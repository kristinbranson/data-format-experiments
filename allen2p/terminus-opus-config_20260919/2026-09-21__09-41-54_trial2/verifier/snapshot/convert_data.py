#!/usr/bin/env python3
"""
Convert the Allen Brain Observatory *Visual Behavior 2P* dataset into the
decoder-ready pickle format.

Usage:
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Design (see /app/CONVERSION_NOTES.md for full justification):
  * unit of a "session"  : one ophys_session_id.  All simultaneously recorded
                           imaging planes (Multiscope) are merged, since they
                           share behaviour, stimulus and clock.
  * unit of a "trial"    : one `go` or `catch` trial (aborted and auto-rewarded
                           trials are excluded, per the task specification).
  * neural signal        : dF/F traces from the Allen pipeline, bin-averaged in
                           100 ms bins (--neural-signal also allows the detected
                           calcium `events` used by the reference paper; dF/F was
                           chosen because it decodes markedly better, see notes).
  * alignment            : trials.change_time (for catch trials this is the SDK's
                           *sham* change time), window [-2, +4] s -> 60 bins.
  * inputs               : none (task specification).
  * outputs              : image identity, image change, running-speed quintile,
                           pupil-diameter quintile, trial outcome.
"""

import argparse
import os
import pickle
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

CACHE_DIR = '/app/data'

# ----------------------------------------------------------------- parameters
BIN_SIZE = 0.1          # seconds (100 ms)
OFF_START = -2.0        # seconds relative to the change
OFF_END = 4.0           # seconds relative to the change
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 60
FLASH_INTERVAL = 0.75   # s, nominal image presentation interval (paper convention)
MAX_FLASH_INTERVAL = 1.0  # s, safety cap for the end of an image interval
NQUANT = 5              # number of percentile bins for running / pupil

OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']


# ------------------------------------------------------------------ utilities
def get_cache():
    from allensdk.brain_observatory.behavior.behavior_project_cache import \
        VisualBehaviorOphysProjectCache
    return VisualBehaviorOphysProjectCache.from_local_cache(cache_dir=CACHE_DIR)


def bin_average(values, timestamps, edges):
    """Average `values` (n_signals, n_samples) within the bins given by `edges`.

    Returns (out, counts) where out is (n_signals, len(edges)-1) and counts is
    the number of samples that fell into each bin.  Bins with no sample are 0.
    """
    lo = np.searchsorted(timestamps, edges[0], side='left')
    hi = np.searchsorted(timestamps, edges[-1], side='left')
    sub_ts = timestamps[lo:hi]
    sub = values[:, lo:hi]
    idx = np.searchsorted(sub_ts, edges, side='left')
    counts = np.diff(idx)
    csum = np.concatenate([np.zeros((sub.shape[0], 1), dtype=np.float64),
                           np.cumsum(sub.astype(np.float64), axis=1)], axis=1)
    sums = csum[:, idx[1:]] - csum[:, idx[:-1]]
    out = np.zeros_like(sums)
    nz = counts > 0
    out[:, nz] = sums[:, nz] / counts[nz]
    return out, counts


def interpolate_nans(x):
    """Linearly interpolate NaNs in a 1-D array (edges use nearest valid value)."""
    x = np.asarray(x, dtype=np.float64).copy()
    bad = ~np.isfinite(x)
    if bad.all():
        return None
    if bad.any():
        good = ~bad
        x[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), x[good])
    return x


def quantile_bins(values, nq=NQUANT):
    """Discretise `values` into `nq` equal-percentile bins.

    Returns (labels, thresholds).  Thresholds are the interior quantiles.
    """
    qs = np.quantile(values, np.arange(1, nq) / nq)
    labels = np.searchsorted(qs, values, side='right')
    return labels.astype(np.int64), qs


# ------------------------------------------------------------ per-session work
def process_session(args):
    """Convert one ophys session.  Returns a dict (or None if the session is skipped)."""
    (ophys_session_id, exp_ids, meta_rows, image_names, show_processing, plot_dir,
     neural_signal) = args
    t_start = time.time()
    timing = {}
    try:
        cache = get_cache()

        # ---------------------------------------------------- load all planes
        t0 = time.time()
        datasets = [cache.get_behavior_ophys_experiment(int(e)) for e in exp_ids]
        timing['load'] = time.time() - t0

        d0 = datasets[0]

        # eye tracking is required (pupil diameter is a decoder output)
        eye = d0.eye_tracking
        if eye is None or len(eye) == 0:
            return {'session_id': ophys_session_id, 'skipped': 'no eye tracking'}

        # ------------------------------------------------------------- trials
        trials = d0.trials
        keep = (trials.go | trials.catch) & trials.change_time.notna()
        tr = trials[keep]
        if len(tr) < 2:
            return {'session_id': ophys_session_id, 'skipped': 'fewer than 2 go/catch trials'}

        # consistency: all planes must share the same behaviour
        for d in datasets[1:]:
            t_other = d.trials
            assert len(t_other) == len(trials), 'trial tables differ across planes'

        change_times = tr.change_time.values.astype(np.float64)

        # -------------------------------------------------- stimulus timeline
        sp = d0.stimulus_presentations
        sp = sp[sp.stimulus_block_name.str.contains('change_detection')]
        sp = sp.sort_values('start_time')
        stim_start = sp.start_time.values.astype(np.float64)
        stim_img = sp.image_name.values.astype(str)
        stim_change = sp.is_change.values.astype(bool)
        # the end of each image-presentation interval is the onset of the next
        # flash (nominally 750 ms later, measured 734-801 ms).  Using the actual
        # next onset avoids leaving unlabelled slivers between intervals.
        stim_end = np.empty_like(stim_start)
        stim_end[:-1] = stim_start[1:]
        stim_end[-1] = stim_start[-1] + FLASH_INTERVAL
        # guard against the (non-existent in this dataset) case of a long gap
        stim_end = np.minimum(stim_end, stim_start + MAX_FLASH_INTERVAL)
        img_to_code = {name: i for i, name in enumerate(image_names)}
        stim_code = np.array([img_to_code[n] for n in stim_img], dtype=np.int64)

        # ------------------------------------------------ behaviour timeseries
        rs = d0.running_speed
        run_ts = rs.timestamps.values.astype(np.float64)
        run_sp = interpolate_nans(rs.speed.values)

        eye_ts = eye.timestamps.values.astype(np.float64)
        pupil_area = eye.pupil_area.values.astype(np.float64)   # NaN where likely_blink
        pupil_diam_raw = 2.0 * np.sqrt(pupil_area / np.pi)
        pupil_diam = interpolate_nans(pupil_diam_raw)
        if pupil_diam is None:
            return {'session_id': ophys_session_id, 'skipped': 'pupil all NaN'}

        # ------------------------------------------------------- neural traces
        t0 = time.time()
        plane_events = []
        plane_ts = []
        region_idx_parts = []
        cell_ids = []
        for d, eid in zip(datasets, exp_ids):
            if neural_signal == 'dff':
                ev = np.vstack(d.dff_traces.dff.values).astype(np.float32)
            elif neural_signal == 'filtered_events':
                ev = np.vstack(d.events.filtered_events.values).astype(np.float32)
            else:
                ev = np.vstack(d.events.events.values).astype(np.float32)
            plane_events.append(ev)
            plane_ts.append(d.ophys_timestamps.astype(np.float64))
            region = meta_rows.loc[eid, 'targeted_structure']
            region_idx_parts.append(np.full(ev.shape[0], region, dtype=object))
            cell_ids.extend(list(d.events.index.values))
        timing['events'] = time.time() - t0

        regions_per_neuron = np.concatenate(region_idx_parts) if region_idx_parts else np.array([])
        n_neurons = sum(e.shape[0] for e in plane_events)
        if n_neurons == 0:
            return {'session_id': ophys_session_id, 'skipped': 'no neurons'}

        # -------------------------------------------------------- trial arrays
        t0 = time.time()
        neural_trials = []
        img_trials = []
        change_trials = []
        run_trials = []
        pupil_trials = []
        empty_bins = 0
        filled_run = 0
        filled_pupil = 0
        for ct in change_times:
            edges = ct + OFF_START + BIN_SIZE * np.arange(NBINS + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])

            # neural: bin each plane on its own timestamps, then stack
            parts = []
            for ev, ts in zip(plane_events, plane_ts):
                b, counts = bin_average(ev, ts, edges)
                empty_bins += int((counts == 0).sum())
                parts.append(b)
            neural_trials.append(np.vstack(parts).astype(np.float32))

            # running speed / pupil diameter (continuous, binned).  Bins that
            # contain no sample (dropped camera frames: up to 13% of bins in one
            # session for the ~30 Hz eye camera) are filled by linear
            # interpolation of the underlying trace at the bin centre, rather
            # than being left at zero.
            rb, rc = bin_average(run_sp[None, :], run_ts, edges)
            rb = rb[0]
            if (rc == 0).any():
                rb[rc == 0] = np.interp(centers[rc == 0], run_ts, run_sp)
                filled_run += int((rc == 0).sum())
            run_trials.append(rb)
            pb, pc = bin_average(pupil_diam[None, :], eye_ts, edges)
            pb = pb[0]
            if (pc == 0).any():
                pb[pc == 0] = np.interp(centers[pc == 0], eye_ts, pupil_diam)
                filled_pupil += int((pc == 0).sum())
            pupil_trials.append(pb)

            # image identity / change: the flash interval containing the bin centre
            j = np.searchsorted(stim_start, centers, side='right') - 1
            j = np.clip(j, 0, len(stim_start) - 1)
            within = centers < stim_end[j]
            img = np.where(within, stim_code[j], img_to_code['omitted'])
            chg = np.where(within, stim_change[j], False).astype(np.int64)
            img_trials.append(img.astype(np.int64))
            change_trials.append(chg)
        timing['bin'] = time.time() - t0

        # ------------------------------------- per-session percentile binning
        run_mat = np.vstack(run_trials)
        pupil_mat = np.vstack(pupil_trials)
        run_lab, run_q = quantile_bins(run_mat.ravel())
        pupil_lab, pupil_q = quantile_bins(pupil_mat.ravel())
        run_lab = run_lab.reshape(run_mat.shape)
        pupil_lab = pupil_lab.reshape(pupil_mat.shape)

        # ------------------------------------------------------- trial outcome
        outcome = np.full(len(tr), -1, dtype=np.int64)
        for k, name in enumerate(OUTCOMES):
            outcome[tr[name].values.astype(bool)] = k
        assert (outcome >= 0).all(), 'trial with no outcome label'

        # --------------------------------------------------------- assemble
        outputs = []
        for i in range(len(tr)):
            out = np.stack([img_trials[i],
                            change_trials[i],
                            run_lab[i],
                            pupil_lab[i],
                            np.full(NBINS, outcome[i], dtype=np.int64)], axis=0)
            outputs.append(out.astype(np.int64))
        inputs = [np.zeros((0, NBINS), dtype=np.float32) for _ in range(len(tr))]

        result = {
            'session_id': int(ophys_session_id),
            'experiment_ids': [int(e) for e in exp_ids],
            'mouse_id': str(meta_rows.iloc[0]['mouse_id']),
            'session_type': str(meta_rows.iloc[0]['session_type']),
            'cre_line': str(meta_rows.iloc[0]['cre_line']),
            'experience_level': str(meta_rows.iloc[0]['experience_level']),
            'project_code': str(meta_rows.iloc[0]['project_code']),
            'equipment_name': str(meta_rows.iloc[0]['equipment_name']),
            'neural': neural_trials,
            'input': inputs,
            'output': outputs,
            'regions': regions_per_neuron,
            'n_neurons': int(n_neurons),
            'n_trials': int(len(tr)),
            'cell_specimen_ids': [int(c) for c in cell_ids],
            'run_quantiles': run_q.tolist(),
            'pupil_quantiles': pupil_q.tolist(),
            'n_go': int(tr.go.sum()), 'n_catch': int(tr.catch.sum()),
            'n_hit': int(tr.hit.sum()), 'n_miss': int(tr.miss.sum()),
            'n_fa': int(tr.false_alarm.sum()), 'n_cr': int(tr.correct_reject.sum()),
            'pupil_nan_frac': float(np.isnan(pupil_diam_raw).mean()),
            'empty_bins': int(empty_bins),
            'filled_run_bins': int(filled_run),
            'filled_pupil_bins': int(filled_pupil),
            'timing': timing,
            'total_time': time.time() - t_start,
        }

        if show_processing:
            try:
                make_processing_plot(result, d0, datasets, tr, sp, run_ts, run_sp,
                                     eye_ts, pupil_diam_raw, pupil_diam,
                                     plane_events, plane_ts, image_names, plot_dir,
                                     neural_signal)
            except Exception as exc:                        # plotting must never kill a run
                print(f'  plotting failed for {ophys_session_id}: {exc}')

        return result

    except Exception as exc:
        import traceback
        return {'session_id': int(ophys_session_id), 'error': f'{exc}',
                'traceback': traceback.format_exc()}


# ----------------------------------------------------------------- plotting
def make_processing_plot(res, d0, datasets, tr, sp, run_ts, run_sp, eye_ts,
                         pupil_raw, pupil_interp, plane_events, plane_ts,
                         image_names, plot_dir, signal_name='dF/F'):
    """Visualise every processing step for one session (2 example trials)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sid = res['session_id']
    change_times = tr.change_time.values.astype(np.float64)
    # pick one go (hit if possible) and one catch trial
    idx_go = int(np.flatnonzero(tr.go.values)[0])
    catch_list = np.flatnonzero(tr.catch.values)
    idx_catch = int(catch_list[0]) if len(catch_list) else int(np.flatnonzero(tr.go.values)[1])

    fig, axes = plt.subplots(6, 2, figsize=(18, 20), sharex='col')
    for col, ti in enumerate([idx_go, idx_catch]):
        ct = change_times[ti]
        edges = ct + OFF_START + BIN_SIZE * np.arange(NBINS + 1)
        centers = 0.5 * (edges[:-1] + edges[1:]) - ct
        t0, t1 = edges[0], edges[-1]

        # ---- 1: raw event traces (all planes) with bin edges
        ax = axes[0, col]
        offset = 0
        for ev, ts in zip(plane_events, plane_ts):
            m = (ts >= t0) & (ts <= t1)
            for k in range(min(ev.shape[0], 10)):
                y = ev[k, m]
                if y.max() > 0:
                    y = y / y.max()
                ax.plot(ts[m] - ct, y + offset, lw=0.8)
                offset += 1.1
        for e in edges:
            ax.axvline(e - ct, color='0.9', lw=0.3, zorder=0)
        ax.set_title(f'session {sid} trial {ti} ({"go" if tr.go.values[ti] else "catch"}) - '
                     f'raw {signal_name} traces (<=10 cells/plane, normalised)')
        ax.set_ylabel('cells')

        # ---- 2: binned neural matrix
        ax = axes[1, col]
        nm = res['neural'][ti]
        ax.imshow(nm, aspect='auto', interpolation='nearest',
                  extent=[centers[0], centers[-1], nm.shape[0], 0], cmap='magma')
        ax.set_title(f'binned {signal_name} (100 ms), {nm.shape[0]} neurons x {nm.shape[1]} bins')
        ax.set_ylabel('neuron')

        # ---- 3: running speed raw vs binned vs quintile
        ax = axes[2, col]
        m = (run_ts >= t0) & (run_ts <= t1)
        ax.plot(run_ts[m] - ct, run_sp[m], color='0.5', lw=1, label='raw 60 Hz')
        rb = np.full(NBINS, np.nan)
        for b in range(NBINS):
            sel = (run_ts >= edges[b]) & (run_ts < edges[b + 1])
            if sel.any():
                rb[b] = run_sp[sel].mean()
        ax.step(centers, rb, where='mid', color='C0', label='binned (100 ms)')
        ax.set_ylabel('running (cm/s)')
        ax2 = ax.twinx()
        ax2.step(centers, res['output'][ti][2], where='mid', color='C3', label='quintile')
        ax2.set_ylim(-0.5, 4.5); ax2.set_ylabel('quintile', color='C3')
        for q in res['run_quantiles']:
            ax.axhline(q, color='C3', ls=':', lw=0.7)
        ax.legend(loc='upper left', fontsize=8)
        ax.set_title('running speed: raw / binned / quintile (dotted = session quintile thresholds)')

        # ---- 4: pupil raw (with blink gaps), interpolated, binned, quintile
        ax = axes[3, col]
        m = (eye_ts >= t0) & (eye_ts <= t1)
        ax.plot(eye_ts[m] - ct, pupil_interp[m], color='C1', lw=1, label='interpolated')
        ax.plot(eye_ts[m] - ct, pupil_raw[m], color='k', lw=1, label='raw (NaN at blinks)')
        pb = np.full(NBINS, np.nan)
        for b in range(NBINS):
            sel = (eye_ts >= edges[b]) & (eye_ts < edges[b + 1])
            if sel.any():
                pb[b] = pupil_interp[sel].mean()
        ax.step(centers, pb, where='mid', color='C0', label='binned (100 ms)')
        for q in res['pupil_quantiles']:
            ax.axhline(q, color='C3', ls=':', lw=0.7)
        ax2 = ax.twinx()
        ax2.step(centers, res['output'][ti][3], where='mid', color='C3')
        ax2.set_ylim(-0.5, 4.5); ax2.set_ylabel('quintile', color='C3')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_ylabel('pupil diameter (px)')
        ax.set_title('pupil diameter: raw / blink-interpolated / binned / quintile')

        # ---- 5: stimulus flashes and decoded image identity
        ax = axes[4, col]
        sel = sp[(sp.end_time >= t0) & (sp.start_time <= t1)]
        for _, row in sel.iterrows():
            if bool(row.omitted):
                ax.axvspan(row.start_time - ct, row.start_time + 0.25 - ct,
                           color='r', alpha=0.15, hatch='//')
            else:
                code = image_names.index(str(row.image_name))
                ax.axvspan(row.start_time - ct, row.start_time + 0.25 - ct,
                           color=plt.cm.tab20(code % 20), alpha=0.6)
            if bool(row.is_change):
                ax.axvline(row.start_time - ct, color='k', lw=2)
        ax.step(centers, res['output'][ti][0], where='mid', color='k', lw=1.5,
                label='image identity code')
        ax.set_ylim(-0.5, len(image_names) - 0.5)
        ax.set_yticks(range(len(image_names)))
        ax.set_yticklabels(image_names, fontsize=6)
        ax.set_title('image identity output vs. actual flashes (coloured bars; black line = change)')

        # ---- 6: image change and trial outcome
        ax = axes[5, col]
        ax.step(centers, res['output'][ti][1], where='mid', color='C2', lw=2, label='image_change')
        ax.step(centers, res['output'][ti][4] / 4.0, where='mid', color='C4', lw=1.5,
                label=f'outcome = {OUTCOMES[int(res["output"][ti][4][0])]} (scaled)')
        ax.axvline(0, color='k', lw=1)
        ax.set_ylim(-0.1, 1.3)
        ax.legend(fontsize=8, loc='upper left')
        ax.set_xlabel('time from (sham) change (s)')
        ax.set_title('image_change output and trial outcome')

    fig.tight_layout()
    fname = os.path.join(plot_dir, f'processing_{sid}.png')
    fig.savefig(fname, dpi=110)
    plt.close(fig)
    print(f'  wrote {fname}')


# --------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', default=True)
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--neural-signal', default='dff',
                    choices=['events', 'dff', 'filtered_events'],
                    help='neural stream to bin. Default dF/F: the canonical '
                         'Allen pipeline output, which decodes substantially '
                         'better than the very sparse event traces at 100 ms bins '
                         '(see CONVERSION_NOTES Step 8).')
    args = ap.parse_args()

    t_all = time.time()
    print('loading project metadata ...')
    cache = get_cache()
    exp_table = cache.get_ophys_experiment_table()

    # only experiments whose NWB file is present locally
    nwb_dir = os.path.join(CACHE_DIR, 'visual-behavior-ophys-1.1.0',
                           'behavior_ophys_experiments')
    local_ids = sorted(int(f.split('_')[-1].split('.')[0])
                       for f in os.listdir(nwb_dir) if f.endswith('.nwb'))
    et = exp_table.loc[exp_table.index.intersection(local_ids)]
    print(f'  {len(et)} experiments available locally (of {len(exp_table)} in the release)')

    # --- curation: keep only ACTIVE (behaving) sessions
    et = et[~et.passive]
    print(f'  {len(et)} active experiments ({et.ophys_session_id.nunique()} sessions, '
          f'{et.mouse_id.nunique()} mice) after removing passive sessions')

    # global image list (image sets A and B) -> consistent output classes
    image_names = sorted({
        'im000', 'im031', 'im035', 'im045', 'im054', 'im073', 'im075', 'im106',
        'im061', 'im062', 'im063', 'im065', 'im066', 'im069', 'im077', 'im085'})
    image_names = image_names + ['omitted']

    session_ids = sorted(et.ophys_session_id.unique())
    if args.sample:
        # one single-plane and one multi-plane session, for a fast end-to-end test
        multi = [s for s in session_ids if (et.ophys_session_id == s).sum() > 1]
        single = [s for s in session_ids if (et.ophys_session_id == s).sum() == 1]
        session_ids = [single[0], multi[0]] if multi else single[:2]
        print(f'  SAMPLE mode: sessions {session_ids}')

    jobs = []
    for sid in session_ids:
        rows = et[et.ophys_session_id == sid].sort_index()
        jobs.append((int(sid), list(rows.index), rows,
                     image_names, args.show_processing, '/app', args.neural_signal))

    print(f'converting {len(jobs)} sessions with {args.workers} workers ...')
    results = []
    t0 = time.time()
    if args.workers > 1 and len(jobs) > 1:
        with Pool(min(args.workers, len(jobs))) as pool:
            for i, r in enumerate(pool.imap_unordered(process_session, jobs)):
                results.append(r)
                done = i + 1
                el = time.time() - t0
                print(f'  [{done}/{len(jobs)}] session {r.get("session_id")} '
                      f'{"ERROR" if "error" in r else r.get("skipped", "ok")} '
                      f'({el:.0f}s elapsed, {el/done:.1f}s/session, '
                      f'eta {(len(jobs)-done)*el/done:.0f}s)')
    else:
        for i, job in enumerate(jobs):
            r = process_session(job)
            results.append(r)
            print(f'  [{i+1}/{len(jobs)}] session {r.get("session_id")} '
                  f'{"ERROR" if "error" in r else r.get("skipped", "ok")} '
                  f'({time.time()-t0:.0f}s elapsed)')

    # ------------------------------------------------------------- assemble
    errors = [r for r in results if 'error' in r]
    for r in errors:
        print(f'ERROR in session {r["session_id"]}: {r["error"]}')
        print(r['traceback'])
    skipped = [r for r in results if 'skipped' in r]
    for r in skipped:
        print(f'SKIPPED session {r["session_id"]}: {r["skipped"]}')

    good = [r for r in results if 'error' not in r and 'skipped' not in r]
    good.sort(key=lambda r: r['session_id'])
    print(f'{len(good)} sessions converted, {len(skipped)} skipped, {len(errors)} errors')

    subjects = sorted({r['mouse_id'] for r in good})
    subject_idx = np.array([subjects.index(r['mouse_id']) for r in good], dtype=np.int64)
    brain_regions = sorted({reg for r in good for reg in np.unique(r['regions'])})
    brain_region_idx = [np.array([brain_regions.index(x) for x in r['regions']],
                                 dtype=np.int64) for r in good]

    data = {
        'neural': [r['neural'] for r in good],
        'input': [r['input'] for r in good],
        'output': [r['output'] for r in good],
        'subjects': subjects,
        'subject_idx': subject_idx,
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': [],
        'output_names': ['image_identity', 'image_change', 'running_speed_bin',
                         'pupil_diameter_bin', 'trial_outcome'],
        'output_values': [
            image_names,
            ['no_change', 'change'],
            [f'speed_quintile_{i+1}' for i in range(NQUANT)],
            [f'pupil_quintile_{i+1}' for i in range(NQUANT)],
            OUTCOMES,
        ],
        'metadata': {
            'task_description': (
                'Allen Brain Observatory Visual Behavior 2P: head-fixed mice perform a '
                'go/no-go visual change-detection task. Images are flashed for 250 ms '
                'every 750 ms; mice earn water by licking when the image identity '
                'changes. Trials are go (real change) or catch (sham change); aborted '
                'and auto-rewarded trials are excluded. Decoded from 2-photon calcium '
                'events: the identity of the currently presented image, whether an image '
                'change just occurred, the quintile of running speed, the quintile of '
                'pupil diameter, and the behavioural outcome of the trial '
                '(hit/miss/false alarm/correct rejection).'),
            'time_bin_size': BIN_SIZE * 1000.0,
            'temporal_alignment_event': (
                'image change time (trials.change_time); for catch trials this is the '
                'sham change time, i.e. when the change would have occurred. Neural data '
                'are binned on the ophys timestamps.'),
            'off_start': OFF_START,
            'off_end': OFF_END,
            'neural_signal': (
                {'events': 'detected calcium events (L0 event detection, AllenSDK '
                           '`events`), mean event magnitude per 100 ms bin',
                 'filtered_events': 'filtered (half-gaussian smoothed) calcium events, '
                                    'mean per 100 ms bin',
                 'dff': 'dF/F traces (Allen pipeline), mean per 100 ms bin'}[args.neural_signal]),
            'dataset': 'Allen Brain Observatory - Visual Behavior 2P (visual-behavior-ophys v1.1.0)',
            'session_info': [
                {k: r[k] for k in ('session_id', 'experiment_ids', 'mouse_id',
                                   'session_type', 'cre_line', 'experience_level',
                                   'project_code', 'equipment_name', 'n_neurons',
                                   'n_trials', 'n_go', 'n_catch', 'n_hit', 'n_miss',
                                   'n_fa', 'n_cr', 'run_quantiles', 'pupil_quantiles',
                                   'pupil_nan_frac', 'cell_specimen_ids')}
                for r in good],
            'curation': (
                'Kept active (behaving) sessions only (passive OPHYS_2/OPHYS_5 sessions '
                'have no licks or rewards). Kept go and catch trials; excluded aborted '
                'and auto-rewarded trials. Sessions without eye tracking were dropped. '
                'Only valid ROIs are present in the released data (upstream ROI '
                'filtering, crosstalk removal, demixing, neuropil subtraction).'),
            'trial_window_seconds': [OFF_START, OFF_END],
        },
    }

    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    # ------------------------------------------------------------ statistics
    ntr = sum(len(x) for x in data['neural'])
    nneu = sum(r['n_neurons'] for r in good)
    print('\n==================== SUMMARY ====================')
    print(f'sessions          : {len(good)}')
    print(f'subjects          : {len(subjects)}')
    print(f'brain regions     : {brain_regions}')
    print(f'neurons (total)   : {nneu}  (mean {nneu/max(len(good),1):.1f} per session)')
    print(f'trials (total)    : {ntr}  (mean {ntr/max(len(good),1):.1f} per session)')
    print(f'go / catch        : {sum(r["n_go"] for r in good)} / {sum(r["n_catch"] for r in good)}')
    print(f'hit/miss/FA/CR    : {sum(r["n_hit"] for r in good)}/{sum(r["n_miss"] for r in good)}/'
          f'{sum(r["n_fa"] for r in good)}/{sum(r["n_cr"] for r in good)}')
    print(f'empty neural bins : {sum(r["empty_bins"] for r in good)}')
    nbins_tot = ntr * NBINS
    print(f'interp-filled bins: running {sum(r["filled_run_bins"] for r in good)} '
          f'({sum(r["filled_run_bins"] for r in good)/max(nbins_tot,1)*100:.3f}%), '
          f'pupil {sum(r["filled_pupil_bins"] for r in good)} '
          f'({sum(r["filled_pupil_bins"] for r in good)/max(nbins_tot,1)*100:.3f}%)')
    print(f'time bin          : {BIN_SIZE*1000:.0f} ms, window [{OFF_START}, {OFF_END}] s, {NBINS} bins')
    tl = np.array([r['total_time'] for r in good])
    print(f'per-session time  : mean {tl.mean():.1f}s, max {tl.max():.1f}s')
    print(f'total wall time   : {time.time()-t_all:.0f}s')
    print(f'written to        : {args.outfile} '
          f'({os.path.getsize(args.outfile)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
