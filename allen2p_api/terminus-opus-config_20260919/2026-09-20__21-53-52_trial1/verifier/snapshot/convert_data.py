#!/usr/bin/env python3
'''
Convert the Allen Brain Observatory Visual Behavior 2P dataset (AllenSDK
VisualBehaviorOphysProjectCache) into the decoder dictionary format.

Usage:  python -u /app/convert_data.py <outpicklefile> [--full|--sample] [--show-processing]

Design (see /app/CONVERSION_NOTES.md Step 5 for full justification):
  session          = one ophys_session_id (all simultaneously recorded imaging planes merged)
  trials           = (go | catch) & ~aborted & ~auto_rewarded
  alignment event  = trials.change_time (real change on go trials, sham change on catch trials)
  window           = [-2.25, +3.75] s around the change (8 x 750 ms flash cycles)
  bin              = 250 ms (image duration) -> 24 bins per trial
  neural           = dF/F (Allen pipeline), averaged per bin (see notes for events comparison)
  inputs           = none
  outputs          = image identity (16), image change (2), running quintile (5),
                     pupil quintile (5), trial outcome (4)
'''
import argparse
import glob
import os
import pickle
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from allensdk.brain_observatory.behavior.behavior_project_cache import (
    VisualBehaviorOphysProjectCache)

CACHE_DIR = '/app/data'
NWB_GLOB = os.path.join(CACHE_DIR, 'visual-behavior-ophys-1.1.0',
                        'behavior_ophys_experiments', '*.nwb')

# ---------------------------------------------------------------- parameters
OFF_START = -2.25       # s relative to change time
OFF_END = 3.75          # s relative to change time
BIN_SIZE = 0.25         # s
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 24
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(N_BINS + 1)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2.0
FLASH_CYCLE = 0.75      # s, image presentation interval (paper)
PUPIL_MAX_INTERP_GAP = 0.5   # s, blink gaps shorter than this are interpolated
N_QUANTILES = 5
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
NEURAL_SIGNAL_DESCRIPTION = {
    'events': ('L0-detected calcium events (AllenSDK BehaviorOphysExperiment.events, column '
               '"events") summed within each 250 ms bin; dF/F and event detection were computed '
               'by the Allen pipeline; only valid ROIs are released'),
    'filtered_events': ('L0-detected calcium events smoothed with the SDK causal half-gaussian '
                        '(filtered_events), summed within each 250 ms bin'),
    'dff': ('dF/F traces computed by the Allen pipeline (BehaviorOphysExperiment.dff_traces), '
            'averaged within each 250 ms bin; only valid ROIs are released'),
}


def get_cache():
    return VisualBehaviorOphysProjectCache.from_local_cache(
        cache_dir=CACHE_DIR, use_static_cache=False)


def available_experiment_table(cache):
    '''Experiment table restricted to experiments whose NWB file is present locally.'''
    et = cache.get_ophys_experiment_table()
    avail = sorted(int(os.path.basename(f).split('_')[-1].split('.')[0])
                   for f in glob.glob(NWB_GLOB))
    sub = et.loc[et.index.isin(avail)].copy()
    return sub


# ------------------------------------------------------------ binning helpers
def bin_sum(values, timestamps, edges_abs):
    '''Sum `values` (n_signals, n_samples) over bins defined by absolute edges.

    edges_abs: (n_trials, N_BINS+1) absolute times.
    Returns (n_signals, n_trials, N_BINS).
    '''
    csum = np.concatenate([np.zeros((values.shape[0], 1), dtype=np.float64),
                           np.cumsum(values, axis=1)], axis=1)
    idx = np.searchsorted(timestamps, edges_abs.ravel())          # (n_trials*(N+1),)
    idx = idx.reshape(edges_abs.shape)
    take = csum[:, idx]                                           # (n_signals, n_trials, N+1)
    return (take[:, :, 1:] - take[:, :, :-1]).astype(np.float32)


def bin_mean_nan(values, timestamps, edges_abs):
    '''Mean of `values` (1d, may contain NaN) within each bin; NaN if no valid sample.'''
    valid = ~np.isnan(values)
    filled = np.where(valid, values, 0.0)
    csum = np.concatenate([[0.0], np.cumsum(filled)])
    ccnt = np.concatenate([[0], np.cumsum(valid.astype(np.int64))])
    idx = np.searchsorted(timestamps, edges_abs.ravel()).reshape(edges_abs.shape)
    s = csum[idx][:, 1:] - csum[idx][:, :-1]
    n = ccnt[idx][:, 1:] - ccnt[idx][:, :-1]
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return out


def interpolate_short_gaps(t, v, max_gap):
    '''Linearly interpolate NaN runs shorter than max_gap seconds; leave longer runs NaN.'''
    v = np.asarray(v, dtype=np.float64).copy()
    isn = np.isnan(v)
    if not isn.any() or isn.all():
        return v
    idx = np.arange(len(v))
    filled = np.interp(idx, idx[~isn], v[~isn])
    d = np.diff(np.concatenate(([0], isn.astype(np.int8), [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    out = filled
    for s0, e0 in zip(starts, ends):
        # gap duration measured between the bracketing valid samples
        t0 = t[s0 - 1] if s0 > 0 else t[0]
        t1 = t[e0] if e0 < len(t) else t[-1]
        if (t1 - t0) > max_gap or s0 == 0 or e0 >= len(t):
            out[s0:e0] = np.nan
    return out


# ------------------------------------------------------------ session loading
def process_session(args):
    '''Process one ophys session (all of its imaging planes).

    Returns a dict with per-trial neural matrices and *continuous* behaviour
    (running speed, pupil diameter) which are discretised globally later.
    '''
    session_id, eids, region_map, want_debug, signal = args
    t_start = time.time()
    cache = get_cache()
    out = {'session_id': int(session_id), 'error': ''}
    try:
        datasets = [cache.get_behavior_ophys_experiment(int(e)) for e in eids]
        t_load = time.time() - t_start
        ds0 = datasets[0]

        # ---------------- trials: go + catch, excluding aborted / auto-rewarded
        trials = ds0.trials
        sel = ((trials['go'] | trials['catch']) & (~trials['aborted'])
               & (~trials['auto_rewarded']))
        trials = trials[sel]
        trials = trials[~trials['change_time'].isna()]
        if len(trials) < 2:
            out['error'] = 'fewer than 2 go/catch trials'
            return out
        change_times = trials['change_time'].values.astype(np.float64)
        edges_abs = change_times[:, None] + BIN_EDGES[None, :]      # (n_trials, N+1)
        centers_abs = change_times[:, None] + BIN_CENTERS[None, :]  # (n_trials, N)
        n_trials = len(change_times)

        # ---------------- neural: L0 detected calcium events, summed per bin
        neural_planes = []
        region_per_neuron = []
        cell_ids = []
        for ds, eid in zip(datasets, eids):
            ts = np.asarray(ds.ophys_timestamps, dtype=np.float64)
            if signal == 'dff':
                ev = np.vstack(ds.dff_traces['dff'].values).astype(np.float64)
            elif signal == 'filtered_events':
                ev = np.vstack(ds.events['filtered_events'].values).astype(np.float64)
            else:
                ev = np.vstack(ds.events['events'].values).astype(np.float64)
            assert ev.shape[1] == ts.shape[0], 'trace/timestamps mismatch'
            n_per_bin = bin_sum(np.ones((1, ts.shape[0])), ts, edges_abs)   # (1, n_trials, N)
            binned = bin_sum(ev, ts, edges_abs)        # (n_cells, n_trials, N)
            if signal == 'dff':
                # average dF/F within the bin (sum would scale with frame rate)
                binned = binned / np.maximum(n_per_bin, 1)
            neural_planes.append(binned)
            region_per_neuron += [region_map[int(eid)]] * ev.shape[0]
            cell_ids += list(ds.events.index.values)
        neural = np.concatenate(neural_planes, axis=0)  # (n_neurons, n_trials, N)

        # ---------------- stimulus: image identity and image change
        sp = ds0.stimulus_presentations
        sp = sp[sp['stimulus_block_name'].astype(str).str.contains('change_detection')]
        shown = sp[sp['image_name'] != 'omitted']
        flash_start = shown['start_time'].values.astype(np.float64)
        flash_image = shown['image_name'].values.astype(str)
        flash_ischange = shown['is_change'].values.astype(bool)
        # index of the most recent shown-image onset at or before each bin centre
        j = np.searchsorted(flash_start, centers_abs, side='right') - 1
        if np.any(j < 0):
            out['error'] = 'bin before first flash'
            return out
        image_name = flash_image[j]                                   # (n_trials, N)
        since = centers_abs - flash_start[j]
        # image change: bins inside the 750 ms presentation interval of a change flash
        image_change = (flash_ischange[j] & (since < FLASH_CYCLE)).astype(np.int64)

        # ---------------- running speed (continuous, discretised globally later)
        run = ds0.running_speed
        run_t = run['timestamps'].values.astype(np.float64)
        run_v = run['speed'].values.astype(np.float64)
        running = bin_mean_nan(run_v, run_t, edges_abs)               # (n_trials, N)

        # ---------------- pupil diameter (continuous, discretised globally later)
        eye = ds0.eye_tracking
        if len(eye) == 0:
            out['error'] = 'no eye tracking data'
            return out
        eye_t = eye['timestamps'].values.astype(np.float64)
        pupil_area = eye['pupil_area'].values.astype(np.float64)
        pupil_diam = 2.0 * np.sqrt(pupil_area / np.pi)
        pupil_diam = interpolate_short_gaps(eye_t, pupil_diam, PUPIL_MAX_INTERP_GAP)
        pupil = bin_mean_nan(pupil_diam, eye_t, edges_abs)            # (n_trials, N)

        # ---------------- trial outcome (static per trial)
        outcome = np.full(n_trials, -1, dtype=np.int64)
        for k, name in enumerate(OUTCOMES):
            outcome[trials[name].values.astype(bool)] = k
        if np.any(outcome < 0):
            out['error'] = 'trial without an outcome label'
            return out

        # ---------------- trial curation: require complete behaviour
        keep = (~np.isnan(pupil).any(axis=1)) & (~np.isnan(running).any(axis=1))
        n_dropped = int((~keep).sum())
        if keep.sum() < 2:
            out['error'] = 'fewer than 2 trials with complete behaviour'
            return out

        out.update(dict(
            neural=neural[:, keep, :].astype(np.float32),
            image_name=image_name[keep],
            image_change=image_change[keep],
            running=running[keep],
            pupil=pupil[keep],
            outcome=outcome[keep],
            change_times=change_times[keep],
            go=trials['go'].values.astype(bool)[keep],
            catch=trials['catch'].values.astype(bool)[keep],
            regions=np.array(region_per_neuron),
            cell_ids=np.array(cell_ids),
            mouse_id=str(ds0.metadata['mouse_id']),
            session_type=str(ds0.metadata['session_type']),
            cre_line=str(ds0.metadata['cre_line']),
            experiment_ids=[int(e) for e in eids],
            n_trials_dropped=n_dropped,
            n_trials_total=n_trials,
            t_load=t_load,
            t_total=time.time() - t_start,
        ))
        if want_debug:
            out['debug'] = dict(
                ophys_timestamps=np.asarray(datasets[0].ophys_timestamps, dtype=np.float64),
                events0=np.vstack(datasets[0].events['events'].values).astype(np.float32),
                run_t=run_t, run_v=run_v, eye_t=eye_t, pupil_diam=pupil_diam,
                flash_start=flash_start, flash_image=flash_image,
                flash_ischange=flash_ischange, keep=keep,
                lick_times=np.asarray(ds0.licks['timestamps'].values, dtype=np.float64),
                reward_times=np.asarray(ds0.rewards['timestamps'].values, dtype=np.float64),
            )
    except Exception as exc:                                   # pragma: no cover
        import traceback
        out['error'] = f'{exc!r}\n{traceback.format_exc()[-800:]}'
    return out


# ------------------------------------------------------------- discretisation
def quantile_bins(values, n_bins=N_QUANTILES):
    '''Global equal-percentile edges (interior) for a 1-d array of values.'''
    qs = np.linspace(0, 100, n_bins + 1)[1:-1]
    return np.percentile(values, qs)


def digitize_with(edges, values):
    return np.digitize(values, edges).astype(np.int64)


# ------------------------------------------------------------------ plotting
def plot_processing(sess, data, session_index, outfile):
    '''Plot every processing step for one session to visually verify the conversion.'''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dbg = sess['debug']
    neural = data['neural'][session_index]
    outputs = data['output'][session_index]
    onames = data['output_names']
    ct = sess['change_times']
    tr = 0 if not sess['catch'].any() else int(np.where(sess['catch'])[0][0])
    trials_to_plot = [0, 1, tr]

    fig, axes = plt.subplots(6, len(trials_to_plot), figsize=(6 * len(trials_to_plot), 18),
                             sharex='col')
    for col, trial in enumerate(trials_to_plot):
        c = ct[trial]
        t0, t1 = c + OFF_START, c + OFF_END
        tt = BIN_CENTERS

        # --- 1: raw events of first plane vs binned neural
        ax = axes[0, col]
        ots = dbg['ophys_timestamps']
        m = (ots >= t0) & (ots <= t1)
        ev = dbg['events0'][:, m]
        nshow = min(15, ev.shape[0])
        for i in range(nshow):
            ax.plot(ots[m] - c, ev[i] + i * 0.5, lw=0.5, color='k', alpha=0.6)
        ax.set_title(f'trial {trial}: raw events (first plane, {nshow} cells)')
        ax.axvline(0, color='r')

        ax = axes[1, col]
        ax.imshow(neural[trial][:min(30, neural[trial].shape[0])], aspect='auto',
                  extent=[OFF_START, OFF_END, 0, min(30, neural[trial].shape[0])],
                  interpolation='nearest', cmap='magma')
        ax.axvline(0, color='c')
        ax.set_title('binned neural (events summed / 250 ms)')

        # --- 2: stimulus flashes vs image identity output
        ax = axes[2, col]
        fs = dbg['flash_start']
        sel = (fs >= t0 - 1) & (fs <= t1)
        for st, im, isch in zip(fs[sel], dbg['flash_image'][sel], dbg['flash_ischange'][sel]):
            ax.axvspan(st - c, st - c + 0.25, color='red' if isch else 'gray', alpha=0.4)
        img_idx = outputs[trial][onames.index('image_identity')]
        ax.step(tt, img_idx, where='mid', color='b')
        ax.axvline(0, color='r')
        ax.set_title('flashes (gray=repeat, red=change) + image_identity code')

        # --- 3: image change binary
        ax = axes[3, col]
        ax.step(tt, outputs[trial][onames.index('image_change')], where='mid', color='m')
        for st, isch in zip(fs[sel], dbg['flash_ischange'][sel]):
            if isch:
                ax.axvspan(st - c, st - c + FLASH_CYCLE, color='red', alpha=0.2)
        ax.axvline(0, color='r')
        ax.set_ylim(-0.2, 1.2)
        ax.set_title('image_change (shaded = 750 ms after a real change)')

        # --- 4: running raw vs discretised
        ax = axes[4, col]
        rt, rv = dbg['run_t'], dbg['run_v']
        m = (rt >= t0) & (rt <= t1)
        ax.plot(rt[m] - c, rv[m], color='k', lw=0.8, label='running (cm/s)')
        ax2 = ax.twinx()
        ax2.step(tt, outputs[trial][onames.index('running_speed_quintile')], where='mid',
                 color='g', label='quintile')
        ax2.set_ylim(-0.5, 4.5)
        ax.axvline(0, color='r')
        ax.set_title('running speed raw (black) vs quintile (green)')

        # --- 5: pupil raw vs discretised, plus licks/rewards
        ax = axes[5, col]
        et_, pd_ = dbg['eye_t'], dbg['pupil_diam']
        m = (et_ >= t0) & (et_ <= t1)
        ax.plot(et_[m] - c, pd_[m], color='k', lw=0.8)
        ax2 = ax.twinx()
        ax2.step(tt, outputs[trial][onames.index('pupil_diameter_quintile')], where='mid',
                 color='purple')
        ax2.set_ylim(-0.5, 4.5)
        lt = dbg['lick_times']
        lt = lt[(lt >= t0) & (lt <= t1)]
        ax.plot(lt - c, np.full(len(lt), np.nanmin(pd_[m]) if m.any() else 0), '|',
                color='b', ms=10)
        rw = dbg['reward_times']
        rw = rw[(rw >= t0) & (rw <= t1)]
        ax.plot(rw - c, np.full(len(rw), np.nanmin(pd_[m]) if m.any() else 0), 'd',
                color='c', ms=8)
        ax.axvline(0, color='r')
        oc = int(outputs[trial][onames.index('trial_outcome')][0])
        ax.set_title(f'pupil raw/quintile; licks(blue) rewards(cyan); outcome={OUTCOMES[oc]}')
        ax.set_xlabel('time from change (s)')

    fig.suptitle(f"session {sess['session_id']} ({sess['session_type']}, {sess['cre_line']}), "
                 f"{neural[0].shape[0]} neurons, {len(neural)} trials")
    fig.tight_layout()
    fig.savefig(outfile, dpi=110)
    plt.close(fig)
    print(f'  wrote {outfile}')


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile', type=str, help='output pickle file')
    ap.add_argument('--full', action='store_true', default=True, help='process all sessions (default)')
    ap.add_argument('--sample', action='store_true', help='process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true',
                    help='plot every processing step for up to 2 sessions')
    ap.add_argument('--nproc', type=int, default=16)
    ap.add_argument('--quantile-scope', type=str, default='global',
                    choices=['global', 'session'],
                    help='compute running/pupil percentile bins over the whole dataset or per session')
    ap.add_argument('--limit', type=int, default=0, help='process only the first N sessions (testing)')
    ap.add_argument('--neural-signal', type=str, default='dff',
                    choices=['events', 'dff', 'filtered_events'],
                    help='neural signal to bin (default: L0 detected calcium events)')
    args = ap.parse_args()

    t_all = time.time()
    cache = get_cache()
    et = available_experiment_table(cache)
    print(f'experiments with local NWB files: {len(et)}')
    active = et[~et['passive']].copy()
    print(f'active-behavior experiments: {len(active)} '
          f'({active["ophys_session_id"].nunique()} ophys sessions, '
          f'{active["mouse_id"].nunique()} mice)')

    region_map = active['targeted_structure'].to_dict()
    region_map = {int(k): str(v) for k, v in region_map.items()}

    groups = active.groupby('ophys_session_id')
    session_ids = sorted(groups.groups.keys())
    if args.sample:
        # one single-plane and one multi-plane session, for a representative sample
        multi = [s for s in session_ids if len(groups.get_group(s)) > 1]
        single = [s for s in session_ids if len(groups.get_group(s)) == 1]
        session_ids = [single[0], multi[0]] if multi else single[:2]
    if args.limit:
        session_ids = session_ids[:args.limit]
    n_debug = 2 if args.show_processing else 0
    jobs = [(int(s), list(groups.get_group(s).index.values), region_map, i < n_debug,
             args.neural_signal)
            for i, s in enumerate(session_ids)]
    print(f'processing {len(jobs)} sessions with {args.nproc} workers ...', flush=True)

    t0 = time.time()
    if len(jobs) == 1 or args.nproc <= 1:
        results = [process_session(j) for j in jobs]
    else:
        with Pool(min(args.nproc, len(jobs))) as pool:
            results = pool.map(process_session, jobs)
    t_proc = time.time() - t0
    print(f'session loading/binning took {t_proc:.1f} s '
          f'({t_proc / max(len(jobs), 1):.2f} s/session)')

    good = [r for r in results if not r['error']]
    bad = [r for r in results if r['error']]
    for r in bad:
        print(f"  EXCLUDED session {r['session_id']}: {r['error'].splitlines()[0]}")
    print(f'sessions kept: {len(good)} / {len(results)}')
    if not good:
        raise RuntimeError('no sessions converted')

    # ------------------------------------------------ global discretisation
    all_run = np.concatenate([r['running'].ravel() for r in good])
    all_pup = np.concatenate([r['pupil'].ravel() for r in good])
    run_edges = quantile_bins(all_run)
    pup_edges = quantile_bins(all_pup)
    print(f'running speed quintile edges (cm/s): {np.round(run_edges, 3)}')
    print(f'pupil diameter quintile edges (px):  {np.round(pup_edges, 3)}')
    if args.quantile_scope == 'session':
        for r in good:
            r['run_edges'] = quantile_bins(r['running'].ravel())
            r['pup_edges'] = quantile_bins(r['pupil'].ravel())
    else:
        for r in good:
            r['run_edges'] = run_edges
            r['pup_edges'] = pup_edges

    image_values = sorted({str(im) for r in good for im in np.unique(r['image_name'])})
    image_to_idx = {im: i for i, im in enumerate(image_values)}
    print(f'image identities ({len(image_values)}): {image_values}')

    output_names = ['image_identity', 'image_change', 'running_speed_quintile',
                    'pupil_diameter_quintile', 'trial_outcome']
    output_values = [
        list(image_values),
        ['no_change', 'change'],
        [f'speed_q{i + 1}' for i in range(N_QUANTILES)],
        [f'pupil_q{i + 1}' for i in range(N_QUANTILES)],
        list(OUTCOMES),
    ]

    # ------------------------------------------------------------- assemble
    data = {'neural': [], 'input': [], 'output': [], 'subjects': [], 'subject_idx': [],
            'brain_regions': [], 'brain_region_idx': [], 'input_names': [],
            'output_names': output_names, 'output_values': output_values, 'metadata': {}}
    subjects, regions = [], []
    session_info = []

    for si, r in enumerate(good):
        n_neurons, n_trials, _ = r['neural'].shape
        img = np.vectorize(lambda x: image_to_idx[str(x)])(r['image_name']).astype(np.int64)
        run_q = digitize_with(r['run_edges'], r['running'])
        pup_q = digitize_with(r['pup_edges'], r['pupil'])
        neural_trials, out_trials, in_trials = [], [], []
        for t in range(n_trials):
            neural_trials.append(np.ascontiguousarray(r['neural'][:, t, :]))
            out_trials.append(np.stack([
                img[t], r['image_change'][t], run_q[t], pup_q[t],
                np.full(N_BINS, r['outcome'][t], dtype=np.int64)]).astype(np.int64))
            in_trials.append(np.zeros((0, N_BINS), dtype=np.float32))
        data['neural'].append(neural_trials)
        data['output'].append(out_trials)
        data['input'].append(in_trials)

        if r['mouse_id'] not in subjects:
            subjects.append(r['mouse_id'])
        data['subject_idx'].append(subjects.index(r['mouse_id']))
        ridx = []
        for reg in [str(x) for x in r['regions']]:
            if reg not in regions:
                regions.append(reg)
            ridx.append(regions.index(reg))
        data['brain_region_idx'].append(np.array(ridx, dtype=np.int64))
        session_info.append(dict(ophys_session_id=r['session_id'],
                                 experiment_ids=r['experiment_ids'],
                                 mouse_id=r['mouse_id'], cre_line=r['cre_line'],
                                 session_type=r['session_type'], n_neurons=int(n_neurons),
                                 n_trials=int(n_trials),
                                 n_trials_dropped_missing_behavior=int(r['n_trials_dropped'])))

    data['subjects'] = subjects
    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
    data['brain_regions'] = regions

    data['metadata'] = dict(
        task_description=(
            'Allen Brain Observatory Visual Behavior 2P. Head-fixed mice perform a go/no-go '
            'visual change detection task: 250 ms natural-image flashes separated by 500 ms of '
            'gray screen; mice lick to report a change in image identity for a water reward. '
            'Trials are go (real change) and catch (sham change); aborted and auto-rewarded '
            'trials are excluded. Decoder outputs (all time-varying at 250 ms resolution except '
            'the trial outcome, which is static per trial and broadcast): image identity of the '
            'currently presented image, image change (1 during the 750 ms presentation interval '
            'of a real change), running speed discretised into 5 global equal-percentile bins, '
            'pupil diameter discretised into 5 global equal-percentile bins, and trial outcome '
            '(hit/miss/false_alarm/correct_reject). No decoder inputs.'),
        time_bin_size=BIN_SIZE * 1000.0,
        temporal_alignment_event=('stimulus change time of the trial (trials.change_time; the '
                                  'sham change time on catch trials), binned on the ophys '
                                  'timestamp clock'),
        off_start=OFF_START,
        off_end=OFF_END,
        neural_signal=(NEURAL_SIGNAL_DESCRIPTION[args.neural_signal]),
        n_time_bins=N_BINS,
        quantile_scope=args.quantile_scope,
        running_speed_quintile_edges_cm_per_s=run_edges.tolist(),
        pupil_diameter_quintile_edges_pixels=pup_edges.tolist(),
        dataset='Visual Behavior Ophys, AllenSDK VisualBehaviorOphysProjectCache manifest v1.1.0',
        sessions_excluded=[dict(session_id=r['session_id'], reason=r['error'].splitlines()[0])
                           for r in bad],
        session_info=session_info,
    )

    # ------------------------------------------------------------ statistics
    ntr = sum(len(x) for x in data['neural'])
    nneu = sum(x[0].shape[0] for x in data['neural'])
    print(f"\nsessions: {len(data['neural'])}, mice: {len(subjects)}, "
          f'neurons: {nneu}, trials: {ntr}')
    print(f'brain regions: {regions}')
    allout = np.concatenate([np.concatenate(o, axis=1) for o in data['output']], axis=1)
    for i, name in enumerate(output_names):
        vals, cnts = np.unique(allout[i], return_counts=True)
        frac = cnts / cnts.sum()
        print(f'  {name}: ' + ', '.join(f'{output_values[i][int(v)]}={f:.3f}'
                                        for v, f in zip(vals, frac)))
    sums = tot = nzero = nnan = 0.0
    nmax = -np.inf
    for sess_neural in data['neural']:
        arr = np.concatenate(sess_neural, axis=1)
        sums += float(arr.sum()); tot += arr.size
        nzero += float((arr == 0).sum()); nnan += float(np.isnan(arr).sum())
        nmax = max(nmax, float(arr.max()))
    print(f'  neural: mean={sums / tot:.4f}, max={nmax:.3f}, '
          f'frac zero bins={nzero / tot:.3f}, nan={int(nnan)}')

    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f"\nwrote {args.outfile} "
          f'({os.path.getsize(args.outfile) / 1e6:.1f} MB) in {time.time() - t_all:.1f} s')

    if args.show_processing:
        for si, r in enumerate(good):
            if 'debug' not in r:
                continue
            plot_processing(r, data, si, f"processing_{r['session_id']}.png")


if __name__ == '__main__':
    main()
