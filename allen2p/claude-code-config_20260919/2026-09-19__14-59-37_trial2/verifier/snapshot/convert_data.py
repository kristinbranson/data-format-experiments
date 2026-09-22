#!/usr/bin/env python3
"""
Convert the Allen Brain Observatory "Visual Behavior 2P" dataset (visual-behavior-ophys-1.1.0)
into the decoder-compatible pickle format described in the task.

Usage:
    python -u /app/convert_data.py <outpicklefile> [--full | --sample]
                                   [--show-processing] [--trace {filtered_events,events,dff}]
                                   [--workers N] [--sessions id1,id2,...]

Design decisions (see /app/CONVERSION_NOTES.md for full justification):
  * session            = one `ophys_session_id` (= one continuous recording). For Multiscope
                         sessions the simultaneously recorded imaging planes ("experiments") are
                         concatenated along the neuron axis.
  * sessions used      = active behavior only (passive OPHYS_2/OPHYS_5 excluded), and only sessions
                         that have eye-tracking data (pupil diameter is a required output).
  * trials             = rows of the SDK `trials` table with `go | catch` (this excludes `aborted`
                         and `auto_rewarded`), window [start_time, stop_time).
  * temporal alignment = ophys frame times. Each trial is divided into bins of BIN_SIZE = 1/31 s
                         starting at `start_time`; the neural value of a bin is the ophys frame
                         nearest the bin centre (identity mapping for the 31 Hz single-plane rigs,
                         zero-order hold for the 11 Hz Multiscope planes).
  * neural             = `dff_traces.dff` (the detrended dF/F produced by the Allen pipeline and
                         described in the whitepaper's "DF/F CALCULATION" section). `events` /
                         `filtered_events` are selectable with --trace; see CONVERSION_NOTES.md
                         Step 7 for the decoding comparison that motivated this default.
  * inputs             = none (dinput = 0), per the task specification.
  * outputs            = [image identity (17), image change (2), running-speed quintile (5),
                          pupil-diameter quintile (5), trial outcome (4)]
"""
import argparse
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

DATA_ROOT = '/app/data/visual-behavior-ophys-1.1.0'
EXP_DIR = os.path.join(DATA_ROOT, 'behavior_ophys_experiments')
META_DIR = os.path.join(DATA_ROOT, 'project_metadata')

# --------------------------------------------------------------------------------------
# Fixed conversion parameters
# --------------------------------------------------------------------------------------
BIN_SIZE = 1.0 / 31.0          # s; nominal single-plane 2P frame period (metadata ophys_frame_rate)
N_QUANTILE_BINS = 5            # "five equal percentile bins"
GRAY_LABEL = 'gray'            # image-identity class for blank/omitted periods
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
BEHAVIOR_BLOCK = 'change_detection_behavior'


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def nearest_index(sorted_times: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Index of the element of `sorted_times` nearest to each element of `query`."""
    idx = np.searchsorted(sorted_times, query)
    idx = np.clip(idx, 1, len(sorted_times) - 1)
    left = sorted_times[idx - 1]
    right = sorted_times[idx]
    idx = np.where(query - left <= right - query, idx - 1, idx)
    return np.clip(idx, 0, len(sorted_times) - 1)


def interp_nonan(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Linear interpolation of y(x) at xq, ignoring NaN samples of y (e.g. blinks)."""
    ok = np.isfinite(y) & np.isfinite(x)
    if not np.any(ok):
        return np.full(xq.shape, np.nan)
    return np.interp(xq, x[ok], y[ok])


def quantile_bin(values: np.ndarray, nbins: int = N_QUANTILE_BINS):
    """Discretise into `nbins` equal-percentile bins. Returns (labels, edges)."""
    edges = np.quantile(values, np.arange(1, nbins) / nbins)
    labels = np.searchsorted(edges, values, side='right').astype(np.int64)
    return np.clip(labels, 0, nbins - 1), edges


def load_experiment(eid: int):
    from allensdk.brain_observatory.behavior.behavior_ophys_experiment import (
        BehaviorOphysExperiment)
    return BehaviorOphysExperiment.from_nwb_path(
        os.path.join(EXP_DIR, 'behavior_ophys_experiment_%d.nwb' % eid))


# --------------------------------------------------------------------------------------
# Per-session conversion
# --------------------------------------------------------------------------------------
def convert_session(job):
    """Convert one ophys session (possibly several imaging planes) to trial lists.

    job = dict(ophys_session_id, experiment_ids (list), trace, show_processing, out_prefix)
    Returns a dict with per-session neural/output/metadata, or {'error': ...}.
    """
    t_start = time.time()
    sid = job['ophys_session_id']
    eids = job['experiment_ids']
    trace_name = job['trace']
    timing = {}

    try:
        # ---------------- load all planes of this session ----------------
        t0 = time.time()
        experiments = [load_experiment(e) for e in eids]
        timing['load_nwb'] = time.time() - t0

        ds0 = experiments[0]
        # Behavior streams are session-level; every plane of a Multiscope session carries an
        # identical copy. Use the first plane, but fall back to another plane if a stream is
        # empty there (defensive: eye tracking is occasionally missing).
        trials = ds0.trials
        stim = ds0.stimulus_presentations
        running = ds0.running_speed
        eye = ds0.eye_tracking
        for ds in experiments[1:]:
            if len(eye) == 0:
                eye = ds.eye_tracking
            if len(running) == 0:
                running = ds.running_speed
        if len(eye) == 0 or not np.any(np.isfinite(eye['pupil_width'].values)):
            return {'ophys_session_id': sid, 'error': 'no eye tracking data'}
        if len(running) == 0:
            return {'ophys_session_id': sid, 'error': 'no running data'}

        # consistency of behavior data across planes
        for ds in experiments[1:]:
            assert len(ds.trials) == len(trials), 'trial tables differ across planes'
            assert np.allclose(ds.trials.start_time.values, trials.start_time.values), \
                'trial start times differ across planes'

        # ---------------- trial selection ----------------
        # go | catch  ==  exclude aborted and auto-rewarded, keep Go and Catch trials
        sel = trials[(trials.go | trials.catch)].sort_values('start_time')
        assert not sel.aborted.any() and not sel.auto_rewarded.any()
        assert sel.change_time.notna().all()

        # ---------------- stimulus table (behavior block only) ----------------
        beh = stim[stim.stimulus_block_name == BEHAVIOR_BLOCK].sort_values('start_time')
        flash_start = beh.start_time.values.astype(float)
        flash_end = beh.end_time.values.astype(float)
        flash_img = beh.image_name.values.astype(str)
        flash_omitted = beh.omitted.values.astype(bool)
        # omitted "flashes" are gray screen; make sure they never match a bin
        flash_end = np.where(flash_omitted | ~np.isfinite(flash_end), -np.inf, flash_end)
        flash_is_change = beh.is_change.values.astype(bool)

        # ---------------- behavior time series ----------------
        run_t = running['timestamps'].values.astype(float)
        run_v = running['speed'].values.astype(float)
        eye_t = eye['timestamps'].values.astype(float)
        eye_v = eye['pupil_width'].values.astype(float)

        # ---------------- neural traces of every plane ----------------
        t0 = time.time()
        plane_traces, plane_ts, plane_region, plane_cellid, plane_eid = [], [], [], [], []
        for ds, eid in zip(experiments, eids):
            if trace_name == 'dff':
                tbl = ds.dff_traces
                mat = np.stack(tbl['dff'].values).astype(np.float32)
            else:
                tbl = ds.events
                mat = np.stack(tbl[trace_name].values).astype(np.float32)
            ts = np.asarray(ds.ophys_timestamps, dtype=float)
            assert mat.shape[1] == len(ts), 'trace length != n ophys timestamps'
            plane_traces.append(mat)
            plane_ts.append(ts)
            plane_region.append(ds.metadata['targeted_structure'])
            plane_cellid.append(np.asarray(tbl.index.values))
            plane_eid.append(eid)
        timing['load_traces'] = time.time() - t0

        # ---------------- per-trial assembly ----------------
        t0 = time.time()
        neural_trials, img_trials, chg_trials = [], [], []
        run_trials, pup_trials, outcome_trials = [], [], []
        trial_info = []
        image_names = sorted(set(flash_img[~flash_omitted]) - {'omitted'})
        img_lookup = {name: i + 1 for i, name in enumerate(image_names)}  # 0 = gray

        for tid, tr in sel.iterrows():
            t_beg, t_stop = float(tr.start_time), float(tr.stop_time)
            nbins = int(np.floor((t_stop - t_beg) / BIN_SIZE))
            if nbins < 2:
                continue
            # bin centres (the sample time of each bin)
            tc = t_beg + (np.arange(nbins) + 0.5) * BIN_SIZE

            # --- require full ophys coverage of the trial ---
            if any(tc[0] < ts[0] or tc[-1] > ts[-1] for ts in plane_ts):
                continue

            # --- neural: nearest ophys frame of each plane ---
            mats = [mat[:, nearest_index(ts, tc)]
                    for mat, ts in zip(plane_traces, plane_ts)]
            neural = np.concatenate(mats, axis=0) if len(mats) > 1 else mats[0]

            # --- image identity (0 = gray / omitted / blank) ---
            j = np.searchsorted(flash_start, tc, side='right') - 1
            j = np.clip(j, 0, len(flash_start) - 1)
            on_screen = (tc >= flash_start[j]) & (tc < flash_end[j])
            img = np.zeros(nbins, dtype=np.int64)
            if np.any(on_screen):
                img[on_screen] = [img_lookup[n] for n in flash_img[j[on_screen]]]

            # --- image change: bins inside the changed-image flash (go trials only) ---
            chg = np.zeros(nbins, dtype=np.int64)
            if bool(tr.go):
                k = int(np.searchsorted(flash_start, float(tr.change_time), side='right') - 1)
                assert flash_is_change[k], 'change_time does not match an is_change flash'
                chg[(tc >= flash_start[k]) & (tc < flash_end[k])] = 1

            # --- running speed and pupil diameter (continuous, binned later) ---
            run_trials.append(interp_nonan(run_t, run_v, tc))
            pup_trials.append(interp_nonan(eye_t, eye_v, tc))

            # --- trial outcome ---
            if bool(tr.hit):
                outcome = 0
            elif bool(tr.miss):
                outcome = 1
            elif bool(tr.false_alarm):
                outcome = 2
            elif bool(tr.correct_reject):
                outcome = 3
            else:
                continue  # go/catch trial with no outcome flag: should not happen
            neural_trials.append(np.ascontiguousarray(neural, dtype=np.float32))
            img_trials.append(img)
            chg_trials.append(chg)
            outcome_trials.append(outcome)
            trial_info.append(dict(trials_id=int(tid), start_time=t_beg, stop_time=t_stop,
                                   change_time=float(tr.change_time), nbins=nbins,
                                   go=bool(tr.go), catch=bool(tr.catch), outcome=outcome))
        timing['trials'] = time.time() - t0

        if len(neural_trials) < 2:
            return {'ophys_session_id': sid, 'error': 'fewer than 2 usable trials'}

        # ---------------- session-level quantile binning of running & pupil ----------------
        run_all = np.concatenate(run_trials)
        pup_all = np.concatenate(pup_trials)
        if not np.all(np.isfinite(pup_all)):
            return {'ophys_session_id': sid, 'error': 'non-finite pupil after interpolation'}
        if not np.all(np.isfinite(run_all)):
            return {'ophys_session_id': sid, 'error': 'non-finite running speed'}
        run_lab, run_edges = quantile_bin(run_all)
        pup_lab, pup_edges = quantile_bin(pup_all)

        outputs = []
        pos = 0
        for i, n in enumerate([len(x) for x in run_trials]):
            out = np.empty((5, n), dtype=np.int64)
            out[0] = img_trials[i]
            out[1] = chg_trials[i]
            out[2] = run_lab[pos:pos + n]
            out[3] = pup_lab[pos:pos + n]
            out[4] = outcome_trials[i]
            outputs.append(out)
            pos += n
        assert pos == len(run_all)

        inputs = [np.zeros((0, x.shape[1]), dtype=np.float32) for x in neural_trials]

        brain_region_per_neuron = np.concatenate(
            [[reg] * mat.shape[0] for reg, mat in zip(plane_region, plane_traces)])

        md0 = ds0.metadata
        info = dict(
            ophys_session_id=int(sid),
            behavior_session_id=int(md0['behavior_session_id']),
            experiment_ids=[int(e) for e in eids],
            mouse_id=str(md0['mouse_id']),
            cre_line=md0['cre_line'],
            session_type=md0['session_type'],
            project_code=md0['project_code'],
            equipment_name=md0['equipment_name'],
            ophys_frame_rate=float(md0['ophys_frame_rate']),
            imaging_depths=[int(ds.metadata['imaging_depth']) for ds in experiments],
            targeted_structures=[ds.metadata['targeted_structure'] for ds in experiments],
            n_neurons=int(sum(m.shape[0] for m in plane_traces)),
            n_trials=len(neural_trials),
            n_timepoints=int(sum(x.shape[1] for x in neural_trials)),
            image_names=image_names,
            running_quantile_edges=[float(x) for x in run_edges],
            pupil_quantile_edges=[float(x) for x in pup_edges],
            n_trials_go=int(sum(t['go'] for t in trial_info)),
            n_trials_catch=int(sum(t['catch'] for t in trial_info)),
            n_trials_table_go=int(trials.go.sum()),
            n_trials_table_catch=int(trials.catch.sum()),
            n_trials_table_aborted=int(trials.aborted.sum()),
            n_trials_table_auto=int(trials.auto_rewarded.sum()),
        )

        result = dict(
            ophys_session_id=int(sid),
            neural=neural_trials,
            input=inputs,
            output=outputs,
            image_names=image_names,
            brain_region_per_neuron=brain_region_per_neuron,
            mouse_id=str(md0['mouse_id']),
            cell_specimen_ids=np.concatenate(plane_cellid),
            info=info,
            trial_info=trial_info,
            timing=timing,
        )

        # ---------------- optional diagnostic plots ----------------
        if job.get('show_processing'):
            # plotting must never lose a session's data
            try:
                _plot_processing(job, experiments, sel, beh, running, eye, plane_traces, plane_ts,
                                 neural_trials, outputs, trial_info, run_trials, pup_trials,
                                 run_edges, pup_edges, image_names, trace_name)
            except Exception as exc:
                import traceback
                print('WARNING: plotting failed for session %d: %r\n%s'
                      % (sid, exc, traceback.format_exc()), flush=True)

        timing['total'] = time.time() - t_start
        return result
    except Exception as exc:  # pragma: no cover - defensive
        import traceback
        return {'ophys_session_id': sid, 'error': repr(exc),
                'traceback': traceback.format_exc()}


# --------------------------------------------------------------------------------------
# Diagnostic plotting
# --------------------------------------------------------------------------------------
def _plot_processing(job, experiments, sel, beh, running, eye, plane_traces, plane_ts,
                     neural_trials, outputs, trial_info, run_trials, pup_trials,
                     run_edges, pup_edges, image_names, trace_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sid = job['ophys_session_id']
    fig, axes = plt.subplots(7, 2, figsize=(26, 24))

    # ---- column 0: raw session-level streams over a 40 s window around trial 3's change ----
    ti = trial_info[3] if len(trial_info) > 3 else trial_info[0]
    t0, t1 = ti['start_time'] - 5, ti['start_time'] + 35
    ax = axes[0, 0]
    ts = plane_ts[0]
    m = (ts >= t0) & (ts <= t1)
    for i in range(min(8, plane_traces[0].shape[0])):
        ax.plot(ts[m], plane_traces[0][i][m] / max(1e-9, plane_traces[0][i][m].max()) + i, lw=.8)
    ax.set_title('RAW neural (%s), plane 0, first 8 cells, native ophys timestamps' % trace_name)
    ax.set_xlim(t0, t1)

    ax = axes[1, 0]
    fl = beh[(beh.start_time >= t0 - 1) & (beh.start_time <= t1)]
    for _, f in fl.iterrows():
        if f.omitted:
            ax.axvline(f.start_time, color='r', ls=':')
        else:
            ax.axvspan(f.start_time, f.end_time,
                       color=plt.cm.tab20(image_names.index(f.image_name) % 20), alpha=.7)
        if f.is_change:
            ax.axvline(f.start_time, color='k', lw=2)
    ax.set_title('RAW stimulus presentations (colour=image, black=change, red dotted=omitted)')
    ax.set_xlim(t0, t1)

    ax = axes[2, 0]
    m = (running['timestamps'].values >= t0) & (running['timestamps'].values <= t1)
    ax.plot(running['timestamps'].values[m], running['speed'].values[m], 'k-', label='raw 60 Hz')
    ax.set_title('RAW running speed (cm/s)')
    ax.set_xlim(t0, t1)
    ax.legend()

    ax = axes[3, 0]
    m = (eye['timestamps'].values >= t0) & (eye['timestamps'].values <= t1)
    ax.plot(eye['timestamps'].values[m], eye['pupil_width'].values[m], 'k.-',
            label='raw 30 Hz (NaN = blink)')
    ax.set_title('RAW pupil width (pixels)')
    ax.set_xlim(t0, t1)
    ax.legend()

    # trial boundaries on all left-column axes
    for ax in axes[:4, 0]:
        for ti2 in trial_info[:8]:
            ax.axvline(ti2['start_time'], color='g', lw=1.5)
            ax.axvline(ti2['change_time'], color='m', lw=1, ls='--')

    # ---- converted versions of the same trial ----
    k = 3 if len(trial_info) > 3 else 0
    tinfo = trial_info[k]
    n = neural_trials[k].shape[1]
    tt = np.arange(n) * BIN_SIZE
    ax = axes[4, 0]
    for i in range(min(8, neural_trials[k].shape[0])):
        ax.plot(tt, neural_trials[k][i] / max(1e-9, neural_trials[k][i].max()) + i, lw=.8)
    ax.set_title('CONVERTED neural, trial %d (aligned to trial start)' % k)
    ax.set_xlabel('time from trial start (s)')

    ax = axes[5, 0]
    ax.step(tt, outputs[k][0], where='post', label='image identity (0=gray)')
    ax.step(tt, outputs[k][1] * 5, where='post', label='image change x5')
    ax.axvline(tinfo['change_time'] - tinfo['start_time'], color='m', ls='--', label='change_time')
    ax.legend()
    ax.set_title('CONVERTED outputs 0,1 for trial %d' % k)

    ax = axes[6, 0]
    ax.step(tt, outputs[k][2], where='post', label='running quintile')
    ax.step(tt, outputs[k][3], where='post', label='pupil quintile')
    ax.step(tt, outputs[k][4], where='post', label='outcome (%s)' %
            OUTCOME_NAMES[tinfo['outcome']])
    ax.legend()
    ax.set_title('CONVERTED outputs 2,3,4 for trial %d' % k)
    ax.set_xlabel('time from trial start (s)')

    # ---- column 1: alignment checks / discretisation ----
    ax = axes[0, 1]
    # neural conversion check: overlay raw trace and converted samples for one cell/trial
    # pick the cell of plane 0 that is most active *in this trial*, so the check is informative
    n0 = plane_traces[0].shape[0]
    cell = int(np.argmax(neural_trials[k][:n0].max(axis=1))) if n0 else 0
    tb = tinfo['start_time'] + (np.arange(n) + .5) * BIN_SIZE
    m = (plane_ts[0] >= tb[0] - .5) & (plane_ts[0] <= tb[-1] + .5)
    ax.plot(plane_ts[0][m], plane_traces[0][cell][m], 'k-', lw=2, label='raw frames')
    ax.plot(tb, neural_trials[k][cell], 'r.--', ms=4, label='converted bins')
    ax.set_title('neural alignment check, cell %d, trial %d' % (cell, k))
    ax.legend()

    ax = axes[1, 1]
    m = (running['timestamps'].values >= tb[0] - .5) & (running['timestamps'].values <= tb[-1] + .5)
    ax.plot(running['timestamps'].values[m], running['speed'].values[m], 'k-', label='raw')
    ax.plot(tb, run_trials[k], 'r.--', ms=4, label='interpolated onto bins')
    for e in run_edges:
        ax.axhline(e, color='b', lw=.5)
    ax.set_title('running speed alignment + quintile edges (blue)')
    ax.legend()

    ax = axes[2, 1]
    m = (eye['timestamps'].values >= tb[0] - .5) & (eye['timestamps'].values <= tb[-1] + .5)
    ax.plot(eye['timestamps'].values[m], eye['pupil_width'].values[m], 'k.-', label='raw')
    ax.plot(tb, pup_trials[k], 'r.--', ms=4, label='interpolated onto bins')
    for e in pup_edges:
        ax.axhline(e, color='b', lw=.5)
    ax.set_title('pupil alignment + quintile edges (blue)')
    ax.legend()

    ax = axes[3, 1]
    allrun = np.concatenate(run_trials)
    ax.hist(allrun, bins=100)
    for e in run_edges:
        ax.axvline(e, color='r')
    ax.set_yscale('log')
    ax.set_title('session running-speed distribution with quintile edges')

    ax = axes[4, 1]
    allpup = np.concatenate(pup_trials)
    ax.hist(allpup, bins=100)
    for e in pup_edges:
        ax.axvline(e, color='r')
    ax.set_title('session pupil-width distribution with quintile edges')

    ax = axes[5, 1]
    frac = [np.mean(np.concatenate([o[i] for o in outputs]) == v)
            for i, vals in enumerate([range(1 + len(image_names)), range(2), range(5),
                                      range(5), range(4)]) for v in vals]
    ax.bar(np.arange(len(frac)), frac)
    ax.set_title('fraction of timepoints per output value (img|chg|run|pupil|outcome)')

    ax = axes[6, 1]
    # image identity around every change in the session (sanity: identity flips at change_time)
    prechg, postchg = [], []
    for i, ti2 in enumerate(trial_info):
        if not ti2['go']:
            continue
        o = outputs[i][0]
        c = outputs[i][1]
        w = np.flatnonzero(c == 1)
        if len(w) == 0:
            continue
        postchg.append(o[w[0]])
        pre = o[:w[0]]
        pre = pre[pre > 0]
        if len(pre):
            prechg.append(pre[-1])
    ax.plot(prechg[:100], 'o-', label='image before change')
    ax.plot(postchg[:100], 's-', label='image at change')
    ax.legend()
    ax.set_title('image identity before/at change (should differ on every go trial)')

    fig.suptitle('Processing steps, ophys_session_id %d (%s, %s, %d planes, %d neurons, %d trials)'
                 % (sid, experiments[0].metadata['session_type'],
                    experiments[0].metadata['project_code'], len(experiments),
                    neural_trials[0].shape[0], len(neural_trials)))
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(job.get('plot_dir', '/app'), 'processing_%d.png' % sid), dpi=90)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def select_sessions():
    """Return a DataFrame of the downloaded, active-behavior experiments grouped by session."""
    exp = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
    have = set()
    for f in os.listdir(EXP_DIR):
        if f.endswith('.nwb'):
            have.add(int(f.split('_')[-1].split('.')[0]))
    exp = exp[exp.ophys_experiment_id.isin(have)]
    exp = exp[~exp.passive]                      # active behavior sessions only
    exp = exp.sort_values(['ophys_session_id', 'ophys_experiment_id'])
    return exp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', help='process all sessions (default)')
    ap.add_argument('--sample', action='store_true', help='process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true',
                    help='save processing_<session_id>.png for up to 2 sessions')
    ap.add_argument('--trace', default='dff',
                    choices=['dff', 'filtered_events', 'events'])
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--sessions', default=None, help='comma separated ophys_session_ids')
    ap.add_argument('--plot-dir', default='/app')
    args = ap.parse_args()

    t_start = time.time()
    exp = select_sessions()
    groups = exp.groupby('ophys_session_id')
    session_ids = list(groups.groups.keys())
    if args.sessions:
        wanted = [int(x) for x in args.sessions.split(',')]
        session_ids = [s for s in session_ids if s in wanted]
    elif args.sample:
        # one single-plane and one Multiscope session, so both paths are exercised
        ms = [s for s in session_ids
              if (exp[exp.ophys_session_id == s].project_code == 'VisualBehaviorMultiscope').all()]
        sp = [s for s in session_ids if s not in ms]
        session_ids = [sp[0], ms[0]] if ms else sp[:2]

    print('Sessions to convert: %d (from %d experiments)' %
          (len(session_ids), sum(len(groups.get_group(s)) for s in session_ids)), flush=True)

    jobs = []
    for i, sid in enumerate(session_ids):
        g = groups.get_group(sid)
        jobs.append(dict(ophys_session_id=int(sid),
                         experiment_ids=g.ophys_experiment_id.tolist(),
                         trace=args.trace,
                         show_processing=args.show_processing and i < 2,
                         plot_dir=args.plot_dir))

    results = []
    nworkers = min(args.workers, max(1, len(jobs)))
    if nworkers > 1:
        with ProcessPoolExecutor(max_workers=nworkers) as ex:
            for i, r in enumerate(ex.map(convert_session, jobs)):
                results.append(r)
                el = time.time() - t_start
                print('[%3d/%3d] session %d  %s  (elapsed %.0fs, %.1fs/session, eta %.0fs)'
                      % (i + 1, len(jobs), r['ophys_session_id'],
                         'ERROR: ' + r['error'] if 'error' in r else
                         'neurons=%d trials=%d T=%d load=%.1fs' %
                         (r['info']['n_neurons'], r['info']['n_trials'],
                          r['info']['n_timepoints'], r['timing']['load_nwb']),
                         el, el / (i + 1), el / (i + 1) * (len(jobs) - i - 1)), flush=True)
    else:
        for i, job in enumerate(jobs):
            r = convert_session(job)
            results.append(r)
            print('[%3d/%3d] session %d %s' % (i + 1, len(jobs), r['ophys_session_id'],
                                               r.get('error', 'ok')), flush=True)

    ok = [r for r in results if 'error' not in r]
    bad = [r for r in results if 'error' in r]
    print('\nConverted %d sessions, %d excluded' % (len(ok), len(bad)))
    for r in bad:
        print('  excluded session %d: %s' % (r['ophys_session_id'], r['error']))
        if 'traceback' in r:
            print(r['traceback'])

    # ---------------- assemble final dictionary ----------------
    ok.sort(key=lambda r: r['ophys_session_id'])
    subjects = sorted({r['mouse_id'] for r in ok})
    brain_regions = sorted({reg for r in ok for reg in set(r['brain_region_per_neuron'])})
    image_names = sorted({n for r in ok for n in r['image_names']})
    output_values = [
        [GRAY_LABEL] + image_names,
        ['no_change', 'change'],
        ['run_q%d' % i for i in range(N_QUANTILE_BINS)],
        ['pupil_q%d' % i for i in range(N_QUANTILE_BINS)],
        OUTCOME_NAMES,
    ]
    # remap per-session image labels (which used the session's own image list) to global indices
    global_img = {name: i + 1 for i, name in enumerate(image_names)}
    for r in ok:
        remap = np.zeros(len(r['image_names']) + 1, dtype=np.int64)
        for local, name in enumerate(r['image_names']):
            remap[local + 1] = global_img[name]
        for out in r['output']:
            out[0] = remap[out[0]]

    data = {
        'neural': [r['neural'] for r in ok],
        'input': [r['input'] for r in ok],
        'output': [r['output'] for r in ok],
        'subjects': subjects,
        'subject_idx': np.array([subjects.index(r['mouse_id']) for r in ok], dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': [np.array([brain_regions.index(x)
                                       for x in r['brain_region_per_neuron']], dtype=np.int64)
                             for r in ok],
        'input_names': [],
        'output_names': ['image_identity', 'image_change', 'running_speed_quintile',
                         'pupil_diameter_quintile', 'trial_outcome'],
        'output_values': output_values,
        'metadata': {
            'task_description':
                'Allen Brain Observatory Visual Behavior 2P: head-fixed mice perform a go/no-go '
                'visual change-detection task while 2-photon calcium imaging is performed in '
                'visual cortex. Natural images are flashed for 250 ms every 750 ms (5% of flashes '
                'omitted); mice earn water rewards by licking within 150-750 ms of a change in '
                'image identity. Trials are Go (image change) or Catch (sham change); Aborted '
                '(early lick) and Auto-rewarded (free reward) trials are excluded. Decoder inputs: '
                'none. Decoder outputs (all per time bin; trial outcome constant within a trial): '
                'identity of the image on the screen (gray screen = "gray"), whether an image '
                'change is on the screen, running speed quintile, pupil-diameter quintile, and '
                'trial outcome (hit/miss/false alarm/correct rejection).',
            'time_bin_size': 1000.0 * BIN_SIZE,
            'temporal_alignment_event':
                'Trial start (trials.start_time of the AllenSDK trials table). Time bins run '
                'forward from trial start; neural data are the ophys (2-photon) frames nearest '
                'each bin centre, and all behavioural/stimulus streams are sampled at the same '
                'bin centres, i.e. everything is aligned on the ophys timestamps.',
            'off_start': 0.0,
            'off_end': None,
            'trial_definition':
                'Rows of the AllenSDK trials table with go|catch; window [start_time, stop_time). '
                'Trial length is variable (mean 8.5 s) because it is set by the behaviour program: '
                'the change (or sham change) occurs 2.25-8.25 s after trial start and the trial '
                'ends ~4.2 s later.',
            'neural_data_type': args.trace,
            'neural_description':
                {'filtered_events': 'AllenSDK events.filtered_events: detected calcium events '
                                    '(magnitudes) convolved with a causal half-normal kernel '
                                    '(scale 2/31 s), as used for all neural analyses in Piet et '
                                    'al. 2024 (detected calcium events).',
                 'events': 'AllenSDK events.events: detected calcium event magnitudes.',
                 'dff': 'AllenSDK dff_traces.dff: detrended dF/F.'}[args.trace],
            'ophys_frame_rate_hz': {r['info']['ophys_frame_rate'] for r in ok}.__str__(),
            'output_descriptions': [
                'image_identity: which natural image is on the screen in this time bin; "gray" '
                'covers the 500 ms inter-flash gray screen and omitted flashes. Image sets A and B '
                'are disjoint (8 images each), so a session only shows 8 of the 16 images.',
                'image_change: 1 for the time bins inside the 250 ms presentation of the changed '
                'image on Go trials (i.e. right after the image identity changes), 0 elsewhere. '
                'Catch trials contain a sham change (the image does not change) and are all 0.',
                'running_speed_quintile: running speed (cm/s, AllenSDK running_speed, 60 Hz, '
                '10 Hz low-pass) linearly interpolated onto the bin centres and discretised into '
                'five equal-percentile bins computed within each session.',
                'pupil_diameter_quintile: pupil width (pixels, AllenSDK eye_tracking.pupil_width, '
                '~30 Hz, NaN on blinks/outliers) interpolated across blinks and onto the bin '
                'centres, then discretised into five equal-percentile bins within each session.',
                'trial_outcome: hit / miss (Go trials) or false alarm / correct rejection (Catch '
                'trials); constant over the trial.',
            ],
            'curation':
                'Kept: active-behavior ophys sessions (OPHYS_1/3/4/6) present in the local data '
                'directory, with eye tracking. Excluded: passive sessions (no licking/reward, so '
                'no trial outcome), sessions without eye-tracking data, aborted and auto-rewarded '
                'trials, and trials not fully covered by the ophys recording. All ROIs returned by '
                'the SDK are used: ROI quality control (motion border, unions, duplicates, '
                'dendrites, low SNR) is already applied by the Allen pipeline.',
            'session_info': [r['info'] for r in ok],
            'excluded_sessions': [{'ophys_session_id': r['ophys_session_id'],
                                   'reason': r['error']} for r in bad],
            'source': 'visual-behavior-ophys-1.1.0 (Allen Institute), NWB files read with '
                      'AllenSDK 2.16.2 BehaviorOphysExperiment.from_nwb_path',
        },
    }

    # ---------------- summary + integrity checks ----------------
    nneurons = [n.shape[0] for r in ok for n in [r['neural'][0]]]
    ntrials = [len(r['neural']) for r in ok]
    ntime = [sum(x.shape[1] for x in r['neural']) for r in ok]
    print('\nSummary: %d sessions, %d mice, %d neurons, %d trials, %d timepoints'
          % (len(ok), len(subjects), sum(nneurons), sum(ntrials), sum(ntime)))
    print('  neurons/session: mean %.1f min %d max %d' %
          (np.mean(nneurons), min(nneurons), max(nneurons)))
    print('  trials/session:  mean %.1f min %d max %d' %
          (np.mean(ntrials), min(ntrials), max(ntrials)))
    print('  brain regions: %s' % ', '.join(
        '%s=%d' % (b, sum(int((idx == i).sum()) for idx in data['brain_region_idx']))
        for i, b in enumerate(brain_regions)))
    for i, name in enumerate(data['output_names']):
        allv = np.concatenate([o[i] for r in ok for o in r['output']])
        cnt = np.bincount(allv, minlength=len(output_values[i]))
        print('  %s: %s' % (name, ', '.join(
            '%s=%.4f' % (v, c / cnt.sum()) for v, c in zip(output_values[i], cnt))))

    print('\nWriting %s ...' % args.outfile, flush=True)
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Wrote %s (%.2f GB) in %.1f s total'
          % (args.outfile, os.path.getsize(args.outfile) / 1e9, time.time() - t_start))


if __name__ == '__main__':
    main()
