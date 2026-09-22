#!/usr/bin/env python3
"""
Convert the Allen Brain Observatory *Visual Behavior 2P* dataset (AllenSDK cache in
/app/data) into the decoder-ready pickle format.

Usage:
    python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Summary of the conversion (see /app/CONVERSION_NOTES.md for full justification):
  * data are read **only** through `VisualBehaviorOphysProjectCache`
  * one output "session" = one *ophys session* (all simultaneously recorded imaging
    planes concatenated along the neuron axis)
  * only active (non-passive) behaviour sessions (OPHYS_1/3/4/6)
  * trials = `go` or `catch` rows of `exp.trials` (aborted / auto_rewarded dropped),
    aligned to `change_time`, window [-3, +3] s, binned at 250 ms -> 24 bins
  * neural = detected calcium `events` (L0), averaged within each 250 ms bin
  * outputs = image identity, image change, running-speed quintile, pupil-diameter
    quintile, trial outcome
"""
import argparse
import glob
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

CACHE_DIR = '/app/data'
NWB_GLOB = '/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/*.nwb'

# ---------------------------------------------------------------- parameters
BIN_SIZE = 0.25           # s, = image presentation duration, 1/3 of the 750 ms flash cycle
OFF_START = -3.0          # s relative to the change (4 flash cycles before)
OFF_END = 3.0             # s relative to the change (4 flash cycles after)
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 24
FLASH_DURATION = 0.75     # s, image presentation interval (image + gray ISI)
NQUANTILES = 5            # percentile bins for running speed and pupil diameter
BEHAVIOR_BLOCK = 'change_detection_behavior'

# 16 natural images used in image sets A and B
IMAGE_NAMES = ['im000', 'im031', 'im035', 'im045', 'im054', 'im061', 'im062', 'im063',
               'im065', 'im066', 'im069', 'im073', 'im075', 'im077', 'im085', 'im106']
IMAGE_VALUES = IMAGE_NAMES + ['omitted']
IMAGE_TO_IDX = {n: i for i, n in enumerate(IMAGE_VALUES)}

OUTCOME_VALUES = ['hit', 'miss', 'false_alarm', 'correct_reject']
OUTPUT_NAMES = ['image_identity', 'image_change', 'running_speed', 'pupil_diameter',
                'trial_outcome']


# ---------------------------------------------------------------- utilities
def get_cache():
    from allensdk.brain_observatory.behavior.behavior_project_cache import (
        VisualBehaviorOphysProjectCache)
    return VisualBehaviorOphysProjectCache.from_local_cache(
        cache_dir=CACHE_DIR, use_static_cache=False)


def downloaded_experiment_table(cache):
    """Experiment-table rows for the NWB files that are actually in the local cache."""
    et = cache.get_ophys_experiment_table()
    ids = sorted(int(os.path.basename(f).split('_')[-1].split('.')[0])
                 for f in glob.glob(NWB_GLOB))
    return et.loc[et.index.isin(ids)].copy()


def bin_means(values, timestamps, edges):
    """Average `values` (n_signals, T) over the samples falling in each bin.

    `edges` is a flat, monotonically increasing-per-trial array of bin edges of shape
    (ntrials, nbins+1). Returns (n_signals, ntrials, nbins) and the per-bin sample count.
    Implemented with a cumulative sum + searchsorted, i.e. O(T + nbins).
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    ntrials, nedge = edges.shape
    idx = np.searchsorted(timestamps, edges.ravel(), side='left').reshape(ntrials, nedge)
    counts = np.diff(idx, axis=1)                      # (ntrials, nbins)
    csum = np.concatenate([np.zeros((values.shape[0], 1)), np.cumsum(values, axis=1)],
                          axis=1)
    sums = csum[:, idx[:, 1:]] - csum[:, idx[:, :-1]]  # (n_signals, ntrials, nbins)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = sums / counts[None, :, :]
    out[np.broadcast_to(counts[None, :, :] == 0, out.shape)] = 0.0
    return out, counts


def interpolate_nans(x, t):
    """Linearly interpolate NaNs of x over t (edges use nearest valid value)."""
    x = np.asarray(x, dtype=np.float64).copy()
    good = np.isfinite(x)
    if good.sum() < 2:
        return None
    x[~good] = np.interp(t[~good], t[good], x[good])
    return x


def quantile_bin(values, nq=NQUANTILES):
    """Discretise into `nq` equal-percentile bins. Returns int labels 0..nq-1."""
    flat = values.ravel()
    edges = np.percentile(flat, np.linspace(0, 100, nq + 1)[1:-1])
    return np.digitize(values, edges, right=False).astype(np.int64)


# ---------------------------------------------------------------- per-session work
def process_session(session_id, experiment_ids, neural_source='events',
                    show_processing=False, plot_path=None, zscore=False):
    """Build the per-trial arrays for one ophys session.

    Returns a dict (or None if the session must be dropped) with:
        neural   : list of (n_neurons, NBINS) float32
        output   : list of (5, NBINS) int64
        region_idx_names : list of str, brain region of each neuron
        info     : dict of session metadata / diagnostics
    """
    t0 = time.time()
    cache = get_cache()

    experiment_ids = list(experiment_ids)
    exps = []
    for eid in experiment_ids:
        try:
            exps.append((eid, cache.get_behavior_ophys_experiment(eid)))
        except Exception as e:                                   # pragma: no cover
            print(f'  [session {session_id}] could not load experiment {eid}: {e}')
    if not exps:
        return None
    t_load = time.time() - t0

    ref = exps[0][1]           # behaviour streams are identical across planes
    meta = ref.metadata

    # ---------------- trials: go & catch only (== trial_masks.contingent_trials)
    trials = ref.trials
    keep = (trials['go'].astype(bool) | trials['catch'].astype(bool))
    tr = trials[keep].copy()
    tr = tr[np.isfinite(tr['change_time'].values)]
    if len(tr) < 2:
        return None

    # ---------------- stimulus presentations (behaviour block only)
    sp = ref.stimulus_presentations
    beh = sp[sp['stimulus_block_name'] == BEHAVIOR_BLOCK].copy()
    pres_start = beh['start_time'].values.astype(np.float64)
    pres_image = beh['image_name'].values.astype(object)
    pres_change = beh['is_change'].astype(bool).values
    order = np.argsort(pres_start)
    pres_start, pres_image, pres_change = pres_start[order], pres_image[order], pres_change[order]

    # ---------------- behaviour streams
    run = ref.running_speed
    run_t = run['timestamps'].values.astype(np.float64)
    run_v = interpolate_nans(run['speed'].values.astype(np.float64), run_t)
    if run_v is None:
        return None

    eye = ref.eye_tracking
    if eye is None or len(eye) == 0:
        return None
    eye_t = eye['timestamps'].values.astype(np.float64)
    # pupil_area = pi * semi_major * semi_minor -> equivalent circular diameter
    pupil_d = 2.0 * np.sqrt(eye['pupil_area'].values.astype(np.float64) / np.pi)
    frac_blink = float(np.mean(~np.isfinite(pupil_d)))
    pupil_d = interpolate_nans(pupil_d, eye_t)
    if pupil_d is None:
        return None

    # ---------------- candidate trial windows, restricted to periods with data
    change_times = tr['change_time'].values.astype(np.float64)
    ophys_t = [e.ophys_timestamps for _, e in exps]
    t_lo = max([t[0] for t in ophys_t] + [run_t[0], eye_t[0], pres_start[0]])
    t_hi = min([t[-1] for t in ophys_t] + [run_t[-1], eye_t[-1],
                                           pres_start[-1] + FLASH_DURATION])
    valid = ((change_times + OFF_START) >= t_lo) & ((change_times + OFF_END) <= t_hi)
    n_dropped_window = int((~valid).sum())
    tr = tr[valid]
    change_times = change_times[valid]
    if len(tr) < 2:
        return None
    ntrials = len(tr)

    edges = change_times[:, None] + (OFF_START + BIN_SIZE * np.arange(NBINS + 1))[None, :]
    centers = 0.5 * (edges[:, :-1] + edges[:, 1:])

    # ---------------- neural: binned calcium events, planes concatenated
    neural_planes, region_names, cell_ids, zero_bins = [], [], [], 0
    for eid, e in exps:
        tbl = e.events if neural_source in ('events', 'filtered_events') else e.dff_traces
        col = {'events': 'events', 'filtered_events': 'filtered_events',
               'dff': 'dff'}[neural_source]
        if len(tbl) == 0:
            continue
        traces = np.vstack(tbl[col].values)                 # (ncells, T)
        if zscore:
            mu = traces.mean(axis=1, keepdims=True)
            sd = traces.std(axis=1, keepdims=True)
            traces = (traces - mu) / np.maximum(sd, 1e-9)
        ts = e.ophys_timestamps
        binned, counts = bin_means(traces, ts, edges)       # (ncells, ntrials, nbins)
        zero_bins += int((counts == 0).sum())
        neural_planes.append(binned.astype(np.float32))
        region_names += [e.metadata['targeted_structure']] * traces.shape[0]
        cell_ids += list(tbl.index.values)
    if not neural_planes:
        return None
    neural = np.concatenate(neural_planes, axis=0)          # (nneurons, ntrials, nbins)

    # ---------------- outputs
    # image identity + change: each bin gets the 750 ms image-presentation interval
    # whose onset most recently preceded the bin centre (paper's convention).
    pidx = np.searchsorted(pres_start, centers.ravel(), side='right') - 1
    pidx = np.clip(pidx, 0, len(pres_start) - 1)
    img_names = pres_image[pidx]
    img_identity = np.array([IMAGE_TO_IDX[n] for n in img_names],
                            dtype=np.int64).reshape(ntrials, NBINS)
    img_change = pres_change[pidx].astype(np.int64).reshape(ntrials, NBINS)

    # running speed / pupil diameter: bin-average then per-session quintiles
    run_binned = bin_means(run_v, run_t, edges)[0][0]        # (ntrials, nbins)
    pupil_binned = bin_means(pupil_d, eye_t, edges)[0][0]
    run_q = quantile_bin(run_binned)
    pupil_q = quantile_bin(pupil_binned)

    # trial outcome (static per trial, broadcast over bins)
    outcome = np.full(ntrials, -1, dtype=np.int64)
    outcome[tr['hit'].astype(bool).values] = 0
    outcome[tr['miss'].astype(bool).values] = 1
    outcome[tr['false_alarm'].astype(bool).values] = 2
    outcome[tr['correct_reject'].astype(bool).values] = 3
    ok = outcome >= 0
    n_dropped_outcome = int((~ok).sum())
    if n_dropped_outcome:
        neural = neural[:, ok, :]
        img_identity, img_change = img_identity[ok], img_change[ok]
        run_q, pupil_q = run_q[ok], pupil_q[ok]
        run_binned, pupil_binned = run_binned[ok], pupil_binned[ok]
        outcome = outcome[ok]
        tr = tr[ok]
        change_times = change_times[ok]
        edges, centers = edges[ok], centers[ok]
        ntrials = int(ok.sum())
    if ntrials < 2:
        return None

    outcome_bins = np.repeat(outcome[:, None], NBINS, axis=1)

    neural_list = [np.ascontiguousarray(neural[:, i, :]) for i in range(ntrials)]
    output_list = [np.stack([img_identity[i], img_change[i], run_q[i], pupil_q[i],
                             outcome_bins[i]], axis=0) for i in range(ntrials)]

    info = dict(
        ophys_session_id=int(session_id),
        ophys_experiment_ids=[int(i) for i, _ in exps],
        behavior_session_id=int(meta['behavior_session_id']),
        mouse_id=str(meta['mouse_id']),
        session_type=str(meta['session_type']),
        project_code=str(meta['project_code']),
        equipment=str(meta['equipment_name']),
        cre_line=str(meta['cre_line']),
        ophys_frame_rate=float(meta['ophys_frame_rate']),
        n_neurons=int(neural.shape[0]),
        n_trials=int(ntrials),
        n_go=int(tr['go'].astype(bool).sum()),
        n_catch=int(tr['catch'].astype(bool).sum()),
        n_hit=int((outcome == 0).sum()), n_miss=int((outcome == 1).sum()),
        n_fa=int((outcome == 2).sum()), n_cr=int((outcome == 3).sum()),
        n_trials_dropped_window=n_dropped_window,
        n_trials_dropped_outcome=n_dropped_outcome,
        n_empty_neural_bins=int(zero_bins),
        frac_blink=frac_blink,
        running_speed_range=[float(run_binned.min()), float(run_binned.max())],
        pupil_diameter_range=[float(pupil_binned.min()), float(pupil_binned.max())],
        load_time_s=t_load, total_time_s=time.time() - t0,
    )

    if show_processing and plot_path is not None:
        try:
            make_processing_plot(plot_path, session_id, exps, tr, change_times, edges,
                                 centers, neural, img_identity, img_change, run_t, run_v,
                                 run_binned, run_q, eye_t, pupil_d, pupil_binned, pupil_q,
                                 pres_start, pres_image, pres_change, outcome,
                                 neural_source)
        except Exception as e:                                   # pragma: no cover
            print(f'  [session {session_id}] plotting failed: {e}')

    return dict(neural=neural_list, output=output_list, region_names=region_names,
                cell_ids=cell_ids, info=info)


# ---------------------------------------------------------------- plotting
def make_processing_plot(path, session_id, exps, tr, change_times, edges, centers,
                         neural, img_identity, img_change, run_t, run_v, run_binned,
                         run_q, eye_t, pupil_d, pupil_binned, pupil_q, pres_start,
                         pres_image, pres_change, outcome, neural_source):
    """Visualise every processing step for one session (2 example trials + summaries)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ntr = min(2, len(change_times))
    fig, axes = plt.subplots(5, ntr + 1, figsize=(7 * (ntr + 1), 18))
    e0 = exps[0][1]
    ts0 = e0.ophys_timestamps
    tbl = e0.events if neural_source in ('events', 'filtered_events') else e0.dff_traces
    col = {'events': 'events', 'filtered_events': 'filtered_events', 'dff': 'dff'}[neural_source]
    traces0 = np.vstack(tbl[col].values)

    for k in range(ntr):
        ct = change_times[k]
        t0, t1 = ct + OFF_START, ct + OFF_END

        # --- row 0: raw neural traces of plane 0 with bin edges
        ax = axes[0, k]
        m = (ts0 >= t0 - 0.5) & (ts0 <= t1 + 0.5)
        ncell = min(8, traces0.shape[0])
        for c in range(ncell):
            ax.plot(ts0[m] - ct, traces0[c, m] / (traces0[c, m].max() + 1e-9) + c,
                    lw=0.8)
        for e in edges[k]:
            ax.axvline(e - ct, color='0.85', lw=0.5)
        ax.axvline(0, color='r', lw=1.5)
        ax.set_title(f'trial {k}: raw {neural_source} (plane 0, first {ncell} cells)\n'
                     f'red = change time, grey = 250 ms bin edges')
        ax.set_xlabel('time from change (s)')

        # --- row 1: binned neural
        ax = axes[1, k]
        im = ax.imshow(neural[:, k, :], aspect='auto', interpolation='nearest',
                       extent=[OFF_START, OFF_END, neural.shape[0], 0])
        ax.axvline(0, color='r', lw=1.5)
        ax.set_title(f'trial {k}: binned neural ({neural.shape[0]} neurons x {NBINS} bins)')
        ax.set_xlabel('time from change (s)'); ax.set_ylabel('neuron')
        plt.colorbar(im, ax=ax)

        # --- row 2: stimulus + image identity output
        ax = axes[2, k]
        sel = (pres_start >= t0 - 1) & (pres_start <= t1 + 1)
        for st, nm, ch in zip(pres_start[sel], pres_image[sel], pres_change[sel]):
            ax.axvspan(st - ct, st - ct + 0.25, color='C3' if ch else '0.8', alpha=0.6)
            ax.text(st - ct, 1.02, str(nm), rotation=90, fontsize=6)
        ax.step(centers[k] - ct, img_identity[k], where='mid', label='image_identity')
        ax.step(centers[k] - ct, img_change[k] * 5, where='mid', color='r',
                label='image_change x5')
        ax.axvline(0, color='r', lw=1.5)
        ax.set_xlim(OFF_START - 0.5, OFF_END + 0.5)
        ax.legend(fontsize=7)
        ax.set_title(f'trial {k}: stimulus flashes (grey=repeat, red=change) vs outputs\n'
                     f'{"GO" if tr.iloc[k]["go"] else "CATCH"}, outcome={OUTCOME_VALUES[outcome[k]]}')
        ax.set_xlabel('time from change (s)')

        # --- row 3: running speed
        ax = axes[3, k]
        m = (run_t >= t0 - 0.5) & (run_t <= t1 + 0.5)
        ax.plot(run_t[m] - ct, run_v[m], color='0.5', lw=0.8, label='raw (60 Hz)')
        ax.step(centers[k] - ct, run_binned[k], where='mid', color='C0', label='binned mean')
        ax2 = ax.twinx()
        ax2.step(centers[k] - ct, run_q[k], where='mid', color='C1', label='quintile')
        ax2.set_ylim(-0.5, 4.5); ax2.set_ylabel('quintile')
        ax.axvline(0, color='r', lw=1.5)
        ax.legend(fontsize=7, loc='upper left')
        ax.set_title(f'trial {k}: running speed (cm/s) -> quintile')
        ax.set_xlabel('time from change (s)')

        # --- row 4: pupil
        ax = axes[4, k]
        m = (eye_t >= t0 - 0.5) & (eye_t <= t1 + 0.5)
        ax.plot(eye_t[m] - ct, pupil_d[m], color='0.5', lw=0.8, label='raw/interp (px)')
        ax.step(centers[k] - ct, pupil_binned[k], where='mid', color='C0', label='binned mean')
        ax2 = ax.twinx()
        ax2.step(centers[k] - ct, pupil_q[k], where='mid', color='C1', label='quintile')
        ax2.set_ylim(-0.5, 4.5); ax2.set_ylabel('quintile')
        ax.axvline(0, color='r', lw=1.5)
        ax.legend(fontsize=7, loc='upper left')
        ax.set_title(f'trial {k}: pupil diameter -> quintile')
        ax.set_xlabel('time from change (s)')

    # --- summary column
    ax = axes[0, ntr]
    ax.hist(neural.ravel(), bins=100, log=True)
    ax.set_title('distribution of binned neural values')

    ax = axes[1, ntr]
    ax.plot(np.nanmean(neural, axis=(0, 1)))
    ax.set_xticks(np.arange(0, NBINS + 1, 4))
    ax.set_xticklabels(OFF_START + BIN_SIZE * np.arange(0, NBINS + 1, 4))
    ax.axvline(-OFF_START / BIN_SIZE - 0.5, color='r')
    ax.set_title('trial-averaged population activity\n(should rise after the change)')
    ax.set_xlabel('time from change (s)')

    ax = axes[2, ntr]
    ax.imshow(img_change, aspect='auto', interpolation='nearest',
              extent=[OFF_START, OFF_END, len(change_times), 0])
    ax.set_title('image_change output (trials x bins)\n1 only during the changed flash')
    ax.set_xlabel('time from change (s)')

    ax = axes[3, ntr]
    ax.hist(run_binned.ravel(), bins=60)
    for q in np.percentile(run_binned.ravel(), [20, 40, 60, 80]):
        ax.axvline(q, color='r')
    ax.set_title('binned running speed with quintile edges')

    ax = axes[4, ntr]
    ax.hist(pupil_binned.ravel(), bins=60)
    for q in np.percentile(pupil_binned.ravel(), [20, 40, 60, 80]):
        ax.axvline(q, color='r')
    ax.set_title('binned pupil diameter with quintile edges')

    fig.suptitle(f'Processing steps, ophys_session_id {session_id}')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    import matplotlib.pyplot as plt2
    plt2.close(fig)


def _worker(args):
    session_id, exp_ids, neural_source, show, path, zscore = args
    try:
        return session_id, process_session(session_id, exp_ids, neural_source, show, path, zscore)
    except Exception as e:                                       # pragma: no cover
        import traceback
        traceback.print_exc()
        print(f'  [session {session_id}] FAILED: {e}')
        return session_id, None


# ---------------------------------------------------------------- main
def main():
    global BIN_SIZE, NBINS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', default=True)
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    ap.add_argument('--neural', default='dff',
                    choices=['events', 'filtered_events', 'dff'])
    ap.add_argument('--bin', type=float, default=BIN_SIZE, help='bin size in seconds')
    ap.add_argument('--zscore', action='store_true', help='z-score each neuron over the session')
    ap.add_argument('--nsample', type=int, default=2,
                    help='number of sessions to process in --sample mode')
    ap.add_argument('--workers', type=int, default=24)
    args = ap.parse_args()

    if abs(args.bin - BIN_SIZE) > 1e-12:
        BIN_SIZE = args.bin
        NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
        print(f'bin size set to {BIN_SIZE}s -> {NBINS} bins per trial')
    t_start = time.time()
    cache = get_cache()
    et = downloaded_experiment_table(cache)
    act = et[~et['passive'].astype(bool)].copy()
    print(f'downloaded experiments: {len(et)}; active (non-passive): {len(act)}')
    print(f'active ophys sessions: {act["ophys_session_id"].nunique()}, '
          f'mice: {act["mouse_id"].nunique()}')

    groups = act.groupby('ophys_session_id')
    session_ids = sorted(groups.groups.keys())
    if args.sample:
        # one single-plane and one multi-plane session, for fast testing
        sp = act[act.project_code == 'VisualBehavior']['ophys_session_id'].unique()
        mp = act[act.project_code == 'VisualBehaviorMultiscope']['ophys_session_id'].unique()
        n = args.nsample
        nmp = max(1, n // 2)
        session_ids = [int(x) for x in sp[:n - nmp]] + [int(x) for x in mp[:nmp]]
    print(f'processing {len(session_ids)} sessions with neural source "{args.neural}"')

    jobs = []
    for i, sid in enumerate(session_ids):
        exp_ids = list(groups.get_group(sid).index.values)
        show = args.show_processing and i < 2
        path = f'/app/processing_{sid}.png' if show else None
        jobs.append((sid, exp_ids, args.neural, show, path, args.zscore))

    results = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, j): j[0] for j in jobs}
        for n, fut in enumerate(as_completed(futs), 1):
            sid, res = fut.result()
            results[sid] = res
            if res is None:
                print(f'[{n}/{len(jobs)}] session {sid}: DROPPED')
            else:
                i = res['info']
                print(f'[{n}/{len(jobs)}] session {sid}: {i["n_neurons"]} neurons, '
                      f'{i["n_trials"]} trials (go {i["n_go"]}, catch {i["n_catch"]}), '
                      f'{i["total_time_s"]:.1f}s '
                      f'[{n / (time.time() - t0):.2f} sessions/s]', flush=True)

    # ------------- assemble
    data = dict(neural=[], input=[], output=[], subjects=[], subject_idx=[],
                brain_regions=[], brain_region_idx=[], input_names=[],
                output_names=OUTPUT_NAMES,
                output_values=[IMAGE_VALUES,
                               ['no_change', 'change'],
                               [f'quintile_{i}' for i in range(NQUANTILES)],
                               [f'quintile_{i}' for i in range(NQUANTILES)],
                               OUTCOME_VALUES],
                metadata={})
    subjects, regions, session_infos = [], [], []
    for sid in session_ids:
        res = results.get(sid)
        if res is None:
            continue
        info = res['info']
        data['neural'].append(res['neural'])
        data['output'].append(res['output'])
        ntr = len(res['neural'])
        nb = res['neural'][0].shape[1]
        data['input'].append([np.zeros((0, nb), dtype=np.float32) for _ in range(ntr)])
        mouse = info['mouse_id']
        if mouse not in subjects:
            subjects.append(mouse)
        data['subject_idx'].append(subjects.index(mouse))
        for r in res['region_names']:
            if r not in regions:
                regions.append(r)
        data['brain_region_idx'].append(
            np.array([regions.index(r) for r in res['region_names']], dtype=np.int64))
        session_infos.append(info)

    data['subjects'] = subjects
    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
    data['brain_regions'] = regions

    ntrials = sum(len(s) for s in data['neural'])
    nneurons = sum(s[0].shape[0] for s in data['neural'])
    data['metadata'] = dict(
        task_description=(
            'Allen Brain Observatory Visual Behavior 2P: mice perform a go/no-go visual '
            'change-detection task with flashed natural images (250 ms image, 500 ms gray, '
            '750 ms cycle, 8 images per session) while 2-photon calcium imaging is '
            'performed in VISp/VISl. Trials are go (image change) and catch (sham change) '
            'trials; aborted and auto-rewarded trials are excluded. The decoder predicts, '
            'from the population calcium-event activity alone: (0) the identity of the '
            'currently presented image (16 images of image sets A and B, plus omitted '
            'flashes), (1) whether the current image presentation is an image change, '
            '(2) the running speed quintile, (3) the pupil-diameter quintile, and '
            '(4) the trial outcome (hit/miss/false alarm/correct reject).'),
        time_bin_size=BIN_SIZE * 1000.0,
        temporal_alignment_event=('time of the image change (`trials.change_time`; the sham '
                                  'change time on catch trials), which always coincides with '
                                  'a stimulus flash onset'),
        off_start=OFF_START,
        off_end=OFF_END,
        neural_signal=(f'{args.neural}: detected calcium events (L0 event detection, Allen '
                       'pipeline), averaged over the ophys frames in each 250 ms bin'
                       if args.neural != 'dff' else
                       'dF/F, averaged over the ophys frames in each 250 ms bin'),
        neural_unit='mean calcium-event magnitude per bin (dF/F units)',
        n_sessions=len(data['neural']), n_trials=ntrials, n_neurons_total=nneurons,
        n_subjects=len(subjects),
        alignment_note=('all data streams (2P frames, running, eye tracking, stimulus, '
                        'trials) are on the common sync clock exposed by the AllenSDK; '
                        'behaviour streams are averaged within the same 250 ms bins as the '
                        'neural data'),
        trial_selection='trials.go | trials.catch (aborted and auto_rewarded excluded)',
        session_selection=('active (non-passive) behaviour+ophys sessions of the Visual '
                           'Behavior 2P release present in the local cache; all imaging '
                           'planes recorded simultaneously in one ophys session are '
                           'concatenated into a single population'),
        neuron_selection=('all cells released by the AllenSDK, i.e. valid ROIs only '
                          '(exclude_invalid_rois=True is the SDK default)'),
        discretization=('running speed and pupil diameter are averaged in each bin and then '
                        'discretised into 5 equal-percentile (quintile) bins computed within '
                        'each session'),
        source='AllenSDK VisualBehaviorOphysProjectCache, visual-behavior-ophys-1.1.0',
        session_info=session_infos,
    )

    # ------------- sanity checks
    print('\n--- sanity checks ---')
    ngo = sum(i['n_go'] for i in session_infos)
    ncatch = sum(i['n_catch'] for i in session_infos)
    print(f'sessions kept: {len(data["neural"])} / {len(session_ids)}')
    print(f'total trials: {ntrials}, total neurons: {nneurons}, mice: {len(subjects)}')
    print(f'go fraction of kept trials: {ngo / max(ngo + ncatch, 1):.4f} (expected ~0.875)')
    nhit = sum(i['n_hit'] for i in session_infos); nmiss = sum(i['n_miss'] for i in session_infos)
    nfa = sum(i['n_fa'] for i in session_infos); ncr = sum(i['n_cr'] for i in session_infos)
    print(f'outcomes: hit {nhit}, miss {nmiss}, false_alarm {nfa}, correct_reject {ncr}')
    print(f'hit+miss == go: {nhit + nmiss == ngo}; fa+cr == catch: {nfa + ncr == ncatch}')
    assert all(t.shape[1] == NBINS for s in data['neural'] for t in s)
    assert all(o.shape == (len(OUTPUT_NAMES), NBINS) for s in data['output'] for o in s)
    for s, b in zip(data['neural'], data['brain_region_idx']):
        assert s[0].shape[0] == len(b)
    outs = np.concatenate([o for s in data['output'] for o in s], axis=1)
    for i, nm in enumerate(OUTPUT_NAMES):
        v, c = np.unique(outs[i], return_counts=True)
        print(f'output {i} ({nm}): values {v.tolist()} fractions '
              f'{np.round(c / c.sum(), 4).tolist()}')
    print(f'empty neural bins: {sum(i["n_empty_neural_bins"] for i in session_infos)}')
    print(f'trials dropped (window outside data): '
          f'{sum(i["n_trials_dropped_window"] for i in session_infos)}')

    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f'\nwrote {args.outfile} '
          f'({os.path.getsize(args.outfile) / 1e6:.1f} MB) in {time.time() - t_start:.1f} s')


if __name__ == '__main__':
    main()
