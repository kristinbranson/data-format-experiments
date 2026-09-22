#!/usr/bin/env python
"""Convert the Allen Brain Observatory Visual Behavior 2P dataset into the
decoder-benchmark pickle format.

Usage:
    python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

See /app/CONVERSION_NOTES.md for the full rationale of every decision.

Summary of the conversion
-------------------------
* Data are read **only** through the AllenSDK `VisualBehaviorOphysProjectCache`.
* Session          = one `ophys_session_id` (all of its active imaging planes are
                     concatenated along the neuron axis).
* Trial            = one `go` or `catch` trial of `dataset.trials`
                     (aborted and auto-rewarded trials are dropped, as instructed).
* Alignment event  = `trials.change_time` (real change on go trials, sham change on
                     catch trials).  It coincides exactly with a stimulus flash onset.
* Window           = [-2.0, +3.0] s around the change, binned into 250 ms bins
                     (20 bins/trial).  250 ms = the image duration and 1/3 of the
                     750 ms flash cycle, so bins are phase locked to the stimulus.
* Neural           = mean dF/F (`dataset.dff_traces`) in each bin.  The reference
                     paper used the L0 calcium events; a controlled comparison on 20
                     sessions showed dF/F decodes better on 4 of 5 outputs and removes
                     the empty-trial problem of the very sparse Vip/Sst planes, so dF/F
                     is the default.  `--neural-signal events|filtered_events|dff`
                     reproduces the paper's choice (events are summed per bin, dF/F is
                     averaged, because dF/F is a rate-like quantity and the number of
                     ophys frames per bin differs between the 31 Hz and 11 Hz rigs).
* Outputs          = image identity (16 classes), image change (1 in the single
                     250 ms bin that starts with a change), running-speed quintile and
                     pupil-diameter quintile (per-session equal-percentile bins) - all
                     time varying - and the trial outcome (hit/miss/false_alarm/
                     correct_reject, static per trial, broadcast over the bins).
* Inputs           = none (per the decoder task specification).
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

# ---------------------------------------------------------------- parameters
BIN_SIZE = 0.25          # s, = image duration, 1/3 of the 750 ms flash cycle
OFF_START = -2.0         # s relative to the change (start of trial window)
OFF_END = 3.0            # s relative to the change (end of trial window)
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 20
FLASH_INTERVAL = 0.75    # s, image presentation interval (image + gray)
NQUANTILES = 5           # quintiles for running speed and pupil diameter
OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']


def get_cache():
    from allensdk.brain_observatory.behavior.behavior_project_cache import (
        VisualBehaviorOphysProjectCache)
    return VisualBehaviorOphysProjectCache.from_local_cache(cache_dir=CACHE_DIR)


# ------------------------------------------------------------------ helpers
def bin_edges_for_trials(change_times):
    """(ntrials, NBINS+1) array of bin edge times (s) for every trial."""
    offsets = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
    return change_times[:, None] + offsets[None, :]


def bin_sum(values_2d, sample_times, edges, chunk=256):
    """Sum `values_2d` (nrows, nsamples) inside each [edge_i, edge_i+1) bin.

    Implemented with a cumulative sum so that empty bins give exactly 0 and bin
    edges outside the sampled interval are handled gracefully.
    Returns (nrows, ntrials, NBINS).
    """
    nrows = values_2d.shape[0]
    idx = np.searchsorted(sample_times, edges.ravel(), side='left')
    out = np.empty((nrows, edges.shape[0], NBINS), dtype=np.float64)
    for a in range(0, nrows, chunk):
        b = min(a + chunk, nrows)
        c = np.concatenate(
            [np.zeros((b - a, 1), dtype=np.float64),
             np.cumsum(values_2d[a:b].astype(np.float64), axis=1)], axis=1)
        v = c[:, idx].reshape(b - a, edges.shape[0], NBINS + 1)
        out[a:b] = np.diff(v, axis=2)
    return out


def bin_mean_1d(values, sample_times, edges):
    """Mean of a 1-D time series inside each bin -> (ntrials, NBINS).

    Bins without samples (should not happen for 30/60 Hz streams) fall back to
    linear interpolation at the bin centre.
    """
    v = values[None, :]
    ones = np.ones_like(v)
    s = bin_sum(v, sample_times, edges)[0]
    n = bin_sum(ones, sample_times, edges)[0]
    with np.errstate(invalid='ignore', divide='ignore'):
        m = s / n
    if np.any(n == 0):
        centres = edges[:, :-1] + BIN_SIZE / 2.0
        fill = np.interp(centres, sample_times, values)
        m = np.where(n == 0, fill, m)
    return m


def interpolate_nans(values):
    """Linearly interpolate NaNs (blinks); edges are filled with nearest value."""
    v = np.asarray(values, dtype=float).copy()
    good = np.isfinite(v)
    if not np.any(good):
        return None
    if not np.all(good):
        x = np.arange(len(v))
        v[~good] = np.interp(x[~good], x[good], v[good])
    return v


def flash_table(dataset):
    """Change-detection flashes with a forward-filled image name."""
    sp = dataset.stimulus_presentations
    sp = sp[sp.stimulus_block_name.str.contains('change_detection')]
    sp = sp.sort_values('start_time')
    names = sp.image_name.astype(str).replace('omitted', np.nan).ffill().bfill().values
    return (sp.start_time.values.astype(float), names,
            sp.is_change.values.astype(bool), sp)


# --------------------------------------------------------- per-session worker
def process_session(args):
    """Build one session of the dataset.  Runs in a worker process."""
    ophys_session_id, experiment_ids, verbose, signal, window, change_window = args
    global OFF_START, OFF_END, NBINS
    OFF_START, OFF_END, NBINS = window
    t_start = time.time()
    bc = get_cache()
    timings = {}

    neural_planes = []
    region_per_neuron = []
    meta = None
    trials = None
    behaviour = None

    for k, eid in enumerate(experiment_ids):
        t0 = time.time()
        ds = bc.get_behavior_ophys_experiment(int(eid))
        timings['load'] = timings.get('load', 0) + time.time() - t0

        if meta is None:
            meta = dict(ds.metadata)
            trials = ds.trials
            keep = trials[(trials.go | trials.catch)]
            if len(keep) < 2:
                return None
            change_times = keep.change_time.values.astype(float)
            edges = bin_edges_for_trials(change_times)
            behaviour = dict(keep=keep, edges=edges, dataset=ds)

        t0 = time.time()
        if signal == 'dff':
            ev = ds.dff_traces
            E = np.vstack(ev['dff'].values).astype(np.float32)
        else:
            ev = ds.events
            E = np.vstack(ev[signal].values).astype(np.float32)   # (ncells, nframes)
        ots = np.asarray(ds.ophys_timestamps, dtype=float)
        edges = behaviour['edges']
        binned = bin_sum(E, ots, edges)                            # (ncells, ntrials, NBINS)
        if signal == 'dff':
            # dF/F is a *rate-like* quantity: the correct bin statistic is the mean,
            # otherwise bins that happen to contain 7 vs 8 ophys frames (31 Hz) or
            # 2 vs 3 frames (11 Hz) would differ by a spurious scale factor.
            counts = bin_sum(np.ones((1, E.shape[1]), dtype=np.float32), ots, edges)
            binned = binned / np.maximum(counts, 1.0)
        neural_planes.append(binned)
        region_per_neuron += [ds.metadata['targeted_structure']] * E.shape[0]
        timings['neural'] = timings.get('neural', 0) + time.time() - t0

    if meta is None:
        return None

    ds = behaviour['dataset']
    keep = behaviour['keep']
    edges = behaviour['edges']
    ntrials = len(keep)
    neural = np.concatenate(neural_planes, axis=0)                 # (nneurons, ntrials, NBINS)

    # ------------------------------------------------- stimulus based outputs
    t0 = time.time()
    starts, names, is_change, sp = flash_table(ds)
    centres = edges[:, :-1] + BIN_SIZE / 2.0                       # (ntrials, NBINS)
    fidx = np.searchsorted(starts, centres.ravel(), side='right') - 1
    fidx = np.clip(fidx, 0, len(starts) - 1)
    image_name = names[fidx].reshape(ntrials, NBINS)
    if change_window == 'interval':
        # 1 for every bin of the 750 ms presentation interval that starts with a change
        change_flag = is_change[fidx].reshape(ntrials, NBINS).astype(np.int64)
    else:
        # 1 only for the single 250 ms bin that starts at the change
        change_flag = np.zeros((ntrials, NBINS), dtype=np.int64)
        flash_start = starts[fidx].reshape(ntrials, NBINS)
        first_bin = (centres - flash_start) < BIN_SIZE
        change_flag[is_change[fidx].reshape(ntrials, NBINS) & first_bin] = 1
    timings['stimulus'] = time.time() - t0

    # ------------------------------------------------- behaviour (continuous)
    t0 = time.time()
    rs = ds.running_speed
    speed = np.asarray(rs['speed'].values, dtype=float)
    speed_t = np.asarray(rs['timestamps'].values, dtype=float)
    speed = interpolate_nans(speed)
    running_binned = bin_mean_1d(speed, speed_t, edges)            # (ntrials, NBINS)

    et = ds.eye_tracking
    pupil_binned = None
    if len(et) > 0:
        diam = 2.0 * np.nanmax(
            np.vstack([et['pupil_width'].values, et['pupil_height'].values]), axis=0)
        diam = interpolate_nans(diam)
        if diam is not None:
            pupil_binned = bin_mean_1d(diam, np.asarray(et['timestamps'].values, float),
                                       edges)
    timings['behaviour'] = time.time() - t0
    if pupil_binned is None:
        # a session without any usable eye tracking cannot supply the pupil output
        return dict(skip=True, ophys_session_id=int(ophys_session_id),
                    reason='no eye tracking')

    # --------------------------------------------------------- trial outcome
    outcome = np.full(ntrials, -1, dtype=np.int64)
    for i, name in enumerate(OUTCOMES):
        outcome[keep[name].values.astype(bool)] = i
    assert np.all(outcome >= 0), 'trial with no hit/miss/FA/CR outcome'

    result = dict(
        skip=False,
        ophys_session_id=int(ophys_session_id),
        experiment_ids=[int(e) for e in experiment_ids],
        neural=neural.astype(np.float32),
        image_name=image_name,
        image_change=change_flag,
        running=running_binned.astype(np.float64),
        pupil=pupil_binned.astype(np.float64),
        outcome=outcome,
        change_times=keep.change_time.values.astype(float),
        go=keep.go.values.astype(bool),
        catch=keep.catch.values.astype(bool),
        regions=region_per_neuron,
        mouse_id=str(meta['mouse_id']),
        cre_line=meta['cre_line'],
        session_type=meta['session_type'],
        equipment=meta['equipment_name'],
        frame_rate=float(meta['ophys_frame_rate']),
        nplanes=len(experiment_ids),
        elapsed=time.time() - t_start,
        timings=timings,
    )
    if verbose:
        print(f"  session {ophys_session_id}: {neural.shape[0]} neurons, "
              f"{ntrials} trials, {len(experiment_ids)} plane(s), "
              f"{result['elapsed']:.1f}s {timings}", flush=True)
    return result


# ------------------------------------------------------------------- driver
def select_sessions(sample=False):
    """Active (non-passive) ophys sessions and their experiment ids."""
    bc = get_cache()
    et = bc.get_ophys_experiment_table()
    local_ids = set(int(f.split('_')[-1].split('.')[0]) for f in os.listdir(
        os.path.join(CACHE_DIR, 'visual-behavior-ophys-1.1.0',
                     'behavior_ophys_experiments')))
    et = et.loc[sorted(set(et.index).intersection(local_ids))]
    et = et[~et.passive]                      # active behaviour sessions only
    groups = []
    for sid, sub in et.groupby('ophys_session_id'):
        groups.append((int(sid), sorted(int(i) for i in sub.index)))
    groups.sort()
    if sample:
        # one single-plane and one multi-plane session, for a quick test
        multi = [g for g in groups if len(g[1]) > 1]
        single = [g for g in groups if len(g[1]) == 1]
        groups = [single[0], multi[0]]
    return groups, et


def discretize(values_list, nq=NQUANTILES, scope='session'):
    """Quantile (equal-percentile) binning of a list of (ntrials, NBINS) arrays.

    scope='session': the percentile bins are computed separately for each session.
        This is the default because both measurements are only comparable *within*
        a session: the pupil diameter is in camera pixels (the eye-camera zoom and
        distance differ between rigs and sessions) and each animal/session has its
        own running baseline.  Every session then contributes exactly 20 % of its
        bins to each of the five classes.
    scope='global': one set of percentile bins for the whole dataset.
    """
    allv = np.concatenate([v.ravel() for v in values_list])
    global_qs = np.percentile(allv, np.linspace(0, 100, nq + 1))
    if scope == 'global':
        out = [np.digitize(v, global_qs[1:-1], right=False).astype(np.int64)
               for v in values_list]
    else:
        out = []
        for v in values_list:
            qs = np.percentile(v.ravel(), np.linspace(0, 100, nq + 1))
            out.append(np.digitize(v, qs[1:-1], right=False).astype(np.int64))
    return out, global_qs


def make_plots(res, images, run_q, pupil_q, outpath):
    """Visualise every processing step for one session."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    bc = get_cache()
    ds = bc.get_behavior_ophys_experiment(res['experiment_ids'][0])
    starts, names, is_change, sp = flash_table(ds)
    ev = ds.events
    E = np.vstack(ev['events'].values).astype(np.float32)
    ots = np.asarray(ds.ophys_timestamps, float)
    rs = ds.running_speed
    et = ds.eye_tracking
    diam = interpolate_nans(2.0 * np.nanmax(
        np.vstack([et['pupil_width'].values, et['pupil_height'].values]), axis=0))
    raw_diam = 2.0 * np.nanmax(
        np.vstack([et['pupil_width'].values, et['pupil_height'].values]), axis=0)

    trial = int(np.argmax(res['go']))          # first go trial
    ct = res['change_times'][trial]
    t0, t1 = ct + OFF_START, ct + OFF_END
    tb = OFF_START + BIN_SIZE * (np.arange(NBINS) + 0.5)

    fig, ax = plt.subplots(6, 1, figsize=(14, 18), sharex=True)
    # 1. raw events of a few cells + bin edges
    m = (ots >= t0 - 1) & (ots <= t1 + 1)
    for c in range(min(5, E.shape[0])):
        ax[0].plot(ots[m] - ct, E[c][m], lw=0.8, label=f'cell {c}')
    for e in OFF_START + BIN_SIZE * np.arange(NBINS + 1):
        ax[0].axvline(e, color='0.85', lw=0.5)
    ax[0].set_title(f"raw calcium events (session {res['ophys_session_id']}, trial {trial}, "
                    f"go={res['go'][trial]}) - vertical lines: 250 ms bin edges")
    ax[0].legend(fontsize=6, ncol=5)

    # 2. binned neural
    im = ax[1].imshow(res['neural'][:, trial, :], aspect='auto', origin='lower',
                      extent=[OFF_START, OFF_END, 0, res['neural'].shape[0]])
    ax[1].set_title('binned event magnitude (neurons x bins)')

    # 3. stimulus: flashes, image identity, change flag
    fm = (starts >= t0 - 1) & (starts <= t1 + 1)
    for s, nm, ch in zip(starts[fm], names[fm], is_change[fm]):
        ax[2].axvspan(s - ct, s - ct + 0.25, color='red' if ch else '0.6', alpha=0.4)
    ax[2].plot(tb, [images.index(x) for x in res['image_name'][trial]], 'o-', label='image identity idx')
    ax[2].plot(tb, res['image_change'][trial] * (len(images) - 1), 's--', label='image change')
    ax[2].axvline(0, color='k', lw=1)
    ax[2].set_title('stimulus flashes (grey=image, red=change) with decoded outputs')
    ax[2].legend(fontsize=7)

    # 4. running speed raw vs binned vs quintile
    rm = (rs.timestamps.values >= t0 - 1) & (rs.timestamps.values <= t1 + 1)
    ax[3].plot(rs.timestamps.values[rm] - ct, rs.speed.values[rm], color='0.5', label='raw 60 Hz')
    ax[3].step(tb, res['running'][trial], where='mid', color='C0', label='bin mean')
    ax3b = ax[3].twinx()
    ax3b.step(tb, run_q[trial], where='mid', color='C3', label='quintile')
    ax3b.set_ylim(-0.5, 4.5)
    ax[3].set_title('running speed (cm/s) and its quintile')
    ax[3].legend(fontsize=7, loc='upper left')

    # 5. pupil raw vs interpolated vs binned vs quintile
    em = (et.timestamps.values >= t0 - 1) & (et.timestamps.values <= t1 + 1)
    ax[4].plot(et.timestamps.values[em] - ct, raw_diam[em], color='0.5', label='raw (NaN at blinks)')
    ax[4].plot(et.timestamps.values[em] - ct, diam[em], color='C2', lw=0.8, label='blink-interpolated')
    ax[4].step(tb, res['pupil'][trial], where='mid', color='C0', label='bin mean')
    ax4b = ax[4].twinx()
    ax4b.step(tb, pupil_q[trial], where='mid', color='C3', label='quintile')
    ax4b.set_ylim(-0.5, 4.5)
    ax[4].set_title('pupil diameter (px) and its quintile')
    ax[4].legend(fontsize=7, loc='upper left')

    # 6. licks / rewards / outcome
    licks = ds.licks.timestamps.values
    rew = ds.rewards.timestamps.values
    lm = (licks >= t0) & (licks <= t1)
    rwm = (rew >= t0) & (rew <= t1)
    ax[5].plot(licks[lm] - ct, np.zeros(lm.sum()), 'k|', ms=20, label='licks')
    ax[5].plot(rew[rwm] - ct, np.zeros(rwm.sum()) + 0.2, 'bd', label='rewards')
    ax[5].set_ylim(-0.5, 0.5)
    ax[5].set_title(f"behaviour; trial outcome = {OUTCOMES[res['outcome'][trial]]}")
    ax[5].set_xlabel('time from change (s)')
    ax[5].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outpath, dpi=110)
    plt.close(fig)

    # second figure: distributions over the whole session
    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    ax[0].hist(res['running'].ravel(), 50)
    ax[0].set_title('running bin means (session)')
    ax[1].hist(run_q.ravel(), np.arange(6) - 0.5)
    ax[1].set_title('running quintiles')
    ax[2].hist(res['pupil'].ravel(), 50)
    ax[2].set_title('pupil bin means (session)')
    ax[3].hist(pupil_q.ravel(), np.arange(6) - 0.5)
    ax[3].set_title('pupil quintiles')
    fig.tight_layout()
    fig.savefig(outpath.replace('.png', '_dist.png'), dpi=110)
    plt.close(fig)


def main():
    global OFF_START, OFF_END, NBINS
    ap = argparse.ArgumentParser()
    ap.add_argument('outfile')
    ap.add_argument('--full', action='store_true', default=True)
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--quantile-scope', choices=['session', 'global'], default='session')
    ap.add_argument('--neural-signal', choices=['events', 'filtered_events', 'dff'], default='dff')
    ap.add_argument('--subset', type=int, default=0, help='convert only the first N sessions (testing)')
    ap.add_argument('--change-window', choices=['interval', 'bin'], default='bin')
    ap.add_argument('--off-start', type=float, default=-2.0)
    ap.add_argument('--off-end', type=float, default=3.0)
    args = ap.parse_args()

    OFF_START, OFF_END = args.off_start, args.off_end
    NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
    t_all = time.time()
    groups, et = select_sessions(sample=args.sample)
    print(f'window [{OFF_START}, {OFF_END}] s -> {NBINS} bins of {BIN_SIZE*1000:.0f} ms', flush=True)
    print(f'{len(groups)} active ophys sessions selected '
          f'({sum(len(g[1]) for g in groups)} imaging planes)', flush=True)

    if args.subset:
        rng = np.random.RandomState(0)
        idx = sorted(rng.choice(len(groups), min(args.subset, len(groups)), replace=False))
        groups = [groups[i] for i in idx]
        print(f'--subset: using {len(groups)} sessions', flush=True)
    jobs = [(sid, eids, True, args.neural_signal, (OFF_START, OFF_END, NBINS), args.change_window)
            for sid, eids in groups]
    t0 = time.time()
    if args.workers > 1 and len(jobs) > 1:
        with Pool(min(args.workers, len(jobs))) as p:
            results = p.map(process_session, jobs)
    else:
        results = [process_session(j) for j in jobs]
    t_sessions = time.time() - t0
    print(f'session processing took {t_sessions:.1f}s '
          f'({t_sessions / max(len(jobs), 1):.1f}s per session)', flush=True)

    skipped = [r for r in results if r is None or r.get('skip')]
    for r in skipped:
        if r is not None:
            print(f"  SKIPPED session {r['ophys_session_id']}: {r['reason']}", flush=True)
    results = [r for r in results if r is not None and not r.get('skip')]
    print(f'{len(results)} sessions kept, {len(skipped)} skipped', flush=True)

    # ------------------------------------------------- global discretization
    run_q, run_edges = discretize([r['running'] for r in results], scope=args.quantile_scope)
    pup_q, pup_edges = discretize([r['pupil'] for r in results], scope=args.quantile_scope)
    print('quantile scope:', args.quantile_scope, flush=True)
    print('running speed quintile edges (cm/s):', np.round(run_edges, 3), flush=True)
    print('pupil diameter quintile edges (px):', np.round(pup_edges, 2), flush=True)

    images = sorted(set(np.concatenate([r['image_name'].ravel() for r in results])))
    print(f'{len(images)} unique images:', images, flush=True)
    img_index = {n: i for i, n in enumerate(images)}

    regions = sorted(set(sum([r['regions'] for r in results], [])))
    subjects = sorted(set(r['mouse_id'] for r in results))

    data = dict(neural=[], input=[], output=[], subjects=subjects,
                subject_idx=[], brain_regions=regions, brain_region_idx=[],
                input_names=[],
                output_names=['image_identity', 'image_change',
                              'running_speed_quintile', 'pupil_diameter_quintile',
                              'trial_outcome'],
                output_values=[images, ['no_change', 'change'],
                               [f'q{i+1}' for i in range(NQUANTILES)],
                               [f'q{i+1}' for i in range(NQUANTILES)],
                               OUTCOMES])
    session_info = []
    for i, r in enumerate(results):
        nneurons, ntrials, T = r['neural'].shape
        img_idx = np.vectorize(img_index.get)(r['image_name']).astype(np.int64)
        neural_trials, input_trials, output_trials = [], [], []
        for t in range(ntrials):
            neural_trials.append(np.ascontiguousarray(r['neural'][:, t, :], dtype=np.float32))
            input_trials.append(np.zeros((0, T), dtype=np.float32))
            out = np.stack([img_idx[t], r['image_change'][t], run_q[i][t], pup_q[i][t],
                            np.full(T, r['outcome'][t], dtype=np.int64)], axis=0)
            output_trials.append(out.astype(np.int64))
        data['neural'].append(neural_trials)
        data['input'].append(input_trials)
        data['output'].append(output_trials)
        data['subject_idx'].append(subjects.index(r['mouse_id']))
        data['brain_region_idx'].append(
            np.array([regions.index(x) for x in r['regions']], dtype=np.int64))
        session_info.append(dict(ophys_session_id=r['ophys_session_id'],
                                 experiment_ids=r['experiment_ids'],
                                 mouse_id=r['mouse_id'], cre_line=r['cre_line'],
                                 session_type=r['session_type'],
                                 equipment=r['equipment'],
                                 ophys_frame_rate=r['frame_rate'],
                                 n_neurons=int(nneurons), n_trials=int(ntrials),
                                 n_go=int(r['go'].sum()), n_catch=int(r['catch'].sum())))
    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)

    data['metadata'] = dict(
        task_description=(
            'Visual change detection (Allen Brain Observatory Visual Behavior 2P). '
            'Head-fixed mice view a series of natural images (250 ms image, 500 ms gray) '
            'and lick to report a change in image identity. Trials are go (real change) '
            'and catch (sham change) trials; aborted and auto-rewarded trials are excluded. '
            'Decoded from 2-photon calcium events: the identity of the presented image, '
            'whether an image change just occurred, the running speed quintile, the pupil '
            'diameter quintile (all time varying) and the trial outcome '
            '(hit/miss/false_alarm/correct_reject, static per trial).'),
        time_bin_size=BIN_SIZE * 1000.0,
        temporal_alignment_event=('stimulus change time (trials.change_time): the onset of the '
                                  'changed image on go trials, of the sham change on catch trials'),
        off_start=OFF_START,
        off_end=OFF_END,
        neural_signal=('sum of L0-detected calcium event magnitudes per 250 ms bin '
                       f'(dataset.events["{args.neural_signal}"])'),
        quantile_scope=args.quantile_scope,
        running_speed_quintile_edges_cm_per_s=run_edges.tolist(),
        pupil_diameter_quintile_edges_px=pup_edges.tolist(),
        pupil_diameter_definition='2 * max(pupil_width, pupil_height) of the SDK ellipse fit, blinks linearly interpolated',
        change_window=args.change_window,
        image_identity_definition=('image whose 750 ms presentation interval contains the bin centre; '
                                   'omitted flashes inherit the previous image'),
        session_definition='one ophys_session_id; all active imaging planes concatenated as neurons',
        excluded_data=('passive sessions (no licking), aborted and auto-rewarded trials, '
                       'sessions without eye tracking'),
        source='AllenSDK VisualBehaviorOphysProjectCache, visual-behavior-ophys-1.1.0',
        session_info=session_info,
    )

    # ----------------------------------------------------------- sanity checks
    ntrials_total = sum(len(s) for s in data['neural'])
    nneurons_total = sum(len(b) for b in data['brain_region_idx'])
    print(f"\nSANITY: {len(data['neural'])} sessions, {len(subjects)} mice, "
          f'{nneurons_total} neurons, {ntrials_total} trials', flush=True)
    ngo = sum(s['n_go'] for s in session_info)
    ncatch = sum(s['n_catch'] for s in session_info)
    print(f'SANITY: go {ngo}, catch {ncatch}, catch fraction {ncatch/(ngo+ncatch):.4f} '
          f'(whitepaper ~0.125)', flush=True)
    allout = np.concatenate([np.concatenate(o, axis=1) for o in data['output']], axis=1)
    for d, name in enumerate(data['output_names']):
        vals, cnts = np.unique(allout[d], return_counts=True)
        print(f'SANITY: {name}: ' +
              ', '.join(f'{data["output_values"][d][v]}={c/allout.shape[1]:.3f}'
                        for v, c in zip(vals, cnts)), flush=True)
    # change flag must be on exactly for bins 8..10 of go trials and never on catch trials
    bad_go = bad_catch = 0
    expect = np.zeros(NBINS, dtype=np.int64)
    nchange_bins = int(FLASH_INTERVAL / BIN_SIZE) if args.change_window == 'interval' else 1
    expect[int((0 - OFF_START) / BIN_SIZE):int((0 - OFF_START) / BIN_SIZE) + nchange_bins] = 1
    for i, r in enumerate(results):
        for t in range(r['image_change'].shape[0]):
            if r['go'][t] and not np.array_equal(r['image_change'][t], expect):
                bad_go += 1
            if r['catch'][t] and r['image_change'][t].sum() > 0:
                bad_catch += 1
    print(f'SANITY: go trials with unexpected change pattern: {bad_go}; '
          f'catch trials with a change flag: {bad_catch} (both should be 0)', flush=True)

    # ------------------------------------------------------------------ plots
    if args.show_processing:
        for i, r in enumerate(results[:2]):
            out = f"/app/processing_{r['ophys_session_id']}.png"
            make_plots(r, images, run_q[i], pup_q[i], out)
            print('wrote', out, flush=True)

    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f'\nwrote {args.outfile} '
          f'({os.path.getsize(args.outfile)/1e6:.1f} MB) in {time.time()-t_all:.1f}s',
          flush=True)


if __name__ == '__main__':
    main()
