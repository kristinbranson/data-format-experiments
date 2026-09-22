#!/usr/bin/env python3
"""
Convert the IBL brain-wide map (BWM) public release into the decoder-ready pickle format.

Usage:
    python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

The conversion follows the reference pipeline of
  Zhang et al., 'Exploiting correlations across trials and behavioral sessions to
  improve neural decoding'  (code in /app/code/code_zhang2025)
and the curation rules of
  IBL et al., 'A brain-wide map of neural activity during complex behaviour'.

Trials are aligned to stimulus onset, window (-0.5, +1.5) s, 20 ms bins -> T = 100.

See /app/CONVERSION_NOTES.md for the full rationale of every decision.
"""
import argparse
import os
import pickle
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------------------
# Parameters -- identical to /app/code/code_zhang2025/src/0_data_caching.py `params`
# --------------------------------------------------------------------------------------
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)      # seconds relative to ALIGN_TIME
BINSIZE = 0.02                 # seconds
NBINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))   # 100

# trial mask parameters -- reference load_trials_and_mask(..., max_trial_len=10.0)
MIN_RT = 0.08
MAX_RT = 2.0
MAX_TRIAL_LEN = 10.0
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']

# neuron curation -- data paper 'well-isolated neurons' + grey matter
QC_LABEL = 1.0
NON_GREY = ('root', 'void')

BWM_RELEASE_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'

INPUT_NAMES = ['time_from_stim_on', 'trial_number_in_block']
OUTPUT_NAMES = ['choice', 'prior_prob_left', 'wheel_speed', 'whisker_motion_energy']
OUTPUT_VALUES = [
    ['left', 'right'],
    ['p_left_0.2', 'p_left_0.5', 'p_left_0.8'],
    ['low', 'medium', 'high'],
    ['low', 'medium', 'high'],
]
TRIAL_IN_BLOCK_SCALE = 100.0   # keeps the input O(1); see CONVERSION_NOTES Step 5


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def get_one():
    """ONE instance that works offline against the staged cache.

    NOTE: ONE() with no arguments reads ~/.one/.caches, which points CACHE_DIR at
    /app/data/one_cache and lets dataset queries be answered from the cached Alyx REST
    responses in .rest/.  Building ONE with an explicit tables_dir instead uses the
    frozen release tables, which do NOT list the revisioned datasets that are actually on
    disk (e.g. alf/#2025-03-03#/_ibl_trials.table.pqt) and silently returns a nearly empty
    trials table.
    """
    from one.api import ONE
    return ONE()


def build_trials_mask(trials):
    """Reproduce reference ibl_data_utils.load_trials_and_mask(max_trial_len=10.).

    Returns a boolean numpy array, True for trials to KEEP.
    """
    query = f'(firstMovement_times - stimOn_times < {MIN_RT})'
    query += f' | (firstMovement_times - stimOn_times > {MAX_RT})'
    query += f' | (feedback_times - goCue_times > {MAX_TRIAL_LEN})'
    for event in NAN_EXCLUDE:
        query += f' | {event}.isnull()'
    query += ' | (choice == 0)'
    return np.asarray(~trials.eval(query), dtype=bool).copy()


def trial_number_in_block(prob_left):
    """0-based index of each trial within its block of constant probabilityLeft."""
    n = len(prob_left)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    # NaN != NaN, so a NaN run would be split; treat NaNs as their own value.
    prev, cur = prob_left[:-1], prob_left[1:]
    is_new[1:] = ~((prev == cur) | (np.isnan(prev) & np.isnan(cur)))
    block_start = np.maximum.accumulate(np.where(is_new, np.arange(n), 0))
    return np.arange(n) - block_start


def bin_spikes_trials(spike_times, spike_clusters, n_clusters, t_beg):
    """Spike counts per (trial, cluster, bin).

    Equivalent to the reference bincount2D(times, clusters, xbin=BINSIZE,
    xlim=[t_beg, t_end]) truncated to NBINS bins, but computed for all trials at once.

    Bin i of trial k covers [t_beg[k] + i*BINSIZE, t_beg[k] + (i+1)*BINSIZE).

    Args:
        spike_times:    (n_spikes,) sorted spike times, seconds
        spike_clusters: (n_spikes,) cluster index in [0, n_clusters)
        n_clusters:     number of clusters
        t_beg:          (n_trials,) window start of each trial, seconds

    Returns:
        (n_trials, n_clusters, NBINS) float32 array of counts
    """
    n_trials = len(t_beg)
    out = np.zeros((n_trials, n_clusters, NBINS), dtype=np.float32)
    if n_trials == 0 or len(spike_times) == 0:
        return out

    t_end = t_beg + NBINS * BINSIZE
    # Trials are not necessarily time-ordered; searchsorted handles each independently.
    i0 = np.searchsorted(spike_times, t_beg, side='left')
    i1 = np.searchsorted(spike_times, t_end, side='left')

    flat = out.reshape(n_trials, n_clusters * NBINS)
    for k in range(n_trials):
        a, b = i0[k], i1[k]
        if b <= a:
            continue
        # bin index within the trial
        bi = ((spike_times[a:b] - t_beg[k]) / BINSIZE).astype(np.int64)
        np.clip(bi, 0, NBINS - 1, out=bi)
        ci = spike_clusters[a:b]
        np.add.at(flat[k], ci * NBINS + bi, 1.0)
    return out


def interp_behavior(times, values, t_beg):
    """Linear interpolation of a continuous behaviour onto the bin RIGHT EDGES.

    Matches reference get_behavior_per_interval, which evaluates at
    np.linspace(interval_beg + binsize, interval_end, n_bins).

    Returns (vals, ok) where vals is (n_trials, NBINS) and ok is (n_trials,) bool:
    a trial is not ok if the trace does not cover its window (the reference
    'target data starts too late' / 'ends too early' checks) or if any interpolated
    value is NaN.
    """
    n_trials = len(t_beg)
    grid = BINSIZE * np.arange(1, NBINS + 1)                     # (NBINS,)
    xi = t_beg[:, None] + grid[None, :]                          # (n_trials, NBINS)
    ok = np.isfinite(t_beg)
    if len(times) < 2:
        return np.zeros((n_trials, NBINS), dtype=np.float32), np.zeros(n_trials, bool)

    # coverage: the reference allows the trace to start/end within one binsize of the
    # interval bounds.
    t_end = t_beg + NBINS * BINSIZE
    ok &= (t_beg >= times[0] - BINSIZE) & (t_end <= times[-1] + BINSIZE)

    safe = np.where(np.isfinite(xi), xi, times[0])
    vals = np.interp(safe, times, values).astype(np.float32)
    ok &= np.isfinite(vals).all(axis=1)
    return vals, ok


def tertile_bins(values, mask):
    """Discretise into 3 per-session classes at the 33.3 / 66.7 percentiles.

    Thresholds are computed over every time bin of every KEPT trial of the session, so
    the three classes are (near) equally populated within a session.
    """
    pool = values[mask].ravel()
    pool = pool[np.isfinite(pool)]
    if pool.size == 0:
        return np.zeros(values.shape, dtype=np.int64), (np.nan, np.nan)
    lo, hi = np.percentile(pool, [100.0 / 3.0, 200.0 / 3.0])
    if not (hi > lo):
        # degenerate (e.g. a constant trace) -- fall back to unique-value splits
        uq = np.unique(pool)
        if uq.size >= 3:
            lo, hi = uq[len(uq) // 3], uq[2 * len(uq) // 3]
        elif uq.size == 2:
            lo = hi = uq[0]
        else:
            lo = hi = uq[0]
    cls = (values > lo).astype(np.int64) + (values > hi).astype(np.int64)
    return cls, (float(lo), float(hi))


# --------------------------------------------------------------------------------------
# Per-session conversion
# --------------------------------------------------------------------------------------
def load_session(eid, probes, show_processing=False):
    """Load, curate, align and bin one session.

    Args:
        eid:     session UUID (str)
        probes:  DataFrame rows of bwm_release.csv for this eid (pid, probe_name)

    Returns a dict with the per-session arrays, or a dict with 'skip' set.
    """
    from brainbox.io.one import SessionLoader, SpikeSortingLoader
    from iblatlas.regions import BrainRegions

    timings = {}
    t_start = time.time()
    one = get_one()
    br = BrainRegions()

    # ---------------- trials ------------------------------------------------------
    t0 = time.time()
    sl = SessionLoader(one=one, eid=eid)
    sl.load_trials()
    trials = sl.trials
    if len(trials) == 0:
        return {'eid': eid, 'skip': 'no trials'}
    trial_mask = build_trials_mask(trials)
    timings['trials'] = time.time() - t0

    align = trials[ALIGN_TIME].to_numpy(dtype=np.float64)
    t_beg = align + TIME_WINDOW[0]
    trial_mask &= np.isfinite(t_beg)

    # ---------------- spikes ------------------------------------------------------
    t0 = time.time()
    spk_times, spk_clusters, acronyms, labels = [], [], [], []
    offset = 0
    for pid, pname in probes:
        ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
        spikes, clusters, channels = ssl.load_spike_sorting()
        if spikes is None or len(spikes) == 0 or 'times' not in spikes:
            continue
        cdf = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
        # merge_probes: shift this probe's cluster ids so they are unique in the session
        spk_times.append(np.asarray(spikes['times'], dtype=np.float64))
        spk_clusters.append(np.asarray(spikes['clusters'], dtype=np.int64) + offset)
        acronyms.append(cdf['acronym'].to_numpy().astype(str))
        labels.append(cdf['label'].to_numpy(dtype=np.float64))
        offset += len(cdf)
    if offset == 0:
        return {'eid': eid, 'skip': 'no spike sorting'}

    spk_times = np.concatenate(spk_times)
    spk_clusters = np.concatenate(spk_clusters)
    acronyms = np.concatenate(acronyms)
    labels = np.concatenate(labels)
    n_units_all = len(acronyms)

    # drop spikes with NaN times (can happen at the edges of a recording)
    finite = np.isfinite(spk_times)
    spk_times, spk_clusters = spk_times[finite], spk_clusters[finite]
    order = np.argsort(spk_times, kind='stable')
    spk_times, spk_clusters = spk_times[order], spk_clusters[order]
    timings['spikes'] = time.time() - t0

    # ---------------- neuron curation --------------------------------------------
    beryl = np.asarray(br.acronym2acronym(acronyms, mapping='Beryl')).astype(str)
    n_good = int((labels >= QC_LABEL).sum())
    keep_unit = (labels >= QC_LABEL) & ~np.isin(beryl, NON_GREY)
    if keep_unit.sum() == 0:
        return {'eid': eid, 'skip': 'no well-isolated grey-matter units',
                'n_units_all': n_units_all, 'n_good': n_good}
    unit_idx = np.flatnonzero(keep_unit)
    remap = np.full(offset, -1, dtype=np.int64)
    remap[unit_idx] = np.arange(len(unit_idx))
    new_clu = remap[spk_clusters]
    sel = new_clu >= 0
    spk_times_k, spk_clusters_k = spk_times[sel], new_clu[sel]
    unit_regions = beryl[unit_idx]
    n_neurons = len(unit_idx)

    # ---------------- behaviour ---------------------------------------------------
    t0 = time.time()
    try:
        sl.load_wheel()
        wheel_t = sl.wheel['times'].to_numpy(dtype=np.float64)
        wheel_v = np.abs(sl.wheel['velocity'].to_numpy(dtype=np.float64))
    except Exception:
        return {'eid': eid, 'skip': 'no wheel'}

    whisker_t = whisker_v = None
    whisker_view = None
    for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
        try:
            sl.load_motion_energy(views=[view])
            me = sl.motion_energy[key]
            wt = me['times'].to_numpy(dtype=np.float64)
            wv = me['whiskerMotionEnergy'].to_numpy(dtype=np.float64)
            good = np.isfinite(wt) & np.isfinite(wv)
            if good.sum() > 1:
                whisker_t, whisker_v, whisker_view = wt[good], wv[good], view
                break
        except Exception:
            continue
    if whisker_t is None:
        return {'eid': eid, 'skip': 'no whisker motion energy'}
    timings['behavior'] = time.time() - t0

    gw = np.isfinite(wheel_t) & np.isfinite(wheel_v)
    wheel_t, wheel_v = wheel_t[gw], wheel_v[gw]

    # Spike-sorting coverage.  The reference applies a coverage test to the behavioural
    # traces (get_behavior_per_interval rejects an interval whose trace starts too late
    # or ends too early) but not to the spikes, so a trial lying in a recording drop-out
    # or past the end of the spike-sorted data silently becomes an all-zero matrix.
    # Two real cases in this release: session 8c2f7f4d has three trials that start after
    # its last spike, and session b182b754 has one trial inside a 2.1 s gap.  Apply the
    # same rule to the neural stream.
    spk_ok = np.zeros(len(t_beg), dtype=bool)
    if len(spk_times_k) > 1:
        t_end_all = t_beg + NBINS * BINSIZE
        inside = np.isfinite(t_beg) & (t_beg >= spk_times_k[0] - BINSIZE) & \
                 (t_end_all <= spk_times_k[-1] + BINSIZE)
        # a window must not fall inside a gap longer than the window itself
        i0 = np.searchsorted(spk_times_k, np.where(inside, t_beg, spk_times_k[0]), 'left')
        i1 = np.searchsorted(spk_times_k, np.where(inside, t_end_all, spk_times_k[0]), 'left')
        spk_ok = inside & (i1 > i0)
    trial_mask &= spk_ok

    ws_vals, ws_ok = interp_behavior(wheel_t, wheel_v, t_beg)
    wm_vals, wm_ok = interp_behavior(whisker_t, whisker_v, t_beg)
    # reference align_spike_behavior: a trial survives only if it is good in the trials
    # mask AND in every behaviour mask
    keep_trial = trial_mask & ws_ok & wm_ok
    if keep_trial.sum() < 2:
        return {'eid': eid, 'skip': f'only {int(keep_trial.sum())} usable trials'}

    # ---------------- discretise continuous outputs -------------------------------
    ws_cls, ws_thr = tertile_bins(ws_vals, keep_trial)
    wm_cls, wm_thr = tertile_bins(wm_vals, keep_trial)

    # ---------------- per-trial variables -----------------------------------------
    choice = trials['choice'].to_numpy(dtype=np.float64)
    # choice == +1 is a LEFT report, choice == -1 a RIGHT report (verified against
    # contrastLeft/contrastRight and feedbackType).  Required coding: left = 0, right = 1.
    choice_cls = (choice < 0).astype(np.int64)

    p_left = trials['probabilityLeft'].to_numpy(dtype=np.float64)
    prior_cls = np.full(len(p_left), -1, dtype=np.int64)
    prior_cls[np.isclose(p_left, 0.2)] = 0
    prior_cls[np.isclose(p_left, 0.5)] = 1
    prior_cls[np.isclose(p_left, 0.8)] = 2
    keep_trial = keep_trial & (prior_cls >= 0)
    if keep_trial.sum() < 2:
        return {'eid': eid, 'skip': 'no trials with a valid probabilityLeft'}

    tib = trial_number_in_block(p_left).astype(np.float32) / TRIAL_IN_BLOCK_SCALE

    # ---------------- bin spikes (kept trials only) --------------------------------
    t0 = time.time()
    kt = np.flatnonzero(keep_trial)
    binned = bin_spikes_trials(spk_times_k, spk_clusters_k, n_neurons, t_beg[kt])
    timings['binning'] = time.time() - t0

    # ---------------- assemble -----------------------------------------------------
    time_axis = (TIME_WINDOW[0] + BINSIZE * np.arange(1, NBINS + 1)).astype(np.float32)

    neural, inputs, outputs = [], [], []
    for j, k in enumerate(kt):
        neural.append(binned[j])
        inp = np.empty((2, NBINS), dtype=np.float32)
        inp[0] = time_axis
        inp[1] = tib[k]
        inputs.append(inp)
        out = np.empty((4, NBINS), dtype=np.int64)
        out[0] = choice_cls[k]
        out[1] = prior_cls[k]
        out[2] = ws_cls[k]
        out[3] = wm_cls[k]
        outputs.append(out)

    res = {
        'eid': eid,
        'skip': None,
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'regions': unit_regions,
        'n_units_all': n_units_all,
        'n_good': n_good,
        'n_neurons': n_neurons,
        'n_trials_raw': int(len(trials)),
        'n_trials_mask': int(trial_mask.sum()),
        'n_trials_kept': int(len(kt)),
        'kept_trial_idx': kt.astype(np.int32),
        'whisker_view': whisker_view,
        'wheel_thresholds': ws_thr,
        'whisker_thresholds': wm_thr,
        'timings': timings,
        'total_time': time.time() - t_start,
    }
    if show_processing:
        res['_plot'] = {
            'trials': trials, 'trial_mask': trial_mask, 'keep_trial': keep_trial,
            'kt': kt, 't_beg': t_beg, 'time_axis': time_axis,
            'wheel_t': wheel_t, 'wheel_v': wheel_v,
            'whisker_t': whisker_t, 'whisker_v': whisker_v,
            'ws_vals': ws_vals, 'wm_vals': wm_vals,
            'ws_cls': ws_cls, 'wm_cls': wm_cls,
            'ws_thr': ws_thr, 'wm_thr': wm_thr,
            'binned': binned, 'spk_times_k': spk_times_k, 'spk_clusters_k': spk_clusters_k,
            'regions': unit_regions, 'p_left': p_left, 'tib': tib,
            'choice_cls': choice_cls, 'prior_cls': prior_cls,
        }
    return res


# --------------------------------------------------------------------------------------
# Diagnostics plot -- one figure per session showing every processing step
# --------------------------------------------------------------------------------------
def make_processing_plot(res, outfile):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    P = res['_plot']
    trials = P['trials']
    kt = P['kt']
    time_axis = P['time_axis']
    tsel = kt[:4] if len(kt) >= 4 else kt
    jsel = list(range(len(tsel)))

    fig = plt.figure(figsize=(22, 26))
    gs = fig.add_gridspec(7, 4, hspace=0.55, wspace=0.28)

    # ---- row 0: trial curation -------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    rt = (trials['firstMovement_times'] - trials['stimOn_times']).to_numpy()
    ax.hist(rt[np.isfinite(rt)], bins=np.linspace(-0.2, 3, 80), color='0.6')
    ax.axvline(MIN_RT, color='r'); ax.axvline(MAX_RT, color='r')
    ax.set_title('reaction time (red = 0.08/2.0 s cutoffs)')
    ax.set_xlabel('firstMovement - stimOn (s)')

    ax = fig.add_subplot(gs[0, 1])
    ax.step(np.arange(len(trials)), P['trial_mask'].astype(int), where='mid', label='trials mask')
    ax.step(np.arange(len(trials)), P['keep_trial'].astype(int) * 0.9, where='mid',
            label='final keep')
    ax.set_ylim(-0.1, 1.3); ax.legend(fontsize=7)
    ax.set_title(f"kept {res['n_trials_kept']} / {res['n_trials_raw']} trials")
    ax.set_xlabel('trial index')

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(P['p_left'], '.-', ms=2, lw=0.5)
    ax.set_title('probabilityLeft (block structure)'); ax.set_xlabel('trial index')

    ax = fig.add_subplot(gs[0, 3])
    ax.plot(P['tib'] * TRIAL_IN_BLOCK_SCALE, '.-', ms=2, lw=0.5)
    ax.set_title('trial number in block (raw, before /100)')
    ax.set_xlabel('trial index')

    # ---- row 1: raster + binned spikes for one trial, alignment check ----------
    k0 = kt[0]
    beg = P['t_beg'][k0]
    ax = fig.add_subplot(gs[1, :2])
    m = (P['spk_times_k'] >= beg) & (P['spk_times_k'] < beg + NBINS * BINSIZE)
    ax.plot(P['spk_times_k'][m] - beg + TIME_WINDOW[0], P['spk_clusters_k'][m], '|k', ms=2)
    ax.axvline(0, color='r', lw=2)
    ax.set_xlim(TIME_WINDOW)
    ax.set_title(f'raw spike raster, trial {k0} (red = stimulus onset)')
    ax.set_xlabel('time from stimOn (s)'); ax.set_ylabel('neuron')

    ax = fig.add_subplot(gs[1, 2:])
    ax.imshow(P['binned'][0], aspect='auto', interpolation='nearest', cmap='Greys',
              extent=[time_axis[0] - BINSIZE, time_axis[-1], P['binned'].shape[1], 0])
    ax.axvline(0, color='r', lw=2)
    ax.set_title('binned spike counts for the SAME trial (must match the raster)')
    ax.set_xlabel('time from stimOn (s)'); ax.set_ylabel('neuron')

    # ---- row 2: PSTH + firing-rate sanity --------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    psth = P['binned'].mean(axis=(0, 1)) / BINSIZE
    ax.plot(time_axis, psth)
    ax.axvline(0, color='r')
    ax.set_title('population PSTH (Hz)'); ax.set_xlabel('time from stimOn (s)')

    ax = fig.add_subplot(gs[2, 1])
    fr = P['binned'].mean(axis=(0, 2)) / BINSIZE
    ax.hist(fr, bins=40, color='0.6')
    ax.set_title(f'per-neuron mean rate (Hz), median {np.median(fr):.1f}')

    ax = fig.add_subplot(gs[2, 2])
    u, c = np.unique(P['regions'], return_counts=True)
    o = np.argsort(-c)[:20]
    ax.barh(np.arange(len(o)), c[o]); ax.set_yticks(np.arange(len(o)))
    ax.set_yticklabels(u[o], fontsize=6); ax.invert_yaxis()
    ax.set_title(f'neurons per Beryl region (n={res["n_neurons"]})')

    ax = fig.add_subplot(gs[2, 3])
    ax.plot(time_axis, np.zeros_like(time_axis) + 0.5, 'k.')
    ax.plot(time_axis, time_axis, 'b.-', ms=3)
    ax.axvline(0, color='r'); ax.axhline(0, color='r')
    ax.set_title('input[0] = time_from_stim_on (bin right edges)')
    ax.set_xlabel('time (s)'); ax.set_ylabel('input value (s)')

    # ---- row 3: wheel speed alignment -------------------------------------------
    for i, (j, k) in enumerate(zip(jsel, tsel)):
        ax = fig.add_subplot(gs[3, i])
        b = P['t_beg'][k]
        m = (P['wheel_t'] >= b - 0.1) & (P['wheel_t'] <= b + NBINS * BINSIZE + 0.1)
        ax.plot(P['wheel_t'][m] - b + TIME_WINDOW[0], P['wheel_v'][m], '-', color='0.7',
                label='raw |velocity|')
        ax.plot(time_axis, P['ws_vals'][k], 'o-', ms=3, color='C0', label='binned')
        ax.axhline(P['ws_thr'][0], color='g', ls='--', lw=1)
        ax.axhline(P['ws_thr'][1], color='m', ls='--', lw=1)
        ax.axvline(0, color='r')
        ax2 = ax.twinx()
        ax2.step(time_axis, P['ws_cls'][k], where='mid', color='C3', alpha=0.6)
        ax2.set_ylim(-0.2, 2.2); ax2.set_yticks([0, 1, 2])
        ax.set_title(f'wheel speed, trial {k}', fontsize=9)
        if i == 0:
            ax.legend(fontsize=6); ax.set_ylabel('|velocity| (rad/s)')
        ax.set_xlabel('time from stimOn (s)')

    # ---- row 4: whisker motion energy alignment ---------------------------------
    for i, (j, k) in enumerate(zip(jsel, tsel)):
        ax = fig.add_subplot(gs[4, i])
        b = P['t_beg'][k]
        m = (P['whisker_t'] >= b - 0.1) & (P['whisker_t'] <= b + NBINS * BINSIZE + 0.1)
        ax.plot(P['whisker_t'][m] - b + TIME_WINDOW[0], P['whisker_v'][m], '-', color='0.7',
                label='raw ME')
        ax.plot(time_axis, P['wm_vals'][k], 'o-', ms=3, color='C1', label='binned')
        ax.axhline(P['wm_thr'][0], color='g', ls='--', lw=1)
        ax.axhline(P['wm_thr'][1], color='m', ls='--', lw=1)
        ax.axvline(0, color='r')
        ax2 = ax.twinx()
        ax2.step(time_axis, P['wm_cls'][k], where='mid', color='C3', alpha=0.6)
        ax2.set_ylim(-0.2, 2.2); ax2.set_yticks([0, 1, 2])
        ax.set_title(f'whisker ME ({res["whisker_view"]}), trial {k}', fontsize=9)
        if i == 0:
            ax.legend(fontsize=6); ax.set_ylabel('motion energy')
        ax.set_xlabel('time from stimOn (s)')

    # ---- row 5: discretisation check --------------------------------------------
    ax = fig.add_subplot(gs[5, 0])
    pool = P['ws_vals'][P['keep_trial']].ravel()
    ax.hist(pool, bins=np.percentile(pool, np.linspace(0, 100, 60)), color='0.6')
    ax.axvline(P['ws_thr'][0], color='g'); ax.axvline(P['ws_thr'][1], color='m')
    ax.set_xscale('symlog', linthresh=1e-3)
    ax.set_title('wheel speed distribution + tertiles')

    ax = fig.add_subplot(gs[5, 1])
    cls = P['ws_cls'][P['keep_trial']].ravel()
    ax.bar([0, 1, 2], [np.mean(cls == i) for i in range(3)])
    ax.set_title('wheel-speed class fractions'); ax.set_ylim(0, 0.6)
    ax.axhline(1 / 3, color='r', ls='--')

    ax = fig.add_subplot(gs[5, 2])
    pool = P['wm_vals'][P['keep_trial']].ravel()
    ax.hist(pool, bins=60, color='0.6')
    ax.axvline(P['wm_thr'][0], color='g'); ax.axvline(P['wm_thr'][1], color='m')
    ax.set_title('whisker ME distribution + tertiles')

    ax = fig.add_subplot(gs[5, 3])
    cls = P['wm_cls'][P['keep_trial']].ravel()
    ax.bar([0, 1, 2], [np.mean(cls == i) for i in range(3)])
    ax.set_title('whisker-ME class fractions'); ax.set_ylim(0, 0.6)
    ax.axhline(1 / 3, color='r', ls='--')

    # ---- row 6: per-trial outputs ------------------------------------------------
    ax = fig.add_subplot(gs[6, 0])
    ax.plot(P['choice_cls'][kt], '.', ms=3)
    ax.set_yticks([0, 1]); ax.set_yticklabels(['left(0)', 'right(1)'])
    fr_ = np.mean(P['choice_cls'][kt])
    ax.set_title(f'output[0] choice, frac right = {fr_:.2f}')
    ax.set_xlabel('kept trial')

    ax = fig.add_subplot(gs[6, 1])
    ax.plot(P['prior_cls'][kt], '.', ms=3)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(['0.2', '0.5', '0.8'])
    ax.set_title('output[1] prior_prob_left'); ax.set_xlabel('kept trial')

    ax = fig.add_subplot(gs[6, 2])
    ax.imshow(P['ws_cls'][kt], aspect='auto', interpolation='nearest', cmap='viridis',
              extent=[time_axis[0], time_axis[-1], len(kt), 0])
    ax.axvline(0, color='r')
    ax.set_title('output[2] wheel speed class'); ax.set_xlabel('time from stimOn (s)')

    ax = fig.add_subplot(gs[6, 3])
    ax.imshow(P['wm_cls'][kt], aspect='auto', interpolation='nearest', cmap='viridis',
              extent=[time_axis[0], time_axis[-1], len(kt), 0])
    ax.axvline(0, color='r')
    ax.set_title('output[3] whisker ME class'); ax.set_xlabel('time from stimOn (s)')

    fig.suptitle(f"session {res['eid']}  |  {res['n_neurons']} neurons, "
                 f"{res['n_trials_kept']} trials, T={NBINS}, bin={BINSIZE*1000:.0f} ms, "
                 f"aligned to {ALIGN_TIME} in {TIME_WINDOW}", fontsize=13)
    fig.savefig(outfile, dpi=90, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {outfile}')


# --------------------------------------------------------------------------------------
# Parallel driver
# --------------------------------------------------------------------------------------
def _worker(task):
    eid, probes, show = task
    try:
        res = load_session(eid, probes, show_processing=show)
    except Exception as exc:
        return {'eid': eid, 'skip': f'EXCEPTION {type(exc).__name__}: {exc}',
                'traceback': traceback.format_exc()}
    if show and res.get('skip') is None:
        try:
            make_processing_plot(res, f'/app/processing_{eid}.png')
        except Exception:
            traceback.print_exc()
    res.pop('_plot', None)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile', type=str, help='output pickle path')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true', default=True,
                   help='process all sessions (default)')
    g.add_argument('--sample', action='store_true',
                   help='process only 2 sessions, for testing')
    ap.add_argument('--show-processing', action='store_true',
                    help='save a per-session diagnostics figure for up to 2 sessions')
    ap.add_argument('--n-workers', type=int, default=24)
    args = ap.parse_args()

    t_all = time.time()

    bwm = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
    print(f'bwm_release.csv: {len(bwm)} insertions, {bwm.eid.nunique()} sessions, '
          f'{bwm.subject.nunique()} subjects, {bwm.lab.nunique()} labs')

    # optional data-limited subset (absent on the full dataset)
    subset_file = '/app/data/DATALIMIT_SUBSET.csv'
    if os.path.exists(subset_file):
        sub = pd.read_csv(subset_file)
        col = 'eid' if 'eid' in sub.columns else sub.columns[0]
        bwm = bwm[bwm.eid.isin(sub[col].astype(str))]
        print(f'DATALIMIT_SUBSET.csv found: restricted to {bwm.eid.nunique()} sessions')

    eids = list(dict.fromkeys(bwm.eid.tolist()))   # preserve file order, unique
    if args.sample:
        eids = eids[:2]
    n_show = 2 if args.show_processing else 0
    print(f'Processing {len(eids)} sessions with {args.n_workers} workers '
          f'(show-processing for the first {n_show})')

    by_eid = {e: list(g[['pid', 'probe_name']].itertuples(index=False, name=None))
              for e, g in bwm.groupby('eid')}
    subj_of = dict(zip(bwm.eid, bwm.subject))
    lab_of = dict(zip(bwm.eid, bwm.lab))

    tasks = [(e, by_eid[e], i < n_show) for i, e in enumerate(eids)]

    results = {}
    done = 0
    t_loop = time.time()
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futs = {ex.submit(_worker, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            results[r['eid']] = r
            done += 1
            el = time.time() - t_loop
            if r.get('skip'):
                print(f"[{done}/{len(tasks)}] SKIP {r['eid']}: {r['skip']}", flush=True)
            else:
                print(f"[{done}/{len(tasks)}] {r['eid']} "
                      f"neurons={r['n_neurons']} trials={r['n_trials_kept']}"
                      f"/{r['n_trials_raw']} t={r['total_time']:.1f}s "
                      f"| elapsed {el/60:.1f} min, eta {el/done*(len(tasks)-done)/60:.1f} min",
                      flush=True)

    # ------------- assemble the final dictionary in a deterministic order --------
    kept = [e for e in eids if results[e].get('skip') is None]
    skipped = [(e, results[e]['skip']) for e in eids if results[e].get('skip') is not None]

    # ---------------- session-level neuron-count curation ------------------------
    # The data paper requires "at least five well-isolated neurons per session" before a
    # recording enters its final analyses.  It states that per *region*, because its
    # decoding analyses are run region by region.  This decoder instead pools every
    # neuron of a session into one population, so applying the threshold per region
    # would discard well-isolated neurons that carry real signal (on this dataset it
    # removes ~40% of them) for no statistical benefit.  The threshold is therefore
    # applied at the level that matters here -- the session population -- which removes
    # the degenerate recordings (one session has a single neuron, whose silent trials
    # produce all-zero neural matrices) while keeping every neuron of every session
    # that survives.
    MIN_NEURONS_PER_SESSION = 5
    for e in list(kept):
        if results[e]['n_neurons'] < MIN_NEURONS_PER_SESSION:
            results[e]['skip'] = (f"only {results[e]['n_neurons']} well-isolated "
                                  f"grey-matter neurons "
                                  f"(< {MIN_NEURONS_PER_SESSION})")
    kept = [e for e in kept if results[e].get('skip') is None]
    skipped = [(e, results[e]['skip']) for e in eids
               if results[e].get('skip') is not None]
    print(f'session neuron-count curation: {len(kept)} sessions with '
          f'>= {MIN_NEURONS_PER_SESSION} well-isolated grey-matter neurons')

    all_regions = sorted({r for e in kept for r in results[e]['regions']})
    region_index = {r: i for i, r in enumerate(all_regions)}

    subjects = sorted({subj_of[e] for e in kept})
    subject_index = {s: i for i, s in enumerate(subjects)}

    data = {
        'neural': [], 'input': [], 'output': [],
        'subjects': subjects,
        'subject_idx': np.array([subject_index[subj_of[e]] for e in kept], dtype=np.int64),
        'brain_regions': all_regions,
        'brain_region_idx': [],
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {},
    }
    session_info = []
    for e in kept:
        r = results[e]
        data['neural'].append(r['neural'])
        data['input'].append(r['input'])
        data['output'].append(r['output'])
        data['brain_region_idx'].append(
            np.array([region_index[x] for x in r['regions']], dtype=np.int64))
        session_info.append({
            'eid': e, 'subject': subj_of[e], 'lab': lab_of[e],
            'n_neurons': r['n_neurons'], 'n_trials': r['n_trials_kept'],
            'n_trials_raw': r['n_trials_raw'], 'n_trials_after_mask': r['n_trials_mask'],
            'n_units_all': r['n_units_all'], 'n_well_isolated': r['n_good'],
            'whisker_camera': r['whisker_view'],
            'kept_trial_idx': r['kept_trial_idx'],
            'wheel_speed_tertiles': r['wheel_thresholds'],
            'whisker_me_tertiles': r['whisker_thresholds'],
            'regions': sorted(set(r['regions'].tolist())),
        })

    n_trials = sum(len(s) for s in data['neural'])
    n_neurons = sum(len(b) for b in data['brain_region_idx'])

    data['metadata'] = {
        'task_description': (
            'IBL decision-making task (brain-wide map). A visual grating of one of five '
            'contrasts appears on the left or right; the mouse turns a wheel to bring it '
            'to the centre. After an initial 90 unbiased trials the stimulus side is '
            'drawn from blocks of 20-100 trials with P(left) = 0.2 or 0.8. Decoder inputs '
            'are the time since stimulus onset and the trial index within the current '
            'block. Decoder outputs are the mouse choice (left/right), the block prior '
            'P(left) (0.2/0.5/0.8), and wheel speed and whisker motion energy each '
            'discretised into three per-session tertile bins.'),
        'time_bin_size': BINSIZE * 1000.0,            # ms
        'temporal_alignment_event': 'stimulus onset (trials.stimOn_times)',
        'off_start': float(TIME_WINDOW[0]),
        'off_end': float(TIME_WINDOW[1]),
        'n_timepoints': NBINS,
        'bin_time_axis_s': (TIME_WINDOW[0] + BINSIZE * np.arange(1, NBINS + 1)).tolist(),
        'bin_time_convention': (
            'element i of a trial covers [stimOn - 0.5 + 0.02*i, stimOn - 0.5 + 0.02*(i+1)); '
            'behavioural traces and input[0] are evaluated at the bin right edge, matching '
            'the reference get_behavior_per_interval'),
        'neural_units': 'spike counts per 20 ms bin (not normalised)',
        'dataset': 'IBL brain-wide map public release (2023), loaded with the ONE API',
        'source_papers': [
            'IBL et al., A brain-wide map of neural activity during complex behaviour',
            'Zhang et al., Exploiting correlations across trials and behavioral sessions '
            'to improve neural decoding'],
        'neuron_curation': (
            'well-isolated neurons only: clusters.label >= 1 (all three RIGOR single-unit '
            'criteria passed: median amplitude > 50 uV, noise cut-off < 20 uV, refractory '
            'period violation) AND Beryl region not root/void (grey matter only). Probes '
            'of a session are merged into one population.'),
        'trial_curation': (
            'reference load_trials_and_mask(min_rt=0.08, max_rt=2.0, max_trial_len=10.0, '
            'exclude_nochoice=True): no NaN in stimOn_times, choice, feedback_times, '
            'probabilityLeft, firstMovement_times, feedbackType; 0.08 s <= '
            'firstMovement_times - stimOn_times <= 2.0 s; feedback_times - goCue_times '
            '<= 10 s; choice != 0. Trials whose window is not covered by the wheel or '
            'whisker trace are additionally dropped (reference align_spike_behavior). '
            'Trials whose window is not covered by the spike sorting (recording gaps, or '
            'behaviour continuing past the last spike) are dropped as well. '
            'The 90 unbiased (P(left) = 0.5) trials are KEPT because they form class 1 of '
            'the prior output.'),
        'output_discretisation': (
            'wheel_speed = |wheel velocity| and whisker_motion_energy are binned into '
            'three classes at the 33.3rd and 66.7th percentiles computed over all kept '
            'time bins of that session, so the classes are equally populated and '
            'comparable across sessions of different absolute scale.'),
        'choice_coding': ('trials.choice == +1 is a leftward report -> class 0 (left); '
                          'trials.choice == -1 -> class 1 (right)'),
        'input_scaling': (f'trial_number_in_block is divided by {TRIAL_IN_BLOCK_SCALE:g} '
                          'so that it is O(1) alongside the neural PCs'),
        'session_curation': (
            'a session is kept only if it has at least 5 well-isolated grey-matter '
            'neurons, at least 2 usable trials, a wheel trace and a whisker '
            'motion-energy trace. The data paper states its 5-neuron threshold per '
            'region because its analyses are per region; this decoder pools all '
            'neurons of a session, so the threshold is applied to the session '
            'population and every neuron of a surviving session is kept.'),
        'n_sessions': len(kept),
        'n_subjects': len(subjects),
        'n_trials_total': n_trials,
        'n_neurons_total': n_neurons,
        'n_brain_regions': len(all_regions),
        'skipped_sessions': skipped,
        'session_info': session_info,
    }

    # ------------- report ---------------------------------------------------------
    print('\n' + '=' * 78)
    print(f'sessions kept      : {len(kept)} / {len(eids)}')
    print(f'sessions skipped   : {len(skipped)}')
    for e, why in skipped:
        print(f'    {e}: {why}')
    print(f'subjects           : {len(subjects)}')
    print(f'brain regions      : {len(all_regions)}')
    print(f'neurons (total)    : {n_neurons}')
    print(f'trials (total)     : {n_trials}')
    if kept:
        nn = np.array([len(b) for b in data['brain_region_idx']])
        nt = np.array([len(s) for s in data['neural']])
        print(f'neurons/session    : mean {nn.mean():.1f} median {np.median(nn):.0f} '
              f'min {nn.min()} max {nn.max()}')
        print(f'trials/session     : mean {nt.mean():.1f} median {np.median(nt):.0f} '
              f'min {nt.min()} max {nt.max()}')
        allout = np.concatenate([o.ravel() for s in data['output'] for o in s]).reshape(-1)
        for i, name in enumerate(OUTPUT_NAMES):
            vals = np.concatenate([o[i] for s in data['output'] for o in s])
            fr = [float(np.mean(vals == c)) for c in range(len(OUTPUT_VALUES[i]))]
            print(f'  output {i} {name:24s} fractions {np.round(fr, 4).tolist()}')
        allin = np.concatenate([o for s in data['input'] for o in s], axis=1)
        for i, name in enumerate(INPUT_NAMES):
            print(f'  input  {i} {name:24s} range [{allin[i].min():.4f}, '
                  f'{allin[i].max():.4f}]')
        tot_units = sum(si['n_units_all'] for si in session_info)
        tot_good = sum(si['n_well_isolated'] for si in session_info)
        print(f'units seen (kept sessions): {tot_units}, well-isolated: {tot_good}')
    print('=' * 78)

    t0 = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'wrote {args.outfile} '
          f'({os.path.getsize(args.outfile)/1e9:.2f} GB) in {time.time()-t0:.1f}s')
    print(f'TOTAL TIME {(time.time()-t_all)/60:.2f} min')


if __name__ == '__main__':
    main()
