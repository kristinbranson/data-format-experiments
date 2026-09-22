"""
Convert the IBL Brain-Wide Map (BWM) dataset into the decoder pickle format expected by
/app/train_decoder.py.

Processing follows the reference implementation of Zhang et al. 2026
(/app/code/code_zhang2025/src/0_data_caching.py + src/utils/ibl_data_utils.py) and the
inclusion criteria of the IBL brain-wide-map data paper.  See /app/CONVERSION_NOTES.md for a
full description of every decision.

Trial geometry (reference `params` in 0_data_caching.py):
    align_time  = 'stimOn_times'
    time_window = (-0.5, +1.5) s
    binsize     = 0.02 s              ->  T = 100 bins per trial

Decoder inputs   : time from stimulus onset (time-varying), trial number in block (per trial)
Decoder outputs  : choice (L/R), prior probability of left (0.2/0.5/0.8),
                   wheel speed (3 bins), whisker motion energy (3 bins)

Usage:
    python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]
"""

import argparse
import os
import pickle
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------------------
# Constants -- these mirror `params` / `beh_names` of the reference caching script
# --------------------------------------------------------------------------------------
ONE_CACHE_DIR = '/app/data/one_cache'
BWM_FREEZE_FILE = '/app/code/code_zhang2025/data/bwm_release.csv'
DATALIMIT_FILE = '/app/data/DATALIMIT_SUBSET.csv'

ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))  # 100

# trial curation (reference `load_trials_and_mask` defaults + prepare_data's max_trial_len)
MIN_RT, MAX_RT, MAX_TRIAL_LEN = 0.08, 2.0, 10.0
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']

# neuron curation (data paper: RIGOR single-unit metrics -> clusters.label >= 1)
QC_LABEL = 1.0
NON_GREY_MATTER = ('root', 'void')

N_OUT_BINS = 3          # wheel speed / whisker motion energy are discretised into 3 classes
MIN_TRIALS_PER_SESSION = 2
# data paper: analyses are restricted to grey-matter regions "[that] contained at least five
# well-isolated neurons per session".  Applied here at the session level, because the decoding
# unit here is the whole grey-matter population of a session.
MIN_NEURONS_PER_SESSION = 5

INPUT_NAMES = ['time_from_stim_onset', 'trial_number_in_block']
OUTPUT_NAMES = ['choice', 'prior_prob_left', 'wheel_speed', 'whisker_motion_energy']
OUTPUT_VALUES = [
    ['left', 'right'],
    ['0.2', '0.5', '0.8'],
    ['low', 'medium', 'high'],
    ['low', 'medium', 'high'],
]


# --------------------------------------------------------------------------------------
# ONE
# --------------------------------------------------------------------------------------
def get_one():
    """Build the ONE client.

    NOTE: no ``password=`` -- supplying one forces a network re-authentication which fails in
    this offline container.  With the staged auth token + `.rest` response cache, ONE answers
    every query from disk.
    """
    from one.api import ONE
    return ONE(base_url='https://openalyx.internationalbrainlab.org',
               silent=True, cache_dir=ONE_CACHE_DIR)


# --------------------------------------------------------------------------------------
# Loading  (mirrors ibl_data_utils.load_spiking_data / merge_probes / load_trials_and_mask)
# --------------------------------------------------------------------------------------
def load_spiking_data(one, pid, eid, pname, qc=QC_LABEL):
    """Load one probe's spike sorting and keep only clusters passing the QC threshold.

    Equivalent to ``ibl_data_utils.load_spiking_data(one, pid, qc=1)`` minus the
    ``raw_electrophysiology`` call (which needs to stream raw AP data over the network and is
    only used to report the sampling frequency).

    Returns
    -------
    times, clusters : np.ndarray
        Spike times (sorted) and the *row index* of each spike's cluster within
        ``clusters_df``.
    clusters_df : pd.DataFrame
        One row per retained cluster, re-indexed 0..n-1.
    """
    from brainbox.io.one import SpikeSortingLoader
    from iblutil.numerical import ismember

    loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    spikes, clusters, channels = loader.load_spike_sorting()
    clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()

    if qc is None:
        sel_clusters = clusters_labeled.reset_index(drop=True)
        sel_times, sel_cl = spikes['times'], spikes['clusters'].astype(np.int64)
    else:
        iok = clusters_labeled['label'] >= qc
        sel_clusters = clusters_labeled[iok]
        spike_idx, ib = ismember(spikes['clusters'], sel_clusters.index)
        sel_clusters = sel_clusters.reset_index(drop=True)
        sel_times = spikes['times'][spike_idx]
        sel_cl = sel_clusters.index.to_numpy()[ib].astype(np.int64)

    return sel_times, sel_cl, sel_clusters, loader.collection


def load_trials_and_mask(one, eid, sess_loader=None):
    """Trials table + keep-mask, identical to ``ibl_data_utils.load_trials_and_mask`` with
    ``min_rt=0.08, max_rt=2.0, max_trial_len=10.0, nan_exclude='default',
    exclude_nochoice=True`` (the settings used by ``prepare_data``).

    Re-implemented only because the reference builds ``SessionLoader(one, eid)`` positionally,
    which the installed brainbox (a keyword-only dataclass) rejects.
    """
    from brainbox.io.one import SessionLoader
    if sess_loader is None:
        sess_loader = SessionLoader(one=one, eid=eid)
    if sess_loader.trials.empty:
        sess_loader.load_trials()
    trials = sess_loader.trials

    query = f'(firstMovement_times - stimOn_times < {MIN_RT})'
    query += f' | (firstMovement_times - stimOn_times > {MAX_RT})'
    query += f' | (feedback_times - goCue_times > {MAX_TRIAL_LEN})'
    for event in NAN_EXCLUDE:
        query += f' | {event}.isnull()'
    query += ' | (choice == 0)'
    mask = ~trials.eval(query)
    return trials, mask.to_numpy()


def load_behaviour_traces(one, eid, sess_loader=None):
    """Load the two continuous behaviours, as ``ibl_data_utils.load_target_behavior`` does.

    wheel-speed             : |velocity| from SessionLoader.load_wheel()
    whisker-motion-energy   : left camera, falling back to the right camera (reference
                              ``bin_behaviors``).
    """
    from brainbox.io.one import SessionLoader
    if sess_loader is None:
        sess_loader = SessionLoader(one=one, eid=eid)

    traces = {}

    sess_loader.load_wheel()
    traces['wheel_speed'] = (sess_loader.wheel['times'].to_numpy(),
                             np.abs(sess_loader.wheel['velocity'].to_numpy()))

    cam_used = None
    for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
        try:
            sess_loader.load_motion_energy(views=[view])
            me = sess_loader.motion_energy[key]
            traces['whisker_motion_energy'] = (me['times'].to_numpy(),
                                               me['whiskerMotionEnergy'].to_numpy())
            cam_used = view
            break
        except Exception:
            continue
    if cam_used is None:
        raise RuntimeError('no whisker motion energy available (neither camera)')

    return traces, cam_used


# --------------------------------------------------------------------------------------
# Binning / interpolation
# --------------------------------------------------------------------------------------
def bin_spikes(times, clusters, n_clusters, interval_begs, binsize=BINSIZE, n_bins=N_BINS):
    """Vectorised equivalent of ``ibl_data_utils.get_spike_data_per_interval``.

    For every interval ``[t_beg, t_beg + n_bins*binsize)`` count the spikes of each cluster in
    bin ``floor((t - t_beg) / binsize)``; spikes are selected with ``t_beg <= t < t_end``,
    exactly as ``bincount2D(..., xlim=[t_beg, t_end])`` followed by ``[:, :n_bins]`` does.

    Parameters
    ----------
    times : (n_spikes,) float array, **sorted**
    clusters : (n_spikes,) int array of cluster row indices in [0, n_clusters)
    interval_begs : (n_trials,) float array

    Returns
    -------
    (n_trials, n_clusters, n_bins) float32 array of spike counts
    """
    n_trials = len(interval_begs)
    out = np.zeros((n_trials, n_clusters, n_bins), dtype=np.float32)
    if n_trials == 0 or times.size == 0:
        return out

    interval_ends = interval_begs + n_bins * binsize
    i0 = np.searchsorted(times, interval_begs, side='left')
    i1 = np.searchsorted(times, interval_ends, side='left')

    # Concatenate the per-trial spike slices into one flat index array, then bincount once.
    counts = (i1 - i0).astype(np.int64)
    counts[counts < 0] = 0
    total = int(counts.sum())
    if total == 0:
        return out

    flat_idx = np.concatenate([np.arange(a, b) for a, b in zip(i0, i1) if b > a])
    trial_of_spike = np.repeat(np.arange(n_trials), counts)
    t_rel = times[flat_idx] - interval_begs[trial_of_spike]
    bin_of_spike = np.floor(t_rel / binsize).astype(np.int64)
    # guard against float round-off putting a spike in bin n_bins
    np.clip(bin_of_spike, 0, n_bins - 1, out=bin_of_spike)

    lin = ((trial_of_spike * n_clusters) + clusters[flat_idx]) * n_bins + bin_of_spike
    out += np.bincount(lin, minlength=n_trials * n_clusters * n_bins).reshape(
        n_trials, n_clusters, n_bins).astype(np.float32)
    return out


def bin_behaviour(target_times, target_vals, interval_begs,
                  binsize=BINSIZE, n_bins=N_BINS):
    """Port of ``ibl_data_utils.get_behavior_per_interval`` (single-process).

    The behaviour value for spike-bin *i* is the trace linearly interpolated at the bin's
    **right edge**, i.e. at ``np.linspace(t_beg + binsize, t_end, n_bins)``.

    Returns
    -------
    vals : (n_trials, n_bins) float array (rows of NaN where the trial is unusable)
    good : (n_trials,) bool array
    """
    from scipy.interpolate import interp1d

    n_trials = len(interval_begs)
    vals = np.full((n_trials, n_bins), np.nan)
    good = np.zeros(n_trials, dtype=bool)
    if n_trials == 0:
        return vals, good

    interval_ends = interval_begs + n_bins * binsize
    if np.all(np.isnan(interval_begs)):
        return vals, good

    idxs_beg = np.searchsorted(target_times, interval_begs, side='right')
    idxs_end = np.searchsorted(target_times, interval_ends, side='left')

    for k in range(n_trials):
        if np.isnan(interval_begs[k]) or np.isnan(interval_ends[k]):
            continue
        tt = target_times[idxs_beg[k]:idxs_end[k]]
        tv = target_vals[idxs_beg[k]:idxs_end[k]]
        if len(tv) == 0:
            continue                                    # 'target data not present'
        if np.abs(interval_begs[k] - tt[0]) > binsize:
            continue                                    # 'target data starts too late'
        if np.abs(interval_ends[k] - tt[-1]) > binsize:
            continue                                    # 'target data ends too early'
        x_interp = np.linspace(interval_begs[k] + binsize, interval_ends[k], n_bins)
        y = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x_interp)
        if not np.all(np.isfinite(y)):
            continue      # NaNs in the trace -> unusable for the decoder (see notes)
        vals[k] = y
        good[k] = True

    return vals, good


# --------------------------------------------------------------------------------------
# Derived task variables
# --------------------------------------------------------------------------------------
def trial_number_in_block(probability_left):
    """0-based index of each trial within its block of constant ``probabilityLeft``.

    Computed on the *full* (uncurated) trials table so that it reflects the animal's true
    position in the block.
    """
    p = np.asarray(probability_left, dtype=float)
    out = np.zeros(len(p), dtype=np.float64)
    counter = 0
    for i in range(1, len(p)):
        same = (p[i] == p[i - 1]) or (np.isnan(p[i]) and np.isnan(p[i - 1]))
        counter = counter + 1 if same else 0
        out[i] = counter
    return out


def discretize_terciles(values):
    """Split ``values`` into 3 classes at the 33.3 % / 66.7 % quantiles.

    Class 0: v <= e1, class 1: e1 < v <= e2, class 2: v > e2.  ``<=`` on the lower edge keeps a
    mass point at the bottom of the range (e.g. a wheel at rest) inside the "low" class.
    Degenerate edges (e1 == e2) are repaired by splitting the remaining mass.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    e1, e2 = np.quantile(v, [1.0 / 3.0, 2.0 / 3.0])
    if e1 == e2:
        above = v[v > e1]
        e2 = np.quantile(above, 0.5) if above.size else e1
        if e2 == e1:                      # still degenerate: trace is (almost) constant
            above = v[v > e1]
            e2 = above.min() if above.size else e1
    return float(e1), float(e2)


def apply_terciles(values, e1, e2):
    return ((values > e1).astype(np.int64) + (values > e2).astype(np.int64))


# --------------------------------------------------------------------------------------
# Per-session conversion
# --------------------------------------------------------------------------------------
def convert_session(session_info, show_processing=False, plot_dir='/app'):
    """Convert a single session.  Returns a dict, or raises on failure."""
    t_start = time.time()
    eid = session_info['eid']
    one = get_one()
    timings = {}

    from brainbox.io.one import SessionLoader
    from iblatlas.regions import BrainRegions

    # ---------------- trials ----------------
    t0 = time.time()
    sess_loader = SessionLoader(one=one, eid=eid)
    trials_df, trials_mask = load_trials_and_mask(one, eid, sess_loader=sess_loader)
    timings['trials'] = time.time() - t0
    n_trials_raw = len(trials_df)

    # trial number in block, computed on the full table (before curation)
    tnb_all = trial_number_in_block(trials_df['probabilityLeft'].to_numpy())

    align_times = trials_df[ALIGN_TIME].to_numpy(dtype=float)
    interval_begs = align_times + TIME_WINDOW[0]

    # ---------------- continuous behaviour ----------------
    t0 = time.time()
    traces, cam_used = load_behaviour_traces(one, eid, sess_loader=sess_loader)
    timings['behaviour_load'] = time.time() - t0

    t0 = time.time()
    beh_vals, beh_good = {}, {}
    for name, (tt, tv) in traces.items():
        vals, good = bin_behaviour(tt, tv, interval_begs)
        beh_vals[name] = vals
        beh_good[name] = good
    timings['behaviour_bin'] = time.time() - t0

    # ---------------- combined trial mask (reference `align_spike_behavior`) ----------------
    keep = np.asarray(trials_mask, dtype=bool).copy()
    keep &= np.isfinite(interval_begs)
    for name in beh_good:
        keep &= beh_good[name]
    keep_idx = np.flatnonzero(keep)
    if len(keep_idx) < MIN_TRIALS_PER_SESSION:
        raise RuntimeError(f'only {len(keep_idx)} usable trials')

    # ---------------- spikes ----------------
    t0 = time.time()
    br = BrainRegions()
    neural_trials, region_acronyms, sorters = [], [], set()
    for pid, pname in zip(session_info['pids'], session_info['probe_names']):
        times, clu, clusters_df, collection = load_spiking_data(one, pid, eid, pname)
        sorters.add(collection)
        if len(clusters_df) == 0:
            continue
        beryl = np.asarray(br.acronym2acronym(clusters_df['acronym'].to_numpy(),
                                              mapping='Beryl'), dtype=object)
        # grey matter only (data paper) AND -- as in the reference `bin_spiking_data` --
        # only clusters that actually emit spikes in this session
        has_spikes = np.zeros(len(clusters_df), dtype=bool)
        if clu.size:
            has_spikes[np.unique(clu)] = True
        sel = (~np.isin(beryl, NON_GREY_MATTER)) & has_spikes
        if not sel.any():
            continue
        # remap kept clusters to rows 0..n-1
        row_of = np.full(len(clusters_df), -1, dtype=np.int64)
        row_of[np.flatnonzero(sel)] = np.arange(int(sel.sum()))
        spike_sel = row_of[clu] >= 0
        binned = bin_spikes(times[spike_sel], row_of[clu[spike_sel]],
                            int(sel.sum()), interval_begs[keep_idx])
        neural_trials.append(binned)
        region_acronyms.append(beryl[sel])
    if not neural_trials:
        raise RuntimeError('no neurons passed curation')
    neural = np.concatenate(neural_trials, axis=1)          # (n_trials, n_neurons, T)
    regions = np.concatenate(region_acronyms)
    if neural.shape[1] < MIN_NEURONS_PER_SESSION:
        raise RuntimeError(f'only {neural.shape[1]} well-isolated grey-matter neurons '
                           f'(< {MIN_NEURONS_PER_SESSION})')
    timings['spikes'] = time.time() - t0

    # ---------------- inputs ----------------
    n_keep = len(keep_idx)
    bin_centres = (TIME_WINDOW[0] + BINSIZE * (np.arange(N_BINS) + 0.5)).astype(np.float32)
    inputs = np.empty((n_keep, 2, N_BINS), dtype=np.float32)
    inputs[:, 0, :] = bin_centres[None, :]
    inputs[:, 1, :] = tnb_all[keep_idx].astype(np.float32)[:, None]

    # ---------------- outputs ----------------
    choice = trials_df['choice'].to_numpy()[keep_idx]        # +1 = left, -1 = right
    choice_cls = (choice < 0).astype(np.int64)               # left -> 0, right -> 1
    pleft = trials_df['probabilityLeft'].to_numpy()[keep_idx]
    pleft_cls = np.full(len(pleft), -1, dtype=np.int64)
    for val, cls in ((0.2, 0), (0.5, 1), (0.8, 2)):
        pleft_cls[np.isclose(pleft, val)] = cls
    if np.any(pleft_cls < 0):
        bad = np.unique(pleft[pleft_cls < 0])
        raise RuntimeError(f'unexpected probabilityLeft values {bad}')

    ws = beh_vals['wheel_speed'][keep_idx]
    wme = beh_vals['whisker_motion_energy'][keep_idx]
    ws_edges = discretize_terciles(ws)
    wme_edges = discretize_terciles(wme)
    ws_cls = apply_terciles(ws, *ws_edges)
    wme_cls = apply_terciles(wme, *wme_edges)

    outputs = np.empty((n_keep, 4, N_BINS), dtype=np.int64)
    outputs[:, 0, :] = choice_cls[:, None]
    outputs[:, 1, :] = pleft_cls[:, None]
    outputs[:, 2, :] = ws_cls
    outputs[:, 3, :] = wme_cls

    # ---------------- sanity checks ----------------
    assert neural.shape[0] == n_keep == inputs.shape[0] == outputs.shape[0]
    assert neural.shape[2] == N_BINS
    assert len(regions) == neural.shape[1]
    assert np.all(np.isfinite(neural)) and np.all(np.isfinite(inputs))
    assert outputs.min() >= 0

    if show_processing:
        _plot_processing(plot_dir, eid, trials_df, keep_idx, interval_begs, traces,
                         beh_vals, ws_edges, wme_edges, neural, inputs, outputs,
                         regions, cam_used)

    result = {
        'eid': eid,
        'subject': session_info['subject'],
        'lab': session_info['lab'],
        'date': str(session_info['date']),
        'neural': [neural[k] for k in range(n_keep)],
        'input': [inputs[k] for k in range(n_keep)],
        'output': [outputs[k] for k in range(n_keep)],
        'regions': regions,
        'n_trials_raw': int(n_trials_raw),
        'n_trials_kept': int(n_keep),
        'n_neurons': int(neural.shape[1]),
        'camera': cam_used,
        'spike_sorter': sorted(sorters),
        'wheel_speed_edges': ws_edges,
        'whisker_me_edges': wme_edges,
        'n_probes': len(session_info['pids']),
        'timings': timings,
        'runtime': time.time() - t_start,
    }
    return result


# --------------------------------------------------------------------------------------
# Diagnostic plots
# --------------------------------------------------------------------------------------
def _plot_processing(plot_dir, eid, trials_df, keep_idx, interval_begs, traces, beh_vals,
                     ws_edges, wme_edges, neural, inputs, outputs, regions, cam_used):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(20, 18))
    t_edges = TIME_WINDOW[0] + BINSIZE * np.arange(N_BINS + 1)
    t_right = TIME_WINDOW[0] + BINSIZE * (np.arange(N_BINS) + 1)
    k = 0                                   # first retained trial
    tk = keep_idx[k]
    t_beg = interval_begs[tk]
    stim_on = trials_df[ALIGN_TIME].to_numpy()[tk]

    # --- 1. raw wheel speed vs interpolated/binned ---
    ax = axes[0, 0]
    tt, tv = traces['wheel_speed']
    m = (tt >= t_beg - 0.1) & (tt <= t_beg + 2.1)
    ax.plot(tt[m] - stim_on, tv[m], 'k-', lw=0.8, label='raw |wheel velocity|')
    ax.plot(t_right, beh_vals['wheel_speed'][tk], 'r.-', ms=4, lw=0.8,
            label='interpolated at bin right edge')
    ax.axhline(ws_edges[0], color='b', ls=':', label='tercile edges')
    ax.axhline(ws_edges[1], color='b', ls=':')
    ax.axvline(0, color='g', label='stimOn')
    ax.set_title(f'{eid[:8]} trial {tk}: wheel speed alignment')
    ax.set_xlabel('time from stimOn (s)'); ax.set_ylabel('rad/s'); ax.legend(fontsize=7)

    # --- 2. raw whisker ME vs interpolated/binned ---
    ax = axes[0, 1]
    tt, tv = traces['whisker_motion_energy']
    m = (tt >= t_beg - 0.1) & (tt <= t_beg + 2.1)
    ax.plot(tt[m] - stim_on, tv[m], 'k.-', lw=0.8, ms=3, label=f'raw ME ({cam_used} cam)')
    ax.plot(t_right, beh_vals['whisker_motion_energy'][tk], 'r.-', ms=4, lw=0.8,
            label='interpolated at bin right edge')
    ax.axhline(wme_edges[0], color='b', ls=':'); ax.axhline(wme_edges[1], color='b', ls=':')
    ax.axvline(0, color='g')
    ax.set_title('whisker motion energy alignment')
    ax.set_xlabel('time from stimOn (s)'); ax.legend(fontsize=7)

    # --- 3. discretisation check ---
    ax = axes[1, 0]
    ax.plot(t_right, beh_vals['wheel_speed'][tk], 'k-', label='wheel speed')
    ax.axhline(ws_edges[0], color='b', ls=':'); ax.axhline(ws_edges[1], color='b', ls=':')
    ax2 = ax.twinx()
    ax2.step(t_right, outputs[k, 2], 'r-', where='mid', label='class')
    ax2.set_ylim(-0.2, 2.2); ax2.set_ylabel('class', color='r')
    ax.set_title('wheel-speed discretisation (3 terciles)'); ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.plot(t_right, beh_vals['whisker_motion_energy'][tk], 'k-', label='whisker ME')
    ax.axhline(wme_edges[0], color='b', ls=':'); ax.axhline(wme_edges[1], color='b', ls=':')
    ax2 = ax.twinx()
    ax2.step(t_right, outputs[k, 3], 'r-', where='mid')
    ax2.set_ylim(-0.2, 2.2); ax2.set_ylabel('class', color='r')
    ax.set_title('whisker-ME discretisation (3 terciles)'); ax.legend(fontsize=7)

    # --- 4. neural raster / binned counts ---
    ax = axes[2, 0]
    nshow = min(60, neural.shape[1])
    ax.imshow(neural[k, :nshow], aspect='auto', origin='lower', interpolation='nearest',
              extent=[t_edges[0], t_edges[-1], 0, nshow], cmap='Greys')
    ax.axvline(0, color='g')
    ax.set_title(f'binned spike counts, trial {tk} (first {nshow} neurons)')
    ax.set_xlabel('time from stimOn (s)'); ax.set_ylabel('neuron')

    ax = axes[2, 1]
    psth = neural.mean(axis=(0, 1)) / BINSIZE
    ax.plot(t_edges[:-1] + BINSIZE / 2, psth, 'k-')
    ax.axvline(0, color='g', label='stimOn')
    med_rt = np.nanmedian((trials_df['firstMovement_times'].to_numpy()
                           - trials_df[ALIGN_TIME].to_numpy())[keep_idx])
    ax.axvline(med_rt, color='m', ls='--', label='median first movement')
    ax.set_title('population PSTH (all retained trials)')
    ax.set_xlabel('time from stimOn (s)'); ax.set_ylabel('spikes/s/neuron')
    ax.legend(fontsize=7)

    # --- 5. inputs ---
    ax = axes[3, 0]
    ax.plot(inputs[k, 0], 'k.-', label='input0: time from stimOn')
    ax.plot(inputs[:, 1, 0][:min(200, len(keep_idx))], 'r.-', ms=3,
            label='input1: trial number in block (first 200 trials)')
    ax.set_title('decoder inputs'); ax.legend(fontsize=7)

    # --- 6. per-trial outputs vs raw trials table ---
    ax = axes[3, 1]
    n = min(200, len(keep_idx))
    ax.plot(trials_df['probabilityLeft'].to_numpy()[keep_idx][:n], 'k-',
            label='probabilityLeft (raw)')
    ax.plot(outputs[:n, 1, 0] * 0.3 + 0.2, 'r--', label='prior class *0.3+0.2')
    ax.plot(0.5 + 0.35 * np.where(outputs[:n, 0, 0] == 0, 1, -1), 'b.', ms=3,
            label='choice class (up=left)')
    ax.plot(0.5 + 0.3 * trials_df['choice'].to_numpy()[keep_idx][:n], 'g+', ms=4,
            label='raw choice (+1=left)')
    ax.set_title('per-trial outputs vs raw trials table'); ax.legend(fontsize=7)

    fig.suptitle(f'Processing checks: {eid}  ({neural.shape[1]} neurons, '
                 f'{len(keep_idx)} trials, regions={len(np.unique(regions))})')
    fig.tight_layout()
    out = os.path.join(plot_dir, f'processing_{eid}.png')
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f'  wrote {out}', flush=True)


# --------------------------------------------------------------------------------------
# Worker entry point
# --------------------------------------------------------------------------------------
def _worker(args):
    session_info, show_processing = args
    try:
        return convert_session(session_info, show_processing=show_processing)
    except Exception as e:                                   # noqa: BLE001
        return {'eid': session_info['eid'], 'error': f'{type(e).__name__}: {e}',
                'traceback': traceback.format_exc()}


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def build_session_list():
    bwm = pd.read_csv(BWM_FREEZE_FILE, index_col=0)
    if os.path.exists(DATALIMIT_FILE):
        sub = pd.read_csv(DATALIMIT_FILE)
        col = 'eid' if 'eid' in sub.columns else sub.columns[0]
        bwm = bwm[bwm.eid.isin(sub[col].astype(str))]
        print(f'DATALIMIT_SUBSET.csv present: restricting to {bwm.eid.nunique()} sessions')
    sessions = []
    for eid, g in bwm.groupby('eid', sort=False):
        g = g.sort_values('probe_name')
        sessions.append({
            'eid': str(eid),
            'pids': [str(p) for p in g.pid],
            'probe_names': list(g.probe_name),
            'subject': str(g.subject.iloc[0]),
            'lab': str(g.lab.iloc[0]),
            'date': str(g.date.iloc[0]),
        })
    sessions.sort(key=lambda s: (s['lab'], s['subject'], s['date']))
    return sessions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile', type=str)
    ap.add_argument('--full', action='store_true', default=True)
    ap.add_argument('--sample', action='store_true',
                    help='process only 2 sessions (for testing)')
    ap.add_argument('--show-processing', action='store_true',
                    help='write processing_<eid>.png for up to 2 sessions')
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()

    t_start = time.time()
    sessions = build_session_list()
    if args.sample:
        sessions = sessions[:2]
    print(f'Converting {len(sessions)} sessions with {args.workers} workers', flush=True)

    tasks = [(s, args.show_processing and i < 2) for i, s in enumerate(sessions)]

    results = []
    failures = []
    if args.workers <= 1:
        for i, task in enumerate(tasks):
            r = _worker(task)
            _report(r, i, len(tasks), t_start)
            (failures if 'error' in r else results).append(r)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, r in enumerate(ex.map(_worker, tasks)):
                _report(r, i, len(tasks), t_start)
                (failures if 'error' in r else results).append(r)

    print(f'\n{len(results)} sessions converted, {len(failures)} failed')
    for f in failures:
        print(f"  FAILED {f['eid']}: {f['error']}")

    # ---------------- assemble ----------------
    subjects = sorted({r['subject'] for r in results})
    subject_index = {s: i for i, s in enumerate(subjects)}
    brain_regions = sorted({reg for r in results for reg in np.unique(r['regions'])})
    region_index = {r: i for i, r in enumerate(brain_regions)}

    data = {
        'neural': [r['neural'] for r in results],
        'input': [r['input'] for r in results],
        'output': [r['output'] for r in results],
        'subjects': subjects,
        'subject_idx': np.array([subject_index[r['subject']] for r in results], dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': [np.array([region_index[x] for x in r['regions']], dtype=np.int64)
                             for r in results],
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'dataset': 'IBL Brain-Wide Map (public release, 459 sessions / 699 insertions)',
            'source_papers': [
                'International Brain Laboratory et al., "A brain-wide map of neural activity '
                'during complex behaviour", Nature 645 (2025)',
                'Zhang et al., "Exploiting correlations across trials and behavioral sessions '
                'to improve neural decoding", Neuron 114 (2026)'],
            'task_description': (
                'IBL decision-making task: a Gabor stimulus of one of 5 contrasts (0, 6, 12.5, '
                '25, 100%) appears at +/-35 deg azimuth and the mouse turns a wheel to bring it '
                'to the centre. After 90 unbiased trials (p(left)=0.5) the stimulus side '
                'follows 20:80 or 80:20 blocks of 20-100 trials. Decoded from population spike '
                'counts: the mouse choice (left/right), the block prior probability that the '
                'stimulus is on the left (0.2/0.5/0.8), and two time-varying behaviours '
                '(wheel speed and whisker-pad motion energy), each discretised into 3 '
                'per-session terciles (low/medium/high).'),
            'time_bin_size': BINSIZE * 1000.0,
            'temporal_alignment_event': 'visual stimulus onset (trials.stimOn_times)',
            'off_start': TIME_WINDOW[0],
            'off_end': TIME_WINDOW[1],
            'n_timepoints': N_BINS,
            'neural_units': 'spike counts per 20 ms bin',
            'input_descriptions': {
                'time_from_stim_onset': 'seconds relative to stimulus onset, bin centre '
                                        '(-0.49 ... +1.49), time-varying',
                'trial_number_in_block': '0-based index of the trial within its block of '
                                         'constant probabilityLeft, computed on the full '
                                         'uncurated trials table; constant within a trial',
            },
            'output_descriptions': {
                'choice': 'wheel-turn choice; trials.choice == +1 -> left (0), == -1 -> '
                          'right (1); constant within a trial',
                'prior_prob_left': 'trials.probabilityLeft; 0.2 -> 0, 0.5 -> 1, 0.8 -> 2; '
                                   'constant within a trial',
                'wheel_speed': '|wheel velocity| (rad/s) interpolated to the right edge of '
                               'each 20 ms bin, discretised at the per-session 33.3/66.7 % '
                               'quantiles',
                'whisker_motion_energy': 'whisker-pad motion energy (left camera, right camera '
                                         'as fallback) interpolated to the right edge of each '
                                         '20 ms bin, discretised at the per-session '
                                         '33.3/66.7 % quantiles',
            },
            'neuron_curation': (
                'well-isolated units only (clusters.label >= 1, i.e. passing all three RIGOR '
                'single-unit metrics: amplitude > 50 uV, noise cut-off < 20 uV, refractory '
                'period violation), restricted to grey matter (Beryl acronym not in '
                '{root, void}); units with no spike in the session are dropped, as in the '
                'reference bin_spiking_data. Probes of a session are merged.'),
            'trial_curation': (
                'reference load_trials_and_mask: no NaN in stimOn_times, choice, '
                'feedback_times, probabilityLeft, firstMovement_times, feedbackType; '
                '0.08 s <= firstMovement_times - stimOn_times <= 2 s; feedback_times - '
                'goCue_times <= 10 s; choice != 0; plus trials whose wheel or whisker trace '
                'does not cover the whole -0.5...+1.5 s window (reference '
                'get_behavior_per_interval / align_spike_behavior).'),
            'brain_region_mapping': 'Allen CCF acronym mapped to the Beryl atlas '
                                    '(iblatlas.regions.BrainRegions.acronym2acronym)',
            'session_info': [
                {k: r[k] for k in ('eid', 'subject', 'lab', 'date', 'n_trials_raw',
                                   'n_trials_kept', 'n_neurons', 'n_probes', 'camera',
                                   'wheel_speed_edges', 'whisker_me_edges')}
                for r in results],
            'failed_sessions': [{'eid': f['eid'], 'error': f['error']} for f in failures],
        },
    }

    # ---------------- summary ----------------
    ntr = np.array([r['n_trials_kept'] for r in results])
    nne = np.array([r['n_neurons'] for r in results])
    print(f'\nSessions: {len(results)}   subjects: {len(subjects)}   '
          f'brain regions: {len(brain_regions)}')
    print(f'Trials kept: total {ntr.sum()}, mean {ntr.mean():.1f}, '
          f'min {ntr.min()}, max {ntr.max()}')
    print(f'Neurons: total {nne.sum()}, mean {nne.mean():.1f}, '
          f'min {nne.min()}, max {nne.max()}')
    for i, name in enumerate(OUTPUT_NAMES):
        vals = np.concatenate([np.asarray(r['output'])[:, i, :].ravel() for r in results])
        frac = np.bincount(vals, minlength=len(OUTPUT_VALUES[i])) / len(vals)
        print(f'  {name}: ' + ', '.join(f'{v}={f:.3f}'
                                        for v, f in zip(OUTPUT_VALUES[i], frac)))

    t0 = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Wrote {args.outfile} '
          f'({os.path.getsize(args.outfile) / 1e9:.2f} GB) in {time.time() - t0:.1f}s')
    print(f'Total run time: {time.time() - t_start:.1f}s')


def _report(r, i, n, t_start):
    el = time.time() - t_start
    if 'error' in r:
        print(f'[{i + 1}/{n}] {r["eid"]} FAILED: {r["error"]}  ({el:.0f}s elapsed)', flush=True)
    else:
        tm = ', '.join(f'{k}={v:.1f}s' for k, v in r['timings'].items())
        print(f'[{i + 1}/{n}] {r["eid"]} {r["subject"]}: '
              f'{r["n_trials_kept"]}/{r["n_trials_raw"]} trials, {r["n_neurons"]} neurons, '
              f'{r["runtime"]:.1f}s ({tm})  [{el:.0f}s elapsed, '
              f'eta {el / (i + 1) * (n - i - 1):.0f}s]', flush=True)


if __name__ == '__main__':
    main()
