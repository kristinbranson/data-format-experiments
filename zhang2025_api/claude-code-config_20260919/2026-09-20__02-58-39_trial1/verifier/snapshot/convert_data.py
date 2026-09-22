#!/usr/bin/env python3
"""
Convert the IBL brain-wide map (BWM) public dataset into the decoder-compatible
dictionary format described in the task specification.

Usage
-----
    python -u /app/convert_data.py <outpicklefile> [--full | --sample]
                                   [--show-processing] [--n-workers N]

Options
-------
    --full             process every session of the BWM freeze (default)
    --sample           process only 2 sessions (quick end-to-end test)
    --show-processing  save a multi-panel diagnostic figure per session
                       (at most 2 sessions) as processing_<eid>.png
    --n-workers N      number of worker processes (default: 24)

Processing follows `code_zhang2025/src/0_data_caching.py` and
`code_zhang2025/src/utils/ibl_data_utils.py`:

  * trials aligned to ``stimOn_times`` over the window (-0.5, +1.5) s
  * spikes binned into 20 ms non-overlapping bins  ->  T = 100
  * trial curation via ``load_trials_and_mask`` (reaction time in [0.08, 2] s,
    no NaN in the key trial events, ``feedback - goCue <= 10 s``, choice != 0)
  * continuous behaviours (wheel speed = |wheel velocity|, whisker motion
    energy from the left camera, falling back to the right camera) linearly
    interpolated onto the right edge of each spike bin, with the reference's
    interval-coverage checks

Deviations from the reference caching script, and why, are documented in
/app/CONVERSION_NOTES.md (Step 4/5). The main ones:
  * neurons are restricted to IBL well-isolated units (`label >= 1`) in grey
    matter (Beryl acronym not in {root, void}), following the data paper
  * trials whose behaviour window contains NaN are dropped instead of being
    NaN-imputed at model-fit time
  * the continuous behaviours are discretised into per-session tertiles, as
    required by the categorical-output specification
"""

import argparse
import os
import pickle
import sys
import time
import traceback
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

# reference code from the method paper
sys.path.insert(0, '/app/code/code_zhang2025/src')

from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# constants -- these mirror `params` in code_zhang2025/src/0_data_caching.py
# ---------------------------------------------------------------------------
PARAMS = {
    'interval_len': 2.0,
    'binsize': 0.02,
    'single_region': False,
    'align_time': 'stimOn_times',
    'time_window': (-0.5, 1.5),
}
N_BINS = int(np.ceil(PARAMS['interval_len'] / PARAMS['binsize']))  # 100

BWM_FREEZE = '/app/code/code_zhang2025/data/bwm_release.csv'
ONE_TABLES_DIR = '/app/data/one_cache/Brainwidemap'
ONE_BASE_URL = 'https://openalyx.internationalbrainlab.org'

#: Beryl acronyms that are not grey-matter brain regions.
NON_REGION_ACRONYMS = ('root', 'void')

#: quantiles used to discretise the continuous behaviours into 3 classes
TERTILES = (1.0 / 3.0, 2.0 / 3.0)

#: minimum number of retained neurons for a session to be kept. The data paper
#: restricts analyses to "regions that ... contained at least five well-isolated
#: neurons per session"; here the decoded population is the whole session, so the
#: same floor is applied at the session level.
MIN_NEURONS_PER_SESSION = 5

OUTPUT_NAMES = ['choice', 'prior', 'wheel_speed', 'whisker_motion_energy']
OUTPUT_VALUES = [
    ['left', 'right'],
    ['0.2 (right block)', '0.5 (unbiased)', '0.8 (left block)'],
    ['low', 'medium', 'high'],
    ['low', 'medium', 'high'],
]
INPUT_NAMES = ['time_from_stim_onset_s', 'trial_number_in_block']


# ---------------------------------------------------------------------------
# per-worker singletons
# ---------------------------------------------------------------------------
_ONE = None
_BRAIN_REGIONS = None


def get_one():
    """Return this process's ONE client (created once, offline-friendly)."""
    global _ONE
    if _ONE is None:
        from one.api import ONE
        _ONE = ONE(base_url=ONE_BASE_URL, silent=True, tables_dir=ONE_TABLES_DIR)
    return _ONE


def get_brain_regions():
    global _BRAIN_REGIONS
    if _BRAIN_REGIONS is None:
        from iblatlas.regions import BrainRegions
        _BRAIN_REGIONS = BrainRegions()
    return _BRAIN_REGIONS


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_session_spikes(one, eid, probe_rows):
    """Load and merge the spike sorting of every probe of a session.

    Mirrors ``ibl_data_utils.prepare_data`` (minus the raw-ephys sampling-rate
    query, which requires streaming the raw binary from the remote server).

    Returns
    -------
    spikes : dict with 'times', 'clusters', ... (spike-time sorted, merged)
    clusters : pandas.DataFrame, one row per cluster, re-indexed across probes
    """
    from brainbox.io.one import SpikeSortingLoader
    from utils.ibl_data_utils import merge_probes

    spikes_list, clusters_list = [], []
    for _, row in probe_rows.iterrows():
        ssl = SpikeSortingLoader(pid=row.pid, one=one, eid=eid, pname=row.probe_name)
        sp, cl, ch = ssl.load_spike_sorting()
        cl_df = SpikeSortingLoader.merge_clusters(sp, cl, ch, compute_metrics=False).to_df()
        cl_df['pid'] = row.pid
        spikes_list.append(sp)
        clusters_list.append(cl_df)
    return merge_probes(spikes_list, clusters_list)


def select_neurons(clusters):
    """Well-isolated units (IBL ``label >= 1``) located in grey matter.

    ``label`` is the fraction of the three RIGOR single-unit metrics passed
    (amplitude > 50 uV, noise cut-off < 20 uV, refractory-period violation), so
    ``label >= 1`` is exactly the data paper's "well-isolated neuron".

    Returns
    -------
    keep_idx : (n_kept,) int array of row indices into `clusters`
    beryl : (n_kept,) array of Beryl acronyms for the kept clusters
    n_good : number of well-isolated units before the grey-matter restriction
    """
    br = get_brain_regions()
    beryl_all = np.asarray(br.acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl'))
    good = clusters['label'].to_numpy() >= 1
    in_brain = ~np.isin(beryl_all, NON_REGION_ACRONYMS)
    keep = good & in_brain
    keep_idx = np.nonzero(keep)[0]
    return keep_idx, beryl_all[keep_idx], int(good.sum())


def load_trials(one, eid):
    """Trials table + reference trial mask (``load_trials_and_mask``)."""
    from brainbox.io.one import SessionLoader
    from utils.ibl_data_utils import load_trials_and_mask

    sess_loader = SessionLoader(one=one, eid=eid)
    trials, mask = load_trials_and_mask(one=one, eid=eid, max_trial_len=10.0,
                                        sess_loader=sess_loader)
    return sess_loader, trials, mask.to_numpy()


def load_continuous_behaviors(one, eid, sess_loader):
    """Wheel speed and whisker motion energy for the whole session.

    Same sources as ``ibl_data_utils.load_target_behavior``:
    wheel speed = |velocity| from ``SessionLoader.load_wheel`` (1 kHz,
    Gaussian-smoothed), whisker motion energy from the left camera ROI motion
    energy, falling back to the right camera when the left is unavailable.
    """
    beh = {}

    sess_loader.load_wheel()
    beh['wheel-speed'] = {
        'times': sess_loader.wheel['times'].to_numpy(),
        'values': np.abs(sess_loader.wheel['velocity'].to_numpy()),
        'source': 'wheel.velocity (abs)',
    }

    me = None
    for view, cam in (('left', 'leftCamera'), ('right', 'rightCamera')):
        try:
            sess_loader.load_motion_energy(views=[view])
            df = sess_loader.motion_energy[cam]
            me = {
                'times': df['times'].to_numpy(),
                'values': df['whiskerMotionEnergy'].to_numpy(),
                'source': f'{cam}.ROIMotionEnergy',
            }
            break
        except Exception:
            continue
    if me is None:
        raise RuntimeError('no whisker motion energy available (left or right camera)')
    beh['whisker-motion-energy'] = me
    return beh


# ---------------------------------------------------------------------------
# binning / interpolation
# ---------------------------------------------------------------------------
def bin_spikes(spike_times, spike_clusters, n_neurons, interval_begs, binsize=0.02,
               n_bins=N_BINS):
    """Spike counts in non-overlapping bins for every interval.

    Reproduces ``bincount2D(times, clusters, xbin=binsize, xlim=[t_beg, t_end])``
    truncated to the first ``n_bins`` columns: bin ``k`` of a trial covers
    ``[t_beg + k*binsize, t_beg + (k+1)*binsize)`` and only spikes with
    ``t_beg <= t < t_end`` are counted.

    `spike_times` must be sorted and `spike_clusters` must already be remapped
    to 0..n_neurons-1.

    Returns (n_intervals, n_neurons, n_bins) float32 array of counts.
    """
    n_intervals = len(interval_begs)
    out = np.zeros((n_intervals, n_neurons, n_bins), dtype=np.float32)
    interval_ends = interval_begs + n_bins * binsize
    i0s = np.searchsorted(spike_times, interval_begs, side='left')
    i1s = np.searchsorted(spike_times, interval_ends, side='left')
    for k in range(n_intervals):
        i0, i1 = i0s[k], i1s[k]
        if i1 <= i0:
            continue
        rel = spike_times[i0:i1] - interval_begs[k]
        bin_idx = np.floor(rel / binsize).astype(np.int64)
        np.clip(bin_idx, 0, n_bins - 1, out=bin_idx)
        flat = spike_clusters[i0:i1].astype(np.int64) * n_bins + bin_idx
        counts = np.bincount(flat, minlength=n_neurons * n_bins)
        out[k] = counts.reshape(n_neurons, n_bins)
    return out


def interpolate_behavior(times, values, interval_begs, binsize=0.02, n_bins=N_BINS):
    """Interpolate a continuous behaviour onto each trial's bin grid.

    Follows ``ibl_data_utils.get_behavior_per_interval``: samples strictly
    inside the interval are selected, then linearly interpolated (with
    extrapolation) onto ``linspace(t_beg + binsize, t_end, n_bins)``, i.e. the
    right edge of each spike bin.

    Returns
    -------
    vals : (n_intervals, n_bins) float array (rows of NaN where invalid)
    good : (n_intervals,) bool array
    reasons : list of str or None, per interval
    """
    interval_ends = interval_begs + n_bins * binsize
    idxs_beg = np.searchsorted(times, interval_begs, side='right')
    idxs_end = np.searchsorted(times, interval_ends, side='left')

    n = len(interval_begs)
    vals = np.full((n, n_bins), np.nan, dtype=np.float64)
    good = np.zeros(n, dtype=bool)
    reasons = [None] * n

    for k in range(n):
        t = times[idxs_beg[k]:idxs_end[k]]
        v = values[idxs_beg[k]:idxs_end[k]]
        if len(v) == 0:
            reasons[k] = 'target data not present'
            continue
        if np.isnan(interval_begs[k]) or np.isnan(interval_ends[k]):
            reasons[k] = 'bad interval data'
            continue
        if np.abs(interval_begs[k] - t[0]) > binsize:
            reasons[k] = 'target data starts too late'
            continue
        if np.abs(interval_ends[k] - t[-1]) > binsize:
            reasons[k] = 'target data ends too early'
            continue
        if np.any(np.isnan(v)):
            # the reference keeps these (allow_nans=True) and imputes the trial
            # mean inside its data loader; a categorical target cannot be
            # imputed without inventing a class, so the trial is dropped
            reasons[k] = 'nans in target data'
            continue
        x = np.linspace(interval_begs[k] + binsize, interval_ends[k], n_bins)
        vals[k] = interp1d(t, v, kind='linear', fill_value='extrapolate')(x)
        good[k] = True
    return vals, good, reasons


def discretize_tertiles(values):
    """Map a (n_trials, n_bins) continuous array onto {0, 1, 2} by tertile.

    Thresholds are the 33.3 / 66.7 percentiles of all values pooled over the
    session's retained trials, so each session's classes are balanced and the
    arbitrary units of motion energy do not have to be comparable across
    sessions.
    """
    flat = values.reshape(-1)
    q1, q2 = np.quantile(flat, TERTILES)
    if not (q1 < q2):
        # degenerate (heavily tied) distribution: fall back to the distinct
        # quantile values that do exist so the labels stay monotone
        edges = np.unique([q1, q2])
        labels = np.digitize(values, edges, right=False)
        return labels.astype(np.int64), (float(q1), float(q2))
    labels = np.digitize(values, [q1, q2], right=False).astype(np.int64)
    return labels, (float(q1), float(q2))


def trial_number_in_block(probability_left):
    """0-based index of each trial within its constant-``probabilityLeft`` block.

    Computed on the *unfiltered* trials table, because the block structure is a
    property of the experiment, not of which trials survive curation.
    """
    p = np.asarray(probability_left, dtype=float)
    # a change (including to/from NaN) starts a new block
    changed = np.ones(len(p), dtype=bool)
    if len(p) > 1:
        same = (p[1:] == p[:-1]) | (np.isnan(p[1:]) & np.isnan(p[:-1]))
        changed[1:] = ~same
    block_id = np.cumsum(changed) - 1
    # index within block = position - position of first trial of that block
    first_of_block = np.zeros(block_id[-1] + 1, dtype=np.int64)
    starts = np.nonzero(changed)[0]
    first_of_block[:] = starts
    return np.arange(len(p)) - first_of_block[block_id]


# ---------------------------------------------------------------------------
# per-session conversion
# ---------------------------------------------------------------------------
def process_session(task):
    """Convert one session. Returns a dict (or a dict with 'error')."""
    eid, probe_rows, subject, lab, show_processing = task
    t_start = time.time()
    timings = {}
    try:
        warnings.filterwarnings('ignore')
        one = get_one()

        # ---- neural -------------------------------------------------------
        t0 = time.time()
        spikes, clusters = load_session_spikes(one, eid, probe_rows)
        timings['load_spikes'] = time.time() - t0

        keep_idx, beryl, n_good = select_neurons(clusters)
        n_neurons = len(keep_idx)
        if n_neurons < MIN_NEURONS_PER_SESSION:
            return {'eid': eid,
                    'error': f'only {n_neurons} well-isolated grey-matter neurons '
                             f'(< {MIN_NEURONS_PER_SESSION})'}

        # remap cluster ids of the retained spikes to 0..n_neurons-1
        remap = np.full(len(clusters), -1, dtype=np.int64)
        remap[keep_idx] = np.arange(n_neurons)
        spike_cl = remap[spikes['clusters']]
        sel = spike_cl >= 0
        spike_times = np.ascontiguousarray(spikes['times'][sel])
        spike_cl = np.ascontiguousarray(spike_cl[sel])

        # ---- trials -------------------------------------------------------
        t0 = time.time()
        sess_loader, trials, trials_mask = load_trials(one, eid)
        timings['load_trials'] = time.time() - t0
        n_trials_raw = len(trials)

        stim_on_all = trials[PARAMS['align_time']].to_numpy()
        tib_all = trial_number_in_block(trials['probabilityLeft'].to_numpy())

        cand = np.nonzero(trials_mask)[0]
        if len(cand) == 0:
            return {'eid': eid, 'error': 'no trials pass the reference trial mask'}
        interval_begs = stim_on_all[cand] + PARAMS['time_window'][0]

        # ---- continuous behaviours ---------------------------------------
        t0 = time.time()
        beh = load_continuous_behaviors(one, eid, sess_loader)
        timings['load_behavior'] = time.time() - t0

        t0 = time.time()
        beh_vals, beh_good, beh_reasons = {}, {}, {}
        for name, d in beh.items():
            v, g, r = interpolate_behavior(d['times'], d['values'], interval_begs,
                                           PARAMS['binsize'], N_BINS)
            beh_vals[name], beh_good[name], beh_reasons[name] = v, g, r
        timings['bin_behavior'] = time.time() - t0

        keep_trial = np.ones(len(cand), dtype=bool)
        for name in beh_vals:
            keep_trial &= beh_good[name]
        if keep_trial.sum() < 2:
            return {'eid': eid,
                    'error': f'only {int(keep_trial.sum())} trials with valid behaviour'}

        trial_idx = cand[keep_trial]
        interval_begs = interval_begs[keep_trial]
        n_trials = len(trial_idx)

        # ---- bin spikes ---------------------------------------------------
        t0 = time.time()
        binned = bin_spikes(spike_times, spike_cl, n_neurons, interval_begs,
                            PARAMS['binsize'], N_BINS)
        timings['bin_spikes'] = time.time() - t0

        # Drop trials in which not a single spike was recorded from the whole
        # population over the 2 s window. With >= 5 simultaneously recorded
        # neurons this indicates a dropout in the recording rather than a
        # genuinely silent population, and such a trial carries no information
        # for a neural decoder.
        has_spikes = binned.any(axis=(1, 2))
        n_zero_spike = int((~has_spikes).sum())
        if n_zero_spike:
            binned = binned[has_spikes]
            trial_idx = trial_idx[has_spikes]
            interval_begs = interval_begs[has_spikes]
            keep_trial[np.nonzero(keep_trial)[0][~has_spikes]] = False
            n_trials = len(trial_idx)
            if n_trials < 2:
                return {'eid': eid, 'error': 'fewer than 2 trials with recorded spikes'}

        # ---- inputs -------------------------------------------------------
        # right edge of each spike bin, relative to stimulus onset
        bin_times = PARAMS['time_window'][0] + PARAMS['binsize'] * np.arange(1, N_BINS + 1)
        tib = tib_all[trial_idx].astype(np.float32)
        inputs = np.empty((n_trials, 2, N_BINS), dtype=np.float32)
        inputs[:, 0, :] = bin_times[None, :]
        inputs[:, 1, :] = tib[:, None]

        # ---- outputs ------------------------------------------------------
        choice = trials['choice'].to_numpy()[trial_idx]
        # choice == +1 is a LEFT report, choice == -1 a RIGHT report
        choice_lbl = ((1 - choice) / 2).astype(np.int64)

        pleft = trials['probabilityLeft'].to_numpy()[trial_idx]
        prior_lbl = np.full(n_trials, -1, dtype=np.int64)
        for val, lbl in ((0.2, 0), (0.5, 1), (0.8, 2)):
            prior_lbl[np.isclose(pleft, val)] = lbl
        if np.any(prior_lbl < 0):
            bad = np.unique(pleft[prior_lbl < 0])
            return {'eid': eid, 'error': f'unexpected probabilityLeft values {bad}'}

        ws = beh_vals['wheel-speed'][keep_trial]
        me = beh_vals['whisker-motion-energy'][keep_trial]
        ws_lbl, ws_thr = discretize_tertiles(ws)
        me_lbl, me_thr = discretize_tertiles(me)

        outputs = np.empty((n_trials, 4, N_BINS), dtype=np.int64)
        outputs[:, 0, :] = choice_lbl[:, None]
        outputs[:, 1, :] = prior_lbl[:, None]
        outputs[:, 2, :] = ws_lbl
        outputs[:, 3, :] = me_lbl

        if show_processing:
            try:
                make_processing_figure(
                    eid, trials, trial_idx, interval_begs, binned, inputs, outputs,
                    beh, ws, me, ws_thr, me_thr, bin_times, beryl, tib_all, trials_mask)
            except Exception:
                traceback.print_exc()

        result = {
            'eid': eid,
            'subject': subject,
            'lab': lab,
            'neural': [binned[k] for k in range(n_trials)],
            'input': [inputs[k] for k in range(n_trials)],
            'output': [outputs[k] for k in range(n_trials)],
            'beryl': beryl,
            'info': {
                'eid': eid,
                'subject': subject,
                'lab': lab,
                'n_probes': int(len(probe_rows)),
                'n_clusters_total': int(len(clusters)),
                'n_good_units': n_good,
                'n_neurons': int(n_neurons),
                'n_trials_raw': int(n_trials_raw),
                'n_trials_mask': int(trials_mask.sum()),
                'n_trials': int(n_trials),
                'n_zero_spike_trials_dropped': n_zero_spike,
                # row indices into the session's (unfiltered) trials table for
                # every retained trial, so the conversion can be re-derived and
                # checked against the raw ONE objects
                'trial_idx': trial_idx.astype(np.int32),
                'whisker_source': beh['whisker-motion-energy']['source'],
                'wheel_speed_tertiles': ws_thr,
                'whisker_me_tertiles': me_thr,
                'mean_firing_rate_hz': float(binned.sum() / (n_neurons * n_trials * 2.0)),
                'behavior_drop_counts': {
                    name: int((~beh_good[name]).sum()) for name in beh_good},
            },
            'timings': timings,
            'total_time': time.time() - t_start,
        }
        return result
    except Exception as e:  # noqa: BLE001 - one bad session must not kill the run
        return {'eid': eid, 'error': f'{type(e).__name__}: {e}',
                'traceback': traceback.format_exc()}


# ---------------------------------------------------------------------------
# diagnostic plots
# ---------------------------------------------------------------------------
def make_processing_figure(eid, trials, trial_idx, interval_begs, binned, inputs,
                           outputs, beh, ws, me, ws_thr, me_thr, bin_times, beryl,
                           tib_all, trials_mask):
    """Save a figure showing every processing step for one session."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_trials = len(trial_idx)
    ex = min(3, n_trials - 1)  # example trial (index into the retained trials)
    t_beg = interval_begs[ex]
    stim_on = trials['stimOn_times'].to_numpy()[trial_idx[ex]]

    fig, axes = plt.subplots(4, 2, figsize=(22, 18))

    # (0,0) raw spike raster of the example trial with bin edges
    ax = axes[0, 0]
    ax.imshow(binned[ex], aspect='auto', interpolation='nearest', origin='lower',
              extent=[bin_times[0] - 0.02, bin_times[-1], 0, binned.shape[1]],
              cmap='Greys')
    ax.axvline(0, color='r', lw=2)
    ax.set_title(f'{eid[:8]} trial {ex}: binned spike counts (n={binned.shape[1]} neurons)')
    ax.set_xlabel('time from stimulus onset (s)')
    ax.set_ylabel('neuron')

    # (0,1) population PSTH over all trials -> should show a stimulus response at t=0
    ax = axes[0, 1]
    psth = binned.mean(axis=(0, 1)) / 0.02
    ax.plot(bin_times, psth, 'k')
    ax.axvline(0, color='r', lw=2, label='stimulus onset')
    ax.set_title('population PSTH (mean firing rate over all trials/neurons)')
    ax.set_xlabel('time from stimulus onset (s)')
    ax.set_ylabel('rate (Hz)')
    ax.legend()

    # (1,0) wheel speed: raw trace vs interpolated bins for the example trial
    ax = axes[1, 0]
    d = beh['wheel-speed']
    m = (d['times'] >= t_beg - 0.1) & (d['times'] <= t_beg + 2.1)
    ax.plot(d['times'][m] - stim_on, d['values'][m], color='0.6', label='raw |velocity|')
    ax.plot(bin_times, ws[ex], 'b.-', label='interpolated to bin edges')
    for thr in ws_thr:
        ax.axhline(thr, color='g', ls='--', lw=1)
    ax.axvline(0, color='r', lw=2)
    ax.set_title('wheel speed: raw vs binned (green = session tertiles)')
    ax.set_xlabel('time from stimulus onset (s)')
    ax.legend()

    # (1,1) wheel speed discretisation
    ax = axes[1, 1]
    ax.plot(bin_times, ws[ex], 'b.-', label='wheel speed')
    ax2 = ax.twinx()
    ax2.step(bin_times, outputs[ex, 2], where='mid', color='m', label='class')
    ax2.set_ylim(-0.2, 2.2)
    ax2.set_ylabel('class (0/1/2)', color='m')
    for thr in ws_thr:
        ax.axhline(thr, color='g', ls='--', lw=1)
    ax.axvline(0, color='r', lw=2)
    ax.set_title('wheel speed discretisation check')
    ax.set_xlabel('time from stimulus onset (s)')

    # (2,0) whisker motion energy raw vs interpolated
    ax = axes[2, 0]
    d = beh['whisker-motion-energy']
    m = (d['times'] >= t_beg - 0.1) & (d['times'] <= t_beg + 2.1)
    ax.plot(d['times'][m] - stim_on, d['values'][m], color='0.6', marker='.',
            label=f"raw ({d['source']})")
    ax.plot(bin_times, me[ex], 'b.-', label='interpolated to bin edges')
    for thr in me_thr:
        ax.axhline(thr, color='g', ls='--', lw=1)
    ax.axvline(0, color='r', lw=2)
    ax.set_title('whisker motion energy: raw vs binned')
    ax.set_xlabel('time from stimulus onset (s)')
    ax.legend()

    # (2,1) whisker ME discretisation
    ax = axes[2, 1]
    ax.plot(bin_times, me[ex], 'b.-')
    ax2 = ax.twinx()
    ax2.step(bin_times, outputs[ex, 3], where='mid', color='m')
    ax2.set_ylim(-0.2, 2.2)
    ax2.set_ylabel('class (0/1/2)', color='m')
    for thr in me_thr:
        ax.axhline(thr, color='g', ls='--', lw=1)
    ax.axvline(0, color='r', lw=2)
    ax.set_title('whisker motion energy discretisation check')
    ax.set_xlabel('time from stimulus onset (s)')

    # (3,0) block structure / trial-in-block input
    ax = axes[3, 0]
    ax.plot(trials['probabilityLeft'].to_numpy(), 'k-', label='probabilityLeft')
    ax.plot(tib_all / 100.0, 'c-', lw=1, label='trial-in-block / 100')
    keep_all = np.zeros(len(trials), dtype=bool)
    keep_all[trial_idx] = True
    ax.plot(np.nonzero(keep_all)[0], np.full(keep_all.sum(), 1.05), '|', color='g',
            ms=4, label='retained trials')
    ax.plot(np.nonzero(trials_mask & ~keep_all)[0],
            np.full(int((trials_mask & ~keep_all).sum()), 1.12), '|', color='orange',
            ms=4, label='dropped (behaviour)')
    ax.set_title('block structure and trial curation')
    ax.set_xlabel('trial index (unfiltered)')
    ax.legend(fontsize=8)

    # (3,1) outputs / inputs summary over trials
    ax = axes[3, 1]
    ax.imshow(outputs[:, 2, :], aspect='auto', interpolation='nearest', cmap='viridis',
              extent=[bin_times[0] - 0.02, bin_times[-1], n_trials, 0])
    ax.axvline(0, color='r', lw=2)
    ax.set_title('wheel-speed class over all trials (rows = trials)')
    ax.set_xlabel('time from stimulus onset (s)')
    ax.set_ylabel('trial')

    fig.suptitle(f'Processing steps for session {eid}   '
                 f'({n_trials} trials, {binned.shape[1]} neurons, '
                 f'{len(np.unique(beryl))} Beryl regions)')
    fig.tight_layout()
    fig.savefig(f'processing_{eid}.png', dpi=110)
    plt.close(fig)
    print(f'  wrote processing_{eid}.png')


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('outfile', type=str)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true', default=True)
    mode.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    ap.add_argument('--n-workers', type=int, default=24)
    args = ap.parse_args()

    t_start = time.time()
    bwm = pd.read_csv(BWM_FREEZE, index_col=0)
    eids = list(dict.fromkeys(bwm.eid.tolist()))  # preserve freeze order
    if args.sample:
        eids = eids[:2]
    print(f'BWM freeze: {len(bwm)} insertions, {bwm.eid.nunique()} sessions, '
          f'{bwm.subject.nunique()} subjects')
    print(f'Processing {len(eids)} sessions with {args.n_workers} workers')

    n_show = 2 if args.show_processing else 0
    tasks = []
    for i, eid in enumerate(eids):
        rows = bwm[bwm.eid == eid]
        tasks.append((eid, rows[['pid', 'probe_name']].copy(),
                      rows.subject.iloc[0], rows.lab.iloc[0], i < n_show))

    results = []
    if args.n_workers <= 1:
        for t in tasks:
            results.append(process_session(t))
            report(results[-1], len(results), len(tasks))
    else:
        import multiprocessing as mp
        ctx = mp.get_context('fork')
        with ctx.Pool(processes=min(args.n_workers, len(tasks))) as pool:
            for res in pool.imap_unordered(process_session, tasks):
                results.append(res)
                report(res, len(results), len(tasks))

    # ---- assemble -------------------------------------------------------
    ok = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    # deterministic session order (freeze order)
    order = {eid: i for i, eid in enumerate(eids)}
    ok.sort(key=lambda r: order[r['eid']])

    print(f'\n{len(ok)} sessions converted, {len(failed)} skipped')
    for r in failed:
        print(f"  SKIPPED {r['eid']}: {r['error']}")

    subjects = sorted({r['subject'] for r in ok})
    subject_index = {s: i for i, s in enumerate(subjects)}
    regions = sorted({a for r in ok for a in np.unique(r['beryl'])})
    region_index = {a: i for i, a in enumerate(regions)}

    data = {
        'neural': [r['neural'] for r in ok],
        'input': [r['input'] for r in ok],
        'output': [r['output'] for r in ok],
        'subjects': subjects,
        'subject_idx': np.array([subject_index[r['subject']] for r in ok], dtype=np.int64),
        'brain_regions': regions,
        'brain_region_idx': [np.array([region_index[a] for a in r['beryl']], dtype=np.int64)
                             for r in ok],
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'dataset': 'IBL brain-wide map (BWM) public release',
            'source_papers': [
                'IBL et al., A brain-wide map of neural activity during complex behaviour',
                'Zhang et al. 2026 Neuron, Exploiting correlations across trials and '
                'behavioral sessions to improve neural decoding',
            ],
            'task_description': (
                'Head-fixed mice report the side of a visual grating by turning a wheel. '
                'Stimulus contrast is drawn from {0, 6.25, 12.5, 25, 100}%; after an initial '
                '90 unbiased trials the prior probability that the stimulus appears on the '
                'left alternates between 0.2 and 0.8 in uncued blocks of 20-100 trials. '
                'Decoded from 20 ms binned spike counts of well-isolated neurons: the '
                "animal's choice (left/right), the block prior probability of left "
                '(0.2/0.5/0.8), and wheel speed and whisker motion energy discretised into '
                'per-session tertiles at every 20 ms bin.'),
            'time_bin_size': PARAMS['binsize'] * 1000.0,  # ms
            'temporal_alignment_event': 'visual stimulus onset (trials.stimOn_times)',
            'off_start': PARAMS['time_window'][0],
            'off_end': PARAMS['time_window'][1],
            'n_timepoints': N_BINS,
            'neural_units': 'spike counts per 20 ms bin',
            'neuron_selection': ("IBL well-isolated units (clusters.label >= 1: amplitude "
                                 "> 50 uV, noise cut-off < 20 uV, refractory-period "
                                 "violation) located in grey matter (Beryl acronym not in "
                                 "{root, void}); probes of a session merged"),
            'trial_selection': ('brainbox load_trials_and_mask: no NaN in stimOn_times, '
                                'choice, feedback_times, probabilityLeft, '
                                'firstMovement_times, feedbackType; reaction time in '
                                '[0.08, 2.0] s; feedback_times - goCue_times <= 10 s; '
                                'choice != 0; plus full behavioural coverage of the '
                                '2 s window with no NaN'),
            'input_description': {
                'time_from_stim_onset_s': ('time (s) of the right edge of each 20 ms bin '
                                           'relative to stimulus onset, -0.48 ... 1.50'),
                'trial_number_in_block': ('0-based index of the trial within its constant-'
                                          'probabilityLeft block, counted on the '
                                          'unfiltered trials table'),
            },
            'output_description': {
                'choice': 'trials.choice: +1 (left report) -> 0, -1 (right report) -> 1',
                'prior': 'trials.probabilityLeft: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2',
                'wheel_speed': ('|wheel velocity| interpolated to bin edges, discretised '
                                'at the session tertiles'),
                'whisker_motion_energy': ('left (else right) camera whisker-pad ROI motion '
                                          'energy interpolated to bin edges, discretised '
                                          'at the session tertiles'),
            },
            'session_info': [r['info'] for r in ok],
            'skipped_sessions': [{'eid': r['eid'], 'error': r['error']} for r in failed],
        },
    }

    # ---- summary --------------------------------------------------------
    n_trials = [len(s) for s in data['neural']]
    n_neurons = [s[0].shape[0] for s in data['neural']]
    print(f"\nSessions: {len(ok)}  subjects: {len(subjects)}  "
          f"Beryl regions: {len(regions)}")
    print(f"Trials: total {sum(n_trials)}, mean/session {np.mean(n_trials):.1f} "
          f"(min {min(n_trials)}, max {max(n_trials)})")
    print(f"Neurons: total {sum(n_neurons)}, mean/session {np.mean(n_neurons):.1f} "
          f"(min {min(n_neurons)}, max {max(n_neurons)})")
    tot_bytes = sum(len(s) * s[0].nbytes for s in data['neural'])
    print(f"Neural array size: {tot_bytes / 1e9:.2f} GB")

    t0 = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Wrote {args.outfile} ({os.path.getsize(args.outfile) / 1e9:.2f} GB) '
          f'in {time.time() - t0:.1f}s')

    # timing report
    agg = defaultdict(float)
    for r in ok:
        for k, v in r['timings'].items():
            agg[k] += v
    print('\nCumulative worker time by stage (s):')
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f'  {k:16s} {v:9.1f}  ({v / max(1, len(ok)):.2f} / session)')
    print(f'Total wall time: {time.time() - t_start:.1f}s')


def report(res, i, n):
    if 'error' in res:
        print(f'[{i}/{n}] {res["eid"]}  ERROR: {res["error"]}', flush=True)
    else:
        info = res['info']
        print(f'[{i}/{n}] {res["eid"]}  {info["n_neurons"]:4d} neurons '
              f'({info["n_good_units"]} good units, {info["n_clusters_total"]} clusters), '
              f'{info["n_trials"]:4d}/{info["n_trials_raw"]:4d} trials, '
              f'{info["mean_firing_rate_hz"]:.2f} Hz, {res["total_time"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
