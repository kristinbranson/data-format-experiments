#!/usr/bin/env python3
"""Convert the IBL Brain-wide Map release into the decoder dataset format.

Usage
-----
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

The processing follows ``code/code_zhang2025/src/0_data_caching.py`` (Zhang et al.,
Neuron 2026) and the inclusion criteria of the IBL Brain-wide Map paper:

  * trials aligned on ``stimOn_times`` over the window (-0.5, +1.5) s,
  * spike counts in 100 non-overlapping 20 ms bins,
  * all probes of a session merged into one population,
  * clusters restricted to well-isolated units (``clusters.label >= 1``) lying in
    grey matter (Beryl acronym not ``root``/``void``),
  * trials curated with ``load_trials_and_mask(..., max_trial_len=10.0)``,
  * wheel speed = |wheel velocity| and whisker motion energy (left camera, right
    camera as fallback) resampled onto the bin right edges with the reference's
    coverage checks.

See CONVERSION_NOTES.md for the full rationale.
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

# The reference repository is imported rather than copied where practical, so that
# trial curation is literally the reference implementation.
REF_SRC = '/app/code/code_zhang2025/src'
if REF_SRC not in sys.path:
    sys.path.insert(0, REF_SRC)

BWM_FREEZE = '/app/code/code_zhang2025/data/bwm_release.csv'
CACHE_DIR = '/app/data/one_cache'
ALYX_URL = 'https://openalyx.internationalbrainlab.org'

# ---------------------------------------------------------------------------
# Trial / binning parameters -- identical to the reference `params` dict in
# code_zhang2025/src/0_data_caching.py
# ---------------------------------------------------------------------------
PARAMS = {
    'interval_len': 2,
    'binsize': 0.02,
    'single_region': False,
    'align_time': 'stimOn_times',
    'time_window': (-0.5, 1.5),
}
BINSIZE = PARAMS['binsize']
WIN = PARAMS['time_window']
NBINS = int(np.ceil((WIN[1] - WIN[0]) / BINSIZE))          # 100
BIN_LEFT_EDGES = WIN[0] + BINSIZE * np.arange(NBINS)        # -0.50 ... 1.48
BIN_RIGHT_EDGES = BIN_LEFT_EDGES + BINSIZE                  # -0.48 ... 1.50

MAX_TRIAL_LEN = 10.0            # reference: load_trials_and_mask(..., max_trial_len=10.0)
QC_LABEL = 1.0                  # BWM "well-isolated neuron" == clusters.label >= 1
NON_GREY = ('root', 'void')     # Beryl acronyms that are not grey matter
N_DISCRETE_BINS = 3             # wheel speed / whisker motion energy -> 3 classes
MIN_TRIALS = 2                  # decoder needs >= 2 trials per session
MIN_NEURONS = 5                 # BWM: >= 5 well-isolated neurons per session
PRIOR_MAP = {0.2: 0, 0.5: 1, 0.8: 2}

INPUT_NAMES = ['time_from_stim_on', 'trial_num_in_block']
OUTPUT_NAMES = ['choice', 'prior_prob_left', 'wheel_speed', 'whisker_motion_energy']
OUTPUT_VALUES = [
    ['left', 'right'],
    ['p(left)=0.2', 'p(left)=0.5', 'p(left)=0.8'],
    ['low', 'medium', 'high'],
    ['low', 'medium', 'high'],
]

_ONE = None
_BR = None


def get_one():
    """Process-local ONE handle (served entirely from the local cache)."""
    global _ONE
    if _ONE is None:
        from one.api import ONE
        _ONE = ONE(base_url=ALYX_URL, silent=True)
    return _ONE


def get_brain_regions():
    global _BR
    if _BR is None:
        from iblatlas.regions import BrainRegions
        _BR = BrainRegions()
    return _BR


# ---------------------------------------------------------------------------
# Neural data
# ---------------------------------------------------------------------------
def load_session_spikes(eid, pids, probe_names):
    """Load and merge the spike sorting of every probe of one session.

    Mirrors ``ibl_data_utils.prepare_data``: ``SpikeSortingLoader.load_spike_sorting``
    followed by ``merge_clusters`` per probe, then ``merge_probes``.

    Returns
    -------
    spikes : dict with 'times' (sorted) and 'clusters'
    clusters : pandas.DataFrame, one row per cluster, re-indexed 0..n-1
    rec_span : (t_first, t_last) -- the interval during which *every* probe of the
        session was producing spikes.  Trials outside it have no neural data and are
        dropped (some sessions' behaviour outlasts the ephys recording).
    """
    from brainbox.io.one import SpikeSortingLoader
    from utils.ibl_data_utils import merge_probes

    one = get_one()
    spikes_list, clusters_list, spans = [], [], []
    for pid, pname in zip(pids, probe_names):
        ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
        sp, cl, ch = ssl.load_spike_sorting()
        if len(sp) == 0 or 'times' not in sp or len(sp['times']) == 0:
            continue
        cld = SpikeSortingLoader.merge_clusters(sp, cl, ch).to_df()
        cld['pid'] = pid
        spikes_list.append(sp)
        clusters_list.append(cld)
        spans.append((float(np.nanmin(sp['times'])), float(np.nanmax(sp['times']))))
    if not spikes_list:
        raise RuntimeError('no spike sorting available')
    if len(spikes_list) == 1:
        spikes, clusters = spikes_list[0], clusters_list[0]
        clusters = clusters.reset_index(drop=True)
    else:
        spikes, clusters = merge_probes(spikes_list, clusters_list)
    rec_span = (max(s[0] for s in spans), min(s[1] for s in spans))
    return spikes, clusters, rec_span


def select_units(clusters):
    """Well-isolated (``label >= 1``) grey-matter units, plus their Beryl acronyms.

    ``clusters.label`` is the fraction of the three RIGOR single-unit metrics
    (amplitude > 50 uV, noise cut-off < 20 uV, refractory-period violation) that the
    cluster passes, so ``label >= 1`` is exactly the BWM paper's "well-isolated
    neuron".  Beryl is the mapping used by ``ibl_data_utils.list_brain_regions``;
    it sends white matter and unassigned channels to ``root``/``void``.
    """
    br = get_brain_regions()
    beryl = np.asarray(br.acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl'))
    label = clusters['label'].to_numpy(dtype=float)
    keep = (label >= QC_LABEL) & ~np.isin(beryl, NON_GREY)
    return np.nonzero(keep)[0], beryl


def bin_spikes(spike_times, spike_units, n_units, interval_begs):
    """Spike counts in ``NBINS`` bins of ``BINSIZE`` starting at each interval.

    Vectorised equivalent of ``ibl_data_utils.get_spike_data_per_interval``: spikes
    with ``t_beg <= t < t_end`` are assigned to ``floor((t - t_beg)/binsize)`` and
    the result is truncated to ``NBINS`` bins.

    Parameters
    ----------
    spike_times : (n_spikes,) float, sorted ascending
    spike_units : (n_spikes,) int in [0, n_units)
    interval_begs : (n_trials,) float, may contain NaN

    Returns
    -------
    (n_trials, n_units, NBINS) float32 array of counts
    """
    n_trials = len(interval_begs)
    out = np.zeros((n_trials, n_units, NBINS), dtype=np.float32)
    if n_units == 0 or len(spike_times) == 0:
        return out
    begs = np.asarray(interval_begs, dtype=float)
    safe = np.where(np.isnan(begs), np.inf, begs)
    i0 = np.searchsorted(spike_times, safe, side='left')
    i1 = np.searchsorted(spike_times, safe + BINSIZE * NBINS, side='left')
    for k in range(n_trials):
        if not np.isfinite(begs[k]) or i1[k] <= i0[k]:
            continue
        tt = spike_times[i0[k]:i1[k]]
        uu = spike_units[i0[k]:i1[k]]
        b = ((tt - begs[k]) / BINSIZE).astype(np.int64)
        np.clip(b, 0, NBINS - 1, out=b)
        counts = np.bincount(uu * NBINS + b, minlength=n_units * NBINS)
        out[k] = counts.reshape(n_units, NBINS)
    return out


# ---------------------------------------------------------------------------
# Continuous behaviour
# ---------------------------------------------------------------------------
def behavior_per_interval(target_times, target_vals, interval_begs):
    """Resample a continuous behaviour onto the 100 bins of each trial.

    Reimplements ``ibl_data_utils.get_behavior_per_interval`` (vectorised, and
    interpolating against the full session trace rather than the per-trial slice --
    linear interpolation is local, so interior values are identical, and the reference's
    ``fill_value='extrapolate'`` only ever applied within one bin of the slice edge).
    The reference's rejection criteria are kept verbatim:

      * no samples inside the interval,
      * interval bounds are NaN,
      * the trace starts more than one bin after the interval start,
      * the trace ends more than one bin before the interval end.

    Returns
    -------
    vals : (n_trials, NBINS) float array (NaN rows where the trial is rejected)
    good : (n_trials,) bool
    """
    n_trials = len(interval_begs)
    vals = np.full((n_trials, NBINS), np.nan)
    good = np.zeros(n_trials, dtype=bool)
    if target_times is None or target_vals is None or len(target_times) == 0:
        return vals, good

    begs = np.asarray(interval_begs, dtype=float)
    ends = begs + BINSIZE * NBINS
    safe_b = np.where(np.isnan(begs), np.inf, begs)
    safe_e = np.where(np.isnan(ends), np.inf, ends)

    # Reference slicing: searchsorted(..., 'right') for the start, 'left' for the end.
    idx_beg = np.searchsorted(target_times, safe_b, side='right')
    idx_end = np.searchsorted(target_times, safe_e, side='left')

    n = len(target_times)
    nonempty = (idx_end > idx_beg) & np.isfinite(begs)
    first_t = np.where(nonempty, target_times[np.clip(idx_beg, 0, n - 1)], np.nan)
    last_t = np.where(nonempty, target_times[np.clip(idx_end - 1, 0, n - 1)], np.nan)
    ok = nonempty & (np.abs(begs - first_t) <= BINSIZE) & (np.abs(ends - last_t) <= BINSIZE)

    if not np.any(ok):
        return vals, good

    # Query grid = bin right edges, exactly linspace(beg + binsize, end, NBINS).
    q = begs[ok][:, None] + BIN_RIGHT_EDGES[None, :] - WIN[0]
    interp = np.interp(q.ravel(), target_times, target_vals).reshape(q.shape)
    vals[ok] = interp
    good[ok] = True
    # A NaN anywhere in the resampled trace makes the trial unusable (the decoder
    # format forbids NaN and the value has to be discretised).
    bad = ~np.all(np.isfinite(vals[ok]), axis=1)
    if np.any(bad):
        idx = np.nonzero(ok)[0][bad]
        good[idx] = False
    return vals, good


def load_wheel_speed(sess_loader):
    """|wheel velocity| -- ``load_target_behavior(one, eid, 'wheel-speed')``."""
    if sess_loader.wheel is None or len(sess_loader.wheel) == 0:
        sess_loader.load_wheel()
    return (sess_loader.wheel['times'].to_numpy(dtype=float),
            np.abs(sess_loader.wheel['velocity'].to_numpy(dtype=float)))


def load_whisker_me(sess_loader):
    """Whisker motion energy, left camera with the right camera as fallback.

    Same preference order as ``ibl_data_utils.bin_behaviors``.
    """
    for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
        try:
            sess_loader.load_motion_energy(views=[view])
            df = sess_loader.motion_energy[key]
            t = df['times'].to_numpy(dtype=float)
            v = df['whiskerMotionEnergy'].to_numpy(dtype=float)
            finite = np.isfinite(t)
            if finite.sum() < 2:
                continue
            return t[finite], v[finite], view
        except Exception:
            continue
    raise RuntimeError('no whisker motion energy available')


def discretize_tertiles(values):
    """Split ``values`` into 3 within-session equal-occupancy bins.

    ``values`` is (n_trials, NBINS).  Thresholds are the 33.33/66.67 percentiles over
    every retained trial x time bin of this session; see CONVERSION_NOTES Step 5,
    decision 5 for why the thresholds are per-session.
    """
    flat = values.ravel()
    edges = np.percentile(flat, [100.0 / N_DISCRETE_BINS * i
                                 for i in range(1, N_DISCRETE_BINS)])
    # Guard against degenerate (tied) thresholds: np.searchsorted still yields a
    # valid class, it just produces fewer than 3 occupied classes.
    binned = np.searchsorted(edges, flat, side='right').reshape(values.shape)
    return binned.astype(np.int64), edges


# ---------------------------------------------------------------------------
# Task variables
# ---------------------------------------------------------------------------
def trial_number_in_block(probability_left):
    """0-based index of each trial within its constant-``probabilityLeft`` block.

    Computed on the complete trials table (before any trial exclusion) so that the
    value reflects the animal's actual position in the block.
    """
    p = np.asarray(probability_left, dtype=float)
    out = np.zeros(len(p), dtype=np.int64)
    count = 0
    for i in range(len(p)):
        if i > 0 and not (p[i] == p[i - 1] or (np.isnan(p[i]) and np.isnan(p[i - 1]))):
            count = 0
        out[i] = count
        count += 1
    return out


def map_prior(probability_left):
    """{0.2, 0.5, 0.8} -> {0, 1, 2}; anything else -> -1 (trial dropped)."""
    p = np.asarray(probability_left, dtype=float)
    out = np.full(len(p), -1, dtype=np.int64)
    for value, code in PRIOR_MAP.items():
        out[np.isclose(p, value)] = code
    return out


# ---------------------------------------------------------------------------
# Per-session conversion
# ---------------------------------------------------------------------------
def convert_session(eid, pids, probe_names, subject, lab, show_processing=False,
                    out_dir='/app'):
    """Convert one session; returns a dict or raises."""
    from brainbox.io.one import SessionLoader
    from utils.ibl_data_utils import load_trials_and_mask

    t_start = time.time()
    timing = {}
    one = get_one()

    # ---- neural -----------------------------------------------------------
    t = time.time()
    spikes, clusters, rec_span = load_session_spikes(eid, pids, probe_names)
    timing['load_spikes'] = time.time() - t

    keep_idx, beryl = select_units(clusters)
    n_all_clusters = len(clusters)
    n_label_good = int((clusters['label'].to_numpy(dtype=float) >= QC_LABEL).sum())
    if len(keep_idx) < MIN_NEURONS:
        raise RuntimeError(f'only {len(keep_idx)} well-isolated grey-matter units')
    regions = beryl[keep_idx]

    # ---- trials -----------------------------------------------------------
    t = time.time()
    sess_loader = SessionLoader(one=one, eid=eid)
    trials, ref_mask = load_trials_and_mask(one=one, eid=eid,
                                            max_trial_len=MAX_TRIAL_LEN,
                                            sess_loader=sess_loader)
    timing['load_trials'] = time.time() - t
    n_trials_raw = len(trials)
    mask = ref_mask.to_numpy().astype(bool)

    prior_code = map_prior(trials['probabilityLeft'].to_numpy())
    mask &= prior_code >= 0
    choice_raw = trials['choice'].to_numpy(dtype=float)
    mask &= np.isin(choice_raw, (-1.0, 1.0))

    align_times = trials[PARAMS['align_time']].to_numpy(dtype=float)
    mask &= np.isfinite(align_times)
    interval_begs = align_times + WIN[0]

    # The trial window has to be covered by the ephys recording of every probe;
    # otherwise the "spike counts" would be zeros that mean "not recorded".
    with np.errstate(invalid='ignore'):
        mask &= (interval_begs >= rec_span[0]) & (interval_begs + BINSIZE * NBINS <= rec_span[1])
    n_trials_in_recording = int(mask.sum())

    block_idx = trial_number_in_block(trials['probabilityLeft'].to_numpy())

    # ---- continuous behaviour --------------------------------------------
    t = time.time()
    wt, wv = load_wheel_speed(sess_loader)
    wheel_vals, wheel_ok = behavior_per_interval(wt, wv, interval_begs)
    mt, mv, me_view = load_whisker_me(sess_loader)
    me_vals, me_ok = behavior_per_interval(mt, mv, interval_begs)
    timing['load_behavior'] = time.time() - t

    mask &= wheel_ok & me_ok
    n_trials_kept = int(mask.sum())
    if n_trials_kept < MIN_TRIALS:
        raise RuntimeError(f'only {n_trials_kept} usable trials')

    # ---- bin spikes on the retained trials -------------------------------
    t = time.time()
    remap = np.full(int(np.max(spikes['clusters'])) + 2, -1, dtype=np.int64)
    remap[keep_idx] = np.arange(len(keep_idx))
    su = remap[spikes['clusters']]
    sel = su >= 0
    binned = bin_spikes(np.ascontiguousarray(spikes['times'][sel]),
                        np.ascontiguousarray(su[sel]),
                        len(keep_idx),
                        interval_begs[mask])
    timing['bin_spikes'] = time.time() - t

    # ---- outputs ----------------------------------------------------------
    choice_out = ((1 - choice_raw[mask]) / 2).astype(np.int64)      # +1 -> 0 (left)
    prior_out = prior_code[mask]
    wheel_kept = wheel_vals[mask]
    me_kept = me_vals[mask]
    wheel_bin, wheel_edges = discretize_tertiles(wheel_kept)
    me_bin, me_edges = discretize_tertiles(me_kept)
    # Tied tertile edges mean the trace is constant over at least a third of the
    # session -- a broken ROI or a stuck wheel, not behaviour. Such a session would
    # contribute an output with no variation at all, so it is dropped.
    for name, edges in (('wheel speed', wheel_edges), ('whisker motion energy', me_edges)):
        if not np.all(np.diff(edges) > 0):
            raise RuntimeError(f'{name} is degenerate (tertile edges {list(edges)})')

    n_kept = n_trials_kept
    outputs = np.empty((n_kept, len(OUTPUT_NAMES), NBINS), dtype=np.int64)
    outputs[:, 0, :] = choice_out[:, None]
    outputs[:, 1, :] = prior_out[:, None]
    outputs[:, 2, :] = wheel_bin
    outputs[:, 3, :] = me_bin

    # ---- inputs -----------------------------------------------------------
    inputs = np.empty((n_kept, len(INPUT_NAMES), NBINS), dtype=np.float32)
    inputs[:, 0, :] = BIN_LEFT_EDGES[None, :]
    inputs[:, 1, :] = block_idx[mask][:, None]

    region_names, region_idx = np.unique(regions, return_inverse=True)

    result = {
        'eid': eid,
        'subject': subject,
        'lab': lab,
        'neural': [np.ascontiguousarray(binned[k]) for k in range(n_kept)],
        'input': [np.ascontiguousarray(inputs[k]) for k in range(n_kept)],
        'output': [np.ascontiguousarray(outputs[k]) for k in range(n_kept)],
        'region_names': list(region_names),
        'region_idx': region_idx.astype(np.int64),
        'info': {
            'eid': eid,
            'subject': subject,
            'lab': lab,
            'n_probes': len(pids),
            'n_clusters_all': int(n_all_clusters),
            'n_clusters_label_good': n_label_good,
            'n_neurons': int(len(keep_idx)),
            'n_trials_raw': int(n_trials_raw),
            'n_trials_kept': int(n_trials_kept),
            'n_trials_ref_mask': int(ref_mask.to_numpy().sum()),
            'n_trials_in_recording': n_trials_in_recording,
            'n_trials_all_zero_neural': int(np.sum(binned.sum(axis=(1, 2)) == 0)),
            'recording_span_s': [float(rec_span[0]), float(rec_span[1])],
            # index of each retained trial in the raw trials table, for provenance
            # and for the independent sanity checks in CONVERSION_NOTES Step 10
            'kept_trial_idx': np.nonzero(mask)[0].astype(np.int32),
            'cluster_uuids': list(clusters['uuids'].to_numpy()[keep_idx]),
            'frac_correct_raw': float(np.nanmean(trials['feedbackType'].to_numpy() == 1)),
            'whisker_camera': me_view,
            'wheel_speed_tertile_edges': [float(x) for x in wheel_edges],
            'whisker_me_tertile_edges': [float(x) for x in me_edges],
            'mean_firing_rate_hz': float(binned.mean() / BINSIZE),
        },
        'timing': timing,
    }
    timing['total'] = time.time() - t_start

    if show_processing:
        try:
            plot_processing(out_dir, eid, trials, mask, spikes, su, sel, keep_idx,
                            interval_begs, wt, wv, mt, mv, wheel_kept, me_kept,
                            wheel_bin, me_bin, wheel_edges, me_edges, binned,
                            inputs, outputs, me_view)
        except Exception:
            traceback.print_exc()

    return result


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------
def plot_processing(out_dir, eid, trials, mask, spikes, su, sel, keep_idx,
                    interval_begs, wt, wv, mt, mv, wheel_kept, me_kept,
                    wheel_bin, me_bin, wheel_edges, me_edges, binned,
                    inputs, outputs, me_view):
    """Every processing step for one session, on one page."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    kept_begs = interval_begs[mask]
    ktrial = min(3, len(kept_begs) - 1)
    t0 = kept_begs[ktrial]
    stim_on = t0 - WIN[0]

    fig, axes = plt.subplots(4, 2, figsize=(18, 20))

    # (0,0) raw raster of the example trial + bin edges
    ax = axes[0, 0]
    st = spikes['times'][sel]
    sc = su[sel]
    m = (st >= t0) & (st < t0 + BINSIZE * NBINS)
    ax.plot((st[m] - stim_on), sc[m], '.', ms=1.5, color='k')
    for e in (BIN_LEFT_EDGES[::10]):
        ax.axvline(e, color='0.85', lw=0.5, zorder=0)
    ax.axvline(0, color='r', lw=1.5)
    ax.set_title(f'{eid[:8]} trial {ktrial}: raw spikes (good grey-matter units)\n'
                 f'red = stimOn, window [{WIN[0]}, {WIN[1]}] s')
    ax.set_xlabel('time from stimulus onset (s)'); ax.set_ylabel('unit')
    ax.set_xlim(WIN)

    # (0,1) binned spike counts for the same trial
    ax = axes[0, 1]
    im = ax.imshow(binned[ktrial], aspect='auto', origin='lower', cmap='Greys',
                   extent=[WIN[0], WIN[1], -0.5, binned.shape[1] - 0.5],
                   interpolation='nearest')
    ax.axvline(0, color='r', lw=1.5)
    ax.set_title(f'binned spike counts, {int(BINSIZE*1000)} ms bins, T={NBINS}\n'
                 f'total spikes raster={int(m.sum())} vs binned={int(binned[ktrial].sum())}')
    ax.set_xlabel('time from stimulus onset (s)'); ax.set_ylabel('unit')
    plt.colorbar(im, ax=ax, label='spikes/bin')

    # (1,0) wheel speed: raw trace vs resampled vs discretised
    ax = axes[1, 0]
    m = (wt >= t0 - 0.2) & (wt <= t0 + BINSIZE * NBINS + 0.2)
    ax.plot(wt[m] - stim_on, wv[m], color='0.6', lw=1, label='raw |velocity| (1 kHz)')
    ax.plot(BIN_RIGHT_EDGES, wheel_kept[ktrial], 'o-', ms=3, color='C0',
            label='resampled onto bin right edges')
    for e in wheel_edges:
        ax.axhline(e, color='C3', ls='--', lw=1)
    ax.axvline(0, color='r', lw=1.5)
    ax.set_yscale('log')
    ax.set_title('wheel speed: raw vs resampled; dashed = session tertile edges')
    ax.set_xlabel('time from stimulus onset (s)'); ax.set_ylabel('rad/s')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.step(BIN_RIGHT_EDGES, wheel_bin[ktrial], where='mid', color='C0')
    ax.axvline(0, color='r', lw=1.5)
    ax.set_ylim(-0.5, 2.5); ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(OUTPUT_VALUES[2])
    ax.set_title('wheel speed discretised (output[2])')
    ax.set_xlabel('time from stimulus onset (s)')

    # (2,0) whisker motion energy
    ax = axes[2, 0]
    m = (mt >= t0 - 0.2) & (mt <= t0 + BINSIZE * NBINS + 0.2)
    ax.plot(mt[m] - stim_on, mv[m], color='0.6', lw=1, marker='.', ms=3,
            label=f'raw {me_view} camera')
    ax.plot(BIN_RIGHT_EDGES, me_kept[ktrial], 'o-', ms=3, color='C2',
            label='resampled onto bin right edges')
    for e in me_edges:
        ax.axhline(e, color='C3', ls='--', lw=1)
    ax.axvline(0, color='r', lw=1.5)
    ax.set_title('whisker motion energy: raw vs resampled; dashed = tertile edges')
    ax.set_xlabel('time from stimulus onset (s)'); ax.set_ylabel('a.u.')
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    ax.step(BIN_RIGHT_EDGES, me_bin[ktrial], where='mid', color='C2')
    ax.axvline(0, color='r', lw=1.5)
    ax.set_ylim(-0.5, 2.5); ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(OUTPUT_VALUES[3])
    ax.set_title('whisker motion energy discretised (output[3])')
    ax.set_xlabel('time from stimulus onset (s)')

    # (3,0) session-level: task variables over trials
    ax = axes[3, 0]
    ax.plot(trials['probabilityLeft'].to_numpy(), color='0.4', label='probabilityLeft')
    ax.plot(np.nonzero(mask)[0], outputs[:, 0, 0] * 0.1 + 0.9, '.', ms=3, color='C1',
            label='choice (0=left, 1=right, offset)')
    ax.plot(np.nonzero(mask)[0], outputs[:, 1, 0] / 10 + 0.05, '.', ms=3, color='C4',
            label='prior code /10')
    ax.set_xlabel('trial'); ax.legend(fontsize=8)
    ax.set_title('block structure and per-trial outputs')

    ax = axes[3, 1]
    ax.plot(np.nonzero(mask)[0], inputs[:, 1, 0], '.', ms=3, color='C5')
    ax.set_xlabel('trial'); ax.set_ylabel('trial number in block')
    ax.set_title('input[1]: trial number within block '
                 f'(input[0] = time, range [{BIN_LEFT_EDGES[0]:.2f}, {BIN_LEFT_EDGES[-1]:.2f}] s)')

    fig.suptitle(f'Processing steps: session {eid}', fontsize=14)
    fig.tight_layout()
    path = os.path.join(out_dir, f'processing_{eid}.png')
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f'  wrote {path}', flush=True)


# ---------------------------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------------------------
def _worker(job):
    eid, pids, pnames, subject, lab, show, out_dir = job
    try:
        res = convert_session(eid, pids, pnames, subject, lab,
                              show_processing=show, out_dir=out_dir)
        return ('ok', eid, res)
    except Exception as exc:                       # mirrors the reference try/except
        return ('fail', eid, f'{type(exc).__name__}: {exc}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('outfile', type=str, help='path of the output pickle')
    ap.add_argument('--full', action='store_true', default=True,
                    help='process all sessions (default)')
    ap.add_argument('--sample', action='store_true',
                    help='process only 2 sessions, for testing')
    ap.add_argument('--show-processing', action='store_true',
                    help='write processing_<eid>.png for up to 2 sessions')
    ap.add_argument('--n-workers', type=int, default=16)
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.outfile)) or '.'
    t_start = time.time()

    bwm = pd.read_csv(BWM_FREEZE, index_col=0)
    eids = list(dict.fromkeys(bwm.eid.tolist()))       # preserve freeze order
    if args.sample:
        # Two sessions from two different mice, so that the sample exercises the
        # subject/brain-region indexing as well as the per-session processing.
        first = eids[0]
        other = next(e for e in eids
                     if bwm[bwm.eid == e].subject.iloc[0] != bwm[bwm.eid == first].subject.iloc[0])
        eids = [first, other]
    print(f'Converting {len(eids)} sessions from {BWM_FREEZE}', flush=True)
    print(f'  align={PARAMS["align_time"]} window={WIN} binsize={BINSIZE} T={NBINS}',
          flush=True)

    jobs = []
    for i, eid in enumerate(eids):
        sub = bwm[bwm.eid == eid]
        show = args.show_processing and i < 2
        jobs.append((eid, sub.pid.tolist(), sub.probe_name.tolist(),
                     sub.subject.iloc[0], sub.lab.iloc[0], show, out_dir))

    results, failures = [], []
    n_workers = min(args.n_workers, max(1, len(jobs)))
    if n_workers == 1 or args.show_processing:
        outs = (_worker(j) for j in jobs)
        outs = list(outs)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            outs = list(pool.map(_worker, jobs, chunksize=1))

    for status, eid, payload in outs:
        if status == 'ok':
            results.append(payload)
        else:
            failures.append((eid, payload))
            print(f'  SKIPPED {eid}: {payload}', flush=True)

    print(f'\nConverted {len(results)} sessions, skipped {len(failures)}', flush=True)

    # ---- assemble the dataset --------------------------------------------
    results.sort(key=lambda r: (r['subject'], r['eid']))

    subjects = sorted({r['subject'] for r in results})
    subj_lookup = {s: i for i, s in enumerate(subjects)}
    brain_regions = sorted({reg for r in results for reg in r['region_names']})
    reg_lookup = {r: i for i, r in enumerate(brain_regions)}

    data = {
        'neural': [r['neural'] for r in results],
        'input': [r['input'] for r in results],
        'output': [r['output'] for r in results],
        'subjects': subjects,
        'subject_idx': np.array([subj_lookup[r['subject']] for r in results], dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': [
            np.array([reg_lookup[r['region_names'][i]] for i in r['region_idx']],
                     dtype=np.int64)
            for r in results
        ],
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'dataset': 'IBL Brain-wide Map (public release), bwm_release.csv freeze',
            'source_papers': [
                'IBL et al., A brain-wide map of neural activity during complex behaviour',
                'Zhang et al., Exploiting correlations across trials and behavioral '
                'sessions to improve neural decoding, Neuron 2026',
            ],
            'task_description':
                'Head-fixed mice turn a wheel to report the side of a visual Gabor stimulus. '
                'Decoded from binned spike counts: the choice (left/right), the block prior '
                'probability that the stimulus appears on the left (0.2/0.5/0.8), and the '
                'within-trial wheel speed and whisker motion energy, each discretised into 3 '
                'within-session tertiles.',
            'time_bin_size': BINSIZE * 1000.0,           # ms
            'temporal_alignment_event': 'visual stimulus onset (trials.stimOn_times)',
            'off_start': float(WIN[0]),
            'off_end': float(WIN[1]),
            'n_time_bins': NBINS,
            'neural_data_type': 'spike counts per 20 ms bin (well-isolated units, '
                                'clusters.label >= 1, grey matter only)',
            'input_descriptions': {
                'time_from_stim_on': 'seconds from stimulus onset to the left edge of the '
                                     'time bin; -0.50 ... 1.48',
                'trial_num_in_block': '0-based index of the trial within its constant-'
                                      'probabilityLeft block (constant within a trial)',
            },
            'output_descriptions': {
                'choice': "mouse's reported stimulus side; 0 = left (trials.choice == +1), "
                          '1 = right (trials.choice == -1); constant within a trial',
                'prior_prob_left': 'block probability that the stimulus appears on the left; '
                                   '0.2 -> 0, 0.5 -> 1, 0.8 -> 2; constant within a trial',
                'wheel_speed': '|wheel velocity| resampled to the 20 ms bins and discretised '
                               'into 3 equal-occupancy within-session bins',
                'whisker_motion_energy': 'whisker-pad motion energy (left camera, right as '
                                         'fallback) resampled to the 20 ms bins and '
                                         'discretised into 3 equal-occupancy within-session '
                                         'bins',
            },
            'trial_inclusion': 'ibl_data_utils.load_trials_and_mask(max_trial_len=10.0): '
                               'reaction time in [0.08, 2.0] s, feedback_times - goCue_times '
                               '<= 10 s, no NaN in stimOn_times/choice/feedback_times/'
                               'probabilityLeft/firstMovement_times/feedbackType, choice != 0; '
                               'plus the wheel and whisker traces and the ephys recording of '
                               'every probe must cover the trial window',
            'neuron_inclusion': 'clusters.label >= 1 (well-isolated: amplitude > 50 uV, '
                                'noise cut-off < 20 uV, refractory-period violation) and '
                                'Beryl region not root/void (grey matter only); probes of a '
                                'session merged; sessions need >= 5 such neurons and >= 2 '
                                'usable trials',
            'session_info': [r['info'] for r in results],
            'failed_sessions': [{'eid': e, 'reason': m} for e, m in failures],
            'n_sessions_in_freeze': len(eids),
        },
    }

    print(f'Writing {args.outfile} ...', flush=True)
    t = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'  wrote in {time.time()-t:.1f}s, '
          f'{os.path.getsize(args.outfile)/1e9:.2f} GB', flush=True)

    # ---- summary ----------------------------------------------------------
    info = data['metadata']['session_info']
    ntr = np.array([i['n_trials_kept'] for i in info])
    nraw = np.array([i['n_trials_raw'] for i in info])
    nneu = np.array([i['n_neurons'] for i in info])
    print('\n--- conversion summary ---')
    print(f'sessions: {len(info)} / {len(eids)} in freeze')
    print(f'subjects: {len(subjects)}')
    print(f'brain regions: {len(brain_regions)}')
    print(f'neurons: total {nneu.sum()}, mean/session {nneu.mean():.1f}, '
          f'min {nneu.min()}, max {nneu.max()}')
    print(f'good units per probe (mean): '
          f'{np.sum([i["n_clusters_label_good"] for i in info]) / np.sum([i["n_probes"] for i in info]):.1f}')
    print(f'trials raw: total {nraw.sum()}, mean/session {nraw.mean():.1f}, '
          f'median {np.median(nraw):.0f}, min {nraw.min()}, max {nraw.max()}')
    print(f'trials kept: total {ntr.sum()}, mean/session {ntr.mean():.1f}, '
          f'min {ntr.min()}, max {ntr.max()}')
    print(f'fraction correct (raw trials, mean over sessions): '
          f'{np.mean([i["frac_correct_raw"] for i in info]):.3f}')
    print(f'mean firing rate: {np.mean([i["mean_firing_rate_hz"] for i in info]):.2f} Hz')
    ttl = [r['timing'] for r in results]
    for k in ('load_spikes', 'load_trials', 'load_behavior', 'bin_spikes', 'total'):
        vals = [t_[k] for t_ in ttl if k in t_]
        if vals:
            print(f'timing {k:14s}: mean {np.mean(vals):.2f}s  total {np.sum(vals):.1f}s')
    print(f'wall clock: {time.time()-t_start:.1f}s')


if __name__ == '__main__':
    main()
