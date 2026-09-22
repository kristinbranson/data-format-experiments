#!/usr/bin/env python3
"""Convert the IBL Brain-Wide Map dataset into the decoder-ready pickle format.

Usage
-----
    python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Processing follows the reference caching pipeline of Zhang et al. 2026
(`/app/code/code_zhang2025/src/0_data_caching.py` + `src/utils/ibl_data_utils.py`):

    params = {'interval_len': 2, 'binsize': 0.02, 'single_region': False,
              'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}

i.e. trials aligned to stimulus onset, a 2 s window from -0.5 s to +1.5 s, binned into
100 non-overlapping 20 ms bins, with all neurons of a session pooled across probes.
Trial curation uses the reference's `load_trials_and_mask(..., max_trial_len=10.0)`,
which implements the data paper's trial exclusions.  Neurons are restricted to the data
paper's "well-isolated neurons" (`clusters.label >= 1`, i.e. all three RIGOR single-unit
metrics passed) lying in grey matter (Beryl acronym not `root`/`void`).

All data access goes through the ONE API and the brainbox loaders.
"""
import argparse
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# The reference implementation, imported rather than re-written where possible.
REF_SRC = Path('/app/code/code_zhang2025/src')
sys.path.insert(0, str(REF_SRC))

from one.api import ONE                                     # noqa: E402
from brainbox.io.one import SpikeSortingLoader, SessionLoader  # noqa: E402
from iblatlas.regions import BrainRegions                    # noqa: E402
from utils.ibl_data_utils import (                           # noqa: E402
    load_trials_and_mask,
    merge_probes,
)

# ----------------------------------------------------------------------------------
# Parameters (identical to the reference caching script)
# ----------------------------------------------------------------------------------
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))   # 100
MAX_TRIAL_LEN = 10.0            # reference: prepare_data(..., max_trial_len=10.0)
MIN_RT, MAX_RT = 0.08, 2.0      # data paper trial exclusions
GOOD_UNIT_LABEL = 1.0           # RIGOR: all three single-unit metrics passed
NON_GREY = ('root', 'void')     # excluded from "grey matter" analyses
MIN_TRIALS_PER_SESSION = 2      # the decoder needs >= 2 trials to be evaluable

BWM_FREEZE = '/app/code/code_zhang2025/data/bwm_release.csv'

OUTPUT_NAMES = ['choice', 'prior', 'wheel_speed', 'whisker_motion_energy']
OUTPUT_VALUES = [
    ['left', 'right'],
    ['p_left=0.2', 'p_left=0.5', 'p_left=0.8'],
    ['low', 'medium', 'high'],
    ['low', 'medium', 'high'],
]
INPUT_NAMES = ['time_from_stimulus_onset', 'trial_number_in_block']

_ONE = None
_BR = None


def ensure_cache_tables():
    """Make sure the local ONE cache has tables that index the files actually on disk.

    The staged release tables do not list the dataset revisions present in the cache
    (e.g. `alf/#2025-03-03#/_ibl_trials.table.pqt`), so ONE would silently return a
    one-column trials table.  See build_one_cache.py.
    """
    tables = Path('/app/data/one_cache/sessions.pqt')
    if not tables.exists():
        print('ONE cache tables missing; rebuilding from the staged cache ...')
        import build_one_cache
        build_one_cache.main()


def get_one():
    """One ONE instance and one BrainRegions per process (both are expensive)."""
    global _ONE, _BR
    if _ONE is None:
        _ONE = ONE(base_url='https://openalyx.internationalbrainlab.org',
                   silent=True, mode='local')
        _BR = BrainRegions()
    return _ONE, _BR


# ----------------------------------------------------------------------------------
# Neural data
# ----------------------------------------------------------------------------------
def load_session_spikes(one, br, eid, probes):
    """Load and merge the spike sorting of every probe of a session.

    Mirrors `ibl_data_utils.prepare_data`: each probe is loaded with
    `SpikeSortingLoader`, clusters are merged with channel information (which attaches
    the anatomical `acronym` and the QC `label`), and probes are concatenated with
    `merge_probes` so that the session is treated as one population -- the data paper
    does the same ("neurons in the same session and region were combined across probes").

    Only `spikes.times` and `spikes.clusters` are read; the other spike attributes
    (`depths`, `amps`) are not used and are several hundred MB per probe.

    Returns
    -------
    spike_times, spike_clusters : np.ndarray
        Session-wide spike train, cluster ids indexing into `clusters`.
    clusters : pd.DataFrame
        Merged cluster table with `label` and `acronym`.
    beryl : np.ndarray of str
        Beryl-mapped acronym per cluster.
    """
    spikes_list, clusters_list = [], []
    for pid, pname in probes:
        ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
        ssl.download_spike_sorting_object('spikes')
        wanted = [f for f in ssl.files['spikes']
                  if f.name.split('.')[1] in ('times', 'clusters')]
        spikes = dict(ssl._load_object(wanted))
        clusters = ssl.load_spike_sorting_object('clusters')
        channels = ssl.load_channels()
        clusters = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
        clusters['pid'] = pid
        spikes_list.append({'times': spikes['times'],
                            'clusters': spikes['clusters'].astype(np.int64)})
        clusters_list.append(clusters)

    spikes, clusters = merge_probes(spikes_list, clusters_list)
    beryl = br.acronym2acronym(clusters['acronym'].to_numpy(), mapping='Beryl')
    return spikes['times'], spikes['clusters'], clusters, np.asarray(beryl)


def bin_spikes(spike_times, spike_clusters, n_clusters, t_begs):
    """Bin spikes into `N_BINS` bins of `BINSIZE` starting at each `t_begs`.

    Equivalent to the reference's `bin_spiking_data`, which calls
    `bincount2D(times, clusters, xbin=binsize, xlim=[t_beg, t_end])` per trial and then
    truncates to `n_bins` columns: bin index = floor((t - t_beg) / binsize) for spikes in
    [t_beg, t_end), i.e. left-closed bins anchored at the window start.  Implemented here
    with a single `np.bincount` over all trials instead of a per-trial multiprocessing
    pool, which is ~100x faster and produces bit-identical counts (verified in Step 10).

    Parameters
    ----------
    spike_times : (n_spikes,) sorted ascending
    spike_clusters : (n_spikes,) int in [0, n_clusters)
    n_clusters : int
    t_begs : (n_trials,) window start times

    Returns
    -------
    (n_trials, n_clusters, N_BINS) float32 spike counts
    """
    n_trials = len(t_begs)
    t_ends = t_begs + N_BINS * BINSIZE

    i0 = np.searchsorted(spike_times, t_begs, side='left')
    i1 = np.searchsorted(spike_times, t_ends, side='left')
    counts = np.maximum(i1 - i0, 0)
    total = int(counts.sum())

    binned = np.zeros((n_trials, n_clusters, N_BINS), dtype=np.float32)
    if total == 0:
        return binned

    # Gather the spike indices of every trial window into one flat array.  Windows of
    # consecutive trials can in principle overlap, so this is a gather, not a slice.
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
    trial_id = np.repeat(np.arange(n_trials), counts)
    idx = np.arange(total) - np.repeat(offsets, counts) + np.repeat(i0, counts)

    rel = spike_times[idx] - t_begs[trial_id]
    b = np.floor(rel / BINSIZE).astype(np.int64)
    ok = (b >= 0) & (b < N_BINS)          # the reference truncates to n_bins columns
    if not ok.all():
        b, trial_id, idx = b[ok], trial_id[ok], idx[ok]

    flat = (trial_id * n_clusters + spike_clusters[idx]) * N_BINS + b
    binned.reshape(-1)[:] = np.bincount(
        flat, minlength=n_trials * n_clusters * N_BINS).astype(np.float32)
    return binned


# ----------------------------------------------------------------------------------
# Continuous behaviour
# ----------------------------------------------------------------------------------
def bin_behavior(target_times, target_vals, t_begs):
    """Interpolate a continuous behavioural trace onto the per-trial bin grid.

    Reproduces the reference's `get_behavior_per_interval`: the trace is evaluated at
    `np.linspace(t_beg + binsize, t_end, n_bins)`, i.e. at the *right edge* of each
    spike-count bin, and a trial is rejected when the trace does not cover the window --
    no samples inside it, the first sample inside starts more than one bin after the
    window start, or the last sample inside ends more than one bin before the window end.

    Returns
    -------
    values : (n_trials, N_BINS) float64, NaN for rejected trials
    good : (n_trials,) bool
    """
    n_trials = len(t_begs)
    t_ends = t_begs + N_BINS * BINSIZE

    values = np.full((n_trials, N_BINS), np.nan)
    good = np.zeros(n_trials, dtype=bool)
    if target_times is None or len(target_times) == 0:
        return values, good

    ib = np.searchsorted(target_times, t_begs, side='right')
    ie = np.searchsorted(target_times, t_ends, side='left')

    nonempty = ie > ib
    first = np.where(nonempty, target_times[np.clip(ib, 0, len(target_times) - 1)], np.nan)
    last = np.where(nonempty, target_times[np.clip(ie - 1, 0, len(target_times) - 1)], np.nan)
    with np.errstate(invalid='ignore'):
        good = (nonempty
                & np.isfinite(t_begs) & np.isfinite(t_ends)
                & (np.abs(t_begs - first) <= BINSIZE)
                & (np.abs(t_ends - last) <= BINSIZE))

    if not good.any():
        return values, good

    # One np.interp call for every query point of every good trial.
    grid = np.linspace(BINSIZE, N_BINS * BINSIZE, N_BINS)       # t_beg + binsize ... t_end
    query = t_begs[good][:, None] + grid[None, :]
    interp = np.interp(query.ravel(), target_times, target_vals).reshape(-1, N_BINS)
    values[good] = interp

    # An interpolated NaN means the trace itself has a gap of NaNs across the window.
    nan_rows = np.isnan(values[good]).any(axis=1)
    if nan_rows.any():
        gi = np.flatnonzero(good)
        good[gi[nan_rows]] = False
    return values, good


def load_wheel_speed(one, eid):
    """Wheel speed = |velocity| of the 1000 Hz interpolated wheel trace.

    Same source as the reference's `load_target_behavior(one, eid, 'wheel-speed')`.
    """
    sl = SessionLoader(one=one, eid=eid)
    sl.load_wheel()
    return (sl.wheel['times'].to_numpy(),
            np.abs(sl.wheel['velocity'].to_numpy()))


def load_whisker_me(one, eid):
    """Whisker-pad motion energy, left camera preferred, right camera as fallback.

    Same preference order as the reference's `bin_behaviors` for
    'whisker-motion-energy'.
    """
    for view in ('left', 'right'):
        try:
            sl = SessionLoader(one=one, eid=eid)
            sl.load_motion_energy(views=[view])
            me = sl.motion_energy[f'{view}Camera']
            return (me['times'].to_numpy(),
                    me['whiskerMotionEnergy'].to_numpy(), view)
        except Exception:
            continue
    return None, None, None


def discretize_tertiles(values):
    """Discretise a continuous (n_trials, T) signal into 3 balanced per-session bins.

    Wheel speed and whisker motion energy are continuous, and the decoder outputs must be
    categorical.  Tertile edges are computed from all binned values of the session, so the
    three classes are equally populated and the balanced-accuracy chance level is exactly
    1/3.  Per session (rather than globally) because whisker motion energy is in arbitrary
    units that depend on the camera and the ROI and is not comparable between sessions.

    Returns
    -------
    labels : (n_trials, T) int64 in {0, 1, 2}
    edges : (2,) the tertile edges actually used
    """
    flat = values.ravel()
    edges = np.percentile(flat, [100.0 / 3.0, 200.0 / 3.0])
    if edges[0] == edges[1]:
        # Degenerate: >1/3 of the signal sits on a single value (e.g. a wheel that never
        # moves).  Fall back to edges that at least separate the distinct values present.
        uniq = np.unique(flat)
        if len(uniq) >= 3:
            edges = np.percentile(uniq, [100.0 / 3.0, 200.0 / 3.0])
        elif len(uniq) == 2:
            edges = np.array([uniq[0], uniq[1]])
        else:
            edges = np.array([uniq[0], uniq[0] + 1.0])
        if edges[0] == edges[1]:
            edges[1] = np.nextafter(edges[1], np.inf)
    return np.digitize(values, edges, right=False).astype(np.int64), edges


# ----------------------------------------------------------------------------------
# Task variables
# ----------------------------------------------------------------------------------
def trial_number_in_block(probability_left):
    """0-based index of each trial within its constant-`probabilityLeft` block.

    The session starts with a 90-trial unbiased block (p = 0.5); afterwards the block
    prior alternates between 0.2 and 0.8 with lengths drawn from a truncated geometric
    distribution.  Block boundaries are exactly the points where `probabilityLeft`
    changes.  Computed on the *full* trials table before any trial exclusion, so that
    excluded trials still advance the counter.
    """
    pl = np.asarray(probability_left, dtype=float)
    change = np.ones(len(pl), dtype=bool)
    change[1:] = pl[1:] != pl[:-1]
    block_id = np.cumsum(change) - 1
    starts = np.flatnonzero(change)
    return (np.arange(len(pl)) - starts[block_id]).astype(np.float32)


def map_choice(choice):
    """IBL `choice` (+1 = left, -1 = right) -> task coding (left = 0, right = 1).

    Verified against the trials table: on correct trials with a left stimulus
    `choice == +1`, with a right stimulus `choice == -1`.
    """
    out = np.full(len(choice), -1, dtype=np.int64)
    out[np.asarray(choice) == 1] = 0
    out[np.asarray(choice) == -1] = 1
    return out


def map_prior(probability_left):
    """`probabilityLeft` -> 0.2 -> 0, 0.5 -> 1, 0.8 -> 2 (per the task spec)."""
    pl = np.asarray(probability_left, dtype=float)
    out = np.full(len(pl), -1, dtype=np.int64)
    out[np.isclose(pl, 0.2)] = 0
    out[np.isclose(pl, 0.5)] = 1
    out[np.isclose(pl, 0.8)] = 2
    return out


# ----------------------------------------------------------------------------------
# Per-session conversion
# ----------------------------------------------------------------------------------
def convert_session(job):
    """Convert one session.  Returns a dict of arrays, or a dict with 'skip' set."""
    t_start = time.time()
    res = _convert_session(job)
    res.setdefault('elapsed', time.time() - t_start)
    return res


def _convert_session(job):
    eid, subject, probes = job
    t_start = time.time()
    timing = {}
    one, br = get_one()
    res = {'eid': eid, 'subject': subject, 'n_probes': len(probes)}

    try:
        # --- trials + inclusion mask (reference: load_trials_and_mask) --------------
        t0 = time.time()
        sl = SessionLoader(one=one, eid=eid)
        sl.load_trials()
        trials, mask = load_trials_and_mask(
            one=one, eid=eid, min_rt=MIN_RT, max_rt=MAX_RT, nan_exclude='default',
            max_trial_len=MAX_TRIAL_LEN, exclude_unbiased=False, exclude_nochoice=True,
            sess_loader=sl)
        mask = np.asarray(mask, dtype=bool)
        res['n_trials_total'] = len(trials)
        res['n_trials_mask'] = int(mask.sum())
        timing['trials'] = time.time() - t0

        # trial-in-block is computed on the full table so exclusions do not renumber it
        tib_all = trial_number_in_block(trials['probabilityLeft'].to_numpy())

        keep = np.flatnonzero(mask)
        if len(keep) < MIN_TRIALS_PER_SESSION:
            res['skip'] = f'only {len(keep)} trials pass the trial mask'
            return res

        t_begs = trials[ALIGN_TIME].to_numpy()[keep] + TIME_WINDOW[0]

        # --- continuous behaviour ---------------------------------------------------
        t0 = time.time()
        wt, wv = load_wheel_speed(one, eid)
        wheel_vals, wheel_good = bin_behavior(wt, wv, t_begs)
        mt, mv, view = load_whisker_me(one, eid)
        if mt is None:
            res['skip'] = 'no whisker motion energy (neither camera)'
            return res
        res['whisker_camera'] = view
        me_vals, me_good = bin_behavior(mt, mv, t_begs)
        timing['behavior'] = time.time() - t0

        beh_good = wheel_good & me_good
        res['n_trials_wheel_bad'] = int((~wheel_good).sum())
        res['n_trials_me_bad'] = int((~me_good).sum())

        # --- spikes ----------------------------------------------------------------
        t0 = time.time()
        spike_times, spike_clusters, clusters, beryl = load_session_spikes(
            one, br, eid, probes)
        res['n_units_all'] = len(clusters)
        label = clusters['label'].to_numpy()
        good_unit = (label >= GOOD_UNIT_LABEL) & ~np.isin(beryl, NON_GREY)
        res['n_units_good'] = int((label >= GOOD_UNIT_LABEL).sum())
        res['n_units_kept'] = int(good_unit.sum())
        if res['n_units_kept'] == 0:
            res['skip'] = 'no well-isolated grey-matter units'
            return res

        # Re-index the kept clusters to 0..n_kept-1 and subselect their spikes.
        kept_ids = np.flatnonzero(good_unit)
        remap = np.full(len(clusters), -1, dtype=np.int64)
        remap[kept_ids] = np.arange(len(kept_ids))
        sel = remap[spike_clusters] >= 0
        spike_times_k = spike_times[sel]
        spike_clusters_k = remap[spike_clusters[sel]]
        order = np.argsort(spike_times_k, kind='stable')
        spike_times_k = spike_times_k[order]
        spike_clusters_k = spike_clusters_k[order]
        timing['spikes_load'] = time.time() - t0

        # A trial whose window falls outside the recorded spike train would silently
        # become an all-zero matrix; drop it instead.
        if len(spike_times_k):
            in_rec = ((t_begs >= spike_times_k[0] - BINSIZE)
                      & (t_begs + N_BINS * BINSIZE <= spike_times_k[-1] + BINSIZE))
        else:
            in_rec = np.zeros(len(t_begs), dtype=bool)
        res['n_trials_outside_recording'] = int((~in_rec).sum())

        valid = beh_good & in_rec
        res['n_trials_valid'] = int(valid.sum())
        if valid.sum() < MIN_TRIALS_PER_SESSION:
            res['skip'] = f'only {int(valid.sum())} trials survive behaviour/spike checks'
            return res

        keep = keep[valid]
        t_begs = t_begs[valid]
        wheel_vals = wheel_vals[valid]
        me_vals = me_vals[valid]

        t0 = time.time()
        binned = bin_spikes(spike_times_k, spike_clusters_k, len(kept_ids), t_begs)
        timing['bin_spikes'] = time.time() - t0

        # --- inputs ----------------------------------------------------------------
        n_trials = len(keep)
        bin_centers = (TIME_WINDOW[0] + (np.arange(N_BINS) + 0.5) * BINSIZE).astype(np.float32)
        inputs = np.empty((n_trials, 2, N_BINS), dtype=np.float32)
        inputs[:, 0, :] = bin_centers[None, :]
        inputs[:, 1, :] = tib_all[keep][:, None]

        # --- outputs ---------------------------------------------------------------
        choice = map_choice(trials['choice'].to_numpy()[keep])
        prior = map_prior(trials['probabilityLeft'].to_numpy()[keep])
        assert choice.min() >= 0, 'unmapped choice value'
        assert prior.min() >= 0, 'unmapped probabilityLeft value'
        wheel_cls, wheel_edges = discretize_tertiles(wheel_vals)
        me_cls, me_edges = discretize_tertiles(me_vals)

        outputs = np.empty((n_trials, 4, N_BINS), dtype=np.int64)
        outputs[:, 0, :] = choice[:, None]
        outputs[:, 1, :] = prior[:, None]
        outputs[:, 2, :] = wheel_cls
        outputs[:, 3, :] = me_cls

        res.update({
            'neural': binned,                       # (n_trials, n_neurons, N_BINS)
            'input': inputs,                        # (n_trials, 2, N_BINS)
            'output': outputs,                      # (n_trials, 4, N_BINS)
            'regions': beryl[kept_ids],
            'trial_idx': keep,                      # index into the full trials table
            'wheel_edges': wheel_edges,
            'me_edges': me_edges,
            'wheel_raw': wheel_vals.astype(np.float32),
            'me_raw': me_vals.astype(np.float32),
            't_begs': t_begs,
            'timing': timing,
            'elapsed': time.time() - t_start,
        })
    except Exception as e:          # keep the run going, report at the end
        import traceback
        res['skip'] = f'{type(e).__name__}: {e}'
        res['traceback'] = traceback.format_exc()
    return res


# ----------------------------------------------------------------------------------
# Diagnostic plots
# ----------------------------------------------------------------------------------
def plot_processing(res, one, out_dir='.'):
    """Plot every processing step for one session, for visual verification."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    eid = res['eid']
    neural, inputs, outputs = res['neural'], res['input'], res['output']
    n_trials = neural.shape[0]
    centers = inputs[0, 0, :]
    right_edges = TIME_WINDOW[0] + (np.arange(N_BINS) + 1) * BINSIZE

    fig, axes = plt.subplots(5, 2, figsize=(22, 26))

    # (1) raw spikes vs binned counts for one example trial
    k = min(5, n_trials - 1)
    t0 = res['t_begs'][k]
    sl = SessionLoader(one=one, eid=eid)
    ax = axes[0, 0]
    ax.imshow(neural[k], aspect='auto', interpolation='nearest', origin='lower',
              extent=[TIME_WINDOW[0], TIME_WINDOW[1], 0, neural.shape[1]], cmap='Greys')
    ax.axvline(0, color='r', lw=2)
    ax.set(title=f'{eid}\nbinned spike counts, trial {k} (red = stimulus onset)',
           xlabel='time from stimulus onset (s)', ylabel='neuron')

    ax = axes[0, 1]
    psth = neural.mean(axis=(0, 1)) / BINSIZE
    ax.plot(centers, psth)
    ax.axvline(0, color='r', lw=2)
    ax.set(title='population PSTH (mean firing rate over all trials/neurons)',
           xlabel='time from stimulus onset (s)', ylabel='spikes/s')

    # (2) raw wheel trace vs binned vs discretised, for one trial
    sl.load_wheel()
    wt = sl.wheel['times'].to_numpy()
    wv = np.abs(sl.wheel['velocity'].to_numpy())
    m = (wt >= t0 - 0.1) & (wt <= t0 + N_BINS * BINSIZE + 0.1)
    ax = axes[1, 0]
    ax.plot(wt[m] - t0 + TIME_WINDOW[0], wv[m], color='0.6', label='raw |wheel velocity|')
    ax.plot(right_edges, res['wheel_raw'][k], 'o-', ms=3, label='interpolated to bins')
    for e in res['wheel_edges']:
        ax.axhline(e, color='g', ls='--', lw=1)
    ax.axvline(0, color='r', lw=2)
    ax.set(title=f'wheel speed, trial {k} (green dashed = tertile edges)',
           xlabel='time from stimulus onset (s)', ylabel='rad/s')
    ax.legend()

    ax = axes[1, 1]
    ax.step(right_edges, outputs[k, 2, :], where='mid')
    ax.axvline(0, color='r', lw=2)
    ax.set(title='discretised wheel speed (output 2)', xlabel='time from stimulus onset (s)',
           ylabel='class', yticks=[0, 1, 2])

    # (3) raw whisker motion energy vs binned vs discretised
    view = res.get('whisker_camera', 'left')
    sl2 = SessionLoader(one=one, eid=eid)
    sl2.load_motion_energy(views=[view])
    me = sl2.motion_energy[f'{view}Camera']
    mt, mv = me['times'].to_numpy(), me['whiskerMotionEnergy'].to_numpy()
    m = (mt >= t0 - 0.1) & (mt <= t0 + N_BINS * BINSIZE + 0.1)
    ax = axes[2, 0]
    ax.plot(mt[m] - t0 + TIME_WINDOW[0], mv[m], color='0.6', label=f'raw {view} whisker ME')
    ax.plot(right_edges, res['me_raw'][k], 'o-', ms=3, label='interpolated to bins')
    for e in res['me_edges']:
        ax.axhline(e, color='g', ls='--', lw=1)
    ax.axvline(0, color='r', lw=2)
    ax.set(title=f'whisker motion energy, trial {k}',
           xlabel='time from stimulus onset (s)', ylabel='a.u.')
    ax.legend()

    ax = axes[2, 1]
    ax.step(right_edges, outputs[k, 3, :], where='mid')
    ax.axvline(0, color='r', lw=2)
    ax.set(title='discretised whisker motion energy (output 3)',
           xlabel='time from stimulus onset (s)', ylabel='class', yticks=[0, 1, 2])

    # (4) discretisation check: class vs value
    ax = axes[3, 0]
    ax.scatter(res['wheel_raw'].ravel()[::13], outputs[:, 2, :].ravel()[::13],
               s=1, alpha=0.2)
    for e in res['wheel_edges']:
        ax.axvline(e, color='g', ls='--')
    ax.set(xscale='symlog', title='wheel speed value -> class', xlabel='rad/s',
           ylabel='class', yticks=[0, 1, 2])
    ax = axes[3, 1]
    ax.scatter(res['me_raw'].ravel()[::13], outputs[:, 3, :].ravel()[::13], s=1, alpha=0.2)
    for e in res['me_edges']:
        ax.axvline(e, color='g', ls='--')
    ax.set(xscale='symlog', title='whisker ME value -> class', xlabel='a.u.',
           ylabel='class', yticks=[0, 1, 2])

    # (5) per-trial task variables and the block-structured input
    ax = axes[4, 0]
    ax.plot(outputs[:, 1, 0], '.-', label='prior class (0=0.2, 1=0.5, 2=0.8)')
    ax.plot(outputs[:, 0, 0] * 0.5 + 2.5, '.', ms=3, label='choice (2.5=left, 3.0=right)')
    ax.set(title='per-trial outputs across the session', xlabel='trial (kept)')
    ax.legend()

    ax = axes[4, 1]
    ax.plot(inputs[:, 1, 0], '.-')
    ax.set(title='input 1: trial number in block', xlabel='trial (kept)',
           ylabel='trials since block start')

    fig.tight_layout()
    path = os.path.join(out_dir, f'processing_{eid}.png')
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f'  wrote {path}')

    # --- alignment figure: spike raster around stimulus onset, several trials -------
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    ax = axes[0]
    ax.imshow(neural.mean(axis=1) / BINSIZE, aspect='auto', interpolation='nearest',
              extent=[TIME_WINDOW[0], TIME_WINDOW[1], n_trials, 0], cmap='viridis')
    ax.axvline(0, color='r', lw=2)
    ax.set(title='mean firing rate per trial x time', xlabel='time from stim onset (s)',
           ylabel='trial')
    ax = axes[1]
    ax.imshow(res['wheel_raw'], aspect='auto', interpolation='nearest',
              extent=[TIME_WINDOW[0], TIME_WINDOW[1], n_trials, 0], cmap='magma')
    ax.axvline(0, color='r', lw=2)
    ax.set(title='wheel speed per trial x time', xlabel='time from stim onset (s)')
    ax = axes[2]
    ax.imshow(res['me_raw'], aspect='auto', interpolation='nearest',
              extent=[TIME_WINDOW[0], TIME_WINDOW[1], n_trials, 0], cmap='magma')
    ax.axvline(0, color='r', lw=2)
    ax.set(title='whisker ME per trial x time', xlabel='time from stim onset (s)')
    fig.suptitle(f'{eid}: temporal alignment of neural, wheel and whisker streams')
    fig.tight_layout()
    path = os.path.join(out_dir, f'processing_{eid}_alignment.png')
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f'  wrote {path}')


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def build_jobs(sample):
    bwm = pd.read_csv(BWM_FREEZE, index_col=0)
    jobs = []
    for eid, g in bwm.groupby('eid', sort=True):
        probes = list(zip(g['pid'], g['probe_name']))
        jobs.append((eid, g['subject'].iloc[0], probes))
    jobs.sort(key=lambda j: j[0])
    if sample:
        jobs = jobs[:2]
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('outfile', type=str)
    ap.add_argument('--full', action='store_true', help='process all sessions (default)')
    ap.add_argument('--sample', action='store_true', help='process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true',
                    help='plot every processing step for up to 2 sessions')
    ap.add_argument('--n-workers', type=int, default=16)
    args = ap.parse_args()

    ensure_cache_tables()
    jobs = build_jobs(args.sample)
    print(f'{len(jobs)} sessions to process, {args.n_workers} workers')
    t_start = time.time()

    results = []
    if args.n_workers > 1 and len(jobs) > 1:
        from multiprocessing import Pool
        with Pool(args.n_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(convert_session, jobs)):
                results.append(res)
                done = i + 1
                el = time.time() - t_start
                print(f'[{done}/{len(jobs)}] {res["eid"]} '
                      f'{"SKIP: " + res["skip"] if "skip" in res else ""}'
                      f'{"" if "skip" in res else f"{res['n_trials_valid']} trials, {res['n_units_kept']} neurons"}'
                      f'  ({res["elapsed"]:.1f}s; elapsed {el:.0f}s, '
                      f'eta {el / done * (len(jobs) - done):.0f}s)', flush=True)
    else:
        for i, job in enumerate(jobs):
            res = convert_session(job)
            results.append(res)
            print(f'[{i + 1}/{len(jobs)}] {res["eid"]} '
                  f'{res.get("skip", "")} ({res["elapsed"]:.1f}s)', flush=True)

    results.sort(key=lambda r: r['eid'])
    ok = [r for r in results if 'skip' not in r]
    skipped = [r for r in results if 'skip' in r]
    print(f'\n{len(ok)} sessions converted, {len(skipped)} skipped')
    for r in skipped:
        print(f'  SKIP {r["eid"]}: {r["skip"]}')
        if 'traceback' in r:
            print(r['traceback'])

    if args.show_processing:
        one, _ = get_one()
        for r in ok[:2]:
            print(f'plotting {r["eid"]}')
            plot_processing(r, one)

    # --- assemble the target structure ---------------------------------------------
    t0 = time.time()
    subjects = sorted({r['subject'] for r in ok})
    sub2idx = {s: i for i, s in enumerate(subjects)}
    regions = sorted({reg for r in ok for reg in r['regions']})
    reg2idx = {reg: i for i, reg in enumerate(regions)}

    data = {
        'neural': [[r['neural'][k] for k in range(r['neural'].shape[0])] for r in ok],
        'input': [[r['input'][k] for k in range(r['input'].shape[0])] for r in ok],
        'output': [[r['output'][k] for k in range(r['output'].shape[0])] for r in ok],
        'subjects': subjects,
        'subject_idx': np.array([sub2idx[r['subject']] for r in ok], dtype=np.int64),
        'brain_regions': regions,
        'brain_region_idx': [np.array([reg2idx[x] for x in r['regions']], dtype=np.int64)
                             for r in ok],
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'dataset': 'IBL Brain-Wide Map (International Brain Laboratory et al., '
                       '"A brain-wide map of neural activity during complex behaviour")',
            'task_description':
                'Mice report the side of a visual stimulus by turning a wheel. Decoded '
                'from population spike counts: the choice (left/right), the block prior '
                'probability that the stimulus appears on the left (0.2/0.5/0.8), and two '
                'time-varying behaviours discretised into tertiles, wheel speed and '
                'whisker-pad motion energy.',
            'time_bin_size': BINSIZE * 1000.0,          # ms
            'temporal_alignment_event': 'visual stimulus onset (trials.stimOn_times)',
            'off_start': TIME_WINDOW[0],
            'off_end': TIME_WINDOW[1],
            'n_time_bins': N_BINS,
            'neural_units': 'spike counts per 20 ms bin',
            'neuron_selection': 'well-isolated units (clusters.label >= 1: amplitude > 50 uV, '
                                'noise cutoff < 20 uV, refractory-period violation passed), '
                                'in grey matter (Beryl acronym not root/void), pooled over '
                                'all probes of a session',
            'trial_selection': 'ibl_data_utils.load_trials_and_mask: no NaN in stimOn_times, '
                               'choice, feedback_times, probabilityLeft, firstMovement_times, '
                               'feedbackType; 0.08 s <= firstMovement_times - stimOn_times '
                               '<= 2 s; feedback_times - goCue_times <= 10 s; choice != 0; '
                               'plus trials whose wheel/whisker traces do not cover the window',
            'brain_region_mapping': 'Allen CCF acronyms mapped to the IBL Beryl atlas',
            'input_descriptions': [
                'time of the bin centre relative to stimulus onset, in seconds '
                '(-0.49 ... 1.49)',
                'number of trials since the start of the current probabilityLeft block '
                '(0-based, counted on the full trials table)',
            ],
            'output_descriptions': [
                'choice: wheel turn reported by the mouse; IBL choice +1 -> 0 (left), '
                '-1 -> 1 (right)',
                'prior: block probability that the stimulus appears on the left; '
                '0.2 -> 0, 0.5 -> 1, 0.8 -> 2',
                'wheel speed |velocity| interpolated to the bin grid then discretised into '
                'per-session tertiles (low/medium/high)',
                'whisker-pad motion energy (left camera, right camera as fallback) '
                'interpolated to the bin grid then discretised into per-session tertiles',
            ],
            'session_info': [
                {'eid': r['eid'], 'subject': r['subject'], 'n_probes': r['n_probes'],
                 'n_trials': int(r['neural'].shape[0]),
                 'n_neurons': int(r['neural'].shape[1]),
                 'n_units_all': r['n_units_all'], 'n_units_good': r['n_units_good'],
                 'n_trials_total': r['n_trials_total'],
                 'n_trials_pass_mask': r['n_trials_mask'],
                 'whisker_camera': r['whisker_camera'],
                 # row index of each kept trial in the session's full trials table, so
                 # every converted trial can be traced back to the source data.  A plain
                 # list of ints, not an ndarray, so that the whole metadata dict stays
                 # JSON-serialisable (train_decoder.py --stats-json dumps it).
                 'trial_idx': [int(x) for x in r['trial_idx']],
                 'wheel_tertile_edges': [float(x) for x in r['wheel_edges']],
                 'whisker_tertile_edges': [float(x) for x in r['me_edges']]}
                for r in ok
            ],
            'skipped_sessions': [{'eid': r['eid'], 'reason': r['skip']} for r in skipped],
            'reference_parameters': {
                'align_time': ALIGN_TIME, 'time_window': list(TIME_WINDOW),
                'binsize': BINSIZE, 'interval_len': 2.0,
                'source': 'code_zhang2025/src/0_data_caching.py',
            },
        },
    }
    print(f'assembled in {time.time() - t0:.1f}s')

    # --- summary --------------------------------------------------------------------
    ntr = np.array([len(s) for s in data['neural']])
    nneu = np.array([s[0].shape[0] for s in data['neural']])
    print(f"sessions {len(ok)}  subjects {len(subjects)}  regions {len(regions)}")
    print(f"trials total {ntr.sum()}  per session mean {ntr.mean():.1f} "
          f"min {ntr.min()} max {ntr.max()}")
    print(f"neurons total {nneu.sum()}  per session mean {nneu.mean():.1f} "
          f"min {nneu.min()} max {nneu.max()}")
    tot = lambda k: sum(r[k] for r in ok)
    print(f"trial accounting over the {len(ok)} converted sessions:")
    print(f"  trials in the trials tables      {tot('n_trials_total')}")
    print(f"  pass load_trials_and_mask        {tot('n_trials_mask')}")
    print(f"  dropped: wheel does not cover    {tot('n_trials_wheel_bad')}")
    print(f"  dropped: whisker does not cover  {tot('n_trials_me_bad')}")
    print(f"  dropped: outside spike recording {tot('n_trials_outside_recording')}")
    print(f"  kept                             {ntr.sum()}")
    print(f"units: {tot('n_units_all')} clusters, {tot('n_units_good')} well-isolated, "
          f"{nneu.sum()} well-isolated in grey matter")
    cams = pd.Series([r['whisker_camera'] for r in ok]).value_counts().to_dict()
    print(f"whisker camera used: {cams}")

    t0 = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_gb = os.path.getsize(args.outfile) / 1e9
    print(f'wrote {args.outfile} ({size_gb:.2f} GB) in {time.time() - t0:.1f}s')
    print(f'TOTAL {time.time() - t_start:.1f}s')


if __name__ == '__main__':
    main()
