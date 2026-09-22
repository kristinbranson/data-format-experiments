#!/usr/bin/env python3
"""
Convert the IBL Brain Wide Map (BWM) public dataset into the decoder-ready format.

Usage
-----
    python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Processing follows the reference pipeline of Zhang et al. (2026), "Exploiting correlations
across trials and behavioral sessions to improve neural decoding"
(`/app/code/code_zhang2025/src/0_data_caching.py` and `src/utils/ibl_data_utils.py`), and
the inclusion criteria of IBL et al. (2025), "A brain-wide map of neural activity during
complex behaviour".

Reference parameters reproduced here (from `0_data_caching.py`):

    params = {'interval_len': 2, 'binsize': 0.02, 'single_region': False,
              'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}

i.e. trials are aligned to stimulus onset and span -0.5 s .. +1.5 s in 100 non-overlapping
20 ms bins.

All data access goes through the ONE API (`one.api.ONE`) and the brainbox loaders
(`brainbox.io.one.SpikeSortingLoader`, `brainbox.io.one.SessionLoader`); no file under
/app/data is read directly.
"""

import argparse
import os
import pickle
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Reference code from the method paper; reused rather than re-implemented where possible.
REFCODE = '/app/code/code_zhang2025/src'
sys.path.insert(0, REFCODE)

from one.api import ONE                                     # noqa: E402
from brainbox.io.one import SessionLoader, SpikeSortingLoader  # noqa: E402
from iblatlas.regions import BrainRegions                    # noqa: E402
from scipy.interpolate import interp1d                       # noqa: E402

from utils.ibl_data_utils import load_trials_and_mask, merge_probes  # noqa: E402

# --------------------------------------------------------------------------------------
# Constants -- these mirror `params` in the reference caching script.
# --------------------------------------------------------------------------------------
ONE_BASE_URL = 'https://openalyx.internationalbrainlab.org'
ONE_CACHE_DIR = '/app/data/one_cache'
BWM_RELEASE_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'

ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)          # seconds relative to the alignment event
BINSIZE = 0.02                     # seconds
NBINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))   # -> 100
MAX_TRIAL_LEN = 10.0               # `load_trials_and_mask(max_trial_len=10.0)` in the ref
QC_LABEL = 1.0                     # cluster['label'] >= 1  <=> IBL "well-isolated neuron"
NON_GREY_MATTER = ('root', 'void')  # Beryl acronyms that are not grey matter
MIN_TRIALS = 2                     # the target format needs >= 2 trials per session
MIN_NEURONS = 5                    # data paper: ">= 5 well-isolated neurons per session"
MAX_SPIKE_GAP = 0.5                # s; a longer gap in the pooled spike train = no data

INPUT_NAMES = ['time_from_stimulus_onset', 'trial_number_in_block']
OUTPUT_NAMES = ['choice', 'prior_probability_left', 'wheel_speed', 'whisker_motion_energy']
OUTPUT_VALUES = [
    ['left', 'right'],
    ['p(left)=0.2', 'p(left)=0.5', 'p(left)=0.8'],
    ['low', 'medium', 'high'],
    ['low', 'medium', 'high'],
]

# Bin centres of the neural bins, relative to stimulus onset (seconds).
BIN_CENTRES = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 0.5)
# Behaviour is sampled at the *right edge* of each bin, exactly as the reference
# `get_behavior_per_interval` does (`np.linspace(beg + binsize, end, n_bins)`).
BIN_RIGHT_EDGES = TIME_WINDOW[0] + BINSIZE * (np.arange(NBINS) + 1)


# --------------------------------------------------------------------------------------
# ONE access
# --------------------------------------------------------------------------------------
_ONE = None
_BRAIN_REGIONS = None


def get_one():
    """Build (once per process) a ONE client against the staged local cache.

    The default (remote) mode is used deliberately: it resolves datasets through the
    cached Alyx REST responses in `<cache>/.rest`, which know about the dataset revisions
    that are actually staged on disk (e.g. `alf/#2025-03-03#/_ibl_trials.table.pqt`).
    The copied `Brainwidemap` release table predates those revisions, so `mode='local'`
    silently fails to find the trials table.
    """
    global _ONE
    if _ONE is None:
        _ONE = ONE(base_url=ONE_BASE_URL, silent=True, cache_dir=ONE_CACHE_DIR)
    return _ONE


def get_brain_regions():
    global _BRAIN_REGIONS
    if _BRAIN_REGIONS is None:
        _BRAIN_REGIONS = BrainRegions()
    return _BRAIN_REGIONS


# --------------------------------------------------------------------------------------
# Neural data
# --------------------------------------------------------------------------------------
def load_session_spikes(one, eid, pids, pnames):
    """Load and merge the spike sorting of every probe of a session.

    Mirrors `ibl_data_utils.prepare_data`: each probe is loaded with
    `SpikeSortingLoader.load_spike_sorting` + `merge_clusters`, then probes are merged with
    the reference `merge_probes` helper (neurons of one session are pooled across probes
    because the probes are not statistically independent -- see the data paper,
    "Overview of decoding").

    The reference additionally queries `raw_electrophysiology(band='ap', stream=True).fs`
    to record the AP sampling rate. That streams raw ephys over the network and is not
    used by any downstream computation, so it is skipped.
    """
    spikes_list, clusters_list = [], []
    for pid, pname in zip(pids, pnames):
        ssl = SpikeSortingLoader(pid=str(pid), one=one, eid=eid, pname=pname)
        sp, cl, ch = ssl.load_spike_sorting()
        if sp is None or 'times' not in sp or len(sp['times']) == 0:
            continue
        if cl is None or len(cl.get('channels', [])) == 0:
            continue
        cld = SpikeSortingLoader.merge_clusters(sp, cl, ch).to_df()
        cld['pid'] = str(pid)
        spikes_list.append(sp)
        clusters_list.append(cld)
    if not spikes_list:
        raise RuntimeError('no spike sorting could be loaded for this session')
    return merge_probes(spikes_list, clusters_list)


def select_neurons(clusters, brain_regions):
    """Apply the data paper's neuron and region inclusion criteria.

    * `label >= 1` keeps only "well-isolated neurons": in `ibllib`, `label` is the fraction
      of the three RIGOR single-unit metrics a cluster passes (amplitude > 50 uV,
      noise cut-off < 20 uV, refractory-period violation), so `label == 1` is exactly the
      data paper's criterion.
    * Beryl acronyms `root` / `void` are dropped, implementing the data paper's restriction
      to regions "designated grey matter in the adult mouse Allen CCF". The reference code
      makes the same exclusion in `MultiRegionDataModule.list_regions`.

    Returns (keep_idx, beryl_acronyms_of_kept_neurons).
    """
    beryl = np.asarray(brain_regions.acronym2acronym(
        clusters['acronym'].to_numpy(), mapping='Beryl'))
    good = clusters['label'].to_numpy() >= QC_LABEL
    grey = ~np.isin(beryl, NON_GREY_MATTER)
    keep = np.nonzero(good & grey)[0]
    return keep, beryl[keep]


def bin_spiking_data_fast(spike_times, spike_clusters, n_clusters, align_times):
    """Bin spikes into (n_trials, n_clusters, NBINS) counts.

    Vectorised equivalent of the reference `bin_spiking_data` /
    `get_spike_data_per_interval`, which calls `iblutil.numerical.bincount2D` once per
    trial inside a multiprocessing pool. `bincount2D` assigns a sample to bin
    `floor((t - t_beg) / binsize)` over `[t_beg, t_end]` and the reference then truncates
    to the first `NBINS` columns, so a spike landing exactly on `t_end` is dropped. The
    same convention is reproduced here (spikes are taken from `[t_beg, t_end)` and bin
    indices outside `[0, NBINS)` are discarded).

    Args:
        spike_times: (n_spikes,) spike times, sorted ascending.
        spike_clusters: (n_spikes,) cluster index in `[0, n_clusters)`.
        n_clusters: number of neurons.
        align_times: (n_trials,) alignment event times.

    Returns:
        (n_trials, n_clusters, NBINS) uint8 array of spike counts.
    """
    beg = align_times + TIME_WINDOW[0]
    end = align_times + TIME_WINDOW[1]
    out = np.zeros((len(align_times), n_clusters, NBINS), dtype=np.uint8)
    finite = np.isfinite(beg) & np.isfinite(end)
    i0 = np.searchsorted(spike_times, np.where(finite, beg, np.inf), side='left')
    i1 = np.searchsorted(spike_times, np.where(finite, end, np.inf), side='left')
    for k in range(len(align_times)):
        if not finite[k]:
            continue
        t = spike_times[i0[k]:i1[k]]
        c = spike_clusters[i0[k]:i1[k]]
        b = np.floor((t - beg[k]) / BINSIZE).astype(np.int64)
        ok = (b >= 0) & (b < NBINS)
        counts = np.bincount(c[ok] * NBINS + b[ok], minlength=n_clusters * NBINS)
        np.clip(counts, 0, 255, out=counts)
        out[k] = counts.reshape(n_clusters, NBINS).astype(np.uint8)
    return out


def neural_data_available(all_spike_times, align_times):
    """Mask of trials whose 2 s window is covered by valid electrophysiology.

    Neither the data paper nor the reference code names a criterion for this, but real
    sessions contain periods with no neural data at all: some recordings stop while the
    behavioural rig keeps running (e.g. eid 8c2f7f4d…: last spike at 1779 s, three more
    trials afterwards), and some spike trains contain multi-second dropouts (e.g. eid
    b182b754…: a 5 s hole at 185-190 s). Those trials are all-zero and carry no
    information, and `verify_data_format` flags them.

    Detection uses the *pooled, unfiltered* spike train of the session (all Kilosort
    clusters of all probes, ~900 per probe firing at a combined rate of thousands of
    spikes per second), so any inter-spike interval longer than `MAX_SPIKE_GAP` = 0.5 s
    is unambiguously missing data rather than silence. A trial is rejected if its window
    intersects such a gap, or starts before the first / ends after the last spike.

    Args:
        all_spike_times: (n,) pooled spike times of every cluster, sorted ascending.
        align_times: (n_trials,) alignment event times.

    Returns:
        (n_trials,) bool mask, True where neural data covers the whole window.
    """
    beg = align_times + TIME_WINDOW[0]
    end = align_times + TIME_WINDOW[1]
    ok = np.isfinite(beg) & np.isfinite(end)
    if len(all_spike_times) == 0:
        return np.zeros(len(align_times), dtype=bool)

    # no-data intervals: before the first spike, after the last, and every long ISI
    dt = np.diff(all_spike_times)
    idx = np.nonzero(dt > MAX_SPIKE_GAP)[0]
    gap_start = np.concatenate([[-np.inf], all_spike_times[idx], [all_spike_times[-1]]])
    gap_end = np.concatenate([[all_spike_times[0]], all_spike_times[idx + 1], [np.inf]])

    # a trial is bad if [beg, end) overlaps any [gap_start, gap_end]
    safe_beg = np.where(ok, beg, 0.0)
    safe_end = np.where(ok, end, 0.0)
    overlaps = (safe_beg[:, None] < gap_end[None, :]) & (safe_end[:, None] > gap_start[None, :])
    return ok & ~overlaps.any(axis=1)


# --------------------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------------------
def load_dynamic_behaviour(sess_loader):
    """Load the two time-varying behaviours the decoder has to predict.

    Reproduces `ibl_data_utils.load_target_behavior`:
      * wheel speed  = |velocity| of the uniformly resampled (~1 kHz) wheel trace,
      * whisker motion energy = `whiskerMotionEnergy` of the **left** camera (60 Hz),
        falling back to the right camera (150 Hz) when the left one is unavailable --
        exactly the fallback in `ibl_data_utils.bin_behaviors`.

    Returns {'wheel-speed': (times, values), 'whisker-motion-energy': (times, values)},
    plus the camera actually used.
    """
    out = {}
    sess_loader.load_wheel()
    out['wheel-speed'] = (sess_loader.wheel['times'].to_numpy(),
                          np.abs(sess_loader.wheel['velocity'].to_numpy()))

    camera_used = None
    for view, key in (('left', 'leftCamera'), ('right', 'rightCamera')):
        try:
            sess_loader.load_motion_energy(views=[view])
            me = sess_loader.motion_energy[key]
            times = me['times'].to_numpy()
            vals = me['whiskerMotionEnergy'].to_numpy()
            if len(times) == 0 or np.all(np.isnan(vals)):
                continue
            out['whisker-motion-energy'] = (times, vals)
            camera_used = view
            break
        except Exception:
            continue
    if camera_used is None:
        raise RuntimeError('no whisker motion energy available')
    return out, camera_used


def bin_behaviour_per_trial(target_times, target_vals, align_times):
    """Resample a behavioural trace onto the 100 trial bins, per trial.

    Faithful re-implementation of `ibl_data_utils.get_behavior_per_interval`:
      * the trace is sliced to the open interval (t_beg, t_end),
      * a trial is rejected when there is no data, when the trace starts more than one bin
        after `t_beg` or ends more than one bin before `t_end`, when the interval time is
        NaN, or when the slice contains NaNs,
      * the remaining trials are linearly interpolated (with linear extrapolation at the
        edges) onto `np.linspace(t_beg + binsize, t_end, n_bins)`, i.e. the right edge of
        each bin.

    The only deviation is that NaNs always reject a trial (the reference's
    `allow_nans=False` branch). The reference caching script passes `allow_nans=True` and
    lets the model impute NaNs with the trial average at fit time; here the converted
    outputs must be valid class labels at every timepoint, and `verify_data_format`
    rejects NaNs outright, so NaN-containing trials are dropped instead.

    Returns (values (n_trials, NBINS) float64 with NaN rows for rejected trials,
             good mask (n_trials,) bool).
    """
    n = len(align_times)
    vals = np.full((n, NBINS), np.nan)
    good = np.zeros(n, dtype=bool)

    beg = align_times + TIME_WINDOW[0]
    end = align_times + TIME_WINDOW[1]
    finite = np.isfinite(beg) & np.isfinite(end)
    if not np.any(finite):
        return vals, good

    idx_beg = np.searchsorted(target_times, np.where(finite, beg, np.inf), side='right')
    idx_end = np.searchsorted(target_times, np.where(finite, end, np.inf), side='left')

    for k in range(n):
        if not finite[k]:
            continue
        tt = target_times[idx_beg[k]:idx_end[k]]
        tv = target_vals[idx_beg[k]:idx_end[k]]
        if len(tv) == 0:
            continue
        if np.isnan(tv).any():
            continue
        if abs(beg[k] - tt[0]) > BINSIZE:      # target data starts too late
            continue
        if abs(end[k] - tt[-1]) > BINSIZE:     # target data ends too early
            continue
        x = np.linspace(beg[k] + BINSIZE, end[k], NBINS)
        vals[k] = interp1d(tt, tv, kind='linear', fill_value='extrapolate')(x)
        good[k] = True
    return vals, good


def discretize_tertiles(values):
    """Discretize a continuous time-varying behaviour into 3 equal-occupancy bins.

    Thresholds are the 33.3rd and 66.7th percentiles of every retained (trial, time-bin)
    sample of **that session**. Per-session thresholds are used because both signals are in
    session-specific units: whisker motion energy is an uncalibrated pixel-intensity
    difference whose scale depends on the camera (left 1280x1024 @60 Hz vs right 640x512
    @150 Hz), on illumination and on ROI placement, and wheel-speed statistics depend on how
    vigorously a given mouse turns the wheel. A single global threshold would map whole
    sessions into one class; per-session tertiles give ~1/3 of samples per class in every
    session, so chance is a well-defined 1/3 and the shared decoder sees a comparable
    target in every session.

    Returns (labels (int) in {0,1,2}, thresholds (2,)).
    """
    finite = values[np.isfinite(values)]
    thresholds = np.quantile(finite, [1. / 3., 2. / 3.])
    labels = np.searchsorted(thresholds, values, side='right').astype(np.int64)
    return labels, thresholds


def trial_number_in_block(probability_left):
    """0-based index of each trial within its block of constant `probabilityLeft`.

    Computed on the *complete* trials table before any trial is excluded, so that the
    numbering reflects the block structure the mouse actually experienced (the IBL task has
    an initial 90-trial unbiased block at pLeft = 0.5, then blocks of 20-100 trials
    alternating between pLeft = 0.8 and 0.2).
    """
    p = np.asarray(probability_left, dtype=float)
    new_block = np.ones(len(p), dtype=bool)
    if len(p) > 1:
        # NaN != NaN, so a NaN pLeft always starts a new block; those trials are excluded
        # later by the trials mask anyway.
        new_block[1:] = ~(p[1:] == p[:-1])
    block_id = np.cumsum(new_block) - 1
    # index within block = running count restarting at each block boundary
    starts = np.nonzero(new_block)[0]
    within = np.arange(len(p)) - starts[block_id]
    return within.astype(np.float32), block_id


# --------------------------------------------------------------------------------------
# Per-session conversion
# --------------------------------------------------------------------------------------
def convert_session(eid, verbose=True, collect_debug=False):
    """Convert one session. Returns a dict, or raises on failure."""
    timing = {}
    one = get_one()
    brain_regions = get_brain_regions()

    t = time.time()
    pids, pnames = one.eid2pid(eid)
    timing['eid2pid'] = time.time() - t

    # ---- neural -----------------------------------------------------------------
    t = time.time()
    spikes, clusters = load_session_spikes(one, eid, pids, pnames)
    timing['load_spikes'] = time.time() - t
    n_clusters_all = len(clusters)

    t = time.time()
    keep_idx, beryl_keep = select_neurons(clusters, brain_regions)
    n_neurons = len(keep_idx)
    if n_neurons < MIN_NEURONS:
        raise RuntimeError(f'only {n_neurons} well-isolated grey-matter neurons '
                           f'(< {MIN_NEURONS})')
    all_spike_times = spikes['times']       # pooled, unfiltered: used for the data-gap test
    remap = np.full(n_clusters_all, -1, dtype=np.int64)
    remap[keep_idx] = np.arange(n_neurons)
    sc = remap[spikes['clusters']]
    sel = sc >= 0
    spike_times = np.ascontiguousarray(spikes['times'][sel])
    spike_clusters = np.ascontiguousarray(sc[sel])
    timing['select_neurons'] = time.time() - t

    # ---- trials -----------------------------------------------------------------
    t = time.time()
    sess_loader = SessionLoader(one=one, eid=eid)
    # `load_trials_and_mask` builds the data paper's trial exclusion mask. Note the
    # reference calls `SessionLoader(one, eid)` positionally, which ibllib 4.0.1 no longer
    # accepts, hence passing an explicit `sess_loader`.
    trials, mask = load_trials_and_mask(
        one=one, eid=eid, max_trial_len=MAX_TRIAL_LEN, sess_loader=sess_loader)
    mask = np.asarray(mask, dtype=bool)
    timing['load_trials'] = time.time() - t
    n_trials_all = len(trials)

    align_times = trials[ALIGN_TIME].to_numpy()
    in_block, _ = trial_number_in_block(trials['probabilityLeft'].to_numpy())

    # ---- behaviour --------------------------------------------------------------
    t = time.time()
    beh_traces, camera_used = load_dynamic_behaviour(sess_loader)
    timing['load_behaviour'] = time.time() - t

    t = time.time()
    wheel_vals, wheel_ok = bin_behaviour_per_trial(*beh_traces['wheel-speed'], align_times)
    me_vals, me_ok = bin_behaviour_per_trial(*beh_traces['whisker-motion-energy'], align_times)
    timing['bin_behaviour'] = time.time() - t

    # ---- trial selection ---------------------------------------------------------
    # Mirrors `align_spike_behavior`: keep trials that pass the trials mask AND for which
    # every behavioural trace could be extracted; additionally require that the neural
    # recording actually covers the trial window.
    has_neural = neural_data_available(all_spike_times, align_times)
    keep_trials = mask & wheel_ok & me_ok & has_neural
    n_keep = int(keep_trials.sum())
    if n_keep < MIN_TRIALS:
        raise RuntimeError(f'only {n_keep} usable trials')

    # ---- bin spikes (only for the trials we keep) --------------------------------
    t = time.time()
    binned = bin_spiking_data_fast(spike_times, spike_clusters, n_neurons,
                                   align_times[keep_trials])
    timing['bin_spikes'] = time.time() - t

    # ---- inputs ------------------------------------------------------------------
    tr = trials[keep_trials]
    inp = np.empty((n_keep, 2, NBINS), dtype=np.float32)
    inp[:, 0, :] = BIN_CENTRES.astype(np.float32)
    inp[:, 1, :] = in_block[keep_trials][:, None]

    # ---- outputs -----------------------------------------------------------------
    # choice: IBL codes +1 / -1. Verified empirically (see CONVERSION_NOTES Step 4):
    # on correct trials with the stimulus on the left, choice == +1; on correct trials with
    # the stimulus on the right, choice == -1. So +1 is a leftward choice.
    choice = tr['choice'].to_numpy()
    choice_out = (choice < 0).astype(np.int64)          # left -> 0, right -> 1

    pleft = tr['probabilityLeft'].to_numpy()
    prior_out = np.full(n_keep, -1, dtype=np.int64)
    for value, code in ((0.2, 0), (0.5, 1), (0.8, 2)):
        prior_out[np.isclose(pleft, value)] = code
    if np.any(prior_out < 0):
        bad = np.unique(pleft[prior_out < 0])
        raise RuntimeError(f'unexpected probabilityLeft values {bad}')

    wheel_kept = wheel_vals[keep_trials]
    me_kept = me_vals[keep_trials]
    wheel_lab, wheel_thr = discretize_tertiles(wheel_kept)
    me_lab, me_thr = discretize_tertiles(me_kept)

    out = np.empty((n_keep, 4, NBINS), dtype=np.int64)
    out[:, 0, :] = choice_out[:, None]
    out[:, 1, :] = prior_out[:, None]
    out[:, 2, :] = wheel_lab
    out[:, 3, :] = me_lab

    result = dict(
        eid=str(eid),
        neural=binned,                      # uint8 (n_keep, n_neurons, NBINS)
        input=inp,
        output=out,
        beryl=beryl_keep,
        n_clusters_all=int(n_clusters_all),
        n_good_all=int((clusters['label'].to_numpy() >= QC_LABEL).sum()),
        n_neurons=int(n_neurons),
        n_probes=len(pids),
        n_trials_all=int(n_trials_all),
        n_trials_mask=int(mask.sum()),
        n_trials_no_neural=int((mask & ~has_neural).sum()),
        n_trials_no_behaviour=int((mask & has_neural & ~(wheel_ok & me_ok)).sum()),
        n_trials_kept=n_keep,
        camera_used=camera_used,
        wheel_thresholds=wheel_thr,
        me_thresholds=me_thr,
        frac_correct=float((trials['feedbackType'].to_numpy() == 1).mean()),
        timing=timing,
    )
    if collect_debug:
        result['debug'] = dict(
            trials=trials, mask=mask, keep_trials=keep_trials,
            align_times=align_times, in_block=in_block,
            wheel_trace=beh_traces['wheel-speed'], me_trace=beh_traces['whisker-motion-energy'],
            wheel_vals=wheel_kept, me_vals=me_kept,
            spike_times=spike_times, spike_clusters=spike_clusters,
        )
    if verbose:
        print(f'  [{eid}] probes={len(pids)} clusters={n_clusters_all} '
              f'good={result["n_good_all"]} kept_neurons={n_neurons} '
              f'trials={n_trials_all}->mask {int(mask.sum())}'
              f'->(-{int((mask & ~has_neural).sum())} no-ephys, '
              f'-{int((mask & has_neural & ~(wheel_ok & me_ok)).sum())} no-behaviour)'
              f'->kept {n_keep} '
              f'camera={camera_used} '
              f'({sum(timing.values()):.1f}s)', flush=True)
    return result


def _worker(eid):
    try:
        return convert_session(eid, verbose=True)
    except Exception as exc:       # noqa: BLE001 -- one bad session must not kill the run
        return {'eid': str(eid), 'error': f'{type(exc).__name__}: {exc}',
                'traceback': traceback.format_exc()}


# --------------------------------------------------------------------------------------
# Diagnostic plots
# --------------------------------------------------------------------------------------
def plot_processing(res, outpath):
    """Plot every processing step for one session, to verify alignment/discretization."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dbg = res['debug']
    trials, keep = dbg['trials'], dbg['keep_trials']
    align = dbg['align_times'][keep]
    neural = res['neural'].astype(np.float32)
    inp, out = res['input'], res['output']
    n_keep = neural.shape[0]
    tr_edges = BIN_RIGHT_EDGES
    tr_ctr = BIN_CENTRES

    fig = plt.figure(figsize=(26, 30))
    gs = fig.add_gridspec(7, 3, hspace=0.45, wspace=0.22)

    # --- 1. population PSTH: must show a stimulus-locked transient at t = 0 -----------
    ax = fig.add_subplot(gs[0, :])
    psth = neural.mean(axis=(0, 1)) / BINSIZE
    ax.plot(tr_ctr, psth, lw=2)
    ax.axvline(0, color='r', ls='--', label='stimulus onset')
    ax.set_xlabel('time from stimulus onset (s)')
    ax.set_ylabel('mean firing rate (sp/s)')
    ax.set_title(f'{res["eid"]}: population PSTH aligned to {ALIGN_TIME} '
                 f'({res["n_neurons"]} neurons, {n_keep} trials)')
    ax.legend()

    # --- 2. single-trial raster reconstructed from raw spike times vs binned matrix ----
    k = min(3, n_keep - 1)
    t0 = align[k]
    st, sc = dbg['spike_times'], dbg['spike_clusters']
    sel = (st >= t0 + TIME_WINDOW[0]) & (st < t0 + TIME_WINDOW[1])
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(st[sel] - t0, sc[sel], '|', ms=2, color='k')
    ax.axvline(0, color='r', ls='--')
    ax.set_xlim(TIME_WINDOW)
    ax.set_title(f'raw spike raster, trial {k}')
    ax.set_xlabel('time from stim onset (s)'); ax.set_ylabel('neuron')

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(neural[k], aspect='auto', interpolation='nearest', origin='lower',
              extent=[TIME_WINDOW[0], TIME_WINDOW[1], 0, res['n_neurons']])
    ax.axvline(0, color='r', ls='--')
    ax.set_title(f'binned spike counts, trial {k} (20 ms bins)')
    ax.set_xlabel('time from stim onset (s)'); ax.set_ylabel('neuron')

    ax = fig.add_subplot(gs[1, 2])
    # per-bin totals from the two routes must agree exactly
    hist = np.zeros(NBINS)
    b = np.floor((st[sel] - (t0 + TIME_WINDOW[0])) / BINSIZE).astype(int)
    b = b[(b >= 0) & (b < NBINS)]
    np.add.at(hist, b, 1)
    ax.step(tr_ctr, hist, where='mid', label='from raw spike times')
    ax.step(tr_ctr, neural[k].sum(axis=0), where='mid', ls='--', label='from binned matrix')
    ax.legend(); ax.set_title('per-bin spike totals agree')
    ax.set_xlabel('time from stim onset (s)')

    # --- 3. wheel speed: raw trace, resampled values, tertile thresholds, labels ------
    wt, wv = dbg['wheel_trace']
    ax = fig.add_subplot(gs[2, :])
    win = (wt >= t0 - 0.7) & (wt <= t0 + 1.7)
    ax.plot(wt[win] - t0, wv[win], color='0.6', lw=1, label='raw |wheel velocity| (~1 kHz)')
    ax.plot(tr_edges, dbg['wheel_vals'][k], 'o-', ms=3, color='C0',
            label='resampled onto bin right edges')
    for thr in res['wheel_thresholds']:
        ax.axhline(thr, color='C3', ls=':', lw=1)
    ax.axvline(0, color='r', ls='--')
    ax.set_yscale('log')
    ax.set_title(f'wheel speed, trial {k} (dotted red = session tertile thresholds '
                 f'{np.round(res["wheel_thresholds"], 4)})')
    ax.set_xlabel('time from stim onset (s)'); ax.set_ylabel('rad/s'); ax.legend()

    ax = fig.add_subplot(gs[3, 0])
    ax.step(tr_edges, out[k, 2], where='post', color='C0')
    ax.set_title('discretized wheel speed (trial %d)' % k)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(OUTPUT_VALUES[2])
    ax.set_xlabel('time from stim onset (s)')

    # --- 4. whisker motion energy ----------------------------------------------------
    mt, mv = dbg['me_trace']
    ax = fig.add_subplot(gs[3, 1:])
    win = (mt >= t0 - 0.7) & (mt <= t0 + 1.7)
    ax.plot(mt[win] - t0, mv[win], color='0.6', lw=1, label='raw whisker ME (camera)')
    ax.plot(tr_edges, dbg['me_vals'][k], 'o-', ms=3, color='C1', label='resampled')
    for thr in res['me_thresholds']:
        ax.axhline(thr, color='C3', ls=':', lw=1)
    ax.axvline(0, color='r', ls='--')
    ax.set_title(f'whisker motion energy ({res["camera_used"]} camera), trial {k}')
    ax.set_xlabel('time from stim onset (s)'); ax.legend()

    ax = fig.add_subplot(gs[4, 0])
    ax.step(tr_edges, out[k, 3], where='post', color='C1')
    ax.set_title('discretized whisker motion energy')
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(OUTPUT_VALUES[3])
    ax.set_xlabel('time from stim onset (s)')

    # --- 5. class balance of the discretized outputs ---------------------------------
    ax = fig.add_subplot(gs[4, 1])
    width = 0.35
    for j, (od, name) in enumerate(((2, 'wheel speed'), (3, 'whisker ME'))):
        frac = [np.mean(out[:, od, :] == c) for c in range(3)]
        ax.bar(np.arange(3) + j * width, frac, width, label=name)
    ax.axhline(1 / 3, color='k', ls=':')
    ax.set_xticks(np.arange(3) + width / 2); ax.set_xticklabels(['low', 'medium', 'high'])
    ax.set_ylabel('fraction of timepoints'); ax.legend()
    ax.set_title('tertile discretization is balanced')

    # --- 6. inputs -------------------------------------------------------------------
    ax = fig.add_subplot(gs[4, 2])
    ax.plot(tr_ctr, inp[k, 0], label=INPUT_NAMES[0])
    ax.plot(tr_ctr, inp[k, 1], label=INPUT_NAMES[1])
    ax.axvline(0, color='r', ls='--'); ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('time from stim onset (s)'); ax.legend()
    ax.set_title('decoder inputs, trial %d' % k)

    # --- 7. block structure and per-trial outputs over the session --------------------
    ax = fig.add_subplot(gs[5, :])
    ax.plot(np.arange(len(trials)), trials['probabilityLeft'].to_numpy(), '.-', ms=3,
            color='0.6', label='probabilityLeft (all trials)')
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(trials)), dbg['in_block'], color='C2', lw=1,
             label='trial number in block')
    ax.set_xlabel('trial index in session'); ax.set_ylabel('pLeft')
    ax2.set_ylabel('trial number in block')
    ax.set_title('block structure: trial-in-block counter resets exactly at pLeft changes')
    ax.legend(loc='upper left'); ax2.legend(loc='upper right')

    ax = fig.add_subplot(gs[6, 0])
    kept_idx = np.nonzero(keep)[0]
    ax.plot(kept_idx, out[:, 1, 0], '.', ms=4, label='prior class')
    ax.plot(np.arange(len(trials)),
            np.select([np.isclose(trials['probabilityLeft'], v) for v in (0.2, 0.5, 0.8)],
                      [0, 1, 2], default=np.nan), '-', lw=0.8, color='0.7',
            label='from trials table')
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(OUTPUT_VALUES[1])
    ax.set_xlabel('trial index'); ax.legend(); ax.set_title('prior output vs trials table')

    ax = fig.add_subplot(gs[6, 1])
    ch = trials['choice'].to_numpy()
    ax.plot(kept_idx, out[:, 0, 0] + 0.02, '.', ms=4, label='converted (0=left,1=right)')
    ax.plot(np.arange(len(trials)), (ch < 0).astype(float), '-', lw=0.8, color='0.7',
            label='(choice<0) from trials table')
    ax.set_yticks([0, 1]); ax.set_yticklabels(OUTPUT_VALUES[0])
    ax.set_xlabel('trial index'); ax.legend(); ax.set_title('choice output vs trials table')

    ax = fig.add_subplot(gs[6, 2])
    # trial-averaged wheel speed split by choice: leftward/rightward turns differ in sign
    # of velocity but not speed, so instead show it split by reaction time tercile
    rt = (trials['firstMovement_times'] - trials['stimOn_times']).to_numpy()[keep]
    order = np.argsort(rt)
    ax.imshow(dbg['wheel_vals'][order], aspect='auto', origin='lower',
              extent=[TIME_WINDOW[0], TIME_WINDOW[1], 0, n_keep],
              vmax=np.nanpercentile(dbg['wheel_vals'], 99))
    ax.plot(rt[order], np.arange(n_keep), 'r-', lw=1, label='first movement time')
    ax.set_xlim(TIME_WINDOW); ax.legend()
    ax.set_title('wheel speed sorted by RT tracks first movement time')
    ax.set_xlabel('time from stim onset (s)'); ax.set_ylabel('trial (sorted by RT)')

    fig.savefig(outpath, dpi=90, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {outpath}', flush=True)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile', type=str, help='output pickle path')
    ap.add_argument('--full', action='store_true', default=True, help='process all sessions')
    ap.add_argument('--sample', action='store_true', help='process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true',
                    help='write processing_<eid>.png for up to 2 sessions')
    ap.add_argument('--n-workers', type=int, default=24)
    args = ap.parse_args()

    t_start = time.time()

    bwm_df = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
    eids = list(pd.unique(bwm_df.eid))
    subject_of_eid = bwm_df.drop_duplicates('eid').set_index('eid')['subject'].to_dict()
    lab_of_eid = bwm_df.drop_duplicates('eid').set_index('eid')['lab'].to_dict()
    date_of_eid = bwm_df.drop_duplicates('eid').set_index('eid')['date'].to_dict()

    if args.sample:
        eids = eids[:2]
    print(f'Converting {len(eids)} sessions ({bwm_df.pid.nunique()} probe insertions, '
          f'{bwm_df.subject.nunique()} subjects) from the BWM release freeze file.',
          flush=True)

    results = []
    failures = []

    if args.show_processing:
        # Serial, with debug payloads, for the first <=2 sessions.
        for eid in eids[:2]:
            try:
                res = convert_session(eid, collect_debug=True)
            except Exception as exc:     # noqa: BLE001
                print(f'  [{eid}] FAILED: {exc}', flush=True)
                failures.append((eid, str(exc)))
                continue
            plot_processing(res, f'/app/processing_{eid}.png')
            del res['debug']
            results.append(res)
        remaining = eids[2:]
    else:
        remaining = eids

    if remaining:
        n_workers = max(1, min(args.n_workers, len(remaining)))
        print(f'Processing {len(remaining)} sessions with {n_workers} workers...', flush=True)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_worker, eid): eid for eid in remaining}
            done = 0
            for fut in as_completed(futures):
                res = fut.result()
                done += 1
                if 'error' in res:
                    print(f'  [{res["eid"]}] FAILED: {res["error"]}', flush=True)
                    failures.append((res['eid'], res['error']))
                else:
                    results.append(res)
                if done % 25 == 0:
                    el = time.time() - t_start
                    print(f'  ... {done}/{len(remaining)} sessions, {el:.0f}s elapsed, '
                          f'{el / done * (len(remaining) - done):.0f}s remaining',
                          flush=True)

    # deterministic session order (release-file order)
    order = {e: i for i, e in enumerate(eids)}
    results.sort(key=lambda r: order[r['eid']])

    print(f'\n{len(results)} sessions converted, {len(failures)} failed.', flush=True)
    for eid, err in failures:
        print(f'  FAILED {eid}: {err}')

    # ---- assemble the target structure ------------------------------------------
    t = time.time()
    all_regions = sorted({r for res in results for r in np.unique(res['beryl'])})
    region_index = {r: i for i, r in enumerate(all_regions)}
    subjects = sorted({subject_of_eid[res['eid']] for res in results})
    subject_index = {s: i for i, s in enumerate(subjects)}

    data = {
        'neural': [], 'input': [], 'output': [],
        'subjects': subjects,
        'subject_idx': np.array([subject_index[subject_of_eid[r['eid']]] for r in results],
                                dtype=np.int64),
        'brain_regions': all_regions,
        'brain_region_idx': [np.array([region_index[x] for x in r['beryl']], dtype=np.int64)
                             for r in results],
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
    }

    for res in results:
        neural = res['neural']
        data['neural'].append([np.ascontiguousarray(neural[k], dtype=np.float32)
                               for k in range(neural.shape[0])])
        data['input'].append([np.ascontiguousarray(res['input'][k]) for k in range(len(neural))])
        data['output'].append([np.ascontiguousarray(res['output'][k]) for k in range(len(neural))])

    ntrials = [len(x) for x in data['neural']]
    nneurons = [r['n_neurons'] for r in results]
    data['metadata'] = {
        'dataset': 'International Brain Laboratory Brain Wide Map (public release, 2023)',
        'task_description': (
            'Mice perform the IBL visual decision-making task: a Gabor patch of one of five '
            'contrasts (0, 6.25, 12.5, 25, 100%) appears 35 degrees to the left or right and '
            'the mouse turns a wheel to bring it to the centre. After an initial 90-trial '
            'unbiased block (p(left)=0.5) the stimulus side is drawn in blocks of 20-100 '
            'trials with p(left)=0.8 or 0.2. Neural activity (Neuropixels spike counts, '
            'well-isolated grey-matter neurons pooled across the probes of a session) is used '
            'to decode: (0) the mouse\'s binary choice (left/right), (1) the block prior '
            'probability that the stimulus is on the left (0.2/0.5/0.8), (2) wheel speed '
            'discretized into 3 per-session tertile bins, and (3) whisker-pad motion energy '
            'discretized into 3 per-session tertile bins. Decoder inputs are the signed time '
            'from stimulus onset and the trial number within the current block.'),
        'time_bin_size': BINSIZE * 1000.0,
        'temporal_alignment_event': 'visual stimulus onset (trials.stimOn_times)',
        'off_start': TIME_WINDOW[0],
        'off_end': TIME_WINDOW[1],
        'n_timepoints': NBINS,
        'neural_units': 'spike counts per 20 ms bin',
        'input_units': ['seconds relative to stimulus onset (bin centre)',
                        'trials since the start of the current pLeft block (0-based)'],
        'output_discretization': (
            'choice: trials.choice==+1 -> 0 (left), ==-1 -> 1 (right). '
            'prior: trials.probabilityLeft 0.2->0, 0.5->1, 0.8->2. '
            'wheel speed and whisker motion energy: per-session tertiles (33.3/66.7 '
            'percentiles over all retained trials and time bins of that session).'),
        'behaviour_sampling': (
            'wheel speed = |velocity| of the ~1 kHz uniformly resampled wheel trace; '
            'whisker motion energy from the left camera (60 Hz) with the right camera '
            '(150 Hz) as fallback; both linearly interpolated onto the right edge of each '
            '20 ms bin, as in the reference get_behavior_per_interval.'),
        'neuron_inclusion': (
            'clusters with ibllib quality label >= 1 (the IBL "well-isolated neuron" '
            'criterion: amplitude > 50 uV, noise cut-off < 20 uV, no refractory-period '
            'violation), restricted to grey matter (Beryl acronym not in {root, void}).'),
        'trial_inclusion': (
            'ibl_data_utils.load_trials_and_mask(max_trial_len=10.0): no NaN in '
            'stimOn_times/choice/feedback_times/probabilityLeft/firstMovement_times/'
            'feedbackType; 0.08 s <= firstMovement_times - stimOn_times <= 2.0 s; '
            'choice != 0; feedback_times - goCue_times <= 10 s; plus a valid wheel and '
            'whisker-motion-energy trace covering the whole 2 s window without NaNs, and '
            'a trial window fully covered by the electrophysiology (no inter-spike gap '
            '> 0.5 s in the pooled unfiltered spike train, and inside the recorded '
            'interval).'),
        'session_inclusion': (
            'sessions of the BWM public release (bwm_release.csv, 459 eids / 699 probe '
            f'insertions) with >= {MIN_NEURONS} well-isolated grey-matter neurons, a '
            'usable whisker-motion-energy trace, and >= 2 usable trials.'),
        'reference_code': ('/app/code/code_zhang2025 (Zhang et al. 2026); parameters '
                           "interval_len=2, binsize=0.02, align_time='stimOn_times', "
                           'time_window=(-0.5, 1.5)'),
        'n_sessions': len(results),
        'n_subjects': len(subjects),
        'n_trials_total': int(np.sum(ntrials)),
        'n_neurons_total': int(np.sum(nneurons)),
        'session_info': [
            {'eid': r['eid'], 'subject': subject_of_eid[r['eid']],
             'lab': lab_of_eid[r['eid']], 'date': str(date_of_eid[r['eid']]),
             'n_probes': r['n_probes'], 'n_clusters_all': r['n_clusters_all'],
             'n_good_clusters': r['n_good_all'], 'n_neurons': r['n_neurons'],
             'n_trials_raw': r['n_trials_all'], 'n_trials_after_mask': r['n_trials_mask'],
             'n_trials_dropped_no_neural_data': r['n_trials_no_neural'],
             'n_trials_dropped_no_behaviour': r['n_trials_no_behaviour'],
             'n_trials_kept': r['n_trials_kept'], 'camera': r['camera_used'],
             'wheel_speed_tertiles': [float(x) for x in r['wheel_thresholds']],
             'whisker_me_tertiles': [float(x) for x in r['me_thresholds']],
             'frac_correct': r['frac_correct']}
            for r in results],
        'failed_sessions': [{'eid': e, 'error': m} for e, m in failures],
    }
    print(f'assembled in {time.time() - t:.1f}s', flush=True)

    # ---- summary ----------------------------------------------------------------
    print('\n=== conversion summary ===')
    print(f'sessions: {len(results)} (of {len(eids)} in the release)')
    print(f'subjects: {len(subjects)}')
    print(f'brain regions: {len(all_regions)}')
    print(f'neurons: total {np.sum(nneurons)}, mean/session {np.mean(nneurons):.1f}, '
          f'min {np.min(nneurons)}, max {np.max(nneurons)}')
    print(f'trials: total {np.sum(ntrials)}, mean/session {np.mean(ntrials):.1f}, '
          f'min {np.min(ntrials)}, max {np.max(ntrials)}')
    print(f'raw trials/session: mean {np.mean([r["n_trials_all"] for r in results]):.1f}, '
          f'median {np.median([r["n_trials_all"] for r in results]):.0f}, '
          f'range {np.min([r["n_trials_all"] for r in results])}-'
          f'{np.max([r["n_trials_all"] for r in results])}')
    print(f'all clusters/probe: {np.sum([r["n_clusters_all"] for r in results]) / np.sum([r["n_probes"] for r in results]):.1f}')
    print(f'good clusters/probe: {np.sum([r["n_good_all"] for r in results]) / np.sum([r["n_probes"] for r in results]):.1f}')
    print(f'trials dropped for missing ephys coverage: '
          f'{np.sum([r["n_trials_no_neural"] for r in results])}; '
          f'for missing behaviour: {np.sum([r["n_trials_no_behaviour"] for r in results])}')
    print(f'fraction correct (all raw trials): '
          f'{np.mean([r["frac_correct"] for r in results]):.3f}')
    for i, name in enumerate(OUTPUT_NAMES):
        vals = np.concatenate([np.concatenate([o[i] for o in sess]) for sess in data['output']])
        frac = np.bincount(vals, minlength=len(OUTPUT_VALUES[i])) / len(vals)
        print(f'output {i} {name}: ' +
              ', '.join(f'{v}={f:.3f}' for v, f in zip(OUTPUT_VALUES[i], frac)))

    # ---- save --------------------------------------------------------------------
    t = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'\nwrote {args.outfile} ({os.path.getsize(args.outfile) / 1e9:.2f} GB) '
          f'in {time.time() - t:.1f}s')
    print(f'TOTAL TIME {time.time() - t_start:.1f}s')


if __name__ == '__main__':
    main()
