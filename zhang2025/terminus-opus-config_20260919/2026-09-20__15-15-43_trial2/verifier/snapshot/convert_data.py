#!/usr/bin/env python3
"""Convert the IBL Brain Wide Map dataset into the decoder-compatible pickle format.

Usage:
    python -u /app/convert_data.py <outpicklefile> [--full | --sample] [--show-processing]

Processing follows the reference caching pipeline of Zhang et al. 2025
(`/app/code/code_zhang2025/src/0_data_caching.py` and `src/utils/ibl_data_utils.py`):

    params = {'interval_len': 2, 'binsize': 0.02, 'single_region': False,
              'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}

i.e. trials are aligned to stimulus onset and span -0.5 s to +1.5 s, binned into 100
non-overlapping 20 ms bins.  Trial curation uses the reference
`load_trials_and_mask(one, eid, max_trial_len=10.0)`.  Neuron curation follows the BWM
data paper: only well-isolated units (IBL single-unit QC label >= 1) located in grey
matter (Beryl acronym not root/void) are kept.  See /app/CONVERSION_NOTES.md.
"""
import argparse
import os
import pickle
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# The reference code lives here; we import its trial-curation function directly so the
# trial mask is *identical* to the one the reference pipeline produces.
sys.path.insert(0, '/app/code/code_zhang2025/src')

# ---------------------------------------------------------------------------
# Configuration - these are the reference pipeline's `params`
# ---------------------------------------------------------------------------
BWM_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'
ONE_KWARGS = dict(cache_dir='/app/data/one_cache',
                  tables_dir='/app/cache/one_tables',
                  mode='local', silent=True)

ALIGN_EVENT = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)      # seconds relative to the alignment event
BIN_SIZE = 0.02                # seconds
N_BINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BIN_SIZE))   # = 100
MAX_TRIAL_LEN = 10.0           # reference `load_trials_and_mask(..., max_trial_len=10.0)`
MIN_TRIALS_PER_SESSION = 2     # target format needs >= 2 trials to evaluate the decoder
N_DISCRETE_BINS = 3            # wheel speed / whisker ME are discretised into 3 bins
NON_GREY = ('root', 'void')    # Beryl acronyms that are not grey matter

INPUT_NAMES = ['time_from_stim_onset', 'stim_onset', 'trial_num_in_block']
OUTPUT_NAMES = ['choice', 'prior_prob_left', 'wheel_speed', 'whisker_motion_energy']
OUTPUT_VALUES = [
    ['left', 'right'],                 # choice: left = 0, right = 1
    ['0.2', '0.5', '0.8'],             # prior probability of left
    ['low', 'medium', 'high'],         # wheel speed tercile
    ['low', 'medium', 'high'],         # whisker motion energy tercile
]

# bin centres relative to the alignment event: -0.49, -0.47, ..., +1.49
BIN_CENTRES = TIME_WINDOW[0] + BIN_SIZE * (np.arange(N_BINS) + 0.5)
# right edge of each bin, used for interpolating the continuous behaviours.  This matches
# the reference `get_behavior_per_interval`, which evaluates the behaviour on
# np.linspace(t_beg + binsize, t_end, n_bins).
BIN_RIGHT_EDGES = TIME_WINDOW[0] + BIN_SIZE * (np.arange(N_BINS) + 1.0)
# index of the bin containing the alignment event itself (t = 0 falls in bin 25)
STIM_ONSET_BIN = int(np.floor((0.0 - TIME_WINDOW[0]) / BIN_SIZE))

_ONE = None          # per-process ONE instance (ONE is NOT thread-safe)
_BRAIN_REGIONS = None


def get_one():
    global _ONE
    if _ONE is None:
        from one.api import ONE
        _ONE = ONE(**ONE_KWARGS)
    return _ONE


def get_brain_regions():
    global _BRAIN_REGIONS
    if _BRAIN_REGIONS is None:
        from iblatlas.regions import BrainRegions
        _BRAIN_REGIONS = BrainRegions()
    return _BRAIN_REGIONS


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def load_session_neurons(eid, probe_rows):
    """Load, quality-filter and merge the spike sorting of all probes of a session.

    Mirrors the reference `load_spiking_data` + `merge_probes`, with the BWM paper's
    neuron inclusion criteria applied (`label >= 1`, grey matter only).

    Returns
    -------
    spike_times : (n_spikes,) float64, sorted
    spike_clusters : (n_spikes,) int64, index into the kept-neuron list
    acronyms : (n_neurons,) str, Beryl acronym of each kept neuron
    counts : dict of diagnostic counts
    """
    from brainbox.io.one import SpikeSortingLoader
    one = get_one()
    br = get_brain_regions()

    times_list, clusters_list, acronyms = [], [], []
    n_offset = 0
    n_all = n_good = 0
    for _, row in probe_rows.iterrows():
        ssl = SpikeSortingLoader(pid=row.pid, one=one, eid=eid, pname=row.probe_name)
        spikes, clusters, channels = ssl.load_spike_sorting()
        if spikes is None or len(spikes) == 0 or 'times' not in spikes:
            continue
        clu = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
        n_all += len(clu)
        # IBL single-unit QC: label is the mean of the three RIGOR criteria, so
        # label >= 1 means all three passed -> "well-isolated neuron" in the BWM paper.
        good = (clu['label'].values >= 1)
        n_good += int(good.sum())
        beryl = np.asarray(br.acronym2acronym(clu['acronym'].to_numpy(), mapping='Beryl'))
        keep = good & ~np.isin(beryl, NON_GREY)
        keep_idx = np.flatnonzero(keep)
        if keep_idx.size == 0:
            continue
        # re-index the kept clusters to 0..n-1 (+ offset for the previous probe), which is
        # the `merge_probes` behaviour of giving every probe a disjoint cluster range
        remap = np.full(len(clu), -1, dtype=np.int64)
        remap[keep_idx] = np.arange(keep_idx.size) + n_offset
        sel = np.isin(spikes['clusters'], keep_idx)
        times_list.append(np.asarray(spikes['times'])[sel])
        clusters_list.append(remap[np.asarray(spikes['clusters'])[sel]])
        acronyms.extend(beryl[keep_idx].tolist())
        n_offset += keep_idx.size

    if n_offset == 0:
        return None, None, [], dict(n_all=n_all, n_good=n_good, n_kept=0)

    spike_times = np.concatenate(times_list)
    spike_clusters = np.concatenate(clusters_list)
    order = np.argsort(spike_times, kind='stable')   # `merge_probes` sorts by spike time
    return (spike_times[order], spike_clusters[order], acronyms,
            dict(n_all=n_all, n_good=n_good, n_kept=n_offset))


def bin_spikes(spike_times, spike_clusters, n_neurons, align_times):
    """Bin spikes into (n_trials, n_neurons, N_BINS) counts aligned to `align_times`.

    Vectorised equivalent of the reference `get_spike_data_per_interval`, which calls
    `iblutil.numerical.bincount2D(..., xbin=BIN_SIZE, xlim=[t_beg, t_end])` per trial.
    Verified to give bit-identical results (see /app/cache/check_binning.py).
    """
    n_trials = len(align_times)
    begs = align_times + TIME_WINDOW[0]
    ends = align_times + TIME_WINDOW[1]
    i0 = np.searchsorted(spike_times, begs, side='left')
    i1 = np.searchsorted(spike_times, ends, side='left')
    out = np.zeros((n_trials, n_neurons, N_BINS), dtype=np.float32)
    for k in range(n_trials):
        a, b = i0[k], i1[k]
        if b <= a:
            continue
        bin_idx = ((spike_times[a:b] - begs[k]) / BIN_SIZE).astype(np.int64)
        np.clip(bin_idx, 0, N_BINS - 1, out=bin_idx)
        np.add.at(out[k], (spike_clusters[a:b], bin_idx), 1.0)
    return out


def interp_behavior(beh_times, beh_values, align_times):
    """Interpolate a continuous behaviour onto the trial time grid.

    Returns (values, valid) where values is (n_trials, N_BINS) and valid is a boolean
    mask of trials whose whole window is covered by the recorded signal and free of NaN.
    The coverage/NaN checks mirror the reference `get_behavior_per_interval`
    ('target data starts too late' / 'ends too early' / 'nans in target data').
    """
    from scipy.interpolate import interp1d
    finite = np.isfinite(beh_times) & np.isfinite(beh_values)
    beh_times, beh_values = beh_times[finite], beh_values[finite]
    order = np.argsort(beh_times, kind='stable')
    beh_times, beh_values = beh_times[order], beh_values[order]

    grid = align_times[:, None] + BIN_RIGHT_EDGES[None, :]
    valid = (grid[:, 0] >= beh_times[0] - BIN_SIZE) & (grid[:, -1] <= beh_times[-1] + BIN_SIZE)
    vals = np.full(grid.shape, np.nan, dtype=np.float64)
    if valid.any():
        f = interp1d(beh_times, beh_values, kind='linear', bounds_error=False,
                     fill_value=(beh_values[0], beh_values[-1]))
        vals[valid] = f(grid[valid])
    valid &= np.isfinite(vals).all(axis=1)
    return vals, valid


def discretize(values, n_bins=N_DISCRETE_BINS):
    """Discretise a continuous (n_trials, N_BINS) signal into `n_bins` equal-count classes.

    Thresholds are the per-session quantiles of all (trial, bin) samples, so the classes
    are balanced within each session and invariant to the arbitrary per-session units of
    wheel speed and whisker motion energy.  Returns (labels int8, edges).
    """
    flat = values.ravel()
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.quantile(flat, qs)
    # Heavily zero-inflated signals can produce duplicate edges; nudge them apart so that
    # np.searchsorted still yields `n_bins` distinct classes wherever the data allow it.
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    labels = np.searchsorted(edges, flat, side='right').astype(np.int8)
    return labels.reshape(values.shape), edges


def trial_number_in_block(prob_left):
    """0-based index of each trial within its block of constant `probabilityLeft`.

    Computed on the *unmasked* trials table, because the position of a trial within its
    block is a property of the experiment and must not change when trials are dropped.
    """
    pl = np.asarray(prob_left, dtype=float)
    new_block = np.ones(len(pl), dtype=bool)
    new_block[1:] = pl[1:] != pl[:-1]
    block_start = np.maximum.accumulate(np.where(new_block, np.arange(len(pl)), 0))
    return np.arange(len(pl)) - block_start


# ---------------------------------------------------------------------------
# Per-session conversion
# ---------------------------------------------------------------------------
def convert_session(eid, probe_rows, show_processing=False, out_dir='.', zscore=True):
    """Convert a single session.  Returns a dict, or None with a 'skip' reason."""
    from brainbox.io.one import SessionLoader
    from utils.ibl_data_utils import load_trials_and_mask

    t_start = time.time()
    one = get_one()
    timing = {}

    # ---- trials + reference curation mask ---------------------------------
    t0 = time.time()
    sess_loader = SessionLoader(one=one, eid=eid)
    trials, mask = load_trials_and_mask(one=one, eid=eid, max_trial_len=MAX_TRIAL_LEN,
                                        sess_loader=sess_loader)
    timing['trials'] = time.time() - t0
    n_trials_raw = len(trials)
    mask = np.asarray(mask, dtype=bool)
    # `trial_in_block` must be computed before the mask is applied
    tnb_all = trial_number_in_block(trials['probabilityLeft'].values)
    if mask.sum() < MIN_TRIALS_PER_SESSION:
        return dict(eid=eid, skip='fewer than %d trials pass curation' % MIN_TRIALS_PER_SESSION,
                    n_trials_raw=n_trials_raw, n_trials_mask=int(mask.sum()))

    align_times = trials[ALIGN_EVENT].values[mask]
    choice_raw = trials['choice'].values[mask]
    prob_left_raw = trials['probabilityLeft'].values[mask]
    tnb = tnb_all[mask]

    # ---- continuous behaviours --------------------------------------------
    t0 = time.time()
    sess_loader.load_wheel()
    wheel_speed_raw, valid_wheel = interp_behavior(
        sess_loader.wheel['times'].to_numpy(),
        np.abs(sess_loader.wheel['velocity'].to_numpy()), align_times)
    timing['wheel'] = time.time() - t0

    # Reference `bin_behaviors`: prefer the left camera, fall back to the right one.
    t0 = time.time()
    me_raw, valid_me, me_view = None, None, None
    for view in ('left', 'right'):
        try:
            sess_loader.load_motion_energy(views=[view])
            me = sess_loader.motion_energy[view + 'Camera']
            me_raw, valid_me = interp_behavior(me['times'].to_numpy(),
                                               me['whiskerMotionEnergy'].to_numpy(),
                                               align_times)
            me_view = view
            break
        except Exception:
            continue
    timing['motion_energy'] = time.time() - t0
    if me_raw is None:
        return dict(eid=eid, skip='no whisker motion energy from either camera',
                    n_trials_raw=n_trials_raw, n_trials_mask=int(mask.sum()))

    keep = valid_wheel & valid_me
    if keep.sum() < MIN_TRIALS_PER_SESSION:
        return dict(eid=eid, skip='fewer than %d trials with complete behaviour' % MIN_TRIALS_PER_SESSION,
                    n_trials_raw=n_trials_raw, n_trials_mask=int(mask.sum()))

    align_times = align_times[keep]
    choice_raw = choice_raw[keep]
    prob_left_raw = prob_left_raw[keep]
    tnb = tnb[keep]
    wheel_speed_raw = wheel_speed_raw[keep]
    me_raw = me_raw[keep]
    n_trials = len(align_times)

    # ---- neurons -----------------------------------------------------------
    t0 = time.time()
    spike_times, spike_clusters, acronyms, counts = load_session_neurons(eid, probe_rows)
    timing['spikes'] = time.time() - t0
    if spike_times is None or len(acronyms) == 0:
        return dict(eid=eid, skip='no well-isolated grey-matter neurons',
                    n_trials_raw=n_trials_raw, n_trials_mask=int(mask.sum()), **counts)
    n_neurons = len(acronyms)

    t0 = time.time()
    spikes_binned = bin_spikes(spike_times, spike_clusters, n_neurons, align_times)
    timing['binning'] = time.time() - t0

    # Per-neuron z-scoring over all (trial, bin) samples of this session.  The provided
    # decoder does no normalisation of its own, and the reference decoding pipeline
    # standardises spike counts before decoding (`standardize_spike_data`).
    raw_counts_example = spikes_binned[0].copy() if show_processing else None
    if zscore:
        # Reference `data_loader_utils.standardize_spike_data`: for each time bin it takes a
        # SINGLE scalar mean and std pooled over all neurons and trials and applies that one
        # affine transform to the whole (trials x neurons) block for that bin.  This removes
        # the session-wide firing-rate scale and the stimulus-locked drift in the overall
        # level, while PRESERVING the relative differences between neurons (which a per-neuron
        # z-score would destroy - measured to cost ~0.02 balanced accuracy on choice/prior).
        # Accumulate the statistics in float64: each bin pools ~n_trials * n_neurons
        # (order 1e5) values, and a float32 summation over that many terms loses enough
        # precision to shift the standardized values by ~1e-3 (measured), which is far
        # more than float32 round-off of the result itself (~2e-6).
        mu = spikes_binned.mean(axis=(0, 1), keepdims=True, dtype=np.float64)
        sd = spikes_binned.std(axis=(0, 1), keepdims=True, dtype=np.float64)
        neural = ((spikes_binned.astype(np.float64) - mu)
                  / np.where(sd > 0, sd, 1.0)).astype(np.float32)
    else:
        neural = spikes_binned.astype(np.float32)

    # ---- inputs ------------------------------------------------------------
    stim_indicator = np.zeros(N_BINS, dtype=np.float32)
    stim_indicator[STIM_ONSET_BIN] = 1.0
    time_axis = BIN_CENTRES.astype(np.float32)
    inputs = []
    for k in range(n_trials):
        arr = np.empty((3, N_BINS), dtype=np.float32)
        arr[0] = time_axis
        arr[1] = stim_indicator
        arr[2] = tnb[k]
        inputs.append(arr)

    # ---- outputs -----------------------------------------------------------
    # choice: ALF +1 == mouse reported LEFT, -1 == reported RIGHT (verified in Step 3)
    choice = np.where(choice_raw > 0, 0, 1).astype(np.int8)
    prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
    prior = np.array([prior_map[round(float(p), 4)] for p in prob_left_raw], dtype=np.int8)
    wheel_lab, wheel_edges = discretize(wheel_speed_raw)
    me_lab, me_edges = discretize(me_raw)

    outputs = []
    for k in range(n_trials):
        arr = np.empty((4, N_BINS), dtype=np.int8)
        arr[0] = choice[k]
        arr[1] = prior[k]
        arr[2] = wheel_lab[k]
        arr[3] = me_lab[k]
        outputs.append(arr)

    if show_processing:
        plot_processing(eid, out_dir, raw_counts_example, neural, inputs, outputs,
                        wheel_speed_raw, me_raw, wheel_edges, me_edges,
                        align_times, sess_loader, me_view, acronyms)

    return dict(
        eid=eid,
        neural=[neural[k] for k in range(n_trials)],
        input=inputs,
        output=outputs,
        acronyms=acronyms,
        n_trials_raw=n_trials_raw,
        n_trials_mask=int(mask.sum()),
        n_trials=n_trials,
        me_view=me_view,
        wheel_edges=wheel_edges,
        me_edges=me_edges,
        mean_rate=float(spikes_binned.mean() / BIN_SIZE),
        timing=timing,
        seconds=time.time() - t_start,
        **counts,
    )


# ---------------------------------------------------------------------------
# Visualisation of every processing step (--show-processing)
# ---------------------------------------------------------------------------
def plot_processing(eid, out_dir, raw_counts, neural, inputs, outputs,
                    wheel_raw, me_raw, wheel_edges, me_edges,
                    align_times, sess_loader, me_view, acronyms):
    """Save a figure showing every step of the conversion for one session."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_trials = len(neural)
    fig, ax = plt.subplots(4, 2, figsize=(20, 18))
    t = BIN_CENTRES

    # (0,0) raw binned spike counts for trial 0
    im = ax[0, 0].imshow(raw_counts, aspect='auto', interpolation='nearest',
                         extent=[t[0], t[-1], raw_counts.shape[0], 0], cmap='Greys')
    ax[0, 0].axvline(0, color='r', lw=1.5)
    ax[0, 0].set_title('1. Raw binned spike counts, trial 0 (%d neurons x %d bins of %g ms)'
                       % (raw_counts.shape[0], N_BINS, BIN_SIZE * 1000))
    ax[0, 0].set_xlabel('time from stimulus onset (s)'); ax[0, 0].set_ylabel('neuron')
    plt.colorbar(im, ax=ax[0, 0])

    # (0,1) z-scored neural data for the same trial
    im = ax[0, 1].imshow(neural[0], aspect='auto', interpolation='nearest',
                         extent=[t[0], t[-1], neural[0].shape[0], 0], cmap='RdBu_r',
                         vmin=-3, vmax=3)
    ax[0, 1].axvline(0, color='k', lw=1.5)
    ax[0, 1].set_title('2. Per-neuron z-scored neural data, trial 0 (stored in `neural`)')
    ax[0, 1].set_xlabel('time from stimulus onset (s)'); ax[0, 1].set_ylabel('neuron')
    plt.colorbar(im, ax=ax[0, 1])

    # (1,0) trial-averaged population activity - should show a stimulus-onset response
    ax[1, 0].plot(t, np.mean([n.mean(axis=0) for n in neural], axis=0), color='k')
    ax[1, 0].axvline(0, color='r', lw=1.5, label='stimulus onset')
    ax[1, 0].set_title('3. Trial-averaged population activity (alignment check)')
    ax[1, 0].set_xlabel('time from stimulus onset (s)')
    ax[1, 0].set_ylabel('mean z-scored rate'); ax[1, 0].legend()

    # (1,1) the three decoder inputs for the first few trials
    for k in range(min(3, n_trials)):
        ax[1, 1].plot(t, inputs[k][0], label='time_from_stim_onset' if k == 0 else None, color='C0')
        ax[1, 1].plot(t, inputs[k][1], label='stim_onset (binary)' if k == 0 else None, color='C1')
        ax[1, 1].plot(t, inputs[k][2] / max(1, inputs[k][2].max()),
                      label='trial_num_in_block (scaled)' if k == 0 else None,
                      color='C2', ls='--')
    ax[1, 1].axvline(0, color='r', lw=1.0)
    ax[1, 1].set_title('4. Decoder inputs (first 3 trials)')
    ax[1, 1].set_xlabel('time from stimulus onset (s)'); ax[1, 1].legend()

    # (2,0) raw wheel speed against the source signal, for trial 0 - alignment check
    k = 0
    w = sess_loader.wheel
    sel = ((w['times'].to_numpy() >= align_times[k] + TIME_WINDOW[0]) &
           (w['times'].to_numpy() <= align_times[k] + TIME_WINDOW[1]))
    ax[2, 0].plot(w['times'].to_numpy()[sel] - align_times[k],
                  np.abs(w['velocity'].to_numpy()[sel]), color='0.6',
                  label='source |wheel velocity|')
    ax[2, 0].plot(BIN_RIGHT_EDGES, wheel_raw[k], 'o-', ms=3, color='C0',
                  label='interpolated onto bins')
    for e in wheel_edges:
        ax[2, 0].axhline(e, color='r', ls=':')
    ax[2, 0].axvline(0, color='r', lw=1.5)
    ax[2, 0].set_title('5. Wheel speed, trial 0 (dotted red = tercile thresholds)')
    ax[2, 0].set_xlabel('time from stimulus onset (s)'); ax[2, 0].legend()

    # (2,1) the discretised wheel-speed output on top of the continuous signal
    ax2 = ax[2, 1].twinx()
    ax[2, 1].plot(t, wheel_raw[k], color='C0', label='wheel speed')
    for e in wheel_edges:
        ax[2, 1].axhline(e, color='r', ls=':')
    ax2.step(t, outputs[k][2], where='mid', color='C3', label='discretised class')
    ax2.set_ylim(-0.2, 2.2); ax2.set_ylabel('class (0=low,1=medium,2=high)')
    ax[2, 1].set_title('6. Wheel speed discretisation check, trial 0')
    ax[2, 1].set_xlabel('time from stimulus onset (s)')

    # (3,0) whisker motion energy, same checks
    me = sess_loader.motion_energy[me_view + 'Camera']
    mt = me['times'].to_numpy()
    sel = ((mt >= align_times[k] + TIME_WINDOW[0]) & (mt <= align_times[k] + TIME_WINDOW[1]))
    ax[3, 0].plot(mt[sel] - align_times[k], me['whiskerMotionEnergy'].to_numpy()[sel],
                  color='0.6', label='source %s-camera ME' % me_view)
    ax[3, 0].plot(BIN_RIGHT_EDGES, me_raw[k], 'o-', ms=3, color='C1',
                  label='interpolated onto bins')
    for e in me_edges:
        ax[3, 0].axhline(e, color='r', ls=':')
    ax[3, 0].axvline(0, color='r', lw=1.5)
    ax[3, 0].set_title('7. Whisker motion energy, trial 0')
    ax[3, 0].set_xlabel('time from stimulus onset (s)'); ax[3, 0].legend()

    # (3,1) class balance of all four outputs
    labels, fracs = [], []
    allout = np.stack([o for o in outputs])          # (n_trials, 4, N_BINS)
    for d, name in enumerate(OUTPUT_NAMES):
        vals = allout[:, d, :].ravel()
        for c in range(len(OUTPUT_VALUES[d])):
            labels.append('%s=%s' % (name[:12], OUTPUT_VALUES[d][c]))
            fracs.append(float((vals == c).mean()))
    ax[3, 1].barh(range(len(fracs)), fracs)
    ax[3, 1].set_yticks(range(len(fracs))); ax[3, 1].set_yticklabels(labels, fontsize=8)
    ax[3, 1].axvline(1 / 3, color='r', ls=':'); ax[3, 1].axvline(0.5, color='g', ls=':')
    ax[3, 1].set_title('8. Output class fractions (red = 1/3, green = 1/2)')
    ax[3, 1].set_xlabel('fraction of samples')

    fig.suptitle('Conversion steps for session %s  (%d trials, %d neurons, %s camera)'
                 % (eid, n_trials, len(acronyms), me_view), fontsize=14)
    fig.tight_layout()
    path = os.path.join(out_dir, 'processing_%s.png' % eid)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print('  wrote %s' % path, flush=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _worker(args):
    eid, probe_rows, show_processing, out_dir, zscore = args
    try:
        return convert_session(eid, probe_rows, show_processing, out_dir, zscore)
    except Exception as exc:                                   # noqa: BLE001
        return dict(eid=eid, skip='exception: %s' % exc,
                    traceback=traceback.format_exc())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outfile', type=str, help='output pickle path')
    ap.add_argument('--full', action='store_true', help='process all sessions (default)')
    ap.add_argument('--sample', action='store_true', help='process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true',
                    help='plot every processing step for up to 2 sessions')
    ap.add_argument('--n-workers', type=int, default=16)
    ap.add_argument('--limit', type=int, default=None,
                    help='process only the first N sessions (diagnostics)')
    ap.add_argument('--no-zscore', action='store_true',
                    help='store raw spike counts instead of per-neuron z-scores (diagnostics)')
    args = ap.parse_args()

    t_start = time.time()
    bwm = pd.read_csv(BWM_CSV, index_col=0)
    print('bwm_release.csv: %d insertions, %d sessions, %d subjects, %d labs'
          % (len(bwm), bwm.eid.nunique(), bwm.subject.nunique(), bwm.lab.nunique()))

    eids = sorted(bwm.eid.unique())
    if args.sample:
        eids = eids[:2]
    if args.limit:
        eids = eids[:args.limit]
    n_show = 2 if args.show_processing else 0
    out_dir = os.path.dirname(os.path.abspath(args.outfile)) or '.'

    tasks = [(eid, bwm[bwm.eid == eid][['pid', 'probe_name']].copy(),
              i < n_show, out_dir, not args.no_zscore) for i, eid in enumerate(eids)]
    print('converting %d sessions with %d workers ...' % (len(tasks), args.n_workers),
          flush=True)

    results = []
    if args.n_workers > 1 and len(tasks) > 1:
        # A ONE instance is not thread-safe, so use processes (each builds its own).
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=min(args.n_workers, len(tasks))) as pool:
            for i, res in enumerate(pool.imap_unordered(_worker, tasks)):
                results.append(res)
                if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                    el = time.time() - t_start
                    print('  %d/%d sessions  %.0fs elapsed  (%.2f s/session, eta %.0fs)'
                          % (i + 1, len(tasks), el, el / (i + 1),
                             el / (i + 1) * (len(tasks) - i - 1)), flush=True)
    else:
        for i, task in enumerate(tasks):
            results.append(_worker(task))
            el = time.time() - t_start
            print('  %d/%d sessions  %.0fs elapsed' % (i + 1, len(tasks), el), flush=True)

    # keep the original session order for reproducibility
    by_eid = {r['eid']: r for r in results}
    results = [by_eid[e] for e in eids if e in by_eid]

    ok = [r for r in results if 'skip' not in r]
    skipped = [r for r in results if 'skip' in r]
    print('\nconverted %d sessions, skipped %d' % (len(ok), len(skipped)))
    from collections import Counter
    for reason, n in Counter(r['skip'].split(':')[0] for r in skipped).items():
        print('  skipped (%s): %d' % (reason, n))
    for r in skipped:
        if 'traceback' in r:
            print('  EXCEPTION in %s:\n%s' % (r['eid'], r['traceback']))

    # ---- assemble the target structure ------------------------------------
    eid2subject = bwm.drop_duplicates('eid').set_index('eid')['subject'].to_dict()
    eid2lab = bwm.drop_duplicates('eid').set_index('eid')['lab'].to_dict()
    subjects = sorted({eid2subject[r['eid']] for r in ok})
    subject_index = {s: i for i, s in enumerate(subjects)}
    brain_regions = sorted({a for r in ok for a in r['acronyms']})
    region_index = {a: i for i, a in enumerate(brain_regions)}

    data = {
        'neural': [r['neural'] for r in ok],
        'input': [r['input'] for r in ok],
        'output': [r['output'] for r in ok],
        'subjects': subjects,
        'subject_idx': np.array([subject_index[eid2subject[r['eid']]] for r in ok],
                                dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': [np.array([region_index[a] for a in r['acronyms']],
                                      dtype=np.int64) for r in ok],
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'task_description': (
                'IBL decision-making task (Brain Wide Map). A Gabor stimulus of one of five '
                'contrasts (0, 6.25, 12.5, 25, 100%) appears left or right; the mouse turns a '
                'wheel to bring it to the centre. After an initial 90-trial unbiased block '
                '(P(left) = 0.5), the stimulus side follows 20:80 or 80:20 blocks of 20-100 '
                'trials. Decoded outputs: the mouse choice (reported side, left/right), the '
                'block prior probability that the stimulus is on the left (0.2/0.5/0.8), and '
                'the wheel speed and whisker-pad motion energy, each discretised into three '
                'per-session equal-count levels (low/medium/high).'),
            'time_bin_size': BIN_SIZE * 1000.0,
            'temporal_alignment_event': 'stimulus onset (trials.stimOn_times)',
            'off_start': TIME_WINDOW[0],
            'off_end': TIME_WINDOW[1],
            'n_time_bins': N_BINS,
            'neural_data_type': ('spike counts in 20 ms bins, z-scored per neuron across all '
                                 'time bins and trials of the session'),
            'neuron_inclusion': ('IBL single-unit QC label >= 1 (amplitude > 50 uV, noise '
                                 'cut-off < 20 uV, no refractory-period violation) and located '
                                 'in grey matter (Beryl acronym not root/void)'),
            'trial_inclusion': ('reference load_trials_and_mask(max_trial_len=10): reaction '
                                'time (firstMovement_times - stimOn_times) in [0.08, 2.0] s, no '
                                'NaN in stimOn_times/choice/feedback_times/probabilityLeft/'
                                'firstMovement_times/feedbackType, feedback_times - goCue_times '
                                '<= 10 s, choice != 0; plus complete, NaN-free wheel and whisker '
                                'motion-energy coverage of the 2 s window'),
            'brain_region_mapping': 'Allen CCF acronyms mapped to the IBL Beryl atlas mapping',
            'input_descriptions': {
                'time_from_stim_onset': 'seconds from stimulus onset, bin centres -0.49 .. +1.49',
                'stim_onset': 'binary indicator, 1 in the bin containing stimulus onset',
                'trial_num_in_block': 'trials since the last change of probabilityLeft (0-based)',
            },
            'output_descriptions': {
                'choice': 'side the mouse reported: left = 0, right = 1 (ALF choice +1/-1)',
                'prior_prob_left': 'probabilityLeft of the block: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2',
                'wheel_speed': '|wheel velocity| discretised into per-session terciles',
                'whisker_motion_energy': ('whisker-pad motion energy (left camera, falling back '
                                          'to right) discretised into per-session terciles'),
            },
            'source': ('International Brain Laboratory Brain Wide Map, public data freeze '
                       'bwm_release.csv, loaded through ONE/ibllib'),
            'session_info': [
                {'eid': r['eid'], 'subject': eid2subject[r['eid']], 'lab': eid2lab[r['eid']],
                 'n_trials': r['n_trials'], 'n_trials_raw': r['n_trials_raw'],
                 'n_trials_after_mask': r['n_trials_mask'], 'n_neurons': len(r['acronyms']),
                 'n_clusters_all': r['n_all'], 'n_clusters_good': r['n_good'],
                 'motion_energy_camera': r['me_view'],
                 'wheel_speed_tercile_edges': [float(x) for x in r['wheel_edges']],
                 'whisker_me_tercile_edges': [float(x) for x in r['me_edges']],
                 'mean_firing_rate_hz': r['mean_rate']}
                for r in ok],
            'skipped_sessions': [{'eid': r['eid'], 'reason': r['skip']} for r in skipped],
        },
    }

    # ---- summary + sanity checks ------------------------------------------
    n_neurons = [len(r['acronyms']) for r in ok]
    n_trials = [r['n_trials'] for r in ok]
    print('\n--- summary ---')
    print('sessions          : %d' % len(ok))
    print('subjects          : %d' % len(subjects))
    print('brain regions     : %d' % len(brain_regions))
    print('neurons total     : %d  (mean %.1f, median %.0f, range %d-%d per session)'
          % (sum(n_neurons), np.mean(n_neurons), np.median(n_neurons),
             min(n_neurons), max(n_neurons)))
    print('trials total      : %d  (mean %.1f, median %.0f, range %d-%d per session)'
          % (sum(n_trials), np.mean(n_trials), np.median(n_trials),
             min(n_trials), max(n_trials)))
    print('clusters seen     : %d all, %d good (label >= 1)'
          % (sum(r['n_all'] for r in ok), sum(r['n_good'] for r in ok)))
    print('timepoints/trial  : %d   bin size %.0f ms   window [%g, %g] s'
          % (N_BINS, BIN_SIZE * 1000, TIME_WINDOW[0], TIME_WINDOW[1]))

    allout = np.concatenate([np.stack(r['output'])[:, :, :] for r in ok], axis=0)
    for d, name in enumerate(OUTPUT_NAMES):
        v = allout[:, d, :].ravel()
        fr = [float((v == c).mean()) for c in range(len(OUTPUT_VALUES[d]))]
        print('output %-24s fractions %s' % (name, np.round(fr, 4)))
    allin = np.concatenate([np.stack(r['input'])[:, :, :] for r in ok], axis=0)
    for d, name in enumerate(INPUT_NAMES):
        v = allin[:, d, :]
        print('input  %-24s range [%.4g, %.4g]' % (name, v.min(), v.max()))

    if ok and 'timing' in ok[0]:
        agg = {}
        for r in ok:
            for k, v in r['timing'].items():
                agg[k] = agg.get(k, 0.0) + v
        tot = sum(r['seconds'] for r in ok)
        print('\ntiming (summed over sessions, %.0f s of worker time):' % tot)
        for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
            print('  %-14s %7.1f s  (%.0f%%)' % (k, v, 100 * v / max(tot, 1e-9)))

    with open(args.outfile, 'wb') as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print('\nwrote %s (%.2f GB) in %.0f s'
          % (args.outfile, os.path.getsize(args.outfile) / 1e9, time.time() - t_start))


if __name__ == '__main__':
    main()
