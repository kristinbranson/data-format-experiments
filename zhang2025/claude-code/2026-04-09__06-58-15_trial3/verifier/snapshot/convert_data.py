#!/usr/bin/env python3
"""
Convert IBL Brain-wide Map data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--sample] [--full] [--show-processing]

References:
    - Zhang et al. (2025) "Exploiting correlations across trials and behavioral sessions"
    - IBL et al. (2024) "A brain-wide map of neural activity during complex behaviour"
"""

import os
import sys
import time
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from iblatlas.regions import BrainRegions
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================
# Constants matching reference code (0_data_caching.py)
# ============================================================
BINSIZE = 0.02          # 20ms bins
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)  # seconds relative to alignment event
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]  # 2.0 seconds
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 time bins

# Trial filtering (matching load_trials_and_mask defaults + prepare_data)
MIN_RT = 0.08
MAX_RT = 2.0
MAX_TRIAL_LEN = 10.0
EXCLUDE_NOCHOICE = True

# NaN exclusion events (matching reference code default)
NAN_EXCLUDE = [
    'stimOn_times', 'choice', 'feedback_times',
    'probabilityLeft', 'firstMovement_times', 'feedbackType'
]

DATA_ROOT = '/app/data/one_cache'
BWM_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'

# ============================================================
# Data loading functions
# ============================================================

def find_file(base_path, filename, revisions=None):
    """Find a file in the base path, checking revision directories."""
    # Check base path first
    direct = os.path.join(base_path, filename)
    if os.path.exists(direct):
        return direct

    # Check revision directories (sorted descending to get latest)
    if revisions is None:
        revisions = []
        for d in sorted(os.listdir(base_path), reverse=True):
            if d.startswith('#') and d.endswith('#'):
                revisions.append(d)

    for rev in revisions:
        candidate = os.path.join(base_path, rev, filename)
        if os.path.exists(candidate):
            return candidate

    return None


def load_spike_data(session_path, probe_name):
    """Load spike times, clusters, and cluster info for a probe."""
    alf_path = os.path.join(session_path, 'alf')
    probe_path = os.path.join(alf_path, probe_name)

    if not os.path.exists(probe_path):
        return None, None

    # Find pykilosort directory
    pks_path = os.path.join(probe_path, 'pykilosort')
    if not os.path.exists(pks_path):
        return None, None

    # Get the latest revision
    revisions = sorted([d for d in os.listdir(pks_path)
                       if d.startswith('#') and d.endswith('#')], reverse=True)
    if not revisions:
        spike_dir = pks_path
    else:
        spike_dir = os.path.join(pks_path, revisions[0])

    # Load spike data
    times_file = os.path.join(spike_dir, 'spikes.times.npy')
    clusters_file = os.path.join(spike_dir, 'spikes.clusters.npy')
    if not os.path.exists(times_file) or not os.path.exists(clusters_file):
        return None, None

    spikes = {
        'times': np.load(times_file).flatten(),
        'clusters': np.load(clusters_file).flatten(),
    }

    # Load cluster info
    channels_file = os.path.join(spike_dir, 'clusters.channels.npy')
    depths_file = os.path.join(spike_dir, 'clusters.depths.npy')
    metrics_file = os.path.join(spike_dir, 'clusters.metrics.pqt')

    clusters = {}
    if os.path.exists(channels_file):
        clusters['channels'] = np.load(channels_file).flatten()
    if os.path.exists(depths_file):
        clusters['depths'] = np.load(depths_file).flatten()
    if os.path.exists(metrics_file):
        clusters['metrics'] = pd.read_parquet(metrics_file)

    # Load brain region info from channels
    # Try electrodeSites first (probe level), then channels (pykilosort level)
    brain_ids = None
    for candidate_dir in [probe_path, spike_dir]:
        for name in ['electrodeSites.brainLocationIds_ccf_2017.npy',
                      'channels.brainLocationIds_ccf_2017.npy']:
            candidate = os.path.join(candidate_dir, name)
            if os.path.exists(candidate):
                brain_ids = np.load(candidate).flatten()
                break
        if brain_ids is not None:
            break

    # Also check revisions of pykilosort dir
    if brain_ids is None:
        for rev in revisions:
            candidate = os.path.join(pks_path, rev, 'channels.brainLocationIds_ccf_2017.npy')
            if os.path.exists(candidate):
                brain_ids = np.load(candidate).flatten()
                break

    clusters['brain_ids'] = brain_ids

    return spikes, clusters


def merge_probes(spikes_list, clusters_list):
    """Merge spikes and clusters from multiple probes. Matches reference code."""
    merged_spikes_parts = []
    merged_brain_ids = []
    merged_channels = []
    merged_depths = []
    cluster_max = 0

    for spikes, clusters in zip(spikes_list, clusters_list):
        s = {
            'times': spikes['times'].copy(),
            'clusters': spikes['clusters'].copy() + cluster_max,
        }

        n_clusters = int(spikes['clusters'].max()) + 1 if len(spikes['clusters']) > 0 else 0

        # Get brain region for each cluster
        if clusters.get('brain_ids') is not None and clusters.get('channels') is not None:
            chan_ids = clusters['channels'].astype(int)
            # Clip to valid range
            chan_ids = np.clip(chan_ids, 0, len(clusters['brain_ids']) - 1)
            cluster_brain_ids = clusters['brain_ids'][chan_ids]
            merged_brain_ids.append(cluster_brain_ids)

        if clusters.get('channels') is not None:
            merged_channels.append(clusters['channels'])
        if clusters.get('depths') is not None:
            merged_depths.append(clusters['depths'])

        cluster_max += n_clusters
        merged_spikes_parts.append(s)

    # Merge and sort by time
    all_times = np.concatenate([s['times'] for s in merged_spikes_parts])
    all_clusters = np.concatenate([s['clusters'] for s in merged_spikes_parts])
    sort_idx = np.argsort(all_times, kind='stable')

    merged_spikes = {
        'times': all_times[sort_idx],
        'clusters': all_clusters[sort_idx],
    }

    merged_brain_ids = np.concatenate(merged_brain_ids) if merged_brain_ids else None

    return merged_spikes, merged_brain_ids


def load_trials(session_path):
    """Load trials table from session."""
    alf_path = os.path.join(session_path, 'alf')
    trials_file = find_file(alf_path, '_ibl_trials.table.pqt')
    if trials_file is None:
        return None
    return pd.read_parquet(trials_file)


def create_trials_mask(trials_df):
    """Create trial exclusion mask matching reference code load_trials_and_mask()."""
    n_trials = len(trials_df)
    mask = np.ones(n_trials, dtype=bool)

    # NaN exclusion
    for event in NAN_EXCLUDE:
        if event in trials_df.columns:
            mask &= ~trials_df[event].isna()

    # Reaction time filter
    if MIN_RT is not None:
        rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
        mask &= (rt >= MIN_RT)
    if MAX_RT is not None:
        rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
        mask &= (rt <= MAX_RT)

    # Trial length filter
    if MAX_TRIAL_LEN is not None:
        if 'goCue_times' in trials_df.columns:
            trial_len = trials_df['feedback_times'] - trials_df['goCue_times']
            mask &= (trial_len <= MAX_TRIAL_LEN) | trial_len.isna()

    # Exclude no-choice trials
    if EXCLUDE_NOCHOICE:
        mask &= (trials_df['choice'] != 0)

    return mask


def load_wheel_speed(session_path):
    """Load wheel speed (absolute velocity) matching brainbox SessionLoader.load_wheel().

    Reproduces the exact processing from brainbox:
    1. Interpolate position to 1000 Hz uniform sampling
    2. Apply Butterworth low-pass filter (order=8, corner=20 Hz)
    3. Compute velocity as filtered diff * fs
    4. Speed = abs(velocity)
    """
    import scipy.signal
    from scipy.interpolate import interp1d as scipy_interp1d

    alf_path = os.path.join(session_path, 'alf')

    pos_file = find_file(alf_path, '_ibl_wheel.position.npy')
    ts_file = find_file(alf_path, '_ibl_wheel.timestamps.npy')

    if pos_file is None or ts_file is None:
        return None, None

    re_pos = np.load(pos_file).flatten()
    re_ts = np.load(ts_file).flatten()

    if len(re_pos) != len(re_ts) or len(re_pos) < 2:
        return None, None

    # Step 1: Interpolate to uniform 1000 Hz (matching interpolate_position)
    fs = 1000
    t = np.arange(re_ts[0], re_ts[-1], 1.0 / fs)
    if len(t) > 0 and t[-1] > re_ts[-1]:
        t = t[:-1]
    if len(t) < 2:
        return None, None

    position = scipy_interp1d(re_ts, re_pos, kind='linear')(t)

    # Step 2: Compute velocity with Butterworth filter (matching velocity_filtered)
    corner_frequency = 20
    order = 8
    sos = scipy.signal.butter(N=order, Wn=corner_frequency / fs * 2,
                               btype='lowpass', output='sos')
    vel = np.insert(np.diff(scipy.signal.sosfiltfilt(sos, position)), 0, 0) * fs

    # Step 3: Speed = abs(velocity)
    speed = np.abs(vel).astype(np.float32)

    return t.astype(np.float64), speed


def load_whisker_motion_energy(session_path):
    """Load whisker motion energy. Try left camera first, fall back to right."""
    alf_path = os.path.join(session_path, 'alf')

    # Try left camera first (matching reference code)
    me_file = find_file(alf_path, 'leftCamera.ROIMotionEnergy.npy')
    times_file = find_file(alf_path, '_ibl_leftCamera.times.npy')

    if me_file is not None and times_file is not None:
        me = np.load(me_file).flatten()
        times = np.load(times_file).flatten()
        if len(me) == len(times):
            return times, me

    # Fall back to right camera
    me_file = find_file(alf_path, 'rightCamera.ROIMotionEnergy.npy')
    times_file = find_file(alf_path, '_ibl_rightCamera.times.npy')

    if me_file is not None and times_file is not None:
        me = np.load(me_file).flatten()
        times = np.load(times_file).flatten()
        if len(me) == len(times):
            return times, me

    return None, None


# ============================================================
# Processing functions
# ============================================================

def bin_spikes_vectorized(spike_times, spike_clusters, interval_begs, interval_ends, n_clusters_total):
    """
    Bin spikes into time bins for each trial. Vectorized implementation.

    Returns:
        binned: array of shape (n_trials, n_clusters, N_BINS) - spike counts
    """
    n_trials = len(interval_begs)
    binned = np.zeros((n_trials, n_clusters_total, N_BINS), dtype=np.float32)

    for trial_idx in range(n_trials):
        t_beg = interval_begs[trial_idx]
        t_end = interval_ends[trial_idx]

        if np.isnan(t_beg) or np.isnan(t_end):
            continue

        # Find spikes in this interval
        idx_start = np.searchsorted(spike_times, t_beg, side='left')
        idx_end = np.searchsorted(spike_times, t_end, side='right')

        if idx_start >= idx_end:
            continue

        trial_times = spike_times[idx_start:idx_end]
        trial_clusters = spike_clusters[idx_start:idx_end]

        # Compute bin indices
        bin_idx = np.floor((trial_times - t_beg) / BINSIZE).astype(int)
        bin_idx = np.clip(bin_idx, 0, N_BINS - 1)

        # Count spikes per cluster per bin
        for spike_i in range(len(trial_times)):
            c = trial_clusters[spike_i]
            b = bin_idx[spike_i]
            if c < n_clusters_total:
                binned[trial_idx, c, b] += 1

    return binned


def bin_spikes_fast(spike_times, spike_clusters, interval_begs, interval_ends, n_clusters_total):
    """
    Fast spike binning using np.bincount with linear indexing.
    ~4x faster than np.add.at approach.
    """
    n_trials = len(interval_begs)
    binned = np.zeros((n_trials, n_clusters_total, N_BINS), dtype=np.float32)

    valid_trials = ~(np.isnan(interval_begs) | np.isnan(interval_ends))
    valid_idx = np.where(valid_trials)[0]

    if len(valid_idx) == 0:
        return binned

    starts = np.searchsorted(spike_times, interval_begs[valid_idx], side='left')
    ends = np.searchsorted(spike_times, interval_ends[valid_idx], side='right')
    minlength = n_clusters_total * N_BINS

    for i, trial_idx in enumerate(valid_idx):
        s, e = starts[i], ends[i]
        if s >= e:
            continue

        t = spike_times[s:e]
        c = spike_clusters[s:e]

        b = np.clip(((t - interval_begs[trial_idx]) / BINSIZE).astype(np.int32), 0, N_BINS - 1)

        valid_mask = c < n_clusters_total
        if not np.all(valid_mask):
            c = c[valid_mask]
            b = b[valid_mask]

        # Linear index: cluster * N_BINS + bin → use bincount for fast accumulation
        lin_idx = c * N_BINS + b
        counts = np.bincount(lin_idx, minlength=minlength)
        binned[trial_idx] = counts[:minlength].reshape(n_clusters_total, N_BINS)

    return binned


def interpolate_behavior(beh_times, beh_values, interval_begs, interval_ends):
    """
    Interpolate continuous behavior to bin times.
    Matching reference code get_behavior_per_interval().

    The reference code checks:
    - abs(interval_beg - first_data_time) > binsize → skip
    - abs(interval_end - last_data_time) > binsize → skip

    Returns:
        binned_beh: array of shape (n_trials, N_BINS)
        good_mask: boolean array of shape (n_trials,)
    """
    n_trials = len(interval_begs)
    binned_beh = np.full((n_trials, N_BINS), np.nan, dtype=np.float32)
    good_mask = np.zeros(n_trials, dtype=bool)

    # Pre-compute search indices (matching reference code)
    idxs_beg = np.searchsorted(beh_times, interval_begs, side='right')
    idxs_end = np.searchsorted(beh_times, interval_ends, side='left')

    for trial_idx in range(n_trials):
        t_beg = interval_begs[trial_idx]
        t_end = interval_ends[trial_idx]

        if np.isnan(t_beg) or np.isnan(t_end):
            continue

        ib = idxs_beg[trial_idx]
        ie = idxs_end[trial_idx]

        trial_times = beh_times[ib:ie]
        trial_vals = beh_values[ib:ie]

        if len(trial_vals) == 0:
            continue

        # Check timing alignment (matching reference code checks exactly)
        if np.abs(t_beg - trial_times[0]) > BINSIZE:
            continue
        if np.abs(t_end - trial_times[-1]) > BINSIZE:
            continue

        # Interpolation target times (matching reference code)
        x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)

        try:
            y_interp = interp1d(trial_times, trial_vals, kind='linear',
                               fill_value='extrapolate')(x_interp)
            binned_beh[trial_idx] = y_interp.astype(np.float32)
            good_mask[trial_idx] = True
        except Exception:
            continue

    return binned_beh, good_mask


def compute_trial_number_in_block(prob_left):
    """
    Compute trial number within the current block.
    A block change occurs when probabilityLeft changes value.
    """
    trial_numbers = np.zeros(len(prob_left), dtype=np.float32)
    current_block_start = 0
    current_val = prob_left[0]

    for i in range(len(prob_left)):
        if prob_left[i] != current_val:
            current_block_start = i
            current_val = prob_left[i]
        trial_numbers[i] = i - current_block_start

    return trial_numbers


def discretize_to_bins(values, n_bins=3):
    """
    Discretize continuous values into n_bins equal-frequency (quantile) bins.
    Returns integer bin labels 0, 1, ..., n_bins-1.
    """
    # Compute quantile thresholds from non-NaN values
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.zeros_like(values, dtype=np.int32)

    quantiles = np.linspace(0, 100, n_bins + 1)
    thresholds = np.percentile(valid, quantiles)

    # Digitize
    result = np.digitize(values, thresholds[1:-1], right=False).astype(np.int32)
    # Clip to valid range
    result = np.clip(result, 0, n_bins - 1)

    return result


# ============================================================
# Session processing
# ============================================================

def process_session(session_info, show_processing=False):
    """
    Process a single session and return converted data.

    Returns dict with keys: neural, input, output, subject, brain_region_ids,
                            n_trials, n_neurons, or None if session fails.
    """
    eid, lab, subject, date, session_number, probe_names = session_info
    session_num_str = str(int(session_number)).zfill(3)
    session_path = os.path.join(DATA_ROOT, lab, 'Subjects', subject, date, session_num_str)

    t0 = time.time()

    if not os.path.exists(session_path):
        print(f"  Session path not found: {session_path}")
        return None

    # 1. Load trials
    trials_df = load_trials(session_path)
    if trials_df is None:
        print(f"  No trials table found for {eid}")
        return None

    # 2. Create trial mask
    mask = create_trials_mask(trials_df)

    # 3. Load spike data from all probes and merge
    spikes_list = []
    clusters_list = []

    for pname in probe_names:
        spikes, clusters = load_spike_data(session_path, pname)
        if spikes is not None:
            spikes_list.append(spikes)
            clusters_list.append(clusters)

    if len(spikes_list) == 0:
        print(f"  No spike data found for {eid}")
        return None

    # Merge probes
    if len(spikes_list) == 1:
        merged_spikes = spikes_list[0]
        brain_ids = clusters_list[0].get('brain_ids')
        if brain_ids is not None and clusters_list[0].get('channels') is not None:
            chan_ids = clusters_list[0]['channels'].astype(int)
            chan_ids = np.clip(chan_ids, 0, len(brain_ids) - 1)
            cluster_brain_ids = brain_ids[chan_ids]
        else:
            cluster_brain_ids = None
    else:
        merged_spikes, cluster_brain_ids = merge_probes(spikes_list, clusters_list)

    n_clusters = int(merged_spikes['clusters'].max()) + 1 if len(merged_spikes['clusters']) > 0 else 0

    if n_clusters == 0:
        print(f"  No clusters found for {eid}")
        return None

    # 4. Get brain regions using Beryl mapping
    br = BrainRegions()
    if cluster_brain_ids is not None:
        cluster_acronyms = br.id2acronym(cluster_brain_ids)
        cluster_beryl = br.acronym2acronym(cluster_acronyms, mapping='Beryl')
    else:
        cluster_beryl = np.array(['void'] * n_clusters)

    # 5. Compute trial intervals
    stim_times = trials_df[ALIGN_TIME].values
    interval_begs = stim_times + TIME_WINDOW[0]
    interval_ends = stim_times + TIME_WINDOW[1]

    # 6. Bin spikes
    t_spike = time.time()
    binned_spikes = bin_spikes_fast(
        merged_spikes['times'], merged_spikes['clusters'],
        interval_begs, interval_ends, n_clusters
    )
    # binned_spikes shape: (n_all_trials, n_clusters, N_BINS)
    t_spike_done = time.time()

    # 7. Load and bin continuous behaviors
    # Wheel speed
    wheel_times, wheel_speed = load_wheel_speed(session_path)
    if wheel_times is not None:
        binned_wheel, wheel_mask = interpolate_behavior(
            wheel_times, wheel_speed, interval_begs, interval_ends)
    else:
        binned_wheel = np.full((len(trials_df), N_BINS), np.nan, dtype=np.float32)
        wheel_mask = np.zeros(len(trials_df), dtype=bool)

    # Whisker motion energy
    me_times, me_values = load_whisker_motion_energy(session_path)
    if me_times is not None:
        binned_me, me_mask = interpolate_behavior(
            me_times, me_values, interval_begs, interval_ends)
    else:
        binned_me = np.full((len(trials_df), N_BINS), np.nan, dtype=np.float32)
        me_mask = np.zeros(len(trials_df), dtype=bool)

    t_beh_done = time.time()

    # 8. Combine masks: trial mask AND behavior masks
    combined_mask = mask & wheel_mask & me_mask

    # Check we have enough trials
    n_good_trials = np.sum(combined_mask)
    if n_good_trials < 2:
        print(f"  Too few valid trials ({n_good_trials}) for {eid}")
        return None

    # 9. Apply mask to all data
    good_indices = np.where(combined_mask)[0]

    neural_data = binned_spikes[good_indices]  # (n_trials, n_clusters, N_BINS)
    wheel_data = binned_wheel[good_indices]    # (n_trials, N_BINS)
    me_data = binned_me[good_indices]          # (n_trials, N_BINS)

    # Per-trial variables
    choice_raw = trials_df['choice'].values[good_indices]       # -1 or 1
    prob_left = trials_df['probabilityLeft'].values[good_indices]  # 0.2, 0.5, 0.8

    # 10. Compute derived variables
    # Choice: -1 (left) -> 0, 1 (right) -> 1
    choice = ((choice_raw + 1) / 2).astype(np.int32)  # 0 or 1

    # Prior probability: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2
    prior = np.zeros(len(prob_left), dtype=np.int32)
    prior[prob_left == 0.2] = 0
    prior[prob_left == 0.5] = 1
    prior[prob_left == 0.8] = 2

    # Trial number in block (use full trials_df for correct block computation)
    all_prob_left = trials_df['probabilityLeft'].values
    all_trial_nums = compute_trial_number_in_block(all_prob_left)
    trial_num_in_block = all_trial_nums[good_indices].astype(np.float32)

    # Discretize wheel speed and whisker ME into 3 bins
    # Use all trials' data for consistent quantile computation
    wheel_flat = wheel_data.flatten()
    wheel_discrete = discretize_to_bins(wheel_flat, n_bins=3)
    wheel_discrete_2d = wheel_discrete.reshape(wheel_data.shape).astype(np.int32)

    me_flat = me_data.flatten()
    me_discrete = discretize_to_bins(me_flat, n_bins=3)
    me_discrete_2d = me_discrete.reshape(me_data.shape).astype(np.int32)

    # 11. Format as lists of trials
    # Time since stimulus onset (same for all trials)
    # Matching reference code: bin centers from interval_beg + binsize to interval_end
    time_since_onset = np.linspace(
        TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
    ).astype(np.float32)

    n_trials = len(good_indices)

    neural_list = []
    input_list = []
    output_list = []

    for trial_idx in range(n_trials):
        # Neural: (n_neurons, n_timepoints)
        # Use uint8 to minimize memory (spike counts in 20ms bins are almost always <255)
        neural_trial = neural_data[trial_idx].astype(np.uint8)  # already (n_clusters, N_BINS)
        neural_list.append(neural_trial)

        # Input: (n_input, n_timepoints) for time-varying, but we have mixed
        # input[0] = time_since_onset (time-varying), input[1] = trial_num_in_block (per-trial)
        # We make input shape (2, N_BINS) by repeating per-trial values
        input_trial = np.zeros((2, N_BINS), dtype=np.float32)
        input_trial[0, :] = time_since_onset
        input_trial[1, :] = trial_num_in_block[trial_idx]  # broadcast scalar
        input_list.append(input_trial)

        # Output: (n_output, n_timepoints) for time-varying outputs
        # output[0] = choice (per-trial, broadcast), output[1] = prior (per-trial, broadcast)
        # output[2] = wheel_speed (time-varying), output[3] = whisker_me (time-varying)
        output_trial = np.zeros((4, N_BINS), dtype=np.int32)
        output_trial[0, :] = choice[trial_idx]
        output_trial[1, :] = prior[trial_idx]
        output_trial[2, :] = wheel_discrete_2d[trial_idx]
        output_trial[3, :] = me_discrete_2d[trial_idx]
        output_list.append(output_trial)

    t_format_done = time.time()

    # Timing info
    print(f"  {eid}: {n_trials} trials, {n_clusters} neurons | "
          f"spike_bin={t_spike_done-t_spike:.1f}s, beh={t_beh_done-t_spike_done:.1f}s, "
          f"total={t_format_done-t0:.1f}s")

    # Show processing plots
    if show_processing:
        plot_processing(eid, neural_data, wheel_data, me_data,
                       wheel_discrete_2d, me_discrete_2d,
                       choice, prior, time_since_onset, trials_df, good_indices)

    return {
        'neural': neural_list,
        'input': input_list,
        'output': output_list,
        'subject': subject,
        'eid': eid,
        'cluster_beryl': cluster_beryl,
        'n_trials': n_trials,
        'n_neurons': n_clusters,
    }


def plot_processing(eid, neural_data, wheel_data, me_data,
                   wheel_discrete, me_discrete,
                   choice, prior, time_axis, trials_df, good_indices):
    """Generate processing verification plots for a session."""
    fig, axes = plt.subplots(4, 3, figsize=(20, 16))
    fig.suptitle(f'Processing: {eid}', fontsize=14)

    n_trials = neural_data.shape[0]
    trial_idx = min(5, n_trials - 1)

    # Row 0: Neural activity
    ax = axes[0, 0]
    n_show = min(50, neural_data.shape[1])
    ax.imshow(neural_data[trial_idx, :n_show, :], aspect='auto', interpolation='none')
    ax.set_title(f'Neural (trial {trial_idx}, first {n_show} neurons)')
    ax.set_xlabel('Time bin')
    ax.set_ylabel('Neuron')

    ax = axes[0, 1]
    mean_fr = neural_data[:, :, :].mean(axis=(0, 2))
    ax.hist(mean_fr, bins=50)
    ax.set_title('Mean firing rate distribution')
    ax.set_xlabel('Mean spike count per bin')

    ax = axes[0, 2]
    psth = neural_data.mean(axis=(0, 1))
    ax.plot(time_axis, psth)
    ax.axvline(0, color='r', linestyle='--', label='Stim onset')
    ax.set_title('PSTH (all neurons, all trials)')
    ax.set_xlabel('Time from stim onset (s)')
    ax.legend()

    # Row 1: Wheel speed
    ax = axes[1, 0]
    ax.plot(time_axis, wheel_data[trial_idx])
    ax.set_title(f'Wheel speed (trial {trial_idx})')
    ax.set_xlabel('Time (s)')

    ax = axes[1, 1]
    ax.hist(wheel_data.flatten(), bins=50)
    ax.set_title('Wheel speed distribution')

    ax = axes[1, 2]
    for val in [0, 1, 2]:
        frac = np.mean(wheel_discrete == val)
        ax.bar(val, frac, label=f'Bin {val}')
    ax.set_title('Wheel speed discretization')
    ax.set_xlabel('Bin')
    ax.set_ylabel('Fraction')
    ax.legend()

    # Row 2: Whisker ME
    ax = axes[2, 0]
    ax.plot(time_axis, me_data[trial_idx])
    ax.set_title(f'Whisker ME (trial {trial_idx})')
    ax.set_xlabel('Time (s)')

    ax = axes[2, 1]
    ax.hist(me_data.flatten(), bins=50)
    ax.set_title('Whisker ME distribution')

    ax = axes[2, 2]
    for val in [0, 1, 2]:
        frac = np.mean(me_discrete == val)
        ax.bar(val, frac, label=f'Bin {val}')
    ax.set_title('Whisker ME discretization')
    ax.set_xlabel('Bin')
    ax.set_ylabel('Fraction')
    ax.legend()

    # Row 3: Choice and prior distributions
    ax = axes[3, 0]
    ax.bar([0, 1], [np.mean(choice == 0), np.mean(choice == 1)])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Left', 'Right'])
    ax.set_title('Choice distribution')

    ax = axes[3, 1]
    ax.bar([0, 1, 2], [np.mean(prior == 0), np.mean(prior == 1), np.mean(prior == 2)])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['0.2', '0.5', '0.8'])
    ax.set_title('Prior (probabilityLeft) distribution')

    ax = axes[3, 2]
    # Check temporal alignment - plot neural + output for one trial
    ax.plot(time_axis, neural_data[trial_idx].mean(axis=0), label='Mean neural', color='blue')
    ax2 = ax.twinx()
    ax2.plot(time_axis, wheel_data[trial_idx], label='Wheel speed', color='orange', alpha=0.7)
    ax.set_title('Temporal alignment check')
    ax.set_xlabel('Time (s)')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f'processing_{eid}.png', dpi=100)
    plt.close()


# ============================================================
# Main conversion
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Convert IBL BWM data to decoder format')
    parser.add_argument('output', type=str, help='Output pickle file path')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--full', action='store_true', help='Process all sessions (default)')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing verification')
    parser.add_argument('--resume-from', type=int, default=0, help='Resume from session index (0-based)')
    args = parser.parse_args()

    t_start = time.time()

    # Load BWM release info
    bwm_df = pd.read_csv(BWM_CSV, index_col=0)

    # Group probes by session
    session_groups = bwm_df.groupby('eid')
    session_list = []
    for eid, group in session_groups:
        row = group.iloc[0]
        probe_names = list(group['probe_name'])
        session_list.append((
            eid, row['lab'], row['subject'], row['date'],
            row['session_number'], probe_names
        ))

    if args.sample:
        session_list = session_list[:2]
        print(f"Sample mode: processing {len(session_list)} sessions")
    else:
        print(f"Full mode: processing {len(session_list)} sessions")

    # Process sessions - save to temporary files to avoid OOM (64 GB cgroup limit)
    import tempfile
    import gc
    tmp_dir = tempfile.mkdtemp(prefix='ibl_convert_')

    subject_set = []
    region_set = []
    session_meta = []  # lightweight metadata per session

    n_processed = 0
    n_skipped = 0

    for i, session_info in enumerate(session_list):
        if i < args.resume_from:
            continue

        eid = session_info[0]
        print(f"\n[{i+1}/{len(session_list)}] Processing {eid}...")

        # Limit show-processing to first 2 sessions
        show = args.show_processing and i < 2

        try:
            result = process_session(session_info, show_processing=show)
        except Exception as e:
            print(f"  ERROR processing {eid}: {type(e).__name__}: {e}")
            n_skipped += 1
            continue

        if result is None:
            n_skipped += 1
            continue

        # Track subjects
        subject = result['subject']
        if subject not in subject_set:
            subject_set.append(subject)
        subject_idx = subject_set.index(subject)

        # Track brain regions
        session_region_idx = np.zeros(result['n_neurons'], dtype=np.int32)
        for neuron_idx in range(result['n_neurons']):
            region = result['cluster_beryl'][neuron_idx]
            if region not in region_set:
                region_set.append(region)
            session_region_idx[neuron_idx] = region_set.index(region)

        # Save session data to temp file to free memory
        tmp_file = os.path.join(tmp_dir, f'session_{n_processed:04d}.pkl')
        with open(tmp_file, 'wb') as f:
            pickle.dump({
                'neural': result['neural'],
                'input': result['input'],
                'output': result['output'],
                'subject_idx': subject_idx,
                'brain_region_idx': session_region_idx,
                'eid': eid,
                'n_trials': result['n_trials'],
                'n_neurons': result['n_neurons'],
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

        session_meta.append({
            'tmp_file': tmp_file,
            'n_trials': result['n_trials'],
            'n_neurons': result['n_neurons'],
        })

        del result
        gc.collect()

        n_processed += 1

        elapsed = time.time() - t_start
        rate = elapsed / (i + 1)
        remaining = rate * (len(session_list) - i - 1)
        print(f"  Progress: {n_processed} processed, {n_skipped} skipped | "
              f"Elapsed: {elapsed:.0f}s, ETA: {remaining:.0f}s")
        sys.stdout.flush()

    print(f"\n{'='*60}")
    print(f"Processed {n_processed} sessions, skipped {n_skipped}")

    # Reassemble from temp files in batches to stay under memory limit
    # Save in two passes: first collect metadata and eids, then stream into final pickle
    print(f"\nReassembling from {n_processed} temp files...")
    t_reassemble = time.time()

    all_eids = []
    all_subject_idx = []
    total_trials = 0
    total_neurons = 0
    n_neurons_list = []

    # Pass 1: collect lightweight metadata
    for meta in session_meta:
        with open(meta['tmp_file'], 'rb') as f:
            sess = pickle.load(f)
        all_eids.append(sess['eid'])
        all_subject_idx.append(sess['subject_idx'])
        total_trials += sess['n_trials']
        total_neurons += sess['n_neurons']
        n_neurons_list.append(sess['n_neurons'])
        del sess

    # Print summary
    print(f"\nFinal dataset:")
    print(f"  Sessions: {n_processed}")
    print(f"  Subjects: {len(subject_set)}")
    print(f"  Total trials: {total_trials}")
    print(f"  Mean trials/session: {total_trials/max(1,n_processed):.1f}")
    print(f"  Brain regions: {len(region_set)}")
    print(f"  Neurons: mean={np.mean(n_neurons_list):.0f}, total={total_neurons}")

    # Pass 2: build and save in batches
    # Load all temp files but process in batches to manage memory
    BATCH_SIZE = 50  # sessions per batch
    print(f"\nSaving to {args.output} in batches of {BATCH_SIZE}...")
    t_save = time.time()

    # We need all lists in memory for pickle - load batch by batch
    all_neural = []
    all_input = []
    all_output = []
    all_brain_region_idx = []

    for batch_start in range(0, len(session_meta), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(session_meta))
        batch_size = batch_end - batch_start
        print(f"  Loading sessions {batch_start}-{batch_end-1}...")

        for idx in range(batch_start, batch_end):
            with open(session_meta[idx]['tmp_file'], 'rb') as f:
                sess = pickle.load(f)
            all_neural.append(sess['neural'])
            all_input.append(sess['input'])
            all_output.append(sess['output'])
            all_brain_region_idx.append(sess['brain_region_idx'])
            del sess

        gc.collect()

    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': subject_set,
        'subject_idx': np.array(all_subject_idx, dtype=np.int64),
        'brain_regions': region_set,
        'brain_region_idx': all_brain_region_idx,
        'input_names': ['time_since_stim_onset', 'trial_number_in_block'],
        'output_names': ['choice', 'prior_probability_left', 'wheel_speed', 'whisker_motion_energy'],
        'output_values': [
            ['left', 'right'],          # choice: 0=left, 1=right
            ['0.2', '0.5', '0.8'],      # prior: 0=0.2, 1=0.5, 2=0.8
            ['low', 'medium', 'high'],  # wheel speed bins
            ['low', 'medium', 'high'],  # whisker ME bins
        ],
        'metadata': {
            'task_description': 'IBL decision-making task: mice rotate wheel to indicate visual stimulus location',
            'time_bin_size': BINSIZE * 1000,  # in ms
            'temporal_alignment_event': 'stimulus onset (stimOn_times)',
            'off_start': TIME_WINDOW[0],  # -0.5s before stimulus
            'off_end': TIME_WINDOW[1],    # 1.5s after stimulus
            'n_time_bins': N_BINS,
            'session_eids': all_eids,
            'trial_filtering': {
                'min_rt': MIN_RT,
                'max_rt': MAX_RT,
                'max_trial_len': MAX_TRIAL_LEN,
                'exclude_nochoice': EXCLUDE_NOCHOICE,
                'nan_exclude': NAN_EXCLUDE,
            },
            'neuron_filtering': 'none (all clusters used, matching Zhang et al. 2025)',
            'spike_sorting': 'Kilosort 2.5 (pykilosort)',
            'brain_region_mapping': 'Beryl (iblatlas)',
        },
    }

    print(f"  Pickling...")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    file_size = os.path.getsize(args.output)
    print(f"Saved ({file_size/1e6:.1f} MB) in {time.time()-t_save:.1f}s")
    print(f"Total time: {time.time()-t_start:.1f}s")

    # Cleanup temp files
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"Cleaned up temp directory")


if __name__ == '__main__':
    main()
