#!/usr/bin/env python3
"""Convert IBL Brain-wide Map data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
"""
import sys
import os
import time
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
import scipy.signal
from iblatlas.regions import BrainRegions
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Constants matching reference code (0_data_caching.py)
# ============================================================
BINSIZE = 0.02  # 20ms bins
TIME_WINDOW = (-0.5, 1.5)  # relative to stimOn_times
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]  # 2.0 seconds
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 bins
ALIGN_TIME = 'stimOn_times'

# Trial filtering parameters (from load_trials_and_mask defaults + prepare_data call)
MIN_RT = 0.08
MAX_RT = 2.0
MAX_TRIAL_LEN = 10.0
EXCLUDE_NOCHOICE = True
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']

BASE_PATH = Path('data/one_cache')
BWM_CSV = 'code/code_zhang2025/data/bwm_release.csv'


def find_session_path(lab, subject, date):
    """Find the session path in the ONE cache."""
    for sess_num in ['001', '002', '003']:
        p = BASE_PATH / lab / 'Subjects' / subject / date / sess_num / 'alf'
        if p.exists():
            return p
    return None


def find_versioned_file(base_dir, filename):
    """Find a file that may be in a versioned (#date#) subdirectory."""
    # First check direct path
    direct = base_dir / filename
    if direct.exists():
        return direct
    # Check versioned subdirectories (sorted to get latest)
    versioned_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('#')], reverse=True)
    for vd in versioned_dirs:
        f = vd / filename
        if f.exists():
            return f
    return None


def load_trials_and_mask(sess_path):
    """Load trials table and create quality mask matching reference code."""
    # Find trials table
    trials_file = find_versioned_file(sess_path, '_ibl_trials.table.pqt')
    if trials_file is None:
        raise FileNotFoundError(f'No trials table found in {sess_path}')
    
    trials_df = pd.read_parquet(trials_file)
    
    # Load additional trial timing files if needed
    # stimOnTrigger_times
    stim_trig_file = find_versioned_file(sess_path, '_ibl_trials.stimOnTrigger_times.npy')
    if stim_trig_file is not None:
        trials_df['stimOnTrigger_times'] = np.load(stim_trig_file)
    
    # goCueTrigger_times
    go_trig_file = find_versioned_file(sess_path, '_ibl_trials.goCueTrigger_times.npy')
    if go_trig_file is None:
        go_trig_file = sess_path / '_ibl_trials.goCueTrigger_times.npy'
    if go_trig_file.exists():
        trials_df['goCueTrigger_times'] = np.load(go_trig_file)
    
    # Create mask matching reference code logic
    mask = pd.Series(True, index=trials_df.index)
    
    # Reaction time filter
    rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
    if MIN_RT is not None:
        mask &= (rt >= MIN_RT)
    if MAX_RT is not None:
        mask &= (rt <= MAX_RT)
    
    # Trial length filter
    if MAX_TRIAL_LEN is not None:
        if 'goCue_times' in trials_df.columns:
            trial_len = trials_df['feedback_times'] - trials_df['goCue_times']
            mask &= (trial_len <= MAX_TRIAL_LEN)
    
    # NaN exclusion
    for event in NAN_EXCLUDE:
        if event in trials_df.columns:
            mask &= ~trials_df[event].isna()
    
    # Exclude no-choice trials
    if EXCLUDE_NOCHOICE:
        mask &= (trials_df['choice'] != 0)
    
    return trials_df, mask


def load_spike_data(sess_path, probe_name):
    """Load spike times, clusters, and cluster info for a probe."""
    probe_path = sess_path / probe_name / 'pykilosort'
    if not probe_path.exists():
        return None, None, None, None
    
    # Find versioned directory
    versioned_dirs = sorted([d for d in probe_path.iterdir() if d.is_dir()], reverse=True)
    if not versioned_dirs:
        return None, None, None, None
    spike_dir = versioned_dirs[0]
    
    # Load spikes
    spike_times_file = spike_dir / 'spikes.times.npy'
    spike_clusters_file = spike_dir / 'spikes.clusters.npy'
    if not spike_times_file.exists() or not spike_clusters_file.exists():
        return None, None, None, None
    
    spike_times = np.load(spike_times_file).flatten()
    spike_clusters = np.load(spike_clusters_file).flatten()
    
    # Load cluster info
    cluster_channels = np.load(spike_dir / 'clusters.channels.npy').flatten()
    
    # Load channel brain locations
    chan_brain_ids = np.load(spike_dir / 'channels.brainLocationIds_ccf_2017.npy').flatten()
    
    # Map clusters to brain regions
    n_clusters = len(cluster_channels)
    cluster_brain_ids = chan_brain_ids[cluster_channels]
    
    return spike_times, spike_clusters, n_clusters, cluster_brain_ids


def merge_probes(spikes_list, clusters_info_list):
    """Merge spike data from multiple probes, matching reference code logic."""
    all_times = []
    all_clusters = []
    all_brain_ids = []
    cluster_offset = 0
    
    for spike_times, spike_clusters, n_clusters, cluster_brain_ids in zip(*[iter(x) for x in [spikes_list]]):
        pass  # This won't work, let me fix
    
    # Actually, spikes_list is list of (spike_times, spike_clusters, n_clusters, cluster_brain_ids)
    for spike_times, spike_clusters, n_clusters, cluster_brain_ids in spikes_list:
        all_times.append(spike_times)
        all_clusters.append(spike_clusters + cluster_offset)
        all_brain_ids.append(cluster_brain_ids)
        cluster_offset += n_clusters
    
    merged_times = np.concatenate(all_times)
    merged_clusters = np.concatenate(all_clusters)
    merged_brain_ids = np.concatenate(all_brain_ids)
    
    # Sort by time
    sort_idx = np.argsort(merged_times, kind='stable')
    merged_times = merged_times[sort_idx]
    merged_clusters = merged_clusters[sort_idx]
    
    return merged_times, merged_clusters, merged_brain_ids


def bin_spikes_vectorized(spike_times, spike_clusters, interval_starts, interval_ends, 
                          n_clusters_total, binsize, n_bins):
    """Bin spikes into trials efficiently using vectorized operations."""
    n_trials = len(interval_starts)
    binned = np.zeros((n_trials, n_clusters_total, n_bins), dtype=np.float32)
    
    for trial_idx in range(n_trials):
        t_start = interval_starts[trial_idx]
        t_end = interval_ends[trial_idx]
        
        if np.isnan(t_start) or np.isnan(t_end):
            continue
        
        # Select spikes in this interval
        mask = (spike_times >= t_start) & (spike_times < t_end)
        trial_times = spike_times[mask]
        trial_clusters = spike_clusters[mask]
        
        if len(trial_times) == 0:
            continue
        
        # Compute bin indices
        bin_idx = np.floor((trial_times - t_start) / binsize).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        
        # Accumulate spike counts
        for s_idx in range(len(trial_times)):
            binned[trial_idx, trial_clusters[s_idx], bin_idx[s_idx]] += 1
    
    return binned


def bin_spikes_fast(spike_times, spike_clusters, interval_starts, interval_ends,
                    n_clusters_total, binsize, n_bins):
    """Fast spike binning using searchsorted and numpy advanced indexing."""
    n_trials = len(interval_starts)
    binned = np.zeros((n_trials, n_clusters_total, n_bins), dtype=np.float32)
    
    # Pre-sort is already done (merged_probes sorts by time)
    for trial_idx in range(n_trials):
        t_start = interval_starts[trial_idx]
        t_end = interval_ends[trial_idx]
        
        if np.isnan(t_start) or np.isnan(t_end):
            continue
        
        # Use searchsorted for fast interval selection
        idx_start = np.searchsorted(spike_times, t_start, side='left')
        idx_end = np.searchsorted(spike_times, t_end, side='left')
        
        trial_times = spike_times[idx_start:idx_end]
        trial_clusters = spike_clusters[idx_start:idx_end]
        
        if len(trial_times) == 0:
            continue
        
        # Compute bin indices
        bin_idx = np.minimum(np.floor((trial_times - t_start) / binsize).astype(int), n_bins - 1)
        
        # Use np.add.at for accumulation (no race conditions)
        np.add.at(binned[trial_idx], (trial_clusters, bin_idx), 1)
    
    return binned


def interpolate_behavior(beh_times, beh_values, interval_starts, interval_ends, 
                         binsize, n_bins):
    """Interpolate behavioral data to match neural time bins.
    
    Matches reference code: get_behavior_per_interval
    x_interp = np.linspace(interval_beg + binsize, interval_end, n_bins)
    """
    n_trials = len(interval_starts)
    result = np.full((n_trials, n_bins), np.nan, dtype=np.float32)
    valid_mask = np.ones(n_trials, dtype=bool)
    
    for trial_idx in range(n_trials):
        t_start = interval_starts[trial_idx]
        t_end = interval_ends[trial_idx]
        
        if np.isnan(t_start) or np.isnan(t_end):
            valid_mask[trial_idx] = False
            continue
        
        # Find behavior data in this interval
        idx_start = np.searchsorted(beh_times, t_start, side='right')
        idx_end = np.searchsorted(beh_times, t_end, side='left')
        
        seg_times = beh_times[idx_start:idx_end]
        seg_vals = beh_values[idx_start:idx_end]
        
        if len(seg_vals) == 0:
            valid_mask[trial_idx] = False
            continue
        
        # Check that behavior data covers the interval (matching reference code)
        if np.abs(t_start - seg_times[0]) > binsize:
            valid_mask[trial_idx] = False
            continue
        if np.abs(t_end - seg_times[-1]) > binsize:
            valid_mask[trial_idx] = False
            continue
        
        # Interpolate to bin centers (matching reference code)
        x_interp = np.linspace(t_start + binsize, t_end, n_bins)
        try:
            f_interp = interp1d(seg_times, seg_vals, kind='linear', fill_value='extrapolate')
            result[trial_idx] = f_interp(x_interp)
        except Exception:
            valid_mask[trial_idx] = False
    
    return result, valid_mask


def compute_wheel_speed(sess_path, fs=1000, corner_frequency=20, order=8):
    """Load wheel data and compute speed (absolute velocity).
    
    Matches reference code: SessionLoader.load_wheel()
    1. Interpolate position to uniform sampling at fs Hz
    2. Apply Butterworth low-pass filter and compute velocity
    3. Speed = abs(velocity)
    """
    pos_file = sess_path / '_ibl_wheel.position.npy'
    ts_file = sess_path / '_ibl_wheel.timestamps.npy'
    
    if not pos_file.exists() or not ts_file.exists():
        return None, None
    
    position = np.load(pos_file).flatten()
    timestamps = np.load(ts_file).flatten()
    
    if len(timestamps) < 2:
        return None, None
    
    # Step 1: Interpolate position to uniform sampling (matching interpolate_position)
    t_uniform = np.arange(timestamps[0], timestamps[-1], 1.0 / fs)
    if len(t_uniform) > 0 and t_uniform[-1] > timestamps[-1]:
        t_uniform = t_uniform[:-1]
    
    pos_interp = interp1d(timestamps, position, kind='linear')(t_uniform)
    
    # Step 2: Apply Butterworth low-pass filter and compute velocity (matching velocity_filtered)
    sos = scipy.signal.butter(N=order, Wn=corner_frequency / fs * 2, btype='lowpass', output='sos')
    vel = np.insert(np.diff(scipy.signal.sosfiltfilt(sos, pos_interp)), 0, 0) * fs
    
    # Step 3: Speed = absolute velocity
    speed = np.abs(vel).astype(np.float32)
    
    return t_uniform, speed


def load_whisker_motion_energy(sess_path):
    """Load whisker motion energy data.
    
    Matches reference code: tries left camera first, then right.
    """
    # Try left camera first
    me_file = find_versioned_file(sess_path, 'leftCamera.ROIMotionEnergy.npy')
    times_file = find_versioned_file(sess_path, '_ibl_leftCamera.times.npy')
    
    if me_file is None or times_file is None:
        # Try right camera
        me_file = find_versioned_file(sess_path, 'rightCamera.ROIMotionEnergy.npy')
        times_file = find_versioned_file(sess_path, '_ibl_rightCamera.times.npy')
    
    if me_file is None or times_file is None:
        return None, None
    
    me_values = np.load(me_file).flatten()
    me_times = np.load(times_file).flatten()
    
    # Ensure same length
    min_len = min(len(me_values), len(me_times))
    me_values = me_values[:min_len]
    me_times = me_times[:min_len]
    
    # Remove NaN values
    valid = ~np.isnan(me_values) & ~np.isnan(me_times)
    me_values = me_values[valid]
    me_times = me_times[valid]
    
    return me_times, me_values


def compute_trial_number_in_block(prob_left):
    """Compute trial number within each block.
    
    A block is a sequence of consecutive trials with the same probabilityLeft.
    Trial number starts at 1 for the first trial in each block.
    """
    trial_nums = np.zeros(len(prob_left), dtype=np.float32)
    current_block = prob_left.iloc[0] if hasattr(prob_left, 'iloc') else prob_left[0]
    count = 0
    
    for i in range(len(prob_left)):
        val = prob_left.iloc[i] if hasattr(prob_left, 'iloc') else prob_left[i]
        if val == current_block:
            count += 1
        else:
            current_block = val
            count = 1
        trial_nums[i] = count
    
    return trial_nums


def discretize_to_bins(values, n_bins=3, quantiles=None):
    """Discretize continuous values into n_bins categories using quantile boundaries.
    
    If quantiles are provided, use them. Otherwise compute from data.
    Returns discretized values and the quantile boundaries used.
    """
    if quantiles is None:
        # Compute quantile boundaries
        flat_vals = values[~np.isnan(values)].flatten()
        quantiles = np.quantile(flat_vals, np.linspace(0, 1, n_bins + 1))
        # Make boundaries slightly wider to include all values
        quantiles[0] = -np.inf
        quantiles[-1] = np.inf
    
    # Discretize
    discretized = np.digitize(values, quantiles[1:-1])  # Returns 0, 1, ..., n_bins-1
    discretized = np.clip(discretized, 0, n_bins - 1)
    
    return discretized.astype(np.float32), quantiles


def process_session(eid, probes_info, show_processing=False):
    """Process a single session.
    
    Returns: dict with session data, or None if session fails.
    """
    t0 = time.time()
    
    # Get session path from first probe
    first_probe = probes_info[0]
    sess_path = find_session_path(first_probe['lab'], first_probe['subject'], first_probe['date'])
    if sess_path is None:
        print(f'  Session {eid}: no session path found')
        return None
    
    # Load trials and mask
    try:
        trials_df, trials_mask = load_trials_and_mask(sess_path)
    except Exception as e:
        print(f'  Session {eid}: failed to load trials: {e}')
        return None
    
    # Load and merge spike data from all probes
    probe_data = []
    for probe_info in probes_info:
        probe_name = probe_info['probe_name']
        result = load_spike_data(sess_path, probe_name)
        if result[0] is not None:
            probe_data.append(result)
        else:
            print(f'  Session {eid}: failed to load probe {probe_name}')
    
    if not probe_data:
        print(f'  Session {eid}: no probe data loaded')
        return None
    
    # Merge probes
    if len(probe_data) == 1:
        spike_times, spike_clusters, n_clusters, cluster_brain_ids = probe_data[0]
    else:
        all_times = []
        all_clusters = []
        all_brain_ids = []
        cluster_offset = 0
        for st, sc, nc, cbi in probe_data:
            all_times.append(st)
            all_clusters.append(sc + cluster_offset)
            all_brain_ids.append(cbi)
            cluster_offset += nc
        spike_times = np.concatenate(all_times)
        spike_clusters = np.concatenate(all_clusters)
        cluster_brain_ids = np.concatenate(all_brain_ids)
        n_clusters = cluster_offset
        # Sort by time
        sort_idx = np.argsort(spike_times, kind='stable')
        spike_times = spike_times[sort_idx]
        spike_clusters = spike_clusters[sort_idx]
    
    # Map brain regions using Beryl
    br = BrainRegions()
    cluster_acronyms_raw = br.id2acronym(cluster_brain_ids)
    cluster_acronyms_beryl = br.acronym2acronym(cluster_acronyms_raw, mapping='Beryl')
    
    # Apply trial mask
    valid_trials = trials_df[trials_mask].copy()
    
    if len(valid_trials) < 2:
        print(f'  Session {eid}: too few valid trials ({len(valid_trials)})')
        return None
    
    # Compute intervals aligned to stimOn_times
    stim_on = valid_trials[ALIGN_TIME].values
    interval_starts = stim_on + TIME_WINDOW[0]
    interval_ends = stim_on + TIME_WINDOW[1]
    
    # Bin spikes
    t1 = time.time()
    binned_spikes = bin_spikes_fast(
        spike_times, spike_clusters, interval_starts, interval_ends,
        n_clusters, BINSIZE, N_BINS
    )
    t_spike = time.time() - t1
    
    # Load and interpolate wheel speed
    t1 = time.time()
    wheel_times, wheel_speed = compute_wheel_speed(sess_path)
    if wheel_times is None:
        print(f'  Session {eid}: no wheel data')
        return None
    
    wheel_binned, wheel_valid = interpolate_behavior(
        wheel_times, wheel_speed, interval_starts, interval_ends, BINSIZE, N_BINS
    )
    t_wheel = time.time() - t1
    
    # Load and interpolate whisker motion energy
    t1 = time.time()
    me_times, me_values = load_whisker_motion_energy(sess_path)
    if me_times is None:
        print(f'  Session {eid}: no whisker ME data')
        return None
    
    me_binned, me_valid = interpolate_behavior(
        me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS
    )
    t_me = time.time() - t1
    
    # Combined validity mask (all data must be valid)
    combined_valid = wheel_valid & me_valid
    
    if np.sum(combined_valid) < 2:
        print(f'  Session {eid}: too few valid trials after behavior filtering ({np.sum(combined_valid)})')
        return None
    
    # Apply validity mask
    binned_spikes = binned_spikes[combined_valid]
    wheel_binned = wheel_binned[combined_valid]
    me_binned = me_binned[combined_valid]
    valid_trials_final = valid_trials[combined_valid].copy()
    valid_trials_final.reset_index(drop=True, inplace=True)
    
    n_trials_final = len(valid_trials_final)
    
    # ---- Construct inputs ----
    # Input 0: Time since stimulus onset (continuous, time-varying)
    # Matches the bin centers used in reference code
    time_since_stim = np.linspace(
        TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
    ).astype(np.float32)
    
    # Input 1: Trial number in block (continuous, per-trial)
    trial_num_in_block = compute_trial_number_in_block(
        valid_trials_final['probabilityLeft']
    )
    
    # ---- Construct outputs ----
    # Output 0: Choice (binary, per-trial) left(-1)->0, right(1)->1
    choice = valid_trials_final['choice'].values.copy()
    choice_binary = np.where(choice == -1, 0, 1).astype(np.float32)
    
    # Output 1: Prior probability of left (per-trial) 0.2->0, 0.5->1, 0.8->2
    prob_left = valid_trials_final['probabilityLeft'].values.copy()
    prior_cat = np.zeros(len(prob_left), dtype=np.float32)
    prior_cat[prob_left == 0.2] = 0
    prior_cat[prob_left == 0.5] = 1
    prior_cat[prob_left == 0.8] = 2
    
    # Output 2 & 3: Wheel speed and Whisker ME (discretized, time-varying)
    # Will be discretized globally after collecting all sessions
    
    t_total = time.time() - t0
    print(f'  Session {eid}: {n_trials_final} trials, {n_clusters} neurons, '
          f'{len(np.unique(cluster_acronyms_beryl))} regions '
          f'(spike:{t_spike:.1f}s, wheel:{t_wheel:.1f}s, ME:{t_me:.1f}s, total:{t_total:.1f}s)')
    
    result = {
        'eid': eid,
        'subject': first_probe['subject'],
        'lab': first_probe['lab'],
        'binned_spikes': binned_spikes,  # (n_trials, n_clusters, n_bins)
        'wheel_binned': wheel_binned,  # (n_trials, n_bins)
        'me_binned': me_binned,  # (n_trials, n_bins)
        'time_since_stim': time_since_stim,  # (n_bins,)
        'trial_num_in_block': trial_num_in_block,  # (n_trials,)
        'choice_binary': choice_binary,  # (n_trials,)
        'prior_cat': prior_cat,  # (n_trials,)
        'cluster_acronyms_beryl': cluster_acronyms_beryl,  # (n_clusters,)
        'n_trials': n_trials_final,
        'n_clusters': n_clusters,
    }
    
    if show_processing:
        result['trials_df'] = valid_trials_final
        result['stim_on'] = stim_on[combined_valid]
    
    return result


def plot_processing(session_data, session_idx, save_prefix='processing'):
    """Plot processing steps for visual verification."""
    eid = session_data['eid']
    fig, axes = plt.subplots(4, 3, figsize=(20, 16))
    fig.suptitle(f'Processing: {eid}', fontsize=14)
    
    # Plot 1: Neural activity raster for first trial
    ax = axes[0, 0]
    trial_data = session_data['binned_spikes'][0]  # (n_clusters, n_bins)
    active = np.where(trial_data.sum(axis=1) > 0)[0]
    if len(active) > 50:
        active = active[:50]
    if len(active) > 0:
        ax.imshow(trial_data[active], aspect='auto', cmap='hot',
                  extent=[TIME_WINDOW[0], TIME_WINDOW[1], len(active), 0])
    ax.set_xlabel('Time from stim onset (s)')
    ax.set_ylabel('Neuron')
    ax.set_title('Trial 1: Neural activity')
    ax.axvline(0, color='white', linestyle='--', alpha=0.7)
    
    # Plot 2: Mean neural activity across trials
    ax = axes[0, 1]
    mean_activity = session_data['binned_spikes'].mean(axis=0)  # (n_clusters, n_bins)
    ax.plot(session_data['time_since_stim'], mean_activity.mean(axis=0))
    ax.set_xlabel('Time from stim onset (s)')
    ax.set_ylabel('Mean spike count')
    ax.set_title('Mean neural activity')
    ax.axvline(0, color='red', linestyle='--', alpha=0.7)
    
    # Plot 3: Wheel speed for first 5 trials
    ax = axes[0, 2]
    for i in range(min(5, session_data['n_trials'])):
        ax.plot(session_data['time_since_stim'], session_data['wheel_binned'][i], alpha=0.5)
    ax.set_xlabel('Time from stim onset (s)')
    ax.set_ylabel('Wheel speed')
    ax.set_title('Wheel speed (first 5 trials)')
    ax.axvline(0, color='red', linestyle='--', alpha=0.7)
    
    # Plot 4: Whisker ME for first 5 trials
    ax = axes[1, 0]
    for i in range(min(5, session_data['n_trials'])):
        ax.plot(session_data['time_since_stim'], session_data['me_binned'][i], alpha=0.5)
    ax.set_xlabel('Time from stim onset (s)')
    ax.set_ylabel('Whisker ME')
    ax.set_title('Whisker ME (first 5 trials)')
    ax.axvline(0, color='red', linestyle='--', alpha=0.7)
    
    # Plot 5: Choice distribution
    ax = axes[1, 1]
    choices, counts = np.unique(session_data['choice_binary'], return_counts=True)
    ax.bar(['Left (0)', 'Right (1)'], [counts[choices==0][0] if 0 in choices else 0,
                                        counts[choices==1][0] if 1 in choices else 0])
    ax.set_title('Choice distribution')
    
    # Plot 6: Prior distribution
    ax = axes[1, 2]
    priors, counts = np.unique(session_data['prior_cat'], return_counts=True)
    labels = ['0.2 (0)', '0.5 (1)', '0.8 (2)']
    ax.bar([labels[int(p)] for p in priors], counts)
    ax.set_title('Prior distribution')
    
    # Plot 7: Trial number in block
    ax = axes[2, 0]
    ax.plot(session_data['trial_num_in_block'])
    ax.set_xlabel('Trial')
    ax.set_ylabel('Trial # in block')
    ax.set_title('Trial number in block')
    
    # Plot 8: Wheel speed histogram
    ax = axes[2, 1]
    ax.hist(session_data['wheel_binned'].flatten(), bins=50)
    ax.set_xlabel('Wheel speed')
    ax.set_title('Wheel speed distribution')
    
    # Plot 9: Whisker ME histogram
    ax = axes[2, 2]
    ax.hist(session_data['me_binned'].flatten(), bins=50)
    ax.set_xlabel('Whisker ME')
    ax.set_title('Whisker ME distribution')
    
    # Plot 10: Brain regions
    ax = axes[3, 0]
    regions, region_counts = np.unique(session_data['cluster_acronyms_beryl'], return_counts=True)
    ax.barh(regions, region_counts)
    ax.set_xlabel('# neurons')
    ax.set_title('Brain regions')
    
    # Plot 11: Mean activity by choice
    ax = axes[3, 1]
    for c, label in [(0, 'Left'), (1, 'Right')]:
        mask = session_data['choice_binary'] == c
        if mask.sum() > 0:
            mean_act = session_data['binned_spikes'][mask].mean(axis=(0, 1))
            ax.plot(session_data['time_since_stim'], mean_act, label=label)
    ax.legend()
    ax.set_xlabel('Time from stim onset (s)')
    ax.set_title('Mean activity by choice')
    ax.axvline(0, color='red', linestyle='--', alpha=0.7)
    
    # Plot 12: Spike count per trial
    ax = axes[3, 2]
    spike_counts = session_data['binned_spikes'].sum(axis=(1, 2))
    ax.plot(spike_counts)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Total spikes')
    ax.set_title('Spikes per trial')
    
    fig.tight_layout()
    fig.savefig(f'{save_prefix}_{eid}.png', dpi=100)
    plt.close(fig)
    print(f'  Saved processing plot: {save_prefix}_{eid}.png')


def build_final_dataset(session_results):
    """Build the final dataset dictionary from processed session results."""
    
    # First pass: collect all wheel speed and whisker ME values for global discretization
    print('Computing global discretization boundaries...')
    all_wheel = []
    all_me = []
    for sess in session_results:
        all_wheel.append(sess['wheel_binned'].flatten())
        all_me.append(sess['me_binned'].flatten())
    
    all_wheel = np.concatenate(all_wheel)
    all_me = np.concatenate(all_me)
    
    # Remove NaN values
    all_wheel = all_wheel[~np.isnan(all_wheel)]
    all_me = all_me[~np.isnan(all_me)]
    
    # Compute quantile boundaries for 3 bins
    wheel_quantiles = np.array([-np.inf, 
                                 np.quantile(all_wheel, 1/3), 
                                 np.quantile(all_wheel, 2/3), 
                                 np.inf])
    me_quantiles = np.array([-np.inf,
                              np.quantile(all_me, 1/3),
                              np.quantile(all_me, 2/3),
                              np.inf])
    
    print(f'  Wheel speed quantiles: {wheel_quantiles}')
    print(f'  Whisker ME quantiles: {me_quantiles}')
    
    # Collect all unique subjects and brain regions
    all_subjects = sorted(list(set(sess['subject'] for sess in session_results)))
    subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
    
    all_regions_set = set()
    for sess in session_results:
        all_regions_set.update(sess['cluster_acronyms_beryl'])
    all_regions = sorted(list(all_regions_set))
    region_to_idx = {r: i for i, r in enumerate(all_regions)}
    
    # Build the dataset
    neural_list = []
    input_list = []
    output_list = []
    subject_idx_list = []
    brain_region_idx_list = []
    
    for sess in session_results:
        n_trials = sess['n_trials']
        n_clusters = sess['n_clusters']
        
        # Neural: list of (n_neurons, n_timepoints) arrays
        session_neural = []
        for trial_idx in range(n_trials):
            # binned_spikes is (n_trials, n_clusters, n_bins)
            session_neural.append(sess['binned_spikes'][trial_idx].astype(np.float32))
        neural_list.append(session_neural)
        
        # Input: (d_input, n_timepoints) or (d_input,) per trial
        session_input = []
        for trial_idx in range(n_trials):
            # input[0]: time since stim onset (time-varying) - same for all trials
            # input[1]: trial number in block (per-trial) - broadcast to time-varying
            inp = np.stack([
                sess['time_since_stim'],  # (n_bins,)
                np.full(N_BINS, sess['trial_num_in_block'][trial_idx], dtype=np.float32),  # (n_bins,)
            ], axis=0)  # (2, n_bins)
            session_input.append(inp)
        input_list.append(session_input)
        
        # Output: (d_output, n_timepoints) or (d_output,) per trial
        session_output = []
        
        # Discretize wheel speed and whisker ME for this session
        wheel_disc = np.digitize(sess['wheel_binned'], wheel_quantiles[1:-1]).astype(np.float32)
        wheel_disc = np.clip(wheel_disc, 0, 2)
        me_disc = np.digitize(sess['me_binned'], me_quantiles[1:-1]).astype(np.float32)
        me_disc = np.clip(me_disc, 0, 2)
        
        for trial_idx in range(n_trials):
            out = np.stack([
                np.full(N_BINS, int(sess["choice_binary"][trial_idx]), dtype=np.int64),  # (n_bins,)
                np.full(N_BINS, int(sess["prior_cat"][trial_idx]), dtype=np.int64),  # (n_bins,)
                wheel_disc[trial_idx].astype(np.int64),  # (n_bins,)
                me_disc[trial_idx].astype(np.int64),  # (n_bins,)
            ], axis=0)  # (4, n_bins)
            session_output.append(out)
        output_list.append(session_output)
        
        # Subject index
        subject_idx_list.append(subject_to_idx[sess['subject']])
        
        # Brain region index
        region_idx = np.array([region_to_idx[r] for r in sess['cluster_acronyms_beryl']], dtype=np.int64)
        brain_region_idx_list.append(region_idx)
    
    data = {
        'neural': neural_list,
        'input': input_list,
        'output': output_list,
        'subjects': all_subjects,
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),
        'brain_regions': all_regions,
        'brain_region_idx': brain_region_idx_list,
        'input_names': ['time_since_stim_onset', 'trial_number_in_block'],
        'output_names': ['choice', 'prior_probability_left', 'wheel_speed', 'whisker_motion_energy'],
        'output_values': [
            ['left', 'right'],  # choice: 0=left, 1=right
            ['0.2', '0.5', '0.8'],  # prior: 0=0.2, 1=0.5, 2=0.8
            ['low', 'medium', 'high'],  # wheel speed bins
            ['low', 'medium', 'high'],  # whisker ME bins
        ],
        'metadata': {
            'task_description': 'IBL Brain-wide Map: mouse performs visual detection task with biased blocks',
            'time_bin_size': BINSIZE * 1000,  # in ms
            'temporal_alignment_event': 'stimulus onset (stimOn_times)',
            'off_start': TIME_WINDOW[0],  # -0.5s before stim onset
            'off_end': TIME_WINDOW[1],  # 1.5s after stim onset
            'binsize_s': BINSIZE,
            'n_bins': N_BINS,
            'align_time': ALIGN_TIME,
            'time_window': TIME_WINDOW,
            'trial_filtering': {
                'min_rt': MIN_RT,
                'max_rt': MAX_RT,
                'max_trial_len': MAX_TRIAL_LEN,
                'exclude_nochoice': EXCLUDE_NOCHOICE,
                'nan_exclude': NAN_EXCLUDE,
            },
            'wheel_speed_quantiles': wheel_quantiles.tolist(),
            'whisker_me_quantiles': me_quantiles.tolist(),
            'source': 'IBL Brain-wide Map',
            'reference_paper': 'IBL et al. 2023, A brain-wide map of neural activity during complex behaviour',
            'methods_paper': 'Zhang et al. 2025, Exploiting correlations across trials and behavioral sessions',
        },
    }
    
    return data


def main():
    parser = argparse.ArgumentParser(description='Convert IBL BWM data to decoder format')
    parser.add_argument('output', type=str, help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    args = parser.parse_args()
    
    if args.sample:
        args.full = False
    
    t_start = time.time()
    
    # Load session list
    bwm_df = pd.read_csv(BWM_CSV, index_col=0)
    print(f'BWM release: {len(bwm_df)} PIDs, {bwm_df.eid.nunique()} sessions, {bwm_df.subject.nunique()} subjects')
    
    # Group by session (eid)
    sessions = {}
    for _, row in bwm_df.iterrows():
        eid = row.eid
        if eid not in sessions:
            sessions[eid] = []
        sessions[eid].append({
            'pid': row.pid,
            'probe_name': row.probe_name,
            'subject': row.subject,
            'lab': row.lab,
            'date': row.date,
        })
    
    # Filter to available sessions
    available_sessions = {}
    for eid, probes in sessions.items():
        sess_path = find_session_path(probes[0]['lab'], probes[0]['subject'], probes[0]['date'])
        if sess_path is not None:
            available_sessions[eid] = probes
    
    print(f'Available sessions: {len(available_sessions)}')
    
    if args.sample:
        # Select 2 sessions for testing
        eids = list(available_sessions.keys())[:2]
        available_sessions = {eid: available_sessions[eid] for eid in eids}
        print(f'Sample mode: processing {len(available_sessions)} sessions')
    
    # Process sessions
    session_results = []
    n_total = len(available_sessions)
    n_failed = 0
    
    for i, (eid, probes) in enumerate(available_sessions.items()):
        print(f'[{i+1}/{n_total}] Processing session {eid}...')
        try:
            result = process_session(eid, probes, show_processing=args.show_processing)
            if result is not None:
                if args.show_processing and len(session_results) < 2:
                    plot_processing(result, len(session_results))
                session_results.append(result)
            else:
                n_failed += 1
        except Exception as e:
            print(f'  Session {eid}: FAILED with error: {e}')
            import traceback
            traceback.print_exc()
            n_failed += 1
    
    print(f'\nProcessed {len(session_results)} sessions successfully, {n_failed} failed')
    
    if len(session_results) == 0:
        print('ERROR: No sessions processed successfully!')
        sys.exit(1)
    
    # Build final dataset
    print('\nBuilding final dataset...')
    data = build_final_dataset(session_results)
    
    # Print summary statistics
    n_sessions = len(data['neural'])
    n_subjects = len(data['subjects'])
    total_trials = sum(len(s) for s in data['neural'])
    total_neurons = sum(data['neural'][s][0].shape[0] for s in range(n_sessions))
    
    print(f'\nDataset Summary:')
    print(f'  Sessions: {n_sessions}')
    print(f'  Subjects: {n_subjects}')
    print(f'  Total trials: {total_trials}')
    print(f'  Total neurons: {total_neurons}')
    print(f'  Mean trials/session: {total_trials/n_sessions:.1f}')
    print(f'  Mean neurons/session: {total_neurons/n_sessions:.1f}')
    print(f'  Brain regions: {data["brain_regions"]}')
    print(f'  Time bins: {N_BINS}')
    print(f'  Input names: {data["input_names"]}')
    print(f'  Output names: {data["output_names"]}')
    
    # Save
    print(f'\nSaving to {args.output}...')
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    
    file_size = os.path.getsize(args.output) / (1024**3)
    print(f'Saved {args.output} ({file_size:.2f} GB)')
    
    t_total = time.time() - t_start
    print(f'Total time: {t_total:.1f}s ({t_total/60:.1f} min)')


if __name__ == '__main__':
    main()
