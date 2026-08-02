#!/usr/bin/env python3
"""Convert Sosa et al. 2025 NWB data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
"""

import sys
import os
import argparse
import time
import glob
import pickle
import numpy as np
import h5py
from collections import defaultdict

# Reward zone definitions from the paper
# A=[80,130], B=[200,250], C=[320,370]
REWARD_ZONES = {
    'A': [80, 130],
    'B': [200, 250],
    'C': [320, 370],
}
TRACK_LENGTH = 450.0
LICK_ERROR_THRESHOLD = 0.35  # from reference code (paper says 0.30)
LICK_SMOOTH_SIGMA = 2  # Gaussian smoothing sigma for licks


def nansmooth(a, sig, axis=-1):
    """Gaussian smoothing with NaN handling (from reference code utilities.py)."""
    if sig == 0:
        return a
    from scipy.ndimage import gaussian_filter1d
    nan_mask = np.isnan(a)
    a_filled = np.copy(a)
    a_filled[nan_mask] = 0
    smoothed = gaussian_filter1d(a_filled, sig, axis=axis)
    # Normalize by the fraction of non-NaN values
    ones = np.ones_like(a)
    ones[nan_mask] = 0
    norm = gaussian_filter1d(ones, sig, axis=axis)
    norm[norm == 0] = 1  # avoid division by zero
    result = smoothed / norm
    result[nan_mask] = np.nan
    return result


def determine_reward_zone_label(rz_start_pos):
    """Determine reward zone label (A/B/C) from start position."""
    if np.isnan(rz_start_pos):
        return None
    for label, (start, end) in REWARD_ZONES.items():
        if start - 20 < rz_start_pos < end + 20:  # allow some tolerance
            return label
    return None


def get_reward_zone_start_per_trial(pos, rzone, trial_start_inds, teleport_inds):
    """Get reward zone start position for each trial."""
    n_trials = len(trial_start_inds)
    rz_starts = np.full(n_trials, np.nan)
    rz_labels = [None] * n_trials
    
    for t in range(n_trials):
        t_start = trial_start_inds[t]
        t_end = teleport_inds[t]
        trial_rzone = rzone[t_start:t_end]
        trial_pos = pos[t_start:t_end]
        
        rz_active = trial_rzone > 0
        if np.any(rz_active):
            # Get the minimum position where rzone is active
            rz_start_pos = trial_pos[rz_active].min()
            rz_starts[t] = rz_start_pos
            rz_labels[t] = determine_reward_zone_label(rz_start_pos)
    
    return rz_starts, rz_labels


def get_reward_zone_for_trial(rz_labels, trial_idx):
    """Get the reward zone label for a specific trial.
    For omission trials, infer from neighboring trials."""
    if rz_labels[trial_idx] is not None:
        return rz_labels[trial_idx]
    
    # Search forward and backward for nearest non-None label
    for offset in range(1, len(rz_labels)):
        if trial_idx - offset >= 0 and rz_labels[trial_idx - offset] is not None:
            return rz_labels[trial_idx - offset]
        if trial_idx + offset < len(rz_labels) and rz_labels[trial_idx + offset] is not None:
            return rz_labels[trial_idx + offset]
    
    return 'A'  # fallback


def compute_distance_to_reward_zone(position, rz_start):
    """Compute signed distance from current position to nearest point in reward zone.
    Reward zone spans [rz_start, rz_start + 50].
    Negative = before reward zone, positive = after."""
    rz_end = rz_start + 50.0
    distance = np.zeros_like(position)
    
    # Before reward zone
    before_mask = position < rz_start
    distance[before_mask] = position[before_mask] - rz_start
    
    # In reward zone
    in_mask = (position >= rz_start) & (position <= rz_end)
    distance[in_mask] = 0.0
    
    # After reward zone
    after_mask = position > rz_end
    distance[after_mask] = position[after_mask] - rz_end
    
    return distance


def discretize_distance(distance):
    """Discretize distance to reward zone into 7 bins.
    0: < -50 cm
    1: -50 to -10 cm
    2: -10 cm to < 0 cm
    3: 0 cm
    4: >0 cm to +10 cm
    5: +10 to +50 cm
    6: > +50 cm
    """
    bins = np.zeros(len(distance), dtype=np.int64)
    bins[distance < -50] = 0
    bins[(distance >= -50) & (distance < -10)] = 1
    bins[(distance >= -10) & (distance < 0)] = 2
    bins[distance == 0] = 3
    bins[(distance > 0) & (distance <= 10)] = 4
    bins[(distance > 10) & (distance <= 50)] = 5
    bins[distance > 50] = 6
    return bins


def discretize_position(position, n_bins=5):
    """Discretize absolute position into n_bins equal-sized bins."""
    bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
    bins = np.digitize(position, bin_edges[1:])  # 0 to n_bins-1
    bins = np.clip(bins, 0, n_bins - 1)
    return bins


def discretize_speed(speed):
    """Discretize speed into 5 bins.
    0: < 2 cm/s
    1: 2-10 cm/s
    2: 10-20 cm/s
    3: 20-40 cm/s
    4: > 40 cm/s
    """
    bins = np.zeros(len(speed), dtype=np.int64)
    bins[speed < 2] = 0
    bins[(speed >= 2) & (speed < 10)] = 1
    bins[(speed >= 10) & (speed < 20)] = 2
    bins[(speed >= 20) & (speed < 40)] = 3
    bins[speed >= 40] = 4
    return bins


def process_licks(lick_data, trial_start_inds, teleport_inds):
    """Process lick data following reference code logic.
    1. Detect and remove lick sensor errors
    2. Clip to [0, 1] (binary)
    3. Smooth with Gaussian
    """
    licks = np.copy(lick_data).astype(np.float64)
    
    # Lick sensor error correction per trial
    for t in range(len(trial_start_inds)):
        t_start = trial_start_inds[t]
        t_end = teleport_inds[t]
        trial_licks = licks[t_start:t_end]
        
        # If >35% of samples have cumulative lick count > 2, set to NaN
        if len(trial_licks) > 0:
            frac_high = np.sum(trial_licks > 2) / len(trial_licks)
            if frac_high > LICK_ERROR_THRESHOLD:
                licks[t_start:t_end] = np.nan
    
    # Clip to [0, 1]
    licks[licks > 1] = 1
    
    return licks


def process_session(nwb_path, show_processing=False, session_idx=0):
    """Process a single NWB file and return trial-level data."""
    t0 = time.time()
    
    f = h5py.File(nwb_path, 'r')
    
    # Extract metadata
    subj_id = f['general']['subject']['subject_id'][()].decode()
    sess_id = f['general']['session_id'][()].decode()
    
    # Extract behavioral data
    beh = f['processing']['behavior']['BehavioralTimeSeries']
    position = beh['position']['data'][:]
    speed_data = beh['speed']['data'][:]
    lick_data = beh['lick']['data'][:]
    rzone = beh['reward_zone']['data'][:]
    env_data = beh['environment']['data'][:]
    trial_start_markers = beh['trial_start']['data'][:]
    teleport_markers = beh['teleport']['data'][:]
    scanning = beh['scanning']['data'][:]
    reward_timestamps = beh['Reward']['timestamps'][:]
    pos_timestamps = beh['position']['timestamps'][:]
    
    # Extract neural data (handle multi-plane animals like m17, m18)
    deconv_group = f['processing']['ophys']['Deconvolved']
    plane_keys = sorted([k for k in deconv_group.keys() if k.startswith('plane')])
    if len(plane_keys) == 1:
        deconv_data = deconv_group[plane_keys[0]]['data'][:]  # (n_timepoints, n_rois)
    else:
        # Concatenate planes along ROI axis
        plane_data = [deconv_group[pk]['data'][:] for pk in plane_keys]
        deconv_data = np.concatenate(plane_data, axis=1)
        print(f"  Multi-plane: {len(plane_keys)} planes, {[d.shape[1] for d in plane_data]} ROIs")
    iscell = f['processing']['ophys']['ImageSegmentation']['PlaneSegmentation']['iscell'][:]
    
    f.close()
    
    # Filter neurons by iscell
    cell_mask = iscell[:, 0] == 1
    neural_all = deconv_data[:, cell_mask].T  # (n_neurons, n_timepoints)
    n_neurons = neural_all.shape[0]
    n_timepoints = neural_all.shape[1]
    
    # Get trial boundaries
    trial_start_inds = np.where(trial_start_markers > 0)[0]
    teleport_inds = np.where(teleport_markers > 0)[0]
    n_trials = len(trial_start_inds)
    
    if n_trials != len(teleport_inds):
        print(f"  WARNING: trial_start ({n_trials}) != teleport ({len(teleport_inds)}) count")
        n_trials = min(n_trials, len(teleport_inds))
        trial_start_inds = trial_start_inds[:n_trials]
        teleport_inds = teleport_inds[:n_trials]
    
    # Compute frame rate
    dt = np.median(np.diff(pos_timestamps))
    
    # Get reward zone info per trial
    rz_starts, rz_labels = get_reward_zone_start_per_trial(
        position, rzone, trial_start_inds, teleport_inds)
    
    # Process licks
    licks_processed = process_licks(lick_data, trial_start_inds, teleport_inds)
    
    # Determine reward per trial
    # Convert reward timestamps to frame indices
    reward_frame_inds = np.searchsorted(pos_timestamps, reward_timestamps)
    reward_frame_inds = np.clip(reward_frame_inds, 0, n_timepoints - 1)
    
    # Build per-trial reward binary
    trial_rewarded = np.zeros(n_trials, dtype=np.int64)
    for t in range(n_trials):
        t_start = trial_start_inds[t]
        t_end = teleport_inds[t]
        # Check if any reward was delivered in this trial
        reward_in_trial = np.any(
            (reward_frame_inds >= t_start) & (reward_frame_inds < t_end)
        )
        trial_rewarded[t] = 1 if reward_in_trial else 0
    
    # Get environment type per trial
    trial_env = np.zeros(n_trials, dtype=np.int64)
    for t in range(n_trials):
        t_start = trial_start_inds[t]
        t_end = teleport_inds[t]
        env_vals = env_data[t_start:t_end]
        env_vals = env_vals[env_vals >= 0]  # exclude -1
        if len(env_vals) > 0:
            trial_env[t] = int(np.median(env_vals))
    
    # Get reward zone label per trial (for output)
    trial_rz_label = np.zeros(n_trials, dtype=np.int64)
    rz_label_map = {'A': 0, 'B': 1, 'C': 2}
    for t in range(n_trials):
        label = get_reward_zone_for_trial(rz_labels, t)
        trial_rz_label[t] = rz_label_map.get(label, 0)
    
    # Get reward zone start position per trial (for distance computation)
    # For omission trials, use the same reward zone as neighboring trials
    trial_rz_start = np.zeros(n_trials)
    for t in range(n_trials):
        label = get_reward_zone_for_trial(rz_labels, t)
        trial_rz_start[t] = REWARD_ZONES[label][0]
    
    # Build trial data
    neural_trials = []
    input_trials = []
    output_trials = []
    
    for t in range(n_trials):
        t_start = trial_start_inds[t]
        t_end = teleport_inds[t]
        n_tp = t_end - t_start
        
        if n_tp < 2:
            continue
        
        # Neural data for this trial
        trial_neural = neural_all[:, t_start:t_end].astype(np.float32)
        
        # --- INPUTS ---
        # Time from start of trial (seconds)
        time_from_start = np.arange(n_tp) * dt
        
        # Environment type (binary, per trial)
        env_type = float(trial_env[t])
        
        # Trial number (continuous, per trial)
        trial_num = float(t)
        
        # Previous trial outcome (binary, per trial)
        if t == 0:
            prev_outcome = 0.0
        else:
            prev_outcome = float(trial_rewarded[t - 1])
        
        # Build input array
        # Time-varying: time_from_start (n_tp,)
        # Per-trial: env_type, trial_num, prev_outcome
        input_data = np.zeros((4, n_tp), dtype=np.float32)
        input_data[0, :] = time_from_start
        input_data[1, :] = env_type
        input_data[2, :] = trial_num
        input_data[3, :] = prev_outcome
        
        # --- OUTPUTS ---
        trial_pos = position[t_start:t_end]
        trial_speed = speed_data[t_start:t_end]
        trial_licks = licks_processed[t_start:t_end]
        
        # Distance to reward zone (time-varying, discretized)
        rz_start = trial_rz_start[t]
        distance = compute_distance_to_reward_zone(trial_pos, rz_start)
        dist_bins = discretize_distance(distance)
        
        # Absolute position (time-varying, discretized into 5 bins)
        pos_bins = discretize_position(trial_pos, n_bins=5)
        
        # Speed (time-varying, discretized)
        speed_bins = discretize_speed(np.abs(trial_speed))  # use absolute speed
        
        # Lick (time-varying, binary)
        lick_binary = np.zeros(n_tp, dtype=np.int64)
        if not np.all(np.isnan(trial_licks)):
            # Smooth licks
            smoothed_licks = nansmooth(trial_licks, LICK_SMOOTH_SIGMA)
            # Convert to binary: > 0.5 threshold after smoothing
            lick_binary = (smoothed_licks > 0.5).astype(np.int64)
            lick_binary[np.isnan(smoothed_licks)] = 0
        
        # Reward zone location (per-trial)
        rz_loc = trial_rz_label[t]
        
        # Reward outcome (per-trial)
        reward_out = trial_rewarded[t]
        
        # Build output array
        # Time-varying: dist_bins, pos_bins, speed_bins, lick_binary
        # Per-trial: rz_loc, reward_out
        output_data = np.zeros((6, n_tp), dtype=np.int64)
        output_data[0, :] = dist_bins
        output_data[1, :] = pos_bins
        output_data[2, :] = speed_bins
        output_data[3, :] = lick_binary
        output_data[4, :] = rz_loc
        output_data[5, :] = reward_out
        
        neural_trials.append(trial_neural)
        input_trials.append(input_data)
        output_trials.append(output_data)
    
    elapsed = time.time() - t0
    print(f"  {subj_id} ses-{sess_id}: {n_neurons} neurons, {len(neural_trials)} trials, "
          f"{n_tp} timepoints (last trial), dt={dt*1000:.1f}ms, {elapsed:.1f}s")
    
    if show_processing and session_idx < 2:
        plot_processing(subj_id, sess_id, position, speed_data, licks_processed,
                       trial_start_inds, teleport_inds, neural_all, 
                       rz_starts, rz_labels, trial_rewarded, trial_env,
                       pos_timestamps, dt, neural_trials, input_trials, output_trials,
                       session_idx)
    
    return {
        'neural_trials': neural_trials,
        'input_trials': input_trials,
        'output_trials': output_trials,
        'subj_id': subj_id,
        'sess_id': sess_id,
        'n_neurons': n_neurons,
        'dt': dt,
    }


def plot_processing(subj_id, sess_id, position, speed, licks,
                   trial_start_inds, teleport_inds, neural_all,
                   rz_starts, rz_labels, trial_rewarded, trial_env,
                   pos_timestamps, dt, neural_trials, input_trials, output_trials,
                   session_idx):
    """Plot processing steps for visual verification."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(6, 2, figsize=(20, 24))
    fig.suptitle(f'{subj_id} ses-{sess_id}', fontsize=16)
    
    n_trials = len(trial_start_inds)
    
    # Plot 1: Position over time for first 10 trials
    ax = axes[0, 0]
    for t in range(min(10, n_trials)):
        t_start = trial_start_inds[t]
        t_end = teleport_inds[t]
        ts = pos_timestamps[t_start:t_end] - pos_timestamps[t_start]
        ax.plot(ts, position[t_start:t_end], alpha=0.7, label=f'T{t}')
    ax.set_xlabel('Time from trial start (s)')
    ax.set_ylabel('Position (cm)')
    ax.set_title('Position traces (first 10 trials)')
    ax.legend(fontsize=6)
    
    # Plot 2: Speed distribution
    ax = axes[0, 1]
    valid_speed = speed[speed > -100]
    ax.hist(valid_speed, bins=100, range=(-5, 80))
    ax.axvline(2, color='r', linestyle='--', label='Speed threshold')
    ax.set_xlabel('Speed (cm/s)')
    ax.set_title('Speed distribution')
    ax.legend()
    
    # Plot 3: Neural activity heatmap (first trial)
    ax = axes[1, 0]
    if len(neural_trials) > 0:
        trial_data = neural_trials[0]
        n_show = min(50, trial_data.shape[0])
        ax.imshow(trial_data[:n_show, :], aspect='auto', cmap='hot')
        ax.set_xlabel('Timepoints')
        ax.set_ylabel('Neuron')
        ax.set_title('Neural activity (Trial 0, first 50 neurons)')
    
    # Plot 4: Reward zone positions across trials
    ax = axes[1, 1]
    colors = {'A': 'red', 'B': 'blue', 'C': 'green', None: 'gray'}
    for t in range(n_trials):
        c = colors.get(rz_labels[t], 'gray')
        ax.scatter(t, rz_starts[t], c=c, s=10)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Reward zone start (cm)')
    ax.set_title('Reward zone locations')
    
    # Plot 5: Distance to reward zone (first trial)
    ax = axes[2, 0]
    if len(output_trials) > 0:
        ax.plot(output_trials[0][0, :], label='Distance bin')
        ax.set_xlabel('Timepoints')
        ax.set_ylabel('Distance bin')
        ax.set_title('Distance to RZ (Trial 0)')
    
    # Plot 6: Position bins (first trial)
    ax = axes[2, 1]
    if len(output_trials) > 0:
        ax.plot(output_trials[0][1, :], label='Position bin')
        ax.set_xlabel('Timepoints')
        ax.set_ylabel('Position bin')
        ax.set_title('Position bins (Trial 0)')
    
    # Plot 7: Speed bins (first trial)
    ax = axes[3, 0]
    if len(output_trials) > 0:
        ax.plot(output_trials[0][2, :], label='Speed bin')
        ax.set_xlabel('Timepoints')
        ax.set_ylabel('Speed bin')
        ax.set_title('Speed bins (Trial 0)')
    
    # Plot 8: Lick (first trial)
    ax = axes[3, 1]
    if len(output_trials) > 0:
        ax.plot(output_trials[0][3, :], label='Lick')
        ax.set_xlabel('Timepoints')
        ax.set_ylabel('Lick')
        ax.set_title('Lick (Trial 0)')
    
    # Plot 9: Input time from start (first trial)
    ax = axes[4, 0]
    if len(input_trials) > 0:
        ax.plot(input_trials[0][0, :], label='Time from start')
        ax.set_xlabel('Timepoints')
        ax.set_ylabel('Time (s)')
        ax.set_title('Time from trial start (Trial 0)')
    
    # Plot 10: Trial reward and environment
    ax = axes[4, 1]
    ax.bar(range(n_trials), trial_rewarded, alpha=0.5, label='Rewarded')
    ax.bar(range(n_trials), trial_env * 0.5, alpha=0.5, label='Env (scaled)')
    ax.set_xlabel('Trial')
    ax.set_title('Trial reward and environment')
    ax.legend()
    
    # Plot 11: Neural activity summary
    ax = axes[5, 0]
    mean_activity = np.nanmean(neural_all, axis=1)
    ax.hist(mean_activity, bins=50)
    ax.set_xlabel('Mean activity')
    ax.set_title('Distribution of mean neural activity per neuron')
    
    # Plot 12: Output distributions
    ax = axes[5, 1]
    if len(output_trials) > 5:
        all_dist = np.concatenate([o[0, :] for o in output_trials])
        all_pos = np.concatenate([o[1, :] for o in output_trials])
        all_spd = np.concatenate([o[2, :] for o in output_trials])
        all_lck = np.concatenate([o[3, :] for o in output_trials])
        ax.bar([0,1,2,3,4,5,6], [np.mean(all_dist==i) for i in range(7)], alpha=0.5, label='Dist')
        ax.set_xlabel('Bin')
        ax.set_title('Distance bin distribution')
    
    plt.tight_layout()
    plt.savefig(f'processing_{subj_id}_ses{sess_id}.png', dpi=100)
    plt.close()
    print(f"  Saved processing plot: processing_{subj_id}_ses{sess_id}.png")


def main():
    parser = argparse.ArgumentParser(description='Convert NWB data to decoder format')
    parser.add_argument('output', help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--show-processing', action='store_true', help='Show processing plots')
    args = parser.parse_args()
    
    if args.sample:
        args.full = False
    
    print("=" * 60)
    print("NWB to Decoder Format Conversion")
    print("=" * 60)
    
    # Find all NWB files
    nwb_files = sorted(glob.glob('data/sub-*/sub-*_ses-*_behavior+ophys.nwb'))
    print(f"Found {len(nwb_files)} NWB files")
    
    if args.sample:
        # Select 2 sessions from different subjects
        nwb_files = [nwb_files[0], nwb_files[14]]  # m11 ses-03 and m12 ses-01
        print(f"Sample mode: processing {len(nwb_files)} sessions")
    
    # Process all sessions
    all_neural = []
    all_input = []
    all_output = []
    all_subj_ids = []
    all_sess_ids = []
    all_n_neurons = []
    all_dts = []
    
    total_t0 = time.time()
    
    for i, nwb_file in enumerate(nwb_files):
        print(f"\nProcessing [{i+1}/{len(nwb_files)}]: {os.path.basename(nwb_file)}")
        result = process_session(nwb_file, show_processing=args.show_processing, session_idx=i)
        
        if len(result['neural_trials']) < 2:
            print(f"  SKIPPING: fewer than 2 trials")
            continue
        
        all_neural.append(result['neural_trials'])
        all_input.append(result['input_trials'])
        all_output.append(result['output_trials'])
        all_subj_ids.append(result['subj_id'])
        all_sess_ids.append(result['sess_id'])
        all_n_neurons.append(result['n_neurons'])
        all_dts.append(result['dt'])
    
    total_elapsed = time.time() - total_t0
    print(f"\nTotal processing time: {total_elapsed:.1f}s")
    
    # Build subjects list
    unique_subjects = sorted(set(all_subj_ids), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
    subject_idx = np.array([unique_subjects.index(s) for s in all_subj_ids])
    
    # Build brain region info
    brain_regions = ['CA1']
    brain_region_idx = []
    for n_neurons in all_n_neurons:
        brain_region_idx.append(np.zeros(n_neurons, dtype=np.int64))  # all CA1
    
    # Compute median dt for metadata
    median_dt = np.median(all_dts)
    
    # Build the data dictionary
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        
        'subjects': unique_subjects,
        'subject_idx': subject_idx,
        
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        
        'input_names': [
            'time_from_trial_start',
            'environment_type',
            'trial_number',
            'previous_trial_outcome',
        ],
        
        'output_names': [
            'distance_to_reward_zone',
            'absolute_position',
            'speed',
            'lick',
            'reward_zone_location',
            'reward_outcome',
        ],
        
        'output_values': [
            ['< -50cm', '-50 to -10cm', '-10 to 0cm', '0cm', '0 to +10cm', '+10 to +50cm', '> +50cm'],
            ['0-90cm', '90-180cm', '180-270cm', '270-360cm', '360-450cm'],
            ['< 2cm/s', '2-10cm/s', '10-20cm/s', '20-40cm/s', '> 40cm/s'],
            ['no_lick', 'lick'],
            ['zone_A', 'zone_B', 'zone_C'],
            ['no_reward', 'reward'],
        ],
        
        'metadata': {
            'task_description': 'Hidden reward zone task on 450cm virtual linear track with 2 environments and 3 possible reward zones (A/B/C). Mice run laps and lick to receive reward. Reward zone switches within session.',
            'time_bin_size': median_dt * 1000,  # in ms
            'temporal_alignment_event': 'trial_start (entry to linear track)',
            'off_start': 0.0,
            'off_end': None,  # variable trial lengths
            'dataset': 'Sosa et al. 2025, Nature Neuroscience',
            'brain_region': 'Hippocampal CA1',
            'species': 'Mus musculus',
            'imaging_modality': '2-photon calcium imaging',
            'neural_signal': 'Deconvolved calcium events (OASIS)',
            'frame_rate_hz': 1.0 / median_dt,
            'track_length_cm': TRACK_LENGTH,
            'reward_zones': REWARD_ZONES,
            'n_sessions': len(all_neural),
            'n_subjects': len(unique_subjects),
        },
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Sessions: {len(all_neural)}")
    print(f"Subjects: {unique_subjects}")
    print(f"Time bin: {median_dt*1000:.2f} ms")
    total_trials = sum(len(s) for s in all_neural)
    total_neurons = sum(n for n in all_n_neurons)
    print(f"Total trials: {total_trials}")
    print(f"Total neurons: {total_neurons}")
    print(f"Neurons per session: {[n for n in all_n_neurons]}")
    
    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    
    file_size = os.path.getsize(args.output) / (1024**2)
    print(f"Saved: {file_size:.1f} MB")
    print("Done!")


if __name__ == '__main__':
    main()
