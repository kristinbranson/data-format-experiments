#!/usr/bin/env python3
"""Convert Sosa et al. 2025 NWB data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
"""

import os
import sys
import glob
import time
import pickle
import argparse
import numpy as np
import scipy.ndimage as ndimage
from pynwb import NWBHDF5IO

# ============================================================
# Helper functions (adapted from reference code)
# ============================================================

def nansmooth(data, sigma_val, axis=-1):
    """Smooth data with Gaussian, handling NaNs.
    Adapted from TwoPUtils.utilities.nansmooth
    
    Parameters:
    -----------
    data : array
    sigma_val : float - standard deviation of Gaussian kernel
    axis : int - axis to smooth along
    """
    if sigma_val <= 0:
        return data.copy()
    
    # Handle NaNs
    nan_mask = np.isnan(data)
    data_filled = np.copy(data)
    data_filled[nan_mask] = 0
    
    smoothed = ndimage.gaussian_filter1d(data_filled, sigma=float(sigma_val), axis=axis)
    
    # Normalize by non-NaN weight
    weight = np.ones_like(data, dtype=np.float64)
    weight[nan_mask] = 0
    weight_smooth = ndimage.gaussian_filter1d(weight, sigma=float(sigma_val), axis=axis)
    weight_smooth[weight_smooth == 0] = 1  # avoid division by zero
    
    result = smoothed / weight_smooth
    result[nan_mask] = np.nan
    
    return result


def compute_dff(F, Fneu, trial_start_inds, teleport_inds, 
                neu_coef=0.7, frame_rate=15.5078125):
    """Compute deltaF/F following the reference code preprocessing pipeline.
    
    Steps:
    1. Neuropil subtraction: F_corrected = F - neu_coef * Fneu
    2. Per-trial baseline: maximin method (smooth, min filter 20s, max filter 20s)
    3. Add back neuropil mean per trial
    4. dFF = (F_corrected - baseline) / |baseline|
    5. Smooth with 2-sample Gaussian
    
    Parameters:
    -----------
    F : array (n_cells, n_timepoints) - raw fluorescence
    Fneu : array (n_cells, n_timepoints) - neuropil fluorescence
    trial_start_inds : array - indices of trial starts
    teleport_inds : array - indices of teleports (trial ends)
    
    Returns:
    --------
    dff : array (n_cells, n_timepoints) - deltaF/F
    """
    n_cells, n_timepoints = F.shape
    
    # Step 1: Neuropil subtraction
    f_ = F - neu_coef * Fneu
    
    # Initialize baseline and dFF arrays
    flow = np.full_like(f_, np.nan)
    dff = np.full_like(f_, np.nan)
    
    # Use trial starts and teleports as start/stop indices
    start_inds = trial_start_inds
    stop_inds = teleport_inds
    
    # Ensure we have matching pairs
    n_trials = min(len(start_inds), len(stop_inds))
    
    window_size = int(20 * frame_rate)  # 20 seconds in frames (~300 frames)
    
    for i in range(n_trials):
        start = start_inds[i]
        stop = stop_inds[i]
        
        if stop <= start:
            continue
        
        # Add back neuropil mean per trial (as in reference code)
        f_[:, start:stop] = f_[:, start:stop] + neu_coef * np.nanmean(
            Fneu[:, start:stop], axis=1, keepdims=True)
        
        # Compute baseline using maximin method
        # Step 1: Smooth with 15-sample Gaussian
        flow[:, start:stop] = nansmooth(f_[:, start:stop], 15, axis=1)
        
        # Step 2: Minimum filter (20s window)
        flow[:, start:stop] = ndimage.minimum_filter1d(
            flow[:, start:stop], window_size, axis=1)
        
        # Step 3: Maximum filter (20s window) - dilation
        flow[:, start:stop] = ndimage.maximum_filter1d(
            flow[:, start:stop], window_size, axis=1)
    
    # Compute dFF
    valid_mask = ~np.isnan(flow) & (np.abs(flow) > 0)
    dff[valid_mask] = (f_[valid_mask] - flow[valid_mask]) / np.abs(flow[valid_mask])
    
    # Smooth dFF with 2-sample Gaussian per trial
    for i in range(n_trials):
        start = start_inds[i]
        stop = stop_inds[i]
        if stop <= start:
            continue
        dff[:, start:stop] = nansmooth(dff[:, start:stop], 2, axis=1)
    
    return dff


def parse_scene_name(identifier):
    """Extract scene name from NWB identifier.
    
    Example identifiers:
    /data/InVivoDA/GCAMP3/03_10_2022/Env1_LocationC_to_A
    /data/InVivoDA/GCAMP11/23_02_2023/Env1_LocationC
    """
    scene = identifier.split('/')[-1]
    return scene


def get_reward_zone_info(scene):
    """Get reward zone coordinates and labels from scene name.
    
    Returns:
    --------
    rz_info : list of (rz_coords, rz_label, change_trial_or_None)
        For non-switch sessions: [(coords, label, None)]
        For switch sessions: [(coords1, label1, 30), (coords2, label2, None)]
    """
    rz_dict = {
        'A': [80, 130],
        'B': [200, 250],
        'C': [320, 370],
    }
    
    # Map scene location names to reward zone labels
    loc_to_label = {
        'LocationA': 'A',
        'LocationB': 'B', 
        'LocationC': 'C',
    }
    
    # Check if it's a switch session
    if '_to_' in scene:
        # Could be within-env switch: Env1_LocationA_to_B
        # Or cross-env switch: Env1_A_to_Env2_B
        
        if 'Env' in scene.split('_to_')[1] and 'Location' not in scene.split('_to_')[1].split('Env')[0]:
            # Cross-environment switch: Env1_A_to_Env2_B
            parts = scene.split('_to_')
            # Before switch
            before_loc = parts[0].split('_')[-1]  # e.g., 'A' or 'B' or 'C'
            # After switch  
            after_loc = parts[1].split('_')[-1]  # e.g., 'A' or 'B' or 'C'
        else:
            # Within-env switch: Env1_LocationA_to_B or Env1_LocationC_to_A
            parts = scene.split('_to_')
            # Before: extract location letter
            for loc_name, label in loc_to_label.items():
                if loc_name in parts[0]:
                    before_loc = label
                    break
            # After: just the letter
            after_loc = parts[1]
        
        return [
            (rz_dict[before_loc], before_loc, 30),
            (rz_dict[after_loc], after_loc, None),
        ]
    else:
        # Non-switch session
        for loc_name, label in loc_to_label.items():
            if loc_name in scene:
                return [(rz_dict[label], label, None)]
        
        # Training or unknown
        return [(None, None, None)]


def get_environment(scene):
    """Get environment type from scene name. Returns 0 for ENV1, 1 for ENV2."""
    if '_to_' in scene and 'Env2' in scene.split('_to_')[1]:
        # Cross-env switch: first part is Env1, second is Env2
        return None  # Will be determined per trial
    elif 'Env2' in scene:
        return 1
    else:
        return 0


def get_reward_zone_label_per_trial(scene, n_trials, change_trial=30):
    """Get reward zone label for each trial.
    
    Returns:
    --------
    rz_labels : array of str, shape (n_trials,)
    rz_coords : array of shape (n_trials, 2) - [start, end] of reward zone
    """
    rz_dict = {'A': [80, 130], 'B': [200, 250], 'C': [320, 370]}
    
    rz_info = get_reward_zone_info(scene)
    
    if rz_info[0][0] is None:
        return None, None
    
    rz_labels = np.empty(n_trials, dtype='U1')
    rz_coords = np.zeros((n_trials, 2))
    
    if len(rz_info) == 1:
        # Non-switch session
        coords, label, _ = rz_info[0]
        rz_labels[:] = label
        rz_coords[:] = coords
    else:
        # Switch session
        coords1, label1, ct = rz_info[0]
        coords2, label2, _ = rz_info[1]
        ct = ct if ct is not None else change_trial
        
        rz_labels[:ct] = label1
        rz_labels[ct:] = label2
        rz_coords[:ct] = coords1
        rz_coords[ct:] = coords2
    
    return rz_labels, rz_coords


def get_env_per_trial(scene, n_trials, change_trial=30):
    """Get environment label per trial.
    
    Returns:
    --------
    env_labels : array of int, shape (n_trials,) - 0=ENV1, 1=ENV2
    """
    env_labels = np.zeros(n_trials, dtype=int)
    
    if '_to_' in scene and 'Env2' in scene.split('_to_')[1] and 'Env1' in scene.split('_to_')[0]:
        # Cross-env switch: Env1 before, Env2 after
        env_labels[:change_trial] = 0
        env_labels[change_trial:] = 1
    elif '_to_' in scene and 'Env1' in scene.split('_to_')[1] and 'Env2' in scene.split('_to_')[0]:
        # Reverse cross-env switch (shouldn't happen in this dataset)
        env_labels[:change_trial] = 1
        env_labels[change_trial:] = 0
    elif 'Env2' in scene:
        env_labels[:] = 1
    else:
        env_labels[:] = 0
    
    return env_labels


def compute_distance_to_reward_zone(position, rz_start, rz_end):
    """Compute signed distance from position to nearest point in reward zone.
    
    Negative = before reward zone, Positive = past reward zone, 0 = inside zone.
    """
    distance = np.zeros_like(position)
    
    before_zone = position < rz_start
    in_zone = (position >= rz_start) & (position <= rz_end)
    after_zone = position > rz_end
    
    distance[before_zone] = position[before_zone] - rz_start
    distance[in_zone] = 0.0
    distance[after_zone] = position[after_zone] - rz_end
    
    return distance


def discretize_distance(distance):
    """Discretize distance to reward zone into 7 bins.
    0: < -50 cm
    1: -50 to -10 cm
    2: -10 cm to < 0 cm
    3: 0 cm (in reward zone)
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


def discretize_position(position):
    """Discretize position into 5 equal bins spanning 450 cm.
    0: < 90 cm
    1: 90 to 180 cm
    2: 180 to 270 cm
    3: 270 to 360 cm
    4: > 360 cm
    """
    bins = np.zeros(len(position), dtype=np.int64)
    bins[position < 90] = 0
    bins[(position >= 90) & (position < 180)] = 1
    bins[(position >= 180) & (position < 270)] = 2
    bins[(position >= 270) & (position < 360)] = 3
    bins[position >= 360] = 4
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


def determine_reward_per_trial(reward_timestamps, trial_start_times, trial_end_times):
    """Determine if reward was delivered on each trial.
    
    Returns:
    --------
    rewarded : array of int, shape (n_trials,) - 0=no, 1=yes
    """
    n_trials = len(trial_start_times)
    rewarded = np.zeros(n_trials, dtype=int)
    
    for i in range(n_trials):
        t_start = trial_start_times[i]
        t_end = trial_end_times[i]
        # Check if any reward was delivered during this trial
        mask = (reward_timestamps >= t_start) & (reward_timestamps <= t_end)
        if np.any(mask):
            rewarded[i] = 1
    
    return rewarded


# ============================================================
# Main conversion function
# ============================================================

def process_session(nwb_file, show_processing=False, session_idx=0):
    """Process a single NWB file and return trial-segmented data.
    
    Returns:
    --------
    result : dict with keys 'neural', 'input', 'output', 'n_neurons', 'subject', 'scene', 'session_id'
    """
    t0 = time.time()
    
    with NWBHDF5IO(nwb_file, 'r') as io:
        nwb = io.read()
        
        # === Extract metadata ===
        subject_id = nwb.subject.subject_id  # e.g., 'm3'
        session_id = nwb.session_id  # e.g., '03'
        identifier = nwb.identifier
        scene = parse_scene_name(identifier)
        frame_rate = nwb.processing['ophys']['Fluorescence']['plane0'].rate
        
        print(f"  Processing {subject_id} ses-{session_id} ({scene})")
        
        # === Load neural data ===
        t1 = time.time()
        
        # Load iscell and planeIdx for filtering
        ps = nwb.processing['ophys']['ImageSegmentation']['PlaneSegmentation']
        iscell = ps['iscell'][:]  # (n_rois, 2)
        plane_idx = ps['planeIdx'][:]  # (n_rois,)
        
        # Determine number of planes
        planes = sorted(list(nwb.processing['ophys']['Fluorescence'].roi_response_series.keys()))
        n_planes = len(planes)
        
        # Load and concatenate data from all planes, filtering by iscell
        F_list = []
        Fneu_list = []
        
        for plane_name in planes:
            plane_num = int(plane_name.replace('plane', ''))
            
            # Get ROIs for this plane
            plane_mask = plane_idx == plane_num
            plane_iscell = iscell[plane_mask, 0] == 1
            
            # Load fluorescence data for this plane
            F_plane = nwb.processing['ophys']['Fluorescence'][plane_name].data[:]  # (n_timepoints, n_rois_plane)
            Fneu_plane = nwb.processing['ophys']['Neuropil'][plane_name].data[:]
            
            # Filter to curated cells
            F_list.append(F_plane[:, plane_iscell].T.astype(np.float64))
            Fneu_list.append(Fneu_plane[:, plane_iscell].T.astype(np.float64))
        
        # Concatenate across planes: (n_cells_total, n_timepoints)
        F = np.concatenate(F_list, axis=0)
        Fneu = np.concatenate(Fneu_list, axis=0)
        
        n_rois = sum(np.sum(plane_idx == int(p.replace('plane', ''))) for p in planes)
        n_cells = F.shape[0]
        print(f"    ROIs: {n_rois}, Cells (iscell): {n_cells}, Planes: {n_planes}")
        
        print(f"    Neural data loaded: {time.time()-t1:.1f}s")
        
        # === Load behavioral data ===
        t1 = time.time()
        bts = nwb.processing['behavior']['BehavioralTimeSeries']
        
        position = bts.time_series['position'].data[:].astype(np.float64)
        speed = bts.time_series['speed'].data[:].astype(np.float64)
        lick = bts.time_series['lick'].data[:].astype(np.float64)
        trial_num = bts.time_series['trial number'].data[:].astype(np.float64)
        tstart = bts.time_series['trial_start'].data[:].astype(np.float64)
        teleport = bts.time_series['teleport'].data[:].astype(np.float64)
        timestamps = bts.time_series['position'].timestamps[:].astype(np.float64)
        
        # Reward delivery timestamps
        reward_ts_obj = bts.time_series['Reward']
        reward_timestamps = reward_ts_obj.timestamps[:]
        
        print(f"    Behavioral data loaded: {time.time()-t1:.1f}s")
        
        # === Find trial boundaries ===
        trial_start_inds = np.where(tstart == 1)[0]
        teleport_inds = np.where(teleport == 1)[0]
        
        n_trials = len(trial_start_inds)
        print(f"    Trials: {n_trials}")
        
        if n_trials < 2:
            print(f"    WARNING: Only {n_trials} trials, skipping session")
            return None
        
        # Match trial starts with teleports
        # Each trial goes from trial_start to the next teleport
        trial_ends = np.zeros(n_trials, dtype=int)
        for i in range(n_trials):
            # Find first teleport after this trial start
            later_teleports = teleport_inds[teleport_inds > trial_start_inds[i]]
            if len(later_teleports) > 0:
                trial_ends[i] = later_teleports[0]
            else:
                # Last trial - use end of data
                trial_ends[i] = len(position) - 1
        
        # === Compute dFF ===
        t1 = time.time()
        dff = compute_dff(F, Fneu, trial_start_inds, trial_ends, 
                         neu_coef=0.7, frame_rate=frame_rate)
        print(f"    dFF computed: {time.time()-t1:.1f}s")
        
        # === Interneuron exclusion ===
        # Exclude cells with Pearson correlation > 0.5 between dFF and speed
        t1 = time.time()
        
        # Handle potential length mismatch between neural and behavioral data
        n_neural_frames = dff.shape[1]
        n_behav_frames = len(speed)
        n_common = min(n_neural_frames, n_behav_frames)
        
        valid_frames = ~np.isnan(dff[0, :n_common])  # frames where dFF is valid
        interneuron_mask = np.ones(n_cells, dtype=bool)  # True = keep
        
        if np.sum(valid_frames) > 100:
            speed_valid = speed[:n_common][valid_frames]
            for c in range(n_cells):
                dff_valid = dff[c, :n_common][valid_frames]
                if np.std(dff_valid) > 0 and np.std(speed_valid) > 0:
                    corr = np.corrcoef(dff_valid, speed_valid)[0, 1]
                    if not np.isnan(corr) and corr > 0.5:
                        interneuron_mask[c] = False
        
        n_interneurons = np.sum(~interneuron_mask)
        n_final_cells = np.sum(interneuron_mask)
        print(f"    Interneurons excluded: {n_interneurons}, Final cells: {n_final_cells}")
        
        # Apply interneuron mask
        dff = dff[interneuron_mask, :]
        
        print(f"    Interneuron filtering: {time.time()-t1:.1f}s")
        
        # === Get reward zone info ===
        rz_labels, rz_coords = get_reward_zone_label_per_trial(scene, n_trials)
        if rz_labels is None:
            print(f"    WARNING: Could not determine reward zone, skipping session")
            return None
        
        # === Get environment per trial ===
        env_per_trial = get_env_per_trial(scene, n_trials)
        
        # === Determine reward per trial ===
        trial_start_times = timestamps[trial_start_inds]
        trial_end_times = timestamps[trial_ends]
        rewarded = determine_reward_per_trial(reward_timestamps, trial_start_times, trial_end_times)
        
        # === Determine previous trial outcome ===
        prev_reward = np.zeros(n_trials, dtype=int)
        prev_reward[1:] = rewarded[:-1]  # First trial has no previous, default to 0
        
        # === Segment data by trial ===
        t1 = time.time()
        neural_trials = []
        input_trials = []
        output_trials = []
        
        dt = 1.0 / frame_rate  # time per frame in seconds
        
        for i in range(n_trials):
            start = trial_start_inds[i]
            end = trial_ends[i]  # teleport index
            
            # Use frames from trial_start up to (but not including) teleport
            # The teleport frame marks the end of the track
            # Handle potential length mismatch
            end = min(end, dff.shape[1], len(position))
            n_frames = end - start  # don't include teleport frame itself
            
            if n_frames < 2:
                continue
            
            # --- Neural data ---
            trial_neural = dff[:, start:end].astype(np.float32)  # (n_cells, n_frames)
            
            # Replace NaNs with 0
            trial_neural = np.nan_to_num(trial_neural, nan=0.0)
            
            neural_trials.append(trial_neural)
            
            # --- Input data ---
            # Time from trial start (seconds)
            time_from_start = (timestamps[start:end] - timestamps[start]).astype(np.float32)
            
            # Environment (per trial, but we broadcast)
            env_val = float(env_per_trial[i])
            
            # Trial number
            trial_number = float(i)
            
            # Previous trial outcome
            prev_outcome = float(prev_reward[i])
            
            # Input: (4, n_frames) for time-varying, or (4,) for per-trial
            # time_from_start is time-varying, others are per-trial
            # We'll make time_from_start (1, n_frames) and others (1,)
            input_time_varying = time_from_start.reshape(1, -1)  # (1, n_frames)
            input_per_trial = np.array([env_val, trial_number, prev_outcome], dtype=np.float32)  # (3,)
            
            # Combine: first row is time-varying, rest are per-trial
            # According to spec: shape (n_input, n_timepoints) or (n_input,)
            # We need to handle mixed time-varying and per-trial
            # Let's broadcast per-trial to time-varying
            input_data = np.zeros((4, n_frames), dtype=np.float32)
            input_data[0, :] = time_from_start
            input_data[1, :] = env_val
            input_data[2, :] = trial_number
            input_data[3, :] = prev_outcome
            
            input_trials.append(input_data)
            
            # --- Output data ---
            trial_pos = position[start:end]
            trial_speed = speed[start:end]
            trial_lick = lick[start:end]
            
            # Distance to reward zone
            rz_start_cm = rz_coords[i, 0]
            rz_end_cm = rz_coords[i, 1]
            dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start_cm, rz_end_cm)
            dist_bins = discretize_distance(dist_to_rz)
            
            # Absolute position
            pos_bins = discretize_position(trial_pos)
            
            # Speed
            speed_bins = discretize_speed(trial_speed)
            
            # Lick (binary: any lick > 0)
            lick_binary = (trial_lick > 0).astype(np.int64)
            
            # Reward zone location (per-trial): A=0, B=1, C=2
            rz_label = rz_labels[i]
            rz_loc = {'A': 0, 'B': 1, 'C': 2}.get(rz_label, 0)
            
            # Reward outcome (per-trial)
            reward_outcome = rewarded[i]
            
            # Output: (6, n_frames) for time-varying, per-trial values broadcast
            output_data = np.zeros((6, n_frames), dtype=np.int64)
            output_data[0, :] = dist_bins
            output_data[1, :] = pos_bins
            output_data[2, :] = speed_bins
            output_data[3, :] = lick_binary
            output_data[4, :] = rz_loc  # per-trial, broadcast
            output_data[5, :] = reward_outcome  # per-trial, broadcast
            
            output_trials.append(output_data)
        
        print(f"    Trial segmentation: {time.time()-t1:.1f}s")
        print(f"    Valid trials: {len(neural_trials)}")
        
        if len(neural_trials) < 2:
            print(f"    WARNING: Less than 2 valid trials, skipping session")
            return None
    
    result = {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'n_neurons': n_final_cells,
        'subject': subject_id,
        'scene': scene,
        'session_id': session_id,
        'env_per_trial': env_per_trial,
        'rz_labels': rz_labels,
        'rewarded': rewarded,
    }
    
    print(f"    Total time: {time.time()-t0:.1f}s")
    
    return result


def show_processing_plots(result, session_idx, output_prefix='processing'):
    """Generate processing visualization plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(6, 2, figsize=(20, 24))
    fig.suptitle(f"Session: {result['subject']} ses-{result['session_id']} ({result['scene']})", fontsize=14)
    
    # Pick a few example trials
    n_trials = len(result['neural'])
    example_trials = [0, n_trials // 2, n_trials - 1]
    
    # Plot 1: Neural activity heatmap for first trial
    trial_idx = example_trials[0]
    ax = axes[0, 0]
    neural = result['neural'][trial_idx]
    im = ax.imshow(neural[:min(50, neural.shape[0]), :], aspect='auto', cmap='hot')
    ax.set_title(f'Neural activity (trial {trial_idx}, first 50 cells)')
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Neuron')
    plt.colorbar(im, ax=ax)
    
    # Plot 2: Neural activity heatmap for middle trial
    trial_idx = example_trials[1]
    ax = axes[0, 1]
    neural = result['neural'][trial_idx]
    im = ax.imshow(neural[:min(50, neural.shape[0]), :], aspect='auto', cmap='hot')
    ax.set_title(f'Neural activity (trial {trial_idx}, first 50 cells)')
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Neuron')
    plt.colorbar(im, ax=ax)
    
    # Plot 3: Input - time from start
    ax = axes[1, 0]
    for ti in example_trials:
        inp = result['input'][ti]
        ax.plot(inp[0, :], label=f'Trial {ti}')
    ax.set_title('Input: Time from trial start (s)')
    ax.legend()
    
    # Plot 4: Input - environment and trial number
    ax = axes[1, 1]
    envs = [result['input'][ti][1, 0] for ti in range(n_trials)]
    ax.plot(envs, 'o-', markersize=2)
    ax.set_title('Input: Environment per trial')
    ax.set_ylabel('Environment (0=ENV1, 1=ENV2)')
    
    # Plot 5: Output - distance to reward zone
    ax = axes[2, 0]
    for ti in example_trials:
        out = result['output'][ti]
        ax.plot(out[0, :], label=f'Trial {ti}')
    ax.set_title('Output: Distance to RZ (discretized)')
    ax.legend()
    
    # Plot 6: Output - position
    ax = axes[2, 1]
    for ti in example_trials:
        out = result['output'][ti]
        ax.plot(out[1, :], label=f'Trial {ti}')
    ax.set_title('Output: Position (discretized)')
    ax.legend()
    
    # Plot 7: Output - speed
    ax = axes[3, 0]
    for ti in example_trials:
        out = result['output'][ti]
        ax.plot(out[2, :], label=f'Trial {ti}')
    ax.set_title('Output: Speed (discretized)')
    ax.legend()
    
    # Plot 8: Output - lick
    ax = axes[3, 1]
    for ti in example_trials:
        out = result['output'][ti]
        ax.plot(out[3, :], label=f'Trial {ti}')
    ax.set_title('Output: Lick (binary)')
    ax.legend()
    
    # Plot 9: Output - reward zone location per trial
    ax = axes[4, 0]
    rz_locs = [result['output'][ti][4, 0] for ti in range(n_trials)]
    ax.plot(rz_locs, 'o-', markersize=3)
    ax.set_title('Output: Reward zone location per trial')
    ax.set_ylabel('0=A, 1=B, 2=C')
    
    # Plot 10: Output - reward outcome per trial
    ax = axes[4, 1]
    rewards = [result['output'][ti][5, 0] for ti in range(n_trials)]
    ax.plot(rewards, 'o-', markersize=3)
    ax.set_title('Output: Reward outcome per trial')
    ax.set_ylabel('0=no, 1=yes')
    
    # Plot 11: Trial durations
    ax = axes[5, 0]
    durations = [result['neural'][ti].shape[1] for ti in range(n_trials)]
    ax.plot(durations, 'o-', markersize=3)
    ax.set_title('Trial duration (frames)')
    ax.set_xlabel('Trial')
    
    # Plot 12: Reward and omission statistics
    ax = axes[5, 1]
    reward_rate = np.mean(rewards)
    ax.bar(['Rewarded', 'Omission'], [reward_rate, 1-reward_rate])
    ax.set_title(f'Reward rate: {reward_rate:.2%}')
    
    plt.tight_layout()
    fname = f'{output_prefix}_{result["subject"]}_ses{result["session_id"]}.png'
    plt.savefig(fname, dpi=100)
    plt.close()
    print(f"    Saved processing plot: {fname}")


def main():
    parser = argparse.ArgumentParser(description='Convert NWB data to decoder format')
    parser.add_argument('output', help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    args = parser.parse_args()
    
    if args.sample:
        args.full = False
    
    print("=" * 60)
    print("Sosa et al. 2025 NWB to Decoder Format Conversion")
    print("=" * 60)
    
    # Find all NWB files
    data_dir = '/app/data'
    subjects = sorted([s for s in os.listdir(data_dir) if s.startswith('sub-')])
    
    all_nwb_files = []
    for sub in subjects:
        files = sorted(glob.glob(os.path.join(data_dir, sub, '*.nwb')))
        all_nwb_files.extend(files)
    
    print(f"Found {len(all_nwb_files)} NWB files from {len(subjects)} subjects")
    
    if args.sample:
        # Select 2 sessions from different subjects for testing
        sample_files = [all_nwb_files[0], all_nwb_files[len(all_nwb_files)//2]]
        nwb_files = sample_files
        print(f"Sample mode: processing {len(nwb_files)} sessions")
    else:
        nwb_files = all_nwb_files
        print(f"Full mode: processing {len(nwb_files)} sessions")
    
    # Process all sessions
    all_neural = []
    all_input = []
    all_output = []
    all_subjects_list = []
    all_subject_idx = []
    all_brain_region_idx = []
    session_info = []
    
    unique_subjects = []
    
    t_total = time.time()
    
    for si, nwb_file in enumerate(nwb_files):
        print(f"\n[{si+1}/{len(nwb_files)}] {os.path.basename(nwb_file)}")
        
        result = process_session(nwb_file, show_processing=args.show_processing, session_idx=si)
        
        if result is None:
            print(f"  SKIPPED")
            continue
        
        # Add to lists
        all_neural.append(result['neural'])
        all_input.append(result['input'])
        all_output.append(result['output'])
        
        # Track subjects
        subj = result['subject']
        if subj not in unique_subjects:
            unique_subjects.append(subj)
        subj_idx = unique_subjects.index(subj)
        all_subject_idx.append(subj_idx)
        
        # Brain region index (all CA1)
        all_brain_region_idx.append(np.zeros(result['n_neurons'], dtype=int))
        
        # Session info
        session_info.append({
            'subject': subj,
            'session_id': result['session_id'],
            'scene': result['scene'],
            'n_neurons': result['n_neurons'],
            'n_trials': len(result['neural']),
        })
        
        if args.show_processing and si < 2:
            show_processing_plots(result, si)
    
    print(f"\n{'='*60}")
    print(f"Total processing time: {time.time()-t_total:.1f}s")
    print(f"Sessions processed: {len(all_neural)}")
    print(f"Subjects: {unique_subjects}")
    
    # Build final data structure
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        
        'subjects': unique_subjects,
        'subject_idx': np.array(all_subject_idx),
        
        'brain_regions': ['CA1'],
        'brain_region_idx': all_brain_region_idx,
        
        'input_names': [
            'time_from_trial_start',
            'environment',
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
            ['< -50cm', '-50 to -10cm', '-10 to 0cm', '0cm (in zone)', '0 to +10cm', '+10 to +50cm', '> +50cm'],
            ['0-90cm', '90-180cm', '180-270cm', '270-360cm', '360-450cm'],
            ['< 2cm/s', '2-10cm/s', '10-20cm/s', '20-40cm/s', '> 40cm/s'],
            ['no_lick', 'lick'],
            ['A', 'B', 'C'],
            ['no_reward', 'reward'],
        ],
        
        'metadata': {
            'task_description': 'Virtual reality navigation task with hidden reward zone switches. '
                               'Mice run on a 450cm linear track with reward zones at locations A (80-130cm), '
                               'B (200-250cm), or C (320-370cm). Reward is omitted on ~15% of trials.',
            'time_bin_size': 1000.0 / 15.5078125,  # ~64.5 ms
            'temporal_alignment_event': 'Trial start (entry to linear track)',
            'off_start': 0.0,  # Trial starts at alignment event
            'off_end': None,  # Variable trial length
            'frame_rate': 15.5078125,
            'session_info': session_info,
            'neural_data_type': 'deltaF/F (computed from raw fluorescence with neuropil subtraction, maximin baseline, 2-sample Gaussian smoothing)',
            'cell_filtering': 'Manual suite2p curation (iscell[:,0]==1) + interneuron exclusion (speed correlation > 0.5)',
        }
    }
    
    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size = os.path.getsize(args.output) / (1024**2)
    print(f"Saved: {file_size:.1f} MB")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    print(f"Subjects: {len(unique_subjects)}")
    print(f"Sessions: {len(all_neural)}")
    total_trials = sum(len(s) for s in all_neural)
    print(f"Total trials: {total_trials}")
    total_neurons = sum(len(br) for br in all_brain_region_idx)
    print(f"Total neurons: {total_neurons}")
    neurons_per_session = [len(br) for br in all_brain_region_idx]
    print(f"Neurons/session: {np.mean(neurons_per_session):.1f} ± {np.std(neurons_per_session):.1f} (range: {np.min(neurons_per_session)}-{np.max(neurons_per_session)})")
    trials_per_session = [len(s) for s in all_neural]
    print(f"Trials/session: {np.mean(trials_per_session):.1f} ± {np.std(trials_per_session):.1f}")
    
    # Reward rate
    all_rewards = []
    for sess_outputs in all_output:
        for trial_out in sess_outputs:
            all_rewards.append(trial_out[5, 0])  # reward outcome
    print(f"Reward rate: {np.mean(all_rewards):.3f}")
    
    print(f"\nDone!")


if __name__ == '__main__':
    main()
