#!/usr/bin/env python3
"""Convert NWB data from Sosa et al. 2025 to decoder-compatible format.

Usage:
    python -u convert_data.py <output_pickle> [--full|--sample] [--show-processing]
"""
import os
import sys
import time
import argparse
import glob
import pickle
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter1d, minimum_filter1d, maximum_filter1d, gaussian_filter
from scipy.stats import pearsonr

# Reward zone coordinates from the reference code (behavior.py)
REWARD_ZONE_DICT = {
    'A': [80, 130],   # LocationA -> X
    'B': [200, 250],  # LocationB -> Y  
    'C': [320, 370],  # LocationC -> Z
}

# Sessions dict mapping: NWB ses number -> exp_day
# Scene names from sessions_dict.py for determining reward zones
SESSIONS_SCENES = {
    'm11': {1: 'Env1_LocationB', 2: 'Env1_LocationB', 3: 'Env1_LocationB_to_A', 4: 'Env1_LocationA', 5: 'Env1_LocationA_to_C', 6: 'Env1_LocationC', 7: 'Env1_LocationC_to_B', 8: 'Env1_B_to_Env2_C', 9: 'Env2_LocationC', 10: 'Env2_LocationC_to_A', 11: 'Env2_LocationA', 12: 'Env2_LocationA_to_B', 13: 'Env2_LocationB', 14: 'Env2_LocationB_to_C'},
    'm12': {1: 'Env1_LocationB', 2: 'Env1_LocationB', 3: 'Env1_LocationB_to_C', 4: 'Env1_LocationC', 5: 'Env1_LocationC_to_A', 6: 'Env1_LocationA', 7: 'Env1_LocationA_to_B', 8: 'Env1_B_to_Env2_A', 9: 'Env2_LocationA', 10: 'Env2_LocationA_to_C', 11: 'Env2_LocationC', 12: 'Env2_LocationC_to_B', 13: 'Env2_LocationB', 14: 'Env2_LocationB_to_A'},
    'm13': {1: 'Env1_LocationC', 2: 'Env1_LocationC', 3: 'Env1_LocationC_to_B', 4: 'Env1_LocationB', 5: 'Env1_LocationB_to_A', 6: 'Env1_LocationA', 7: 'Env1_LocationA_to_C', 8: 'Env1_C_to_Env2_A', 9: 'Env2_LocationA', 10: 'Env2_LocationA_to_B', 11: 'Env2_LocationB', 12: 'Env2_LocationB_to_C', 13: 'Env2_LocationC', 14: 'Env2_LocationC_to_A'},
    'm14': {1: 'Env1_LocationA', 2: 'Env1_LocationA', 3: 'Env1_LocationA_to_B', 4: 'Env1_LocationB', 5: 'Env1_LocationB_to_C', 6: 'Env1_LocationC', 7: 'Env1_LocationC_to_A', 8: 'Env1_A_to_Env2_C', 9: 'Env2_LocationC', 10: 'Env2_LocationC_to_B', 11: 'Env2_LocationB', 12: 'Env2_LocationB_to_A', 13: 'Env2_LocationA', 14: 'Env2_LocationA_to_C'},
    'm15': {1: 'Env1_LocationA', 2: 'Env1_LocationA', 3: 'Env1_LocationA_to_C', 4: 'Env1_LocationC', 5: 'Env1_LocationC_to_B', 6: 'Env1_LocationB', 7: 'Env1_LocationB_to_A', 8: 'Env1_A_to_Env2_B', 9: 'Env2_LocationB', 10: 'Env2_LocationB_to_C', 11: 'Env2_LocationC', 12: 'Env2_LocationC_to_A', 13: 'Env2_LocationA', 14: 'Env2_LocationA_to_B'},
    'm17': {1: 'Env2_LocationB', 2: 'Env2_LocationB', 3: 'Env2_LocationB_to_C', 4: 'Env2_LocationC', 5: 'Env2_LocationC_to_A', 6: 'Env2_LocationA', 7: 'Env2_LocationA_to_B', 8: 'Env2_B_to_Env1_A', 9: 'Env1_LocationA', 10: 'Env1_LocationA_to_C', 11: 'Env1_LocationC', 12: 'Env1_LocationC_to_B', 13: 'Env1_LocationB', 14: 'Env1_LocationB_to_A'},
    'm18': {1: 'Env2_LocationA', 2: 'Env2_LocationA', 3: 'Env2_LocationA_to_B', 4: 'Env2_LocationB', 5: 'Env2_LocationB_to_C', 6: 'Env2_LocationC', 7: 'Env2_LocationC_to_A', 8: 'Env2_A_to_Env1_C', 9: 'Env1_LocationC', 10: 'Env1_LocationC_to_B', 11: 'Env1_LocationB', 12: 'Env1_LocationB_to_A', 13: 'Env1_LocationA', 14: 'Env1_LocationA_to_C'},
    'm19': {1: 'Env1_LocationC', 2: 'Env1_LocationC', 3: 'Env1_LocationC_to_B', 4: 'Env1_LocationB', 5: 'Env1_LocationB_to_A', 6: 'Env1_LocationA', 7: 'Env1_LocationA_to_C', 8: 'Env1_C_to_Env2_A', 9: 'Env2_LocationA', 10: 'Env2_LocationA_to_B', 11: 'Env2_LocationB', 12: 'Env2_LocationB_to_C', 13: 'Env2_LocationC', 14: 'Env2_LocationC_to_A'},
    'm3': {1: 'Env1_LocationC', 2: 'Env1_LocationC', 3: 'Env1_LocationC_to_A', 4: 'Env1_LocationA', 5: 'Env1_LocationA_to_B', 6: 'Env1_LocationB', 7: 'Env1_LocationB_to_C', 8: 'Env1_C_to_Env2_B', 9: 'Env2_LocationB', 10: 'Env2_LocationB_to_A', 11: 'Env2_LocationA', 12: 'Env2_LocationA_to_C', 13: 'Env2_LocationC', 14: 'Env2_LocationC_to_B'},
    'm4': {1: 'Env1_LocationB', 2: 'Env1_LocationB', 3: 'Env1_LocationB_to_A', 4: 'Env1_LocationA', 5: 'Env1_LocationA_to_C', 6: 'Env1_LocationC', 7: 'Env1_LocationC_to_B', 8: 'Env1_B_to_Env2_C', 9: 'Env2_LocationC', 10: 'Env2_LocationC_to_A', 11: 'Env2_LocationA', 12: 'Env2_LocationA_to_B', 13: 'Env2_LocationB', 14: 'Env2_LocationB_to_C'},
    'm7': {1: 'Env1_LocationA', 2: 'Env1_LocationA', 3: 'Env1_LocationA_to_C', 4: 'Env1_LocationC', 5: 'Env1_LocationC_to_B', 6: 'Env1_LocationB', 7: 'Env1_LocationB_to_A', 8: 'Env1_A_to_Env2_B', 9: 'Env2_LocationB', 10: 'Env2_LocationB_to_C', 11: 'Env2_LocationC', 12: 'Env2_LocationC_to_A', 13: 'Env2_LocationA', 14: 'Env2_LocationA_to_B'},
}


def get_reward_zone_for_trial(scene, trial_idx, n_trials, change_trial=30):
    """Get reward zone label and coordinates for a given trial.
    
    Based on get_reward_zones() from behavior.py.
    Returns: (rz_start, rz_end), rz_label ('A', 'B', or 'C')
    """
    if 'LocationA' in scene and '_to_' not in scene:
        return REWARD_ZONE_DICT['A'], 'A'
    elif 'LocationB' in scene and '_to_' not in scene:
        return REWARD_ZONE_DICT['B'], 'B'
    elif 'LocationC' in scene and '_to_' not in scene:
        return REWARD_ZONE_DICT['C'], 'C'
    elif '_to_' in scene or '_to_Env' in scene:
        # Switch session: determine pre and post zones
        # Parse scene name to get zones
        # Examples: 'Env1_LocationA_to_B', 'Env1_LocationA_to_C', 'Env1_B_to_Env2_A'
        parts = scene.split('_to_')
        pre_part = parts[0]  # e.g., 'Env1_LocationA' or 'Env1_B'
        post_part = parts[1]  # e.g., 'B' or 'Env2_A' or 'LocationC'
        
        # Determine pre-switch zone
        if 'LocationA' in pre_part or pre_part.endswith('_A'):
            pre_zone = 'A'
        elif 'LocationB' in pre_part or pre_part.endswith('_B'):
            pre_zone = 'B'
        elif 'LocationC' in pre_part or pre_part.endswith('_C'):
            pre_zone = 'C'
        else:
            raise ValueError(f"Cannot determine pre-zone from scene: {scene}")
        
        # Determine post-switch zone
        if 'LocationA' in post_part or post_part.endswith('A') or post_part == 'A':
            post_zone = 'A'
        elif 'LocationB' in post_part or post_part.endswith('B') or post_part == 'B':
            post_zone = 'B'
        elif 'LocationC' in post_part or post_part.endswith('C') or post_part == 'C':
            post_zone = 'C'
        else:
            raise ValueError(f"Cannot determine post-zone from scene: {scene}")
        
        if trial_idx < change_trial:
            return REWARD_ZONE_DICT[pre_zone], pre_zone
        else:
            return REWARD_ZONE_DICT[post_zone], post_zone
    else:
        raise ValueError(f"Unknown scene: {scene}")


def nansmooth(a, sig, axis=-1):
    """Gaussian smoothing that handles NaNs.
    Reimplementation of utilities.nansmooth from the reference code.
    """
    nan_inds = np.isnan(a)
    a_nanless = np.copy(a)
    a_nanless[nan_inds] = 0
    
    one = np.ones(a.shape)
    one[nan_inds] = 0.001
    
    a_nanless = gaussian_filter1d(a_nanless, sig, axis=axis)
    one = gaussian_filter1d(one, sig, axis=axis)
    
    return a_nanless / one


def compute_dff(F, Fneu, trial_start_idx, teleport_idx, neu_coef=0.7):
    """Compute dF/F following the reference paper's method.
    
    Steps:
    1. Neuropil subtraction: F -= neu_coef * Fneu
    2. For each trial:
       a. Add back neuropil mean within trial
       b. Maximin baseline (smooth, min filter 300, max filter 300)
    3. dF/F = (F - baseline) / |baseline|
    4. Smooth dF/F with 2-sample Gaussian per trial
    
    Args:
        F: (n_cells, n_timepoints) raw fluorescence
        Fneu: (n_cells, n_timepoints) neuropil fluorescence
        trial_start_idx: array of trial start frame indices
        teleport_idx: array of teleport (trial end) frame indices
        neu_coef: neuropil coefficient (default 0.7)
    
    Returns:
        dff: (n_cells, n_timepoints) delta F/F
    """
    n_cells, n_timepoints = F.shape
    
    # Copy to avoid modifying originals
    f_ = F.astype(np.float64).copy()
    f_neu_ = Fneu.astype(np.float64).copy()
    
    # Step 1: Neuropil subtraction
    f_ -= neu_coef * f_neu_
    
    # Initialize baseline and dff arrays
    flow = np.full_like(f_, np.nan)
    dff = np.full_like(f_, np.nan)
    
    # Step 2: Compute baseline per trial
    for i in range(len(trial_start_idx)):
        start = trial_start_idx[i]
        stop = teleport_idx[i]
        if stop <= start:
            continue
        
        # Add back neuropil mean within trial
        f_[:, start:stop] = f_[:, start:stop] + neu_coef * np.nanmean(
            f_neu_[:, start:stop], axis=1, keepdims=True)
        
        # Smooth signal with Gaussian sigma=15
        trial_data = f_[:, start:stop]
        flow[:, start:stop] = nansmooth(trial_data, 15, axis=1)
        
        # Minimum filter with window=300 frames (~20s)
        flow[:, start:stop] = minimum_filter1d(flow[:, start:stop], 300, axis=-1)
        # Maximum filter with same window (dilation)
        flow[:, start:stop] = maximum_filter1d(flow[:, start:stop], 300, axis=-1)
    
    # Step 3: Compute dF/F
    valid = ~np.isnan(flow) & (np.abs(flow) > 0)
    dff[valid] = (f_[valid] - flow[valid]) / np.abs(flow[valid])
    
    # Step 4: Smooth dF/F with 2-sample Gaussian per trial
    for i in range(len(trial_start_idx)):
        start = trial_start_idx[i]
        stop = teleport_idx[i]
        if stop <= start:
            continue
        dff[:, start:stop] = nansmooth(dff[:, start:stop], 2, axis=1)
    
    return dff


def detect_interneurons(dff, speed, threshold=0.5):
    """Detect putative interneurons by correlation with running speed.
    
    From methods: 'Pearson correlation of >0.5 between dF/F and running speed'
    Vectorized implementation for speed.
    
    Returns: boolean mask of interneurons (True = interneuron)
    """
    n_cells = dff.shape[0]
    is_interneuron = np.zeros(n_cells, dtype=bool)
    
    # Find valid timepoints (not NaN in speed)
    speed_valid = ~np.isnan(speed)
    speed_clean = speed.copy()
    speed_clean[~speed_valid] = 0
    
    for i in range(n_cells):
        valid = ~np.isnan(dff[i]) & speed_valid
        n_valid = valid.sum()
        if n_valid > 10:
            d = dff[i, valid]
            s = speed_clean[valid]
            # Fast Pearson correlation
            d_mean = d.mean()
            s_mean = s.mean()
            d_centered = d - d_mean
            s_centered = s - s_mean
            num = np.dot(d_centered, s_centered)
            denom = np.sqrt(np.dot(d_centered, d_centered) * np.dot(s_centered, s_centered))
            if denom > 0:
                r = num / denom
                if r > threshold:
                    is_interneuron[i] = True
    
    return is_interneuron


def discretize_distance_to_rz(distance):
    """Discretize distance to reward zone into 7 bins.
    Bins: 0: <-50, 1: -50 to -10, 2: -10 to <0, 3: 0, 4: >0 to +10, 5: +10 to +50, 6: >+50
    """
    bins = np.full_like(distance, -1, dtype=np.int64)
    bins[distance < -50] = 0
    bins[(distance >= -50) & (distance < -10)] = 1
    bins[(distance >= -10) & (distance < 0)] = 2
    bins[distance == 0] = 3
    bins[(distance > 0) & (distance <= 10)] = 4
    bins[(distance > 10) & (distance <= 50)] = 5
    bins[distance > 50] = 6
    return bins


def discretize_position(position):
    """Discretize position into 5 equal bins of 90cm spanning 450cm track.
    Bins: 0: <90, 1: 90-180, 2: 180-270, 3: 270-360, 4: >360
    """
    bins = np.full_like(position, -1, dtype=np.int64)
    bins[position < 90] = 0
    bins[(position >= 90) & (position < 180)] = 1
    bins[(position >= 180) & (position < 270)] = 2
    bins[(position >= 270) & (position < 360)] = 3
    bins[position >= 360] = 4
    return bins


def discretize_speed(speed):
    """Discretize speed into 5 bins.
    Bins: 0: <2, 1: 2-10, 2: 10-20, 3: 20-40, 4: >40
    """
    bins = np.full_like(speed, -1, dtype=np.int64)
    bins[speed < 2] = 0
    bins[(speed >= 2) & (speed < 10)] = 1
    bins[(speed >= 10) & (speed < 20)] = 2
    bins[(speed >= 20) & (speed < 40)] = 3
    bins[speed >= 40] = 4
    return bins


def process_session(nwb_path, subject_id, ses_num, show_processing=False):
    """Process a single NWB session file.
    
    Returns dict with processed data or None if session should be skipped.
    """
    t0 = time.time()
    print(f"  Processing {subject_id} ses-{ses_num:02d}...")
    
    scene = SESSIONS_SCENES[subject_id][ses_num]
    
    with h5py.File(nwb_path, 'r') as f:
        # Load neural data (handle multi-plane)
        fluor_group = f['processing']['ophys']['Fluorescence']
        plane_keys = sorted([k for k in fluor_group.keys() if k.startswith('plane')])
        
        F_planes = []
        Fneu_planes = []
        for pk in plane_keys:
            F_planes.append(f['processing']['ophys']['Fluorescence'][pk]['data'][()].T)
            Fneu_planes.append(f['processing']['ophys']['Neuropil'][pk]['data'][()].T)
        
        F_raw = np.concatenate(F_planes, axis=0)  # (n_rois, n_timepoints)
        Fneu_raw = np.concatenate(Fneu_planes, axis=0)
        iscell = f['processing']['ophys']['ImageSegmentation']['PlaneSegmentation']['iscell'][()]
        
        # Load behavioral data
        beh = f['processing']['behavior']['BehavioralTimeSeries']
        position = beh['position']['data'][()]
        speed = beh['speed']['data'][()]
        lick = beh['lick']['data'][()]
        environment = beh['environment']['data'][()]
        trial_number = beh['trial number']['data'][()]
        scanning = beh['scanning']['data'][()]
        trial_start_arr = beh['trial_start']['data'][()]
        teleport_arr = beh['teleport']['data'][()]
        autoreward = beh['autoreward']['data'][()]
        reward_zone = beh['reward_zone']['data'][()]
        timestamps = beh['position']['timestamps'][()]
        
        # Reward delivery timestamps
        reward_ts = beh['Reward']['timestamps'][()]
    
    n_rois, n_timepoints_neural = F_raw.shape
    n_timepoints_beh = len(position)
    
    # Handle potential off-by-one between neural and behavioral data
    n_timepoints = min(n_timepoints_neural, n_timepoints_beh)
    if n_timepoints_neural != n_timepoints_beh:
        print(f"    Note: neural ({n_timepoints_neural}) and behavioral ({n_timepoints_beh}) timepoints differ, truncating to {n_timepoints}")
        F_raw = F_raw[:, :n_timepoints]
        Fneu_raw = Fneu_raw[:, :n_timepoints]
        position = position[:n_timepoints]
        speed = speed[:n_timepoints]
        lick = lick[:n_timepoints]
        environment = environment[:n_timepoints]
        trial_number = trial_number[:n_timepoints]
        scanning = scanning[:n_timepoints]
        trial_start_arr = trial_start_arr[:n_timepoints]
        teleport_arr = teleport_arr[:n_timepoints]
        autoreward = autoreward[:n_timepoints]
        reward_zone = reward_zone[:n_timepoints]
        timestamps = timestamps[:n_timepoints]
    
    dt = np.mean(np.diff(timestamps))  # ~0.0645s
    
    # Cell selection: iscell == 1
    cell_mask = iscell[:, 0] == 1
    
    # Get trial boundaries
    tstart_idx = np.where(trial_start_arr == 1)[0]
    teleport_idx = np.where(teleport_arr == 1)[0]
    n_trials = min(len(tstart_idx), len(teleport_idx))
    
    if n_trials < 2:
        print(f"    Skipping: only {n_trials} trials")
        return None
    
    tstart_idx = tstart_idx[:n_trials]
    teleport_idx = teleport_idx[:n_trials]
    
    # Compute dF/F
    print(f"    Computing dF/F for {cell_mask.sum()} cells, {n_timepoints} timepoints...")
    dff = compute_dff(F_raw, Fneu_raw, tstart_idx, teleport_idx, neu_coef=0.7)
    
    # Filter to cells only
    dff_cells = dff[cell_mask]
    
    # Detect and exclude interneurons
    is_interneuron = detect_interneurons(dff_cells, speed, threshold=0.5)
    n_interneurons = is_interneuron.sum()
    print(f"    Excluding {n_interneurons} interneurons ({n_interneurons/cell_mask.sum()*100:.1f}%)")
    
    # Final cell mask: iscell & not interneuron
    cell_indices = np.where(cell_mask)[0]
    valid_cell_indices = cell_indices[~is_interneuron]
    dff_valid = dff[valid_cell_indices]  # (n_valid_cells, n_timepoints)
    n_cells = len(valid_cell_indices)
    
    if n_cells < 2:
        print(f"    Skipping: only {n_cells} valid cells")
        return None
    
    # Determine reward per trial
    rewarded_trials = np.zeros(n_trials, dtype=bool)
    for t in range(n_trials):
        t_start_time = timestamps[tstart_idx[t]]
        t_end_time = timestamps[teleport_idx[t]]
        rewarded_trials[t] = np.any((reward_ts >= t_start_time) & (reward_ts <= t_end_time))
    
    # Determine reward zone per trial
    rz_labels = []
    rz_coords = []
    for t in range(n_trials):
        coords, label = get_reward_zone_for_trial(scene, t, n_trials, change_trial=30)
        rz_labels.append(label)
        rz_coords.append(coords)
    rz_labels = np.array(rz_labels)
    rz_coords = np.array(rz_coords)  # (n_trials, 2) - [start, end]
    
    # Determine environment type per trial (from scanning frames)
    env_per_trial = np.zeros(n_trials, dtype=int)
    for t in range(n_trials):
        start = tstart_idx[t]
        stop = teleport_idx[t]
        env_vals = environment[start:stop]
        env_vals = env_vals[env_vals >= 0]  # exclude -1
        if len(env_vals) > 0:
            env_per_trial[t] = int(np.median(env_vals))
    
    # Build trial-level data
    neural_trials = []
    input_trials = []
    output_trials = []
    
    for t in range(n_trials):
        start = tstart_idx[t]
        stop = teleport_idx[t]
        n_tp = stop - start
        
        if n_tp < 2:
            continue
        
        # Neural data: (n_cells, n_timepoints)
        neural = dff_valid[:, start:stop].astype(np.float32)
        
        # Replace NaNs with 0 for neural data
        neural = np.nan_to_num(neural, nan=0.0)
        
        # Time from trial start (seconds)
        time_from_start = timestamps[start:stop] - timestamps[start]
        
        # Input variables
        # 1. Time from start of trial (continuous, time-varying)
        # 2. Environment type (binary, per trial)
        # 3. Trial number (continuous, per trial)
        # 4. Previous trial outcome (binary, per trial)
        prev_outcome = float(rewarded_trials[t-1]) if t > 0 else 0.0  # first trial: no previous
        
        inp = np.zeros((4, n_tp), dtype=np.float32)
        inp[0, :] = time_from_start
        inp[1, :] = env_per_trial[t]
        inp[2, :] = t  # trial number (0-indexed)
        inp[3, :] = prev_outcome
        
        # Output variables (time-varying)
        trial_pos = position[start:stop]
        trial_speed = speed[start:stop]
        trial_lick = lick[start:stop]
        
        # Distance to reward zone
        rz_start_cm, rz_end_cm = rz_coords[t]
        dist_to_rz = np.where(trial_pos < rz_start_cm, trial_pos - rz_start_cm,
                              np.where(trial_pos > rz_end_cm, trial_pos - rz_end_cm, 0.0))
        
        # Discretize outputs
        dist_bins = discretize_distance_to_rz(dist_to_rz)
        pos_bins = discretize_position(trial_pos)
        speed_bins = discretize_speed(trial_speed)
        lick_binary = (trial_lick > 0).astype(np.int64)
        
        # Per-trial outputs
        rz_label_int = {'A': 0, 'B': 1, 'C': 2}[rz_labels[t]]
        reward_outcome = int(rewarded_trials[t])
        
        # Build output array: 6 outputs
        # Time-varying: dist_to_rz, position, speed, lick (4 x n_tp)
        # Per-trial: reward_zone_location, reward_outcome (2,)
        out_tv = np.zeros((4, n_tp), dtype=np.int64)
        out_tv[0, :] = dist_bins
        out_tv[1, :] = pos_bins
        out_tv[2, :] = speed_bins
        out_tv[3, :] = lick_binary
        
        out_trial = np.array([rz_label_int, reward_outcome], dtype=np.int64)
        
        # Combine: first 4 are time-varying, last 2 are per-trial
        # Store as (6, n_tp) where per-trial values are broadcast
        out = np.zeros((6, n_tp), dtype=np.int64)
        out[:4, :] = out_tv
        out[4, :] = rz_label_int
        out[5, :] = reward_outcome
        
        neural_trials.append(neural)
        input_trials.append(inp)
        output_trials.append(out)
    
    if len(neural_trials) < 2:
        print(f"    Skipping: only {len(neural_trials)} valid trials after filtering")
        return None
    
    elapsed = time.time() - t0
    print(f"    Done: {n_cells} cells, {len(neural_trials)} trials, {elapsed:.1f}s")
    
    if show_processing:
        plot_processing(subject_id, ses_num, neural_trials, input_trials, output_trials,
                       position, speed, lick, timestamps, tstart_idx, teleport_idx,
                       dff_valid, rz_coords, rz_labels, rewarded_trials, environment)
    
    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'n_cells': n_cells,
        'subject_id': subject_id,
        'ses_num': ses_num,
        'scene': scene,
    }


def plot_processing(subject_id, ses_num, neural_trials, input_trials, output_trials,
                   position, speed, lick, timestamps, tstart_idx, teleport_idx,
                   dff_valid, rz_coords, rz_labels, rewarded_trials, environment):
    """Plot processing visualizations for quality checking."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(6, 2, figsize=(20, 24))
    fig.suptitle(f'{subject_id} ses-{ses_num:02d}', fontsize=16)
    
    # Plot 1: Raw position over time
    ax = axes[0, 0]
    for t in range(min(5, len(neural_trials))):
        tp = input_trials[t][0, :]  # time from start
        pos = np.argmax(output_trials[t][1, :] != output_trials[t][1, 0]) if output_trials[t].shape[1] > 1 else 0
        # Plot position as continuous
        n_tp = neural_trials[t].shape[1]
        start = tstart_idx[t]
        trial_pos = position[start:start+n_tp]
        ax.plot(tp, trial_pos, alpha=0.7, label=f'Trial {t}')
    ax.set_xlabel('Time from trial start (s)')
    ax.set_ylabel('Position (cm)')
    ax.set_title('Position vs Time (first 5 trials)')
    ax.legend(fontsize=8)
    
    # Plot 2: Neural activity heatmap (first trial)
    ax = axes[0, 1]
    if len(neural_trials) > 0:
        trial_neural = neural_trials[0]
        n_show = min(50, trial_neural.shape[0])
        ax.imshow(trial_neural[:n_show], aspect='auto', cmap='viridis')
        ax.set_xlabel('Timepoint')
        ax.set_ylabel('Neuron')
        ax.set_title('Neural activity (Trial 0, first 50 cells)')
    
    # Plot 3: Distance to reward zone
    ax = axes[1, 0]
    for t in range(min(5, len(output_trials))):
        tp = input_trials[t][0, :]
        dist = output_trials[t][0, :]
        ax.plot(tp, dist, alpha=0.7, label=f'Trial {t}')
    ax.set_xlabel('Time from trial start (s)')
    ax.set_ylabel('Distance bin')
    ax.set_title('Distance to reward zone (discretized)')
    ax.legend(fontsize=8)
    
    # Plot 4: Speed distribution
    ax = axes[1, 1]
    all_speeds = np.concatenate([output_trials[t][2, :] for t in range(len(output_trials))])
    ax.hist(all_speeds, bins=np.arange(-0.5, 5.5, 1), edgecolor='black')
    ax.set_xlabel('Speed bin')
    ax.set_ylabel('Count')
    ax.set_title('Speed distribution')
    
    # Plot 5: Lick distribution
    ax = axes[2, 0]
    all_licks = np.concatenate([output_trials[t][3, :] for t in range(len(output_trials))])
    ax.hist(all_licks, bins=np.arange(-0.5, 2.5, 1), edgecolor='black')
    ax.set_xlabel('Lick (0/1)')
    ax.set_ylabel('Count')
    ax.set_title('Lick distribution')
    
    # Plot 6: Reward zone per trial
    ax = axes[2, 1]
    rz_per_trial = [output_trials[t][4, 0] for t in range(len(output_trials))]
    ax.plot(rz_per_trial, 'o-', markersize=3)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Reward zone (0=A, 1=B, 2=C)')
    ax.set_title('Reward zone location per trial')
    
    # Plot 7: Reward outcome per trial
    ax = axes[3, 0]
    rew_per_trial = [output_trials[t][5, 0] for t in range(len(output_trials))]
    ax.plot(rew_per_trial, 'o-', markersize=3)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Reward (0/1)')
    ax.set_title('Reward outcome per trial')
    
    # Plot 8: Environment per trial
    ax = axes[3, 1]
    env_per_trial = [input_trials[t][1, 0] for t in range(len(input_trials))]
    ax.plot(env_per_trial, 'o-', markersize=3)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Environment (0/1)')
    ax.set_title('Environment per trial')
    
    # Plot 9: Position distribution
    ax = axes[4, 0]
    all_pos = np.concatenate([output_trials[t][1, :] for t in range(len(output_trials))])
    ax.hist(all_pos, bins=np.arange(-0.5, 5.5, 1), edgecolor='black')
    ax.set_xlabel('Position bin')
    ax.set_ylabel('Count')
    ax.set_title('Position distribution')
    
    # Plot 10: Trial duration distribution
    ax = axes[4, 1]
    durations = [input_trials[t][0, -1] for t in range(len(input_trials))]
    ax.hist(durations, bins=30, edgecolor='black')
    ax.set_xlabel('Trial duration (s)')
    ax.set_ylabel('Count')
    ax.set_title('Trial duration distribution')
    
    # Plot 11: Neural activity mean across cells for first trial
    ax = axes[5, 0]
    if len(neural_trials) > 0:
        mean_neural = np.mean(neural_trials[0], axis=0)
        tp = input_trials[0][0, :]
        ax.plot(tp, mean_neural)
        ax.set_xlabel('Time from trial start (s)')
        ax.set_ylabel('Mean dF/F')
        ax.set_title('Mean neural activity (Trial 0)')
    
    # Plot 12: Previous trial outcome
    ax = axes[5, 1]
    prev_out = [input_trials[t][3, 0] for t in range(len(input_trials))]
    ax.plot(prev_out, 'o-', markersize=3)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Previous outcome (0/1)')
    ax.set_title('Previous trial outcome')
    
    plt.tight_layout()
    plt.savefig(f'/app/processing_{subject_id}_ses{ses_num:02d}.png', dpi=100)
    plt.close()
    print(f"    Saved processing plot")


def main():
    parser = argparse.ArgumentParser(description='Convert NWB data to decoder format')
    parser.add_argument('output', help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    args = parser.parse_args()
    
    data_dir = '/app/data'
    subjects = sorted([d.replace('sub-', '') for d in os.listdir(data_dir) if d.startswith('sub-')])
    
    print(f"Found {len(subjects)} subjects: {subjects}")
    
    # Collect all session files
    all_sessions = []
    for subj in subjects:
        subj_dir = os.path.join(data_dir, f'sub-{subj}')
        nwb_files = sorted(glob.glob(os.path.join(subj_dir, '*.nwb')))
        for nwb_file in nwb_files:
            ses_str = os.path.basename(nwb_file).split('ses-')[1].split('_')[0]
            ses_num = int(ses_str)
            all_sessions.append((subj, ses_num, nwb_file))
    
    print(f"Total sessions: {len(all_sessions)}")
    
    if args.sample:
        # Pick 2 sessions from different subjects for testing
        all_sessions = [all_sessions[0], all_sessions[len(all_sessions)//2]]
        print(f"Sample mode: processing {len(all_sessions)} sessions")
    
    # Process sessions
    neural_all = []
    input_all = []
    output_all = []
    subject_list = []
    subject_idx_list = []
    brain_region_idx_list = []
    session_info_list = []
    
    subjects_seen = []
    total_t0 = time.time()
    
    for i, (subj, ses_num, nwb_file) in enumerate(all_sessions):
        print(f"\n[{i+1}/{len(all_sessions)}] {subj} ses-{ses_num:02d}")
        
        result = process_session(nwb_file, subj, ses_num, show_processing=args.show_processing)
        
        if result is None:
            continue
        
        neural_all.append(result['neural'])
        input_all.append(result['input'])
        output_all.append(result['output'])
        
        if subj not in subjects_seen:
            subjects_seen.append(subj)
        subject_idx_list.append(subjects_seen.index(subj))
        
        # Brain region: all CA1
        brain_region_idx_list.append(np.zeros(result['n_cells'], dtype=np.int64))
        
        session_info_list.append({
            'subject': subj,
            'session': ses_num,
            'scene': result['scene'],
            'n_cells': result['n_cells'],
            'n_trials': len(result['neural']),
        })
    
    total_elapsed = time.time() - total_t0
    print(f"\n\nTotal processing time: {total_elapsed:.1f}s")
    print(f"Processed {len(neural_all)} sessions")
    
    # Build final data structure
    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,
        'subjects': subjects_seen,
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),
        'brain_regions': ['CA1'],
        'brain_region_idx': brain_region_idx_list,
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
            ['< -50 cm', '-50 to -10 cm', '-10 to 0 cm', '0 cm', '>0 to +10 cm', '+10 to +50 cm', '> +50 cm'],
            ['0-90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '360+ cm'],
            ['< 2 cm/s', '2-10 cm/s', '10-20 cm/s', '20-40 cm/s', '> 40 cm/s'],
            ['no lick', 'lick'],
            ['A (80-130)', 'B (200-250)', 'C (320-370)'],
            ['no reward', 'reward'],
        ],
        'metadata': {
            'task_description': 'Virtual reality navigation with hidden reward zones. Mice traverse a 450cm linear track with reward zones at 3 possible locations (A/B/C). Reward zone switches every other day.',
            'time_bin_size': 64.5,  # ms, approximate
            'temporal_alignment_event': 'start of trial (entry to linear track)',
            'off_start': 0.0,  # trial starts at alignment event
            'off_end': None,  # variable trial duration
            'session_info': session_info_list,
            'brain_region': 'CA1 hippocampus',
            'species': 'Mus musculus',
            'imaging_method': 'two-photon calcium imaging (GCaMP7f)',
            'neural_signal': 'dF/F (neuropil-subtracted, maximin baseline)',
            'frame_rate_hz': 15.5,
        },
    }
    
    # Print summary statistics
    total_trials = sum(len(s) for s in neural_all)
    total_neurons = sum(info['n_cells'] for info in session_info_list)
    neurons_per_session = [info['n_cells'] for info in session_info_list]
    trials_per_session = [info['n_trials'] for info in session_info_list]
    
    print(f"\n=== Summary ===")
    print(f"Subjects: {len(subjects_seen)}")
    print(f"Sessions: {len(neural_all)}")
    print(f"Total trials: {total_trials}")
    print(f"Total neurons: {total_neurons}")
    print(f"Neurons/session: {np.mean(neurons_per_session):.0f} ± {np.std(neurons_per_session):.0f} (range: {np.min(neurons_per_session)}-{np.max(neurons_per_session)})")
    print(f"Trials/session: {np.mean(trials_per_session):.1f} ± {np.std(trials_per_session):.1f} (range: {np.min(trials_per_session)}-{np.max(trials_per_session)})")
    
    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    
    file_size = os.path.getsize(args.output) / (1024**2)
    print(f"Saved: {file_size:.1f} MB")
    print("Done!")


if __name__ == '__main__':
    main()
