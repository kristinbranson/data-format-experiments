#!/usr/bin/env python3
"""Convert NWB data from Sosa et al. (2025) to decoder-compatible format.

Usage:
    python -u convert_data.py <output_pickle_file> [--sample] [--full] [--show-processing]
"""

import sys
import os
import argparse
import time
import pickle
import numpy as np
import h5py
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Sessions dictionary (from reference code)
# ============================================================
# Auto-generated from reference code sessions_dict
_all_sessions = {}
_all_sessions['GCAMP2'] = (
    {'exp_day': 1, 'scene': 'Env1_LocationA'},
    {'exp_day': 2, 'scene': 'Env1_LocationA'},
    {'exp_day': 3, 'scene': 'Env1_LocationA'},
    {'exp_day': 4, 'scene': 'Env1_LocationA'},
    {'exp_day': 5, 'scene': 'Env1_LocationA'},
    {'exp_day': 6, 'scene': 'Env1_LocationA'},
    {'exp_day': 7, 'scene': 'Env1_LocationA'},
    {'exp_day': 8, 'scene': 'Env1_LocationA'},
    {'exp_day': 9, 'scene': 'Env1_LocationA'},
    {'exp_day': 10, 'scene': 'Env1_LocationA'},
    {'exp_day': 11, 'scene': 'Env1_LocationA'},
    {'exp_day': 12, 'scene': 'Env1_LocationA'},
    {'exp_day': 13, 'scene': 'Env1_LocationA'},
    {'exp_day': 14, 'scene': 'Env1_LocationA'},
)
_all_sessions['GCAMP3'] = (
    {'exp_day': 1, 'scene': 'Env1_LocationC'},
    {'exp_day': 2, 'scene': 'Env1_LocationC'},
    {'exp_day': 3, 'scene': 'Env1_LocationC_to_A'},
    {'exp_day': 4, 'scene': 'Env1_LocationA'},
    {'exp_day': 5, 'scene': 'Env1_LocationA_to_B'},
    {'exp_day': 6, 'scene': 'Env1_LocationB'},
    {'exp_day': 7, 'scene': 'Env1_LocationB_to_C'},
    {'exp_day': 8, 'scene': 'Env1_C_to_Env2_B'},
    {'exp_day': 9, 'scene': 'Env2_LocationB'},
    {'exp_day': 10, 'scene': 'Env2_LocationB_to_A'},
    {'exp_day': 11, 'scene': 'Env2_LocationA'},
    {'exp_day': 12, 'scene': 'Env2_LocationA_to_C'},
    {'exp_day': 13, 'scene': 'Env2_LocationC'},
    {'exp_day': 14, 'scene': 'Env2_LocationC_to_B'},
)
_all_sessions['GCAMP4'] = (
    {'exp_day': 1, 'scene': 'Env1_LocationB'},
    {'exp_day': 2, 'scene': 'Env1_LocationB'},
    {'exp_day': 3, 'scene': 'Env1_LocationB_to_A'},
    {'exp_day': 4, 'scene': 'Env1_LocationA'},
    {'exp_day': 5, 'scene': 'Env1_LocationA_to_C'},
    {'exp_day': 6, 'scene': 'Env1_LocationC'},
    {'exp_day': 7, 'scene': 'Env1_LocationC_to_B'},
    {'exp_day': 8, 'scene': 'Env1_B_to_Env2_C'},
    {'exp_day': 9, 'scene': 'Env2_LocationC'},
    {'exp_day': 10, 'scene': 'Env2_LocationC_to_A'},
    {'exp_day': 11, 'scene': 'Env2_LocationA'},
    {'exp_day': 12, 'scene': 'Env2_LocationA_to_B'},
    {'exp_day': 13, 'scene': 'Env2_LocationB'},
    {'exp_day': 14, 'scene': 'Env2_LocationB_to_C'},
)
_all_sessions['GCAMP6'] = (
    {'exp_day': 1, 'scene': 'Env1_LocationB'},
    {'exp_day': 2, 'scene': 'Env1_LocationB'},
    {'exp_day': 3, 'scene': 'Env1_LocationB'},
    {'exp_day': 4, 'scene': 'Env1_LocationB'},
    {'exp_day': 5, 'scene': 'Env1_LocationB'},
    {'exp_day': 6, 'scene': 'Env1_LocationB'},
    {'exp_day': 7, 'scene': 'Env1_LocationB'},
    {'exp_day': 8, 'scene': 'Env1_LocationB'},
    {'exp_day': 9, 'scene': 'Env1_LocationB'},
    {'exp_day': 10, 'scene': 'Env1_LocationB'},
    {'exp_day': 11, 'scene': 'Env1_LocationB'},
    {'exp_day': 12, 'scene': 'Env1_LocationB'},
    {'exp_day': 13, 'scene': 'Env1_LocationB'},
    {'exp_day': 14, 'scene': 'Env1_LocationB'},
)
_all_sessions['GCAMP7'] = (
    {'exp_day': 1, 'scene': 'Env1_LocationA'},
    {'exp_day': 2, 'scene': 'Env1_LocationA'},
    {'exp_day': 3, 'scene': 'Env1_LocationA_to_C'},
    {'exp_day': 4, 'scene': 'Env1_LocationC'},
    {'exp_day': 5, 'scene': 'Env1_LocationC_to_B'},
    {'exp_day': 6, 'scene': 'Env1_LocationB'},
    {'exp_day': 7, 'scene': 'Env1_LocationB_to_A'},
    {'exp_day': 8, 'scene': 'Env1_A_to_Env2_B'},
    {'exp_day': 9, 'scene': 'Env2_LocationB'},
    {'exp_day': 10, 'scene': 'Env2_LocationB_to_C'},
    {'exp_day': 11, 'scene': 'Env2_LocationC'},
    {'exp_day': 12, 'scene': 'Env2_LocationC_to_A'},
    {'exp_day': 13, 'scene': 'Env2_LocationA'},
    {'exp_day': 14, 'scene': 'Env2_LocationA_to_B'},
)
_all_sessions['GCAMP10'] = (
    {'exp_day': 1, 'scene': 'Env1_LocationC'},
    {'exp_day': 2, 'scene': 'Env1_LocationC'},
    {'exp_day': 3, 'scene': 'Env1_LocationC'},
    {'exp_day': 4, 'scene': 'Env1_LocationC'},
    {'exp_day': 5, 'scene': 'Env1_LocationC'},
    {'exp_day': 6, 'scene': 'Env1_LocationC'},
    {'exp_day': 7, 'scene': 'Env1_LocationC'},
    {'exp_day': 8, 'scene': 'Env1_LocationC'},
    {'exp_day': 9, 'scene': 'Env1_LocationC'},
    {'exp_day': 10, 'scene': 'Env1_LocationC'},
    {'exp_day': 11, 'scene': 'Env1_LocationC'},
    {'exp_day': 12, 'scene': 'Env1_LocationC'},
    {'exp_day': 13, 'scene': 'Env1_LocationC'},
    {'exp_day': 14, 'scene': 'Env1_LocationC'},
)
_all_sessions['GCAMP11'] = (
    {'exp_day': 1, 'scene': 'Env1_LocationB'},
    {'exp_day': 2, 'scene': 'Env1_LocationB'},
    {'exp_day': 3, 'scene': 'Env1_LocationB_to_A'},
    {'exp_day': 4, 'scene': 'Env1_LocationA'},
    {'exp_day': 5, 'scene': 'Env1_LocationA_to_C'},
    {'exp_day': 6, 'scene': 'Env1_LocationC'},
    {'exp_day': 7, 'scene': 'Env1_LocationC_to_B'},
    {'exp_day': 8, 'scene': 'Env1_B_to_Env2_C'},
    {'exp_day': 9, 'scene': 'Env2_LocationC'},
    {'exp_day': 10, 'scene': 'Env2_LocationC_to_A'},
    {'exp_day': 11, 'scene': 'Env2_LocationA'},
    {'exp_day': 12, 'scene': 'Env2_LocationA_to_B'},
    {'exp_day': 13, 'scene': 'Env2_LocationB'},
    {'exp_day': 14, 'scene': 'Env2_LocationB_to_C'},
)
_all_sessions['GCAMP12'] = (
    {'exp_day': 1, 'scene': 'Env1_LocationB'},
    {'exp_day': 2, 'scene': 'Env1_LocationB'},
    {'exp_day': 3, 'scene': 'Env1_LocationB_to_C'},
    {'exp_day': 4, 'scene': 'Env1_LocationC'},
    {'exp_day': 5, 'scene': 'Env1_LocationC_to_A'},
    {'exp_day': 6, 'scene': 'Env1_LocationA'},
    {'exp_day': 7, 'scene': 'Env1_LocationA_to_B'},
    {'exp_day': 8, 'scene': 'Env1_B_to_Env2_A'},
    {'exp_day': 9, 'scene': 'Env2_LocationA'},
    {'exp_day': 10, 'scene': 'Env2_LocationA_to_C'},
    {'exp_day': 11, 'scene': 'Env2_LocationC'},
    {'exp_day': 12, 'scene': 'Env2_LocationC_to_B'},
    {'exp_day': 13, 'scene': 'Env2_LocationB'},
    {'exp_day': 14, 'scene': 'Env2_LocationB_to_A'},
)
_all_sessions['GCAMP13'] = (
    {'exp_day': 1, 'scene': 'Env1_LocationC'},
    {'exp_day': 2, 'scene': 'Env1_LocationC'},
    {'exp_day': 3, 'scene': 'Env1_LocationC_to_B'},
    {'exp_day': 4, 'scene': 'Env1_LocationB'},
    {'exp_day': 5, 'scene': 'Env1_LocationB_to_A'},
    {'exp_day': 6, 'scene': 'Env1_LocationA'},
    {'exp_day': 7, 'scene': 'Env1_LocationA_to_C'},
    {'exp_day': 8, 'scene': 'Env1_C_to_Env2_A'},
    {'exp_day': 9, 'scene': 'Env2_LocationA'},
    {'exp_day': 10, 'scene': 'Env2_LocationA_to_B'},
    {'exp_day': 11, 'scene': 'Env2_LocationB'},
    {'exp_day': 12, 'scene': 'Env2_LocationB_to_C'},
    {'exp_day': 13, 'scene': 'Env2_LocationC'},
    {'exp_day': 14, 'scene': 'Env2_LocationC_to_A'},
)
_all_sessions['GCAMP14'] = (
    {'exp_day': 1, 'scene': 'Env1_LocationA'},
    {'exp_day': 2, 'scene': 'Env1_LocationA'},
    {'exp_day': 3, 'scene': 'Env1_LocationA_to_B'},
    {'exp_day': 4, 'scene': 'Env1_LocationB'},
    {'exp_day': 5, 'scene': 'Env1_LocationB_to_C'},
    {'exp_day': 6, 'scene': 'Env1_LocationC'},
    {'exp_day': 7, 'scene': 'Env1_LocationC_to_A'},
    {'exp_day': 8, 'scene': 'Env1_A_to_Env2_C'},
    {'exp_day': 9, 'scene': 'Env2_LocationC'},
    {'exp_day': 10, 'scene': 'Env2_LocationC_to_B'},
    {'exp_day': 11, 'scene': 'Env2_LocationB'},
    {'exp_day': 12, 'scene': 'Env2_LocationB_to_A'},
    {'exp_day': 13, 'scene': 'Env2_LocationA'},
    {'exp_day': 14, 'scene': 'Env2_LocationA_to_C'},
)
_all_sessions['GCAMP15'] = (
    {'exp_day': 1, 'scene': 'Env1_LocationA'},
    {'exp_day': 2, 'scene': 'Env1_LocationA'},
    {'exp_day': 3, 'scene': 'Env1_LocationA_to_C'},
    {'exp_day': 4, 'scene': 'Env1_LocationC'},
    {'exp_day': 5, 'scene': 'Env1_LocationC_to_B'},
    {'exp_day': 6, 'scene': 'Env1_LocationB'},
    {'exp_day': 7, 'scene': 'Env1_LocationB_to_A'},
    {'exp_day': 8, 'scene': 'Env1_A_to_Env2_B'},
    {'exp_day': 9, 'scene': 'Env2_LocationB'},
    {'exp_day': 10, 'scene': 'Env2_LocationB_to_C'},
    {'exp_day': 11, 'scene': 'Env2_LocationC'},
    {'exp_day': 12, 'scene': 'Env2_LocationC_to_A'},
    {'exp_day': 13, 'scene': 'Env2_LocationA'},
    {'exp_day': 14, 'scene': 'Env2_LocationA_to_B'},
)
_all_sessions['GCAMP19'] = (
    {'exp_day': 1, 'scene': 'Env1_LocationC'},
    {'exp_day': 2, 'scene': 'Env1_LocationC'},
    {'exp_day': 3, 'scene': 'Env1_LocationC_to_B'},
    {'exp_day': 4, 'scene': 'Env1_LocationB'},
    {'exp_day': 5, 'scene': 'Env1_LocationB_to_A'},
    {'exp_day': 6, 'scene': 'Env1_LocationA'},
    {'exp_day': 7, 'scene': 'Env1_LocationA_to_C'},
    {'exp_day': 8, 'scene': 'Env1_C_to_Env2_A'},
    {'exp_day': 9, 'scene': 'Env2_LocationA'},
    {'exp_day': 10, 'scene': 'Env2_LocationA_to_B'},
    {'exp_day': 11, 'scene': 'Env2_LocationB'},
    {'exp_day': 12, 'scene': 'Env2_LocationB_to_C'},
    {'exp_day': 13, 'scene': 'Env2_LocationC'},
    {'exp_day': 14, 'scene': 'Env2_LocationC_to_A'},
)
_all_sessions['GCAMP17'] = (
    {'exp_day': 1, 'scene': 'Env2_LocationB'},
    {'exp_day': 2, 'scene': 'Env2_LocationB'},
    {'exp_day': 3, 'scene': 'Env2_LocationB_to_C'},
    {'exp_day': 4, 'scene': 'Env2_LocationC'},
    {'exp_day': 5, 'scene': 'Env2_LocationC_to_A'},
    {'exp_day': 6, 'scene': 'Env2_LocationA'},
    {'exp_day': 7, 'scene': 'Env2_LocationA_to_B'},
    {'exp_day': 8, 'scene': 'Env2_B_to_Env1_A'},
    {'exp_day': 9, 'scene': 'Env1_LocationA'},
    {'exp_day': 10, 'scene': 'Env1_LocationA_to_C'},
    {'exp_day': 11, 'scene': 'Env1_LocationC'},
    {'exp_day': 12, 'scene': 'Env1_LocationC_to_B'},
    {'exp_day': 13, 'scene': 'Env1_LocationB'},
    {'exp_day': 14, 'scene': 'Env1_LocationB_to_A'},
)
_all_sessions['GCAMP18'] = (
    {'exp_day': 1, 'scene': 'Env2_LocationA'},
    {'exp_day': 2, 'scene': 'Env2_LocationA'},
    {'exp_day': 3, 'scene': 'Env2_LocationA_to_B'},
    {'exp_day': 4, 'scene': 'Env2_LocationB'},
    {'exp_day': 5, 'scene': 'Env2_LocationB_to_C'},
    {'exp_day': 6, 'scene': 'Env2_LocationC'},
    {'exp_day': 7, 'scene': 'Env2_LocationC_to_A'},
    {'exp_day': 8, 'scene': 'Env2_A_to_Env1_C'},
    {'exp_day': 9, 'scene': 'Env1_LocationC'},
    {'exp_day': 10, 'scene': 'Env1_LocationC_to_B'},
    {'exp_day': 11, 'scene': 'Env1_LocationB'},
    {'exp_day': 12, 'scene': 'Env1_LocationB_to_A'},
    {'exp_day': 13, 'scene': 'Env1_LocationA'},
    {'exp_day': 14, 'scene': 'Env1_LocationA_to_C'},
)

all_sessions_dict = _all_sessions

# Reward zone coordinates
REWARD_ZONE_DICT = {
    'X': [80, 130],   # Zone A
    'Y': [200, 250],  # Zone B  
    'Z': [320, 370],  # Zone C
}

# Map location labels to zone labels
LOCATION_TO_ZONE = {
    'A': 'X',  # A -> X -> [80, 130]
    'B': 'Y',  # B -> Y -> [200, 250]
    'C': 'Z',  # C -> Z -> [320, 370]
}

ZONE_TO_LABEL = {'X': 'A', 'Y': 'B', 'Z': 'C'}

TRACK_LENGTH = 450  # cm
CHANGE_TRIAL = 30   # reward switches after this trial


def get_scene_for_session(subject_id, session_num):
    """Get scene name from sessions_dict for a given subject and session."""
    gcamp_name = f'GCAMP{subject_id}'
    if gcamp_name not in all_sessions_dict:
        raise ValueError(f"Subject {gcamp_name} not found in sessions dict")
    
    sessions = all_sessions_dict[gcamp_name]
    for s in sessions:
        if s['exp_day'] == session_num:
            return s['scene']
    raise ValueError(f"Session {session_num} not found for {gcamp_name}")


def get_reward_zones_from_scene(scene, n_trials, change_trial=CHANGE_TRIAL):
    """Get reward zone coordinates and labels for each trial based on scene name.
    
    Returns:
        rz_coords: (n_trials, 2) array of [start, stop] positions
        rz_labels: (n_trials,) array of zone labels ('A', 'B', 'C')
    """
    rz_coords = np.zeros((n_trials, 2))
    rz_labels = np.empty(n_trials, dtype='U1')
    
    # Parse scene name to determine reward zone(s)
    if '_to_' in scene or '_to_Env' in scene:
        # Switch session - parse before and after locations
        # Handle cross-environment switches like 'Env1_C_to_Env2_B'
        parts = scene.split('_to_')
        
        # Get the "before" location (last character of first part)
        before_loc = parts[0][-1]  # e.g., 'A', 'B', or 'C'
        # Get the "after" location (last character of second part)
        after_loc = parts[1][-1]  # e.g., 'A', 'B', or 'C'
        
        before_zone = LOCATION_TO_ZONE[before_loc]
        after_zone = LOCATION_TO_ZONE[after_loc]
        
        rz_coords[:change_trial] = REWARD_ZONE_DICT[before_zone]
        rz_labels[:change_trial] = ZONE_TO_LABEL[before_zone]
        rz_coords[change_trial:] = REWARD_ZONE_DICT[after_zone]
        rz_labels[change_trial:] = ZONE_TO_LABEL[after_zone]
    else:
        # Non-switch session - single location
        # Extract location from scene name like 'Env1_LocationA'
        loc = scene[-1]  # Last character: A, B, or C
        zone = LOCATION_TO_ZONE[loc]
        rz_coords[:] = REWARD_ZONE_DICT[zone]
        rz_labels[:] = ZONE_TO_LABEL[zone]
    
    return rz_coords, rz_labels


def get_environment_from_scene(scene, n_trials, change_trial=CHANGE_TRIAL):
    """Get environment type (0=ENV1, 1=ENV2) for each trial."""
    env = np.zeros(n_trials, dtype=int)
    
    if '_to_Env' in scene:
        # Cross-environment switch
        parts = scene.split('_to_')
        before_env = 0 if 'Env1' in parts[0] else 1
        after_env = 0 if 'Env1' in parts[1] else 1
        env[:change_trial] = before_env
        env[change_trial:] = after_env
    else:
        # Single environment
        env[:] = 0 if 'Env1' in scene else 1
    
    return env


def discretize_distance_to_reward(distances):
    """Discretize distance to reward zone into 7 bins.
    
    Bins:
        0: < -50 cm
        1: -50 to -10 cm
        2: -10 to < 0 cm
        3: 0 cm (in reward zone)
        4: >0 to +10 cm  
        5: +10 to +50 cm
        6: > +50 cm
    """
    bins = np.zeros(len(distances), dtype=int)
    bins[distances < -50] = 0
    bins[(distances >= -50) & (distances < -10)] = 1
    bins[(distances >= -10) & (distances < 0)] = 2
    bins[distances == 0] = 3
    bins[(distances > 0) & (distances <= 10)] = 4
    bins[(distances > 10) & (distances <= 50)] = 5
    bins[distances > 50] = 6
    return bins


def discretize_position(positions, n_bins=5):
    """Discretize absolute position into n_bins equal-sized bins."""
    bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
    bins = np.digitize(positions, bin_edges[1:])  # 0 to n_bins-1
    bins = np.clip(bins, 0, n_bins - 1)
    return bins


def discretize_speed(speeds):
    """Discretize speed into 5 bins.
    
    Bins:
        0: < 2 cm/s
        1: 2-10 cm/s
        2: 10-20 cm/s
        3: 20-40 cm/s
        4: > 40 cm/s
    """
    bins = np.zeros(len(speeds), dtype=int)
    bins[speeds < 2] = 0
    bins[(speeds >= 2) & (speeds < 10)] = 1
    bins[(speeds >= 10) & (speeds < 20)] = 2
    bins[(speeds >= 20) & (speeds < 40)] = 3
    bins[speeds >= 40] = 4
    return bins


def compute_distance_to_reward_zone(position, rz_start, rz_end):
    """Compute signed distance from position to nearest edge of reward zone.
    
    Returns:
        distance: negative if before zone, 0 if in zone, positive if past zone
    """
    distance = np.zeros_like(position)
    # Before reward zone
    before = position < rz_start
    distance[before] = position[before] - rz_start
    # In reward zone
    in_zone = (position >= rz_start) & (position <= rz_end)
    distance[in_zone] = 0
    # After reward zone
    after = position > rz_end
    distance[after] = position[after] - rz_end
    return distance


def process_session(nwb_path, subject_id, session_num, show_processing=False):
    """Process a single NWB file and return trial-level data.
    
    Returns dict with keys: neural, input, output, n_neurons, session_info
    or None if session should be skipped.
    """
    t0 = time.time()
    
    # Get scene info
    scene = get_scene_for_session(subject_id, session_num)
    
    # Load NWB data
    f = h5py.File(nwb_path, 'r')
    behav = f['processing']['behavior']['BehavioralTimeSeries']
    ophys = f['processing']['ophys']
    
    # Load behavioral data
    position = behav['position']['data'][()]
    speed = behav['speed']['data'][()]
    lick_raw = behav['lick']['data'][()]
    trial_num = behav['trial number']['data'][()]
    trial_start_signal = behav['trial_start']['data'][()]
    teleport_signal = behav['teleport']['data'][()]
    env_data = behav['environment']['data'][()]
    rzone_data = behav['reward_zone']['data'][()]
    scanning = behav['scanning']['data'][()]
    timestamps = behav['position']['timestamps'][()]
    
    # Load reward events
    reward_data = behav['Reward']['data'][()]
    reward_ts = behav['Reward']['timestamps'][()]
    
    # Load autoreward
    autoreward = behav['autoreward']['data'][()]
    
    # Load neural data - handle multi-plane animals
    planes = sorted(ophys['Deconvolved'].keys())
    deconv_parts = []
    for plane in planes:
        deconv_parts.append(ophys['Deconvolved'][plane]['data'][()])
    deconv = np.concatenate(deconv_parts, axis=1)  # (n_timepoints, total_rois)
    iscell = ophys['ImageSegmentation']['PlaneSegmentation']['iscell'][()]
    
    # Get imaging rate
    imaging_rate = f['general']['optophysiology']['ImagingPlane']['imaging_rate'][()]
    dt = 1.0 / imaging_rate  # seconds per frame
    
    f.close()
    
    n_timepoints = len(position)
    
    # ---- Cell filtering ----
    cell_mask = iscell[:, 0] == 1
    n_cells = cell_mask.sum()
    
    if n_cells < 5:
        print(f"  Skipping: only {n_cells} cells")
        return None
    
    # Filter neural data to cells only
    neural_all = deconv[:, cell_mask]  # (n_timepoints, n_cells)
    
    # ---- Trial boundaries ----
    # Find trial start and teleport indices
    trial_start_inds = np.where(trial_start_signal > 0)[0]
    teleport_inds = np.where(teleport_signal > 0)[0]
    
    n_trials = min(len(trial_start_inds), len(teleport_inds))
    
    # Match trial starts to teleports
    # Each trial goes from trial_start to the next teleport
    matched_starts = []
    matched_teleports = []
    
    for i in range(len(trial_start_inds)):
        start = trial_start_inds[i]
        # Find the next teleport after this start
        future_teleports = teleport_inds[teleport_inds > start]
        if len(future_teleports) > 0:
            matched_starts.append(start)
            matched_teleports.append(future_teleports[0])
    
    trial_start_inds = np.array(matched_starts)
    teleport_inds = np.array(matched_teleports)
    n_trials = len(trial_start_inds)
    
    if n_trials < 2:
        print(f"  Skipping: only {n_trials} trials")
        return None
    
    # ---- Get reward zone info from scene ----
    rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
    
    # ---- Get environment type ----
    env_per_trial = get_environment_from_scene(scene, n_trials)
    
    # ---- Determine reward/omission per trial ----
    isreward = np.zeros(n_trials, dtype=int)
    for i in range(n_trials):
        start = trial_start_inds[i]
        stop = teleport_inds[i]
        # Check if any reward was delivered in this trial
        trial_rewards = np.sum((reward_ts >= timestamps[start]) & (reward_ts <= timestamps[stop]))
        # Also check if rzone was active (to distinguish true omissions)
        rzone_active = np.any(rzone_data[start:stop+1] > 0)
        isreward[i] = 1 if (trial_rewards > 0 and rzone_active) else 0
    
    # ---- Previous trial outcome ----
    prev_outcome = np.zeros(n_trials, dtype=int)
    prev_outcome[1:] = isreward[:-1]  # First trial has no previous, default to 0
    
    # ---- Reward zone label per trial (0=A, 1=B, 2=C) ----
    rz_label_idx = np.zeros(n_trials, dtype=int)
    for i in range(n_trials):
        if rz_labels[i] == 'A':
            rz_label_idx[i] = 0
        elif rz_labels[i] == 'B':
            rz_label_idx[i] = 1
        elif rz_labels[i] == 'C':
            rz_label_idx[i] = 2
    
    # ---- Process each trial ----
    neural_trials = []
    input_trials = []
    output_trials = []
    
    for i in range(n_trials):
        start = trial_start_inds[i]
        stop = teleport_inds[i]
        
        # Trial data (start to teleport, inclusive of start, exclusive of teleport)
        trial_len = stop - start
        if trial_len < 3:
            # Very short trial, skip
            continue
        
        # Neural data: (n_cells, trial_len)
        trial_neural = neural_all[start:stop, :].T.astype(np.float32)  # (n_cells, trial_len)
        
        # ---- Inputs ----
        # Time from start of trial in seconds
        time_from_start = np.arange(trial_len) * dt  # (trial_len,)
        
        # Environment type (binary, per trial)
        env_type = env_per_trial[i]
        
        # Trial number (per trial)
        trial_number = i
        
        # Previous trial outcome (per trial)
        prev_out = prev_outcome[i]
        
        # Stack inputs: (4, trial_len) for time-varying, or (4,) mixed
        # time_from_start is time-varying, others are per-trial
        trial_input = np.array([time_from_start[0], float(env_type), float(trial_number), float(prev_out)], dtype=np.float32)
        # Actually, time_from_start is time-varying, so we need shape (4, trial_len) or mixed
        # Per the format: (n_input, n_timepoints) or (n_input,)
        # Since time_from_start varies, we need (4, trial_len)
        trial_input = np.zeros((4, trial_len), dtype=np.float32)
        trial_input[0, :] = time_from_start
        trial_input[1, :] = float(env_type)
        trial_input[2, :] = float(trial_number)
        trial_input[3, :] = float(prev_out)
        
        # ---- Outputs ----
        trial_pos = position[start:stop]
        trial_speed = speed[start:stop]
        trial_lick = lick_raw[start:stop].copy()
        
        # Lick processing: clip to binary
        # First check for lick sensor error (>35% samples with cumulative lick > 2)
        if np.sum(trial_lick > 2) / len(trial_lick) > 0.35:
            trial_lick[:] = 0  # Set to 0 for error trials
        else:
            trial_lick[trial_lick > 1] = 1
        trial_lick = trial_lick.astype(int)
        
        # Distance to reward zone
        rz_start = rz_coords[i, 0]
        rz_end = rz_coords[i, 1]
        dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
        dist_bins = discretize_distance_to_reward(dist_to_rz)
        
        # Absolute position bins
        # Clip position to [0, TRACK_LENGTH] for binning
        pos_clipped = np.clip(trial_pos, 0, TRACK_LENGTH)
        pos_bins = discretize_position(pos_clipped)
        
        # Speed bins
        speed_bins = discretize_speed(np.abs(trial_speed))
        
        # Reward zone location (per trial)
        rz_loc = rz_label_idx[i]
        
        # Reward outcome (per trial)
        rew_out = isreward[i]
        
        # Stack outputs: mix of time-varying and per-trial
        # Time-varying: dist_bins, pos_bins, speed_bins, lick (4 vars)
        # Per-trial: rz_loc, rew_out (2 vars)
        # Total: 6 outputs
        # Shape: (6, trial_len) for time-varying, (6,) for per-trial
        n_outputs = 6
        trial_output = np.zeros((n_outputs, trial_len), dtype=np.int64)
        trial_output[0, :] = dist_bins
        trial_output[1, :] = pos_bins
        trial_output[2, :] = speed_bins
        trial_output[3, :] = trial_lick
        trial_output[4, :] = rz_loc  # per-trial, broadcast
        trial_output[5, :] = rew_out  # per-trial, broadcast
        
        neural_trials.append(trial_neural)
        input_trials.append(trial_input)
        output_trials.append(trial_output)
    
    if len(neural_trials) < 2:
        print(f"  Skipping: only {len(neural_trials)} valid trials")
        return None
    
    elapsed = time.time() - t0
    print(f"  Processed {len(neural_trials)} trials, {n_cells} neurons in {elapsed:.1f}s")
    
    # ---- Visualization ----
    if show_processing:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(6, 2, figsize=(16, 20))
            fig.suptitle(f'm{subject_id} ses-{session_num:02d} ({scene})', fontsize=14)
            
            # Pick a sample trial
            sample_trial = min(5, len(neural_trials) - 1)
            trial_len_s = neural_trials[sample_trial].shape[1]
            t_axis = np.arange(trial_len_s) * dt
            
            # Plot neural activity (mean across neurons)
            axes[0, 0].plot(t_axis, neural_trials[sample_trial].mean(axis=0))
            axes[0, 0].set_title(f'Mean neural activity (trial {sample_trial})')
            axes[0, 0].set_xlabel('Time (s)')
            
            # Plot position
            axes[1, 0].plot(t_axis, output_trials[sample_trial][1, :])
            axes[1, 0].set_title('Position bin')
            
            # Plot speed
            axes[2, 0].plot(t_axis, output_trials[sample_trial][2, :])
            axes[2, 0].set_title('Speed bin')
            
            # Plot lick
            axes[3, 0].plot(t_axis, output_trials[sample_trial][3, :])
            axes[3, 0].set_title('Lick')
            
            # Plot distance to reward
            axes[4, 0].plot(t_axis, output_trials[sample_trial][0, :])
            axes[4, 0].set_title('Distance to reward bin')
            
            # Plot inputs
            axes[5, 0].plot(t_axis, input_trials[sample_trial][0, :])
            axes[5, 0].set_title('Time from trial start (s)')
            
            # Right column: distributions across all trials
            all_dist = np.concatenate([o[0, :] for o in output_trials])
            axes[0, 1].hist(all_dist, bins=7, range=(-0.5, 6.5))
            axes[0, 1].set_title('Distance to reward distribution')
            
            all_pos = np.concatenate([o[1, :] for o in output_trials])
            axes[1, 1].hist(all_pos, bins=5, range=(-0.5, 4.5))
            axes[1, 1].set_title('Position bin distribution')
            
            all_speed = np.concatenate([o[2, :] for o in output_trials])
            axes[2, 1].hist(all_speed, bins=5, range=(-0.5, 4.5))
            axes[2, 1].set_title('Speed bin distribution')
            
            all_lick = np.concatenate([o[3, :] for o in output_trials])
            axes[3, 1].hist(all_lick, bins=2, range=(-0.5, 1.5))
            axes[3, 1].set_title('Lick distribution')
            
            rz_locs = np.array([o[4, 0] for o in output_trials])
            axes[4, 1].hist(rz_locs, bins=3, range=(-0.5, 2.5))
            axes[4, 1].set_title('Reward zone location')
            
            rew_outs = np.array([o[5, 0] for o in output_trials])
            axes[5, 1].bar(['No', 'Yes'], [np.sum(rew_outs==0), np.sum(rew_outs==1)])
            axes[5, 1].set_title('Reward outcome')
            
            plt.tight_layout()
            plt.savefig(f'processing_m{subject_id}_ses{session_num:02d}.png', dpi=100)
            plt.close()
            print(f"  Saved processing plot")
        except Exception as e:
            print(f"  Warning: Could not create processing plot: {e}")
    
    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'n_neurons': n_cells,
        'session_info': {
            'subject': f'm{subject_id}',
            'session': session_num,
            'scene': scene,
            'n_trials': len(neural_trials),
            'n_neurons': n_cells,
            'imaging_rate': imaging_rate,
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Convert NWB data to decoder format')
    parser.add_argument('output', help='Output pickle file path')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--full', action='store_true', help='Process all sessions (default)')
    parser.add_argument('--show-processing', action='store_true', help='Save processing plots')
    args = parser.parse_args()
    
    if not args.sample:
        args.full = True
    
    data_dir = 'data'
    
    # Discover all NWB files
    all_files = []
    for sub_dir in sorted(os.listdir(data_dir)):
        if not sub_dir.startswith('sub-'):
            continue
        subject_id = sub_dir.replace('sub-m', '')
        sub_path = os.path.join(data_dir, sub_dir)
        for fname in sorted(os.listdir(sub_path)):
            if not fname.endswith('.nwb'):
                continue
            ses_num = int(fname.split('_ses-')[1].split('_')[0])
            all_files.append({
                'path': os.path.join(sub_path, fname),
                'subject_id': subject_id,
                'session_num': ses_num,
            })
    
    print(f"Found {len(all_files)} NWB files")
    
    if args.sample:
        # Pick 2 sessions from different subjects
        all_files = [all_files[0], all_files[len(all_files)//2]]
        print(f"Sample mode: processing {len(all_files)} sessions")
    
    # Process all sessions
    neural_list = []
    input_list = []
    output_list = []
    subjects = []
    subject_idx = []
    brain_region_idx_list = []
    session_infos = []
    
    subject_map = {}  # subject_id -> index
    
    total_t0 = time.time()
    
    for i, file_info in enumerate(all_files):
        subject_id = file_info['subject_id']
        session_num = file_info['session_num']
        nwb_path = file_info['path']
        
        print(f"\n[{i+1}/{len(all_files)}] Processing m{subject_id} ses-{session_num:02d}...")
        
        try:
            result = process_session(
                nwb_path, subject_id, session_num,
                show_processing=args.show_processing
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        if result is None:
            continue
        
        # Track subject
        sub_name = f'm{subject_id}'
        if sub_name not in subject_map:
            subject_map[sub_name] = len(subjects)
            subjects.append(sub_name)
        
        neural_list.append(result['neural'])
        input_list.append(result['input'])
        output_list.append(result['output'])
        subject_idx.append(subject_map[sub_name])
        brain_region_idx_list.append(np.zeros(result['n_neurons'], dtype=int))  # All CA1
        session_infos.append(result['session_info'])
    
    total_elapsed = time.time() - total_t0
    print(f"\n\nTotal processing time: {total_elapsed:.1f}s")
    print(f"Processed {len(neural_list)} sessions")
    
    if len(neural_list) == 0:
        print("ERROR: No sessions processed!")
        sys.exit(1)
    
    # Get imaging rate from first session
    imaging_rate = session_infos[0]['imaging_rate']
    dt_ms = 1000.0 / imaging_rate
    
    # Build output data structure
    data = {
        'neural': neural_list,
        'input': input_list,
        'output': output_list,
        
        'subjects': subjects,
        'subject_idx': np.array(subject_idx),
        
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
            ['< -50cm', '-50 to -10cm', '-10 to 0cm', '0cm (in zone)', '>0 to +10cm', '+10 to +50cm', '> +50cm'],
            ['0-90cm', '90-180cm', '180-270cm', '270-360cm', '360-450cm'],
            ['< 2cm/s', '2-10cm/s', '10-20cm/s', '20-40cm/s', '> 40cm/s'],
            ['no_lick', 'lick'],
            ['A', 'B', 'C'],
            ['no_reward', 'reward'],
        ],
        
        'metadata': {
            'task_description': 'Virtual reality navigation with hidden reward zone in hippocampal CA1. '
                               'Mice navigate a 450cm linear track with a hidden 50cm reward zone at one of '
                               'three locations (A, B, C). Reward zone switches between locations across sessions.',
            'time_bin_size': dt_ms,
            'temporal_alignment_event': 'trial_start',
            'off_start': 0.0,
            'off_end': None,
            'track_length_cm': TRACK_LENGTH,
            'reward_zones': {'A': [80, 130], 'B': [200, 250], 'C': [320, 370]},
            'imaging_rate_hz': imaging_rate,
            'session_info': session_infos,
            'n_sessions': len(neural_list),
            'n_subjects': len(subjects),
        }
    }
    
    # Print summary
    total_trials = sum(len(s) for s in neural_list)
    total_neurons = sum(s[0].shape[0] for s in neural_list)
    neurons_per_session = [s[0].shape[0] for s in neural_list]
    trials_per_session = [len(s) for s in neural_list]
    
    print(f"\n=== Summary ===")
    print(f"Subjects: {len(subjects)}")
    print(f"Sessions: {len(neural_list)}")
    print(f"Total trials: {total_trials}")
    print(f"Trials/session: {np.mean(trials_per_session):.1f} +/- {np.std(trials_per_session):.1f}")
    print(f"Total neurons: {total_neurons}")
    print(f"Neurons/session: {np.mean(neurons_per_session):.1f} +/- {np.std(neurons_per_session):.1f}")
    print(f"  range: {np.min(neurons_per_session)}-{np.max(neurons_per_session)}")
    print(f"Time bin: {dt_ms:.2f} ms")
    
    # Output distributions
    all_outputs = {}
    for out_idx, out_name in enumerate(data['output_names']):
        vals = []
        for sess in output_list:
            for trial in sess:
                vals.append(trial[out_idx, :])
        all_vals = np.concatenate(vals)
        unique, counts = np.unique(all_vals, return_counts=True)
        fracs = counts / counts.sum()
        print(f"\n{out_name}:")
        for u, c, fr in zip(unique, counts, fracs):
            print(f"  {int(u)}: {c} ({fr:.3f})")
    
    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
    
    file_size = os.path.getsize(args.output)
    print(f"Saved {file_size / 1e6:.1f} MB")
    print("Done!")


if __name__ == '__main__':
    main()
