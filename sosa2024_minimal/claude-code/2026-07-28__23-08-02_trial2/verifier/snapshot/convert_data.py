"""
Convert NWB data from Sosa, Plitt & Giocomo (2025) to decoder format.

Reference: "A flexible hippocampal population code for experience relative to reward"
Data: 2-photon calcium imaging of CA1 neurons in mice performing a VR navigation task.
"""

import numpy as np
import h5py
import os
import pickle
import sys
import warnings
from collections import defaultdict

# ============================================================
# Constants from the paper and code
# ============================================================
TRACK_LENGTH = 450  # cm
REWARD_ZONES = {
    'A': (80, 130),   # zone A: 80-130 cm (50 cm span)
    'B': (200, 250),  # zone B: 200-250 cm
    'C': (320, 370),  # zone C: 320-370 cm
}
SWITCH_TRIAL = 30  # reward zone switches after 30 trials
SPEED_CORR_THRESHOLD = 0.5  # for excluding putative interneurons
LICK_ERROR_THRESHOLD = 0.3  # fraction of samples with cumulative lick > 2

# Subject mapping: NWB sub-mN -> GCAMPN
SUBJECT_MAP = {
    'sub-m3': 'GCAMP3',
    'sub-m4': 'GCAMP4',
    'sub-m7': 'GCAMP7',
    'sub-m11': 'GCAMP11',
    'sub-m12': 'GCAMP12',
    'sub-m13': 'GCAMP13',
    'sub-m14': 'GCAMP14',
    'sub-m15': 'GCAMP15',
    'sub-m17': 'GCAMP17',
    'sub-m18': 'GCAMP18',
    'sub-m19': 'GCAMP19',
}

# Session scene info from sessions_dict.py
# Maps (subject, exp_day) -> scene name
# We only need to encode the reward zone info
SESSIONS_INFO = {
    'GCAMP3': {
        1: 'Env1_LocationC', 2: 'Env1_LocationC',
        3: 'Env1_LocationC_to_A', 4: 'Env1_LocationA',
        5: 'Env1_LocationA_to_B', 6: 'Env1_LocationB',
        7: 'Env1_LocationB_to_C', 8: 'Env1_C_to_Env2_B',
        9: 'Env2_LocationB', 10: 'Env2_LocationB_to_A',
        11: 'Env2_LocationA', 12: 'Env2_LocationA_to_C',
        13: 'Env2_LocationC', 14: 'Env2_LocationC_to_B',
    },
    'GCAMP4': {
        1: 'Env1_LocationB', 2: 'Env1_LocationB',
        3: 'Env1_LocationB_to_A', 4: 'Env1_LocationA',
        5: 'Env1_LocationA_to_C', 6: 'Env1_LocationC',
        7: 'Env1_LocationC_to_B', 8: 'Env1_B_to_Env2_C',
        9: 'Env2_LocationC', 10: 'Env2_LocationC_to_A',
        11: 'Env2_LocationA', 12: 'Env2_LocationA_to_B',
        13: 'Env2_LocationB', 14: 'Env2_LocationB_to_C',
    },
    'GCAMP7': {
        1: 'Env1_LocationA', 2: 'Env1_LocationA',
        3: 'Env1_LocationA_to_C', 4: 'Env1_LocationC',
        5: 'Env1_LocationC_to_B', 6: 'Env1_LocationB',
        7: 'Env1_LocationB_to_A', 8: 'Env1_A_to_Env2_B',
        9: 'Env2_LocationB', 10: 'Env2_LocationB_to_C',
        11: 'Env2_LocationC', 12: 'Env2_LocationC_to_A',
        13: 'Env2_LocationA', 14: 'Env2_LocationA_to_B',
    },
    'GCAMP11': {
        3: 'Env1_LocationB_to_A', 4: 'Env1_LocationA',
        5: 'Env1_LocationA_to_C', 6: 'Env1_LocationC',
        7: 'Env1_LocationC_to_B', 8: 'Env1_B_to_Env2_C',
        9: 'Env2_LocationC', 10: 'Env2_LocationC_to_A',
        11: 'Env2_LocationA', 12: 'Env2_LocationA_to_B',
        13: 'Env2_LocationB', 14: 'Env2_LocationB_to_C',
    },
    'GCAMP12': {
        1: 'Env1_LocationB', 2: 'Env1_LocationB',
        3: 'Env1_LocationB_to_C', 4: 'Env1_LocationC',
        5: 'Env1_LocationC_to_A', 6: 'Env1_LocationA',
        7: 'Env1_LocationA_to_B', 8: 'Env1_B_to_Env2_A',
        9: 'Env2_LocationA', 10: 'Env2_LocationA_to_C',
        11: 'Env2_LocationC', 12: 'Env2_LocationC_to_B',
        13: 'Env2_LocationB', 14: 'Env2_LocationB_to_A',
    },
    'GCAMP13': {
        1: 'Env1_LocationC', 2: 'Env1_LocationC',
        3: 'Env1_LocationC_to_B', 4: 'Env1_LocationB',
        5: 'Env1_LocationB_to_A', 6: 'Env1_LocationA',
        7: 'Env1_LocationA_to_C', 8: 'Env1_C_to_Env2_A',
        9: 'Env2_LocationA', 10: 'Env2_LocationA_to_B',
        11: 'Env2_LocationB', 12: 'Env2_LocationB_to_C',
        13: 'Env2_LocationC', 14: 'Env2_LocationC_to_A',
    },
    'GCAMP14': {
        1: 'Env1_LocationA', 2: 'Env1_LocationA',
        3: 'Env1_LocationA_to_B', 4: 'Env1_LocationB',
        5: 'Env1_LocationB_to_C', 6: 'Env1_LocationC',
        7: 'Env1_LocationC_to_A', 8: 'Env1_A_to_Env2_C',
        9: 'Env2_LocationC', 10: 'Env2_LocationC_to_B',
        11: 'Env2_LocationB', 12: 'Env2_LocationB_to_A',
        13: 'Env2_LocationA', 14: 'Env2_LocationA_to_C',
    },
    'GCAMP15': {
        1: 'Env1_LocationA', 2: 'Env1_LocationA',
        3: 'Env1_LocationA_to_C', 4: 'Env1_LocationC',
        5: 'Env1_LocationC_to_B', 6: 'Env1_LocationB',
        7: 'Env1_LocationB_to_A', 8: 'Env1_A_to_Env2_B',
        9: 'Env2_LocationB', 10: 'Env2_LocationB_to_C',
        11: 'Env2_LocationC', 12: 'Env2_LocationC_to_A',
        13: 'Env2_LocationA', 14: 'Env2_LocationA_to_B',
    },
    'GCAMP17': {
        1: 'Env2_LocationB', 2: 'Env2_LocationB',
        3: 'Env2_LocationB_to_C', 4: 'Env2_LocationC',
        5: 'Env2_LocationC_to_A', 6: 'Env2_LocationA',
        7: 'Env2_LocationA_to_B', 8: 'Env2_B_to_Env1_A',
        9: 'Env1_LocationA', 10: 'Env1_LocationA_to_C',
        11: 'Env1_LocationC', 12: 'Env1_LocationC_to_B',
        13: 'Env1_LocationB', 14: 'Env1_LocationB_to_A',
    },
    'GCAMP18': {
        1: 'Env2_LocationA', 2: 'Env2_LocationA',
        3: 'Env2_LocationA_to_B', 4: 'Env2_LocationB',
        5: 'Env2_LocationB_to_C', 6: 'Env2_LocationC',
        7: 'Env2_LocationC_to_A', 8: 'Env2_A_to_Env1_C',
        9: 'Env1_LocationC', 10: 'Env1_LocationC_to_B',
        11: 'Env1_LocationB', 12: 'Env1_LocationB_to_A',
        13: 'Env1_LocationA', 14: 'Env1_LocationA_to_C',
    },
    'GCAMP19': {
        1: 'Env1_LocationC', 2: 'Env1_LocationC',
        3: 'Env1_LocationC_to_B', 4: 'Env1_LocationB',
        5: 'Env1_LocationB_to_A', 6: 'Env1_LocationA',
        7: 'Env1_LocationA_to_C', 8: 'Env1_C_to_Env2_A',
        9: 'Env2_LocationA', 10: 'Env2_LocationA_to_B',
        11: 'Env2_LocationB', 12: 'Env2_LocationB_to_C',
        13: 'Env2_LocationC', 14: 'Env2_LocationC_to_A',
    },
}


def parse_scene_reward_zones(scene, n_trials, change_trial=SWITCH_TRIAL):
    """Parse the scene name to determine reward zone labels per trial.

    Returns array of 'A', 'B', or 'C' per trial.
    """
    rz_labels = np.empty(n_trials, dtype='U1')

    # Determine the zone(s) from scene name
    # Single-zone sessions
    if scene.endswith('_LocationA') or scene.endswith('LocationA'):
        rz_labels[:] = 'A'
    elif scene.endswith('_LocationB') or scene.endswith('LocationB'):
        rz_labels[:] = 'B'
    elif scene.endswith('_LocationC') or scene.endswith('LocationC'):
        rz_labels[:] = 'C'
    else:
        # Switch session: parse from->to
        # Patterns: "LocationX_to_Y", "X_to_EnvN_Y"
        # Extract the two zone letters
        zone_before, zone_after = parse_switch_zones(scene)
        ct = min(change_trial, n_trials)
        rz_labels[:ct] = zone_before
        rz_labels[ct:] = zone_after

    return rz_labels


def parse_switch_zones(scene):
    """Parse switch session scene name to get before/after zones."""
    # Patterns:
    # 'Env1_LocationA_to_B' -> A, B
    # 'Env1_LocationA_to_C' -> A, C
    # 'Env1_A_to_Env2_B' -> A, B (environment switch)
    # 'Env1_B_to_Env2_C' -> B, C
    # 'Env2_B_to_Env1_A' -> B, A

    scene_upper = scene.upper()

    # Find 'TO' separator
    if '_TO_' in scene_upper:
        parts = scene.split('_to_')
        if len(parts) != 2:
            parts = scene.split('_to_')

        before_part = parts[0]
        after_part = parts[1]

        # Extract zone letter from before part (last character of Location*)
        zone_before = None
        for z in ['A', 'B', 'C']:
            if before_part.endswith(z) or f'Location{z}' in before_part or f'_{z}_' in before_part:
                zone_before = z
                break
        if zone_before is None:
            # Try pattern like 'Env1_C_to_Env2_A'
            for z in ['A', 'B', 'C']:
                if f'_{z}' in before_part:
                    zone_before = z

        # Extract zone letter from after part
        zone_after = None
        for z in ['A', 'B', 'C']:
            if after_part.endswith(z) or f'Location{z}' in after_part or z == after_part[-1]:
                zone_after = z
                break
        if zone_after is None:
            for z in ['A', 'B', 'C']:
                if z in after_part:
                    zone_after = z

        if zone_before and zone_after:
            return zone_before, zone_after

    raise ValueError(f"Cannot parse switch zones from scene: {scene}")


def parse_scene_environment(scene):
    """Determine environment type from scene name.
    Returns 0 for ENV1, 1 for ENV2.
    For switch sessions with env change (day 8), returns (env_before, env_after).
    """
    if 'Env1' in scene and 'Env2' not in scene:
        return 0, None
    elif 'Env2' in scene and 'Env1' not in scene:
        return 1, None
    elif 'Env1' in scene and 'Env2' in scene:
        # Environment switch (day 8)
        # e.g., 'Env1_A_to_Env2_B' or 'Env2_B_to_Env1_A'
        parts = scene.split('_to_')
        env_before = 0 if 'Env1' in parts[0] else 1
        env_after = 0 if 'Env1' in parts[1] else 1
        return env_before, env_after
    else:
        return 0, None  # default


def get_reward_zone_coords(zone_label):
    """Get (start, end) coordinates for a reward zone label."""
    return REWARD_ZONES[zone_label]


def compute_distance_to_reward_zone(position, rz_start, rz_end):
    """Compute signed distance from position to nearest point in reward zone.

    Negative = before reward zone, positive = after reward zone, 0 = inside.
    """
    dist = np.zeros_like(position, dtype=float)
    before = position < rz_start
    inside = (position >= rz_start) & (position <= rz_end)
    after = position > rz_end

    dist[before] = position[before] - rz_start
    dist[inside] = 0.0
    dist[after] = position[after] - rz_end

    return dist


def discretize_distance_to_reward(distance):
    """Discretize distance to reward zone.
    Bins:
      0: < -50 cm
      1: -50 to -10 cm
      2: -10 cm to < 0 cm
      3: 0 cm (inside zone)
      4: >0 cm to +10 cm
      5: +10 to +50 cm
      6: > +50 cm
    """
    out = np.zeros_like(distance, dtype=int)
    out[distance < -50] = 0
    out[(distance >= -50) & (distance < -10)] = 1
    out[(distance >= -10) & (distance < 0)] = 2
    out[distance == 0] = 3
    out[(distance > 0) & (distance <= 10)] = 4
    out[(distance > 10) & (distance <= 50)] = 5
    out[distance > 50] = 6
    return out


def discretize_position(position, n_bins=5):
    """Discretize position into n_bins equal-sized bins across 0-450 cm.
    Bins: 0-90, 90-180, 180-270, 270-360, 360-450
    """
    bin_size = TRACK_LENGTH / n_bins
    out = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
    return out


def discretize_speed(speed):
    """Discretize speed.
    Bins:
      0: < 2 cm/s
      1: 2-10 cm/s
      2: 10-20 cm/s
      3: 20-40 cm/s
      4: > 40 cm/s
    """
    out = np.zeros_like(speed, dtype=int)
    out[speed < 2] = 0
    out[(speed >= 2) & (speed < 10)] = 1
    out[(speed >= 10) & (speed < 20)] = 2
    out[(speed >= 20) & (speed < 40)] = 3
    out[speed >= 40] = 4
    return out


def correct_lick_sensor_errors(lick_data, tstart_indices, teleport_indices, threshold=LICK_ERROR_THRESHOLD):
    """Correct lick sensor errors per the paper's method.

    Trials where >threshold fraction of samples have cumulative lick count >2
    are set to NaN.
    """
    licks = np.copy(lick_data)
    error_trials = []

    for i, (start, end) in enumerate(zip(tstart_indices, teleport_indices)):
        trial_licks = licks[start:end]
        if len(trial_licks) == 0:
            continue
        frac_high = np.sum(trial_licks > 2) / len(trial_licks)
        if frac_high > threshold:
            licks[start:end] = np.nan
            error_trials.append(i)

    return licks, error_trials


def identify_interneurons(neural_data, speed_data, valid_mask, threshold=SPEED_CORR_THRESHOLD):
    """Identify putative interneurons by correlation with running speed.

    Per the paper: Pearson correlation > 0.5 between dF/F and speed.
    We use the deconvolved activity here as we don't have separate dF/F,
    but the NWB files contain fluorescence which we can use instead.

    Returns boolean mask where True = interneuron.
    """
    n_neurons = neural_data.shape[1]
    is_interneuron = np.zeros(n_neurons, dtype=bool)

    for c in range(n_neurons):
        neural_ts = neural_data[valid_mask, c]
        speed_ts = speed_data[valid_mask]

        if np.std(neural_ts) == 0 or np.std(speed_ts) == 0:
            continue

        r = np.corrcoef(neural_ts, speed_ts)[0, 1]
        if r > threshold:
            is_interneuron[c] = True

    return is_interneuron


def load_nwb_session(filepath):
    """Load data from a single NWB file."""
    with h5py.File(filepath, 'r') as f:
        bts = f['processing/behavior/BehavioralTimeSeries']
        ophys = f['processing/ophys']
        seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']

        # Behavioral data
        position = bts['position/data'][:]
        speed = bts['speed/data'][:]
        lick = bts['lick/data'][:]
        trial_start = bts['trial_start/data'][:]
        teleport = bts['teleport/data'][:]
        trial_num = bts['trial number/data'][:]
        environment = bts['environment/data'][:]
        reward_zone = bts['reward_zone/data'][:]
        timestamps = bts['position/timestamps'][:]
        autoreward = bts['autoreward/data'][:]

        # Reward events (separate timestamps)
        reward_data = bts['Reward/data'][:]
        reward_ts = bts['Reward/timestamps'][:]

        # ROI info
        iscell = seg['iscell'][:]
        plane_idx = seg['planeIdx'][:]

        # Neural data - handle multi-plane
        deconv_keys = list(ophys['Deconvolved'].keys())
        fluor_keys = list(ophys['Fluorescence'].keys())

        # Concatenate planes (paper says "ROIs were identified separately per plane,
        # but planes were pooled for all analyses")
        deconv_list = []
        fluor_list = []
        for pk in sorted(deconv_keys):
            deconv_list.append(ophys['Deconvolved'][pk]['data'][:])
        for fk in sorted(fluor_keys):
            fluor_list.append(ophys['Fluorescence'][fk]['data'][:])

        deconvolved = np.concatenate(deconv_list, axis=1)
        fluorescence = np.concatenate(fluor_list, axis=1)

        # Ensure behavioral and neural data have the same number of timepoints
        n_behavior = len(position)
        n_neural = deconvolved.shape[0]
        if n_behavior != n_neural:
            min_len = min(n_behavior, n_neural)
            position = position[:min_len]
            speed = speed[:min_len]
            lick = lick[:min_len]
            trial_start = trial_start[:min_len]
            teleport = teleport[:min_len]
            trial_num = trial_num[:min_len]
            environment = environment[:min_len]
            reward_zone = reward_zone[:min_len]
            timestamps = timestamps[:min_len]
            autoreward = autoreward[:min_len]
            deconvolved = deconvolved[:min_len]
            fluorescence = fluorescence[:min_len]

        # Imaging rate
        imaging_rate = f['acquisition/TwoPhotonSeries/imaging_plane/imaging_rate'][()]

        # Subject info
        subject_id = f['general/subject/subject_id'][()]
        if isinstance(subject_id, bytes):
            subject_id = subject_id.decode()
        session_id = f['general/session_id'][()]
        if isinstance(session_id, bytes):
            session_id = session_id.decode()

    return {
        'position': position,
        'speed': speed,
        'lick': lick,
        'trial_start': trial_start,
        'teleport': teleport,
        'trial_num': trial_num,
        'environment': environment,
        'reward_zone_ts': reward_zone,
        'timestamps': timestamps,
        'autoreward': autoreward,
        'reward_data': reward_data,
        'reward_ts': reward_ts,
        'iscell': iscell,
        'plane_idx': plane_idx,
        'deconvolved': deconvolved,
        'fluorescence': fluorescence,
        'imaging_rate': imaging_rate,
        'subject_id': subject_id,
        'session_id': session_id,
    }


def process_session(nwb_data, scene, exp_day):
    """Process a single session into trials.

    Returns dict with neural, input, output arrays per trial,
    plus metadata.
    """
    # Get trial boundaries
    tstart_idx = np.where(nwb_data['trial_start'] > 0)[0]
    teleport_idx = np.where(nwb_data['teleport'] > 0)[0]

    n_trials = min(len(tstart_idx), len(teleport_idx))
    tstart_idx = tstart_idx[:n_trials]
    teleport_idx = teleport_idx[:n_trials]

    if n_trials < 2:
        print(f"  Skipping session: only {n_trials} trials")
        return None

    # Filter neurons by iscell (manual curation from Suite2P)
    cell_mask = nwb_data['iscell'][:, 0] == 1
    n_total_rois = len(cell_mask)
    n_cells_after_iscell = cell_mask.sum()

    # Get neural data for valid cells
    deconvolved = nwb_data['deconvolved'][:, cell_mask]
    fluorescence = nwb_data['fluorescence'][:, cell_mask]

    # Identify and exclude putative interneurons using fluorescence (dF/F proxy)
    # Per the paper: correlation of dF/F with running speed > 0.5
    # Use non-NaN timepoints where animal is on the track
    valid_mask = nwb_data['position'] > 0  # exclude teleport period (pos = -500)

    is_interneuron = identify_interneurons(
        fluorescence, nwb_data['speed'], valid_mask,
        threshold=SPEED_CORR_THRESHOLD
    )

    n_interneurons = is_interneuron.sum()
    neuron_mask = ~is_interneuron
    n_neurons = neuron_mask.sum()

    if n_neurons < 1:
        print(f"  Skipping session: no valid neurons")
        return None

    # Apply neuron mask
    neural = deconvolved[:, neuron_mask]  # (timepoints, n_neurons)

    # Correct lick sensor errors
    licks_corrected, error_trials = correct_lick_sensor_errors(
        nwb_data['lick'], tstart_idx, teleport_idx,
        threshold=LICK_ERROR_THRESHOLD
    )

    # Determine reward zone labels per trial
    rz_labels = parse_scene_reward_zones(scene, n_trials)

    # Determine environment type per trial
    env_before, env_after = parse_scene_environment(scene)
    env_per_trial = np.full(n_trials, env_before, dtype=int)
    if env_after is not None:
        # Environment switch session
        ct = min(SWITCH_TRIAL, n_trials)
        env_per_trial[ct:] = env_after

    # Determine reward outcome per trial
    reward_per_trial = np.zeros(n_trials, dtype=int)
    timestamps = nwb_data['timestamps']
    reward_ts = nwb_data['reward_ts']

    for i in range(n_trials):
        start_t = timestamps[tstart_idx[i]]
        end_t = timestamps[teleport_idx[i]]
        if np.any((reward_ts >= start_t) & (reward_ts <= end_t)):
            reward_per_trial[i] = 1

    # Previous trial outcome (first trial has no previous, set to 0)
    prev_outcome = np.zeros(n_trials, dtype=int)
    prev_outcome[1:] = reward_per_trial[:-1]

    # Frame time
    frame_time = 1.0 / nwb_data['imaging_rate']

    # Process each trial
    trial_neural = []
    trial_input = []
    trial_output = []
    valid_trial_indices = []

    for i in range(n_trials):
        start = tstart_idx[i]
        end = teleport_idx[i]

        if end <= start:
            continue

        n_timepoints = end - start
        if n_timepoints < 2:
            continue

        # Neural data: (n_neurons, n_timepoints)
        trial_n = neural[start:end, :].T.astype(np.float32)

        # Position and speed for this trial
        pos = nwb_data['position'][start:end]
        spd = np.abs(nwb_data['speed'][start:end])  # use absolute speed
        lck = licks_corrected[start:end]

        # Handle NaN licks (from error correction) - set to 0
        lck = np.nan_to_num(lck, nan=0.0)
        # Binarize licks: >0 = 1
        lck_binary = (lck > 0).astype(int)

        # Time from start of trial (seconds)
        time_from_start = (np.arange(n_timepoints) * frame_time).astype(np.float32)

        # ---- INPUTS ----
        # 1. Time from start of trial (continuous)
        # 2. Environment type (binary, per trial)
        # 3. Trial number (continuous, per trial)
        # 4. Previous trial outcome (binary, per trial)

        input_arr = np.array([
            time_from_start[0] if False else 0,  # placeholder
        ])
        # Actually, inputs can be time-varying or per-trial
        # Time from start is time-varying, rest are per-trial
        # Shape: (n_input, n_timepoints) or (n_input,)
        # Per spec: (n_input, n_timepoints) or (n_input)
        # Let's make time_from_start time-varying and others scalar

        input_time_varying = time_from_start  # (n_timepoints,)
        input_per_trial = np.array([
            float(env_per_trial[i]),
            float(i),  # trial number
            float(prev_outcome[i]),
        ], dtype=np.float32)

        # Combine: first row is time-varying, rest are repeated per trial
        input_arr = np.zeros((4, n_timepoints), dtype=np.float32)
        input_arr[0, :] = time_from_start
        input_arr[1, :] = env_per_trial[i]
        input_arr[2, :] = float(i)
        input_arr[3, :] = float(prev_outcome[i])

        # ---- OUTPUTS ----
        # 1. Distance to reward zone (discretized, time-varying)
        rz_start, rz_end = get_reward_zone_coords(rz_labels[i])
        dist = compute_distance_to_reward_zone(pos, rz_start, rz_end)
        dist_disc = discretize_distance_to_reward(dist)

        # 2. Absolute position (discretized into 5 bins, time-varying)
        pos_clipped = np.clip(pos, 0, TRACK_LENGTH)
        pos_disc = discretize_position(pos_clipped)

        # 3. Speed (discretized, time-varying)
        speed_disc = discretize_speed(spd)

        # 4. Lick (binary, time-varying)

        # 5. Reward zone location (per trial): 0=A, 1=B, 2=C
        rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_labels[i]]

        # 6. Reward outcome (per trial): 0=no, 1=yes

        output_arr = np.zeros((6, n_timepoints), dtype=np.int64)
        output_arr[0, :] = dist_disc
        output_arr[1, :] = pos_disc
        output_arr[2, :] = speed_disc
        output_arr[3, :] = lck_binary
        output_arr[4, :] = rz_loc
        output_arr[5, :] = reward_per_trial[i]

        trial_neural.append(trial_n)
        trial_input.append(input_arr)
        trial_output.append(output_arr)
        valid_trial_indices.append(i)

    if len(trial_neural) < 2:
        print(f"  Skipping session: only {len(trial_neural)} valid trials")
        return None

    return {
        'neural': trial_neural,
        'input': trial_input,
        'output': trial_output,
        'n_neurons': n_neurons,
        'n_total_rois': n_total_rois,
        'n_cells_after_iscell': n_cells_after_iscell,
        'n_interneurons': n_interneurons,
        'n_trials': len(trial_neural),
        'n_lick_error_trials': len(error_trials),
        'rz_labels': rz_labels[valid_trial_indices],
        'env_per_trial': env_per_trial[valid_trial_indices],
        'reward_per_trial': reward_per_trial[valid_trial_indices],
        'scene': scene,
        'exp_day': exp_day,
    }


def convert_all_data(data_dir, sample_only=False):
    """Convert all NWB files to the decoder format."""

    subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])

    all_neural = []
    all_input = []
    all_output = []
    all_subject_idx = []
    all_brain_region_idx = []

    subject_names = []
    session_info_list = []

    total_sessions = 0
    total_trials = 0
    total_neurons_list = []

    for subj_dir_name in subjects:
        subj_path = os.path.join(data_dir, subj_dir_name)
        subj_id = subj_dir_name  # e.g., 'sub-m11'

        if subj_id not in SUBJECT_MAP:
            print(f"Warning: Unknown subject {subj_id}, skipping")
            continue

        gcamp_name = SUBJECT_MAP[subj_id]
        mouse_name = subj_id.replace('sub-', '')  # e.g., 'm11'

        if mouse_name not in subject_names:
            subject_names.append(mouse_name)
        subj_idx = subject_names.index(mouse_name)

        session_files = sorted([f for f in os.listdir(subj_path) if f.endswith('.nwb')])

        if sample_only:
            session_files = session_files[:2]

        for sess_file in session_files:
            filepath = os.path.join(subj_path, sess_file)

            # Extract exp_day from filename
            # e.g., 'sub-m11_ses-03_behavior+ophys.nwb' -> ses-03 -> 3
            ses_part = sess_file.split('_')[1]  # 'ses-03'
            exp_day = int(ses_part.split('-')[1])

            # Get scene from sessions info
            if gcamp_name not in SESSIONS_INFO:
                print(f"Warning: No session info for {gcamp_name}, skipping")
                continue
            if exp_day not in SESSIONS_INFO[gcamp_name]:
                print(f"Warning: No session info for {gcamp_name} day {exp_day}, skipping")
                continue

            scene = SESSIONS_INFO[gcamp_name][exp_day]

            print(f"Processing {subj_id} ses-{exp_day:02d} ({scene})...")

            try:
                nwb_data = load_nwb_session(filepath)
                result = process_session(nwb_data, scene, exp_day)
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
                continue

            if result is None:
                continue

            all_neural.append(result['neural'])
            all_input.append(result['input'])
            all_output.append(result['output'])
            all_subject_idx.append(subj_idx)
            all_brain_region_idx.append(np.zeros(result['n_neurons'], dtype=int))

            session_info_list.append({
                'subject': mouse_name,
                'gcamp_name': gcamp_name,
                'exp_day': exp_day,
                'scene': scene,
                'n_neurons': result['n_neurons'],
                'n_total_rois': result['n_total_rois'],
                'n_cells_after_iscell': result['n_cells_after_iscell'],
                'n_interneurons': result['n_interneurons'],
                'n_trials': result['n_trials'],
                'n_lick_error_trials': result['n_lick_error_trials'],
            })

            total_sessions += 1
            total_trials += result['n_trials']
            total_neurons_list.append(result['n_neurons'])

            print(f"  {result['n_neurons']} neurons, {result['n_trials']} trials "
                  f"({result['n_interneurons']} interneurons excluded, "
                  f"{result['n_lick_error_trials']} lick error trials)")

    # Build the output dictionary
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,

        'subjects': subject_names,
        'subject_idx': np.array(all_subject_idx, dtype=int),

        'brain_regions': ['CA1'],
        'brain_region_idx': all_brain_region_idx,

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
            ['< -50 cm', '-50 to -10 cm', '-10 to 0 cm', '0 cm (in zone)',
             '0 to +10 cm', '+10 to +50 cm', '> +50 cm'],
            ['0-90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '360-450 cm'],
            ['< 2 cm/s', '2-10 cm/s', '10-20 cm/s', '20-40 cm/s', '> 40 cm/s'],
            ['no lick', 'lick'],
            ['A', 'B', 'C'],
            ['no reward', 'reward'],
        ],

        'metadata': {
            'task_description': (
                'Virtual reality navigation task with hidden reward zones. '
                'Mice navigate a 450 cm linear track with a hidden 50 cm reward zone '
                'at one of three possible locations (A: 80-130 cm, B: 200-250 cm, C: 320-370 cm). '
                'Reward zone switches occur after 30 trials. Two environments are used.'
            ),
            'time_bin_size': 1000.0 / 15.5078125,  # ~64.5 ms
            'temporal_alignment_event': 'start of trial (trial_start event)',
            'off_start': 0.0,
            'off_end': None,
            'imaging_rate_hz': 15.5078125,
            'track_length_cm': TRACK_LENGTH,
            'reward_zones': REWARD_ZONES,
            'brain_region': 'hippocampal CA1',
            'species': 'mouse',
            'n_subjects': len(subject_names),
            'n_sessions': total_sessions,
            'n_trials_total': total_trials,
            'neurons_per_session': total_neurons_list,
            'session_info': session_info_list,
        },
    }

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Conversion Summary")
    print(f"{'='*60}")
    print(f"Subjects: {len(subject_names)} ({subject_names})")
    print(f"Sessions: {total_sessions}")
    print(f"Total trials: {total_trials}")
    print(f"Neurons per session: min={min(total_neurons_list)}, "
          f"max={max(total_neurons_list)}, "
          f"mean={np.mean(total_neurons_list):.0f}")
    print(f"Time bin size: {data['metadata']['time_bin_size']:.2f} ms")

    return data


def run_sanity_checks(data):
    """Run sanity checks on the converted data."""
    print(f"\n{'='*60}")
    print(f"Sanity Checks")
    print(f"{'='*60}")

    n_sessions = len(data['neural'])

    # Check 1: Consistent dimensions
    print(f"\n1. Dimension consistency:")
    all_ok = True
    for s in range(n_sessions):
        n_trials = len(data['neural'][s])
        n_trials_in = len(data['input'][s])
        n_trials_out = len(data['output'][s])
        if n_trials != n_trials_in or n_trials != n_trials_out:
            print(f"  Session {s}: MISMATCH neural={n_trials}, input={n_trials_in}, output={n_trials_out}")
            all_ok = False

        for t in range(n_trials):
            n_tp_neural = data['neural'][s][t].shape[1]
            n_tp_input = data['input'][s][t].shape[1]
            n_tp_output = data['output'][s][t].shape[1]
            if n_tp_neural != n_tp_input or n_tp_neural != n_tp_output:
                print(f"  Session {s}, Trial {t}: timepoint mismatch "
                      f"neural={n_tp_neural}, input={n_tp_input}, output={n_tp_output}")
                all_ok = False
    if all_ok:
        print("  All dimensions consistent.")

    # Check 2: Trial counts match paper (~80.5 +/- 7.4 per session)
    trial_counts = [len(data['neural'][s]) for s in range(n_sessions)]
    mean_trials = np.mean(trial_counts)
    std_trials = np.std(trial_counts)
    print(f"\n2. Trial counts: mean={mean_trials:.1f} +/- {std_trials:.1f} "
          f"(paper: 80.5 +/- 7.4)")

    # Check 3: Neuron counts (155-2172 per session per paper)
    neuron_counts = [data['neural'][s][0].shape[0] for s in range(n_sessions)]
    print(f"\n3. Neuron counts: min={min(neuron_counts)}, max={max(neuron_counts)} "
          f"(paper: 155-2172)")

    # Check 4: Reward omission rate (~15%)
    total_rewarded = 0
    total_trial_count = 0
    for s in range(n_sessions):
        for t in range(len(data['output'][s])):
            total_trial_count += 1
            if data['output'][s][t][5, 0] == 1:
                total_rewarded += 1
    omission_rate = 1.0 - total_rewarded / total_trial_count
    print(f"\n4. Reward omission rate: {omission_rate*100:.1f}% (paper: ~15%)")

    # Check 5: Number of subjects
    print(f"\n5. Subjects: {len(data['subjects'])} "
          f"(paper: 11 switch + 3 fixed = 14 total, we have 11 switch mice)")

    # Check 6: Track length check via position output
    print(f"\n6. Output value ranges:")
    for s in range(min(3, n_sessions)):
        for t in range(min(3, len(data['output'][s]))):
            out = data['output'][s][t]
            print(f"  S{s}T{t}: dist_rz=[{out[0].min()}-{out[0].max()}], "
                  f"pos=[{out[1].min()}-{out[1].max()}], "
                  f"speed=[{out[2].min()}-{out[2].max()}], "
                  f"lick=[{out[3].min()}-{out[3].max()}]")

    # Check 7: Reward zone distribution
    rz_counts = {0: 0, 1: 0, 2: 0}
    for s in range(n_sessions):
        for t in range(len(data['output'][s])):
            rz = data['output'][s][t][4, 0]  # per-trial value
            rz_counts[rz] += 1
    print(f"\n7. Reward zone distribution: A={rz_counts[0]}, B={rz_counts[1]}, C={rz_counts[2]}")

    print(f"\nSanity checks complete.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data', help='Path to NWB data directory')
    parser.add_argument('--output', default='converted_data.pkl', help='Output pickle file')
    parser.add_argument('--sample', action='store_true', help='Only process 2 sessions per subject')
    parser.add_argument('--sample-output', default='sample_data.pkl', help='Sample output file')
    args = parser.parse_args()

    if args.sample:
        print("Running in SAMPLE mode (2 sessions per subject)")
        data = convert_all_data(args.data_dir, sample_only=True)
        output_file = args.sample_output
    else:
        data = convert_all_data(args.data_dir, sample_only=False)
        output_file = args.output

    run_sanity_checks(data)

    print(f"\nSaving to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved successfully. File size: {os.path.getsize(output_file) / 1e6:.1f} MB")
