#!/usr/bin/env python3
"""
Convert NWB data from Sosa et al. 2025 to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--sample] [--full] [--show-processing]
"""

import argparse
import glob
import os
import pickle
import time
import warnings

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Constants
# ============================================================
TRACK_LENGTH = 450.0  # cm
REWARD_ZONE_DICT = {
    'A': (80, 130),
    'B': (200, 250),
    'C': (320, 370),
}
REWARD_ZONE_LABEL_MAP = {'A': 0, 'B': 1, 'C': 2}
SPEED_THRESHOLD = 2.0  # cm/s  (exclude below this)
LICK_ERROR_THRESHOLD = 0.35  # fraction of frames with cumulative lick >2
SWITCH_TRIAL = 30  # reward zone changes after this many trials on switch days
DATA_DIR = '/app/data'


def parse_args():
    parser = argparse.ArgumentParser(description='Convert NWB data to decoder format.')
    parser.add_argument('outfile', type=str, help='Output pickle file path')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    parser.add_argument('--full', action='store_true', help='Process all sessions (default)')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing steps')
    return parser.parse_args()


def get_all_nwb_files():
    """Get all NWB file paths, sorted by subject and session."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
    return files


def parse_scene(identifier):
    """Extract scene name from NWB identifier field."""
    if '/' in identifier:
        return identifier.split('/')[-1]
    return identifier


def get_reward_zones_from_scene(scene, n_trials, change_trial=SWITCH_TRIAL):
    """
    Determine reward zone labels and coordinates for each trial from scene name.

    Returns:
        rz_coords: (n_trials, 2) array of [start, end] positions
        rz_labels: (n_trials,) array of zone labels ('A', 'B', 'C')
    """
    rz_coords = np.zeros((n_trials, 2))
    rz_labels = np.empty(n_trials, dtype='U1')

    # Parse scene name to determine reward zone(s)
    # Single-location scenes
    for loc in ['A', 'B', 'C']:
        if scene.endswith(f'Location{loc}') and '_to_' not in scene:
            rz_coords[:] = REWARD_ZONE_DICT[loc]
            rz_labels[:] = loc
            return rz_coords, rz_labels

    # Switch scenes within same environment: X_to_Y
    if '_to_' in scene and 'Env' not in scene.split('_to_')[1].split('_')[0] if '_' in scene.split('_to_')[1] else True:
        # e.g. "Env1_LocationA_to_B" or "Env1_LocationA_to_C"
        parts = scene.split('_to_')
        # Get first location
        loc1 = parts[0][-1]  # Last char before _to_
        loc2 = parts[1][-1]  # Last char after _to_

        rz_coords[:change_trial] = REWARD_ZONE_DICT[loc1]
        rz_labels[:change_trial] = loc1
        rz_coords[change_trial:] = REWARD_ZONE_DICT[loc2]
        rz_labels[change_trial:] = loc2
        return rz_coords, rz_labels

    # Cross-environment switch: e.g. "Env1_B_to_Env2_C" or "Env1_C_to_Env2_B"
    if '_to_Env' in scene:
        parts = scene.split('_to_')
        # First part: e.g. "Env1_B" -> location is last char
        loc1 = parts[0][-1]
        # Second part: e.g. "Env2_C" -> location is last char
        loc2 = parts[1][-1]

        rz_coords[:change_trial] = REWARD_ZONE_DICT[loc1]
        rz_labels[:change_trial] = loc1
        rz_coords[change_trial:] = REWARD_ZONE_DICT[loc2]
        rz_labels[change_trial:] = loc2
        return rz_coords, rz_labels

    raise ValueError(f"Cannot parse scene: {scene}")


def get_environment_from_scene(scene, n_trials, change_trial=SWITCH_TRIAL):
    """
    Determine environment type (0=ENV1, 1=ENV2) for each trial.
    """
    env = np.zeros(n_trials, dtype=np.int64)

    if '_to_Env' in scene:
        # Cross-environment switch
        parts = scene.split('_to_')
        env1_num = int(parts[0][3])  # "Env1..." -> 1
        env2_num = int(parts[1][3])  # "Env2..." -> 2
        env[:change_trial] = env1_num - 1  # 0-indexed
        env[change_trial:] = env2_num - 1
    elif scene.startswith('Env1'):
        env[:] = 0
    elif scene.startswith('Env2'):
        env[:] = 1
    else:
        raise ValueError(f"Cannot determine environment from scene: {scene}")

    return env


def compute_distance_to_reward_zone(position, rz_start, rz_end):
    """
    Compute signed distance from position to the nearest point in the reward zone.

    Negative = before zone (position < rz_start)
    Zero = inside zone
    Positive = after zone (position > rz_end)
    """
    distance = np.zeros_like(position)
    before = position < rz_start
    after = position > rz_end
    inside = ~before & ~after

    distance[before] = position[before] - rz_start
    distance[after] = position[after] - rz_end
    distance[inside] = 0.0

    return distance


def discretize_distance(distance):
    """Discretize distance to reward zone into 7 bins."""
    out = np.zeros(len(distance), dtype=np.int64)
    out[distance < -50] = 0
    out[(distance >= -50) & (distance < -10)] = 1
    out[(distance >= -10) & (distance < 0)] = 2
    out[distance == 0] = 3  # in zone
    out[(distance > 0) & (distance <= 10)] = 4
    out[(distance > 10) & (distance <= 50)] = 5
    out[distance > 50] = 6
    return out


def discretize_position(position):
    """Discretize absolute position into 5 equal bins spanning 450 cm."""
    out = np.zeros(len(position), dtype=np.int64)
    out[position < 90] = 0
    out[(position >= 90) & (position < 180)] = 1
    out[(position >= 180) & (position < 270)] = 2
    out[(position >= 270) & (position < 360)] = 3
    out[position >= 360] = 4
    return out


def discretize_speed(speed):
    """Discretize speed into 5 bins."""
    out = np.zeros(len(speed), dtype=np.int64)
    out[speed < 2] = 0
    out[(speed >= 2) & (speed < 10)] = 1
    out[(speed >= 10) & (speed < 20)] = 2
    out[(speed >= 20) & (speed < 40)] = 3
    out[speed >= 40] = 4
    return out


def process_session(nwb_path, show_processing=False, session_label=''):
    """
    Process a single NWB file and return trial-segmented data.

    Returns dict with keys: neural, input_data, output_data, n_neurons,
        subject, session_id, scene, or None if session is invalid.
    """
    t0 = time.time()

    with h5py.File(nwb_path, 'r') as f:
        # --- Metadata ---
        identifier = f['identifier'][()].decode() if isinstance(f['identifier'][()], bytes) else str(f['identifier'][()])
        subject_id = f['general/subject/subject_id'][()].decode() if isinstance(f['general/subject/subject_id'][()], bytes) else str(f['general/subject/subject_id'][()])
        session_id = f['general/session_id'][()].decode() if isinstance(f['general/session_id'][()], bytes) else str(f['general/session_id'][()])
        imaging_rate = float(f['acquisition/TwoPhotonSeries/imaging_plane/imaging_rate'][()])
        scene = parse_scene(identifier)

        # --- Neural data ---
        iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:, 0]
        cell_mask = iscell == 1
        n_neurons = int(cell_mask.sum())

        if n_neurons == 0:
            print(f"  WARNING: No cells in {session_label}, skipping")
            return None

        # Load deconvolved events from all planes and concatenate
        deconv_group = f['processing/ophys/Deconvolved']
        plane_keys = sorted([k for k in deconv_group.keys() if k.startswith('plane')])
        plane_data = [deconv_group[pk]['data'][:] for pk in plane_keys]
        deconv_all = np.concatenate(plane_data, axis=1)  # (T, total_ROIs)

        # Apply iscell mask
        deconv = deconv_all[:, cell_mask]  # (T, N_cells)

        # --- Behavioral data ---
        bts = f['processing/behavior/BehavioralTimeSeries']
        position = bts['position/data'][:]
        speed = bts['speed/data'][:]
        lick_raw = bts['lick/data'][:]
        trial_start_data = bts['trial_start/data'][:]
        teleport_data = bts['teleport/data'][:]
        trial_number_data = bts['trial number/data'][:]
        environment_data = bts['environment/data'][:]
        timestamps = bts['position/timestamps'][:]

        # Reward events
        reward_timestamps = bts['Reward/timestamps'][:]

    n_timepoints = len(position)

    # --- Trial boundaries ---
    trial_start_inds = np.where(trial_start_data > 0)[0]
    teleport_inds = np.where(teleport_data > 0)[0]
    n_trials = len(trial_start_inds)

    if n_trials < 2:
        print(f"  WARNING: Only {n_trials} trials in {session_label}, skipping")
        return None

    # Ensure matching number of starts and teleports
    if len(teleport_inds) != n_trials:
        # Trim to match
        min_len = min(len(trial_start_inds), len(teleport_inds))
        trial_start_inds = trial_start_inds[:min_len]
        teleport_inds = teleport_inds[:min_len]
        n_trials = min_len

    # --- Reward zone from scene ---
    rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
    env_per_trial = get_environment_from_scene(scene, n_trials)

    # --- Determine reward per trial ---
    reward_per_trial = np.zeros(n_trials, dtype=np.int64)
    for i in range(n_trials):
        ts = trial_start_inds[i]
        te = teleport_inds[i]
        # Find reward events within this trial's time window
        trial_time_start = timestamps[ts]
        trial_time_end = timestamps[te] if te < len(timestamps) else timestamps[-1]
        rewards_in_trial = np.sum((reward_timestamps >= trial_time_start) &
                                   (reward_timestamps <= trial_time_end))
        reward_per_trial[i] = 1 if rewards_in_trial > 0 else 0

    # --- Previous trial outcome ---
    prev_outcome = np.zeros(n_trials, dtype=np.int64)
    prev_outcome[1:] = reward_per_trial[:-1]  # First trial: 0 (no previous)

    # --- Lick sensor error correction ---
    lick_corrected = lick_raw.copy()
    for i in range(n_trials):
        ts = trial_start_inds[i]
        te = teleport_inds[i]
        trial_licks = lick_corrected[ts:te]
        if len(trial_licks) > 0:
            frac_error = np.sum(trial_licks > 2) / len(trial_licks)
            if frac_error > LICK_ERROR_THRESHOLD:
                lick_corrected[ts:te] = np.nan

    # Binarize licks
    lick_binary = np.zeros_like(lick_corrected)
    lick_binary[lick_corrected > 0] = 1
    lick_binary[np.isnan(lick_corrected)] = 0  # NaN licks -> 0

    # --- Process each trial ---
    neural_trials = []
    input_trials = []
    output_trials = []

    frame_period = 1.0 / imaging_rate  # seconds per frame

    for i in range(n_trials):
        ts = trial_start_inds[i]
        te = teleport_inds[i]

        if te <= ts:
            continue

        # Extract trial data
        trial_pos = position[ts:te]
        trial_speed = speed[ts:te]
        trial_lick = lick_binary[ts:te]
        trial_neural = deconv[ts:te, :]  # (T_trial, N)
        trial_timestamps = timestamps[ts:te]

        n_trial_tp = len(trial_pos)
        if n_trial_tp < 2:
            continue

        # Speed threshold mask: keep timepoints with speed >= 2 cm/s
        speed_mask = trial_speed >= SPEED_THRESHOLD

        # Also exclude NaN neural data
        neural_nan_mask = ~np.isnan(trial_neural[:, 0])

        # Combined valid mask
        valid_mask = speed_mask & neural_nan_mask

        if valid_mask.sum() < 2:
            continue

        # Apply mask
        trial_pos_valid = trial_pos[valid_mask]
        trial_speed_valid = trial_speed[valid_mask]
        trial_lick_valid = trial_lick[valid_mask]
        trial_neural_valid = trial_neural[valid_mask, :]  # (T_valid, N)
        trial_times_valid = trial_timestamps[valid_mask]

        # Replace NaN in neural data with 0 (as in reference code: X[np.isnan(X)] = 0)
        trial_neural_valid = np.nan_to_num(trial_neural_valid, nan=0.0)

        # --- Neural: (n_neurons, n_timepoints) ---
        neural_matrix = trial_neural_valid.T.astype(np.float32)  # (N, T)

        # --- Inputs ---
        # Time from start of trial in seconds
        time_from_start = trial_times_valid - trial_timestamps[0]  # seconds

        # Per-trial inputs (broadcast to match)
        env_type = env_per_trial[i]
        trial_num = float(i)  # 0-indexed trial number
        prev_out = float(prev_outcome[i])

        # input shape: (n_input, n_timepoints) for time-varying, (n_input,) for per-trial
        # Mix of time-varying and per-trial
        n_tp = len(time_from_start)
        input_data = np.zeros((4, n_tp), dtype=np.float32)
        input_data[0, :] = time_from_start
        input_data[1, :] = env_type
        input_data[2, :] = trial_num
        input_data[3, :] = prev_out

        # --- Outputs ---
        # Distance to reward zone
        rz_start = rz_coords[i, 0]
        rz_end = rz_coords[i, 1]
        dist = compute_distance_to_reward_zone(trial_pos_valid, rz_start, rz_end)
        dist_disc = discretize_distance(dist)

        # Absolute position
        pos_disc = discretize_position(trial_pos_valid)

        # Speed
        speed_disc = discretize_speed(trial_speed_valid)

        # Lick
        lick_disc = trial_lick_valid.astype(np.int64)

        # Reward zone location (per-trial)
        rz_label = REWARD_ZONE_LABEL_MAP[rz_labels[i]]

        # Reward outcome (per-trial)
        reward_out = reward_per_trial[i]

        # output shape: (n_output, n_timepoints) for time-varying, (n_output,) for per-trial
        # 4 time-varying + 2 per-trial = 6 outputs
        # Time-varying outputs: (4, n_tp)
        # Per-trial outputs: scalar
        n_output_tv = 4  # distance, position, speed, lick
        n_output_pt = 2  # reward zone, reward outcome

        output_data = np.zeros((n_output_tv + n_output_pt, n_tp), dtype=np.int64)
        output_data[0, :] = dist_disc
        output_data[1, :] = pos_disc
        output_data[2, :] = speed_disc
        output_data[3, :] = lick_disc
        output_data[4, :] = rz_label  # broadcast per-trial
        output_data[5, :] = reward_out  # broadcast per-trial

        neural_trials.append(neural_matrix)
        input_trials.append(input_data)
        output_trials.append(output_data)

    if len(neural_trials) < 2:
        print(f"  WARNING: Only {len(neural_trials)} valid trials in {session_label}, skipping")
        return None

    t1 = time.time()
    print(f"  {session_label}: {n_neurons} cells, {len(neural_trials)} trials, {t1-t0:.1f}s")

    result = {
        'neural': neural_trials,
        'input_data': input_trials,
        'output_data': output_trials,
        'n_neurons': n_neurons,
        'subject': subject_id,
        'session_id': session_id,
        'scene': scene,
        'imaging_rate': imaging_rate,
        'n_trials_original': n_trials,
        'reward_rate': float(reward_per_trial.sum()) / n_trials if n_trials > 0 else 0,
    }

    # --- Show processing plots ---
    if show_processing:
        plot_processing(result, nwb_path, session_label,
                        position, speed, lick_raw, lick_binary,
                        trial_start_inds, teleport_inds, deconv, cell_mask,
                        timestamps, rz_coords, rz_labels, reward_per_trial)

    return result


def plot_processing(result, nwb_path, session_label,
                    position, speed, lick_raw, lick_binary,
                    trial_start_inds, teleport_inds, deconv, cell_mask,
                    timestamps, rz_coords, rz_labels, reward_per_trial):
    """Plot processing steps for visual verification."""
    fig, axes = plt.subplots(6, 2, figsize=(20, 24))
    fig.suptitle(f'Processing: {session_label}', fontsize=14)

    # Plot 1: Raw position and speed over time
    ax = axes[0, 0]
    ax.plot(timestamps[:1000], position[:1000], 'b-', linewidth=0.5)
    ax.set_ylabel('Position (cm)')
    ax.set_title('Position (first 1000 frames)')
    for i in range(min(3, len(trial_start_inds))):
        ts = trial_start_inds[i]
        if ts < 1000:
            ax.axvline(timestamps[ts], color='g', linewidth=0.5, alpha=0.5)

    ax = axes[0, 1]
    ax.plot(timestamps[:1000], speed[:1000], 'r-', linewidth=0.5)
    ax.axhline(SPEED_THRESHOLD, color='k', linestyle='--', linewidth=0.5)
    ax.set_ylabel('Speed (cm/s)')
    ax.set_title('Speed (first 1000 frames)')

    # Plot 2: Trial-segmented neural activity (first trial)
    if len(result['neural']) > 0:
        ax = axes[1, 0]
        trial_data = result['neural'][0]
        n_show = min(20, trial_data.shape[0])
        ax.imshow(trial_data[:n_show], aspect='auto', cmap='hot')
        ax.set_ylabel('Neuron')
        ax.set_xlabel('Time (frames)')
        ax.set_title(f'Neural activity - Trial 0 (first {n_show} neurons)')

    # Plot 3: Outputs for first trial
    if len(result['output_data']) > 0:
        out = result['output_data'][0]
        ax = axes[1, 1]
        ax.plot(out[0], label='Dist to RZ')
        ax.plot(out[1], label='Position bin')
        ax.plot(out[2], label='Speed bin')
        ax.legend(fontsize=8)
        ax.set_title('Outputs - Trial 0')

    # Plot 4: Position discretization check
    if len(result['input_data']) > 0 and len(result['output_data']) > 0:
        ax = axes[2, 0]
        inp = result['input_data'][0]
        out = result['output_data'][0]
        ax.scatter(inp[0], out[1], s=1, alpha=0.5)
        ax.set_xlabel('Time from trial start (s)')
        ax.set_ylabel('Position bin')
        ax.set_title('Position bins vs time - Trial 0')

    # Plot 5: Distance to reward zone
    if len(result['output_data']) > 0:
        ax = axes[2, 1]
        for t_i in range(min(5, len(result['output_data']))):
            ax.plot(result['output_data'][t_i][0], alpha=0.5, linewidth=0.5)
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel('Distance bin')
        ax.set_title('Distance to RZ - first 5 trials')

    # Plot 6: Reward zone locations across trials
    ax = axes[3, 0]
    rz_mid = (rz_coords[:, 0] + rz_coords[:, 1]) / 2
    ax.plot(rz_mid, 'o-', markersize=2)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Reward zone center (cm)')
    ax.set_title('Reward zone location across trials')

    # Plot 7: Reward delivery
    ax = axes[3, 1]
    ax.plot(reward_per_trial, 'ko', markersize=2)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Reward (0/1)')
    ax.set_title(f'Reward delivery (rate={reward_per_trial.mean():.2f})')

    # Plot 8: Lick data
    ax = axes[4, 0]
    if len(result['output_data']) > 0:
        for t_i in range(min(5, len(result['output_data']))):
            ax.plot(result['output_data'][t_i][3], alpha=0.5, linewidth=0.5)
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Lick (0/1)')
    ax.set_title('Lick - first 5 trials')

    # Plot 9: Speed output distribution
    ax = axes[4, 1]
    if len(result['output_data']) > 0:
        all_speed = np.concatenate([o[2] for o in result['output_data']])
        bins = np.arange(-0.5, 5.5, 1)
        ax.hist(all_speed, bins=bins, edgecolor='black')
        ax.set_xlabel('Speed bin')
        ax.set_ylabel('Count')
        ax.set_title('Speed bin distribution')

    # Plot 10: Input distributions
    ax = axes[5, 0]
    if len(result['input_data']) > 0:
        all_time = np.concatenate([inp[0] for inp in result['input_data']])
        ax.hist(all_time, bins=50, edgecolor='black')
        ax.set_xlabel('Time from trial start (s)')
        ax.set_ylabel('Count')
        ax.set_title('Time from trial start distribution')

    ax = axes[5, 1]
    if len(result['output_data']) > 0:
        all_dist = np.concatenate([o[0] for o in result['output_data']])
        bins = np.arange(-0.5, 7.5, 1)
        ax.hist(all_dist, bins=bins, edgecolor='black')
        ax.set_xlabel('Distance bin')
        ax.set_ylabel('Count')
        ax.set_title('Distance to RZ bin distribution')

    plt.tight_layout()
    safe_label = session_label.replace(' ', '_').replace('/', '_')
    fig.savefig(f'/app/processing_{safe_label}.png', dpi=100)
    plt.close(fig)
    print(f"  Saved processing plot: /app/processing_{safe_label}.png")


def main():
    args = parse_args()

    t_start = time.time()

    # Get all NWB files
    all_files = get_all_nwb_files()
    print(f"Found {len(all_files)} NWB files")

    if args.sample:
        # Take 2 sessions from different subjects
        all_files = [all_files[0], all_files[len(all_files)//2]]
        print(f"Sample mode: processing {len(all_files)} sessions")

    # Process all sessions
    neural_all = []
    input_all = []
    output_all = []
    subjects_list = []
    subject_idx_list = []
    brain_region_idx_all = []
    session_info = []

    unique_subjects = []

    for fpath in all_files:
        fname = os.path.basename(fpath)
        session_label = fname.replace('_behavior+ophys.nwb', '')

        result = process_session(fpath,
                                 show_processing=args.show_processing,
                                 session_label=session_label)

        if result is None:
            continue

        # Track subjects
        subj = result['subject']
        if subj not in unique_subjects:
            unique_subjects.append(subj)
        subj_idx = unique_subjects.index(subj)

        neural_all.append(result['neural'])
        input_all.append(result['input_data'])
        output_all.append(result['output_data'])
        subject_idx_list.append(subj_idx)
        brain_region_idx_all.append(np.zeros(result['n_neurons'], dtype=np.int64))  # All CA1

        session_info.append({
            'subject': subj,
            'session_id': result['session_id'],
            'scene': result['scene'],
            'n_neurons': result['n_neurons'],
            'n_trials': len(result['neural']),
            'n_trials_original': result['n_trials_original'],
            'reward_rate': result['reward_rate'],
            'imaging_rate': result['imaging_rate'],
        })

    n_sessions = len(neural_all)
    print(f"\nProcessed {n_sessions} sessions from {len(unique_subjects)} subjects")

    # --- Build output dictionary ---
    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,

        'subjects': unique_subjects,
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),

        'brain_regions': ['CA1'],
        'brain_region_idx': brain_region_idx_all,

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
            ['<-50cm', '-50 to -10cm', '-10 to 0cm', 'in zone', '0 to +10cm', '+10 to +50cm', '>+50cm'],
            ['0-90cm', '90-180cm', '180-270cm', '270-360cm', '360-450cm'],
            ['<2 cm/s', '2-10 cm/s', '10-20 cm/s', '20-40 cm/s', '>40 cm/s'],
            ['no lick', 'lick'],
            ['zone A', 'zone B', 'zone C'],
            ['no reward', 'reward'],
        ],

        'metadata': {
            'task_description': 'Virtual reality navigation with hidden reward zone switches. Mice navigate a 450cm linear track with a hidden 50cm reward zone at one of three locations (A, B, C). Reward zone switches every few days. Two environments used.',
            'time_bin_size': 64.5,  # ms (approximate, ~1/15.5 Hz)
            'temporal_alignment_event': 'start of trial (entry to virtual track)',
            'off_start': 0.0,  # trial starts at alignment event
            'off_end': None,  # variable trial length
            'session_info': session_info,
            'speed_threshold_cm_s': SPEED_THRESHOLD,
            'track_length_cm': TRACK_LENGTH,
            'reward_zones': REWARD_ZONE_DICT,
            'imaging_rate_hz': 15.5,
            'n_subjects': len(unique_subjects),
            'n_sessions': n_sessions,
        }
    }

    # --- Print summary statistics ---
    total_trials = sum(len(s) for s in neural_all)
    total_neurons = sum(si['n_neurons'] for si in session_info)
    neurons_per_session = [si['n_neurons'] for si in session_info]
    trials_per_session = [si['n_trials'] for si in session_info]
    reward_rates = [si['reward_rate'] for si in session_info]

    print(f"\n=== Summary Statistics ===")
    print(f"Subjects: {len(unique_subjects)} ({', '.join(unique_subjects)})")
    print(f"Sessions: {n_sessions}")
    print(f"Total trials (valid): {total_trials}")
    print(f"Trials/session: {np.mean(trials_per_session):.1f} +/- {np.std(trials_per_session):.1f} (range [{min(trials_per_session)}, {max(trials_per_session)}])")
    print(f"Neurons/session: {np.mean(neurons_per_session):.1f} +/- {np.std(neurons_per_session):.1f} (range [{min(neurons_per_session)}, {max(neurons_per_session)}])")
    print(f"Mean reward rate: {np.mean(reward_rates):.3f}")

    # --- Save ---
    t_save = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size = os.path.getsize(args.outfile) / (1024**2)
    t_end = time.time()
    print(f"\nSaved to {args.outfile} ({file_size:.1f} MB)")
    print(f"Total time: {t_end - t_start:.1f}s (save: {t_end - t_save:.1f}s)")


if __name__ == '__main__':
    main()
