#!/usr/bin/env python3
"""
Convert NWB data from Sosa et al. 2025 to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--sample] [--full] [--show-processing]
"""
import argparse
import os
import sys
import time
import pickle
import numpy as np
import h5py
from glob import glob

# Reward zone coordinates from behavior.py in reference code
REWARD_ZONE_DICT = {
    'A': [80, 130],   # mapped from 'X' in code
    'B': [200, 250],  # mapped from 'Y' in code
    'C': [320, 370],  # mapped from 'Z' in code
}

TRACK_LENGTH = 450  # cm
IMAGING_RATE_NOMINAL = 15.5078125  # Hz

def parse_args():
    parser = argparse.ArgumentParser(description='Convert NWB data to decoder format')
    parser.add_argument('output', type=str, help='Output pickle file path')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    parser.add_argument('--full', action='store_true', help='Process all sessions (default)')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    return parser.parse_args()


def find_nwb_files(data_dir='data'):
    """Find all NWB files organized by subject."""
    subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
    sessions = []
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        files = sorted(glob(os.path.join(subj_dir, '*.nwb')))
        for fpath in files:
            sessions.append({
                'subject': subj.replace('sub-', ''),
                'filepath': fpath,
                'filename': os.path.basename(fpath),
            })
    return sessions


def parse_scene(identifier):
    """Parse scene name from NWB identifier to determine reward zones.

    Returns:
        scene: str, e.g. 'Env1_LocationB_to_A'
    """
    return identifier.split('/')[-1]


def get_reward_zone_labels(scene, n_trials, change_trial=30):
    """Determine reward zone label (A/B/C) and environment for each trial based on scene name.

    Scene formats:
    - Single zone: Env1_LocationA, Env2_LocationB, etc.
    - Within-env switch: Env1_LocationB_to_A, Env2_LocationA_to_C, etc.
    - Cross-env switch: Env1_A_to_Env2_B, Env2_B_to_Env1_A, etc.

    Returns:
        labels: (n_trials,) array of zone labels ('A', 'B', 'C')
        coords: (n_trials, 2) array of [zone_start, zone_end] coordinates
        env_per_trial: (n_trials,) array of environment values (0=Env1, 1=Env2)
    """
    labels = np.empty(n_trials, dtype='U1')
    coords = np.zeros((n_trials, 2))
    env_per_trial = np.zeros(n_trials, dtype=np.int64)

    if '_to_' in scene:
        # Switch session - need to parse from and to
        parts = scene.split('_')
        to_idx = parts.index('to')

        # Parse "from" side
        from_parts = parts[:to_idx]
        # Parse "to" side
        to_parts = parts[to_idx+1:]

        # Extract zone letter and env from each side
        from_zone, from_env = _parse_zone_and_env(from_parts)
        to_zone, to_env = _parse_zone_and_env(to_parts)

        labels[:change_trial] = from_zone
        labels[change_trial:] = to_zone
        coords[:change_trial] = REWARD_ZONE_DICT[from_zone]
        coords[change_trial:] = REWARD_ZONE_DICT[to_zone]
        env_per_trial[:change_trial] = from_env
        env_per_trial[change_trial:] = to_env
    else:
        # Single zone session: Env1_LocationA or similar
        parts = scene.split('_')
        zone, env = _parse_zone_and_env(parts)
        labels[:] = zone
        coords[:] = REWARD_ZONE_DICT[zone]
        env_per_trial[:] = env

    return labels, coords, env_per_trial


def _parse_zone_and_env(parts):
    """Parse zone letter and env number from scene name parts.

    Examples:
        ['Env1', 'LocationA'] -> ('A', 0)
        ['Env2', 'B'] -> ('B', 1)
        ['Env1', 'LocationB'] -> ('B', 0)
    """
    env = 0
    zone = 'A'

    for p in parts:
        if p.startswith('Env'):
            env_num = int(p.replace('Env', ''))
            env = env_num - 1  # 0-indexed
        elif p.startswith('Location'):
            zone = p.replace('Location', '')
        elif len(p) == 1 and p in 'ABC':
            zone = p

    return zone, env


def compute_distance_to_reward_zone(position, rz_start, rz_end):
    """Compute signed distance from position to reward zone.

    Negative = before zone (position < zone start)
    Positive = after zone (position > zone end)
    Zero = inside zone
    """
    dist = np.zeros_like(position)
    before = position < rz_start
    after = position > rz_end
    inside = ~before & ~after

    dist[before] = position[before] - rz_start  # negative
    dist[after] = position[after] - rz_end       # positive
    dist[inside] = 0.0

    return dist


def discretize_distance(distance):
    """Discretize distance to reward zone into 7 bins.
    0: < -50 cm
    1: -50 to -10 cm
    2: -10 cm to < 0 cm
    3: 0 cm (in zone)
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
    """Discretize absolute position into 5 equal bins (0-90, 90-180, 180-270, 270-360, 360-450)."""
    bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
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


def process_session(filepath, show_processing=False, session_idx=0):
    """Process a single NWB file and return trial-segmented data.

    Returns dict with keys: neural, input, output, n_neurons, subject,
           scene, n_trials, brain_region, or None if session should be skipped.
    """
    t0 = time.time()

    with h5py.File(filepath, 'r') as f:
        # === Metadata ===
        subject_id = f['general/subject/subject_id'][()].decode()
        identifier = f['identifier'][()].decode()
        scene = parse_scene(identifier)
        imaging_rate = f['general/optophysiology/ImagingPlane/imaging_rate'][()]
        location = f['general/optophysiology/ImagingPlane/location'][()].decode()

        # === Neural data (handle multi-plane) ===
        iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:]  # (n_ROIs_total, 2)
        planeIdx = f['processing/ophys/ImageSegmentation/PlaneSegmentation/planeIdx'][:]

        # Determine available planes
        deconv_group = f['processing/ophys/Deconvolved']
        planes = sorted(deconv_group.keys())  # e.g. ['plane0'] or ['plane0', 'plane1']

        # Load and concatenate data from all planes
        deconv_list = []
        flu_list = []
        neu_list = []
        plane_cell_offset = 0
        plane_offsets = {}
        for plane in planes:
            plane_num = int(plane.replace('plane', ''))
            plane_mask = planeIdx == plane_num
            plane_offsets[plane_num] = np.where(plane_mask)[0]

            d = f[f'processing/ophys/Deconvolved/{plane}/data'][:]  # (n_timepoints, n_rois_plane)
            fl = f[f'processing/ophys/Fluorescence/{plane}/data'][:]
            ne = f[f'processing/ophys/Neuropil/{plane}/data'][:]
            deconv_list.append(d)
            flu_list.append(fl)
            neu_list.append(ne)

        # Concatenate across planes: (n_timepoints, n_rois_total_in_data)
        deconv = np.concatenate(deconv_list, axis=1)
        fluorescence = np.concatenate(flu_list, axis=1)
        neuropil_data = np.concatenate(neu_list, axis=1)

        # Build mapping from concatenated column index to iscell row index
        concat_to_iscell = []
        for plane in planes:
            plane_num = int(plane.replace('plane', ''))
            concat_to_iscell.append(plane_offsets[plane_num])
        concat_to_iscell = np.concatenate(concat_to_iscell)

        # === Behavior ===
        position = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
        speed = f['processing/behavior/BehavioralTimeSeries/speed/data'][:]
        lick = f['processing/behavior/BehavioralTimeSeries/lick/data'][:]
        environment = f['processing/behavior/BehavioralTimeSeries/environment/data'][:]
        trial_start = f['processing/behavior/BehavioralTimeSeries/trial_start/data'][:]
        teleport_sig = f['processing/behavior/BehavioralTimeSeries/teleport/data'][:]
        behav_timestamps = f['processing/behavior/BehavioralTimeSeries/position/timestamps'][:]

        # Reward events (sparse)
        reward_timestamps = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]

    n_timepoints, n_rois_concat = deconv.shape

    # === Align behavior and neural data lengths ===
    n_behav = len(position)
    if n_timepoints != n_behav:
        min_len = min(n_timepoints, n_behav)
        print(f"  NOTE: Aligned neural ({n_timepoints}) and behavior ({n_behav}) to {min_len} timepoints")
        deconv = deconv[:min_len]
        fluorescence = fluorescence[:min_len]
        neuropil_data = neuropil_data[:min_len]
        position = position[:min_len]
        speed = speed[:min_len]
        lick = lick[:min_len]
        environment = environment[:min_len]
        trial_start = trial_start[:min_len]
        teleport_sig = teleport_sig[:min_len]
        behav_timestamps = behav_timestamps[:min_len]
        n_timepoints = min_len

    # === Cell filtering ===
    # Step 1: iscell filter - map to concatenated indices
    cell_mask_concat = np.array([iscell[concat_to_iscell[i], 0] == 1 for i in range(n_rois_concat)])

    # Step 2: Interneuron exclusion - compute dF/F correlation with speed
    f_corrected = fluorescence - 0.7 * neuropil_data
    f_median = np.median(f_corrected, axis=0, keepdims=True)
    f_median[f_median == 0] = 1
    dff_simple = (f_corrected - f_median) / np.abs(f_median)

    valid_mask = (speed > 0) & (position >= 0) & ~np.isnan(speed)
    interneuron_mask = np.zeros(n_rois_concat, dtype=bool)

    if valid_mask.sum() > 100:
        speed_valid = speed[valid_mask]
        accepted_cols = np.where(cell_mask_concat)[0]
        if len(accepted_cols) > 0:
            # Vectorized correlation: z-score speed and dff, then dot product
            speed_z = (speed_valid - speed_valid.mean()) / (speed_valid.std() + 1e-10)
            dff_accepted = dff_simple[valid_mask][:, accepted_cols]
            dff_mean = dff_accepted.mean(axis=0, keepdims=True)
            dff_std = dff_accepted.std(axis=0, keepdims=True)
            dff_std[dff_std == 0] = 1e-10
            dff_z = (dff_accepted - dff_mean) / dff_std
            corrs = (speed_z @ dff_z) / len(speed_z)
            for i, col_idx in enumerate(accepted_cols):
                if corrs[i] > 0.5:
                    interneuron_mask[col_idx] = True

    # Final cell mask
    final_cell_mask = cell_mask_concat & ~interneuron_mask
    n_accepted = cell_mask_concat.sum()
    n_interneurons = (cell_mask_concat & interneuron_mask).sum()
    n_final = final_cell_mask.sum()

    # Filter neural data to accepted cells
    neural_all = deconv[:, final_cell_mask].T  # (n_neurons, n_timepoints)
    neural_all = neural_all.astype(np.float32)

    # === Trial boundaries ===
    trial_start_inds = np.where(trial_start > 0)[0]
    teleport_inds = np.where(teleport_sig > 0)[0]
    n_trials = len(trial_start_inds)

    if n_trials < 2:
        print(f"  WARNING: Only {n_trials} trials, skipping session")
        return None

    # Match teleport to trial start (ensure same count)
    if len(teleport_inds) != n_trials:
        # Try to match: each teleport should come after corresponding trial start
        matched_teleports = []
        for ts in trial_start_inds:
            tp_after = teleport_inds[teleport_inds > ts]
            if len(tp_after) > 0:
                matched_teleports.append(tp_after[0])
        teleport_inds = np.array(matched_teleports)
        n_trials = min(n_trials, len(teleport_inds))
        trial_start_inds = trial_start_inds[:n_trials]

    # === Reward zone labels ===
    rz_labels, rz_coords, env_per_trial = get_reward_zone_labels(scene, n_trials)

    # === Map reward events to trial indices ===
    # Create frame-aligned reward signal
    reward_frames = np.zeros(n_timepoints, dtype=np.float32)
    for rt in reward_timestamps:
        frame_idx = np.argmin(np.abs(behav_timestamps - rt))
        reward_frames[frame_idx] = 1.0

    # Determine reward outcome per trial
    trial_rewarded = np.zeros(n_trials, dtype=np.int64)
    for t in range(n_trials):
        si = trial_start_inds[t]
        ei = teleport_inds[t]
        if np.any(reward_frames[si:ei] > 0):
            trial_rewarded[t] = 1

    # === Environment type per trial ===
    # Use env_per_trial from scene parsing (more reliable for cross-env switches)
    # Fall back to NWB environment field where scene doesn't specify
    trial_env = env_per_trial.copy()

    # === Process licks ===
    # Following reference code: lick is cumulative per frame, set >1 to 1
    lick_binary = lick.copy()
    lick_binary[lick_binary > 0] = 1
    # Lick sensor error correction per trial (>30% frames with count>2 -> NaN)
    for t in range(n_trials):
        si = trial_start_inds[t]
        ei = teleport_inds[t]
        trial_lick = lick[si:ei]
        if len(trial_lick) > 0 and (np.sum(trial_lick > 2) / len(trial_lick)) > 0.30:
            lick_binary[si:ei] = 0  # Set to 0 instead of NaN for cleaner output

    # === Previous trial outcome (lagged) ===
    prev_outcome = np.zeros(n_trials, dtype=np.int64)
    for t in range(1, n_trials):
        prev_outcome[t] = trial_rewarded[t - 1]
    # Trial 0: no previous, use 0 (omission)

    # === Segment into trials ===
    neural_trials = []
    input_trials = []
    output_trials = []

    frame_time = 1.0 / imaging_rate  # seconds per frame

    for t in range(n_trials):
        si = trial_start_inds[t]
        ei = teleport_inds[t]
        n_tp = ei - si

        if n_tp < 2:
            continue

        # Neural: (n_neurons, n_timepoints)
        trial_neural = neural_all[:, si:ei].copy()
        # Replace NaN with 0 in neural data
        trial_neural = np.nan_to_num(trial_neural, nan=0.0)

        # === Inputs ===
        # Time from start of trial (seconds)
        time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time

        # Per-trial inputs
        env_type = np.float32(trial_env[t])
        trial_num = np.float32(t)
        prev_out = np.float32(prev_outcome[t])

        # input shape: (n_input, n_timepoints) for time-varying, or (n_input,) for per-trial
        # We have 1 time-varying + 3 per-trial
        # Time-varying: time_from_start (1, n_tp)
        # Per-trial: env_type, trial_num, prev_outcome (3,)
        trial_input_tv = time_from_start.reshape(1, -1)  # (1, n_tp)
        trial_input_pt = np.array([env_type, trial_num, prev_out], dtype=np.float32)  # (3,)
        # Combine: first dim is time-varying, rest are per-trial
        # Per spec: (n_input, n_timepoints) or (n_input,)
        # We need to combine: expand per-trial to time-varying by repeating
        trial_input = np.vstack([
            trial_input_tv,
            np.full((1, n_tp), env_type, dtype=np.float32),
            np.full((1, n_tp), trial_num, dtype=np.float32),
            np.full((1, n_tp), prev_out, dtype=np.float32),
        ])  # (4, n_tp)

        # === Outputs ===
        trial_pos = position[si:ei]
        trial_speed = speed[si:ei]
        trial_lick = lick_binary[si:ei]

        # Distance to reward zone
        rz_start = rz_coords[t, 0]
        rz_end = rz_coords[t, 1]
        signed_dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
        dist_bins = discretize_distance(signed_dist)

        # Absolute position bins
        pos_bins = discretize_position(trial_pos)

        # Speed bins
        speed_bins = discretize_speed(np.abs(trial_speed))  # use absolute speed

        # Lick binary
        lick_out = (trial_lick > 0).astype(np.int64)

        # Reward zone location (per-trial): A=0, B=1, C=2
        zone_map = {'A': 0, 'B': 1, 'C': 2}
        rz_loc = zone_map.get(rz_labels[t], 0)

        # Reward outcome (per-trial): 0=no, 1=yes
        rew_outcome = int(trial_rewarded[t])

        # Combine outputs: time-varying + per-trial
        # Time-varying: dist_bins, pos_bins, speed_bins, lick (4, n_tp)
        # Per-trial: rz_loc, rew_outcome (2,)
        trial_output = np.vstack([
            dist_bins.reshape(1, -1),
            pos_bins.reshape(1, -1),
            speed_bins.reshape(1, -1),
            lick_out.reshape(1, -1),
            np.full((1, n_tp), rz_loc, dtype=np.int64),
            np.full((1, n_tp), rew_outcome, dtype=np.int64),
        ])  # (6, n_tp)

        neural_trials.append(trial_neural.astype(np.float32))
        input_trials.append(trial_input.astype(np.float32))
        output_trials.append(trial_output.astype(np.int64))

    t1 = time.time()
    print(f"  {os.path.basename(filepath)}: {n_final} neurons ({n_accepted} accepted, {n_interneurons} interneurons removed), "
          f"{len(neural_trials)} trials, scene={scene}, env={np.unique(trial_env)}, "
          f"reward_rate={trial_rewarded.mean():.2f}, time={t1-t0:.1f}s")

    if show_processing and session_idx < 2:
        plot_processing(filepath, neural_trials, input_trials, output_trials,
                       position, speed, lick_binary, trial_start_inds, teleport_inds,
                       rz_coords, rz_labels, trial_rewarded, behav_timestamps,
                       subject_id, scene, session_idx)

    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'n_neurons': n_final,
        'subject': subject_id,
        'scene': scene,
        'n_trials': len(neural_trials),
        'brain_region': location,
    }


def plot_processing(filepath, neural_trials, input_trials, output_trials,
                   position, speed, lick_binary, trial_start_inds, teleport_inds,
                   rz_coords, rz_labels, trial_rewarded, timestamps,
                   subject_id, scene, session_idx):
    """Plot processing visualizations for a session."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 2, figsize=(20, 24))
    fig.suptitle(f'{subject_id} - {scene}', fontsize=14)

    # Pick 2 example trials
    for col, trial_idx in enumerate([5, min(35, len(neural_trials)-1)]):
        if trial_idx >= len(neural_trials):
            continue

        neural = neural_trials[trial_idx]
        inp = input_trials[trial_idx]
        out = output_trials[trial_idx]
        n_tp = neural.shape[1]
        time_vec = inp[0, :]  # time from trial start

        # Panel 1: Neural activity (mean across neurons)
        ax = axes[0, col]
        ax.plot(time_vec, np.mean(neural, axis=0), 'k', linewidth=0.5)
        ax.set_title(f'Trial {trial_idx}: Mean neural activity')
        ax.set_xlabel('Time (s)')

        # Panel 2: Position and reward zone
        ax = axes[1, col]
        si = trial_start_inds[trial_idx]
        ei = teleport_inds[trial_idx]
        trial_pos = position[si:ei]
        ax.plot(time_vec[:len(trial_pos)], trial_pos, 'b')
        ax.axhspan(rz_coords[trial_idx, 0], rz_coords[trial_idx, 1], alpha=0.3, color='green')
        ax.set_title(f'Position (zone {rz_labels[trial_idx]})')
        ax.set_ylabel('Position (cm)')

        # Panel 3: Distance to reward zone (discretized)
        ax = axes[2, col]
        ax.plot(time_vec, out[0, :], 'r')
        ax.set_title('Distance to reward zone (bins)')
        ax.set_ylabel('Bin')

        # Panel 4: Speed (discretized)
        ax = axes[3, col]
        trial_speed = speed[si:ei]
        ax.plot(time_vec[:len(trial_speed)], trial_speed, 'b', alpha=0.5, label='raw')
        ax2 = ax.twinx()
        ax2.plot(time_vec, out[2, :], 'r', alpha=0.7, label='binned')
        ax.set_title('Speed')
        ax.set_ylabel('cm/s')

        # Panel 5: Lick
        ax = axes[4, col]
        ax.plot(time_vec, out[3, :], 'k')
        ax.set_title('Lick (binary)')

        # Panel 6: Inputs
        ax = axes[5, col]
        ax.text(0.1, 0.8, f'Env: {inp[1, 0]:.0f}', transform=ax.transAxes)
        ax.text(0.1, 0.6, f'Trial#: {inp[2, 0]:.0f}', transform=ax.transAxes)
        ax.text(0.1, 0.4, f'Prev outcome: {inp[3, 0]:.0f}', transform=ax.transAxes)
        ax.text(0.1, 0.2, f'Reward zone: {rz_labels[trial_idx]} (bin {out[4, 0]})', transform=ax.transAxes)
        ax.text(0.1, 0.0, f'Reward: {out[5, 0]} (actual: {trial_rewarded[trial_idx]})', transform=ax.transAxes)
        ax.set_title('Trial info')

    plt.tight_layout()
    fig.savefig(f'processing_{session_idx}.png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved processing_{session_idx}.png")


def main():
    args = parse_args()

    t_start = time.time()

    # Find all NWB files
    sessions_info = find_nwb_files('data')
    print(f"Found {len(sessions_info)} NWB files")

    if args.sample:
        # Take 2 sessions from different subjects
        sessions_info = [sessions_info[0], sessions_info[len(sessions_info)//2]]
        print(f"Sample mode: processing {len(sessions_info)} sessions")

    # Process all sessions
    all_neural = []
    all_input = []
    all_output = []
    all_subject_ids = []
    all_brain_regions_per_session = []
    all_n_neurons = []

    for i, sess_info in enumerate(sessions_info):
        print(f"\n[{i+1}/{len(sessions_info)}] Processing {sess_info['filename']}...")
        result = process_session(
            sess_info['filepath'],
            show_processing=args.show_processing,
            session_idx=i
        )

        if result is None:
            continue

        if result['n_trials'] < 2:
            print(f"  Skipping: fewer than 2 valid trials")
            continue

        all_neural.append(result['neural'])
        all_input.append(result['input'])
        all_output.append(result['output'])
        all_subject_ids.append(result['subject'])
        all_brain_regions_per_session.append(result['brain_region'])
        all_n_neurons.append(result['n_neurons'])

    n_sessions = len(all_neural)
    print(f"\n=== Processed {n_sessions} sessions ===")

    # Build subject index
    unique_subjects = sorted(set(all_subject_ids))
    subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])

    # Build brain region index
    unique_regions = sorted(set(all_brain_regions_per_session))
    brain_region_idx = []
    for i in range(n_sessions):
        region = all_brain_regions_per_session[i]
        region_id = unique_regions.index(region)
        brain_region_idx.append(np.full(all_n_neurons[i], region_id, dtype=np.int64))

    # Compute time bin size
    time_bin_ms = 1000.0 / IMAGING_RATE_NOMINAL

    # Build the output dictionary
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,

        'subjects': unique_subjects,
        'subject_idx': subject_idx,

        'brain_regions': unique_regions,
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
            ['< -50cm', '-50 to -10cm', '-10 to 0cm', '0cm (in zone)', '0 to +10cm', '+10 to +50cm', '> +50cm'],
            ['0-90cm', '90-180cm', '180-270cm', '270-360cm', '360-450cm'],
            ['< 2 cm/s', '2-10 cm/s', '10-20 cm/s', '20-40 cm/s', '> 40 cm/s'],
            ['no lick', 'lick'],
            ['A', 'B', 'C'],
            ['no reward', 'reward'],
        ],

        'metadata': {
            'task_description': 'Hidden reward zone navigation in virtual linear track with reward zone switches. '
                               'Mice run on 450cm track, reward zone (50cm) at one of 3 locations (A/B/C), '
                               'switches after trial 30 on switch days. ~15% reward omission.',
            'time_bin_size': time_bin_ms,
            'temporal_alignment_event': 'start of trial (entry to linear track)',
            'off_start': 0.0,
            'off_end': None,  # variable trial length
            'imaging_rate_hz': IMAGING_RATE_NOMINAL,
            'track_length_cm': TRACK_LENGTH,
            'reward_zones': REWARD_ZONE_DICT,
            'n_sessions': n_sessions,
            'n_subjects': len(unique_subjects),
            'neural_data_type': 'deconvolved calcium events',
            'cell_filtering': 'Suite2p iscell + interneuron exclusion (speed-dFF corr > 0.5)',
        }
    }

    # Print summary statistics
    total_trials = sum(len(s) for s in all_neural)
    total_neurons = sum(all_n_neurons)
    mean_neurons = np.mean(all_n_neurons)
    trial_counts = [len(s) for s in all_neural]

    print(f"\n=== Summary ===")
    print(f"Subjects: {len(unique_subjects)} ({unique_subjects})")
    print(f"Sessions: {n_sessions}")
    print(f"Total trials: {total_trials}")
    print(f"Trials/session: {np.mean(trial_counts):.1f} +/- {np.std(trial_counts):.1f}")
    print(f"Total neurons: {total_neurons}")
    print(f"Neurons/session: {mean_neurons:.1f} +/- {np.std(all_n_neurons):.1f}")
    print(f"Brain regions: {unique_regions}")
    print(f"Time bin: {time_bin_ms:.2f} ms")

    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    file_size = os.path.getsize(args.output) / (1024 * 1024)
    t_end = time.time()
    print(f"Saved {args.output} ({file_size:.1f} MB)")
    print(f"Total time: {t_end - t_start:.1f}s")


if __name__ == '__main__':
    main()
