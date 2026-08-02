#!/usr/bin/env python3
"""Convert NWB data from Sosa et al. 2025 to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
"""

import sys
import os
import glob
import time
import argparse
import pickle
import numpy as np
import h5py
from collections import defaultdict

# Reward zone dictionary (from behavior.py)
REWARD_ZONE_DICT = {
    'A': [80, 130],   # X in code
    'B': [200, 250],  # Y in code  
    'C': [320, 370],  # Z in code
}
REWARD_ZONE_CENTERS = {k: (v[0] + v[1]) / 2 for k, v in REWARD_ZONE_DICT.items()}
# A=105, B=225, C=345

TRACK_LENGTH = 450  # cm
TRACK_START = 0     # cm
SAMPLING_RATE = 15.5  # Hz approximate
CHANGE_TRIAL = 30   # default switch trial


def parse_scene(identifier):
    """Parse scene name from NWB identifier to get environment and reward zone info.
    
    Returns:
        env_before: 'Env1' or 'Env2' (environment before switch, or only env)
        env_after: 'Env1' or 'Env2' or None (environment after switch)
        rz_before: 'A', 'B', or 'C' (reward zone before switch, or only zone)
        rz_after: 'A', 'B', or 'C' or None (reward zone after switch)
        is_switch: bool
        is_env_switch: bool
    """
    scene = identifier.split('/')[-1]
    
    # Environment switch: e.g., "Env1_C_to_Env2_B"
    if '_to_Env' in scene:
        # e.g., Env1_C_to_Env2_B
        parts = scene.split('_to_')
        before_part = parts[0]  # e.g., Env1_C
        after_part = parts[1]   # e.g., Env2_B
        
        env_before = before_part.split('_')[0]  # Env1
        rz_before_letter = before_part.split('_')[1]  # C
        
        env_after = after_part.split('_')[0]  # Env2
        rz_after_letter = after_part.split('_')[1]  # B
        
        return env_before, env_after, rz_before_letter, rz_after_letter, True, True
    
    # Reward zone switch: e.g., "Env1_LocationA_to_B"
    elif '_to_' in scene:
        parts = scene.split('_to_')
        before_part = parts[0]  # e.g., Env1_LocationA
        after_letter = parts[1]  # e.g., B
        
        env = before_part.split('_')[0]  # Env1
        before_letter = before_part.split('_')[1].replace('Location', '')  # A
        
        return env, None, before_letter, after_letter, True, False
    
    # No switch: e.g., "Env1_LocationC"
    else:
        env = scene.split('_')[0]  # Env1
        rz_letter = scene.split('_')[1].replace('Location', '')  # C
        
        return env, None, rz_letter, None, False, False


def get_reward_zone_per_trial(scene_info, n_trials, change_trial=CHANGE_TRIAL):
    """Get reward zone label (A, B, C) for each trial.
    
    Returns:
        rz_labels: list of str, length n_trials
        rz_coords: list of [start, end] positions, length n_trials
    """
    env_before, env_after, rz_before, rz_after, is_switch, is_env_switch = scene_info
    
    rz_labels = []
    rz_coords = []
    
    for i in range(n_trials):
        if is_switch and i >= change_trial:
            label = rz_after
        else:
            label = rz_before
        rz_labels.append(label)
        rz_coords.append(REWARD_ZONE_DICT[label])
    
    return rz_labels, rz_coords


def get_env_per_trial(scene_info, n_trials, change_trial=CHANGE_TRIAL):
    """Get environment type (0=Env1, 1=Env2) for each trial."""
    env_before, env_after, rz_before, rz_after, is_switch, is_env_switch = scene_info
    
    env_map = {'Env1': 0, 'Env2': 1}
    env_vals = []
    
    for i in range(n_trials):
        if is_env_switch and i >= change_trial:
            env_vals.append(env_map.get(env_after, 0))
        else:
            env_vals.append(env_map.get(env_before, 0))
    
    return env_vals


def distance_to_reward_zone(position, rz_start, rz_end):
    """Compute signed distance from position to nearest point in reward zone.
    
    Negative = before reward zone, positive = after reward zone, 0 = in reward zone.
    """
    dist = np.zeros_like(position, dtype=float)
    
    before_mask = position < rz_start
    in_mask = (position >= rz_start) & (position <= rz_end)
    after_mask = position > rz_end
    
    dist[before_mask] = position[before_mask] - rz_start
    dist[in_mask] = 0.0
    dist[after_mask] = position[after_mask] - rz_end
    
    return dist


def discretize_distance(dist):
    """Discretize distance to reward zone.
    Bins:
        0: < -50 cm
        1: -50 to -10 cm
        2: -10 cm to < 0 cm
        3: 0 cm
        4: >0 cm to +10 cm
        5: +10 to +50 cm
        6: > +50 cm
    """
    bins = np.zeros(len(dist), dtype=int)
    bins[dist < -50] = 0
    bins[(dist >= -50) & (dist < -10)] = 1
    bins[(dist >= -10) & (dist < 0)] = 2
    bins[dist == 0] = 3
    bins[(dist > 0) & (dist <= 10)] = 4
    bins[(dist > 10) & (dist <= 50)] = 5
    bins[dist > 50] = 6
    return bins


def discretize_position(position, n_bins=5):
    """Discretize absolute position into n_bins equal-sized bins.
    Track is 0-450 cm, so bins are 90 cm each.
    """
    bin_edges = np.linspace(TRACK_START, TRACK_LENGTH, n_bins + 1)
    bins = np.digitize(position, bin_edges[1:-1])  # 0 to n_bins-1
    bins = np.clip(bins, 0, n_bins - 1)
    return bins


def discretize_speed(speed):
    """Discretize speed.
    Bins:
        0: < 2 cm/s
        1: 2-10 cm/s
        2: 10-20 cm/s
        3: 20-40 cm/s
        4: > 40 cm/s
    """
    bins = np.zeros(len(speed), dtype=int)
    bins[speed < 2] = 0
    bins[(speed >= 2) & (speed < 10)] = 1
    bins[(speed >= 10) & (speed < 20)] = 2
    bins[(speed >= 20) & (speed < 40)] = 3
    bins[speed >= 40] = 4
    return bins


def process_session(nwb_path, show_processing=False, session_idx=0):
    """Process a single NWB file into the decoder format.
    
    Returns:
        neural_trials: list of (n_neurons, n_timepoints) arrays
        input_trials: list of (n_input, n_timepoints) or (n_input,) arrays
        output_trials: list of (n_output, n_timepoints) or (n_output,) arrays
        subject: str
        n_neurons: int
        session_info: dict with metadata
    """
    t0 = time.time()
    
    with h5py.File(nwb_path, 'r') as f:
        # --- Metadata ---
        identifier = f['identifier'][()].decode()
        scene = identifier.split('/')[-1]
        sess_id = f['general/session_id'][()].decode()
        subj_name = os.path.basename(nwb_path).split('_')[0]  # e.g., sub-m11
        
        # Parse scene info
        scene_info = parse_scene(identifier)
        
        # --- Load behavioral data ---
        behav = f['processing/behavior/BehavioralTimeSeries']
        position = behav['position/data'][()]
        speed = behav['speed/data'][()]
        lick = behav['lick/data'][()]
        reward_zone_signal = behav['reward_zone/data'][()]
        environment = behav['environment/data'][()]
        trial_start_signal = behav['trial_start/data'][()]
        teleport_signal = behav['teleport/data'][()]
        trial_number = behav['trial number/data'][()]
        timestamps = behav['position/timestamps'][()]
        
        # Reward events
        reward_timestamps = behav['Reward/timestamps'][()]
        
        # --- Load neural data ---
        iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][()]
        
        # Handle multi-plane sessions (m17, m18 have 2 planes)
        planes = sorted(f['processing/ophys/Deconvolved'].keys())
        deconv_planes = []
        for plane in planes:
            deconv_planes.append(f[f'processing/ophys/Deconvolved/{plane}/data'][()])
        deconv_data = np.concatenate(deconv_planes, axis=1)  # (n_timepoints, total_rois)
        
        # Verify dimensions match
        assert deconv_data.shape[1] == iscell.shape[0], \
            f"Mismatch: deconv has {deconv_data.shape[1]} ROIs but iscell has {iscell.shape[0]}"
    

    # Handle length mismatch between neural and behavioral data
    n_behav = len(position)
    n_neural = deconv_data.shape[0]
    if n_behav != n_neural:
        min_len = min(n_behav, n_neural)
        position = position[:min_len]
        speed = speed[:min_len]
        lick = lick[:min_len]
        reward_zone_signal = reward_zone_signal[:min_len]
        environment = environment[:min_len]
        trial_start_signal = trial_start_signal[:min_len]
        teleport_signal = teleport_signal[:min_len]
        trial_number = trial_number[:min_len]
        timestamps = timestamps[:min_len]
        deconv_data = deconv_data[:min_len, :]

    t_load = time.time() - t0
    
    # --- Cell filtering ---
    cell_mask = iscell[:, 0] == 1
    n_cells = int(np.sum(cell_mask))
    
    # Compute dF/F for interneuron exclusion
    # Paper: Pearson r > 0.5 between dF/F and speed -> interneuron
    # Since computing full dF/F with maximin baseline is complex,
    # we use a simpler approach: correlate deconvolved events with speed
    # This is a reasonable approximation since deconvolved events are derived from dF/F
    speed_valid = speed.copy()
    speed_valid[speed_valid < 0] = 0  # clip negative speeds
    
    # Only use valid timepoints (during trials)
    valid_mask = trial_number >= 0
    
    if np.sum(valid_mask) > 100:
        speed_corr = np.zeros(deconv_data.shape[1])
        for c in range(deconv_data.shape[1]):
            if cell_mask[c]:
                valid_neural = deconv_data[valid_mask, c]
                valid_speed = speed_valid[valid_mask]
                # Remove NaN
                both_valid = ~np.isnan(valid_neural) & ~np.isnan(valid_speed)
                if np.sum(both_valid) > 100:
                    r = np.corrcoef(valid_neural[both_valid], valid_speed[both_valid])[0, 1]
                    speed_corr[c] = r if not np.isnan(r) else 0
        
        # Exclude putative interneurons (r > 0.5 as per paper)
        interneuron_mask = speed_corr > 0.5
        n_interneurons = int(np.sum(cell_mask & interneuron_mask))
        cell_mask = cell_mask & ~interneuron_mask
    else:
        n_interneurons = 0
    
    n_cells_final = int(np.sum(cell_mask))
    cell_indices = np.where(cell_mask)[0]
    
    # Filter neural data to only cells
    neural_data = deconv_data[:, cell_indices]  # (n_timepoints, n_cells_final)
    
    # --- Trial boundaries ---
    trial_starts = np.where(trial_start_signal > 0)[0]
    teleports = np.where(teleport_signal > 0)[0]
    n_trials = len(trial_starts)
    
    if len(teleports) != n_trials:
        print(f"  WARNING: {nwb_path} has {n_trials} trial starts but {len(teleports)} teleports")
        n_trials = min(n_trials, len(teleports))
        trial_starts = trial_starts[:n_trials]
        teleports = teleports[:n_trials]
    
    # --- Get reward zone and environment per trial ---
    rz_labels, rz_coords = get_reward_zone_per_trial(scene_info, n_trials)
    env_per_trial = get_env_per_trial(scene_info, n_trials)
    
    # Verify reward zone assignment by checking position where rz signal > 0
    # (sanity check for first few trials)
    
    # --- Determine reward per trial ---
    is_rewarded = np.zeros(n_trials, dtype=int)
    for i in range(n_trials):
        start = trial_starts[i]
        stop = teleports[i]
        trial_start_time = timestamps[start]
        trial_end_time = timestamps[stop - 1] if stop > start else timestamps[start]
        # Check if any reward event falls within this trial's time window
        reward_in_trial = np.any(
            (reward_timestamps >= trial_start_time) & 
            (reward_timestamps <= trial_end_time)
        )
        is_rewarded[i] = int(reward_in_trial)
    
    # --- Lick sensor error correction ---
    # Following the reference code: if >35% of samples have cumulative lick count >2, set to NaN
    # (Paper says >30%, code uses 0.35)
    lick_corrected = lick.copy()
    for i in range(n_trials):
        start = trial_starts[i]
        stop = teleports[i]
        trial_licks = lick_corrected[start:stop]
        if len(trial_licks) > 0:
            frac_high = np.sum(trial_licks > 2) / len(trial_licks)
            if frac_high > 0.35:
                lick_corrected[start:stop] = np.nan
    
    # Convert licks to binary (as in reference code)
    lick_binary = lick_corrected.copy()
    lick_binary[lick_binary > 1] = 1
    lick_binary[np.isnan(lick_binary)] = 0  # treat NaN licks as 0
    
    # --- Compute time bin size ---
    dt = np.median(np.diff(timestamps))
    
    # --- Build trial-level data ---
    neural_trials = []
    input_trials = []
    output_trials = []
    
    # Previous trial outcome (for first trial, use 0 = no previous)
    prev_outcome = np.zeros(n_trials, dtype=int)
    for i in range(1, n_trials):
        prev_outcome[i] = is_rewarded[i - 1]
    
    for i in range(n_trials):
        start = trial_starts[i]
        stop = teleports[i]
        
        if stop <= start:
            continue
        
        n_tp = stop - start
        
        # --- Neural data ---
        # (n_neurons, n_timepoints)
        trial_neural = neural_data[start:stop, :].T.astype(np.float32)
        
        # --- Time from trial start ---
        time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
        
        # --- Position ---
        trial_pos = position[start:stop].astype(np.float32)
        # Clip position to valid range
        trial_pos = np.clip(trial_pos, TRACK_START, TRACK_LENGTH)
        
        # --- Speed ---
        trial_speed = speed[start:stop].astype(np.float32)
        trial_speed = np.clip(trial_speed, 0, None)  # clip negative speeds
        
        # --- Lick ---
        trial_lick = lick_binary[start:stop].astype(np.float32)
        
        # --- Distance to reward zone ---
        rz_start, rz_end = rz_coords[i]
        trial_dist = distance_to_reward_zone(trial_pos, rz_start, rz_end)
        
        # --- Discretize outputs ---
        dist_disc = discretize_distance(trial_dist)
        pos_disc = discretize_position(trial_pos)
        speed_disc = discretize_speed(trial_speed)
        lick_disc = (trial_lick > 0).astype(int)
        
        # Reward zone location: A=0, B=1, C=2
        rz_label_map = {'A': 0, 'B': 1, 'C': 2}
        rz_loc = rz_label_map[rz_labels[i]]
        
        # Reward outcome: 0=no, 1=yes
        reward_out = is_rewarded[i]
        
        # --- Build input array ---
        # Inputs: time_from_start, env_type, trial_number, prev_outcome
        # time_from_start: time-varying
        # env_type: per-trial (broadcast)
        # trial_number: per-trial (broadcast)
        # prev_outcome: per-trial (broadcast)
        input_arr = np.array([
            time_from_start,
            np.full(n_tp, env_per_trial[i], dtype=np.float32),
            np.full(n_tp, i, dtype=np.float32),  # trial number within session
            np.full(n_tp, prev_outcome[i], dtype=np.float32),
        ], dtype=np.float32)  # (4, n_tp)
        
        # --- Build output array ---
        # Time-varying outputs: dist_disc, pos_disc, speed_disc, lick_disc
        # Per-trial outputs: rz_loc, reward_out
        output_arr = np.array([
            dist_disc,
            pos_disc,
            speed_disc,
            lick_disc,
            np.full(n_tp, rz_loc, dtype=int),
            np.full(n_tp, reward_out, dtype=int),
        ], dtype=int)  # (6, n_tp)
        
        neural_trials.append(trial_neural)
        input_trials.append(input_arr)
        output_trials.append(output_arr)
    
    t_process = time.time() - t0
    
    session_info = {
        'scene': scene,
        'sess_id': sess_id,
        'n_rois': len(iscell),
        'n_cells_iscell': n_cells,
        'n_interneurons_excluded': n_interneurons,
        'n_cells_final': n_cells_final,
        'n_trials': len(neural_trials),
        'dt': dt,
        'identifier': identifier,
    }
    
    print(f"  {subj_name} ses-{sess_id} ({scene}): {n_cells_final} cells, {len(neural_trials)} trials, "
          f"load={t_load:.1f}s, total={t_process:.1f}s")
    
    # --- Show processing plots ---
    if show_processing:
        plot_processing(nwb_path, neural_trials, input_trials, output_trials, 
                       session_info, timestamps, trial_starts, teleports,
                       position, speed, lick_binary, is_rewarded, rz_labels, rz_coords,
                       env_per_trial, session_idx)
    
    return neural_trials, input_trials, output_trials, subj_name, n_cells_final, session_info


def plot_processing(nwb_path, neural_trials, input_trials, output_trials,
                   session_info, timestamps, trial_starts, teleports,
                   position, speed, lick, is_rewarded, rz_labels, rz_coords,
                   env_per_trial, session_idx):
    """Plot processing visualizations for verification."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(6, 2, figsize=(20, 24))
    fig.suptitle(f"Processing: {session_info['scene']} (ses-{session_info['sess_id']})", fontsize=14)
    
    n_trials_to_show = min(5, len(neural_trials))
    
    # Plot 1: Position traces for first few trials
    ax = axes[0, 0]
    for i in range(n_trials_to_show):
        t = input_trials[i][0]  # time from start
        pos = output_trials[i][1]  # discretized position
        ax.plot(t, pos, alpha=0.7, label=f'Trial {i}')
    ax.set_xlabel('Time from trial start (s)')
    ax.set_ylabel('Position bin')
    ax.set_title('Discretized Position')
    ax.legend(fontsize=8)
    
    # Plot 2: Speed traces
    ax = axes[0, 1]
    for i in range(n_trials_to_show):
        t = input_trials[i][0]
        spd = output_trials[i][2]  # discretized speed
        ax.plot(t, spd, alpha=0.7, label=f'Trial {i}')
    ax.set_xlabel('Time from trial start (s)')
    ax.set_ylabel('Speed bin')
    ax.set_title('Discretized Speed')
    ax.legend(fontsize=8)
    
    # Plot 3: Distance to reward zone
    ax = axes[1, 0]
    for i in range(n_trials_to_show):
        t = input_trials[i][0]
        dist = output_trials[i][0]  # discretized distance
        ax.plot(t, dist, alpha=0.7, label=f'Trial {i} RZ={rz_labels[i]}')
    ax.set_xlabel('Time from trial start (s)')
    ax.set_ylabel('Distance bin')
    ax.set_title('Discretized Distance to Reward Zone')
    ax.legend(fontsize=8)
    
    # Plot 4: Neural activity (mean across neurons)
    ax = axes[1, 1]
    for i in range(n_trials_to_show):
        t = input_trials[i][0]
        mean_neural = np.mean(neural_trials[i], axis=0)
        ax.plot(t, mean_neural, alpha=0.7, label=f'Trial {i}')
    ax.set_xlabel('Time from trial start (s)')
    ax.set_ylabel('Mean deconv activity')
    ax.set_title('Mean Neural Activity')
    ax.legend(fontsize=8)
    
    # Plot 5: Lick raster
    ax = axes[2, 0]
    for i in range(n_trials_to_show):
        t = input_trials[i][0]
        lk = output_trials[i][3]  # lick
        ax.plot(t, lk + i * 1.5, alpha=0.7, label=f'Trial {i}')
    ax.set_xlabel('Time from trial start (s)')
    ax.set_ylabel('Lick (offset)')
    ax.set_title('Lick Events')
    
    # Plot 6: Reward zone location distribution
    ax = axes[2, 1]
    rz_counts = {}
    for rl in rz_labels:
        rz_counts[rl] = rz_counts.get(rl, 0) + 1
    ax.bar(rz_counts.keys(), rz_counts.values())
    ax.set_xlabel('Reward Zone')
    ax.set_ylabel('Count')
    ax.set_title('Reward Zone Distribution')
    
    # Plot 7: Reward outcome distribution
    ax = axes[3, 0]
    n_rewarded = sum(is_rewarded)
    n_omitted = len(is_rewarded) - n_rewarded
    ax.bar(['Rewarded', 'Omitted'], [n_rewarded, n_omitted])
    ax.set_title(f'Reward Outcomes (omission rate: {n_omitted/len(is_rewarded):.2%})')
    
    # Plot 8: Environment distribution
    ax = axes[3, 1]
    env_counts = {}
    for e in env_per_trial:
        env_counts[e] = env_counts.get(e, 0) + 1
    ax.bar([f'Env{k+1}' for k in sorted(env_counts.keys())], 
           [env_counts[k] for k in sorted(env_counts.keys())])
    ax.set_title('Environment Distribution')
    
    # Plot 9: Neural activity heatmap for one trial
    ax = axes[4, 0]
    trial_idx = 0
    n_show = min(50, neural_trials[trial_idx].shape[0])
    ax.imshow(neural_trials[trial_idx][:n_show, :], aspect='auto', cmap='viridis')
    ax.set_xlabel('Time bins')
    ax.set_ylabel('Neuron')
    ax.set_title(f'Neural Activity Heatmap (Trial {trial_idx}, first {n_show} neurons)')
    
    # Plot 10: Input variables
    ax = axes[4, 1]
    for i in range(n_trials_to_show):
        t = input_trials[i][0]
        ax.plot(t, input_trials[i][1], 'o', markersize=1, alpha=0.3)  # env
    ax.set_xlabel('Time from trial start (s)')
    ax.set_ylabel('Environment')
    ax.set_title('Environment per trial')
    
    # Plot 11: Trial duration distribution
    ax = axes[5, 0]
    durations = [input_trials[i][0][-1] for i in range(len(input_trials))]
    ax.hist(durations, bins=30)
    ax.set_xlabel('Trial duration (s)')
    ax.set_ylabel('Count')
    ax.set_title('Trial Duration Distribution')
    
    # Plot 12: Neurons per trial check
    ax = axes[5, 1]
    n_neurons_per_trial = [neural_trials[i].shape[0] for i in range(len(neural_trials))]
    ax.plot(n_neurons_per_trial)
    ax.set_xlabel('Trial')
    ax.set_ylabel('N neurons')
    ax.set_title('Neurons per Trial (should be constant)')
    
    plt.tight_layout()
    plt.savefig(f'processing_session{session_idx}.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved processing_session{session_idx}.png")


def main():
    parser = argparse.ArgumentParser(description='Convert NWB data to decoder format')
    parser.add_argument('output', help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    args = parser.parse_args()
    
    if args.sample:
        args.full = False
    
    # Find all NWB files
    all_files = sorted(glob.glob('data/sub-*/*.nwb'))
    print(f"Found {len(all_files)} NWB files")
    
    if args.sample:
        # Select 2 sessions from different subjects with different characteristics
        # Pick one with switch and one without
        sample_files = [all_files[0], all_files[14]]  # sub-m11 ses-03, sub-m12 ses-01
        files_to_process = sample_files
        print(f"Sample mode: processing {len(files_to_process)} sessions")
    else:
        files_to_process = all_files
        print(f"Full mode: processing {len(files_to_process)} sessions")
    
    # Process all sessions
    all_neural = []
    all_input = []
    all_output = []
    all_subjects = []
    all_subject_idx = []
    all_brain_region_idx = []
    session_infos = []
    
    subject_list = []  # unique subjects
    
    total_start = time.time()
    
    for file_idx, nwb_path in enumerate(files_to_process):
        print(f"\nProcessing {file_idx + 1}/{len(files_to_process)}: {os.path.basename(nwb_path)}")
        
        show = args.show_processing and file_idx < 2
        
        result = process_session(nwb_path, show_processing=show, session_idx=file_idx)
        neural_trials, input_trials, output_trials, subj_name, n_cells, session_info = result
        
        if len(neural_trials) < 2:
            print(f"  SKIPPING: only {len(neural_trials)} trials")
            continue
        
        # Track subjects
        if subj_name not in subject_list:
            subject_list.append(subj_name)
        subj_idx = subject_list.index(subj_name)
        
        all_neural.append(neural_trials)
        all_input.append(input_trials)
        all_output.append(output_trials)
        all_subject_idx.append(subj_idx)
        all_brain_region_idx.append(np.zeros(n_cells, dtype=int))  # all CA1
        session_infos.append(session_info)
    
    total_time = time.time() - total_start
    print(f"\nTotal processing time: {total_time:.1f}s")
    print(f"Average per session: {total_time / len(files_to_process):.1f}s")
    
    # Get time bin size from first session
    dt_ms = session_infos[0]['dt'] * 1000  # convert to ms
    
    # Build the output dictionary
    data = {
        'neural': all_neural,
        
        'input': all_input,
        
        'output': all_output,
        
        'subjects': subject_list,
        'subject_idx': np.array(all_subject_idx),
        
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
            ['< -50cm', '-50 to -10cm', '-10 to 0cm', '0cm (in zone)', '>0 to +10cm', '+10 to +50cm', '> +50cm'],
            ['0-90cm', '90-180cm', '180-270cm', '270-360cm', '360-450cm'],
            ['< 2cm/s', '2-10cm/s', '10-20cm/s', '20-40cm/s', '> 40cm/s'],
            ['no_lick', 'lick'],
            ['zone_A', 'zone_B', 'zone_C'],
            ['no_reward', 'reward'],
        ],
        
        'metadata': {
            'task_description': 'Head-fixed mice navigate a 450cm virtual linear track with a hidden 50cm reward zone. '
                               'Reward zone location switches between sessions. Two environments (Env1, Env2) with distinct visual features. '
                               'Reward randomly omitted on ~15% of trials.',
            'time_bin_size': dt_ms,
            'temporal_alignment_event': 'trial_start',
            'off_start': 0.0,
            'off_end': None,  # variable trial length
            'track_length_cm': TRACK_LENGTH,
            'reward_zones': REWARD_ZONE_DICT,
            'imaging_rate_hz': 1000.0 / dt_ms,
            'neural_data_type': 'deconvolved_calcium_events',
            'cell_filtering': 'suite2p_iscell + interneuron_exclusion (speed_corr > 0.5)',
            'session_info': [s for s in session_infos],
            'source': 'Sosa et al. 2025, Nature Neuroscience',
            'species': 'mouse',
            'brain_area': 'hippocampal CA1',
        }
    }
    
    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    
    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Saved {args.output} ({file_size:.1f} MB)")
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Subjects: {len(subject_list)} ({', '.join(subject_list)})")
    print(f"Sessions: {len(all_neural)}")
    print(f"Total trials: {sum(len(s) for s in all_neural)}")
    print(f"Total neurons: {sum(len(br) for br in all_brain_region_idx)}")
    print(f"Neurons/session: {np.mean([len(br) for br in all_brain_region_idx]):.1f} "
          f"(range: {min(len(br) for br in all_brain_region_idx)}-{max(len(br) for br in all_brain_region_idx)})")
    print(f"Trials/session: {np.mean([len(s) for s in all_neural]):.1f} "
          f"(range: {min(len(s) for s in all_neural)}-{max(len(s) for s in all_neural)})")
    print(f"Time bin: {dt_ms:.2f} ms")
    
    # Print output distributions
    print(f"\n=== Output Distributions ===")
    for out_idx, out_name in enumerate(data['output_names']):
        all_vals = []
        for sess_outputs in all_output:
            for trial_output in sess_outputs:
                all_vals.extend(trial_output[out_idx].tolist())
        all_vals = np.array(all_vals)
        unique, counts = np.unique(all_vals, return_counts=True)
        fracs = counts / len(all_vals)
        print(f"  {out_name}:")
        for u, c, fr in zip(unique, counts, fracs):
            val_name = data['output_values'][out_idx][int(u)] if int(u) < len(data['output_values'][out_idx]) else f'val_{int(u)}'
            print(f"    {val_name}: {c} ({fr:.3f})")


if __name__ == '__main__':
    main()
