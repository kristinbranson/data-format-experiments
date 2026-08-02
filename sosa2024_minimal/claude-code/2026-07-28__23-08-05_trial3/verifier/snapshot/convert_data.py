#!/usr/bin/env python3
"""
Convert NWB data from Sosa et al. (2025) "A flexible hippocampal population code
for experience relative to reward" into the standardized decoder format.

Processing follows the reference paper and code:
- Neural data: deconvolved calcium activity (OASIS algorithm, already in NWB)
- Cell filtering: manual curation (iscell) + putative interneuron exclusion (dF/F-speed corr > 0.5)
- Trial structure: trial_start to teleport boundaries
- Temporal alignment: start of each trial
- Reward zones: A=[80,130], B=[200,250], C=[320,370] cm
- Environment encoding: 0=ENV1, 1=ENV2
- Imaging rate: ~15.5 Hz per plane
"""

import numpy as np
import h5py
import os
import pickle
import sys
from glob import glob
from scipy import ndimage, stats

# ============================================================
# Configuration
# ============================================================

DATA_DIR = '/app/data'
OUTPUT_FILE = '/app/converted_data.pkl'
SAMPLE_OUTPUT_FILE = '/app/sample_data.pkl'

# Reward zone definitions (from behavior.py in reference code)
REWARD_ZONES = {
    'A': [80, 130],   # Zone X in code
    'B': [200, 250],  # Zone Y in code
    'C': [320, 370],  # Zone Z in code
}

TRACK_LENGTH = 450  # cm

# Interneuron exclusion threshold (from methods)
INTERNEURON_SPEED_CORR_THRESHOLD = 0.5

# Lick sensor error threshold (from behavior.py: correction_thr=0.35)
LICK_CORRECTION_THR = 0.35


# ============================================================
# Helper functions
# ============================================================

def get_reward_zone_label(position_when_in_rzone):
    """Determine reward zone label from positions where rzone > 0."""
    if len(position_when_in_rzone) == 0:
        return None
    median_pos = np.median(position_when_in_rzone)
    for label, (start, end) in REWARD_ZONES.items():
        if start - 20 <= median_pos <= end + 20:
            return label
    return None


def determine_reward_zone_per_trial(pos, rzone, tstart_inds, teleport_inds):
    """Determine the active reward zone for each trial.

    For trials where the animal enters the reward zone (rzone > 0),
    determine which zone from position. For omission trials or trials
    where the zone is not entered, inherit from neighboring trials.
    """
    n_trials = len(tstart_inds)
    zone_labels = [None] * n_trials

    # First pass: determine from rzone activation
    for i in range(n_trials):
        s, e = tstart_inds[i], teleport_inds[i]
        rz_mask = rzone[s:e] > 0
        if np.any(rz_mask):
            rz_positions = pos[s:e][rz_mask]
            zone_labels[i] = get_reward_zone_label(rz_positions)

    # Second pass: fill in missing labels from neighbors
    # Forward fill first
    last_label = None
    for i in range(n_trials):
        if zone_labels[i] is not None:
            last_label = zone_labels[i]
        elif last_label is not None:
            zone_labels[i] = last_label

    # Backward fill for any remaining
    last_label = None
    for i in range(n_trials - 1, -1, -1):
        if zone_labels[i] is not None:
            last_label = zone_labels[i]
        elif last_label is not None:
            zone_labels[i] = last_label

    return zone_labels


def compute_distance_to_reward_zone(position, rz_start, rz_end):
    """Compute signed distance from position to nearest edge of reward zone.

    Negative = before zone, 0 = inside zone, positive = after zone.
    """
    distance = np.zeros_like(position, dtype=float)
    before = position < rz_start
    inside = (position >= rz_start) & (position <= rz_end)
    after = position > rz_end
    distance[before] = position[before] - rz_start
    distance[inside] = 0
    distance[after] = position[after] - rz_end
    return distance


def discretize_distance(distance):
    """Discretize distance to reward zone into 7 bins.

    0: < -50 cm
    1: -50 to -10 cm
    2: -10 to < 0 cm
    3: 0 cm (in reward zone)
    4: > 0 to +10 cm
    5: +10 to +50 cm
    6: > +50 cm
    """
    bins = np.zeros_like(distance, dtype=int)
    bins[distance < -50] = 0
    bins[(distance >= -50) & (distance < -10)] = 1
    bins[(distance >= -10) & (distance < 0)] = 2
    bins[distance == 0] = 3
    bins[(distance > 0) & (distance <= 10)] = 4
    bins[(distance > 10) & (distance <= 50)] = 5
    bins[distance > 50] = 6
    return bins


def discretize_position(position, n_bins=5):
    """Discretize absolute position into n_bins equal-sized bins.

    Track is 0-450 cm, so bins are 90 cm each.
    """
    bin_size = TRACK_LENGTH / n_bins
    bins = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
    return bins


def discretize_speed(speed):
    """Discretize speed into 5 bins.

    0: < 2 cm/s
    1: 2-10 cm/s
    2: 10-20 cm/s
    3: 20-40 cm/s
    4: > 40 cm/s
    """
    bins = np.zeros_like(speed, dtype=int)
    bins[speed < 2] = 0
    bins[(speed >= 2) & (speed < 10)] = 1
    bins[(speed >= 10) & (speed < 20)] = 2
    bins[(speed >= 20) & (speed < 40)] = 3
    bins[speed >= 40] = 4
    return bins


def correct_lick_sensor_error(licks, tstart_inds, teleport_inds, correction_thr=0.35):
    """Detect and remove trials with erroneous lick detection.

    From behavior.py: trials where >correction_thr fraction of samples
    have cumulative lick count > 2 are set to NaN.
    This corresponds to the paper's >30% threshold with cumulative lick > 2
    (sustained lick rate >= 20 Hz).
    """
    licks_corrected = np.copy(licks)
    error_trials = []
    for i, (s, e) in enumerate(zip(tstart_inds, teleport_inds)):
        trial_licks = licks_corrected[s:e]
        if len(trial_licks) > 0:
            frac_high = np.sum(trial_licks > 2) / len(trial_licks)
            if frac_high > correction_thr:
                licks_corrected[s:e] = 0  # Set to 0 instead of NaN for binary
                error_trials.append(i)
    return licks_corrected, error_trials


def compute_dff_simple(fluorescence, neuropil, tstart_inds, teleport_inds,
                       neu_coef=0.7, baseline_window=300):
    """Compute simplified dF/F for interneuron detection.

    Following the reference code's approach:
    1. Neuropil subtraction: F_corrected = F - 0.7 * F_neuropil
    2. Maximin baseline per trial with 20s window (300 frames at 15 Hz)
    3. dF/F = (F_corrected - baseline) / |baseline|
    """
    n_cells = fluorescence.shape[1]
    n_timepoints = fluorescence.shape[0]

    # Neuropil correction
    f_corrected = fluorescence - neu_coef * neuropil
    # Add back neuropil mean per trial to avoid dividing by small numbers

    dff = np.full((n_timepoints, n_cells), np.nan, dtype=np.float32)

    for s, e in zip(tstart_inds, teleport_inds):
        if e <= s:
            continue
        trial_f = f_corrected[s:e, :]
        # Add back neuropil mean
        trial_f = trial_f + neu_coef * np.nanmean(neuropil[s:e, :], axis=0, keepdims=True)

        # Smooth
        trial_smooth = ndimage.uniform_filter1d(trial_f, size=15, axis=0)

        # Maximin baseline: minimum then maximum filter with 20s window
        win = min(baseline_window, trial_f.shape[0])
        baseline = ndimage.minimum_filter1d(trial_smooth, size=win, axis=0)
        baseline = ndimage.maximum_filter1d(baseline, size=win, axis=0)

        # dF/F
        abs_baseline = np.abs(baseline)
        abs_baseline[abs_baseline < 1e-6] = 1e-6
        dff[s:e, :] = (trial_f - baseline) / abs_baseline

    return dff


def detect_interneurons(dff, speed, tstart_inds, teleport_inds, threshold=0.5):
    """Detect putative interneurons by correlation between dF/F and speed.

    From the paper: Pearson correlation > 0.5 between dF/F and running speed.
    """
    n_cells = dff.shape[1]
    is_interneuron = np.zeros(n_cells, dtype=bool)

    # Create mask for valid (within-trial) timepoints
    valid_mask = np.zeros(dff.shape[0], dtype=bool)
    for s, e in zip(tstart_inds, teleport_inds):
        valid_mask[s:e] = True

    valid_mask &= ~np.isnan(dff[:, 0])

    if np.sum(valid_mask) < 10:
        return is_interneuron

    speed_valid = speed[valid_mask]

    for cell in range(n_cells):
        dff_valid = dff[valid_mask, cell]
        if np.std(dff_valid) < 1e-10 or np.std(speed_valid) < 1e-10:
            continue
        r, _ = stats.pearsonr(dff_valid, speed_valid)
        if r > threshold:
            is_interneuron[cell] = True

    return is_interneuron


def determine_reward_outcome(reward_ts, timestamps, tstart_inds, teleport_inds):
    """Determine if reward was delivered on each trial.

    Match reward timestamps to trial time windows.
    """
    n_trials = len(tstart_inds)
    outcomes = np.zeros(n_trials, dtype=int)

    for i in range(n_trials):
        t_start = timestamps[tstart_inds[i]]
        t_end = timestamps[teleport_inds[i]]
        n_rewards = np.sum((reward_ts >= t_start) & (reward_ts <= t_end))
        outcomes[i] = 1 if n_rewards > 0 else 0

    return outcomes


# ============================================================
# Main conversion
# ============================================================

def convert_session(nwb_path, subject_id, session_num):
    """Convert a single NWB session to the target format.

    Returns trial-level data or None if session should be skipped.
    """
    with h5py.File(nwb_path, 'r') as f:
        # Extract imaging rate
        imaging_rate = f['acquisition/TwoPhotonSeries/imaging_plane/imaging_rate'][()]

        # ---- Behavioral data ----
        pos = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
        speed = f['processing/behavior/BehavioralTimeSeries/speed/data'][:]
        lick = f['processing/behavior/BehavioralTimeSeries/lick/data'][:]
        rzone = f['processing/behavior/BehavioralTimeSeries/reward_zone/data'][:]
        env = f['processing/behavior/BehavioralTimeSeries/environment/data'][:]
        trial_start = f['processing/behavior/BehavioralTimeSeries/trial_start/data'][:]
        teleport = f['processing/behavior/BehavioralTimeSeries/teleport/data'][:]
        timestamps = f['processing/behavior/BehavioralTimeSeries/position/timestamps'][:]

        # Reward events (sparse)
        reward_data = f['processing/behavior/BehavioralTimeSeries/Reward/data'][:]
        reward_ts = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]

        # ---- Neural data ----
        iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:]
        planeIdx = f['processing/ophys/ImageSegmentation/PlaneSegmentation/planeIdx'][:]

        # Check for multi-plane
        deconv_keys = list(f['processing/ophys/Deconvolved'].keys())
        has_multi_plane = 'plane1' in deconv_keys

        # Load deconvolved data and fluorescence
        deconv_p0 = f['processing/ophys/Deconvolved/plane0/data'][:]
        fluor_p0 = f['processing/ophys/Fluorescence/plane0/data'][:]
        neuro_p0 = f['processing/ophys/Neuropil/plane0/data'][:]

        if has_multi_plane:
            deconv_p1 = f['processing/ophys/Deconvolved/plane1/data'][:]
            fluor_p1 = f['processing/ophys/Fluorescence/plane1/data'][:]
            neuro_p1 = f['processing/ophys/Neuropil/plane1/data'][:]

    # ---- Combine multi-plane neural data ----
    curated_mask = iscell[:, 0] == 1
    n_total_cells = iscell.shape[0]

    if has_multi_plane:
        n_p0 = deconv_p0.shape[1]
        n_p1 = deconv_p1.shape[1]
        assert n_p0 + n_p1 == n_total_cells, \
            f"Cell count mismatch: {n_p0}+{n_p1} != {n_total_cells}"
        deconv_all = np.concatenate([deconv_p0, deconv_p1], axis=1)
        fluor_all = np.concatenate([fluor_p0, fluor_p1], axis=1)
        neuro_all = np.concatenate([neuro_p0, neuro_p1], axis=1)
    else:
        deconv_all = deconv_p0
        fluor_all = fluor_p0
        neuro_all = neuro_p0

    # ---- Align lengths ----
    # Truncate all data to minimum length (occasional off-by-one in NWB)
    min_len = min(len(pos), deconv_all.shape[0])
    pos = pos[:min_len]
    speed = speed[:min_len]
    lick = lick[:min_len]
    rzone = rzone[:min_len]
    env = env[:min_len]
    trial_start = trial_start[:min_len]
    teleport = teleport[:min_len]
    timestamps = timestamps[:min_len]
    deconv_all = deconv_all[:min_len]
    fluor_all = fluor_all[:min_len]
    neuro_all = neuro_all[:min_len]

    # ---- Trial boundaries ----
    tstart_inds = np.where(trial_start > 0)[0]
    teleport_inds = np.where(teleport > 0)[0]

    # Ensure matching number of starts and teleports
    n_trials = min(len(tstart_inds), len(teleport_inds))
    tstart_inds = tstart_inds[:n_trials]
    teleport_inds = teleport_inds[:n_trials]

    # Remove trials where teleport comes before or at trial start
    valid = teleport_inds > tstart_inds
    tstart_inds = tstart_inds[valid]
    teleport_inds = teleport_inds[valid]
    n_trials = len(tstart_inds)

    if n_trials < 2:
        print(f"  Skipping {subject_id} ses-{session_num}: only {n_trials} valid trials")
        return None

    # ---- Interneuron detection ----
    # Compute dF/F for interneuron detection
    dff = compute_dff_simple(fluor_all, neuro_all, tstart_inds, teleport_inds)
    is_interneuron = detect_interneurons(dff, speed, tstart_inds, teleport_inds,
                                          threshold=INTERNEURON_SPEED_CORR_THRESHOLD)

    # Apply cell filtering: curated AND not interneuron
    cell_mask = curated_mask & ~is_interneuron
    n_curated = int(curated_mask.sum())
    n_interneurons = int((curated_mask & is_interneuron).sum())
    n_kept = int(cell_mask.sum())

    if n_kept < 2:
        print(f"  Skipping {subject_id} ses-{session_num}: only {n_kept} cells after filtering")
        return None

    # Filter neural data
    deconv_filtered = deconv_all[:, cell_mask]

    # ---- Lick sensor error correction ----
    lick_corrected, error_trials = correct_lick_sensor_error(
        lick, tstart_inds, teleport_inds, correction_thr=LICK_CORRECTION_THR)
    # Convert licks to binary
    lick_binary = (lick_corrected > 0).astype(float)

    # ---- Reward zone per trial ----
    zone_labels = determine_reward_zone_per_trial(pos, rzone, tstart_inds, teleport_inds)

    # ---- Reward outcomes ----
    reward_outcomes = determine_reward_outcome(reward_ts, timestamps, tstart_inds, teleport_inds)

    # ---- Environment type per trial ----
    env_per_trial = np.zeros(n_trials, dtype=int)
    for i in range(n_trials):
        s, e = tstart_inds[i], teleport_inds[i]
        env_vals = env[s:e]
        env_valid = env_vals[env_vals >= 0]
        if len(env_valid) > 0:
            env_per_trial[i] = int(np.median(env_valid))
        else:
            env_per_trial[i] = 0

    # ---- Build trial-level data ----
    neural_trials = []
    input_trials = []
    output_trials = []

    frame_time = 1.0 / imaging_rate  # seconds per frame

    for i in range(n_trials):
        s, e = tstart_inds[i], teleport_inds[i]
        n_timepoints = e - s

        if n_timepoints < 2:
            continue

        # Neural: (n_neurons, n_timepoints)
        neural = deconv_filtered[s:e, :].T.astype(np.float32)

        # Position and speed for this trial
        trial_pos = pos[s:e]
        trial_speed = speed[s:e]
        trial_lick = lick_binary[s:e]

        # ---- Inputs ----
        # 1. Time from start of trial (seconds)
        time_from_start = np.arange(n_timepoints) * frame_time

        # 2. Environment type (binary, per trial) - broadcast to match format
        env_type = float(env_per_trial[i])

        # 3. Trial number (continuous, per trial)
        trial_number = float(i + 1)

        # 4. Previous trial outcome (binary, per trial)
        if i == 0:
            prev_outcome = 0.0  # No previous trial
        else:
            prev_outcome = float(reward_outcomes[i - 1])

        # Input array: (4, n_timepoints) for time-varying, or (4,) for per-trial
        # Time from start is time-varying, others are per-trial
        # Use (4, n_timepoints) with broadcast for consistency
        input_data = np.zeros((4, n_timepoints), dtype=np.float32)
        input_data[0, :] = time_from_start
        input_data[1, :] = env_type
        input_data[2, :] = trial_number
        input_data[3, :] = prev_outcome

        # ---- Outputs ----
        # Get reward zone for this trial
        rz_label = zone_labels[i]
        if rz_label is None:
            rz_label = 'A'  # fallback
        rz_start, rz_end = REWARD_ZONES[rz_label]

        # 1. Distance to reward zone (discretized, time-varying)
        dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
        dist_disc = discretize_distance(dist)

        # 2. Absolute position (discretized into 5 bins, time-varying)
        # Clip position to valid range
        pos_clipped = np.clip(trial_pos, 0, TRACK_LENGTH)
        pos_disc = discretize_position(pos_clipped, n_bins=5)

        # 3. Speed (discretized, time-varying)
        speed_disc = discretize_speed(np.abs(trial_speed))

        # 4. Lick (binary, time-varying)
        lick_disc = trial_lick.astype(int)

        # 5. Reward zone location (per-trial): 0=A, 1=B, 2=C
        rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_label]

        # 6. Reward outcome (per-trial): 0=no, 1=yes
        rew_out = int(reward_outcomes[i])

        # Output: (6, n_timepoints) for time-varying, per-trial values broadcast
        output_data = np.zeros((6, n_timepoints), dtype=np.int64)
        output_data[0, :] = dist_disc
        output_data[1, :] = pos_disc
        output_data[2, :] = speed_disc
        output_data[3, :] = lick_disc
        output_data[4, :] = rz_loc
        output_data[5, :] = rew_out

        neural_trials.append(neural)
        input_trials.append(input_data)
        output_trials.append(output_data)

    if len(neural_trials) < 2:
        print(f"  Skipping {subject_id} ses-{session_num}: only {len(neural_trials)} valid trials after filtering")
        return None

    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'n_curated': n_curated,
        'n_interneurons': n_interneurons,
        'n_kept': n_kept,
        'n_trials': len(neural_trials),
        'imaging_rate': imaging_rate,
        'zone_labels': zone_labels,
        'reward_outcomes': reward_outcomes,
        'env_per_trial': env_per_trial,
        'lick_error_trials': error_trials,
    }


def convert_all(data_dir, sample_only=False, max_sessions_per_subject=None):
    """Convert all NWB files to the target format."""

    # Discover subjects
    subjects_dirs = sorted([d for d in os.listdir(data_dir)
                           if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])

    subjects = [d.replace('sub-', '') for d in subjects_dirs]
    print(f"Found {len(subjects)} subjects: {subjects}")

    all_neural = []
    all_input = []
    all_output = []
    subject_idx_list = []
    brain_region_idx_list = []

    total_sessions = 0
    total_trials = 0
    total_cells_kept = 0
    total_interneurons = 0

    for subj_i, (subj_dir, subj_id) in enumerate(zip(subjects_dirs, subjects)):
        subj_path = os.path.join(data_dir, subj_dir)
        nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))

        if max_sessions_per_subject:
            nwb_files = nwb_files[:max_sessions_per_subject]

        print(f"\nProcessing {subj_id} ({len(nwb_files)} sessions)...")

        for nwb_file in nwb_files:
            fname = os.path.basename(nwb_file)
            session_num = fname.split('ses-')[1].split('_')[0]

            print(f"  Session {session_num}...", end=" ")

            result = convert_session(nwb_file, subj_id, session_num)

            if result is None:
                continue

            all_neural.append(result['neural'])
            all_input.append(result['input'])
            all_output.append(result['output'])
            subject_idx_list.append(subj_i)
            # All cells are from CA1
            brain_region_idx_list.append(np.zeros(result['n_kept'], dtype=int))

            total_sessions += 1
            total_trials += result['n_trials']
            total_cells_kept += result['n_kept']
            total_interneurons += result['n_interneurons']

            print(f"OK: {result['n_trials']} trials, {result['n_kept']} cells "
                  f"({result['n_curated']} curated, {result['n_interneurons']} interneurons removed, "
                  f"{len(result['lick_error_trials'])} lick error trials)")

    print(f"\n{'='*60}")
    print(f"Total: {total_sessions} sessions, {total_trials} trials, "
          f"mean {total_cells_kept/total_sessions:.0f} cells/session")
    print(f"Total interneurons removed: {total_interneurons}")

    # ---- Assemble final data dictionary ----
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,

        'subjects': subjects,
        'subject_idx': np.array(subject_idx_list, dtype=int),

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
            ['< -50cm', '-50 to -10cm', '-10 to 0cm', '0cm (in zone)', '0 to +10cm', '+10 to +50cm', '> +50cm'],
            ['0-90cm', '90-180cm', '180-270cm', '270-360cm', '360-450cm'],
            ['< 2cm/s', '2-10cm/s', '10-20cm/s', '20-40cm/s', '> 40cm/s'],
            ['no_lick', 'lick'],
            ['zone_A', 'zone_B', 'zone_C'],
            ['no_reward', 'reward'],
        ],

        'metadata': {
            'task_description': 'Virtual reality navigation with hidden reward zones in hippocampal CA1 imaging. '
                               'Mice navigate a 450cm linear track with one of three possible hidden reward zones '
                               '(A: 80-130cm, B: 200-250cm, C: 320-370cm). Reward zone location switches across days.',
            'time_bin_size': 1000.0 / 15.5078125,  # ms (1/imaging_rate * 1000)
            'temporal_alignment_event': 'start of trial (re-entry into virtual environment at position 0)',
            'off_start': 0.0,
            'off_end': None,  # Variable trial length
            'track_length_cm': 450,
            'reward_zones': REWARD_ZONES,
            'n_subjects': len(subjects),
            'n_sessions': total_sessions,
            'n_trials_total': total_trials,
            'imaging_rate_hz': 15.5078125,
            'neural_data_type': 'deconvolved calcium activity (OASIS algorithm)',
            'cell_filtering': 'Manual curation (Suite2P) + putative interneuron exclusion (dF/F-speed correlation > 0.5)',
            'paper': 'Sosa, Plitt & Giocomo (2025) Nature Neuroscience',
        }
    }

    return data


def create_sample(data, max_sessions=5, max_trials=20):
    """Create a smaller sample dataset for quick testing."""
    import copy
    sample = copy.deepcopy(data)

    # Limit sessions
    n_sessions = min(max_sessions, len(data['neural']))

    sample['neural'] = []
    sample['input'] = []
    sample['output'] = []
    sample['brain_region_idx'] = []
    subject_idx = []

    for i in range(n_sessions):
        n_trials = min(max_trials, len(data['neural'][i]))
        sample['neural'].append(data['neural'][i][:n_trials])
        sample['input'].append(data['input'][i][:n_trials])
        sample['output'].append(data['output'][i][:n_trials])
        sample['brain_region_idx'].append(data['brain_region_idx'][i])
        subject_idx.append(data['subject_idx'][i])

    sample['subject_idx'] = np.array(subject_idx, dtype=int)

    return sample


def print_sanity_checks(data):
    """Print sanity checks comparing our data to the paper's statistics."""
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    n_sessions = len(data['neural'])
    n_subjects = len(data['subjects'])

    # Paper says: 11 switch mice, 14 days each (but m11 starts day 3 = 12 sessions)
    print(f"\n1. Number of subjects: {n_subjects} (paper: 11 switch mice)")

    # Sessions per subject
    for subj in range(n_subjects):
        n_sess = np.sum(data['subject_idx'] == subj)
        print(f"   {data['subjects'][subj]}: {n_sess} sessions")

    # Paper says: mean 80.5 +/- 7.4 trials per session
    trials_per_session = [len(data['neural'][i]) for i in range(n_sessions)]
    print(f"\n2. Trials per session: mean={np.mean(trials_per_session):.1f} "
          f"+/- {np.std(trials_per_session):.1f} (paper: 80.5 +/- 7.4)")

    # Paper says: 155-2172 neurons per session
    neurons_per_session = [data['neural'][i][0].shape[0] for i in range(n_sessions)]
    print(f"\n3. Neurons per session: range=[{min(neurons_per_session)}, {max(neurons_per_session)}] "
          f"(paper: 155-2172)")
    print(f"   Mean: {np.mean(neurons_per_session):.0f}")

    # Imaging rate
    print(f"\n4. Time bin size: {data['metadata']['time_bin_size']:.2f} ms "
          f"(= 1/{data['metadata']['imaging_rate_hz']:.1f} Hz)")

    # Reward omission rate (paper: ~15%)
    all_outcomes = []
    for i in range(n_sessions):
        for t in range(len(data['output'][i])):
            all_outcomes.append(data['output'][i][t][5, 0])  # reward_outcome
    all_outcomes = np.array(all_outcomes)
    omission_rate = 1 - np.mean(all_outcomes)
    print(f"\n5. Reward omission rate: {omission_rate*100:.1f}% (paper: ~15%)")

    # Check reward zone distribution
    rz_counts = {0: 0, 1: 0, 2: 0}
    for i in range(n_sessions):
        for t in range(len(data['output'][i])):
            rz = int(data['output'][i][t][4, 0])
            rz_counts[rz] += 1
    total = sum(rz_counts.values())
    print(f"\n6. Reward zone distribution: A={rz_counts[0]/total*100:.1f}%, "
          f"B={rz_counts[1]/total*100:.1f}%, C={rz_counts[2]/total*100:.1f}%")

    # Environment distribution
    env_counts = {0: 0, 1: 0}
    for i in range(n_sessions):
        for t in range(len(data['input'][i])):
            env_val = int(data['input'][i][t][1, 0])
            env_counts[env_val] = env_counts.get(env_val, 0) + 1
    total = sum(env_counts.values())
    print(f"\n7. Environment distribution: ENV1={env_counts.get(0,0)/total*100:.1f}%, "
          f"ENV2={env_counts.get(1,0)/total*100:.1f}%")

    # Speed distribution check
    print(f"\n8. Speed bin distribution (sample):")
    speed_counts = np.zeros(5)
    for i in range(min(5, n_sessions)):
        for t in range(len(data['output'][i])):
            for b in range(5):
                speed_counts[b] += np.sum(data['output'][i][t][2, :] == b)
    total = speed_counts.sum()
    speed_labels = ['<2', '2-10', '10-20', '20-40', '>40']
    for b in range(5):
        print(f"   {speed_labels[b]} cm/s: {speed_counts[b]/total*100:.1f}%")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert NWB data to decoder format')
    parser.add_argument('--sample-only', action='store_true',
                       help='Only create sample dataset (faster)')
    parser.add_argument('--max-sessions', type=int, default=None,
                       help='Max sessions per subject')
    args = parser.parse_args()

    print("Converting NWB data to decoder format...")
    print(f"Data directory: {DATA_DIR}")

    data = convert_all(DATA_DIR, sample_only=args.sample_only,
                       max_sessions_per_subject=args.max_sessions)

    # Sanity checks
    print_sanity_checks(data)

    # Save full dataset
    print(f"\nSaving full dataset to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    file_size = os.path.getsize(OUTPUT_FILE) / (1024**3)
    print(f"Full dataset saved: {file_size:.2f} GB")

    # Save sample dataset
    sample = create_sample(data)
    print(f"Saving sample dataset to {SAMPLE_OUTPUT_FILE}...")
    with open(SAMPLE_OUTPUT_FILE, 'wb') as f:
        pickle.dump(sample, f, protocol=4)
    sample_size = os.path.getsize(SAMPLE_OUTPUT_FILE) / (1024**2)
    print(f"Sample dataset saved: {sample_size:.1f} MB")

    print("\nConversion complete!")
