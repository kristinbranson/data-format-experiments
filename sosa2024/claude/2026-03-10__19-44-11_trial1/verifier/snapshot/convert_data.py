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
from scipy.ndimage import gaussian_filter1d, minimum_filter1d, maximum_filter1d
from scipy.stats import pearsonr

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================
# Constants from reference code/paper
# ============================================================
TRACK_LENGTH = 450.0  # cm
REWARD_ZONE_DICT = {
    'A': [80, 130],   # = 'X' in code
    'B': [200, 250],  # = 'Y' in code
    'C': [320, 370],  # = 'Z' in code
}
DEFAULT_CHANGE_TRIAL = 30  # 0-indexed trial where reward zone switches
INTERNEURON_SPEED_CORR_THRESHOLD = 0.5
LICK_ERROR_FRACTION_THRESHOLD = 0.35  # from reference code (>35% samples with lick>2)
NEUROPIL_COEF = 0.7
DFF_SMOOTH_SIGMA = 2  # samples
BASELINE_WINDOW = 300  # samples (~20 sec at 15.5 Hz)

# Position bins for output discretization
POSITION_BINS = 5  # equal-size bins over [0, 450]
POSITION_BIN_EDGES = np.linspace(0, TRACK_LENGTH, POSITION_BINS + 1)

# Speed bins
SPEED_BIN_EDGES = [0, 2, 10, 20, 40, np.inf]

# Distance to reward zone bins
DIST_RZ_BIN_EDGES = [-np.inf, -50, -10, 0, 0, 10, 50, np.inf]
# Special handling: bin 3 is exactly 0 (inside reward zone)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert NWB data to decoder format')
    parser.add_argument('output', type=str, help='Output pickle file path')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--full', action='store_true', help='Process all sessions (default)')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    return parser.parse_args()


def get_scene_from_identifier(identifier):
    """Extract scene name from NWB identifier field.
    e.g. '/data/InVivoDA/GCAMP11/23_02_2023/Env1_LocationB_to_A' -> 'Env1_LocationB_to_A'
    """
    return identifier.split('/')[-1]


def get_reward_zones(scene, n_trials, change_trial=DEFAULT_CHANGE_TRIAL):
    """Map scene name to reward zone coordinates per trial.
    Mirrors behavior.get_reward_zones() from reference code.
    The reference code checks for 'X_to' in scene and scene[-1] to handle
    both within-env (Env1_LocationA_to_B) and cross-env (Env1_A_to_Env2_B) switches.

    Returns:
        rz_coords: (n_trials, 2) array of [start, stop] positions
        rz_labels: (n_trials,) array of 'A', 'B', or 'C' labels
    """
    rz_coords = np.zeros((n_trials, 2))
    rz_labels = np.empty(n_trials, dtype='U1')

    # Single-location sessions (no switch)
    if scene.endswith('LocationA') or (scene.endswith('_A') and '_to' not in scene.lower()):
        # Matches Env1_LocationA, Env2_LocationA
        # But NOT Env1_LocationB_to_A (has '_to')
        pass  # fall through to switch check below

    # Use same logic as reference code: check 'X_to' and last character
    if 'Location' in scene and '_to' not in scene:
        # Single location: Env1_LocationA, Env2_LocationB, etc.
        loc = scene.split('Location')[-1]  # 'A', 'B', or 'C'
        rz_coords[:] = REWARD_ZONE_DICT[loc]
        rz_labels[:] = loc
    elif 'A_to' in scene and scene[-1] == 'B':
        rz_coords[:change_trial] = REWARD_ZONE_DICT['A']
        rz_labels[:change_trial] = 'A'
        rz_coords[change_trial:] = REWARD_ZONE_DICT['B']
        rz_labels[change_trial:] = 'B'
    elif 'B_to' in scene and scene[-1] == 'A':
        rz_coords[:change_trial] = REWARD_ZONE_DICT['B']
        rz_labels[:change_trial] = 'B'
        rz_coords[change_trial:] = REWARD_ZONE_DICT['A']
        rz_labels[change_trial:] = 'A'
    elif 'A_to' in scene and scene[-1] == 'C':
        rz_coords[:change_trial] = REWARD_ZONE_DICT['A']
        rz_labels[:change_trial] = 'A'
        rz_coords[change_trial:] = REWARD_ZONE_DICT['C']
        rz_labels[change_trial:] = 'C'
    elif 'C_to' in scene and scene[-1] == 'A':
        rz_coords[:change_trial] = REWARD_ZONE_DICT['C']
        rz_labels[:change_trial] = 'C'
        rz_coords[change_trial:] = REWARD_ZONE_DICT['A']
        rz_labels[change_trial:] = 'A'
    elif 'B_to' in scene and scene[-1] == 'C':
        rz_coords[:change_trial] = REWARD_ZONE_DICT['B']
        rz_labels[:change_trial] = 'B'
        rz_coords[change_trial:] = REWARD_ZONE_DICT['C']
        rz_labels[change_trial:] = 'C'
    elif 'C_to' in scene and scene[-1] == 'B':
        rz_coords[:change_trial] = REWARD_ZONE_DICT['C']
        rz_labels[:change_trial] = 'C'
        rz_coords[change_trial:] = REWARD_ZONE_DICT['B']
        rz_labels[change_trial:] = 'B'
    elif 'Training' in scene:
        rz_coords[:] = [275, 325]
        rz_labels[:] = 'T'
    else:
        print(f"  WARNING: Unrecognized scene '{scene}', defaulting to zone A")
        rz_coords[:] = REWARD_ZONE_DICT['A']
        rz_labels[:] = 'A'

    return rz_coords, rz_labels


def compute_dff_trial(F_trial, Fneu_trial, baseline_window=BASELINE_WINDOW):
    """Compute dF/F for a single trial using maximin baseline.
    Mirrors preprocessing.dff() from reference code.

    Args:
        F_trial: (n_cells, n_timepoints) raw fluorescence for one trial
        Fneu_trial: (n_cells, n_timepoints) neuropil fluorescence
    Returns:
        dff: (n_cells, n_timepoints) dF/F
    """
    # Neuropil subtraction
    F_corr = F_trial - NEUROPIL_COEF * Fneu_trial

    n_cells, n_t = F_corr.shape
    if n_t < 3:
        return np.zeros_like(F_corr)

    # Maximin baseline per cell
    # 1. Gaussian smooth
    smoothed = gaussian_filter1d(F_corr, sigma=15, axis=1)
    # 2. Minimum filter
    baseline = minimum_filter1d(smoothed, size=min(baseline_window, n_t), axis=1)
    # 3. Maximum filter (morphological dilation)
    baseline = maximum_filter1d(baseline, size=min(baseline_window, n_t), axis=1)

    # dF/F = (F - baseline) / |baseline|
    abs_baseline = np.abs(baseline)
    abs_baseline[abs_baseline < 1e-10] = 1e-10  # avoid division by zero
    dff = (F_corr - baseline) / abs_baseline

    # Smooth dF/F
    dff = gaussian_filter1d(dff, sigma=DFF_SMOOTH_SIGMA, axis=1)

    return dff


def detect_interneurons(dff_all, speed_all, threshold=INTERNEURON_SPEED_CORR_THRESHOLD):
    """Detect putative interneurons by correlation of dF/F with speed.
    Vectorized implementation for speed.
    Args:
        dff_all: (n_cells, n_total_timepoints) - dF/F for all timepoints in session
        speed_all: (n_total_timepoints,) - speed for all timepoints
    Returns:
        is_interneuron: (n_cells,) boolean mask
    """
    n_cells = dff_all.shape[0]
    is_interneuron = np.zeros(n_cells, dtype=bool)

    # Only use valid (non-NaN) timepoints
    valid = ~np.isnan(speed_all) & ~np.isnan(dff_all[0])
    if valid.sum() < 10:
        return is_interneuron

    speed_valid = speed_all[valid].astype(np.float64)
    dff_valid = dff_all[:, valid].astype(np.float64)  # (n_cells, n_valid)

    # Vectorized Pearson correlation
    speed_mean = speed_valid.mean()
    speed_centered = speed_valid - speed_mean
    speed_std = np.sqrt(np.sum(speed_centered ** 2))

    if speed_std < 1e-10:
        return is_interneuron

    dff_mean = dff_valid.mean(axis=1, keepdims=True)
    dff_centered = dff_valid - dff_mean
    dff_std = np.sqrt(np.sum(dff_centered ** 2, axis=1))

    # Correlation = dot(dff_centered, speed_centered) / (dff_std * speed_std)
    corr = dff_centered @ speed_centered / (dff_std * speed_std + 1e-10)

    is_interneuron = corr > threshold

    return is_interneuron


def compute_distance_to_reward_zone(position, rz_start, rz_end):
    """Compute signed distance from position to nearest point in reward zone.
    Negative = before reward zone, 0 = inside, positive = after.
    """
    dist = np.zeros_like(position)
    before = position < rz_start
    inside = (position >= rz_start) & (position <= rz_end)
    after = position > rz_end
    dist[before] = position[before] - rz_start  # negative
    dist[inside] = 0.0
    dist[after] = position[after] - rz_end  # positive
    return dist


def discretize_distance_to_rz(dist):
    """Discretize distance to reward zone into 7 bins.
    0: < -50 cm
    1: -50 to -10 cm
    2: -10 cm to < 0 cm
    3: 0 cm (inside reward zone)
    4: >0 cm to +10 cm
    5: +10 to +50 cm
    6: > +50 cm
    """
    out = np.zeros(len(dist), dtype=np.int64)
    out[dist < -50] = 0
    out[(dist >= -50) & (dist < -10)] = 1
    out[(dist >= -10) & (dist < 0)] = 2
    out[dist == 0] = 3
    out[(dist > 0) & (dist <= 10)] = 4
    out[(dist > 10) & (dist <= 50)] = 5
    out[dist > 50] = 6
    return out


def discretize_position(position, n_bins=POSITION_BINS):
    """Discretize position into equal-sized bins."""
    bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
    binned = np.digitize(position, bin_edges) - 1
    binned = np.clip(binned, 0, n_bins - 1)
    return binned


def discretize_speed(speed):
    """Discretize speed into 5 bins.
    0: < 2 cm/s
    1: 2-10 cm/s
    2: 10-20 cm/s
    3: 20-40 cm/s
    4: > 40 cm/s
    """
    out = np.zeros(len(speed), dtype=np.int64)
    out[speed < 2] = 0
    out[(speed >= 2) & (speed < 10)] = 1
    out[(speed >= 10) & (speed < 20)] = 2
    out[(speed >= 20) & (speed < 40)] = 3
    out[speed >= 40] = 4
    return out


def rz_label_to_idx(label):
    """Convert reward zone label to index: A=0, B=1, C=2."""
    mapping = {'A': 0, 'B': 1, 'C': 2}
    return mapping.get(label, -1)


def detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time):
    """Check if any reward was delivered during the trial."""
    if len(reward_timestamps) == 0:
        return False
    return np.any((reward_timestamps >= trial_start_time) & (reward_timestamps <= trial_end_time))


def process_session(nwb_path, show_processing=False, session_label=""):
    """Process a single NWB file and return session data.

    Returns dict with keys: neural_trials, input_trials, output_trials,
        subject_id, n_neurons, brain_region, session_info, or None if invalid.
    """
    t0 = time.time()

    with h5py.File(nwb_path, 'r') as f:
        # --- Metadata ---
        subject_id = f['general/subject/subject_id'][()].decode()
        session_id = f['general/session_id'][()].decode()
        identifier = f['identifier'][()].decode()
        imaging_rate = f['general/optophysiology/ImagingPlane/imaging_rate'][()]
        scene = get_scene_from_identifier(identifier)

        bts = f['processing/behavior/BehavioralTimeSeries']

        # --- Behavioral data ---
        position = bts['position/data'][:]
        speed = bts['speed/data'][:]
        lick_raw = bts['lick/data'][:]
        rzone_cumul = bts['reward_zone/data'][:]
        trial_num = bts['trial number/data'][:]
        trial_start_flag = bts['trial_start/data'][:]
        teleport_flag = bts['teleport/data'][:]
        environment = bts['environment/data'][:]
        scanning = bts['scanning/data'][:]
        pos_timestamps = bts['position/timestamps'][:]

        # Reward: stored with separate timestamps
        reward_timestamps = bts['Reward/timestamps'][:]
        reward_data = bts['Reward/data'][:]

        # --- Neural data ---
        ophys = f['processing/ophys']
        iscell = ophys['ImageSegmentation/PlaneSegmentation/iscell'][:]
        plane_idx = ophys['ImageSegmentation/PlaneSegmentation/planeIdx'][:]

        # Get fluorescence and deconvolved data from all planes
        fluor_planes = sorted([k for k in ophys['Fluorescence'].keys() if k.startswith('plane')])

        # Load deconvolved events (already computed in NWB)
        deconv_list = []
        F_list = []
        Fneu_list = []
        plane_assignment = []

        for pi, plane_name in enumerate(fluor_planes):
            deconv_data = ophys[f'Deconvolved/{plane_name}/data'][:]  # (n_timepoints, n_rois_plane)
            F_data = ophys[f'Fluorescence/{plane_name}/data'][:]
            Fneu_data = ophys[f'Neuropil/{plane_name}/data'][:]
            deconv_list.append(deconv_data)
            F_list.append(F_data)
            Fneu_list.append(Fneu_data)

        # Concatenate across planes: (n_timepoints, n_total_rois)
        deconv_all = np.concatenate(deconv_list, axis=1)
        F_all = np.concatenate(F_list, axis=1)
        Fneu_all = np.concatenate(Fneu_list, axis=1)

    # Align lengths: neural and behavioral may differ by 1 frame in multi-plane recordings
    n_timepoints_neural = deconv_all.shape[0]
    n_timepoints_behav = len(position)
    n_timepoints_total = min(n_timepoints_neural, n_timepoints_behav)
    if n_timepoints_neural != n_timepoints_behav:
        deconv_all = deconv_all[:n_timepoints_total]
        F_all = F_all[:n_timepoints_total]
        Fneu_all = Fneu_all[:n_timepoints_total]
        position = position[:n_timepoints_total]
        speed = speed[:n_timepoints_total]
        lick_raw = lick_raw[:n_timepoints_total]
        rzone_cumul = rzone_cumul[:n_timepoints_total]
        trial_num = trial_num[:n_timepoints_total]
        trial_start_flag = trial_start_flag[:n_timepoints_total]
        teleport_flag = teleport_flag[:n_timepoints_total]
        environment = environment[:n_timepoints_total]
        scanning = scanning[:n_timepoints_total]
        pos_timestamps = pos_timestamps[:n_timepoints_total]

    n_rois_total = deconv_all.shape[1]

    # For multi-plane, effective rate per plane is imaging_rate / n_planes
    n_planes = len(fluor_planes)
    effective_rate = imaging_rate if n_planes == 1 else imaging_rate / n_planes
    time_bin_ms = 1000.0 / effective_rate

    # --- Filter by iscell ---
    cell_mask = iscell[:, 0].astype(bool)
    n_cells_iscell = cell_mask.sum()

    if n_cells_iscell < 2:
        print(f"  Skipping {session_label}: only {n_cells_iscell} cells after iscell filter")
        return None

    # --- Find trial boundaries ---
    # trial_start_flag and teleport_flag are binary indicators
    trial_start_inds = np.where(trial_start_flag > 0)[0]
    teleport_inds = np.where(teleport_flag > 0)[0]

    n_trials = min(len(trial_start_inds), len(teleport_inds))
    if n_trials < 2:
        print(f"  Skipping {session_label}: only {n_trials} trials")
        return None

    trial_start_inds = trial_start_inds[:n_trials]
    teleport_inds = teleport_inds[:n_trials]

    # --- Get reward zones ---
    rz_coords, rz_labels = get_reward_zones(scene, n_trials)

    # --- Compute dF/F for interneuron detection ---
    # We compute dF/F per trial, then concatenate for speed correlation
    F_cells = F_all[:, cell_mask].T  # (n_cells, n_timepoints)
    Fneu_cells = Fneu_all[:, cell_mask].T
    deconv_cells = deconv_all[:, cell_mask].T  # (n_cells, n_timepoints)

    # Compute dF/F per trial for interneuron detection
    dff_full = np.full_like(F_cells, np.nan)
    for i in range(n_trials):
        s = trial_start_inds[i]
        e = teleport_inds[i]
        if e <= s:
            continue
        dff_full[:, s:e] = compute_dff_trial(F_cells[:, s:e], Fneu_cells[:, s:e])

    # Detect interneurons
    is_interneuron = detect_interneurons(dff_full, speed)
    n_interneurons = is_interneuron.sum()

    # Final cell mask: iscell AND not interneuron
    final_cell_mask = ~is_interneuron
    n_final_cells = final_cell_mask.sum()

    if n_final_cells < 2:
        print(f"  Skipping {session_label}: only {n_final_cells} cells after interneuron exclusion")
        return None

    # Apply final cell filter to deconvolved events
    neural_data = deconv_cells[final_cell_mask]  # (n_final_cells, n_timepoints)

    # --- Lick processing ---
    lick = lick_raw.copy()

    # --- Process trials ---
    neural_trials = []
    input_trials = []
    output_trials = []
    valid_trial_count = 0

    prev_trial_rewarded = 0  # For the first trial, assume no previous reward

    for i in range(n_trials):
        s = trial_start_inds[i]
        e = teleport_inds[i]

        if e <= s or (e - s) < 2:
            # Still need to track prev_trial_rewarded
            trial_start_time = pos_timestamps[s] if s < len(pos_timestamps) else 0
            trial_end_time = pos_timestamps[min(e, len(pos_timestamps) - 1)]
            was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
            prev_trial_rewarded = int(was_rewarded)
            continue

        # Extract data for this trial
        trial_pos = position[s:e]
        trial_speed = speed[s:e]
        trial_lick = lick[s:e].copy()
        trial_neural = neural_data[:, s:e]  # (n_cells, n_timepoints)
        trial_env = environment[s:e]
        trial_timestamps = pos_timestamps[s:e]

        n_t = e - s

        # Lick sensor error correction (from reference code):
        # if >35% of samples have cumulative lick count > 2, set licks to NaN for this trial
        if np.sum(trial_lick > 2) / n_t > LICK_ERROR_FRACTION_THRESHOLD:
            trial_lick[:] = 0  # set to 0 instead of NaN for decoder output
        # Cap licks at 1 (binary)
        trial_lick[trial_lick > 1] = 1
        trial_lick = (trial_lick > 0).astype(np.float32)

        # --- Determine reward for this trial ---
        trial_start_time = trial_timestamps[0]
        trial_end_time = trial_timestamps[-1]
        was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)

        # --- Inputs ---
        # Time from start of trial in seconds
        time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)

        # Environment type (per trial scalar)
        env_val = np.median(trial_env[trial_env >= 0]) if np.any(trial_env >= 0) else 0.0
        env_type = np.float32(env_val)

        # Trial number (per trial scalar)
        trial_number = np.float32(i)

        # Previous trial outcome (per trial scalar)
        prev_outcome = np.float32(prev_trial_rewarded)

        # Build input array: time_from_start is (1, n_t), others are (1,) scalars
        input_data = np.array([
            time_from_start,                     # (n_t,) time-varying
        ], dtype=np.float32)  # shape (1, n_t)
        # Add per-trial scalars
        trial_scalars = np.array([env_type, trial_number, prev_outcome], dtype=np.float32)
        # Combine: first row is time-varying, then per-trial scalars as separate 1D arrays
        # Per spec: (n_input, n_timepoints) or (n_input,) for per-trial
        # We'll put time-varying first, then per-trial scalars as single values
        # Actually, the format allows mixed: time-varying have shape (n_input_tv, n_t)
        # and per-trial have shape (n_input_pt,)
        # But the spec says input_data shape: (n_input, n_timepoints) or (n_input,)
        # Let's make time-varying inputs as (1, n_t) and per-trial as (3,)
        # Actually re-reading the spec: it's a single array per trial
        # Best approach: stack all as (n_input, n_timepoints) where per-trial values are repeated
        input_arr = np.zeros((4, n_t), dtype=np.float32)
        input_arr[0, :] = time_from_start
        input_arr[1, :] = env_type  # broadcast
        input_arr[2, :] = trial_number  # broadcast
        input_arr[3, :] = prev_outcome  # broadcast

        # --- Outputs ---
        # Distance to reward zone
        rz_start, rz_end = rz_coords[i]
        dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
        dist_to_rz_binned = discretize_distance_to_rz(dist_to_rz)

        # Absolute position
        pos_binned = discretize_position(trial_pos)

        # Speed
        speed_binned = discretize_speed(trial_speed)

        # Lick (already binary)
        lick_binned = trial_lick.astype(np.int64)

        # Reward zone location (per-trial)
        rz_loc = rz_label_to_idx(rz_labels[i])

        # Reward outcome (per-trial)
        reward_outcome = int(was_rewarded)

        # Build output array: time-varying outputs + per-trial
        # Time-varying: (4, n_t), per-trial: (2,)
        # Again, make everything (n_output, n_t) with per-trial broadcast
        output_arr = np.zeros((6, n_t), dtype=np.int64)
        output_arr[0, :] = dist_to_rz_binned
        output_arr[1, :] = pos_binned
        output_arr[2, :] = speed_binned
        output_arr[3, :] = lick_binned
        output_arr[4, :] = rz_loc  # broadcast
        output_arr[5, :] = reward_outcome  # broadcast

        # Store
        neural_trials.append(trial_neural.astype(np.float32))
        input_trials.append(input_arr)
        output_trials.append(output_arr)

        # Update previous trial reward
        prev_trial_rewarded = int(was_rewarded)
        valid_trial_count += 1

    if valid_trial_count < 2:
        print(f"  Skipping {session_label}: only {valid_trial_count} valid trials")
        return None

    elapsed = time.time() - t0
    print(f"  {session_label}: {n_final_cells} neurons, {valid_trial_count} trials, "
          f"{n_interneurons} interneurons removed, {elapsed:.1f}s")

    # --- Plotting ---
    if show_processing and valid_trial_count > 0:
        plot_processing(
            neural_trials, input_trials, output_trials,
            position, speed, lick, trial_start_inds, teleport_inds,
            rz_coords, rz_labels, reward_timestamps, pos_timestamps,
            effective_rate, session_label, n_trials
        )

    return {
        'neural_trials': neural_trials,
        'input_trials': input_trials,
        'output_trials': output_trials,
        'subject_id': subject_id,
        'n_neurons': n_final_cells,
        'brain_region': 'CA1',
        'n_interneurons': n_interneurons,
        'n_iscell': n_cells_iscell,
        'session_info': f"{subject_id}_ses-{session_id}_{scene}",
        'imaging_rate': effective_rate,
    }


def plot_processing(neural_trials, input_trials, output_trials,
                    position, speed, lick, trial_start_inds, teleport_inds,
                    rz_coords, rz_labels, reward_timestamps, pos_timestamps,
                    effective_rate, session_label, n_trials):
    """Plot processing visualizations for a session."""
    fig, axes = plt.subplots(6, 2, figsize=(20, 24))
    fig.suptitle(f'Processing: {session_label}', fontsize=14)

    # Pick 2 example trials
    n_valid = len(neural_trials)
    trial_idxs = [0, min(n_valid - 1, n_valid // 2)]

    for col, ti in enumerate(trial_idxs):
        neural = neural_trials[ti]
        inp = input_trials[ti]
        out = output_trials[ti]
        n_t = neural.shape[1]
        time_axis = inp[0, :]  # time from trial start in seconds

        # Row 0: Neural activity (first 10 neurons)
        n_show = min(10, neural.shape[0])
        im = axes[0, col].imshow(neural[:n_show], aspect='auto',
                                  extent=[time_axis[0], time_axis[-1], n_show, 0])
        axes[0, col].set_title(f'Trial {ti}: Neural (first {n_show} neurons)')
        axes[0, col].set_ylabel('Neuron')

        # Row 1: Position output (discretized) and raw position overlay
        axes[1, col].plot(time_axis, out[1], 'b-', label='Position bin')
        axes[1, col].set_title('Position (discretized)')
        axes[1, col].set_ylabel('Bin')
        axes[1, col].legend()

        # Row 2: Speed output
        axes[2, col].plot(time_axis, out[2], 'g-', label='Speed bin')
        axes[2, col].set_title('Speed (discretized)')
        axes[2, col].set_ylabel('Bin')

        # Row 3: Distance to reward zone
        axes[3, col].plot(time_axis, out[0], 'r-', label='Dist to RZ bin')
        axes[3, col].set_title('Distance to Reward Zone')
        axes[3, col].set_ylabel('Bin')

        # Row 4: Lick
        axes[4, col].plot(time_axis, out[3], 'k-', label='Lick')
        axes[4, col].set_title('Lick')
        axes[4, col].set_ylabel('0/1')

        # Row 5: Inputs
        axes[5, col].plot(time_axis, inp[0], label='Time from start')
        axes[5, col].axhline(inp[1, 0], color='r', ls='--', label=f'Env={inp[1,0]:.0f}')
        axes[5, col].axhline(inp[3, 0], color='g', ls='--', label=f'PrevReward={inp[3,0]:.0f}')
        axes[5, col].set_title(f'Inputs (trial#{inp[2,0]:.0f})')
        axes[5, col].set_xlabel('Time (s)')
        axes[5, col].legend(fontsize=8)

    plt.tight_layout()
    safe_label = session_label.replace('/', '_').replace(' ', '_')
    plt.savefig(f'processing_{safe_label}.png', dpi=100)
    plt.close(fig)


def main():
    args = parse_args()
    t_start = time.time()

    # Find all NWB files
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*', '*.nwb')))

    if len(nwb_files) == 0:
        print(f"No NWB files found in {data_dir}")
        return

    print(f"Found {len(nwb_files)} NWB files")

    if args.sample:
        # Select 2 sessions from different subjects
        nwb_files = [nwb_files[0], nwb_files[len(nwb_files) // 2]]
        print(f"Sample mode: processing {len(nwb_files)} sessions")

    # Process all sessions
    all_sessions = []
    subjects_set = set()
    session_count = 0

    for nwb_path in nwb_files:
        fname = os.path.basename(nwb_path)
        session_label = fname.replace('_behavior+ophys.nwb', '')

        result = process_session(
            nwb_path,
            show_processing=args.show_processing and session_count < 2,
            session_label=session_label
        )

        if result is not None:
            all_sessions.append(result)
            subjects_set.add(result['subject_id'])
        session_count += 1

    if len(all_sessions) == 0:
        print("No valid sessions found!")
        return

    # Build the output data structure
    subjects = sorted(subjects_set)
    brain_regions = ['CA1']

    neural_list = []
    input_list = []
    output_list = []
    subject_idx = []
    brain_region_idx_list = []

    for sess in all_sessions:
        neural_list.append(sess['neural_trials'])
        input_list.append(sess['input_trials'])
        output_list.append(sess['output_trials'])
        subject_idx.append(subjects.index(sess['subject_id']))
        brain_region_idx_list.append(np.zeros(sess['n_neurons'], dtype=np.int64))

    # Get effective time bin from first session
    time_bin_ms = 1000.0 / all_sessions[0]['imaging_rate']

    data = {
        'neural': neural_list,
        'input': input_list,
        'output': output_list,

        'subjects': subjects,
        'subject_idx': np.array(subject_idx, dtype=np.int64),

        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx_list,

        'input_names': [
            'time_from_trial_start_s',
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
            ['<-50cm', '-50to-10cm', '-10to0cm', '0cm_in_zone', '0to10cm', '10to50cm', '>50cm'],
            ['0-90cm', '90-180cm', '180-270cm', '270-360cm', '360-450cm'],
            ['<2cm/s', '2-10cm/s', '10-20cm/s', '20-40cm/s', '>40cm/s'],
            ['no_lick', 'lick'],
            ['zone_A', 'zone_B', 'zone_C'],
            ['no_reward', 'reward'],
        ],

        'metadata': {
            'task_description': 'Virtual reality spatial navigation with hidden reward zone switches. '
                                'Mice traverse a 450cm linear track with operant reward delivery in a 50cm zone.',
            'time_bin_size': time_bin_ms,
            'temporal_alignment_event': 'Start of trial (entry to linear track)',
            'off_start': 0.0,
            'off_end': None,  # variable trial length
            'imaging_rate_hz': all_sessions[0]['imaging_rate'],
            'track_length_cm': TRACK_LENGTH,
            'reward_zone_size_cm': 50,
            'reward_zones': REWARD_ZONE_DICT,
            'n_sessions': len(all_sessions),
            'session_info': [s['session_info'] for s in all_sessions],
            'brain_region': 'hippocampal CA1',
            'species': 'mouse',
            'indicator': 'GCaMP7f',
            'neural_data_type': 'deconvolved calcium events (OASIS)',
        },
    }

    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    file_size = os.path.getsize(args.output) / (1024 * 1024)
    elapsed = time.time() - t_start
    print(f"Saved {args.output} ({file_size:.1f} MB)")
    print(f"Total time: {elapsed:.1f}s")

    # Print summary
    total_neurons = sum(s['n_neurons'] for s in all_sessions)
    total_trials = sum(len(s['neural_trials']) for s in all_sessions)
    print(f"\nSummary:")
    print(f"  Subjects: {len(subjects)} ({', '.join(subjects)})")
    print(f"  Sessions: {len(all_sessions)}")
    print(f"  Total neurons: {total_neurons}")
    print(f"  Mean neurons/session: {total_neurons/len(all_sessions):.1f}")
    print(f"  Total trials: {total_trials}")
    print(f"  Mean trials/session: {total_trials/len(all_sessions):.1f}")
    print(f"  Time bin: {time_bin_ms:.2f} ms")


if __name__ == '__main__':
    main()
