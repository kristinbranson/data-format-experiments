#!/usr/bin/env python3
"""
Convert Sosa et al. 2025 NWB data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full | --sample] [--show-processing]
"""

import argparse
import os
import sys
import time
import pickle
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter1d, minimum_filter1d, maximum_filter1d
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Constants
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
IMAGING_RATE = 15.5078125  # Hz
FRAME_PERIOD = 1.0 / IMAGING_RATE  # seconds
TRACK_LENGTH = 450.0  # cm
SPEED_THRESHOLD = 2.0  # cm/s
INTERNEURON_CORR_THRESHOLD = 0.5
LICK_ERROR_FRACTION = 0.35  # fraction of samples with cumulative lick > 2
NEUROPIL_COEF = 0.7
BASELINE_WINDOW = 300  # frames (~20s at 15.5 Hz)
DFF_SMOOTH_SIGMA = 2  # frames

# Reward zone definitions (cm)
REWARD_ZONES = {
    'A': (80, 130),
    'B': (200, 250),
    'C': (320, 370),
}

# Speed bins for output
SPEED_BINS = [0, 2, 10, 20, 40, np.inf]  # edges: <2, 2-10, 10-20, 20-40, >40

# Position bins (5 equal bins of 90 cm each)
POS_BIN_EDGES = np.linspace(0, TRACK_LENGTH, 6)  # [0, 90, 180, 270, 360, 450]

# Distance to reward zone bins
DIST_BINS = [-np.inf, -50, -10, 0, 0, 10, 50, np.inf]
# Special handling: bin 3 = exactly 0 (in reward zone)


# ============================================================
# Helper Functions
# ============================================================

def get_all_nwb_files(data_dir, sample=False):
    """Get all NWB files organized by subject."""
    subjects = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])

    all_files = []
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
        for nwb_file in nwb_files:
            all_files.append({
                'subject': subj.replace('sub-', ''),
                'filepath': os.path.join(subj_dir, nwb_file),
                'filename': nwb_file,
            })

    if sample:
        # Select 2 sessions from different subjects for testing
        # Pick one small (m11) and one larger subject
        sample_files = []
        for f in all_files:
            if f['subject'] == 'm11' and 'ses-03' in f['filename']:
                sample_files.append(f)
            elif f['subject'] == 'm3' and 'ses-03' in f['filename']:
                sample_files.append(f)
        return sample_files

    return all_files


def compute_dff(F, Fneu, trial_starts, teleports):
    """
    Compute dF/F following the reference code preprocessing.dff().

    Steps:
    1. Neuropil subtraction (coef=0.7)
    2. Mask out-of-trial data
    3. Maximin baseline per trial
    4. dF/F = (F - baseline) / |baseline|
    5. Gaussian smooth (sigma=2 frames)

    Parameters:
        F: (n_samples, n_neurons) raw fluorescence
        Fneu: (n_samples, n_neurons) neuropil fluorescence
        trial_starts: array of trial start indices
        teleports: array of trial end indices

    Returns:
        dff: (n_samples, n_neurons) delta F/F, NaN outside trials
    """
    n_samples, n_neurons = F.shape

    # Neuropil subtraction
    F_corr = F - NEUROPIL_COEF * Fneu

    # Initialize dFF with NaN
    dff = np.full_like(F_corr, np.nan)

    for i, (start, stop) in enumerate(zip(trial_starts, teleports)):
        if start >= n_samples or stop >= n_samples:
            continue
        trial_F = F_corr[start:stop, :].copy()

        if len(trial_F) < 10:
            continue

        # Maximin baseline per trial
        smoothed = gaussian_filter1d(trial_F, sigma=15, axis=0)
        min_filtered = minimum_filter1d(smoothed, size=BASELINE_WINDOW, axis=0)
        baseline = maximum_filter1d(min_filtered, size=BASELINE_WINDOW, axis=0)

        # dF/F
        abs_baseline = np.abs(baseline)
        abs_baseline[abs_baseline < 1e-10] = 1e-10
        trial_dff = (trial_F - baseline) / abs_baseline

        # Gaussian smooth within trial (combined loop)
        dff[start:stop, :] = gaussian_filter1d(trial_dff, sigma=DFF_SMOOTH_SIGMA, axis=0)

    return dff


def identify_interneurons(dff, speed, iscell_mask):
    """
    Identify putative interneurons by correlation of dF/F with speed.
    Vectorized implementation for performance.

    Parameters:
        dff: (n_samples, n_neurons) delta F/F
        speed: (n_samples,) speed timeseries
        iscell_mask: boolean mask for which ROIs are cells

    Returns:
        is_interneuron: boolean array of length n_iscell (True = interneuron)
    """
    n_neurons = iscell_mask.sum()
    dff_cells = dff[:, iscell_mask]  # (n_samples, n_neurons)

    # Find globally valid timepoints (speed is valid)
    speed_valid = ~np.isnan(speed)

    # For each neuron, valid = non-NaN dff AND non-NaN speed
    # Since dFF NaN pattern is the same for all neurons (based on trial boundaries),
    # we can use the first neuron's pattern
    neuron_valid = ~np.isnan(dff_cells[:, 0]) if n_neurons > 0 else np.zeros(len(speed), dtype=bool)
    valid = speed_valid & neuron_valid

    if valid.sum() < 100:
        return np.zeros(n_neurons, dtype=bool)

    # Vectorized Pearson correlation
    X = dff_cells[valid, :]  # (n_valid, n_neurons)
    Y = speed[valid]  # (n_valid,)

    # Demean
    X_mean = X.mean(axis=0, keepdims=True)
    Y_mean = Y.mean()
    X_centered = X - X_mean
    Y_centered = Y - Y_mean

    # Correlation = cov(X, Y) / (std(X) * std(Y))
    n = valid.sum()
    cov_XY = (X_centered * Y_centered[:, np.newaxis]).sum(axis=0) / (n - 1)
    std_X = np.sqrt((X_centered ** 2).sum(axis=0) / (n - 1))
    std_Y = np.sqrt((Y_centered ** 2).sum() / (n - 1))

    # Avoid division by zero
    std_X[std_X < 1e-10] = 1e-10

    r = cov_XY / (std_X * std_Y)

    is_interneuron = r > INTERNEURON_CORR_THRESHOLD
    return is_interneuron


def identify_reward_zone(position, reward_zone_signal, trial_start, trial_end):
    """
    Identify which reward zone (A, B, or C) is active for a trial.

    Uses position where reward_zone > 0 to determine zone location.
    """
    pos_trial = position[trial_start:trial_end]
    rz_trial = reward_zone_signal[trial_start:trial_end]

    in_rz = rz_trial > 0
    if not np.any(in_rz):
        return None  # No reward zone entry (omission or no entry)

    rz_pos = pos_trial[in_rz]
    mean_rz_pos = np.mean(rz_pos)

    # Map to zone A, B, or C based on center position
    best_zone = None
    best_dist = np.inf
    for zone_name, (zone_start, zone_end) in REWARD_ZONES.items():
        zone_center = (zone_start + zone_end) / 2
        dist = abs(mean_rz_pos - zone_center)
        if dist < best_dist:
            best_dist = dist
            best_zone = zone_name

    return best_zone


def compute_distance_to_reward_zone(position, rz_start, rz_end):
    """
    Compute signed distance from position to nearest point in reward zone.

    Negative = before zone (position < rz_start)
    Zero = within zone
    Positive = past zone (position > rz_end)
    """
    distance = np.zeros_like(position)

    before = position < rz_start
    inside = (position >= rz_start) & (position <= rz_end)
    after = position > rz_end

    distance[before] = position[before] - rz_start  # negative
    distance[inside] = 0.0
    distance[after] = position[after] - rz_end  # positive

    return distance


def discretize_distance(distance):
    """
    Discretize distance to reward zone into 7 bins:
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
    bins[distance == 0] = 3  # in reward zone
    bins[(distance > 0) & (distance <= 10)] = 4
    bins[(distance > 10) & (distance <= 50)] = 5
    bins[distance > 50] = 6

    return bins


def discretize_speed(speed):
    """
    Discretize speed into 5 bins:
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


def discretize_position(position):
    """
    Discretize position into 5 equal bins of 90 cm each.
    0: 0-90 cm
    1: 90-180 cm
    2: 180-270 cm
    3: 270-360 cm
    4: 360-450 cm
    """
    bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
    return bins


def process_session(filepath, show_processing=False, session_idx=0):
    """
    Process a single NWB file and return trial-organized data.

    Returns:
        session_data: dict with neural, input, output data per trial
        session_info: dict with metadata about the session
    """
    t0 = time.time()

    with h5py.File(filepath, 'r') as f:
        # ---- Load metadata ----
        subject_id = f['general']['subject']['subject_id'][()].decode() if isinstance(
            f['general']['subject']['subject_id'][()], bytes) else str(f['general']['subject']['subject_id'][()])
        session_id = f['general']['session_id'][()].decode() if isinstance(
            f['general']['session_id'][()], bytes) else str(f['general']['session_id'][()])
        imaging_rate = f['general']['optophysiology']['ImagingPlane']['imaging_rate'][()]

        # ---- Load behavioral data ----
        behav = f['processing']['behavior']['BehavioralTimeSeries']
        position = behav['position']['data'][:]
        speed = behav['speed']['data'][:]
        lick = behav['lick']['data'][:]
        reward_zone_signal = behav['reward_zone']['data'][:]
        trial_start_signal = behav['trial_start']['data'][:]
        teleport_signal = behav['teleport']['data'][:]
        trial_number = behav['trial number']['data'][:]
        environment = behav['environment']['data'][:]
        scanning = behav['scanning']['data'][:]

        # Reward events
        reward_data = behav['Reward']['data'][:]
        reward_timestamps = behav['Reward']['timestamps'][:]
        behav_timestamps = behav['position']['timestamps'][:]

        # ---- Load neural data ----
        ophys = f['processing']['ophys']
        seg = ophys['ImageSegmentation']['PlaneSegmentation']
        iscell = seg['iscell'][:, 0].astype(bool)
        plane_idx = seg['planeIdx'][:]

        # Load fluorescence and deconvolved for all planes
        planes = sorted(ophys['Deconvolved'].keys())

        if len(planes) == 1:
            deconv_data = ophys['Deconvolved']['plane0']['data'][:]  # (n_samples, n_rois)
            fluor_data = ophys['Fluorescence']['plane0']['data'][:]
            neuropil_data = ophys['Neuropil']['plane0']['data'][:]
            n_planes = 1
        else:
            # Multi-plane: concatenate ROIs across planes
            deconv_parts = []
            fluor_parts = []
            neuro_parts = []
            for plane in planes:
                deconv_parts.append(ophys['Deconvolved'][plane]['data'][:])
                fluor_parts.append(ophys['Fluorescence'][plane]['data'][:])
                neuro_parts.append(ophys['Neuropil'][plane]['data'][:])
            deconv_data = np.concatenate(deconv_parts, axis=1)
            fluor_data = np.concatenate(fluor_parts, axis=1)
            neuropil_data = np.concatenate(neuro_parts, axis=1)
            n_planes = len(planes)

    t_load = time.time() - t0

    # Handle potential length mismatch between behavioral and neural data
    n_behav_samples = len(position)
    n_neural_samples = deconv_data.shape[0]
    n_samples = min(n_behav_samples, n_neural_samples)
    if n_behav_samples != n_neural_samples:
        # Truncate to common length
        position = position[:n_samples]
        speed = speed[:n_samples]
        lick = lick[:n_samples]
        reward_zone_signal = reward_zone_signal[:n_samples]
        trial_start_signal = trial_start_signal[:n_samples]
        teleport_signal = teleport_signal[:n_samples]
        trial_number = trial_number[:n_samples]
        environment = environment[:n_samples]
        scanning = scanning[:n_samples]
        behav_timestamps = behav_timestamps[:n_samples]
        deconv_data = deconv_data[:n_samples, :]
        fluor_data = fluor_data[:n_samples, :]
        neuropil_data = neuropil_data[:n_samples, :]

    n_total_rois = deconv_data.shape[1]

    # ---- Find trial boundaries ----
    trial_starts = np.where(trial_start_signal > 0)[0]
    teleports = np.where(teleport_signal > 0)[0]

    # Ensure matching number of starts and teleports
    n_trials = min(len(trial_starts), len(teleports))
    trial_starts = trial_starts[:n_trials]
    teleports = teleports[:n_trials]

    # Ensure each teleport comes after its corresponding trial start
    valid = teleports > trial_starts
    trial_starts = trial_starts[valid]
    teleports = teleports[valid]
    n_trials = len(trial_starts)

    # ---- Compute dF/F for interneuron detection ----
    t1 = time.time()
    dff = compute_dff(fluor_data, neuropil_data, trial_starts, teleports)
    t_dff = time.time() - t1

    # ---- Identify interneurons ----
    t1 = time.time()
    is_interneuron = identify_interneurons(dff, speed, iscell)
    t_int = time.time() - t1

    # Create final neuron mask: iscell AND not interneuron
    iscell_indices = np.where(iscell)[0]
    non_interneuron = ~is_interneuron
    final_neuron_mask = np.zeros(n_total_rois, dtype=bool)
    final_neuron_mask[iscell_indices[non_interneuron]] = True

    n_neurons = final_neuron_mask.sum()
    n_interneurons = is_interneuron.sum()

    # ---- Get deconvolved data for selected neurons ----
    neural_all = deconv_data[:, final_neuron_mask]  # (n_samples, n_neurons)

    # ---- Process reward events ----
    # Map reward timestamps to frame indices
    reward_frame_indices = np.searchsorted(behav_timestamps, reward_timestamps)
    reward_frame_indices = np.clip(reward_frame_indices, 0, n_samples - 1)

    # Determine which trials have rewards
    trial_rewarded = np.zeros(n_trials, dtype=bool)
    for t in range(n_trials):
        start = trial_starts[t]
        end = teleports[t]
        trial_rewards = np.any((reward_frame_indices >= start) & (reward_frame_indices < end))
        trial_rewarded[t] = trial_rewards

    # ---- Identify reward zones per trial ----
    trial_rz_label = []  # 'A', 'B', 'C', or None
    trial_rz_start = np.zeros(n_trials)
    trial_rz_end = np.zeros(n_trials)

    # First pass: identify zones where animal entered
    last_known_zone = None
    for t in range(n_trials):
        zone = identify_reward_zone(position, reward_zone_signal, trial_starts[t], teleports[t])
        if zone is not None:
            last_known_zone = zone
        trial_rz_label.append(zone if zone is not None else last_known_zone)

    # Fill backward for any leading None values
    if trial_rz_label[0] is None:
        for t in range(n_trials):
            if trial_rz_label[t] is not None:
                for tt in range(t):
                    trial_rz_label[tt] = trial_rz_label[t]
                break

    # Set reward zone coordinates
    for t in range(n_trials):
        if trial_rz_label[t] is not None:
            trial_rz_start[t], trial_rz_end[t] = REWARD_ZONES[trial_rz_label[t]]
        else:
            # Fallback - shouldn't happen
            trial_rz_start[t], trial_rz_end[t] = 200, 250

    # Map reward zone labels to indices: A=0, B=1, C=2
    rz_label_to_idx = {'A': 0, 'B': 1, 'C': 2}
    trial_rz_idx = np.array([rz_label_to_idx.get(lbl, 0) for lbl in trial_rz_label])

    # ---- Process lick data ----
    # Following reference: binarize licks (clip to 0/1), apply error correction
    lick_binary = np.clip(lick, 0, 1).astype(np.float64)

    # Lick error correction per trial
    lick_error_trials = []
    for t in range(n_trials):
        start = trial_starts[t]
        end = teleports[t]
        trial_lick = lick[start:end]
        if len(trial_lick) > 0:
            frac_bad = np.sum(trial_lick > 2) / len(trial_lick)
            if frac_bad > LICK_ERROR_FRACTION:
                lick_binary[start:end] = np.nan
                lick_error_trials.append(t)

    # ---- Get environment per trial ----
    trial_env = np.zeros(n_trials, dtype=np.int64)
    for t in range(n_trials):
        start = trial_starts[t]
        end = teleports[t]
        env_vals = environment[start:end]
        valid_env = env_vals[env_vals >= 0]
        if len(valid_env) > 0:
            trial_env[t] = int(np.median(valid_env))
        else:
            trial_env[t] = 0

    # ---- Previous trial outcome ----
    prev_trial_outcome = np.zeros(n_trials, dtype=np.int64)
    for t in range(1, n_trials):
        prev_trial_outcome[t] = int(trial_rewarded[t - 1])
    # First trial: no previous, default to 0 (unknown/omitted)

    # ---- Build per-trial data ----
    neural_trials = []
    input_trials = []
    output_trials = []
    valid_trials = []

    for t in range(n_trials):
        start = trial_starts[t]
        end = teleports[t]
        n_timepoints = end - start

        if n_timepoints < 5:
            continue  # Skip very short trials

        # Neural: (n_neurons, n_timepoints)
        trial_neural = neural_all[start:end, :].T.copy()  # (n_neurons, n_timepoints)

        # Replace NaN with 0 in neural data (following reference: X[np.isnan(X)] = 0)
        trial_neural[np.isnan(trial_neural)] = 0

        # Input construction
        # [0] Time from trial start in seconds (time-varying)
        time_from_start = np.arange(n_timepoints) * FRAME_PERIOD

        # [1] Environment type (per trial, broadcast)
        env_val = float(trial_env[t])

        # [2] Trial number (per trial)
        trial_num = float(t)

        # [3] Previous trial outcome (per trial)
        prev_outcome = float(prev_trial_outcome[t])

        # Input array: (4,) for per-trial values, (4, n_timepoints) for time-varying
        # Time from start is time-varying; others are per-trial
        # Following the spec: (n_input, n_timepoints) or (n_input,)
        # Since time_from_start is time-varying, we need mixed format
        # Use (4,) for per-trial inputs and time_from_start as time-varying
        # Actually, the spec says input can be (n_input, n_timepoints) or (n_input)
        # Let's make time_from_start time-varying and others per-trial
        # We'll pack as: row 0 = time_from_start (time-varying)
        # rows 1-3 = per-trial values broadcast to time-varying
        trial_input = np.zeros((4, n_timepoints), dtype=np.float32)
        trial_input[0, :] = time_from_start
        trial_input[1, :] = env_val
        trial_input[2, :] = trial_num
        trial_input[3, :] = prev_outcome

        # Output construction
        trial_pos = position[start:end]
        trial_speed = speed[start:end]
        trial_lick = lick_binary[start:end]

        # [0] Distance to reward zone (time-varying, discretized)
        dist_to_rz = compute_distance_to_reward_zone(trial_pos, trial_rz_start[t], trial_rz_end[t])
        dist_bins = discretize_distance(dist_to_rz)

        # [1] Absolute position (time-varying, discretized)
        pos_bins = discretize_position(trial_pos)

        # [2] Speed (time-varying, discretized)
        speed_bins = discretize_speed(trial_speed)

        # [3] Lick (time-varying, binary)
        lick_vals = np.where(np.isnan(trial_lick), 0, trial_lick).astype(np.int64)

        # [4] Reward zone location (per trial)
        rz_loc = trial_rz_idx[t]

        # [5] Reward outcome (per trial)
        reward_out = int(trial_rewarded[t])

        # Output: time-varying outputs as (n_output, n_timepoints), per-trial as (n_output,)
        # Mix: first 4 are time-varying, last 2 are per-trial
        # Pack all as (6, n_timepoints) with per-trial broadcast
        trial_output = np.zeros((6, n_timepoints), dtype=np.int64)
        trial_output[0, :] = dist_bins
        trial_output[1, :] = pos_bins
        trial_output[2, :] = speed_bins
        trial_output[3, :] = lick_vals
        trial_output[4, :] = rz_loc  # broadcast per-trial
        trial_output[5, :] = reward_out  # broadcast per-trial

        neural_trials.append(trial_neural.astype(np.float32))
        input_trials.append(trial_input)
        output_trials.append(trial_output)
        valid_trials.append(t)

    t_total = time.time() - t0

    session_info = {
        'subject': subject_id,
        'session': session_id,
        'n_neurons': n_neurons,
        'n_interneurons': n_interneurons,
        'n_iscell': int(iscell.sum()),
        'n_trials': len(valid_trials),
        'n_trials_total': n_trials,
        'n_lick_error_trials': len(lick_error_trials),
        'n_rewarded': int(trial_rewarded.sum()),
        'imaging_rate': imaging_rate,
        'n_planes': n_planes,
        'load_time': t_load,
        'dff_time': t_dff,
        'interneuron_time': t_int,
        'total_time': t_total,
        'reward_zones': trial_rz_label,
        'environments': trial_env.tolist(),
    }

    print(f"  {subject_id} ses-{session_id}: {n_neurons} neurons ({n_interneurons} interneurons excluded), "
          f"{len(valid_trials)} trials, {int(trial_rewarded.sum())}/{n_trials} rewarded, "
          f"{len(lick_error_trials)} lick-error trials, "
          f"{t_total:.1f}s")

    # ---- Optional processing visualization ----
    if show_processing and session_idx < 2:
        plot_processing(filepath, subject_id, session_id,
                        position, speed, lick_binary, trial_starts, teleports,
                        neural_all, trial_rz_start, trial_rz_end,
                        trial_rewarded, trial_env, valid_trials,
                        neural_trials, input_trials, output_trials,
                        session_idx)

    return neural_trials, input_trials, output_trials, session_info


def plot_processing(filepath, subject_id, session_id,
                    position, speed, lick, trial_starts, teleports,
                    neural_all, trial_rz_start, trial_rz_end,
                    trial_rewarded, trial_env, valid_trials,
                    neural_trials, input_trials, output_trials,
                    session_idx):
    """Plot processing steps for visual verification."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 2, figsize=(20, 24))
    fig.suptitle(f'Processing: {subject_id} ses-{session_id}', fontsize=16)

    # Pick a representative trial
    trial_idx = min(5, len(valid_trials) - 1)
    t = valid_trials[trial_idx]
    start = trial_starts[t]
    end = teleports[t]
    n_tp = end - start
    time_sec = np.arange(n_tp) * FRAME_PERIOD

    # Panel 1: Raw position and speed for one trial
    ax = axes[0, 0]
    ax.plot(time_sec, position[start:end], label='Position (cm)')
    ax.axhline(y=trial_rz_start[t], color='r', linestyle='--', label=f'RZ start ({trial_rz_start[t]:.0f} cm)')
    ax.axhline(y=trial_rz_end[t], color='r', linestyle=':', label=f'RZ end ({trial_rz_end[t]:.0f} cm)')
    ax.set_ylabel('Position (cm)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8)
    ax.set_title(f'Trial {t}: Position')

    ax = axes[0, 1]
    ax.plot(time_sec, speed[start:end], label='Speed (cm/s)')
    ax.axhline(y=SPEED_THRESHOLD, color='r', linestyle='--', label=f'Threshold ({SPEED_THRESHOLD} cm/s)')
    ax.set_ylabel('Speed (cm/s)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8)
    ax.set_title(f'Trial {t}: Speed')

    # Panel 2: Neural data (first 10 neurons)
    ax = axes[1, 0]
    n_show = min(10, neural_trials[trial_idx].shape[0])
    for i in range(n_show):
        ax.plot(time_sec, neural_trials[trial_idx][i, :] + i * 0.5, linewidth=0.5)
    ax.set_ylabel('Neuron (offset)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'Trial {t}: Deconvolved neural (first {n_show})')

    # Panel 3: Discretized outputs
    ax = axes[1, 1]
    ax.plot(time_sec, output_trials[trial_idx][0, :], label='Dist to RZ bin')
    ax.set_ylabel('Bin')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8)
    ax.set_title(f'Trial {t}: Distance to RZ (discretized)')

    ax = axes[2, 0]
    ax.plot(time_sec, output_trials[trial_idx][1, :], label='Position bin')
    ax.set_ylabel('Bin')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8)
    ax.set_title(f'Trial {t}: Position (discretized)')

    ax = axes[2, 1]
    ax.plot(time_sec, output_trials[trial_idx][2, :], label='Speed bin')
    ax.set_ylabel('Bin')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8)
    ax.set_title(f'Trial {t}: Speed (discretized)')

    # Panel 4: Lick and reward
    ax = axes[3, 0]
    ax.plot(time_sec, output_trials[trial_idx][3, :], label='Lick')
    ax.set_ylabel('Lick (0/1)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8)
    ax.set_title(f'Trial {t}: Lick')

    # Panel 5: Continuous distance to RZ overlay with position
    ax = axes[3, 1]
    dist = compute_distance_to_reward_zone(position[start:end], trial_rz_start[t], trial_rz_end[t])
    ax.plot(time_sec, dist, label='Distance to RZ (cm)')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axhline(y=-50, color='gray', linestyle='--', linewidth=0.5)
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=0.5)
    ax.axhline(y=-10, color='gray', linestyle=':', linewidth=0.5)
    ax.axhline(y=10, color='gray', linestyle=':', linewidth=0.5)
    ax.set_ylabel('Distance (cm)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8)
    ax.set_title(f'Trial {t}: Continuous distance to RZ')

    # Panel 6: Session-level statistics
    ax = axes[4, 0]
    rz_counts = [sum(1 for lbl in [None]*0), 0, 0]  # placeholder
    reward_frac = np.mean(trial_rewarded)
    ax.bar(['Rewarded', 'Omitted'], [reward_frac, 1 - reward_frac])
    ax.set_ylabel('Fraction of trials')
    ax.set_title(f'Reward rate: {reward_frac:.2f}')

    ax = axes[4, 1]
    trial_lengths = [teleports[valid_trials[i]] - trial_starts[valid_trials[i]] for i in range(len(valid_trials))]
    ax.hist(trial_lengths, bins=20)
    ax.set_xlabel('Trial length (frames)')
    ax.set_ylabel('Count')
    ax.set_title('Trial length distribution')

    # Panel 7: Inputs overview
    ax = axes[5, 0]
    envs = [input_trials[i][1, 0] for i in range(len(input_trials))]
    ax.plot(envs, 'o', markersize=2)
    ax.set_ylabel('Environment')
    ax.set_xlabel('Trial')
    ax.set_title('Environment per trial')

    ax = axes[5, 1]
    prev_outcomes = [input_trials[i][3, 0] for i in range(len(input_trials))]
    ax.plot(prev_outcomes, 'o', markersize=2)
    ax.set_ylabel('Prev outcome')
    ax.set_xlabel('Trial')
    ax.set_title('Previous trial outcome')

    plt.tight_layout()
    plt.savefig(f'processing_{subject_id}_ses{session_id}.png', dpi=100)
    plt.close()
    print(f"  Saved processing plot: processing_{subject_id}_ses{session_id}.png")


def build_dataset(nwb_files, show_processing=False):
    """
    Build the full dataset from NWB files.

    Returns:
        data: dict in the target format
    """
    all_neural = []
    all_input = []
    all_output = []
    all_subject_ids = []
    all_session_infos = []
    all_brain_region_idx = []

    subjects_list = []
    subject_idx_list = []

    for i, file_info in enumerate(nwb_files):
        print(f"\nProcessing session {i+1}/{len(nwb_files)}: {file_info['filename']}")

        neural_trials, input_trials, output_trials, session_info = process_session(
            file_info['filepath'], show_processing=show_processing, session_idx=i)

        if len(neural_trials) < 2:
            print(f"  WARNING: Skipping session with <2 valid trials")
            continue

        all_neural.append(neural_trials)
        all_input.append(input_trials)
        all_output.append(output_trials)
        all_session_infos.append(session_info)

        subj = session_info['subject']
        if subj not in subjects_list:
            subjects_list.append(subj)
        subject_idx_list.append(subjects_list.index(subj))

        # Brain region idx: all neurons are CA1
        n_neurons = session_info['n_neurons']
        all_brain_region_idx.append(np.zeros(n_neurons, dtype=np.int64))

    # Build final data structure
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,

        'subjects': subjects_list,
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),

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
            ['< -50 cm', '-50 to -10 cm', '-10 to 0 cm', '0 cm (in zone)', '>0 to +10 cm', '+10 to +50 cm', '> +50 cm'],
            ['0-90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '360-450 cm'],
            ['< 2 cm/s', '2-10 cm/s', '10-20 cm/s', '20-40 cm/s', '> 40 cm/s'],
            ['no lick', 'lick'],
            ['A (80-130)', 'B (200-250)', 'C (320-370)'],
            ['no reward', 'reward'],
        ],

        'metadata': {
            'task_description': 'Head-fixed mice navigate a 450 cm virtual linear track with a hidden 50 cm reward zone. Reward zone switches between 3 locations across sessions. Two environments used. ~15% reward omission.',
            'time_bin_size': 1000.0 / IMAGING_RATE,  # ms per frame
            'temporal_alignment_event': 'start of each trial (first imaging frame on track after teleport)',
            'off_start': 0.0,
            'off_end': None,  # variable trial length
            'imaging_rate_hz': IMAGING_RATE,
            'track_length_cm': TRACK_LENGTH,
            'speed_threshold_cm_s': SPEED_THRESHOLD,
            'neural_data_type': 'deconvolved calcium events (OASIS)',
            'session_info': all_session_infos,
        }
    }

    return data


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Convert NWB data to decoder format.')
    parser.add_argument('output', type=str, help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')

    args = parser.parse_args()

    if args.sample:
        args.full = False

    print("=" * 60)
    print(f"NWB to Decoder Format Conversion")
    print(f"Mode: {'SAMPLE (2 sessions)' if args.sample else 'FULL (all sessions)'}")
    print(f"Output: {args.output}")
    print("=" * 60)

    t_start = time.time()

    # Get NWB files
    nwb_files = get_all_nwb_files(DATA_DIR, sample=args.sample)
    print(f"\nFound {len(nwb_files)} NWB files to process")

    # Build dataset
    data = build_dataset(nwb_files, show_processing=args.show_processing)

    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)

    file_size = os.path.getsize(args.output) / (1024 * 1024)
    t_total = time.time() - t_start

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Conversion complete!")
    print(f"Total time: {t_total:.1f}s")
    print(f"Output file: {args.output} ({file_size:.1f} MB)")
    print(f"Sessions: {len(data['neural'])}")
    print(f"Subjects: {data['subjects']}")
    total_trials = sum(len(s) for s in data['neural'])
    total_neurons = sum(data['neural'][i][0].shape[0] for i in range(len(data['neural'])) if len(data['neural'][i]) > 0)
    print(f"Total trials: {total_trials}")
    print(f"Neurons per session: {[data['neural'][i][0].shape[0] for i in range(len(data['neural'])) if len(data['neural'][i]) > 0]}")

    # Print per-session timing
    print(f"\nPer-session timing:")
    for info in data['metadata']['session_info']:
        print(f"  {info['subject']} ses-{info['session']}: "
              f"load={info['load_time']:.1f}s, dff={info['dff_time']:.1f}s, "
              f"int={info['interneuron_time']:.1f}s, total={info['total_time']:.1f}s")


if __name__ == '__main__':
    main()
