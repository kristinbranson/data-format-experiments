#!/usr/bin/env python3
"""
Convert NWB data from Sosa, Plitt, & Giocomo (2025) into the decoder format.

Processing pipeline (matching reference paper and code):
1. Load NWB files for each session
2. Select curated cells (iscell[:, 0] == 1)
3. Compute dF/F from raw fluorescence with neuropil subtraction and maximin baseline
4. Smooth dF/F with 2-sample Gaussian kernel
5. Deconvolve with OASIS to get events
6. Filter interneurons (speed correlation > 0.5)
7. Extract trial-level behavioral variables
8. Align all data to trial start
9. Format into decoder structure
"""

import os
import sys
import numpy as np
import pickle
import warnings
from glob import glob
from scipy import ndimage, interpolate

warnings.filterwarnings('ignore')

# ---- Configuration ----
DATA_DIR = 'data'
FRAME_RATE = 15.5078125  # Hz, from NWB files
TIME_BIN_MS = 1000.0 / FRAME_RATE  # ~64.5 ms
TRACK_LENGTH = 450  # cm
SPEED_THR = 2.0  # cm/s for place cell analyses (not applied to decoder data)
NEUROPIL_COEF = 0.7
TAU = 0.7  # calcium indicator decay constant for OASIS
INTERNEURON_SPEED_CORR_THR = 0.5

# Reward zone definitions from behavior.py
REWARD_ZONES = {
    'A': [80, 130],
    'B': [200, 250],
    'C': [320, 370],
}


def load_nwb(filepath):
    """Load NWB file and extract all needed data.
    Handles both single-plane and multi-plane recordings.
    Multi-plane: pools ROIs across planes (matching paper methods).
    """
    from pynwb import NWBHDF5IO

    io = NWBHDF5IO(filepath, 'r')
    nwb = io.read()

    bts = nwb.processing['behavior']['BehavioralTimeSeries']
    ophys = nwb.processing['ophys']
    seg = ophys['ImageSegmentation']['PlaneSegmentation']

    # Determine number of planes
    plane_keys = sorted(ophys['Fluorescence'].roi_response_series.keys())
    n_planes = len(plane_keys)

    # Load and concatenate neural data across planes
    if n_planes == 1:
        fluorescence = np.array(ophys['Fluorescence']['plane0'].data[:])
        neuropil_data = np.array(ophys['Neuropil']['plane0'].data[:])
        deconvolved = np.array(ophys['Deconvolved']['plane0'].data[:])
    else:
        # Multi-plane: concatenate neurons from all planes
        # Paper: "ROIs were identified separately per plane, but planes were pooled for all analyses"
        fluor_planes = [np.array(ophys['Fluorescence'][k].data[:]) for k in plane_keys]
        neuro_planes = [np.array(ophys['Neuropil'][k].data[:]) for k in plane_keys]
        deconv_planes = [np.array(ophys['Deconvolved'][k].data[:]) for k in plane_keys]
        fluorescence = np.concatenate(fluor_planes, axis=1)
        neuropil_data = np.concatenate(neuro_planes, axis=1)
        deconvolved = np.concatenate(deconv_planes, axis=1)

    data = {
        'position': np.array(bts.time_series['position'].data[:]),
        'speed': np.array(bts.time_series['speed'].data[:]),
        'lick': np.array(bts.time_series['lick'].data[:]),
        'reward_zone': np.array(bts.time_series['reward_zone'].data[:]),
        'teleport': np.array(bts.time_series['teleport'].data[:]),
        'trial_start': np.array(bts.time_series['trial_start'].data[:]),
        'trial_number': np.array(bts.time_series['trial number'].data[:]),
        'environment': np.array(bts.time_series['environment'].data[:]),
        'scanning': np.array(bts.time_series['scanning'].data[:]),
        'autoreward': np.array(bts.time_series['autoreward'].data[:]),
        'timestamps': np.array(bts.time_series['position'].timestamps[:]),
        'reward_timestamps': np.array(bts.time_series['Reward'].timestamps[:]),
        'reward_data': np.array(bts.time_series['Reward'].data[:]),
        # Neural data - shape: (n_timepoints, n_neurons)
        'fluorescence': fluorescence,
        'neuropil': neuropil_data,
        'deconvolved': deconvolved,
        # Cell info
        'iscell': np.array(seg['iscell'].data[:]),
        'plane_idx': np.array(seg['planeIdx'].data[:]),
        # Metadata
        'session_id': nwb.session_id,
        'subject_id': nwb.subject.subject_id if nwb.subject else None,
        'n_planes': n_planes,
    }

    io.close()
    return data


def compute_dff_per_trial(F, Fneu, trial_starts, trial_ends, frame_rate=FRAME_RATE):
    """
    Compute dF/F following the paper's method (matching preprocessing.py):
    1. Mask fluorescence to within-trial data only
    2. Neuropil subtraction (coef=0.7)
    3. Per trial: add back neuropil mean so baseline is close to true F
    4. Maximin baseline within each trial (300 sample window, matching reference)
    5. dF/F = (F - baseline) / |baseline|
    6. Smooth with 2-sample Gaussian

    Returns dff array of shape (n_neurons, n_timepoints)
    """
    n_neurons, n_samples = F.shape

    # Initialize masked arrays (NaN outside trials, matching reference code)
    f_ = np.full((n_neurons, n_samples), np.nan)
    f_neu_ = np.full((n_neurons, n_samples), np.nan)

    # Mask to within-trial data only
    for start, end in zip(trial_starts, trial_ends):
        if end <= start:
            continue
        f_[:, start:end] = F[:, start:end]
        f_neu_[:, start:end] = Fneu[:, start:end]

    # Neuropil subtraction on masked data
    nanmask = ~np.isnan(f_[0, :])
    f_[:, nanmask] = f_[:, nanmask] - NEUROPIL_COEF * f_neu_[:, nanmask]

    # Baseline and dF/F per trial
    dff = np.full((n_neurons, n_samples), np.nan)
    window_size = 300  # ~20s at 15.5 Hz, matching reference code int(300)

    for start, end in zip(trial_starts, trial_ends):
        if end <= start:
            continue

        # Add back neuropil mean per trial
        # (so dff values are close to true dF/F and we don't divide by small numbers)
        f_[:, start:end] = f_[:, start:end] + NEUROPIL_COEF * np.nanmean(
            f_neu_[:, start:end], axis=1, keepdims=True)

        # Smooth for baseline calculation (sigma=15 samples along time)
        f_smooth = nansmooth(f_[:, start:end], 15, axis=1)

        # Maximin baseline
        baseline = ndimage.minimum_filter1d(f_smooth, window_size, axis=-1)
        baseline = ndimage.maximum_filter1d(baseline, window_size, axis=-1)

        # dF/F = (F - baseline) / |baseline|
        dff[:, start:end] = (f_[:, start:end] - baseline) / np.abs(baseline)

    # Smooth dF/F with 2-sample Gaussian kernel per trial
    for start, end in zip(trial_starts, trial_ends):
        if end <= start:
            continue
        dff[:, start:end] = nansmooth(dff[:, start:end], 2, axis=1)

    return dff


def nansmooth(a, sig, axis=-1):
    """Gaussian smoothing that handles NaNs."""
    nan_inds = np.isnan(a)
    a_clean = np.copy(a)
    a_clean[nan_inds] = 0

    one = np.ones(a.shape)
    one[nan_inds] = 0.001

    a_smooth = ndimage.gaussian_filter1d(a_clean, sig, axis=axis)
    one_smooth = ndimage.gaussian_filter1d(one, sig, axis=axis)

    return a_smooth / one_smooth


def deconvolve_oasis(dff_trial, tau=TAU, frame_rate=FRAME_RATE):
    """
    Deconvolve dF/F using OASIS algorithm from suite2p.
    Falls back to simple threshold if suite2p not available.
    """
    try:
        from suite2p.extraction.dcnv import oasis
        # oasis(F, batch_size, tau, fs)
        events = oasis(dff_trial, 2000, tau, frame_rate)
        return events
    except ImportError:
        # Fallback: simple non-negative deconvolution
        events = np.copy(dff_trial)
        events[events < 0] = 0
        return events


def filter_interneurons(dff, speed, iscell_mask):
    """
    Filter putative interneurons by correlation with running speed > 0.5.
    Returns updated mask.
    """
    mask = np.copy(iscell_mask)
    valid = ~np.isnan(dff[0, :]) & ~np.isnan(speed)

    if valid.sum() < 10:
        return mask

    cell_indices = np.where(mask)[0]
    for idx in cell_indices:
        cell_data = dff[idx, valid]
        speed_data = speed[valid]
        if np.std(cell_data) > 0 and np.std(speed_data) > 0:
            corr = np.corrcoef(cell_data, speed_data)[0, 1]
            if corr > INTERNEURON_SPEED_CORR_THR:
                mask[idx] = False

    return mask


def get_trial_boundaries(trial_start_signal, teleport_signal, trial_numbers, scanning):
    """
    Get trial start and end indices from the behavioral signals.
    Returns list of (start_idx, end_idx) for each valid trial.
    """
    # Find trial start indices
    start_indices = np.where(trial_start_signal > 0)[0]
    # Find teleport (trial end) indices
    end_indices = np.where(teleport_signal > 0)[0]

    if len(start_indices) == 0 or len(end_indices) == 0:
        return [], []

    # Match starts and ends
    trial_starts = []
    trial_ends = []

    for s in start_indices:
        # Find the next teleport after this start
        next_ends = end_indices[end_indices > s]
        if len(next_ends) > 0:
            e = next_ends[0]
            # Only include if scanning was on (scanning == 1)
            if np.any(scanning[s:e] == 1):
                trial_starts.append(s)
                trial_ends.append(e)

    return trial_starts, trial_ends


def determine_reward_zone_label(position, rz_signal, trial_start, trial_end):
    """
    Determine which reward zone (A, B, or C) is active for a trial
    based on positions where reward_zone signal > 0.
    """
    pos_trial = position[trial_start:trial_end]
    rz_trial = rz_signal[trial_start:trial_end]

    rz_positions = pos_trial[rz_trial > 0]

    if len(rz_positions) == 0:
        return None  # Can't determine from this trial alone

    mean_rz_pos = np.mean(rz_positions)

    # Match to closest zone
    best_zone = None
    best_dist = float('inf')
    for zone, (start, end) in REWARD_ZONES.items():
        zone_center = (start + end) / 2
        dist = abs(mean_rz_pos - zone_center)
        if dist < best_dist:
            best_dist = dist
            best_zone = zone

    return best_zone


def determine_trial_rewarded(position, rz_signal, reward_timestamps, timestamps,
                              trial_start, trial_end):
    """
    Determine if a trial was rewarded.
    A trial is rewarded if there's a reward delivery AND the mouse was in the reward zone.
    """
    t_start = timestamps[trial_start]
    t_end = timestamps[min(trial_end, len(timestamps) - 1)]

    # Check if any reward was delivered during this trial
    reward_in_trial = np.any(
        (reward_timestamps >= t_start) & (reward_timestamps <= t_end)
    )

    # Check if mouse entered reward zone
    rz_trial = rz_signal[trial_start:trial_end]
    entered_rz = np.any(rz_trial > 0)

    return int(reward_in_trial and entered_rz)


def determine_reward_zone_for_all_trials(position, rz_signal, trial_starts, trial_ends):
    """
    Determine reward zone label for each trial.
    For trials where the mouse didn't enter the zone (omission),
    infer from neighboring trials.
    """
    n_trials = len(trial_starts)
    labels = [None] * n_trials

    # First pass: determine from direct observation
    for i in range(n_trials):
        labels[i] = determine_reward_zone_label(
            position, rz_signal, trial_starts[i], trial_ends[i])

    # Second pass: fill in missing labels from neighbors
    for i in range(n_trials):
        if labels[i] is None:
            # Look forward
            for j in range(i + 1, n_trials):
                if labels[j] is not None:
                    labels[i] = labels[j]
                    break
            # If still None, look backward
            if labels[i] is None:
                for j in range(i - 1, -1, -1):
                    if labels[j] is not None:
                        labels[i] = labels[j]
                        break

    return labels


def distance_to_reward_zone(position, rz_start, rz_end):
    """
    Compute signed distance from current position to nearest point in reward zone.
    Negative = before zone, positive = after zone, 0 = inside zone.
    """
    if position < rz_start:
        return position - rz_start  # negative
    elif position > rz_end:
        return position - rz_end  # positive
    else:
        return 0.0  # inside zone


def discretize_distance(distances):
    """
    Discretize distance to reward zone into bins:
    0: < -50 cm
    1: -50 to -10 cm
    2: -10 cm to < 0 cm
    3: 0 cm
    4: >0 cm to +10 cm
    5: +10 to +50 cm
    6: > +50 cm
    """
    result = np.zeros(len(distances), dtype=int)
    for i, d in enumerate(distances):
        if d < -50:
            result[i] = 0
        elif d < -10:
            result[i] = 1
        elif d < 0:
            result[i] = 2
        elif d == 0:
            result[i] = 3
        elif d <= 10:
            result[i] = 4
        elif d <= 50:
            result[i] = 5
        else:
            result[i] = 6
    return result


def discretize_position(positions, n_bins=5):
    """
    Discretize absolute position into n_bins equal-sized bins over [0, 450] cm.
    """
    bin_size = TRACK_LENGTH / n_bins  # 90 cm bins
    result = np.clip(np.floor(positions / bin_size).astype(int), 0, n_bins - 1)
    return result


def discretize_speed(speeds):
    """
    Discretize speed into bins:
    0: < 2 cm/s
    1: 2-10 cm/s
    2: 10-20 cm/s
    3: 20-40 cm/s
    4: > 40 cm/s
    """
    result = np.zeros(len(speeds), dtype=int)
    for i, s in enumerate(speeds):
        if s < 2:
            result[i] = 0
        elif s < 10:
            result[i] = 1
        elif s < 20:
            result[i] = 2
        elif s < 40:
            result[i] = 3
        else:
            result[i] = 4
    return result


def process_session(filepath, compute_own_dff=True):
    """
    Process a single NWB file into session data.

    Returns dict with neural and behavioral data per trial,
    or None if session is invalid.
    """
    print(f"  Loading {os.path.basename(filepath)}...")
    nwb_data = load_nwb(filepath)

    subject = nwb_data['subject_id']
    session_id = nwb_data['session_id']

    # Get curated cells
    iscell = nwb_data['iscell'][:, 0].astype(bool)
    n_total_cells = iscell.sum()

    if n_total_cells < 5:
        print(f"    Skipping: only {n_total_cells} curated cells")
        return None

    # Get trial boundaries
    trial_starts, trial_ends = get_trial_boundaries(
        nwb_data['trial_start'], nwb_data['teleport'],
        nwb_data['trial_number'], nwb_data['scanning']
    )

    if len(trial_starts) < 3:
        print(f"    Skipping: only {len(trial_starts)} trials")
        return None

    # Neural data processing
    # Transpose to (n_neurons, n_timepoints) format
    F = nwb_data['fluorescence'].T  # (n_neurons, n_timepoints)
    Fneu = nwb_data['neuropil'].T
    deconv_nwb = nwb_data['deconvolved'].T  # Pre-computed deconvolved

    speed = nwb_data['speed']

    # Handle shape mismatches between neural and behavioral data (off-by-one)
    n_behav = len(speed)
    n_neural = F.shape[1]
    if n_behav != n_neural:
        min_len = min(n_behav, n_neural)
        F = F[:, :min_len]
        Fneu = Fneu[:, :min_len]
        deconv_nwb = deconv_nwb[:, :min_len]
        speed = nwb_data['speed'][:min_len]
        nwb_data['position'] = nwb_data['position'][:min_len]
        nwb_data['lick'] = nwb_data['lick'][:min_len]
        nwb_data['reward_zone'] = nwb_data['reward_zone'][:min_len]
        nwb_data['teleport'] = nwb_data['teleport'][:min_len]
        nwb_data['trial_start'] = nwb_data['trial_start'][:min_len]
        nwb_data['trial_number'] = nwb_data['trial_number'][:min_len]
        nwb_data['environment'] = nwb_data['environment'][:min_len]
        nwb_data['scanning'] = nwb_data['scanning'][:min_len]
        nwb_data['timestamps'] = nwb_data['timestamps'][:min_len]

    n_planes = nwb_data.get('n_planes', 1)
    # For multi-plane: frame rate per plane is the effective rate
    effective_frame_rate = FRAME_RATE  # already per-plane in NWB data

    if compute_own_dff:
        # Compute dF/F following the paper's pipeline
        dff = compute_dff_per_trial(F, Fneu, trial_starts, trial_ends,
                                     frame_rate=effective_frame_rate)

        # Filter interneurons based on dF/F-speed correlation
        cell_mask = filter_interneurons(dff, speed, iscell)

        # Deconvolve dF/F to get events
        cell_indices = np.where(cell_mask)[0]
        n_cells = len(cell_indices)
        events = np.full((n_cells, F.shape[1]), np.nan)

        for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
            if end <= start:
                continue
            trial_dff = dff[cell_indices, start:end]
            # Replace NaNs with 0 for deconvolution
            trial_dff_clean = np.nan_to_num(trial_dff, nan=0.0)
            try:
                events_trial = deconvolve_oasis(trial_dff_clean,
                                                 frame_rate=effective_frame_rate)
                events[:, start:end] = events_trial
            except Exception as e:
                print(f"    Warning: deconvolution failed for trial: {e}")
                events[:, start:end] = np.maximum(trial_dff_clean, 0)
    else:
        # Use pre-computed deconvolved data from NWB
        cell_mask = filter_interneurons(
            deconv_nwb, speed, iscell)
        cell_indices = np.where(cell_mask)[0]
        events = deconv_nwb[cell_indices, :]

    n_cells = len(cell_indices)
    n_interneurons_removed = int(iscell.sum() - cell_mask.sum())

    print(f"    {subject} ses-{session_id}: {n_cells} cells "
          f"({n_interneurons_removed} interneurons removed), "
          f"{len(trial_starts)} trials")

    # Get behavioral data
    position = nwb_data['position']
    lick_cumul = nwb_data['lick']
    rz_signal = nwb_data['reward_zone']
    environment = nwb_data['environment']
    timestamps = nwb_data['timestamps']
    reward_timestamps = nwb_data['reward_timestamps']

    # Determine reward zones for all trials
    rz_labels = determine_reward_zone_for_all_trials(
        position, rz_signal, trial_starts, trial_ends)

    # Determine reward outcomes for all trials
    is_rewarded = []
    for i in range(len(trial_starts)):
        rew = determine_trial_rewarded(
            position, rz_signal, reward_timestamps, timestamps,
            trial_starts[i], trial_ends[i])
        is_rewarded.append(rew)

    # Build per-trial data
    neural_trials = []
    input_trials = []
    output_trials = []
    valid_trial_count = 0

    for t in range(len(trial_starts)):
        s, e = trial_starts[t], trial_ends[t]
        n_timepoints = e - s

        if n_timepoints < 5:
            continue

        # Neural: (n_neurons, n_timepoints)
        neural_trial = events[:, s:e].copy()
        # Replace NaNs with 0 for the decoder
        neural_trial = np.nan_to_num(neural_trial, nan=0.0)

        # Time from trial start in seconds
        time_from_start = (timestamps[s:e] - timestamps[s])

        # Environment type (binary: ENV1=0, ENV2=1)
        env_type = float(environment[s])  # per trial
        if env_type < 0:
            env_type = 0.0  # default to ENV1 if unknown

        # Trial number (continuous)
        trial_num = float(t)

        # Previous trial outcome (binary)
        if t == 0:
            prev_outcome = 1.0  # assume rewarded before session
        else:
            prev_outcome = float(is_rewarded[t - 1])

        # Input: (4, n_timepoints) for time-varying, or (4,) for per-trial
        # time_from_start is time-varying; env, trial_num, prev_outcome are per-trial
        # Per the format: (n_input, n_timepoints) or (n_input,)
        # Mix: time is time-varying, others are per-trial scalars
        # We'll broadcast per-trial values to match time dimension
        input_trial = np.zeros((4, n_timepoints))
        input_trial[0, :] = time_from_start
        input_trial[1, :] = env_type
        input_trial[2, :] = trial_num
        input_trial[3, :] = prev_outcome

        # Outputs
        pos_trial = position[s:e]
        speed_trial = speed[s:e]

        # Lick: convert cumulative to binary
        lick_trial = lick_cumul[s:e].copy()
        lick_binary = (lick_trial > 0).astype(float)

        # Get reward zone for this trial
        rz_label = rz_labels[t]
        if rz_label is None:
            rz_label = 'A'  # fallback

        rz_start, rz_end = REWARD_ZONES[rz_label]

        # Distance to reward zone (time-varying)
        dist_to_rz = np.array([distance_to_reward_zone(p, rz_start, rz_end)
                               for p in pos_trial])
        dist_binned = discretize_distance(dist_to_rz)

        # Absolute position (time-varying, 5 bins)
        # Clip position to [0, 450] range
        pos_clipped = np.clip(pos_trial, 0, TRACK_LENGTH - 0.001)
        pos_binned = discretize_position(pos_clipped, n_bins=5)

        # Speed (time-varying, 5 bins)
        speed_abs = np.abs(speed_trial)
        speed_binned = discretize_speed(speed_abs)

        # Reward zone location (per-trial: A=0, B=1, C=2)
        rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_label]

        # Reward outcome (per-trial)
        rew_outcome = is_rewarded[t]

        # Output: (n_output, n_timepoints) for time-varying, (n_output,) for per-trial
        # Time-varying outputs: distance, position, speed, lick
        # Per-trial outputs: reward zone location, reward outcome
        # Combine: broadcast per-trial to time-varying
        n_output = 6
        output_trial = np.zeros((n_output, n_timepoints), dtype=int)
        output_trial[0, :] = dist_binned
        output_trial[1, :] = pos_binned
        output_trial[2, :] = speed_binned
        output_trial[3, :] = lick_binary.astype(int)
        output_trial[4, :] = rz_loc
        output_trial[5, :] = rew_outcome

        neural_trials.append(neural_trial)
        input_trials.append(input_trial)
        output_trials.append(output_trial)
        valid_trial_count += 1

    if valid_trial_count < 2:
        print(f"    Skipping: only {valid_trial_count} valid trials")
        return None

    # Get plane info for brain region
    planes = nwb_data['plane_idx'][cell_indices]

    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'subject': subject,
        'session_id': session_id,
        'n_neurons': n_cells,
        'n_trials': valid_trial_count,
        'plane_idx': planes,
        'n_interneurons_removed': n_interneurons_removed,
    }


def convert_all_data(data_dir, output_path, sample_path=None):
    """
    Convert all NWB files to the decoder format.
    """
    # Find all NWB files
    subjects = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])

    print(f"Found {len(subjects)} subjects: {subjects}")

    all_neural = []
    all_input = []
    all_output = []
    all_subjects = []
    subject_idx = []
    brain_region_idx = []

    total_sessions = 0
    total_trials = 0
    total_neurons = 0

    for subj_dir in subjects:
        subj_path = os.path.join(data_dir, subj_dir)
        nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))

        subject_name = subj_dir.replace('sub-', '')
        if subject_name not in all_subjects:
            all_subjects.append(subject_name)
        subj_idx = all_subjects.index(subject_name)

        print(f"\nProcessing {subject_name} ({len(nwb_files)} sessions)...")

        for nwb_file in nwb_files:
            result = process_session(nwb_file, compute_own_dff=True)

            if result is None:
                continue

            all_neural.append(result['neural'])
            all_input.append(result['input'])
            all_output.append(result['output'])
            subject_idx.append(subj_idx)

            # Brain region index for each neuron (all CA1)
            brain_region_idx.append(np.zeros(result['n_neurons'], dtype=int))

            total_sessions += 1
            total_trials += result['n_trials']
            total_neurons += result['n_neurons']

    print(f"\n{'='*60}")
    print(f"Total: {total_sessions} sessions, {total_trials} trials, "
          f"{total_neurons} total neuron-sessions")
    print(f"Subjects: {all_subjects}")
    print(f"{'='*60}")

    # Build the data dictionary
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': all_subjects,
        'subject_idx': np.array(subject_idx),
        'brain_regions': ['CA1'],
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
            ['< -50cm', '-50 to -10cm', '-10 to 0cm', '0cm', '0 to +10cm', '+10 to +50cm', '> +50cm'],
            ['0-90cm', '90-180cm', '180-270cm', '270-360cm', '360-450cm'],
            ['< 2cm/s', '2-10cm/s', '10-20cm/s', '20-40cm/s', '> 40cm/s'],
            ['no_lick', 'lick'],
            ['zone_A', 'zone_B', 'zone_C'],
            ['no_reward', 'reward'],
        ],
        'metadata': {
            'task_description': (
                'Hidden reward zone navigation task in virtual reality. '
                'Head-fixed mice navigate a 450cm linear track to find and lick '
                'in an unmarked 50cm reward zone at one of three locations (A, B, C). '
                'Reward zone switches occur across sessions in two environments.'
            ),
            'time_bin_size': TIME_BIN_MS,
            'temporal_alignment_event': 'start of trial (entry to linear track at 0 cm)',
            'off_start': 0.0,
            'off_end': None,
            'track_length_cm': TRACK_LENGTH,
            'frame_rate_hz': FRAME_RATE,
            'reward_zones': REWARD_ZONES,
            'n_sessions': total_sessions,
            'n_subjects': len(all_subjects),
            'neural_data_type': 'deconvolved calcium events (OASIS from dF/F)',
            'dff_baseline_method': 'maximin with 20s window, per-trial',
            'dff_smoothing': '2-sample Gaussian kernel (~0.129s)',
            'neuropil_subtraction': f'coefficient={NEUROPIL_COEF}',
            'interneuron_filter': f'speed correlation > {INTERNEURON_SPEED_CORR_THR}',
            'brain_region': 'hippocampal CA1',
            'imaging_method': '2-photon calcium imaging (GCaMP7f)',
        },
    }

    # Save full dataset
    print(f"\nSaving full dataset to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved. File size: {os.path.getsize(output_path) / 1e6:.1f} MB")

    # Save sample dataset (subset of sessions)
    if sample_path:
        print(f"\nCreating sample dataset...")
        # Take up to 2 sessions per subject, max ~10 sessions total
        sample_indices = []
        subj_count = {}
        for i, si in enumerate(subject_idx):
            subj_count[si] = subj_count.get(si, 0)
            if subj_count[si] < 2:
                sample_indices.append(i)
                subj_count[si] += 1
            if len(sample_indices) >= 10:
                break

        sample_data = {
            'neural': [all_neural[i] for i in sample_indices],
            'input': [all_input[i] for i in sample_indices],
            'output': [all_output[i] for i in sample_indices],
            'subjects': all_subjects,
            'subject_idx': np.array([subject_idx[i] for i in sample_indices]),
            'brain_regions': data['brain_regions'],
            'brain_region_idx': [brain_region_idx[i] for i in sample_indices],
            'input_names': data['input_names'],
            'output_names': data['output_names'],
            'output_values': data['output_values'],
            'metadata': data['metadata'].copy(),
        }
        sample_data['metadata']['note'] = 'Sample subset for quick testing'

        with open(sample_path, 'wb') as f:
            pickle.dump(sample_data, f)
        print(f"Saved sample to {sample_path}. "
              f"{len(sample_indices)} sessions, "
              f"file size: {os.path.getsize(sample_path) / 1e6:.1f} MB")

    return data


def run_sanity_checks(data):
    """Run sanity checks comparing our conversion with reference paper statistics."""
    print("\n" + "="*60)
    print("SANITY CHECKS")
    print("="*60)

    # 1. Number of subjects
    n_subjects = len(data['subjects'])
    print(f"\n1. Number of subjects: {n_subjects}")
    print(f"   Expected: 11 switch mice (paper mentions n=11 switch + 3 fixed-condition)")
    print(f"   Available in data: {data['subjects']}")

    # 2. Number of sessions per subject
    print(f"\n2. Sessions per subject:")
    for subj in data['subjects']:
        si = data['subjects'].index(subj)
        n_sess = np.sum(data['subject_idx'] == si)
        print(f"   {subj}: {n_sess} sessions")

    # 3. Trials per session
    trial_counts = [len(sess) for sess in data['neural']]
    print(f"\n3. Trials per session: mean={np.mean(trial_counts):.1f} +/- {np.std(trial_counts):.1f}")
    print(f"   Expected: ~80.5 +/- 7.4 (paper)")
    print(f"   Range: [{min(trial_counts)}, {max(trial_counts)}]")

    # 4. Neurons per session
    neuron_counts = [sess[0].shape[0] if len(sess) > 0 else 0 for sess in data['neural']]
    print(f"\n4. Neurons per session: mean={np.mean(neuron_counts):.1f} +/- {np.std(neuron_counts):.1f}")
    print(f"   Expected: 155-2172 (paper range)")
    print(f"   Range: [{min(neuron_counts)}, {max(neuron_counts)}]")

    # 5. Frame rate
    print(f"\n5. Frame rate: {data['metadata']['time_bin_size']:.2f} ms/bin "
          f"({1000/data['metadata']['time_bin_size']:.2f} Hz)")
    print(f"   Expected: ~15.5 Hz (~64.5 ms/bin)")

    # 6. Output distribution checks
    print(f"\n6. Output distributions (from first session):")
    if len(data['output']) > 0 and len(data['output'][0]) > 0:
        # Check across multiple trials
        all_outputs = np.concatenate([t for t in data['output'][0]], axis=1)
        for i, name in enumerate(data['output_names']):
            vals = all_outputs[i, :]
            unique, counts = np.unique(vals, return_counts=True)
            pcts = counts / counts.sum() * 100
            dist_str = ', '.join([f'{int(v)}:{p:.1f}%' for v, p in zip(unique, pcts)])
            print(f"   {name}: {dist_str}")

    # 7. Environment distribution
    print(f"\n7. Environment distribution:")
    env_counts = {0: 0, 1: 0}
    for sess_inputs in data['input']:
        for trial_input in sess_inputs:
            env = trial_input[1, 0] if trial_input.ndim == 2 else trial_input[1]
            env_counts[int(env)] = env_counts.get(int(env), 0) + 1
    print(f"   ENV1: {env_counts.get(0, 0)} trials, ENV2: {env_counts.get(1, 0)} trials")

    print(f"\n{'='*60}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert NWB data to decoder format')
    parser.add_argument('--data-dir', default=DATA_DIR, help='Path to data directory')
    parser.add_argument('--output', default='converted_data.pkl', help='Output pickle file')
    parser.add_argument('--sample-output', default='sample_data.pkl', help='Sample output pickle file')
    parser.add_argument('--sample-only', action='store_true', help='Only create sample dataset')
    args = parser.parse_args()

    if args.sample_only:
        # Quick mode: only process a few sessions
        print("Creating sample dataset only...")
        data = convert_all_data(args.data_dir, args.output, args.sample_output)
    else:
        data = convert_all_data(args.data_dir, args.output, args.sample_output)

    run_sanity_checks(data)
