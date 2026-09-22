"""
Convert NWB data from Sosa, Plitt & Giocomo (2025) into the standardized decoder format.

Data: Two-photon calcium imaging of hippocampal CA1 neurons in head-fixed mice
performing a virtual reality navigation task with hidden reward zones.

Processing decisions (matching reference paper/code):
- Neural data: Use deconvolved calcium activity from NWB (OASIS deconvolution of
  custom dF/F with maximin baseline, neuropil subtraction coef=0.7).
- Cell filtering: Suite2p iscell classification, then exclude putative interneurons
  (Pearson r > 0.5 between dF/F and running speed, per paper methods).
- Trial segmentation: Each trial runs from trial_start to teleport event.
  Only include on-track timepoints (position >= 0 cm, <= 450 cm).
- Lick correction: Trials with >35% of frames having lick count > 2 are flagged
  as sensor errors; lick set to 0 for those trials (matching code threshold of 0.35).
- Reward zone identification: Parsed from NWB session identifier (scene name).
  Switch sessions change reward zone at trial 30.
- Time bin size: Native imaging frame rate (~15.5 Hz, ~64.5 ms per bin).
- No speed threshold applied (decoder predicts at all timepoints including stationary).
"""

import h5py
import numpy as np
import pickle
import os
import re
from scipy.ndimage import gaussian_filter1d, minimum_filter1d, maximum_filter1d
from scipy.stats import pearsonr

# ============================================================
# Constants
# ============================================================
TRACK_LENGTH = 450.0  # cm
REWARD_ZONES = {
    'A': (80.0, 130.0),
    'B': (200.0, 250.0),
    'C': (320.0, 370.0),
}
REWARD_ZONE_LABELS = {'A': 0, 'B': 1, 'C': 2}
CHANGE_TRIAL = 30  # Trial index where reward zone switches
INTERNEURON_SPEED_CORR_THR = 0.5
NEUROPIL_COEF = 0.7
LICK_CORRECTION_THR = 0.35  # Fraction of frames with lick > 2 to flag bad sensor
BASELINE_SMOOTH_SIGMA = 15  # Gaussian sigma for baseline smoothing (frames)
BASELINE_WINDOW = 300  # Window for min/max filter (frames, ~20s at 15.5 Hz)
DFF_SMOOTH_SIGMA = 2  # Gaussian sigma for dF/F smoothing (frames, ~0.129s)

DATA_DIR = '/app/data'


# ============================================================
# Scene parsing
# ============================================================
def parse_scene_from_identifier(identifier):
    """Parse the scene name from NWB identifier path."""
    scene = identifier.strip('/').split('/')[-1]
    return scene


def get_session_info(scene):
    """
    Parse scene name to extract environment(s) and reward zone(s).

    Returns:
        dict with keys:
            'is_switch': bool
            'env_before': int (0=Env1, 1=Env2)
            'env_after': int (0=Env1, 1=Env2)
            'zone_before': str ('A', 'B', or 'C')
            'zone_after': str ('A', 'B', or 'C')
    """
    env_map = {'Env1': 0, 'Env2': 1}

    # Cross-environment switch: e.g., "Env1_B_to_Env2_C"
    m = re.match(r'(Env[12])_([ABC])_to_(Env[12])_([ABC])', scene)
    if m:
        return {
            'is_switch': True,
            'env_before': env_map[m.group(1)],
            'env_after': env_map[m.group(3)],
            'zone_before': m.group(2),
            'zone_after': m.group(4),
        }

    # Within-environment switch: e.g., "Env1_LocationA_to_B"
    m = re.match(r'(Env[12])_Location([ABC])_to_([ABC])', scene)
    if m:
        env = env_map[m.group(1)]
        return {
            'is_switch': True,
            'env_before': env,
            'env_after': env,
            'zone_before': m.group(2),
            'zone_after': m.group(3),
        }

    # Stable session: e.g., "Env1_LocationA"
    m = re.match(r'(Env[12])_Location([ABC])', scene)
    if m:
        env = env_map[m.group(1)]
        zone = m.group(2)
        return {
            'is_switch': False,
            'env_before': env,
            'env_after': env,
            'zone_before': zone,
            'zone_after': zone,
        }

    # Training sessions - skip these
    return None


def get_per_trial_info(session_info, n_trials):
    """
    Return per-trial reward zone label and environment.

    Returns:
        zones: list of str ('A', 'B', or 'C') per trial
        envs: list of int (0 or 1) per trial
    """
    zones = []
    envs = []
    for t in range(n_trials):
        if session_info['is_switch'] and t >= CHANGE_TRIAL:
            zones.append(session_info['zone_after'])
            envs.append(session_info['env_after'])
        else:
            zones.append(session_info['zone_before'])
            envs.append(session_info['env_before'])
    return zones, envs


# ============================================================
# dF/F computation (for interneuron detection)
# ============================================================
def compute_dff_for_interneuron_detection(fluorescence, neuropil, trial_starts, trial_ends):
    """
    Compute dF/F using the paper's maximin baseline method.
    Used only for interneuron detection (speed correlation).

    Args:
        fluorescence: (n_timepoints, n_cells) raw fluorescence
        neuropil: (n_timepoints, n_cells) neuropil fluorescence
        trial_starts: list of trial start indices
        trial_ends: list of trial end indices

    Returns:
        dff: (n_timepoints, n_cells) dF/F values (NaN outside trials)
    """
    n_timepoints, n_cells = fluorescence.shape

    # Neuropil subtraction
    f_corr = fluorescence - NEUROPIL_COEF * neuropil

    dff = np.full_like(f_corr, np.nan)

    for t_start, t_end in zip(trial_starts, trial_ends):
        if t_end <= t_start:
            continue
        trial_f = f_corr[t_start:t_end, :]

        # Maximin baseline per trial
        # 1. Gaussian smooth
        smoothed = gaussian_filter1d(trial_f, sigma=BASELINE_SMOOTH_SIGMA, axis=0)
        # 2. Minimum filter
        baseline = minimum_filter1d(smoothed, size=BASELINE_WINDOW, axis=0)
        # 3. Maximum filter
        baseline = maximum_filter1d(baseline, size=BASELINE_WINDOW, axis=0)

        # dF/F
        abs_baseline = np.abs(baseline)
        abs_baseline[abs_baseline < 1e-6] = 1e-6  # avoid division by zero
        trial_dff = (trial_f - baseline) / abs_baseline

        # Smooth dF/F with 2-sample Gaussian kernel
        trial_dff = gaussian_filter1d(trial_dff, sigma=DFF_SMOOTH_SIGMA, axis=0)

        dff[t_start:t_end, :] = trial_dff

    return dff


def detect_interneurons(dff, speed, iscell_mask, threshold=INTERNEURON_SPEED_CORR_THR):
    """
    Detect putative interneurons by Pearson correlation between dF/F and speed.

    Args:
        dff: (n_timepoints, n_cells) - only cells passing iscell
        speed: (n_timepoints,) running speed
        iscell_mask: boolean mask of which ROIs are cells
        threshold: correlation threshold (default 0.5)

    Returns:
        is_interneuron: boolean array of shape (n_cells_passing_iscell,)
    """
    n_cells = dff.shape[1]
    is_interneuron = np.zeros(n_cells, dtype=bool)

    # Only use valid (non-NaN) timepoints
    valid = ~np.isnan(dff[:, 0]) & ~np.isnan(speed)

    if np.sum(valid) < 10:
        return is_interneuron

    speed_valid = speed[valid]

    for i in range(n_cells):
        cell_dff = dff[valid, i]
        if np.std(cell_dff) < 1e-10 or np.std(speed_valid) < 1e-10:
            continue
        r, _ = pearsonr(cell_dff, speed_valid)
        if r > threshold:
            is_interneuron[i] = True

    return is_interneuron


# ============================================================
# Trial extraction
# ============================================================
def get_trial_boundaries(trial_number, trial_start_signal, teleport_signal):
    """
    Get start and end indices for each trial.

    Returns:
        trial_starts: list of start indices
        trial_ends: list of end indices (exclusive)
        trial_ids: list of trial numbers
    """
    trial_starts = []
    trial_ends = []
    trial_ids = []

    # Find trial start frames
    start_frames = np.where(trial_start_signal > 0.5)[0]
    teleport_frames = np.where(teleport_signal > 0.5)[0]

    for sf in start_frames:
        tid = int(trial_number[sf])
        if tid < 0:
            continue

        # Find the next teleport after this start
        future_teleports = teleport_frames[teleport_frames > sf]
        if len(future_teleports) == 0:
            # No teleport found; use end of data or next trial start
            next_starts = start_frames[start_frames > sf]
            if len(next_starts) > 0:
                ef = next_starts[0]
            else:
                ef = len(trial_number)
        else:
            ef = future_teleports[0] + 1  # include the teleport frame

        trial_starts.append(sf)
        trial_ends.append(ef)
        trial_ids.append(tid)

    return trial_starts, trial_ends, trial_ids


def extract_on_track_indices(position, start_idx, end_idx):
    """
    Get indices within a trial where the mouse is on the track (0 <= pos <= 450).
    """
    trial_pos = position[start_idx:end_idx]
    on_track = (trial_pos >= 0) & (trial_pos <= TRACK_LENGTH + 5)  # small buffer
    indices = np.where(on_track)[0] + start_idx
    return indices


# ============================================================
# Discretization functions
# ============================================================
def discretize_distance_to_reward(position, reward_zone):
    """
    Compute signed distance to reward zone and discretize.

    Args:
        position: array of positions (cm)
        reward_zone: tuple (start, end) in cm

    Returns:
        bins: array of discretized distances (0-6)
    """
    rz_start, rz_end = reward_zone
    distance = np.where(
        position < rz_start,
        position - rz_start,  # negative (approaching)
        np.where(
            position > rz_end,
            position - rz_end,  # positive (past)
            0.0  # inside zone
        )
    )

    bins = np.zeros_like(distance, dtype=np.int64)
    bins[distance < -50] = 0
    bins[(distance >= -50) & (distance < -10)] = 1
    bins[(distance >= -10) & (distance < 0)] = 2
    bins[distance == 0] = 3
    bins[(distance > 0) & (distance <= 10)] = 4
    bins[(distance > 10) & (distance <= 50)] = 5
    bins[distance > 50] = 6

    return bins


def discretize_position(position):
    """Discretize position into 5 equal bins spanning 450 cm."""
    bin_size = TRACK_LENGTH / 5  # 90 cm
    bins = np.clip(np.floor(position / bin_size).astype(np.int64), 0, 4)
    return bins


def discretize_speed(speed):
    """Discretize speed into 5 bins."""
    abs_speed = np.abs(speed)
    bins = np.zeros_like(speed, dtype=np.int64)
    bins[abs_speed < 2] = 0
    bins[(abs_speed >= 2) & (abs_speed < 10)] = 1
    bins[(abs_speed >= 10) & (abs_speed < 20)] = 2
    bins[(abs_speed >= 20) & (abs_speed < 40)] = 3
    bins[abs_speed >= 40] = 4
    return bins


def check_lick_sensor_error(lick_data, threshold=LICK_CORRECTION_THR):
    """
    Check if a trial has a stuck lick sensor.
    Returns True if the trial should have its lick data zeroed.
    """
    if len(lick_data) == 0:
        return False
    frac_high = np.mean(lick_data > 2)
    return frac_high > threshold


# ============================================================
# Main processing
# ============================================================
def load_and_process_session(nwb_path, subject_name):
    """
    Load one NWB file and extract all trial data.

    Returns:
        session_data dict or None if session should be skipped.
    """
    with h5py.File(nwb_path, 'r') as f:
        # Parse scene from identifier
        identifier = f['identifier'][()]
        if isinstance(identifier, bytes):
            identifier = identifier.decode()
        scene = parse_scene_from_identifier(identifier)
        session_info = get_session_info(scene)

        if session_info is None:
            print(f"  Skipping training session: {scene}")
            return None

        # Load behavioral data
        beh = f['processing']['behavior']['BehavioralTimeSeries']
        position = beh['position']['data'][:]
        speed = beh['speed']['data'][:]
        lick = beh['lick']['data'][:]
        trial_number = beh['trial number']['data'][:]
        trial_start_sig = beh['trial_start']['data'][:]
        teleport_sig = beh['teleport']['data'][:]

        # Get timestamps for computing time
        beh_timestamps = beh['position']['timestamps'][:]

        # Load reward event timestamps
        reward_timestamps = beh['Reward']['timestamps'][:]

        # Load neural data - handle single-plane and multi-plane animals
        ophys = f['processing']['ophys']
        iscell_all = ophys['ImageSegmentation']['PlaneSegmentation']['iscell'][:, 0]

        # Check for multiple planes
        has_plane1 = 'plane1' in ophys['Deconvolved']

        if has_plane1:
            # Multi-plane animal (m17, m18): pool ROIs from both planes
            plane_idx = ophys['ImageSegmentation']['PlaneSegmentation']['planeIdx'][:]
            deconv0 = ophys['Deconvolved']['plane0']['data'][:]
            deconv1 = ophys['Deconvolved']['plane1']['data'][:]
            fluor0 = ophys['Fluorescence']['plane0']['data'][:]
            fluor1 = ophys['Fluorescence']['plane1']['data'][:]
            neuro0 = ophys['Neuropil']['plane0']['data'][:]
            neuro1 = ophys['Neuropil']['plane1']['data'][:]

            # Concatenate planes: plane0 ROIs first, then plane1
            deconvolved = np.concatenate([deconv0, deconv1], axis=1)
            fluorescence = np.concatenate([fluor0, fluor1], axis=1)
            neuropil_data = np.concatenate([neuro0, neuro1], axis=1)

            # iscell is ordered plane0 then plane1 (matching planeIdx order)
            iscell = iscell_all
        else:
            deconvolved = ophys['Deconvolved']['plane0']['data'][:]
            fluorescence = ophys['Fluorescence']['plane0']['data'][:]
            neuropil_data = ophys['Neuropil']['plane0']['data'][:]
            iscell = iscell_all

    # Ensure all arrays have the same number of timepoints
    n_time = min(
        deconvolved.shape[0], fluorescence.shape[0], neuropil_data.shape[0],
        len(position), len(speed), len(lick), len(trial_number),
        len(trial_start_sig), len(teleport_sig), len(beh_timestamps)
    )
    deconvolved = deconvolved[:n_time]
    fluorescence = fluorescence[:n_time]
    neuropil_data = neuropil_data[:n_time]
    position = position[:n_time]
    speed = speed[:n_time]
    lick = lick[:n_time]
    trial_number = trial_number[:n_time]
    trial_start_sig = trial_start_sig[:n_time]
    teleport_sig = teleport_sig[:n_time]
    beh_timestamps = beh_timestamps[:n_time]

    # Apply iscell filter
    cell_mask = iscell == 1
    n_cells_raw = int(np.sum(cell_mask))
    deconvolved_cells = deconvolved[:, cell_mask]
    fluorescence_cells = fluorescence[:, cell_mask]
    neuropil_cells = neuropil_data[:, cell_mask]

    # Get trial boundaries
    trial_starts, trial_ends, trial_ids = get_trial_boundaries(
        trial_number, trial_start_sig, teleport_sig
    )
    n_trials = len(trial_starts)

    if n_trials < 2:
        print(f"  Skipping session with < 2 trials: {scene}")
        return None

    # Compute dF/F for interneuron detection
    dff = compute_dff_for_interneuron_detection(
        fluorescence_cells, neuropil_cells, trial_starts, trial_ends
    )

    # Detect interneurons
    is_interneuron = detect_interneurons(dff, speed, cell_mask)
    n_interneurons = int(np.sum(is_interneuron))

    # Final cell selection: cells passing iscell AND not interneurons
    good_cells = ~is_interneuron
    n_good_cells = int(np.sum(good_cells))

    if n_good_cells < 1:
        print(f"  Skipping session with 0 good cells: {scene}")
        return None

    neural_data = deconvolved_cells[:, good_cells]  # (n_timepoints, n_good_cells)

    # Get per-trial reward zone and environment
    zones, envs = get_per_trial_info(session_info, n_trials)

    # Determine reward delivery per trial
    is_rewarded = np.zeros(n_trials, dtype=int)
    for i, (ts, te) in enumerate(zip(trial_starts, trial_ends)):
        t_start_time = beh_timestamps[ts]
        t_end_time = beh_timestamps[min(te - 1, len(beh_timestamps) - 1)]
        # Check if any reward was delivered during this trial
        if np.any((reward_timestamps >= t_start_time) & (reward_timestamps <= t_end_time)):
            is_rewarded[i] = 1

    # Process each trial
    neural_trials = []
    input_trials = []
    output_trials = []
    valid_trial_indices = []

    for i in range(n_trials):
        ts = trial_starts[i]
        te = trial_ends[i]

        # Get on-track indices
        on_track_idx = extract_on_track_indices(position, ts, te)

        if len(on_track_idx) < 5:  # Skip very short trials
            continue

        # Neural data for this trial: (n_neurons, n_timepoints)
        trial_neural = neural_data[on_track_idx, :].T  # (n_good_cells, n_timepoints)

        # Time from trial start (seconds)
        trial_times = beh_timestamps[on_track_idx] - beh_timestamps[on_track_idx[0]]

        # Position
        trial_pos = position[on_track_idx]
        trial_pos = np.clip(trial_pos, 0, TRACK_LENGTH)

        # Speed
        trial_speed = speed[on_track_idx]

        # Lick
        trial_lick = lick[on_track_idx]

        # Check lick sensor error
        if check_lick_sensor_error(trial_lick):
            trial_lick = np.zeros_like(trial_lick)

        # Binarize lick
        trial_lick_binary = (trial_lick > 0).astype(np.int64)

        # Environment
        trial_env = envs[i]

        # Trial number
        trial_num = trial_ids[i]

        # Previous trial outcome
        if i == 0:
            prev_outcome = 0
        else:
            prev_outcome = is_rewarded[i - 1]

        # Reward zone for this trial
        zone_label = zones[i]
        rz = REWARD_ZONES[zone_label]

        # --- Build input array (4, n_timepoints) ---
        n_tp = len(on_track_idx)
        inp = np.zeros((4, n_tp), dtype=np.float32)
        inp[0, :] = trial_times                # time from trial start (s)
        inp[1, :] = float(trial_env)           # environment type
        inp[2, :] = float(trial_num)           # trial number
        inp[3, :] = float(prev_outcome)        # previous trial outcome

        # --- Build output array (6, n_timepoints) ---
        out = np.zeros((6, n_tp), dtype=np.int64)
        out[0, :] = discretize_distance_to_reward(trial_pos, rz)
        out[1, :] = discretize_position(trial_pos)
        out[2, :] = discretize_speed(trial_speed)
        out[3, :] = trial_lick_binary
        out[4, :] = REWARD_ZONE_LABELS[zone_label]  # reward zone location
        out[5, :] = is_rewarded[i]                    # reward outcome

        # Check for NaN/Inf
        if np.any(np.isnan(trial_neural)) or np.any(np.isinf(trial_neural)):
            # Replace NaN/Inf with 0 in neural data
            trial_neural = np.nan_to_num(trial_neural, nan=0.0, posinf=0.0, neginf=0.0)

        neural_trials.append(trial_neural.astype(np.float32))
        input_trials.append(inp)
        output_trials.append(out)
        valid_trial_indices.append(i)

    if len(neural_trials) < 2:
        print(f"  Skipping session with < 2 valid trials: {scene}")
        return None

    print(f"  {subject_name} {scene}: {n_good_cells} cells ({n_interneurons} interneurons removed), "
          f"{len(neural_trials)} trials")

    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'n_cells': n_good_cells,
        'subject': subject_name,
        'scene': scene,
        'session_info': session_info,
    }


def main():
    # Discover all subjects
    subjects_dirs = sorted([
        d for d in os.listdir(DATA_DIR)
        if d.startswith('sub-') and os.path.isdir(os.path.join(DATA_DIR, d))
    ])

    all_neural = []
    all_input = []
    all_output = []
    all_subject_idx = []
    all_brain_region_idx = []
    subjects = []
    subject_to_idx = {}
    session_metadata = []

    for subj_dir in subjects_dirs:
        subject_name = subj_dir.replace('sub-', '')
        subj_path = os.path.join(DATA_DIR, subj_dir)
        nwb_files = sorted([
            f for f in os.listdir(subj_path) if f.endswith('.nwb')
        ])

        print(f"\nProcessing {subject_name} ({len(nwb_files)} sessions)...")

        for nwb_file in nwb_files:
            nwb_path = os.path.join(subj_path, nwb_file)
            result = load_and_process_session(nwb_path, subject_name)

            if result is None:
                continue

            # Register subject
            if subject_name not in subject_to_idx:
                subject_to_idx[subject_name] = len(subjects)
                subjects.append(subject_name)

            all_neural.append(result['neural'])
            all_input.append(result['input'])
            all_output.append(result['output'])
            all_subject_idx.append(subject_to_idx[subject_name])
            # All neurons are from CA1
            all_brain_region_idx.append(np.zeros(result['n_cells'], dtype=np.int64))
            session_metadata.append({
                'subject': subject_name,
                'scene': result['scene'],
                'n_cells': result['n_cells'],
                'n_trials': len(result['neural']),
            })

    # Build output dictionary
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,

        'subjects': subjects,
        'subject_idx': np.array(all_subject_idx, dtype=np.int64),

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
            'position',
            'speed',
            'lick',
            'reward_zone_location',
            'reward_outcome',
        ],

        'output_values': [
            # distance_to_reward_zone (7 bins)
            ['< -50 cm', '-50 to -10 cm', '-10 to 0 cm', '0 cm (in zone)',
             '0 to +10 cm', '+10 to +50 cm', '> +50 cm'],
            # position (5 bins)
            ['0-90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '360-450 cm'],
            # speed (5 bins)
            ['< 2 cm/s', '2-10 cm/s', '10-20 cm/s', '20-40 cm/s', '> 40 cm/s'],
            # lick (2 bins)
            ['no lick', 'lick'],
            # reward_zone_location (3 bins)
            ['A', 'B', 'C'],
            # reward_outcome (2 bins)
            ['no reward', 'reward'],
        ],

        'metadata': {
            'task_description': (
                'Hidden reward zone navigation task in virtual reality. '
                'Head-fixed mice run on a 450 cm linear track with a hidden 50 cm reward zone '
                'at one of three possible locations (A: 80-130 cm, B: 200-250 cm, C: 320-370 cm). '
                'Reward zone switches mid-session on switch days (after trial 30). '
                'Two distinct virtual environments used across 14 days.'
            ),
            'time_bin_size': None,  # will be filled with actual value
            'temporal_alignment_event': 'Start of trial (mouse enters track at position 0 cm)',
            'off_start': 0.0,  # trial starts at alignment event
            'off_end': None,   # variable trial duration
            'track_length_cm': 450.0,
            'reward_zones_cm': {'A': [80, 130], 'B': [200, 250], 'C': [320, 370]},
            'imaging_rate_hz': 15.5,
            'brain_region': 'hippocampus CA1',
            'species': 'Mus musculus',
            'indicator': 'GCaMP7f',
            'neural_signal': 'deconvolved calcium activity (OASIS)',
            'switch_trial': CHANGE_TRIAL,
            'session_info': session_metadata,
        },
    }

    # Compute actual time bin size from data
    all_dt = []
    for sess_inputs in data['input']:
        for trial_inp in sess_inputs:
            if trial_inp.ndim == 2 and trial_inp.shape[1] > 1:
                dt = np.diff(trial_inp[0, :])
                all_dt.extend(dt.tolist())
    if all_dt:
        median_dt_ms = np.median(all_dt) * 1000  # convert to ms
        data['metadata']['time_bin_size'] = float(median_dt_ms)

    # Save
    output_path = '/app/converted_data.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"\nSaved converted data to {output_path}")
    print(f"Total sessions: {len(all_neural)}")
    print(f"Total subjects: {len(subjects)}")
    print(f"Total trials: {sum(len(s) for s in all_neural)}")


if __name__ == '__main__':
    main()
