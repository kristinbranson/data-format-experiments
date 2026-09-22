"""
Convert NWB data from the reward-relative hippocampal coding paper into
the standard decoder format.

Processing pipeline follows the paper's methods:
1. Neural data: Neuropil subtraction (coef=0.7), maximin baseline per trial,
   dF/F, 2-sample Gaussian smoothing, OASIS deconvolution (tau=0.7).
2. Neuron filtering: Suite2p iscell, then exclude putative interneurons
   (Pearson r > 0.5 between dF/F and running speed).
3. Trials: Defined by trial_start and teleport markers. Teleport periods excluded.
4. Temporal alignment: Start of each trial (time 0 = first frame of trial).
5. All sessions from all 11 switch-task mice included (152 sessions total).
"""

import numpy as np
import h5py
import os
import pickle
from scipy.ndimage import minimum_filter1d, maximum_filter1d, gaussian_filter1d
from scipy.stats import pearsonr
from suite2p.extraction import dcnv

# ──────────────────────────────────────────────────────────────────────────────
# Constants from the paper
# ──────────────────────────────────────────────────────────────────────────────
TRACK_LENGTH = 450  # cm
NEU_COEF = 0.7
BASELINE_WINDOW = 300  # frames (~20 s at 15.5 Hz)
BASELINE_SMOOTH_SIGMA = 15
DFF_SMOOTH_SIGMA = 2
TAU = 0.7  # calcium decay time constant for OASIS
INTERNEURON_SPEED_CORR_THR = 0.5
LICK_SENSOR_ERROR_THR = 0.5  # paper uses 0.3 (>30% of frames with cumulative lick > 2)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# Reward zone positions from behavior.py: X=[80,130], Y=[200,250], Z=[320,370]
# Labels: A->X, B->Y, C->Z
REWARD_ZONES = {
    'A': (80, 130),
    'B': (200, 250),
    'C': (320, 370),
}

# Session metadata: mapping subject -> list of (session_number, scene, exp_day)
# Derived from sessions_dict.py (GCAMP# -> sub-m#)
SESSIONS_META = {
    'm3': [
        (1, 'Env1_LocationC', 1), (2, 'Env1_LocationC', 2),
        (3, 'Env1_LocationC_to_A', 3), (4, 'Env1_LocationA', 4),
        (5, 'Env1_LocationA_to_B', 5), (6, 'Env1_LocationB', 6),
        (7, 'Env1_LocationB_to_C', 7), (8, 'Env1_C_to_Env2_B', 8),
        (9, 'Env2_LocationB', 9), (10, 'Env2_LocationB_to_A', 10),
        (11, 'Env2_LocationA', 11), (12, 'Env2_LocationA_to_C', 12),
        (13, 'Env2_LocationC', 13), (14, 'Env2_LocationC_to_B', 14),
    ],
    'm4': [
        (1, 'Env1_LocationB', 1), (2, 'Env1_LocationB', 2),
        (3, 'Env1_LocationB_to_A', 3), (4, 'Env1_LocationA', 4),
        (5, 'Env1_LocationA_to_C', 5), (6, 'Env1_LocationC', 6),
        (7, 'Env1_LocationC_to_B', 7), (8, 'Env1_B_to_Env2_C', 8),
        (9, 'Env2_LocationC', 9), (10, 'Env2_LocationC_to_A', 10),
        (11, 'Env2_LocationA', 11), (12, 'Env2_LocationA_to_B', 12),
        (13, 'Env2_LocationB', 13), (14, 'Env2_LocationB_to_C', 14),
    ],
    'm7': [
        (1, 'Env1_LocationA', 1), (2, 'Env1_LocationA', 2),
        (3, 'Env1_LocationA_to_C', 3), (4, 'Env1_LocationC', 4),
        (5, 'Env1_LocationC_to_B', 5), (6, 'Env1_LocationB', 6),
        (7, 'Env1_LocationB_to_A', 7), (8, 'Env1_A_to_Env2_B', 8),
        (9, 'Env2_LocationB', 9), (10, 'Env2_LocationB_to_C', 10),
        (11, 'Env2_LocationC', 11), (12, 'Env2_LocationC_to_A', 12),
        (13, 'Env2_LocationA', 13), (14, 'Env2_LocationA_to_B', 14),
    ],
    'm11': [
        (3, 'Env1_LocationB_to_A', 3), (4, 'Env1_LocationA', 4),
        (5, 'Env1_LocationA_to_C', 5), (6, 'Env1_LocationC', 6),
        (7, 'Env1_LocationC_to_B', 7), (8, 'Env1_B_to_Env2_C', 8),
        (9, 'Env2_LocationC', 9), (10, 'Env2_LocationC_to_A', 10),
        (11, 'Env2_LocationA', 11), (12, 'Env2_LocationA_to_B', 12),
        (13, 'Env2_LocationB', 13), (14, 'Env2_LocationB_to_C', 14),
    ],
    'm12': [
        (1, 'Env1_LocationB', 1), (2, 'Env1_LocationB', 2),
        (3, 'Env1_LocationB_to_C', 3), (4, 'Env1_LocationC', 4),
        (5, 'Env1_LocationC_to_A', 5), (6, 'Env1_LocationA', 6),
        (7, 'Env1_LocationA_to_B', 7), (8, 'Env1_B_to_Env2_A', 8),
        (9, 'Env2_LocationA', 9), (10, 'Env2_LocationA_to_C', 10),
        (11, 'Env2_LocationC', 11), (12, 'Env2_LocationC_to_B', 12),
        (13, 'Env2_LocationB', 13), (14, 'Env2_LocationB_to_A', 14),
    ],
    'm13': [
        (1, 'Env1_LocationC', 1), (2, 'Env1_LocationC', 2),
        (3, 'Env1_LocationC_to_B', 3), (4, 'Env1_LocationB', 4),
        (5, 'Env1_LocationB_to_A', 5), (6, 'Env1_LocationA', 6),
        (7, 'Env1_LocationA_to_C', 7), (8, 'Env1_C_to_Env2_A', 8),
        (9, 'Env2_LocationA', 9), (10, 'Env2_LocationA_to_B', 10),
        (11, 'Env2_LocationB', 11), (12, 'Env2_LocationB_to_C', 12),
        (13, 'Env2_LocationC', 13), (14, 'Env2_LocationC_to_A', 14),
    ],
    'm14': [
        (1, 'Env1_LocationA', 1), (2, 'Env1_LocationA', 2),
        (3, 'Env1_LocationA_to_B', 3), (4, 'Env1_LocationB', 4),
        (5, 'Env1_LocationB_to_C', 5), (6, 'Env1_LocationC', 6),
        (7, 'Env1_LocationC_to_A', 7), (8, 'Env1_A_to_Env2_C', 8),
        (9, 'Env2_LocationC', 9), (10, 'Env2_LocationC_to_B', 10),
        (11, 'Env2_LocationB', 11), (12, 'Env2_LocationB_to_A', 12),
        (13, 'Env2_LocationA', 13), (14, 'Env2_LocationA_to_C', 14),
    ],
    'm15': [
        (1, 'Env1_LocationA', 1), (2, 'Env1_LocationA', 2),
        (3, 'Env1_LocationA_to_C', 3), (4, 'Env1_LocationC', 4),
        (5, 'Env1_LocationC_to_B', 5), (6, 'Env1_LocationB', 6),
        (7, 'Env1_LocationB_to_A', 7), (8, 'Env1_A_to_Env2_B', 8),
        (9, 'Env2_LocationB', 9), (10, 'Env2_LocationB_to_C', 10),
        (11, 'Env2_LocationC', 11), (12, 'Env2_LocationC_to_A', 12),
        (13, 'Env2_LocationA', 13), (14, 'Env2_LocationA_to_B', 14),
    ],
    'm17': [
        (1, 'Env2_LocationB', 1), (2, 'Env2_LocationB', 2),
        (3, 'Env2_LocationB_to_C', 3), (4, 'Env2_LocationC', 4),
        (5, 'Env2_LocationC_to_A', 5), (6, 'Env2_LocationA', 6),
        (7, 'Env2_LocationA_to_B', 7), (8, 'Env2_B_to_Env1_A', 8),
        (9, 'Env1_LocationA', 9), (10, 'Env1_LocationA_to_C', 10),
        (11, 'Env1_LocationC', 11), (12, 'Env1_LocationC_to_B', 12),
        (13, 'Env1_LocationB', 13), (14, 'Env1_LocationB_to_A', 14),
    ],
    'm18': [
        (1, 'Env2_LocationA', 1), (2, 'Env2_LocationA', 2),
        (3, 'Env2_LocationA_to_B', 3), (4, 'Env2_LocationB', 4),
        (5, 'Env2_LocationB_to_C', 5), (6, 'Env2_LocationC', 6),
        (7, 'Env2_LocationC_to_A', 7), (8, 'Env2_A_to_Env1_C', 8),
        (9, 'Env1_LocationC', 9), (10, 'Env1_LocationC_to_B', 10),
        (11, 'Env1_LocationB', 11), (12, 'Env1_LocationB_to_A', 12),
        (13, 'Env1_LocationA', 13), (14, 'Env1_LocationA_to_C', 14),
    ],
    'm19': [
        (1, 'Env1_LocationC', 1), (2, 'Env1_LocationC', 2),
        (3, 'Env1_LocationC_to_B', 3), (4, 'Env1_LocationB', 4),
        (5, 'Env1_LocationB_to_A', 5), (6, 'Env1_LocationA', 6),
        (7, 'Env1_LocationA_to_C', 7), (8, 'Env1_C_to_Env2_A', 8),
        (9, 'Env2_LocationA', 9), (10, 'Env2_LocationA_to_B', 10),
        (11, 'Env2_LocationB', 11), (12, 'Env2_LocationB_to_C', 12),
        (13, 'Env2_LocationC', 13), (14, 'Env2_LocationC_to_A', 14),
    ],
}


def nansmooth(arr, sigma, axis=-1):
    """Gaussian smoothing that handles NaNs by interpolation."""
    if sigma <= 0:
        return arr
    nan_mask = np.isnan(arr)
    arr_filled = np.copy(arr)
    arr_filled[nan_mask] = 0
    smoothed = gaussian_filter1d(arr_filled, sigma, axis=axis)
    # normalize by the fraction of valid data
    ones = np.ones_like(arr)
    ones[nan_mask] = 0
    norm = gaussian_filter1d(ones, sigma, axis=axis)
    norm[norm == 0] = 1
    result = smoothed / norm
    result[nan_mask] = np.nan
    return result


def compute_dff_and_deconvolve(F, Fneu, trial_starts, trial_ends, frame_rate, n_planes=1):
    """
    Compute dF/F and deconvolved activity following the paper's pipeline.

    Steps:
    1. Mask to only include within-trial data (exclude teleport periods)
    2. Neuropil subtraction: F - 0.7 * Fneu
    3. Per-trial maximin baseline
    4. dF/F = (F_corr - baseline) / |baseline|
    5. Smooth with 2-sample Gaussian
    6. OASIS deconvolution (tau=0.7)
    """
    n_cells, n_frames = F.shape

    # Initialize with NaN (teleport periods will remain NaN)
    f_ = np.full((n_cells, n_frames), np.nan)

    # Copy only within-trial data
    for start, stop in zip(trial_starts, trial_ends):
        f_[:, start:stop] = F[:, start:stop]

    f_neu_ = np.full((n_cells, n_frames), np.nan)
    for start, stop in zip(trial_starts, trial_ends):
        f_neu_[:, start:stop] = Fneu[:, start:stop]

    # Neuropil subtraction
    nanmask = ~np.isnan(f_[0, :])
    f_[:, nanmask] = f_[:, nanmask] - NEU_COEF * f_neu_[:, nanmask]

    # Add back neuropil mean per trial (as in the paper's code)
    # and compute baseline
    flow = np.full((n_cells, n_frames), np.nan)
    for start, stop in zip(trial_starts, trial_ends):
        # Add back neuropil mean per trial so dF/F values are reasonable
        f_[:, start:stop] = f_[:, start:stop] + NEU_COEF * np.nanmean(
            f_neu_[:, start:stop], axis=1, keepdims=True
        )
        # Maximin baseline
        flow[:, start:stop] = nansmooth(f_[:, start:stop], BASELINE_SMOOTH_SIGMA, axis=1)
        flow[:, start:stop] = minimum_filter1d(flow[:, start:stop], BASELINE_WINDOW, axis=-1)
        flow[:, start:stop] = maximum_filter1d(flow[:, start:stop], BASELINE_WINDOW, axis=-1)

    # Compute dF/F
    dff = np.full((n_cells, n_frames), np.nan)
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])

    # Smooth dF/F and deconvolve per trial
    spks = np.full((n_cells, n_frames), np.nan)
    for start, stop in zip(trial_starts, trial_ends):
        dff[:, start:stop] = nansmooth(dff[:, start:stop], DFF_SMOOTH_SIGMA, axis=1)
        spks[:, start:stop] = dcnv.oasis(
            dff[:, start:stop], 2000, TAU, frame_rate / n_planes
        )

    return dff, spks


def get_reward_zone_for_trial(scene, trial_idx, n_trials, change_trial=30):
    """
    Determine the reward zone label for a given trial based on the scene name.
    Returns 'A', 'B', or 'C'.
    """
    # Single zone sessions
    if 'LocationA' in scene and '_to_' not in scene and '_to_Env' not in scene.split('Location')[-1]:
        return 'A'
    if 'LocationB' in scene and '_to_' not in scene and '_to_Env' not in scene.split('Location')[-1]:
        return 'B'
    if 'LocationC' in scene and '_to_' not in scene and '_to_Env' not in scene.split('Location')[-1]:
        return 'C'

    # Switch sessions - parse zone before and after switch
    # Handle cross-environment switches like 'Env1_C_to_Env2_B'
    if '_to_Env' in scene:
        # e.g. 'Env1_C_to_Env2_B' -> before='C', after='B'
        parts = scene.split('_to_')
        before_zone = parts[0].split('_')[-1]  # last char before _to_
        after_zone = parts[1].split('_')[-1]  # last char
    elif '_to_' in scene:
        # e.g. 'Env1_LocationA_to_B' -> before='A', after='B'
        parts = scene.split('_to_')
        before_zone = parts[0].split('Location')[-1]
        after_zone = parts[1]
    else:
        # Shouldn't happen
        return 'A'

    if trial_idx < change_trial:
        return before_zone
    else:
        return after_zone


def get_environment(scene):
    """Return 0 for Env1, 1 for Env2."""
    if 'Env2' in scene:
        # For cross-env switches, the environment changes at the switch
        if 'Env1' in scene:
            return None  # mixed - handle per trial
        return 1
    return 0


def get_environment_per_trial(scene, trial_idx, change_trial=30):
    """Return environment for a specific trial (handles cross-env switches)."""
    if '_to_Env' in scene:
        parts = scene.split('_to_Env')
        if trial_idx < change_trial:
            return 0 if 'Env1' in parts[0] else 1
        else:
            return 0 if parts[1].startswith('1') else 1
    elif scene.startswith('Env2'):
        return 1
    else:
        return 0


def discretize_distance_to_reward(position, rz_start, rz_end):
    """
    Compute signed distance from position to nearest point in reward zone,
    then discretize.

    Distance convention:
    - Negative = before the zone (position < rz_start)
    - Zero = inside the zone
    - Positive = after the zone (position > rz_end)

    Bins:
    0: < -50 cm
    1: -50 to -10 cm
    2: -10 to < 0 cm
    3: 0 cm (in zone)
    4: >0 to +10 cm
    5: +10 to +50 cm
    6: > +50 cm
    """
    dist = np.where(
        position < rz_start, position - rz_start,
        np.where(position > rz_end, position - rz_end, 0.0)
    )
    bins = np.zeros(len(dist), dtype=int)
    bins[dist < -50] = 0
    bins[(dist >= -50) & (dist < -10)] = 1
    bins[(dist >= -10) & (dist < 0)] = 2
    bins[dist == 0] = 3  # inside zone
    bins[(dist > 0) & (dist <= 10)] = 4
    bins[(dist > 10) & (dist <= 50)] = 5
    bins[dist > 50] = 6
    return bins


def discretize_position(position):
    """Discretize position into 5 equal bins spanning 450 cm."""
    bins = np.clip((position / 90).astype(int), 0, 4)
    return bins


def discretize_speed(speed):
    """Discretize speed into bins."""
    bins = np.zeros(len(speed), dtype=int)
    bins[speed < 2] = 0
    bins[(speed >= 2) & (speed < 10)] = 1
    bins[(speed >= 10) & (speed < 20)] = 2
    bins[(speed >= 20) & (speed < 40)] = 3
    bins[speed >= 40] = 4
    return bins


def rz_label_to_idx(label):
    """Map reward zone label to index: A=0, B=1, C=2."""
    return {'A': 0, 'B': 1, 'C': 2}[label]


def process_session(nwb_path, subject_id, scene, exp_day):
    """Process a single NWB session and return trial-organized data."""
    print(f"  Processing {subject_id} day {exp_day} ({scene})")

    with h5py.File(nwb_path, 'r') as f:
        bts = f['processing']['behavior']['BehavioralTimeSeries']

        # Load behavioral data
        position = bts['position']['data'][:]
        speed = bts['speed']['data'][:]
        lick = bts['lick']['data'][:]
        trial_start_signal = bts['trial_start']['data'][:]
        teleport_signal = bts['teleport']['data'][:]
        timestamps = bts['position']['timestamps'][:]

        # Load reward events
        reward_data = bts['Reward']['data'][:]
        reward_timestamps = bts['Reward']['timestamps'][:]

        # Load neural data - handle multi-plane
        iscell = f['processing']['ophys']['ImageSegmentation']['PlaneSegmentation']['iscell'][:]
        imaging_rate = f['acquisition']['TwoPhotonSeries']['imaging_plane']['imaging_rate'][()]

        seg = f['processing']['ophys']['ImageSegmentation']['PlaneSegmentation']
        if 'planeIdx' in seg:
            plane_idx = seg['planeIdx'][:]
            unique_planes = np.unique(plane_idx)
            n_planes = len(unique_planes)
        else:
            plane_idx = np.zeros(iscell.shape[0], dtype=int)
            n_planes = 1

        # Load and concatenate F/Fneu across planes, maintaining ROI order
        fluor_group = f['processing']['ophys']['Fluorescence']
        neuro_group = f['processing']['ophys']['Neuropil']
        F_list = []
        Fneu_list = []
        for p in sorted(unique_planes if n_planes > 1 else [0]):
            plane_key = f'plane{int(p)}'
            F_list.append(fluor_group[plane_key]['data'][:].T)  # (n_rois_plane, n_frames)
            Fneu_list.append(neuro_group[plane_key]['data'][:].T)
        F = np.concatenate(F_list, axis=0)  # (n_rois_total, n_frames)
        Fneu = np.concatenate(Fneu_list, axis=0)

    # Ensure behavioral and neural data have same length
    n_frames = min(len(position), F.shape[1])
    position = position[:n_frames]
    speed = speed[:n_frames]
    lick = lick[:n_frames]
    trial_start_signal = trial_start_signal[:n_frames]
    teleport_signal = teleport_signal[:n_frames]
    timestamps = timestamps[:n_frames]
    F = F[:, :n_frames]
    Fneu = Fneu[:, :n_frames]

    # Get frame time
    frame_time = np.median(np.diff(timestamps))

    # Find trial boundaries
    tstart_inds = np.where(trial_start_signal > 0)[0]
    teleport_inds = np.where(teleport_signal > 0)[0]

    n_trials = min(len(tstart_inds), len(teleport_inds))
    tstart_inds = tstart_inds[:n_trials]
    teleport_inds = teleport_inds[:n_trials]

    # Ensure each trial_start comes before its teleport
    valid = []
    for i in range(n_trials):
        if tstart_inds[i] < teleport_inds[i]:
            valid.append(i)
    tstart_inds = tstart_inds[valid]
    teleport_inds = teleport_inds[valid]
    n_trials = len(tstart_inds)

    if n_trials < 2:
        print(f"    Skipping: only {n_trials} valid trials")
        return None

    # Filter cells by iscell
    cell_mask = iscell[:, 0] == 1
    cell_indices = np.where(cell_mask)[0]
    F_cells = F[cell_indices, :]
    Fneu_cells = Fneu[cell_indices, :]

    # Compute dF/F and deconvolved activity
    dff, spks = compute_dff_and_deconvolve(
        F_cells, Fneu_cells, tstart_inds, teleport_inds, imaging_rate, n_planes
    )

    # Filter putative interneurons (speed correlation > 0.5 with dF/F)
    # Compute correlation only on valid (non-NaN) within-trial data
    valid_mask = ~np.isnan(dff[0, :])
    speed_valid = speed[valid_mask]
    keep_neuron = np.ones(dff.shape[0], dtype=bool)
    for n_idx in range(dff.shape[0]):
        dff_valid = dff[n_idx, valid_mask]
        if np.std(dff_valid) > 0 and np.std(speed_valid) > 0:
            r, _ = pearsonr(dff_valid, speed_valid)
            if r > INTERNEURON_SPEED_CORR_THR:
                keep_neuron[n_idx] = False

    n_interneurons = (~keep_neuron).sum()
    spks = spks[keep_neuron, :]
    dff = dff[keep_neuron, :]
    n_neurons = spks.shape[0]

    if n_neurons < 1:
        print(f"    Skipping: no neurons after filtering")
        return None

    print(f"    {n_neurons} neurons ({n_interneurons} putative interneurons removed), {n_trials} trials")

    # Determine reward delivery per trial
    # A trial is rewarded if a reward event falls within the trial time window
    trial_rewarded = np.zeros(n_trials, dtype=int)
    for i in range(n_trials):
        t_start_time = timestamps[tstart_inds[i]]
        t_end_time = timestamps[teleport_inds[i]]
        # Check if any reward timestamp falls in this trial
        reward_in_trial = (reward_timestamps >= t_start_time) & (reward_timestamps <= t_end_time)
        if np.any(reward_in_trial):
            trial_rewarded[i] = 1

    # Correct lick sensor errors per trial
    lick_corrected = np.copy(lick)
    for i in range(n_trials):
        t0, t1 = tstart_inds[i], teleport_inds[i]
        trial_licks = lick_corrected[t0:t1]
        n_frames_trial = len(trial_licks)
        if n_frames_trial > 0:
            # Paper: >30% of frames with cumulative lick count > 2
            frac_high = np.sum(trial_licks > 2) / n_frames_trial
            if frac_high > 0.3:
                lick_corrected[t0:t1] = np.nan

    # Build per-trial data
    neural_trials = []
    input_trials = []
    output_trials = []

    for i in range(n_trials):
        t0, t1 = tstart_inds[i], teleport_inds[i]
        n_timepoints = t1 - t0

        if n_timepoints < 2:
            continue

        # Neural activity (deconvolved): (n_neurons, n_timepoints)
        trial_neural = spks[:, t0:t1]
        # Replace any remaining NaN with 0
        trial_neural = np.nan_to_num(trial_neural, nan=0.0)

        # Time from start of trial (seconds)
        time_from_start = (timestamps[t0:t1] - timestamps[t0])

        # Environment type (per trial)
        env_type = get_environment_per_trial(scene, i)

        # Trial number
        trial_number = float(i)

        # Previous trial outcome (0=omission, 1=rewarded)
        if i == 0:
            prev_outcome = 1.0  # first trial: no previous, default to rewarded
        else:
            prev_outcome = float(trial_rewarded[i - 1])

        # Inputs: (4, n_timepoints) for time_from_start, rest are per-trial
        trial_input = np.array([
            time_from_start,
            np.full(n_timepoints, env_type, dtype=float),
            np.full(n_timepoints, trial_number, dtype=float),
            np.full(n_timepoints, prev_outcome, dtype=float),
        ])

        # Outputs
        rz_label = get_reward_zone_for_trial(scene, i, n_trials)
        rz_start, rz_end = REWARD_ZONES[rz_label]

        pos_trial = position[t0:t1]
        speed_trial = np.abs(speed[t0:t1])  # use absolute speed
        lick_trial = lick_corrected[t0:t1]

        # Discretize outputs
        dist_to_rz = discretize_distance_to_reward(pos_trial, rz_start, rz_end)
        abs_position = discretize_position(pos_trial)
        speed_binned = discretize_speed(speed_trial)
        lick_binary = np.where(np.isnan(lick_trial), 0, (lick_trial > 0).astype(int))
        rz_location = rz_label_to_idx(rz_label)
        reward_outcome = trial_rewarded[i]

        # Output: (6, n_timepoints) for time-varying, per-trial for constant
        # Use int type so decoder can use values as category indices
        trial_output = np.array([
            dist_to_rz,
            abs_position,
            speed_binned,
            lick_binary,
            np.full(n_timepoints, rz_location, dtype=int),
            np.full(n_timepoints, reward_outcome, dtype=int),
        ], dtype=int)

        neural_trials.append(trial_neural)
        input_trials.append(trial_input)
        output_trials.append(trial_output)

    if len(neural_trials) < 2:
        print(f"    Skipping: fewer than 2 valid trials after processing")
        return None

    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'n_neurons': n_neurons,
        'subject_id': subject_id,
        'scene': scene,
        'exp_day': exp_day,
        'imaging_rate': imaging_rate,
        'frame_time': frame_time,
    }


def main():
    all_neural = []
    all_input = []
    all_output = []
    all_subject_idx = []
    all_brain_region_idx = []
    session_info = []

    subjects = sorted(SESSIONS_META.keys())

    for sub_idx, subject_id in enumerate(subjects):
        sub_dir = os.path.join(DATA_DIR, f'sub-{subject_id}')
        if not os.path.exists(sub_dir):
            print(f"WARNING: {sub_dir} not found, skipping")
            continue

        sessions = SESSIONS_META[subject_id]
        print(f"Subject {subject_id}: {len(sessions)} sessions")

        for ses_num, scene, exp_day in sessions:
            # NWB session number matches exp_day directly
            nwb_filename = f'sub-{subject_id}_ses-{exp_day:02d}_behavior+ophys.nwb'
            nwb_path = os.path.join(sub_dir, nwb_filename)

            if not os.path.exists(nwb_path):
                print(f"  WARNING: {nwb_path} not found, skipping")
                continue

            result = process_session(nwb_path, subject_id, scene, exp_day)
            if result is None:
                continue

            all_neural.append(result['neural'])
            all_input.append(result['input'])
            all_output.append(result['output'])
            all_subject_idx.append(sub_idx)
            all_brain_region_idx.append(
                np.zeros(result['n_neurons'], dtype=int)  # all CA1
            )
            session_info.append({
                'subject': subject_id,
                'exp_day': exp_day,
                'scene': scene,
                'imaging_rate': result['imaging_rate'],
            })

    # Compute median time bin size across all sessions
    time_bin_sizes = []
    for info in session_info:
        time_bin_sizes.append(1000.0 / info['imaging_rate'])
    median_time_bin = np.median(time_bin_sizes) if time_bin_sizes else 64.5

    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': subjects,
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
            ['< -50 cm', '-50 to -10 cm', '-10 to 0 cm', 'in zone', '0 to +10 cm', '+10 to +50 cm', '> +50 cm'],
            ['0-90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '360-450 cm'],
            ['< 2 cm/s', '2-10 cm/s', '10-20 cm/s', '20-40 cm/s', '> 40 cm/s'],
            ['no lick', 'lick'],
            ['A', 'B', 'C'],
            ['no reward', 'reward'],
        ],
        'metadata': {
            'task_description': 'Virtual reality hidden reward zone navigation task with reward zone switches across two environments. Mice traverse a 450 cm linear track and lick to receive reward in a hidden 50 cm zone.',
            'time_bin_size': median_time_bin,
            'temporal_alignment_event': 'start of trial (first imaging frame after teleport)',
            'off_start': 0.0,
            'off_end': None,
            'session_info': session_info,
            'track_length_cm': 450,
            'reward_zones': {'A': [80, 130], 'B': [200, 250], 'C': [320, 370]},
        },
    }

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'converted_data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"\nSaved converted data to {output_path}")
    print(f"  {len(all_neural)} sessions, {len(subjects)} subjects")
    for i, sub in enumerate(subjects):
        n_sess = sum(1 for idx in all_subject_idx if idx == i)
        print(f"  {sub}: {n_sess} sessions")


if __name__ == '__main__':
    main()
