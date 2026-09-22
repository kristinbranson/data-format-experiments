"""
Convert NWB data from Sosa et al. (2025) "A flexible hippocampal population code
for experience relative to reward" into the standardized decoder format.

Key decisions:
- Neural data: Use pre-computed Deconvolved (OASIS) activity from NWB files. This is
  the result of the full processing pipeline (neuropil subtraction, dF/F baseline
  correction with maximin method, Gaussian smoothing, OASIS deconvolution) as described
  in the paper and implemented in the reference code.
- Cell filtering: (1) Suite2P manual curation (iscell), (2) exclude putative interneurons
  identified by Pearson correlation > 0.5 between dF/F and running speed (paper methods).
- dF/F for interneuron detection: Computed from raw fluorescence with neuropil subtraction
  (F - 0.7*Fneu), maximin baseline (20s window = 300 samples at ~15.5 Hz), and 2-sample
  Gaussian smoothing, following the reference preprocessing.py code.
- Trial segmentation: Using trial_start and teleport signals from NWB behavioral data.
- Temporal alignment: Aligned to trial start (time=0 at first frame of each trial).
- Time bins: Native imaging frame rate (~15.5 Hz, ~64.5 ms per bin).
- No speed threshold: All timepoints within trials are included (speed threshold of
  2 cm/s is only used in the paper for spatial analyses like place cell detection,
  not for raw activity extraction).
- Reward zone identification: From sessions_dict scene names, with switch at trial 30
  on switch days (days 3, 5, 7, 8, 10, 12, 14).
- Environment: From NWB environment time series (0=ENV1, 1=ENV2).
- Reward outcome: Determined by mapping Reward event timestamps to trial time windows.
- Lick: Binarized from the NWB lick signal (any lick > 0 → 1).
- All 11 switch-task mice included; fixed-condition mice (m2, m6, m10) are not in the
  NWB dataset. Sessions with exp_day 0 (running training) excluded as they lack the
  main task structure.
"""

import numpy as np
import scipy.ndimage as ndi
import h5py
import pickle
import os
from collections import OrderedDict

# ============================================================
# Session metadata from the reference code's sessions_dict.py
# Maps GCAMP name -> list of session dicts with scene and exp_day
# ============================================================

SESSIONS_DICT = {
    'm3': [
        {'scene': 'Env1_LocationC', 'exp_day': 1},
        {'scene': 'Env1_LocationC', 'exp_day': 2},
        {'scene': 'Env1_LocationC_to_A', 'exp_day': 3},
        {'scene': 'Env1_LocationA', 'exp_day': 4},
        {'scene': 'Env1_LocationA_to_B', 'exp_day': 5},
        {'scene': 'Env1_LocationB', 'exp_day': 6},
        {'scene': 'Env1_LocationB_to_C', 'exp_day': 7},
        {'scene': 'Env1_C_to_Env2_B', 'exp_day': 8},
        {'scene': 'Env2_LocationB', 'exp_day': 9},
        {'scene': 'Env2_LocationB_to_A', 'exp_day': 10},
        {'scene': 'Env2_LocationA', 'exp_day': 11},
        {'scene': 'Env2_LocationA_to_C', 'exp_day': 12},
        {'scene': 'Env2_LocationC', 'exp_day': 13},
        {'scene': 'Env2_LocationC_to_B', 'exp_day': 14},
    ],
    'm4': [
        {'scene': 'Env1_LocationB', 'exp_day': 1},
        {'scene': 'Env1_LocationB', 'exp_day': 2},
        {'scene': 'Env1_LocationB_to_A', 'exp_day': 3},
        {'scene': 'Env1_LocationA', 'exp_day': 4},
        {'scene': 'Env1_LocationA_to_C', 'exp_day': 5},
        {'scene': 'Env1_LocationC', 'exp_day': 6},
        {'scene': 'Env1_LocationC_to_B', 'exp_day': 7},
        {'scene': 'Env1_B_to_Env2_C', 'exp_day': 8},
        {'scene': 'Env2_LocationC', 'exp_day': 9},
        {'scene': 'Env2_LocationC_to_A', 'exp_day': 10},
        {'scene': 'Env2_LocationA', 'exp_day': 11},
        {'scene': 'Env2_LocationA_to_B', 'exp_day': 12},
        {'scene': 'Env2_LocationB', 'exp_day': 13},
        {'scene': 'Env2_LocationB_to_C', 'exp_day': 14},
    ],
    'm7': [
        {'scene': 'Env1_LocationA', 'exp_day': 1},
        {'scene': 'Env1_LocationA', 'exp_day': 2},
        {'scene': 'Env1_LocationA_to_C', 'exp_day': 3},
        {'scene': 'Env1_LocationC', 'exp_day': 4},
        {'scene': 'Env1_LocationC_to_B', 'exp_day': 5},
        {'scene': 'Env1_LocationB', 'exp_day': 6},
        {'scene': 'Env1_LocationB_to_A', 'exp_day': 7},
        {'scene': 'Env1_A_to_Env2_B', 'exp_day': 8},
        {'scene': 'Env2_LocationB', 'exp_day': 9},
        {'scene': 'Env2_LocationB_to_C', 'exp_day': 10},
        {'scene': 'Env2_LocationC', 'exp_day': 11},
        {'scene': 'Env2_LocationC_to_A', 'exp_day': 12},
        {'scene': 'Env2_LocationA', 'exp_day': 13},
        {'scene': 'Env2_LocationA_to_B', 'exp_day': 14},
    ],
    'm11': [
        {'scene': 'Env1_LocationB_to_A', 'exp_day': 3},
        {'scene': 'Env1_LocationA', 'exp_day': 4},
        {'scene': 'Env1_LocationA_to_C', 'exp_day': 5},
        {'scene': 'Env1_LocationC', 'exp_day': 6},
        {'scene': 'Env1_LocationC_to_B', 'exp_day': 7},
        {'scene': 'Env1_B_to_Env2_C', 'exp_day': 8},
        {'scene': 'Env2_LocationC', 'exp_day': 9},
        {'scene': 'Env2_LocationC_to_A', 'exp_day': 10},
        {'scene': 'Env2_LocationA', 'exp_day': 11},
        {'scene': 'Env2_LocationA_to_B', 'exp_day': 12},
        {'scene': 'Env2_LocationB', 'exp_day': 13},
        {'scene': 'Env2_LocationB_to_C', 'exp_day': 14},
    ],
    'm12': [
        {'scene': 'Env1_LocationB', 'exp_day': 1},
        {'scene': 'Env1_LocationB', 'exp_day': 2},
        {'scene': 'Env1_LocationB_to_C', 'exp_day': 3},
        {'scene': 'Env1_LocationC', 'exp_day': 4},
        {'scene': 'Env1_LocationC_to_A', 'exp_day': 5},
        {'scene': 'Env1_LocationA', 'exp_day': 6},
        {'scene': 'Env1_LocationA_to_B', 'exp_day': 7},
        {'scene': 'Env1_B_to_Env2_A', 'exp_day': 8},
        {'scene': 'Env2_LocationA', 'exp_day': 9},
        {'scene': 'Env2_LocationA_to_C', 'exp_day': 10},
        {'scene': 'Env2_LocationC', 'exp_day': 11},
        {'scene': 'Env2_LocationC_to_B', 'exp_day': 12},
        {'scene': 'Env2_LocationB', 'exp_day': 13},
        {'scene': 'Env2_LocationB_to_A', 'exp_day': 14},
    ],
    'm13': [
        {'scene': 'Env1_LocationC', 'exp_day': 1},
        {'scene': 'Env1_LocationC', 'exp_day': 2},
        {'scene': 'Env1_LocationC_to_B', 'exp_day': 3},
        {'scene': 'Env1_LocationB', 'exp_day': 4},
        {'scene': 'Env1_LocationB_to_A', 'exp_day': 5},
        {'scene': 'Env1_LocationA', 'exp_day': 6},
        {'scene': 'Env1_LocationA_to_C', 'exp_day': 7},
        {'scene': 'Env1_C_to_Env2_A', 'exp_day': 8},
        {'scene': 'Env2_LocationA', 'exp_day': 9},
        {'scene': 'Env2_LocationA_to_B', 'exp_day': 10},
        {'scene': 'Env2_LocationB', 'exp_day': 11},
        {'scene': 'Env2_LocationB_to_C', 'exp_day': 12},
        {'scene': 'Env2_LocationC', 'exp_day': 13},
        {'scene': 'Env2_LocationC_to_A', 'exp_day': 14},
    ],
    'm14': [
        {'scene': 'Env1_LocationA', 'exp_day': 1},
        {'scene': 'Env1_LocationA', 'exp_day': 2},
        {'scene': 'Env1_LocationA_to_B', 'exp_day': 3},
        {'scene': 'Env1_LocationB', 'exp_day': 4},
        {'scene': 'Env1_LocationB_to_C', 'exp_day': 5},
        {'scene': 'Env1_LocationC', 'exp_day': 6},
        {'scene': 'Env1_LocationC_to_A', 'exp_day': 7},
        {'scene': 'Env1_A_to_Env2_C', 'exp_day': 8},
        {'scene': 'Env2_LocationC', 'exp_day': 9},
        {'scene': 'Env2_LocationC_to_B', 'exp_day': 10},
        {'scene': 'Env2_LocationB', 'exp_day': 11},
        {'scene': 'Env2_LocationB_to_A', 'exp_day': 12},
        {'scene': 'Env2_LocationA', 'exp_day': 13},
        {'scene': 'Env2_LocationA_to_C', 'exp_day': 14},
    ],
    'm15': [
        {'scene': 'Env1_LocationA', 'exp_day': 1},
        {'scene': 'Env1_LocationA', 'exp_day': 2},
        {'scene': 'Env1_LocationA_to_C', 'exp_day': 3},
        {'scene': 'Env1_LocationC', 'exp_day': 4},
        {'scene': 'Env1_LocationC_to_B', 'exp_day': 5},
        {'scene': 'Env1_LocationB', 'exp_day': 6},
        {'scene': 'Env1_LocationB_to_A', 'exp_day': 7},
        {'scene': 'Env1_A_to_Env2_B', 'exp_day': 8},
        {'scene': 'Env2_LocationB', 'exp_day': 9},
        {'scene': 'Env2_LocationB_to_C', 'exp_day': 10},
        {'scene': 'Env2_LocationC', 'exp_day': 11},
        {'scene': 'Env2_LocationC_to_A', 'exp_day': 12},
        {'scene': 'Env2_LocationA', 'exp_day': 13},
        {'scene': 'Env2_LocationA_to_B', 'exp_day': 14},
    ],
    'm17': [
        {'scene': 'Env2_LocationB', 'exp_day': 1},
        {'scene': 'Env2_LocationB', 'exp_day': 2},
        {'scene': 'Env2_LocationB_to_C', 'exp_day': 3},
        {'scene': 'Env2_LocationC', 'exp_day': 4},
        {'scene': 'Env2_LocationC_to_A', 'exp_day': 5},
        {'scene': 'Env2_LocationA', 'exp_day': 6},
        {'scene': 'Env2_LocationA_to_B', 'exp_day': 7},
        {'scene': 'Env2_B_to_Env1_A', 'exp_day': 8},
        {'scene': 'Env1_LocationA', 'exp_day': 9},
        {'scene': 'Env1_LocationA_to_C', 'exp_day': 10},
        {'scene': 'Env1_LocationC', 'exp_day': 11},
        {'scene': 'Env1_LocationC_to_B', 'exp_day': 12},
        {'scene': 'Env1_LocationB', 'exp_day': 13},
        {'scene': 'Env1_LocationB_to_A', 'exp_day': 14},
    ],
    'm18': [
        {'scene': 'Env2_LocationA', 'exp_day': 1},
        {'scene': 'Env2_LocationA', 'exp_day': 2},
        {'scene': 'Env2_LocationA_to_B', 'exp_day': 3},
        {'scene': 'Env2_LocationB', 'exp_day': 4},
        {'scene': 'Env2_LocationB_to_C', 'exp_day': 5},
        {'scene': 'Env2_LocationC', 'exp_day': 6},
        {'scene': 'Env2_LocationC_to_A', 'exp_day': 7},
        {'scene': 'Env2_A_to_Env1_C', 'exp_day': 8},
        {'scene': 'Env1_LocationC', 'exp_day': 9},
        {'scene': 'Env1_LocationC_to_B', 'exp_day': 10},
        {'scene': 'Env1_LocationB', 'exp_day': 11},
        {'scene': 'Env1_LocationB_to_A', 'exp_day': 12},
        {'scene': 'Env1_LocationA', 'exp_day': 13},
        {'scene': 'Env1_LocationA_to_C', 'exp_day': 14},
    ],
    'm19': [
        {'scene': 'Env1_LocationC', 'exp_day': 1},
        {'scene': 'Env1_LocationC', 'exp_day': 2},
        {'scene': 'Env1_LocationC_to_B', 'exp_day': 3},
        {'scene': 'Env1_LocationB', 'exp_day': 4},
        {'scene': 'Env1_LocationB_to_A', 'exp_day': 5},
        {'scene': 'Env1_LocationA', 'exp_day': 6},
        {'scene': 'Env1_LocationA_to_C', 'exp_day': 7},
        {'scene': 'Env1_C_to_Env2_A', 'exp_day': 8},
        {'scene': 'Env2_LocationA', 'exp_day': 9},
        {'scene': 'Env2_LocationA_to_B', 'exp_day': 10},
        {'scene': 'Env2_LocationB', 'exp_day': 11},
        {'scene': 'Env2_LocationB_to_C', 'exp_day': 12},
        {'scene': 'Env2_LocationC', 'exp_day': 13},
        {'scene': 'Env2_LocationC_to_A', 'exp_day': 14},
    ],
}

# Reward zone boundaries in cm
REWARD_ZONES = {
    'A': (80, 130),
    'B': (200, 250),
    'C': (320, 370),
}

SWITCH_TRIAL = 30  # Trial index at which reward zone switches (0-indexed)

DATA_DIR = '/app/data'


def parse_scene(scene):
    """Parse scene name to extract reward zone(s) and environment(s).

    Returns (env_before, zone_before, env_after, zone_after).
    If no switch, env_after and zone_after are None.
    """
    # Handle environment switch scenes like 'Env1_C_to_Env2_B'
    if '_to_Env' in scene:
        # e.g., 'Env1_C_to_Env2_B'
        parts = scene.split('_to_')
        before = parts[0]  # 'Env1_C'
        after = parts[1]   # 'Env2_B'
        env_before = int(before[3])  # 1 or 2
        zone_before = before.split('_')[1]  # 'C'
        env_after = int(after[3])
        zone_after = after.split('_')[1]
        return env_before, zone_before, env_after, zone_after

    # Handle within-environment switch: 'Env1_LocationA_to_B'
    if '_to_' in scene:
        parts = scene.split('_to_')
        before = parts[0]  # 'Env1_LocationA'
        zone_after = parts[1]  # 'B'
        env = int(before[3])
        zone_before = before.split('Location')[1]
        return env, zone_before, env, zone_after

    # No switch: 'Env1_LocationA'
    env = int(scene[3])
    zone = scene.split('Location')[1]
    return env, zone, None, None


def get_reward_zone_for_trial(scene_info, trial_idx):
    """Get the active reward zone letter for a given trial.

    On switch days, trials 0-29 use zone_before, trials 30+ use zone_after.
    """
    env_before, zone_before, env_after, zone_after = parse_scene(scene_info)
    if zone_after is not None and trial_idx >= SWITCH_TRIAL:
        return zone_after
    return zone_before


def get_env_for_trial(scene_info, trial_idx):
    """Get the environment (1 or 2) for a given trial."""
    env_before, _, env_after, zone_after = parse_scene(scene_info)
    if zone_after is not None and trial_idx >= SWITCH_TRIAL:
        return env_after if env_after is not None else env_before
    return env_before


def compute_distance_to_reward_zone(position, zone_start, zone_end):
    """Compute signed distance from position to nearest point of reward zone.

    Negative = before zone, 0 = in zone, positive = past zone.
    """
    dist = np.zeros_like(position, dtype=float)
    before = position < zone_start
    inside = (position >= zone_start) & (position <= zone_end)
    after = position > zone_end
    dist[before] = position[before] - zone_start
    dist[inside] = 0.0
    dist[after] = position[after] - zone_end
    return dist


def discretize_distance(dist):
    """Discretize distance to reward zone into 7 bins."""
    out = np.zeros(len(dist), dtype=np.int64)
    out[dist < -50] = 0
    out[(dist >= -50) & (dist < -10)] = 1
    out[(dist >= -10) & (dist < 0)] = 2
    out[dist == 0] = 3
    out[(dist > 0) & (dist <= 10)] = 4
    out[(dist > 10) & (dist <= 50)] = 5
    out[dist > 50] = 6
    return out


def discretize_position(position):
    """Discretize position into 5 equal bins spanning 450 cm track."""
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


def compute_dff_for_interneuron_detection(F, Fneu, trial_starts, trial_ends, frame_rate=15.5):
    """Compute dF/F following the reference code's preprocessing pipeline.

    Steps (per the reference code preprocessing.py):
    1. Set fluorescence outside trials to NaN
    2. Neuropil subtraction: F -= 0.7 * Fneu  (single-channel default)
    3. Add back neuropil mean per trial (to avoid dividing by small numbers)
    4. Maximin baseline: smooth -> minimum filter (300 samples) -> maximum filter (300 samples)
    5. dF/F = (F_corrected - baseline) / |baseline|
    6. Smooth with 2-sample Gaussian per trial

    Args:
        F: (n_timepoints, n_cells) raw fluorescence
        Fneu: (n_timepoints, n_cells) neuropil fluorescence
        trial_starts: list of trial start frame indices
        trial_ends: list of trial end frame indices
        frame_rate: imaging frame rate

    Returns:
        dff: (n_cells, n_timepoints) dF/F traces (NaN outside trials)
    """
    n_frames, n_cells = F.shape
    neu_coef = 0.7

    # Transpose to (n_cells, n_timepoints) for processing
    F_t = F.T.astype(np.float64).copy()
    Fneu_t = Fneu.T.astype(np.float64).copy()

    # Initialize with NaN (outside trials)
    f_ = np.full_like(F_t, np.nan)
    fneu_ = np.full_like(Fneu_t, np.nan)

    # Copy only trial data
    for start, stop in zip(trial_starts, trial_ends):
        f_[:, start:stop] = F_t[:, start:stop]
        fneu_[:, start:stop] = Fneu_t[:, start:stop]

    # Neuropil subtraction (single channel default in reference code)
    f_ -= neu_coef * fneu_

    # Baseline computation per trial
    baseline = np.full_like(f_, np.nan)
    dff = np.full_like(f_, np.nan)

    window = 300  # ~20 seconds at 15.5 Hz

    for start, stop in zip(trial_starts, trial_ends):
        trial_len = stop - start
        if trial_len < 3:
            continue

        # Add back neuropil mean per trial (reference code line 434)
        f_[:, start:stop] += neu_coef * np.nanmean(
            fneu_[:, start:stop], axis=1, keepdims=True)

        # Smooth with Gaussian sigma=15 (reference code line 440: nansmooth with [0, 15])
        bl = np.copy(f_[:, start:stop])
        for c in range(n_cells):
            bl[c] = ndi.gaussian_filter1d(bl[c], sigma=15)

        # Minimum filter then maximum filter (maximin baseline)
        bl = ndi.minimum_filter1d(bl, size=window, axis=-1)
        bl = ndi.maximum_filter1d(bl, size=window, axis=-1)
        baseline[:, start:stop] = bl

    # Compute dF/F
    nanmask = ~np.isnan(f_[0, :])
    dff[:, nanmask] = (f_[:, nanmask] - baseline[:, nanmask]) / np.abs(baseline[:, nanmask])

    # Smooth dF/F with 2-sample Gaussian per trial
    for start, stop in zip(trial_starts, trial_ends):
        if stop - start < 3:
            continue
        for c in range(n_cells):
            dff[c, start:stop] = ndi.gaussian_filter1d(dff[c, start:stop], sigma=2)

    return dff


def detect_interneurons(dff, speed, threshold=0.5):
    """Detect putative interneurons by speed-dF/F correlation.

    Following the reference code spatial.py is_putative_interneuron():
    Cells with Pearson correlation > threshold between their dF/F and
    running speed are flagged as interneurons.

    Args:
        dff: (n_cells, n_timepoints) dF/F traces
        speed: (n_timepoints,) running speed
        threshold: correlation threshold (0.5 per paper, matching int_thresh in notebooks)

    Returns:
        is_interneuron: (n_cells,) boolean mask
    """
    n_cells = dff.shape[0]
    is_int = np.zeros(n_cells, dtype=bool)
    nanmask = ~np.isnan(dff[0, :])

    for c in range(n_cells):
        if nanmask.sum() > 10:
            r = np.corrcoef(dff[c, nanmask], speed[nanmask])[0, 1]
            if np.isnan(r):
                continue
            is_int[c] = r > threshold

    return is_int


def process_session(mouse_id, exp_day, nwb_path, scene):
    """Process a single NWB session file.

    Returns:
        neural_trials: list of (n_neurons, n_timepoints) arrays
        input_trials: list of (n_input, n_timepoints) or (n_input,) arrays
        output_trials: list of (n_output, n_timepoints) or (n_output,) arrays
        n_neurons: number of neurons after filtering
        info: dict with session metadata
    """
    f = h5py.File(nwb_path, 'r')

    # --- Load neural data (handle multi-plane) ---
    dec_group = f['processing/ophys/Deconvolved']
    fl_group = f['processing/ophys/Fluorescence']
    neu_group = f['processing/ophys/Neuropil']
    planes = sorted([k for k in dec_group.keys() if k.startswith('plane')])

    dec_planes = [dec_group[p]['data'][:] for p in planes]
    fl_planes = [fl_group[p]['data'][:] for p in planes]
    neu_planes = [neu_group[p]['data'][:] for p in planes]

    # Concatenate across planes (paper: "planes were pooled for all analyses")
    deconvolved = np.concatenate(dec_planes, axis=1)   # (n_frames, n_rois)
    fluorescence = np.concatenate(fl_planes, axis=1)
    neuropil_data = np.concatenate(neu_planes, axis=1)

    iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:]

    # Get frame rate from the ImagingPlane
    frame_rate = f['general/optophysiology/ImagingPlane/imaging_rate'][()]

    # --- Load behavioral data ---
    beh = f['processing/behavior/BehavioralTimeSeries']
    position = beh['position/data'][:]
    speed = beh['speed/data'][:]
    lick = beh['lick/data'][:]
    timestamps = beh['position/timestamps'][:]
    trial_start_signal = beh['trial_start/data'][:]
    teleport_signal = beh['teleport/data'][:]
    env_signal = beh['environment/data'][:]

    # Reward timestamps
    reward_ts = beh['Reward/timestamps'][:]

    f.close()

    # --- Cell filtering ---
    # Step 1: Suite2P manual curation
    cell_mask = iscell[:, 0].astype(bool)

    # Ensure neural and behavioral data have same number of frames
    n_frames_neural = deconvolved.shape[0]
    n_frames_behav = len(position)
    n_frames = min(n_frames_neural, n_frames_behav)

    deconvolved = deconvolved[:n_frames]
    fluorescence = fluorescence[:n_frames]
    neuropil_data = neuropil_data[:n_frames]
    position = position[:n_frames]
    speed = speed[:n_frames]
    lick = lick[:n_frames]
    timestamps = timestamps[:n_frames]
    trial_start_signal = trial_start_signal[:n_frames]
    teleport_signal = teleport_signal[:n_frames]
    env_signal = env_signal[:n_frames]

    # Re-compute trial boundaries after truncation
    trial_start_idx = np.where(trial_start_signal > 0)[0]
    teleport_idx = np.where(teleport_signal > 0)[0]

    n_trials = min(len(trial_start_idx), len(teleport_idx))
    trial_start_idx = trial_start_idx[:n_trials]
    teleport_idx = teleport_idx[:n_trials]

    # Compute dF/F for interneuron detection (using all ROIs that pass iscell)
    cell_indices = np.where(cell_mask)[0]
    F_cells = fluorescence[:, cell_indices]
    Fneu_cells = neuropil_data[:, cell_indices]

    dff = compute_dff_for_interneuron_detection(
        F_cells, Fneu_cells, trial_start_idx, teleport_idx, frame_rate)

    # Step 3: Detect interneurons
    is_int = detect_interneurons(dff, speed, threshold=0.5)

    n_interneurons = is_int.sum()
    print(f"  {mouse_id} day {exp_day}: {cell_mask.sum()} cells, {n_interneurons} interneurons excluded")

    # Final cell mask: iscell AND NOT interneuron
    # is_int indexes into cell_indices, so we need to map back
    final_cell_indices = cell_indices[~is_int]
    n_neurons = len(final_cell_indices)

    if n_neurons == 0:
        return None, None, None, 0, {}

    # Get filtered deconvolved data: (n_frames, n_neurons)
    deconv_filtered = deconvolved[:, final_cell_indices]

    # --- Process trials ---
    neural_trials = []
    input_trials = []
    output_trials = []

    # Parse scene for reward zone info
    env_before, zone_before, env_after, zone_after = parse_scene(scene)

    dt = 1.0 / frame_rate  # time per frame in seconds

    prev_rewarded = 0  # For the first trial, no previous trial → 0

    for t in range(n_trials):
        t_start = trial_start_idx[t]
        t_end = teleport_idx[t]

        if t_end <= t_start:
            continue

        # --- Neural data ---
        # (n_neurons, n_timepoints)
        neural = deconv_filtered[t_start:t_end, :].T.astype(np.float32)

        n_tp = neural.shape[1]
        if n_tp < 2:
            continue

        # --- Position and behavior ---
        pos_trial = position[t_start:t_end]
        speed_trial = speed[t_start:t_end]
        lick_trial = lick[t_start:t_end]

        # Clamp position to [0, 450]
        pos_trial = np.clip(pos_trial, 0, 450)

        # --- Determine active reward zone ---
        zone_letter = get_reward_zone_for_trial(scene, t)
        zone_start, zone_end = REWARD_ZONES[zone_letter]
        zone_idx = {'A': 0, 'B': 1, 'C': 2}[zone_letter]

        # --- Determine environment ---
        env = get_env_for_trial(scene, t)
        env_binary = 0 if env == 1 else 1  # ENV1=0, ENV2=1

        # --- Determine reward outcome ---
        t_start_time = timestamps[t_start]
        t_end_time = timestamps[t_end - 1]
        rewarded = int(np.any((reward_ts >= t_start_time) & (reward_ts <= t_end_time)))

        # --- Compute outputs ---
        # 1. Distance to reward zone (time-varying, discretized)
        dist = compute_distance_to_reward_zone(pos_trial, zone_start, zone_end)
        dist_disc = discretize_distance(dist)

        # 2. Absolute position (time-varying, discretized)
        pos_disc = discretize_position(pos_trial)

        # 3. Speed (time-varying, discretized)
        speed_disc = discretize_speed(np.abs(speed_trial))

        # 4. Lick (time-varying, binary)
        lick_binary = (lick_trial > 0).astype(np.int64)

        # 5. Reward zone location (per-trial)
        # 6. Reward outcome (per-trial)

        # Stack time-varying outputs: (n_output, n_timepoints)
        # Time-varying: dist, pos, speed, lick (4 vars, n_timepoints each)
        # Per-trial: reward_zone, reward_outcome (2 vars, scalar each)
        # For per-trial outputs, broadcast to length 1
        output = np.stack([
            dist_disc,
            pos_disc,
            speed_disc,
            lick_binary,
            np.full(n_tp, zone_idx, dtype=np.int64),
            np.full(n_tp, rewarded, dtype=np.int64),
        ], axis=0)  # (6, n_timepoints)

        # --- Compute inputs ---
        # 1. Time from start of trial (seconds, continuous, time-varying)
        time_from_start = np.arange(n_tp, dtype=np.float32) * dt

        # 2. Environment type (binary, per-trial)
        # 3. Trial number (continuous, per-trial)
        # 4. Previous trial outcome (binary, per-trial)

        input_data = np.stack([
            time_from_start,
            np.full(n_tp, env_binary, dtype=np.float32),
            np.full(n_tp, t, dtype=np.float32),
            np.full(n_tp, prev_rewarded, dtype=np.float32),
        ], axis=0)  # (4, n_timepoints)

        neural_trials.append(neural)
        input_trials.append(input_data)
        output_trials.append(output)

        # Update previous trial outcome for next iteration
        prev_rewarded = rewarded

    info = {
        'mouse_id': mouse_id,
        'exp_day': exp_day,
        'scene': scene,
        'frame_rate': frame_rate,
        'n_neurons': n_neurons,
        'n_trials': len(neural_trials),
        'n_interneurons_excluded': int(n_interneurons),
    }

    return neural_trials, input_trials, output_trials, n_neurons, info


def main():
    # Collect all sessions
    all_neural = []
    all_input = []
    all_output = []
    all_subject_idx = []
    all_brain_region_idx = []
    all_session_info = []

    subjects = sorted(SESSIONS_DICT.keys(), key=lambda x: int(x[1:]))

    for subj in subjects:
        subj_dir = os.path.join(DATA_DIR, f'sub-{subj}')
        if not os.path.isdir(subj_dir):
            print(f"Warning: directory {subj_dir} not found, skipping")
            continue

        # Get available NWB files
        nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])

        for nwb_file in nwb_files:
            # Extract session day from filename: sub-m11_ses-03_behavior+ophys.nwb
            ses_str = nwb_file.split('_ses-')[1].split('_')[0]
            exp_day = int(ses_str)

            # Find matching scene from sessions dict
            scene = None
            for entry in SESSIONS_DICT[subj]:
                if entry['exp_day'] == exp_day:
                    scene = entry['scene']
                    break

            if scene is None:
                print(f"Warning: no scene found for {subj} day {exp_day}, skipping")
                continue

            nwb_path = os.path.join(subj_dir, nwb_file)
            print(f"Processing {subj} day {exp_day}: {scene}")

            neural, inp, out, n_neurons, info = process_session(
                subj, exp_day, nwb_path, scene)

            if neural is None or len(neural) < 2:
                print(f"  Skipping {subj} day {exp_day}: insufficient data")
                continue

            all_neural.append(neural)
            all_input.append(inp)
            all_output.append(out)
            all_subject_idx.append(subjects.index(subj))
            all_brain_region_idx.append(np.zeros(n_neurons, dtype=np.int64))
            all_session_info.append(info)

    # Get frame rate (should be consistent)
    frame_rates = [info['frame_rate'] for info in all_session_info]
    median_frame_rate = np.median(frame_rates)
    time_bin_ms = 1000.0 / median_frame_rate

    # Build the final data structure
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
            'environment',
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
            ['< -50cm', '-50 to -10cm', '-10 to 0cm', '0cm (in zone)', '>0 to 10cm', '10 to 50cm', '> 50cm'],
            ['0-90cm', '90-180cm', '180-270cm', '270-360cm', '360-450cm'],
            ['< 2cm/s', '2-10cm/s', '10-20cm/s', '20-40cm/s', '> 40cm/s'],
            ['no_lick', 'lick'],
            ['zone_A', 'zone_B', 'zone_C'],
            ['no_reward', 'reward'],
        ],

        'metadata': {
            'task_description': (
                'Mice navigate a 450cm virtual linear track with a hidden 50cm reward zone '
                'at one of three locations (A: 80-130cm, B: 200-250cm, C: 320-370cm). '
                'Reward zone switches across days; two environments (ENV1, ENV2) are used. '
                'Reward is operantly delivered for licking in the zone, randomly omitted ~15% of trials.'
            ),
            'time_bin_size': time_bin_ms,
            'temporal_alignment_event': 'start of trial (first frame of track entry)',
            'off_start': 0.0,
            'off_end': None,  # Variable across trials
            'track_length_cm': 450,
            'reward_zone_A': [80, 130],
            'reward_zone_B': [200, 250],
            'reward_zone_C': [320, 370],
            'imaging_indicator': 'GCaMP7f',
            'brain_region': 'hippocampus CA1',
            'session_info': all_session_info,
        }
    }

    # Save
    output_path = '/app/converted_data.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"\nSaved converted data to {output_path}")
    print(f"  Sessions: {len(all_neural)}")
    print(f"  Subjects: {len(subjects)}")
    total_trials = sum(len(s) for s in all_neural)
    print(f"  Total trials: {total_trials}")
    total_neurons = sum(info['n_neurons'] for info in all_session_info)
    print(f"  Total neurons: {total_neurons}")


if __name__ == '__main__':
    main()
