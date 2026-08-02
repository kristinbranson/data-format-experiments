#!/usr/bin/env python3
"""
Convert Allen Brain Observatory Visual Behavior 2P data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]

The script loads NWB experiment files, extracts neural events, behavioral data,
and trial structure, then saves in the standardized decoder format.
"""

import argparse
import os
import sys
import time
import pickle
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import h5py

warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# Constants
# ============================================================================
TARGET_RATE_HZ = 30.0  # Paper: "linearly interpolating onto a consistent set of 30hz timestamps"
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ  # ~33.33 ms

DATA_DIR = Path('data/visual-behavior-ophys-1.1.0')
NWB_DIR = DATA_DIR / 'behavior_ophys_experiments'
META_DIR = DATA_DIR / 'project_metadata'

# Session types to include (active behavior only, exclude passive)
ACTIVE_SESSION_TYPES = [
    'OPHYS_1_images_A', 'OPHYS_3_images_A',
    'OPHYS_4_images_B', 'OPHYS_6_images_B',
]

# ============================================================================
# Helper Functions
# ============================================================================

def interpolate_to_regular_grid(timestamps, data, target_timestamps):
    """Linearly interpolate data from irregular timestamps to regular grid.

    Args:
        timestamps: (n_orig,) original timestamps
        data: (n_orig,) or (n_orig, n_features) data array
        target_timestamps: (n_target,) target timestamps

    Returns:
        interpolated data of shape (n_target,) or (n_target, n_features)
    """
    if data.ndim == 1:
        return np.interp(target_timestamps, timestamps, data)
    else:
        result = np.zeros((len(target_timestamps), data.shape[1]), dtype=np.float32)
        for i in range(data.shape[1]):
            result[:, i] = np.interp(target_timestamps, timestamps, data[:, i])
        return result


def compute_percentile_bins(values, n_bins=5):
    """Compute bin edges from equal percentiles, ignoring NaN.

    Returns:
        bin_edges: array of n_bins+1 edges
    """
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.linspace(0, 1, n_bins + 1)
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    # Ensure unique edges (handle constant values)
    # Add small epsilon to make edges strictly increasing
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-10
    return edges


def digitize_to_bins(values, bin_edges):
    """Digitize values into bins defined by bin_edges.

    Returns integer bin indices from 0 to n_bins-1.
    NaN values get bin 0.
    """
    n_bins = len(bin_edges) - 1
    binned = np.digitize(values, bin_edges[1:-1])  # 0 to n_bins-1
    binned = np.clip(binned, 0, n_bins - 1)
    # NaN handling
    binned[np.isnan(values)] = 0
    return binned.astype(np.int64)


def interpolate_nans(arr):
    """Linearly interpolate NaN values in a 1D array."""
    nans = np.isnan(arr)
    if not nans.any():
        return arr.copy()
    if nans.all():
        return np.zeros_like(arr)
    result = arr.copy()
    x = np.arange(len(arr))
    result[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return result


# ============================================================================
# NWB Loading Functions
# ============================================================================

def load_experiment_data(nwb_path, experiment_id):
    """Load all needed data from a single NWB file.

    Returns a dict with all extracted data, or None if loading fails.
    """
    try:
        with h5py.File(nwb_path, 'r') as f:
            # --- Cell specimen table ---
            cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
            valid_roi = cell_table['valid_roi'][()].astype(bool)
            cell_specimen_ids = cell_table['cell_specimen_id'][()]

            n_valid = valid_roi.sum()
            if n_valid == 0:
                print(f"  WARNING: No valid ROIs in experiment {experiment_id}, skipping")
                return None

            # --- Ophys timestamps ---
            ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][()]

            # --- Neural events (filtered by valid_roi) ---
            events_data = f['processing']['ophys']['event_detection']['data'][()]
            # events_data shape: (n_timepoints, n_all_cells)
            events_valid = events_data[:, valid_roi]  # (n_timepoints, n_valid_cells)

            # --- Trials ---
            trials_grp = f['intervals']['trials']
            n_trials = len(trials_grp['start_time'][()])
            trials = {
                'start_time': trials_grp['start_time'][()],
                'stop_time': trials_grp['stop_time'][()],
                'change_time': trials_grp['change_time'][()],
                'go': trials_grp['go'][()].astype(bool),
                'catch': trials_grp['catch'][()].astype(bool),
                'aborted': trials_grp['aborted'][()].astype(bool),
                'auto_rewarded': trials_grp['auto_rewarded'][()].astype(bool),
                'hit': trials_grp['hit'][()].astype(bool),
                'miss': trials_grp['miss'][()].astype(bool),
                'false_alarm': trials_grp['false_alarm'][()].astype(bool),
                'correct_reject': trials_grp['correct_reject'][()].astype(bool),
                'is_change': trials_grp['is_change'][()].astype(bool),
                'initial_image_name': trials_grp['initial_image_name'][()],
                'change_image_name': trials_grp['change_image_name'][()],
            }
            # Decode bytes
            if isinstance(trials['initial_image_name'][0], bytes):
                trials['initial_image_name'] = np.array([x.decode() for x in trials['initial_image_name']])
                trials['change_image_name'] = np.array([x.decode() for x in trials['change_image_name']])

            # --- Stimulus presentations ---
            # Find the natural images presentation table
            stim_key = None
            for k in f['intervals'].keys():
                if k != 'trials' and 'spontaneous' not in k.lower() and 'movie' not in k.lower():
                    stim_key = k
                    break

            if stim_key is None:
                print(f"  WARNING: No stimulus presentations found in {experiment_id}, skipping")
                return None

            stim = f['intervals'][stim_key]
            stim_data = {
                'start_time': stim['start_time'][()],
                'stop_time': stim['stop_time'][()],
                'image_name': stim['image_name'][()],
                'is_change': stim['is_change'][()].astype(bool),
                'omitted': stim['omitted'][()],
            }
            if isinstance(stim_data['image_name'][0], bytes):
                stim_data['image_name'] = np.array([x.decode() for x in stim_data['image_name']])

            # Handle omitted field (might be float)
            if stim_data['omitted'].dtype == float:
                stim_data['omitted'] = stim_data['omitted'].astype(bool)

            # --- Running speed ---
            running_speed = f['processing']['running']['speed']['data'][()]
            running_ts = f['processing']['running']['speed']['timestamps'][()]

            # --- Pupil tracking ---
            pupil_area = None
            pupil_ts = None
            likely_blink = None
            try:
                pupil_tracking = f['acquisition']['EyeTracking']['pupil_tracking']
                pupil_area = pupil_tracking['area'][()]
                pupil_ts = pupil_tracking['timestamps'][()]
                likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
            except (KeyError, Exception):
                print(f"  WARNING: No pupil tracking data in {experiment_id}")

            return {
                'experiment_id': experiment_id,
                'valid_roi': valid_roi,
                'cell_specimen_ids': cell_specimen_ids[valid_roi],
                'n_valid_cells': n_valid,
                'ophys_ts': ophys_ts,
                'events': events_valid,
                'trials': trials,
                'stim': stim_data,
                'running_speed': running_speed,
                'running_ts': running_ts,
                'pupil_area': pupil_area,
                'pupil_ts': pupil_ts,
                'likely_blink': likely_blink,
            }
    except Exception as e:
        print(f"  ERROR loading {experiment_id}: {e}")
        return None


# ============================================================================
# Trial Processing
# ============================================================================

def get_image_at_timepoints(timepoints, stim_data, all_image_names):
    """For each timepoint, determine which image is currently being shown.

    Returns array of image indices (into all_image_names) for each timepoint.
    During gray screen periods, returns the last shown image.
    For omitted flashes, continues with the previous image.
    """
    n_tp = len(timepoints)
    image_idx = np.zeros(n_tp, dtype=np.int64)

    # Build a sorted list of (start_time, image_name) for non-omitted stimuli
    non_omitted = ~stim_data['omitted']
    stim_starts = stim_data['start_time'][non_omitted]
    stim_names = stim_data['image_name'][non_omitted]

    # Map image names to indices
    name_to_idx = {name: i for i, name in enumerate(all_image_names)}

    # For each timepoint, find the most recent stimulus onset
    # Use searchsorted for efficiency
    insert_idx = np.searchsorted(stim_starts, timepoints, side='right') - 1

    for i in range(n_tp):
        if insert_idx[i] >= 0:
            img_name = stim_names[insert_idx[i]]
            image_idx[i] = name_to_idx.get(img_name, 0)
        else:
            image_idx[i] = 0  # Before first stimulus

    return image_idx


def get_image_change_at_timepoints(timepoints, stim_data):
    """For each timepoint, determine if an image change is occurring.

    Returns binary array: 1 during the first flash after a change, 0 otherwise.
    """
    n_tp = len(timepoints)
    change_signal = np.zeros(n_tp, dtype=np.int64)

    # Get change stimulus presentations (non-omitted, is_change=True)
    change_mask = stim_data['is_change'] & ~stim_data['omitted']
    change_starts = stim_data['start_time'][change_mask]
    change_stops = stim_data['stop_time'][change_mask]

    # For each change, mark timepoints within the 750ms image presentation interval
    for cs, ce in zip(change_starts, change_stops):
        # Mark the full image interval (stimulus + gray) as change
        # Use 750ms window from change start
        mask = (timepoints >= cs) & (timepoints < cs + 0.75)
        change_signal[mask] = 1

    return change_signal


def process_single_experiment(raw_data, exp_meta, target_rate=TARGET_RATE_HZ):
    """Process a single experiment into trial-segmented format.

    Args:
        raw_data: dict from load_experiment_data
        exp_meta: dict with 'targeted_structure', 'mouse_id' etc.
        target_rate: target sampling rate in Hz

    Returns:
        dict with processed trial data, or None if processing fails.
    """
    trials = raw_data['trials']

    # Filter trials: keep Go and Catch, exclude Aborted and Auto-rewarded
    valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < 2:
        print(f"  WARNING: Only {len(valid_indices)} valid trials in experiment {raw_data['experiment_id']}, skipping")
        return None

    # --- Resample neural events to target rate ---
    ophys_ts = raw_data['ophys_ts']
    events = raw_data['events']  # (n_timepoints, n_cells)

    # Create regular timestamp grid at target rate
    t_start = ophys_ts[0]
    t_end = ophys_ts[-1]
    dt = 1.0 / target_rate
    regular_ts = np.arange(t_start, t_end, dt)

    # Interpolate events to regular grid
    events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
    # Ensure non-negative (events should be >= 0)
    events_resampled = np.maximum(events_resampled, 0)

    # --- Resample running speed ---
    running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])

    # --- Resample pupil area ---
    pupil_resampled = None
    if raw_data['pupil_area'] is not None:
        pupil_area = raw_data['pupil_area'].copy().astype(float)
        pupil_ts = raw_data['pupil_ts']
        likely_blink = raw_data['likely_blink']

        # Set blink frames to NaN
        if likely_blink is not None:
            pupil_area[likely_blink] = np.nan

        # Interpolate NaNs in pupil area before resampling
        pupil_area = interpolate_nans(pupil_area)

        # Resample to regular grid
        pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)

    # --- Get all unique image names (excluding 'omitted') ---
    all_stim_images = sorted(set(
        raw_data['stim']['image_name'][raw_data['stim']['image_name'] != 'omitted']
    ))

    # --- Process each valid trial ---
    neural_trials = []
    output_trials = []
    trial_outcomes = []

    for trial_idx in valid_indices:
        t_trial_start = trials['start_time'][trial_idx]
        t_trial_stop = trials['stop_time'][trial_idx]

        # Find regular grid indices within this trial
        trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
        trial_time_indices = np.where(trial_mask)[0]

        if len(trial_time_indices) < 3:
            continue

        trial_ts = regular_ts[trial_time_indices]
        n_tp = len(trial_ts)

        # Neural data: (n_cells, n_timepoints)
        neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)

        # Image identity at each timepoint
        image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)

        # Image change signal
        change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])

        # Running speed (already resampled)
        running_trial = running_resampled[trial_time_indices]

        # Pupil area (already resampled)
        if pupil_resampled is not None:
            pupil_trial = pupil_resampled[trial_time_indices]
        else:
            pupil_trial = np.full(n_tp, np.nan)

        # Trial outcome (static)
        if trials['hit'][trial_idx]:
            outcome = 0  # hit
        elif trials['miss'][trial_idx]:
            outcome = 1  # miss
        elif trials['false_alarm'][trial_idx]:
            outcome = 2  # false_alarm
        elif trials['correct_reject'][trial_idx]:
            outcome = 3  # correct_reject
        else:
            continue  # Unknown outcome, skip

        neural_trials.append(neural_trial)
        # Store raw values for now; discretization happens after collecting all data
        output_trials.append({
            'image_identity': image_idx,
            'image_change': change_signal,
            'running_speed': running_trial.astype(np.float32),
            'pupil_area': pupil_trial.astype(np.float32),
            'trial_outcome': outcome,
        })

    if len(neural_trials) < 2:
        print(f"  WARNING: Only {len(neural_trials)} processed trials in experiment {raw_data['experiment_id']}, skipping")
        return None

    return {
        'experiment_id': raw_data['experiment_id'],
        'neural_trials': neural_trials,
        'output_trials': output_trials,
        'n_cells': raw_data['n_valid_cells'],
        'all_image_names': all_stim_images,
        'targeted_structure': exp_meta['targeted_structure'],
        'mouse_id': str(exp_meta['mouse_id']),
    }


# ============================================================================
# Main Conversion
# ============================================================================

def get_experiment_list(sample=False):
    """Get list of active experiment IDs with metadata."""
    exp_table = pd.read_csv(META_DIR / 'ophys_experiment_table.csv')

    # Get NWB files on disk
    nwb_files = list(NWB_DIR.glob('*.nwb'))
    nwb_ids = set()
    for f in nwb_files:
        try:
            eid = int(f.stem.split('_')[-1])
            nwb_ids.add(eid)
        except ValueError:
            pass

    # Filter to on-disk, active sessions
    mask = (
        exp_table['ophys_experiment_id'].isin(nwb_ids) &
        exp_table['session_type'].isin(ACTIVE_SESSION_TYPES)
    )
    active_exps = exp_table[mask].copy()

    if sample:
        # Pick 2 sessions (preferably from different mice, different equipment)
        # Pick one single-plane and one multi-plane if possible
        single = active_exps[active_exps['equipment_name'].str.startswith('CAM')]
        if len(single) > 0:
            first = single.iloc[0]
        else:
            first = active_exps.iloc[0]

        remaining = active_exps[active_exps['mouse_id'] != first['mouse_id']]
        if len(remaining) > 0:
            second = remaining.iloc[0]
        else:
            second = active_exps.iloc[1] if len(active_exps) > 1 else first

        active_exps = active_exps[
            active_exps['ophys_experiment_id'].isin([
                first['ophys_experiment_id'],
                second['ophys_experiment_id']
            ])
        ]

    print(f"Selected {len(active_exps)} experiments from {active_exps['mouse_id'].nunique()} mice")
    return active_exps


def convert_all(exp_table, show_processing=False):
    """Convert all experiments to the target format."""

    all_neural = []
    all_input = []
    all_output = []
    all_subject_idx = []
    all_brain_region_idx = []

    subjects = sorted(exp_table['mouse_id'].unique().astype(str))
    subject_to_idx = {s: i for i, s in enumerate(subjects)}

    brain_regions = sorted(exp_table['targeted_structure'].unique())
    region_to_idx = {r: i for i, r in enumerate(brain_regions)}

    # Collect all image names across experiments to build a global mapping
    all_image_names_set = set()

    # First pass: determine global image names
    print("\n--- Pass 1: Collecting global image names ---")
    t_start_global = time.time()

    # Quick scan of a few files to get image names
    for _, row in exp_table.iterrows():
        eid = row['ophys_experiment_id']
        nwb_path = NWB_DIR / f'behavior_ophys_experiment_{eid}.nwb'
        try:
            with h5py.File(nwb_path, 'r') as f:
                for k in f['intervals'].keys():
                    if k != 'trials' and 'spontaneous' not in k.lower() and 'movie' not in k.lower():
                        img_names = f['intervals'][k]['image_name'][()]
                        if isinstance(img_names[0], bytes):
                            img_names = [x.decode() for x in img_names]
                        all_image_names_set.update([n for n in img_names if n != 'omitted'])
                        break
        except Exception as e:
            print(f"  Error scanning {eid}: {e}")

    global_image_names = sorted(all_image_names_set)
    print(f"  Global image names ({len(global_image_names)}): {global_image_names}")

    # --- Collect all running speed and pupil values for global percentile computation ---
    print("\n--- Pass 2: Collecting running speed and pupil data for percentile bins ---")
    all_running_values = []
    all_pupil_values = []

    for idx, (_, row) in enumerate(exp_table.iterrows()):
        eid = row['ophys_experiment_id']
        nwb_path = NWB_DIR / f'behavior_ophys_experiment_{eid}.nwb'
        try:
            with h5py.File(nwb_path, 'r') as f:
                running = f['processing']['running']['speed']['data'][()]
                all_running_values.append(running.astype(np.float32))

                try:
                    pupil_area = f['acquisition']['EyeTracking']['pupil_tracking']['area'][()].astype(float)
                    blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
                    pupil_area[blink] = np.nan
                    valid_pupil = pupil_area[~np.isnan(pupil_area)]
                    if len(valid_pupil) > 0:
                        all_pupil_values.append(valid_pupil.astype(np.float32))
                except (KeyError, Exception):
                    pass
        except Exception as e:
            print(f"  Error collecting stats for {eid}: {e}")

    all_running_cat = np.concatenate(all_running_values) if all_running_values else np.array([0.0])
    all_pupil_cat = np.concatenate(all_pupil_values) if all_pupil_values else np.array([0.0])

    running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
    pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)

    print(f"  Running speed bin edges: {running_bin_edges}")
    print(f"  Pupil area bin edges: {pupil_bin_edges}")
    del all_running_values, all_pupil_values, all_running_cat, all_pupil_cat

    # --- Main processing pass ---
    print("\n--- Pass 3: Processing experiments ---")

    total_trials = 0
    total_neurons = 0
    experiments_processed = 0

    for idx, (_, row) in enumerate(exp_table.iterrows()):
        eid = row['ophys_experiment_id']
        nwb_path = NWB_DIR / f'behavior_ophys_experiment_{eid}.nwb'

        t0 = time.time()
        print(f"\n[{idx+1}/{len(exp_table)}] Processing experiment {eid}...", flush=True)

        # Load raw data
        raw_data = load_experiment_data(nwb_path, eid)
        if raw_data is None:
            continue

        t_load = time.time() - t0

        # Process into trials
        exp_meta = {
            'targeted_structure': row['targeted_structure'],
            'mouse_id': row['mouse_id'],
        }
        result = process_single_experiment(raw_data, exp_meta)
        if result is None:
            continue

        t_process = time.time() - t0 - t_load

        # --- Build output arrays with global image names and discretization ---
        session_neural = []
        session_output = []
        session_input = []

        # Map local image names to global indices
        local_to_global = {}
        for local_idx, name in enumerate(result['all_image_names']):
            if name in global_image_names:
                local_to_global[local_idx] = global_image_names.index(name)
            else:
                local_to_global[local_idx] = 0

        for trial_data_neural, trial_data_out in zip(result['neural_trials'], result['output_trials']):
            n_tp = trial_data_neural.shape[1]

            # Neural
            session_neural.append(trial_data_neural)

            # Input: empty (no inputs for this task)
            session_input.append(np.zeros((0, n_tp), dtype=np.float32))

            # Output: 5 variables
            # 1. Image identity (categorical, time-varying)
            img_id_global = np.array([local_to_global.get(v, 0) for v in trial_data_out['image_identity']], dtype=np.int64)

            # 2. Image change (binary, time-varying)
            img_change = trial_data_out['image_change'].astype(np.int64)

            # 3. Running speed (discretized, time-varying)
            running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)

            # 4. Pupil diameter (discretized, time-varying)
            pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)

            # 5. Trial outcome (static)
            trial_outcome = np.array([trial_data_out['trial_outcome']], dtype=np.int64)

            # Stack outputs: (n_output, n_timepoints) for time-varying, (n_output,) for static
            # For mixed time-varying and static, we need to handle carefully
            # Time-varying outputs: shape (n_output_tv, n_timepoints)
            # Static outputs: shape (n_output_static,)
            # Combined: (n_output, n_timepoints) where static is broadcast
            output_tv = np.stack([img_id_global, img_change, running_binned, pupil_binned], axis=0)  # (4, n_tp)
            # Append trial outcome as constant across time
            outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
            output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)  # (5, n_tp)

            session_output.append(output_combined)

        all_neural.append(session_neural)
        all_input.append(session_input)
        all_output.append(session_output)
        all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])

        # Brain region index for each neuron
        region_idx = np.full(result['n_cells'], region_to_idx[row['targeted_structure']], dtype=np.int64)
        all_brain_region_idx.append(region_idx)

        total_trials += len(session_neural)
        total_neurons += result['n_cells']
        experiments_processed += 1

        t_total = time.time() - t0
        print(f"  Loaded in {t_load:.1f}s, processed in {t_process:.1f}s, total {t_total:.1f}s")
        print(f"  {result['n_cells']} neurons, {len(session_neural)} trials")

        # Show processing plots for first 2 sessions
        if show_processing and experiments_processed <= 2:
            plot_processing(raw_data, result, session_output, global_image_names,
                          running_bin_edges, pupil_bin_edges, eid)

    print(f"\n=== Conversion Summary ===")
    print(f"Experiments processed: {experiments_processed}")
    print(f"Total trials: {total_trials}")
    print(f"Total neurons: {total_neurons}")
    print(f"Subjects: {len(subjects)}")
    print(f"Brain regions: {brain_regions}")

    # Build output names and values
    output_names = ['image_identity', 'image_change', 'running_speed', 'pupil_diameter', 'trial_outcome']

    output_values = [
        global_image_names,  # image identity values
        ['no_change', 'change'],  # image change values
        [f'bin_{i}' for i in range(5)],  # running speed bins
        [f'bin_{i}' for i in range(5)],  # pupil diameter bins
        ['hit', 'miss', 'false_alarm', 'correct_reject'],  # trial outcomes
    ]

    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': [str(s) for s in subjects],
        'subject_idx': np.array(all_subject_idx, dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': all_brain_region_idx,
        'input_names': [],
        'output_names': output_names,
        'output_values': output_values,
        'metadata': {
            'task_description': 'Visual change detection task: decode image identity, image change, running speed, pupil diameter, and trial outcome from neural activity',
            'time_bin_size': TIME_BIN_MS,
            'temporal_alignment_event': 'Trial start time (first stimulus onset of trial)',
            'off_start': 0.0,
            'off_end': None,
            'target_rate_hz': TARGET_RATE_HZ,
            'neural_signal': 'calcium events (FastLZeroSpikeInference)',
            'running_speed_bin_edges': running_bin_edges.tolist(),
            'pupil_area_bin_edges': pupil_bin_edges.tolist(),
            'active_session_types': ACTIVE_SESSION_TYPES,
            'trial_filter': 'Go and Catch trials only (excluded Aborted and Auto-rewarded)',
            'valid_roi_filter': True,
        }
    }

    return data


# ============================================================================
# Visualization
# ============================================================================

def plot_processing(raw_data, result, session_output, global_image_names,
                    running_bin_edges, pupil_bin_edges, eid):
    """Plot processing steps for a single experiment."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 2, figsize=(24, 20))
    fig.suptitle(f'Processing: Experiment {eid}', fontsize=14)

    ophys_ts = raw_data['ophys_ts']
    events = raw_data['events']

    # Plot 1: Raw events (first 5 neurons, first 60s)
    ax = axes[0, 0]
    t_mask = ophys_ts < ophys_ts[0] + 60
    n_show = min(5, events.shape[1])
    for i in range(n_show):
        ax.plot(ophys_ts[t_mask], events[t_mask, i] + i*0.5, alpha=0.7, linewidth=0.5)
    ax.set_title('Raw events (first 60s, first 5 neurons)')
    ax.set_xlabel('Time (s)')

    # Plot 2: Trial structure
    ax = axes[0, 1]
    trials = raw_data['trials']
    valid = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
    go_valid = trials['go'] & valid
    catch_valid = trials['catch'] & valid

    for i in np.where(go_valid)[0][:50]:
        ax.axvspan(trials['start_time'][i], trials['stop_time'][i], alpha=0.3, color='blue')
    for i in np.where(catch_valid)[0][:50]:
        ax.axvspan(trials['start_time'][i], trials['stop_time'][i], alpha=0.3, color='red')
    ax.set_title('Trial structure (blue=Go, red=Catch)')
    ax.set_xlim(trials['start_time'][valid][0], trials['start_time'][valid][min(49, valid.sum()-1)])

    # Plot 3: Running speed (raw vs resampled)
    ax = axes[1, 0]
    t0 = raw_data['running_ts'][0]
    rmask = raw_data['running_ts'] < t0 + 60
    ax.plot(raw_data['running_ts'][rmask], raw_data['running_speed'][rmask], alpha=0.5, linewidth=0.5, label='Raw (60Hz)')
    ax.set_title('Running speed (first 60s)')
    ax.set_ylabel('cm/s')
    ax.legend()

    # Plot 4: Pupil area
    ax = axes[1, 1]
    if raw_data['pupil_area'] is not None:
        pmask = raw_data['pupil_ts'] < raw_data['pupil_ts'][0] + 60
        ax.plot(raw_data['pupil_ts'][pmask], raw_data['pupil_area'][pmask], alpha=0.5, linewidth=0.5)
        ax.set_title('Pupil area (first 60s)')
    else:
        ax.set_title('No pupil data')

    # Plot 5-6: Sample trials (neural + outputs)
    for t_idx in range(min(2, len(result['neural_trials']))):
        trial_neural = result['neural_trials'][t_idx]
        trial_output = session_output[t_idx]

        ax = axes[2 + t_idx, 0]
        n_show = min(10, trial_neural.shape[0])
        for i in range(n_show):
            ax.plot(trial_neural[i] + i*0.3, alpha=0.7, linewidth=0.5)
        ax.set_title(f'Trial {t_idx}: Neural events ({trial_neural.shape[0]} neurons)')

        ax = axes[2 + t_idx, 1]
        output_names = ['img_id', 'img_change', 'running', 'pupil', 'outcome']
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        for o_idx in range(min(4, trial_output.shape[0])):
            ax.plot(trial_output[o_idx] / max(trial_output[o_idx].max(), 1) + o_idx,
                   alpha=0.7, linewidth=1, color=colors[o_idx], label=output_names[o_idx])
        ax.legend(fontsize=8)
        ax.set_title(f'Trial {t_idx}: Outputs')

    # Plot 7: Running speed histogram with bin edges
    ax = axes[4, 0]
    all_running = np.concatenate([t['running_speed'] for t in result['output_trials']])
    ax.hist(all_running, bins=100, alpha=0.7)
    for edge in running_bin_edges:
        ax.axvline(edge, color='red', linestyle='--', alpha=0.5)
    ax.set_title('Running speed distribution + bin edges')

    # Plot 8: Pupil histogram with bin edges
    ax = axes[4, 1]
    all_pupil = np.concatenate([t['pupil_area'] for t in result['output_trials']])
    valid_pupil = all_pupil[~np.isnan(all_pupil)]
    if len(valid_pupil) > 0:
        ax.hist(valid_pupil, bins=100, alpha=0.7)
        for edge in pupil_bin_edges:
            ax.axvline(edge, color='red', linestyle='--', alpha=0.5)
    ax.set_title('Pupil area distribution + bin edges')

    # Plot 9: Image identity over time for one trial
    ax = axes[5, 0]
    if len(result['output_trials']) > 0:
        img_ids = result['output_trials'][0]['image_identity']
        ax.step(range(len(img_ids)), img_ids, where='post')
        ax.set_title('Trial 0: Image identity over time')
        ax.set_ylabel('Image index')

    # Plot 10: Trial outcome distribution
    ax = axes[5, 1]
    outcomes = [t['trial_outcome'] for t in result['output_trials']]
    outcome_names = ['hit', 'miss', 'FA', 'CR']
    counts = [outcomes.count(i) for i in range(4)]
    ax.bar(outcome_names, counts)
    ax.set_title('Trial outcome distribution')

    plt.tight_layout()
    plt.savefig(f'processing_{eid}.png', dpi=100)
    plt.close()
    print(f"  Saved processing plot: processing_{eid}.png")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Convert Visual Behavior 2P data')
    parser.add_argument('output', type=str, help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    args = parser.parse_args()

    if args.sample:
        args.full = False

    print(f"=== Visual Behavior 2P Data Conversion ===")
    print(f"Mode: {'sample (2 sessions)' if args.sample else 'full'}")
    print(f"Output: {args.output}")
    print(f"Show processing: {args.show_processing}")

    t_total_start = time.time()

    # Get experiment list
    exp_table = get_experiment_list(sample=args.sample)

    # Convert
    data = convert_all(exp_table, show_processing=args.show_processing)

    # Save
    print(f"\nSaving to {args.output}...")
    t_save = time.time()
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Saved {file_size:.1f} MB in {time.time() - t_save:.1f}s")

    total_time = time.time() - t_total_start
    print(f"\nTotal conversion time: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == '__main__':
    main()
