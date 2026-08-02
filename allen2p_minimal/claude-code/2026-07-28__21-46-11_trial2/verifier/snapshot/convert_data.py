"""
Convert Allen Brain Observatory Visual Behavior 2P data to decoder format.

Usage:
    python convert_data.py [--sample] [--output OUTPUT_PATH]

The script loads NWB files from the Visual Behavior 2P dataset and converts
them into the standardized decoder format.

Data selection (matching reference papers):
- Active behavior sessions only (no passive viewing)
- Go and Catch trials only (exclude Aborted and Auto-rewarded)
- Neural data: detected calcium events (not raw DFF)
- Temporally aligned to ophys timestamps

Decoder outputs:
1. Image identity (categorical, time-varying)
2. Image change (binary, time-varying)
3. Running speed (5 percentile bins, time-varying)
4. Pupil diameter (5 percentile bins, time-varying)
5. Trial outcome (static per-trial: hit/miss/false_alarm/correct_reject)
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import h5py
from collections import defaultdict

# Constants
DATA_DIR = 'data/visual-behavior-ophys-1.1.0'
NWB_DIR = os.path.join(DATA_DIR, 'behavior_ophys_experiments')
METADATA_DIR = os.path.join(DATA_DIR, 'project_metadata')

# Trial outcome mapping
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']


def load_experiment_table():
    """Load experiment metadata table and filter for available NWB files."""
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))

    # Get available NWB file IDs
    nwb_files = [f for f in os.listdir(NWB_DIR) if f.endswith('.nwb')]
    nwb_ids = [int(f.split('_')[-1].split('.')[0]) for f in nwb_files]

    # Filter for experiments with NWB files
    exp_table = exp_table[exp_table['ophys_experiment_id'].isin(nwb_ids)]

    # Filter for active behavior only
    exp_table = exp_table[exp_table['passive'] == False]

    print(f"Active experiments with NWB files: {len(exp_table)}")
    print(f"  Unique mice: {exp_table['mouse_id'].nunique()}")
    print(f"  Unique sessions: {exp_table['ophys_session_id'].nunique()}")
    print(f"  Cre lines: {exp_table['cre_line'].unique()}")
    print(f"  Structures: {exp_table['targeted_structure'].unique()}")
    print(f"  Projects: {exp_table['project_code'].unique()}")

    return exp_table


def load_nwb_data(nwb_path):
    """Load all relevant data from a single NWB file."""
    data = {}

    with h5py.File(nwb_path, 'r') as f:
        # Ophys timestamps (common timebase for neural data)
        ophys_ts = f['processing/ophys/dff/traces/timestamps'][:]
        data['ophys_timestamps'] = ophys_ts

        # Neural data: detected calcium events
        events = f['processing/ophys/event_detection/data'][:]
        events_ts = f['processing/ophys/event_detection/timestamps'][:]
        data['events'] = events  # shape: (n_timepoints, n_cells)
        data['events_timestamps'] = events_ts

        # DFF traces (for fallback / comparison)
        dff = f['processing/ophys/dff/traces/data'][:]
        data['dff'] = dff  # shape: (n_timepoints, n_cells)

        # Cell specimen table
        cst = f['processing/ophys/image_segmentation/cell_specimen_table']
        data['cell_specimen_ids'] = cst['cell_specimen_id'][:]
        data['valid_roi'] = cst['valid_roi'][:]

        # Stimulus presentations
        stim_keys = [k for k in f['intervals'] if 'Natural' in k or 'natural_scene' in k.lower()]
        if len(stim_keys) == 0:
            stim_keys = [k for k in f['intervals'] if k not in ['trials', 'spontaneous_presentations', 'natural_movie_one_presentations']]
        stim_key = stim_keys[0]
        stim = f[f'intervals/{stim_key}']

        data['stim_start_times'] = stim['start_time'][:]
        data['stim_stop_times'] = stim['stop_time'][:]
        data['stim_image_names'] = np.array([s.decode() if isinstance(s, bytes) else s for s in stim['image_name'][:]])
        data['stim_is_change'] = stim['is_change'][:]
        data['stim_omitted'] = stim['omitted'][:]

        # Trials
        trials = f['intervals/trials']
        data['trial_start_times'] = trials['start_time'][:]
        data['trial_stop_times'] = trials['stop_time'][:]
        data['trial_change_times'] = trials['change_time'][:]
        data['trial_aborted'] = trials['aborted'][:]
        data['trial_auto_rewarded'] = trials['auto_rewarded'][:]
        data['trial_go'] = trials['go'][:]
        data['trial_catch'] = trials['catch'][:]
        data['trial_hit'] = trials['hit'][:]
        data['trial_miss'] = trials['miss'][:]
        data['trial_false_alarm'] = trials['false_alarm'][:]
        data['trial_correct_reject'] = trials['correct_reject'][:]
        data['trial_is_change'] = trials['is_change'][:]
        data['trial_change_image'] = np.array([s.decode() if isinstance(s, bytes) else s for s in trials['change_image_name'][:]])
        data['trial_initial_image'] = np.array([s.decode() if isinstance(s, bytes) else s for s in trials['initial_image_name'][:]])

        # Running speed
        data['running_speed'] = f['processing/running/speed/data'][:]
        data['running_timestamps'] = f['processing/running/speed/timestamps'][:]

        # Pupil tracking
        if 'acquisition/EyeTracking/pupil_tracking' in f:
            pupil = f['acquisition/EyeTracking/pupil_tracking']
            # Use width as pupil diameter (fitted ellipse width)
            data['pupil_width'] = pupil['width'][:]
            data['pupil_area'] = pupil['area'][:]
            eye_ts = f['acquisition/EyeTracking/eye_tracking/timestamps'][:]
            data['eye_timestamps'] = eye_ts
            data['likely_blink'] = f['acquisition/EyeTracking/likely_blink/data'][:]
        else:
            data['pupil_width'] = None
            data['eye_timestamps'] = None
            data['likely_blink'] = None

        # Imaging rate
        data['imaging_rate'] = f['general/optophysiology/imaging_plane_1/imaging_rate'][()]

    return data


def interpolate_to_ophys(signal, signal_ts, ophys_ts):
    """Interpolate a behavioral signal to ophys timestamps."""
    return np.interp(ophys_ts, signal_ts, signal)


def get_trial_mask(trial_idx, nwb_data):
    """Check if a trial should be included (Go or Catch, not Aborted or Auto-rewarded)."""
    if nwb_data['trial_aborted'][trial_idx]:
        return False
    if nwb_data['trial_auto_rewarded'][trial_idx]:
        return False
    if not (nwb_data['trial_go'][trial_idx] or nwb_data['trial_catch'][trial_idx]):
        return False
    return True


def get_trial_outcome(trial_idx, nwb_data):
    """Get trial outcome as integer index into OUTCOME_NAMES."""
    if nwb_data['trial_hit'][trial_idx]:
        return 0  # hit
    elif nwb_data['trial_miss'][trial_idx]:
        return 1  # miss
    elif nwb_data['trial_false_alarm'][trial_idx]:
        return 2  # false_alarm
    elif nwb_data['trial_correct_reject'][trial_idx]:
        return 3  # correct_reject
    else:
        return -1  # unknown


def process_experiment(nwb_data, all_image_names, running_percentiles=None, pupil_percentiles=None):
    """Process a single experiment into trial-segmented data.

    Returns:
        neural_trials: list of (n_neurons, n_timepoints) arrays
        output_trials: list of (n_output, n_timepoints) or (n_output,) arrays
        trial_outcomes: list of outcome indices
        running_at_ophys: running speed interpolated to ophys timestamps
        pupil_at_ophys: pupil diameter interpolated to ophys timestamps
    """
    ophys_ts = nwb_data['ophys_timestamps']

    # Use calcium events as neural data (transpose to n_cells x n_timepoints)
    # Events timestamps should match ophys timestamps
    events = nwb_data['events']  # (n_timepoints, n_cells)

    # Filter for valid ROIs only
    valid = nwb_data['valid_roi']
    events = events[:, valid]

    n_timepoints_total, n_neurons = events.shape

    if n_neurons == 0:
        return None, None, None, None, None, None

    # Interpolate running speed to ophys timestamps
    running_at_ophys = interpolate_to_ophys(
        nwb_data['running_speed'],
        nwb_data['running_timestamps'],
        ophys_ts
    )

    # Interpolate pupil to ophys timestamps
    if nwb_data['pupil_width'] is not None and nwb_data['eye_timestamps'] is not None:
        pupil_raw = nwb_data['pupil_width'].copy()
        blink_mask = nwb_data['likely_blink']
        if blink_mask is not None:
            pupil_raw[blink_mask] = np.nan

        # Interpolate pupil, handling NaN by forward/backward fill first
        valid_mask = ~np.isnan(pupil_raw)
        if np.sum(valid_mask) > 10:
            pupil_at_ophys = np.interp(
                ophys_ts,
                nwb_data['eye_timestamps'][valid_mask],
                pupil_raw[valid_mask]
            )
        else:
            pupil_at_ophys = np.full(len(ophys_ts), np.nan)
    else:
        pupil_at_ophys = np.full(len(ophys_ts), np.nan)

    # Build image identity time series at ophys resolution
    # For each ophys timepoint, find the current image being shown
    image_at_ophys = np.full(len(ophys_ts), -1, dtype=int)
    stim_starts = nwb_data['stim_start_times']
    stim_stops = nwb_data['stim_stop_times']
    stim_names = nwb_data['stim_image_names']
    stim_omitted = nwb_data['stim_omitted']

    # Map image names to indices
    image_name_to_idx = {name: idx for idx, name in enumerate(all_image_names)}

    for i in range(len(stim_starts)):
        if stim_omitted[i] == 1.0:
            continue  # Skip omitted stimuli (gray screen)
        img_name = stim_names[i]
        if img_name == 'omitted':
            continue
        if img_name not in image_name_to_idx:
            continue
        img_idx = image_name_to_idx[img_name]
        # Find ophys frames during this stimulus
        mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
        image_at_ophys[mask] = img_idx

    # Build image change time series
    is_change_at_ophys = np.zeros(len(ophys_ts), dtype=int)
    stim_is_change = nwb_data['stim_is_change']
    for i in range(len(stim_starts)):
        if stim_is_change[i] == 1.0:
            mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
            is_change_at_ophys[mask] = 1

    # Segment into trials
    neural_trials = []
    output_trials = []
    trial_outcomes = []

    n_trials_total = len(nwb_data['trial_start_times'])

    for trial_idx in range(n_trials_total):
        if not get_trial_mask(trial_idx, nwb_data):
            continue

        outcome = get_trial_outcome(trial_idx, nwb_data)
        if outcome == -1:
            continue

        t_start = nwb_data['trial_start_times'][trial_idx]
        t_stop = nwb_data['trial_stop_times'][trial_idx]

        # Find ophys frames in this trial
        frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
        frame_indices = np.where(frame_mask)[0]

        if len(frame_indices) < 2:
            continue

        # Neural data for this trial: (n_neurons, n_timepoints)
        trial_neural = events[frame_indices, :].T  # (n_neurons, n_timepoints)

        # Image identity for this trial
        trial_image = image_at_ophys[frame_indices]

        # During inter-stimulus intervals (gray screen), image_at_ophys = -1
        # We need to fill these with the last shown image for continuity
        # The "image identity of the image presented during the non-grey screen"
        # For gray screen periods, carry forward the last image
        last_img = -1
        for k in range(len(trial_image)):
            if trial_image[k] >= 0:
                last_img = trial_image[k]
            elif last_img >= 0:
                trial_image[k] = last_img
        # If trial starts before first stim, fill with first image in trial
        if trial_image[0] < 0:
            first_valid = np.where(trial_image >= 0)[0]
            if len(first_valid) > 0:
                trial_image[:first_valid[0]] = trial_image[first_valid[0]]
            else:
                # No stimulus in this trial, skip
                continue

        # Image change for this trial
        trial_change = is_change_at_ophys[frame_indices]

        # Running speed for this trial (will be binned later)
        trial_running = running_at_ophys[frame_indices]

        # Pupil for this trial (will be binned later)
        trial_pupil = pupil_at_ophys[frame_indices]

        # Stack time-varying outputs: image_identity, image_change, running, pupil
        # Trial outcome is static
        # Output shape: (n_output, n_timepoints) for time-varying, (n_output,) for static
        # We'll store running and pupil raw here, bin them later globally

        neural_trials.append(trial_neural.astype(np.float32))
        output_trials.append({
            'image_identity': trial_image,
            'image_change': trial_change,
            'running_speed': trial_running,
            'pupil_diameter': trial_pupil,
            'trial_outcome': outcome,
        })
        trial_outcomes.append(outcome)

    return neural_trials, output_trials, trial_outcomes, running_at_ophys, pupil_at_ophys, events


def compute_percentile_bins(values, n_bins=5):
    """Compute percentile bin edges from values, excluding NaN."""
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.linspace(0, 1, n_bins + 1)
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    return edges


def digitize_to_bins(values, edges):
    """Digitize values into bins defined by edges. Returns 0-indexed bin indices."""
    bins = np.digitize(values, edges[1:-1])  # bins: 0 to n_bins-1
    return bins


def convert_data(sample_mode=False, output_path='converted_data.pkl'):
    """Main conversion function."""

    print("=" * 60)
    print("Allen Visual Behavior 2P Data Conversion")
    print("=" * 60)

    # Load experiment table
    exp_table = load_experiment_table()

    if sample_mode:
        # Use a small subset for testing: 2 mice, ~5 experiments
        mice = sorted(exp_table['mouse_id'].unique())
        # Pick mice with decent cell counts from different cre lines
        sample_mice = []
        for cre in exp_table['cre_line'].unique():
            cre_mice = exp_table[exp_table['cre_line'] == cre]['mouse_id'].unique()
            if len(cre_mice) > 0:
                sample_mice.append(cre_mice[0])
            if len(sample_mice) >= 3:
                break
        exp_table = exp_table[exp_table['mouse_id'].isin(sample_mice)]
        # Limit to max 3 experiments per mouse
        limited = []
        for mid in sample_mice:
            mouse_exps = exp_table[exp_table['mouse_id'] == mid].head(3)
            limited.append(mouse_exps)
        exp_table = pd.concat(limited)
        print(f"\nSAMPLE MODE: Using {len(exp_table)} experiments from {len(sample_mice)} mice")

    # Collect all image names across all experiments
    print("\nCollecting image names...")
    all_image_names_set = set()
    for _, row in exp_table.iterrows():
        eid = row['ophys_experiment_id']
        nwb_path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{eid}.nwb')
        with h5py.File(nwb_path, 'r') as f:
            stim_keys = [k for k in f['intervals'] if 'Natural' in k]
            if stim_keys:
                names = f[f'intervals/{stim_keys[0]}/image_name'][:]
                for n in names:
                    n_str = n.decode() if isinstance(n, bytes) else n
                    if n_str != 'omitted':
                        all_image_names_set.add(n_str)

    all_image_names = sorted(all_image_names_set)
    print(f"Found {len(all_image_names)} unique images: {all_image_names}")

    # First pass: collect all running and pupil values for global percentile binning
    print("\nFirst pass: collecting behavioral statistics for binning...")
    all_running = []
    all_pupil = []

    for idx, (_, row) in enumerate(exp_table.iterrows()):
        eid = row['ophys_experiment_id']
        nwb_path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{eid}.nwb')
        nwb_data = load_nwb_data(nwb_path)

        # Interpolate to ophys timestamps
        running = interpolate_to_ophys(
            nwb_data['running_speed'],
            nwb_data['running_timestamps'],
            nwb_data['ophys_timestamps']
        )
        all_running.append(running)

        if nwb_data['pupil_width'] is not None:
            pupil_raw = nwb_data['pupil_width'].copy()
            if nwb_data['likely_blink'] is not None:
                pupil_raw[nwb_data['likely_blink']] = np.nan
            valid_mask = ~np.isnan(pupil_raw)
            if np.sum(valid_mask) > 10:
                pupil = np.interp(
                    nwb_data['ophys_timestamps'],
                    nwb_data['eye_timestamps'][valid_mask],
                    pupil_raw[valid_mask]
                )
                all_pupil.append(pupil)

    all_running_flat = np.concatenate(all_running)
    running_edges = compute_percentile_bins(all_running_flat, n_bins=5)
    print(f"Running speed bin edges: {running_edges}")

    if all_pupil:
        all_pupil_flat = np.concatenate(all_pupil)
        pupil_edges = compute_percentile_bins(all_pupil_flat, n_bins=5)
        print(f"Pupil diameter bin edges: {pupil_edges}")
    else:
        pupil_edges = None

    # Second pass: process each experiment
    print("\nSecond pass: processing experiments...")

    neural_all = []
    input_all = []
    output_all = []
    subjects_list = []
    subject_idx_list = []
    brain_region_idx_all = []

    all_subjects = sorted(exp_table['mouse_id'].unique().astype(str))
    subject_to_idx = {s: i for i, s in enumerate(all_subjects)}

    all_brain_regions = sorted(exp_table['targeted_structure'].unique())
    region_to_idx = {r: i for i, r in enumerate(all_brain_regions)}

    session_info = []
    skipped_experiments = []

    for idx, (_, row) in enumerate(exp_table.iterrows()):
        eid = row['ophys_experiment_id']
        mouse_id = str(row['mouse_id'])
        structure = row['targeted_structure']
        cre_line = row['cre_line']
        session_type = row['session_type']
        project = row['project_code']

        print(f"  [{idx+1}/{len(exp_table)}] Experiment {eid}: mouse={mouse_id}, "
              f"structure={structure}, cre={cre_line}, session={session_type}")

        nwb_path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{eid}.nwb')
        nwb_data = load_nwb_data(nwb_path)

        result = process_experiment(nwb_data, all_image_names)
        neural_trials, raw_output_trials, trial_outcomes, running_at_ophys, pupil_at_ophys, events_full = result

        if neural_trials is None or len(neural_trials) < 2:
            print(f"    SKIPPED: insufficient valid trials ({0 if neural_trials is None else len(neural_trials)})")
            skipped_experiments.append(eid)
            continue

        n_neurons = neural_trials[0].shape[0]
        if n_neurons < 1:
            print(f"    SKIPPED: no valid neurons")
            skipped_experiments.append(eid)
            continue

        # Now bin running and pupil for this experiment's trials
        final_output_trials = []
        valid_trial_indices = []

        for t_idx, raw_out in enumerate(raw_output_trials):
            img_id = raw_out['image_identity']
            img_change = raw_out['image_change']
            running = raw_out['running_speed']
            pupil = raw_out['pupil_diameter']
            outcome = raw_out['trial_outcome']

            # Bin running speed
            running_binned = digitize_to_bins(running, running_edges)

            # Bin pupil diameter
            if pupil_edges is not None and not np.all(np.isnan(pupil)):
                pupil_binned = digitize_to_bins(pupil, pupil_edges)
            else:
                # If no pupil data, use median bin
                pupil_binned = np.full(len(pupil), 2, dtype=int)

            n_t = len(img_id)

            # Combine all outputs into integer array (categorical)
            output_combined = np.stack([
                img_id.astype(np.int64),
                img_change.astype(np.int64),
                running_binned.astype(np.int64),
                pupil_binned.astype(np.int64),
                np.full(n_t, outcome, dtype=np.int64),
            ], axis=0)  # (5, n_timepoints)

            final_output_trials.append(output_combined)
            valid_trial_indices.append(t_idx)

        if len(valid_trial_indices) < 2:
            print(f"    SKIPPED: insufficient valid trials after processing")
            skipped_experiments.append(eid)
            continue

        # Select valid neural trials
        final_neural = [neural_trials[i] for i in valid_trial_indices]

        # No decoder inputs for this task
        # But we still need input arrays - use empty arrays
        final_input = [np.zeros((0, t.shape[1]), dtype=np.float32) for t in final_neural]

        neural_all.append(final_neural)
        input_all.append(final_input)
        output_all.append(final_output_trials)
        subject_idx_list.append(subject_to_idx[mouse_id])

        # Brain region index: all neurons in this experiment have the same region
        region_idx = region_to_idx[structure]
        brain_region_idx_all.append(np.full(n_neurons, region_idx, dtype=int))

        session_info.append({
            'ophys_experiment_id': int(eid),
            'mouse_id': mouse_id,
            'cre_line': cre_line,
            'targeted_structure': structure,
            'session_type': session_type,
            'project_code': project,
            'n_neurons': n_neurons,
            'n_trials': len(final_neural),
            'imaging_rate_hz': float(nwb_data['imaging_rate']),
        })

        print(f"    OK: {n_neurons} neurons, {len(final_neural)} trials")

    # Compute time bin size (use median across sessions)
    imaging_rates = [s['imaging_rate_hz'] for s in session_info]
    median_rate = np.median(imaging_rates)
    time_bin_ms = 1000.0 / median_rate

    print(f"\n{'=' * 60}")
    print(f"Conversion Summary")
    print(f"{'=' * 60}")
    print(f"Sessions: {len(neural_all)}")
    print(f"Skipped experiments: {len(skipped_experiments)}")
    print(f"Subjects: {len(all_subjects)}")
    print(f"Brain regions: {all_brain_regions}")
    print(f"Image names: {all_image_names}")
    print(f"Median imaging rate: {median_rate:.2f} Hz")
    print(f"Time bin size: {time_bin_ms:.2f} ms")

    total_trials = sum(len(s) for s in neural_all)
    total_neurons = sum(s[0].shape[0] for s in neural_all)
    print(f"Total trials: {total_trials}")
    print(f"Total neuron-sessions: {total_neurons}")

    # Trial outcome distribution
    all_outcomes = []
    for session_outputs in output_all:
        for trial_out in session_outputs:
            all_outcomes.append(int(trial_out[4, 0]))  # trial outcome is 5th output
    outcome_counts = {name: sum(1 for o in all_outcomes if o == i) for i, name in enumerate(OUTCOME_NAMES)}
    print(f"Trial outcome distribution: {outcome_counts}")

    # Construct output dictionary
    output_names = ['image_identity', 'image_change', 'running_speed', 'pupil_diameter', 'trial_outcome']

    # Output values for each output variable
    output_values = [
        all_image_names,  # image_identity values
        ['no_change', 'change'],  # image_change values
        [f'bin_{i}' for i in range(5)],  # running_speed bins
        [f'bin_{i}' for i in range(5)],  # pupil_diameter bins
        OUTCOME_NAMES,  # trial outcome values
    ]

    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,
        'subjects': all_subjects,
        'subject_idx': np.array(subject_idx_list, dtype=int),
        'brain_regions': all_brain_regions,
        'brain_region_idx': brain_region_idx_all,
        'input_names': [],
        'output_names': output_names,
        'output_values': output_values,
        'metadata': {
            'task_description': 'Visual change detection task: mice detect changes in natural image identity. '
                              'Outputs are image identity, image change, running speed (5 bins), '
                              'pupil diameter (5 bins), and trial outcome (hit/miss/false_alarm/correct_reject).',
            'time_bin_size': time_bin_ms,
            'temporal_alignment_event': 'Ophys imaging frame timestamps',
            'off_start': None,
            'off_end': None,
            'dataset': 'Allen Brain Observatory Visual Behavior 2P',
            'neural_data_type': 'Detected calcium events',
            'running_speed_bin_edges': running_edges.tolist(),
            'pupil_diameter_bin_edges': pupil_edges.tolist() if pupil_edges is not None else None,
            'image_names': all_image_names,
            'session_info': session_info,
            'skipped_experiments': skipped_experiments,
            'trial_inclusion': 'Go and Catch trials only (Aborted and Auto-rewarded excluded)',
            'n_sessions': len(neural_all),
            'n_total_trials': total_trials,
            'outcome_distribution': outcome_counts,
        }
    }

    # Sanity checks
    print(f"\n{'=' * 60}")
    print("Sanity Checks")
    print(f"{'=' * 60}")

    # Check dimensions consistency
    for s_idx in range(len(data['neural'])):
        n_trials = len(data['neural'][s_idx])
        assert len(data['input'][s_idx]) == n_trials, f"Session {s_idx}: input trial count mismatch"
        assert len(data['output'][s_idx]) == n_trials, f"Session {s_idx}: output trial count mismatch"

        n_neurons_session = data['neural'][s_idx][0].shape[0]
        for t_idx in range(n_trials):
            nt = data['neural'][s_idx][t_idx]
            assert nt.shape[0] == n_neurons_session, f"Session {s_idx}, trial {t_idx}: neuron count mismatch"
            out = data['output'][s_idx][t_idx]
            assert out.shape[1] == nt.shape[1], f"Session {s_idx}, trial {t_idx}: timepoint mismatch"
            assert out.shape[0] == 5, f"Session {s_idx}, trial {t_idx}: expected 5 outputs"

    print("  Dimension checks: PASSED")

    # Check no NaN/Inf in neural data
    has_nan = False
    for s_idx in range(len(data['neural'])):
        for t_idx in range(len(data['neural'][s_idx])):
            if np.any(np.isnan(data['neural'][s_idx][t_idx])) or np.any(np.isinf(data['neural'][s_idx][t_idx])):
                print(f"  WARNING: NaN/Inf in neural data session {s_idx}, trial {t_idx}")
                has_nan = True
    if not has_nan:
        print("  No NaN/Inf in neural data: PASSED")

    # Check output ranges
    for s_idx in range(len(data['output'])):
        for t_idx in range(len(data['output'][s_idx])):
            out = data['output'][s_idx][t_idx]
            # Image identity should be 0..n_images-1
            assert np.all(out[0] >= 0) and np.all(out[0] < len(all_image_names)), \
                f"Session {s_idx}, trial {t_idx}: image identity out of range"
            # Image change should be 0 or 1
            assert np.all((out[1] == 0) | (out[1] == 1)), \
                f"Session {s_idx}, trial {t_idx}: image change not binary"
            # Running bins should be 0..4
            assert np.all(out[2] >= 0) and np.all(out[2] <= 4), \
                f"Session {s_idx}, trial {t_idx}: running bins out of range"
            # Pupil bins should be 0..4
            assert np.all(out[3] >= 0) and np.all(out[3] <= 4), \
                f"Session {s_idx}, trial {t_idx}: pupil bins out of range"
            # Trial outcome should be 0..3
            assert np.all(out[4] >= 0) and np.all(out[4] <= 3), \
                f"Session {s_idx}, trial {t_idx}: trial outcome out of range"
    print("  Output range checks: PASSED")

    # Check subject_idx and brain_region_idx
    assert len(data['subject_idx']) == len(data['neural'])
    assert len(data['brain_region_idx']) == len(data['neural'])
    assert all(0 <= idx < len(data['subjects']) for idx in data['subject_idx'])
    print("  Index checks: PASSED")

    # Paper comparison: number of cells per cre line
    print(f"\n  Cell counts by cre line (compare to paper):")
    cre_counts = defaultdict(int)
    for s in session_info:
        cre_counts[s['cre_line']] += s['n_neurons']
    for cre, count in sorted(cre_counts.items()):
        print(f"    {cre}: {count} neuron-sessions")

    # Save
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved successfully ({file_size:.1f} MB)")

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Visual Behavior 2P data to decoder format')
    parser.add_argument('--sample', action='store_true', help='Use sample subset for testing')
    parser.add_argument('--output', type=str, default='converted_data.pkl', help='Output file path')
    args = parser.parse_args()

    convert_data(sample_mode=args.sample, output_path=args.output)
