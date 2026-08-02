#!/usr/bin/env python3
"""
Convert Allen Brain Observatory Visual Behavior 2P data to decoder format.

References:
- Allen Brain Observatory: Visual Behavior 2P Technical Whitepaper
- "Behavioral strategy shapes activation of the Vip-Sst disinhibitory circuit in visual cortex"

Data selection:
- All active behavior ophys experiments available as NWB files
- Neural data: detected calcium events (matching paper methodology)
- Trials: Go and Catch only (exclude Aborted and Auto-rewarded)
- Temporal alignment: ophys timestamps, resampled to common ~11Hz bin size

Decoder outputs:
- Image identity (categorical, time-varying)
- Image change (binary, time-varying)
- Running speed (5 equal percentile bins, time-varying)
- Pupil diameter (5 equal percentile bins, time-varying)
- Trial outcome (static per trial: hit/miss/false_alarm/correct_reject)
"""

import sys
import os
import pickle
import warnings
import argparse
import numpy as np
import pandas as pd
from scipy import interpolate

warnings.filterwarnings('ignore')

sys.path.insert(0, '/app/code')
from allensdk.brain_observatory.behavior.behavior_ophys_experiment import BehaviorOphysExperiment

# Constants
NWB_DIR = '/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/'
METADATA_DIR = '/app/data/visual-behavior-ophys-1.1.0/project_metadata/'
COMMON_BIN_SIZE = 1.0 / 11.0  # ~91ms, matching multiplane frame rate
MIN_TRIAL_FRAMES = 3  # minimum frames per trial to include
MIN_NEURONS = 5  # minimum neurons per experiment to include


def load_experiment_table():
    """Load and filter the experiment table to active behavior sessions."""
    exp = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))

    # Get available NWB files
    nwb_ids = set()
    for f in os.listdir(NWB_DIR):
        if f.endswith('.nwb'):
            eid = int(f.replace('behavior_ophys_experiment_', '').replace('.nwb', ''))
            nwb_ids.add(eid)

    # Filter: have NWB file + active behavior only
    exp = exp[exp['ophys_experiment_id'].isin(nwb_ids)]
    exp = exp[exp['passive'] == False]

    print(f"Found {len(exp)} active behavior experiments with NWB files")
    print(f"  Project codes: {exp['project_code'].value_counts().to_dict()}")
    print(f"  Cre lines: {exp['cre_line'].value_counts().to_dict()}")
    print(f"  Mice: {exp['mouse_id'].nunique()}")
    print(f"  Brain regions: {exp['targeted_structure'].value_counts().to_dict()}")

    return exp


def resample_to_common_bins(timestamps, data, trial_start, trial_end):
    """
    Resample time series data to common time bins within a trial.

    Args:
        timestamps: original timestamps (1D array)
        data: data array, shape (n_features, n_orig_timepoints) or (n_orig_timepoints,)
        trial_start: start time of trial
        trial_end: end time of trial

    Returns:
        resampled_data: shape (n_features, n_bins) or (n_bins,)
        bin_centers: center times of bins
    """
    # Create common time bins
    bin_centers = np.arange(trial_start + COMMON_BIN_SIZE / 2, trial_end, COMMON_BIN_SIZE)
    if len(bin_centers) == 0:
        return None, None

    is_1d = data.ndim == 1
    if is_1d:
        data = data[np.newaxis, :]

    n_features = data.shape[0]
    n_bins = len(bin_centers)
    resampled = np.zeros((n_features, n_bins))

    for b, tc in enumerate(bin_centers):
        t_lo = tc - COMMON_BIN_SIZE / 2
        t_hi = tc + COMMON_BIN_SIZE / 2
        mask = (timestamps >= t_lo) & (timestamps < t_hi)
        if mask.sum() > 0:
            resampled[:, b] = data[:, mask].mean(axis=1)
        else:
            # Nearest neighbor for sparse data
            idx = np.argmin(np.abs(timestamps - tc))
            resampled[:, b] = data[:, idx]

    if is_1d:
        resampled = resampled[0]

    return resampled, bin_centers


def interpolate_to_bins(timestamps, values, bin_centers):
    """Interpolate a 1D time series to target bin centers."""
    valid = ~np.isnan(values)
    if valid.sum() < 2:
        return np.full(len(bin_centers), np.nan)

    f = interpolate.interp1d(
        timestamps[valid], values[valid],
        kind='linear', bounds_error=False, fill_value='extrapolate'
    )
    return f(bin_centers)


def get_image_identity_at_times(stim_presentations, bin_centers, image_to_idx):
    """
    Get image identity (categorical index) at each time bin.
    During grey screen (inter-stimulus), use the last shown image.
    Vectorized for performance.
    """
    # Sort and filter stimulus presentations
    stim = stim_presentations[stim_presentations['image_name'] != 'omitted'].sort_values('start_time')
    if len(stim) == 0:
        return np.zeros(len(bin_centers), dtype=int)

    stim_times = stim['start_time'].values
    stim_idx = np.array([image_to_idx.get(n, 0) for n in stim['image_name'].values])

    # For each bin center, find the most recent stimulus
    indices = np.searchsorted(stim_times, bin_centers, side='right') - 1
    result = np.where(indices >= 0, stim_idx[np.clip(indices, 0, len(stim_idx) - 1)], 0)
    return result.astype(int)


def get_image_change_at_times(stim_presentations, bin_centers):
    """
    Get binary image change indicator. Value of 1 at the time bin immediately
    after a change in image identity, otherwise 0.
    Vectorized for performance.
    """
    result = np.zeros(len(bin_centers), dtype=int)
    changes = stim_presentations[stim_presentations['is_change'] == True]
    if len(changes) == 0:
        return result

    change_times = changes['start_time'].values
    for ct in change_times:
        mask = (bin_centers >= ct) & (bin_centers < ct + COMMON_BIN_SIZE * 2)
        result[mask] = 1

    return result


def get_trial_outcome(trial):
    """Get trial outcome as categorical index."""
    if trial['hit']:
        return 0  # hit
    elif trial['miss']:
        return 1  # miss
    elif trial['false_alarm']:
        return 2  # false_alarm
    elif trial['correct_reject']:
        return 3  # correct_reject
    else:
        return -1  # unknown


def process_experiment(exp_row, sample_mode=False):
    """
    Process a single experiment (NWB file) into decoder format.

    Returns dict with neural, output data for all valid trials, or None if failed.
    """
    eid = exp_row['ophys_experiment_id']
    nwb_path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{eid}.nwb')

    try:
        dataset = BehaviorOphysExperiment.from_nwb_path(nwb_path)
    except Exception as e:
        print(f"  ERROR loading {eid}: {e}")
        return None

    # Get metadata
    meta = dataset.metadata
    mouse_id = str(meta['mouse_id'])
    targeted_structure = meta['targeted_structure']
    ophys_frame_rate = meta['ophys_frame_rate']

    # Get neural data (dF/F traces - continuous fluorescence measure)
    try:
        dff_df = dataset.dff_traces
        neural_data = np.array([row['dff'] for _, row in dff_df.iterrows()])  # (n_neurons, n_frames)
    except Exception as e:
        print(f"  ERROR getting dff_traces for {eid}: {e}")
        return None

    n_neurons = neural_data.shape[0]
    if n_neurons < MIN_NEURONS:
        print(f"  SKIP {eid}: only {n_neurons} neurons (min {MIN_NEURONS})")
        return None

    ophys_ts = dataset.ophys_timestamps

    # Get running speed
    try:
        running = dataset.running_speed
        running_ts = running['timestamps'].values
        running_speed = running['speed'].values
    except Exception as e:
        print(f"  WARNING: no running speed for {eid}: {e}")
        running_ts = None
        running_speed = None

    # Get eye tracking (pupil diameter)
    try:
        eye = dataset.eye_tracking
        eye_ts = eye['timestamps'].values
        # Compute pupil diameter as mean of width and height
        pupil_diam = ((eye['pupil_width'].values + eye['pupil_height'].values) / 2.0)
        # Set likely blinks to NaN
        likely_blink = eye['likely_blink'].values
        pupil_diam[likely_blink] = np.nan
    except Exception as e:
        print(f"  WARNING: no eye tracking for {eid}: {e}")
        eye_ts = None
        pupil_diam = None

    # Get stimulus presentations (change detection block only)
    try:
        stim = dataset.stimulus_presentations
        if 'stimulus_block_name' in stim.columns:
            stim_cd = stim[stim['stimulus_block_name'].str.contains('change_detection', na=False)]
        else:
            stim_cd = stim[stim['image_name'] != 'spontaneous']
    except Exception as e:
        print(f"  ERROR getting stimuli for {eid}: {e}")
        return None

    # Get trials
    try:
        trials = dataset.trials
    except Exception as e:
        print(f"  ERROR getting trials for {eid}: {e}")
        return None

    # Filter trials: Go or Catch, not Aborted, not Auto-rewarded
    valid_trials = trials[
        ((trials['go'] == True) | (trials['catch'] == True)) &
        (trials['aborted'] == False) &
        (trials['auto_rewarded'] == False)
    ]

    if len(valid_trials) < 2:
        print(f"  SKIP {eid}: only {len(valid_trials)} valid trials")
        return None

    # Build unique image list from this experiment
    image_names = sorted(stim_cd[
        (stim_cd['image_name'] != 'omitted') &
        (stim_cd['image_name'].notna())
    ]['image_name'].unique().tolist())

    # Process each trial
    trial_neural = []
    trial_outputs = []
    trial_count = 0

    for trial_idx, (_, trial) in enumerate(valid_trials.iterrows()):
        t_start = trial['start_time']
        t_end = trial['stop_time']

        if np.isnan(t_start) or np.isnan(t_end) or t_end <= t_start:
            continue

        # Get neural data frames in trial window
        frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_end)
        if frame_mask.sum() < MIN_TRIAL_FRAMES:
            continue

        trial_ophys_ts = ophys_ts[frame_mask]
        trial_neural_raw = neural_data[:, frame_mask]

        # Resample neural data to common bins
        neural_resampled, bin_centers = resample_to_common_bins(
            trial_ophys_ts, trial_neural_raw, t_start, t_end
        )
        if neural_resampled is None or len(bin_centers) < MIN_TRIAL_FRAMES:
            continue

        n_bins = len(bin_centers)

        # Get image identity at each bin
        img_identity = get_image_identity_at_times(stim_cd, bin_centers,
                                                    {name: i for i, name in enumerate(image_names)})

        # Get image change indicator
        img_change = get_image_change_at_times(stim_cd, bin_centers)

        # Get running speed at each bin
        if running_ts is not None and running_speed is not None:
            run_at_bins = interpolate_to_bins(running_ts, running_speed, bin_centers)
        else:
            run_at_bins = np.zeros(n_bins)

        # Get pupil diameter at each bin
        if eye_ts is not None and pupil_diam is not None:
            pupil_at_bins = interpolate_to_bins(eye_ts, pupil_diam, bin_centers)
        else:
            pupil_at_bins = np.full(n_bins, np.nan)

        # Get trial outcome
        outcome = get_trial_outcome(trial)
        if outcome == -1:
            continue

        # Store trial data
        trial_neural.append(neural_resampled)  # (n_neurons, n_bins)

        # Output: [image_identity, image_change, running_speed_raw, pupil_diam_raw, trial_outcome]
        # Running speed and pupil will be discretized later
        trial_outputs.append({
            'image_identity': img_identity,
            'image_change': img_change,
            'running_speed_raw': run_at_bins,
            'pupil_diam_raw': pupil_at_bins,
            'trial_outcome': outcome,
        })
        trial_count += 1

        if sample_mode and trial_count >= 20:
            break

    if trial_count < 2:
        print(f"  SKIP {eid}: only {trial_count} processed trials")
        return None

    print(f"  OK {eid}: {n_neurons} neurons, {trial_count} trials, "
          f"mouse={mouse_id}, region={targeted_structure}, rate={ophys_frame_rate}Hz")

    return {
        'neural': trial_neural,
        'outputs': trial_outputs,
        'mouse_id': mouse_id,
        'targeted_structure': targeted_structure,
        'n_neurons': n_neurons,
        'image_names': image_names,
        'ophys_experiment_id': eid,
        'cre_line': meta['cre_line'],
        'session_type': meta['session_type'],
        'ophys_frame_rate': ophys_frame_rate,
        'n_go_trials': int(valid_trials['go'].sum()),
        'n_catch_trials': int(valid_trials['catch'].sum()),
    }


def discretize_continuous(all_values, n_bins=5):
    """
    Compute percentile bin edges from all values, return edges.
    """
    valid = all_values[~np.isnan(all_values)]
    if len(valid) == 0:
        return np.linspace(0, 1, n_bins + 1)
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    # Ensure unique edges
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def apply_discretization(values, edges):
    """Discretize values using precomputed bin edges. Returns 0-indexed bin labels."""
    result = np.digitize(values, edges[1:-1])  # bins 0 to n_bins-1
    result = np.clip(result, 0, len(edges) - 2)
    return result


def main():
    parser = argparse.ArgumentParser(description='Convert Allen VB data to decoder format')
    parser.add_argument('--sample', action='store_true', help='Process only a few experiments for testing')
    parser.add_argument('--output', type=str, default='converted_data.pkl', help='Output file path')
    args = parser.parse_args()

    print("=" * 60)
    print("Allen Visual Behavior 2P Data Conversion")
    print("=" * 60)

    # Load experiment table
    exp_table = load_experiment_table()

    if args.sample:
        # Take a small sample: 5 experiments from different mice
        exp_table = exp_table.groupby('mouse_id').first().reset_index().head(5)
        exp_table = exp_table.sort_values('ophys_experiment_id')
        print(f"\nSAMPLE MODE: processing {len(exp_table)} experiments")

    # Process all experiments
    results = []
    for i, (_, row) in enumerate(exp_table.iterrows()):
        print(f"\nProcessing experiment {i+1}/{len(exp_table)}: {row['ophys_experiment_id']}")
        result = process_experiment(row, sample_mode=args.sample)
        if result is not None:
            results.append(result)

    print(f"\n{'=' * 60}")
    print(f"Successfully processed {len(results)} / {len(exp_table)} experiments")

    if len(results) == 0:
        print("ERROR: No experiments processed successfully!")
        sys.exit(1)

    # Collect all unique subjects, brain regions, image names
    all_subjects = sorted(set(r['mouse_id'] for r in results))
    all_regions = sorted(set(r['targeted_structure'] for r in results))

    # Build global image name list from all experiments
    all_image_names = set()
    for r in results:
        all_image_names.update(r['image_names'])
    all_image_names = sorted(all_image_names)
    global_img_to_idx = {name: i for i, name in enumerate(all_image_names)}

    print(f"\nSubjects: {len(all_subjects)}")
    print(f"Brain regions: {all_regions}")
    print(f"Image names: {all_image_names}")

    # Collect all running speed and pupil diameter values for discretization
    print("\nComputing discretization bins...")
    all_running = []
    all_pupil = []
    for r in results:
        for out in r['outputs']:
            all_running.append(out['running_speed_raw'])
            all_pupil.append(out['pupil_diam_raw'])

    all_running_flat = np.concatenate(all_running)
    all_pupil_flat = np.concatenate(all_pupil)

    running_edges = discretize_continuous(all_running_flat, n_bins=5)
    pupil_edges = discretize_continuous(all_pupil_flat, n_bins=5)

    print(f"  Running speed edges: {running_edges}")
    print(f"  Pupil diameter edges: {pupil_edges}")

    # Build running speed bin labels
    valid_running = all_running_flat[~np.isnan(all_running_flat)]
    running_bin_labels = []
    for i in range(5):
        lo = running_edges[i] if i > 0 else valid_running.min()
        hi = running_edges[i + 1] if i < 4 else valid_running.max()
        running_bin_labels.append(f"bin{i}_{lo:.1f}_{hi:.1f}")

    valid_pupil = all_pupil_flat[~np.isnan(all_pupil_flat)]
    pupil_bin_labels = []
    for i in range(5):
        lo = pupil_edges[i] if i > 0 else valid_pupil.min()
        hi = pupil_edges[i + 1] if i < 4 else valid_pupil.max()
        pupil_bin_labels.append(f"bin{i}_{lo:.1f}_{hi:.1f}")

    # Now remap image identities to global indices and discretize continuous outputs
    print("\nAssembling final dataset...")
    neural_all = []
    input_all = []
    output_all = []
    subject_idx = []
    brain_region_idx_all = []

    n_images = len(all_image_names)
    outcome_names = ['hit', 'miss', 'false_alarm', 'correct_reject']

    # Precompute median pupil value for NaN filling (avoid recomputing in loop)
    pupil_fill_value = np.nanmedian(all_pupil_flat)

    # Free large intermediate arrays
    del all_running, all_pupil, all_running_flat, all_pupil_flat

    for r in results:
        session_neural = []
        session_output = []
        session_input = []

        # Map local image indices to global using lookup array
        local_to_global_arr = np.array([global_img_to_idx[name] for name in r['image_names']])

        for trial_idx, (neural, out) in enumerate(zip(r['neural'], r['outputs'])):
            session_neural.append(neural.astype(np.float32))

            n_bins = neural.shape[1]

            # Remap image identity to global indices using vectorized lookup
            img_id_global = local_to_global_arr[out['image_identity']]

            # Discretize running speed
            run_disc = apply_discretization(out['running_speed_raw'], running_edges)

            # Discretize pupil diameter (handle NaN: assign to median bin)
            pupil_raw = out['pupil_diam_raw'].copy()
            nan_mask = np.isnan(pupil_raw)
            pupil_raw[nan_mask] = pupil_fill_value
            pupil_disc = apply_discretization(pupil_raw, pupil_edges)

            # Build output array: (5, n_bins) for time-varying, (5,) for static+time-varying mix
            # Outputs: image_identity, image_change, running_speed, pupil_diameter (all time-varying)
            #          trial_outcome (static)
            output_data = np.zeros((5, n_bins), dtype=np.int64)
            output_data[0, :] = img_id_global        # image identity
            output_data[1, :] = out['image_change']   # image change
            output_data[2, :] = run_disc              # running speed (discretized)
            output_data[3, :] = pupil_disc            # pupil diameter (discretized)
            output_data[4, :] = out['trial_outcome']  # trial outcome (static, same value each bin)

            session_output.append(output_data)

            # No decoder inputs specified in task
            session_input.append(np.zeros((0, n_bins), dtype=np.float32))

        neural_all.append(session_neural)
        output_all.append(session_output)
        input_all.append(session_input)

        # Subject index
        subject_idx.append(all_subjects.index(r['mouse_id']))

        # Brain region index - all neurons in this experiment have same region
        region_idx = all_regions.index(r['targeted_structure'])
        brain_region_idx_all.append(np.full(r['n_neurons'], region_idx, dtype=np.int64))

    # Build output_values
    output_values = [
        all_image_names,                                    # image identity values
        ['no_change', 'change'],                           # image change values
        [f"speed_{running_bin_labels[i]}" for i in range(5)],  # running speed bin labels
        [f"pupil_{pupil_bin_labels[i]}" for i in range(5)],   # pupil diameter bin labels
        outcome_names,                                      # trial outcome values
    ]

    # Build session info for metadata
    session_info = []
    for r in results:
        session_info.append({
            'ophys_experiment_id': r['ophys_experiment_id'],
            'mouse_id': r['mouse_id'],
            'cre_line': r['cre_line'],
            'targeted_structure': r['targeted_structure'],
            'session_type': r['session_type'],
            'ophys_frame_rate': r['ophys_frame_rate'],
            'n_neurons': r['n_neurons'],
            'n_trials': len(r['neural']),
            'n_go_trials': r['n_go_trials'],
            'n_catch_trials': r['n_catch_trials'],
        })

    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,
        'subjects': all_subjects,
        'subject_idx': np.array(subject_idx, dtype=np.int64),
        'brain_regions': all_regions,
        'brain_region_idx': brain_region_idx_all,
        'input_names': [],
        'output_names': [
            'image_identity',
            'image_change',
            'running_speed',
            'pupil_diameter',
            'trial_outcome',
        ],
        'output_values': output_values,
        'metadata': {
            'task_description': 'Visual change detection go/no-go task. Mice detect changes in flashed natural images (250ms on, 500ms off). Go trials have image changes; Catch trials have no change. Outcomes: hit, miss, false_alarm, correct_reject.',
            'time_bin_size': COMMON_BIN_SIZE * 1000,  # in ms
            'temporal_alignment_event': 'Trial start time (onset of first stimulus in trial)',
            'off_start': 0.0,
            'off_end': None,  # variable trial duration
            'neural_data_type': 'dF/F traces (normalized change in fluorescence)',
            'dataset': 'Allen Brain Observatory Visual Behavior 2P',
            'running_speed_discretization': f'5 equal percentile bins, edges: {running_edges.tolist()}',
            'pupil_diameter_discretization': f'5 equal percentile bins, edges: {pupil_edges.tolist()}',
            'session_info': session_info,
            'n_sessions': len(results),
            'n_subjects': len(all_subjects),
            'total_trials': sum(len(s) for s in neural_all),
            'total_neurons': sum(r['n_neurons'] for r in results),
        },
    }

    # Print summary statistics
    print(f"\n{'=' * 60}")
    print("DATASET SUMMARY")
    print(f"{'=' * 60}")
    print(f"Sessions: {len(neural_all)}")
    print(f"Subjects: {len(all_subjects)}")
    print(f"Brain regions: {all_regions}")
    print(f"Total trials: {sum(len(s) for s in neural_all)}")
    print(f"Total neurons: {sum(r['n_neurons'] for r in results)}")
    print(f"Time bin size: {COMMON_BIN_SIZE * 1000:.1f} ms")
    print(f"Output variables: {data['output_names']}")
    print(f"Image names ({len(all_image_names)}): {all_image_names}")

    # Per-session stats
    trial_counts = [len(s) for s in neural_all]
    neuron_counts = [r['n_neurons'] for r in results]
    print(f"\nTrials per session: min={min(trial_counts)}, max={max(trial_counts)}, "
          f"mean={np.mean(trial_counts):.1f}")
    print(f"Neurons per session: min={min(neuron_counts)}, max={max(neuron_counts)}, "
          f"mean={np.mean(neuron_counts):.1f}")

    # Trial duration stats
    all_durations = []
    for session in neural_all:
        for trial in session:
            all_durations.append(trial.shape[1] * COMMON_BIN_SIZE)
    print(f"Trial duration: min={min(all_durations):.1f}s, max={max(all_durations):.1f}s, "
          f"mean={np.mean(all_durations):.1f}s")

    # Save
    output_path = os.path.join('/app', args.output)
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f"Saved! File size: {os.path.getsize(output_path) / 1e6:.1f} MB")

    # Sanity checks
    print(f"\n{'=' * 60}")
    print("SANITY CHECKS")
    print(f"{'=' * 60}")

    # Check 1: All trials have consistent neuron count within sessions
    for si, session in enumerate(neural_all):
        n_neurons_first = session[0].shape[0]
        for ti, trial in enumerate(session):
            assert trial.shape[0] == n_neurons_first, \
                f"Session {si}, trial {ti}: {trial.shape[0]} neurons vs {n_neurons_first}"
    print("PASS: Consistent neuron count within sessions")

    # Check 2: Output dimensions match
    for si, session in enumerate(output_all):
        for ti, out in enumerate(session):
            assert out.shape[0] == 5, f"Session {si}, trial {ti}: output has {out.shape[0]} dims, expected 5"
            assert out.shape[1] == neural_all[si][ti].shape[1], \
                f"Session {si}, trial {ti}: output time {out.shape[1]} != neural time {neural_all[si][ti].shape[1]}"
    print("PASS: Output dimensions consistent")

    # Check 3: Brain region indices valid
    for si, br_idx in enumerate(brain_region_idx_all):
        assert len(br_idx) == neural_all[si][0].shape[0], \
            f"Session {si}: brain_region_idx length {len(br_idx)} != n_neurons {neural_all[si][0].shape[0]}"
    print("PASS: Brain region indices valid")

    # Check 4: Output values in valid range
    for si, session in enumerate(output_all):
        for ti, out in enumerate(session):
            assert np.all(out[0] >= 0) and np.all(out[0] < n_images), \
                f"Session {si}, trial {ti}: image identity out of range"
            assert np.all((out[1] == 0) | (out[1] == 1)), \
                f"Session {si}, trial {ti}: image change not binary"
            assert np.all(out[2] >= 0) and np.all(out[2] < 5), \
                f"Session {si}, trial {ti}: running speed bin out of range"
            assert np.all(out[3] >= 0) and np.all(out[3] < 5), \
                f"Session {si}, trial {ti}: pupil diameter bin out of range"
            assert np.all(out[4] >= 0) and np.all(out[4] < 4), \
                f"Session {si}, trial {ti}: trial outcome out of range"
    print("PASS: Output values in valid range")

    # Check 5: Subject indices valid
    assert len(subject_idx) == len(neural_all)
    assert all(0 <= s < len(all_subjects) for s in subject_idx)
    print("PASS: Subject indices valid")

    print(f"\nAll sanity checks passed!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
