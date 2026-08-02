"""
Convert Allen Brain Observatory Visual Behavior 2P dataset to decoder format.

References:
- "Allen Brain Observatory: Visual Behavior 2P Technical Whitepaper" (whitepaper.pdf)
- "Behavioral strategy shapes activation of the Vip-Sst disinhibitory circuit
   in visual cortex" (paper.pdf)

Data selection (following paper.pdf methods):
- Active behavior sessions only (not passive viewing)
- Familiar image set sessions
- All available cre lines (Slc17a7, Sst, Vip)
- Both single-plane (Scientifica) and multi-plane (Multiscope) data
- All available brain regions (VISp, VISl, VISal, VISam)

Trial selection:
- Include Go and Catch trials only
- Exclude Aborted and Auto-rewarded trials

Neural data:
- Use detected calcium events (as in paper.pdf)
- Aligned to ophys timestamps

Outputs:
- Image identity (categorical, time-varying)
- Image change (binary, time-varying)
- Running speed (5 percentile bins, time-varying)
- Pupil diameter (5 percentile bins, time-varying)
- Trial outcome (static: hit, miss, false_alarm, correct_reject)

Temporal alignment:
- All signals aligned to ophys timestamps
- Common time bin size across sessions (93.2 ms, ~10.7 Hz, matching multiscope rate)
- Single-plane data (~31 Hz) downsampled by 3x averaging
"""

import sys
sys.path.insert(0, 'code')

import numpy as np
import pandas as pd
import pickle
import os
import re
import warnings
import argparse
from scipy import interpolate

warnings.filterwarnings('ignore')


def get_experiment_table(data_dir='data'):
    """Load experiment table and filter to available active behavior experiments."""
    exp_table = pd.read_csv(
        os.path.join(data_dir, 'visual-behavior-ophys-1.1.0/project_metadata/ophys_experiment_table.csv')
    )

    # Get available NWB files
    nwb_dir = os.path.join(data_dir, 'visual-behavior-ophys-1.1.0/behavior_ophys_experiments/')
    nwb_files = os.listdir(nwb_dir)
    exp_ids = [int(re.search(r'(\d+)', f).group(1)) for f in nwb_files]

    # Filter to available, active behavior, familiar sessions
    available = exp_table[exp_table['ophys_experiment_id'].isin(exp_ids)]
    filtered = available[
        (available['behavior_type'] == 'active_behavior') &
        (available['experience_level'] == 'Familiar')
    ].copy()

    filtered = filtered.sort_values(['mouse_id', 'ophys_session_id', 'ophys_experiment_id'])

    print(f"Total available experiments: {len(available)}")
    print(f"Active behavior, familiar: {len(filtered)}")
    print(f"  Unique mice: {filtered['mouse_id'].nunique()}")
    print(f"  Unique sessions: {filtered['ophys_session_id'].nunique()}")
    print(f"  Cre lines: {filtered['cre_line'].value_counts().to_dict()}")
    print(f"  Brain regions: {filtered['targeted_structure'].value_counts().to_dict()}")
    print(f"  Equipment: {filtered['equipment_name'].value_counts().to_dict()}")

    return filtered


def load_experiment(cache, ophys_experiment_id):
    """Load a single experiment using the SDK."""
    return cache.get_behavior_ophys_experiment(ophys_experiment_id)


def get_trial_outcome(trial):
    """Get trial outcome as string."""
    if trial['hit']:
        return 'hit'
    elif trial['miss']:
        return 'miss'
    elif trial['false_alarm']:
        return 'false_alarm'
    elif trial['correct_reject']:
        return 'correct_reject'
    else:
        return 'unknown'


def interpolate_to_ophys(signal_timestamps, signal_values, ophys_timestamps):
    """Interpolate a signal to ophys timestamps using linear interpolation."""
    if len(signal_timestamps) == 0 or len(signal_values) == 0:
        return np.full(len(ophys_timestamps), np.nan)

    # Remove NaN values for interpolation
    valid = ~np.isnan(signal_values)
    if valid.sum() < 2:
        return np.full(len(ophys_timestamps), np.nan)

    f = interpolate.interp1d(
        signal_timestamps[valid], signal_values[valid],
        kind='linear', bounds_error=False, fill_value=np.nan
    )
    return f(ophys_timestamps)


def get_image_at_timepoints(stim_presentations, ophys_timestamps, image_to_idx):
    """Get image identity at each ophys timepoint.

    During gray screen (between images), assign the most recent image.
    Before any image, assign the first image.
    """
    image_indices = np.zeros(len(ophys_timestamps), dtype=np.int64)

    # For each timepoint, find which image is being shown or was most recently shown
    stim_starts = stim_presentations['start_time'].values
    stim_ends = stim_presentations['end_time'].values
    stim_images = stim_presentations['image_name'].values

    # Build a timeline of image identity
    current_image_idx = -1
    stim_idx = 0

    for t_idx, t in enumerate(ophys_timestamps):
        # Advance stim_idx to find current or most recent stimulus
        while stim_idx < len(stim_starts) - 1 and stim_starts[stim_idx + 1] <= t:
            stim_idx += 1

        if stim_idx < len(stim_starts) and stim_starts[stim_idx] <= t:
            img = stim_images[stim_idx]
            if img in image_to_idx and img != 'omitted':
                current_image_idx = image_to_idx[img]
            # For omitted stimuli, keep the previous image identity

        if current_image_idx >= 0:
            image_indices[t_idx] = current_image_idx

    return image_indices


def get_image_change_signal(stim_presentations, ophys_timestamps):
    """Create binary image change signal.

    Value of 1 at ophys timepoints right after a change in image identity.
    """
    change_signal = np.zeros(len(ophys_timestamps), dtype=np.float32)

    # Get change times from stimulus presentations
    changes = stim_presentations[stim_presentations['is_change'] == True]

    for _, change in changes.iterrows():
        change_time = change['start_time']
        # Mark the ophys frame(s) during the change stimulus presentation
        # The change image is shown for 250ms
        mask = (ophys_timestamps >= change_time) & (ophys_timestamps < change_time + 0.75)
        change_signal[mask] = 1.0

    return change_signal


def downsample_by_factor(data, factor):
    """Downsample a 1D or 2D array by averaging every 'factor' samples along last axis."""
    if data.ndim == 1:
        n = len(data)
        n_new = n // factor
        if n_new == 0:
            return data[:1]
        return data[:n_new * factor].reshape(n_new, factor).mean(axis=1)
    elif data.ndim == 2:
        n = data.shape[1]
        n_new = n // factor
        if n_new == 0:
            return data[:, :1]
        return data[:, :n_new * factor].reshape(data.shape[0], n_new, factor).mean(axis=2)


def downsample_categorical_by_factor(data, factor):
    """Downsample categorical data by taking the mode in each bin."""
    n = len(data)
    n_new = n // factor
    if n_new == 0:
        return data[:1]
    result = np.zeros(n_new, dtype=data.dtype)
    for i in range(n_new):
        chunk = data[i*factor:(i+1)*factor]
        values, counts = np.unique(chunk, return_counts=True)
        result[i] = values[np.argmax(counts)]
    return result


def process_experiment(cache, exp_info, image_names, image_to_idx,
                       target_dt, all_running_speeds, all_pupil_diameters):
    """Process a single experiment and return trial data.

    Returns:
        neural_trials: list of (n_neurons, n_timepoints) arrays
        output_trials: list of (n_output, n_timepoints) or (n_output,) arrays
        trial_outcomes: list of outcome strings
        n_neurons: number of neurons
        brain_region: targeted structure
        skip_reason: string if experiment should be skipped, None otherwise
    """
    exp_id = int(exp_info['ophys_experiment_id'])

    try:
        ds = load_experiment(cache, exp_id)
    except Exception as e:
        return None, None, None, None, None, f"Failed to load: {e}"

    # Get ophys timestamps and frame rate
    ophys_ts = ds.ophys_timestamps
    dt = np.median(np.diff(ophys_ts))

    # Determine downsample factor to match target_dt
    ds_factor = max(1, int(round(target_dt / dt)))
    actual_dt = dt * ds_factor

    # Get neural data (calcium events as used in paper)
    try:
        events = ds.events
        neural_full = np.vstack(events['events'].values).astype(np.float32)
    except Exception as e:
        return None, None, None, None, None, f"Failed to get events: {e}"

    n_neurons = neural_full.shape[0]
    if n_neurons == 0:
        return None, None, None, None, None, "No neurons"

    # Get stimulus presentations (change detection block only)
    stim = ds.stimulus_presentations
    stim_cd = stim[stim['stimulus_block_name'].str.contains('change_detection', na=False)].copy()
    # Exclude omitted stimuli for image identity tracking, but keep them in the table
    stim_images = stim_cd[stim_cd['image_name'] != 'omitted']

    # Get trials (Go and Catch only, exclude Aborted and Auto-rewarded)
    trials = ds.trials
    valid_trials = trials[
        ((trials['go'] == True) | (trials['catch'] == True)) &
        (trials['aborted'] == False) &
        (trials['auto_rewarded'] == False)
    ]

    if len(valid_trials) < 2:
        return None, None, None, None, None, f"Only {len(valid_trials)} valid trials"

    # Get running speed interpolated to ophys timestamps
    rs = ds.running_speed
    running_at_ophys = interpolate_to_ophys(
        rs['timestamps'].values, rs['speed'].values, ophys_ts
    )
    # Replace NaN with 0 for running speed
    running_at_ophys = np.nan_to_num(running_at_ophys, nan=0.0)

    # Collect all running speeds for percentile computation
    all_running_speeds.extend(running_at_ophys.tolist())

    # Get pupil diameter interpolated to ophys timestamps
    try:
        et = ds.eye_tracking
        # Use pupil width as diameter (from tutorial: pupil_width)
        pupil_vals = et['pupil_width'].values
        pupil_ts = et['timestamps'].values
        pupil_at_ophys = interpolate_to_ophys(pupil_ts, pupil_vals, ophys_ts)
    except Exception:
        pupil_at_ophys = np.full(len(ophys_ts), np.nan)

    # Collect non-NaN pupil values for percentile computation
    valid_pupil = pupil_at_ophys[~np.isnan(pupil_at_ophys)]
    all_pupil_diameters.extend(valid_pupil.tolist())

    # Process each trial
    neural_trials = []
    output_trials = []
    trial_outcomes = []

    for trial_idx, (_, trial) in enumerate(valid_trials.iterrows()):
        t_start = trial['start_time']
        t_stop = trial['stop_time']

        # Get ophys frame indices for this trial
        frame_mask = (ophys_ts >= t_start) & (ophys_ts <= t_stop)
        frame_indices = np.where(frame_mask)[0]

        if len(frame_indices) < ds_factor:
            continue

        # Extract neural data for this trial
        neural_trial = neural_full[:, frame_indices]

        # Get trial ophys timestamps
        trial_ophys_ts = ophys_ts[frame_indices]

        # Get stimulus presentations for this trial
        trial_stim = stim_cd[
            (stim_cd['start_time'] >= t_start - 0.5) &
            (stim_cd['start_time'] <= t_stop + 0.5)
        ]
        trial_stim_no_omit = trial_stim[trial_stim['image_name'] != 'omitted']

        # Image identity at each timepoint
        image_idx = get_image_at_timepoints(trial_stim_no_omit, trial_ophys_ts, image_to_idx)

        # Image change signal
        change_signal = get_image_change_signal(trial_stim, trial_ophys_ts)

        # Running speed for this trial
        running_trial = running_at_ophys[frame_indices]

        # Pupil diameter for this trial
        pupil_trial = pupil_at_ophys[frame_indices]

        # Downsample if needed (for single-plane ~31Hz data)
        if ds_factor > 1:
            neural_trial = downsample_by_factor(neural_trial, ds_factor)
            image_idx = downsample_categorical_by_factor(image_idx, ds_factor)
            change_signal = downsample_by_factor(change_signal, ds_factor)
            # Binarize change signal after downsampling
            change_signal = (change_signal > 0.0).astype(np.float32)
            running_trial = downsample_by_factor(running_trial, ds_factor)
            pupil_trial = downsample_by_factor(pupil_trial, ds_factor)

        # Trial outcome
        outcome = get_trial_outcome(trial)

        # Store raw values (will bin later with global percentiles)
        neural_trials.append(neural_trial.astype(np.float32))
        output_trials.append({
            'image_idx': image_idx.astype(np.int64),
            'change_signal': change_signal.astype(np.float32),
            'running': running_trial.astype(np.float32),
            'pupil': pupil_trial.astype(np.float32),
            'outcome': outcome,
        })
        trial_outcomes.append(outcome)

    if len(neural_trials) < 2:
        return None, None, None, None, None, f"Only {len(neural_trials)} usable trials after processing"

    brain_region = exp_info['targeted_structure']

    return neural_trials, output_trials, trial_outcomes, n_neurons, brain_region, None


def discretize_to_percentile_bins(values, n_bins, percentiles):
    """Discretize values into n_bins based on pre-computed percentile boundaries."""
    result = np.digitize(values, percentiles)
    result = np.clip(result, 0, n_bins - 1)
    return result.astype(np.int64)


def convert_data(data_dir='data', sample_only=False, max_experiments=None):
    """Main conversion function."""
    from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache

    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=data_dir)

    # Get filtered experiment table
    exp_table = get_experiment_table(data_dir)

    if sample_only:
        # For quick testing, use a small subset
        # Pick 3 experiments from different mice/sessions
        sample = exp_table.groupby('mouse_id').first().reset_index()
        if len(sample) > 3:
            sample = sample.head(3)
        exp_ids = sample['ophys_experiment_id'].values
        exp_table = exp_table[exp_table['ophys_experiment_id'].isin(exp_ids)]
        print(f"\nSample mode: using {len(exp_table)} experiments")

    if max_experiments is not None:
        exp_table = exp_table.head(max_experiments)
        print(f"\nLimiting to {len(exp_table)} experiments")

    # Get all unique image names from the dataset
    # Load one experiment to discover image names
    first_exp = cache.get_behavior_ophys_experiment(int(exp_table.iloc[0]['ophys_experiment_id']))
    stim = first_exp.stimulus_presentations
    stim_cd = stim[stim['stimulus_block_name'].str.contains('change_detection', na=False)]
    all_image_names = sorted([n for n in stim_cd['image_name'].unique()
                              if n != 'omitted' and isinstance(n, str)])

    # Verify we have 8 images as expected
    print(f"\nImage names found: {all_image_names}")
    print(f"Number of images: {len(all_image_names)}")

    image_to_idx = {name: idx for idx, name in enumerate(all_image_names)}

    # Outcome categories
    outcome_names = ['hit', 'miss', 'false_alarm', 'correct_reject']
    outcome_to_idx = {name: idx for idx, name in enumerate(outcome_names)}

    # Target time bin (matching multiscope ~10.7 Hz)
    target_dt = 0.0932  # seconds

    # First pass: process all experiments and collect running/pupil values
    print(f"\nProcessing {len(exp_table)} experiments...")
    all_running_speeds = []
    all_pupil_diameters = []

    experiment_results = []
    for i, (_, exp_info) in enumerate(exp_table.iterrows()):
        exp_id = int(exp_info['ophys_experiment_id'])
        print(f"  [{i+1}/{len(exp_table)}] Experiment {exp_id} "
              f"(mouse={exp_info['mouse_id']}, {exp_info['cre_line']}, "
              f"{exp_info['targeted_structure']})...", end='')

        result = process_experiment(
            cache, exp_info, all_image_names, image_to_idx,
            target_dt, all_running_speeds, all_pupil_diameters
        )

        neural_trials, output_trials, trial_outcomes, n_neurons, brain_region, skip_reason = result

        if skip_reason:
            print(f" SKIPPED: {skip_reason}")
            continue

        print(f" OK: {n_neurons} neurons, {len(neural_trials)} trials")
        experiment_results.append({
            'exp_info': exp_info,
            'neural_trials': neural_trials,
            'output_trials': output_trials,
            'trial_outcomes': trial_outcomes,
            'n_neurons': n_neurons,
            'brain_region': brain_region,
        })

    print(f"\nSuccessfully processed {len(experiment_results)} experiments")

    if len(experiment_results) == 0:
        raise ValueError("No experiments were successfully processed!")

    # Compute global percentile boundaries for running speed and pupil
    n_bins = 5
    running_arr = np.array(all_running_speeds)
    pupil_arr = np.array(all_pupil_diameters)

    running_percentiles = np.percentile(running_arr, np.linspace(0, 100, n_bins + 1)[1:-1])
    print(f"\nRunning speed percentile boundaries: {running_percentiles}")

    if len(pupil_arr) > 0:
        pupil_percentiles = np.percentile(pupil_arr, np.linspace(0, 100, n_bins + 1)[1:-1])
        print(f"Pupil diameter percentile boundaries: {pupil_percentiles}")
    else:
        pupil_percentiles = np.array([0, 1, 2, 3])
        print("WARNING: No valid pupil data!")

    # Build output data structure
    neural_all = []
    input_all = []
    output_all = []
    subjects = []
    subject_idx_list = []
    brain_regions_list = []
    brain_region_idx_all = []

    # Collect unique subjects and brain regions
    all_mice = sorted(set(r['exp_info']['mouse_id'] for r in experiment_results))
    mouse_to_idx = {m: i for i, m in enumerate(all_mice)}
    subjects = [str(m) for m in all_mice]

    all_regions = sorted(set(r['brain_region'] for r in experiment_results))
    region_to_idx = {r: i for i, r in enumerate(all_regions)}
    brain_regions_list = list(all_regions)

    for result in experiment_results:
        exp_info = result['exp_info']
        neural_trials = result['neural_trials']
        output_trials_raw = result['output_trials']
        n_neurons = result['n_neurons']
        brain_region = result['brain_region']

        # Subject index
        subject_idx_list.append(mouse_to_idx[exp_info['mouse_id']])

        # Brain region index for each neuron (all same region for one experiment)
        region_idx = region_to_idx[brain_region]
        brain_region_idx_all.append(np.full(n_neurons, region_idx, dtype=np.int64))

        # Process output trials: apply binning
        session_neural = []
        session_input = []  # No inputs specified
        session_output = []

        for t_idx, (neural_trial, out_raw) in enumerate(zip(neural_trials, output_trials_raw)):
            n_timepoints = neural_trial.shape[1]

            # Neural
            session_neural.append(neural_trial)

            # Input: empty (no inputs for this task)
            session_input.append(np.zeros((0, n_timepoints), dtype=np.float32))

            # Outputs (all time-varying except trial outcome)
            # 1. Image identity (categorical)
            image_idx = out_raw['image_idx']

            # 2. Image change (binary)
            change_signal = out_raw['change_signal']

            # 3. Running speed (5 bins)
            running_binned = discretize_to_percentile_bins(
                out_raw['running'], n_bins, running_percentiles
            ).astype(np.float32)

            # 4. Pupil diameter (5 bins)
            pupil_vals = out_raw['pupil']
            # Replace NaN with median bin
            nan_mask = np.isnan(pupil_vals)
            if nan_mask.all():
                pupil_binned = np.full(len(pupil_vals), 2, dtype=np.float32)  # median bin
            else:
                pupil_clean = pupil_vals.copy()
                if nan_mask.any():
                    # Forward fill then backward fill NaN
                    valid_indices = np.where(~nan_mask)[0]
                    if len(valid_indices) > 0:
                        for j in range(len(pupil_clean)):
                            if nan_mask[j]:
                                # Find nearest valid value
                                dists = np.abs(valid_indices - j)
                                nearest = valid_indices[np.argmin(dists)]
                                pupil_clean[j] = pupil_clean[nearest]
                pupil_binned = discretize_to_percentile_bins(
                    pupil_clean, n_bins, pupil_percentiles
                ).astype(np.float32)

            # 5. Trial outcome (static)
            outcome = out_raw['outcome']
            outcome_idx = outcome_to_idx.get(outcome, 0)

            # Stack outputs: 4 time-varying + 1 static
            # Time-varying outputs: (4, n_timepoints)
            # Static output: (1,)
            # We need to combine them.
            # Output format: (n_output, n_timepoints) for time-varying
            # or (n_output,) for static
            # Since we have a mix, we'll make time-varying ones (4, T)
            # and static one separate
            # Actually, re-reading the format: output can be (n_output, n_timepoints) or (n_output,)
            # But all trials in a session should have same format
            # Let's make trial outcome time-varying too (constant over trial)

            output_trial = np.stack([
                image_idx.astype(np.int64),
                change_signal.astype(np.int64),
                running_binned.astype(np.int64),
                pupil_binned.astype(np.int64),
                np.full(n_timepoints, outcome_idx, dtype=np.int64),
            ], axis=0)

            session_output.append(output_trial)

        neural_all.append(session_neural)
        input_all.append(session_input)
        output_all.append(session_output)

    # Running speed bin labels
    running_bin_labels = [f"bin_{i+1}" for i in range(n_bins)]
    pupil_bin_labels = [f"bin_{i+1}" for i in range(n_bins)]

    # Construct the final data dictionary
    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,
        'subjects': subjects,
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),
        'brain_regions': brain_regions_list,
        'brain_region_idx': brain_region_idx_all,
        'input_names': [],
        'output_names': ['image_identity', 'image_change', 'running_speed',
                         'pupil_diameter', 'trial_outcome'],
        'output_values': [
            all_image_names,                # image identity values
            ['no_change', 'change'],        # image change values
            running_bin_labels,             # running speed bins
            pupil_bin_labels,               # pupil diameter bins
            outcome_names,                  # trial outcome values
        ],
        'metadata': {
            'task_description': 'Visual change detection go/no-go task. Mice view flashed natural images (250ms on, 500ms gray) and report changes by licking.',
            'time_bin_size': target_dt * 1000,  # in ms
            'temporal_alignment_event': 'Trial start (aligned to ophys timestamps)',
            'off_start': 0.0,  # trial starts at trial start_time
            'off_end': None,  # variable trial length
            'dataset': 'Allen Brain Observatory Visual Behavior 2P',
            'neural_signal': 'Detected calcium events (deconvolved from dF/F)',
            'frame_rate_hz': 1000.0 / (target_dt * 1000),
            'n_sessions': len(experiment_results),
            'n_subjects': len(subjects),
            'experiment_selection': 'Active behavior, familiar image set',
            'trial_selection': 'Go and Catch trials (excluding Aborted and Auto-rewarded)',
            'running_speed_percentiles': running_percentiles.tolist(),
            'pupil_diameter_percentiles': pupil_percentiles.tolist(),
        }
    }

    # Print summary statistics
    total_trials = sum(len(s) for s in data['neural'])
    total_neurons = sum(data['neural'][i][0].shape[0] for i in range(len(data['neural'])) if len(data['neural'][i]) > 0)
    trial_lengths = [data['neural'][i][j].shape[1] for i in range(len(data['neural'])) for j in range(len(data['neural'][i]))]

    print(f"\n=== CONVERSION SUMMARY ===")
    print(f"Sessions: {len(data['neural'])}")
    print(f"Subjects: {len(subjects)}")
    print(f"Total trials: {total_trials}")
    print(f"Brain regions: {brain_regions_list}")
    print(f"Time bin size: {target_dt*1000:.1f} ms")
    print(f"Trial length range: {min(trial_lengths)}-{max(trial_lengths)} timepoints")
    print(f"Mean trial length: {np.mean(trial_lengths):.1f} timepoints ({np.mean(trial_lengths)*target_dt:.1f} s)")

    # Per-session stats
    for i in range(len(data['neural'])):
        n_trials_i = len(data['neural'][i])
        if n_trials_i > 0:
            n_neurons_i = data['neural'][i][0].shape[0]
        else:
            n_neurons_i = 0
        print(f"  Session {i}: {n_neurons_i} neurons, {n_trials_i} trials, "
              f"subject={data['subjects'][data['subject_idx'][i]]}, "
              f"region={data['brain_regions'][data['brain_region_idx'][i][0]]}")

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', action='store_true', help='Only process a small sample')
    parser.add_argument('--output', type=str, default='converted_data.pkl', help='Output file path')
    parser.add_argument('--max-experiments', type=int, default=None, help='Max experiments to process')
    args = parser.parse_args()

    output_file = args.output

    data = convert_data(
        data_dir='data',
        sample_only=args.sample,
        max_experiments=args.max_experiments,
    )

    print(f"\nSaving to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved successfully. File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")
