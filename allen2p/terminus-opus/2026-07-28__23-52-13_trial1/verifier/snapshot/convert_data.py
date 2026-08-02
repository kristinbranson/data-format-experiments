#!/usr/bin/env python3
"""Convert Allen Brain Observatory Visual Behavior 2P data to decoder format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
"""

import sys
import os
import re
import time
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

sys.path.insert(0, '/app/code')
from allensdk.brain_observatory.behavior.behavior_ophys_experiment import BehaviorOphysExperiment

# Parse arguments
parser = argparse.ArgumentParser(description='Convert Visual Behavior 2P data to decoder format.')
parser.add_argument('outfile', type=str, help='Output pickle file path')
parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
parser.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
args = parser.parse_args()

if args.sample:
    args.full = False

# Constants
DATA_DIR = '/app/data'
NWB_DIR = os.path.join(DATA_DIR, 'visual-behavior-ophys-1.1.0', 'behavior_ophys_experiments')
METADATA_DIR = os.path.join(DATA_DIR, 'visual-behavior-ophys-1.1.0', 'project_metadata')

TRIAL_OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']


def get_experiment_table():
    """Get filtered experiment table for active behavior experiments."""
    nwb_files = os.listdir(NWB_DIR)
    all_exp_ids = [int(re.search(r'(\d+)', f).group(1)) for f in nwb_files if f.endswith('.nwb')]
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    exp_table = exp_table[exp_table['ophys_experiment_id'].isin(all_exp_ids)]
    exp_table = exp_table[exp_table['behavior_type'] == 'active_behavior']
    # Exclude Multiscope (11 Hz) sessions - different frame rate than Scientifica (31 Hz)
    # The format requires consistent time bins across all sessions
    exp_table = exp_table[exp_table['equipment_name'] != 'MESO.1']
    return exp_table


def load_experiment(exp_id):
    """Load a single experiment from NWB file."""
    nwb_path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{exp_id}.nwb')
    return BehaviorOphysExperiment.from_nwb_path(nwb_path)


def resample_to_ophys(signal_timestamps, signal_values, ophys_timestamps):
    """Resample a signal to ophys timestamps using linear interpolation."""
    return np.interp(ophys_timestamps, signal_timestamps, signal_values).astype(np.float32)


def get_image_at_ophys(sp_active, ophys_ts, image_to_idx):
    """Get image identity index at each ophys timestamp.
    
    During image presentation: index of current image.
    Between presentations (gray screen): index of last shown image.
    Before first image: -1 (will be handled per trial).
    """
    n_tp = len(ophys_ts)
    image_idx = np.full(n_tp, -1, dtype=np.int32)
    
    # Vectorized: for each stimulus presentation, find ophys frames
    starts = sp_active['start_time'].values
    ends = sp_active['end_time'].values
    names = sp_active['image_name'].values
    
    for j in range(len(starts)):
        img_name = names[j]
        if not isinstance(img_name, str) or img_name == 'omitted':
            continue
        if img_name not in image_to_idx:
            continue
        mask = (ophys_ts >= starts[j]) & (ophys_ts < ends[j])
        image_idx[mask] = image_to_idx[img_name]
    
    # Forward-fill
    last_img = -1
    for i in range(n_tp):
        if image_idx[i] >= 0:
            last_img = image_idx[i]
        elif last_img >= 0:
            image_idx[i] = last_img
    
    return image_idx


def get_change_at_ophys(sp_active, ophys_ts):
    """Get binary change signal at ophys timestamps."""
    n_tp = len(ophys_ts)
    change = np.zeros(n_tp, dtype=np.int32)
    
    change_sp = sp_active[sp_active['is_change'] == True]
    for _, row in change_sp.iterrows():
        mask = (ophys_ts >= row['start_time']) & (ophys_ts < row['end_time'])
        change[mask] = 1
    
    return change


def get_trial_outcome(trial_row):
    """Get trial outcome as integer: 0=hit, 1=miss, 2=false_alarm, 3=correct_reject."""
    if trial_row['hit']: return 0
    if trial_row['miss']: return 1
    if trial_row['false_alarm']: return 2
    if trial_row['correct_reject']: return 3
    return -1


def extract_session_data(dataset, image_to_idx):
    """Extract all needed raw data from a dataset.
    
    Returns dict with raw data, or None if session should be skipped.
    """
    metadata = dataset.metadata
    ophys_ts = dataset.ophys_timestamps
    dt = np.median(np.diff(ophys_ts))
    
    # Neural data (events)
    events = dataset.events
    events_array = np.vstack(events['events'].values).astype(np.float32)
    n_neurons = events_array.shape[0]
    
    if n_neurons < 2:
        return None
    
    # Trials
    trials = dataset.trials
    valid_trials = trials[(trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']]
    if len(valid_trials) < 2:
        return None
    
    # Stimulus presentations (change detection block only)
    sp = dataset.stimulus_presentations
    sp_active = sp[sp['stimulus_block_name'] == 'change_detection_behavior'].copy()
    
    # Image identity at ophys timestamps
    image_idx_full = get_image_at_ophys(sp_active, ophys_ts, image_to_idx)
    
    # Image change signal
    change_full = get_change_at_ophys(sp_active, ophys_ts)
    
    # Running speed resampled to ophys timestamps
    running = dataset.running_speed
    running_full = resample_to_ophys(running['timestamps'].values, running['speed'].values, ophys_ts)
    
    # Pupil diameter resampled to ophys timestamps
    try:
        eye = dataset.eye_tracking
        pupil_raw = eye['pupil_width'].values
        pupil_ts = eye['timestamps'].values
        valid_mask = ~np.isnan(pupil_raw)
        if valid_mask.sum() > 10:
            pupil_full = resample_to_ophys(pupil_ts[valid_mask], pupil_raw[valid_mask], ophys_ts)
        else:
            pupil_full = np.full(len(ophys_ts), np.nan, dtype=np.float32)
    except:
        pupil_full = np.full(len(ophys_ts), np.nan, dtype=np.float32)
    
    return {
        'ophys_ts': ophys_ts,
        'events_array': events_array,
        'n_neurons': n_neurons,
        'valid_trials': valid_trials,
        'image_idx_full': image_idx_full,
        'change_full': change_full,
        'running_full': running_full,
        'pupil_full': pupil_full,
        'metadata': metadata,
        'dt': dt,
        'sp_active': sp_active,
    }


def segment_trials(session_data, running_edges, pupil_edges):
    """Segment session data into trials and apply binning.
    
    Returns lists of neural and output arrays per trial.
    """
    ophys_ts = session_data['ophys_ts']
    events_array = session_data['events_array']
    valid_trials = session_data['valid_trials']
    image_idx_full = session_data['image_idx_full']
    change_full = session_data['change_full']
    running_full = session_data['running_full']
    pupil_full = session_data['pupil_full']
    
    # Bin running speed
    running_binned = np.clip(np.digitize(running_full, running_edges[1:-1]), 0, 4).astype(np.int32)
    
    # Bin pupil diameter
    valid_pupil_mask = ~np.isnan(pupil_full)
    pupil_binned = np.zeros(len(ophys_ts), dtype=np.int32)
    if valid_pupil_mask.any() and pupil_edges is not None:
        pupil_binned[valid_pupil_mask] = np.clip(
            np.digitize(pupil_full[valid_pupil_mask], pupil_edges[1:-1]), 0, 4
        ).astype(np.int32)
    
    neural_trials = []
    output_trials = []
    
    for _, trial_row in valid_trials.iterrows():
        start_time = trial_row['start_time']
        stop_time = trial_row['stop_time']
        
        frame_mask = (ophys_ts >= start_time) & (ophys_ts < stop_time)
        frame_indices = np.where(frame_mask)[0]
        
        if len(frame_indices) < 5:
            continue
        
        # Neural
        neural_trial = events_array[:, frame_indices]
        
        # Outputs
        img_trial = image_idx_full[frame_indices]
        change_trial = change_full[frame_indices]
        running_trial = running_binned[frame_indices]
        pupil_trial = pupil_binned[frame_indices]
        outcome = get_trial_outcome(trial_row)
        outcome_trial = np.full(len(frame_indices), outcome, dtype=np.int32)
        
        # Fix any -1 image indices (before first stimulus)
        neg_mask = img_trial < 0
        if neg_mask.any():
            first_valid = img_trial[~neg_mask][0] if (~neg_mask).any() else 0
            img_trial[neg_mask] = first_valid
        
        output_trial = np.stack([img_trial, change_trial, running_trial, pupil_trial, outcome_trial], axis=0)
        
        neural_trials.append(neural_trial)
        output_trials.append(output_trial)
    
    return neural_trials, output_trials


def plot_processing(exp_id, session_data, running_edges, pupil_edges, neural_trials, output_trials):
    """Plot processing steps for visual verification."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    ophys_ts = session_data['ophys_ts']
    events_array = session_data['events_array']
    valid_trials = session_data['valid_trials']
    
    fig, axes = plt.subplots(7, 1, figsize=(20, 24), sharex=True)
    
    # Select a time window around a trial
    trial = valid_trials.iloc[min(5, len(valid_trials)-1)]
    t_start = trial['start_time'] - 5
    t_end = trial['stop_time'] + 5
    mask = (ophys_ts >= t_start) & (ophys_ts <= t_end)
    t = ophys_ts[mask]
    
    # 1. Neural events
    ax = axes[0]
    n_show = min(5, events_array.shape[0])
    for i in range(n_show):
        ax.plot(t, events_array[i, mask] + i * 0.5, alpha=0.7, linewidth=0.5)
    ax.set_ylabel('Events')
    ax.set_title(f'Experiment {exp_id} - Neural Events')
    ax.axvline(trial['start_time'], color='green', linestyle='--', alpha=0.5, label='trial start')
    ax.axvline(trial['stop_time'], color='red', linestyle='--', alpha=0.5, label='trial stop')
    if pd.notna(trial.get('change_time', np.nan)):
        ax.axvline(trial['change_time'], color='blue', linestyle='--', alpha=0.5, label='change')
    ax.legend(fontsize=8)
    
    # 2. Image identity
    ax = axes[1]
    ax.plot(t, session_data['image_idx_full'][mask], 'b-', linewidth=0.5)
    ax.set_ylabel('Image Index')
    ax.set_title('Image Identity')
    
    # 3. Image change
    ax = axes[2]
    ax.plot(t, session_data['change_full'][mask], 'r-', linewidth=0.5)
    ax.set_ylabel('Change')
    ax.set_title('Image Change Signal')
    
    # 4. Running speed (continuous)
    ax = axes[3]
    ax.plot(t, session_data['running_full'][mask], 'g-', linewidth=0.5)
    ax.set_ylabel('Speed (cm/s)')
    ax.set_title('Running Speed (continuous)')
    
    # 5. Running speed (binned)
    running_binned = np.clip(np.digitize(session_data['running_full'], running_edges[1:-1]), 0, 4)
    ax = axes[4]
    ax.plot(t, running_binned[mask], 'm-', linewidth=0.5)
    ax.set_ylabel('Speed Bin')
    ax.set_title('Running Speed (5 percentile bins)')
    
    # 6. Pupil diameter (continuous)
    ax = axes[5]
    ax.plot(t, session_data['pupil_full'][mask], 'c-', linewidth=0.5)
    ax.set_ylabel('Pupil Width')
    ax.set_title('Pupil Diameter')
    
    # 7. Pupil binned
    valid_p = ~np.isnan(session_data['pupil_full'])
    pupil_binned = np.zeros(len(ophys_ts), dtype=np.int32)
    if valid_p.any() and pupil_edges is not None:
        pupil_binned[valid_p] = np.clip(np.digitize(session_data['pupil_full'][valid_p], pupil_edges[1:-1]), 0, 4)
    ax = axes[6]
    ax.plot(t, pupil_binned[mask], 'k-', linewidth=0.5)
    ax.set_ylabel('Pupil Bin')
    ax.set_title('Pupil Diameter (5 percentile bins)')
    ax.set_xlabel('Time (s)')
    
    for ax in axes:
        ax.axvline(trial['start_time'], color='green', linestyle='--', alpha=0.3)
        ax.axvline(trial['stop_time'], color='red', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'processing_{exp_id}.png', dpi=100)
    plt.close()
    print(f'  Saved processing plot: processing_{exp_id}.png')


def main():
    total_start = time.time()
    
    # Get experiment table
    exp_table = get_experiment_table()
    print(f'Found {len(exp_table)} active behavior experiments')
    
    if args.sample:
        exp_table = exp_table.head(2)
        print(f'Sample mode: processing {len(exp_table)} experiments')
    
    exp_ids = exp_table['ophys_experiment_id'].values
    n_exp = len(exp_ids)
    
    # ===== PASS 1: Load all experiments, collect global stats =====
    print('\n=== Pass 1: Loading data and collecting global statistics ===')
    all_session_data = []
    all_running_values = []
    all_pupil_values = []
    all_image_names = set()
    
    for i, exp_id in enumerate(exp_ids):
        t0 = time.time()
        try:
            dataset = load_experiment(exp_id)
        except Exception as e:
            print(f'  ERROR loading experiment {exp_id}: {e}')
            all_session_data.append(None)
            continue
        
        # Collect image names from stimulus presentations
        sp = dataset.stimulus_presentations
        sp_active = sp[sp['stimulus_block_name'] == 'change_detection_behavior']
        for name in sp_active['image_name'].unique():
            if isinstance(name, str) and name != 'omitted':
                all_image_names.add(name)
        
        # Build image_to_idx (temporary, will rebuild after collecting all names)
        temp_image_to_idx = {name: idx for idx, name in enumerate(sorted(all_image_names))}
        
        # Extract session data
        session_data = extract_session_data(dataset, temp_image_to_idx)
        
        if session_data is None:
            print(f'  Skipping experiment {exp_id}: insufficient neurons or trials')
            all_session_data.append(None)
            continue
        
        # Collect running and pupil values for global percentile computation
        all_running_values.append(session_data['running_full'])
        valid_pupil = session_data['pupil_full'][~np.isnan(session_data['pupil_full'])]
        if len(valid_pupil) > 0:
            all_pupil_values.append(valid_pupil)
        
        all_session_data.append(session_data)
        
        t_elapsed = time.time() - t0
        if (i + 1) % 10 == 0 or i == 0:
            print(f'  [{i+1}/{n_exp}] Experiment {exp_id}: {session_data["n_neurons"]} neurons, '
                  f'{len(session_data["valid_trials"])} trials, {t_elapsed:.1f}s')
    
    # Compute global image name mapping
    all_image_names = sorted(list(all_image_names))
    image_to_idx = {name: idx for idx, name in enumerate(all_image_names)}
    print(f'\nAll image names ({len(all_image_names)}): {all_image_names}')
    
    # Compute global percentile edges
    all_running_concat = np.concatenate(all_running_values)
    running_edges = np.percentile(all_running_concat, np.linspace(0, 100, 6))
    print(f'Running speed percentile edges: {running_edges}')
    
    if len(all_pupil_values) > 0:
        all_pupil_concat = np.concatenate(all_pupil_values)
        pupil_edges = np.percentile(all_pupil_concat, np.linspace(0, 100, 6))
        print(f'Pupil diameter percentile edges: {pupil_edges}')
    else:
        pupil_edges = None
        print('WARNING: No pupil data available')
    
    print(f'Pass 1 completed in {time.time() - total_start:.1f}s')
    
    # ===== PASS 2: Re-extract with correct image mapping and apply binning =====
    print('\n=== Pass 2: Segmenting trials with global binning ===')
    
    all_neural = []
    all_input = []
    all_output = []
    all_subject_idx = []
    all_brain_region_idx = []
    session_info = []
    
    subject_set = {}  # mouse_id -> index
    brain_region_set = {}  # region -> index
    
    for i, exp_id in enumerate(exp_ids):
        if all_session_data[i] is None:
            continue
        
        t0 = time.time()
        
        # Re-extract image indices with global mapping if needed
        session_data = all_session_data[i]
        
        # Re-compute image indices with global image_to_idx
        session_data['image_idx_full'] = get_image_at_ophys(session_data['sp_active'], session_data['ophys_ts'], image_to_idx)
        
        # Segment trials
        neural_trials, output_trials = segment_trials(session_data, running_edges, pupil_edges)
        
        if len(neural_trials) < 2:
            print(f'  Skipping experiment {exp_id}: only {len(neural_trials)} trials after segmentation')
            continue
        
        # Input: empty (no decoder inputs specified)
        input_trials = [np.zeros((0, t.shape[1]), dtype=np.float32) for t in neural_trials]
        
        all_neural.append(neural_trials)
        all_input.append(input_trials)
        all_output.append(output_trials)
        
        # Subject tracking
        meta = session_data['metadata']
        mouse_id = str(meta['mouse_id'])
        if mouse_id not in subject_set:
            subject_set[mouse_id] = len(subject_set)
        all_subject_idx.append(subject_set[mouse_id])
        
        # Brain region tracking
        region = meta['targeted_structure']
        if region not in brain_region_set:
            brain_region_set[region] = len(brain_region_set)
        all_brain_region_idx.append(
            np.full(session_data['n_neurons'], brain_region_set[region], dtype=np.int64)
        )
        
        session_info.append({
            'exp_id': exp_id,
            'session_id': int(meta['ophys_session_id']),
            'mouse_id': mouse_id,
            'cre_line': meta['cre_line'],
            'session_type': meta['session_type'],
            'targeted_structure': region,
            'n_neurons': session_data['n_neurons'],
            'n_trials': len(neural_trials),
            'ophys_frame_rate': float(1.0 / session_data['dt']),
        })
        
        # Processing visualization
        if args.show_processing and len(all_neural) <= 2:
            plot_processing(exp_id, session_data, running_edges, pupil_edges, neural_trials, output_trials)
        
        t_elapsed = time.time() - t0
        if (len(all_neural)) % 10 == 0 or len(all_neural) == 1:
            print(f'  [{len(all_neural)}] Experiment {exp_id}: {session_data["n_neurons"]} neurons, '
                  f'{len(neural_trials)} trials, {t_elapsed:.1f}s')
    
    if len(all_neural) == 0:
        print('ERROR: No experiments processed successfully!')
        sys.exit(1)
    
    # ===== Build output dictionary =====
    print(f'\n=== Building output dictionary ===')
    
    subjects = [''] * len(subject_set)
    for mouse_id, idx in subject_set.items():
        subjects[idx] = mouse_id
    
    brain_regions = [''] * len(brain_region_set)
    for region, idx in brain_region_set.items():
        brain_regions[idx] = region
    
    output_names = ['image_identity', 'image_change', 'running_speed', 'pupil_diameter', 'trial_outcome']
    output_values = [
        all_image_names,
        ['no_change', 'change'],
        [f'bin_{i}' for i in range(5)],
        [f'bin_{i}' for i in range(5)],
        TRIAL_OUTCOME_NAMES,
    ]
    
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': subjects,
        'subject_idx': np.array(all_subject_idx, dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': all_brain_region_idx,
        'input_names': [],
        'output_names': output_names,
        'output_values': output_values,
        'metadata': {
            'task_description': 'Visual Behavior change detection task - mice report image identity changes by licking',
            'time_bin_size': 1000.0 / 31.0,  # ~32.3 ms
            'temporal_alignment_event': 'Trial start time',
            'off_start': 0.0,
            'off_end': None,
            'session_info': session_info,
            'all_image_names': all_image_names,
            'running_percentiles': running_edges.tolist(),
            'pupil_percentiles': pupil_edges.tolist() if pupil_edges is not None else None,
        }
    }
    
    # Save
    print(f'\nSaving to {args.outfile}...')
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    
    file_size = os.path.getsize(args.outfile) / (1024**2)
    print(f'Saved {args.outfile} ({file_size:.1f} MB)')
    
    # Summary
    total_time = time.time() - total_start
    total_trials = sum(len(s) for s in all_neural)
    total_neurons = sum(info['n_neurons'] for info in session_info)
    print(f'\n=== Summary ===')
    print(f'Sessions: {len(all_neural)}')
    print(f'Subjects: {len(subjects)}')
    print(f'Brain regions: {brain_regions}')
    print(f'Total trials: {total_trials}')
    print(f'Total neurons: {total_neurons}')
    print(f'Mean trials/session: {total_trials/len(all_neural):.1f}')
    print(f'Mean neurons/session: {total_neurons/len(all_neural):.1f}')
    print(f'Output names: {output_names}')
    print(f'Total processing time: {total_time:.1f}s ({total_time/60:.1f} min)')


if __name__ == '__main__':
    main()
