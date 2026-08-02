#!/usr/bin/env python3
"""
Convert Allen Brain Observatory Visual Behavior 2P data to decoder-compatible format.

Usage:
    python -u convert_data.py <output_pickle_file> [--full|--sample] [--show-processing]
"""

import argparse
import glob
import os
import pickle
import sys
import time
import warnings

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate

warnings.filterwarnings('ignore', category=FutureWarning)

# ==============================================================================
# Constants
# ==============================================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
NWB_DIR = os.path.join(DATA_DIR, 'visual-behavior-ophys-1.1.0', 'behavior_ophys_experiments')
METADATA_DIR = os.path.join(DATA_DIR, 'visual-behavior-ophys-1.1.0', 'project_metadata')

# Image names (excluding 'omitted' and gray screen)
GRAY_LABEL = 'gray'

# ==============================================================================
# Helper functions
# ==============================================================================

def load_experiment_table():
    """Load the ophys experiment table and filter to active sessions with downloaded NWB files."""
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))

    # Get list of downloaded NWB experiment IDs
    nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
    downloaded_ids = set()
    for f in nwb_files:
        eid = int(os.path.basename(f).replace('behavior_ophys_experiment_', '').replace('.nwb', ''))
        downloaded_ids.add(eid)

    # Filter to downloaded experiments
    exp_table = exp_table[exp_table['ophys_experiment_id'].isin(downloaded_ids)].copy()

    # Filter to active sessions only (exclude passive)
    exp_table = exp_table[~exp_table['session_type'].str.contains('passive', case=False)].copy()

    # Sort by experiment ID for reproducibility
    exp_table = exp_table.sort_values('ophys_experiment_id').reset_index(drop=True)

    return exp_table


def load_nwb_data(nwb_path):
    """Load all relevant data from a single NWB file."""
    data = {}
    with h5py.File(nwb_path, 'r') as f:
        # Ophys timestamps
        data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]

        # DFF traces: shape (n_frames, n_cells) -> transpose to (n_cells, n_frames)
        dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
        data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)

        # Cell specimen IDs from image segmentation
        if 'image_segmentation' in f['processing']['ophys']:
            seg = f['processing']['ophys']['image_segmentation']
            # Find the plane segmentation
            for key in seg.keys():
                if 'id' in seg[key]:
                    data['cell_roi_ids'] = seg[key]['id'][:]
                    break

        # Running speed
        data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
        data['running_speed'] = f['processing']['running']['speed']['data'][:]

        # Eye tracking / pupil
        if 'EyeTracking' in f.get('acquisition', {}):
            et = f['acquisition']['EyeTracking']
            if 'pupil_tracking' in et:
                pt = et['pupil_tracking']
                data['pupil_area'] = pt['area']['data'][:] if 'data' in pt['area'] else pt['area'][:]
                data['pupil_timestamps'] = pt['timestamps'][:]
                data['likely_blink'] = et['likely_blink']['data'][:]
            else:
                data['pupil_area'] = None
        else:
            data['pupil_area'] = None

        # Trials
        trials = f['intervals']['trials']
        trial_data = {}
        for key in ['start_time', 'stop_time', 'go', 'catch', 'aborted', 'auto_rewarded',
                     'hit', 'miss', 'false_alarm', 'correct_reject', 'change_time',
                     'initial_image_name', 'change_image_name', 'is_change']:
            if key in trials:
                trial_data[key] = trials[key][:]
        data['trials'] = trial_data

        # Stimulus presentations - find the image presentations interval
        stim_key = None
        for key in f['intervals'].keys():
            if 'Natural_Images' in key or 'natural_images' in key:
                stim_key = key
                break

        if stim_key is None:
            # Try other naming conventions
            for key in f['intervals'].keys():
                if key not in ['trials', 'spontaneous_presentations', 'natural_movie_one_presentations']:
                    if 'presentation' in key.lower():
                        stim_key = key
                        break

        if stim_key is not None:
            stim = f['intervals'][stim_key]
            stim_data = {}
            for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
                if key in stim:
                    stim_data[key] = stim[key][:]
            data['stimulus'] = stim_data
        else:
            data['stimulus'] = None

    return data


def get_valid_trials(trial_data):
    """Get indices of valid trials (Go + Catch, excluding Aborted and Auto-rewarded)."""
    go = trial_data['go'].astype(bool)
    catch = trial_data['catch'].astype(bool)
    aborted = trial_data['aborted'].astype(bool)
    auto_rewarded = trial_data['auto_rewarded'].astype(bool)

    valid = (go | catch) & ~aborted & ~auto_rewarded
    return np.where(valid)[0]


def get_trial_outcome(trial_data, idx):
    """Get the trial outcome for a given trial index."""
    if trial_data['hit'][idx]:
        return 'hit'
    elif trial_data['miss'][idx]:
        return 'miss'
    elif trial_data['false_alarm'][idx]:
        return 'false_alarm'
    elif trial_data['correct_reject'][idx]:
        return 'correct_reject'
    else:
        return 'unknown'


def build_image_identity_trace(ophys_ts, stim_data, trial_start, trial_stop, image_names_list):
    """Build a time-varying image identity trace aligned to ophys timestamps for a trial window.

    During image presentation: index of the image in image_names_list
    During gray screen (ISI): index of GRAY_LABEL in image_names_list
    """
    # Get ophys frame indices within trial
    trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
    trial_ts = ophys_ts[trial_mask]
    n_frames = len(trial_ts)

    if n_frames == 0:
        return np.array([], dtype=np.int64), trial_mask

    # Initialize to gray (ISI)
    gray_idx = image_names_list.index(GRAY_LABEL)
    trace = np.full(n_frames, gray_idx, dtype=np.int64)

    # Map stimulus presentations onto ophys frames
    stim_starts = stim_data['start_time']
    stim_stops = stim_data['stop_time']
    stim_names = stim_data['image_name']

    # Only consider stimulus presentations that overlap with trial
    for si in range(len(stim_starts)):
        s_start = stim_starts[si]
        s_stop = stim_stops[si]

        if s_stop < trial_start:
            continue
        if s_start >= trial_stop:
            break

        name = stim_names[si]
        if isinstance(name, bytes):
            name = name.decode('utf-8')

        # Skip omitted stimuli (they're gray screen)
        if name == 'omitted':
            continue

        if name in image_names_list:
            img_idx = image_names_list.index(name)
        else:
            continue

        # Find ophys frames during this stimulus
        frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
        trace[frame_mask] = img_idx

    return trace, trial_mask


def build_image_change_trace(ophys_ts, stim_data, trial_start, trial_stop):
    """Build a binary trace that is 1 at the first frame after an image change."""
    trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
    trial_ts = ophys_ts[trial_mask]
    n_frames = len(trial_ts)

    if n_frames == 0:
        return np.array([], dtype=np.int64)

    trace = np.zeros(n_frames, dtype=np.int64)

    stim_starts = stim_data['start_time']
    is_change = stim_data['is_change']

    for si in range(len(stim_starts)):
        if not is_change[si]:
            continue

        s_start = stim_starts[si]
        if s_start < trial_start or s_start >= trial_stop:
            continue

        # Find the first ophys frame at or after the change onset
        frame_idx = np.searchsorted(trial_ts, s_start)
        if frame_idx < n_frames:
            trace[frame_idx] = 1

    return trace


def interpolate_to_ophys(signal, signal_ts, ophys_ts_trial):
    """Interpolate a behavioral signal to ophys timestamps using linear interpolation."""
    if len(signal) == 0 or len(ophys_ts_trial) == 0:
        return np.full(len(ophys_ts_trial), np.nan)

    f = interpolate.interp1d(signal_ts, signal, kind='linear',
                             bounds_error=False, fill_value=np.nan)
    return f(ophys_ts_trial)


def discretize_percentile(values, n_bins=5):
    """Discretize values into n_bins equal percentile bins.

    Returns bin indices (0 to n_bins-1). NaN values get bin 0 (will be handled separately).
    Percentiles computed from non-NaN values only.
    """
    valid = ~np.isnan(values)
    if valid.sum() == 0:
        return np.zeros(len(values), dtype=np.int64)

    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(values[valid], percentiles)

    # Make edges slightly wider to include boundary values
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    result = np.zeros(len(values), dtype=np.int64)
    result[valid] = np.clip(np.digitize(values[valid], bin_edges[1:-1]), 0, n_bins - 1)
    result[~valid] = 0  # NaN gets bin 0 (lowest bin)

    return result


def compute_session_percentile_edges(values, n_bins=5):
    """Compute percentile bin edges from session-wide values (excluding NaN)."""
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.linspace(0, 1, n_bins + 1)

    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def apply_percentile_bins(values, edges, n_bins=5):
    """Apply pre-computed percentile bin edges to values."""
    valid = ~np.isnan(values)
    result = np.zeros(len(values), dtype=np.int64)
    if valid.sum() > 0:
        result[valid] = np.clip(np.digitize(values[valid], edges[1:-1]), 0, n_bins - 1)
    result[~valid] = 0
    return result


def process_experiment(exp_id, exp_row, image_names_list, outcome_names, show_processing=False):
    """Process a single experiment and return trial-level data."""
    t0 = time.time()

    nwb_path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{exp_id}.nwb')
    if not os.path.exists(nwb_path):
        print(f"  WARNING: NWB file not found for experiment {exp_id}")
        return None

    # Load data
    nwb_data = load_nwb_data(nwb_path)
    t_load = time.time() - t0

    ophys_ts = nwb_data['ophys_timestamps']
    dff = nwb_data['dff_traces']  # (n_cells, n_frames)
    n_cells, n_frames = dff.shape

    if n_cells == 0:
        print(f"  WARNING: No cells in experiment {exp_id}")
        return None

    trial_data = nwb_data['trials']
    stim_data = nwb_data['stimulus']

    if stim_data is None:
        print(f"  WARNING: No stimulus data in experiment {exp_id}")
        return None

    # Get valid trial indices
    valid_trial_idx = get_valid_trials(trial_data)

    if len(valid_trial_idx) < 2:
        print(f"  WARNING: Only {len(valid_trial_idx)} valid trials in experiment {exp_id}")
        return None

    # Prepare behavioral signals for interpolation to ophys timestamps
    # Running speed - interpolate entire session to ophys timestamps
    running_at_ophys = interpolate_to_ophys(
        nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
    )

    # Pupil diameter - compute from area, handle blinks
    if nwb_data['pupil_area'] is not None:
        pupil_area = nwb_data['pupil_area'].copy()
        likely_blink = nwb_data['likely_blink']
        pupil_ts = nwb_data['pupil_timestamps']

        # Set blink frames to NaN
        pupil_area[likely_blink] = np.nan

        # Compute diameter from area: diameter = 2 * sqrt(area / pi)
        pupil_diameter = np.full_like(pupil_area, np.nan)
        valid_pupil = ~np.isnan(pupil_area) & (pupil_area > 0)
        pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)

        # Interpolate to ophys timestamps
        pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
    else:
        pupil_at_ophys = np.full(len(ophys_ts), np.nan)

    # Compute session-wide percentile bin edges
    running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)
    pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)

    # Process each valid trial
    neural_trials = []
    output_trials = []

    dt = np.median(np.diff(ophys_ts))  # time bin size

    for trial_idx in valid_trial_idx:
        t_start = trial_data['start_time'][trial_idx]
        t_stop = trial_data['stop_time'][trial_idx]

        # Get ophys frames for this trial
        frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
        trial_ts = ophys_ts[frame_mask]
        n_trial_frames = frame_mask.sum()

        if n_trial_frames < 2:
            continue

        # Neural data: dF/F for this trial
        neural = dff[:, frame_mask].astype(np.float32)  # (n_cells, n_trial_frames)

        # Output 0: Image identity (time-varying, categorical)
        img_trace, _ = build_image_identity_trace(
            ophys_ts, stim_data, t_start, t_stop, image_names_list
        )

        # Output 1: Image change (time-varying, binary)
        change_trace = build_image_change_trace(ophys_ts, stim_data, t_start, t_stop)

        # Output 2: Running speed binned (time-varying)
        running_trial = running_at_ophys[frame_mask]
        running_binned = apply_percentile_bins(running_trial, running_edges, n_bins=5)

        # Output 3: Pupil diameter binned (time-varying)
        pupil_trial = pupil_at_ophys[frame_mask]
        pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)

        # Output 4: Trial outcome (static per trial)
        outcome = get_trial_outcome(trial_data, trial_idx)
        outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0

        # Stack outputs
        # Time-varying outputs: (4, n_trial_frames)
        # Static output: (1,)
        output_tv = np.stack([img_trace, change_trace, running_binned, pupil_binned], axis=0).astype(np.int64)
        output_static = np.array([outcome_idx], dtype=np.int64)

        # Combine: (5, n_trial_frames) for time-varying, last row repeated
        # Actually, static outputs should be (n_output,) shape
        # Let's make time-varying as (4, T) and static as (1,)
        # Combined: need to handle mixed time-varying and static
        # The format says: (n_output, n_timepoints) or (n_output,)
        # We'll put all 5 outputs together: first 4 are (T,), last is scalar
        # But the format expects a single array per trial...
        # Let's make output as (5, T) where last row is constant
        output_full = np.zeros((5, n_trial_frames), dtype=np.int64)
        output_full[0] = img_trace
        output_full[1] = change_trace
        output_full[2] = running_binned
        output_full[3] = pupil_binned
        output_full[4] = outcome_idx  # broadcast scalar to all timepoints

        neural_trials.append(neural)
        output_trials.append(output_full)

    t_process = time.time() - t0 - t_load

    if len(neural_trials) < 2:
        print(f"  WARNING: Only {len(neural_trials)} valid trials after processing for experiment {exp_id}")
        return None

    # Brain region
    brain_region = exp_row['targeted_structure']
    mouse_id = str(exp_row['mouse_id'])

    result = {
        'neural': neural_trials,
        'output': output_trials,
        'brain_region': brain_region,
        'mouse_id': mouse_id,
        'n_cells': n_cells,
        'n_trials': len(neural_trials),
        'dt': float(dt),
        'exp_id': int(exp_id),
        'ophys_session_id': int(exp_row['ophys_session_id']),
        'session_type': exp_row['session_type'],
        'cre_line': exp_row['cre_line'],
        'imaging_depth': int(exp_row['imaging_depth']),
        't_load': t_load,
        't_process': t_process,
    }

    if show_processing:
        result['ophys_ts'] = ophys_ts
        result['running_at_ophys'] = running_at_ophys
        result['pupil_at_ophys'] = pupil_at_ophys
        result['running_edges'] = running_edges
        result['pupil_edges'] = pupil_edges
        result['trial_data'] = trial_data
        result['stim_data'] = stim_data
        result['valid_trial_idx'] = valid_trial_idx

    return result


def get_all_image_names(exp_table):
    """Scan a few NWB files to collect all unique image names across sessions."""
    all_names = set()
    for _, row in exp_table.iterrows():
        eid = row['ophys_experiment_id']
        nwb_path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{eid}.nwb')
        try:
            with h5py.File(nwb_path, 'r') as f:
                for key in f['intervals'].keys():
                    if 'Natural_Images' in key or 'natural_images' in key:
                        names = f['intervals'][key]['image_name'][:]
                        for n in names:
                            if isinstance(n, bytes):
                                n = n.decode('utf-8')
                            if n != 'omitted':
                                all_names.add(n)
                        break
        except Exception as e:
            print(f"  WARNING: Could not read images from {eid}: {e}")

    return sorted(all_names)


def plot_processing(result, session_idx, image_names_list, output_dir='.'):
    """Plot processing visualizations for a session."""
    ophys_ts = result['ophys_ts']
    trial_data = result['trial_data']
    stim_data = result['stim_data']
    valid_idx = result['valid_trial_idx']

    # Pick first 3 valid trials to plot
    n_plot = min(3, len(valid_idx))

    fig, axes = plt.subplots(n_plot, 5, figsize=(30, 4 * n_plot))
    if n_plot == 1:
        axes = axes[np.newaxis, :]

    for ti in range(n_plot):
        trial_i = valid_idx[ti]
        t_start = trial_data['start_time'][trial_i]
        t_stop = trial_data['stop_time'][trial_i]
        frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
        trial_ts = ophys_ts[frame_mask]
        trial_ts_rel = trial_ts - t_start

        # Col 0: Neural dF/F (first 5 cells)
        ax = axes[ti, 0]
        neural = result['neural'][ti]
        n_show = min(5, neural.shape[0])
        for ci in range(n_show):
            ax.plot(trial_ts_rel, neural[ci] + ci * 1.0, linewidth=0.5)
        ax.set_ylabel('dF/F + offset')
        ax.set_title(f'Trial {trial_i}: Neural (first {n_show} cells)')

        # Col 1: Image identity
        ax = axes[ti, 1]
        output = result['output'][ti]
        ax.plot(trial_ts_rel, output[0], 'k-', linewidth=0.8)
        ax.set_ylabel('Image ID')
        ax.set_title('Image Identity')
        ax.set_yticks(range(len(image_names_list)))
        ax.set_yticklabels(image_names_list, fontsize=6)

        # Col 2: Image change
        ax = axes[ti, 2]
        ax.plot(trial_ts_rel, output[1], 'r-', linewidth=0.8)
        ax.set_ylabel('Change')
        ax.set_title('Image Change')

        # Col 3: Running speed (binned)
        ax = axes[ti, 3]
        ax.plot(trial_ts_rel, output[2], 'b-', linewidth=0.8)
        running_raw = result['running_at_ophys'][frame_mask]
        ax2 = ax.twinx()
        ax2.plot(trial_ts_rel, running_raw, 'b--', alpha=0.3, linewidth=0.5)
        ax2.set_ylabel('cm/s', color='b', alpha=0.3)
        ax.set_ylabel('Speed Bin')
        ax.set_title('Running Speed')

        # Col 4: Pupil diameter (binned)
        ax = axes[ti, 4]
        ax.plot(trial_ts_rel, output[3], 'g-', linewidth=0.8)
        pupil_raw = result['pupil_at_ophys'][frame_mask]
        ax2 = ax.twinx()
        ax2.plot(trial_ts_rel, pupil_raw, 'g--', alpha=0.3, linewidth=0.5)
        ax2.set_ylabel('pixels', color='g', alpha=0.3)
        ax.set_ylabel('Pupil Bin')
        ax.set_title('Pupil Diameter')

        for col in range(5):
            axes[ti, col].set_xlabel('Time from trial start (s)')

    fig.suptitle(f'Session {session_idx} (exp {result["exp_id"]}): Processing Visualization', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'processing_{result["exp_id"]}.png'), dpi=100)
    plt.close(fig)
    print(f"  Saved processing plot for experiment {result['exp_id']}")


# ==============================================================================
# Main conversion
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Convert Visual Behavior 2P data to decoder format.')
    parser.add_argument('output', type=str, help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing steps')
    args = parser.parse_args()

    t_start_total = time.time()

    print("=" * 60)
    print("Visual Behavior 2P Data Conversion")
    print("=" * 60)

    # Load experiment table
    print("\nLoading experiment table...")
    exp_table = load_experiment_table()
    print(f"  Total active experiments with NWB files: {len(exp_table)}")
    print(f"  Unique sessions: {exp_table['ophys_session_id'].nunique()}")
    print(f"  Unique mice: {exp_table['mouse_id'].nunique()}")
    print(f"  Targeted structures: {exp_table['targeted_structure'].value_counts().to_dict()}")

    if args.sample:
        # Pick 2 experiments from different mice
        sample_mice = exp_table['mouse_id'].unique()[:2]
        exp_table = exp_table[exp_table['mouse_id'].isin(sample_mice)].head(2).copy()
        print(f"\n  SAMPLE MODE: Processing {len(exp_table)} experiments")

    # Collect all image names across experiments
    print("\nCollecting image names across all experiments...")
    t0 = time.time()
    all_image_names = get_all_image_names(exp_table)
    # Add gray screen label
    image_names_list = [GRAY_LABEL] + all_image_names
    print(f"  Found {len(all_image_names)} unique images: {all_image_names}")
    print(f"  Image names list (with gray): {image_names_list}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # Define output variables
    outcome_names = ['hit', 'miss', 'false_alarm', 'correct_reject']

    output_names = ['image_identity', 'image_change', 'running_speed', 'pupil_diameter', 'trial_outcome']
    output_values = [
        image_names_list,          # image identity categories
        ['no_change', 'change'],   # image change: 0=no change, 1=change
        [f'bin_{i}' for i in range(5)],  # running speed percentile bins
        [f'bin_{i}' for i in range(5)],  # pupil diameter percentile bins
        outcome_names,             # trial outcomes
    ]

    # No decoder inputs
    input_names = []

    # Process each experiment
    print(f"\nProcessing {len(exp_table)} experiments...")

    all_neural = []
    all_input = []
    all_output = []
    all_subject_idx = []
    all_brain_region_idx = []
    all_subjects = []
    all_brain_regions = []
    session_metadata = []

    subject_map = {}  # mouse_id -> index
    region_map = {}   # brain_region -> index

    for idx, (_, row) in enumerate(exp_table.iterrows()):
        exp_id = row['ophys_experiment_id']
        t0 = time.time()
        print(f"\n[{idx+1}/{len(exp_table)}] Processing experiment {exp_id} "
              f"(mouse={row['mouse_id']}, region={row['targeted_structure']}, "
              f"type={row['session_type']})...")

        result = process_experiment(
            exp_id, row, image_names_list, outcome_names,
            show_processing=args.show_processing
        )

        if result is None:
            continue

        t_total = time.time() - t0
        print(f"  Cells: {result['n_cells']}, Trials: {result['n_trials']}, "
              f"dt: {result['dt']*1000:.1f}ms, "
              f"Time: {t_total:.1f}s (load={result['t_load']:.1f}s, process={result['t_process']:.1f}s)")

        # Map subject
        mouse_id = result['mouse_id']
        if mouse_id not in subject_map:
            subject_map[mouse_id] = len(all_subjects)
            all_subjects.append(mouse_id)

        # Map brain region
        brain_region = result['brain_region']
        if brain_region not in region_map:
            region_map[brain_region] = len(all_brain_regions)
            all_brain_regions.append(brain_region)

        # Neural and output data
        all_neural.append(result['neural'])
        all_output.append(result['output'])

        # Input: empty for each trial
        all_input.append([np.zeros((0, trial.shape[1]), dtype=np.float32) for trial in result['neural']])

        all_subject_idx.append(subject_map[mouse_id])
        all_brain_region_idx.append(
            np.full(result['n_cells'], region_map[brain_region], dtype=np.int64)
        )

        session_metadata.append({
            'exp_id': result['exp_id'],
            'ophys_session_id': result['ophys_session_id'],
            'session_type': result['session_type'],
            'cre_line': result['cre_line'],
            'imaging_depth': result['imaging_depth'],
            'mouse_id': mouse_id,
            'brain_region': brain_region,
            'n_cells': result['n_cells'],
            'n_trials': result['n_trials'],
            'dt_ms': result['dt'] * 1000,
        })

        # Show processing plots
        if args.show_processing:
            plot_processing(result, idx, image_names_list)

    if len(all_neural) == 0:
        print("ERROR: No valid sessions processed!")
        sys.exit(1)

    # Compute median time bin size across sessions
    all_dts = [m['dt_ms'] for m in session_metadata]
    median_dt = np.median(all_dts)

    # Assemble final data structure
    print(f"\nAssembling final data structure...")
    print(f"  Sessions: {len(all_neural)}")
    print(f"  Subjects: {len(all_subjects)}")
    print(f"  Brain regions: {all_brain_regions}")
    total_trials = sum(len(s) for s in all_neural)
    total_neurons = sum(m['n_cells'] for m in session_metadata)
    print(f"  Total trials: {total_trials}")
    print(f"  Total neurons: {total_neurons}")

    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': all_subjects,
        'subject_idx': np.array(all_subject_idx, dtype=np.int64),
        'brain_regions': all_brain_regions,
        'brain_region_idx': all_brain_region_idx,
        'input_names': input_names,
        'output_names': output_names,
        'output_values': output_values,
        'metadata': {
            'task_description': 'Visual behavior change detection task: mice view flashed natural images and lick to report image identity changes. Decoder predicts image identity, change events, running speed, pupil diameter, and trial outcome from dF/F neural activity.',
            'time_bin_size': median_dt,
            'temporal_alignment_event': 'Aligned to ophys (2-photon imaging) timestamps. Each trial spans from trial start_time to stop_time.',
            'off_start': None,
            'off_end': None,
            'session_info': session_metadata,
            'stimulus_duration_ms': 250.0,
            'inter_stimulus_interval_ms': 500.0,
            'flash_cycle_ms': 750.0,
            'ophys_frame_rate_hz': 1000.0 / median_dt,
            'n_images': len(all_image_names),
            'image_names': all_image_names,
        }
    }

    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"  File size: {file_size:.1f} MB")

    t_total = time.time() - t_start_total
    print(f"\nTotal conversion time: {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"Average time per session: {t_total/len(all_neural):.1f}s")
    print("Done!")


if __name__ == '__main__':
    main()
