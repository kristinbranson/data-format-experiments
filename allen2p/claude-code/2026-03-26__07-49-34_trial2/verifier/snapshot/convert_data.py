#!/usr/bin/env python3
"""
Convert Allen Brain Observatory Visual Behavior 2P data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--sample] [--show-processing]

    --full (default): Process all active sessions
    --sample: Process only 2 sessions for testing
    --show-processing: Plot visualizations of processing steps (up to 2 sessions)
"""

import argparse
import numpy as np
import h5py
import pandas as pd
import pickle
import time
import os
import glob
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# Configuration
# ============================================================================
NWB_DIR = 'data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/'
METADATA_DIR = 'data/visual-behavior-ophys-1.1.0/project_metadata/'
TIME_BIN_MS = 750.0  # One bin per stimulus presentation (250ms image + 500ms grey)
N_PERCENTILE_BINS = 5  # For running speed and pupil diameter discretization

# Image names for consistent categorical encoding
IMAGE_NAMES_GLOBAL = None  # Will be populated from data

# ============================================================================
# Helper functions
# ============================================================================

def get_experiment_metadata():
    """Load experiment metadata and identify available NWB files."""
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
    nwb_map = {}
    for f in nwb_files:
        eid = int(os.path.basename(f).replace('behavior_ophys_experiment_', '').replace('.nwb', ''))
        nwb_map[eid] = f

    our_exps = exp_table[exp_table['ophys_experiment_id'].isin(nwb_map.keys())].copy()
    # Only active behavior sessions
    active_exps = our_exps[our_exps['passive'] == False].copy()
    active_exps = active_exps.sort_values('ophys_experiment_id').reset_index(drop=True)
    return active_exps, nwb_map


def interpolate_nans(values):
    """Linearly interpolate NaN values in a 1D array."""
    valid = ~np.isnan(values)
    if valid.sum() == 0:
        return values  # All NaN, can't interpolate
    if valid.sum() == len(values):
        return values  # No NaN
    result = values.copy()
    x = np.arange(len(values))
    result[~valid] = np.interp(x[~valid], x[valid], values[valid])
    return result


def find_stim_key(f):
    """Find the stimulus presentations key for change detection task."""
    for key in f['intervals']:
        if key == 'trials' or key.startswith('spontaneous') or key.startswith('natural_movie'):
            continue
        # Should be the natural images / change detection key
        return key
    return None


def process_experiment(nwb_path, exp_info, collect_stats_only=False):
    """
    Process a single NWB experiment file.

    Returns:
        dict with 'neural_trials', 'output_trials', 'input_trials',
        'n_neurons', 'brain_region', 'mouse_id', 'experiment_id'

    If collect_stats_only=True, returns only running/pupil values for percentile computation.
    """
    t0 = time.time()

    with h5py.File(nwb_path, 'r') as f:
        experiment_id = exp_info['ophys_experiment_id']

        # --- Load neural data ---
        dff_data = f['processing']['ophys']['dff']['traces']['data'][:]  # (timepoints, neurons)
        ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][:]

        # Load cell info and filter to valid ROIs
        cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
        valid_roi = cell_table['valid_roi'][:].astype(bool)
        n_total_rois = dff_data.shape[1]

        dff_valid = dff_data[:, valid_roi]  # (timepoints, n_valid_neurons)
        n_neurons = dff_valid.shape[1]

        if n_neurons == 0:
            print(f"  WARNING: Experiment {experiment_id} has 0 valid neurons, skipping")
            return None

        # --- Load stimulus presentations ---
        stim_key = find_stim_key(f)
        if stim_key is None:
            print(f"  WARNING: No stimulus presentations found for {experiment_id}, skipping")
            return None

        stim = f['intervals'][stim_key]
        stim_start = stim['start_time'][:]
        stim_stop = stim['stop_time'][:]
        stim_image_name = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in stim['image_name'][:]])
        stim_is_change = stim['is_change'][:]
        stim_omitted = stim['omitted'][:]
        stim_trials_id = stim['trials_id'][:]

        # --- Load trials ---
        trials = f['intervals']['trials']
        trial_ids = trials['id'][:]
        trial_go = trials['go'][:].astype(bool)
        trial_catch = trials['catch'][:].astype(bool)
        trial_aborted = trials['aborted'][:].astype(bool)
        trial_auto = trials['auto_rewarded'][:].astype(bool)
        trial_hit = trials['hit'][:].astype(bool)
        trial_miss = trials['miss'][:].astype(bool)
        trial_fa = trials['false_alarm'][:].astype(bool)
        trial_cr = trials['correct_reject'][:].astype(bool)
        trial_start = trials['start_time'][:]
        trial_stop = trials['stop_time'][:]
        trial_change_time = trials['change_time'][:]

        # Filter: Go + Catch, exclude Aborted and Auto-rewarded
        trial_mask = (trial_go | trial_catch) & ~trial_aborted & ~trial_auto
        valid_trial_ids = trial_ids[trial_mask]

        # --- Load running speed ---
        running_speed = f['processing']['running']['speed']['data'][:]
        running_ts = f['processing']['running']['speed']['timestamps'][:]

        # --- Load pupil data ---
        has_eye_tracking = 'EyeTracking' in f['acquisition']
        if has_eye_tracking:
            pupil_area_raw = f['acquisition']['EyeTracking']['pupil_tracking']['area'][:]
            pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][:]
            likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][:].astype(bool)

            # Set blink frames to NaN and interpolate
            pupil_area = pupil_area_raw.copy().astype(float)
            pupil_area[likely_blink] = np.nan
            # Also NaN out negative or zero values
            pupil_area[pupil_area <= 0] = np.nan
            # Compute diameter from area: d = 2*sqrt(area/pi)
            pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
            # Interpolate NaNs
            pupil_diameter = interpolate_nans(pupil_diameter)
        else:
            pupil_ts = None
            pupil_diameter = None

        # --- Precompute bin assignments for vectorized averaging ---
        # For each stimulus presentation, find ophys frame indices in [start, start+0.75)
        bin_duration = TIME_BIN_MS / 1000.0  # 0.75s

        # Precompute all stimulus bin boundaries for the entire session
        all_stim_starts = stim_start
        all_stim_ends = all_stim_starts + bin_duration

        # Assign ophys frames to stimulus bins using searchsorted
        ophys_bin_starts = np.searchsorted(ophys_ts, all_stim_starts, side='left')
        ophys_bin_ends = np.searchsorted(ophys_ts, all_stim_ends, side='left')

        # Precompute cumulative sum for fast bin averaging of neural data
        dff_cumsum = np.cumsum(dff_valid, axis=0)
        dff_cumsum = np.vstack([np.zeros((1, n_neurons), dtype=dff_cumsum.dtype), dff_cumsum])

        # Precompute running speed bin assignments
        run_bin_starts = np.searchsorted(running_ts, all_stim_starts, side='left')
        run_bin_ends = np.searchsorted(running_ts, all_stim_ends, side='left')
        run_cumsum = np.cumsum(running_speed)
        run_cumsum = np.concatenate([[0], run_cumsum])

        # Precompute pupil bin assignments
        if has_eye_tracking and pupil_diameter is not None:
            pup_bin_starts = np.searchsorted(pupil_ts, all_stim_starts, side='left')
            pup_bin_ends = np.searchsorted(pupil_ts, all_stim_ends, side='left')
            # For pupil, need to handle NaN so can't use simple cumsum
            # Use cumsum of valid values and counts
            pup_valid = ~np.isnan(pupil_diameter)
            pup_filled = np.where(pup_valid, pupil_diameter, 0.0)
            pup_cumsum = np.cumsum(pup_filled)
            pup_cumsum = np.concatenate([[0], pup_cumsum])
            pup_count_cumsum = np.cumsum(pup_valid.astype(np.float64))
            pup_count_cumsum = np.concatenate([[0], pup_count_cumsum])

        # Build image name to index mapping
        img_to_idx = {name: i for i, name in enumerate(IMAGE_NAMES_GLOBAL)}

        # --- Build per-trial data ---
        neural_trials = []
        output_trials = []
        input_trials = []
        running_values = []
        pupil_values = []

        for trial_idx in np.where(trial_mask)[0]:
            tid = trial_ids[trial_idx]

            # Find stimulus presentations for this trial
            trial_stim_indices = np.where(stim_trials_id == tid)[0]
            if len(trial_stim_indices) == 0:
                continue

            n_bins = len(trial_stim_indices)

            # --- Neural data: vectorized bin averaging ---
            neural_matrix = np.zeros((n_neurons, n_bins), dtype=np.float32)
            for bi, si in enumerate(trial_stim_indices):
                s, e = ophys_bin_starts[si], ophys_bin_ends[si]
                n_frames = e - s
                if n_frames > 0:
                    neural_matrix[:, bi] = (dff_cumsum[e] - dff_cumsum[s]) / n_frames

            # --- Image identity ---
            images = stim_image_name[trial_stim_indices].copy()
            # Forward-fill omitted presentations
            for bi in range(len(images)):
                if images[bi] == 'omitted':
                    if bi > 0:
                        images[bi] = images[bi - 1]
                    else:
                        for bj in range(bi + 1, len(images)):
                            if images[bj] != 'omitted':
                                images[bi] = images[bj]
                                break

            image_indices = np.array([img_to_idx.get(img, 0) for img in images], dtype=np.int64)

            # --- Image change ---
            change_flags = np.nan_to_num(stim_is_change[trial_stim_indices], nan=0).astype(np.int64)

            # --- Running speed: vectorized bin averaging ---
            running_binned = np.zeros(n_bins, dtype=np.float32)
            for bi, si in enumerate(trial_stim_indices):
                s, e = run_bin_starts[si], run_bin_ends[si]
                n_pts = e - s
                if n_pts > 0:
                    running_binned[bi] = (run_cumsum[e] - run_cumsum[s]) / n_pts

            running_values.extend(running_binned.tolist())

            # --- Pupil diameter: vectorized bin averaging ---
            pupil_binned = np.full(n_bins, np.nan, dtype=np.float32)
            if has_eye_tracking and pupil_diameter is not None:
                for bi, si in enumerate(trial_stim_indices):
                    s, e = pup_bin_starts[si], pup_bin_ends[si]
                    n_valid = pup_count_cumsum[e] - pup_count_cumsum[s]
                    if n_valid > 0:
                        pupil_binned[bi] = (pup_cumsum[e] - pup_cumsum[s]) / n_valid

            pupil_values.extend(pupil_binned.tolist())

            # --- Trial outcome ---
            if trial_hit[trial_idx]:
                outcome = 0  # Hit
            elif trial_miss[trial_idx]:
                outcome = 1  # Miss
            elif trial_fa[trial_idx]:
                outcome = 2  # False Alarm
            elif trial_cr[trial_idx]:
                outcome = 3  # Correct Rejection
            else:
                outcome = 1  # Default to Miss

            if collect_stats_only:
                continue

            neural_trials.append(neural_matrix)
            output_trials.append({
                'image_identity': image_indices,
                'image_change': change_flags,
                'running_speed_raw': running_binned,
                'pupil_diameter_raw': pupil_binned,
                'trial_outcome': outcome,
                'n_bins': n_bins,
            })
            input_trials.append(np.zeros((0, n_bins), dtype=np.float32))

    elapsed = time.time() - t0

    if collect_stats_only:
        return {
            'running_values': running_values,
            'pupil_values': pupil_values,
        }

    return {
        'neural_trials': neural_trials,
        'output_trials': output_trials,
        'input_trials': input_trials,
        'n_neurons': n_neurons,
        'brain_region': exp_info['targeted_structure'],
        'mouse_id': str(exp_info['mouse_id']),
        'experiment_id': experiment_id,
        'n_trials': len(neural_trials),
        'elapsed': elapsed,
    }


def collect_all_image_names(active_exps, nwb_map):
    """Collect all unique image names across experiments (sample a few to be fast)."""
    all_images = set()
    # Sample up to 10 experiments to collect image names (they use the same 8 images)
    sample_exps = active_exps.head(min(10, len(active_exps)))
    for _, row in sample_exps.iterrows():
        nwb_path = nwb_map[row['ophys_experiment_id']]
        with h5py.File(nwb_path, 'r') as f:
            stim_key = find_stim_key(f)
            if stim_key is None:
                continue
            names = f['intervals'][stim_key]['image_name'][:]
            for n in names:
                name = n.decode() if isinstance(n, bytes) else str(n)
                if name != 'omitted':
                    all_images.add(name)
    return sorted(all_images)


def discretize_values(values, n_bins, bin_edges=None):
    """Discretize continuous values into equal percentile bins.

    Returns:
        binned: array of bin indices (0 to n_bins-1)
        bin_edges: percentile bin edges used
    """
    if bin_edges is None:
        valid = values[~np.isnan(values)]
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(valid, percentiles)
        # Ensure unique edges
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

    binned = np.digitize(values, bin_edges[1:-1]).astype(np.int64)  # 0 to n_bins-1
    # Clip to valid range
    binned = np.clip(binned, 0, n_bins - 1)
    return binned, bin_edges


def plot_processing(session_data, session_idx, nwb_path, exp_info):
    """Plot processing visualizations for a single session."""
    fig, axes = plt.subplots(6, 2, figsize=(20, 24))
    fig.suptitle(f"Processing Visualization - Experiment {exp_info['ophys_experiment_id']}\n"
                 f"Mouse {exp_info['mouse_id']}, {exp_info['targeted_structure']}, {exp_info['cre_line']}",
                 fontsize=14)

    neural_trials = session_data['neural_trials']
    output_trials = session_data['output_trials']

    if len(neural_trials) == 0:
        plt.savefig(f"processing_{exp_info['ophys_experiment_id']}.png", dpi=100)
        plt.close()
        return

    # Pick 2 representative trials
    trial_indices = [0, min(len(neural_trials) - 1, len(neural_trials) // 2)]

    for col, ti in enumerate(trial_indices):
        nt = neural_trials[ti]
        ot = output_trials[ti]
        n_bins = ot['n_bins']
        time_axis = np.arange(n_bins) * 0.75  # seconds

        # 1. Neural activity heatmap
        ax = axes[0, col]
        n_show = min(nt.shape[0], 20)
        if n_show > 0:
            ax.imshow(nt[:n_show, :], aspect='auto', interpolation='nearest',
                     extent=[0, n_bins * 0.75, n_show, 0])
            ax.set_ylabel('Neuron')
        ax.set_title(f'Trial {ti}: dF/F (first {n_show} neurons)')
        ax.set_xlabel('Time (s)')

        # 2. Image identity
        ax = axes[1, col]
        ax.plot(time_axis, ot['image_identity'], 'b-o', markersize=3)
        ax.set_ylabel('Image Index')
        ax.set_title(f'Image Identity')
        ax.set_xlabel('Time (s)')

        # 3. Image change
        ax = axes[2, col]
        ax.plot(time_axis, ot['image_change'], 'r-o', markersize=3)
        ax.set_ylabel('Change (0/1)')
        ax.set_title(f'Image Change')
        ax.set_xlabel('Time (s)')

        # 4. Running speed (raw)
        ax = axes[3, col]
        ax.plot(time_axis, ot['running_speed_raw'], 'g-', linewidth=1)
        ax.set_ylabel('Speed (cm/s)')
        ax.set_title(f'Running Speed (raw)')
        ax.set_xlabel('Time (s)')

        # 5. Pupil diameter (raw)
        ax = axes[4, col]
        ax.plot(time_axis, ot['pupil_diameter_raw'], 'm-', linewidth=1)
        ax.set_ylabel('Diameter (px)')
        ax.set_title(f'Pupil Diameter (raw)')
        ax.set_xlabel('Time (s)')

        # 6. Trial outcome
        ax = axes[5, col]
        outcome_names = ['Hit', 'Miss', 'False Alarm', 'Correct Rejection']
        outcome = ot['trial_outcome']
        ax.text(0.5, 0.5, f"Trial Outcome: {outcome_names[outcome]}",
                transform=ax.transAxes, fontsize=14, ha='center', va='center')
        ax.set_title('Trial Outcome')

    plt.tight_layout()
    plt.savefig(f"processing_{exp_info['ophys_experiment_id']}.png", dpi=100, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main conversion
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Convert Visual Behavior 2P data')
    parser.add_argument('output_file', type=str, help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    args = parser.parse_args()

    total_start = time.time()

    # ---- Step 1: Load metadata ----
    print("=" * 60)
    print("Loading experiment metadata...")
    t0 = time.time()
    active_exps, nwb_map = get_experiment_metadata()
    print(f"  Found {len(active_exps)} active experiments from {active_exps['mouse_id'].nunique()} mice")
    print(f"  Brain regions: {active_exps['targeted_structure'].value_counts().to_dict()}")
    print(f"  Metadata loaded in {time.time()-t0:.1f}s")

    if args.sample:
        # Select 2 experiments from different mice if possible
        sample_exps = active_exps.groupby('mouse_id').first().reset_index()
        sample_exps = sample_exps.head(2)
        exp_ids = sample_exps['ophys_experiment_id'].values
        active_exps = active_exps[active_exps['ophys_experiment_id'].isin(exp_ids)].copy()
        print(f"  SAMPLE MODE: Processing {len(active_exps)} experiments")

    # ---- Step 2: Collect all image names ----
    global IMAGE_NAMES_GLOBAL
    print("\nCollecting image names across all experiments...")
    t0 = time.time()
    IMAGE_NAMES_GLOBAL = collect_all_image_names(active_exps, nwb_map)
    print(f"  Found {len(IMAGE_NAMES_GLOBAL)} unique images: {IMAGE_NAMES_GLOBAL}")
    print(f"  Image names collected in {time.time()-t0:.1f}s")

    # ---- Step 3: First pass - collect running/pupil stats for percentile bins ----
    print("\nPass 1: Collecting running speed and pupil statistics...")
    t0 = time.time()
    all_running = []
    all_pupil = []

    for idx, (_, row) in enumerate(active_exps.iterrows()):
        eid = row['ophys_experiment_id']
        nwb_path = nwb_map[eid]
        stats = process_experiment(nwb_path, row, collect_stats_only=True)
        if stats is not None:
            all_running.extend(stats['running_values'])
            all_pupil.extend(stats['pupil_values'])
        if (idx + 1) % 20 == 0 or idx == len(active_exps) - 1:
            print(f"  Pass 1: {idx+1}/{len(active_exps)} experiments processed")

    all_running = np.array(all_running, dtype=np.float64)
    all_pupil = np.array(all_pupil, dtype=np.float64)

    # Compute percentile bin edges
    _, running_bin_edges = discretize_values(all_running, N_PERCENTILE_BINS)

    # For pupil, exclude NaN
    valid_pupil = all_pupil[~np.isnan(all_pupil)]
    if len(valid_pupil) > 0:
        _, pupil_bin_edges = discretize_values(valid_pupil, N_PERCENTILE_BINS)
    else:
        pupil_bin_edges = np.array([-np.inf, 0, 1, 2, 3, np.inf])

    print(f"  Running speed percentile edges: {running_bin_edges}")
    print(f"  Pupil diameter percentile edges: {pupil_bin_edges}")
    print(f"  Pass 1 completed in {time.time()-t0:.1f}s")

    # ---- Step 4: Second pass - full processing ----
    print("\nPass 2: Full data conversion...")
    t0 = time.time()

    all_sessions_neural = []
    all_sessions_input = []
    all_sessions_output = []
    all_brain_region_idx = []
    all_subject_idx = []

    subjects_list = sorted(active_exps['mouse_id'].unique().astype(str).tolist())
    brain_regions_list = sorted(active_exps['targeted_structure'].unique().tolist())

    session_info = []
    skipped = 0

    for idx, (_, row) in enumerate(active_exps.iterrows()):
        eid = row['ophys_experiment_id']
        nwb_path = nwb_map[eid]

        result = process_experiment(nwb_path, row, collect_stats_only=False)

        if result is None or result['n_trials'] < 2:
            skipped += 1
            reason = 'no result' if result is None else f'only {result["n_trials"]} trials'
            print(f"  Skipped experiment {eid}: {reason}")
            continue

        # Discretize running speed and pupil for this session's trials
        session_output = []
        for ot in result['output_trials']:
            n_bins = ot['n_bins']

            # Discretize running speed
            running_disc, _ = discretize_values(ot['running_speed_raw'], N_PERCENTILE_BINS, running_bin_edges)

            # Discretize pupil diameter
            pupil_raw = ot['pupil_diameter_raw']
            # Handle remaining NaN in pupil (fill with median bin)
            nan_mask = np.isnan(pupil_raw)
            if nan_mask.all():
                pupil_disc = np.full(n_bins, N_PERCENTILE_BINS // 2, dtype=np.int64)  # middle bin
            else:
                if nan_mask.any():
                    pupil_raw = interpolate_nans(pupil_raw)
                pupil_disc, _ = discretize_values(pupil_raw, N_PERCENTILE_BINS, pupil_bin_edges)

            # Build output array: (5, n_bins) for time-varying, (5,) for trial_outcome
            # Time-varying outputs: image_identity, image_change, running_speed, pupil_diameter
            # Static output: trial_outcome (replicate across time)
            output_arr = np.stack([
                ot['image_identity'].astype(np.int64),
                ot['image_change'].astype(np.int64),
                running_disc.astype(np.int64),
                pupil_disc.astype(np.int64),
                np.full(n_bins, ot['trial_outcome'], dtype=np.int64),
            ], axis=0)  # (5, n_bins)

            session_output.append(output_arr)

        # Store session data
        all_sessions_neural.append(result['neural_trials'])
        all_sessions_input.append(result['input_trials'])
        all_sessions_output.append(session_output)

        # Brain region index for each neuron
        region_idx = np.full(result['n_neurons'],
                            brain_regions_list.index(result['brain_region']),
                            dtype=np.int64)
        all_brain_region_idx.append(region_idx)

        # Subject index
        subject_idx = subjects_list.index(result['mouse_id'])
        all_subject_idx.append(subject_idx)

        session_info.append({
            'experiment_id': eid,
            'mouse_id': result['mouse_id'],
            'brain_region': result['brain_region'],
            'n_neurons': result['n_neurons'],
            'n_trials': result['n_trials'],
        })

        if (idx + 1) % 10 == 0 or idx == len(active_exps) - 1:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            remaining = (len(active_exps) - idx - 1) / rate
            print(f"  Pass 2: {idx+1}/{len(active_exps)} experiments "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

        # Plot processing if requested
        if args.show_processing and idx < 2:
            plot_processing(result, idx, nwb_path, row)
            print(f"  Saved processing plot: processing_{eid}.png")

    print(f"\nPass 2 completed in {time.time()-t0:.1f}s")
    print(f"  Processed: {len(all_sessions_neural)} sessions, Skipped: {skipped}")

    # ---- Step 5: Build output dictionary ----
    print("\nBuilding output dictionary...")

    # Output value labels
    image_value_names = IMAGE_NAMES_GLOBAL  # List of image names
    change_value_names = ['no_change', 'change']
    running_value_names = [f'speed_q{i+1}' for i in range(N_PERCENTILE_BINS)]
    pupil_value_names = [f'pupil_q{i+1}' for i in range(N_PERCENTILE_BINS)]
    outcome_value_names = ['hit', 'miss', 'false_alarm', 'correct_rejection']

    data = {
        'neural': all_sessions_neural,
        'input': all_sessions_input,
        'output': all_sessions_output,

        'subjects': subjects_list,
        'subject_idx': np.array(all_subject_idx, dtype=np.int64),

        'brain_regions': brain_regions_list,
        'brain_region_idx': all_brain_region_idx,

        'input_names': [],
        'output_names': ['image_identity', 'image_change', 'running_speed', 'pupil_diameter', 'trial_outcome'],
        'output_values': [
            image_value_names,
            change_value_names,
            running_value_names,
            pupil_value_names,
            outcome_value_names,
        ],

        'metadata': {
            'task_description': 'Go/no-go visual change detection: mice lick when image identity changes. '
                              'Outputs: image identity, image change, running speed (5 bins), '
                              'pupil diameter (5 bins), trial outcome (hit/miss/FA/CR).',
            'time_bin_size': TIME_BIN_MS,
            'temporal_alignment_event': 'Stimulus presentation onset (each 750ms image flash)',
            'off_start': None,
            'off_end': None,
            'running_speed_bin_edges': running_bin_edges.tolist(),
            'pupil_diameter_bin_edges': pupil_bin_edges.tolist(),
            'session_info': session_info,
            'n_sessions': len(all_sessions_neural),
            'n_subjects': len(subjects_list),
            'brain_regions': brain_regions_list,
        }
    }

    # ---- Step 6: Save ----
    print(f"\nSaving to {args.output_file}...")
    t0 = time.time()
    with open(args.output_file, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    file_size = os.path.getsize(args.output_file) / (1024 * 1024)
    print(f"  Saved {file_size:.1f} MB in {time.time()-t0:.1f}s")

    # ---- Summary statistics ----
    total_neurons = sum(s[0].shape[0] for s in all_sessions_neural if len(s) > 0)
    total_trials = sum(len(s) for s in all_sessions_neural)
    neurons_per_session = [s[0].shape[0] for s in all_sessions_neural if len(s) > 0]
    trials_per_session = [len(s) for s in all_sessions_neural]

    print(f"\n{'='*60}")
    print(f"Conversion Summary")
    print(f"{'='*60}")
    print(f"Sessions: {len(all_sessions_neural)}")
    print(f"Subjects: {len(subjects_list)}")
    print(f"Brain regions: {brain_regions_list}")
    print(f"Total trials: {total_trials}")
    print(f"Trials/session: mean={np.mean(trials_per_session):.1f}, "
          f"min={np.min(trials_per_session)}, max={np.max(trials_per_session)}")
    print(f"Neurons/session: mean={np.mean(neurons_per_session):.1f}, "
          f"min={np.min(neurons_per_session)}, max={np.max(neurons_per_session)}")
    print(f"Total elapsed time: {time.time()-total_start:.1f}s")

    # Output distribution
    all_outcomes = []
    all_images = []
    all_changes = []
    for session_out in all_sessions_output:
        for trial_out in session_out:
            all_outcomes.append(trial_out[4, 0])  # trial outcome (static)
            all_images.extend(trial_out[0, :].tolist())  # image identity
            all_changes.extend(trial_out[1, :].tolist())  # image change

    all_outcomes = np.array(all_outcomes)
    print(f"\nTrial outcome distribution:")
    for i, name in enumerate(outcome_value_names):
        count = (all_outcomes == i).sum()
        print(f"  {name}: {count} ({100*count/len(all_outcomes):.1f}%)")

    all_changes = np.array(all_changes)
    print(f"\nImage change distribution:")
    print(f"  no_change: {(all_changes==0).sum()} ({100*(all_changes==0).mean():.1f}%)")
    print(f"  change: {(all_changes==1).sum()} ({100*(all_changes==1).mean():.1f}%)")

    print(f"\nImage identity distribution:")
    all_images = np.array(all_images)
    for i, name in enumerate(IMAGE_NAMES_GLOBAL):
        count = (all_images == i).sum()
        print(f"  {name}: {count} ({100*count/len(all_images):.1f}%)")


if __name__ == '__main__':
    main()
