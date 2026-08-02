#!/usr/bin/env python3
"""Convert CA1 neural recording data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]
"""

import argparse
import os
import sys
import time
import pickle
import numpy as np
import joblib
from collections import OrderedDict

# ============================================================================
# Constants
# ============================================================================
DATA_DIR = 'data'
ALL_ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
               "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
FPS = 30  # Recording frame rate
TIME_BIN_SEC = 1.0  # Time bin size in seconds
TIME_BIN_FRAMES = int(FPS * TIME_BIN_SEC)  # 30 frames per time bin
TRIAL_DURATION_SEC = 60  # Trial duration in seconds
TRIAL_DURATION_BINS = int(TRIAL_DURATION_SEC / TIME_BIN_SEC)  # 60 time bins per trial
N_SPATIAL_BINS = 3  # 3x3 grid
N_OUTPUT_CLASSES = N_SPATIAL_BINS ** 2  # 9
BRAIN_REGION = 'CA1'

# ============================================================================
# Environment geometry functions
# ============================================================================
def get_env_mat(env):
    """Get binary 3x3 matrix for environment geometry.
    Copied from reference code utils.py:215."""
    env_mats = {
        'square': [[1,1,1],[1,1,1],[1,1,1]],
        'o': [[1,1,1],[1,0,1],[1,1,1]],
        't': [[0,1,0],[0,1,0],[1,1,1]],
        'u': [[1,1,1],[1,0,0],[1,1,1]],
        'rectangle': [[0,1,1],[0,1,1],[0,1,1]],
        '+': [[0,1,0],[1,1,1],[0,1,0]],
        'i': [[1,1,1],[0,1,0],[1,1,1]],
        'l': [[1,1,1],[1,0,0],[1,0,0]],
        'bit donut': [[1,1,1],[1,0,1],[0,1,1]],
        'glenn': [[1,1,0],[1,1,1],[0,1,1]],
    }
    if env in env_mats:
        return np.array(env_mats[env], dtype=float)
    else:
        raise ValueError(f"Unknown environment: {env}")


# ============================================================================
# Data processing functions
# ============================================================================
def bin_trace_temporal(trace, time_bin_frames):
    """Bin binary trace data into time bins by averaging.
    
    Args:
        trace: (n_cells, n_timepoints) binary trace data
        time_bin_frames: number of frames per time bin
    
    Returns:
        (n_cells, n_timebins) averaged firing rates
    """
    n_cells, n_timepoints = trace.shape
    n_bins = n_timepoints // time_bin_frames
    # Truncate to exact multiple of time_bin_frames
    trace_truncated = trace[:, :n_bins * time_bin_frames]
    # Reshape and average
    trace_binned = trace_truncated.reshape(n_cells, n_bins, time_bin_frames).mean(axis=2)
    return trace_binned.astype(np.float32)


def bin_position_to_grid(position, n_spatial_bins=3):
    """Bin continuous position into spatial grid.
    
    Args:
        position: (2, n_timepoints) x,y position data
        n_spatial_bins: number of bins per dimension
    
    Returns:
        (n_timepoints,) integer bin indices (0 to n_spatial_bins^2 - 1)
    """
    buffer = 1e-5
    pos_max = np.nanmax(position) + buffer
    bin_size = pos_max / n_spatial_bins
    pos_binned = np.floor(position / bin_size).astype(int)
    # Clip to valid range
    pos_binned = np.clip(pos_binned, 0, n_spatial_bins - 1)
    # Convert to single index: x * n_bins + y
    bin_idx = pos_binned[0] * n_spatial_bins + pos_binned[1]
    return bin_idx


def bin_position_temporal(position, time_bin_frames, n_spatial_bins=3):
    """Bin position into time bins and spatial grid.
    
    For each time bin, take the mode (most frequent) spatial bin.
    
    Args:
        position: (2, n_timepoints) x,y position
        time_bin_frames: frames per time bin
        n_spatial_bins: spatial bins per dimension
    
    Returns:
        (n_timebins,) integer bin indices
    """
    # First get spatial bin for each frame
    bin_idx = bin_position_to_grid(position, n_spatial_bins)
    
    n_timepoints = len(bin_idx)
    n_bins = n_timepoints // time_bin_frames
    bin_idx_truncated = bin_idx[:n_bins * time_bin_frames]
    bin_idx_reshaped = bin_idx_truncated.reshape(n_bins, time_bin_frames)
    
    # Mode for each time bin (vectorized)
    result = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        values, counts = np.unique(bin_idx_reshaped[i], return_counts=True)
        result[i] = values[np.argmax(counts)]
    
    return result


def split_into_trials(data, trial_length):
    """Split a time series into fixed-length trials.
    
    Args:
        data: array with time as last dimension
        trial_length: number of time bins per trial
    
    Returns:
        list of arrays, each with trial_length time bins
    """
    if data.ndim == 1:
        n_timebins = len(data)
        n_trials = n_timebins // trial_length
        trials = []
        for t in range(n_trials):
            start = t * trial_length
            end = start + trial_length
            trials.append(data[start:end])
        return trials
    else:
        # data shape: (n_features, n_timebins)
        n_timebins = data.shape[-1]
        n_trials = n_timebins // trial_length
        trials = []
        for t in range(n_trials):
            start = t * trial_length
            end = start + trial_length
            trials.append(data[..., start:end])
        return trials


# ============================================================================
# Main conversion function
# ============================================================================
def convert_data(animals, data_dir, show_processing=False):
    """Convert CA1 data to decoder format.
    
    Args:
        animals: list of animal IDs to process
        data_dir: path to data directory
        show_processing: whether to generate processing plots
    
    Returns:
        data dict in decoder format
    """
    t_start = time.time()
    
    # Initialize output structure
    neural_sessions = []
    input_sessions = []
    output_sessions = []
    subjects_list = []
    subject_idx_list = []
    brain_region_idx_sessions = []
    
    # Get unique subjects
    all_subjects = sorted(set(animals))
    
    total_sessions = 0
    total_trials = 0
    total_neurons_sum = 0
    
    for animal in animals:
        t_animal_start = time.time()
        print(f"\nLoading {animal}...")
        dat = joblib.load(os.path.join(data_dir, animal))
        d = dat[animal]
        
        n_days = d['trace'].shape[0]
        n_cells_total = d['trace'].shape[1]
        n_timepoints = d['trace'].shape[2]
        
        subject_id = animal
        if subject_id not in all_subjects:
            all_subjects.append(subject_id)
        subj_idx = all_subjects.index(subject_id)
        
        print(f"  {n_days} days, {n_cells_total} total cells, {n_timepoints} timepoints")
        
        for day in range(n_days):
            t_day_start = time.time()
            
            # Get environment name
            env_name = str(d['envs'][day, 0])
            
            # Get trace data for this day: (n_cells, n_timepoints)
            trace_day = d['trace'][day]  # (n_cells, n_timepoints)
            
            # Find valid (registered) cells
            valid_mask = ~np.isnan(trace_day[:, 0])
            valid_trace = trace_day[valid_mask]  # (n_valid_cells, n_timepoints)
            n_valid = valid_mask.sum()
            
            # Get position data: (2, n_timepoints)
            pos_day = d['position'][day]  # (2, n_timepoints)
            
            # Temporal binning of neural data (1-second bins)
            neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
            # shape: (n_valid_cells, n_timebins)
            
            # Bin position into 3x3 grid and temporal bins
            pos_binned = bin_position_temporal(pos_day, TIME_BIN_FRAMES, N_SPATIAL_BINS)
            # shape: (n_timebins,)
            
            # Split into 1-minute trials
            neural_trials = split_into_trials(neural_binned, TRIAL_DURATION_BINS)
            output_trials_raw = split_into_trials(pos_binned, TRIAL_DURATION_BINS)
            
            n_trials = len(neural_trials)
            
            if n_trials < 2:
                print(f"  WARNING: Day {day} ({env_name}) has only {n_trials} trials, skipping")
                continue
            
            # Environment geometry input (static per trial)
            env_mat = get_env_mat(env_name).flatten()  # (9,)
            
            # Build trial lists for this session
            session_neural = []
            session_input = []
            session_output = []
            
            for trial_idx in range(n_trials):
                # Neural: (n_valid_cells, TRIAL_DURATION_BINS)
                session_neural.append(neural_trials[trial_idx])
                
                # Input: environment geometry, static per trial (9,)
                session_input.append(env_mat.astype(np.float32))
                
                # Output: position bin index, time-varying (1, TRIAL_DURATION_BINS)
                session_output.append(output_trials_raw[trial_idx].reshape(1, -1).astype(int))
            
            neural_sessions.append(session_neural)
            input_sessions.append(session_input)
            output_sessions.append(session_output)
            subject_idx_list.append(subj_idx)
            brain_region_idx_sessions.append(np.zeros(n_valid, dtype=int))  # All CA1
            
            total_sessions += 1
            total_trials += n_trials
            total_neurons_sum += n_valid
            
            t_day_end = time.time()
            if day % 10 == 0 or day == n_days - 1:
                print(f"  Day {day}/{n_days-1} ({env_name}): {n_valid} cells, {n_trials} trials, "
                      f"{t_day_end - t_day_start:.1f}s")
        
        # Generate processing plots if requested
        if show_processing:
            plot_processing(d, animal, data_dir)
        
        del dat
        t_animal_end = time.time()
        print(f"  {animal} done in {t_animal_end - t_animal_start:.1f}s")
    
    # Build output names and values
    output_names = ['position']
    output_values = [[f'bin_{i}' for i in range(N_OUTPUT_CLASSES)]]
    
    input_names = [f'env_grid_{i}' for i in range(N_OUTPUT_CLASSES)]
    
    # Build final data dict
    data = {
        'neural': neural_sessions,
        'input': input_sessions,
        'output': output_sessions,
        'subjects': all_subjects,
        'subject_idx': np.array(subject_idx_list),
        'brain_regions': [BRAIN_REGION],
        'brain_region_idx': brain_region_idx_sessions,
        'input_names': input_names,
        'output_names': output_names,
        'output_values': output_values,
        'metadata': {
            'task_description': 'Decode mouse position (3x3 grid) from CA1 neural activity during geometric environment exploration',
            'time_bin_size': TIME_BIN_SEC * 1000,  # in ms
            'temporal_alignment_event': 'Start of recording session',
            'off_start': 0.0,
            'off_end': TRIAL_DURATION_SEC,
            'fps': FPS,
            'arena_size_cm': 75,
            'n_spatial_bins': N_SPATIAL_BINS,
            'trial_duration_sec': TRIAL_DURATION_SEC,
            'source_paper': 'Lee, Keinath, Cianfarano & Brandon (2025). Neuron 113(2): 307-320.',
            'total_unique_neurons': 5413,
            'total_sessions': total_sessions,
            'total_trials': total_trials,
        }
    }
    
    t_end = time.time()
    print(f"\n=== Conversion Summary ===")
    print(f"Total sessions: {total_sessions}")
    print(f"Total trials: {total_trials}")
    print(f"Mean trials/session: {total_trials/total_sessions:.1f}")
    print(f"Subjects: {all_subjects}")
    print(f"Total time: {t_end - t_start:.1f}s")
    
    return data


def plot_processing(d, animal, data_dir):
    """Generate processing visualization plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(4, 3, figsize=(20, 16))
    fig.suptitle(f'Processing Visualization: {animal}', fontsize=16)
    
    # Plot first 3 days
    for col, day in enumerate(range(min(3, d['trace'].shape[0]))):
        env_name = str(d['envs'][day, 0])
        
        # Row 0: Raw position trajectory
        ax = axes[0, col]
        pos = d['position'][day]
        ax.plot(pos[0, :3000], pos[1, :3000], 'b-', alpha=0.3, linewidth=0.5)
        ax.set_title(f'Day {day}: {env_name} - Position')
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_xlim(0, 75)
        ax.set_ylim(0, 75)
        ax.set_aspect('equal')
        
        # Row 1: Environment geometry
        ax = axes[1, col]
        env_mat = get_env_mat(env_name)
        ax.imshow(env_mat, cmap='RdYlGn', vmin=0, vmax=1, origin='upper')
        ax.set_title(f'Env geometry: {env_name}')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{int(env_mat[i,j])}', ha='center', va='center', fontsize=14)
        
        # Row 2: Neural activity (first 5 valid cells, first 1000 time bins)
        ax = axes[2, col]
        trace_day = d['trace'][day]
        valid_mask = ~np.isnan(trace_day[:, 0])
        valid_trace = trace_day[valid_mask]
        neural_binned = bin_trace_temporal(valid_trace, TIME_BIN_FRAMES)
        n_show = min(5, neural_binned.shape[0])
        t_show = min(300, neural_binned.shape[1])
        for i in range(n_show):
            ax.plot(np.arange(t_show), neural_binned[i, :t_show] + i * 0.5, linewidth=0.5)
        ax.set_title(f'Neural (binned, {valid_mask.sum()} cells)')
        ax.set_xlabel('Time bin (1s)')
        
        # Row 3: Position bins over time
        ax = axes[3, col]
        pos_binned = bin_position_temporal(d['position'][day], TIME_BIN_FRAMES, N_SPATIAL_BINS)
        ax.plot(pos_binned[:300], 'k-', linewidth=0.5)
        ax.set_title('Position bin (3x3 grid)')
        ax.set_xlabel('Time bin (1s)')
        ax.set_ylabel('Bin index (0-8)')
        ax.set_ylim(-0.5, 8.5)
    
    plt.tight_layout()
    plt.savefig(f'processing_{animal}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved processing_{animal}.png")


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert CA1 data to decoder format')
    parser.add_argument('output', help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True,
                        help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true',
                        help='Process only 2 sessions for testing')
    parser.add_argument('--show-processing', action='store_true',
                        help='Generate processing visualization plots')
    
    args = parser.parse_args()
    
    if args.sample:
        # Use first 2 animals for sample
        animals = ALL_ANIMALS[:2]
        print(f"SAMPLE MODE: Processing {len(animals)} animals: {animals}")
    else:
        animals = ALL_ANIMALS
        print(f"FULL MODE: Processing {len(animals)} animals")
    
    data = convert_data(animals, DATA_DIR, show_processing=args.show_processing)
    
    print(f"\nSaving to {args.output}...")
    t_save = time.time()
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f"Saved in {time.time() - t_save:.1f}s")
    print(f"File size: {os.path.getsize(args.output) / 1e6:.1f} MB")
    print("Done!")
