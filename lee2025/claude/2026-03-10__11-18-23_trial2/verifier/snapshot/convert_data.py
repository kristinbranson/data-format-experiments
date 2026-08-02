#!/usr/bin/env python3
"""
Convert QLAK-CA1 geometric representations dataset to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]

Reference: Lee et al. (2025) Neuron 113(2): 307-320
"""

import os
import sys
import time
import argparse
import pickle
import numpy as np
import joblib
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add reference code to path for get_env_mat
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'georepca1', 'src'))

# ============================================================================
# Constants matching reference code
# ============================================================================
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
FPS = 30  # recording frame rate (Hz)
TEMPORAL_BIN_SIZE = 3  # frames per time bin (from reference fit_decoder)
TRIAL_DURATION_SEC = 60  # 1 minute trials
FRAMES_PER_TRIAL = TRIAL_DURATION_SEC * FPS  # 1800 frames per trial
BINS_PER_TRIAL = FRAMES_PER_TRIAL // TEMPORAL_BIN_SIZE  # 600 time bins per trial
ARENA_SIZE_CM = 75.0  # arena is 75x75 cm
N_SPATIAL_BINS = 3  # 3x3 output grid
SPATIAL_BIN_SIZE = ARENA_SIZE_CM / N_SPATIAL_BINS  # 25 cm per bin
TIME_BIN_MS = (TEMPORAL_BIN_SIZE / FPS) * 1000  # 100 ms

# ============================================================================
# Environment geometry (from reference code get_env_mat)
# ============================================================================
ENV_MATRICES = {
    'square':    np.array([[1,1,1],[1,1,1],[1,1,1]]),
    'o':         np.array([[1,1,1],[1,0,1],[1,1,1]]),
    't':         np.array([[0,1,0],[0,1,0],[1,1,1]]),
    'u':         np.array([[1,1,1],[1,0,0],[1,1,1]]),
    'rectangle': np.array([[0,1,1],[0,1,1],[0,1,1]]),
    '+':         np.array([[0,1,0],[1,1,1],[0,1,0]]),
    'i':         np.array([[1,1,1],[0,1,0],[1,1,1]]),
    'l':         np.array([[1,1,1],[1,0,0],[1,0,0]]),
    'bit donut': np.array([[1,1,1],[1,0,1],[0,1,1]]),
    'glenn':     np.array([[1,1,0],[1,1,1],[0,1,1]]),
}


def get_env_input(env_name):
    """Get flattened 3x3 environment matrix as decoder input (9 values)."""
    mat = ENV_MATRICES.get(env_name)
    if mat is None:
        raise ValueError(f"Unknown environment: {env_name}")
    return mat.flatten().astype(np.float32)


def temporal_bin_trace(trace_2d, sigma=TEMPORAL_BIN_SIZE, bin_size=TEMPORAL_BIN_SIZE):
    """
    Temporally bin trace data matching reference fit_decoder:
    1. Gaussian smooth along time axis (sigma=3 frames)
    2. Average pool with kernel=3, stride=3

    Args:
        trace_2d: (n_cells, n_frames) binary trace data
    Returns:
        (n_cells, n_bins) smoothed and binned trace
    """
    # Gaussian smooth along time axis (axis=1 for cells x time)
    smoothed = gaussian_filter1d(trace_2d.astype(np.float64), sigma=sigma, axis=1)
    # Average pool: reshape and mean
    n_cells, n_frames = smoothed.shape
    n_bins = n_frames // bin_size
    trimmed = smoothed[:, :n_bins * bin_size]
    binned = trimmed.reshape(n_cells, n_bins, bin_size).mean(axis=2)
    return binned.astype(np.float32)


def discretize_position(position_2d, bin_size=SPATIAL_BIN_SIZE, n_bins=N_SPATIAL_BINS):
    """
    Discretize x,y position into 3x3 grid bins.

    Args:
        position_2d: (2, n_frames) x,y position in cm
    Returns:
        (n_frames,) integer bin indices 0-8
    """
    # Bin: floor(pos / bin_size), clamp to [0, n_bins-1]
    x_bin = np.clip(np.floor(position_2d[0] / bin_size).astype(int), 0, n_bins - 1)
    y_bin = np.clip(np.floor(position_2d[1] / bin_size).astype(int), 0, n_bins - 1)
    return x_bin * n_bins + y_bin


def bin_position(position_2d, bin_size=TEMPORAL_BIN_SIZE):
    """
    Temporally bin position by taking the mean of each bin (matching reference behavior binning).

    Args:
        position_2d: (2, n_frames) position data
    Returns:
        (2, n_bins) binned position
    """
    n_frames = position_2d.shape[1]
    n_bins = n_frames // bin_size
    trimmed = position_2d[:, :n_bins * bin_size]
    binned = trimmed.reshape(2, n_bins, bin_size).mean(axis=2)
    return binned


def process_session(animal_data, day_idx, show_processing=False, animal_name="", fig_axes=None):
    """
    Process one session (day) into trials.

    Returns:
        neural_trials: list of (n_neurons, n_timepoints) arrays
        input_trials: list of (9,) arrays (static environment)
        output_trials: list of (1, n_timepoints) arrays (position bin)
        n_valid_cells: number of valid cells this session
    """
    trace = animal_data['trace'][day_idx]  # (n_cells, n_frames)
    position = animal_data['position'][day_idx]  # (2, n_frames)
    env_name = str(animal_data['envs'][day_idx].item() if hasattr(animal_data['envs'][day_idx], 'item')
                   else animal_data['envs'][day_idx])
    if isinstance(env_name, np.ndarray):
        env_name = str(env_name.squeeze())

    n_cells, n_frames = trace.shape

    # Identify valid (registered) cells for this day
    valid_mask = ~np.isnan(trace[:, 0])
    valid_trace = trace[valid_mask]  # (n_valid, n_frames)
    n_valid = valid_mask.sum()

    if n_valid == 0:
        return [], [], [], 0

    # Temporal bin the trace (gaussian smooth + average pool)
    binned_trace = temporal_bin_trace(valid_trace)  # (n_valid, n_total_bins)

    # Temporal bin position and discretize
    binned_pos = bin_position(position)  # (2, n_total_bins)
    pos_bins = discretize_position(binned_pos)  # (n_total_bins,)

    n_total_bins = binned_trace.shape[1]

    # Get environment input (static per trial)
    env_input = get_env_input(env_name)  # (9,)

    # Split into 1-minute trials
    neural_trials = []
    input_trials = []
    output_trials = []

    n_trials = n_total_bins // BINS_PER_TRIAL
    for t in range(n_trials):
        start = t * BINS_PER_TRIAL
        end = (t + 1) * BINS_PER_TRIAL

        trial_neural = binned_trace[:, start:end]  # (n_valid, 600)
        trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)  # (1, 600)

        neural_trials.append(trial_neural)
        input_trials.append(env_input)  # (9,) static
        output_trials.append(trial_output)

    # Visualization for --show-processing
    if show_processing and fig_axes is not None and n_trials > 0:
        ax_neural, ax_pos, ax_output, ax_env = fig_axes

        # Show first trial
        trial_idx = 0
        t_start = trial_idx * BINS_PER_TRIAL
        t_end = (trial_idx + 1) * BINS_PER_TRIAL
        time_axis = np.arange(BINS_PER_TRIAL) * (TIME_BIN_MS / 1000)  # seconds

        # Neural activity (first 20 cells)
        n_show = min(20, n_valid)
        ax_neural.imshow(binned_trace[:n_show, t_start:t_end], aspect='auto',
                        extent=[0, TRIAL_DURATION_SEC, n_show, 0], cmap='hot')
        ax_neural.set_ylabel('Neuron')
        ax_neural.set_title(f'{animal_name} Day {day_idx} - {env_name}\nNeural (first {n_show} cells)')

        # Position trace
        ax_pos.plot(time_axis, binned_pos[0, t_start:t_end], label='x', alpha=0.7)
        ax_pos.plot(time_axis, binned_pos[1, t_start:t_end], label='y', alpha=0.7)
        for b in range(1, N_SPATIAL_BINS):
            ax_pos.axhline(b * SPATIAL_BIN_SIZE, color='gray', ls='--', alpha=0.5)
        ax_pos.set_ylabel('Position (cm)')
        ax_pos.legend(loc='upper right')
        ax_pos.set_title('Position trace with bin boundaries')

        # Discretized output
        ax_output.plot(time_axis, pos_bins[t_start:t_end], 'k-', alpha=0.7)
        ax_output.set_ylabel('Position bin (0-8)')
        ax_output.set_xlabel('Time (s)')
        ax_output.set_yticks(range(9))
        ax_output.set_title('Discretized position (3x3)')

        # Environment geometry
        env_mat = ENV_MATRICES[env_name]
        ax_env.imshow(env_mat, cmap='Blues', vmin=0, vmax=1)
        ax_env.set_title(f'Environment: {env_name}')
        for i in range(3):
            for j in range(3):
                ax_env.text(j, i, str(i*3+j), ha='center', va='center',
                           color='red' if env_mat[i,j]==0 else 'black')
        ax_env.set_xticks([])
        ax_env.set_yticks([])

    return neural_trials, input_trials, output_trials, n_valid


def convert_dataset(animals_to_process, data_dir='data', show_processing=False):
    """
    Convert the full dataset to decoder format.

    Args:
        animals_to_process: list of animal names
        data_dir: path to data directory
        show_processing: if True, generate processing plots
    Returns:
        data dict in target format
    """
    all_neural = []
    all_input = []
    all_output = []
    subject_idx_list = []
    brain_region_idx_list = []

    subjects = list(animals_to_process)
    brain_regions = ['CA1']

    total_sessions = 0
    total_trials = 0
    total_valid_cell_days = 0

    for a_idx, animal in enumerate(animals_to_process):
        t0 = time.time()
        print(f"\nLoading {animal}...")
        dat = joblib.load(os.path.join(data_dir, animal))
        animal_data = dat[animal]
        n_days = animal_data['trace'].shape[0]
        print(f"  {n_days} days, {animal_data['trace'].shape[1]} total cells, "
              f"{animal_data['trace'].shape[2]} frames/day")

        for day in range(n_days):
            t1 = time.time()

            # Determine if we should show processing for this session
            do_plot = show_processing and total_sessions < 2
            fig = None
            fig_axes = None
            if do_plot:
                fig, axes = plt.subplots(2, 2, figsize=(16, 10))
                fig_axes = (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1])

            env_name = str(animal_data['envs'][day].squeeze())
            neural_trials, input_trials, output_trials, n_valid = process_session(
                animal_data, day,
                show_processing=do_plot,
                animal_name=animal,
                fig_axes=fig_axes
            )

            if len(neural_trials) < 2:
                print(f"  Day {day}: skipped (< 2 trials)")
                if fig is not None:
                    plt.close(fig)
                continue

            all_neural.append(neural_trials)
            all_input.append(input_trials)
            all_output.append(output_trials)
            subject_idx_list.append(a_idx)
            brain_region_idx_list.append(np.zeros(n_valid, dtype=np.int64))

            total_sessions += 1
            total_trials += len(neural_trials)
            total_valid_cell_days += n_valid

            elapsed = time.time() - t1
            if day == 0 or (day + 1) % 10 == 0 or day == n_days - 1:
                print(f"  Day {day}: {env_name}, {n_valid} cells, "
                      f"{len(neural_trials)} trials, {elapsed:.2f}s")

            if do_plot and fig is not None:
                fig.suptitle(f'{animal} Day {day} ({env_name}) - Processing Overview', fontsize=14)
                fig.tight_layout()
                fig.savefig(f'processing_{animal}_day{day}.png', dpi=100)
                plt.close(fig)

        animal_time = time.time() - t0
        print(f"  {animal} done in {animal_time:.1f}s")
        del dat

    # Output position bin names
    output_values_position = []
    for i in range(N_SPATIAL_BINS):
        for j in range(N_SPATIAL_BINS):
            output_values_position.append(f"row{i}_col{j}")

    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': subjects,
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx_list,
        'input_names': [f'env_partition_{i}' for i in range(9)],
        'output_names': ['position'],
        'output_values': [output_values_position],
        'metadata': {
            'task_description': 'Decode mouse position (3x3 spatial bins) from CA1 neural activity during geometric environment exploration',
            'time_bin_size': TIME_BIN_MS,
            'temporal_alignment_event': 'Start of recording session',
            'off_start': 0.0,
            'off_end': float(TRIAL_DURATION_SEC),
            'recording_fps': FPS,
            'temporal_bin_frames': TEMPORAL_BIN_SIZE,
            'trial_duration_sec': TRIAL_DURATION_SEC,
            'arena_size_cm': ARENA_SIZE_CM,
            'spatial_bin_size_cm': SPATIAL_BIN_SIZE,
            'n_spatial_bins': N_SPATIAL_BINS,
            'total_sessions': total_sessions,
            'total_trials': total_trials,
            'total_valid_cell_days': total_valid_cell_days,
            'reference': 'Lee et al. (2025) Neuron 113(2): 307-320',
        }
    }

    print(f"\n=== Conversion Summary ===")
    print(f"Subjects: {len(subjects)}")
    print(f"Sessions: {total_sessions}")
    print(f"Total trials: {total_trials}")
    print(f"Total valid cell-days: {total_valid_cell_days}")
    print(f"Time bin: {TIME_BIN_MS} ms")
    print(f"Bins per trial: {BINS_PER_TRIAL}")

    return data


def main():
    parser = argparse.ArgumentParser(description='Convert QLAK-CA1 data to decoder format')
    parser.add_argument('output', type=str, help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    args = parser.parse_args()

    if args.sample:
        # Use first 2 animals with limited days for sample
        animals_to_process = ANIMALS[:2]
        print(f"SAMPLE MODE: Processing {animals_to_process}")
    else:
        animals_to_process = ANIMALS
        print(f"FULL MODE: Processing all {len(ANIMALS)} animals")

    t_start = time.time()
    data = convert_dataset(animals_to_process, data_dir='data',
                          show_processing=args.show_processing)

    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    file_size = os.path.getsize(args.output) / (1024 * 1024)
    total_time = time.time() - t_start
    print(f"Saved {args.output} ({file_size:.1f} MB)")
    print(f"Total conversion time: {total_time:.1f}s")


if __name__ == '__main__':
    main()
