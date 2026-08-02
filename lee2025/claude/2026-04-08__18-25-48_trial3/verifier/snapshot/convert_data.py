#!/usr/bin/env python3
"""
Convert CA1 calcium imaging data from Lee et al. (2025) to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--sample] [--show-processing]

Options:
    --full (default): Process all sessions
    --sample: Process only 2 sessions for testing
    --show-processing: Plot visualizations of every processing step for up to 2 sessions
"""

import argparse
import os
import sys
import time
import pickle
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ============================================================================
# Environment geometry functions (from reference code)
# ============================================================================

def get_env_mat(env):
    """
    Get binary 3x3 matrix for the corresponding string environment name.
    1 = accessible partition, 0 = blocked partition.
    From reference code: georepca1/src/utils.py
    """
    env_mats = {
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
    return env_mats.get(env, np.full((3,3), np.nan)).astype(float)


def discretize_position_3x3(position, env_size=75.0):
    """
    Discretize x,y position into 3x3 grid (9 bins).

    Args:
        position: shape (2, n_frames), x-y coordinates in [0, env_size]
        env_size: size of the environment in cm

    Returns:
        bin_ids: shape (n_frames,), integer bin IDs 0-8
    """
    bin_size = env_size / 3.0
    # Clip to valid range and compute bin indices
    x = np.clip(position[0], 0, env_size - 1e-10)
    y = np.clip(position[1], 0, env_size - 1e-10)
    x_bin = np.floor(x / bin_size).astype(int)
    y_bin = np.floor(y / bin_size).astype(int)
    # Clamp to [0, 2]
    x_bin = np.clip(x_bin, 0, 2)
    y_bin = np.clip(y_bin, 0, 2)
    # Combine into single bin ID: row-major order
    bin_ids = x_bin * 3 + y_bin
    return bin_ids


def process_animal(animal_name, data_dir, trial_duration_frames=1800, show_processing=False):
    """
    Process one animal's data into sessions of trials.

    Args:
        animal_name: e.g. 'QLAK-CA1-08'
        data_dir: path to data directory
        trial_duration_frames: frames per trial (1800 = 1 min at 30 Hz)
        show_processing: whether to generate processing plots

    Returns:
        dict with 'neural', 'input', 'output', 'envs', 'n_active_cells' per session
    """
    t0 = time.time()
    print(f"Loading {animal_name}...", flush=True)
    dat = joblib.load(os.path.join(data_dir, animal_name))
    d = dat[animal_name]
    t_load = time.time() - t0
    print(f"  Loaded in {t_load:.1f}s", flush=True)

    trace = d['trace']       # (n_days, n_cells, n_frames)
    position = d['position'] # (n_days, 2, n_frames)
    envs = d['envs'].flatten()  # (n_days,) string array

    n_days, n_cells_total, n_frames_total = trace.shape

    sessions_neural = []
    sessions_input = []
    sessions_output = []
    sessions_env = []
    active_cells_per_session = []

    for day in range(n_days):
        t_day = time.time()

        # Get trace and position for this day
        trace_day = trace[day]       # (n_cells, n_frames)
        pos_day = position[day]      # (2, n_frames)
        env_name = str(envs[day])

        # Identify active (registered) cells for this day
        # A cell is registered if its trace is not all NaN
        active_mask = ~np.all(np.isnan(trace_day), axis=1)
        active_trace = trace_day[active_mask]  # (n_active, n_frames)
        n_active = active_mask.sum()

        # Replace any remaining NaN with 0 (shouldn't happen for active cells, but safety)
        active_trace = np.nan_to_num(active_trace, nan=0.0)

        # Get environment geometry as input (flattened 3x3 binary matrix)
        env_mat = get_env_mat(env_name).flatten()  # (9,)

        # Discretize position into 3x3 grid
        bin_ids = discretize_position_3x3(pos_day)  # (n_frames,)

        # Split into 1-minute trials
        n_trials = n_frames_total // trial_duration_frames

        trials_neural = []
        trials_input = []
        trials_output = []

        for trial_idx in range(n_trials):
            start = trial_idx * trial_duration_frames
            end = start + trial_duration_frames

            # Neural: (n_active, trial_duration_frames)
            trial_neural = active_trace[:, start:end].astype(np.float32)

            # Input: environment geometry, static per trial (9,)
            trial_input = env_mat.astype(np.float32)

            # Output: position bin, time-varying (1, trial_duration_frames)
            # Must be integer-valued for indexing into output_values
            trial_output = bin_ids[start:end].reshape(1, -1).astype(np.int64)

            trials_neural.append(trial_neural)
            trials_input.append(trial_input)
            trials_output.append(trial_output)

        sessions_neural.append(trials_neural)
        sessions_input.append(trials_input)
        sessions_output.append(trials_output)
        sessions_env.append(env_name)
        active_cells_per_session.append(n_active)

        dt = time.time() - t_day
        if day < 3 or day == n_days - 1:
            print(f"  Day {day}/{n_days}: {env_name}, {n_active} active cells, "
                  f"{n_trials} trials, {dt:.2f}s", flush=True)

    # Generate processing plots if requested
    if show_processing:
        plot_processing(animal_name, d, sessions_neural, sessions_input,
                       sessions_output, sessions_env, active_cells_per_session,
                       trial_duration_frames)

    t_total = time.time() - t0
    print(f"  {animal_name} complete: {n_days} sessions, "
          f"{sum(len(s) for s in sessions_neural)} total trials, {t_total:.1f}s", flush=True)

    del dat, d  # Free memory

    return {
        'neural': sessions_neural,
        'input': sessions_input,
        'output': sessions_output,
        'envs': sessions_env,
        'active_cells': active_cells_per_session,
    }


def plot_processing(animal_name, raw_data, sessions_neural, sessions_input,
                   sessions_output, sessions_env, active_cells, trial_duration):
    """Plot processing visualizations for an animal."""

    n_days = len(sessions_neural)
    n_plot_days = min(2, n_days)

    fig, axes = plt.subplots(n_plot_days, 4, figsize=(24, 6*n_plot_days))
    if n_plot_days == 1:
        axes = axes[np.newaxis, :]

    for d_idx in range(n_plot_days):
        day = d_idx  # Plot first two days
        env = sessions_env[day]

        # 1. Raw position trace
        ax = axes[d_idx, 0]
        pos = raw_data['position'][day]  # (2, n_frames)
        t_sec = np.arange(pos.shape[1]) / 30.0
        ax.plot(t_sec[:3000], pos[0, :3000], alpha=0.7, label='x')
        ax.plot(t_sec[:3000], pos[1, :3000], alpha=0.7, label='y')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (cm)')
        ax.set_title(f'Day {day}: {env}\nRaw position (first 100s)')
        ax.legend()
        ax.axhline(25, color='gray', ls='--', alpha=0.3)
        ax.axhline(50, color='gray', ls='--', alpha=0.3)

        # 2. Discretized position
        ax = axes[d_idx, 1]
        trial0_output = sessions_output[day][0]  # (1, 1800)
        t_trial = np.arange(trial0_output.shape[1]) / 30.0
        ax.plot(t_trial, trial0_output[0], '.', markersize=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position bin (0-8)')
        ax.set_title(f'Discretized output (trial 0)')
        ax.set_yticks(range(9))

        # 3. Environment geometry input
        ax = axes[d_idx, 2]
        env_mat = sessions_input[day][0].reshape(3, 3)
        ax.imshow(env_mat, cmap='gray_r', vmin=0, vmax=1)
        ax.set_title(f'Env geometry input: {env}')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{int(env_mat[i,j])}', ha='center', va='center',
                       color='red' if env_mat[i,j] == 0 else 'white', fontsize=14)

        # 4. Neural raster (first 20 cells, first trial)
        ax = axes[d_idx, 3]
        trial0_neural = sessions_neural[day][0]  # (n_active, 1800)
        n_show = min(20, trial0_neural.shape[0])
        for i in range(n_show):
            spikes = np.where(trial0_neural[i] > 0)[0]
            ax.scatter(spikes / 30.0, np.full_like(spikes, i), s=0.5, c='black')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Cell #')
        ax.set_title(f'Neural raster (first {n_show} cells, trial 0)')
        ax.set_ylim(-0.5, n_show - 0.5)

    fig.suptitle(f'Processing: {animal_name}', fontsize=14)
    fig.tight_layout()
    fig.savefig(f'processing_{animal_name}.png', dpi=150)
    plt.close(fig)
    print(f"  Saved processing_{animal_name}.png")


def main():
    parser = argparse.ArgumentParser(description='Convert CA1 data to decoder format.')
    parser.add_argument('outfile', type=str, help='Output pickle file path')
    parser.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    parser.add_argument('--full', action='store_true', help='Process all sessions (default)')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    args = parser.parse_args()

    data_dir = 'data'
    animals = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
               'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']

    fps = 30  # recording frame rate
    trial_duration_sec = 60  # 1 minute trials
    trial_duration_frames = fps * trial_duration_sec  # 1800 frames

    # Determine which animals/sessions to process
    if args.sample:
        # Process first 2 sessions (days) from first animal only
        animals_to_process = [animals[0]]
        max_sessions = 2
        print(f"SAMPLE mode: processing {max_sessions} sessions from {animals_to_process[0]}")
    else:
        animals_to_process = animals
        max_sessions = None
        print(f"FULL mode: processing all {len(animals)} animals")

    # Process all animals
    all_neural = []
    all_input = []
    all_output = []
    subjects = animals  # All 7 animals are subjects
    subject_idx_list = []
    brain_region_idx_list = []

    t_total_start = time.time()

    for animal in animals_to_process:
        subject_id = animals.index(animal)

        show = args.show_processing
        result = process_animal(animal, data_dir, trial_duration_frames,
                               show_processing=show)

        n_sessions = len(result['neural'])
        if max_sessions is not None:
            n_sessions = min(n_sessions, max_sessions)

        for s in range(n_sessions):
            all_neural.append(result['neural'][s])
            all_input.append(result['input'][s])
            all_output.append(result['output'][s])
            subject_idx_list.append(subject_id)
            # Brain region: all neurons are CA1
            n_neurons_session = result['active_cells'][s]
            brain_region_idx_list.append(np.zeros(n_neurons_session, dtype=int))

    t_total = time.time() - t_total_start

    # Compute statistics
    total_sessions = len(all_neural)
    total_trials = sum(len(s) for s in all_neural)

    print(f"\n=== Conversion Summary ===")
    print(f"Total sessions: {total_sessions}")
    print(f"Total trials: {total_trials}")
    print(f"Trials per session: {[len(s) for s in all_neural[:5]]}...")
    print(f"Neurons per session: {[all_neural[i][0].shape[0] for i in range(min(5, total_sessions))]}...")
    print(f"Total processing time: {t_total:.1f}s")

    # Output bin names
    output_bin_names = []
    for i in range(3):
        for j in range(3):
            x_start = i * 25
            x_end = (i + 1) * 25
            y_start = j * 25
            y_end = (j + 1) * 25
            output_bin_names.append(f"x[{x_start}-{x_end}]_y[{y_start}-{y_end}]")

    # Input names: 9 partition accessibility values
    input_names = [f'partition_{i}' for i in range(9)]

    # Build the output data dictionary
    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,

        'subjects': subjects,
        'subject_idx': np.array(subject_idx_list, dtype=int),

        'brain_regions': ['CA1'],
        'brain_region_idx': brain_region_idx_list,

        'input_names': input_names,
        'output_names': ['position'],
        'output_values': [output_bin_names],

        'metadata': {
            'task_description': 'Decode mouse position (3x3 spatial bins) from CA1 calcium imaging during geometric deformation task',
            'time_bin_size': 1000.0 / fps,  # ~33.33 ms
            'temporal_alignment_event': 'Start of 1-minute trial segment within recording session',
            'off_start': 0.0,
            'off_end': float(trial_duration_sec),
            'fps': fps,
            'trial_duration_sec': trial_duration_sec,
            'trial_duration_frames': trial_duration_frames,
            'env_size_cm': 75.0,
            'n_spatial_bins': 9,
            'spatial_bin_size_cm': 25.0,
            'source_paper': 'Lee, Keinath, Cianfarano & Brandon (2025). Neuron 113(2): 307-320',
            'brain_region': 'CA1 (dorsal hippocampus)',
            'recording_method': 'Miniscope calcium imaging (GCaMP6f)',
            'neural_data_type': 'Binary calcium events (binarized rising-phase transients)',
        }
    }

    # Save
    print(f"\nSaving to {args.outfile}...", flush=True)
    t_save = time.time()
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved in {time.time() - t_save:.1f}s")

    file_size = os.path.getsize(args.outfile)
    print(f"File size: {file_size / 1e6:.1f} MB")
    print("Done.")


if __name__ == '__main__':
    main()
