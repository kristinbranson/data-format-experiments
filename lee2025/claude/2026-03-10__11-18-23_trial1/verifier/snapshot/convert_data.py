#!/usr/bin/env python3
"""
Convert georepca1 CA1 calcium imaging data to decoder-compatible format.

Usage:
    python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]

Reference: Lee, Keinath, Cianfarano & Brandon (2025). Identifying representational
structure in CA1 to benchmark theoretical models of cognitive mapping. Neuron 113(2).
"""

import sys
import os
import argparse
import time
import pickle
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# Constants from reference code and paper
# ============================================================================
ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FPS = 30  # Recording frame rate (Hz)
TRIAL_DURATION_S = 60  # 1-minute trials
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S  # 1800 frames
MIN_TRIAL_FRAMES = FPS * 30  # Minimum 30s for a partial trial at end
N_POS_BINS = 3  # 3x3 position grid for output
ENV_SIZE_CM = 75.0  # Environment is 75x75 cm

# ============================================================================
# Environment geometry functions (from reference code utils.py)
# ============================================================================
def get_env_mat(env):
    """Get binary 3x3 matrix for environment geometry. From reference code."""
    if env == 'square':
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(float)
    elif env == 'o':
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]).astype(float)
    elif env == 't':
        return np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]]).astype(float)
    elif env == 'u':
        return np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]]).astype(float)
    elif env == 'rectangle':
        return np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1]]).astype(float)
    elif env == '+':
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype(float)
    elif env == 'i':
        return np.array([[1, 1, 1], [0, 1, 0], [1, 1, 1]]).astype(float)
    elif env == 'l':
        return np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]]).astype(float)
    elif env == 'bit donut':
        return np.array([[1, 1, 1], [1, 0, 1], [0, 1, 1]]).astype(float)
    elif env == 'glenn':
        return np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]]).astype(float)
    else:
        raise ValueError(f"Unknown environment: {env}")


def bin_position_3x3(position, env_size=ENV_SIZE_CM):
    """
    Bin x,y position into 3x3 grid. Returns integer bin index 0-8 (row-major).
    position: (2, n_frames) array of x, y coordinates
    Returns: (n_frames,) array of bin indices 0-8
    """
    buffer = 1e-5
    bin_size = (env_size + buffer) / N_POS_BINS
    x_bin = np.clip(np.floor(position[0] / bin_size).astype(int), 0, N_POS_BINS - 1)
    y_bin = np.clip(np.floor(position[1] / bin_size).astype(int), 0, N_POS_BINS - 1)
    # Row-major: bin_idx = x_bin * 3 + y_bin
    # Note: x is the first spatial dimension, y is the second
    # Using same convention as get_env_mat: row=x, col=y
    bin_idx = x_bin * N_POS_BINS + y_bin
    return bin_idx


def load_animal_data(animal):
    """Load preprocessed data for one animal from joblib file."""
    filepath = os.path.join(DATA_DIR, animal)
    dat = joblib.load(filepath)
    return dat[animal]


def process_animal(animal, show_processing=False):
    """
    Process all sessions for one animal.

    Returns lists of (neural_trials, input_trials, output_trials) for each session,
    plus metadata about the animal.
    """
    t0 = time.time()
    print(f"\nProcessing {animal}...")
    d = load_animal_data(animal)

    n_days = d['trace'].shape[0]
    n_cells_total = d['trace'].shape[1]
    n_frames_total = d['trace'].shape[2]
    envs = d['envs'].squeeze()

    print(f"  {n_cells_total} total cells, {n_days} days, {n_frames_total} frames/day")

    all_neural = []
    all_input = []
    all_output = []
    session_info = []

    for day in range(n_days):
        t_day = time.time()
        env_name = envs[day]

        # Get trace for this day: (n_cells, n_frames)
        trace = d['trace'][day]
        position = d['position'][day]  # (2, n_frames)
        n_frames = trace.shape[1]

        # Identify registered cells (non-NaN on this day)
        registered_mask = ~np.isnan(trace[:, 0])
        n_registered = np.sum(registered_mask)

        if n_registered == 0:
            print(f"  Day {day} ({env_name}): No registered cells, skipping")
            continue

        # Extract registered cells' traces
        trace_registered = trace[registered_mask]  # (n_registered, n_frames)

        # Get environment geometry as input (3x3 flattened to 9)
        env_mat = get_env_mat(env_name).flatten()  # (9,)

        # Bin position to 3x3 grid
        pos_bins = bin_position_3x3(position)  # (n_frames,)

        # Split into 1-minute trials
        trial_starts = list(range(0, n_frames, FRAMES_PER_TRIAL))
        neural_trials = []
        input_trials = []
        output_trials = []

        for start in trial_starts:
            end = min(start + FRAMES_PER_TRIAL, n_frames)
            trial_len = end - start

            # Skip short partial trials
            if trial_len < MIN_TRIAL_FRAMES:
                continue

            # Neural: (n_registered, trial_len)
            neural_trial = trace_registered[:, start:end].astype(np.float32)

            # Input: environment geometry, static per trial -> (9,)
            input_trial = env_mat.astype(np.float32)

            # Output: position bin, time-varying -> (1, trial_len)
            output_trial = pos_bins[start:end].astype(np.int64).reshape(1, -1)

            neural_trials.append(neural_trial)
            input_trials.append(input_trial)
            output_trials.append(output_trial)

        n_trials = len(neural_trials)
        if n_trials < 2:
            print(f"  Day {day} ({env_name}): Only {n_trials} trial(s), skipping (need >=2)")
            continue

        all_neural.append(neural_trials)
        all_input.append(input_trials)
        all_output.append(output_trials)
        session_info.append({
            'animal': animal,
            'day': day,
            'env': env_name,
            'n_registered': int(n_registered),
            'n_trials': n_trials,
        })

        dt = time.time() - t_day
        print(f"  Day {day} ({env_name}): {n_registered} cells, {n_trials} trials, {dt:.1f}s")

    if show_processing and len(all_neural) > 0:
        plot_processing(animal, d, all_neural, all_input, all_output, session_info)

    dt_total = time.time() - t0
    print(f"  {animal} done: {len(all_neural)} sessions, {dt_total:.1f}s total")

    del d  # Free memory
    return all_neural, all_input, all_output, session_info


def plot_processing(animal, d, all_neural, all_input, all_output, session_info):
    """Plot processing visualizations for up to 2 sessions."""
    n_to_plot = min(2, len(all_neural))

    fig, axes = plt.subplots(n_to_plot, 4, figsize=(24, 6 * n_to_plot))
    if n_to_plot == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_to_plot):
        info = session_info[i]
        neural_trials = all_neural[i]
        input_trials = all_input[i]
        output_trials = all_output[i]

        # Plot 1: Neural activity heatmap (first trial, first 50 cells)
        ax = axes[i, 0]
        n_show = min(50, neural_trials[0].shape[0])
        t_show = min(600, neural_trials[0].shape[1])  # First 20 seconds
        ax.imshow(neural_trials[0][:n_show, :t_show], aspect='auto', cmap='binary',
                  interpolation='none')
        ax.set_title(f"Day {info['day']} ({info['env']})\nNeural (first {n_show} cells, {t_show} frames)")
        ax.set_xlabel('Frame')
        ax.set_ylabel('Cell')

        # Plot 2: Position trace and binned output
        ax = axes[i, 1]
        day = info['day']
        pos = d['position'][day]
        t_range = slice(0, min(FRAMES_PER_TRIAL, pos.shape[1]))
        ax.plot(pos[0, t_range], pos[1, t_range], 'b-', alpha=0.3, linewidth=0.5)
        ax.set_title(f"Position trace (trial 0)")
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_xlim(0, ENV_SIZE_CM)
        ax.set_ylim(0, ENV_SIZE_CM)
        ax.set_aspect('equal')
        # Draw 3x3 grid
        for g in range(1, N_POS_BINS):
            ax.axhline(g * ENV_SIZE_CM / N_POS_BINS, color='r', linewidth=1, alpha=0.5)
            ax.axvline(g * ENV_SIZE_CM / N_POS_BINS, color='r', linewidth=1, alpha=0.5)

        # Plot 3: Output (position bin) over time
        ax = axes[i, 2]
        out_trial = output_trials[0][0]  # (trial_len,)
        ax.plot(out_trial[:600], 'k-', linewidth=0.5)
        ax.set_title(f"Output: position bin (trial 0)")
        ax.set_xlabel('Frame')
        ax.set_ylabel('Bin (0-8)')
        ax.set_ylim(-0.5, 8.5)

        # Plot 4: Environment geometry input
        ax = axes[i, 3]
        env_mat = input_trials[0].reshape(3, 3)
        ax.imshow(env_mat, cmap='binary', vmin=0, vmax=1)
        ax.set_title(f"Input: env geometry ({info['env']})")
        for r in range(3):
            for c in range(3):
                ax.text(c, r, f"{int(env_mat[r, c])}", ha='center', va='center',
                       color='red' if env_mat[r, c] == 0 else 'blue', fontsize=14)

    fig.suptitle(f"Processing: {animal}", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"processing_{animal}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved processing_{animal}.png")


def build_dataset(animals, sample_mode=False, show_processing=False):
    """Build the complete dataset dictionary."""
    t0 = time.time()

    if sample_mode:
        # Use first 2 animals for sample
        animals = animals[:2]
        print(f"SAMPLE MODE: Processing {len(animals)} animals: {animals}")
    else:
        print(f"FULL MODE: Processing {len(animals)} animals")

    all_neural = []
    all_input = []
    all_output = []
    all_session_info = []
    subject_idx_list = []
    brain_region_idx_list = []

    for animal in animals:
        neural, inp, out, sess_info = process_animal(
            animal, show_processing=show_processing
        )

        subject_id = ANIMALS.index(animal)

        for s_idx in range(len(neural)):
            all_neural.append(neural[s_idx])
            all_input.append(inp[s_idx])
            all_output.append(out[s_idx])
            all_session_info.append(sess_info[s_idx])
            subject_idx_list.append(subject_id)
            # All neurons are from CA1
            n_neurons = neural[s_idx][0].shape[0]
            brain_region_idx_list.append(np.zeros(n_neurons, dtype=np.int64))

    # Build output names: position bin labels
    output_names = ['position_bin']
    output_values = [
        [f"x{r}y{c}" for r in range(N_POS_BINS) for c in range(N_POS_BINS)]
    ]

    # Input names: environment geometry partitions
    input_names = [f"env_partition_{r}{c}" for r in range(N_POS_BINS) for c in range(N_POS_BINS)]

    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': ANIMALS,  # All animal names even in sample mode
        'subject_idx': np.array(subject_idx_list, dtype=np.int64),
        'brain_regions': ['CA1'],
        'brain_region_idx': brain_region_idx_list,
        'input_names': input_names,
        'output_names': output_names,
        'output_values': output_values,
        'metadata': {
            'task_description': 'Decode mouse position (3x3 spatial bins) from CA1 calcium imaging during geometric deformation task',
            'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
            'temporal_alignment_event': 'Start of each 1-minute trial segment within a 40-minute recording session',
            'off_start': 0.0,
            'off_end': float(TRIAL_DURATION_S),
            'session_info': all_session_info,
            'recording_fps': FPS,
            'environment_size_cm': ENV_SIZE_CM,
            'trial_duration_s': TRIAL_DURATION_S,
            'n_position_bins': N_POS_BINS,
            'paper': 'Lee et al. (2025) Neuron 113(2): 307-320',
        }
    }

    dt = time.time() - t0
    n_sessions = len(all_neural)
    n_trials = sum(len(s) for s in all_neural)
    print(f"\nDataset built: {n_sessions} sessions, {n_trials} trials, {dt:.1f}s")

    return data


def main():
    parser = argparse.ArgumentParser(description='Convert georepca1 data to decoder format')
    parser.add_argument('outfile', type=str, help='Output pickle file path')
    parser.add_argument('--full', action='store_true', default=True, help='Process all sessions (default)')
    parser.add_argument('--sample', action='store_true', help='Process only 2 animals for testing')
    parser.add_argument('--show-processing', action='store_true', help='Plot processing visualizations')
    args = parser.parse_args()

    sample_mode = args.sample

    data = build_dataset(
        ANIMALS,
        sample_mode=sample_mode,
        show_processing=args.show_processing
    )

    # Save
    t0 = time.time()
    print(f"\nSaving to {args.outfile}...")
    with open(args.outfile, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    dt = time.time() - t0
    filesize = os.path.getsize(args.outfile) / (1024**2)
    print(f"Saved {filesize:.1f} MB in {dt:.1f}s")

    # Print summary stats
    print("\n=== Summary ===")
    print(f"Animals: {len(set(data['subject_idx']))}")
    print(f"Sessions: {len(data['neural'])}")
    n_trials = sum(len(s) for s in data['neural'])
    print(f"Trials: {n_trials}")
    neurons_per_session = [s[0].shape[0] for s in data['neural']]
    print(f"Neurons/session: min={min(neurons_per_session)}, max={max(neurons_per_session)}, mean={np.mean(neurons_per_session):.0f}")
    print(f"Time bin: {data['metadata']['time_bin_size']:.2f} ms")

    # Output distribution
    all_out = np.concatenate([
        np.concatenate([t[0] for t in sess]) for sess in data['output']
    ])
    unique, counts = np.unique(all_out, return_counts=True)
    print(f"\nOutput distribution (position_bin):")
    for u, c in zip(unique, counts):
        print(f"  Bin {int(u)} ({data['output_values'][0][int(u)]}): {c/len(all_out)*100:.1f}%")


if __name__ == '__main__':
    main()
