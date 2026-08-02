"""
Convert georepca1 calcium imaging data to standardized decoder format.

Reference: Lee et al. (2025) "Identifying representational structure in CA1
to benchmark theoretical models of cognitive mapping", Neuron 113, 307-320.

Data: CA1 calcium imaging in mice navigating 10 geometric environments.
- 7 mice, 207 sessions total, 5413 unique neurons
- Binary rising-phase transients at 30 Hz
- Sessions are 40 min each, split into 1-min trials
- Position discretized into 3x3 spatial bins
- Environment geometry encoded as 3x3 binary matrix (input to decoder)
"""

import numpy as np
import pickle
import sys
import os
from mat73 import loadmat

# Constants
FPS = 30  # frames per second
TRIAL_DURATION_S = 60  # 1 minute trials
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S  # 1800 frames per trial
N_SPATIAL_BINS = 3  # 3x3 grid for position output
BUFFER = 1e-5

ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def get_env_mat(env):
    """Get binary 3x3 matrix for environment geometry. From reference code."""
    env_mats = {
        'square':    [[1,1,1],[1,1,1],[1,1,1]],
        'o':         [[1,1,1],[1,0,1],[1,1,1]],
        't':         [[0,1,0],[0,1,0],[1,1,1]],
        'u':         [[1,1,1],[1,0,0],[1,1,1]],
        'rectangle': [[0,1,1],[0,1,1],[0,1,1]],
        '+':         [[0,1,0],[1,1,1],[0,1,0]],
        'i':         [[1,1,1],[0,1,0],[1,1,1]],
        'l':         [[1,1,1],[1,0,0],[1,0,0]],
        'bit donut': [[1,1,1],[1,0,1],[0,1,1]],
        'glenn':     [[1,1,0],[1,1,1],[0,1,1]],
    }
    return np.array(env_mats.get(env, [[0,0,0],[0,0,0],[0,0,0]])).astype(float)


def discretize_position(position, n_bins=N_SPATIAL_BINS):
    """
    Discretize 2D position into n_bins x n_bins grid.
    Position is (2, T) with values in [0, ~75].
    Returns bin index 0..n_bins^2-1 for each timepoint.
    """
    pos = position.copy()
    # Bin each dimension
    max_vals = np.nanmax(pos, axis=1, keepdims=True)
    bin_size = (max_vals + BUFFER) / n_bins
    binned = np.floor(pos / bin_size).astype(int)
    # Clip to valid range
    binned = np.clip(binned, 0, n_bins - 1)
    # Convert 2D bin to single index (row-major): bin_idx = x * n_bins + y
    bin_idx = binned[0] * n_bins + binned[1]
    return bin_idx


def convert_data(save_path="converted_data.pkl", sample_path="sample_data.pkl"):
    """Convert all animal data to standardized format."""

    all_neural = []
    all_input = []
    all_output = []
    subject_idx_list = []
    brain_region_idx_list = []

    total_sessions = 0
    total_trials = 0

    for animal_idx, animal in enumerate(ANIMALS):
        print(f"\nLoading {animal}...")
        dat = loadmat(os.path.join(DATA_DIR, f"{animal}.mat"))

        n_sessions = len(dat['trace'])
        n_neurons_total = dat['trace'][0].shape[0]

        print(f"  {n_sessions} sessions, {n_neurons_total} tracked neurons")

        for sess_idx in range(n_sessions):
            trace = np.array(dat['trace'][sess_idx])  # (n_neurons, n_timepoints)
            position = np.array(dat['position'][sess_idx])  # (2, n_timepoints)
            env_name = dat['envs'][sess_idx][0]

            n_timepoints = trace.shape[1]

            # Identify valid (non-NaN) neurons for this session
            valid_neurons = ~np.isnan(trace).all(axis=1)
            n_valid = valid_neurons.sum()

            if n_valid == 0:
                print(f"  Session {sess_idx} ({env_name}): skipped, no valid neurons")
                continue

            # Extract valid neuron traces, replace any remaining NaN with 0
            # Data is binary 0/1, use float32 for compatibility with decoder
            trace_valid = trace[valid_neurons].copy()
            trace_valid = np.nan_to_num(trace_valid, nan=0.0).astype(np.float32)

            # Get environment geometry matrix (3x3 flattened to 9)
            env_mat = get_env_mat(env_name).flatten().astype(np.float32)  # (9,)

            # Discretize position into 3x3 bins
            pos_bins = discretize_position(position, N_SPATIAL_BINS)  # (n_timepoints,)

            # Split into 1-minute trials
            n_full_trials = n_timepoints // FRAMES_PER_TRIAL

            if n_full_trials < 2:
                print(f"  Session {sess_idx} ({env_name}): skipped, < 2 trials")
                continue

            session_neural = []
            session_input = []
            session_output = []

            for trial_idx in range(n_full_trials):
                start = trial_idx * FRAMES_PER_TRIAL
                end = start + FRAMES_PER_TRIAL

                # Neural: (n_valid_neurons, FRAMES_PER_TRIAL)
                trial_neural = trace_valid[:, start:end]

                # Input: environment geometry, static per trial (9,)
                trial_input = env_mat.copy()

                # Output: discretized position (1, FRAMES_PER_TRIAL)
                trial_output = pos_bins[start:end].reshape(1, -1).astype(np.int64)

                session_neural.append(trial_neural)
                session_input.append(trial_input)
                session_output.append(trial_output)

            all_neural.append(session_neural)
            all_input.append(session_input)
            all_output.append(session_output)
            subject_idx_list.append(animal_idx)
            brain_region_idx_list.append(np.zeros(n_valid, dtype=int))  # all CA1

            total_sessions += 1
            total_trials += n_full_trials
            print(f"  Session {sess_idx} ({env_name}): {n_valid} neurons, {n_full_trials} trials")

    # Build output value names for 3x3 position grid
    pos_labels = []
    for i in range(N_SPATIAL_BINS):
        for j in range(N_SPATIAL_BINS):
            pos_labels.append(f"bin_({i},{j})")

    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,

        'subjects': ANIMALS,
        'subject_idx': np.array(subject_idx_list, dtype=int),

        'brain_regions': ['CA1'],
        'brain_region_idx': brain_region_idx_list,

        'input_names': [f'geometry_{i}' for i in range(9)],
        'output_names': ['position'],
        'output_values': [pos_labels],

        'metadata': {
            'task_description': 'Decode mouse position from CA1 neural activity during free exploration of geometric environments',
            'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
            'temporal_alignment_event': 'Start of recording session',
            'off_start': 0.0,
            'off_end': float(TRIAL_DURATION_S),
            'recording_fps': FPS,
            'trial_duration_s': TRIAL_DURATION_S,
            'n_spatial_bins': N_SPATIAL_BINS,
            'arena_size_cm': 75,
            'session_duration_min': 40,
            'n_animals': len(ANIMALS),
            'n_sessions': total_sessions,
            'n_trials': total_trials,
            'n_unique_neurons': 5413,  # from paper: 5,413 unique neurons across 207 sessions
            'environments': ['square', 'o', 't', 'u', 'rectangle', '+', 'i', 'l', 'bit donut', 'glenn'],
            'neural_data_type': 'Binary rising-phase calcium transients',
        }
    }

    print(f"\n--- Summary ---")
    print(f"Total sessions: {total_sessions}")
    print(f"Total trials: {total_trials}")
    print(f"Subjects: {len(ANIMALS)}")

    # Save full dataset
    print(f"\nSaving full dataset to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved ({os.path.getsize(save_path) / 1e9:.2f} GB)")

    # Create sample dataset (first 2 sessions per animal, or fewer)
    print(f"\nCreating sample dataset...")
    sample_data = create_sample(data, max_sessions_per_animal=2)
    with open(sample_path, 'wb') as f:
        pickle.dump(sample_data, f)
    print(f"Saved sample to {sample_path} ({os.path.getsize(sample_path) / 1e6:.1f} MB)")

    return data


def create_sample(data, max_sessions_per_animal=2):
    """Create a smaller sample dataset for quick testing."""
    sample = {
        'subjects': data['subjects'],
        'brain_regions': data['brain_regions'],
        'input_names': data['input_names'],
        'output_names': data['output_names'],
        'output_values': data['output_values'],
        'metadata': data['metadata'].copy(),
    }

    neural, inp, out = [], [], []
    subject_idx_list = []
    brain_region_idx_list = []

    # Count sessions per subject
    subject_counts = {}
    for i, si in enumerate(data['subject_idx']):
        si_int = int(si)
        if subject_counts.get(si_int, 0) < max_sessions_per_animal:
            neural.append(data['neural'][i])
            inp.append(data['input'][i])
            out.append(data['output'][i])
            subject_idx_list.append(si)
            brain_region_idx_list.append(data['brain_region_idx'][i])
            subject_counts[si_int] = subject_counts.get(si_int, 0) + 1

    sample['neural'] = neural
    sample['input'] = inp
    sample['output'] = out
    sample['subject_idx'] = np.array(subject_idx_list, dtype=int)
    sample['brain_region_idx'] = brain_region_idx_list

    return sample


if __name__ == "__main__":
    data = convert_data()
