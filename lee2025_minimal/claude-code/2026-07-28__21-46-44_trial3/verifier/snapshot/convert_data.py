"""
Convert georepca1 calcium imaging data to the standard decoder format.

Reference: Lee, Keinath, Cianfarano & Brandon (2025). Identifying representational
structure in CA1 to benchmark theoretical models of cognitive mapping. Neuron 113(2): 307-320.

Data structure:
- Each animal has multiple recording sessions (days), each ~40 min at 30 Hz
- trace: binary calcium transient events (n_days, n_cells, n_timepoints)
- position: x-y coordinates (n_days, 2, n_timepoints)
- envs: environment name per day
- blocked: which 3x3 grid partitions are blocked per day

Processing:
- Sessions = recording days
- Trials = 1-minute segments within each session
- Neural: binary traces for registered cells only (non-NaN)
- Input: 3x3 environment geometry (flattened, static per trial)
- Output: position discretized into 3x3 = 9 spatial bins (time-varying)
"""

import numpy as np
import joblib
import pickle
import os
import sys

# Constants
FPS = 30  # recording frame rate
TRIAL_DURATION_S = 60  # 1-minute trials
FRAMES_PER_TRIAL = FPS * TRIAL_DURATION_S  # 1800 frames per trial
N_SPATIAL_BINS = 3  # 3x3 grid for output position
ARENA_SIZE = 75.0  # cm, square arena

ANIMALS = [
    "QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
    "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"
]

DATA_DIR = "/app/data"


def get_env_mat(env):
    """
    Get binary 3x3 matrix for environment geometry.
    1 = open partition, 0 = blocked partition.
    Directly from reference code (utils.py).
    """
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
    if env in env_mats:
        return np.array(env_mats[env], dtype=float)
    else:
        raise ValueError(f"Unknown environment: {env}")


def discretize_position(position, n_bins=N_SPATIAL_BINS, arena_size=ARENA_SIZE):
    """
    Discretize x-y position into n_bins x n_bins grid.
    Returns categorical bin index (0 to n_bins*n_bins - 1).

    Position is in [0, arena_size] for both x and y.
    Bins are row-major: bin = row * n_bins + col
    where row corresponds to x (first position dim) and col to y (second position dim).
    """
    # Clip to arena bounds and bin
    x = np.clip(position[0], 0, arena_size - 1e-10)
    y = np.clip(position[1], 0, arena_size - 1e-10)

    bin_size = arena_size / n_bins
    x_bin = np.floor(x / bin_size).astype(int)
    y_bin = np.floor(y / bin_size).astype(int)

    # Clamp to valid range
    x_bin = np.clip(x_bin, 0, n_bins - 1)
    y_bin = np.clip(y_bin, 0, n_bins - 1)

    # Combine into single bin index (row-major)
    bin_idx = x_bin * n_bins + y_bin
    return bin_idx


def convert_data(data_dir=DATA_DIR, animals=ANIMALS, sample=False):
    """
    Convert georepca1 data to standard decoder format.

    Args:
        data_dir: path to data directory
        animals: list of animal IDs
        sample: if True, only use first 2 animals for quick testing
    """
    if sample:
        animals = animals[:2]

    neural_all = []
    input_all = []
    output_all = []
    subject_idx_list = []
    brain_region_idx_all = []

    subjects = list(animals)

    total_sessions = 0
    total_trials = 0
    total_neurons_per_session = []

    for animal_idx, animal in enumerate(animals):
        print(f"Loading {animal}...")
        dat = joblib.load(os.path.join(data_dir, animal))
        d = dat[animal]

        n_days = d['trace'].shape[0]
        n_cells_total = d['trace'].shape[1]
        n_timepoints = d['trace'].shape[2]

        print(f"  {n_days} days, {n_cells_total} total cells, {n_timepoints} timepoints/day")

        for day in range(n_days):
            trace_day = d['trace'][day]  # (n_cells, n_timepoints)
            pos_day = d['position'][day]  # (2, n_timepoints)
            env_name = str(d['envs'][day, 0])

            # Get registered cells (non-NaN traces)
            registered = ~np.all(np.isnan(trace_day), axis=1)
            n_registered = registered.sum()

            if n_registered < 5:
                print(f"  Day {day}: skipping, only {n_registered} registered cells")
                continue

            # Extract registered cell traces
            traces = trace_day[registered]  # (n_registered, n_timepoints)
            # Replace any remaining NaN with 0 (shouldn't happen but safety)
            traces = np.nan_to_num(traces, nan=0.0)

            # Get environment geometry as input (3x3 flattened to 9)
            env_mat = get_env_mat(env_name)
            env_input = env_mat.flatten()  # shape (9,)

            # Discretize position into 3x3 bins
            pos_bins = discretize_position(pos_day)  # shape (n_timepoints,)

            # Split into 1-minute trials
            n_trials = n_timepoints // FRAMES_PER_TRIAL
            if n_trials < 2:
                print(f"  Day {day}: skipping, only {n_trials} possible trials")
                continue

            neural_trials = []
            input_trials = []
            output_trials = []

            for t in range(n_trials):
                start = t * FRAMES_PER_TRIAL
                end = (t + 1) * FRAMES_PER_TRIAL

                trial_traces = traces[:, start:end]  # (n_registered, FRAMES_PER_TRIAL)
                trial_pos = pos_bins[start:end]  # (FRAMES_PER_TRIAL,)

                neural_trials.append(trial_traces.astype(np.float32))
                input_trials.append(env_input.astype(np.float32))  # static per trial, shape (9,)
                # Output: one-hot style categorical position (1, n_timepoints)
                output_trials.append(trial_pos[np.newaxis, :].astype(np.int64))  # (1, FRAMES_PER_TRIAL)

            neural_all.append(neural_trials)
            input_all.append(input_trials)
            output_all.append(output_trials)
            subject_idx_list.append(animal_idx)
            brain_region_idx_all.append(np.zeros(n_registered, dtype=int))

            total_sessions += 1
            total_trials += len(neural_trials)
            total_neurons_per_session.append(n_registered)

            print(f"  Day {day}: {env_name}, {n_registered} neurons, {len(neural_trials)} trials")

        del dat

    # Build output value names for 3x3 position bins
    # Row-major order: bin = row * 3 + col
    position_bin_names = []
    for r in range(N_SPATIAL_BINS):
        for c in range(N_SPATIAL_BINS):
            position_bin_names.append(f"row{r}_col{c}")

    # Build input names for 3x3 environment geometry
    input_names_list = []
    for r in range(N_SPATIAL_BINS):
        for c in range(N_SPATIAL_BINS):
            input_names_list.append(f"grid_{r}_{c}")

    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,

        'subjects': subjects,
        'subject_idx': np.array(subject_idx_list, dtype=int),

        'brain_regions': ['CA1'],
        'brain_region_idx': brain_region_idx_all,

        'input_names': input_names_list,
        'output_names': ['position'],
        'output_values': [position_bin_names],

        'metadata': {
            'task_description': 'Decode mouse position (3x3 spatial bins) from CA1 calcium imaging during geometric deformation of environment',
            'time_bin_size': 1000.0 / FPS,  # ~33.33 ms
            'temporal_alignment_event': 'Start of recording session',
            'off_start': 0.0,
            'off_end': TRIAL_DURATION_S,
            'recording_fps': FPS,
            'trial_duration_s': TRIAL_DURATION_S,
            'frames_per_trial': FRAMES_PER_TRIAL,
            'arena_size_cm': ARENA_SIZE,
            'n_spatial_bins': N_SPATIAL_BINS,
            'n_animals': len(animals),
            'n_sessions': total_sessions,
            'n_trials': total_trials,
            'reference': 'Lee, Keinath, Cianfarano & Brandon (2025). Neuron 113(2): 307-320.',
        }
    }

    # Print summary stats
    print(f"\n=== Conversion Summary ===")
    print(f"Animals: {len(animals)}")
    print(f"Sessions: {total_sessions}")
    print(f"Total trials: {total_trials}")
    print(f"Neurons per session: mean={np.mean(total_neurons_per_session):.1f}, "
          f"min={np.min(total_neurons_per_session)}, max={np.max(total_neurons_per_session)}")
    print(f"Frames per trial: {FRAMES_PER_TRIAL}")
    print(f"Time bin size: {1000.0/FPS:.2f} ms")

    return data


def sanity_checks(data):
    """Run sanity checks on converted data."""
    print("\n=== Sanity Checks ===")

    n_sessions = len(data['neural'])
    print(f"Number of sessions: {n_sessions}")
    print(f"Number of subjects: {len(data['subjects'])}")
    print(f"Subject idx shape: {data['subject_idx'].shape}")

    # Check dimensions consistency
    for s in range(n_sessions):
        n_trials = len(data['neural'][s])
        for t in range(n_trials):
            neural = data['neural'][s][t]
            output = data['output'][s][t]
            inp = data['input'][s][t]

            assert neural.shape[1] == output.shape[1], \
                f"Session {s} trial {t}: neural timepoints {neural.shape[1]} != output timepoints {output.shape[1]}"
            assert neural.shape[1] == FRAMES_PER_TRIAL, \
                f"Session {s} trial {t}: unexpected n_timepoints {neural.shape[1]}"
            assert output.shape[0] == 1, \
                f"Session {s} trial {t}: output should have 1 dim, got {output.shape[0]}"
            assert inp.shape == (9,), \
                f"Session {s} trial {t}: input shape {inp.shape} != (9,)"

    # Check output values are valid
    for s in range(n_sessions):
        for t in range(len(data['output'][s])):
            vals = data['output'][s][t]
            assert np.all((vals >= 0) & (vals < 9)), \
                f"Session {s} trial {t}: output values out of range"

    # Check neural values are binary-ish (0s and 1s with possible intermediate)
    sample_neural = data['neural'][0][0]
    unique_vals = np.unique(sample_neural)
    print(f"Sample neural unique values: {unique_vals}")

    # Cross-check with paper stats
    # Paper: 5,413 unique neurons across 207 sessions in 7 animals
    # We have 7 animals with 21-31 days each
    total_neurons = sum(data['neural'][s][0].shape[0] for s in range(n_sessions))
    print(f"Total neuron-sessions: {total_neurons}")
    print(f"Total sessions: {n_sessions}")

    # Paper says mean cells per animal = 773 +/- 68 SE, min = 515
    # Check unique neurons per animal (max registered across days)
    for subj_id, subj_name in enumerate(data['subjects']):
        session_mask = data['subject_idx'] == subj_id
        session_indices = np.where(session_mask)[0]
        n_neurons_list = [data['neural'][s][0].shape[0] for s in session_indices]
        print(f"  {subj_name}: {len(session_indices)} sessions, neurons/session: "
              f"min={min(n_neurons_list)}, max={max(n_neurons_list)}, mean={np.mean(n_neurons_list):.0f}")

    print("\nAll sanity checks passed!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert georepca1 data to decoder format')
    parser.add_argument('--sample', action='store_true', help='Only use first 2 animals (for quick testing)')
    parser.add_argument('--output', type=str, default=None, help='Output pickle file path')
    args = parser.parse_args()

    if args.sample:
        output_path = args.output or '/app/sample_data.pkl'
    else:
        output_path = args.output or '/app/converted_data.pkl'

    data = convert_data(sample=args.sample)
    sanity_checks(data)

    print(f"\nSaving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f"Saved successfully. File size: {os.path.getsize(output_path) / 1e6:.1f} MB")
