"""
Convert CA1 geometric deformation data from Lee et al. (2025) to the standardized decoder format.

Experiment: Mice freely explored 10 geometrically distinct environments (created by blocking
partitions of a 75x75 cm square arena treated as a 3x3 grid). Sessions were 40 minutes each,
recorded at 30 Hz with miniscope calcium imaging. Neural data is binary (rise-extracted calcium
transients). Each day = one session = one environment geometry.

Decoder task: Predict mouse position (discretized into 3x3 = 9 spatial bins) from neural activity,
with environment geometry as decoder input.

Processing:
- Each day/session for each animal becomes a session in the converted data
- Each session is split into 1-minute trials (1800 frames at 30 Hz)
- Neural data is binned into 1-second time bins (sum of 30 binary frames)
- Position is discretized into 3x3 spatial bins
- Only cells registered (non-NaN) on a given day are included
- Environment geometry represented as flattened 3x3 binary matrix (9 values)
"""

import os
import sys
import pickle
import numpy as np
import joblib
from copy import deepcopy

# ---- Configuration ----
DATA_DIR = "data"
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
FPS = 30  # frames per second
TRIAL_DURATION_S = 60  # trial duration in seconds
TRIAL_FRAMES = FPS * TRIAL_DURATION_S  # 1800 frames per trial
TIME_BIN_FRAMES = FPS  # 30 frames per time bin = 1 second
TIME_BIN_MS = 1000.0 / FPS * TIME_BIN_FRAMES  # 1000 ms
N_SPATIAL_BINS = 3  # 3x3 grid for position output
POSITION_BUFFER = 1e-5  # small buffer for binning edge positions


def get_env_mat(env):
    """Get 3x3 binary matrix for environment geometry (1=accessible, 0=blocked).
    Matches get_env_mat from reference code."""
    envs = {
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
    if env in envs:
        return np.array(envs[env], dtype=float)
    else:
        raise ValueError(f"Unknown environment: {env}")


def discretize_position(position, n_bins=3):
    """Discretize x-y position into spatial bin index (0 to n_bins^2 - 1).

    Args:
        position: shape (2, n_frames) - x,y coordinates
        n_bins: number of bins per dimension (3 for 3x3 grid)

    Returns:
        bin_indices: shape (n_frames,) - integer bin index 0 to n_bins^2-1
    """
    x = position[0]
    y = position[1]

    x_max = np.nanmax(x) + POSITION_BUFFER
    y_max = np.nanmax(y) + POSITION_BUFFER

    x_bin = np.floor(x / (x_max / n_bins)).astype(int)
    y_bin = np.floor(y / (y_max / n_bins)).astype(int)

    # Clip to valid range
    x_bin = np.clip(x_bin, 0, n_bins - 1)
    y_bin = np.clip(y_bin, 0, n_bins - 1)

    # Combined bin index: row * n_cols + col
    bin_idx = x_bin * n_bins + y_bin
    return bin_idx


def bin_neural_data(trace, time_bin_frames):
    """Bin binary neural trace into time bins by summing frames.

    Args:
        trace: shape (n_cells, n_frames) - binary trace
        time_bin_frames: number of frames per time bin

    Returns:
        binned: shape (n_cells, n_timebins) - event counts per bin
    """
    n_cells, n_frames = trace.shape
    n_bins = n_frames // time_bin_frames
    # Truncate to exact multiple
    truncated = trace[:, :n_bins * time_bin_frames]
    # Reshape and sum
    reshaped = truncated.reshape(n_cells, n_bins, time_bin_frames)
    return reshaped.sum(axis=2).astype(float)


def bin_position(bin_indices, time_bin_frames):
    """Bin position indices by taking the mode within each time bin.

    Args:
        bin_indices: shape (n_frames,) - position bin indices
        time_bin_frames: number of frames per time bin

    Returns:
        binned: shape (n_timebins,) - position bin per time bin
    """
    n_frames = len(bin_indices)
    n_bins = n_frames // time_bin_frames
    truncated = bin_indices[:n_bins * time_bin_frames]
    reshaped = truncated.reshape(n_bins, time_bin_frames)
    # Take mode (most common bin in each time window)
    from scipy.stats import mode
    binned = mode(reshaped, axis=1, keepdims=False)[0].astype(int)
    return binned


def convert_data(data_dir, animals, output_path, sample=False, max_sessions_per_animal=None):
    """Convert the CA1 data to the standardized decoder format.

    Args:
        data_dir: path to data directory
        animals: list of animal IDs
        output_path: path to save the converted data
        sample: if True, only use first 2 animals with limited sessions
        max_sessions_per_animal: limit number of sessions per animal (for sample)
    """
    neural_all = []
    input_all = []
    output_all = []
    subjects = animals.copy()
    subject_idx_all = []
    brain_region_idx_all = []

    total_sessions = 0
    total_trials = 0
    total_neurons_unique = 0

    for a_idx, animal in enumerate(animals):
        print(f"Processing {animal}...")
        dat = joblib.load(os.path.join(data_dir, animal))
        d = dat[animal]

        n_days = d['envs'].shape[0]
        n_cells_total = d['trace'].shape[1]
        n_frames = d['trace'].shape[2]
        total_neurons_unique += n_cells_total

        if max_sessions_per_animal is not None:
            n_days = min(n_days, max_sessions_per_animal)

        for day in range(n_days):
            env_name = str(d['envs'][day, 0])
            trace_day = d['trace'][day]  # (n_cells, n_frames)
            position_day = d['position'][day]  # (2, n_frames)

            # Filter to registered cells (non-NaN)
            registered = ~np.all(np.isnan(trace_day), axis=1)
            trace_registered = trace_day[registered]  # (n_registered, n_frames)
            n_registered = registered.sum()

            if n_registered == 0:
                print(f"  Skipping {animal} day {day}: no registered cells")
                continue

            # Replace any remaining NaNs with 0 (shouldn't happen but be safe)
            trace_registered = np.nan_to_num(trace_registered, nan=0.0)

            # Get environment geometry
            env_mat = get_env_mat(env_name)  # 3x3 binary
            env_flat = env_mat.flatten()  # (9,)

            # Discretize position into 3x3 bins
            pos_bins = discretize_position(position_day, N_SPATIAL_BINS)  # (n_frames,)

            # Split into 1-minute trials
            n_trials = n_frames // TRIAL_FRAMES
            if n_trials < 2:
                print(f"  Skipping {animal} day {day}: only {n_trials} trial(s)")
                continue

            session_neural = []
            session_input = []
            session_output = []

            for trial in range(n_trials):
                t_start = trial * TRIAL_FRAMES
                t_end = (trial + 1) * TRIAL_FRAMES

                trial_trace = trace_registered[:, t_start:t_end]  # (n_registered, 1800)
                trial_pos_bins = pos_bins[t_start:t_end]  # (1800,)

                # Bin neural data into 1-second time bins
                trial_neural = bin_neural_data(trial_trace, TIME_BIN_FRAMES)  # (n_registered, 60)

                # Bin position data
                trial_output = bin_position(trial_pos_bins, TIME_BIN_FRAMES)  # (60,)
                trial_output = trial_output.reshape(1, -1)  # (1, 60)

                session_neural.append(trial_neural)
                session_input.append(env_flat)  # (9,) static per trial
                session_output.append(trial_output)

            neural_all.append(session_neural)
            input_all.append(session_input)
            output_all.append(session_output)
            subject_idx_all.append(a_idx)
            brain_region_idx_all.append(np.zeros(n_registered, dtype=int))

            total_sessions += 1
            total_trials += len(session_neural)

        del dat

    # Build output names for 3x3 position bins
    position_labels = []
    for row in range(N_SPATIAL_BINS):
        for col in range(N_SPATIAL_BINS):
            position_labels.append(f"row{row}_col{col}")

    # Build input names
    input_labels = []
    for row in range(N_SPATIAL_BINS):
        for col in range(N_SPATIAL_BINS):
            input_labels.append(f"geometry_row{row}_col{col}")

    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,
        'subjects': subjects,
        'subject_idx': np.array(subject_idx_all),
        'brain_regions': ['CA1'],
        'brain_region_idx': brain_region_idx_all,
        'input_names': input_labels,
        'output_names': ['position'],
        'output_values': [position_labels],
        'metadata': {
            'task_description': 'Decode mouse position (3x3 spatial bins) from CA1 calcium activity during free exploration of geometrically deformed environments',
            'time_bin_size': TIME_BIN_MS,
            'temporal_alignment_event': 'Start of recording session',
            'off_start': 0.0,
            'off_end': float(TRIAL_DURATION_S),
            'recording_fps': FPS,
            'trial_duration_s': TRIAL_DURATION_S,
            'session_duration_min': 40,
            'arena_size_cm': 75,
            'n_spatial_bins': N_SPATIAL_BINS,
            'neural_data_type': 'Binary calcium transient event counts (summed over 1s time bins)',
            'source_paper': 'Lee, Keinath, Cianfarano & Brandon (2025). Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping. Neuron 113(2): 307-320.',
            'total_unique_neurons': total_neurons_unique,
            'total_sessions': total_sessions,
            'total_trials': total_trials,
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"\nConversion complete:")
    print(f"  Total sessions: {total_sessions}")
    print(f"  Total trials: {total_trials}")
    print(f"  Total unique neurons: {total_neurons_unique}")
    print(f"  Subjects: {subjects}")
    print(f"  Saved to: {output_path}")

    return data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert CA1 data to decoder format')
    parser.add_argument('--sample', action='store_true', help='Create sample dataset (2 animals, 4 sessions each)')
    parser.add_argument('--output', type=str, default=None, help='Output path')
    args = parser.parse_args()

    if args.sample:
        output_path = args.output or "sample_data.pkl"
        data = convert_data(DATA_DIR, ANIMALS[:2], output_path, sample=True, max_sessions_per_animal=4)
    else:
        output_path = args.output or "converted_data.pkl"
        data = convert_data(DATA_DIR, ANIMALS, output_path)
