"""
Convert the CA1 geometric-deformation dataset of Lee, Keinath, Cianfarano & Brandon
(2025, Neuron) into the decoder format described in the task.

Source data (Zenodo 10.5281/zenodo.13993254, one joblib file per mouse) contains, for
each of the ~31 recording days of each mouse:
    trace     (n_cells, n_frames)  binarised rising phase of calcium transients (30 Hz).
                                   Cells not registered on that day are all-NaN rows.
    position  (2, n_frames)        head position in cm (x, y) in the 75 x 75 cm arena.
    envs      str                  name of that day's geometry.
    blocked                        indices of the blocked partitions in the 3 x 3 grid,
                                   indexed [[0,1,2],[3,4,5],[6,7,8]], or -1 if none.

Decoder task
    input  : environment geometry - which of the 9 partitions are blocked (static/trial)
    output : the animal's position discretised into the 3 x 3 = 9 partitions (per time bin)

Processing follows the within-session Bayesian position decoding of the paper
(`decode_position_within` / `fit_decoder` in georepca1/src/utils.py):
  * neural: binarised transient-rising-phase vector ("firing rate" in the paper),
    smoothed along time with a Gaussian of sigma = 3 frames and average-pooled into
    non-overlapping 3-frame (100 ms) bins.
  * position: average-pooled into the same 100 ms bins, then divided by the spatial bin
    width and floored. The bin width is derived from the maximum position across all of
    an animal's sessions (as in `decode_position_within`), so the grid is identical for
    every session of a mouse and lines up with the physical partitions.
  * cells: only cells registered on that day (non-NaN) and with more than 5 transients in
    the session are kept (the paper's `cell_threshold` for decoding).

Deviation from the paper, required by the decoder format: the paper's decoder analysis
keeps only frames in which the animal ran faster than 5 cm/s. Here every frame of the
session is kept, because the task asks for contiguous 1-minute trials covering the
session and because the animal's spatial bin is equally well defined while it is
immobile. The speed filter in the paper serves rate-map/place-coding estimation, which
is biased by long stationary epochs; it is not needed to define the decoding target.

Each 40-min session is cut into consecutive, non-overlapping 1-min trials (600 bins of
100 ms); the incomplete remainder at the end of the session is dropped.
"""

import os
import pickle

import joblib
import numpy as np
from scipy.ndimage import gaussian_filter1d

DATA_DIR = "/app/data"
OUT_FILE = "/app/converted_data.pkl"

ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50",
           "QLAK-CA1-51", "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]

FPS = 30                      # acquisition rate of both imaging and behaviour streams
TEMPORAL_BIN_FRAMES = 3       # paper's `temporal_bin_size` for decoding -> 100 ms bins
TRIAL_SECONDS = 60            # 1-minute trials
N_SPATIAL_BINS = 3            # 3 x 3 partition grid of the arena
ARENA_CM = 75.0               # full square environment is 75 x 75 cm
MIN_EVENTS = 5                # paper's `cell_threshold` for the decoding analysis
BUFFER = 1e-15                # paper's rounding buffer for spatial binning

TRIAL_BINS = TRIAL_SECONDS * FPS // TEMPORAL_BIN_FRAMES   # 600 bins per trial
TIME_BIN_MS = 1000.0 * TEMPORAL_BIN_FRAMES / FPS          # 100 ms


def pool_time(x):
    """Average non-overlapping bins of TEMPORAL_BIN_FRAMES along the last axis.

    Mirrors torch.nn.AvgPool1d(kernel_size=3, stride=3) used in the paper's decoder:
    frames left over at the end of the session are dropped.
    """
    n = x.shape[-1] // TEMPORAL_BIN_FRAMES
    return x[..., :n * TEMPORAL_BIN_FRAMES].reshape(
        x.shape[:-1] + (n, TEMPORAL_BIN_FRAMES)).mean(-1)


def session_neural(trace_day):
    """(n_cells, n_frames) binary trace -> (n_kept_cells, n_bins) binned rate + cell mask."""
    registered = ~np.isnan(trace_day[:, 0])
    events = np.nansum(trace_day, axis=1)
    keep = registered & (events > MIN_EVENTS)
    trace = trace_day[keep].astype(np.float32)
    # Gaussian smoothing along time followed by average pooling, as in `fit_decoder`.
    smoothed = gaussian_filter1d(trace, sigma=TEMPORAL_BIN_FRAMES, axis=1)
    return pool_time(smoothed).astype(np.float32), keep


def session_position_bins(position_day, bin_width):
    """(2, n_frames) position in cm -> (n_bins,) index of the occupied partition (0..8).

    Partition index is 3 * row + column with row = y // bin_width and column = x //
    bin_width, which reproduces the indexing of the dataset's `blocked` field (verified:
    partitions listed as blocked are exactly the partitions with zero occupancy).
    """
    pooled = pool_time(position_day.astype(np.float64))          # (2, n_bins), cm
    xy = np.clip((pooled / bin_width).astype(int), 0, N_SPATIAL_BINS - 1)
    return (N_SPATIAL_BINS * xy[1] + xy[0]).astype(np.int64)


def blocked_vector(blocked_day):
    """Dataset `blocked` entry -> (9,) float32, 1 where the partition is walled off."""
    idx = np.array(blocked_day[0]).ravel().astype(int)
    vec = np.zeros(N_SPATIAL_BINS ** 2, dtype=np.float32)
    if idx.size and idx[0] != -1:
        vec[idx] = 1.0
    return vec


def convert():
    data = {
        'neural': [], 'input': [], 'output': [],
        'subjects': list(ANIMALS), 'subject_idx': [],
        'brain_regions': ['CA1'], 'brain_region_idx': [],
        'input_names': [f'partition_{i}_blocked' for i in range(N_SPATIAL_BINS ** 2)],
        'output_names': ['position_bin'],
        'output_values': [[f'partition_{i}_x{i % 3}_y{i // 3}'
                           for i in range(N_SPATIAL_BINS ** 2)]],
        'metadata': {},
    }
    session_info = []

    for animal_idx, animal in enumerate(ANIMALS):
        print(f'loading {animal}', flush=True)
        dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
        trace, position = dat['trace'], dat['position']
        envs, blocked = dat['envs'], dat['blocked']
        n_days = trace.shape[0]

        # One spatial grid for the whole animal, from the largest coordinate reached in
        # any session (as `decode_position_within` does), so that the 3 x 3 grid is the
        # physical partition grid in every geometry, including those the animal cannot
        # fully cover.
        bin_width = (np.nanmax(position) + BUFFER) / N_SPATIAL_BINS

        for day in range(n_days):
            neural, keep = session_neural(trace[day])
            bins = session_position_bins(position[day], bin_width)
            geometry = blocked_vector(blocked[day])

            n_trials = min(neural.shape[1], bins.shape[0]) // TRIAL_BINS
            neural_trials, input_trials, output_trials = [], [], []
            for t in range(n_trials):
                sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
                neural_trials.append(np.ascontiguousarray(neural[:, sl]))
                input_trials.append(geometry.copy())
                output_trials.append(bins[sl][np.newaxis, :])

            data['neural'].append(neural_trials)
            data['input'].append(input_trials)
            data['output'].append(output_trials)
            data['subject_idx'].append(animal_idx)
            data['brain_region_idx'].append(np.zeros(int(keep.sum()), dtype=np.int64))
            session_info.append({
                'subject': animal,
                'day': int(day),
                'sequence': int(day // 10) + 1,
                'environment': str(envs[day][0]),
                'blocked_partitions': np.where(geometry > 0)[0].tolist(),
                'n_neurons': int(keep.sum()),
                'n_trials': int(n_trials),
            })
            print(f'  day {day:2d} {str(envs[day][0]):10s} '
                  f'neurons {keep.sum():3d} trials {n_trials}', flush=True)
        del dat, trace, position

    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
    data['metadata'] = {
        'task_description': (
            'Mice freely explore a 75 x 75 cm square arena partitioned into a 3 x 3 grid. '
            'On each day (session) a different subset of the 9 partitions is walled off '
            'with 25 cm inserts, producing 10 distinct geometries presented in a sequence '
            'that is repeated up to three times per mouse. The decoder is given the '
            'geometry of the environment (which partitions are blocked, static within a '
            'session) and CA1 population calcium activity, and must predict which of the '
            '9 partitions the animal currently occupies.'),
        'time_bin_size': TIME_BIN_MS,
        'temporal_alignment_event': (
            'Start of each 1-minute trial. Sessions are continuous 40-minute free '
            'exploration recordings with no discrete trial events, so each session is cut '
            'into consecutive non-overlapping 1-minute trials.'),
        'off_start': 0.0,
        'off_end': float(TRIAL_SECONDS),
        'recording_modality': (
            'one-photon miniscope calcium imaging of dorsal CA1 (GCaMP6f), 30 Hz'),
        'neural_data_description': (
            'Binarised rising phase of calcium transients (treated as firing rate in the '
            'paper), Gaussian-smoothed along time (sigma = 3 frames) and averaged within '
            'non-overlapping 100 ms bins.'),
        'neuron_curation': (
            f'Per session: cells registered on that day (non-NaN trace) with more than '
            f'{MIN_EVENTS} transients in the session, as in the paper\'s within-session '
            f'decoding analysis. Cells are tracked across days, so the same neuron can '
            f'appear in several sessions of a mouse.'),
        'output_description': (
            'Index of the occupied partition of the 3 x 3 grid, 3 * (y // 25 cm) + '
            '(x // 25 cm), computed from the head position averaged within each 100 ms '
            'bin. Partitions blocked in a given geometry are never occupied in that '
            'session.'),
        'input_description': (
            'Binary vector of length 9, 1 for each partition of the 3 x 3 grid that is '
            'walled off in that session (all zeros for the full square).'),
        'speed_filter': (
            'None. Unlike the paper\'s decoding analysis, frames with running speed below '
            '5 cm/s are retained so that trials are contiguous 1-minute epochs; the '
            'occupied partition is equally well defined during immobility.'),
        'arena_size_cm': ARENA_CM,
        'spatial_bin_size_cm': ARENA_CM / N_SPATIAL_BINS,
        'original_sampling_rate_hz': FPS,
        'trial_duration_s': float(TRIAL_SECONDS),
        'source': ('Lee, Keinath, Cianfarano & Brandon (2025) Neuron 113(2):307-320, '
                   'dataset at https://doi.org/10.5281/zenodo.13993254'),
        'session_info': session_info,
    }
    return data


if __name__ == '__main__':
    data = convert()
    n_trials = sum(len(s) for s in data['neural'])
    n_neurons = sum(len(b) for b in data['brain_region_idx'])
    print(f'{len(data["neural"])} sessions, {n_trials} trials, '
          f'{n_neurons} neuron-sessions')
    with open(OUT_FILE, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f'saved {OUT_FILE} ({os.path.getsize(OUT_FILE) / 1e9:.2f} GB)')
