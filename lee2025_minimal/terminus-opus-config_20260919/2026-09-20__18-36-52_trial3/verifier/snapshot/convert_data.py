"""
Convert the CA1 geometric-deformation dataset of Lee et al. (2025), "Identifying
representational structure in CA1 to benchmark theoretical models of cognitive
mapping" (Neuron 113:307-320), into the decoder dataset format.

Decoder task: predict the mouse's discretized position (3 x 3 = 9 spatial bins)
from CA1 calcium activity, given the environment geometry (which of the 9
partitions are blocked) as input.

Processing follows the reference code (georepca1/src/utils.py):
  * data are loaded from the joblib files distributed with the paper
  * neural activity is the binarized transient rising-phase vector ('trace'),
    treated as the firing rate in all analyses of the paper
  * as in utils.fit_decoder/test_decoder, traces are smoothed with a gaussian
    (sigma = 3 frames) and average-pooled in non-overlapping bins of 3 frames
    (100 ms at 30 Hz)
  * cells not registered on a given day are NaN and are dropped for that session;
    as in utils.decode_position_within (cell_threshold=5) cells with <= 5
    transients in a session are also dropped
  * position is binned into a 3 x 3 grid of the 75 x 75 cm arena (25 cm bins),
    matching the 3 x 3 partition design of the experiment. Bin index is
    ybin*3 + xbin, the same convention used by the 'blocked' field.
"""

import os
import pickle
import numpy as np
import joblib
from scipy.ndimage import gaussian_filter1d

DATA_DIR = '/app/data'
OUT_FILE = '/app/converted_data.pkl'

ANIMALS = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']

FPS = 30.0                 # acquisition rate of both behavior and imaging streams
TEMPORAL_BIN_FRAMES = 3    # utils.fit_decoder default -> 100 ms bins
SMOOTH_SIGMA = 3           # frames, utils.fit_decoder gaussian_filter1d sigma
TRIAL_SECONDS = 60.0       # 1-minute trials
ARENA_SIZE = 75.0          # cm, full square environment
N_SPATIAL_BINS = 3         # 3 x 3 partition design
EVENT_THRESHOLD = 5        # utils.decode_position_within cell_threshold

BIN_MS = TEMPORAL_BIN_FRAMES / FPS * 1000.0
BINS_PER_TRIAL = int(round(TRIAL_SECONDS * FPS / TEMPORAL_BIN_FRAMES))


def bin_time(x, n_frames_bin):
    """Average non-overlapping bins of n_frames_bin along the last axis."""
    n = x.shape[-1] // n_frames_bin
    x = x[..., :n * n_frames_bin]
    return x.reshape(*x.shape[:-1], n, n_frames_bin).mean(axis=-1)


def geometry_vector(blocked_day):
    """9-dim binary vector, 1 if that partition of the 3x3 grid is blocked."""
    vec = np.zeros(N_SPATIAL_BINS * N_SPATIAL_BINS, dtype=np.float32)
    b = np.atleast_1d(np.array(blocked_day[0]).ravel())
    for i in b:
        if i >= 0:
            vec[int(i)] = 1.0
    return vec


def main():
    neural_all, input_all, output_all = [], [], []
    subject_idx, brain_region_idx = [], []
    session_info = []

    n_dropped_silent = 0
    n_cells_total = 0

    for ai, animal in enumerate(ANIMALS):
        print(f'Loading {animal} ...', flush=True)
        dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
        trace = dat['trace']        # (n_days, n_cells, n_frames), binary, NaN if unregistered
        position = dat['position']  # (n_days, 2, n_frames), cm
        envs = [e[0] for e in dat['envs']]
        blocked = dat['blocked']
        n_days = trace.shape[0]

        for day in range(n_days):
            tr = trace[day]
            # keep only cells registered on this day (unregistered cells are all-NaN)
            registered = ~np.isnan(tr).any(axis=1)
            # drop near-silent cells (<= 5 transients in the session), as in the
            # paper's within-session decoding analysis
            events = np.nansum(tr, axis=1)
            keep = registered & (events > EVENT_THRESHOLD)
            n_cells_total += registered.sum()
            n_dropped_silent += int(registered.sum() - keep.sum())
            tr = tr[keep].astype(np.float32)

            # temporal smoothing + binning of neural data (as in utils.fit_decoder)
            tr = gaussian_filter1d(tr, sigma=SMOOTH_SIGMA, axis=1)
            neural = bin_time(tr, TEMPORAL_BIN_FRAMES).astype(np.float32)

            # position: average within the same time bins, then spatially discretize
            pos = bin_time(position[day].astype(np.float64), TEMPORAL_BIN_FRAMES)
            bin_size_cm = ARENA_SIZE / N_SPATIAL_BINS
            xb = np.clip((pos[0] // bin_size_cm).astype(int), 0, N_SPATIAL_BINS - 1)
            yb = np.clip((pos[1] // bin_size_cm).astype(int), 0, N_SPATIAL_BINS - 1)
            pos_bin = (yb * N_SPATIAL_BINS + xb).astype(np.int64)

            geom = geometry_vector(blocked[day])

            n_bins = min(neural.shape[1], pos_bin.shape[0])
            n_trials = n_bins // BINS_PER_TRIAL
            if n_trials < 2:
                print(f'  skipping {animal} day {day}: too short')
                continue

            neural_trials, input_trials, output_trials = [], [], []
            for t in range(n_trials):
                sl = slice(t * BINS_PER_TRIAL, (t + 1) * BINS_PER_TRIAL)
                neural_trials.append(np.ascontiguousarray(neural[:, sl]))
                input_trials.append(geom.copy())
                output_trials.append(pos_bin[sl][None, :].copy())

            neural_all.append(neural_trials)
            input_all.append(input_trials)
            output_all.append(output_trials)
            subject_idx.append(ai)
            brain_region_idx.append(np.zeros(neural.shape[0], dtype=np.int64))
            session_info.append({'animal': animal, 'day': int(day),
                                 'environment': str(envs[day]),
                                 'n_neurons': int(neural.shape[0]),
                                 'n_trials': n_trials})
        del dat, trace, position

    print(f'Sessions: {len(neural_all)}, cells (registered): {n_cells_total}, '
          f'dropped near-silent: {n_dropped_silent}')

    output_values = []
    for idx in range(N_SPATIAL_BINS * N_SPATIAL_BINS):
        x = idx % N_SPATIAL_BINS
        y = idx // N_SPATIAL_BINS
        output_values.append(f'x{x}_y{y}')

    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,
        'subjects': ANIMALS,
        'subject_idx': np.array(subject_idx, dtype=np.int64),
        'brain_regions': ['CA1'],
        'brain_region_idx': brain_region_idx,
        'input_names': [f'blocked_x{ i % N_SPATIAL_BINS }_y{ i // N_SPATIAL_BINS }'
                        for i in range(N_SPATIAL_BINS * N_SPATIAL_BINS)],
        'output_names': ['position_bin'],
        'output_values': [output_values],
        'metadata': {
            'task_description': (
                'Mice freely foraged for 40 min/day in a 75x75 cm arena partitioned '
                'into a 3x3 grid, whose geometry was deformed across days by blocking '
                'subsets of the 9 partitions (10 distinct geometries, sequence repeated '
                'up to 3 times). CA1 populations were recorded with miniscope calcium '
                'imaging. Decoder inputs the neural activity plus the environment '
                'geometry (which of the 9 partitions are blocked, static within a trial) '
                'and predicts the animal position discretized into the 9 spatial bins.'),
            'time_bin_size': BIN_MS,
            'temporal_alignment_event': (
                'start of the recording session; each session is cut into consecutive '
                'non-overlapping 1-minute trials'),
            'off_start': 0.0,
            'off_end': TRIAL_SECONDS,
            'neural_data_type': (
                'binarized rising phase of calcium transients (paper firing rate), '
                'gaussian-smoothed (sigma=3 frames) and averaged in 100 ms bins'),
            'spatial_bin_size_cm': ARENA_SIZE / N_SPATIAL_BINS,
            'position_bin_convention': 'bin index = ybin*3 + xbin, same as the blocked field',
            'input_convention': '1 = partition blocked (inaccessible), 0 = accessible',
            'recording_rate_hz': FPS,
            'session_info': session_info,
            'source': ('Lee, Keinath, Cianfarano & Brandon (2025) Neuron 113:307-320; '
                       'data from Zenodo 10.5281/zenodo.13993254'),
        },
    }

    with open(OUT_FILE, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print('saved', OUT_FILE)


if __name__ == '__main__':
    main()
