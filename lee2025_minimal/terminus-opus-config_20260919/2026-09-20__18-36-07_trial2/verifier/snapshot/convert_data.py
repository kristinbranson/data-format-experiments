"""
Convert the CA1 geometric-deformation dataset of Lee et al. (2025, Neuron),
"Identifying representational structure in CA1 to benchmark theoretical models of
cognitive mapping", into the standard decoding format.

Decoder task: predict the mouse's position (discretized into the 3 x 3 grid of
arena partitions) from CA1 calcium-event activity, given the environment geometry
(which of the 9 partitions are blocked) as a static per-trial input.

Processing follows the original paper / repository (georepca1/src/utils.py):
  * data loaded from the authors' joblib files (identical content to the .mat files)
  * one session per recording day (40 min, 30 Hz), all 7 mice / 207 sessions kept
  * only cells registered on a given day are kept for that session (others are NaN)
  * running-speed filter: speed smoothed with a gaussian (sigma = 5 frames) and
    thresholded at 5 cm/s (utils.decode_position_within, v_filt_size=5, v_thresh=5)
  * cells with <= 5 events during running in that session are excluded
    (utils.decode_position_within, cell_threshold=5)
  * traces smoothed with a gaussian (sigma = 3 frames) and average-pooled over
    3 frames -> 100 ms time bins (utils.fit_decoder, temporal_bin_size=3)
  * position average-pooled over the same 3-frame bins and discretized
  * long sessions are split into consecutive 1-minute trials (600 bins)
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

FPS = 30.0            # acquisition rate of miniscope / behaviour (Hz)
TEMPORAL_BIN = 3      # frames per time bin -> 100 ms (as in utils.fit_decoder)
TRIAL_SEC = 60.0      # trial length in seconds
BINS_PER_TRIAL = int(round(TRIAL_SEC * FPS / TEMPORAL_BIN))  # 600 bins = 1 min
MIN_BINS_PER_TRIAL = 30   # >= 3 s of running data required to keep a trial
V_FILT_SIGMA = 5      # gaussian sigma (frames) for speed estimate
V_THRESH = 5.0        # cm/s, running threshold
CELL_THRESH = 5       # minimum number of events during running per session
ARENA_SIZE = 75.0     # cm (75 x 75 cm square)
N_SPACE_BINS = 3      # 3 x 3 partitions

# partition index = 3 * y_bin + x_bin, layout [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
PART_NAMES = [f'x{i % 3}y{i // 3}' for i in range(9)]


def get_env_mat(env):
    """Binary 3x3 matrix of open (1) / blocked (0) partitions, copied verbatim from
    utils.get_env_mat of the paper's repository. Rows there run from the top (north)
    of the arena downwards, i.e. this matrix is the vertical flip of the dataset's
    'blocked' partition indexing (3 * y_bin + x_bin); used here only as a
    cross-check of the 'blocked' field."""
    m = {
        'square':    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        'o':         [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
        't':         [[0, 1, 0], [0, 1, 0], [1, 1, 1]],
        'u':         [[1, 1, 1], [1, 0, 0], [1, 1, 1]],
        'rectangle': [[0, 1, 1], [0, 1, 1], [0, 1, 1]],
        '+':         [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        'i':         [[1, 1, 1], [0, 1, 0], [1, 1, 1]],
        'l':         [[1, 1, 1], [1, 0, 0], [1, 0, 0]],
        'bit donut': [[1, 1, 1], [1, 0, 1], [0, 1, 1]],
        'glenn':     [[1, 1, 0], [1, 1, 1], [0, 1, 1]],
    }
    return np.array(m[str(env)], dtype=float)


def blocked_vector(env_name, blocked_field):
    """9-d binary vector, 1 = partition blocked.

    Taken from the dataset's own 'blocked' field, whose partition order is
    [[0, 1, 2], [3, 4, 5], [6, 7, 8]] (README of the paper's repository), i.e.
    partition index = 3 * y_bin + x_bin (verified against the animals' occupancy:
    the animal is never found in a blocked partition). This is the vertical flip
    of utils.get_env_mat, which draws the same geometries with the first row at
    the top; the check below confirms the two describe identical geometries.
    """
    b = np.atleast_1d(np.asarray(blocked_field, dtype=float).ravel())
    vec = np.zeros(9, dtype=np.float32)
    if not (b.size == 1 and b[0] == -1):
        vec[b.astype(int)] = 1.0
    from_mat = (1.0 - np.flipud(get_env_mat(env_name)).ravel()).astype(np.float32)
    assert np.array_equal(vec, from_mat), f'blocked mismatch for env {env_name}'
    return vec


def pool_mean(x, k):
    """Non-overlapping average pooling over the last axis (drops the remainder)."""
    n = (x.shape[-1] // k) * k
    return x[..., :n].reshape(x.shape[:-1] + (n // k, k)).mean(axis=-1)


def position_to_bin(pos_xy):
    """Discretize x-y position (cm) into the 3x3 partition index.

    Partition index = 3 * y_bin + x_bin, matching the convention of the dataset's
    'blocked' field ([[0, 1, 2], [3, 4, 5], [6, 7, 8]]), verified against the
    animals' occupancy in each geometry.
    """
    edge = ARENA_SIZE / N_SPACE_BINS
    xb = np.clip(np.floor(pos_xy[0] / edge), 0, N_SPACE_BINS - 1).astype(np.int64)
    yb = np.clip(np.floor(pos_xy[1] / edge), 0, N_SPACE_BINS - 1).astype(np.int64)
    return 3 * yb + xb


def main():
    neural_all, input_all, output_all = [], [], []
    subject_idx, brain_region_idx, session_info = [], [], []

    for ai, animal in enumerate(ANIMALS):
        print(f'--- {animal}')
        dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
        traces, positions = dat['trace'], dat['position']
        envs = [str(e[0]) for e in dat['envs']]
        blocked = dat['blocked']
        n_days = traces.shape[0]

        for day in range(n_days):
            env = envs[day]
            inp = blocked_vector(env, blocked[day])

            pos = np.asarray(positions[day], dtype=np.float64)     # (2, T)
            tr = np.asarray(traces[day])                           # (n_cells, T)

            # keep only cells registered (tracked) on this day
            registered = ~np.isnan(tr[:, 0])
            tr = tr[registered].astype(np.float32)

            # ---- running speed (cm/s), smoothed as in decode_position_within
            speed = np.zeros(pos.shape[1])
            speed[1:] = np.linalg.norm(np.diff(pos, axis=1), axis=0) * FPS
            speed = gaussian_filter1d(speed, sigma=V_FILT_SIGMA)
            running_frames = speed > V_THRESH

            # ---- exclude poorly active cells (>5 events while running)
            active = tr[:, running_frames].sum(axis=1) > CELL_THRESH
            tr = tr[active]
            if tr.shape[0] == 0:
                print(f'  day {day} ({env}): no cells pass criteria, skipped')
                continue

            # ---- temporal smoothing + 100 ms binning (as in fit_decoder)
            tr_s = gaussian_filter1d(tr, sigma=TEMPORAL_BIN, axis=1)
            neural = pool_mean(tr_s, TEMPORAL_BIN).astype(np.float32)   # (n_cells, nbins)
            pos_b = pool_mean(pos, TEMPORAL_BIN)                        # (2, nbins)
            speed_b = pool_mean(speed[np.newaxis], TEMPORAL_BIN)[0]     # (nbins,)
            run_b = speed_b > V_THRESH

            out_bins = position_to_bin(pos_b)[np.newaxis, :]            # (1, nbins)
            nbins = neural.shape[1]

            # sanity check: the animal should essentially never be in a blocked partition
            occ = np.bincount(out_bins[0], minlength=9) / nbins
            assert occ[inp.astype(bool)].sum() < 0.02, (
                f'{animal} day {day} ({env}): occupancy in blocked partitions '
                f'{occ[inp.astype(bool)].sum():.3f}')

            # ---- split into consecutive 1-minute trials, keep running bins only
            sess_neural, sess_input, sess_output = [], [], []
            for start in range(0, nbins, BINS_PER_TRIAL):
                sl = slice(start, min(start + BINS_PER_TRIAL, nbins))
                keep = run_b[sl]
                if keep.sum() < MIN_BINS_PER_TRIAL:
                    continue
                sess_neural.append(np.ascontiguousarray(neural[:, sl][:, keep]))
                sess_output.append(np.ascontiguousarray(out_bins[:, sl][:, keep]))
                sess_input.append(inp.copy())
            if len(sess_neural) < 2:
                print(f'  day {day} ({env}): < 2 usable trials, skipped')
                continue

            neural_all.append(sess_neural)
            input_all.append(sess_input)
            output_all.append(sess_output)
            subject_idx.append(ai)
            brain_region_idx.append(np.zeros(sess_neural[0].shape[0], dtype=np.int64))
            session_info.append({'subject': animal, 'day': int(day), 'environment': env,
                                 'blocked_partitions': np.nonzero(inp)[0].tolist(),
                                 'n_neurons': int(sess_neural[0].shape[0]),
                                 'n_trials': len(sess_neural)})
            print(f'  day {day:2d} ({env:9s}): {sess_neural[0].shape[0]} cells, '
                  f'{len(sess_neural)} trials, '
                  f'{sum(t.shape[1] for t in sess_neural)} bins')
        del dat, traces, positions

    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,
        'subjects': ANIMALS,
        'subject_idx': np.array(subject_idx, dtype=np.int64),
        'brain_regions': ['CA1'],
        'brain_region_idx': brain_region_idx,
        'input_names': [f'blocked_{i}_{PART_NAMES[i]}' for i in range(9)],
        'output_names': ['position_bin'],
        'output_values': [[f'{i}_{PART_NAMES[i]}' for i in range(9)]],
        'metadata': {
            'task_description': (
                'Mice freely foraged for 40 min/day in a 75 x 75 cm arena partitioned '
                'into a 3 x 3 grid, whose geometry was changed across days by blocking '
                'partitions (10 geometries, repeated up to 3 times per mouse). CA1 '
                'populations were recorded with miniscope calcium imaging (Lee et al., '
                '2025, Neuron). The decoder predicts which of the 9 spatial partitions '
                'the mouse occupies from CA1 activity, given the environment geometry '
                '(which partitions are blocked) as a static input.'),
            'time_bin_size': 100.0,
            'temporal_alignment_event': (
                'start of the recording session; each session is cut into consecutive '
                '1-minute trials'),
            'off_start': 0.0,
            'off_end': 60.0,
            'neural_data_type': (
                'binarized calcium-transient rising phases (events), gaussian smoothed '
                '(sigma = 3 frames) and averaged within 100 ms bins (mean event rate)'),
            'sampling_rate_original_hz': FPS,
            'arena_size_cm': ARENA_SIZE,
            'spatial_bin_size_cm': ARENA_SIZE / N_SPACE_BINS,
            'partition_layout': 'partition index = 3 * y_bin + x_bin, i.e. [[0, 1, 2], [3, 4, 5], [6, 7, 8]] as in the datasets blocked field',
            'speed_filter': (
                f'time bins with running speed <= {V_THRESH} cm/s excluded (speed '
                f'smoothed with gaussian sigma = {V_FILT_SIGMA} frames), as in the '
                'papers within-session position decoding'),
            'neuron_curation': (
                'per session: only cells registered (tracked) on that day and with more '
                f'than {CELL_THRESH} events during running were kept'),
            'trial_curation': (
                f'trials with fewer than {MIN_BINS_PER_TRIAL} running time bins and '
                'sessions with fewer than 2 usable trials were dropped'),
            'session_info': session_info,
            'source': 'Lee, Keinath, Cianfarano & Brandon (2025) Neuron 113:307-320; '
                      'Zenodo 10.5281/zenodo.13993254',
        },
    }

    n_sessions = len(neural_all)
    n_trials = sum(len(s) for s in neural_all)
    n_bins = sum(t.shape[1] for s in neural_all for t in s)
    print(f'\nSessions: {n_sessions}, trials: {n_trials}, time bins: {n_bins}')

    with open(OUT_FILE, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print(f'Saved {OUT_FILE}')


if __name__ == '__main__':
    main()
