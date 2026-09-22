"""
Convert the Lee et al. (2025) CA1 geometric deformation dataset
("Identifying representational structure in CA1 to benchmark theoretical models
of cognitive mapping", Neuron 113:307-320) into the standard decoding format.

Decoder task: decode the animal's position (discretized into the 3 x 3 grid of
partitions that defines the experimental geometries) from CA1 population
activity, given the environment geometry (which partitions are blocked) as an
additional input.

Data source (per animal joblib file, as described in the paper's repo README):
  'trace'    : (n_days, n_cells, n_frames) binarized rising phase of calcium
               transients at 30 Hz ('1' = significant event). Cells that were
               not registered on a given day are NaN for that day.
  'position' : (n_days, 2, n_frames) x-y position (cm) from DeepLabCut tracking,
               simultaneously acquired with the imaging at 30 Hz.
  'envs'     : (n_days, 1) name of the geometry recorded on each day.
  'blocked'  : list (n_days) of the indices of the blocked (occluded) partitions
               in the 3 x 3 grid, indexed [[0, 1, 2], [3, 4, 5], [6, 7, 8]];
               -1 if nothing is blocked (full square).

Processing decisions (see comments below for justification):
  * one session = one recording day (40 min of free exploration); all 207
    sessions from all 7 animals are used.
  * neural data = the binary transient-rising-phase vector used as "firing rate"
    throughout the paper, smoothed and temporally binned exactly as in the
    paper's own position-decoding code (utils.fit_decoder / utils.test_decoder):
    gaussian_filter1d(sigma = 3 frames) followed by average pooling over 3
    frames -> 100 ms time bins.
  * neurons: only cells registered on that day (non-NaN) and with more than 5
    transients in the session are kept (activity-sparsity criterion of
    utils.decode_position_within, cell_threshold=5).
  * output = position discretized into the 3 x 3 partition grid (25 cm bins of
    the 75 x 75 cm arena), labelled with the same indexing as the 'blocked'
    field, i.e. label = 3 * y_bin + x_bin.
  * input = the static (per-trial) environment geometry: 9 binary values, 1 if
    that partition is blocked in the current geometry.
  * trials = consecutive, non-overlapping 1 min segments of each session (600
    bins of 100 ms); the trailing incomplete segment is dropped.
  * the paper's within-session Bayesian decoding additionally discards frames in
    which the animal moves slower than 5 cm/s. That filter is NOT applied here:
    it would remove ~50% of the frames scattered through the recording and make
    it impossible to cut the session into contiguous 1-min trials with a common
    time base for neural, input and output streams (as required here). The
    animal still occupies a well defined partition while immobile, so every time
    bin carries a valid position label. Correspondingly the cell activity
    criterion (> 5 transients) is evaluated over the whole session rather than
    over the moving frames only.
"""

import os
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d
import joblib

DATA_DIR = '/app/data'
OUT_FILE = '/app/converted_data.pkl'

ANIMALS = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51',
           'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']

FPS = 30.0              # acquisition rate of both imaging and behavior streams
TEMPORAL_BIN = 3        # frames per time bin (paper's decoder: temporal_bin_size=3)
BIN_MS = 1000.0 * TEMPORAL_BIN / FPS   # = 100 ms
TRIAL_SEC = 60.0        # 1 minute trials
TRIAL_BINS = int(round(TRIAL_SEC * 1000.0 / BIN_MS))   # 600 bins per trial
ARENA_SIZE = 75.0       # cm, full square environment
NGRID = 3               # 3 x 3 partitions
PART_SIZE = ARENA_SIZE / NGRID   # 25 cm
CELL_EVENT_THRESHOLD = 5   # cell_threshold in utils.decode_position_within


def pool_mean(x, k):
    """Average pool along the last axis with kernel = stride = k (drops remainder)."""
    n = (x.shape[-1] // k) * k
    return x[..., :n].reshape(x.shape[:-1] + (n // k, k)).mean(axis=-1)


def partition_labels(pos_binned_xy, open_mask):
    """Convert continuous x-y position (cm, shape (T, 2)) to 3x3 partition labels.

    Labels follow the indexing of the dataset's 'blocked' field:
    [[0, 1, 2], [3, 4, 5], [6, 7, 8]] with label = 3 * y_bin + x_bin (verified
    against the occupancy of the blocked partitions in every session).
    Rare samples that fall inside a blocked partition (tracking noise near the
    partition walls, <1% of samples in a single session) are snapped to the
    nearest accessible partition, as the paper's decoding code snaps actual and
    predicted positions to the visitable bins of the rate map.
    """
    xb = np.clip(np.floor(pos_binned_xy[:, 0] / PART_SIZE).astype(int), 0, NGRID - 1)
    yb = np.clip(np.floor(pos_binned_xy[:, 1] / PART_SIZE).astype(int), 0, NGRID - 1)
    labels = NGRID * yb + xb
    bad = ~open_mask[labels]
    if np.any(bad):
        open_idx = np.where(open_mask)[0]
        centers = np.stack([(open_idx % NGRID + 0.5) * PART_SIZE,
                            (open_idx // NGRID + 0.5) * PART_SIZE], axis=1)
        d = np.linalg.norm(pos_binned_xy[bad][:, None, :] - centers[None], axis=2)
        labels[bad] = open_idx[np.argmin(d, axis=1)]
    return labels


def convert(animals=ANIMALS, out_file=OUT_FILE):
    data = {
        'neural': [], 'input': [], 'output': [],
        'subjects': list(animals), 'subject_idx': [],
        'brain_regions': ['CA1'], 'brain_region_idx': [],
        'input_names': [f'blocked_partition_{i}' for i in range(NGRID * NGRID)],
        'output_names': ['position_bin'],
        'output_values': [[f'partition_{i}' for i in range(NGRID * NGRID)]],
        'metadata': {},
    }
    session_info = []

    for ai, animal in enumerate(animals):
        print(f'loading {animal} ...', flush=True)
        dat = joblib.load(os.path.join(DATA_DIR, animal))[animal]
        pos_all, trace_all = dat['position'], dat['trace']
        envs = [str(e[0]) for e in dat['envs']]
        n_days = trace_all.shape[0]
        for day in range(n_days):
            env = envs[day]
            blocked = np.array(dat['blocked'][day]).ravel().astype(int)
            blocked = blocked[blocked >= 0]
            open_mask = np.ones(NGRID * NGRID, dtype=bool)
            open_mask[blocked] = False

            pos = pos_all[day].T.astype(np.float64)          # (n_frames, 2), cm
            trace = trace_all[day]                            # (n_cells, n_frames)

            # keep cells registered on this day with > 5 transients in the session
            registered = ~np.all(np.isnan(trace), axis=1)
            cell_ids = np.where(registered)[0]
            tr = trace[cell_ids]
            enough = np.nansum(tr, axis=1) > CELL_EVENT_THRESHOLD
            cell_ids, tr = cell_ids[enough], tr[enough]
            if tr.shape[0] == 0:
                print(f'  skipping {animal} day {day}: no cells pass criteria')
                continue

            # temporal processing identical to the paper's decoder:
            # gaussian smoothing (sigma = 3 frames) then 3-frame average pooling
            tr_s = gaussian_filter1d(tr.astype(np.float32), sigma=TEMPORAL_BIN, axis=1)
            neural = pool_mean(tr_s, TEMPORAL_BIN).astype(np.float32)   # (n_cells, n_bins)
            pos_b = pool_mean(pos.T, TEMPORAL_BIN).T                    # (n_bins, 2)
            labels = partition_labels(pos_b, open_mask).astype(np.int64)

            n_bins = min(neural.shape[1], labels.shape[0])
            n_trials = n_bins // TRIAL_BINS
            if n_trials < 2:
                print(f'  skipping {animal} day {day}: < 2 full trials')
                continue

            geom = np.zeros(NGRID * NGRID, dtype=np.float32)
            geom[blocked] = 1.0

            neural_trials, input_trials, output_trials = [], [], []
            for t in range(n_trials):
                sl = slice(t * TRIAL_BINS, (t + 1) * TRIAL_BINS)
                neural_trials.append(np.ascontiguousarray(neural[:, sl]))
                input_trials.append(geom.copy())
                output_trials.append(labels[sl][None, :].copy())

            data['neural'].append(neural_trials)
            data['input'].append(input_trials)
            data['output'].append(output_trials)
            data['subject_idx'].append(ai)
            data['brain_region_idx'].append(np.zeros(len(cell_ids), dtype=np.int64))
            session_info.append({
                'subject': animal, 'day': int(day), 'environment': env,
                'blocked_partitions': blocked.tolist(),
                'n_neurons': int(len(cell_ids)), 'n_trials': int(n_trials),
                'cell_ids': [int(c) for c in cell_ids],
            })
            print(f'  {animal} day {day:2d} env {env:10s} cells {len(cell_ids):4d} '
                  f'trials {n_trials}', flush=True)
        del dat, pos_all, trace_all

    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
    data['metadata'] = {
        'task_description': (
            'Mice freely explore a 75 x 75 cm arena partitioned into a 3 x 3 grid, '
            'whose geometry is changed across days by occluding partitions (10 '
            'geometries). CA1 population activity was recorded with miniscope '
            'calcium imaging at 30 Hz while position was tracked with DeepLabCut. '
            'The decoder receives the population activity and the static geometry '
            'of the environment (which of the 9 partitions are blocked) and must '
            'predict which of the 9 partitions the animal occupies at each time bin.'),
        'time_bin_size': BIN_MS,
        'temporal_alignment_event': (
            'start of each 1-min trial, i.e. arbitrary segmentation of the '
            'continuous 40-min free-exploration session (no task events)'),
        'off_start': 0.0,
        'off_end': TRIAL_SEC,
        'session_info': session_info,
        'neural_data_description': (
            'Binarized rising phase of calcium transients (the "firing rate" used '
            'in the paper), smoothed with a gaussian kernel (sigma = 3 frames) and '
            'average-pooled over 3 frames (100 ms bins), as in the paper\'s '
            'position-decoding code (utils.fit_decoder / utils.test_decoder). '
            'Only cells registered on the session with > 5 transients are kept.'),
        'input_description': (
            'Static per-trial 9-dim binary vector of the environment geometry: 1 if '
            'the corresponding partition of the 3 x 3 grid is blocked (occluded), '
            '0 if accessible. Indexing [[0, 1, 2], [3, 4, 5], [6, 7, 8]] as in the '
            'dataset\'s "blocked" field.'),
        'output_description': (
            'Partition of the 3 x 3 grid occupied by the animal (25 cm spatial '
            'bins), label = 3 * y_bin + x_bin, same indexing as the geometry input. '
            'Samples falling in a blocked partition (tracking noise) are snapped to '
            'the nearest accessible partition.'),
        'velocity_filter': (
            'none: unlike the paper\'s Bayesian decoding analysis (which kept only '
            'frames with speed > 5 cm/s), all time bins are kept so that each '
            'session can be split into contiguous 1-min trials with a common time '
            'base across neural, input and output streams.'),
        'sampling_rate_original': FPS,
        'session_duration_min': 40,
        'reference': ('Lee, Keinath, Cianfarano, Brandon (2025) Identifying '
                      'representational structure in CA1 to benchmark theoretical '
                      'models of cognitive mapping. Neuron 113:307-320'),
    }

    print(f'saving to {out_file} ...', flush=True)
    with open(out_file, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    print('done')
    return data


if __name__ == '__main__':
    convert()
