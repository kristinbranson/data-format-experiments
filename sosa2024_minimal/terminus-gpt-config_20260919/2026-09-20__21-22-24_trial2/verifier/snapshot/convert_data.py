#!/usr/bin/env python3
"""Convert the Sosa et al. NWB release to decoder-ready trial data.

Decisions follow the deposited processing: Suite2p deconvolved activity is used
at its native imaging rate, and only ROIs marked as cells by Suite2p are kept.
Behavior in each NWB is already resampled to the imaging frames. Trials are
bounded by the explicit trial_start and teleport streams rather than the trial
number stream (the latter changes during ITIs in a few files). All deposited
behavior+ophys sessions are included.
"""
import glob, os, pickle
import h5py
import numpy as np

ROOT = '/app/data'
OUT = '/app/converted_data.pkl'
RATE = 15.5078125
DT_MS = 1000.0 / RATE
ZONE_STARTS = np.array([80.0, 200.0, 320.0], dtype=np.float32)
ZONE_WIDTH = 50.0
B = 'processing/behavior/BehavioralTimeSeries/'


def nearest_fill(labels):
    """Fill missing trial zone labels from the nearest observed trial.

    Zone-entry is absent on some omission/aborted traversals. Nearest filling
    retains the experimentally abrupt reward-location switches.
    """
    labels = labels.copy()
    known = np.flatnonzero(labels >= 0)
    if not len(known):
        raise ValueError('session has no reward-zone entries')
    missing = np.flatnonzero(labels < 0)
    for i in missing:
        labels[i] = labels[known[np.argmin(np.abs(known - i))]]
    return labels


def trial_pairs(start, end):
    """Pair each start with the first subsequent end before the next start."""
    pairs = []
    j = 0
    for k, a in enumerate(start):
        while j < len(end) and end[j] <= a:
            j += 1
        if j == len(end):
            break
        b = int(end[j])
        next_a = int(start[k + 1]) if k + 1 < len(start) else np.iinfo(np.int64).max
        if b < next_a and b > a:
            # teleport is the first ITI frame, hence use it as exclusive end
            pairs.append((int(a), b))
            j += 1
    return pairs


def convert_session(path):
    with h5py.File(path, 'r') as f:
        def beh(name):
            return f[B + name + '/data'][:]

        ts = f[B + 'position/timestamps'][:]
        position = beh('position')
        speed = beh('speed')
        lick = beh('lick')
        environment = beh('environment')
        zone_event = beh('reward_zone')
        starts = np.flatnonzero(beh('trial_start') > 0)
        ends = np.flatnonzero(beh('teleport') > 0)
        pairs = trial_pairs(starts, ends)
        if len(pairs) < 2:
            raise ValueError(f'fewer than two complete trials: {path}')

        # Infer A/B/C from entry position. The task's three 50-cm zones begin
        # at 80, 200 and 320 cm. The event can be sampled after boundary entry,
        # so classify by nearest nominal start, then fill omission trials.
        zones = np.full(len(pairs), -1, dtype=np.int8)
        for i, (a, b) in enumerate(pairs):
            p = position[a:b][zone_event[a:b] > 0]
            p = p[np.isfinite(p) & (p >= 0) & (p <= 450)]
            if len(p):
                zones[i] = int(np.argmin(np.abs(ZONE_STARTS - np.median(p))))
        zones = nearest_fill(zones)

        reward_times = f[B + 'Reward/timestamps'][:]
        rewarded = np.array([
            np.any((reward_times >= ts[a]) & (reward_times < ts[b]))
            for a, b in pairs
        ], dtype=np.int8)

        iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:, 0] == 1
        dset = f['processing/ophys/Deconvolved/plane0/data']
        # NWB series columns are a DynamicTableRegion into PlaneSegmentation.
        # Most files contain all rows in order; a few contain a subset, whose
        # row indices are deposited in the companion Fluorescence `rois` set.
        rois_path = 'processing/ophys/Deconvolved/plane0/rois'
        if rois_path in f:
            roi_rows = np.asarray(f[rois_path][:], dtype=int)
        else:
            roi_rows = np.asarray(f['processing/ophys/Fluorescence/plane0/rois'][:], dtype=int)
        if len(roi_rows) != dset.shape[1]:
            raise ValueError(f'ROI mapping length {len(roi_rows)} != data columns {dset.shape[1]}: {path}')
        cell_idx = np.flatnonzero(iscell[roi_rows])
        deconv = dset[:, cell_idx]

        neural_trials, input_trials, output_trials = [], [], []
        prev = 0
        for i, (a, b) in enumerate(pairs):
            n = b - a
            neural_trials.append(np.asarray(deconv[a:b, :], dtype=np.float32).T.copy())

            # Environment is constant on ordinary sessions and changes on the
            # ENV1->ENV2 switch session. Median within each trial avoids ITI -1.
            ev = environment[a:b]
            ev = ev[ev >= 0]
            env = int(np.rint(np.median(ev))) if len(ev) else 0
            inp = np.empty((4, n), dtype=np.float32)
            inp[0] = np.arange(n, dtype=np.float32) / RATE
            inp[1] = env
            inp[2] = i
            inp[3] = prev
            input_trials.append(inp)

            pos = position[a:b]
            spd = speed[a:b]
            z0 = float(ZONE_STARTS[zones[i]])
            z1 = z0 + ZONE_WIDTH
            dist = np.where(pos < z0, pos - z0, np.where(pos > z1, pos - z1, 0.0))

            out = np.empty((6, n), dtype=np.int8)
            # np.digitize with right=False gives exact requested edge behavior.
            out[0] = np.digitize(dist, [-50, -10, 0, np.nextafter(0.0, 1.0), 10, 50])
            out[1] = np.digitize(pos, [90, 180, 270, 360])
            out[2] = np.digitize(spd, [2, 10, 20, 40])
            out[3] = (lick[a:b] > 0).astype(np.int8)
            out[4] = zones[i]
            out[5] = rewarded[i]
            output_trials.append(out)
            prev = int(rewarded[i])

    return neural_trials, input_trials, output_trials, len(cell_idx), len(pairs)


def main():
    files = sorted(glob.glob(os.path.join(ROOT, 'sub-*', '*_behavior+ophys.nwb')))
    subjects = sorted({os.path.basename(os.path.dirname(p)).replace('sub-', '') for p in files},
                      key=lambda x: int(x[1:]) if x.startswith('m') and x[1:].isdigit() else x)
    submap = {s: i for i, s in enumerate(subjects)}
    data = {
        'neural': [], 'input': [], 'output': [],
        'subjects': subjects, 'subject_idx': [],
        'brain_regions': ['dorsal CA1'], 'brain_region_idx': [],
        'input_names': ['time from trial start (s)', 'environment type',
                        'trial number', 'previous trial outcome'],
        'output_names': ['distance to reward zone', 'absolute position', 'speed',
                         'lick', 'reward zone location', 'reward outcome'],
        'output_values': [
            ['< -50 cm', '-50 to -10 cm', '-10 to <0 cm', '0 cm',
             '>0 to +10 cm', '+10 to +50 cm', '> +50 cm'],
            ['<90 cm', '90-180 cm', '180-270 cm', '270-360 cm', '>360 cm'],
            ['<2 cm/s', '2-10 cm/s', '10-20 cm/s', '20-40 cm/s', '>40 cm/s'],
            ['no', 'yes'], ['A', 'B', 'C'], ['no', 'yes']
        ],
        'metadata': {
            'task_description': ('Head-fixed mice traverse a 450-cm virtual linear corridor; '
                'neural activity predicts position, speed, licking, reward-zone identity and outcome.'),
            'time_bin_size': DT_MS,
            'temporal_alignment_event': 'entry to linear track (trial_start)',
            'off_start': 0.0,
            'off_end': None,
            'neural_signal': 'Suite2p deconvolved calcium activity for iscell ROIs',
            'trial_interval': 'trial_start inclusive to teleport exclusive',
            'reward_zone_bounds_cm': {'A': [80, 130], 'B': [200, 250], 'C': [320, 370]},
            'source': 'Sosa et al. 2025 DANDI 001361'
        }
    }
    session_info = []
    for k, path in enumerate(files):
        neu, inp, out, nc, nt = convert_session(path)
        subj = os.path.basename(os.path.dirname(path)).replace('sub-', '')
        data['neural'].append(neu); data['input'].append(inp); data['output'].append(out)
        data['subject_idx'].append(submap[subj])
        data['brain_region_idx'].append(np.zeros(nc, dtype=np.int8))
        session_info.append({'file': os.path.basename(path), 'subject': subj,
                             'n_neurons': nc, 'n_trials': nt})
        print(f'[{k+1:3d}/{len(files)}] {os.path.basename(path)}: {nc} cells, {nt} trials', flush=True)
    data['subject_idx'] = np.asarray(data['subject_idx'], dtype=np.int16)
    data['metadata']['session_info'] = session_info
    with open(OUT, 'wb') as fh:
        pickle.dump(data, fh, protocol=4)
    print('saved', OUT)

if __name__ == '__main__':
    main()
