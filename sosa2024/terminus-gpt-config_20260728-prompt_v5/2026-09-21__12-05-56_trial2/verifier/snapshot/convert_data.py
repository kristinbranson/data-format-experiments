#!/usr/bin/env python3
import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np
from pynwb import NWBHDF5IO

CANONICAL_ZONE_CENTERS = np.array([85.0, 205.0, 325.0], dtype=np.float32)  # A, B, C


def load_nwb_session(path):
    with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
        nwb = io.read()
        beh = nwb.processing['behavior']['BehavioralTimeSeries'].time_series
        ophys = nwb.processing['ophys']
        seg = list(ophys['ImageSegmentation'].plane_segmentations.values())[0]
        iscell = np.asarray(seg['iscell'].data[:])
        if iscell.ndim > 1:
            # suite2p-style arrays may store two columns; use first column as confidence if present
            iscell = iscell[:, 0]
        if iscell.dtype.fields is not None:
            first_field = list(iscell.dtype.fields)[0]
            iscell = iscell[first_field]
        iscell = np.asarray(iscell, dtype=np.float32).reshape(-1)
        plane_idx = np.asarray(seg['planeIdx'].data[:], dtype=np.int32)
        deconv = next(iter(ophys['Deconvolved'].roi_response_series.values()))
        neural = np.asarray(deconv.data[:], dtype=np.float32)  # time x roi
        roi_idx = np.asarray(deconv.rois.data[:], dtype=np.int64) if deconv.rois is not None else np.arange(neural.shape[1], dtype=np.int64)
        iscell = iscell[roi_idx]
        plane_idx = plane_idx[roi_idx]
        timestamps = np.asarray(deconv.timestamps[:] if deconv.timestamps is not None else np.arange(neural.shape[0]) / float(deconv.rate), dtype=np.float64)
        rate = float(np.median(1.0 / np.diff(timestamps))) if len(timestamps) > 1 else float(getattr(deconv, 'rate', np.nan))
        out = {
            'subject': path.parent.name,
            'session': path.stem,
            'timestamps': timestamps,
            'rate': rate,
            'neural': neural,
            'iscell': iscell,
            'plane_idx': plane_idx,
        }
        for key in ['Reward', 'autoreward', 'environment', 'lick', 'position', 'reward_zone', 'scanning', 'speed', 'teleport', 'trial number', 'trial_start']:
            ts = beh[key]
            out[key] = np.asarray(ts.data[:])
            out[key + '_t'] = np.asarray(ts.timestamps[:]) if ts.timestamps is not None else timestamps.copy()
        return out


def infer_zone_center(pos_trial, reward_zone_trial):
    m = reward_zone_trial > 0
    if np.any(m):
        center = float(np.nanmean(pos_trial[m]))
    else:
        # fallback: nearest canonical center to trial position mode in plausible corridor region
        valid = pos_trial[(pos_trial >= 0) & (pos_trial <= 450)]
        center = float(np.nanmedian(valid)) if len(valid) else 205.0
        center = float(CANONICAL_ZONE_CENTERS[np.argmin(np.abs(CANONICAL_ZONE_CENTERS - center))])
    return center


def zone_label_from_center(center):
    return int(np.argmin(np.abs(CANONICAL_ZONE_CENTERS - center)))


def discretize_distance(dist):
    out = np.full(dist.shape, -1, dtype=np.int64)
    out[dist < -50] = 0
    out[(dist >= -50) & (dist <= -10)] = 1
    out[(dist > -10) & (dist < 0)] = 2
    out[dist == 0] = 3
    out[(dist > 0) & (dist <= 10)] = 4
    out[(dist > 10) & (dist <= 50)] = 5
    out[dist > 50] = 6
    return out


def discretize_position(pos):
    out = np.full(pos.shape, -1, dtype=np.int64)
    out[pos < 90] = 0
    out[(pos >= 90) & (pos < 180)] = 1
    out[(pos >= 180) & (pos < 270)] = 2
    out[(pos >= 270) & (pos < 360)] = 3
    out[pos >= 360] = 4
    return out


def discretize_speed(speed):
    out = np.full(speed.shape, -1, dtype=np.int64)
    out[speed < 2] = 0
    out[(speed >= 2) & (speed < 10)] = 1
    out[(speed >= 10) & (speed < 20)] = 2
    out[(speed >= 20) & (speed < 40)] = 3
    out[speed >= 40] = 4
    return out


def build_trials(sess):
    neural = sess['neural']
    timestamps = sess['timestamps']
    assert neural.shape[0] == len(timestamps)
    n_time, n_roi = neural.shape

    iscell = sess['iscell'] > 0.5
    neural = neural[:, iscell]
    plane_idx = sess['plane_idx'][iscell]

    trnum = np.asarray(sess['trial number']).astype(int)
    tstart = np.asarray(sess['trial_start']) > 0
    scanning = np.asarray(sess['scanning']) > 0
    pos = np.asarray(sess['position'], dtype=np.float32)
    speed = np.asarray(sess['speed'], dtype=np.float32)
    lick = (np.asarray(sess['lick']) > 0).astype(np.int64)
    env = np.asarray(sess['environment']).astype(int)
    reward_zone = np.asarray(sess['reward_zone']).astype(int)

    valid = (trnum >= 0) & scanning & np.isfinite(pos) & np.isfinite(speed)
    trial_ids = np.unique(trnum[valid])
    start_trial_ids = set(trnum[tstart & (trnum >= 0)].tolist())
    reward_times = np.asarray(sess['Reward_t'], dtype=np.float64)

    trial_reward = {}
    for trial in trial_ids:
        if trial not in start_trial_ids:
            continue
        m = valid & (trnum == trial)
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            continue
        t0 = timestamps[idx[0]]
        t1 = timestamps[idx[-1]]
        trial_reward[trial] = int(np.any((reward_times >= t0) & (reward_times <= t1)))

    neural_trials, input_trials, output_trials = [], [], []
    zone_centers = []
    zone_labels = []
    for trial in trial_ids:
        m = valid & (trnum == trial)
        idx = np.flatnonzero(m)
        if len(idx) < 2:
            continue
        # align to explicit trial_start pulse when present inside trial
        start_candidates = idx[tstart[idx]]
        start_idx = int(start_candidates[0]) if len(start_candidates) else int(idx[0])
        idx = idx[idx >= start_idx]
        if len(idx) < 2:
            continue
        trial_t = timestamps[idx] - timestamps[start_idx]
        trial_pos = pos[idx]
        trial_speed = speed[idx]
        trial_lick = lick[idx]
        trial_env = env[idx]
        trial_rz = reward_zone[idx]
        center = infer_zone_center(trial_pos, trial_rz)
        zlabel = zone_label_from_center(center)
        zone_centers.append(center)
        zone_labels.append(zlabel)
        dist = trial_pos - center
        prev_outcome = trial_reward.get(trial - 1, 0)
        this_outcome = trial_reward.get(trial, 0)

        inp = np.vstack([
            trial_t.astype(np.float32),
            np.full(len(idx), int(np.round(np.median(trial_env[trial_env >= 0]))) if np.any(trial_env >= 0) else 0, dtype=np.float32),
            np.full(len(idx), float(trial), dtype=np.float32),
            np.full(len(idx), float(prev_outcome), dtype=np.float32),
        ])
        out = np.vstack([
            discretize_distance(dist),
            discretize_position(np.clip(trial_pos, 0, 450)),
            discretize_speed(trial_speed),
            trial_lick.astype(np.int64),
            np.full(len(idx), zlabel, dtype=np.int64),
            np.full(len(idx), this_outcome, dtype=np.int64),
        ])
        neu = neural[idx].T.astype(np.float32)
        neural_trials.append(neu)
        input_trials.append(inp)
        output_trials.append(out)

    return neural_trials, input_trials, output_trials, plane_idx, zone_centers, zone_labels


def convert_dataset(files):
    subjects = []
    subject_to_idx = {}
    brain_regions = ['CA1']
    all_neural, all_input, all_output = [], [], []
    subject_idx = []
    brain_region_idx = []
    session_info = []
    for path in files:
        t0 = time.time()
        sess = load_nwb_session(path)
        ntrials = int((np.asarray(sess['trial_start']) > 0).sum())
        neural_trials, input_trials, output_trials, plane_idx, zone_centers, zone_labels = build_trials(sess)
        if len(neural_trials) < 2:
            continue
        subj = sess['subject']
        if subj not in subject_to_idx:
            subject_to_idx[subj] = len(subjects)
            subjects.append(subj)
        subject_idx.append(subject_to_idx[subj])
        all_neural.append(neural_trials)
        all_input.append(input_trials)
        all_output.append(output_trials)
        brain_region_idx.append(np.zeros(np.sum(sess['iscell'] > 0.5), dtype=np.int64))
        session_info.append({
            'session_id': sess['session'],
            'source_file': str(path),
            'native_rate_hz': sess['rate'],
            'n_trials_raw': ntrials,
            'n_trials_kept': len(neural_trials),
            'zone_centers_median': float(np.median(zone_centers)) if zone_centers else None,
            'zone_labels_unique': sorted(set(zone_labels)),
            'duration_sec': float(sess['timestamps'][-1] - sess['timestamps'][0]),
            'convert_sec': time.time() - t0,
        })
        print(f'converted {path.name}: kept {len(neural_trials)} trials, {all_neural[-1][0].shape[0]} neurons')

    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': subjects,
        'subject_idx': np.asarray(subject_idx, dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': ['time_from_trial_start_sec', 'environment_type', 'trial_number', 'previous_trial_outcome'],
        'output_names': ['distance_to_reward_zone', 'absolute_position', 'speed', 'lick', 'reward_zone_location', 'reward_outcome'],
        'output_values': [
            ['lt_-50', '-50_to_-10', '-10_to_0', '0', '0_to_10', '10_to_50', 'gt_50'],
            ['lt_90', '90_to_180', '180_to_270', '270_to_360', 'gt_360'],
            ['lt_2', '2_to_10', '10_to_20', '20_to_40', 'gt_40'],
            ['no', 'yes'],
            ['A', 'B', 'C'],
            ['no', 'yes'],
        ],
        'metadata': {
            'task_description': 'Hippocampal calcium imaging during virtual linear track navigation with changing reward-zone locations; decode trial context and behavior aligned to trial start.',
            'time_bin_size': float(1000.0 / np.median([s['native_rate_hz'] for s in session_info])) if session_info else None,
            'temporal_alignment_event': 'trial_start pulse from behavioral time series',
            'off_start': 0.0,
            'off_end': None,
            'neural_source': 'NWB ophys Deconvolved ROI response series',
            'neuron_curation': 'iscell > 0.5',
            'session_info': session_info,
        },
    }
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true')
    g.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    args = ap.parse_args()

    files = sorted(Path('/app/data').glob('sub-*/*.nwb'))
    if args.sample:
        files = files[:2]
    data = convert_dataset(files)
    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    print('wrote', args.outpicklefile)
    print('n_sessions', len(data['neural']))
    print('n_subjects', len(data['subjects']))

if __name__ == '__main__':
    main()
