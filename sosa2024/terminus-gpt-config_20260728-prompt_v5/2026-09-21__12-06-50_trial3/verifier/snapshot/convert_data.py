#!/usr/bin/env python3
import argparse
import pickle
import time
from pathlib import Path

import h5py
import numpy as np


TRACK_LEN = 450.0
OUTPUT_NAMES = [
    'distance_to_reward_zone',
    'absolute_position',
    'speed',
    'lick',
    'reward_zone_location',
    'reward_outcome',
]
INPUT_NAMES = [
    'time_from_trial_start',
    'environment_type',
    'trial_number',
    'previous_trial_outcome',
]
OUTPUT_VALUES = [
    ['lt_-50', '-50_to_-10', '-10_to_lt_0', '0', 'gt_0_to_10', '10_to_50', 'gt_50'],
    ['bin0', 'bin1', 'bin2', 'bin3', 'bin4'],
    ['lt_2', '2_to_10', '10_to_20', '20_to_40', 'gt_40'],
    ['no', 'yes'],
    ['A', 'B', 'C'],
    ['no', 'yes'],
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpickle')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true')
    mode.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    return ap.parse_args()


def find_nwb_files():
    return sorted(Path('/app/data').rglob('*.nwb'))


def read_scalar(h5, path, default='UNKNOWN'):
    if path not in h5:
        return default
    x = h5[path][()]
    if isinstance(x, bytes):
        return x.decode()
    return str(x)


def first_valid(arr):
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return arr[0]


def discretize_distance(d):
    out = np.full(d.shape, -1, dtype=np.int64)
    out[d < -50] = 0
    out[(d >= -50) & (d <= -10)] = 1
    out[(d > -10) & (d < 0)] = 2
    out[np.isclose(d, 0)] = 3
    out[(d > 0) & (d <= 10)] = 4
    out[(d > 10) & (d <= 50)] = 5
    out[d > 50] = 6
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


def reward_zone_label_from_position(rz_val):
    # provisional mapping by sorted unique in-trial zone codes to A/B/C
    return rz_val


def get_trial_bounds(trial_start, teleport, trial_number):
    starts = np.where(np.diff(trial_start.astype(int), prepend=0) > 0)[0]
    ends = np.where(np.diff(teleport.astype(int), prepend=0) > 0)[0]
    bounds = []
    for s in starts:
        e_candidates = ends[ends > s]
        if e_candidates.size == 0:
            continue
        e = int(e_candidates[0])
        if np.any(trial_number[s:e] >= 0):
            bounds.append((int(s), int(e)))
    return bounds


def compute_env_mapping(env_trials):
    vals = sorted({int(v) for v in env_trials if v >= 0})
    return {v: i for i, v in enumerate(vals[:2])}


def compute_rz_mapping_from_centers(centers):
    vals = sorted({round(float(v), 1) for v in centers if v is not None and np.isfinite(v)})
    if not vals:
        return {}
    # compress nearby centers and keep sorted order as A/B/C
    merged = []
    for v in vals:
        if not merged or abs(v - merged[-1]) > 30:
            merged.append(v)
    return {v: min(i, 2) for i, v in enumerate(merged[:3])}


def convert_session(fpath):
    with h5py.File(fpath, 'r') as f:
        subj = read_scalar(f, 'general/subject/subject_id')
        sess_id = read_scalar(f, 'general/session_id', fpath.stem)
        neural = np.asarray(f['processing/ophys/Deconvolved/plane0/data'], dtype=np.float32)
        fluor = np.asarray(f['processing/ophys/Fluorescence/plane0/data'], dtype=np.float32)
        ts = np.asarray(f['processing/behavior/BehavioralTimeSeries/position/timestamps'], dtype=np.float64)
        base = 'processing/behavior/BehavioralTimeSeries'
        beh = {k: np.asarray(f[f'{base}/{k}/data']) for k in ['environment', 'reward_zone', 'teleport', 'lick', 'speed', 'position', 'trial number', 'trial_start', 'scanning']}
        reward = np.asarray(f[f'{base}/Reward/data']) if f'{base}/Reward/data' in f else None
        reward_ts = np.asarray(f[f'{base}/Reward/timestamps']) if f'{base}/Reward/timestamps' in f else None

    assert neural.shape == fluor.shape
    lengths = [neural.shape[0], ts.shape[0]] + [v.shape[0] for v in beh.values()]
    common_len = min(lengths)
    if len(set(lengths)) != 1:
        print(f'length mismatch in {fpath.name}: lengths={lengths}, trimming to {common_len}')
    neural = neural[:common_len]
    fluor = fluor[:common_len]
    ts = ts[:common_len]
    beh = {k: v[:common_len] for k, v in beh.items()}
    n_time, n_neurons = neural.shape
    bounds = get_trial_bounds(beh['trial_start'], beh['teleport'], beh['trial number'])

    env_per_trial = []
    rz_per_trial = []
    reward_outcomes = []
    trial_nums = []
    for s, e in bounds:
        valid = beh['trial number'][s:e] >= 0
        env_vals = beh['environment'][s:e][valid]
        rz_vals = beh['reward_zone'][s:e][valid]
        tn_vals = beh['trial number'][s:e][valid]
        env_per_trial.append(int(first_valid(env_vals)) if env_vals.size else -1)
        rz_per_trial.append(int(first_valid(rz_vals)) if rz_vals.size else -1)
        trial_nums.append(float(first_valid(tn_vals)) if tn_vals.size else len(trial_nums))
        if reward is not None and reward_ts is not None and reward.shape[0] != ts.shape[0]:
            t_start = ts[s]
            t_end = ts[e-1] if e-1 < len(ts) else ts[-1]
            m_rew = (reward_ts >= t_start) & (reward_ts <= t_end)
            reward_outcomes.append(int(np.any(reward[m_rew] > 0)))
        elif reward is not None:
            reward_outcomes.append(int(np.any(reward[s:e][valid] > 0)))
        else:
            reward_outcomes.append(0)

    env_map = compute_env_mapping(env_per_trial)
    rz_centers = []
    for i, (s, e) in enumerate(bounds):
        valid = beh['trial number'][s:e] >= 0
        pos_vals = beh['position'][s:e][valid]
        rz_vals = beh['reward_zone'][s:e][valid]
        m = rz_vals > 0
        rz_centers.append(float(np.nanmedian(pos_vals[m])) if np.any(m) else None)
    rz_map = compute_rz_mapping_from_centers(rz_centers)

    session_neural, session_input, session_output = [], [], []
    for i, (s, e) in enumerate(bounds):
        valid = (beh['trial number'][s:e] >= 0) & (beh['scanning'][s:e] > 0)
        if valid.sum() < 2:
            continue
        idx = np.where(valid)[0] + s
        t0 = ts[idx[0]]
        t_rel = (ts[idx] - t0).astype(np.float32)
        pos = beh['position'][idx].astype(np.float32)
        speed = beh['speed'][idx].astype(np.float32)
        lick = (beh['lick'][idx] > 0).astype(np.int64)
        rz_series = beh['reward_zone'][idx].astype(np.float32)
        rz_center = rz_centers[i] if i < len(rz_centers) else None
        if rz_center is None or not np.isfinite(rz_center):
            rz_center = float(np.nanmedian(pos))
        # map center to A/B/C by nearest discovered center
        if rz_map:
            nearest = min(rz_map.keys(), key=lambda x: abs(x - rz_center))
            rz_cat = rz_map[nearest]
        else:
            rz_cat = 0
        dist = pos - float(rz_center)

        inp = np.vstack([
            t_rel,
            np.full_like(t_rel, float(env_per_trial[i] if env_per_trial[i] >= 0 else 0), dtype=np.float32),
            np.full_like(t_rel, trial_nums[i], dtype=np.float32),
            np.full_like(t_rel, reward_outcomes[i-1] if i > 0 else 0, dtype=np.float32),
        ]).astype(np.float32)
        out = np.vstack([
            discretize_distance(dist),
            discretize_position(np.clip(pos, 0, TRACK_LEN)),
            discretize_speed(np.maximum(speed, 0)),
            lick,
            np.full(t_rel.shape, rz_cat, dtype=np.int64),
            np.full(t_rel.shape, reward_outcomes[i], dtype=np.int64),
        ]).astype(np.int64)
        neu = neural[idx].T.astype(np.float32)
        session_neural.append(neu)
        session_input.append(inp)
        session_output.append(out)

    return subj, sess_id, n_neurons, session_neural, session_input, session_output


def main():
    args = parse_args()
    t_start = time.time()
    files = find_nwb_files()
    if args.sample:
        files = files[:2]
    print(f'processing {len(files)} sessions')

    subjects = []
    subject_to_idx = {}
    neural_all, input_all, output_all = [], [], []
    subject_idx = []
    brain_regions = ['CA1']
    brain_region_idx = []
    session_info = []

    for i, fpath in enumerate(files):
        t0 = time.time()
        subj, sess_id, n_neurons, n_list, in_list, out_list = convert_session(fpath)
        if len(n_list) < 2:
            print(f'skipping {fpath.name}: fewer than 2 valid trials after filtering')
            continue
        if subj not in subject_to_idx:
            subject_to_idx[subj] = len(subjects)
            subjects.append(subj)
        neural_all.append(n_list)
        input_all.append(in_list)
        output_all.append(out_list)
        subject_idx.append(subject_to_idx[subj])
        brain_region_idx.append(np.zeros(n_neurons, dtype=np.int64))
        session_info.append({'file': str(fpath), 'session_id': sess_id})
        print(f'[{i+1}/{len(files)}] {fpath.name}: trials={len(n_list)} neurons={n_neurons} time={time.time()-t0:.2f}s')

    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,
        'subjects': subjects,
        'subject_idx': np.asarray(subject_idx, dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'task_description': 'Hippocampal reward-relative navigation task with trial-aligned temporal decoding of position, reward-zone distance, speed, lick, reward-zone identity, and reward outcome.',
            'time_bin_size': None,
            'temporal_alignment_event': 'trial start',
            'off_start': 0.0,
            'off_end': None,
            'signal_type': 'deconvolved calcium activity',
            'track_length_cm': TRACK_LEN,
            'session_info': session_info,
        },
    }

    outpath = Path(args.outpickle)
    with outpath.open('wb') as f:
        pickle.dump(data, f)
    print(f'saved {outpath}')
    print(f'total_time_sec {time.time()-t_start:.2f}')


if __name__ == '__main__':
    main()
