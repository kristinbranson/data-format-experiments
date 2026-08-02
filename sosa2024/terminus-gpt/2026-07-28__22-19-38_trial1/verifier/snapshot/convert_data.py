#!/usr/bin/env python3
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
from pynwb import NWBHDF5IO


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpickle')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true', help='Process all sessions')
    g.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    ap.add_argument('--show-processing', action='store_true', help='Placeholder; save processing plots for up to 2 sessions')
    return ap.parse_args()


def get_files(sample=False):
    files = sorted(Path('data').glob('sub-*/*.nwb'))
    return files[:2] if sample else files


def _ts_data(ts):
    data = np.asarray(ts.data[:])
    t = np.asarray(ts.timestamps[:]) if ts.timestamps is not None else None
    return data, t


def infer_region(nwb):
    try:
        ophys = nwb.processing['ophys']
        seg = ophys.data_interfaces['ImageSegmentation']
        plane = next(iter(seg.plane_segmentations.values()))
        desc = getattr(plane.imaging_plane, 'description', '') or ''
        loc = getattr(plane.imaging_plane, 'location', '') or ''
        region = loc if loc else desc
        return region if region else 'unknown'
    except Exception:
        return 'unknown'


def contiguous_segments(mask):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)
    return [(g[0], g[-1] + 1) for g in groups if len(g)]


def trial_bounds_from_trial_start(trial_start):
    starts = np.flatnonzero(trial_start > 0)
    bounds = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(trial_start)
        if e > s:
            bounds.append((s, e))
    return bounds


def reward_zone_centers(position, reward_zone):
    centers = {}
    for code in sorted(c for c in np.unique(reward_zone) if c > 0):
        pos = position[reward_zone == code]
        pos = pos[np.isfinite(pos)]
        if len(pos):
            centers[int(code)] = float(np.median(pos))
    return centers


def collapse_zone_position_to_abc(pos):
    if pos < 150:
        return 0
    if pos < 260:
        return 1
    return 2


def distance_bin(d):
    if d < -50:
        return 0
    if d < -10:
        return 1
    if d < 0:
        return 2
    if d == 0:
        return 3
    if d <= 10:
        return 4
    if d <= 50:
        return 5
    return 6


def speed_bin(v):
    if v < 2:
        return 0
    if v < 10:
        return 1
    if v < 20:
        return 2
    if v < 40:
        return 3
    return 4


def pos_bins(pos, lo=0.0, hi=450.0):
    edges = np.linspace(lo, hi, 6)
    out = np.digitize(np.clip(pos, lo, hi - 1e-6), edges[1:-1], right=False)
    return out.astype(np.int64)


def process_file(fpath):
    with NWBHDF5IO(str(fpath), 'r', load_namespaces=True) as io:
        nwb = io.read()
        subj = getattr(nwb.subject, 'subject_id', fpath.parent.name)
        sess_id = nwb.session_id
        bts = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series
        deconv = nwb.processing['ophys'].data_interfaces['Deconvolved'].roi_response_series['plane0']

        neural = np.asarray(deconv.data[:], dtype=np.float32)
        timestamps = np.asarray(deconv.timestamps[:]) if deconv.timestamps is not None else np.arange(neural.shape[0]) / float(deconv.rate)
        rate = float(deconv.rate) if getattr(deconv, 'rate', None) is not None else float(1.0 / np.median(np.diff(timestamps)))

        behavior = {}
        for k in ['trial_start', 'trial number', 'environment', 'reward_zone', 'teleport', 'Reward', 'lick', 'speed', 'position', 'autoreward', 'scanning']:
            behavior[k] = _ts_data(bts[k])

        trial_start = np.asarray(behavior['trial_start'][0]).ravel()
        trial_num = np.asarray(behavior['trial number'][0]).ravel()
        env = np.asarray(behavior['environment'][0]).ravel()
        reward_zone = np.asarray(behavior['reward_zone'][0]).ravel()
        lick = np.asarray(behavior['lick'][0]).ravel()
        speed = np.asarray(behavior['speed'][0]).ravel()
        position = np.asarray(behavior['position'][0]).ravel()
        reward_ts = np.asarray(behavior['Reward'][1]).ravel() if behavior['Reward'][1] is not None else np.array([])

        bounds = trial_bounds_from_trial_start(trial_start)
        zone_centers = reward_zone_centers(position, reward_zone)

        sess_neural, sess_input, sess_output = [], [], []
        reward_outcomes = []
        trial_zone_labels = []

        for ti, (s, e) in enumerate(bounds):
            if e - s < 2:
                continue
            if np.nanmax(trial_num[s:e]) < 0:
                continue
            if np.all(position[s:e] <= -100):
                continue

            t0 = timestamps[s]
            t1 = timestamps[e - 1]
            rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
            reward_outcomes.append(rew)

            rz_vals = reward_zone[s:e]
            rz_nz = rz_vals[rz_vals > 0]
            if len(rz_nz):
                code = int(np.bincount(rz_nz.astype(int)).argmax())
                center = zone_centers.get(code, np.nan)
            elif rew:
                ridx = np.searchsorted(timestamps, reward_ts[(reward_ts >= t0) & (reward_ts <= t1 + 1e-9)][0])
                ridx = min(ridx, len(position) - 1)
                center = float(position[ridx])
            else:
                center = np.nanmedian(position[s:e])
            zone_label = collapse_zone_position_to_abc(center if np.isfinite(center) else 225.0)
            trial_zone_labels.append(zone_label)

            zone_center_lookup = {0: 90.0, 1: 205.0, 2: 325.0}
            zc = zone_center_lookup[zone_label]
            d = position[s:e] - zc
            out_dist = np.array([distance_bin(float(x)) for x in d], dtype=np.int64)
            out_pos = pos_bins(position[s:e])
            out_speed = np.array([speed_bin(float(max(0.0, x))) for x in speed[s:e]], dtype=np.int64)
            out_lick = (lick[s:e] > 0).astype(np.int64)
            out_rz = np.full(e - s, zone_label, dtype=np.int64)
            out_rew = np.full(e - s, rew, dtype=np.int64)
            output = np.vstack([out_dist, out_pos, out_speed, out_lick, out_rz, out_rew])

            env_valid = env[s:e][env[s:e] >= 0]
            env_label = int(np.round(np.median(env_valid))) if len(env_valid) else 0
            tr_valid = trial_num[s:e][trial_num[s:e] >= 0]
            tr_label = float(np.median(tr_valid)) if len(tr_valid) else float(ti)
            prev_rew = reward_outcomes[-2] if len(reward_outcomes) >= 2 else 0
            rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
            inp = np.vstack([
                rel_time,
                np.full(e - s, env_label, dtype=np.float32),
                np.full(e - s, tr_label, dtype=np.float32),
                np.full(e - s, prev_rew, dtype=np.float32),
            ])

            trial_neural = neural[s:e, :].T.astype(np.float32)
            sess_neural.append(trial_neural)
            sess_input.append(inp.astype(np.float32))
            sess_output.append(output)

        region = infer_region(nwb)
        n_neurons = neural.shape[1]
        region_idx = np.zeros(n_neurons, dtype=np.int64)
        return {
            'subject': subj,
            'session_id': sess_id,
            'region': region,
            'rate': rate,
            'neural': sess_neural,
            'input': sess_input,
            'output': sess_output,
            'brain_region_idx': region_idx,
        }


def main():
    args = parse_args()
    sample = args.sample and not args.full
    files = get_files(sample=sample)
    t0 = time.time()
    sessions = []
    for i, f in enumerate(files, 1):
        st = time.time()
        sess = process_file(f)
        if len(sess['neural']) >= 2:
            sessions.append(sess)
        print(f'processed {i}/{len(files)} {f.name}: trials={len(sess["neural"])} neurons={(sess["brain_region_idx"].shape[0])} time={time.time()-st:.2f}s', flush=True)

    subjects = sorted({s['subject'] for s in sessions})
    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    brain_regions = sorted({s['region'] for s in sessions})
    region_to_idx = {r: i for i, r in enumerate(brain_regions)}

    data = {
        'neural': [s['neural'] for s in sessions],
        'input': [s['input'] for s in sessions],
        'output': [s['output'] for s in sessions],
        'subjects': subjects,
        'subject_idx': np.array([subject_to_idx[s['subject']] for s in sessions], dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': [np.full(len(s['brain_region_idx']), region_to_idx[s['region']], dtype=np.int64) for s in sessions],
        'input_names': ['time_from_trial_start', 'environment_type', 'trial_number', 'previous_trial_outcome'],
        'output_names': ['distance_to_reward_zone', 'absolute_position_bin', 'speed_bin', 'lick', 'reward_zone_location', 'reward_outcome'],
        'output_values': [
            ['lt_neg50', 'neg50_to_neg10', 'neg10_to_0', '0', '0_to_10', '10_to_50', 'gt_50'],
            ['bin0', 'bin1', 'bin2', 'bin3', 'bin4'],
            ['lt2', '2_to_10', '10_to_20', '20_to_40', 'gt40'],
            ['no', 'yes'],
            ['A', 'B', 'C'],
            ['no', 'yes'],
        ],
        'metadata': {
            'task_description': 'Hippocampal virtual corridor reward-switch task; decode position/reward-related and behavioral variables from deconvolved neural activity.',
            'time_bin_size': float(1000.0 / sessions[0]['rate']) if sessions else None,
            'temporal_alignment_event': 'trial_start',
            'off_start': 0.0,
            'off_end': None,
            'n_sessions': len(sessions),
            'source_format': 'NWB',
            'neural_signal': 'deconvolved ophys activity',
        },
    }

    with open(args.outpickle, 'wb') as f:
        pickle.dump(data, f)
    print(f'saved {args.outpickle} with {len(sessions)} sessions in {time.time()-t0:.2f}s', flush=True)


if __name__ == '__main__':
    main()
