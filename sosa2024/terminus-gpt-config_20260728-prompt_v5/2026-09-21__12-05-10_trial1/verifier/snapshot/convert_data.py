#!/usr/bin/env python3
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
from pynwb import NWBHDF5IO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


INPUT_NAMES = [
    'time_from_trial_start_s',
    'environment_type',
    'trial_number',
    'previous_trial_outcome',
]

OUTPUT_NAMES = [
    'distance_to_reward_zone',
    'absolute_position',
    'speed',
    'lick',
    'reward_zone_location',
    'reward_outcome',
]

OUTPUT_VALUES = [
    ['lt_-50', '-50_to_-10', '-10_to_lt0', '0', 'gt0_to_10', '10_to_50', 'gt_50'],
    ['lt_90', '90_to_180', '180_to_270', '270_to_360', 'gt_360'],
    ['lt_2', '2_to_10', '10_to_20', '20_to_40', 'gt_40'],
    ['no', 'yes'],
    ['A', 'B', 'C'],
    ['no', 'yes'],
]


def list_sessions(data_root: Path):
    return sorted(data_root.rglob('*.nwb'))


def align_series_to_frame(ts, frame_timestamps):
    data = np.asarray(ts.data[:])
    ts_t = np.asarray(ts.timestamps[:], dtype=np.float64) if ts.timestamps is not None else None
    if data.ndim > 1:
        data = np.squeeze(data)
    if ts_t is not None and len(data) == len(frame_timestamps) and len(ts_t) == len(frame_timestamps):
        return data
    if ts_t is None:
        if len(data) == len(frame_timestamps):
            return data
        out = np.full(len(frame_timestamps), np.nan, dtype=np.float64)
        n = min(len(data), len(out))
        out[:n] = data[:n]
        return out
    if len(data) == 0:
        return np.zeros(len(frame_timestamps), dtype=np.float64)
    # sparse event-like series: place values at nearest frame timestamps
    if len(data) < len(frame_timestamps):
        out = np.zeros(len(frame_timestamps), dtype=np.float64)
        idx = np.searchsorted(frame_timestamps, ts_t)
        idx = np.clip(idx, 0, len(frame_timestamps) - 1)
        left = np.maximum(idx - 1, 0)
        choose_left = np.abs(frame_timestamps[left] - ts_t) < np.abs(frame_timestamps[idx] - ts_t)
        idx[choose_left] = left[choose_left]
        out[idx] = data
        return out
    # fallback interpolation / nearest carry-forward
    order = np.argsort(ts_t)
    ts_t = ts_t[order]
    data = data[order]
    return np.interp(frame_timestamps, ts_t, data)


def read_session(nwb_path: Path):
    with NWBHDF5IO(str(nwb_path), 'r', load_namespaces=True) as io:
        nwb = io.read()
        subj = nwb.subject.subject_id if nwb.subject is not None else nwb_path.parent.name.replace('sub-', '')
        sess_id = getattr(nwb, 'session_id', nwb_path.stem)

        beh = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries']
        ophys = nwb.processing['ophys']
        deconv = np.asarray(ophys.data_interfaces['Deconvolved'].roi_response_series['plane0'].data[:], dtype=np.float32)
        roi_count = deconv.shape[1]

        frame_timestamps = np.asarray(beh.time_series['trial_start'].timestamps[:], dtype=np.float64)
        ts_names = ['Reward', 'environment', 'lick', 'position', 'reward_zone', 'scanning', 'speed', 'teleport', 'trial number', 'trial_start']
        beh_data = {k: align_series_to_frame(beh.time_series[k], frame_timestamps) for k in ts_names}

        return {
            'subject': str(subj),
            'session_id': str(sess_id),
            'path': str(nwb_path),
            'neural_t_by_n': deconv,
            'timestamps': frame_timestamps,
            'behavior': beh_data,
            'roi_count': roi_count,
        }


def contiguous_segments(start_idxs, n_time):
    starts = list(start_idxs)
    segs = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else n_time
        if e > s:
            segs.append((s, e))
    return segs


def map_binary_environment(env_trial_vals):
    vals = env_trial_vals[~np.isnan(env_trial_vals)]
    uniq = np.unique(vals)
    if len(uniq) == 0:
        return np.zeros_like(env_trial_vals, dtype=np.int64), {}
    uniq_sorted = sorted(uniq.tolist())
    mapping = {v: i for i, v in enumerate(uniq_sorted[:2])}
    out = np.array([mapping.get(v, 0) for v in env_trial_vals], dtype=np.int64)
    return out, mapping


def map_reward_zone(zone_trial_vals):
    vals = zone_trial_vals[~np.isnan(zone_trial_vals)]
    uniq = sorted(np.unique(vals).tolist())
    mapping = {v: i for i, v in enumerate(uniq[:3])}
    out = np.array([mapping.get(v, 0) for v in zone_trial_vals], dtype=np.int64)
    return out, mapping


def infer_zone_interval_and_location(rz_vals, pos_vals):
    m = (rz_vals > 0) & np.isfinite(pos_vals) & (pos_vals >= 0) & (pos_vals <= 450)
    if np.any(m):
        lo = float(np.min(pos_vals[m]))
        hi = float(np.max(pos_vals[m]))
    else:
        c = float(np.nanmedian(pos_vals[np.isfinite(pos_vals)]))
        lo, hi = c - 10.0, c + 10.0
    center = 0.5 * (lo + hi)
    if center < 200:
        loc = 0
    elif center < 300:
        loc = 1
    else:
        loc = 2
    return lo, hi, loc


def signed_distance_to_interval(pos, lo, hi):
    dist = np.zeros_like(pos, dtype=np.float32)
    dist[pos < lo] = pos[pos < lo] - lo
    dist[pos > hi] = pos[pos > hi] - hi
    return dist


def bin_distance(dist):
    out = np.full(dist.shape, 6, dtype=np.int64)
    out[dist < -50] = 0
    out[(dist >= -50) & (dist < -10)] = 1
    out[(dist >= -10) & (dist < 0)] = 2
    out[dist == 0] = 3
    out[(dist > 0) & (dist <= 10)] = 4
    out[(dist > 10) & (dist <= 50)] = 5
    out[dist > 50] = 6
    return out


def bin_position(pos):
    out = np.full(pos.shape, 0, dtype=np.int64)
    out[(pos >= 90) & (pos < 180)] = 1
    out[(pos >= 180) & (pos < 270)] = 2
    out[(pos >= 270) & (pos < 360)] = 3
    out[pos >= 360] = 4
    return out


def bin_speed(speed):
    s = np.maximum(speed, 0)
    out = np.full(s.shape, 0, dtype=np.int64)
    out[(s >= 2) & (s < 10)] = 1
    out[(s >= 10) & (s < 20)] = 2
    out[(s >= 20) & (s < 40)] = 3
    out[s >= 40] = 4
    return out


def process_session(sess, make_plot=False):
    neural = sess['neural_t_by_n']
    t = sess['timestamps']
    b = sess['behavior']

    valid = np.isfinite(t) & (b['trial number'] >= 0)
    if 'scanning' in b:
        valid &= (b['scanning'] > 0)

    trial_start_idx = np.flatnonzero((b['trial_start'] > 0) & valid)
    if len(trial_start_idx) < 2:
        return [], [], [], np.zeros(sess['roi_count'], dtype=np.int64)

    segs = contiguous_segments(trial_start_idx, len(t))
    trial_nums = []
    for s, e in segs:
        tr = b['trial number'][s:e]
        tr_valid = tr[tr >= 0]
        if len(tr_valid) == 0:
            continue
        trial_nums.append(int(np.round(np.median(tr_valid))))
    if not trial_nums:
        return [], [], [], np.zeros(sess['roi_count'], dtype=np.int64)

    env_per_trial = []
    rz_per_trial = []
    rz_interval_per_trial = []
    rew_per_trial = []
    kept_segs = []
    for (s, e), trn in zip(segs, trial_nums):
        mask = valid[s:e]
        if mask.sum() < 5:
            continue
        pos = np.asarray(b['position'][s:e])[mask]
        if np.sum((pos >= 0) & (pos <= 450)) < 5:
            continue
        env = np.asarray(b['environment'][s:e])[mask]
        rz = np.asarray(b['reward_zone'][s:e])[mask]
        rw = np.asarray(b['Reward'][s:e])[mask]
        zlo, zhi, zloc = infer_zone_interval_and_location(rz, pos)
        env_per_trial.append(np.nanmedian(env))
        rz_per_trial.append(zloc)
        rz_interval_per_trial.append((zlo, zhi))
        rew_per_trial.append(int(np.any(rw > 0)))
        kept_segs.append((s, e))

    if len(kept_segs) < 2:
        return [], [], [], np.zeros(sess['roi_count'], dtype=np.int64)

    env_codes, env_map = map_binary_environment(np.asarray(env_per_trial, dtype=float))
    rz_codes = np.asarray(rz_per_trial, dtype=np.int64)
    rz_map = {0: 0, 1: 1, 2: 2}
    prev_rew = np.array([0] + rew_per_trial[:-1], dtype=np.int64)

    neural_trials, input_trials, output_trials = [], [], []
    for i, (s, e) in enumerate(kept_segs):
        mask = valid[s:e]
        idx = np.flatnonzero(mask) + s
        tt = t[idx] - t[idx[0]]
        nn = neural[idx, :].T.astype(np.float32)

        pos = np.asarray(b['position'][idx], dtype=np.float32)
        speed = np.asarray(b['speed'][idx], dtype=np.float32)
        lick = (np.asarray(b['lick'][idx]) > 0).astype(np.int64)
        pos_clip = np.clip(pos, 0, 450)
        zlo, zhi = rz_interval_per_trial[i]
        dist = signed_distance_to_interval(pos_clip, zlo, zhi)

        inp = np.vstack([
            tt.astype(np.float32),
            np.full(len(idx), env_codes[i], dtype=np.float32),
            np.full(len(idx), i, dtype=np.float32),
            np.full(len(idx), prev_rew[i], dtype=np.float32),
        ])
        out = np.vstack([
            bin_distance(dist),
            bin_position(pos_clip),
            bin_speed(speed),
            lick,
            np.full(len(idx), rz_codes[i], dtype=np.int64),
            np.full(len(idx), rew_per_trial[i], dtype=np.int64),
        ]).astype(np.int64)

        neural_trials.append(nn)
        input_trials.append(inp)
        output_trials.append(out)

    brain_region_idx = np.zeros(sess['roi_count'], dtype=np.int64)

    if make_plot and neural_trials:
        fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=False)
        tr = 0
        axs[0].imshow(neural_trials[tr][:min(40, neural_trials[tr].shape[0])], aspect='auto', interpolation='nearest')
        axs[0].set_title(f"{sess['subject']} session {sess['session_id']} neural sample")
        axs[1].plot(input_trials[tr][0], label='time_from_start')
        axs[1].legend()
        axs[2].plot(output_trials[tr][1], label='pos_bin')
        axs[2].plot(output_trials[tr][2], label='speed_bin')
        axs[2].plot(output_trials[tr][3], label='lick')
        axs[2].legend()
        axs[3].plot(output_trials[tr][0], label='dist_to_rz_bin')
        axs[3].plot(output_trials[tr][4], label='rz')
        axs[3].plot(output_trials[tr][5], label='reward')
        axs[3].legend()
        fig.tight_layout()
        fig.savefig(f"/app/processing_{sess['subject']}_{sess['session_id']}.png", dpi=150)
        plt.close(fig)

    return neural_trials, input_trials, output_trials, brain_region_idx


def convert(data_root: Path, out_path: Path, sample=False, show_processing=False):
    start = time.time()
    files = list_sessions(data_root)
    if sample:
        files = files[:2]

    subjects = []
    subject_to_idx = {}
    neural_all, input_all, output_all = [], [], []
    subject_idx = []
    brain_regions = ['CA1']
    brain_region_idx = []
    session_info = []
    dt_list = []

    for j, f in enumerate(files):
        t0 = time.time()
        sess = read_session(f)
        if sess['subject'] not in subject_to_idx:
            subject_to_idx[sess['subject']] = len(subjects)
            subjects.append(sess['subject'])
        dts = np.diff(sess['timestamps'])
        dts = dts[np.isfinite(dts) & (dts > 0)]
        if len(dts):
            dt_list.append(float(np.median(dts)))
        ntr, itr, otr, bri = process_session(sess, make_plot=show_processing and j < 2)
        if len(ntr) >= 2:
            neural_all.append(ntr)
            input_all.append(itr)
            output_all.append(otr)
            subject_idx.append(subject_to_idx[sess['subject']])
            brain_region_idx.append(bri)
            session_info.append({'path': sess['path'], 'session_id': sess['session_id'], 'subject': sess['subject'], 'n_trials': len(ntr), 'n_neurons': int(bri.shape[0])})
        print(f'processed {f.name}: kept_trials={len(ntr)} neurons={sess["roi_count"]} elapsed={time.time()-t0:.2f}s')

    time_bin = float(np.median(dt_list) * 1000.0) if dt_list else np.nan
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
            'task_description': 'Hippocampal 2P imaging during virtual linear-track reward task; decode trial-aligned behavioral and task variables from neural activity.',
            'time_bin_size': time_bin,
            'temporal_alignment_event': 'trial_start (entry to the linear track)',
            'off_start': 0.0,
            'off_end': None,
            'session_info': session_info,
            'source_data_root': str(data_root),
            'neural_signal': 'ophys/Deconvolved/plane0',
            'notes': 'Trials reconstructed from BehavioralTimeSeries trial_start and trial number; position clipped to [0,450] for discretization.',
        },
    }
    with open(out_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'saved {out_path}')
    print(f'total_sessions_kept={len(neural_all)} total_subjects={len(subjects)} total_time={time.time()-start:.2f}s')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true')
    g.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    args = ap.parse_args()
    sample = args.sample and not args.full
    convert(Path('/app/data'), Path(args.outpicklefile), sample=sample, show_processing=args.show_processing)
