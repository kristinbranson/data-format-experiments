#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def decode_scalar(x):
    if isinstance(x, bytes):
        return x.decode()
    if isinstance(x, np.ndarray) and x.shape == ():
        return decode_scalar(x[()])
    return x


def load_behavior_series(f):
    grp = f['processing/behavior/BehavioralTimeSeries']
    out = {}
    for k in grp.keys():
        g = grp[k]
        if 'data' in g:
            out[k] = np.asarray(g['data'][:])
        if 'timestamps' in g:
            out[k + '__timestamps'] = np.asarray(g['timestamps'][:])
    return out


def load_neural_series(f):
    for base in ['processing/ophys/DfOverF', 'processing/ophys/Fluorescence']:
        if base in f:
            grp = f[base]
            for k in grp.keys():
                g = grp[k]
                if 'data' in g:
                    data = np.asarray(g['data'][:])
                    ts = np.asarray(g['timestamps'][:]) if 'timestamps' in g else None
                    return data, ts, base, k
    raise RuntimeError('No DfOverF or Fluorescence dataset found')


def load_n_rois_and_regions(f):
    n_rois = None
    regions = None
    if 'processing/ophys/ImageSegmentation' in f:
        for k in f['processing/ophys/ImageSegmentation'].keys():
            ps = f['processing/ophys/ImageSegmentation'][k]
            if 'id' in ps:
                n_rois = len(ps['id'])
            if 'location' in ps:
                try:
                    loc = ps['location'][:]
                    regions = [decode_scalar(x) for x in loc]
                except Exception:
                    pass
            break
    return n_rois, regions


def rising_edges(x, thr=0.5):
    x = np.asarray(x)
    return np.where((x[1:] > thr) & (x[:-1] <= thr))[0] + 1


def infer_trial_bounds(b):
    trial_num = np.asarray(b['trial number'])
    trial_start = np.asarray(b['trial_start'])
    valid = trial_num >= 0
    starts = rising_edges(trial_start, 0.5)
    starts = starts[valid[starts]]
    if len(starts) == 0:
        changes = np.where(np.diff(trial_num) > 0)[0] + 1
        starts = changes[valid[changes]]
    trial_ids = trial_num[starts].astype(int)
    bounds = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(trial_num)
        if e - s > 1:
            bounds.append((int(trial_ids[i]), int(s), int(e)))
    return bounds


def map_environment(x):
    vals = np.unique(x[np.isfinite(x)])
    vals = [v for v in vals if v >= 0 or v == -1 or v == 1]
    uniq = sorted(set(v for v in vals if v != -1))
    if len(uniq) >= 2:
        lo, hi = uniq[0], uniq[-1]
        return np.where(x == hi, 1, 0)
    return (x > 0).astype(np.int64)


def infer_trial_reward_positions(pos, rz_signal, trials):
    trial_reward_pos = {}
    for tid, s, e in trials:
        sl = slice(s, e)
        nz = rz_signal[sl] > 0
        trial_reward_pos[tid] = float(np.nanmedian(pos[sl][nz])) if np.any(nz) else np.nan
    vals = np.array([v for v in trial_reward_pos.values() if np.isfinite(v)])
    if len(vals) == 0:
        centers = np.array([np.nan, np.nan, np.nan])
    else:
        k = min(3, len(np.unique(np.round(vals, 3))))
        km = KMeans(n_clusters=k, random_state=0, n_init=20).fit(vals.reshape(-1, 1))
        centers = np.sort(km.cluster_centers_.ravel())
    return trial_reward_pos, centers

def assign_reward_location_labels(trial_reward_pos, centers):
    labels = {}
    finite_centers = [c for c in centers if np.isfinite(c)]
    for tid, rp in trial_reward_pos.items():
        if not np.isfinite(rp) or len(finite_centers) == 0:
            labels[tid] = 0
        else:
            labels[tid] = int(np.argmin(np.abs(np.asarray(finite_centers) - rp)))
    return labels


def discretize_speed(speed):
    bins = np.zeros_like(speed, dtype=np.int64)
    bins[(speed >= 2) & (speed < 10)] = 1
    bins[(speed >= 10) & (speed < 20)] = 2
    bins[(speed >= 20) & (speed < 40)] = 3
    bins[speed >= 40] = 4
    return bins


def discretize_position(pos, valid_mask, n_bins=5):
    valid_pos = pos[valid_mask]
    lo, hi = np.nanmin(valid_pos), np.nanmax(valid_pos)
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(pos, edges[1:-1], right=False), 0, n_bins - 1)
    return idx, edges


def discretize_dist_to_reward(pos, reward_center):
    dist = pos - reward_center
    out = np.full(dist.shape, 0, dtype=np.int64)
    out[(dist >= -50) & (dist < -10)] = 1
    out[(dist >= -10) & (dist < 0)] = 2
    out[np.isclose(dist, 0)] = 3
    out[(dist > 0) & (dist <= 10)] = 4
    out[(dist > 10) & (dist <= 50)] = 5
    out[dist > 50] = 6
    out[np.isnan(dist)] = 0
    return out, dist


def infer_reward_outcomes_from_events(trials, behavior_timestamps, reward_event_timestamps):
    outcomes = []
    for _, s, e in trials:
        t0 = behavior_timestamps[s]
        t1 = behavior_timestamps[e - 1]
        rewarded = int(np.any((reward_event_timestamps >= t0) & (reward_event_timestamps <= t1)))
        outcomes.append(rewarded)
    return np.asarray(outcomes, dtype=np.int64)


def choose_zone_centers(pos, rz_code):
    centers = {}
    for z in sorted(set(rz_code[rz_code >= 0].tolist())):
        mask = rz_code == z
        if np.any(mask):
            centers[int(z)] = float(np.nanmedian(pos[mask]))
    return centers


def convert_session(path, show_processing=False):
    with h5py.File(path, 'r') as f:
        subj = decode_scalar(f['general/subject/subject_id'][()])
        sess = decode_scalar(f['general/session_id'][()])
        b = load_behavior_series(f)
        neural, neural_t, neural_base, neural_key = load_neural_series(f)
        n_rois, regions = load_n_rois_and_regions(f)

    # orient neural as neurons x time
    if neural.ndim != 2:
        raise RuntimeError(f'Unexpected neural shape {neural.shape}')
    if n_rois is not None:
        if neural.shape[0] == n_rois:
            neural_nt = neural
        elif neural.shape[1] == n_rois:
            neural_nt = neural.T
        else:
            neural_nt = neural if neural.shape[0] < neural.shape[1] else neural.T
    else:
        neural_nt = neural if neural.shape[0] < neural.shape[1] else neural.T

    bt = b.get('position__timestamps', None)
    if bt is None:
        any_ts = [v for k, v in b.items() if k.endswith('__timestamps')]
        bt = any_ts[0]
    pos = np.asarray(b['position'])
    speed = np.asarray(b['speed'])
    lick = np.asarray(b['lick'])
    env = map_environment(np.asarray(b['environment']))
    teleport = np.asarray(b['teleport'])
    trial_num = np.asarray(b['trial number']).astype(int)
    trial_start = np.asarray(b['trial_start'])
    rz_raw = np.asarray(b['reward_zone'])

    valid_mask = (trial_num >= 0) & (pos > -400)
    trials = infer_trial_bounds(b)
    trial_reward_pos, reward_loc_centers = infer_trial_reward_positions(pos, rz_raw, trials)
    trial_reward_loc = assign_reward_location_labels(trial_reward_pos, reward_loc_centers)
    reward_event_timestamps = b.get('Reward__timestamps', np.asarray([]))
    reward_outcomes = infer_reward_outcomes_from_events(trials, bt, reward_event_timestamps)

    abs_pos_bins, pos_edges = discretize_position(pos, valid_mask, n_bins=5)
    speed_bins = discretize_speed(speed)

    if neural_t is None:
        if bt is None:
            raise RuntimeError('No timestamps for neural or behavior')
        neural_t = np.linspace(bt[0], bt[-1], neural_nt.shape[1])

    session_neural = []
    session_input = []
    session_output = []

    prev_out = 0
    for i, (tid, s, e) in enumerate(trials):
        sl = slice(s, e)
        t0 = bt[s]
        t1 = bt[e - 1]
        nmask = (neural_t >= t0) & (neural_t <= t1)
        if np.sum(nmask) < 2 or (e - s) < 2:
            continue
        nt = neural_t[nmask]
        bidx = np.searchsorted(bt, nt, side='left')
        bidx = np.clip(bidx, 0, len(bt) - 1)

        neural_trial = neural_nt[:, nmask].astype(np.float32)
        time_from_start = (nt - t0).astype(np.float32)
        env_trial = np.full(nt.shape, int(np.round(np.nanmedian(env[sl]))), dtype=np.float32)
        trialnum_trial = np.full(nt.shape, tid, dtype=np.float32)
        prev_out_trial = np.full(nt.shape, prev_out, dtype=np.float32)
        inp = np.vstack([time_from_start, env_trial, trialnum_trial, prev_out_trial])

        reward_center = float(trial_reward_pos.get(tid, np.nan))
        dist_bins, _ = discretize_dist_to_reward(pos[bidx], reward_center)
        out = np.vstack([
            dist_bins,
            abs_pos_bins[bidx],
            speed_bins[bidx],
            (lick[bidx] > 0.5).astype(np.int64),
            np.full(nt.shape, int(trial_reward_loc.get(tid, 0)), dtype=np.int64),
            np.full(nt.shape, int(reward_outcomes[i]), dtype=np.int64),
        ]).astype(np.int64)

        session_neural.append(neural_trial)
        session_input.append(inp)
        session_output.append(out)
        prev_out = int(reward_outcomes[i])

    if show_processing:
        fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
        axs[0].plot(bt, pos, lw=0.5)
        axs[0].set_title(f'{subj} session {sess} position')
        axs[1].plot(bt, speed, lw=0.5)
        axs[1].set_title('speed')
        axs[2].plot(bt, lick, lw=0.5)
        axs[2].set_title('lick')
        axs[3].plot(bt, trial_num, lw=0.5)
        axs[3].set_title('trial number')
        fig.tight_layout()
        fig.savefig(f'processing_{subj}_{sess}.png', dpi=150)
        plt.close(fig)

    brain_region_idx = np.zeros(neural_nt.shape[0], dtype=np.int64)
    return {
        'subject': str(subj),
        'session': str(sess),
        'neural': session_neural,
        'input': session_input,
        'output': session_output,
        'brain_region_idx': brain_region_idx,
        'n_trials': len(session_neural),
        'zone_centers': reward_loc_centers.tolist() if hasattr(reward_loc_centers, 'tolist') else reward_loc_centers,
        'neural_source': f'{neural_base}/{neural_key}',
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true')
    g.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    args = ap.parse_args()

    files = sorted(Path('data').rglob('*.nwb'))
    if args.sample:
        # choose two diverse sessions: early env0 and later env1 if possible
        chosen = []
        env0 = None
        env1 = None
        for fp in files:
            with h5py.File(fp, 'r') as f:
                env = f['processing/behavior/BehavioralTimeSeries']['environment']['data'][:]
                tn = f['processing/behavior/BehavioralTimeSeries']['trial number']['data'][:]
                valid = tn >= 0
                vals = sorted(set(np.unique(env[valid]).tolist()))
                if vals == [0.0] and env0 is None:
                    env0 = fp
                if vals == [1.0] and env1 is None:
                    env1 = fp
                if env0 is not None and env1 is not None:
                    break
        files = [x for x in [env0, env1] if x is not None]
        if len(files) < 2:
            files = sorted(Path('data').rglob('*.nwb'))[:2]

    sessions = []
    for fp in files:
        print('Converting', fp, flush=True)
        sess = convert_session(fp, show_processing=args.show_processing)
        if sess['n_trials'] >= 2:
            sessions.append(sess)
        else:
            print('Skipping session with <2 trials after processing:', fp, flush=True)

    subjects = sorted(set(s['subject'] for s in sessions))
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    data = {
        'neural': [s['neural'] for s in sessions],
        'input': [s['input'] for s in sessions],
        'output': [s['output'] for s in sessions],
        'subjects': subjects,
        'subject_idx': np.asarray([subj_to_idx[s['subject']] for s in sessions], dtype=np.int64),
        'brain_regions': ['CA1'],
        'brain_region_idx': [s['brain_region_idx'] for s in sessions],
        'input_names': ['time_from_trial_start_sec', 'environment', 'trial_number', 'previous_trial_outcome'],
        'output_names': ['distance_to_reward_zone', 'absolute_position', 'speed', 'lick', 'reward_zone_location', 'reward_outcome'],
        'output_values': [
            ['lt_-50', '-50_to_-10', '-10_to_0', '0', '0_to_10', '10_to_50', 'gt_50'],
            [f'pos_bin_{i}' for i in range(5)],
            ['lt_2', '2_to_10', '10_to_20', '20_to_40', 'gt_40'],
            ['no', 'yes'],
            ['A', 'B', 'C'],
            ['no', 'yes'],
        ],
        'metadata': {
            'task_description': 'Virtual linear track reward-switch task; decode behavior and task variables from hippocampal imaging activity.',
            'time_bin_size': float(np.nan),
            'temporal_alignment_event': 'trial_start',
            'off_start': 0.0,
            'off_end': None,
            'n_sessions': len(sessions),
            'source_format': 'NWB',
        }
    }
    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    print('Saved', args.outpicklefile, 'with', len(sessions), 'sessions', flush=True)


if __name__ == '__main__':
    main()
