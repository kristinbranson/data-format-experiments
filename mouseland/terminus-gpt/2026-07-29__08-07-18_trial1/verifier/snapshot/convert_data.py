#!/usr/bin/env python3
import argparse
import pickle
import re
import time
from pathlib import Path

import numpy as np

SESSION_RE = re.compile(r'^[A-Za-z0-9]+_\d{4}_\d{2}_\d{2}_\d+(?:_swap[12])?$')


def _behavior_richness(dat):
    score = len(dat)
    if 'ft_trInd' in dat:
        try:
            score += int(np.asarray(dat['ft_trInd']).size)
        except Exception:
            pass
    return score


def load_behavior_sessions(data_dir: Path):
    sessions = {}
    session_source = {}
    for p in sorted((data_dir / 'beh').glob('*.npy')):
        obj = np.load(p, allow_pickle=True).item()
        if not isinstance(obj, dict):
            continue
        for sess, dat in obj.items():
            if isinstance(sess, str) and SESSION_RE.match(sess) and isinstance(dat, dict):
                if sess not in sessions or _behavior_richness(dat) > _behavior_richness(sessions[sess]):
                    sessions[sess] = dat
                    session_source[sess] = p.name
    return sessions, session_source


def load_neural_files(data_dir: Path):
    files = {}
    for p in sorted(data_dir.rglob('*_neural_data.npy')):
        files[p.name.replace('_neural_data.npy', '')] = p
    return files


def concat_spks(spks_list):
    return np.concatenate([np.asarray(x, dtype=np.float32) for x in spks_list], axis=0)


def infer_day(source_name: str):
    name = source_name.lower()
    if 'before_learning' in name or 'before_grating' in name:
        return 1.0
    if 'after_learning' in name or 'after_grating' in name:
        return 5.0
    m = re.search(r'test(\d+)', name)
    if m:
        return float(m.group(1))
    m = re.search(r'train(\d+)', name)
    if m:
        return float(m.group(1))
    return 0.0


def build_session(beh, beh_source, neural_path):
    nobj = np.load(neural_path, allow_pickle=True).item()
    spk = concat_spks(nobj['spks'])

    ft_tr = np.asarray(beh['ft_trInd'], dtype=float)
    ft_pos = np.asarray(beh['ft_PosCum'], dtype=float)
    ft_move = np.asarray(beh['ft_move'], dtype=float)
    ft_wall = np.asarray(beh['ft_WallID'])
    n_frames = min(spk.shape[1], len(ft_tr), len(ft_pos), len(ft_move), len(ft_wall))
    spk = spk[:, :n_frames]
    ft_tr = ft_tr[:n_frames]
    ft_pos = ft_pos[:n_frames]
    ft_move = ft_move[:n_frames]
    ft_wall = ft_wall[:n_frames]

    valid_frame = np.isfinite(ft_tr)
    ft_tr_int = np.full(n_frames, -1, dtype=int)
    ft_tr_int[valid_frame] = ft_tr[valid_frame].astype(int)

    wall = np.asarray(beh['WallName'])
    is_rew = np.asarray(beh['isRew']).astype(np.int64)
    tstart = np.asarray(beh['Trial_start_time'], dtype=float)
    tend = np.asarray(beh['Trial_end_time'], dtype=float)
    sound = np.asarray(beh['SoundTime'], dtype=float)
    ntrials = int(np.asarray(beh['ntrials']).item())
    corridor_len_arr = np.asarray(beh.get('Corridor_Length', 4.0))
    corridor_len = float(corridor_len_arr.item()) if corridor_len_arr.shape == () else 4.0
    day = infer_day(beh_source)

    trial_dur_sec = (tend - tstart) * 86400.0
    frame_periods = []
    for tr in range(min(ntrials, len(trial_dur_sec))):
        nfr = np.sum(ft_tr_int == tr)
        if nfr > 1 and np.isfinite(trial_dur_sec[tr]) and trial_dur_sec[tr] > 0:
            frame_periods.append(trial_dur_sec[tr] / nfr)
    dt_frame = float(np.nanmedian(frame_periods)) if frame_periods else 0.1

    valid_idx = np.flatnonzero(ft_tr_int >= 0)
    valid_tr = ft_tr_int[valid_idx]
    uniq_tr, starts = np.unique(valid_tr, return_index=True)
    trial_frame_idx = {int(tr): valid_idx[starts[i]:starts[i+1]] if i+1 < len(starts) else valid_idx[starts[i]:] for i, tr in enumerate(uniq_tr)}

    lick_series = np.zeros(n_frames, dtype=np.int64)
    lick_tr = np.asarray(beh.get('LickTrind', []))
    lick_time = np.asarray(beh.get('LickTime', []), dtype=float)
    if lick_tr.size and lick_time.size:
        lick_tr_i = lick_tr.astype(int)
        for tr in np.unique(lick_tr_i):
            idx = trial_frame_idx.get(int(tr))
            if idx is None or idx.size == 0 or tr >= len(tstart):
                continue
            rel = (lick_time[lick_tr_i == tr] - tstart[int(tr)]) * 86400.0
            rel = rel[np.isfinite(rel)]
            if rel.size == 0:
                continue
            bins = np.clip(np.floor(rel / dt_frame).astype(int), 0, idx.size - 1)
            lick_series[idx[np.unique(bins)]] = 1

    neural_trials, input_trials, output_trials = [], [], []
    all_speeds = []
    cache = []
    for tr, idx in trial_frame_idx.items():
        if idx.size < 2:
            continue
        if tr >= len(wall) or tr >= len(is_rew) or tr >= len(sound) or tr >= len(tstart):
            continue
        tr_spk = spk[:, idx].astype(np.float32, copy=False)
        pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
        time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
        sound_sec = float((sound[tr] - tstart[tr]) * 86400.0) if np.isfinite(sound[tr]) and np.isfinite(tstart[tr]) else np.nan
        time_to_sound = (sound_sec - time_since).astype(np.float32) if np.isfinite(sound_sec) else np.full(idx.size, np.nan, dtype=np.float32)
        day_arr = np.full(idx.size, day, dtype=np.float32)
        reward_arr = np.full(idx.size, int(is_rew[tr]), dtype=np.float32)
        inp = np.vstack([time_to_sound, day_arr, time_since, reward_arr]).astype(np.float32)

        pos_bin = np.clip((pos / max(corridor_len, 1e-6) * 4).astype(int), 0, 3)
        speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
        lick = lick_series[idx]
        cache.append((tr_spk, inp, str(wall[tr]), lick.astype(np.int64), pos_bin.astype(np.int64), speed.astype(np.float32)))
        all_speeds.append(speed)

    if not cache:
        return [], [], []

    speed_all = np.concatenate(all_speeds)
    qs = np.quantile(speed_all, [0.25, 0.5, 0.75]) if speed_all.size else np.array([0, 0, 0], dtype=np.float32)
    for tr_spk, inp, wall_name, lick, pos_bin, speed in cache:
        speed_bin = np.digitize(speed, qs, right=False).astype(np.int64)
        neural_trials.append(tr_spk)
        input_trials.append(inp)
        output_trials.append((wall_name, np.vstack([
            np.full(tr_spk.shape[1], -1, dtype=np.int64),
            lick,
            pos_bin,
            speed_bin,
        ])))
    return neural_trials, input_trials, output_trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    ap.add_argument('--full', action='store_true', default=False)
    ap.add_argument('--sample', action='store_true', default=False)
    ap.add_argument('--show-processing', action='store_true', default=False)
    args = ap.parse_args()
    if not args.full and not args.sample:
        args.full = True

    t0 = time.time()
    data_dir = Path('data')
    beh_sessions, beh_source = load_behavior_sessions(data_dir)
    neural_files = load_neural_files(data_dir)
    common = sorted(set(beh_sessions) & set(neural_files))
    if args.sample:
        def sample_score(sess):
            d = beh_sessions[sess]
            lick_n = len(np.asarray(d.get('LickTime', [])))
            rew = np.asarray(d.get('isRew', []))
            rew_frac = float(np.mean(rew.astype(float))) if rew.size else 0.0
            return (lick_n, rew_frac)
        common = sorted(common, key=sample_score, reverse=True)[:2]
    print(f'Loaded {len(beh_sessions)} behavior sessions, {len(neural_files)} neural sessions, {len(common)} common sessions')

    wall_names = sorted({str(w) for s in common for w in np.asarray(beh_sessions[s]['WallName'])})
    wall_to_id = {w: i for i, w in enumerate(wall_names)}
    subjects = sorted({s.split('_')[0] for s in common})
    subj_to_id = {s: i for i, s in enumerate(subjects)}

    data = {
        'neural': [],
        'input': [],
        'output': [],
        'subjects': subjects,
        'subject_idx': [],
        'brain_regions': ['unknown'],
        'brain_region_idx': [],
        'input_names': ['time_to_sound_cue', 'day_of_training', 'time_since_trial_start', 'reward_availability'],
        'output_names': ['visual_stimulus_category', 'licking', 'corridor_position_bin', 'running_speed_bin'],
        'output_values': [wall_names, ['no_lick', 'lick'], ['bin1', 'bin2', 'bin3', 'bin4'], ['q1', 'q2', 'q3', 'q4']],
        'metadata': {
            'task_description': 'Visual corridor task with rewarded and unrewarded corridors; decode stimulus, licking, position, and running speed from deconvolved neural activity.',
            'time_bin_size': None,
            'temporal_alignment_event': 'corridor entry / trial start',
            'off_start': 0.0,
            'off_end': None,
        },
    }

    kept = 0
    for sess in common:
        neural_trials, input_trials, output_trials = build_session(beh_sessions[sess], beh_source[sess], neural_files[sess])
        if len(neural_trials) < 2:
            continue
        sess_out = []
        for wall_name, out in output_trials:
            out = out.copy()
            out[0, :] = wall_to_id[wall_name]
            sess_out.append(out)
        data['neural'].append(neural_trials)
        data['input'].append(input_trials)
        data['output'].append(sess_out)
        data['subject_idx'].append(subj_to_id[sess.split('_')[0]])
        data['brain_region_idx'].append(np.zeros(neural_trials[0].shape[0], dtype=np.int64))
        kept += 1
        print('kept', sess, 'trials', len(neural_trials), 'neurons', neural_trials[0].shape[0])

    data['subject_idx'] = np.asarray(data['subject_idx'], dtype=np.int64)
    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    print(f'Wrote {args.outpicklefile} with {kept} sessions in {time.time()-t0:.2f}s')


if __name__ == '__main__':
    main()
