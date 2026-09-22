#!/usr/bin/env python3
import argparse
import pickle
import re
from pathlib import Path

import numpy as np

DATA_DIR = Path('/app/data')
BEH_DIR = DATA_DIR / 'beh'
SPK_DIR = DATA_DIR / 'spk'
SESSION_RE = re.compile(r'^[A-Z0-9]+_\d{4}_\d{2}_\d{2}_[0-9]+(?:_swap[0-9]+)?$')


def is_session_key(k):
    return isinstance(k, str) and SESSION_RE.match(k) is not None


def load_behavior_index():
    beh = {}
    for f in sorted(BEH_DIR.glob('*.npy')):
        obj = np.load(f, allow_pickle=True).item()
        for k, v in obj.items():
            if is_session_key(k):
                beh[k] = {'file': f.name, 'data': v}
    return beh


def load_neural_session(session_id):
    f = SPK_DIR / f'{session_id}_neural_data.npy'
    if not f.exists():
        return None
    mats = np.load(f, allow_pickle=True).item()['spks']
    return np.concatenate(mats, axis=0).astype(np.float32)


def get_subject(session_id):
    return session_id.split('_')[0]


def get_day_value(session_id, beh_file):
    name = beh_file.lower()
    if 'before_learning' in name or 'before_grating' in name:
        return 0.0
    if 'after_learning' in name or 'after_grating' in name:
        return 1.0
    m = re.search(r'train(\d+)', name)
    if m:
        return float(m.group(1))
    m = re.search(r'test(\d+)', name)
    if m:
        return float(m.group(1))
    return 0.0


def session_category_map(all_wall_names):
    cats = sorted(str(x) for x in np.unique(all_wall_names))
    return {c: i for i, c in enumerate(cats)}, cats


def discretize_speed(all_speeds):
    qs = np.quantile(all_speeds, [0.25, 0.5, 0.75])
    return qs


def apply_speed_bins(x, qs):
    return np.digitize(x, qs, right=False).astype(np.int64)


def convert_session(session_id, beh_entry, speed_qs, global_cat_map, max_trials=None):
    b = beh_entry['data']
    neural = load_neural_session(session_id)
    if neural is None:
        return None

    ft = np.asarray(b['ft'])
    ft_tr = np.asarray(b['ft_trInd'])
    ft_pos = np.asarray(b['ft_Pos'])
    ft_speed = np.asarray(b['ft_RunSpeed'])
    ft_wall = np.asarray(b['ft_WallID']).astype(str)
    T = min(neural.shape[1], len(ft), len(ft_tr), len(ft_pos), len(ft_speed), len(ft_wall))
    neural = neural[:, :T]
    ft = ft[:T]
    ft_tr = ft_tr[:T]
    ft_pos = ft_pos[:T]
    ft_speed = ft_speed[:T]
    ft_wall = ft_wall[:T]
    lick = np.zeros(T, dtype=np.int64)
    lick_fr = np.asarray(b.get('LickFr', []))
    if lick_fr.size:
        lick_fr = lick_fr.astype(int)
        lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < T)]
        lick[lick_fr] = 1

    valid = ~np.isnan(ft_tr)
    tr_ids = np.unique(ft_tr[valid].astype(int))
    if max_trials is not None:
        tr_ids = tr_ids[:max_trials]

    wall_names = np.asarray(b['WallName']).astype(str)
    is_rew = np.asarray(b['isRew']).astype(int)
    sound_times = np.asarray(b['SoundTime'])
    trial_start_times = np.asarray(b['Trial_start_time'])

    neural_trials = []
    input_trials = []
    output_trials = []

    for tr in tr_ids:
        idx = np.where(ft_tr.astype(float) == float(tr))[0]
        if idx.size < 2:
            continue
        t0 = ft[idx[0]]
        times = (ft[idx] - t0) * 24 * 3600

        time_to_cue = np.full(idx.size, np.nan, dtype=np.float32)
        if tr < len(sound_times):
            cue_rel = float((sound_times[tr] - trial_start_times[tr]) * 24 * 3600)
            time_to_cue = (cue_rel - times).astype(np.float32)

        day_val = np.full(idx.size, get_day_value(session_id, beh_entry['file']), dtype=np.float32)
        time_since = times.astype(np.float32)
        rew_avail = np.full(idx.size, int(is_rew[tr]) if tr < len(is_rew) else 0, dtype=np.float32)
        inp = np.vstack([time_to_cue, day_val, time_since, rew_avail])

        trial_wall = wall_names[tr] if tr < len(wall_names) else ft_wall[idx[0]]
        stim_cat = np.full(idx.size, global_cat_map.get(str(trial_wall), 0), dtype=np.int64)
        lick_out = lick[idx].astype(np.int64)
        pos_bin = np.clip((ft_pos[idx] // 1.0).astype(int), 0, 3).astype(np.int64)
        speed_bin = apply_speed_bins(ft_speed[idx], speed_qs)
        out = np.vstack([stim_cat, lick_out, pos_bin, speed_bin])

        neural_trials.append(neural[:, idx].astype(np.float32))
        input_trials.append(inp)
        output_trials.append(out)

    if len(neural_trials) < 2:
        return None
    return neural_trials, input_trials, output_trials, neural.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    args = ap.parse_args()

    beh = load_behavior_index()
    common_sessions = [sid for sid in sorted(beh.keys()) if (SPK_DIR / f'{sid}_neural_data.npy').exists()]
    if args.sample:
        scored = []
        for sid in common_sessions:
            b = beh[sid]['data']
            lick_n = len(np.asarray(b.get('LickFr', [])))
            rew_n = int(np.asarray(b.get('isRew', [])).sum()) if 'isRew' in b else 0
            scored.append((rew_n > 0, lick_n > 0, rew_n, lick_n, sid))
        scored.sort(reverse=True)
        common_sessions = [x[-1] for x in scored[:2]]

    print('n_common_sessions', len(common_sessions))

    all_wall = []
    all_speed = []
    for sid in common_sessions:
        b = beh[sid]['data']
        all_wall.extend([str(x) for x in np.asarray(b['WallName']).astype(str)])
        all_speed.extend(np.asarray(b['ft_RunSpeed'])[np.isfinite(np.asarray(b['ft_RunSpeed']))].tolist())
    cat_map, cat_names = session_category_map(np.array(all_wall))
    speed_qs = discretize_speed(np.asarray(all_speed))
    print('speed_quantiles', speed_qs)
    print('stim_categories', cat_names)

    subjects = sorted({get_subject(sid) for sid in common_sessions})
    subject_to_idx = {s: i for i, s in enumerate(subjects)}

    data = {
        'neural': [],
        'input': [],
        'output': [],
        'subjects': subjects,
        'subject_idx': [],
        'brain_regions': ['unknown_block0', 'unknown_block1', 'unknown_block2'],
        'brain_region_idx': [],
        'input_names': ['time_to_sound_cue', 'day_of_training', 'time_since_trial_start', 'reward_availability'],
        'output_names': ['visual_stimulus_category', 'licking', 'position_bin', 'running_speed_bin'],
        'output_values': [cat_names, ['no_lick', 'lick'], ['bin1', 'bin2', 'bin3', 'bin4'], ['q1', 'q2', 'q3', 'q4']],
        'metadata': {
            'task_description': 'Decode visual category, licking, position bin, and running speed bin from deconvolved calcium activity in VR corridor task.',
            'time_bin_size': float(np.median(np.diff(np.asarray(beh[common_sessions[0]]['data']['ft'])))) * 24 * 3600 * 1000 if common_sessions else None,
            'temporal_alignment_event': 'trial start / corridor entry',
            'off_start': 0.0,
            'off_end': None,
        }
    }

    for sid in common_sessions:
        print('processing_session', sid)
        converted = convert_session(sid, beh[sid], speed_qs, cat_map, max_trials=(20 if args.sample else None))
        if converted is None:
            print('skip_session', sid)
            continue
        neural_trials, input_trials, output_trials, nneu = converted
        data['neural'].append(neural_trials)
        data['input'].append(input_trials)
        data['output'].append(output_trials)
        data['subject_idx'].append(subject_to_idx[get_subject(sid)])
        data['brain_region_idx'].append(np.concatenate([
            np.full(19408 if '2022_07_12_1' in sid else nneu // 3, 0, dtype=int),
            np.full(19408 if '2022_07_12_1' in sid else nneu // 3, 1, dtype=int),
            np.full(nneu - 2 * (19408 if '2022_07_12_1' in sid else nneu // 3), 2, dtype=int),
        ]))
        print('converted_session', sid, 'n_trials', len(neural_trials), 'n_neurons', nneu)

    data['subject_idx'] = np.asarray(data['subject_idx'], dtype=int)

    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    print('wrote', args.outpicklefile)


if __name__ == '__main__':
    main()
