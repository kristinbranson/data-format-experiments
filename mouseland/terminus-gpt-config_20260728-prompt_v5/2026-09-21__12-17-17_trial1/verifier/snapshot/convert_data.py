#!/usr/bin/env python3
import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DATA_ROOT = Path('/app/data')
SPK_DIR = DATA_ROOT / 'spk'
BEH_DIR = DATA_ROOT / 'beh'
RET_DIR = DATA_ROOT / 'retinotopy'

INPUT_NAMES = [
    'time_to_sound_cue',
    'day_of_training',
    'time_since_trial_start',
    'reward_availability',
]

OUTPUT_NAMES = [
    'visual_stimulus_category',
    'licking',
    'corridor_position_bin',
    'running_speed_bin',
]


def parse_args():
    ap = argparse.ArgumentParser(description='Convert neuroscience dataset to decoder format')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true', help='Process all sessions (default)')
    mode.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    ap.add_argument('--show-processing', action='store_true', help='Save processing visualizations of up to 2 sessions')
    ap.add_argument('outpicklefile', help='Output pickle path')
    return ap.parse_args()


def load_behavior_maps():
    beh_maps = {}
    session_to_group = defaultdict(list)
    for f in sorted(BEH_DIR.glob('Beh_*.npy')):
        group = f.stem.replace('Beh_', '')
        d = np.load(f, allow_pickle=True).item()
        beh_maps[group] = d
        for sess in d:
            session_to_group[sess].append(group)
    return beh_maps, dict(session_to_group)


def load_spk_session(session_id):
    obj = np.load(SPK_DIR / f'{session_id}_neural_data.npy', allow_pickle=True).item()
    return np.concatenate(list(obj['spks']), axis=0).astype(np.float32)


def find_behavior_for_session(session_id, beh_maps, session_to_group):
    groups = session_to_group.get(session_id, [])
    if not groups:
        return None, None
    # Prefer supervised/unsupervised test/train groups over naive if multiple
    pref = sorted(groups, key=lambda g: (('sup' not in g and 'unsup' not in g), g))
    g = pref[0]
    return g, beh_maps[g][session_id]


def infer_session_day(session_id, ordered_sessions):
    return float(ordered_sessions.index(session_id) + 1)


def infer_rewarded_wallnames(beh):
    wall = np.asarray(beh['WallName'])
    isrew = np.asarray(beh['isRew']).astype(bool)
    uniq = np.asarray(beh['UniqWalls'])
    rewarded = set(wall[isrew].tolist())
    if not rewarded:
        return set()
    return rewarded


def safe_int_frames(x):
    return np.asarray(np.round(x), dtype=int)


def build_lick_vector(beh, start_fr, end_fr, trial_idx):
    n_t = end_fr - start_fr + 1
    lick = np.zeros(n_t, dtype=np.int64)
    lick_fr = np.asarray(beh['LickFr']).astype(int)
    lick_tr = np.asarray(beh['LickTrind']).astype(int)
    mask = lick_tr == int(trial_idx)
    rel = lick_fr[mask] - int(start_fr)
    rel = rel[(rel >= 0) & (rel < n_t)]
    if rel.size:
        lick[np.unique(rel)] = 1
    return lick


def build_trial_io(beh, trial_idx, start_fr, end_fr, day_val, stim_to_idx, speed_edges, rewarded_wallnames):
    n_t = end_fr - start_fr + 1
    frame_idx = np.arange(start_fr, end_fr + 1)
    sound_fr = int(round(float(beh['SoundFr'][trial_idx])))
    time_to_cue = (sound_fr - frame_idx).astype(np.float32)
    time_since_start = np.arange(n_t, dtype=np.float32)
    wall_name = str(beh['WallName'][trial_idx])
    reward_avail = np.full(n_t, int(wall_name in rewarded_wallnames), dtype=np.float32)
    day_arr = np.full(n_t, day_val, dtype=np.float32)
    inp = np.vstack([time_to_cue, day_arr, time_since_start, reward_avail]).astype(np.float32)

    stim_name = str(beh['TrialStim'][trial_idx])
    if stim_name == 'stimulus_of_trial':
        stim_name = str(beh['WallName'][trial_idx])
    stim_idx = stim_to_idx[stim_name]
    stim_arr = np.full(n_t, stim_idx, dtype=np.int64)

    lick = build_lick_vector(beh, start_fr, end_fr, trial_idx)

    ft_pos = np.asarray(beh['ft_Pos'], dtype=float)
    ft_speed = np.asarray(beh['ft_RunSpeed'], dtype=float)
    pos_seg = ft_pos[start_fr:end_fr + 1]
    speed_seg = ft_speed[start_fr:end_fr + 1]

    corridor_len = float(beh.get('Corridor_Length', 40.0))
    bin_w = corridor_len / 4.0
    pos_bin = np.clip(np.floor(pos_seg / bin_w).astype(int), 0, 3)
    speed_bin = np.digitize(speed_seg, speed_edges[1:-1], right=False).astype(int)
    speed_bin = np.clip(speed_bin, 0, 3)

    out = np.vstack([
        stim_arr,
        lick.astype(np.int64),
        pos_bin.astype(np.int64),
        speed_bin.astype(np.int64),
    ])
    return inp, out


def collect_speed_values(session_ids, beh_maps, session_to_group):
    vals = []
    for sess in session_ids:
        _, beh = find_behavior_for_session(sess, beh_maps, session_to_group)
        if beh is None:
            continue
        v = np.asarray(beh['ft_RunSpeed'], dtype=float)
        m = np.isfinite(v)
        if 'ft_CorrSpc' in beh:
            m &= np.asarray(beh['ft_CorrSpc'], dtype=bool)
        if 'ft_isMoving' in beh:
            m &= np.asarray(beh['ft_isMoving'], dtype=bool)
        vals.append(v[m])
    if not vals:
        return np.array([0, 1, 2, 3, 4], dtype=float)
    allv = np.concatenate(vals)
    qs = np.quantile(allv, [0, 0.25, 0.5, 0.75, 1.0])
    qs[-1] = np.nextafter(qs[-1], np.inf)
    return qs


def main():
    args = parse_args()
    if not args.full and not args.sample:
        args.full = True

    beh_maps, session_to_group = load_behavior_maps()
    spk_sessions = sorted(p.stem.replace('_neural_data', '') for p in SPK_DIR.glob('*_neural_data.npy'))
    matched = [s for s in spk_sessions if s in session_to_group]
    if args.sample:
        sup = [s for s in matched if any('sup' in g and 'unsup' not in g for g in session_to_group.get(s, []))]
        unsup = [s for s in matched if any('unsup' in g for g in session_to_group.get(s, []))]
        pick = []
        if sup:
            pick.append(sorted(sup)[0])
        if unsup:
            cand = sorted(unsup)[0]
            if cand not in pick:
                pick.append(cand)
        for s in matched:
            if len(pick) >= 2:
                break
            if s not in pick:
                pick.append(s)
        matched = pick[:2]

    print('spk sessions:', len(spk_sessions))
    print('matched sessions:', len(matched))

    # stimulus vocabulary and speed bins from selected sessions
    stim_names = sorted({(lambda beh,i: (str(beh['WallName'][i]) if str(beh['TrialStim'][i]) == 'stimulus_of_trial' else str(beh['TrialStim'][i])))(find_behavior_for_session(s, beh_maps, session_to_group)[1], i)
                         for s in matched
                         for i in range(int(find_behavior_for_session(s, beh_maps, session_to_group)[1]['ntrials']))})
    stim_to_idx = {s: i for i, s in enumerate(stim_names)}
    speed_edges = collect_speed_values(matched, beh_maps, session_to_group)
    print('stim categories:', stim_names)
    print('speed edges:', speed_edges)

    subjects = sorted({s.split('_')[0] for s in matched})
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    brain_regions = ['unknown_group0', 'unknown_group1', 'unknown_group2']

    data = {
        'neural': [],
        'input': [],
        'output': [],
        'subjects': subjects,
        'subject_idx': [],
        'brain_regions': brain_regions,
        'brain_region_idx': [],
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': [
            stim_names,
            ['no_lick', 'lick'],
            ['0-1m', '1-2m', '2-3m', '3-4m'],
            ['q1', 'q2', 'q3', 'q4'],
        ],
        'metadata': {
            'task_description': 'Visual discrimination in 4 m virtual corridors with cue and reward; decode stimulus, licking, position bin, and running speed bin from neural activity.',
            'time_bin_size': 1.0,
            'temporal_alignment_event': 'trial start / corridor entry',
            'off_start': 0.0,
            'off_end': None,
            'notes': 'Neural traces are deconvolved activity; sessions reconstructed from full-session traces segmented by behavior StartFr/EndFr.',
        },
    }

    kept_sessions = []
    for sess in matched:
        group, beh = find_behavior_for_session(sess, beh_maps, session_to_group)
        spk = load_spk_session(sess)
        n_neu, n_fr = spk.shape
        ft_tr = np.asarray(beh['ft_trInd'])[:n_fr]
        ft_corr = np.asarray(beh['ft_CorrSpc'])[:n_fr].astype(bool)
        valid = []
        for tr in range(int(beh['ntrials'])):
            cols = np.flatnonzero((ft_tr == tr) & ft_corr)
            if cols.size < 2:
                continue
            valid.append((tr, cols))
        if len(valid) < 2:
            print('skip', sess, 'valid_trials', len(valid), 'shape', spk.shape)
            continue

        sess_neural, sess_input, sess_output = [], [], []
        ordered_sessions = sorted([s for s in matched if s.split('_')[0] == sess.split('_')[0]], key=lambda x: tuple(x.split('_')[1:4]))
        day_val = infer_session_day(sess, ordered_sessions)
        rewarded_wallnames = infer_rewarded_wallnames(beh)
        for tr, cols in valid:
            st, en = int(cols[0]), int(cols[-1])
            sess_neural.append(spk[:, cols].astype(np.float32))
            inp, out = build_trial_io(beh, tr, st, en, day_val, stim_to_idx, speed_edges, rewarded_wallnames)
            # remap framewise outputs/inputs to corridor-frame columns only
            inp = inp[:, cols - st]
            out = out[:, cols - st]
            sess_input.append(inp)
            sess_output.append(out)

        # brain region idx: split concatenated neurons into 3 groups as placeholders
        raw = np.load(SPK_DIR / f'{sess}_neural_data.npy', allow_pickle=True).item()['spks']
        group_sizes = [arr.shape[0] for arr in raw]
        bri = np.concatenate([np.full(sz, i, dtype=np.int64) for i, sz in enumerate(group_sizes)])

        data['neural'].append(sess_neural)
        data['input'].append(sess_input)
        data['output'].append(sess_output)
        data['subject_idx'].append(subj_to_idx[sess.split('_')[0]])
        data['brain_region_idx'].append(bri)
        kept_sessions.append((sess, group, len(valid), n_neu, n_fr))
        print('kept', sess, 'group', group, 'trials', len(valid), 'neurons', n_neu, 'frames', n_fr)

    data['subject_idx'] = np.asarray(data['subject_idx'], dtype=np.int64)
    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)

    print('saved', args.outpicklefile)
    print('n_sessions_out', len(data['neural']))
    print('n_subjects_out', len(data['subjects']))
    print('session_trial_counts', [len(x) for x in data['neural']])


if __name__ == '__main__':
    main()
