#!/usr/bin/env python3
import argparse
import os
import re
import time
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np

DATA_ROOT = Path('/app/data')
BEH_ROOT = DATA_ROOT / 'beh'
SPK_ROOT = DATA_ROOT / 'spk'
RET_ROOT = DATA_ROOT / 'retinotopy'

INPUT_NAMES = [
    'time_to_sound_cue',
    'day_of_training',
    'time_since_trial_start',
    'reward_availability',
]

OUTPUT_NAMES = [
    'visual_stimulus_category',
    'licking',
    'position_bin',
    'running_speed_bin',
]


def parse_session_id(session_id):
    parts = session_id.split('_')
    subj = parts[0]
    date = '_'.join(parts[1:4])
    run = parts[4] if len(parts) > 4 else None
    return subj, date, run


def normalize_session_for_retino(session_id):
    parts = session_id.split('_')
    return '_'.join(parts[:4])


def load_all_behavior_entries():
    entries = {}
    source_file = {}
    for f in sorted(BEH_ROOT.glob('*.npy')):
        dat = np.load(f, allow_pickle=True).item()
        for sid, sess in dat.items():
            entries[sid] = sess
            source_file[sid] = f.name
    return entries, source_file


def choose_behavior_for_spike_session(spike_sid, beh_entries):
    if spike_sid in beh_entries:
        return spike_sid
    candidates = [k for k in beh_entries if k == spike_sid or k.startswith(spike_sid + '_')]
    if candidates:
        for c in candidates:
            if 'swap' not in c:
                return c
        return candidates[0]
    return None


def load_spike_session(spike_sid):
    obj = np.load(SPK_ROOT / f'{spike_sid}_neural_data.npy', allow_pickle=True).item()
    spks = obj['spks']
    return np.concatenate([x.astype(np.float32, copy=False) for x in spks], axis=0)


def load_retino_for_session(spike_sid):
    rid = normalize_session_for_retino(spike_sid)
    path = RET_ROOT / f'{rid}_trans.npz'
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.keys()}


def area_names_and_idx(iarea):
    area_names = ['All', 'V1', 'medial', 'anterior', 'lateral']
    iarea = np.asarray(iarea)
    idx = np.zeros((len(iarea),), dtype=np.int64)
    idx[(iarea == 7) | (iarea == 8)] = 1
    idx[(iarea == 1) | (iarea == 2)] = 2
    idx[(iarea == 3) | (iarea == 4)] = 3
    idx[(iarea == 5) | (iarea == 6)] = 4
    return area_names, idx


def get_stimulus_categories(sess):
    if 'TrialStim' in sess:
        vals = np.asarray(sess['TrialStim']).astype(str)
        cats = sorted(np.unique(vals).tolist())
        mapping = {c: i for i, c in enumerate(cats)}
        return vals, cats, np.array([mapping[v] for v in vals], dtype=np.int64)
    if 'WallName' in sess:
        vals = np.asarray(sess['WallName']).astype(str)
        cats = sorted(np.unique(vals).tolist())
        mapping = {c: i for i, c in enumerate(cats)}
        return vals, cats, np.array([mapping[v] for v in vals], dtype=np.int64)
    raise KeyError('No TrialStim or WallName in behavior session')


def quartile_bins(x):
    x = np.asarray(x, dtype=float)
    qs = np.nanquantile(x[np.isfinite(x)], [0.25, 0.5, 0.75])
    return np.digitize(x, qs, right=False)


def infer_trial_frame_bounds(sess, n_frames):
    ntrials = int(sess['ntrials'])
    candidates = [k for k in sess.keys() if 'tr' in k.lower() and hasattr(sess[k], 'shape')]
    sound_fr = np.asarray(sess['SoundFr']).astype(int) if 'SoundFr' in sess else None
    if sound_fr is not None and sound_fr.shape[0] == ntrials:
        starts = np.zeros(ntrials, dtype=int)
        starts[1:] = np.maximum(sound_fr[1:] - np.maximum(np.diff(sound_fr), 1), 0)
        ends = np.empty(ntrials, dtype=int)
        ends[:-1] = np.maximum(starts[1:] - 1, starts[:-1])
        ends[-1] = n_frames - 1
        return starts, ends, {'method': 'soundfr_heuristic', 'candidates': candidates}
    edges = np.linspace(0, n_frames, ntrials + 1).astype(int)
    starts = edges[:-1]
    ends = edges[1:] - 1
    return starts, ends, {'method': 'uniform_fallback', 'candidates': candidates}


def build_trial_matrices(spk, sess, day_value, stim_cats_global, frame_stride=5):
    n_neurons, n_frames = spk.shape
    ntrials = int(sess['ntrials'])
    starts, ends, bounds_info = infer_trial_frame_bounds(sess, n_frames)
    stim_vals, stim_names_local, stim_ids = get_stimulus_categories(sess)
    stim_map_global = {name: i for i, name in enumerate(stim_cats_global)}
    stim_ids_global = np.array([stim_map_global[v] for v in stim_vals], dtype=np.int64)

    lick_fr = np.asarray(sess.get('LickFr', []), dtype=int)
    lick_tr = np.asarray(sess.get('LickTrind', []), dtype=int)
    sound_fr = np.asarray(sess.get('SoundFr', np.full(ntrials, -1)), dtype=int)
    run = np.asarray(sess.get('Run', sess.get('RunFr', np.zeros(n_frames))), dtype=float).reshape(-1)
    if run.shape[0] != n_frames:
        run = np.resize(run, n_frames)
    speed_bins = quartile_bins(run)

    pos = None
    for k in ['Pos', 'pos', 'AccumPos', 'AccPos', 'LickPos']:
        if k in sess and hasattr(sess[k], 'shape') and np.asarray(sess[k]).ndim == 1 and len(sess[k]) == n_frames:
            pos = np.asarray(sess[k], dtype=float)
            break
    if pos is None:
        pos = np.linspace(0, 4, n_frames)
    pos_bin = np.clip(np.floor(np.minimum(pos, 3.999999) / 1.0).astype(int), 0, 3)

    reward_avail = None
    if 'isRew' in sess:
        reward_avail = np.asarray(sess['isRew']).astype(int)
        if reward_avail.shape[0] != ntrials:
            reward_avail = np.resize(reward_avail, ntrials)
    else:
        reward_avail = np.zeros(ntrials, dtype=int)

    neural_trials, input_trials, output_trials = [], [], []
    for tr in range(ntrials):
        s = int(max(0, starts[tr]))
        e = int(min(n_frames - 1, max(starts[tr], ends[tr])))
        trial_spk = spk[:, s:e+1][:, ::frame_stride].astype(np.float16, copy=False)
        T = trial_spk.shape[1]
        ts = (np.arange(T, dtype=np.float32) * frame_stride)
        cue = sound_fr[tr] - s if tr < len(sound_fr) else -1
        time_to_cue = (cue - ts).astype(np.float32)
        inp = np.vstack([
            time_to_cue,
            np.full(T, float(day_value), dtype=np.float32),
            ts.astype(np.float32),
            np.full(T, float(reward_avail[tr]), dtype=np.float32),
        ])
        lick = np.zeros(T, dtype=np.int64)
        if lick_tr.size and lick_fr.size:
            idx = np.where(lick_tr == tr)[0]
            idx = idx[idx < lick_fr.shape[0]]
            if idx.size:
                lf = ((lick_fr[idx] - s) // frame_stride).astype(int)
                lf = lf[(lf >= 0) & (lf < T)]
                if lf.size:
                    lick[lf] = 1
        out = np.vstack([
            np.full(T, stim_ids_global[tr], dtype=np.int64),
            lick,
            pos_bin[s:e+1:frame_stride].astype(np.int64, copy=False),
            speed_bins[s:e+1:frame_stride].astype(np.int64, copy=False),
        ])
        neural_trials.append(trial_spk)
        input_trials.append(inp)
        output_trials.append(out)
    return neural_trials, input_trials, output_trials, bounds_info, stim_names_local


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true')
    mode.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    args = ap.parse_args()

    t0 = time.time()
    beh_entries, beh_source = load_all_behavior_entries()
    spike_sessions = sorted(f.name.replace('_neural_data.npy', '') for f in SPK_ROOT.glob('*_neural_data.npy'))

    matched = []
    for sid in spike_sessions:
        b = choose_behavior_for_spike_session(sid, beh_entries)
        if b is not None:
            matched.append((sid, b))

    rewarded = []
    for sid, b in matched:
        sess = beh_entries[b]
        isrew = np.asarray(sess.get('isRew', []))
        lickn = len(np.asarray(sess.get('LickFr', [])))
        if isrew.size and bool(isrew.any()) and lickn > 0:
            rewarded.append((sid, b))
    if args.sample:
        matched = rewarded[:2] if len(rewarded) >= 2 else matched[:2]
    else:
        matched = rewarded

    all_stim_names = set()
    for _, b in matched:
        sess = beh_entries[b]
        vals, cats, ids = get_stimulus_categories(sess)
        all_stim_names.update(cats)
    all_stim_names = sorted(all_stim_names)

    subjects = []
    subject_to_idx = {}
    brain_regions = ['All', 'V1', 'medial', 'anterior', 'lateral']

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
            all_stim_names,
            ['no_lick', 'lick'],
            ['bin0_0to1m', 'bin1_1to2m', 'bin2_2to3m', 'bin3_3to4m'],
            ['q1', 'q2', 'q3', 'q4'],
        ],
        'metadata': {
            'task_description': 'Visual discrimination in virtual reality corridors; decode stimulus category, licking, position bin, and running speed bin from neural activity.',
            'time_bin_size': 5.0,
            'temporal_alignment_event': 'trial start / corridor entry',
            'off_start': 0.0,
            'off_end': None,
            'notes': 'Neural activity loaded from processed deconvolved traces (`spks`) and concatenated across groups per session. Trial boundaries currently inferred heuristically and should be refined from behavior variables if available.',
        }
    }

    for day_value, (sid, bkey) in enumerate(matched):
        subj, date, run = parse_session_id(sid)
        if subj not in subject_to_idx:
            subject_to_idx[subj] = len(subjects)
            subjects.append(subj)
        spk = load_spike_session(sid)
        sess = beh_entries[bkey]
        neural_trials, input_trials, output_trials, bounds_info, stim_names_local = build_trial_matrices(spk, sess, day_value, all_stim_names, frame_stride=5)
        data['neural'].append(neural_trials)
        data['input'].append(input_trials)
        data['output'].append(output_trials)
        data['subject_idx'].append(subject_to_idx[subj])
        ret = load_retino_for_session(sid)
        if ret is not None and 'iarea' in ret:
            _, bri = area_names_and_idx(ret['iarea'])
            if len(bri) != spk.shape[0]:
                bri = np.resize(bri, spk.shape[0])
        else:
            bri = np.zeros((spk.shape[0],), dtype=np.int64)
        data['brain_region_idx'].append(bri)
        print(f'processed {sid} with behavior {bkey}: neurons={spk.shape[0]} frames={spk.shape[1]} trials={len(neural_trials)} bounds={bounds_info["method"]}')

    data['subject_idx'] = np.asarray(data['subject_idx'], dtype=np.int64)
    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    print(f'saved {args.outpicklefile} in {time.time()-t0:.2f}s with {len(data["neural"])} sessions')

if __name__ == '__main__':
    main()
