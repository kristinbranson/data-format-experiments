import argparse
import os
import pickle
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DATA_DIR = Path('data')
BEH_DIR = DATA_DIR / 'beh'
SPK_DIR = DATA_DIR / 'spk'
RET_DIR = DATA_DIR / 'retinotopy'


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true', help='Process all sessions')
    g.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    ap.add_argument('--show-processing', action='store_true', help='Save processing plots for up to 2 sessions')
    return ap.parse_args()


def load_all_behavior():
    beh_by_session = {}
    beh_source = {}
    for bf in sorted(BEH_DIR.glob('Beh_*.npy')):
        obj = np.load(bf, allow_pickle=True).item()
        for k, v in obj.items():
            beh_by_session[k] = v
            beh_source[k] = bf.name
    return beh_by_session, beh_source


def base_session_name(session):
    if session.endswith('_swap1') or session.endswith('_swap2'):
        return session.rsplit('_', 1)[0]
    return session


def load_spk_session(session_base):
    fn = SPK_DIR / f'{session_base}_neural_data.npy'
    obj = np.load(fn, allow_pickle=True).item()
    spk = np.concatenate([x for x in obj['spks']], axis=0).astype(np.float32)
    return spk


def load_retino_for_session(session_base):
    parts = session_base.split('_')
    mouse = parts[0]
    datexp = '_'.join(parts[1:4])
    fn = RET_DIR / f'{mouse}_{datexp}_trans.npz'
    if not fn.exists():
        return None
    r = np.load(fn, allow_pickle=True)
    return {k: r[k] for k in r.files}


def neu_area_ID(iarea):
    idx = {}
    iarea = np.asarray(iarea)
    for ar in ['all', 'V1', 'medial', 'anterior', 'lateral', 'aHV']:
        if ar == 'all':
            idx[ar] = np.ones(len(iarea), dtype=bool)
        elif ar == 'V1':
            idx[ar] = iarea == 0
        elif ar == 'medial':
            idx[ar] = iarea == 1
        elif ar == 'anterior':
            idx[ar] = iarea == 2
        elif ar == 'lateral':
            idx[ar] = iarea == 3
        elif ar == 'aHV':
            idx[ar] = (iarea == 3) | (iarea == 4)
    return idx


def get_cat_id(WallName, isRew):
    uniqW = np.unique(WallName)
    rewStim = WallName[isRew][0]
    cid = np.zeros(len(uniqW))
    cid[uniqW == rewStim] = 2
    if rewStim[-1] == '1':
        cid[uniqW == rewStim[:-1] + '2'] = 3
    nrewStim = uniqW[cid == 0][0]
    if nrewStim[-1] == '1':
        cid[uniqW == nrewStim[:-1] + '2'] = 1
    return cid.astype(int)


def session_day_value(session_name):
    parts = session_name.split('_')
    y, m, d = map(int, parts[1:4])
    return y * 10000 + m * 100 + d


def matlab_days_to_seconds(x):
    return np.asarray(x, dtype=float) * 24.0 * 3600.0


def build_global_mappings(beh_by_session, selected_sessions):
    stim_names = set()
    for sess in selected_sessions:
        beh = beh_by_session[sess]
        if 'TrialStim' in beh:
            vals = [str(x) for x in np.unique(beh['TrialStim']) if str(x) != 'stimulus_of_trial']
            stim_names.update(vals)
    stim_names = sorted(stim_names)
    stim_to_idx = {s: i for i, s in enumerate(stim_names)}
    return stim_names, stim_to_idx


def compute_speed_edges(beh_by_session, selected_sessions):
    vals = []
    for sess in selected_sessions:
        x = np.asarray(beh_by_session[sess]['ft_RunSpeed'], dtype=float)
        x = x[np.isfinite(x)]
        if x.size:
            vals.append(x)
    allv = np.concatenate(vals) if vals else np.array([0.0, 1.0])
    edges = np.quantile(allv, [0.25, 0.5, 0.75])
    return edges.astype(float)


def discretize_position(pos, corridor_length):
    pos = np.asarray(pos, dtype=float)
    bins = np.floor(np.clip(pos, 0, corridor_length - 1e-8) / (corridor_length / 4.0)).astype(int)
    return np.clip(bins, 0, 3)


def discretize_speed(speed, edges):
    speed = np.asarray(speed, dtype=float)
    out = np.zeros(speed.shape, dtype=int)
    out[speed > edges[0]] = 1
    out[speed > edges[1]] = 2
    out[speed > edges[2]] = 3
    return out

def resample_matrix_time(mat, n_bins=60):
    mat = np.asarray(mat)
    if mat.ndim == 1:
        mat = mat[None, :]
    t = mat.shape[1]
    if t == n_bins:
        return mat.astype(np.float32, copy=False)
    if t == 1:
        return np.repeat(mat, n_bins, axis=1).astype(np.float32, copy=False)
    idx = np.round(np.linspace(0, t - 1, n_bins)).astype(int)
    return mat[:, idx].astype(np.float32, copy=False)


def resample_labels_1d(arr, n_bins=60):
    arr = np.asarray(arr)
    t = arr.shape[0]
    if t == n_bins:
        return arr
    if t == 1:
        return np.repeat(arr, n_bins)
    idx = np.round(np.linspace(0, t - 1, n_bins)).astype(int)
    return arr[idx]


def extract_session(session_name, beh, stim_to_idx, speed_edges, show_processing=False):
    t0 = time.time()
    base = base_session_name(session_name)
    spk = load_spk_session(base)

    frame_keys = ['ft', 'ft_trInd', 'ft_Pos', 'ft_PosCum', 'ft_RunSpeed', 'BefCueFr', 'AftCueFr']
    frame_lengths = [len(np.asarray(beh[k])) for k in frame_keys if k in beh]
    frame_lengths.append(spk.shape[1])
    nfr = min(frame_lengths)

    spk = spk[:, :nfr]
    ft = matlab_days_to_seconds(np.asarray(beh['ft'], dtype=float)[:nfr])
    ft_trInd = np.asarray(beh['ft_trInd'], dtype=float)[:nfr]
    ft_pos = np.asarray(beh['ft_Pos'], dtype=float)[:nfr]
    ft_speed = np.asarray(beh['ft_RunSpeed'], dtype=float)[:nfr]

    valid = np.isfinite(ft_trInd)
    tr_idx = ft_trInd[valid].astype(int)
    uniq_trials = np.unique(tr_idx)

    ntrials_declared = int(np.asarray(beh['ntrials']).item())
    trialstim = np.asarray(beh['TrialStim']).astype(str)
    isrew = np.asarray(beh['isRew']).astype(bool)
    soundtime = matlab_days_to_seconds(np.asarray(beh['SoundTime'], dtype=float))
    trialstart = matlab_days_to_seconds(np.asarray(beh['Trial_start_time'], dtype=float))
    corridor_length = float(beh.get('Corridor_Length', 40.0))

    neural_trials = []
    input_trials = []
    output_trials = []
    kept_trial_ids = []

    day_val = float(session_day_value(base))

    for tr in uniq_trials:
        if tr < 0 or tr >= ntrials_declared:
            continue
        mask = valid.copy()
        mask[valid] = tr_idx == tr
        if mask.sum() < 2:
            continue

        nmat_raw = spk[:, mask].astype(np.float32)
        tvec = ft[mask]
        pos = ft_pos[mask]
        speed = ft_speed[mask]

        time_since_start = tvec - tvec[0]
        cue_rel = soundtime[tr] - tvec
        reward_available = np.full(mask.sum(), 1.0 if isrew[tr] else 0.0, dtype=np.float32)
        day_arr = np.full(mask.sum(), float(day_val), dtype=np.float32)
        inp_raw = np.vstack([
            cue_rel.astype(np.float32),
            day_arr,
            time_since_start.astype(np.float32),
            reward_available,
        ])

        nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
        inp = resample_matrix_time(inp_raw, n_bins=60).astype(np.float32)

        stim_idx = stim_to_idx[str(trialstim[tr])]
        lick = np.zeros(mask.sum(), dtype=np.uint8)
        if len(beh['LickTrind']) > 0:
            lick_times = np.asarray(beh['LickTime'])[np.asarray(beh['LickTrind']) == tr]
            if len(lick_times) > 0:
                inds = np.searchsorted(tvec, lick_times, side='left')
                inds = inds[(inds >= 0) & (inds < len(tvec))]
                lick[inds] = 1
        pos_bin = discretize_position(pos, corridor_length)
        speed_bin = discretize_speed(speed, speed_edges)
        stim_arr = np.full(mask.sum(), stim_idx, dtype=np.uint8)
        out_raw = np.vstack([stim_arr, lick, pos_bin.astype(np.uint8), speed_bin.astype(np.uint8)])
        out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60) for i in range(out_raw.shape[0])]).astype(np.uint8)

        neural_trials.append(nmat)
        input_trials.append(inp)
        output_trials.append(out)
        kept_trial_ids.append(int(tr))

    if show_processing:
        fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
        axs[0].plot(ft[:min(2000, len(ft))], ft_pos[:min(2000, len(ft))])
        axs[0].set_title(f'{session_name}: position vs time')
        axs[1].plot(ft[:min(2000, len(ft))], ft_speed[:min(2000, len(ft))])
        axs[1].set_title('running speed')
        if neural_trials:
            ex = neural_trials[0][:min(20, neural_trials[0].shape[0])].astype(float)
            axs[2].imshow(ex, aspect='auto', interpolation='nearest')
            axs[2].set_title('example neural trial (first 20 neurons)')
            axs[3].imshow(output_trials[0], aspect='auto', interpolation='nearest')
            axs[3].set_title('example output trial')
        fig.tight_layout()
        fig.savefig(f'processing_{session_name}.png', dpi=150)
        plt.close(fig)

    info = {
        'session_name': session_name,
        'base_session': base,
        'n_frames_common': int(nfr),
        'n_trials_kept': len(neural_trials),
        'elapsed_sec': time.time() - t0,
    }
    return neural_trials, input_trials, output_trials, info, spk.shape[0]


def build_brain_region_idx(session_name, n_neurons):
    base = base_session_name(session_name)
    ret = load_retino_for_session(base)
    if ret is None or 'iarea' not in ret:
        return np.zeros(n_neurons, dtype=np.int64), ['unknown']
    iarea = np.asarray(ret['iarea']).astype(int)
    labels = ['V1' if x == 0 else 'medial' if x == 1 else 'anterior' if x == 2 else 'lateral' if x == 3 else 'aHV' if x == 4 else 'unknown' for x in iarea]
    uniq = []
    for x in labels:
        if x not in uniq:
            uniq.append(x)
    lab_to_idx = {x: i for i, x in enumerate(uniq)}
    idx = np.array([lab_to_idx[x] for x in labels], dtype=np.int64)
    if len(idx) != n_neurons:
        m = min(len(idx), n_neurons)
        idx = idx[:m]
        if m < n_neurons:
            pad = np.full(n_neurons - m, lab_to_idx.get('unknown', 0), dtype=np.int64)
            idx = np.concatenate([idx, pad])
    return idx, uniq


def main():
    args = parse_args()
    if not args.full and not args.sample:
        args.full = True

    beh_by_session, beh_source = load_all_behavior()
    spk_sessions = set(p.name.replace('_neural_data.npy', '') for p in SPK_DIR.glob('*_neural_data.npy'))

    selected = []
    for sess in sorted(beh_by_session.keys()):
        base = base_session_name(sess)
        if base not in spk_sessions:
            continue
        ts = np.asarray(beh_by_session[sess].get('TrialStim', [])).astype(str)
        if ts.size and 'stimulus_of_trial' in set(ts.tolist()):
            continue
        selected.append(sess)

    if args.sample:
        informative = []
        for sess in selected:
            beh = beh_by_session[sess]
            nrew = int(np.asarray(beh.get('isRew', []), dtype=int).sum()) if 'isRew' in beh else 0
            nlick = len(beh.get('LickTime', []))
            informative.append((nrew > 0 and nlick > 0, nrew, nlick, sess))
        informative.sort(reverse=True)
        selected = [x[3] for x in informative[:2]]

    print(f'selected_sessions={len(selected)}')

    stim_names, stim_to_idx = build_global_mappings(beh_by_session, selected)
    speed_edges = compute_speed_edges(beh_by_session, selected)
    print('stim_names', stim_names)
    print('speed_edges', speed_edges)

    subjects = []
    subj_to_idx = {}
    subject_idx = []
    all_brain_regions = []
    brain_region_idx = []
    neural = []
    inputs = []
    outputs = []
    session_info = []

    for i, sess in enumerate(selected):
        print(f'processing {i+1}/{len(selected)}: {sess}')
        subj = sess.split('_')[0]
        if subj not in subj_to_idx:
            subj_to_idx[subj] = len(subjects)
            subjects.append(subj)
        nt, it, ot, info, n_neu = extract_session(sess, beh_by_session[sess], stim_to_idx, speed_edges, show_processing=args.show_processing and i < 2)
        if len(nt) < 2:
            print(f'skipping {sess}: fewer than 2 valid trials')
            continue
        neural.append(nt)
        inputs.append(it)
        outputs.append(ot)
        subject_idx.append(subj_to_idx[subj])
        bri, br_names = build_brain_region_idx(sess, n_neu)
        local_to_global = {}
        for b in br_names:
            if b not in all_brain_regions:
                all_brain_regions.append(b)
            local_to_global[br_names.index(b)] = all_brain_regions.index(b)
        brain_region_idx.append(np.array([local_to_global[x] for x in bri], dtype=np.int64))
        session_info.append(info)
        print(f"completed {sess}: trials={info['n_trials_kept']} frames={info['n_frames_common']} neurons={n_neu} elapsed={info['elapsed_sec']:.2f}s")

    data = {
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'subjects': subjects,
        'subject_idx': np.array(subject_idx, dtype=np.int64),
        'brain_regions': all_brain_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': ['time_to_sound_cue', 'day_of_training', 'time_since_trial_start', 'reward_availability'],
        'output_names': ['visual_stimulus_category', 'licking', 'position_bin', 'running_speed_bin'],
        'output_values': [
            stim_names,
            ['no_lick', 'lick'],
            ['bin0', 'bin1', 'bin2', 'bin3'],
            ['q1', 'q2', 'q3', 'q4'],
        ],
        'metadata': {
            'task_description': 'Virtual linear corridor task with visual textures, sound cue, reward contingency, licking, running, and position decoding targets.',
            'time_bin_size': None,
            'temporal_alignment_event': 'trial start / corridor entry',
            'off_start': 0.0,
            'off_end': None,
            'reference_processing': 'Neural traces concatenated across planes and segmented on frame-wise trial indices; reference code uses position-based interpolation of deconvolved traces.',
            'speed_bin_edges': speed_edges.tolist(),
            'session_info': session_info,
        }
    }

    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    print(f'saved {args.outpicklefile}')
    print(f'n_sessions={len(neural)} n_subjects={len(subjects)}')


if __name__ == '__main__':
    main()
