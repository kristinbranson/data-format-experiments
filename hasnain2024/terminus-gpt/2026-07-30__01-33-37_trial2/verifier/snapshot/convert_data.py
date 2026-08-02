#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true', help='Process all sessions')
    g.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    ap.add_argument('--show-processing', action='store_true', help='Save processing plots for up to 2 sessions')
    return ap.parse_args()


def decode_char_dataset(ds):
    arr = np.array(ds)
    chars = []
    for x in arr.reshape(-1):
        try:
            ix = int(x)
        except Exception:
            continue
        if ix != 0:
            chars.append(chr(ix))
    return ''.join(chars)


def safe_decode_ref_string(h, ref):
    try:
        obj = deref(h, ref)
        if isinstance(obj, h5py.Dataset):
            arr = obj[()]
            if arr.dtype == h5py.ref_dtype or str(arr.dtype) == 'object':
                flat = arr.reshape(-1)
                if len(flat):
                    return safe_decode_ref_string(h, flat[0])
            return decode_char_dataset(obj)
    except Exception:
        pass
    return ''


def deref(h, ref):
    return h[ref]


def read_dataset_maybe_refs(h, ds):
    arr = ds[()]
    if ds.attrs.get('MATLAB_class', b'') == b'char':
        return decode_char_dataset(ds)
    return arr


def read_char_ref_array(h, ds):
    arr = ds[()]
    out = []
    for r in arr.reshape(-1):
        out.append(decode_char_dataset(deref(h, r)))
    return np.array(out, dtype=object).reshape(arr.shape)


def read_numeric_ref_array(h, ds):
    arr = ds[()]
    out = []
    for r in arr.reshape(-1):
        out.append(np.array(deref(h, r)))
    return np.array(out, dtype=object).reshape(arr.shape)


def load_data_structure(path):
    out = {}
    try:
        hfile = h5py.File(path, 'r')
    except OSError:
        return None
    with hfile as h:
        obj = h['obj']

        bp = obj['bp']
        bp_out = {}
        for name in ['L', 'R', 'Ntrials', 'autowater', 'bitRand', 'early', 'hit', 'miss', 'no']:
            if name in bp:
                bp_out[name] = np.array(bp[name])
        if 'ev' in bp:
            ev = bp['ev']
            ev_out = {}
            for name in ['bitStart', 'delay', 'goCue', 'lickL', 'lickR', 'reward', 'sample']:
                if name in ev:
                    ev_out[name] = np.array(ev[name])
            bp_out['ev'] = ev_out
        out['bp'] = bp_out

        clu_ref = obj['clu'][()].reshape(-1)[0]
        clu = deref(h, clu_ref)
        clu_out = {}
        for name in ['tm', 'site', 'quality', 'trialtm', 'trial']:
            if name in clu:
                ds = clu[name]
                if ds.dtype == h5py.ref_dtype or str(ds.dtype) == 'object':
                    clu_out[name] = [np.array(deref(h, r)).reshape(-1) for r in ds[()].reshape(-1)]
                else:
                    clu_out[name] = np.array(ds)
        out['clu'] = clu_out

        if 'traj' in obj:
            traj_views = []
            traj_arr = obj['traj'][()]
            for view_ref in traj_arr.reshape(-1):
                traj = deref(h, view_ref)
                traj_out = {}
                if 'featNames' in traj:
                    feat_arr = traj['featNames'][()]
                    if feat_arr.size:
                        feat_cell = np.array(deref(h, feat_arr.reshape(-1)[0]))
                        traj_out['featNames'] = [safe_decode_ref_string(h, r) for r in feat_cell.reshape(-1)]
                    else:
                        traj_out['featNames'] = []
                for name in ['ts', 'frameTimes', 'fn', 'NdroppedFrames']:
                    if name in traj:
                        ds = traj[name]
                        if ds.dtype == h5py.ref_dtype or str(ds.dtype) == 'object':
                            traj_out[name] = [np.array(deref(h, r)) for r in ds[()].reshape(-1)]
                        else:
                            traj_out[name] = np.array(ds)
                traj_views.append(traj_out)
            out['traj'] = traj_views

        if 'meta' in obj:
            meta = obj['meta']
            meta_out = {}
            for name in meta.keys():
                child = meta[name]
                if isinstance(child, h5py.Dataset) and child.attrs.get('MATLAB_class', b'') == b'char':
                    meta_out[name] = decode_char_dataset(child)
            out['meta'] = meta_out
    return out


def load_motion_energy(path):
    return sio.loadmat(path, simplify_cells=True)['me']


def discover_sessions(data_root):
    sessions = {}
    for subdir in Path(data_root).iterdir():
        if not subdir.is_dir():
            continue
        for f in subdir.glob('data_structure_*.mat'):
            stem = f.stem.replace('data_structure_', '')
            sessions.setdefault(stem, {})['data_structure'] = f
        for f in subdir.glob('motionEnergy_*.mat'):
            stem = f.stem.replace('motionEnergy_', '')
            sessions.setdefault(stem, {})['motion_energy'] = f
    return sessions


def valid_trials(bp):
    n = int(np.array(bp['Ntrials']).reshape(-1)[0])
    mask = np.ones(n, dtype=bool)
    for name in ['early', 'no']:
        if name in bp:
            arr = np.array(bp[name], dtype=float).reshape(-1)
            if arr.size == n:
                mask &= (arr == 0)
    return mask


def infer_subject_session(stem):
    parts = stem.split('_')
    return parts[0], '_'.join(parts[1:])


def build_time_grid(bin_size_s=0.075, t_start=-2.5, t_end=2.5):
    edges = np.arange(t_start, t_end + 1e-9, bin_size_s)
    centers = edges[:-1] + bin_size_s / 2
    return edges, centers


def bin_spikes_for_session(obj, trial_mask, edges):
    clu = obj['clu']
    if 'trialtm' not in clu or 'trial' not in clu:
        return None, None
    trialtm = clu['trialtm']
    trialid = clu['trial']
    neural_trials = None
    kept_units = []
    for ui, (st, tr) in enumerate(zip(trialtm, trialid)):
        st = np.array(st, dtype=float).reshape(-1)
        tr = np.array(tr, dtype=float).reshape(-1).astype(int) - 1
        if st.size == 0:
            continue
        approx_duration = (edges[-1] - edges[0]) * max(trial_mask.sum(), 1)
        fr = st.size / max(approx_duration, 1e-9)
        if fr <= 1.0:
            continue
        kept_units.append(ui)
        unit_trials = []
        for raw_t in np.where(trial_mask)[0]:
            counts, _ = np.histogram(st[tr == raw_t], bins=edges)
            unit_trials.append(counts.astype(np.float32))
        if neural_trials is None:
            neural_trials = [[u] for u in unit_trials]
        else:
            for i, u in enumerate(unit_trials):
                neural_trials[i].append(u)
    if neural_trials is None:
        return [], np.array([], dtype=int)
    return [np.stack(x, axis=0) for x in neural_trials], np.array(kept_units, dtype=int)


def build_trial_labels(obj, trial_mask):
    bp = obj['bp']
    idx = np.where(trial_mask)[0]
    L = np.array(bp['L'], dtype=float).reshape(-1)
    R = np.array(bp['R'], dtype=float).reshape(-1)
    autowater = np.array(bp['autowater'], dtype=float).reshape(-1)
    hit = np.array(bp['hit'], dtype=float).reshape(-1)
    miss = np.array(bp['miss'], dtype=float).reshape(-1)
    lick_dir = np.where(R[idx] > L[idx], 1, 0).astype(np.int64)
    context = np.where(autowater[idx] > 0, 0, 1).astype(np.int64)
    outcome = np.where(hit[idx] > miss[idx], 1, 0).astype(np.int64)
    return idx, lick_dir, context, outcome


def rebin_variable_trace(values, n_bins):
    arr = np.array(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.zeros(n_bins, dtype=np.float32)
    xp = np.linspace(0, 1, arr.size)
    xnew = np.linspace(0, 1, n_bins)
    return np.interp(xnew, xp, arr).astype(np.float32)


def interp_feature_velocity(ts, frame_times, align_time, centers, feat_idx, tongue=False):
    arr = np.array(ts, dtype=float)
    if arr.ndim != 3 or feat_idx >= arr.shape[2]:
        return np.zeros(len(centers), dtype=np.float32)
    xy = arr[:, :2, feat_idx]
    ft = np.array(frame_times, dtype=float).reshape(-1)
    if ft.size != xy.shape[0]:
        if ft.size == 0:
            ft = np.arange(1, xy.shape[0] + 1, dtype=float) / 400.0
        else:
            ft = np.linspace(ft.min(), ft.max(), xy.shape[0])
    rel_t = (ft - 0.5) - float(align_time)
    x = np.interp(centers, rel_t, xy[:, 0], left=np.nan, right=np.nan)
    y = np.interp(centers, rel_t, xy[:, 1], left=np.nan, right=np.nan)
    xv = np.gradient(x)
    yv = np.gradient(y)
    if tongue:
        xv[np.isnan(xv)] = 0
        yv[np.isnan(yv)] = 0
    else:
        if np.all(np.isnan(xv)) or np.all(np.isnan(yv)):
            return np.zeros(len(centers), dtype=np.float32)
        xv = np.where(np.isnan(xv), np.nanmedian(xv), xv)
        yv = np.where(np.isnan(yv), np.nanmedian(yv), yv)
        xv = xv - np.nanmedian(xv)
        yv = yv - np.nanmedian(yv)
    return np.sqrt(xv**2 + yv**2).astype(np.float32)


def build_traj_outputs(obj, trial_idx, centers):
    traj_views = obj.get('traj', [])
    if len(traj_views) < 2:
        return None, None
    side = traj_views[0]
    bottom = traj_views[1]
    side_names = [str(x) for x in side.get('featNames', [])]
    bottom_names = [str(x) for x in bottom.get('featNames', [])]
    go = np.array(obj['bp']['ev']['goCue'], dtype=float).reshape(-1)

    def pick_feature(feat_names, candidates):
        for cand in candidates:
            for i, name in enumerate(feat_names):
                if cand == name:
                    return i
        for cand in candidates:
            for i, name in enumerate(feat_names):
                if cand in name:
                    return i
        return None

    tongue_idx = pick_feature(side_names, ['tongue', 'left_tongue', 'right_tongue'])
    paw_idx = pick_feature(bottom_names, ['top_paw', 'bottom_paw', 'paw'])
    if paw_idx is None:
        return None, None
    tongue = []
    paw = []
    for tr in trial_idx:
        ts_side = side['ts'][tr] if tr < len(side.get('ts', [])) else np.array([])
        ft_side = side['frameTimes'][tr] if tr < len(side.get('frameTimes', [])) else np.array([])
        ts_bot = bottom['ts'][tr] if tr < len(bottom.get('ts', [])) else np.array([])
        ft_bot = bottom['frameTimes'][tr] if tr < len(bottom.get('frameTimes', [])) else np.array([])
        if tongue_idx is None:
            tongue.append(np.zeros(len(centers), dtype=np.float32))
        else:
            tongue.append(interp_feature_velocity(ts_side, ft_side, go[tr], centers, tongue_idx, tongue=True))
        paw.append(interp_feature_velocity(ts_bot, ft_bot, go[tr], centers, paw_idx, tongue=False))
    tongue = np.stack(tongue, axis=0)
    paw = np.stack(paw, axis=0)
    tongue_disc = (tongue > np.nanmedian(tongue)).astype(np.int64)
    paw_disc = (paw > np.nanmedian(paw)).astype(np.int64)
    return tongue_disc, paw_disc


def normalize_motion_energy_data(me):
    data = me
    if isinstance(data, dict):
        data = data.get('data', data)
    elif hasattr(data, 'dtype') and getattr(data.dtype, 'names', None):
        names = list(data.dtype.names)
        if 'data' in names:
            data = data['data']
    if isinstance(data, dict):
        for key in ['data', 'trials', 'values']:
            if key in data:
                data = data[key]
                break
    if isinstance(data, np.ndarray):
        if data.dtype == object:
            return list(data.reshape(-1))
        if data.ndim == 0:
            try:
                item = data.item()
                if isinstance(item, dict):
                    return normalize_motion_energy_data(item)
            except Exception:
                pass
        if data.ndim == 1:
            return list(data)
        return [row for row in data]
    if isinstance(data, (list, tuple)):
        return list(data)
    try:
        return list(data)
    except Exception:
        return []


def process_session(stem, files, edges, centers):
    obj = load_data_structure(files['data_structure'])
    if obj is None:
        print('  skip_reason: unreadable_data_structure', flush=True)
        return None
    if 'motion_energy' not in files:
        print('  skip_reason: no_motion_energy', flush=True)
        return None
    bp = obj['bp']
    trial_mask = valid_trials(bp)
    neural, kept_units = bin_spikes_for_session(obj, trial_mask, edges)
    if neural is None or kept_units is None:
        print('  skip_reason: missing_trial_aligned_spikes', flush=True)
        return None
    if len(neural) < 2 or len(kept_units) < 10:
        print(f'  skip_reason: insufficient_neural trials={len(neural)} units={len(kept_units)}', flush=True)
        return None
    idx, lick_dir, context, outcome = build_trial_labels(obj, trial_mask)
    me = load_motion_energy(files['motion_energy'])
    me_data = normalize_motion_energy_data(me)
    n_bins = len(centers)
    if len(me_data) == 0 or max(idx) >= len(me_data):
        print(f'  skip_reason: bad_motion_energy len_me={len(me_data)} max_trial={max(idx) if len(idx) else None}', flush=True)
        return None
    me_trials = [rebin_variable_trace(me_data[int(i)], n_bins) for i in idx]
    me_stack = np.stack(me_trials, axis=0)
    me_disc = (me_stack > np.nanmedian(me_stack)).astype(np.int64)
    tongue_disc, paw_disc = build_traj_outputs(obj, idx, centers)
    if tongue_disc is None or paw_disc is None:
        print('  skip_reason: no_paw_feature', flush=True)
        return None
    inputs = [centers[None, :].astype(np.float32) for _ in idx]
    outputs = []
    for i in range(len(idx)):
        outputs.append(np.vstack([
            np.full(n_bins, lick_dir[i], dtype=np.int64),
            np.full(n_bins, context[i], dtype=np.int64),
            np.full(n_bins, outcome[i], dtype=np.int64),
            tongue_disc[i],
            paw_disc[i],
            me_disc[i],
        ]))
    subject, _ = infer_subject_session(stem)
    return {
        'subject': subject,
        'neural': neural,
        'input': inputs,
        'output': outputs,
        'brain_region_idx': np.zeros(len(kept_units), dtype=np.int64),
    }


def main():
    args = parse_args()
    sessions = discover_sessions('data')
    keys = sorted(k for k,v in sessions.items() if 'data_structure' in v and 'motion_energy' in v)
    edges, centers = build_time_grid()
    out_sessions = []
    subjects = []
    subject_to_idx = {}
    for stem in keys:
        print(f'processing {stem}', flush=True)
        sess = process_session(stem, sessions[stem], edges, centers)
        if sess is None:
            print(f'skipped {stem}', flush=True)
            continue
        print(f'kept {stem} with {len(sess["neural"])} trials', flush=True)
        out_sessions.append(sess)
        if sess['subject'] not in subject_to_idx:
            subject_to_idx[sess['subject']] = len(subjects)
            subjects.append(sess['subject'])
        if args.sample and len(out_sessions) >= 2:
            break
    data = {
        'neural': [s['neural'] for s in out_sessions],
        'input': [s['input'] for s in out_sessions],
        'output': [s['output'] for s in out_sessions],
        'subjects': subjects,
        'subject_idx': np.array([subject_to_idx[s['subject']] for s in out_sessions], dtype=np.int64),
        'brain_regions': ['ALM'],
        'brain_region_idx': [s['brain_region_idx'] for s in out_sessions],
        'input_names': ['time_from_go_cue'],
        'output_names': ['lick_direction', 'behavioral_context', 'outcome', 'tongue_velocity', 'paw_velocity', 'motion_energy'],
        'output_values': [
            ['left', 'right'],
            ['WC', 'DR'],
            ['incorrect', 'correct'],
            ['low', 'high'],
            ['low', 'high'],
            ['low', 'high'],
        ],
        'metadata': {
            'task_description': 'Go-cue aligned neural decoding of lick direction, context, outcome, tongue velocity, paw velocity, and motion energy.',
            'time_bin_size': 75.0,
            'temporal_alignment_event': 'Go cue onset',
            'off_start': float(edges[0]),
            'off_end': float(edges[-1]),
        }
    }
    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    print(f'saved {args.outpicklefile} with {len(out_sessions)} sessions')


if __name__ == '__main__':
    main()
