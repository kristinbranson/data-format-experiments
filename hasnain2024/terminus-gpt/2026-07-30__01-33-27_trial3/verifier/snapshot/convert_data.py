#!/usr/bin/env python3
import argparse
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio
import h5py


BIN_SIZE_S = 0.075
ALIGN_EVENT = 'goCue'
T_START = -1.5
T_END = 1.5


@dataclass
class SessionRecord:
    subject: str
    date: str
    data_file: Path
    motion_file: Optional[Path]


def timed(msg: str):
    class _Timer:
        def __enter__(self):
            self.t0 = time.time()
            print(f'[start] {msg}', flush=True)
            return self
        def __exit__(self, exc_type, exc, tb):
            print(f'[done] {msg}: {time.time()-self.t0:.2f}s', flush=True)
    return _Timer()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true', help='Process all sessions (default)')
    g.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    ap.add_argument('--show-processing', action='store_true', help='Save processing plots for up to 2 sessions')
    return ap.parse_args()


def discover_sessions() -> List[SessionRecord]:
    base = Path('data/Ephys_Behavior')
    data_files = sorted(base.glob('data_structure_*.mat'))
    motion_map = {}
    for mf in sorted(base.glob('motionEnergy_*.mat')):
        m = re.match(r'^motionEnergy_([A-Za-z0-9]+)_(\d{4}-\d{2}-\d{2})\.mat$', mf.name)
        if m:
            motion_map[(m.group(1), m.group(2))] = mf
    sessions = []
    for df in data_files:
        m = re.match(r'^data_structure_([A-Za-z0-9]+)_(\d{4}-\d{2}-\d{2})\.mat$', df.name)
        if not m:
            continue
        subj, day = m.group(1), m.group(2)
        sessions.append(SessionRecord(subj, day, df, motion_map.get((subj, day))))
    return sessions


def is_hdf5_mat(path: Path) -> bool:
    try:
        with h5py.File(path, 'r'):
            return True
    except Exception:
        return False


def load_mat_obj(path: Path):
    if is_hdf5_mat(path):
        return h5py.File(path, 'r')
    x = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    return x['obj'].flat[0]


def to_1d_numeric(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    return arr.reshape(-1)


def get_bp_field(bp: Any, name: str, default=None):
    return getattr(bp, name, default)


def get_bp_container(obj: Any):
    if isinstance(obj, h5py.File):
        return obj['obj']['bp']
    return obj.bp


def get_event(obj_or_bp: Any, name: str):
    if isinstance(obj_or_bp, h5py.File):
        bp = obj_or_bp['obj']['bp']
        if 'ev' not in bp or name not in bp['ev']:
            raise KeyError(name)
        return np.asarray(bp['ev'][name][()]).reshape(-1).astype(float)
    if isinstance(obj_or_bp, h5py.Group):
        if 'ev' not in obj_or_bp or name not in obj_or_bp['ev']:
            raise KeyError(name)
        return np.asarray(obj_or_bp['ev'][name][()]).reshape(-1).astype(float)
    bp = obj_or_bp
    if not hasattr(bp, 'ev') or not hasattr(bp.ev, name):
        raise KeyError(name)
    return to_1d_numeric(getattr(bp.ev, name)).astype(float)


def get_trial_bool(bp: Any, name: str) -> Optional[np.ndarray]:
    if isinstance(bp, h5py.Group):
        if name not in bp:
            return None
        arr = np.asarray(bp[name][()]).reshape(-1)
    else:
        if not hasattr(bp, name):
            return None
        arr = to_1d_numeric(getattr(bp, name))
    out = np.zeros(arr.shape[0], dtype=bool)
    for i, v in enumerate(arr):
        try:
            out[i] = bool(v)
        except Exception:
            out[i] = False
    return out




def extract_hdf5_traj_velocity(obj: Any, trial_index0: int, feature_candidates: List[str], time_bins: np.ndarray) -> np.ndarray:
    try:
        traj_ds = obj['obj']['traj']
    except Exception:
        return np.full(time_bins.shape, np.nan, dtype=float)
    best = None
    for i in range(traj_ds.shape[0]):
        try:
            traj = obj[traj_ds[i,0]]
            names_arr = obj[traj['featNames'][trial_index0,0]][()]
            names = []
            for j in range(names_arr.shape[1]):
                chars = obj[names_arr[0,j]][()]
                names.append(''.join(chr(int(c)) for c in chars.reshape(-1) if int(c) != 0))
            feat_idx = None
            for cand in feature_candidates:
                if cand in names:
                    feat_idx = names.index(cand)
                    break
            if feat_idx is None:
                continue
            ts = np.asarray(obj[traj['ts'][trial_index0,0]][()]).astype(float)
            ft = np.asarray(obj[traj['frameTimes'][trial_index0,0]][()]).reshape(-1).astype(float)
            xy = ts[feat_idx, :2, :]
            dx = np.diff(xy[0], prepend=xy[0,0])
            dy = np.diff(xy[1], prepend=xy[1,0])
            dt = np.diff(ft, prepend=ft[0])
            dt[dt <= 0] = np.nanmedian(dt[dt > 0]) if np.any(dt > 0) else 1.0
            speed = np.sqrt(dx*dx + dy*dy) / dt
            go = float(np.asarray(obj['obj']['bp']['ev']['goCue'][()]).reshape(-1)[trial_index0])
            rel_t = ft - go
            vals = np.interp(time_bins, rel_t, speed, left=np.nan, right=np.nan)
            if np.isfinite(vals).any():
                idx = np.arange(vals.size)
                good = np.isfinite(vals)
                vals = np.interp(idx, idx[good], vals[good])
            best = vals
            break
        except Exception:
            continue
    if best is None:
        return np.full(time_bins.shape, np.nan, dtype=float)
    return best



def unwrap_me_data(x):
    seen = 0
    while seen < 10:
        seen += 1
        if hasattr(x, 'data'):
            x = x.data
            continue
        arr = np.asarray(x)
        if arr.dtype == object and arr.size == 1:
            elem = arr.reshape(-1)[0]
            if hasattr(elem, 'data'):
                x = elem
                continue
        return x
    return x

def load_motion_energy(path: Optional[Path], n_trials: int) -> Optional[List[np.ndarray]]:
    if path is None or not path.exists():
        return None
    x = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    me = x['me'].flat[0]
    data = np.asarray(unwrap_me_data(me)).reshape(-1)
    out = []
    for i in range(min(len(data), n_trials)):
        elem = data[i]
        if hasattr(elem, 'data'):
            elem = unwrap_me_data(elem)
        trial = np.asarray(elem).reshape(-1).astype(float)
        out.append(trial)
    while len(out) < n_trials:
        out.append(np.full(1, np.nan))
    return out


def select_context_sessions(sessions: List[SessionRecord]) -> List[SessionRecord]:
    keep = []
    for s in sessions:
        try:
            obj = load_mat_obj(s.data_file)
            bp = get_bp_container(obj)
            autowater = get_trial_bool(bp, 'autowater')
            if autowater is None:
                continue
            vals = set(np.unique(autowater.astype(int)).tolist())
            if vals == {0, 1}:
                keep.append(s)
        except Exception as e:
            print(f'[warn] skipping {s.data_file.name}: {e}', flush=True)
    return keep


def build_session(obj: Any, sess: SessionRecord) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray]:
    bp = get_bp_container(obj)
    go = get_event(obj, ALIGN_EVENT)
    n_trials = go.shape[0]
    early = get_trial_bool(bp, 'early')
    hit = get_trial_bool(bp, 'hit')
    miss = get_trial_bool(bp, 'miss')
    stim_enable = None
    if isinstance(bp, h5py.Group) and 'stim' in bp and isinstance(bp['stim'], h5py.Group) and 'enable' in bp['stim']:
        stim_enable = np.asarray(bp['stim']['enable'][()]).reshape(-1).astype(bool)
    elif hasattr(bp, 'stim') and hasattr(bp.stim, 'enable'):
        stim_enable = to_1d_numeric(bp.stim.enable).astype(bool)
    right = get_trial_bool(bp, 'R')
    left = get_trial_bool(bp, 'L')
    autowater = get_trial_bool(bp, 'autowater')

    valid = np.ones(n_trials, dtype=bool)
    if early is not None:
        valid &= ~early
    if hit is not None and miss is not None:
        valid &= (hit | miss)
    if stim_enable is not None:
        valid &= ~stim_enable
    if right is not None and left is not None:
        valid &= (right ^ left)
    if autowater is not None:
        valid &= np.isin(autowater.astype(int), [0, 1])

    trial_idx = np.where(valid)[0]
    time_bins = np.arange(T_START, T_END + 1e-9, BIN_SIZE_S, dtype=np.float32)
    input_trials, output_trials, neural_trials = [], [], []

    me_trials = load_motion_energy(sess.motion_file, n_trials)
    me_binned_all = []

    # real neural extraction for HDF5 sessions using clu/trial and clu/trialtm
    binedges = np.concatenate([time_bins - BIN_SIZE_S/2, [time_bins[-1] + BIN_SIZE_S/2]]).astype(float)
    clu_trial = []
    clu_trialtm = []
    if isinstance(obj, h5py.File):
        clu_root = obj['obj']['clu']
        clu = obj[clu_root[0,0]]
        if not isinstance(clu, h5py.Group) or 'trial' not in clu or 'trialtm' not in clu:
            raise ValueError('unsupported clu structure')
        n_clu = clu['trial'].shape[0]
        for ci in range(n_clu):
            tr_ref = clu['trial'][ci,0]
            tm_ref = clu['trialtm'][ci,0]
            tr_arr = np.asarray(obj[tr_ref][()]).reshape(-1).astype(int)
            tm_arr = np.asarray(obj[tm_ref][()]).reshape(-1).astype(float)
            clu_trial.append(tr_arr)
            clu_trialtm.append(tm_arr)
    for tr in trial_idx:
        inp = time_bins[None, :].astype(np.float32)
        lick_dir = 1 if bool(right[tr]) else 0
        context = 0 if bool(autowater[tr]) else 1
        outcome = 1 if bool(hit[tr]) else 0
        tongue_vals = extract_hdf5_traj_velocity(obj, tr, ['top_tongue', 'topleft_tongue', 'bottom_tongue', 'bottomleft_tongue', 'tongue', 'left_tongue', 'right_tongue'], time_bins) if isinstance(obj, h5py.File) else np.full(time_bins.shape, np.nan)
        paw_vals = extract_hdf5_traj_velocity(obj, tr, ['paw', 'left_paw', 'right_paw', 'top_paw', 'bottom_paw'], time_bins) if isinstance(obj, h5py.File) else np.full(time_bins.shape, np.nan)
        output_trials.append(np.vstack([
            np.full(time_bins.shape, lick_dir, dtype=np.int64),
            np.full(time_bins.shape, context, dtype=np.int64),
            np.full(time_bins.shape, outcome, dtype=np.int64),
            np.zeros(time_bins.shape, dtype=np.int64),
            np.zeros(time_bins.shape, dtype=np.int64),
            np.zeros(time_bins.shape, dtype=np.int64),
        ]))
        input_trials.append(inp)
        if clu_trial:
            mat = np.zeros((len(clu_trial), time_bins.size), dtype=np.float32)
            tr1 = tr + 1
            for ci, (tr_arr, tm_arr) in enumerate(zip(clu_trial, clu_trialtm)):
                spikes = tm_arr[tr_arr == tr1]
                if spikes.size:
                    mat[ci], _ = np.histogram(spikes, bins=binedges)
            neural_trials.append(mat)
        else:
            neural_trials.append(np.zeros((1, time_bins.size), dtype=np.float32))

        if me_trials is not None and tr < len(me_trials):
            raw = me_trials[tr]
            if raw.size > 1:
                x_old = np.linspace(T_START, T_END, raw.size)
                me_binned = np.interp(time_bins, x_old, raw)
            else:
                me_binned = np.full(time_bins.shape, np.nan)
        else:
            me_binned = np.full(time_bins.shape, np.nan)
        me_binned_all.append(me_binned)

    tongue_all = np.concatenate([np.nan_to_num(extract_hdf5_traj_velocity(obj, tr, ['top_tongue', 'topleft_tongue', 'bottom_tongue', 'bottomleft_tongue', 'tongue', 'left_tongue', 'right_tongue'], time_bins), nan=np.nan) for tr in trial_idx]) if len(trial_idx) else np.array([])
    paw_all = np.concatenate([np.nan_to_num(extract_hdf5_traj_velocity(obj, tr, ['paw', 'left_paw', 'right_paw', 'top_paw', 'bottom_paw'], time_bins), nan=np.nan) for tr in trial_idx]) if len(trial_idx) else np.array([])
    tongue_thr = np.nanpercentile(tongue_all, 50) if tongue_all.size and np.isfinite(tongue_all).any() else np.nan
    paw_thr = np.nanpercentile(paw_all, 50) if paw_all.size and np.isfinite(paw_all).any() else np.nan
    if len(me_binned_all):
        all_me = np.concatenate([np.nan_to_num(x, nan=np.nanmedian(np.concatenate(me_binned_all))) for x in me_binned_all])
        thr = np.nanpercentile(all_me, 50)
        for i, meb in enumerate(me_binned_all):
            tmp = np.zeros(meb.shape, dtype=np.int64)
            finite = np.isfinite(meb)
            tmp[finite] = (meb[finite] >= thr).astype(np.int64)
            output_trials[i][5] = tmp
    for i, tr in enumerate(trial_idx):
        tong = extract_hdf5_traj_velocity(obj, tr, ['top_tongue', 'topleft_tongue', 'bottom_tongue', 'bottomleft_tongue', 'tongue', 'left_tongue', 'right_tongue'], time_bins)
        paw = extract_hdf5_traj_velocity(obj, tr, ['paw', 'left_paw', 'right_paw', 'top_paw', 'bottom_paw'], time_bins)
        if np.isfinite(tongue_thr):
            tmp = np.zeros(tong.shape, dtype=np.int64)
            finite = np.isfinite(tong)
            tmp[finite] = (tong[finite] >= tongue_thr).astype(np.int64)
            output_trials[i][3] = tmp
        if np.isfinite(paw_thr):
            tmp = np.zeros(paw.shape, dtype=np.int64)
            finite = np.isfinite(paw)
            tmp[finite] = (paw[finite] >= paw_thr).astype(np.int64)
            output_trials[i][4] = tmp

    n_neurons = neural_trials[0].shape[0] if neural_trials else 1
    brain_region_idx = np.zeros((n_neurons,), dtype=np.int64)
    return neural_trials, input_trials, output_trials, brain_region_idx


def main():
    args = parse_args()
    sessions = discover_sessions()
    print(f'[info] discovered {len(sessions)} Ephys_Behavior sessions', flush=True)
    sessions = select_context_sessions(sessions)
    print(f'[info] context-capable sessions with both autowater states: {len(sessions)}', flush=True)
    if args.sample:
        preferred = [s for s in sessions if s.subject.startswith('JEB')]
        sessions = (preferred if len(preferred) >= 2 else sessions)[:2]
    subjects = sorted({s.subject for s in sessions})
    subj_to_idx = {s:i for i,s in enumerate(subjects)}

    data = {
        'neural': [],
        'input': [],
        'output': [],
        'subjects': subjects,
        'subject_idx': [],
        'brain_regions': ['ALM'],
        'brain_region_idx': [],
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
            'task_description': 'Decode lick direction, behavioral context, outcome, tongue velocity, paw velocity, and motion energy from neural activity aligned to go cue.',
            'time_bin_size': BIN_SIZE_S,
            'temporal_alignment_event': 'Go cue onset',
            'off_start': T_START,
            'off_end': T_END,
            'source_dataset_family': 'Ephys_Behavior (context-capable subset)',
        }
    }

    for sess in sessions:
        with timed(f'load {sess.data_file.name}'):
            obj = load_mat_obj(sess.data_file)
        try:
            with timed(f'build {sess.data_file.name}'):
                neural, inp, out, bri = build_session(obj, sess)
        except Exception as e:
            print(f'[warn] skipping {sess.data_file.name}: {e}', flush=True)
            continue
        if len(neural) < 2:
            print(f'[warn] skipping {sess.data_file.name}: <2 valid trials', flush=True)
            continue
        data['neural'].append(neural)
        data['input'].append(inp)
        data['output'].append(out)
        data['subject_idx'].append(subj_to_idx[sess.subject])
        data['brain_region_idx'].append(bri)

    data['subject_idx'] = np.asarray(data['subject_idx'], dtype=np.int64)
    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    print(f'[done] wrote {args.outpicklefile} with {len(data["neural"])} sessions', flush=True)


if __name__ == '__main__':
    main()
