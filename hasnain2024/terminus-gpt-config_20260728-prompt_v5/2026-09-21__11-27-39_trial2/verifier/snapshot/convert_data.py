#!/usr/bin/env python3
import argparse
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import scipy.io as sio

DATA_ROOT = Path('/app/data')
EPHYS_DIRS = ['Ephys_Behavior', 'RandomizedDelay_Ephys_Behavior']

# Initial implementation uses a conservative common window/binning; if reference
# params indicate otherwise, these will be updated after inspection/validation.
BIN_SIZE_S = 0.025
T_START = -2.5
T_END = 2.5


@dataclass
class SessionInfo:
    animal: str
    date: str
    task_dir: str
    data_file: Path
    motion_file: Optional[Path]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('outpicklefile')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true', help='Process all sessions')
    g.add_argument('--sample', action='store_true', help='Process only 2 sessions')
    p.add_argument('--show-processing', action='store_true')
    return p.parse_args()


def discover_sessions() -> List[SessionInfo]:
    sessions = []
    for task_dir in EPHYS_DIRS:
        d = DATA_ROOT / task_dir
        for data_file in sorted(d.glob('data_structure_*.mat')):
            m = re.match(r'data_structure_([^_]+)_(.+)\.mat', data_file.name)
            if not m:
                continue
            animal, date = m.groups()
            motion_file = d / f'motionEnergy_{animal}_{date}.mat'
            if not motion_file.exists():
                motion_file = None
            sessions.append(SessionInfo(animal, date, task_dir, data_file, motion_file))
    return sessions


def mat_fields(attrs) -> List[str]:
    mf = attrs.get('MATLAB_fields', None)
    out = []
    if mf is None:
        return out
    for x in mf:
        try:
            out.append(''.join(ch.decode() if isinstance(ch, (bytes, np.bytes_)) else str(ch) for ch in x))
        except Exception:
            out.append(str(x))
    return out


def decode_char(arr) -> str:
    arr = np.array(arr)
    if arr.dtype.kind not in 'ui':
        return ''
    return ''.join(chr(int(x)) for x in arr.flatten() if int(x) != 0)


def read_dataset(h: h5py.File, obj) -> Any:
    arr = obj[()]
    arr = np.array(arr)
    if obj.attrs.get('MATLAB_class', b'') == b'char':
        return decode_char(arr)
    return arr


def read_any(h: h5py.File, obj) -> Any:
    if isinstance(obj, h5py.Group):
        return {k: read_any(h, obj[k]) for k in obj.keys()}
    arr = np.array(obj[()])
    if obj.attrs.get('MATLAB_class', b'') == b'char':
        return decode_char(arr)
    if arr.dtype == object:
        vals = [read_any(h, h[r]) for r in arr.flatten(order='F')]
        try:
            return np.array(vals, dtype=object).reshape(arr.shape, order='F')
        except Exception:
            return vals
    return arr


def deref_cell(h: h5py.File, ds) -> List[Any]:
    arr = np.array(ds[()])
    if arr.dtype != object:
        return [arr]
    out = []
    for ref in arr.flatten(order='F'):
        out.append(h[ref])
    return out


def read_bp(h: h5py.File) -> Dict[str, Any]:
    return read_any(h, h['/obj/bp'])


def read_probe_locations(h: h5py.File) -> List[str]:
    try:
        meta = h['/obj/meta']
    except Exception:
        return []
    if not isinstance(meta, h5py.Group) or 'probe' not in meta:
        return []
    probe = meta['probe']
    if not isinstance(probe, h5py.Group) or 'loc' not in probe:
        return []
    loc = probe['loc']
    arr = np.array(loc[()])
    if arr.dtype != object:
        s = decode_char(arr)
        return [s] if s else []
    vals = []
    for ref in arr.flatten(order='F'):
        try:
            s = decode_char(h[ref][()])
        except Exception:
            s = ''
        if s:
            vals.append(s)
    return vals


def read_clu_units(h: h5py.File) -> List[Dict[str, Any]]:
    clu = h['/obj/clu']
    probe_groups = deref_cell(h, clu)
    units = []
    for probe_idx, grp in enumerate(probe_groups):
        if not isinstance(grp, h5py.Group):
            continue
        fields = {k: grp[k] for k in grp.keys()}
        if 'trial' not in fields or 'trialtm' not in fields:
            continue
        n_units = fields['trial'].shape[0]
        for i in range(n_units):
            unit = {'probe_idx': probe_idx}
            for key in ['tm', 'trial', 'trialtm', 'quality', 'site']:
                if key not in fields:
                    continue
                ds = fields[key]
                ref = ds[i, 0]
                target = h[ref]
                unit[key] = np.array(target[()]).squeeze()
            units.append(unit)
    return units


def filter_units(units: List[Dict[str, Any]], n_trials: int) -> List[Dict[str, Any]]:
    kept = []
    session_dur = max(n_trials * 1.0, 1.0)
    for u in units:
        tm = np.array(u['tm']).astype(float).ravel()
        mean_fr = tm.size / session_dur
        if mean_fr > 1.0:
            kept.append(u)
    return kept


def common_time_axis() -> np.ndarray:
    edges = np.arange(T_START, T_END + BIN_SIZE_S, BIN_SIZE_S)
    centers = edges[:-1] + BIN_SIZE_S / 2
    return centers.astype(np.float32)


def build_neural_trials(units: List[Dict[str, Any]], n_trials: int) -> List[np.ndarray]:
    time = common_time_axis()
    edges = np.concatenate([[time[0] - BIN_SIZE_S / 2], time + BIN_SIZE_S / 2])
    n_time = time.size
    n_units = len(units)
    mats = [np.zeros((n_units, n_time), dtype=np.float32) for _ in range(n_trials)]
    for ui, u in enumerate(units):
        trials = np.array(u['trial']).astype(int).ravel()
        trialtm = np.array(u['trialtm']).astype(float).ravel()
        # assume MATLAB 1-based trial indices
        for tr in range(1, n_trials + 1):
            mask = trials == tr
            if np.any(mask):
                mats[tr - 1][ui] = np.histogram(trialtm[mask], bins=edges)[0].astype(np.float32)
    return mats


def infer_outcomes(bp: Dict[str, Any], n_trials: int) -> np.ndarray:
    hit = np.ravel(bp['hit'])[:n_trials] > 0
    miss = np.ravel(bp['miss'])[:n_trials] > 0
    no = np.ravel(bp['no'])[:n_trials] > 0
    out = np.full(n_trials, 2, dtype=np.int64)
    out[miss] = 0
    out[hit] = 1
    out[no] = 2
    return out


def infer_lick_direction(bp: Dict[str, Any], outcomes: np.ndarray, n_trials: int) -> np.ndarray:
    R = np.ravel(bp['R'])[:n_trials] > 0
    L = np.ravel(bp['L'])[:n_trials] > 0
    lick = np.full(n_trials, 2, dtype=np.int64)
    lick[L] = 0
    lick[R] = 1
    lick[outcomes == 2] = 2
    return lick


def infer_context(sess: SessionInfo, bp: Dict[str, Any], n_trials: int) -> np.ndarray:
    # Conservative initial mapping: randomized-delay sessions are DR; standard ephys
    # sessions default DR unless protocol nums show multiple contexts.
    ctx = np.zeros(n_trials, dtype=np.int64)
    protocol = bp.get('protocol', {})
    nums = protocol.get('nums', None) if isinstance(protocol, dict) else None
    if nums is not None:
        vals = np.ravel(nums)[:n_trials]
        uniq = [u for u in np.unique(vals) if np.isfinite(u)]
        if len(uniq) == 2:
            ctx = (vals == uniq.max()).astype(np.int64)
    return ctx




def read_traj_trial_refs(h: h5py.File):
    traj = h['/obj/traj']
    arr = np.array(traj[()])
    if arr.dtype != object or arr.size == 0:
        return None
    return h[arr.flatten(order='F')[0]]


def align_tongue_speed(h: h5py.File, bp: Dict[str, Any], n_trials: int, time: np.ndarray) -> Optional[np.ndarray]:
    tg = read_traj_trial_refs(h)
    if tg is None or 'ts' not in tg or 'frameTimes' not in tg:
        return None
    ts_refs = np.array(tg['ts'][()]).flatten(order='F')
    ft_refs = np.array(tg['frameTimes'][()]).flatten(order='F')
    align_times = np.ravel(bp['ev']['goCue']).astype(float)[:n_trials]
    out = np.full((n_trials, time.size), np.nan, dtype=np.float32)
    n = min(n_trials, len(ts_refs), len(ft_refs))
    for tr in range(n):
        ts = np.array(h[ts_refs[tr]][()])
        ft = np.array(h[ft_refs[tr]][()]).astype(float).ravel()
        if ts.ndim != 3 or ts.shape[0] < 1 or ts.shape[1] < 3:
            continue
        x = ts[0, 0, :].astype(float)
        y = ts[0, 1, :].astype(float)
        p = ts[0, 2, :].astype(float)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(ft) & (p > 0.5)
        if valid.sum() < 3:
            continue
        xv = x[valid]
        yv = y[valid]
        tv = ft[valid] - align_times[tr]
        dt = np.diff(tv)
        good = dt > 0
        if good.sum() < 2:
            continue
        speed = np.sqrt(np.diff(xv)**2 + np.diff(yv)**2) / dt
        tmid = (tv[:-1] + tv[1:]) / 2
        tmid = tmid[good]
        speed = speed[good]
        finite = np.isfinite(tmid) & np.isfinite(speed)
        if finite.sum() < 2:
            continue
        yint = np.interp(time, tmid[finite], speed[finite], left=np.nan, right=np.nan)
        idx = np.where(np.isfinite(yint))[0]
        if idx.size == 0:
            continue
        first, last = idx[0], idx[-1]
        yint[:first] = np.nan
        yint[last+1:] = np.nan
        out[tr] = yint.astype(np.float32)
    return out



def extract_numeric_motion_trial(x):
    if isinstance(x, np.ndarray):
        try:
            return np.asarray(x, dtype=float).ravel()
        except Exception:
            pass
        if x.dtype == object and x.size == 1:
            return extract_numeric_motion_trial(x.flat[0])
    if hasattr(x, '__dict__'):
        for name in ['data', 'me', 'motionEnergy', 'values', 'value']:
            if hasattr(x, name):
                out = extract_numeric_motion_trial(getattr(x, name))
                if out is not None:
                    return out
    try:
        return np.asarray(x, dtype=float).ravel()
    except Exception:
        return None

def load_motion_energy(path: Optional[Path]):
    if path is None:
        return None
    m = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    return m['me'].flat[0]


def align_motion_energy(me, bp: Dict[str, Any], n_trials: int, time: np.ndarray) -> Optional[np.ndarray]:
    if me is None:
        return None
    data = getattr(me, 'data', None)
    if data is None:
        return None
    try:
        data_arr = np.asarray(data, dtype=object)
    except Exception:
        try:
            data_arr = np.array(data, dtype=object)
        except Exception:
            data_arr = np.array([data], dtype=object)
    align_times = np.ravel(bp['ev']['goCue']).astype(float)[:n_trials]
    out = np.full((n_trials, time.size), np.nan, dtype=np.float32)
    for tr in range(min(n_trials, data_arr.size)):
        x = extract_numeric_motion_trial(data_arr.flat[tr])
        if x is None or x.size == 0:
            continue
        if x.size == 0:
            continue
        frame_times = np.arange(1, x.size + 1, dtype=float) / 400.0
        old_t = frame_times - 0.5 - align_times[tr]
        valid = np.isfinite(old_t) & np.isfinite(x)
        if valid.sum() < 2:
            continue
        y = np.interp(time, old_t[valid], x[valid], left=np.nan, right=np.nan)
        if np.all(np.isnan(y)):
            continue
        # nearest fill for edge NaNs to mimic reference helper behavior
        idx = np.where(np.isfinite(y))[0]
        first, last = idx[0], idx[-1]
        y[:first] = y[first]
        y[last+1:] = y[last]
        out[tr] = y.astype(np.float32)
    return out


def discretize_session_median(arr2d: Optional[np.ndarray], missing_code: int) -> np.ndarray:
    if arr2d is None:
        return np.full((0, 0), missing_code, dtype=np.int64)
    valid = np.isfinite(arr2d)
    if not np.any(valid):
        return np.full(arr2d.shape, missing_code, dtype=np.int64)
    thr = np.nanmedian(arr2d[valid])
    out = np.full(arr2d.shape, missing_code, dtype=np.int64)
    out[valid & (arr2d < thr)] = 0
    out[valid & (arr2d >= thr)] = 1
    return out


def build_session(sess: SessionInfo):
    with h5py.File(sess.data_file, 'r') as h:
        bp = read_bp(h)
        n_trials = int(np.array(bp['Ntrials']).squeeze())
        units = read_clu_units(h)
        units = filter_units(units, n_trials)
        neural_trials = build_neural_trials(units, n_trials)
        probe_locs = read_probe_locations(h)
        if not probe_locs:
            probe_locs = ['ALM']
        brain_region_idx = np.zeros(len(units), dtype=np.int64)
        time = common_time_axis()
        tongue_aligned = align_tongue_speed(h, bp, n_trials, time)

    outcomes = infer_outcomes(bp, n_trials)
    lick = infer_lick_direction(bp, outcomes, n_trials)
    context = infer_context(sess, bp, n_trials)
    me = load_motion_energy(sess.motion_file)
    motion_aligned = align_motion_energy(me, bp, n_trials, time)
    motion_disc = discretize_session_median(motion_aligned, missing_code=2) if motion_aligned is not None else None
    tongue_disc = discretize_session_median(tongue_aligned, missing_code=2) if tongue_aligned is not None else None

    input_trials = []
    output_trials = []
    for tr in range(n_trials):
        input_trials.append(time[None, :].astype(np.float32))
        tongue = tongue_disc[tr] if tongue_disc is not None else np.full(time.shape, 2, dtype=np.int64)
        paw = np.full(time.shape, 2, dtype=np.int64)
        motion = motion_disc[tr] if motion_disc is not None else np.full(time.shape, 2, dtype=np.int64)
        out = np.vstack([
            np.full((1, time.size), lick[tr], dtype=np.int64),
            np.full((1, time.size), context[tr], dtype=np.int64),
            np.full((1, time.size), outcomes[tr], dtype=np.int64),
            tongue[None, :],
            paw[None, :],
            motion[None, :],
        ])
        output_trials.append(out)

    return neural_trials, input_trials, output_trials, sess.animal, brain_region_idx


def main():
    args = parse_args()
    sessions = discover_sessions()
    if args.sample:
        sessions = sessions[:2]

    data = {
        'neural': [],
        'input': [],
        'output': [],
        'subjects': [],
        'subject_idx': [],
        'brain_regions': ['ALM'],
        'brain_region_idx': [],
        'input_names': ['time_from_go_cue_s'],
        'output_names': ['lick_direction', 'behavioral_context', 'outcome', 'tongue_velocity', 'paw_velocity', 'motion_energy'],
        'output_values': [
            ['left', 'right', 'none'],
            ['DR', 'WC'],
            ['incorrect', 'correct', 'ignore'],
            ['lt50', 'ge50', 'not_visible'],
            ['lt50', 'ge50', 'not_visible'],
            ['lt50', 'ge50', 'no_video'],
        ],
        'metadata': {
            'task_description': 'Mouse ALM delayed-response / water-cued licking task with go-cue aligned neural and behavioral outputs',
            'time_bin_size': BIN_SIZE_S,
            'temporal_alignment_event': 'go cue onset',
            'off_start': T_START,
            'off_end': T_END,
            'source_dirs': EPHYS_DIRS,
        }
    }

    subject_to_idx = {}
    skipped = []
    for sess in sessions:
        print(f'Processing {sess.animal} {sess.date} {sess.task_dir}')
        try:
            neural_trials, input_trials, output_trials, subject, bri = build_session(sess)
        except Exception as e:
            print(f'SKIP {sess.data_file.name}: {e!r}')
            skipped.append((sess.data_file.name, repr(e)))
            continue
        if subject not in subject_to_idx:
            subject_to_idx[subject] = len(data['subjects'])
            data['subjects'].append(subject)
        data['subject_idx'].append(subject_to_idx[subject])
        data['neural'].append(neural_trials)
        data['input'].append(input_trials)
        data['output'].append(output_trials)
        data['brain_region_idx'].append(bri)

    data['subject_idx'] = np.asarray(data['subject_idx'], dtype=np.int64)
    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved {args.outpicklefile}')
    if 'skipped' in locals() and skipped:
        print('Skipped sessions:')
        for name, err in skipped:
            print(name, err)


if __name__ == '__main__':
    main()


# --- scipy fallback for non-HDF5 data_structure files ---
def scipy_obj_to_bp(obj_bp):
    bp = obj_bp.flat[0] if isinstance(obj_bp, np.ndarray) else obj_bp
    out = {}
    for name in ['Ntrials','hit','miss','no','early','autowater','bitRand','R','L']:
        if hasattr(bp, name):
            out[name] = np.array(getattr(bp, name))
    if hasattr(bp, 'ev'):
        ev = bp.ev
        out['ev'] = {}
        for name in ['bitStart','sample','delay','goCue','reward','lickL','lickR']:
            if hasattr(ev, name):
                out['ev'][name] = np.array(getattr(ev, name))
    if hasattr(bp, 'protocol'):
        prot = bp.protocol
        out['protocol'] = {}
        for name in ['types','nums']:
            if hasattr(prot, name):
                out['protocol'][name] = np.array(getattr(prot, name))
    return out


def scipy_read_units(obj_clu):
    clu_cell = obj_clu.flat[0] if isinstance(obj_clu, np.ndarray) else obj_clu
    units = []
    for i, u in enumerate(clu_cell.flat):
        unit = {'probe_idx': 0}
        for key in ['tm','trial','trialtm','quality','site','channel']:
            if hasattr(u, key):
                unit[key] = np.array(getattr(u, key)).squeeze()
        units.append(unit)
    return units


def scipy_align_tongue(obj_traj, bp, n_trials, time):
    try:
        cam = obj_traj.flat[0]
    except Exception:
        return None
    out = np.full((n_trials, time.size), np.nan, dtype=np.float32)
    align_times = np.ravel(bp['ev']['goCue']).astype(float)[:n_trials]
    trials = list(cam.flat)
    n = min(n_trials, len(trials))
    for tr in range(n):
        t = trials[tr]
        if not hasattr(t, 'ts') or not hasattr(t, 'frameTimes'):
            continue
        ts = np.array(t.ts)
        ft = np.array(t.frameTimes).astype(float).ravel()
        # scipy layout is (time, 3, landmarks)
        if ts.ndim != 3 or ts.shape[1] < 3 or ts.shape[2] < 1:
            continue
        x = ts[:, 0, 0].astype(float)
        y = ts[:, 1, 0].astype(float)
        p = ts[:, 2, 0].astype(float)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(ft) & (p > 0.5)
        if valid.sum() < 3:
            continue
        xv, yv, tv = x[valid], y[valid], ft[valid] - align_times[tr]
        dt = np.diff(tv)
        good = dt > 0
        if good.sum() < 2:
            continue
        speed = np.sqrt(np.diff(xv)**2 + np.diff(yv)**2) / dt
        tmid = ((tv[:-1] + tv[1:]) / 2)[good]
        speed = speed[good]
        finite = np.isfinite(tmid) & np.isfinite(speed)
        if finite.sum() < 2:
            continue
        yint = np.interp(time, tmid[finite], speed[finite], left=np.nan, right=np.nan)
        idx = np.where(np.isfinite(yint))[0]
        if idx.size == 0:
            continue
        yint[:idx[0]] = np.nan
        yint[idx[-1]+1:] = np.nan
        out[tr] = yint.astype(np.float32)
    return out


def build_session_scipy(sess: SessionInfo):
    m = sio.loadmat(sess.data_file, squeeze_me=False, struct_as_record=False)
    obj = m['obj'].flat[0]
    bp = scipy_obj_to_bp(obj.bp)
    n_trials = int(np.array(bp['Ntrials']).squeeze())
    units = scipy_read_units(obj.clu)
    units = filter_units(units, n_trials)
    neural_trials = build_neural_trials(units, n_trials)
    brain_region_idx = np.zeros(len(units), dtype=np.int64)
    time = common_time_axis()
    tongue_aligned = scipy_align_tongue(obj.traj, bp, n_trials, time)
    outcomes = infer_outcomes(bp, n_trials)
    lick = infer_lick_direction(bp, outcomes, n_trials)
    context = infer_context(sess, bp, n_trials)
    me = load_motion_energy(sess.motion_file)
    motion_aligned = align_motion_energy(me, bp, n_trials, time)
    motion_disc = discretize_session_median(motion_aligned, missing_code=2) if motion_aligned is not None else None
    tongue_disc = discretize_session_median(tongue_aligned, missing_code=2) if tongue_aligned is not None else None
    input_trials, output_trials = [], []
    for tr in range(n_trials):
        input_trials.append(time[None, :].astype(np.float32))
        tongue = tongue_disc[tr] if tongue_disc is not None else np.full(time.shape, 2, dtype=np.int64)
        paw = np.full(time.shape, 2, dtype=np.int64)
        motion = motion_disc[tr] if motion_disc is not None else np.full(time.shape, 2, dtype=np.int64)
        out = np.vstack([
            np.full((1, time.size), lick[tr], dtype=np.int64),
            np.full((1, time.size), context[tr], dtype=np.int64),
            np.full((1, time.size), outcomes[tr], dtype=np.int64),
            tongue[None, :],
            paw[None, :],
            motion[None, :],
        ])
        output_trials.append(out)
    return neural_trials, input_trials, output_trials, sess.animal, brain_region_idx
