#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
import numpy as np
import scipy.io
import h5py

BIN_SIZE = 0.075
T_START = -2.5
T_END = 2.5
RAND_SESSIONS = [
    'JEB11_2022-05-10','JEB11_2022-05-11',
    'JEB12_2022-05-12','JEB12_2022-05-13',
    'JEB23_2023-10-10','JEB23_2023-10-11','JEB23_2023-10-12','JEB23_2023-10-13','JEB23_2023-10-18','JEB23_2023-10-19','JEB23_2023-10-21',
    'JEB24_2023-10-23','JEB24_2023-10-24','JEB24_2023-10-25','JEB24_2023-10-26','JEB24_2023-10-27','JEB24_2023-10-31','JEB24_2023-11-02','JEB24_2023-11-03'
]


def decode_char(arr):
    arr = np.asarray(arr)
    if arr.dtype.kind in ('u', 'i'):
        return ''.join(chr(int(x)) for x in arr.ravel() if int(x) != 0)
    return str(arr)


def deref(f, x):
    return f[x]


def unwrap_obj(x):
    while isinstance(x, np.ndarray) and x.dtype == object:
        if x.size == 0:
            return x
        x = x.flat[0]
    return x




def normalize_motion_elem(x):
    while True:
        if hasattr(x, '_fieldnames'):
            if hasattr(x, 'data'):
                x = x.data
                continue
            x = getattr(x, x._fieldnames[0])
            continue
        if isinstance(x, np.ndarray) and x.dtype == object:
            if x.size == 0:
                return np.array([], dtype=np.float32)
            x = x.flat[0]
            continue
        return np.asarray(x).ravel().astype(np.float32)

def load_motion_energy(path):
    me = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)['me']
    move_thresh = getattr(me, 'moveThresh', None) if hasattr(me, '_fieldnames') else None
    raw = me
    while hasattr(raw, '_fieldnames') and hasattr(raw, 'data'):
        raw = raw.data
    arr = np.asarray(raw, dtype=object)
    out = [normalize_motion_elem(x) for x in arr.flat]
    return out, move_thresh


def extract_bp_h5(f):
    bp = f['obj/bp']
    out = {}
    for k in ['L', 'R', 'autowater', 'bitRand', 'early', 'hit', 'miss', 'no']:
        out[k] = np.asarray(bp[k][()]).astype(np.float32).ravel()
    out['Ntrials'] = int(np.asarray(bp['Ntrials'][()]).squeeze())
    ev = f['obj/bp/ev']
    for k in ['bitStart', 'sample', 'delay', 'goCue', 'reward']:
        out[k] = np.asarray(ev[k][()]).astype(np.float32).ravel()
    return out


def extract_clu_h5(f):
    clu = deref(f, f['obj/clu'][()].flat[0])
    trialtm = [np.asarray(deref(f, r)[()]).ravel().astype(np.float32) for r in clu['trialtm'][()].flat]
    trial = [np.asarray(deref(f, r)[()]).ravel().astype(int) for r in clu['trial'][()].flat]
    n_units = len(trialtm)
    return trialtm, trial, n_units


def get_traj_stream_h5(f, stream_i):
    traj = deref(f, f['obj/traj'][()].flat[stream_i])
    feat_names = [decode_char(deref(f, rr)[()]) for rr in deref(f, traj['featNames'][()].flat[0])[()].flat]
    ts_refs = traj['ts'][()]
    ft_refs = traj['frameTimes'][()]
    def get_trial(trial_i):
        ts = np.asarray(deref(f, ts_refs[trial_i, 0])[()])
        ft = np.asarray(deref(f, ft_refs[trial_i, 0])[()]).ravel().astype(np.float32)
        return ts, ft
    return feat_names, get_trial


def extract_bp_old(obj):
    bp = obj.bp
    out = {}
    for k in ['L', 'R', 'autowater', 'bitRand', 'early', 'hit', 'miss', 'no']:
        out[k] = np.asarray(getattr(bp, k)).astype(np.float32).ravel()
    out['Ntrials'] = int(np.asarray(bp.Ntrials).squeeze())
    for k in ['bitStart', 'sample', 'delay', 'goCue', 'reward']:
        out[k] = np.asarray(getattr(bp.ev, k)).astype(np.float32).ravel()
    return out


def extract_clu_old(obj):
    clu_arr = np.asarray(obj.clu, dtype=object).ravel()
    trialtm = [np.asarray(unwrap_obj(c).trialtm).ravel().astype(np.float32) for c in clu_arr]
    trial = [np.asarray(unwrap_obj(c).trial).ravel().astype(int) for c in clu_arr]
    return trialtm, trial, len(clu_arr)


def get_traj_stream_old(obj, stream_i):
    stream_trials = np.asarray(obj.traj, dtype=object).ravel()[stream_i]
    first_trial = unwrap_obj(np.asarray(stream_trials, dtype=object).ravel()[0])
    feat_names = [str(x) for x in np.asarray(first_trial.featNames).ravel()]
    def get_trial(trial_i):
        tr = unwrap_obj(np.asarray(stream_trials, dtype=object).ravel()[trial_i])
        ts = np.asarray(unwrap_obj(tr.ts))
        if ts.ndim == 3 and ts.shape[1] == 3:
            ts = np.transpose(ts, (2, 1, 0))
        ft = np.asarray(unwrap_obj(tr.frameTimes)).ravel().astype(np.float32)
        return ts, ft
    return feat_names, get_trial


def bin_unit_spikes_for_trials(trialtm_list, trial_list, n_trials, time_edges):
    n_units = len(trialtm_list)
    n_bins = len(time_edges) - 1
    per_trial = [np.zeros((n_units, n_bins), dtype=np.float32) for _ in range(n_trials)]
    for ui, (ttm, tr) in enumerate(zip(trialtm_list, trial_list)):
        for trial_idx in range(1, n_trials + 1):
            mask = tr == trial_idx
            if np.any(mask):
                counts, _ = np.histogram(ttm[mask], bins=time_edges)
                per_trial[trial_idx - 1][ui] = counts.astype(np.float32)
    return per_trial


def make_time_input(n_trials, time_centers):
    return [time_centers[None, :].astype(np.float32).copy() for _ in range(n_trials)]


def nanbin_mean(times, values, edges):
    out = np.full((len(edges) - 1,), np.nan, dtype=np.float32)
    idx = np.digitize(times, edges) - 1
    for bi in range(len(out)):
        m = idx == bi
        if np.any(m):
            vv = values[m]
            if np.any(np.isfinite(vv)):
                out[bi] = np.nanmean(vv)
    return out


def speed_from_ts(ts, frame_times):
    xy = ts[:, :2, :].astype(np.float32)
    dt = np.diff(frame_times).astype(np.float32)
    dt[dt <= 0] = np.nan
    dxy = np.diff(xy, axis=2)
    spd = np.sqrt(np.nansum(dxy ** 2, axis=1)) / dt[None, :]
    tmid = (frame_times[:-1] + frame_times[1:]) / 2
    return spd, tmid


def choose_feature(names, preferred):
    for p in preferred:
        if p in names:
            return names.index(p)
    return None


def extract_binned_behavior_generic(bp, motion_data, time_edges, get_stream0, get_stream1, names0, names1):
    tongue_idx0 = choose_feature(names0, ['tongue', 'left_tongue', 'right_tongue'])
    tongue_idx1 = choose_feature(names1, ['top_tongue', 'bottom_tongue', 'topleft_tongue', 'bottomleft_tongue'])
    paw_idx1 = choose_feature(names1, ['top_paw', 'bottom_paw'])
    n_trials = bp['Ntrials']
    tongue_vals, paw_vals, me_vals = [], [], []
    for ti in range(n_trials):
        go = bp['goCue'][ti]
        tongue_bin = np.full((len(time_edges)-1,), np.nan, dtype=np.float32)
        paw_bin = np.full((len(time_edges)-1,), np.nan, dtype=np.float32)
        for idx, getter in [(tongue_idx1, get_stream1), (tongue_idx0, get_stream0)]:
            if idx is None:
                continue
            ts, ft = getter(ti)
            spd, tmid = speed_from_ts(ts, ft)
            candidate = nanbin_mean(tmid - go, spd[idx], time_edges)
            if np.any(np.isfinite(candidate)):
                tongue_bin = candidate
                break
        if paw_idx1 is not None:
            ts, ft = get_stream1(ti)
            spd, tmid = speed_from_ts(ts, ft)
            paw_bin = nanbin_mean(tmid - go, spd[paw_idx1], time_edges)
        if ti < len(motion_data):
            me = np.asarray(motion_data[ti]).ravel().astype(np.float32)
            _, ft0 = get_stream0(ti)
            me_bin = nanbin_mean(ft0 - go, me, time_edges) if len(ft0) == len(me) else np.full((len(time_edges)-1,), np.nan, dtype=np.float32)
        else:
            me_bin = np.full((len(time_edges)-1,), np.nan, dtype=np.float32)
        tongue_vals.append(tongue_bin)
        paw_vals.append(paw_bin)
        me_vals.append(me_bin)
    return tongue_vals, paw_vals, me_vals


def threshold_session_bins(arr_list):
    finite_chunks = [a[np.isfinite(a)] for a in arr_list if np.any(np.isfinite(a))]
    allv = np.concatenate(finite_chunks) if finite_chunks else np.array([])
    thr = np.nanmedian(allv) if allv.size else np.nan
    out = []
    for a in arr_list:
        b = np.zeros_like(a, dtype=np.int64)
        if np.isfinite(thr):
            valid = np.isfinite(a)
            b[valid] = (a[valid] >= thr).astype(np.int64)
        out.append(b)
    return out, thr


def process_session(data_path, motion_path):
    time_edges = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE, dtype=np.float32)
    time_centers = (time_edges[:-1] + time_edges[1:]) / 2
    motion_data, move_thresh = load_motion_energy(motion_path)
    try:
        with h5py.File(data_path, 'r') as f:
            bp = extract_bp_h5(f)
            trialtm, trial, n_units = extract_clu_h5(f)
            names0, get0 = get_traj_stream_h5(f, 0)
            names1, get1 = get_traj_stream_h5(f, 1)
            tongue_vals, paw_vals, me_vals = extract_binned_behavior_generic(bp, motion_data, time_edges, get0, get1, names0, names1)
    except OSError:
        obj = scipy.io.loadmat(data_path, squeeze_me=True, struct_as_record=False)['obj']
        bp = extract_bp_old(obj)
        trialtm, trial, n_units = extract_clu_old(obj)
        names0, get0 = get_traj_stream_old(obj, 0)
        names1, get1 = get_traj_stream_old(obj, 1)
        tongue_vals, paw_vals, me_vals = extract_binned_behavior_generic(bp, motion_data, time_edges, get0, get1, names0, names1)
    n_trials = bp['Ntrials']
    neural = bin_unit_spikes_for_trials(trialtm, trial, n_trials, time_edges)
    brain_region_idx = np.zeros((n_units,), dtype=int)
    tongue_bin, tongue_thr = threshold_session_bins(tongue_vals)
    paw_bin, paw_thr = threshold_session_bins(paw_vals)
    me_bin, me_thr = threshold_session_bins(me_vals)
    valid = (bp['early'] == 0) & (bp['no'] == 0)
    lick_dir = np.where(bp['R'] > bp['L'], 1, 0).astype(np.int64)
    context = np.where(bp['autowater'] > 0, 0, 1).astype(np.int64)
    outcome = np.where(bp['hit'] > 0, 1, 0).astype(np.int64)
    input_trials = make_time_input(n_trials, time_centers)
    output_trials = []
    for ti in range(n_trials):
        output_trials.append(np.vstack([
            np.full((len(time_centers),), lick_dir[ti], dtype=np.int64),
            np.full((len(time_centers),), context[ti], dtype=np.int64),
            np.full((len(time_centers),), outcome[ti], dtype=np.int64),
            tongue_bin[ti], paw_bin[ti], me_bin[ti],
        ]))
    keep_idx = np.where(valid)[0]
    keep_idx = [i for i in keep_idx if not np.all(neural[i] == 0)]
    neural = [neural[i] for i in keep_idx]
    input_trials = [input_trials[i] for i in keep_idx]
    output_trials = [output_trials[i] for i in keep_idx]
    info = {
        'n_trials_raw': n_trials,
        'n_trials_kept': len(keep_idx),
        'moveThresh': move_thresh,
        'tongue_threshold': float(tongue_thr) if np.isfinite(tongue_thr) else None,
        'paw_threshold': float(paw_thr) if np.isfinite(paw_thr) else None,
        'motion_threshold': float(me_thr) if np.isfinite(me_thr) else None,
    }
    return neural, input_trials, output_trials, brain_region_idx, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    args = ap.parse_args()

    base = Path('data/RandomizedDelay_Ephys_Behavior')
    keys = RAND_SESSIONS[:2] if args.sample else RAND_SESSIONS
    all_neural, all_input, all_output = [], [], []
    subject_names, subject_idx = [], []
    brain_region_idx = []
    session_info = []
    for key in keys:
        subj = key.split('_')[0]
        if subj not in subject_names:
            subject_names.append(subj)
        df = base / f'data_structure_{key}.mat'
        mf = base / f'motionEnergy_{key}.mat'
        neural, inp, out, bri, info = process_session(df, mf)
        all_neural.append(neural)
        all_input.append(inp)
        all_output.append(out)
        brain_region_idx.append(bri)
        subject_idx.append(subject_names.index(subj))
        info['session'] = key
        session_info.append(info)
        print(key, info, 'n_units', bri.shape[0], 'n_timebins', neural[0].shape[1] if neural else None)

    data = {
        'neural': all_neural,
        'input': all_input,
        'output': all_output,
        'subjects': subject_names,
        'subject_idx': np.asarray(subject_idx, dtype=int),
        'brain_regions': ['ALM'],
        'brain_region_idx': brain_region_idx,
        'input_names': ['time_from_go_cue'],
        'output_names': ['lick_direction', 'behavioral_context', 'outcome', 'tongue_velocity', 'paw_velocity', 'motion_energy'],
        'output_values': [['left', 'right'], ['WC', 'DR'], ['incorrect', 'correct'], ['low', 'high'], ['low', 'high'], ['low', 'high']],
        'metadata': {
            'task_description': 'Neural decoding aligned to go cue in randomized-delay ALM sessions',
            'time_bin_size': BIN_SIZE,
            'temporal_alignment_event': 'go cue onset',
            'off_start': T_START,
            'off_end': T_END,
            'session_info': session_info,
            'notes': 'Mixed-format randomized-delay converter using reference session list.'
        }
    }
    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    print('wrote', args.outpicklefile)


if __name__ == '__main__':
    main()
