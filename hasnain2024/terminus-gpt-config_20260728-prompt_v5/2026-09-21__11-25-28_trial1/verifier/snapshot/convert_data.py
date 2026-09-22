#!/usr/bin/env python3
import argparse
import math
import os
import pickle
import re
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio

BASE = Path('/app/data/RandomizedDelay_Ephys_Behavior')

OUTPUT_NAMES = [
    'lick_direction',
    'behavioral_context',
    'outcome',
    'tongue_velocity',
    'paw_velocity',
    'motion_energy',
]
OUTPUT_VALUES = [
    ['left', 'right', 'none'],
    ['WC', 'DR'],
    ['incorrect', 'correct', 'ignore'],
    ['lt50', 'ge50', 'not_visible'],
    ['lt50', 'ge50', 'not_visible'],
    ['lt50', 'ge50', 'no_video'],
]
INPUT_NAMES = ['time_from_go_cue']

KEEP_QUALITY = {'multi', 'fair', 'good', 'great', 'excellent'}


def normq(q):
    q = str(q).strip().lower().replace('\x00', '')
    if q == 'mutli':
        q = 'multi'
    return q


def gaussian_smooth(x, sigma_bins):
    if sigma_bins is None or sigma_bins <= 0:
        return x
    radius = max(1, int(math.ceil(4 * sigma_bins)))
    t = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (t / sigma_bins) ** 2)
    k /= k.sum()
    return np.convolve(x, k, mode='same')


def decode_h5_char(h, ref):
    arr = np.array(h[ref][()]).squeeze()
    return ''.join(chr(int(v)) for v in np.ravel(arr))


def load_motion_energy_sidecar(session_name):
    f = BASE / f'motionEnergy_{session_name}.mat'
    if not f.exists():
        return None, None
    try:
        me = sio.loadmat(str(f), squeeze_me=True, struct_as_record=False)['me']
        if hasattr(me, 'data') and hasattr(me, 'moveThresh'):
            return np.array(me.data, dtype=object), float(me.moveThresh)
        if isinstance(me, np.ndarray) and me.dtype.names is not None:
            names = set(me.dtype.names)
            if 'data' in names:
                data = np.array(me['data'].item() if me.shape == () else me['data'], dtype=object)
                thresh = None
                if 'moveThresh' in names:
                    try:
                        thresh = float(me['moveThresh'].item() if me.shape == () else np.array(me['moveThresh']).squeeze()[0])
                    except Exception:
                        thresh = None
                return data, thresh
    except Exception:
        pass
    return None, None


def load_session(path):
    try:
        with h5py.File(path, 'r') as h:
            return load_session_h5(path, h)
    except OSError:
        mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
        return load_session_mat(path, mat['obj'])


def load_session_h5(path, h):
    obj = h['obj']
    bp = obj['bp']
    ntrials = int(np.array(bp['Ntrials'][()]).squeeze())
    out = {
        'format': 'h5py',
        'session_name': path.stem.replace('data_structure_', ''),
        'subject': path.stem.split('_')[2],
        'ntrials': ntrials,
        'goCue': np.array(bp['ev']['goCue'][()]).squeeze().astype(float),
        'sample': np.array(bp['ev']['sample'][()]).squeeze().astype(float) if 'sample' in bp['ev'] else None,
        'reward': np.array(bp['ev']['reward'][()]).squeeze().astype(float) if 'reward' in bp['ev'] else None,
        'R': np.array(bp['R'][()]).squeeze().astype(bool),
        'L': np.array(bp['L'][()]).squeeze().astype(bool),
        'hit': np.array(bp['hit'][()]).squeeze().astype(bool),
        'miss': np.array(bp['miss'][()]).squeeze().astype(bool),
        'no': np.array(bp['no'][()]).squeeze().astype(bool),
        'early': np.array(bp['early'][()]).squeeze().astype(bool) if 'early' in bp else np.zeros(ntrials, dtype=bool),
        'autowater': np.array(bp['autowater'][()]).squeeze().astype(bool),
        'has_clu': 'clu' in obj,
        'brain_region': 'ALM',
    }
    if out['has_clu']:
        clu = h[obj['clu'][0, 0]]
        units = []
        for i in range(clu['tm'].shape[0]):
            q = normq(decode_h5_char(h, clu['quality'][i, 0]))
            tm = np.array(h[clu['tm'][i, 0]][()]).squeeze().astype(float)
            trial = np.array(h[clu['trial'][i, 0]][()]).squeeze().astype(int)
            trialtm = np.array(h[clu['trialtm'][i, 0]][()]).squeeze().astype(float)
            site = int(np.array(h[clu['site'][i, 0]][()]).squeeze()) if 'site' in clu else -1
            units.append({'quality': q, 'tm': tm, 'trial': trial, 'trialtm': trialtm, 'site': site})
        out['units'] = units
    else:
        out['units'] = []
    out['traj'] = None
    out['me_obj'] = None
    return out


def load_session_mat(path, obj):
    ntrials = int(np.array(obj.bp.Ntrials).squeeze())
    units = []
    if hasattr(obj, 'clu'):
        clu = obj.clu
        if not isinstance(clu, np.ndarray):
            clu = np.array([clu], dtype=object)
        for u in clu:
            units.append({
                'quality': normq(u.quality),
                'tm': np.array(u.tm).squeeze().astype(float),
                'trial': np.array(u.trial).squeeze().astype(int),
                'trialtm': np.array(u.trialtm).squeeze().astype(float),
                'site': int(np.array(u.site).squeeze()) if hasattr(u, 'site') else -1,
            })
    return {
        'format': 'loadmat',
        'session_name': path.stem.replace('data_structure_', ''),
        'subject': path.stem.split('_')[2],
        'ntrials': ntrials,
        'goCue': np.array(obj.bp.ev.goCue).squeeze().astype(float),
        'sample': np.array(obj.bp.ev.sample).squeeze().astype(float) if hasattr(obj.bp.ev, 'sample') else None,
        'reward': np.array(obj.bp.ev.reward).squeeze().astype(float) if hasattr(obj.bp.ev, 'reward') else None,
        'R': np.array(obj.bp.R).squeeze().astype(bool),
        'L': np.array(obj.bp.L).squeeze().astype(bool),
        'hit': np.array(obj.bp.hit).squeeze().astype(bool),
        'miss': np.array(obj.bp.miss).squeeze().astype(bool),
        'no': np.array(obj.bp.no).squeeze().astype(bool),
        'early': np.array(obj.bp.early).squeeze().astype(bool) if hasattr(obj.bp, 'early') else np.zeros(ntrials, dtype=bool),
        'autowater': np.array(obj.bp.autowater).squeeze().astype(bool),
        'has_clu': len(units) > 0,
        'units': units,
        'traj': getattr(obj, 'traj', None),
        'me_obj': getattr(obj, 'me', None),
        'brain_region': 'ALM',
    }


def unit_fr_gt1(unit):
    tm = unit['tm']
    if tm.size < 2:
        return False
    dur = float(tm.max() - tm.min())
    if dur <= 0:
        return False
    return (tm.size / dur) > 1.0


def select_units(session):
    keep = [u for u in session['units'] if u['quality'] in KEEP_QUALITY and unit_fr_gt1(u)]
    return keep


def build_neural_trials(session, units, tmin=-2.4, tmax=2.0, dt=0.02, smooth_sigma=2.0):
    edges = np.arange(tmin, tmax + dt, dt)
    centers = edges[:-1] + dt / 2
    neural_trials = []
    for tr in range(1, session['ntrials'] + 1):
        arr = np.zeros((len(units), len(centers)), dtype=np.float32)
        for i, u in enumerate(units):
            mask = (u['trial'] == tr)
            if np.any(mask):
                counts, _ = np.histogram(u['trialtm'][mask], bins=edges)
                arr[i] = gaussian_smooth(counts.astype(np.float32) / dt, smooth_sigma)
        neural_trials.append(arr)
    return neural_trials, centers




def resample_trace_to_bins(trace, ntime):
    trace = np.asarray(trace, dtype=float).squeeze()
    if trace.ndim != 1 or trace.size == 0:
        return None
    x_old = np.linspace(0.0, 1.0, trace.size)
    x_new = np.linspace(0.0, 1.0, ntime)
    return np.interp(x_new, x_old, trace).astype(np.float32)



def extract_h5_traj_view(h, traj_group, trial_idx):
    feat_cell = h[traj_group['featNames'][trial_idx, 0]]
    feat_names = [decode_h5_char(h, r) for r in feat_cell[()].ravel()]
    ts = np.array(h[traj_group['ts'][trial_idx, 0]][()])
    frame_times = np.array(h[traj_group['frameTimes'][trial_idx, 0]][()]).squeeze().astype(float)
    return feat_names, ts, frame_times

def speed_from_ts(ts, feat_names, feature_candidates, like_thresh=0.5):
    for feat in feature_candidates:
        if feat in feat_names:
            i = feat_names.index(feat)
            xy = np.asarray(ts[i, 0:2, :], dtype=float)
            like = np.asarray(ts[i, 2, :], dtype=float)
            valid = np.isfinite(xy).all(axis=0) & np.isfinite(like) & (like >= like_thresh)
            if valid.sum() < 2:
                continue
            dx = np.diff(xy[0], prepend=np.nan)
            dy = np.diff(xy[1], prepend=np.nan)
            spd = np.sqrt(dx**2 + dy**2)
            spd[~valid] = np.nan
            return spd, valid
    return None, None

def build_video_outputs_h5(session, ntime):
    ntr = session['ntrials']
    tongue_vals = []
    paw_vals = []
    tongue_res = [None] * ntr
    paw_res = [None] * ntr
    with h5py.File(BASE / f"data_structure_{session['session_name']}.mat", 'r') as h:
        traj_refs = h['obj']['traj'][()]
        if traj_refs.size < 2:
            return None, None
        view0 = h[traj_refs[0,0] if traj_refs.ndim == 2 else traj_refs.ravel()[0]]
        view1 = h[traj_refs[1,0] if traj_refs.ndim == 2 else traj_refs.ravel()[1]]
        for tr in range(ntr):
            try:
                feat0, ts0, ft0 = extract_h5_traj_view(h, view0, tr)
                spd0, valid0 = speed_from_ts(ts0, feat0, ['tongue','left_tongue','right_tongue'])
                if spd0 is not None:
                    rr = resample_trace_to_bins(np.nan_to_num(spd0, nan=np.nanmedian(spd0[np.isfinite(spd0)]) if np.isfinite(spd0).any() else 0.0), ntime)
                    vv = resample_trace_to_bins(valid0.astype(float), ntime)
                    tongue_res[tr] = (rr, vv >= 0.5)
                    tongue_vals.extend(rr[(vv >= 0.5)].tolist())
            except Exception:
                pass
            try:
                feat1, ts1, ft1 = extract_h5_traj_view(h, view1, tr)
                spd1, valid1 = speed_from_ts(ts1, feat1, ['top_paw','bottom_paw'])
                if spd1 is not None:
                    rr = resample_trace_to_bins(np.nan_to_num(spd1, nan=np.nanmedian(spd1[np.isfinite(spd1)]) if np.isfinite(spd1).any() else 0.0), ntime)
                    vv = resample_trace_to_bins(valid1.astype(float), ntime)
                    paw_res[tr] = (rr, vv >= 0.5)
                    paw_vals.extend(rr[(vv >= 0.5)].tolist())
            except Exception:
                pass
    tongue_out = np.full((ntr, ntime), 2, dtype=np.int64)
    paw_out = np.full((ntr, ntime), 2, dtype=np.int64)
    if len(tongue_vals) > 0:
        med = np.nanmedian(np.asarray(tongue_vals, dtype=float))
        for tr, item in enumerate(tongue_res):
            if item is not None:
                rr, valid = item
                tongue_out[tr, valid] = (rr[valid] >= med).astype(np.int64)
    if len(paw_vals) > 0:
        med = np.nanmedian(np.asarray(paw_vals, dtype=float))
        for tr, item in enumerate(paw_res):
            if item is not None:
                rr, valid = item
                paw_out[tr, valid] = (rr[valid] >= med).astype(np.int64)
    return tongue_out, paw_out

def build_outputs(session, ntime):
    ntr = session['ntrials']
    lick = np.full(ntr, 2, dtype=np.int64)
    lick[session['L']] = 0
    lick[session['R']] = 1
    context = np.where(session['autowater'], 0, 1).astype(np.int64)  # WC, DR
    outcome = np.full(ntr, 2, dtype=np.int64)
    outcome[session['miss']] = 0
    outcome[session['hit']] = 1
    outcome[session['no']] = 2

    tongue = np.full((ntr, ntime), 2, dtype=np.int64)
    paw = np.full((ntr, ntime), 2, dtype=np.int64)
    motion = np.full((ntr, ntime), 2, dtype=np.int64)

    if session.get('format') == 'h5py':
        try:
            t_out, p_out = build_video_outputs_h5(session, ntime)
            if t_out is not None:
                tongue = t_out
            if p_out is not None:
                paw = p_out
        except Exception:
            pass

    me_data, me_thresh = load_motion_energy_sidecar(session['session_name'])
    try:
        if me_data is not None:
            trial_traces = np.asarray(me_data, dtype=object).ravel()
            if trial_traces.shape[0] == ntr:
                session_vals = []
                resampled = []
                for trc in trial_traces:
                    rr = resample_trace_to_bins(trc, ntime)
                    if rr is None:
                        resampled.append(None)
                    else:
                        resampled.append(rr)
                        session_vals.extend(rr.tolist())
                if len(session_vals) > 0:
                    med = np.nanmedian(np.asarray(session_vals, dtype=float))
                    for i, rr in enumerate(resampled):
                        if rr is not None:
                            motion[i] = (rr >= med).astype(np.int64)
        elif session.get('me_obj', None) is not None:
            vals = np.asarray(session['me_obj']).squeeze()
            if vals.ndim == 1 and vals.shape[0] == ntr and np.issubdtype(vals.dtype, np.number):
                vals = vals.astype(float)
                med = np.nanmedian(vals)
                cls = (vals >= med).astype(np.int64)
                motion[:] = cls[:, None]
    except Exception:
        pass

    outputs = []
    for tr in range(ntr):
        out = np.vstack([
            np.full(ntime, lick[tr], dtype=np.int64),
            np.full(ntime, context[tr], dtype=np.int64),
            np.full(ntime, outcome[tr], dtype=np.int64),
            tongue[tr],
            paw[tr],
            motion[tr],
        ])
        outputs.append(out)
    return outputs


def build_inputs(centers, ntrials):
    arr = centers.astype(np.float32)[None, :]
    return [arr.copy() for _ in range(ntrials)]


def convert(sample=False):
    files = sorted(BASE.glob('data_structure_*.mat'))
    if sample:
        files = files[:2]
    sessions = [load_session(f) for f in files]
    sessions = [s for s in sessions if s['has_clu']]
    sessions = [s for s in sessions if len(select_units(s)) >= 10]

    subjects = sorted({s['subject'] for s in sessions})
    subject_map = {s: i for i, s in enumerate(subjects)}
    brain_regions = ['ALM']

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
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'task_description': 'Randomized-delay licking task aligned to go cue; decode lick direction, context, outcome, tongue velocity, paw velocity, and motion energy from ALM neural activity.',
            'time_bin_size': 20.0,
            'temporal_alignment_event': 'Go cue onset',
            'off_start': -2.4,
            'off_end': 2.0,
            'unit_filter': 'quality in {multi,fair,good,great,excellent} and firing rate > 1 Hz; sessions require >=10 retained units',
            'session_exclusion_note': 'Sessions lacking neural data (`clu`) are excluded.',
        },
    }

    for si, sess in enumerate(sessions, start=1):
        t_sess = time.time()
        units = select_units(sess)
        print(f'processing_session {si}/{len(sessions)} {sess["session_name"]} ntrials={sess["ntrials"]} nunits={len(units)}', flush=True)
        neural_trials, centers = build_neural_trials(sess, units)
        inputs = build_inputs(centers, sess['ntrials'])
        outputs = build_outputs(sess, len(centers))
        data['neural'].append(neural_trials)
        data['input'].append(inputs)
        data['output'].append(outputs)
        data['subject_idx'].append(subject_map[sess['subject']])
        data['brain_region_idx'].append(np.zeros(len(units), dtype=np.int64))
        print(f'finished_session {sess["session_name"]} elapsed_sec={time.time()-t_sess:.3f}', flush=True)

    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true')
    g.add_argument('--sample', action='store_true')
    ap.add_argument('--show-processing', action='store_true')
    args = ap.parse_args()

    t0 = time.time()
    data = convert(sample=args.sample)
    with open(args.outpicklefile, 'wb') as f:
        pickle.dump(data, f)
    print('wrote', args.outpicklefile)
    print('n_sessions', len(data['neural']))
    print('elapsed_sec', time.time() - t0)

if __name__ == '__main__':
    main()
