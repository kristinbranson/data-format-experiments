#!/usr/bin/env python3
import argparse
import os
import re
import time
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import mat73
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
    ['lt_median', 'ge_median', 'not_visible'],
    ['lt_median', 'ge_median', 'not_visible'],
    ['lt_median', 'ge_median', 'no_video'],
]
INPUT_NAMES = ['time_from_go_cue']


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true', help='Process all sessions')
    g.add_argument('--sample', action='store_true', help='Process only 2 sessions for testing')
    ap.add_argument('--show-processing', action='store_true')
    return ap.parse_args()


def discover_sessions(base=Path('/app/data')):
    # Start with Ephys_Behavior because it matches the paper's 25-session DR dataset
    files = sorted((base / 'Ephys_Behavior').glob('data_structure_*.mat'))
    return files


def flatten_units(clu_list):
    units = []
    if not isinstance(clu_list, list):
        return units
    for group in clu_list:
        if not isinstance(group, dict):
            continue
        n = min(len(group.get('tm', [])), len(group.get('trial', [])), len(group.get('trialtm', [])))
        for i in range(n):
            units.append({
                'quality': group.get('quality', [None] * n)[i] if len(group.get('quality', [])) > i else None,
                'site': group.get('site', [None] * n)[i] if len(group.get('site', [])) > i else None,
                'tm': np.asarray(group.get('tm', [])[i]).astype(np.float32),
                'trial': np.asarray(group.get('trial', [])[i]).astype(np.int32),
                'trialtm': np.asarray(group.get('trialtm', [])[i]).astype(np.float32),
            })
    return units


def quality_ok(q):
    if q is None:
        return True
    if isinstance(q, str):
        qs = q.strip().lower()
        return qs not in {'garbage', 'noisy'}
    return True


def infer_context_per_trial(bp):
    ntr = int(np.asarray(bp['Ntrials']).item())
    stim_enable = np.asarray(bp.get('stim', {}).get('enable', np.zeros(ntr))).astype(bool)
    autowater = np.asarray(bp.get('autowater', np.zeros(ntr))).astype(bool)
    early = np.asarray(bp.get('early', np.zeros(ntr))).astype(bool)
    out = np.full(ntr, -1, dtype=np.int64)
    # Reference code: DR = ~stim.enable & ~autowater & ~early ; WC = ~stim.enable & autowater & ~early
    dr = (~stim_enable) & (~autowater) & (~early)
    wc = (~stim_enable) & autowater & (~early)
    out[wc] = 0
    out[dr] = 1
    labels = np.array(['unknown'] * ntr, dtype=object)
    labels[wc] = 'WC'
    labels[dr] = 'DR'
    return out, labels


def infer_lick_direction(bp):
    L = np.asarray(bp.get('L')).astype(bool)
    R = np.asarray(bp.get('R')).astype(bool)
    no = np.asarray(bp.get('no')).astype(bool)
    out = np.full(L.shape[0], 2, dtype=np.int64)
    out[L] = 0
    out[R] = 1
    out[~(L | R) | no] = 2
    return out


def infer_outcome(bp):
    hit = np.asarray(bp.get('hit')).astype(bool)
    miss = np.asarray(bp.get('miss')).astype(bool)
    no = np.asarray(bp.get('no')).astype(bool)
    early = np.asarray(bp.get('early', np.zeros_like(hit))).astype(bool)
    out = np.full(hit.shape[0], 2, dtype=np.int64)
    out[hit] = 1
    out[miss] = 0
    out[no | early] = 2
    return out


def bin_unit_trialtm(units, n_trials, t_edges):
    n_units = len(units)
    n_bins = len(t_edges) - 1
    trial_mats = [np.zeros((n_units, n_bins), dtype=np.float32) for _ in range(n_trials)]
    for ui, u in enumerate(units):
        tr = u['trial']
        tt = u['trialtm']
        keep = np.isfinite(tr) & np.isfinite(tt)
        tr = tr[keep].astype(int)
        tt = tt[keep]
        for tr_ix in np.unique(tr):
            if tr_ix < 1 or tr_ix > n_trials:
                continue
            x = tt[tr == tr_ix]
            counts, _ = np.histogram(x, bins=t_edges)
            trial_mats[tr_ix - 1][ui] = counts.astype(np.float32)
    return trial_mats


def discretize_trace_per_session(values, visible_mask):
    arr = np.asarray(values, dtype=np.float32)
    out = np.full(arr.shape, 2, dtype=np.int64)
    vis = np.asarray(visible_mask).astype(bool)
    if np.any(vis):
        thr = np.nanmedian(arr[vis])
        out[vis] = (arr[vis] >= thr).astype(np.int64)
    return out


def placeholder_timevarying_outputs(n_trials, n_bins, have_vid):
    tongue = []
    paw = []
    me = []
    for i in range(n_trials):
        hv = bool(have_vid[i]) if i < len(have_vid) else False
        if hv:
            z = np.zeros(n_bins, dtype=np.int64)
            tongue.append(z.copy())
            paw.append(z.copy())
            me.append(z.copy())
        else:
            tongue.append(np.full(n_bins, 2, dtype=np.int64))
            paw.append(np.full(n_bins, 2, dtype=np.int64))
            me.append(np.full(n_bins, 2, dtype=np.int64))
    return tongue, paw, me




def load_motion_energy_for_session(session_path):
    m = re.match(r'data_structure_([^_]+)_(\d{4}-\d{2}-\d{2})\.mat', session_path.name)
    if not m:
        return None
    subj, date = m.groups()
    mefile = session_path.parent / f'motionEnergy_{subj}_{date}.mat'
    if not mefile.exists():
        return None
    import scipy.io as sio
    return sio.loadmat(str(mefile), squeeze_me=True, struct_as_record=False).get('me')


def bin_timeseries_to_edges(times, values, t_edges):
    times = np.asarray(times)
    values = np.asarray(values)
    out = np.full(len(t_edges)-1, np.nan, dtype=np.float32)
    for i in range(len(t_edges)-1):
        m = (times >= t_edges[i]) & (times < t_edges[i+1]) & np.isfinite(values)
        if np.any(m):
            out[i] = np.nanmean(values[m])
    return out


def extract_speed_from_traj_cam(cam, trial_idx, feature_keywords):
    ts = np.asarray(cam['ts'][trial_idx])
    ft = np.asarray(cam['frameTimes'][trial_idx]).astype(np.float32)
    feat_names = [x[0] if isinstance(x, list) and len(x)==1 else str(x) for x in cam['featNames'][trial_idx]]
    feat_idx = [i for i, name in enumerate(feat_names) if any(k in str(name).lower() for k in feature_keywords)]
    if len(feat_idx) == 0 or ts.ndim != 3 or ts.shape[0] != ft.shape[0]:
        return ft, np.full(ft.shape, np.nan, dtype=np.float32), np.zeros(ft.shape, dtype=bool)
    xy = ts[:, :2, :][:, :, feat_idx]
    conf = ts[:, 2, :][:, feat_idx]
    visible = np.any(np.isfinite(xy), axis=(1,2)) & np.any(conf > 0.5, axis=1)
    mean_xy = np.nanmean(xy, axis=2)
    dxy = np.diff(mean_xy, axis=0)
    dt = np.diff(ft)
    speed = np.full(ft.shape, np.nan, dtype=np.float32)
    good = np.isfinite(dxy).all(axis=1) & np.isfinite(dt) & (dt > 0)
    sp = np.full(dt.shape, np.nan, dtype=np.float32)
    sp[good] = np.sqrt((dxy[good]**2).sum(axis=1)) / dt[good]
    speed[1:] = sp
    visible[~np.isfinite(speed)] = False
    return ft, speed, visible

def process_session(path, show_processing=False, sample_plot_dir=Path('/app')):
    t0 = time.time()
    obj = mat73.loadmat(str(path))['obj']
    bp = obj['bp']
    n_trials = int(np.asarray(bp['Ntrials']).item())
    units = [u for u in flatten_units(obj.get('clu', [])) if quality_ok(u.get('quality'))]

    # trial validity
    have_ephys = np.asarray(obj.get('trials', {}).get('bp', {}).get('haveEphys', np.ones(n_trials))).astype(bool)
    have_vid = np.asarray(obj.get('trials', {}).get('bp', {}).get('haveVid', np.zeros(n_trials))).astype(bool)

    # fixed binning around go cue using trial-relative spike times
    bin_size = 0.01
    t_edges = np.arange(-2.5, 2.5001, bin_size)
    t_centers = (t_edges[:-1] + t_edges[1:]) / 2

    neural_all = bin_unit_trialtm(units, n_trials, t_edges)
    context, context_labels = infer_context_per_trial(bp)
    lick_dir = infer_lick_direction(bp)
    outcome = infer_outcome(bp)
    tongue_tv, paw_tv, me_tv = [], [], []
    me_struct = load_motion_energy_for_session(path)
    cam_for_kin = obj['traj'][1] if isinstance(obj.get('traj'), list) and len(obj.get('traj')) > 1 else None
    for tr in range(n_trials):
        if cam_for_kin is not None and tr < len(cam_for_kin['ts']):
            tt, tongue_speed, tongue_vis = extract_speed_from_traj_cam(cam_for_kin, tr, ['tongue'])
            pt, paw_speed, paw_vis = extract_speed_from_traj_cam(cam_for_kin, tr, ['paw'])
            tongue_tv.append(discretize_trace_per_session(bin_timeseries_to_edges(tt, tongue_speed, t_edges), bin_timeseries_to_edges(tt, tongue_vis.astype(float), t_edges) > 0))
            paw_tv.append(discretize_trace_per_session(bin_timeseries_to_edges(pt, paw_speed, t_edges), bin_timeseries_to_edges(pt, paw_vis.astype(float), t_edges) > 0))
        else:
            tongue_tv.append(np.full(len(t_centers), 2, dtype=np.int64))
            paw_tv.append(np.full(len(t_centers), 2, dtype=np.int64))
        me_data = getattr(me_struct, 'data', None) if me_struct is not None else None
        if me_data is not None:
            try:
                me_seq = list(me_data)
            except TypeError:
                me_seq = np.asarray(me_data).ravel().tolist()
        else:
            me_seq = None
        if me_seq is not None and tr < len(me_seq):
            try:
                md = np.asarray(me_seq[tr]).astype(np.float32).squeeze()
            except Exception:
                md = None
            if md is not None and getattr(md, 'ndim', None) == 1 and md.shape[0] > 0:
                mt = np.linspace(t_edges[0], t_edges[-1], num=md.shape[0], dtype=np.float32)
                mb = bin_timeseries_to_edges(mt, md, t_edges)
                me_tv.append(discretize_trace_per_session(mb, np.isfinite(mb)))
            else:
                me_tv.append(np.full(len(t_centers), 2, dtype=np.int64))
        else:
            me_tv.append(np.full(len(t_centers), 2, dtype=np.int64))

    keep_trials = [i for i in range(n_trials) if have_ephys[i] and context[i] >= 0]
    neural = [neural_all[i] for i in keep_trials]
    inputs = [t_centers[None, :].astype(np.float32) for _ in keep_trials]
    outputs = []
    for i in keep_trials:
        out = np.vstack([
            np.full(len(t_centers), lick_dir[i], dtype=np.int64),
            np.full(len(t_centers), context[i], dtype=np.int64),
            np.full(len(t_centers), outcome[i], dtype=np.int64),
            tongue_tv[i],
            paw_tv[i],
            me_tv[i],
        ])
        outputs.append(out)

    subject = obj.get('meta', {}).get('anm', re.search(r'data_structure_([^_]+)_', path.name).group(1))
    # region currently unavailable from inspected fields; use site-based placeholder single region
    region_names = ['ALM']
    region_idx = np.zeros(len(units), dtype=np.int64)

    if show_processing:
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
        axs[0].imshow(neural[0], aspect='auto', origin='lower')
        axs[0].set_title(f'{path.stem}: trial 1 neural binned activity')
        axs[1].plot(t_centers, outputs[0][0], label='lick_dir')
        axs[1].plot(t_centers, outputs[0][1], label='context')
        axs[1].plot(t_centers, outputs[0][2], label='outcome')
        axs[1].legend(loc='upper right', ncol=3)
        axs[2].plot(t_centers, outputs[0][3], label='tongue')
        axs[2].plot(t_centers, outputs[0][4], label='paw')
        axs[2].plot(t_centers, outputs[0][5], label='motion')
        axs[2].legend(loc='upper right', ncol=3)
        fig.savefig(sample_plot_dir / f'processing_{path.stem}.png', dpi=150)
        plt.close(fig)

    info = {
        'subject': subject,
        'brain_regions': region_names,
        'brain_region_idx': region_idx,
        'n_units': len(units),
        'n_trials_total': n_trials,
        'n_trials_kept': len(keep_trials),
        'context_labels_sample': list(context_labels[:10]),
        'elapsed_sec': time.time() - t0,
    }
    print(f'Processed {path.name}: units={len(units)} kept_trials={len(keep_trials)}/{n_trials} elapsed={info["elapsed_sec"]:.2f}s')
    return neural, inputs, outputs, info


def main():
    args = parse_args()
    session_files = discover_sessions()
    if args.sample:
        session_files = session_files[:2]
    print(f'Found {len(session_files)} sessions to process')

    neural_all, input_all, output_all = [], [], []
    subjects = []
    subject_map = {}
    subject_idx = []
    brain_regions = ['ALM']
    brain_region_idx = []
    session_info = []

    for sf in session_files:
        neural, inputs, outputs, info = process_session(sf, show_processing=args.show_processing)
        if len(neural) < 2:
            print(f'Skipping {sf.name}: fewer than 2 valid trials')
            continue
        if info['n_units'] < 10:
            print(f"Skipping {sf.name}: only {info['n_units']} units (<10 inclusion threshold)")
            continue
        neural_all.append(neural)
        input_all.append(inputs)
        output_all.append(outputs)
        subj = info['subject']
        if subj not in subject_map:
            subject_map[subj] = len(subjects)
            subjects.append(subj)
        subject_idx.append(subject_map[subj])
        brain_region_idx.append(info['brain_region_idx'])
        session_info.append({'file': sf.name, **info})

    data = {
        'neural': neural_all,
        'input': input_all,
        'output': output_all,
        'subjects': subjects,
        'subject_idx': np.asarray(subject_idx, dtype=np.int64),
        'brain_regions': brain_regions,
        'brain_region_idx': brain_region_idx,
        'input_names': INPUT_NAMES,
        'output_names': OUTPUT_NAMES,
        'output_values': OUTPUT_VALUES,
        'metadata': {
            'task_description': 'Mouse ALM electrophysiology during directional licking task; decode lick direction, context, outcome, and video-derived movement variables from neural activity.',
            'time_bin_size': 10.0,
            'temporal_alignment_event': 'Go cue onset',
            'off_start': -2.5,
            'off_end': 2.5,
            'session_info': session_info,
            'source_subset': 'Ephys_Behavior',
            'notes': 'Initial conversion focused on Ephys_Behavior sessions; video-derived outputs currently use availability placeholders pending deeper traj/me parsing.'
        }
    }

    out = Path(args.outpicklefile)
    with out.open('wb') as f:
        pickle.dump(data, f)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
