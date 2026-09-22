#!/usr/bin/env python3
"""Convert Allen Visual Behavior Ophys NWBs to decoder format.

Usage: python -u /app/convert_data.py OUT.pkl [--full|--sample] [--show-processing]
"""
import argparse, glob, os, re, time, pickle
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

DATA_ROOT = Path('/app/data')
BIN_S = 0.100


def find_files():
    files = sorted(glob.glob(str(DATA_ROOT / '**' / 'behavior_ophys_experiment_*.nwb'), recursive=True))
    return {int(re.search(r'(\d+)\.nwb$', f).group(1)): f for f in files}


def metadata_table(name):
    paths = glob.glob(str(DATA_ROOT / '**' / name), recursive=True)
    if len(paths) != 1:
        raise RuntimeError(f'Expected one {name}, found {paths}')
    return pd.read_csv(paths[0])


def has_eye(path):
    with h5py.File(path, 'r') as h:
        return 'acquisition/EyeTracking/pupil_tracking/data' in h


def decode_strings(a):
    return np.asarray([x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in a])


def select_experiments(files):
    meta = metadata_table('ophys_experiment_table.csv')
    meta = meta[meta.ophys_experiment_id.isin(files)].copy()
    meta = meta[~meta.session_type.str.contains('passive', case=False, na=False)].copy()
    eye_map = {eid: has_eye(files[eid]) for eid in meta.ophys_experiment_id.astype(int)}
    missing = sorted(eid for eid, ok in eye_map.items() if not ok)
    meta = meta[meta.ophys_experiment_id.map(eye_map)].copy()
    meta = meta.sort_values('ophys_experiment_id').reset_index(drop=True)
    print(f'Curation: {len(files)} local experiments -> {len(meta)} active experiments with eye tracking')
    print(f'Excluded active experiments without eye tracking: {missing}')
    return meta


def image_vocabulary(meta, files):
    names = set()
    for eid in meta.ophys_experiment_id.astype(int):
        with h5py.File(files[eid], 'r') as h:
            if 'stimulus/templates' in h:
                for g in h['stimulus/templates'].values():
                    if 'control_description' in g:
                        names.update(decode_strings(g['control_description'][:]).tolist())
    return ['gray'] + sorted(n for n in names if n and n.lower() not in {'gray', 'omitted'})


def interp_valid(t_new, t, x, valid=None):
    t = np.asarray(t, float); x = np.asarray(x, float)
    good = np.isfinite(t) & np.isfinite(x)
    if valid is not None:
        good &= np.asarray(valid, bool)
    if good.sum() < 2:
        return np.full(len(t_new), np.nan, dtype=np.float32)
    # np.interp uses nearest valid endpoint outside support, as documented.
    return np.interp(t_new, t[good], x[good]).astype(np.float32)


def event_bin_means(events, ts, starts, centers):
    """Mean native frame samples in 100-ms bins; interpolate empty bins."""
    nbin = len(centers); nc = events.shape[1]
    out = np.empty((nc, nbin), dtype=np.float32)
    edges = starts + np.arange(nbin + 1) * BIN_S
    lo = np.searchsorted(ts, edges[:-1], side='left')
    hi = np.searchsorted(ts, edges[1:], side='left')
    for j, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            out[:, j] = np.asarray(events[a:b], dtype=np.float32).mean(axis=0)
        else:
            k = np.searchsorted(ts, centers[j])
            k = min(max(k, 1), len(ts)-1)
            t0, t1 = ts[k-1], ts[k]
            w = 0.0 if t1 == t0 else (centers[j]-t0)/(t1-t0)
            out[:, j] = (1-w)*events[k-1] + w*events[k]
    return out


def presentation_streams(h):
    """Return flash onsets and corresponding image names from NWB TimeSeries."""
    onsets, names = [], []
    if 'stimulus/presentation' not in h:
        return np.array([]), np.array([], dtype=str)
    templates = h.get('stimulus/templates')
    for gname, g in h['stimulus/presentation'].items():
        if 'timestamps' not in g or 'data' not in g:
            continue
        t = np.asarray(g['timestamps'], float)
        idx = np.asarray(g['data']).astype(int)
        desc = None
        if templates is not None and gname in templates and 'control_description' in templates[gname]:
            desc = decode_strings(templates[gname]['control_description'][:])
        elif templates is not None:
            # Match the sole natural-image template if group names differ slightly.
            candidates = [q for q in templates.values() if 'control_description' in q]
            if len(candidates) == 1:
                desc = decode_strings(candidates[0]['control_description'][:])
        if desc is None:
            continue
        good = (idx >= 0) & (idx < len(desc)) & np.isfinite(t)
        onsets.extend(t[good]); names.extend(desc[idx[good]])
    if not onsets:
        return np.array([]), np.array([], dtype=str)
    order = np.argsort(onsets)
    return np.asarray(onsets)[order], np.asarray(names)[order]


def quintile_labels(values):
    allv = np.concatenate([v[np.isfinite(v)] for v in values if np.isfinite(v).any()])
    if len(allv) == 0:
        raise ValueError('No finite samples for quintile discretization')
    edges = np.quantile(allv, [0.2, 0.4, 0.6, 0.8])
    labels = [np.searchsorted(edges, v, side='right').astype(np.int64) for v in values]
    return labels, edges


def process_experiment(row, path, image_to_idx, show_plot=False):
    eid = int(row.ophys_experiment_id); t0 = time.time()
    with h5py.File(path, 'r') as h:
        ev = np.asarray(h['processing/ophys/event_detection/data'], dtype=np.float32)
        ots = np.asarray(h['processing/ophys/event_detection/timestamps'], dtype=float)
        if ev.shape[0] != len(ots):
            raise ValueError(f'{eid}: events/timestamps mismatch')
        tr = h['intervals/trials']
        flags = {k: np.asarray(tr[k]).astype(bool) for k in
                 ['go','catch','aborted','auto_rewarded','hit','miss','false_alarm','correct_reject']}
        keep = (flags['go'] | flags['catch']) & ~flags['aborted'] & ~flags['auto_rewarded']
        tids = np.flatnonzero(keep)
        starts = np.asarray(tr['start_time'], float); stops = np.asarray(tr['stop_time'], float)
        changes = np.asarray(tr['change_time'], float)

        run_t = np.asarray(h['processing/running/speed/timestamps'], float)
        run_x = np.asarray(h['processing/running/speed/data'], float)
        eye_t = np.asarray(h['acquisition/EyeTracking/pupil_tracking/timestamps'], float)
        axes = np.asarray(h['acquisition/EyeTracking/pupil_tracking/data'], float)
        blink = np.asarray(h['acquisition/EyeTracking/likely_blink/data']).astype(bool)
        # EllipseSeries stores width and height (diameters); equivalent circular diameter.
        pupil = np.sqrt(np.maximum(axes[:, 0] * axes[:, 1], 0.0))
        pupil_good = (~blink) & np.isfinite(pupil) & (pupil > 0)
        flash_t, flash_name = presentation_streams(h)

        neural, inputs, prelim, kept_ids = [], [], [], []
        for ti in tids:
            nbin = int(np.floor((stops[ti] - starts[ti]) / BIN_S + 1e-9))
            if nbin < 1:
                continue
            centers = starts[ti] + (np.arange(nbin) + 0.5) * BIN_S
            r = interp_valid(centers, run_t, run_x)
            p = interp_valid(centers, eye_t, pupil, pupil_good)
            if not (np.isfinite(r).all() and np.isfinite(p).all()):
                continue
            n = event_bin_means(ev, ots, starts[ti], centers)
            image = np.zeros(nbin, dtype=np.int64)
            # A natural image is visible for 250 ms after each flash onset.
            left = np.searchsorted(flash_t, centers - 0.250, side='left')
            right = np.searchsorted(flash_t, centers, side='right')
            for j, (a, b) in enumerate(zip(left, right)):
                if b > a:
                    ft = flash_t[b-1]
                    if ft <= centers[j] < ft + 0.250:
                        image[j] = image_to_idx.get(str(flash_name[b-1]), 0)
            change = np.zeros(nbin, dtype=np.int64)
            if flags['go'][ti] and np.isfinite(changes[ti]):
                j = int(np.ceil((changes[ti] - starts[ti]) / BIN_S - 1e-12))
                if 0 <= j < nbin:
                    change[j] = 1
            if flags['hit'][ti]: outcome = 0
            elif flags['miss'][ti]: outcome = 1
            elif flags['false_alarm'][ti]: outcome = 2
            elif flags['correct_reject'][ti]: outcome = 3
            else: raise ValueError(f'{eid} trial {ti}: retained trial lacks outcome')
            neural.append(n); inputs.append(np.empty((0, nbin), dtype=np.float32))
            prelim.append((image, change, r, p, outcome, centers)); kept_ids.append(int(ti))

    run_lab, run_edges = quintile_labels([q[2] for q in prelim])
    pup_lab, pup_edges = quintile_labels([q[3] for q in prelim])
    outputs = []
    for q, rl, pl in zip(prelim, run_lab, pup_lab):
        image, change, _, _, outcome, centers = q
        outputs.append(np.vstack([image, change, rl, pl,
                                  np.full(len(centers), outcome, dtype=np.int64)]).astype(np.int64))
    if len(neural) < 2:
        raise ValueError(f'{eid}: fewer than 2 usable trials')
    for n, i, o in zip(neural, inputs, outputs):
        assert n.ndim == 2 and i.shape == (0, n.shape[1]) and o.shape == (5, n.shape[1])
        assert np.isfinite(n).all() and np.isfinite(o).all()
    info = {'ophys_experiment_id': eid, 'ophys_session_id': int(row.ophys_session_id),
            'behavior_session_id': int(row.behavior_session_id), 'mouse_id': str(row.mouse_id),
            'session_type': row.session_type, 'targeted_structure': row.targeted_structure,
            'imaging_depth': int(row.imaging_depth), 'n_neurons': int(neural[0].shape[0]),
            'n_trials': len(neural), 'raw_retained_trials': int(keep.sum()),
            'native_rate_hz': float(1/np.median(np.diff(ots))),
            'running_quintile_edges': run_edges.tolist(), 'pupil_quintile_edges': pup_edges.tolist(),
            'source_trial_indices': kept_ids, 'seconds': time.time()-t0}
    if show_plot:
        plot_processing(eid, prelim, neural, outputs, run_edges, pup_edges)
    print(f"{eid}: {len(neural)} trials, {neural[0].shape[0]} neurons, {info['native_rate_hz']:.2f} Hz, {info['seconds']:.2f}s")
    return neural, inputs, outputs, info


def plot_processing(eid, prelim, neural, outputs, run_edges, pup_edges):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    q = prelim[0]; out = outputs[0]; n = neural[0]
    t = np.arange(n.shape[1])*BIN_S
    fig, ax = plt.subplots(5, 1, figsize=(13, 12), sharex=True)
    ax[0].imshow(n[:min(50, len(n))], aspect='auto', origin='lower', extent=[0,t[-1]+BIN_S,0,min(50,len(n))])
    ax[0].set_ylabel('neurons'); ax[0].set_title(f'Experiment {eid}: binned detected events')
    ax[1].step(t, out[0], where='mid', label='image identity'); ax[1].step(t, out[1]*out[0].max(), where='mid', label='change (scaled)'); ax[1].legend()
    ax[2].plot(t, q[2], label='running cm/s'); ax[2].step(t, out[2], where='mid', label='quintile'); ax[2].legend(); ax[2].set_title(f'edges={np.round(run_edges,2)}')
    ax[3].plot(t, q[3], label='pupil equivalent diameter'); ax[3].step(t, out[3], where='mid', label='quintile'); ax[3].legend(); ax[3].set_title(f'edges={np.round(pup_edges,4)}')
    ax[4].step(t, out[4], where='mid'); ax[4].set_ylabel('outcome'); ax[4].set_xlabel('seconds from trial start')
    fig.tight_layout(); fig.savefig(f'/app/processing_{eid}.png', dpi=140); plt.close(fig)


def validate(data):
    ns = len(data['neural'])
    assert ns == len(data['input']) == len(data['output']) == len(data['subject_idx']) == len(data['brain_region_idx'])
    assert len(data['input_names']) == 0 and len(data['output_names']) == 5
    total_trials = total_neurons = 0
    for s in range(ns):
        assert len(data['neural'][s]) >= 2
        assert len(data['neural'][s]) == len(data['input'][s]) == len(data['output'][s])
        nr = data['neural'][s][0].shape[0]; total_neurons += nr; total_trials += len(data['neural'][s])
        assert data['brain_region_idx'][s].shape == (nr,)
        for n, x, y in zip(data['neural'][s], data['input'][s], data['output'][s]):
            assert n.shape[0] == nr and x.shape == (0,n.shape[1]) and y.shape == (5,n.shape[1])
            assert np.isfinite(n).all() and np.isfinite(y).all()
            assert set(np.unique(y[1])).issubset({0,1})
            assert y[2].min() >= 0 and y[2].max() <= 4 and y[3].min() >= 0 and y[3].max() <= 4
            assert y[4].min() >= 0 and y[4].max() <= 3 and np.unique(y[4]).size == 1
    print(f'Internal validation passed: {ns} sessions, {total_trials} trials, {total_neurons} neurons')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('outpicklefile')
    g=ap.add_mutually_exclusive_group(); g.add_argument('--full',action='store_true'); g.add_argument('--sample',action='store_true')
    ap.add_argument('--show-processing',action='store_true'); args=ap.parse_args()
    started=time.time(); files=find_files(); meta=select_experiments(files); vocab=image_vocabulary(meta,files)
    if args.sample: meta=meta.iloc[:2].copy()
    print('Image vocabulary:',vocab); imap={n:i for i,n in enumerate(vocab)}
    neural=[]; inputs=[]; outputs=[]; infos=[]
    for k,row in meta.iterrows():
        eid=int(row.ophys_experiment_id)
        n,x,y,info=process_experiment(row,files[eid],imap,args.show_processing and len(infos)<2)
        neural.append(n); inputs.append(x); outputs.append(y); infos.append(info)
    subjects=sorted({z['mouse_id'] for z in infos}); smap={x:i for i,x in enumerate(subjects)}
    regions=sorted({z['targeted_structure'] for z in infos}); rmap={x:i for i,x in enumerate(regions)}
    data={'neural':neural,'input':inputs,'output':outputs,'subjects':subjects,
          'subject_idx':np.asarray([smap[z['mouse_id']] for z in infos],dtype=np.int64),
          'brain_regions':regions,
          'brain_region_idx':[np.full(z['n_neurons'],rmap[z['targeted_structure']],dtype=np.int64) for z in infos],
          'input_names':[],
          'output_names':['image_identity','image_change','running_speed_bin','pupil_diameter_bin','trial_outcome'],
          'output_values':[vocab,['no_change','change'],['0-20%','20-40%','40-60%','60-80%','80-100%'],
                           ['0-20%','20-40%','40-60%','60-80%','80-100%'],
                           ['hit','miss','false_alarm','correct_reject']],
          'metadata':{'task_description':'Decode visual image identity/change, running-speed quintile, pupil-diameter quintile, and go/catch trial outcome from detected calcium events.',
                      'time_bin_size':100.0,'temporal_alignment_event':'trial start time on the ophys session clock',
                      'off_start':0.0,'off_end':None,'neural_signal':'detected calcium event magnitude, mean in 100 ms bins',
                      'trial_filter':'active sessions; go or catch; exclude aborted and auto-rewarded; require eye tracking',
                      'session_unit':'ophys experiment / imaging plane','session_info':infos,
                      'image_visibility_seconds':0.250,'source_release':'Visual Behavior Ophys 1.1.0 local subset'}}
    validate(data)
    with open(args.outpicklefile,'wb') as f: pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
    print(f'Saved {args.outpicklefile} ({os.path.getsize(args.outpicklefile)/2**30:.3f} GiB) in {time.time()-started:.1f}s')

if __name__=='__main__': main()
