#!/usr/bin/env python3
"""Convert Sosa et al. CA1 NWB files to trial-aligned decoder format.

Usage: python -u /app/convert_data.py OUT.pkl [--full|--sample] [--show-processing]
"""
import argparse
import pickle
import time
from pathlib import Path

import h5py
import numpy as np

DATA_ROOT = Path('/app/data')
ZONE_STARTS = np.array([80.0, 200.0, 320.0], dtype=np.float32)
ZONE_WIDTH = 50.0
BIN_MS = 1000.0 / 15.5078125


def discretize_distance(position, zone_start):
    """Signed distance to closed reward-zone interval, then requested 7 bins."""
    p = np.asarray(position)
    distance = np.where(p < zone_start, p-zone_start,
                        np.where(p > zone_start+ZONE_WIDTH,
                                 p-(zone_start+ZONE_WIDTH), 0.0))
    out = np.empty(p.shape, dtype=np.int8)
    out[distance < -50] = 0
    out[(distance >= -50) & (distance < -10)] = 1
    out[(distance >= -10) & (distance < 0)] = 2
    out[distance == 0] = 3
    out[(distance > 0) & (distance <= 10)] = 4
    out[(distance > 10) & (distance <= 50)] = 5
    out[distance > 50] = 6
    return out, distance.astype(np.float32)


def discretize_position(position):
    # <90, [90,180), [180,270), [270,360], >360
    p = np.asarray(position)
    return np.select([p < 90, p < 180, p < 270, p <= 360],
                     [0, 1, 2, 3], default=4).astype(np.int8)


def discretize_speed(speed):
    # <2, [2,10), [10,20), [20,40], >40
    v = np.asarray(speed)
    return np.select([v < 2, v < 10, v < 20, v <= 40],
                     [0, 1, 2, 3], default=4).astype(np.int8)


def nearest_fill(labels):
    """Fill missing trial labels from nearest trial with observed zone entry."""
    labels = np.asarray(labels, dtype=np.float32).copy()
    known = np.flatnonzero(np.isfinite(labels))
    missing = np.flatnonzero(~np.isfinite(labels))
    if not len(known):
        raise ValueError('session has no observed reward-zone entries')
    if len(missing):
        nearest = np.argmin(np.abs(missing[:, None] - known[None, :]), axis=1)
        labels[missing] = labels[known[nearest]]
    return labels.astype(np.int8)


def inspect_session(path):
    """Read small behavior arrays and derive trial metadata."""
    with h5py.File(path, 'r') as f:
        b = f['processing/behavior/BehavioralTimeSeries']
        trial = b['trial number/data'][:]
        ts = b['trial number/timestamps'][:]
        position = b['position/data'][:]
        zone_event = b['reward_zone/data'][:]
        environment = b['environment/data'][:]
        reward_ts = b['Reward/timestamps'][:]
        trial_ids = np.unique(trial[trial >= 0]).astype(int)
        if not np.array_equal(trial_ids, np.arange(len(trial_ids))):
            raise ValueError(f'{path.name}: non-contiguous trial numbers')
        zones = np.full(len(trial_ids), np.nan, dtype=np.float32)
        outcomes = np.zeros(len(trial_ids), dtype=np.int8)
        envs = np.zeros(len(trial_ids), dtype=np.int8)
        has_start = np.zeros(len(trial_ids), dtype=bool)
        bounds = []
        for tid in trial_ids:
            idx = np.flatnonzero(trial == tid)
            if not len(idx) or np.any(np.diff(idx) != 1):
                raise ValueError(f'{path.name}: trial {tid} is empty/noncontiguous')
            lo, hi = int(idx[0]), int(idx[-1])
            bounds.append((lo, hi + 1))
            has_start[tid] = bool(np.any(b['trial_start/data'][idx] > 0))
            event_idx = idx[zone_event[idx] > 0]
            if len(event_idx):
                entry_pos = position[event_idx[0]]
                zones[tid] = int(np.argmin(np.abs(ZONE_STARTS-entry_pos)))
            outcomes[tid] = int(np.any((reward_ts >= ts[lo]) & (reward_ts <= ts[hi])))
            ev = environment[idx]
            ev = ev[(ev == 0) | (ev == 1)]
            if not len(ev):
                raise ValueError(f'{path.name}: no valid environment in trial {tid}')
            envs[tid] = int(np.bincount(ev.astype(int), minlength=2).argmax())
        zones = nearest_fill(zones)
        # Exclude incomplete recording-edge fragments lacking a trial-start marker.
        # Preserve native trial number and previous native-trial outcome.
        previous = np.r_[0, outcomes[:-1]].astype(np.int8)
        keep = has_start
        if np.sum(~keep):
            print(f'  {path.name}: excluding {int(np.sum(~keep))} numbered fragment(s) without trial_start', flush=True)
        trial_ids = trial_ids[keep]
        bounds = [x for x, k in zip(bounds, keep) if k]
        zones = zones[keep]; outcomes = outcomes[keep]; envs = envs[keep]; previous = previous[keep]
        changes = int(np.sum(np.diff(zones) != 0))
        if changes > 1:
            raise ValueError(f'{path.name}: inferred {changes} reward-zone switches')
        ps = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
        iscell = ps['iscell'][:]
        accepted = iscell[:, 0] > 0 if iscell.ndim == 2 else iscell > 0
        plane_idx = ps['planeIdx'][:].astype(int) if 'planeIdx' in ps else np.zeros(len(accepted), dtype=int)
        plane_names = sorted(f['processing/ophys/Deconvolved'].keys(), key=lambda x: int(x.replace('plane','')))
        plane_masks = []
        plane_shapes = []
        for plane_name in plane_names:
            plane_num = int(plane_name.replace('plane',''))
            shape = f['processing/ophys/Deconvolved/'+plane_name+'/data'].shape
            mask = accepted[plane_idx == plane_num]
            if shape[1] != len(mask):
                raise ValueError(f'{path.name}: {plane_name} ROI mask/data mismatch {len(mask)} != {shape[1]}')
            plane_masks.append(mask); plane_shapes.append(shape)
        subject = f['general/subject/subject_id'][()].decode()
    return dict(trial=trial, timestamps=ts, trial_ids=trial_ids, bounds=bounds,
                zones=zones, outcomes=outcomes, previous_outcomes=previous, environments=envs,
                plane_names=plane_names, plane_masks=plane_masks, plane_shapes=plane_shapes,
                n_rois=len(accepted), n_cells=int(accepted.sum()), subject=subject)


def convert_session(path, make_plot=False):
    t0 = time.perf_counter()
    info = inspect_session(path)
    trials_neural, trials_input, trials_output = [], [], []
    plot_data = None
    with h5py.File(path, 'r') as f:
        b = f['processing/behavior/BehavioralTimeSeries']
        neural_series = [f['processing/ophys/Deconvolved/'+name+'/data'] for name in info['plane_names']]
        cell_indices = [np.flatnonzero(mask) for mask in info['plane_masks']]
        position = b['position/data'][:]
        speed = b['speed/data'][:]
        lick = b['lick/data'][:]
        timestamps = info['timestamps']
        n_common = min([len(position)] + [ds.shape[0] for ds in neural_series])
        for trial_idx, (lo, hi) in enumerate(info['bounds']):
            tid = int(info['trial_ids'][trial_idx])
            hi = min(hi, n_common)
            if hi <= lo:
                raise ValueError(f'{path.name}: no common samples in trial {tid}')
            # One contiguous HDF5 read per plane, then accepted-cell selection and concatenation.
            plane_data = [np.asarray(ds[lo:hi, :][:, idx].T, dtype=np.float32)
                          for ds, idx in zip(neural_series, cell_indices)]
            neural = np.concatenate(plane_data, axis=0)
            if not np.all(np.isfinite(neural)):
                neural = np.nan_to_num(neural, copy=False)
            t_rel = (timestamps[lo:hi] - timestamps[lo]).astype(np.float32)
            n = hi-lo
            previous = info['previous_outcomes'][trial_idx]
            inp = np.vstack((t_rel,
                             np.full(n, info['environments'][trial_idx], np.float32),
                             np.full(n, tid, np.float32),
                             np.full(n, previous, np.float32))).astype(np.float32)
            zone_start = float(ZONE_STARTS[info['zones'][trial_idx]])
            dist_cls, signed_dist = discretize_distance(position[lo:hi], zone_start)
            out = np.vstack((dist_cls,
                             discretize_position(position[lo:hi]),
                             discretize_speed(speed[lo:hi]),
                             (lick[lo:hi] > 0).astype(np.int8),
                             np.full(n, info['zones'][trial_idx], np.int8),
                             np.full(n, info['outcomes'][trial_idx], np.int8))).astype(np.int8)
            if neural.shape[1] != inp.shape[1] or inp.shape[1] != out.shape[1]:
                raise ValueError(f'{path.name}: trial {tid} temporal shape mismatch')
            trials_neural.append(neural); trials_input.append(inp); trials_output.append(out)
            if make_plot and plot_data is None and trial_idx == min(4, len(info['bounds'])-1):
                plot_data=(tid, neural, t_rel, position[lo:hi], speed[lo:hi],
                           lick[lo:hi], signed_dist, out)
    if make_plot and plot_data is not None:
        plot_processing(path.stem, plot_data)
    elapsed=time.perf_counter()-t0
    print(f'  {path.name}: {len(trials_neural)} trials, {info["n_rois"]} ROIs -> '
          f'{info["n_cells"]} cells, {sum(x.shape[1] for x in trials_neural)} samples, {elapsed:.2f}s', flush=True)
    summary=dict(file=path.name, subject=info['subject'], n_trials=len(trials_neural),
                 n_cells=info['n_cells'], zone_counts=np.bincount(info['zones'], minlength=3).tolist(),
                 rewarded=int(info['outcomes'].sum()), environments=np.unique(info['environments']).tolist(),
                 seconds=elapsed)
    return trials_neural, trials_input, trials_output, info['subject'], summary


def plot_processing(session_id, d):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tid, neural, t, pos, speed, lick, distance, out = d
    fig, ax = plt.subplots(6, 1, figsize=(12, 13), sharex=True, constrained_layout=True)
    show=neural[:min(80,len(neural))]
    ax[0].imshow(show, aspect='auto', origin='lower', extent=[t[0],t[-1],0,len(show)], cmap='viridis')
    ax[0].set_ylabel('accepted cells'); ax[0].set_title(f'{session_id}, trial {tid}: deconvolved activity')
    ax[1].plot(t,pos,label='position (cm)'); ax[1].plot(t,distance,label='signed distance'); ax[1].legend(); ax[1].set_ylabel('cm')
    ax[2].plot(t,speed); ax[2].set_ylabel('speed cm/s')
    ax[3].plot(t,lick,label='raw within-frame count'); ax[3].step(t,out[3],where='mid',label='binary'); ax[3].legend(); ax[3].set_ylabel('lick')
    ax[4].step(t,out[0],where='mid',label='distance class'); ax[4].step(t,out[1],where='mid',label='position class'); ax[4].step(t,out[2],where='mid',label='speed class'); ax[4].legend(); ax[4].set_ylabel('class')
    ax[5].step(t,out[4],where='mid',label='zone A/B/C'); ax[5].step(t,out[5],where='mid',label='rewarded'); ax[5].legend(); ax[5].set_ylabel('trial labels'); ax[5].set_xlabel('seconds from trial start')
    fig.savefig(f'/app/processing_{session_id}.png', dpi=140); plt.close(fig)


def validate_data(data):
    ns=len(data['neural'])
    assert ns==len(data['input'])==len(data['output'])==len(data['subject_idx'])==len(data['brain_region_idx'])
    for s in range(ns):
        assert len(data['neural'][s])==len(data['input'][s])==len(data['output'][s])>=2
        ncell=data['neural'][s][0].shape[0]
        assert data['brain_region_idx'][s].shape==(ncell,)
        for n,i,o in zip(data['neural'][s],data['input'][s],data['output'][s]):
            assert n.ndim==i.ndim==o.ndim==2 and i.shape[0]==4 and o.shape[0]==6
            assert n.shape[0]==ncell and n.shape[1]==i.shape[1]==o.shape[1]
            assert np.isfinite(n).all() and np.isfinite(i).all()
    return True


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument('--full',action='store_true',help='process all sessions (default)')
    mode.add_argument('--sample',action='store_true',help='process first 2 sessions')
    ap.add_argument('--show-processing',action='store_true')
    args=ap.parse_args()
    files=sorted(DATA_ROOT.glob('sub-*/*.nwb'))
    if args.sample: files=files[:2]
    print(f'Converting {len(files)} of {len(list(DATA_ROOT.glob("sub-*/*.nwb")))} sessions',flush=True)
    neural=[]; inputs=[]; outputs=[]; session_subjects=[]; summaries=[]
    t0=time.perf_counter()
    for si,path in enumerate(files):
        print(f'[{si+1}/{len(files)}] {path}',flush=True)
        n,i,o,subject,summary=convert_session(path, args.show_processing and si<2)
        neural.append(n); inputs.append(i); outputs.append(o); session_subjects.append(subject); summaries.append(summary)
    subjects=sorted(set(session_subjects), key=lambda x:(int(x[1:]) if x[1:].isdigit() else x))
    subject_idx=np.array([subjects.index(x) for x in session_subjects],dtype=np.int64)
    data=dict(
      neural=neural, input=inputs, output=outputs,
      subjects=subjects, subject_idx=subject_idx,
      brain_regions=['CA1'],
      brain_region_idx=[np.zeros(sess[0].shape[0],dtype=np.int64) for sess in neural],
      input_names=['time from trial start (s)','environment type','trial number','previous trial outcome'],
      output_names=['distance to reward zone','absolute position','speed','lick','reward zone location','reward outcome'],
      output_values=[
       ['< -50 cm','-50 to < -10 cm','-10 to < 0 cm','inside reward zone','> 0 to 10 cm','> 10 to 50 cm','> 50 cm'],
       ['< 90 cm','90 to < 180 cm','180 to < 270 cm','270 to 360 cm','> 360 cm'],
       ['< 2 cm/s','2 to < 10 cm/s','10 to < 20 cm/s','20 to 40 cm/s','> 40 cm/s'],
       ['no lick','lick'],['A','B','C'],['omitted','rewarded']],
      metadata=dict(
       task_description='Mouse runs along a 450 cm virtual corridor with reward at one of three zones; decode position, speed, licking, reward location, and outcome from CA1 deconvolved calcium activity.',
       time_bin_size=float(BIN_MS), temporal_alignment_event='start of numbered corridor trial (entry to linear track)',
       off_start=0.0, off_end=None, neural_signal='Suite2p deconvolved calcium events from accepted cells',
       brain_region='hippocampus CA1', track_length_cm=450.0,
       reward_zone_starts_cm=ZONE_STARTS.tolist(), reward_zone_width_cm=ZONE_WIDTH,
       source_format='NWB 2.5', session_info=summaries,
       notes='Variable-duration trials retain all numbered in-trial frames; intertrial/teleport samples are excluded.')
    )
    validate_data(data)
    out=Path(args.outpicklefile); out.parent.mkdir(parents=True,exist_ok=True)
    print(f'Writing {out} ...',flush=True)
    with out.open('wb') as fh: pickle.dump(data,fh,protocol=pickle.HIGHEST_PROTOCOL)
    dt=time.perf_counter()-t0
    print(f'Done: {len(neural)} sessions, {sum(map(len,neural))} trials, {sum(x[0].shape[0] for x in neural)} session-cells; {out.stat().st_size/1e9:.3f} GB; {dt:.2f}s',flush=True)

if __name__=='__main__': main()
