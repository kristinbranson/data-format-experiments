#!/usr/bin/env python3
"""Convert Sosa et al. NWB behavior+ophys sessions to decoder format.

Usage: python -u /app/convert_data.py OUT.pkl [--full|--sample] [--show-processing]
"""
import argparse, pickle, re, time
from pathlib import Path
import h5py
import numpy as np

DATA_ROOT = Path('/app/data')
DT = 1.0 / 15.5078125
ZONE_INTERVALS = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
ZONE_CODES = {'A': 0, 'B': 1, 'C': 2}


def scalar_text(ds):
    x = ds[()]
    return x.decode(errors='replace') if isinstance(x, bytes) else str(x)


def scene_from_identifier(identifier):
    return identifier.rstrip('/').split('/')[-1]


def scene_zone_sequence(scene, n_trials, change_trial=30):
    """Match reward_relative.behavior.get_reward_zones for X/Y/Z=A/B/C."""
    fixed = re.search(r'Location([ABC])$', scene)
    if fixed:
        labels = [fixed.group(1)] * n_trials
    else:
        # Handles Env1_LocationA_to_B and Env1_A_to_Env2_B forms.
        m = re.search(r'(?:Location)?([ABC])_to_(?:Env[123]_)?(?:Location)?([ABC])$', scene)
        if not m:
            raise ValueError(f'Cannot parse reward-zone sequence from scene {scene!r}')
        before, after = m.groups()
        c = min(int(change_trial), n_trials)
        labels = [before] * c + [after] * (n_trials - c)
    coords = np.asarray([ZONE_INTERVALS[x] for x in labels], dtype=np.float32)
    codes = np.asarray([ZONE_CODES[x] for x in labels], dtype=np.int64)
    return labels, coords, codes


def distance_to_interval(pos, start, stop):
    return np.where(pos < start, pos - start, np.where(pos > stop, pos - stop, 0.0))


def bin_distance(x):
    y = np.empty(x.shape, np.int64)
    y[x < -50] = 0
    y[(x >= -50) & (x < -10)] = 1
    y[(x >= -10) & (x < 0)] = 2
    y[x == 0] = 3
    y[(x > 0) & (x <= 10)] = 4
    y[(x > 10) & (x <= 50)] = 5
    y[x > 50] = 6
    return y


def bin_position(x):
    y = np.empty(x.shape, np.int64)
    y[x < 90] = 0
    y[(x >= 90) & (x < 180)] = 1
    y[(x >= 180) & (x < 270)] = 2
    y[(x >= 270) & (x <= 360)] = 3
    y[x > 360] = 4
    return y


def bin_speed(x):
    y = np.empty(x.shape, np.int64)
    y[x < 2] = 0
    y[(x >= 2) & (x < 10)] = 1
    y[(x >= 10) & (x < 20)] = 2
    y[(x >= 20) & (x <= 40)] = 3
    y[x > 40] = 4
    return y


def pair_trials(start_signal, teleport_signal, n_samples):
    starts = np.flatnonzero(start_signal > 0)
    teleports = np.flatnonzero(teleport_signal > 0)
    pairs = []
    for s in starts:
        k = np.searchsorted(teleports, s, side='left')
        if k < len(teleports) and teleports[k] > s:
            pairs.append((int(s), int(teleports[k])))  # end is exclusive
    return pairs


def load_session(path):
    with h5py.File(path, 'r') as f:
        b = f['processing/behavior/BehavioralTimeSeries']
        aligned_names = ['autoreward','environment','lick','position','reward_zone',
                         'scanning','speed','teleport','trial number','trial_start']
        beh = {k: np.asarray(b[k]['data']) for k in aligned_names}
        timestamps = np.asarray(b['position']['timestamps'], dtype=np.float64)
        reward_times = np.asarray(b['Reward']['timestamps'], dtype=np.float64)
        n_beh = len(timestamps)
        if any(len(v) != n_beh for v in beh.values()):
            raise ValueError(f'Frame-aligned behavior length mismatch in {path.name}')

        seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
        iscell0 = np.asarray(seg['iscell'])
        iscell = (iscell0[:, 0] if iscell0.ndim > 1 else iscell0) > 0
        plane_idx = np.asarray(seg['planeIdx']).astype(int)
        pieces = []
        selected_global = []
        for plane_name in sorted(f['processing/ophys/Deconvolved'].keys()):
            plane = int(plane_name.replace('plane',''))
            global_ids = np.flatnonzero(plane_idx == plane)
            ds = f['processing/ophys/Deconvolved'][plane_name]['data']
            if ds.shape[1] != len(global_ids):
                raise ValueError(f'ROI mapping mismatch {path.name} {plane_name}')
            local_keep = np.flatnonzero(iscell[global_ids])
            # h5py reads efficiently with monotonic column indices.
            arr = np.asarray(ds[:, local_keep], dtype=np.float32)
            pieces.append(arr[:n_beh])
            selected_global.extend(global_ids[local_keep].tolist())
        if not pieces:
            raise ValueError(f'No deconvolved series in {path}')
        neural_tn = np.concatenate(pieces, axis=1)
        if neural_tn.shape[0] < n_beh:
            raise ValueError(f'Neural shorter than behavior in {path.name}')
        # In ten source files neural has one extra terminal row; slicing above trims it.
        if neural_tn.shape[1] != int(iscell.sum()):
            raise ValueError(f'Accepted-cell count mismatch in {path.name}')

        identifier = scalar_text(f['identifier'])
        subject = scalar_text(f['general/subject/subject_id'])

    pairs = pair_trials(beh['trial_start'], beh['teleport'], n_beh)
    labels, zone_coords, zone_codes = scene_zone_sequence(scene_from_identifier(identifier), len(pairs))
    outcomes = np.asarray([
        int(np.any((reward_times >= timestamps[s]) & (reward_times <= timestamps[e])))
        for s, e in pairs
    ], dtype=np.int64)

    neural_trials, input_trials, output_trials = [], [], []
    plot_info = []
    for j, (s, e) in enumerate(pairs):
        if e <= s:
            continue
        sl = slice(s, e)  # start included; teleport excluded
        pos = np.asarray(beh['position'][sl], dtype=np.float32)
        speed = np.asarray(beh['speed'][sl], dtype=np.float32)
        lick = (np.asarray(beh['lick'][sl]) > 0).astype(np.int64)
        env_values = np.asarray(beh['environment'][sl])
        env = int(np.rint(np.median(env_values)))
        source_trial = float(beh['trial number'][s])
        reltime = (timestamps[sl] - timestamps[s]).astype(np.float32)
        prev = int(outcomes[j-1]) if j > 0 else 0
        inp = np.vstack([
            reltime,
            np.full(len(pos), env, np.float32),
            np.full(len(pos), source_trial, np.float32),
            np.full(len(pos), prev, np.float32),
        ]).astype(np.float32, copy=False)
        zstart, zstop = zone_coords[j]
        signed_dist = distance_to_interval(pos, zstart, zstop)
        out = np.vstack([
            bin_distance(signed_dist), bin_position(pos), bin_speed(speed), lick,
            np.full(len(pos), zone_codes[j], np.int64),
            np.full(len(pos), outcomes[j], np.int64),
        ]).astype(np.int64, copy=False)
        neu = np.asarray(neural_tn[sl].T, dtype=np.float32)
        if not (neu.shape[1] == inp.shape[1] == out.shape[1] > 0):
            raise ValueError(f'Trial shape mismatch {path.name} trial {j}')
        if abs(float(inp[0,0])) > 1e-7 or np.any(~np.isfinite(neu)) or np.any(~np.isfinite(inp)):
            raise ValueError(f'Nonfinite data or bad alignment {path.name} trial {j}')
        neural_trials.append(neu); input_trials.append(inp); output_trials.append(out)
        plot_info.append((pos, speed, lick, signed_dist, out, labels[j]))

    info = dict(path=str(path), session=path.stem, identifier=identifier,
                subject=subject, n_trials=len(neural_trials), n_neurons=neural_tn.shape[1],
                n_rois=len(iscell), reward_events=len(reward_times), outcomes=outcomes,
                zone_labels=labels, pairs=pairs, timestamps=timestamps,
                selected_global=np.asarray(selected_global), plot_info=plot_info)
    return neural_trials, input_trials, output_trials, info


def make_plot(info):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    trials = info['plot_info'][:6]
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=False)
    offset = 0
    for j, (pos, speed, lick, dist, out, label) in enumerate(trials):
        x = np.arange(len(pos)) + offset
        axes[0].plot(x, pos, lw=.8, label=f'T{j} zone {label}')
        axes[1].plot(x, dist, lw=.8)
        axes[2].plot(x, speed, lw=.8)
        axes[3].step(x, out[0], where='mid', lw=.8)
        axes[3].scatter(x[lick > 0], np.full(np.sum(lick > 0), 7.2), s=5, c='r')
        for ax in axes: ax.axvline(offset, color='k', alpha=.15)
        offset += len(pos) + 5
    axes[0].set_ylabel('position cm'); axes[0].legend(ncol=3, fontsize=7)
    axes[1].set_ylabel('signed zone dist cm'); axes[1].axhline(0,c='k',lw=.5)
    axes[2].set_ylabel('speed cm/s')
    axes[3].set_ylabel('distance bin\n(red=lick)'); axes[3].set_xlabel('concatenated native samples')
    fig.suptitle(f"{info['session']} | {info['identifier']} | trial starts aligned at t=0")
    fig.tight_layout(); fig.savefig(f"/app/processing_{info['session']}.png", dpi=140); plt.close(fig)


def convert(files, show_processing=False):
    subjects = sorted({p.parent.name.replace('sub-','') for p in files}, key=lambda x: int(re.sub(r'\D','',x)))
    subject_lookup = {s:i for i,s in enumerate(subjects)}
    neural, inputs, outputs, region_idx, session_info = [], [], [], [], []
    t0=time.perf_counter()
    for i,p in enumerate(files):
        q0=time.perf_counter(); n,x,y,info=load_session(p)
        neural.append(n); inputs.append(x); outputs.append(y)
        region_idx.append(np.zeros(info['n_neurons'], dtype=np.int64))
        info_clean={k:v for k,v in info.items() if k not in ('plot_info','timestamps','pairs','selected_global','outcomes','zone_labels')}
        info_clean['subject_idx']=subject_lookup[info['subject']]
        session_info.append(info_clean)
        if show_processing and i<2: make_plot(info)
        print(f"[{i+1}/{len(files)}] {p.name}: {info['n_trials']} trials, {info['n_neurons']} cells, {time.perf_counter()-q0:.2f}s",flush=True)
    total_trials=sum(map(len,neural)); total_cells=sum(len(x) for x in region_idx)
    data={
      'neural':neural, 'input':inputs, 'output':outputs,
      'subjects':subjects,
      'subject_idx':np.asarray([z['subject_idx'] for z in session_info],dtype=np.int64),
      'brain_regions':['CA1'], 'brain_region_idx':region_idx,
      'input_names':['time from trial start (s)','environment type','trial number','previous trial outcome'],
      'output_names':['distance to reward zone','absolute position','speed','lick','reward zone location','reward outcome'],
      'output_values':[
        ['< -50 cm','-50 to -10 cm','-10 to < 0 cm','inside reward zone','>0 to +10 cm','+10 to +50 cm','> +50 cm'],
        ['< 90 cm','90 to <180 cm','180 to <270 cm','270 to 360 cm','> 360 cm'],
        ['< 2 cm/s','2 to <10 cm/s','10 to <20 cm/s','20 to 40 cm/s','> 40 cm/s'],
        ['no lick','lick'], ['A','B','C'], ['omitted','rewarded']],
      'metadata':{
        'task_description':'Decode position, reward-relative distance, speed, licking, reward-zone identity, and outcome from dorsal CA1 deconvolved calcium activity during virtual linear-track trials.',
        'time_bin_size':DT*1000.0,
        'temporal_alignment_event':'trial_start pulse: entry onto the 450 cm linear track',
        'off_start':0.0, 'off_end':None,
        'neural_signal':'Suite2p/OASIS deconvolved calcium activity for iscell ROIs; not interpreted as spike rate',
        'trial_window':'trial_start sample inclusive to teleport sample exclusive; variable-duration trials',
        'reward_zone_intervals_cm':{'A':[80,130],'B':[200,250],'C':[320,370]},
        'source':'DANDI 001361; Sosa et al., A flexible hippocampal population code for experience relative to reward',
        'session_info':session_info,
      }}
    print(f'Converted {len(files)} sessions, {total_trials} trials, {total_cells} session-cells in {time.perf_counter()-t0:.2f}s',flush=True)
    return data


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('outpicklefile'); g=ap.add_mutually_exclusive_group(); g.add_argument('--full',action='store_true'); g.add_argument('--sample',action='store_true'); ap.add_argument('--show-processing',action='store_true'); args=ap.parse_args()
    files=sorted(DATA_ROOT.rglob('*.nwb'))
    if args.sample: files=files[:2]
    if not files: raise RuntimeError('No NWB files found')
    data=convert(files,args.show_processing)
    out=Path(args.outpicklefile); t=time.perf_counter()
    with out.open('wb') as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved {out} ({out.stat().st_size/1e6:.1f} MB) in {time.perf_counter()-t:.2f}s',flush=True)

if __name__=='__main__': main()
