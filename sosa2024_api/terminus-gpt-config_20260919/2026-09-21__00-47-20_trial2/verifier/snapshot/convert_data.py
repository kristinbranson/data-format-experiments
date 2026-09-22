#!/usr/bin/env python3
"""Convert Sosa et al. reward-relative NWB files to decoder format.

All NWB access is through pynwb. Usage:
  python -u /app/convert_data.py OUT.pkl [--full|--sample] [--show-processing]
"""
import argparse
import pickle
import re
import time
from pathlib import Path

import numpy as np
from pynwb import NWBHDF5IO

DATA_ROOT = Path('/app/data')
DT = 0.06448362720402656
ZONE_BOUNDS = {'A': (80.0, 100.0), 'B': (200.0, 220.0), 'C': (320.0, 340.0)}
ZONE_INDEX = {'A': 0, 'B': 1, 'C': 2}


def parse_scene(scene, trial_id, env_mode):
    """Return (environment, zone letter) for all released scene spellings."""
    env = int(round(float(env_mode)))
    # Combined environment+reward switches use Env1_A_to_Env2_B and switch
    # at trial 30 in the release. The aligned environment stream is authoritative.
    m = re.fullmatch(r'Env([12])_([ABC])_to_Env([12])_([ABC])', scene)
    if m:
        first_env, first_zone = int(m.group(1))-1, m.group(2)
        second_env, second_zone = int(m.group(3))-1, m.group(4)
        if env == first_env:
            return env, first_zone
        if env == second_env:
            return env, second_zone
        raise ValueError(f'Environment mismatch scene={scene}, trial={trial_id}: data={env}')
    # Stable or reward-only switch scenes; reward switch follows trial 39.
    m = re.search(r'Env([12])_Location([ABC])(?:_to_([ABC]))?', scene)
    if not m:
        raise ValueError(f'Cannot parse reward location from scene {scene!r}')
    expected_env, first, second = int(m.group(1))-1, m.group(2), m.group(3)
    if env != expected_env:
        raise ValueError(f'Environment mismatch scene={scene}, trial={trial_id}: data={env}, expected={expected_env}')
    zone = second if second is not None and trial_id >= 40 else first
    return env, zone


def signed_distance_to_interval(position, low, high):
    """Negative before, zero within, positive after a closed interval."""
    return np.where(position < low, position-low,
                    np.where(position > high, position-high, 0.0))


def discretize_distance(d):
    """Task-prescribed seven classes, including exact-boundary conventions."""
    y = np.empty(d.shape, dtype=np.int8)
    y[d < -50] = 0
    y[(d >= -50) & (d <= -10)] = 1
    y[(d > -10) & (d < 0)] = 2
    y[d == 0] = 3
    y[(d > 0) & (d <= 10)] = 4
    y[(d > 10) & (d <= 50)] = 5
    y[d > 50] = 6
    return y


def discretize_position(x):
    # Spec says >360 for class 4, hence exact 360 remains class 3.
    y = np.zeros(x.shape, dtype=np.int8)
    y[(x >= 90) & (x < 180)] = 1
    y[(x >= 180) & (x < 270)] = 2
    y[(x >= 270) & (x <= 360)] = 3
    y[x > 360] = 4
    return y


def discretize_speed(x):
    y = np.zeros(x.shape, dtype=np.int8)
    y[(x >= 2) & (x < 10)] = 1
    y[(x >= 10) & (x < 20)] = 2
    y[(x >= 20) & (x <= 40)] = 3
    y[x > 40] = 4
    return y


def nearest_frame_indices(frame_times, event_times):
    idx = np.clip(np.searchsorted(frame_times, event_times), 0, len(frame_times)-1)
    left = np.maximum(idx-1, 0)
    return np.where(np.abs(frame_times[left]-event_times) < np.abs(frame_times[idx]-event_times), left, idx)


def convert_session(path, make_plot=False):
    t0 = time.time()
    with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
        nwb = io.read()
        subject = str(nwb.subject.subject_id)
        scene = str(nwb.identifier).rstrip('/').split('/')[-1]
        bts = nwb.processing['behavior']['BehavioralTimeSeries']
        needed = ['environment', 'lick', 'position', 'scanning', 'speed',
                  'teleport', 'trial number', 'trial_start']
        arrays = {k: np.asarray(bts.time_series[k].data[:]).squeeze() for k in needed}
        deconv_container = nwb.processing['ophys']['Deconvolved']
        # Single-plane files have one RoiResponseSeries; multi-plane files have
        # plane0/plane1 series linked to disjoint rows of one segmentation table.
        series = list(deconv_container.roi_response_series.values())
        # Ten released sessions contain one additional terminal neural frame.
        common_n = min([r.data.shape[0] for r in series] + [len(v) for v in arrays.values()])
        arrays = {k: v[:common_n] for k, v in arrays.items()}
        neural_parts = []
        n_rois = 0
        neural_native_lengths = []
        for r in series:
            linked_rows = np.asarray(r.rois.data[:], dtype=np.int64)
            if r.data.shape[1] != len(linked_rows):
                raise ValueError(f'RoiResponseSeries/DynamicTableRegion mismatch in {path}: {r.name}')
            iscell = np.asarray(r.rois.table['iscell'][:])
            binary = iscell[:, 0] if iscell.ndim == 2 else iscell
            local_cell_mask = (binary[linked_rows] == 1)
            neural_parts.append(np.asarray(r.data[:common_n, local_cell_mask], dtype=np.float32))
            n_rois += len(linked_rows)
            neural_native_lengths.append(int(r.data.shape[0]))
        neural_all = np.ascontiguousarray(np.concatenate(neural_parts, axis=1), dtype=np.float32)
        n_cells = int(neural_all.shape[1])
        # Position timestamps define the common imaging-frame timebase.
        pts = bts.time_series['position']
        if pts.timestamps is not None:
            frame_times = np.asarray(pts.timestamps[:common_n], dtype=np.float64)
        else:
            frame_times = float(pts.starting_time) + np.arange(common_n)/float(pts.rate)
        reward = bts.time_series['Reward']
        reward_times = np.asarray(reward.timestamps[:], dtype=np.float64)
        event_frames = nearest_frame_indices(frame_times, reward_times) if len(reward_times) else np.array([], dtype=int)
        event_trial_ids = arrays['trial number'][event_frames] if len(event_frames) else np.array([])
        rewarded_ids = set(int(x) for x in event_trial_ids if np.isfinite(x) and x >= 0)

        trial_ids = np.unique(arrays['trial number'][np.isfinite(arrays['trial number']) &
                                                     (arrays['trial number'] >= 0)]).astype(int)
        valid_ids = []
        for tid in trial_ids:
            ix = np.flatnonzero(arrays['trial number'] == tid)
            complete = (len(ix) >= 20 and np.sum(arrays['trial_start'][ix] > 0) >= 1
                        and np.nanmax(arrays['position'][ix]) >= 440
                        and np.all(arrays['scanning'][ix] == 1))
            if complete:
                valid_ids.append(tid)
        neural_trials, input_trials, output_trials = [], [], []
        plot_info = []
        prev_outcome = 0
        for tid in valid_ids:
            ix = np.flatnonzero(arrays['trial number'] == tid)
            nt = len(ix)
            pos = arrays['position'][ix].astype(np.float32)
            speed = arrays['speed'][ix].astype(np.float32)
            lick = arrays['lick'][ix]
            env_values = arrays['environment'][ix]
            u, c = np.unique(env_values[np.isfinite(env_values)], return_counts=True)
            if not len(u):
                raise ValueError(f'No finite environment in {path}, trial {tid}')
            env_mode = u[np.argmax(c)]
            env, zone = parse_scene(scene, tid, env_mode)
            outcome = int(tid in rewarded_ids)
            low, high = ZONE_BOUNDS[zone]
            dist = signed_distance_to_interval(pos, low, high)

            inp = np.empty((4, nt), dtype=np.float32)
            inp[0] = np.arange(nt, dtype=np.float32) * np.float32(DT)
            inp[1] = env
            inp[2] = tid
            inp[3] = prev_outcome
            out = np.empty((6, nt), dtype=np.int8)
            out[0] = discretize_distance(dist)
            out[1] = discretize_position(pos)
            out[2] = discretize_speed(speed)
            out[3] = (lick > 0).astype(np.int8)
            out[4] = ZONE_INDEX[zone]
            out[5] = outcome
            neu = np.ascontiguousarray(neural_all[ix].T, dtype=np.float32)
            if neu.shape[1] != nt or inp.shape[1] != nt or out.shape[1] != nt:
                raise AssertionError('Within-trial temporal mismatch')
            if not (np.all(np.isfinite(neu)) and np.all(np.isfinite(inp))):
                raise ValueError(f'Nonfinite neural/input values in {path}, trial {tid}')
            neural_trials.append(neu); input_trials.append(inp); output_trials.append(out)
            if len(plot_info) < 4:
                plot_info.append((tid, pos.copy(), speed.copy(), (lick > 0).copy(), dist.copy(), out.copy()))
            prev_outcome = outcome

        if len(valid_ids) < 2:
            raise ValueError(f'Fewer than two valid trials in {path}')
        info = {
            'file': str(path), 'subject': subject, 'scene': scene,
            'n_cells': n_cells, 'n_rois': int(n_rois), 'n_planes': len(series),
            'n_trials': len(valid_ids), 'native_trial_ids': valid_ids,
            'common_n': common_n, 'neural_native_n': neural_native_lengths,
            'reward_events': len(reward_times), 'rewarded_trials': len(set(valid_ids) & rewarded_ids),
            'elapsed_s': time.time()-t0,
        }
    if make_plot:
        plot_processing(info, plot_info)
    return neural_trials, input_trials, output_trials, info


def plot_processing(info, trials):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(trials), 4, figsize=(16, 3.2*len(trials)), squeeze=False)
    for row, (tid, pos, speed, lick, dist, out) in enumerate(trials):
        t=np.arange(len(pos))*DT
        axes[row,0].plot(t,pos,label='position'); axes[row,0].plot(t,dist,label='signed zone distance'); axes[row,0].legend(fontsize=7)
        axes[row,1].plot(t,speed,label='speed'); axes[row,1].scatter(t[lick],np.zeros(np.sum(lick)),s=4,label='lick'); axes[row,1].legend(fontsize=7)
        axes[row,2].imshow(out[:4],aspect='auto',interpolation='nearest',extent=[0,t[-1] if len(t)>1 else 0,3.5,-.5]); axes[row,2].set_yticks(range(4),['distance','position','speed','lick'])
        axes[row,3].imshow(out[4:],aspect='auto',interpolation='nearest',extent=[0,t[-1] if len(t)>1 else 0,1.5,-.5]); axes[row,3].set_yticks(range(2),['zone','outcome'])
        for ax in axes[row]: ax.set_xlabel('seconds from trial start'); ax.set_title(f'trial {tid}')
    fig.suptitle(f"{info['subject']} {info['scene']}: raw behavior and categorical transforms")
    fig.tight_layout()
    safe=re.sub(r'[^A-Za-z0-9_.-]+','_',f"{info['subject']}_{Path(info['file']).stem}")
    fig.savefig(f'/app/processing_{safe}.png',dpi=130); plt.close(fig)


def validate_complete(data):
    ns=len(data['neural'])
    assert ns == len(data['input']) == len(data['output']) == len(data['subject_idx']) == len(data['brain_region_idx'])
    for s in range(ns):
        assert len(data['neural'][s]) == len(data['input'][s]) == len(data['output'][s]) >= 2
        nn=data['neural'][s][0].shape[0]
        assert data['brain_region_idx'][s].shape == (nn,)
        for n,i,o in zip(data['neural'][s],data['input'][s],data['output'][s]):
            assert n.ndim==2 and i.shape[0]==4 and o.shape[0]==6
            assert n.shape[0]==nn and n.shape[1]==i.shape[1]==o.shape[1]
            assert np.all(np.isfinite(n)) and np.all(np.isfinite(i))
            for j,k in enumerate((7,5,5,2,3,2)):
                assert o[j].min()>=0 and o[j].max()<k


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('outpicklefile')
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument('--full',action='store_true',help='process all sessions (default)')
    mode.add_argument('--sample',action='store_true',help='process first 2 sessions')
    ap.add_argument('--show-processing',action='store_true')
    args=ap.parse_args()
    files=sorted(DATA_ROOT.rglob('*.nwb'))
    if args.sample: files=files[:2]
    if not files: raise FileNotFoundError('No NWB files found')
    print(f'Converting {len(files)} sessions with pynwb; output={args.outpicklefile}',flush=True)
    neural=[]; inputs=[]; outputs=[]; infos=[]; subjects=[]
    t0=time.time()
    for j,path in enumerate(files):
        sn,si,so,info=convert_session(path,args.show_processing and j<2)
        neural.append(sn); inputs.append(si); outputs.append(so); infos.append(info); subjects.append(info['subject'])
        print(f"[{j+1}/{len(files)}] {path.relative_to(DATA_ROOT)} scene={info['scene']} cells={info['n_cells']} trials={info['n_trials']} rewarded={info['rewarded_trials']} time={info['elapsed_s']:.2f}s total={time.time()-t0:.1f}s",flush=True)
    subject_names=sorted(set(subjects))
    data={
      'neural':neural, 'input':inputs, 'output':outputs,
      'subjects':subject_names,
      'subject_idx':np.asarray([subject_names.index(x) for x in subjects],dtype=np.int32),
      'brain_regions':['CA1'],
      'brain_region_idx':[np.zeros(x[0].shape[0],dtype=np.int32) for x in neural],
      'input_names':['time from trial start (s)','environment type','trial number','previous trial outcome'],
      'output_names':['distance to reward zone','absolute position','speed','lick','reward zone location','reward outcome'],
      'output_values':[
        ['< -50 cm','-50 to -10 cm','-10 to < 0 cm','inside reward zone (0 cm)','>0 to +10 cm','+10 to +50 cm','> +50 cm'],
        ['< 90 cm','90 to 180 cm','180 to 270 cm','270 to 360 cm','> 360 cm'],
        ['< 2 cm/s','2-10 cm/s','10-20 cm/s','20-40 cm/s','> 40 cm/s'],
        ['no','yes'],['A','B','C'],['no','yes']],
      'metadata':{
        'task_description':'Head-fixed mice run a 450 cm virtual corridor in ENV1/ENV2 with hidden reward zones A/B/C; decode position, speed, licking, active zone, and reward outcome from CA1 deconvolved calcium activity.',
        'time_bin_size':DT*1000.0,
        'temporal_alignment_event':'start of numbered trial (trial_start; first aligned imaging frame)',
        'off_start':0.0, 'off_end':None,
        'sampling_rate_hz':1.0/DT,
        'neural_signal':'Suite2p deconvolved calcium activity for iscell-classified ROIs',
        'trial_window':'all frame-aligned samples belonging to each complete native numbered trial; variable duration',
        'reward_zone_bounds_cm':{'A':[80.0,100.0],'B':[200.0,220.0],'C':[320.0,340.0]},
        'session_info':infos,
        'source_format':'NWB read exclusively with pynwb',
      }}
    validate_complete(data)
    out=Path(args.outpicklefile); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('wb') as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Wrote {out} ({out.stat().st_size/1e9:.3f} GB) in {time.time()-t0:.1f}s; sessions={len(files)}, trials={sum(map(len,neural))}, cells={sum(x[0].shape[0] for x in neural)}',flush=True)

if __name__=='__main__': main()
