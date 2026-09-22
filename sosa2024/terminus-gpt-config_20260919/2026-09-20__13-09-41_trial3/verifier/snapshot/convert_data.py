#!/usr/bin/env python3
"""Convert Sosa et al. 2025 NWB ophys data to decoder-compatible pickle.

Usage: python -u /app/convert_data.py OUT.pkl [--full|--sample] [--show-processing]
"""
import argparse, glob, os, pickle, re, time
from pathlib import Path
import h5py
import numpy as np

DATA_ROOT = '/app/data'
ZONE_COORDS = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
ZONE_INDEX = {'A': 0, 'B': 1, 'C': 2}
DT_REFERENCE = 1.0 / 15.5078125


def natural_key(path):
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', path)]


def parse_scene(identifier):
    """Return ordered A/B/C schedule labels from final identifier component."""
    scene = identifier.rstrip('/').split('/')[-1]
    labels = re.findall(r'(?:Location)?([ABC])', scene)
    if len(labels) not in (1, 2):
        raise ValueError(f'Cannot parse reward schedule from {scene!r}: {labels}')
    return scene, labels


def zone_for_trial(labels, chronological_index):
    # Matches behavior.get_reward_zones(..., change_trial=30).
    return labels[0] if len(labels) == 1 or chronological_index < 30 else labels[1]


def distance_classes(position, start, stop):
    distance = np.where(position < start, position-start,
                        np.where(position > stop, position-stop, 0.0))
    out = np.empty(distance.shape, dtype=np.int64)
    out[distance < -50] = 0
    out[(distance >= -50) & (distance < -10)] = 1
    out[(distance >= -10) & (distance < 0)] = 2
    out[distance == 0] = 3
    out[(distance > 0) & (distance <= 10)] = 4
    out[(distance > 10) & (distance <= 50)] = 5
    out[distance > 50] = 6
    return out, distance


def find_complete_trials(b):
    """Pair each global start with first teleport before next start."""
    start_flags = np.asarray(b['trial_start/data'][:])
    teleport_flags = np.asarray(b['teleport/data'][:])
    starts = np.flatnonzero(start_flags > 0)
    teleports = np.flatnonzero(teleport_flags > 0)
    trials = []
    for j, start in enumerate(starts):
        next_start = starts[j+1] if j+1 < len(starts) else len(start_flags)
        candidates = teleports[(teleports > start) & (teleports < next_start)]
        if len(candidates) != 1:
            raise ValueError(f'start {start}: expected one following teleport before {next_start}, got {candidates}')
        trials.append((int(start), int(candidates[0])))  # stop is exclusive
    return trials


def reward_outcomes(b, timestamps, trials):
    reward_t = np.asarray(b['Reward/timestamps'][:], dtype=float)
    return np.asarray([np.any((reward_t >= timestamps[a]) & (reward_t < timestamps[z]))
                       for a, z in trials], dtype=np.int64)


def load_neural_and_regions(h):
    """Concatenate ophys planes in PlaneSegmentation order and apply Suite2p iscell."""
    dec = h['processing/ophys/Deconvolved']
    plane_names = sorted((k for k, v in dec.items() if isinstance(v, h5py.Group)), key=natural_key)
    arrays = [np.asarray(dec[p]['data'][:], dtype=np.float32) for p in plane_names]
    if not arrays or len({x.shape[0] for x in arrays}) != 1:
        raise ValueError('Missing planes or unequal plane time dimensions')
    full = np.concatenate(arrays, axis=1)
    seg = h['processing/ophys/ImageSegmentation/PlaneSegmentation']
    iscell = np.asarray(seg['iscell'][:, 0]) > 0
    plane_idx = np.asarray(seg['planeIdx'][:], dtype=np.int16)
    if full.shape[1] != len(iscell) or len(plane_idx) != len(iscell):
        raise ValueError(f'plane concatenation {full.shape} != segmentation {len(iscell)}')
    return full[:, iscell], plane_idx[iscell], plane_names


def process_session(path, make_plot=False):
    t0 = time.time()
    with h5py.File(path, 'r') as h:
        b = h['processing/behavior/BehavioralTimeSeries']
        names = ['environment', 'position', 'speed', 'lick', 'trial number']
        streams = {n: np.asarray(b[n+'/data'][:]) for n in names}
        timestamps = np.asarray(b['position/timestamps'][:], dtype=np.float64)
        ntime = len(timestamps)
        if any(len(x) != ntime for x in streams.values()):
            raise ValueError('Behavior stream length mismatch')
        dt = float(np.median(np.diff(timestamps)))
        if not np.isclose(dt, DT_REFERENCE, rtol=0, atol=2e-5):
            raise ValueError(f'Unexpected behavior dt {dt}')
        trials = find_complete_trials(b)
        outcomes = reward_outcomes(b, timestamps, trials)
        neural_all, curated_planes, plane_names = load_neural_and_regions(h)
        neural_excess_frames = int(neural_all.shape[0] - ntime)
        # Ten released two-plane NWBs have exactly one unmatched trailing neural row.
        # It lies after the final behavior/teleport sample and cannot belong to a valid trial.
        if neural_excess_frames not in (0, 1):
            raise ValueError(f'Unexpected neural/behavior row difference: {neural_all.shape[0]} vs {ntime}')
        if neural_excess_frames:
            neural_all = neural_all[:ntime]
        subject = h['general/subject/subject_id'][()].decode()
        identifier = h['identifier'][()].decode()
        scene, labels = parse_scene(identifier)

        neural_trials, input_trials, output_trials = [], [], []
        kept_info = []
        bad_lick = 0
        for j, (start, stop) in enumerate(trials):
            q = slice(start, stop)
            n = stop-start
            lick_raw = streams['lick'][q]
            # Reference correct_lick_sensor_error: fraction of frames whose cumulative count is >2.
            if np.mean(lick_raw > 2) > 0.30:
                bad_lick += 1
                continue
            trial_id = int(round(float(streams['trial number'][start])))
            env_values = np.unique(streams['environment'][q])
            env_values = env_values[env_values >= 0]
            if len(env_values) != 1 or env_values[0] not in (0, 1):
                raise ValueError(f'Non-binary/nonconstant environment trial {j}: {env_values}')
            env = int(env_values[0])
            zone = zone_for_trial(labels, j)
            zstart, zstop = ZONE_COORDS[zone]
            pos = streams['position'][q].astype(np.float32)
            speed = streams['speed'][q].astype(np.float32)
            dclass, distance = distance_classes(pos, zstart, zstop)
            posclass = np.digitize(pos, [90, 180, 270, 360], right=False).astype(np.int64)
            speedclass = np.digitize(speed, [2, 10, 20, 40], right=False).astype(np.int64)
            lickclass = (lick_raw > 0).astype(np.int64)
            prev_outcome = int(outcomes[j-1]) if j > 0 else 0
            inp = np.vstack([
                (timestamps[q]-timestamps[start]).astype(np.float32),
                np.full(n, env, dtype=np.float32),
                np.full(n, trial_id, dtype=np.float32),
                np.full(n, prev_outcome, dtype=np.float32),
            ])
            out = np.vstack([
                dclass, posclass, speedclass, lickclass,
                np.full(n, ZONE_INDEX[zone], dtype=np.int64),
                np.full(n, outcomes[j], dtype=np.int64),
            ])
            neu = np.ascontiguousarray(neural_all[q, :].T, dtype=np.float32)
            if neu.shape[1] != n or inp.shape != (4, n) or out.shape != (6, n):
                raise AssertionError('Trial shape mismatch')
            if not (np.isfinite(neu).all() and np.isfinite(inp).all()):
                raise ValueError('Nonfinite neural/input values')
            neural_trials.append(neu); input_trials.append(inp); output_trials.append(out)
            kept_info.append(dict(source_trial_index=j, trial_id=trial_id, start=start, stop=stop,
                                  zone=zone, outcome=int(outcomes[j]), environment=env,
                                  distance_min=float(distance.min()), distance_max=float(distance.max())))

        if len(neural_trials) < 2:
            raise ValueError(f'Only {len(neural_trials)} valid trials')
        info = dict(file=os.path.relpath(path, DATA_ROOT), subject=subject, scene=scene,
                    identifier=identifier, n_frames=ntime, time_bin_s=dt,
                    n_rois_raw=int(len(h['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'])),
                    n_neurons=int(neural_all.shape[1]), planes=plane_names,
                    curated_plane_idx=curated_planes, neural_excess_frames_trimmed=neural_excess_frames,
                    complete_trials=len(trials),
                    excluded_bad_lick=bad_lick, kept_trials=len(neural_trials), trials=kept_info,
                    seconds=float(time.time()-t0))
    if make_plot:
        plot_processing(path, neural_trials, input_trials, output_trials, info)
    return neural_trials, input_trials, output_trials, info


def plot_processing(path, neural, inputs, outputs, info):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    k = min(2, len(neural)-1); t = inputs[k][0]
    show_n = min(25, neural[k].shape[0])
    ax[0].imshow(neural[k][:show_n], aspect='auto', origin='lower', extent=[t[0],t[-1],0,show_n]); ax[0].set_ylabel('deconv cells')
    ax[1].plot(t, outputs[k][1], label='position class'); ax[1].plot(t, outputs[k][0], label='zone-distance class'); ax[1].legend()
    ax[2].plot(t, outputs[k][2], label='speed class'); ax[2].plot(t, outputs[k][3], label='lick binary', alpha=.7); ax[2].legend()
    ax[3].plot(t, inputs[k][1], label='environment'); ax[3].plot(t, outputs[k][4], label='zone A/B/C'); ax[3].legend()
    ax[4].plot(t, inputs[k][0], label='time input'); ax[4].set_xlabel('seconds from trial start'); ax[4].legend()
    fig.suptitle(f"{info['file']} trial {info['trials'][k]['trial_id']} start-aligned; raw→classes")
    fig.tight_layout()
    session_id=Path(path).stem.replace('_behavior+ophys','')
    fig.savefig(f'/app/processing_{session_id}.png', dpi=140); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('outpicklefile')
    mode=ap.add_mutually_exclusive_group(); mode.add_argument('--full',action='store_true'); mode.add_argument('--sample',action='store_true')
    ap.add_argument('--show-processing',action='store_true'); args=ap.parse_args()
    files=sorted(glob.glob(DATA_ROOT+'/**/*.nwb',recursive=True),key=natural_key)
    if args.sample: files=files[:2]
    print(f'Processing {len(files)} sessions to {args.outpicklefile}',flush=True)
    neural=[]; inputs=[]; outputs=[]; infos=[]
    subjects=[]; subject_idx=[]; region_idx=[]
    t0=time.time()
    for i,f in enumerate(files):
        ns,xs,ys,info=process_session(f, make_plot=args.show_processing and i<2)
        if info['subject'] not in subjects: subjects.append(info['subject'])
        subject_idx.append(subjects.index(info['subject']))
        neural.append(ns); inputs.append(xs); outputs.append(ys)
        region_idx.append(np.zeros(info['n_neurons'],dtype=np.int64)); infos.append(info)
        print(f"[{i+1}/{len(files)}] {info['file']}: neurons={info['n_neurons']} trials={info['kept_trials']}/{info['complete_trials']} badlick={info['excluded_bad_lick']} time={info['seconds']:.2f}s",flush=True)
    dts=np.asarray([x['time_bin_s'] for x in infos])
    data=dict(neural=neural,input=inputs,output=outputs,subjects=subjects,
              subject_idx=np.asarray(subject_idx,dtype=np.int64),brain_regions=['dorsal CA1'],
              brain_region_idx=region_idx,
              input_names=['time from trial start (s)','environment type','trial number','previous trial outcome'],
              output_names=['distance to reward zone','absolute position','speed','lick','reward zone location','reward outcome'],
              output_values=[['< -50 cm','-50 to < -10 cm','-10 to < 0 cm','inside reward zone (0 cm)','> 0 to 10 cm','> 10 to 50 cm','> 50 cm'],
                             ['< 90 cm','90 to < 180 cm','180 to < 270 cm','270 to < 360 cm','>= 360 cm'],
                             ['< 2 cm/s','2 to < 10 cm/s','10 to < 20 cm/s','20 to < 40 cm/s','>= 40 cm/s'],
                             ['no lick','lick'],['A','B','C'],['omitted','rewarded']],
              metadata=dict(task_description='Decode framewise navigation and licking plus trial reward context/outcome from dorsal CA1 deconvolved calcium activity.',
                            time_bin_size=float(np.median(dts)*1000),temporal_alignment_event='start of trial (trial_start pulse)',
                            off_start=0.0,off_end=None,source='DANDI 001361; Sosa et al., A flexible hippocampal population code for experience relative to reward',
                            neural_signal='Suite2p deconvolved calcium events, Suite2p iscell ROIs only',
                            trial_window='trial_start inclusive to teleport exclusive',session_info=infos))
    with open(args.outpicklefile,'wb') as f: pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
    total_trials=sum(map(len,neural)); total_cells=sum(x.shape[0] for x in region_idx)
    print(f'SAVED {args.outpicklefile} sessions={len(neural)} trials={total_trials} summed_neurons={total_cells} elapsed={time.time()-t0:.1f}s size={os.path.getsize(args.outpicklefile)/1e9:.3f}GB',flush=True)

if __name__=='__main__': main()
