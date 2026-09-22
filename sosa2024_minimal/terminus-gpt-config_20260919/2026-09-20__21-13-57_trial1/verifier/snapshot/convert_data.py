#!/usr/bin/env python3
"""Convert the Sosa et al. reward-relative CA1 NWB release for neural decoding.

Decisions:
* Use the supplied OASIS-deconvolved activity and Suite2p/manual `iscell` curation.
* Align each lap at the NWB trial_start pulse (the zero-cm crossing), and stop at
  teleport onset; teleport activity is not part of a corridor trial.
* The recordings have rates 15.5078125 and 31.015625 Hz.  Average blocks of 4
  and 8 frames respectively, giving one common 257.934 ms bin. Behavioral
  continuous quantities are averaged, lick is any lick in a bin.
* Reward-zone coordinates and labels follow reward_relative.behavior:
  A/X=80--130, B/Y=200--250, C/Z=320--370 cm. NWB identifiers preserve scene
  names. Switches occur at trial 30, as in the task/repository default.
* Reward outcome is presence of an NWB Reward event between trial start and
  teleport; previous outcome is chronological within session (first trial=0).
"""
import glob, os, pickle
import h5py
import numpy as np

DATA_ROOT = '/app/data'
OUT = '/app/converted_data.pkl'
TARGET_DT = 4 / 15.5078125
ZONE_COORDS = {'A': (80.,130.), 'B': (200.,250.), 'C': (320.,370.)}
ZONE_ID = {'A':0, 'B':1, 'C':2}


def scene_info(identifier):
    scene = identifier.decode() if isinstance(identifier, bytes) else str(identifier)
    scene = scene.rstrip('/').split('/')[-1]
    # LocationA, LocationA_to_B, and cross-environment A_to_Env2_B forms.
    import re
    locs = re.findall(r'(?:Location)?([ABC])', scene)
    if not locs:
        raise ValueError(f'Cannot identify reward location from {scene}')
    z0, z1 = locs[0], (locs[-1] if len(locs) > 1 else locs[0])
    switched = z1 != z0
    return scene, z0, z1, switched


def binned_mean(x, starts, block):
    # x is time x features (or a vector). Drop only the final incomplete block.
    return np.stack([np.asarray(x[s:min(s+block, len(x))]).mean(axis=0)
                     for s in starts], axis=0)


def discretize_distance(pos, lo, hi):
    d = np.where(pos < lo, pos-lo, np.where(pos > hi, pos-hi, 0.0))
    y = np.empty(d.shape, np.int8)
    y[d < -50] = 0
    y[(d >= -50) & (d < -10)] = 1
    y[(d >= -10) & (d < 0)] = 2
    y[d == 0] = 3
    y[(d > 0) & (d <= 10)] = 4
    y[(d > 10) & (d <= 50)] = 5
    y[d > 50] = 6
    return y


def convert_session(path):
    with h5py.File(path, 'r') as f:
        b = f['processing/behavior/BehavioralTimeSeries']
        scene, z0, z1, switched = scene_info(f['identifier'][()])
        dg = f['processing/ophys/Deconvolved']
        plane_names = sorted(dg.keys(), key=lambda x: int(x.replace('plane','')))
        rates = [float(dg[n]['starting_time'].attrs['rate']) for n in plane_names]
        if not np.allclose(rates, rates[0]):
            raise ValueError(f'Plane frame rates disagree in {path}: {rates}')
        # Arrays are aligned by frame index. In 28 NWBs the ophys rate attribute
        # is the resonant scanner line/plane rate (31 Hz), while shared frame
        # timestamps are 15.5 Hz. Use those timestamps as the authoritative rate.
        bt = b['position/timestamps']
        dt = float(np.median(np.diff(bt[:min(len(bt), 2000)])))
        rate = 1.0 / dt
        block = int(round(TARGET_DT / dt))
        if not np.isclose(block * dt, TARGET_DT, rtol=0, atol=2e-6):
            raise ValueError(f'Unexpected aligned-frame interval {dt} in {path}')

        seg = f['processing/ophys/ImageSegmentation/PlaneSegmentation']
        iscell = seg['iscell'][:,0] == 1
        plane_idx = seg['planeIdx'][:].astype(int) if 'planeIdx' in seg else np.zeros(len(iscell), int)
        plane_series = []
        for name in plane_names:
            pi = int(name.replace('plane',''))
            dset = dg[name]['data']
            mask = iscell[plane_idx == pi]
            if dset.shape[1] != len(mask):
                raise ValueError(f'ROI count mismatch for {name}: {dset.shape[1]} vs {len(mask)}')
            plane_series.append((dset, np.flatnonzero(mask)))
        n_cells = sum(len(idx) for _,idx in plane_series)

        pos = b['position/data'][:]
        speed = b['speed/data'][:]
        lick = b['lick/data'][:]
        env = b['environment/data'][:]
        trial_num = b['trial number/data'][:]
        trial_start = b['trial_start/data'][:]
        teleport = b['teleport/data'][:]
        timestamps = b['position/timestamps'][:]
        starts = np.flatnonzero(trial_start > 0)

        # Reward is a TimeSeries: event values are in data and event times in timestamps.
        rg = b['Reward']
        reward_times = rg['timestamps'][:] if 'timestamps' in rg else np.empty(0)

        neural_trials, input_trials, output_trials = [], [], []
        trial_records = []
        for s in starts:
            tq = np.flatnonzero(teleport[s:] > 0)
            e = s + int(tq[0]) if len(tq) else len(pos)
            if e <= s:
                continue
            # Ignore a clipped final trial if it never reaches teleport.
            if not len(tq):
                continue
            offsets = np.arange(s, e, block, dtype=int)
            if len(offsets) < 2:
                continue
            tr = int(round(float(np.median(trial_num[s:e]))))
            zone = z1 if (switched and tr >= 30) else z0
            lo, hi = ZONE_COORDS[zone]

            # Read only this trial and selected accepted cells from HDF5.
            raw_parts = [dset[s:e, :][:, idx] for dset,idx in plane_series]
            raw = np.concatenate(raw_parts, axis=1)
            nbin = binned_mean(raw, np.arange(0, e-s, block), block).T.astype(np.float16)
            pbin = binned_mean(pos[s:e], np.arange(0, e-s, block), block)
            sbin = binned_mean(speed[s:e], np.arange(0, e-s, block), block)
            lbin = np.asarray([np.any(lick[q:min(q+block,e)] > 0) for q in offsets], np.int8)
            # Environment is a per-trial input. Use the NWB stream, converting ENV1/2 to 0/1.
            ev = int(round(float(np.median(env[s:e]))))
            ev = 1 if ev > 0 else 0
            t0, t1 = timestamps[s], timestamps[e-1] + 1/rate
            rewarded = int(np.any((reward_times >= t0) & (reward_times < t1)))

            T = len(offsets)
            inp = np.vstack([
                np.arange(T, dtype=np.float32) * np.float32(TARGET_DT),
                np.full(T, ev, np.float32),
                np.full(T, tr, np.float32),
                np.zeros(T, np.float32),  # filled after chronological outcomes known
            ])
            dist_cls = discretize_distance(pbin, lo, hi)
            pos_cls = np.digitize(pbin, [90,180,270,360], right=False).astype(np.int8)
            speed_cls = np.digitize(sbin, [2,10,20,40], right=False).astype(np.int8)
            out = np.vstack([
                dist_cls, pos_cls, speed_cls, lbin,
                np.full(T, ZONE_ID[zone], np.int8),
                np.full(T, rewarded, np.int8),
            ])
            trial_records.append((tr, rewarded, nbin, inp, out))

        # Previous outcome follows actual chronological trial order, not list accidents.
        trial_records.sort(key=lambda x: x[0])
        prev = 0
        for tr, rewarded, nbin, inp, out in trial_records:
            inp[3,:] = prev
            prev = rewarded
            neural_trials.append(nbin)
            input_trials.append(inp)
            output_trials.append(out)
        return neural_trials, input_trials, output_trials, n_cells, scene, rate, block


def main():
    files = sorted(glob.glob(os.path.join(DATA_ROOT, 'sub-*', '*_behavior+ophys.nwb')))
    if not files:
        raise FileNotFoundError('No NWB files found')
    subjects = sorted({os.path.basename(os.path.dirname(p)).replace('sub-','') for p in files})
    subj_to_idx = {s:i for i,s in enumerate(subjects)}
    data = {
        'neural': [], 'input': [], 'output': [],
        'subjects': subjects, 'subject_idx': [],
        'brain_regions': ['CA1'], 'brain_region_idx': [],
        'input_names': ['time from trial start (s)', 'environment type', 'trial number',
                        'previous trial outcome'],
        'output_names': ['distance to reward zone', 'absolute corridor position', 'speed',
                         'lick', 'reward zone location', 'reward outcome'],
        'output_values': [
            ['< -50 cm','-50 to -10 cm','-10 to <0 cm','in reward zone','>0 to +10 cm','+10 to +50 cm','> +50 cm'],
            ['<90 cm','90-180 cm','180-270 cm','270-360 cm','>360 cm'],
            ['<2 cm/s','2-10 cm/s','10-20 cm/s','20-40 cm/s','>40 cm/s'],
            ['no lick','lick'], ['A','B','C'], ['omitted','rewarded']],
        'metadata': {
            'task_description': 'Head-fixed mice navigate a 450 cm virtual corridor for hidden rewards; reward location switches among A, B, and C and the environment changes between ENV1 and ENV2.',
            'time_bin_size': float(TARGET_DT*1000),
            'temporal_alignment_event': 'trial_start pulse at the 0 cm corridor crossing',
            'off_start': 0.0, 'off_end': None,
            'neural_signal': 'OASIS-deconvolved calcium activity supplied in NWB, block averaged',
            'trial_end': 'first teleport sample after trial start',
            'roi_filter': 'NWB PlaneSegmentation iscell == 1 (Suite2p plus manual curation)',
            'reward_zone_coordinates_cm': {'A':[80,130], 'B':[200,250], 'C':[320,370]},
            'source': 'Sosa et al., A flexible hippocampal population code for experience relative to reward',
            'session_info': []
        }
    }
    for i,p in enumerate(files):
        nt, it, ot, nc, scene, rate, block = convert_session(p)
        if len(nt) < 2:
            print('SKIP (<2 trials)', p, flush=True)
            continue
        subj = os.path.basename(os.path.dirname(p)).replace('sub-','')
        data['neural'].append(nt); data['input'].append(it); data['output'].append(ot)
        data['subject_idx'].append(subj_to_idx[subj])
        data['brain_region_idx'].append(np.zeros(nc, dtype=np.int8))
        data['metadata']['session_info'].append({
            'file': os.path.relpath(p, DATA_ROOT), 'scene':scene,
            'source_frame_rate_hz':rate, 'frames_per_bin':block,
            'n_trials':len(nt), 'n_neurons':nc})
        print(f'[{i+1}/{len(files)}] {subj} {scene}: {len(nt)} trials, {nc} cells', flush=True)
    data['subject_idx'] = np.asarray(data['subject_idx'], dtype=np.int16)
    with open(OUT, 'wb') as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Wrote {OUT}: {len(data["neural"])} sessions, '
          f'{sum(map(len,data["neural"]))} trials, {os.path.getsize(OUT)/2**30:.2f} GiB')

if __name__ == '__main__':
    main()
