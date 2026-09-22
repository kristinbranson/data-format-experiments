#!/usr/bin/env python3
"""Convert the Sosa et al. reward-relative CA1 NWB release for decoding.

Decisions matching the paper/repository
---------------------------------------
* Every released session is retained.  The 77-session number in the paper is
  the reward-switch subset (there are 77 switch and 75 fixed-zone NWBs), not a
  general quality filter.
* Suite2p deconvolved events are used, as in the paper's population/GLM
  analyses. Only ROIs for which Suite2p ``iscell[:, 0]`` is true are retained.
* Behavior and events are already resampled to imaging frames in the NWBs.
  Complete trials are [trial_start, teleport), matching get_trial_types and
  trial-matrix calls in the supplied repository. No speed filtering is used:
  the <2 cm/s exclusion in the paper applies only to specified spatial maps.
* Scene identity is encoded in the NWB identifier. Paper code maps task labels
  A/B/C to physical zones X/Y/Z = [80,130]/[200,250]/[320,370] cm and uses
  trial 30 as the default switch. Distance is signed to the nearest point in
  the active zone: negative before it, zero within it, positive after it.
* Reward outcome is actual reward delivery within the trial. Licks are frame
  counts and are converted to presence/absence.
"""
from pathlib import Path
import glob, os, pickle, re
import h5py
import numpy as np

DATA_ROOT = '/app/data'
OUT = '/app/converted_data.pkl'
ZONE_COORDS = {'A': (80.0, 130.0), 'B': (200.0, 250.0), 'C': (320.0, 370.0)}
ZONE_INDEX = {'A': 0, 'B': 1, 'C': 2}

def text(x):
    return x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)

def parse_scene(identifier):
    """Return scene, environment (0/1), first/second zone labels."""
    scene = text(identifier).rstrip('/').split('/')[-1]
    # Environment type requested is binary. Cross-environment sessions switch
    # environment along with zone; ordinary sessions have one EnvN prefix.
    pairs = re.findall(r'Env([12])(?:_Location)?([ABC])', scene)
    if len(pairs) >= 2:
        (e0,z0),(e1,z1) = pairs[0],pairs[1]
    else:
        em = re.search(r'Env([12])', scene)
        if not em:
            raise ValueError(f'Cannot parse environment from {scene}')
        e0=e1=em.group(1)
        # Handles LocationA, LocationA_to_B, and A_to_Env1_C forms.
        zs = re.findall(r'(?:Location)?([ABC])', scene)
        if not zs:
            raise ValueError(f'Cannot parse reward location from {scene}')
        z0, z1 = zs[0], (zs[-1] if '_to_' in scene else zs[0])
    return scene, int(e0)-1, int(e1)-1, z0, z1

def distance_class(pos, lo, hi):
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

def convert():
    files = sorted(glob.glob(DATA_ROOT + '/sub-*/*.nwb'))
    if not files:
        raise FileNotFoundError('No NWB files found')
    subjects = sorted({Path(f).parent.name.removeprefix('sub-') for f in files},
                      key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
    subj_map = {s:i for i,s in enumerate(subjects)}
    neural, inputs, outputs, region_idx, subject_idx, session_info = [],[],[],[],[],[]
    for si,f in enumerate(files):
        with h5py.File(f,'r') as h:
            subject = text(h['general/subject/subject_id'][()])
            session_id = text(h['general/session_id'][()])
            scene,e0,e1,z0,z1 = parse_scene(h['identifier'][()])
            b = h['processing/behavior/BehavioralTimeSeries']
            starts = np.flatnonzero(b['trial_start/data'][:])
            ends = np.flatnonzero(b['teleport/data'][:])
            if len(starts) != len(ends):
                raise ValueError(f'Unmatched boundaries in {f}')
            valid = (ends > starts)
            starts, ends = starts[valid], ends[valid]
            ts = b['position/timestamps'][:]
            pos_all = b['position/data'][:]
            speed_all = b['speed/data'][:]
            lick_all = b['lick/data'][:]
            trialnum_all = b['trial number/data'][:]
            # Sparse reward-delivery timestamps (seconds on same clock).
            reward_times = b['Reward/timestamps'][:]
            # The 28 dual-plane NWBs have one concatenated segmentation table
            # but one deconvolved response series per plane. ROI table order is
            # plane0 followed by plane1 (widths sum exactly to table length).
            deconv = h['processing/ophys/Deconvolved']
            plane_names = sorted(deconv.keys(), key=lambda q: int(re.search(r'([0-9]+)$',q).group(1)))
            event_sets = [deconv[q]['data'] for q in plane_names]
            # Ten dual-plane files contain one unused trailing neural frame.
            # Both streams start at t=0 at the imaging rate; align by frame and
            # ignore that terminal sample (all trial teleports precede it).
            if any(d.shape[0] < len(ts) or d.shape[0]-len(ts) > 1 for d in event_sets):
                raise ValueError(f'Behavior/neural frame mismatch in {f}')
            iscell = h['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:,0] > 0
            widths = [d.shape[1] for d in event_sets]
            if sum(widths) != len(iscell):
                raise ValueError(f'Plane ROI widths do not match segmentation in {f}')
            plane_cell_ids=[]; off=0
            for width in widths:
                plane_cell_ids.append(np.flatnonzero(iscell[off:off+width]))
                off += width
            n_cells = int(sum(map(len,plane_cell_ids)))
            ns, xs, ys = [], [], []
            outcomes = []
            # Determine outcomes first so previous outcome can be assigned.
            for a,e in zip(starts,ends):
                outcomes.append(int(np.any((reward_times >= ts[a]) & (reward_times < ts[e]))))
            switched = '_to_' in scene
            for ti,(a,e) in enumerate(zip(starts,ends)):
                sl = slice(int(a),int(e))
                ntime = e-a
                after = switched and ti >= 30
                env = e1 if after else e0
                zone = z1 if after else z0
                lo,hi = ZONE_COORDS[zone]
                pos = np.asarray(pos_all[sl], dtype=np.float32)
                speed = np.asarray(speed_all[sl], dtype=np.float32)
                # Small negative values are numerical differentiation noise and
                # properly fall in the specified <2 cm/s category.
                t = np.asarray(ts[sl]-ts[a], dtype=np.float32)
                trnum = float(trialnum_all[a])
                if not np.isfinite(trnum) or trnum < 0: trnum = float(ti)
                prev = float(outcomes[ti-1] if ti else 0)
                x = np.vstack((t,
                    np.full(ntime,env,np.float32),
                    np.full(ntime,trnum,np.float32),
                    np.full(ntime,prev,np.float32)))
                y = np.vstack((
                    distance_class(pos,lo,hi),
                    np.digitize(pos,[90,180,270,360],right=False).astype(np.int8),
                    np.digitize(speed,[2,10,20,40],right=False).astype(np.int8),
                    (np.asarray(lick_all[sl])>0).astype(np.int8),
                    np.full(ntime,ZONE_INDEX[zone],np.int8),
                    np.full(ntime,outcomes[ti],np.int8)))
                # HDF5 selection is time x selected ROI; transpose to neuron x time.
                n = np.concatenate([np.asarray(d[sl,ids],dtype=np.float32).T
                                    for d,ids in zip(event_sets,plane_cell_ids)],axis=0)
                ns.append(n); xs.append(x); ys.append(y)
            if len(ns) < 2 or n_cells==0:
                continue
            neural.append(ns); inputs.append(xs); outputs.append(ys)
            subject_idx.append(subj_map[subject])
            region_idx.append(np.zeros(n_cells,dtype=np.int16))
            session_info.append({'file':os.path.relpath(f,DATA_ROOT),'subject':subject,
                'session_id':session_id,'scene':scene,'n_trials':len(ns),
                'n_cells':n_cells,'switch_trial':30 if switched else None})
        print(f'[{si+1:3d}/{len(files)}] {subject} ses-{session_id}: {scene}, {len(ns)} trials, {n_cells} cells',flush=True)
    data = {
      'neural': neural, 'input': inputs, 'output': outputs,
      'subjects': subjects, 'subject_idx': np.asarray(subject_idx,dtype=np.int16),
      'brain_regions': ['CA1'], 'brain_region_idx': region_idx,
      'input_names': ['time from trial start','environment type','trial number','previous trial outcome'],
      'output_names': ['distance to reward zone','absolute position','speed','lick','reward zone location','reward outcome'],
      'output_values': [
        ['< -50 cm','-50 to -10 cm','-10 to <0 cm','in reward zone','>0 to +10 cm','+10 to +50 cm','> +50 cm'],
        ['<90 cm','90-180 cm','180-270 cm','270-360 cm','>360 cm'],
        ['<2 cm/s','2-10 cm/s','10-20 cm/s','20-40 cm/s','>40 cm/s'],
        ['no','yes'], ['A','B','C'], ['no reward','rewarded']],
      'metadata': {
        'task_description':'Head-fixed mice navigate a 450 cm virtual linear corridor; CA1 activity predicts position, movement, licking, active reward location, and reward delivery.',
        'time_bin_size':1000.0/15.5078125,
        'temporal_alignment_event':'start of trial (entry to linear track)',
        'off_start':0.0, 'off_end':None,
        'neural_measure':'Suite2p deconvolved calcium events from iscell-classified CA1 ROIs',
        'trial_interval':'trial_start frame inclusive to teleport frame exclusive',
        'reward_zone_coordinates_cm':{'A':[80,130],'B':[200,250],'C':[320,370]},
        'session_info':session_info}}
    with open(OUT+'.tmp','wb') as fp: pickle.dump(data,fp,protocol=4)
    os.replace(OUT+'.tmp',OUT)
    print(f'Wrote {OUT}: {len(neural)} sessions, {sum(map(len,neural))} trials')
if __name__=='__main__': convert()
