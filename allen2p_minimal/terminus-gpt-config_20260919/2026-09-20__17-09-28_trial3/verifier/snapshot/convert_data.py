#!/usr/bin/env python3
"""Convert Allen Visual Behavior Ophys NWBs to the neural-decoder format.

Decisions follow the AllenSDK and supplied tutorials:
* An ophys experiment is a session: it has one timestamp stream and one population.
* SDK dF/F is used as neural activity. It is demixed, neuropil-corrected, detrended,
  and contains only ROIs passing the released dataset's cell/ROI curation.
* Trials retain their experimental start/stop boundaries and native ophys frames.
* Go and Catch trials are retained; aborted and auto-rewarded trials are removed.
* SDK filtered running speed and eye-tracking pupil area are interpolated to ophys
  timestamps. Equivalent circular diameter is derived from area (a monotone transform,
  so its percentile classes equal pupil-area percentile classes).
* Quintile boundaries are calculated independently within each recording, avoiding
  across-mouse calibration differences in running wheels and eye cameras.
"""
import argparse, gc, pickle, warnings
from pathlib import Path
import numpy as np
from allensdk.brain_observatory.behavior.behavior_ophys_experiment import BehaviorOphysExperiment


def interp_finite(t_new, t, x):
    t=np.asarray(t,float); x=np.asarray(x,float)
    good=np.isfinite(t)&np.isfinite(x)
    if good.sum()==0: return np.zeros(len(t_new),dtype=np.float32)
    if good.sum()==1: return np.full(len(t_new),x[good][0],dtype=np.float32)
    # np.interp uses nearest endpoint outside the measured interval and linear
    # interpolation internally; both behavior streams are sampled at >= ophys rate.
    return np.interp(t_new,t[good],x[good]).astype(np.float32)


def quintiles(x):
    """Codes 0..4 using equal-percentile boundaries; robust to tied boundaries."""
    x=np.asarray(x,float)
    edges=np.nanquantile(x,[.2,.4,.6,.8])
    return np.searchsorted(edges,x,side='right').astype(np.int16), edges.tolist()


def outcome_code(row):
    if bool(row['hit']): return 0
    if bool(row['miss']): return 1
    if bool(row['false_alarm']): return 2
    if bool(row['correct_reject']): return 3
    raise ValueError('Retained Go/Catch trial has no standard outcome')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',default='/app/data')
    ap.add_argument('--output',default='/app/converted_data.pkl')
    ap.add_argument('--limit',type=int,default=None,help='testing: process only first N NWBs')
    args=ap.parse_args()
    paths=sorted(Path(args.data_dir).rglob('behavior_ophys_experiment_*.nwb'))
    if args.limit is not None: paths=paths[:args.limit]
    if not paths: raise FileNotFoundError('No behavior ophys experiment NWBs found')

    neural=[]; inputs=[]; outputs=[]; subject_ids=[]; session_regions=[]
    session_info=[]; image_to_code={'gray':0}; image_values=['gray']

    for si,p in enumerate(paths):
        print(f'[{si+1}/{len(paths)}] {p.name}',flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ds=BehaviorOphysExperiment.from_nwb_path(str(p))
        ts=np.asarray(ds.ophys_timestamps,float)
        dff=np.vstack(ds.dff_traces['dff'].values).astype(np.float32)
        if dff.shape[1]!=len(ts): raise ValueError(f'dF/F timestamp mismatch in {p}')

        # AllenSDK's running_speed is the transient-corrected, 10-Hz low-pass signal.
        run=interp_finite(ts,ds.running_speed['timestamps'],ds.running_speed['speed'])
        eye=ds.eye_tracking
        area=interp_finite(ts,eye['timestamps'],eye['pupil_area'])
        diameter=(2*np.sqrt(np.maximum(area,0)/np.pi)).astype(np.float32)
        run_bin,run_edges=quintiles(run); pupil_bin,pupil_edges=quintiles(diameter)

        stim=ds.stimulus_presentations
        trials=ds.trials
        keep=(trials['go'].astype(bool)|trials['catch'].astype(bool)) \
             & ~trials['aborted'].astype(bool) & ~trials['auto_rewarded'].astype(bool)
        trials=trials.loc[keep]
        sn=[]; sx=[]; sy=[]
        for tid,row in trials.iterrows():
            lo=int(np.searchsorted(ts,float(row.start_time),side='left'))
            hi=int(np.searchsorted(ts,float(row.stop_time),side='right'))
            if hi<=lo: continue
            tt=ts[lo:hi]; nt=len(tt)
            image=np.zeros(nt,dtype=np.int16)  # gray between 250-ms flashes
            change=np.zeros(nt,dtype=np.int16)
            # Restrict to presentations overlapping this trial. Omitted presentations
            # remain gray. is_change denotes the task's actual image identity change,
            # rather than every image-to-gray flash transition.
            sub=stim[(stim.start_time<=tt[-1]) & (stim.end_time>=tt[0])]
            for _,pr in sub.iterrows():
                name=pr.get('image_name',np.nan)
                if not isinstance(name,str) or name=='omitted': continue
                if name not in image_to_code:
                    image_to_code[name]=len(image_values); image_values.append(name)
                a=int(np.searchsorted(tt,float(pr.start_time),side='left'))
                b=int(np.searchsorted(tt,float(pr.end_time),side='left'))
                if b>a: image[a:b]=image_to_code[name]
                if bool(pr.get('is_change',False)) and a<nt: change[a]=1
            oc=outcome_code(row)
            # Outcome is static by definition. It is repeated because temporal and
            # static targets share a rectangular output matrix in the decoder format.
            y=np.vstack((image,change,run_bin[lo:hi],pupil_bin[lo:hi],
                         np.full(nt,oc,dtype=np.int16)))
            sn.append(dff[:,lo:hi]); sx.append(np.empty((0,nt),dtype=np.float32)); sy.append(y)
        if len(sn)<2:
            print('  skipped: fewer than two retained trials',flush=True); del ds; gc.collect(); continue
        neural.append(sn); inputs.append(sx); outputs.append(sy)
        m=ds.metadata; subject_ids.append(str(m['mouse_id'])); session_regions.append(str(m['targeted_structure']))
        session_info.append({'ophys_experiment_id':int(m['ophys_experiment_id']),
          'behavior_session_id':int(m['behavior_session_id']), 'ophys_session_id':int(m['ophys_session_id']),
          'session_type':str(m['session_type']), 'imaging_depth_um':int(m['imaging_depth']),
          'targeted_structure':str(m['targeted_structure']), 'n_cells':int(dff.shape[0]),
          'n_trials':len(sn), 'running_quintile_edges_cm_s':run_edges,
          'pupil_diameter_quintile_edges':pupil_edges})
        del ds,dff,run,area,diameter,run_bin,pupil_bin; gc.collect()

    subjects=sorted(set(subject_ids)); smap={x:i for i,x in enumerate(subjects)}
    regions=sorted(set(session_regions)); rmap={x:i for i,x in enumerate(regions)}
    data={'neural':neural,'input':inputs,'output':outputs,
      'subjects':subjects,'subject_idx':np.asarray([smap[x] for x in subject_ids],dtype=np.int64),
      'brain_regions':regions,
      'brain_region_idx':[np.full(len(sess[0]),rmap[r],dtype=np.int64) for sess,r in zip(neural,session_regions)],
      'input_names':[],
      'output_names':['image_identity','image_change','running_speed_quintile','pupil_diameter_quintile','trial_outcome'],
      'output_values':[image_values,['no_change','change'],
        ['quintile_1','quintile_2','quintile_3','quintile_4','quintile_5'],
        ['quintile_1','quintile_2','quintile_3','quintile_4','quintile_5'],
        ['hit','miss','false_alarm','correct_reject']],
      'metadata':{'task_description':'Visual change-detection task; decode flashed image identity, true image-change onset, running-speed quintile, pupil-diameter quintile, and four-class Go/Catch trial outcome from calcium activity.',
        'time_bin_size':float(1000/31.0),'temporal_alignment_event':'Native ophys timestamps within each experimental trial (trial start to trial stop).',
        'off_start':0.0,'off_end':None,'neural_measure':'AllenSDK demixed, neuropil-corrected, detrended dF/F',
        'trial_filter':'Go or Catch; aborted and auto-rewarded excluded',
        'pupil_measure':'Equivalent circular diameter derived from AllenSDK pupil_area',
        'trial_outcome_encoding':'Static per trial; repeated over time to share the temporal output matrix',
        'session_info':session_info}}
    with open(args.output,'wb') as f: pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
    print(f'Wrote {args.output}: {len(neural)} sessions, {sum(map(len,neural))} trials, {len(subjects)} subjects',flush=True)

if __name__=='__main__': main()
