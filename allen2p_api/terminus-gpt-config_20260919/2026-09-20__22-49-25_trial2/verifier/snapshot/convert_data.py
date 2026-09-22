#!/usr/bin/env python3
"""Convert local Allen Visual Behavior Ophys data to decoder format.

All experiment content is loaded through VisualBehaviorOphysProjectCache; NWB
files are never opened directly. Usage:
  python -u /app/convert_data.py OUT.pkl [--full|--sample] [--show-processing]
"""
import argparse, gc, pickle, re, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache

CACHE_DIR = Path('/app/data')
BIN_S = 0.1
MISSING_EYE_IDS = {795953296, 833631914, 806456687}
IMAGE_NAMES = ['im000','im031','im035','im045','im054','im061','im062','im063',
               'im065','im066','im069','im073','im075','im077','im085','im106']
IMAGE_TO_ID = {x:i+1 for i,x in enumerate(IMAGE_NAMES)}
OUTCOMES = ['hit','miss','false_alarm','correct_reject']


def local_ids():
    root=CACHE_DIR/'visual-behavior-ophys-1.1.0'/'behavior_ophys_experiments'
    return sorted(int(re.search(r'_(\d+)\.nwb$',p.name).group(1)) for p in root.glob('behavior_ophys_experiment_*.nwb'))


def interp_valid(t, v, q):
    t=np.asarray(t,float); v=np.asarray(v,float)
    good=np.isfinite(t)&np.isfinite(v)
    if good.sum()<2: raise ValueError('fewer than two valid samples')
    t=t[good]; v=v[good]; order=np.argsort(t); t=t[order]; v=v[order]
    return np.interp(q,t,v,left=v[0],right=v[-1])


def quintile_edges(v):
    v=np.asarray(v,float); v=v[np.isfinite(v)]
    if len(v)<5: raise ValueError('insufficient samples for quintiles')
    return np.quantile(v,[.2,.4,.6,.8])


def discretize(v, edges):
    return np.clip(np.searchsorted(edges,v,side='right'),0,4).astype(np.int64)


def aggregate_events(event_cumsum, timestamps, edges):
    """Sum event magnitude by time bin from one experiment-level cumulative sum."""
    lo=np.searchsorted(timestamps,edges[:-1],side='left')
    hi=np.searchsorted(timestamps,edges[1:],side='left')
    return (event_cumsum[:,hi]-event_cumsum[:,lo]).astype(np.float32)


def task_stimuli(exp):
    sp=exp.stimulus_presentations
    return sp[sp.stimulus_block_name.eq('change_detection_behavior')].copy()


def outcome_value(row):
    vals=[bool(row[k]) for k in OUTCOMES]
    if sum(vals)!=1: raise ValueError(f'non-exclusive outcome flags {vals}')
    return int(np.flatnonzero(vals)[0])


def convert_experiment(exp, row, image_to_id):
    xid=int(row.name); ot=np.asarray(exp.ophys_timestamps,float)
    ev=exp.events
    traces=np.stack(ev.filtered_events.to_numpy()).astype(np.float32)
    if traces.shape[1]!=len(ot): raise ValueError('event/timestamp length mismatch')
    if not np.isfinite(traces).all(): raise ValueError('nonfinite filtered events')
    # Compute once per experiment; recomputing this for every trial is quadratic in trial count.
    event_cumsum=np.concatenate([np.zeros((traces.shape[0],1),np.float32),
                                  np.cumsum(traces,dtype=np.float32,axis=1)],axis=1)
    trials=exp.trials
    trials=trials[trials.go.astype(bool)|trials.catch.astype(bool)].copy()
    if len(trials)<2: raise ValueError('fewer than two valid trials')
    # Continuous streams and experiment-level quintiles.
    run=exp.running_speed
    rt=run.timestamps.to_numpy(float); rv=run.speed.to_numpy(float)
    eye=exp.eye_tracking
    if len(eye)==0: raise ValueError('missing eye tracking')
    pupil=2.0*np.sqrt(eye.pupil_area.to_numpy(float)/np.pi)
    pt=eye.timestamps.to_numpy(float)
    run_edges=quintile_edges(rv)
    pupil_edges=quintile_edges(pupil)
    sp=task_stimuli(exp)
    real=sp.image_name.isin(image_to_id)
    presentations=sp[real][['start_time','end_time','image_name','is_change']]
    neural=[]; inputs=[]; outputs=[]; durations=[]
    raw_plot=None
    for ti,(_,tr) in enumerate(trials.iterrows()):
        start=float(tr.start_time); stop=float(tr.stop_time)
        n=max(1,int(np.ceil((stop-start)/BIN_S)))
        edges=start+np.arange(n+1)*BIN_S; edges[-1]=stop
        centers=(edges[:-1]+edges[1:])/2
        neu=aggregate_events(event_cumsum,ot,edges)
        image=np.zeros(n,np.int64); change=np.zeros(n,np.int64)
        # Only presentations overlapping the trial can contribute.
        cand=presentations[(presentations.end_time>start)&(presentations.start_time<stop)]
        for _,pr in cand.iterrows():
            m=(centers>=float(pr.start_time))&(centers<float(pr.end_time))
            image[m]=image_to_id[str(pr.image_name)]
            if bool(pr.is_change) and start<=float(pr.start_time)<stop:
                bi=min(n-1,max(0,int(np.floor((float(pr.start_time)-start)/BIN_S))))
                change[bi]=1
        run_cont=interp_valid(rt,rv,centers)
        pupil_cont=interp_valid(pt,pupil,centers)
        out=np.empty((5,n),np.int64)
        out[0]=image; out[1]=change
        out[2]=discretize(run_cont,run_edges)
        out[3]=discretize(pupil_cont,pupil_edges)
        # Validator/model accepts static output as (doutput,), but outputs are
        # jointly represented; repeat static outcome over time for a uniform
        # (5,T) output and true static semantics.
        out[4]=outcome_value(tr)
        inp=np.empty((0,n),np.float32)
        assert neu.shape[1]==inp.shape[1]==out.shape[1]
        assert np.isfinite(neu).all() and np.isfinite(out).all()
        neural.append(neu); inputs.append(inp); outputs.append(out); durations.append(stop-start)
        if raw_plot is None:
            raw_plot=(centers,neu[:min(8,len(neu))],image,change,run_cont,pupil_cont,out[2],out[3])
    info={'ophys_experiment_id':xid,'ophys_session_id':int(row.ophys_session_id),
          'behavior_session_id':int(row.behavior_session_id),'mouse_id':str(row.mouse_id),
          'session_type':str(row.session_type),'targeted_structure':str(row.targeted_structure),
          'cre_line':str(row.cre_line),'n_neurons':int(traces.shape[0]),
          'n_trials':len(trials),'ophys_frame_rate_measured':float(1/np.median(np.diff(ot))),
          'trial_duration_min_s':float(np.min(durations)),'trial_duration_max_s':float(np.max(durations)),
          'running_quintile_edges':run_edges.tolist(),'pupil_diameter_quintile_edges':pupil_edges.tolist()}
    return neural,inputs,outputs,info,raw_plot


def make_plot(xid, raw, info):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t,neu,img,ch,run,pup,rb,pb=raw; tr=t-t[0]
    fig,ax=plt.subplots(5,1,figsize=(12,11),sharex=True)
    ax[0].imshow(neu,aspect='auto',extent=[tr[0],tr[-1],neu.shape[0],0]); ax[0].set_ylabel('event cells')
    ax[1].step(tr,img,where='mid',label='image class'); ax[1].step(tr,ch*img.max(),where='mid',label='change pulse'); ax[1].legend(); ax[1].set_ylabel('stimulus')
    ax[2].plot(tr,run,label='continuous'); ax[2].step(tr,rb,where='mid',label='quintile'); ax[2].legend(); ax[2].set_ylabel('running')
    ax[3].plot(tr,pup,label='diameter'); ax[3].step(tr,pb,where='mid',label='quintile'); ax[3].legend(); ax[3].set_ylabel('pupil')
    ax[4].hist(neu.ravel(),bins=50); ax[4].set_ylabel('event histogram'); ax[4].set_xlabel('seconds from trial start / event magnitude')
    fig.suptitle(f"Experiment {xid}: aligned 100 ms processing\n{info['session_type']} {info['targeted_structure']}")
    fig.tight_layout(); fig.savefig(f'/app/processing_{xid}.png',dpi=140); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('outpicklefile')
    mode=ap.add_mutually_exclusive_group(); mode.add_argument('--full',action='store_true'); mode.add_argument('--sample',action='store_true')
    ap.add_argument('--show-processing',action='store_true'); args=ap.parse_args()
    warnings.filterwarnings('ignore',message='Ignoring cached namespace')
    warnings.filterwarnings('ignore',category=FutureWarning)
    t0=time.time(); cache=VisualBehaviorOphysProjectCache.from_local_cache(CACHE_DIR)
    table=cache.get_ophys_experiment_table(); ids=table.index.intersection(local_ids())
    tab=table.loc[ids].sort_index()
    tab=tab[~tab.session_type.str.contains('passive',case=False,na=False)]
    tab=tab[~tab.index.isin(MISSING_EYE_IDS)]
    if args.sample:
        # Deliberately include one multiscope (~11 Hz) and one single-plane (~31 Hz).
        multi=tab[tab.project_code.eq('VisualBehaviorMultiscope')].index[0]
        single=tab[tab.project_code.eq('VisualBehavior')].index[0]
        tab=tab.loc[[multi,single]]
    print(f'Candidate experiments: {len(tab)}',flush=True)
    subjects=sorted(tab.mouse_id.astype(str).unique()); regions=sorted(tab.targeted_structure.unique())
    neural=[]; inputs=[]; outputs=[]; subject_idx=[]; region_idx=[]; infos=[]
    for j,(xid,row) in enumerate(tab.iterrows(),1):
        ts=time.time(); exp=cache.get_behavior_ophys_experiment(int(xid))
        n,i,o,info,raw=convert_experiment(exp,row,IMAGE_TO_ID)
        neural.append(n); inputs.append(i); outputs.append(o)
        subject_idx.append(subjects.index(str(row.mouse_id)))
        region_idx.append(np.full(info['n_neurons'],regions.index(row.targeted_structure),np.int64))
        infos.append(info)
        if args.show_processing and j<=2: make_plot(int(xid),raw,info)
        print(f'[{j}/{len(tab)}] {xid}: cells={info["n_neurons"]} trials={info["n_trials"]} rate={info["ophys_frame_rate_measured"]:.2f} Hz time={time.time()-ts:.1f}s',flush=True)
        del exp,n,i,o,raw; gc.collect()
    data={'neural':neural,'input':inputs,'output':outputs,
          'subjects':subjects,'subject_idx':np.asarray(subject_idx,np.int64),
          'brain_regions':regions,'brain_region_idx':region_idx,
          'input_names':[],
          'output_names':['image_identity','image_change','running_speed_quintile','pupil_diameter_quintile','trial_outcome'],
          'output_values':[['gray']+IMAGE_NAMES,['no_change','change'],
                           ['Q1_slowest','Q2','Q3','Q4','Q5_fastest'],
                           ['Q1_smallest','Q2','Q3','Q4','Q5_largest'],
                           ['hit','miss','false_alarm','correct_reject']],
          'metadata':{'task_description':'Go/no-go natural-image change detection; decode stimulus, change events, running, pupil diameter, and behavioral outcome from detected calcium events.',
                      'time_bin_size':100.0,'temporal_alignment_event':'SDK trial start_time; bins follow absolute ophys time',
                      'off_start':0.0,'off_end':None,'trial_window':'variable SDK start_time to stop_time',
                      'neural_signal':'AllenSDK filtered_events summed in 100 ms bins',
                      'source_cache':'Visual Behavior Ophys 1.1.0 via VisualBehaviorOphysProjectCache',
                      'session_unit':'ophys experiment (imaging plane)','excluded_passive':True,
                      'excluded_missing_eye_tracking':sorted(MISSING_EYE_IDS),'image_class_0':'gray screen or omitted flash',
                      'percentile_scope':'within ophys experiment over full valid continuous stream',
                      'session_info':infos}}
    # Whole-dataset structural checks.
    assert len(neural)==len(inputs)==len(outputs)==len(subject_idx)==len(region_idx)==len(infos)
    assert all(len(s)>=2 for s in neural)
    assert all(len(n)==len(i)==len(o) for n,i,o in zip(neural,inputs,outputs))
    with open(args.outpicklefile,'wb') as f: pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
    total_trials=sum(map(len,neural)); total_cells=sum(x.shape[0] for x in region_idx)
    print(f'Saved {args.outpicklefile}: sessions={len(neural)} trials={total_trials} experiment-cells={total_cells} size={Path(args.outpicklefile).stat().st_size/1e9:.3f} GB total_time={time.time()-t0:.1f}s',flush=True)

if __name__=='__main__': main()
