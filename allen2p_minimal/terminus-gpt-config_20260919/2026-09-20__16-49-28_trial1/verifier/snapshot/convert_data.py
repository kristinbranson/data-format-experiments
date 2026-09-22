#!/usr/bin/env python3
"""Convert Allen Visual Behavior Ophys NWBs to decoder format.

Decisions follow the released AllenSDK/NWB products:
* one behavior_ophys_experiment NWB (one imaging plane/population) is a session;
* use released, ROI-curated dF/F (not recomputed fluorescence or inferred events);
* use SDK-defined trial flags and retain Go/Catch only, excluding aborted and
  auto-rewarded trials;
* use native ophys frames (~31 Hz) as the common clock and trial bins;
* use released filtered running speed and filtered pupil area.  Pupil diameter
  is 2*sqrt(area/pi). Missing blink/outlier samples are linearly interpolated;
* discretize running and pupil separately within each recording into quintiles
  using all retained trial frames (the requested equal percentile bins);
* image flashes last 250 ms in this task and are followed by 500 ms gray. Omitted
  flashes therefore remain gray. An image-change impulse is placed at the first
  ophys frame at/after the SDK display-lag-corrected trial change_time.
"""
import os, glob, pickle, re
import h5py
import numpy as np

ROOT='/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments'
OUT='/app/converted_data.pkl'
FILES=sorted(glob.glob(os.path.join(ROOT,'*.nwb')))

def dec(x):
    if isinstance(x,(bytes,np.bytes_)): return x.decode()
    return str(x)

def interp_finite(tnew,t,x):
    t=np.asarray(t,float); x=np.asarray(x,float)
    ok=np.isfinite(t)&np.isfinite(x)
    if ok.sum()==0: return np.zeros(len(tnew),dtype=np.float32)
    if ok.sum()==1: return np.full(len(tnew),x[ok][0],dtype=np.float32)
    return np.interp(tnew,t[ok],x[ok]).astype(np.float32)

def quintile(x, mask):
    """Codes 0..4; ties are handled deterministically by percentile edges."""
    vals=x[mask & np.isfinite(x)]
    if vals.size==0: return np.zeros(x.size,dtype=np.int8), [np.nan]*4
    edges=np.quantile(vals,[.2,.4,.6,.8])
    return np.searchsorted(edges,x,side='right').astype(np.int8), edges.tolist()

def find_dset(f, candidates):
    for q in candidates:
        if q in f: return f[q]
    raise KeyError(candidates)

# Global vocabularies are based on actual task trial labels, avoiding passive
# movie/template names unrelated to the Visual Behavior task.
subjects=[]; regions=[]; images=set()
for p in FILES:
    with h5py.File(p,'r') as f:
        sid=dec(f['general/subject/subject_id'][()])
        if sid not in subjects: subjects.append(sid)
        plane=next(iter(f['general/optophysiology'].values()))
        reg=dec(plane['location'][()])
        if reg not in regions: regions.append(reg)
        tr=f['intervals/trials']
        for col in ('initial_image_name','change_image_name'):
            images.update(dec(v) for v in tr[col][:] if dec(v) not in ('','nan','None'))
subjects=sorted(subjects); regions=sorted(regions); images=sorted(images)
image_values=['gray']+images
image_code={v:i for i,v in enumerate(image_values)}

neural=[]; inputs=[]; outputs=[]; subject_idx=[]; brain_region_idx=[]
session_info=[]; percentile_edges=[]
for fi,p in enumerate(FILES):
    with h5py.File(p,'r') as f:
        tr=f['intervals/trials']
        # Pupil diameter is a required decoder output; exclude recordings in
        # which the release has no eye-tracking stream rather than inventing it.
        if 'acquisition/EyeTracking/pupil_tracking' not in f:
            print(f'{fi+1}/{len(FILES)} skip: no pupil tracking {os.path.basename(p)}', flush=True)
            continue
        if 'stimulus/presentation' not in f or len(f['stimulus/presentation']) == 0:
            print(f'{fi+1}/{len(FILES)} skip: no stimulus presentations {os.path.basename(p)}', flush=True)
            continue
        keep=(tr['go'][:].astype(bool)|tr['catch'][:].astype(bool))
        keep &= ~tr['aborted'][:].astype(bool)
        keep &= ~tr['auto_rewarded'][:].astype(bool)
        tids=np.flatnonzero(keep)
        if len(tids)<2:
            print('skip <2 trials',p,flush=True); continue

        ot=find_dset(f,['processing/ophys/dff/traces/timestamps'])[:]
        dff=find_dset(f,['processing/ophys/dff/traces/data'])
        ncell=dff.shape[1]

        # Processed running speed supplied by AllenSDK/NWB.
        rt=f['processing/running/speed/timestamps'][:]
        rv=f['processing/running/speed/data'][:]
        run=interp_finite(ot,rt,rv)

        # Released filtered pupil area; timestamps are shared by EyeTracking.
        pg=f['acquisition/EyeTracking/pupil_tracking']
        pt=pg['timestamps'][:] if 'timestamps' in pg else f['acquisition/EyeTracking/eye_tracking/timestamps'][:]
        pa=pg['area'][:]
        if np.isfinite(pa).sum() < 2:
            print(f'{fi+1}/{len(FILES)} skip: insufficient pupil samples {os.path.basename(p)}', flush=True)
            continue
        diam=interp_finite(ot,pt,2*np.sqrt(np.maximum(pa,0)/np.pi))

        # Mask of frames belonging to retained trials, for session-level bins.
        retained=np.zeros(ot.size,bool); bounds=[]
        starts=tr['start_time'][:]; stops=tr['stop_time'][:]
        for j in tids:
            a=np.searchsorted(ot,starts[j],side='left')
            b=np.searchsorted(ot,stops[j],side='right')
            a=max(0,a); b=min(ot.size,b)
            if b>a: retained[a:b]=True
            bounds.append((j,a,b))
        runbin,redges=quintile(run,retained)
        pupbin,pedges=quintile(diam,retained)

        # Active task image onsets. Raw series omits omitted flashes; assigning
        # exactly 250 ms after each onset naturally leaves omissions/ISI gray.
        pres=f['stimulus/presentation']
        series=next(iter(pres.values()))
        pon=series['timestamps'][:]; pind=series['data'][:].astype(int)
        templ=next(iter(f['stimulus/templates'].values()))
        if 'control_description' in templ:
            labels=[dec(x) for x in templ['control_description'][:]]
        else:
            # Current release uses timestamps as image indices if descriptions
            # are absent; trial labels provide a conservative fallback.
            labels=images
        # In this release template control_description is indexed by control.
        img=np.zeros(ot.size,dtype=np.int16)
        for onset,ii in zip(pon,pind):
            if ii>=len(labels): continue
            label=labels[ii]
            if label not in image_code: continue
            a=np.searchsorted(ot,onset,'left'); b=np.searchsorted(ot,onset+0.250,'left')
            img[a:b]=image_code[label]

        ns=[]; ins=[]; outs=[]
        for j,a,b in bounds:
            if b<=a: continue
            # h5py reads time x cells; target is cells x time.
            x=np.asarray(dff[a:b,:],dtype=np.float32).T
            # Very rare nonfinite dF/F values are invalid for the decoder.
            if not np.isfinite(x).all():
                x=np.nan_to_num(x,nan=0.0,posinf=0.0,neginf=0.0)
            T=b-a
            change=np.zeros(T,dtype=np.int8)
            ct=float(tr['change_time'][j])
            # Catch trials have a sham change_time but no identity change.
            if np.isfinite(ct) and bool(tr['go'][j]):
                k=np.searchsorted(ot[a:b],ct,'left')
                if k<T: change[k]=1
            if bool(tr['hit'][j]): outcome=0
            elif bool(tr['miss'][j]): outcome=1
            elif bool(tr['false_alarm'][j]): outcome=2
            elif bool(tr['correct_reject'][j]): outcome=3
            else: raise ValueError(f'unclassified retained trial {j} in {p}')
            y=np.vstack((img[a:b],change,runbin[a:b],pupbin[a:b],
                         np.full(T,outcome,dtype=np.int8))).astype(np.int16)
            ns.append(x); ins.append(np.empty((0,T),dtype=np.float32)); outs.append(y)
        if len(ns)<2: continue
        sid=dec(f['general/subject/subject_id'][()])
        plane=next(iter(f['general/optophysiology'].values())); reg=dec(plane['location'][()])
        eid=int(re.search(r'(\d+)\.nwb$',p).group(1))
        neural.append(ns); inputs.append(ins); outputs.append(outs)
        subject_idx.append(subjects.index(sid))
        brain_region_idx.append(np.full(ncell,regions.index(reg),dtype=np.int16))
        session_info.append({'ophys_experiment_id':eid,'mouse_id':sid,
          'targeted_structure':reg,'n_cells':ncell,'n_trials':len(ns),
          'ophys_frame_rate_hz':float(plane['imaging_rate'][()])})
        percentile_edges.append({'running_speed_cm_per_s':redges,'pupil_diameter_pixels':pedges})
    print(f'{fi+1}/{len(FILES)} {os.path.basename(p)}: {ncell} cells, {len(ns)} trials',flush=True)

data={'neural':neural,'input':inputs,'output':outputs,
 'subjects':subjects,'subject_idx':np.asarray(subject_idx,dtype=np.int16),
 'brain_regions':regions,'brain_region_idx':brain_region_idx,
 'input_names':[],
 'output_names':['image identity','image change','running speed quintile','pupil diameter quintile','trial outcome'],
 'output_values':[image_values,['no change','change'],['Q1','Q2','Q3','Q4','Q5'],
                  ['Q1','Q2','Q3','Q4','Q5'],['hit','miss','false alarm','correct reject']],
 'metadata':{
  'task_description':'Visual change-detection task; decode flashed image, image changes, locomotion, pupil diameter, and Go/Catch trial outcome from two-photon dF/F.',
  'time_bin_size':1000.0/31.0,
  'temporal_alignment_event':'Native ophys timestamps; each trial spans SDK trial start_time through stop_time.',
  'off_start':None,'off_end':None,
  'neural_measure':'Allen released detrended dF/F from valid curated ROIs',
  'trial_filter':'(go OR catch) AND NOT aborted AND NOT auto_rewarded; sessions lacking required pupil tracking excluded',
  'image_flash_duration_s':0.250,
  'continuous_discretization':'Per-recording quintiles over retained trial frames; filtered missing pupil samples linearly interpolated.',
  'session_info':session_info,'percentile_edges':percentile_edges,
  'source_release':'visual-behavior-ophys 1.1.0'}}
with open(OUT+'.tmp','wb') as fh: pickle.dump(data,fh,protocol=pickle.HIGHEST_PROTOCOL)
os.replace(OUT+'.tmp',OUT)
print('saved',OUT,os.path.getsize(OUT),flush=True)
