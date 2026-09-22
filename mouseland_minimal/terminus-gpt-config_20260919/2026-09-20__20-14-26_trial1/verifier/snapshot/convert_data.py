#!/usr/bin/env python3
"""Convert the published corridor-imaging data to the neural-decoder format.

The supplied Beh_*.npy objects already contain behavior interpolated to imaging
frames.  Neural loading follows the paper repository's utils.load_spk exactly:
the three arrays in the `spks` list are concatenated on the neuron axis.
"""
import os, glob, pickle, gc, re
from collections import defaultdict
import numpy as np

ROOT='/app/data'
OUT='/app/converted_data.pkl'

# Collect each physical session once. Several behavior files are analysis aliases
# (e.g. train/test before grating) and test3 metadata also repeats entries.
beh_by_session={}
source_by_session={}
for fn in sorted(glob.glob(os.path.join(ROOT,'beh','Beh_*.npy'))):
    B=np.load(fn,allow_pickle=True).item()
    for raw_sid,b in B.items():
        # Test3 stores two analysis views (_swap1/_swap2) of one physical
        # recording. Both have the same synchronized frames and include the
        # literal swapped WallName labels; match either view to the one spike file.
        sid=re.sub(r'_swap[12]$', '', raw_sid)
        if sid not in beh_by_session:
            beh_by_session[sid]=b
            source_by_session[sid]=os.path.basename(fn)+(':'+raw_sid if raw_sid != sid else '')
spk_files={os.path.basename(f).replace('_neural_data.npy',''):f
           for f in glob.glob(os.path.join(ROOT,'spk','*_neural_data.npy'))}
def parse_sid(s):
    p=s.split('_')
    return p[0], '_'.join(p[1:4]), p[4]

sessions=sorted(set(beh_by_session)&set(spk_files), key=parse_sid)
if set(spk_files)-set(beh_by_session):
    raise RuntimeError('Spike sessions without synchronized behavior: '+repr(sorted(set(spk_files)-set(beh_by_session))))

# Subject-specific chronological session rank is a reproducible continuous day
# covariate (calendar days from that animal's first included recording).
subject_dates=defaultdict(list)
for sid in sessions:
    m,d,_=parse_sid(sid); subject_dates[m].append(d)
first_date={m:min(v) for m,v in subject_dates.items()}
from datetime import datetime
def training_day(sid):
    m,d,_=parse_sid(sid)
    return float((datetime.strptime(d,'%Y_%m_%d')-datetime.strptime(first_date[m],'%Y_%m_%d')).days)

# Global quartiles over valid corridor frames, as requested (each bin is 25% of data).
all_speed=[]
for sid in sessions:
    b=beh_by_session[sid]; n=len(b['ft'])
    for st,en in zip(np.asarray(b['StartFr']),np.asarray(b['GrayFr'])):
        i=max(0,int(np.ceil(st))); j=min(n,int(np.ceil(en)))
        if j>i: all_speed.append(np.asarray(b['ft_RunSpeed'])[i:j])
speed_edges=np.nanquantile(np.concatenate(all_speed),[.25,.5,.75]).astype(float)
del all_speed

# Global visual labels are the literal published WallName categories.
visual_values=sorted({str(x) for sid in sessions for x in np.asarray(beh_by_session[sid]['WallName'])})
visual_id={x:i for i,x in enumerate(visual_values)}
subjects=sorted({parse_sid(s)[0] for s in sessions})
subject_id={s:i for i,s in enumerate(subjects)}
brain_regions=['V1','medial','anterior','lateral','unassigned']

neural=[]; inputs=[]; outputs=[]; region_idx=[]; session_info=[]
for si,sid in enumerate(sessions):
    b=beh_by_session[sid]
    obj=np.load(spk_files[sid],allow_pickle=True).item()
    planes=obj['spks']
    # Same operation as code/utils.py:load_spk.
    spk=np.concatenate([np.asarray(x) for x in planes],axis=0)
    nframes=min(spk.shape[1],len(b['ft']))
    ft=np.asarray(b['ft'],float); pos=np.asarray(b['ft_Pos'],float)
    speed=np.asarray(b['ft_RunSpeed'],float)
    starts=np.asarray(b['StartFr']); ends=np.asarray(b['GrayFr'])
    sounds=np.asarray(b['SoundFr']); walls=np.asarray(b['WallName']).astype(str)
    rewarded=np.asarray(b['isRew'],bool)
    lickfr=np.rint(np.asarray(b['LickFr'],float)).astype(int)
    lickset=set(lickfr[(lickfr>=0)&(lickfr<nframes)].tolist())
    ns=spk.shape[0]

    # Retinotopy iarea uses 1=V1, 2=medial, 3=anterior, 4=lateral;
    # one map applies to each of the three concatenated arrays.
    mname,date,block=parse_sid(sid)
    rf=os.path.join(ROOT,'retinotopy',mname+'_'+date+'_trans.npz')
    if os.path.exists(rf):
        ar=np.asarray(np.load(rf)['iarea']).astype(int)
        rid=np.full(ar.shape,4,dtype=np.int64)
        for raw,outid in ((1,0),(2,1),(3,2),(4,3)): rid[ar==raw]=outid
        reps=int(np.ceil(ns/len(rid))); rid=np.tile(rid,reps)[:ns]
    else: rid=np.full(ns,4,dtype=np.int64)

    sn=[]; sx=[]; sy=[]; kept=[]
    dt=float(np.nanmedian(np.diff(ft))*86400.0)
    for tr,(st,en,sf,wall,rw) in enumerate(zip(starts,ends,sounds,walls,rewarded)):
        i=max(0,int(np.ceil(st))); j=min(nframes,int(np.ceil(en)))
        if j<=i: continue
        inds=np.arange(i,j); T=j-i
        # Neural data are published deconvolved activity, no added smoothing/normalization.
        nn=np.asarray(spk[:,i:j],dtype=np.float32)
        t_since=(ft[i:j]-float(st if st<nframes else i)*0)*86400.0
        # Use frame interval rather than absolute MATLAB datenums to avoid cancellation.
        t_since=(inds-float(st))*dt
        t_sound=(float(sf)-inds)*dt
        xx=np.vstack((t_sound,
                      np.full(T,training_day(sid)),
                      t_since,
                      np.full(T,float(rw)))).astype(np.float32)
        lick=np.fromiter((1 if q in lickset else 0 for q in inds),dtype=np.int64,count=T)
        pbin=np.clip(np.floor(pos[i:j]/10.0),0,3).astype(np.int64)
        sbin=np.digitize(speed[i:j],speed_edges,right=False).astype(np.int64)
        yy=np.vstack((np.full(T,visual_id[wall],dtype=np.int64),lick,pbin,sbin))
        sn.append(nn); sx.append(xx); sy.append(yy); kept.append(tr)
    del spk,obj,planes
    if len(sn)<2: continue
    neural.append(sn); inputs.append(sx); outputs.append(sy); region_idx.append(rid)
    session_info.append({'session_id':sid,'behavior_source':source_by_session[sid],
                         'subject':mname,'date':date,
                         'block':block,'training_day':training_day(sid),
                         'n_trials_source':int(b['ntrials']),'kept_trial_indices':kept,
                         'n_neurons':int(ns),'median_frame_interval_s':dt})
    print(f'[{si+1}/{len(sessions)}] {sid}: {ns} neurons, {len(sn)} trials',flush=True)
    gc.collect()

data={'neural':neural,'input':inputs,'output':outputs,
      'subjects':subjects,
      'subject_idx':np.asarray([subject_id[x['subject']] for x in session_info],dtype=np.int64),
      'brain_regions':brain_regions,'brain_region_idx':region_idx,
      'input_names':['time_to_sound_cue','day_of_training','time_since_trial_start','reward_availability'],
      'output_names':['visual_stimulus_category','licking','corridor_position_bin','running_speed_quartile'],
      'output_values':[visual_values,['not licking','licking'],
                       ['0-1 m','1-2 m','2-3 m','3-4 m'],
                       ['0-25%','25-50%','50-75%','75-100%']],
      'metadata':{
       'task_description':'4-m virtual visual-corridor task; decode corridor identity, licking, position, and speed from published deconvolved two-photon activity.',
       'time_bin_size':float(np.median([x['median_frame_interval_s'] for x in session_info]))*1000.0,
       'temporal_alignment_event':'trial start (entry into the 4-m visual corridor; StartFr)',
       'off_start':0.0,'off_end':None,
       'neural_processing':'Published spks; concatenated exactly as repository utils.load_spk; no additional smoothing, normalization, or neuron filtering.',
       'trial_window':'ceil(StartFr) through frame before ceil(GrayFr), excluding the following gray space.',
       'frame_index_rounding':'Trial bounds use ceil; event frame for licking uses nearest frame.',
       'position_processing':'Published ft_Pos in 10 units/m, floor(ft_Pos/10), clipped to bins 0..3.',
       'speed_quartile_edges':speed_edges.tolist(),
       'speed_quartile_scope':'All valid corridor frames across all unique sessions.',
       'day_definition':'Calendar days since each subject first included imaging session.',
       'time_units':'seconds','session_info':session_info,
       'deduplication':'One physical session per *_neural_data.npy; duplicate behavior analysis aliases removed.'}}
print('Writing',OUT,flush=True)
with open(OUT,'wb') as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
print('Done:',len(neural),'sessions',sum(map(len,neural)),'trials',flush=True)
