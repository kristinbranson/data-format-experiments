#!/usr/bin/env python3
"""Convert Allen Visual Behavior 2P NWBs for neural decoding.

Decisions follow the released NWB processing and the paper: detected calcium
Events (rather than dF/F) are the neural signal; only active change-detection
sessions and Go/Catch trials are retained.  Streams are aligned in the common
ophys clock and binned at 100 ms around the scheduled image change.
"""
import glob, os, pickle, warnings
import h5py
import numpy as np

DATA_GLOB='/app/data/**/*.nwb'
OUT='/app/converted_data.pkl'
DT=0.100
OFF0,OFF1=-3.0,4.2
NB=int(round((OFF1-OFF0)/DT))
ACTIVE={'OPHYS_1_images_A','OPHYS_3_images_A','OPHYS_4_images_B','OPHYS_6_images_B'}

def text(x):
    x=x[()]
    return x.decode() if isinstance(x,bytes) else str(x)

def interp_clean(t,x,q):
    t=np.asarray(t); x=np.asarray(x,dtype=float)
    good=np.isfinite(t)&np.isfinite(x)
    if good.sum()<2: return np.zeros(len(q),dtype=np.float32)
    return np.interp(q,t[good],x[good]).astype(np.float32)

def quintile_reference(x):
    x=np.asarray(x,dtype=float); x=x[np.isfinite(x)]
    if len(x)==0: return np.array([-np.inf]*4)
    return np.quantile(x,[.2,.4,.6,.8])

def main():
    files=[]
    for f in sorted(glob.glob(DATA_GLOB,recursive=True)):
        with h5py.File(f,'r') as h:
            if text(h['session_description']) in ACTIVE: files.append(f)
    # Global categorical image names are index-defined separately for familiar
    # and novel sets, hence prefix by set A/B. Gray is class zero.
    image_values=['gray']+[f'{s}_image_{i}' for s in ('A','B') for i in range(8)]
    subjects=sorted({text(h5py.File(f,'r')['general/subject/subject_id']) for f in files})
    subjmap={v:i for i,v in enumerate(subjects)}
    neural=[]; inputs=[]; outputs=[]; subject_idx=[]; region_idx=[]; session_info=[]
    regions=[]; regionmap={}
    for fi,f in enumerate(files):
      with h5py.File(f,'r') as h:
        stype=text(h['session_description']); mouse=text(h['general/subject/subject_id'])
        expid=int(text(h['identifier'])); setoff=1 if 'images_A' in stype else 9
        ev=h['processing/ophys/event_detection/data']
        ot=np.asarray(h['processing/ophys/event_detection/timestamps'],dtype=float)
        ncell=ev.shape[1]
        # Structure is represented by the imaging-plane location attribute.
        plane=h['general/optophysiology']
        loc='visual cortex'
        for _,g in plane.items():
            if isinstance(g,h5py.Group) and 'location' in g:
                loc=text(g['location']); break
        if loc not in regionmap: regionmap[loc]=len(regions); regions.append(loc)
        # Trial columns are already AllenSDK-derived trial definitions.
        tr=h['intervals/trials']
        go=np.asarray(tr['go']); catch=np.asarray(tr['catch'])
        abort=np.asarray(tr['aborted']); auto=np.asarray(tr['auto_rewarded'])
        change=np.asarray(tr['change_time'],dtype=float)
        keep=np.flatnonzero((go|catch)&~abort&~auto&np.isfinite(change))
        # Continuous streams, on their native synchronized clocks.
        rt=np.asarray(h['processing/running/speed/timestamps']); rv=np.asarray(h['processing/running/speed/data'])
        pbase='acquisition/EyeTracking/pupil_tracking'
        if pbase in h:
            pt=np.asarray(h['acquisition/EyeTracking/eye_tracking/timestamps'])
            pv=np.asarray(h[pbase+'/width'])
        else: pt=np.array([]); pv=np.array([])
        rq=quintile_reference(rv)
        pq=quintile_reference(pv)
        # Image presentation TimeSeries: image index at each 250-ms presentation.
        preskeys=list(h['stimulus/presentation'].keys())
        pg=h['stimulus/presentation/'+preskeys[0]]
        stim_t=np.asarray(pg['timestamps'],dtype=float); stim_i=np.asarray(pg['data'])
        # Values outside 0..7 (omissions) are gray.
        sn=[]; si=[]; so=[]
        hit=np.asarray(tr['hit']); miss=np.asarray(tr['miss'])
        fa=np.asarray(tr['false_alarm']); cr=np.asarray(tr['correct_reject'])
        for j in keep:
            edges=change[j]+OFF0+np.arange(NB+1)*DT
            centers=(edges[:-1]+edges[1:])/2
            # Average event magnitude in each temporal bin. Reading the common
            # contiguous frame slab once per trial avoids loading full recordings.
            a=np.searchsorted(ot,edges[0]); b=np.searchsorted(ot,edges[-1])
            slab=np.asarray(ev[a:b,:],dtype=np.float32)
            tt=ot[a:b]; mat=np.zeros((ncell,NB),dtype=np.float32)
            bi=np.floor((tt-edges[0])/DT).astype(int)
            for k in range(NB):
                z=slab[bi==k]
                if len(z): mat[:,k]=z.mean(axis=0)
            run=interp_clean(rt,rv,centers); pup=interp_clean(pt,pv,centers)
            runbin=np.digitize(run,rq).astype(np.int16)
            pupbin=np.digitize(pup,pq).astype(np.int16)
            # Screen is gray except during [onset,onset+.25). Omitted indices are gray.
            pos=np.searchsorted(stim_t,centers,side='right')-1
            img=np.zeros(NB,dtype=np.int16)
            ok=(pos>=0)
            pp=np.clip(pos,0,len(stim_t)-1)
            shown=ok&(centers<stim_t[pp]+.25)&(stim_i[pp]<8)
            img[shown]=(setoff+stim_i[pp[shown]].astype(int)).astype(np.int16)
            ch=np.zeros(NB,dtype=np.int16); ch[np.argmin(abs(centers-change[j]))]=1
            outcome=0 if hit[j] else 1 if miss[j] else 2 if fa[j] else 3
            out=np.vstack([img,ch,runbin,pupbin,np.full(NB,outcome,dtype=np.int16)])
            sn.append(mat); si.append(np.empty((0,NB),dtype=np.float32)); so.append(out)
        if len(sn)>=2:
            neural.append(sn); inputs.append(si); outputs.append(so)
            subject_idx.append(subjmap[mouse]); region_idx.append(np.full(ncell,regionmap[loc],dtype=np.int32))
            session_info.append({'ophys_experiment_id':expid,'mouse_id':mouse,'session_type':stype,
                                 'n_cells':ncell,'n_trials':len(sn),'source_file':os.path.basename(f)})
        print(f'[{fi+1}/{len(files)}] {expid}: {ncell} cells, {len(sn)} trials',flush=True)
    data={'neural':neural,'input':inputs,'output':outputs,'subjects':subjects,
          'subject_idx':np.asarray(subject_idx,dtype=np.int32),'brain_regions':regions,
          'brain_region_idx':region_idx,'input_names':[],
          'output_names':['image identity','image change','running speed quintile','pupil diameter quintile','trial outcome'],
          'output_values':[image_values,['no change','image change'],
                           ['0-20%','20-40%','40-60%','60-80%','80-100%'],
                           ['0-20%','20-40%','40-60%','60-80%','80-100%'],
                           ['hit','miss','false alarm','correct reject']],
          'metadata':{'task_description':'Active visual image change-detection task; decode stimulus, behavior, and Go/Catch outcome from detected calcium events.',
                      'time_bin_size':DT*1000,'temporal_alignment_event':'scheduled image change time (Go change or Catch sham-change)',
                      'off_start':OFF0,'off_end':OFF1,'neural_signal':'Allen event_detection calcium events, mean per 100-ms bin',
                      'trial_filter':'Go and Catch; aborted and auto-rewarded excluded',
                      'behavior_binning':'within-session quintiles; pupil diameter is tracked pupil width',
                      'session_info':session_info}}
    with open(OUT,'wb') as fp: pickle.dump(data,fp,pickle.HIGHEST_PROTOCOL)
    print('saved',OUT,'sessions',len(neural),'trials',sum(map(len,neural)))
if __name__=='__main__': main()
