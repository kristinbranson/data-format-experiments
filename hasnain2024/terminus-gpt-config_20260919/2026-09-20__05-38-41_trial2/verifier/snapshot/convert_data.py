#!/usr/bin/env python3
"""Convert the paper's two-context ALM ephys data to decoder format.

Usage: python -u /app/convert_data.py OUT.pkl [--full|--sample] [--show-processing]
"""
import argparse, pickle, time, warnings
from pathlib import Path
import h5py, numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d

ROOT=Path('/app/data/Ephys_Behavior')
TMIN,TMAX,DT=-3.0,2.5,0.01
EDGES=np.arange(TMIN,TMAX+DT/2,DT); CENTERS=(EDGES[:-1]+EDGES[1:])/2
VALID_QUAL={'poor','fair','good','great','excellent','multi'}
# Exact active records from Figure 8 metadata loaders; MATLAB probe indices are 1-based.
SESSIONS=[
 ('JEB6','2021-04-18',2),('JEB7','2021-04-29',1),('JEB7','2021-04-30',1),
 ('EKH1','2021-08-07',2),('EKH3','2021-08-11',2),
 ('JGR2','2021-11-16',1),('JGR2','2021-11-17',1),('JGR3','2021-11-18',1),
 ('JEB19','2023-04-21',1),('JEB19','2023-04-20',1),
 ('JEB19','2023-04-19',1),('JEB19','2023-04-18',1)]

def h5arr(f, obj):
    """Read dataset or MATLAB object reference, squeeze MATLAB singleton axes."""
    if isinstance(obj,h5py.Reference): obj=f[obj]
    return np.asarray(obj).squeeze()

def h5str(f, obj):
    a=h5arr(f,obj)
    if a.dtype.kind in 'ui': return ''.join(chr(int(z)) for z in np.asarray(a).ravel(order='F')).rstrip('\x00')
    return str(a.item() if a.ndim==0 else a)

def cell_arrays(f, ds):
    return [h5arr(f,r) for r in np.asarray(ds).ravel()]

def feat_names(f, ref):
    cell=f[ref]
    return [h5str(f,r) for r in np.asarray(cell).ravel()]

def causal_smooth(x, n=15):
    """Exact translation of reference mySmooth(x,15): reflect-pad and causal gausswin."""
    # MATLAB gausswin(N) default alpha=2.5: exp(-0.5*(alpha*nidx)^2),
    # nidx linearly spans -1..1. Reference zeros indices 1:floor(N/2).
    idx=np.linspace(-1.0,1.0,int(n),dtype=float)
    k=np.exp(-0.5*(2.5*idx)**2)
    k[:len(k)//2]=0.0; k/=k.sum()
    out=np.empty_like(x,dtype=np.float32)
    for i,row in enumerate(x):
        # mySmooth default bctype='reflect': prepend flipud(x(1:N)).
        padded=np.concatenate([row[:n][::-1],row])
        y=np.convolve(padded,k,mode='same')
        out[i]=y[n:]
    return out

def behavior(f):
    bp=f['obj/bp']; n=np.asarray(bp['L']).size
    get=lambda k:np.asarray(bp[k]).squeeze()
    ev=bp['ev']; trials=f['obj/trials/bp']
    d={k:get(k) for k in ['L','R','hit','miss','no','early','autowater']}
    d['stim']=np.asarray(bp['stim/enable']).squeeze() if 'stim/enable' in bp else np.zeros(n)
    d['go']=np.asarray(ev['goCue']).squeeze()
    d['haveEphys']=np.asarray(trials['haveEphys']).squeeze().astype(bool)
    d['haveVid']=np.asarray(trials['haveVid']).squeeze().astype(bool)
    d['lickL']=cell_arrays(f,ev['lickL']); d['lickR']=cell_arrays(f,ev['lickR'])
    return d

def first_post_go(a,go):
    x=np.atleast_1d(a).astype(float); x=x[np.isfinite(x)&(x>=go)]
    return np.min(x) if x.size else np.inf

def load_clusters(f, probe, go, trial_keep):
    pref=np.asarray(f['obj/clu']).ravel()[probe-1]; g=f[pref]
    if not isinstance(g,h5py.Group): raise ValueError('selected probe is empty')
    mats=[]; qualities=[]
    for rt,ri,rq in zip(np.asarray(g['trialtm']).ravel(),np.asarray(g['trial']).ravel(),np.asarray(g['quality']).ravel()):
        q=h5str(f,rq).lower().strip()
        if q not in VALID_QUAL: continue
        tm=np.atleast_1d(h5arr(f,rt)).astype(float); tr=np.atleast_1d(h5arr(f,ri)).astype(int)-1
        mat=np.zeros((trial_keep.size,CENTERS.size),np.float32)
        pos={int(t):i for i,t in enumerate(trial_keep)}
        for old in np.unique(tr):
            if old not in pos or old<0 or old>=len(go): continue
            z=tm[tr==old]-go[old]
            mat[pos[old]]=np.histogram(z,EDGES)[0]/DT
        mats.append(mat); qualities.append(q)
    if not mats: raise ValueError('no curated clusters')
    raw=np.stack(mats,axis=1) # retained decoder trials, neurons, time
    sm=np.stack([causal_smooth(x) for x in raw],axis=0)
    # Reference low-FR curation uses condition 1 (all hit|miss|no trials),
    # computes a trial-averaged PSTH, applies mySmooth, then averages over time.
    rates=[]
    for rt,ri,rq in zip(np.asarray(g['trialtm']).ravel(),np.asarray(g['trial']).ravel(),np.asarray(g['quality']).ravel()):
        q=h5str(f,rq).lower().strip()
        if q not in VALID_QUAL: continue
        tm=np.atleast_1d(h5arr(f,rt)).astype(float); tr=np.atleast_1d(h5arr(f,ri)).astype(int)-1
        good=(tr>=0)&(tr<len(go)); aligned=tm[good]-go[tr[good]]
        psth=np.histogram(aligned,EDGES)[0].astype(np.float32)/(len(go)*DT)
        rates.append(float(np.mean(causal_smooth(psth[None,:])[0])))
    use=np.asarray(rates)>1.0
    return sm[:,use,:],np.asarray(qualities)[use],use

def nearest_bin(times, values):
    """Nearest-frame resampling, unavailable outside source support or at nonfinite values."""
    t=np.asarray(times,float).ravel(); v=np.asarray(values,float).ravel()
    ok=np.isfinite(t); t=t[ok]; v=v[ok]
    out=np.full(CENTERS.size,np.nan)
    if t.size<2:return out
    order=np.argsort(t);t=t[order];v=v[order]
    ix=np.searchsorted(t,CENTERS); ix=np.clip(ix,1,len(t)-1)
    lo=ix-1; pick=np.where(np.abs(t[ix]-CENTERS)<np.abs(t[lo]-CENTERS),ix,lo)
    inside=(CENTERS>=t[0])&(CENTERS<=t[-1]); out[inside]=v[pick[inside]]
    return out

def video_continuous(f, trial_ids, go):
    """Return tongue speed, paw speed and camera frame times per retained trial."""
    cams=np.asarray(f['obj/traj']).ravel(); results=[]
    decoded=[]
    for cref in cams:
        g=f[cref]; decoded.append((g, np.asarray(g['ts']).ravel(),np.asarray(g['frameTimes']).ravel(),np.asarray(g['featNames']).ravel()))
    for old in trial_ids:
        vals=[]
        for ci,(g,tsrefs,ftrefs,nrefs) in enumerate(decoded):
            arr=np.asarray(f[tsrefs[old]],float) # feature, xyz, frame
            ft=np.asarray(f[ftrefs[old]],float).squeeze()-go[old]
            names=[x.lower() for x in feat_names(f,nrefs[old])]
            vals.append((arr,ft,names))
        def speed_for(cam,patterns):
            arr,ft,names=vals[cam]
            inds=[i for i,n in enumerate(names) if any(p in n for p in patterns)]
            if not inds:return np.full(CENTERS.size,np.nan)
            speeds=[]
            for j in inds:
                x,y=arr[j,0],arr[j,1]
                visible=np.isfinite(x)&np.isfinite(y)
                if arr.shape[1]>2: visible &= np.isfinite(arr[j,2])&(arr[j,2]>0.9)
                dx=np.gradient(x,ft);dy=np.gradient(y,ft); sp=np.hypot(dx,dy);sp[~visible]=np.nan
                speeds.append(nearest_bin(ft,sp))
            z=np.stack(speeds); finite=np.isfinite(z); count=finite.sum(axis=0)
            return np.divide(np.nansum(z,axis=0),count,out=np.full(z.shape[1],np.nan),where=count>0)
        tongue=speed_for(0,['tongue'])
        paw=speed_for(1,['paw'])
        results.append((tongue,paw,vals[0][1]))
    return results

def motion_continuous(subject,date,trial_ids,go,frame_times):
    p=ROOT/f'motionEnergy_{subject}_{date}.mat'; out=[]
    if not p.exists(): return [np.full(CENTERS.size,np.nan) for _ in trial_ids]
    me=loadmat(p,simplify_cells=True)['me']; data=np.atleast_1d(me['data'])
    for old,ft in zip(trial_ids,frame_times):
        if old>=len(data):out.append(np.full(CENTERS.size,np.nan));continue
        v=np.asarray(data[old],float).squeeze(); t=np.asarray(ft,float).squeeze()
        n=min(len(v),len(t)); out.append(nearest_bin(t[:n],v[:n]))
    return out

def discretize_session(a):
    a=np.asarray(a,float); valid=np.isfinite(a); med=float(np.nanpercentile(a,50)) if valid.any() else np.nan
    y=np.full(a.shape,2,np.int64); y[valid & (a<med)]=0; y[valid & (a>=med)]=1
    return y,med

def process_session(subject,date,probe,show=False):
    t0=time.time(); p=ROOT/f'data_structure_{subject}_{date}.mat'
    with h5py.File(p,'r') as f:
        b=behavior(f); n=len(b['go'])
        valid=b['haveEphys']&np.isfinite(b['go'])&(np.asarray(b['early'])==0)&(np.asarray(b['stim'])==0)
        outcome_sum=np.asarray(b['hit'])+np.asarray(b['miss'])+np.asarray(b['no'])
        valid &= outcome_sum==1
        ids=np.flatnonzero(valid)
        neural,qualities,_=load_clusters(f,probe,b['go'],ids)
        vid=video_continuous(f,ids,b['go'])
        tongue=np.stack([z[0] for z in vid]);paw=np.stack([z[1] for z in vid]);fts=[z[2] for z in vid]
        motion=np.stack(motion_continuous(subject,date,ids,b['go'],fts))
        # Trial scalar labels.
        lick=[];context=[];outcome=[]
        for old in ids:
            l=first_post_go(b['lickL'][old],b['go'][old]);r=first_post_go(b['lickR'][old],b['go'][old])
            lick.append(0 if l<r else (1 if r<l else 2))
            context.append(1 if b['autowater'][old] else 0)
            outcome.append(0 if b['miss'][old] else (1 if b['hit'][old] else 2))
    tv,tmed=discretize_session(tongue);pv,pmed=discretize_session(paw);mv,mmed=discretize_session(motion)
    outputs=[];inputs=[];neur=[]
    for i in range(len(ids)):
        o=np.vstack([np.full(CENTERS.size,lick[i]),np.full(CENTERS.size,context[i]),np.full(CENTERS.size,outcome[i]),tv[i],pv[i],mv[i]]).astype(np.int64)
        neur.append(neural[i].astype(np.float32));inputs.append(CENTERS[None,:].astype(np.float32));outputs.append(o)
    info={'id':f'{subject}_{date}','source_trials':n,'trials':len(ids),'neurons':neural.shape[1],
          'probe':probe,'quality_counts':{q:int(np.sum(qualities==q)) for q in np.unique(qualities)},
          'thresholds':{'tongue_velocity':tmed,'paw_velocity':pmed,'motion_energy':mmed},'seconds':time.time()-t0}
    if show:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,ax=plt.subplots(4,1,figsize=(11,9),sharex=True)
        ax[0].imshow(neural[0],aspect='auto',extent=[TMIN,TMAX,neural.shape[1],0]);ax[0].set_ylabel('neurons')
        ax[1].plot(CENTERS,tongue[0],label='tongue');ax[1].plot(CENTERS,paw[0],label='paw');ax[1].legend()
        ax[2].plot(CENTERS,motion[0]);ax[2].set_ylabel('motion energy')
        ax[3].step(CENTERS,outputs[0][3:].T);ax[3].set_ylabel('classes');ax[3].set_xlabel('s from go cue')
        fig.suptitle(info['id']);fig.tight_layout();fig.savefig(f'/app/processing_{subject}_{date}.png',dpi=140);plt.close(fig)
    print(f"{info['id']}: {info['trials']}/{n} trials, {info['neurons']} neurons, {info['seconds']:.1f}s",flush=True)
    return neur,inputs,outputs,info

def main():
    ap=argparse.ArgumentParser();ap.add_argument('outpicklefile');g=ap.add_mutually_exclusive_group();g.add_argument('--full',action='store_true');g.add_argument('--sample',action='store_true');ap.add_argument('--show-processing',action='store_true');a=ap.parse_args()
    sessions=SESSIONS[:2] if a.sample else SESSIONS
    neural=[];inputs=[];outputs=[];infos=[];subjects=[];subject_idx=[];regions=[]
    for si,(sub,date,probe) in enumerate(sessions):
        n,x,y,info=process_session(sub,date,probe,a.show_processing and si<2)
        if len(n)<2 or n[0].shape[0]<10: raise ValueError(f'insufficient data {sub}_{date}')
        if sub not in subjects:subjects.append(sub)
        neural.append(n);inputs.append(x);outputs.append(y);infos.append(info);subject_idx.append(subjects.index(sub));regions.append(np.zeros(n[0].shape[0],np.int64))
    data={'neural':neural,'input':inputs,'output':outputs,'subjects':subjects,'subject_idx':np.asarray(subject_idx,np.int64),
          'brain_regions':['ALM'],'brain_region_idx':regions,'input_names':['time_from_go_cue_s'],
          'output_names':['lick_direction','behavioral_context','outcome','tongue_velocity','paw_velocity','motion_energy'],
          'output_values':[['left','right','none'],['DR','WC'],['incorrect','correct','ignore'],
             ['below_session_median','at_or_above_session_median','not_visible'],
             ['below_session_median','at_or_above_session_median','not_visible'],
             ['below_session_median','at_or_above_session_median','no_video']],
          'metadata':{'task_description':'Two-context delayed-response (DR) and water-cued (WC) licking; decode behavior from ALM activity.',
             'time_bin_size':10.0,'temporal_alignment_event':'go cue onset (water drop in WC)','off_start':TMIN,'off_end':TMAX,
             'neural_representation':'spike firing rate (Hz), causal Gaussian smoothed','session_info':infos,
             'curation':'Figure 8 ALM probes; curated quality labels; mean FR >1 Hz; early/stim trials excluded; ignore retained for requested class.'}}
    with open(a.outpicklefile,'wb') as f:pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
    print(f'Wrote {a.outpicklefile}: {len(neural)} sessions, {sum(map(len,neural))} trials, {sum(x[0].shape[0] for x in neural)} neurons',flush=True)
if __name__=='__main__':main()
