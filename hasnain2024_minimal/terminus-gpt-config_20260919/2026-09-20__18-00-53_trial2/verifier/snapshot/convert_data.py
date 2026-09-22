#!/usr/bin/env python3
"""Convert the released two-context ALM recordings to decoder format.

Decisions follow the paper repository: Figure-8 sessions only; ALM probe map from
load*ALMVideo.m; 5-ms bins from -2.5 to +2.5 s around go cue; 15-bin causal
Gaussian smoothing; units with mean rate >1 Hz. Early-lick trials are omitted as
in the paper, while no-response trials are retained because ignore is a required
class. WC is bp.autowater and DR is its complement, as in fig8 code.
"""
import os, glob, pickle
import numpy as np
import mat73
from scipy.io import loadmat
from scipy.signal.windows import gaussian
from scipy.ndimage import convolve1d

ROOT='/app/data/Ephys_Behavior'
DT=.005; T0=-2.5; T1=2.5
EDGES=np.arange(T0,T1+DT/2,DT); TIME=EDGES[:-1]+DT/2
# Figure 8's six mice; all their released sessions (11 files; paper reports 12).
MICE={'JEB6','JEB7','EKH3','JGR2','JGR3','JEB19'}
PROBES={
 ('EKH3','2021-08-11'):[2], ('JEB6','2021-04-18'):[2],
 ('JEB7','2021-04-29'):[1], ('JEB7','2021-04-30'):[1],
 ('JGR2','2021-11-16'):[1], ('JGR2','2021-11-17'):[1],
 ('JGR3','2021-11-18'):[1],
 ('JEB19','2023-04-18'):[1], ('JEB19','2023-04-19'):[1],
 ('JEB19','2023-04-20'):[1], ('JEB19','2023-04-21'):[1]}

def smooth_rates(x):
    # MATLAB mySmooth: gausswin(15), first floor(N/2) samples set to zero.
    k=gaussian(15, std=(15-1)/(2*2.5), sym=True); k[:7]=0; k/=k.sum()
    # np.convolve(x,k,'same') equals convolution with kernel as written.
    return np.apply_along_axis(lambda z:np.convolve(z,k,mode='same'),1,x)

def interp_nearest(t,y,target,fill=np.nan):
    t=np.asarray(t,float).ravel(); y=np.asarray(y,float).ravel()
    ok=np.isfinite(t)&np.isfinite(y)
    if ok.sum()<2:return np.full(target.shape,fill,float)
    t=t[ok]; y=y[ok]; ix=np.argsort(t); t=t[ix]; y=y[ix]
    return np.interp(target,t,y,left=fill,right=fill)

def trial_kin(traj, trial, go):
    # Side camera (cam 1): first four tongue landmarks, top/bottom paw landmarks.
    if len(traj)<2 or trial>=len(traj[1]['ts']):
        return np.full(TIME.size,np.nan),np.zeros(TIME.size,bool),np.full(TIME.size,np.nan),np.zeros(TIME.size,bool),None
    cam=traj[1]; ts=np.asarray(cam['ts'][trial],float); ft=np.asarray(cam['frameTimes'][trial],float).ravel()-go
    if ts.ndim!=3 or ts.shape[0]!=ft.size:return np.full(TIME.size,np.nan),np.zeros(TIME.size,bool),np.full(TIME.size,np.nan),np.zeros(TIME.size,bool),ft
    def speed(ids):
        ids=[i for i in ids if i<ts.shape[2]]
        xy=ts[:,0:2,ids]; lk=ts[:,2,ids] if ts.shape[1]>2 else np.ones((len(ft),len(ids)))
        vis=np.any(lk>=.9,axis=1)
        # centroid of visible landmarks; preserve invisibility rather than paper's zero fill
        xy=np.where((lk>=.9)[:,None,:],xy,np.nan)
        pos=np.nanmean(xy,axis=2)
        # MATLAB findVelocity uses gradient per frame (pixel/frame), not /dt.
        vx=np.gradient(pos[:,0]); vy=np.gradient(pos[:,1]); sp=np.hypot(vx,vy)
        sp[~vis]=np.nan
        return interp_nearest(ft,sp,TIME), interp_nearest(ft,vis.astype(float),TIME,0)>=.5
    tongue,tvis=speed([0,1,2,3]); paw,pvis=speed([4,5])
    return tongue,tvis,paw,pvis,ft

def main():
    files=[]
    for f in glob.glob(ROOT+'/data_structure_*.mat'):
        base=os.path.basename(f)[15:-4]; mouse,day=base.split('_',1)
        if (mouse,day) in PROBES: files.append((mouse,day,f))
    files.sort()
    neural=[]; inputs=[]; outputs=[]; region_idx=[]; session_info=[]
    subjects=sorted(MICE); subject_idx=[]
    for mouse,day,f in files:
        print('loading',mouse,day,flush=True); o=mat73.loadmat(f)['obj']; b=o['bp']; n=int(b['Ntrials'])
        early=np.asarray(b['early']).astype(bool); keep=np.flatnonzero(~early)
        go=np.asarray(b['ev']['goCue'],float).ravel()
        units=[]
        for p in PROBES[(mouse,day)]:
            c=o['clu'][p-1]
            for tr,tm in zip(c['trial'],c['trialtm']): units.append((np.asarray(tr,int).ravel()-1,np.asarray(tm,float).ravel()))
        # Build smoothed rates unit x trial x time directly from spike lists.
        rates=np.zeros((len(units),len(keep),TIME.size),np.float32)
        keepmap={int(t):j for j,t in enumerate(keep)}
        for ui,(tr,tm) in enumerate(units):
            for t in np.unique(tr):
                j=keepmap.get(int(t));
                if j is None or not np.isfinite(go[t]): continue
                h=np.histogram(tm[tr==t]-go[t],EDGES)[0].astype(float)/DT
                rates[ui,j]=smooth_rates(h[None,:])[0]
        use=rates.mean(axis=(1,2))>1.0; rates=rates[use]
        mef=os.path.join(ROOT,f'motionEnergy_{mouse}_{day}.mat'); me=None
        if os.path.exists(mef): me=loadmat(mef,squeeze_me=True,struct_as_record=False)['me'].data
        tongue=[];tvis=[];paw=[];pvis=[];mes=[];mvis=[]
        for t in keep:
            tv,tvok,pv,pvok,ft=trial_kin(o['traj'],int(t),go[t]); tongue.append(tv);tvis.append(tvok);paw.append(pv);pvis.append(pvok)
            if me is not None and ft is not None and t<len(me):
                mv=np.asarray(me[t],float).ravel(); m=min(len(mv),len(ft)); z=interp_nearest(ft[:m],mv[:m],TIME); mes.append(z);mvis.append(np.isfinite(z))
            else: mes.append(np.full(TIME.size,np.nan));mvis.append(np.zeros(TIME.size,bool))
        tongue=np.asarray(tongue);tvis=np.asarray(tvis);paw=np.asarray(paw);pvis=np.asarray(pvis);mes=np.asarray(mes);mvis=np.asarray(mvis)
        def classes(x,vis):
            th=np.nanpercentile(x[vis],50) if np.any(vis) else np.nan
            z=np.full(x.shape,2,np.int8); z[vis]=(x[vis]>=th).astype(np.int8); return z,float(th)
        tc,tth=classes(tongue,tvis); pc,pth=classes(paw,pvis); mc,mth=classes(mes,mvis)
        L=np.asarray(b['L']).astype(bool); R=np.asarray(b['R']).astype(bool); hit=np.asarray(b['hit']).astype(bool); miss=np.asarray(b['miss']).astype(bool); no=np.asarray(b['no']).astype(bool); aw=np.asarray(b['autowater']).astype(bool)
        ns=[];ins=[];outs=[]
        for j,t in enumerate(keep):
            ns.append(rates[:,j,:])
            ins.append(TIME[None,:].astype(np.float32))
            # Actual lick: instructed direction on hit, opposite on miss, none on no/ignore.
            lick=2 if no[t] or (not hit[t] and not miss[t]) else (0 if (hit[t] and L[t]) or (miss[t] and R[t]) else 1)
            outcome=2 if no[t] or (not hit[t] and not miss[t]) else (1 if hit[t] else 0)
            const=np.vstack([np.full(TIME.size,lick),np.full(TIME.size,int(not aw[t])),np.full(TIME.size,outcome)])
            outs.append(np.vstack([const,tc[j],pc[j],mc[j]]).astype(np.int8))
        neural.append(ns);inputs.append(ins);outputs.append(outs);region_idx.append(np.zeros(rates.shape[0],int));subject_idx.append(subjects.index(mouse))
        session_info.append({'subject':mouse,'date':day,'n_trials_raw':n,'n_trials':len(keep),'n_units_raw':len(units),'n_units':int(use.sum()),'tongue_median':tth,'paw_median':pth,'motion_energy_median':mth})
        del o,rates
    data={'neural':neural,'input':inputs,'output':outputs,'subjects':subjects,'subject_idx':np.asarray(subject_idx,int),'brain_regions':['ALM'],'brain_region_idx':region_idx,'input_names':['time from go cue'],'output_names':['lick direction','behavioral context','outcome','tongue velocity','paw velocity','motion energy'],'output_values':[['left','right','none'],['WC','DR'],['incorrect','correct','ignore'],['below session median','at or above session median','not visible'],['below session median','at or above session median','not visible'],['below session median','at or above session median','no video']], 'metadata':{'task_description':'Delayed-response and water-cued licking task; decode choice, context, outcome and movements from ALM activity.','time_bin_size':DT*1000,'temporal_alignment_event':'go cue onset (water delivery in WC trials)','off_start':T0,'off_end':T1,'neural_representation':'5 ms firing rates smoothed with repository 15-bin causal Gaussian; units with mean rate >1 Hz','trial_filter':'Figure-8 two-context sessions; early-lick trials excluded; ignore trials retained','session_info':session_info}}
    with open('/app/converted_data.pkl','wb') as q:pickle.dump(data,q,pickle.HIGHEST_PROTOCOL)
    print('saved',len(neural),'sessions',sum(map(len,neural)),'trials',sum(len(x) for x in region_idx),'units')
if __name__=='__main__':main()
