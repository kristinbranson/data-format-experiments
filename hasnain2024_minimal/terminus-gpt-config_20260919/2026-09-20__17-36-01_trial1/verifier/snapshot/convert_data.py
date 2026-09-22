#!/usr/bin/env python3
"""Convert the paper's curated two-context ephys sessions for decoder use.

Decisions follow Figure8a_thru_c.m and paper methods: its 12 active sessions and
probe selections, all unit qualities but session firing rate >1 Hz, go-cue
alignment, [-3, 2.5) s at 10 ms, and the causal 15-bin Gaussian smoothing used
by getPSTHs.m. Hit/miss/no trials are retained (condition 1 in that script).
DLC x/y/likelihood and frameTimes are aligned as in getKinematicsFromVideo.
The release lacks raw videos / separate motion-energy files, so motion energy
is represented by aggregate visible-landmark image motion; missing video is 2.
"""
import os, pickle, h5py
import numpy as np
from scipy.signal.windows import gaussian
from scipy.ndimage import convolve1d

ROOT='/app/data/Ephys_Behavior'
SESSIONS=[
 ('JEB6','2021-04-18',2),('JEB7','2021-04-29',1),('JEB7','2021-04-30',1),
 ('EKH1','2021-08-07',2),('EKH3','2021-08-11',2),
 ('JGR2','2021-11-16',1),('JGR2','2021-11-17',1),('JGR3','2021-11-18',1),
 ('JEB19','2023-04-21',1),('JEB19','2023-04-20',1),
 ('JEB19','2023-04-19',1),('JEB19','2023-04-18',1)]
DT=.01; TMIN=-3.; TMAX=2.5
TIME=np.arange(TMIN,TMAX,DT); EDGES=np.arange(TMIN,TMAX+DT/2,DT)


def refs(a): return np.asarray(a).ravel(order='F')
def arr(f, ref): return np.asarray(f[ref]).squeeze()
def matlab_text(a): return ''.join(chr(int(x)) for x in np.asarray(a).ravel(order='F') if x)

def causal_smooth(x, n=15):
    # MATLAB gausswin(15), first floor(15/2) entries zeroed, conv(...,'same').
    k=gaussian(n, std=(n-1)/2/2.5, sym=True)
    k[:n//2]=0; k/=k.sum()
    return convolve1d(x, k, axis=-1, mode='constant', origin=0)

def interp_trace(old_t, val):
    good=np.isfinite(old_t)&np.isfinite(val)
    out=np.full(TIME.size,np.nan,dtype=np.float32)
    if good.sum()>=2:
        # np.interp edge values match MATLAB's subsequent fillmissing nearest.
        out[:]=np.interp(TIME,old_t[good],val[good]).astype(np.float32)
        out[TIME < old_t[good].min()]=np.nan; out[TIME > old_t[good].max()]=np.nan
    return out

def camera_trial(f,traj,ti):
    try:
        ts=arr(f,refs(traj['ts'])[ti]).astype(float)
        ft=arr(f,refs(traj['frameTimes'])[ti]).astype(float)
    except Exception:return None,None,None
    if ts.ndim!=3:return None,None,None
    # HDF5 dimensions are feature, coordinate(x,y,likelihood), frame.
    if ts.shape[1]!=3:return None,None,None
    names_obj=f[refs(traj['featNames'])[ti]]
    names=[matlab_text(f[r]) for r in refs(names_obj)]
    return ts,names,np.ravel(ft)

def landmark_speed(ts, indices):
    if not indices:return None,None
    xy=ts[indices,:2,:]
    lk=ts[indices,2,:]
    vis=lk>=0.9
    xy=np.where(vis[:,None,:],xy,np.nan)
    cx=np.nanmean(xy[:,0,:],axis=0); cy=np.nanmean(xy[:,1,:],axis=0)
    visible=np.any(vis,axis=0)
    speed=np.r_[np.nan, np.hypot(np.diff(cx),np.diff(cy))*400.]
    speed[~visible]=np.nan
    return speed,visible

def session_video(f, tids, go):
    trajrefs=refs(f['obj/traj']); traj=[f[r] for r in trajrefs]
    tongue=[]; paw=[]; motion=[]
    for ti in tids:
        tv=np.full(TIME.size,np.nan,np.float32); pv=tv.copy(); components=[]
        for cami,tr in enumerate(traj):
            ts,names,ft=camera_trial(f,tr,ti)
            if ts is None:continue
            old=ft-0.5-go[ti] # exact offset used by paper loadMotionEnergy/getPosition
            tong=[j for j,n in enumerate(names) if 'tongue' in n]
            paws=[j for j,n in enumerate(names) if 'paw' in n]
            s,_=landmark_speed(ts,tong)
            if s is not None:
                q=interp_trace(old,s); components.append(q)
                if cami==0 or np.all(~np.isfinite(tv)): tv=q
            s,_=landmark_speed(ts,paws)
            if s is not None:
                pv=interp_trace(old,s); components.append(pv)
            # Aggregate all confidently visible landmarks as image-motion proxy.
            s,_=landmark_speed(ts,list(range(len(names))))
            if s is not None: components.append(interp_trace(old,s))
        if components:
            with np.errstate(invalid='ignore'): me=np.nanmean(np.stack(components),axis=0).astype(np.float32)
        else: me=np.full(TIME.size,np.nan,np.float32)
        tongue.append(tv); paw.append(pv); motion.append(me)
    return [np.stack(tongue),np.stack(paw),np.stack(motion)]

def discretize_session(stream):
    finite=np.isfinite(stream)
    threshold=float(np.nanpercentile(stream,50)) if finite.any() else np.nan
    out=np.full(stream.shape,2,dtype=np.int16)
    out[finite & (stream<threshold)]=0; out[finite & (stream>=threshold)]=1
    return out,threshold

def convert_session(animal,date,probe):
    path=f'{ROOT}/data_structure_{animal}_{date}.mat'
    with h5py.File(path,'r') as f:
        bp=f['obj/bp']; n=int(np.asarray(bp['Ntrials']).squeeze())
        label={k:np.asarray(bp[k]).ravel().astype(bool) for k in ['R','L','hit','miss','no','early','autowater']}
        go=np.asarray(bp['ev/goCue']).ravel().astype(float)
        # Figure 8 condition 1: hit|miss|no. Keep early trials because that exact
        # all-trial condition does not exclude them and outcome decoding needs no.
        keep=(label['hit']|label['miss']|label['no']) & np.isfinite(go)
        tids=np.flatnonzero(keep)
        clu=f[refs(f['obj/clu'])[probe-1]]
        trial_cells=refs(clu['trial']); tm_cells=refs(clu['trialtm'])
        alltm=refs(clu['tm'])
        # removeLowFRClusters calls getFiringRate on trial-aligned data: mean
        # over every time bin in [-3,2.5) and every source trial. Reproduce
        # that denominator rather than including inter-trial wall-clock time.
        good=[]
        for ui in range(len(alltm)):
            tr_all=arr(f,trial_cells[ui]).astype(int)-1
            rel_all=arr(f,tm_cells[ui]).astype(float)
            aligned=rel_all-go[tr_all]
            nwin=np.count_nonzero((aligned>=TMIN)&(aligned<TMAX))
            rate=nwin/(n*(TMAX-TMIN))
            if rate>1.: good.append(ui)
        neural_trials=[np.zeros((len(good),TIME.size),np.float32) for _ in tids]
        tid_to_out={int(t):j for j,t in enumerate(tids)}
        for ni,ui in enumerate(good):
            tr=arr(f,trial_cells[ui]).astype(int)-1; rel=arr(f,tm_cells[ui]).astype(float)
            for ti,oj in tid_to_out.items():
                sp=rel[tr==ti]-go[ti]
                counts=np.histogram(sp,bins=EDGES)[0].astype(np.float32)/DT
                neural_trials[oj][ni]=causal_smooth(counts)
        rawvid=session_video(f,tids,go)
        disc=[]; thresholds=[]
        for x in rawvid:
            d,th=discretize_session(x); disc.append(d); thresholds.append(th)
        inputs=[]; outputs=[]
        time_row=TIME.astype(np.float32)[None,:]
        for j,ti in enumerate(tids):
            inputs.append(time_row.copy())
            # Actual lick direction: correct follows instructed side; incorrect is
            # opposite; ignored/no-response has no lick. left=0,right=1,none=2.
            if label['no'][ti]: lick=2
            elif label['hit'][ti]: lick=0 if label['L'][ti] else 1
            else: lick=1 if label['L'][ti] else 0
            context=1 if label['autowater'][ti] else 0 # DR=0, WC=1
            outcome=2 if label['no'][ti] else (1 if label['hit'][ti] else 0)
            const=np.array([lick,context,outcome],dtype=np.int16)[:,None]
            const=np.repeat(const,TIME.size,axis=1)
            outputs.append(np.vstack([const,disc[0][j],disc[1][j],disc[2][j]]).astype(np.int16))
        info={'animal':animal,'date':date,'probe':probe,'source_file':os.path.basename(path),
              'n_source_trials':n,'source_trial_indices':tids.tolist(),
              'n_units_probe':len(alltm),'n_units_rate_gt_1Hz':len(good),
              'video_median_thresholds':{'tongue_velocity':thresholds[0],
                 'paw_velocity':thresholds[1],'motion_energy_proxy':thresholds[2]}}
        print(animal,date,'trials',len(tids),'units',len(good),flush=True)
        return neural_trials,inputs,outputs,info

def main():
    neural=[]; inp=[]; out=[]; infos=[]
    for s in SESSIONS:
        a,b,c,i=convert_session(*s); neural.append(a); inp.append(b); out.append(c); infos.append(i)
    subjects=[]; subject_idx=[]
    for animal,_,_ in SESSIONS:
        if animal not in subjects:subjects.append(animal)
        subject_idx.append(subjects.index(animal))
    data={'neural':neural,'input':inp,'output':out,
      'subjects':subjects,'subject_idx':np.asarray(subject_idx,dtype=np.int64),
      'brain_regions':['ALM'],'brain_region_idx':[np.zeros(x[0].shape[0],dtype=np.int64) for x in neural],
      'input_names':['time from go cue onset'],
      'output_names':['lick direction','behavioral context','outcome','tongue velocity','paw velocity','motion energy'],
      'output_values':[['left','right','none'],['DR','WC'],['incorrect','correct','ignore'],
                       ['below session median','at or above session median','not visible'],
                       ['below session median','at or above session median','not visible'],
                       ['below session median','at or above session median','no video']],
      'metadata':{'task_description':'Two-context delayed response (DR) and water-cued/autowater (WC) licking task.',
       'time_bin_size':10.0,'temporal_alignment_event':'go cue onset','off_start':-3.0,'off_end':2.5,
       'neural_processing':'10-ms firing rates, causal 15-bin Gaussian smoothing; all qualities, >1 Hz session rate',
       'video_processing':'DLC likelihood >=0.9; velocity from framewise x/y displacement; session-median discretization. Motion energy is an aggregate landmark-motion proxy because released files contain no raw video/pixel-motion stream.',
       'session_info':infos}}
    with open('/app/converted_data.pkl','wb') as fh:pickle.dump(data,fh,pickle.HIGHEST_PROTOCOL)
if __name__=='__main__':main()
