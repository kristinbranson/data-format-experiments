#!/usr/bin/env python3
"""Convert Steinmetz-lab released data for the WC/DR neural decoder.

Decisions follow the released Figure 8 pipeline: its exact 12 two-context
sessions and selected probe, go-cue alignment, 5-ms bins over [-2.5,2.5],
all curated units followed by >1 Hz filtering, and the causal 15-bin Gaussian
smoother. Unlike correct-trial paper panels, all non-stim, non-early trials are
retained because outcome is a requested decoder target. Video missingness is
preserved as an explicit category rather than filled as in plotting analyses.
"""
import os, pickle, warnings
import numpy as np
import mat73
from scipy.signal.windows import gaussian
from scipy.ndimage import convolve1d

DATA='/app/data/Ephys_Behavior'
DT=.005; TMIN=-2.5; TMAX=2.5
# MATLAB colon -2.5:0.005:2.5 gives 1001 sample centers. getSeq histogram
# uses edges centered on these samples; output therefore has 1001 points.
TIME=np.arange(TMIN,TMAX+DT/2,DT)
SESS=[
 ('JEB6','2021-04-18',2),('JEB7','2021-04-29',1),('JEB7','2021-04-30',1),
 ('EKH1','2021-08-07',2),('EKH3','2021-08-11',2),
 ('JGR2','2021-11-16',1),('JGR2','2021-11-17',1),('JGR3','2021-11-18',1),
 ('JEB19','2023-04-21',1),('JEB19','2023-04-20',1),
 ('JEB19','2023-04-19',1),('JEB19','2023-04-18',1)]

def arr(x): return np.asarray(x, dtype=float).reshape(-1)
def trial_item(x,i):
    if isinstance(x,list): return x[i]
    a=np.asarray(x, dtype=object)
    return a.reshape(-1)[i]
def names_at(traj,i):
    z=trial_item(traj['featNames'],i)
    if not isinstance(z,list): z=np.asarray(z,dtype=object).reshape(-1).tolist()
    return [str(q[0] if isinstance(q,list) else q) for q in z]

def matlab_smooth(x):
    # mySmooth: gausswin(15), first floor(15/2) entries zero, normalize,
    # conv(...,'same'). scipy convolve1d needs the equivalent correlation origin.
    k=gaussian(15,std=(15-1)/6, sym=True) # MATLAB gausswin default alpha=2.5: corrected below
    # Exact MATLAB gausswin(N): exp(-.5*(2.5*n/((N-1)/2))^2)
    n=np.arange(15)-(14/2); k=np.exp(-.5*(2.5*n/(14/2))**2)
    k[:7]=0; k/=k.sum()
    # np.convolve independently is unambiguous and matches MATLAB conv same.
    return np.stack([np.convolve(row,k,'same') for row in x],axis=0)

def interp_stream(ft,val):
    ft=arr(ft); val=arr(val)
    ok=np.isfinite(ft)&np.isfinite(val)
    if ok.sum()<2:return np.full(TIME.size,np.nan)
    order=np.argsort(ft[ok]); xx=ft[ok][order]; yy=val[ok][order]
    return np.interp(TIME,xx,yy,left=np.nan,right=np.nan)

def video_trial(o,i,go):
    """Return tongue speed, paw speed, ME on the common go-aligned grid."""
    out=[]
    # tongue: side camera landmark 'tongue'; paw: mean top/bottom paw in bottom camera.
    for cam, feats in [(0,['tongue']),(1,['top_paw','bottom_paw'])]:
        tr=o['traj'][cam]; ts=np.asarray(trial_item(tr['ts'],i),float)
        ft=arr(trial_item(tr['frameTimes'],i))-go
        nm=names_at(tr,i)
        coords=[]
        for feat in feats:
            if feat not in nm: continue
            j=nm.index(feat); xy=ts[:,0:2,j].copy(); lk=ts[:,2,j]
            # DLC convention used by preprocessing: low-confidence labels are missing.
            xy[lk<.9]=np.nan; coords.append(xy)
        if not coords: out.append(np.full(TIME.size,np.nan)); continue
        xy=np.nanmean(np.stack(coords),axis=0)
        # Velocity magnitude in pixels/frame, matching findVelocity's gradient of
        # interpolated coordinates. Interpolate positions first onto video frames;
        # preserve visibility separately through NaNs.
        visible=np.all(np.isfinite(xy),axis=1)
        vx=np.gradient(xy[:,0]); vy=np.gradient(xy[:,1])
        speed=np.hypot(vx,vy); speed[~visible]=np.nan
        out.append(interp_stream(ft,speed))
    meobj=o.get('me',{})
    me=meobj.get('data',[]) if isinstance(meobj,dict) else (meobj if isinstance(meobj,list) else [])
    try:
        m=trial_item(me,i); ft=arr(trial_item(o['traj'][0]['frameTimes'],i))-go
        while isinstance(m,list) and len(m)==1: m=m[0]
        out.append(interp_stream(ft,m))
    except Exception: out.append(np.full(TIME.size,np.nan))
    return out

def process(path,probe):
    print('Loading',os.path.basename(path),flush=True)
    o=mat73.loadmat(path)['obj']; b=o['bp']; n=int(float(np.asarray(b['Ntrials'])))
    go=arr(b['ev']['goCue']); stim=arr(b['stim']['enable']); early=arr(b['early'])
    valid=np.flatnonzero(np.isfinite(go)&(stim==0)&(early==0))
    # selected MATLAB probe numbers are 1-based
    clu=o['clu'][probe-1] if isinstance(o['clu'],list) else o['clu']
    quals=clu['quality']; keep0=[]
    for u,q in enumerate(quals):
        q=str(q).lower()
        # findClusters quality='all' means all non-garbage clusters.
        if q not in ('garbage','noise','nan','none',''): keep0.append(u)
    edges=np.r_[TIME-DT/2,TIME[-1]+DT/2]
    neural=np.zeros((len(valid),len(keep0),TIME.size),np.float32)
    for jj,u in enumerate(keep0):
        st=arr(clu['trialtm'][u]); tri=arr(clu['trial'][u]).astype(int)-1
        for ii,t in enumerate(valid):
            z=st[tri==t]-go[t]
            neural[ii,jj]=matlab_smooth((np.histogram(z,edges)[0]/DT)[None,:])[0]
    # Paper's general-analysis criterion: average firing rate strictly >1 Hz.
    unitkeep=np.nanmean(neural,axis=(0,2))>1
    neural=neural[:,unitkeep,:]
    inp=[]; outs=[]; continuous=[]
    L=arr(b['L']); R=arr(b['R']); aw=arr(b['autowater'])
    hit=arr(b['hit']); miss=arr(b['miss']); no=arr(b['no'])
    for t in valid:
        inp.append(TIME[None,:].astype(np.float32))
        outcome=1 if hit[t]>0 else (0 if miss[t]>0 else 2)
        if outcome==2:
            lick=2
        elif hit[t]>0:
            lick=0 if L[t]>0 else 1
        else:
            lick=1 if L[t]>0 else 0
        v=video_trial(o,t,go[t]); continuous.append(v)
        outs.append([lick,int(not (aw[t]>0)),outcome])
    # Per-session medians over all visible samples, as requested.
    thresholds=[]
    for k in range(3):
        z=np.concatenate([x[k][np.isfinite(x[k])] for x in continuous])
        thresholds.append(float(np.percentile(z,50)) if z.size else np.nan)
    output=[]
    for base,v in zip(outs,continuous):
        rows=[np.full(TIME.size,x,dtype=np.int16) for x in base]
        for k,z in enumerate(v):
            y=np.full(TIME.size,2,dtype=np.int16); ok=np.isfinite(z)
            y[ok]=(z[ok]>=thresholds[k]).astype(np.int16); rows.append(y)
        output.append(np.stack(rows))
    print(' ',n,'raw trials,',len(valid),'retained,',len(keep0),'curated,',neural.shape[1],'>1Hz',flush=True)
    return [x for x in neural],inp,output,thresholds

def main():
    D={'neural':[],'input':[],'output':[],'subjects':[], 'subject_idx':[],
       'brain_regions':['ALM'],'brain_region_idx':[],
       'input_names':['time from go cue onset'],
       'output_names':['lick direction','behavioral context','outcome','tongue velocity','paw velocity','motion energy'],
       'output_values':[['left','right','none'],['WC','DR'],['incorrect','correct','ignore'],
                        ['below session median','at or above session median','not visible'],
                        ['below session median','at or above session median','not visible'],
                        ['below session median','at or above session median','no video']],
       'metadata':{'task_description':'Alternating delayed-response (DR) and water-cued (WC) licking task; decode trial direction, context, outcome, and movement from ALM activity.',
        'time_bin_size':5.0,'temporal_alignment_event':'go cue onset (water delivery in WC trials)',
        'off_start':TMIN,'off_end':TMAX,'neural_measure':'single-trial firing rate (Hz), causal Gaussian smoothed',
        'neural_filter':'released Figure 8 selected probe; non-garbage curated units with mean firing rate >1 Hz',
        'trial_filter':'no photostimulation, no early lick; includes correct, incorrect and ignore trials',
        'session_info':[]}}
    for anm,date,probe in SESS:
        p=f'{DATA}/data_structure_{anm}_{date}.mat'
        neu,inp,out,thr=process(p,probe)
        if len(neu)<2 or not neu or neu[0].shape[0]==0:
            warnings.warn('dropping unusable session '+p); continue
        if anm not in D['subjects']:D['subjects'].append(anm)
        D['subject_idx'].append(D['subjects'].index(anm)); D['neural'].append(neu)
        D['input'].append(inp); D['output'].append(out)
        D['brain_region_idx'].append(np.zeros(neu[0].shape[0],dtype=np.int16))
        D['metadata']['session_info'].append({'subject':anm,'date':date,'probe':probe,
          'n_trials':len(neu),'n_neurons':neu[0].shape[0],
          'video_median_thresholds':dict(zip(['tongue_velocity','paw_velocity','motion_energy'],thr))})
    D['subject_idx']=np.asarray(D['subject_idx'],dtype=np.int16)
    with open('/app/converted_data.pkl','wb') as f:pickle.dump(D,f,pickle.HIGHEST_PROTOCOL)
    print('Saved /app/converted_data.pkl:',len(D['neural']),'sessions',sum(map(len,D['neural'])),'trials')
if __name__=='__main__':main()
