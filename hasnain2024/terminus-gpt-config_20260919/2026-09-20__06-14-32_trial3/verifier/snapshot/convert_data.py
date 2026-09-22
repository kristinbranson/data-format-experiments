#!/usr/bin/env python3
"""Convert Hasnain/Birnbaum ALM ephys and video data to decoder format."""
import argparse, pickle, re, time
from pathlib import Path
from collections import Counter
import numpy as np
import h5py
from scipy.io import loadmat

ROOT=Path('/app'); DATA=ROOT/'data'; CODE=ROOT/'code'
DT=0.005; TMIN=-2.5; TMAX=2.5
EDGES=np.arange(TMIN,TMAX+DT/2,DT,dtype=np.float64)
TIME=((EDGES[:-1]+EDGES[1:])/2).astype(np.float32)
BADQ={'garbage','gabrga'}

def parse_sessions():
    """Parse active author loader entries, including scalar or dual-probe choices."""
    rows=[]
    p=CODE/'DataLoadingScripts'/'Recording and video'
    for f in sorted(p.glob('*.m')):
        cur={}; active=False
        for raw in f.read_text(errors='ignore').splitlines():
            line=raw.strip()
            if not line or line.startswith('%'): continue
            if re.match(r'meta\(end\+1\)',line):
                if active and {'anm','date'}<=cur.keys(): rows.append((cur['anm'],cur['date'],cur.get('probe',[])))
                cur=cur.copy(); active=True
            for key in ('anm','date'):
                m=re.search(rf"meta\(end(?:\+1)?\)\.{key}\s*=\s*'([^']+)'",line)
                if m: cur[key]=m.group(1)
            m=re.search(r'meta\(end(?:\+1)?\)\.probe\s*=\s*(\[[^]]+\]|\d+)',line)
            if m: cur['probe']=[int(x) for x in re.findall(r'\d+',m.group(1))]
        if active and {'anm','date'}<=cur.keys(): rows.append((cur['anm'],cur['date'],cur.get('probe',[])))
    active={f'{a}_{d}':pr for a,d,pr in rows}
    out=[]
    for f in sorted(DATA.glob('*Ephys_Behavior/data_structure_*.mat')):
        key=f.stem.removeprefix('data_structure_')
        if key in active and active[key]: out.append((key,f,active[key]))
    return out

def hchars(h,obj):
    a=np.asarray(obj if isinstance(obj,np.ndarray) else h[obj])
    return ''.join(chr(int(x)) for x in a.flatten(order='F')).strip('\x00 ')

def nearest_fill(x):
    x=np.asarray(x,float).copy(); good=np.isfinite(x)
    if not good.any(): return x
    ix=np.arange(x.size); x[~good]=np.interp(ix[~good],ix[good],x[good]); return x

def interp_visible(t,y,target):
    """Linear interpolation inside support; leave outside/insufficient samples NaN."""
    t=np.asarray(t,float).ravel(); y=np.asarray(y,float).ravel(); ok=np.isfinite(t)&np.isfinite(y)
    if ok.sum()<2: return np.full(target.shape,np.nan)
    t=t[ok]; y=y[ok]; order=np.argsort(t); t=t[order]; y=y[order]
    u=np.r_[True,np.diff(t)>0]; t=t[u]; y=y[u]
    return np.interp(target,t,y,left=np.nan,right=np.nan) if t.size>1 else np.full(target.shape,np.nan)

def decode_h5_obj(h,ds,i): return np.asarray(h[ds[i,0]])

def load_h5(f,probes):
    h=h5py.File(f,'r'); bp=h['obj/bp']; ev=bp['ev']
    scalar=lambda p: np.asarray(h[p]).ravel()
    B={k:scalar('obj/bp/'+k).astype(bool) for k in ('R','L','hit','miss','no','early','autowater')}
    # stimulation may be a struct; canonical sessions generally have none.
    if 'stim' in bp and isinstance(bp['stim'],h5py.Group) and 'enable' in bp['stim']:
        B['stim']=np.asarray(bp['stim/enable']).ravel().astype(bool)
    else: B['stim']=np.zeros_like(B['hit'])
    B['go']=scalar('obj/bp/ev/goCue').astype(float); B['n']=len(B['go'])
    B['haveE']=scalar('obj/trials/bp/haveEphys').astype(bool)[:B['n']]
    B['haveV']=scalar('obj/trials/bp/haveVid').astype(bool)[:B['n']]
    units=[]
    for probe in probes:
        g=h[h['obj/clu'][probe-1,0]]
        if not isinstance(g,h5py.Group) or 'quality' not in g: continue
        for i in range(g['quality'].shape[0]):
            q=hchars(h,g['quality'][i,0]).strip().lower()
            if q in BADQ: continue
            tr=np.asarray(h[g['trial'][i,0]],int).ravel()-1
            tm=np.asarray(h[g['trialtm'][i,0]],float).ravel()
            units.append((q,tr,tm))
    traj=[]
    for ref in h['obj/traj'][:,0]:
        g=h[ref]; trials=[]
        if not isinstance(g,h5py.Group): traj.append(trials); continue
        for j in range(B['n']):
            try:
                ts=np.asarray(h[g['ts'][j,0]],float).transpose(2,1,0) # frames, xyz, features
                ft=np.asarray(h[g['frameTimes'][j,0]],float).ravel()
                fn=h[g['featNames'][j,0]]
                names=[hchars(h,r) for r in np.asarray(fn).flat]
                trials.append((ft,ts,names))
            except Exception: trials.append(None)
        traj.append(trials)
    return h,B,units,traj

def load_v5(f,probes):
    o=loadmat(f,squeeze_me=True,struct_as_record=False,variable_names=['obj'])['obj']; bp=o.bp
    B={k:np.asarray(getattr(bp,k)).ravel().astype(bool) for k in ('R','L','hit','miss','no','early','autowater')}
    B['stim']=np.asarray(getattr(bp.stim,'enable',np.zeros_like(B['hit']))).ravel().astype(bool)
    B['go']=np.asarray(bp.ev.goCue,float).ravel(); B['n']=len(B['go'])
    B['haveE']=np.asarray(o.trials.bp.haveEphys).ravel().astype(bool)[:B['n']]
    B['haveV']=np.asarray(o.trials.bp.haveVid).ravel().astype(bool)[:B['n']]
    units=[]
    for c in np.asarray(o.clu).flat: # later v5 objects already contain selected probe
        q=str(c.quality).strip().lower()
        if q in BADQ: continue
        units.append((q,np.asarray(c.trial,int).ravel()-1,np.asarray(c.trialtm,float).ravel()))
    traj=[]
    for x in np.asarray(o.traj).flat:
        arr=np.asarray(x).flat; trials=[]
        for j in range(B['n']):
            if j>=len(arr): trials.append(None); continue
            try:
                t=arr[j]; trials.append((np.asarray(t.frameTimes,float).ravel(),np.asarray(t.ts,float),[str(z) for z in np.asarray(t.featNames).flat]))
            except Exception: trials.append(None)
        traj.append(trials)
    return None,B,units,traj

def neural_arrays(B,units,trial_ids):
    n=B['n']; rates=[]; binned=[]
    for q,tr,tm in units:
        ok=(tr>=0)&(tr<n); tr=tr[ok]; al=tm[ok]-B['go'][tr]
        rate=np.count_nonzero((al>=TMIN)&(al<TMAX))/(n*(TMAX-TMIN))
        if rate<=0.5: continue
        mat=np.zeros((len(trial_ids),len(TIME)),np.float32)
        for oi,tid in enumerate(trial_ids):
            vals=al[tr==tid]; mat[oi]=np.histogram(vals,EDGES)[0].astype(np.float32)/DT
        # Exact reference mySmooth: gausswin(15, alpha=2.5), first half
        # zeroed for a causal kernel, normalized, conv(...,'same').
        N=15; alpha=2.5
        nn=np.arange(N,dtype=float)-(N-1)/2
        kern=np.exp(-0.5*(alpha*nn/((N-1)/2))**2)
        kern[:N//2]=0; kern/=kern.sum()
        mat=np.stack([np.convolve(row,kern,mode='same') for row in mat]).astype(np.float32)
        binned.append(mat); rates.append(rate)
    if not binned: raise RuntimeError('No neurons survive curation')
    cube=np.stack(binned,axis=1) # trial, neuron, time
    return cube,np.asarray(rates,np.float32)

def feature_speed(traj,trial_ids,view,name,B):
    out=[]
    trials=traj[view] if view<len(traj) else []
    for tid in trial_ids:
        z=trials[tid] if tid<len(trials) else None
        if z is None or not B['haveV'][tid]: out.append(np.full(len(TIME),np.nan,np.float32)); continue
        ft,ts,names=z
        try: fi=names.index(name)
        except ValueError: out.append(np.full(len(TIME),np.nan,np.float32)); continue
        # Normalize layouts to frames x 3(x,y,likelihood) x features.
        a=np.asarray(ts,float)
        if a.ndim!=3: out.append(np.full(len(TIME),np.nan,np.float32)); continue
        if a.shape[1]==3: pass
        elif a.shape[0]==3: a=np.transpose(a,(2,0,1))
        elif a.shape[2]==3: a=np.transpose(a,(0,2,1))
        else: out.append(np.full(len(TIME),np.nan,np.float32)); continue
        m=min(len(ft),a.shape[0]); ft=np.asarray(ft[:m],float); xy=a[:m,:2,fi]
        # Raw DLC stores invisible coordinates as NaN (reference findPosition).
        vis=np.isfinite(xy).all(1)
        vel=np.full(m,np.nan); dx=np.gradient(xy[:,0]); dy=np.gradient(xy[:,1]); vel[vis]=np.hypot(dx[vis],dy[vis])
        rel=ft-B['go'][tid]
        speed=interp_visible(rel,vel,TIME)
        # Visibility interpolation must not bridge long invisible intervals: nearest raw-frame mask.
        vok=np.isfinite(rel)
        if vok.sum()>1:
            order=np.argsort(rel[vok]); rr=rel[vok][order]; vv=vis[vok][order].astype(float)
            vi=np.interp(TIME,rr,vv,left=0,right=0)>=0.5; speed[~vi]=np.nan
        out.append(speed.astype(np.float32))
    return np.stack(out)

def motion_arrays(f,traj,B,trial_ids):
    mf=f.with_name(f.name.replace('data_structure_','motionEnergy_'))
    if not mf.exists(): return np.full((len(trial_ids),len(TIME)),np.nan,np.float32),False
    me=loadmat(mf,squeeze_me=True,struct_as_record=False)['me']
    raw=me.data
    # Reference loadMotionEnergy unwraps legacy struct-valued me.data.
    while hasattr(raw,'data'):
        raw=raw.data
    vals=np.asarray(raw).flat
    out=[]
    for tid in trial_ids:
        if tid>=len(vals) or not B['haveV'][tid] or not traj or tid>=len(traj[0]) or traj[0][tid] is None:
            out.append(np.full(len(TIME),np.nan,np.float32)); continue
        ft=traj[0][tid][0]; y=np.asarray(vals[tid],float).ravel(); m=min(len(ft),len(y)); rel=np.asarray(ft[:m])-B['go'][tid]
        z=interp_visible(rel,y[:m],TIME); z=nearest_fill(z).astype(np.float32); out.append(z)
    return np.stack(out),True

def discretize(x,missing_class=True):
    med=float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan
    y=np.full(x.shape,2 if missing_class else 0,np.int8); ok=np.isfinite(x)
    if np.isfinite(med): y[ok]=(x[ok]>=med).astype(np.int8)
    return y,med

def convert_session(key,f,probes,show=False):
    t0=time.time(); is_h5=h5py.is_hdf5(f)
    holder,B,units,traj=load_h5(f,probes) if is_h5 else load_v5(f,probes)
    valid=B['haveE']&np.isfinite(B['go'])&(B['go']>0)&(~B['early'])&(~B['stim'])
    tids=np.flatnonzero(valid)
    neural,rates=neural_arrays(B,units,tids)
    # Exclude native trailing/invalid periods with no activity in any retained
    # unit; stream flags are incorrect for these periods in two JEB24 files.
    neural_valid=np.any(neural!=0,axis=(1,2))
    invalid_zero_trials=tids[~neural_valid].tolist()
    tids=tids[neural_valid]; neural=neural[neural_valid]
    tongue=feature_speed(traj,tids,0,'tongue',B)
    paw=feature_speed(traj,tids,1,'top_paw',B)
    motion,hasme=motion_arrays(f,traj,B,tids)
    td,tmed=discretize(tongue); pd,pmed=discretize(paw); md,mmed=discretize(motion)
    trialsN=[]; trialsI=[]; trialsO=[]
    for j,tid in enumerate(tids):
        lick=0 if (B['L'][tid] and B['hit'][tid]) or (B['R'][tid] and B['miss'][tid]) else 1 if (B['R'][tid] and B['hit'][tid]) or (B['L'][tid] and B['miss'][tid]) else 2
        outcome=1 if B['hit'][tid] else 0 if B['miss'][tid] else 2
        trialsN.append(neural[j]); trialsI.append(TIME[None,:].copy())
        trialsO.append(np.vstack([np.full(len(TIME),lick,np.int8),np.full(len(TIME),int(B['autowater'][tid]),np.int8),np.full(len(TIME),outcome,np.int8),td[j],pd[j],md[j]]))
    if holder is not None: holder.close()
    info={'session_id':key,'source_file':str(f),'probes':probes,'source_trials':B['n'],'included_trials':len(tids),'trial_indices_0based':tids.tolist(),'excluded_all_zero_neural_trial_indices_0based':invalid_zero_trials,'neurons':len(rates),'mean_rates_hz':rates,'thresholds':{'tongue_velocity':tmed,'paw_velocity':pmed,'motion_energy':mmed},'has_motion_energy':hasme,'task_family':f.parent.name}
    print(f"{key}: trials {len(tids)}/{B['n']}, neurons {len(rates)}, {time.time()-t0:.2f}s",flush=True)
    if show:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(4,1,figsize=(12,10),sharex=True)
        ax[0].imshow(neural[0],aspect='auto',extent=[TMIN,TMAX,0,neural.shape[1]],origin='lower'); ax[0].set_ylabel('neurons')
        ax[1].plot(TIME,tongue[0]); ax[1].axhline(tmed,color='r'); ax[1].set_ylabel('tongue speed')
        ax[2].plot(TIME,paw[0]); ax[2].axhline(pmed,color='r'); ax[2].set_ylabel('paw speed')
        ax[3].plot(TIME,motion[0]); ax[3].axhline(mmed,color='r'); ax[3].set_ylabel('motion energy'); ax[3].set_xlabel('time from go cue (s)')
        for a in ax: a.axvline(0,color='k',ls='--',lw=.8)
        fig.suptitle(key+' alignment and discretization'); fig.tight_layout(); fig.savefig(ROOT/f'processing_{key}.png',dpi=140); plt.close(fig)
    return trialsN,trialsI,trialsO,info

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('outpicklefile'); g=ap.add_mutually_exclusive_group(); g.add_argument('--full',action='store_true'); g.add_argument('--sample',action='store_true'); ap.add_argument('--show-processing',action='store_true'); args=ap.parse_args()
    sessions=parse_sessions();
    if args.sample: sessions=sessions[:2]
    print(f'Processing {len(sessions)} sessions on {len(TIME)} bins ({DT*1000:g} ms)',flush=True)
    neural=[]; inputs=[]; outputs=[]; infos=[]; animals=[]
    for i,(key,f,probes) in enumerate(sessions):
        n,x,y,info=convert_session(key,f,probes,args.show_processing and i<2)
        if len(n)<2: print('Skipping session with <2 valid trials:',key); continue
        neural.append(n); inputs.append(x); outputs.append(y); infos.append(info); animals.append(key.split('_')[0])
    subjects=sorted(set(animals)); subject_idx=np.asarray([subjects.index(a) for a in animals],dtype=np.int64)
    data={'neural':neural,'input':inputs,'output':outputs,'subjects':subjects,'subject_idx':subject_idx,'brain_regions':['ALM'],'brain_region_idx':[np.zeros(len(s[0]),dtype=np.int64) for s in neural],'input_names':['time_from_go_cue_s'],'output_names':['lick_direction','behavioral_context','outcome','tongue_velocity','paw_velocity','motion_energy'],'output_values':[['left','right','none'],['DR','WC'],['incorrect','correct','ignore'],['below_session_median','at_or_above_session_median','not_visible'],['below_session_median','at_or_above_session_median','not_visible'],['below_session_median','at_or_above_session_median','no_video']],'metadata':{'task_description':'Decode lick direction, behavioral context, outcome, and discretized tongue velocity, paw velocity, and motion energy from ALM activity.','time_bin_size':DT*1000,'temporal_alignment_event':'go cue onset (trial-specific Bpod goCue)','off_start':TMIN,'off_end':TMAX,'neural_representation':'go-cue-aligned single-trial firing rate (spikes/s), Gaussian smoothed; curated non-garbage units >0.5 Hz','session_info':infos,'conversion_version':'1.0'}}
    # Internal invariants.
    assert len(neural)==len(inputs)==len(outputs)==len(subject_idx)==len(data['brain_region_idx'])
    assert all(a.shape[1]==len(TIME) for s in neural for a in s)
    assert all(x.shape==(1,len(TIME)) for s in inputs for x in s)
    assert all(y.shape==(6,len(TIME)) and np.issubdtype(y.dtype,np.integer) for s in outputs for y in s)
    with open(args.outpicklefile,'wb') as fh: pickle.dump(data,fh,pickle.HIGHEST_PROTOCOL)
    print(f'Saved {args.outpicklefile} ({Path(args.outpicklefile).stat().st_size/1e6:.1f} MB)')
    print('Totals: sessions',len(neural),'trials',sum(map(len,neural)),'neurons',sum(len(s[0]) for s in neural),'subjects',len(subjects))

if __name__=='__main__': main()
