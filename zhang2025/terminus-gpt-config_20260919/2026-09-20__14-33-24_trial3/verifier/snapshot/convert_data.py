#!/usr/bin/env python3
"""Convert the staged IBL Brain-Wide Map release to decoder-compatible pickle format.

Usage: python -u /app/convert_data.py OUTFILE [--full|--sample] [--show-processing]
"""
from __future__ import annotations
import argparse, json, pickle, time, warnings
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from brainbox.behavior.wheel import interpolate_position, velocity_filtered
from iblatlas.regions import BrainRegions

ROOT = Path('/app/data/one_cache')
COHORT = Path('/app/code/code_zhang2025/data/bwm_release.csv')
CACHE = Path('/app/cache/conversion_behavior')
BIN = 0.02
OFF0, OFF1 = -0.5, 1.5
NBIN = 100
REL_TIME = np.linspace(OFF0 + BIN, OFF1, NBIN, dtype=np.float32)
REQ_TRIAL = {'goCue_times','choice','stimOn_times','probabilityLeft','feedback_times',
             'feedbackType','firstMovement_times'}


def session_path(r):
    return ROOT / str(r.lab) / 'Subjects' / str(r.subject) / str(r.date) / f'{int(r.session_number):03d}'


def full_trial_table(alf: Path):
    candidates=[]
    for p in alf.glob('**/_ibl_trials.table.pqt'):
        try:
            x=pd.read_parquet(p)
            if REQ_TRIAL.issubset(x.columns): candidates.append((len(x),str(p),p,x))
        except Exception: pass
    if not candidates: raise FileNotFoundError(f'No full trial table in {alf}')
    return max(candidates,key=lambda z:(z[0],z[1]))[2:]


def newest(paths):
    paths=list(paths)
    if not paths: return None
    return max(paths,key=lambda p:str(p))


def paired_stream(alf: Path, side: str):
    times=list(alf.glob(f'**/_ibl_{side}Camera.times.npy'))
    vals=list(alf.glob(f'**/{side}Camera.ROIMotionEnergy.npy'))
    # Try every pair, preferring newest, and require equal lengths.
    for tp in sorted(times,key=str,reverse=True):
        t=np.load(tp,mmap_mode='r')
        for vp in sorted(vals,key=str,reverse=True):
            v=np.load(vp,mmap_mode='r')
            if len(t)==len(v) and len(t)>1:
                return tp,vp
    return None,None


def trial_mask(x):
    m=np.ones(len(x),bool)
    for c in ['stimOn_times','choice','feedback_times','probabilityLeft','firstMovement_times','feedbackType']:
        m &= x[c].notna().to_numpy()
    rt=x.firstMovement_times.to_numpy()-x.stimOn_times.to_numpy()
    m &= (rt >= .08) & (rt <= 2.)
    m &= x.choice.to_numpy()!=0
    m &= (x.feedback_times.to_numpy()-x.goCue_times.to_numpy() <= 10.)
    return m


def trial_in_block(prob):
    prob=np.asarray(prob)
    change=np.r_[True, prob[1:] != prob[:-1]]
    starts=np.maximum.accumulate(np.where(change,np.arange(len(prob)),0))
    return (np.arange(len(prob))-starts+1).astype(np.float32)


def derive_wheel(alf):
    tp=newest(alf.glob('**/_ibl_wheel.timestamps.npy'))
    pp=newest(alf.glob('**/_ibl_wheel.position.npy'))
    if tp is None or pp is None: raise FileNotFoundError('wheel stream missing')
    t=np.asarray(np.load(tp),dtype=float); p=np.asarray(np.load(pp),dtype=float)
    if len(t)!=len(p) or len(t)<20: raise ValueError('invalid wheel arrays')
    pos,ti=interpolate_position(t,p,freq=1000)
    vel,_=velocity_filtered(pos,1000)
    return np.asarray(ti),np.abs(np.asarray(vel))


def interp_trials(times, values, stim, base_mask):
    """Reference linear interpolation at bin ends; return values and valid indices."""
    times=np.asarray(times,dtype=float); values=np.asarray(values,dtype=float)
    order=np.argsort(times,kind='stable'); times=times[order]; values=values[order]
    # Remove duplicate timestamps to keep interp deterministic.
    keep=np.r_[True,np.diff(times)>0]; times=times[keep]; values=values[keep]
    idx=np.flatnonzero(base_mask)
    out=[]; kept=[]
    for i in idx:
        beg,end=stim[i]+OFF0,stim[i]+OFF1
        ib=np.searchsorted(times,beg,'left'); ie=np.searchsorted(times,end,'left')
        if ie<=ib or ib>=len(times): continue
        tx=times[ib:ie]; vy=values[ib:ie]
        if len(tx)<2 or abs(beg-tx[0])>BIN or abs(end-tx[-1])>BIN or not np.isfinite(vy).all(): continue
        grid=stim[i]+REL_TIME.astype(float)
        y=interp1d(tx,vy,kind='linear',fill_value='extrapolate')(grid)
        if np.isfinite(y).all(): out.append(y.astype(np.float32)); kept.append(i)
    if not out: return np.empty((0,NBIN),np.float32),np.empty(0,int)
    return np.stack(out),np.asarray(kept,int)


def preprocess_behavior(row,eid,cache_file):
    alf=session_path(row)/'alf'; trial_p,x=full_trial_table(alf)
    base=trial_mask(x); stim=x.stimOn_times.to_numpy(dtype=float)
    side=None; tp=vp=None
    for sd in ('left','right'):
        tp,vp=paired_stream(alf,sd)
        if tp is not None: side=sd; break
    if side is None: raise FileNotFoundError('no matched camera time/motion stream')
    ct=np.load(tp,mmap_mode='r'); cv=np.load(vp,mmap_mode='r')
    cam,ci=interp_trials(ct,cv,stim,base)
    wt,wv=derive_wheel(alf); wheel,wi=interp_trials(wt,wv,stim,base)
    common=np.intersect1d(ci,wi,assume_unique=True)
    if len(common)<2: raise ValueError(f'only {len(common)} jointly valid trials')
    cmap={v:j for j,v in enumerate(ci)}; wmap={v:j for j,v in enumerate(wi)}
    cam=np.stack([cam[cmap[i]] for i in common]); wheel=np.stack([wheel[wmap[i]] for i in common])
    probs=x.probabilityLeft.to_numpy(dtype=float); choices=x.choice.to_numpy(dtype=float)
    tib=trial_in_block(probs)
    np.savez(cache_file,trial_idx=common.astype(np.int32),wheel=wheel,whisker=cam,
             choice=choices[common].astype(np.int8),prior=probs[common].astype(np.float32),
             trial_in_block=tib[common],stim=stim[common],side=np.array(side),trial_path=np.array(str(trial_p)))
    return {'eid':eid,'trials':len(common),'side':side,'raw':len(x),'base_valid':int(base.sum())}


def resolve_probe(row):
    base=session_path(row)/'alf'/str(row.probe_name)
    candidates=[]
    for p in base.glob('**/clusters.metrics.pqt'):
        try:
            x=pd.read_parquet(p)
            if 'cluster_id' in x and 'label' in x: candidates.append((len(x),str(p),p,x))
        except Exception: pass
    if not candidates: raise FileNotFoundError(f'cluster metrics missing {row.pid}')
    _,_,mp,metrics=max(candidates,key=lambda z:(z[0],z[1]))
    d=mp.parent
    def local_or_glob(name):
        p=d/name
        if p.exists(): return p
        q=newest(base.glob(f'**/{name}'))
        if q is None: raise FileNotFoundError(f'{name} missing for {row.pid}')
        return q
    return metrics, {n:local_or_glob(n) for n in ['spikes.times.npy','spikes.clusters.npy','clusters.channels.npy','channels.brainLocationIds_ccf_2017.npy']}


def bin_probe(row,stim):
    metrics,p=resolve_probe(row)
    cluster_ids=metrics.cluster_id.to_numpy(dtype=int)
    n=len(cluster_ids); maxid=max(int(cluster_ids.max()),int(np.max(np.load(p['spikes.clusters.npy'],mmap_mode='r'))))
    lookup=np.full(maxid+1,-1,dtype=np.int32); lookup[cluster_ids]=np.arange(n,dtype=np.int32)
    st=np.load(p['spikes.times.npy'],mmap_mode='r'); sc=np.load(p['spikes.clusters.npy'],mmap_mode='r')
    order=None
    if len(st)>1 and np.any(np.diff(st[:min(len(st),100000)])<0): order=np.argsort(st,kind='stable'); st=np.asarray(st)[order]; sc=np.asarray(sc)[order]
    mats=[]; max_count=0
    for s in stim:
        beg=s+OFF0; end=s+OFF1; ib=np.searchsorted(st,beg,'left'); ie=np.searchsorted(st,end,'left')
        spike_t=np.asarray(st[ib:ie]); ids=np.asarray(sc[ib:ie],dtype=int)
        valid_id=(ids>=0)&(ids<len(lookup)); spike_t=spike_t[valid_id]; ids=ids[valid_id]
        mapped=lookup[ids]; valid_map=mapped>=0; spike_t=spike_t[valid_map]; mapped=mapped[valid_map]
        tb=np.floor((spike_t-beg)/BIN).astype(int)
        valid2=(tb>=0)&(tb<NBIN); flat=mapped[valid2]*NBIN+tb[valid2]
        a=np.bincount(flat,minlength=n*NBIN).reshape(n,NBIN)
        max_count=max(max_count,int(a.max(initial=0))); mats.append(a)
    ch=np.asarray(np.load(p['clusters.channels.npy']),dtype=int)
    atlas=np.asarray(np.load(p['channels.brainLocationIds_ccf_2017.npy']),dtype=int)
    if len(ch)!=n: raise ValueError(f'cluster channel length {len(ch)} != metrics {n}')
    if np.any((ch<0)|(ch>=len(atlas))): raise ValueError('cluster channel out of range')
    br=BrainRegions(); native=br.id2acronym(atlas[ch]); regions=br.acronym2acronym(native,mapping='Beryl').astype(str)
    return mats,regions,max_count


def thresholds(values):
    x=np.concatenate([v.reshape(-1) for v in values])
    q=np.quantile(x,[1/3,2/3])
    if not np.isfinite(q).all() or q[0]>=q[1]:
        raise ValueError(f'Collapsed/nonfinite tertiles {q}; rank fallback not implemented because class meaning would vary among ties')
    return q.astype(float)


def make_plot(eid,z,neural,out,outpath,wthr,mthr):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(4,1,figsize=(12,10),sharex=True)
    ax[0].imshow(neural[0][:min(80,neural[0].shape[0])],aspect='auto',origin='lower',extent=[OFF0,OFF1,0,min(80,neural[0].shape[0])]); ax[0].set_ylabel('neurons'); ax[0].set_title(f'{eid}: spike counts, trial 1')
    ax[1].plot(REL_TIME,z['wheel'][0]); ax[1].axhline(wthr[0],ls='--'); ax[1].axhline(wthr[1],ls='--'); ax[1].set_ylabel('wheel speed')
    ax[2].plot(REL_TIME,z['whisker'][0]); ax[2].axhline(mthr[0],ls='--'); ax[2].axhline(mthr[1],ls='--'); ax[2].set_ylabel('whisker ME')
    ax[3].step(REL_TIME,out[0][2],where='mid',label='wheel bin'); ax[3].step(REL_TIME,out[0][3],where='mid',label='whisker bin'); ax[3].legend(); ax[3].set_xlabel('seconds from stimulus onset'); ax[3].set_ylabel('class')
    for a in ax: a.axvline(0,color='r',alpha=.5)
    fig.tight_layout(); fig.savefig(outpath,dpi=140); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('outpicklefile'); g=ap.add_mutually_exclusive_group(); g.add_argument('--full',action='store_true'); g.add_argument('--sample',action='store_true'); ap.add_argument('--show-processing',action='store_true'); args=ap.parse_args()
    t0=time.time(); CACHE.mkdir(parents=True,exist_ok=True)
    rel=pd.read_csv(COHORT,index_col=0); sessions=rel.drop_duplicates('eid',keep='first')
    if args.sample: sessions=sessions.iloc[:2]
    print(f'Candidate sessions: {len(sessions)} (sample={args.sample})',flush=True)
    kept=[]; logs=[]
    for k,(_,r) in enumerate(sessions.iterrows(),1):
        eid=str(r.eid); cf=CACHE/f'{eid}.npz'; ts=time.time()
        try:
            info=preprocess_behavior(r,eid,cf); kept.append((r,eid,cf)); info['seconds']=time.time()-ts; logs.append(info); print(f'behavior {k}/{len(sessions)} {eid}: {info}',flush=True)
        except Exception as e: print(f'EXCLUDE {eid}: {type(e).__name__}: {e}',flush=True)
    if not kept: raise RuntimeError('No valid sessions')
    wz=[np.load(cf)['wheel'] for _,_,cf in kept]; mz=[np.load(cf)['whisker'] for _,_,cf in kept]
    wthr=thresholds(wz); mthr=thresholds(mz); del wz,mz
    print('Global wheel tertiles',wthr,'whisker tertiles',mthr,flush=True)
    neural_all=[]; inputs_all=[]; outputs_all=[]; subj=[]; region_names=[]; session_regions=[]; session_info=[]
    for si,(r,eid,cf) in enumerate(kept):
        ts=time.time(); z=np.load(cf); stim=z['stim']; probe_rows=rel[rel.eid.astype(str)==eid]
        pmats=[]; preg=[]; maxcount=0
        for _,pr in probe_rows.iterrows():
            m,rg,mx=bin_probe(pr,stim); pmats.append(m); preg.extend(rg.tolist()); maxcount=max(maxcount,mx)
        if maxcount>255: dtype=np.uint16
        else: dtype=np.uint8
        ntr=len(stim); neural=[np.concatenate([pmats[p][j] for p in range(len(pmats))],axis=0).astype(dtype,copy=False) for j in range(ntr)]
        tib=z['trial_in_block']; inp=[np.vstack((REL_TIME,np.full(NBIN,tib[j],np.float32))).astype(np.float32) for j in range(ntr)]
        wc=np.searchsorted(wthr,z['wheel'],side='right').astype(np.uint8); mc=np.searchsorted(mthr,z['whisker'],side='right').astype(np.uint8)
        cmap={-1:0,1:1}; pmap={0.2:0,0.5:1,0.8:2}; out=[]
        for j in range(ntr):
            c=cmap[int(z['choice'][j])]; pv=float(z['prior'][j]); key=min(pmap,key=lambda q:abs(q-pv))
            if abs(key-pv)>1e-6: raise ValueError(f'unexpected prior {pv}')
            out.append(np.vstack((np.full(NBIN,c,np.uint8),np.full(NBIN,pmap[key],np.uint8),wc[j],mc[j])))
        neural_all.append(neural); inputs_all.append(inp); outputs_all.append(out); subj.append(str(r.subject)); session_regions.append(preg)
        session_info.append({'eid':eid,'subject':str(r.subject),'n_trials':ntr,'n_neurons':len(preg),'camera_side':str(z['side']),'max_spike_count':maxcount,'neural_dtype':str(np.dtype(dtype))})
        if args.show_processing and si<2: make_plot(eid,z,neural,out,f'/app/processing_{eid}.png',wthr,mthr)
        print(f'convert {si+1}/{len(kept)} {eid}: trials={ntr} neurons={len(preg)} maxcount={maxcount} sec={time.time()-ts:.2f}',flush=True)
    subjects=list(dict.fromkeys(subj)); subject_idx=np.array([subjects.index(x) for x in subj],dtype=np.int32)
    brain_regions=sorted(set(x for ss in session_regions for x in ss)); bri=[np.array([brain_regions.index(x) for x in ss],dtype=np.int32) for ss in session_regions]
    data={'neural':neural_all,'input':inputs_all,'output':outputs_all,'subjects':subjects,'subject_idx':subject_idx,
          'brain_regions':brain_regions,'brain_region_idx':bri,'input_names':['time since stimulus onset','trial number in block'],
          'output_names':['choice','prior probability of left','wheel speed','whisker motion energy'],
          'output_values':[['left','right'],['0.2','0.5','0.8'],['low','medium','high'],['low','medium','high']],
          'metadata':{'task_description':'IBL visual decision task; decode choice, block prior, wheel speed and whisker motion energy from stimulus-aligned spike counts.',
          'time_bin_size':20.0,'temporal_alignment_event':'visual stimulus onset (stimOn_times)','off_start':OFF0,'off_end':OFF1,
          'behavior_sample_times':'20 ms bin ends from -0.48 to +1.50 s','wheel_speed_thresholds':wthr.tolist(),'whisker_motion_energy_thresholds':mthr.tolist(),
          'discretization':'global empirical tertiles across retained session-trial-time samples; searchsorted side=right','neural_representation':'20 ms spike counts; all Kilosort clusters; probes merged by session',
          'trial_filter':'required events finite; 0.08<=reaction time<=2 s; choice nonzero; feedback-goCue<=10 s; complete finite wheel/camera window',
          'session_info':session_info,'source_cohort':'code_zhang2025/data/bwm_release.csv'}}
    for sidx in range(len(neural_all)):
        assert len(neural_all[sidx])==len(inputs_all[sidx])==len(outputs_all[sidx])>=2
        assert len(bri[sidx])==neural_all[sidx][0].shape[0]
        for n,i,o in zip(neural_all[sidx],inputs_all[sidx],outputs_all[sidx]): assert n.shape[1]==i.shape[1]==o.shape[1]==NBIN and np.isfinite(i).all()
    with open(args.outpicklefile,'wb') as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
    total_trials=sum(map(len,neural_all)); print(f'Saved {args.outpicklefile}: sessions={len(neural_all)} subjects={len(subjects)} trials={total_trials} summed_session_neurons={sum(len(x) for x in bri)} regions={len(brain_regions)} elapsed={time.time()-t0:.2f}s',flush=True)
    print('choice counts',Counter(int(o[0,0]) for ss in outputs_all for o in ss)); print('prior counts',Counter(int(o[1,0]) for ss in outputs_all for o in ss)); print('wheel bins',Counter(np.concatenate([o[2] for ss in outputs_all for o in ss]).tolist())); print('whisker bins',Counter(np.concatenate([o[3] for ss in outputs_all for o in ss]).tolist()))

if __name__=='__main__': main()
