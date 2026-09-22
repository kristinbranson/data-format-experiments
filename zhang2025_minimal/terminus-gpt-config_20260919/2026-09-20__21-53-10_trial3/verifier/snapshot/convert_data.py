#!/usr/bin/env python3
"""Convert the IBL Brain-Wide Map release to the decoder interchange format.

Processing choices follow code_zhang2025/src/utils/ibl_data_utils.py: BWM release
sessions/probes, all sorted clusters (qc=None), valid trials with finite essential
timestamps, no choice==0 trials, maximum goCue-to-feedback duration 10 s, and
100 20-ms bins spanning [-0.5, 1.5) s relative to stimulus onset.  Existing ALF
camera arrays are read directly because current SessionLoader tries to write derived
features into the intentionally read-only source cache.
"""
import argparse, gc, os, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader

BIN=.02; OFF0=-.5; OFF1=1.5; NBIN=100
CENTERS=np.arange(NBIN,dtype=np.float64)*BIN + OFF0 + BIN/2
RELEASE=Path('/app/code/code_zhang2025/data/bwm_release.csv')
CACHE=Path('/app/session_cache'); OUT=Path('/app/converted_data.pkl')

def trial_table(one,eid):
    tr=one.load_object(eid,'trials',collection='alf')
    keys=['goCue_times','choice','feedback_times','stimOn_times','response_times',
          'firstMovement_times','probabilityLeft','feedbackType']
    d={k:np.asarray(tr[k]) for k in keys}
    n=len(d['choice']); good=np.ones(n,bool)
    for k in keys: good &= np.isfinite(d[k])
    rt=d['firstMovement_times']-d['stimOn_times']
    good &= (rt >= 0.08) & (rt <= 2.0)
    good &= (d['feedback_times']-d['goCue_times'] <= 10.0)
    good &= d['choice'] != 0
    return d,good

def load_camera(one,eid):
    # Reference uses left whisker ME and falls back to right if unavailable.
    for cam in ('left','right'):
        try:
            t=np.asarray(one.load_dataset(eid,f'_ibl_{cam}Camera.times.npy',collection='alf'),float)
            v=np.asarray(one.load_dataset(eid,f'{cam}Camera.ROIMotionEnergy.npy',collection='alf'),float).squeeze()
            n=min(len(t),len(v)); t,v=t[:n],v[:n]
            ok=np.isfinite(t)&np.isfinite(v)
            if ok.sum()>2: return t[ok],v[ok],cam
        except Exception: pass
    raise RuntimeError('no whisker motion-energy stream')

def load_wheel(one,eid):
    w=one.load_object(eid,'wheel',collection='alf')
    t=np.asarray(w['timestamps'],float); p=np.asarray(w['position'],float)
    ok=np.isfinite(t)&np.isfinite(p); t,p=t[ok],p[ok]
    u=np.r_[True,np.diff(t)>0]; t,p=t[u],p[u]
    # Central-difference angular velocity, matching SessionLoader's velocity meaning.
    speed=np.abs(np.gradient(p,t))
    return t,speed

def load_spikes(one,rows):
    times=[]; clus=[]; regions=[]; offset=0
    for row in rows.itertuples():
        sl=SpikeSortingLoader(one=one,pid=row.pid,eid=row.eid,pname=row.probe_name)
        s,c,ch=sl.load_spike_sorting()
        merged=sl.merge_clusters(s,c,ch).to_df()
        st=np.asarray(s['times']); sc=np.asarray(s['clusters'],np.int64)
        # Cluster IDs index the merged table. Keep all clusters as in prepare_data(qc=None).
        n=len(merged); valid=(sc>=0)&(sc<n)
        times.append(st[valid]); clus.append(sc[valid]+offset)
        reg=np.asarray(merged['acronym'].fillna('void').astype(str)) if 'acronym' in merged else np.repeat('void',n)
        regions.extend(reg.tolist()); offset += n
    if not times: raise RuntimeError('no release probes')
    t=np.concatenate(times); c=np.concatenate(clus)
    order=np.argsort(t,kind='mergesort')
    return t[order],c[order],regions

def block_trial_numbers(prior):
    out=np.zeros(len(prior),np.float32); k=0
    for i in range(1,len(prior)):
        k = 0 if prior[i] != prior[i-1] else k+1
        out[i]=k
    return out

def convert_session(one,eid,rows):
    tr,good=trial_table(one,eid)
    idx=np.flatnonzero(good)
    if len(idx)<2: raise RuntimeError('fewer than two valid trials')
    wt,wv=load_wheel(one,eid); mt,mv,cam=load_camera(one,eid)
    st,sc,regions=load_spikes(one,rows)
    ncl=len(regions); prior_all=tr['probabilityLeft']; blockno=block_trial_numbers(prior_all)
    neural=[]; inputs=[]; outputs=[]; wheel_raw=[]; whisk_raw=[]
    for j in idx:
        onset=tr['stimOn_times'][j]; edges=onset+OFF0+np.arange(NBIN+1)*BIN
        lo=np.searchsorted(st,edges[0],'left'); hi=np.searchsorted(st,edges[-1],'left')
        tt=st[lo:hi]; cc=sc[lo:hi]
        tb=np.floor((tt-edges[0])/BIN).astype(np.int64)
        flat=cc*NBIN+tb
        counts=np.bincount(flat,minlength=ncl*NBIN).reshape(ncl,NBIN)
        counts=np.minimum(counts,255).astype(np.uint8)
        x=onset+CENTERS
        ws=np.interp(x,wt,wv).astype(np.float32)
        me=np.interp(x,mt,mv).astype(np.float32)
        inp=np.vstack((CENTERS.astype(np.float32),np.full(NBIN,blockno[j],np.float32)))
        choice=np.uint8(0 if tr['choice'][j] == -1 else 1)
        pv=float(prior_all[j]); prior=np.uint8({.2:0,.5:1,.8:2}[round(pv,1)])
        # Raw behavior occupies rows 2/3 temporarily; global tertiles are applied later.
        out=np.empty((4,NBIN),np.float32); out[0]=choice; out[1]=prior; out[2]=ws; out[3]=me
        neural.append(counts); inputs.append(inp); outputs.append(out)
        wheel_raw.append(ws); whisk_raw.append(me)
    return {'eid':eid,'subject':str(rows.subject.iloc[0]),'neural':neural,'input':inputs,
            'output':outputs,'regions':regions,'wheel':np.concatenate(wheel_raw),
            'whisk':np.concatenate(whisk_raw),'camera':cam,'n_source_trials':len(good)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--limit',type=int); ap.add_argument('--rebuild',action='store_true')
    ap.add_argument('--output',default=str(OUT)); ap.add_argument('--shard',type=str); ap.add_argument('--cache-only',action='store_true'); args=ap.parse_args()
    CACHE.mkdir(exist_ok=True); rel=pd.read_csv(RELEASE)
    eids=list(dict.fromkeys(rel.eid));
    subset=Path('/app/data/DATALIMIT_SUBSET.csv')
    if subset.exists():
        ss=pd.read_csv(subset); allowed=set(ss['eid'] if 'eid' in ss else ss.iloc[:,0]); eids=[e for e in eids if e in allowed]
    if args.limit: eids=eids[:args.limit]
    if args.shard:
        si,sn=map(int,args.shard.split('/')); eids=eids[si::sn]
    one=ONE(); failures=[]
    for k,eid in enumerate(eids):
        f=CACHE/f'{eid}.pkl'
        if f.exists() and not args.rebuild:
            print(f'[{k+1}/{len(eids)}] cached {eid}',flush=True); continue
        try:
            print(f'[{k+1}/{len(eids)}] converting {eid}',flush=True)
            s=convert_session(one,eid,rel[rel.eid==eid])
            tmp=f.with_suffix('.tmp'); pickle.dump(s,open(tmp,'wb'),protocol=5); os.replace(tmp,f)
            print(f"  {len(s['neural'])} trials, {len(s['regions'])} neurons",flush=True)
        except Exception as ex:
            warnings.warn(f'{eid}: {ex}'); failures.append((eid,repr(ex)))
        gc.collect()
    if args.cache_only:
        print(f'Cache-only shard complete; failures={len(failures)}')
        return
    sessions=[]
    for eid in eids:
        f=CACHE/f'{eid}.pkl'
        if f.exists(): sessions.append(pickle.load(open(f,'rb')))
    if not sessions: raise RuntimeError('no sessions converted')
    # Global finite tertiles give reproducible, balanced three-class behavior outputs.
    qw=np.quantile(np.concatenate([s['wheel'] for s in sessions]),[1/3,2/3])
    qm=np.quantile(np.concatenate([s['whisk'] for s in sessions]),[1/3,2/3])
    allreg=sorted(set(r for s in sessions for r in s['regions'])); rmap={r:i for i,r in enumerate(allreg)}
    subjects=sorted(set(s['subject'] for s in sessions)); smap={s:i for i,s in enumerate(subjects)}
    for s in sessions:
        for o in s['output']:
            o[2]=np.digitize(o[2],qw).astype(np.float32); o[3]=np.digitize(o[3],qm).astype(np.float32)
            o[:]=o.astype(np.uint8)
    data={'neural':[s['neural'] for s in sessions], 'input':[s['input'] for s in sessions],
          'output':[[o.astype(np.uint8) for o in s['output']] for s in sessions],
          'subjects':subjects,'subject_idx':np.asarray([smap[s['subject']] for s in sessions],np.int64),
          'brain_regions':allreg,
          'brain_region_idx':[np.asarray([rmap[r] for r in s['regions']],np.int64) for s in sessions],
          'input_names':['time since stimulus onset','trial number in block'],
          'output_names':['choice','prior probability of left','wheel speed','whisker motion energy'],
          'output_values':[['left','right'],['0.2','0.5','0.8'],['low','medium','high'],['low','medium','high']],
          'metadata':{'task_description':'IBL visual decision task: decode choice, block prior, wheel speed, and whisker motion energy.',
            'time_bin_size':20.0,'temporal_alignment_event':'visual stimulus onset (stimOn_times)',
            'off_start':-0.5,'off_end':1.5,'source_release':'Zhang 2025 bwm_release.csv / IBL Brain-Wide Map',
            'neural_measure':'spike counts per 20 ms bin; all release clusters (reference qc=None)',
            'trial_filter':'reference filter: finite required fields; 0.08 <= firstMovement-stimOn <= 2 s; choice != 0; feedback-goCue <= 10 s',
            'behavior_bin_edges':{'wheel_speed':qw.tolist(),'whisker_motion_energy':qm.tolist()},
            'session_info':[{'eid':s['eid'],'subject':s['subject'],'n_source_trials':s['n_source_trials'],'whisker_camera':s['camera']} for s in sessions],
            'failed_sessions':failures}}
    tmp=Path(args.output+'.tmp'); pickle.dump(data,open(tmp,'wb'),protocol=5); os.replace(tmp,args.output)
    print(f'Wrote {args.output}: {len(sessions)} sessions, {sum(map(len,data["neural"]))} trials; thresholds {qw}, {qm}')
if __name__=='__main__': main()
