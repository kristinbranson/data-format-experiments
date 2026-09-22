#!/usr/bin/env python3
"""Convert mesoscope VR data to decoder-compatible trial-wise format."""
import argparse, gc, pickle, re, sys, time
from pathlib import Path
import numpy as np

ROOT=Path('/app/data')
DT_GLOBAL=0.31480416655540466
REGIONS=['V1','medial higher visual','lateral higher visual','anterior higher visual','other visual area','unassigned']

def base_id(k): return re.sub(r'_swap[12]$','',k)

def load_sources():
    records={}; record_files={}
    for f in sorted((ROOT/'beh').glob('Beh_*.npy')):
        o=np.load(f,allow_pickle=True).item()
        for k,v in o.items():
            if isinstance(v,dict) and 'ntrials' in v:
                records[k]=v; record_files[k]=f.name
    info=np.load(ROOT/'beh'/'Imaging_Exp_info.npy',allow_pickle=True).item()
    db={}
    for group,arr in info.items():
        for d in arr:
            sid=f"{d['mname']}_{d['datexp']}_{d['blk']}"
            db.setdefault(sid,[]).append((group,d))
    neural=sorted(re.sub(r'_neural_data$','',f.stem) for f in (ROOT/'spk').glob('*_neural_data.npy'))
    aliases={sid:sorted([k for k in records if base_id(k)==sid],key=lambda k:('_swap' in k,k)) for sid in neural}
    if any(not x for x in aliases.values()): raise RuntimeError('Missing behavior: '+str([k for k,v in aliases.items() if not v]))
    return records,record_files,db,neural,aliases

def merged_stimuli(alias_names,records):
    """Return physical visual wall identity; aliases share identical WallName."""
    out=np.asarray(records[alias_names[0]]['WallName']).astype(str).copy()
    for k in alias_names[1:]:
        other=np.asarray(records[k]['WallName']).astype(str)
        if not np.array_equal(out,other):
            raise ValueError(f'Behavior aliases disagree on WallName: {alias_names}')
    return out

def session_day(sid,db):
    vals=[]; groups=[]
    for group,d in db.get(sid,[]):
        groups.append(group); v=d.get('sess#')
        if v is not None:
            try: vals.append(float(v))
            except Exception: pass
    if vals: return float(max(vals)),False,groups
    # Explicit stage fallback: before=0, first after/test=1, later train2 after=2.
    text=' '.join(groups)
    if 'before' in text: return 0.0,True,groups
    if 'train2_after' in text: return 2.0,True,groups
    return 1.0,True,groups

def area_indices(sid):
    datebase='_'.join(sid.split('_')[:-1])
    with np.load(ROOT/'retinotopy'/f'{datebase}_trans.npz') as z: a=z['iarea'].astype(int)
    out=np.full(len(a),5,dtype=np.int16)
    out[a==8]=0; out[np.isin(a,[0,1,2,9])]=1; out[np.isin(a,[5,6])]=2
    out[np.isin(a,[3,4])]=3; out[a==7]=4; out[a==-1]=5
    return out

def choose_behavior(sid,aliases,records): return records[aliases[sid][0]]

def retained_mask(b,nfr=None):
    n=len(b['ft_trInd']) if nfr is None else min(nfr,len(b['ft_trInd']))
    return np.asarray(b['ft_CorrSpc'][:n],bool)&(np.asarray(b['ft_move'][:n])>0)

def scan(records,db,sids,aliases):
    speeds=[]; stimset=set(); stats={'trials_source':0,'trials_kept':0,'frames':0,'empty':0}
    for sid in sids:
        b=choose_behavior(sid,aliases,records); st=merged_stimuli(aliases[sid],records)
        m=retained_mask(b); tri=np.where(np.isfinite(np.asarray(b['ft_trInd'][:len(m)],float)),np.asarray(b['ft_trInd'][:len(m)],float),-1).astype(int)
        speed=np.asarray(b['ft_RunSpeed'][:len(m)],np.float32)
        speeds.append(speed[m]); stimset.update(map(str,st)); stats['trials_source']+=int(b['ntrials'])
        for t in range(int(b['ntrials'])):
            z=int(np.sum(m&(tri==t))); stats['frames']+=z
            if z: stats['trials_kept']+=1
            else: stats['empty']+=1
    speed=np.concatenate(speeds); qs=np.quantile(speed,[.25,.5,.75]).astype(np.float32)
    return qs,sorted(stimset),stats

def load_neural(sid):
    o=np.load(ROOT/'spk'/f'{sid}_neural_data.npy',allow_pickle=True).item()['spks']
    return np.concatenate([np.asarray(x,dtype=np.float32) for x in o],axis=0)

def convert_session(sid,records,db,aliases,stim_to_idx,speed_q,plot=False):
    t0=time.time(); b=choose_behavior(sid,aliases,records); stimuli=merged_stimuli(aliases[sid],records)
    neural=load_neural(sid); nfr=min(neural.shape[1],len(b['ft_trInd']))
    neural=neural[:,:nfr]; areas=area_indices(sid)
    if len(areas)!=neural.shape[0]: raise ValueError(f'{sid}: area/neuron mismatch {len(areas)} {neural.shape[0]}')
    tri_raw=np.asarray(b['ft_trInd'][:nfr],float); tri=np.where(np.isfinite(tri_raw),tri_raw,-1).astype(int); mask=retained_mask(b,nfr)
    pos=np.asarray(b['ft_Pos'][:nfr],np.float32); speed=np.asarray(b['ft_RunSpeed'][:nfr],np.float32)
    ft=np.asarray(b['ft'][:nfr],float); d=np.diff(ft)*86400; d=d[np.isfinite(d)&(d>0)]
    dt=float(np.median(d)) if len(d) else DT_GLOBAL
    day,imputed,groups=session_day(sid,db)
    lick_by_trial={t:[] for t in range(int(b['ntrials']))}
    lf=np.asarray(b['LickFr'],float); lt=np.asarray(b['LickTrind'],float)
    for f,t in zip(lf,lt):
        if np.isfinite(f) and np.isfinite(t): lick_by_trial.setdefault(int(t),[]).append(float(f))
    ns=[]; ins=[]; outs=[]; kept_ids=[]
    rew=np.asarray(b['isRew']).astype(int); sound=np.asarray(b['SoundFr'],float)
    # Build one selected contiguous matrix in trial order. Trial matrices below are
    # lightweight views, avoiding hundreds of persistent advanced-indexing bases.
    trial_indices=[]; trial_ids=[]
    for t in range(int(b['ntrials'])):
        idx=np.flatnonzero(mask&(tri==t))
        if len(idx): trial_indices.append(idx); trial_ids.append(t)
    ordered=np.concatenate(trial_indices) if trial_indices else np.empty(0,dtype=int)
    selected=np.ascontiguousarray(neural[:,ordered],dtype=np.float32)
    del neural; gc.collect()
    offset=0
    for t,idx in zip(trial_ids,trial_indices):
        T=len(idx); ntrial=selected[:,offset:offset+T]; offset+=T
        cue=(sound[t]-idx)*dt
        since=(idx-idx[0])*dt
        inp=np.vstack([cue,np.full(T,day),since,np.full(T,rew[t])]).astype(np.float32)
        lick=np.zeros(T,dtype=np.int16)
        for f in lick_by_trial.get(t,[]):
            j=int(np.argmin(np.abs(idx-f)))
            if abs(float(idx[j])-f)<=0.5001: lick[j]=1
        position=np.clip((pos[idx]/10).astype(np.int16),0,3)
        speedbin=np.digitize(speed[idx],speed_q,right=False).astype(np.int16)
        stim=np.full(T,stim_to_idx[str(stimuli[t])],dtype=np.int16)
        out=np.vstack([stim,lick,position,speedbin]).astype(np.int16)
        ns.append(ntrial); ins.append(inp); outs.append(out); kept_ids.append(t)
    meta={'session_id':sid,'behavior_aliases':aliases[sid],'source_trial_ids':kept_ids,'training_day':day,
          'training_day_imputed':imputed,'experiment_groups':groups,'session_dt_seconds':dt,
          'n_source_frames':nfr,'n_retained_frames':sum(x.shape[1] for x in ns)}
    print(f'{sid}: neurons={len(areas)} trials={len(ns)} frames={meta["n_retained_frames"]} dt={dt:.6f}s time={time.time()-t0:.1f}s',flush=True)
    return ns,ins,outs,areas,meta

def make_plot(sid,neural,inp,out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    n=min(3,len(neural)); fig,ax=plt.subplots(4,n,figsize=(5*n,10),squeeze=False)
    for j in range(n):
        ax[0,j].imshow(neural[j][:min(100,neural[j].shape[0])],aspect='auto'); ax[0,j].set_title(f'{sid} trial {j}: neural')
        ax[1,j].plot(inp[j][2],inp[j][0]); ax[1,j].axhline(0,color='r'); ax[1,j].set_ylabel('time to cue (s)')
        ax[2,j].step(inp[j][2],out[j][2],where='mid'); ax[2,j].set_ylabel('position class')
        ax[3,j].step(inp[j][2],out[j][1],label='lick'); ax[3,j].step(inp[j][2],out[j][3]/3,label='speed/3'); ax[3,j].legend()
    fig.tight_layout(); fig.savefig(f'/app/processing_{sid}.png',dpi=120); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('outpicklefile'); g=ap.add_mutually_exclusive_group(); g.add_argument('--full',action='store_true'); g.add_argument('--sample',action='store_true'); ap.add_argument('--show-processing',action='store_true'); args=ap.parse_args()
    t0=time.time(); records,record_files,db,all_sids,aliases=load_sources(); 
    if args.sample:
        # Representative supervised sessions with both reward classes and lick events.
        candidates=[]
        for sid in all_sids:
            b=choose_behavior(sid,aliases,records)
            if len(np.asarray(b['LickFr'])) and len(np.unique(np.asarray(b['isRew']).astype(int)))>1: candidates.append(sid)
        sids=candidates[:2]
        if len(sids)<2: raise RuntimeError('Could not find two representative sample sessions')
    else: sids=all_sids
    speed_q,stim_values,scan_stats=scan(records,db,sids,aliases); stim_to_idx={x:i for i,x in enumerate(stim_values)}
    print('scan',scan_stats,'speed quartiles',speed_q.tolist(),'stimuli',stim_values,flush=True)
    data={'neural':[],'input':[],'output':[]}; br=[]; sinfo=[]
    for i,sid in enumerate(sids):
        n,x,y,a,m=convert_session(sid,records,db,aliases,stim_to_idx,speed_q)
        data['neural'].append(n); data['input'].append(x); data['output'].append(y); br.append(a); sinfo.append(m)
        if args.show_processing and i<2: make_plot(sid,n,x,y)
    subjects=sorted({s.split('_')[0] for s in sids}); subjmap={x:i for i,x in enumerate(subjects)}
    data.update(subjects=subjects,subject_idx=np.asarray([subjmap[s.split('_')[0]] for s in sids],dtype=np.int32),
      brain_regions=REGIONS,brain_region_idx=br,
      input_names=['time to sound cue','day of training','time since trial start','reward availability'],
      output_names=['visual stimulus category','licking','position in corridor','running speed quartile'],
      output_values=[stim_values,['not licking','licking'],['0-1 m','1-2 m','2-3 m','3-4 m'],['Q1','Q2','Q3','Q4']],
      metadata={'task_description':'Decode visual stimulus, licking, 1-m corridor position, and running-speed quartile from population deconvolved calcium activity.',
       'time_bin_size':DT_GLOBAL*1000,'temporal_alignment_event':'entry into 4-m visual corridor (first retained running corridor frame)',
       'off_start':0.0,'off_end':None,'source_dataset':'Unsupervised pretraining in biological neural networks',
       'neural_processing':'Suite2p nonnegative deconvolved fluorescence; three imaging planes concatenated; all cells retained',
       'trial_filter':'ft_CorrSpc and ft_move > 0; streams truncated to neural frame count',
       'position_source_units_per_meter':10.0,'speed_quartile_thresholds':speed_q.tolist(),
       'session_info':sinfo,'scan_statistics':scan_stats})
    out=Path(args.outpicklefile); print(f'Writing {out}...',flush=True)
    with open(out,'wb') as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Wrote {out.stat().st_size/2**30:.3f} GiB in {time.time()-t0:.1f}s',flush=True)
if __name__=='__main__': main()
