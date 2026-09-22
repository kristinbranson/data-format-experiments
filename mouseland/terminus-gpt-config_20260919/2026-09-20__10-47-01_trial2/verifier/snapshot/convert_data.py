#!/usr/bin/env python3
"""Convert Zhong et al. imaging data to decoder-compatible trial lists.

Usage: python -u /app/convert_data.py OUT.pkl [--full|--sample] [--show-processing]
"""
import argparse, gc, json, pickle, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np

DATA = Path('/app/data')
SPEED_EDGES = np.array([0.0, 8.32701545, 30.15676260], dtype=np.float32)
REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'unmapped']
STIMULI = ['circle1','circle2','circle3','leaf1','leaf1_swap1','leaf1_swap2',
           'leaf2','leaf3','rock1','rock2','wood1','wood1_swap1','wood1_swap2','wood2','wood5']


def load_catalog():
    """Return one canonical behavior record and experiment labels per physical session."""
    info = np.load(DATA/'beh/Imaging_Exp_info.npy', allow_pickle=True).item()
    beh, labels = {}, defaultdict(list)
    for exp, rows in info.items():
        d = np.load(DATA/f'beh/Beh_{exp}.npy', allow_pickle=True).item()
        for r in rows:
            sid = f"{r['mname']}_{r['datexp']}_{r['blk']}"
            keys = [sid, sid + (f"_{r['stimtype']}" if 'stimtype' in r else '')]
            hit = next((k for k in keys if k in d), None)
            if hit is None:
                raise KeyError(f'No behavior key for {sid} in {exp}')
            if sid in beh:
                # Exploration established duplicate memberships are identical.
                if int(beh[sid]['ntrials']) != int(d[hit]['ntrials']):
                    raise ValueError(f'Conflicting duplicate behavior for {sid}')
            else:
                beh[sid] = d[hit]
            if exp not in labels[sid]: labels[sid].append(exp)
    neural_ids = {p.name.removesuffix('_neural_data.npy') for p in (DATA/'spk').glob('*_neural_data.npy')}
    if neural_ids != set(beh):
        raise ValueError(f'Neural/behavior IDs differ: neural-only={neural_ids-set(beh)}, behavior-only={set(beh)-neural_ids}')
    return beh, labels


def day_offsets(session_ids):
    dates = {s: datetime.strptime('_'.join(s.split('_')[1:4]), '%Y_%m_%d') for s in session_ids}
    first = {}
    for s,d in dates.items():
        mouse=s.split('_')[0]; first[mouse]=min(first.get(mouse,d),d)
    return {s: float((d-first[s.split('_')[0]]).days) for s,d in dates.items()}


def region_indices(sid, nneurons):
    rp = DATA/'retinotopy'/f"{sid.rsplit('_',1)[0]}_trans.npz"
    with np.load(rp) as z: ia = np.asarray(z['iarea'])
    if len(ia) != nneurons: raise ValueError(f'{sid}: retinotopy {len(ia)} != neurons {nneurons}')
    out=np.full(nneurons,4,dtype=np.int16)
    out[ia==8]=0; out[np.isin(ia,[0,1,2,9])]=1; out[np.isin(ia,[5,6])]=2; out[np.isin(ia,[3,4])]=3
    return out


def make_plot(sid, trials, inputs, outputs):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n=min(4,len(trials)); fig,axs=plt.subplots(n,4,figsize=(16,3*n),squeeze=False)
    for j in range(n):
        neu=trials[j]; inp=inputs[j]; out=outputs[j]; t=inp[2]
        axs[j,0].imshow(neu[:min(100,len(neu))].astype(np.float32),aspect='auto',interpolation='nearest')
        axs[j,0].set_title(f'trial {j}: first 100 neural traces')
        axs[j,1].plot(t,inp[0],label='time to cue'); axs[j,1].plot(t,t,label='since start'); axs[j,1].axhline(0,c='k',lw=.5); axs[j,1].legend()
        axs[j,2].step(t,out[2],where='mid',label='position bin'); axs[j,2].step(t,out[3],where='mid',label='speed bin'); axs[j,2].legend()
        axs[j,3].step(t,out[1],where='mid',label='lick'); axs[j,3].set_ylim(-.1,1.1); axs[j,3].legend()
    fig.suptitle(sid); fig.tight_layout(); fig.savefig(f'/app/processing_{sid}.png',dpi=130); plt.close(fig)


def convert_session(sid, b, day, show=False):
    t0=time.time(); raw=np.load(DATA/'spk'/f'{sid}_neural_data.npy',allow_pickle=True).item()['spks']
    nfr=min(a.shape[1] for a in raw)
    if any(a.dtype != np.float32 for a in raw): raise TypeError(f'{sid}: expected float32 spks')
    spk=np.concatenate(raw,axis=0); del raw
    nfr=min(nfr,spk.shape[1],len(b['ft_trInd']))
    spk=spk[:,:nfr]
    tri=np.asarray(b['ft_trInd'])[:nfr]; pos=np.asarray(b['ft_Pos'],float)[:nfr]
    speed=np.asarray(b['ft_RunSpeed'],float)[:nfr]; ft=np.asarray(b['ft'],float)[:nfr]*86400.0
    dt=float(np.median(np.diff(ft)))
    wall=np.asarray(b['WallName']).astype(str); rew=np.asarray(b['isRew'],bool); sound=np.asarray(b['SoundFr'],float)
    lickfr=np.asarray(b['LickFr'],float); licktr=np.asarray(b['LickTrind'],int)
    stim_to_idx={x:i for i,x in enumerate(STIMULI)}
    neural=[]; inputs=[]; outputs=[]
    for tr in range(int(b['ntrials'])):
        ix=np.flatnonzero((tri==tr)&np.isfinite(pos)&(pos>=0)&(pos<40))
        if len(ix)<2: raise ValueError(f'{sid} trial {tr}: only {len(ix)} corridor frames')
        # Actual timestamps can have small jitter; re-zero at corridor entry.
        elapsed=(ft[ix]-ft[ix[0]]).astype(np.float32)
        cue=((sound[tr]-ix.astype(float))*dt).astype(np.float32)
        inp=np.vstack([cue, np.full(len(ix),day,np.float32), elapsed,
                       np.full(len(ix),float(rew[tr]),np.float32)]).astype(np.float32)
        if wall[tr] not in stim_to_idx: raise ValueError(f'Unknown stimulus {wall[tr]}')
        lick=np.zeros(len(ix),dtype=np.int16)
        for ev in lickfr[licktr==tr]:
            k=int(np.argmin(np.abs(ix.astype(float)-ev)))
            # Keep only events whose nearest native frame lies in this retained corridor.
            if abs(float(ix[k])-float(ev)) <= 1.0: lick[k]=1
        pbin=np.clip(np.floor(pos[ix]/10.0),0,3).astype(np.int16)
        sbin=np.searchsorted(SPEED_EDGES,speed[ix],side='right').astype(np.int16)
        out=np.vstack([np.full(len(ix),stim_to_idx[wall[tr]],np.int16),lick,pbin,sbin])
        neural.append(np.asarray(spk[:,ix],dtype=np.float16,order='C')); inputs.append(inp); outputs.append(out)
    ridx=region_indices(sid,spk.shape[0]); del spk; gc.collect()
    if show: make_plot(sid,neural,inputs,outputs)
    print(f'{sid}: neurons={len(ridx):,}, trials={len(neural):,}, timepoints={sum(x.shape[1] for x in neural):,}, {time.time()-t0:.2f}s',flush=True)
    return neural,inputs,outputs,ridx,dt


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('outpicklefile'); g=ap.add_mutually_exclusive_group(); g.add_argument('--full',action='store_true'); g.add_argument('--sample',action='store_true'); ap.add_argument('--show-processing',action='store_true'); args=ap.parse_args()
    start=time.time(); beh,labels=load_catalog(); ids=sorted(beh)
    if args.sample:
        # Two deterministic but behaviorally informative sessions: unsupervised + rewarded task.
        preferred=['DR10_2022_07_12_1','TX60_2021_06_07_1']
        ids=[x for x in preferred if x in beh]
        if len(ids)<2: ids=sorted(beh)[:2]
    days=day_offsets(sorted(beh)); subjects=sorted({x.split('_')[0] for x in ids}); subidx={x:i for i,x in enumerate(subjects)}
    neural=[]; inp=[]; out=[]; regions=[]; dts=[]; sinfo=[]
    def job(item):
        i,sid=item
        result=convert_session(sid,beh[sid],days[sid],args.show_processing and i<2)
        return sid,result
    items=list(enumerate(ids))
    if args.sample or len(items)<3:
        results=map(job,items)
    else:
        # NumPy I/O, slicing, and casting release the GIL. Three bounded workers reduce
        # wall time without multiprocessing copies of multi-GB return objects.
        from concurrent.futures import ThreadPoolExecutor
        executor=ThreadPoolExecutor(max_workers=3)
        results=executor.map(job,items)
    for sid,(n,x,y,r,dt) in results:
        neural.append(n); inp.append(x); out.append(y); regions.append(r); dts.append(dt)
        sinfo.append({'session_id':sid,'experiment_types':sorted(labels[sid]),'training_day_elapsed':days[sid],
                      'n_trials':len(n),'n_neurons':len(r),'median_frame_interval_ms':dt*1000})
    if not (args.sample or len(items)<3): executor.shutdown()
    data={'neural':neural,'input':inp,'output':out,'subjects':subjects,
          'subject_idx':np.array([subidx[x.split('_')[0]] for x in ids],dtype=np.int16),
          'brain_regions':REGIONS,'brain_region_idx':regions,
          'input_names':['time to sound cue (s)','day of training (elapsed days)','time since trial start (s)','reward availability'],
          'output_names':['visual stimulus category','licking','position in corridor','running speed quartile'],
          'output_values':[STIMULI,['not licking','licking'],['0-1 m','1-2 m','2-3 m','3-4 m'],
                           ['speed Q1','speed Q2','speed Q3','speed Q4']],
          'metadata':{'task_description':'Decode concrete visual stimulus identity, binary licking, 1-m corridor position, and global running-speed quartile from deconvolved calcium activity.',
                      'time_bin_size':float(np.median(dts)*1000),'temporal_alignment_event':'corridor entry (first imaging frame assigned to trial; ceil StartFr)',
                      'off_start':0.0,'off_end':None,'native_neural_signal':'Suite2p non-negative deconvolved fluorescence (spks)',
                      'neural_storage_dtype':'float16','speed_quartile_edges_cm_s':SPEED_EDGES.tolist(),
                      'position_source_units':'decimeters','corridor_window_m':[0.0,4.0],
                      'training_day_definition':'calendar days elapsed from subject first supplied imaging session',
                      'session_info':sinfo}}
    # Structural assertions before writing.
    assert len(neural)==len(inp)==len(out)==len(regions)==len(ids)
    for ns,xs,ys in zip(neural,inp,out):
        assert len(ns)==len(xs)==len(ys) and len(ns)>=2
        for n,x,y in zip(ns,xs,ys): assert n.shape[1]==x.shape[1]==y.shape[1] and x.shape[0]==4 and y.shape[0]==4
    print(f'Writing {args.outpicklefile} ...',flush=True); t=time.time()
    with open(args.outpicklefile,'wb') as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Wrote {Path(args.outpicklefile).stat().st_size/2**30:.3f} GiB in {time.time()-t:.2f}s; total {time.time()-start:.2f}s',flush=True)

if __name__=='__main__': main()
