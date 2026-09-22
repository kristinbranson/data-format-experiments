#!/usr/bin/env python3
"""Convert the Zhou et al. VR mesoscope data for time-resolved decoding.

Decisions follow the released analysis code where applicable: Suite2p deconvolved
traces are concatenated over planes; retinotopic IDs are grouped with
utils.neu_area_ID; and only moving samples in the 4-m textured corridor are used.
Native imaging frames (3.17 Hz) are retained because this task requires temporal
lick, cue-time, speed and elapsed-time variables (the paper's 60-bin spatial
interpolation would discard their natural timing).
"""
import argparse, collections, datetime, gc, glob, os, pickle
import numpy as np

ROOT='/app/data'; FS=3.17
REGIONS=['V1','mHV','lHV','aHV','unassigned']

def area_index(ia):
    ia=np.asarray(ia); z=np.full(ia.shape,4,dtype=np.int8)
    z[ia==8]=0
    z[np.isin(ia,[0,1,2,9])]=1
    z[np.isin(ia,[5,6])]=2
    z[np.isin(ia,[3,4])]=3
    return z

def sid_date(s): return datetime.datetime.strptime('_'.join(s.split('_')[1:4]),'%Y_%m_%d').date()

def load_behavior_sessions():
    out={}; source={}
    # Stable precedence. Duplicate entries represent alternate paper comparisons;
    # frame-aligned behavior itself is the same recording.
    for p in sorted(glob.glob(ROOT+'/beh/Beh_*.npy')):
        try: d=np.load(p,allow_pickle=True).item()
        except Exception: continue
        for sid,b in d.items():
            if isinstance(b,dict) and 'ft' in b and sid not in out:
                out[sid]=b; source[sid]=os.path.basename(p)
    return out,source

def trial_mask(b,tr,n):
    tri=np.asarray(b['ft_trInd'])[:n]
    pos=np.asarray(b['ft_Pos'])[:n]
    moving=np.asarray(b['ft_isMoving'],bool)[:n]
    corr=np.asarray(b['ft_CorrSpc'],bool)[:n]
    # Corridor positions are represented as 0..40 (source position unit = 0.1 m).
    return (tri==tr) & moving & corr & np.isfinite(pos) & (pos>=0) & (pos<40)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default='/app/converted_data.pkl')
    ap.add_argument('--max-sessions',type=int,default=None,help='diagnostic only')
    ap.add_argument('--max-neurons',type=int,default=2000)
    a=ap.parse_args()
    beh,bsrc=load_behavior_sessions()
    spk={os.path.basename(p).replace('_neural_data.npy',''):p for p in glob.glob(ROOT+'/spk/*_neural_data.npy')}
    sids=sorted(set(beh)&set(spk),key=lambda s:(s.split('_')[0],sid_date(s),s))
    if a.max_sessions: sids=sids[:a.max_sessions]
    subjects=sorted({s.split('_')[0] for s in sids}); subjmap={s:i for i,s in enumerate(subjects)}
    firstdate={m:min(sid_date(s) for s in sids if s.split('_')[0]==m) for m in subjects}

    # First pass: global category vocabulary and running-speed quartiles over all
    # retained moving corridor frames (not per-session, ensuring common classes).
    categories=set(); speeds=[]
    for sid in sids:
        b=beh[sid]; categories.update(map(str,np.asarray(b['WallName']).reshape(-1)))
        n=len(b['ft']); tri=np.asarray(b['ft_trInd'])[:n]; pos=np.asarray(b['ft_Pos'])[:n]
        ok=np.isfinite(tri)&np.asarray(b['ft_isMoving'],bool)[:n]&np.asarray(b['ft_CorrSpc'],bool)[:n]&np.isfinite(pos)&(pos>=0)&(pos<40)
        v=np.asarray(b['ft_RunSpeed'],float)[:n][ok]; speeds.append(v[np.isfinite(v)])
    categories=sorted(categories); catmap={x:i for i,x in enumerate(categories)}
    speed_edges=np.quantile(np.concatenate(speeds),[.25,.5,.75]); del speeds
    print('sessions',len(sids),'subjects',len(subjects),'categories',categories,'speed quartiles',speed_edges,flush=True)

    neural=[]; inputs=[]; outputs=[]; region_idx=[]; session_info=[]
    for si,sid in enumerate(sids):
        b=beh[sid]
        obj=np.load(spk[sid],allow_pickle=True).item()['spks']
        n=min(len(b['ft']),min(x.shape[1] for x in obj))
        X=np.concatenate([x[:,:n] for x in obj],axis=0)
        del obj
        rp=glob.glob(ROOT+'/retinotopy/'+('_'.join(sid.split('_')[:4]))+'_trans.npz')
        if rp:
            ia=np.load(rp[0])['iarea']; ri=area_index(ia)
        else: ri=np.full(X.shape[0],4,dtype=np.int8)
        if len(ri)!=X.shape[0]:
            print('WARNING region/neuron mismatch',sid,len(ri),X.shape[0],flush=True)
            ri=np.resize(ri,X.shape[0]).astype(np.int8)
        # Deterministic proportional stratified sample, preserving every region.
        if a.max_neurons and X.shape[0]>a.max_neurons:
            rng=np.random.default_rng(0)
            chosen=[]
            for r in range(len(REGIONS)):
                ids=np.flatnonzero(ri==r)
                k=int(round(a.max_neurons*len(ids)/len(ri)))
                if len(ids) and k==0: k=1
                chosen.extend(rng.choice(ids,min(k,len(ids)),replace=False).tolist())
            if len(chosen)>a.max_neurons: chosen=chosen[:a.max_neurons]
            if len(chosen)<a.max_neurons:
                rem=np.setdiff1d(np.arange(len(ri)),chosen,assume_unique=False)
                chosen.extend(rng.choice(rem,a.max_neurons-len(chosen),replace=False).tolist())
            chosen=np.sort(np.asarray(chosen)); X=X[chosen]; ri=ri[chosen]
        ft=np.asarray(b['ft'],float)[:n]
        # Assign each lick to its nearest imaging frame.
        lick=np.zeros(n,dtype=np.int8)
        lt=np.asarray(b['LickTime']).reshape(-1)
        if lt.size and np.issubdtype(lt.dtype,np.number):
            lt=lt[np.isfinite(lt.astype(float))].astype(float)
            q=np.searchsorted(ft,lt); q=np.clip(q,1,n-1)
            q-=((lt-ft[q-1]) <= (ft[q]-lt)).astype(int)
            lick[np.unique(q)]=1
        sn=[]; ii=[]; oo=[]; kept=[]
        day=float((sid_date(sid)-firstdate[sid.split('_')[0]]).days)
        for tr in range(int(b['ntrials'])):
            mask=trial_mask(b,tr,n); ix=np.flatnonzero(mask)
            if ix.size<2: continue
            name=str(np.asarray(b['WallName'])[tr]); pos=np.asarray(b['ft_Pos'],float)[:n][ix]
            speed=np.asarray(b['ft_RunSpeed'],float)[:n][ix]
            t0=float(np.asarray(b['Trial_start_time'])[tr]); cue=float(np.asarray(b['SoundTime'])[tr])
            elapsed=(ft[ix]-t0)*86400.0; tocue=(cue-ft[ix])*86400.0
            inp=np.vstack([tocue,np.full(ix.size,day),elapsed,np.full(ix.size,int(np.asarray(b['isRew'])[tr]))]).astype(np.float32)
            out=np.vstack([np.full(ix.size,catmap[name]),lick[ix],np.clip((pos//10).astype(int),0,3),np.digitize(speed,speed_edges)]).astype(np.int8)
            sn.append(np.asarray(X[:,ix],dtype=np.float32)); ii.append(inp); oo.append(out); kept.append(tr)
        del X
        if len(sn)<2:
            print('skip',sid,'fewer than 2 usable trials',flush=True); continue
        neural.append(sn); inputs.append(ii); outputs.append(oo); region_idx.append(ri)
        session_info.append({'session_id':sid,'behavior_source':bsrc[sid],'recording_date':str(sid_date(sid)),'training_day_elapsed_from_first_recording':day,'source_trials':int(b['ntrials']),'retained_trials':len(sn),'retained_trial_indices':kept,'neurons_retained':int(len(ri))})
        print(si+1,'/',len(sids),sid,'neurons',len(ri),'trials',len(sn),'timepoints',sum(x.shape[1] for x in sn),flush=True)
        gc.collect()
    used_subjects=sorted({x['session_id'].split('_')[0] for x in session_info}); um={s:i for i,s in enumerate(used_subjects)}
    data={'neural':neural,'input':inputs,'output':outputs,'subjects':used_subjects,
          'subject_idx':np.asarray([um[x['session_id'].split('_')[0]] for x in session_info],dtype=np.int32),
          'brain_regions':REGIONS,'brain_region_idx':region_idx,
          'input_names':['time_to_sound_cue_s','training_day_elapsed','time_since_trial_start_s','reward_available'],
          'output_names':['visual_stimulus_category','licking','corridor_position_bin','running_speed_quartile'],
          'output_values':[categories,['not_licking','licking'],['0-1 m','1-2 m','2-3 m','3-4 m'],['Q1 (slowest)','Q2','Q3','Q4 (fastest)']],
          'metadata':{'task_description':'Decode visual corridor identity, licking, 1-m corridor position, and global running-speed quartile from visual-cortex deconvolved activity.','time_bin_size':1000.0/FS,'temporal_alignment_event':'trial start (entry into textured corridor)','off_start':0.0,'off_end':None,'native_imaging_rate_hz':FS,'neural_signal':'Suite2p non-negative deconvolved fluorescence trace','frame_filter':'ft_isMoving AND ft_CorrSpc AND 0 <= ft_Pos < 40; matches paper restriction to running in the textured 4-m corridor','alignment':'Behavior arrays are supplied at neural-frame timestamps; neural and behavior streams truncated to shared minimum length (typically neural is one frame shorter).','position_conversion':'Source ft_Pos units are 0.1 m; bins [0,10), [10,20), [20,30), [30,40).','speed_quartile_edges':speed_edges.tolist(),'speed_quartile_scope':'all retained timepoints across included sessions','lick_binning':'each lick assigned to nearest native imaging frame','training_day_definition':'elapsed calendar days from each subject first included imaging recording','neuron_selection':f'deterministic region-stratified sample capped at {a.max_neurons} neurons/session; reference plane concatenation retained before sampling','excluded_neural_sessions_without_behavior':sorted(set(spk)-set(beh)),'session_info':session_info}}
    with open(a.output,'wb') as f: pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
    print('saved',a.output,os.path.getsize(a.output)/1e9,'GB')
if __name__=='__main__': main()
