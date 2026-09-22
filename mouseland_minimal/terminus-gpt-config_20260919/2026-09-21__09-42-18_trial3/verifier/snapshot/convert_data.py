#!/usr/bin/env python3
"""Convert Zhong et al. imaging data to the neural-decoder schema.

Decisions matching the source repository:
* Sessions are those listed in Imaging_Exp_info with both behavior and neural files.
* All three Suite2p `spks` arrays are concatenated over neurons, exactly as
  code/utils.py:load_spk. These are non-negative deconvolved fluorescence traces.
* Suite2p cell classification is the supplied curation; no post-hoc selectivity filter.
* Trials are aligned to corridor entry using ft_trInd/StartFr and end at corridor exit
  (GrayFr). Native imaging is ~3.2 Hz. We average in non-overlapping 1 s bins to make
  the full dataset tractable while retaining temporal cue/lick structure.
* Stored corridor coordinates span 60 units = 4 m for this decoder specification;
  positions are mapped linearly to four 1 m bins.
"""
import os, glob, pickle, re, gc
import numpy as np
ROOT='/app/data'; BIN_S=1.0

def sid_of(r): return f"{r['mname']}_{r['datexp']}_{r['blk']}" + (f"_{r['stimtype']}" if 'stimtype' in r else '')
def base_sid(r): return f"{r['mname']}_{r['datexp']}_{r['blk']}"
def date_ordinal(date):
    return float(np.datetime64(date.replace('_','-'),'D').astype(int))
def region_indices(sid,n):
    p=sid.rsplit('_',1)[0]
    f=os.path.join(ROOT,'retinotopy',p+'_trans.npz')
    if not os.path.exists(f): return np.zeros(n,dtype=np.int16)
    a=np.asarray(np.load(f)['iarea']).ravel()
    # Repository neu_area_ID: 1=V1, 2=medial, 3=anterior, 4=lateral; 0=unassigned.
    out=np.zeros(len(a),dtype=np.int16)
    out[a==1]=1; out[np.isin(a,[2,3])]=2; out[np.isin(a,[4,5])]=3; out[np.isin(a,[6,7])]=4
    if len(out)!=n:
        # Retinotopy corresponds to concatenated planes; guard malformed metadata.
        z=np.zeros(n,dtype=np.int16); z[:min(n,len(out))]=out[:min(n,len(out))]; out=z
    return out

info=np.load(os.path.join(ROOT,'beh','Imaging_Exp_info.npy'),allow_pickle=True).item()
# Behavior keyed by both full keys and base session keys. Preserve first occurrence;
# duplicate analysis files contain the same raw frame streams.
beh={}; source_group={}; rowmap={}
for group,rows in info.items():
    bf=os.path.join(ROOT,'beh','Beh_'+group+'.npy')
    if not os.path.exists(bf): continue
    bd=np.load(bf,allow_pickle=True).item()
    for r in rows:
        full=sid_of(r); base=base_sid(r)
        key=full if full in bd else base
        if key in bd and base not in beh:
            beh[base]=bd[key]; source_group[base]=group; rowmap[base]=r
spkfiles={os.path.basename(f).replace('_neural_data.npy',''):f for f in glob.glob(os.path.join(ROOT,'spk','*_neural_data.npy'))}
sessions=sorted(set(beh)&set(spkfiles))
subject_day0={}
for s in sessions:
    mouse=s.split('_')[0]; d=date_ordinal(rowmap[s]['datexp'])
    subject_day0[mouse]=min(subject_day0.get(mouse,d),d)
print('usable sessions',len(sessions),flush=True)
# Global speed quartiles from the same valid corridor frames included below.
speeds=[]
for s in sessions:
    b=beh[s]; tri=np.asarray(b['ft_trInd']); pos=np.asarray(b['ft_Pos']); v=np.asarray(b['ft_RunSpeed'])
    ok=np.isfinite(tri)&np.isfinite(pos)&np.isfinite(v)&np.asarray(b['ft_CorrSpc'],dtype=bool)&(pos>=0)&(pos<float(b['Texture_Length']))
    speeds.append(v[ok].astype(np.float32))
q=np.quantile(np.concatenate(speeds),[.25,.5,.75]).astype(np.float32); del speeds
print('speed quartiles',q,flush=True)
# Rewarded corridor is an experimental property. Infer the rewarded visual category
# only where rewards exist; task sessions consistently identify it. Unsupervised/naive
# sessions have no rewarded corridor, hence all zeros.
reward_stim={}
for s in sessions:
    b=beh[s]; st=np.asarray(b['TrialStim']); rw=np.asarray(b['RewardFr'])
    counts={x:int(np.isfinite(rw[st==x]).sum()) for x in np.unique(st)}
    reward_stim[s]=max(counts,key=counts.get) if counts and max(counts.values())>0 else None

neural=[]; inputs=[]; outputs=[]; bridx=[]; subj=[]; session_info=[]
stim_names=sorted({str(x) for s in sessions for x in np.unique(beh[s]['TrialStim'])}); stim_id={x:i for i,x in enumerate(stim_names)}
for si,s in enumerate(sessions):
    b=beh[s]; raw=np.load(spkfiles[s],allow_pickle=True).item()['spks']; spk=np.concatenate(raw,axis=0); del raw
    nfr=min(spk.shape[1],len(b['ft_trInd']))
    tri=np.asarray(b['ft_trInd'][:nfr]); pos=np.asarray(b['ft_Pos'][:nfr]); speed=np.asarray(b['ft_RunSpeed'][:nfr]); ft=np.asarray(b['ft'][:nfr])
    # MATLAB datenums -> elapsed seconds.
    tsec=(ft-ft[0])*86400.0
    ns=[]; ins=[]; outs=[]; kept=[]
    ntr=int(b['ntrials']); trialstim=np.asarray(b['TrialStim']); sound=np.asarray(b['SoundFr']); lick=np.asarray(b['LickFr'])
    for tr in range(min(ntr,len(trialstim))):
        ix=np.where((tri==tr)&np.asarray(b['ft_CorrSpc'][:nfr],dtype=bool)&np.isfinite(pos)&(pos>=0)&(pos<float(b['Texture_Length'])))[0]
        if ix.size<2: continue
        t0=tsec[ix[0]]; rel=tsec[ix]-t0; nb=max(1,int(np.floor(rel[-1]/BIN_S))+1); bins=np.minimum((rel/BIN_S).astype(int),nb-1)
        # Cue event is a fractional global imaging-frame index.
        cue_rel=(np.interp(sound[tr],np.arange(nfr),tsec)-t0) if tr<len(sound) and np.isfinite(sound[tr]) else np.nan
        N=np.empty((spk.shape[0],nb),np.float32); P=np.empty(nb,np.float32); V=np.empty(nb,np.float32)
        L=np.zeros(nb,np.int16)
        for j in range(nb):
            jj=ix[bins==j]
            if jj.size: N[:,j]=spk[:,jj].mean(1,dtype=np.float32); P[j]=np.nanmean(pos[jj]); V[j]=np.nanmean(speed[jj])
            else: N[:,j]=N[:,j-1] if j else 0; P[j]=P[j-1] if j else 0; V[j]=V[j-1] if j else 0
        # Any lick in a temporal bin -> binary licking.
        lf=lick[np.isfinite(lick)] if lick.size else lick
        if lf.size:
            lt=np.interp(lf,np.arange(nfr),tsec)-t0; lb=(lt/BIN_S).astype(int); lb=lb[(lb>=0)&(lb<nb)]; L[np.unique(lb)]=1
        elapsed=(np.arange(nb,dtype=np.float32)+.5)*BIN_S
        tcue=(cue_rel-elapsed).astype(np.float32) if np.isfinite(cue_rel) else np.full(nb,np.nan,np.float32)
        day=np.full(nb,date_ordinal(rowmap[s]['datexp'])-subject_day0[s.split('_')[0]]+1.0,np.float32)
        rewarded=np.full(nb,int(reward_stim[s] is not None and str(trialstim[tr])==reward_stim[s]),np.float32)
        inp=np.vstack([tcue,day,elapsed,rewarded]).astype(np.float32)
        cat=np.full(nb,stim_id[str(trialstim[tr])],np.int16)
        # Linear mapping required by task: 4 equal 1 m bins over each session corridor.
        pcat=np.clip((P/float(b['Texture_Length'])*4).astype(int),0,3).astype(np.int16)
        vcat=np.digitize(V,q,right=False).astype(np.int16)
        out=np.vstack([cat,L,pcat,vcat]).astype(np.int16)
        ns.append(N); ins.append(inp); outs.append(out); kept.append(tr)
    if len(ns)<2: print('skip',s,'too few trials'); continue
    neural.append(ns); inputs.append(ins); outputs.append(outs); bridx.append(region_indices(s,spk.shape[0])); subj.append(s.split('_')[0])
    session_info.append({'session_id':s,'experiment_group':source_group[s],'date':rowmap[s]['datexp'],'n_source_trials':ntr,'trial_indices':kept,'rewarded_stimulus':reward_stim[s]})
    print(si+1,'/',len(sessions),s,spk.shape,'trials',len(ns),'bins',sum(x.shape[1] for x in ns),flush=True)
    del spk; gc.collect()
subjects=sorted(set(subj)); subject_idx=np.array([subjects.index(x) for x in subj],dtype=np.int16)
data={'neural':neural,'input':inputs,'output':outputs,'subjects':subjects,'subject_idx':subject_idx,
 'brain_regions':['unassigned','V1','medial visual areas','anterior visual areas','lateral visual areas'],'brain_region_idx':bridx,
 'input_names':['time to sound cue (s)','day of training (days since subject first session)','time since trial start (s)','reward availability'],
 'output_names':['visual stimulus category','licking','position in corridor','running speed quartile'],
 'output_values':[stim_names,['not licking','licking'],['0-1 m','1-2 m','2-3 m','3-4 m'],['Q1','Q2','Q3','Q4']],
 'metadata':{'task_description':'Head-fixed mice run through visual virtual corridors; decode corridor identity and behavior from deconvolved calcium activity.','time_bin_size':1000.0,'temporal_alignment_event':'trial start / corridor entry','off_start':0.0,'off_end':None,'neural_signal':'Suite2p non-negative deconvolved fluorescence, all supplied planes concatenated','source_frame_rate_hz':'approximately 3.2, variable','temporal_binning':'non-overlapping 1 s means from corridor entry to exit','speed_quartile_edges':q.tolist(),'position_mapping':'Visual-corridor coordinates (ft_CorrSpc; Texture_Length=40 units) mapped to four equal 1 m bins; gray space excluded','session_info':session_info}}
with open('/app/converted_data.pkl','wb') as f: pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
print('saved /app/converted_data.pkl',os.path.getsize('/app/converted_data.pkl')/1e9,'GB')
