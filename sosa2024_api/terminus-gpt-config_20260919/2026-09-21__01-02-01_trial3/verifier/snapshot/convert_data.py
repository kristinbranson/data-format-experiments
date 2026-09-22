#!/usr/bin/env python3
"""Convert reward-relative CA1 NWBs to decoder format. NWB access is pynwb-only."""
import argparse, pickle, time
from pathlib import Path
import numpy as np
from pynwb import NWBHDF5IO

DT=0.1
ZONES=np.array([[80.,100.],[200.,220.],[320.,340.]])

def nearest_idx(t,q):
    i=np.searchsorted(t,q); i=np.clip(i,1,len(t)-1)
    return np.where(q-t[i-1] <= t[i]-q,i-1,i)

def zone_for_trial(pos):
    # The active zone is the fixed zone traversed around reward-zone activation.
    # If no activation (omission), infer session block from position where the native
    # reward_zone signal activates in neighboring trials; caller supplies block fallback.
    return int(np.argmin(np.abs(ZONES.mean(1)-pos)))

def discretize_distance(d):
    y=np.empty(d.shape,np.int64)
    y[d < -50]=0; y[(d>=-50)&(d<-10)]=1; y[(d>=-10)&(d<0)]=2
    y[d==0]=3; y[(d>0)&(d<=10)]=4; y[(d>10)&(d<=50)]=5; y[d>50]=6
    return y

def process_file(fn, show=False):
    t0=time.time()
    with NWBHDF5IO(str(fn),'r',load_namespaces=True) as io:
        nwb=io.read(); beh=nwb.processing['behavior']['BehavioralTimeSeries'].time_series
        bt=np.asarray(beh['trial_start'].timestamps[:],float)
        B={k:np.asarray(v.data[:]) for k,v in beh.items() if v.data.shape[0]==len(bt)}
        starts=np.flatnonzero(B['trial_start']>0); tele=np.flatnonzero(B['teleport']>0)
        pairs=[]
        for j,s in enumerate(starts):
            e0=tele[tele>=s]
            if len(e0) and (j+1==len(starts) or e0[0]<starts[j+1]): pairs.append((int(s),int(e0[0])))
        if len(pairs)<2: raise ValueError(f'{fn}: fewer than 2 complete trials')
        rew_t=np.asarray(beh['Reward'].timestamps[:],float)
        # Each RoiResponseSeries references its own rows of the segmentation table.
        # Single-plane sessions have one series; multi-plane sessions have plane0/plane1.
        plane_data=[]; rates=[]
        dec=nwb.processing['ophys']['Deconvolved'].roi_response_series
        n_planes=len(dec)
        for plane_name,rr in sorted(dec.items()):
            region=np.asarray(rr.rois.data[:],dtype=int)
            table=rr.rois.table
            iscell=np.asarray(table['iscell'].data[:])[:,0]>0
            valid=iscell[region]
            nt=rr.data.shape[0]; nominal_rate=float(rr.rate)
            # m17/m18 planes are acquired interleaved at ~31 Hz aggregate, giving
            # ~15.5 Hz per plane (Methods). NWB stores aggregate rate on each series.
            if rr.timestamps is not None:
                neural_t=np.asarray(rr.timestamps[:],float)
                effective_rate=1.0/np.median(np.diff(neural_t))
            else:
                effective_rate=nominal_rate/n_planes
                neural_t=float(rr.starting_time)+np.arange(nt)/effective_rate
            rates.append(effective_rate)
            neural_all=np.asarray(rr.data[:, :],np.float32)[:,valid]
            plane_data.append((plane_name,neural_t,neural_all))
        n_valid=sum(x[2].shape[1] for x in plane_data)
        # Retain only trials fully covered by every neural plane. Behavioral recording
        # can continue after imaging ends; nearest-neighbor extrapolation is invalid.
        neural_start=max(x[1][0] for x in plane_data)
        neural_end=min(x[1][-1] for x in plane_data)
        n_pairs_raw=len(pairs)
        pairs=[(s,e) for s,e in pairs if bt[s]>=neural_start and bt[e]<=neural_end]
        excluded_no_neural=n_pairs_raw-len(pairs)
        if len(pairs)<2: raise ValueError(f'{fn}: fewer than 2 fully imaging-covered trials')
        if excluded_no_neural:
            print(f'{fn.name}: excluded {excluded_no_neural} trials outside complete neural coverage',flush=True)
        # Establish active zone per trial from native reward_zone activation position.
        raw_zone=[]
        for s,e in pairs:
            hit=np.flatnonzero(B['reward_zone'][s:e+1]>0)
            if len(hit): raw_zone.append(zone_for_trial(float(np.median(B['position'][s+hit]))))
            else: raw_zone.append(-1)
        # Fill omission/no-activation labels from nearest labeled trial; blocks are contiguous.
        good=np.flatnonzero(np.asarray(raw_zone)>=0)
        if not len(good): raise ValueError(f'{fn}: cannot infer reward zone')
        zones=np.asarray(raw_zone)
        bad=np.flatnonzero(zones<0)
        zones[bad]=zones[good[np.argmin(abs(good[:,None]-bad),axis=0)]] if len(bad) else zones[bad]
        neural_trials=[]; inputs=[]; outputs=[]; outcomes=[]
        for ti,(s,e) in enumerate(pairs):
            t_start,t_end=bt[s],bt[e]
            centers=t_start+DT/2+np.arange(max(1,int(np.floor((t_end-t_start)/DT))))*DT
            centers=centers[centers<=t_end]
            # Mean native deconvolved samples in each temporal bin.
            binned_planes=[]
            for plane_name,neural_t,neural_all in plane_data:
                ni=np.searchsorted(neural_t,centers-DT/2); nj=np.searchsorted(neural_t,centers+DT/2)
                xp=np.empty((neural_all.shape[1],len(centers)),np.float32)
                for k,(a,b) in enumerate(zip(ni,nj)):
                    if b>a: xp[:,k]=neural_all[a:b].mean(0)
                    else: xp[:,k]=neural_all[nearest_idx(neural_t,np.array([centers[k]]))[0]]
                binned_planes.append(xp)
            X=np.concatenate(binned_planes,axis=0)
            pos=np.interp(centers,bt,B['position']).astype(np.float32)
            speed=np.interp(centers,bt,B['speed']).astype(np.float32)
            jj=nearest_idx(bt,centers)
            lick=(B['lick'][jj]>0).astype(np.int64)
            env=int(round(float(B['environment'][s]))); env=max(0,min(1,env))
            trialnum=float(B['trial number'][s]); rewarded=int(np.any((rew_t>=t_start)&(rew_t<=t_end)))
            prev=outcomes[-1] if outcomes else 0; outcomes.append(rewarded)
            z=int(zones[ti]); lo,hi=ZONES[z]
            dist=np.where(pos<lo,pos-lo,np.where(pos>hi,pos-hi,0.)).astype(np.float32)
            poscls=np.digitize(pos,[90,180,270,360],right=False).astype(np.int64)
            speedcls=np.digitize(speed,[2,10,20,40],right=False).astype(np.int64)
            inp=np.vstack([centers-t_start,np.full(len(centers),env),np.full(len(centers),trialnum),np.full(len(centers),prev)]).astype(np.float32)
            out=np.vstack([discretize_distance(dist),poscls,speedcls,lick,np.full(len(centers),z),np.full(len(centers),rewarded)]).astype(np.int64)
            assert X.shape[1]==inp.shape[1]==out.shape[1] and np.isfinite(X).all()
            neural_trials.append(X); inputs.append(inp); outputs.append(out)
        sid=f'{nwb.subject.subject_id}_{nwb.session_id or fn.stem}'
        info=dict(session_id=sid,file=str(fn),n_trials=len(pairs),n_neurons=int(n_valid),rate=float(rates[0]),plane_rates=rates,n_planes=len(plane_data),zones=np.bincount(zones,minlength=3).tolist(),rewarded=int(sum(outcomes)),excluded_no_neural=excluded_no_neural)
    print(f'{sid}: {info["n_trials"]} trials, {info["n_neurons"]} cells, {time.time()-t0:.2f}s',flush=True)
    return neural_trials,inputs,outputs,nwb.subject.subject_id,info

def plot_session(info,neural,inp,out):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(5,1,figsize=(12,12),sharex=True); t=inp[0][0]
    ax[0].imshow(neural[0],aspect='auto',extent=[t[0],t[-1],0,neural[0].shape[0]]); ax[0].set_ylabel('cell')
    ax[1].plot(t,out[0][1]); ax[1].set_ylabel('position class')
    ax[2].plot(t,out[0][0]); ax[2].set_ylabel('distance class')
    ax[3].plot(t,out[0][2]); ax[3].set_ylabel('speed class')
    ax[4].step(t,out[0][3]); ax[4].set_ylabel('lick'); ax[4].set_xlabel('s from trial start')
    fig.suptitle(info['session_id']); fig.tight_layout(); fig.savefig(f"/app/processing_{info['session_id']}.png",dpi=140); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('output'); g=ap.add_mutually_exclusive_group(); g.add_argument('--full',action='store_true'); g.add_argument('--sample',action='store_true'); ap.add_argument('--show-processing',action='store_true'); a=ap.parse_args()
    files=sorted(Path('/app/data').rglob('*.nwb')); files=files[:2] if a.sample else files
    neural=[]; inputs=[]; outputs=[]; subjects=[]; sidx=[]; br=[]; infos=[]
    start=time.time()
    for i,fn in enumerate(files):
        N,I,O,sub,info=process_file(fn,a.show_processing); neural.append(N);inputs.append(I);outputs.append(O);infos.append(info)
        if sub not in subjects: subjects.append(sub)
        sidx.append(subjects.index(sub)); br.append(np.zeros(N[0].shape[0],np.int64))
        if a.show_processing and i<2: plot_session(info,N,I,O)
    data={'neural':neural,'input':inputs,'output':outputs,'subjects':subjects,'subject_idx':np.asarray(sidx,np.int64),'brain_regions':['CA1'],'brain_region_idx':br,
      'input_names':['time from trial start (s)','environment type (ENV1=0, ENV2=1)','trial number','previous trial outcome (omitted=0, rewarded=1)'],
      'output_names':['distance to reward zone','absolute position','speed','lick','reward zone location','reward outcome'],
      'output_values':[['< -50 cm','-50 to -10 cm','-10 to <0 cm','in reward zone (0 cm)','>0 to +10 cm','+10 to +50 cm','> +50 cm'],['<90 cm','90-180 cm','180-270 cm','270-360 cm','>360 cm'],['<2 cm/s','2-10 cm/s','10-20 cm/s','20-40 cm/s','>40 cm/s'],['no','yes'],['A','B','C'],['no','yes']],
      'metadata':{'task_description':'Head-fixed mice traverse a 450 cm virtual corridor with blockwise reward zones A/B/C and omission trials. Decode behavior and task variables from CA1 deconvolved calcium events.','time_bin_size':DT*1000,'temporal_alignment_event':'explicit trial_start pulse','off_start':0.0,'off_end':None,'neural_signal':'Suite2p deconvolved calcium events; iscell curated; mean in 100 ms bins','reward_zones_cm':{'A':[80,100],'B':[200,220],'C':[320,340]},'session_info':infos}}
    with open(a.output,'wb') as f: pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
    print(f'Wrote {a.output}; sessions={len(files)}, trials={sum(map(len,neural))}, neurons(session-sum)={sum(x[0].shape[0] for x in neural)}, elapsed={time.time()-start:.1f}s',flush=True)
if __name__=='__main__': main()
