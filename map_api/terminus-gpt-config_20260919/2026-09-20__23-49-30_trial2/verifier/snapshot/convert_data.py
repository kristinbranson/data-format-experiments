#!/usr/bin/env python3
"""Convert MAP NWB sessions to the neural-decoder pickle format.

All NWB access uses pynwb. Usage:
  python -u /app/convert_data.py OUT.pkl [--full|--sample] [--show-processing]
"""
import argparse, pickle, time
from pathlib import Path
import numpy as np
from pynwb import NWBHDF5IO

BIN_S=0.050
OFF_START=-2.5
OFF_END=1.5
EDGES_REL=np.arange(OFF_START, OFF_END + BIN_S/2, BIN_S, dtype=np.float64)
CENTERS_REL=(EDGES_REL[:-1]+EDGES_REL[1:])/2
N_TIME=len(CENTERS_REL)
LIKELIHOOD_THRESHOLD=0.90


def arrcol(table, name, dtype=None):
    a=np.asarray(table[name].data[:])
    return a.astype(dtype) if dtype is not None else a


def nearest_indices(sorted_times, query):
    """Nearest indices in monotonically increasing sorted_times."""
    j=np.searchsorted(sorted_times, query, side='left')
    j=np.clip(j, 0, len(sorted_times)-1)
    prev=np.maximum(j-1,0)
    use_prev=np.abs(query-sorted_times[prev]) <= np.abs(sorted_times[j]-query)
    return np.where(use_prev,prev,j)


def select_units_and_trials(nwb, structurally_valid_trials):
    """Select classifier-good units and trials valid for every selected unit.

    is_good_trials is local to each unit's obs_intervals, not necessarily the
    whole behavioral table. Map observation intervals to behavior rows by
    maximum temporal overlap, then intersect valid trial IDs across units.
    """
    cls=arrcol(nwb.units,'classification').astype(str)
    units=np.flatnonzero(cls=='good')
    if units.size==0:
        return units, np.asarray([],dtype=int)
    starts=arrcol(nwb.trials,'start_time',np.float64)
    stops=arrcol(nwb.trials,'stop_time',np.float64)
    common=set(map(int,structurally_valid_trials))
    map_cache={}
    for j in units:
        obs=np.asarray(nwb.units['obs_intervals'][j],dtype=np.float64)
        valid=np.asarray(nwb.units['is_good_trials'][j],dtype=bool)
        if obs.ndim!=2 or obs.shape[0]!=valid.size:
            raise ValueError(f'obs_intervals/is_good_trials mismatch for unit {j}')
        key=(obs.shape,obs.tobytes())
        mapped=map_cache.get(key)
        if mapped is None:
            # Trials and observation intervals are ordered. Candidate is latest
            # behavioral trial starting before each observation interval ends.
            mapped=np.searchsorted(starts,obs[:,1],side='left')-1
            mapped=np.clip(mapped,0,len(starts)-1)
            overlap=np.minimum(stops[mapped],obs[:,1])-np.maximum(starts[mapped],obs[:,0])
            mapped=np.where(overlap>0,mapped,-1).astype(int)
            map_cache[key]=mapped
        common.intersection_update(map(int,mapped[valid & (mapped>=0)]))
        if not common: break
    return units, np.asarray(sorted(common),dtype=int)


def get_tongue(nwb):
    series=nwb.acquisition['BehavioralTimeSeries'].time_series
    keys=sorted(k for k in series if 'TongueTracking' in k)
    if not keys:
        return None,None,None,None,(np.nan,np.nan)
    key='Camera0_side_TongueTracking' if 'Camera0_side_TongueTracking' in keys else keys[0]
    ts=series[key]
    times=np.asarray(ts.timestamps[:],dtype=np.float64)
    data=np.asarray(ts.data[:],dtype=np.float32)
    y=data[:,1]; likelihood=data[:,2]
    visible=np.isfinite(y)&np.isfinite(likelihood)&(likelihood>=LIKELIHOOD_THRESHOLD)
    thresholds=tuple(np.percentile(y[visible],[40,60]).tolist()) if visible.any() else (np.nan,np.nan)
    return key,times,y,likelihood,thresholds


def trial_choice(instruction,outcome):
    if outcome=='ignore': return 2
    if outcome=='hit': return 0 if instruction=='left' else 1
    if outcome=='miss': return 1 if instruction=='left' else 0
    raise ValueError(f'Unknown outcome {outcome!r}')


def convert_session(path, make_plot=False):
    t0=time.time()
    with NWBHDF5IO(str(path),'r',load_namespaces=True) as io:
        nwb=io.read(); tr=nwb.trials
        ev=nwb.acquisition['BehavioralEvents'].time_series
        go=np.asarray(ev['go_start_times'].timestamps[:],dtype=np.float64)
        sample=np.asarray(ev['sample_start_times'].timestamps[:],dtype=np.float64)
        n=min(len(tr),len(go))
        # Structurally valid completed trials: one indexed go and a preceding tone.
        ix=np.searchsorted(sample,go[:n],side='right')-1
        structural_idx=np.flatnonzero(ix>=0)
        if structural_idx.size<2: return None
        units,trial_idx=select_units_and_trials(nwb,structural_idx)
        candidates=units
        if units.size==0 or trial_idx.size<2: return None
        go=go[trial_idx]; tone=sample[ix[trial_idx]]
        names=arrcol(nwb.units,'anno_name').astype(str)[units]
        if np.any(names==''): raise ValueError(f'Blank anatomy among selected units: {path.name}')

        instruction=arrcol(tr,'trial_instruction').astype(str)[trial_idx]
        outcome_s=arrcol(tr,'outcome').astype(str)[trial_idx]
        early_s=arrcol(tr,'early_lick').astype(str)[trial_idx]
        pstart=np.asarray(ev['photostim_start_times'].timestamps[:],dtype=np.float64)
        pstop=np.asarray(ev['photostim_stop_times'].timestamps[:],dtype=np.float64)
        tongue_key,vt,vy,vl,(p40,p60)=get_tongue(nwb)

        # Read each selected ragged spike vector once; histogramming below is in C.
        spikes=[np.asarray(nwb.units['spike_times'][j],dtype=np.float64) for j in units]
        # Vectorize binning across all trials: one searchsorted call per unit.
        # np.diff is along each trial's 81 edges, so no cross-trial bins are introduced.
        edge_matrix=go[:,None]+EDGES_REL[None,:]
        rates=np.empty((len(go),len(units),N_TIME),dtype=np.float32)
        for ui,sp in enumerate(spikes):
            rates[:,ui,:]=np.diff(np.searchsorted(sp,edge_matrix,side='left'),axis=1).astype(np.float32)/BIN_S
        # Remove neural recording-gap trials: no selected unit emitted any
        # spike anywhere in the requested four-second window.
        neural_valid=np.any(rates>0,axis=(1,2))
        dropped_zero_neural=int((~neural_valid).sum())
        trial_idx=trial_idx[neural_valid]
        go=go[neural_valid]; tone=tone[neural_valid]; rates=rates[neural_valid]
        instruction=instruction[neural_valid]; outcome_s=outcome_s[neural_valid]; early_s=early_s[neural_valid]
        if len(go)<2: return None
        neural=[]; inputs=[]; outputs=[]
        tongue_visible=[]
        for k,(g,to) in enumerate(zip(go,tone)):
            fr=rates[k]
            centers=g+CENTERS_REL
            inp=np.empty((2,N_TIME),dtype=np.float32)
            inp[0]=(centers-to).astype(np.float32)
            if len(pstart):
                # true at center if any half-open stimulation interval contains center
                jj=np.searchsorted(pstart,centers,side='right')-1
                valid=jj>=0; stim=np.zeros(N_TIME,dtype=bool)
                stim[valid]=centers[valid] < pstop[jj[valid]]
                inp[1]=stim.astype(np.float32)
            else: inp[1]=0
            out=np.empty((4,N_TIME),dtype=np.int64)
            out[0]=trial_choice(instruction[k],outcome_s[k])
            out[1]={'ignore':0,'miss':1,'hit':2}[outcome_s[k]]
            out[2]={'no early':0,'early':1}[early_s[k]]
            tc=np.full(N_TIME,3,dtype=np.int64); vis=np.zeros(N_TIME,dtype=bool)
            if vt is not None and len(vt):
                q=nearest_indices(vt,centers)
                # Do not extrapolate beyond video coverage; nearest frame must also be temporally close.
                covered=(centers>=vt[0])&(centers<=vt[-1])&(np.abs(vt[q]-centers)<=0.010)
                vis=covered&np.isfinite(vy[q])&np.isfinite(vl[q])&(vl[q]>=LIKELIHOOD_THRESHOLD)&np.isfinite(p40)&np.isfinite(p60)
                yy=vy[q]; tc[vis & (yy<p40)]=0; tc[vis & (yy>=p40)&(yy<=p60)]=1; tc[vis & (yy>p60)]=2
            out[3]=tc
            neural.append(fr); inputs.append(inp); outputs.append(out); tongue_visible.append(vis)

        info=dict(file=path.name,identifier=nwb.identifier,subject=str(nwb.subject.subject_id),raw_trials=len(tr),structurally_valid_trials=len(structural_idx),observation_valid_trials=int(len(neural_valid)),dropped_zero_neural_trials=dropped_zero_neural,retained_trials=len(trial_idx),raw_units=len(nwb.units),classifier_good_units=len(candidates),retained_units=len(units),tongue_series=tongue_key,tongue_likelihood_threshold=LIKELIHOOD_THRESHOLD,tongue_y_p40=float(p40),tongue_y_p60=float(p60),conversion_seconds=time.time()-t0)
        plot_payload=None
        if make_plot:
            plot_payload=(go,tone,pstart,pstop,vt,vy,vl,p40,p60,neural,inputs,outputs,np.asarray(tongue_visible),info)
        return dict(neural=neural,input=inputs,output=outputs,subject=info['subject'],regions=names.tolist(),info=info,plot=plot_payload)


def plot_processing(payload,outpath):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    go,tone,ps,pe,vt,vy,vl,p40,p60,neural,inputs,outputs,visible,info=payload
    ti=min(4,len(go)-1); g=go[ti]; centers=g+CENTERS_REL
    fig,ax=plt.subplots(5,1,figsize=(12,14),constrained_layout=True)
    im=ax[0].imshow(neural[ti][:min(80,len(neural[ti]))],aspect='auto',extent=[OFF_START,OFF_END,min(80,len(neural[ti])),0],cmap='viridis'); fig.colorbar(im,ax=ax[0],label='Hz'); ax[0].set_title('Go-aligned 50-ms firing rates (first 80 neurons)')
    ax[1].plot(CENTERS_REL,inputs[ti][0],label='time from tone'); ax[1].plot(CENTERS_REL,inputs[ti][1],label='photostim on'); ax[1].axvline(0,color='k',ls='--'); ax[1].legend(); ax[1].set_title('Decoder inputs')
    if vt is not None:
        m=(vt>=g+OFF_START)&(vt<=g+OFF_END); ax[2].plot(vt[m]-g,vy[m],lw=.7,label='tongue y'); ax2=ax[2].twinx(); ax2.plot(vt[m]-g,vl[m],color='gray',alpha=.5,label='likelihood'); ax[2].axhline(p40,color='C1',ls='--'); ax[2].axhline(p60,color='C2',ls='--'); ax2.axhline(LIKELIHOOD_THRESHOLD,color='r',ls=':')
    ax[2].set_title('Raw tongue y, confidence, and session percentile thresholds')
    ax[3].step(CENTERS_REL,outputs[ti][3],where='mid'); ax[3].set_yticks(range(4),['<p40','p40-p60','>p60','not visible']); ax[3].set_title('Discretized tongue output')
    vals=np.concatenate([o[3] for o in outputs]); ax[4].bar(range(4),np.bincount(vals,minlength=4)/len(vals)); ax[4].set_xticks(range(4),['<p40','p40-p60','>p60','not visible']); ax[4].set_title('Session tongue-class fractions')
    for a in ax: a.set_xlabel('seconds from go cue')
    fig.suptitle(f"{info['identifier']} | trial {ti} | {info['retained_units']} units")
    fig.savefig(outpath,dpi=140); plt.close(fig)


def validate(data):
    ns=len(data['neural']); assert ns==len(data['input'])==len(data['output'])==len(data['brain_region_idx'])==len(data['subject_idx'])
    assert N_TIME==80 and len(EDGES_REL)==81
    for s in range(ns):
        assert len(data['neural'][s])>=2 and len(data['neural'][s])==len(data['input'][s])==len(data['output'][s])
        nn=data['neural'][s][0].shape[0]; assert len(data['brain_region_idx'][s])==nn
        for n,i,o in zip(data['neural'][s],data['input'][s],data['output'][s]):
            assert n.shape==(nn,N_TIME) and i.shape==(2,N_TIME) and o.shape==(4,N_TIME)
            assert np.isfinite(n).all() and np.isfinite(i).all() and (n>=0).all()
            assert set(np.unique(i[1])).issubset({0.,1.}); assert o.min()>=0
            assert o[0].max()<3 and o[1].max()<3 and o[2].max()<2 and o[3].max()<4


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('outpicklefile'); mode=ap.add_mutually_exclusive_group(); mode.add_argument('--full',action='store_true'); mode.add_argument('--sample',action='store_true'); ap.add_argument('--show-processing',action='store_true'); args=ap.parse_args()
    files=sorted(Path('/app/data').rglob('*.nwb')); target=2 if args.sample else None
    print(f'Discovered {len(files)} NWB files; mode={"sample" if args.sample else "full"}',flush=True)
    sessions=[]; t0=time.time()
    for p in files:
        st=time.time(); res=convert_session(p,make_plot=args.show_processing and len(sessions)<2)
        if res is None: print(f'SKIP {p.name}: insufficient valid neurons/trials',flush=True); continue
        sessions.append(res); q=res['info']; print(f"[{len(sessions)}] {p.name}: {q['retained_trials']} trials, {q['retained_units']}/{q['classifier_good_units']} QC units, {time.time()-st:.2f}s",flush=True)
        if target and len(sessions)>=target: break
    subjects=sorted({x['subject'] for x in sessions}); regions=sorted({r for x in sessions for r in x['regions']}); smap={v:i for i,v in enumerate(subjects)}; rmap={v:i for i,v in enumerate(regions)}
    data={'neural':[x['neural'] for x in sessions],'input':[x['input'] for x in sessions],'output':[x['output'] for x in sessions],
          'subjects':subjects,'subject_idx':np.asarray([smap[x['subject']] for x in sessions],dtype=np.int64),
          'brain_regions':regions,'brain_region_idx':[np.asarray([rmap[r] for r in x['regions']],dtype=np.int64) for x in sessions],
          'input_names':['time from tone onset (s)','photostimulation on'],
          'output_names':['lick direction choice','outcome','early lick','tongue y-position'],
          'output_values':[['left','right','no lick'],['ignore','miss','hit'],['no','yes'],['<40th percentile','40th to 60th percentile','>60th percentile','not visible']],
          'metadata':{'task_description':'Auditory delayed-response task; decode lick choice, outcome, early licking, and time-varying tongue y-position from neural firing rates.','time_bin_size':50.0,'time_bin_units':'ms','neural_units':'Hz','temporal_alignment_event':'go cue onset','off_start':OFF_START,'off_end':OFF_END,'n_timepoints':N_TIME,'bin_centers_seconds':CENTERS_REL.astype(np.float32),'neuron_filter':'NWB classification == good; retained trials are the intersection of per-unit obs_intervals/is_good_trials mappings','tongue_visibility_likelihood_threshold':LIKELIHOOD_THRESHOLD,'tongue_discretization':'Per-session visible-frame y percentiles: <p40, p40-p60, >p60; low-confidence/uncovered=not visible','session_info':[x['info'] for x in sessions]}}
    validate(data)
    with open(args.outpicklefile,'wb') as f: pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
    if args.show_processing:
        for x in sessions[:2]:
            if x['plot'] is not None: plot_processing(x['plot'],f"/app/processing_{x['info']['identifier']}.png")
    ntr=sum(map(len,data['neural'])); nneu=sum(x[0].shape[0] for x in data['neural']); print(f'SAVED {args.outpicklefile}: sessions={len(sessions)} subjects={len(subjects)} trials={ntr} summed_neurons={nneu} regions={len(regions)} elapsed={time.time()-t0:.1f}s',flush=True)

if __name__=='__main__': main()
