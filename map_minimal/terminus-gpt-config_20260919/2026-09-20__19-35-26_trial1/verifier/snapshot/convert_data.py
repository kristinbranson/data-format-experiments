#!/usr/bin/env python3
"""Convert the International Brain Laboratory/Janelia brain-wide NWB release.

Decisions follow the supplied movement-encoding methods: sessions in the release
are already behaviorally curated; units are retained only when the regional
classifier labels them ``good``. Spikes and all behavioral streams use the NWB
master clock. Exactly 80 non-overlapping 50-ms bins cover [-2.5, 1.5) s around
the go-cue onset. DLC confidence >= .9 defines a visible tongue. Session-wide
40/60 percentiles are computed from all visible tongue-y samples, as requested.
"""
import os, glob, pickle, json
import numpy as np
from pynwb import NWBHDF5IO

ROOT='/app/data'; OUT='/app/converted_data.pkl'
DT=.05; OFF0=-2.5; OFF1=1.5
EDGES=np.arange(OFF0, OFF1+DT/2, DT, dtype=np.float64)
CENTERS=(EDGES[:-1]+EDGES[1:])/2
DLC_VISIBLE=.9

def s(x):
    return x.decode() if isinstance(x,(bytes,np.bytes_)) else str(x)

def probe_region(unit_group):
    """Use broad recording location from probe metadata, not fine CCF layer."""
    loc=s(getattr(unit_group,'location','')).strip()
    try:
        d=json.loads(loc)
        return str(d.get('brain_regions',loc)).strip()
    except Exception:
        return loc or 'unknown'

def one_file(path):
    with NWBHDF5IO(path,'r',load_namespaces=True) as io:
        n=io.read(); tr=n.trials; u=n.units
        subject=s(n.subject.subject_id)
        ntr=len(tr)
        # Classifier label is the QC criterion described in methods.txt.
        cls=np.asarray([s(x).lower() for x in u['classification'][:]])
        keep=np.flatnonzero(cls=='good')
        if not len(keep):
            return None
        regions=[probe_region(u['electrode_group'][int(i)]) for i in keep]

        ev=n.acquisition['BehavioralEvents'].time_series
        go=np.asarray(ev['go_start_times'].timestamps[:],dtype=float)
        sample=np.asarray(ev['sample_start_times'].timestamps[:],dtype=float)
        if len(go)!=ntr:
            raise RuntimeError(f'{path}: {len(go)} go events != {ntr} trials')
        # Occasional extra sample events are replays after an early lick. Associate
        # each trial with the last sample onset preceding its unique go cue.
        sample_for_trial=np.empty(ntr,float)
        for j,g in enumerate(go):
            q=sample[sample < g+1e-9]
            sample_for_trial[j]=q[-1] if len(q) else np.nan

        pstart=np.asarray(ev['photostim_start_times'].timestamps[:],float)
        pstop=np.asarray(ev['photostim_stop_times'].timestamps[:],float)
        tongue=n.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
        tt=np.asarray(tongue.timestamps[:],float)
        td=np.asarray(tongue.data[:],dtype=np.float32)
        visible=np.isfinite(td[:,1]) & np.isfinite(td[:,2]) & (td[:,2]>=DLC_VISIBLE)
        if visible.any(): q40,q60=np.percentile(td[visible,1],[40,60])
        else: q40,q60=np.nan,np.nan

        instr=np.asarray([s(x).lower() for x in tr['trial_instruction'][:]])
        outcome=np.asarray([s(x).lower() for x in tr['outcome'][:]])
        early=np.asarray([s(x).lower() for x in tr['early_lick'][:]])
        neural=[]; inputs=[]; outputs=[]
        # Materialize sorted spike vectors and bin all trials at once. Using
        # left insertion indices gives counts in [edge_i, edge_i+1), matching
        # the half-open convention in the supplied reference implementation.
        spikes=[np.asarray(u['spike_times'][int(i)],float) for i in keep]
        all_edges=go[:,None]+EDGES[None,:]
        frcube=np.empty((len(keep),ntr,len(CENTERS)),dtype=np.float32)
        for k,st in enumerate(spikes):
            frcube[k]=np.diff(np.searchsorted(st,all_edges,side='left'),axis=1)/DT
        for j,g in enumerate(go):
            fr=frcube[:,j,:].copy()
            times=g+CENTERS
            tone_elapsed=(times-sample_for_trial[j]).astype(np.float32)
            photo=np.zeros(len(CENTERS),dtype=np.float32)
            # State at bin centers for every stimulation interval.
            for a,b in zip(pstart,pstop): photo[(times>=a)&(times<b)]=1

            if outcome[j]=='ignore': choice=2
            elif outcome[j]=='hit': choice=0 if instr[j]=='left' else 1
            elif outcome[j]=='miss': choice=1 if instr[j]=='left' else 0
            else: raise ValueError(f'Unknown outcome {outcome[j]}')
            omap={'ignore':0,'miss':1,'hit':2}
            ecat=1 if early[j].startswith('early') else 0
            # Nearest camera frame at each bin center; no extrapolation beyond video.
            ix=np.searchsorted(tt,times); ix=np.clip(ix,1,len(tt)-1)
            ix-=((times-tt[ix-1]) <= (tt[ix]-times))
            vis=(td[ix,2]>=DLC_VISIBLE)&np.isfinite(td[ix,1])
            ycat=np.full(len(CENTERS),3,dtype=np.int8)
            ycat[vis & (td[ix,1]<q40)]=0
            ycat[vis & (td[ix,1]>=q40) & (td[ix,1]<=q60)]=1
            ycat[vis & (td[ix,1]>q60)]=2
            out=np.vstack((np.full(len(CENTERS),choice,np.int8),
                           np.full(len(CENTERS),omap[outcome[j]],np.int8),
                           np.full(len(CENTERS),ecat,np.int8),ycat))
            neural.append(fr)
            inputs.append(np.vstack((tone_elapsed,photo)).astype(np.float32))
            outputs.append(out)
        info={'file':os.path.basename(path),'subject':subject,'n_trials':ntr,
              'n_good_units':len(keep),'tongue_y_percentiles':[float(q40),float(q60)],
              'dlc_visibility_threshold':DLC_VISIBLE}
        return subject,regions,neural,inputs,outputs,info

def main():
    files=sorted(glob.glob(os.path.join(ROOT,'sub-*','*.nwb')))
    if not files: raise FileNotFoundError('No NWB files found')
    data={'neural':[],'input':[],'output':[],'subjects':[], 'subject_idx':[],
          'brain_regions':[],'brain_region_idx':[],
          'input_names':['time from tone onset (s)','photostimulation on'],
          'output_names':['lick direction choice','outcome','early lick','tongue y-position'],
          'output_values':[['left','right','no lick'],['ignore','miss','hit'],
                           ['no','yes'],['below 40th percentile','40th to 60th percentile',
                                        'above 60th percentile','not visible']],
          'metadata':{'task_description':'Auditory delayed-response task: decode choice, outcome, early licking, and tongue y-position.',
                      'time_bin_size':50.0,'temporal_alignment_event':'go cue onset',
                      'off_start':-2.5,'off_end':1.5,'n_time_bins':80,
                      'neural_measure':'firing rate (spikes/s) in non-overlapping half-open bins',
                      'unit_filter':'classification == good (region-specific classifier QC)',
                      'tongue_visibility':'DeepLabCut confidence >= 0.9',
                      'session_info':[]}}
    for z,path in enumerate(files,1):
        result=one_file(path)
        if result is None:
            print(f'[{z}/{len(files)}] SKIP {os.path.basename(path)}: no classifier-good units',flush=True)
            continue
        sub,regs,neu,inp,out,info=result
        if sub not in data['subjects']: data['subjects'].append(sub)
        data['subject_idx'].append(data['subjects'].index(sub))
        rid=[]
        for r in regs:
            if r not in data['brain_regions']: data['brain_regions'].append(r)
            rid.append(data['brain_regions'].index(r))
        data['brain_region_idx'].append(np.asarray(rid,dtype=np.int16))
        data['neural'].append(neu); data['input'].append(inp); data['output'].append(out)
        data['metadata']['session_info'].append(info)
        print(f'[{z}/{len(files)}] {os.path.basename(path)}: {len(neu)} trials, {len(regs)} units',flush=True)
    data['subject_idx']=np.asarray(data['subject_idx'],dtype=np.int16)
    tmp=OUT+'.tmp'
    with open(tmp,'wb') as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp,OUT)
    print('WROTE',OUT,os.path.getsize(OUT),flush=True)
if __name__=='__main__': main()
