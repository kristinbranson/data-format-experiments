#!/usr/bin/env python3
"""Convert the MAP auditory delayed-response NWB files for neural decoding.

Decisions:
* Keep every supplied session/trial: ignore and early-lick trials are required targets.
* Keep units classified 'good', matching the paper's region-specific QC classifiers.
* Trial start is 2.5 s before the go cue; sample-tone onset is 0.5 s after trial
  start. Histograms use left-closed 50-ms bins over go-2.5 through go+1.5.
* DLC tongue detections with likelihood < .9 are treated as not visible. Session
  percentiles are computed from all visible samples falling in retained windows.
"""
import argparse, glob, os, pickle
import h5py
import numpy as np

DT=.05; NBIN=80; GO_FROM_START=2.5; TONE_FROM_START=.5; DLC_THRESHOLD=.9

def text(a):
    return np.asarray([x.decode() if isinstance(x,(bytes,np.bytes_)) else str(x) for x in a])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--limit',type=int); ap.add_argument('--output',default='/app/converted_data.pkl'); args=ap.parse_args()
    files=sorted(glob.glob('/app/data/sub-*/*.nwb'))
    # A recording with no QC-passing units cannot furnish decoder input; the paper's
    # analyzed set likewise contains 173 sessions rather than all 174 NWB assets.
    kept=[]
    for p in files:
      with h5py.File(p,'r') as f:
        if np.any(text(f['units/classification'][:])=='good'): kept.append(p)
        else: print('Excluding session with no good units:',os.path.basename(p),flush=True)
    files=kept
    if args.limit: files=files[:args.limit]
    subjects=sorted({os.path.basename(os.path.dirname(p)).removeprefix('sub-') for p in files})
    subjmap={s:i for i,s in enumerate(subjects)}
    neural=[]; inputs=[]; outputs=[]; region_names=[]; region_map={}; region_indices=[]; sinfo=[]
    # Bin centers relative to go; tone starts at -2.0 s.
    rel_centers=-GO_FROM_START+(np.arange(NBIN)+.5)*DT
    time_from_tone=rel_centers-(-GO_FROM_START+TONE_FROM_START)
    for fi,p in enumerate(files):
      print(f'[{fi+1}/{len(files)}] {os.path.basename(p)}',flush=True)
      with h5py.File(p,'r') as f:
        tr=f['intervals/trials']; starts=np.asarray(tr['start_time'][:],float); ntr=len(starts)
        outcome=text(tr['outcome'][:]); instr=text(tr['trial_instruction'][:]); early=text(tr['early_lick'][:])
        pon=text(tr['photostim_onset'][:]); pdur=text(tr['photostim_duration'][:]); ppow=text(tr['photostim_power'][:])
        u=f['units']; cls=text(u['classification'][:]); good=np.flatnonzero(cls=='good')
        # Paper analyses use anatomical annotation attached to each classified unit.
        if 'anno_name' in u:
          regs=text(u['anno_name'][:])[good]
        else:
          regs=np.repeat('unknown',len(good))
        regs=np.asarray([r if r else 'unknown' for r in regs])
        rid=[]
        for r in regs:
          if r not in region_map: region_map[r]=len(region_names); region_names.append(r)
          rid.append(region_map[r])
        region_indices.append(np.asarray(rid,dtype=np.int32))
        # Read ragged spike vectors once and histogram each good unit across all trial bins.
        allsp=np.asarray(u['spike_times'][:],float); ends=np.asarray(u['spike_times_index'][:],int)
        begins=np.r_[0,ends[:-1]]
        sess_n=[np.empty((len(good),NBIN),dtype=np.float32) for _ in range(ntr)]
        for row,ui in enumerate(good):
          sp=allsp[begins[ui]:ends[ui]]
          for ti,s in enumerate(starts):
            # integer indexing is stable and gives [start,start+4) bins
            lo=np.searchsorted(sp,s,'left'); hi=np.searchsorted(sp,s+NBIN*DT,'left')
            idx=np.floor((sp[lo:hi]-s)/DT).astype(np.int64)
            sess_n[ti][row]=np.bincount(idx,minlength=NBIN)[:NBIN]/DT
        # Tongue side view: columns x, y, DLC likelihood. Camera0 exists in all files.
        tg=f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
        td=np.asarray(tg['data'][:],float); tt=np.asarray(tg['timestamps'][:],float)
        # Nearest video frame to each 50-ms center (video is ~300 Hz).
        sample_t=(starts[:,None]+(np.arange(NBIN)+.5)*DT).ravel()
        jj=np.searchsorted(tt,sample_t); jj=np.clip(jj,1,len(tt)-1)
        jj-=((sample_t-tt[jj-1]) <= (tt[jj]-sample_t))
        yy=td[jj,1].reshape(ntr,NBIN); conf=td[jj,2].reshape(ntr,NBIN)
        visible=np.isfinite(yy)&np.isfinite(conf)&(conf>=DLC_THRESHOLD)
        visvals=yy[visible]
        q40,q60=np.percentile(visvals,[40,60]) if len(visvals) else (np.nan,np.nan)
        tongue=np.full((ntr,NBIN),3,dtype=np.int64)
        tongue[visible&(yy<q40)]=0; tongue[visible&(yy>=q40)&(yy<=q60)]=1; tongue[visible&(yy>q60)]=2
        sess_i=[]; sess_o=[]
        for ti in range(ntr):
          stim=np.zeros(NBIN,dtype=np.float32)
          if ppow[ti] != 'N/A':
            a=float(pon[ti])-GO_FROM_START; b=a+float(pdur[ti])
            # state at bin centers
            stim[(rel_centers>=a)&(rel_centers<b)]=1
          sess_i.append(np.stack((time_from_tone.astype(np.float32),stim)))
          # Actual lick choice: correct instruction on hit, opposite on miss, no lick on ignore.
          if outcome[ti]=='ignore': choice=2
          elif outcome[ti]=='hit': choice=0 if instr[ti]=='left' else 1
          else: choice=1 if instr[ti]=='left' else 0
          oc={'ignore':0,'miss':1,'hit':2}[outcome[ti]]
          el=1 if early[ti]=='early' else 0
          sess_o.append(np.vstack((np.full(NBIN,choice,dtype=np.int64),np.full(NBIN,oc,dtype=np.int64),np.full(NBIN,el,dtype=np.int64),tongue[ti])))
        neural.append(sess_n); inputs.append(sess_i); outputs.append(sess_o)
        sub=os.path.basename(os.path.dirname(p)).removeprefix('sub-')
        sinfo.append({'file':os.path.basename(p),'subject':sub,'n_trials':ntr,'n_good_units':len(good),'tongue_y_percentiles':[float(q40),float(q60)],'dlc_likelihood_threshold':DLC_THRESHOLD})
    data={'neural':neural,'input':inputs,'output':outputs,'subjects':subjects,
      'subject_idx':np.asarray([subjmap[x['subject']] for x in sinfo],dtype=np.int32),
      'brain_regions':region_names,'brain_region_idx':region_indices,
      'input_names':['time from tone onset','photostimulation on'],
      'output_names':['lick direction choice','outcome','early lick','tongue y-position'],
      'output_values':[['left','right','no lick'],['ignore','miss','hit'],['no','yes'],['below 40th percentile','40th to 60th percentile','above 60th percentile','not visible']],
      'metadata':{'task_description':'Auditory delayed-response task: decode lick choice, outcome, early licking, and discretized side-view tongue y-position.','time_bin_size':50.0,'temporal_alignment_event':'auditory Go cue onset','off_start':-2.5,'off_end':1.5,'neural_units':'firing rate (spikes/s)','bin_convention':'left-closed, right-open; values sampled at bin centers','go_cue_from_trial_start_s':2.5,'tone_onset_from_trial_start_s':0.5,'unit_filter':"NWB units.classification == 'good' (paper QC classifier)",'trial_filter':'all trials retained because ignore and early lick are requested outputs; sessions with zero QC-passing units excluded','tongue_processing':'nearest Camera0 side frame to each bin center; DLC likelihood >=0.9 visible; visible y discretized by per-session 40th/60th percentiles','session_info':sinfo}}
    print('Writing',args.output,flush=True)
    with open(args.output,'wb') as fh: pickle.dump(data,fh,protocol=pickle.HIGHEST_PROTOCOL)
if __name__=='__main__': main()
