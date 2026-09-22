#!/usr/bin/env python3
"""Convert a deterministic 10-session IBL BWM subset to decoder format."""
import argparse, hashlib, pickle, sys, time, urllib.request, urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from brainbox.behavior.wheel import interpolate_position, velocity_filtered
from iblatlas.regions import BrainRegions

ROOT=Path('/app'); RAW=ROOT/'data'/'raw_cache'; RAW.mkdir(parents=True,exist_ok=True)
QPATH=ROOT/'data/one_cache/Brainwidemap/datasets.pqt'; SPATH=ROOT/'data/one_cache/Brainwidemap/sessions.pqt'
RELEASE=ROOT/'code/code_zhang2025/data/bwm_release.csv'
OFF0,OFF1,DT=-.6,1.5,.02
EDGES=np.arange(OFF0,OFF1+DT/2,DT); CENTERS=(EDGES[:-1]+EDGES[1:])/2
BR=BrainRegions(); ID2AC={int(i):str(a) for i,a in zip(BR.id,BR.acronym)}

def selected_eids(n):
 x=pd.read_csv(RELEASE,index_col=0); np.random.seed(42)
 subs=np.random.choice(np.unique(x.subject),10,replace=False); by=x.groupby('subject').indices
 eids=[str(x.iloc[by[s][0]].eid) for s in subs]
 return list(zip(subs[:n],eids[:n]))

def choose(g, needle, unrevisioned=False):
 z=g[g.rel_path.str.contains(needle,case=False,regex=False)].copy()
 if unrevisioned: z=z[~z.rel_path.str.contains('/#',regex=False)]
 if z.empty: raise KeyError(needle)
 # Exact unrevisioned cache record preferred; otherwise most recent listed path.
 return z.iloc[-1]

def record_url(eid,did,r,srow):
 stem,ext=r.rel_path.rsplit('.',1); rel=f'{stem}.{did}.{ext}'
 return f'https://ibl-brain-wide-map-public.s3.amazonaws.com/data/{srow.lab}/Subjects/{srow.subject}/{srow.date}/{int(srow.number):03d}/'+urllib.parse.quote(rel, safe='/._-')

def fetch(eid,did,r,srow):
 ext=r.rel_path.rsplit('.',1)[-1]; out=RAW/eid/(str(did)+'.'+ext); out.parent.mkdir(parents=True,exist_ok=True)
 expected=int(r.file_size) if pd.notna(r.file_size) else -1
 if out.exists() and (expected<0 or out.stat().st_size==expected): return out
 url=record_url(eid,did,r,srow); tmp=out.with_suffix(out.suffix+'.part')
 for attempt in range(5):
  try:
   urllib.request.urlretrieve(url,tmp)
   if expected>=0 and tmp.stat().st_size!=expected: raise IOError(f'size {tmp.stat().st_size}!={expected}')
   tmp.replace(out); return out
  except Exception as e:
   tmp.unlink(missing_ok=True)
   if attempt==4: raise RuntimeError(f'download failed {url}: {e}')
   time.sleep(2**attempt)

def get_record(g,needle,unrevisioned=False):
 z=g[g.rel_path.str.contains(needle,case=False,regex=False)]
 if unrevisioned: z=z[~z.rel_path.str.contains('/#',regex=False)]
 if z.empty: raise KeyError(needle)
 did=z.index[-1]; return did,z.iloc[-1]

def load_record(eid,g,srow,needle,unrevisioned=False,allow_pickle=False):
 did,r=get_record(g,needle,unrevisioned); f=fetch(eid,did,r,srow)
 return pd.read_parquet(f) if f.suffix=='.pqt' else np.load(f,allow_pickle=allow_pickle)

def probe_names(g):
 return sorted(g.rel_path.str.extract(r'/(probe\d+)/',expand=False).dropna().unique())

def load_session(eid,subject,q,srow):
 t0=time.time(); g=q.loc[eid]
 trials=load_record(eid,g,srow,'trials.table')
 # Behavior
 wt=load_record(eid,g,srow,'wheel.timestamps'); wp=load_record(eid,g,srow,'wheel.position')
 wpos,wtime=interpolate_position(wt,wp,freq=1000); wvel,_=velocity_filtered(wpos,fs=1000)
 wspeed=np.abs(wvel)
 side=None
 for v in ('left','right'):
  try:
   ct=load_record(eid,g,srow,f'{v}Camera.times'); cm=load_record(eid,g,srow,f'{v}Camera.ROIMotionEnergy')
   n=min(len(ct),len(cm)); ct=np.asarray(ct[:n],float); cm=np.asarray(cm[:n],float)
   if n>10 and np.isfinite(cm).mean()>.95: side=v; break
  except Exception as e:
   print(f'Camera {v} unavailable for {eid}: {e}', flush=True)
   continue
 if side is None: raise ValueError('no valid whisker motion stream')
 # Good neurons from every physical probe; unrevisioned reference pykilosort.
 probe_data=[]; regions=[]; labels_all=[]
 for probe in probe_names(g):
  base=f'{probe}/pykilosort/'
  try:
   st=np.asarray(load_record(eid,g,srow,base+'spikes.times',True),float)
   sc=np.asarray(load_record(eid,g,srow,base+'spikes.clusters',True),int)
   met=load_record(eid,g,srow,base+'clusters.metrics',True)
   ch=np.asarray(load_record(eid,g,srow,base+'clusters.channels',True),int)
   labels=np.asarray(met['label'],float); good=np.flatnonzero(labels>=1)
   # anatomy: channel atlas IDs; fallback void if unavailable.
   try: atlas=np.asarray(load_record(eid,g,srow,f'{probe}/channels.brainLocationIds_ccf_2017'))
   except Exception:
    try: atlas=np.asarray(load_record(eid,g,srow,f'{probe}/pykilosort/channels.brainLocationIds_ccf_2017',True))
    except Exception: atlas=np.zeros(max(ch.max()+1,1),int)
   regs=[ID2AC.get(int(atlas[c]),'void') if 0<=c<len(atlas) else 'void' for c in ch[good]]
   probe_data.append((st,sc,good)); regions.extend(regs); labels_all.extend(labels[good])
  except KeyError: continue
 if not probe_data or not regions: raise ValueError('no good neurons')
 stim=np.asarray(trials.stimOn_times,float); choice=np.asarray(trials.choice,float); prior=np.asarray(trials.probabilityLeft,float)
 valid=np.isfinite(stim)&np.isin(choice,[-1,1])&np.isin(np.round(prior,1),[.2,.5,.8])
 # ensure behavior coverage and finite interpolants
 valid &= (stim+OFF0>=wtime[0])&(stim+OFF1<=wtime[-1])&(stim+OFF0>=ct[0])&(stim+OFF1<=ct[-1])
 idx=np.flatnonzero(valid); neural=[]; rawwheel=[]; rawwhisk=[]; kept=[]
 for ti in idx:
  times=stim[ti]+CENTERS
  ww=np.interp(times,wtime,wspeed); mm=np.interp(times,ct,cm)
  if not (np.isfinite(ww).all() and np.isfinite(mm).all()): continue
  mats=[]
  for st,sc,good in probe_data:
   lo=np.searchsorted(st,stim[ti]+OFF0); hi=np.searchsorted(st,stim[ti]+OFF1)
   rel=st[lo:hi]-stim[ti]; cid=sc[lo:hi]
   M=np.zeros((len(good),len(CENTERS)),np.float32); lut={int(c):j for j,c in enumerate(good)}
   bins=np.floor((rel-OFF0)/DT).astype(int)
   for c,b in zip(cid,bins):
    j=lut.get(int(c));
    if j is not None and 0<=b<M.shape[1]: M[j,b]+=1
   mats.append(M)
  neural.append(np.concatenate(mats)); rawwheel.append(ww.astype(np.float32)); rawwhisk.append(mm.astype(np.float32)); kept.append(ti)
 if len(kept)<2: raise ValueError('fewer than two valid trials')
 kept=np.array(kept); pr=np.round(prior,1); blocknum=np.zeros(len(trials),np.float32)
 for i in range(1,len(trials)): blocknum[i]=0 if pr[i]!=pr[i-1] else blocknum[i-1]+1
 print(f'Loaded {eid} {subject}: {len(kept)}/{len(trials)} trials, {len(regions)} neurons, camera={side}, {time.time()-t0:.1f}s',flush=True)
 return dict(eid=eid,subject=str(subject),neural=neural,regions=regions,labels=np.array(labels_all),trial_idx=kept,
             choice=choice[kept],prior=pr[kept],blocknum=blocknum[kept],wheel=rawwheel,whisk=rawwhisk,camera=side)

def plot_session(x,out):
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(2,2,figsize=(11,7)); im=ax[0,0].imshow(x['neural'][0],aspect='auto',interpolation='none'); fig.colorbar(im,ax=ax[0,0]); ax[0,0].set_title('Trial 1 spike counts')
 ax[0,1].plot(CENTERS,x['wheel'][0]); ax[0,1].set_title('Aligned wheel speed')
 ax[1,0].plot(CENTERS,x['whisk'][0]); ax[1,0].set_title(f"Aligned {x['camera']} whisker motion")
 ax[1,1].hist(np.concatenate(x['wheel']),50,alpha=.6,label='wheel'); ax[1,1].hist(np.concatenate(x['whisk']),50,alpha=.6,label='whisker'); ax[1,1].set_yscale('log'); ax[1,1].legend()
 for a in ax.flat: a.set_xlabel('time / value');
 fig.suptitle(x['eid']); fig.tight_layout(); fig.savefig(out,dpi=140); plt.close(fig)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('outpicklefile'); m=ap.add_mutually_exclusive_group(); m.add_argument('--full',action='store_true'); m.add_argument('--sample',action='store_true'); ap.add_argument('--show-processing',action='store_true'); args=ap.parse_args()
 n=2 if args.sample else 10; q=pd.read_parquet(QPATH); ss=pd.read_parquet(SPATH); chosen=selected_eids(n)
 print('Selected sessions:',chosen,flush=True); sessions=[]
 for subject,eid in chosen:
  try: sessions.append(load_session(eid,subject,q,ss.loc[eid]))
  except Exception as e: print(f'ERROR session {eid}: {e}',file=sys.stderr,flush=True); raise
 if args.show_processing:
  for x in sessions[:2]: plot_session(x,ROOT/f"processing_{x['eid']}.png")
 wheel=np.concatenate([np.concatenate(x['wheel']) for x in sessions]); whisk=np.concatenate([np.concatenate(x['whisk']) for x in sessions])
 wq=np.quantile(wheel,[1/3,2/3]); mq=np.quantile(whisk,[1/3,2/3]); print('tertiles wheel',wq,'whisker',mq)
 subjects=[]
 for x in sessions:
  if x['subject'] not in subjects: subjects.append(x['subject'])
 allregs=sorted(set(r for x in sessions for r in x['regions'])); rid={r:i for i,r in enumerate(allregs)}
 data={'neural':[],'input':[],'output':[],'subjects':subjects,'subject_idx':np.array([subjects.index(x['subject']) for x in sessions],np.int64),
       'brain_regions':allregs,'brain_region_idx':[],'input_names':['time_since_stimulus_onset','trial_number_in_block'],
       'output_names':['choice','prior_probability_left','wheel_speed','whisker_motion_energy'],
       'output_values':[['left','right'],['0.2','0.5','0.8'],['low','medium','high'],['low','medium','high']],
       'metadata':{'task_description':'Decode choice, prior, wheel-speed tertile and whisker-motion-energy tertile from stimulus-aligned spike counts.',
       'time_bin_size':20.0,'temporal_alignment_event':'visual stimulus onset (trials.stimOn_times)','off_start':OFF0,'off_end':OFF1,
       'neural_representation':'spike counts per 20 ms bin; clusters.metrics label >= 1','discretization':'pooled value tertiles over converted sessions',
       'wheel_speed_thresholds':wq.tolist(),'whisker_motion_energy_thresholds':mq.tolist(),'session_info':[]}}
 for x in sessions:
  ins=[]; outs=[]
  for c,p,b,w,m in zip(x['choice'],x['prior'],x['blocknum'],x['wheel'],x['whisk']):
   ins.append(np.vstack([CENTERS,np.full(len(CENTERS),b)]).astype(np.float32))
   cc=0 if c==-1 else 1; pp={.2:0,.5:1,.8:2}[float(p)]
   outs.append(np.vstack([np.full(len(CENTERS),cc),np.full(len(CENTERS),pp),np.digitize(w,wq),np.digitize(m,mq)]).astype(np.int64))
  data['neural'].append(x['neural']); data['input'].append(ins); data['output'].append(outs); data['brain_region_idx'].append(np.array([rid[r] for r in x['regions']],np.int64))
  data['metadata']['session_info'].append({'eid':x['eid'],'subject':x['subject'],'n_trials':len(x['neural']),'n_neurons':len(x['regions']),'camera':x['camera']})
 with open(args.outpicklefile,'wb') as f: pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
 print(f'Saved {args.outpicklefile}: {len(sessions)} sessions, {sum(map(len,data["neural"]))} trials, {sum(len(x) for x in data["brain_region_idx"])} session-neurons, {Path(args.outpicklefile).stat().st_size/1e6:.1f} MB',flush=True)
if __name__=='__main__': main()
