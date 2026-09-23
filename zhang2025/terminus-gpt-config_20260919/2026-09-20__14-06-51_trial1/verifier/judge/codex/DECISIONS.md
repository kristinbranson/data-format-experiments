# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not load the full release through ONE. It reads local metadata tables from `datasets.pqt`, `sessions.pqt`, and `bwm_release.csv`, picks a deterministic 10-subject subset, and then downloads per-session payload files directly from the public S3 bucket into `/app/data/raw_cache`. Session contents are then loaded record-by-record with `load_record`.

ii.
```python
QPATH=ROOT/'data/one_cache/Brainwidemap/datasets.pqt'; SPATH=ROOT/'data/one_cache/Brainwidemap/sessions.pqt'
RELEASE=ROOT/'code/code_zhang2025/data/bwm_release.csv'
```

```python
def selected_eids(n):
 x=pd.read_csv(RELEASE,index_col=0); np.random.seed(42)
 subs=np.random.choice(np.unique(x.subject),10,replace=False); by=x.groupby('subject').indices
 eids=[str(x.iloc[by[s][0]].eid) for s in subs]
 return list(zip(subs[:n],eids[:n]))
```

```python
def fetch(eid,did,r,srow):
 ext=r.rel_path.rsplit('.',1)[-1]; out=RAW/eid/(str(did)+'.'+ext)
 ...
 url=record_url(eid,did,r,srow)
 ...
 urllib.request.urlretrieve(url,tmp)
```

iii. In `CONVERSION_NOTES.md` Step 5 and the trajectory, the AI justifies this as a practicality decision: the local cache only contains metadata, direct S3 access was reachable, and processing all stream-complete sessions was estimated to be too large, so it chose a deterministic 10-session subset with metadata-driven downloads.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are taken from the `subject` column of `bwm_release.csv`. The AI samples 10 unique subjects with `np.random.seed(42)` and keeps one session per chosen subject. In the final dataset, `subjects` preserves first-seen order from the processed sessions.

ii.
```python
subs=np.random.choice(np.unique(x.subject),10,replace=False); by=x.groupby('subject').indices
eids=[str(x.iloc[by[s][0]].eid) for s in subs]
return list(zip(subs[:n],eids[:n]))
```

```python
subjects=[]
for x in sessions:
 if x['subject'] not in subjects: subjects.append(x['subject'])
...
'subject_idx':np.array([subjects.index(x['subject']) for x in sessions],np.int64),
```

iii. The notes justify this as matching the AI's chosen 10-session canonical subset, with one selected session per sampled subject.

## 1-c. How are the data split into sessions?

i. Sessions are defined by EIDs chosen from the release CSV. The AI does not iterate all available sessions; it processes the single selected EID for each sampled subject.

ii.
```python
eids=[str(x.iloc[by[s][0]].eid) for s in subs]
return list(zip(subs[:n],eids[:n]))
```

```python
n=2 if args.sample else 10; q=pd.read_parquet(QPATH); ss=pd.read_parquet(SPATH); chosen=selected_eids(n)
...
for subject,eid in chosen:
 try: sessions.append(load_session(eid,subject,q,ss.loc[eid]))
```

iii. The justification in the notes is again scope control: a deterministic 10-session subset was treated as the intended workload.

## 1-d. How are the data split into trials?

i. Trials are taken from the rows of the session's `trials.table` parquet. After a validity mask is applied, each retained row becomes one converted trial.

ii.
```python
trials=load_record(eid,g,srow,'trials.table')
```

```python
stim=np.asarray(trials.stimOn_times,float); choice=np.asarray(trials.choice,float); prior=np.asarray(trials.probabilityLeft,float)
valid=np.isfinite(stim)&np.isin(choice,[-1,1])&np.isin(np.round(prior,1),[.2,.5,.8])
...
idx=np.flatnonzero(valid)
for ti in idx:
    ...
    kept.append(ti)
```

iii. No separate justification is given beyond using the trial table as the natural per-trial unit.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials with finite `stimOn_times`, `choice` in `{-1, 1}`, and `probabilityLeft` rounded to one decimal in `{0.2, 0.5, 0.8}`. It also requires the full `[-0.6, 1.5]` s window to lie inside the wheel and camera time ranges, and later drops any trial whose interpolated wheel or whisker trace contains non-finite values. It does not apply a reaction-time filter.

ii.
```python
stim=np.asarray(trials.stimOn_times,float); choice=np.asarray(trials.choice,float); prior=np.asarray(trials.probabilityLeft,float)
valid=np.isfinite(stim)&np.isin(choice,[-1,1])&np.isin(np.round(prior,1),[.2,.5,.8])
# ensure behavior coverage and finite interpolants
valid &= (stim+OFF0>=wtime[0])&(stim+OFF1<=wtime[-1])&(stim+OFF0>=ct[0])&(stim+OFF1<=ct[-1])
```

```python
ww=np.interp(times,wtime,wspeed); mm=np.interp(times,ct,cm)
if not (np.isfinite(ww).all() and np.isfinite(mm).all()): continue
```

iii. In the notes the AI justifies this as a common validity mask for all streams under the task-mandated shared stimulus-aligned tensor.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural tensor is derived from spike times and spike cluster assignments per probe, with cluster metrics and channel/atlas metadata used to filter units and assign regions.

ii.
```python
st=np.asarray(load_record(eid,g,srow,base+'spikes.times',True),float)
sc=np.asarray(load_record(eid,g,srow,base+'spikes.clusters',True),int)
met=load_record(eid,g,srow,base+'clusters.metrics',True)
ch=np.asarray(load_record(eid,g,srow,base+'clusters.channels',True),int)
```

```python
try: atlas=np.asarray(load_record(eid,g,srow,f'{probe}/channels.brainLocationIds_ccf_2017'))
except Exception:
 ...
regs=[ID2AC.get(int(atlas[c]),'void') if 0<=c<len(atlas) else 'void' for c in ch[good]]
```

iii. The justification in the notes is that this matches the reference spike-sorting plus QC/anatomy metadata pipeline, while using direct S3 retrieval instead of ONE loaders.

## 2-b. How is the `neural` data processed?

i. For each retained trial, spikes are sliced from `-0.6` to `+1.5` s around stimulus onset, assigned to 20 ms bins by `floor`, counted into per-neuron matrices, and concatenated across probes. The saved values are spike counts, not firing rates.

ii.
```python
OFF0,OFF1,DT=-.6,1.5,.02
EDGES=np.arange(OFF0,OFF1+DT/2,DT); CENTERS=(EDGES[:-1]+EDGES[1:])/2
```

```python
lo=np.searchsorted(st,stim[ti]+OFF0); hi=np.searchsorted(st,stim[ti]+OFF1)
rel=st[lo:hi]-stim[ti]; cid=sc[lo:hi]
M=np.zeros((len(good),len(CENTERS)),np.float32); lut={int(c):j for j,c in enumerate(good)}
bins=np.floor((rel-OFF0)/DT).astype(int)
for c,b in zip(cid,bins):
    j=lut.get(int(c));
    if j is not None and 0<=b<M.shape[1]: M[j,b]+=1
```

iii. The notes explicitly justify the `[-0.6, 1.5]` window as the union of the AI's reading of the prior and choice windows, and justify 20 ms bins as preserving the dynamic-behavior resolution while staying stimulus-aligned.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered by `clusters.metrics.label >= 1`. Region labels are assigned from `channels.brainLocationIds_ccf_2017`; if anatomy lookup fails, the code falls back to zeros or `'void'`, but it does not explicitly remove `'void'` units.

ii.
```python
labels=np.asarray(met['label'],float); good=np.flatnonzero(labels>=1)
```

```python
try: atlas=np.asarray(load_record(eid,g,srow,f'{probe}/channels.brainLocationIds_ccf_2017'))
except Exception:
 try: atlas=np.asarray(load_record(eid,g,srow,f'{probe}/pykilosort/channels.brainLocationIds_ccf_2017',True))
 except Exception: atlas=np.zeros(max(ch.max()+1,1),int)
regs=[ID2AC.get(int(atlas[c]),'void') if 0<=c<len(atlas) else 'void' for c in ch[good]]
```

iii. In `CONVERSION_NOTES.md` the AI says it wanted to match the reference `good_clusters = (label >= 1)` rule and preserve per-neuron anatomy.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `trials.stimOn_times`. Spike times within the trial window are converted to time-relative-to-stimulus by subtraction of that trial's stimulus onset.

ii.
```python
stim=np.asarray(trials.stimOn_times,float)
```

```python
lo=np.searchsorted(st,stim[ti]+OFF0); hi=np.searchsorted(st,stim[ti]+OFF1)
rel=st[lo:hi]-stim[ti]
```

iii. The notes repeatedly justify stimulus-onset alignment as required by the task and consistent with the reference choice/prior analysis.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 20 ms bins. There is no later temporal rebinning; the spike counts are formed directly on that grid.

ii.
```python
OFF0,OFF1,DT=-.6,1.5,.02
EDGES=np.arange(OFF0,OFF1+DT/2,DT); CENTERS=(EDGES[:-1]+EDGES[1:])/2
```

```python
bins=np.floor((rel-OFF0)/DT).astype(int)
```

iii. The AI's notes justify 20 ms bins as the chosen compromise between the stimulus-aligned static targets and the paper's dynamic behavioral outputs.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from `trials.stimOn_times` plus the hard-coded bin grid `CENTERS`.

ii.
```python
OFF0,OFF1,DT=-.6,1.5,.02
EDGES=np.arange(OFF0,OFF1+DT/2,DT); CENTERS=(EDGES[:-1]+EDGES[1:])/2
```

```python
stim=np.asarray(trials.stimOn_times,float)
...
times=stim[ti]+CENTERS
```

iii. The notes justify this as the shared stimulus-relative time axis required by the decoder format.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No raw signal is transformed. The input is simply the fixed vector of bin centers from `-0.59` to `1.49` s, repeated for every retained trial.

ii.
```python
EDGES=np.arange(OFF0,OFF1+DT/2,DT); CENTERS=(EDGES[:-1]+EDGES[1:])/2
```

```python
ins.append(np.vstack([CENTERS,np.full(len(CENTERS),b)]).astype(np.float32))
```

iii. The AI's notes treat this as a task-defined decoder input rather than a variable that needs further preprocessing.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The same `CENTERS` vector is used both to define the neural bins and to evaluate the continuous wheel and whisker traces, so the time input is on the exact same per-bin axis as the neural matrix.

ii.
```python
EDGES=np.arange(OFF0,OFF1+DT/2,DT); CENTERS=(EDGES[:-1]+EDGES[1:])/2
```

```python
times=stim[ti]+CENTERS
ww=np.interp(times,wtime,wspeed); mm=np.interp(times,ct,cm)
```

iii. The notes justify this as a common stimulus-centered tensor for all modalities.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `trials.probabilityLeft`, after rounding to one decimal place.

ii.
```python
prior=np.asarray(trials.probabilityLeft,float)
...
pr=np.round(prior,1); blocknum=np.zeros(len(trials),np.float32)
```

iii. The justification is implicit: the prior value is treated as the block identity signal.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI walks through the raw trial order and increments a counter while the rounded prior stays the same; the counter resets to zero when `probabilityLeft` changes. The count is computed before filtering and then indexed by retained trials.

ii.
```python
pr=np.round(prior,1); blocknum=np.zeros(len(trials),np.float32)
for i in range(1,len(trials)): blocknum[i]=0 if pr[i]!=pr[i-1] else blocknum[i-1]+1
```

```python
return dict(..., blocknum=blocknum[kept], ...)
```

iii. The notes justify preserving chronological order because block structure is part of the decoder inputs.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from `trials.choice`.

ii.
```python
choice=np.asarray(trials.choice,float)
```

```python
return dict(..., choice=choice[kept], ...)
```

iii. The notes state that raw IBL choice values are taken from the trial table and remapped to the task's binary coding.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Trials with choice not equal to `-1` or `1` are dropped. The retained values are then recoded as `-1 -> 0` and `+1 -> 1`, and the categorical label is broadcast across all time bins in the trial.

ii.
```python
valid=np.isfinite(stim)&np.isin(choice,[-1,1])&np.isin(np.round(prior,1),[.2,.5,.8])
```

```python
cc=0 if c==-1 else 1
outs.append(np.vstack([np.full(len(CENTERS),cc), ... ]).astype(np.int64))
```

iii. In the notes, the AI explicitly justified this as satisfying the task's left/right coding requirement, although that interpretation differs from the human reference.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from `trials.probabilityLeft`.

ii.
```python
prior=np.asarray(trials.probabilityLeft,float)
```

```python
return dict(..., prior=pr[kept], ...)
```

iii. The notes say this is the task's block prior variable and should be mapped directly from the trial table.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI rounds `probabilityLeft` to one decimal place, filters to `{0.2, 0.5, 0.8}`, maps those values to `{0, 1, 2}`, and broadcasts the resulting label across all bins of the trial.

ii.
```python
valid=np.isfinite(stim)&np.isin(choice,[-1,1])&np.isin(np.round(prior,1),[.2,.5,.8])
...
pr=np.round(prior,1)
```

```python
pp={.2:0,.5:1,.8:2}[float(p)]
outs.append(np.vstack([np.full(len(CENTERS),cc),np.full(len(CENTERS),pp), ... ]).astype(np.int64))
```

iii. The notes justify this as the exact categorical mapping requested in the task description.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from `wheel.timestamps` and `wheel.position`.

ii.
```python
wt=load_record(eid,g,srow,'wheel.timestamps'); wp=load_record(eid,g,srow,'wheel.position')
wpos,wtime=interpolate_position(wt,wp,freq=1000); wvel,_=velocity_filtered(wpos,fs=1000)
wspeed=np.abs(wvel)
```

iii. The notes justify this as matching the official IBL wheel-processing functions.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The code resamples wheel position to 1000 Hz with `interpolate_position`, computes filtered velocity with `velocity_filtered`, takes the absolute value as speed, and interpolates that continuous trace onto the stimulus-centered `CENTERS` grid for each retained trial.

ii.
```python
wpos,wtime=interpolate_position(wt,wp,freq=1000); wvel,_=velocity_filtered(wpos,fs=1000)
wspeed=np.abs(wvel)
```

```python
times=stim[ti]+CENTERS
ww=np.interp(times,wtime,wspeed)
```

iii. The notes explicitly cite the IBL wheel utilities as the intended reference behavior.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. After all sessions are loaded, the AI pools every aligned continuous wheel-speed value across the converted sessions, computes the global `1/3` and `2/3` quantiles, and discretizes each trial's wheel-speed trace with `np.digitize`.

ii.
```python
wheel=np.concatenate([np.concatenate(x['wheel']) for x in sessions])
wq=np.quantile(wheel,[1/3,2/3])
```

```python
outs.append(np.vstack([..., np.digitize(w,wq), ...]).astype(np.int64))
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI justifies pooled tertiles as producing comparable low/medium/high categories across sessions.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel-speed trace is interpolated at `stimOn_times + CENTERS`, so it shares the same stimulus-centered bin grid as the neural counts.

ii.
```python
times=stim[ti]+CENTERS
ww=np.interp(times,wtime,wspeed)
```

iii. The notes justify this as the required common time axis for all decoder inputs and outputs.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `<side>Camera.times` and `<side>Camera.ROIMotionEnergy`, preferring the left camera and falling back to the right camera.

ii.
```python
for v in ('left','right'):
 try:
  ct=load_record(eid,g,srow,f'{v}Camera.times'); cm=load_record(eid,g,srow,f'{v}Camera.ROIMotionEnergy')
  ...
  if n>10 and np.isfinite(cm).mean()>.95: side=v; break
```

iii. The notes justify this as following the reference code's left-camera preference with right-camera fallback.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion-energy trace is used directly. After choosing the camera side, the code truncates times and values to a common length, checks that most samples are finite, and interpolates the trace onto the stimulus-centered `CENTERS` grid for each retained trial.

ii.
```python
ct=load_record(eid,g,srow,f'{v}Camera.times'); cm=load_record(eid,g,srow,f'{v}Camera.ROIMotionEnergy')
n=min(len(ct),len(cm)); ct=np.asarray(ct[:n],float); cm=np.asarray(cm[:n],float)
if n>10 and np.isfinite(cm).mean()>.95: side=v; break
```

```python
times=stim[ti]+CENTERS
mm=np.interp(times,ct,cm)
```

iii. The notes say motion energy is already produced by the IBL pipeline and should not be recomputed from video.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. As with wheel speed, the AI pools all aligned whisker-motion-energy samples across the converted sessions, computes global tertile thresholds, and discretizes each per-trial whisker trace with `np.digitize`.

ii.
```python
whisk=np.concatenate([np.concatenate(x['whisk']) for x in sessions])
mq=np.quantile(whisk,[1/3,2/3])
```

```python
outs.append(np.vstack([..., np.digitize(m,mq)]).astype(np.int64))
```

iii. The notes justify global pooled tertiles as creating dataset-wide categories rather than per-session categories.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The trace is interpolated at `stimOn_times + CENTERS`, so it shares the neural time axis exactly.

ii.
```python
times=stim[ti]+CENTERS
mm=np.interp(times,ct,cm)
```

iii. The justification is the same common-grid argument used for neural and wheel data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missingness by retrying downloads, skipping missing probe datasets via `except KeyError`, falling back from left to right camera, using fallback atlas arrays when anatomy files are missing, dropping trials with invalid labels or uncovered behavior windows, and raising an error if a session ends up with no good neurons or fewer than two valid trials.

ii.
```python
for attempt in range(5):
 try:
  urllib.request.urlretrieve(url,tmp)
  ...
 except Exception as e:
  tmp.unlink(missing_ok=True)
  if attempt==4: raise RuntimeError(f'download failed {url}: {e}')
```

```python
for v in ('left','right'):
 try:
  ...
 except Exception as e:
  print(f'Camera {v} unavailable for {eid}: {e}', flush=True)
  continue
if side is None: raise ValueError('no valid whisker motion stream')
```

```python
except Exception:
 try: atlas=np.asarray(load_record(eid,g,srow,f'{probe}/pykilosort/channels.brainLocationIds_ccf_2017',True))
 except Exception: atlas=np.zeros(max(ch.max()+1,1),int)
...
if len(kept)<2: raise ValueError('fewer than two valid trials')
```

iii. The notes justify this as conservative exclusion rather than imputation, with extra download/anatomy fallbacks needed because the AI chose direct S3 access instead of ONE.

## 10-a. What are the most time-consuming steps of the code?

i. Based on the code and notes, the AI's slowest steps are downloading the raw payloads from S3 and then looping through every retained trial, probe, and spike to build binned neural matrices. The notes also discuss the potential cost of processing the full release and motivate the 10-session subset partly on that basis.

ii.
```python
urllib.request.urlretrieve(url,tmp)
```

```python
for ti in idx:
 ...
 for st,sc,good in probe_data:
  ...
  for c,b in zip(cid,bins):
   ...
```

iii. `CONVERSION_NOTES.md` Step 6 says full-release conversion would require hundreds of GB and highlights per-spike Python work as a major cost, so the AI intentionally limited scope.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities are the trial loop, the probe-within-trial loop, and especially the inner spike loop that updates `M[j, b]` one spike at a time. Trial-wise interpolation of wheel and whisker traces could also be vectorized further.

ii.
```python
for ti in idx:
  times=stim[ti]+CENTERS
  ww=np.interp(times,wtime,wspeed); mm=np.interp(times,ct,cm)
```

```python
for st,sc,good in probe_data:
 ...
 bins=np.floor((rel-OFF0)/DT).astype(int)
 for c,b in zip(cid,bins):
  j=lut.get(int(c));
  if j is not None and 0<=b<M.shape[1]: M[j,b]+=1
```

iii. The notes mention "per-spike Python work" and binary-search slicing, implying the AI was aware of this efficiency tradeoff but accepted it for the smaller chosen subset.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly rebuilds the `lut` cluster-to-row dictionary for every trial and every probe, repeatedly interpolates wheel and whisker traces one trial at a time, and repeatedly computes `stim[ti] + CENTERS` inside the loop.

ii.
```python
for ti in idx:
  times=stim[ti]+CENTERS
  ww=np.interp(times,wtime,wspeed); mm=np.interp(times,ct,cm)
  ...
  for st,sc,good in probe_data:
   ...
   M=np.zeros((len(good),len(CENTERS)),np.float32); lut={int(c):j for j,c in enumerate(good)}
```

iii. This is not directly justified in the notes; it appears to be a simplicity tradeoff in the implementation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores `labels` and `trial_idx` in each session dictionary but never uses them in the final saved dataset. It also keeps continuous `wheel` and `whisk` traces for every retained trial only to compute global thresholds and then discards them after converting outputs to categories. Optional plotting is likewise outside the downstream decoder input.

ii.
```python
return dict(eid=eid,subject=str(subject),neural=neural,regions=regions,labels=np.array(labels_all),trial_idx=kept,
            choice=choice[kept],prior=pr[kept],blocknum=blocknum[kept],wheel=rawwheel,whisk=rawwhisk,camera=side)
```

```python
wheel=np.concatenate([np.concatenate(x['wheel']) for x in sessions]); whisk=np.concatenate([np.concatenate(x['whisk']) for x in sessions])
wq=np.quantile(wheel,[1/3,2/3]); mq=np.quantile(whisk,[1/3,2/3])
```

```python
if args.show_processing:
 for x in sessions[:2]: plot_session(x,ROOT/f"processing_{x['eid']}.png")
```

iii. The notes justify pooled thresholds and diagnostic plots, but the extra retained fields themselves are not explicitly justified.
