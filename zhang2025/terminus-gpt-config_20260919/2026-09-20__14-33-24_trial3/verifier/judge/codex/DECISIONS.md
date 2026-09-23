# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the ONE API as the main loader. It starts from `bwm_release.csv`, uses each row's `lab`, `subject`, `date`, and `session_number` to construct a filesystem path under `/app/data/one_cache`, then scans ALF directories directly. It chooses a "best" trial table by requiring a full set of trial columns and taking the candidate with the largest row count and latest lexical path. It likewise resolves camera, wheel, and probe files by direct globbing.

ii. 
```python
ROOT = Path('/app/data/one_cache')
COHORT = Path('/app/code/code_zhang2025/data/bwm_release.csv')

def session_path(r):
    return ROOT / str(r.lab) / 'Subjects' / str(r.subject) / str(r.date) / f'{int(r.session_number):03d}'
```

```python
def full_trial_table(alf: Path):
    for p in alf.glob('**/_ibl_trials.table.pqt'):
        ...
    return max(candidates,key=lambda z:(z[0],z[1]))[2:]
```

iii. In `CONVERSION_NOTES.md`, the AI says it avoided generic ONE/`SessionLoader` loading because the staged cache had mixed revisions and could select partial legacy objects; it therefore treated explicit file resolution as a robustness fix.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are taken from the `subject` column of `bwm_release.csv`. During assembly, the script records one subject string per retained session and then forms `subjects` by first appearance order, with `subject_idx` built by indexing back into that list.

ii. 
```python
rel=pd.read_csv(COHORT,index_col=0); sessions=rel.drop_duplicates('eid',keep='first')
...
neural_all.append(neural); ...; subj.append(str(r.subject))
...
subjects=list(dict.fromkeys(subj)); subject_idx=np.array([subjects.index(x) for x in subj],dtype=np.int32)
```

iii. The notes say `bwm_release.csv` is the authoritative cohort table and already maps each retained session to a subject, so no extra derivation is needed.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values in `bwm_release.csv`. The script drops duplicate rows on `eid`, uses one retained record per session for behavior preprocessing, and later gathers all probe rows for that same `eid` during neural conversion.

ii. 
```python
rel=pd.read_csv(COHORT,index_col=0); sessions=rel.drop_duplicates('eid',keep='first')
...
for k,(_,r) in enumerate(sessions.iterrows(),1):
    eid=str(r.eid)
```

```python
probe_rows=rel[rel.eid.astype(str)==eid]
for _,pr in probe_rows.iterrows():
    m,rg,mx=bin_probe(pr,stim)
```

iii. The justification in the notes is that the Zhang cohort CSV enumerates the exact 459-session release and each unique `eid` is the session unit.

## 1-d. How are the data split into trials?

i. Trials come from rows of the selected `_ibl_trials.table.pqt` file. The script computes a boolean trial mask on that table, aligns wheel and camera traces against each trial's `stimOn_times`, and then keeps the shared set of valid trial indices.

ii. 
```python
def preprocess_behavior(row,eid,cache_file):
    alf=session_path(row)/'alf'; trial_p,x=full_trial_table(alf)
    base=trial_mask(x); stim=x.stimOn_times.to_numpy(dtype=float)
    ...
    common=np.intersect1d(ci,wi,assume_unique=True)
```

iii. The notes say the full trials table is already one row per trial; the additional work is to decide which rows survive the event/coverage filters.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials with non-missing `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, and `feedbackType`; reaction time between 0.08 and 2.0 s; nonzero choice; `feedback_times - goCue_times <= 10`; and complete finite wheel and camera coverage across the full `[-0.5, 1.5]` stimulus-aligned window. It then requires at least two jointly valid wheel+camera trials per session.

ii. 
```python
def trial_mask(x):
    m=np.ones(len(x),bool)
    for c in ['stimOn_times','choice','feedback_times','probabilityLeft','firstMovement_times','feedbackType']:
        m &= x[c].notna().to_numpy()
    rt=x.firstMovement_times.to_numpy()-x.stimOn_times.to_numpy()
    m &= (rt >= .08) & (rt <= 2.)
    m &= x.choice.to_numpy()!=0
    m &= (x.feedback_times.to_numpy()-x.goCue_times.to_numpy() <= 10.)
    return m
```

```python
cam,ci=interp_trials(ct,cv,stim,base)
wt,wv=derive_wheel(alf); wheel,wi=interp_trials(wt,wv,stim,base)
common=np.intersect1d(ci,wi,assume_unique=True)
if len(common)<2: raise ValueError(f'only {len(common)} jointly valid trials')
```

iii. The AI justifies this as matching the broader Zhang trial mask plus an explicit intersection of wheel and camera validity masks, which it says is a deliberate fix for a bug in the original reference code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The trial neural matrices are derived from `spikes.times.npy` and `spikes.clusters.npy` for each probe. The script also uses `clusters.metrics.pqt`, `clusters.channels.npy`, and `channels.brainLocationIds_ccf_2017.npy` to map cluster ids and brain regions, but the per-bin neural values themselves come from spike times and cluster assignments.

ii. 
```python
return metrics, {n:local_or_glob(n) for n in ['spikes.times.npy','spikes.clusters.npy','clusters.channels.npy','channels.brainLocationIds_ccf_2017.npy']}
```

```python
st=np.load(p['spikes.times.npy'],mmap_mode='r'); sc=np.load(p['spikes.clusters.npy'],mmap_mode='r')
```

iii. The notes say the conversion should use the methods-paper representation of temporally binned spike-sorted electrophysiology, not imaging-style preprocessing.

## 2-b. How is the `neural` data processed?

i. For each probe and each retained trial, the AI bins spikes into 100 bins of width 20 ms over `[-0.5, 1.5]` s relative to `stimOn_times`. It keeps these as integer spike counts, not firing rates, and concatenates the per-probe count matrices within each session. The final per-trial matrix is `neurons x 100`, stored as `uint8` when possible and `uint16` otherwise.

ii. 
```python
tb=np.floor((spike_t-beg)/BIN).astype(int)
valid2=(tb>=0)&(tb<NBIN); flat=mapped[valid2]*NBIN+tb[valid2]
a=np.bincount(flat,minlength=n*NBIN).reshape(n,NBIN)
```

```python
if maxcount>255: dtype=np.uint16
else: dtype=np.uint8
ntr=len(stim); neural=[np.concatenate([pmats[p][j] for p in range(len(pmats))],axis=0).astype(dtype,copy=False) for j in range(ntr)]
```

iii. The notes explicitly say the AI chose "20 ms spike counts; all Kilosort clusters; probes merged by session" and defended integer storage as a lossless memory optimization for a very large dataset.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not filter neurons by cluster quality label and does not exclude `void` or other anatomical labels. It keeps all cluster rows listed in the selected `clusters.metrics.pqt` tables for retained sessions, then maps every cluster's channel to a Beryl acronym.

ii. 
```python
cluster_ids=metrics.cluster_id.to_numpy(dtype=int)
n=len(cluster_ids)
...
lookup=np.full(maxid+1,-1,dtype=np.int32); lookup[cluster_ids]=np.arange(n,dtype=np.int32)
```

```python
br=BrainRegions(); native=br.id2acronym(atlas[ch]); regions=br.acronym2acronym(native,mapping='Beryl').astype(str)
return mats,regions,max_count
```

iii. The AI's notes justify this as following the Zhang executable pipeline's `qc=None` behavior and prose about using "all Kilosort clusters", even though the data paper separately reports a good-unit subset.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to visual stimulus onset. For each retained trial, the script defines `beg = stimOn + OFF0` and `end = stimOn + OFF1`, slices spikes in that absolute interval, and bins them relative to `beg`, which corresponds to `[-0.5, 1.5]` s from stimulus onset.

ii. 
```python
OFF0, OFF1 = -0.5, 1.5
...
for s in stim:
    beg=s+OFF0; end=s+OFF1; ib=np.searchsorted(st,beg,'left'); ie=np.searchsorted(st,end,'left')
```

iii. The notes say the task-required common alignment event is `stimOn_times`, and that this should override variable-specific alignments used elsewhere in the papers.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural representation uses 20 ms bins and exactly 100 time bins per trial across a 2 s window. There is no further temporal rebinning beyond counting spikes directly into those bins.

ii. 
```python
BIN = 0.02
OFF0, OFF1 = -0.5, 1.5
NBIN = 100
```

```python
tb=np.floor((spike_t-beg)/BIN).astype(int)
```

iii. The notes say this matches the methods-paper common representation and the task's required 2 s stimulus-aligned decoder window.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The time input is not read from a separate raw array. It is a constructed relative-time vector anchored to each trial's `stimOn_times`, using the same stimulus-aligned trial windows that are also used for behavior interpolation.

ii. 
```python
REL_TIME = np.linspace(OFF0 + BIN, OFF1, NBIN, dtype=np.float32)
...
stim=x.stimOn_times.to_numpy(dtype=float)
```

iii. The notes say this is a task-defined input whose only raw anchor is the alignment event `stimOn_times`.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI creates a fixed 100-sample vector from `-0.48` to `+1.50` s using `np.linspace(OFF0 + BIN, OFF1, NBIN)`. This is then reused for every retained trial as the first input row.

ii. 
```python
REL_TIME = np.linspace(OFF0 + BIN, OFF1, NBIN, dtype=np.float32)
...
inp=[np.vstack((REL_TIME,np.full(NBIN,tib[j],np.float32))).astype(np.float32) for j in range(ntr)]
```

iii. The notes justify this as matching the "reference behavior interpolation" grid at trial bin ends rather than bin centers.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The AI uses the same stimulus-aligned trial window for neural binning and for the time input. However, the time row represents the interpolation grid `[-0.48, ..., 1.50]` rather than spike-bin centers; it is intended to match the behavioral resampling times used for wheel and whisker traces.

ii. 
```python
grid=stim[i]+REL_TIME.astype(float)
y=interp1d(tx,vy,kind='linear',fill_value='extrapolate')(grid)
```

```python
inp=[np.vstack((REL_TIME,np.full(NBIN,tib[j],np.float32))).astype(np.float32) for j in range(ntr)]
```

iii. The notes say this was an intentional choice so that the explicit time input shares the same sampled timestamps as the interpolated behavioral outputs.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the raw `probabilityLeft` sequence in the trials table. A new block is detected whenever `probabilityLeft` changes from one trial to the next.

ii. 
```python
def trial_in_block(prob):
    prob=np.asarray(prob)
    change=np.r_[True, prob[1:] != prob[:-1]]
```

iii. The notes say there is no explicit block id in the raw trials table, so block structure must be reconstructed from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI computes a 1-indexed within-block count on the full unfiltered trial sequence, resetting to 1 whenever `probabilityLeft` changes. It then selects the surviving trials and broadcasts each retained scalar value across all 100 time bins.

ii. 
```python
starts=np.maximum.accumulate(np.where(change,np.arange(len(prob)),0))
return (np.arange(len(prob))-starts+1).astype(np.float32)
```

```python
tib=trial_in_block(probs)
...
inp=[np.vstack((REL_TIME,np.full(NBIN,tib[j],np.float32))).astype(np.float32) for j in range(ntr)]
```

iii. The notes justify computing it before filtering so excluded trials still advance block position, which the AI viewed as preserving the animal's true place in the block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the raw `choice` column in the selected trials table and saved in the behavior cache before final output assembly.

ii. 
```python
choices=x.choice.to_numpy(dtype=float)
...
np.savez(cache_file, ... choice=choices[common].astype(np.int8), ...)
```

iii. The notes say this follows the standard IBL trial variable and excludes no-response trials earlier in the trial mask.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps raw choice `-1` to decoder class `0` and raw choice `+1` to decoder class `1`, then broadcasts that scalar label across all 100 time bins of the trial.

ii. 
```python
cmap={-1:0,1:1}
...
c=cmap[int(z['choice'][j])]
out.append(np.vstack((np.full(NBIN,c,np.uint8), ... )))
```

iii. The notes describe this as mapping native `-1/+1` choices to the task's `0/1` binary output, with no-response trials already removed.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the raw `probabilityLeft` column in the trials table and carried through the behavior cache as a per-trial scalar.

ii. 
```python
probs=x.probabilityLeft.to_numpy(dtype=float)
...
np.savez(cache_file, ... prior=probs[common].astype(np.float32), ...)
```

iii. The notes say the IBL task only uses the three protocol values `0.2`, `0.5`, and `0.8`, so the decoder output can be mapped directly from that column.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, checking each cached prior against the nearest allowed key and raising an error if it is not effectively exact. The chosen class is broadcast across the full trial.

ii. 
```python
pmap={0.2:0,0.5:1,0.8:2}
...
pv=float(z['prior'][j]); key=min(pmap,key=lambda q:abs(q-pv))
if abs(key-pv)>1e-6: raise ValueError(f'unexpected prior {pv}')
out.append(np.vstack((..., np.full(NBIN,pmap[key],np.uint8), ... )))
```

iii. The notes justify this as an exact task-specified recoding of the protocol's three prior levels.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`. The AI loads those raw arrays directly, reconstructs a dense wheel trajectory, differentiates it to velocity, and then takes the absolute value.

ii. 
```python
tp=newest(alf.glob('**/_ibl_wheel.timestamps.npy'))
pp=newest(alf.glob('**/_ibl_wheel.position.npy'))
...
t=np.asarray(np.load(tp),dtype=float); p=np.asarray(np.load(pp),dtype=float)
```

```python
pos,ti=interpolate_position(t,p,freq=1000)
vel,_=velocity_filtered(pos,1000)
return np.asarray(ti),np.abs(np.asarray(vel))
```

iii. The notes justify this as matching the Brainbox wheel-processing utilities used by the reference pipeline.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI interpolates raw wheel position to 1 kHz, computes filtered velocity with `velocity_filtered`, takes the absolute value, sorts and deduplicates timestamps, and linearly interpolates the resulting speed trace onto a 100-sample stimulus-aligned grid for each retained trial.

ii. 
```python
pos,ti=interpolate_position(t,p,freq=1000)
vel,_=velocity_filtered(pos,1000)
return np.asarray(ti),np.abs(np.asarray(vel))
```

```python
order=np.argsort(times,kind='stable'); times=times[order]; values=values[order]
keep=np.r_[True,np.diff(times)>0]; times=times[keep]; values=values[keep]
...
y=interp1d(tx,vy,kind='linear',fill_value='extrapolate')(grid)
```

iii. The notes say this was chosen specifically to reproduce Brainbox's wheel-velocity derivation and the reference interpolation logic.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. After all retained sessions have been behavior-preprocessed, the AI pools every retained wheel-speed sample across sessions and trials, computes global 1/3 and 2/3 quantiles, and assigns class `0/1/2` with `np.searchsorted(..., side='right')`.

ii. 
```python
def thresholds(values):
    x=np.concatenate([v.reshape(-1) for v in values])
    q=np.quantile(x,[1/3,2/3])
```

```python
wz=[np.load(cf)['wheel'] for _,_,cf in kept]
wthr=thresholds(wz)
...
wc=np.searchsorted(wthr,z['wheel'],side='right').astype(np.uint8)
```

iii. The notes justify this as giving all sessions a common physical meaning for "low/medium/high" wheel speed, rather than session-specific categories.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to each trial's `stimOn_times` and sampled on the fixed relative grid `REL_TIME = [-0.48, ..., 1.50]` used by the behavior interpolation code. That same 100-column grid is used when wheel classes are written into the output tensor.

ii. 
```python
grid=stim[i]+REL_TIME.astype(float)
y=interp1d(tx,vy,kind='linear',fill_value='extrapolate')(grid)
```

```python
wc=np.searchsorted(wthr,z['wheel'],side='right').astype(np.uint8)
out.append(np.vstack((..., wc[j], ...)))
```

iii. The notes justify this as using a single common stimulus-aligned temporal basis for neural counts and dynamic behavioral outputs.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from `<side>Camera.ROIMotionEnergy.npy` and `_ibl_<side>Camera.times.npy`, with left camera preferred and right camera used only as fallback when no valid left pair exists.

ii. 
```python
def paired_stream(alf: Path, side: str):
    times=list(alf.glob(f'**/_ibl_{side}Camera.times.npy'))
    vals=list(alf.glob(f'**/{side}Camera.ROIMotionEnergy.npy'))
```

```python
for sd in ('left','right'):
    tp,vp=paired_stream(alf,sd)
    if tp is not None: side=sd; break
```

iii. The notes say left-camera preference was chosen to mirror the reference behavior stream and because the papers describe the left whisker stream as the canonical 60 Hz source when available.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI uses the released motion-energy values directly, after sorting and deduplicating timestamps, and linearly interpolates them onto the same 100-sample per-trial stimulus-aligned grid used for wheel speed.

ii. 
```python
ct=np.load(tp,mmap_mode='r'); cv=np.load(vp,mmap_mode='r')
cam,ci=interp_trials(ct,cv,stim,base)
```

```python
order=np.argsort(times,kind='stable'); times=times[order]; values=values[order]
keep=np.r_[True,np.diff(times)>0]; times=times[keep]; values=values[keep]
...
y=interp1d(tx,vy,kind='linear',fill_value='extrapolate')(grid)
```

iii. The notes justify this as using the released whisker motion-energy trace as-is and only changing its sampling grid to match the decoder format.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Like wheel speed, whisker motion energy is discretized with global thresholds computed from all retained whisker samples pooled across sessions and trials, followed by `np.searchsorted(..., side='right')` into classes `0/1/2`.

ii. 
```python
mz=[np.load(cf)['whisker'] for _,_,cf in kept]
mthr=thresholds(mz)
...
mc=np.searchsorted(mthr,z['whisker'],side='right').astype(np.uint8)
```

iii. The notes say the goal was to keep one dataset-wide interpretation of "low/medium/high" whisker motion rather than making categories session-relative.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The whisker trace is aligned to each trial's `stimOn_times` and sampled on the same 100-sample relative grid used for wheel speed and the time input, then written into the output tensor as a time-varying categorical row.

ii. 
```python
grid=stim[i]+REL_TIME.astype(float)
y=interp1d(tx,vy,kind='linear',fill_value='extrapolate')(grid)
```

```python
mc=np.searchsorted(mthr,z['whisker'],side='right').astype(np.uint8)
out.append(np.vstack((..., mc[j])))
```

iii. The notes justify this as part of the single common stimulus-aligned temporal basis used for all dynamic outputs.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or inconsistent files usually trigger exclusion rather than imputation. The script skips sessions with no matched camera stream or fewer than two jointly valid wheel+camera trials, rejects wheel arrays with bad lengths, requires finite interpolated traces, raises on missing probe files or bad channel indices, and chooses explicit "best" revisions for mixed ALF objects.

ii. 
```python
if side is None: raise FileNotFoundError('no matched camera time/motion stream')
...
if len(common)<2: raise ValueError(f'only {len(common)} jointly valid trials')
```

```python
if len(t)!=len(p) or len(t)<20: raise ValueError('invalid wheel arrays')
...
if np.any((ch<0)|(ch>=len(atlas))): raise ValueError('cluster channel out of range')
```

iii. The AI's notes justify this as a robustness strategy for a staged cache with mixed revisions and incomplete camera coverage, preferring explicit exclusion over silent repair.

## 10-a. What are the most time-consuming steps of the code?

i. The AI's code spends most of its time in the two-pass session loop: first behavior preprocessing/interpolation and then per-probe spike loading and per-trial spike binning, followed by serializing a very large pickle.

ii. 
```python
for k,(_,r) in enumerate(sessions.iterrows(),1):
    info=preprocess_behavior(r,eid,cf)
```

```python
for _,pr in probe_rows.iterrows():
    m,rg,mx=bin_probe(pr,stim)
...
with open(args.outpicklefile,'wb') as f: pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes call out the dataset's size, memory-mapped spike processing, and large serialization cost as the main performance constraints.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining Python loops are over retained trials during behavior interpolation and spike binning, plus the loop that assembles final per-trial neural/input/output arrays. These loops could in principle be vectorized further, although the AI deliberately left the core trial slicing structure intact.

ii. 
```python
for i in idx:
    beg,end=stim[i]+OFF0,stim[i]+OFF1
    ...
    y=interp1d(tx,vy,kind='linear',fill_value='extrapolate')(grid)
```

```python
for s in stim:
    beg=s+OFF0; end=s+OFF1; ib=np.searchsorted(st,beg,'left'); ie=np.searchsorted(st,end,'left')
    ...
    a=np.bincount(flat,minlength=n*NBIN).reshape(n,NBIN)
```

iii. The notes explicitly mention that Python trial loops remain but were paired with memory mapping, binary search, and `np.bincount` to keep runtime acceptable.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats some work across its two-pass design. It writes per-session behavior caches, then reloads them once to compute global thresholds and again during final conversion. It also re-filters the cohort table to collect `probe_rows` for each `eid`, and `resolve_probe` re-scans probe directories for files each time a probe is processed.

ii. 
```python
wz=[np.load(cf)['wheel'] for _,_,cf in kept]; mz=[np.load(cf)['whisker'] for _,_,cf in kept]
...
for si,(r,eid,cf) in enumerate(kept):
    ts=time.time(); z=np.load(cf); stim=z['stim']; probe_rows=rel[rel.eid.astype(str)==eid]
```

```python
for p in base.glob('**/clusters.metrics.pqt'):
    ...
return metrics, {n:local_or_glob(n) for n in [...]}
```

iii. The notes acknowledge the two-pass cache design but justify it as avoiding the more expensive alternative of recomputing wheel filtering and interpolation during spike conversion.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The core conversion is mostly used, but the script does some extra work that is not part of the final decoder input: it accumulates an unused `logs` list, stores `trial_path` inside intermediate `.npz` caches even though that path is not copied into the final pickle, and optionally generates processing plots that are not consumed downstream.

ii. 
```python
kept=[]; logs=[]
...
info=preprocess_behavior(r,eid,cf); kept.append((r,eid,cf)); info['seconds']=time.time()-ts; logs.append(info)
```

```python
np.savez(cache_file, ... side=np.array(side),trial_path=np.array(str(trial_p)))
...
if args.show_processing and si<2: make_plot(...)
```

iii. The notes frame the caches and plots as validation/debugging aids rather than part of the final decoder dataset itself.
