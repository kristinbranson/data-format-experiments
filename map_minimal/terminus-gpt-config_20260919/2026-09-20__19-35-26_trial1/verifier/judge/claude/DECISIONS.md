# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is stored as one NWB file per session under `data/sub-<subject_id>/`. All sessions are found with a glob over that layout, sorted, and each file is opened with `pynwb`. Subjects, trials, units, and behavioral events are read from within each file.

ii.
```python
ROOT='/app/data'
files=sorted(glob.glob(os.path.join(ROOT,'sub-*','*.nwb')))
...
for z,path in enumerate(files,1):
    result=one_file(path)
```

Loading one session:
```python
with NWBHDF5IO(path,'r',load_namespaces=True) as io:
    n=io.read(); tr=n.trials; u=n.units
    subject=s(n.subject.subject_id)
    ...
    ev=n.acquisition['BehavioralEvents'].time_series
    go=np.asarray(ev['go_start_times'].timestamps[:],dtype=float)
```

iii. The agent identified that the NWB release is the canonical data format and provides all needed fields in a structured, synchronized way. It used PyNWB over raw HDF5 access. The glob pattern captures all 174 NWB files across subject directories.

## 1-b. How are the data split into subjects?

i. Each NWB file records its animal in `n.subject.subject_id`. That value is read per session and used to build the `subjects` list incrementally, with `subject_idx` tracking each session's index into that list.

ii.
```python
subject=s(n.subject.subject_id)
...
if sub not in data['subjects']: data['subjects'].append(sub)
data['subject_idx'].append(data['subjects'].index(sub))
```

iii. The agent noted that subject ID is directly available in the NWB metadata, and that 28 subject directories containing 174 NWB files matched the dataset description.

## 1-c. How are the data split into sessions?

i. One NWB file corresponds to one session. No grouping or splitting is needed. Session order follows the sorted file list.

ii.
```python
files=sorted(glob.glob(os.path.join(ROOT,'sub-*','*.nwb')))
```

iii. The agent noted the NWB release stores one session per file and that the paper reports 173 analyzed sessions, with one session excluded for having no classifier-good units (reducing 174 to 173).

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`n.trials`), one row per behavioral trial. The number of go-cue events is checked against the trial count.

ii.
```python
ntr=len(tr)
...
go=np.asarray(ev['go_start_times'].timestamps[:],dtype=float)
...
if len(go)!=ntr:
    raise RuntimeError(f'{path}: {len(go)} go events != {ntr} trials')
```

iii. The agent noted that the trials table directly enumerates trials, with one go-cue event per trial. It performs a consistency check between go events and trial count.

## 1-e. How are trials filtered based on quality controls?

i. The AI does **not** filter trials. All trials from each session are included -- no exclusion of trials outside `obs_intervals`, no exclusion of `free_water` trials, and no exclusion of early-lick or ignore trials. The only filtering is at the session level: sessions with no `classification == 'good'` units are dropped.

ii.
```python
for j,g in enumerate(go):
    # All trials processed - no filtering logic
    fr=frcube[:,j,:].copy()
    ...
    neural.append(fr)
    inputs.append(...)
    outputs.append(out)
```

iii. The agent interpreted early lick and ignore trials as decoder output variables to predict rather than exclusion criteria (since the task specification lists "early lick" and "outcome" including "ignore" as decoder outputs). However, the agent did not address `obs_intervals` or `free_water` filtering at all -- it noted these fields exist but provided no explicit justification for omitting them.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units['spike_times']`, the sorted spike times of each unit. Only units with `classification == 'good'` contribute. Go-cue times from `BehavioralEvents/go_start_times` are used to place bin edges.

ii.
```python
spikes=[np.asarray(u['spike_times'][int(i)],float) for i in keep]
all_edges=go[:,None]+EDGES[None,:]
frcube=np.empty((len(keep),ntr,len(CENTERS)),dtype=np.float32)
for k,st in enumerate(spikes):
    frcube[k]=np.diff(np.searchsorted(st,all_edges,side='left'),axis=1)/DT
```

iii. The agent noted that spike_times is the only neural representation in the NWB files, so firing rates must be computed from it directly.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to per-bin firing rates in Hz. For each good unit, bin edges for all trials are constructed relative to go-cue times. `np.searchsorted` gives running spike counts at each edge, differencing gives counts per bin, and dividing by bin width (0.05s) gives Hz. No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
all_edges=go[:,None]+EDGES[None,:]
frcube=np.empty((len(keep),ntr,len(CENTERS)),dtype=np.float32)
for k,st in enumerate(spikes):
    frcube[k]=np.diff(np.searchsorted(st,all_edges,side='left'),axis=1)/DT
```

iii. The agent noted this matches the reference code's `sliding_histogram` approach with half-open bins. It initially used `np.histogram` per unit per trial, then optimized to vectorized `np.searchsorted`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` (case-insensitive) are kept. A session with no such units is dropped entirely. No individual quality metric thresholds are applied.

ii.
```python
cls=np.asarray([s(x).lower() for x in u['classification'][:]])
keep=np.flatnonzero(cls=='good')
if not len(keep):
    return None
```

iii. The agent stated: "Classifier label is the QC criterion described in methods.txt." It confirmed that one session had all classifier labels as NaN, reducing 174 to 173 sessions, matching the paper's count of 173 analyzed sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times and event times are on the same session-absolute NWB clock. Bin edges are constructed as go-cue time plus relative offsets, and spikes are binned against those edges using `searchsorted`.

ii.
```python
go=np.asarray(ev['go_start_times'].timestamps[:],dtype=float)
...
all_edges=go[:,None]+EDGES[None,:]
...
frcube[k]=np.diff(np.searchsorted(st,all_edges,side='left'),axis=1)/DT
```

iii. The agent noted that the reference code comments state spike_times are relative to go cue in the original data, but in NWB they use absolute timestamps, so conversion is done by adding go-cue time to relative edges.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Spike times are binned into 80 non-overlapping 50 ms bins spanning -2.5 s to +1.5 s relative to the go cue. The bin grid is defined once as 81 edges.

ii.
```python
DT=.05; OFF0=-2.5; OFF1=1.5
EDGES=np.arange(OFF0, OFF1+DT/2, DT, dtype=np.float64)
CENTERS=(EDGES[:-1]+EDGES[1:])/2
```

iii. The window and 50 ms bin width are specified by the task instructions. The agent chose `np.arange(OFF0, OFF1+DT/2, DT)` to ensure 81 edges are generated.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` (tone onsets) in the BehavioralEvents, together with the go cue of each trial. The tone for a trial is the last `sample_start_times` timestamp before that trial's go cue.

ii.
```python
sample=np.asarray(ev['sample_start_times'].timestamps[:],dtype=float)
...
sample_for_trial=np.empty(ntr,float)
for j,g in enumerate(go):
    q=sample[sample < g+1e-9]
    sample_for_trial[j]=q[-1] if len(q) else np.nan
```

iii. The agent noted: "Occasional extra sample events are replays after an early lick. Associate each trial with the last sample onset preceding its unique go cue."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, at each bin center (absolute time = go + center_offset), the elapsed time since tone onset is computed as `bin_absolute_time - sample_time_for_trial`.

ii.
```python
times=g+CENTERS
tone_elapsed=(times-sample_for_trial[j]).astype(np.float32)
```

iii. No additional processing beyond finding the tone onset and computing elapsed time.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both neural data and tone elapsed time use the same bin centers defined relative to the go cue. The bin centers are absolute times (`go + CENTERS`), and tone elapsed is computed at those same time points.

ii.
```python
times=g+CENTERS
tone_elapsed=(times-sample_for_trial[j]).astype(np.float32)
```

iii. The shared bin grid ensures alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_start_times` and `photostim_stop_times` in the BehavioralEvents time series -- these are global event timestamps for photostimulation intervals across the whole session.

ii.
```python
pstart=np.asarray(ev['photostim_start_times'].timestamps[:],float)
pstop=np.asarray(ev['photostim_stop_times'].timestamps[:],float)
```

iii. The agent read the NWB photostim event series and confirmed that photostimulation trials have explicit onset/offset timestamps.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time-varying signal. For each bin center (as an absolute time), the code checks whether it falls within any `[photostim_start, photostim_stop)` interval. If so, value is 1; otherwise 0.

ii.
```python
photo=np.zeros(len(CENTERS),dtype=np.float32)
for a,b in zip(pstart,pstop): photo[(times>=a)&(times<b)]=1
```

iii. The agent chose half-open interval checks `[start, stop)` matching the bin convention.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The photostim start/stop times are on the same session-absolute clock as the neural bin centers. The bin centers (`go + CENTERS`) are compared directly against the photostim intervals.

ii.
```python
times=g+CENTERS
...
for a,b in zip(pstart,pstop): photo[(times>=a)&(times<b)]=1
```

iii. The shared NWB master clock ensures alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `trial_instruction` (left/right) and `outcome` (hit/miss/ignore) in the trials table. No direct choice column exists.

ii.
```python
instr=np.asarray([s(x).lower() for x in tr['trial_instruction'][:]])
outcome=np.asarray([s(x).lower() for x in tr['outcome'][:]])
...
if outcome[j]=='ignore': choice=2
elif outcome[j]=='hit': choice=0 if instr[j]=='left' else 1
elif outcome[j]=='miss': choice=1 if instr[j]=='left' else 0
```

iii. The agent stated: "Choice should represent the actual response: right/left for hit or miss (derived from instruction plus correctness), and no lick for ignore." Hit means licked correctly (matching instruction), miss means licked the wrong side.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded as 0=left, 1=right, 2=no lick. Per-trial value broadcast across all 80 time bins.

ii.
```python
out=np.vstack((np.full(len(CENTERS),choice,np.int8), ...))
```

iii. The decoder expects `(d_output, T)` shaped outputs per trial, so per-trial values are broadcast.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From the `outcome` column of the trials table, which holds `'ignore'`, `'miss'`, and `'hit'`.

ii.
```python
outcome=np.asarray([s(x).lower() for x in tr['outcome'][:]])
...
omap={'ignore':0,'miss':1,'hit':2}
```

iii. The trials table stores outcome explicitly with the three categories requested.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to 0=ignore, 1=miss, 2=hit. Per-trial value broadcast across all 80 bins.

ii.
```python
omap={'ignore':0,'miss':1,'hit':2}
...
np.full(len(CENTERS),omap[outcome[j]],np.int8)
```

iii. Direct mapping from the instructions.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds strings like `'no early'` and `'early'`.

ii.
```python
early=np.asarray([s(x).lower() for x in tr['early_lick'][:]])
...
ecat=1 if early[j].startswith('early') else 0
```

iii. The trials table flags early licking explicitly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Binary: 1 if the early_lick field starts with `'early'`, 0 otherwise. Per-trial value broadcast across all 80 bins.

ii.
```python
ecat=1 if early[j].startswith('early') else 0
...
np.full(len(CENTERS),ecat,np.int8)
```

iii. Simple binary mapping.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose data is `(n_frames, 3)` = x, y, likelihood, with matching timestamps.

ii.
```python
tongue=n.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
tt=np.asarray(tongue.timestamps[:],float)
td=np.asarray(tongue.data[:],dtype=np.float32)
```

iii. This is the only tongue measurement in the NWB files.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with DLC confidence < 0.9 are considered invisible. Session-wide 40th and 60th percentiles are computed on all visible raw y-values (not binned means). Per bin, the nearest camera frame to the bin center is found; if visible, it is categorized by percentile thresholds; if not visible, category 3.

ii.
```python
DLC_VISIBLE=.9
visible=np.isfinite(td[:,1]) & np.isfinite(td[:,2]) & (td[:,2]>=DLC_VISIBLE)
if visible.any(): q40,q60=np.percentile(td[visible,1],[40,60])
else: q40,q60=np.nan,np.nan
...
ix=np.searchsorted(tt,times); ix=np.clip(ix,1,len(tt)-1)
ix-=((times-tt[ix-1]) <= (tt[ix]-times))
vis=(td[ix,2]>=DLC_VISIBLE)&np.isfinite(td[ix,1])
ycat=np.full(len(CENTERS),3,dtype=np.int8)
ycat[vis & (td[ix,1]<q40)]=0
ycat[vis & (td[ix,1]>=q40) & (td[ix,1]<=q60)]=1
ycat[vis & (td[ix,1]>q60)]=2
```

iii. The agent stated it uses "the conventional 0.9 threshold" for DLC visibility. It computes percentiles on raw visible frame values rather than on binned means.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Categories: 0 = below 40th percentile, 1 = 40th to 60th percentile (inclusive both sides), 2 = above 60th percentile, 3 = not visible. Percentile thresholds are computed per-session from all visible raw tongue y values.

ii.
```python
ycat[vis & (td[ix,1]<q40)]=0
ycat[vis & (td[ix,1]>=q40) & (td[ix,1]<=q60)]=1
ycat[vis & (td[ix,1]>q60)]=2
```

iii. The boundary definitions follow the task instructions (below 40th, 40th to 60th, above 60th).

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each bin center (absolute time = go + center_offset), the nearest camera frame is found using `searchsorted` on camera timestamps. The nearest-neighbor frame is used (no interpolation or averaging within the bin).

ii.
```python
ix=np.searchsorted(tt,times); ix=np.clip(ix,1,len(tt)-1)
ix-=((times-tt[ix-1]) <= (tt[ix]-times))
```

iii. The camera timestamps share the NWB master clock with spikes and events, so alignment is straightforward.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Two cases handled:
- **Session with no good units**: If classification is NaN for all units, the session is skipped entirely.
- **Missing tongue data**: If no frames have confidence >= 0.9, percentiles are set to NaN and all tongue bins become category 3.

Notably, trials outside `obs_intervals` and `free_water` trials are NOT handled -- they are included with potentially zero or meaningless neural data.

ii.
```python
if not len(keep):
    return None
...
if visible.any(): q40,q60=np.percentile(td[visible,1],[40,60])
else: q40,q60=np.nan,np.nan
```

iii. The agent addressed the NaN classification issue but did not address obs_intervals or free_water filtering.

## 10-a. What are the most time-consuming steps of the code?

i. Reading each NWB file with PyNWB is the dominant cost. Within a session, materializing spike times and computing per-unit searchsorted are the main computational steps. The agent initially used a slow per-unit-per-trial histogram approach but optimized to vectorized searchsorted.

ii.
```python
spikes=[np.asarray(u['spike_times'][int(i)],float) for i in keep]
all_edges=go[:,None]+EDGES[None,:]
frcube=np.empty((len(keep),ntr,len(CENTERS)),dtype=np.float32)
for k,st in enumerate(spikes):
    frcube[k]=np.diff(np.searchsorted(st,all_edges,side='left'),axis=1)/DT
```

iii. The agent noted the optimization improved throughput from ~4 sessions/minute to ~50 sessions/minute.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two loops remain: the per-unit loop for spike binning (one searchsorted per unit, but all trials vectorized), and the per-trial loop for processing outputs (iterating over trials to build inputs and outputs). Additionally, the tone-onset association loop iterates per trial.

ii.
```python
# Per-unit loop (partially vectorized)
for k,st in enumerate(spikes):
    frcube[k]=np.diff(np.searchsorted(st,all_edges,side='left'),axis=1)/DT

# Per-trial loop for all outputs
for j,g in enumerate(go):
    fr=frcube[:,j,:].copy()
    ...

# Per-trial tone association
for j,g in enumerate(go):
    q=sample[sample < g+1e-9]
    sample_for_trial[j]=q[-1] if len(q) else np.nan
```

iii. The per-trial output loop could be vectorized (as done in the reference), and the tone association could use searchsorted instead of a per-trial filter.

## 10-c. What processing does the code repeat multiple times?

i. The photostimulation check iterates over all global photostim intervals for every trial, even though most trials have zero or one relevant interval. The per-trial loop recomputes `times = g + CENTERS` each iteration.

ii.
```python
for j,g in enumerate(go):
    times=g+CENTERS
    ...
    for a,b in zip(pstart,pstop): photo[(times>=a)&(times<b)]=1
```

iii. The nested loop over photostim intervals within the trial loop is redundant work; most intervals are irrelevant to most trials.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code processes ALL trials in each session, including trials that fall outside `obs_intervals` (where no spike data was recorded) and `free_water` trials (which have no meaningful neural data). These trials contain zero or near-zero firing rates that would be misleading for downstream analysis. The reference solution filters these out.

ii.
```python
# No trial filtering - all ntr trials are processed
for j,g in enumerate(go):
    fr=frcube[:,j,:].copy()
    ...
    neural.append(fr)
```

iii. The agent did not address this issue explicitly.
