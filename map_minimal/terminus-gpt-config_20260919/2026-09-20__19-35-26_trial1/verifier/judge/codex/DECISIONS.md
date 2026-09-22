# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globs every `sub-*/*.nwb` file, sorts the paths, opens each with `NWBHDF5IO`, and reads the subject, trials, units, acquisitions, and spike data. It skips only a file having no classifier-good units.

ii.
```python
files=sorted(glob.glob(os.path.join(ROOT,'sub-*','*.nwb')))
result=one_file(path)
```
```python
with NWBHDF5IO(path,'r',load_namespaces=True) as io:
    n=io.read(); tr=n.trials; u=n.units
```

iii. The trajectory identified NWB as the canonical synchronized release and one file as one session. It reports finding 174 files and retaining the 173 paper-qualified sessions.

## 1-b. How are the data split into subjects?

i. Subject identity is read from each NWB's `n.subject.subject_id`. Unique subjects are accumulated in first-seen file order, and each retained session gets the corresponding integer index.

ii.
```python
subject=s(n.subject.subject_id)
if sub not in data['subjects']: data['subjects'].append(sub)
data['subject_idx'].append(data['subjects'].index(sub))
```

iii. The agent considered the NWB subject field the canonical mouse identifier. Sorted file traversal makes the first-seen ordering deterministic.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Its trial lists and neuron-region vector are appended once to the top-level session lists; the file basename is saved in session metadata.

ii.
```python
data['neural'].append(neu); data['input'].append(inp); data['output'].append(out)
info={'file':os.path.basename(path),'subject':subject,'n_trials':ntr,
      'n_good_units':len(keep), ...}
```

iii. The trajectory concluded that the release's file boundaries are session boundaries and that 173 analyzed sessions are expected.

## 1-d. How are the data split into trials?

i. Trial rows come directly from `n.trials`. The agent requires one go-cue timestamp per row and builds one neural/input/output array for every row.

ii.
```python
ntr=len(tr)
go=np.asarray(ev['go_start_times'].timestamps[:],dtype=float)
if len(go)!=ntr:
    raise RuntimeError(...)
for j,g in enumerate(go):
    ...
    neural.append(fr); inputs.append(...); outputs.append(out)
```

iii. It found that go cues uniquely map trials, while sample events may repeat after early licking, so it used rows/go cues as the trial definition.

## 1-e. How are trials filtered based on quality controls?

i. They are not filtered. Every trials-table row is retained, including free-water trials and behavioral trials outside unit observation intervals. Only a session with no good units is skipped.

ii.
```python
ntr=len(tr)
...
for j,g in enumerate(go):
```

iii. The agent reasoned that the sessions were already behaviorally curated. Although it noticed `is_good_trials`, it did not implement trial-level QC and did not investigate `obs_intervals` or `free_water`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `units['spike_times']` for units whose `classification` is `good`, with `go_start_times` supplying each trial's absolute bin edges.

ii.
```python
cls=np.asarray([s(x).lower() for x in u['classification'][:]])
keep=np.flatnonzero(cls=='good')
spikes=[np.asarray(u['spike_times'][int(i)],float) for i in keep]
all_edges=go[:,None]+EDGES[None,:]
```

iii. The trajectory tied `classification == 'good'` to the newer classifier QC described in the supplied methods and treated spike times as the canonical neural stream.

## 2-b. How is the `neural` data processed?

i. Sorted spike timestamps are counted in half-open bins using `searchsorted` differences and divided by 0.05 s to produce firing rates in spikes/s. There is no smoothing or normalization.

ii.
```python
frcube=np.empty((len(keep),ntr,len(CENTERS)),dtype=np.float32)
for k,st in enumerate(spikes):
    frcube[k]=np.diff(np.searchsorted(st,all_edges,side='left'),axis=1)/DT
```

iii. The agent explicitly replaced a slow unit-by-trial histogram with one search over all trial edges per unit, reasoning that it preserves the same half-open counts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only `classification == 'good'` units are retained. A session with no such units is omitted. Other unit metrics and per-trial unit validity are not used.

ii.
```python
keep=np.flatnonzero(cls=='good')
if not len(keep):
    return None
```

iii. The agent selected this field because it represents the supplied classifier-based QC. It inspected the excluded file and found all classifier/anatomy labels missing, consistent with 174 source files versus 173 analyzed sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. A fixed relative edge grid is added to every absolute go-cue timestamp, and spikes are counted within those absolute intervals.

ii.
```python
EDGES=np.arange(OFF0, OFF1+DT/2, DT, dtype=np.float64)
all_edges=go[:,None]+EDGES[None,:]
```

iii. The agent determined that spikes and behavioral events share the NWB master clock, so no offset correction or interpolation is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. There are 80 non-overlapping 50-ms bins over `[-2.5, 1.5)` seconds relative to go cue. Raw timestamps are binned once; there is no later rebinning.

ii.
```python
DT=.05; OFF0=-2.5; OFF1=1.5
EDGES=np.arange(OFF0, OFF1+DT/2, DT, dtype=np.float64)
CENTERS=(EDGES[:-1]+EDGES[1:])/2
```

iii. The trajectory chose 80 bins because the requested four-second window divided by 50 ms naturally yields 80 and the target requires consistent timepoints.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `BehavioralEvents/sample_start_times`, `go_start_times`, and the common bin centers. The last sample onset at or before each go cue is treated as that trial's tone.

ii.
```python
sample=np.asarray(ev['sample_start_times'].timestamps[:],dtype=float)
q=sample[sample < g+1e-9]
sample_for_trial[j]=q[-1] if len(q) else np.nan
```

iii. The agent found that early licking can replay the sample epoch, so the last preceding sample is the relevant tone.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Each absolute bin-center time is subtracted from the selected tone onset, yielding elapsed seconds as `float32`.

ii.
```python
times=g+CENTERS
tone_elapsed=(times-sample_for_trial[j]).astype(np.float32)
```

iii. It regarded subtraction on the shared master clock as sufficient; no scaling beyond seconds is applied.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Values are evaluated at the same go-relative bin centers whose surrounding edges define the neural bins.

ii.
```python
times=g+CENTERS
tone_elapsed=(times-sample_for_trial[j]).astype(np.float32)
fr=frcube[:,j,:].copy()
```

iii. The shared center grid was chosen to synchronize every time-varying stream with the 80 neural bins.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses the acquisition event timestamps `photostim_start_times` and `photostim_stop_times`.

ii.
```python
pstart=np.asarray(ev['photostim_start_times'].timestamps[:],float)
pstop=np.asarray(ev['photostim_stop_times'].timestamps[:],float)
```

iii. The agent found explicit, synchronized stimulation intervals in the NWB and preferred them to reconstructing intervals from trial fields.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. It initializes an all-zero vector and sets a bin to one when its center is within any half-open stimulation interval.

ii.
```python
photo=np.zeros(len(CENTERS),dtype=np.float32)
for a,b in zip(pstart,pstop): photo[(times>=a)&(times<b)]=1
```

iii. The trajectory states that the decoder requires a time-varying binary state, so stimulation is represented at every bin center.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute stimulation intervals and absolute neural-bin centers share the NWB clock; interval membership is evaluated at each neural bin center.

ii.
```python
times=g+CENTERS
photo[(times>=a)&(times<b)]=1
```

iii. The agent relied on the synchronized NWB master clock and applied no additional correction.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from trials-table `trial_instruction` and `outcome`: hit selects the instructed side, miss selects the opposite side, and ignore means no lick.

ii.
```python
instr=np.asarray([s(x).lower() for x in tr['trial_instruction'][:]])
outcome=np.asarray([s(x).lower() for x in tr['outcome'][:]])
if outcome[j]=='ignore': choice=2
elif outcome[j]=='hit': choice=0 if instr[j]=='left' else 1
elif outcome[j]=='miss': choice=1 if instr[j]=='left' else 0
```

iii. The trajectory reasoned that actual response is fully determined by instructed side plus correctness, with ignore representing no response.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is encoded as 0 left, 1 right, or 2 no lick, then broadcast across all 80 bins.

ii.
```python
out=np.vstack((np.full(len(CENTERS),choice,np.int8), ...))
```

iii. The agent confirmed the decoder accepts time-varying outputs and broadcast trial-level labels so all four outputs can share one rectangular array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trials-table `outcome` field.

ii.
```python
outcome=np.asarray([s(x).lower() for x in tr['outcome'][:]])
```

iii. The agent observed that the field directly contains the requested categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. `ignore`, `miss`, and `hit` map to 0, 1, and 2 and are repeated over all bins.

ii.
```python
omap={'ignore':0,'miss':1,'hit':2}
np.full(len(CENTERS),omap[outcome[j]],np.int8)
```

iii. The mapping follows the requested output order; broadcasting gives a common output shape.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes from the trials-table `early_lick` field.

ii.
```python
early=np.asarray([s(x).lower() for x in tr['early_lick'][:]])
```

iii. The agent found this trial label directly in NWB, so it did not derive it from lick timestamps.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Values beginning with `early` map to 1 and all others to 0; the result is repeated across all bins.

ii.
```python
ecat=1 if early[j].startswith('early') else 0
np.full(len(CENTERS),ecat,np.int8)
```

iii. This implements the requested yes/no categories while tolerating the source string wording.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses `Camera0_side_TongueTracking` timestamps and data, with column 1 as y-position and column 2 as tracking confidence.

ii.
```python
tongue=n.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
tt=np.asarray(tongue.timestamps[:],float)
td=np.asarray(tongue.data[:],dtype=np.float32)
```

iii. The trajectory identified the three tracking columns as x, y, and DeepLabCut confidence and regarded this as the synchronized tongue stream.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with finite y and confidence at least 0.9 define the session percentile population. For each neural bin center, the nearest camera frame is selected; its y is used only if visible, otherwise category 3 is emitted. There is no within-bin averaging.

ii.
```python
visible=np.isfinite(td[:,1]) & np.isfinite(td[:,2]) & (td[:,2]>=DLC_VISIBLE)
q40,q60=np.percentile(td[visible,1],[40,60])
ix=np.searchsorted(tt,times); ix=np.clip(ix,1,len(tt)-1)
ix-=((times-tt[ix-1]) <= (tt[ix]-times))
vis=(td[ix,2]>=DLC_VISIBLE)&np.isfinite(td[ix,1])
```

iii. The agent selected 0.9 as a conventional confidence threshold and interpreted the requested percentiles as percentiles of all confident session samples. It chose nearest-frame sampling to synchronize video to bin centers.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Session-wide raw visible-frame y values provide the 40th and 60th percentiles. Visible values below q40 are 0, values from q40 through q60 are 1, values above q60 are 2, and invisible values are 3.

ii.
```python
ycat=np.full(len(CENTERS),3,dtype=np.int8)
ycat[vis & (td[ix,1]<q40)]=0
ycat[vis & (td[ix,1]>=q40) & (td[ix,1]<=q60)]=1
ycat[vis & (td[ix,1]>q60)]=2
```

iii. The 40/60 boundaries and per-session scope come directly from the task. The agent used confidence to operationalize “not visible.”

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The closest video frame to each absolute neural-bin center is selected using timestamp search.

ii.
```python
times=g+CENTERS
ix=np.searchsorted(tt,times); ix=np.clip(ix,1,len(tt)-1)
ix-=((times-tt[ix-1]) <= (tt[ix]-times))
```

iii. The agent reasoned that both streams share the NWB clock and chose center sampling rather than aggregating all frames falling inside each 50-ms neural interval.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. A session with no good unit is skipped; absent tone history becomes NaN; an entirely non-visible tongue session gets NaN percentile edges; non-finite/invisible tongue samples map to category 3. A go/trial count mismatch raises an error. Missing spike coverage and free-water trials are not handled.

ii.
```python
if not len(keep): return None
if len(go)!=ntr: raise RuntimeError(...)
sample_for_trial[j]=q[-1] if len(q) else np.nan
if visible.any(): q40,q60=np.percentile(...)
else: q40,q60=np.nan,np.nan
ycat=np.full(len(CENTERS),3,dtype=np.int8)
```

iii. The agent investigated the all-unlabelled session and deliberately omitted it. It otherwise relied on the release being curated and did not inspect observation intervals, leaving some trials with fabricated all-zero neural activity.

## 10-a. What are the most time-consuming steps of the code?

i. Reading large NWBs and dense tongue arrays, loading ragged spike trains, per-unit spike binning, retaining a large in-memory result, and serializing the 12.31-GB pickle dominate. The initial nested histogram implementation was especially slow.

ii.
```python
td=np.asarray(tongue.data[:],dtype=np.float32)
spikes=[np.asarray(u['spike_times'][int(i)],float) for i in keep]
for k,st in enumerate(spikes):
    frcube[k]=np.diff(np.searchsorted(st,all_edges,side='left'),axis=1)/DT
pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory measured only about four sessions/minute with nested histograms, then reported a major speedup after vectorizing across trials; final conversion still produced a 12.31-GB object.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The original unit-by-trial histogram loop was vectorized across trials. Remaining loops include per-unit ragged spike searches, per-trial stream/output construction, scanning every stimulation interval for every trial, and repeated Python loops for subject/region indexing. The photostimulation work could especially be restricted or vectorized.

ii.
```python
for k,st in enumerate(spikes):
    frcube[k]=np.diff(np.searchsorted(st,all_edges,side='left'),axis=1)/DT
for j,g in enumerate(go):
    ...
    for a,b in zip(pstart,pstop): photo[(times>=a)&(times<b)]=1
```

iii. The agent explicitly recognized and fixed the costly unit-by-trial histogram loop. It left the per-unit loop because spike vectors are ragged and did not discuss the quadratic trial-by-all-stimulation-interval scan.

## 10-c. What processing does the code repeat multiple times?

i. For every trial it scans all session photostimulation intervals, performs nearest-frame timestamp searches, allocates/broadcasts categorical vectors, and copies the trial's firing-rate slice. Subject and region membership/index searches are also repeated linearly during assembly.

ii.
```python
for j,g in enumerate(go):
    fr=frcube[:,j,:].copy()
    ...
    for a,b in zip(pstart,pstop): ...
    ix=np.searchsorted(tt,times)
```
```python
if r not in data['brain_regions']: data['brain_regions'].append(r)
rid.append(data['brain_regions'].index(r))
```

iii. The trajectory focused on removing repeated spike histograms but gave no explicit justification for these remaining repeated operations.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads and bins trials that the reference later excludes for absent spike observations/free water, producing neural/input/output arrays that should not be in the dataset. It also computes and stores metadata percentiles and repeatedly copies matrices, though those are minor compared with the invalid retained trials.

ii.
```python
for j,g in enumerate(go):
    fr=frcube[:,j,:].copy()
    ...
info={...,'tongue_y_percentiles':[float(q40),float(q60)],...}
```

iii. The agent believed sessions were already curated, so it did not recognize the discarded/invalid trial processing. It did not identify any unnecessary downstream work in its final rationale.
