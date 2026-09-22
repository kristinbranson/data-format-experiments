# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively finds every local `behavior_ophys_experiment_*.nwb` under `/app/data`, sorts the paths, and loads each directly with AllenSDK. It does not use the project cache/experiment table or group files before loading.

ii.
```python
paths=sorted(Path(args.data_dir).rglob('behavior_ophys_experiment_*.nwb'))
...
ds=BehaviorOphysExperiment.from_nwb_path(str(p))
```

iii. The trajectory identified the files as local Visual Behavior Ophys NWBs and chose to process all 284 downloaded files one at a time for memory efficiency. It explicitly reasoned that an experiment file should be preserved as a session because it has its own neural population and timestamps.

## 1-b. How are the data split into subjects?

i. A subject ID is read from each retained experiment's metadata. The final subject list is the sorted unique set, and each session receives the corresponding index.

ii.
```python
subject_ids.append(str(m['mouse_id']))
subjects=sorted(set(subject_ids)); smap={x:i for i,x in enumerate(subjects)}
'subject_idx':np.asarray([smap[x] for x in subject_ids],dtype=np.int64)
```

iii. The trajectory inspected metadata across files and used `mouse_id` as AllenSDK's animal identifier.

## 1-c. How are the data split into sessions?

i. Every NWB/`ophys_experiment_id` is treated as a separate session, even when multiple experiments are imaging planes from the same `ophys_session_id`.

ii.
```python
for si,p in enumerate(paths):
    ds=BehaviorOphysExperiment.from_nwb_path(str(p))
    ...
    neural.append(sn)
```

iii. The agent reasoned that each experiment has a distinct cell population and ophys stream and therefore should remain a separate neural session. The trajectory later reaffirmed this while processing multi-plane groups.

## 1-d. How are the data split into trials?

i. Trials come from `ds.trials`. For each retained row, frames span the experimental `start_time` through `stop_time`; the start uses a left insertion point and the end a right insertion point, producing variable-length trials.

ii.
```python
for tid,row in trials.iterrows():
    lo=int(np.searchsorted(ts,float(row.start_time),side='left'))
    hi=int(np.searchsorted(ts,float(row.stop_time),side='right'))
    if hi<=lo: continue
    tt=ts[lo:hi]
```

iii. The agent chose native experimental trial boundaries so pre-change and post-change activity and time-varying targets are retained.

## 1-e. How are trials filtered based on quality controls?

i. Only rows marked Go or Catch are retained; aborted and auto-rewarded rows are removed. Empty frame windows are skipped, and an experiment is omitted if fewer than two trials remain.

ii.
```python
keep=(trials['go'].astype(bool)|trials['catch'].astype(bool)) \
     & ~trials['aborted'].astype(bool) & ~trials['auto_rewarded'].astype(bool)
...
if hi<=lo: continue
...
if len(sn)<2: ... continue
```

iii. This directly follows the instruction to include Go/Catch and exclude aborted/auto-rewarded trials, plus the validator's two-trial minimum.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from the AllenSDK `dff_traces['dff']` arrays and uses `ophys_timestamps` as its time axis.

ii.
```python
ts=np.asarray(ds.ophys_timestamps,float)
dff=np.vstack(ds.dff_traces['dff'].values).astype(np.float32)
```

iii. The agent concluded that SDK dF/F is the appropriate calcium-activity measure and described it as already demixed, neuropil-corrected, detrended, and curated.

## 2-b. How is the `neural` data processed?

i. Cell traces within one experiment are vertically stacked and cast to `float32`, checked against the timestamp length, and sliced by trial. No normalization, filtering, or temporal resampling is added. Neurons from simultaneous imaging-plane experiments are not combined.

ii.
```python
dff=np.vstack(ds.dff_traces['dff'].values).astype(np.float32)
if dff.shape[1]!=len(ts): raise ValueError(...)
...
sn.append(dff[:,lo:hi])
```

iii. The agent relied on the AllenSDK pipeline's existing fluorescence processing and wanted to preserve native signals.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional cell-level filter is applied; all cells exposed in the released `dff_traces` table are used.

ii.
```python
dff=np.vstack(ds.dff_traces['dff'].values).astype(np.float32)
```

iii. The agent stated that the released AllenSDK traces already contain ROIs passing dataset cell/ROI curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples are indexed on native ophys timestamps from trial start to trial stop. Thus each segment begins at the first frame at or after `start_time` and includes through the last frame at or before `stop_time`.

ii.
```python
lo=int(np.searchsorted(ts,float(row.start_time),side='left'))
hi=int(np.searchsorted(ts,float(row.stop_time),side='right'))
sn.append(dff[:,lo:hi])
```

iii. The agent used ophys timestamps, as required, and retained complete trial windows rather than imposing a fixed change-centered window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied: arrays remain at each experiment's native ophys frames. However, metadata hard-codes `1000/31` = 32.26 ms rather than measuring the actual timestamp interval (about 93 ms in the reference).

ii.
```python
sn.append(dff[:,lo:hi])
...
'time_bin_size':float(1000/31.0)
```

iii. The trajectory correctly recognized native ophys sampling but appears to have inferred an incorrect 31 Hz rate; it did not justify the hard-coded value.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity comes from `ds.stimulus_presentations`, principally `start_time`, `end_time`, and `image_name`.

ii.
```python
stim=ds.stimulus_presentations
...
name=pr.get('image_name',np.nan)
```

iii. The agent examined the stimulus-presentation schema and chose it to represent the actual flashed image rather than treating a trial's initial/change names as continuously displayed.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Each trial starts as the integer code for `gray`. Valid, non-omitted image presentations are assigned global integer codes and written only over their presentation interval; omitted and inter-flash periods remain gray.

ii.
```python
image=np.zeros(nt,dtype=np.int16)
...
if not isinstance(name,str) or name=='omitted': continue
if name not in image_to_code:
    image_to_code[name]=len(image_values); image_values.append(name)
...
if b>a: image[a:b]=image_to_code[name]
```

iii. The agent justified this from the task wording “image presented during the non-grey screen” and the experiment's 250-ms flashes separated by gray intervals.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Overlapping presentations are converted to indices by searching within the same trial ophys timestamp vector used for neural slicing.

ii.
```python
sub=stim[(stim.start_time<=tt[-1]) & (stim.end_time>=tt[0])]
a=int(np.searchsorted(tt,float(pr.start_time),side='left'))
b=int(np.searchsorted(tt,float(pr.end_time),side='left'))
if b>a: image[a:b]=image_to_code[name]
```

iii. The agent chose one label per native ophys bin so output and neural columns have identical lengths.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations.is_change` and presentation start times.

ii.
```python
if bool(pr.get('is_change',False)) and a<nt: change[a]=1
```

iii. The agent reasoned that `is_change` marks a real identity change and avoids labeling image-to-gray transitions or Catch sham changes.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector is created and exactly the first ophys bin of every overlapping presentation marked `is_change` is set to one.

ii.
```python
change=np.zeros(nt,dtype=np.int16)
...
if bool(pr.get('is_change',False)) and a<nt: change[a]=1
```

iii. The trajectory described this as a sparse, one-frame impulse representing change onset.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is inherently binary: 0 means no true change onset in that frame and 1 means a true change onset. There is no numerical threshold.

ii.
```python
change=np.zeros(nt,dtype=np.int16)
...
change[a]=1
```

iii. The raw `is_change` boolean already supplies the category.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Presentation onset is mapped into the trial's ophys timestamp vector; the corresponding neural column is labeled 1.

ii.
```python
a=int(np.searchsorted(tt,float(pr.start_time),side='left'))
...
change[a]=1
```

iii. This uses the same ophys-based trial vector as neural data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses the AllenSDK `running_speed` table's `timestamps` and `speed` columns.

ii.
```python
run=interp_finite(ts,ds.running_speed['timestamps'],ds.running_speed['speed'])
```

iii. The agent identified this as the SDK's transient-corrected, low-pass running-wheel signal.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Finite samples are linearly interpolated to ophys timestamps, with endpoint values extended outside the sampled range. Quintiles are then computed separately over the whole recording.

ii.
```python
good=np.isfinite(t)&np.isfinite(x)
return np.interp(t_new,t[good],x[good]).astype(np.float32)
...
run_bin,run_edges=quintiles(run)
```

iii. Interpolation aligns clocks; per-recording quantiles were chosen to avoid across-mouse running-wheel calibration differences.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The 20th, 40th, 60th, and 80th percentiles of each experiment's full interpolated running trace define codes 0–4; ties use right insertion.

ii.
```python
edges=np.nanquantile(x,[.2,.4,.6,.8])
return np.searchsorted(edges,x,side='right').astype(np.int16), edges.tolist()
```

iii. The agent wanted five equal-percentile classes within each recording and noted tie robustness.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is interpolated to the full ophys timebase first, then sliced with the same `lo:hi` indices as dF/F.

ii.
```python
run=interp_finite(ts,...)
...
y=np.vstack((image,change,run_bin[lo:hi],...))
sn.append(dff[:,lo:hi])
```

iii. The agent noted the behavior streams are synchronized and sampled at least as fast as ophys, making interpolation appropriate.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses eye-tracking `timestamps` and `pupil_area`, not the reference's `pupil_width` and `likely_blink` fields.

ii.
```python
eye=ds.eye_tracking
area=interp_finite(ts,eye['timestamps'],eye['pupil_area'])
```

iii. The agent observed that no literal diameter field was central in its inspection and chose area as a size measurement from which an equivalent diameter can be calculated.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Finite pupil-area samples are linearly interpolated to ophys time, negative values are clipped to zero, and equivalent circular diameter is calculated as `2*sqrt(area/pi)`. It is then binned per recording. Blink rows are not explicitly excluded.

ii.
```python
area=interp_finite(ts,eye['timestamps'],eye['pupil_area'])
diameter=(2*np.sqrt(np.maximum(area,0)/np.pi)).astype(np.float32)
pupil_bin,pupil_edges=quintiles(diameter)
```

iii. The agent justified equivalent circular diameter as a monotonic transformation of area, so percentile membership is unchanged.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Per-experiment 20/40/60/80 percentile boundaries of equivalent diameter produce integer codes 0–4.

ii.
```python
edges=np.nanquantile(x,[.2,.4,.6,.8])
return np.searchsorted(edges,x,side='right').astype(np.int16), edges.tolist()
```

iii. The agent chose within-recording quintiles to reduce eye-camera and animal calibration differences.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil area is interpolated onto `ophys_timestamps`, transformed/binned there, and sliced with the same trial indices as neural data.

ii.
```python
area=interp_finite(ts,eye['timestamps'],eye['pupil_area'])
...
y=np.vstack((...,pupil_bin[lo:hi],...))
sn.append(dff[:,lo:hi])
```

iii. The shared ophys timebase guarantees column-wise alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome uses the trial table's mutually exclusive `hit`, `miss`, `false_alarm`, and `correct_reject` booleans.

ii.
```python
if bool(row['hit']): return 0
if bool(row['miss']): return 1
if bool(row['false_alarm']): return 2
if bool(row['correct_reject']): return 3
```

iii. The agent treated these as the four canonical outcomes for retained Go/Catch trials and raises an error if none applies.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The four outcomes map to codes 0–3. Although static per trial, the code is repeated over all time bins to fit the rectangular output matrix.

ii.
```python
oc=outcome_code(row)
...
np.full(nt,oc,dtype=np.int16)
```

iii. The code comment explicitly explains that repetition reconciles a static target with the shared time-varying output representation.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Non-finite behavior samples are omitted before interpolation; no samples produces zeros, one produces a constant, and ordinary gaps are bridged linearly. Extrapolation uses nearest endpoints. Negative pupil area is clipped. Timestamp mismatch raises, empty trials are skipped, missing outcome raises, and sessions with fewer than two trials are skipped. There is no per-session exception handler.

ii.
```python
good=np.isfinite(t)&np.isfinite(x)
if good.sum()==0: return np.zeros(...)
if good.sum()==1: return np.full(...)
return np.interp(...)
...
if dff.shape[1]!=len(ts): raise ValueError(...)
if hi<=lo: continue
```

iii. The agent emphasized robustness to sparse/missing behavior and performed full-format and finiteness validation after conversion.

## 9-a. What are the most time-consuming steps of the code?

i. Loading and parsing all 284 large NWBs through AllenSDK, materializing full-session dF/F, and serializing the multi-gigabyte pickle dominate runtime.

ii.
```python
ds=BehaviorOphysExperiment.from_nwb_path(str(p))
dff=np.vstack(ds.dff_traces['dff'].values).astype(np.float32)
...
pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory anticipated AllenSDK loading as the bottleneck, monitored the sequential full conversion for several minutes, and released session objects between files.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The loop over trials and nested `stimulus_presentations.iterrows()` loop could be reduced by bulk-searching trial/presentation boundaries and assigning intervals more directly. Outcome selection is also scalar per row.

ii.
```python
for tid,row in trials.iterrows():
    ...
    for _,pr in sub.iterrows():
```

iii. The trajectory did not explicitly analyze vectorization; it prioritized clarity and the I/O-heavy full run.

## 9-c. What processing does the code repeat multiple times?

i. For every trial it filters the entire stimulus-presentation table for overlap, then searches presentation boundaries in the trial-local timestamp vector. Shared trials are independently processed again in every imaging-plane NWB belonging to the same behavioral/ophys session.

ii.
```python
sub=stim[(stim.start_time<=tt[-1]) & (stim.end_time>=tt[0])]
for _,pr in sub.iterrows():
    a=int(np.searchsorted(tt,float(pr.start_time),side='left'))
```

iii. The repeated behavior processing follows from the agent's deliberate one-experiment-per-session model; it did not identify it as duplication.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Continuous running, interpolated pupil area, and equivalent diameter are created only to produce categorical bins and then deleted. Trial-local `tt`, overlapping-presentation DataFrames, and loop variable `tid` are temporary; `tid` is unused. Full-session bins outside retained trials are computed but never saved. Garbage collection is forced after each experiment.

ii.
```python
run=interp_finite(...); area=interp_finite(...)
diameter=(2*np.sqrt(...)).astype(np.float32)
run_bin,run_edges=quintiles(run); pupil_bin,pupil_edges=quintiles(diameter)
...
del ds,dff,run,area,diameter,run_bin,pupil_bin; gc.collect()
```

iii. These intermediates support interpolation and per-recording percentile estimation; the agent intentionally deletes them to control memory, though computing bins over non-trial periods is unnecessary for the final trial-only outputs.
