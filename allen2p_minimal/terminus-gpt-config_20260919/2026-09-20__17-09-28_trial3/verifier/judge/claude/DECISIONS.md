# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Data are loaded by scanning the `/app/data` directory for all NWB files matching `behavior_ophys_experiment_*.nwb`. Each NWB file is loaded individually using `BehaviorOphysExperiment.from_nwb_path()` from the AllenSDK. No project cache or experiment table is used; the agent directly reads the local NWB files.

ii.
```python
paths=sorted(Path(args.data_dir).rglob('behavior_ophys_experiment_*.nwb'))
...
for si,p in enumerate(paths):
    ds=BehaviorOphysExperiment.from_nwb_path(str(p))
```

iii. The agent identified the NWB files on disk and chose to load them directly rather than through the SDK's `VisualBehaviorOphysProjectCache`. The agent reasoned that since all the data were already downloaded as NWB files, direct loading was sufficient and avoids the overhead of the cache infrastructure. (Trajectory step 11: "Create a documented sequential converter... It will use AllenSDK-loaded, quality-curated dF/F traces.")

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `mouse_id` field in each experiment's metadata. Unique mouse IDs are collected across all sessions and sorted to form the subjects list.

ii.
```python
m=ds.metadata; subject_ids.append(str(m['mouse_id']))
...
subjects=sorted(set(subject_ids)); smap={x:i for i,x in enumerate(subjects)}
```

iii. The agent used the standard AllenSDK metadata field `mouse_id` to identify subjects. This is the canonical identifier for each animal.

## 1-c. How are the data split into sessions?

i. Each individual NWB file (ophys experiment) is treated as a separate session. There is no grouping of multiple imaging planes from the same behavioral session. Each experiment has its own neuron population and ophys timestamp stream, so the agent treated it as an independent session.

ii.
```python
paths=sorted(Path(args.data_dir).rglob('behavior_ophys_experiment_*.nwb'))
...
for si,p in enumerate(paths):
    ...
    neural.append(sn); inputs.append(sx); outputs.append(sy)
```

iii. The agent reasoned (step 3): "the target session grouping likely should preserve each ophys experiment as a neural session because each has its own neurons and timestamps." And (step 11): "Each ophys experiment will be a session because it has a distinct neuron population, imaging plane, and ophys timestamp stream."

## 1-d. How are the data split into trials?

i. Trials are defined using the built-in `ds.trials` table. Each trial is segmented from `start_time` to `stop_time` using ophys timestamps, giving variable-length trial windows. Go and Catch trials are retained; aborted and auto-rewarded trials are excluded.

ii.
```python
trials=ds.trials
keep=(trials['go'].astype(bool)|trials['catch'].astype(bool)) \
     & ~trials['aborted'].astype(bool) & ~trials['auto_rewarded'].astype(bool)
trials=trials.loc[keep]
for tid,row in trials.iterrows():
    lo=int(np.searchsorted(ts,float(row.start_time),side='left'))
    hi=int(np.searchsorted(ts,float(row.stop_time),side='right'))
    if hi<=lo: continue
```

iii. The agent followed the instructions to include Go and Catch trials and exclude aborted and auto-rewarded trials. Trial boundaries use the SDK's built-in start/stop times with variable length.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) requiring `go` or `catch` to be true, (2) excluding `aborted` trials, (3) excluding `auto_rewarded` trials, (4) skipping trials where `hi <= lo` (empty window), and (5) skipping sessions with fewer than 2 retained trials.

ii.
```python
keep=(trials['go'].astype(bool)|trials['catch'].astype(bool)) \
     & ~trials['aborted'].astype(bool) & ~trials['auto_rewarded'].astype(bool)
...
if hi<=lo: continue
...
if len(sn)<2:
    print('  skipped: fewer than two retained trials',flush=True); ...continue
```

iii. The agent followed the task instructions for trial filtering. The minimum 2-trial requirement ensures decoder evaluation is possible.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dff_traces` (dF/F calcium fluorescence traces) accessed via `ds.dff_traces['dff']`.

ii.
```python
dff=np.vstack(ds.dff_traces['dff'].values).astype(np.float32)
```

iii. The agent reasoned (step 8): "The official tutorials primarily demonstrate detrended dF/F as the neural activity... this strongly supports using dF/F rather than inferred events." And (step 9): "dF/F is the least transformed neural activity and is what the tutorials use for trial comparisons."

## 2-b. How is the `neural` data processed?

i. The dF/F traces are vertically stacked and cast to float32. No additional processing (filtering, normalization, etc.) is applied beyond what the AllenSDK pipeline already provides.

ii.
```python
dff=np.vstack(ds.dff_traces['dff'].values).astype(np.float32)
...
sn.append(dff[:,lo:hi])
```

iii. The agent relied on the AllenSDK's pre-processing pipeline (demixing, neuropil correction, detrending) and did not apply additional transformations.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied. All neurons present in the SDK's `dff_traces` are included, relying on the SDK's own cell/ROI curation.

ii. N/A (no filtering code)

iii. The agent stated in the code docstring: "SDK dF/F is used as neural activity. It is demixed, neuropil-corrected, detrended, and contains only ROIs passing the released dataset's cell/ROI curation."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to ophys timestamps. Each trial's data is extracted by finding the ophys frame indices corresponding to `start_time` (left searchsorted) and `stop_time` (right searchsorted), giving variable-length windows.

ii.
```python
ts=np.asarray(ds.ophys_timestamps,float)
...
lo=int(np.searchsorted(ts,float(row.start_time),side='left'))
hi=int(np.searchsorted(ts,float(row.stop_time),side='right'))
...
sn.append(dff[:,lo:hi])
```

iii. The instructions specify "Temporally align based on ophys timestamp." The agent uses `searchsorted` with `side='left'` for start and `side='right'` for stop to find the inclusive ophys frame window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. Data is kept at the native ophys frame rate. The time bin size is hardcoded as `1000/31.0` ms (~32.26 ms) in the metadata.

ii.
```python
'time_bin_size':float(1000/31.0),
```

iii. The agent observed from inspecting one experiment that the ophys sampling rate was ~31 Hz and hardcoded this value. No resampling is performed.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, specifically the `image_name` column, `start_time`, and `end_time` of each presentation. The agent also uses the `is_change` field. Periods between stimulus flashes default to "gray" (code 0).

ii.
```python
stim=ds.stimulus_presentations
...
sub=stim[(stim.start_time<=tt[-1]) & (stim.end_time>=tt[0])]
for _,pr in sub.iterrows():
    name=pr.get('image_name',np.nan)
    if not isinstance(name,str) or name=='omitted': continue
    if name not in image_to_code:
        image_to_code[name]=len(image_values); image_values.append(name)
    a=int(np.searchsorted(tt,float(pr.start_time),side='left'))
    b=int(np.searchsorted(tt,float(pr.end_time),side='left'))
    if b>a: image[a:b]=image_to_code[name]
```

iii. The agent chose to use stimulus_presentations rather than trial-level image names to get fine-grained, per-frame image identity including the gray inter-stimulus intervals. This captures the actual visual stimulus on screen at each ophys frame.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes using a dynamic dictionary built as new images are encountered. The code initializes with `{'gray': 0}` and adds new images as they appear. For each trial, the image identity starts as all zeros (gray), then each overlapping stimulus presentation fills in the corresponding frames with the image code. Omitted presentations are skipped.

ii.
```python
image_to_code={'gray':0}; image_values=['gray']
...
image=np.zeros(nt,dtype=np.int16)  # gray between 250-ms flashes
...
if name not in image_to_code:
    image_to_code[name]=len(image_values); image_values.append(name)
a=int(np.searchsorted(tt,float(pr.start_time),side='left'))
b=int(np.searchsorted(tt,float(pr.end_time),side='left'))
if b>a: image[a:b]=image_to_code[name]
```

iii. The agent's approach captures the actual stimulus on screen at each frame, including gray periods between flashes, which is more temporally precise than using trial-level image names.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed per ophys frame within the trial window using `searchsorted` on the ophys timestamps. The same `lo:hi` indices define both neural and stimulus data.

ii.
```python
tt=ts[lo:hi]
...
a=int(np.searchsorted(tt,float(pr.start_time),side='left'))
b=int(np.searchsorted(tt,float(pr.end_time),side='left'))
```

iii. By using the same ophys timestamp array for both neural and stimulus alignment, the image identity is guaranteed to be aligned with the neural data frame-by-frame.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in the `stimulus_presentations` table. For each stimulus presentation that has `is_change=True`, the frame at its start time is marked as 1.

ii.
```python
if bool(pr.get('is_change',False)) and a<nt: change[a]=1
```

iii. The agent used the stimulus presentations' `is_change` flag to identify the actual image change events, rather than deriving it from the trials table.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary array is initialized to zeros. For each stimulus presentation within the trial that has `is_change=True`, only the single frame at the presentation start is set to 1. This creates a single-frame impulse at the change onset.

ii.
```python
change=np.zeros(nt,dtype=np.int16)
...
if bool(pr.get('is_change',False)) and a<nt: change[a]=1
```

iii. The agent chose a single-frame impulse representation. This marks the exact onset of the image change rather than a sustained window.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1). No thresholding is applied; it is directly computed as a binary indicator from the `is_change` field.

ii. See 4-b above.

iii. The instructions specify "binary variable. Have value of 1 right after a change in image identity, otherwise 0." The agent implemented this directly.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same alignment as image identity - computed per ophys frame within the trial window using `searchsorted` on the same timestamp array.

ii.
```python
a=int(np.searchsorted(tt,float(pr.start_time),side='left'))
...
if bool(pr.get('is_change',False)) and a<nt: change[a]=1
```

iii. Frame-level alignment is guaranteed by using the same ophys timestamp indices.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ds.running_speed`, using the `timestamps` and `speed` columns.

ii.
```python
run=interp_finite(ts,ds.running_speed['timestamps'],ds.running_speed['speed'])
```

iii. The agent used the AllenSDK's running_speed attribute, which provides the transient-corrected, 10-Hz low-pass filtered running speed signal.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is interpolated from its native timestamps to the ophys timebase using `np.interp` (via a custom `interp_finite` function that handles NaN/infinite values). It is then discretized into 5 equal-percentile bins using session-level quintile boundaries.

ii.
```python
def interp_finite(t_new, t, x):
    t=np.asarray(t,float); x=np.asarray(x,float)
    good=np.isfinite(t)&np.isfinite(x)
    ...
    return np.interp(t_new,t[good],x[good]).astype(np.float32)

run=interp_finite(ts,ds.running_speed['timestamps'],ds.running_speed['speed'])
run_bin,run_edges=quintiles(run)
```

iii. The agent chose session-level quintile binning rather than global binning, reasoning that calibration differences between running wheels across mice/sessions could make global bins inappropriate.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal-percentile bins (quintiles) computed per-session. The `quintiles` function computes edges at the 20th, 40th, 60th, and 80th percentiles and uses `searchsorted` with `side='right'` to assign bin codes 0-4.

ii.
```python
def quintiles(x):
    x=np.asarray(x,float)
    edges=np.nanquantile(x,[.2,.4,.6,.8])
    return np.searchsorted(edges,x,side='right').astype(np.int16), edges.tolist()
```

iii. The agent chose per-session quintiles to avoid cross-session calibration issues. The `side='right'` ensures values exactly on an edge boundary go to the lower bin.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the ophys timebase before trial segmentation, so it shares the same time indices as the neural data. The same `lo:hi` slice is used.

ii.
```python
run=interp_finite(ts,ds.running_speed['timestamps'],ds.running_speed['speed'])
...
y=np.vstack((image,change,run_bin[lo:hi],pupil_bin[lo:hi],...))
sn.append(dff[:,lo:hi])
```

iii. By pre-interpolating to ophys timestamps, alignment is guaranteed.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `pupil_area` in the `eye_tracking` table. The agent converts pupil area to equivalent circular diameter via `2*sqrt(area/pi)`.

ii.
```python
eye=ds.eye_tracking
area=interp_finite(ts,eye['timestamps'],eye['pupil_area'])
diameter=(2*np.sqrt(np.maximum(area,0)/np.pi)).astype(np.float32)
```

iii. The agent reasoned (step 8): "They use SDK pupil_area even where prose loosely says pupil diameter, suggesting pupil_area is the intended available pupil-size measure." The agent then converted area to diameter to match the output name.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area is first interpolated to the ophys timebase using `interp_finite` (which handles NaN/infinite values but does NOT filter blinks). The area is then converted to equivalent circular diameter (`2*sqrt(area/pi)`). Finally, it is discretized into 5 per-session quintile bins.

ii.
```python
area=interp_finite(ts,eye['timestamps'],eye['pupil_area'])
diameter=(2*np.sqrt(np.maximum(area,0)/np.pi)).astype(np.float32)
pupil_bin,pupil_edges=quintiles(diameter)
```

iii. The agent noted in the code docstring: "Equivalent circular diameter is derived from area (a monotone transform, so its percentile classes equal pupil-area percentile classes)."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed - discretized into 5 equal-percentile bins (quintiles) per session.

ii.
```python
pupil_bin,pupil_edges=quintiles(diameter)
```

iii. Per-session quintiles, same approach as running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed - pupil data is pre-interpolated to ophys timestamps and sliced with the same `lo:hi` indices.

ii.
```python
area=interp_finite(ts,eye['timestamps'],eye['pupil_area'])
...
y=np.vstack((image,change,run_bin[lo:hi],pupil_bin[lo:hi],...))
```

iii. Same frame-level alignment as neural and running data.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
def outcome_code(row):
    if bool(row['hit']): return 0
    if bool(row['miss']): return 1
    if bool(row['false_alarm']): return 2
    if bool(row['correct_reject']): return 3
    raise ValueError('Retained Go/Catch trial has no standard outcome')
```

iii. These four columns are the standard SDK trial outcome labels for the change detection task. The agent raises an error if no outcome matches, ensuring data integrity.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_reject). The code is static per trial but repeated across all time bins in the output matrix to share the rectangular format.

ii.
```python
oc=outcome_code(row)
y=np.vstack((image,change,run_bin[lo:hi],pupil_bin[lo:hi],
             np.full(nt,oc,dtype=np.int16)))
```

iii. The agent noted in metadata: `'trial_outcome_encoding':'Static per trial; repeated over time to share the temporal output matrix'`.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **NaN/infinite values in behavioral streams**: The `interp_finite` function filters out non-finite values before interpolation, using only valid data points. If all values are NaN, it returns zeros; if only one valid point, it fills with that constant.
- **Empty trials**: Trials where `hi <= lo` (no ophys frames) are skipped.
- **Sessions with few trials**: Sessions with fewer than 2 valid trials are skipped.
- **Negative pupil area**: `np.maximum(area, 0)` clamps negative areas to zero before sqrt.
- **Omitted stimuli**: Stimulus presentations with `image_name='omitted'` are skipped.

ii.
```python
def interp_finite(t_new, t, x):
    good=np.isfinite(t)&np.isfinite(x)
    if good.sum()==0: return np.zeros(len(t_new),dtype=np.float32)
    if good.sum()==1: return np.full(len(t_new),x[good][0],dtype=np.float32)
    return np.interp(t_new,t[good],x[good]).astype(np.float32)
...
if hi<=lo: continue
...
if len(sn)<2: ...continue
...
diameter=(2*np.sqrt(np.maximum(area,0)/np.pi)).astype(np.float32)
```

iii. The agent's `interp_finite` is a robust interpolation helper that gracefully handles edge cases without crashing.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each NWB file via `BehaviorOphysExperiment.from_nwb_path()`, which reads large neural and behavioral data arrays from disk. The agent observed this taking several seconds per file across 284 experiments.

ii. N/A

iii. The trajectory shows the full conversion took many minutes, with the agent polling for progress repeatedly (steps 14-33).

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop iterates over each stimulus presentation within each trial to compute image identity and change indicators. This nested loop (trials x presentations) could potentially be vectorized using array operations on the stimulus_presentations table.

ii.
```python
for _,pr in sub.iterrows():
    name=pr.get('image_name',np.nan)
    ...
    a=int(np.searchsorted(tt,float(pr.start_time),side='left'))
    b=int(np.searchsorted(tt,float(pr.end_time),side='left'))
    if b>a: image[a:b]=image_to_code[name]
```

iii. The stimulus-level loop within each trial is more complex than necessary. Since each trial already has `initial_image_name` and `change_image_name`, the per-presentation iteration could be avoided.

## 9-c. What processing does the code repeat multiple times?

i. The code processes each experiment fully in a single pass - there is no repeated processing. However, since each experiment is treated as a separate session (rather than grouping by ophys_session_id), behavioral data (running speed, eye tracking, trials) may be loaded redundantly for experiments that share the same behavioral session.

ii. N/A

iii. Multiple imaging planes from the same behavioral session share identical running speed, eye tracking, and trial data. By treating each experiment independently, this behavioral data is loaded and processed multiple times.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The conversion of pupil area to equivalent circular diameter (`2*sqrt(area/pi)`) is unnecessary because the subsequent quintile discretization produces the same bin assignments whether applied to area or diameter (since sqrt is a monotone transform). The agent acknowledges this in the docstring.

ii.
```python
diameter=(2*np.sqrt(np.maximum(area,0)/np.pi)).astype(np.float32)
```

iii. The agent noted: "Equivalent circular diameter is derived from area (a monotone transform, so its percentile classes equal pupil-area percentile classes)." This is correct but means the sqrt transformation is unnecessary computation.
