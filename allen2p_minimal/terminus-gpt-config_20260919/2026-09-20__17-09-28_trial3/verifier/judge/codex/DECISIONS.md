# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all data by recursively finding every local NWB file matching `behavior_ophys_experiment_*.nwb` under `/app/data`, then opening each file with `BehaviorOphysExperiment.from_nwb_path`. It does not use the Allen project cache or experiment table.

ii.
```python
paths=sorted(Path(args.data_dir).rglob('behavior_ophys_experiment_*.nwb'))
if args.limit is not None: paths=paths[:args.limit]
if not paths: raise FileNotFoundError('No behavior ophys experiment NWBs found')

for si,p in enumerate(paths):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ds=BehaviorOphysExperiment.from_nwb_path(str(p))
```

iii. In the trajectory, the AI said the provided data were large local per-experiment NWB files and decided to process them “every NWB sequentially” rather than reconstructing the dataset through the SDK cache. It justified this as a feasible, memory-conscious way to cover all downloaded files.

## 1-b. How are the data split into subjects?

i. Subjects are split by `mouse_id` from each NWB file’s metadata. The final `subjects` list is the sorted set of those IDs, and each session gets a subject index derived from that mapping.

ii.
```python
m=ds.metadata; subject_ids.append(str(m['mouse_id']))
...
subjects=sorted(set(subject_ids)); smap={x:i for i,x in enumerate(subjects)}
...
'subject_idx':np.asarray([smap[x] for x in subject_ids],dtype=np.int64),
```

iii. In the trajectory, the AI repeatedly described the dataset in terms of mouse IDs and reported subject counts from those metadata fields. There was no alternate subject definition proposed.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file, i.e. each `behavior_ophys_experiment`, as one session. It does not group multiple experiments by shared `ophys_session_id`.

ii.
```python
for si,p in enumerate(paths):
    ...
    ds=BehaviorOphysExperiment.from_nwb_path(str(p))
    ...
    neural.append(sn); inputs.append(sx); outputs.append(sy)
    m=ds.metadata; subject_ids.append(str(m['mouse_id'])); session_regions.append(str(m['targeted_structure']))
    session_info.append({'ophys_experiment_id':int(m['ophys_experiment_id']),
      'behavior_session_id':int(m['behavior_session_id']), 'ophys_session_id':int(m['ophys_session_id']),
```

iii. In the trajectory, the AI explicitly decided that “an ophys experiment is a session” because each file has one timestamp stream, one imaging plane, and one neuron population. Later it reiterated that “multi-plane experiment groups are being treated as separate neural sessions.”

## 1-d. How are the data split into trials?

i. Trials are taken from `ds.trials`. For each retained row, the AI extracts all ophys frames from trial `start_time` through `stop_time`, keeping native variable-length trials.

ii.
```python
trials=ds.trials
...
for tid,row in trials.iterrows():
    lo=int(np.searchsorted(ts,float(row.start_time),side='left'))
    hi=int(np.searchsorted(ts,float(row.stop_time),side='right'))
    if hi<=lo: continue
    tt=ts[lo:hi]; nt=len(tt)
    ...
    sn.append(dff[:,lo:hi])
```

iii. In the trajectory, the AI said official tutorials “slice streams from each trial’s start_time through stop_time,” and it decided to preserve “native ophys samples and variable trial durations” rather than time-warping trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only `go` or `catch` trials, removes `aborted` and `auto_rewarded` trials, drops any trial whose extracted frame window is empty, and discards sessions with fewer than two retained trials. It does not explicitly filter out trials with missing `change_time`.

ii.
```python
keep=(trials['go'].astype(bool)|trials['catch'].astype(bool)) \
     & ~trials['aborted'].astype(bool) & ~trials['auto_rewarded'].astype(bool)
trials=trials.loc[keep]
...
if hi<=lo: continue
...
if len(sn)<2:
    print('  skipped: fewer than two retained trials',flush=True); del ds; gc.collect(); continue
```

iii. In the trajectory, the AI stated it would “include Go/Catch and exclude aborted/auto-rewarded trials.” It also checked the validator specifically because the decoder requires at least two trials per session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived directly from the AllenSDK dF/F traces in `ds.dff_traces['dff']`.

ii.
```python
ts=np.asarray(ds.ophys_timestamps,float)
dff=np.vstack(ds.dff_traces['dff'].values).astype(np.float32)
if dff.shape[1]!=len(ts): raise ValueError(f'dF/F timestamp mismatch in {p}')
```

iii. In the trajectory, the AI explicitly compared dF/F to event traces and chose dF/F because the tutorials use it directly for trial comparisons and it is the “least transformed neural activity.”

## 2-b. How is the `neural` data processed?

i. The AI does almost no further neural processing: it vertically stacks all cell dF/F traces from one NWB file into a neuron-by-time matrix, converts to `float32`, and slices those traces per trial. It does not merge multiple imaging planes into one behavioral session.

ii.
```python
dff=np.vstack(ds.dff_traces['dff'].values).astype(np.float32)
...
sn.append(dff[:,lo:hi])
```

iii. In the trajectory, the AI justified this by saying SDK dF/F is already “demixed, neuropil-corrected, detrended” and that each ophys experiment should stand alone as a session. It therefore saw no need for further neural normalization or aggregation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level filtering is applied in the script. The AI relies on the released AllenSDK dataset’s existing ROI/cell curation.

ii.
```python
"""... SDK dF/F is used as neural activity. It is demixed, neuropil-corrected, detrended,
  and contains only ROIs passing the released dataset's cell/ROI curation.
"""
...
dff=np.vstack(ds.dff_traces['dff'].values).astype(np.float32)
```

iii. In the trajectory, the AI repeatedly described `dff_traces` as already quality curated by the Allen pipeline and chose to “retain all SDK-curated cells.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to native ophys timestamps within each trial window. For each trial, frames are selected between the trial’s `start_time` and `stop_time`; no secondary alignment to `change_time` is applied.

ii.
```python
lo=int(np.searchsorted(ts,float(row.start_time),side='left'))
hi=int(np.searchsorted(ts,float(row.stop_time),side='right'))
...
sn.append(dff[:,lo:hi])
...
'temporal_alignment_event':'Native ophys timestamps within each experimental trial (trial start to trial stop).',
```

iii. In the trajectory, the AI said it would use “ophys-grid trial slices” and “native ophys timestamps within each experimental trial,” following the tutorials’ start-to-stop trial slicing.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the data at the native ophys frame rate and does not rebin it. In metadata it hard-codes the time bin size as `1000/31.0` ms, implying about 31 Hz sampling.

ii.
```python
'metadata':{'task_description':'Visual change-detection task; decode flashed image identity, true image-change onset, running-speed quintile, pupil-diameter quintile, and four-class Go/Catch trial outcome from calcium activity.',
  'time_bin_size':float(1000/31.0),'temporal_alignment_event':'Native ophys timestamps within each experimental trial (trial start to trial stop).',
```

iii. In the trajectory, the AI said valid trials should “retain native start-to-stop boundaries and ~31 Hz bins rather than be time-warped,” so it deliberately avoided any resampling of the neural timebase.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `ds.stimulus_presentations`, specifically each presentation’s `image_name`, `start_time`, and `end_time`. The AI does not use `initial_image_name` and `change_image_name` from the trials table.

ii.
```python
stim=ds.stimulus_presentations
...
sub=stim[(stim.start_time<=tt[-1]) & (stim.end_time>=tt[0])]
for _,pr in sub.iterrows():
    name=pr.get('image_name',np.nan)
    ...
    a=int(np.searchsorted(tt,float(pr.start_time),side='left'))
    b=int(np.searchsorted(tt,float(pr.end_time),side='left'))
```

iii. In the trajectory, the AI said it would “map image intervals to image/gray categories” and use “true experimental image-change onsets,” which reflects its choice to use the stimulus presentation table rather than only trial-level before/after labels.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI initializes each trial’s image identity to `gray`, then fills in frames overlapped by non-omitted stimulus presentations with integer image codes. New image names are added to a global dictionary as they are encountered across files.

ii.
```python
session_info=[]; image_to_code={'gray':0}; image_values=['gray']
...
image=np.zeros(nt,dtype=np.int16)  # gray between 250-ms flashes
...
if not isinstance(name,str) or name=='omitted': continue
if name not in image_to_code:
    image_to_code[name]=len(image_values); image_values.append(name)
...
if b>a: image[a:b]=image_to_code[name]
```

iii. In the trajectory, the AI explicitly planned to “map image intervals to image/gray categories.” It also validated the resulting image label range on a two-file test set before launching the full conversion.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned by converting stimulus presentation start and end times into indices on the same per-trial ophys timestamp vector `tt` used for neural slicing.

ii.
```python
tt=ts[lo:hi]; nt=len(tt)
...
a=int(np.searchsorted(tt,float(pr.start_time),side='left'))
b=int(np.searchsorted(tt,float(pr.end_time),side='left'))
if b>a: image[a:b]=image_to_code[name]
```

iii. In the trajectory, the AI described this as aligning all streams to the “ophys grid” and to “native ophys timestamps,” so image identity uses the same trial-specific ophys frame indices as the neural data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus presentation table’s `is_change` flag together with each presentation’s onset time. It is not derived from the trial table’s `change_time` and `go` fields.

ii.
```python
sub=stim[(stim.start_time<=tt[-1]) & (stim.end_time>=tt[0])]
for _,pr in sub.iterrows():
    ...
    if bool(pr.get('is_change',False)) and a<nt: change[a]=1
```

iii. In the trajectory, the AI said it would “mark the first ophys frame at each true image change” and use “true experimental image-change onsets,” which is why it chose `stimulus_presentations.is_change`.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI creates a binary vector of zeros for each trial and sets only the first ophys frame of any presentation marked `is_change` to 1. This produces a sparse onset indicator rather than a sustained post-change window.

ii.
```python
change=np.zeros(nt,dtype=np.int16)
...
if bool(pr.get('is_change',False)) and a<nt: change[a]=1
```

iii. In the trajectory, the AI explicitly planned to “mark the first ophys frame at each true image change,” and later noted in its test-set sanity check that image change was a “sparse one-frame impulse.”

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is applied beyond direct binary coding: `0` for no change and `1` for a detected change onset frame.

ii.
```python
change=np.zeros(nt,dtype=np.int16)
...
'output_values':[image_values,['no_change','change'],
```

iii. The trajectory treats image change as a binary event variable rather than a continuous signal, so no discretization step was proposed.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is aligned by converting each change presentation onset into an index on the same trial-specific ophys timestamp vector `tt` used for neural activity.

ii.
```python
tt=ts[lo:hi]; nt=len(tt)
...
a=int(np.searchsorted(tt,float(pr.start_time),side='left'))
...
if bool(pr.get('is_change',False)) and a<nt: change[a]=1
```

iii. In the trajectory, the AI described all outputs as being generated on the same “ophys-grid trial slices,” so image change uses the same frame basis as the neural data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ds.running_speed['speed']` and its timestamps.

ii.
```python
run=interp_finite(ts,ds.running_speed['timestamps'],ds.running_speed['speed'])
```

iii. In the trajectory, the AI called `running_speed` the SDK’s already processed running measure and said it would use “SDK running speed” rather than recomputing behavior from raw wheel signals.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI linearly interpolates running speed to ophys timestamps with `np.interp` after dropping non-finite samples. It then discretizes the full-session running trace into quintiles for that recording and slices those per-frame quintile labels into each trial.

ii.
```python
def interp_finite(t_new, t, x):
    t=np.asarray(t,float); x=np.asarray(x,float)
    good=np.isfinite(t)&np.isfinite(x)
    if good.sum()==0: return np.zeros(len(t_new),dtype=np.float32)
    if good.sum()==1: return np.full(len(t_new),x[good][0],dtype=np.float32)
    return np.interp(t_new,t[good],x[good]).astype(np.float32)

def quintiles(x):
    x=np.asarray(x,float)
    edges=np.nanquantile(x,[.2,.4,.6,.8])
    return np.searchsorted(edges,x,side='right').astype(np.int16), edges.tolist()

run=interp_finite(ts,ds.running_speed['timestamps'],ds.running_speed['speed'])
run_bin,run_edges=quintiles(run)
...
y=np.vstack((image,change,run_bin[lo:hi],pupil_bin[lo:hi],
```

iii. In the trajectory, the AI justified session-level quintiles by saying they avoid “across-mouse calibration differences in running wheels.” It also said it would use interpolation onto the ophys grid so all streams share the same timebase.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five quintile categories computed separately within each recording session, yielding integer labels `0` through `4`.

ii.
```python
def quintiles(x):
    """Codes 0..4 using equal-percentile boundaries; robust to tied boundaries."""
    x=np.asarray(x,float)
    edges=np.nanquantile(x,[.2,.4,.6,.8])
    return np.searchsorted(edges,x,side='right').astype(np.int16), edges.tolist()
```

iii. In the trajectory, the AI explicitly chose “session-level quintiles” to avoid calibration differences between mice and sessions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is first resampled onto the full-session ophys timestamps and then sliced with the same trial indices `lo:hi` used for neural activity.

ii.
```python
run=interp_finite(ts,ds.running_speed['timestamps'],ds.running_speed['speed'])
...
lo=int(np.searchsorted(ts,float(row.start_time),side='left'))
hi=int(np.searchsorted(ts,float(row.stop_time),side='right'))
...
y=np.vstack((image,change,run_bin[lo:hi],pupil_bin[lo:hi],
...
sn.append(dff[:,lo:hi])
```

iii. In the trajectory, the AI repeatedly described this as aligning behavioral streams to the “ophys grid” before trial segmentation.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The AI derives pupil size from `ds.eye_tracking['pupil_area']` and eye-tracking timestamps, then converts area to an equivalent circular diameter. It does not use `pupil_width`.

ii.
```python
eye=ds.eye_tracking
area=interp_finite(ts,eye['timestamps'],eye['pupil_area'])
diameter=(2*np.sqrt(np.maximum(area,0)/np.pi)).astype(np.float32)
```

iii. In the trajectory, the AI noted that tutorials often use `pupil_area` even when prose says “pupil diameter,” so it chose an equivalent-diameter transform from area as its pupil-size measure.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI interpolates `pupil_area` to ophys timestamps while ignoring non-finite samples, converts the interpolated area to equivalent circular diameter, computes quintiles separately within each session, and uses those per-frame labels as the output. It does not explicitly remove blink frames via `likely_blink`.

ii.
```python
eye=ds.eye_tracking
area=interp_finite(ts,eye['timestamps'],eye['pupil_area'])
diameter=(2*np.sqrt(np.maximum(area,0)/np.pi)).astype(np.float32)
run_bin,run_edges=quintiles(run); pupil_bin,pupil_edges=quintiles(diameter)
...
y=np.vstack((image,change,run_bin[lo:hi],pupil_bin[lo:hi],
```

iii. In the trajectory, the AI justified the area-to-diameter conversion as a monotone transform whose percentile classes match those from area, and it used the same per-recording quintile rationale as for running speed.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five session-specific quintile categories with codes `0` through `4`.

ii.
```python
run_bin,run_edges=quintiles(run); pupil_bin,pupil_edges=quintiles(diameter)
...
'output_values':[image_values,['no_change','change'],
  ['quintile_1','quintile_2','quintile_3','quintile_4','quintile_5'],
  ['quintile_1','quintile_2','quintile_3','quintile_4','quintile_5'],
```

iii. In the trajectory, the AI said it would use “session-level quintiles” for both running and pupil to avoid across-session calibration differences.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil size is first interpolated to the ophys timestamp vector `ts` for the whole experiment, then each trial uses the same `lo:hi` indices as the neural data.

ii.
```python
area=interp_finite(ts,eye['timestamps'],eye['pupil_area'])
...
lo=int(np.searchsorted(ts,float(row.start_time),side='left'))
hi=int(np.searchsorted(ts,float(row.stop_time),side='right'))
...
y=np.vstack((image,change,run_bin[lo:hi],pupil_bin[lo:hi],
...
sn.append(dff[:,lo:hi])
```

iii. In the trajectory, the AI described running and pupil together as behavioral streams that should be interpolated onto the ophys timebase before trial slicing.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table fields `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
def outcome_code(row):
    if bool(row['hit']): return 0
    if bool(row['miss']): return 1
    if bool(row['false_alarm']): return 2
    if bool(row['correct_reject']): return 3
```

iii. In the trajectory, the AI said it would encode “four trial outcomes,” referring to these standard Allen trial labels.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps the four mutually exclusive booleans to integer codes `0..3`, then repeats the single outcome code across all time bins in the trial so it can live inside the same rectangular output matrix as the time-varying variables.

ii.
```python
oc=outcome_code(row)
...
y=np.vstack((image,change,run_bin[lo:hi],pupil_bin[lo:hi],
             np.full(nt,oc,dtype=np.int16)))
...
'trial_outcome_encoding':'Static per trial; repeated over time to share the temporal output matrix',
```

iii. In the trajectory, the AI explicitly said the “static trial outcome must be repeated across each trial’s time axis” because the decoder format stores outputs in one matrix.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing behavior samples inside `interp_finite` by dropping non-finite timestamps/values, returning all zeros if no finite values exist, returning a constant if only one finite value exists, and otherwise using `np.interp`, which also extends endpoint values beyond the observed range. Trials with empty extracted windows are skipped, and sessions with fewer than two retained trials are skipped. There is no explicit try/except around bad sessions and no explicit blink masking.

ii.
```python
def interp_finite(t_new, t, x):
    t=np.asarray(t,float); x=np.asarray(x,float)
    good=np.isfinite(t)&np.isfinite(x)
    if good.sum()==0: return np.zeros(len(t_new),dtype=np.float32)
    if good.sum()==1: return np.full(len(t_new),x[good][0],dtype=np.float32)
    return np.interp(t_new,t[good],x[good]).astype(np.float32)
...
if hi<=lo: continue
...
if len(sn)<2:
    print('  skipped: fewer than two retained trials',flush=True); del ds; gc.collect(); continue
```

iii. The trajectory does not give a detailed separate rationale for these fallback rules. Its explicit emphasis was on a robust sequential converter that could process all files while releasing memory between sessions.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive step is opening each large NWB file through AllenSDK and reading the full-session neural and behavioral arrays before trial slicing. The conversion is organized around one full-file load per experiment.

ii.
```python
for si,p in enumerate(paths):
    print(f'[{si+1}/{len(paths)}] {p.name}',flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ds=BehaviorOphysExperiment.from_nwb_path(str(p))
```

iii. In the trajectory, the AI explicitly called loading one AllenSDK object per NWB “the bottleneck” and then monitored the long full-dataset conversion as one file at a time.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The code keeps Python loops over trials and, inside each trial, over overlapping stimulus presentations. Those loops could be vectorized or pre-indexed, especially the repeated `stimulus_presentations` filtering and repeated `np.searchsorted` calls.

ii.
```python
for tid,row in trials.iterrows():
    ...
    sub=stim[(stim.start_time<=tt[-1]) & (stim.end_time>=tt[0])]
    for _,pr in sub.iterrows():
        ...
        a=int(np.searchsorted(tt,float(pr.start_time),side='left'))
        b=int(np.searchsorted(tt,float(pr.end_time),side='left'))
```

iii. The trajectory did not explicitly defend these loops beyond describing the converter as sequential and readable. It instead focused optimization effort on processing one file at a time and freeing memory.

## 9-c. What processing does the code repeat multiple times?

i. Within each session, the code repeatedly filters `stimulus_presentations` for every trial and repeatedly converts stimulus/trial times to indices with `np.searchsorted`. It also recomputes quintile thresholds independently for every recording.

ii.
```python
run_bin,run_edges=quintiles(run); pupil_bin,pupil_edges=quintiles(diameter)
...
for tid,row in trials.iterrows():
    ...
    sub=stim[(stim.start_time<=tt[-1]) & (stim.end_time>=tt[0])]
    for _,pr in sub.iterrows():
        ...
        a=int(np.searchsorted(tt,float(pr.start_time),side='left'))
        b=int(np.searchsorted(tt,float(pr.end_time),side='left'))
```

iii. The trajectory explicitly endorsed per-recording quintiles. It did not explicitly discuss the repeated per-trial filtering of stimulus presentations; that repetition is mainly evident from the code structure.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest unnecessary processing is converting `pupil_area` to equivalent circular diameter before quintiling, even though the AI itself noted this is a monotone transform and therefore preserves percentile-class assignments. The code also stores per-session quintile edges and other metadata that are not used by downstream decoding.

ii.
```python
area=interp_finite(ts,eye['timestamps'],eye['pupil_area'])
diameter=(2*np.sqrt(np.maximum(area,0)/np.pi)).astype(np.float32)
run_bin,run_edges=quintiles(run); pupil_bin,pupil_edges=quintiles(diameter)
...
session_info.append({'ophys_experiment_id':int(m['ophys_experiment_id']),
  ...
  'n_trials':len(sn), 'running_quintile_edges_cm_s':run_edges,
  'pupil_diameter_quintile_edges':pupil_edges})
```

iii. In the trajectory, the AI explicitly justified the diameter conversion by saying it is a monotone transform whose percentile classes match pupil-area percentile classes. That same justification implies the conversion is not needed for the categorical output actually saved.
