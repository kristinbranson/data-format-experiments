# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data via `VisualBehaviorOphysProjectCache.from_local_cache()`, discovers locally available experiment IDs by scanning NWB filenames in the cache directory, and intersects them with the experiment table. It does NOT filter by `project_code == 'VisualBehavior'` — it includes all locally available experiments (both VisualBehavior and VisualBehaviorMultiscope). It then excludes passive sessions and 3 experiments with missing eye tracking data. Each experiment is loaded via `cache.get_behavior_ophys_experiment(int(xid))`.

ii.
```python
cache=VisualBehaviorOphysProjectCache.from_local_cache(CACHE_DIR)
table=cache.get_ophys_experiment_table(); ids=table.index.intersection(local_ids())
tab=table.loc[ids].sort_index()
tab=tab[~tab.session_type.str.contains('passive',case=False,na=False)]
tab=tab[~tab.index.isin(MISSING_EYE_IDS)]
```

```python
def local_ids():
    root=CACHE_DIR/'visual-behavior-ophys-1.1.0'/'behavior_ophys_experiments'
    return sorted(int(re.search(r'_(\d+)\.nwb$',p.name).group(1))
                  for p in root.glob('behavior_ophys_experiment_*.nwb'))
```

iii. The AI justified this by noting that the local cache contains a subset of the full release and that passive sessions should be excluded because passive replayed changes lack meaningful behavioral outcomes, consistent with the paper's exclusion of passive data.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified as unique `mouse_id` values from the filtered experiment table, sorted and converted to strings.

ii.
```python
subjects=sorted(tab.mouse_id.astype(str).unique())
```

iii. The AI uses mouse_id from the experiment metadata, consistent with the SDK's subject identification.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id` (i.e., each imaging plane) is treated as a separate decoder session. Multi-plane sessions from VisualBehaviorMultiscope contribute one decoder session per plane, sharing the same behavioral data but different neural populations.

ii.
```python
for j,(xid,row) in enumerate(tab.iterrows(),1):
    ts=time.time(); exp=cache.get_behavior_ophys_experiment(int(xid))
    n,i,o,info,raw=convert_experiment(exp,row,IMAGE_TO_ID)
    neural.append(n); inputs.append(i); outputs.append(o)
```

iii. The AI justified this by noting that the paper's decoding analyses are performed per imaging plane, and that each plane has a distinct neural population. Multiscope sessions legitimately share behavioral labels across planes.

## 1-d. How are the data split into trials?

i. Trials are defined using the SDK's trials table. Only `go` and `catch` trials are retained (which inherently excludes aborted and auto-rewarded trials). Each trial spans `start_time` to `stop_time` with variable-length duration, rebinned into 100ms time bins.

ii.
```python
trials=exp.trials
trials=trials[trials.go.astype(bool)|trials.catch.astype(bool)].copy()
if len(trials)<2: raise ValueError('fewer than two valid trials')
```

```python
for ti,(_,tr) in enumerate(trials.iterrows()):
    start=float(tr.start_time); stop=float(tr.stop_time)
    n=max(1,int(np.ceil((stop-start)/BIN_S)))
    edges=start+np.arange(n+1)*BIN_S; edges[-1]=stop
```

iii. The AI uses the SDK's go/catch flags, which are mutually exclusive with aborted and auto-rewarded. The variable-length trial window captures both pre-change stimulus flashes and the post-change response window.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) only go|catch trials retained, (2) sessions with <2 valid trials raise an error and are skipped, (3) passive sessions are excluded at the experiment level, (4) 3 experiments with missing eye tracking are excluded. The trial outcome flags are validated as mutually exclusive.

ii.
```python
trials=trials[trials.go.astype(bool)|trials.catch.astype(bool)].copy()
if len(trials)<2: raise ValueError('fewer than two valid trials')
```

```python
tab=tab[~tab.session_type.str.contains('passive',case=False,na=False)]
tab=tab[~tab.index.isin(MISSING_EYE_IDS)]
```

```python
def outcome_value(row):
    vals=[bool(row[k]) for k in OUTCOMES]
    if sum(vals)!=1: raise ValueError(f'non-exclusive outcome flags {vals}')
```

iii. The go|catch filter matches the instructions to include Go and Catch trials while excluding Aborted and Auto-rewarded. Passive sessions are excluded because their replayed changes lack meaningful behavioral outcomes. Missing eye tracking sessions are excluded because pupil diameter is a required output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `exp.events.filtered_events` — the SDK's inferred and filtered calcium event magnitudes, not dF/F traces.

ii.
```python
ev=exp.events
traces=np.stack(ev.filtered_events.to_numpy()).astype(np.float32)
```

iii. The AI justified this by citing the paper: "For all analysis of neural data we used the detected calcium events." The filtered events remove slow GCaMP decay dynamics and are recommended by the SDK for neural analysis. This matches the paper's methodology.

## 2-b. How is the `neural` data processed?

i. Filtered event magnitudes are summed within 100ms time bins using a cumulative-sum-based aggregation. Bin edges are computed from trial start to stop in 100ms increments, with the final bin edge clipped to the trial stop time.

ii.
```python
event_cumsum=np.concatenate([np.zeros((traces.shape[0],1),np.float32),
                              np.cumsum(traces,dtype=np.float32,axis=1)],axis=1)
```

```python
def aggregate_events(event_cumsum, timestamps, edges):
    lo=np.searchsorted(timestamps,edges[:-1],side='left')
    hi=np.searchsorted(timestamps,edges[1:],side='left')
    return (event_cumsum[:,hi]-event_cumsum[:,lo]).astype(np.float32)
```

iii. The cumulative sum approach is efficient (O(n) per experiment rather than O(n*T) per trial). Summing event magnitudes in bins preserves total detected activity per bin. The 100ms bin size was chosen to provide a common temporal grid across different frame rates (31 Hz single-plane and 11 Hz multiscope).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied beyond the SDK's built-in curation. The SDK's `cell_specimen_table` and `events` already expose only valid ROIs. The code asserts that all values are finite.

ii.
```python
if traces.shape[1]!=len(ot): raise ValueError('event/timestamp length mismatch')
if not np.isfinite(traces).all(): raise ValueError('nonfinite filtered events')
```

iii. The SDK pipeline already applies quality control (cell segmentation, ROI validation, neuropil correction). No further filtering was deemed necessary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's neural data is aligned to the trial's `start_time` from the SDK trials table. The 100ms bin edges are computed as `start + k*0.1` for k=0,...,n, with the final edge clipped to `stop_time`. Neural events are aggregated within these bins using the ophys timestamps.

ii.
```python
start=float(tr.start_time); stop=float(tr.stop_time)
n=max(1,int(np.ceil((stop-start)/BIN_S)))
edges=start+np.arange(n+1)*BIN_S; edges[-1]=stop
centers=(edges[:-1]+edges[1:])/2
neu=aggregate_events(event_cumsum,ot,edges)
```

iii. Trial start serves as the temporal anchor. Using bin edges aligned to absolute time and aggregating events by timestamp ensures correct alignment regardless of frame rate.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is rebinned to 100ms time bins. This is a fixed physical-time bin applied uniformly across all experiments, regardless of native frame rate.

ii.
```python
BIN_S = 0.1
```

```python
'time_bin_size':100.0
```

iii. The AI justified 100ms bins as necessary because the dataset includes both single-plane (~31 Hz, ~32ms frames) and multiscope (~11 Hz, ~93ms frames) recordings. A common physical time bin ensures consistent temporal resolution across all sessions, as required by the format specification ("Time bins should be the same size for all trials and sessions").

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, specifically the `image_name`, `start_time`, and `end_time` columns. Only presentations from the `change_detection_behavior` block with recognized image names are used.

ii.
```python
def task_stimuli(exp):
    sp=exp.stimulus_presentations
    return sp[sp.stimulus_block_name.eq('change_detection_behavior')].copy()
```

```python
sp=task_stimuli(exp)
real=sp.image_name.isin(image_to_id)
presentations=sp[real][['start_time','end_time','image_name','is_change']]
```

iii. Using stimulus_presentations provides the exact timing of each image presentation, rather than relying on the trial table's summary fields. This enables precise assignment of image identity to each time bin.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image identity is encoded as integers 0-16: 0 for gray/no-image periods, 1-16 for 16 hardcoded natural image names. For each time bin, the image identity is assigned based on whether the bin center falls within a stimulus presentation interval. Bins not covered by any recognized image presentation are assigned 0 (gray).

ii.
```python
IMAGE_NAMES = ['im000','im031','im035','im045','im054','im061','im062','im063',
               'im065','im066','im069','im073','im075','im077','im085','im106']
IMAGE_TO_ID = {x:i+1 for i,x in enumerate(IMAGE_NAMES)}
```

```python
image=np.zeros(n,np.int64); change=np.zeros(n,np.int64)
cand=presentations[(presentations.end_time>start)&(presentations.start_time<stop)]
for _,pr in cand.iterrows():
    m=(centers>=float(pr.start_time))&(centers<float(pr.end_time))
    image[m]=image_to_id[str(pr.image_name)]
```

iii. Using interval membership rather than simple switching at change_time ensures accurate image identity even during grey inter-stimulus intervals. The gray class (0) captures periods when no image is on screen.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned by assigning values based on whether each 100ms bin center falls within a stimulus presentation interval. The same bin edges are used for neural data and image identity.

ii.
```python
centers=(edges[:-1]+edges[1:])/2
cand=presentations[(presentations.end_time>start)&(presentations.start_time<stop)]
for _,pr in cand.iterrows():
    m=(centers>=float(pr.start_time))&(centers<float(pr.end_time))
    image[m]=image_to_id[str(pr.image_name)]
```

iii. Using the same bin centers for all variables ensures temporal alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` flag and `start_time` in the `stimulus_presentations` table (filtered to the change_detection_behavior block).

ii.
```python
if bool(pr.is_change) and start<=float(pr.start_time)<stop:
    bi=min(n-1,max(0,int(np.floor((float(pr.start_time)-start)/BIN_S))))
    change[bi]=1
```

iii. The `is_change` flag in stimulus_presentations marks actual identity changes (go trials). Catch trials (sham changes) have `is_change=False`.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Image change is a binary variable. A single 100ms bin is set to 1 at the time bin containing the change onset. All other bins are 0. This applies only to go trials (where `is_change=True`).

ii.
```python
change=np.zeros(n,np.int64)
if bool(pr.is_change) and start<=float(pr.start_time)<stop:
    bi=min(n-1,max(0,int(np.floor((float(pr.start_time)-start)/BIN_S))))
    change[bi]=1
```

iii. The single-bin pulse captures the moment of image change. This is a strict interpretation of "right after a change in image identity."

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1). No thresholding is applied — 0 for no change, 1 for change. Only a single bin per trial is marked as 1 (at change onset for go trials).

ii. See 4-b above.

iii. Binary categorization directly matches the instruction specification.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The change bin index is computed from the change onset time relative to trial start, using `floor((change_time - start) / 0.1)`. This uses the same 100ms bin grid as neural data.

ii.
```python
bi=min(n-1,max(0,int(np.floor((float(pr.start_time)-start)/BIN_S))))
change[bi]=1
```

iii. Same bin grid as neural and other outputs ensures alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `exp.running_speed`, using the `timestamps` and `speed` columns.

ii.
```python
run=exp.running_speed
rt=run.timestamps.to_numpy(float); rv=run.speed.to_numpy(float)
```

iii. The SDK's `running_speed` is the standard interface for locomotion data from the running wheel encoder.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to 100ms bin centers using `np.interp`. Out-of-range values are filled with edge values (not NaN). It is then discretized into 5 quintile bins using per-experiment percentile edges.

ii.
```python
def interp_valid(t, v, q):
    t=np.asarray(t,float); v=np.asarray(v,float)
    good=np.isfinite(t)&np.isfinite(v)
    t=t[good]; v=v[order];
    return np.interp(q,t,v,left=v[0],right=v[-1])
```

```python
run_edges=quintile_edges(rv)
run_cont=interp_valid(rt,rv,centers)
out[2]=discretize(run_cont,run_edges)
```

```python
def quintile_edges(v):
    v=np.asarray(v,float); v=v[np.isfinite(v)]
    return np.quantile(v,[.2,.4,.6,.8])

def discretize(v, edges):
    return np.clip(np.searchsorted(edges,v,side='right'),0,4).astype(np.int64)
```

iii. The AI computes quintile edges per experiment (not globally), ensuring equal bin occupancy within each recording. Linear interpolation preserves the signal shape while resampling to the common time grid.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins (0-4) using per-experiment quintile edges at the 20th, 40th, 60th, and 80th percentiles. `np.searchsorted` with `side='right'` assigns values to bins, clipped to [0,4].

ii.
```python
run_edges=quintile_edges(rv)
```

```python
def quintile_edges(v):
    return np.quantile(v,[.2,.4,.6,.8])

def discretize(v, edges):
    return np.clip(np.searchsorted(edges,v,side='right'),0,4).astype(np.int64)
```

iii. Per-experiment quintiles ensure balanced class distributions within each session, which is beneficial for decoding.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the 100ms bin centers, which are the same centers used for neural data aggregation. This ensures temporal alignment.

ii.
```python
run_cont=interp_valid(rt,rv,centers)
```

iii. Same bin centers guarantee alignment with neural and other output variables.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `exp.eye_tracking.pupil_area`, converted to diameter via `2*sqrt(pupil_area/pi)`.

ii.
```python
eye=exp.eye_tracking
if len(eye)==0: raise ValueError('missing eye tracking')
pupil=2.0*np.sqrt(eye.pupil_area.to_numpy(float)/np.pi)
pt=eye.timestamps.to_numpy(float)
```

iii. The AI followed the whitepaper's definition: the processed pupil area treats the observed ellipse major axis as the diameter of an underlying circular pupil. `2*sqrt(area/pi)` recovers this diameter from the area.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter is computed from pupil_area, then linearly interpolated to 100ms bin centers using `interp_valid` (which filters out NaN/non-finite values, including blink frames where pupil_area is NaN). It is then discretized into 5 per-experiment quintile bins.

ii.
```python
pupil=2.0*np.sqrt(eye.pupil_area.to_numpy(float)/np.pi)
pt=eye.timestamps.to_numpy(float)
pupil_edges=quintile_edges(pupil)
pupil_cont=interp_valid(pt,pupil,centers)
out[3]=discretize(pupil_cont,pupil_edges)
```

iii. The `interp_valid` function naturally handles blinks because `pupil_area` is NaN during blinks, so `np.isfinite` in `interp_valid` filters them out before interpolation. Per-experiment quintile edges ensure balanced bins within each recording.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: discretized into 5 bins (0-4) using per-experiment quintile edges.

ii.
```python
pupil_edges=quintile_edges(pupil)
out[3]=discretize(pupil_cont,pupil_edges)
```

iii. Same approach as running speed for consistency.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed: interpolated to 100ms bin centers shared with neural data.

ii.
```python
pupil_cont=interp_valid(pt,pupil,centers)
```

iii. Same bin centers guarantee alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
OUTCOMES = ['hit','miss','false_alarm','correct_reject']

def outcome_value(row):
    vals=[bool(row[k]) for k in OUTCOMES]
    if sum(vals)!=1: raise ValueError(f'non-exclusive outcome flags {vals}')
    return int(np.flatnonzero(vals)[0])
```

iii. These four flags are the SDK's canonical trial outcome labels for the change detection task. They are mutually exclusive for non-aborted, non-auto-rewarded trials. The validation ensures exactly one flag is True.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is mapped to integer codes (hit=0, miss=1, false_alarm=2, correct_reject=3) and repeated across all time bins in the trial as a static per-trial value in the joint `(5, T)` output matrix.

ii.
```python
out[4]=outcome_value(tr)
```

iii. The outcome code is constant across the trial, matching the instruction's "Static per-trial" specification. Repeating across time bins ensures a uniform `(5, T)` output shape.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: 3 experiments (795953296, 833631914, 806456687) are excluded entirely because pupil diameter is a required output.
- **Passive sessions**: Excluded because replayed changes lack behavioral outcomes.
- **Few trials**: Sessions with <2 valid trials raise an error and are skipped.
- **Non-exclusive outcomes**: Validated with assertion; non-exclusive flags raise an error.
- **Non-finite values**: Validated with assertions on neural data.
- **Out-of-range interpolation**: `interp_valid` fills with edge values rather than NaN.

ii.
```python
MISSING_EYE_IDS = {795953296, 833631914, 806456687}
tab=tab[~tab.index.isin(MISSING_EYE_IDS)]
```

```python
if len(eye)==0: raise ValueError('missing eye tracking')
if not np.isfinite(traces).all(): raise ValueError('nonfinite filtered events')
```

iii. The AI chose to exclude rather than impute missing eye tracking data, arguing that whole-session imputation is unjustified. Other edge cases are handled with assertions that fail fast.

## 9-a. What are the most time-consuming steps of the code?

i. Loading each experiment via `cache.get_behavior_ophys_experiment()` is the dominant bottleneck, as it reads large NWB files from disk. The AI measured ~4-10 seconds per experiment, with total runtime ~18.5 minutes for 199 experiments.

ii. N/A

iii. The I/O cost of reading NWB files through the SDK is inherent and cannot be significantly reduced.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_experiment` iterates over each trial sequentially. Within each trial, the stimulus presentation assignment loop iterates over candidate presentations. The cumulative-sum-based neural aggregation is already vectorized across neurons.

ii.
```python
for ti,(_,tr) in enumerate(trials.iterrows()):
    ...
    cand=presentations[(presentations.end_time>start)&(presentations.start_time<stop)]
    for _,pr in cand.iterrows():
        ...
```

iii. The per-trial loop is not a bottleneck compared to data loading. The inner presentation loop operates on a small number of candidate presentations per trial.

## 9-c. What processing does the code repeat multiple times?

i. The initial version recomputed the experiment-wide event cumulative sum per trial, which was identified as a severe bottleneck and fixed. In the final version, the cumsum is computed once per experiment and reused across all trials. No other redundant processing was identified.

ii.
```python
event_cumsum=np.concatenate([np.zeros((traces.shape[0],1),np.float32),
                              np.cumsum(traces,dtype=np.float32,axis=1)],axis=1)
```

iii. The AI documented this optimization in the conversion notes, noting it reduced runtime from projected >60 minutes to 18.5 minutes.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores raw continuous running and pupil values in the `raw_plot` tuple for processing visualization, but this is only used when `--show-processing` is enabled and only for the first trial/first 2 experiments. The `info` dict per experiment stores detailed metadata that goes into `session_info` in the final pickle but may not be used by the decoder.

ii.
```python
if raw_plot is None:
    raw_plot=(centers,neu[:min(8,len(neu))],image,change,run_cont,pupil_cont,out[2],out[3])
```

iii. The raw_plot storage is minimal (only one trial per experiment) and conditional. The session_info metadata is useful for debugging and provenance.
