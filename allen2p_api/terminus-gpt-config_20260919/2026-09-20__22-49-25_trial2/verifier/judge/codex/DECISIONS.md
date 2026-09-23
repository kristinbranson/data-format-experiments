# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the Allen cache with `VisualBehaviorOphysProjectCache.from_local_cache`, discovers locally available experiments by parsing the NWB filenames under the cache directory, intersects those IDs with the experiment table, removes passive sessions and three hard-coded missing-eye experiments, and then loads each retained experiment with `get_behavior_ophys_experiment()`. It therefore operates on locally present experiments, not on the full `VisualBehavior` project table.

ii.
```python
def local_ids():
    root=CACHE_DIR/'visual-behavior-ophys-1.1.0'/'behavior_ophys_experiments'
    return sorted(int(re.search(r'_(\d+)\.nwb$',p.name).group(1))
                  for p in root.glob('behavior_ophys_experiment_*.nwb'))

cache=VisualBehaviorOphysProjectCache.from_local_cache(CACHE_DIR)
table=cache.get_ophys_experiment_table()
ids=table.index.intersection(local_ids())
tab=table.loc[ids].sort_index()
tab=tab[~tab.session_type.str.contains('passive',case=False,na=False)]
tab=tab[~tab.index.isin(MISSING_EYE_IDS)]

for j,(xid,row) in enumerate(tab.iterrows(),1):
    exp=cache.get_behavior_ophys_experiment(int(xid))
```

iii. In `CONVERSION_NOTES.md` Step 1 and Step 5, and in the trajectory, the AI explicitly justified using AllenSDK only, avoiding direct NWB access, restricting work to the provided local subset, excluding passive sessions because trial outcome should reflect real behavior, and excluding the three experiments without eye data because pupil diameter was a required output.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the sorted unique `mouse_id` values from the filtered experiment table, stored as strings. Each experiment/session later stores a `subject_idx` pointing back into that subject list.

ii.
```python
subjects=sorted(tab.mouse_id.astype(str).unique())
...
subject_idx.append(subjects.index(str(row.mouse_id)))
```

iii. The notes repeatedly treat `mouse_id` as the canonical animal identifier from AllenSDK metadata. The trajectory describes subject handling as a simple metadata lookup rather than an additional curation step.

## 1-c. How are the data split into sessions?

i. The AI treats each `ophys_experiment_id` imaging plane as one decoder session. It does not group multiple planes sharing the same `ophys_session_id`; instead, every retained row of the experiment table becomes one session in the output.

ii.
```python
tab=table.loc[ids].sort_index()
...
for j,(xid,row) in enumerate(tab.iterrows(),1):
    exp=cache.get_behavior_ophys_experiment(int(xid))
    n,i,o,info,raw=convert_experiment(exp,row,IMAGE_TO_ID)
    neural.append(n); inputs.append(i); outputs.append(o)
```

iii. In Step 4 and Step 5 of the notes, and in trajectory step 24, the AI states that multiscope planes share trial timing but contain different cell populations, and that the paper’s decoders were reported per imaging plane. On that basis it deliberately chose `ophys_experiment_id` rather than `ophys_session_id` as the session unit.

## 1-d. How are the data split into trials?

i. Trials are taken from `exp.trials`, filtered to `go | catch`, and segmented from SDK `start_time` to `stop_time`. Each trial gets a variable-length sequence of 100 ms bins spanning that full trial interval.

ii.
```python
trials=exp.trials
trials=trials[trials.go.astype(bool)|trials.catch.astype(bool)].copy()
...
for ti,(_,tr) in enumerate(trials.iterrows()):
    start=float(tr.start_time); stop=float(tr.stop_time)
    n=max(1,int(np.ceil((stop-start)/BIN_S)))
    edges=start+np.arange(n+1)*BIN_S; edges[-1]=stop
    centers=(edges[:-1]+edges[1:])/2
```

iii. The Step 5 notes say to “retain SDK go/catch trial boundaries” and to use native `start_time` to `stop_time` windows rather than event-centered cropping. The trajectory also records that the AI wanted variable-length trials while keeping a common physical bin size.

## 1-e. How are trials filtered based on quality controls?

i. Trial rows are kept only if they are `go` or `catch`, which implicitly removes aborted and auto-rewarded trials in this dataset. Experiments are also excluded wholesale if they are passive sessions, if they are in the hard-coded missing-eye set, or if they have fewer than two retained trials.

ii.
```python
tab=tab[~tab.session_type.str.contains('passive',case=False,na=False)]
tab=tab[~tab.index.isin(MISSING_EYE_IDS)]
...
trials=trials[trials.go.astype(bool)|trials.catch.astype(bool)].copy()
if len(trials)<2: raise ValueError('fewer than two valid trials')
```

iii. The notes justify passive-session exclusion because passive replay does not provide a meaningful behavioral outcome target, and they justify excluding the three missing-eye experiments because pupil diameter is mandatory. The AI also states in the notes that AllenSDK’s go/catch flags already separate valid go/catch trials from aborted and auto-rewarded rows.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the SDK’s inferred event representation, specifically `exp.events.filtered_events`, together with `exp.ophys_timestamps`.

ii.
```python
ot=np.asarray(exp.ophys_timestamps,float)
ev=exp.events
traces=np.stack(ev.filtered_events.to_numpy()).astype(np.float32)
```

iii. In Step 3 and Step 5, the AI argues that the paper says neural analyses used detected calcium events rather than dF/F. The trajectory shows this was a deliberate departure from the dF/F route because the AI viewed `filtered_events` as the paper-faithful neural representation.

## 2-b. How is the `neural` data processed?

i. The AI converts event traces into per-trial neural matrices by summing event magnitudes into consecutive 100 ms bins between each trial’s `start_time` and `stop_time`. It precomputes a cumulative sum per experiment and uses `searchsorted` to aggregate bins efficiently.

ii.
```python
event_cumsum=np.concatenate([np.zeros((traces.shape[0],1),np.float32),
                              np.cumsum(traces,dtype=np.float32,axis=1)],axis=1)
...
def aggregate_events(event_cumsum, timestamps, edges):
    lo=np.searchsorted(timestamps,edges[:-1],side='left')
    hi=np.searchsorted(timestamps,edges[1:],side='left')
    return (event_cumsum[:,hi]-event_cumsum[:,lo]).astype(np.float32)
...
neu=aggregate_events(event_cumsum,ot,edges)
```

iii. The Step 5 notes say the purpose of 100 ms bins was to harmonize 10.7 Hz and 30.9 Hz rigs, preserve event magnitude, and control memory usage. The trajectory also records that the AI first implemented this inefficiently, then optimized by moving the cumulative-sum construction outside the trial loop.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no additional neuron-level inclusion filter beyond the AllenSDK experiment object. The code asserts that `filtered_events` are finite and implicitly relies on AllenSDK’s curated valid ROIs.

ii.
```python
traces=np.stack(ev.filtered_events.to_numpy()).astype(np.float32)
if traces.shape[1]!=len(ot): raise ValueError('event/timestamp length mismatch')
if not np.isfinite(traces).all(): raise ValueError('nonfinite filtered events')
```

iii. Step 1 and Step 3 of the notes say AllenSDK already exposes only valid ROIs and that no extra electrophysiology-style cell QC is needed for this two-photon dataset. The AI therefore treated Allen’s QC as sufficient.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start, not to change time. The bins are built on the ophys clock from each trial’s `start_time` to `stop_time`, so alignment is trial-based on absolute ophys time.

ii.
```python
start=float(tr.start_time); stop=float(tr.stop_time)
n=max(1,int(np.ceil((stop-start)/BIN_S)))
edges=start+np.arange(n+1)*BIN_S; edges[-1]=stop
centers=(edges[:-1]+edges[1:])/2
neu=aggregate_events(event_cumsum,ot,edges)
```

iii. The notes explicitly say “No alignment-event cropping” and describe trial start as the alignment event in metadata. The trajectory also frames this as preserving the full trial, including pre-change and post-change periods.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output uses fixed 100 ms bins (`BIN_S = 0.1`). Yes, the neural data are rebinned from the native frame times into those physical-time bins.

ii.
```python
BIN_S = 0.1
...
n=max(1,int(np.ceil((stop-start)/BIN_S)))
edges=start+np.arange(n+1)*BIN_S; edges[-1]=stop
...
'time_bin_size':100.0,
'neural_signal':'AllenSDK filtered_events summed in 100 ms bins',
```

iii. The Step 5 notes justify 100 ms bins as a common temporal resolution across mixed frame-rate rigs, and the trajectory repeats that this was chosen specifically to support both multiscope and single-plane experiments in one decoder-ready format.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity comes from `stimulus_presentations`, specifically `image_name`, `start_time`, and `end_time` after restricting to the active task block.

ii.
```python
def task_stimuli(exp):
    sp=exp.stimulus_presentations
    return sp[sp.stimulus_block_name.eq('change_detection_behavior')].copy()

sp=task_stimuli(exp)
real=sp.image_name.isin(image_to_id)
presentations=sp[real][['start_time','end_time','image_name','is_change']]
```

iii. The notes say the cache contains multiple stimulus blocks and that only `change_detection_behavior` rows should be used for the task outputs. The AI also justified using stimulus intervals directly because the target variable is the image currently on screen.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI uses a fixed global mapping for 16 named images (`im000` ... `im106`), reserves `0` for gray or omitted periods, and fills each 100 ms bin according to whether its center falls inside an overlapping stimulus interval.

ii.
```python
IMAGE_NAMES = ['im000','im031','im035','im045','im054','im061','im062','im063',
               'im065','im066','im069','im073','im075','im077','im085','im106']
IMAGE_TO_ID = {x:i+1 for i,x in enumerate(IMAGE_NAMES)}
...
image=np.zeros(n,np.int64)
for _,pr in cand.iterrows():
    m=(centers>=float(pr.start_time))&(centers<float(pr.end_time))
    image[m]=image_to_id[str(pr.image_name)]
```

iii. In Step 5, the AI justifies a dedicated gray class because the task asks for the image shown during the non-gray screen, so gray and omitted flashes should not be treated as image identities. The notes also say there are 16 real image identities across the supplied image sets.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned on the same 100 ms trial bins as the neural data. For each trial, the code looks at overlapping stimulus intervals and labels bins by testing their centers against those intervals.

ii.
```python
centers=(edges[:-1]+edges[1:])/2
neu=aggregate_events(event_cumsum,ot,edges)
...
m=(centers>=float(pr.start_time))&(centers<float(pr.end_time))
image[m]=image_to_id[str(pr.image_name)]
```

iii. The notes explicitly say interval membership was chosen instead of nearest-neighbor assignment so stimulus onsets would not be smeared across bin edges. The trajectory presents this as a direct temporal-alignment choice.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the task stimulus table, using `is_change` and the stimulus `start_time` of each overlapping presentation.

ii.
```python
presentations=sp[real][['start_time','end_time','image_name','is_change']]
...
if bool(pr.is_change) and start<=float(pr.start_time)<stop:
    bi=min(n-1,max(0,int(np.floor((float(pr.start_time)-start)/BIN_S))))
    change[bi]=1
```

iii. The notes describe image change as a stimulus event, not a trial-table label, and the trajectory says the AI wanted the change variable to come from the actual presentation intervals rather than from a coarser trial summary.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI creates a binary pulse train with a single positive 100 ms bin at each true change onset within a retained trial. It does not mark the rest of the post-change flash or post-change trial as `1`.

ii.
```python
change=np.zeros(n,np.int64)
...
if bool(pr.is_change) and start<=float(pr.start_time)<stop:
    bi=min(n-1,max(0,int(np.floor((float(pr.start_time)-start)/BIN_S))))
    change[bi]=1
```

iii. In Step 5 and Step 12, the AI defends this as matching the instruction that image change should be `1` “right after” a change, and argues that widening the pulse would distort timing. The trajectory contrasts this with the paper’s different, more balanced decoding setup.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is not thresholded from a continuous value; it is directly encoded as two categories: `0` for no change and `1` for change.

ii.
```python
change=np.zeros(n,np.int64)
...
change[bi]=1
...
'output_values':[['gray']+IMAGE_NAMES,['no_change','change'],
```

iii. The justification is implicit in both the task instructions and the notes: image change is a binary event variable, so the AI represented it as a binary categorical output rather than applying an additional thresholding rule.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is aligned to the same 100 ms trial bins as neural data by converting each change onset time into a bin index relative to trial start.

ii.
```python
centers=(edges[:-1]+edges[1:])/2
neu=aggregate_events(event_cumsum,ot,edges)
...
bi=min(n-1,max(0,int(np.floor((float(pr.start_time)-start)/BIN_S))))
change[bi]=1
```

iii. The notes say all streams should share a common physical-time grid based on ophys time. The trajectory also emphasizes that the bin containing the exact onset is the aligned event location.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is taken from `exp.running_speed`, using its `timestamps` and `speed` columns.

ii.
```python
run=exp.running_speed
rt=run.timestamps.to_numpy(float)
rv=run.speed.to_numpy(float)
```

iii. The notes identify `BehaviorSession.running_speed` as the AllenSDK source for locomotion. No extra source table is used.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes quintile edges from the full experiment-level running trace, linearly interpolates running speed to the per-trial 100 ms bin centers, and then discretizes each bin with those experiment-specific edges.

ii.
```python
run_edges=quintile_edges(rv)
...
run_cont=interp_valid(rt,rv,centers)
...
out[2]=discretize(run_cont,run_edges)
```

iii. In Step 5, the AI says experiment-wise quintiles avoid cross-rig calibration differences and keep the five classes roughly balanced within each recording. The trajectory repeats that this was a deliberate decoder-specific adaptation, not a direct paper method.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into five percentile bins using the 20th, 40th, 60th, and 80th percentiles of the experiment-level running-speed distribution. The resulting labels are clipped to integers `0` through `4`.

ii.
```python
def quintile_edges(v):
    v=np.asarray(v,float); v=v[np.isfinite(v)]
    return np.quantile(v,[.2,.4,.6,.8])

def discretize(v, edges):
    return np.clip(np.searchsorted(edges,v,side='right'),0,4).astype(np.int64)
```

iii. The Step 5 notes explicitly call for “five percentile bins” and say repeated quantile edges are handled monotonically with clipping. This is the AI’s category-thresholding rule for running.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned to neural data by interpolating from running timestamps to the same 100 ms bin centers used for the neural matrix in each trial.

ii.
```python
centers=(edges[:-1]+edges[1:])/2
neu=aggregate_events(event_cumsum,ot,edges)
...
run_cont=interp_valid(rt,rv,centers)
```

iii. The notes say all streams must be sampled on a common physical-time grid tied to ophys timestamps. The trajectory repeatedly describes interpolation to bin centers as the alignment mechanism.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `exp.eye_tracking`, specifically `pupil_area` and `timestamps`. The code converts area to diameter with `2 * sqrt(area / pi)`.

ii.
```python
eye=exp.eye_tracking
if len(eye)==0: raise ValueError('missing eye tracking')
pupil=2.0*np.sqrt(eye.pupil_area.to_numpy(float)/np.pi)
pt=eye.timestamps.to_numpy(float)
```

iii. In Step 3 and Step 5, the AI cites the whitepaper’s processed pupil-area definition and says diameter can be recovered from it. The trajectory also notes that processed `pupil_area` already marks invalid blink/outlier frames as non-finite.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI converts pupil area to diameter, keeps only finite samples through `interp_valid`, computes experiment-level quintile edges, interpolates to the 100 ms trial bin centers, and discretizes those interpolated values.

ii.
```python
def interp_valid(t, v, q):
    good=np.isfinite(t)&np.isfinite(v)
    ...
    return np.interp(q,t,v,left=v[0],right=v[-1])

pupil=2.0*np.sqrt(eye.pupil_area.to_numpy(float)/np.pi)
pupil_edges=quintile_edges(pupil)
...
pupil_cont=interp_valid(pt,pupil,centers)
out[3]=discretize(pupil_cont,pupil_edges)
```

iii. The notes justify using processed pupil area rather than raw eye geometry, and they justify excluding entire experiments with no eye data instead of imputing a required output stream. The AI’s trajectory also treats finite-sample interpolation as the way to bridge blink-induced gaps.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into five experiment-wise percentile bins using the same quintile machinery as running speed, again producing labels `0` through `4`.

ii.
```python
pupil_edges=quintile_edges(pupil)
...
out[3]=discretize(pupil_cont,pupil_edges)
...
'output_values':[
    ['Q1_smallest','Q2','Q3','Q4','Q5_largest'],
```

iii. The Step 5 notes say pupil should be represented with five percentile bins and that per-experiment binning avoids calibration differences across recordings. The AI therefore reuses the same quantile-thresholding rule used for running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned to neural data by interpolation to the same 100 ms trial bin centers used for the neural and running outputs.

ii.
```python
centers=(edges[:-1]+edges[1:])/2
...
pupil_cont=interp_valid(pt,pupil,centers)
```

iii. The notes explicitly say all streams share the same physical-time bin grid. The trajectory also describes pupil alignment identically to running alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome comes from the boolean trial columns `hit`, `miss`, `false_alarm`, and `correct_reject` in `exp.trials`.

ii.
```python
OUTCOMES = ['hit','miss','false_alarm','correct_reject']

def outcome_value(row):
    vals=[bool(row[k]) for k in OUTCOMES]
    if sum(vals)!=1: raise ValueError(f'non-exclusive outcome flags {vals}')
    return int(np.flatnonzero(vals)[0])
```

iii. The notes identify those four AllenSDK flags as the valid behavioral outcomes for retained go/catch trials. The trajectory also stresses that exactly one should be true for each retained trial.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome flags are converted to integer class IDs `0` to `3`, and that single class is repeated across all time bins within the trial so the output remains a uniform `(5, T)` matrix.

ii.
```python
def outcome_value(row):
    ...
    return int(np.flatnonzero(vals)[0])
...
out=np.empty((5,n),np.int64)
...
out[4]=outcome_value(tr)
```

iii. The notes say the semantics are “static per trial,” but the trajectory explains that the code repeats the value over time because the validator/model accepted a uniform time-varying output matrix more easily than a mixed static/time-varying representation.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly handles problems by exclusion or by interpolation over valid samples. It hard-codes three experiments to exclude for missing eye tracking, raises errors if a stream has fewer than two valid samples or if event traces are non-finite, uses endpoint-filled interpolation for running and pupil, and keeps sparse all-zero neural trials rather than discarding them.

ii.
```python
MISSING_EYE_IDS = {795953296, 833631914, 806456687}
...
good=np.isfinite(t)&np.isfinite(v)
if good.sum()<2: raise ValueError('fewer than two valid samples')
return np.interp(q,t,v,left=v[0],right=v[-1])
...
if len(eye)==0: raise ValueError('missing eye tracking')
if not np.isfinite(traces).all(): raise ValueError('nonfinite filtered events')
```

iii. The notes justify excluding whole experiments with no pupil stream because that output is required and should not be fabricated. Step 10 of the notes also explicitly argues that all-zero neural trials are valid sparse-event intervals, not missing data, so they are retained.

## 9-a. What are the most time-consuming steps of the code?

i. The AI identifies AllenSDK experiment loading as the dominant runtime cost, and it also notes that an early implementation mistake recomputed the event cumulative sum once per trial until that bottleneck was removed.

ii.
```python
for j,(xid,row) in enumerate(tab.iterrows(),1):
    ts=time.time()
    exp=cache.get_behavior_ophys_experiment(int(xid))
    n,i,o,info,raw=convert_experiment(exp,row,IMAGE_TO_ID)
```

iii. Step 6 of the notes says AllenSDK object creation dominates runtime. The trajectory then documents the separate optimization step where per-trial cumulative-sum recomputation was identified as a severe avoidable bottleneck and fixed.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Two obvious Python-level loops remain: the outer per-trial loop and the inner loop over overlapping stimulus presentations within each trial. Both could be vectorized further, although the AI already vectorized event aggregation itself.

ii.
```python
for ti,(_,tr) in enumerate(trials.iterrows()):
    ...
    cand=presentations[(presentations.end_time>start)&(presentations.start_time<stop)]
    for _,pr in cand.iterrows():
        m=(centers>=float(pr.start_time))&(centers<float(pr.end_time))
        image[m]=image_to_id[str(pr.image_name)]
```

iii. The notes’ efficiency plan says to use `searchsorted` and cumulative sums where possible, but it also acknowledges that only overlapping stimulus rows are still iterated in Python. The trajectory treats those loops as acceptable after the bigger event-aggregation bottleneck was removed.

## 9-c. What processing does the code repeat multiple times?

i. The code repeatedly filters the stimulus presentation table per trial and repeatedly interpolates running and pupil values per trial. Those operations are done again for every trial even though they all come from session- or experiment-level streams.

ii.
```python
for ti,(_,tr) in enumerate(trials.iterrows()):
    ...
    cand=presentations[(presentations.end_time>start)&(presentations.start_time<stop)]
    ...
    run_cont=interp_valid(rt,rv,centers)
    pupil_cont=interp_valid(pt,pupil,centers)
```

iii. The AI’s notes emphasize that experiments are loaded only once, but they do not claim all downstream computations are precomputed. The repeated per-trial interpolation and overlap filtering are visible in the final code and reflect a readability-over-vectorization tradeoff.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code always constructs `raw_plot` data for the first trial of every experiment even when `--show-processing` is not requested, and it also computes `durations` only to store summary metadata that are not used by the downstream decoder. Those computations are retained for diagnostics rather than for the converted dataset’s core training path.

ii.
```python
neural=[]; inputs=[]; outputs=[]; durations=[]
raw_plot=None
...
neural.append(neu); inputs.append(inp); outputs.append(out); durations.append(stop-start)
if raw_plot is None:
    raw_plot=(centers,neu[:min(8,len(neu))],image,change,run_cont,pupil_cont,out[2],out[3])
...
'trial_duration_min_s':float(np.min(durations)),
'trial_duration_max_s':float(np.max(durations)),
```

iii. The notes describe these diagnostics as useful for validation and plotting, not as part of the model input/output format itself. In the final code, `raw_plot` is discarded unless optional plotting is enabled, so that work is not necessary for the default conversion output.
