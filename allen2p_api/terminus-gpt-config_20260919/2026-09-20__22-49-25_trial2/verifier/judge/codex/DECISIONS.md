# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent constructs an AllenSDK local cache at `/app/data`, intersects the SDK experiment table with experiment IDs inferred from locally present NWB filenames, removes passive experiments and three experiments with missing eye data, and loads each retained experiment through `get_behavior_ophys_experiment`. Thus it converts the supplied local active subset, including both `VisualBehavior` and `VisualBehaviorMultiscope`, rather than all `VisualBehavior` project sessions listed by the SDK.

ii.
```python
cache=VisualBehaviorOphysProjectCache.from_local_cache(CACHE_DIR)
table=cache.get_ophys_experiment_table(); ids=table.index.intersection(local_ids())
tab=table.loc[ids].sort_index()
tab=tab[~tab.session_type.str.contains('passive',case=False,na=False)]
tab=tab[~tab.index.isin(MISSING_EYE_IDS)]
...
exp=cache.get_behavior_ophys_experiment(int(xid))
```

iii. The notes say the local cache contains only 284 of 1,936 listed experiments, so intersecting local IDs avoids downloading absent data. They justify excluding passive sessions because the paper did not analyze passive viewing and excluding the three eye-less experiments because pupil diameter is required.

## 1-b. How are the data split into subjects?

i. Subjects are sorted unique string-valued `mouse_id`s from the retained experiment table; each experiment gets the corresponding index.

ii.
```python
subjects=sorted(tab.mouse_id.astype(str).unique())
...
subject_idx.append(subjects.index(str(row.mouse_id)))
```

iii. The agent identifies `mouse_id` as the stable animal identifier and uses a global sorted list for deterministic indexing.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id` (one imaging plane) becomes one output “session.” Simultaneous planes sharing an `ophys_session_id` are not merged.

ii.
```python
for j,(xid,row) in enumerate(tab.iterrows(),1):
    exp=cache.get_behavior_ophys_experiment(int(xid))
    n,i,o,info,raw=convert_experiment(exp,row,IMAGE_TO_ID)
    neural.append(n)
```

iii. The notes argue that paper decoding was performed per imaging plane, that this keeps each population homogeneous in area, and that duplicated behavioral labels across simultaneous multiscope planes are legitimate.

## 1-d. How are the data split into trials?

i. The SDK `trials` table defines trials. Rows flagged `go` or `catch` are retained, and each is represented from its SDK `start_time` through `stop_time` using variable-length 100 ms bins.

ii.
```python
trials=exp.trials
trials=trials[trials.go.astype(bool)|trials.catch.astype(bool)].copy()
...
start=float(tr.start_time); stop=float(tr.stop_time)
n=max(1,int(np.ceil((stop-start)/BIN_S)))
edges=start+np.arange(n+1)*BIN_S; edges[-1]=stop
```

iii. The agent says `go | catch` directly implements the requested inclusion and inherently excludes aborted and auto-rewarded attempts; retaining full SDK trial boundaries preserves pre- and post-change content.

## 1-e. How are trials filtered based on quality controls?

i. Only go/catch trials are retained; an experiment is rejected if fewer than two remain. Passive experiments and three experiments without eye tracking are excluded before trial conversion, and malformed neural/outcome data raise errors.

ii.
```python
trials=trials[trials.go.astype(bool)|trials.catch.astype(bool)].copy()
if len(trials)<2: raise ValueError('fewer than two valid trials')
...
if sum(vals)!=1: raise ValueError(f'non-exclusive outcome flags {vals}')
```

iii. The notes cite the explicit go/catch instruction, the decoder’s two-trial minimum, the paper’s exclusion of passive viewing, and the need for a pupil output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the SDK’s inferred, filtered calcium events (`events.filtered_events`) and `ophys_timestamps`, not dF/F.

ii.
```python
ot=np.asarray(exp.ophys_timestamps,float)
ev=exp.events
traces=np.stack(ev.filtered_events.to_numpy()).astype(np.float32)
```

iii. The agent cites the paper’s statement that its analyses used detected calcium events and prefers SDK-filtered events to slow dF/F dynamics.

## 2-b. How is the `neural` data processed?

i. Per-frame filtered-event magnitudes are cumulatively summed once, then summed into consecutive 100 ms physical-time bins for each trial. No normalization is applied.

ii.
```python
event_cumsum=np.concatenate([np.zeros((traces.shape[0],1),np.float32),
                              np.cumsum(traces,dtype=np.float32,axis=1)],axis=1)
...
neu=aggregate_events(event_cumsum,ot,edges)
```

iii. The notes say summing preserves event magnitude, provides a common physical grid across 11 and 31 Hz rigs, and reduces memory while retaining resolution for 250 ms flashes.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent relies on AllenSDK’s valid-cell/ROI curation and filtered-event pipeline, then rejects length mismatches or nonfinite event arrays. It applies no additional cell selection.

ii.
```python
if traces.shape[1]!=len(ot): raise ValueError('event/timestamp length mismatch')
if not np.isfinite(traces).all(): raise ValueError('nonfinite filtered events')
```

iii. The notes state that `cell_specimen_table`/SDK experiment traces already exclude invalid ROIs such as duplicates, unions, ghosts, and non-cells, so additional filtering would duplicate SDK QC.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to SDK `start_time`; 100 ms bin edges are placed in absolute ophys time from `start_time` to `stop_time`, and ophys samples falling in each bin are summed.

ii.
```python
edges=start+np.arange(n+1)*BIN_S; edges[-1]=stop
centers=(edges[:-1]+edges[1:])/2
neu=aggregate_events(event_cumsum,ot,edges)
```

iii. The agent says the requested unit is a full trial rather than a fixed change-centered crop, so trial start is the alignment event and `off_end` remains variable.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is fixed at 100 ms. Native filtered events are summed into these bins, so explicit temporal rebinning is applied.

ii.
```python
BIN_S = 0.1
...
'time_bin_size':100.0
```

iii. The agent chose 100 ms to harmonize different native sampling rates, preserve 250 ms stimulus timing, and limit dataset size.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from active task `stimulus_presentations` fields `image_name`, `start_time`, and `end_time`; only 16 predefined real image names are retained.

ii.
```python
sp=task_stimuli(exp)
real=sp.image_name.isin(image_to_id)
presentations=sp[real][['start_time','end_time','image_name','is_change']]
```

iii. The agent says presentation records, unlike trial-level initial/change fields, accurately distinguish the 250 ms non-gray screen from the 500 ms gray interval and omissions.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Sixteen fixed image names map to IDs 1–16. Each bin starts as class 0 (gray/omission), and a real image ID is assigned when the bin center falls within that presentation’s interval.

ii.
```python
IMAGE_TO_ID = {x:i+1 for i,x in enumerate(IMAGE_NAMES)}
image=np.zeros(n,np.int64)
...
m=(centers>=float(pr.start_time))&(centers<float(pr.end_time))
image[m]=image_to_id[str(pr.image_name)]
```

iii. The notes interpret “image presented during the non-grey screen” literally and reserve class 0 for gray periods and omitted flashes.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is evaluated at the same 100 ms bin centers whose edges define the neural event sums.

ii.
```python
centers=(edges[:-1]+edges[1:])/2
neu=aggregate_events(event_cumsum,ot,edges)
...
m=(centers>=float(pr.start_time))&(centers<float(pr.end_time))
```

iii. A shared physical-time grid is intended to align stimulus labels and neural bins across acquisition rates.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It is derived from active task `stimulus_presentations.is_change` and `start_time`, restricted to recognized real-image presentations overlapping the trial.

ii.
```python
presentations=sp[real][['start_time','end_time','image_name','is_change']]
...
if bool(pr.is_change) and start<=float(pr.start_time)<stop:
```

iii. The agent regards `is_change` as the SDK’s direct change-event annotation.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector is created and the single 100 ms bin containing each real change onset is set to one.

ii.
```python
change=np.zeros(n,np.int64)
bi=min(n-1,max(0,int(np.floor((float(pr.start_time)-start)/BIN_S))))
change[bi]=1
```

iii. The notes describe image change as an onset pulse, not a persistent post-change state.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary: 0 means no change onset and 1 marks the onset bin; no numeric threshold is estimated.

ii.
```python
change=np.zeros(n,np.int64)
...
change[bi]=1
```

iii. The raw SDK boolean supplies the category directly.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change onset is converted to a bin index relative to trial start using the same 100 ms width used for neural aggregation.

ii.
```python
bi=int(np.floor((float(pr.start_time)-start)/BIN_S))
```

iii. The shared trial-relative physical-time grid is the stated alignment mechanism.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses `exp.running_speed.timestamps` and `exp.running_speed.speed`.

ii.
```python
run=exp.running_speed
rt=run.timestamps.to_numpy(float); rv=run.speed.to_numpy(float)
```

iii. The notes identify this as the SDK’s synchronized wheel-speed stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Finite samples are sorted, linearly interpolated to each 100 ms bin center with constant endpoint extrapolation, and then discretized using quintile edges calculated from the experiment’s entire native speed stream.

ii.
```python
run_edges=quintile_edges(rv)
run_cont=interp_valid(rt,rv,centers)
out[2]=discretize(run_cont,run_edges)
```

iii. The agent argues per-experiment quintiles yield balanced categories within recordings and avoid rig/calibration differences.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The 20th, 40th, 60th, and 80th percentiles define five integer classes 0–4.

ii.
```python
return np.quantile(v,[.2,.4,.6,.8])
...
return np.clip(np.searchsorted(edges,v,side='right'),0,4).astype(np.int64)
```

iii. Equal-percentile bins were required and are intended to approximately balance decoder classes.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is linearly interpolated to the centers of the same 100 ms bins used to aggregate neural events.

ii.
```python
centers=(edges[:-1]+edges[1:])/2
neu=aggregate_events(event_cumsum,ot,edges)
run_cont=interp_valid(rt,rv,centers)
```

iii. The agent relies on SDK-corrected synchronized timestamps and a common bin grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses `eye_tracking.pupil_area` and `eye_tracking.timestamps`, deriving a circular-equivalent diameter.

ii.
```python
eye=exp.eye_tracking
pupil=2.0*np.sqrt(eye.pupil_area.to_numpy(float)/np.pi)
pt=eye.timestamps.to_numpy(float)
```

iii. The notes cite the whitepaper’s processed pupil-area convention and treat its NaNs as blink/missing-frame exclusions.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Area is converted to diameter, finite samples are linearly interpolated to bin centers with endpoint extrapolation, and experiment-wide full-stream quintiles are applied.

ii.
```python
pupil_edges=quintile_edges(pupil)
pupil_cont=interp_valid(pt,pupil,centers)
out[3]=discretize(pupil_cont,pupil_edges)
```

iii. The agent says interpolation bridges blink NaNs while retaining all trial bins, and per-experiment quantiles normalize recording-to-recording scale differences.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Finite full-experiment diameters determine 20/40/60/80 percentile edges; values map to classes 0–4.

ii.
```python
pupil_edges=quintile_edges(pupil)
...
out[3]=discretize(pupil_cont,pupil_edges)
```

iii. This implements five equal-percentile categories within each experiment.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Diameter is interpolated by timestamp to the same 100 ms bin centers used for neural event aggregation.

ii.
```python
pupil_cont=interp_valid(pt,pupil,centers)
neu=aggregate_events(event_cumsum,ot,edges)
```

iii. The notes state that SDK timestamps are synchronized and explicit interpolation avoids assuming matching sample rates.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It comes from the trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
OUTCOMES = ['hit','miss','false_alarm','correct_reject']
vals=[bool(row[k]) for k in OUTCOMES]
```

iii. The agent identifies these as the four mutually exclusive valid outcomes for retained go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Exactly one flag must be true; its fixed list position gives code 0–3, repeated across all time bins of the trial.

ii.
```python
if sum(vals)!=1: raise ValueError(f'non-exclusive outcome flags {vals}')
return int(np.flatnonzero(vals)[0])
...
out[4]=outcome_value(tr)
```

iii. Repetition gives static semantics in the jointly time-varying `(5,T)` output representation accepted by the validator.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Nonfinite stream samples are removed before interpolation; interpolation requires at least two values and fills outside the observed interval with endpoint values. Three known eye-less experiments are excluded. Invalid event arrays, nonexclusive outcomes, too few trials, or absent eye data raise errors rather than being silently repaired; there is no per-experiment recovery in `main`.

ii.
```python
good=np.isfinite(t)&np.isfinite(v)
if good.sum()<2: raise ValueError('fewer than two valid samples')
return np.interp(q,t,v,left=v[0],right=v[-1])
...
tab=tab[~tab.index.isin(MISSING_EYE_IDS)]
```

iii. The notes favor explicit exclusions and assertions, and say blink-related NaNs can be bridged using neighboring valid pupil samples.

## 9-a. What are the most time-consuming steps of the code?

i. Loading each large SDK experiment is I/O-heavy; conversion also computes full-trace cumulative sums and repeatedly constructs hundreds of trial arrays. The recorded run prints per-experiment timings.

ii.
```python
exp=cache.get_behavior_ophys_experiment(int(xid))
...
event_cumsum=np.concatenate([... np.cumsum(traces,dtype=np.float32,axis=1)],axis=1)
```

iii. The agent’s notes identify SDK loading/decoding and event processing as the dominant work, with cumulative sums used to avoid a much worse per-trial scan.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The outer experiment loop is necessarily load-oriented, but the per-trial loop and especially the nested loop over overlapping stimulus presentations could be vectorized or replaced by bulk interval/bin indexing. Subject/region list lookups could also use dictionaries.

ii.
```python
for ti,(_,tr) in enumerate(trials.iterrows()):
    ...
    for _,pr in cand.iterrows():
```

iii. The implementation prioritizes clear interval logic; its main explicit optimization is vectorized cumulative-sum aggregation of neural events.

## 9-c. What processing does the code repeat multiple times?

i. For every trial it filters the presentation table for overlaps, interpolates the same full running and pupil streams, and allocates output arrays. It also performs repeated linear `list.index` lookups for subject and region indices.

ii.
```python
cand=presentations[(presentations.end_time>start)&(presentations.start_time<stop)]
run_cont=interp_valid(rt,rv,centers)
pupil_cont=interp_valid(pt,pupil,centers)
...
subject_idx.append(subjects.index(str(row.mouse_id)))
```

iii. The agent explicitly avoided repeating neural integration by computing one cumulative sum per experiment, but did not similarly precompute behavioral interpolation or presentation-to-bin mappings.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `convert_experiment` always saves first-trial continuous signals and neural slices in `raw_plot`, even when `--show-processing` is false; those values are discarded after return. It also computes duration summaries and extensive metadata not consumed by decoder training, and calls garbage collection after every experiment.

ii.
```python
if raw_plot is None:
    raw_plot=(centers,neu[:min(8,len(neu))],image,change,run_cont,pupil_cont,out[2],out[3])
...
del exp,n,i,o,raw; gc.collect()
```

iii. The plotting payload and metadata support validation/documentation, but are unnecessary for the downstream decoder; the unconditional plotting payload is a small avoidable cost.
