# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent creates an AllenSDK `VisualBehaviorOphysProjectCache` at `/app/data`, intersects the SDK experiment table with locally present experiment IDs, retains `active_behavior` experiments, and loads each with `get_behavior_ophys_experiment`. Full mode uses eight spawned workers.

ii. ```python
cache=get_cache(); tab=selected_table(cache)
exp=cache.get_behavior_ophys_experiment(int(eid))
```

iii. The notes justify this as using the required SDK exclusively, avoiding absent-cache downloads, and parallelizing the estimated 20+ minute serial workload.

## 1-b. How are the data split into subjects?

i. Subjects are sorted unique string-valued `mouse_id`s from the selected experiment table. Experiments point to them through `subject_idx`; subjects left with no retained experiments are removed and indices remapped.

ii. ```python
subjects=sorted(tab.mouse_id.astype(str).unique())
subjmap={x:i for i,x in enumerate(subjects)}
data['subject_idx'].append(subjmap[str(row.mouse_id)])
```

iii. The notes regard SDK mouse IDs as the canonical animal identifiers and verify 38 retained mice.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id` (one plane) becomes one output session. Although `ophys_session_id` is recorded in metadata, simultaneous experiments are not grouped.

ii. ```python
tasks=[(int(eid),row.to_dict()) for eid,row in tab.iterrows()]
data['neural'].append(res['neural'])
```

iii. The notes explicitly describe 199 retained “plane sessions” and experiment-level parallelism, treating an experiment as the decoding session.

## 1-d. How are the data split into trials?

i. The SDK trials table defines trials. Each retained trial spans `start_time` to `stop_time`; complete, non-overlapping 100 ms bins are constructed, so trial lengths vary and the trailing partial bin is discarded.

ii. ```python
n = int(np.floor((float(row.stop_time)-float(row.start_time))/BIN_S + 1e-9))
edges = float(row.start_time) + np.arange(n+1)*BIN_S
centers = edges[:-1] + BIN_S/2
```

iii. The notes justify native trial boundaries, half-open bins, and variable lengths as preserving SDK trial semantics while producing a common resolution.

## 1-e. How are trials filtered based on quality controls?

i. Only go or catch trials are retained; aborted and auto-rewarded trials are excluded. Trials must have exactly one of four outcomes and at least one complete bin. Experiments need at least two eligible trials.

ii. ```python
mask = (tr['go'].astype(bool) | tr['catch'].astype(bool))
mask &= ~tr['aborted'].astype(bool) & ~tr['auto_rewarded'].astype(bool)
exact = tr[OUTCOME_COLS].astype(int).sum(axis=1).eq(1)
```

iii. The agent cites the task’s explicit go/catch inclusion and aborted/auto-reward exclusion, and uses exclusive outcomes to prevent ambiguous labels.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `exp.events['events']`, ordered by `exp.cell_specimen_table.index`, plus `exp.ophys_timestamps` for temporal binning.

ii. ```python
evdf=exp.events
cell_ids=exp.cell_specimen_table.index.to_numpy()
events=np.stack(evdf.loc[cell_ids]['events'].to_numpy()).astype(np.float32)
```

iii. The notes say detected calcium-event magnitudes match the paper and are preferable to dF/F for its analyses.

## 2-b. How is the `neural` data processed?

i. Per-cell detected-event magnitudes are summed in half-open 100 ms bins. A float64 cumulative sum is computed once per experiment and differenced for every trial, then saved as float32.

ii. ```python
np.cumsum(event_matrix, axis=1, dtype=np.float64, out=cs[:,1:])
return (event_cumsum[:,hi] - event_cumsum[:,lo]).astype(np.float32, copy=False)
```

iii. The notes justify summation as magnitude-preserving aggregation and float64 accumulation as a fix for late-recording cancellation errors.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It uses SDK-valid cells from `cell_specimen_table`, requires exact event/timestamp dimensions and finite binned values, but does not remove silent trials or cells.

ii. ```python
if events.shape != (len(cell_ids),len(ts)):
    return None, 'event shape mismatch ...'
assert np.isfinite(n).all()
```

iii. The agent argues that all-zero event trials are genuine sparsity and that activity-dependent removal would bias trial selection.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial-relative bins begin exactly at SDK `start_time`; event timestamps are assigned to those bins with `searchsorted`. Metadata calls trial start the alignment event.

ii. ```python
edges = float(row.start_time) + np.arange(n+1)*BIN_S
lo = np.searchsorted(event_times, edges[:-1], side='left')
```

iii. The notes say synchronized ophys coordinates and half-open bins avoid duplication at boundaries.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output resolution is fixed at 100 ms. Native event samples are explicitly rebinned by summing event magnitudes.

ii. ```python
BIN_S = 0.100
'time_bin_size':100.0
```

iii. The agent chose a common 100 ms decoder grid, describing it as compatible with the paper’s response-window analyses.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from change-detection rows of `exp.stimulus_presentations`, specifically `image_name`, `omitted`, `start_time`, and `end_time`.

ii. ```python
sp=exp.stimulus_presentations
sp=sp[sp['stimulus_block_name'].astype(str).str.contains('change_detection',na=False)]
```

iii. The notes state that presentation intervals best represent the actual image/non-image epochs.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Seventeen fixed categories are used: gray plus 16 images. Bins default to gray; non-omitted known-image presentation intervals overwrite them with an integer image ID.

ii. ```python
image = np.zeros(len(centers), dtype=np.int64)
if not omitted and str(name) in IMAGE_TO_ID:
    image[a:b] = IMAGE_TO_ID[str(name)]
```

iii. The agent justifies retaining gray and omission intervals because the requested identity applies to actual displayed content.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Presentation boundaries are mapped onto the same 100 ms trial-bin centers used for neural bins.

ii. ```python
a=np.searchsorted(centers, float(r.start_time), side='left')
b=np.searchsorted(centers, float(r.end_time), side='left')
```

iii. The notes report independent raw-trial comparisons and synchronized SDK timestamps.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It uses `stimulus_presentations.is_change` and each real change presentation’s `start_time`.

ii. ```python
if bool(r.is_change):
    a=np.searchsorted(centers, float(r.start_time), side='left')
```

iii. The agent says this excludes catch sham changes and labels only real changes.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary array initialized to zero is set to one for 400 ms following each real change onset.

ii. ```python
b=np.searchsorted(centers, float(r.start_time)+0.400, side='left')
change[a:b]=1
```

iii. After a one-bin version decoded poorly, the agent chose 400 ms based on the paper’s first-400-ms change decoder window and documented improved accuracy.

## 4-c. How is `output` *Image change* thresholded into categories?

i. There is no numeric threshold: `is_change=True` produces category 1 during the post-onset window; all other bins are category 0.

ii. ```python
change = np.zeros(len(centers), dtype=np.int64)
if bool(r.is_change): change[a:b]=1
```

iii. The notes define output values as `no_change` and `change`, with catch trials remaining zero.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change onset and offset are located against the same 100 ms bin centers as every other output and the corresponding neural bins.

ii. ```python
a=np.searchsorted(centers, float(r.start_time), side='left')
b=np.searchsorted(centers, float(r.start_time)+0.400, side='left')
```

iii. The notes cite synchronized SDK time coordinates and raw spot checks.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from `exp.running_speed['timestamps']` and `['speed']`.

ii. ```python
run=exp.running_speed
all_run=interp_valid(run['timestamps'],run['speed'],all_centers)
```

iii. The agent treats the SDK’s processed running-speed stream as canonical.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Finite, unique samples are linearly interpolated at all eligible bin centers. Per-experiment 20/40/60/80 percentiles define five bins; missing interpolates are median-filled before categorization.

ii. ```python
out = np.interp(query, t, v, left=np.nan, right=np.nan)
run_edges=quintile_edges(all_run)
rb,mr=labels_from_edges(rv,run_edges,run_fill)
```

iii. The notes say session-specific quintiles achieve essentially exact class balance and median filling avoids inventing extreme behavior.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize` applies four per-experiment percentile edges to yield integer categories 0–4 (`Q1`–`Q5`).

ii. ```python
np.nanpercentile(v, [20,40,60,80])
np.digitize(v, edges, right=False).astype(np.int64)
```

iii. The requested five equal percentile bins motivate quintiles; the agent validates near-20% fractions in each.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is interpolated directly at the centers of the same trial bins used for neural event sums.

ii. ```python
all_centers=np.concatenate([g[2] for g in grids])
all_run=interp_valid(...,all_centers)
```

iii. The notes rely on hardware-synchronized SDK timestamps and independent reconstruction checks.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses `exp.eye_tracking` timestamps, `pupil_width`, and `pupil_height`.

ii. ```python
eye=exp.eye_tracking
pupil=2.0*np.maximum(eye['pupil_width'].to_numpy(float),eye['pupil_height'].to_numpy(float))
```

iii. The notes call this a processed pupil-axis diameter and report excluding experiments without a usable stream.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Diameter is defined as twice the larger pupil axis, interpolated at trial-bin centers, and converted into per-experiment quintiles. Non-finite interpolates are median-filled. The code does not explicitly use `likely_blink`.

ii. ```python
all_pupil=interp_valid(eye['timestamps'],pupil,all_centers)
pupil_edges=quintile_edges(all_pupil)
pb,mp=labels_from_edges(pv,pupil_edges,pupil_fill)
```

iii. The agent asserts SDK-filtered pupil axes are appropriate and reports no imputed bins in retained full-data experiments.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four per-experiment percentile edges (20, 40, 60, 80) are applied with `np.digitize`, yielding categories 0–4.

ii. ```python
pupil_edges=quintile_edges(all_pupil)
np.digitize(v, edges, right=False).astype(np.int64)
```

iii. The five requested equal percentile categories motivate this and are checked for balance.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Eye data are interpolated at the same 100 ms trial-bin centers corresponding to neural event bins.

ii. ```python
all_pupil=interp_valid(eye['timestamps'],pupil,all_centers)
pv=all_pupil[pos:pos+T]
```

iii. The agent cites synchronized timestamps and three independent trial reconstructions.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome comes from the trials-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. ```python
OUTCOME_COLS = ['hit','miss','false_alarm','correct_reject']
outcome=int(np.flatnonzero(rowt[OUTCOME_COLS].to_numpy(bool))[0])
```

iii. The notes use the SDK’s mutually exclusive canonical outcome flags.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trials without exactly one outcome are dropped. The true column’s fixed index (0–3) is repeated across every time bin of the trial.

ii. ```python
exact = tr[OUTCOME_COLS].astype(int).sum(axis=1).eq(1)
np.full(T,outcome,dtype=np.int64)
```

iii. The agent requires an unambiguous static label and repeats it to satisfy the common `(5,T)` output shape.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Non-finite interpolation sources are ignored; duplicate timestamps are collapsed. Missing queried behavior is median-filled before binning. Experiments are skipped for insufficient running/pupil data, event-shape mismatch, or fewer than two trials. Skips are recorded; subjects are remapped. Finite values and valid ranges are asserted.

ii. ```python
good = np.isfinite(times) & np.isfinite(values)
v[missing] = fill
if all_pupil is None: return None,'insufficient processed pupil data'
```

iii. The notes favor skipping wholly unusable pupil streams, median-imputing only isolated gaps, and recording provenance rather than fabricating an entire signal.

## 9-a. What are the most time-consuming steps of the code?

i. AllenSDK experiment loading and processing full event/behavior arrays dominate. The notes measured roughly 5–8 seconds per experiment serially and used eight workers for the full run.

ii. ```python
pool=mp.get_context('spawn').Pool(nworkers,initializer=init_worker)
result_iter=pool.imap(worker_process,tasks,chunksize=1)
```

iii. The agent’s timing estimates motivated safe experiment-level parallelism; full conversion took about 245 seconds.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. `make_grids`, the per-trial assembly loop, and the per-presentation loop in `image_and_change` remain Python loops. Trial event binning is already vectorized within each trial, and interpolation/percentiles are performed over concatenated centers.

ii. ```python
for idx,row in trials.iterrows():
for tidx,edges,centers in grids:
for r in sp.itertuples():
```

iii. The notes emphasize that replacing per-bin work and repeated cumulative sums produced the important speedup; remaining variable-length interval loops are simpler to retain.

## 9-c. What processing does the code repeat multiple times?

i. Each worker initializes its own cache; each experiment independently filters presentations, constructs grids, interpolates streams, calculates quintiles, and loops through presentations once per trial via `image_and_change`. Thus the entire presentation table is rescanned for every trial.

ii. ```python
for tidx,edges,centers in grids:
    im,ch=image_and_change(sp,centers,edges)
```

iii. The notes mainly discuss eliminating the earlier repeated full event cumulative sum; they do not acknowledge the repeated presentation scans.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Up to three trials’ `plot_info` (centers, neural arrays, outputs, and continuous behavior) are retained for every experiment even when plotting is disabled; in full mode these objects are transferred from workers and then discarded. The `row` argument to `process_experiment` is unused. Continuous behavior is otherwise needed to form categories.

ii. ```python
if len(plot_info)<3: plot_info.append((centers,n,out,rv,pv))
result=dict(...,plot_info=plot_info,...)
```

iii. The notes justify plot data for visual validation but do not explain retaining it during non-plotting full conversion.
