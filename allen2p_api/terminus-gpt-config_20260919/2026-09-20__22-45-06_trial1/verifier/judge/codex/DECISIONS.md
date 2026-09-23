# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data through AllenSDK `VisualBehaviorOphysProjectCache`, but it first restricts the experiment table to NWB files that are already present locally under `/app/data`. It then keeps only rows with `behavior_type == 'active_behavior'` and processes each retained experiment by calling `get_behavior_ophys_experiment()`.

ii. ```python
def get_cache():
    from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache
    return VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=str(CACHE_DIR))

def local_experiment_ids():
    paths = CACHE_DIR.glob('visual-behavior-ophys-*/behavior_ophys_experiments/behavior_ophys_experiment_*.nwb')
    return sorted(int(re.search(r'(\d+)\.nwb$', p.name).group(1)) for p in paths)

def selected_table(cache):
    tab = cache.get_ophys_experiment_table()
    ids = tab.index.intersection(local_experiment_ids())
    tab = tab.loc[ids]
    tab = tab[tab['behavior_type'].eq('active_behavior')].sort_index()
    return tab
...
cache=get_cache(); tab=selected_table(cache)
...
res, err = process_experiment(_WORKER_CACHE, eid, row, show=False)
```

iii. In `CONVERSION_NOTES.md` Step 4-5, the agent says the full paper cohort is not locally cached, so it will use "all 202 locally cached active task experiments" and exclude passive sessions. Trajectory steps 20-23 repeat that rationale and frame it as using all supplied active task data while staying within AllenSDK-only loading.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values from the selected experiment table, converted to strings and stored in sorted order. Each retained experiment/session gets a `subject_idx` derived from that `mouse_id`.

ii. ```python
subjects=sorted(tab.mouse_id.astype(str).unique())
subjmap={x:i for i,x in enumerate(subjects)}
...
data['subject_idx'].append(subjmap[str(row.mouse_id)])
```

iii. The notes' Step 5 mapping explicitly says `mouse_id` becomes `subjects` and `subject_idx`, with one entry per retained plane experiment. There is no more elaborate subject grouping beyond Allen metadata.

## 1-c. How are the data split into sessions?

i. The AI treats each `ophys_experiment_id` as one session. It does not merge experiments that share an `ophys_session_id`; instead, each imaging plane is a separate target session.

ii. ```python
tasks=[(int(eid),row.to_dict()) for eid,row in tab.iterrows()]
...
for k,(eid,rowdict,res,err) in enumerate(result_iter,1):
    ...
    data['neural'].append(res['neural'])
    data['input'].append(res['input'])
    data['output'].append(res['output'])
    ...
    info={'ophys_experiment_id':int(eid),'ophys_session_id':int(row.ophys_session_id), ...}
```

iii. Step 4 and Step 5 in `CONVERSION_NOTES.md` say "Session unit: one ophys experiment/imaging plane per target session" because the agent wanted plane-wise analysis and to avoid merging planes with different timestamp grids. Trajectory steps 20-23 repeat that decision.

## 1-d. How are the data split into trials?

i. Trials come from `exp.trials`. After filtering, each trial is represented by a fixed-width 100 ms grid spanning `[start_time, stop_time)`. Trials with fewer than one 100 ms bin are dropped.

ii. ```python
def prepare_trials(exp):
    tr = exp.trials.copy()
    ...
    return tr

def make_grids(trials):
    grids=[]
    for idx,row in trials.iterrows():
        n = int(np.floor((float(row.stop_time)-float(row.start_time))/BIN_S + 1e-9))
        if n < 1: continue
        edges = float(row.start_time) + np.arange(n+1)*BIN_S
        centers = edges[:-1] + BIN_S/2
        grids.append((idx, edges, centers))
    return grids
```

iii. The notes say trial boundaries come from SDK `trials.start_time` and `stop_time`, and that the agent intentionally kept native trial windows instead of cropping around change time. Trajectory step 23 adds that a common bin size was imposed afterward.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only go or catch trials, excludes aborted and auto-rewarded trials, and drops any trial that does not have exactly one of `hit`, `miss`, `false_alarm`, or `correct_reject`. Trials shorter than one 100 ms bin are omitted implicitly in `make_grids`, and experiments with fewer than two eligible trials are skipped entirely.

ii. ```python
mask = (tr['go'].astype(bool) | tr['catch'].astype(bool))
mask &= ~tr['aborted'].astype(bool) & ~tr['auto_rewarded'].astype(bool)
tr = tr.loc[mask].copy()
exact = tr[OUTCOME_COLS].astype(int).sum(axis=1).eq(1)
if (~exact).any():
    print(f'  warning: dropping {(~exact).sum()} trials without exactly one outcome')
    tr = tr.loc[exact]
...
if len(grids)<2: return None, f'fewer than 2 eligible trials ({len(grids)})'
```

iii. Step 5 in the notes states the trial filter as `(go OR catch) AND NOT aborted AND NOT auto_rewarded`, plus exactly one recognized outcome and at least two retained trials per experiment. The exact-outcome filter is the agent's extra consistency check.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural activity from `exp.events['events']`, indexed to the valid cells in `exp.cell_specimen_table` and synchronized with `exp.ophys_timestamps`.

ii. ```python
ts=np.asarray(exp.ophys_timestamps,float)
evdf=exp.events
cell_ids=exp.cell_specimen_table.index.to_numpy()
evdf=evdf.loc[cell_ids]
events=np.stack(evdf['events'].to_numpy()).astype(np.float32)
```

iii. In Step 4 of `CONVERSION_NOTES.md`, the agent explicitly reverses an earlier dF/F plan and says the paper used detected/regressed calcium events, so `events` are a closer match than `dff_traces`. Trajectory steps 19-20 document that change.

## 2-b. How is the `neural` data processed?

i. For each experiment, the event matrix is converted into a cumulative sum once, then event magnitudes are summed inside 100 ms trial bins. The saved trial matrices are `float32` arrays of shape `(cells, time_bins)`.

ii. ```python
def cumulative_events(event_matrix):
    cs = np.empty((event_matrix.shape[0], event_matrix.shape[1]+1), dtype=np.float64)
    cs[:,0] = 0.0
    np.cumsum(event_matrix, axis=1, dtype=np.float64, out=cs[:,1:])
    return cs

def bin_events(event_times, event_cumsum, edges):
    lo = np.searchsorted(event_times, edges[:-1], side='left')
    hi = np.searchsorted(event_times, edges[1:], side='left')
    return (event_cumsum[:,hi] - event_cumsum[:,lo]).astype(np.float32, copy=False)
...
n=bin_events(ts,event_cs,edges)
```

iii. Step 5 and Step 6 say the agent wanted a common 100 ms grid and chose to sum detected event magnitudes within bins. Trajectory steps 23-24 describe this as a compromise that preserves multiscope compatibility and avoids per-trial recomputation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no extra neuron-level filtering beyond taking the SDK's valid `cell_specimen_table` ordering and the corresponding `events` rows. The code checks only that the event matrix shape matches the timestamp length.

ii. ```python
cell_ids=exp.cell_specimen_table.index.to_numpy()
evdf=evdf.loc[cell_ids]
events=np.stack(evdf['events'].to_numpy()).astype(np.float32)
if events.shape != (len(cell_ids),len(ts)):
    return None, f'event shape mismatch {events.shape} vs {len(cell_ids),len(ts)}'
```

iii. The notes say to rely on AllenSDK valid segmented cells and not add extra ROI/cell filtering. This is justified there as following the released SDK curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start. Each trial's bins begin at `row.start_time`, proceed in 100 ms steps to `row.stop_time`, and event sums are computed over those bin edges using synchronized ophys timestamps.

ii. ```python
n = int(np.floor((float(row.stop_time)-float(row.start_time))/BIN_S + 1e-9))
edges = float(row.start_time) + np.arange(n+1)*BIN_S
centers = edges[:-1] + BIN_S/2
...
n=bin_events(ts,event_cs,edges)
```

iii. The notes' metadata section says the temporal alignment event is native SDK trial start. Trajectory step 23 states that the agent repeated the static outcome row because all outputs had to share the same per-trial time axis.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 100 ms bins (`BIN_S = 0.100`), and yes, the original AllenSDK streams are rebinned or interpolated onto that common grid.

ii. ```python
BIN_S = 0.100
...
'metadata':{'task_description': ...,
            'time_bin_size':100.0,
            'temporal_alignment_event':'Native SDK trial start; 100 ms bins defined in synchronized ophys timestamp coordinates.',
            ...}
```

iii. Step 5 says the agent imposed a common 100 ms grid because it mixed recordings with different native frame rates. Trajectory step 23 explicitly cites the requirement that time bins be the same across sessions.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity comes from `exp.stimulus_presentations`, specifically `image_name`, `start_time`, `end_time`, and `omitted`, after restricting to stimulus blocks whose name contains `change_detection`.

ii. ```python
sp=exp.stimulus_presentations
sp=sp[sp['stimulus_block_name'].astype(str).str.contains('change_detection',na=False)].copy()
...
name = r.image_name
omitted = bool(r.omitted) if pd.notna(r.omitted) else False
if not omitted and pd.notna(name) and str(name) in IMAGE_TO_ID:
    a=np.searchsorted(centers, float(r.start_time), side='left')
    b=np.searchsorted(centers, float(r.end_time), side='left')
    image[a:b] = IMAGE_TO_ID[str(name)]
```

iii. The notes say the stimulus table had multiple irrelevant blocks and that only the `change_detection` block should drive image labels. Step 5 also says omissions and inter-image gray periods should map to gray/no-image.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI uses a fixed global 17-class vocabulary: `gray` plus 16 natural images. Every 100 ms bin is initialized to `gray`, then bins overlapping non-omitted stimulus intervals are overwritten with the corresponding image ID.

ii. ```python
IMAGE_VALUES = ['gray','im000','im031','im035','im045','im054','im061','im062',
                'im063','im065','im066','im069','im073','im075','im077','im085','im106']
IMAGE_TO_ID = {v:i for i,v in enumerate(IMAGE_VALUES)}
...
image = np.zeros(len(centers), dtype=np.int64)
...
image[a:b] = IMAGE_TO_ID[str(name)]
```

iii. Step 5 of the notes states that active sessions use image sets A and B, giving 16 identities, and that omitted flashes and inter-flash gray periods should not become a separate image identity. The agent therefore added an explicit `gray` class.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is generated on the same per-trial 100 ms grid as the neural data. The code uses the same `centers` array for both, and each stimulus presentation interval is mapped into those trial bins.

ii. ```python
for tidx,edges,centers in grids:
    ...
    n=bin_events(ts,event_cs,edges)
    im,ch=image_and_change(sp,centers,edges)
    ...
    out=np.vstack([im,ch,rb,pb,np.full(T,outcome,dtype=np.int64)])
```

iii. The notes describe all streams as being mapped onto the same ophys-derived trial grid. The agent's rationale is that this guarantees output rows and neural bins share a single time axis.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` flag and `start_time` in the filtered `stimulus_presentations` table, not from the trials table's `change_time` column.

ii. ```python
if bool(r.is_change):
    a=np.searchsorted(centers, float(r.start_time), side='left')
    b=np.searchsorted(centers, float(r.start_time)+0.400, side='left')
    change[a:b]=1
```

iii. The notes say the stimulus table is the relevant source for flashed image events once the code was switched to interval-based image labeling. The agent wanted real stimulus change onsets rather than inferred trial-level timing.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code initializes a zero vector and marks bins as 1 for the first 400 ms after each real image-change onset. Catch sham changes remain zero because they are not flagged by `is_change`.

ii. ```python
change = np.zeros(len(centers), dtype=np.int64)
...
if bool(r.is_change):
    a=np.searchsorted(centers, float(r.start_time), side='left')
    b=np.searchsorted(centers, float(r.start_time)+0.400, side='left')
    change[a:b]=1
```

iii. Trajectory step 27 records that the agent originally used a one-bin label, found poor sample decoder performance, and changed to a 400 ms post-change window because the paper's decoder also used the first 400 ms after image onset. Step 5 in the notes was updated to describe that as "right after change."

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is a binary variable with categories 0 = `no_change` and 1 = `change`.

ii. ```python
'output_values':[IMAGE_VALUES,['no_change','change'],['Q1','Q2','Q3','Q4','Q5'],['Q1','Q2','Q3','Q4','Q5'],OUTCOME_COLS]
...
change = np.zeros(len(centers), dtype=np.int64)
...
change[a:b]=1
```

iii. The agent treated image change as a simple binary decoding target throughout the notes and code. No multiclass thresholding was introduced.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is aligned on the same 100 ms trial grid used for neural event bins. Change bins are found by searching the trial-bin centers for each change onset.

ii. ```python
for tidx,edges,centers in grids:
    n=bin_events(ts,event_cs,edges)
    im,ch=image_and_change(sp,centers,edges)
    ...
```

iii. The notes say all streams are synchronized on the common ophys-based trial grid, and the code uses the same `centers` array for both neural and image-change rows.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `exp.running_speed`, specifically its `timestamps` and `speed` columns.

ii. ```python
run=exp.running_speed
...
all_run=interp_valid(run['timestamps'],run['speed'],all_centers)
```

iii. The notes explicitly say to use the SDK-processed `running_speed` stream rather than raw wheel signals, because that matches the AllenSDK/whitepaper processing.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The code linearly interpolates running speed onto all trial-bin centers in the experiment, computes experiment-specific quintile edges from those values, and then discretizes each trial's running trace on that basis. Missing samples are filled with the experiment median before digitization.

ii. ```python
all_run=interp_valid(run['timestamps'],run['speed'],all_centers)
...
run_edges=quintile_edges(all_run)
run_fill=float(np.nanmedian(all_run))
...
rb,mr=labels_from_edges(rv,run_edges,run_fill)
```

iii. Step 5 says running should be aligned at bin centers and binned by 20/40/60/80 percentiles within each experiment/session. The agent justified session-wise discretization as preventing between-mouse calibration differences from dominating.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five categories using the 20th, 40th, 60th, and 80th percentiles computed from that experiment's eligible binned samples. `np.digitize` produces integer labels 0-4, corresponding to `Q1`-`Q5`.

ii. ```python
def quintile_edges(values):
    ...
    return np.nanpercentile(v, [20,40,60,80])

def labels_from_edges(values, edges, fill):
    ...
    return np.digitize(v, edges, right=False).astype(np.int64), int(missing.sum())
```

iii. The notes describe these as session-level equal-frequency bins. Missing values are not given a special category; they are median-filled before binning and the number of filled samples is tracked.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned to the same 100 ms trial-bin centers used for neural activity, then sliced per trial in lockstep with the neural bins.

ii. ```python
all_centers=np.concatenate([g[2] for g in grids])
all_run=interp_valid(run['timestamps'],run['speed'],all_centers)
...
rv=all_run[pos:pos+T]
...
out=np.vstack([im,ch,rb,pb,np.full(T,outcome,dtype=np.int64)])
```

iii. The notes consistently describe running as being resampled onto the common ophys-derived bin centers so it can share a time axis with neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter comes from `exp.eye_tracking`, using `timestamps`, `pupil_width`, and `pupil_height`. The code defines diameter as `2 * max(width, height)` for each sample.

ii. ```python
eye=exp.eye_tracking
pupil=2.0*np.maximum(eye['pupil_width'].to_numpy(float),eye['pupil_height'].to_numpy(float))
all_pupil=interp_valid(eye['timestamps'],pupil,all_centers)
```

iii. Step 3 and Step 5 in the notes say the whitepaper describes pupil diameter as the major axis of the fitted ellipse, so the agent intentionally used both axes rather than `pupil_width` alone.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code uses the processed eye-tracking columns directly, interpolates finite pupil-diameter values onto trial-bin centers, computes experiment-specific quintile edges, and discretizes each trial after median-filling any remaining missing bins.

ii. ```python
all_pupil=interp_valid(eye['timestamps'],pupil,all_centers)
if all_pupil is None: return None,'insufficient processed pupil data'
...
pupil_edges=quintile_edges(all_pupil)
pupil_fill=float(np.nanmedian(all_pupil))
...
pb,mp=labels_from_edges(pv,pupil_edges,pupil_fill)
```

iii. The notes say the agent wanted to preserve SDK blink/outlier handling by using processed values only, not raw blink frames. Sessions with too little usable pupil support are skipped; residual missing bins are imputed to the session median and counted.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Like running speed, pupil diameter is discretized into five experiment-specific quintile bins using the 20/40/60/80 percentiles and `np.digitize`, yielding labels 0-4 (`Q1`-`Q5`).

ii. ```python
pupil_edges=quintile_edges(all_pupil)
...
pb,mp=labels_from_edges(pv,pupil_edges,pupil_fill)
...
'output_values':[IMAGE_VALUES,['no_change','change'],['Q1','Q2','Q3','Q4','Q5'],['Q1','Q2','Q3','Q4','Q5'],OUTCOME_COLS]
```

iii. Step 5 gives the same percentile-bin policy for pupil as for running, and the sample/full statistics in the notes emphasize the resulting balanced quintiles.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated onto the same 100 ms trial-bin centers used for neural bins and then split trial by trial in parallel with the neural data.

ii. ```python
all_centers=np.concatenate([g[2] for g in grids])
all_pupil=interp_valid(eye['timestamps'],pupil,all_centers)
...
rv=all_run[pos:pos+T]; pv=all_pupil[pos:pos+T]; pos+=T
```

iii. The agent's stated approach in the notes is to put all behavioral streams on the common ophys-derived trial grid so every output row lines up with the neural matrix.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial columns `hit`, `miss`, `false_alarm`, and `correct_reject` in `exp.trials`.

ii. ```python
OUTCOME_COLS = ['hit','miss','false_alarm','correct_reject']
...
exact = tr[OUTCOME_COLS].astype(int).sum(axis=1).eq(1)
...
outcome=int(np.flatnonzero(rowt[OUTCOME_COLS].to_numpy(bool))[0])
```

iii. The notes use those four AllenSDK outcome flags as the canonical task outcomes and add an exact-one-outcome check before coding them.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. After filtering trials, the code takes the index of the one true outcome flag and repeats that integer across every time bin in the trial, making the variable static in meaning but time-shaped in storage.

ii. ```python
outcome=int(np.flatnonzero(rowt[OUTCOME_COLS].to_numpy(bool))[0])
out=np.vstack([im,ch,rb,pb,np.full(T,outcome,dtype=np.int64)])
```

iii. Step 5 and trajectory step 23 explain that the agent repeated the trial outcome across time because the downstream validator expects a single `(n_output, T)` array per trial rather than a mix of dynamic rows and scalars.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: it skips experiments with insufficient running or pupil data, drops malformed outcome rows, ignores too-short trials, counts and median-imputes missing running/pupil bins after interpolation, and skips any experiment with fewer than two eligible trials. Shape mismatches in the event matrix also cause the experiment to be skipped.

ii. ```python
if good.sum() < 2:
    return None
...
exact = tr[OUTCOME_COLS].astype(int).sum(axis=1).eq(1)
...
if len(grids)<2: return None, f'fewer than 2 eligible trials ({len(grids)})'
...
if events.shape != (len(cell_ids),len(ts)):
    return None, f'event shape mismatch {events.shape} vs {len(cell_ids),len(ts)}'
...
v[missing] = fill
return np.digitize(v, edges, right=False).astype(np.int64), int(missing.sum())
```

iii. Step 5 says the agent did not want to use raw blink/outlier pupil values, so it chose interpolation over valid samples and median imputation for residual holes, while skipping sessions with no usable processed pupil stream. The trajectory also shows that preserving decoder-trainable outputs was an explicit concern.

## 9-a. What are the most time-consuming steps of the code?

i. The AI treats experiment loading through AllenSDK as the main runtime cost, with per-experiment event binning and behavioral interpolation as the main CPU-side work once an experiment is loaded. That is why the full run uses multiprocessing across experiments.

ii. ```python
exp=cache.get_behavior_ophys_experiment(int(eid))
...
event_cs=cumulative_events(events)
...
all_run=interp_valid(run['timestamps'],run['speed'],all_centers)
all_pupil=interp_valid(eye['timestamps'],pupil,all_centers)
...
pool=mp.get_context('spawn').Pool(nworkers,initializer=init_worker)
result_iter=pool.imap(worker_process,tasks,chunksize=1)
```

iii. In Step 6-9 notes and trajectory steps 24, 29, and 31, the agent repeatedly identifies data loading and per-experiment processing as the expensive part and adds an 8-worker pool to reduce wall time.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still has two obvious Python loops: the loop over trials in `make_grids` / `process_experiment`, and the loop over every stimulus-presentation row inside `image_and_change`. The agent also documents that an earlier per-trial cumulative-sum recomputation was a performance problem and vectorized that part away.

ii. ```python
for idx,row in trials.iterrows():
    ...
    grids.append((idx, edges, centers))
...
for r in sp.itertuples():
    ...
    image[a:b] = IMAGE_TO_ID[str(name)]
    ...
for tidx,edges,centers in grids:
    ...
    im,ch=image_and_change(sp,centers,edges)
```

iii. Step 6 explicitly says the original implementation repeated a full cumulative-event computation per trial and that this was replaced with a per-experiment cumulative array. The remaining per-trial and per-stimulus loops are not similarly vectorized.

## 9-c. What processing does the code repeat multiple times?

i. Within each experiment, `image_and_change` scans the full filtered stimulus table separately for every trial, and the per-trial loop repeatedly slices/interpolates outputs after the all-trial interpolation has already been computed. Earlier versions also recomputed the cumulative event array per trial before that was fixed.

ii. ```python
for tidx,edges,centers in grids:
    ...
    im,ch=image_and_change(sp,centers,edges)
    rv=all_run[pos:pos+T]; pv=all_pupil[pos:pos+T]; pos+=T
```

iii. The notes themselves flag one repeated computation that was optimized away, and the final code still repeats stimulus-table scanning on every trial. That repeated work is not reused across trials even though many trials belong to the same experiment.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `process_experiment` always builds `plot_info` for up to three trials even when `show=False`, and it also returns per-experiment quintile edges and missing-value counts that are only used for metadata/logging, not for the downstream decoder inputs/outputs themselves.

ii. ```python
neural=[]; inputs=[]; outputs=[]; plot_info=[]; pos=0; miss_run=miss_pupil=0
...
if len(plot_info)<3: plot_info.append((centers,n,out,rv,pv))
result=dict(neural=neural,input=inputs,output=outputs,cell_ids=cell_ids,
            run_edges=run_edges,pupil_edges=pupil_edges,missing_run=miss_run,
            missing_pupil=miss_pupil,plot_info=plot_info,
            n_bins=sum(x.shape[1] for x in neural),load_seconds=time.time()-t0)
if show: plot_processing(eid,result)
```

iii. The agent added `plot_info` for diagnostic plotting during development. In normal full conversion it is still populated, but then only metadata and the core trial arrays are retained; the plotting payload itself is unused downstream.
