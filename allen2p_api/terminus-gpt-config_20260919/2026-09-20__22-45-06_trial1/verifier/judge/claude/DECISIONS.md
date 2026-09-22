# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All access goes through the AllenSDK `VisualBehaviorOphysProjectCache`, built with `from_s3_cache(cache_dir='/app/data')`. The AI enumerates candidate experiments from `cache.get_ophys_experiment_table()`, then **intersects that table with the experiment IDs that are actually present as local `.nwb` files** (found with a filesystem glob on filenames only — the NWB files are never opened directly). It then keeps only rows with `behavior_type == 'active_behavior'` (dropping the 82 passive-viewing planes) and sorts by experiment ID. Each retained experiment is loaded with `cache.get_behavior_ophys_experiment(eid)`, from which `events`, `cell_specimen_table`, `ophys_timestamps`, `trials`, `stimulus_presentations`, `running_speed` and `eye_tracking` are pulled. Notably, the AI does **not** filter on `project_code`, so both `VisualBehavior` (single-plane) and `VisualBehaviorMultiscope` experiments are included. Loading is parallelised across 8 spawned worker processes, each of which builds its own cache handle in `init_worker`. Net cohort: 202 candidate experiments → 199 retained sessions, 38 mice, 29,168 cells, 51,075 trials.

ii.
```python
CACHE_DIR = Path('/app/data')

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
```
```python
def process_experiment(cache, eid, row, show=False):
    t0=time.time(); exp=cache.get_behavior_ophys_experiment(int(eid))
    ...
    ts=np.asarray(exp.ophys_timestamps,float)
    evdf=exp.events
    cell_ids=exp.cell_specimen_table.index.to_numpy()
    evdf=evdf.loc[cell_ids]
    events=np.stack(evdf['events'].to_numpy()).astype(np.float32)
```
```python
pool=mp.get_context('spawn').Pool(nworkers,initializer=init_worker)
result_iter=pool.imap(worker_process,tasks,chunksize=1)
```

iii. From CONVERSION_NOTES Step 1/Step 4: "`/app/code` is the AllenSDK repository. Its Visual Behavior ophys documentation explicitly recommends loading and interacting with NWB-backed data through AllenSDK; conversion will use `VisualBehaviorOphysProjectCache` only and never open NWB files with h5py/pynwb." Restricting to locally cached files was justified as "Conversion will process only locally available experiment files and will not download missing release data" (the SDK metadata tables describe the full 1,936-experiment release, but only 284 files exist locally). Excluding passive planes was justified by the paper: "Paper states passive viewing was not analyzed → Exclude passive experiments." Parallelism was added because the serial estimate (~22–25 min) exceeded the 15-minute budget in the instructions.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values (as strings) of the selected experiment table, sorted. `subject_idx` is the index of each session's mouse into that list. After conversion, any subject that ended up with no retained session (because of skips) is removed and the indices are remapped so that `subjects` contains only represented mice. Result: 38 mice.

ii.
```python
subjects=sorted(tab.mouse_id.astype(str).unique()); subjmap={x:i for i,x in enumerate(subjects)}
...
data['subject_idx'].append(subjmap[str(row.mouse_id)])
...
# Remove subjects not represented after any skips and remap indices.
used=sorted(set(data['subject_idx'].tolist()))
remap={old:new for new,old in enumerate(used)}
data['subjects']=[subjects[i] for i in used]
data['subject_idx']=np.asarray([remap[i] for i in data['subject_idx']],dtype=np.int64)
```

iii. CONVERSION_NOTES Step 5 mapping table: "`mouse_id` → `subjects`, `subject_idx`; String IDs and zero-based lookup; One entry per retained plane experiment." The cohort count (38 local active-task mice) was cross-checked against the SDK metadata in Step 9/Step 10 ("mice raw/converted 38 38").

## 1-c. How are the data split into sessions?

i. **One target "session" = one ophys experiment = one imaging plane.** Physical multiscope sessions that contain several simultaneously imaged planes are therefore split into several target sessions (each with its own neurons but the same behavioural trials). No merging across planes is performed. 202 active plane experiments (from 174 physical ophys sessions) become up to 202 target sessions; 199 survive filtering.

ii.
```python
tasks=[(int(eid),row.to_dict()) for eid,row in tab.iterrows()]
...
for k,(eid,rowdict,res,err) in enumerate(result_iter,1):
    ...
    data['neural'].append(res['neural']); data['input'].append(res['input']); data['output'].append(res['output'])
    data['subject_idx'].append(subjmap[str(row.mouse_id)])
    ridx=REGIONS.index(str(row.targeted_structure)); data['brain_region_idx'].append(np.full(len(res['cell_ids']),ridx,dtype=np.int64))
```

iii. CONVERSION_NOTES Step 4 discrepancy table: "'Session' granularity — Cache has physical sessions with one or multiple plane experiments / 202 active plane experiments represent 174 physical sessions / Paper decoder analyzed each imaging plane separately → Represent each ophys experiment/imaging plane as a target session. This avoids combining neurons with different plane timestamps and matches the paper's plane-wise analysis." Step 5 Key Decision 2 repeats: "one ophys experiment/imaging plane per target session, matching paper plane-wise decoding and avoiding invalid merging of distinct timestamp grids."

## 1-d. How are the data split into trials?

i. Trials come from the SDK's built-in `exp.trials` table. The retained set is `(go OR catch) AND NOT aborted AND NOT auto_rewarded`, further requiring exactly one of `hit/miss/false_alarm/correct_reject` to be True. Each trial spans the native SDK window `[start_time, stop_time)`, which is then tiled with a fixed 100 ms grid anchored at `start_time`; the trailing partial bin is dropped (`floor`). Trials therefore have variable length (70–125 bins, mean 84.2 bins ≈ 8.4 s). Trials yielding fewer than 1 bin are dropped, and an experiment with fewer than 2 remaining trials is skipped entirely.

ii.
```python
def prepare_trials(exp):
    tr = exp.trials.copy()
    mask = (tr['go'].astype(bool) | tr['catch'].astype(bool))
    mask &= ~tr['aborted'].astype(bool) & ~tr['auto_rewarded'].astype(bool)
    tr = tr.loc[mask].copy()
    exact = tr[OUTCOME_COLS].astype(int).sum(axis=1).eq(1)
    if (~exact).any():
        print(f'  warning: dropping {(~exact).sum()} trials without exactly one outcome')
        tr = tr.loc[exact]
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
```python
trials=prepare_trials(exp); grids=make_grids(trials)
if len(grids)<2: return None, f'fewer than 2 eligible trials ({len(grids)})'
```

iii. CONVERSION_NOTES Step 1: "Trial semantics in `Trial._get_trial_data`: go = real stimulus change; catch = sham change; auto-rewarded and aborted are distinct mutually exclusive categories. The requested trial set is therefore `(go OR catch) AND NOT auto_rewarded AND NOT aborted`." Step 5 Key Decision 4: "SDK `[start_time, stop_time)` boundaries. Trial starts vary relative to scheduled change by experimental design; do not crop to a fixed change-centered window because the task requests individual experimental trials." Step 10 edge-case review: "Half-open `[start, stop)` trial and event bins avoid duplicate boundary samples. Last partial 100 ms bins are discarded consistently."

## 1-e. How are trials filtered based on quality controls?

i. Filters applied, in order: (1) `aborted` and `auto_rewarded` trials removed, only `go`/`catch` kept (instruction-mandated); (2) trials whose four outcome booleans do not sum to exactly 1 are dropped with a printed warning; (3) trials shorter than one 100 ms bin are dropped; (4) whole experiments with <2 eligible trials are skipped; (5) whole experiments whose processed running or pupil stream has <2 finite samples are skipped (3 experiments: 795953296, 806456687, 833631914). No activity-based trial rejection is done — trials with all-zero detected events are deliberately kept (2,502 such trials across 88 sessions), which the verifier flags as warnings.

ii.
```python
exact = tr[OUTCOME_COLS].astype(int).sum(axis=1).eq(1)
if (~exact).any():
    print(f'  warning: dropping {(~exact).sum()} trials without exactly one outcome')
    tr = tr.loc[exact]
```
```python
if len(grids)<2: return None, f'fewer than 2 eligible trials ({len(grids)})'
...
if all_run is None: return None,'insufficient running data'
if all_pupil is None: return None,'insufficient processed pupil data'
if run_edges is None or pupil_edges is None: return None,'cannot estimate quintiles'
```

iii. Step 5 Key Decision 3: "`(go OR catch) AND NOT aborted AND NOT auto_rewarded`; require exactly one recognized outcome and at least two retained trials/session." Step 9: "Experiments 795953296, 806456687, and 833631914 had fewer than two finite processed pupil samples and were excluded. They cannot support the required pupil-diameter output without inventing an entire behavioral stream." Step 9 on zero-activity trials: "These are genuine trials from sparse detected-event traces... Removing such trials would introduce activity-dependent trial selection; filling them or using dF/F would contradict the paper's detected-event processing."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses the SDK's **detected calcium events**, `exp.events['events']` (unfiltered event magnitudes), re-indexed to the cells listed in `exp.cell_specimen_table.index`, together with `exp.ophys_timestamps`. It explicitly rejects `dff_traces` (its provisional Step 1 choice) after reading the paper.

ii.
```python
ts=np.asarray(exp.ophys_timestamps,float)
evdf=exp.events
cell_ids=exp.cell_specimen_table.index.to_numpy()
evdf=evdf.loc[cell_ids]
events=np.stack(evdf['events'].to_numpy()).astype(np.float32)
if events.shape != (len(cell_ids),len(ts)):
    return None, f'event shape mismatch {events.shape} vs {len(cell_ids),len(ts)}'
```

iii. Step 4 discrepancy table: "Paper explicitly used detected/regressed calcium-event magnitudes rather than dF/F → Use SDK `events` (unfiltered detected magnitudes), not dF/F. This supersedes the provisional Step 1 dF/F choice and most closely matches the paper." This tracks `methods.txt`: "We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces, thus removing the slow decay dynamics of the calcium indicator GCaMP6f" and "For all analysis of neural data we used the detected calcium events as described in Garrett et al."

## 2-b. How is the `neural` data processed?

i. Event magnitudes are **summed within each 100 ms bin**. To make that cheap, a per-experiment cumulative sum over time is built once (in float64 to avoid cancellation), and each trial's bin sums are obtained by differencing the cumulative array at bin-edge indices found by `np.searchsorted` on `ophys_timestamps`. Results are stored as float32 `(n_cells, T)` per trial. No smoothing, normalisation, z-scoring or neuron-wise rescaling is applied. Neurons are never merged across planes (each plane is its own session).

ii.
```python
def cumulative_events(event_matrix):
    # One cumulative array per experiment; avoids recomputing it for every trial.
    cs = np.empty((event_matrix.shape[0], event_matrix.shape[1]+1), dtype=np.float64)
    cs[:,0] = 0.0
    # float64 prevents cancellation error when late-bin sums subtract large cumulative totals.
    np.cumsum(event_matrix, axis=1, dtype=np.float64, out=cs[:,1:])
    return cs

def bin_events(event_times, event_cumsum, edges):
    # Sum each cell's detected event magnitudes in [edge_i, edge_i+1).
    lo = np.searchsorted(event_times, edges[:-1], side='left')
    hi = np.searchsorted(event_times, edges[1:], side='left')
    return (event_cumsum[:,hi] - event_cumsum[:,lo]).astype(np.float32, copy=False)
```

iii. Step 5 mapping table: "Sum detected event magnitudes in common 100 ms timestamp bins, yielding cells x time. Matches paper's detected-event signal; valid cells only. Summing preserves event magnitude across source rates." Step 10 documents that an initial float32 cumulative sum produced a precision mismatch late in long recordings, found by an independent raw-data check on trial 38, and was fixed by accumulating in float64 while still saving float32.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level quality filtering beyond what the released pipeline already did. The AI takes exactly the cells in `cell_specimen_table.index` and reindexes `exp.events` to that order, and asserts that the resulting array shape equals `(n_cells, n_ophys_timestamps)` — an experiment failing that assertion is skipped. There is no `valid_roi` threshold, no SNR/event-rate threshold, and no removal of silent cells.

ii.
```python
cell_ids=exp.cell_specimen_table.index.to_numpy()
evdf=evdf.loc[cell_ids]
events=np.stack(evdf['events'].to_numpy()).astype(np.float32)
if events.shape != (len(cell_ids),len(ts)):
    return None, f'event shape mismatch {events.shape} vs {len(cell_ids),len(ts)}'
```

iii. Step 1: "Cell curation: rely on the released `cell_specimen_table` and dF/F trace index, which represent valid segmented cells exposed by AllenSDK. No electrophysiology quality filtering applies." Step 3: "Use SDK `cell_specimen_table`/events indexing so invalid ROIs are excluded by the released pipeline." Step 10 verified "Cell indices/counts agree exactly" and "cell_total raw/converted 29168 29168".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The instruction is "temporally align based on ophys timestamp," and trials are the SDK trials. The AI anchors each trial's bin grid at the SDK `trials.start_time` and lays down 100 ms edges from there; events are assigned to bins by comparing `ophys_timestamps` to those absolute edge times (`np.searchsorted`, half-open `[edge_i, edge_{i+1})`). Behavioural streams are interpolated at the **centres** of the same bins, so all five output rows and the neural matrix share one index axis. `metadata['temporal_alignment_event']` is set to "Native SDK trial start; 100 ms bins defined in synchronized ophys timestamp coordinates", and `off_start`/`off_end` are `None` because the window is the native variable-length trial rather than a fixed offset around an event.

ii.
```python
edges = float(row.start_time) + np.arange(n+1)*BIN_S
centers = edges[:-1] + BIN_S/2
...
for tidx,edges,centers in grids:
    T=len(centers); rowt=trials.loc[tidx]
    n=bin_events(ts,event_cs,edges)
    im,ch=image_and_change(sp,centers,edges)
    rv=all_run[pos:pos+T]; pv=all_pupil[pos:pos+T]; pos+=T
```
```python
'temporal_alignment_event':'Native SDK trial start; 100 ms bins defined in synchronized ophys timestamp coordinates.',
'off_start':None,'off_end':None,
```

iii. Step 1: "Temporal alignment: SDK data streams carry synchronized timestamps. Neural bins are defined by `ophys_timestamps`; stimulus, running, and pupil streams will be mapped onto that timebase." Step 5 Key Decision 11: "`off_start`/`off_end` are `None` because trials use native variable boundaries rather than a fixed event-centered window." Alignment was spot-checked by independently reconstructing trials 0, 5 and 38 from the SDK and comparing with `np.allclose` (all passed).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **Yes — the data are rebinned.** The native ophys rate is ~31 Hz for single-plane experiments and ~11 Hz for multiscope planes. The AI imposes a single common bin of **100 ms** (`BIN_S = 0.100`, `metadata['time_bin_size'] = 100.0`) on every trial in every session. Neural events are summed within bins; running and pupil are linearly interpolated at bin centres; stimulus labels are assigned by which presentation interval contains the bin centre. Trailing partial bins are discarded.

ii.
```python
BIN_S = 0.100
...
n = int(np.floor((float(row.stop_time)-float(row.start_time))/BIN_S + 1e-9))
edges = float(row.start_time) + np.arange(n+1)*BIN_S
centers = edges[:-1] + BIN_S/2
...
'time_bin_size':100.0,
```

iii. Step 5 Key Decision 5: "100 ms common bins anchored to each trial start. This is compatible with both ~31 Hz single-plane and ~11 Hz multiscope acquisition. Neural event magnitudes are summed by timestamp; behavioral values use bin centers. Last partial bins are excluded, preventing off-by-one inclusion at trial stop." Step 4 adds that native samples "cannot be saved directly" at a single bin size because the two rigs differ, so "a common 100 ms grid is appropriate and preserves the slower multiscope resolution while allowing event magnitudes to be summed within bins." This also satisfies the format requirement that "Time bins should be the same size for all trials and sessions."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. `exp.stimulus_presentations`, restricted to the release-1.1 block(s) whose `stimulus_block_name` contains `change_detection`; within that table the columns `image_name`, `start_time`, `end_time` and `omitted` are used. It is **not** derived from the trials table's `initial_image_name`/`change_image_name`.

ii.
```python
sp=exp.stimulus_presentations
sp=sp[sp['stimulus_block_name'].astype(str).str.contains('change_detection',na=False)].copy()
```
```python
for r in sp.itertuples():
    name = r.image_name
    omitted = bool(r.omitted) if pd.notna(r.omitted) else False
    if not omitted and pd.notna(name) and str(name) in IMAGE_TO_ID:
        a=np.searchsorted(centers, float(r.start_time), side='left')
        b=np.searchsorted(centers, float(r.end_time), side='left')
        image[a:b] = IMAGE_TO_ID[str(name)]
```

iii. Step 2/Step 4: "Stimulus tables in release 1.1 have multiple blocks. The representative table contains 4,806 `change_detection_behavior` image presentations, 9,000 natural-movie rows, and gray-screen rows; only the block whose name contains `change_detection` is relevant to task image outputs." Using the presentation table (rather than the per-trial image names) is what lets the label follow the actual 250 ms flash / 500 ms grey cycle demanded by the instruction's parenthetical "(of the image presented during the non-grey screen)".

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A **17-class** categorical label per 100 ms bin: class 0 is `gray` (inter-flash grey screen, omitted flashes, and any time outside a presentation interval), classes 1–16 are the 16 natural images of sets A and B in a hard-coded, sorted, global order. A bin gets an image label if its centre falls in `[presentation.start_time, presentation.end_time)`; otherwise it stays `gray`. Omitted presentations are deliberately mapped to `gray` rather than to a 17th image class. The mapping is fixed globally across all sessions (so codes are comparable) and is also exported in `metadata['image_values']` / `output_values[0]`.

ii.
```python
IMAGE_VALUES = ['gray','im000','im031','im035','im045','im054','im061','im062',
                'im063','im065','im066','im069','im073','im075','im077','im085','im106']
IMAGE_TO_ID = {v:i for i,v in enumerate(IMAGE_VALUES)}
...
def image_and_change(sp, centers, edges):
    image = np.zeros(len(centers), dtype=np.int64)
    change = np.zeros(len(centers), dtype=np.int64)
    # Presentation intervals are non-overlapping; omitted rows stay gray.
    for r in sp.itertuples():
        ...
```

iii. Step 5 Key Decision 6: "deterministic global order `gray`, then sorted IDs from sets A/B: im000, im031, ... im106." Step 4/trajectory: "Omissions are no-image events and should map to gray rather than become a seventeenth image identity." The resulting distribution was used as a consistency check: gray = 0.665 of all bins, i.e. images occupy ~33.5% of time, matching the 250 ms-image / 750 ms-cycle design, and each of the 16 images occupies 1.9–2.3%.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is computed directly on the same 100 ms bin grid as the neural matrix: a presentation's `[start_time, end_time)` is converted to bin indices with `np.searchsorted` over the trial's bin **centres**, so a bin is labelled with an image exactly when its centre lies inside the flash. The resulting row is stacked with the other outputs into a `(5, T)` array whose `T` is asserted equal to the neural `T`.

ii.
```python
a=np.searchsorted(centers, float(r.start_time), side='left')
b=np.searchsorted(centers, float(r.end_time), side='left')
image[a:b] = IMAGE_TO_ID[str(name)]
...
out=np.vstack([im,ch,rb,pb,np.full(T,outcome,dtype=np.int64)])
assert n.shape[1]==T and out.shape==(5,T) and np.isfinite(n).all()
```

iii. Step 4 "Final Understanding": "all signals are sampled on the native ophys timestamps within those half-open intervals." Step 10 alignment check: "Timestamp search on `ophys_timestamps`; behavior interpolated at 100 ms centers... Three raw trial spot-checks performed" — the independent script reproduces image identity from the SDK and `np.allclose` passes for trials 0, 5, 38.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The `is_change` boolean and `start_time` of rows in the change-detection block of `exp.stimulus_presentations`. (The trials table's `go`/`change_time` columns are not used for this row.)

ii.
```python
if bool(r.is_change):
    # 'Right after change': label the 400 ms post-onset response window
    # used by the reference paper's image-change decoder.
    a=np.searchsorted(centers, float(r.start_time), side='left')
    b=np.searchsorted(centers, float(r.start_time)+0.400, side='left')
    change[a:b]=1
```

iii. Step 5 mapping table: "change-detection `stimulus_presentations.is_change/start_time` → output row 1, `image_change`... Catch sham changes remain 0 because image identity does not change." `is_change` is the SDK's flag for a presentation whose image differs from the previous one, which is exactly "a change in image identity"; on catch trials the sham "change" re-presents the same image, so `is_change` is False there.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary `(T,)` row, initialised to 0, set to 1 for the bins covering the **first 400 ms after each real change onset** (4 bins at 100 ms). Nothing else — no smoothing, no pre-change marking, no marking of catch/sham changes. Across the full dataset 4.1% of bins are labelled `change`.

ii.
```python
change = np.zeros(len(centers), dtype=np.int64)
...
a=np.searchsorted(centers, float(r.start_time), side='left')
b=np.searchsorted(centers, float(r.start_time)+0.400, side='left')
change[a:b]=1
```

iii. This was revised during Step 8. Iteration 1 used a single 100 ms impulse: "Issue: image change was slightly below chance. Its initial single-100-ms-bin target occupied only 1.1% of samples and was narrower than both calcium response dynamics and the reference paper's decoding window. Fix: define 'right after change' as the first 400 ms after a real change onset, matching the paper's first-400-ms change decoder window." Step 5 Key Decision 8: "'right after change' is the first 400 ms after a real image-change onset, matching the paper decoder window and calcium response timescale; catch sham changes and pre-change bins remain 0." (Note: the Step 5 *mapping table* still describes the superseded one-bin impulse — a stale documentation line, not a code discrepancy.)

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding of a continuous quantity is needed — the variable is natively binary. The only "threshold" is the temporal one: the 400 ms post-onset window defines the extent of the `1` label; `output_values[1] = ['no_change','change']`.

ii.
```python
'output_values':[IMAGE_VALUES,['no_change','change'],['Q1','Q2','Q3','Q4','Q5'],['Q1','Q2','Q3','Q4','Q5'],OUTCOME_COLS],
```

iii. Same as 4-b: the 400 ms extent is justified by the reference paper's change/repeat random-forest decoder, which used "the first 400 ms after image", and by the calcium response timescale.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identically to image identity — computed on the trial's 100 ms bin centres via `np.searchsorted`, then stacked as row 1 of the `(5, T)` output array, with `T` asserted equal to the neural `T`.

ii.
```python
im,ch=image_and_change(sp,centers,edges)
...
out=np.vstack([im,ch,rb,pb,np.full(T,outcome,dtype=np.int64)])
assert n.shape[1]==T and out.shape==(5,T)
```

iii. Step 10 edge-case review: "Change labels contain only post-onset bins and do not mark catch sham changes." Independently verified against raw SDK data for trials 0, 5 and 38 with `np.allclose`.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `exp.running_speed`, using its `timestamps` and `speed` columns — the SDK's *processed* running speed (cm/s), not `raw_running_speed`.

ii.
```python
run=exp.running_speed
...
all_run=interp_valid(run['timestamps'],run['speed'],all_centers)
```

iii. Step 1: "`BehaviorSession.running_speed` ... processing unwraps wheel voltage, computes angular derivative, clips wrap artifacts, applies a 10 Hz low-pass filter, and converts to cm/s"; and "Running: use `running_speed`, not `raw_running_speed`, to retain the released SDK processing."

## 5-b. What processing is involved in computing `output` *Running speed*?

i. All eligible trials' bin centres for the experiment are concatenated into one query vector; running speed is linearly interpolated onto that vector (`np.interp`, with NaN outside the sampled range, after dropping non-finite timestamps/values and de-duplicating timestamps). The interpolated values are then discretised into quintiles (see 5-c) and the per-trial slices are cut back out by running offset. Interpolation is done once per experiment rather than once per trial.

ii.
```python
def interp_valid(times, values, query):
    times = np.asarray(times, float); values = np.asarray(values, float)
    good = np.isfinite(times) & np.isfinite(values)
    if good.sum() < 2:
        return None
    t, idx = np.unique(times[good], return_index=True)
    v = values[good][idx]
    if len(t) < 2:
        return None
    out = np.interp(query, t, v, left=np.nan, right=np.nan)
    return out
```
```python
all_centers=np.concatenate([g[2] for g in grids])
all_run=interp_valid(run['timestamps'],run['speed'],all_centers)
...
rv=all_run[pos:pos+T]; pv=all_pupil[pos:pos+T]; pos+=T
```

iii. Step 5 mapping table: "Linear interpolation at bin centers, then session-wise five equal-frequency percentile bins. SDK-processed cm/s; thresholds computed from eligible-trial samples." Step 6 lists "Vectorized interpolation and percentile calculations over concatenated eligible-trial bin centers" as a deliberate speed-up.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-frequency bins from the **20/40/60/80th percentiles computed per experiment/session** over the interpolated bin-centre samples of that session's eligible trials only. `np.digitize(..., right=False)` yields labels 0–4 (`Q1`–`Q5`). Any non-finite value is replaced by the session's valid median before digitising, and the number of such imputations is recorded per session (0 for every retained session). Verified distribution: each class 0.200 ± 0.00002.

ii.
```python
def quintile_edges(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    return np.nanpercentile(v, [20,40,60,80])

def labels_from_edges(values, edges, fill):
    v = np.asarray(values, float).copy()
    missing = ~np.isfinite(v)
    v[missing] = fill
    return np.digitize(v, edges, right=False).astype(np.int64), int(missing.sum())
```
```python
run_edges=quintile_edges(all_run); pupil_edges=quintile_edges(all_pupil)
run_fill=float(np.nanmedian(all_run)); pupil_fill=float(np.nanmedian(all_pupil))
...
rb,mr=labels_from_edges(rv,run_edges,run_fill)
```

iii. Step 5 Key Decision 7: "thresholds are 20/40/60/80 percentiles over all eligible binned samples within each experiment/session. `np.digitize` gives labels 0..4. Session-level discretization prevents between-mouse calibration differences from dominating while producing approximately equal class occupancy." Per-session edges are also stored in `metadata['session_info'][i]['running_quintile_edges']`.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. The interpolation query points *are* the neural bin centres, so alignment is structural: `all_centers` is the concatenation of every trial's bin centres in trial order, and the per-trial slice is taken with a running offset `pos` that advances by exactly `T` per trial. The result becomes row 2 of the `(5, T)` output array.

ii.
```python
all_centers=np.concatenate([g[2] for g in grids])
all_run=interp_valid(run['timestamps'],run['speed'],all_centers)
...
pos=0
for tidx,edges,centers in grids:
    T=len(centers)
    rv=all_run[pos:pos+T]; pv=all_pupil[pos:pos+T]; pos+=T
```

iii. Step 1/Step 4: SDK streams carry hardware-synchronised timestamps, so interpolating onto the ophys-derived bin centres is valid: "Temporal synchronization of all data-streams ... was achieved by recording all experimental clocks on a single NI PCI-6612 digital IO board" (methods.txt). Confirmed by the independent reconstruction (`np.allclose` on the running row for trials 0, 5, 38) and by the `--show-processing` plots overlaying continuous speed with its quintile labels.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `exp.eye_tracking`, columns `timestamps`, `pupil_width` and `pupil_height`. The AI does not subset on `likely_blink` explicitly; it relies on the fact that the SDK's `filter_on_blinks()` has already set `pupil_width`/`pupil_height` to NaN on likely-blink/outlier frames, and then drops non-finite values inside `interp_valid`.

ii.
```python
eye=exp.eye_tracking
pupil=2.0*np.maximum(eye['pupil_width'].to_numpy(float),eye['pupil_height'].to_numpy(float))
...
all_pupil=interp_valid(eye['timestamps'],pupil,all_centers)
```

iii. Step 1: "`BehaviorSession.eye_tracking` ... SDK flags likely blinks/outliers (default modified z threshold 3, dilation 2 frames) and replaces affected ellipse values with NaN." Step 2: "SDK-filtered pupil ellipse fields can be NaN around likely blinks/outliers (9.2% filtered pupil-area missingness in the representative experiment)". Step 5 Key Decision 9: "never use raw blink/outlier pupil values."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Diameter is defined as `2 * max(pupil_width, pupil_height)`, i.e. twice the major semi-axis of the fitted pupil ellipse. That is then interpolated onto the trial bin centres exactly as running speed, and quintile-binned. Blink/outlier NaNs are excluded from both the interpolation support and the percentile estimation; residual NaNs are median-filled and counted.

ii.
```python
pupil=2.0*np.maximum(eye['pupil_width'].to_numpy(float),eye['pupil_height'].to_numpy(float))
all_pupil=interp_valid(eye['timestamps'],pupil,all_centers)
...
pupil_edges=quintile_edges(all_pupil)
pupil_fill=float(np.nanmedian(all_pupil))
pb,mp=labels_from_edges(pv,pupil_edges,pupil_fill)
```

iii. Step 3: "Pupil diameter should use the major axis of the fitted pupil ellipse (`2 * max(pupil_width, pupil_height)` if SDK width/height are half-axes), preserving SDK blink/outlier exclusions." This matches the SDK's own `compute_circular_area(df_row) = np.pi * max_dim * max_dim`, which treats `max(pupil_width, pupil_height)` as the pupil **radius**, so `2*max(...)` is the diameter in the SDK's own convention.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same scheme as running speed: per-session 20/40/60/80th percentiles of the interpolated bin-centre samples of that session's eligible trials, `np.digitize` → labels 0–4 (`Q1`–`Q5`), NaN → session median before digitising, imputation count recorded. Sessions with fewer than 2 finite pupil samples are dropped entirely rather than fabricated (3 sessions). Verified distribution: each class 0.200 ± 0.00002, 0 imputed bins across retained sessions.

ii.
```python
if all_pupil is None: return None,'insufficient processed pupil data'
run_edges=quintile_edges(all_run); pupil_edges=quintile_edges(all_pupil)
if run_edges is None or pupil_edges is None: return None,'cannot estimate quintiles'
...
pb,mp=labels_from_edges(pv,pupil_edges,pupil_fill)
```

iii. Step 5 Key Decisions 7 and 9: "Session-level discretization prevents between-mouse calibration differences from dominating"; "values outside valid temporal support or in sessions with insufficient valid points are median-imputed and explicitly reported. Sessions with no usable pupil stream are skipped because the requested output cannot be constructed responsibly."

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Identically to running speed — interpolated at the neural bin centres for all eligible trials at once, then sliced per trial with the same running offset, and stacked as row 3 of the `(5, T)` output array.

ii.
```python
all_pupil=interp_valid(eye['timestamps'],pupil,all_centers)
...
rv=all_run[pos:pos+T]; pv=all_pupil[pos:pos+T]; pos+=T
...
out=np.vstack([im,ch,rb,pb,np.full(T,outcome,dtype=np.int64)])
```

iii. Same hardware-synchronisation argument as running speed (Step 4 "Final Understanding": "all signals are sampled on the native ophys timestamps within those half-open intervals"), confirmed by the independent `np.allclose` reconstruction of the pupil row for three trials.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the SDK trials table: `hit`, `miss`, `false_alarm`, `correct_reject`.

ii.
```python
OUTCOME_COLS = ['hit','miss','false_alarm','correct_reject']
...
outcome=int(np.flatnonzero(rowt[OUTCOME_COLS].to_numpy(bool))[0])
```

iii. Step 1: "`BehaviorSession.trials` / `Trials` — Return trial boundaries, mutually exclusive go/catch/auto-rewarded/aborted categories, and hit/miss/false-alarm/correct-reject outcomes." Step 3 lists "Trial outcome classes: hit, miss, false alarm, correct reject" as sourced from the whitepaper and SDK trial definitions.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The index of the single True column gives an integer class 0–3 (order = `['hit','miss','false_alarm','correct_reject']`). Because the target format stores all five outputs in a single `(5, T)` array, the constant class is **broadcast across all T bins** of the trial. Trials that do not have exactly one True outcome are dropped upstream in `prepare_trials`, so the `flatnonzero(...)[0]` lookup is always well-defined. `validate()` asserts the row is constant within each trial. Observed distribution over bins: hit 0.303, miss 0.571, false alarm 0.017, correct reject 0.108.

ii.
```python
outcome=int(np.flatnonzero(rowt[OUTCOME_COLS].to_numpy(bool))[0])
out=np.vstack([im,ch,rb,pb,np.full(T,outcome,dtype=np.int64)])
...
assert o[4].min()>=0 and o[4].max()<4 and np.all(o[4]==o[4,0])
```

iii. Step 5 mapping table: "Classes 0..3; repeat constant class across T bins. Semantically static per trial. Repetition is required because the validator stores all five outputs in one `(5,T)` array."

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **Missing/blink pupil samples**: excluded from interpolation support and from percentile estimation (`interp_valid` drops non-finite pairs); duplicate timestamps de-duplicated via `np.unique`.
- **Values outside a stream's temporal support**: `np.interp` returns NaN (`left`/`right`), which is then replaced by the session's valid median and *counted* (`imputed_running_bins`, `imputed_pupil_bins` in session metadata). Across the full run these counts are all 0.
- **Streams too sparse to use at all** (<2 finite samples): the whole experiment is skipped with a reason string, recorded in `metadata['skipped_experiments']` (3 experiments).
- **Trials with malformed outcome flags** (not exactly one of hit/miss/FA/CR): dropped with a printed warning.
- **Degenerate trials** (window shorter than one bin) and **degenerate sessions** (<2 eligible trials): skipped.
- **Event/timestamp shape mismatch**: experiment skipped with a diagnostic message.
- **Numerical precision**: cumulative event sums accumulated in float64 to prevent catastrophic cancellation late in long recordings (a bug found and fixed in Step 10).
- **Genuinely zero activity**: trials with all-zero detected events are deliberately *kept*, since deleting them would be activity-dependent selection.
There is no blanket `try/except` around per-experiment processing, so an unanticipated exception would propagate rather than be skipped.

ii.
```python
good = np.isfinite(times) & np.isfinite(values)
if good.sum() < 2:
    return None
t, idx = np.unique(times[good], return_index=True)
...
out = np.interp(query, t, v, left=np.nan, right=np.nan)
```
```python
def labels_from_edges(values, edges, fill):
    v = np.asarray(values, float).copy()
    missing = ~np.isfinite(v)
    v[missing] = fill
    return np.digitize(v, edges, right=False).astype(np.int64), int(missing.sum())
```
```python
if events.shape != (len(cell_ids),len(ts)):
    return None, f'event shape mismatch {events.shape} vs {len(cell_ids),len(ts)}'
...
if err:
    print('  SKIP:',err); skipped.append((int(eid),err)); continue
...
data['metadata']['skipped_experiments']=skipped
```

iii. Step 5 Key Decision 9 and Step 9/10. On imputation: "residual unavailable values use session valid median and are counted." On skipping: "Sessions with no usable pupil stream are skipped because the requested output cannot be constructed responsibly... They cannot support the required pupil-diameter output without inventing an entire behavioral stream." On zero-activity trials: "Removing such trials would introduce activity-dependent trial selection; filling them or using dF/F would contradict the paper's detected-event processing." On float precision: "float32 cumulative sums suffered cancellation when differenced late in a recording. Changed cumulative accumulation to float64 while retaining float32 saved matrices."

## 9-a. What are the most time-consuming steps of the code?

i. Loading each experiment through the SDK — `cache.get_behavior_ophys_experiment(eid)` and the attribute accesses it triggers (`events`, `stimulus_presentations`, `eye_tracking`, `running_speed`), which deserialise large NWB-backed tables. The script times this (`load_seconds` → `processing_seconds` per session) and reports ~4.8–8.2 s per experiment, of which the SDK load dominates; the pure numeric work (one cumulative sum, searchsorted, interpolation) is a small fraction. Secondarily, the per-trial Python loop over the full `stimulus_presentations` table in `image_and_change` is the largest *compute* cost. The full run took 245 s wall-clock with 8 processes for 202 experiments (~9.7 s/experiment serial-equivalent).

ii.
```python
def process_experiment(cache, eid, row, show=False):
    t0=time.time(); exp=cache.get_behavior_ophys_experiment(int(eid))
    ...
    result=dict(..., load_seconds=time.time()-t0)
```
```python
import multiprocessing as mp
nworkers=min(8,len(tasks))
pool=mp.get_context('spawn').Pool(nworkers,initializer=init_worker)
result_iter=pool.imap(worker_process,tasks,chunksize=1)
```

iii. Step 7: "Because the serial estimate exceeds 15 minutes, safe experiment-level parallelism will be added after checking available CPU/RAM." Step 6 also notes the earlier bottleneck it removed: "Initial implementation recomputed a full cell-by-time cumulative event array for every trial, which would scale poorly and produce excessive temporary memory allocations" → "Compute one float32 [later float64] cumulative event array per experiment, then obtain every trial/bin sum by vectorized timestamp search and cumulative differences."

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest candidate is `image_and_change`: it is called once **per trial** and each call iterates over the **entire session's** change-detection presentation table with `sp.itertuples()` (~4,800 rows). With ~250 eligible trials that is ~1.2 M Python-level iterations per experiment, nearly all of which do nothing because the presentation lies outside the trial window. This could be a single vectorised pass: compute presentation-index-per-bin for all bins at once with `np.searchsorted(sp.start_time, all_centers)` plus an `end_time` validity mask, and build the change mask from the `is_change` onsets with one broadcasted comparison. A secondary candidate is `make_grids`, which uses `trials.iterrows()`; the bin counts and edge offsets could be computed from the `start_time`/`stop_time` columns vectorised. The per-trial main loop itself (`bin_events`, slicing, `labels_from_edges`) could also be collapsed by binning the whole concatenated edge vector once instead of once per trial. The AI did not identify any of these in its notes.

ii.
```python
def image_and_change(sp, centers, edges):
    ...
    for r in sp.itertuples():          # full session table, re-scanned for every trial
        ...
```
```python
for idx,row in trials.iterrows():      # row-wise pandas iteration
    n = int(np.floor((float(row.stop_time)-float(row.start_time))/BIN_S + 1e-9))
```
```python
for tidx,edges,centers in grids:
    n=bin_events(ts,event_cs,edges)    # searchsorted over full ts, once per trial
```

iii. The AI's notes only claim the vectorisations it *did* implement ("Vectorized timestamp search/interpolation — Avoids per-bin Python loops"; "Vectorized interpolation and percentile calculations over concatenated eligible-trial bin centers"). It never flags the quadratic trial × presentation scan. Its stated justification for stopping optimisation was that the target was met: the full conversion ran in 245 s, well under the 15-minute budget.

## 9-c. What processing does the code repeat multiple times?

i. Repeated work that remains in the final script:
- **Full stimulus-presentation table re-scanned once per trial** in `image_and_change` (the main redundancy; the table is identical for every trial of an experiment).
- **`np.searchsorted` over the full `ophys_timestamps` array once per trial** in `bin_events`, rather than once per experiment over all concatenated edges.
- **`pd.Series(rowdict)` reconstructed twice** per experiment — once inside `worker_process` and again in the parent loop.
- **`get_cache()` called in the parent and again in every worker** (unavoidable with `spawn`, and amortised).
- `selected_table` globs the cache directory (`local_experiment_ids`) — once, cheap.
The one large repetition the AI *did* eliminate was the per-trial cumulative-sum recomputation, replaced by one cumulative array per experiment.

ii.
```python
def worker_process(task):
    eid, row_dict = task
    row = pd.Series(row_dict)          # built here ...
    res, err = process_experiment(_WORKER_CACHE, eid, row, show=False)
    return eid, row_dict, res, err
...
for k,(eid,rowdict,res,err) in enumerate(result_iter,1):
    row=pd.Series(rowdict)             # ... and again here
```
```python
def bin_events(event_times, event_cumsum, edges):
    lo = np.searchsorted(event_times, edges[:-1], side='left')
    hi = np.searchsorted(event_times, edges[1:], side='left')
```

iii. Step 6: "Initial implementation recomputed a full cell-by-time cumulative event array for every trial... Compute one float32 cumulative event array per experiment, then obtain every trial/bin sum by vectorized timestamp search and cumulative differences." The remaining repetitions are not acknowledged in the notes.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Identified waste:
- **`plot_info` is always collected**, even when `--show-processing` is not passed: up to three trials' `(centers, neural, output, running, pupil)` arrays are accumulated and, in `--full` mode, **pickled across the multiprocessing boundary** back to the parent, where they are never used. This is pure IPC and memory overhead on every one of the 202 experiments.
- **`edges` is passed to `image_and_change` but never used** inside it (all indexing is done on `centers`), so `make_grids` returns and ships a value the consumer ignores.
- The **cumulative event array is built over the whole recording** (`n_cells × (T+1)` float64, e.g. 666 × 140,205 ≈ 750 MB for the largest plane) although only the ~35% of time inside eligible trials is ever read.
- Within `image_and_change`, the vast majority of the per-trial presentation scan produces empty slices (`a == b`), i.e. computation whose result is discarded.
- `pupil = 2*max(width, height)` is computed for the entire eye-tracking table, including the blink rows that are immediately discarded by `interp_valid`.
- `row_dict` is returned from each worker only so the parent can rebuild a `pd.Series` it already had in `tab`.
- The output arrays are stored as **`int64`** although every label is in 0–16 (`int8` would suffice), costing ~8× more bytes than necessary for the output stream and inflating the pickle.
- The `(0, T)` empty input arrays are materialised per trial; they are required by the format spec, but they are genuinely empty payloads.

ii.
```python
neural=[]; inputs=[]; outputs=[]; plot_info=[]; ...
for tidx,edges,centers in grids:
    ...
    if len(plot_info)<3: plot_info.append((centers,n,out,rv,pv))   # collected unconditionally
result=dict(neural=neural,...,plot_info=plot_info,...)             # shipped back even when show=False
```
```python
def image_and_change(sp, centers, edges):    # `edges` never referenced in the body
```
```python
cs = np.empty((event_matrix.shape[0], event_matrix.shape[1]+1), dtype=np.float64)
```

iii. The AI's notes do not identify any of these; its efficiency discussion (Step 6/Step 7) covers only the cumulative-sum restructuring, vectorised interpolation, "Process and release one experiment at a time", and "write only float32 neural arrays and integer categorical arrays". The implicit justification is that the conversion already met the runtime target (245 s for the full dataset), so no further profiling was pursued.
