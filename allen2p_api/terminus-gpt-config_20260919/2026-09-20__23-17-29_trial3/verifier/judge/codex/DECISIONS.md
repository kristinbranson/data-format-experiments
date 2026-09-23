# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI constructs an Allen SDK `VisualBehaviorOphysProjectCache` rooted at `/app/data`, gets the SDK experiment table, then limits work to locally cached `.nwb` resources discovered by scanning filenames. It further filters to `active_behavior` and non-passive experiments, and loads each retained experiment with `get_behavior_ophys_experiment()`.

ii.
```python
cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=DATA_DIR)
table = cache.get_ophys_experiment_table()
ids = local_experiment_ids(table)
local = table.loc[ids]
active = local[(local["behavior_type"] == "active_behavior") & (~local["passive"].astype(bool))]

exp = cache.get_behavior_ophys_experiment(int(eid))
```

iii. In `CONVERSION_NOTES.md`, the AI says it must use the Allen SDK exclusively, exclude passive sessions because the decoder task needs task trials and outcomes, and restrict processing to resources actually present in `/app/data`. The trajectory also states that the local cache contains 202 active experiments and that scientific contents are accessed only through `VisualBehaviorOphysProjectCache`.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values from the prepared experiment records, stored as strings and mapped to integer `subject_idx` values.

ii.
```python
subjects = sorted({s["mouse_id"] for s in prepared})
subject_map = {x: i for i, x in enumerate(subjects)}
...
"subject_idx": np.array([subject_map[s["mouse_id"]] for s in prepared], dtype=np.int16),
```

iii. The notes say `mouse_id` is the subject identifier and that 38 mice are expected in the active subset. The trajectory repeatedly summarizes the dataset in terms of unique mice counted from experiment metadata.

## 1-c. How are the data split into sessions?

i. The AI treats each `ophys_experiment_id` as one session, not each `ophys_session_id`. In practice, one imaging plane becomes one output session.

ii.
```python
def prepare_experiment(cache, eid, row, verbose=True):
    exp = cache.get_behavior_ophys_experiment(int(eid))
    ...
    result = {
        "eid": int(eid),
        "mouse_id": str(row.mouse_id),
        "region": str(row.targeted_structure),
        ...
    }
```

iii. `CONVERSION_NOTES.md` explicitly says “session unit is one ophys experiment/imaging plane” because simultaneous planes can have offset timestamps and the paper analyzed planes separately. The trajectory also says Step 4 resolved the session unit in favor of experiment-level plane sessions.

## 1-d. How are the data split into trials?

i. Trials come from `exp.trials`. After filtering, each trial is segmented from SDK `start_time` to `stop_time` and represented on a 100 ms grid whose bin centers are anchored to trial start.

ii.
```python
trials = exp.trials.copy()
...
for trial_id, tr in trials.iterrows():
    grid = trial_grid(tr.start_time, tr.stop_time)
    if grid.size == 0:
        exclusion["empty_grid"] += 1
        continue
```

```python
def trial_grid(start, stop):
    n = int(np.floor((float(stop) - float(start)) / BIN_SEC + 1e-9))
    if n < 1:
        return np.empty(0, dtype=np.float64)
    return float(start) + BIN_SEC * (np.arange(n, dtype=np.float64) + 0.5)
```

iii. The notes say the AI kept SDK `start_time`/`stop_time` trial boundaries, preserved variable trial lengths, and used a common 100 ms ophys-clock grid to satisfy the instruction that all sessions share one bin size. The trajectory records that the original idea of native frame rates was corrected in Step 5 for that reason.

## 1-e. How are trials filtered based on quality controls?

i. Trials are retained only if they are go or catch, not aborted, not auto-rewarded, have exactly one recognized outcome, yield at least one 100 ms bin, lie completely inside the ophys support, and have complete aligned running and pupil values. Sessions with fewer than two retained trials are excluded.

ii.
```python
eligible = (
    (trials["go"].fillna(False) | trials["catch"].fillna(False))
    & ~trials["aborted"].fillna(False)
    & ~trials["auto_rewarded"].fillna(False)
)
trials = trials.loc[eligible]
...
outcome_flags = [bool(tr[x]) if pd.notna(tr[x]) else False for x in OUTCOMES]
if sum(outcome_flags) != 1:
    exclusion["ambiguous_outcome"] += 1
    continue
...
if grid[0] < ophys_t[0] or grid[-1] > ophys_t[-1]:
    exclusion["outside_ophys_support"] += 1
    continue
...
if not np.isfinite(run).all():
    exclusion["missing_running"] += 1
    continue
...
if not np.isfinite(pup).all():
    exclusion["missing_pupil"] += 1
    continue
```

```python
if len(sess["records"]) >= 2:
    prepared.append(sess)
else:
    failed.append((eid, "fewer_than_two_retained_trials"))
```

iii. The notes justify the go/catch and aborted/auto-rewarded filter from the task definition, and justify excluding trials with invalid pupil/running because the requested outputs must be in five real percentile bins rather than a sixth missing-data class. The trajectory later notes that three experiments were excluded for insufficient pupil-valid trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the Allen SDK released dF/F traces, aligned to the SDK `cell_specimen_table` order and `ophys_timestamps`.

ii.
```python
cells = experiment.cell_specimen_table
dff = experiment.dff_traces
if not cells.index.equals(dff.index):
    dff = dff.loc[cells.index]
traces = np.stack(dff["dff"].to_numpy()).astype(np.float32, copy=False)
```

iii. The notes say the AI intentionally used SDK-provided released `dff_traces` rather than recomputing fluorescence metrics, because that is the documented Allen processed product and keeps loading within the SDK API.

## 2-b. How is the `neural` data processed?

i. The AI does not merge multiple imaging planes. For each experiment, it stacks dF/F rows in cell-specimen order, checks finiteness, and linearly interpolates every cell trace from native ophys timestamps onto the per-trial 100 ms grid.

ii.
```python
def linear_sample_matrix(source_t, matrix, target_t):
    ...
    hi = np.searchsorted(source_t, target_t, side="left")
    hi = np.clip(hi, 1, source_t.size - 1)
    lo = hi - 1
    den = source_t[hi] - source_t[lo]
    alpha = ((target_t - source_t[lo]) / den).astype(np.float32)
    return matrix[:, lo] * (1.0 - alpha)[None, :] + matrix[:, hi] * alpha[None, :]
...
neural = linear_sample_matrix(ophys_t, dff, grid).astype(np.float32, copy=False)
```

iii. The notes justify this as a consequence of choosing a uniform 100 ms bin size across all sessions while staying on the ophys clock. They also say not to merge simultaneous planes because plane timestamps can be offset.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no additional neuron-level biological QC beyond the Allen SDK curation implicit in `cell_specimen_table` and `dff_traces`. The code does assert that trace lengths match timestamps and that the released traces are finite.

ii.
```python
if traces.shape[0] != len(cells):
    raise ValueError("cell table and dF/F row count differ")
if not np.isfinite(traces).all():
    raise ValueError("nonfinite values in released dF/F traces")
```

iii. The notes say to “retain exactly SDK cells and dF/F rows” and not redo segmentation, ROI filtering, or dF/F computation. The extra assertions are integrity checks rather than a separate curation rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start on the ophys clock. Each trial uses absolute ophys-time bin centers from `start_time + 0.05 + 0.1*k` up to `< stop_time`.

ii.
```python
def trial_grid(start, stop):
    ...
    return float(start) + BIN_SEC * (np.arange(n, dtype=np.float64) + 0.5)
```

```python
grid = trial_grid(tr.start_time, tr.stop_time)
neural = linear_sample_matrix(ophys_t, dff, grid).astype(np.float32, copy=False)
```

iii. The notes explicitly describe the temporal alignment event as trial start with bins in absolute ophys timestamp coordinates. The trajectory says this was chosen to satisfy the instruction to align based on ophys timestamps while keeping a common bin size.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 100 ms bins for all sessions and trials. Yes: the AI rebins by linearly interpolating native ophys traces and behavior streams onto that 100 ms grid.

ii.
```python
BIN_SEC = 0.100
...
"time_bin_size": 100.0,
```

```python
grid = trial_grid(tr.start_time, tr.stop_time)
run = interp_vector(running["timestamps"], running["speed"], grid)
pup = interp_vector(eye["timestamps"], pupil, grid, max_gap=MAX_PUPIL_GAP_SEC)
neural = linear_sample_matrix(ophys_t, dff, grid).astype(np.float32, copy=False)
```

iii. The notes justify 100 ms as the highest common rate below the slowest native plane rate and say the common bin size was required by the target format. The trajectory shows this was a deliberate correction after first considering native frame rates.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stimulus_presentations`, specifically `image_name`, `start_time`, `end_time`, `omitted`, and the `stimulus_block_name` filter for change-detection rows.

ii.
```python
stim = exp.stimulus_presentations
stim = stim[stim["stimulus_block_name"].str.contains("change_detection", na=False)].copy()
stim = stim.sort_values("start_time")
...
for row in relevant.itertuples():
    name = str(row.image_name)
    omitted = bool(row.omitted) if pd.notna(row.omitted) else False
```

iii. The notes justify using the stimulus table rather than only trial metadata because image identity is time-varying within a trial and should be defined only when a real image is on screen.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each 100 ms bin, the AI marks the visible image only when that bin falls inside a non-omitted change-detection presentation interval. Gray periods and omitted flashes are set to a dedicated `none/gray` category. Image names are globally encoded into integer categories.

ii.
```python
def visible_images_and_changes(stim, grid):
    image = np.empty(grid.size, dtype=object)
    image[:] = None
    ...
    for row in relevant.itertuples():
        name = str(row.image_name)
        omitted = bool(row.omitted) if pd.notna(row.omitted) else False
        if not omitted and name not in {"omitted", "nan", "None"}:
            mask = (grid >= float(row.start_time)) & (grid < float(row.end_time))
            image[mask] = name
```

```python
real_images = sorted({str(x) for s in prepared for r in s["records"] for x in r["image_names"] if x is not None})
image_values = ["none/gray"] + real_images
image_to_code = {name: i for i, name in enumerate(image_values)}
...
image = np.array([image_to_code.get(x, 0) for x in rec["image_names"]], dtype=np.int16)
```

iii. The notes say the task asks for the image “during the non-grey screen,” so gray and omitted intervals should not be forced to one of the real image identities. The trajectory also records that `none/gray` was chosen after inspecting the stimulus timing and the presence of omitted flashes.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is evaluated on the same per-trial 100 ms grid used for neural data, so each neural time bin gets the image identity corresponding to that bin’s absolute ophys-clock time.

ii.
```python
grid = trial_grid(tr.start_time, tr.stop_time)
neural = linear_sample_matrix(ophys_t, dff, grid).astype(np.float32, copy=False)
image_names, changes = visible_images_and_changes(stim, grid)
```

iii. The notes say all streams are sampled onto one common ophys-based trial grid. The justification is to keep every output explicitly aligned with the rebinned neural activity.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the change-detection stimulus table, primarily `is_change`, `start_time`, `end_time`, `image_name`, and `omitted`. The implementation does not directly compute it from trial `go` or `change_time`.

ii.
```python
for row in relevant.itertuples():
    ...
    is_change = bool(row.is_change) if pd.notna(row.is_change) else False
    if is_change and not omitted:
        k = int(np.floor((float(row.start_time) - (grid[0] - BIN_SEC / 2)) / BIN_SEC))
        if 0 <= k < grid.size:
            change[k] = 1
```

iii. The notes say the AI checked that SDK `change_time` and presentation onsets agree, then used the presentation table because it directly marks real image-change events and naturally leaves catch trials at zero.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI represents image change as a sparse onset event: exactly one 100 ms bin is set to 1 for each real image-change onset, and all other bins are 0.

ii.
```python
change = np.zeros(grid.size, dtype=np.int8)
...
if is_change and not omitted:
    k = int(np.floor((float(row.start_time) - (grid[0] - BIN_SEC / 2)) / BIN_SEC))
    if 0 <= k < grid.size:
        change[k] = 1
```

iii. The notes explicitly say “Change is an onset event” and justify that as closer to “right after a change” than labeling an extended post-change window.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is a binary categorical output: 0 for `no_change` and 1 for `change`.

ii.
```python
"output_values": [
    image_values,
    ["no_change", "change"],
    ["q1_slowest", "q2", "q3", "q4", "q5_fastest"],
    ["q1_smallest", "q2", "q3", "q4", "q5_largest"],
    OUTCOMES,
],
```

iii. The task itself asks for a binary image-change output, and the notes describe it as a binary event variable.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is placed on the same 100 ms trial grid as neural activity, with the positive bin chosen from the absolute stimulus onset time relative to that grid.

ii.
```python
grid = trial_grid(tr.start_time, tr.stop_time)
neural = linear_sample_matrix(ophys_t, dff, grid).astype(np.float32, copy=False)
image_names, changes = visible_images_and_changes(stim, grid)
```

iii. The notes say all outputs are built directly on the common ophys-clock trial grid, and that SDK `change_time` matched presentation onsets in the checks used to justify this alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `exp.running_speed`, using its `timestamps` and `speed` columns.

ii.
```python
running = exp.running_speed.sort_values("timestamps")
...
run = interp_vector(running["timestamps"], running["speed"], grid)
```

iii. The notes identify running speed as an SDK-provided timestamped behavioral stream and say it should be aligned to the ophys timebase.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated onto each trial’s 100 ms grid without extrapolation, then discretized using global dataset-wide percentile edges.

ii.
```python
run = interp_vector(running["timestamps"], running["speed"], grid)
...
run_edges = percentile_edges([r["running"] for s in prepared for r in s["records"]])
...
discretize(rec["running"], run_edges),
```

```python
def percentile_edges(values):
    x = np.concatenate([np.asarray(v, dtype=np.float64) for v in values])
    x = x[np.isfinite(x)]
    edges = np.percentile(x, [0, 20, 40, 60, 80, 100]).astype(np.float64)
    ...
```

iii. The notes justify global quintiles so categories have the same meaning across sessions, and justify interpolation because all streams must be aligned to the ophys-derived trial grid.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded into five global percentile bins using the 0th, 20th, 40th, 60th, 80th, and 100th percentiles over all retained aligned running samples.

ii.
```python
run_edges = percentile_edges([r["running"] for s in prepared for r in s["records"]])
...
def discretize(x, edges):
    return np.searchsorted(edges[1:-1], x, side="right").astype(np.int16)
```

iii. The notes explicitly call these “global quintiles” and say per-session bins would break comparability across sessions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is interpolated directly onto the same 100 ms grid used for the neural trial matrix, so the running category at each column corresponds to the same time point as the neural column.

ii.
```python
grid = trial_grid(tr.start_time, tr.stop_time)
run = interp_vector(running["timestamps"], running["speed"], grid)
neural = linear_sample_matrix(ophys_t, dff, grid).astype(np.float32, copy=False)
```

iii. The notes say the ophys clock is the master alignment grid for all streams. Running is one of the examples given there.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `exp.eye_tracking`, using `timestamps`, `pupil_width`, `pupil_height`, and `likely_blink`.

ii.
```python
eye = exp.eye_tracking
...
pupil = np.sqrt(
    eye["pupil_width"].to_numpy(float) * eye["pupil_height"].to_numpy(float)
)
blink = eye["likely_blink"].fillna(True).to_numpy(bool)
pupil[blink] = np.nan
```

iii. The notes justify this as an explicit diameter-like quantity derived from the fitted pupil ellipse, with blink-invalid frames removed before interpolation.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI computes pupil diameter as `sqrt(width * height)`, masks likely blinks to `NaN`, linearly interpolates onto the 100 ms trial grid only across gaps of at most 0.5 s, and then discretizes the valid values with global percentile edges.

ii.
```python
MAX_PUPIL_GAP_SEC = 0.500
...
pupil = np.sqrt(
    eye["pupil_width"].to_numpy(float) * eye["pupil_height"].to_numpy(float)
)
blink = eye["likely_blink"].fillna(True).to_numpy(bool)
pupil[blink] = np.nan
...
pup = interp_vector(
    eye["timestamps"], pupil, grid, max_gap=MAX_PUPIL_GAP_SEC
)
...
pupil_edges = percentile_edges([r["pupil"] for s in prepared for r in s["records"]])
```

iii. The notes say this diameter is an orientation-invariant equivalent-area diameter, and that long blink/missing gaps should not be bridged. The trajectory also mentions pupil missingness as a key reason for trial and session exclusions.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is thresholded into five global percentile bins over all retained aligned pupil samples.

ii.
```python
pupil_edges = percentile_edges([r["pupil"] for s in prepared for r in s["records"]])
...
discretize(rec["pupil"], pupil_edges),
```

iii. The notes call these global quintiles, using the same rationale as running speed: consistent category meaning across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. It is interpolated onto the same 100 ms trial grid as neural activity, and trials are dropped if any aligned pupil bins remain invalid.

ii.
```python
grid = trial_grid(tr.start_time, tr.stop_time)
pup = interp_vector(
    eye["timestamps"], pupil, grid, max_gap=MAX_PUPIL_GAP_SEC
)
...
if not np.isfinite(pup).all():
    exclusion["missing_pupil"] += 1
    continue
```

iii. The notes say all outputs must align to the common ophys-based grid, and that trials with unresolved pupil gaps were excluded rather than assigned an invented missing category.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
OUTCOMES = ["hit", "miss", "false_alarm", "correct_reject"]
...
outcome_flags = [bool(tr[x]) if pd.notna(tr[x]) else False for x in OUTCOMES]
```

iii. The notes say these four labels are the task’s canonical non-aborted, non-auto-rewarded outcomes and that exactly one must be true for any retained trial.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI requires exactly one of the four outcome flags, converts it to an integer code by position in `OUTCOMES`, and then broadcasts that static value across every time bin in the trial output matrix.

ii.
```python
outcome_flags = [bool(tr[x]) if pd.notna(tr[x]) else False for x in OUTCOMES]
if sum(outcome_flags) != 1:
    exclusion["ambiguous_outcome"] += 1
    continue
...
"outcome": int(np.flatnonzero(outcome_flags)[0]),
```

```python
out = np.vstack([
    image,
    rec["changes"].astype(np.int16),
    discretize(rec["running"], run_edges),
    discretize(rec["pupil"], pupil_edges),
    np.full(T, rec["outcome"], dtype=np.int16),
])
```

iii. The notes first discuss a static per-trial representation, then state that if validator compatibility required it the outcome would be broadcast over time. The final code takes that broadcasted representation.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or problematic data conservatively: it uses assertions for malformed neural arrays, avoids extrapolation when interpolating, rejects trials that extend outside ophys support or still contain missing running/pupil values after interpolation, limits pupil interpolation to short gaps, and rejects experiments with fewer than two retained trials. It records exclusion reasons per experiment.

ii.
```python
if target_t.size == 0 or target_t[0] < source_t[0] or target_t[-1] > source_t[-1]:
    raise ValueError("target grid outside ophys timestamp support")
...
out = np.full(target_t.shape, np.nan, dtype=np.float32)
...
if max_gap is not None and inside.any():
    ...
    out[ii[~gap_ok]] = np.nan
...
if not np.isfinite(run).all():
    exclusion["missing_running"] += 1
    continue
...
if not np.isfinite(pup).all():
    exclusion["missing_pupil"] += 1
    continue
```

iii. The notes say this avoids fabricating invalid quintile labels for missing values, especially for pupil diameter. The trajectory explicitly highlights three sessions dropped for unusable pupil streams and frames that this as a deliberate QC rule rather than a crash.

## 9-a. What are the most time-consuming steps of the code?

i. The expensive parts are loading experiments through the Allen SDK and doing per-experiment trial extraction/interpolation over large dF/F arrays. The AI recognized that serial full conversion would be too slow and added multiprocessing for full mode.

ii.
```python
exp = cache.get_behavior_ophys_experiment(int(eid))
...
with ProcessPoolExecutor(max_workers=workers) as pool:
    futures = {pool.submit(prepare_experiment_worker, eid): eid for eid in ids}
```

iii. The trajectory says the serial estimate for full conversion was roughly 18–20 minutes and that parallelization was added to bring runtime under the required threshold. Multiple progress updates identify experiment loading and per-experiment processing as the bottleneck.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest non-vectorized loops are the per-trial loop over `trials.iterrows()` and the per-stimulus-row loop in `visible_images_and_changes()`. Both repeatedly build masks and interpolate arrays on a trial-by-trial basis.

ii.
```python
for trial_id, tr in trials.iterrows():
    ...
    run = interp_vector(running["timestamps"], running["speed"], grid)
    pup = interp_vector(
        eye["timestamps"], pupil, grid, max_gap=MAX_PUPIL_GAP_SEC
    )
    neural = linear_sample_matrix(ophys_t, dff, grid).astype(np.float32, copy=False)
    image_names, changes = visible_images_and_changes(stim, grid)
```

```python
for row in relevant.itertuples():
    ...
    mask = (grid >= float(row.start_time)) & (grid < float(row.end_time))
    image[mask] = name
```

iii. The AI did not document a specific vectorization rationale here; this conclusion follows from the structure of the implementation. The trajectory instead focused on coarse-grained parallelism across experiments.

## 9-c. What processing does the code repeat multiple times?

i. The code repeatedly interpolates running, pupil, and neural data separately for every retained trial, and repeatedly scans relevant stimulus rows per trial to rebuild image/change labels. In full mode it also re-creates the SDK cache and reloads the experiment table inside each worker process.

ii.
```python
for trial_id, tr in trials.iterrows():
    grid = trial_grid(tr.start_time, tr.stop_time)
    run = interp_vector(running["timestamps"], running["speed"], grid)
    pup = interp_vector(
        eye["timestamps"], pupil, grid, max_gap=MAX_PUPIL_GAP_SEC
    )
    neural = linear_sample_matrix(ophys_t, dff, grid).astype(np.float32, copy=False)
    image_names, changes = visible_images_and_changes(stim, grid)
```

```python
def prepare_experiment_worker(eid):
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=DATA_DIR)
    table = cache.get_ophys_experiment_table()
    return prepare_experiment(cache, int(eid), table.loc[int(eid)], verbose=False)
```

iii. There is no explicit justification in the notes for these repeated operations beyond the general decision to use a per-trial common grid and per-worker isolated SDK access. The trajectory justifies multiprocessing for speed, not these repeated substeps.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and keeps intermediate per-trial `grid`, `start_time`, `stop_time`, and `trial_id` values during conversion, but those are not written into the final dataset. It also stores `cell_ids` even though only the count is later used for `brain_region_idx`, and it can optionally generate processing plots that are not needed downstream by the decoder.

ii.
```python
records.append(
    {
        "trial_id": int(trial_id),
        "grid": grid,
        "neural": neural,
        "running": run,
        "pupil": pup,
        "image_names": image_names,
        "changes": changes,
        "outcome": int(np.flatnonzero(outcome_flags)[0]),
        "start_time": float(tr.start_time),
        "stop_time": float(tr.stop_time),
    }
)
```

```python
region_indices.append(np.full(len(sess["cell_ids"]), region_map[sess["region"]], dtype=np.int16))
...
if args.show_processing:
    for sess in prepared[:2]:
        make_plot(sess, run_edges, pupil_edges, image_to_code)
```

iii. The notes do not explicitly justify these as necessary for downstream decoding. They mainly exist for QC, plotting, metadata, and assembly convenience during conversion.
