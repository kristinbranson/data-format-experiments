# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `VisualBehaviorOphysProjectCache.from_s3_cache` to create a cache object, then discovers locally cached experiment NWB files by scanning the data directory for filenames matching experiment IDs in the experiment table. It filters the experiment table to active-behavior experiments (using `behavior_type == "active_behavior"` and `~passive`). Each experiment is loaded individually via `cache.get_behavior_ophys_experiment(eid)`.

ii.
```python
cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=DATA_DIR)
table = cache.get_ophys_experiment_table()
ids = local_experiment_ids(table)
local = table.loc[ids]
active = local[(local["behavior_type"] == "active_behavior") & (~local["passive"].astype(bool))]
# ...
exp = cache.get_behavior_ophys_experiment(int(eid))
```

iii. The AI identified locally cached experiments by scanning for NWB filenames rather than using the full experiment table, ensuring only available data is processed. Filtering to active-behavior experiments excludes passive-viewing sessions as instructed.

## 1-b. How are the data split into subjects?

i. Subjects correspond to unique `mouse_id` values extracted from prepared experiment metadata, sorted alphabetically.

ii.
```python
subjects = sorted({s["mouse_id"] for s in prepared})
subject_map = {x: i for i, x in enumerate(subjects)}
```

iii. The `mouse_id` field is the SDK's unique identifier for each animal.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id` (one imaging plane) is treated as one session. Simultaneous imaging planes from the same `ophys_session_id` are NOT merged; each plane becomes its own session in the output.

ii.
```python
ids = local_experiment_ids(table)
# ...
for i, eid in enumerate(ids, 1):
    sess = prepare_experiment(cache, eid, active.loc[eid])
```

iii. The AI justified this by noting that simultaneous planes have different ophys timestamps (offset by up to ~23 ms) and that the paper analyzes each plane independently. This avoids artificial cross-plane interpolation.

## 1-d. How are the data split into trials?

i. Trials are defined using the SDK's `exp.trials` table. Eligible trials are those that are go OR catch, not aborted, not auto-rewarded. The trial window spans from SDK `start_time` to `stop_time`. A 100 ms grid of bin centers is created within each trial window, giving variable-length trials.

ii.
```python
trials = exp.trials.copy()
eligible = (
    (trials["go"].fillna(False) | trials["catch"].fillna(False))
    & ~trials["aborted"].fillna(False)
    & ~trials["auto_rewarded"].fillna(False)
)
trials = trials.loc[eligible]
# ...
grid = trial_grid(tr.start_time, tr.stop_time)
```

```python
def trial_grid(start, stop):
    n = int(np.floor((float(stop) - float(start)) / BIN_SEC + 1e-9))
    if n < 1:
        return np.empty(0, dtype=np.float64)
    return float(start) + BIN_SEC * (np.arange(n, dtype=np.float64) + 0.5)
```

iii. The SDK's trials table provides pre-computed trial metadata. The full trial window is used rather than a fixed window around change_time. 100 ms bin centers are anchored to trial start.

## 1-e. How are trials filtered based on quality controls?

i. Multiple quality filters are applied:
1. Only go or catch trials (excludes aborted and auto-rewarded).
2. Exactly one of the four outcome flags (hit/miss/false_alarm/correct_reject) must be true.
3. Trial grid must fall within ophys timestamp support.
4. Running speed must have finite values for all grid points.
5. Pupil data must have finite values for all grid points (with max interpolation gap of 0.5s).
6. Sessions with fewer than 2 retained trials are excluded.

ii.
```python
outcome_flags = [bool(tr[x]) if pd.notna(tr[x]) else False for x in OUTCOMES]
if sum(outcome_flags) != 1:
    exclusion["ambiguous_outcome"] += 1
    continue
# ...
if grid[0] < ophys_t[0] or grid[-1] > ophys_t[-1]:
    exclusion["outside_ophys_support"] += 1
    continue
run = interp_vector(running["timestamps"], running["speed"], grid)
if not np.isfinite(run).all():
    exclusion["missing_running"] += 1
    continue
# ...
pup = interp_vector(eye["timestamps"], pupil, grid, max_gap=MAX_PUPIL_GAP_SEC)
if not np.isfinite(pup).all():
    exclusion["missing_pupil"] += 1
    continue
```

iii. The AI documented all exclusions per session and ensured that excluding trials with missing behavioral data avoids inventing a sixth "missing" bin category for the five-bin output. This is more conservative than mapping NaN to bin 0.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dff_traces` (dF/F calcium fluorescence traces) from each experiment, accessed via `experiment.dff_traces`.

ii.
```python
def stack_dff(experiment):
    cells = experiment.cell_specimen_table
    dff = experiment.dff_traces
    if not cells.index.equals(dff.index):
        dff = dff.loc[cells.index]
    traces = np.stack(dff["dff"].to_numpy()).astype(np.float32, copy=False)
    return cells.index.to_numpy(), traces
```

iii. dF/F is the standard measure of neural activity for two-photon calcium imaging. The Allen SDK provides it pre-computed with neuropil correction and baseline normalization.

## 2-b. How is the `neural` data processed?

i. The dF/F traces are linearly interpolated from native ophys timestamps onto a uniform 100 ms grid (10 Hz) anchored to each trial's start time. This is done using a vectorized linear interpolation function that processes all neurons simultaneously.

ii.
```python
def linear_sample_matrix(source_t, matrix, target_t):
    hi = np.searchsorted(source_t, target_t, side="left")
    hi = np.clip(hi, 1, source_t.size - 1)
    lo = hi - 1
    den = source_t[hi] - source_t[lo]
    alpha = ((target_t - source_t[lo]) / den).astype(np.float32)
    return matrix[:, lo] * (1.0 - alpha)[None, :] + matrix[:, hi] * alpha[None, :]
# ...
neural = linear_sample_matrix(ophys_t, dff, grid).astype(np.float32, copy=False)
```

iii. Vectorized interpolation processes all neurons in one matrix operation. The 100 ms bin size ensures a uniform temporal resolution across single-plane (~31 Hz) and multiplane (~10.7 Hz) experiments.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied. All neurons present in the SDK's `cell_specimen_table` and `dff_traces` are included. The code asserts that cell table and dF/F row counts match and that no non-finite values exist.

ii.
```python
if traces.shape[0] != len(cells):
    raise ValueError("cell table and dF/F row count differ")
if not np.isfinite(traces).all():
    raise ValueError("nonfinite values in released dF/F traces")
```

iii. The Allen SDK pipeline already applies quality control (segmentation, neuropil correction). No further filtering was deemed necessary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's neural data is aligned to the trial start. A 100 ms grid of bin centers is computed from `start_time` to `stop_time`, and dF/F is linearly interpolated onto this grid.

ii.
```python
grid = trial_grid(tr.start_time, tr.stop_time)
# grid = start + 0.1 * (arange(n) + 0.5), i.e., centers at start+50ms, start+150ms, ...
neural = linear_sample_matrix(ophys_t, dff, grid)
```

iii. Using `searchsorted` and linear interpolation ensures the neural data is sampled at the exact 100 ms grid points. The grid must fall within ophys timestamp support.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is rebinned to a uniform 100 ms (10 Hz) temporal resolution. The native ophys frame rate varies between ~10.7 Hz (multiplane) and ~31 Hz (single-plane), so rebinning is applied to ensure a consistent time bin size across all sessions.

ii.
```python
BIN_SEC = 0.100
# ...
def trial_grid(start, stop):
    n = int(np.floor((float(stop) - float(start)) / BIN_SEC + 1e-9))
    return float(start) + BIN_SEC * (np.arange(n, dtype=np.float64) + 0.5)
```

iii. A 100 ms grid was chosen because it is just below the slowest native frame rate (~10.7 Hz for multiplane), avoiding information-free upsampling while satisfying the requirement that all sessions have the same bin size.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, specifically the `image_name`, `start_time`, `end_time`, `omitted`, and `is_change` columns. Only presentations from the `change_detection` stimulus block are used.

ii.
```python
stim = exp.stimulus_presentations
stim = stim[stim["stimulus_block_name"].str.contains("change_detection", na=False)].copy()
# ...
def visible_images_and_changes(stim, grid):
    for row in relevant.itertuples():
        name = str(row.image_name)
        omitted = bool(row.omitted) if pd.notna(row.omitted) else False
        if not omitted and name not in {"omitted", "nan", "None"}:
            mask = (grid >= float(row.start_time)) & (grid < float(row.end_time))
            image[mask] = name
```

iii. Using stimulus_presentations gives precise onset/offset timing for each image flash. This allows distinguishing between image-on and gray/inter-stimulus periods.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global mapping. Gray/inter-stimulus periods and omitted flashes are assigned category 0 ("none/gray"). Real image names form categories 1+. The mapping is built from the sorted union of all image names across all sessions.

ii.
```python
real_images = sorted({str(x) for s in prepared for r in s["records"]
                      for x in r["image_names"] if x is not None})
image_values = ["none/gray"] + real_images
image_to_code = {name: i for i, name in enumerate(image_values)}
# ...
image = np.array([image_to_code.get(x, 0) for x in rec["image_names"]], dtype=np.int16)
```

iii. Including "none/gray" as a category captures the gray inter-stimulus intervals. The instructions say "of the image presented during the non-grey screen" — the AI interpreted this as requiring a separate category for gray periods rather than excluding them.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed per bin on the same 100 ms grid as the neural data. Each bin is assigned the image visible during that time period based on stimulus presentation start/end times.

ii.
```python
mask = (grid >= float(row.start_time)) & (grid < float(row.end_time))
image[mask] = name
```

iii. Using the same grid guarantees alignment with neural data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in `stimulus_presentations`, filtered to the change_detection block.

ii.
```python
is_change = bool(row.is_change) if pd.notna(row.is_change) else False
if is_change and not omitted:
    k = int(np.floor((float(row.start_time) - (grid[0] - BIN_SEC / 2)) / BIN_SEC))
    if 0 <= k < grid.size:
        change[k] = 1
```

iii. The AI used `is_change` from stimulus presentations rather than the trial-level `change_time`, getting the change event from the source closest to the actual stimulus.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Image change is a binary variable. A value of 1 is assigned to the single 100 ms bin that contains the change onset time. All other bins are 0. Both go and catch trial changes are marked (the code checks `is_change` from presentations, not the trial `go` flag).

ii. See 4-a code snippet.

iii. The AI chose to mark change as a single-bin event at the onset, interpreting "right after a change" literally as the bin containing the change.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is applied. Image change is inherently binary (0 or 1).

ii. N/A

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same 100 ms grid as neural data. The change bin is determined by computing which grid bin the presentation onset falls into.

ii. See 4-a.

iii. Same grid guarantees alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `experiment.running_speed`, which provides `timestamps` and `speed` columns.

ii.
```python
running = exp.running_speed.sort_values("timestamps")
# ...
run = interp_vector(running["timestamps"], running["speed"], grid)
```

iii. The `running_speed` attribute is the SDK's standard interface for locomotion data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the 100 ms trial grid. Then discretized into 5 bins using global percentile edges (0/20/40/60/80/100 percentiles) computed across all retained trials. Trials with any non-finite running values are excluded.

ii.
```python
run = interp_vector(running["timestamps"], running["speed"], grid)
if not np.isfinite(run).all():
    exclusion["missing_running"] += 1
    continue
# ...
run_edges = percentile_edges([r["running"] for s in prepared for r in s["records"]])
# ...
discretize(rec["running"], run_edges)
```

iii. Linear interpolation preserves the signal while resampling. Global percentile bins ensure roughly equal class counts and consistent categories across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal percentile bins using global edges computed from all retained data. The edges are at 0th, 20th, 40th, 60th, 80th, and 100th percentiles.

ii.
```python
def percentile_edges(values):
    edges = np.percentile(x, [0, 20, 40, 60, 80, 100]).astype(np.float64)
    return edges

def discretize(x, edges):
    return np.searchsorted(edges[1:-1], x, side="right").astype(np.int16)
```

iii. Global edges preserve consistent physical meaning across sessions. Equal percentile bins approximate 20% occupancy per bin globally.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same 100 ms grid as neural data using `interp_vector`.

ii.
```python
run = interp_vector(running["timestamps"], running["speed"], grid)
```

iii. Using the same grid guarantees alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `experiment.eye_tracking`, using `pupil_width` and `pupil_height` columns. The diameter is computed as `sqrt(pupil_width * pupil_height)` (geometric mean of ellipse axes). Blink frames (`likely_blink == True`) are invalidated.

ii.
```python
eye = exp.eye_tracking
pupil = np.sqrt(
    eye["pupil_width"].to_numpy(float) * eye["pupil_height"].to_numpy(float)
)
blink = eye["likely_blink"].fillna(True).to_numpy(bool)
pupil[blink] = np.nan
```

iii. The geometric mean of width and height provides an orientation-invariant equivalent-area diameter. Blink frames are set to NaN before interpolation.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. After blink removal and geometric mean computation, pupil diameter is linearly interpolated to the 100 ms grid with a maximum gap of 0.5 seconds. Trials with any remaining NaN values (from long blink gaps) are excluded entirely. Then discretized into 5 global percentile bins.

ii.
```python
MAX_PUPIL_GAP_SEC = 0.500
# ...
pup = interp_vector(eye["timestamps"], pupil, grid, max_gap=MAX_PUPIL_GAP_SEC)
if not np.isfinite(pup).all():
    exclusion["missing_pupil"] += 1
    continue
# ...
pupil_edges = percentile_edges([r["pupil"] for s in prepared for r in s["records"]])
discretize(rec["pupil"], pupil_edges)
```

iii. The 0.5s max interpolation gap prevents bridging long blink periods with artificial values. Excluding trials with remaining NaN avoids the need for a "missing" category.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 equal percentile bins using global edges.

ii.
```python
pupil_edges = percentile_edges([r["pupil"] for s in prepared for r in s["records"]])
discretize(rec["pupil"], pupil_edges)
```

iii. Same rationale as running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same 100 ms grid as neural data.

ii.
```python
pup = interp_vector(eye["timestamps"], pupil, grid, max_gap=MAX_PUPIL_GAP_SEC)
```

iii. Same grid guarantees alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
OUTCOMES = ["hit", "miss", "false_alarm", "correct_reject"]
# ...
outcome_flags = [bool(tr[x]) if pd.notna(tr[x]) else False for x in OUTCOMES]
if sum(outcome_flags) != 1:
    exclusion["ambiguous_outcome"] += 1
    continue
```

iii. Exactly one outcome must be true; ambiguous trials are excluded. These four outcomes are the SDK's canonical labels for the change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome is encoded as an integer (hit=0, miss=1, false_alarm=2, correct_reject=3) and broadcast as a constant value across all time bins in the trial.

ii.
```python
outcome = int(np.flatnonzero(outcome_flags)[0])
# ...
np.full(T, rec["outcome"], dtype=np.int16)
```

iii. Trial outcome is static per trial as specified in the instructions. Broadcasting over time allows it to fit the same (5, T) output array format as the time-varying outputs.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several error cases are handled:
- **Failed experiments**: Try-except around experiment loading; failures are recorded.
- **Empty trial grids**: Trials with no 100ms bins are excluded.
- **Grid outside ophys support**: Trials whose grid falls outside ophys timestamp range are excluded.
- **Missing running data**: Trials with non-finite interpolated running speed are excluded.
- **Missing pupil data**: Trials with non-finite interpolated pupil (including long blink gaps) are excluded.
- **Ambiguous outcomes**: Trials without exactly one outcome flag are excluded.
- **Sessions with <2 trials**: Excluded after filtering.
- **Missing eye tracking table**: All trials in that session are excluded.
- All exclusions are counted and reported per session.

ii.
```python
try:
    sess = prepare_experiment(cache, eid, active.loc[eid])
except Exception as exc:
    failed.append((eid, f"{type(exc).__name__}: {exc}"))
# ...
exclusion = Counter()
if eye is None:
    exclusion["missing_eye_table"] += len(trials)
# ...
if not np.isfinite(run).all():
    exclusion["missing_running"] += 1
    continue
```

iii. Each exclusion reason is tracked and documented in metadata. This conservative approach avoids propagating missing data into outputs.

## 9-a. What are the most time-consuming steps of the code?

i. Loading experiments via `cache.get_behavior_ophys_experiment()` dominates runtime. For full conversion, the AI uses parallel processing with `ProcessPoolExecutor` (up to 4 workers) to mitigate this.

ii.
```python
workers = min(4, os.cpu_count() or 1)
with ProcessPoolExecutor(max_workers=workers) as pool:
    futures = {pool.submit(prepare_experiment_worker, eid): eid for eid in ids}
```

iii. SDK experiment loading is I/O bound. Parallel processing reduced full conversion from ~18 min (serial) to ~5 min.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `prepare_experiment` iterates over each trial sequentially, performing interpolation and stimulus label assignment individually. The stimulus label assignment via `visible_images_and_changes` iterates over stimulus presentations within each trial.

ii.
```python
for trial_id, tr in trials.iterrows():
    # ...per-trial processing...
for row in relevant.itertuples():
    # ...per-presentation image assignment...
```

iii. Neural interpolation is already vectorized over neurons. The per-trial and per-presentation loops could theoretically be vectorized but are not the bottleneck compared to data loading.

## 9-c. What processing does the code repeat multiple times?

i. No significant processing is repeated. Each experiment is loaded once and all trial data is extracted in a single pass. Global percentile edges are computed once from the collected data.

ii. N/A

iii. The two-pass structure (experiment loading, then global assembly) was designed to avoid reloading data.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The trial outcome is broadcast as a constant across all time bins (shape `(1, T)` per trial) even though it is a per-trial static variable. This creates redundant data that could have been stored more compactly. Additionally, the "none/gray" image identity category occupies ~66% of time bins but carries no stimulus information, adding decode complexity without information.

ii.
```python
np.full(T, rec["outcome"], dtype=np.int16)
```

iii. Broadcasting the outcome was done for consistency with the (5, T) output format. The none/gray category was included to capture the full temporal structure of visual stimulation.
