# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `VisualBehaviorOphysProjectCache.from_s3_cache()` to create a cache object, then calls `cache.get_ophys_experiment_table()` to get the experiment metadata. It enumerates locally available `.nwb` files to determine which experiments are present on disk, then filters to `project_code == 'VisualBehavior'` and `passive == False`. Each experiment is loaded independently via `cache.get_behavior_ophys_experiment(experiment_id)` in parallel using `ProcessPoolExecutor`.

ii.
```python
def local_experiment_ids(cache) -> list[int]:
    table = cache.get_ophys_experiment_table()
    file_ids = {
        int(path.stem.rsplit("_", 1)[1])
        for path in (DATA_DIR / "visual-behavior-ophys-1.1.0" /
                     "behavior_ophys_experiments").glob("*.nwb")
    }
    keep = (
        table.index.isin(file_ids)
        & table["project_code"].eq("VisualBehavior")
        & ~table["passive"].astype(bool)
    )
    return sorted(table.index[keep].astype(int).tolist())
```

```python
with ProcessPoolExecutor(max_workers=workers) as pool:
    futures = {pool.submit(extract_session, eid): eid for eid in ids}
```

iii. The AI chose to enumerate local NWB files to avoid downloading missing experiments from S3. The passive filter was added because passive sessions have no genuine behavioral responses (the lick spout is retracted), which would create artificial trial outcomes. ProcessPoolExecutor was used to parallelize I/O-bound experiment loading.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mouse_id` from each experiment's metadata. Unique mouse IDs are collected across all processed experiments and sorted.

ii.
```python
subjects = sorted({s["mouse_id"] for s in raw_sessions})
subject_map = {name: i for i, name in enumerate(subjects)}
```

In `extract_session`:
```python
"mouse_id": str(meta["mouse_id"]),
```

iii. `mouse_id` is the SDK's canonical unique identifier for each animal. Sorting ensures deterministic ordering.

## 1-c. How are the data split into sessions?

i. Each experiment (one imaging plane) is treated as a separate session. For the exact `VisualBehavior` project, this is one-to-one with `ophys_session_id` since these are single-plane recordings. The AI does not merge multiple planes per session.

ii.
```python
def extract_session(experiment_id: int) -> dict:
    cache = make_cache()
    exp = cache.get_behavior_ophys_experiment(int(experiment_id))
    ...
```

iii. The AI's CONVERSION_NOTES state: "These are single-plane VISp recordings, so session and experiment are one-to-one and no asynchronous plane merge is needed." For the VisualBehavior project (not Multiscope), each session has exactly one imaging plane, so experiment = session.

## 1-d. How are the data split into trials?

i. Trials are defined using the SDK's `exp.trials` table, sorted by `start_time`. Each trial spans `[start_time, stop_time)` on a 30 Hz time grid. The grid is computed as `start + arange(n) * (1/30)` where `n = ceil((stop - start) * 30)`, keeping only points `< stop` (half-open interval).

ii.
```python
def _trial_grid(start: float, stop: float) -> np.ndarray:
    n = int(np.ceil((stop - start) * FS - 1e-10))
    grid = start + np.arange(max(0, n), dtype=np.float64) * DT
    return grid[grid < stop]

grids = [_trial_grid(float(r.start_time), float(r.stop_time))
         for r in trials.itertuples()]
```

iii. The half-open `[start, stop)` interval prevents boundary duplication between consecutive trials. The 30 Hz grid matches the paper's interpolation rate.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to include only `go` or `catch` trials, excluding `aborted` and `auto_rewarded` trials. Passive sessions are excluded at the experiment level. Sessions with fewer than 2 eligible trials are rejected. Sessions with empty eye tracking tables are also excluded.

ii.
```python
eligible = (
    (trials["go"] | trials["catch"])
    & ~trials["aborted"]
    & ~trials["auto_rewarded"]
)
trials = trials.loc[eligible].copy()
if len(trials) < 2:
    raise ValueError(f"{experiment_id}: fewer than two eligible trials")
```

Passive filtering in `local_experiment_ids`:
```python
keep = (
    table.index.isin(file_ids)
    & table["project_code"].eq("VisualBehavior")
    & ~table["passive"].astype(bool)
)
```

Eye tracking check:
```python
if eye is None or eye.empty:
    raise MissingRequiredData("eye tracking table is missing or empty")
```

iii. The instructions say to include Go and Catch trials and exclude Aborted and Auto-rewarded. Passive sessions were excluded because the paper states passive imaging "was not analyzed" and passive sessions generate artificial all-miss/all-CR outcomes. Sessions with empty eye tracking were excluded because the pupil output cannot be computed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `exp.events` — the raw L0 calcium event magnitude traces, not dF/F.

ii.
```python
event_table = exp.events
cell_ids = event_table.index.to_numpy(copy=True)
events = np.vstack(event_table["events"].to_numpy()).astype(np.float32)
```

iii. The AI's CONVERSION_NOTES state: "The paper's neural analyses—including its random-forest decoding—use detected calcium event magnitude traces rather than ΔF/F, explicitly to remove slow GCaMP6f decay. Event-based neural input is therefore the strongest reference-matching choice."

## 2-b. How is the `neural` data processed?

i. Raw L0 event traces are linearly interpolated from native ophys timestamps to a 30 Hz trial-relative grid. A custom vectorized interpolation function (`interp_rows_shared_time`) performs the timestamp bracketing once and applies it to all neurons simultaneously.

ii.
```python
ophys_t = np.asarray(exp.ophys_timestamps, dtype=np.float64)
neural_all = interp_rows_shared_time(ophys_t, events, all_t)
neural = [neural_all[:, offsets[i]:offsets[i + 1]]
          for i in range(len(lengths))]
```

```python
def interp_rows_shared_time(source_t, source_x, target_t):
    right = np.searchsorted(source_t, target_t, side="right")
    right = np.clip(right, 1, len(source_t) - 1)
    left = right - 1
    denom = source_t[right] - source_t[left]
    weight = ((target_t - source_t[left]) / denom).astype(np.float32)
    lo = source_x[:, left]
    hi = source_x[:, right]
    out = lo + (hi - lo) * weight[None, :]
    ...
```

iii. The 30 Hz resampling matches the paper's interpolation rate ("event-triggered neural and running traces were linearly interpolated to a common 30 Hz time series"). The vectorized interpolation avoids per-neuron `np.interp` calls.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied beyond SDK defaults. All cells present in the events table (which only includes valid ROIs) are used.

ii.
```python
event_table = exp.events
cell_ids = event_table.index.to_numpy(copy=True)
events = np.vstack(event_table["events"].to_numpy()).astype(np.float32)
```

iii. The AI's CONVERSION_NOTES state: "Published data passed experiment and container QC. ROI filtering excludes unions/duplicates, motion-border ROIs, dendrites, and ROIs that are too small, narrow, or dim; invalid ROIs are absent from the SDK valid-cell table."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (`start_time` from the SDK trials table). A 30 Hz grid starting at `start_time` is created for each trial, and neural events are interpolated onto this grid.

ii.
```python
grids = [_trial_grid(float(r.start_time), float(r.stop_time))
         for r in trials.itertuples()]
all_t = np.concatenate(grids)
neural_all = interp_rows_shared_time(ophys_t, events, all_t)
```

iii. The SDK's `start_time` is the synchronized trial start on the ophys clock. The grid extends to `stop_time`, capturing the full variable-length trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is resampled to exactly 30 Hz (33.333 ms time bins). All streams (neural, running, pupil, stimulus) are interpolated or sampled onto this common 30 Hz grid.

ii.
```python
FS = 30.0
DT = 1.0 / FS

def _trial_grid(start: float, stop: float) -> np.ndarray:
    n = int(np.ceil((stop - start) * FS - 1e-10))
    grid = start + np.arange(max(0, n), dtype=np.float64) * DT
    return grid[grid < stop]
```

Metadata:
```python
"time_bin_size": 1000.0 / FS,  # 33.333 ms
```

iii. The paper explicitly states neural and running traces were "linearly interpolated to a common 30 Hz time series." The native single-plane ophys rate is ~31 Hz, so 30 Hz resampling creates a uniform grid close to the native rate.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, specifically the `image_name`, `start_time`, `end_time`, `omitted`, and `is_change` columns from presentations in the `change_detection` stimulus block. Periods between presentations are labeled "gray."

ii.
```python
stim = exp.stimulus_presentations
task = stim[
    stim["stimulus_block_name"].str.contains("change_detection", na=False)
].sort_values("start_time")
for row in task.itertuples():
    start = float(row.start_time)
    end = float(row.end_time)
    ...
    if hi > lo and not bool(row.omitted):
        name = str(row.image_name)
        image_all[idx[valid]] = IMAGE_TO_INT[name]
```

iii. Using `stimulus_presentations` directly gives frame-accurate image timing, including explicit gray periods during inter-stimulus intervals. The AI notes: "User wording 'image identity (of the image presented during the non-grey screen)' requires explicit gray handling rather than carrying the last image through ISI."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes using a fixed global mapping of 17 classes (gray + 16 natural images). The mapping is hardcoded in `IMAGE_VALUES` and `IMAGE_TO_INT`. Image identity is 0 (gray) by default and set to the specific image code only during non-omitted presentation intervals.

ii.
```python
IMAGE_VALUES = [
    "gray", "im000", "im031", "im035", "im045", "im054", "im061",
    "im062", "im063", "im065", "im066", "im069", "im073", "im075",
    "im077", "im085", "im106",
]
IMAGE_TO_INT = {name: idx for idx, name in enumerate(IMAGE_VALUES)}

image_all = np.zeros(len(all_t), dtype=np.int16)  # default gray=0
```

iii. A hardcoded mapping was used rather than a data-derived one for determinism. The 17 classes cover all images across both image sets A and B plus gray.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same 30 Hz trial grid as the neural data. `np.searchsorted` maps stimulus presentation times to grid indices.

ii.
```python
lo = int(np.searchsorted(all_t, start, side="left"))
hi = int(np.searchsorted(all_t, end, side="left"))
if hi > lo and not bool(row.omitted):
    idx = np.arange(lo, hi)
    valid = (all_t[idx] >= start) & (all_t[idx] < end)
    image_all[idx[valid]] = IMAGE_TO_INT[name]
```

iii. Using the same concatenated time grid (`all_t`) for both neural and image data guarantees temporal alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` flag in the `stimulus_presentations` table, combined with presentation `start_time` and `end_time`.

ii.
```python
if bool(row.is_change):
    change_all[idx[valid]] = 1
```

iii. `is_change` is the SDK's precomputed flag indicating whether a stimulus presentation is a change event.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Image change is a binary time-varying variable. It is 1 throughout the 250 ms changed-image flash presentation (all 30 Hz samples within `[start_time, end_time)` of a presentation with `is_change == True`), and 0 otherwise.

ii.
```python
change_all = np.zeros(len(all_t), dtype=np.int16)
...
if bool(row.is_change):
    change_all[idx[valid]] = 1
```

iii. The AI initially tried a one-bin impulse but found it decoded below chance. The final implementation marks the entire 250 ms changed-image flash, which the AI describes as "a better operationalization of 'right after,' robust to sampling phase."

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 = no change, 1 = change). No thresholding is needed.

ii.
```python
["no_change", "change"]
```

iii. The binary nature of image change means no discretization is required.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity — computed on the same 30 Hz trial grid using `np.searchsorted` to map stimulus times to grid indices.

ii. Same code as 3-c.

iii. Same alignment mechanism as all other output variables.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `exp.running_speed`, which provides SDK-filtered speed (cm/s) with synchronized timestamps.

ii.
```python
running = exp.running_speed
run_all = finite_interp(running["timestamps"], running["speed"],
                        all_t, "running speed")
```

iii. The SDK's `running_speed` attribute applies the standard 10 Hz low-pass Butterworth filter and artifact rejection described in the whitepaper.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the 30 Hz trial grid using `finite_interp`, which first removes non-finite samples. Then it is discretized into 5 percentile-based bins using cohort-wide 20th/40th/60th/80th percentile edges computed from all retained sessions.

ii.
```python
def finite_interp(source_t, source_x, target_t, label: str):
    good = np.isfinite(source_t) & np.isfinite(source_x)
    t = source_t[good]
    x = source_x[good]
    order = np.argsort(t, kind="stable")
    t, x = t[order], x[order]
    unique = np.r_[True, np.diff(t) > 0]
    t, x = t[unique], x[unique]
    return np.interp(target_t, t, x).astype(np.float32)

run_edges = percentile_edges([s["running"] for s in raw_sessions], "running")
```

```python
def percentile_edges(values, label):
    joined = np.concatenate(values).astype(np.float32, copy=False)
    edges = np.quantile(joined, [0.2, 0.4, 0.6, 0.8]).astype(np.float64)
    return edges
```

iii. `finite_interp` handles NaN/infinite values by excluding them before interpolation, using `np.interp` which clamps to edge values outside the source range. Percentile edges are computed globally for cross-session consistency.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins (0-4) using 4 percentile edges (20th, 40th, 60th, 80th percentiles) computed across all sessions. `np.searchsorted` with `side="right"` maps values to bins.

ii.
```python
run_edges = percentile_edges([s["running"] for s in raw_sessions], "running")
run_bin = np.searchsorted(run_edges, s["running"], side="right").astype(np.int16)
```

iii. Equal percentile bins ensure approximately 20% of data per bin, which is important for balanced decoding.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same 30 Hz trial grid as neural data, so alignment is guaranteed by sharing the same time vector.

ii.
```python
run_all = finite_interp(running["timestamps"], running["speed"],
                        all_t, "running speed")
```

iii. All streams are interpolated to the same `all_t` grid, ensuring alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `exp.eye_tracking`, using `pupil_width` and `pupil_height` columns. The diameter is computed as `2 * max(pupil_width, pupil_height)`.

ii.
```python
eye = exp.eye_tracking
pupil_diameter = 2.0 * np.maximum(
    eye["pupil_width"].to_numpy(dtype=np.float64),
    eye["pupil_height"].to_numpy(dtype=np.float64),
)
```

iii. The AI's CONVERSION_NOTES state: "SDK/whitepaper define width and height as ellipse half-axes and diameter as the major full axis. Cleaned fields are NaN for likely blinks."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter (`2 * max(width, height)`) is linearly interpolated from its native timestamps to the 30 Hz trial grid using `finite_interp`, which excludes NaN/non-finite values (including blink frames where cleaned fields are NaN). Then it is discretized into 5 percentile-based bins.

ii.
```python
pupil_diameter = 2.0 * np.maximum(
    eye["pupil_width"].to_numpy(dtype=np.float64),
    eye["pupil_height"].to_numpy(dtype=np.float64),
)
pupil_all = finite_interp(eye["timestamps"], pupil_diameter,
                          all_t, "pupil diameter")

pupil_edges = percentile_edges([s["pupil"] for s in raw_sessions], "pupil")
```

iii. Blink frames produce NaN in cleaned pupil fields and are automatically excluded by `finite_interp`'s finite-check. Linear interpolation bridges over blink gaps.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed — 5 bins using 20th/40th/60th/80th percentile edges computed globally.

ii.
```python
pupil_edges = percentile_edges([s["pupil"] for s in raw_sessions], "pupil")
pupil_bin = np.searchsorted(pupil_edges, s["pupil"], side="right").astype(np.int16)
```

iii. Same percentile-based discretization as running speed for consistency.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — interpolated to the same 30 Hz trial grid.

ii.
```python
pupil_all = finite_interp(eye["timestamps"], pupil_diameter,
                          all_t, "pupil diameter")
```

iii. Same alignment mechanism as all other streams.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the SDK trials table.

ii.
```python
OUTCOME_COLUMNS = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_bool = trials[OUTCOME_COLUMNS].to_numpy(dtype=bool)
if not np.all(outcome_bool.sum(axis=1) == 1):
    bad = trials.index[outcome_bool.sum(axis=1) != 1].tolist()
    raise ValueError(f"{experiment_id}: nonexclusive outcomes in {bad[:5]}")
outcomes = outcome_bool.argmax(axis=1).astype(np.int16)
```

iii. The four outcome columns are mutually exclusive for eligible (go/catch, non-aborted, non-auto-rewarded) trials. The code validates this with an explicit check.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are encoded as integers 0-3 via `argmax` over the four boolean outcome columns. The static per-trial value is repeated across all time bins within the trial.

ii.
```python
outcomes = outcome_bool.argmax(axis=1).astype(np.int16)
...
np.full(int(length), s["outcome"][i], dtype=np.int16),
```

iii. Repeating the static outcome across time is required because the output format uses a single `(5, T)` matrix per trial and other outputs are time-varying.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: Sessions with empty eye tracking tables are excluded entirely (3 sessions excluded) and logged in metadata.
- **Non-finite running/pupil values**: `finite_interp` removes NaN/infinite samples before interpolation; `np.interp` clamps to edge values at boundaries.
- **Non-exclusive outcomes**: The code raises an error if outcome flags are not exactly one-hot.
- **Empty trial grids**: Raises an error if any eligible trial produces an empty time grid.
- **Failed experiments**: Caught by `MissingRequiredData` or generic exceptions; excluded sessions are logged.

ii.
```python
if eye is None or eye.empty:
    raise MissingRequiredData("eye tracking table is missing or empty")

def finite_interp(source_t, source_x, target_t, label):
    good = np.isfinite(source_t) & np.isfinite(source_x)
    if good.sum() < 2:
        raise ValueError(f"fewer than two finite {label} samples")
    ...

except MissingRequiredData as exc:
    exclusions.append({"experiment_id": int(eid), "reason": str(exc)})
```

iii. The AI chose to exclude sessions with missing data rather than fill with defaults, documenting each exclusion. This is more conservative than mapping NaN to a default bin value.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading and decompressing trace data from NWB files via `cache.get_behavior_ophys_experiment()`. The AI addressed this with 16-worker `ProcessPoolExecutor` parallelism, reducing full conversion from a projected >15 minutes to 73 seconds.

ii.
```python
workers = min(16, len(ids), max(1, (os.cpu_count() or 2) // 2))
with ProcessPoolExecutor(max_workers=workers) as pool:
    futures = {pool.submit(extract_session, eid): eid for eid in ids}
```

iii. The AI's CONVERSION_NOTES document the optimization: "Four-worker trace decompression projected beyond the 15-minute goal. Sixteen workers reduced final conversion to 73.32 s."

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-stimulus-presentation loop in `extract_session` iterates over each presentation in the task block to assign image identity and change labels. This could potentially be vectorized using interval-based array operations.

ii.
```python
for row in task.itertuples():
    start = float(row.start_time)
    end = float(row.end_time)
    lo = int(np.searchsorted(all_t, start, side="left"))
    hi = int(np.searchsorted(all_t, end, side="left"))
    if hi > lo and not bool(row.omitted):
        idx = np.arange(lo, hi)
        valid = (all_t[idx] >= start) & (all_t[idx] < end)
        ...
```

iii. The loop processes ~4800 stimulus presentations per session. While vectorizable, this is fast compared to data loading and the vectorized neural interpolation optimization had greater impact.

## 9-c. What processing does the code repeat multiple times?

i. Each parallel worker creates its own `VisualBehaviorOphysProjectCache` instance via `make_cache()`, since the cache object cannot be pickled across process boundaries. This repeats the cache initialization for each experiment.

ii.
```python
def extract_session(experiment_id: int) -> dict:
    cache = make_cache()
    exp = cache.get_behavior_ophys_experiment(int(experiment_id))
```

iii. Cache creation is fast (it reads a manifest JSON), so the repeated initialization is negligible compared to the experiment loading it enables in parallel.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores detailed per-session metadata (`session_info`) including counts, image set, experience level, and ophys frame rate that are not used by the downstream decoder. The processing plots (`--show-processing`) generate visualizations that are not needed for the final output.

ii.
```python
session_info.append({k: s[k] for k in [
    "experiment_id", "ophys_session_id", "mouse_id", "region",
    "session_type", "image_set", "experience_level", "source_trial_count",
    "eligible_trial_count", "go_count", "catch_count", "outcome_counts",
    "ophys_frame_rate",
]})
```

iii. The metadata is useful for documentation and debugging but adds minimal overhead. The resampling to 30 Hz could also be considered unnecessary since native ophys rate (~31 Hz) is close to 30 Hz and the downstream decoder can handle slight rate variations.
