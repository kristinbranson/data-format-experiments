# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `VisualBehaviorOphysProjectCache.from_s3_cache()` to access the Allen SDK cache. It calls `get_ophys_experiment_table()` to get metadata, then filters to only locally cached experiments using filename enumeration (`cached_experiment_ids()`). It further filters to `behavior_type == 'active_behavior'` via the `active_experiment_table()` function. Each experiment is loaded individually using `cache.get_behavior_ophys_experiment()`. This differs from the reference, which filters by `project_code == 'VisualBehavior'` instead.

ii.
```python
def cached_experiment_ids() -> list[int]:
    prefix = "behavior_ophys_experiment_"
    return sorted(int(p.stem.removeprefix(prefix)) for p in EXPERIMENT_DIR.glob(f"{prefix}*.nwb"))

def active_experiment_table(cache) -> pd.DataFrame:
    table = cache.get_ophys_experiment_table()
    ids = [x for x in cached_experiment_ids() if x in table.index]
    selected = table.loc[ids]
    selected = selected[selected["behavior_type"].eq("active_behavior")]
    return selected.sort_index()
```

iii. The AI justified filtering to `active_behavior` because passive sessions lack behavioral trial outcomes (hit/miss/FA/CR) required by the decoder task. It also used filename enumeration to restrict to cached experiments before loading. The AI's CONVERSION_NOTES document this rationale under Step 4.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values from the experiment table. A sorted list of unique mouse IDs is created.

ii.
```python
subjects = sorted({s["mouse_id"] for s in sessions})
subject_map = {x: i for i, x in enumerate(subjects)}
```

iii. The AI used `mouse_id` as the canonical subject identifier, which is the SDK's unique identifier for each animal.

## 1-c. How are the data split into sessions?

i. Each ophys experiment (imaging plane) is treated as a separate session. This differs from the reference, which groups experiments by `ophys_session_id` (combining multiple imaging planes from the same recording session). The AI processes one experiment at a time via `convert_experiment()`.

ii.
```python
for i, (experiment_id, meta) in enumerate(table.iterrows(), start=1):
    session, exclusions = convert_experiment(
        cache, int(experiment_id), meta, ...)
    if session is not None:
        sessions.append(session)
```

iii. The AI justified this by noting that the paper's unit of decoding was per imaging plane, and that a neural matrix cannot combine different frame clocks/regions/planes. Each target session is one ophys experiment.

## 1-d. How are the data split into trials?

i. Trials are defined using the SDK's `trials` table. For each eligible trial, the window from `start_time` to `stop_time` is divided into fixed 100ms bins. The number of bins is `floor((stop - start) / 0.1)`, giving variable-length trials.

ii.
```python
start, stop = float(row.start_time), float(row.stop_time)
nbins = int(np.floor((stop - start) / DT + 1e-9))
if nbins < 2:
    dropped_short += 1
    continue
edges = start + np.arange(nbins + 1, dtype=float) * DT
centers = edges[:-1] + DT / 2.0
```

iii. The AI chose 100ms bins to normalize across different acquisition rates (31Hz single-plane, 11Hz multiscope) while retaining multiple bins per 250ms image flash.

## 1-e. How are trials filtered based on quality controls?

i. Trials must satisfy: (1) `go` or `catch`, (2) not `aborted`, (3) not `auto_rewarded`, (4) exactly one of the four outcome flags is true, (5) finite and positive start/stop times. Additionally, trials are dropped if running or pupil interpolation fails (coverage), if the trial is too short (<2 bins), or if go/catch stimulus consistency checks fail. Sessions with <2 retained trials or 0 cells are excluded. Three experiments lacking pupil data were excluded entirely.

ii.
```python
contingent = trials["go"].fillna(False).astype(bool) | trials["catch"].fillna(False).astype(bool)
non_aborted = ~trials["aborted"].fillna(False).astype(bool)
non_auto = ~trials["auto_rewarded"].fillna(False).astype(bool)
one_outcome = flags.sum(axis=1).eq(1)
finite_bounds = np.isfinite(trials["start_time"]) & np.isfinite(trials["stop_time"])
positive = trials["stop_time"].to_numpy() > trials["start_time"].to_numpy()
mask = contingent & non_aborted & non_auto & one_outcome & finite_bounds & positive
```

iii. The AI added more quality checks than the reference (which only required `~aborted & ~auto_rewarded & change_time.notna()`). The AI justified the additional checks as ensuring data integrity: exactly one outcome prevents ambiguous trials, finite bounds prevent invalid time windows, and coverage checks ensure all modalities have data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses `events` (SDK-provided FastLZero inferred calcium events) accessed via `obj.events`, not `dff_traces` (dF/F). This is a major difference from the reference which uses `dff_traces.dff`.

ii.
```python
events, cell_ids = stack_event_traces(obj.events, len(timestamps))

def stack_event_traces(events: pd.DataFrame, nframes: int) -> tuple[np.ndarray, np.ndarray]:
    cell_ids = events.index.to_numpy(dtype=np.int64)
    traces = np.vstack(events["events"].to_numpy()).astype(np.float32, copy=False)
    return traces, cell_ids
```

iii. The AI justified using events by citing the paper: "all neural analyses used detected calcium events to remove slow GCaMP decay." The AI chose unfiltered `events` over `filtered_events` because the latter is documented as visualization smoothing only.

## 2-b. How is the `neural` data processed?

i. Event traces are stacked from the events table, then summed within 100ms bins using a vectorized cumulative-sum approach. The reference simply stacks dF/F traces at native frame rate with no additional processing.

ii.
```python
def bin_events(events: np.ndarray, timestamps: np.ndarray, edges: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(timestamps, edges, side="left")
    lo, hi = int(indices[0]), int(indices[-1])
    local = events[:, lo:hi]
    cumulative = np.empty((events.shape[0], local.shape[1] + 1), dtype=np.float32)
    cumulative[:, 0] = 0.0
    np.cumsum(local, axis=1, dtype=np.float32, out=cumulative[:, 1:])
    rel = indices - lo
    return (cumulative[:, rel[1:]] - cumulative[:, rel[:-1]]).astype(np.float32, copy=False)
```

iii. The AI noted that summing event magnitudes within bins preserves discrete event signal and approximates the slowest frame period.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied. The AI relies on the SDK's `cell_specimen_table`/events rows, which already contain only valid ROIs. The AI explicitly chose not to filter on activity/SNR.

ii. No filtering code beyond what the SDK provides.

iii. The AI noted: "Do not impose a new downstream cell filter absent from the reference analysis" and "Neither paper nor decoder specification calls for it."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial start time (`start_time` from the trials table). The 100ms bins start at `start_time` and extend to `stop_time`. This gives variable-length trials.

ii.
```python
start, stop = float(row.start_time), float(row.stop_time)
nbins = int(np.floor((stop - start) / DT + 1e-9))
edges = start + np.arange(nbins + 1, dtype=float) * DT
neural = bin_events(events, timestamps, edges)
```

iii. The AI set `temporal_alignment_event` to "Variable-length trial start on the common synchronized clock; bins follow ophys timestamps" and `off_start`/`off_end` to `None` because alignment is to absolute timestamps rather than a fixed event window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses a fixed 100ms bin size (`DT = 0.100` seconds) for all experiments. Events are summed within each bin. This differs from the reference which uses the native ophys frame rate (~31Hz for single-plane, ~91ms per frame) with no rebinning.

ii.
```python
DT = 0.100  # seconds
...
"time_bin_size": DT * 1000.0,  # 100.0 ms
```

iii. The AI justified 100ms bins because: (1) it normalizes across different acquisition rates (31Hz single-plane vs 11Hz multiscope), (2) it is close to the slower multiscope frame period, (3) it retains multiple bins per 250ms image flash and the 400ms decoding window used in the paper.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, using `image_name`, `start_time`, `end_time`, `omitted`, and `active` columns. This differs from the reference which uses `initial_image_name` and `change_image_name` from the trials table.

ii.
```python
stim = active_stimulus_table(obj.stimulus_presentations)
...
image_labels, change, nchanges = label_stimuli(trial_stim, centers, edges)

def label_stimuli(stim, centers, edges):
    labels = np.full(len(centers), "gray", dtype=object)
    ...
    shown = valid_idx & (centers < ends[rows]) & (~omitted[rows]) & (image_name[rows] != "") & (image_name[rows] != "omitted")
    labels[shown] = image_name[rows[shown]]
```

iii. The AI used the stimulus_presentations table because it provides precise timing of when each image is actually on screen (250ms on, 500ms gray). This captures inter-stimulus gray periods as "gray" rather than attributing the pre-change or post-change image to the entire trial segment.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. At each 100ms bin center, the AI looks up which stimulus presentation is active. If a non-omitted image is being shown, that image name is used; otherwise "gray" is assigned. A global vocabulary is built: `["gray"] + sorted(unique image names)`, and names are mapped to integer codes.

ii.
```python
image_values = ["gray"] + sorted({str(x) for s in sessions for r in s["records"]
                                    for x in r["image_labels"] if x != "gray"})
image_map = {x: i for i, x in enumerate(image_values)}
image = np.fromiter((image_map[str(x)] for x in r["image_labels"]), dtype=np.int16, count=T)
```

iii. The AI justified including "gray" because inter-stimulus periods and omitted presentations genuinely show a gray screen, and the decoder should know the actual visual stimulus at each moment.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is determined at each 100ms bin center using the stimulus_presentations table. The same bin centers define both the neural data bins and the image label queries, ensuring alignment.

ii.
```python
centers = edges[:-1] + DT / 2.0
...
image_labels, change, nchanges = label_stimuli(trial_stim, centers, edges)
```

iii. Same temporal grid as neural data ensures alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the `stimulus_presentations` table, combined with stimulus interval timing (`start_time`, `end_time`). This differs from the reference which uses `change_time` from the trials table and `go` flag.

ii.
```python
is_change = stim["is_change"].fillna(False).astype(bool).to_numpy()
change_rows = np.flatnonzero(is_change & (starts < edges[-1]) & (ends > edges[0]))
for row in change_rows:
    changes[(centers >= starts[row]) & (centers < ends[row])] = 1
```

iii. The AI used `is_change` from the stimulus_presentations table because it provides the display-lag-corrected timing of the actual changed image presentation.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The image change flag is set to 1 at bin centers that fall within the on-screen interval of a stimulus presentation marked with `is_change=True`. This spans approximately 250ms (the image presentation duration). The reference uses a 750ms window (one flash + gray). Additionally, the AI asserts that each go trial has exactly 1 change presentation and each catch trial has 0.

ii.
```python
for row in change_rows:
    changes[(centers >= starts[row]) & (centers < ends[row])] = 1
...
if (is_go and nchanges != 1) or (is_catch and nchanges != 0):
    dropped_stim_consistency += 1
    continue
```

iii. The AI reasoned that `is_change` labels the newly shown image presentation interval, so the change signal spans only the ~250ms on-screen period. The consistency check ensures data integrity.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is a binary variable (0 or 1) with no thresholding needed. Value 1 marks the change presentation interval; 0 is used otherwise.

ii. See 4-b code.

iii. Binary by definition per the instructions.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Aligned using the same 100ms bin centers as neural data and image identity.

ii. See 4-b code.

iii. Same temporal grid ensures alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `obj.running_speed`, which provides SDK-filtered speed (cm/s) and timestamps from the running wheel encoder.

ii.
```python
running = obj.running_speed
run = interpolate_finite(running["timestamps"].to_numpy(), running["speed"].to_numpy(), centers)
```

iii. The SDK's `running_speed` property provides the processed signal (unwrapped, artifact-removed, low-pass filtered).

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the 100ms bin centers using `np.interp`. Non-finite values are excluded before interpolation. Then discretized into 5 quintile bins using pooled percentile thresholds across all retained sessions.

ii.
```python
def interpolate_finite(times, values, query):
    good = np.isfinite(times) & np.isfinite(values)
    times, values = times[good], values[good]
    return np.interp(query, times, values).astype(np.float32)

run_edges = percentile_edges([r["running"] for s in sessions for r in s["records"]])

def percentile_edges(values):
    joined = np.concatenate(values).astype(np.float64, copy=False)
    edges = np.quantile(joined[np.isfinite(joined)], [0.2, 0.4, 0.6, 0.8])
    return edges
```

iii. Linear interpolation preserves signal shape. Percentile-based binning ensures equal class counts.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins (0-4) using the 20th, 40th, 60th, and 80th percentile thresholds computed from all retained data. `np.searchsorted(edges, values, side='right')` maps values to bin indices.

ii.
```python
def discretize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, values, side="right").astype(np.int16)
```

iii. This is mathematically equivalent to the reference's `np.digitize(values, edges[1:-1])` approach, producing 5 equal-percentile bins.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same 100ms bin centers used for neural data binning, ensuring alignment.

ii. See 5-b code.

iii. Same temporal grid as neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `eye_tracking['pupil_area']`, converted to equivalent circular diameter via `2 * sqrt(pupil_area / pi)`. This differs from the reference which uses `pupil_width` directly.

ii.
```python
eye = obj.eye_tracking
pupil_area = eye["pupil_area"].to_numpy(float)
pupil_diameter = 2.0 * np.sqrt(np.maximum(pupil_area, 0.0) / np.pi)
```

iii. The AI reasoned that converting area to equivalent diameter preserves ellipse size while the SDK documentation describes the pupil as an ellipse fit. The whitepaper documents ellipse half-axes/area.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Processed `pupil_area` (already NaN during blinks/outliers via SDK processing) is converted to equivalent diameter, then linearly interpolated to 100ms bin centers. Non-finite values are excluded before interpolation. Trials where pupil interpolation fails are dropped. Three experiments lacking `pupil_area` data were excluded entirely. Then discretized into 5 quintile bins.

ii.
```python
pupil = interpolate_finite(eye["timestamps"].to_numpy(), pupil_diameter, centers)
if pupil is None:
    dropped_coverage += 1
    continue
```

iii. The AI noted that processed pupil values are already NaN during blink/outlier frames (masked upstream by SDK), so explicit blink removal was not needed. The reference explicitly filters `likely_blink` before interpolation using `pupil_width`.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 quintile bins using pooled percentile thresholds.

ii. Same `percentile_edges` and `discretize` functions as running speed.

iii. Equal-percentile binning for balanced decoding.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the same 100ms bin centers as neural data.

ii. See 6-b code.

iii. Same temporal grid as neural data.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the four boolean outcome columns: `hit`, `miss`, `false_alarm`, `correct_reject` in the trials table. Exactly one must be True.

ii.
```python
OUTCOME_COLUMNS = ["hit", "miss", "false_alarm", "correct_reject"]

def outcome_code(row: pd.Series) -> int:
    flags = np.asarray([bool(row[x]) for x in OUTCOME_COLUMNS])
    if flags.sum() != 1:
        raise ValueError("trial does not have exactly one outcome")
    return int(np.flatnonzero(flags)[0])
```

iii. These four columns are the SDK's canonical trial outcomes for the change detection task. The AI requires exactly one to be True, which is more strict than the reference's "first match" approach.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome is mapped to an integer code (0-3) based on position in `OUTCOME_COLUMNS`. The integer code is constant across all time bins within a trial (repeated as a full time-series).

ii.
```python
code = outcome_code(row)
outcome = np.full(T, r["outcome"], dtype=np.int16)
output = np.vstack([..., outcome, ...])
```

iii. Static per-trial but represented as time-varying so all outputs coexist in one `(5, T)` array.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Missing pupil data**: 3 experiments without processed pupil data are excluded entirely. Trials where running or pupil interpolation fails (insufficient coverage) are dropped.
- **Short trials**: Trials with <2 bins are dropped.
- **Inconsistent stimulus**: Go trials without exactly 1 change presentation, or catch trials with any change, are dropped.
- **Failed experiments**: Exceptions during conversion cause the experiment to be skipped with an error log.
- **Non-finite values**: Non-finite timestamps/values are excluded from interpolation.
- **Missing outcomes**: Trials without exactly one outcome flag are excluded by the valid_trials filter.

ii.
```python
if eye is None or len(eye) == 0 or "pupil_area" not in eye:
    return None, exclusions
if run is None or pupil is None:
    dropped_coverage += 1
    continue
if (is_go and nchanges != 1) or (is_catch and nchanges != 0):
    dropped_stim_consistency += 1
    continue
```

iii. The AI took a more conservative approach than the reference: rather than mapping NaN to bin 0, it excludes trials/sessions with insufficient data. All exclusion counts are tracked in metadata.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each experiment via `cache.get_behavior_ophys_experiment()`, which reads NWB files from the cache. The AI used `ProcessPoolExecutor` with up to 4 workers for parallel processing.

ii.
```python
use_parallel = not args.sample and not args.show_processing and len(table) > 2
if use_parallel:
    workers = min(4, len(table))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (experiment_id, session, exclusions) in enumerate(pool.map(convert_experiment_worker, items)):
            ...
```

iii. Each experiment contains full-session neural traces plus behavioral data. The AI estimated ~5.3s per experiment sequentially, ~6-8 min total with 4 workers. Full conversion completed in 348s (5.8 min).

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_experiment()` iterates sequentially over eligible trials. The `label_stimuli` function has a Python loop over change_rows. The image-label-to-integer conversion uses a generator expression rather than a vectorized lookup.

ii.
```python
for trial_id, row in eligible.iterrows():
    ...
for row in change_rows:
    changes[(centers >= starts[row]) & (centers < ends[row])] = 1
```

iii. These loops are not bottlenecks compared to SDK data loading. The neural binning is already vectorized with cumulative sums.

## 9-c. What processing does the code repeat multiple times?

i. No significant processing is repeated. Each experiment is loaded once. Percentile edges and image vocabularies are computed once from collected data. The parallel worker approach does create independent SDK cache instances per worker, but this is necessary for process isolation.

ii. N/A

iii. The AI designed a two-pass approach: first collect all trial data, then compute global statistics and assemble. No redundant computation was identified.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI stores extensive metadata including per-session `session_info` with experiment IDs, cell specimen IDs, trial IDs, start/stop times, and exclusion counts. The diagnostic data (ophys_timestamps, event_mean, running/pupil raw samples) is optionally retained for plotting but discarded in non-diagnostic mode. The AI also performs stimulus consistency checks (asserting go trials have exactly 1 change) which add robustness but are not strictly required.

ii.
```python
"diagnostic": { ... } if keep_diagnostic else None,
session_info.append({...})
```

iii. The extra metadata supports reproducibility and debugging but is not used by the decoder.
