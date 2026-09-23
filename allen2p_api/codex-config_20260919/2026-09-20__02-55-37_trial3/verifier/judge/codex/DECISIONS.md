# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `VisualBehaviorOphysProjectCache.from_s3_cache` to read the Allen cache, but it first enumerates cached experiment `.nwb` filenames to get a local subset of experiment IDs. It then filters the Allen experiment table to those cached IDs and to `behavior_type == 'active_behavior'`, and loads each retained experiment independently with `get_behavior_ophys_experiment()`. In full mode it parallelizes this over processes.

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

obj = cache.get_behavior_ophys_experiment(int(experiment_id))
```

iii. In `CONVERSION_NOTES.md`, the AI says the local cache is only a subset of the release, so enumerating cached IDs avoids fetching absent experiments from S3. It also justifies excluding passive experiments because the requested trial-outcome output is only meaningful for active behavior sessions.

## 1-b. How are the data split into subjects?

i. Subjects are unique retained `mouse_id` values, converted to strings and sorted globally after session conversion.

ii.
```python
subjects = sorted({s["mouse_id"] for s in sessions})
subject_map = {x: i for i, x in enumerate(subjects)}
...
"subject_idx": np.asarray([subject_map[s["mouse_id"]] for s in sessions], dtype=np.int64),
```

iii. The AI’s notes say `mouse_id` is the canonical Allen SDK animal identifier, so it uses that directly for subject identity.

## 1-c. How are the data split into sessions?

i. The AI treats each ophys experiment / imaging plane as one target session. It does not merge experiments sharing an `ophys_session_id`; instead it stores that ID only as metadata while keeping one converted session per experiment.

ii.
```python
for i, (experiment_id, meta) in enumerate(table.iterrows(), start=1):
    ...
    session, exclusions = convert_experiment(cache, int(experiment_id), meta, ...)
    ...
    if session is not None:
        sessions.append(session)
...
"experiment_id": int(experiment_id),
"ophys_session_id": int(meta.ophys_session_id),
```

iii. In `CONVERSION_NOTES.md`, the AI explicitly resolves the “session vs experiment” ambiguity by arguing that multiscope planes can differ in region and frame timing, so each imaging plane should be a separate decoding session and shared behavior can be repeated per plane.

## 1-d. How are the data split into trials?

i. Trials come from the Allen SDK `obj.trials` table. For each retained row, the trial window is the native `start_time` to `stop_time` interval, but it is represented as a variable-length sequence of 100 ms bins rather than native ophys frames.

ii.
```python
eligible, exclusions = valid_trials(obj.trials)
...
for trial_id, row in eligible.iterrows():
    start, stop = float(row.start_time), float(row.stop_time)
    nbins = int(np.floor((stop - start) / DT + 1e-9))
    ...
    edges = start + np.arange(nbins + 1, dtype=float) * DT
    centers = edges[:-1] + DT / 2.0
```

iii. The AI’s notes say trial boundaries should follow the Allen SDK trial definition, with variable trial duration preserved, while a common 100 ms temporal grid is used inside each trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only contingent go/catch trials that are not aborted, not auto-rewarded, have exactly one of the four outcome flags, have finite and positive time bounds, and have at least 2 full 100 ms bins. It additionally drops trials lacking running or pupil coverage at all bin centers and trials whose active stimulus table is inconsistent with go/catch expectations. Whole experiments are excluded if they end up with fewer than 2 retained trials or zero cells.

ii.
```python
contingent = trials["go"].fillna(False).astype(bool) | trials["catch"].fillna(False).astype(bool)
non_aborted = ~trials["aborted"].fillna(False).astype(bool)
non_auto = ~trials["auto_rewarded"].fillna(False).astype(bool)
one_outcome = flags.sum(axis=1).eq(1)
finite_bounds = np.isfinite(trials["start_time"]) & np.isfinite(trials["stop_time"])
positive = trials["stop_time"].to_numpy() > trials["start_time"].to_numpy()
mask = contingent & non_aborted & non_auto & one_outcome & finite_bounds & positive
...
if nbins < 2:
    dropped_short += 1
    continue
...
if run is None or pupil is None:
    dropped_coverage += 1
    continue
...
if (is_go and nchanges != 1) or (is_catch and nchanges != 0):
    dropped_stim_consistency += 1
    continue
...
if len(records) < 2 or len(cell_ids) == 0:
    return None, exclusions
```

iii. The AI justifies this in its notes as enforcing the requested go/catch-only contingent trials and avoiding “dishonest” imputation when synchronized behavioral coverage is missing.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from the Allen SDK `events` table, specifically the unfiltered `events` array for each cell, together with `ophys_timestamps`.

ii.
```python
timestamps = np.asarray(obj.ophys_timestamps, dtype=float)
events, cell_ids = stack_event_traces(obj.events, len(timestamps))

def stack_event_traces(events: pd.DataFrame, nframes: int) -> tuple[np.ndarray, np.ndarray]:
    cell_ids = events.index.to_numpy(dtype=np.int64)
    traces = np.vstack(events["events"].to_numpy()).astype(np.float32, copy=False)
```

iii. The AI’s notes say the paper used detected calcium events rather than dF/F to reduce slow GCaMP decay, so it chose unfiltered FastLZero events instead of fluorescence traces.

## 2-b. How is the `neural` data processed?

i. The AI stacks all cell event traces for an experiment into a cell-by-frame matrix, then rebins them into non-overlapping 100 ms trial-local bins by summing event magnitudes between timestamp bin edges. The final per-trial matrices are stored as `float32`.

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
...
neural = bin_events(events, timestamps, edges)
```

iii. In the notes, the AI argues that event magnitudes better match the paper and that 100 ms bins provide one common time resolution across single-plane and multiscope recordings.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not apply an additional downstream neuron-quality filter. It trusts the Allen SDK’s valid ROI / event rows, while still rejecting experiments with zero cells and raising errors if event arrays are malformed or non-finite.

ii.
```python
if len(cell_ids) == 0:
    return np.empty((0, nframes), np.float32), cell_ids
...
if traces.shape != (len(cell_ids), nframes):
    raise ValueError(...)
if not np.isfinite(traces).all():
    raise ValueError("non-finite inferred event value")
...
if len(records) < 2 or len(cell_ids) == 0:
    return None, exclusions
```

iii. The AI’s notes say Allen’s upstream pipeline has already removed invalid ROIs, so no extra SNR/activity filter should be imposed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned on the Allen synchronized ophys clock within each trial, using bins that start at the trial `start_time`. The effective trial alignment event is trial start, with neural samples assigned by timestamp into bins covering the trial interval.

ii.
```python
start, stop = float(row.start_time), float(row.stop_time)
edges = start + np.arange(nbins + 1, dtype=float) * DT
centers = edges[:-1] + DT / 2.0
neural = bin_events(events, timestamps, edges)
...
"temporal_alignment_event": "Variable-length trial start on the common synchronized clock; bins follow ophys timestamps.",
```

iii. The AI justifies this by citing the whitepaper’s shared synchronization clock and the need to keep all modalities on one timestamp-based axis.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 100 ms bin size for every session and trial. Yes: the native neural event stream is rebinned by summing within 100 ms bins.

ii.
```python
DT = 0.100  # seconds
...
nbins = int(np.floor((stop - start) / DT + 1e-9))
edges = start + np.arange(nbins + 1, dtype=float) * DT
...
"time_bin_size": DT * 1000.0,
"neural_signal": "AllenSDK unfiltered FastLZero inferred event magnitudes summed in 100 ms bins",
```

iii. The AI’s notes say 100 ms is close to the slower multiscope sampling interval and creates one uniform decoder timebase across rigs.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the active-task `stimulus_presentations` table, using `image_name` along with `start_time`, `end_time`, and `omitted`.

ii.
```python
stim = active_stimulus_table(obj.stimulus_presentations)
...
starts = stim["start_time"].to_numpy(float)
ends = stim["end_time"].to_numpy(float)
image_name = stim["image_name"].fillna("").astype(str).to_numpy()
omitted = stim["omitted"].fillna(False).astype(bool).to_numpy() if "omitted" in stim else np.zeros(len(stim), bool)
```

iii. The AI’s notes say this better tracks the image actually on screen, including gray intervals and omissions, than trial-level initial/change image fields alone.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each 100 ms bin center, the AI labels the bin with the active image name only if the bin falls inside a non-omitted stimulus presentation interval; otherwise it assigns `"gray"`. It then builds one global categorical mapping with `"gray"` first and all other image names sorted.

ii.
```python
labels = np.full(len(centers), "gray", dtype=object)
...
shown = valid_idx & (centers < ends[rows]) & (~omitted[rows]) & (image_name[rows] != "") & (image_name[rows] != "omitted")
labels[shown] = image_name[rows[shown]]
...
image_values = ["gray"] + sorted({str(x) for s in sessions for r in s["records"] for x in r["image_labels"] if x != "gray"})
image_map = {x: i for i, x in enumerate(image_values)}
```

iii. The AI justifies this as matching the requested “image presented during the non-grey screen” rather than assuming the same image label fills the entire trial.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same per-trial 100 ms bin centers used for neural event binning, so each image label aligns one-to-one with a neural time bin.

ii.
```python
edges = start + np.arange(nbins + 1, dtype=float) * DT
centers = edges[:-1] + DT / 2.0
image_labels, change, nchanges = label_stimuli(trial_stim, centers, edges)
...
image = np.fromiter((image_map[str(x)] for x in r["image_labels"]), dtype=np.int16, count=T)
```

iii. The AI’s notes say all modalities are queried against the same synchronized bin centers, so no separate alignment step is needed once `centers` are defined.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the active-task `stimulus_presentations.is_change` flag and the corresponding presentation intervals. The go/catch trial flags are then used only as consistency checks.

ii.
```python
is_change = stim["is_change"].fillna(False).astype(bool).to_numpy()
change_rows = np.flatnonzero(is_change & (starts < edges[-1]) & (ends > edges[0]))
...
is_go, is_catch = bool(row.go), bool(row.catch)
if (is_go and nchanges != 1) or (is_catch and nchanges != 0):
    dropped_stim_consistency += 1
    continue
```

iii. In the notes, the AI says `is_change` in the stimulus table is display-lag corrected and directly marks the changed image presentation, so it is preferable to trial `change_time` for labeling the stimulus stream.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI creates a binary vector initialized to 0 and sets it to 1 for every 100 ms bin center that falls within the full on-screen interval of the changed image presentation. It does not mark catch trials as changed, and it drops stimulus-trial combinations that violate go/catch expectations.

ii.
```python
changes = np.zeros(len(centers), dtype=np.int16)
...
for row in change_rows:
    changes[(centers >= starts[row]) & (centers < ends[row])] = 1
...
if (is_go and nchanges != 1) or (is_catch and nchanges != 0):
    dropped_stim_consistency += 1
    continue
```

iii. The AI’s notes say it originally tried a one-bin impulse, found that it decoded poorly, and then changed to the full changed-image interval because that is what `is_change` semantically labels and better matches “right after” a change.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is a binary categorical output: `0 = no_change`, `1 = change`.

ii.
```python
changes = np.zeros(len(centers), dtype=np.int16)
...
"output_values": [
    image_values,
    ["no_change", "change"],
    ["Q1_slowest", "Q2", "Q3", "Q4", "Q5_fastest"],
```

iii. No extra thresholding is applied beyond the binary change/no-change labeling.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The change label is computed at the same 100 ms bin centers and with the same trial edges as the neural event data.

ii.
```python
edges = start + np.arange(nbins + 1, dtype=float) * DT
centers = edges[:-1] + DT / 2.0
image_labels, change, nchanges = label_stimuli(trial_stim, centers, edges)
...
output = np.vstack([
    image,
    r["change"].astype(np.int16, copy=False),
```

iii. The AI’s justification is the same common-clock argument used for image identity and other outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `obj.running_speed`, using its `timestamps` and `speed` columns.

ii.
```python
running = obj.running_speed
...
run = interpolate_finite(running["timestamps"].to_numpy(), running["speed"].to_numpy(), centers)
```

iii. The AI’s notes say the SDK `running_speed` stream is already the processed locomotion signal, so it uses that directly.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI linearly interpolates running speed to the 100 ms trial bin centers, pools all retained running samples across sessions/trials, computes global quintile thresholds, and later discretizes each trial’s aligned running values with those pooled edges.

ii.
```python
run = interpolate_finite(running["timestamps"].to_numpy(), running["speed"].to_numpy(), centers)
...
def percentile_edges(values: list[np.ndarray]) -> np.ndarray:
    joined = np.concatenate(values).astype(np.float64, copy=False)
    edges = np.quantile(joined[np.isfinite(joined)], [0.2, 0.4, 0.6, 0.8])
    return edges
...
run_edges = percentile_edges([r["running"] for s in sessions for r in s["records"]])
...
discretize(r["running"], run_edges)
```

iii. The AI justifies this as using the SDK’s processed signal while enforcing one common decoder timebase and one dataset-wide categorical definition.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five percentile bins using pooled 20th/40th/60th/80th percentile edges across all retained aligned samples. Category assignment uses `np.searchsorted(..., side="right")`, yielding integer classes 0 through 4.

ii.
```python
edges = np.quantile(joined[np.isfinite(joined)], [0.2, 0.4, 0.6, 0.8])
...
def discretize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, values, side="right").astype(np.int16)
...
["Q1_slowest", "Q2", "Q3", "Q4", "Q5_fastest"],
```

iii. The AI’s notes describe this as pooled quintiles so the class definitions are consistent across sessions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated directly onto the same 100 ms bin centers used for the neural bins inside each trial.

ii.
```python
centers = edges[:-1] + DT / 2.0
run = interpolate_finite(running["timestamps"].to_numpy(), running["speed"].to_numpy(), centers)
...
output = np.vstack([
    image,
    r["change"].astype(np.int16, copy=False),
    discretize(r["running"], run_edges),
```

iii. The AI’s notes explicitly treat `centers` as the shared alignment axis for all modalities.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `obj.eye_tracking`, using the processed `pupil_area` signal and its timestamps rather than `pupil_width`.

ii.
```python
eye = obj.eye_tracking
...
if eye is None or len(eye) == 0 or "pupil_area" not in eye:
    ...
pupil_area = eye["pupil_area"].to_numpy(float)
```

iii. The AI’s notes say the task asks for diameter, while the SDK’s processed area is already blink/outlier masked, so it converts area to an equivalent circular diameter.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI converts processed pupil area to equivalent diameter with `2*sqrt(area/pi)`, interpolates finite samples to 100 ms bin centers without extrapolating beyond valid coverage, uses pooled global quintile edges for discretization, and drops trials or experiments when pupil coverage is insufficient.

ii.
```python
pupil_area = eye["pupil_area"].to_numpy(float)
pupil_diameter = 2.0 * np.sqrt(np.maximum(pupil_area, 0.0) / np.pi)
...
pupil = interpolate_finite(eye["timestamps"].to_numpy(), pupil_diameter, centers)
if run is None or pupil is None:
    dropped_coverage += 1
    continue
...
pupil_edges = percentile_edges([r["pupil"] for s in sessions for r in s["records"]])
```

iii. The AI’s notes justify this as preserving the processed pupil metric, respecting masked bad samples, and avoiding extrapolation outside the actual eye-tracking support.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five pooled percentile bins using the same `percentile_edges` / `discretize` scheme as running speed.

ii.
```python
pupil_edges = percentile_edges([r["pupil"] for s in sessions for r in s["records"]])
...
def discretize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, values, side="right").astype(np.int16)
...
["Q1_smallest", "Q2", "Q3", "Q4", "Q5_largest"],
```

iii. The AI’s notes describe these as global quintiles computed after alignment and trial filtering.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated onto the same 100 ms trial bin centers used for neural data and all other time-varying outputs.

ii.
```python
centers = edges[:-1] + DT / 2.0
pupil = interpolate_finite(eye["timestamps"].to_numpy(), pupil_diameter, centers)
...
output = np.vstack([
    image,
    r["change"].astype(np.int16, copy=False),
    discretize(r["running"], run_edges),
    discretize(r["pupil"], pupil_edges),
```

iii. The AI’s notes use the same shared-bin-center justification as for running speed.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
OUTCOME_COLUMNS = ["hit", "miss", "false_alarm", "correct_reject"]

def outcome_code(row: pd.Series) -> int:
    flags = np.asarray([bool(row[x]) for x in OUTCOME_COLUMNS])
```

iii. The AI treats these four Allen SDK fields as the canonical contingent-trial outcome definition.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI requires exactly one of the four outcome flags to be true, maps the true flag’s index to an integer 0 to 3, and repeats that scalar outcome across every time bin of the trial.

ii.
```python
def outcome_code(row: pd.Series) -> int:
    flags = np.asarray([bool(row[x]) for x in OUTCOME_COLUMNS])
    if flags.sum() != 1:
        raise ValueError("trial does not have exactly one outcome")
    return int(np.flatnonzero(flags)[0])
...
outcome = np.full(T, r["outcome"], dtype=np.int16)
```

iii. The AI’s notes say a constant time series is the simplest way to include the static per-trial label in the shared `(n_output, T)` output matrix.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly handles bad or missing data by exclusion rather than imputation. It raises errors for malformed/non-finite neural event arrays, excludes experiments with no processed pupil stream, drops short trials, drops trials with incomplete running/pupil coverage at the requested bin centers, drops stimulus-trial inconsistencies, and excludes experiments with fewer than 2 retained trials or no cells. It also catches experiment-level exceptions so one failure does not stop the full conversion.

ii.
```python
if not np.isfinite(traces).all():
    raise ValueError("non-finite inferred event value")
...
if eye is None or len(eye) == 0 or "pupil_area" not in eye:
    exclusions["no_eye_session"] = len(eligible)
    return None, exclusions
...
if nbins < 2:
    dropped_short += 1
    continue
...
if run is None or pupil is None:
    dropped_coverage += 1
    continue
...
if (is_go and nchanges != 1) or (is_catch and nchanges != 0):
    dropped_stim_consistency += 1
    continue
...
except Exception as exc:
    print(f"  experiment {experiment_id}: ERROR {exc!r}", flush=True)
    return experiment_id, None, {"fatal_error": repr(exc)}
```

iii. The AI’s notes say it wants to avoid fabricating labels outside genuine synchronized coverage, while still allowing internal interpolation across finite samples and logging all exclusions for auditability.

## 9-a. What are the most time-consuming steps of the code?

i. The AI identifies full Allen SDK experiment loading as the main bottleneck, with full-session event stacking and trial-level binning/interpolation as the next most expensive work.

ii.
```python
obj = cache.get_behavior_ophys_experiment(int(experiment_id))
timestamps = np.asarray(obj.ophys_timestamps, dtype=float)
events, cell_ids = stack_event_traces(obj.events, len(timestamps))
...
for trial_id, row in eligible.iterrows():
    ...
    neural = bin_events(events, timestamps, edges)
```

iii. In `CONVERSION_NOTES.md`, the AI says SDK object construction and loading of large arrays dominate runtime, which is why it adds optional process-level parallelism.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining Python loops are the per-trial loop in `convert_experiment`, the per-session/per-trial assembly loop in `assemble`, and the loop over changed stimulus rows in `label_stimuli`. The AI already vectorized the heaviest inner numeric work (`searchsorted`, interpolation, cumulative-sum event binning), so it considered data loading rather than these loops the primary performance issue.

ii.
```python
for trial_id, row in eligible.iterrows():
    ...
for row in change_rows:
    changes[(centers >= starts[row]) & (centers < ends[row])] = 1
...
for s in sessions:
    sn, si, so = [], [], []
    for r in s["records"]:
```

iii. The notes explicitly mention replacing neuron/bin Python loops with vectorized cumulative-sum binning and say remaining speed limits are mostly in SDK loading.

## 9-c. What processing does the code repeat multiple times?

i. The AI code repeats some work across trial iterations and across conversion stages. Inside each trial loop it repeatedly converts running and eye timestamps to NumPy arrays and repeatedly slices the stimulus table for the current trial. Later, after storing continuous aligned running/pupil traces per trial, it makes a second pass in `assemble()` to discretize them and remap image labels into integer codes.

ii.
```python
for trial_id, row in eligible.iterrows():
    ...
    run = interpolate_finite(running["timestamps"].to_numpy(), running["speed"].to_numpy(), centers)
    pupil = interpolate_finite(eye["timestamps"].to_numpy(), pupil_diameter, centers)
    ...
    trial_stim = stim[(stim["start_time"] < edges[-1]) & (stim["end_time"] > edges[0])]
...
for s in sessions:
    ...
    image = np.fromiter((image_map[str(x)] for x in r["image_labels"]), dtype=np.int16, count=T)
    ...
    discretize(r["running"], run_edges),
    discretize(r["pupil"], pupil_edges),
```

iii. The notes frame this as an acceptable tradeoff for clarity, validation, and metadata/audit bookkeeping rather than as the dominant runtime cost.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does a small amount of extra bookkeeping and optional diagnostics that are not needed by the downstream decoder. In particular, it creates an unused local `trial_ids` list, can retain diagnostic arrays for plotting when `--show-processing` is enabled, and stores extensive `session_info` / exclusion metadata that the decoder does not consume.

ii.
```python
trial_ids = []
...
trial_ids.append(int(trial_id))
...
"diagnostic": {
    "ophys_timestamps": timestamps,
    "event_mean": events.mean(axis=0, dtype=np.float64).astype(np.float32),
    "running_times": running["timestamps"].to_numpy(float),
    "running_speed": running["speed"].to_numpy(float),
    "eye_times": eye["timestamps"].to_numpy(float),
    "pupil_diameter": pupil_diameter,
} if keep_diagnostic else None,
...
"session_info": session_info,
```

iii. The AI’s notes justify these extras as audit/debug support rather than as part of the decoder input-output representation itself.
