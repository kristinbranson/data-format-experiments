# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script creates an AllenSDK `VisualBehaviorOphysProjectCache` rooted at `/app/data`, reads the ophys experiment table, and then restricts processing to locally cached experiment files whose metadata satisfy `project_code == "VisualBehavior"` and `passive == False`. It then loads each kept experiment with `get_behavior_ophys_experiment()`. Trials are not loaded globally up front; they are loaded per experiment inside `extract_session()`.

ii.
```python
def make_cache():
    cls = _import_cache_class()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls.from_s3_cache(cache_dir=DATA_DIR)

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
def extract_session(experiment_id: int) -> dict:
    cache = make_cache()
    exp = cache.get_behavior_ophys_experiment(int(experiment_id))
```

iii. In `CONVERSION_NOTES.md`, the AI says it wanted to use the AllenSDK cache only, avoid direct NWB reads, stay on the exact `VisualBehavior` project, and exclude passive replay sessions because the paper "did not analyze" them and because passive sessions produce artificial all-miss/all-correct-reject outcome tables.

## 1-b. How are the data split into subjects?

i. Subjects are defined by unique `mouse_id` values collected from the per-experiment metadata returned by the SDK. The final `subjects` list is sorted, and each retained experiment/session gets a `subject_idx` pointing into that list.

ii.
```python
result = {
    ...
    "mouse_id": str(meta["mouse_id"]),
    ...
}
```

```python
subjects = sorted({s["mouse_id"] for s in raw_sessions})
subject_map = {name: i for i, name in enumerate(subjects)}
...
"subjects": subjects,
"subject_idx": np.asarray([subject_map[s["mouse_id"]] for s in raw_sessions],
                          dtype=np.int32),
```

iii. The notes describe `mouse_id` as the canonical SDK animal identifier and report cohort counts in terms of distinct mice after the experiment-level filtering.

## 1-c. How are the data split into sessions?

i. The AI effectively treats each retained `ophys_experiment_id` as one decoder session. It preserves `ophys_session_id` in metadata, but it does not merge multiple experiments into one larger session object. Each returned `result` from `extract_session()` becomes one entry in the final session lists.

ii.
```python
def extract_session(experiment_id: int) -> dict:
    ...
    result = {
        "experiment_id": int(experiment_id),
        "ophys_session_id": int(meta["ophys_session_id"]),
        ...
    }
    return result
```

```python
with ProcessPoolExecutor(max_workers=workers) as pool:
    futures = {pool.submit(extract_session, eid): eid for eid in ids}
    ...
    raw_sessions.append(result)
...
for s in raw_sessions:
    ...
    neural.append(s["neural"])
```

iii. In the notes, the AI argues that the exact `VisualBehavior` subset is single-plane, so "session and experiment are one-to-one" for its chosen cohort and no cross-plane merge is needed.

## 1-d. How are the data split into trials?

i. Trials are taken from the AllenSDK `exp.trials` table. The script sorts that table by `start_time`, keeps only eligible go/catch rows, and then creates one half-open time grid per trial from `start_time` to `stop_time`. Each trial gets its own neural and output slices on that grid.

ii.
```python
trials = exp.trials.sort_values("start_time")
eligible = (
    (trials["go"] | trials["catch"])
    & ~trials["aborted"]
    & ~trials["auto_rewarded"]
)
trials = trials.loc[eligible].copy()
```

```python
def _trial_grid(start: float, stop: float) -> np.ndarray:
    n = int(np.ceil((stop - start) * FS - 1e-10))
    grid = start + np.arange(max(0, n), dtype=np.float64) * DT
    return grid[grid < stop]

grids = [_trial_grid(float(r.start_time), float(r.stop_time))
         for r in trials.itertuples()]
```

iii. The notes say the AI followed the SDK trial table because it already contains task-defined start/stop times and trial-type/outcome labels, and it intentionally used half-open `[start_time, stop_time)` trial windows to avoid duplicating boundary samples.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to `(go OR catch) AND NOT aborted AND NOT auto_rewarded`. Sessions with fewer than two eligible trials are rejected. Trials whose generated 30 Hz grids are empty are also rejected by raising an error. The code does not explicitly require `change_time.notna()`.

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
...
if any(len(g) == 0 for g in grids):
    raise ValueError(f"{experiment_id}: empty eligible trial grid")
```

iii. The notes justify these exclusions as matching the task instructions exactly for trial type, and they say passive sessions were separately removed because their outcomes are not genuine behavior.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from the AllenSDK event table, specifically `exp.events["events"]`, not from `dff_traces`.

ii.
```python
event_table = exp.events
cell_ids = event_table.index.to_numpy(copy=True)
events = np.vstack(event_table["events"].to_numpy()).astype(np.float32)
ophys_t = np.asarray(exp.ophys_timestamps, dtype=np.float64)
```

iii. The notes explicitly say the AI chose raw L0 calcium event magnitudes because the supplied paper’s neural analyses and decoder used event traces to reduce slow calcium-decay effects.

## 2-b. How is the `neural` data processed?

i. The event traces are stacked into a cell-by-time matrix, checked against `ophys_timestamps`, linearly interpolated from the native ophys timestamps onto one concatenated 30 Hz trial grid, and then split back into per-trial matrices. The saved matrices are `float32`.

ii.
```python
events = np.vstack(event_table["events"].to_numpy()).astype(np.float32)
ophys_t = np.asarray(exp.ophys_timestamps, dtype=np.float64)
if events.shape[1] != len(ophys_t):
    raise ValueError(f"{experiment_id}: event/timestamp length mismatch")
neural_all = interp_rows_shared_time(ophys_t, events, all_t)
neural = [neural_all[:, offsets[i]:offsets[i + 1]]
          for i in range(len(lengths))]
```

iii. The notes justify this as keeping the AllenSDK preprocessing of the event traces, while matching the paper’s use of 30 Hz interpolation and avoiding repeated per-neuron, per-trial interpolation work.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no extra per-cell quality filter in the script. The AI relies on the SDK’s valid-cell/event tables and only applies higher-level experiment/session filters such as exact `VisualBehavior` and active behavior.

ii.
```python
event_table = exp.events
cell_ids = event_table.index.to_numpy(copy=True)
events = np.vstack(event_table["events"].to_numpy()).astype(np.float32)
```

```python
keep = (
    table.index.isin(file_ids)
    & table["project_code"].eq("VisualBehavior")
    & ~table["passive"].astype(bool)
)
```

iii. The notes say the SDK already exposes QC-filtered valid ROIs/cells, so no extra ad hoc neural thresholding was applied.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial-level neural data are aligned to trial start. For each eligible trial, the script builds a relative 30 Hz grid spanning `[start_time, stop_time)`, interpolates the full-session event traces onto the concatenated trial timestamps, and then slices the resulting array trial by trial.

ii.
```python
def _trial_grid(start: float, stop: float) -> np.ndarray:
    n = int(np.ceil((stop - start) * FS - 1e-10))
    grid = start + np.arange(max(0, n), dtype=np.float64) * DT
    return grid[grid < stop]
```

```python
grids = [_trial_grid(float(r.start_time), float(r.stop_time))
         for r in trials.itertuples()]
all_t = np.concatenate(grids)
neural_all = interp_rows_shared_time(ophys_t, events, all_t)
```

iii. The notes say the temporal origin is "trial start (AllenSDK trials.start_time)" and that alignment still comes from the synchronized ophys timestamps even though the final trial grid is resampled to 30 Hz.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 30 Hz time base (`DT = 1/30 s`, `time_bin_size = 1000/30 ms`) for all sessions and trials. Yes: the neural and behavioral streams are resampled/interpolated onto that common 30 Hz grid.

ii.
```python
FS = 30.0
DT = 1.0 / FS
```

```python
"metadata": {
    ...
    "time_bin_size": 1000.0 / FS,
    "sampling_rate_hz": FS,
    "neural_resampling": "linear interpolation from synchronized ophys timestamps",
    ...
}
```

iii. The notes justify 30 Hz by pointing to the paper’s 30 Hz interpolation and to the fact that running and eye signals are naturally near that scale, while also giving every session exactly the same bin size.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `exp.stimulus_presentations`, specifically the task-block rows whose `stimulus_block_name` contains `change_detection`, together with each row’s `image_name`, `start_time`, `end_time`, and `omitted` flags.

ii.
```python
stim = exp.stimulus_presentations
task = stim[
    stim["stimulus_block_name"].str.contains("change_detection", na=False)
].sort_values("start_time")
```

```python
for row in task.itertuples():
    start = float(row.start_time)
    end = float(row.end_time)
    ...
    if hi > lo and not bool(row.omitted):
        ...
        name = str(row.image_name)
        if name not in IMAGE_TO_INT:
            raise ValueError(f"{experiment_id}: unknown image {name}")
        image_all[idx[valid]] = IMAGE_TO_INT[name]
```

iii. The notes say the AI used stimulus presentations because they give the actual on-screen flash intervals and omitted flashes, which the trial table alone does not fully specify for a time-varying identity signal.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The script initializes image labels to `0`, which corresponds to `"gray"`, and then overwrites only the non-omitted image-presentation intervals with integer codes from a fixed global 17-class mapping (`gray` plus 16 image IDs). The mapping is predefined, not discovered from the data.

ii.
```python
IMAGE_VALUES = [
    "gray", "im000", "im031", "im035", "im045", "im054", "im061",
    "im062", "im063", "im065", "im066", "im069", "im073", "im075",
    "im077", "im085", "im106",
]
IMAGE_TO_INT = {name: idx for idx, name in enumerate(IMAGE_VALUES)}
```

```python
image_all = np.zeros(len(all_t), dtype=np.int16)
...
image_all[idx[valid]] = IMAGE_TO_INT[name]
```

iii. The notes justify the explicit `"gray"` class because the task has 250 ms image flashes separated by 500 ms gray screens, and the requested output asks for image identity only during the non-gray screen.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the exact same concatenated 30 Hz trial timestamps (`all_t`) used for neural interpolation. For each stimulus presentation, the script finds the overlapping sample indices in `all_t` and labels only those samples that actually fall inside `[start_time, end_time)`.

ii.
```python
lo = int(np.searchsorted(all_t, start, side="left"))
hi = int(np.searchsorted(all_t, end, side="left"))
if hi > lo and not bool(row.omitted):
    idx = np.arange(lo, hi)
    valid = (all_t[idx] >= start) & (all_t[idx] < end)
    image_all[idx[valid]] = IMAGE_TO_INT[name]
```

iii. The notes explicitly say the same synchronized ophys-derived time base is used for neural and stimulus signals, so the identity labels are assigned directly onto the decoder time grid.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the same filtered `stimulus_presentations` table, using each row’s `is_change`, `start_time`, `end_time`, and `omitted` fields rather than `trials.change_time`.

ii.
```python
for row in task.itertuples():
    start = float(row.start_time)
    end = float(row.end_time)
    ...
    if hi > lo and not bool(row.omitted):
        ...
        if bool(row.is_change):
            change_all[idx[valid]] = 1
```

iii. The notes say this choice ties "image change" to the actual changed-image presentation interval instead of to a trial-table scalar time.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The script starts with an all-zero vector and sets the value to `1` only during time bins that overlap a changed, non-omitted image presentation. It does not extend the label through the following gray period.

ii.
```python
change_all = np.zeros(len(all_t), dtype=np.int16)
...
if bool(row.is_change):
    change_all[idx[valid]] = 1
```

iii. The notes explain that the AI first tried a one-bin impulse, then broadened it to the full changed-image flash because that was a more stable operationalization of "right after" for pointwise decoding.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is not thresholded from a continuous value. It is directly constructed as a binary categorical output with values `0 = no_change` and `1 = change`.

ii.
```python
change_all = np.zeros(len(all_t), dtype=np.int16)
...
if bool(row.is_change):
    change_all[idx[valid]] = 1
```

```python
"output_values": [
    IMAGE_VALUES,
    ["no_change", "change"],
    ...
],
```

iii. The notes describe image change as an event label, not a measured continuous signal, so there is no extra thresholding stage.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The change labels are assigned on the same `all_t` trial timestamps used for the neural data, using interval overlap with each changed-image presentation.

ii.
```python
lo = int(np.searchsorted(all_t, start, side="left"))
hi = int(np.searchsorted(all_t, end, side="left"))
...
if bool(row.is_change):
    change_all[idx[valid]] = 1
```

iii. The notes justify this the same way as image identity alignment: the label lives on the common decoder time grid derived from synchronized ophys time.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `exp.running_speed`, using the SDK-provided `timestamps` and `speed` columns.

ii.
```python
running = exp.running_speed
...
run_all = finite_interp(running["timestamps"], running["speed"],
                        all_t, "running speed")
```

iii. The notes say the AI intentionally used the SDK-computed running speed rather than recomputing wheel velocity itself.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The script removes nonfinite samples inside `finite_interp()`, requires at least two finite points, linearly interpolates the running speed onto the concatenated 30 Hz trial timestamps, then computes cohort-wide quintile cut points and discretizes each sample with `np.searchsorted`.

ii.
```python
def finite_interp(source_t, source_x, target_t, label: str):
    source_t = np.asarray(source_t, dtype=np.float64)
    source_x = np.asarray(source_x, dtype=np.float64)
    good = np.isfinite(source_t) & np.isfinite(source_x)
    if good.sum() < 2:
        raise ValueError(f"fewer than two finite {label} samples")
    ...
    return np.interp(target_t, t, x).astype(np.float32)
```

```python
run_edges = percentile_edges([s["running"] for s in raw_sessions], "running")
...
run_bin = np.searchsorted(run_edges, s["running"], side="right").astype(np.int16)
```

iii. The notes justify this as using the SDK-filtered locomotion signal, aligning it to the decoder time base, and making five balanced categories across the cohort.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into five percentile bins using the 20th, 40th, 60th, and 80th percentiles computed across all retained sessions.

ii.
```python
def percentile_edges(values: list[np.ndarray], label: str) -> np.ndarray:
    joined = np.concatenate(values).astype(np.float32, copy=False)
    ...
    edges = np.quantile(joined, [0.2, 0.4, 0.6, 0.8]).astype(np.float64)
    ...
    return edges
```

```python
run_bin = np.searchsorted(run_edges, s["running"], side="right").astype(np.int16)
...
"output_values": [
    ...,
    ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
    ...
],
```

iii. The notes explicitly say the bins are cohort-wide quintiles so the categories are comparable across sessions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated directly onto the same concatenated 30 Hz trial timestamps used for neural interpolation, then split into the same per-trial lengths.

ii.
```python
run_all = finite_interp(running["timestamps"], running["speed"],
                        all_t, "running speed")
...
run_trials = split_vector(run_bin, lengths)
...
out_trials.append(np.vstack([
    image_trials[i], change_trials[i], run_trials[i], pupil_trials[i],
    np.full(int(length), s["outcome"][i], dtype=np.int16),
]).astype(np.int16, copy=False))
```

iii. The notes say all continuous streams were aligned to the same ophys-derived decoder grid before trial splitting.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `exp.eye_tracking`, specifically from the cleaned `pupil_width` and `pupil_height` columns. The code defines diameter as `2 * max(width, height)` per sample.

ii.
```python
eye = exp.eye_tracking
if eye is None or eye.empty:
    raise MissingRequiredData("eye tracking table is missing or empty")
...
pupil_diameter = 2.0 * np.maximum(
    eye["pupil_width"].to_numpy(dtype=np.float64),
    eye["pupil_height"].to_numpy(dtype=np.float64),
)
```

iii. The notes say the SDK/whitepaper treat width and height as ellipse half-axes and that the major full axis is the intended pupil diameter measure.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. After computing `2 * max(width, height)`, the script keeps only finite samples through `finite_interp()`, linearly interpolates onto the 30 Hz trial timestamps, computes cohort-wide quintile edges, and discretizes each sample with `np.searchsorted`. Sessions with missing or empty eye-tracking tables are excluded.

ii.
```python
try:
    pupil_all = finite_interp(eye["timestamps"], pupil_diameter,
                              all_t, "pupil diameter")
except ValueError as exc:
    raise MissingRequiredData(str(exc)) from exc
```

```python
pupil_edges = percentile_edges([s["pupil"] for s in raw_sessions], "pupil")
...
pupil_bin = np.searchsorted(pupil_edges, s["pupil"], side="right").astype(np.int16)
```

iii. The notes justify this as using the cleaned SDK pupil geometry, interpolating through bounded blink/missing gaps rather than creating a sixth "missing" class, and dropping the few sessions where the signal is unusable.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five cohort-wide percentile bins using the 20th, 40th, 60th, and 80th percentiles across all retained sessions.

ii.
```python
pupil_edges = percentile_edges([s["pupil"] for s in raw_sessions], "pupil")
...
pupil_bin = np.searchsorted(pupil_edges, s["pupil"], side="right").astype(np.int16)
```

```python
"output_values": [
    ...,
    ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
    ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
    ...
],
```

iii. The notes say the thresholds are cohort-wide quintiles for class balance and cross-session consistency.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated onto the same concatenated 30 Hz trial timestamps used for neural interpolation, then split into per-trial segments with the same lengths as the neural trials.

ii.
```python
pupil_all = finite_interp(eye["timestamps"], pupil_diameter,
                          all_t, "pupil diameter")
...
pupil_trials = split_vector(pupil_bin, lengths)
```

iii. The notes say pupil, running, stimulus labels, and neural traces all share the same ophys-derived decoder time grid.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the trial-table boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
OUTCOME_COLUMNS = ["hit", "miss", "false_alarm", "correct_reject"]
...
outcome_bool = trials[OUTCOME_COLUMNS].to_numpy(dtype=bool)
if not np.all(outcome_bool.sum(axis=1) == 1):
    bad = trials.index[outcome_bool.sum(axis=1) != 1].tolist()
    raise ValueError(f"{experiment_id}: nonexclusive outcomes in {bad[:5]}")
outcomes = outcome_bool.argmax(axis=1).astype(np.int16)
```

iii. The notes say these are the SDK’s mutually exclusive task outcomes for the eligible trial set.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code checks that exactly one of the four outcome flags is true for each eligible trial, maps that one-hot row to integer codes `0..3` with `argmax`, and then repeats that static outcome value across all time bins of that trial in the final output matrix.

ii.
```python
outcome_bool = trials[OUTCOME_COLUMNS].to_numpy(dtype=bool)
...
outcomes = outcome_bool.argmax(axis=1).astype(np.int16)
```

```python
out_trials.append(np.vstack([
    image_trials[i], change_trials[i], run_trials[i], pupil_trials[i],
    np.full(int(length), s["outcome"][i], dtype=np.int16),
]).astype(np.int16, copy=False))
```

iii. The notes say repeating the static trial outcome through time was necessary because the target format expects one output array per trial while the other outputs are time-varying.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles several problems explicitly: it requires at least two eligible trials; it raises on empty trial grids; it checks event/timestamp length consistency; it removes nonfinite running/pupil samples before interpolation; it excludes sessions when running or pupil lack enough finite values; and it excludes sessions with missing or empty eye-tracking tables. For interpolation, `np.interp` is used, so edge samples are filled with the nearest finite value rather than being left as NaN.

ii.
```python
class MissingRequiredData(RuntimeError):
    """A published session cannot supply one of the required decoder outputs."""
```

```python
good = np.isfinite(source_t) & np.isfinite(source_x)
if good.sum() < 2:
    raise ValueError(f"fewer than two finite {label} samples")
...
return np.interp(target_t, t, x).astype(np.float32)
```

```python
eye = exp.eye_tracking
if eye is None or eye.empty:
    raise MissingRequiredData("eye tracking table is missing or empty")
...
except MissingRequiredData as exc:
    exclusions.append({"experiment_id": int(eid), "reason": str(exc)})
    print(f"[{done}/{len(ids)}] {eid}: EXCLUDED ({exc})", flush=True)
    continue
```

iii. The notes say the AI preferred excluding truly unusable sessions over inventing pupil labels, while interpolating across bounded finite gaps for otherwise valid sessions.

## 9-a. What are the most time-consuming steps of the code?

i. The expensive part is loading each experiment and decompressing/constructing the large AllenSDK data objects, especially the neural traces. The notes also identify interpolation over large full-session arrays as a performance concern, but the dominant wall-time cost in the final full run is SDK trace decompression/I/O.

ii.
```python
def extract_session(experiment_id: int) -> dict:
    ...
    cache = make_cache()
    exp = cache.get_behavior_ophys_experiment(int(experiment_id))
```

```python
workers = min(16, len(ids), max(1, (os.cpu_count() or 2) // 2))
with ProcessPoolExecutor(max_workers=workers) as pool:
    futures = {pool.submit(extract_session, eid): eid for eid in ids}
```

iii. In Step 6 and Step 10, the notes say the full-cache bottleneck was SDK trace decompression and that increasing worker count reduced total conversion time substantially.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI specifically targeted the expensive "interpolate each neuron separately" pattern and replaced it with a vectorized interpolation routine that computes timestamp brackets once and applies them to all cells. Trial timestamps are also concatenated before splitting so repeated per-trial interpolation is avoided.

ii.
```python
def interp_rows_shared_time(source_t: np.ndarray, source_x: np.ndarray,
                            target_t: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source_t, target_t, side="right")
    right = np.clip(right, 1, len(source_t) - 1)
    left = right - 1
    denom = source_t[right] - source_t[left]
    weight = ((target_t - source_t[left]) / denom).astype(np.float32)
    lo = source_x[:, left]
    hi = source_x[:, right]
    out = lo + (hi - lo) * weight[None, :]
```

iii. The notes explicitly say the earlier naive approach repeated binary searches for every neuron and that this shared-time interpolation was added as the main vectorization improvement.

## 9-c. What processing does the code repeat multiple times?

i. The code still repeats some setup per experiment: every worker calls `make_cache()` again, every experiment reloads its own SDK object, and every session reruns trial-grid construction and stimulus-interval assignment. What it intentionally avoids repeating is per-trial/per-neuron interpolation: neural, running, and pupil are each interpolated once per session onto one concatenated target vector.

ii.
```python
def extract_session(experiment_id: int) -> dict:
    ...
    cache = make_cache()
    exp = cache.get_behavior_ophys_experiment(int(experiment_id))
```

```python
grids = [_trial_grid(float(r.start_time), float(r.stop_time))
         for r in trials.itertuples()]
all_t = np.concatenate(grids)
...
for row in task.itertuples():
    ...
```

iii. The notes emphasize that repeated interpolation was removed by concatenating trial grids and interpolating each stream once per session, but they also describe independent worker loads for each experiment as part of the performance strategy.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes and stores several metadata fields that the downstream decoder does not need, such as `image_set`, `experience_level`, `source_trial_count`, `eligible_trial_count`, `go_count`, `catch_count`, `outcome_counts`, and `ophys_frame_rate`. It also supports optional plotting via `--show-processing`. These are useful for documentation and sanity checks, but not used by the decoder itself.

ii.
```python
result = {
    ...
    "image_set": str(meta["session_type"]).split("images_")[-1][:1],
    "experience_level": "Familiar" if "images_A" in str(meta["session_type"])
                        else "Novel",
    ...
    "source_trial_count": int(len(exp.trials)),
    "eligible_trial_count": int(len(trials)),
    "go_count": int(trials["go"].sum()),
    "catch_count": int(trials["catch"].sum()),
    "outcome_counts": outcome_bool.sum(axis=0).astype(int).tolist(),
    "ophys_frame_rate": float(meta["ophys_frame_rate"]),
    ...
}
```

```python
if args.show_processing:
    plot_processing(raw_sessions, data)
```

iii. The notes explicitly frame these as documentation, sanity-check, and audit fields added for validation and reproducibility rather than as features required by downstream decoding.
