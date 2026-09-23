# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data through `VisualBehaviorOphysProjectCache.from_s3_cache`, starts from the Allen SDK experiment table, and then restricts to rows with `project_code == "VisualBehavior"`, `behavior_type == "active_behavior"`, and a locally present NWB file. It then loads each selected experiment with `get_behavior_ophys_experiment()`.

ii.
```python
def make_cache():
    return VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=DATA_DIR)

def selected_table(cache):
    table = cache.get_ophys_experiment_table()
    mask = (
        (table["project_code"] == "VisualBehavior")
        & (table["behavior_type"] == "active_behavior")
        & table.index.isin(local_experiment_ids())
    )
    return table.loc[mask].sort_index()

ds = cache.get_behavior_ophys_experiment(int(experiment_id))
```

iii. In `CONVERSION_NOTES.md`, the AI says it wanted the task-relevant subset only, argued that active behavior was required for meaningful go/catch outcomes, and avoided experiment-table rows without a local backing file.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values from the filtered experiment table, converted to sorted strings and later mapped to `subject_idx`.

ii.
```python
subjects = sorted(kept["mouse_id"].astype(str).unique())
subject_map = {x: i for i, x in enumerate(subjects)}
subject_idx = np.asarray([subject_map[str(x)] for x in kept["mouse_id"]], dtype=np.int64)
```

iii. The notes repeatedly describe mice as being identified by SDK `mouse_id`, and the trajectory shows the AI treating those as the canonical subject identifiers.

## 1-c. How are the data split into sessions?

i. The AI treats each selected `ophys_experiment_id` as one session. It does not group multiple experiments by `ophys_session_id`; instead, it assumes that for this VisualBehavior subset, one experiment is one session.

ii.
```python
if args.sample:
    ids = chosen[:2]
else:
    ids = [int(x) for x in table.index]

for k, experiment_id in enumerate(ids, 1):
    result = convert_experiment(cache, experiment_id, image_to_idx, ...)
```

iii. In `CONVERSION_NOTES.md`, the AI explicitly says that VisualBehavior is single-plane and therefore “each selected experiment is exactly one recording session.”

## 1-d. How are the data split into trials?

i. Trials are taken from `ds.trials`. For each kept trial, the AI converts `start_time` and `stop_time` into frame indices with `np.searchsorted` on `ophys_timestamps`, and slices the neural/output arrays on that half-open interval.

ii.
```python
trials = ds.trials
...
for trial_id, row in trials.iterrows():
    lo = int(np.searchsorted(ts, float(row.start_time), side="left"))
    hi = int(np.searchsorted(ts, float(row.stop_time), side="left"))
    if hi - lo < 2:
        continue
    x = events[:, lo:hi]
    y[0] = image[lo:hi]
```

iii. The notes say the SDK `trials` table is authoritative and that the AI intentionally uses the SDK interval `[start_time, stop_time)` as the trial definition.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only go or catch trials, excludes aborted and auto-rewarded trials, skips trials with ambiguous outcome flags, skips trials with fewer than 2 ophys frames, skips trials with nonfinite neural/output values, and rejects sessions with fewer than 2 surviving trials.

ii.
```python
keep = (trials["go"] | trials["catch"]) & ~trials["aborted"] & ~trials["auto_rewarded"]
trials = trials.loc[keep]
...
flags = np.asarray([bool(row[x]) for x in OUTCOMES])
if flags.sum() != 1:
    continue
...
if hi - lo < 2:
    continue
...
if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
    continue
...
if len(neural_trials) < 2:
    raise ValueError(f"only {len(neural_trials)} valid trials")
```

iii. The notes justify the go/catch and aborted/auto-rewarded filters from the task instructions, and justify the “exactly one outcome” rule plus trial/session length checks as structural validation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural data are derived from `ds.events["events"]`, not from `dff_traces`.

ii.
```python
ts = np.asarray(ds.ophys_timestamps, dtype=np.float64)
events = np.stack(ds.events["events"].to_numpy()).astype(np.float32, copy=False)
```

iii. The trajectory and notes show that the AI initially planned to use dF/F, then changed to Allen SDK L0 calcium events because it interpreted the supplied paper as saying neural analyses should use detected events.

## 2-b. How is the `neural` data processed?

i. The AI stacks the per-cell event arrays into a 2D matrix, truncates timestamps and events to a common minimum length if needed, casts to `float32`, and then slices per trial. It does not further smooth, normalize, or rebin the neural signal.

ii.
```python
events = np.stack(ds.events["events"].to_numpy()).astype(np.float32, copy=False)
n = min(ts.size, events.shape[1])
ts, events = ts[:n], events[:, :n]
...
x = events[:, lo:hi]
neural_trials.append(x)
```

iii. The notes say the AI wanted “reference-matched L0 calcium events” at native ophys frames and considered additional processing unnecessary because the Allen SDK signal was already the intended processed source.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no extra neuron-level selection beyond whatever the Allen SDK already exposes. The AI only rejects entire sessions or trials if timestamps or event values are invalid/nonfinite.

ii.
```python
if len(ts) < 2 or np.any(np.diff(ts) <= 0):
    raise ValueError("invalid ophys timestamps")
if not np.all(np.isfinite(events)):
    raise ValueError("nonfinite calcium events")
...
if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
    continue
```

iii. In the notes, the AI says it is relying on the release/QC-valid SDK objects and does not invent an extra ROI-quality threshold of its own.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to the native ophys timestamps and trialized by taking the first ophys frame at or after `start_time` through the first frame at or after `stop_time`, i.e. the SDK half-open trial interval.

ii.
```python
lo = int(np.searchsorted(ts, float(row.start_time), side="left"))
hi = int(np.searchsorted(ts, float(row.stop_time), side="left"))
x = events[:, lo:hi]
```

iii. The notes say the common time base should be the synchronized ophys frame timestamps, and that trial alignment should use the SDK `[start_time, stop_time)` interval exactly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps one native ophys frame per bin, computes metadata `time_bin_size` from the median frame interval across converted sessions, and does not apply temporal rebinning.

ii.
```python
median_bin = float(np.median([x["median_frame_interval_ms"] for x in infos]))
...
"time_bin_size": median_bin,
...
"temporal_alignment_event": "Native synchronized ophys frame timestamps; each trial is the half-open SDK interval [start_time, stop_time).",
```

iii. The notes explicitly say “do not re-bin or interpolate neural traces” and to preserve the native ophys frame grid.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table for the `change_detection` block, specifically `image_name`, `start_time`, `end_time`, and `omitted`.

ii.
```python
def task_stimuli(dataset):
    stim = dataset.stimulus_presentations
    names = stim["stimulus_block_name"].fillna("").astype(str)
    return stim.loc[names.str.contains("change_detection", case=False)].sort_values("start_time")

stim = task_stimuli(dataset)
...
omitted = bool(row.omitted) if not np.isnan(row.omitted) else False
name = str(row.image_name)
```

iii. The notes justify this by saying the newer SDK stimulus table can contain multiple blocks and that only the change-detection block should define task image labels.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI creates a global vocabulary of image names by loading one representative experiment per image set, prepends a synthetic `"gray"` class, fills the entire ophys timeline with the gray code by default, then overwrites intervals covered by non-omitted image presentations.

ii.
```python
def discover_images(cache, table) -> list[str]:
    found: set[str] = set()
    for _, group in table.groupby("image_set", dropna=False):
        ds = cache.get_behavior_ophys_experiment(int(group.index[0]))
        stim = task_stimuli(ds)
        vals = stim.loc[~stim["omitted"].fillna(False), "image_name"].dropna().astype(str)
        found.update(v for v in vals.unique() if v.lower() not in {"omitted", "nan"})
    return ["gray"] + sorted(found)

image = np.full(n, image_to_idx["gray"], dtype=np.int16)
...
if not omitted and name in image_to_idx and end > start:
    image[start:end] = image_to_idx[name]
```

iii. The notes say the AI wanted a deterministic global vocabulary and explicitly chose to model the gray inter-stimulus and omission periods as a dedicated class.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The image-identity array is built directly on the same ophys timestamp grid as the neural data, then sliced with the same `[lo:hi]` trial indices.

ii.
```python
start = int(np.searchsorted(timestamps, float(row.start_time), side="left"))
end = int(np.searchsorted(timestamps, float(row.end_time), side="left"))
...
y[0] = image[lo:hi]
```

iii. The notes say that all labels should be sampled on the synchronized ophys clock, so image identity and neural activity share one framewise time base.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations.is_change` and the corresponding presentation `start_time` within the task block.

ii.
```python
stim = task_stimuli(dataset)
for row in stim.itertuples():
    start = int(np.searchsorted(timestamps, float(row.start_time), side="left"))
    ...
    if bool(row.is_change) and start < n:
        change[start] = 1
```

iii. The notes justify this as using the task stimulus table itself, rather than the trial table, to mark real image-identity transitions.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI initializes a zero vector over the full ophys timeline and places a one-frame impulse at the first ophys frame of each presentation marked `is_change`.

ii.
```python
change = np.zeros(n, dtype=np.int16)
...
if bool(row.is_change) and start < n:
    change[start] = 1
```

iii. The notes explicitly describe the target as a “change impulse” at the first ophys frame at or after a true image change.

## 4-c. How is `output` *Image change* thresholded into categories?

i. The AI uses a binary categorical coding: `0` for no change and `1` for change.

ii.
```python
change = np.zeros(n, dtype=np.int16)
...
"output_values": [images, ["no_change", "change"],
                  ["q1_lowest", "q2", "q3", "q4", "q5_highest"],
                  ["q1_lowest", "q2", "q3", "q4", "q5_highest"], OUTCOMES],
```

iii. The notes describe this as a sparse binary event channel rather than a continuous value needing threshold selection.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The change vector is built on the native ophys timestamps and then sliced by the same trial indices used for neural data.

ii.
```python
start = int(np.searchsorted(timestamps, float(row.start_time), side="left"))
...
y[1] = change[lo:hi]
```

iii. The AI’s notes say that all outputs should share the ophys frame time base, so change and neural slices stay aligned by construction.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, using its `timestamps` and `speed` columns.

ii.
```python
run = dataset.running_speed
speed = interpolate_valid(run["timestamps"], run["speed"], timestamps)
```

iii. The notes say this is the Allen SDK filtered running-speed stream and should be used instead of recreating wheel processing.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI linearly interpolates running speed onto the ophys timestamps and then discretizes the aligned session-wide values into five bins using the session’s own 20/40/60/80 percentiles.

ii.
```python
def percentile_bins(values: np.ndarray) -> np.ndarray:
    edges = np.percentile(values, [20, 40, 60, 80])
    return np.searchsorted(edges, values, side="right").astype(np.int16)

run = dataset.running_speed
speed = interpolate_valid(run["timestamps"], run["speed"], timestamps)
speed_bin = percentile_bins(speed)
```

iii. The notes justify interpolation because behavior and ophys timestamps are synchronized, and justify per-session quintiles as a way to control session-specific scale/calibration differences.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded into five ordinal categories by applying `np.percentile(values, [20, 40, 60, 80])` to the aligned running-speed samples for each session and using `np.searchsorted(..., side="right")` to assign labels `0` through `4`.

ii.
```python
edges = np.percentile(values, [20, 40, 60, 80])
return np.searchsorted(edges, values, side="right").astype(np.int16)
```

iii. The notes call these “per-session 20/40/60/80 percentiles on aligned continuous samples.”

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is first interpolated to the ophys timestamps and then trialized with the same `[lo:hi]` indices used for neural data.

ii.
```python
speed = interpolate_valid(run["timestamps"], run["speed"], timestamps)
...
y[2] = speed_bin[lo:hi]
```

iii. The notes repeatedly state that the synchronized ophys frame clock is the shared alignment grid for all signals.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`, specifically `pupil_area` and `timestamps`.

ii.
```python
eye = dataset.eye_tracking
area = eye["pupil_area"].to_numpy(dtype=np.float64)
diameter = 2.0 * np.sqrt(np.maximum(area, 0.0) / np.pi)
```

iii. The notes say the AI preferred a diameter-equivalent computed from area rather than directly using pupil width, because it considered that a more geometric definition of diameter.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI converts pupil area to diameter-equivalent values, filters implicitly to finite samples inside `interpolate_valid`, interpolates to ophys timestamps, and then bins the aligned session-wide values into five per-session percentile bins.

ii.
```python
def interpolate_valid(t_src, value_src, t_dst):
    valid = np.isfinite(t_src) & np.isfinite(value_src)
    if valid.sum() < 2:
        raise ValueError("fewer than two finite source samples")
    order = np.argsort(t_src[valid])
    return np.interp(t_dst, t_src[valid][order], value_src[valid][order])

area = eye["pupil_area"].to_numpy(dtype=np.float64)
diameter = 2.0 * np.sqrt(np.maximum(area, 0.0) / np.pi)
pupil = interpolate_valid(eye["timestamps"], diameter, timestamps)
pupil_bin = percentile_bins(pupil)
```

iii. The notes justify finite-sample interpolation as a way to avoid fabricating blink values, and justify per-session quintiles for the same normalization reason given for running speed.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is thresholded into five ordinal categories using the session’s 20/40/60/80 percentile cut points on aligned pupil-diameter samples, with labels `0` through `4`.

ii.
```python
edges = np.percentile(values, [20, 40, 60, 80])
return np.searchsorted(edges, values, side="right").astype(np.int16)
```

iii. The notes describe these as per-session quintiles rather than a global set of edges shared across all sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil values are interpolated onto the same ophys timestamp grid as the neural data, and trial slices use the same `[lo:hi]` indices.

ii.
```python
pupil = interpolate_valid(eye["timestamps"], diameter, timestamps)
...
y[3] = pupil_bin[lo:hi]
```

iii. The notes say pupil, running, stimulus labels, and neural activity should all live on the synchronized ophys timeline.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
flags = np.asarray([bool(row[x]) for x in OUTCOMES])
...
OUTCOMES = ["hit", "miss", "false_alarm", "correct_reject"]
```

iii. The notes and trajectory both describe these four SDK fields as the canonical mutually exclusive outcome labels for valid trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI encodes the unique true outcome flag as integers `0` to `3` in the order given by `OUTCOMES`, and repeats that integer across every frame of the trial.

ii.
```python
flags = np.asarray([bool(row[x]) for x in OUTCOMES])
if flags.sum() != 1:
    continue
outcome = int(np.flatnonzero(flags)[0])
...
y[4] = outcome
```

iii. The notes justify repeating the static trial label across time because the output format expects one `(n_output, T)` matrix per trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI skips invalid sessions rather than repairing them. It raises on missing/insufficient eye-tracking support, invalid timestamps, or nonfinite event matrices; skips trials with ambiguous outcomes, too few frames, or nonfinite trial slices; and lets `np.interp` extrapolate from the nearest valid endpoint when the target ophys times fall outside the source support.

ii.
```python
if valid.sum() < 2:
    raise ValueError("fewer than two finite source samples")
...
if eye is None:
    raise ValueError("eye tracking unavailable")
...
if len(ts) < 2 or np.any(np.diff(ts) <= 0):
    raise ValueError("invalid ophys timestamps")
if not np.all(np.isfinite(events)):
    raise ValueError("nonfinite calcium events")
...
if flags.sum() != 1:
    continue
if hi - lo < 2:
    continue
if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
    continue
```

iii. The notes frame this as a conservative policy: do not fabricate required targets, especially pupil bins, and drop sessions/trials when the synchronized source data are structurally inadequate.

## 9-a. What are the most time-consuming steps of the code?

i. The main cost is loading each experiment through the Allen SDK cache and materializing full-session arrays before trial slicing.

ii.
```python
ds = cache.get_behavior_ophys_experiment(int(experiment_id))
...
for k, experiment_id in enumerate(ids, 1):
    result = convert_experiment(cache, experiment_id, image_to_idx, ...)
```

iii. The notes say SDK construction/loading dominates runtime and report per-session conversion times largely driven by Allen SDK I/O.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The most obvious vectorizable loops are the per-stimulus loop in `build_continuous_outputs()` and the per-trial loop in `convert_experiment()`. Both currently use Python iteration with repeated `searchsorted` and slicing.

ii.
```python
for row in stim.itertuples():
    start = int(np.searchsorted(timestamps, float(row.start_time), side="left"))
    end = int(np.searchsorted(timestamps, float(row.end_time), side="left"))
    ...

for trial_id, row in trials.iterrows():
    lo = int(np.searchsorted(ts, float(row.start_time), side="left"))
    hi = int(np.searchsorted(ts, float(row.stop_time), side="left"))
    ...
```

iii. The notes do not emphasize vectorization, and instead argue that I/O and SDK loading dominate runtime, which explains why these loops were left in simple Python form.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats some session loading during image-vocabulary discovery: `discover_images()` loads one representative session per image set, and those sessions are then loaded again later during full conversion. It also recomputes percentile bins separately for each session.

ii.
```python
for _, group in table.groupby("image_set", dropna=False):
    ds = cache.get_behavior_ophys_experiment(int(group.index[0]))
    ...

for k, experiment_id in enumerate(ids, 1):
    result = convert_experiment(cache, experiment_id, image_to_idx, ...)
```

iii. The notes describe representative-session image discovery as a deliberate tradeoff to build a deterministic vocabulary up front, and describe per-session quintiles as intentional normalization rather than a shared global binning pass.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest extra work is representative-session loading purely to discover image names before the main conversion pass, and bookkeeping for diagnostic plotting (`plot_rows`, continuous `speed`/`pupil`) that is not part of the saved downstream dataset.

ii.
```python
images = discover_images(cache, table)
...
plot_rows = []
...
image, change, speed_bin, pupil_bin, speed, pupil = build_continuous_outputs(
    ds, ts, image_to_idx
)
...
if len(plot_rows) < 6:
    plot_rows.append((lo, hi, outcome, int(trial_id)))
```

iii. The notes justify these as diagnostic and documentation aids, not as quantities needed by the final decoder-facing pickle format itself.
