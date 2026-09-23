# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `VisualBehaviorOphysProjectCache.from_s3_cache()` to access the Allen SDK cache. It calls `get_ophys_experiment_table()` and filters to `project_code == 'VisualBehavior'` and `behavior_type == 'active_behavior'`. It further restricts to locally available experiment IDs by scanning the NWB file directory. Each experiment is loaded via `cache.get_behavior_ophys_experiment(experiment_id)`.

ii.
```python
cache = make_cache()
table = selected_table(cache)

def selected_table(cache):
    table = cache.get_ophys_experiment_table()
    mask = (
        (table["project_code"] == "VisualBehavior")
        & (table["behavior_type"] == "active_behavior")
        & table.index.isin(local_experiment_ids())
    )
    return table.loc[mask].sort_index()

def local_experiment_ids() -> set[int]:
    folder = DATA_DIR / "visual-behavior-ophys-1.1.0" / "behavior_ophys_experiments"
    return {int(p.stem.rsplit("_", 1)[1]) for p in folder.glob("*.nwb")}
```

iii. The AI filtered to `active_behavior` because passive sessions lack meaningful go/catch behavioral outcomes required for the trial outcome output variable. The local experiment ID check ensures only locally cached data is used. This is documented in CONVERSION_NOTES.md Steps 2 and 4.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values from the filtered experiment table, sorted and stored as strings.

ii.
```python
subjects = sorted(kept["mouse_id"].astype(str).unique())
subject_map = {x: i for i, x in enumerate(subjects)}
subject_idx = np.asarray([subject_map[str(x)] for x in kept["mouse_id"]], dtype=np.int64)
```

iii. The `mouse_id` field is the SDK's unique identifier for each animal.

## 1-c. How are the data split into sessions?

i. Each experiment (single imaging plane) is treated as its own session. Since VisualBehavior is single-plane, each ophys_session_id maps one-to-one to an experiment. The AI does NOT group multiple experiments into a single session.

ii.
```python
ids = [int(x) for x in table.index]  # each experiment ID is a session
for k, experiment_id in enumerate(ids, 1):
    result = convert_experiment(cache, experiment_id, image_to_idx, ...)
```

iii. The AI documented in CONVERSION_NOTES.md Step 4 that "VisualBehavior is single-plane, so all selected ophys session IDs map one-to-one to experiments" and decided to treat each experiment as one session.

## 1-d. How are the data split into trials?

i. Trials are defined using the SDK's `dataset.trials` table. The AI keeps trials where `(go | catch) & ~aborted & ~auto_rewarded` and where exactly one outcome flag is True. Trials use the `[start_time, stop_time)` interval from the trials table, giving variable-length trials.

ii.
```python
trials = ds.trials
keep = (trials["go"] | trials["catch"]) & ~trials["aborted"] & ~trials["auto_rewarded"]
trials = trials.loc[keep]
for trial_id, row in trials.iterrows():
    flags = np.asarray([bool(row[x]) for x in OUTCOMES])
    if flags.sum() != 1:
        continue
    lo = int(np.searchsorted(ts, float(row.start_time), side="left"))
    hi = int(np.searchsorted(ts, float(row.stop_time), side="left"))
    if hi - lo < 2:
        continue
```

iii. The AI used go/catch filtering as required by the instructions. The additional check for exactly one valid outcome ensures clean trial labeling.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters out: (1) aborted trials, (2) auto-rewarded trials, (3) trials without exactly one valid outcome flag (hit/miss/false_alarm/correct_reject), (4) trials with fewer than 2 ophys frames, (5) trials with non-finite neural or output data. Sessions with fewer than 2 valid trials are excluded. Three sessions were excluded due to insufficient valid pupil data.

ii.
```python
keep = (trials["go"] | trials["catch"]) & ~trials["aborted"] & ~trials["auto_rewarded"]
# ...
if flags.sum() != 1:
    continue
if hi - lo < 2:
    continue
if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
    continue
# ...
if len(neural_trials) < 2:
    raise ValueError(f"only {len(neural_trials)} valid trials")
```

iii. The AI documented these filters as ensuring data quality. Sessions with insufficient pupil data were excluded rather than fabricating labels.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dataset.events["events"]` -- the L0-detected calcium event magnitudes, NOT dF/F traces. The AI initially used dF/F but switched to events during Critical Review 1 after identifying that the reference paper explicitly uses detected events for neural analyses.

ii.
```python
events = np.stack(ds.events["events"].to_numpy()).astype(np.float32, copy=False)
n = min(ts.size, events.shape[1])
ts, events = ts[:n], events[:, :n]
```

iii. The AI's CONVERSION_NOTES.md Step 10 documents the switch: "Initial dF/F activity differed from the paper's explicit detected-event stream. Switched to SDK unfiltered L0 events and reran all affected work."

## 2-b. How is the `neural` data processed?

i. The events array is stacked from the SDK, cast to float32, and length-matched to the ophys timestamps. No additional filtering, normalization, or smoothing is applied.

ii.
```python
events = np.stack(ds.events["events"].to_numpy()).astype(np.float32, copy=False)
n = min(ts.size, events.shape[1])
ts, events = ts[:n], events[:, :n]
```

iii. The SDK pipeline already applies its own quality control. The AI chose to use the raw L0 events without additional processing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied. The AI uses all SDK-valid ROIs (cells that passed the Allen pipeline's segmentation and QC). Non-finite event values cause the entire trial to be skipped.

ii.
```python
if not np.all(np.isfinite(events)):
    raise ValueError("nonfinite calcium events")
# ...
if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
    continue
```

iii. The AI relies on the Allen release QC which includes checks for saturation, bleaching, targeting, z-drift, etc.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's neural data is aligned to the trial's `start_time` from the SDK trials table. The ophys frames in `[start_time, stop_time)` are extracted using `np.searchsorted` with `side="left"`.

ii.
```python
lo = int(np.searchsorted(ts, float(row.start_time), side="left"))
hi = int(np.searchsorted(ts, float(row.stop_time), side="left"))
x = events[:, lo:hi]
```

iii. The AI uses native synchronized ophys frame timestamps. Each trial spans from `start_time` to `stop_time` as defined by the SDK.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The native ophys frame rate is used (~31 Hz for single-plane, ~32.26 ms per frame). The time bin size is reported as the median inter-frame interval across sessions.

ii.
```python
median_bin = float(np.median([x["median_frame_interval_ms"] for x in infos]))
# ...
"time_bin_size": median_bin,
```

iii. The AI keeps the native temporal resolution, which is appropriate since single-plane VisualBehavior sessions have a consistent ~31 Hz frame rate.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, specifically the `image_name`, `start_time`, `end_time`, `omitted`, and `is_change` columns. The AI filters to the `change_detection` stimulus block. A "gray" class covers inter-stimulus intervals and omitted presentations.

ii.
```python
def task_stimuli(dataset):
    stim = dataset.stimulus_presentations
    names = stim["stimulus_block_name"].fillna("").astype(str)
    return stim.loc[names.str.contains("change_detection", case=False)].sort_values("start_time")

def build_continuous_outputs(dataset, timestamps, image_to_idx):
    n = len(timestamps)
    image = np.full(n, image_to_idx["gray"], dtype=np.int16)
    stim = task_stimuli(dataset)
    for row in stim.itertuples():
        start = int(np.searchsorted(timestamps, float(row.start_time), side="left"))
        end = int(np.searchsorted(timestamps, float(row.end_time), side="left"))
        omitted = bool(row.omitted) if not np.isnan(row.omitted) else False
        name = str(row.image_name)
        if not omitted and name in image_to_idx and end > start:
            image[start:end] = image_to_idx[name]
```

iii. The AI used the stimulus_presentations table because it provides exact stimulus onset/offset times, enabling frame-by-frame image identity tracking including the gray inter-stimulus intervals.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global image vocabulary is built from all unique image names across all sessions, with "gray" prepended. Image names are mapped to integer indices. For each session, a continuous image identity trace is constructed by defaulting to "gray" and overwriting with the image code during each non-omitted stimulus presentation interval.

ii.
```python
images = discover_images(cache, table)  # ["gray"] + sorted(found)
image_to_idx = {name: i for i, name in enumerate(images)}
# ...
image = np.full(n, image_to_idx["gray"], dtype=np.int16)
# overwrite during stimulus presentations
if not omitted and name in image_to_idx and end > start:
    image[start:end] = image_to_idx[name]
```

iii. The AI chose to include "gray" as an explicit class because most frames show the inter-stimulus gray screen, and the instructions ask for "Image identity (of the image presented during the non-grey screen)."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is constructed at every ophys timestamp, so it shares the same time axis as the neural data. The same `lo:hi` indices are used to slice both.

ii.
```python
y[0] = image[lo:hi]  # same indices as neural: events[:, lo:hi]
```

iii. Both are indexed by ophys frame timestamps, ensuring alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the `stimulus_presentations` table.

ii.
```python
if bool(row.is_change) and start < n:
    change[start] = 1
```

iii. The AI uses the stimulus_presentations table's `is_change` flag to identify true image changes.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary impulse (single frame = 1) is placed at the first ophys frame at or after each true image-change onset. All other frames are 0.

ii.
```python
change = np.zeros(n, dtype=np.int16)
# ...
if bool(row.is_change) and start < n:
    change[start] = 1
```

iii. The AI marks only a single frame as the change impulse, rather than a window.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary: 0 (no change) or 1 (change). No thresholding is applied.

ii. See 4-b.

iii. The variable is already binary by definition.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity -- the change indicator is constructed at ophys timestamps and sliced with the same `lo:hi` indices.

ii.
```python
y[1] = change[lo:hi]
```

iii. Alignment is guaranteed by shared ophys frame indexing.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, which provides timestamps and speed values from the SDK's filtered running wheel data.

ii.
```python
run = dataset.running_speed
speed = interpolate_valid(run["timestamps"], run["speed"], timestamps)
```

iii. The `running_speed` attribute is the SDK's standard interface for locomotion data with 10-Hz low-pass filtering already applied.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the ophys timebase using `np.interp` (after filtering out non-finite values). Then it is discretized into 5 percentile-based bins computed per-session.

ii.
```python
def interpolate_valid(t_src, value_src, t_dst):
    valid = np.isfinite(t_src) & np.isfinite(value_src)
    order = np.argsort(t_src[valid])
    return np.interp(t_dst, t_src[valid][order], value_src[valid][order])

def percentile_bins(values):
    edges = np.percentile(values, [20, 40, 60, 80])
    return np.searchsorted(edges, values, side="right").astype(np.int16)

speed_bin = percentile_bins(speed)
```

iii. Linear interpolation resamples to the ophys timebase. Per-session percentile binning controls for session-specific speed distributions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal percentile bins (quintiles) using the 20th, 40th, 60th, and 80th percentile edges, computed per-session. The bins are labeled 0-4.

ii.
```python
edges = np.percentile(values, [20, 40, 60, 80])
return np.searchsorted(edges, values, side="right").astype(np.int16)
```

iii. Per-session percentile binning ensures approximately equal class counts within each session.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to ophys timestamps before trial segmentation, then sliced with the same `lo:hi` indices as neural data.

ii.
```python
speed = interpolate_valid(run["timestamps"], run["speed"], timestamps)
# ...
y[2] = speed_bin[lo:hi]
```

iii. Interpolation to the common ophys timebase ensures alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`, specifically the `pupil_area` column. The AI computes diameter-equivalent as `2 * sqrt(pupil_area / pi)`.

ii.
```python
eye = dataset.eye_tracking
area = eye["pupil_area"].to_numpy(dtype=np.float64)
diameter = 2.0 * np.sqrt(np.maximum(area, 0.0) / np.pi)
pupil = interpolate_valid(eye["timestamps"], diameter, timestamps)
```

iii. The AI chose `pupil_area` over `pupil_width` and computed a diameter-equivalent using the geometric formula, as documented in CONVERSION_NOTES.md Step 4: "Compute diameter-equivalent `2*sqrt(pupil_area/pi)` from valid area, interpolate only within valid support."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area is converted to diameter-equivalent via `2*sqrt(area/pi)`. Non-finite values (including blink-flagged NaN values) are filtered out before interpolation to ophys timestamps. Then per-session percentile quintile binning is applied.

ii.
```python
diameter = 2.0 * np.sqrt(np.maximum(area, 0.0) / np.pi)
pupil = interpolate_valid(eye["timestamps"], diameter, timestamps)
pupil_bin = percentile_bins(pupil)
```

iii. The AI uses `interpolate_valid` which filters non-finite values before interpolation, effectively removing blink artifacts. Note: the AI does NOT explicitly use the `likely_blink` column for filtering; instead it relies on the SDK setting derived pupil values to NaN during blinks, and then `interpolate_valid` removes all non-finite values.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 equal percentile bins (quintiles) computed per-session.

ii.
```python
pupil_bin = percentile_bins(pupil)
```

iii. Per-session percentile binning.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed -- interpolated to ophys timestamps, then sliced with same indices.

ii.
```python
pupil = interpolate_valid(eye["timestamps"], diameter, timestamps)
y[3] = pupil_bin[lo:hi]
```

iii. Shared ophys timebase ensures alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
OUTCOMES = ["hit", "miss", "false_alarm", "correct_reject"]
flags = np.asarray([bool(row[x]) for x in OUTCOMES])
if flags.sum() != 1:
    continue
outcome = int(np.flatnonzero(flags)[0])
```

iii. The four outcomes are mutually exclusive for valid go/catch trials. Trials without exactly one true flag are excluded.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3) based on position in the OUTCOMES list. The code is repeated across all time bins in the trial to fit the (5, T) output matrix shape.

ii.
```python
y[4] = outcome  # broadcast scalar to all time bins
```

iii. Static outcome is repeated over time to coexist with time-varying outputs in the same matrix, as documented in CONVERSION_NOTES.md Step 5.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Failed sessions**: If `convert_experiment` throws an exception, the session is skipped with a warning.
- **Short trials**: Trials with fewer than 2 ophys frames are skipped.
- **Non-finite data**: Trials with non-finite neural or output data are skipped.
- **Insufficient pupil data**: Sessions with fewer than 2 finite pupil samples raise an error and are excluded (3 sessions).
- **Missing outcomes**: Trials without exactly one valid outcome flag are skipped.
- **Timestamp/event length mismatch**: Neural events and timestamps are truncated to the shorter length.

ii.
```python
n = min(ts.size, events.shape[1])
ts, events = ts[:n], events[:, :n]
# ...
if flags.sum() != 1:
    continue
if hi - lo < 2:
    continue
if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
    continue
# ...
except Exception as exc:
    print(f"SKIP {experiment_id}: ...")
    continue
```

iii. The try/except ensures a single bad session doesn't crash the pipeline. The finite-data check avoids propagating NaN/Inf values.

## 9-a. What are the most time-consuming steps of the code?

i. Loading each experiment via `cache.get_behavior_ophys_experiment()` dominates runtime, as it reads large NWB-backed neural and behavioral data from the SDK cache. The AI reports full conversion took ~13.3 minutes for 168 sessions.

ii. N/A

iii. The AI's CONVERSION_NOTES.md Step 7 estimates ~4.4-4.5 s per session for loading and conversion.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-stimulus-presentation loop in `build_continuous_outputs` iterates over each stimulus row to set image identity and change indicators. This could potentially be vectorized using interval-based array operations. The per-trial loop in the main conversion function is sequential but does minimal computation.

ii.
```python
for row in stim.itertuples():
    start = int(np.searchsorted(timestamps, float(row.start_time), side="left"))
    end = int(np.searchsorted(timestamps, float(row.end_time), side="left"))
    # ...
```

iii. The I/O-bound SDK loading dominates runtime, so vectorizing these loops would yield minimal speedup.

## 9-c. What processing does the code repeat multiple times?

i. The `discover_images` function loads one representative experiment per image set to build the global image vocabulary before the main conversion loop. This means some experiments are loaded twice (once for vocabulary discovery, once for conversion). The `percentile_bins` function is computed independently per session, repeating the percentile calculation.

ii.
```python
def discover_images(cache, table) -> list[str]:
    for _, group in table.groupby("image_set", dropna=False):
        ds = cache.get_behavior_ophys_experiment(int(group.index[0]))
        # ...
```

iii. The vocabulary discovery loads only one experiment per image set (not all), so the redundancy is minimal.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes continuous (pre-binning) speed and pupil values that are only used for plotting in `--show-processing` mode. In normal mode, these continuous values are computed but only the binned versions are stored. The AI also computes and returns `speed` and `pupil` continuous values from `build_continuous_outputs` even when not plotting.

ii.
```python
return image, change, speed_bin, pupil_bin, speed, pupil
# speed and pupil are only used in plot_processing()
```

iii. The continuous values are small relative to the neural data and the overhead is negligible.
