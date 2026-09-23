# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads a CSV experiment table from disk (`ophys_experiment_table.csv`) and discovers which NWB files are locally available by globbing the NWB directory. It filters to active Visual Behavior experiments with local NWB files. Each experiment is loaded individually using `BehaviorOphysExperiment.from_nwb_path()`.

ii.
```python
table = pd.read_csv(EXPERIMENT_TABLE, index_col="ophys_experiment_id")
local_ids = _local_experiment_ids()
table = table.loc[table.index.intersection(local_ids)].copy()
active = table[table["behavior_type"].eq("active_behavior") & ~table["passive"]]
...
dataset = BehaviorOphysExperiment.from_nwb_path(str(path))
```

iii. The agent investigated the local data directory structure and found NWB files and a CSV experiment table. It chose to load directly from NWB files rather than using the S3 cache, since the data was provided locally. The agent explicitly filtered for active behavior sessions to exclude passive replay sessions which lack genuine behavioral outcomes.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the `mouse_id` field in each experiment's metadata. Unique mouse IDs are collected across all converted experiments and sorted.

ii.
```python
subjects = sorted({session["mouse_id"] for session in converted})
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The `mouse_id` is the Allen SDK's canonical identifier for each animal. Sorting ensures deterministic ordering.

## 1-c. How are the data split into sessions?

i. Each imaging plane (`ophys_experiment_id`) is treated as a separate decoder session. This means multi-plane (Multiscope) recordings are split into individual plane-level sessions rather than grouped by `ophys_session_id`.

ii.
```python
def convert_experiment(experiment_id: int) -> dict:
    """Load and convert one imaging plane/session."""
    path = NWB_DIR / f"behavior_ophys_experiment_{experiment_id}.nwb"
    dataset = BehaviorOphysExperiment.from_nwb_path(str(path))
```

iii. The agent reasoned: "Each imaging plane (`ophys_experiment_id`) is one decoder session, matching the plane-wise decoding analysis and avoiding artificial interpolation among interleaved Multiscope planes." The agent noted that Multiscope planes are acquired at different frame rates and interleaved, so grouping them would require artificial interpolation.

## 1-d. How are the data split into trials?

i. Trials are defined using the SDK's built-in `dataset.trials` table. Go and catch trials are retained; aborted and auto-rewarded trials are excluded. Each trial spans from `start_time` to `stop_time` (variable length). The trial duration is divided into 100ms time bins, with the number of bins being `floor(duration / 0.1)`.

ii.
```python
keep = (
    (trials["go"].astype(bool) | trials["catch"].astype(bool))
    & ~trials["aborted"].astype(bool)
    & ~trials["auto_rewarded"].astype(bool)
)
trials = trials.loc[keep].copy()
...
duration = float(row["stop_time"] - row["start_time"])
n_bins = int(np.floor(duration / BIN_SEC))
centers = float(row["start_time"]) + (np.arange(n_bins) + 0.5) * BIN_SEC
```

iii. The agent followed the instructions to include Go and Catch trials while excluding Aborted and Auto-rewarded trials. The trial window uses the SDK's start/stop times, giving variable-length trials (~8s, ~80 bins at 100ms).

## 1-e. How are trials filtered based on quality controls?

i. Aborted and auto-rewarded trials are excluded. Trials where the duration yields fewer than 1 time bin are skipped. Sessions with fewer than 2 valid trials are excluded. Sessions where eye tracking data is entirely invalid (all blinks) are excluded entirely. Passive sessions are excluded.

ii.
```python
keep = (
    (trials["go"].astype(bool) | trials["catch"].astype(bool))
    & ~trials["aborted"].astype(bool)
    & ~trials["auto_rewarded"].astype(bool)
)
...
if len(trials) < 2:
    raise ValueError(f"only {len(trials)} retained trials")
...
if n_bins < 1:
    continue
...
if len(trial_centers) < 2:
    raise ValueError(f"only {len(trial_centers)} nonempty retained trials")
```

iii. The agent excluded passive sessions because "Passive replay sessions have synthetic go/catch labels but no genuine choice/outcome." Sessions with entirely invalid pupil data were excluded because "Missing pupil data prevents constructing a required decoder output; such a session is unusable rather than safely imputable."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `events` table (inferred calcium events) from the Allen SDK, not from `dff_traces` (dF/F).

ii.
```python
events = np.vstack(dataset.events["events"].to_numpy()).astype(np.float32, copy=False)
```

iii. The agent reasoned: "The neural signal is the inferred calcium-event trace, not dF/F. This avoids carrying the slow GCaMP decay into later stimulus epochs, as in the paper."

## 2-b. How is the `neural` data processed?

i. The events are stacked into a (n_neurons, n_timepoints) matrix. For each trial, the nearest ophys timestamp to each 100ms bin center is found, and the corresponding event values are selected. No additional normalization or filtering is applied.

ii.
```python
events = np.vstack(dataset.events["events"].to_numpy()).astype(np.float32, copy=False)
...
nearest = _nearest_indices(ophys_times, centers)
neural_trials.append(events[:, nearest].astype(np.float32, copy=False))
```

iii. The agent relied on the Allen SDK's pre-existing QC: "NWB files and cells have already passed Allen ophys/ROI QC." The nearest-neighbor resampling was chosen because the inferred events are sparse point-like signals where interpolation would be inappropriate.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied beyond what the Allen SDK already provides. All neurons present in the `events` table are included.

ii. N/A - no filtering code exists.

iii. The agent stated: "NWB files and cells have already passed Allen ophys/ROI QC." The agent also noted that silent trials (all-zero neural activity) were expected for sparse inferred-event traces and should be retained.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial start time. For each trial, 100ms bin centers are computed starting 50ms after `start_time`. The nearest ophys timestamp to each bin center is found, and the corresponding neural data is selected.

ii.
```python
centers = float(row["start_time"]) + (np.arange(n_bins) + 0.5) * BIN_SEC
...
nearest = _nearest_indices(ophys_times, centers)
neural_trials.append(events[:, nearest].astype(np.float32, copy=False))
```

iii. The metadata records: "experiment-defined trial start on the synchronized ophys clock; 100 ms bin centers begin 50 ms after trial start." The agent aligned to `start_time` (trial start) as the temporal alignment event, with `off_start: 0.0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is rebinned to a 100ms common grid (10 Hz). This is a departure from the native ophys frame rate (~11 Hz for Multiscope, ~31 Hz for Scientifica).

ii.
```python
BIN_SEC = 0.100
...
n_bins = int(np.floor(duration / BIN_SEC))
centers = float(row["start_time"]) + (np.arange(n_bins) + 0.5) * BIN_SEC
```

iii. The agent explained: "A common grid is required because Scientifica and Multiscope planes were acquired at about 31 Hz and 11 Hz, respectively. Ten Hz is still finer than the roughly 200 ms effective resolution of the inferred events." The metadata records `time_bin_size: 100.0` (ms).

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, using the `image_name`, `start_time`, and `end_time` columns. This is different from the reference which uses `initial_image_name` and `change_image_name` from the trials table.

ii.
```python
presentations = dataset.stimulus_presentations
...
for _, stim in overlap.iterrows():
    name = stim["image_name"]
    if pd.notna(name) and name != "omitted":
        on = (centers >= float(stim["start_time"])) & (
            centers < float(stim["end_time"])
        )
        image[on] = IMAGE_TO_CLASS[name]
```

iii. The agent used the stimulus_presentations table to get fine-grained, time-varying image identity at each time bin. This allows capturing the gray inter-stimulus intervals and omitted flashes as a separate "gray" category (class 0).

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes using a fixed global mapping of 17 categories: "gray" (class 0) plus 16 natural images. Inter-stimulus gray intervals and omitted flashes are both mapped to the "gray" class. The mapping is hardcoded rather than discovered from data.

ii.
```python
IMAGE_NAMES = [
    "gray",
    "im000", "im031", "im035", "im045", "im054", "im073", "im075", "im106",
    "im061", "im062", "im063", "im065", "im066", "im069", "im077", "im085",
]
IMAGE_TO_CLASS = {name: idx for idx, name in enumerate(IMAGE_NAMES)}
...
image = np.zeros(len(centers), dtype=np.uint8)  # default to gray
for _, stim in overlap.iterrows():
    name = stim["image_name"]
    if pd.notna(name) and name != "omitted":
        image[on] = IMAGE_TO_CLASS[name]
```

iii. The agent hardcoded the image names based on the two counterbalanced image sets used in the Visual Behavior task. The "gray" class captures both inter-stimulus intervals and omitted flashes.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the same 100ms bin centers used for neural data. Each bin is labeled with the image being displayed at that time (based on stimulus presentation start/end times), or "gray" if no image is on screen.

ii.
```python
def _stimulus_labels(centers, presentations):
    image = np.zeros(len(centers), dtype=np.uint8)
    overlap = presentations[
        (presentations["start_time"] < centers[-1] + BIN_SEC / 2)
        & (presentations["end_time"] > centers[0] - BIN_SEC / 2)
    ]
    for _, stim in overlap.iterrows():
        ...
        on = (centers >= float(stim["start_time"])) & (
            centers < float(stim["end_time"])
        )
        image[on] = IMAGE_TO_CLASS[name]
```

iii. By computing image identity at the same bin centers as neural data, perfect temporal alignment is guaranteed.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the `stimulus_presentations` table and the stimulus `start_time`.

ii.
```python
if bool(stim.get("is_change", False)):
    idx = int(np.searchsorted(centers, float(stim["start_time"]), side="left"))
    if idx < len(changed):
        changed[idx] = 1
```

iii. The agent used `is_change` from stimulus_presentations, which marks the specific stimulus flash where the image identity changed.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary time-varying variable. Value 1 is placed at the single time bin whose center falls at or just after the change onset; all other bins are 0. This is an instantaneous event marker, not a window.

ii.
```python
changed = np.zeros(len(centers), dtype=np.uint8)
...
if bool(stim.get("is_change", False)):
    idx = int(np.searchsorted(centers, float(stim["start_time"]), side="left"))
    if idx < len(changed):
        changed[idx] = 1
```

iii. The agent described this as: "A change is an event, not a 250 ms state: label the first common time bin whose center is on or after its display-lag-corrected onset."

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 = no change, 1 = change). No thresholding is needed.

ii.
```python
changed = np.zeros(len(centers), dtype=np.uint8)
# ... set to 1 at change bin
```

iii. The binary nature of image change makes thresholding unnecessary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity — computed at the same 100ms bin centers as the neural data.

ii. See 4-a/4-b code.

iii. Same bin-center alignment as all other variables.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, using the `timestamps` and `speed` columns.

ii.
```python
running = dataset.running_speed
...
run_values = _interp_valid(
    running["timestamps"].to_numpy(dtype=np.float64),
    running["speed"].to_numpy(dtype=np.float64),
    all_centers,
)
```

iii. The `running_speed` attribute is the SDK's standard interface for locomotion data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated to the 100ms bin centers (after removing NaN/inf samples), then discretized into 5 percentile-based bins computed *per session* using the 20th/40th/60th/80th percentiles.

ii.
```python
run_values = _interp_valid(
    running["timestamps"].to_numpy(dtype=np.float64),
    running["speed"].to_numpy(dtype=np.float64),
    all_centers,
)
run_class = _percentile_classes(run_values)
...
def _percentile_classes(values):
    cuts = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
    return np.searchsorted(cuts, values, side="right").astype(np.uint8)
```

iii. The agent used per-session percentiles to ensure balanced class distributions within each session. The `_interp_valid` function removes NaN/inf before interpolation for robustness.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using the 20th, 40th, 60th, and 80th percentiles of the session's running speed values. The `searchsorted` with `side="right"` ensures deterministic handling of ties.

ii.
```python
def _percentile_classes(values):
    cuts = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
    return np.searchsorted(cuts, values, side="right").astype(np.uint8)
```

iii. Per-session percentile binning ensures approximately 20% of samples in each bin per session.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same 100ms bin centers used for neural data, ensuring exact temporal alignment.

ii.
```python
all_centers = np.concatenate(trial_centers)
run_values = _interp_valid(
    running["timestamps"].to_numpy(dtype=np.float64),
    running["speed"].to_numpy(dtype=np.float64),
    all_centers,
)
```

iii. By interpolating to the same time grid, all data streams share the same temporal alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from the `pupil_area` column in `dataset.eye_tracking`, converted to diameter via `2 * sqrt(area / pi)`.

ii.
```python
eye = dataset.eye_tracking
pupil_area = eye["pupil_area"].to_numpy(dtype=np.float64)
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
```

iii. The agent noted: "SDK pupil_area is pi * max(ellipse radius)^2 after blink/outlier removal; converting it to a diameter is monotonic (and therefore preserves the requested percentile classes)."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area is converted to diameter, then linearly interpolated to 100ms bin centers (after removing NaN/inf for blink filtering), then discretized into 5 per-session percentile bins.

ii.
```python
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
pupil_values = _interp_valid(
    eye["timestamps"].to_numpy(dtype=np.float64),
    pupil_diameter,
    all_centers,
)
pupil_class = _percentile_classes(pupil_values)
```

iii. The `_interp_valid` function implicitly handles blink removal by filtering out NaN/inf values before interpolation. The SDK's blink detection sets affected samples to NaN in `pupil_area`.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 bins using per-session 20/40/60/80 percentiles.

ii.
```python
pupil_class = _percentile_classes(pupil_values)
```

iii. Same per-session percentile approach as running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — interpolated to the same 100ms bin centers as neural data.

ii.
```python
pupil_values = _interp_valid(
    eye["timestamps"].to_numpy(dtype=np.float64),
    pupil_diameter,
    all_centers,
)
```

iii. Same bin-center alignment as all other variables.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the `hit`, `miss`, `false_alarm`, and `correct_reject` boolean columns in the trials table.

ii.
```python
OUTCOME_NAMES = ["hit", "miss", "false_alarm", "correct_reject"]

def _trial_outcome(row):
    flags = np.asarray([bool(row[name]) for name in OUTCOME_NAMES])
    if flags.sum() != 1:
        raise ValueError(...)
    return int(np.flatnonzero(flags)[0])
```

iii. The agent enforces that exactly one outcome flag is set per trial, raising an error otherwise. This provides a stronger validation than the reference solution's fallback to "other".

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome is mapped to an integer code (0-3) corresponding to hit/miss/false_alarm/correct_reject. The code is constant across all time bins within a trial.

ii.
```python
outcome_row = np.full(n_bins, outcome, dtype=np.uint8)
output_trials.append(
    np.vstack([
        image, change,
        run_class[cursor:cursor + n_bins],
        pupil_class[cursor:cursor + n_bins],
        outcome_row,
    ]).astype(np.uint8, copy=False)
)
```

iii. The mapping order matches `OUTCOME_NAMES = ["hit", "miss", "false_alarm", "correct_reject"]`.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Failed sessions**: If `convert_experiment` throws an exception, the session is skipped and the reason is recorded in metadata.
- **Missing eye tracking**: Sessions without valid eye tracking data are excluded entirely (not imputed).
- **NaN/inf values**: The `_interp_valid` function removes NaN/inf samples before interpolation.
- **Short trials**: Trials with fewer than 1 time bin are skipped.
- **Few trials**: Sessions with fewer than 2 valid trials raise an error and are skipped.

ii.
```python
try:
    converted.append(convert_experiment(int(experiment_id)))
except Exception as exc:
    skipped.append({
        "ophys_experiment_id": int(experiment_id),
        "reason": f"{type(exc).__name__}: {exc}",
    })
...
valid = np.isfinite(times) & np.isfinite(values)
if valid.sum() < 2:
    raise ValueError("fewer than two finite samples available for interpolation")
```

iii. The agent chose to exclude rather than impute missing data: "Missing pupil data prevents constructing a required decoder output; such a session is unusable rather than safely imputable." Skipped sessions and reasons are preserved in metadata for transparency.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each NWB file via `BehaviorOphysExperiment.from_nwb_path()`, which reads large neural and behavioral data arrays from disk.

ii. N/A

iii. Each NWB file contains full-session neural traces, running speed, eye tracking, stimulus presentations, and trial data. Loading and parsing these large files dominates runtime.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `_stimulus_labels` function iterates over stimulus presentations with a Python for-loop. This could potentially be vectorized using interval-based operations. The `_trial_outcome` function also uses a loop over outcome names.

ii.
```python
for _, stim in overlap.iterrows():
    name = stim["image_name"]
    ...
    on = (centers >= float(stim["start_time"])) & (centers < float(stim["end_time"]))
    image[on] = IMAGE_TO_CLASS[name]
```

iii. The stimulus presentation loop iterates over ~300-400 presentations per trial window, computing boolean masks for each. This could be replaced with a vectorized interval assignment. However, data loading likely dominates runtime.

## 9-c. What processing does the code repeat multiple times?

i. Running speed and pupil diameter are interpolated to all trial bin centers at once (concatenated), then split back per trial using a cursor. This is efficient — no processing is repeated.

ii.
```python
all_centers = np.concatenate(trial_centers)
run_values = _interp_valid(..., all_centers)
...
run_class[cursor:cursor + n_bins]  # slice per trial
cursor += n_bins
```

iii. The agent's approach of concatenating all bin centers and interpolating once is efficient and avoids redundant computation.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The conversion from `pupil_area` to `pupil_diameter` via `2 * sqrt(area / pi)` is unnecessary because the square root is a monotonic transformation that preserves percentile bin assignments. The same bins would result from using area directly.

ii.
```python
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
```

iii. The agent acknowledged this: "converting it to a diameter is monotonic (and therefore preserves the requested percentile classes)." While not wrong, it's an unnecessary computation step.
