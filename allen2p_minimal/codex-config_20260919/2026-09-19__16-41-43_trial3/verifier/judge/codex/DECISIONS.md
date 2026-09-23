# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local Allen Visual Behavior manifest CSV directly from `/app/data`, enumerates the NWB files actually present on disk, intersects the manifest with those local experiment ids, filters to active non-passive experiments, and then loads each remaining experiment from its NWB file with `BehaviorOphysExperiment.from_nwb_path`. It does not use `VisualBehaviorOphysProjectCache`.

ii.
```python
def _local_experiment_ids() -> set[int]:
    ids: set[int] = set()
    for path in NWB_DIR.glob("behavior_ophys_experiment_*.nwb"):
        match = re.search(r"_(\d+)\.nwb$", path.name)
        if match:
            ids.add(int(match.group(1)))
    return ids

def select_experiments(max_sessions: int | None = None) -> pd.DataFrame:
    table = pd.read_csv(EXPERIMENT_TABLE, index_col="ophys_experiment_id")
    local_ids = _local_experiment_ids()
    table = table.loc[table.index.intersection(local_ids)].copy()
    active = table[table["behavior_type"].eq("active_behavior") & ~table["passive"]]
    return active

def convert_experiment(experiment_id: int) -> dict:
    path = NWB_DIR / f"behavior_ophys_experiment_{experiment_id}.nwb"
    dataset = BehaviorOphysExperiment.from_nwb_path(str(path))
```

iii. In the trajectory, the agent said it would "use every locally provided QC-passed active-behavior experiment" and exclude passive replays, and it explicitly chose direct NWB loading from the local data bundle rather than using the Allen cache (step 29).

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the unique `mouse_id` values found in the converted experiments' metadata. The final `subjects` list is the sorted set of mouse ids, and each converted session stores a `mouse_id` used to build `subject_idx`.

ii.
```python
meta = dataset.metadata
result = {
    "mouse_id": str(meta["mouse_id"]),
    ...
}

subjects = sorted({session["mouse_id"] for session in converted})
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
"subject_idx": np.asarray(
    [subject_lookup[session["mouse_id"]] for session in converted], dtype=np.int64
),
```

iii. The trajectory does not give a separate argument for subject splitting beyond using experiment metadata throughout. This is the natural organization once the agent decided each experiment would be converted independently.

## 1-c. How are the data split into sessions?

i. The agent treats each `ophys_experiment_id` NWB file, i.e. each imaging plane, as one decoder session. It does not group multiple experiments that share an `ophys_session_id`.

ii.
```python
def convert_experiment(experiment_id: int) -> dict:
    path = NWB_DIR / f"behavior_ophys_experiment_{experiment_id}.nwb"
    dataset = BehaviorOphysExperiment.from_nwb_path(str(path))
    ...
    result = {
        "session_info": {
            "ophys_experiment_id": int(experiment_id),
            "ophys_session_id": int(meta["ophys_session_id"]),
            ...
        },
    }

for number, experiment_id in enumerate(experiments.index, start=1):
    converted.append(convert_experiment(int(experiment_id)))
```

iii. The agent explicitly justified this in the trajectory: it would "treat each imaging plane as a decoder session as in the paper’s plane-wise decoding" and retain Multiscope planes separately with their own synchronized timestamps (steps 29 and 48).

## 1-d. How are the data split into trials?

i. Trials come from `dataset.trials`. For each retained trial row, the agent uses the experiment-defined `start_time` and `stop_time`, computes a sequence of 100 ms bin centers within that interval, and makes one converted trial per row.

ii.
```python
trials = dataset.trials
...
trial_centers: list[np.ndarray] = []
for _, row in trials.iterrows():
    duration = float(row["stop_time"] - row["start_time"])
    n_bins = int(np.floor(duration / BIN_SEC))
    if n_bins < 1:
        continue
    centers = float(row["start_time"]) + (np.arange(n_bins) + 0.5) * BIN_SEC
    trial_centers.append(centers)
```

iii. The trajectory says the NWB files contain the experiment-defined trial flags and that the converted outputs were aligned to trial bins on a common 100 ms grid (steps 17 and 29). The agent kept the Allen trial definitions but changed the temporal sampling within each trial.

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps only trials marked as `go` or `catch`, excludes `aborted` and `auto_rewarded` trials, discards trials shorter than one 100 ms bin, and rejects experiments with fewer than two retained nonempty trials.

ii.
```python
keep = (
    (trials["go"].astype(bool) | trials["catch"].astype(bool))
    & ~trials["aborted"].astype(bool)
    & ~trials["auto_rewarded"].astype(bool)
)
trials = trials.loc[keep].copy()
if len(trials) < 2:
    raise ValueError(f"only {len(trials)} retained trials")

...
n_bins = int(np.floor(duration / BIN_SEC))
if n_bins < 1:
    continue
...
if len(trial_centers) < 2:
    raise ValueError(f"only {len(trial_centers)} nonempty retained trials")
```

iii. The trajectory repeatedly says the agent retained "Go/Catch trials" while excluding aborted and auto-rewarded trials, because passive or unusable sessions would distort the requested outputs (steps 29, 39, and 68).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The agent derives `neural` from the AllenSDK inferred calcium-event matrix, specifically `dataset.events["events"]`, not from `dff_traces`.

ii.
```python
dataset = BehaviorOphysExperiment.from_nwb_path(str(path))

events = np.vstack(dataset.events["events"].to_numpy()).astype(np.float32, copy=False)
ophys_times = np.asarray(dataset.ophys_timestamps, dtype=np.float64)
```

iii. The trajectory explicitly states that the agent chose "detected calcium events" and not dF/F because this would avoid carrying slow GCaMP decay into later epochs (step 29). Earlier it had verified that a representative NWB exposed both `events` and `dff_traces` (step 16).

## 2-b. How is the `neural` data processed?

i. The event matrix is used largely as-is. The main processing is to select, for each 100 ms trial bin center, the nearest ophys timestamp and take the event values from that frame. Because each imaging plane is treated as a separate session, there is no across-plane stacking within a session.

ii.
```python
def _nearest_indices(source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source_times, target_times, side="left")
    right = np.clip(right, 0, len(source_times) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(target_times - source_times[left]) <= np.abs(
        source_times[right] - target_times
    )
    return np.where(choose_left, left, right)

...
nearest = _nearest_indices(ophys_times, centers)
neural_trials.append(events[:, nearest].astype(np.float32, copy=False))
```

iii. The trajectory justification is the common-bin-size requirement: the agent said it would use a common 100 ms grid so both ~31 Hz and ~11 Hz recordings satisfy the decoder format while keeping plane-wise sessions separate (step 29).

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent applies no extra neuron-level quality filter beyond using the QC-passed NWB content. It does, however, reject entire experiments that contain zero cells.

ii.
```python
events = np.vstack(dataset.events["events"].to_numpy()).astype(np.float32, copy=False)
...
if events.shape[0] == 0:
    raise ValueError("no QC-passed cells")
```

iii. The script header says the "NWB files and cells have already passed Allen ophys/ROI QC" and the trajectory repeatedly refers to "QC-passed" experiments and cells (steps 17 and 29), so the agent treated the Allen preprocessing as sufficient.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The per-trial neural data are aligned to trial start. For each retained trial, the agent creates 100 ms bin centers beginning 50 ms after `start_time` and samples the nearest ophys frame at each bin center.

ii.
```python
centers = float(row["start_time"]) + (np.arange(n_bins) + 0.5) * BIN_SEC
...
nearest = _nearest_indices(ophys_times, centers)
neural_trials.append(events[:, nearest].astype(np.float32, copy=False))
...
"temporal_alignment_event": (
    "experiment-defined trial start on the synchronized ophys clock; "
    "100 ms bin centers begin 50 ms after trial start"
),
```

iii. The trajectory says all streams would be placed on "a common 100 ms grid anchored at each trial start" to reconcile different acquisition frame rates (step 29).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 100 ms bins. Yes, temporal rebinning/resampling is applied: neural activity is downsampled by nearest-neighbor selection at 100 ms centers.

ii.
```python
BIN_SEC = 0.100
...
n_bins = int(np.floor(duration / BIN_SEC))
centers = float(row["start_time"]) + (np.arange(n_bins) + 0.5) * BIN_SEC
...
"time_bin_size": BIN_SEC * 1000.0,
"resampling": (
    "nearest ophys event sample at each 100 ms center; linear interpolation "
    "of running speed and blink-filtered pupil diameter"
),
```

iii. The agent explicitly justified 100 ms bins in the trajectory as a common resolution suitable for both acquisition regimes (~31 Hz and ~11 Hz) and still finer than the effective event resolution (step 29).

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. `Image identity` is derived from `dataset.stimulus_presentations`, using the per-stimulus `image_name`, `start_time`, and `end_time` fields over the trial's 100 ms bin centers.

ii.
```python
presentations = dataset.stimulus_presentations
...
overlap = presentations[
    (presentations["start_time"] < centers[-1] + BIN_SEC / 2)
    & (presentations["end_time"] > centers[0] - BIN_SEC / 2)
]
for _, stim in overlap.iterrows():
    name = stim["image_name"]
    ...
    on = (centers >= float(stim["start_time"])) & (
        centers < float(stim["end_time"])
    )
    image[on] = IMAGE_TO_CLASS[name]
```

iii. The trajectory shows the agent inspected `stimulus_presentations` in the NWB and decided to use the flashed-stimulus table rather than the trial table for image labeling (steps 16 and 29).

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The agent uses a fixed global mapping `IMAGE_NAMES -> IMAGE_TO_CLASS`. It labels only actual image flashes, while gray periods and omitted flashes are assigned class 0 (`"gray"`). This makes image identity time-varying at 100 ms resolution.

ii.
```python
IMAGE_NAMES = [
    "gray",
    "im000", "im031", "im035", "im045", "im054", "im073", "im075", "im106",
    "im061", "im062", "im063", "im065", "im066", "im069", "im077", "im085",
]
IMAGE_TO_CLASS = {name: idx for idx, name in enumerate(IMAGE_NAMES)}

image = np.zeros(len(centers), dtype=np.uint8)
...
if pd.notna(name) and name != "omitted":
    image[on] = IMAGE_TO_CLASS[name]
```

iii. The header comment explains the rationale: the two counterbalanced image sets are disjoint and class zero should represent the gray/no-image state, including omitted flashes. The trajectory later notes that both image sets were seen without label mismatches (step 58).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same 100 ms bin centers used for the neural data, based on overlap between each bin center and the flashed-stimulus intervals.

ii.
```python
for centers, outcome in zip(trial_centers, outcomes):
    nearest = _nearest_indices(ophys_times, centers)
    neural_trials.append(events[:, nearest].astype(np.float32, copy=False))

    image, change = _stimulus_labels(centers, presentations)
    ...
    output_trials.append(
        np.vstack(
            [image, change, ...]
        ).astype(np.uint8, copy=False)
    )
```

iii. The trajectory says "all streams are placed on a common 100 ms grid anchored at each trial start" so that outputs and neural activity share the same bins (step 29).

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. `Image change` is derived from `dataset.stimulus_presentations`, specifically each overlapping row's `is_change` flag and `start_time`.

ii.
```python
for _, stim in overlap.iterrows():
    ...
    if bool(stim.get("is_change", False)):
        idx = int(np.searchsorted(centers, float(stim["start_time"]), side="left"))
        if idx < len(changed):
            changed[idx] = 1
```

iii. The trajectory shows the agent inspected the NWB stimulus table and chose to use its explicit change annotations rather than deriving change only from the trial table (steps 16 and 29).

## 4-b. What processing is involved in computing `output` *Image change*?

i. The agent treats image change as an event, not a state. It scans overlapping stimulus presentations and, for each row flagged `is_change`, sets exactly one time bin to 1: the first bin whose center is on or after the change onset.

ii.
```python
changed = np.zeros(len(centers), dtype=np.uint8)
...
if bool(stim.get("is_change", False)):
    idx = int(np.searchsorted(centers, float(stim["start_time"]), side="left"))
    if idx < len(changed):
        changed[idx] = 1
```

iii. The code comment gives the main rationale: "A change is an event, not a 250 ms state." The trajectory also emphasizes creating categorical outputs aligned to the 100 ms grid (steps 29 and 32).

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is binary: `0` for `no_change` and `1` for `change`.

ii.
```python
changed = np.zeros(len(centers), dtype=np.uint8)
...
"output_values": [
    IMAGE_NAMES,
    ["no_change", "change"],
    ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
    ...
],
```

iii. The trajectory does not add further discussion here. The binary choice follows directly from the task requirement that image change be a two-category time-varying output.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image-change labels are produced on the same 100 ms bin centers used for neural sampling, so the binary event array is frame-aligned with the converted neural matrix after temporal resampling.

ii.
```python
for centers, outcome in zip(trial_centers, outcomes):
    nearest = _nearest_indices(ophys_times, centers)
    neural_trials.append(events[:, nearest].astype(np.float32, copy=False))

    image, change = _stimulus_labels(centers, presentations)
    output_trials.append(
        np.vstack([image, change, ...]).astype(np.uint8, copy=False)
    )
```

iii. The same trajectory rationale applies as for the other outputs: one common 100 ms grid for all modalities (step 29).

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `dataset.running_speed`, using its `timestamps` and `speed` columns.

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

iii. The trajectory explicitly notes that the NWB contains a 60 Hz running stream, and the agent kept that stream as the source for the running-speed output (step 17).

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to all 100 ms trial bin centers in the experiment, then discretized into five percentile classes computed separately within that experiment.

ii.
```python
def _interp_valid(times: np.ndarray, values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    valid = np.isfinite(times) & np.isfinite(values)
    ...
    return np.interp(targets, x, y).astype(np.float32, copy=False)

def _percentile_classes(values: np.ndarray) -> np.ndarray:
    cuts = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
    return np.searchsorted(cuts, values, side="right").astype(np.uint8)

run_values = _interp_valid(..., all_centers)
run_class = _percentile_classes(run_values)
```

iii. The trajectory says the percentile outputs were "exactly balanced within sessions" and that a common 100 ms grid was the organizing time base (step 32). That reflects the decision to compute percentile classes per experiment rather than globally.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The categories are the five within-session percentile bins implied by the 20/40/60/80% quantile cut points: class 0 through class 4, labeled in metadata as `0-20%`, `20-40%`, `40-60%`, `60-80%`, `80-100%`.

ii.
```python
cuts = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
return np.searchsorted(cuts, values, side="right").astype(np.uint8)

"output_values": [
    IMAGE_NAMES,
    ["no_change", "change"],
    ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
    ...
],
"percentile_scope": "separately within each session over retained trial bins",
```

iii. The trajectory justification is the balanced-class goal: the agent highlighted that the percentile outputs were balanced within sessions after validation (step 32).

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. The running-speed values are interpolated directly to the same 100 ms trial bin centers used for neural sampling, then the corresponding class labels are sliced trial by trial using a running cursor.

ii.
```python
all_centers = np.concatenate(trial_centers)
run_values = _interp_valid(..., all_centers)
run_class = _percentile_classes(run_values)

for centers, outcome in zip(trial_centers, outcomes):
    n_bins = len(centers)
    nearest = _nearest_indices(ophys_times, centers)
    ...
    output_trials.append(
        np.vstack(
            [
                image,
                change,
                run_class[cursor : cursor + n_bins],
                ...
            ]
        ).astype(np.uint8, copy=False)
    )
    cursor += n_bins
```

iii. The trajectory repeatedly frames the solution as one common 100 ms grid for all streams (steps 29 and 32).

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`, using the processed `pupil_area` column and converting it to a diameter. The code does not explicitly filter `likely_blink`; instead it relies on the processed `pupil_area` values and finite-value checks.

ii.
```python
eye = dataset.eye_tracking
if eye is None:
    raise ValueError("eye tracking is unavailable")

pupil_area = eye["pupil_area"].to_numpy(dtype=np.float64)
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
```

iii. In the trajectory, the agent inspected AllenSDK eye-tracking processing code and saw that `pupil_area` is produced after blink/outlier handling, then chose to convert area to diameter as a monotonic transformation (step 18). It also said sessions lacking usable pupil data should be skipped rather than imputed (steps 32 and 39).

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The processed pupil-area signal is converted to a diameter, linearly interpolated to all 100 ms trial bin centers, and discretized into five percentile classes computed separately within each experiment. If too few finite samples remain, the whole experiment is skipped.

ii.
```python
pupil_area = eye["pupil_area"].to_numpy(dtype=np.float64)
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)

pupil_values = _interp_valid(
    eye["timestamps"].to_numpy(dtype=np.float64),
    pupil_diameter,
    all_centers,
)
pupil_class = _percentile_classes(pupil_values)
```

iii. The trajectory justification is explicit: any session lacking usable pupil data would be "explicitly skipped and recorded in metadata rather than silently imputed" (step 32), and later such exclusions are described as necessary to avoid inventing a pupil target (step 39).

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The thresholding is the same as running speed: five within-session percentile bins using the 20/40/60/80% cut points, represented as integer classes 0-4.

ii.
```python
def _percentile_classes(values: np.ndarray) -> np.ndarray:
    cuts = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
    return np.searchsorted(cuts, values, side="right").astype(np.uint8)

pupil_class = _percentile_classes(pupil_values)
...
"percentile_scope": "separately within each session over retained trial bins",
```

iii. The trajectory says the percentile outputs were balanced within sessions (step 32), which is the direct consequence of this per-session thresholding rule.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil values are interpolated to the same 100 ms trial bin centers used for neural sampling, and the resulting percentile classes are then assigned to each trial's bins via the shared cursor over `all_centers`.

ii.
```python
all_centers = np.concatenate(trial_centers)
pupil_values = _interp_valid(
    eye["timestamps"].to_numpy(dtype=np.float64),
    pupil_diameter,
    all_centers,
)
pupil_class = _percentile_classes(pupil_values)

for centers, outcome in zip(trial_centers, outcomes):
    ...
    output_trials.append(
        np.vstack(
            [
                image,
                change,
                run_class[cursor : cursor + n_bins],
                pupil_class[cursor : cursor + n_bins],
                outcome_row,
            ]
        ).astype(np.uint8, copy=False)
    )
```

iii. The same trajectory explanation applies: all streams were intentionally forced onto a shared 100 ms trial-centered grid (step 29).

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the four boolean trial columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
OUTCOME_NAMES = ["hit", "miss", "false_alarm", "correct_reject"]

def _trial_outcome(row: pd.Series) -> int:
    flags = np.asarray([bool(row[name]) for name in OUTCOME_NAMES])
    if flags.sum() != 1:
        raise ValueError(...)
    return int(np.flatnonzero(flags)[0])
```

iii. The trajectory says the agent chose "four mutually exclusive outcomes" and rejected passive sessions specifically because their apparent outcomes are synthetic rather than genuine behavior (steps 22 and 29).

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The agent maps the four outcome flags to integer codes 0-3 in the fixed order `["hit", "miss", "false_alarm", "correct_reject"]`, requires exactly one flag to be true, and repeats that code across every time bin in the trial.

ii.
```python
def _trial_outcome(row: pd.Series) -> int:
    flags = np.asarray([bool(row[name]) for name in OUTCOME_NAMES])
    if flags.sum() != 1:
        raise ValueError(...)
    return int(np.flatnonzero(flags)[0])

...
outcome_row = np.full(n_bins, outcome, dtype=np.uint8)
output_trials.append(
    np.vstack([image, change, ..., outcome_row]).astype(np.uint8, copy=False)
)
```

iii. The trajectory explicitly mentions the choice of "four mutually exclusive outcomes" (step 29), and the passive-session inspection in step 22 was used to justify trusting those four flags only in active sessions.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles unusable data conservatively by skipping whole experiments when a required stream cannot support the requested outputs. Examples include missing eye tracking, fewer than two finite interpolable samples for running or pupil, zero cells, or fewer than two retained trials. It does not impute missing pupil values, and `np.interp` uses endpoint values at synchronization edges.

ii.
```python
if eye is None:
    raise ValueError("eye tracking is unavailable")

if valid.sum() < 2:
    raise ValueError("fewer than two finite samples available for interpolation")

if events.shape[0] == 0:
    raise ValueError("no QC-passed cells")
...
except Exception as exc:
    skipped.append(
        {
            "ophys_experiment_id": int(experiment_id),
            "reason": f"{type(exc).__name__}: {exc}",
        }
    )
```

iii. The trajectory emphasizes this policy several times: sessions lacking usable pupil data should be skipped and recorded rather than imputed (steps 32, 39, 58, and 80). The code comments also say such sessions are "unusable rather than safely imputable."

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive parts are loading each large NWB file with `BehaviorOphysExperiment.from_nwb_path`, materializing the event/running/eye/stimulus tables, and then converting every experiment one by one across the full manifest.

ii.
```python
for number, experiment_id in enumerate(experiments.index, start=1):
    print(f"[{number}/{total}] converting experiment {experiment_id}", flush=True)
    try:
        converted.append(convert_experiment(int(experiment_id)))
```

iii. The trajectory repeatedly refers to "many large NWB files" and long full-dataset passes over 202 experiments, which shows that session loading/conversion is the dominant cost (steps 9, 39, 48, 58, and 62).

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities are the per-trial loop in `convert_experiment` and the nested loop over overlapping `stimulus_presentations` rows inside `_stimulus_labels`. The percentile slicing with `cursor` is already vectorized once the full interpolated arrays are built.

ii.
```python
for _, row in trials.iterrows():
    duration = float(row["stop_time"] - row["start_time"])
    ...
    trial_centers.append(centers)
    outcomes.append(_trial_outcome(row))

for _, stim in overlap.iterrows():
    ...
    if bool(stim.get("is_change", False)):
        idx = int(np.searchsorted(centers, float(stim["start_time"]), side="left"))
        ...
```

iii. The trajectory does not explicitly discuss vectorization. This conclusion comes from the structure of the written code after the agent chose readability plus experiment-by-experiment processing.

## 9-c. What processing does the code repeat multiple times?

i. The code repeatedly scans overlapping stimulus presentations for every trial, repeatedly constructs per-trial bin-center arrays, and repeatedly runs `gc.collect()` after each experiment. It also recomputes within-session percentile thresholds separately for running and pupil in every experiment.

ii.
```python
for centers, outcome in zip(trial_centers, outcomes):
    ...
    image, change = _stimulus_labels(centers, presentations)
    ...

def _stimulus_labels(centers: np.ndarray, presentations: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    overlap = presentations[...]
    for _, stim in overlap.iterrows():
        ...

run_class = _percentile_classes(run_values)
pupil_class = _percentile_classes(pupil_values)
...
gc.collect()
```

iii. The trajectory does not call out this repetition directly. It follows from the agent's decision to convert each experiment independently and label each trial from the stimulus table on demand.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Two clear examples are: converting `pupil_area` to diameter even though only percentile rank is used later, and storing extensive `session_info` / `skipped_sessions` metadata that the decoder does not use. The downstream decoder only consumes the discretized outputs and neural matrices.

ii.
```python
pupil_area = eye["pupil_area"].to_numpy(dtype=np.float64)
pupil_diameter = 2.0 * np.sqrt(pupil_area / np.pi)
...
"session_info": [session["session_info"] for session in converted],
"skipped_sessions": skipped,
```

iii. The trajectory justifies the metadata retention as documentation of skipped sessions (steps 32 and 80), but it does not claim that this extra metadata is needed by the decoder itself.
