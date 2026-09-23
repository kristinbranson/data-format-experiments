# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the Allen SDK cache loader. It enumerates local NWB files under `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments`, joins them to `project_metadata/ophys_experiment_table.csv`, drops passive experiments, drops experiments lacking eye-tracking, then reopens each retained NWB with `h5py` to extract trials and signals.

ii. 
```python
def discover(data_root: Path):
    nwb_dir = data_root / "behavior_ophys_experiments"
    table_path = data_root / "project_metadata/ophys_experiment_table.csv"
    table = pd.read_csv(table_path).set_index("ophys_experiment_id")
    paths = {experiment_id(path): path for path in nwb_dir.glob("*.nwb")}
    ...
    supplied = table.loc[sorted(paths)].copy()
    active = supplied[~supplied["passive"].astype(bool)].copy()
    ...
    for exp_id in active.index:
        with h5py.File(paths[int(exp_id)], "r") as nwb:
            if "acquisition/EyeTracking/pupil_tracking/area" in nwb:
                usable_ids.append(int(exp_id))

for number, (exp_id, row) in enumerate(selected.iterrows(), start=1):
    neural, inputs, outputs, regions, info = convert_experiment(
        paths[int(exp_id)], row, image_to_index, region_to_index
    )
```

iii. In trajectory step 14, the AI says the supplied files are a deliberate local subset and therefore treats those local NWBs as the dataset. In the script docstring it says direct `h5py` reading was chosen to avoid materializing AllenSDK objects it considered irrelevant to decoding.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values from the retained experiment metadata table.

ii.
```python
subjects = sorted({str(int(x)) for x in selected["mouse_id"]})
subject_to_index = {name: i for i, name in enumerate(subjects)}
...
subject_idx.append(subject_to_index[str(int(row["mouse_id"]))])
```

iii. The trajectory does not give a separate argument for this choice; it follows the Allen metadata schema. Step 49 reports the resulting subject count.

## 1-c. How are the data split into sessions?

i. One `behavior_ophys_experiment` NWB file, i.e. one imaging plane, is treated as one decoder session. The AI does not group experiments that share an `ophys_session_id`.

ii.
```python
"""...
* One released ``behavior_ophys_experiment`` (one imaging plane) is one decoder
  session.
"""
...
for number, (exp_id, row) in enumerate(selected.iterrows(), start=1):
    neural, inputs, outputs, regions, info = convert_experiment(
        paths[int(exp_id)], row, image_to_index, region_to_index
    )
    neural_sessions.append(neural)
```

iii. In trajectory step 14, the AI explicitly says it is treating each NWB experiment as a decoder session because that is the SDK’s experiment abstraction and, in its view, combining planes would require resampling asynchronous plane timestamps and would discard many supplied single-plane recordings.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. For each retained row, the AI uses the full `start_time` to `stop_time` interval and converts it into a variable-length sequence of 100 ms bins anchored at trial start.

ii.
```python
def load_trial_table(group: h5py.Group) -> dict[str, np.ndarray]:
    names = (
        "start_time", "stop_time", "go", "catch", "aborted",
        "auto_rewarded", *OUTCOME_COLUMNS,
    )
    return {name: read_array(group, name) for name in names}

def make_trial_grids(trials: dict[str, np.ndarray], keep: np.ndarray):
    grids = []
    for row in np.flatnonzero(keep):
        start = float(trials["start_time"][row])
        stop = float(trials["stop_time"][row])
        n_bins = int(np.floor((stop - start) / BIN_SECONDS + 1e-9))
        ...
        edges = start + np.arange(n_bins + 1, dtype=np.float64) * BIN_SECONDS
        centers = edges[:-1] + BIN_SECONDS / 2.0
        grids.append((int(row), edges, centers))
```

iii. In trajectory step 19, the AI says it will “retain full valid trials from trial start to stop” while putting all streams onto a common 100 ms grid anchored at trial start.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept when `(go OR catch) AND NOT aborted AND NOT auto_rewarded`. Trials shorter than one 100 ms bin are skipped implicitly by `make_trial_grids`, and experiments with fewer than two retained trials are rejected. The AI does not require `change_time` to be non-null.

ii.
```python
keep = (
    (trials["go"].astype(bool) | trials["catch"].astype(bool))
    & ~trials["aborted"].astype(bool)
    & ~trials["auto_rewarded"].astype(bool)
)
grids = make_trial_grids(trials, keep)
if len(grids) < 2:
    raise ValueError(f"{path.name}: fewer than two retained trials")
...
if n_bins < 1:
    continue
```

iii. The module docstring says these filters were chosen “exactly as requested,” and trajectory step 26 says the smoke test confirmed the intended trial filtering after excluding passive experiments and experiments without eye tracking.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is taken from `processing/ophys/event_detection/data`, with timestamps from `processing/ophys/event_detection/timestamps`, rather than from dF/F.

ii.
```python
ophys = nwb["processing/ophys"]
event_group = ophys["event_detection"]
ophys_timestamps = read_array(event_group, "timestamps", np.float64)
events = read_array(event_group, "data", np.float32)
```

iii. In trajectory step 19, the AI states that the source paper uses detected calcium events rather than dF/F, and the docstring repeats that this converter uses detected calcium-event magnitudes.

## 2-b. How is the `neural` data processed?

i. The AI filters ROIs by `valid_roi`, computes an in-place cumulative sum over event magnitudes, and then for each trial sums event magnitudes into 100 ms bins using the ophys timestamps. It does not merge multiple planes into one session because each experiment is already treated as a session.

ii.
```python
valid_roi = read_array(
    ophys["image_segmentation/cell_specimen_table"], "valid_roi"
).astype(bool)
events = np.ascontiguousarray(events[:, valid_roi], dtype=np.float32)
...
np.cumsum(events, axis=0, dtype=np.float32, out=events)
...
neural = cumulative_bin_sums(events, ophys_timestamps, edges)
...
def cumulative_bin_sums(cumulative_events: np.ndarray,
                        ophys_timestamps: np.ndarray,
                        edges: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(ophys_timestamps, edges, side="left")
    ...
    return (right - left).T
```

iii. Trajectory step 19 says the AI wanted neural matrices to contain binned event magnitudes on a common 100 ms grid while still using ophys timestamps for alignment. The docstring says this satisfies the decoder’s common-bin requirement.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies `valid_roi` and drops experiments with zero valid neurons. It otherwise keeps all remaining neurons, including fully silent event-binned trials.

ii.
```python
valid_roi = read_array(
    ophys["image_segmentation/cell_specimen_table"], "valid_roi"
).astype(bool)
events = np.ascontiguousarray(events[:, valid_roi], dtype=np.float32)
if events.shape[1] == 0:
    raise ValueError(f"{path.name}: no valid neurons")
```

iii. The docstring says released experiments and ROIs have passed Allen QC but that `valid_roi` is applied explicitly anyway. In trajectory step 44, the AI adds that it intentionally retained silent event trials to avoid activity-based trial selection bias.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start. Each trial becomes a set of 100 ms bins whose edges begin at `start_time`, and events are summed into those bins using `event_detection/timestamps`.

ii.
```python
start = float(trials["start_time"][row])
stop = float(trials["stop_time"][row])
edges = start + np.arange(n_bins + 1, dtype=np.float64) * BIN_SECONDS
centers = edges[:-1] + BIN_SECONDS / 2.0
...
neural = cumulative_bin_sums(events, ophys_timestamps, edges)
```

iii. In trajectory step 19, the AI explicitly says all streams will be “on a common 100 ms grid anchored to trial start” and aligned using ophys timestamps rather than frame number.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses fixed 100 ms bins. Yes, the native ophys sampling is rebinned.

ii.
```python
BIN_SECONDS = 0.100
...
"time_bin_size": BIN_SECONDS * 1000.0,
...
edges = start + np.arange(n_bins + 1, dtype=np.float64) * BIN_SECONDS
```

iii. Trajectory steps 19, 31, and 49 all state that the converter uses 100 ms bins so all sessions share a common bin size.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity comes from the NWB stimulus-presentation interval group that contains `image_name`, together with each presentation’s `start_time`, `stop_time`, and `omitted` flag.

ii.
```python
stim = stimulus_group(nwb)
stim_start = read_array(stim, "start_time", np.float64)
stim_stop = read_array(stim, "stop_time", np.float64)
stim_names = decode_strings(read_array(stim, "image_name"))
omitted = read_array(stim, "omitted").astype(bool)
```

iii. The trajectory does not separately discuss this variable, but the docstring says image identity should have an explicit `gray` class outside displayed-image intervals, including omissions.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI first builds a global image vocabulary from all retained experiments, prepending a synthetic `gray` class. For each trial, it initializes every bin as `gray`, finds overlapping stimulus presentations, and writes the displayed image into bins whose centers fall within each presentation’s `start_time`/`stop_time`.

ii.
```python
def collect_image_names(selected: pd.DataFrame, paths: dict[int, Path]):
    names = set()
    ...
    names.discard("omitted")
    return ["gray", *sorted(names)]
...
image_identity = np.full(n_time, gray_index, dtype=np.int16)
overlaps = np.flatnonzero(
    (stim_stop > edges[0]) & (stim_start < edges[-1])
)
for presentation in overlaps:
    if not omitted[presentation] and stim_names[presentation] != "omitted":
        shown = ((centers >= stim_start[presentation])
                 & (centers < stim_stop[presentation]))
        image_identity[shown] = image_to_index[stim_names[presentation]]
```

iii. The docstring explicitly justifies the `gray` class as a way to represent off-image periods and omissions. The trajectory’s alignment justification in step 19 also applies here because the AI wanted all streams on the same 100 ms grid.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is aligned by evaluating each stimulus presentation on the same per-trial 100 ms bin centers used for the neural data.

ii.
```python
centers = edges[:-1] + BIN_SECONDS / 2.0
...
shown = ((centers >= stim_start[presentation])
         & (centers < stim_stop[presentation]))
image_identity[shown] = image_to_index[stim_names[presentation]]
```

iii. Trajectory step 19 says all streams, including outputs, are resampled onto the same 100 ms grid anchored at trial start.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table’s `is_change` field together with each presentation’s `start_time`.

ii.
```python
is_change_raw = read_array(stim, "is_change", np.float64)
is_change = np.isfinite(is_change_raw) & is_change_raw.astype(bool)
...
if is_change[presentation]:
    change_bin = int(np.searchsorted(
        centers, stim_start[presentation], side="left"
    ))
```

iii. The docstring says image change should be one only in the first bin centered at or after a true image-change onset.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each overlapping stimulus presentation marked as a true change, the AI finds the first 100 ms bin whose center is at or after that presentation’s onset and sets only that single bin to 1. All other bins remain 0.

ii.
```python
image_change = np.zeros(n_time, dtype=np.int16)
...
if is_change[presentation]:
    change_bin = int(np.searchsorted(
        centers, stim_start[presentation], side="left"
    ))
    if change_bin < n_time:
        image_change[change_bin] = 1
```

iii. The only explicit justification is in the docstring, which says the target should mark the first bin centered at or after a true image-change onset.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is converted to a binary categorical variable with two labels: `no change` and `change`.

ii.
```python
image_change = np.zeros(n_time, dtype=np.int16)
...
"output_values": [
    image_values,
    ["no change", "change"],
```

iii. The trajectory does not add more detail; this is an implementation-level choice encoded directly in the output vocabulary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is aligned on the same 100 ms trial-relative bin centers used for neural activity and other outputs.

ii.
```python
centers = edges[:-1] + BIN_SECONDS / 2.0
...
change_bin = int(np.searchsorted(
    centers, stim_start[presentation], side="left"
))
```

iii. Trajectory step 19 says every stream is put onto the same 100 ms grid anchored at trial start.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is read from `processing/running/speed`, using its `data` and `timestamps` arrays.

ii.
```python
running_group = nwb["processing/running/speed"]
running_t = read_array(running_group, "timestamps", np.float64)
running_v = read_array(running_group, "data", np.float64)
```

iii. The trajectory does not separately justify the source variable; step 19 only states that running will be aligned onto the same timestamp-based trial grid.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI interpolates running speed onto all trial-bin centers with `np.interp`, concatenates all retained bins within the experiment, computes 20/40/60/80% quantiles within that experiment, and assigns quintile labels 0-4.

ii.
```python
def interpolate_finite(timestamps: np.ndarray, values: np.ndarray,
                       targets: np.ndarray, label: str) -> np.ndarray:
    valid = np.isfinite(timestamps) & np.isfinite(values)
    ...
    return np.interp(targets, timestamps[valid], values[valid]).astype(np.float32)

def quintile(values: np.ndarray) -> np.ndarray:
    edges = np.quantile(values, (0.2, 0.4, 0.6, 0.8))
    return np.searchsorted(edges, values, side="right").astype(np.int16)

running_aligned = interpolate_finite(
    running_t, running_v, all_centers, "running-speed"
)
running_bins = quintile(running_aligned)
```

iii. The docstring says this within-experiment quintiling was chosen to remove between-animal and camera-scale differences while still producing percentile classes. Trajectory step 19 adds that interpolation is done on the common 100 ms grid.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is split into five within-experiment quintile categories, corresponding to 0-20%, 20-40%, 40-60%, 60-80%, and 80-100%.

ii.
```python
def quintile(values: np.ndarray) -> np.ndarray:
    """Return labels 0..4 using empirical 20/40/60/80 percentile edges."""
    edges = np.quantile(values, (0.2, 0.4, 0.6, 0.8))
...
"output_values": [
    image_values,
    ["no change", "change"],
    ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
```

iii. The docstring explicitly says running values are quintiled separately within each experiment.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by interpolating it to the same 100 ms trial-bin centers used for the neural data.

ii.
```python
all_centers = np.concatenate([centers for _, _, centers in grids])
running_aligned = interpolate_finite(
    running_t, running_v, all_centers, "running-speed"
)
...
output[2] = running_bins[offset:offset + n_time]
```

iii. Trajectory step 19 says every stream is aligned to the common timestamp-based 100 ms grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/area`, using eye-tracking timestamps from `acquisition/EyeTracking/eye_tracking/timestamps`.

ii.
```python
pupil_group = nwb["acquisition/EyeTracking/pupil_tracking"]
eye_t = read_array(
    nwb["acquisition/EyeTracking/eye_tracking"], "timestamps", np.float64
)
pupil_area = read_array(pupil_group, "area", np.float64)
```

iii. The docstring says the NWB `area` field is already SDK-processed so blink and outlier frames appear as NaNs, which the AI then handles by interpolation.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI converts pupil area to equivalent circular diameter with `2*sqrt(area/pi)`, interpolates finite values to all trial-bin centers, then computes within-experiment quintile labels across all retained bins.

ii.
```python
pupil_area = read_array(pupil_group, "area", np.float64)
pupil_diameter = 2.0 * np.sqrt(np.maximum(pupil_area, 0.0) / np.pi)
...
pupil_aligned = interpolate_finite(
    eye_t, pupil_diameter, all_centers, "pupil-diameter"
)
pupil_bins = quintile(pupil_aligned)
```

iii. The docstring gives the full rationale: equivalent circular diameter is monotonic with area, blink gaps become NaNs in the processed trace, and within-experiment quintiles are used to normalize scale differences.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is split into five within-experiment quintile categories, with output labels `0-20%` through `80-100%`.

ii.
```python
pupil_bins = quintile(pupil_aligned)
...
"output_values": [
    image_values,
    ["no change", "change"],
    ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
    ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
```

iii. The docstring explicitly says pupil values are quintiled separately within each experiment.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by interpolating it to the same 100 ms trial-bin centers used for neural activity.

ii.
```python
all_centers = np.concatenate([centers for _, _, centers in grids])
pupil_aligned = interpolate_finite(
    eye_t, pupil_diameter, all_centers, "pupil-diameter"
)
...
output[3] = pupil_bins[offset:offset + n_time]
```

iii. Trajectory step 19 says all streams are placed on the same timestamp-based 100 ms grid.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the four trial-table boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
OUTCOME_COLUMNS = ("hit", "miss", "false_alarm", "correct_reject")
...
outcomes = np.column_stack(
    [trials[name].astype(bool) for name in OUTCOME_COLUMNS]
)
if not np.all(outcomes[keep].sum(axis=1) == 1):
    raise ValueError(f"{path.name}: retained trial lacks a unique outcome")
```

iii. The AI does not justify the raw columns separately; it treats these booleans as the canonical trial outcomes and enforces that each retained trial has exactly one.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI encodes the active outcome as an integer 0-3 by taking the index of the true outcome column, then repeats that code across every time bin in the trial.

ii.
```python
outcome = int(np.flatnonzero(outcomes[trial_row])[0])
output = np.empty((5, n_time), dtype=np.int16)
...
output[4].fill(outcome)
```

iii. The docstring explicitly says trial outcome is repeated over time so static and time-varying targets can share one rectangular output matrix.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI excludes active experiments missing eye-tracking entirely. Within retained experiments, it linearly interpolates over finite running and pupil samples, so short NaN gaps are filled. It raises errors if there are fewer than two finite behavior samples, if no valid neurons remain, if no unique trial outcome exists, or if fewer than two trials survive filtering. Trials shorter than one 100 ms bin are dropped. There is no top-level `try/except` to continue past a bad experiment.

ii.
```python
for exp_id in active.index:
    with h5py.File(paths[int(exp_id)], "r") as nwb:
        if "acquisition/EyeTracking/pupil_tracking/area" in nwb:
            usable_ids.append(int(exp_id))
...
valid = np.isfinite(timestamps) & np.isfinite(values)
if np.count_nonzero(valid) < 2:
    raise ValueError(f"Fewer than two finite {label} samples")
return np.interp(targets, timestamps[valid], values[valid]).astype(np.float32)
...
if events.shape[1] == 0:
    raise ValueError(f"{path.name}: no valid neurons")
...
if len(grids) < 2:
    raise ValueError(f"{path.name}: fewer than two retained trials")
```

iii. Trajectory step 26 explicitly mentions excluding three active experiments without eye tracking. The docstring justifies linear interpolation across short blink gaps in the processed pupil trace. Step 44 separately says silent neural trials were intentionally retained rather than filtered out.

## 9-a. What are the most time-consuming steps of the code?

i. The heavy work is reading each NWB’s large event matrix from disk, cumulative-summing it, and then reopening the same NWB files multiple times during discovery, image-name collection, and full conversion.

ii.
```python
for exp_id in active.index:
    with h5py.File(paths[int(exp_id)], "r") as nwb:
        ...

for exp_id in selected.index:
    with h5py.File(paths[int(exp_id)], "r") as nwb:
        ...

with h5py.File(path, "r") as nwb:
    ...
    events = read_array(event_group, "data", np.float32)
    ...
    np.cumsum(events, axis=0, dtype=np.float32, out=events)
```

iii. The trajectory does not explicitly profile runtime, but step 31 notes the conversion produced a 2.55 GiB dataset retaining every valid trial and neuron, which is consistent with large I/O and event-array processing dominating runtime.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest non-vectorized loops are the per-file scan for missing eye tracking, the per-file scan used to collect image names, the per-trial loop over `grids`, and the nested per-presentation loop used to write image identity and image change into each trial.

ii.
```python
for exp_id in active.index:
    with h5py.File(paths[int(exp_id)], "r") as nwb:
        ...

for exp_id in selected.index:
    with h5py.File(paths[int(exp_id)], "r") as nwb:
        ...

for trial_row, edges, centers in grids:
    ...
    for presentation in overlaps:
        ...
```

iii. The trajectory does not explicitly discuss vectorization. This is an inference from the code structure.

## 9-c. What processing does the code repeat multiple times?

i. The code reopens each NWB file multiple times: once in `discover`, once in `collect_image_names`, and once in `convert_experiment`. It also rereads stimulus-presentation tables both while collecting image names and while converting each experiment.

ii.
```python
def discover(data_root: Path):
    ...
    for exp_id in active.index:
        with h5py.File(paths[int(exp_id)], "r") as nwb:
            ...

def collect_image_names(selected: pd.DataFrame, paths: dict[int, Path]):
    for exp_id in selected.index:
        with h5py.File(paths[int(exp_id)], "r") as nwb:
            stim = stimulus_group(nwb)
            ...

def convert_experiment(path: Path, row: pd.Series,
                       image_to_index: dict[str, int],
                       region_to_index: dict[str, int]):
    with h5py.File(path, "r") as nwb:
        stim = stimulus_group(nwb)
        ...
```

iii. The trajectory does not explicitly call out this repetition. It is directly visible in the implementation.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI does not explicitly describe any discarded processing. In the implementation, the main avoidable work is creating empty `input` arrays for every trial even though there are no decoder inputs, and building verbose `session_info` metadata that is not used by the decoder itself. It also computes explicit gray-state image labels and time-expanded trial-outcome rows, which are design choices rather than requirements of the downstream trainer.

ii.
```python
input_trials.append(np.empty((0, n_time), dtype=np.float32))
...
info = {
    "ophys_experiment_id": int(row.name),
    "ophys_session_id": int(row["ophys_session_id"]),
    ...
}
...
"metadata": {
    ...
    "session_info": session_info,
}
```

iii. The trajectory does not explicitly justify these as necessary for downstream analysis; the only stated rationale is that outcomes are repeated over time so all outputs fit a rectangular matrix.
