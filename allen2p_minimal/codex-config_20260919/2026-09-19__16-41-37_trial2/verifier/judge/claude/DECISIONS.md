# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads NWB files directly using `h5py` rather than the AllenSDK's Python API. It discovers experiments by scanning for `.nwb` files in the `behavior_ophys_experiments` directory and cross-references them with `ophys_experiment_table.csv`. It filters to active (non-passive) experiments with valid eye tracking data.

ii.
```python
def discover(data_root: Path):
    nwb_dir = data_root / "behavior_ophys_experiments"
    table_path = data_root / "project_metadata/ophys_experiment_table.csv"
    table = pd.read_csv(table_path).set_index("ophys_experiment_id")
    paths = {experiment_id(path): path for path in nwb_dir.glob("*.nwb")}
    ...
    active = supplied[~supplied["passive"].astype(bool)].copy()
```

iii. The agent stated: "I'm treating each NWB experiment (one imaging plane) as a decoder session, consistent with the paper's plane-wise decoding and the SDK's experiment abstraction." The agent used direct h5py reading to "avoid materializing image templates and ROI masks that are irrelevant to decoding."

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values from the experiment table CSV. They are sorted and mapped to indices.

ii.
```python
subjects = sorted({str(int(x)) for x in selected["mouse_id"]})
subject_to_index = {name: i for i, name in enumerate(subjects)}
```

iii. The agent uses the standard mouse_id field from the experiment metadata to identify unique subjects.

## 1-c. How are the data split into sessions?

i. Each individual `behavior_ophys_experiment` (one imaging plane) is treated as a separate decoder session. This differs from the reference, which groups multiple experiments from the same `ophys_session_id` into a single session.

ii.
```python
for number, (exp_id, row) in enumerate(selected.iterrows(), start=1):
    neural, inputs, outputs, regions, info = convert_experiment(
        paths[int(exp_id)], row, image_to_index, region_to_index
    )
```

iii. The agent justified this: "treating each NWB experiment (one imaging plane) as a decoder session, consistent with the paper's plane-wise decoding and the SDK's experiment abstraction; combining planes would discard most supplied single-plane recordings and require resampling asynchronous plane timestamps."

## 1-d. How are the data split into trials?

i. Trials are defined using the NWB `intervals/trials` table. The AI retains trials where `go` OR `catch` is true, AND `aborted` is false AND `auto_rewarded` is false. The trial window spans from `start_time` to `stop_time` (variable length), binned into 100ms bins.

ii.
```python
keep = (
    (trials["go"].astype(bool) | trials["catch"].astype(bool))
    & ~trials["aborted"].astype(bool)
    & ~trials["auto_rewarded"].astype(bool)
)
grids = make_trial_grids(trials, keep)
```

```python
def make_trial_grids(trials, keep):
    grids = []
    for row in np.flatnonzero(keep):
        start = float(trials["start_time"][row])
        stop = float(trials["stop_time"][row])
        n_bins = int(np.floor((stop - start) / BIN_SECONDS + 1e-9))
        ...
```

iii. The agent stated the trial filter is "(go OR catch) AND NOT aborted AND NOT auto_rewarded", matching the instruction requirements.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) must be go or catch, (2) not aborted, (3) not auto-rewarded, (4) must have at least 1 time bin (n_bins >= 1), (5) sessions must have at least 2 valid trials. Additionally, passive sessions and sessions without eye tracking are excluded at the experiment level.

ii.
```python
keep = (
    (trials["go"].astype(bool) | trials["catch"].astype(bool))
    & ~trials["aborted"].astype(bool)
    & ~trials["auto_rewarded"].astype(bool)
)
...
if len(grids) < 2:
    raise ValueError(f"{path.name}: fewer than two retained trials")
```

For experiment-level filtering:
```python
active = supplied[~supplied["passive"].astype(bool)].copy()
...
if "acquisition/EyeTracking/pupil_tracking/area" in nwb:
    usable_ids.append(int(exp_id))
```

iii. The agent noted: "199 usable active experiments after excluding 82 passive recordings and three active recordings without eye tracking."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `event_detection` (detected calcium-event magnitudes) in the NWB processing/ophys group, NOT from dF/F traces.

ii.
```python
ophys = nwb["processing/ophys"]
event_group = ophys["event_detection"]
ophys_timestamps = read_array(event_group, "timestamps", np.float64)
events = read_array(event_group, "data", np.float32)
```

iii. The agent stated: "The source paper uses detected calcium events (not dF/F), so neural matrices will contain binned event magnitudes."

## 2-b. How is the `neural` data processed?

i. The AI filters neurons by `valid_roi`, then sums calcium event magnitudes into 100ms time bins using a cumulative sum approach. Bins are defined by trial start/stop times divided into 100ms intervals.

ii.
```python
valid_roi = read_array(
    ophys["image_segmentation/cell_specimen_table"], "valid_roi"
).astype(bool)
events = np.ascontiguousarray(events[:, valid_roi], dtype=np.float32)
np.cumsum(events, axis=0, dtype=np.float32, out=events)
...
def cumulative_bin_sums(cumulative_events, ophys_timestamps, edges):
    indices = np.searchsorted(ophys_timestamps, edges, side="left")
    starts, stops = indices[:-1], indices[1:]
    ...
    return (right - left).T
```

iii. The agent noted this approach "satisfies the common-bin requirement while respecting native ophys timestamps from both ~11 Hz multiplane and ~31 Hz single-plane recordings."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by the `valid_roi` flag from the cell specimen table. Only neurons where `valid_roi` is True are retained. Sessions with zero valid neurons raise an error.

ii.
```python
valid_roi = read_array(
    ophys["image_segmentation/cell_specimen_table"], "valid_roi"
).astype(bool)
events = np.ascontiguousarray(events[:, valid_roi], dtype=np.float32)
if events.shape[1] == 0:
    raise ValueError(f"{path.name}: no valid neurons")
```

iii. The agent stated: "Released experiments/ROIs have passed Allen QC; valid_roi is nevertheless applied explicitly."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start. Time bins of 100ms are anchored at each trial's `start_time`. Calcium event magnitudes whose ophys timestamps fall within each bin are summed.

ii.
```python
edges = start + np.arange(n_bins + 1, dtype=np.float64) * BIN_SECONDS
centers = edges[:-1] + BIN_SECONDS / 2.0
...
neural = cumulative_bin_sums(events, ophys_timestamps, edges)
```

iii. The agent stated: "All streams are placed on 100 ms bins anchored at each trial start."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins to 100ms (0.1s) time bins. This is a departure from the native ophys frame rate (~11Hz for multiplane, ~31Hz for single-plane). The time bin size is stored as 100.0 ms in the metadata.

ii.
```python
BIN_SECONDS = 0.100
...
"time_bin_size": BIN_SECONDS * 1000.0,
```

iii. The agent justified this: "100 ms bins anchored at each trial start... satisfies the common-bin requirement while respecting native ophys timestamps."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table in the NWB file (`intervals` group with `image_name`), using `start_time`, `stop_time`, `image_name`, and `omitted` fields. An explicit `gray` class is assigned to time bins outside any displayed image.

ii.
```python
stim = stimulus_group(nwb)
stim_start = read_array(stim, "start_time", np.float64)
stim_stop = read_array(stim, "stop_time", np.float64)
stim_names = decode_strings(read_array(stim, "image_name"))
omitted = read_array(stim, "omitted").astype(bool)
```

iii. The agent noted: "Image identity is an explicit 'gray' class outside a displayed image interval (including omissions)."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each trial, each time bin is assigned the image currently displayed (from the stimulus presentations table). Bins where no image is on screen (inter-stimulus gray periods, omitted flashes) are assigned `gray` (index 0). Image names are mapped to integer indices via a global mapping.

ii.
```python
image_identity = np.full(n_time, gray_index, dtype=np.int16)
...
for presentation in overlaps:
    if not omitted[presentation] and stim_names[presentation] != "omitted":
        shown = ((centers >= stim_start[presentation])
                 & (centers < stim_stop[presentation]))
        image_identity[shown] = image_to_index[stim_names[presentation]]
```

iii. The AI's approach uses stimulus presentations rather than the trials table columns, providing finer-grained image identity tracking including gray periods between flashes.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the same 100ms bin centers as the neural data, using the stimulus presentation start/stop times. Alignment is inherent since both use the same time grid.

ii.
```python
shown = ((centers >= stim_start[presentation])
         & (centers < stim_stop[presentation]))
image_identity[shown] = image_to_index[stim_names[presentation]]
```

iii. Alignment is guaranteed by computing image identity at the same bin centers used for neural data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in the stimulus presentations table. It is a binary variable: 1 at the first bin at or after a true image-change onset, 0 otherwise.

ii.
```python
is_change_raw = read_array(stim, "is_change", np.float64)
is_change = np.isfinite(is_change_raw) & is_change_raw.astype(bool)
```

iii. The agent noted: "Image-change is one only in the first bin centered at or after a true image-change onset."

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each stimulus presentation marked as `is_change`, a single time bin is set to 1 — the first bin whose center is at or after the change onset. This differs from the reference which marks a 750ms window.

ii.
```python
if is_change[presentation]:
    change_bin = int(np.searchsorted(
        centers, stim_start[presentation], side="left"
    ))
    if change_bin < n_time:
        image_change[change_bin] = 1
```

iii. The AI marks only a single bin rather than a window, creating a point indicator of the change event.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1), so no thresholding is needed. The output values are `["no change", "change"]`.

ii.
```python
image_change = np.zeros(n_time, dtype=np.int16)
...
image_change[change_bin] = 1
```

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity — computed at the same 100ms bin centers as neural data.

ii. See 4-b code snippet.

iii. Alignment is guaranteed by using the same time grid.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from the `processing/running/speed` group in the NWB file, using `timestamps` and `data` arrays.

ii.
```python
running_group = nwb["processing/running/speed"]
running_t = read_array(running_group, "timestamps", np.float64)
running_v = read_array(running_group, "data", np.float64)
```

iii. The agent uses the standard running speed data from the NWB processing pipeline.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated to the 100ms bin centers, then discretized into 5 quintile bins. Quintiles are computed within each experiment (not globally).

ii.
```python
running_aligned = interpolate_finite(
    running_t, running_v, all_centers, "running-speed"
)
running_bins = quintile(running_aligned)
```

```python
def quintile(values):
    edges = np.quantile(values, (0.2, 0.4, 0.6, 0.8))
    return np.searchsorted(edges, values, side="right").astype(np.int16)
```

iii. The agent justified within-experiment quintiling: "This removes between-animal/camera scale differences and gives percentile classes for the data actually decoded."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal percentile bins (quintiles: 0-20%, 20-40%, 40-60%, 60-80%, 80-100%) computed within each experiment.

ii.
```python
def quintile(values):
    edges = np.quantile(values, (0.2, 0.4, 0.6, 0.8))
    return np.searchsorted(edges, values, side="right").astype(np.int16)
```

iii. Within-experiment quintiling ensures consistent percentile categories despite scale differences between animals/sessions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same 100ms bin centers as neural data for each trial. All centers across a session are concatenated, interpolated once, then split back per trial.

ii.
```python
all_centers = np.concatenate([centers for _, _, centers in grids])
running_aligned = interpolate_finite(
    running_t, running_v, all_centers, "running-speed"
)
...
output[2] = running_bins[offset:offset + n_time]
```

iii. By interpolating at bin centers, running speed is aligned to the same time points as neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/area` (pupil area) in the NWB file. The area is converted to equivalent circular diameter. Timestamps come from the eye tracking group.

ii.
```python
pupil_group = nwb["acquisition/EyeTracking/pupil_tracking"]
eye_t = read_array(
    nwb["acquisition/EyeTracking/eye_tracking"], "timestamps", np.float64
)
pupil_area = read_array(pupil_group, "area", np.float64)
pupil_diameter = 2.0 * np.sqrt(np.maximum(pupil_area, 0.0) / np.pi)
```

iii. The agent noted: "The NWB 'area' field is SDK-processed: outliers and dilated likely blink frames are NaN. Equivalent circular diameter is in camera pixels."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area is converted to equivalent circular diameter via `2*sqrt(area/pi)`. The SDK-processed area already has blink frames as NaN. The diameter is then linearly interpolated to bin centers (NaN values are handled by `interpolate_finite` which only uses finite samples). Finally, quintile-binned within each experiment.

ii.
```python
pupil_diameter = 2.0 * np.sqrt(np.maximum(pupil_area, 0.0) / np.pi)
...
pupil_aligned = interpolate_finite(
    eye_t, pupil_diameter, all_centers, "pupil-diameter"
)
pupil_bins = quintile(pupil_aligned)
```

iii. The agent noted: "Since this is monotonic, its percentile labels equal pupil-area percentile labels. Short blink gaps (NaNs in the SDK-processed trace) are linearly interpolated."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed — discretized into 5 quintile bins within each experiment.

ii.
```python
pupil_bins = quintile(pupil_aligned)
```

iii. Same within-experiment quintiling approach as running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — interpolated to the same 100ms bin centers as neural data.

ii.
```python
pupil_aligned = interpolate_finite(
    eye_t, pupil_diameter, all_centers, "pupil-diameter"
)
```

iii. Same alignment approach as running speed.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB trials table.

ii.
```python
OUTCOME_COLUMNS = ("hit", "miss", "false_alarm", "correct_reject")
...
outcomes = np.column_stack(
    [trials[name].astype(bool) for name in OUTCOME_COLUMNS]
)
```

iii. The agent validates that each retained trial has exactly one outcome set: `if not np.all(outcomes[keep].sum(axis=1) == 1): raise ValueError(...)`.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome is determined by which of the four boolean columns is True, mapped to an integer index (0-3). The outcome code is repeated across all time bins for the trial.

ii.
```python
outcome = int(np.flatnonzero(outcomes[trial_row])[0])
...
output[4].fill(outcome)
```

iii. The outcome is static per trial but repeated across time bins so all output variables have the same shape.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: Experiments without eye tracking data are excluded entirely (3 experiments excluded).
- **Passive sessions**: Excluded as they don't contain Go/Catch trials.
- **Blink-corrupted pupil data**: The SDK-processed area already has blinks as NaN; `interpolate_finite` only uses finite samples.
- **NaN running/pupil values**: `interpolate_finite` uses `np.interp` which extrapolates using nearest valid endpoint.
- **Outcome validation**: Raises error if a retained trial doesn't have exactly one outcome.
- **Few trials**: Sessions with fewer than 2 trials raise an error.
- **No valid neurons**: Sessions with zero valid ROIs raise an error.

ii.
```python
# Excluding sessions without eye tracking
if "acquisition/EyeTracking/pupil_tracking/area" in nwb:
    usable_ids.append(int(exp_id))
else:
    excluded_no_eye.append(int(exp_id))

# interpolate_finite only uses finite values
valid = np.isfinite(timestamps) & np.isfinite(values)
```

iii. The agent stated: "Excluding sessions in which the NWB lacks eye tracking is preferable to inventing a constant/imputed signal."

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) the `discover` function which opens every NWB file to check for eye tracking data, and (2) `convert_experiment` which reads large event, running, and pupil arrays from each NWB file. Additionally, `collect_image_names` opens every NWB file a second time to collect image names.

ii.
```python
for exp_id in active.index:
    with h5py.File(paths[int(exp_id)], "r") as nwb:
        if "acquisition/EyeTracking/pupil_tracking/area" in nwb:
            ...
```

iii. NWB file I/O dominates runtime, as each file contains large neural and behavioral arrays.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_experiment` that iterates over stimulus presentations (`for presentation in overlaps`) could potentially be vectorized. The main per-trial loop over `grids` is harder to vectorize due to variable trial lengths.

ii.
```python
for trial_row, edges, centers in grids:
    ...
    for presentation in overlaps:
        ...
```

iii. The loops are not performance bottlenecks compared to I/O.

## 9-c. What processing does the code repeat multiple times?

i. The code opens each NWB file multiple times: once in `discover()` to check for eye tracking, once in `collect_image_names()` to gather image names, and once in `convert_experiment()` for actual conversion. This triples the file I/O overhead.

ii.
```python
# In discover():
for exp_id in active.index:
    with h5py.File(paths[int(exp_id)], "r") as nwb: ...

# In collect_image_names():
for exp_id in selected.index:
    with h5py.File(paths[int(exp_id)], "r") as nwb: ...

# In convert_experiment():
with h5py.File(path, "r") as nwb: ...
```

iii. This is a design choice for code clarity — each function is self-contained, but it means each NWB file is opened 3 times.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code collects extensive metadata (`session_info` with IDs, session types, experience levels) that is stored in the metadata dict but not used by the decoder. The `collect_image_names` pass is also somewhat redundant since image names could be collected during the main conversion pass.

ii.
```python
info = {
    "ophys_experiment_id": int(row.name),
    "ophys_session_id": int(row["ophys_session_id"]),
    "behavior_session_id": int(row["behavior_session_id"]),
    ...
}
```

iii. The metadata is useful for debugging and provenance but adds I/O overhead.
