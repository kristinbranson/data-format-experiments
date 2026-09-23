# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the AllenSDK object API at all. It reads the locally released NWB files directly with `h5py`, enumerating every `behavior_ophys_experiment_<id>.nwb` in `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments` and joining each file to its row of `project_metadata/ophys_experiment_table.csv` (the id is parsed out of the filename). It asserts that every supplied NWB has a metadata row. All 284 supplied experiments are considered (239 single-plane `VisualBehavior` + 45 multiplane `VisualBehaviorMultiscope` planes from one mouse); **no `project_code` filter is applied**. Experiments are then filtered to active behaviour (`~passive`, removing 82) and to those that actually contain eye tracking (removing 3), leaving 199 converted experiments. Only the specific HDF5 datasets needed are read (trials table, `event_detection` data/timestamps, `valid_roi`, running speed, eye tracking, stimulus presentations); image templates, ROI masks, raw fluorescence, etc. are never materialised.

ii.
```python
def discover(data_root: Path):
    nwb_dir = data_root / "behavior_ophys_experiments"
    table_path = data_root / "project_metadata/ophys_experiment_table.csv"
    table = pd.read_csv(table_path).set_index("ophys_experiment_id")
    paths = {experiment_id(path): path for path in nwb_dir.glob("*.nwb")}
    missing_metadata = sorted(set(paths) - set(table.index))
    if missing_metadata:
        raise ValueError(f"No metadata for experiment ids {missing_metadata}")

    supplied = table.loc[sorted(paths)].copy()
    active = supplied[~supplied["passive"].astype(bool)].copy()

    # Pupil diameter is a required output. Excluding sessions in which the NWB
    # lacks eye tracking is preferable to inventing a constant/imputed signal.
    usable_ids = []
    excluded_no_eye = []
    for exp_id in active.index:
        with h5py.File(paths[int(exp_id)], "r") as nwb:
            if "acquisition/EyeTracking/pupil_tracking/area" in nwb:
                usable_ids.append(int(exp_id))
            else:
                excluded_no_eye.append(int(exp_id))
    return table.loc[usable_ids].copy(), paths, excluded_no_eye, len(supplied)
```
```python
for number, (exp_id, row) in enumerate(selected.iterrows(), start=1):
    print(f"[{number:3d}/{total}] experiment {exp_id}", flush=True)
    neural, inputs, outputs, regions, info = convert_experiment(
        paths[int(exp_id)], row, image_to_index, region_to_index
    )
```

iii. From the converter docstring: *"The converter reads NWB arrays directly with h5py. Paths and semantics are those exposed by AllenSDK 2.16 for this dataset, but direct reading avoids materializing image templates and ROI masks that are irrelevant to decoding."* In the trajectory (step 14) the AI first inventoried the supplied files and concluded: *"The supplied files are a deliberate 284-experiment subset: all 239 single-plane `VisualBehavior` experiments plus 45 multiplane experiments from one mouse"*, and chose to convert everything that is active behaviour rather than restrict to one project code. It verified against the SDK once (step 18, `BehaviorOphysExperiment.from_nwb_path`) before committing to the raw-HDF5 path, and it grepped the SDK source for `eye_tracking`, `event_detection`, `valid_roi` semantics (step 21) so that the raw datasets it reads are the same ones the SDK exposes.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values of the selected experiment table, cast to `str` and sorted; each converted session (= experiment) stores an index into that list. 38 subjects result (37 single-plane mice + the one multiscope mouse).

ii.
```python
subjects = sorted({str(int(x)) for x in selected["mouse_id"]})
subject_to_index = {name: i for i, name in enumerate(subjects)}
...
subject_idx.append(subject_to_index[str(int(row["mouse_id"]))])
...
"subjects": subjects,
"subject_idx": np.asarray(subject_idx, dtype=np.int16),
```

iii. No explicit discussion in the trajectory; `mouse_id` is the canonical animal identifier in the Allen experiment table, and the AI cross-checked mouse counts per project code while inventorying the data (steps 12–14). It also records `mouse_id` per session inside `metadata['session_info']`.

## 1-c. How are the data split into sessions?

i. **One NWB `behavior_ophys_experiment` (i.e. one imaging plane) is one decoder session.** For the 239 single-plane `VisualBehavior` experiments this is identical to grouping by `ophys_session_id` (there is exactly one plane per session). For the multiscope mouse, the 34 active planes recorded in ~5 simultaneous multiplane sessions become 34 separate "sessions" that share the same behaviour, trials and outputs; that single mouse therefore supplies 34 of the 199 sessions. Passive replay experiments (82) and experiments without eye tracking (3) are dropped, so no purely-passive session enters the dataset. No per-session performance/engagement threshold is applied.

ii.
```python
* One released ``behavior_ophys_experiment`` (one imaging plane) is one decoder
  session.  This is the AllenSDK unit that owns one simultaneously sampled
  neural population, and it is also the unit used for plane-wise decoding in
  the reference paper.
* Only active-behavior experiments are used.  Passive replay sessions do not
  contain an animal performing Go/Catch trials.
```
```python
active = supplied[~supplied["passive"].astype(bool)].copy()
...
"session_unit": "One behavior_ophys_experiment / imaging plane",
"session_filter": (
    "Active behavior, released QC-passed experiments, valid ROIs, "
    "and eye tracking available"
),
```

iii. Trajectory step 14: *"I'm treating each NWB experiment (one imaging plane) as a decoder session, consistent with the paper's plane-wise decoding and the SDK's experiment abstraction; combining planes would discard most supplied single-plane recordings and require resampling asynchronous plane timestamps."* The passive exclusion is justified in the docstring: *"Passive replay sessions do not contain an animal performing Go/Catch trials."* Step 26 confirms the arithmetic: *"199 usable active experiments after excluding 82 passive recordings and three active recordings without eye tracking."*

## 1-d. How are the data split into trials?

i. Trials are the NWB/SDK `intervals/trials` intervals, i.e. the experiment's own trial definition. The retained window is the full trial interval `start_time → stop_time` (variable length, ~7–12.5 s), which spans the pre-change flashes and the post-change response window. The window is then tiled with whole 100 ms bins anchored at `start_time`; `n_bins = floor((stop - start)/0.1)`, so at most one sub-bin remainder (<100 ms) at the trial end is dropped. Resulting trial lengths are 70–125 bins.

ii.
```python
def make_trial_grids(trials: dict[str, np.ndarray], keep: np.ndarray):
    grids = []
    for row in np.flatnonzero(keep):
        start = float(trials["start_time"][row])
        stop = float(trials["stop_time"][row])
        n_bins = int(np.floor((stop - start) / BIN_SECONDS + 1e-9))
        if n_bins < 1:
            continue
        edges = start + np.arange(n_bins + 1, dtype=np.float64) * BIN_SECONDS
        centers = edges[:-1] + BIN_SECONDS / 2.0
        grids.append((int(row), edges, centers))
    return grids
```
```python
"trial_window": "Full NWB trial interval; duration varies by trial",
```

iii. Docstring: *"Trials are the SDK/NWB trial intervals."* The instruction asked to "segment each recording session into individual trials based on how they are defined in the experiment", and the AI took the trials table verbatim rather than imposing a fixed window around the change; keeping `start_time → stop_time` is what makes the time-varying outputs (image identity, image change) meaningful within a trial.

## 1-e. How are trials filtered based on quality controls?

i. Retained trials are `(go OR catch) AND NOT aborted AND NOT auto_rewarded`, exactly as instructed. Additional guards: a trial that cannot fit one whole 100 ms bin is skipped; every retained trial must have exactly one of `hit/miss/false_alarm/correct_reject` set, otherwise the converter raises; an experiment with fewer than two retained trials (or zero valid ROIs) raises. No engagement, d-prime, lick-rate or activity-based trial selection is imposed, and silent (all-zero event) trials are deliberately kept.

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

outcomes = np.column_stack(
    [trials[name].astype(bool) for name in OUTCOME_COLUMNS]
)
if not np.all(outcomes[keep].sum(axis=1) == 1):
    raise ValueError(f"{path.name}: retained trial lacks a unique outcome")
```
```python
"trial_filter": "(go OR catch) AND NOT aborted AND NOT auto_rewarded",
```

iii. Docstring: *"We retain (Go OR Catch) AND NOT aborted AND NOT auto-rewarded trials, exactly as requested. No engagement or performance threshold is imposed."* On silent trials, trajectory step 44: *"The verifier's only warnings are expected fully silent event trials from sparse detected events, which I retained to avoid activity-based trial selection bias."*

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `processing/ophys/event_detection/data` — the Allen-detected discrete calcium events (magnitudes per ROI per ophys frame) — together with `processing/ophys/event_detection/timestamps` (the ophys timestamps) and `processing/ophys/image_segmentation/cell_specimen_table/valid_roi`. dF/F is deliberately not used, and neither is the SDK's smoothed `filtered_events`.

ii.
```python
ophys = nwb["processing/ophys"]
event_group = ophys["event_detection"]
ophys_timestamps = read_array(event_group, "timestamps", np.float64)
events = read_array(event_group, "data", np.float32)
valid_roi = read_array(
    ophys["image_segmentation/cell_specimen_table"], "valid_roi"
).astype(bool)
events = np.ascontiguousarray(events[:, valid_roi], dtype=np.float32)
```
```python
"neural_measure": (
    "Allen event_detection calcium-event magnitudes, summed in 100 ms bins"
),
```

iii. Docstring: *"Neural activity is the detected calcium-event magnitude used in the paper, not dF/F or the SDK's display-oriented filtered events."* Trajectory step 19: *"The source paper uses detected calcium events (not dF/F), so neural matrices will contain binned event magnitudes."* This follows `methods.txt` directly: *"For all analysis of neural data we used the detected calcium events … This process produces, for each cell, a set of calcium events each with a time and magnitude"*, and *"We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces, thus removing the slow decay dynamics of the calcium indicator GCaMP6f."*

## 2-b. How is the `neural` data processed?

i. Events are restricted to valid ROIs and then **summed into 100 ms bins** anchored at each trial start; bin membership is decided by the ophys timestamp of each frame. The summation is implemented once per experiment as an in-place cumulative sum along time, after which any trial's bin sums are obtained as differences of the cumulative trace at the bin edges. No normalisation, smoothing, z-scoring or neuron-level rescaling is applied, and each session's matrix is a single plane (no cross-plane stacking). Output dtype is float32, shape (n_neurons, n_bins).

ii.
```python
# In-place cumulative sums let arbitrary trial-relative bins be obtained
# without rereading the large event dataset or allocating a second copy.
np.cumsum(events, axis=0, dtype=np.float32, out=events)
```
```python
def cumulative_bin_sums(cumulative_events, ophys_timestamps, edges):
    """Sum native-frame event magnitudes into bins; return neuron x time."""
    indices = np.searchsorted(ophys_timestamps, edges, side="left")
    starts, stops = indices[:-1], indices[1:]
    ...
    right[has_right] = cumulative_events[stops[has_right] - 1]
    left[has_left] = cumulative_events[starts[has_left] - 1]
    return (right - left).T
```

iii. Docstring: *"All streams are placed on 100 ms bins anchored at each trial start. Calcium event magnitudes whose *ophys timestamps* fall in a bin are summed."* Summation (rather than averaging) is the natural aggregation for sparse event magnitudes: it conserves total event magnitude regardless of how many native frames land in a bin, which matters because the supplied data mix ~31 Hz single-plane and ~11 Hz multiplane recordings.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only the released `valid_roi` mask is applied; an experiment with zero valid ROIs would raise. Nothing else is filtered — no SNR, event-rate, or activity criteria, and no dropping of silent cells/trials. (In the released files `valid_roi` is all-True, so the mask is an explicit safeguard rather than an effective filter.)

ii.
```python
valid_roi = read_array(
    ophys["image_segmentation/cell_specimen_table"], "valid_roi"
).astype(bool)
events = np.ascontiguousarray(events[:, valid_roi], dtype=np.float32)
if events.shape[1] == 0:
    raise ValueError(f"{path.name}: no valid neurons")
```

iii. Docstring: *"Released experiments/ROIs have passed Allen QC; `valid_roi` is nevertheless applied explicitly."* The AI grepped the SDK for `valid_roi` usage (step 21) before adding the mask, and refused activity-based curation (step 44) to avoid selection bias.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **trial start** (`metadata['temporal_alignment_event'] = 'Trial start; synchronized streams binned/aligned using ophys timestamps on the NWB clock'`, `off_start = 0.0`, `off_end = None` because trial length varies). The bin grid starts exactly at `start_time`, and each event frame is assigned to a bin by comparing its own ophys timestamp against the bin edges (`np.searchsorted` on `ophys_timestamps`), never by frame index or nominal frame rate. Running speed and pupil are sampled at the bin centres of the same grid, so all streams share one timebase.

ii.
```python
edges = start + np.arange(n_bins + 1, dtype=np.float64) * BIN_SECONDS
centers = edges[:-1] + BIN_SECONDS / 2.0
...
indices = np.searchsorted(ophys_timestamps, edges, side="left")
```
```python
"temporal_alignment_event": (
    "Trial start; synchronized streams binned/aligned using ophys "
    "timestamps on the NWB clock"
),
"off_start": 0.0,
"off_end": None,
```

iii. Docstring: *"Thus the ophys timestamps, rather than frame number or nominal acquisition rate, define alignment."* The instruction said to "temporally align based on ophys timestamp"; `methods.txt` notes all streams are hardware-synchronised on a single clock, which is why the AI can put behaviour and neural data on a common trial-start-anchored grid.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 100 ms fixed bins for every trial and every session (`metadata['time_bin_size'] = 100.0`). This **is** a rebinning: native ophys sampling is ~32.3 ms (31 Hz) for the single-plane experiments and ~90 ms (11 Hz) for the multiplane ones, and both are re-expressed on the common 100 ms grid by summing event magnitudes. Behavioural streams are point-sampled (interpolated) at the 100 ms bin centres.

ii.
```python
BIN_SECONDS = 0.100
...
"time_bin_size": BIN_SECONDS * 1000.0,
```

iii. Trajectory step 19: *"I'll … resample every stream onto a common 100 ms grid anchored to trial start, and use event sums within each bin. This satisfies the common-bin requirement while respecting native ophys timestamps from both ~11 Hz multiplane and ~31 Hz single-plane recordings."* The target format requires "Time bins … the same size for all trials and sessions", and the AI's decision to include both rigs makes a common re-binning necessary; 100 ms is also convenient relative to the 250 ms flash / 500 ms grey stimulus cycle.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The stimulus presentations interval table (the one NWB interval group that has an `image_name` column — the natural-image task table, explicitly distinguished from `natural_movie_one_presentations` and `spontaneous_presentations`): `image_name`, `start_time`, `stop_time`, `omitted`. The trials table's `initial_image_name` / `change_image_name` are **not** used.

ii.
```python
def stimulus_group(nwb: h5py.File) -> h5py.Group:
    """Return the natural-image task presentation table (not movie/spontaneous)."""
    candidates = []
    for name, group in nwb["intervals"].items():
        if isinstance(group, h5py.Group) and "image_name" in group:
            candidates.append((name, group))
    if len(candidates) != 1:
        names = [name for name, _ in candidates]
        raise ValueError(f"Expected one task image table, found {names}")
    return candidates[0][1]
```
```python
stim = stimulus_group(nwb)
stim_start = read_array(stim, "start_time", np.float64)
stim_stop = read_array(stim, "stop_time", np.float64)
stim_names = decode_strings(read_array(stim, "image_name"))
omitted = read_array(stim, "omitted").astype(bool)
```

iii. The presentations table is the authoritative per-flash record of what was actually on the monitor (including omissions), which is what the AI needed in order to represent the grey inter-stimulus interval explicitly (see 3-b). The AI enumerated the interval groups and image-name vocabularies across all files before writing the converter (step 20) and hard-fails if more than one image table is found.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Per-bin categorical label. A global, deterministic vocabulary is built by scanning every selected NWB's `image_name` column: `['gray', <16 sorted image names>]`, so `gray` is class 0 and there are 17 classes. Each bin is labelled `gray` by default and is overwritten with the image index if the bin centre falls inside a non-omitted presentation interval `[start_time, stop_time)`. Omitted flashes therefore stay `gray`. Because the stimulus cycle is 250 ms image / 500 ms grey, ~66.5% of all bins are `gray`.

ii.
```python
def collect_image_names(selected, paths):
    names = set()
    for exp_id in selected.index:
        with h5py.File(paths[int(exp_id)], "r") as nwb:
            stim = stimulus_group(nwb)
            image_names = decode_strings(read_array(stim, "image_name"))
            omitted = read_array(stim, "omitted").astype(bool)
            names.update(image_names[~omitted])
    names.discard("omitted")
    return ["gray", *sorted(names)]
```
```python
image_identity = np.full(n_time, gray_index, dtype=np.int16)
...
for presentation in overlaps:
    if not omitted[presentation] and stim_names[presentation] != "omitted":
        shown = ((centers >= stim_start[presentation])
                 & (centers < stim_stop[presentation]))
        image_identity[shown] = image_to_index[stim_names[presentation]]
```

iii. Docstring: *"Image identity is an explicit `gray` class outside a displayed image interval (including omissions)."* The AI read the instruction's "(of the image presented during the non-grey screen)" as meaning that the label should reflect the actual screen content, so that the grey inter-stimulus period and 5% omissions get their own class rather than inheriting the previous image.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is computed on exactly the same per-trial bin grid as the neural matrix: a bin takes an image label iff its centre lies in that presentation's `[start_time, stop_time)` interval. Only presentations overlapping the trial window are considered. The row is stored as row 0 of the trial's `(5, n_time)` output matrix, so it is element-wise aligned with the neural `(n_neurons, n_time)` matrix.

ii.
```python
overlaps = np.flatnonzero(
    (stim_stop > edges[0]) & (stim_start < edges[-1])
)
...
shown = ((centers >= stim_start[presentation])
         & (centers < stim_stop[presentation]))
image_identity[shown] = image_to_index[stim_names[presentation]]
...
output = np.empty((5, n_time), dtype=np.int16)
output[0] = image_identity
```

iii. Same rationale as 2-d: the stimulus times and ophys timestamps live on the same synchronised NWB clock, so testing presentation intervals against the bin centres of the trial-start-anchored grid puts stimulus and neural data on one timebase.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The `is_change` column of the stimulus presentations table, plus that presentation's `start_time`. (In these files `is_change` is stored as float and is True only for genuine image changes on go trials; catch/sham changes are not flagged, so catch trials are automatically all-zero.) The trials table's `change_time`/`go` are not used for this variable.

ii.
```python
is_change_raw = read_array(stim, "is_change", np.float64)
is_change = np.isfinite(is_change_raw) & is_change_raw.astype(bool)
```

iii. Not explicitly narrated in the trajectory beyond the docstring line *"Image-change is one only in the first bin centered at or after a true image-change onset"*; the AI inspected the stimulus table columns (steps 20–25) and used the dataset's own change flag as the definition of "a change in image identity", with the NaN-safe cast because the column is float-typed.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary per-bin row, zero everywhere except a single 1 in the bin whose centre is the first at or after the change-flash onset. No smoothing, no widening to the flash or response window. Over the whole dataset 1.03% of bins are 1.

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
...
output[1] = image_change
```

iii. Docstring: *"Image-change is one only in the first bin centered at or after a true image-change onset."* This is a literal reading of the instruction "Have value of 1 right after a change in image identity, otherwise 0".

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary — two classes, `['no change', 'change']`, with no thresholding of a continuous quantity. The only implicit "threshold" is the temporal width of the positive class: exactly one 100 ms bin per genuine change, and zero positives on catch trials.

ii.
```python
"output_values": [
    image_values,
    ["no change", "change"],
    ...
]
```

iii. As in 4-b: the AI chose the narrowest representation consistent with "value of 1 right after a change", rather than marking the whole post-change flash/response window.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same trial bin grid as the neural data; the positive bin is located with `np.searchsorted(centers, stim_start)`, i.e. the first bin whose centre is at or after the change onset, and it is bounds-checked against the trial length. Stored as row 1 of the same `(5, n_time)` matrix.

ii.
```python
change_bin = int(np.searchsorted(centers, stim_start[presentation], side="left"))
if change_bin < n_time:
    image_change[change_bin] = 1
```

iii. Identical alignment logic and justification as image identity (3-c) and the neural binning (2-d): everything is indexed off the trial-start-anchored 100 ms grid defined on the synchronised NWB clock.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed` (`data` + `timestamps`) — the SDK's filtered running-speed stream, not `speed_unfiltered` and not the raw encoder `dx`.

ii.
```python
running_group = nwb["processing/running/speed"]
running_t = read_array(running_group, "timestamps", np.float64)
running_v = read_array(running_group, "data", np.float64)
```

iii. Not narrated in detail; this is the NWB path that backs the SDK's `running_speed` property (the AI inspected the SDK/tutorial usage of `running_speed` in steps 9/18/21), i.e. the same stream the reference tooling exposes.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Finite samples are linearly interpolated onto the concatenated bin centres of all retained trials of that experiment (`np.interp`, which clamps to the nearest endpoint outside the sampled range), and the resulting values are then quintile-labelled. Fewer than two finite samples is a hard error. No smoothing, clipping or sign handling is applied.

ii.
```python
def interpolate_finite(timestamps, values, targets, label):
    valid = np.isfinite(timestamps) & np.isfinite(values)
    if np.count_nonzero(valid) < 2:
        raise ValueError(f"Fewer than two finite {label} samples")
    # np.interp also supplies the nearest valid endpoint outside the sampled
    # range. Retained trials normally lie strictly inside both behavior traces.
    return np.interp(targets, timestamps[valid], values[valid]).astype(np.float32)

all_centers = np.concatenate([centers for _, _, centers in grids])
running_aligned = interpolate_finite(running_t, running_v, all_centers, "running-speed")
running_bins = quintile(running_aligned)
```

iii. Docstring: *"Running speed and processed (blink-filtered) pupil area are interpolated at bin centers on the common synchronized NWB clock."* Interpolating at bin centres (rather than resampling the neural data to the behaviour clock) keeps behaviour on the ophys-derived timebase required by the instructions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile bins (quintiles), with the 20/40/60/80th percentile edges computed **separately for each experiment**, over exactly the bins that are retained in the converted trials (not the whole session). Labels 0–4 named `['0-20%', …, '80-100%']`. Marginal class frequencies come out at 20% each.

ii.
```python
def quintile(values: np.ndarray) -> np.ndarray:
    """Return labels 0..4 using empirical 20/40/60/80 percentile edges."""
    edges = np.quantile(values, (0.2, 0.4, 0.6, 0.8))
    return np.searchsorted(edges, values, side="right").astype(np.int16)
...
"percentile_scope": "Within experiment over retained trial bins",
```

iii. Docstring: *"Running and pupil values are quintiled separately within each experiment, using all retained trial bins. This removes between-animal/camera scale differences and gives percentile classes for the data actually decoded."*

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is sampled at the same bin centres as the neural bins. All trials of the experiment are concatenated for one interpolation/quintile call, then sliced back per trial with a running offset; the slice is stored as row 2 of the trial's output matrix.

ii.
```python
offset = 0
for trial_row, edges, centers in grids:
    n_time = len(centers)
    neural = cumulative_bin_sums(events, ophys_timestamps, edges)
    ...
    output[2] = running_bins[offset:offset + n_time]
    output[3] = pupil_bins[offset:offset + n_time]
    offset += n_time
```

iii. Same synchronisation argument as 2-d: running timestamps and ophys timestamps are on one hardware-synced clock (`methods.txt`: all experimental clocks recorded on a single IO board at 100 kHz), so interpolation onto the ophys-anchored grid is valid; concatenating first guarantees the offsets line up with the same `grids` order used for the neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/area` (the SDK-processed pupil-ellipse area, in which likely-blink/outlier frames are already NaN) together with `acquisition/EyeTracking/eye_tracking/timestamps`. Diameter is the equivalent circular diameter `2*sqrt(area/pi)`. `area_raw` and the `likely_blink` column are not read (the NaNs in `area` coincide exactly with `likely_blink`).

ii.
```python
pupil_group = nwb["acquisition/EyeTracking/pupil_tracking"]
eye_t = read_array(
    nwb["acquisition/EyeTracking/eye_tracking"], "timestamps", np.float64
)
pupil_area = read_array(pupil_group, "area", np.float64)
# The NWB ``area`` field is SDK-processed: outliers and dilated likely
# blink frames are NaN. Equivalent circular diameter is in camera pixels.
pupil_diameter = 2.0 * np.sqrt(np.maximum(pupil_area, 0.0) / np.pi)
```

iii. Docstring: *"Pupil diameter is the equivalent circular diameter 2*sqrt(area/pi). Since this is monotonic, its percentile labels equal pupil-area percentile labels. Short blink gaps (NaNs in the SDK-processed trace) are linearly interpolated."* The AI checked the SDK's eye-tracking/blink handling in the source (step 21) before deciding it could rely on the NaNs already present in `area`.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Area → equivalent circular diameter, NaN (blink) samples dropped, the remaining finite samples linearly interpolated onto the trial bin centres (so blink gaps are bridged by interpolation), then quintile-labelled per experiment. Experiments with no eye-tracking data at all were already removed in `discover()`; an experiment with <2 finite samples raises.

ii.
```python
pupil_aligned = interpolate_finite(eye_t, pupil_diameter, all_centers, "pupil-diameter")
pupil_bins = quintile(pupil_aligned)
...
"pupil_measure": (
    "Equivalent circular diameter from SDK blink-filtered pupil area; "
    "finite samples linearly interpolated"
),
```

iii. Docstring, as quoted in 6-a: the monotonic area→diameter transform is reported because it makes the output interpretable as a diameter while leaving percentile classes unchanged; interpolating only over finite samples is what removes blink artefacts.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identical scheme to running speed: five equal-percentile classes from within-experiment 20/40/60/80 percentiles of the retained trial bins, labelled `['0-20%', …, '80-100%']`, stored as row 3.

ii.
```python
running_bins = quintile(running_aligned)
pupil_bins = quintile(pupil_aligned)
...
"percentile_scope": "Within experiment over retained trial bins",
```

iii. Docstring: *"Running and pupil values are quintiled separately within each experiment … This removes between-animal/camera scale differences and gives percentile classes for the data actually decoded."* Pupil area is measured in camera pixels, so its absolute scale depends on the rig/camera geometry — the stated motivation for per-experiment percentiles applies most strongly here.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolated at the same bin centres as the neural bins and sliced per trial with the same offset bookkeeping as running speed.

ii.
```python
all_centers = np.concatenate([centers for _, _, centers in grids])
pupil_aligned = interpolate_finite(eye_t, pupil_diameter, all_centers, "pupil-diameter")
...
output[3] = pupil_bins[offset:offset + n_time]
```

iii. As for running speed: the eye camera is synchronised to the same master clock as the 2P frames, so interpolating onto the ophys-anchored 100 ms grid is a valid way to put it in register with the neural data.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the trials table: `hit`, `miss`, `false_alarm`, `correct_reject`.

ii.
```python
OUTCOME_COLUMNS = ("hit", "miss", "false_alarm", "correct_reject")
OUTCOME_NAMES = ("hit", "miss", "false alarm", "correct rejection")
...
outcomes = np.column_stack(
    [trials[name].astype(bool) for name in OUTCOME_COLUMNS]
)
if not np.all(outcomes[keep].sum(axis=1) == 1):
    raise ValueError(f"{path.name}: retained trial lacks a unique outcome")
```

iii. These are the dataset's canonical change-detection outcome labels; because aborted and auto-rewarded trials are already excluded, exactly one of the four must be true, and the AI enforces that invariant instead of silently falling back.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. A single integer per trial (index of the true column, 0=hit, 1=miss, 2=false alarm, 3=correct rejection), broadcast constant across all bins of the trial so that the static variable fits the same rectangular `(5, n_time)` output matrix as the time-varying ones.

ii.
```python
outcome = int(np.flatnonzero(outcomes[trial_row])[0])
output = np.empty((5, n_time), dtype=np.int16)
...
output[4].fill(outcome)
```
```python
"output_values": [..., list(OUTCOME_NAMES)],
```

iii. Docstring: *"Trial outcome is repeated over time solely so static and time-varying targets fit one rectangular output matrix."*

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handling is a mixture of explicit exclusion, interpolation and fail-fast:
- **Missing eye tracking**: the 3 active experiments whose NWB has no `pupil_tracking/area` are dropped entirely (ids recorded in `metadata['excluded_missing_eye_tracking_ids']`) rather than given an imputed pupil signal.
- **Passive experiments**: 82 dropped up front.
- **Blink / NaN pupil samples**: excluded from the interpolation source, so gaps are bridged linearly (gaps run from a few hundred ms up to ~1 min in some sessions).
- **Behaviour outside the sampled range**: `np.interp` clamps to the nearest endpoint (documented in a comment) rather than producing NaN.
- **NaN-typed `is_change`**: guarded with `np.isfinite` before the boolean cast.
- **Ragged trial ends**: a remainder shorter than one 100 ms bin is dropped; a trial that cannot fit a whole bin is skipped.
- **Structural anomalies** (no valid ROIs, <2 retained trials, a retained trial without a unique outcome, more than one image-presentation table, an NWB without a metadata row) raise immediately. There is no try/except around `convert_experiment`, so any such case aborts the whole conversion rather than skipping that experiment; none of them triggered on this dataset.
- **Silent trials** (2–3% of trials have no detected events) are kept on purpose.
- The pickle is written to a temporary file and atomically renamed.

ii.
```python
if "acquisition/EyeTracking/pupil_tracking/area" in nwb:
    usable_ids.append(int(exp_id))
else:
    excluded_no_eye.append(int(exp_id))
```
```python
valid = np.isfinite(timestamps) & np.isfinite(values)
if np.count_nonzero(valid) < 2:
    raise ValueError(f"Fewer than two finite {label} samples")
return np.interp(targets, timestamps[valid], values[valid]).astype(np.float32)
```
```python
is_change = np.isfinite(is_change_raw) & is_change_raw.astype(bool)
...
n_bins = int(np.floor((stop - start) / BIN_SECONDS + 1e-9))
if n_bins < 1:
    continue
...
temporary = output_path.with_suffix(output_path.suffix + ".tmp")
with temporary.open("wb") as stream:
    pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
temporary.replace(output_path)
```

iii. Docstring/inline comments: *"Pupil diameter is a required output. Excluding sessions in which the NWB lacks eye tracking is preferable to inventing a constant/imputed signal"*; *"Short blink gaps (NaNs in the SDK-processed trace) are linearly interpolated"*; *"np.interp also supplies the nearest valid endpoint outside the sampled range. Retained trials normally lie strictly inside both behavior traces."* Trajectory step 44 justifies keeping silent trials: *"expected fully silent event trials from sparse detected events, which I retained to avoid activity-based trial selection bias."*

## 9-a. What are the most time-consuming steps of the code?

i. The dominant costs are I/O and the per-experiment event array work: reading `event_detection/data` (≈150k frames × N ROIs float32, tens of MB per file) for each of 199 files, the in-place `np.cumsum` over the full session for every experiment, and finally pickling the 2.55 GiB dense output. Two extra full passes over all NWB files (the eye-tracking probe in `discover()` and `collect_image_names()`) add HDF5 open/read overhead before conversion starts. The AI deliberately avoided the much larger cost of instantiating SDK `BehaviorOphysExperiment` objects. The whole conversion ran in a few minutes.

ii.
```python
events = read_array(event_group, "data", np.float32)
...
np.cumsum(events, axis=0, dtype=np.float32, out=events)
...
pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Docstring: *"direct reading avoids materializing image templates and ROI masks that are irrelevant to decoding"*, and the cumulative-sum comment: *"In-place cumulative sums let arbitrary trial-relative bins be obtained without rereading the large event dataset or allocating a second copy."* `read_array` also casts during the HDF5 read to avoid a second buffer.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The heavy work is already vectorised (one cumsum per experiment; all bin edges of a trial resolved with a single `np.searchsorted`; all trials' behaviour interpolated and quintiled in one call). What remains as Python loops: the per-trial loop in `convert()`/`convert_experiment` (all trials' edges could be concatenated and binned in one `searchsorted`/difference call instead of one call per trial), the inner loop over overlapping stimulus presentations (could be replaced by a single `searchsorted` of bin centres into the presentation start/stop arrays plus fancy indexing), `make_trial_grids`, `decode_strings` (element-wise Python decode of ~70k strings per file, done twice per file), and the two per-file discovery loops. None of these dominate runtime relative to HDF5 reads.

ii.
```python
for trial_row, edges, centers in grids:
    n_time = len(centers)
    neural = cumulative_bin_sums(events, ophys_timestamps, edges)
    ...
    for presentation in overlaps:
        ...
```
```python
def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
         for x in values], dtype=object)
```

iii. Not discussed explicitly in the trajectory; the design comments show the AI optimised where it judged it mattered (single cumulative-sum pass, cast-on-read, one interpolation/quintile call per experiment) and left readable per-trial loops elsewhere.

## 9-c. What processing does the code repeat multiple times?

i. Each NWB file is opened and read **three times**: once in `discover()` to test for eye tracking, once in `collect_image_names()` to read and decode the `image_name` column, and once in `convert_experiment()` (which decodes the same `image_name` column again). The experiment metadata table is re-indexed several times (`table.loc[...]` in `discover`, then `selected.iterrows()`). Within `convert_experiment`, the presentation-overlap test is recomputed per trial even though presentations are sorted, and the quintile edges are computed once but the concatenate/slice bookkeeping re-walks the trial list. Session-level quantities such as `ophys_timestamps` are read once, which is good.

ii.
```python
selected, paths, excluded_no_eye, supplied_count = discover(data_root)   # opens every NWB
image_values = collect_image_names(selected, paths)                      # opens every NWB again
...
neural, inputs, outputs, regions, info = convert_experiment(...)         # opens every NWB a third time
```

iii. No justification given for the repeated passes; they are the price of building the global image vocabulary and the usable-experiment list before conversion begins, which the AI needed so that image codes are consistent across all sessions.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several small items:
- The cumulative sum is computed over the **entire** session (~150k frames) although only the retained trial windows (roughly 5–15% of the session) are ever read out.
- The full event matrix, the full running trace and the full eye-tracking trace are read into memory even though only trial windows are used.
- `pupil_area → 2*sqrt(area/pi)` is a monotonic transform that cannot change the quintile labels, as the docstring itself notes; only the (unsaved) continuous values would have differed.
- `valid_roi` masking and the `np.ascontiguousarray` copy are no-ops on released data where all ROIs are valid.
- Each trial stores an empty `np.empty((0, n_time))` input array (the spec allows no inputs; a single empty array per trial would do), and the static trial outcome is materialised as a full-length int16 row per trial.
- A rich `session_info` record is stored per session in metadata, and neural data are stored as dense float32 even though ~99.8% of event entries are exactly zero — giving a 2.55 GiB pickle where a sparse representation would be far smaller (the dense layout is, however, what the target format and decoder expect).

ii.
```python
np.cumsum(events, axis=0, dtype=np.float32, out=events)   # whole session
...
pupil_diameter = 2.0 * np.sqrt(np.maximum(pupil_area, 0.0) / np.pi)
...
input_trials.append(np.empty((0, n_time), dtype=np.float32))
...
output[4].fill(outcome)
```

iii. The docstring justifies the diameter conversion on interpretability grounds while acknowledging it does not change the labels (*"Since this is monotonic, its percentile labels equal pupil-area percentile labels"*), and the cumulative-sum comment justifies the whole-session cumsum as the cheapest way to serve arbitrary trial-relative bins without re-reading the event dataset. Step 31 notes the size consequence: *"Conversion completed successfully: 199 sessions and 2.55 GiB, with all 51k-ish eligible trials represented at 100 ms resolution."*
