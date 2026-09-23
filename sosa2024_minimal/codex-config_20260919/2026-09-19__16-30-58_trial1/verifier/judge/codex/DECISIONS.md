# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads every `*_behavior+ophys.nwb` file under `/app/data/sub-*` using `Path.glob`, sorts them by parsed mouse/session numbers, and processes each file as one session with `h5py`. Within each session it reads behavior streams and ophys groups directly from the NWB hierarchy.

ii.
```python
def convert(data_dir: Path) -> dict:
    paths = sorted(data_dir.glob("sub-*/*_behavior+ophys.nwb"), key=natural_key)
    if not paths:
        raise FileNotFoundError(f"No NWB files found below {data_dir}")
```
```python
with h5py.File(path, "r") as nwb:
    behavior = nwb["processing/behavior/BehavioralTimeSeries"]
    position = read_behavior(behavior, "position").astype(np.float64, copy=False)
    speed = read_behavior(behavior, "speed").astype(np.float64, copy=False)
    ...
    segmentation = nwb["processing/ophys/ImageSegmentation/PlaneSegmentation"]
    fluorescence = nwb["processing/ophys/Fluorescence"]
    neuropil = nwb["processing/ophys/Neuropil"]
```

iii. In trajectory step 6 the agent says it will inspect the raw data layout and implement a reproducible converter. In step 20 it says the NWBs contain synchronized frame-level behavior plus raw Suite2P fluorescence/neuropil, and that it will reconstruct the paper’s neural signal from those fields rather than rely on stored deconvolution.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the `sub-mXX` component of each file path. The final `subjects` list is the sorted unique mouse identifiers, formatted as `m11`, `m12`, etc.

ii.
```python
def natural_key(path: Path) -> tuple[int, int]:
    match = re.search(r"sub-m(\d+).*ses-(\d+)", str(path))
    ...
    return int(match.group(1)), int(match.group(2))
```
```python
subject_names = [f"m{mouse}" for mouse in sorted({natural_key(path)[0] for path in paths})]
subject_lookup = {subject: index for index, subject in enumerate(subject_names)}
```

iii. The trajectory does not contain a separate explicit argument for subject splitting beyond step 20’s summary that the dataset is a multi-mouse NWB release and that all 152 sessions are processed.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Sessions are identified from the `ses-XX` portion of the filename and kept in sorted mouse/session order.

ii.
```python
def natural_key(path: Path) -> tuple[int, int]:
    match = re.search(r"sub-m(\d+).*ses-(\d+)", str(path))
    ...
    return int(match.group(1)), int(match.group(2))
```
```python
for path in paths:
    neural_session, input_session, output_session, info = convert_session(path)
    ...
```

iii. In step 24 the agent reports a one-session dry run and then says the “full conversion is now running across all 152 available imaging sessions,” confirming the one-file-per-session interpretation.

## 1-d. How are the data split into trials?

i. Trials are defined by the `trial_start` and `teleport` behavior streams. The AI finds all positive `trial_start` samples as starts and all positive `teleport` samples as stops, then slices each trial as `[start, stop)`, including the trial-start sample and excluding the teleport sample.

ii.
```python
starts = np.flatnonzero(read_behavior(behavior, "trial_start") > 0)
stops = np.flatnonzero(read_behavior(behavior, "teleport") > 0)

if len(starts) != len(stops) or np.any(stops <= starts):
    raise ValueError(f"Invalid start/teleport pairing in {path}")
```
```python
for trial, (start, stop) in enumerate(zip(starts, stops)):
    # The alignment marker is included as t=0; the teleport marker is
    # excluded, leaving only the 450 cm virtual corridor.
    trial_slice = slice(int(start), int(stop))
```

iii. In trajectory step 20 the agent states: “Trials will span the trial-start marker through the sample before teleport.” That is exactly the slicing implemented here.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops trials with corrupted lick traces: if more than 30% of corridor frames in a trial have `lick > 2`, that trial is skipped. It does not apply the human reference’s minimum-length trial filter.

ii.
```python
LICK_ERROR_FRACTION = 0.30
...
trial_lick = lick[trial_slice]
if np.mean(trial_lick > 2) > LICK_ERROR_FRACTION:
    dropped_lick_trials.append(trial)
    continue
```
```python
"trial_filtering": (
    "Trials with lick count >2 in more than 30% of corridor frames were "
    "dropped because the paper identifies these as lick-circuit errors. No "
    "running-speed filter was applied because stopped/slow frames are a "
    "requested speed class."
),
```

iii. Step 20 explicitly says it will “drop only trials with the paper-defined corrupted lick pattern.” Step 46 says this removed exactly 81 trials, which the agent cites as matching the paper.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from the raw `Fluorescence` and `Neuropil` ROI response series in the NWB file, after restricting to ROIs marked as `iscell`.

ii.
```python
segmentation = nwb["processing/ophys/ImageSegmentation/PlaneSegmentation"]
iscell = np.asarray(segmentation["iscell"][:, 0]) > 0
plane_index = np.asarray(segmentation["planeIdx"], dtype=np.int64)
...
fluorescence = nwb["processing/ophys/Fluorescence"]
neuropil = nwb["processing/ophys/Neuropil"]
```
```python
f_parts.append(
    np.asarray(
        fluorescence[f"plane{plane}"]["data"][start:stop, local_indices],
        dtype=np.float32,
    ).T
)
fneu_parts.append(
    np.asarray(
        neuropil[f"plane{plane}"]["data"][start:stop, local_indices],
        dtype=np.float32,
    ).T
)
```

iii. Step 20 says the NWB `Deconvolved` array is not the paper’s analysis signal and that the converter will instead reconstruct from “raw Suite2P ROI/neuropil fluorescence” from curated `iscell` ROIs.

## 2-b. How is the `neural` data processed?

i. The AI computes trial-wise dF/F from fluorescence and neuropil using the paper’s maximin-style baseline pipeline, but stops there. It does not deconvolve the dF/F into events/spikes, and it does not implement the session-specific `keep_teleports` handling from the paper/reference code.

ii.
```python
corrected = f - NEUROPIL_COEFFICIENT * fneu
corrected += NEUROPIL_COEFFICIENT * np.mean(fneu, axis=1, keepdims=True)
baseline = gaussian_filter1d(corrected, 15, axis=1)
baseline = minimum_filter1d(baseline, 300, axis=1)
baseline = maximum_filter1d(baseline, 300, axis=1)
...
dff = (corrected - baseline) / denominator
dff = gaussian_filter1d(dff, 2, axis=1)
return np.asarray(dff, dtype=np.float32)
```
```python
"neural_signal": "trial-wise neuropil-corrected, maximin-baselined, Gaussian-smoothed dF/F",
```

iii. Step 20 says it will reconstruct the signal via “0.7 neuropil subtraction, per-trial 20 s maximin baseline, ΔF/F, and the reported two-frame Gaussian smoothing.” The trajectory never claims it will deconvolve; the saved metadata also describes the final signal only as dF/F.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data are filtered twice: first by `iscell` manual curation, then by excluding putative interneurons whose dF/F correlates with running speed above 0.5.

ii.
```python
iscell = np.asarray(segmentation["iscell"][:, 0]) > 0
...
cell_indices_by_plane = [
    np.flatnonzero(iscell[plane_index == plane]) for plane in plane_values
]
```
```python
speed_correlations = correlations_from_sums(
    sum_x, sum_x2, sum_xy, sum_y, sum_y2, n_samples
)
keep_neurons = speed_correlations <= INTERNEURON_R_THRESHOLD
neural_trials = [trial[keep_neurons] for trial in neural_trials]
```

iii. Step 20 states it will use curated `iscell` ROIs and “exclude the additional dF/F–speed-correlation >0.5 interneuron candidates.” Step 46 reports that 400 such candidates were excluded.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to trial start by trial slicing itself: each trial matrix begins at the `trial_start` sample and ends before `teleport`. No separate realignment is applied after slicing.

ii.
```python
for trial, (start, stop) in enumerate(zip(starts, stops)):
    # The alignment marker is included as t=0; the teleport marker is
    # excluded, leaving only the 450 cm virtual corridor.
    trial_slice = slice(int(start), int(stop))
    ...
    neural_trials.append(dff)
```

iii. Step 20 explicitly says the trials span from trial-start to pre-teleport, and the metadata says the temporal alignment event is “entry into the virtual corridor (trial-start marker).”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native frame rate and uses a fixed bin size of `1000 / 15.5078125 = 64.48 ms`. It does not rebin or resample the neural data.

ii.
```python
FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
```
```python
frame_intervals = np.diff(timestamps)
if not np.isclose(np.median(frame_intervals), 1.0 / FRAME_RATE_HZ, rtol=0, atol=1e-6):
    raise ValueError(f"Unexpected behavior sampling interval in {path}")
```
```python
"time_bin_size": TIME_BIN_MS,
```

iii. In step 20 the agent says it will retain the “native synchronized 64.48 ms frames.” No trajectory message suggests any temporal binning beyond preserving the native sampling grid.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavior timestamps attached to the `position` time series.

ii.
```python
timestamps = np.asarray(behavior["position"]["timestamps"], dtype=np.float64)
...
time_from_start = timestamps[trial_slice] - timestamps[start]
```

iii. The trajectory does not separately justify why `position` timestamps were chosen over another behavior series; the apparent assumption is that the frame-level behavior streams are synchronized and share the same timeline.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the timestamp at the trial start is subtracted from every timestamp in that trial slice.

ii.
```python
time_from_start = timestamps[trial_slice] - timestamps[start]
...
inputs = np.vstack(
    [
        time_from_start,
        ...
    ]
).astype(np.float32)
```

iii. This is implied by step 20’s plan to align everything at trial start. No extra trajectory explanation beyond that is recorded.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses the same `trial_slice` and same per-frame timeline as the neural trial, so it is frame-aligned to the neural matrix by construction.

ii.
```python
trial_slice = slice(int(start), int(stop))
...
dff = paper_dff_trial(
    fluorescence, neuropil, cell_indices_by_plane, int(start), int(stop)
)
...
time_from_start = timestamps[trial_slice] - timestamps[start]
```

iii. Step 20 frames the whole conversion around “native synchronized” frame-level data and a shared trial-start alignment event.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the `environment` behavior time series.

ii.
```python
environment = read_behavior(behavior, "environment")
...
env_values = environment[trial_slice]
```

iii. The trajectory does not include a separate explanation for this field; it is inferred from the code and from the general step-20 statement that the NWBs contain synchronized frame-level behavior fields needed for decoder inputs.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI takes the median nonnegative environment label within each trial, rounds it to an integer, and repeats that single per-trial value across all time bins in the trial.

ii.
```python
env_values = environment[trial_slice]
valid_env = env_values[env_values >= 0]
if valid_env.size == 0:
    raise ValueError(f"No environment label in {path}, trial {trial}")
env = float(np.rint(np.median(valid_env)))
...
np.full(dff.shape[1], env),
```

iii. The trajectory does not state this explicitly. The code suggests the AI assumed environment is a trial-level condition and used a robust summary to guard against occasional invalid samples.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the `trial number` behavior time series, but only after trial boundaries are defined from `trial_start` and `teleport`. The AI checks that the median trial-number value inside each sliced trial matches the loop index.

ii.
```python
trial_number_stream = read_behavior(behavior, "trial number")
...
source_trial_number = float(np.rint(np.median(trial_number_stream[trial_slice])))
if source_trial_number != trial:
    raise ValueError(
        f"Unexpected trial number {source_trial_number} for marker {trial} in {path}"
    )
```

iii. There is no explicit trajectory defense of this choice. The code indicates the AI wanted to use the stored trial-number stream while still validating it against the marker-defined trial ordering.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The median trial-number value over the trial slice is rounded and then broadcast across all timepoints in that trial.

ii.
```python
source_trial_number = float(np.rint(np.median(trial_number_stream[trial_slice])))
...
np.full(dff.shape[1], source_trial_number),
```

iii. The trajectory contains no separate explanation beyond the general claim in step 20 that the data streams are synchronized and trial-sliced on the native frame grid.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from reward delivery timestamps (`Reward/timestamps`) together with the trial start/stop indices and behavior timestamps.

ii.
```python
reward_timestamps = np.asarray(behavior["Reward"]["timestamps"], dtype=np.float64)
...
outcomes = reward_outcomes(timestamps, starts, stops, reward_timestamps)
previous_outcomes = np.r_[0, outcomes[:-1]].astype(np.float32)
```

iii. The trajectory does not isolate this variable, but step 20 says omission trials should be handled correctly and that reward-based variables will be inferred from actual reward deliveries.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI first computes a binary reward outcome for each trial, then shifts that vector by one trial and prepends `0` for the first trial. The resulting previous-outcome value is repeated across all time bins of each trial.

ii.
```python
def reward_outcomes(
    timestamps: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    reward_timestamps: np.ndarray,
) -> np.ndarray:
    outcomes = np.zeros(len(starts), dtype=np.int8)
    for trial, (start, stop) in enumerate(zip(starts, stops)):
        outcomes[trial] = np.any(
            (reward_timestamps >= timestamps[start])
            & (reward_timestamps <= timestamps[stop])
        )
    return outcomes
```
```python
previous_outcomes = np.r_[0, outcomes[:-1]].astype(np.float32)
...
np.full(dff.shape[1], previous_outcomes[trial]),
```

iii. The metadata records the rationale for the first trial: “Set to 0 because the outcome of the pre-imaging warm-up trial is unavailable.”

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from `position` plus an inferred reward-zone label for each trial. The zone label is not taken from the NWB `reward_zone` time series; instead it is inferred from reward delivery timestamps and interpolated reward positions, separately for trials `0-29` and `>=30`.

ii.
```python
position = read_behavior(behavior, "position").astype(np.float64, copy=False)
reward_timestamps = np.asarray(behavior["Reward"]["timestamps"], dtype=np.float64)
...
zones, block_zone_labels, block_reward_medians = infer_zone_by_trial(
    timestamps, position, starts, stops, reward_timestamps
)
```
```python
def infer_zone_by_trial(...):
    boundaries = [(0, min(SWITCH_TRIAL, n_trials)), (min(SWITCH_TRIAL, n_trials), n_trials)]
    ...
    reward_positions = np.interp(block_rewards, timestamps, position)
    median_position = float(np.median(reward_positions))
    label = int(np.argmin(np.abs(ZONE_CENTERS - median_position)))
    labels[first:last] = label
```

iii. Step 20 explicitly justifies this: “Reward-zone labels will be inferred per pre/post-switch block from actual reward delivery positions and the known A/B/C coordinates; omission trials inherit their block’s active zone.”

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each frame, the AI computes signed linear distance to the nearest point in the active reward zone: negative before the zone, zero inside the zone, positive after the zone. It then maps those signed distances into the 7 requested categories.

ii.
```python
def distance_classes(position: np.ndarray, zone: int) -> np.ndarray:
    zone_start, zone_stop = ZONE_BOUNDS[zone]
    distance = np.where(
        position < zone_start,
        position - zone_start,
        np.where(position > zone_stop, position - zone_stop, 0.0),
    )
    result = np.full(position.shape, 3, dtype=np.int8)
    result[distance < -50.0] = 0
    result[(distance >= -50.0) & (distance < -10.0)] = 1
    result[(distance >= -10.0) & (distance < 0.0)] = 2
    result[(distance > 0.0) & (distance <= 10.0)] = 4
    result[(distance > 10.0) & (distance <= 50.0)] = 5
    result[distance > 50.0] = 6
    return result
```

iii. The metadata says the distance definition is “Signed linear distance to the nearest point in the active reward zone,” and step 20 says omitted trials inherit the block’s active zone before this distance computation.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI hard-codes the requested 7 bins using comparison logic rather than `np.digitize`: `< -50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, and `> 50`.

ii.
```python
result = np.full(position.shape, 3, dtype=np.int8)
result[distance < -50.0] = 0
result[(distance >= -50.0) & (distance < -10.0)] = 1
result[(distance >= -10.0) & (distance < 0.0)] = 2
result[(distance > 0.0) & (distance <= 10.0)] = 4
result[(distance > 10.0) & (distance <= 50.0)] = 5
result[distance > 50.0] = 6
```

iii. No separate trajectory argument is given, but the implementation matches the bin specification the agent was working from.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from `trial_position = position[trial_slice]`, where `trial_slice` is the same slice used for the neural dF/F trial, so alignment is by shared frame indices within each trial.

ii.
```python
trial_slice = slice(int(start), int(stop))
...
trial_position = position[trial_slice]
...
distance_classes(trial_position, int(zones[trial])),
```

iii. Step 20 emphasizes native synchronized frame-level behavior and trial-start alignment, which is the basis of this framewise alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the `position` behavior time series.

ii.
```python
position = read_behavior(behavior, "position").astype(np.float64, copy=False)
...
trial_position = position[trial_slice]
```

iii. The trajectory does not discuss this field separately; it is straightforwardly read from the synchronized behavior stream.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI takes the raw per-trial position samples and discretizes them into five 90 cm bins with `np.digitize`.

ii.
```python
np.digitize(trial_position, [90.0, 180.0, 270.0, 360.0]).astype(np.int8)
```

iii. The trajectory does not include additional justification beyond using the requested decoder categories.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The thresholds are `(-inf,90)`, `[90,180)`, `[180,270)`, `[270,360)`, and `[360,inf)`, implemented through `np.digitize` with inner boundaries `[90, 180, 270, 360]`.

ii.
```python
np.digitize(trial_position, [90.0, 180.0, 270.0, 360.0]).astype(np.int8)
```

iii. No separate trajectory note appears; the binning follows the task specification directly.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same `trial_slice` as the neural data and therefore the same within-trial frame indices.

ii.
```python
trial_slice = slice(int(start), int(stop))
...
trial_position = position[trial_slice]
...
outputs = np.vstack(
    [
        ...,
        np.digitize(trial_position, [90.0, 180.0, 270.0, 360.0]).astype(np.int8),
        ...
    ]
)
```

iii. Step 20’s “native synchronized” description is the only explicit trajectory justification.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the `lick` behavior time series.

ii.
```python
lick = read_behavior(behavior, "lick")
...
trial_lick = lick[trial_slice]
```

iii. The trajectory focuses on lick mainly as a quality-control signal; step 20 says the agent will drop trials with the paper-defined corrupted lick pattern.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI binarizes the lick trace so that any positive lick value becomes `1` and zero becomes `0`.

ii.
```python
(trial_lick > 0).astype(np.int8)
```

iii. No separate trajectory discussion is recorded beyond the broader handling of lick corruption in steps 20 and 46.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is sliced with the same `trial_slice` used for the neural dF/F matrix, so it is aligned frame by frame within each trial.

ii.
```python
trial_slice = slice(int(start), int(stop))
trial_lick = lick[trial_slice]
...
(trial_lick > 0).astype(np.int8),
```

iii. This follows the same synchronized-frame assumption stated in step 20.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The AI derives reward-zone location from reward delivery timestamps plus position, using the same blockwise inference described for distance-to-zone. It does not use the `reward_zone` behavior stream.

ii.
```python
zones, block_zone_labels, block_reward_medians = infer_zone_by_trial(
    timestamps, position, starts, stops, reward_timestamps
)
...
np.full(dff.shape[1], zones[trial], dtype=np.int8),
```

iii. Step 20 gives the rationale: infer the active zone per pre/post-switch block from actual reward delivery positions so omission trials can inherit a label.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI splits each session into two blocks (`0-29` and `>=30`), interpolates reward delivery positions within each block, takes the median reward position, assigns the nearest of the known zone centers A/B/C, and broadcasts that block label to every trial in the block.

ii.
```python
boundaries = [(0, min(SWITCH_TRIAL, n_trials)), (min(SWITCH_TRIAL, n_trials), n_trials)]
...
reward_positions = np.interp(block_rewards, timestamps, position)
median_position = float(np.median(reward_positions))
label = int(np.argmin(np.abs(ZONE_CENTERS - median_position)))
labels[first:last] = label
```

iii. The exact explanation appears in step 20 and is repeated in the saved metadata under `reward_zone_inference`.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/timestamps`, together with the trial boundaries and behavior timestamps.

ii.
```python
reward_timestamps = np.asarray(behavior["Reward"]["timestamps"], dtype=np.float64)
...
outcomes = reward_outcomes(timestamps, starts, stops, reward_timestamps)
```

iii. The trajectory treats reward deliveries as the authoritative source both for omission/reward outcomes and for reward-zone inference.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the AI marks reward outcome as `1` if any reward timestamp falls between the trial’s start and stop timestamps, otherwise `0`. That single binary value is then repeated across all time bins in the trial.

ii.
```python
for trial, (start, stop) in enumerate(zip(starts, stops)):
    outcomes[trial] = np.any(
        (reward_timestamps >= timestamps[start])
        & (reward_timestamps <= timestamps[stop])
    )
```
```python
np.full(dff.shape[1], outcomes[trial], dtype=np.int8),
```

iii. The trajectory does not discuss this variable separately, but step 20 says the converter will use actual reward delivery events and handle omission trials explicitly.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly handles unexpected data by failing fast rather than repairing it. It raises errors for invalid start/stop pairings, non-contiguous plane labels, unexpected frame intervals, missing environment labels, absent reward deliveries in a block, and trial-number mismatches. It also guards against zero fluorescence baselines and drops lick-corrupt trials. It does not implement the human reference’s cropping for neural/behavior length mismatches or missing-reward-zone smoothing.

ii.
```python
if len(starts) != len(stops) or np.any(stops <= starts):
    raise ValueError(f"Invalid start/teleport pairing in {path}")
...
if not np.isclose(np.median(frame_intervals), 1.0 / FRAME_RATE_HZ, rtol=0, atol=1e-6):
    raise ValueError(f"Unexpected behavior sampling interval in {path}")
...
if valid_env.size == 0:
    raise ValueError(f"No environment label in {path}, trial {trial}")
...
if block_rewards.size == 0:
    raise ValueError(f"No reward deliveries in trial block [{first}, {last})")
```
```python
denominator = np.abs(baseline)
denominator[denominator == 0] = np.finfo(np.float32).eps
```

iii. The trajectory suggests the agent preferred strict validation: step 43 says it preserved all 152 sessions and then ran the format verifier; step 46 emphasizes that verification passed with no warnings after the chosen filtering.

## 13-a. What are the most time-consuming steps of the code?

i. The likely expensive steps are reading each NWB file, computing trial-wise dF/F from fluorescence and neuropil for every trial, computing speed-correlation statistics for interneuron filtering, and serializing the final 9.52 GB pickle. Unlike the human reference, this AI code does not do a separate survey pass over the data.

ii.
```python
with h5py.File(path, "r") as nwb:
    ...
for trial, (start, stop) in enumerate(zip(starts, stops)):
    ...
    dff = paper_dff_trial(
        fluorescence, neuropil, cell_indices_by_plane, int(start), int(stop)
    )
```
```python
with args.output.open("wb") as stream:
    pickle.dump(converted, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory supports this: steps 38-42 show the long-running session-by-session conversion; step 43 reports the artifact is 9.52 GB; step 52 notes the decoder run is waiting on later training rather than on conversion format checks.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The biggest non-vectorized work is the per-trial conversion loop, including repeated behavioral slicing and repeated `paper_dff_trial` calls, plus the `reward_outcomes` loop and the small block loop in `infer_zone_by_trial`. Some behavioral discretizations could be done session-wide before splitting, but the dF/F computation is inherently trial-local in this implementation.

ii.
```python
for trial, (start, stop) in enumerate(zip(starts, stops)):
    ...
    dff = paper_dff_trial(...)
    ...
    outputs = np.vstack(
        [
            distance_classes(trial_position, int(zones[trial])),
            np.digitize(trial_position, [90.0, 180.0, 270.0, 360.0]).astype(np.int8),
            np.digitize(trial_speed, [2.0, 10.0, 20.0, 40.0]).astype(np.int8),
            (trial_lick > 0).astype(np.int8),
            ...
        ]
    )
```
```python
for trial, (start, stop) in enumerate(zip(starts, stops)):
    outcomes[trial] = np.any(
        (reward_timestamps >= timestamps[start])
        & (reward_timestamps <= timestamps[stop])
    )
```

iii. The trajectory does not include an explicit efficiency analysis. The agent instead seems to have prioritized avoiding a second survey pass while keeping the trial-local preprocessing faithful to its chosen interpretation of the paper.

## 13-c. What processing does the code repeat multiple times?

i. The AI avoids the human reference’s largest repetition, namely re-reading every file in a separate survey phase. Within its single pass, it still repeats per-trial fluorescence loading, Gaussian/filter operations, per-trial slicing, and per-trial binning for every trial in every session.

ii.
```python
for trial, (start, stop) in enumerate(zip(starts, stops)):
    ...
    dff = paper_dff_trial(
        fluorescence, neuropil, cell_indices_by_plane, int(start), int(stop)
    )
```
```python
for plane, local_indices in enumerate(cell_indices_by_plane):
    ...
    fluorescence[f"plane{plane}"]["data"][start:stop, local_indices]
    ...
    neuropil[f"plane{plane}"]["data"][start:stop, local_indices]
```

iii. The trajectory indicates this was a deliberate simplification: step 24 says a one-session dry run succeeded and then the same conversion path was applied to all 152 sessions without any separate exploratory pass.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes some values mainly for validation or metadata rather than for downstream decoder inputs/outputs: the `source_trial_number` median is checked then only repeated into the input because the task asked for trial number, block reward medians are stored only in metadata, and bookkeeping fields such as kept/dropped source-trial lists are saved but not used by the decoder. It also performs strict assertions on frame timing and trial numbering that serve validation rather than downstream modeling.

ii.
```python
source_trial_number = float(np.rint(np.median(trial_number_stream[trial_slice])))
if source_trial_number != trial:
    raise ValueError(
        f"Unexpected trial number {source_trial_number} for marker {trial} in {path}"
    )
```
```python
info = {
    "source_trial_count": int(len(starts)),
    "kept_source_trials": kept_source_trials,
    "dropped_corrupt_lick_trials": dropped_lick_trials,
    ...
    "reward_zone_by_block": block_zone_labels,
    "median_reward_position_cm_by_block": block_reward_medians,
}
```

iii. The trajectory does not discuss unnecessary work directly. The closest evidence is step 43-46, where the agent emphasizes verification, statistics, and metadata reporting in addition to producing the decoder-ready pickle.
