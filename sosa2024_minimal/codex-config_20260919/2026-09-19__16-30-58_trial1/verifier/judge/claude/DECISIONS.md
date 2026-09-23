# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All NWB files matching the pattern `sub-*/*_behavior+ophys.nwb` under the data directory are discovered using `Path.glob()`, sorted by mouse and session number. Each NWB file is opened with `h5py.File` (not pynwb) and data is accessed directly via HDF5 paths (e.g. `nwb["processing/behavior/BehavioralTimeSeries"]`, `nwb["processing/ophys/Fluorescence"]`).

ii.
```python
paths = sorted(data_dir.glob("sub-*/*_behavior+ophys.nwb"), key=natural_key)
...
with h5py.File(path, "r") as nwb:
    behavior = nwb["processing/behavior/BehavioralTimeSeries"]
    position = read_behavior(behavior, "position").astype(np.float64, copy=False)
    ...
```

iii. The agent checked that both h5py and pynwb were available but chose h5py for direct, low-level HDF5 access. No explicit justification was given for preferring h5py over pynwb. The glob pattern discovers all NWB files across all subject directories.

## 1-b. How are the data split into subjects?

i. Subject names are extracted from file paths using a regex pattern `sub-m(\d+)`. Unique mouse numbers are collected from all NWB file paths, sorted, and formatted as `"m{number}"`.

ii.
```python
def natural_key(path: Path) -> tuple[int, int]:
    match = re.search(r"sub-m(\d+).*ses-(\d+)", str(path))
    return int(match.group(1)), int(match.group(2))
...
subject_names = [f"m{mouse}" for mouse in sorted({natural_key(path)[0] for path in paths})]
```

iii. Subject identifiers are parsed directly from the file naming convention `sub-m<number>`.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are identified by their file path and sorted by mouse and session number.

ii.
```python
paths = sorted(data_dir.glob("sub-*/*_behavior+ophys.nwb"), key=natural_key)
...
for path in paths:
    neural_session, input_session, output_session, info = convert_session(path)
```

iii. Each NWB file is a separate recording session, matching the naming convention `ses-<number>`.

## 1-d. How are the data split into trials?

i. Trial boundaries are identified from the `trial_start` and `teleport` behavior time series. The trial start is the index where `trial_start > 0`, and the trial end is where `teleport > 0`. Each trial spans `[start, stop)` — the teleport frame is excluded.

ii.
```python
starts = np.flatnonzero(read_behavior(behavior, "trial_start") > 0)
stops = np.flatnonzero(read_behavior(behavior, "teleport") > 0)
...
trial_slice = slice(int(start), int(stop))
```

iii. The agent verified empirically that position goes up to ~449 cm and then drops to -50 at teleport, confirming the teleport marks the end of the corridor traversal. Metadata states: "teleport onset (excluded from each trial)".

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on lick circuit errors: trials where more than 30% of frames have lick count > 2 are dropped. No minimum trial length filter is applied.

ii.
```python
LICK_ERROR_FRACTION = 0.30
...
if np.mean(trial_lick > 2) > LICK_ERROR_FRACTION:
    dropped_lick_trials.append(trial)
    continue
```

iii. The agent justified this by referencing the paper, which identifies these as lick-circuit errors. Validation confirmed 81 trials were removed, matching the paper's reported count. No running-speed filter was applied because stopped/slow frames are a requested speed class in the decoder.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw `Fluorescence` (F) and `Neuropil` (Fneu) series in the NWB ophys processing module, NOT the NWB's `Deconvolved` field.

ii.
```python
fluorescence = nwb["processing/ophys/Fluorescence"]
neuropil = nwb["processing/ophys/Neuropil"]
...
f_parts.append(
    np.asarray(fluorescence[f"plane{plane}"]["data"][start:stop, local_indices], dtype=np.float32).T
)
fneu_parts.append(
    np.asarray(neuropil[f"plane{plane}"]["data"][start:stop, local_indices], dtype=np.float32).T
)
```

iii. The agent explicitly reasoned that the NWB's "Deconvolved" array is Suite2P's own deconvolution of unnormalized fluorescence, not the paper's processed signal. The paper computes its own dF/F pipeline from the raw fluorescence.

## 2-b. How is the `neural` data processed?

i. The agent reproduces the paper's dF/F pipeline but WITHOUT deconvolution. Per trial: subtract 0.7 * neuropil, add back the neuropil mean, Gaussian smooth with sigma=15 samples, apply 300-sample minimum filter then 300-sample maximum filter (maximin baseline), compute (corrected - baseline) / |baseline|, and smooth dF/F with sigma=2 Gaussian. Cells from multiple planes are pooled. The output is dF/F, not deconvolved events.

ii.
```python
def paper_dff_trial(fluorescence, neuropil, cell_indices_by_plane, start, stop):
    ...
    corrected = f - NEUROPIL_COEFFICIENT * fneu
    corrected += NEUROPIL_COEFFICIENT * np.mean(fneu, axis=1, keepdims=True)
    baseline = gaussian_filter1d(corrected, 15, axis=1)
    baseline = minimum_filter1d(baseline, 300, axis=1)
    baseline = maximum_filter1d(baseline, 300, axis=1)
    denominator = np.abs(baseline)
    denominator[denominator == 0] = np.finfo(np.float32).eps
    dff = (corrected - baseline) / denominator
    dff = gaussian_filter1d(dff, 2, axis=1)
    return np.asarray(dff, dtype=np.float32)
```

iii. The agent stated: "The conversion deliberately follows the paper's fluorescence processing rather than using the NWB Deconvolved series." However, the agent chose NOT to apply the OASIS deconvolution step that the reference paper applies (converting dF/F into "events"), outputting dF/F directly instead.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) Only `iscell`-curated ROIs are kept (Suite2P's manual curation). (2) Putative interneurons with Pearson r(dF/F, speed) > 0.5 are excluded, computed via streaming sufficient statistics across all retained corridor frames.

ii.
```python
segmentation = nwb["processing/ophys/ImageSegmentation/PlaneSegmentation"]
iscell = np.asarray(segmentation["iscell"][:, 0]) > 0
...
cell_indices_by_plane = [np.flatnonzero(iscell[plane_index == plane]) for plane in plane_values]
...
speed_correlations = correlations_from_sums(sum_x, sum_x2, sum_xy, sum_y, sum_y2, n_samples)
keep_neurons = speed_correlations <= INTERNEURON_R_THRESHOLD
neural_trials = [trial[keep_neurons] for trial in neural_trials]
```

iii. The agent found the paper's code uses `int_thresh = 0.5` and `int_method = 'speed'`. Validation confirmed 400 speed-correlated candidates were excluded across all sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to the start of the trial (trial_start marker). Since neural and behavioral data share the same time indices, no additional alignment is needed beyond splitting at trial boundaries.

ii.
```python
trial_slice = slice(int(start), int(stop))
dff = paper_dff_trial(fluorescence, neuropil, cell_indices_by_plane, int(start), int(stop))
```

iii. The metadata records `"temporal_alignment_event": "entry into the virtual corridor (trial-start marker)"` and `"off_start": 0.0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The time bin size is determined by the frame rate: `TIME_BIN_MS = 1000.0 / 15.5078125 ≈ 64.48 ms`.

ii.
```python
FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
```

iii. The frame rate was confirmed from NWB metadata. The agent hardcoded the rate as a constant rather than reading it from each file.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `timestamps` array of the `position` behavior time series.

ii.
```python
timestamps = np.asarray(behavior["position"]["timestamps"], dtype=np.float64)
...
time_from_start = timestamps[trial_slice] - timestamps[start]
```

iii. The timestamps are the common time base for all behavioral variables.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at the trial start is subtracted from all timestamps within the trial.

ii.
```python
time_from_start = timestamps[trial_slice] - timestamps[start]
```

iii. Straightforward computation: each trial starts at t=0.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same time indices (same sampling rate), so indexing by the same trial slice ensures alignment. No interpolation is needed.

ii.
```python
trial_slice = slice(int(start), int(stop))
dff = paper_dff_trial(..., int(start), int(stop))
time_from_start = timestamps[trial_slice] - timestamps[start]
```

iii. The agent verified that the median frame interval matches 1/FRAME_RATE_HZ.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
environment = read_behavior(behavior, "environment")
...
env_values = environment[trial_slice]
valid_env = env_values[env_values >= 0]
env = float(np.rint(np.median(valid_env)))
```

iii. The environment variable is constant within each trial and binary (0 or 1), corresponding to ENV1 and ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The median of valid (>=0) environment values within the trial is taken and rounded. This is a robustness measure since the value should be constant within a trial.

ii.
```python
valid_env = env_values[env_values >= 0]
env = float(np.rint(np.median(valid_env)))
...
np.full(dff.shape[1], env),
```

iii. Taking the median handles any edge cases with negative sentinel values.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number` behavior time series in the NWB file, verified against the sequential trial index from `trial_start` markers.

ii.
```python
trial_number_stream = read_behavior(behavior, "trial number")
...
source_trial_number = float(np.rint(np.median(trial_number_stream[trial_slice])))
if source_trial_number != trial:
    raise ValueError(...)
np.full(dff.shape[1], source_trial_number),
```

iii. The agent uses the NWB's stored trial number but cross-checks it against the marker-derived index, raising an error if they disagree.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The median of the `trial number` stream within the trial slice is taken and rounded. This is verified to match the zero-based marker index.

ii.
```python
source_trial_number = float(np.rint(np.median(trial_number_stream[trial_slice])))
```

iii. No additional processing beyond reading and verifying.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps and the trial boundaries (from `trial_start` and `teleport`).

ii.
```python
reward_timestamps = np.asarray(behavior["Reward"]["timestamps"], dtype=np.float64)
...
outcomes = reward_outcomes(timestamps, starts, stops, reward_timestamps)
previous_outcomes = np.r_[0, outcomes[:-1]].astype(np.float32)
```

iii. The Reward time series has its own timestamps separate from the behavior sampling rate.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. First, per-trial reward outcomes are computed by checking if any reward timestamp falls within [trial_start_time, teleport_time]. Then previous trial outcome is a shifted version of this array, with the first trial set to 0.

ii.
```python
def reward_outcomes(timestamps, starts, stops, reward_timestamps):
    outcomes = np.zeros(len(starts), dtype=np.int8)
    for trial, (start, stop) in enumerate(zip(starts, stops)):
        outcomes[trial] = np.any(
            (reward_timestamps >= timestamps[start])
            & (reward_timestamps <= timestamps[stop])
        )
    return outcomes
...
previous_outcomes = np.r_[0, outcomes[:-1]].astype(np.float32)
```

iii. The first trial's previous outcome is set to 0 because the outcome of the pre-imaging warm-up trial is unavailable.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the inferred reward zone label for the current trial. Reward zone labels are inferred from `Reward` timestamps and position (not from the `reward_zone` behavior variable).

ii.
```python
position = read_behavior(behavior, "position").astype(np.float64, copy=False)
reward_timestamps = np.asarray(behavior["Reward"]["timestamps"], dtype=np.float64)
zones, block_zone_labels, block_reward_medians = infer_zone_by_trial(
    timestamps, position, starts, stops, reward_timestamps
)
```

iii. The agent infers reward zone labels per block (trials 0-29 and trials 30+) from median actual reward delivery position, matched to the nearest of the known A/B/C zone centers. This differs from the reference, which uses the `reward_zone` behavior variable with a Viterbi algorithm.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed linear distance to the nearest edge of the active 50 cm reward zone: negative before the zone, zero throughout the zone, positive after. Zone bounds: A=[80,130], B=[200,250], C=[320,370].

ii.
```python
def distance_classes(position, zone):
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

iii. Distance is computed relative to the known zone boundaries from the paper's behavior code.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Direct conditional assignment into 7 bins: 0 (<-50), 1 (-50 to -10), 2 (-10 to 0), 3 (in zone, distance=0), 4 (>0 to 10), 5 (10 to 50), 6 (>50).

ii.
```python
result = np.full(position.shape, 3, dtype=np.int8)  # default: in zone
result[distance < -50.0] = 0
result[(distance >= -50.0) & (distance < -10.0)] = 1
result[(distance >= -10.0) & (distance < 0.0)] = 2
result[(distance > 0.0) & (distance <= 10.0)] = 4
result[(distance > 10.0) & (distance <= 50.0)] = 5
result[distance > 50.0] = 6
```

iii. Matches the instruction specification. Uses explicit conditionals rather than `np.digitize`.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same time indices as the neural data within each trial — both use the same `trial_slice`.

ii.
```python
trial_slice = slice(int(start), int(stop))
trial_position = position[trial_slice]
dff = paper_dff_trial(..., int(start), int(stop))
```

iii. Same sampling rate and time indexing ensures alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = read_behavior(behavior, "position").astype(np.float64, copy=False)
trial_position = position[trial_slice]
```

iii. The position variable directly records the animal's position in the VR corridor in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice and discretizing.

ii.
```python
np.digitize(trial_position, [90.0, 180.0, 270.0, 360.0]).astype(np.int8)
```

iii. Raw position values used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins using `np.digitize` with bin edges [90, 180, 270, 360]. This gives bins 0 (<90), 1 (90-180), 2 (180-270), 3 (270-360), 4 (>=360).

ii.
```python
np.digitize(trial_position, [90.0, 180.0, 270.0, 360.0]).astype(np.int8)
```

iii. Five 90 cm bins spanning the 450 cm track as specified in the instructions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data — same `trial_slice`.

ii. `trial_position = position[trial_slice]`

iii. Same sampling rate and indexing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = read_behavior(behavior, "lick")
trial_lick = lick[trial_slice]
```

iii. The lick variable records lick events at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value mapped to 1, otherwise 0.

ii.
```python
(trial_lick > 0).astype(np.int8)
```

iii. The instructions specify binary output (no/yes). Raw lick values can be >1, so thresholding at >0 converts to binary.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data — same `trial_slice`.

ii. `trial_lick = lick[trial_slice]`

iii. Same sampling rate and indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `Reward` timestamps and `position` behavior time series. The `reward_zone` behavior variable in the NWB is NOT used. Instead, reward zone labels are inferred per block from the median actual reward delivery position.

ii.
```python
zones, block_zone_labels, block_reward_medians = infer_zone_by_trial(
    timestamps, position, starts, stops, reward_timestamps
)
...
np.full(dff.shape[1], zones[trial], dtype=np.int8)
```

iii. The agent infers one label per block (trials 0-29 and trials >=30), matching it to the nearest A/B/C zone center. On stay days both blocks get the same label. Omission trials inherit their block's label.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Reward positions are linearly interpolated at exact delivery timestamps. The median reward position within each block (pre-switch and post-switch) is computed and matched to the nearest zone center. Output is encoded as 0=A, 1=B, 2=C.

ii.
```python
def infer_zone_by_trial(timestamps, position, starts, stops, reward_timestamps):
    boundaries = [(0, min(SWITCH_TRIAL, n_trials)), (min(SWITCH_TRIAL, n_trials), n_trials)]
    for first, last in boundaries:
        block_rewards = reward_timestamps[(reward_timestamps >= lo) & (reward_timestamps <= hi)]
        reward_positions = np.interp(block_rewards, timestamps, position)
        median_position = float(np.median(reward_positions))
        label = int(np.argmin(np.abs(ZONE_CENTERS - median_position)))
        labels[first:last] = label
```

iii. The agent used block-level assignment rather than per-trial assignment to make omission trials well-defined and to be more robust than per-trial zone inference.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps.

ii.
```python
reward_timestamps = np.asarray(behavior["Reward"]["timestamps"], dtype=np.float64)
outcomes = reward_outcomes(timestamps, starts, stops, reward_timestamps)
```

iii. The Reward time series records reward delivery events with separate timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward timestamp falls within [trial_start_time, teleport_time]. Binary output: 1 if rewarded, 0 if not. Constant across all timepoints in the trial.

ii.
```python
def reward_outcomes(timestamps, starts, stops, reward_timestamps):
    outcomes = np.zeros(len(starts), dtype=np.int8)
    for trial, (start, stop) in enumerate(zip(starts, stops)):
        outcomes[trial] = np.any(
            (reward_timestamps >= timestamps[start])
            & (reward_timestamps <= timestamps[stop])
        )
    return outcomes
```

iii. Validation confirmed ~84.2% rewarded, ~15.8% not rewarded, consistent with the paper's ~15% omission rate.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several checks are made:
- Start/stop pairing is validated (same count, stops after starts)
- Frame interval is verified against expected rate
- Environment label validity is checked (must have valid values >=0)
- Trial number is cross-checked against marker-derived index
- Zero-division in dF/F baseline is guarded with epsilon
- Lick-error trials are dropped
- Sessions must have at least 2 usable trials

ii.
```python
if len(starts) != len(stops) or np.any(stops <= starts):
    raise ValueError(f"Invalid start/teleport pairing in {path}")
...
denominator[denominator == 0] = np.finfo(np.float32).eps
...
if len(neural_session) < 2:
    raise ValueError(f"Fewer than two usable trials in {path}")
```

iii. Errors raise exceptions rather than silently handling mismatches. The agent chose to be strict rather than permissive.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files and reading large fluorescence/neuropil arrays via h5py
2. Computing the dF/F pipeline per trial (Gaussian smoothing, min/max filters) — done for every trial individually
3. The streaming correlation computation for interneuron filtering

ii. N/A

iii. The per-trial dF/F computation is particularly expensive because it processes fluorescence data trial-by-trial rather than in bulk.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates over each trial, computing dF/F independently for each. The `reward_outcomes` function loops over trials to check reward timestamps. The Viterbi-like zone inference uses a loop over blocks (minimal). The streaming correlation accumulator loops over trials but this is inherent to the streaming approach.

ii. N/A

iii. The per-trial dF/F computation is necessary because the paper's pipeline processes each trial independently (per-trial baseline). However, reading fluorescence data from HDF5 per trial is much slower than reading the whole session array once.

## 13-c. What processing does the code repeat multiple times?

i. The code reads each NWB file only once (no separate survey step). However, fluorescence data is read from HDF5 per trial (each `paper_dff_trial` call reads a slice from the HDF5 dataset), which involves repeated I/O to the same file.

ii. N/A

iii. The h5py-based per-trial reading avoids materializing the full session-wide fluorescence matrix in memory, but at the cost of repeated file I/O.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does not apply OASIS deconvolution, so dF/F is the final neural signal. No obviously discarded processing is performed. However, the code does NOT apply deconvolution which the reference solution does, meaning the neural signal is in a different format (dF/F vs deconvolved events).

ii. N/A

iii. The agent chose dF/F as the neural signal, which is a different representation than the deconvolved events the reference paper uses for its analyses.
