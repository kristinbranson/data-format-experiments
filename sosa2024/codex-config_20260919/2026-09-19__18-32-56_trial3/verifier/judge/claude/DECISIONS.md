# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All NWB files under `/app/data/sub-*/` are discovered via glob, sorted by numeric mouse ID and session ID. Each file is opened with `h5py` (not `pynwb`) and all behavior, neural, and metadata fields are read directly from HDF5 paths. All 152 sessions across 11 subjects are included.

ii.
```python
def discover_files(sample: bool) -> list[Path]:
    files = sorted(DATA_ROOT.glob("sub-*/*.nwb"), key=natural_file_key)
    ...
    return files
```
```python
with h5py.File(path, "r") as nwb:
    behavior = nwb[BEHAVIOR_ROOT]
    timestamps = behavior["position/timestamps"][:]
    position = behavior["position/data"][:]
    ...
```

iii. The AI explored the data directory structure in Step 2 of CONVERSION_NOTES.md and confirmed 11 subjects and 152 sessions, matching the paper's description. Using `h5py` instead of `pynwb` was a deliberate choice for direct HDF5 access.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from the NWB file paths by parsing the `sub-m<N>` pattern. The subject ID is also read from within the NWB metadata (`general/subject/subject_id`). All unique subjects are sorted by numeric mouse ID.

ii.
```python
def natural_file_key(path: Path) -> tuple[int, int]:
    match = re.search(r"sub-m(\d+)_ses-(\d+)", path.name)
    return int(match.group(1)), int(match.group(2))
...
subjects = sorted(
    {f"m{natural_file_key(path)[0]}" for path in files},
    key=lambda value: int(value[1:]),
)
```

iii. The AI verified 11 subjects matching the paper's count.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Session IDs are parsed from file names (`ses-<N>`). All sessions per subject are included.

ii.
```python
files = sorted(DATA_ROOT.glob("sub-*/*.nwb"), key=natural_file_key)
...
session_id = decode_text(nwb["general/session_id"])
```

iii. Consistent with one NWB file per imaging day as described in the paper.

## 1-d. How are the data split into trials?

i. Trials are delimited by frames where `trial_start > 0` (start) and `teleport > 0` (end). The trial spans `[start, stop)`, excluding the teleport frame itself. Validation checks ensure starts and stops are matched, ordered, and non-overlapping.

ii.
```python
def trial_bounds(group: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    starts = np.flatnonzero(group["trial_start/data"][:] > 0)
    stops = np.flatnonzero(group["teleport/data"][:] > 0)
    if starts.size != stops.size or starts.size < 2:
        raise ValueError(...)
    if np.any(starts >= stops) or np.any(starts[1:] <= stops[:-1]):
        raise ValueError(...)
    return starts, stops
```

iii. The AI found that `trial_start` and `teleport` provide clean, matched trial boundaries. The teleport frame is excluded because the paper notes it contains position-ambiguous interpolation.

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded when more than 30% of frames have cumulative lick count >2 (the paper's lick sensor fault criterion). This removes exactly 81 trials from 12,216 source trials, leaving 12,135 retained trials. No minimum-timepoint filter is applied.

ii.
```python
LICK_FAULT_FRACTION = 0.30
...
bad_trials = np.array(
    [np.mean(lick[a:b] > 2) > LICK_FAULT_FRACTION for a, b in zip(starts, stops)],
    dtype=bool,
)
...
for raw_trial, (start, stop) in enumerate(zip(starts, stops)):
    if bad_trials[raw_trial]:
        continue
```

iii. The AI found the paper's lick QC description: "n = 81 out of 12,376 trials" with >30% threshold. The code uses fraction >0.30 matching the paper (not the code's stale 0.35 threshold). The CONVERSION_NOTES document this decision extensively in Step 4.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the NWB's pre-stored `Deconvolved` arrays at `processing/ophys/Deconvolved/plane*/data`. These are described by the AI as "author-provided OASIS-deconvolved calcium events." Raw Fluorescence and Neuropil are only read transiently for the interneuron QC computation.

ii.
```python
def load_deconvolved(nwb: h5py.File, keep_global: np.ndarray, n_behavior: int) -> np.ndarray:
    deconvolved = nwb[f"{OPHYS_ROOT}/Deconvolved"]
    for plane_name in sorted(deconvolved):
        group = deconvolved[plane_name]
        global_ids = group["rois"][:].astype(np.int64)
        local = np.flatnonzero(keep_global[global_ids])
        if local.size:
            chunks.append(group["data"][:n_behavior, local].astype(np.float32, copy=False))
            ids_all.append(global_ids[local])
    ...
    return activity[:, order]
```

iii. The AI's CONVERSION_NOTES (Step 5) state: "Use author-provided OASIS-deconvolved calcium events, the stream used by their decoder. Recomputing deconvolution would add numerical differences without benefit." The AI assumed the NWB Deconvolved stream is the same as the paper's processed events.

## 2-b. How is the `neural` data processed?

i. No processing is applied to the neural data beyond loading. The AI uses the pre-stored Deconvolved arrays directly, filtered to `iscell` and non-interneuron cells, concatenated across planes in global ROI order, and cropped to behavior length.

ii.
```python
activity = load_deconvolved(nwb, keep, n_behavior)
...
neural = activity[start:stop].T.copy()
```

iii. The AI decided against recomputing dF/F and deconvolution, stating it would "add numerical differences without benefit." Fluorescence and Neuropil are only used for the interneuron QC mask computation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) Suite2p manual curation via `iscell[:,0] == 1`, and (2) putative interneuron exclusion where dF/F-speed Pearson correlation > 0.5. The dF/F for interneuron QC is computed transiently using the paper's exact preprocessing: 0.7 neuropil subtraction, Gaussian sigma-15 smoothing, 300-frame min/max filters for baseline, dF/F as (F-baseline)/|baseline|, Gaussian sigma-2 smoothing.

ii.
```python
iscell = segmentation["iscell"][:, 0].astype(bool)
keep, speed_correlations = compute_interneuron_mask(nwb, iscell, speed, starts, stops)
...
activity = load_deconvolved(nwb, keep, n_behavior)
```

The `compute_interneuron_mask` function computes dF/F per block of 128 cells:
```python
corrected = f - 0.7 * f_neu + 0.7 * f_neu.mean(axis=1, keepdims=True)
baseline = ndimage.gaussian_filter1d(corrected, 15, axis=1)
baseline = ndimage.minimum_filter1d(baseline, 300, axis=1)
baseline = ndimage.maximum_filter1d(baseline, 300, axis=1)
dff = (corrected - baseline) / np.abs(baseline)
dff = ndimage.gaussian_filter1d(dff, 2, axis=1)
```

iii. The AI documented this in CONVERSION_NOTES Step 4: the paper states "Additional putative interneurons were detected for exclusion" with r > 0.5. The AI's QC implementation matches the paper's dF/F pipeline parameters.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start by slicing `activity[start:stop]` where `start` is the frame with `trial_start > 0`. Since neural and behavior data share the same imaging-frame time grid, no temporal shifting is needed.

ii.
```python
neural = activity[start:stop].T.copy()
```

iii. The AI verified that neural and behavior timestamps are on the same grid and that slicing by the same indices ensures alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging frame rate (~15.5078125 Hz, ~64.48 ms per frame) is preserved. No temporal rebinning is applied. The time bin size is computed as `DT_SECONDS * 1000.0` milliseconds.

ii.
```python
DT_SECONDS = 1.0 / 15.5078125
...
"time_bin_size": DT_SECONDS * 1000.0,
```

iii. The AI confirmed all sessions have identical frame intervals matching the paper's ~15.5 Hz per plane.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the dense behavior timestamps at `processing/behavior/BehavioralTimeSeries/position/timestamps`.

ii.
```python
timestamps = behavior["position/timestamps"][:]
...
relative_time = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. All dense behavioral time series share identical timestamps, so any one could be used. The AI chose position timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at the trial start frame is subtracted from all timestamps within the trial, giving relative time starting at 0.

ii.
```python
relative_time = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. Straightforward computation yielding time in seconds from trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same imaging-frame time grid. Both are sliced with the same `[start:stop)` indices, so they are inherently aligned.

ii.
```python
relative_time = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
...
neural = activity[start:stop].T.copy()
```

iii. The AI validated that timestamps are uniform and that all dense behavior arrays have the same length as neural data (within one terminal frame for two-plane sessions).

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` dense behavior time series at `processing/behavior/BehavioralTimeSeries/environment/data`.

ii.
```python
environment = behavior["environment/data"][:]
...
env_values = np.unique(environment[start:stop])
...
np.full(T, env_values[0], dtype=np.float32),
```

iii. The AI verified that environment is constant within each trial and takes values 0 (ENV1) or 1 (ENV2).

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The unique environment value within the trial is extracted and broadcast across all timepoints. A validation check ensures exactly one unique value in {0, 1} per trial.

ii.
```python
env_values = np.unique(environment[start:stop])
if env_values.size != 1 or env_values[0] not in (0, 1):
    raise ValueError(...)
np.full(T, env_values[0], dtype=np.float32),
```

iii. No transformation needed; the raw values directly encode ENV1=0, ENV2=1.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number` dense behavior time series at `processing/behavior/BehavioralTimeSeries/trial number/data`. This is the experimental trial number assigned by the VR system.

ii.
```python
trial_number = behavior["trial number/data"][:]
...
trial_values = np.unique(trial_number[start:stop])
...
np.full(T, trial_values[0], dtype=np.float32),
```

iii. The AI validates that trial_number is unique within each trial. This uses the native experimental trial number rather than a sequential loop counter.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The unique trial number within the trial is extracted and broadcast across all timepoints. A validation check ensures exactly one unique value per trial.

ii.
```python
trial_values = np.unique(trial_number[start:stop])
if trial_values.size != 1:
    raise ValueError(...)
np.full(T, trial_values[0], dtype=np.float32),
```

iii. The native zero-based experimental trial number is used directly without transformation.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the sparse `Reward` timestamps and the `reward_zone` dense behavior signal. Both are used together to determine reward outcome per trial.

ii.
```python
reward_timestamps = behavior["Reward/timestamps"][:]
reward_zone_signal = behavior["reward_zone/data"][:]
...
outcomes = reward_outcomes(timestamps, reward_timestamps, reward_zone_signal, starts, stops)
previous_outcomes = np.r_[0, outcomes[:-1]].astype(np.int8)
```

iii. The AI used both delivery AND reward-zone evidence to match the paper's `get_trial_types` function.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each raw trial, outcome is 1 if a sparse reward delivery falls within the trial's time range AND the reward_zone signal is positive during the trial; otherwise 0. The previous trial's outcome is then used: `previous_outcomes = np.r_[0, outcomes[:-1]]`. The first trial defaults to 0. The outcome of the preceding raw trial is used, even if that trial was excluded by lick QC.

ii.
```python
def reward_outcomes(timestamps, reward_timestamps, reward_zone_signal, starts, stops):
    out = np.zeros(starts.size, dtype=np.int8)
    for i, (start, stop) in enumerate(zip(starts, stops)):
        delivered = np.any(
            (reward_timestamps >= timestamps[start])
            & (reward_timestamps < timestamps[stop])
        )
        zone_evidence = np.any(reward_zone_signal[start:stop] > 0)
        out[i] = int(delivered and zone_evidence)
    return out
...
previous_outcomes = np.r_[0, outcomes[:-1]].astype(np.int8)
```

iii. Matches the paper's `get_trial_types` which requires both delivery and zone evidence. The first trial has no predecessor, so defaults to 0 (omitted).

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone label parsed from the NWB session identifier (scene protocol name, e.g. `Env1_LocationB_to_C`). Zone bounds are defined as A=(80,130), B=(200,250), C=(320,370) cm. For switch sessions, the zone changes after trial 30.

ii.
```python
ZONE_BOUNDS = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}

def scene_zone_labels(scene: str, n_trials: int) -> np.ndarray:
    labels = re.findall(r"(?:Location)?([ABC])", scene)
    if len(labels) == 1:
        return np.full(n_trials, labels[0], dtype="<U1")
    if len(labels) == 2:
        out = np.full(n_trials, labels[1], dtype="<U1")
        out[:30] = labels[0]
        return out
...
scene = decode_text(nwb["identifier"]).rsplit("/", 1)[-1]
labels = scene_zone_labels(scene, starts.size)
```

iii. This approach directly mirrors the paper's `get_reward_zones` function which parses zone labels from the scene/protocol name and applies the switch at trial 30.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, signed distance is computed: negative if before the zone (position - zone_start), zero if inside the zone [zone_start, zone_stop], positive if past the zone (position - zone_stop). Uses `np.where` for vectorized computation.

ii.
```python
label = labels[raw_trial]
zone_start, zone_stop = ZONE_BOUNDS[label]
signed_distance = np.where(
    trial_position < zone_start,
    trial_position - zone_start,
    np.where(trial_position > zone_stop, trial_position - zone_stop, 0.0),
)
```

iii. Matches the paper's concept of reward-relative distance with linear signed distance to the zone interval.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit conditional logic: 0 (<-50), 1 ([-50,-10)), 2 ([-10,0)), 3 (==0), 4 ((0,10]), 5 ((10,50]), 6 (>50).

ii.
```python
def discretize_distance(distance: np.ndarray) -> np.ndarray:
    out = np.empty(distance.shape, dtype=np.int8)
    out[distance < -50] = 0
    out[(distance >= -50) & (distance < -10)] = 1
    out[(distance >= -10) & (distance < 0)] = 2
    out[distance == 0] = 3
    out[(distance > 0) & (distance <= 10)] = 4
    out[(distance > 10) & (distance <= 50)] = 5
    out[distance > 50] = 6
    return out
```

iii. The AI verified edge cases with `verify_discretization_edges()`. The bin boundaries match the instructions. Note: boundary values like exactly 10 fall in class 4 (<=10), and exactly 50 fall in class 5 (<=50).

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same `[start:stop)` indices are used for both neural and behavioral data, ensuring alignment on the shared imaging-frame grid.

ii.
```python
trial_position = position[start:stop]
...
neural = activity[start:stop].T.copy()
```

iii. No additional alignment needed since all data share the same time grid.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` dense behavior time series.

ii.
```python
position = behavior["position/data"][:]
...
trial_position = position[start:stop]
```

iii. The `position` variable records the animal's position in the VR corridor in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond slicing to the trial interval. The raw position values in cm are used directly.

ii.
```python
trial_position = position[start:stop]
discretize_position(trial_position)
```

iii. Position values within trials range from ~0 to ~450 cm on the 450 cm track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins using explicit conditional logic: 0 (<90), 1 ([90,180)), 2 ([180,270)), 3 ([270,360]), 4 (>360).

ii.
```python
def discretize_position(position: np.ndarray) -> np.ndarray:
    out = np.empty(position.shape, dtype=np.int8)
    out[position < 90] = 0
    out[(position >= 90) & (position < 180)] = 1
    out[(position >= 180) & (position < 270)] = 2
    out[(position >= 270) & (position <= 360)] = 3
    out[position > 360] = 4
    return out
```

iii. Five 90-cm bins spanning the 450-cm track, matching the instructions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `[start:stop)` indices for both neural and position data.

ii.
```python
trial_position = position[start:stop]
neural = activity[start:stop].T.copy()
```

iii. Inherently aligned on the shared imaging-frame grid.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` dense behavior time series.

ii.
```python
lick = behavior["lick/data"][:]
...
trial_lick = lick[start:stop]
```

iii. The `lick` variable records cumulative lick counts per imaging frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick count is mapped to 1, otherwise 0.

ii.
```python
(trial_lick > 0).astype(np.int8),
```

iii. The instructions specify binary output (no/yes). The raw lick values can be >1, so thresholding at >0 converts to binary.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[start:stop)` indices for both neural and lick data.

ii.
```python
trial_lick = lick[start:stop]
neural = activity[start:stop].T.copy()
```

iii. Inherently aligned on the shared imaging-frame grid.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB session identifier string (scene/protocol name) which encodes the reward zone letters (A, B, C) and switch information.

ii.
```python
scene = decode_text(nwb["identifier"]).rsplit("/", 1)[-1]
labels = scene_zone_labels(scene, starts.size)
```

iii. The AI followed the paper's `get_reward_zones` approach: parsing zone labels from the scene name with switch at trial 30.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene string is parsed with regex to extract zone letters. For fixed sessions (one letter), all trials get that zone. For switch sessions (two letters), the first 30 trials get the first zone and remaining trials get the second. Zone letters are mapped to classes: A=0, B=1, C=2.

ii.
```python
def scene_zone_labels(scene: str, n_trials: int) -> np.ndarray:
    labels = re.findall(r"(?:Location)?([ABC])", scene)
    if len(labels) == 1:
        return np.full(n_trials, labels[0], dtype="<U1")
    if len(labels) == 2:
        out = np.full(n_trials, labels[1], dtype="<U1")
        out[:30] = labels[0]
        return out
...
ZONE_TO_CLASS = {"A": 0, "B": 1, "C": 2}
np.full(T, ZONE_TO_CLASS[label], dtype=np.int8),
```

iii. Matches the paper's description: "Each switch occurred after 30 trials."

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the sparse `Reward` timestamps and the `reward_zone` dense behavior signal.

ii.
```python
reward_timestamps = behavior["Reward/timestamps"][:]
reward_zone_signal = behavior["reward_zone/data"][:]
outcomes = reward_outcomes(timestamps, reward_timestamps, reward_zone_signal, starts, stops)
```

iii. Both variables are used together to match the paper's `get_trial_types` semantics.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is marked as rewarded (1) if a sparse reward delivery timestamp falls within the trial's time range AND the reward_zone signal is positive during the trial. Otherwise 0. The value is broadcast across all timepoints.

ii.
```python
def reward_outcomes(timestamps, reward_timestamps, reward_zone_signal, starts, stops):
    out = np.zeros(starts.size, dtype=np.int8)
    for i, (start, stop) in enumerate(zip(starts, stops)):
        delivered = np.any(
            (reward_timestamps >= timestamps[start])
            & (reward_timestamps < timestamps[stop])
        )
        zone_evidence = np.any(reward_zone_signal[start:stop] > 0)
        out[i] = int(delivered and zone_evidence)
    return out
...
np.full(T, outcomes[raw_trial], dtype=np.int8),
```

iii. The AI's CONVERSION_NOTES document ~84.64% reward rate, matching the paper's ~85%.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- **Lick sensor faults**: Trials with >30% frames having lick count >2 are excluded entirely (81 trials).
- **Two-plane terminal frame mismatch**: Ten two-plane sessions have one extra neural row beyond behavior; all trial spans lie within the shared behavior length, so it's ignored.
- **Dense behavior length validation**: All dense behavior arrays are checked to have identical lengths; a mismatch raises an error.
- **Timestamp uniformity**: Validated that all timestamps match the expected frame interval.
- **Scanning validation**: Asserts all frames within trials have scanning==1.
- **Environment/trial number validation**: Asserts exactly one unique value per trial.
- **Non-finite values**: Checks that no NaN/Inf values exist in converted neural or input data.

ii.
```python
dense_lengths = {
    name: behavior[f"{name}/data"].shape[0]
    for name in ["position", "speed", "lick", ...]
}
if set(dense_lengths.values()) != {n_behavior}:
    raise ValueError(...)
if not np.allclose(np.diff(timestamps), DT_SECONDS, rtol=0, atol=1e-10):
    raise ValueError(...)
...
bad_trials = np.array(
    [np.mean(lick[a:b] > 2) > LICK_FAULT_FRACTION for a, b in zip(starts, stops)],
    dtype=bool,
)
```

iii. The AI documented these checks extensively in CONVERSION_NOTES Steps 4 and 10. The lick QC threshold of 0.30 follows the paper rather than the code's stale 0.35.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Interneuron QC computation**: Reading raw Fluorescence/Neuropil and computing per-cell dF/F-speed correlations for each session.
2. **Loading deconvolved neural data**: Reading large HDF5 arrays from disk.
3. **Pickle serialization**: Writing the ~8.86 GiB output file.

ii. N/A (timing information from CONVERSION_NOTES: 383.45s conversion + 7.79s serialization for 152 sessions)

iii. The AI reported total conversion time of ~391s (~6.5 minutes), well under the 15-minute threshold.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The `reward_outcomes` function iterates over trials in a Python loop, checking delivery timestamps against each trial's time range. This could be vectorized with `np.searchsorted`. Similarly, the inner `compute_interneuron_mask` loop over QC segments could potentially be batched, though it already processes cells in blocks of 128.

ii.
```python
def reward_outcomes(timestamps, reward_timestamps, reward_zone_signal, starts, stops):
    out = np.zeros(starts.size, dtype=np.int8)
    for i, (start, stop) in enumerate(zip(starts, stops)):
        delivered = np.any(...)
        zone_evidence = np.any(...)
        out[i] = int(delivered and zone_evidence)
    return out
```

iii. The per-trial loops are the natural structure given variable-length trials. The AI already optimized the interneuron QC with blockwise processing and sufficient statistics.

## 13-c. What processing does the code repeat multiple times?

i. The code reads each NWB file only once for conversion. However, the interneuron QC reads raw Fluorescence and Neuropil for computing dF/F correlations, and then separately loads the Deconvolved stream for the actual neural data. These access different HDF5 datasets but open the same file handle.

ii. N/A

iii. The AI designed the code to process each session in a single pass, avoiding the survey-then-convert pattern that would require loading files twice.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The interneuron QC computes full dF/F (neuropil subtraction, baseline, smoothing) for all curated cells only to compute speed correlations. The actual dF/F values are discarded; only the correlation mask is retained.

ii.
```python
def compute_interneuron_mask(...):
    ...
    corrected = f - 0.7 * f_neu + 0.7 * f_neu.mean(axis=1, keepdims=True)
    baseline = ndimage.gaussian_filter1d(corrected, 15, axis=1)
    ...
    dff = (corrected - baseline) / np.abs(baseline)
    ...
    # Only the correlation is kept
    corr = numerator / denominator
    keep[ids[corr > INTERNEURON_R_THRESHOLD]] = False
```

iii. This is necessary for the QC step but the intermediate dF/F data is discarded. The AI uses sufficient statistics (running sums) rather than storing full dF/F arrays, which is memory-efficient.
