# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All NWB files are discovered by globbing `DATA_ROOT/sub-*/*.nwb` and sorted by a natural numeric key (mouse number, session number). Each file is loaded with `pynwb.NWBHDF5IO`. Behavior time series are accessed via `nwb.processing["behavior"]["BehavioralTimeSeries"].time_series` and neural data via `nwb.processing["ophys"]`.

ii.
```python
DATA_ROOT = Path("/app/data")

def discover_files(sample: bool) -> list[Path]:
    files = sorted(DATA_ROOT.glob("sub-*/*.nwb"), key=natural_session_key)
    ...

with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
    nwb = io.read()
    behavior = nwb.processing["behavior"]["BehavioralTimeSeries"].time_series
```

iii. The code discovers all NWB files under the data root directory and processes them in natural order. The CONVERSION_NOTES document that there are 152 NWB files across 11 mice, matching the paper.

## 1-b. How are the data split into subjects?

i. Subject identity is read from the NWB metadata field `nwb.subject.subject_id` for each session. Unique subjects are collected and sorted numerically.

ii.
```python
subject_id = str(nwb.subject.subject_id)
...
subjects = sorted(set(subject_ids), key=lambda value: int(re.search(r"\d+", value).group()))
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
subject_idx = np.asarray([subject_lookup[subject] for subject in subject_ids], dtype=np.int64)
```

iii. The subject ID comes directly from the NWB metadata, which is authoritative for each file.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session day is parsed from the filename (e.g., `sub-m3_ses-08_behavior+ophys.nwb` -> session day 8).

ii.
```python
def natural_session_key(path: Path) -> tuple[int, int]:
    match = re.search(r"sub-m(\d+)_ses-(\d+)", path.name)
    return int(match.group(1)), int(match.group(2))
```

iii. Filenames follow a consistent naming convention encoding mouse and session day.

## 1-d. How are the data split into trials?

i. Trial starts are identified where the `trial_start` behavior signal is positive. Trial ends (stops) are identified where the `teleport` signal is positive. Trials span `[start, stop)`.

ii.
```python
starts = np.flatnonzero(np.asarray(behavior["trial_start"].data[:]) > 0)
stops = np.flatnonzero(np.asarray(behavior["teleport"].data[:]) > 0)
if len(starts) != len(stops) or np.any(stops <= starts):
    raise ValueError(f"Invalid trial bounds in {path.name}")
```

iii. The agent validates that starts and stops match in count and that each stop follows its corresponding start. This correctly identifies the track traversal epoch for each trial.

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded when more than 30% of their samples have cumulative lick count >2, matching the paper's corrupt lick sensor quality control criterion. In total, 81 trials are removed across the dataset.

ii.
```python
LICK_ERROR_FRACTION = 0.30
...
corrupt_lick = np.asarray([
    np.mean(lick[start:stop] > 2) > LICK_ERROR_FRACTION
    for start, stop in zip(starts, stops)
])
...
if corrupt_lick[raw_idx]:
    continue
```

iii. The paper describes removing trials where lick sensor error affects >30% of samples (cumulative count >2). The CONVERSION_NOTES document that this matches the paper's reported 81 corrupt trials (0.65% of 12,376).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw suite2p traces: `Fluorescence` (F) and `Neuropil` (Fneu) series from `processing/ophys`. The NWB `Deconvolved` field is NOT used, as the CONVERSION_NOTES document it is suite2p's own deconvolution rather than the paper's trial-wise processing.

ii.
```python
fluorescence_series = ophys["Fluorescence"].roi_response_series
neuropil_series = ophys["Neuropil"].roi_response_series
...
fluorescence = np.asarray(f_series.data[:len(speed), local_keep], dtype=np.float32)
neuropil = np.asarray(n_series.data[:len(speed), local_keep], dtype=np.float32)
```

iii. The CONVERSION_NOTES extensively document the finding that NWB `Deconvolved` differs substantially from the paper's processing (median per-neuron r=0.483), making it unsuitable as the neural signal.

## 2-b. How is the `neural` data processed?

i. The agent independently reimplements the paper's trial-wise dF/F and OASIS event extraction. Per trial: subtract `0.7 * Fneu`, add back `0.7 * mean(Fneu)`, Gaussian smooth with sigma 15 for baseline, 300-sample minimum filter then 300-sample maximum filter (maximin baseline), compute `(corrected - baseline) / |baseline|`, Gaussian smooth with sigma 2, then OASIS deconvolution with `tau=0.7` at `EFFECTIVE_FS_HZ=15.5078125`. The processing is always restricted to track samples `[start:stop)` without extending to inter-trial intervals.

ii.
```python
def trial_dff_and_events(fluorescence, neuropil, starts, stops, ...):
    for trial_idx, (start, stop) in enumerate(zip(starts, stops)):
        f_trial = fluorescence[start:stop]
        n_trial = neuropil[start:stop]
        corrected = f_trial - NEUROPIL_COEF * n_trial
        corrected += NEUROPIL_COEF * np.mean(n_trial, axis=0, keepdims=True)
        baseline = gaussian_filter1d(corrected, 15, axis=0)
        baseline = minimum_filter1d(baseline, 300, axis=0)
        baseline = maximum_filter1d(baseline, 300, axis=0)
        denom = np.abs(baseline)
        trial_dff = gaussian_filter1d((corrected - baseline) / denom, 2, axis=0)
        dff[start:stop] = trial_dff
    ...
    events[start:stop] = dcnv.oasis(
        dff[start:stop].T, 2000, OASIS_TAU_S, EFFECTIVE_FS_HZ
    ).T
```

iii. The CONVERSION_NOTES document that this matches the paper's Methods description. However, the agent does NOT implement the `keep_teleports` metadata from the paper's `teleport_metadata.py`, which specifies sessions where the laser was not blanked between trials and the baseline window extends into the inter-trial interval. Instead, the agent always restricts processing to track samples only.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) Manual curation via `iscell[:,0] > 0` from the ROI segmentation table. (2) Putative interneuron exclusion: cells with dF/F-speed Pearson r > 0.5. Both match the paper's described curation pipeline.

ii.
```python
iscell = np.asarray(segmentation["iscell"].data[:])[:, 0] > 0
...
local_keep = np.flatnonzero(iscell[roi_ids])
...
correlations = speed_correlations(dff, speed, starts, stops)
keep = np.isfinite(correlations) & (correlations <= 0.5)
interneuron_count += int(np.sum(~keep))
plane_events.append(events[:, keep])
```

iii. The paper describes manual Suite2p curation and exclusion of cells with dF/F-speed r > 0.5 (0.42 +/- 0.85%). The speed correlation is computed using a vectorized implementation over track samples.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to the trial start. Since neural and behavioral data are already synchronized at the same sampling rate, slicing by the same trial boundary indices aligns them. No additional temporal shifting is needed.

ii.
```python
neural = np.concatenate([events[start:stop].T for events in plane_events], axis=0)
```

iii. The instructions specify alignment to trial start. The CONVERSION_NOTES document that behavior/imaging timestamps are synchronized, so slicing is sufficient.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin is the native per-plane imaging rate: ~64.48 ms (1000/15.5078125 Hz). No rebinning is applied. Both single-plane and dual-plane sessions are already sampled at this effective rate per plane.

ii.
```python
EFFECTIVE_FS_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / EFFECTIVE_FS_HZ
...
"time_bin_size": float(TIME_BIN_MS),
```

iii. The CONVERSION_NOTES document that dual-plane sessions (m17/m18) have 31 Hz aggregate rate but 15.5 Hz per plane. The hardcoded effective frequency is consistent across all sessions.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position` behavior time series' timestamps array.

ii.
```python
timestamps = np.asarray(behavior["position"].timestamps[:], dtype=np.float64)
...
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The behavior timestamps are synchronized with imaging and provide the most accurate time information.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Subtract the timestamp at the trial start from all timestamps within the trial.

ii.
```python
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. Straightforward offset subtraction to get time relative to trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data are already synchronized (one row per imaging frame), so both use the same `[start:stop)` indexing within each trial.

ii. Same indexing: `timestamps[start:stop]` and `events[start:stop]`.

iii. The CONVERSION_NOTES document that timestamps and neural data have the same number of rows per session.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
environment = np.asarray(behavior["environment"].data[:], dtype=np.float32)
...
env_values = np.unique(environment[start:stop])
env_values = env_values[env_values >= 0]
```

iii. Environment values are 0 (ENV1) or 1 (ENV2), with -1 for invalid periods. The agent validates exactly one valid environment per trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The per-trial environment value is extracted and validated to be a single value (0 or 1). It is repeated as a constant across all timepoints in the trial.

ii.
```python
if len(env_values) != 1 or env_values[0] not in (0, 1):
    raise ValueError(f"Trial {raw_idx} has invalid environment values {env_values}")
...
np.full(stop - start, env_values[0], dtype=np.float32),
```

iii. Environment is constant within a trial, so the unique value is broadcast.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the NWB `trial number` behavior time series. The value at the first sample of each trial is used.

ii.
```python
trial_number = np.asarray(behavior["trial number"].data[:], dtype=np.float32)
...
source_trial_number = trial_number[start]
...
np.full(stop - start, source_trial_number, dtype=np.float32),
```

iii. The agent validates that the NWB trial number matches the raw trial index (`raw_idx`), raising an error on mismatch.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial number from the NWB data is used directly as a constant per trial. It is validated to match the expected sequential index.

ii.
```python
if not np.isclose(source_trial_number, raw_idx):
    raise ValueError(
        f"Trial-number mismatch in {path.name}: row {raw_idx}, value {source_trial_number}"
    )
```

iii. No transformation beyond validation and broadcasting across timepoints.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the sparse `Reward` timestamps in the behavior data, combined with the trial start/stop boundaries.

ii.
```python
def reward_outcomes(reward_times, timestamps, starts, stops):
    reward_times = np.sort(np.asarray(reward_times, dtype=np.float64))
    left = np.searchsorted(reward_times, timestamps[starts], side="left")
    right = np.searchsorted(reward_times, timestamps[stops], side="left")
    return (right > left).astype(np.int8)

outcomes = reward_outcomes(
    np.asarray(behavior["Reward"].timestamps[:]), timestamps, starts, stops
)
```

iii. Reward events have their own timestamps separate from the behavior sampling rate. The function checks whether any reward delivery falls within each trial's time window.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the previous raw trial's outcome is used (0 = omitted, 1 = rewarded). For the first trial, the value is 0. The reference is to the previous RAW trial, not the previous retained trial (important when corrupt lick trials are excluded).

ii.
```python
previous_outcome = outcomes[raw_idx - 1] if raw_idx > 0 else 0
...
np.full(stop - start, previous_outcome, dtype=np.float32),
```

iii. The instructions specify binary previous trial outcome. Using the raw trial index ensures chronological correctness even when some trials are excluded.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone schedule parsed from `NWBFile.identifier`. The identifier encodes the scene name which specifies which zone (A, B, or C) is active and, for switch sessions, the switch point.

ii.
```python
def parse_zone_schedule(scene: str, n_trials: int) -> np.ndarray:
    if "_to_" in scene:
        left, right = scene.split("_to_", 1)
        left_match = re.search(r"([ABC])$", left)
        right_match = re.search(r"([ABC])$", right)
        labels = [left_match.group(1)] * min(30, n_trials)
        labels.extend([right_match.group(1)] * max(0, n_trials - 30))
        return np.asarray(labels)
    match = re.search(r"Location([ABC])$", scene)
    return np.full(n_trials, match.group(1))

scene = nwb.identifier.rsplit("/", 1)[-1]
zone_labels_raw = parse_zone_schedule(scene, len(starts))
```

iii. The NWB identifier encodes the full scene/reward schedule. Switch sessions change zone after trial 30, matching the paper's documented switch schedule.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance to the nearest point of the active reward zone: negative before the zone, zero inside, positive after. Zone bounds are A=[80,130], B=[200,250], C=[320,370] cm.

ii.
```python
ZONE_BOUNDS = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}
...
zone_start, zone_stop = ZONE_BOUNDS[zone_label]
signed_distance = np.where(
    trial_position < zone_start,
    trial_position - zone_start,
    np.where(trial_position > zone_stop, trial_position - zone_stop, 0.0),
)
```

iii. This follows the paper's reward zone definitions and the instructions' requirement for distance to any location in the reward zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins with explicit boundary conditions:
- 0: < -50 cm
- 1: [-50, -10)
- 2: [-10, 0)
- 3: exactly 0 (inside zone)
- 4: (0, 10]
- 5: (10, 50]
- 6: > 50 cm

ii.
```python
def distance_classes(distance: np.ndarray) -> np.ndarray:
    out = np.full(distance.shape, 6, dtype=np.int8)
    out[distance < -50.0] = 0
    out[(distance >= -50.0) & (distance < -10.0)] = 1
    out[(distance >= -10.0) & (distance < 0.0)] = 2
    out[distance == 0.0] = 3
    out[(distance > 0.0) & (distance <= 10.0)] = 4
    out[(distance > 10.0) & (distance <= 50.0)] = 5
    return out
```

iii. The boundary conditions are explicitly tested with a `check_discretization_boundaries()` function that verifies correct classification at all edge values.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Both position (used to compute distance) and neural data are indexed by the same `[start:stop)` trial boundaries, ensuring alignment.

ii. Same indexing: `trial_position = position[start:stop]` and `events[start:stop]`.

iii. Synchronized behavior/imaging timestamps ensure alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = np.asarray(behavior["position"].data[:], dtype=np.float32)
...
trial_position = position[start:stop]
```

iii. The position variable directly records the animal's position in the VR corridor in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice and discretizing into 5 equal bins of 90 cm each.

ii.
```python
def position_classes(position: np.ndarray) -> np.ndarray:
    out = np.full(position.shape, 4, dtype=np.int8)
    out[position < 90.0] = 0
    out[(position >= 90.0) & (position < 180.0)] = 1
    out[(position >= 180.0) & (position < 270.0)] = 2
    out[(position >= 270.0) & (position <= 360.0)] = 3
    return out
```

iii. Five 90-cm bins cover the 450-cm track. The position data is used directly from the VR.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins:
- 0: < 90 cm
- 1: [90, 180)
- 2: [180, 270)
- 3: [270, 360]
- 4: > 360 cm

ii. See 8-b code snippet.

iii. The bins match the instructions. Position = 360 is correctly assigned to bin 3 (matching "270 to 360 cm").

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `[start:stop)` indexing as neural data within each trial.

ii. Same indexing: `trial_position = position[start:stop]`.

iii. Synchronized behavior/imaging timestamps.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = np.asarray(behavior["lick"].data[:], dtype=np.float32)
...
trial_lick = lick[start:stop]
```

iii. The lick variable records lick events at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
(trial_lick > 0).astype(np.int8),
```

iii. The instructions specify binary output (no/yes). Raw lick values can be >1 (cumulative counts), so thresholding at >0 converts to binary.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[start:stop)` indexing as neural data within each trial.

ii. Same indexing: `trial_lick = lick[start:stop]`.

iii. Synchronized timestamps.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from `NWBFile.identifier` which encodes the scene/reward schedule, parsed by `parse_zone_schedule()`. See 7-a.

ii. See 7-a code snippet.

iii. The scene identifier deterministically specifies which zone is active for each trial.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The zone label (A/B/C) is mapped to a class index (0/1/2) and broadcast as a constant across all timepoints in the trial.

ii.
```python
ZONE_TO_CLASS = {"A": 0, "B": 1, "C": 2}
...
np.full(stop - start, ZONE_TO_CLASS[zone_label], dtype=np.int8),
```

iii. For switch sessions, the first 30 trials use the first zone and trials >=30 use the second zone, matching the paper's switch protocol.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the sparse `Reward` timestamps in the behavior data.

ii.
```python
outcomes = reward_outcomes(
    np.asarray(behavior["Reward"].timestamps[:]), timestamps, starts, stops
)
...
np.full(stop - start, outcomes[raw_idx], dtype=np.int8),
```

iii. The Reward time series records reward delivery events with separate timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check whether any reward delivery timestamp falls within the trial's time window `[start_timestamp, stop_timestamp)`. If yes, outcome = 1; otherwise 0. The value is constant across all timepoints in the trial.

ii.
```python
def reward_outcomes(reward_times, timestamps, starts, stops):
    reward_times = np.sort(np.asarray(reward_times, dtype=np.float64))
    left = np.searchsorted(reward_times, timestamps[starts], side="left")
    right = np.searchsorted(reward_times, timestamps[stops], side="left")
    return (right > left).astype(np.int8)
```

iii. Uses searchsorted for efficient mapping of sparse reward events to trial windows. The reward rate (84.6%) matches the paper's ~85%.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Ophys/behavior length mismatch**: Ten dual-plane sessions have exactly one extra terminal ophys frame after the last behavior timestamp. This frame is discarded; any other mismatch raises an error.
- **Corrupt lick trials**: Trials with >30% cumulative lick >2 are excluded entirely (81 trials).
- **Environment validation**: Trials with invalid or mixed environment values raise errors.
- **Trial number validation**: Trials where NWB trial number doesn't match expected index raise errors.
- **Finite value checks**: All converted neural and input arrays are verified finite.

ii.
```python
extra_tail = n_ophys_frames - len(speed)
if extra_tail > 1:
    raise ValueError(...)
discarded_tail_frames = max(discarded_tail_frames, extra_tail)
fluorescence = np.asarray(f_series.data[:len(speed), local_keep], dtype=np.float32)
```

iii. The CONVERSION_NOTES document the precise characterization of the +1 terminal ophys frame issue in ten sessions (m17 sessions 4, 6 and m18 sessions 1, 5, 7, 10-14).

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files via PyNWB (I/O bound, ~92 GB dataset)
2. dF/F computation and OASIS deconvolution per trial per plane
3. Speed correlation computation for neuron filtering
4. Writing the final pickle file (~9.5 GB)

ii. N/A (timing logged in conversion output: total ~301 s)

iii. Per-session times range from 0.4 s to 5 s, with larger sessions (more neurons) taking longer.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `trial_dff_and_events` iterates over each trial sequentially for baseline/dF/F computation. This is inherent to the trial-wise processing requirement (baselines are independent per trial) but could potentially be batched for equal-length trials. The trial-level loop in `process_session` for constructing input/output arrays is also sequential but involves variable-length operations.

ii. N/A

iii. The speed correlation computation IS vectorized (accumulates sufficient statistics over trials then computes correlation in one step), which is more efficient than per-cell loops.

## 13-c. What processing does the code repeat multiple times?

i. The code processes each NWB file exactly once in a single pass. No survey/pre-scan step is used. This avoids the duplication of reading NWB files twice.

ii. N/A

iii. The single-pass approach is more I/O-efficient than a two-pass approach that surveys then converts.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code captures processing traces and plots for `--show-processing` mode, which are diagnostic only. The `check_discretization_boundaries()` function runs every time but is instantaneous. Otherwise, there is minimal unnecessary processing.

ii. N/A

iii. The code is focused and does not compute signals that are discarded downstream.
