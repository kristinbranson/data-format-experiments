# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All NWB files are discovered by globbing `sub-*/sub-*_behavior+ophys.nwb` under the data root. Files are sorted by subject number then session number. Each NWB file is opened with `h5py` (not `pynwb`) and all behavioral time series, fluorescence/neuropil/deconvolved neural data, and segmentation metadata are read directly from HDF5 paths.

ii.
```python
paths = sort_session_paths(glob.glob(str(DATA_ROOT / "sub-*" / "sub-*_behavior+ophys.nwb")))
# ...
with h5py.File(path, "r") as f:
    behavior = f["processing/behavior/BehavioralTimeSeries"]
    frame_times = np.asarray(behavior["position/timestamps"][:], dtype=np.float64)
    position = np.asarray(behavior["position/data"][:], dtype=np.float32)
    # ...
```

iii. The agent explored the NWB file structure extensively, confirmed the directory layout, and decided to use `h5py` for direct HDF5 access rather than `pynwb`. The agent verified that all sessions are captured by the glob pattern.

## 1-b. How are the data split into subjects?

i. Subject IDs are extracted from the NWB file's `general/subject/subject_id` field. Unique subjects are collected from all converted sessions and sorted numerically.

ii.
```python
subject = f["general/subject/subject_id"][()].decode()
# ...
subjects = sorted({session["subject"] for session in session_results}, key=lambda s: int(s[1:]))
```

iii. The agent read the NWB metadata to obtain subject IDs directly from the file rather than parsing directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session ID is read from the NWB file's `general/session_id` field.

ii.
```python
session_id = f["general/session_id"][()].decode()
```

iii. The agent confirmed that each NWB file represents a unique session and used the embedded session ID.

## 1-d. How are the data split into trials?

i. Trials are segmented by pairing `trial_start` rising edges with the next `teleport` rising edge. The `pair_trial_segments` function finds each `trial_start > 0` and pairs it with the next `teleport > 0` that occurs after it.

ii.
```python
def pair_trial_segments(trial_start_signal, teleport_signal):
    starts = np.flatnonzero(trial_start_signal > 0)
    teleports = np.flatnonzero(teleport_signal > 0)
    segments = []
    teleport_ptr = 0
    for start in starts:
        while teleport_ptr < len(teleports) and teleports[teleport_ptr] <= start:
            teleport_ptr += 1
        if teleport_ptr >= len(teleports):
            break
        stop = teleports[teleport_ptr]
        teleport_ptr += 1
        if stop - start >= 1:
            segments.append((int(start), int(stop)))
    return segments
```

iii. The agent investigated the trial boundary signals and confirmed that pairing `trial_start` with the next `teleport` correctly delineates trials, matching the paper's definition.

## 1-e. How are trials filtered based on quality controls?

i. Two filters are applied: (1) Trials flagged as "lick error" trials are excluded — a trial is flagged if more than 35% of its frames have lick counts > 2.0. (2) Trials where temporal binning yields fewer than 1 bin (i.e., fewer than 8 frames) are excluded. Additionally, sessions with fewer than 2 usable trials are dropped entirely.

ii.
```python
LICK_ERROR_THRESHOLD = 0.35
# ...
lick_error = bool(np.mean(lick_counts[start:stop] > 2.0) > LICK_ERROR_THRESHOLD)
# ...
if info["lick_error"]:
    lick_error_trials += 1
    continue
n_bins = n_frames // BIN_FRAMES
if n_bins < 1:
    continue
# ...
if len(converted["neural"]) < 2:
    continue
```

iii. The agent introduced a lick-error filter that is not described in the reference paper or code. The agent reasoned that trials with excessive lick counts indicated sensor errors, but this is an additional quality control not present in the reference solution.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the NWB's `Deconvolved` field (suite2p deconvolution stored in the NWB file), NOT from the raw `Fluorescence` and `Neuropil` traces.

ii.
```python
deconvolved = load_roi_response_matrix(
    f,
    "processing/ophys/Deconvolved",
    curated_cell_idx,
    plane_idx_all,
).T
deconvolved = deconvolved[keep_cells]
```

iii. The agent explored the NWB structure and decided to use the pre-computed `Deconvolved` signal. However, the agent's trajectory shows awareness that the paper computes its own dF/F and deconvolution from raw fluorescence (step 28: "I'm checking ... whether to trust the released Deconvolved series as-is or reproduce any extra filtering"), but ultimately chose to use the stored Deconvolved data directly.

## 2-b. How is the `neural` data processed?

i. The deconvolved data is read directly from the NWB file with no further signal processing (no dF/F computation, no baseline correction, no custom deconvolution). The only processing is: (1) temporal rebinning into 8-frame bins by averaging, and (2) NaN replacement with 0.

ii.
```python
neural = bin_2d_mean(deconvolved[:, start:stop_trimmed], BIN_FRAMES)
neural = np.nan_to_num(neural, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)
```

iii. The agent decided to use the stored deconvolved signal rather than reproducing the paper's full dF/F + OASIS pipeline. The agent's reasoning was that this signal was already available in the NWB file.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) Only cells with `iscell[:, 0] > 0.5` (suite2p manual curation) are kept. (2) Putative interneurons are excluded using dF/F-speed correlation > 0.5. The interneuron filter is computed from raw Fluorescence/Neuropil (computing dF/F on the fly for this purpose), even though the final neural signal uses the stored Deconvolved data.

ii.
```python
curated_cell_idx = np.flatnonzero(iscell[:, 0] > 0.5)
# ...
is_interneuron, speed_corr = find_putative_interneurons(
    fluorescence, neuropil, speed, trial_segments,
)
keep_cells = ~is_interneuron
deconvolved = deconvolved[keep_cells]
```

iii. The agent confirmed that `iscell` is the standard suite2p curation array and implemented the interneuron exclusion based on the paper's described threshold of r > 0.5 for speed correlation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start by slicing the deconvolved array from the trial start index to the trial end (teleport) index. No additional temporal shifting is needed.

ii.
```python
neural = bin_2d_mean(deconvolved[:, start:stop_trimmed], BIN_FRAMES)
```

iii. The agent's trial segmentation already defines start as the trial onset, so alignment to trial start is implicit.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is rebinned from the native frame rate (~15.5 Hz) into 8-frame bins, yielding a time bin size of approximately 516 ms (`8 / 15.5078125 * 1000` ms). This is a significant downsampling from the native rate.

ii.
```python
FRAME_RATE_HZ = 15.5078125
BIN_FRAMES = 8
TIME_BIN_SIZE_S = BIN_FRAMES / FRAME_RATE_HZ
# ...
neural = bin_2d_mean(deconvolved[:, start:stop_trimmed], BIN_FRAMES)
pos_binned = bin_1d_mean(position[start:stop_trimmed], BIN_FRAMES)
```

iii. The agent decided to rebin because "The full 15.5 Hz frame stream is too large to serialize and train on directly" (step 63). The agent chose 8-frame bins as a "defensible temporal bin size that preserves the published alignment while keeping the dataset trainable."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position/timestamps` behavioral time series (frame times).

ii.
```python
frame_times = np.asarray(behavior["position/timestamps"][:], dtype=np.float64)
# ...
time_from_start = (
    frame_times[start:stop_trimmed:BIN_FRAMES] - frame_times[start]
).astype(np.float32)
```

iii. The agent used the position timestamps as the reference time base for all behavioral streams.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The frame times at every 8th frame (matching the bin boundaries) are taken, and the first frame time is subtracted to get time from trial start. This means each bin's time value is the time of the first frame in that bin, relative to the first frame of the trial.

ii.
```python
time_from_start = (
    frame_times[start:stop_trimmed:BIN_FRAMES] - frame_times[start]
).astype(np.float32)
```

iii. Straightforward computation of relative time from the trial start timestamp.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both neural and time-from-start use the same frame indices and the same 8-frame binning, so they are aligned by construction.

ii. Both use `start:stop_trimmed` with `BIN_FRAMES` binning.

iii. Same indexing scheme ensures alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The agent uses two sources: the `environment` behavior time series from the NWB file, and as a fallback/validation, the `scene` identifier parsed from the NWB file's `identifier` field.

ii.
```python
environment = np.asarray(behavior["environment/data"][:], dtype=np.float32)
# ...
env_slice = environment[start:stop]
env_slice = env_slice[np.isfinite(env_slice) & (env_slice >= 0)]
if env_slice.size:
    env_value = int(np.rint(np.nanmedian(env_slice)))
else:
    env_value = expected_envs[trial_idx]
```

iii. The agent uses the median of the environment signal within the trial, with a fallback to the expected environment from the scene parser. Environment mismatches are counted but not used for filtering.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The environment signal within the trial is filtered for finite, non-negative values, then the median is taken and rounded. If no valid values exist, the expected environment from scene parsing is used as fallback.

ii.
```python
env_slice = env_slice[np.isfinite(env_slice) & (env_slice >= 0)]
if env_slice.size:
    env_value = int(np.rint(np.nanmedian(env_slice)))
else:
    env_value = expected_envs[trial_idx]
```

iii. The agent adds robustness by filtering invalid values and using a fallback, but the core value is the same as simply reading the environment signal.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The trial number is derived from the `trial number` behavior time series in the NWB file.

ii.
```python
trial_number_signal = np.asarray(behavior["trial number/data"][:], dtype=np.float32)
# ...
trialnum_slice = trial_number_signal[start:stop]
trialnum_slice = trialnum_slice[np.isfinite(trialnum_slice) & (trialnum_slice >= 0)]
if trialnum_slice.size:
    trial_number = int(np.rint(np.nanmedian(trialnum_slice)))
else:
    trial_number = trial_idx
```

iii. The agent uses the stored trial number signal from the NWB file rather than a simple loop counter.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial number signal within each trial is filtered for finite non-negative values, then the median is rounded to get an integer trial number. Falls back to the trial index if no valid values exist.

ii.
```python
trialnum_slice = trialnum_slice[np.isfinite(trialnum_slice) & (trialnum_slice >= 0)]
if trialnum_slice.size:
    trial_number = int(np.rint(np.nanmedian(trialnum_slice)))
else:
    trial_number = trial_idx
```

iii. The agent chose to use the NWB's trial number signal, which may differ from the sequential trial index used in the reference.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward/timestamps` behavior time series — the timestamps at which rewards were delivered.

ii.
```python
reward_timestamps = np.asarray(behavior["Reward/timestamps"][:], dtype=np.float64)
# ...
reward_outcome = int(np.any(
    (reward_timestamps >= frame_times[start]) & (reward_timestamps < frame_times[stop])
))
reward_outcomes.append(reward_outcome)
# ...
prev_outcome = reward_outcomes[trial_idx - 1] if trial_idx > 0 else 0
```

iii. The agent checks whether any reward timestamp falls within the previous trial's time window to determine the previous trial's outcome.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the reward outcome is determined by checking if any reward timestamp falls within [frame_times[start], frame_times[stop]). For the first trial, previous outcome defaults to 0. For subsequent trials, it is the reward outcome of the preceding trial (before any filtering).

ii.
```python
prev_outcome = reward_outcomes[trial_idx - 1] if trial_idx > 0 else 0
```

iii. The agent uses reward timestamps rather than a frame-aligned reward signal. Note that the previous trial outcome uses the raw trial index (before lick-error filtering), so `trial_idx - 1` refers to the immediately preceding trial even if it was filtered out.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone label for the trial. The reward zone label is determined from the NWB `identifier` field (scene name), which encodes the environment and reward zone location(s), parsed by `parse_scene()` and `expected_trial_labels()`.

ii.
```python
scene = f["identifier"][()].decode().split("/")[-1]
zone_labels, expected_envs = expected_trial_labels(scene, len(trial_segments))
# ...
zone_bounds = REWARD_ZONES[info["zone_label"]]
distance_binned = reward_distance_cm(pos_binned.astype(np.float32), zone_bounds)
```

iii. The agent parsed the scene identifier to determine which reward zone (A, B, or C) applies to each trial. Switch days are handled by changing the zone label at trial 30 (the `SWITCH_TRIAL` constant).

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each time bin, the signed distance from the animal's binned position to the nearest edge of the reward zone is computed. Distance is 0 inside the zone, negative before, positive after. The position is first averaged over 8-frame bins before computing distance.

ii.
```python
def reward_distance_cm(position_cm, zone_bounds):
    start_cm, end_cm = zone_bounds
    distance = np.zeros_like(position_cm, dtype=np.float32)
    before = position_cm < start_cm
    after = position_cm > end_cm
    distance[before] = position_cm[before] - start_cm
    distance[after] = position_cm[after] - end_cm
    return distance
```

iii. The computation matches the standard signed-distance-to-interval formula used in the reference.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit conditional logic matching the instruction bin edges: < -50, -50 to -10, -10 to 0, exactly 0, >0 to 10, 10 to 50, > 50.

ii.
```python
def discretize_reward_distance(distance_cm):
    out = np.empty(distance_cm.shape, dtype=np.uint8)
    out[distance_cm < -50.0] = 0
    out[(distance_cm >= -50.0) & (distance_cm < -10.0)] = 1
    out[(distance_cm >= -10.0) & (distance_cm < 0.0)] = 2
    out[distance_cm == 0.0] = 3
    out[(distance_cm > 0.0) & (distance_cm <= 10.0)] = 4
    out[(distance_cm > 10.0) & (distance_cm <= 50.0)] = 5
    out[distance_cm > 50.0] = 6
    return out
```

iii. The bin edges match the instructions. The implementation uses explicit conditions rather than `np.digitize`.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position is binned using the same 8-frame mean binning as neural data, then distance is computed from the binned position. Alignment is ensured by using the same frame indices and bin size.

ii.
```python
pos_binned = bin_1d_mean(position[start:stop_trimmed], BIN_FRAMES)
distance_binned = reward_distance_cm(pos_binned.astype(np.float32), zone_bounds)
```

iii. Same binning scheme ensures temporal alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = np.asarray(behavior["position/data"][:], dtype=np.float32)
pos_binned = bin_1d_mean(position[start:stop_trimmed], BIN_FRAMES)
```

iii. The position variable directly records the animal's VR position.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is averaged over 8-frame bins and then clipped to [0, 450] cm before discretization.

ii.
```python
pos_binned = bin_1d_mean(position[start:stop_trimmed], BIN_FRAMES)
pos_binned = np.clip(pos_binned, 0.0, TRACK_LENGTH_CM)
```

iii. The clipping ensures all values fall within the track range.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins: < 90, 90 to 180, 180 to 270, 270 to 360, >= 360.

ii.
```python
def discretize_position(position_cm):
    out = np.empty(position_cm.shape, dtype=np.uint8)
    out[position_cm < 90.0] = 0
    out[(position_cm >= 90.0) & (position_cm < 180.0)] = 1
    out[(position_cm >= 180.0) & (position_cm < 270.0)] = 2
    out[(position_cm >= 270.0) & (position_cm < 360.0)] = 3
    out[position_cm >= 360.0] = 4
    return out
```

iii. Five equal 90 cm bins spanning the 450 cm track, matching the instructions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same 8-frame binning as neural data ensures alignment.

ii. `pos_binned = bin_1d_mean(position[start:stop_trimmed], BIN_FRAMES)`

iii. Same indexing scheme.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick_counts = np.asarray(behavior["lick/data"][:], dtype=np.float32)
```

iii. The `lick` variable records lick events at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick values > 1.0 are capped at 1.0, then max-pooled over 8-frame bins, and binarized (> 0 becomes 1).

ii.
```python
lick_trial = lick_counts[start:stop_trimmed].copy()
lick_trial[lick_trial > 1.0] = 1.0
lick_binned = bin_1d_max(lick_trial, BIN_FRAMES)
lick_binned = (lick_binned > 0).astype(np.uint8)
```

iii. Max-pooling preserves lick events during binning. Values > 1.0 are clipped first (possibly related to the lick error threshold logic).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same 8-frame binning (using max instead of mean, but same frame boundaries) as neural data.

ii. Same `start:stop_trimmed` and `BIN_FRAMES`.

iii. Same binning scheme ensures alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB `identifier` field, which encodes the scene name (e.g., `Env1_LocationC_to_A`). The `parse_scene()` function extracts the reward zone label(s) and `expected_trial_labels()` assigns the correct zone to each trial, accounting for switch trials.

ii.
```python
scene = f["identifier"][()].decode().split("/")[-1]
zone_labels, expected_envs = expected_trial_labels(scene, len(trial_segments))
# ...
output_trial[4] = np.full(n_bins, ord(info["zone_label"]) - ord("A"), dtype=np.uint8)
```

iii. The agent parsed the scene identifier to determine reward zone labels deterministically rather than using the `reward_zone` behavior time series or a Viterbi-based approach.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene name is parsed to extract zone labels before and after any switch. For switch sessions (e.g., `LocationC_to_A`), the zone changes at trial 30. The zone label (A/B/C) is converted to integer (0/1/2) using ASCII offset.

ii.
```python
def expected_trial_labels(scene, n_trials):
    info = parse_scene(scene)
    if info["switch_trial"] is None:
        zones = [info["zone_before"]] * n_trials
    else:
        pre_n = min(info["switch_trial"], n_trials)
        post_n = max(0, n_trials - pre_n)
        zones = [info["zone_before"]] * pre_n + [info["zone_after"]] * post_n
    return zones, envs
```

iii. The agent uses a deterministic approach based on the session's scene identifier, with the switch point hardcoded at trial 30 based on the experimental protocol.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward/timestamps` behavior time series.

ii.
```python
reward_timestamps = np.asarray(behavior["Reward/timestamps"][:], dtype=np.float64)
reward_outcome = int(np.any(
    (reward_timestamps >= frame_times[start]) & (reward_timestamps < frame_times[stop])
))
```

iii. Reward timestamps are checked against trial boundaries.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Binary: 1 if any reward timestamp falls within the trial's time window [start, stop), 0 otherwise. The value is constant for all time bins in the trial.

ii.
```python
reward_outcome = int(np.any(
    (reward_timestamps >= frame_times[start]) & (reward_timestamps < frame_times[stop])
))
# ...
np.full(n_bins, info["reward_outcome"], dtype=np.uint8)
```

iii. Standard per-trial binary reward outcome.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several defensive measures: (1) NaN values in neural data are replaced with 0. (2) Environment and trial number signals are filtered for finite non-negative values with fallbacks to expected values from scene parsing. (3) Position is clipped to [0, 450]. (4) Lick values > 1 are clipped. (5) Sessions with < 2 trials are skipped. (6) Trials with lick sensor errors (>35% of frames with lick > 2) are excluded. (7) Speed NaN/inf replaced with 0.

ii.
```python
neural = np.nan_to_num(neural, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)
pos_binned = np.clip(pos_binned, 0.0, TRACK_LENGTH_CM)
speed_binned = np.nan_to_num(speed_binned, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The agent added multiple defensive checks during data exploration. The lick error filter and position clipping are additional processing not in the reference solution.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading NWB files via h5py (I/O bound), (2) The interneuron detection step which computes dF/F from raw fluorescence per session, (3) The full conversion loop over all 152 sessions.

ii. N/A

iii. The agent noted conversion took several minutes for the full dataset.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The `find_putative_interneurons` function loops over trial segments to accumulate correlation statistics. The per-cell correlation loop could potentially be vectorized. The `pair_trial_segments` function uses a pointer-based loop that could use vectorized operations.

ii. N/A

iii. The agent's code is generally well-vectorized, using numpy operations where possible.

## 13-c. What processing does the code repeat multiple times?

i. The interneuron detection computes dF/F from fluorescence/neuropil just for the purpose of speed correlation, but this computation is not reused. The fluorescence and neuropil are loaded, processed for interneuron detection, then deleted — the final neural data comes from the separate Deconvolved field. This means the fluorescence processing is done once for filtering but never used for the actual neural signal.

ii.
```python
is_interneuron, speed_corr = find_putative_interneurons(fluorescence, neuropil, speed, trial_segments)
keep_cells = ~is_interneuron
del fluorescence
del neuropil
deconvolved = load_roi_response_matrix(f, "processing/ophys/Deconvolved", ...)
```

iii. The agent loads fluorescence just for interneuron detection, then loads deconvolved separately.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The lick error detection and filtering is an additional quality control not present in the reference solution. The position clipping to [0, 450] modifies raw data values. The computation of `session_metadata` with detailed per-session statistics is stored but not used by the decoder.

ii.
```python
lick_error = bool(np.mean(lick_counts[start:stop] > 2.0) > LICK_ERROR_THRESHOLD)
session_metadata = { ... }  # extensive metadata not used by decoder
```

iii. These are defensive measures the agent added that go beyond the reference processing.
