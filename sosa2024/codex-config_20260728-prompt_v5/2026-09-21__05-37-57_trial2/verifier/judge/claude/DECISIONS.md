# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all NWB files by globbing `sub-*/sub-*_behavior+ophys.nwb` under `/app/data`, sorted alphabetically. Each NWB file is opened with `pynwb.NWBHDF5IO` and read in full. All subjects, sessions, and trials found are included.

ii.
```python
data_root = Path("/app/data")
files = sorted(data_root.glob("sub-*/sub-*_behavior+ophys.nwb"))
...
with NWBHDF5IO(str(file_path), "r", load_namespaces=True) as io:
    nwb = io.read()
```

iii. The AI finds all NWB files via a glob pattern matching the DANDI-style directory structure. This is documented in the CONVERSION_NOTES.md data exploration step.

## 1-b. How are the data split into subjects?

i. Subjects are derived from the parent directory name of each NWB file (e.g., `sub-m3` → `m3`). Unique subjects are accumulated as sessions are processed; subject-to-index mapping is built incrementally.

ii.
```python
subject = file_path.parent.name.replace("sub-", "")
...
if meta.subject not in subject_to_idx:
    subject_to_idx[meta.subject] = len(subjects)
    subjects.append(meta.subject)
subject_idx.append(subject_to_idx[meta.subject])
```

iii. The directory structure follows the DANDI convention `sub-<id>`, and the AI uses this to extract subject identity.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session ID is parsed from the filename using a regex matching `_ses-(\d+)_`.

ii.
```python
session_match = re.search(r"_ses-(\d+)_", file_path.name)
session_id = session_match.group(1) if session_match else "unknown"
```

iii. The NWB filenames follow the pattern `sub-<mouse>_ses-<NN>_behavior+ophys.nwb`, making each file a distinct session.

## 1-d. How are the data split into trials?

i. Trial boundaries are found using the `trial_start` and `teleport` behavior time series. For each start (where `trial_start > 0`), the code searches forward for the next `teleport > 0` sample. This pairs each trial start with its corresponding teleport end.

ii.
```python
def pair_trial_bounds(trial_start: np.ndarray, teleport: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(trial_start > 0)
    stops = np.flatnonzero(teleport > 0)
    bounds: list[tuple[int, int]] = []
    stop_idx = 0
    for start in starts:
        while stop_idx < len(stops) and stops[stop_idx] <= start:
            stop_idx += 1
        if stop_idx >= len(stops):
            break
        stop = int(stops[stop_idx])
        bounds.append((int(start), stop))
        stop_idx += 1
    return bounds
```

iii. The AI uses a forward-search pairing: for each trial_start pulse, find the next teleport pulse after it. This handles any edge cases where teleport signals might occur before the first trial start.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered if: (1) `stop <= start`, (2) `stop > deconv.shape[0]` (exceeds neural data length), or (3) the environment variable has no valid values (`env_valid.size == 0`). After all filtering, sessions with fewer than 2 usable trials raise an error. There is no explicit minimum trial length filter.

ii.
```python
for trial_idx, (start, stop) in enumerate(trial_bounds):
    if stop <= start:
        continue
    if stop > deconv.shape[0]:
        raise ValueError(...)
    env_trial = environment[start:stop]
    env_valid = env_trial[env_trial >= 0]
    if env_valid.size == 0:
        continue
...
if len(neural_trials) < 2:
    raise ValueError(f"{meta.session_name}: fewer than 2 usable trials after conversion")
```

iii. The AI checks for degenerate trials (zero or negative length) and invalid environment periods. Sessions must have at least 2 usable trials for decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the NWB's `Deconvolved` module under `processing/ophys/Deconvolved`. This is suite2p's own deconvolution of raw fluorescence, stored directly in the NWB file.

ii.
```python
def load_curated_deconvolved(nwb) -> tuple[np.ndarray, np.ndarray, int]:
    ...
    deconv_mod = nwb.processing["ophys"]["Deconvolved"]
    plane_names = sorted(deconv_mod.roi_response_series.keys(), key=get_plane_number)
    arrays = []
    for plane_name in plane_names:
        rs = deconv_mod.roi_response_series[plane_name]
        ...
        plane_data = np.asarray(rs.data[:, plane_keep_local], dtype=np.float32)
        arrays.append(plane_data)
    deconv = np.concatenate(arrays, axis=1)
```

iii. The AI's CONVERSION_NOTES.md metadata states: "Deconvolved calcium events from NWB processing/ophys/Deconvolved." The AI treats the NWB's stored deconvolved signal as sufficient for the decoder task.

## 2-b. How is the `neural` data processed?

i. The deconvolved data is loaded directly from the NWB file. NaN values are replaced with zeros using `np.nan_to_num`. The data is cast to float16 for memory efficiency. No additional dF/F computation, baseline correction, smoothing, or re-deconvolution is performed.

ii.
```python
neural_trial = np.nan_to_num(deconv[start:stop, :].T, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16, copy=False)
```

iii. The AI uses the pre-computed deconvolved signal directly rather than recomputing it from raw fluorescence and neuropil traces.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells are filtered by the `iscell` column from the `ImageSegmentation/PlaneSegmentation` table. Only ROIs where `iscell[:, 0] == 1` are retained. No additional filtering for putative interneurons is applied.

ii.
```python
seg = nwb.processing["ophys"]["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]
iscell = np.asarray(seg["iscell"].data[:] ...)
plane_idx = np.asarray(seg["planeIdx"].data[:] ...).astype(int)
keep = iscell[:, 0] == 1
...
plane_keep_mask = keep & (plane_idx == plane_num)
```

iii. The AI uses suite2p's manual curation flag. It does not implement the paper's additional putative interneuron filter (speed correlation > 0.5).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is sliced using the same `[start:stop]` indices as the behavioral data within each trial. Since alignment is to trial start and the trial boundaries come from the behavior time series, no additional temporal shifting is needed.

ii.
```python
neural_trial = np.nan_to_num(deconv[start:stop, :].T, ...)
```

iii. Both neural and behavioral data share the same sampling grid (~15.5 Hz imaging frames), so indexing by the same trial boundaries aligns them.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is ~64.48 ms (1/15.5078125 Hz). No rebinning is applied. The AI validates that every session's frame dt matches the expected value.

ii.
```python
EXPECTED_FRAME_DT_S = 1.0 / 15.5078125
...
frame_dt_s = float(np.median(np.diff(frame_times)))
if not np.isclose(frame_dt_s, EXPECTED_FRAME_DT_S, atol=1e-9, rtol=0):
    raise ValueError(...)
```

iii. The AI hardcodes the expected frame rate and validates all sessions match. For multi-plane sessions (~31 Hz scanner rate), the behavior timestamps are still at ~15.5 Hz per plane.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position` behavior time series timestamps (`beh.time_series["position"].timestamps[:]`).

ii.
```python
frame_times = np.asarray(beh.time_series["position"].timestamps[:], dtype=np.float64)
...
trial_time_s = (frame_times[start:stop] - frame_times[start]).astype(np.float32, copy=False)
```

iii. All behavior time series share the same timestamps. The AI chose `position` timestamps as the source.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The trial start timestamp is subtracted from each timepoint's timestamp within the trial, yielding time in seconds from trial onset.

ii.
```python
trial_time_s = (frame_times[start:stop] - frame_times[start]).astype(np.float32, copy=False)
```

iii. Standard approach: subtract the first timestamp of the trial to get relative time.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both neural and behavioral data use the same frame-level indexing (`start:stop`), so they are inherently aligned. No additional alignment is needed.

ii. Same `start:stop` indices used for both neural and input data.

iii. The data is already on the same sampling grid (~15.5 Hz imaging frames).

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
environment = np.asarray(beh.time_series["environment"].data[:], dtype=np.float32)
...
env_trial = environment[start:stop]
env_valid = env_trial[env_trial >= 0]
env_code = float(mode_int(env_valid))
```

iii. The `environment` variable stores 0 or 1 for ENV1/ENV2. Pre-synchronization samples use -1 as a sentinel.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the mode (most frequent value) of valid environment samples (those >= 0) is computed. This single value is broadcast across all timepoints in the trial.

ii.
```python
def mode_int(values: np.ndarray) -> int:
    uniq, counts = np.unique(values.astype(int), return_counts=True)
    return int(uniq[np.argmax(counts)])
...
env_code = float(mode_int(env_valid))
input_trial = np.vstack([
    ...
    np.full(trial_time_s.shape, env_code, dtype=np.float32),
    ...
])
```

iii. Taking the mode handles any edge cases where a trial boundary might include a sentinel value. Environment is constant within a trial, so the mode returns the correct value.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the NWB `trial number` behavior time series. The value at the trial start is rounded to the nearest integer. Falls back to the loop index if the native value is negative.

ii.
```python
trial_number = np.asarray(beh.time_series["trial number"].data[:], dtype=np.float32)
...
trial_num_native = int(round(float(trial_number[start])))
if trial_num_native < 0:
    trial_num_native = trial_idx
```

iii. The native `trial number` stored in the NWB file is used when valid (>= 0). Pre-synchronization periods may have negative trial numbers, which trigger the fallback.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The raw trial number value at the start of the trial is rounded to an integer. It is broadcast as a constant across all timepoints in the trial.

ii.
```python
trial_num_native = int(round(float(trial_number[start])))
...
np.full(trial_time_s.shape, float(trial_num_native), dtype=np.float32),
```

iii. Rounding handles any floating-point precision issues in the stored data.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps. Reward event times are mapped to frame indices, producing a binary reward-per-frame array.

ii.
```python
reward_times = np.asarray(beh.time_series["Reward"].timestamps[:], dtype=np.float64)
reward_frames = rasterize_reward_events(frame_times, reward_times)
```

iii. The `Reward` time series uses its own timestamps, which are mapped to frame indices via `searchsorted`.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Per-trial reward outcomes are computed first (any reward frame within the trial → 1). Previous trial outcome is then a shifted version: `prev_reward_outcomes[1:] = reward_outcomes[:-1]`, with the first trial set to 0.

ii.
```python
reward_outcomes = trial_reward_outcomes(trial_bounds, reward_frames)
prev_reward_outcomes = np.zeros_like(reward_outcomes)
if reward_outcomes.size > 1:
    prev_reward_outcomes[1:] = reward_outcomes[:-1]
...
np.full(trial_time_s.shape, float(prev_reward_outcomes[trial_idx]), dtype=np.float32),
```

iii. The shift-by-one approach correctly assigns the previous trial's outcome. The first trial defaults to 0 (no previous trial).

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone label for the current trial. Reward zone labels are determined from the NWB `identifier` string, which encodes the session's scene name (e.g., `Env1_LocationA`, `Env1_LocationA_to_B`). The `scene_to_reward_labels()` function maps scene names to per-trial reward zone labels (A, B, or C), handling switch sessions where the zone changes after trial 30.

ii.
```python
meta = parse_identifier(nwb.identifier, file_path)
...
reward_labels = scene_to_reward_labels(meta.scene, len(trial_bounds))
...
rz_label = reward_labels[trial_idx]
rz_start, rz_stop = REWARD_ZONE_COORDS_CM[rz_label]
rz_dist = signed_distance_to_interval(pos_trial, rz_start, rz_stop)
```

```python
def scene_to_reward_labels(scene: str, ntrials: int, change_trial: int = SCENE_SWITCH_TRIAL) -> list[str]:
    if scene in {"Env1_LocationA", "Env2_LocationA", "Env3_LocationA"}:
        return ["A"] * ntrials
    ...
    if "A_to" in scene and scene.endswith("B"):
        return ["A"] * first_n + ["B"] * second_n
```

iii. The scene string from the NWB identifier encodes the experimental condition. The AI parses this to determine reward zones, including switch sessions where the zone changes after 30 trials.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from the animal's position to the nearest edge of the reward zone interval. Zero when inside the zone, negative when before, positive when past.

ii.
```python
def signed_distance_to_interval(position_cm, start_cm, stop_cm):
    out = np.zeros_like(position_cm, dtype=np.float32)
    below = position_cm < start_cm
    above = position_cm > stop_cm
    out[below] = position_cm[below] - start_cm
    out[above] = position_cm[above] - stop_cm
    return out
```

iii. Standard signed-distance-to-interval computation, consistent with the paper's reward-relative position concept.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit conditional assignments matching the instruction bins: `< -50`, `[-50, -10)`, `[-10, 0)`, `== 0`, `(0, 10]`, `(10, 50]`, `> 50`.

ii.
```python
def discretize_reward_distance(distance_cm: np.ndarray) -> np.ndarray:
    out = np.full(distance_cm.shape, 6, dtype=np.int8)
    out[distance_cm < -50.0] = 0
    out[(distance_cm >= -50.0) & (distance_cm < -10.0)] = 1
    out[(distance_cm >= -10.0) & (distance_cm < 0.0)] = 2
    out[distance_cm == 0.0] = 3
    out[(distance_cm > 0.0) & (distance_cm <= 10.0)] = 4
    out[(distance_cm > 10.0) & (distance_cm <= 50.0)] = 5
    out[distance_cm > 50.0] = 6
    return out
```

iii. The bin edges match the task instructions. The function uses explicit conditions rather than `np.digitize`.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same `start:stop` indices used for both position (used to compute distance) and neural data. No additional alignment needed.

ii.
```python
pos_trial = np.nan_to_num(position[start:stop], ...)
rz_dist = signed_distance_to_interval(pos_trial, rz_start, rz_stop)
```

iii. Data shares the same sampling grid.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = np.asarray(beh.time_series["position"].data[:], dtype=np.float32)
...
pos_trial = np.nan_to_num(position[start:stop], ...)
```

iii. The `position` variable records the animal's position in the VR corridor in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. NaN values are replaced with 0. The raw position is then discretized. No other processing.

ii.
```python
pos_trial = np.nan_to_num(position[start:stop], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
...
discretize_position(pos_trial)
```

iii. Position values are used directly from the NWB file after NaN handling.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins: `< 90`, `[90, 180)`, `[180, 270)`, `[270, 360)`, `>= 360`. This corresponds to 90 cm wide bins spanning the 450 cm track.

ii.
```python
def discretize_position(position_cm: np.ndarray) -> np.ndarray:
    out = np.full(position_cm.shape, 4, dtype=np.int8)
    out[position_cm < 90.0] = 0
    out[(position_cm >= 90.0) & (position_cm < 180.0)] = 1
    out[(position_cm >= 180.0) & (position_cm < 270.0)] = 2
    out[(position_cm >= 270.0) & (position_cm < 360.0)] = 3
    out[position_cm >= 360.0] = 4
    return out
```

iii. Five equal 90 cm bins over a 450 cm track, matching the instructions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same `start:stop` frame indices. No additional alignment needed.

ii. Same indexing as neural data.

iii. Data shares the same sampling grid.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick_counts = np.asarray(beh.time_series["lick"].data[:], dtype=np.float32)
```

iii. The `lick` variable records lick sensor counts per frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI applies a lick sensor correction: if more than 35% of timepoints in a trial have lick counts exceeding 2, the entire trial's lick data is zeroed out (flagged as a bad lick sensor trial). After this correction, lick counts are binarized (> 0 → 1).

ii.
```python
LICK_SENSOR_COUNT_THRESHOLD = 2.0
LICK_SENSOR_BAD_FRAC = 0.35

def correct_lick_trial(lick_counts: np.ndarray) -> tuple[np.ndarray, bool]:
    corrected = lick_counts.copy()
    bad_trial = float(np.mean(corrected > LICK_SENSOR_COUNT_THRESHOLD)) > LICK_SENSOR_BAD_FRAC
    if bad_trial:
        corrected[:] = 0.0
    corrected = (corrected > 0).astype(np.int8)
    return corrected, bad_trial
```

iii. The AI's metadata documents this rule: trials where the lick sensor appears to malfunction (excessive counts) have their lick data zeroed. This matches a pattern from the reference code's `get_timeseries_data` which performs similar lick sensor cleanup.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `start:stop` frame indices. No additional alignment needed.

ii. Same indexing as neural data.

iii. Data shares the same sampling grid.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB `identifier` string, which encodes the session's scene name. The `scene_to_reward_labels()` function maps scene names to per-trial reward zone labels.

ii.
```python
meta = parse_identifier(nwb.identifier, file_path)
...
reward_labels = scene_to_reward_labels(meta.scene, len(trial_bounds))
reward_codes = np.array([reward_label_to_code(label) for label in reward_labels], dtype=np.int8)
```

iii. The scene string provides the session's reward zone condition. For switch sessions, the zone changes after trial 30 (the `SCENE_SWITCH_TRIAL` constant).

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene string is parsed to determine the reward zone(s). For fixed-zone sessions, all trials get the same label. For switch sessions (e.g., `LocationA_to_B`), the first 30 trials get the pre-switch zone and remaining trials get the post-switch zone. Labels are mapped to codes: A=0, B=1, C=2.

ii.
```python
SCENE_SWITCH_TRIAL = 30

def scene_to_reward_labels(scene, ntrials, change_trial=SCENE_SWITCH_TRIAL):
    ...
    first_n = min(change_trial, ntrials)
    second_n = max(0, ntrials - change_trial)
    if "A_to" in scene and scene.endswith("B"):
        return ["A"] * first_n + ["B"] * second_n
    ...
```

iii. The switch after 30 trials matches the paper: "Each switch occurred after 30 trials."

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps. Reward events are rasterized to frame indices.

ii.
```python
reward_times = np.asarray(beh.time_series["Reward"].timestamps[:], dtype=np.float64)
reward_frames = rasterize_reward_events(frame_times, reward_times)
```

iii. The `Reward` time series records reward delivery events with separate timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward frame falls within the trial boundaries. Binary: 1 if rewarded, 0 if not. The value is constant across all timepoints in the trial.

ii.
```python
def trial_reward_outcomes(bounds, reward_frames):
    outcomes = np.zeros(len(bounds), dtype=np.int8)
    for idx, (start, stop) in enumerate(bounds):
        outcomes[idx] = int(np.any(reward_frames[start:stop] > 0))
    return outcomes
...
np.full(trial_time_s.shape, reward_outcomes[trial_idx], dtype=np.int8),
```

iii. Standard per-trial binary reward outcome.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Neural/behavior frame mismatch**: If neural data has exactly 1 extra frame, it is trimmed. Other mismatches raise an error.
- **NaN values**: `np.nan_to_num` replaces NaN, inf, -inf with 0 in neural, position, and speed data.
- **Invalid environment**: Trials with no valid environment values (all negative) are skipped.
- **Reward timestamp alignment**: An assertion checks that reward times align exactly to frame timestamps (within 1e-9).
- **Bad lick sensor**: Trials with excessive lick counts are zeroed out.

ii.
```python
frame_delta = deconv.shape[0] - frame_times.shape[0]
if frame_delta == 1:
    deconv = deconv[: frame_times.shape[0], :]
elif frame_delta != 0:
    raise ValueError(...)
...
neural_trial = np.nan_to_num(deconv[start:stop, :].T, nan=0.0, ...)
```

iii. These are defensive checks discovered during data exploration.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files** — I/O bound, reading large arrays from disk
2. **Processing each session** — loading deconvolved data and behavior variables
3. **Saving the pickle file**

ii. N/A

iii. The code is single-pass (no separate survey step), so each NWB file is loaded once.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates over each trial sequentially. Operations like discretization and distance computation could be applied to full session arrays before splitting, but variable trial lengths make this awkward. The `rasterize_reward_events` function and `trial_reward_outcomes` function use per-element loops that could be vectorized.

ii. N/A

iii. The per-trial loop is natural given variable-length trials.

## 13-c. What processing does the code repeat multiple times?

i. When `show_processing` is enabled, the code re-opens the NWB file to re-read behavior data for plotting. The `pair_trial_bounds` function is called multiple times in the plotting code. Otherwise, the main conversion is single-pass.

ii.
```python
if show_processing and session_idx < 2:
    ...
    with NWBHDF5IO(str(file_path), "r", load_namespaces=True) as io:
        nwb = io.read()
        beh = nwb.processing["behavior"]["BehavioralTimeSeries"]
        frame_times = np.asarray(beh.time_series["position"].timestamps[:], ...)
```

iii. The re-opening for plotting is a minor inefficiency that only affects 2 sessions.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes `n_planes` per session by counting planes in the deconvolved data, and stores per-cell plane indices. Since all neurons are labeled as brain region "CA1" regardless of plane, the plane-level information is not used in the final dataset structure (it's only in metadata). The lick sensor correction identifies bad trials but the trial itself is still included (just with zeroed lick data rather than being excluded).

ii.
```python
plane_per_cell = np.full(plane_data.shape[1], plane_num, dtype=np.int16)
...
# plane_per_cell is computed but only used in session_brain_region_idx which returns all zeros
```

iii. The plane information is tracked for metadata but doesn't affect the downstream brain_region_idx.
