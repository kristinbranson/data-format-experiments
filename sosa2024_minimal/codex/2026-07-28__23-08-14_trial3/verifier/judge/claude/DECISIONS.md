# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `h5py` (not `pynwb`) to directly read the HDF5-formatted NWB files. It discovers all NWB files via a sorted glob pattern `sub-*/sub-*_behavior+ophys.nwb` under the data directory. Each file is opened with `h5py.File(path, "r")` and the relevant datasets are read from the HDF5 hierarchy. The AI switched from pynwb to h5py because pynwb was slow, had version mismatch warnings, and the NWB files lacked standard `trials`/`units` tables.

ii.
```python
def nwb_files(data_dir: str):
    paths = sorted(Path(data_dir).glob("sub-*/sub-*_behavior+ophys.nwb"))
    if not paths:
        raise FileNotFoundError(f"No NWB files found under {data_dir}")
    return paths
...
def convert_session(path: Path, lick_error_threshold: float):
    subject, day = parse_subject_and_day(path)
    with h5py.File(path, "r") as f:
        beh = f["processing/behavior/BehavioralTimeSeries"]
        ...
```

iii. The AI chose h5py over pynwb for pragmatic reasons: pynwb was slow (>1s per file vs 0.01s for h5py), had NWB schema version mismatches, and the NWB files lacked standard trials/units tables. Since all needed data was accessible as raw HDF5 datasets, h5py was simpler and faster. The glob pattern ensures all NWB files are found.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from the parent directory name of each NWB file by stripping the `sub-` prefix. All unique subjects are collected and sorted.

ii.
```python
def parse_subject_and_day(path: Path):
    subject = path.parent.name.replace("sub-", "")
    ...
...
subjects = sorted({record["subject"] for record in session_records})
```

iii. The directory structure `sub-<id>/` directly encodes subject identity. Sorting ensures deterministic ordering.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session number (day) is parsed from the filename using a regex pattern `ses-(\d+)`.

ii.
```python
def parse_subject_and_day(path: Path):
    subject = path.parent.name.replace("sub-", "")
    match = re.search(r"ses-(\d+)", path.name)
    if match is None:
        raise ValueError(f"Could not parse session/day from {path.name}")
    return subject, int(match.group(1))
```

iii. The NWB filenames follow the BIDS-like convention `sub-<subject>_ses-<number>_behavior+ophys.nwb`, making session parsing straightforward.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from framewise behavioral channels: `trial_start` marks trial onsets and `teleport` marks trial ends. The AI uses `np.flatnonzero(signal > 0)` for both signals. Each trial spans from `trial_start` (inclusive) to `teleport` (exclusive).

ii.
```python
trial_start_signal = beh["trial_start"]["data"][()]
teleport_signal = beh["teleport"]["data"][()]
trial_starts = np.flatnonzero(trial_start_signal > 0)
teleports = np.flatnonzero(teleport_signal > 0)
if trial_starts.size != teleports.size:
    raise ValueError(...)
...
for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    T = int(stop - start)
    ...
    pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. The AI notes that NWB files lack standard trials tables, so trials must be reconstructed from framewise signals. The assertion that `trial_starts.size == teleports.size` validates the reconstruction. Note: the AI does NOT perform onset detection for teleport (it takes all frames where teleport > 0, not just the transition from 0 to positive). This works if teleport is a single-frame pulse in the data.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies three trial-level filters: (1) lick sensor error trials are dropped when more than 35% of frames have lick count > 2, (2) trials with T <= 0 are dropped, (3) trials where the zone or environment label could not be inferred (None) are dropped. A total of 69 out of 12216 trials were dropped for lick-sensor error.

ii.
```python
def drop_lick_error_trials(lick, trial_starts, teleports, threshold):
    drop_mask = np.zeros(len(trial_starts), dtype=bool)
    error_fraction = np.zeros(len(trial_starts), dtype=np.float32)
    for idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
        lick_trial = lick[start:stop]
        frac = float(np.mean(lick_trial > 2))
        error_fraction[idx] = frac
        if frac > threshold:
            drop_mask[idx] = True
    return drop_mask, error_fraction
```

iii. The lick error filtering comes from the paper's reference code (`correct_lick_sensor_error`), which uses a 0.35 threshold. The AI justifies dropping (rather than NaN-filling) these trials because the decoder format requires dense arrays.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `Deconvolved` calcium event time series, specifically `processing/ophys/Deconvolved/plane0/data`. Only `plane0` is used, even for multi-plane animals (m17, m18).

ii.
```python
deconv = f["processing/ophys/Deconvolved/plane0/data"]
...
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. The paper states analyses used deconvolved calcium events. The AI only uses plane0 because multi-plane mice (m17, m18) only have the plane0 response series linked in NWB; using unlinked segmentation rows from the second plane could cause misindexing.

## 2-b. How is the `neural` data processed?

i. No additional processing (smoothing, rebinning) is applied. The deconvolved data is read directly, filtered by the `iscell` mask, and transposed to (n_neurons, n_timepoints) format.

ii.
```python
rois = f["processing/ophys/Fluorescence/plane0/rois"][()]
iscell = f["processing/ophys/ImageSegmentation/PlaneSegmentation/iscell"][()][:, 0].astype(bool)
selected_mask = iscell[rois]
selected_indices = np.flatnonzero(selected_mask)
...
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. The AI reads ROI indices from the Fluorescence plane0 table, applies the iscell mask from PlaneSegmentation, and only selects neurons that pass both criteria.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data is filtered using the `iscell` flag from `processing/ophys/ImageSegmentation/PlaneSegmentation/iscell`. Only ROIs with `iscell == True` that are also referenced by the `plane0` response series are retained.

ii.
```python
iscell = f["processing/ophys/ImageSegmentation/PlaneSegmentation/iscell"][()][:, 0].astype(bool)
selected_mask = iscell[rois]
selected_indices = np.flatnonzero(selected_mask)
selected_rois = rois[selected_mask]
```

iii. The iscell mask is the standard Suite2p quality control flag. The AI's approach restricts to plane0-linked ROIs to avoid misindexing for multi-plane animals.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start. Since trials are defined by trial_start indices and neural data uses the same frame indices as behavior, no additional alignment is needed. The neural data for each trial is simply `deconv[start:stop, :]`.

ii.
```python
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. Both neural and behavioral data share the same framewise indexing in the NWB file, so slicing by the same start:stop indices aligns them.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The time bin size is computed as the median of inter-timestamp differences from the behavioral timestamps, yielding approximately 64.48 ms across all sessions.

ii.
```python
dt_sec = float(np.median(np.diff(times)))
...
"time_bin_size": float(np.median(time_bin_sizes_ms)),
```

iii. The bin size is consistent across all sessions including multi-plane mice. No rebinning is needed since the data is already at the imaging frame rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position` timestamps in `processing/behavior/BehavioralTimeSeries/position/timestamps`.

ii.
```python
times = beh["position"]["timestamps"][()].astype(np.float64)
...
time_trial = (times[start:stop] - times[start]).astype(np.float32)
```

iii. The position timestamps are used as the reference time base for all behavioral data. They are identical across all behavior channels.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp within each trial is subtracted from all timestamps in that trial, giving time relative to trial start in seconds.

ii.
```python
time_trial = (times[start:stop] - times[start]).astype(np.float32)
```

iii. Straightforward computation: subtract the trial start time to get time-from-trial-start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The behavioral timestamps and neural data share the same frame indices. Both are sliced with the same `start:stop` range, so they are inherently aligned.

ii.
```python
time_trial = (times[start:stop] - times[start]).astype(np.float32)
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. Same indexing ensures alignment. No interpolation or resampling needed.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series in `processing/behavior/BehavioralTimeSeries/environment/data`.

ii.
```python
environment = beh["environment"]["data"][()].astype(np.float32)
```

iii. The environment channel records the virtual reality environment type (0 or 1) at each frame.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the AI takes the median of valid environment values (>= 0) within the trial. It then applies a 30-trial switch schedule inference: the mode of the first 30 trials is the "pre" label, and the mode of trials after 30 is the "post" label. If pre != post, the session is classified as a switch session. Missing or inconsistent values are filled using this schedule.

ii.
```python
observed_env = []
for start, stop in zip(trial_starts, teleports):
    env_trial = environment[start:stop]
    env_trial = env_trial[env_trial >= 0]
    observed_env.append(int(np.rint(np.nanmedian(env_trial))) if env_trial.size else None)

env_schedule = infer_constant_or_switch_schedule(observed_env)
...
env_label = env_schedule["filled"][trial_idx]
np.full(T, env_label, dtype=np.float32)
```

iii. The AI uses the 30-trial switch boundary from the paper's experimental design. This handles edge cases where some trials might have missing or inconsistent environment values.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number` behavior time series in `processing/behavior/BehavioralTimeSeries/trial number/data`. Falls back to the loop index (`trial_idx`) if no valid values exist.

ii.
```python
trial_number = beh["trial number"]["data"][()].astype(np.float32)
...
trial_number_values = trial_number[start:stop]
trial_number_values = trial_number_values[trial_number_values >= 0]
trial_number_scalar = (
    float(np.nanmedian(trial_number_values))
    if trial_number_values.size
    else float(trial_idx)
)
```

iii. The AI reads the per-trial number from the recorded NWB channel and takes the median of valid (>= 0) values within each trial window.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The median of valid `trial number` values (>= 0) within each trial's frame range is computed. If no valid values exist, the sequential trial index is used as fallback. The result is a scalar repeated across all timepoints.

ii.
```python
trial_number_values = trial_number[start:stop]
trial_number_values = trial_number_values[trial_number_values >= 0]
trial_number_scalar = (
    float(np.nanmedian(trial_number_values))
    if trial_number_values.size
    else float(trial_idx)
)
np.full(T, trial_number_scalar, dtype=np.float32)
```

iii. The AI uses the recorded trial number rather than a simple loop counter, which could better reflect the actual experimental trial numbering.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps in `processing/behavior/BehavioralTimeSeries/Reward/timestamps`.

ii.
```python
reward_times = beh["Reward"]["timestamps"][()].astype(np.float64)
```

iii. Reward delivery events have their own timestamps separate from the behavior sampling rate.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, reward outcome is determined by checking if any reward timestamp falls within the trial's time window. Previous trial outcome is then computed as a shifted version: `[0] + reward_outcomes[:-1]`. The first trial defaults to 0.

ii.
```python
def reward_outcomes_per_trial(reward_times, times, trial_starts, teleports):
    outcomes = []
    for start, stop in zip(trial_starts, teleports):
        has_reward = np.any((reward_times >= times[start]) & (reward_times < times[stop]))
        outcomes.append(int(has_reward))
    return outcomes
...
previous_reward_outcomes = [0] + reward_outcomes[:-1]
```

iii. The instructions specify previous trial outcome as binary (omitted=0, rewarded=1). The shift-by-one approach is clean and equivalent to checking the previous trial's reward status.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the inferred reward zone label for the current trial. The reward zone boundaries come from the paper's defined zones (A: 80-130cm, B: 200-250cm, C: 320-370cm).

ii.
```python
pos = beh["position"]["data"][()].astype(np.float32)
...
zone_bounds_cm = ZONE_BOUNDS_CM[zone_label]
dist_trial = signed_distance_to_zone(pos_trial, zone_bounds_cm)
```

iii. The zone label is inferred from the framewise `reward_zone` signal and animal position, then filled using the 30-trial switch schedule.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed: 0 when inside the zone, negative when before the zone (position < zone start), positive when past the zone (position > zone end).

ii.
```python
def signed_distance_to_zone(pos_cm, zone_bounds_cm):
    zone_start, zone_end = zone_bounds_cm
    dist = np.zeros_like(pos_cm, dtype=np.float32)
    before = pos_cm < zone_start
    after = pos_cm > zone_end
    dist[before] = pos_cm[before] - zone_start
    dist[after] = pos_cm[after] - zone_end
    return dist
```

iii. This matches the paper's concept of signed distance relative to reward zone boundaries.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous distance is binned into 7 categories using explicit conditional logic matching the instruction bins.

ii.
```python
def bin_distance_to_zone(dist_cm):
    out = np.zeros(dist_cm.shape, dtype=np.int64)
    out[dist_cm < -50.0] = 0
    out[(dist_cm >= -50.0) & (dist_cm < -10.0)] = 1
    out[(dist_cm >= -10.0) & (dist_cm < 0.0)] = 2
    out[dist_cm == 0.0] = 3
    out[(dist_cm > 0.0) & (dist_cm <= 10.0)] = 4
    out[(dist_cm > 10.0) & (dist_cm <= 50.0)] = 5
    out[dist_cm > 50.0] = 6
    return out
```

iii. The bin boundaries match the instructions exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position data and neural data share the same frame indices. Both are sliced with `start:stop`, ensuring alignment.

ii.
```python
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. Same indexing as neural data ensures alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series in `processing/behavior/BehavioralTimeSeries/position/data`.

ii.
```python
pos = beh["position"]["data"][()].astype(np.float32)
...
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. The position variable records the animal's position in the VR corridor in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450] cm before binning. No other processing is applied.

ii.
```python
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. Clipping ensures positions stay within the defined corridor range.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The 450 cm corridor is divided into 5 equal 90 cm bins: [0, 90), [90, 180), [180, 270), [270, 360), [360, 450]. Implementation uses integer division by 90, capped at 4.

ii.
```python
def bin_absolute_position(pos_cm):
    pos_clipped = np.clip(pos_cm, 0.0, 450.0 - 1e-6)
    return np.minimum((pos_clipped / 90.0).astype(np.int64), 4)
```

iii. The instructions specify "5 equal-sized bins." The corridor is 450 cm, so 90 cm bins are mathematically equal.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indexing as neural data. Both sliced with `start:stop`.

ii.
```python
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. Verified by shared frame indices.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series in `processing/behavior/BehavioralTimeSeries/lick/data`.

ii.
```python
lick = beh["lick"]["data"][()].astype(np.float32)
```

iii. The lick channel records lick events at each frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
lick_trial = (lick[start:stop] > 0).astype(np.int64)
```

iii. The instructions specify binary output (0 = no, 1 = yes). Raw lick values can be > 1, so thresholding at > 0 converts to binary.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indexing as neural data via `start:stop` slicing.

ii.
```python
lick_trial = (lick[start:stop] > 0).astype(np.int64)
```

iii. Shared frame indices ensure alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` behavior time series, animal `position`, and `Reward` timestamps. The zone label is inferred by taking the median position when the reward_zone signal is active, finding the closest defined zone center, then applying a 30-trial switch schedule to fill gaps.

ii.
```python
def infer_trial_zone_positions(pos, reward_zone_signal, times, reward_times, trial_starts, teleports):
    observed_labels = []
    observed_positions = []
    for start, stop in zip(trial_starts, teleports):
        pos_trial = pos[start:stop]
        zone_signal_trial = reward_zone_signal[start:stop]
        zone_frames = np.flatnonzero(zone_signal_trial > 0)
        zone_position_cm = np.nan
        if zone_frames.size > 0:
            zone_position_cm = float(np.nanmedian(pos_trial[zone_frames]))
        else:
            in_trial_reward = reward_times[(reward_times >= times[start]) & (reward_times < times[stop])]
            if in_trial_reward.size > 0:
                reward_idx = np.searchsorted(times, in_trial_reward[0], side="left")
                reward_idx = min(max(reward_idx, start), stop - 1)
                zone_position_cm = float(pos[reward_idx])
        observed_positions.append(zone_position_cm)
        observed_labels.append(infer_zone_from_position(zone_position_cm))
    return observed_labels, observed_positions
```

iii. The AI infers zone labels from position data, using the closest zone center, and fills missing observations with the dominant pre/post-switch schedule.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Two-step process: (1) Infer per-trial zone from median position during active reward_zone signal, mapping to nearest zone center. If no zone signal is active, use position at reward delivery time. (2) Apply `infer_constant_or_switch_schedule` with a 30-trial boundary: take mode of first 30 trials vs. remaining trials. If modes differ, fill with pre-mode for trials 0-29 and post-mode for trials 30+. Output is encoded as 0=A, 1=B, 2=C.

ii.
```python
def infer_constant_or_switch_schedule(observed_values, switch_trial=30):
    ntrials = len(observed_values)
    pre = mode_or_none(observed_values[: min(switch_trial, ntrials)])
    post = mode_or_none(observed_values[switch_trial:]) if ntrials > switch_trial else None
    overall = mode_or_none(observed_values)
    is_switch = pre is not None and post is not None and pre != post
    filled = []
    if is_switch:
        for trial_idx in range(ntrials):
            filled.append(pre if trial_idx < switch_trial else post)
    else:
        filled = [overall for _ in range(ntrials)]
    return { ... }
...
zone_schedule = infer_constant_or_switch_schedule(observed_zone_labels)
...
np.full(T, ZONE_TO_INDEX[zone_label], dtype=np.int64)
```

iii. The 30-trial switch boundary matches the paper's experimental design. The schedule-filling approach handles omission trials that lack direct reward zone observations.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` timestamps in `processing/behavior/BehavioralTimeSeries/Reward/timestamps`.

ii.
```python
reward_times = beh["Reward"]["timestamps"][()].astype(np.float64)
```

iii. Reward delivery events have their own timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward timestamp falls within the trial's time window `[times[start], times[stop])`. Output is binary: 1 if rewarded, 0 otherwise. The value is constant across all timepoints in the trial.

ii.
```python
def reward_outcomes_per_trial(reward_times, times, trial_starts, teleports):
    outcomes = []
    for start, stop in zip(trial_starts, teleports):
        has_reward = np.any((reward_times >= times[start]) & (reward_times < times[stop]))
        outcomes.append(int(has_reward))
    return outcomes
...
np.full(T, reward_outcomes[trial_idx], dtype=np.int64)
```

iii. Simple binary check per trial. The `np.any` aggregation handles the case where multiple reward events could occur.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Lick sensor errors**: Trials with >35% of frames having lick > 2 are dropped entirely (69 trials out of 12216).
- **Zero-length trials**: Trials with T <= 0 are dropped.
- **Missing zone/environment labels**: Trials where zone or environment could not be inferred (None) are dropped.
- **Missing reward zone observations**: On omission trials, the zone is inferred from reward timestamps or filled using the 30-trial switch schedule.
- **Position clipping**: Position values are clipped to [0, 450] cm.
- **Multi-plane handling**: Only plane0 is used to avoid misindexing with unlinked segmentation data.

ii.
```python
if drop_mask[trial_idx]:
    dropped_trials.append(trial_idx)
    continue
...
if T <= 0:
    dropped_trials.append(trial_idx)
    continue
...
if zone_label is None or env_label is None:
    dropped_trials.append(trial_idx)
    continue
```

iii. The AI prioritizes data integrity by dropping questionable trials rather than attempting to fix them. The lick error threshold comes from the paper's reference code.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files with h5py** and reading large deconvolved neural data arrays for each session
2. **Per-trial loop** in `convert_session` that slices and processes each trial individually
3. **Saving the pickle file** with the full dataset

ii. N/A

iii. Using h5py instead of pynwb significantly speeds up file loading. The single-pass design (no separate survey step) is more efficient than the reference's two-pass approach.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates over each trial sequentially. Operations like `signed_distance_to_zone`, `bin_distance_to_zone`, `bin_absolute_position`, and `bin_speed` are all applied per-trial but could theoretically be vectorized over the full session before splitting into trials. The `reward_outcomes_per_trial` loop and `drop_lick_error_trials` loop could also potentially be vectorized.

ii. N/A

iii. Variable trial lengths make full vectorization awkward (would require padding/masking), so the per-trial loop is a natural choice.

## 13-c. What processing does the code repeat multiple times?

i. The AI's code makes a single pass through all NWB files, so there is no repeated loading. However, some per-trial operations compute values that could be precomputed once per session (e.g., reward outcome checking for each trial scans the full reward_times array each time).

ii. N/A

iii. The single-pass design avoids the main source of repeated work (file I/O). Minor repeated computations within the trial loop are negligible.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI's code computes and stores extensive session metadata (observed zone positions, observed labels, lick error fractions per trial, environment schedules, etc.) that is included in the metadata dict but not used by the downstream decoder. The `infer_constant_or_switch_schedule` function computes switch detection that is only used for filling missing labels.

ii.
```python
session_info["observed_zone_positions_cm"] = observed_zone_positions
session_info["observed_zone_labels"] = observed_zone_labels
session_info["observed_environment"] = observed_env
session_info["lick_error_fraction_per_trial"] = lick_error_fraction.tolist()
```

iii. This extra metadata is useful for debugging and validation but adds overhead to the pickle file size without affecting decoder performance.
