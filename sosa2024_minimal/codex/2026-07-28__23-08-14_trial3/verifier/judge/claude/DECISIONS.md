# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all NWB files under the data directory using `Path.glob("sub-*/sub-*_behavior+ophys.nwb")`. Files are sorted alphabetically. Each NWB file is opened with `h5py` (not `pynwb`) and data is read directly from HDF5 paths. All subjects and sessions found under the data directory are processed.

ii.
```python
def nwb_files(data_dir: str):
    paths = sorted(Path(data_dir).glob("sub-*/sub-*_behavior+ophys.nwb"))
    if not paths:
        raise FileNotFoundError(f"No NWB files found under {data_dir}")
    return paths
...
with h5py.File(path, "r") as f:
    beh = f["processing/behavior/BehavioralTimeSeries"]
```

iii. The AI chose `h5py` over `pynwb` for direct HDF5 access. The CONVERSION_NOTES state that NWB files do not expose NWB trials/units tables, so direct HDF5 access was preferred.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the parent directory name of each NWB file by stripping the `sub-` prefix. All unique subjects are collected and sorted.

ii.
```python
def parse_subject_and_day(path: Path):
    subject = path.parent.name.replace("sub-", "")
    ...
...
subjects = sorted({record["subject"] for record in session_records})
```

iii. Directory naming convention `sub-<id>` is used to identify subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session/day number is parsed from the filename using regex.

ii.
```python
def parse_subject_and_day(path: Path):
    subject = path.parent.name.replace("sub-", "")
    match = re.search(r"ses-(\d+)", path.name)
    ...
    return subject, int(match.group(1))
```

iii. Filename convention `ses-<number>` identifies sessions. The CONVERSION_NOTES confirm 152 sessions across 11 subjects.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from framewise behavior channels: `trial_start` marks the start of each trial, and `teleport` marks the end. Frames from `trial_start` inclusive to `teleport` exclusive are used.

ii.
```python
trial_starts = np.flatnonzero(trial_start_signal > 0)
teleports = np.flatnonzero(teleport_signal > 0)
...
for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    ...
    T = int(stop - start)
    ...
    pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. CONVERSION_NOTES: "The NWB files do not expose a ready-made NWB trials table, so trials were reconstructed from the framewise behavior channels." Trials use frames from trial_start inclusive to teleport exclusive.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on two criteria: (1) lick sensor error - trials where more than 35% of frames have lick count > 2 are dropped, (2) trials with T <= 0 or where zone/environment label is None are dropped.

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
...
if drop_mask[trial_idx]:
    dropped_trials.append(trial_idx)
    continue
if T <= 0:
    dropped_trials.append(trial_idx)
    continue
```

iii. CONVERSION_NOTES: "The reference code includes correct_lick_sensor_error, which marks a trial as invalid when more than a threshold fraction of frames have lick count > 2." 69 trials dropped for lick error out of 12216.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `processing/ophys/Deconvolved/plane0/data`, the deconvolved calcium event time series.

ii.
```python
deconv = f["processing/ophys/Deconvolved/plane0/data"]
...
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. CONVERSION_NOTES: "The paper states that analyses used deconvolved calcium event time series sampled at the imaging frame rate."

## 2-b. How is the `neural` data processed?

i. The AI reads only `plane0` data and selects ROIs based on the `iscell` flag from `ImageSegmentation/PlaneSegmentation`. For multi-plane animals (m17, m18), only the published `plane0` response series is used. The ROI indices from `Fluorescence/plane0/rois` are used to map into the `iscell` array.

ii.
```python
deconv = f["processing/ophys/Deconvolved/plane0/data"]
rois = f["processing/ophys/Fluorescence/plane0/rois"][()]
iscell = f["processing/ophys/ImageSegmentation/PlaneSegmentation/iscell"][()][:, 0].astype(bool)
selected_mask = iscell[rois]
selected_indices = np.flatnonzero(selected_mask)
...
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. CONVERSION_NOTES: "Multi-plane mice m17 and m18 include only the published plane0 response series in NWB; the converter uses the available response series rather than unlinked segmentation rows from the second plane."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data is filtered using the `iscell` flag from `PlaneSegmentation`. Only ROIs where `iscell` is true are included. Only ROIs referenced by `Fluorescence/plane0/rois` are considered.

ii.
```python
rois = f["processing/ophys/Fluorescence/plane0/rois"][()]
iscell = f["processing/ophys/ImageSegmentation/PlaneSegmentation/iscell"][()][:, 0].astype(bool)
selected_mask = iscell[rois]
selected_indices = np.flatnonzero(selected_mask)
selected_rois = rois[selected_mask]
```

iii. CONVERSION_NOTES: "ROIs referenced by processing/ophys/Fluorescence/plane0/rois and retained only if the corresponding PlaneSegmentation iscell flag is true."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start. Neural data is sliced using the same `start:stop` indices as behavior, where `start` is the `trial_start` frame index. No additional alignment processing is needed since neural and behavior share the same frame indices.

ii.
```python
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. The trial start is the alignment event; slicing from `start` achieves this directly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is kept at the native imaging frame rate (~64.48 ms). No rebinning is applied. The time bin size is computed from the median of timestamp differences.

ii.
```python
dt_sec = float(np.median(np.diff(times)))
...
"time_bin_size": float(np.median(time_bin_sizes_ms)),
```

iii. CONVERSION_NOTES: "No extra temporal smoothing or rebinning was added. The recovered bin size is constant across all sessions: 64.48362720402656 ms."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position` timestamps (`processing/behavior/BehavioralTimeSeries/position/timestamps`).

ii.
```python
times = beh["position"]["timestamps"][()].astype(np.float64)
...
time_trial = (times[start:stop] - times[start]).astype(np.float32)
```

iii. The position timestamps provide the frame times; subtracting the first frame's time gives time from trial start.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp within the trial is subtracted from all timestamps in that trial.

ii.
```python
time_trial = (times[start:stop] - times[start]).astype(np.float32)
```

iii. Standard approach to compute relative time from trial onset.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both neural and behavior data use the same frame indices (`start:stop`), so they are inherently aligned.

ii.
```python
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
time_trial = (times[start:stop] - times[start]).astype(np.float32)
```

iii. Same indexing ensures alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
environment = beh["environment"]["data"][()].astype(np.float32)
...
observed_env.append(int(np.rint(np.nanmedian(env_trial))) if env_trial.size else None)
```

iii. The `environment` channel records the VR environment type per frame.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI computes the median environment value per trial, then uses a "constant or switch" schedule inference. If the pre-switch (first 30 trials) and post-switch modes differ, a switch is detected and labels are filled accordingly. Otherwise a constant value is used for all trials.

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
input_trial = np.vstack([
    ...
    np.full(T, env_label, dtype=np.float32),
    ...
])
```

iii. CONVERSION_NOTES: "Environment identity is inferred the same way from the framewise environment channel, again using the 30-trial switch rule when needed."

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number` behavior time series in the NWB file.

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

iii. The NWB `trial number` variable provides the trial index; the AI takes its median value within the trial window.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The median of the `trial number` values within the trial window (filtered for values >= 0) is used. If no valid values exist, the loop index `trial_idx` is used as fallback. The value is constant across all timepoints in the trial.

ii.
```python
trial_number_values = trial_number[start:stop]
trial_number_values = trial_number_values[trial_number_values >= 0]
trial_number_scalar = (
    float(np.nanmedian(trial_number_values))
    if trial_number_values.size
    else float(trial_idx)
)
...
np.full(T, trial_number_scalar, dtype=np.float32),
```

iii. The AI chose to use the stored trial number variable rather than a simple loop counter.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from `Reward/timestamps` in the behavior time series. Reward times are compared against trial windows to determine if a reward occurred.

ii.
```python
reward_times = beh["Reward"]["timestamps"][()].astype(np.float64)
...
def reward_outcomes_per_trial(reward_times, times, trial_starts, teleports):
    outcomes = []
    for start, stop in zip(trial_starts, teleports):
        has_reward = np.any((reward_times >= times[start]) & (reward_times < times[stop]))
        outcomes.append(int(has_reward))
    return outcomes
```

iii. Reward timestamps are checked against each trial's time window.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, reward outcome is determined by checking if any reward timestamp falls within the trial window. Previous trial outcome is then the reward outcome of the preceding trial, with the first trial set to 0.

ii.
```python
reward_outcomes = reward_outcomes_per_trial(reward_times, times, trial_starts, teleports)
previous_reward_outcomes = [0] + reward_outcomes[:-1]
...
np.full(T, float(previous_reward_outcomes[trial_idx]), dtype=np.float32),
```

iii. Simple shift of reward outcomes by one trial. First trial defaults to 0.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the inferred reward zone label for each trial. Reward zone labels are inferred from the `reward_zone` channel and animal position using a "constant or switch" schedule with a 30-trial switch boundary.

ii.
```python
pos = beh["position"]["data"][()].astype(np.float32)
reward_zone_signal = beh["reward_zone"]["data"][()].astype(np.float32)
...
observed_zone_labels, observed_zone_positions = infer_trial_zone_positions(
    pos=pos, reward_zone_signal=reward_zone_signal, times=times,
    reward_times=reward_times, trial_starts=trial_starts, teleports=teleports,
)
zone_schedule = infer_constant_or_switch_schedule(observed_zone_labels)
...
zone_label = zone_schedule["filled"][trial_idx]
zone_bounds_cm = ZONE_BOUNDS_CM[zone_label]
dist_trial = signed_distance_to_zone(pos_trial, zone_bounds_cm)
```

iii. CONVERSION_NOTES: "Reward-zone labels are inferred from framewise reward_zone observations and filled with the dominant pre/post-switch label when omission trials lack direct observations."

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed from the animal position to the nearest edge of the assigned reward zone. Distance is negative before the zone, 0 inside, and positive after. Position is clipped to [0, 450].

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

iii. Standard signed distance computation relative to a zone interval.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit conditional logic:
- 0: < -50 cm
- 1: -50 to -10 cm
- 2: -10 to < 0 cm
- 3: exactly 0 cm
- 4: > 0 to 10 cm
- 5: 10 to 50 cm
- 6: > 50 cm

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

iii. Bin edges match the specification in the instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indices (`start:stop`) are used for both neural and position data, ensuring alignment.

ii.
```python
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. Same indexing ensures alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
pos = beh["position"]["data"][()].astype(np.float32)
...
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. The position variable records the animal's location in the VR corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450] range, then discretized into 5 equal bins of 90 cm each (0-90, 90-180, 180-270, 270-360, 360-450).

ii.
```python
def bin_absolute_position(pos_cm):
    pos_clipped = np.clip(pos_cm, 0.0, 450.0 - 1e-6)
    return np.minimum((pos_clipped / 90.0).astype(np.int64), 4)
```

iii. The corridor spans 0-450 cm, so 5 equal bins of 90 cm each are used.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is divided by 90 and truncated to integer, capped at 4. This creates 5 bins: [0, 90), [90, 180), [180, 270), [270, 360), [360, 450].

ii.
```python
def bin_absolute_position(pos_cm):
    pos_clipped = np.clip(pos_cm, 0.0, 450.0 - 1e-6)
    return np.minimum((pos_clipped / 90.0).astype(np.int64), 4)
```

iii. Output values defined as: "0 to <90 cm", "90 to <180 cm", "180 to <270 cm", "270 to <360 cm", "360 to 450 cm".

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices (`start:stop`) used for both.

ii.
```python
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. Same indexing ensures alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = beh["lick"]["data"][()].astype(np.float32)
...
lick_trial = (lick[start:stop] > 0).astype(np.int64)
```

iii. The `lick` channel records lick events per frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any lick value > 0 is set to 1, otherwise 0.

ii.
```python
lick_trial = (lick[start:stop] > 0).astype(np.int64)
```

iii. Instructions specify binary output (no/yes).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices (`start:stop`) used for both neural and lick data.

ii.
```python
lick_trial = (lick[start:stop] > 0).astype(np.int64)
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. Same indexing ensures alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` behavior time series, `position`, and `Reward/timestamps`. The median position when `reward_zone > 0` is computed, and the closest zone center (A, B, or C) is assigned. Missing labels are filled using a constant-or-switch schedule with a 30-trial boundary.

ii.
```python
def infer_trial_zone_positions(pos, reward_zone_signal, times, reward_times, trial_starts, teleports):
    observed_labels = []
    for start, stop in zip(trial_starts, teleports):
        zone_frames = np.flatnonzero(zone_signal_trial > 0)
        zone_position_cm = np.nan
        if zone_frames.size > 0:
            zone_position_cm = float(np.nanmedian(pos_trial[zone_frames]))
        ...
        observed_labels.append(infer_zone_from_position(zone_position_cm))
    return observed_labels, observed_positions

def infer_zone_from_position(zone_position_cm):
    return min(ZONE_CENTERS_CM, key=lambda label: abs(zone_position_cm - ZONE_CENTERS_CM[label]))

zone_schedule = infer_constant_or_switch_schedule(observed_zone_labels)
```

iii. CONVERSION_NOTES: "The converter infers the reward-zone label of each trial from the framewise reward_zone channel and animal position."

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Per trial: (1) find frames where `reward_zone > 0`, (2) compute median position at those frames, (3) assign to nearest zone center. If no reward zone frames, check for reward timestamps in the trial and use position at reward time. Missing labels are filled using the mode of the first 30 trials (pre-switch) and remaining trials (post-switch). Final output is 0=A, 1=B, 2=C.

ii.
```python
zone_schedule = infer_constant_or_switch_schedule(observed_zone_labels)
...
zone_label = zone_schedule["filled"][trial_idx]
...
np.full(T, ZONE_TO_INDEX[zone_label], dtype=np.int64),
```

iii. CONVERSION_NOTES: "On omission trials where the zone may not be directly observed, the converter fills missing labels using the dominant pre-switch and post-switch labels with the paper's 30-trial switch boundary."

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from `Reward/timestamps` in the behavior time series.

ii.
```python
reward_times = beh["Reward"]["timestamps"][()].astype(np.float64)
...
def reward_outcomes_per_trial(reward_times, times, trial_starts, teleports):
    outcomes = []
    for start, stop in zip(trial_starts, teleports):
        has_reward = np.any((reward_times >= times[start]) & (reward_times < times[stop]))
        outcomes.append(int(has_reward))
    return outcomes
```

iii. Reward timestamps are compared against trial time windows.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Binary: 1 if any reward timestamp falls within the trial window, 0 otherwise. Value is constant across all timepoints in the trial.

ii.
```python
has_reward = np.any((reward_times >= times[start]) & (reward_times < times[stop]))
outcomes.append(int(has_reward))
...
np.full(T, reward_outcomes[trial_idx], dtype=np.int64),
```

iii. Per-trial binary output as specified in instructions.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Lick sensor errors**: Trials where >35% of frames have lick count >2 are dropped entirely (69 trials dropped).
- **Missing reward zone**: When `reward_zone` signal is never active in a trial, the code falls back to reward timestamp position, and ultimately fills with the mode-based schedule.
- **Missing environment**: Similar fill logic using mode/switch schedule.
- **Empty trials**: Trials with T <= 0 or missing zone/env labels are dropped.
- **Position clipping**: Position is clipped to [0, 450] to handle edge cases.

ii.
```python
if drop_mask[trial_idx]:
    dropped_trials.append(trial_idx)
    continue
if T <= 0:
    dropped_trials.append(trial_idx)
    continue
if zone_label is None or env_label is None:
    dropped_trials.append(trial_idx)
    continue
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. CONVERSION_NOTES: "Trials flagged by the reference lick-error logic were dropped instead of retaining NaNs, because the decoder format expects dense arrays."

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files with `h5py` and reading large neural data arrays from disk
2. Reading the full deconvolved neural data per trial (I/O bound)
3. Saving the pickle files

ii. N/A

iii. The NWB files contain full neural recordings and are large.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates sequentially over trials. Some operations could be vectorized:
- `reward_outcomes_per_trial` loops over trials checking reward timestamps
- `drop_lick_error_trials` loops over trials computing error fractions
- `infer_trial_zone_positions` loops over trials computing zone positions
- The neural data slicing `deconv[start:stop, selected_indices]` is done per-trial rather than reading the full matrix once

ii. N/A

iii. Variable trial lengths make full vectorization awkward but some operations like reward outcome detection could use searchsorted for all trials at once.

## 13-c. What processing does the code repeat multiple times?

i. The code reads from the HDF5 file only once per session (single `with h5py.File` block), so there is no repeated file I/O. However, the neural data is read per-trial from the HDF5 dataset (`deconv[start:stop, selected_indices]`) rather than loading the full matrix into memory once, which may cause repeated disk reads.

ii.
```python
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. This is within the `with h5py.File` context, so the file handle is open, but each slice is a separate HDF5 read operation.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores extensive session metadata including observed zone positions, observed environment values, lick error fractions per trial, plane indices, and detailed schedule information. Most of this is stored in `metadata['session_info']` but is not used by the downstream decoder. The position clipping to [0, 450] may also be unnecessary processing if positions are already within range.

ii.
```python
session_info["observed_zone_positions_cm"] = observed_zone_positions
session_info["observed_zone_labels"] = observed_zone_labels
session_info["observed_environment"] = observed_env
session_info["reward_outcomes_all_trials"] = reward_outcomes
session_info["lick_error_fraction_per_trial"] = lick_error_fraction.tolist()
```

iii. This metadata is useful for debugging and documentation but not consumed by the decoder.
