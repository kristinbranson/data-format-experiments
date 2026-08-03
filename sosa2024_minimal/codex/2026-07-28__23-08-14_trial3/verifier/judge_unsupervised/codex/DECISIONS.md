# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all session files by globbing `/app/data/sub-*/sub-*_behavior+ophys.nwb`, sorting the paths, and converting each file one session at a time with `convert_session()`. The full dataset is then assembled from the per-session records with `build_dataset()`.

ii. 
```python
def nwb_files(data_dir: str):
    paths = sorted(Path(data_dir).glob("sub-*/sub-*_behavior+ophys.nwb"))
    if not paths:
        raise FileNotFoundError(f"No NWB files found under {data_dir}")
    return paths

files = nwb_files(args.data_dir)
for idx, path in enumerate(files, start=1):
    record = convert_session(path, lick_error_threshold=args.lick_error_threshold)
    session_records.append(record)

full_data = build_dataset(session_records)
```

iii. The justification in `CONVERSION_NOTES.md` is that the source materials are the NWB files under `/app/data/sub-*/sub-*_behavior+ophys.nwb`, and the converter should stay close to the paper/code while producing the decoder-ready format.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the parent directory name of each NWB file, e.g. `sub-m3` becomes `m3`. The final `subjects` list is the sorted set of those IDs, and `subject_idx` maps each session to that list.

ii. 
```python
def parse_subject_and_day(path: Path):
    subject = path.parent.name.replace("sub-", "")
    match = re.search(r"ses-(\d+)", path.name)
    return subject, int(match.group(1))

subjects = sorted({record["subject"] for record in session_records})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The notes justify this by reporting recovered mouse counts and matching the paper's statement that there are 11 switch-task mice and that `m11` starts imaging later.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session identity is effectively the file path plus the parsed `ses-XX` day number.

ii. 
```python
paths = sorted(Path(data_dir).glob("sub-*/sub-*_behavior+ophys.nwb"))

def parse_subject_and_day(path: Path):
    subject = path.parent.name.replace("sub-", "")
    match = re.search(r"ses-(\d+)", path.name)
    return subject, int(match.group(1))
```

iii. The justification is implicit in the paper/methods setup of one imaging session per day and in the notes' session-level sanity checks.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from framewise behavioral channels. A trial starts at each `trial_start > 0` frame and ends at the matching `teleport > 0` frame; frames are taken from start inclusive to teleport exclusive.

ii. 
```python
trial_start_signal = beh["trial_start"]["data"][()]
teleport_signal = beh["teleport"]["data"][()]

trial_starts = np.flatnonzero(trial_start_signal > 0)
teleports = np.flatnonzero(teleport_signal > 0)

for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    T = int(stop - start)
```

iii. `CONVERSION_NOTES.md` explicitly says the NWB files do not provide a ready-made trials table and that this reconstruction matches the reference preprocessing logic.

## 1-e. How are trials filtered based on quality controls?

i. The agent drops whole trials if more than `lick_error_threshold` of frames have `lick > 2` (default `0.35`). It also drops trials with nonpositive length, or trials whose inferred reward-zone label or environment label is missing.

ii. 
```python
def drop_lick_error_trials(lick, trial_starts, teleports, threshold):
    for idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
        frac = float(np.mean(lick[start:stop] > 2))
        if frac > threshold:
            drop_mask[idx] = True

if drop_mask[trial_idx]:
    continue
if T <= 0:
    continue
if zone_label is None or env_label is None:
    continue
```

iii. The notes say this follows the reference lick-error logic in spirit, but because the decoder format expects dense arrays, the agent chose to drop such trials instead of keeping them with `NaN` licks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` data come from `processing/ophys/Deconvolved/plane0/data`, then are restricted to ROIs referenced by `processing/ophys/Fluorescence/plane0/rois` whose segmentation rows have `iscell == True`.

ii. 
```python
deconv = f["processing/ophys/Deconvolved/plane0/data"]
rois = f["processing/ophys/Fluorescence/plane0/rois"][()]
iscell = f["processing/ophys/ImageSegmentation/PlaneSegmentation/iscell"][()][:, 0].astype(bool)
selected_mask = iscell[rois]
```

iii. The notes justify this as using the published deconvolved calcium event series and the linked ROI list rather than every segmentation row.

## 2-b. How is the `neural` data processed?

i. The agent does not recompute dF/F or deconvolution. It slices the already deconvolved framewise matrix into trials and transposes each trial to `neurons x time`.

ii. 
```python
dt_sec = float(np.median(np.diff(times)))
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
neural_trials.append(neural)
```

iii. The notes justify this by citing the paper's use of deconvolved calcium event time series at the imaging frame rate and stating that no extra smoothing or rebinning was added.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural QC implemented is an `iscell` filter applied to the linked ROI indices for `plane0`. The code does not implement the paper's additional speed-correlation-based interneuron exclusion, and it does not pool `plane1` neurons for the multi-plane mice.

ii. 
```python
rois = f["processing/ophys/Fluorescence/plane0/rois"][()]
iscell = f["processing/ophys/ImageSegmentation/PlaneSegmentation/iscell"][()][:, 0].astype(bool)
selected_mask = iscell[rois]
selected_indices = np.flatnonzero(selected_mask)
```

iii. The trajectory shows the agent considered the paper's extra interneuron exclusion but decided to skip it after a quick spot check, reasoning that it looked negligible relative to `iscell` curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to trial start by taking exactly the frames between each `trial_start` and `teleport`, then later representing time within that same frame window relative to `times[start]`.

ii. 
```python
for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    neural = deconv[start:stop, selected_indices].T.astype(np.float32)
    time_trial = (times[start:stop] - times[start]).astype(np.float32)
```

iii. The notes explicitly state that the decoder specification required alignment to trial start and that the converter uses `trial_start` inclusive to `teleport` exclusive.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the original imaging frame interval, estimated per session as the median difference between behavior timestamps and reported as a single constant dataset-level median in milliseconds. No temporal rebinning is applied.

ii. 
```python
dt_sec = float(np.median(np.diff(times)))
time_bin_sizes_ms = [1000.0 * info["time_bin_size_sec"] for info in session_info]
"time_bin_size": float(np.median(time_bin_sizes_ms))
```

iii. `CONVERSION_NOTES.md` says the recovered constant bin size is about `64.48 ms` and that no extra temporal smoothing or rebinning was added.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavior timestamps stored alongside `position`, using the same frame indices that define each trial.

ii. 
```python
times = beh["position"]["timestamps"][()].astype(np.float64)
time_trial = (times[start:stop] - times[start]).astype(np.float32)
```

iii. The justification is implicit: the NWB behavior streams are already synchronized to imaging frames, so the same timestamps can define decoder time.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the code subtracts the first trial timestamp from every timestamp in that trial window, yielding elapsed seconds since trial start.

ii. 
```python
time_trial = (times[start:stop] - times[start]).astype(np.float32)
```

iii. No separate written justification was recorded beyond the requirement to align everything to trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is taken over exactly the same `[start:stop]` frame range used for the trial's neural matrix, so it has one value per neural time bin.

ii. 
```python
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
time_trial = (times[start:stop] - times[start]).astype(np.float32)
input_trial = np.vstack([time_trial, ...])
```

iii. The decoder-alignment justification in the notes applies here as well: all streams are aligned to trial start using the same frame window.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the framewise `processing/behavior/BehavioralTimeSeries/environment/data` channel.

ii. 
```python
environment = beh["environment"]["data"][()].astype(np.float32)
env_trial = environment[start:stop]
```

iii. The notes say environment identity was inferred from the framewise `environment` channel and checked against the expected day-8 environment switch and the fact that `m17` and `m18` start in ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial, the code discards negative values, takes the rounded median of the remaining frames, then fills the full session with either a constant label or a pre/post-switch label split at trial 30.

ii. 
```python
env_trial = environment[start:stop]
env_trial = env_trial[env_trial >= 0]
observed_env.append(int(np.rint(np.nanmedian(env_trial))) if env_trial.size else None)

env_schedule = infer_constant_or_switch_schedule(observed_env)
np.full(T, env_label, dtype=np.float32)
```

iii. The written justification is that this recovers the expected environment schedule, including the day-8 switch and the special starting environment of `m17` and `m18`.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/trial number/data`.

ii. 
```python
trial_number = beh["trial number"]["data"][()].astype(np.float32)
trial_number_values = trial_number[start:stop]
```

iii. No separate textual justification was given; the choice is direct from the available synchronized behavior signal.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. For each trial, the code removes negative values, takes the median remaining value as the per-trial scalar, and repeats that scalar over all time bins in the trial. If no valid value is present, it falls back to the loop index `trial_idx`.

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

iii. The only implied justification is robustness to invalid pre-TTL values such as `-1`.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from reward delivery times in `processing/behavior/BehavioralTimeSeries/Reward/timestamps`, aggregated per trial and then shifted by one trial.

ii. 
```python
reward_times = beh["Reward"]["timestamps"][()].astype(np.float64)
reward_outcomes = reward_outcomes_per_trial(reward_times, times, trial_starts, teleports)
previous_reward_outcomes = [0] + reward_outcomes[:-1]
```

iii. The notes explicitly state that per-trial reward outcome is computed from `Reward/timestamps` within each trial window, then previous-trial outcome is defined from that sequence with the first trial set to `0`.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Each trial gets a binary `reward_outcome` equal to whether any reward timestamp falls inside that trial window; `previous_trial_outcome` is then the previous element of that list, with the first trial forced to `0`. The resulting scalar is repeated across all bins of the current trial.

ii. 
```python
def reward_outcomes_per_trial(reward_times, times, trial_starts, teleports):
    for start, stop in zip(trial_starts, teleports):
        has_reward = np.any((reward_times >= times[start]) & (reward_times < times[stop]))
        outcomes.append(int(has_reward))

previous_reward_outcomes = [0] + reward_outcomes[:-1]
np.full(T, float(previous_reward_outcomes[trial_idx]), dtype=np.float32)
```

iii. The justification is explicitly documented in `CONVERSION_NOTES.md`.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from trial-frame positions plus the trial's inferred reward-zone identity. The zone identity itself is inferred from `reward_zone`, `position`, and, when necessary, `Reward/timestamps`.

ii. 
```python
reward_zone_signal = beh["reward_zone"]["data"][()].astype(np.float32)
observed_zone_labels, observed_zone_positions = infer_trial_zone_positions(
    pos=pos, reward_zone_signal=reward_zone_signal, times=times,
    reward_times=reward_times, trial_starts=trial_starts, teleports=teleports,
)
zone_bounds_cm = ZONE_BOUNDS_CM[zone_label]
dist_trial = signed_distance_to_zone(pos_trial, zone_bounds_cm)
```

iii. The notes justify this by saying the converter had to infer per-trial reward-zone labels from the framewise `reward_zone` channel and animal position, then fill omissions with the dominant pre/post-switch labels.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The code clips position to `[0, 450]`, uses the active zone's start and end coordinates, sets distance to `0` inside the zone, negative distance before the zone start, and positive distance after the zone end.

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

iii. The justification is implicit in the decoder specification, which asked for distance relative to the reward zone rather than raw position.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is binned into the seven requested categories using hard numeric thresholds at `-50`, `-10`, `0`, `10`, and `50` cm.

ii. 
```python
out[dist_cm < -50.0] = 0
out[(dist_cm >= -50.0) & (dist_cm < -10.0)] = 1
out[(dist_cm >= -10.0) & (dist_cm < 0.0)] = 2
out[dist_cm == 0.0] = 3
out[(dist_cm > 0.0) & (dist_cm <= 10.0)] = 4
out[(dist_cm > 10.0) & (dist_cm <= 50.0)] = 5
out[dist_cm > 50.0] = 6
```

iii. This follows the decoder task specification directly; no extra justification was recorded.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from the same clipped per-frame positions over the exact `[start:stop]` trial slice used for neural activity, giving one categorical value per neural frame.

ii. 
```python
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
dist_trial = signed_distance_to_zone(pos_trial, zone_bounds_cm)
output_trial = np.vstack([bin_distance_to_zone(dist_trial), ...])
```

iii. The trial-start alignment rationale in the notes applies here.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/position/data`.

ii. 
```python
pos = beh["position"]["data"][()].astype(np.float32)
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. No special justification was recorded beyond the decoder output requirement.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The code clips each trial's position values to the track range `[0, 450)` and keeps them at the native frame rate.

ii. 
```python
def bin_absolute_position(pos_cm):
    pos_clipped = np.clip(pos_cm, 0.0, 450.0 - 1e-6)
    return np.minimum((pos_clipped / 90.0).astype(np.int64), 4)
```

iii. The clipping reflects the track bounds in the paper and avoids teleport/pre-TTL negative values.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is discretized into five equal 90 cm bins spanning the 450 cm track.

ii. 
```python
pos_clipped = np.clip(pos_cm, 0.0, 450.0 - 1e-6)
return np.minimum((pos_clipped / 90.0).astype(np.int64), 4)
```

iii. This follows the decoder task specification directly.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position uses the same trial-frame slice as the neural data, so the position category at column `t` corresponds to the neural column `t`.

ii. 
```python
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
output_trial = np.vstack([..., bin_absolute_position(pos_trial), ...])
```

iii. The justification is the same framewise trial-start alignment used throughout the converter.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/lick/data`.

ii. 
```python
lick = beh["lick"]["data"][()].astype(np.float32)
lick_trial = (lick[start:stop] > 0).astype(np.int64)
```

iii. The notes say the reference code treats corrupted lick trials specially, and the converter uses the lick stream directly after its chosen trial filtering.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The code binarizes framewise lick counts to `0/1` with the rule `lick > 0`. Trials above the lick-error threshold are dropped before this binarization is used downstream.

ii. 
```python
drop_mask, lick_error_fraction = drop_lick_error_trials(...)
lick_trial = (lick[start:stop] > 0).astype(np.int64)
```

iii. The notes justify dropping gross lick-sensor-error trials because the decoder format could not easily carry `NaN` lick values.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is sliced on the same `[start:stop]` frame interval as the neural matrix and then placed into the trial's output matrix with one value per time bin.

ii. 
```python
lick_trial = (lick[start:stop] > 0).astype(np.int64)
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
output_trial = np.vstack([..., lick_trial, ...])
```

iii. The justification is the same trial-start/framewise alignment used for all time-varying variables.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from `reward_zone`, `position`, and, if no positive reward-zone frames are present, `Reward/timestamps`, which are used to infer a per-trial zone position and then map it to `A/B/C`.

ii. 
```python
zone_frames = np.flatnonzero(zone_signal_trial > 0)
if zone_frames.size > 0:
    zone_position_cm = float(np.nanmedian(pos_trial[zone_frames]))
else:
    in_trial_reward = reward_times[(reward_times >= times[start]) & (reward_times < times[stop])]
    if in_trial_reward.size > 0:
        reward_idx = np.searchsorted(times, in_trial_reward[0], side="left")
        zone_position_cm = float(pos[reward_idx])
observed_labels.append(infer_zone_from_position(zone_position_cm))
```

iii. The notes justify this as recovering the hidden reward-zone identity from the aligned framewise signals when a direct session-level reward-zone table was not available.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The agent first infers an observed zone label per trial from median in-zone position or reward position, converts that position to the nearest of the three zone centers, then fills the session using either a constant label or a pre/post-switch schedule split at trial 30. The final categorical zone label is repeated across all time bins of the trial.

ii. 
```python
zone_schedule = infer_constant_or_switch_schedule(observed_zone_labels)
zone_label = zone_schedule["filled"][trial_idx]
np.full(T, ZONE_TO_INDEX[zone_label], dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` explicitly cites the paper's 30-trial switch boundary as the fill rule for omission trials or otherwise missing direct observations.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/timestamps` within each reconstructed trial window. This is the same computation used above when constructing previous-trial outcome.

ii. 
```python
reward_times = beh["Reward"]["timestamps"][()].astype(np.float64)
reward_outcomes = reward_outcomes_per_trial(reward_times, times, trial_starts, teleports)
```

iii. `CONVERSION_NOTES.md` explicitly records this.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The code checks for any reward timestamp between `times[start]` and `times[stop]`, converts that to `0/1`, and repeats the trial-level category across all frame bins in the output matrix.

ii. 
```python
def reward_outcomes_per_trial(...):
    has_reward = np.any((reward_times >= times[start]) & (reward_times < times[stop]))
    outcomes.append(int(has_reward))

np.full(T, reward_outcomes[trial_idx], dtype=np.int64)
```

iii. The notes justify repeating trial-level outputs across time because it is the simplest decoder-compatible representation.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles several issues heuristically. Negative environment/trial-number values are ignored before per-trial reduction; missing reward-zone observations are imputed from reward position and then from the dominant pre/post-switch schedule; trials with bad lick-sensor behavior are dropped; trials with missing zone/env labels or nonpositive length are dropped; and unmatched counts of `trial_start` and `teleport` raise an error rather than being corrected.

ii. 
```python
if trial_starts.size != teleports.size:
    raise ValueError(...)

env_trial = env_trial[env_trial >= 0]
trial_number_values = trial_number_values[trial_number_values >= 0]

if zone_frames.size == 0 and in_trial_reward.size > 0:
    reward_idx = np.searchsorted(times, in_trial_reward[0], side="left")

if frac > threshold:
    drop_mask[idx] = True
```

iii. The notes justify the main deviation here: dropping lick-error trials instead of keeping `NaN` values because the decoder format expects dense arrays.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive work is per-session I/O from large NWB files and the per-trial slicing/stacking of neural and behavioral matrices inside `convert_session()`. Zone inference and lick-error checks also loop over every trial.

ii. 
```python
with h5py.File(path, "r") as f:
    ...
    for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
        ...
        neural = deconv[start:stop, selected_indices].T.astype(np.float32)
        input_trial = np.vstack([...])
        output_trial = np.vstack([...])
```

iii. No explicit optimization rationale was recorded; the code is written for clarity and reproducibility rather than speed.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are scalar Python loops that could have been vectorized or batch-processed: `infer_trial_zone_positions()`, `reward_outcomes_per_trial()`, `drop_lick_error_trials()`, the per-trial environment reduction, and the main trial-building loop in `convert_session()`.

ii. 
```python
for start, stop in zip(trial_starts, teleports):
    ...

for idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    ...

for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    ...
```

iii. No written justification was given for keeping these as Python loops beyond implementation simplicity.

## 13-c. What processing does the code repeat multiple times?

i. The code repeatedly scans the same trial boundaries to infer zone labels, infer environment, compute reward outcomes, compute lick-error fractions, and then build trial tensors. It also repeats per-trial `np.full()` expansions for trial-level variables.

ii. 
```python
observed_zone_labels, observed_zone_positions = infer_trial_zone_positions(...)
for start, stop in zip(trial_starts, teleports):
    ...
reward_outcomes = reward_outcomes_per_trial(...)
drop_mask, lick_error_fraction = drop_lick_error_trials(...)
for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    ...
```

iii. The trajectory does not show a separate justification for this repetition; it appears to be the simplest direct implementation the agent chose.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The converter computes and stores several metadata products that the downstream decoder does not use: sample-session selection, dataset summaries, per-session schedule printouts, observed zone positions/labels, plane indices, and extensive `session_info` bookkeeping. It also creates a sample dataset even though the full decoder run only needs the full dataset.

ii. 
```python
session_info["observed_zone_positions_cm"] = observed_zone_positions
session_info["observed_zone_labels"] = observed_zone_labels
session_info["selected_plane_indices"] = sorted(np.unique(plane_idx).astype(int).tolist())

full_summary = summarize_dataset(full_data)
print_subject_schedules(full_data)

sample_records = select_sample_records(session_records)
sample_data = build_dataset(sample_records)
```

iii. The notes justify most of this as sanity checking and validation support rather than as inputs needed by downstream decoding.
