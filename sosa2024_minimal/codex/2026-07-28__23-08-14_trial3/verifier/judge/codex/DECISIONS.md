# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads every NWB file matching `sub-*/sub-*_behavior+ophys.nwb` under `/app/data`, sorts them, and processes them one file at a time with `h5py`. Each file is treated as one session and is passed into `convert_session`.

ii.
```python
def nwb_files(data_dir: str):
    paths = sorted(Path(data_dir).glob("sub-*/sub-*_behavior+ophys.nwb"))
    if not paths:
        raise FileNotFoundError(f"No NWB files found under {data_dir}")
    return paths

def main():
    args = parse_args()
    files = nwb_files(args.data_dir)
    ...
    for idx, path in enumerate(files, start=1):
        record = convert_session(path, lick_error_threshold=args.lick_error_threshold)
```

iii. In the trajectory, the AI said it had confirmed the dataset was NWB-based with one subject directory per mouse and that it would use the NWB framewise channels directly. It also justified this as the safest approach because the top-level NWB `trials` and `units` tables were absent.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by directory name. The subject ID is the parent directory name with the `sub-` prefix removed, and the dataset subject list is the sorted set of these IDs across all sessions.

ii.
```python
def parse_subject_and_day(path: Path):
    subject = path.parent.name.replace("sub-", "")
    ...
    return subject, int(match.group(1))

subjects = sorted({record["subject"] for record in session_records})
```

iii. The trajectory shows the AI inspecting `/app/data/sub-m11`, `/app/data/sub-m12`, etc., and treating these directories as the mouse split.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The session/day number is parsed from the filename component `ses-XX`.

ii.
```python
def parse_subject_and_day(path: Path):
    subject = path.parent.name.replace("sub-", "")
    match = re.search(r"ses-(\d+)", path.name)
    if match is None:
        raise ValueError(f"Could not parse session/day from {path.name}")
    return subject, int(match.group(1))
```

iii. In the trajectory, the AI repeatedly referred to each NWB file as a session and used the parsed `ses-` number as the day/session identifier.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from framewise behavior signals. Trial starts are the indices where `trial_start > 0`; trial ends are the indices where `teleport > 0`. Trial data use Python slices `[start:stop]`, so `trial_start` is included and the teleport sample itself is excluded.

ii.
```python
trial_start_signal = beh["trial_start"]["data"][()]
teleport_signal = beh["teleport"]["data"][()]

trial_starts = np.flatnonzero(trial_start_signal > 0)
teleports = np.flatnonzero(teleport_signal > 0)
...
for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    ...
    pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
    neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. The trajectory explicitly says the converter had to reconstruct trials from `trial_start` and `teleport` because the NWB trial table was absent, and that this matched the paper’s framewise trial structure.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops trials if the lick trace suggests a sensor error, defined as more than `0.35` of frames with `lick > 2`. It also drops trials with non-positive length and trials where it cannot infer a reward-zone label or environment label. It does not implement the reference solution’s short-trial `< 50` frame filter.

ii.
```python
def drop_lick_error_trials(lick, trial_starts, teleports, threshold):
    drop_mask = np.zeros(len(trial_starts), dtype=bool)
    ...
    for idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
        lick_trial = lick[start:stop]
        frac = float(np.mean(lick_trial > 2))
        ...
        if frac > threshold:
            drop_mask[idx] = True

for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    if drop_mask[trial_idx]:
        dropped_trials.append(trial_idx)
        continue
    T = int(stop - start)
    if T <= 0:
        dropped_trials.append(trial_idx)
        continue
    ...
    if zone_label is None or env_label is None:
        dropped_trials.append(trial_idx)
        continue
```

iii. The trajectory says the AI decided to “drop only trials with the published lick-sensor failure pattern,” and its notes justify dropping those trials because the decoder format could not carry NaN lick traces.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural data are taken directly from `processing/ophys/Deconvolved/plane0/data`. ROI selection is constrained using `processing/ophys/Fluorescence/plane0/rois` and `processing/ophys/ImageSegmentation/PlaneSegmentation/iscell`.

ii.
```python
deconv = f["processing/ophys/Deconvolved/plane0/data"]
rois = f["processing/ophys/Fluorescence/plane0/rois"][()]
iscell = f["processing/ophys/ImageSegmentation/PlaneSegmentation/iscell"][()][:, 0].astype(bool)
selected_mask = iscell[rois]
selected_indices = np.flatnonzero(selected_mask)
```

iii. The trajectory states that the NWB files expose deconvolved events plus fluorescence, then later concludes that for the multi-plane mice only the published `plane0` response series should be used. The AI therefore chose the stored deconvolved matrix rather than recomputing the paper’s event signal.

## 2-b. How is the `neural` data processed?

i. The AI does not recompute dF/F or deconvolution. It slices the stored deconvolved activity into trials, selects curated ROIs, transposes to `(neurons, time)`, and casts to `float32`.

ii.
```python
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
neural_trials.append(neural)
```

iii. The trajectory shows the AI debating whether to recompute the paper’s curation and signal-processing path, but it decided to keep the converter simpler and use the published `plane0` deconvolved series after checking that the stored response series matched the framewise behavior timing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural quality-control filter kept in the final code is the Suite2p `iscell` mask, restricted to ROIs referenced by the `plane0` response series. The AI does not implement the paper/reference interneuron exclusion step.

ii.
```python
iscell = f["processing/ophys/ImageSegmentation/PlaneSegmentation/iscell"][()][:, 0].astype(bool)
selected_mask = iscell[rois]
selected_indices = np.flatnonzero(selected_mask)
...
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. In the trajectory, the AI checked whether the extra putative-interneuron exclusion “materially changes the retained cell set.” Its quick test on 200 cells found zero cells above the correlation threshold, so it chose to keep only the `iscell` curation and skip the extra filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to trial start by using the same `[start:stop]` windows defined from `trial_start` and `teleport`. No additional shifting or interpolation is applied.

ii.
```python
for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    ...
    neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. The trajectory explicitly says the converter would align each trial from `trial_start` to `teleport`, and that the goal was trial-start alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native sampling interval inferred from the behavior timestamps, about `64.48 ms` per frame. No temporal rebinning or resampling is applied.

ii.
```python
dt_sec = float(np.median(np.diff(times)))
...
"time_bin_size": float(np.median(time_bin_sizes_ms)),
```

iii. The trajectory notes that the behavior timestamps had a single unique spacing across sessions and that the multi-plane sessions still had `64.48 ms` frame spacing, so the AI left the data at the native frame rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/position/timestamps`.

ii.
```python
times = beh["position"]["timestamps"][()].astype(np.float64)
...
time_trial = (times[start:stop] - times[start]).astype(np.float32)
```

iii. The trajectory did not record a separate justification for choosing `position/timestamps`; the choice appears to be driven by the fact that the position timestamps were already being used as the frame clock for the aligned behavior arrays.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the first timestamp in the trial window is subtracted so the trial starts at zero seconds.

ii.
```python
time_trial = (times[start:stop] - times[start]).astype(np.float32)
...
input_trial = np.vstack(
    [
        time_trial,
        ...
    ]
)
```

iii. The trajectory does not contain a separate discussion of this point; it is implicit in the implementation.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by taking the same trial slice `[start:stop]` that is used for neural activity.

ii.
```python
time_trial = (times[start:stop] - times[start]).astype(np.float32)
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. In the trajectory, the AI checked that the response matrix length and behavior timestamp length matched in representative sessions, including the multi-plane cases, and treated the frame indices as already aligned.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/environment/data`.

ii.
```python
environment = beh["environment"]["data"][()].astype(np.float32)
...
env_trial = environment[start:stop]
```

iii. The trajectory explicitly lists `environment` among the framewise behavior channels the AI verified in the NWB files.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI first computes a per-trial environment label as the rounded median of nonnegative environment samples in that trial. It then applies `infer_constant_or_switch_schedule(..., switch_trial=30)` to fill the whole session with either a constant label or a pre/post-30-trial switch schedule, and repeats the chosen label across all time bins in each trial.

ii.
```python
observed_env = []
for start, stop in zip(trial_starts, teleports):
    env_trial = environment[start:stop]
    env_trial = env_trial[env_trial >= 0]
    observed_env.append(int(np.rint(np.nanmedian(env_trial))) if env_trial.size else None)

env_schedule = infer_constant_or_switch_schedule(observed_env)
...
np.full(T, env_label, dtype=np.float32)
```

iii. The trajectory shows the AI validating that the recovered environment schedules produced the expected day-8 switches, and the final handoff says it used a “30-trial switch rule” for environment inference.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/trial number/data`, using only the samples inside each reconstructed trial. If no nonnegative values are available, it falls back to the loop index.

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

iii. The trajectory explicitly says the AI changed the implementation to “read the per-trial number from the recorded channel rather than assuming sequential indexing.”

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The AI takes the median of the nonnegative `trial number` values within the trial window to obtain a scalar trial number, then repeats that scalar across all time bins in the trial.

ii.
```python
trial_number_scalar = (
    float(np.nanmedian(trial_number_values))
    if trial_number_values.size
    else float(trial_idx)
)
...
np.full(T, trial_number_scalar, dtype=np.float32)
```

iii. The trajectory justifies this as using the recorded channel instead of imposing a synthetic sequential index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/Reward/timestamps` together with the reconstructed trial windows from `trial_start` and `teleport`.

ii.
```python
reward_times = beh["Reward"]["timestamps"][()].astype(np.float64)
...
reward_outcomes = reward_outcomes_per_trial(reward_times, times, trial_starts, teleports)
previous_reward_outcomes = [0] + reward_outcomes[:-1]
```

iii. The trajectory confirms that reward timestamps were available as a separate behavior channel and that per-trial reward outcomes would be computed from those windows.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI first computes the current-trial reward outcome as whether any reward timestamp falls inside each trial window. It then shifts that list by one trial, prepending `0` for the first trial, and repeats the resulting scalar across time bins within the current trial.

ii.
```python
def reward_outcomes_per_trial(reward_times, times, trial_starts, teleports):
    outcomes = []
    for start, stop in zip(trial_starts, teleports):
        has_reward = np.any((reward_times >= times[start]) & (reward_times < times[stop]))
        outcomes.append(int(has_reward))
    return outcomes

previous_reward_outcomes = [0] + reward_outcomes[:-1]
...
np.full(T, float(previous_reward_outcomes[trial_idx]), dtype=np.float32)
```

iii. The trajectory does not contain a separate extended discussion of this point; the justification is implicit in the trial-wise reward computation and shift.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from `position`, `reward_zone`, `Reward/timestamps`, and the reconstructed trial windows. The AI infers a per-trial reward-zone label from the median position of frames where `reward_zone > 0`, or from the first reward timestamp if the zone signal is absent, and then uses that zone label with position.

ii.
```python
observed_zone_labels, observed_zone_positions = infer_trial_zone_positions(
    pos=pos,
    reward_zone_signal=reward_zone_signal,
    times=times,
    reward_times=reward_times,
    trial_starts=trial_starts,
    teleports=teleports,
)
...
zone_label = zone_schedule["filled"][trial_idx]
zone_bounds_cm = ZONE_BOUNDS_CM[zone_label]
dist_trial = signed_distance_to_zone(pos_trial, zone_bounds_cm)
```

iii. The trajectory shows the AI validating “per-trial reward-zone inference,” then deciding to reconstruct per-trial zone schedules from observed frames and fill missing labels with a 30-trial switch rule.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. After inferring the trial’s reward-zone identity, the AI computes signed distance to the reward-zone edges: negative before the zone, zero inside, positive after.

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

iii. The trajectory does not separately debate this formula; the discussion focused on how to infer the reward-zone identity per trial.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is thresholded into 7 categories using manual comparisons at `-50`, `-10`, `0`, `10`, and `50` cm.

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
```

iii. The trajectory does not give an extra rationale here beyond following the task’s requested bins.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned using the same trial slice and same per-frame time axis as neural activity.

ii.
```python
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
dist_trial = signed_distance_to_zone(pos_trial, zone_bounds_cm)
```

iii. The trajectory says the converter would use the NWB framewise channels directly and align each trial from `trial_start` to `teleport`.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/position/data`.

ii.
```python
pos = beh["position"]["data"][()].astype(np.float32)
...
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. The trajectory explicitly lists `position` among the verified framewise behavior channels.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI slices the per-trial position trace and clips it to the `[0, 450]` cm track range before discretization.

ii.
```python
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. The trajectory does not include a separate written justification for clipping; it appears to be a pragmatic choice to keep positions inside the nominal track range.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is discretized into five 90 cm bins by dividing by `90.0`, converting to integers, and capping at `4`.

ii.
```python
def bin_absolute_position(pos_cm):
    pos_clipped = np.clip(pos_cm, 0.0, 450.0 - 1e-6)
    return np.minimum((pos_clipped / 90.0).astype(np.int64), 4)
```

iii. The trajectory does not discuss this separately; it follows the decoder specification of five equal bins across the 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same `[start:stop]` frame slice as the neural data for each trial.

ii.
```python
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. The trajectory justification is the same as for other framewise outputs: shared trial windows from the NWB channels.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/lick/data`.

ii.
```python
lick = beh["lick"]["data"][()].astype(np.float32)
...
lick_trial = (lick[start:stop] > 0).astype(np.int64)
```

iii. The trajectory explicitly lists `lick` among the behavior channels it verified in the NWB files.

## 9-b. What processing is involved in computing `output` *Lick*?

i. First, entire trials can be dropped by the lick-sensor error rule. For kept trials, lick is binarized so any value `> 0` becomes `1` and all other values become `0`.

ii.
```python
drop_mask, lick_error_fraction = drop_lick_error_trials(
    lick=lick,
    trial_starts=trial_starts,
    teleports=teleports,
    threshold=lick_error_threshold,
)
...
lick_trial = (lick[start:stop] > 0).astype(np.int64)
```

iii. The trajectory cites the paper code’s lick-sensor correction logic and says the converter would drop only trials with that failure pattern because the decoder format needed dense arrays.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned by taking the same frame indices `[start:stop]` used for the neural data.

ii.
```python
lick_trial = (lick[start:stop] > 0).astype(np.int64)
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. The trajectory treats lick as one of the synchronized framewise behavior channels aligned to the same trial windows.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the same inferred per-trial reward-zone identity used for distance-to-zone: `reward_zone`, `position`, `Reward/timestamps`, and the reconstructed trial boundaries.

ii.
```python
observed_zone_labels, observed_zone_positions = infer_trial_zone_positions(...)
zone_schedule = infer_constant_or_switch_schedule(observed_zone_labels)
...
np.full(T, ZONE_TO_INDEX[zone_label], dtype=np.int64)
```

iii. The trajectory shows the AI focusing on “per-trial reward-zone inference” as one of the main uncertain decisions and then using a 30-trial switch-rule fill for missing trials.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI infers observed labels from median in-zone position or reward timing, fills a constant-or-switch schedule with `infer_constant_or_switch_schedule`, converts labels `A/B/C` to `0/1/2`, and repeats that category across every time bin in the trial.

ii.
```python
zone_schedule = infer_constant_or_switch_schedule(observed_zone_labels)
...
output_trial = np.vstack(
    [
        ...,
        np.full(T, ZONE_TO_INDEX[zone_label], dtype=np.int64),
        ...
    ]
)
```

iii. The trajectory justifies this by saying the recovered zone schedules looked internally consistent and by highlighting the expected switch-session structure.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/timestamps` together with the trial windows from `trial_start` and `teleport`.

ii.
```python
reward_times = beh["Reward"]["timestamps"][()].astype(np.float64)
...
reward_outcomes = reward_outcomes_per_trial(reward_times, times, trial_starts, teleports)
```

iii. The trajectory confirms that reward timestamps were present and that per-trial reward outcomes were computed from them.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is marked rewarded if any reward timestamp falls within that trial’s `[start, stop)` time window. The resulting binary label is repeated across all time bins in the trial.

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

iii. The trajectory does not contain a separate argument here beyond the general choice to use the reward timestamps directly.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases:
- it raises an error if the counts of `trial_start` and `teleport` indices differ;
- it drops trials with lick-sensor-error fractions above threshold;
- it drops trials with non-positive duration;
- it drops trials where inferred reward-zone or environment labels are missing;
- when using reward timestamps to infer reward-zone position, it clamps the chosen frame index into the trial bounds;
- it clips position values into the track range.

It does not implement the reference solution’s neural/behavior length cropping or short-trial filter.

ii.
```python
if trial_starts.size != teleports.size:
    raise ValueError(
        f"{path.name}: trial starts ({trial_starts.size}) and teleports ({teleports.size}) do not match"
    )
...
reward_idx = np.searchsorted(times, in_trial_reward[0], side="left")
reward_idx = min(max(reward_idx, start), stop - 1)
...
if drop_mask[trial_idx]:
    ...
if T <= 0:
    ...
if zone_label is None or env_label is None:
    ...
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. The trajectory and `CONVERSION_NOTES.md` justify the lick-error handling as adapting the paper’s `correct_lick_sensor_error` logic to a dense decoder format. The rest of the handling appears to be defensive implementation rather than a separately discussed decision.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming parts are opening all 152 NWB files, reading large HDF5 arrays (`position`, `lick`, `reward_zone`, deconvolved activity, ROI metadata), iterating through every trial in every session, and serializing the final dataset. The later validator and decoder training runs were also long, but those are outside the converter itself.

ii.
```python
for idx, path in enumerate(files, start=1):
    record = convert_session(path, lick_error_threshold=args.lick_error_threshold)
    session_records.append(record)
...
with h5py.File(path, "r") as f:
    ...
    for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
        ...
save_pickle(args.full_out, full_data)
```

iii. The trajectory shows the end-to-end conversion taking long enough to require polling, which is consistent with repeated HDF5 I/O and per-trial processing. It does not include a dedicated efficiency discussion.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are naturally vectorizable but were left explicit:
- `infer_trial_zone_positions`;
- `reward_outcomes_per_trial`;
- `drop_lick_error_trials`;
- the loop that computes `observed_env`;
- the main per-trial assembly loop inside `convert_session`.

ii.
```python
for start, stop in zip(trial_starts, teleports):
    ...

for idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    ...

for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    ...
```

iii. The trajectory does not record an explicit justification for leaving these as loops. The structure suggests the AI favored straightforward per-trial logic over vectorized restructuring.

## 13-c. What processing does the code repeat multiple times?

i. Within each session, the code makes multiple passes over the same trial boundaries: once for reward-zone inference, once for environment inference, once for reward-outcome computation, once for lick-error screening, and once to build the final trial arrays. This is repeated work over the same session-level signals, even though the code does avoid rereading files for the sample dataset by reusing `session_records`.

ii.
```python
observed_zone_labels, observed_zone_positions = infer_trial_zone_positions(...)
...
for start, stop in zip(trial_starts, teleports):
    env_trial = environment[start:stop]
...
reward_outcomes = reward_outcomes_per_trial(...)
...
drop_mask, lick_error_fraction = drop_lick_error_trials(...)
...
for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    ...
```

iii. The trajectory does not explicitly justify these repeated passes. The implementation favors a sequence of simple helper functions over a more fused trial-processing pass.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The converter computes and stores a large amount of session-summary metadata that the downstream decoder does not need, including observed zone positions, observed labels, observed environments, per-trial lick-error fractions, selected plane indices, and subject schedule summaries. It also builds and saves a separate sample dataset, which is useful for validation but not used by downstream analysis of the full dataset.

ii.
```python
session_info["trial_indices_kept"] = kept_trial_indices
session_info["trial_indices_dropped"] = dropped_trials
session_info["observed_zone_positions_cm"] = observed_zone_positions
session_info["observed_zone_labels"] = observed_zone_labels
session_info["observed_environment"] = observed_env
session_info["reward_outcomes_all_trials"] = reward_outcomes
session_info["previous_reward_outcomes_all_trials"] = previous_reward_outcomes
session_info["lick_error_fraction_per_trial"] = lick_error_fraction.tolist()
...
sample_records = select_sample_records(session_records)
sample_data = build_dataset(sample_records)
save_pickle(args.sample_out, sample_data)
```

iii. The trajectory shows the AI intentionally generating these summaries to validate the conversion and document its choices. They were used for validation and notes, not for the decoder inputs themselves.
