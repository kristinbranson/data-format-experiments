# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent finds every NWB file matching `sub-*/sub-*_behavior+ophys.nwb` under `/app/data`, sorts the paths, and processes each file as one session. It opens the files directly with `h5py` and reads the needed HDF5 datasets from the behavior and ophys groups.

ii.
```python
def nwb_files(data_dir: str):
    paths = sorted(Path(data_dir).glob("sub-*/sub-*_behavior+ophys.nwb"))
    if not paths:
        raise FileNotFoundError(f"No NWB files found under {data_dir}")
    return paths

def convert_session(path: Path, lick_error_threshold: float):
    subject, day = parse_subject_and_day(path)
    with h5py.File(path, "r") as f:
        beh = f["processing/behavior/BehavioralTimeSeries"]
```

iii. The justification in the trajectory and notes is that the published NWB files are the authoritative source, all files live one directory deep under `sub-*`, and direct HDF5 access is enough to map the paper variables without reconstructing a fuller NWB object.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are inferred from the `sub-*` directory names. The subject id is taken from the parent directory of each NWB file and later deduplicated into the dataset-level `subjects` list.

ii.
```python
def parse_subject_and_day(path: Path):
    subject = path.parent.name.replace("sub-", "")
    match = re.search(r"ses-(\d+)", path.name)
    ...
    return subject, int(match.group(1))

subjects = sorted({record["subject"] for record in session_records})
```

iii. The notes treat each `sub-*` directory as one mouse and cite the recovered 11 subjects as matching the dataset described in the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The session/day number is parsed from the `ses-XX` part of the filename.

ii.
```python
def parse_subject_and_day(path: Path):
    subject = path.parent.name.replace("sub-", "")
    match = re.search(r"ses-(\d+)", path.name)
    if match is None:
        raise ValueError(f"Could not parse session/day from {path.name}")
    return subject, int(match.group(1))
```

iii. The trajectory says the agent mapped sessions directly onto NWB files because that is how the data are organized on disk and because the manuscript reports day-by-day sessions.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from framewise behavior channels. A trial starts at frames where `trial_start > 0` and ends at frames where `teleport > 0`; within each trial the agent uses slices `start:stop`, so `trial_start` is inclusive and `teleport` is exclusive.

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
```

iii. `CONVERSION_NOTES.md` says the NWB files do not provide an NWB `trials` table, so the agent intentionally reproduced trial structure from `trial_start` and `teleport` as the closest match to the paper’s preprocessing.

## 1-e. How are trials filtered based on quality controls?

i. The agent drops trials whose lick channel looks corrupted: if more than `0.35` of frames in a trial have `lick > 2`, the whole trial is removed. It also drops any trial with `T <= 0`, or with unresolved reward-zone/environment labels, but it does not apply a minimum-length filter.

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
```

iii. The notes justify this as following the paper code’s lick-sensor error logic while adapting it to the decoder format: because the decoder expects dense arrays, trials flagged by the lick-error heuristic are dropped rather than retained with invalid lick values.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural matrices come from `processing/ophys/Deconvolved/plane0/data`, with the ROI list taken from `processing/ophys/Fluorescence/plane0/rois` and cell filtering taken from `processing/ophys/ImageSegmentation/PlaneSegmentation/iscell`.

ii.
```python
deconv = f["processing/ophys/Deconvolved/plane0/data"]
rois = f["processing/ophys/Fluorescence/plane0/rois"][()]
iscell = f["processing/ophys/ImageSegmentation/PlaneSegmentation/iscell"][()][:, 0].astype(bool)
selected_mask = iscell[rois]
selected_indices = np.flatnonzero(selected_mask)
```

iii. The agent’s justification is explicit in the notes: it believed the safe published-data interpretation was to use only the response series linked to `plane0`, especially for `m17` and `m18`, rather than all segmentation rows.

## 2-b. How is the `neural` data processed?

i. The agent does no temporal reprocessing or normalization. It slices the deconvolved event matrix by trial and transposes it to neuron-by-time. The only curation step before that is ROI selection through the linked `iscell` mask.

ii.
```python
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
...
neural_trials.append(neural)
```

iii. The notes say the paper uses deconvolved calcium events at the imaging frame rate, so the agent intentionally avoided extra smoothing or rebinning.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs are filtered by the Suite2p-style `iscell` flag, but only among the ROIs referenced by `Fluorescence/plane0/rois`. All retained neurons are labeled as `CA1`.

ii.
```python
rois = f["processing/ophys/Fluorescence/plane0/rois"][()]
iscell = f["processing/ophys/ImageSegmentation/PlaneSegmentation/iscell"][()][:, 0].astype(bool)
selected_mask = iscell[rois]
...
"brain_region_idx": np.zeros(int(selected_mask.sum()), dtype=np.int64),
```

iii. The agent argues in the notes that the `iscell` mask is the key published quality-control field and that restricting to linked ROIs avoids overcounting neurons in multi-plane animals.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start by slicing the deconvolved session matrix with the same `start:stop` frame indices used for the behavioral trial window. No separate offset or interpolation is applied.

ii.
```python
for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    ...
    time_trial = (times[start:stop] - times[start]).astype(np.float32)
    neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. The notes say trial-start alignment is achieved directly by reconstructing trials from `trial_start` to `teleport`, which the agent viewed as matching the decoder instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent uses the native frame spacing already present in the NWB data. It estimates the bin size as the median difference between behavior timestamps, which it reports as about `64.48` ms, and it performs no temporal rebinning.

ii.
```python
dt_sec = float(np.median(np.diff(times)))
...
"time_bin_size": float(np.median(time_bin_sizes_ms)),
```

iii. The notes justify this by stating that the paper uses deconvolved events at the imaging rate and that all sessions recover the same bin size, so no extra resampling is needed.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavior timestamps attached to the `position` time series.

ii.
```python
times = beh["position"]["timestamps"][()].astype(np.float64)
...
time_trial = (times[start:stop] - times[start]).astype(np.float32)
```

iii. There is no longer justification than that the behavior channels are frame-aligned and the position timestamps provide the session time base used for slicing trials.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Within each trial, the first timestamp is subtracted so the time axis starts at `0` seconds and increases frame by frame.

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

iii. The implied justification is that the decoder input is explicitly “time from start of trial”, so a within-trial zeroing operation is the minimal required processing.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time vector and neural matrix use the same `start:stop` slice, so each time bin in `time_trial` corresponds to the same frame index used in the neural matrix.

ii.
```python
time_trial = (times[start:stop] - times[start]).astype(np.float32)
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. The notes and trajectory both describe the behavior and deconvolved data as already frame-aligned in NWB, so the agent did not add further alignment logic.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the framewise `environment` behavior channel.

ii.
```python
environment = beh["environment"]["data"][()].astype(np.float32)
...
env_trial = environment[start:stop]
```

iii. The notes say the paper uses two environments, and the agent treated this channel as the direct source for ENV1/ENV2 identity.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The agent takes the median environment value within each trial, rounds it to an integer, and then fills the whole trial with that value. If a session looks like a switch session, it uses a simple 30-trial pre/post mode fill.

ii.
```python
observed_env = []
for start, stop in zip(trial_starts, teleports):
    env_trial = environment[start:stop]
    env_trial = env_trial[env_trial >= 0]
    observed_env.append(int(np.rint(np.nanmedian(env_trial))) if env_trial.size else None)

env_schedule = infer_constant_or_switch_schedule(observed_env)
...
np.full(T, env_label, dtype=np.float32),
```

iii. The notes justify this as a way to recover the expected day-8 environment switch and to handle trials with incomplete direct observations while staying close to the paper’s task structure.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the framewise `trial number` behavior channel, with fallback to the trial loop index only if no nonnegative values are present in that trial window.

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

iii. The trajectory explicitly says the agent tightened the implementation to “read the per-trial number from the recorded channel rather than assuming sequential indexing.”

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The within-trial `trial number` values are reduced to one scalar by taking the median of nonnegative samples, and that scalar is repeated over all time bins in the trial.

ii.
```python
trial_number_scalar = (
    float(np.nanmedian(trial_number_values))
    if trial_number_values.size
    else float(trial_idx)
)
...
np.full(T, trial_number_scalar, dtype=np.float32),
```

iii. The justification is the decoder format: `trial_number` is a per-trial covariate, so the agent stored a constant value over the trial.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `Reward/timestamps`, after first computing whether each reconstructed trial contains any reward event.

ii.
```python
reward_times = beh["Reward"]["timestamps"][()].astype(np.float64)
...
reward_outcomes = reward_outcomes_per_trial(reward_times, times, trial_starts, teleports)
previous_reward_outcomes = [0] + reward_outcomes[:-1]
```

iii. The notes justify this as the natural per-trial interpretation of rewarded versus omitted trials in the paper’s task.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The agent computes a binary reward outcome for each trial, shifts that list by one trial, assigns `0` to the first trial, and repeats the resulting scalar across all time bins in the current trial.

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
np.full(T, float(previous_reward_outcomes[trial_idx]), dtype=np.float32),
```

iii. `CONVERSION_NOTES.md` says the first trial is assigned `0` because there is no previous trial, and the remaining trials inherit the previous trial’s rewarded/omitted status.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the animal’s `position` and an inferred per-trial reward-zone label. That label is inferred from the framewise `reward_zone` signal, the animal’s position while `reward_zone > 0`, and, if needed, the first reward timestamp in a trial.

ii.
```python
def infer_trial_zone_positions(pos, reward_zone_signal, times, reward_times, trial_starts, teleports):
    ...
    zone_frames = np.flatnonzero(zone_signal_trial > 0)
    ...
    if zone_frames.size > 0:
        zone_position_cm = float(np.nanmedian(pos_trial[zone_frames]))
    else:
        in_trial_reward = reward_times[(reward_times >= times[start]) & (reward_times < times[stop])]
        if in_trial_reward.size > 0:
            reward_idx = np.searchsorted(times, in_trial_reward[0], side="left")
            ...
            zone_position_cm = float(pos[reward_idx])
```

iii. The notes justify this as a practical way to reconstruct A/B/C reward-zone identity from the NWB behavior channels, especially on omission trials where direct reward-zone observations may be missing.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. After assigning a reward-zone label to the trial, the agent computes signed distance to the nearest edge of that zone. Distances are negative before the zone, zero inside the zone, and positive after the zone.

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

iii. The notes say this was intended to match the paper’s reward-relative coding by expressing location relative to the active 50 cm reward zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The signed distance is converted into seven bins matching the decoder instructions: `< -50`, `[-50, -10)`, `[-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50`.

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

iii. The justification is simply that these thresholds were required by the decoder task, so the agent implemented them directly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by computing the distance from `pos[start:stop]`, the same trial slice used to build `deconv[start:stop, ...]`.

ii.
```python
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
dist_trial = signed_distance_to_zone(pos_trial, zone_bounds_cm)
```

iii. The notes repeatedly describe the decoder outputs as frame-aligned to the trial-start-sliced neural data, so no extra temporal offset is introduced here.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from the `position` behavior channel.

ii.
```python
pos = beh["position"]["data"][()].astype(np.float32)
...
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. The justification is direct: this channel records the animal’s position on the 450 cm virtual track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Before binning, the agent clips position to the interval `[0, 450)` and then bins those values. It therefore discards negative positions rather than preserving them as a separate low-position range.

ii.
```python
def bin_absolute_position(pos_cm):
    pos_clipped = np.clip(pos_cm, 0.0, 450.0 - 1e-6)
    return np.minimum((pos_clipped / 90.0).astype(np.int64), 4)
```

iii. There is no detailed written justification beyond keeping the track range at `0-450 cm` in metadata and using equal-sized bins over that range.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The agent uses five 90 cm bins spanning `0-450 cm`: `0-<90`, `90-<180`, `180-<270`, `270-<360`, and `360-450`.

ii.
```python
def bin_absolute_position(pos_cm):
    pos_clipped = np.clip(pos_cm, 0.0, 450.0 - 1e-6)
    return np.minimum((pos_clipped / 90.0).astype(np.int64), 4)

OUTPUT_VALUES = [
    ...
    [
        "0 to <90 cm",
        "90 to <180 cm",
        "180 to <270 cm",
        "270 to <360 cm",
        "360 to 450 cm",
    ],
```

iii. The implied justification is that the instructions asked for “5 equal-sized bins,” and the agent interpreted the corridor as running from `0` to `450 cm`.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned by slicing `position[start:stop]` with the same indices used to slice neural activity for that trial.

ii.
```python
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. The agent’s overall alignment rule is trial-wise frame alignment from `trial_start` to `teleport`, so position inherits that same alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the framewise `lick` behavior channel.

ii.
```python
lick = beh["lick"]["data"][()].astype(np.float32)
...
lick_trial = (lick[start:stop] > 0).astype(np.int64)
```

iii. The notes frame lick as one of the behavioral decoder outputs and also use it for the lick-sensor quality-control heuristic.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick counts are binarized: any frame with `lick > 0` becomes `1`, otherwise `0`.

ii.
```python
lick_trial = (lick[start:stop] > 0).astype(np.int64)
...
output_trial = np.vstack(
    [
        ...
        lick_trial,
        ...
    ]
)
```

iii. The decoder task requires a binary lick output, so the agent applied a simple positive/nonpositive threshold.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned by using the same `start:stop` slice as the neural data for each trial.

ii.
```python
lick_trial = (lick[start:stop] > 0).astype(np.int64)
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. The notes treat lick as one of the frame-aligned behavioral channels already synchronized to the imaging frames.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the framewise `reward_zone` signal, the animal’s `position`, and, on some omission trials, `Reward/timestamps` to infer the likely zone position. The final output is a trial-level A/B/C label.

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
np.full(T, ZONE_TO_INDEX[zone_label], dtype=np.int64),
```

iii. The notes justify this as recovering the hidden reward-zone schedule specified in the paper from the published behavioral channels.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The agent first maps within-trial reward-zone observations to a zone position, converts that to the nearest zone label A/B/C, then applies a simple session-level fill rule: if the dominant label before trial 30 differs from the dominant label after trial 30, the session is treated as a switch session at trial 30; otherwise one constant label is used for the whole session.

ii.
```python
def infer_constant_or_switch_schedule(observed_values, switch_trial=30):
    ...
    is_switch = pre is not None and post is not None and pre != post
    if is_switch:
        for trial_idx in range(ntrials):
            filled.append(pre if trial_idx < switch_trial else post)
    else:
        filled = [overall for _ in range(ntrials)]

zone_schedule = infer_constant_or_switch_schedule(observed_zone_labels)
```

iii. `CONVERSION_NOTES.md` explicitly justifies this as using the paper’s 30-trial switch boundary to fill omission trials and recover expected switch schedules.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/timestamps`.

ii.
```python
reward_times = beh["Reward"]["timestamps"][()].astype(np.float64)
reward_outcomes = reward_outcomes_per_trial(reward_times, times, trial_starts, teleports)
```

iii. The notes treat reward outcome as the per-trial rewarded-versus-omitted label implied by reward deliveries in the task.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each reconstructed trial window, the agent checks whether any reward timestamp falls between `times[start]` and `times[stop]`. The result is a binary trial-level value repeated over all time bins in the trial.

ii.
```python
def reward_outcomes_per_trial(reward_times, times, trial_starts, teleports):
    outcomes = []
    for start, stop in zip(trial_starts, teleports):
        has_reward = np.any((reward_times >= times[start]) & (reward_times < times[stop]))
        outcomes.append(int(has_reward))
    return outcomes

np.full(T, reward_outcomes[trial_idx], dtype=np.int64),
```

iii. The notes justify this as matching the rewarded/omitted distinction used for `previous_trial_outcome` and the decoder output.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles several edge cases, but in a different style from the reference. Missing direct reward-zone observations can be filled from reward timestamps or from the dominant pre/post-switch zone label. Environment identity is similarly filled from per-trial medians and the same 30-trial rule. Trials with excessive lick counts are dropped. Trials with no valid labels or nonpositive length are also dropped. There is no explicit neural/behavior length-cropping safeguard and no minimum-length trial filter.

ii.
```python
if zone_frames.size > 0:
    zone_position_cm = float(np.nanmedian(pos_trial[zone_frames]))
else:
    in_trial_reward = reward_times[(reward_times >= times[start]) & (reward_times < times[stop])]
    if in_trial_reward.size > 0:
        ...
        zone_position_cm = float(pos[reward_idx])

drop_mask, lick_error_fraction = drop_lick_error_trials(...)
...
if drop_mask[trial_idx]:
    dropped_trials.append(trial_idx)
    continue
...
if zone_label is None or env_label is None:
    dropped_trials.append(trial_idx)
    continue
```

iii. The notes justify these choices as pragmatic adaptations to the decoder format, especially for omission trials and lick-sensor errors, but they do not document additional mismatch-handling beyond those cases.

## 13-a. What are the most time-consuming steps of the code?

i. The code spends most of its time opening all 152 NWB files with `h5py`, reading large behavior and deconvolved arrays, iterating through every trial to build `neural`, `input`, and `output`, and then serializing the full and sample datasets.

ii.
```python
files = nwb_files(args.data_dir)
...
for idx, path in enumerate(files, start=1):
    record = convert_session(path, lick_error_threshold=args.lick_error_threshold)
    session_records.append(record)
...
save_pickle(args.full_out, full_data)
save_pickle(args.sample_out, sample_data)
```

iii. This is not spelled out in the notes, but it follows directly from the implementation structure and from the trajectory, which treats the full conversion and validation runs as the longest steps.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious candidates are the repeated per-trial loops in `infer_trial_zone_positions`, `reward_outcomes_per_trial`, `drop_lick_error_trials`, and the main `for trial_idx` conversion loop. The switch-schedule fill loop could also be replaced by vectorized array construction.

ii.
```python
for start, stop in zip(trial_starts, teleports):
    ...

for idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    ...

for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    ...
```

iii. The agent did not explicitly discuss vectorization, but these loops are the main pure-Python hotspots in the converter.

## 13-c. What processing does the code repeat multiple times?

i. The code traverses the trial boundaries several separate times in one session: once to infer reward-zone observations, once to summarize environment, once to compute reward outcomes, once to compute lick-error fractions, and once more to actually build converted trials. It also computes dataset summaries twice, once for the full set and once for the sample subset.

ii.
```python
observed_zone_labels, observed_zone_positions = infer_trial_zone_positions(...)
...
for start, stop in zip(trial_starts, teleports):
    env_trial = environment[start:stop]
...
reward_outcomes = reward_outcomes_per_trial(reward_times, times, trial_starts, teleports)
drop_mask, lick_error_fraction = drop_lick_error_trials(...)
...
for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    ...
full_summary = summarize_dataset(full_data)
sample_summary = summarize_dataset(sample_data)
```

iii. There is no explicit written justification for this repetition; it appears to be a straightforward decomposition into helper functions rather than an optimized pass structure.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code records substantial metadata that the decoder does not use directly, including observed zone positions, observed zone labels, observed environment values, per-trial lick-error fractions, linked plane indices, and printed subject schedule summaries. It also builds and summarizes a sample dataset in addition to the full dataset.

ii.
```python
session_info["observed_zone_positions_cm"] = observed_zone_positions
session_info["observed_zone_labels"] = observed_zone_labels
session_info["observed_environment"] = observed_env
session_info["lick_error_fraction_per_trial"] = lick_error_fraction.tolist()
session_info["selected_plane_indices"] = sorted(np.unique(plane_idx).astype(int).tolist())
...
sample_records = select_sample_records(session_records)
sample_data = build_dataset(sample_records)
sample_summary = summarize_dataset(sample_data)
```

iii. The notes justify some of this as sanity checking and documentation, but these computations are not needed for the downstream decoder itself.
