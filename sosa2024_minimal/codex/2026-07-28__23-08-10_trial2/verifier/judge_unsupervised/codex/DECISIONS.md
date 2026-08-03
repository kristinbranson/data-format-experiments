# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads every NWB file matching `sub-*/sub-*_behavior+ophys.nwb`, extracts subject/date/scene/session metadata, sorts sessions by subject and session number, then opens each NWB with `h5py` inside `process_session`. Within each session it loads behavior streams from `processing/behavior/BehavioralTimeSeries` and imaging streams from `processing/ophys`.

ii. ```python
def list_nwb_files(data_root: Path) -> list[SessionMeta]:
    session_meta: list[SessionMeta] = []
    for file_path in sorted(data_root.glob("sub-*/sub-*_behavior+ophys.nwb")):
        with h5py.File(file_path, "r") as f:
            subject = decode_scalar(f["general/subject/subject_id"][()])
            identifier = decode_scalar(f["identifier"][()])
        ...
        session_meta.append(SessionMeta(...))
    session_meta.sort(key=lambda s: (int(s.subject[1:]), s.session_num))
    return session_meta

def process_session(session_meta: SessionMeta, block_size: int = 256) -> tuple[dict, dict]:
    with h5py.File(session_meta.file_path, "r") as f:
        behavior = f["processing/behavior/BehavioralTimeSeries"]
        ophys = f["processing/ophys"]
        imgseg = ophys["ImageSegmentation"]["PlaneSegmentation"]
```

iii. `CONVERSION_NOTES.md` says the conversion uses the supplied NWB sessions in `data/sub-*/sub-*_behavior+ophys.nwb`, and the trajectory shows the agent deliberately switched from reconstructing raw Scanbox/SQLite loading to using the released NWBs as the working input format.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the NWB field `general/subject/subject_id`. The dataset keeps a sorted unique `subjects` list and builds `subject_idx` by mapping each session’s subject string into that list.

ii. ```python
with h5py.File(file_path, "r") as f:
    subject = decode_scalar(f["general/subject/subject_id"][()])
...
subjects = sorted({meta.subject for meta in session_meta}, key=lambda s: int(s[1:]))
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
dataset["subject_idx"].append(subject_to_idx[meta.subject])
```

iii. `CONVERSION_NOTES.md` states the final full dataset contains 11 subjects, and the trajectory shows the agent checked session inventory before implementing subject bookkeeping.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session identity comes from the filename `ses-XX` plus the scene/date parsed from the NWB identifier string. Sessions are ordered by subject number and session number.

ii. ```python
identifier = decode_scalar(f["identifier"][()])
parts = identifier.strip("/").split("/")
date = parts[-2]
scene = parts[-1]
session_num = int(re.search(r"ses-(\d+)", file_path.name).group(1))
...
session_meta.sort(key=lambda s: (int(s.subject[1:]), s.session_num))
```

iii. `CONVERSION_NOTES.md` calls these out as NWB sessions and reports 152 total sessions. The agent’s trajectory also shows it inventoried the file tree up front and used one file per session.

## 1-d. How are the data split into trials?

i. Within each session, trial starts are the indices where `trial_start > 0`, and trial stops are the indices where `teleport > 0`. Every trial is then sliced as `[start:stop)`, so trial-aligned data include the trial start frame and exclude the teleport frame itself.

ii. ```python
trial_start_series = behavior["trial_start"]["data"][:]
teleport_series = behavior["teleport"]["data"][:]
...
trial_start_idx = np.where(trial_start_series > 0)[0]
trial_stop_idx = np.where(teleport_series > 0)[0]
...
kept_events[:, start:stop]
...
trial_times = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
trial_pos = np.clip(position[start:stop], 0.0, 450.0).astype(np.float32)
trial_lick = (lick_counts[start:stop] > 0).astype(np.int8)
```

iii. `CONVERSION_NOTES.md` explicitly says trials are aligned to trial start, the kept range is `[trial_start_idx : teleport_idx)`, and the teleport sample is excluded.

## 1-e. How are trials filtered based on quality controls?

i. The only explicit trial-level QC is a lick-sensor artifact rule: the agent marks a trial bad if more than 30% of the samples inside that trial have `lick count > 2`, then drops those trials from the final dataset.

ii. ```python
def bad_lick_trial_mask(lick_counts: np.ndarray, start_idx: np.ndarray, stop_idx: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(start_idx), dtype=bool)
    for trial_idx, (start, stop) in enumerate(zip(start_idx, stop_idx)):
        if np.mean(lick_counts[start:stop] > 2.0) > 0.3:
            mask[trial_idx] = True
    return mask

...
lick_error_mask = bad_lick_trial_mask(lick_counts, trial_start_idx, trial_stop_idx)
kept_trial_indices = np.where(~lick_error_mask)[0]
```

iii. `CONVERSION_NOTES.md` says this matches the paper’s lick-sensor error criterion and notes that reward outcome / previous trial outcome were computed before removing those bad trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from Suite2p fluorescence and neuropil traces in each plane, filtered to curated ROIs using `iscell`. Trial boundaries from `trial_start` and `teleport` define which samples are kept, and running speed is used only for interneuron exclusion.

ii. ```python
fluorescence_ds = ophys["Fluorescence"][plane_key]["data"]
neuropil_ds = ophys["Neuropil"][plane_key]["data"]
...
iscell = imgseg["iscell"][:]
...
fluorescence = fluorescence_ds[:session_len, block_local].T.astype(np.float32)
neuropil = neuropil_ds[:session_len, block_local].T.astype(np.float32)
...
block_corr = np.array(
    [
        np.corrcoef(trace, speed_valid)[0, 1]
        if np.nanstd(trace) > 0 and np.nanstd(speed_valid) > 0
        else np.nan
        for trace in dff_valid
    ],
    dtype=np.float32,
)
```

iii. `CONVERSION_NOTES.md` says the saved neural activity is recomputed from Suite2p fluorescence and neuropil rather than copied from NWB deconvolved output, and trajectory step 24 says the agent intentionally mirrored the repo’s dF/F and OASIS pipeline.

## 2-b. How is the `neural` data processed?

i. The agent reconstructs trial-only fluorescence arrays, subtracts neuropil with coefficient 0.7, adds back the per-trial neuropil mean before baseline normalization, computes a per-trial maximin baseline, converts to dF/F, smooths dF/F with a 2-sample Gaussian, then deconvolves with Suite2p’s OASIS. It stores deconvolved events, cast to `float16`, as the trial neural matrices.

ii. ```python
for start, stop in zip(start_idx, stop_idx):
    f[:, start:stop] = fluorescence[:, start:stop]
    f_neu[:, start:stop] = neuropil[:, start:stop]

f -= neu_coef * f_neu
...
for start, stop in zip(start_idx, stop_idx):
    sl = slice(start, stop)
    f[:, sl] = f[:, sl] + neu_coef * np.nanmean(f_neu[:, sl], axis=1, keepdims=True)
    baseline[:, sl] = nansmooth_2d(f[:, sl], sigma=15)
    baseline[:, sl] = minimum_filter1d(baseline[:, sl], size=300, axis=1, mode="nearest")
    baseline[:, sl] = maximum_filter1d(baseline[:, sl], size=300, axis=1, mode="nearest")

dff[:, valid] = (f[:, valid] - baseline[:, valid]) / np.abs(baseline[:, valid])
...
dff[:, sl] = nansmooth_2d(dff[:, sl], sigma=2)
events[:, sl] = dcnv.oasis(dff[:, sl], 2000, tau, frame_rate_hz)
...
neural_trial = np.concatenate(block_list, axis=0).astype(np.float16, copy=False)
```

iii. `CONVERSION_NOTES.md` lists this exact 7-step neural pipeline. The recovered reference `preprocessing.dff` from the trajectory uses the same maximin baseline, `neu_coef=0.7`, 2-sample smoothing, and OASIS deconvolution.

## 2-c. How is the `neural` data filtered based on quality controls?

i. First, only ROIs with `iscell[:, 0] == 1` are kept. Then putative interneurons are removed if the Pearson correlation between that ROI’s in-trial dF/F trace and running speed exceeds 0.5.

ii. ```python
plane_iscell = iscell[plane_slice, 0] == 1
curated_plane_indices = np.where(plane_iscell)[0]
...
valid_mask = np.isfinite(dff[0])
speed_valid = speed[valid_mask]
dff_valid = dff[:, valid_mask]
...
block_interneuron = np.isfinite(block_corr) & (block_corr > 0.5)
keep_block = ~block_interneuron
```

iii. `CONVERSION_NOTES.md` says ROI curation follows `iscell` and that additional putative interneurons are excluded at correlation `> 0.5`. The recovered methods text and `dayData.py` documentation in the trajectory both cite the same 0.5 threshold.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to trial start. For each kept trial, the agent slices the deconvolved session-long event array from `trial_start_idx[trial]` up to but not including `trial_stop_idx[trial]`, then stores that `(neurons, time)` matrix as the trial’s neural array.

ii. ```python
trial_start_idx = np.where(trial_start_series > 0)[0]
trial_stop_idx = np.where(teleport_series > 0)[0]
...
for out_idx, trial_idx in enumerate(kept_trial_indices):
    start = trial_start_idx[trial_idx]
    stop = trial_stop_idx[trial_idx]
    session_trial_blocks[out_idx].append(
        kept_events[:, start:stop].astype(np.float16, copy=False)
    )
```

iii. `CONVERSION_NOTES.md` explicitly names the alignment event as trial start and says teleport samples are excluded.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin is the native aligned sample spacing inferred from behavior timestamps, about 15.5078 Hz or 64.48 ms per bin. No further temporal rebinning is applied.

ii. ```python
def infer_frame_rate(position_timestamps: np.ndarray) -> float:
    diffs = np.diff(position_timestamps.astype(np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(1.0 / np.median(diffs))

...
frame_rate_hz = infer_frame_rate(timestamps)
...
mean_frame_rate = float(np.mean([stats["frame_rate_hz"] for stats in session_stats]))
dataset["metadata"]["time_bin_size"] = 1000.0 / mean_frame_rate
```

iii. `CONVERSION_NOTES.md` says the mean frame rate is `15.507813` Hz and that the resulting `metadata["time_bin_size"]` is `64.4836` ms. It also says the agent inferred this from timestamps instead of trusting NWB rate metadata.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavior timestamps attached to the `position` time series, together with the `trial_start` index that selects the first sample of each trial.

ii. ```python
timestamps = behavior["position"]["timestamps"][:].astype(np.float64)
trial_start_series = behavior["trial_start"]["data"][:]
...
trial_start_idx = np.where(trial_start_series > 0)[0]
...
trial_times = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` says `input[0]` is computed from behavior timestamps relative to the first frame of each trial.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the agent subtracts the timestamp of the first trial sample from every timestamp in that trial slice, yielding a continuous time-from-start vector in seconds.

ii. ```python
start = trial_start_idx[trial_idx]
stop = trial_stop_idx[trial_idx]
trial_times = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` describes this as “computed from the behavior timestamps relative to the first frame of each trial.”

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses the exact same `[start:stop)` trial slice as the neural data, so the input time vector has one sample per neural time bin in the trial.

ii. ```python
neural_trial = np.concatenate(block_list, axis=0).astype(np.float16, copy=False)
...
trial_times = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
...
trial_input = np.vstack(
    [
        trial_times,
        ...
    ]
)
```

iii. The notes and code both state all trial-varying inputs are built on the same trial-aligned sample range as the neural events.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The actual label is derived from the session scene string parsed from NWB metadata, not directly from the `environment` behavior time series. The `environment` behavior stream is only used as a consistency check.

ii. ```python
trial_labels = parse_scene(session_meta.scene, len(trial_start_idx))
...
trial_input = np.vstack(
    [
        trial_times,
        np.full(trial_times.shape, float(label.env), dtype=np.float32),
        ...
    ]
)

env_in_trial = env_timeseries[start:stop]
env_in_trial = env_in_trial[env_in_trial >= 0]
if env_in_trial.size:
    env_mode = int(round(float(np.median(env_in_trial))))
    if env_mode != label.env:
        raise RuntimeError(...)
```

iii. `CONVERSION_NOTES.md` says environment labels are “derived from the session scene name” and cross-checked against the aligned `environment` stream on every kept trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. `parse_scene` regex-parses scene names such as `Env1_LocationA`, `Env1_LocationA_to_B`, or `Env1_A_to_Env2_B`, converts `Env1` to `0` and `Env2` to `1`, applies a switch point at trial 30 for switch sessions, and then repeats that scalar label across every timepoint in the trial.

ii. ```python
def parse_scene(scene: str, n_trials: int, switch_trial_count: int = 30) -> list[TrialLabel]:
    match = SCENE_LOCATION_RE.match(scene)
    if match:
        env = int(match.group("env")) - 1
        ...
    match = SCENE_SWITCH_RE.match(scene)
    if match:
        env = int(match.group("env")) - 1
        ...
        return [
            TrialLabel(env=env, zone=zone0 if trial_idx < switch_trial_count else zone1)
            for trial_idx in range(n_trials)
        ]
    match = SCENE_ENV_SWITCH_RE.match(scene)
    if match:
        ...
        return [
            TrialLabel(
                env=env0 if trial_idx < switch_trial_count else env1,
                zone=zone0 if trial_idx < switch_trial_count else zone1,
            )
            for trial_idx in range(n_trials)
        ]
```

iii. `CONVERSION_NOTES.md` names the supported scene formats and says the switch point was set to trial 30 to match the paper/task structure.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the behavior time series `trial number`.

ii. ```python
trial_number_series = behavior["trial number"]["data"][:].astype(np.float32)
...
np.full(trial_times.shape, trial_number_series[start], dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` says trial number is “copied from the aligned NWB `trial number` stream at trial start.”

## 5-b. What processing is involved in computing `input` *Trial number*?

i. For each trial, the agent reads the value of `trial_number_series` at the trial’s start index and broadcasts that scalar across all timepoints in the trial input matrix.

ii. ```python
trial_input = np.vstack(
    [
        trial_times,
        np.full(trial_times.shape, float(label.env), dtype=np.float32),
        np.full(trial_times.shape, trial_number_series[start], dtype=np.float32),
        np.full(trial_times.shape, prev_reward_outcomes[trial_idx], dtype=np.float32),
    ]
)
```

iii. The notes describe it as a per-trial decoder input copied at trial start and repeated across the trial.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from reward delivery times in `behavior["Reward"]["timestamps"]`, together with the trial start and teleport times that define each trial window.

ii. ```python
reward_timestamps = behavior["Reward"]["timestamps"][:].astype(np.float64)
...
reward_outcomes = reward_outcome_per_trial(
    reward_timestamps,
    timestamps[trial_start_idx],
    timestamps[trial_stop_idx],
)
prev_reward_outcomes = np.concatenate([[0], reward_outcomes[:-1]]).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` says reward outcome is computed from reward timestamps falling between trial start and teleport, and previous-trial outcome is then formed from the actual previous trial, with the session’s first trial set to 0.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The agent first converts reward timestamps into a binary per-trial reward outcome by checking whether any reward timestamp falls between that trial’s start time and stop time. It then shifts the vector by one trial, prepends 0 for the first trial, and repeats that scalar across the time bins of each trial. This is done before lick-artifact trial removal.

ii. ```python
def reward_outcome_per_trial(reward_timestamps: np.ndarray, start_times: np.ndarray, stop_times: np.ndarray) -> np.ndarray:
    outcomes = np.zeros(len(start_times), dtype=np.int8)
    reward_idx = 0
    for trial_idx, (start_time, stop_time) in enumerate(zip(start_times, stop_times)):
        while reward_idx < len(reward_timestamps) and reward_timestamps[reward_idx] < start_time:
            reward_idx += 1
        probe_idx = reward_idx
        while probe_idx < len(reward_timestamps) and reward_timestamps[probe_idx] < stop_time:
            outcomes[trial_idx] = 1
            probe_idx += 1
    return outcomes

prev_reward_outcomes = np.concatenate([[0], reward_outcomes[:-1]]).astype(np.float32)
...
np.full(trial_times.shape, prev_reward_outcomes[trial_idx], dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` explicitly justifies the ordering: previous-trial outcome is computed before excluding bad lick trials so that a kept trial still refers to the actual immediately previous trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from per-sample position (`behavior["position"]["data"]`) plus reward-zone bounds inferred from the parsed scene label for that trial.

ii. ```python
position = behavior["position"]["data"][:].astype(np.float32)
...
label = trial_labels[trial_idx]
zone_bounds = ZONE_TO_BOUNDS_CM[label.zone]
...
discretize_distance_to_zone(trial_pos, zone_bounds)
```

iii. `CONVERSION_NOTES.md` says the active reward zone comes from the scene-derived trial label and that distance is the signed nearest distance to that active zone.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position is clipped into the 0 to 450 cm corridor. For each sample, the agent computes signed distance to the active zone: `position - zone_start` before the zone, `0` inside the zone, and `position - zone_stop` after the zone.

ii. ```python
trial_pos = np.clip(position[start:stop], 0.0, 450.0).astype(np.float32)
...
signed_distance = np.where(
    position_cm < zone_start,
    position_cm - zone_start,
    np.where(position_cm > zone_stop, position_cm - zone_stop, 0.0),
)
```

iii. The notes state these exact before-zone / in-zone / after-zone rules.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The signed distances are placed into seven categories with hard thresholds: `< -50`, `[-50, -10]`, `(-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50`.

ii. ```python
bins[signed_distance < -50.0] = 0
bins[(signed_distance >= -50.0) & (signed_distance <= -10.0)] = 1
bins[(signed_distance > -10.0) & (signed_distance < 0.0)] = 2
bins[signed_distance == 0.0] = 3
bins[(signed_distance > 0.0) & (signed_distance <= 10.0)] = 4
bins[(signed_distance > 10.0) & (signed_distance <= 50.0)] = 5
bins[signed_distance > 50.0] = 6
```

iii. `CONVERSION_NOTES.md` says this output was built to satisfy the task prompt, but the exact inclusive/exclusive edges only appear in the code.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from `trial_pos = position[start:stop]` for the same trial slice used to extract neural events, then stacked into the per-trial output matrix.

ii. ```python
start = trial_start_idx[trial_idx]
stop = trial_stop_idx[trial_idx]
trial_pos = np.clip(position[start:stop], 0.0, 450.0).astype(np.float32)
...
trial_output = np.vstack(
    [
        discretize_distance_to_zone(trial_pos, zone_bounds),
        ...
    ]
)
```

iii. The notes say all behavioral outputs are built on the aligned trial sample range `[trial_start_idx : teleport_idx)`.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from `behavior["position"]["data"]`.

ii. ```python
position = behavior["position"]["data"][:].astype(np.float32)
...
trial_pos = np.clip(position[start:stop], 0.0, 450.0).astype(np.float32)
...
discretize_absolute_position(trial_pos)
```

iii. `CONVERSION_NOTES.md` describes it as corridor position binned across the 450 cm track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The agent clips position to the corridor limits and then maps it to equal-width 90 cm bins using `floor(position / 90)`.

ii. ```python
def discretize_absolute_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, 0.0, 449.999999)
    bins = np.floor(clipped / 90.0).astype(np.int8)
    bins[bins > 4] = 4
    return bins
```

iii. `CONVERSION_NOTES.md` says the corridor is clipped to 0 to 450 cm and binned into 5 equal bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The categories are `[0, 90)`, `[90, 180)`, `[180, 270)`, `[270, 360)`, and `[360, 450]` cm, implemented by clipping to just under 450 before flooring by 90 cm and capping bin indices at 4.

ii. ```python
clipped = np.clip(position_cm, 0.0, 449.999999)
bins = np.floor(clipped / 90.0).astype(np.int8)
bins[bins > 4] = 4
```

iii. The category names are also written into `output_values` as `0_to_90cm`, `90_to_180cm`, `180_to_270cm`, `270_to_360cm`, and `360_to_450cm`.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same trial slice `[start:stop)` as the neural array and is stacked alongside the other time-varying outputs for that trial.

ii. ```python
trial_pos = np.clip(position[start:stop], 0.0, 450.0).astype(np.float32)
...
trial_output = np.vstack(
    [
        discretize_distance_to_zone(trial_pos, zone_bounds),
        discretize_absolute_position(trial_pos),
        ...
    ]
)
```

iii. This follows the same trial-start alignment described in the notes for neural and behavioral variables.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavior time series `lick`.

ii. ```python
lick_counts = behavior["lick"]["data"][:].astype(np.float32)
...
trial_lick = (lick_counts[start:stop] > 0).astype(np.int8)
```

iii. `CONVERSION_NOTES.md` says the lick output is binary `lick_count > 0`.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The agent thresholds the raw per-sample lick counts into a binary event series, assigning 1 wherever `lick_count > 0` and 0 otherwise.

ii. ```python
trial_lick = (lick_counts[start:stop] > 0).astype(np.int8)
...
trial_output = np.vstack(
    [
        ...,
        trial_lick,
        ...
    ]
)
```

iii. The notes justify this as the decoder-task-required binary lick variable.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is sliced over the same `[start:stop)` trial window used for neural data and stored as a time-varying row of the output matrix.

ii. ```python
start = trial_start_idx[trial_idx]
stop = trial_stop_idx[trial_idx]
trial_lick = (lick_counts[start:stop] > 0).astype(np.int8)
...
trial_output = np.vstack([... , trial_lick, ...])
```

iii. `CONVERSION_NOTES.md` says the behavioral outputs are all aligned to the same per-trial sample span as neural activity.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The location label is derived from the NWB session scene string, parsed into zone labels `A/B/C`; it is not read from a raw per-sample behavior variable.

ii. ```python
trial_labels = parse_scene(session_meta.scene, len(trial_start_idx))
...
label = trial_labels[trial_idx]
zone_idx = ZONE_TO_INDEX[label.zone]
```

iii. `CONVERSION_NOTES.md` says reward-zone labels are derived from the session scene name and, for switch sessions, use a switch point at trial 30.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The agent regex-parses the scene, assigns the appropriate zone label on each trial, converts `A/B/C` to `0/1/2`, and repeats that scalar across all timepoints in the trial output array.

ii. ```python
ZONE_TO_INDEX = {"A": 0, "B": 1, "C": 2}
...
label = trial_labels[trial_idx]
zone_idx = ZONE_TO_INDEX[label.zone]
...
np.full(trial_times.shape, zone_idx, dtype=np.int8)
```

iii. `CONVERSION_NOTES.md` calls this a per-trial label repeated across timepoints and says the scene parser handles stay, same-environment switch, and environment-switch sessions.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `behavior["Reward"]["timestamps"]` and trial boundaries from `trial_start` / `teleport`.

ii. ```python
reward_timestamps = behavior["Reward"]["timestamps"][:].astype(np.float64)
...
reward_outcomes = reward_outcome_per_trial(
    reward_timestamps,
    timestamps[trial_start_idx],
    timestamps[trial_stop_idx],
)
```

iii. `CONVERSION_NOTES.md` says reward outcome is determined from reward timestamps falling between trial start and teleport.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The agent marks a trial rewarded if any reward timestamp occurs between that trial’s start and stop times, otherwise omitted. It then repeats that scalar across every timepoint in the trial output matrix.

ii. ```python
while probe_idx < len(reward_timestamps) and reward_timestamps[probe_idx] < stop_time:
    outcomes[trial_idx] = 1
    probe_idx += 1
...
np.full(trial_times.shape, reward_outcomes[trial_idx], dtype=np.int8)
```

iii. `CONVERSION_NOTES.md` says `reward_outcome` is a trial-level label repeated across timepoints and coded as `0 = omitted`, `1 = rewarded`.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent uses a few ad hoc fixes rather than a general imputation pipeline. It truncates all behavior arrays, timestamps, and fluorescence planes to the shortest common session length; assigns previous-trial outcome 0 on the first trial; ignores negative environment samples when cross-checking environment; leaves non-trial fluorescence samples as NaN during dF/F/event computation; and raises an error if a kept trial ends up with no neurons.

ii. ```python
session_len = min(
    len(position),
    len(speed),
    len(lick_counts),
    len(env_timeseries),
    len(trial_number_series),
    len(trial_start_series),
    len(teleport_series),
    len(timestamps),
    *plane_lengths,
)
position = position[:session_len]
...
prev_reward_outcomes = np.concatenate([[0], reward_outcomes[:-1]]).astype(np.float32)
...
env_in_trial = env_timeseries[start:stop]
env_in_trial = env_in_trial[env_in_trial >= 0]
...
f = np.full((n_cells, n_time), np.nan, dtype=np.float32)
...
if not block_list:
    raise RuntimeError(...)
```

iii. `CONVERSION_NOTES.md` specifically justifies the common-length truncation for multi-plane sessions with one-frame mismatches, and trajectory step 101 says that fix was added after an off-by-one failure during a full run.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive parts are the per-plane / per-block neural preprocessing pass, especially dF/F reconstruction and OASIS deconvolution over all curated ROIs, followed by repeatedly slicing and concatenating per-trial event matrices across all kept trials.

ii. ```python
for plane_key in plane_keys:
    ...
    for block_start in range(0, len(curated_plane_indices), block_size):
        ...
        dff, events = compute_dff_and_events(...)
        ...
        for out_idx, trial_idx in enumerate(kept_trial_indices):
            ...
            session_trial_blocks[out_idx].append(
                kept_events[:, start:stop].astype(np.float16, copy=False)
            )
```

iii. The trajectory shows the full conversion taking a long enough time that the agent waited several minutes for the full pass, and the code structure makes the neural preprocessing the dominant cost.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are scalar or Python-list based and could be vectorized: the lick-artifact trial loop, the reward-outcome loop, the per-cell speed-correlation list comprehension, the per-trial `session_trial_blocks` append loop, and the later per-trial assembly of input/output arrays.

ii. ```python
for trial_idx, (start, stop) in enumerate(zip(start_idx, stop_idx)):
    if np.mean(lick_counts[start:stop] > 2.0) > 0.3:
        mask[trial_idx] = True

for trial_idx, (start_time, stop_time) in enumerate(zip(start_times, stop_times)):
    ...

block_corr = np.array(
    [
        np.corrcoef(trace, speed_valid)[0, 1]
        if np.nanstd(trace) > 0 and np.nanstd(speed_valid) > 0
        else np.nan
        for trace in dff_valid
    ],
    dtype=np.float32,
)

for out_idx, trial_idx in enumerate(kept_trial_indices):
    ...
```

iii. No explicit justification was given for these choices; this is just what the implementation does.

## 13-c. What processing does the code repeat multiple times?

i. The code revisits trial boundaries repeatedly. It loops over trials once to build NaN-masked fluorescence, again to compute baselines, again to smooth/deconvolve, again to apply lick QC, again to compute reward outcomes, again to append per-trial neural blocks, and again to build final per-trial input/output arrays.

ii. ```python
for start, stop in zip(start_idx, stop_idx):
    f[:, start:stop] = fluorescence[:, start:stop]
    ...

for start, stop in zip(start_idx, stop_idx):
    sl = slice(start, stop)
    ...

for start, stop in zip(start_idx, stop_idx):
    sl = slice(start, stop)
    ...

for trial_idx, (start, stop) in enumerate(zip(start_idx, stop_idx)):
    ...

for out_idx, trial_idx in enumerate(kept_trial_indices):
    ...
```

iii. The repetition is visible directly in `compute_dff_and_events`, `bad_lick_trial_mask`, `reward_outcome_per_trial`, and the session assembly loop.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes full-session dF/F traces even though only deconvolved events are saved; it loads `planeIdx` but never uses it; and it performs an environment consistency check whose result is discarded unless it throws an exception. It also accumulates several detailed stats arrays used only for reporting/metadata.

ii. ```python
plane_idx = imgseg["planeIdx"][:].astype(np.int16)
...
dff, events = compute_dff_and_events(...)
...
valid_mask = np.isfinite(dff[0])
...
env_in_trial = env_timeseries[start:stop]
env_in_trial = env_in_trial[env_in_trial >= 0]
if env_in_trial.size:
    env_mode = int(round(float(np.median(env_in_trial))))
    if env_mode != label.env:
        raise RuntimeError(...)
...
stats = {
    ...
    "trial_lengths": [int(arr.shape[1]) for arr in neural_trials],
    "plane_keys": plane_keys,
    "plane_curated_counts": curated_counts_by_plane,
    "plane_kept_counts": kept_counts_by_plane,
    ...
}
```

iii. No explicit justification was given for these inefficiencies. They appear to be pragmatic implementation choices for validation and reporting rather than values consumed by the final decoder input pipeline.
