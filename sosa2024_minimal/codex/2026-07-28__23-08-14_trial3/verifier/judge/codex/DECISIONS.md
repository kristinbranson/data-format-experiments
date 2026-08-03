# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all NWB files matching `sub-*/sub-*_behavior+ophys.nwb` under the data directory, sorts them, and converts each file as one session. It reads the raw NWB contents directly with `h5py` from the HDF5 paths rather than using `pynwb`.

ii. 
```python
def nwb_files(data_dir: str):
    paths = sorted(Path(data_dir).glob("sub-*/sub-*_behavior+ophys.nwb"))
    if not paths:
        raise FileNotFoundError(f"No NWB files found under {data_dir}")
    return paths
...
for idx, path in enumerate(files, start=1):
    record = convert_session(path, lick_error_threshold=args.lick_error_threshold)
```
```python
def convert_session(path: Path, lick_error_threshold: float):
    subject, day = parse_subject_and_day(path)
    with h5py.File(path, "r") as f:
        beh = f["processing/behavior/BehavioralTimeSeries"]
```

iii. `CONVERSION_NOTES.md` says the source data are `/app/data/sub-*/sub-*_behavior+ophys.nwb` and that the converter kept the loading/alignment logic close to the published data layout. In trajectory step 30, the agent notes that NWB `trials`/`units` tables are absent and that it therefore needs to map directly from the framewise NWB channels.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the parent directory name `sub-<mouse>` and then deduplicated/sorted when the final dataset is built.

ii. 
```python
def parse_subject_and_day(path: Path):
    subject = path.parent.name.replace("sub-", "")
    match = re.search(r"ses-(\d+)", path.name)
    ...
    return subject, int(match.group(1))
```
```python
subjects = sorted({record["subject"] for record in session_records})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The notes justify this by the recovered dataset structure: 11 unique subjects and the expected session counts per mouse. The file naming convention is treated as authoritative.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session/day, parsed from the `ses-##` token in the filename.

ii. 
```python
def parse_subject_and_day(path: Path):
    subject = path.parent.name.replace("sub-", "")
    match = re.search(r"ses-(\d+)", path.name)
    ...
    return subject, int(match.group(1))
```
```python
for idx, path in enumerate(files, start=1):
    record = convert_session(path, lick_error_threshold=args.lick_error_threshold)
    session_records.append(record)
```

iii. `CONVERSION_NOTES.md` reports 152 recovered sessions and the expected reduced count for `m11`, so the agent used filename/session identity as the session split.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from the framewise behavior channels. The agent takes all indices with `trial_start > 0` as trial starts, all indices with `teleport > 0` as trial ends, and slices each trial as frames `start:stop`, so the teleport frame itself is excluded.

ii. 
```python
trial_start_signal = beh["trial_start"]["data"][()]
teleport_signal = beh["teleport"]["data"][()]

trial_starts = np.flatnonzero(trial_start_signal > 0)
teleports = np.flatnonzero(teleport_signal > 0)
if trial_starts.size != teleports.size:
    raise ValueError(
        f"{path.name}: trial starts ({trial_starts.size}) and teleports ({teleports.size}) do not match"
    )
...
for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    ...
    T = int(stop - start)
```

iii. The notes say this matches the reference preprocessing idea of using `trial_start` to `teleport` and excluding the teleport sample. Trajectory step 30 makes the same justification: the NWB files do not expose a trials table, so trial structure must be reconstructed from those channels.

## 1-e. How are trials filtered based on quality controls?

i. The main quality-control filter is dropping trials with suspected lick-sensor failure: more than `0.35` of frames in the trial have `lick > 2`. The code also drops zero-length trials and any trial whose inferred reward-zone label or environment label is missing. It does not implement the human reference's `<50`-timepoint minimum-length filter.

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
    return drop_mask, error_fraction
```
```python
for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    if drop_mask[trial_idx]:
        dropped_trials.append(trial_idx)
        continue

    T = int(stop - start)
    if T <= 0:
        dropped_trials.append(trial_idx)
        continue

    zone_label = zone_schedule["filled"][trial_idx]
    env_label = env_schedule["filled"][trial_idx]
    if zone_label is None or env_label is None:
        dropped_trials.append(trial_idx)
        continue
```

iii. `CONVERSION_NOTES.md` explicitly cites the paper code's `correct_lick_sensor_error` logic and says the decoder format could not keep NaN licks while preserving dense arrays, so the agent chose to drop those trials entirely. Trajectory step 71 describes this as the only intentional trial-quality filter beyond structural validity.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The agent derives `neural` from the deconvolved calcium-event matrix `processing/ophys/Deconvolved/plane0/data`, using ROI linkage information from `processing/ophys/Fluorescence/plane0/rois` and ROI metadata from `processing/ophys/ImageSegmentation/PlaneSegmentation`.

ii. 
```python
deconv = f["processing/ophys/Deconvolved/plane0/data"]
rois = f["processing/ophys/Fluorescence/plane0/rois"][()]
iscell = f["processing/ophys/ImageSegmentation/PlaneSegmentation/iscell"][()][:, 0].astype(bool)
selected_mask = iscell[rois]
selected_indices = np.flatnonzero(selected_mask)
selected_rois = rois[selected_mask]
plane_idx = f["processing/ophys/ImageSegmentation/PlaneSegmentation/planeIdx"][()][selected_rois]
```

iii. The notes say the paper used deconvolved calcium event time series and that, for the published NWB, the "safest published-data interpretation" was to use the linked `plane0` response series plus its linked ROI list, especially for `m17` and `m18`.

## 2-b. How is the `neural` data processed?

i. The agent keeps the stored deconvolved values, selects only the linked `iscell` ROIs, slices each trial by frame index, and transposes to `(n_neurons, n_timepoints)`. It does not smooth, normalize, or rebin the neural data.

ii. 
```python
selected_mask = iscell[rois]
selected_indices = np.flatnonzero(selected_mask)
...
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
...
neural_trials.append(neural)
```

iii. `CONVERSION_NOTES.md` says "No extra temporal smoothing or rebinning was added" and that the recovered neural bin size was taken as the published imaging-frame sampling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural QC implemented is filtering ROIs by the boolean `iscell` flag for the ROIs linked to the chosen response series.

ii. 
```python
rois = f["processing/ophys/Fluorescence/plane0/rois"][()]
iscell = f["processing/ophys/ImageSegmentation/PlaneSegmentation/iscell"][()][:, 0].astype(bool)
selected_mask = iscell[rois]
selected_indices = np.flatnonzero(selected_mask)
```

iii. The notes justify this as matching the published segmentation metadata and say it brought neuron counts closer to the paper's reported range. Trajectory steps 44 and 50 show the agent explicitly checking whether additional exclusions were necessary and deciding they were not.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start implicitly by trial slicing. For each reconstructed trial, the agent uses the same `start:stop` frame window for neural and behavior, so each neural trial begins at the trial-start frame.

ii. 
```python
for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    ...
    time_trial = (times[start:stop] - times[start]).astype(np.float32)
    neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. The notes say the decoder specification required alignment to trial start and that reconstructing trials from `trial_start` to `teleport` satisfied that requirement. No extra offsetting of neural samples was added.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent uses one stored imaging/behavior frame per output time bin, with no temporal rebinning. The bin size is estimated from the median difference of the behavior timestamps and stored in milliseconds in the metadata.

ii. 
```python
times = beh["position"]["timestamps"][()].astype(np.float64)
...
dt_sec = float(np.median(np.diff(times)))
```
```python
"metadata": {
    ...
    "time_bin_size": float(np.median(time_bin_sizes_ms)),
    ...
}
```

iii. `CONVERSION_NOTES.md` says the paper used deconvolved calcium event time series at about 15.5 Hz and that the recovered bin size was a constant `64.48362720402656` ms, with no added rebinning.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavior timestamps stored alongside the position channel, `processing/behavior/BehavioralTimeSeries/position/timestamps`.

ii. 
```python
times = beh["position"]["timestamps"][()].astype(np.float64)
...
time_trial = (times[start:stop] - times[start]).astype(np.float32)
```

iii. The agent did not separately justify choosing `position` timestamps over another behavior stream. The implied justification, from the notes and trajectory, is that the framewise behavior channels are synchronized and any one of their timestamps can provide trial-relative time.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the code subtracts the first timestamp in that trial so time starts at zero.

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

iii. The justification is implicit: the decoder input is "time from start of trial", and this subtraction is the direct way to implement that alignment.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by using exactly the same `start:stop` frame window as the neural slice for each trial.

ii. 
```python
time_trial = (times[start:stop] - times[start]).astype(np.float32)
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. The agent's notes and trajectory assume the framewise behavior and deconvolved signals are synchronized. Unlike the human reference, the agent did not add explicit timestamp-equality checks or cropping logic here.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the framewise `environment` behavior channel.

ii. 
```python
environment = beh["environment"]["data"][()].astype(np.float32)
...
for start, stop in zip(trial_starts, teleports):
    env_trial = environment[start:stop]
    env_trial = env_trial[env_trial >= 0]
    observed_env.append(int(np.rint(np.nanmedian(env_trial))) if env_trial.size else None)
```

iii. `CONVERSION_NOTES.md` says environment identity was inferred from the framewise `environment` channel and that this recovered the expected day-8 environment switches and the starting environment for `m17` and `m18`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The agent first compresses each trial's environment stream to a single trial label using the median of valid values, then forces the session to be either constant or a single switch at trial 30 via `infer_constant_or_switch_schedule`. The resulting label is repeated across every time bin of the trial.

ii. 
```python
def infer_constant_or_switch_schedule(observed_values, switch_trial=30):
    ...
    is_switch = pre is not None and post is not None and pre != post
    ...
    if is_switch:
        for trial_idx in range(ntrials):
            filled.append(pre if trial_idx < switch_trial else post)
    else:
        filled = [overall for _ in range(ntrials)]
```
```python
env_schedule = infer_constant_or_switch_schedule(observed_env)
...
np.full(T, env_label, dtype=np.float32)
```

iii. The notes explicitly justify this with a "30-trial switch rule when needed" and describe it as a way to recover the known environment-switch structure from the paper.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the raw `trial number` behavior channel within each reconstructed trial, with fallback to the loop index if no valid values exist.

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

iii. Trajectory step 76 gives the key justification: the agent deliberately changed the implementation to "read the per-trial number from the recorded channel rather than assuming sequential indexing."

## 5-b. What processing is involved in computing `input` *Trial number*?

i. Within each trial, it takes the median of the nonnegative `trial number` values and repeats that scalar across all time bins in the trial. If the channel is empty for a trial, it falls back to the Python loop index.

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
np.full(T, trial_number_scalar, dtype=np.float32)
```

iii. The only explicit justification is trajectory step 76, where the agent says it wants to use the recorded channel instead of assuming a sequential counter.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the reward-event timestamps in `processing/behavior/BehavioralTimeSeries/Reward/timestamps`.

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

iii. `CONVERSION_NOTES.md` states that per-trial reward outcome is computed from `Reward/timestamps` inside each reconstructed trial window.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The agent computes reward outcome for every trial, shifts that list by one position, and sets the first trial's previous outcome to `0`. The previous-outcome value is then repeated across every time bin of the current trial.

ii. 
```python
reward_outcomes = reward_outcomes_per_trial(reward_times, times, trial_starts, teleports)
previous_reward_outcomes = [0] + reward_outcomes[:-1]
...
np.full(T, float(previous_reward_outcomes[trial_idx]), dtype=np.float32)
```

iii. The notes justify this directly: "`previous_trial_outcome` is then the previous trial's reward outcome, with the first trial in a session assigned `0`."

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from per-trial position plus an inferred trial-level reward-zone identity. That reward-zone identity is inferred from the framewise `reward_zone` channel and, if needed, from reward timestamps combined with position.

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
```python
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
zone_bounds_cm = ZONE_BOUNDS_CM[zone_label]
dist_trial = signed_distance_to_zone(pos_trial, zone_bounds_cm)
```

iii. The notes say the converter infers reward-zone labels from `reward_zone` plus position and fills omission-trial labels using the dominant pre/post-switch identity.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Once a zone label is assigned to the trial, the code computes signed distance to the nearest edge of that zone: negative before the zone, zero inside the zone, positive after the zone.

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

iii. The notes describe this as preserving the paper's reward-relative geometry while adapting it to the requested decoder outputs.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is binned into seven categories with explicit threshold logic matching the decoder-task bins.

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

iii. The justification is the decoder specification itself; the code implements those requested bins directly rather than keeping a continuous distance value.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by computing distance from `pos[start:stop]` while neural data use `deconv[start:stop, ...]`, so both use the same frame indices within each trial.

ii. 
```python
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
...
dist_trial = signed_distance_to_zone(pos_trial, zone_bounds_cm)
```

iii. The notes and trajectory both treat the published behavior and imaging streams as frame-aligned once trial windows are reconstructed.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the framewise `position` behavior channel.

ii. 
```python
pos = beh["position"]["data"][()].astype(np.float32)
...
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
```

iii. The agent did not separately justify this choice because `position` is the direct raw measurement needed for this decoder output.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The agent clips per-trial position to `[0, 450)` cm and then converts it to five 90-cm-wide bins by computing `floor(position / 90)`, capped at 4.

ii. 
```python
def bin_absolute_position(pos_cm):
    pos_clipped = np.clip(pos_cm, 0.0, 450.0 - 1e-6)
    return np.minimum((pos_clipped / 90.0).astype(np.int64), 4)
```
```python
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
...
bin_absolute_position(pos_trial)
```

iii. The notes encode the assumed corridor range as `position_range_cm: [0.0, 450.0]`, so the agent appears to have chosen five equal-width bins across that span to follow the decoder instructions literally.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The categories are `0-<90`, `90-<180`, `180-<270`, `270-<360`, and `360-450` cm, implemented by integer division after clipping.

ii. 
```python
OUTPUT_VALUES = [
    ...,
    [
        "0 to <90 cm",
        "90 to <180 cm",
        "180 to <270 cm",
        "270 to <360 cm",
        "360 to 450 cm",
    ],
    ...
]
```
```python
def bin_absolute_position(pos_cm):
    pos_clipped = np.clip(pos_cm, 0.0, 450.0 - 1e-6)
    return np.minimum((pos_clipped / 90.0).astype(np.int64), 4)
```

iii. The apparent justification is again the decoder request for five equal-sized bins across the corridor.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned by slicing `position` with the same `start:stop` frame window used for the neural trial.

ii. 
```python
pos_trial = np.clip(pos[start:stop], 0.0, 450.0).astype(np.float32)
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. The notes assume framewise synchronization between behavior and neural streams after trial reconstruction.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the framewise `lick` behavior channel.

ii. 
```python
lick = beh["lick"]["data"][()].astype(np.float32)
...
lick_trial = (lick[start:stop] > 0).astype(np.int64)
```

iii. The notes discuss the lick channel both as the source for the binary decoder output and as the source of a trial-quality filter for lick-sensor artifacts.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The lick output is binarized by thresholding the raw lick counts at `> 0`. Separately, trials with severe lick-sensor artifacts (`> 35%` of frames above 2) are dropped before conversion.

ii. 
```python
lick_trial = (lick[start:stop] > 0).astype(np.int64)
```
```python
frac = float(np.mean(lick_trial > 2))
...
if frac > threshold:
    drop_mask[idx] = True
```

iii. `CONVERSION_NOTES.md` says the decoder wants a binary lick output and that bad lick-sensor trials were dropped because dense decoder-ready arrays cannot preserve those trials with NaNs.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned by slicing the lick array with the same `start:stop` frame indices used for the neural trial.

ii. 
```python
lick_trial = (lick[start:stop] > 0).astype(np.int64)
neural = deconv[start:stop, selected_indices].T.astype(np.float32)
```

iii. The agent's approach relies on the published framewise synchronization of behavior and neural samples.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the framewise `reward_zone` behavior channel plus position, with reward timestamps used as a fallback when a trial has no positive `reward_zone` frames.

ii. 
```python
def infer_trial_zone_positions(pos, reward_zone_signal, times, reward_times, trial_starts, teleports):
    ...
    zone_signal_trial = reward_zone_signal[start:stop]
    zone_frames = np.flatnonzero(zone_signal_trial > 0)
    ...
    in_trial_reward = reward_times[(reward_times >= times[start]) & (reward_times < times[stop])]
```

iii. The notes explicitly say reward-zone identity is inferred from `reward_zone` plus position and that omission-trial labels are filled from the session schedule.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For each trial, the agent estimates a zone position from the median position of `reward_zone > 0` frames, maps that position to the nearest canonical zone center (`A`, `B`, or `C`), and if needed falls back to the position at the first reward timestamp. It then enforces a session schedule that is either constant or a single switch at trial 30, and repeats the resulting zone index across all time bins in the trial.

ii. 
```python
def infer_zone_from_position(zone_position_cm):
    if not np.isfinite(zone_position_cm):
        return None
    return min(
        ZONE_CENTERS_CM,
        key=lambda label: abs(zone_position_cm - ZONE_CENTERS_CM[label]),
    )
```
```python
zone_schedule = infer_constant_or_switch_schedule(observed_zone_labels)
...
np.full(T, ZONE_TO_INDEX[zone_label], dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` gives the direct rationale: use the framewise observations when available and fill omission trials using the dominant pre-switch and post-switch zone labels with the paper's 30-trial switch boundary.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/Reward/timestamps`.

ii. 
```python
reward_times = beh["Reward"]["timestamps"][()].astype(np.float64)
...
reward_outcomes = reward_outcomes_per_trial(reward_times, times, trial_starts, teleports)
```

iii. The notes say reward outcome is computed from `Reward/timestamps` within each reconstructed trial.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is labeled rewarded if any reward timestamp falls inside that trial's `[start, stop)` time window. The result is then repeated across all time bins of that trial in the output array.

ii. 
```python
def reward_outcomes_per_trial(reward_times, times, trial_starts, teleports):
    outcomes = []
    for start, stop in zip(trial_starts, teleports):
        has_reward = np.any((reward_times >= times[start]) & (reward_times < times[stop]))
        outcomes.append(int(has_reward))
    return outcomes
```
```python
np.full(T, reward_outcomes[trial_idx], dtype=np.int64)
```

iii. The notes say trial-level outputs like reward outcome were repeated across time bins because that is the simplest shape compatible with the decoder format.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles missing or suspect data mainly by dropping trials or filling schedule labels. It drops lick-error trials, zero-length trials, and trials whose inferred zone/environment label is missing. For omission trials with no `reward_zone` frames, it tries reward-time/position fallback and then fills remaining unknowns from a per-session pre/post-switch schedule. The code does not add the human reference's explicit neural/behavior length-cropping logic or reward-time alignment assertion.

ii. 
```python
if zone_frames.size > 0:
    zone_position_cm = float(np.nanmedian(pos_trial[zone_frames]))
else:
    in_trial_reward = reward_times[(reward_times >= times[start]) & (reward_times < times[stop])]
    if in_trial_reward.size > 0:
        reward_idx = np.searchsorted(times, in_trial_reward[0], side="left")
        ...
        zone_position_cm = float(pos[reward_idx])
```
```python
if drop_mask[trial_idx]:
    dropped_trials.append(trial_idx)
    continue
...
if zone_label is None or env_label is None:
    dropped_trials.append(trial_idx)
    continue
```

iii. The notes justify lick-error dropping as necessary for dense decoder arrays, and they justify schedule filling as a way to recover omission-trial labels consistent with the paper's switch structure. The handling of time-length mismatches is not discussed, implying the agent assumed the stored frame indexing was adequate.

## 13-a. What are the most time-consuming steps of the code?

i. The likely slowest steps are opening all 152 NWB/HDF5 files, reading the large behavior and deconvolved arrays from each file, iterating over every trial to build per-trial arrays, and writing the full pickle outputs.

ii. 
```python
for idx, path in enumerate(files, start=1):
    record = convert_session(path, lick_error_threshold=args.lick_error_threshold)
```
```python
with h5py.File(path, "r") as f:
    ...
    times = beh["position"]["timestamps"][()].astype(np.float64)
    pos = beh["position"]["data"][()].astype(np.float32)
    speed = beh["speed"]["data"][()].astype(np.float32)
    lick = beh["lick"]["data"][()].astype(np.float32)
    ...
    deconv = f["processing/ophys/Deconvolved/plane0/data"]
```

iii. The code structure itself makes this clear, and the trajectory describes the full conversion and validation runs as the long-running steps.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious non-vectorized loops are the per-trial zone inference loop, the per-trial reward-outcome loop, the per-trial lick-error loop, the per-trial environment summarization loop, and the main per-trial assembly loop that slices every variable one trial at a time.

ii. 
```python
for start, stop in zip(trial_starts, teleports):
    ...
```
```python
for idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    ...
```
```python
for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    ...
    input_trial = np.vstack([...])
    output_trial = np.vstack([...])
```

iii. The agent did not explicitly justify these loops, but the implementation clearly favors simple per-trial control flow over vectorization.

## 13-c. What processing does the code repeat multiple times?

i. Within each session, the code repeatedly walks the same trial boundaries to compute zone observations, environment observations, reward outcomes, lick-error fractions, and finally the actual per-trial neural/input/output arrays. Across datasets, it reuses the already converted `session_records` to build the sample dataset, so it does not reread the NWB files for the sample.

ii. 
```python
observed_zone_labels, observed_zone_positions = infer_trial_zone_positions(...)
...
for start, stop in zip(trial_starts, teleports):
    ...
    observed_env.append(...)
...
reward_outcomes = reward_outcomes_per_trial(...)
...
drop_mask, lick_error_fraction = drop_lick_error_trials(...)
...
for trial_idx, (start, stop) in enumerate(zip(trial_starts, teleports)):
    ...
```

iii. There is no explicit justification in the notes. The repetition seems to come from keeping each processing stage separate and easy to inspect rather than tightly fusing the passes.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores a substantial amount of per-session bookkeeping that the decoder itself does not consume: observed reward-zone positions/labels, observed environment labels, dropped-trial indices, lick-error fractions, available ROI-series metadata, selected plane indices, and schedule summaries. These are used for documentation and sanity checks rather than for the actual decoder inputs/outputs.

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
session_info["available_roi_series"] = "processing/ophys/Deconvolved/plane0"
session_info["available_response_series_roi_count"] = int(len(rois))
session_info["selected_roi_count_after_iscell"] = int(selected_mask.sum())
session_info["selected_plane_indices"] = sorted(np.unique(plane_idx).astype(int).tolist())
```

iii. The notes emphasize dataset-level sanity checks and decision documentation, so this extra processing appears intentional for auditability even though it is discarded by downstream decoder training.
