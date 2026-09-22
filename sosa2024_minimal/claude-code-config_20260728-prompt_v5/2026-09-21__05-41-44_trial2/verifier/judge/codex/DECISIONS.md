# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent scans `/app/data` for subject directories named `sub-*`, then scans each subject directory for `.nwb` files and processes each file as one session. It loads NWB content directly with `h5py`, not `pynwb`, and reads behavioral arrays from `processing/behavior/BehavioralTimeSeries` and neural arrays from `processing/ophys`. Sessions whose identifier does not match the expected scene patterns are skipped as training sessions.

ii.
```python
subjects_dirs = sorted([
    d for d in os.listdir(DATA_DIR)
    if d.startswith('sub-') and os.path.isdir(os.path.join(DATA_DIR, d))
])
...
for subj_dir in subjects_dirs:
    ...
    nwb_files = sorted([
        f for f in os.listdir(subj_path) if f.endswith('.nwb')
    ])
    ...
    for nwb_file in nwb_files:
        nwb_path = os.path.join(subj_path, nwb_file)
        result = load_and_process_session(nwb_path, subject_name)
```

```python
with h5py.File(nwb_path, 'r') as f:
    beh = f['processing']['behavior']['BehavioralTimeSeries']
    ...
    ophys = f['processing']['ophys']
```

iii. In the trajectory, the agent said it had found 152 NWB files and described the plan as processing whatever session files were present. It explicitly decided to use the NWB structure directly with `h5py` after exploring the file layout, and to skip files whose scene names looked like non-task training sessions.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level subdirectories under `/app/data` whose names start with `sub-`. The stored subject name is the directory name with the `sub-` prefix removed.

ii.
```python
subjects_dirs = sorted([
    d for d in os.listdir(DATA_DIR)
    if d.startswith('sub-') and os.path.isdir(os.path.join(DATA_DIR, d))
])
...
subject_name = subj_dir.replace('sub-', '')
```

iii. In the trajectory, the agent summarized the available subjects as the 11 switch-task mice present in the folder and treated each `sub-*` directory as one mouse.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file inside a subject directory is treated as one session. The agent does not merge files or split a file into multiple sessions; even switch sessions remain a single session.

ii.
```python
nwb_files = sorted([
    f for f in os.listdir(subj_path) if f.endswith('.nwb')
])
...
for nwb_file in nwb_files:
    nwb_path = os.path.join(subj_path, nwb_file)
    result = load_and_process_session(nwb_path, subject_name)
```

iii. In the trajectory, the agent said “each NWB file represents one imaging session” and that switch sessions should remain one-to-one with files rather than being split into pre/post-switch sessions.

## 1-d. How are the data split into trials?

i. Trial starts are taken from frames where `trial_start > 0.5`. Trial ends are the next `teleport > 0.5` frame, with the end index stored as that teleport frame plus one. Converted trials are then further reduced to “on-track” samples within those start/end bounds by keeping only frames with `0 <= position <= 455`.

ii.
```python
start_frames = np.where(trial_start_signal > 0.5)[0]
teleport_frames = np.where(teleport_signal > 0.5)[0]
...
future_teleports = teleport_frames[teleport_frames > sf]
...
ef = future_teleports[0] + 1  # include the teleport frame
...
trial_starts.append(sf)
trial_ends.append(ef)
trial_ids.append(tid)
```

```python
def extract_on_track_indices(position, start_idx, end_idx):
    trial_pos = position[start_idx:end_idx]
    on_track = (trial_pos >= 0) & (trial_pos <= TRACK_LENGTH + 5)
    indices = np.where(on_track)[0] + start_idx
    return indices
```

iii. The trajectory shows the agent reasoning that trials “start at `trial_start=1` and run until `teleport=1`,” but that teleport/tunnel samples were not part of the actual track. It decided to restrict each converted trial to on-track positions between 0 and 450 cm, because the decoder should focus on the real corridor rather than the teleport segment.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered out if they have fewer than 5 kept on-track samples. Sessions are skipped if they have fewer than 2 detected trials overall or fewer than 2 valid trials after filtering.

ii.
```python
if n_trials < 2:
    print(f"  Skipping session with < 2 trials: {scene}")
    return None
...
on_track_idx = extract_on_track_indices(position, ts, te)

if len(on_track_idx) < 5:  # Skip very short trials
    continue
...
if len(neural_trials) < 2:
    print(f"  Skipping session with < 2 valid trials: {scene}")
    return None
```

iii. In the trajectory, the agent said it wanted to exclude trivial or unusable trial fragments after on-track cropping while retaining as much decoder data as possible. It did not mention the reference solution’s stricter 50-sample minimum.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` arrays are derived from the NWB `Deconvolved` dataset, after cell filtering. `Fluorescence` and `Neuropil` are only used to compute a dF/F-like signal for interneuron detection, not for the saved neural activity itself.

ii.
```python
if has_plane1:
    deconv0 = ophys['Deconvolved']['plane0']['data'][:]
    deconv1 = ophys['Deconvolved']['plane1']['data'][:]
    fluor0 = ophys['Fluorescence']['plane0']['data'][:]
    fluor1 = ophys['Fluorescence']['plane1']['data'][:]
    neuro0 = ophys['Neuropil']['plane0']['data'][:]
    neuro1 = ophys['Neuropil']['plane1']['data'][:]
    ...
else:
    deconvolved = ophys['Deconvolved']['plane0']['data'][:]
    fluorescence = ophys['Fluorescence']['plane0']['data'][:]
    neuropil_data = ophys['Neuropil']['plane0']['data'][:]
```

```python
neural_data = deconvolved_cells[:, good_cells]
```

iii. The trajectory records a long deliberation here. The agent noticed the NWB file had `Fluorescence`, `Neuropil`, and `Deconvolved`, recognized that the paper used a custom dF/F pipeline, but decided that recomputing everything from raw fluorescence would be too complex and error-prone. It therefore settled on using the NWB `Deconvolved` signal directly for the decoder while still using a dF/F computation for interneuron detection.

## 2-b. How is the `neural` data processed?

i. The saved neural activity is not recomputed from raw fluorescence. The agent applies `iscell` filtering, computes a per-trial maximin-style dF/F only for interneuron detection, removes putative interneurons, and then saves the remaining NWB `Deconvolved` traces restricted to on-track trial samples.

ii.
```python
cell_mask = iscell == 1
deconvolved_cells = deconvolved[:, cell_mask]
fluorescence_cells = fluorescence[:, cell_mask]
neuropil_cells = neuropil_data[:, cell_mask]
...
dff = compute_dff_for_interneuron_detection(
    fluorescence_cells, neuropil_cells, trial_starts, trial_ends
)
...
good_cells = ~is_interneuron
...
neural_data = deconvolved_cells[:, good_cells]
...
trial_neural = neural_data[on_track_idx, :].T
```

iii. In the trajectory, the agent explicitly chose not to recompute the paper’s full event pipeline. It reasoned that the NWB `Deconvolved` field was likely the published processed output, and that a simpler paper-inspired dF/F routine was sufficient for the small interneuron-exclusion step.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered in two stages: `iscell == 1` from Suite2p curation, then removal of putative interneurons with Pearson correlation above 0.5 between dF/F and running speed. Sessions with zero surviving cells are dropped.

ii.
```python
cell_mask = iscell == 1
...
is_interneuron = detect_interneurons(dff, speed, cell_mask)
...
good_cells = ~is_interneuron
n_good_cells = int(np.sum(good_cells))

if n_good_cells < 1:
    print(f"  Skipping session with 0 good cells: {scene}")
    return None
```

iii. The trajectory says the agent wanted to match the paper’s two quoted neural QC steps: Suite2p `iscell` curation and speed-correlation-based interneuron exclusion at `r > 0.5`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The code first segments by `trial_start` to `teleport`, but the saved per-trial neural data are actually aligned to the first kept on-track sample inside each trial, because only `on_track_idx` is retained. No extra interpolation or temporal shifting is applied.

ii.
```python
on_track_idx = extract_on_track_indices(position, ts, te)
...
trial_neural = neural_data[on_track_idx, :].T
...
trial_times = beh_timestamps[on_track_idx] - beh_timestamps[on_track_idx[0]]
```

iii. In the trajectory, the agent said trials should be aligned to trial start, but it also said it wanted to exclude tunnel/teleport samples and keep only valid corridor positions. The implemented result is that time zero is effectively the first retained on-track frame rather than the raw `trial_start` sample if the trial begins with negative-position tunnel samples.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the native frame rate, with no temporal rebinning or resampling. The stored `time_bin_size` is computed from the median spacing of the trial time input and comes out to about 64.5 ms.

ii.
```python
all_dt = []
for sess_inputs in data['input']:
    for trial_inp in sess_inputs:
        if trial_inp.ndim == 2 and trial_inp.shape[1] > 1:
            dt = np.diff(trial_inp[0, :])
            all_dt.extend(dt.tolist())
if all_dt:
    median_dt_ms = np.median(all_dt) * 1000
    data['metadata']['time_bin_size'] = float(median_dt_ms)
```

iii. The trajectory states that the agent would keep the native imaging rate of about 15.5 Hz and not rebin, because the behavioral timestamps already matched that sampling interval.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. This input is derived from `position` timestamps in the behavior group, after subsetting to the trial’s retained on-track indices.

ii.
```python
beh_timestamps = beh['position']['timestamps'][:]
...
trial_times = beh_timestamps[on_track_idx] - beh_timestamps[on_track_idx[0]]
...
inp[0, :] = trial_times
```

iii. The trajectory says the agent found shared behavioral timestamps sampled at the imaging rate and considered any of those behavior timestamps acceptable for trial-relative time.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The code subtracts the first retained timestamp in each trial from all retained timestamps in that trial. Because of on-track cropping, this is time from first kept on-track sample, not necessarily from the raw `trial_start` sample.

ii.
```python
trial_times = beh_timestamps[on_track_idx] - beh_timestamps[on_track_idx[0]]
...
inp[0, :] = trial_times
```

iii. In the trajectory, the agent described this as “align time relative to trial start,” but also justified restricting the decoder to `position` values on the real track. That rationale produced a cropped trial clock in the actual code.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time input and neural data use the same `on_track_idx` indices within each trial, so they are sample-by-sample aligned after cropping.

ii.
```python
trial_neural = neural_data[on_track_idx, :].T
trial_times = beh_timestamps[on_track_idx] - beh_timestamps[on_track_idx[0]]
inp[0, :] = trial_times
```

iii. The trajectory says the agent believed the behavior and imaging streams were already synchronized at the same rate, so simple indexing was sufficient and no interpolation was needed.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The saved environment input is derived from the NWB `identifier` string, not from the behavior `environment` timeseries. The scene name is parsed into `Env1`/`Env2`, then converted to `0`/`1`.

ii.
```python
identifier = f['identifier'][()]
...
scene = parse_scene_from_identifier(identifier)
session_info = get_session_info(scene)
```

```python
env_map = {'Env1': 0, 'Env2': 1}
...
trial_env = envs[i]
...
inp[1, :] = float(trial_env)
```

iii. The trajectory records that the agent decided the scene identifier was the clearest source of per-session environment identity, especially for switch sessions, and preferred that metadata over inferring environment from the behavior stream.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The code uses regex parsing of the scene name, constructs per-trial environment labels with a switch at trial 30 for switch sessions, and then broadcasts the resulting scalar across all timepoints in the trial.

ii.
```python
def get_session_info(scene):
    env_map = {'Env1': 0, 'Env2': 1}
    ...
    m = re.match(r'(Env[12])_([ABC])_to_(Env[12])_([ABC])', scene)
    ...
    m = re.match(r'(Env[12])_Location([ABC])_to_([ABC])', scene)
    ...
    m = re.match(r'(Env[12])_Location([ABC])', scene)
```

```python
for t in range(n_trials):
    if session_info['is_switch'] and t >= CHANGE_TRIAL:
        ...
    else:
        ...
...
inp[1, :] = float(trial_env)
```

iii. In the trajectory, the agent said stable sessions keep one environment, within-environment switch sessions keep the same environment throughout, and cross-environment switch sessions flip at trial 30. That logic is what the parser and broadcaster implement.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the NWB behavior `trial number` signal sampled at the detected `trial_start` frame for each trial. It is not recomputed as a simple loop index.

ii.
```python
tid = int(trial_number[sf])
...
trial_ids.append(tid)
...
trial_num = trial_ids[i]
...
inp[2, :] = float(trial_num)
```

iii. The trajectory says the agent treated the trial-number stream as the cleanest way to preserve the session’s own trial indexing once trial starts had been identified.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transformation is applied beyond taking the integer `trial_number` value at trial start and broadcasting it across the trial’s timepoints.

ii.
```python
trial_num = trial_ids[i]
...
inp[2, :] = float(trial_num)
```

iii. The trajectory describes this as a per-trial contextual scalar rather than a time-varying quantity, so the code repeats it across time.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from reward event timestamps in the behavior `Reward` series, combined with per-trial start/end times from the behavior timestamps.

ii.
```python
reward_timestamps = beh['Reward']['timestamps'][:]
...
for i, (ts, te) in enumerate(zip(trial_starts, trial_ends)):
    t_start_time = beh_timestamps[ts]
    t_end_time = beh_timestamps[min(te - 1, len(beh_timestamps) - 1)]
    if np.any((reward_timestamps >= t_start_time) & (reward_timestamps <= t_end_time)):
        is_rewarded[i] = 1
```

iii. In the trajectory, the agent said previous-trial outcome should be determined from whether a reward event occurred during the previous trial, and that omission versus missed lick did not matter for this binary variable.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The code first creates a per-trial `is_rewarded` flag. It then sets the previous-outcome input to `0` for the first trial and otherwise copies `is_rewarded[i - 1]`, broadcasting that scalar across all timepoints in the current trial.

ii.
```python
if i == 0:
    prev_outcome = 0
else:
    prev_outcome = is_rewarded[i - 1]
...
inp[3, :] = float(prev_outcome)
```

iii. The trajectory explicitly states that the first trial should default to zero because there is no previous trial, and that the value should represent whether the immediately preceding trial was rewarded.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from `position` and from the reward-zone identity inferred from the NWB `identifier` scene string. The raw `reward_zone` behavior timeseries is not used for the saved output.

ii.
```python
trial_pos = position[on_track_idx]
trial_pos = np.clip(trial_pos, 0, TRACK_LENGTH)
...
zone_label = zones[i]
rz = REWARD_ZONES[zone_label]
...
out[0, :] = discretize_distance_to_reward(trial_pos, rz)
```

iii. The trajectory says the agent initially considered using the `reward_zone` signal directly, but concluded that reward-zone identity “needs to be inferred from the session identifier (scene name), not from the `reward_zone` signal in the NWB.”

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each retained timepoint, the code computes signed distance to the current trial’s reward-zone interval: negative before the zone, zero inside it, positive after it. Position is first clipped into `[0, 450]`, then the continuous signed distance is immediately discretized.

ii.
```python
distance = np.where(
    position < rz_start,
    position - rz_start,
    np.where(
        position > rz_end,
        position - rz_end,
        0.0
    )
)
```

```python
trial_pos = position[on_track_idx]
trial_pos = np.clip(trial_pos, 0, TRACK_LENGTH)
...
out[0, :] = discretize_distance_to_reward(trial_pos, rz)
```

iii. The trajectory explains that the agent wanted the signed interpretation implied by the task: negative when approaching the zone, zero in the zone, positive after it. It also justified excluding teleport/tunnel positions so that only real track locations contributed to this output.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The code thresholds signed distance into 7 bins using explicit inequalities: `< -50`, `[-50, -10)`, `[-10, 0)`, `== 0`, `(0, 10]`, `(10, 50]`, and `> 50`.

ii.
```python
bins[distance < -50] = 0
bins[(distance >= -50) & (distance < -10)] = 1
bins[(distance >= -10) & (distance < 0)] = 2
bins[distance == 0] = 3
bins[(distance > 0) & (distance <= 10)] = 4
bins[(distance > 10) & (distance <= 50)] = 5
bins[distance > 50] = 6
```

iii. In the trajectory, the agent said it was matching the requested signed-distance interpretation with a dedicated “in-zone” bin at zero.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by using the same `on_track_idx` subset that is used for `trial_neural`, so one distance label is produced per retained neural time bin.

ii.
```python
trial_neural = neural_data[on_track_idx, :].T
trial_pos = position[on_track_idx]
...
out[0, :] = discretize_distance_to_reward(trial_pos, rz)
```

iii. The trajectory says the agent believed no extra alignment step was necessary because behavior and imaging were already sampled in sync; it therefore aligned everything by shared within-trial indexing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is derived from the behavior `position` timeseries.

ii.
```python
position = beh['position']['data'][:]
...
trial_pos = position[on_track_idx]
```

iii. The trajectory repeatedly identifies `position` as the direct source of corridor location.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The code keeps only on-track samples, clips them into the range `[0, 450]`, and then bins those clipped positions into five equal 90 cm bins.

ii.
```python
trial_pos = position[on_track_idx]
trial_pos = np.clip(trial_pos, 0, TRACK_LENGTH)
...
out[1, :] = discretize_position(trial_pos)
```

```python
bin_size = TRACK_LENGTH / 5  # 90 cm
bins = np.clip(np.floor(position / bin_size).astype(np.int64), 0, 4)
```

iii. The trajectory says the agent wanted absolute position specifically on the real 450 cm track, not during the tunnel or teleport period, and therefore restricted and clipped positions to the corridor span.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is converted to integer bins `0` through `4` by dividing by 90 cm, taking the floor, and clipping into the valid range.

ii.
```python
bin_size = TRACK_LENGTH / 5  # 90 cm
bins = np.clip(np.floor(position / bin_size).astype(np.int64), 0, 4)
```

iii. The trajectory explicitly notes that the 450 cm track should be split into five equal bins, which is why the code uses 90 cm intervals.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The code indexes both position and neural data with the same `on_track_idx` samples, so the binned position label is aligned to each retained neural sample.

ii.
```python
trial_neural = neural_data[on_track_idx, :].T
trial_pos = position[on_track_idx]
...
out[1, :] = discretize_position(trial_pos)
```

iii. The trajectory says the agent viewed shared sample indices as sufficient alignment because both streams already lived on the same native sampling grid.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick is derived from the behavior `lick` timeseries.

ii.
```python
lick = beh['lick']['data'][:]
...
trial_lick = lick[on_track_idx]
```

iii. The trajectory identifies the NWB `lick` signal as a per-frame lick count that should be collapsed to a binary decoder output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The code first optionally zeroes the whole trial’s lick trace if more than 35% of retained samples exceed 2 licks per frame, treating that as a stuck-sensor artifact. It then binarizes the remaining lick values with `> 0`.

ii.
```python
def check_lick_sensor_error(lick_data, threshold=LICK_CORRECTION_THR):
    if len(lick_data) == 0:
        return False
    frac_high = np.mean(lick_data > 2)
    return frac_high > threshold
```

```python
if check_lick_sensor_error(trial_lick):
    trial_lick = np.zeros_like(trial_lick)
...
trial_lick_binary = (trial_lick > 0).astype(np.int64)
...
out[3, :] = trial_lick_binary
```

iii. The trajectory shows the agent debating how to handle the paper’s lick-sensor correction, noticing a paper/code mismatch between 30% and 35%, and deciding to follow the code-level threshold `0.35`. Because NaNs are not allowed by the decoder, it chose to zero out bad trials instead of inserting missing values.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural data use the same retained `on_track_idx` sample indices within each trial, so the binary lick output is aligned one-to-one with the saved neural time bins.

ii.
```python
trial_neural = neural_data[on_track_idx, :].T
trial_lick = lick[on_track_idx]
...
out[3, :] = trial_lick_binary
```

iii. The trajectory says the agent assumed the behavior and neural streams were already synchronized at the frame level, so shared indexing was enough.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward zone location is derived from the NWB `identifier` scene metadata, not from the behavior `reward_zone` timeseries. The scene name is parsed into A/B/C zone labels and, for switch sessions, split at trial 30.

ii.
```python
scene = parse_scene_from_identifier(identifier)
session_info = get_session_info(scene)
...
zones, envs = get_per_trial_info(session_info, n_trials)
...
zone_label = zones[i]
...
out[4, :] = REWARD_ZONE_LABELS[zone_label]
```

iii. The trajectory explicitly says the agent concluded that “reward zone identity needs to be inferred from the session identifier (scene name), not from the `reward_zone` signal in the NWB.”

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The code parses three scene-name patterns with regex, maps them to zone labels A/B/C, applies a trial-30 switch for switch sessions, and then broadcasts the resulting zone label across all timepoints in that trial.

ii.
```python
m = re.match(r'(Env[12])_([ABC])_to_(Env[12])_([ABC])', scene)
...
m = re.match(r'(Env[12])_Location([ABC])_to_([ABC])', scene)
...
m = re.match(r'(Env[12])_Location([ABC])', scene)
```

```python
for t in range(n_trials):
    if session_info['is_switch'] and t >= CHANGE_TRIAL:
        zones.append(session_info['zone_after'])
    else:
        zones.append(session_info['zone_before'])
...
out[4, :] = REWARD_ZONE_LABELS[zone_label]
```

iii. In the trajectory, the agent said this was the simplest and most robust way to recover the active reward zone, because the identifier already encoded the intended task condition and switch structure.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from reward event timestamps in the behavior `Reward` series together with trial start/end times from the behavior timestamps.

ii.
```python
reward_timestamps = beh['Reward']['timestamps'][:]
...
for i, (ts, te) in enumerate(zip(trial_starts, trial_ends)):
    t_start_time = beh_timestamps[ts]
    t_end_time = beh_timestamps[min(te - 1, len(beh_timestamps) - 1)]
    if np.any((reward_timestamps >= t_start_time) & (reward_timestamps <= t_end_time)):
        is_rewarded[i] = 1
```

iii. The trajectory says the agent only cared whether a reward event was present or absent within each trial, regardless of whether absence reflected omission or failure to lick.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The code creates a per-trial binary `is_rewarded` flag by checking whether any reward timestamp falls between the trial’s start and end times. That binary value is then broadcast across all timepoints of the trial.

ii.
```python
is_rewarded = np.zeros(n_trials, dtype=int)
for i, (ts, te) in enumerate(zip(trial_starts, trial_ends)):
    ...
    if np.any((reward_timestamps >= t_start_time) & (reward_timestamps <= t_end_time)):
        is_rewarded[i] = 1
...
out[5, :] = is_rewarded[i]
```

iii. The trajectory states that reward outcome should be a per-trial binary output driven by the actual presence or absence of a reward event.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles several irregularities defensively: it truncates all streams to the minimum shared length, skips sessions with too few trials or no surviving cells, skips trials with too few retained on-track samples, zeroes whole-trial lick traces flagged as sensor artifacts, and replaces any NaN/Inf values in trial neural data with zeros.

ii.
```python
n_time = min(
    deconvolved.shape[0], fluorescence.shape[0], neuropil_data.shape[0],
    len(position), len(speed), len(lick), len(trial_number),
    len(trial_start_sig), len(teleport_sig), len(beh_timestamps)
)
deconvolved = deconvolved[:n_time]
...
```

```python
if check_lick_sensor_error(trial_lick):
    trial_lick = np.zeros_like(trial_lick)
...
if np.any(np.isnan(trial_neural)) or np.any(np.isinf(trial_neural)):
    trial_neural = np.nan_to_num(trial_neural, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The trajectory mentions an off-by-one neural/behavior length mismatch in some sessions and says the agent chose truncation as the safest fix. It also says the decoder cannot accept NaNs, which is why bad lick trials were zeroed and invalid neural values were converted to zero instead of being left missing.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive parts of this code are likely session-level NWB loading, array materialization for fluorescence/deconvolved/behavior data, the per-trial dF/F computation used for interneuron detection, the per-cell Pearson correlation loop in `detect_interneurons`, and then the per-trial extraction/building of neural-input-output arrays.

ii.
```python
with h5py.File(nwb_path, 'r') as f:
    ...
    deconv0 = ophys['Deconvolved']['plane0']['data'][:]
    ...
    position = beh['position']['data'][:]
```

```python
for t_start, t_end in zip(trial_starts, trial_ends):
    ...
    smoothed = gaussian_filter1d(trial_f, sigma=BASELINE_SMOOTH_SIGMA, axis=0)
    baseline = minimum_filter1d(smoothed, size=BASELINE_WINDOW, axis=0)
    baseline = maximum_filter1d(baseline, size=BASELINE_WINDOW, axis=0)
```

```python
for i in range(n_cells):
    ...
    r, _ = pearsonr(cell_dff, speed_valid)
```

iii. The trajectory repeatedly focuses on multi-plane loading, array-size mismatches, very long trials, and decoder runtime, which is consistent with large I/O and these Python-level loops being the main cost centers.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The cell-by-cell Pearson loop in `detect_interneurons` could be vectorized. The per-trial dF/F loop and the per-trial session-construction loop could also be partly vectorized or batched, though variable trial lengths make the latter less straightforward.

ii.
```python
for i in range(n_cells):
    cell_dff = dff[valid, i]
    ...
    r, _ = pearsonr(cell_dff, speed_valid)
```

```python
for t_start, t_end in zip(trial_starts, trial_ends):
    ...
```

```python
for i in range(n_trials):
    ...
    neural_trials.append(...)
    input_trials.append(...)
    output_trials.append(...)
```

iii. The trajectory does not explicitly propose vectorization, but it does show the agent reasoning about these loops and their impact while debugging runtime and data-shape issues.

## 13-c. What processing does the code repeat multiple times?

i. The code rereads the full contents of every NWB session once per run, and within each session it traverses the same trial boundaries multiple times: once for dF/F, once for reward detection, and once again for final trial extraction. It also computes on-track filtering separately for every trial after already having trial boundaries.

ii.
```python
dff = compute_dff_for_interneuron_detection(
    fluorescence_cells, neuropil_cells, trial_starts, trial_ends
)
...
for i, (ts, te) in enumerate(zip(trial_starts, trial_ends)):
    ...
    if np.any((reward_timestamps >= t_start_time) & (reward_timestamps <= t_end_time)):
        is_rewarded[i] = 1
...
for i in range(n_trials):
    ts = trial_starts[i]
    te = trial_ends[i]
    on_track_idx = extract_on_track_indices(position, ts, te)
```

iii. The trajectory shows the agent incrementally layering trial-level computations rather than consolidating them, especially after deciding to add on-track cropping, lick correction, and per-trial reward handling.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The largest discarded computation is the dF/F pipeline used only for interneuron detection; those dF/F traces are never saved. The code also loads fluorescence and neuropil solely to support that filter, stores `valid_trial_indices` without using them later, reads `planeIdx` without using it, and computes metadata from the already-built inputs instead of using the native sampling metadata directly.

ii.
```python
fluorescence_cells = fluorescence[:, cell_mask]
neuropil_cells = neuropil_data[:, cell_mask]
...
dff = compute_dff_for_interneuron_detection(
    fluorescence_cells, neuropil_cells, trial_starts, trial_ends
)
...
valid_trial_indices.append(i)
```

```python
plane_idx = ophys['ImageSegmentation']['PlaneSegmentation']['planeIdx'][:]
...
all_dt = []
for sess_inputs in data['input']:
    for trial_inp in sess_inputs:
        if trial_inp.ndim == 2 and trial_inp.shape[1] > 1:
            dt = np.diff(trial_inp[0, :])
            all_dt.extend(dt.tolist())
```

iii. The trajectory makes clear that the agent only needed dF/F to support the paper-inspired interneuron filter and otherwise intended to use the stored deconvolved signal. That made fluorescence/neuropil loading and dF/F computation an auxiliary step whose outputs are discarded afterward.
