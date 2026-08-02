# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans `/app/data` for subject directories named `sub-*`, then scans each subject directory for `*.nwb` files. Each NWB file is treated as one session, and the code reads only the behavioral and ophys arrays it needs directly with `h5py` rather than via `pynwb`.

ii.
```python
subjects_dirs = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
...
for subj_i, (subj_dir, subj_id) in enumerate(zip(subjects_dirs, subjects)):
    subj_path = os.path.join(data_dir, subj_dir)
    nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
```

```python
with h5py.File(nwb_path, 'r') as f:
    pos = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
    speed = f['processing/behavior/BehavioralTimeSeries/speed/data'][:]
    ...
    deconv_p0 = f['processing/ophys/Deconvolved/plane0/data'][:]
```

iii. `CONVERSION_NOTES.md` says the dataset contains the 11 switch-task mice and 12-14 sessions per mouse, so the AI justified this directory traversal as capturing the intended NWB sessions. The trajectory does not add a different loading rationale beyond exploring the data layout and then reading named HDF5 paths directly.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level directories under `/app/data` whose names start with `sub-`; the stored subject id is the suffix after `sub-`.

ii.
```python
subjects_dirs = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])

subjects = [d.replace('sub-', '') for d in subjects_dirs]
```

iii. `CONVERSION_NOTES.md` explicitly maps the NWB subject directories (`sub-m3`, `sub-m4`, etc.) to the paper’s mouse identities, so the AI’s subject split is justified from the filesystem structure.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file inside a subject directory is treated as one session. The session number is parsed from the filename substring after `ses-`.

ii.
```python
nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
...
for nwb_file in nwb_files:
    fname = os.path.basename(nwb_file)
    session_num = fname.split('ses-')[1].split('_')[0]
    result = convert_session(nwb_file, subj_id, session_num)
```

iii. The notes summarize session counts per subject and describe them as daily imaging sessions, which is the justification the AI used for mapping one NWB file to one session.

## 1-d. How are the data split into trials?

i. Trial starts are every index where `trial_start > 0`. Trial ends are every index where `teleport > 0`. The code truncates the two index lists to the same length, then drops pairs where `teleport` is not after `trial_start`.

ii.
```python
tstart_inds = np.where(trial_start > 0)[0]
teleport_inds = np.where(teleport > 0)[0]

n_trials = min(len(tstart_inds), len(teleport_inds))
tstart_inds = tstart_inds[:n_trials]
teleport_inds = teleport_inds[:n_trials]

valid = teleport_inds > tstart_inds
tstart_inds = tstart_inds[valid]
teleport_inds = teleport_inds[valid]
```

iii. `CONVERSION_NOTES.md` says trial boundaries are `trial_start` to `teleport` and that trial data include only the on-track period. The code does not mention the reference solution’s use of teleport onset edges; it uses all positive teleport samples instead.

## 1-e. How are trials filtered based on quality controls?

i. Trials are only lightly filtered. The code drops sessions with fewer than 2 valid trials, and inside a kept session it skips trials with fewer than 2 timepoints. It also removes malformed trial pairs where `teleport <= trial_start`.

ii.
```python
if n_trials < 2:
    print(f"  Skipping {subject_id} ses-{session_num}: only {n_trials} valid trials")
    return None
...
for i in range(n_trials):
    s, e = tstart_inds[i], teleport_inds[i]
    n_timepoints = e - s

    if n_timepoints < 2:
        continue
```

iii. The notes emphasize matching paper-level trial counts and keeping the on-track period, but they do not describe any explicit short-trial cutoff beyond these validity checks. The trajectory does not show a stronger trial-QC rule being adopted.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final saved neural matrix comes from deconvolved calcium activity, but neuron inclusion also depends on raw fluorescence, neuropil, and running speed because those are used to detect and exclude putative interneurons.

ii.
```python
deconv_p0 = f['processing/ophys/Deconvolved/plane0/data'][:]
fluor_p0 = f['processing/ophys/Fluorescence/plane0/data'][:]
neuro_p0 = f['processing/ophys/Neuropil/plane0/data'][:]
...
dff = compute_dff_simple(fluor_all, neuro_all, tstart_inds, teleport_inds)
is_interneuron = detect_interneurons(dff, speed, tstart_inds, teleport_inds,
                                      threshold=INTERNEURON_SPEED_CORR_THRESHOLD)
```

iii. `CONVERSION_NOTES.md` says the source neural signal is deconvolved calcium activity and separately documents interneuron exclusion based on dF/F-speed correlation, which is the stated reason these extra raw variables influence the final neural dataset.

## 2-b. How is the `neural` data processed?

i. The AI concatenates multi-plane sessions across planes, truncates all streams to a common minimum length, computes a simplified trialwise dF/F only for interneuron detection, removes those cells, then slices per trial, transposes to `(neurons, time)`, and casts to `float32`.

ii.
```python
if has_multi_plane:
    deconv_all = np.concatenate([deconv_p0, deconv_p1], axis=1)
    fluor_all = np.concatenate([fluor_p0, fluor_p1], axis=1)
    neuro_all = np.concatenate([neuro_p0, neuro_p1], axis=1)
...
min_len = min(len(pos), deconv_all.shape[0])
pos = pos[:min_len]
...
deconv_all = deconv_all[:min_len]
```

```python
dff = compute_dff_simple(fluor_all, neuro_all, tstart_inds, teleport_inds)
is_interneuron = detect_interneurons(dff, speed, tstart_inds, teleport_inds,
                                      threshold=INTERNEURON_SPEED_CORR_THRESHOLD)
cell_mask = curated_mask & ~is_interneuron
deconv_filtered = deconv_all[:, cell_mask]
...
neural = deconv_filtered[s:e, :].T.astype(np.float32)
```

iii. The notes justify this by citing the paper’s use of deconvolved activity, plane pooling for m17/m18, and putative interneuron removal. They present the dF/F step as a preprocessing step only for filtering, not as the saved neural signal.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells are kept only if `iscell[:, 0] == 1` and they are not flagged as putative interneurons by Pearson correlation greater than 0.5 between simplified dF/F and speed.

ii.
```python
curated_mask = iscell[:, 0] == 1
...
is_interneuron = detect_interneurons(dff, speed, tstart_inds, teleport_inds,
                                      threshold=INTERNEURON_SPEED_CORR_THRESHOLD)
...
cell_mask = curated_mask & ~is_interneuron
```

iii. `CONVERSION_NOTES.md` explicitly lists the two-stage filter and says the paper reported additional interneuron exclusion, which is the AI’s justification for going beyond `iscell`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned implicitly by cutting each session into trial windows from `trial_start` to `teleport`. Within each trial, the first frame is treated as time zero for the decoder.

ii.
```python
for i in range(n_trials):
    s, e = tstart_inds[i], teleport_inds[i]
    ...
    neural = deconv_filtered[s:e, :].T.astype(np.float32)
```

```python
frame_time = 1.0 / imaging_rate
time_from_start = np.arange(n_timepoints) * frame_time
```

iii. The notes say temporal alignment is “start of each trial” and describe `trial_start` as re-entry into the virtual environment at position 0, so the AI treated trial slicing itself as the alignment step.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the native imaging frame rate. The code uses `1 / imaging_rate` seconds per frame and does not rebin or resample neural data.

ii.
```python
imaging_rate = f['acquisition/TwoPhotonSeries/imaging_plane/imaging_rate'][()]
...
frame_time = 1.0 / imaging_rate  # seconds per frame
```

```python
'time_bin_size': 1000.0 / 15.5078125,
```

iii. `CONVERSION_NOTES.md` reports a 64.48 ms time bin and says it matches the 2-photon imaging frame rate. The AI treats the recorded frame rate as the authoritative bin size.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the imaging rate and the trial start/end indices, not from the behavior timestamps. The number of frames in the trial determines the length of the time vector.

ii.
```python
imaging_rate = f['acquisition/TwoPhotonSeries/imaging_plane/imaging_rate'][()]
...
frame_time = 1.0 / imaging_rate  # seconds per frame
...
n_timepoints = e - s
time_from_start = np.arange(n_timepoints) * frame_time
```

iii. The notes say the time bin is fixed by imaging rate and that behavioral streams are already aligned to imaging frames. That is the AI’s rationale for using frame count times frame duration instead of timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the code makes a uniformly spaced vector `0, frame_time, 2*frame_time, ...` of length equal to the number of frames in that trial.

ii.
```python
frame_time = 1.0 / imaging_rate  # seconds per frame
...
time_from_start = np.arange(n_timepoints) * frame_time
...
input_data[0, :] = time_from_start
```

iii. The justification in the notes is that the frame interval is constant at about 15.5 Hz, so an evenly spaced time axis is sufficient.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time vector is aligned by using the same trial slices and the same number of frames as the neural matrix. There is no separate interpolation or timestamp matching step.

ii.
```python
s, e = tstart_inds[i], teleport_inds[i]
n_timepoints = e - s
neural = deconv_filtered[s:e, :].T.astype(np.float32)
time_from_start = np.arange(n_timepoints) * frame_time
```

iii. `CONVERSION_NOTES.md` states that speed and other behavioral data are already aligned to imaging frames, so the AI assumed a shared frame index was enough for alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the raw `environment` behavioral time series.

ii.
```python
env = f['processing/behavior/BehavioralTimeSeries/environment/data'][:]
```

iii. The notes explicitly say “Directly from NWB `environment` field: 0 = ENV1, 1 = ENV2.”

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI computes one environment value per trial by taking the median of the non-negative `environment` samples in that trial, then broadcasts that scalar across the trial’s timepoints.

ii.
```python
env_per_trial = np.zeros(n_trials, dtype=int)
for i in range(n_trials):
    s, e = tstart_inds[i], teleport_inds[i]
    env_vals = env[s:e]
    env_valid = env_vals[env_vals >= 0]
    if len(env_valid) > 0:
        env_per_trial[i] = int(np.median(env_valid))
```

```python
env_type = float(env_per_trial[i])
...
input_data[1, :] = env_type
```

iii. The notes say the environment is a binary per-trial variable and is constant within trials, so the AI collapsed the per-frame series to a trial label.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is not taken from a stored NWB variable. It is derived from the loop index after the AI has segmented trials using `trial_start` and `teleport`.

ii.
```python
for i in range(n_trials):
    ...
    trial_number = float(i + 1)
```

iii. The notes describe trial number as “1-indexed trial number within session,” which matches this loop-counter implementation.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The AI uses a 1-indexed sequential session-local trial counter and broadcasts it across all timepoints in the trial.

ii.
```python
trial_number = float(i + 1)
...
input_data[2, :] = trial_number
```

iii. The justification in `CONVERSION_NOTES.md` is simply that trial number is a per-trial contextual variable.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from reward event timestamps in the `Reward` time series, together with per-trial time windows defined using the behavioral timestamps and trial boundaries.

ii.
```python
reward_ts = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
timestamps = f['processing/behavior/BehavioralTimeSeries/position/timestamps'][:]
...
reward_outcomes = determine_reward_outcome(reward_ts, timestamps, tstart_inds, teleport_inds)
```

```python
def determine_reward_outcome(reward_ts, timestamps, tstart_inds, teleport_inds):
    ...
    n_rewards = np.sum((reward_ts >= t_start) & (reward_ts <= t_end))
    outcomes[i] = 1 if n_rewards > 0 else 0
```

iii. The notes say reward outcome is determined by matching sparse reward-event timestamps to trial windows, and previous-trial outcome is then derived from those per-trial reward labels.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The code first builds a binary reward outcome for each trial. For trial `i`, it then assigns `reward_outcomes[i - 1]`; the first trial is forced to 0.

ii.
```python
reward_outcomes = determine_reward_outcome(reward_ts, timestamps, tstart_inds, teleport_inds)
...
if i == 0:
    prev_outcome = 0.0
else:
    prev_outcome = float(reward_outcomes[i - 1])
...
input_data[3, :] = prev_outcome
```

iii. `CONVERSION_NOTES.md` explicitly describes previous trial outcome as binary, constant within a trial, with the first trial of the session set to 0.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from `position` plus a per-trial reward-zone label. That reward-zone label is itself inferred from `reward_zone` and `position` by looking at positions where `reward_zone > 0`.

ii.
```python
pos = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
rzone = f['processing/behavior/BehavioralTimeSeries/reward_zone/data'][:]
...
zone_labels = determine_reward_zone_per_trial(pos, rzone, tstart_inds, teleport_inds)
```

```python
rz_mask = rzone[s:e] > 0
if np.any(rz_mask):
    rz_positions = pos[s:e][rz_mask]
    zone_labels[i] = get_reward_zone_label(rz_positions)
```

iii. The notes say reward-zone location is determined from position samples where the reward-zone field is positive, with omission trials inheriting the neighboring trial’s label.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Once a trial’s reward-zone boundaries are known, the code computes signed distance to the nearest zone edge: negative before the zone, zero inside it, positive after it.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    distance = np.zeros_like(position, dtype=float)
    before = position < rz_start
    inside = (position >= rz_start) & (position <= rz_end)
    after = position > rz_end
    distance[before] = position[before] - rz_start
    distance[inside] = 0
    distance[after] = position[after] - rz_end
    return distance
```

iii. `CONVERSION_NOTES.md` describes the same signed-distance definition and links it to the reward-zone ranges A, B, and C.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI uses explicit threshold comparisons to assign one of 7 classes matching the decoder specification.

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

iii. The notes reproduce the same 7-bin decoder definition and present this as a direct implementation of the task instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by slicing `position` and `neural` with the same per-trial start/end indices and then computing the distance on that trial slice.

ii.
```python
s, e = tstart_inds[i], teleport_inds[i]
neural = deconv_filtered[s:e, :].T.astype(np.float32)
trial_pos = pos[s:e]
...
dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
output_data[0, :] = dist_disc
```

iii. The notes say the behavioral variables are already aligned to imaging frames and that temporal alignment is at trial start, so the AI relied on shared frame indices.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the raw `position` behavioral time series.

ii.
```python
pos = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
...
trial_pos = pos[s:e]
```

iii. The notes describe absolute position as “Position discretized into 5 equal 90cm bins,” implying direct use of the corridor position signal.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Before binning, the AI clips trial positions to the range `[0, 450]` cm, then converts them to integer bin indices with equal-width bins.

ii.
```python
pos_clipped = np.clip(trial_pos, 0, TRACK_LENGTH)
pos_disc = discretize_position(pos_clipped, n_bins=5)
```

iii. The notes justify this by treating the corridor as a 450 cm track and describing the output as five equal 90 cm bins from 0 to 450 cm.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The AI divides the 0-450 cm track into 5 equal 90 cm bins and assigns `floor(position / 90)` clipped to `[0, 4]`.

ii.
```python
def discretize_position(position, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins
    bins = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
    return bins
```

iii. `CONVERSION_NOTES.md` explicitly describes the bins as `0-90`, `90-180`, `180-270`, `270-360`, and `360-450`.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position is aligned to neural data by using the same per-trial start/end slice and the same frame count as the neural matrix.

ii.
```python
s, e = tstart_inds[i], teleport_inds[i]
neural = deconv_filtered[s:e, :].T.astype(np.float32)
trial_pos = pos[s:e]
...
output_data[1, :] = pos_disc
```

iii. The notes state that behavioral data are already aligned to imaging frames, so no separate realignment was attempted.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the raw `lick` behavioral time series.

ii.
```python
lick = f['processing/behavior/BehavioralTimeSeries/lick/data'][:]
```

iii. The notes describe lick as a binary time-varying output built from the NWB lick field.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI first runs a lick-sensor error correction: trials where more than 35% of samples have lick values greater than 2 are zeroed out. It then binarizes the corrected lick signal using `> 0`.

ii.
```python
def correct_lick_sensor_error(licks, tstart_inds, teleport_inds, correction_thr=0.35):
    ...
    frac_high = np.sum(trial_licks > 2) / len(trial_licks)
    if frac_high > correction_thr:
        licks_corrected[s:e] = 0
```

```python
lick_corrected, error_trials = correct_lick_sensor_error(
    lick, tstart_inds, teleport_inds, correction_thr=LICK_CORRECTION_THR)
lick_binary = (lick_corrected > 0).astype(float)
...
lick_disc = trial_lick.astype(int)
```

iii. `CONVERSION_NOTES.md` says this follows `behavior.py:correct_lick_sensor_error` and cites the paper’s discussion of lick sensor errors as the justification.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned by slicing the corrected lick vector with the same per-trial frame indices used for neural data and other behavioral outputs.

ii.
```python
s, e = tstart_inds[i], teleport_inds[i]
neural = deconv_filtered[s:e, :].T.astype(np.float32)
trial_lick = lick_binary[s:e]
...
output_data[3, :] = lick_disc
```

iii. The notes say the lick field is already aligned to imaging frames, so no additional interpolation or timestamp matching is used.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the raw `reward_zone` time series and the raw `position` time series.

ii.
```python
pos = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
rzone = f['processing/behavior/BehavioralTimeSeries/reward_zone/data'][:]
...
zone_labels = determine_reward_zone_per_trial(pos, rzone, tstart_inds, teleport_inds)
```

iii. The notes’ “Reward Zone Detection” section says zone location is determined from position samples when the `reward_zone` field is positive.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI infers the zone per trial by taking the median position of frames where `reward_zone > 0`, matching that median to A/B/C with a ±20 cm tolerance, then forward-filling and backward-filling missing trial labels. If a label is still missing when writing outputs, it falls back to `A`.

ii.
```python
def get_reward_zone_label(position_when_in_rzone):
    median_pos = np.median(position_when_in_rzone)
    for label, (start, end) in REWARD_ZONES.items():
        if start - 20 <= median_pos <= end + 20:
            return label
```

```python
for i in range(n_trials):
    s, e = tstart_inds[i], teleport_inds[i]
    rz_mask = rzone[s:e] > 0
    if np.any(rz_mask):
        rz_positions = pos[s:e][rz_mask]
        zone_labels[i] = get_reward_zone_label(rz_positions)
...
elif last_label is not None:
    zone_labels[i] = last_label
```

```python
rz_label = zone_labels[i]
if rz_label is None:
    rz_label = 'A'
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_label]
```

iii. `CONVERSION_NOTES.md` says omission trials inherit neighboring labels by forward/backward fill and that the position ranges for A/B/C come from the reference behavior code.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the `Reward` event timestamps, using behavioral timestamps and trial boundaries to decide which trial each reward belongs to.

ii.
```python
reward_ts = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
timestamps = f['processing/behavior/BehavioralTimeSeries/position/timestamps'][:]
...
reward_outcomes = determine_reward_outcome(reward_ts, timestamps, tstart_inds, teleport_inds)
```

iii. The notes say reward outcome is determined by matching sparse reward-event timestamps to trial time windows.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the AI checks whether any reward timestamp falls between the trial start time and trial end time. That binary trial label is then broadcast across the trial’s output timepoints.

ii.
```python
for i in range(n_trials):
    t_start = timestamps[tstart_inds[i]]
    t_end = timestamps[teleport_inds[i]]
    n_rewards = np.sum((reward_ts >= t_start) & (reward_ts <= t_end))
    outcomes[i] = 1 if n_rewards > 0 else 0
```

```python
rew_out = int(reward_outcomes[i])
...
output_data[5, :] = rew_out
```

iii. `CONVERSION_NOTES.md` reports an omission rate near the paper’s 15%, which is the AI’s main validation for this reward-window method.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several problems defensively and mostly by truncation or fallback: all streams are truncated to the minimum common length, unmatched start/end trial lists are truncated to the smaller count, trials with `teleport <= trial_start` are dropped, missing environment values default to 0, trials with missing reward-zone detections are filled from neighbors, and unresolved reward-zone labels fall back to `A`. Sessions with fewer than 2 valid trials or fewer than 2 kept cells are skipped.

ii.
```python
min_len = min(len(pos), deconv_all.shape[0])
pos = pos[:min_len]
...
deconv_all = deconv_all[:min_len]
```

```python
n_trials = min(len(tstart_inds), len(teleport_inds))
tstart_inds = tstart_inds[:n_trials]
teleport_inds = teleport_inds[:n_trials]
valid = teleport_inds > tstart_inds
```

```python
if len(env_valid) > 0:
    env_per_trial[i] = int(np.median(env_valid))
else:
    env_per_trial[i] = 0
...
if rz_label is None:
    rz_label = 'A'
```

iii. The notes present these as practical fixes to preserve session counts and keep reward-zone distributions, omission rate, and trial counts close to the paper’s reported statistics.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive parts are loading large NWB arrays with `h5py`, concatenating multi-plane recordings, computing trialwise dF/F for every cell, computing per-cell Pearson correlations for interneuron detection, and then building per-trial neural/input/output arrays for all sessions.

ii.
```python
with h5py.File(nwb_path, 'r') as f:
    ...
    deconv_p0 = f['processing/ophys/Deconvolved/plane0/data'][:]
    fluor_p0 = f['processing/ophys/Fluorescence/plane0/data'][:]
    neuro_p0 = f['processing/ophys/Neuropil/plane0/data'][:]
```

```python
dff = compute_dff_simple(fluor_all, neuro_all, tstart_inds, teleport_inds)
is_interneuron = detect_interneurons(dff, speed, tstart_inds, teleport_inds,
                                      threshold=INTERNEURON_SPEED_CORR_THRESHOLD)
...
for i in range(n_trials):
    ...
    neural_trials.append(neural)
```

iii. The notes emphasize full-dataset statistics over 152 sessions and specifically call out interneuron detection and lick correction as added processing steps, so those are the main sources of cost beyond I/O.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious vectorization targets are the per-trial loop in `correct_lick_sensor_error`, the per-trial loop in `compute_dff_simple`, the per-cell loop in `detect_interneurons`, the per-trial `env_per_trial` loop, and the per-trial construction loop for `neural_trials`, `input_trials`, and `output_trials`.

ii.
```python
for i, (s, e) in enumerate(zip(tstart_inds, teleport_inds)):
    trial_licks = licks_corrected[s:e]
    ...
```

```python
for s, e in zip(tstart_inds, teleport_inds):
    ...
for cell in range(n_cells):
    ...
for i in range(n_trials):
    ...
```

iii. The trajectory and notes do not show the AI trying to optimize these loops; the emphasis was on matching its interpretation of the paper and getting valid decoder outputs.

## 13-c. What processing does the code repeat multiple times?

i. The code repeats several passes over the same trial boundaries: once for lick correction, once for dF/F computation, once for reward-zone detection, once for reward-outcome computation, once for environment aggregation, and once to assemble the saved trial arrays. It also deep-copies the full dataset again when creating the sample dataset.

ii.
```python
lick_corrected, error_trials = correct_lick_sensor_error(...)
zone_labels = determine_reward_zone_per_trial(pos, rzone, tstart_inds, teleport_inds)
reward_outcomes = determine_reward_outcome(reward_ts, timestamps, tstart_inds, teleport_inds)
...
for i in range(n_trials):
    ...
```

```python
sample = copy.deepcopy(data)
```

iii. The notes do not justify this as an optimization choice; it is a straightforward implementation that recomputes different per-trial summaries in separate passes.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `reward_data` and `planeIdx` but never uses them. It also loads fluorescence and neuropil and computes dF/F solely to filter cells; those intermediates are discarded rather than saved. The returned per-session bookkeeping fields like `zone_labels`, `reward_outcomes`, `env_per_trial`, and `lick_error_trials` are used for logging but are not kept in the final saved dataset.

ii.
```python
reward_data = f['processing/behavior/BehavioralTimeSeries/Reward/data'][:]
...
planeIdx = f['processing/ophys/ImageSegmentation/PlaneSegmentation/planeIdx'][:]
```

```python
fluor_p0 = f['processing/ophys/Fluorescence/plane0/data'][:]
neuro_p0 = f['processing/ophys/Neuropil/plane0/data'][:]
...
dff = compute_dff_simple(fluor_all, neuro_all, tstart_inds, teleport_inds)
```

```python
return {
    ...
    'zone_labels': zone_labels,
    'reward_outcomes': reward_outcomes,
    'env_per_trial': env_per_trial,
    'lick_error_trials': error_trials,
}
```

iii. The notes justify some of this as sanity-checking and paper-matching, but none of those extra arrays are part of the final decoder dataset written to `converted_data.pkl`.
