# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers every `sub-*` directory under `/app/data`, strips `sub-` to create the `subjects` list, sorts each subject's `.nwb` files, and calls `convert_session(...)` once per file. Inside `convert_session`, it opens the NWB with `h5py.File(...)` and loads the behavioral streams (`position`, `speed`, `lick`, `reward_zone`, `environment`, `trial_start`, `teleport`, timestamps, reward timestamps) plus ophys streams (`iscell`, `planeIdx`, `Deconvolved`, `Fluorescence`, `Neuropil`).

ii. 
```python
subjects_dirs = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])

subjects = [d.replace('sub-', '') for d in subjects_dirs]

for subj_i, (subj_dir, subj_id) in enumerate(zip(subjects_dirs, subjects)):
    subj_path = os.path.join(data_dir, subj_dir)
    nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
    ...
    result = convert_session(nwb_file, subj_id, session_num)
```

```python
with h5py.File(nwb_path, 'r') as f:
    pos = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
    speed = f['processing/behavior/BehavioralTimeSeries/speed/data'][:]
    lick = f['processing/behavior/BehavioralTimeSeries/lick/data'][:]
    rzone = f['processing/behavior/BehavioralTimeSeries/reward_zone/data'][:]
    env = f['processing/behavior/BehavioralTimeSeries/environment/data'][:]
    trial_start = f['processing/behavior/BehavioralTimeSeries/trial_start/data'][:]
    teleport = f['processing/behavior/BehavioralTimeSeries/teleport/data'][:]
    timestamps = f['processing/behavior/BehavioralTimeSeries/position/timestamps'][:]
    reward_ts = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
```

iii. `CONVERSION_NOTES.md` says the source data are NWB files from DANDI and that the conversion covers the 11 switch-task mice and 152 sessions. The trajectory also says the agent verified that the NWB data contain 11 subjects and 152 total sessions.

## 1-b. How are the data split into subjects?

i. Each `sub-*` directory is treated as one mouse. The agent converts directory names like `sub-m3` into subject ids like `m3`, stores those in `subjects`, and records the subject index for each kept session in `subject_idx`.

ii. 
```python
subjects_dirs = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])

subjects = [d.replace('sub-', '') for d in subjects_dirs]
```

```python
all_neural.append(result['neural'])
all_input.append(result['input'])
all_output.append(result['output'])
subject_idx_list.append(subj_i)
```

iii. `CONVERSION_NOTES.md` includes an explicit subject-mapping table from NWB ids (`sub-m*`) to the paper/code names (`GCAMP*`) and states that there are 11 switch-task mice in the NWB dataset.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file inside a subject directory is treated as one session. Files are globbed and sorted, the session number is parsed from the filename for logging, and each successful file contributes one session entry to `neural`, `input`, and `output`.

ii. 
```python
subj_path = os.path.join(data_dir, subj_dir)
nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))

for nwb_file in nwb_files:
    fname = os.path.basename(nwb_file)
    session_num = fname.split('ses-')[1].split('_')[0]
    result = convert_session(nwb_file, subj_id, session_num)
```

iii. `CONVERSION_NOTES.md` describes session counts per mouse and notes that most mice have 14 sessions while `m11` has 12.

## 1-d. How are the data split into trials?

i. Trials are defined by pairing indices where `trial_start > 0` with indices where `teleport > 0`. After matching counts and removing invalid pairs where teleport is not after the start, each trial is the slice `[s:e]`, so the trial begins at `trial_start` and ends just before the teleport sample. The neural, input, and output trial objects are then built by looping over these start/end pairs.

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

```python
for i in range(n_trials):
    s, e = tstart_inds[i], teleport_inds[i]
    n_timepoints = e - s
    ...
    neural = deconv_filtered[s:e, :].T.astype(np.float32)
```

iii. `CONVERSION_NOTES.md` explicitly says trial boundaries are `trial_start` to `teleport` and that temporal alignment is the start of each trial. The trajectory repeats that the agent chose `trial_start` to `teleport` boundaries.

## 1-e. How are trials filtered based on quality controls?

i. The agent drops trial pairs with invalid boundaries (`teleport <= trial_start`), skips any session with fewer than 2 valid trials, and skips individual trials with fewer than 2 timepoints. It does not apply any stronger minimum trial-length filter.

ii. 
```python
valid = teleport_inds > tstart_inds
tstart_inds = tstart_inds[valid]
teleport_inds = teleport_inds[valid]
...
if n_trials < 2:
    return None
```

```python
for i in range(n_trials):
    s, e = tstart_inds[i], teleport_inds[i]
    n_timepoints = e - s

    if n_timepoints < 2:
        continue
```

iii. No explicit justification is given in `CONVERSION_NOTES.md` for the `<2` threshold. The only clear rationale in the code/comments is preserving valid trial boundaries and preserving the decoder requirement that a session must contain at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final saved neural data are derived from the NWB deconvolved calcium activity arrays (`Deconvolved/plane0/data` and optionally `plane1/data`) after cell filtering. The agent also loads `Fluorescence` and `Neuropil`, but only to compute a helper dF/F signal for interneuron exclusion; those arrays are not saved as `neural`.

ii. 
```python
deconv_p0 = f['processing/ophys/Deconvolved/plane0/data'][:]
fluor_p0 = f['processing/ophys/Fluorescence/plane0/data'][:]
neuro_p0 = f['processing/ophys/Neuropil/plane0/data'][:]

if has_multi_plane:
    deconv_p1 = f['processing/ophys/Deconvolved/plane1/data'][:]
    fluor_p1 = f['processing/ophys/Fluorescence/plane1/data'][:]
    neuro_p1 = f['processing/ophys/Neuropil/plane1/data'][:]
```

iii. `CONVERSION_NOTES.md` says the source for neural data is the deconvolved calcium activity in NWB and that the raw fluorescence and neuropil are only used to build a simplified dF/F signal for interneuron detection.

## 2-b. How is the `neural` data processed?

i. The agent concatenates planes for multi-plane sessions, truncates all modalities to a common minimum length, filters cells, and then segments the deconvolved traces into per-trial matrices. Each trial is transposed to `(n_neurons, n_timepoints)` and stored as `float32`. There is no temporal rebinning or additional normalization of the final saved `neural` arrays.

ii. 
```python
if has_multi_plane:
    deconv_all = np.concatenate([deconv_p0, deconv_p1], axis=1)
    fluor_all = np.concatenate([fluor_p0, fluor_p1], axis=1)
    neuro_all = np.concatenate([neuro_p0, neuro_p1], axis=1)
else:
    deconv_all = deconv_p0
```

```python
min_len = min(len(pos), deconv_all.shape[0])
pos = pos[:min_len]
...
deconv_all = deconv_all[:min_len]
```

```python
deconv_filtered = deconv_all[:, cell_mask]
...
neural = deconv_filtered[s:e, :].T.astype(np.float32)
```

iii. `CONVERSION_NOTES.md` says the agent used deconvolved activity, concatenated multi-plane sessions for `m17`/`m18`, and kept the imaging-frame sampling rather than any rebinning.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells are kept only if `iscell[:, 0] == 1` and they are not classified as putative interneurons. Interneurons are defined as cells whose simplified dF/F trace has Pearson correlation greater than `0.5` with running speed. Sessions with fewer than 2 kept cells are dropped.

ii. 
```python
curated_mask = iscell[:, 0] == 1
...
dff = compute_dff_simple(fluor_all, neuro_all, tstart_inds, teleport_inds)
is_interneuron = detect_interneurons(dff, speed, tstart_inds, teleport_inds,
                                      threshold=INTERNEURON_SPEED_CORR_THRESHOLD)

cell_mask = curated_mask & ~is_interneuron
```

```python
if n_kept < 2:
    print(f"  Skipping {subject_id} ses-{session_num}: only {n_kept} cells after filtering")
    return None
```

iii. `CONVERSION_NOTES.md` explicitly justifies this as matching the paper's manual curation plus putative interneuron exclusion rule, and the trajectory says the agent expected only a small fraction of cells to be excluded as interneurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start. For each trial, the saved neural matrix uses exactly the frame slice from `trial_start` up to `teleport`, so the first neural column is the trial-start frame.

ii. 
```python
tstart_inds = np.where(trial_start > 0)[0]
teleport_inds = np.where(teleport > 0)[0]
...
neural = deconv_filtered[s:e, :].T.astype(np.float32)
```

```python
'temporal_alignment_event': 'start of trial (re-entry into virtual environment at position 0)',
'off_start': 0.0,
```

iii. `CONVERSION_NOTES.md` says "Temporal alignment: start of each trial", and the metadata repeats that exact choice.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the imaging frame rate. Within a session, `frame_time = 1 / imaging_rate`, and metadata record a 64.48 ms bin (`1000 / 15.5078125`). No temporal rebinning is applied.

ii. 
```python
imaging_rate = f['acquisition/TwoPhotonSeries/imaging_plane/imaging_rate'][()]
...
frame_time = 1.0 / imaging_rate  # seconds per frame
```

```python
'time_bin_size': 1000.0 / 15.5078125,
'imaging_rate_hz': 15.5078125,
```

iii. `CONVERSION_NOTES.md` explicitly states a 64.48 ms time bin and says it matches the ~15.5 Hz two-photon frame rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. In the agent's code, this input is derived from the session imaging rate and the number of frames in the trial, not from a raw timestamp time series. It uses frame indices `0..n_timepoints-1` multiplied by `1 / imaging_rate`.

ii. 
```python
frame_time = 1.0 / imaging_rate  # seconds per frame
...
time_from_start = np.arange(n_timepoints) * frame_time
```

iii. `CONVERSION_NOTES.md` only justifies the bin size by citing the imaging frame rate; it does not separately justify preferring frame count over the raw behavioral timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The processing is just a linear ramp starting at 0 seconds: `0, frame_time, 2*frame_time, ...` for the length of the trial. No interpolation, resampling, or timestamp subtraction from the NWB timestamp arrays is used.

ii. 
```python
time_from_start = np.arange(n_timepoints) * frame_time
...
input_data[0, :] = time_from_start
```

iii. The stated rationale in `CONVERSION_NOTES.md` is that the data are kept at the 15.5 Hz imaging frame rate.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by construction because it is built with the same `n_timepoints = e - s` used for the neural slice and written into the same trial matrix width. The first entry corresponds to the first neural frame of the trial.

ii. 
```python
n_timepoints = e - s
...
neural = deconv_filtered[s:e, :].T.astype(np.float32)
...
input_data = np.zeros((4, n_timepoints), dtype=np.float32)
input_data[0, :] = time_from_start
```

iii. `CONVERSION_NOTES.md` says all variables are aligned to the start of the trial; sharing the trial slice with the neural data is how the code implements that.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the NWB behavioral `environment` time series.

ii. 
```python
env = f['processing/behavior/BehavioralTimeSeries/environment/data'][:]
```

iii. `CONVERSION_NOTES.md` says environment type comes directly from the NWB `environment` field and encodes `0 = ENV1`, `1 = ENV2`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the agent takes `env[s:e]`, drops any negative values, uses the median of the remaining samples as the trial label, falls back to 0 if nothing valid remains, and broadcasts that single value across every timepoint in the trial.

ii. 
```python
env_per_trial = np.zeros(n_trials, dtype=int)
for i in range(n_trials):
    s, e = tstart_inds[i], teleport_inds[i]
    env_vals = env[s:e]
    env_valid = env_vals[env_vals >= 0]
    if len(env_valid) > 0:
        env_per_trial[i] = int(np.median(env_valid))
    else:
        env_per_trial[i] = 0
```

```python
env_type = float(env_per_trial[i])
...
input_data[1, :] = env_type
```

iii. `CONVERSION_NOTES.md` says the agent verified the direct `0/1` encoding and specifically checked the mice that start in `ENV2`.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is not derived from the NWB `trial number` time series. Instead, it is generated from the Python loop index `i` during per-trial assembly.

ii. 
```python
for i in range(n_trials):
    ...
    trial_number = float(i + 1)
```

iii. There is no explicit raw-data justification in the notes. `CONVERSION_NOTES.md` only states the result: "1-indexed trial number within session."

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The agent uses a one-based within-session index (`i + 1`) and broadcasts it across all time bins of the trial.

ii. 
```python
trial_number = float(i + 1)
...
input_data[2, :] = trial_number
```

iii. `CONVERSION_NOTES.md` explicitly says the saved variable is "1-indexed trial number within session."

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the raw reward event timestamps. Those timestamps are first converted into a per-trial `reward_outcomes` array, and then each trial uses the previous entry from that array.

ii. 
```python
reward_ts = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
...
reward_outcomes = determine_reward_outcome(reward_ts, timestamps, tstart_inds, teleport_inds)
```

```python
if i == 0:
    prev_outcome = 0.0
else:
    prev_outcome = float(reward_outcomes[i - 1])
```

iii. `CONVERSION_NOTES.md` says the first trial is set to 0 and subsequent trials encode omitted vs rewarded from the prior trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The helper function marks a trial rewarded if any reward timestamp falls within that trial window. During trial assembly, the first trial gets 0, and every later trial gets the previous trial's reward outcome broadcast across time.

ii. 
```python
def determine_reward_outcome(reward_ts, timestamps, tstart_inds, teleport_inds):
    outcomes = np.zeros(n_trials, dtype=int)
    for i in range(n_trials):
        t_start = timestamps[tstart_inds[i]]
        t_end = timestamps[teleport_inds[i]]
        n_rewards = np.sum((reward_ts >= t_start) & (reward_ts <= t_end))
        outcomes[i] = 1 if n_rewards > 0 else 0
    return outcomes
```

```python
input_data[3, :] = prev_outcome
```

iii. `CONVERSION_NOTES.md` describes the exact binary encoding and says reward omission rate was checked against the paper.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the raw position trace plus a per-trial reward-zone label inferred from the raw `reward_zone` signal and position within each trial. The code does not use a precomputed session-wide reward-zone label; it infers the zone from where `rzone > 0` within that trial and fills gaps from neighboring trials.

ii. 
```python
def determine_reward_zone_per_trial(pos, rzone, tstart_inds, teleport_inds):
    ...
    rz_mask = rzone[s:e] > 0
    if np.any(rz_mask):
        rz_positions = pos[s:e][rz_mask]
        zone_labels[i] = get_reward_zone_label(rz_positions)
```

```python
trial_pos = pos[s:e]
...
rz_label = zone_labels[i]
rz_start, rz_end = REWARD_ZONES[rz_label]
dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. `CONVERSION_NOTES.md` says reward-zone location is determined from the position values seen when `reward_zone > 0`, and omission trials inherit the label from neighboring trials.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Once a trial's reward-zone label is chosen, the agent maps that label to a start and end position in centimeters and computes signed distance to the nearest edge: negative before the zone, zero inside it, and positive after it.

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

iii. `CONVERSION_NOTES.md` describes the same signed-distance convention.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is mapped into 7 categories with hand-coded thresholds: `<-50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, and `>50` cm.

ii. 
```python
def discretize_distance(distance):
    bins = np.zeros_like(distance, dtype=int)
    bins[distance < -50] = 0
    bins[(distance >= -50) & (distance < -10)] = 1
    bins[(distance >= -10) & (distance < 0)] = 2
    bins[distance == 0] = 3
    bins[(distance > 0) & (distance <= 10)] = 4
    bins[(distance > 10) & (distance <= 50)] = 5
    bins[distance > 50] = 6
    return bins
```

iii. The bins in `CONVERSION_NOTES.md` exactly match these thresholds.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The distance is computed from `trial_pos = pos[s:e]`, the same per-trial slice used to build the neural matrix, and written into an output array with exactly the same number of time bins.

ii. 
```python
trial_pos = pos[s:e]
...
dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
dist_disc = discretize_distance(dist)
...
output_data[0, :] = dist_disc
```

iii. `CONVERSION_NOTES.md` describes this output as time-varying and aligned to trial start.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the raw NWB position trace, restricted to each trial window.

ii. 
```python
trial_pos = pos[s:e]
```

iii. `CONVERSION_NOTES.md` says this variable is the animal's absolute position in the corridor, discretized after extracting the raw position trace.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The agent clips position values into `[0, 450]` cm and then bins them with `floor(position / 90)` into five bins across the 450 cm track.

ii. 
```python
pos_clipped = np.clip(trial_pos, 0, TRACK_LENGTH)
pos_disc = discretize_position(pos_clipped, n_bins=5)
```

```python
def discretize_position(position, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins
    bins = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
    return bins
```

iii. `CONVERSION_NOTES.md` justifies this by calling the bins "5 equal 90 cm bins" on a 450 cm track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The thresholds are the 90 cm boundaries implied by the helper above: `0-90`, `90-180`, `180-270`, `270-360`, and `360-450` cm.

ii. 
```python
'output_values': [
    ...
    ['0-90cm', '90-180cm', '180-270cm', '270-360cm', '360-450cm'],
    ...
]
```

iii. `CONVERSION_NOTES.md` lists the same five bins and describes them as equal-sized.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position is sliced with the same `[s:e]` trial boundaries as the neural data and written into a per-trial output matrix with the same number of time bins.

ii. 
```python
trial_pos = pos[s:e]
...
output_data[1, :] = pos_disc
```

iii. The notes describe absolute position as a time-varying decoder output, which is implemented here by sharing the trial slice with the neural data.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the raw NWB `lick` time series.

ii. 
```python
lick = f['processing/behavior/BehavioralTimeSeries/lick/data'][:]
```

iii. `CONVERSION_NOTES.md` says lick comes from the NWB lick field and is then corrected and binarized.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The agent first runs a lick-sensor error correction pass: if more than 35% of samples in a trial have cumulative lick count above 2, the whole trial is zeroed. It then binarizes the corrected signal as `lick > 0`.

ii. 
```python
def correct_lick_sensor_error(licks, tstart_inds, teleport_inds, correction_thr=0.35):
    licks_corrected = np.copy(licks)
    for i, (s, e) in enumerate(zip(tstart_inds, teleport_inds)):
        trial_licks = licks_corrected[s:e]
        if len(trial_licks) > 0:
            frac_high = np.sum(trial_licks > 2) / len(trial_licks)
            if frac_high > correction_thr:
                licks_corrected[s:e] = 0
    return licks_corrected, error_trials
```

```python
lick_corrected, error_trials = correct_lick_sensor_error(...)
lick_binary = (lick_corrected > 0).astype(float)
```

iii. `CONVERSION_NOTES.md` explicitly says the agent followed a lick correction routine from the reference behavior code and then binarized the signal.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. After correction and binarization, lick is sliced as `lick_binary[s:e]` for each trial and stored in an output matrix with the same number of frames as the neural data.

ii. 
```python
trial_lick = lick_binary[s:e]
...
lick_disc = trial_lick.astype(int)
...
output_data[3, :] = lick_disc
```

iii. `CONVERSION_NOTES.md` describes lick as a framewise binary output aligned to the imaging frames.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the raw `reward_zone` time series plus the raw position trace. The agent looks at positions where `reward_zone > 0` inside each trial, infers whether those positions correspond to A/B/C, and fills missing trials from neighboring labels.

ii. 
```python
def get_reward_zone_label(position_when_in_rzone):
    median_pos = np.median(position_when_in_rzone)
    for label, (start, end) in REWARD_ZONES.items():
        if start - 20 <= median_pos <= end + 20:
            return label
```

```python
zone_labels = determine_reward_zone_per_trial(pos, rzone, tstart_inds, teleport_inds)
```

iii. `CONVERSION_NOTES.md` says reward-zone location is determined from position samples during `reward_zone > 0` and that omission trials inherit neighboring trial labels.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The median reward-zone-entry position is mapped to A/B/C using the hard-coded spatial ranges, missing labels are forward- then backward-filled, unresolved labels fall back to `A` at output assembly time, and the final per-trial code is `A->0`, `B->1`, `C->2`.

ii. 
```python
for i in range(n_trials):
    s, e = tstart_inds[i], teleport_inds[i]
    rz_mask = rzone[s:e] > 0
    if np.any(rz_mask):
        rz_positions = pos[s:e][rz_mask]
        zone_labels[i] = get_reward_zone_label(rz_positions)
```

```python
for i in range(n_trials):
    if zone_labels[i] is not None:
        last_label = zone_labels[i]
    elif last_label is not None:
        zone_labels[i] = last_label
...
if rz_label is None:
    rz_label = 'A'
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_label]
```

iii. The notes justify the forward/backward inheritance step for omission trials and list the A/B/C spatial ranges.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the raw reward event timestamps. The actual reward amount array is loaded, but the code uses only the timestamps to decide whether a reward occurred in a trial.

ii. 
```python
reward_data = f['processing/behavior/BehavioralTimeSeries/Reward/data'][:]
reward_ts = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
...
reward_outcomes = determine_reward_outcome(reward_ts, timestamps, tstart_inds, teleport_inds)
```

iii. `CONVERSION_NOTES.md` says reward outcome was determined by matching sparse reward-event timestamps to the trial windows.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is labeled rewarded if at least one reward timestamp falls between that trial's start and teleport timestamps; otherwise it is an omission. The resulting 0/1 value is broadcast across the trial in `output_data[5, :]`.

ii. 
```python
n_rewards = np.sum((reward_ts >= t_start) & (reward_ts <= t_end))
outcomes[i] = 1 if n_rewards > 0 else 0
```

```python
rew_out = int(reward_outcomes[i])
...
output_data[5, :] = rew_out
```

iii. `CONVERSION_NOTES.md` describes the binary rewarded/omitted encoding and checks that the omission rate matches the paper.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles several cases defensively: all streams are truncated to the smallest common length; invalid trial boundaries are dropped; missing/invalid environment values default to 0; missing reward-zone labels are forward/backward-filled and ultimately default to `A`; lick-sensor-error trials are zeroed; sessions with too few valid trials or too few cells are skipped.

ii. 
```python
min_len = min(len(pos), deconv_all.shape[0])
pos = pos[:min_len]
...
deconv_all = deconv_all[:min_len]
```

```python
valid = teleport_inds > tstart_inds
...
if len(env_valid) > 0:
    env_per_trial[i] = int(np.median(env_valid))
else:
    env_per_trial[i] = 0
...
if rz_label is None:
    rz_label = 'A'
```

iii. `CONVERSION_NOTES.md` explicitly mentions handling occasional off-by-one NWB length mismatches and filling omission-trial reward-zone labels from neighboring trials. The other fallbacks are only implicit in the code.

## 13-a. What are the most time-consuming steps of the code?

i. The main expensive steps are loading large NWB arrays, computing the helper dF/F signal for every trial and every cell, and then looping over cells again to compute speed correlations for interneuron detection. The final trial-building loop is also large, but the dF/F and correlation passes are the most avoidable heavy work.

ii. 
```python
def compute_dff_simple(fluorescence, neuropil, tstart_inds, teleport_inds,
                       neu_coef=0.7, baseline_window=300):
    ...
    for s, e in zip(tstart_inds, teleport_inds):
        ...
        trial_smooth = ndimage.uniform_filter1d(trial_f, size=15, axis=0)
        baseline = ndimage.minimum_filter1d(trial_smooth, size=win, axis=0)
        baseline = ndimage.maximum_filter1d(baseline, size=win, axis=0)
```

```python
for cell in range(n_cells):
    dff_valid = dff[valid_mask, cell]
    ...
    r, _ = stats.pearsonr(dff_valid, speed_valid)
    if r > threshold:
        is_interneuron[cell] = True
```

iii. `CONVERSION_NOTES.md` justifies this extra work as the mechanism for putative interneuron exclusion.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-cell Pearson-correlation loop, the per-trial lick-correction loop, the per-trial reward-outcome loop, the per-trial environment summarization loop, and much of the final per-trial assembly loop are all written as Python loops and could have been vectorized or batched more aggressively.

ii. 
```python
for cell in range(n_cells):
    ...
    r, _ = stats.pearsonr(dff_valid, speed_valid)
```

```python
for i, (s, e) in enumerate(zip(tstart_inds, teleport_inds)):
    trial_licks = licks_corrected[s:e]
    ...
```

```python
for i in range(n_trials):
    t_start = timestamps[tstart_inds[i]]
    t_end = timestamps[teleport_inds[i]]
    ...
```

iii. No explicit performance justification appears in the notes or trajectory; these choices seem to have been made for implementation simplicity.

## 13-c. What processing does the code repeat multiple times?

i. The code revisits the same trial boundaries many times in separate passes: once to compute trialwise dF/F baselines, again to build the valid mask for interneuron detection, again for lick correction, again for reward-zone inference, again for reward outcomes, again for environment summaries, and then again for final neural/input/output assembly. It also repeatedly slices the same `s:e` windows from the behavioral arrays.

ii. 
```python
for s, e in zip(tstart_inds, teleport_inds):
    ...
```

```python
for i, (s, e) in enumerate(zip(tstart_inds, teleport_inds)):
    ...
```

```python
for i in range(n_trials):
    s, e = tstart_inds[i], teleport_inds[i]
    ...
```

iii. The notes do not explicitly justify these repeated passes.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The largest discarded work is computing the helper dF/F and speed correlations even though the final saved neural data remain the deconvolved traces. The code also loads `reward_data` and `planeIdx` without using them and accepts a `sample_only` argument that is not used inside `convert_all`.

ii. 
```python
reward_data = f['processing/behavior/BehavioralTimeSeries/Reward/data'][:]
...
planeIdx = f['processing/ophys/ImageSegmentation/PlaneSegmentation/planeIdx'][:]
```

```python
dff = compute_dff_simple(fluor_all, neuro_all, tstart_inds, teleport_inds)
is_interneuron = detect_interneurons(dff, speed, tstart_inds, teleport_inds,
                                      threshold=INTERNEURON_SPEED_CORR_THRESHOLD)
```

```python
def convert_all(data_dir, sample_only=False, max_sessions_per_subject=None):
    """Convert all NWB files to the target format."""
```

iii. No explicit downstream justification is given for these extra steps beyond the notes' general rationale for interneuron exclusion.
