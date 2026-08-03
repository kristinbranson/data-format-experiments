# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent scans `/app/data` for subject directories named `sub-*`, then scans each subject directory for `*.nwb` files and converts each file as one session. It opens each NWB file directly with `h5py` and reads only specific HDF5 datasets for behavior and ophys data instead of using `pynwb`.

ii.
```python
subjects_dirs = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
...
nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
...
with h5py.File(nwb_path, 'r') as f:
    pos = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
    ...
    deconv_p0 = f['processing/ophys/Deconvolved/plane0/data'][:]
```

iii. The justification in `CONVERSION_NOTES.md` is that the DANDI NWB files contain the 11 switch-task mice and all relevant variables, so iterating all `sub-*` directories and all `.nwb` files loads the full dataset. The trajectory also shows the agent explicitly inspecting the NWB tree and then choosing direct `h5py` access.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the `sub-*` directory names under `/app/data`, and the stored subject IDs are those directory names with the `sub-` prefix removed.

ii.
```python
subjects_dirs = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])

subjects = [d.replace('sub-', '') for d in subjects_dirs]
```

iii. `CONVERSION_NOTES.md` says the NWB dataset contains 11 switch-task mice such as `sub-m3`, `sub-m4`, and `sub-m17`, and uses that directory-level split as the subject definition.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file is treated as one session. The session number is parsed from the filename substring after `ses-`.

ii.
```python
nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
...
fname = os.path.basename(nwb_file)
session_num = fname.split('ses-')[1].split('_')[0]
result = convert_session(nwb_file, subj_id, session_num)
```

iii. The justification in the notes is that the NWB export has one file per session and subject/session naming is explicit in the filenames.

## 1-d. How are the data split into trials?

i. Trials are split by taking every index where `trial_start > 0` as a trial start and every index where `teleport > 0` as a trial end, then truncating the two lists to equal length and dropping pairs where the teleport index is not after the start index.

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

iii. The agent’s notes say trial boundaries are `trial_start` to `teleport`, aligned to trial start. The trajectory shows it explored those fields in the NWB files and adopted them as the trial structure.

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps any trial with at least 2 timepoints and drops sessions with fewer than 2 valid trials. It does not implement the reference solution’s `< 50` timepoint trial filter.

ii.
```python
if n_trials < 2:
    print(f"  Skipping {subject_id} ses-{session_num}: only {n_trials} valid trials")
    return None
...
n_timepoints = e - s
if n_timepoints < 2:
    continue
...
if len(neural_trials) < 2:
    print(f"  Skipping {subject_id} ses-{session_num}: only {len(neural_trials)} valid trials after filtering")
    return None
```

iii. The only explicit justification present is the target-format requirement that each session must have at least two trials. No note or trajectory evidence justifies lowering the per-trial minimum from 50 to 2 samples.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural matrices saved to output come from `processing/ophys/Deconvolved/plane*/data`, but the cell-selection step also depends on `Fluorescence`, `Neuropil`, `iscell`, and behavioral `speed`.

ii.
```python
deconv_p0 = f['processing/ophys/Deconvolved/plane0/data'][:]
fluor_p0 = f['processing/ophys/Fluorescence/plane0/data'][:]
neuro_p0 = f['processing/ophys/Neuropil/plane0/data'][:]
iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:]
...
is_interneuron = detect_interneurons(dff, speed, tstart_inds, teleport_inds,
                                      threshold=INTERNEURON_SPEED_CORR_THRESHOLD)
```

iii. `CONVERSION_NOTES.md` says the source neural signal is deconvolved calcium activity, but also says cells are further filtered by manual curation and putative interneuron exclusion using dF/F-speed correlation.

## 2-b. How is the `neural` data processed?

i. The agent concatenates multi-plane sessions across planes, computes a simplified dF/F from fluorescence and neuropil for interneuron detection, excludes inferred interneurons, and then outputs filtered deconvolved activity transposed to `(neurons, time)`.

ii.
```python
if has_multi_plane:
    deconv_all = np.concatenate([deconv_p0, deconv_p1], axis=1)
    fluor_all = np.concatenate([fluor_p0, fluor_p1], axis=1)
    neuro_all = np.concatenate([neuro_p0, neuro_p1], axis=1)
...
dff = compute_dff_simple(fluor_all, neuro_all, tstart_inds, teleport_inds)
is_interneuron = detect_interneurons(dff, speed, tstart_inds, teleport_inds,
                                      threshold=INTERNEURON_SPEED_CORR_THRESHOLD)
...
deconv_filtered = deconv_all[:, cell_mask]
...
neural = deconv_filtered[s:e, :].T.astype(np.float32)
```

iii. The notes justify this as matching the paper’s use of deconvolved activity, pooling multi-plane recordings, and excluding putative interneurons by dF/F-speed correlation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells are kept only if `iscell[:, 0] == 1` and they are not flagged as putative interneurons by a Pearson correlation above 0.5 between simplified dF/F and speed.

ii.
```python
curated_mask = iscell[:, 0] == 1
...
is_interneuron = detect_interneurons(dff, speed, tstart_inds, teleport_inds,
                                      threshold=INTERNEURON_SPEED_CORR_THRESHOLD)
...
cell_mask = curated_mask & ~is_interneuron
```

iii. `CONVERSION_NOTES.md` explicitly cites manual Suite2P curation and a `dF/F-speed corr > 0.5` interneuron exclusion threshold as the rationale.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start simply by slicing each trial from `trial_start` index to the paired `teleport` index, so timepoint 0 in each saved trial is the start of the trial.

ii.
```python
for i in range(n_trials):
    s, e = tstart_inds[i], teleport_inds[i]
    ...
    neural = deconv_filtered[s:e, :].T.astype(np.float32)
```

iii. The notes state the temporal alignment event is the start of each trial and describe `trial_start` as re-entry into the virtual environment at position 0.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent does not rebin the neural data. It uses the stored imaging-frame resolution, but in two different ways: trial time vectors use `1 / imaging_rate` from the NWB acquisition metadata, while dataset metadata hardcodes `1000 / 15.5078125` ms.

ii.
```python
imaging_rate = f['acquisition/TwoPhotonSeries/imaging_plane/imaging_rate'][()]
...
frame_time = 1.0 / imaging_rate  # seconds per frame
...
'time_bin_size': 1000.0 / 15.5078125,
```

iii. The notes justify this as “matches the 2-photon imaging frame rate” and say no temporal rebinning is applied.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The saved time-from-start input is derived from trial length and the imaging rate, not from behavioral timestamps.

ii.
```python
imaging_rate = f['acquisition/TwoPhotonSeries/imaging_plane/imaging_rate'][()]
...
n_timepoints = e - s
time_from_start = np.arange(n_timepoints) * frame_time
```

iii. `CONVERSION_NOTES.md` says the time bin is taken from imaging rate and that temporal alignment is at trial start.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the agent constructs a regularly spaced vector `0, frame_time, 2*frame_time, ...` using `np.arange`, then broadcasts that row into the first input channel.

ii.
```python
frame_time = 1.0 / imaging_rate
...
time_from_start = np.arange(n_timepoints) * frame_time
...
input_data[0, :] = time_from_start
```

iii. The notes frame this as using the 2-photon imaging frame rate directly.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time-from-start vector has one sample per neural frame in the sliced trial and is therefore aligned by construction to the neural sample indices for that trial.

ii.
```python
neural = deconv_filtered[s:e, :].T.astype(np.float32)
...
n_timepoints = e - s
time_from_start = np.arange(n_timepoints) * frame_time
input_data = np.zeros((4, n_timepoints), dtype=np.float32)
```

iii. The agent’s stated rationale is that both streams are already frame-aligned in the NWB export and the imaging rate defines the temporal spacing.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Environment type comes from `processing/behavior/BehavioralTimeSeries/environment/data`.

ii.
```python
env = f['processing/behavior/BehavioralTimeSeries/environment/data'][:]
```

iii. The notes say this field directly encodes `0 = ENV1` and `1 = ENV2`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the agent takes all nonnegative environment values within the trial, uses their median as the per-trial label, and broadcasts that constant across timepoints.

ii.
```python
env_per_trial = np.zeros(n_trials, dtype=int)
for i in range(n_trials):
    s, e = tstart_inds[i], teleport_inds[i]
    env_vals = env[s:e]
    env_valid = env_vals[env_vals >= 0]
    if len(env_valid) > 0:
        env_per_trial[i] = int(np.median(env_valid))
...
input_data[1, :] = env_type
```

iii. The notes justify this by saying the environment field is already binary and constant within a trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is not taken from the stored `trial number` series. It is derived from the trial loop index after trials have been segmented by `trial_start` and `teleport`.

ii.
```python
for i in range(n_trials):
    ...
    trial_number = float(i + 1)
```

iii. The notes say trial number is a within-session index and the trajectory shows the agent distrusted some NWB bookkeeping variables, preferring its own trial segmentation.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The agent assigns `i + 1` for trial `i` and broadcasts that constant value across the whole trial.

ii.
```python
trial_number = float(i + 1)
...
input_data[2, :] = trial_number
```

iii. `CONVERSION_NOTES.md` explicitly says the output is a 1-indexed trial number within session.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome comes from reward event timestamps in `Reward/timestamps`, compared to the timestamp ranges of each segmented trial.

ii.
```python
reward_ts = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
timestamps = f['processing/behavior/BehavioralTimeSeries/position/timestamps'][:]
...
reward_outcomes = determine_reward_outcome(reward_ts, timestamps, tstart_inds, teleport_inds)
```

iii. The notes say reward outcome is determined by matching sparse reward timestamps to trial windows, then previous-trial outcome is derived from that binary result.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The agent first computes a per-trial `reward_outcomes` vector, then uses the previous entry for the current trial; the first trial gets 0.

ii.
```python
for i in range(n_trials):
    t_start = timestamps[tstart_inds[i]]
    t_end = timestamps[teleport_inds[i]]
    n_rewards = np.sum((reward_ts >= t_start) & (reward_ts <= t_end))
    outcomes[i] = 1 if n_rewards > 0 else 0
...
if i == 0:
    prev_outcome = 0.0
else:
    prev_outcome = float(reward_outcomes[i - 1])
```

iii. The notes justify this as matching the decoder input definition “omitted = 0, rewarded = 1.”

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Distance to reward zone is derived from the per-trial position trace plus a reward-zone label inferred from `reward_zone > 0` samples and position. If no zone is observed on a trial, the label is inherited from neighboring trials, with final fallback to zone A.

ii.
```python
rz_mask = rzone[s:e] > 0
if np.any(rz_mask):
    rz_positions = pos[s:e][rz_mask]
    zone_labels[i] = get_reward_zone_label(rz_positions)
...
if rz_label is None:
    rz_label = 'A'
rz_start, rz_end = REWARD_ZONES[rz_label]
dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The notes justify this by saying reward-zone location can be recovered from positions where `reward_zone` is active, and omission trials should inherit nearby zone identity.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. After choosing a reward-zone interval, the agent computes signed distance to the nearest edge: negative before the zone, zero inside, positive after the zone.

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

iii. The notes say this follows the paper’s reward-relative framing and uses reward-zone definitions A/B/C at 80-130, 200-250, and 320-370 cm.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The agent uses seven hard-coded categories matching the task instructions.

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

iii. `CONVERSION_NOTES.md` reproduces those seven bins and states that they match the decoder specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by applying the same trial slice `s:e` used for the neural matrix and computing distance from the position samples in that slice.

ii.
```python
neural = deconv_filtered[s:e, :].T.astype(np.float32)
trial_pos = pos[s:e]
...
dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
output_data[0, :] = dist_disc
```

iii. The notes state that behavioral variables are already aligned to imaging frames in the NWB data.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is derived from `processing/behavior/BehavioralTimeSeries/position/data`.

ii.
```python
pos = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
...
trial_pos = pos[s:e]
```

iii. The notes describe this as the mouse’s position along the 450 cm virtual track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The agent clips position to `[0, 450]` cm and then assigns five equal 90 cm bins across the track.

ii.
```python
pos_clipped = np.clip(trial_pos, 0, TRACK_LENGTH)
pos_disc = discretize_position(pos_clipped, n_bins=5)
...
bin_size = TRACK_LENGTH / n_bins
bins = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
```

iii. The notes justify this as “5 equal 90 cm bins” on a 450 cm track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The five categories are `0-90`, `90-180`, `180-270`, `270-360`, and `360-450` cm after clipping to track bounds.

ii.
```python
def discretize_position(position, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins
    bins = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
    return bins
```

iii. `CONVERSION_NOTES.md` explicitly lists the five 90 cm bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position is aligned by taking the same trial slice `s:e` that is used for neural data.

ii.
```python
neural = deconv_filtered[s:e, :].T.astype(np.float32)
trial_pos = pos[s:e]
...
output_data[1, :] = pos_disc
```

iii. The notes say position is already aligned to imaging frames in the NWB export.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick output comes from `processing/behavior/BehavioralTimeSeries/lick/data`.

ii.
```python
lick = f['processing/behavior/BehavioralTimeSeries/lick/data'][:]
...
trial_lick = lick_binary[s:e]
```

iii. The notes describe the NWB lick field as the frame-aligned lick signal.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The agent first applies a lick-sensor-error correction that zeroes whole trials where more than 35% of samples have `lick > 2`, then binarizes with `> 0`.

ii.
```python
def correct_lick_sensor_error(licks, tstart_inds, teleport_inds, correction_thr=0.35):
    ...
    frac_high = np.sum(trial_licks > 2) / len(trial_licks)
    if frac_high > correction_thr:
        licks_corrected[s:e] = 0
...
lick_binary = (lick_corrected > 0).astype(float)
...
lick_disc = trial_lick.astype(int)
```

iii. `CONVERSION_NOTES.md` says this follows a `behavior.py`-style lick sensor correction and then binarizes licks for the decoder.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Licks are aligned by slicing the corrected lick trace with the same trial indices used for neural data.

ii.
```python
neural = deconv_filtered[s:e, :].T.astype(np.float32)
trial_lick = lick_binary[s:e]
...
output_data[3, :] = lick_disc
```

iii. The notes state that the behavioral time series are already aligned to imaging frames.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward zone location is inferred from the `reward_zone` activity trace plus position during each trial; missing trials inherit neighboring labels and can fall back to zone A.

ii.
```python
rz_mask = rzone[s:e] > 0
if np.any(rz_mask):
    rz_positions = pos[s:e][rz_mask]
    zone_labels[i] = get_reward_zone_label(rz_positions)
...
if rz_label is None:
    rz_label = 'A'
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_label]
```

iii. The notes justify this by saying the reward-zone field and positions reveal the active zone, while omission trials need inherited labels.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The agent assigns a zone from the median position of samples with `reward_zone > 0`, then forward-fills and backward-fills missing labels across trials, and finally converts `A/B/C` to `0/1/2`.

ii.
```python
def get_reward_zone_label(position_when_in_rzone):
    median_pos = np.median(position_when_in_rzone)
    for label, (start, end) in REWARD_ZONES.items():
        if start - 20 <= median_pos <= end + 20:
            return label
...
for i in range(n_trials):
    ...
    elif last_label is not None:
        zone_labels[i] = last_label
...
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_label]
```

iii. `CONVERSION_NOTES.md` says omission trials inherit reward-zone identity from neighboring trials via forward/backward fill.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from `Reward/timestamps` compared with each trial’s start and end timestamps.

ii.
```python
reward_ts = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
...
t_start = timestamps[tstart_inds[i]]
t_end = timestamps[teleport_inds[i]]
n_rewards = np.sum((reward_ts >= t_start) & (reward_ts <= t_end))
```

iii. The notes say reward outcome is determined by matching sparse reward events to trial windows.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, if any reward timestamp falls between the trial start and end timestamps, the outcome is 1; otherwise 0. That scalar is then broadcast across all timepoints in the saved output matrix.

ii.
```python
for i in range(n_trials):
    t_start = timestamps[tstart_inds[i]]
    t_end = timestamps[teleport_inds[i]]
    n_rewards = np.sum((reward_ts >= t_start) & (reward_ts <= t_end))
    outcomes[i] = 1 if n_rewards > 0 else 0
...
rew_out = int(reward_outcomes[i])
output_data[5, :] = rew_out
```

iii. `CONVERSION_NOTES.md` ties this to the omission-rate sanity check and the binary decoder target.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent truncates behavior and neural arrays to a common minimum length, drops trials with invalid start/end ordering or fewer than 2 samples, skips sessions with too few trials or too few remaining cells, fills missing environment values with 0, fills missing reward-zone labels from neighboring trials, and falls back to zone A if a label is still missing. It also zeroes entire lick-error trials.

ii.
```python
min_len = min(len(pos), deconv_all.shape[0])
pos = pos[:min_len]
...
valid = teleport_inds > tstart_inds
...
if n_timepoints < 2:
    continue
...
if n_kept < 2:
    return None
...
if len(env_valid) > 0:
    env_per_trial[i] = int(np.median(env_valid))
else:
    env_per_trial[i] = 0
...
if rz_label is None:
    rz_label = 'A'
```

iii. The notes justify these as pragmatic fixes for “occasional off-by-one in NWB,” omission trials, lick sensor artifacts, and missing values, though not all of these fallbacks are explicitly tied back to the reference code.

## 13-a. What are the most time-consuming steps of the code?

i. The slowest steps in the agent’s code are reading large NWB arrays for every session, computing simplified dF/F for all cells, running per-cell Pearson correlations for interneuron detection, and then iterating through every trial to build trial-level arrays. Saving the full pickle and deep-copying a sample dataset are additional costs.

ii.
```python
with h5py.File(nwb_path, 'r') as f:
    ...
dff = compute_dff_simple(fluor_all, neuro_all, tstart_inds, teleport_inds)
is_interneuron = detect_interneurons(dff, speed, tstart_inds, teleport_inds,
                                      threshold=INTERNEURON_SPEED_CORR_THRESHOLD)
...
for i in range(n_trials):
    ...
sample = create_sample(data)
```

iii. This follows directly from the code structure. The notes also emphasize full-dataset conversion, sanity checks, and sample creation.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial dF/F loop, the per-cell Pearson-correlation loop, the per-trial environment loop, the per-trial reward-outcome loop, and the main per-trial trial-construction loop could all be partially vectorized or fused.

ii.
```python
for s, e in zip(tstart_inds, teleport_inds):
    ...
for cell in range(n_cells):
    ...
for i in range(n_trials):
    ...
for i in range(n_trials):
    t_start = timestamps[tstart_inds[i]]
    ...
for i in range(n_trials):
    s, e = tstart_inds[i], teleport_inds[i]
    ...
```

iii. This is an inference from the implementation itself. The agent did not document these loops as an efficiency concern in its notes.

## 13-c. What processing does the code repeat multiple times?

i. The code makes several separate passes over the same trial boundaries: one pass for lick correction, one for reward-zone labeling, one for reward outcomes, one for environment labeling, and one to build the final trial tensors. It also makes extra full-dataset passes in `print_sanity_checks`, and `create_sample` deep-copies the full converted dataset before truncating it.

ii.
```python
lick_corrected, error_trials = correct_lick_sensor_error(...)
zone_labels = determine_reward_zone_per_trial(...)
reward_outcomes = determine_reward_outcome(...)
...
for i in range(n_trials):
    ...
print_sanity_checks(data)
sample = create_sample(data)
```

iii. The justification is implicit rather than stated: the agent factored the code into several small passes instead of one fused pass.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `reward_data` and `planeIdx` but never uses them, computes and stores several session-summary fields used only for console output or notes, deep-copies the full dataset to make a sample file, and runs sanity-check traversals that do not affect the saved full dataset. It also computes simplified dF/F only to discard the dF/F array after using it for cell exclusion.

ii.
```python
reward_data = f['processing/behavior/BehavioralTimeSeries/Reward/data'][:]
...
planeIdx = f['processing/ophys/ImageSegmentation/PlaneSegmentation/planeIdx'][:]
...
dff = compute_dff_simple(fluor_all, neuro_all, tstart_inds, teleport_inds)
...
sample = create_sample(data)
print_sanity_checks(data)
```

iii. No explicit justification is given beyond wanting additional validation outputs and sample artifacts. The notes present these as convenience and sanity-check steps rather than required downstream processing.
