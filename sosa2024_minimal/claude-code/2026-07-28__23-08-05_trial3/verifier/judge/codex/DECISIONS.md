# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent scans `/app/data` for all `sub-*` directories, then loads every `.nwb` file inside each subject directory as a session. It reads NWB content directly with `h5py` and only pulls the arrays it needs for the conversion, rather than using `pynwb`.

ii.
```python
subjects_dirs = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])

for subj_i, (subj_dir, subj_id) in enumerate(zip(subjects_dirs, subjects)):
    subj_path = os.path.join(data_dir, subj_dir)
    nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
```
```python
with h5py.File(nwb_path, 'r') as f:
    pos = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
    speed = f['processing/behavior/BehavioralTimeSeries/speed/data'][:]
    deconv_p0 = f['processing/ophys/Deconvolved/plane0/data'][:]
```

iii. In the trajectory, the agent said it had identified “11 subjects, 152 total sessions” and described the NWB fields it planned to use. It explicitly concluded that the NWB files already exposed the needed behavior and neural arrays and then wrote the converter around direct HDF5 paths.

## 1-b. How are the data split into subjects?

i. Subjects are the `sub-*` directories under `/app/data`, with the `sub-` prefix stripped to form subject IDs such as `m11` and `m17`.

ii.
```python
subjects_dirs = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])

subjects = [d.replace('sub-', '') for d in subjects_dirs]
```

iii. The trajectory repeatedly notes that the dataset contains 11 subjects with directory names like `sub-m3`, `sub-m4`, and `sub-m11`, and that these map directly to the final subject IDs.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file within a subject directory is treated as one session. The session number is parsed from the filename fragment after `ses-`.

ii.
```python
nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))

for nwb_file in nwb_files:
    fname = os.path.basename(nwb_file)
    session_num = fname.split('ses-')[1].split('_')[0]
```

iii. In the trajectory, the agent described the data as “152 total sessions” and used the filename convention to reason about session numbering, especially for `m11` starting at `ses-03`.

## 1-d. How are the data split into trials?

i. Trials are split by taking every index where `trial_start > 0` as a start and every index where `teleport > 0` as an end. The code truncates the two index arrays to the same length, removes any pair where the teleport index is not after the start index, and then slices trials as `[start:end)`.

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
```

iii. The trajectory says “trial boundaries come from the trial_start and teleport signals,” and later says the resulting trial-length distribution looked reasonable enough to proceed. There is no evidence the agent noticed that the reference detects teleport onset transitions rather than every positive teleport sample.

## 1-e. How are trials filtered based on quality controls?

i. The agent does not apply the reference solution’s `< 50` timepoint trial filter. Instead, it only skips a trial if `n_timepoints < 2`, and it skips entire sessions if they end up with fewer than 2 trials. Lick-sensor-error trials are corrected to zeros rather than removed.

ii.
```python
if n_trials < 2:
    print(f"  Skipping {subject_id} ses-{session_num}: only {n_trials} valid trials")
    return None
```
```python
for i in range(n_trials):
    s, e = tstart_inds[i], teleport_inds[i]
    n_timepoints = e - s

    if n_timepoints < 2:
        continue
```
```python
lick_corrected, error_trials = correct_lick_sensor_error(
    lick, tstart_inds, teleport_inds, correction_thr=LICK_CORRECTION_THR)
```

iii. The trajectory shows the agent debugging session-level shape mismatches and checking gross trial statistics, but it does not mention adopting the human reference’s minimum-length trial filter. The only explicit QC rationale it recorded was that the trial-length distribution “looked reasonable.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` data are taken from the NWB `Deconvolved` arrays. The raw `Fluorescence` and `Neuropil` arrays are loaded too, but only to compute a simplified dF/F for interneuron detection.

ii.
```python
deconv_p0 = f['processing/ophys/Deconvolved/plane0/data'][:]
fluor_p0 = f['processing/ophys/Fluorescence/plane0/data'][:]
neuro_p0 = f['processing/ophys/Neuropil/plane0/data'][:]
```
```python
deconv_filtered = deconv_all[:, cell_mask]
neural = deconv_filtered[s:e, :].T.astype(np.float32)
```

iii. In the trajectory, the agent explicitly wrote that “Neural data: Fluorescence (raw), Neuropil, Deconvolved (already processed)” and then decided to use the already deconvolved NWB signal as the output neural representation.

## 2-b. How is the `neural` data processed?

i. The agent concatenates planes when present, crops neural and behavioral streams to a common minimum length, filters cells, and then slices the already-stored deconvolved activity into trials. It does not recompute the paper’s dF/F and OASIS events for the final output.

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
deconv_all = deconv_all[:min_len]
...
deconv_filtered = deconv_all[:, cell_mask]
neural = deconv_filtered[s:e, :].T.astype(np.float32)
```

iii. The trajectory says the NWB files already contain processed deconvolved activity and that the agent planned to use it directly. Later it only revisited dF/F in the narrower context of checking whether the interneuron detector seemed reasonable.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent keeps only `iscell[:, 0] == 1` ROIs and excludes putative interneurons whose simplified dF/F trace has Pearson correlation above `0.5` with running speed.

ii.
```python
curated_mask = iscell[:, 0] == 1
...
is_interneuron = detect_interneurons(dff, speed, tstart_inds, teleport_inds,
                                      threshold=INTERNEURON_SPEED_CORR_THRESHOLD)
cell_mask = curated_mask & ~is_interneuron
```
```python
if r > threshold:
    is_interneuron[cell] = True
```

iii. The trajectory states that the agent needed to “filter cells using the `iscell[:,0] == 1` criterion” and also “exclude putative interneurons based on a Pearson correlation threshold with speed (> 0.5).” It later justified the small number removed by citing the paper’s reported exclusion rate.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent aligns neural data by starting each trial at the `trial_start` index and slicing the deconvolved activity over the same `[start:end)` window used for other trial data.

ii.
```python
for i in range(n_trials):
    s, e = tstart_inds[i], teleport_inds[i]
    ...
    neural = deconv_filtered[s:e, :].T.astype(np.float32)
```

iii. The script docstring says “Temporal alignment: start of each trial,” and the trajectory also says trial boundaries come from `trial_start` and `teleport`. There is no extra temporal shifting beyond trial slicing.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent assumes a single imaging-frame resolution for all converted data and applies no temporal rebinning. It uses `1 / imaging_rate` for trial time vectors and stores a fixed metadata time bin size of `1000.0 / 15.5078125` ms.

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

iii. In the trajectory, the agent repeatedly described the imaging rate as about 15.5 Hz and never discussed any resampling. It treated the stored frame rate as the dataset’s native time resolution.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The agent derives this variable from the imaging frame rate and the trial length in frames, not from the behavior timestamps. It uses the number of samples in the trial and constructs a synthetic elapsed-time vector.

ii.
```python
imaging_rate = f['acquisition/TwoPhotonSeries/imaging_plane/imaging_rate'][()]
...
frame_time = 1.0 / imaging_rate  # seconds per frame
time_from_start = np.arange(n_timepoints) * frame_time
```

iii. There is no separate trajectory note defending this choice. The surrounding reasoning shows the agent treated the data streams as already aligned at a fixed imaging rate and then generated trial-relative time from frame count.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The code builds a regularly spaced vector starting at 0 with step `1 / imaging_rate`, one value per frame in the trial, and writes it into the first input row.

ii.
```python
time_from_start = np.arange(n_timepoints) * frame_time
...
input_data[0, :] = time_from_start
```

iii. The trajectory does not show a deeper justification here; the implemented assumption is simply that equally spaced imaging frames are an adequate clock for trial-relative time.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time vector is aligned by construction: it is created with exactly `n_timepoints = e - s` entries for the same `[s:e)` trial slice used for the neural matrix.

ii.
```python
s, e = tstart_inds[i], teleport_inds[i]
n_timepoints = e - s
neural = deconv_filtered[s:e, :].T.astype(np.float32)
time_from_start = np.arange(n_timepoints) * frame_time
```

iii. The agent’s reasoning implicitly assumes that once trial boundaries are shared, the streams are aligned. Its trajectory emphasizes matched trial slicing and a fixed imaging cadence rather than timestamp-based re-alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the NWB behavior `environment` time series.

ii.
```python
env = f['processing/behavior/BehavioralTimeSeries/environment/data'][:]
```

iii. The trajectory explicitly states “Environment encoding: 0=ENV1, 1=ENV2 directly,” showing that the agent treated the NWB `environment` values as the source of this input.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The agent collapses each trial’s environment values to a single per-trial label by taking the median of nonnegative entries in that trial, defaulting to `0` if no valid value exists. It then broadcasts that per-trial value across all timepoints in the trial.

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
input_data[1, :] = env_type
```

iii. The trajectory only gives a brief justification: the agent concluded the encoding is direct (`0=ENV1`, `1=ENV2`) and the decoder specification called this a per-trial variable, so it reduced each trial to one label.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the trial loop index after trial segmentation, not from any NWB `trial number` time series.

ii.
```python
for i in range(n_trials):
    ...
    trial_number = float(i + 1)
```

iii. The trajectory does not separately justify this. The code shows the agent treated “trial number” as the ordinal number of the segmented trial within a session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index is converted to a 1-based float (`i + 1`) and broadcast across the entire trial.

ii.
```python
trial_number = float(i + 1)
...
input_data[2, :] = trial_number
```

iii. No explicit trajectory discussion was recorded beyond the general plan to provide the requested per-trial decoder input.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the reward event timestamps in the NWB `Reward` time series. The agent first computes a per-trial reward outcome and then shifts it back by one trial.

ii.
```python
reward_ts = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
...
reward_outcomes = determine_reward_outcome(reward_ts, timestamps, tstart_inds, teleport_inds)
```

iii. In the trajectory, the agent explicitly paused to “check the reward outcome and understand the Reward field better,” and inspected how many rewards landed inside example trial windows before implementing this logic.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The first trial gets `0.0`. Every later trial gets the binary reward outcome from the immediately previous trial, then that scalar is broadcast across all timepoints in the current trial.

ii.
```python
if i == 0:
    prev_outcome = 0.0
else:
    prev_outcome = float(reward_outcomes[i - 1])
...
input_data[3, :] = prev_outcome
```

iii. The trajectory shows the agent had already decided reward outcome should be computed per trial from reward timestamps, so using the prior trial’s binary result for this input followed directly from that design.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the `position` behavior time series plus a per-trial reward-zone label inferred from the `reward_zone` behavior signal and the positions where `reward_zone > 0`.

ii.
```python
pos = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
rzone = f['processing/behavior/BehavioralTimeSeries/reward_zone/data'][:]
...
zone_labels = determine_reward_zone_per_trial(pos, rzone, tstart_inds, teleport_inds)
```
```python
trial_pos = pos[s:e]
rz_label = zone_labels[i]
rz_start, rz_end = REWARD_ZONES[rz_label]
dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The trajectory shows the agent inspecting reward-zone positions and concluding that the reward-zone detection “works,” including a visible C-to-A switch at the expected trial boundary. That inspection drove its choice to infer zone identity directly from `reward_zone` activity and position.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each trial, the agent infers a reward-zone label, falls back to `'A'` if none is available, and computes signed distance from position to the nearest reward-zone edge: negative before the zone, zero inside it, positive after it.

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
```python
rz_label = zone_labels[i]
if rz_label is None:
    rz_label = 'A'
rz_start, rz_end = REWARD_ZONES[rz_label]
dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The trajectory says the agent believed the direct reward-zone detection behaved as expected on an example switch session. It did not mention the human reference’s Viterbi segmentation; instead it trusted its simpler per-trial inference plus neighbor filling.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous signed distance is binned into 7 categories with manual threshold comparisons implementing the requested decoder bins.

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

iii. The trajectory does not add extra justification beyond trying to satisfy the task specification. The thresholds in code directly mirror the decoder instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by using the same per-trial `[s:e)` slice of position samples as the neural data slice, then writing the discretized result into a trial matrix of the same width.

ii.
```python
neural = deconv_filtered[s:e, :].T.astype(np.float32)
trial_pos = pos[s:e]
...
output_data[0, :] = dist_disc
```

iii. The agent’s general alignment strategy throughout the script is to treat shared trial indices as sufficient alignment. That same assumption is used here.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the `position` behavior time series.

ii.
```python
pos = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
...
trial_pos = pos[s:e]
```

iii. The trajectory treats position as one of the core behavioral streams already aligned to the imaging frames and uses it directly in downstream outputs.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The code takes the per-trial position slice, clips it to the nominal track range `[0, 450]`, then discretizes that clipped position.

ii.
```python
pos_clipped = np.clip(trial_pos, 0, TRACK_LENGTH)
pos_disc = discretize_position(pos_clipped, n_bins=5)
```

iii. The agent’s converter hard-codes the 450 cm track length from the task description and uses clipping to keep values within that range. There is no separate trajectory note justifying the clip itself.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The clipped position is divided into five 90 cm bins by computing `floor(position / 90)` and clipping the result to the integer range `0` to `4`.

ii.
```python
def discretize_position(position, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins
    bins = np.clip(np.floor(position / bin_size).astype(int), 0, n_bins - 1)
    return bins
```

iii. The trajectory does not separately discuss this choice. The implementation follows the requested 5 equal-sized bins but realizes them with floor-and-clip logic rather than open-ended `np.digitize` bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned by taking position from the same trial slice `[s:e)` as the neural data and storing a category for each frame in that slice.

ii.
```python
neural = deconv_filtered[s:e, :].T.astype(np.float32)
trial_pos = pos[s:e]
...
output_data[1, :] = pos_disc
```

iii. As elsewhere, the agent relied on common frame indices within each segmented trial as its alignment method.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the `lick` behavior time series.

ii.
```python
lick = f['processing/behavior/BehavioralTimeSeries/lick/data'][:]
...
trial_lick = lick_binary[s:e]
```

iii. The trajectory lists `lick` among the core behavioral streams discovered in the NWB files and later adds a sensor-correction step based on the paper’s code comments.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The agent first applies a lick-sensor error correction: if more than 35% of samples in a trial have lick values above 2, the entire trial’s lick signal is zeroed. After that, any remaining positive lick value is binarized to 1.

ii.
```python
def correct_lick_sensor_error(licks, tstart_inds, teleport_inds, correction_thr=0.35):
    ...
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

iii. The script comments cite `behavior.py: correction_thr=0.35`, and the trajectory later reports the number of “lick error trials” per session, showing the agent deliberately carried this correction into the conversion.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned by slicing the corrected lick array with the same `[s:e)` trial window as the neural data.

ii.
```python
neural = deconv_filtered[s:e, :].T.astype(np.float32)
trial_lick = lick_binary[s:e]
...
output_data[3, :] = lick_disc
```

iii. The agent did not perform any additional temporal interpolation or timestamp matching for lick; it assumed shared frame indices implied alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the `reward_zone` time series together with `position`, by inferring which spatial reward zone was active on each trial.

ii.
```python
pos = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
rzone = f['processing/behavior/BehavioralTimeSeries/reward_zone/data'][:]
zone_labels = determine_reward_zone_per_trial(pos, rzone, tstart_inds, teleport_inds)
```

iii. The trajectory shows the agent testing reward-zone positions against expected zone ranges and concluding the inferred switches looked correct.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The agent marks each trial’s `rzone > 0` samples, takes the median position of those samples, assigns zone `A/B/C` if the median falls near one of the expected reward-zone ranges, then forward-fills and backward-fills missing trials from neighbors. The final categorical output is encoded as `0=A`, `1=B`, `2=C`.

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
```
```python
for i in range(n_trials):
    if zone_labels[i] is not None:
        last_label = zone_labels[i]
    elif last_label is not None:
        zone_labels[i] = last_label
...
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_label]
```

iii. The trajectory explicitly says “the reward zone detection works” after visual inspection of an expected zone switch. That inspection is the main stated justification for this heuristic.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the `Reward` event timestamps in the NWB behavior data, together with trial start and end timestamps.

ii.
```python
reward_ts = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
...
reward_outcomes = determine_reward_outcome(reward_ts, timestamps, tstart_inds, teleport_inds)
```

iii. The trajectory includes an explicit inspection of the `Reward` field, where the agent checked that reward events were sparse, had constant amplitude, and landed within expected trial windows.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the code compares reward timestamps to the trial’s start and end timestamps, sets the trial outcome to 1 if any reward event falls in that time window, and broadcasts that scalar across the trial in the output array.

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
rew_out = int(reward_outcomes[i])
output_data[5, :] = rew_out
```

iii. After the reward-field inspection in the trajectory, the agent chose direct timestamp-window tests rather than mapping rewards onto behavior bins first.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code crops neural and behavioral streams to a shared minimum length to handle occasional off-by-one mismatches. It silently truncates mismatched start/end trial lists, drops invalid `teleport <= start` trial pairs, skips sessions with too few valid trials or cells, zeroes lick-error trials, fills missing reward-zone labels from neighboring trials, and falls back to reward zone `A` if a trial still has no label.

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
tstart_inds = tstart_inds[valid]
teleport_inds = teleport_inds[valid]
```
```python
if n_kept < 2:
    return None
...
if frac_high > correction_thr:
    licks_corrected[s:e] = 0
...
if rz_label is None:
    rz_label = 'A'
```

iii. The trajectory explicitly documents the off-by-one mismatch fix: “The neural data has one more timepoint than behavioral data. Let me fix this by truncating to the minimum length.” The other fallback behaviors are encoded in the script without much separate defense.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading large NWB arrays for every session, computing simplified dF/F and per-cell speed correlations for interneuron detection, looping over all trials to build trial matrices, and serializing the large pickle output.

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
```
```python
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. The trajectory shows the agent spending effort on end-to-end full conversion and explicitly debugging performance-relevant array-length issues. The main bottlenecks are evident from the script’s repeated full-session array loads and nested trial/cell loops.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The biggest vectorization candidates are the per-cell Pearson-correlation loop in `detect_interneurons`, the per-trial loops in `correct_lick_sensor_error`, `determine_reward_zone_per_trial`, `determine_reward_outcome`, and the final trial-construction loop in `convert_session`.

ii.
```python
for cell in range(n_cells):
    dff_valid = dff[valid_mask, cell]
    ...
    r, _ = stats.pearsonr(dff_valid, speed_valid)
```
```python
for i, (s, e) in enumerate(zip(tstart_inds, teleport_inds)):
    trial_licks = licks_corrected[s:e]
```
```python
for i in range(n_trials):
    s, e = tstart_inds[i], teleport_inds[i]
    ...
    neural_trials.append(neural)
```

iii. The trajectory does not contain an explicit efficiency analysis, but these loops dominate the code’s Python-level work and could be reduced with more session-level vectorization or batch trial handling.

## 13-c. What processing does the code repeat multiple times?

i. The code repeatedly scans the same trial boundaries for different purposes: dF/F estimation, interneuron detection, lick correction, reward-zone inference, reward-outcome inference, environment summarization, and final trial assembly. It also loads fluorescence and neuropil even though the final saved neural output comes from the deconvolved signal.

ii.
```python
dff = compute_dff_simple(fluor_all, neuro_all, tstart_inds, teleport_inds)
is_interneuron = detect_interneurons(dff, speed, tstart_inds, teleport_inds,
                                      threshold=INTERNEURON_SPEED_CORR_THRESHOLD)
lick_corrected, error_trials = correct_lick_sensor_error(
    lick, tstart_inds, teleport_inds, correction_thr=LICK_CORRECTION_THR)
zone_labels = determine_reward_zone_per_trial(pos, rzone, tstart_inds, teleport_inds)
reward_outcomes = determine_reward_outcome(reward_ts, timestamps, tstart_inds, teleport_inds)
```

iii. The trajectory focuses on getting a working full conversion rather than eliminating repeated passes. The final script reflects that: it makes several separate passes over the same segmented trials.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and retains several intermediate or reporting-only artifacts that do not enter the saved neural/input/output tensors: raw fluorescence and neuropil loads, simplified dF/F, interneuron flags, lick-error trial indices, per-trial zone labels, per-trial reward outcomes, and per-trial environment summaries. It also reads `reward_data` and `planeIdx` but never uses them.

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
return {
    'neural': neural_trials,
    'input': input_trials,
    'output': output_trials,
    'n_curated': n_curated,
    'n_interneurons': n_interneurons,
    'n_kept': n_kept,
    'n_trials': len(neural_trials),
    'imaging_rate': imaging_rate,
    'zone_labels': zone_labels,
    'reward_outcomes': reward_outcomes,
    'env_per_trial': env_per_trial,
    'lick_error_trials': error_trials,
}
```

iii. The trajectory makes clear that many of these computations were useful for validation and sanity checking. But most of them are discarded once the session’s final trial tensors are appended to the top-level dataset.
