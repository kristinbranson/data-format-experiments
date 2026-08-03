# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script enumerates every `sub-m*` directory under `data/`, treats every `.nwb` file as one session, and calls `process_session()` on each file. Inside each session it opens the NWB file with `h5py`, reads selected behavioral streams from `processing/behavior/BehavioralTimeSeries`, reads deconvolved neural activity from `processing/ophys/Deconvolved`, and then splits those continuous arrays into trials later in the function.

ii. ```python
for sub_dir in sorted(os.listdir(data_dir)):
    if not sub_dir.startswith('sub-'):
        continue
    subject_id = sub_dir.replace('sub-m', '')
    sub_path = os.path.join(data_dir, sub_dir)
    for fname in sorted(os.listdir(sub_path)):
        if not fname.endswith('.nwb'):
            continue
        ses_num = int(fname.split('_ses-')[1].split('_')[0])
        all_files.append({
            'path': os.path.join(sub_path, fname),
            'subject_id': subject_id,
            'session_num': ses_num,
        })
```

```python
f = h5py.File(nwb_path, 'r')
behav = f['processing']['behavior']['BehavioralTimeSeries']
ophys = f['processing']['ophys']

position = behav['position']['data'][()]
speed = behav['speed']['data'][()]
lick_raw = behav['lick']['data'][()]
trial_start_signal = behav['trial_start']['data'][()]
teleport_signal = behav['teleport']['data'][()]
reward_ts = behav['Reward']['timestamps'][()]
```

iii. `CONVERSION_NOTES.md` Step 2 says the dataset is organized as one NWB file per subject/session and lists the behavior and ophys streams the agent found. Step 5 says “All 152 sessions processed,” and Step 10 says the NWB loading was intended to match the reference `sess` object structure.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the folder name `sub-mXX`. The code strips `sub-m` to get `subject_id`, converts it back to `mXX` when storing metadata, and builds `subjects` plus `subject_idx` in first-seen order across sessions.

ii. ```python
subject_id = sub_dir.replace('sub-m', '')
```

```python
sub_name = f'm{subject_id}'
if sub_name not in subject_map:
    subject_map[sub_name] = len(subjects)
    subjects.append(sub_name)

subject_idx.append(subject_map[sub_name])
```

iii. Step 1 of `CONVERSION_NOTES.md` explicitly notes the “GCAMP_N -> m_N naming convention,” and Step 9 reports 11 subjects after processing all files.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The session number is parsed from the filename (`ses-XX`) and is also used to look up the session scene in the hard-coded `sessions_dict` copy.

ii. ```python
ses_num = int(fname.split('_ses-')[1].split('_')[0])
all_files.append({
    'path': os.path.join(sub_path, fname),
    'subject_id': subject_id,
    'session_num': ses_num,
})
```

```python
scene = get_scene_for_session(subject_id, session_num)
```

iii. `CONVERSION_NOTES.md` Step 2 says “NWB files: `data/sub-mX/sub-mX_ses-YY_behavior+ophys.nwb`.” The trajectory also shows the agent fixing a bad manual `sessions_dict` copy and re-generating it from the reference `sessions_dict` before finalizing the conversion.

## 1-d. How are the data split into trials?

i. Trials are defined from each `trial_start` pulse to the next `teleport` pulse. The code finds all indices where those binary streams are positive, pairs each start with the next teleport after it, and then slices each trial as `start:stop` (start inclusive, teleport exclusive).

ii. ```python
trial_start_inds = np.where(trial_start_signal > 0)[0]
teleport_inds = np.where(teleport_signal > 0)[0]
```

```python
matched_starts = []
matched_teleports = []

for i in range(len(trial_start_inds)):
    start = trial_start_inds[i]
    future_teleports = teleport_inds[teleport_inds > start]
    if len(future_teleports) > 0:
        matched_starts.append(start)
        matched_teleports.append(future_teleports[0])
```

```python
start = trial_start_inds[i]
stop = teleport_inds[i]
trial_len = stop - start
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 says “Trial boundaries: trial_start to teleport signals,” and Step 10 says the agent believed this matched the reference code’s `trial_start_inds` and `teleport_inds`.

## 1-e. How are trials filtered based on quality controls?

i. The script does very little trial-level filtering. It skips sessions with fewer than 2 matched trials, skips individual trials shorter than 3 samples, and otherwise keeps trials even if they have lick-sensor problems. For bad lick-sensor trials it does not drop the trial; it keeps the trial and sets the lick output to all zeros.

ii. ```python
if n_trials < 2:
    print(f"  Skipping: only {n_trials} trials")
    return None
```

```python
trial_len = stop - start
if trial_len < 3:
    continue
```

```python
if np.sum(trial_lick > 2) / len(trial_lick) > 0.35:
    trial_lick[:] = 0
```

iii. In `CONVERSION_NOTES.md`, Step 1 identifies lick-sensor error correction from the reference behavior code, and Step 4 says the agent chose the code threshold `0.35`. The notes do not describe any broader trial rejection policy beyond this.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is taken from the NWB deconvolved calcium activity arrays plus the `iscell` ROI-classification array. For multi-plane recordings, all deconvolved planes are concatenated column-wise before cell filtering.

ii. ```python
planes = sorted(ophys['Deconvolved'].keys())
deconv_parts = []
for plane in planes:
    deconv_parts.append(ophys['Deconvolved'][plane]['data'][()])
deconv = np.concatenate(deconv_parts, axis=1)
iscell = ophys['ImageSegmentation']['PlaneSegmentation']['iscell'][()]
```

iii. `CONVERSION_NOTES.md` Step 2 lists “Deconvolved events” and `iscell` as the relevant ophys datasets. Step 5 says “Use deconvolved events from NWB (matches reference code `sess.timeseries['events']`).”

## 2-b. How is the `neural` data processed?

i. The script concatenates planes if needed, filters ROIs by `iscell[:, 0] == 1`, slices the continuous deconvolved array into trials, transposes each trial to neuron-by-time format, and casts to `float32`. It does not recompute dF/F or deconvolution.

ii. ```python
cell_mask = iscell[:, 0] == 1
neural_all = deconv[:, cell_mask]
```

```python
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
```

iii. Step 5 of `CONVERSION_NOTES.md` lists the key decision: “Use deconvolved events from NWB.” The trajectory also shows the agent concluding that the NWB `Deconvolved` arrays already correspond to the reference `sess.timeseries['events']`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural QC is limited to `iscell[:, 0] == 1` and an extra session-level minimum of 5 retained cells. The code does not implement the methods-described interneuron exclusion based on high correlation between dF/F and running speed.

ii. ```python
cell_mask = iscell[:, 0] == 1
n_cells = cell_mask.sum()

if n_cells < 5:
    print(f"  Skipping: only {n_cells} cells")
    return None
```

iii. `CONVERSION_NOTES.md` Step 3 notes the paper’s additional interneuron exclusion, but Step 5 reduces the curation rule to “iscell=1,” and trajectory step 104 explicitly says the speed-correlation-based interneuron exclusion “was not implemented.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each neural trial begins at the detected `trial_start` index, so time 0 for the per-trial neural matrices is the start of trial.

ii. ```python
start = trial_start_inds[i]
stop = teleport_inds[i]
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 3 states “Temporal alignment: Align to trial start,” and Step 10 says the temporal alignment was intended to match the reference code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the native imaging frame rate. The code reads `imaging_rate`, sets `dt = 1.0 / imaging_rate`, and uses that directly; no temporal rebinning or resampling is applied.

ii. ```python
imaging_rate = f['general']['optophysiology']['ImagingPlane']['imaging_rate'][()]
dt = 1.0 / imaging_rate
```

```python
dt_ms = 1000.0 / imaging_rate
...
'time_bin_size': dt_ms,
```

iii. `CONVERSION_NOTES.md` Step 5 says “Time bin: native imaging rate (~64.5 ms)” and “No additional binning.”

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the session imaging rate plus the matched trial boundaries. The code uses the number of samples in the current trial (`trial_len = stop - start`) and the per-frame duration `dt = 1 / imaging_rate`.

ii. ```python
imaging_rate = f['general']['optophysiology']['ImagingPlane']['imaging_rate'][()]
dt = 1.0 / imaging_rate
```

```python
trial_len = stop - start
time_from_start = np.arange(trial_len) * dt
```

iii. `CONVERSION_NOTES.md` Step 5 maps this input as “frame_index * dt,” and Step 10 says the agent spot-checked this against the original NWB-derived timing.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. It is a simple per-trial ramp that starts at 0 and increases by one frame duration for each sample in the trial.

ii. ```python
time_from_start = np.arange(trial_len) * dt
trial_input[0, :] = time_from_start
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly planned this as “frame_index * dt,” and Step 10 reports an “Input time check” that the agent said was an exact match.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses exactly the same `trial_len` and `start:stop` boundaries as the neural slice, and is stored as row 0 of the same per-trial input matrix.

ii. ```python
trial_neural = neural_all[start:stop, :].T.astype(np.float32)
time_from_start = np.arange(trial_len) * dt

trial_input = np.zeros((4, trial_len), dtype=np.float32)
trial_input[0, :] = time_from_start
```

iii. The justification is the same as for trial alignment generally: `CONVERSION_NOTES.md` Step 3 says align to trial start, and Step 10 says the time input was checked against the converted trial boundaries.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is not derived from the NWB `environment` stream even though that array is loaded. Instead, the script derives environment type from the per-session scene label in the copied `sessions_dict`.

ii. ```python
env_data = behav['environment']['data'][()]
...
scene = get_scene_for_session(subject_id, session_num)
env_per_trial = get_environment_from_scene(scene, n_trials)
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly states “Environment (0/1) From scene name,” and the trajectory shows the agent reasoning that scene metadata was needed to recover task condition and reward-zone identity.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The script parses whether the scene contains `Env1` or `Env2`. If the scene indicates a cross-environment switch (`'_to_Env'`), it assigns one environment for the first 30 trials and the other for later trials; otherwise it uses one constant value for the whole session.

ii. ```python
def get_environment_from_scene(scene, n_trials, change_trial=CHANGE_TRIAL):
    env = np.zeros(n_trials, dtype=int)

    if '_to_Env' in scene:
        parts = scene.split('_to_')
        before_env = 0 if 'Env1' in parts[0] else 1
        after_env = 0 if 'Env1' in parts[1] else 1
        env[:change_trial] = before_env
        env[change_trial:] = after_env
    else:
        env[:] = 0 if 'Env1' in scene else 1
```

iii. `CONVERSION_NOTES.md` Step 3 and the copied methods text both note that environment switches happen at fixed trial counts during the task. The agent’s notes say it intentionally used scene metadata rather than the raw stream.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Although the NWB `trial number` array is loaded, the stored decoder input is actually derived from the loop index over matched trials. So it is based on the order produced by the detected `trial_start` and `teleport` events, not on the raw `trial number` values themselves.

ii. ```python
trial_num = behav['trial number']['data'][()]
...
for i in range(n_trials):
    ...
    trial_number = i
```

iii. `CONVERSION_NOTES.md` Step 5 says “Trial number: 0-indexed,” which matches the loop-index implementation rather than the raw per-frame `trial number` stream.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index `i` is taken as the per-trial number and then broadcast across every time bin of the trial as input row 2.

ii. ```python
trial_number = i
...
trial_input[2, :] = float(trial_number)
```

iii. The agent’s notes justify this only at a high level: Step 5 says the value should be “0-indexed” and per-trial.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `isreward`, which the script computes using the NWB reward timestamps together with the per-frame `reward_zone` activity inside each matched trial.

ii. ```python
reward_ts = behav['Reward']['timestamps'][()]
rzone_data = behav['reward_zone']['data'][()]
```

```python
trial_rewards = np.sum((reward_ts >= timestamps[start]) & (reward_ts <= timestamps[stop]))
rzone_active = np.any(rzone_data[start:stop+1] > 0)
isreward[i] = 1 if (trial_rewards > 0 and rzone_active) else 0
```

iii. The trajectory shows the agent reading the reference `get_trial_types(sess)` function, which uses reward presence plus reward-zone occupancy to define rewarded trials. Step 5 of `CONVERSION_NOTES.md` maps previous-trial outcome from reward outcome.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The first trial is assigned 0 by default. Every later trial gets the immediately preceding trial’s `isreward` value, and that scalar is broadcast across the full time axis of the current trial.

ii. ```python
prev_outcome = np.zeros(n_trials, dtype=int)
prev_outcome[1:] = isreward[:-1]
...
prev_out = prev_outcome[i]
trial_input[3, :] = float(prev_out)
```

iii. `CONVERSION_NOTES.md` Step 10 says the agent explicitly checked `prev_outcome[i] = reward_outcome[i-1]` and considered it correct.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the raw position stream plus reward-zone coordinates inferred from the session scene and the copied reward-zone dictionary.

ii. ```python
position = behav['position']['data'][()]
...
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
```

```python
trial_pos = position[start:stop]
rz_start = rz_coords[i, 0]
rz_end = rz_coords[i, 1]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. `CONVERSION_NOTES.md` Step 5 says “Distance to reward zone: Signed distance, 7 bins,” and the trajectory shows the agent using the reference `get_reward_zones(sess)` logic from `behavior.py` as the model for scene-based reward-zone lookup.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each position sample, the code computes a signed linear distance to the nearest part of the active reward zone: negative before the zone, zero inside it, positive after it.

ii. ```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    distance = np.zeros_like(position)
    before = position < rz_start
    distance[before] = position[before] - rz_start
    in_zone = (position >= rz_start) & (position <= rz_end)
    distance[in_zone] = 0
    after = position > rz_end
    distance[after] = position[after] - rz_end
    return distance
```

iii. Step 5 of `CONVERSION_NOTES.md` says this output is “Signed distance,” which is exactly the helper function’s behavior.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is discretized into the seven task-specified bins: `< -50`, `[-50, -10)`, `[-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50` cm.

ii. ```python
bins[distances < -50] = 0
bins[(distances >= -50) & (distances < -10)] = 1
bins[(distances >= -10) & (distances < 0)] = 2
bins[distances == 0] = 3
bins[(distances > 0) & (distances <= 10)] = 4
bins[(distances > 10) & (distances <= 50)] = 5
bins[distances > 50] = 6
```

iii. The bin edges come directly from the decoder-task instructions, and `CONVERSION_NOTES.md` Step 5 says the agent planned “7 bins” for this variable.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from `trial_pos = position[start:stop]`, so it has the same number of time bins and the same trial boundaries as the neural matrix for that trial.

ii. ```python
trial_pos = position[start:stop]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
dist_bins = discretize_distance_to_reward(dist_to_rz)
trial_output[0, :] = dist_bins
```

iii. `CONVERSION_NOTES.md` Step 10 says the agent’s spot checks compared converted trial variables against the original NWB trial segmentation and judged them aligned.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the per-frame raw `position` data within each matched trial.

ii. ```python
position = behav['position']['data'][()]
...
trial_pos = position[start:stop]
```

iii. `CONVERSION_NOTES.md` Step 5 maps “Absolute position” directly from the NWB position stream.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The position samples are clipped into the physical track range `[0, 450]` cm and then digitized into equal-width bins.

ii. ```python
pos_clipped = np.clip(trial_pos, 0, TRACK_LENGTH)
pos_bins = discretize_position(pos_clipped)
```

```python
bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
bins = np.digitize(positions, bin_edges[1:])
bins = np.clip(bins, 0, n_bins - 1)
```

iii. `CONVERSION_NOTES.md` Step 5 says this output should be “5 equal bins,” matching the implemented processing.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is discretized into 5 equal 90 cm bins over the 450 cm track.

ii. ```python
bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
```

```python
'output_values': [
    ...
    ['0-90cm', '90-180cm', '180-270cm', '270-360cm', '360-450cm'],
    ...
]
```

iii. The thresholds come from the task instruction “Absolute position in corridor, discretized into 5 equal-sized bins,” and Step 5 of the notes reflects that choice.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same `position[start:stop]` slice and is stored as a time-varying output with one value per neural time bin.

ii. ```python
trial_pos = position[start:stop]
pos_bins = discretize_position(pos_clipped)
trial_output[1, :] = pos_bins
```

iii. The agent’s Step 10 sanity checks say the position discretization matched the original trial extraction.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the raw per-frame NWB `lick` stream.

ii. ```python
lick_raw = behav['lick']['data'][()]
...
trial_lick = lick_raw[start:stop].copy()
```

iii. `CONVERSION_NOTES.md` Step 2 lists `lick` as one of the key behavioral series, and Step 5 maps it directly to the decoder output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The code first checks for a lick-sensor error trial using the fraction of samples with cumulative lick count `> 2`. If that fraction exceeds `0.35`, it sets the entire trial’s lick vector to zeros. Otherwise, it clips all counts above 1 down to 1 and casts the result to integer.

ii. ```python
if np.sum(trial_lick > 2) / len(trial_lick) > 0.35:
    trial_lick[:] = 0
else:
    trial_lick[trial_lick > 1] = 1
trial_lick = trial_lick.astype(int)
```

iii. The agent justified this using the reference lick QC it had read: `CONVERSION_NOTES.md` Step 1 notes “if >35% of samples in trial have cumulative lick >2,” and trajectory step 15 summarizes the reference behavior code as applying this rule plus binary clipping. The code changes the bad-trial handling from NaN to zero, but the agent’s stated justification was still the reference lick-sensor correction rule.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is sliced with the same `start:stop` trial boundaries as the neural data and stored as one value per neural time bin.

ii. ```python
trial_lick = lick_raw[start:stop].copy()
...
trial_output[3, :] = trial_lick
```

iii. `CONVERSION_NOTES.md` Step 10 says the lick processing was spot-checked against the NWB-derived trial extraction and judged an exact match.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the scene string in the copied `sessions_dict`, not from the NWB `reward_zone` time series. The script infers whether the active location is A, B, or C for each trial from the scene label and the fixed switch trial.

ii. ```python
scene = get_scene_for_session(subject_id, session_num)
rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)
```

```python
if rz_labels[i] == 'A':
    rz_label_idx[i] = 0
elif rz_labels[i] == 'B':
    rz_label_idx[i] = 1
elif rz_labels[i] == 'C':
    rz_label_idx[i] = 2
```

iii. `CONVERSION_NOTES.md` Step 5 says “Reward zone from `sessions_dict` scene mapping (imported from reference code),” and the trajectory shows the agent fixing a wrong manual scene mapping before rerunning the conversion.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For a non-switch session, one zone label is used for all trials. For a switch session, the code parses the scene string, assigns the “before” location to trials 0-29 and the “after” location to trials 30 onward, then maps A/B/C to 0/1/2.

ii. ```python
if '_to_' in scene or '_to_Env' in scene:
    parts = scene.split('_to_')
    before_loc = parts[0][-1]
    after_loc = parts[1][-1]
    ...
    rz_coords[:change_trial] = REWARD_ZONE_DICT[before_zone]
    rz_labels[:change_trial] = ZONE_TO_LABEL[before_zone]
    rz_coords[change_trial:] = REWARD_ZONE_DICT[after_zone]
    rz_labels[change_trial:] = ZONE_TO_LABEL[after_zone]
```

iii. This directly reflects the reference `behavior.get_reward_zones(sess)` logic the agent copied from the trajectory, and `CONVERSION_NOTES.md` Step 5 lists this mapping as a key decision.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the NWB reward timestamps and the per-frame `reward_zone` stream within each matched trial.

ii. ```python
reward_ts = behav['Reward']['timestamps'][()]
rzone_data = behav['reward_zone']['data'][()]
...
trial_rewards = np.sum((reward_ts >= timestamps[start]) & (reward_ts <= timestamps[stop]))
rzone_active = np.any(rzone_data[start:stop+1] > 0)
```

iii. The agent read the reference `get_trial_types(sess)` function, which defines `isreward` from reward delivery and reward-zone occupancy. `CONVERSION_NOTES.md` Step 5 maps reward outcome directly from that per-trial reward status.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is labeled rewarded if at least one reward timestamp falls between the trial start and trial end and the reward-zone signal is active somewhere in the same trial. That binary value is then broadcast across the full trial in output row 5.

ii. ```python
isreward[i] = 1 if (trial_rewards > 0 and rzone_active) else 0
...
rew_out = isreward[i]
trial_output[5, :] = rew_out
```

iii. `CONVERSION_NOTES.md` Step 10 says the agent checked reward-zone labels and previous-outcome propagation and considered the reward-outcome logic consistent with the task.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The script mostly does not repair missing or inconsistent data. It catches exceptions at the session level and skips failed sessions, skips sessions with too few cells or trials, skips very short trials, clips position and lick values locally, and zeroes lick-sensor-error trials instead of representing them as missing. Arrays such as `env_data`, `trial_num`, `reward_data`, and `autoreward` are loaded but never used to resolve inconsistencies.

ii. ```python
try:
    result = process_session(...)
except Exception as e:
    ...
    continue
```

```python
if n_cells < 5:
    return None
...
if n_trials < 2:
    return None
...
if trial_len < 3:
    continue
```

```python
if np.sum(trial_lick > 2) / len(trial_lick) > 0.35:
    trial_lick[:] = 0
```

iii. `CONVERSION_NOTES.md` Step 4 frames the main data “mistake” issue as lick-sensor errors, and trajectory step 104 explicitly notes that some reference filtering steps were left unimplemented.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive parts are repeated HDF5 reads of full behavioral and deconvolved arrays for every session, the Python trial loop inside `process_session()`, the final all-output concatenations used only for summary printing, and especially pickling the final 9.8 GB dataset.

ii. ```python
for i, file_info in enumerate(all_files):
    ...
    result = process_session(...)
```

```python
for i in range(n_trials):
    ...
    neural_trials.append(trial_neural)
    input_trials.append(trial_input)
    output_trials.append(trial_output)
```

```python
for out_idx, out_name in enumerate(data['output_names']):
    vals = []
    for sess in output_list:
        for trial in sess:
            vals.append(trial[out_idx, :])
    all_vals = np.concatenate(vals)
```

iii. `CONVERSION_NOTES.md` Step 7 estimates the runtime, and trajectory step 63 calls out the huge `converted_data.pkl` size as a major cost driver.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial reward computation loop, the reward-zone-label mapping loop, the start/teleport pairing loop, and much of the per-trial input/output construction could have been vectorized or at least reduced with array operations over all trials.

ii. ```python
for i in range(len(trial_start_inds)):
    start = trial_start_inds[i]
    future_teleports = teleport_inds[teleport_inds > start]
    ...
```

```python
for i in range(n_trials):
    start = trial_start_inds[i]
    stop = teleport_inds[i]
    ...
    isreward[i] = 1 if (trial_rewards > 0 and rzone_active) else 0
```

```python
for i in range(n_trials):
    if rz_labels[i] == 'A':
        rz_label_idx[i] = 0
    elif rz_labels[i] == 'B':
        rz_label_idx[i] = 1
    elif rz_labels[i] == 'C':
        rz_label_idx[i] = 2
```

iii. This is an inference from the code rather than an explicit self-justification in the notes. The code structure in `process_session()` makes these vectorization opportunities clear.

## 13-c. What processing does the code repeat multiple times?

i. It traverses the trial boundaries multiple times: once to match starts and teleports, once to compute `isreward`, and once again to build the actual per-trial neural/input/output arrays. It also re-concatenates all outputs after conversion just to print dataset-wide histograms.

ii. ```python
for i in range(len(trial_start_inds)):
    ...
```

```python
for i in range(n_trials):
    ...
    isreward[i] = ...
```

```python
for i in range(n_trials):
    ...
    neural_trials.append(trial_neural)
    input_trials.append(trial_input)
    output_trials.append(trial_output)
```

iii. This repeated traversal is visible directly in `convert_data.py`; the notes do not present it as an intended optimization.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several raw arrays are loaded and then never used (`trial_num`, `env_data`, `scanning`, `reward_data`, `autoreward`). The code also creates an initial 1D `trial_input` array and immediately overwrites it with a 2D array, allocates `all_outputs = {}` without using it, and recomputes full-dataset output summaries that are not stored in the final pickle.

ii. ```python
trial_num = behav['trial number']['data'][()]
env_data = behav['environment']['data'][()]
scanning = behav['scanning']['data'][()]
reward_data = behav['Reward']['data'][()]
autoreward = behav['autoreward']['data'][()]
```

```python
trial_input = np.array([time_from_start[0], float(env_type), float(trial_number), float(prev_out)], dtype=np.float32)
trial_input = np.zeros((4, trial_len), dtype=np.float32)
```

```python
all_outputs = {}
for out_idx, out_name in enumerate(data['output_names']):
    ...
```

iii. This is again an inference from the code itself. The trajectory’s late-stage runtime and file-size discussion suggests the agent noticed size/performance issues, but these unused loads and overwritten allocations remained in the final script.
