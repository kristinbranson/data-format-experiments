# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `h5py` to open each NWB file individually. It finds all NWB files using `glob.glob(os.path.join(data_dir, 'sub-*', '*.nwb'))`, which matches the directory structure `data/sub-{mouse}/sub-{mouse}_ses-{session}_behavior+ophys.nwb`. Each file is processed by `process_session()`, and results are accumulated into lists. In `--sample` mode, only 2 sessions are processed; in `--full` mode (default), all 152 sessions are processed.

ii.
```python
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*', '*.nwb')))
# ...
for nwb_path in nwb_files:
    result = process_session(nwb_path, ...)
    if result is not None:
        all_sessions.append(result)
```

iii. The AI documented that NWB files are organized as `data/sub-{mouse}/sub-{mouse}_ses-{session}_behavior+ophys.nwb` and found 152 total sessions across 11 subjects (CONVERSION_NOTES Step 2).

## 1-b. How are the data split into subjects (mice)?

i. Each NWB file contains a `general/subject/subject_id` field identifying the mouse. The AI reads this per-session and collects unique subjects. Subject indices are assigned based on sorted order of unique subject IDs.

ii.
```python
subject_id = f['general/subject/subject_id'][()].decode()
# ...
subjects = sorted(subjects_set)
subject_idx.append(subjects.index(sess['subject_id']))
```

iii. The AI noted 11 subjects (m3, m4, m7, m11-m15, m17-m19) in CONVERSION_NOTES Step 2, matching the paper's "n = 11 mice" for the switch group.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The AI treats each file as a separate session. Sessions are identified by a combination of subject_id, session_id, and scene name extracted from the identifier field.

ii.
```python
session_id = f['general/session_id'][()].decode()
identifier = f['identifier'][()].decode()
scene = get_scene_from_identifier(identifier)
# ...
'session_info': f"{subject_id}_ses-{session_id}_{scene}",
```

iii. The AI documented 152 total sessions (14 per subject, 12 for m11) matching the paper.

## 1-d. How are the data split into trials?

i. Trials are identified by the `trial_start` and `teleport` binary flag signals in the behavioral timeseries. The AI finds indices where `trial_start_flag > 0` and `teleport_flag > 0`, pairing them to define trial boundaries. Each trial spans from the trial_start index to the teleport index.

ii.
```python
trial_start_inds = np.where(trial_start_flag > 0)[0]
teleport_inds = np.where(teleport_flag > 0)[0]
n_trials = min(len(trial_start_inds), len(teleport_inds))
trial_start_inds = trial_start_inds[:n_trials]
teleport_inds = teleport_inds[:n_trials]
# ...
for i in range(n_trials):
    s = trial_start_inds[i]
    e = teleport_inds[i]
```

iii. The AI noted that trial boundaries are defined by `trial_start` and `teleport` flags in the behavioral timeseries, consistent with the reference code's `start-1` and `stop-1` indexing.

## 1-e. How are trials filtered based on quality controls?

i. Trials are only excluded if they have fewer than 2 timepoints (`e - s < 2`). Sessions are excluded if they have fewer than 2 valid trials or fewer than 2 cells after filtering. Lick sensor error correction is applied per trial (setting lick values to 0 if >35% of samples have lick>2), but this does not remove trials. No speed-based trial filtering is applied.

ii.
```python
if e <= s or (e - s) < 2:
    # Skip trial but track prev_trial_rewarded
    continue
# ...
if valid_trial_count < 2:
    print(f"  Skipping {session_label}: only {valid_trial_count} valid trials")
    return None
```

iii. The AI documented that lick sensor errors (>35% threshold) are handled by zeroing lick data rather than removing trials. The AI decided not to apply speed thresholding because speed is a decoder output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the pre-computed deconvolved calcium events stored in the NWB files at `processing/ophys/Deconvolved/plane{N}/data`. This data was originally computed by the suite2p OASIS deconvolution algorithm.

ii.
```python
deconv_data = ophys[f'Deconvolved/{plane_name}/data'][:]
# ...
deconv_all = np.concatenate(deconv_list, axis=1)
# ...
deconv_cells = deconv_all[:, cell_mask].T  # (n_cells, n_timepoints)
# ...
neural_data = deconv_cells[final_cell_mask]  # (n_final_cells, n_timepoints)
```

iii. The AI justified this by noting that "The paper's decoder uses deconvolved events" and "NWB has them pre-computed" (CONVERSION_NOTES Steps 4 and 5). The AI also loaded raw fluorescence (F) and neuropil (Fneu) data for computing dF/F, which is needed only for interneuron detection.

## 2-b. How is the `neural` data processed?

i. The deconvolved events are used directly from the NWB files with no additional processing beyond filtering. The raw fluorescence and neuropil data are loaded separately to compute dF/F for interneuron detection, but dF/F is not used as the neural data itself.

ii.
```python
# Deconvolved events used directly
neural_data = deconv_cells[final_cell_mask]
# ...
trial_neural = neural_data[:, s:e]  # (n_cells, n_timepoints)
neural_trials.append(trial_neural.astype(np.float32))
```

iii. The AI documented that the NWB files contain pre-computed deconvolved events, and the decoder in the paper uses deconvolved events (CONVERSION_NOTES Step 1).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filtering steps are applied:
1. **iscell filter**: Only ROIs where `iscell[:, 0] == 1` (suite2p cell classification with manual curation) are kept.
2. **Interneuron exclusion**: Putative interneurons are detected by computing Pearson correlation between dF/F and running speed. Cells with correlation > 0.5 are excluded.

ii.
```python
# iscell filter
cell_mask = iscell[:, 0].astype(bool)
# ...
# Interneuron detection via speed correlation
is_interneuron = detect_interneurons(dff_full, speed)
final_cell_mask = ~is_interneuron
neural_data = deconv_cells[final_cell_mask]
```
For dF/F used in interneuron detection:
```python
def compute_dff_trial(F_trial, Fneu_trial, baseline_window=BASELINE_WINDOW):
    F_corr = F_trial - NEUROPIL_COEF * Fneu_trial  # NEUROPIL_COEF = 0.7
    smoothed = gaussian_filter1d(F_corr, sigma=15, axis=1)
    baseline = minimum_filter1d(smoothed, size=min(baseline_window, n_t), axis=1)
    baseline = maximum_filter1d(baseline, size=min(baseline_window, n_t), axis=1)
    dff = (F_corr - baseline) / abs_baseline
    dff = gaussian_filter1d(dff, sigma=DFF_SMOOTH_SIGMA, axis=1)  # sigma=2
    return dff
```

iii. The AI documented the interneuron exclusion threshold of 0.5 from the reference code, and noted ~0.42+/-0.85% interneurons per session from the paper (CONVERSION_NOTES Step 3).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of each trial. Each trial's neural data spans from the `trial_start` index to the `teleport` index, so time 0 corresponds to the trial start (entry to the linear track). The instruction specifies "Temporally align based on start of the trial."

ii.
```python
for i in range(n_trials):
    s = trial_start_inds[i]
    e = teleport_inds[i]
    trial_neural = neural_data[:, s:e]
```

iii. The AI documented this as "Trial alignment = trial start: As specified in Decoder Task section" (CONVERSION_NOTES Step 5).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native imaging frame rate (~15.5 Hz for single-plane, ~31 Hz total for multi-plane with effective ~15.5 Hz per plane). This corresponds to ~64.5 ms per time bin. No temporal rebinning is applied.

ii.
```python
n_planes = len(fluor_planes)
effective_rate = imaging_rate if n_planes == 1 else imaging_rate / n_planes
time_bin_ms = 1000.0 / effective_rate
```

iii. The AI documented "Time bin = imaging frame: ~64.5 ms per frame at ~15.5 Hz. This matches the native sampling rate." (CONVERSION_NOTES Step 5).

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. This is computed from the trial index position relative to the trial start, using the effective imaging rate. It is not directly read from any raw variable but is derived from the frame index within the trial and the imaging rate.

ii.
```python
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
```

iii. The AI's mapping plan listed: "Time from trial start: (t - trial_start) / imaging_rate, continuous seconds" (CONVERSION_NOTES Step 5).

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, an array of length `n_t` (number of timepoints in the trial) is created where each element is the frame index divided by the effective imaging rate, yielding seconds from trial start.

ii.
```python
n_t = e - s
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
```

iii. No special processing beyond dividing frame indices by the imaging rate.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both the neural data and the time-from-start input use the same frame indexing. The neural data for the trial spans frames `s:e` and the time input covers `n_t = e - s` frames starting from 0. They are inherently aligned because they share the same time axis.

ii.
```python
trial_neural = neural_data[:, s:e]
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
input_arr[0, :] = time_from_start
```

iii. The AI ensured alignment by using the same trial boundaries for both neural and input data.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavioral timeseries in the NWB file (`processing/behavior/BehavioralTimeSeries/environment/data`).

ii.
```python
environment = bts['environment/data'][:]
# ...
trial_env = environment[s:e]
env_val = np.median(trial_env[trial_env >= 0]) if np.any(trial_env >= 0) else 0.0
env_type = np.float32(env_val)
```

iii. The AI documented that environment type is "0=ENV1, 1=ENV2, per-trial scalar" from the `environment` behavioral timeseries (CONVERSION_NOTES Step 5).

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The environment values within a trial are filtered to keep only non-negative values (environment can be -1 during inter-trial intervals), then the median of valid values is taken as the per-trial environment type. This is broadcast across all timepoints in the trial.

ii.
```python
env_val = np.median(trial_env[trial_env >= 0]) if np.any(trial_env >= 0) else 0.0
env_type = np.float32(env_val)
input_arr[1, :] = env_type  # broadcast
```

iii. The AI treated environment as a per-trial binary (0 or 1) scalar broadcast across timepoints.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The AI uses the loop index `i` (0-indexed trial number within the session) rather than reading the `trial number` field from the behavioral timeseries.

ii.
```python
trial_number = np.float32(i)
input_arr[2, :] = trial_number  # broadcast
```

iii. The AI's mapping states "Trial number within session: 0-indexed, per-trial scalar" (CONVERSION_NOTES Step 5).

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial number is simply the 0-based loop index, cast to float32 and broadcast across all timepoints in the trial.

ii.
```python
trial_number = np.float32(i)
input_arr[2, :] = trial_number
```

iii. No complex processing. The AI chose to use the loop index rather than the NWB `trial number` data field, which contains the same values (verified: both are 0-indexed).

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps and data in the NWB behavioral timeseries, combined with the `position/timestamps` for determining trial time boundaries.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
# ...
def detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time):
    if len(reward_timestamps) == 0:
        return False
    return np.any((reward_timestamps >= trial_start_time) & (reward_timestamps <= trial_end_time))
```

iii. The AI checks whether any reward event timestamp falls within the previous trial's time window. The first trial defaults to `prev_trial_rewarded = 0`.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the AI checks whether any reward timestamp falls within the trial's time window (from position timestamps at trial start to trial end). The result (0 or 1) is stored and used as `previous_trial_outcome` for the next trial. The first trial's previous outcome is set to 0.

ii.
```python
prev_trial_rewarded = 0  # For the first trial
# ...
was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
# ...
prev_outcome = np.float32(prev_trial_rewarded)
input_arr[3, :] = prev_outcome
# ...
prev_trial_rewarded = int(was_rewarded)
```

iii. The AI noted ~85% reward rate and ~15% omission rate, matching the paper's description (CONVERSION_NOTES Step 9).

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from two sources:
1. The `position` behavioral timeseries (animal's position on the track)
2. The reward zone coordinates, determined from the scene name in the NWB `identifier` field and the trial index (for sessions with reward zone switches)

ii.
```python
trial_pos = position[s:e]
rz_start, rz_end = rz_coords[i]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The AI documented the reward zone coordinates as A=[80,130], B=[200,250], C=[320,370] matching the paper and reference code (CONVERSION_NOTES Steps 1, 3).

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed as:
- Before reward zone: `position - rz_start` (negative)
- Inside reward zone: 0
- After reward zone: `position - rz_end` (positive)

The reward zone location for each trial is determined by parsing the scene name and using the switch trial logic (trial 30 for within-session switches).

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    dist = np.zeros_like(position)
    before = position < rz_start
    inside = (position >= rz_start) & (position <= rz_end)
    after = position > rz_end
    dist[before] = position[before] - rz_start  # negative
    dist[inside] = 0.0
    dist[after] = position[after] - rz_end  # positive
    return dist
```

iii. The AI documented the distance computation and reward zone mapping in CONVERSION_NOTES Step 5.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Distance is discretized into 7 bins matching the instructions:
- 0: < -50 cm
- 1: -50 to -10 cm
- 2: -10 to < 0 cm
- 3: 0 cm (inside reward zone)
- 4: >0 to +10 cm
- 5: +10 to +50 cm
- 6: > +50 cm

ii.
```python
def discretize_distance_to_rz(dist):
    out = np.zeros(len(dist), dtype=np.int64)
    out[dist < -50] = 0
    out[(dist >= -50) & (dist < -10)] = 1
    out[(dist >= -10) & (dist < 0)] = 2
    out[dist == 0] = 3
    out[(dist > 0) & (dist <= 10)] = 4
    out[(dist > 10) & (dist <= 50)] = 5
    out[dist > 50] = 6
    return out
```

iii. The AI's discretization matches the instruction specifications exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The distance to reward zone is computed from the position data at the same frame indices as the neural data (`s:e`), so they share the same time alignment.

ii.
```python
trial_pos = position[s:e]
# ...
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
output_arr[0, :] = dist_to_rz_binned
```

iii. Alignment is inherent from using the same trial boundary indices for both neural and behavioral data.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral timeseries in the NWB file.

ii.
```python
trial_pos = position[s:e]
pos_binned = discretize_position(trial_pos)
```

iii. The AI documented this as coming from the `position` behavioral timeseries.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The raw position values (0-450 cm) are discretized into 5 equal-sized bins using `np.digitize` with edges at [0, 90, 180, 270, 360, 450].

ii.
```python
def discretize_position(position, n_bins=POSITION_BINS):
    bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
    binned = np.digitize(position, bin_edges) - 1
    binned = np.clip(binned, 0, n_bins - 1)
    return binned
```

iii. The AI used 5 equal bins of 90 cm each over the 450 cm track, matching the instruction for "5 equal-sized bins."

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is discretized into 5 bins:
- 0: 0-90 cm
- 1: 90-180 cm
- 2: 180-270 cm
- 3: 270-360 cm
- 4: 360-450 cm

ii.
```python
POSITION_BINS = 5
POSITION_BIN_EDGES = np.linspace(0, TRACK_LENGTH, POSITION_BINS + 1)
# results in [0, 90, 180, 270, 360, 450]
```

iii. The AI documented the 5-bin discretization with labels '0-90cm', '90-180cm', etc.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position data uses the same frame indices as neural data (both extracted from the same `s:e` range), so they are inherently aligned.

ii.
```python
trial_pos = position[s:e]
trial_neural = neural_data[:, s:e]
```

iii. Same alignment mechanism as all other variables.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral timeseries in the NWB file.

ii.
```python
lick_raw = bts['lick/data'][:]
# ...
trial_lick = lick[s:e].copy()
```

iii. The AI noted that lick data contains cumulative lick counts per frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Processing involves two steps:
1. **Lick sensor error correction**: If >35% of samples in a trial have cumulative lick count >2, all lick values in that trial are set to 0.
2. **Binarization**: Lick values are capped at 1, then converted to binary (0 or 1).

ii.
```python
if np.sum(trial_lick > 2) / n_t > LICK_ERROR_FRACTION_THRESHOLD:
    trial_lick[:] = 0
trial_lick[trial_lick > 1] = 1
trial_lick = (trial_lick > 0).astype(np.float32)
```

iii. The AI documented the lick error correction from the reference code (CONVERSION_NOTES Step 1), noting the >35% threshold from code (vs >30% in paper text).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick data uses the same frame indices (`s:e`) as neural data, providing inherent alignment.

ii.
```python
trial_lick = lick[s:e].copy()
output_arr[3, :] = lick_binned
```

iii. Same alignment as other variables.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB `identifier` field (which contains the scene name, e.g., "Env1_LocationB_to_A") and the trial index within the session.

ii.
```python
identifier = f['identifier'][()].decode()
scene = get_scene_from_identifier(identifier)
rz_coords, rz_labels = get_reward_zones(scene, n_trials)
rz_loc = rz_label_to_idx(rz_labels[i])
```

iii. The AI maps scene names to reward zone coordinates using logic matching the reference code's `behavior.get_reward_zones()`, handling within-environment switches (e.g., A_to_B) and cross-environment switches (e.g., Env1_A_to_Env2_B).

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene name is parsed to determine:
1. Whether it's a single-location or switch session
2. For switch sessions, which zones are before/after the switch (trial 30)
Zone labels (A/B/C) are mapped to indices (0/1/2). The result is broadcast across all timepoints.

ii.
```python
def get_reward_zones(scene, n_trials, change_trial=DEFAULT_CHANGE_TRIAL):
    # Pattern matching on scene name
    if 'Location' in scene and '_to' not in scene:
        loc = scene.split('Location')[-1]
        rz_coords[:] = REWARD_ZONE_DICT[loc]
        rz_labels[:] = loc
    elif 'A_to' in scene and scene[-1] == 'B':
        rz_coords[:change_trial] = REWARD_ZONE_DICT['A']
        rz_coords[change_trial:] = REWARD_ZONE_DICT['B']
    # ... etc.

def rz_label_to_idx(label):
    mapping = {'A': 0, 'B': 1, 'C': 2}
    return mapping.get(label, -1)
```

iii. The AI documented the reward zone mapping and switch trial logic (trial 30) from the reference code (CONVERSION_NOTES Step 1).

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` timestamps in the NWB behavioral timeseries, compared against the trial's time boundaries from `position/timestamps`.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
# ...
was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
reward_outcome = int(was_rewarded)
```

iii. The AI detects reward by checking whether any reward event timestamp falls within the trial window.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the start and end timestamps are determined from `pos_timestamps`. If any reward timestamp falls within this range, the trial is marked as rewarded (1), otherwise not (0). The result is broadcast across all timepoints.

ii.
```python
trial_start_time = trial_timestamps[0]
trial_end_time = trial_timestamps[-1]
was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
reward_outcome = int(was_rewarded)
output_arr[5, :] = reward_outcome  # broadcast
```

iii. The AI noted ~85% reward rate matching the paper's ~15% omission rate (CONVERSION_NOTES Step 9).

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several data quality issues are handled:
1. **Lick sensor errors**: Trials with >35% of samples having lick>2 get lick values zeroed out (not removed).
2. **Neural-behavioral length mismatch**: When neural and behavioral data differ in length (by 1 frame in multi-plane recordings), both are truncated to the minimum length.
3. **Short/invalid trials**: Trials with fewer than 2 timepoints are skipped.
4. **Sessions with few cells**: Sessions with fewer than 2 cells after filtering are skipped.
5. **Negative environment values**: Environment values of -1 (inter-trial) are filtered out when computing per-trial environment.

ii.
```python
# Length mismatch
n_timepoints_total = min(n_timepoints_neural, n_timepoints_behav)
# ...
# Short trials
if e <= s or (e - s) < 2:
    continue
# Lick error
if np.sum(trial_lick > 2) / n_t > LICK_ERROR_FRACTION_THRESHOLD:
    trial_lick[:] = 0
# Environment filtering
env_val = np.median(trial_env[trial_env >= 0]) if np.any(trial_env >= 0) else 0.0
```

iii. The AI documented these edge cases in CONVERSION_NOTES Steps 5 and 10 (Check 5: Edge cases).

## 13-a. What are the most time-consuming steps of the code?

i. The main bottlenecks are:
1. **NWB file I/O**: Reading large HDF5 files for each session, especially for multi-plane animals (m17, m18) with >2000 ROIs.
2. **dF/F computation for interneuron detection**: Computing dF/F per trial involves Gaussian smoothing, minimum and maximum filtering across all cells.
3. **Interneuron correlation computation**: Computing Pearson correlation between dF/F and speed for all cells.

The full conversion took ~681 seconds (11.4 minutes) for 152 sessions, averaging ~4.5 seconds per session.

ii. The conversion log shows timing per session, with larger sessions (e.g., m18 with >2000 neurons) taking 6-10 seconds.

iii. The AI estimated ~9 minutes initially and implemented vectorized interneuron detection to speed up processing (CONVERSION_NOTES Step 7).

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop that could be further vectorized:
1. **dF/F computation loop**: The per-trial dF/F computation iterates over trials (`for i in range(n_trials)`), computing dF/F separately for each trial. This could potentially be vectorized by pre-allocating and using batch operations.
2. **Trial processing loop**: The main trial loop processes each trial sequentially for extracting neural, input, and output data.

ii.
```python
# dF/F per trial loop
for i in range(n_trials):
    s = trial_start_inds[i]
    e = teleport_inds[i]
    dff_full[:, s:e] = compute_dff_trial(F_cells[:, s:e], Fneu_cells[:, s:e])
```

iii. The AI noted that vectorized interneuron detection was implemented, reducing processing time from ~18s to ~7s for 2 sessions (CONVERSION_NOTES Step 7).

## 13-c. What processing does the code repeat multiple times?

i. The code loads F (raw fluorescence) and Fneu (neuropil) data to compute dF/F for interneuron detection. This computation is done once per session. The reward zone determination logic is computed once and then indexed per trial. No significant repeated processing is evident.

ii. The main processing pipeline is fairly streamlined - each session is processed once, and each trial within a session is processed once.

iii. No major redundancies documented.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several processing steps could be considered unnecessary:
1. **dF/F computation**: Computed solely for interneuron detection (speed-dFF correlation), not used as the neural data itself. This involves neuropil subtraction, maximin baseline estimation, and Gaussian smoothing for every cell and trial.
2. **Speed output**: Speed is computed and included as an output, but the instruction specifies it as a decoder output - so this is necessary.
3. **F and Fneu loading**: Raw fluorescence and neuropil data are loaded into memory even though only deconvolved events are used for neural data. These are only needed for the dF/F interneuron detection step.

ii.
```python
F_data = ophys[f'Fluorescence/{plane_name}/data'][:]
Fneu_data = ophys[f'Neuropil/{plane_name}/data'][:]
# Only used for:
dff_full[:, s:e] = compute_dff_trial(F_cells[:, s:e], Fneu_cells[:, s:e])
# Which is only used for:
is_interneuron = detect_interneurons(dff_full, speed)
```

iii. The dF/F computation is the most computationally expensive "unnecessary" step, but it's required for the interneuron exclusion quality control step that the reference code implements.
