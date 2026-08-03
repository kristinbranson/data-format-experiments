# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all NWB files by globbing `data/sub-*/*.nwb`. It opens each file using `h5py` (not `pynwb`), reads neural data from `processing/ophys/Deconvolved/`, `Fluorescence/`, and `Neuropil/`, and behavioral data from `processing/behavior/BehavioralTimeSeries/`. Metadata (subject_id, session_id, scene) is read from `general/` and `identifier` fields.

ii.
```python
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*', '*.nwb')))
...
with h5py.File(nwb_path, 'r') as f:
    subject_id = f['general/subject/subject_id'][()].decode()
    session_id = f['general/session_id'][()].decode()
    ...
    bts = f['processing/behavior/BehavioralTimeSeries']
    position = bts['position/data'][:]
    ...
    ophys = f['processing/ophys']
    deconv_data = ophys[f'Deconvolved/{plane_name}/data'][:]
```

iii. The AI chose `h5py` for direct HDF5 access to NWB files. CONVERSION_NOTES.md documents the NWB file structure and confirms 11 subjects and 152 sessions matching the paper.

## 1-b. How are the data split into subjects?

i. Subjects are determined from the `general/subject/subject_id` field within each NWB file. Unique subjects are collected into a sorted set.

ii.
```python
subject_id = f['general/subject/subject_id'][()].decode()
...
subjects_set = set()
...
subjects_set.add(result['subject_id'])
...
subjects = sorted(subjects_set)
```

iii. Subject IDs are extracted from the NWB metadata rather than directory names, but both approaches yield the same set of 11 subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. All NWB files are discovered by globbing `data/sub-*/*.nwb` and sorted.

ii.
```python
nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*', '*.nwb')))
...
for nwb_path in nwb_files:
    result = process_session(nwb_path, ...)
    if result is not None:
        all_sessions.append(result)
```

iii. Each NWB file is one session. The AI confirmed 152 total sessions (14 per subject, 12 for m11).

## 1-d. How are the data split into trials?

i. Trial boundaries are found using `trial_start_flag > 0` for trial starts and `teleport_flag > 0` for trial ends. The number of trials is the minimum of the two arrays' lengths, and the first `n_trials` indices from each are paired.

ii.
```python
trial_start_inds = np.where(trial_start_flag > 0)[0]
teleport_inds = np.where(teleport_flag > 0)[0]

n_trials = min(len(trial_start_inds), len(teleport_inds))
...
trial_start_inds = trial_start_inds[:n_trials]
teleport_inds = teleport_inds[:n_trials]
...
for i in range(n_trials):
    s = trial_start_inds[i]
    e = teleport_inds[i]
    ...
    trial_pos = position[s:e]
```

iii. The AI uses all indices where teleport > 0 rather than detecting rising edges (transitions from <=0 to >0) as the reference does. This works if teleport is a single-frame pulse but would fail if teleport stays positive for multiple frames.

## 1-e. How are trials filtered based on quality controls?

i. Trials are skipped if `e <= s` or `(e - s) < 2` (fewer than 2 timepoints). No minimum timepoint threshold like the reference's 50-timepoint filter.

ii.
```python
if e <= s or (e - s) < 2:
    trial_start_time = pos_timestamps[s] if s < len(pos_timestamps) else 0
    trial_end_time = pos_timestamps[min(e, len(pos_timestamps) - 1)]
    was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
    prev_trial_rewarded = int(was_rewarded)
    continue
```

iii. The AI's minimal filtering (< 2 timepoints) allows very short trials through that the reference would exclude. The AI still tracks previous trial reward for skipped trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is from the `Deconvolved` field in the NWB ophys processing module, filtered by `iscell`.

ii.
```python
deconv_data = ophys[f'Deconvolved/{plane_name}/data'][:]  # (n_timepoints, n_rois_plane)
...
deconv_all = np.concatenate(deconv_list, axis=1)
...
cell_mask = iscell[:, 0].astype(bool)
...
deconv_cells = deconv_all[:, cell_mask].T  # (n_cells, n_timepoints)
```

iii. The paper's decoder uses deconvolved events. The NWB files contain pre-computed deconvolved events from suite2p/OASIS.

## 2-b. How is the `neural` data processed?

i. The AI applies two processing steps beyond basic loading:
1. Concatenation of ROIs across imaging planes
2. Interneuron exclusion: computes dF/F from raw fluorescence and neuropil (neuropil subtraction with coefficient 0.7, maximin baseline, Gaussian smoothing), then correlates dF/F with speed to identify and exclude putative interneurons (Pearson r > 0.5).

ii.
```python
def compute_dff_trial(F_trial, Fneu_trial, baseline_window=BASELINE_WINDOW):
    F_corr = F_trial - NEUROPIL_COEF * Fneu_trial
    smoothed = gaussian_filter1d(F_corr, sigma=15, axis=1)
    baseline = minimum_filter1d(smoothed, size=min(baseline_window, n_t), axis=1)
    baseline = maximum_filter1d(baseline, size=min(baseline_window, n_t), axis=1)
    ...
    dff = (F_corr - baseline) / abs_baseline
    dff = gaussian_filter1d(dff, sigma=DFF_SMOOTH_SIGMA, axis=1)
    return dff

def detect_interneurons(dff_all, speed_all, threshold=INTERNEURON_SPEED_CORR_THRESHOLD):
    ...
    corr = dff_centered @ speed_centered / (dff_std * speed_std + 1e-10)
    is_interneuron = corr > threshold
    return is_interneuron

# In process_session:
dff_full = np.full_like(F_cells, np.nan)
for i in range(n_trials):
    ...
    dff_full[:, s:e] = compute_dff_trial(F_cells[:, s:e], Fneu_cells[:, s:e])
is_interneuron = detect_interneurons(dff_full, speed)
final_cell_mask = ~is_interneuron
neural_data = deconv_cells[final_cell_mask]
```

iii. The AI noted that the paper and reference analysis code exclude interneurons (speed correlation > 0.5). The dF/F computation follows the reference code's `preprocessing.dff()` function.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering: (1) `iscell` flag from suite2p, (2) interneuron exclusion via speed-dF/F correlation > 0.5.

ii.
```python
cell_mask = iscell[:, 0].astype(bool)
...
is_interneuron = detect_interneurons(dff_full, speed)
final_cell_mask = ~is_interneuron
n_final_cells = final_cell_mask.sum()
neural_data = deconv_cells[final_cell_mask]
```

iii. The paper describes excluding putative interneurons (0.42+/-0.85% of neurons). The AI implemented both iscell filtering and interneuron exclusion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start by extracting data between `trial_start_inds[i]` and `teleport_inds[i]`. No additional temporal shifting is needed since the trial start IS the alignment event.

ii.
```python
for i in range(n_trials):
    s = trial_start_inds[i]
    e = teleport_inds[i]
    ...
    trial_neural = neural_data[:, s:e]
```

iii. Instructions specify alignment to "start of the trial." The data extraction naturally aligns to trial start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging frame rate (~15.5 Hz, ~64.5 ms per frame) is used without rebinning. For multi-plane animals (m17, m18), the effective rate is imaging_rate / n_planes.

ii.
```python
n_planes = len(fluor_planes)
effective_rate = imaging_rate if n_planes == 1 else imaging_rate / n_planes
time_bin_ms = 1000.0 / effective_rate
```

iii. No temporal rebinning is applied. The time bin is approximately 64.5 ms at ~15.5 Hz effective rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the frame count within a trial and the effective imaging rate, NOT from stored timestamps.

ii.
```python
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
```

iii. The AI computed time from the frame index and rate rather than using stored timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Frame indices (0, 1, 2, ..., n_t-1) are divided by the effective imaging rate to get time in seconds.

ii.
```python
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
...
input_arr[0, :] = time_from_start
```

iii. This assumes perfectly uniform frame spacing. The reference uses actual timestamps which could capture timing irregularities.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Since both neural and behavioral data are indexed by the same frame indices within each trial, they are inherently aligned. The time variable starts at 0 for the first frame of each trial.

ii.
```python
input_arr[0, :] = time_from_start  # shape matches trial_neural.shape[1]
```

iii. Neural and behavioral data share the same frame indices.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. From the `environment` behavioral time series in the NWB file.

ii.
```python
environment = bts['environment/data'][:]
...
trial_env = environment[s:e]
```

iii. The environment variable indicates ENV1 vs ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI takes the median of non-negative environment values within the trial, then broadcasts it as a per-trial scalar across all timepoints.

ii.
```python
env_val = np.median(trial_env[trial_env >= 0]) if np.any(trial_env >= 0) else 0.0
env_type = np.float32(env_val)
...
input_arr[1, :] = env_type  # broadcast
```

iii. The median is used as a robust summary since environment should be constant within a trial. The reference takes the raw per-timepoint values directly.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the loop counter over trials within a session (0-indexed).

ii.
```python
for i in range(n_trials):
    ...
    trial_number = np.float32(i)
    input_arr[2, :] = trial_number
```

iii. The loop index is used rather than the stored `trial number` behavioral time series.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond converting the loop index to float32. The value is constant across all timepoints within a trial.

ii.
```python
trial_number = np.float32(i)
input_arr[2, :] = trial_number  # broadcast
```

iii. Simple sequential indexing within the session.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from `Reward/timestamps` — the reward delivery timestamps stored separately from the behavioral time series.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
...
def detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time):
    if len(reward_timestamps) == 0:
        return False
    return np.any((reward_timestamps >= trial_start_time) & (reward_timestamps <= trial_end_time))
```

iii. Reward delivery is indicated by timestamps in the `Reward` time series.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, check if any reward timestamp falls within the previous trial's time window. For the first trial, set to 0. The value is tracked across skipped trials.

ii.
```python
prev_trial_rewarded = 0  # For the first trial
for i in range(n_trials):
    ...
    was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
    ...
    prev_outcome = np.float32(prev_trial_rewarded)
    input_arr[3, :] = prev_outcome  # broadcast
    ...
    prev_trial_rewarded = int(was_rewarded)
```

iii. The AI correctly handles skipped trials by updating `prev_trial_rewarded` even when a trial is skipped. The reference instead indexes into the previous trial's time indices directly.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` behavioral time series and reward zone coordinates determined by parsing the scene name from the NWB `identifier` field using `get_reward_zones()`.

ii.
```python
identifier = f['identifier'][()].decode()
scene = get_scene_from_identifier(identifier)
...
rz_coords, rz_labels = get_reward_zones(scene, n_trials)
...
rz_start, rz_end = rz_coords[i]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The AI's approach mirrors the reference analysis code's `behavior.get_reward_zones()` which parses scene names. The reference conversion code uses a Viterbi algorithm on reward_zone behavioral data instead.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from position to the nearest edge of the reward zone. Negative when before the zone, 0 when inside, positive when past it.

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

iii. The computation logic matches the reference's `compute_distance_to_reward_zone()` function.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit conditional logic rather than `np.digitize`.

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

iii. The bin definitions match the instructions. The AI uses `dist == 0` for strict equality to identify the "inside reward zone" category. The reference uses `np.digitize` with a small epsilon (1e-6) boundary to separate 0 from slightly positive values.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position data is extracted using the same frame indices as neural data within each trial, ensuring alignment.

ii.
```python
trial_pos = position[s:e]
trial_neural = neural_data[:, s:e]
```

iii. Same indexing ensures alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series.

ii.
```python
position = bts['position/data'][:]
...
trial_pos = position[s:e]
```

iii. The position variable records the animal's location in the VR corridor in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is discretized into 5 equal-sized bins over [0, 450] cm using `np.linspace(0, 450, 6)` = [0, 90, 180, 270, 360, 450], then clipped to [0, 4].

ii.
```python
POSITION_BIN_EDGES = np.linspace(0, TRACK_LENGTH, POSITION_BINS + 1)
# = [0, 90, 180, 270, 360, 450]

def discretize_position(position, n_bins=POSITION_BINS):
    bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)
    binned = np.digitize(position, bin_edges) - 1
    binned = np.clip(binned, 0, n_bins - 1)
    return binned
```

iii. The AI interpreted "5 equal-sized bins" as 90 cm bins over the 450 cm track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. 5 bins: [0, 90), [90, 180), [180, 270), [270, 360), [360, 450]. Values outside [0, 450] are clipped to the nearest bin.

ii.
```python
binned = np.digitize(position, bin_edges) - 1
binned = np.clip(binned, 0, n_bins - 1)
```

iii. The reference uses different bin edges: [-inf, 50, 150, 250, 350, inf], giving approximately 100 cm bins centered differently. The AI's bins are 90 cm each over [0, 450].

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices as neural data within each trial.

ii.
```python
trial_pos = position[s:e]
trial_neural = neural_data[:, s:e]
```

iii. Same indexing ensures alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series.

ii.
```python
lick_raw = bts['lick/data'][:]
...
trial_lick = lick[s:e].copy()
```

iii. The lick variable records lick events at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI applies lick sensor error correction from the reference analysis code: if >35% of samples in a trial have cumulative lick > 2, licks are set to 0 for that trial. Then licks are capped at 1 and binarized.

ii.
```python
# Lick sensor error correction:
if np.sum(trial_lick > 2) / n_t > LICK_ERROR_FRACTION_THRESHOLD:
    trial_lick[:] = 0
# Cap licks at 1 (binary)
trial_lick[trial_lick > 1] = 1
trial_lick = (trial_lick > 0).astype(np.float32)
```

iii. The reference analysis code applies this correction. The reference conversion code does NOT apply it — it simply binarizes with `(licks_curr > 0).astype(int)`.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices as neural data within each trial.

ii.
```python
trial_lick = lick[s:e].copy()
trial_neural = neural_data[:, s:e]
```

iii. Same indexing ensures alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `identifier` field in the NWB file, which contains the scene name (e.g., `Env1_LocationB_to_A`). The scene name encodes which reward zone is active and when switches occur.

ii.
```python
identifier = f['identifier'][()].decode()
scene = get_scene_from_identifier(identifier)
...
rz_coords, rz_labels = get_reward_zones(scene, n_trials)
...
rz_loc = rz_label_to_idx(rz_labels[i])
output_arr[4, :] = rz_loc
```

iii. The AI followed the reference analysis code's approach (`behavior.get_reward_zones()`) which parses scene names. The reference conversion code uses a Viterbi algorithm on reward_zone behavioral data instead.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene name is parsed to determine reward zone identity and switch points. For switch sessions (e.g., `Env1_LocationB_to_A`), the first 30 trials use the initial zone and remaining trials use the target zone (change_trial=30). Labels are mapped to indices: A=0, B=1, C=2.

ii.
```python
def get_reward_zones(scene, n_trials, change_trial=DEFAULT_CHANGE_TRIAL):
    ...
    if 'Location' in scene and '_to' not in scene:
        loc = scene.split('Location')[-1]
        rz_coords[:] = REWARD_ZONE_DICT[loc]
        rz_labels[:] = loc
    elif 'A_to' in scene and scene[-1] == 'B':
        rz_coords[:change_trial] = REWARD_ZONE_DICT['A']
        rz_labels[:change_trial] = 'A'
        rz_coords[change_trial:] = REWARD_ZONE_DICT['B']
        rz_labels[change_trial:] = 'B'
    ...

def rz_label_to_idx(label):
    mapping = {'A': 0, 'B': 1, 'C': 2}
    return mapping.get(label, -1)
```

iii. The change_trial=30 default comes from the reference analysis code. The scene parsing logic covers all A-B, B-A, A-C, C-A, B-C, C-B switch combinations and single-location sessions.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from `Reward/timestamps` — the reward delivery timestamps.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
```

iii. Reward events have separate timestamps from the behavioral sampling rate.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward timestamp falls within the trial's time window (trial_start_time to trial_end_time). Binary: 1 if rewarded, 0 if not. Per-trial value broadcast across all timepoints.

ii.
```python
def detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time):
    if len(reward_timestamps) == 0:
        return False
    return np.any((reward_timestamps >= trial_start_time) & (reward_timestamps <= trial_end_time))
...
was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
reward_outcome = int(was_rewarded)
output_arr[5, :] = reward_outcome
```

iii. The reference instead maps reward timestamps to behavioral frame indices using `searchsorted` and checks if any mapped index falls within the trial's frame range. Both achieve the same result.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Neural/behavior length mismatch**: Truncated to minimum of the two lengths.
- **Short/invalid trials**: Trials with `e <= s` or `(e - s) < 2` are skipped.
- **Sessions with too few cells**: Sessions with < 2 cells after filtering are skipped.
- **Sessions with too few trials**: Sessions with < 2 valid trials are skipped.
- **Lick sensor errors**: Trials with >35% of samples having lick > 2 get licks set to 0.

ii.
```python
if n_timepoints_neural != n_timepoints_behav:
    deconv_all = deconv_all[:n_timepoints_total]
    position = position[:n_timepoints_total]
    ...
if e <= s or (e - s) < 2:
    ...
    continue
if n_final_cells < 2:
    return None
if valid_trial_count < 2:
    return None
```

iii. These are defensive checks found during data exploration.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files** via h5py and reading large arrays (fluorescence, deconvolved, neuropil data)
2. **Computing dF/F** for interneuron detection (Gaussian smoothing, min/max filtering for each trial)
3. **Interneuron detection** (correlation computation across all neurons)

ii. N/A

iii. The dF/F computation and interneuron detection are additional steps not present in the reference conversion code, adding significant processing time.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial dF/F computation loop (iterating over trials to compute `compute_dff_trial`) and the per-trial extraction loop could potentially be partially vectorized, though variable trial lengths make full vectorization difficult.

ii.
```python
for i in range(n_trials):
    s = trial_start_inds[i]
    e = teleport_inds[i]
    if e <= s:
        continue
    dff_full[:, s:e] = compute_dff_trial(F_cells[:, s:e], Fneu_cells[:, s:e])
```

iii. The AI vectorized the interneuron detection (matrix correlation) but the per-trial dF/F loop remains sequential.

## 13-c. What processing does the code repeat multiple times?

i. The AI loads each NWB file only once (no separate survey step), so there is no repeated I/O. However, fluorescence data (`F_all`, `Fneu_all`) is loaded and processed for dF/F computation even though only the deconvolved events are used for the final neural data.

ii. N/A

iii. The fluorescence and neuropil data are loaded solely for interneuron detection. The reference conversion code avoids this by not performing interneuron exclusion.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads and processes fluorescence (`F_all`) and neuropil (`Fneu_all`) data to compute dF/F for interneuron detection. This is a substantial amount of data that is only used to identify a small number of interneurons (~0.42% of neurons). The deconvolved events (which are the final neural data) are loaded separately. Additionally, the lick sensor error correction processing is applied but the reference conversion code doesn't include it.

ii.
```python
F_list.append(F_data)
Fneu_list.append(Fneu_data)
...
F_all = np.concatenate(F_list, axis=1)
Fneu_all = np.concatenate(Fneu_list, axis=1)
...
dff_full[:, s:e] = compute_dff_trial(F_cells[:, s:e], Fneu_cells[:, s:e])
```

iii. Loading F and Fneu and computing dF/F is computationally expensive and only removes ~0.42% of neurons. The reference conversion avoids this entirely.
