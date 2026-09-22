# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all subject directories under `/app/data` that start with `sub-`, and for each subject finds all `.nwb` files. Each NWB file is loaded using `h5py` (not `pynwb`). All behavioral and neural data arrays are read directly from the HDF5 group structure.

ii.
```python
subjects_dirs = sorted([
    d for d in os.listdir(DATA_DIR)
    if d.startswith('sub-') and os.path.isdir(os.path.join(DATA_DIR, d))
])
...
for subj_dir in subjects_dirs:
    subj_path = os.path.join(DATA_DIR, subj_dir)
    nwb_files = sorted([
        f for f in os.listdir(subj_path) if f.endswith('.nwb')
    ])
    for nwb_file in nwb_files:
        nwb_path = os.path.join(subj_path, nwb_file)
        result = load_and_process_session(nwb_path, subject_name)
```
Loading with h5py:
```python
with h5py.File(nwb_path, 'r') as f:
    beh = f['processing']['behavior']['BehavioralTimeSeries']
    position = beh['position']['data'][:]
    ...
```

iii. The agent explored the data directory structure with sub-agents and discovered the `sub-*` directory convention. No explicit justification was given for choosing `h5py` over `pynwb`; the agent's exploration scripts simply used h5py.

## 1-b. How are the data split into subjects?

i. Subjects correspond to subdirectories of the data directory starting with `sub-`. The subject name is extracted by removing the `sub-` prefix.

ii.
```python
subjects_dirs = sorted([
    d for d in os.listdir(DATA_DIR)
    if d.startswith('sub-') and os.path.isdir(os.path.join(DATA_DIR, d))
])
...
subject_name = subj_dir.replace('sub-', '')
```

iii. The agent discovered the directory structure during exploration. No additional justification needed.

## 1-c. How are the data split into sessions?

i. Each NWB file within a subject directory corresponds to one session. Sessions are processed sequentially.

ii.
```python
nwb_files = sorted([
    f for f in os.listdir(subj_path) if f.endswith('.nwb')
])
for nwb_file in nwb_files:
    nwb_path = os.path.join(subj_path, nwb_file)
    result = load_and_process_session(nwb_path, subject_name)
```

iii. Each NWB file is treated as a separate session. The agent also parses the scene name from the NWB identifier to determine session properties (environment, reward zone).

## 1-d. How are the data split into trials?

i. Trial boundaries are found using the `trial_start` signal (frames where it exceeds 0.5) and `teleport` signal (next teleport frame after each start). The trial end is the teleport frame + 1 (inclusive of the teleport frame). Additionally, only "on-track" timepoints where position is between 0 and 455 cm are kept within each trial.

ii.
```python
def get_trial_boundaries(trial_number, trial_start_signal, teleport_signal):
    start_frames = np.where(trial_start_signal > 0.5)[0]
    teleport_frames = np.where(teleport_signal > 0.5)[0]
    for sf in start_frames:
        tid = int(trial_number[sf])
        if tid < 0:
            continue
        future_teleports = teleport_frames[teleport_frames > sf]
        if len(future_teleports) == 0:
            next_starts = start_frames[start_frames > sf]
            if len(next_starts) > 0:
                ef = next_starts[0]
            else:
                ef = len(trial_number)
        else:
            ef = future_teleports[0] + 1
        trial_starts.append(sf)
        trial_ends.append(ef)
        trial_ids.append(tid)
    return trial_starts, trial_ends, trial_ids
```
On-track filtering:
```python
def extract_on_track_indices(position, start_idx, end_idx):
    trial_pos = position[start_idx:end_idx]
    on_track = (trial_pos >= 0) & (trial_pos <= TRACK_LENGTH + 5)
    indices = np.where(on_track)[0] + start_idx
    return indices
```

iii. The agent reasoned that timepoints within the teleport zone (negative positions) should be excluded since they don't represent actual track positions. Trials with negative trial numbers are skipped. The on-track filtering ensures only valid positions are included.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 5 on-track timepoints are skipped. Sessions with fewer than 2 valid trials are skipped. Sessions where the scene cannot be parsed (training sessions) are skipped.

ii.
```python
if len(on_track_idx) < 5:  # Skip very short trials
    continue
...
if len(neural_trials) < 2:
    print(f"  Skipping session with < 2 valid trials: {scene}")
    return None
```

iii. The agent chose a minimum of 5 timepoints for a trial to be valid. Training sessions are skipped because their scene names don't match the expected patterns.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the NWB `Deconvolved` field (suite2p's stored deconvolution). For multi-plane animals (m17, m18), data from both `plane0` and `plane1` are concatenated.

ii.
```python
deconvolved = ophys['Deconvolved']['plane0']['data'][:]
# or for multi-plane:
deconv0 = ophys['Deconvolved']['plane0']['data'][:]
deconv1 = ophys['Deconvolved']['plane1']['data'][:]
deconvolved = np.concatenate([deconv0, deconv1], axis=1)
...
neural_data = deconvolved_cells[:, good_cells]
```

iii. The agent reasoned: "Suite2p already subtracts neuropil and computes dF/F internally before deconvolving, so the NWB's deconvolved field has likely already incorporated that processing... I'll treat the Deconvolved field as the paper's finished pipeline output and use it directly." The agent also noted that "Reprocessing from raw fluorescence would be complex and error-prone."

## 2-b. How is the `neural` data processed?

i. The `Deconvolved` data from NWB is used directly with no further processing beyond cell filtering (iscell and interneuron removal). NaN/Inf values are replaced with 0.

ii.
```python
neural_data = deconvolved_cells[:, good_cells]
...
trial_neural = neural_data[on_track_idx, :].T
...
if np.any(np.isnan(trial_neural)) or np.any(np.isinf(trial_neural)):
    trial_neural = np.nan_to_num(trial_neural, nan=0.0, posinf=0.0, neginf=0.0)
neural_trials.append(trial_neural.astype(np.float32))
```

iii. The agent decided to use the NWB deconvolved data directly rather than recomputing the paper's full dF/F pipeline (neuropil subtraction with mean re-addition, maximin baseline, smoothing, OASIS deconvolution). A separate simplified dF/F was computed only for interneuron detection.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) `iscell` from suite2p's classification, and (2) putative interneuron removal based on speed-dF/F correlation > 0.5. A simplified dF/F is computed from Fluorescence and Neuropil for the interneuron detection step only.

ii.
```python
cell_mask = iscell == 1
deconvolved_cells = deconvolved[:, cell_mask]
...
dff = compute_dff_for_interneuron_detection(
    fluorescence_cells, neuropil_cells, trial_starts, trial_ends
)
is_interneuron = detect_interneurons(dff, speed, cell_mask)
good_cells = ~is_interneuron
neural_data = deconvolved_cells[:, good_cells]
```

iii. The agent followed the paper's methods for both filters: iscell for manual curation and interneuron exclusion with Pearson r > 0.5 threshold. The dF/F for interneuron detection uses a simplified version of the paper's maximin baseline method.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start. The neural data is indexed by the on-track indices within each trial, which start from the trial start frame (where position >= 0).

ii.
```python
trial_neural = neural_data[on_track_idx, :].T
```

iii. No additional alignment is needed since the trial start is the alignment event and neural/behavioral data share the same time indices.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging frame rate is used (~15.5 Hz, ~64.5 ms per bin). No rebinning is applied. The time bin size is computed from the median inter-sample interval of the behavioral timestamps.

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

iii. The agent noted the native imaging rate is approximately 15.5 Hz and decided not to resample.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the behavioral timestamps (`beh_timestamps`) from the `position` time series.

ii.
```python
beh_timestamps = beh['position']['timestamps'][:]
...
trial_times = beh_timestamps[on_track_idx] - beh_timestamps[on_track_idx[0]]
```

iii. The agent used the position timestamps as the time base for computing within-trial time.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at the first on-track frame of the trial is subtracted from all timestamps in the trial.

ii.
```python
trial_times = beh_timestamps[on_track_idx] - beh_timestamps[on_track_idx[0]]
```

iii. Standard approach to compute time relative to trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same time indices (both indexed by `on_track_idx`), so no additional alignment is needed.

ii.
```python
on_track_idx = extract_on_track_indices(position, ts, te)
trial_neural = neural_data[on_track_idx, :].T
trial_times = beh_timestamps[on_track_idx] - beh_timestamps[on_track_idx[0]]
```

iii. Both neural and behavioral data are indexed by the same on-track indices.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Environment type is derived from parsing the scene name in the NWB `identifier` field, not from the `environment` behavioral variable.

ii.
```python
identifier = f['identifier'][()]
scene = parse_scene_from_identifier(identifier)
session_info = get_session_info(scene)
...
def get_session_info(scene):
    env_map = {'Env1': 0, 'Env2': 1}
    m = re.match(r'(Env[12])_([ABC])_to_(Env[12])_([ABC])', scene)
    ...
```

iii. The agent parsed the scene name to determine the environment. For switch sessions, the environment can change at trial 30.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The scene name is parsed with regex to extract environment labels (`Env1` or `Env2`), mapped to 0 or 1. For switch sessions, trials before trial 30 use the "before" environment and trials 30+ use the "after" environment.

ii.
```python
def get_per_trial_info(session_info, n_trials):
    zones = []
    envs = []
    for t in range(n_trials):
        if session_info['is_switch'] and t >= CHANGE_TRIAL:
            zones.append(session_info['zone_after'])
            envs.append(session_info['env_after'])
        else:
            zones.append(session_info['zone_before'])
            envs.append(session_info['env_before'])
    return zones, envs
```

iii. The agent reasoned that the scene name encodes the session type and environment, allowing per-trial assignment.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is read from the NWB `trial number` behavioral variable at the trial start frame.

ii.
```python
trial_ids = []
...
tid = int(trial_number[sf])
trial_ids.append(tid)
...
trial_num = trial_ids[i]
inp[2, :] = float(trial_num)
```

iii. The agent used the stored trial number from the NWB file directly.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial number at the trial start frame is read and assigned as a constant across all timepoints in the trial.

ii.
```python
inp[2, :] = float(trial_num)
```

iii. No processing beyond reading the value and broadcasting it.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps in the NWB behavioral data. Reward delivery is checked by whether any reward timestamp falls within the trial's time window.

ii.
```python
reward_timestamps = beh['Reward']['timestamps'][:]
...
is_rewarded = np.zeros(n_trials, dtype=int)
for i, (ts, te) in enumerate(zip(trial_starts, trial_ends)):
    t_start_time = beh_timestamps[ts]
    t_end_time = beh_timestamps[min(te - 1, len(beh_timestamps) - 1)]
    if np.any((reward_timestamps >= t_start_time) & (reward_timestamps <= t_end_time)):
        is_rewarded[i] = 1
```

iii. The agent checked whether any reward event timestamp fell within each trial's time boundaries.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the previous trial's reward status is looked up. For the first trial, the value defaults to 0.

ii.
```python
if i == 0:
    prev_outcome = 0
else:
    prev_outcome = is_rewarded[i - 1]
inp[3, :] = float(prev_outcome)
```

iii. The agent noted: "For the very first trial where there's no previous outcome, I'll just default that input to zero."

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavioral variable and the reward zone boundaries, which are determined from parsing the NWB session identifier (scene name).

ii.
```python
zone_label = zones[i]
rz = REWARD_ZONES[zone_label]
trial_pos = position[on_track_idx]
trial_pos = np.clip(trial_pos, 0, TRACK_LENGTH)
out[0, :] = discretize_distance_to_reward(trial_pos, rz)
```

iii. The agent derived the reward zone from the scene name and used the known zone boundaries (`A: [80,130], B: [200,250], C: [320,370]`) to compute signed distance.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed: negative when before the zone, 0 when inside, positive when past it. Position is first clipped to [0, 450].

ii.
```python
def discretize_distance_to_reward(position, reward_zone):
    rz_start, rz_end = reward_zone
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

iii. Standard signed distance computation matching the paper's concept.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit conditional logic matching the instruction bins.

ii.
```python
bins = np.zeros_like(distance, dtype=np.int64)
bins[distance < -50] = 0
bins[(distance >= -50) & (distance < -10)] = 1
bins[(distance >= -10) & (distance < 0)] = 2
bins[distance == 0] = 3
bins[(distance > 0) & (distance <= 10)] = 4
bins[(distance > 10) & (distance <= 50)] = 5
bins[distance > 50] = 6
```

iii. Bin edges match the instruction specifications.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same on-track indices are used for both neural and position data, so alignment is automatic.

ii.
```python
trial_neural = neural_data[on_track_idx, :].T
trial_pos = position[on_track_idx]
```

iii. Both use the same `on_track_idx` array.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral variable.

ii.
```python
position = beh['position']['data'][:]
...
trial_pos = position[on_track_idx]
trial_pos = np.clip(trial_pos, 0, TRACK_LENGTH)
```

iii. Direct use of the position variable.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450] and then discretized into 5 bins using `np.floor(position / 90)`, clipped to [0, 4].

ii.
```python
trial_pos = np.clip(trial_pos, 0, TRACK_LENGTH)
...
def discretize_position(position):
    bin_size = TRACK_LENGTH / 5  # 90 cm
    bins = np.clip(np.floor(position / bin_size).astype(np.int64), 0, 4)
    return bins
```

iii. The 450 cm track is divided into 5 equal bins of 90 cm each.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is divided by 90, floored, and clipped to [0, 4]. This creates bins: [0,90) -> 0, [90,180) -> 1, [180,270) -> 2, [270,360) -> 3, [360,450] -> 4.

ii.
```python
bins = np.clip(np.floor(position / bin_size).astype(np.int64), 0, 4)
```

iii. Equal-width bins of 90 cm spanning the 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same on-track indices for both neural and position data.

ii. Same indexing via `on_track_idx`.

iii. No additional alignment needed.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral variable.

ii.
```python
lick = beh['lick']['data'][:]
...
trial_lick = lick[on_track_idx]
```

iii. Direct use of the lick variable.

## 9-b. What processing is involved in computing `output` *Lick*?

i. A lick sensor error check is applied: if >35% of frames in a trial have lick count > 2, the lick data for that trial is zeroed out. Then lick is binarized (>0 -> 1, otherwise 0).

ii.
```python
def check_lick_sensor_error(lick_data, threshold=LICK_CORRECTION_THR):
    frac_high = np.mean(lick_data > 2)
    return frac_high > threshold
...
if check_lick_sensor_error(trial_lick):
    trial_lick = np.zeros_like(trial_lick)
trial_lick_binary = (trial_lick > 0).astype(np.int64)
```

iii. The agent reasoned that lick counts > 2 on more than 35% of frames indicate a stuck sensor. This matches a threshold from the paper's code (0.35). The binarization follows the instruction specification of 0=no, 1=yes.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same on-track indices for both neural and lick data.

ii. Same indexing via `on_track_idx`.

iii. No additional alignment needed.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from parsing the scene name in the NWB `identifier` field. The scene name encodes the environment and reward zone (e.g., "Env1_LocationB" or "Env1_B_to_Env2_C").

ii.
```python
identifier = f['identifier'][()]
scene = parse_scene_from_identifier(identifier)
session_info = get_session_info(scene)
zones, envs = get_per_trial_info(session_info, n_trials)
...
zone_label = zones[i]
out[4, :] = REWARD_ZONE_LABELS[zone_label]  # 0=A, 1=B, 2=C
```

iii. The agent determined from exploration that the `reward_zone` behavioral variable does not directly encode zone identity (A/B/C), so it parsed the scene name instead.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene name is parsed with regex. For switch sessions, trials 0-29 use the "before" zone and trials >= 30 use the "after" zone. Zones are mapped to integers: A=0, B=1, C=2.

ii.
```python
CHANGE_TRIAL = 30
...
def get_per_trial_info(session_info, n_trials):
    for t in range(n_trials):
        if session_info['is_switch'] and t >= CHANGE_TRIAL:
            zones.append(session_info['zone_after'])
        else:
            zones.append(session_info['zone_before'])
```

iii. The agent found that switch sessions change reward zone at trial 30, consistent with the paper's experimental design.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` timestamps. Reward delivery is checked by whether any reward timestamp falls within the trial's time window.

ii.
```python
reward_timestamps = beh['Reward']['timestamps'][:]
...
is_rewarded = np.zeros(n_trials, dtype=int)
for i, (ts, te) in enumerate(zip(trial_starts, trial_ends)):
    t_start_time = beh_timestamps[ts]
    t_end_time = beh_timestamps[min(te - 1, len(beh_timestamps) - 1)]
    if np.any((reward_timestamps >= t_start_time) & (reward_timestamps <= t_end_time)):
        is_rewarded[i] = 1
```

iii. The agent checks whether any reward delivery event occurred within the trial's time boundaries.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Binary: 1 if any reward event occurred during the trial, 0 otherwise. The value is constant across all timepoints in the trial.

ii.
```python
out[5, :] = is_rewarded[i]
```

iii. Per-trial binary output matching the instruction specification.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Neural/behavior length mismatch**: All arrays are cropped to the minimum shared length.
- **Short trials**: Trials with fewer than 5 on-track timepoints are skipped.
- **NaN/Inf in neural data**: Replaced with 0.
- **Lick sensor errors**: Trials with >35% high-lick frames have lick data zeroed.
- **Training sessions**: Sessions with unparseable scene names are skipped.
- **Sessions with too few cells or trials**: Skipped.

ii.
```python
n_time = min(deconvolved.shape[0], fluorescence.shape[0], ..., len(beh_timestamps))
...
if len(on_track_idx) < 5:
    continue
...
trial_neural = np.nan_to_num(trial_neural, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. These are defensive checks added during data exploration.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files** with h5py and reading large arrays (I/O bound)
2. **Computing dF/F for interneuron detection** (per-trial baseline estimation with Gaussian smoothing, min/max filtering)
3. **Interneuron detection** (computing Pearson correlation for each cell)

ii. N/A

iii. NWB files are large and loading all data arrays is I/O bound.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The interneuron detection loop over cells (`for i in range(n_cells): pearsonr(...)`) could be vectorized using `np.corrcoef` on the full matrix. The per-trial loop in `load_and_process_session` iterates over trials sequentially for discretization operations that could be applied to full session arrays.

ii.
```python
for i in range(n_cells):
    cell_dff = dff[valid, i]
    r, _ = pearsonr(cell_dff, speed_valid)
```

iii. The per-cell loop is the natural structure but could use matrix-level correlation.

## 13-c. What processing does the code repeat multiple times?

i. The code loads each NWB file once and processes it in a single pass. There is no repeated loading. However, the dF/F computation for interneuron detection duplicates some work that could have been avoided if the NWB deconvolved data were not used directly for the neural signal.

ii. N/A

iii. The agent chose to compute dF/F separately for interneuron detection while using the NWB deconvolved data for the neural signal, creating some redundancy.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The lick sensor error detection (`check_lick_sensor_error`) adds processing that the reference solution does not perform. The on-track position filtering (`extract_on_track_indices`) adds an extra step that may discard valid timepoints. Position clipping to [0, 450] modifies data that would otherwise be naturally handled by binning.

ii.
```python
if check_lick_sensor_error(trial_lick):
    trial_lick = np.zeros_like(trial_lick)
...
on_track = (trial_pos >= 0) & (trial_pos <= TRACK_LENGTH + 5)
...
trial_pos = np.clip(trial_pos, 0, TRACK_LENGTH)
```

iii. These are extra processing steps the agent added that go beyond the reference approach.
