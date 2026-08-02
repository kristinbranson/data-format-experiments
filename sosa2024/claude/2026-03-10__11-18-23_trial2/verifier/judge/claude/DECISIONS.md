# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all NWB files organized by subject directories (`sub-*`) in the `data` folder. It uses `h5py` (not `pynwb`) to directly read HDF5 fields from each NWB file. All subjects and all NWB files within each subject directory are processed.

ii.
```python
def find_nwb_files(data_dir='data'):
    subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
    sessions = []
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        files = sorted(glob(os.path.join(subj_dir, '*.nwb')))
        for fpath in files:
            sessions.append({
                'subject': subj.replace('sub-', ''),
                'filepath': fpath,
                'filename': os.path.basename(fpath),
            })
    return sessions
```

Loading data with h5py:
```python
with h5py.File(filepath, 'r') as f:
    subject_id = f['general/subject/subject_id'][()].decode()
    identifier = f['identifier'][()].decode()
    # ... reads all neural and behavior fields directly
```

iii. The AI documented that 11 subjects and 152 sessions were found, matching the paper. It chose h5py over pynwb for performance reasons.

## 1-b. How are the data split into subjects?

i. Subjects are identified from subdirectory names matching `sub-*` pattern. The subject ID is also extracted from the NWB file's `general/subject/subject_id` field.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
# ...
subject_id = f['general/subject/subject_id'][()].decode()
```

iii. The AI confirmed 11 subjects matching the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. All `.nwb` files within each subject directory are collected and sorted.

ii.
```python
files = sorted(glob(os.path.join(subj_dir, '*.nwb')))
for fpath in files:
    sessions.append({...})
```

iii. The AI found 152 total sessions, consistent with the paper (10 subjects x 14 sessions + 1 subject x 12 sessions).

## 1-d. How are the data split into trials?

i. Trial starts are identified by `trial_start > 0` and trial ends by `teleport > 0`. When counts don't match, each trial start is matched to the first teleport event after it. Data between `[trial_start, teleport)` constitutes a trial.

ii.
```python
trial_start_inds = np.where(trial_start > 0)[0]
teleport_inds = np.where(teleport_sig > 0)[0]
n_trials = len(trial_start_inds)

if len(teleport_inds) != n_trials:
    matched_teleports = []
    for ts in trial_start_inds:
        tp_after = teleport_inds[teleport_inds > ts]
        if len(tp_after) > 0:
            matched_teleports.append(tp_after[0])
    teleport_inds = np.array(matched_teleports)
```

iii. CONVERSION_NOTES.md states that trial_start and teleport signals define trial boundaries, consistent with the paper.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 timepoints (`n_tp < 2`) are skipped. Sessions with fewer than 2 trials are also skipped.

ii.
```python
if n_tp < 2:
    continue
# ...
if n_trials < 2:
    print(f"  WARNING: Only {n_trials} trials, skipping session")
    return None
```

iii. CONVERSION_NOTES.md mentions short trial filtering, but the actual code threshold is 2, not 50 as the notes suggest.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `Deconvolved` processing module in the NWB file, reading data from each imaging plane (e.g., `plane0`, `plane1`).

ii.
```python
deconv_group = f['processing/ophys/Deconvolved']
planes = sorted(deconv_group.keys())
for plane in planes:
    d = f[f'processing/ophys/Deconvolved/{plane}/data'][:]
    deconv_list.append(d)
deconv = np.concatenate(deconv_list, axis=1)
```

iii. The CONVERSION_NOTES.md states that NWB Deconvolved data is already fully processed through the dF/F pipeline, so it should be used directly.

## 2-b. How is the `neural` data processed?

i. After loading deconvolved data from all planes: (1) cells are filtered by `iscell`, (2) an additional interneuron exclusion step removes cells with speed-dF/F correlation > 0.5, (3) data is converted to float32.

ii.
```python
# iscell filtering
cell_mask_concat = np.array([iscell[concat_to_iscell[i], 0] == 1 for i in range(n_rois_concat)])

# Interneuron exclusion via speed-dFF correlation
f_corrected = fluorescence - 0.7 * neuropil_data
f_median = np.median(f_corrected, axis=0, keepdims=True)
f_median[f_median == 0] = 1
dff_simple = (f_corrected - f_median) / np.abs(f_median)
# ... computes correlation with speed ...
for i, col_idx in enumerate(accepted_cols):
    if corrs[i] > 0.5:
        interneuron_mask[col_idx] = True

final_cell_mask = cell_mask_concat & ~interneuron_mask
neural_all = deconv[:, final_cell_mask].T
neural_all = neural_all.astype(np.float32)
```

iii. CONVERSION_NOTES.md documents: "Applied iscell filter + interneuron exclusion (speed-dFF corr > 0.5)". The interneuron exclusion was inspired by the reference code's `get_timeseries_data()` which filters cells with high speed correlation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering: (1) Suite2p `iscell` classification (iscell[:,0] == 1), and (2) interneuron exclusion based on speed-dF/F correlation > 0.5.

ii. See 2-b code snippets.

iii. CONVERSION_NOTES.md documents that interneuron exclusion rate was "0.42 +/- 0.85%" which was low and consistent with observations.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start by extracting the slice from `trial_start_ind` to `teleport_ind`. The first timepoint of each trial corresponds to the trial start event.

ii.
```python
trial_neural = neural_all[:, si:ei].copy()
```

iii. Since all data shares the same frame-rate time axis, slicing by trial boundaries automatically aligns to trial start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is computed from the nominal imaging rate (15.5078125 Hz), giving ~64.48 ms per bin. No rebinning is applied; the data is kept at the native imaging frame rate.

ii.
```python
IMAGING_RATE_NOMINAL = 15.5078125  # Hz
# ...
time_bin_ms = 1000.0 / IMAGING_RATE_NOMINAL
```

iii. CONVERSION_NOTES.md confirms "Time bin: 64.48 ms" and that behavior data is already aligned to the imaging frame rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the frame index within the trial and the imaging rate, not from stored timestamps.

ii.
```python
frame_time = 1.0 / imaging_rate  # seconds per frame
# ...
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
```

iii. The AI used a computed time axis rather than the behavior timestamps stored in the NWB file.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Frame indices (0 to n_tp-1) are multiplied by the frame duration (1/imaging_rate). The result is in seconds.

ii.
```python
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
```

iii. This gives uniformly spaced time values starting at 0, assuming constant frame rate.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time array has the same number of elements as the neural trial slice (n_tp = ei - si), ensuring 1:1 alignment with neural time bins.

ii.
```python
n_tp = ei - si
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
trial_input_tv = time_from_start.reshape(1, -1)  # (1, n_tp)
```

iii. Same time axis by construction.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The AI derives environment type from the NWB `identifier` field (scene name), NOT from the `environment` behavior time series. The scene name is parsed to determine Env1 (0) vs Env2 (1) for each trial.

ii.
```python
identifier = f['identifier'][()].decode()
scene = parse_scene(identifier)
# ...
rz_labels, rz_coords, env_per_trial = get_reward_zone_labels(scene, n_trials)
# ...
trial_env = env_per_trial.copy()
```

```python
def _parse_zone_and_env(parts):
    for p in parts:
        if p.startswith('Env'):
            env_num = int(p.replace('Env', ''))
            env = env_num - 1  # 0-indexed
```

iii. CONVERSION_NOTES.md says "Use env_per_trial from scene parsing (more reliable for cross-env switches)".

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The scene name is parsed to extract environment numbers. For single-zone sessions, one environment applies to all trials. For switch sessions (containing `_to_`), the first 30 trials get the "from" environment and remaining trials get the "to" environment.

ii.
```python
def get_reward_zone_labels(scene, n_trials, change_trial=30):
    if '_to_' in scene:
        from_zone, from_env = _parse_zone_and_env(from_parts)
        to_zone, to_env = _parse_zone_and_env(to_parts)
        env_per_trial[:change_trial] = from_env
        env_per_trial[change_trial:] = to_env
    else:
        zone, env = _parse_zone_and_env(parts)
        env_per_trial[:] = env
```

iii. The hardcoded `change_trial=30` assumes all switches happen at trial 30, based on the paper's description.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the loop counter over trials (0-indexed within session), not from the `trial number` behavior time series in the NWB file.

ii.
```python
for t in range(n_trials):
    # ...
    trial_num = np.float32(t)
```

iii. The AI noted that the stored `trial number` variable didn't always agree with `trial_start`.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing; the trial number is the 0-based loop index, broadcast to all timepoints in the trial.

ii.
```python
trial_num = np.float32(t)
# ...
np.full((1, n_tp), trial_num, dtype=np.float32)
```

iii. Simple sequential indexing within the session.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps. Reward event timestamps are matched to behavior frame times to create a binary reward-per-frame signal.

ii.
```python
reward_timestamps = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
# ...
reward_frames = np.zeros(n_timepoints, dtype=np.float32)
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
    reward_frames[frame_idx] = 1.0
```

iii. Uses the sparse reward event timestamps mapped to frame indices.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the reward outcome is determined by checking if any reward frame fell within [trial_start, teleport). Previous trial outcome for trial t is the reward outcome of trial t-1. Trial 0 gets 0 (no previous trial).

ii.
```python
trial_rewarded = np.zeros(n_trials, dtype=np.int64)
for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    if np.any(reward_frames[si:ei] > 0):
        trial_rewarded[t] = 1

prev_outcome = np.zeros(n_trials, dtype=np.int64)
for t in range(1, n_trials):
    prev_outcome[t] = trial_rewarded[t - 1]
```

iii. Binary previous-trial reward outcome as specified in instructions.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` behavior time series and the reward zone coordinates. The reward zone label (A/B/C) is determined by parsing the NWB identifier/scene name, and the zone coordinates come from a hardcoded dictionary.

ii.
```python
REWARD_ZONE_DICT = {
    'A': [80, 130],
    'B': [200, 250],
    'C': [320, 370],
}
# ...
position = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
# ...
rz_labels, rz_coords, env_per_trial = get_reward_zone_labels(scene, n_trials)
# ...
rz_start = rz_coords[t, 0]
rz_end = rz_coords[t, 1]
signed_dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. Reward zone coordinates from `behavior.py` in the reference code, mapped from X/Y/Z to A/B/C.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance: negative when before zone (position < zone start), positive when past zone (position > zone end), zero when inside zone.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    dist = np.zeros_like(position)
    before = position < rz_start
    after = position > rz_end
    dist[before] = position[before] - rz_start  # negative
    dist[after] = position[after] - rz_end       # positive
    return dist
```

iii. Matches the paper's concept of reward-relative distance.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using manual conditional logic with boundaries at -50, -10, 0, +10, +50.

ii.
```python
def discretize_distance(distance):
    bins = np.zeros(len(distance), dtype=np.int64)
    bins[distance < -50] = 0
    bins[(distance >= -50) & (distance < -10)] = 1
    bins[(distance >= -10) & (distance < 0)] = 2
    bins[distance == 0] = 3
    bins[(distance > 0) & (distance <= 10)] = 4
    bins[(distance > 10) & (distance <= 50)] = 5
    bins[distance > 50] = 6
    return bins
```

iii. Bin boundaries match the instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position data is sliced using the same trial boundary indices as neural data (`si:ei`), ensuring temporal alignment.

ii.
```python
trial_pos = position[si:ei]
signed_dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. Same frame indexing as neural data.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
# ...
trial_pos = position[si:ei]
```

iii. Direct position from NWB file.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extraction and discretization.

ii.
```python
trial_pos = position[si:ei]
pos_bins = discretize_position(trial_pos)
```

iii. Raw position values are used.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins using `floor(position / 90)` clipped to [0, 4], giving bins [0-90), [90-180), [180-270), [270-360), [360+) cm.

ii.
```python
def discretize_position(position):
    bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
    return bins
```

iii. The AI chose 90 cm bins to span the 450 cm track in 5 equal divisions (450/5 = 90).

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indexing as neural data.

ii.
```python
trial_pos = position[si:ei]
```

iii. Aligned by construction through shared trial boundary indices.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = f['processing/behavior/BehavioralTimeSeries/lick/data'][:]
```

iii. Direct lick data from NWB.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Two processing steps: (1) Lick sensor error correction: if >30% of frames in a trial have lick count > 2, that trial's licks are set to 0. (2) Binary thresholding: any positive lick value becomes 1.

ii.
```python
# Error correction
lick_binary = lick.copy()
lick_binary[lick_binary > 0] = 1
for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    trial_lick = lick[si:ei]
    if len(trial_lick) > 0 and (np.sum(trial_lick > 2) / len(trial_lick)) > 0.30:
        lick_binary[si:ei] = 0

# In trial loop:
lick_out = (trial_lick > 0).astype(np.int64)
```

iii. CONVERSION_NOTES.md references the reference code function `correct_lick_sensor_error()` in `behavior.py`.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indexing as neural data.

ii.
```python
trial_lick = lick_binary[si:ei]
```

iii. Aligned by construction.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB `identifier` field which contains the scene name (e.g., `Env1_LocationB_to_A`). NOT from the `reward_zone` behavior time series.

ii.
```python
identifier = f['identifier'][()].decode()
scene = parse_scene(identifier)
rz_labels, rz_coords, env_per_trial = get_reward_zone_labels(scene, n_trials)
# ...
zone_map = {'A': 0, 'B': 1, 'C': 2}
rz_loc = zone_map.get(rz_labels[t], 0)
```

iii. The AI parsed the scene name to determine reward zone labels, with a hardcoded switch at trial 30 for switch sessions.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene name parsing determines zone label (A/B/C) per trial. For switch sessions, trials 0-29 get the "from" zone and trials 30+ get the "to" zone. Mapped to integers: A=0, B=1, C=2.

ii.
```python
def get_reward_zone_labels(scene, n_trials, change_trial=30):
    if '_to_' in scene:
        labels[:change_trial] = from_zone
        labels[change_trial:] = to_zone
    else:
        labels[:] = zone
    return labels, coords, env_per_trial
```

iii. Based on the paper's description that reward zone switches occur after trial 30.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps.

ii.
```python
reward_timestamps = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
# ...
reward_frames = np.zeros(n_timepoints, dtype=np.float32)
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
    reward_frames[frame_idx] = 1.0
```

iii. Sparse reward events mapped to imaging frames.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are mapped to nearest behavior frames. Per trial, reward outcome is 1 if any reward frame falls within the trial boundaries, 0 otherwise.

ii.
```python
trial_rewarded = np.zeros(n_trials, dtype=np.int64)
for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    if np.any(reward_frames[si:ei] > 0):
        trial_rewarded[t] = 1
# ...
rew_outcome = int(trial_rewarded[t])
```

iii. Binary per-trial output matching instructions.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases handled:
- **Neural/behavior length mismatch**: Truncated to minimum of the two with a warning.
- **Short trials**: Trials with < 2 timepoints skipped.
- **Trial start/teleport mismatch**: Matching algorithm pairs each trial start to its nearest subsequent teleport.
- **NaN in neural data**: Replaced with 0.
- **Lick sensor errors**: Trials with >30% high-count lick frames set to 0.
- **Sessions with <2 trials**: Skipped entirely.

ii.
```python
if n_timepoints != n_behav:
    min_len = min(n_timepoints, n_behav)
    # ... truncate both to min_len

trial_neural = np.nan_to_num(trial_neural, nan=0.0)

if n_tp < 2:
    continue
```

iii. CONVERSION_NOTES.md documents off-by-one behavior/neural mismatches in 10 multi-plane sessions.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files with h5py (I/O bound, large files)
2. Computing dF/F for interneuron exclusion (loads fluorescence and neuropil data)
3. Reward timestamp matching (per-event argmin loop)
4. Saving the large pickle file (~9.4 GB)

ii. N/A

iii. Total conversion time was 762.5 seconds for 152 sessions (~5s per session).

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The reward timestamp matching uses a Python loop with `np.argmin` per event:
```python
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
    reward_frames[frame_idx] = 1.0
```
This could be vectorized with `np.searchsorted`. The trial-matching loop for teleport indices could also be vectorized.

ii. See above.

iii. The per-trial processing loop is the natural structure given variable-length trials.

## 13-c. What processing does the code repeat multiple times?

i. The code processes each session in a single pass (no separate survey step). However, the interneuron exclusion loads fluorescence and neuropil data in addition to the deconvolved data, which adds overhead. The fluorescence/neuropil data is only used for this filtering step and discarded afterward.

ii. N/A

iii. Unlike the reference which has a separate survey step, the AI processes everything in one pass per session.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads and processes fluorescence and neuropil data (computing a simple dF/F) solely for the interneuron exclusion step. This additional processing is not part of the reference solution's approach to cell filtering and adds both computational overhead and algorithmic complexity. The interneuron exclusion itself (speed-dFF correlation > 0.5) goes beyond the reference's iscell-only filtering.

ii.
```python
fl = f[f'processing/ophys/Fluorescence/{plane}/data'][:]
ne = f[f'processing/ophys/Neuropil/{plane}/data'][:]
# ...
f_corrected = fluorescence - 0.7 * neuropil_data
f_median = np.median(f_corrected, axis=0, keepdims=True)
dff_simple = (f_corrected - f_median) / np.abs(f_median)
```

iii. The speed output variable uses speed discretization bins including a `< 2 cm/s` category but does NOT apply a speed threshold to exclude low-speed timepoints from neural data, which the reference code's decoder does apply.
