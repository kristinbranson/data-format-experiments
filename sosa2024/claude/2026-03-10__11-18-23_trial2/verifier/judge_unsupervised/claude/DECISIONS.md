# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files from the `data/` directory by iterating over subject directories (`sub-*`) and finding `.nwb` files within each. Each NWB file corresponds to one session. Files are opened with `h5py` and data is extracted from standardized HDF5 paths within the NWB structure. Neural data (Deconvolved, Fluorescence, Neuropil), behavioral data (position, speed, lick, environment, trial_start, teleport, Reward), and metadata (subject_id, identifier/scene, imaging_rate, location) are loaded per session. Multi-plane data (for subjects m17, m18) is handled by concatenating across planes.

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

```python
with h5py.File(filepath, 'r') as f:
    subject_id = f['general/subject/subject_id'][()].decode()
    identifier = f['identifier'][()].decode()
    # ... load all neural and behavioral data
```

iii. The AI documented in CONVERSION_NOTES.md that data is in NWB format organized as `data/sub-{id}/sub-{id}_ses-{ses}_behavior+ophys.nwb`, with 11 subjects and 152 sessions total. The approach of loading from subject directories is consistent with the data organization.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the directory names (`sub-m3`, `sub-m4`, etc.) and from the `general/subject/subject_id` field in each NWB file. After processing all sessions, unique subject IDs are collected, sorted, and a `subject_idx` array maps each session to its subject.

ii.
```python
unique_subjects = sorted(set(all_subject_ids))
subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
```

iii. The AI identified 11 subjects matching the paper's description. Subject IDs are extracted from NWB metadata rather than directory names for the actual mapping.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Sessions are processed sequentially, and results from each are appended to lists. Sessions with fewer than 2 valid trials are skipped. The `--sample` mode selects 2 sessions (first and middle).

ii.
```python
for i, sess_info in enumerate(sessions_info):
    result = process_session(sess_info['filepath'], ...)
    if result is None:
        continue
    if result['n_trials'] < 2:
        continue
    all_neural.append(result['neural'])
    # ...
```

iii. One NWB file = one session is consistent with the data organization described in CONVERSION_NOTES.md (152 NWB files = 152 sessions).

## 1-d. How are the data split into trials?

i. Trials are identified using the `trial_start` and `teleport` signals from the behavioral timeseries in the NWB file. Trial boundaries are defined from the frame where `trial_start > 0` to the frame where `teleport > 0`. Trials with fewer than 2 timepoints are skipped.

ii.
```python
trial_start_inds = np.where(trial_start > 0)[0]
teleport_inds = np.where(teleport_sig > 0)[0]
n_trials = len(trial_start_inds)

# Match teleport to trial start
if len(teleport_inds) != n_trials:
    matched_teleports = []
    for ts in trial_start_inds:
        tp_after = teleport_inds[teleport_inds > ts]
        if len(tp_after) > 0:
            matched_teleports.append(tp_after[0])
    teleport_inds = np.array(matched_teleports)

# Per trial:
for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    n_tp = ei - si
    if n_tp < 2:
        continue
```

iii. The AI documented that trial boundaries use `trial_start` and `teleport` signals. The reference code uses similar trial segmentation logic. Matching teleports to trial starts handles edge cases where counts differ.

## 1-e. How are trials filtered based on quality controls?

i. The only trial-level filtering is: (1) trials with fewer than 2 timepoints are skipped, and (2) sessions with fewer than 2 valid trials are skipped entirely. There is no filtering based on behavioral criteria (e.g., minimum running speed, minimum distance traveled). The lick sensor error correction zeroes out lick data for trials where >30% of frames have cumulative lick count > 2, but the trial itself is not excluded.

ii.
```python
if n_tp < 2:
    continue

# Lick sensor error correction per trial
for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    trial_lick = lick[si:ei]
    if len(trial_lick) > 0 and (np.sum(trial_lick > 2) / len(trial_lick)) > 0.30:
        lick_binary[si:ei] = 0
```

iii. The AI noted that no speed threshold is applied for frame inclusion because speed is a decoder output. This is a reasonable decision given the decoder task specifications, though the reference code uses speed < 2 cm/s filtering for place cell analyses.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the `Deconvolved` data stored in the NWB files at `processing/ophys/Deconvolved/{plane}/data`. For cell filtering, the `Fluorescence` and `Neuropil` data are also loaded, along with the `iscell` and `planeIdx` arrays from `ImageSegmentation/PlaneSegmentation`.

ii.
```python
deconv_list = []
flu_list = []
neu_list = []
for plane in planes:
    d = f[f'processing/ophys/Deconvolved/{plane}/data'][:]
    fl = f[f'processing/ophys/Fluorescence/{plane}/data'][:]
    ne = f[f'processing/ophys/Neuropil/{plane}/data'][:]
    deconv_list.append(d)
    flu_list.append(fl)
    neu_list.append(ne)

deconv = np.concatenate(deconv_list, axis=1)
fluorescence = np.concatenate(flu_list, axis=1)
neuropil_data = np.concatenate(neu_list, axis=1)
```

iii. The AI correctly identified that NWB files contain already-processed deconvolved events (the full dF/F pipeline was already applied during NWB creation). The Fluorescence and Neuropil are only loaded for interneuron exclusion, not for the final neural data.

## 2-b. How is the `neural` data processed?

i. The neural data processing involves: (1) Concatenating deconvolved data across planes for multi-plane sessions, (2) Applying cell filtering (iscell + interneuron exclusion), (3) Transposing to (n_neurons, n_timepoints), (4) Converting to float32, (5) Segmenting into per-trial arrays, and (6) Replacing NaNs with 0.

ii.
```python
neural_all = deconv[:, final_cell_mask].T  # (n_neurons, n_timepoints)
neural_all = neural_all.astype(np.float32)

# Per trial:
trial_neural = neural_all[:, si:ei].copy()
trial_neural = np.nan_to_num(trial_neural, nan=0.0)
```

iii. The AI documented that the NWB deconvolved data is used directly since it's already fully processed through the dF/F pipeline (neuropil subtraction, maximin baseline, smoothing, OASIS deconvolution). No additional temporal processing (rebinning, smoothing) is applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage cell filtering is applied: (1) Suite2p's `iscell` classification (iscell[:,0] == 1 means accepted cell), and (2) Interneuron exclusion based on correlation between dF/F and running speed (threshold > 0.5). No additional quality filtering (e.g., minimum firing rate, signal-to-noise) is applied.

ii.
```python
# Step 1: iscell filter
cell_mask_concat = np.array([iscell[concat_to_iscell[i], 0] == 1 for i in range(n_rois_concat)])

# Step 2: Interneuron exclusion
f_corrected = fluorescence - 0.7 * neuropil_data
f_median = np.median(f_corrected, axis=0, keepdims=True)
f_median[f_median == 0] = 1
dff_simple = (f_corrected - f_median) / np.abs(f_median)

valid_mask = (speed > 0) & (position >= 0) & ~np.isnan(speed)
# ... vectorized correlation computation ...
for i, col_idx in enumerate(accepted_cols):
    if corrs[i] > 0.5:
        interneuron_mask[col_idx] = True

final_cell_mask = cell_mask_concat & ~interneuron_mask
```

iii. The AI documented the interneuron exclusion rate as 0.42 +/- 0.85% matching the paper. The dF/F computation for interneuron detection uses a simplified approach (median-based) rather than the full maximin baseline from the reference code, but this is only used for the correlation threshold check, not for the actual neural data.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of each trial. Each trial's neural data spans from `trial_start_inds[t]` to `teleport_inds[t]` (exclusive). The trial start frame is the first frame in the trial, so off_start = 0.0. Trial end varies per trial (off_end = None).

ii.
```python
trial_neural = neural_all[:, si:ei].copy()
# where si = trial_start_inds[t], ei = teleport_inds[t]
```

```python
'temporal_alignment_event': 'start of trial (entry to linear track)',
'off_start': 0.0,
'off_end': None,
```

iii. The instructions specify "Temporally align based on start of the trial." The AI correctly aligns to trial start and notes that trial end is variable.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native imaging frame rate of ~15.5078125 Hz, corresponding to ~64.48 ms per time bin. No temporal rebinning is applied; the data remains at the original imaging frame rate.

ii.
```python
IMAGING_RATE_NOMINAL = 15.5078125  # Hz
time_bin_ms = 1000.0 / IMAGING_RATE_NOMINAL
# metadata:
'time_bin_size': time_bin_ms,  # 64.48 ms
```

iii. The AI documented the imaging rate as matching the paper's ~15.5 Hz. The reference code also works at the native imaging rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Time from start of trial is not derived from any raw data variable. It is computed from the frame index within each trial and the imaging rate.

ii.
```python
frame_time = 1.0 / imaging_rate  # seconds per frame
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
```

iii. This is a straightforward computation: frame 0 = 0.0 seconds, frame k = k * (1/imaging_rate) seconds.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Time is computed as frame_index * (1/imaging_rate). The imaging_rate is read from the NWB file's `general/optophysiology/ImagingPlane/imaging_rate` field. The result is a float32 array of shape (n_timepoints,).

ii.
```python
frame_time = 1.0 / imaging_rate
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
```

iii. Simple linear computation from frame indices.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Time from trial start is computed over the same frame range as the neural data (si:ei), so it is inherently aligned. Both have n_tp = ei - si timepoints. The time array starts at 0.0 for the first frame of the trial.

ii.
```python
n_tp = ei - si
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
trial_neural = neural_all[:, si:ei].copy()
```

iii. Same frame indices ensure alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Environment type is derived from parsing the scene name in the NWB `identifier` field. For cross-environment switch sessions, the scene name contains both environments (e.g., `Env1_B_to_Env2_C`). For within-environment sessions, it's a single environment. The change happens at trial 30 (hardcoded).

ii.
```python
scene = parse_scene(identifier)  # e.g. 'Env1_LocationB_to_A'
rz_labels, rz_coords, env_per_trial = get_reward_zone_labels(scene, n_trials)
trial_env = env_per_trial.copy()
```

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

iii. The AI chose to parse the scene name from the identifier rather than using the NWB `environment` field directly. The AI stated this is "more reliable for cross-env switches." The NWB environment field actually does contain per-frame environment values that correctly reflect the switch, so this introduces potential issues for within-env switches where the scene parser assigns the same env to both before and after the switch.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Environment type is encoded as binary: 0 = Env1, 1 = Env2. It is a per-trial value that is broadcast to all timepoints within the trial by creating a (1, n_tp) array filled with the value.

ii.
```python
env_type = np.float32(trial_env[t])
trial_input = np.vstack([
    trial_input_tv,
    np.full((1, n_tp), env_type, dtype=np.float32),
    # ...
])
```

iii. Binary encoding matches the instruction specification (ENV1 vs ENV2).

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the loop index `t` in the trial iteration, representing the 0-indexed trial number within the session. It is NOT derived from the `trial number` field in the NWB behavioral timeseries.

ii.
```python
trial_num = np.float32(t)
```

iii. The AI uses the iteration index rather than the NWB's `trial number` field. These should generally be equivalent for valid trials, though the NWB field might contain the "official" trial numbering from the experiment.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial number is simply the 0-indexed loop counter `t`, cast to float32 and broadcast to all timepoints as a per-trial input.

ii.
```python
trial_num = np.float32(t)
trial_input = np.vstack([
    # ...
    np.full((1, n_tp), trial_num, dtype=np.float32),
    # ...
])
```

iii. No special processing. Note that trials with < 2 timepoints are skipped, so the trial numbering may have gaps if any trials are dropped (though the `continue` statement skips those trials).

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from the `Reward/timestamps` field in the NWB file, which contains sparse timestamps of reward delivery events. These timestamps are matched to behavior frame times to determine which trials were rewarded.

ii.
```python
reward_timestamps = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]

# Match to frames
reward_frames = np.zeros(n_timepoints, dtype=np.float32)
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
    reward_frames[frame_idx] = 1.0

# Per-trial reward
trial_rewarded = np.zeros(n_trials, dtype=np.int64)
for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    if np.any(reward_frames[si:ei] > 0):
        trial_rewarded[t] = 1
```

iii. The AI matches reward timestamps to imaging frames using nearest-neighbor matching, then checks if any reward frame falls within each trial's boundaries.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Previous trial outcome is a lagged version of the trial reward outcome. For trial t, it is the reward outcome of trial t-1. For the first trial (t=0), it defaults to 0 (omission). Binary: 0 = omitted, 1 = rewarded.

ii.
```python
prev_outcome = np.zeros(n_trials, dtype=np.int64)
for t in range(1, n_trials):
    prev_outcome[t] = trial_rewarded[t - 1]
```

iii. Simple one-trial lag. The instruction specifies "omitted = 0, rewarded = 1" which matches.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Distance to reward zone is derived from: (1) the `position` behavioral timeseries from NWB, and (2) the reward zone coordinates determined by parsing the scene name from the NWB identifier. Reward zone coordinates are hardcoded: A=[80,130], B=[200,250], C=[320,370].

ii.
```python
REWARD_ZONE_DICT = {
    'A': [80, 130],
    'B': [200, 250],
    'C': [320, 370],
}

trial_pos = position[si:ei]
rz_start = rz_coords[t, 0]
rz_end = rz_coords[t, 1]
signed_dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The AI uses hardcoded reward zone coordinates from the reference code's `behavior.py`. The mapping from code variables to zone labels is: X->A, Y->B, Z->C.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed as: negative if position is before the zone start, positive if after the zone end, zero if inside the zone. This is the distance to the nearest edge of the reward zone.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    dist = np.zeros_like(position)
    before = position < rz_start
    after = position > rz_end
    inside = ~before & ~after
    dist[before] = position[before] - rz_start  # negative
    dist[after] = position[after] - rz_end       # positive
    dist[inside] = 0.0
    return dist
```

iii. This computes distance to the nearest edge of the reward zone, with sign indicating direction.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Distance is discretized into 7 bins matching the instruction specification:
- 0: < -50 cm
- 1: -50 to -10 cm
- 2: -10 cm to < 0 cm
- 3: 0 cm (in zone)
- 4: >0 cm to +10 cm
- 5: +10 to +50 cm
- 6: > +50 cm

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

iii. Bin boundaries match the instruction specification exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Distance is computed from the position data for the same frame range (si:ei) as the neural data, so alignment is inherent.

ii.
```python
trial_pos = position[si:ei]
signed_dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
dist_bins = discretize_distance(signed_dist)
```

iii. Same frame indices as neural data ensure alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is derived from the `position` behavioral timeseries in the NWB file.

ii.
```python
trial_pos = position[si:ei]
pos_bins = discretize_position(trial_pos)
```

iii. Direct use of the position field.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is discretized into 5 equal-sized bins of 90 cm each (track length = 450 cm): [0-90), [90-180), [180-270), [270-360), [360-450].

ii.
```python
def discretize_position(position):
    bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
    return bins
```

iii. The 450cm track divided into 5 bins gives 90cm per bin, matching the instruction for "5 equal-sized bins."

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is divided into 5 equal bins using floor division by 90, clipped to [0, 4]:
- 0: 0-90 cm
- 1: 90-180 cm
- 2: 180-270 cm
- 3: 270-360 cm
- 4: 360-450 cm

ii.
```python
bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
```

iii. Equal binning of the 450cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position is from the same frame range (si:ei) as neural data, ensuring inherent alignment.

ii.
```python
trial_pos = position[si:ei]
```

iii. Same frame indices.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick is derived from the `lick` behavioral timeseries in the NWB file.

ii.
```python
lick = f['processing/behavior/BehavioralTimeSeries/lick/data'][:]
```

iii. Direct loading from NWB.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick processing involves: (1) binarizing cumulative lick counts (any value > 0 becomes 1), (2) correcting lick sensor errors by setting licks to 0 for trials where >30% of frames have cumulative count > 2.

ii.
```python
lick_binary = lick.copy()
lick_binary[lick_binary > 0] = 1

# Lick sensor error correction per trial
for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    trial_lick = lick[si:ei]
    if len(trial_lick) > 0 and (np.sum(trial_lick > 2) / len(trial_lick)) > 0.30:
        lick_binary[si:ei] = 0

# Per trial:
lick_out = (trial_lick > 0).astype(np.int64)
```

iii. The AI documented that the reference code's `correct_lick_sensor_error()` function uses a similar approach. The threshold of 30% matches the reference code behavior.py.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick data is from the same frame range (si:ei) as neural data.

ii.
```python
trial_lick = lick_binary[si:ei]
```

iii. Same frame indices.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward zone location is derived from parsing the scene name in the NWB `identifier` field, using hardcoded reward zone coordinates. The scene name encodes which reward zone(s) are active and when switches occur.

ii.
```python
identifier = f['identifier'][()].decode()
scene = parse_scene(identifier)
rz_labels, rz_coords, env_per_trial = get_reward_zone_labels(scene, n_trials)

# Per trial:
zone_map = {'A': 0, 'B': 1, 'C': 2}
rz_loc = zone_map.get(rz_labels[t], 0)
```

iii. The AI maps reward zone labels (A/B/C) to integers (0/1/2). For switch sessions, the first 30 trials get the "from" zone and trials 30+ get the "to" zone (hardcoded change_trial=30).

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene name is parsed to extract zone labels. For switch sessions (containing '_to_'), the zone switches at trial 30. The zone letter is mapped to an integer: A=0, B=1, C=2. This is a per-trial value broadcast to all timepoints.

ii.
```python
def get_reward_zone_labels(scene, n_trials, change_trial=30):
    if '_to_' in scene:
        labels[:change_trial] = from_zone
        labels[change_trial:] = to_zone
    else:
        labels[:] = zone

# Broadcast to timepoints:
np.full((1, n_tp), rz_loc, dtype=np.int64)
```

iii. The hardcoded change_trial=30 matches the paper's description of reward zone switches after trial 30.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from the `Reward/timestamps` field in the NWB file, which contains sparse event timestamps for reward delivery.

ii.
```python
reward_timestamps = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
```

iii. Sparse reward event timestamps from NWB.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are matched to behavior frame timestamps using nearest-neighbor matching. A binary frame-aligned reward signal is created, then checked per-trial: if any frame within the trial has a reward event, the trial is marked as rewarded (1), otherwise not (0). This per-trial value is broadcast to all timepoints.

ii.
```python
reward_frames = np.zeros(n_timepoints, dtype=np.float32)
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
    reward_frames[frame_idx] = 1.0

trial_rewarded = np.zeros(n_trials, dtype=np.int64)
for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    if np.any(reward_frames[si:ei] > 0):
        trial_rewarded[t] = 1

# Broadcast:
np.full((1, n_tp), rew_outcome, dtype=np.int64)
```

iii. The AI documented the ~15% omission rate matching the paper's description.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several data quality issues are handled:
1. **Neural/behavior length mismatch**: For multi-plane sessions where neural and behavior arrays have different lengths, both are truncated to the shorter length.
2. **NaN values in neural data**: Replaced with 0.0 using `np.nan_to_num`.
3. **Teleport/trial_start count mismatch**: Teleports are matched to trial starts by finding the first teleport after each trial start.
4. **Lick sensor errors**: Trials with >30% frames having cumulative lick count > 2 have lick data zeroed out.
5. **Sessions with < 2 trials**: Skipped entirely.
6. **Trials with < 2 timepoints**: Skipped.
7. **Division by zero in dF/F**: Median values of 0 are replaced with 1 to prevent division by zero.

ii.
```python
if n_timepoints != n_behav:
    min_len = min(n_timepoints, n_behav)
    deconv = deconv[:min_len]
    position = position[:min_len]
    # ...

trial_neural = np.nan_to_num(trial_neural, nan=0.0)

f_median[f_median == 0] = 1
```

iii. The AI handled edge cases discovered during development, particularly the multi-plane length mismatch issue which was fixed after the initial full conversion crashed.

## 13-a. What are the most time-consuming steps of the code?

i. Based on the conversion output, the most time-consuming steps are:
1. **Loading NWB files via h5py**: Reading large arrays from disk (especially multi-plane sessions).
2. **Interneuron exclusion**: Computing dF/F and correlations with speed for all accepted cells, involving neuropil subtraction, z-scoring, and dot product.
3. **Reward timestamp matching**: Using `np.argmin(np.abs(...))` in a loop for each reward event.

ii.
```python
# Interneuron check (vectorized but still computes over all cells):
f_corrected = fluorescence - 0.7 * neuropil_data
# ... full dF/F computation ...
corrs = (speed_z @ dff_z) / len(speed_z)

# Reward matching (loop over reward events):
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
```

iii. The AI noted that the interneuron correlation check is the bottleneck for large sessions and attempted to vectorize it. Total conversion time was ~762 seconds for 152 sessions (~5 seconds/session average).

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The following loops could potentially be vectorized:
1. **Reward timestamp matching** (lines 346-348): Each reward timestamp is matched to frames individually using `np.argmin`.
2. **Trial rewarded determination** (lines 352-356): Loops over each trial to check for rewards.
3. **Lick sensor error correction** (lines 368-373): Loops over trials to check lick error condition.
4. **iscell mapping** (line 282): List comprehension to build cell mask could use fancy indexing.

ii.
```python
# Could be vectorized:
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))

for t in range(n_trials):
    if np.any(reward_frames[si:ei] > 0):
        trial_rewarded[t] = 1
```

iii. The AI vectorized the interneuron correlation computation but left several per-trial loops.

## 13-c. What processing does the code repeat multiple times?

i. The code loads Fluorescence and Neuropil data for every session solely for the interneuron exclusion check. This includes computing a simplified dF/F (neuropil subtraction + median normalization) which is a separate computation from the already-processed deconvolved data. This is repeated for every session even though the interneuron exclusion rate is very low (~0.4%).

ii.
```python
# Loaded every session, only for interneuron check:
fl = f[f'processing/ophys/Fluorescence/{plane}/data'][:]
ne = f[f'processing/ophys/Neuropil/{plane}/data'][:]

f_corrected = fluorescence - 0.7 * neuropil_data
f_median = np.median(f_corrected, axis=0, keepdims=True)
dff_simple = (f_corrected - f_median) / np.abs(f_median)
```

iii. Loading Fluorescence and Neuropil data for interneuron detection is necessary for correctness, even if the exclusion rate is low.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and includes **speed** as a decoder output, which was specified in the instructions. However, the conversion also loads and processes several data streams that are used only for intermediate computations:
1. Fluorescence and Neuropil data are loaded only for interneuron detection but not included in the output.
2. The `scanning` and `autoreward` fields in the NWB behavior data are available but not used.
3. The `reward_zone` field in the NWB is not used (replaced by scene parsing).

ii.
```python
# Data loaded but not in final output:
fluorescence = np.concatenate(flu_list, axis=1)
neuropil_data = np.concatenate(neu_list, axis=1)
```

iii. The Fluorescence/Neuropil loading is necessary for the interneuron check, so it's not truly unnecessary. The unused NWB fields (scanning, autoreward, reward_zone) are not loaded either, so no processing is wasted on them.
