# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all NWB files by scanning subdirectories of `data/` that start with `sub-`. It collects all `.nwb` files sorted per subject directory using `glob`. Each NWB file is opened with `h5py` (not `pynwb`) and data is read directly from HDF5 paths. All subjects, sessions, and trials within each NWB file are included.

ii.
```python
def find_nwb_files(data_dir='data'):
    """Find all NWB files organized by subject."""
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

Data loading uses `h5py`:
```python
with h5py.File(filepath, 'r') as f:
    subject_id = f['general/subject/subject_id'][()].decode()
    identifier = f['identifier'][()].decode()
    ...
    deconv = ...  # from processing/ophys/Deconvolved
    position = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
    ...
```

iii. The AI documented finding 152 NWB files across 11 subjects, matching the paper. The `h5py` approach is a valid alternative to `pynwb` for reading NWB files since NWB uses HDF5 format.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the subdirectory names (e.g., `sub-m3` -> `m3`). A unique sorted list is built from all session subject IDs.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
...
unique_subjects = sorted(set(all_subject_ids))
subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
```

iii. Standard approach matching the data organization. 11 subjects found, consistent with the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are processed individually in `process_session()`.

ii.
```python
for i, sess_info in enumerate(sessions_info):
    result = process_session(sess_info['filepath'], ...)
```

iii. Standard approach. 152 sessions total, consistent with the paper.

## 1-d. How are the data split into trials?

i. Trial boundaries are found using the `trial_start` and `teleport` behavior time series from the NWB file. Trial starts are indices where `trial_start > 0`. For trial ends, all indices where `teleport > 0` are found, then each trial start is matched to the first teleport index after it.

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
    n_trials = min(n_trials, len(teleport_inds))
    trial_start_inds = trial_start_inds[:n_trials]
```

iii. The AI used `trial_start` and `teleport` signals consistent with the reference. The matching approach handles cases where teleport counts differ from trial start counts. However, the reference uses teleport *onset* detection (`(teleport[1:] > 0) & (teleport[:-1] <= 0)`) to find the first sample of each teleport event, while the AI takes any `teleport > 0` index and matches to each trial start. This means the AI's teleport index may point to the first sample where teleport is positive, which is functionally similar but uses a different matching strategy.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 timepoints are skipped.

ii.
```python
if n_tp < 2:
    continue
```

iii. The AI uses a very lenient threshold (2 timepoints), while the reference uses 50 timepoints as minimum. The AI's CONVERSION_NOTES do not justify this threshold.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses the NWB `Deconvolved` field directly, reading from `processing/ophys/Deconvolved/{plane}/data`. It does NOT compute dF/F from raw Fluorescence and Neuropil.

ii.
```python
deconv_group = f['processing/ophys/Deconvolved']
planes = sorted(deconv_group.keys())
deconv_list = []
for plane in planes:
    d = f[f'processing/ophys/Deconvolved/{plane}/data'][:]
    deconv_list.append(d)
deconv = np.concatenate(deconv_list, axis=1)
```

iii. The AI's CONVERSION_NOTES state: "NWB `Deconvolved` data is the deconvolved events AFTER full dF/F processing pipeline (computed from the multi_anim_sess notebook)" and "Used deconvolved events from NWB directly (already fully processed)". The AI believed these were already properly processed by the paper's pipeline. However, the reference solution explicitly states that the NWB's `Deconvolved` array is NOT the signal the paper analyses -- it is suite2p's own deconvolution, not the paper's custom dF/F + OASIS pipeline.

## 2-b. How is the `neural` data processed?

i. The AI uses the Deconvolved data directly from the NWB file with no further processing (no dF/F computation, no neuropil subtraction, no baseline correction, no smoothing, no OASIS deconvolution). NaN values are replaced with 0.

ii.
```python
neural_all = deconv[:, final_cell_mask].T  # (n_neurons, n_timepoints)
neural_all = neural_all.astype(np.float32)
...
trial_neural = neural_all[:, si:ei].copy()
trial_neural = np.nan_to_num(trial_neural, nan=0.0)
```

iii. The AI assumed the NWB Deconvolved data was already fully processed. The reference solution instead copies the paper's `dff()` function and computes dF/F from raw Fluorescence and Neuropil, including neuropil subtraction (coef=0.7), maximin baseline estimation, smoothing, and OASIS deconvolution with tau=0.7.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) `iscell` filter from Suite2p curation, and (2) interneuron exclusion based on speed-dFF correlation > 0.5. For interneuron detection, the AI computes a simplified dF/F (neuropil-subtracted fluorescence divided by median) and correlates with speed.

ii.
```python
# iscell filter
cell_mask_concat = np.array([iscell[concat_to_iscell[i], 0] == 1 for i in range(n_rois_concat)])

# Interneuron exclusion
f_corrected = fluorescence - 0.7 * neuropil_data
f_median = np.median(f_corrected, axis=0, keepdims=True)
f_median[f_median == 0] = 1
dff_simple = (f_corrected - f_median) / np.abs(f_median)

valid_mask = (speed > 0) & (position >= 0) & ~np.isnan(speed)
...
speed_z = (speed_valid - speed_valid.mean()) / (speed_valid.std() + 1e-10)
dff_accepted = dff_simple[valid_mask][:, accepted_cols]
...
corrs = (speed_z @ dff_z) / len(speed_z)
for i, col_idx in enumerate(accepted_cols):
    if corrs[i] > 0.5:
        interneuron_mask[col_idx] = True

final_cell_mask = cell_mask_concat & ~interneuron_mask
```

iii. The `iscell` filter matches the reference. The interneuron exclusion uses the same threshold (0.5) as the reference. However, the dF/F used for interneuron correlation is a simplified version (median-based) rather than the paper's full maximin baseline dF/F. The reference computes the full dF/F using the paper's pipeline and then checks correlation. Also, the AI's `valid_mask` uses `speed > 0` and `position >= 0`, while the reference uses the NaN mask from the dF/F computation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start by slicing from the trial start index to the teleport index. No additional temporal shifting is needed since alignment is to trial start.

ii.
```python
trial_neural = neural_all[:, si:ei].copy()
```

iii. Aligning to trial start is the correct approach per the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the original imaging rate (~15.5 Hz, ~64.5 ms per frame). No rebinning is applied. The time bin size is computed as `1000.0 / IMAGING_RATE_NOMINAL` where `IMAGING_RATE_NOMINAL = 15.5078125`.

ii.
```python
IMAGING_RATE_NOMINAL = 15.5078125  # Hz
...
time_bin_ms = 1000.0 / IMAGING_RATE_NOMINAL
```

iii. No rebinning matches the reference approach. However, the AI hardcodes the imaging rate rather than reading it from each file, and does not account for multi-plane sessions where the per-plane rate is `rate/nplanes`. The reference reads the rate from each NWB file and computes `nplanes/rate*1000` for the time bin size.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the frame index and the hardcoded imaging rate, not from behavior timestamps.

ii.
```python
frame_time = 1.0 / imaging_rate  # seconds per frame
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
```

iii. The AI computes time from frame index rather than using behavior timestamps. The reference uses actual behavior timestamps (`timestamps[idx] - timestamps[idx][0]`). These should be very similar since the imaging rate is constant, but using actual timestamps is more precise.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The frame index (0, 1, 2, ...) is multiplied by `1.0 / imaging_rate` to get seconds from trial start.

ii.
```python
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
```

iii. Simple multiplication. The reference subtracts the first timestamp from all timestamps in the trial.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data use the same frame indices within each trial, so alignment is implicit by using the same slicing.

ii.
```python
si = trial_start_inds[t]
ei = teleport_inds[t]
trial_neural = neural_all[:, si:ei].copy()
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
```

iii. Both use the same indices, so alignment is automatic.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The AI derives environment type from parsing the NWB `identifier` field (scene name), using `get_reward_zone_labels()` and `_parse_zone_and_env()`. For switch sessions, the environment is parsed from the scene name format (e.g., `Env1_B_to_Env2_C`).

ii.
```python
identifier = f['identifier'][()].decode()
scene = parse_scene(identifier)
...
rz_labels, rz_coords, env_per_trial = get_reward_zone_labels(scene, n_trials)
...
trial_env = env_per_trial.copy()
```

```python
def _parse_zone_and_env(parts):
    env = 0
    zone = 'A'
    for p in parts:
        if p.startswith('Env'):
            env_num = int(p.replace('Env', ''))
            env = env_num - 1
        elif p.startswith('Location'):
            zone = p.replace('Location', '')
        elif len(p) == 1 and p in 'ABC':
            zone = p
    return zone, env
```

iii. The AI prefers the scene name approach because it's "more reliable for cross-env switches." The reference uses the `environment` behavior time series directly from the NWB file.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Scene name is parsed to extract environment number. For switch sessions, the environment changes at trial 30 (hardcoded `change_trial=30`). The value is 0 for Env1, 1 for Env2.

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

iii. Hardcoding the switch at trial 30 is based on the paper's description of switch sessions. The reference reads the environment variable directly from the behavior timeseries, which is more robust.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the loop index `t` over trials within a session.

ii.
```python
for t in range(n_trials):
    ...
    trial_num = np.float32(t)
```

iii. Same approach as the reference (sequential index within session).

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing; the loop index is used directly as the trial number, constant across all timepoints in the trial.

ii.
```python
trial_num = np.float32(t)
...
np.full((1, n_tp), trial_num, dtype=np.float32),
```

iii. Same as reference.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps. Reward timestamps are matched to behavior frame times.

ii.
```python
reward_timestamps = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
...
reward_frames = np.zeros(n_timepoints, dtype=np.float32)
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
    reward_frames[frame_idx] = 1.0
```

iii. Same source variable as the reference, though the AI uses `argmin(abs(...))` for matching while the reference uses `searchsorted`.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, check if any reward occurred in the previous trial. Trial 0 gets 0.

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

iii. Same logic as the reference: binary previous trial outcome. However, note that if a trial is filtered out (n_tp < 2), the "previous trial" is still computed correctly for non-filtered trials because the trial_rewarded array covers all original trials.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone coordinates. The reward zone for each trial is determined from the scene name in the NWB identifier.

ii.
```python
trial_pos = position[si:ei]
rz_start = rz_coords[t, 0]
rz_end = rz_coords[t, 1]
signed_dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The AI determines reward zone from scene name parsing, while the reference uses the `reward_zone` behavior variable with Viterbi smoothing. Both ultimately use the same reward zone coordinates (A=[80,130], B=[200,250], C=[320,370]).

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from position to nearest edge of the reward zone. Negative before zone, zero inside, positive after. Then discretized into 7 bins.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    dist = np.zeros_like(position)
    before = position < rz_start
    after = position > rz_end
    inside = ~before & ~after
    dist[before] = position[before] - rz_start
    dist[after] = position[after] - rz_end
    dist[inside] = 0.0
    return dist
```

iii. Same logic as the reference's `compute_distance_to_reward_zone()`.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized using explicit boolean indexing rather than `np.digitize`. Bin boundaries: < -50, -50 to -10, -10 to 0, exactly 0, 0 to 10, 10 to 50, > 50.

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

iii. The bin edges match the instructions. The reference uses `np.digitize` with `[-inf, -50, -10, 0, 1e-6, 10, 50, inf]`. The AI's approach is functionally equivalent but uses explicit boolean conditions. One difference: the AI uses `distance == 0` for bin 3 (exact zero only), while the reference uses `0` to `1e-6` which is essentially the same. Also, the AI uses `<= 10` and `<= 50` for the positive side boundaries while the instructions say `>0 to +10` and `+10 to +50`, which creates a slight ambiguity at exact boundaries.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indices used for both neural and position data within each trial.

ii.
```python
trial_pos = position[si:ei]
trial_neural = neural_all[:, si:ei].copy()
```

iii. Alignment is automatic through shared indexing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
...
trial_pos = position[si:ei]
```

iii. Same as reference.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Discretized into 5 bins using `np.clip(np.floor(position / 90.0), 0, 4)`.

ii.
```python
def discretize_position(position):
    bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
    return bins
```

iii. The formula `floor(position/90)` with clipping produces the same 5 bins as `np.digitize` with edges `[-inf, 90, 180, 270, 360, inf]` used by the reference, since the track is 450 cm (5 * 90 cm bins).

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five bins of 90 cm each: 0-90, 90-180, 180-270, 270-360, 360-450.

ii.
```python
bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
```

iii. Matches the instructions' "5 equal-sized bins spanning the 450 cm track."

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices, no additional alignment needed.

ii. Same `si:ei` slicing as neural data.

iii. Alignment is automatic.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = f['processing/behavior/BehavioralTimeSeries/lick/data'][:]
```

iii. Same as reference.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any value > 0 is mapped to 1. Additionally, a lick sensor error correction is applied: if more than 30% of frames in a trial have lick count > 2, the trial's lick values are set to 0.

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
```

iii. The binarization matches the reference. The lick sensor error correction is inspired by the paper's `correct_lick_sensor_error()` function. The reference does not include this correction. Setting erroneous licks to 0 instead of NaN is a reasonable choice for decoder compatibility.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices, no additional alignment needed.

ii. Same `si:ei` slicing as neural data.

iii. Alignment is automatic.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB `identifier` field, which contains the scene name (e.g., `Env1_LocationB_to_A`). The scene is parsed to determine reward zone labels per trial.

ii.
```python
identifier = f['identifier'][()].decode()
scene = parse_scene(identifier)
rz_labels, rz_coords, env_per_trial = get_reward_zone_labels(scene, n_trials)
...
zone_map = {'A': 0, 'B': 1, 'C': 2}
rz_loc = zone_map.get(rz_labels[t], 0)
```

iii. The AI parses the scene name to determine reward zones. The reference uses the `reward_zone` behavior time series combined with `position` to detect where the reward zone is, then uses a Viterbi algorithm to assign consistent labels. The scene-name approach is simpler but assumes the switch always happens at trial 30 (hardcoded), which may not perfectly match reality.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene name parsing determines the "from" and "to" reward zones. For switch sessions, the transition is assumed to occur at trial 30. Zone labels are mapped to integers: A=0, B=1, C=2.

ii.
```python
def get_reward_zone_labels(scene, n_trials, change_trial=30):
    if '_to_' in scene:
        labels[:change_trial] = from_zone
        labels[change_trial:] = to_zone
        ...
    else:
        labels[:] = zone
        ...
```

iii. The hardcoded switch at trial 30 is based on the paper's description. The reference approach (Viterbi on actual reward positions) is more data-driven and handles edge cases better.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps.

ii.
```python
reward_timestamps = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
...
reward_frames = np.zeros(n_timepoints, dtype=np.float32)
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
    reward_frames[frame_idx] = 1.0
```

iii. Same source as reference.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are matched to nearest behavior frame. Per trial, the output is 1 if any reward event occurred within the trial window, 0 otherwise.

ii.
```python
trial_rewarded = np.zeros(n_trials, dtype=np.int64)
for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    if np.any(reward_frames[si:ei] > 0):
        trial_rewarded[t] = 1
...
rew_outcome = int(trial_rewarded[t])
```

iii. Same logic as reference. Uses `argmin(abs(...))` instead of `searchsorted` for timestamp matching, but functionally equivalent.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Neural/behavior length mismatch**: If neural and behavior arrays differ in length, both are cropped to the minimum.
- **Short trials**: Trials with fewer than 2 timepoints are skipped.
- **NaN neural data**: NaN values in neural data are replaced with 0 using `np.nan_to_num`.
- **Lick sensor errors**: Trials with >30% frames having lick count >2 have lick set to 0.
- **Trial start/teleport mismatch**: If teleport count differs from trial start count, teleports are matched to the nearest trial start.

ii.
```python
if n_timepoints != n_behav:
    min_len = min(n_timepoints, n_behav)
    ...

if n_tp < 2:
    continue

trial_neural = np.nan_to_num(trial_neural, nan=0.0)
```

iii. The neural/behavior mismatch handling matches the reference. The NaN replacement with 0 is a reasonable choice for the decoder but masks potential issues. The lick sensor correction is an additional step not in the reference.

## 13-a. What are the most time-consuming steps of the code?

i. The AI's code has a simpler structure since it doesn't compute dF/F:
1. Loading NWB files with h5py (I/O bound)
2. Interneuron correlation computation (vectorized but still requires loading fluorescence/neuropil)
3. Saving the large pickle file

ii. N/A

iii. Total processing time was 762.5 seconds for 152 sessions.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The reward timestamp matching loop uses `argmin(abs(...))` per reward event, which could be vectorized with `searchsorted`. The lick sensor error correction loop iterates per trial.

ii.
```python
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
    reward_frames[frame_idx] = 1.0
```

iii. The `searchsorted` approach used by the reference is more efficient.

## 13-c. What processing does the code repeat multiple times?

i. The code loads fluorescence and neuropil data even though the neural data used is the Deconvolved data (which is loaded separately). This is done solely for the interneuron correlation check.

ii.
```python
fl = f[f'processing/ophys/Fluorescence/{plane}/data'][:]
ne = f[f'processing/ophys/Neuropil/{plane}/data'][:]
...
d = f[f'processing/ophys/Deconvolved/{plane}/data'][:]
```

iii. Reading fluorescence and neuropil for interneuron detection while using deconvolved for neural output is somewhat redundant I/O.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads and reads the NWB `identifier` field, parses scene names, and computes environment labels from scene names. The reference reads environment directly from the behavior time series, which is simpler. The AI also computes a simplified dF/F for interneuron detection that is not used for the final neural data.

ii. N/A

iii. The scene name parsing is more complex than necessary given that environment type is directly available in the behavior data.
