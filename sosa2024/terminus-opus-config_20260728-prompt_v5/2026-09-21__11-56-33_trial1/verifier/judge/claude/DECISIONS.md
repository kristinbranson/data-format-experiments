# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all NWB files by scanning sorted `sub-*` directories under `/app/data` and collecting all `.nwb` files with `glob.glob`. Each NWB file is opened with `pynwb.NWBHDF5IO` and read in full. All subjects and sessions are included; in `--sample` mode, 2 files are selected.

ii.
```python
data_dir = '/app/data'
subjects = sorted([s for s in os.listdir(data_dir) if s.startswith('sub-')])

all_nwb_files = []
for sub in subjects:
    files = sorted(glob.glob(os.path.join(data_dir, sub, '*.nwb')))
    all_nwb_files.extend(files)
```

```python
with NWBHDF5IO(nwb_file, 'r') as io:
    nwb = io.read()
```

iii. The AI documented finding 152 NWB files from 11 subjects, matching the expected count. Standard pynwb loading is used.

## 1-b. How are the data split into subjects?

i. Subjects are identified from subdirectory names (`sub-*`) and from `nwb.subject.subject_id` within each NWB file. Unique subjects are tracked in a list built during iteration.

ii.
```python
subjects = sorted([s for s in os.listdir(data_dir) if s.startswith('sub-')])
...
subj = result['subject']  # from nwb.subject.subject_id
if subj not in unique_subjects:
    unique_subjects.append(subj)
```

iii. The AI verified 11 subjects matching the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Session ID is extracted from `nwb.session_id`.

ii.
```python
session_id = nwb.session_id  # e.g., '03'
```

iii. The AI noted 152 sessions total across 11 subjects.

## 1-d. How are the data split into trials?

i. Trial boundaries are found using `trial_start` (where value == 1) and `teleport` (where value == 1) behavioral time series. For each trial start, the first teleport occurring after it is found.

ii.
```python
trial_start_inds = np.where(tstart == 1)[0]
teleport_inds = np.where(teleport == 1)[0]
...
for i in range(n_trials):
    later_teleports = teleport_inds[teleport_inds > trial_start_inds[i]]
    if len(later_teleports) > 0:
        trial_ends[i] = later_teleports[0]
    else:
        trial_ends[i] = len(position) - 1
```

iii. The AI documented that trial boundaries come from `trial_start` and `teleport` signals.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 frames are skipped. Sessions with fewer than 2 valid trials are also skipped.

ii.
```python
if n_frames < 2:
    continue
...
if len(neural_trials) < 2:
    print(f"    WARNING: Less than 2 valid trials, skipping session")
    return None
```

iii. The AI used a minimal threshold of 2 frames for trial validity.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw `Fluorescence` (F) and `Neuropil` (Fneu) time series stored in the NWB ophys processing module. The AI explicitly chose NOT to use the NWB `Deconvolved` field.

ii.
```python
F_plane = nwb.processing['ophys']['Fluorescence'][plane_name].data[:]
Fneu_plane = nwb.processing['ophys']['Neuropil'][plane_name].data[:]
```

iii. The AI documented that the NWB `Deconvolved` is suite2p's default deconvolution from raw F, not the paper's custom pipeline, so raw F and Fneu are used instead.

## 2-b. How is the `neural` data processed?

i. The AI computes dF/F but does NOT deconvolve. The pipeline: (1) neuropil subtraction F - 0.7*Fneu, (2) add back neuropil mean per trial, (3) maximin baseline (smooth sigma=15, min filter ~20s window, max filter ~20s window), (4) dFF = (F-baseline)/|baseline|, (5) smooth with 2-sample Gaussian per trial. Multi-plane data is pooled. NaNs are replaced with 0 for the decoder.

ii.
```python
def compute_dff(F, Fneu, trial_start_inds, teleport_inds,
                neu_coef=0.7, frame_rate=15.5078125):
    f_ = F - neu_coef * Fneu
    ...
    window_size = int(20 * frame_rate)  # 20 seconds in frames
    for i in range(n_trials):
        f_[:, start:stop] = f_[:, start:stop] + neu_coef * np.nanmean(
            Fneu[:, start:stop], axis=1, keepdims=True)
        flow[:, start:stop] = nansmooth(f_[:, start:stop], 15, axis=1)
        flow[:, start:stop] = ndimage.minimum_filter1d(flow[:, start:stop], window_size, axis=1)
        flow[:, start:stop] = ndimage.maximum_filter1d(flow[:, start:stop], window_size, axis=1)
    ...
    dff[valid_mask] = (f_[valid_mask] - flow[valid_mask]) / np.abs(flow[valid_mask])
    for i in range(n_trials):
        dff[:, start:stop] = nansmooth(dff[:, start:stop], 2, axis=1)
    return dff
```

iii. The AI documented following the reference code's dFF pipeline. However, the AI chose to use dFF (not deconvolved events) as the final neural representation, arguing it "preserves more information."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) `iscell` from suite2p manual curation, (2) interneuron exclusion based on dFF-speed Pearson correlation > 0.5.

ii.
```python
plane_iscell = iscell[plane_mask, 0] == 1
F_list.append(F_plane[:, plane_iscell].T.astype(np.float64))
...
for c in range(n_cells):
    dff_valid = dff[c, :n_common][valid_frames]
    if np.std(dff_valid) > 0 and np.std(speed_valid) > 0:
        corr = np.corrcoef(dff_valid, speed_valid)[0, 1]
        if not np.isnan(corr) and corr > 0.5:
            interneuron_mask[c] = False
```

iii. Both filters match the paper's methods section.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start. Since neural and behavioral data share the same frame indices, no additional alignment is needed beyond slicing from `trial_start_inds[i]` to `trial_ends[i]`.

ii.
```python
trial_neural = dff[:, start:end].astype(np.float32)
```

iii. The AI verified temporal alignment between neural and behavioral data.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Data is kept at the native frame rate. The time bin size is `1000.0 / 15.5078125` ms (~64.5 ms). No rebinning is applied.

ii.
```python
'time_bin_size': 1000.0 / 15.5078125,  # ~64.5 ms
```

iii. The AI used the scanner frame rate directly as the time bin size.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `timestamps` of the `position` behavioral time series.

ii.
```python
timestamps = bts.time_series['position'].timestamps[:].astype(np.float64)
...
time_from_start = (timestamps[start:end] - timestamps[start]).astype(np.float32)
```

iii. The AI used position timestamps, which are the same as other behavioral timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Subtract the first timestamp of the trial from all timestamps in the trial.

ii.
```python
time_from_start = (timestamps[start:end] - timestamps[start]).astype(np.float32)
```

iii. Straightforward subtraction.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same frame indices, so they are inherently aligned by using the same start/end indices.

ii.
```python
trial_neural = dff[:, start:end]
time_from_start = (timestamps[start:end] - timestamps[start])
```

iii. Verified that neural and behavioral data have the same frame count (with cropping for mismatches).

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the scene name parsed from `nwb.identifier`, NOT from the `environment` behavioral time series.

ii.
```python
identifier = nwb.identifier
scene = parse_scene_name(identifier)
...
def get_env_per_trial(scene, n_trials, change_trial=30):
    if '_to_' in scene and 'Env2' in scene.split('_to_')[1]:
        env_labels[:change_trial] = 0
        env_labels[change_trial:] = 1
    elif 'Env2' in scene:
        env_labels[:] = 1
    else:
        env_labels[:] = 0
```

iii. The AI parsed environment from the session scene name, using the switch logic from the reference code.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Scene name parsing to determine Env1 (0) or Env2 (1). For cross-environment switch sessions, environment changes at trial 30.

ii. See 4-a code snippet.

iii. The AI followed the reference code's switch trial convention (change_trial=30).

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the loop counter over trials (0-indexed sequential trial number within the session).

ii.
```python
trial_number = float(i)
```

iii. The AI used a sequential index rather than the NWB `trial number` variable.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing; the loop index `i` is used directly. The value is broadcast to all timepoints in the trial.

ii.
```python
input_data[2, :] = trial_number
```

iii. Simple assignment.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavioral time series timestamps.

ii.
```python
reward_timestamps = reward_ts_obj.timestamps[:]
...
rewarded = determine_reward_per_trial(reward_timestamps, trial_start_times, trial_end_times)
prev_reward = np.zeros(n_trials, dtype=int)
prev_reward[1:] = rewarded[:-1]
```

iii. The AI matches reward delivery timestamps to trial time windows.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, check if any reward timestamp falls within the trial's start-to-end time window. The previous trial's reward status is shifted by one trial. First trial gets 0.

ii.
```python
def determine_reward_per_trial(reward_timestamps, trial_start_times, trial_end_times):
    for i in range(n_trials):
        mask = (reward_timestamps >= t_start) & (reward_timestamps <= t_end)
        if np.any(mask):
            rewarded[i] = 1
    return rewarded
...
prev_reward[1:] = rewarded[:-1]
```

iii. The AI documented the logic in CONVERSION_NOTES.md.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavioral time series and the reward zone coordinates. Reward zone coordinates are determined from the scene name in the NWB identifier (not from the `reward_zone` behavioral variable).

ii.
```python
rz_labels, rz_coords = get_reward_zone_label_per_trial(scene, n_trials)
...
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start_cm, rz_end_cm)
```

iii. Reward zone locations A=[80,130], B=[200,250], C=[320,370] are from the reference code.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance: negative before zone, 0 inside zone, positive after zone.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    distance[before_zone] = position[before_zone] - rz_start
    distance[in_zone] = 0.0
    distance[after_zone] = position[after_zone] - rz_end
    return distance
```

iii. Standard signed distance computation.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit conditional assignments matching the instruction bins.

ii.
```python
def discretize_distance(distance):
    bins[distance < -50] = 0
    bins[(distance >= -50) & (distance < -10)] = 1
    bins[(distance >= -10) & (distance < 0)] = 2
    bins[distance == 0] = 3
    bins[(distance > 0) & (distance <= 10)] = 4
    bins[(distance > 10) & (distance <= 50)] = 5
    bins[distance > 50] = 6
    return bins
```

iii. Bin edges match the instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indices used for neural and behavioral data within each trial.

ii.
```python
trial_pos = position[start:end]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start_cm, rz_end_cm)
```

iii. Inherently aligned by shared indexing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series.

ii.
```python
position = bts.time_series['position'].data[:].astype(np.float64)
...
trial_pos = position[start:end]
```

iii. Direct extraction from behavioral data.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond slicing and discretization.

ii.
```python
pos_bins = discretize_position(trial_pos)
```

iii. Raw position values are used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins of 90 cm each (0-90, 90-180, 180-270, 270-360, 360+).

ii.
```python
def discretize_position(position):
    bins[position < 90] = 0
    bins[(position >= 90) & (position < 180)] = 1
    bins[(position >= 180) & (position < 270)] = 2
    bins[(position >= 270) & (position < 360)] = 3
    bins[position >= 360] = 4
    return bins
```

iii. Matches the instructions (5 equal-sized bins over 450 cm).

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices as neural data.

ii.
```python
trial_pos = position[start:end]
```

iii. Inherently aligned.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series.

ii.
```python
lick = bts.time_series['lick'].data[:].astype(np.float64)
...
trial_lick = lick[start:end]
```

iii. Direct extraction.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any value > 0 maps to 1, otherwise 0.

ii.
```python
lick_binary = (trial_lick > 0).astype(np.int64)
```

iii. Matches the binary lick specification.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices as neural data.

ii.
```python
trial_lick = lick[start:end]
```

iii. Inherently aligned.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the scene name parsed from `nwb.identifier`. The AI does NOT use the `reward_zone` behavioral variable in the NWB file.

ii.
```python
scene = parse_scene_name(identifier)
rz_labels, rz_coords = get_reward_zone_label_per_trial(scene, n_trials)
...
rz_loc = {'A': 0, 'B': 1, 'C': 2}.get(rz_label, 0)
```

iii. The AI parsed the scene name to determine which reward zone (A, B, or C) is active, with switches occurring at trial 30.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene name parsing to extract location labels. For switch sessions (`_to_` in name), the zone changes at trial 30. Labels are mapped to integers: A=0, B=1, C=2.

ii.
```python
def get_reward_zone_label_per_trial(scene, n_trials, change_trial=30):
    rz_info = get_reward_zone_info(scene)
    if len(rz_info) == 1:
        rz_labels[:] = label
    else:
        rz_labels[:ct] = label1
        rz_labels[ct:] = label2
```

iii. The switch trial is hardcoded to 30, matching the reference code's convention.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavioral time series timestamps.

ii.
```python
reward_timestamps = reward_ts_obj.timestamps[:]
rewarded = determine_reward_per_trial(reward_timestamps, trial_start_times, trial_end_times)
```

iii. Reward delivery events have separate timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward timestamp falls within the trial's time window (using behavior timestamps, not frame indices). Binary per-trial value broadcast to all timepoints.

ii.
```python
def determine_reward_per_trial(reward_timestamps, trial_start_times, trial_end_times):
    for i in range(n_trials):
        mask = (reward_timestamps >= t_start) & (reward_timestamps <= t_end)
        if np.any(mask):
            rewarded[i] = 1
    return rewarded
...
output_data[5, :] = reward_outcome
```

iii. Uses actual timestamps for matching rather than frame indices.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Neural/behavior length mismatch**: Data is cropped to the minimum of neural and behavioral frame counts.
- **Short trials**: Trials with < 2 frames are skipped.
- **NaN in neural data**: NaNs in dFF are replaced with 0 using `np.nan_to_num`.
- **Last trial without teleport**: Falls back to end of data.

ii.
```python
n_common = min(n_neural_frames, n_behav_frames)
...
if n_frames < 2:
    continue
...
trial_neural = np.nan_to_num(trial_neural, nan=0.0)
```

iii. The AI documented handling edge cases in CONVERSION_NOTES.md Step 10.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files (I/O bound)
2. Computing dFF (per-trial baseline computation)
3. Interneuron correlation filtering (per-cell loop)

ii. N/A

iii. The AI documented total conversion time of ~7 minutes for 152 sessions.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-cell interneuron correlation check loops over all cells. The per-trial reward determination also uses a Python loop. The discretization functions are already vectorized.

ii.
```python
for c in range(n_cells):
    dff_valid = dff[c, :n_common][valid_frames]
    corr = np.corrcoef(dff_valid, speed_valid)[0, 1]
```

iii. The AI noted that vectorization was applied where possible.

## 13-c. What processing does the code repeat multiple times?

i. Each NWB file is loaded only once. However, the dFF computation re-implements processing that could potentially reuse the NWB's stored Deconvolved data (though the AI chose not to use it for correctness reasons).

ii. N/A

iii. No obvious repeated processing since there's no survey step; the AI processes each file in a single pass.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes dFF but does not deconvolve it, whereas the reference solution computes both dFF and deconvolved events and uses the events. The AI's dFF includes NaN values that are then replaced with 0, which adds a processing step. The `result` dictionary stores extra fields (`env_per_trial`, `rz_labels`, `rewarded`) that are not part of the final output format.

ii.
```python
result = {
    ...
    'env_per_trial': env_per_trial,
    'rz_labels': rz_labels,
    'rewarded': rewarded,
}
```

iii. These extra fields are used for plotting and debugging but not saved in the final pickle.
