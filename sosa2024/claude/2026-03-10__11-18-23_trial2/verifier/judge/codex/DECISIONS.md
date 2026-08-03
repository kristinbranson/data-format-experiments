# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script scans `data/` for all `sub-*` directories, glob-matches every `*.nwb` file in each subject directory, and treats each NWB file as one session. It reads the NWB/HDF5 contents directly with `h5py` rather than `pynwb`, then later splits each session into trials inside `process_session()`.

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
...
with h5py.File(filepath, 'r') as f:
```

iii. In `CONVERSION_NOTES.md` Step 2, the agent documented the directory layout as `data/sub-{id}/sub-{id}_ses-{ses}_behavior+ophys.nwb` and counted 11 subjects and 152 sessions. Step 4 says direct use of the NWB contents was a resolved consistency decision.

## 1-b. How are the data split into subjects?

i. Subjects are defined by directory names beginning with `sub-`, and the stored subject id is also read from each file. Session-level subject labels are accumulated from `subject_id`.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
...
subject_id = f['general/subject/subject_id'][()].decode()
...
all_subject_ids.append(result['subject'])
unique_subjects = sorted(set(all_subject_ids))
subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
```

iii. Step 2 of the notes explicitly lists the subject directories and says the dataset contains 11 subjects. The trajectory summary also reports “11 subjects”.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The script builds one `sessions_info` entry per file and calls `process_session()` once for each file.

ii.
```python
files = sorted(glob(os.path.join(subj_dir, '*.nwb')))
for fpath in files:
    sessions.append({
        'subject': subj.replace('sub-', ''),
        'filepath': fpath,
        'filename': os.path.basename(fpath),
    })
...
for i, sess_info in enumerate(sessions_info):
    result = process_session(sess_info['filepath'], ...)
```

iii. The notes state “NWB files organized: `data/sub-{id}/sub-{id}_ses-{ses}_behavior+ophys.nwb`” and report 152 total sessions, so the AI justified session boundaries from file organization.

## 1-d. How are the data split into trials?

i. Trial starts are every frame where `trial_start > 0`; trial ends are every frame where `teleport > 0`. If the counts do not match, the script greedily matches each start to the first later teleport frame.

ii.
```python
trial_start_inds = np.where(trial_start > 0)[0]
teleport_inds = np.where(teleport_sig > 0)[0]
...
if len(teleport_inds) != n_trials:
    matched_teleports = []
    for ts in trial_start_inds:
        tp_after = teleport_inds[teleport_inds > ts]
        if len(tp_after) > 0:
            matched_teleports.append(tp_after[0])
    teleport_inds = np.array(matched_teleports)
```

iii. Step 1 of the notes says “Trial boundaries: `trial_start` and `teleport` signals in behavior timeseries”, and Step 4 lists trial-boundary handling as part of the consistency pass. There is no separate justification for using all positive `teleport` frames instead of teleport rising edges.

## 1-e. How are trials filtered based on quality controls?

i. The AI only keeps sessions with at least 2 trials and only drops individual trials shorter than 2 frames. It does not apply the reference script’s `min_ntimepoints=50` filter.

ii.
```python
if n_trials < 2:
    print(f"  WARNING: Only {n_trials} trials, skipping session")
    return None
...
for t in range(n_trials):
    ...
    n_tp = ei - si
    if n_tp < 2:
        continue
```

iii. The notes emphasize the target-format requirement that sessions must have at least two trials. No explicit note justifies the much looser per-trial length filter; the decision is only visible in the code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural matrices come from `processing/ophys/Deconvolved/<plane>/data`. Additional raw variables (`iscell`, `Fluorescence`, `Neuropil`) are used for filtering, but the kept activity values are the deconvolved traces.

ii.
```python
d = f[f'processing/ophys/Deconvolved/{plane}/data'][:]
fl = f[f'processing/ophys/Fluorescence/{plane}/data'][:]
ne = f[f'processing/ophys/Neuropil/{plane}/data'][:]
...
neural_all = deconv[:, final_cell_mask].T
```

iii. Step 1 of the notes says the NWB `Deconvolved` data are already the output of the reference dF/F plus deconvolution pipeline and “we should use these directly.”

## 2-b. How is the `neural` data processed?

i. The AI concatenates planes, filters neurons, transposes to `(n_neurons, n_timepoints)`, casts to `float32`, and replaces trial-level NaNs with zeros. It does not recompute deconvolution for the saved signal values.

ii.
```python
deconv = np.concatenate(deconv_list, axis=1)
...
neural_all = deconv[:, final_cell_mask].T
neural_all = neural_all.astype(np.float32)
...
trial_neural = neural_all[:, si:ei].copy()
trial_neural = np.nan_to_num(trial_neural, nan=0.0)
```

iii. The notes justify plane concatenation and direct use of NWB deconvolved events, especially for multi-plane sessions. The NaN-to-zero replacement is not separately justified in the notes.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies two neuron filters: Suite2p `iscell[:, 0] == 1`, then an additional interneuron exclusion based on correlation between a simple dF/F estimate and speed (`corr > 0.5`).

ii.
```python
cell_mask_concat = np.array([iscell[concat_to_iscell[i], 0] == 1 for i in range(n_rois_concat)])
...
f_corrected = fluorescence - 0.7 * neuropil_data
f_median = np.median(f_corrected, axis=0, keepdims=True)
dff_simple = (f_corrected - f_median) / np.abs(f_median)
...
if corrs[i] > 0.5:
    interneuron_mask[col_idx] = True
...
final_cell_mask = cell_mask_concat & ~interneuron_mask
```

iii. Step 3 of the notes claims the paper excludes putative interneurons with Pearson correlation of dF/F versus speed `> 0.5`, and Step 5 lists that filter as a “Key Decision”. Step 6-8 says the correlation was vectorized for efficiency.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start implicitly by slicing each trial from its `trial_start` frame to its teleport frame. Within each trial, time zero is the first imaging frame.

ii.
```python
for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    trial_neural = neural_all[:, si:ei].copy()
```

iii. Step 5 of the notes says “Temporal alignment: Align to trial start (first frame of each trial).”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses one imaging frame per bin and does not rebin. It stores a single nominal bin size of `1000 / 15.5078125` ms for all sessions.

ii.
```python
IMAGING_RATE_NOMINAL = 15.5078125  # Hz
...
frame_time = 1.0 / imaging_rate
...
time_bin_ms = 1000.0 / IMAGING_RATE_NOMINAL
...
'time_bin_size': time_bin_ms,
```

iii. Step 5 of the notes says “Time bin: One imaging frame (~64.5ms).” The notes treat the data as already aligned to imaging frames and say no rebinning is needed.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The saved time-from-start input is derived from the imaging rate and the number of frames in the trial, not from the NWB behavior timestamps.

ii.
```python
imaging_rate = f['general/optophysiology/ImagingPlane/imaging_rate'][()]
...
frame_time = 1.0 / imaging_rate  # seconds per frame
...
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
```

iii. Step 5 of the notes explicitly maps this variable as “Frame index * time_bin_size, in seconds”.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the AI constructs a synthetic evenly spaced time vector starting at 0 and incrementing by `1 / imaging_rate` seconds per frame.

ii.
```python
frame_time = 1.0 / imaging_rate  # seconds per frame
...
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
```

iii. The notes justify this as one time bin per imaging frame and describe the output as time-varying seconds from trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The vector is generated with exactly one sample per neural frame in the same `si:ei` trial slice, so alignment is by shared frame index within each trial.

ii.
```python
trial_neural = neural_all[:, si:ei].copy()
...
n_tp = ei - si
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
trial_input = np.vstack([
    trial_input_tv,
    np.full((1, n_tp), env_type, dtype=np.float32),
    ...
])
```

iii. Step 1 and Step 3 of the notes say the behavior and imaging timeseries are aligned to the imaging frame rate; the code operationalizes that by generating one time sample per kept frame.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The AI derives environment type from the NWB `identifier` string by parsing the scene name, not from the frame-aligned `environment` timeseries it also loaded.

ii.
```python
identifier = f['identifier'][()].decode()
scene = parse_scene(identifier)
...
rz_labels, rz_coords, env_per_trial = get_reward_zone_labels(scene, n_trials)
...
trial_env = env_per_trial.copy()
```

iii. Step 4 of the notes says scene info in `identifier` was used to determine switch structure, and Step 6-8 says scene parsing handled “all 26 unique scene name formats”.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The scene string is parsed into an environment code (`Env1 -> 0`, `Env2 -> 1`), and for switch sessions the environment changes after trial 30. The resulting per-trial value is repeated across all frames of the trial.

ii.
```python
if p.startswith('Env'):
    env_num = int(p.replace('Env', ''))
    env = env_num - 1
...
labels[:change_trial] = from_zone
labels[change_trial:] = to_zone
...
env_per_trial[:change_trial] = from_env
env_per_trial[change_trial:] = to_env
...
np.full((1, n_tp), env_type, dtype=np.float32)
```

iii. Step 4 lists cross-environment scene parsing as a resolved discrepancy, and Step 5 explicitly says to parse the scene and switch at trial 30.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is not read from the NWB `trial number` series. It is derived from the within-session trial loop index after the trial boundaries are computed from `trial_start` and `teleport`.

ii.
```python
for t in range(n_trials):
    ...
    trial_num = np.float32(t)
```

iii. Step 5 maps trial number to an “Integer per trial”, and the code makes that the sequential trial index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No additional processing is applied. The loop index `t` is repeated across every frame in the trial.

ii.
```python
trial_num = np.float32(t)
...
np.full((1, n_tp), trial_num, dtype=np.float32)
```

iii. The notes describe trial number as a per-trial contextual variable, so the AI expanded it to a constant per-frame row.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome comes from `Reward/timestamps`, which are converted into a frame-aligned binary `reward_frames` array using the behavior timestamps.

ii.
```python
behav_timestamps = f['processing/behavior/BehavioralTimeSeries/position/timestamps'][:]
reward_timestamps = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
...
reward_frames = np.zeros(n_timepoints, dtype=np.float32)
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
    reward_frames[frame_idx] = 1.0
```

iii. Step 4 and Step 6-8 of the notes repeatedly justify this as converting sparse reward timestamps into frame-aligned reward indicators.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. First the AI computes `trial_rewarded[t]` by checking whether any frame-aligned reward occurs inside each trial. Then it shifts that binary outcome by one trial: trial 0 gets 0, and every later trial gets the previous trial’s outcome. The per-trial result is repeated over the trial’s frames.

ii.
```python
trial_rewarded = np.zeros(n_trials, dtype=np.int64)
for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    if np.any(reward_frames[si:ei] > 0):
        trial_rewarded[t] = 1
...
prev_outcome = np.zeros(n_trials, dtype=np.int64)
for t in range(1, n_trials):
    prev_outcome[t] = trial_rewarded[t - 1]
```

iii. Step 5 maps this variable to “Binary: 0=omission, 1=rewarded” and explicitly calls it lagged.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The AI uses trial position from the `position` behavior series and reward-zone coordinates inferred from the scene identifier. It does not use the NWB `reward_zone` behavior series to infer the active zone.

ii.
```python
position = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
identifier = f['identifier'][()].decode()
scene = parse_scene(identifier)
...
rz_labels, rz_coords, env_per_trial = get_reward_zone_labels(scene, n_trials)
...
trial_pos = position[si:ei]
rz_start = rz_coords[t, 0]
rz_end = rz_coords[t, 1]
```

iii. Step 4 of the notes says the AI concluded the NWB `reward_zone` field was not directly usable for zone labels and therefore decided to parse scene names from `identifier` instead.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each frame, the AI computes signed distance from position to the current trial’s reward-zone interval: negative before the zone, zero inside it, positive after it. It then discretizes the result into 7 classes.

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
...
signed_dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
dist_bins = discretize_distance(signed_dist)
```

iii. Step 5 of the notes gives the same signed-distance interpretation and cites the reward-zone coordinates A/B/C from the reference behavior code.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI uses manual boolean thresholds to assign seven categories: `<-50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, `>50`.

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

iii. Step 5 lists the intended seven bins and says they follow the task specification. No separate note discusses the exact handling of edge cases like `distance == -10`.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data are sliced with the same trial start/end indices, so the binned distance vector has one value per neural frame.

ii.
```python
trial_neural = neural_all[:, si:ei].copy()
trial_pos = position[si:ei]
signed_dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The notes justify all behavioral signals as imaging-frame aligned; the code preserves that alignment by using the same `si:ei` slice.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is taken directly from the `position` behavior time series.

ii.
```python
position = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
...
trial_pos = position[si:ei]
```

iii. Step 2 of the notes lists `position` as one of the main frame-aligned behavioral variables in the NWB files.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI discretizes raw corridor position by taking `floor(position / 90)` and clipping to `[0, 4]`, producing five 90 cm bins over a 450 cm track.

ii.
```python
def discretize_position(position):
    bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
    return bins
...
pos_bins = discretize_position(trial_pos)
```

iii. Step 5 of the notes says “Discretize 0-450cm into 5 equal bins (90cm each)”, which is the explicit justification the AI recorded.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Thresholding is the same `floor(position / 90)` rule, clipped into categories 0 through 4: `0-90`, `90-180`, `180-270`, `270-360`, `360-450`.

ii.
```python
def discretize_position(position):
    bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
    return bins
...
'output_values': [
    ['0-90cm', '90-180cm', '180-270cm', '270-360cm', '360-450cm'],
```

iii. The notes justify this as “5 equal bins (90cm each)” based on the 450 cm track length.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position is sliced with the same trial frame indices as the neural data, so there is one position bin per neural frame.

ii.
```python
trial_neural = neural_all[:, si:ei].copy()
trial_pos = position[si:ei]
pos_bins = discretize_position(trial_pos)
```

iii. Step 3 of the notes says all behavior timeseries are aligned to imaging frames, and the code uses shared frame slices.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick output comes from the `lick` behavior time series.

ii.
```python
lick = f['processing/behavior/BehavioralTimeSeries/lick/data'][:]
...
trial_lick = lick_binary[si:ei]
```

iii. Step 2 of the notes lists `lick` among the main behavior fields. Step 3 says the NWB lick values are cumulative per frame and must be binarized.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI first binarizes lick counts by setting any positive value to 1. It then performs an additional stuck-sensor correction: if more than 30% of frames in a trial have lick counts above 2, the whole trial’s lick output is forced to 0 instead of NaN. The saved output is `(trial_lick > 0).astype(int)`.

ii.
```python
lick_binary = lick.copy()
lick_binary[lick_binary > 0] = 1
for t in range(n_trials):
    ...
    if len(trial_lick) > 0 and (np.sum(trial_lick > 2) / len(trial_lick)) > 0.30:
        lick_binary[si:ei] = 0
...
lick_out = (trial_lick > 0).astype(np.int64)
```

iii. Step 1 and Step 3 of the notes cite the reference `correct_lick_sensor_error()` rule and claim lick values are cumulative per frame. The code comment says the trial is set to 0 “for cleaner output”.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned by slicing the same trial frame range used for neural activity.

ii.
```python
trial_neural = neural_all[:, si:ei].copy()
trial_lick = lick_binary[si:ei]
lick_out = (trial_lick > 0).astype(np.int64)
```

iii. The notes repeatedly state that behavioral timeseries are already aligned to imaging frames, and the code keeps that one-sample-per-frame structure.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from the scene name embedded in the NWB `identifier`, not from the `reward_zone` behavior series.

ii.
```python
identifier = f['identifier'][()].decode()
scene = parse_scene(identifier)
rz_labels, rz_coords, env_per_trial = get_reward_zone_labels(scene, n_trials)
...
zone_map = {'A': 0, 'B': 1, 'C': 2}
rz_loc = zone_map.get(rz_labels[t], 0)
```

iii. Step 4 of the notes says the AI resolved the reward-zone labeling problem by parsing scene info from `identifier`, and Step 5 repeats that as a key design decision.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI parses reward-zone letters and environment labels from the scene string, assumes switch sessions change at trial 30, assigns per-trial zone labels `A/B/C`, and maps them to `0/1/2`. The per-trial value is then repeated over the trial’s frames.

ii.
```python
def get_reward_zone_labels(scene, n_trials, change_trial=30):
    ...
    if '_to_' in scene:
        ...
        labels[:change_trial] = from_zone
        labels[change_trial:] = to_zone
    else:
        ...
        labels[:] = zone
...
rz_loc = zone_map.get(rz_labels[t], 0)
np.full((1, n_tp), rz_loc, dtype=np.int64)
```

iii. Step 3 of the notes says “Switch trial: 30”, and Step 6-8 says scene parsing handled all observed naming patterns.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from `Reward/timestamps`, converted to a frame-aligned reward indicator via the behavior timestamps.

ii.
```python
behav_timestamps = f['processing/behavior/BehavioralTimeSeries/position/timestamps'][:]
reward_timestamps = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
...
reward_frames = np.zeros(n_timepoints, dtype=np.float32)
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
    reward_frames[frame_idx] = 1.0
```

iii. Step 4 and Step 6-8 of the notes justify this as converting sparse reward events into frame-aligned trial outcomes.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Each reward timestamp is assigned to the nearest behavior frame. A trial is labeled rewarded if any such reward frame falls between its start and end indices. The binary outcome is repeated across the trial’s frames.

ii.
```python
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
    reward_frames[frame_idx] = 1.0
...
for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    if np.any(reward_frames[si:ei] > 0):
        trial_rewarded[t] = 1
...
rew_outcome = int(trial_rewarded[t])
```

iii. Step 5 maps reward outcome to a per-trial binary variable and Step 9 of the notes cites the resulting reward rate as matching the paper’s omission rate.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several issues in code: neural/behavior length mismatches are truncated to the shorter array; mismatched `trial_start` and `teleport` counts are greedily reconciled; lick-sensor error trials are zeroed; neural NaNs are replaced with 0; sessions with fewer than 2 trials are skipped; and sub-2-frame trials are dropped.

ii.
```python
if n_timepoints != n_behav:
    min_len = min(n_timepoints, n_behav)
    deconv = deconv[:min_len]
    ...
if len(teleport_inds) != n_trials:
    matched_teleports = []
    ...
    teleport_inds = np.array(matched_teleports)
...
if len(trial_lick) > 0 and (np.sum(trial_lick > 2) / len(trial_lick)) > 0.30:
    lick_binary[si:ei] = 0
...
trial_neural = np.nan_to_num(trial_neural, nan=0.0)
```

iii. Step 4 of the notes explicitly justifies truncating off-by-one behavior/neural mismatches in multi-plane data. Step 1 and Step 3 justify the lick-sensor rule. The other defensive behaviors are only implicit in the code.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive parts of the AI code are reading large NWB arrays, concatenating multi-plane fluorescence/neuropil/deconvolved matrices, computing the speed-dF/F correlation filter, looping over trials to build outputs, and finally pickling the full dataset.

ii.
```python
with h5py.File(filepath, 'r') as f:
    ...
    d = f[f'processing/ophys/Deconvolved/{plane}/data'][:]
    fl = f[f'processing/ophys/Fluorescence/{plane}/data'][:]
    ne = f[f'processing/ophys/Neuropil/{plane}/data'][:]
...
dff_accepted = dff_simple[valid_mask][:, accepted_cols]
...
for t in range(n_trials):
    ...
with open(args.output, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. Step 6-8 highlights vectorized interneuron exclusion as a key implementation detail, and Step 9 records a full-conversion runtime of 762.5 seconds for all 152 sessions.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized the dF/F-speed correlation, but several remaining loops could still be vectorized or reduced: per-reward nearest-frame matching, per-trial lick correction, per-trial reward-outcome computation, the per-trial construction loop, and the small per-cell loop that writes `interneuron_mask`.

ii.
```python
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
    reward_frames[frame_idx] = 1.0
...
for t in range(n_trials):
    ...
    if np.any(reward_frames[si:ei] > 0):
        trial_rewarded[t] = 1
...
for i, col_idx in enumerate(accepted_cols):
    if corrs[i] > 0.5:
        interneuron_mask[col_idx] = True
```

iii. The notes explicitly say the speed-dF/F correlation was vectorized, which implies the AI was thinking about performance. No separate note enumerates the remaining vectorization opportunities.

## 13-c. What processing does the code repeat multiple times?

i. Within each trial the code repeatedly allocates constant per-frame rows with `np.full`, recomputes reward-related summaries after already creating `reward_frames`, and repeats the same scene-derived trial metadata expansion for every trial. Across sessions it also recomputes the dF/F-style filtering pipeline from fluorescence and neuropil even though the saved neural output comes from deconvolved traces.

ii.
```python
trial_input = np.vstack([
    trial_input_tv,
    np.full((1, n_tp), env_type, dtype=np.float32),
    np.full((1, n_tp), trial_num, dtype=np.float32),
    np.full((1, n_tp), prev_out, dtype=np.float32),
])
...
trial_output = np.vstack([
    ...
    np.full((1, n_tp), rz_loc, dtype=np.int64),
    np.full((1, n_tp), rew_outcome, dtype=np.int64),
])
...
f_corrected = fluorescence - 0.7 * neuropil_data
```

iii. The notes justify the extra fluorescence/neuropil pass by the claimed interneuron rule and justify expanding per-trial variables so everything fits a `(n_var, n_timepoints)` layout.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several intermediate computations are not part of the saved decoder data: fluorescence and neuropil loading plus simple dF/F computation are only used for the extra interneuron filter; the raw `environment` series is loaded but then ignored; `trial_input_pt` is created and never used; and `rz_coords` are computed even though only the discretized outputs are saved.

ii.
```python
flu_list = []
neu_list = []
...
fluorescence = np.concatenate(flu_list, axis=1)
neuropil_data = np.concatenate(neu_list, axis=1)
...
environment = f['processing/behavior/BehavioralTimeSeries/environment/data'][:]
...
trial_input_pt = np.array([env_type, trial_num, prev_out], dtype=np.float32)
```

iii. Step 4 and Step 5 of the notes justify these extras as part of the AI’s chosen consistency plan, especially the interneuron filter and scene parsing. The unused `trial_input_pt` has no recorded justification and appears to be leftover code.
