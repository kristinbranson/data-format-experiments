# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans `data/` for subject directories whose names start with `sub-`, then scans each subject directory for all `.nwb` files. Each file is treated as one session and is opened directly with `h5py`, with neural and behavioral arrays read from NWB paths inside the file.

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

with h5py.File(filepath, 'r') as f:
    subject_id = f['general/subject/subject_id'][()].decode()
    deconv_group = f['processing/ophys/Deconvolved']
    position = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
```

iii. `CONVERSION_NOTES.md` says the dataset has 11 subjects and 152 sessions and describes the NWB layout under `processing/ophys` and `processing/behavior`. The notes justify direct NWB reads as sufficient because the needed arrays are already present in those paths.

## 1-b. How are the data split into subjects?

i. Subjects are split by subdirectory name and by the `subject_id` stored in each NWB file. The output `subjects` list is the sorted set of session-level subject IDs seen during conversion.

ii. 
```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
...
subject_id = f['general/subject/subject_id'][()].decode()
...
unique_subjects = sorted(set(all_subject_ids))
subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
```

iii. The notes say the directory structure is `data/sub-{id}/sub-{id}_ses-{ses}_behavior+ophys.nwb` and that there are 11 mouse directories, so the AI treated subject folders as the canonical mouse split.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file is treated as a separate session. The converter does not merge sessions across days or across files.

ii. 
```python
files = sorted(glob(os.path.join(subj_dir, '*.nwb')))
for fpath in files:
    sessions.append({
        'subject': subj.replace('sub-', ''),
        'filepath': fpath,
        'filename': os.path.basename(fpath),
    })
```

iii. `CONVERSION_NOTES.md` reports 152 sessions total and repeatedly refers to processing one NWB file at a time, so the AI's justification was that the file organization already defines the session split.

## 1-d. How are the data split into trials?

i. Trials start wherever `trial_start > 0`. Trial ends are taken from indices where `teleport > 0`; if the counts do not match, the AI matches each trial start to the first later teleport index. Trial data are then sliced with `si:ei`.

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
...
si = trial_start_inds[t]
ei = teleport_inds[t]
trial_neural = neural_all[:, si:ei].copy()
```

iii. The notes state that "trial boundaries: `trial_start` and `teleport` signals in behavior timeseries" and that these were used to mark trial start/end. No stronger justification than that appears in the notes.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps sessions only if they have at least two detected trials, and it drops individual trials only when they contain fewer than 2 timepoints. It does not implement the reference solution's `< 50` frame filter.

ii. 
```python
if n_trials < 2:
    print(f"  WARNING: Only {n_trials} trials, skipping session")
    return None
...
for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    n_tp = ei - si

    if n_tp < 2:
        continue
```

iii. The notes emphasize only the decoder requirement that each session contain at least two trials. I did not find an explicit justification for the `n_tp < 2` threshold beyond avoiding empty or degenerate trial slices.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final saved neural data come from the NWB `Deconvolved` arrays. The AI also loads `Fluorescence` and `Neuropil`, but only to build a simplified dF/F estimate for interneuron filtering.

ii. 
```python
deconv_group = f['processing/ophys/Deconvolved']
...
d = f[f'processing/ophys/Deconvolved/{plane}/data'][:]
fl = f[f'processing/ophys/Fluorescence/{plane}/data'][:]
ne = f[f'processing/ophys/Neuropil/{plane}/data'][:]
...
neural_all = deconv[:, final_cell_mask].T
```

iii. The notes explicitly say: "NWB `Deconvolved` data is the deconvolved events AFTER full dF/F processing pipeline ... We should use these directly." That is the AI's stated justification for not recomputing the paper's events.

## 2-b. How is the `neural` data processed?

i. The AI does not rerun the paper's maximin dF/F and OASIS deconvolution pipeline. It concatenates the stored `Deconvolved` planes, filters cells, casts to `float32`, and replaces any NaNs with zeros after trial slicing.

ii. 
```python
deconv = np.concatenate(deconv_list, axis=1)
...
final_cell_mask = cell_mask_concat & ~interneuron_mask
neural_all = deconv[:, final_cell_mask].T
neural_all = neural_all.astype(np.float32)
...
trial_neural = neural_all[:, si:ei].copy()
trial_neural = np.nan_to_num(trial_neural, nan=0.0)
```

iii. The notes justify this by claiming the stored deconvolved NWB data are already the fully processed events used by the paper, so recomputing dF/F and deconvolution was considered unnecessary.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies two filters: `iscell` ROI curation and an interneuron exclusion step. For the latter it computes a simplified per-cell dF/F from `Fluorescence - 0.7 * Neuropil`, correlates that with speed on samples where `speed > 0`, `position >= 0`, and speed is not NaN, and removes cells with correlation `> 0.5`.

ii. 
```python
cell_mask_concat = np.array([iscell[concat_to_iscell[i], 0] == 1 for i in range(n_rois_concat)])

f_corrected = fluorescence - 0.7 * neuropil_data
f_median = np.median(f_corrected, axis=0, keepdims=True)
f_median[f_median == 0] = 1
dff_simple = (f_corrected - f_median) / np.abs(f_median)

valid_mask = (speed > 0) & (position >= 0) & ~np.isnan(speed)
...
if corrs[i] > 0.5:
    interneuron_mask[col_idx] = True

final_cell_mask = cell_mask_concat & ~interneuron_mask
```

iii. `CONVERSION_NOTES.md` cites the Methods for "Suite2p iscell + interneuron exclusion (speed-dFF corr > 0.5)" and presents that as the rationale. The notes do not justify the simplified median-baseline dF/F used to compute the correlation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start implicitly by defining each trial slice from `trial_start` to the matched teleport index. No extra temporal shift is applied.

ii. 
```python
si = trial_start_inds[t]
ei = teleport_inds[t]
trial_neural = neural_all[:, si:ei].copy()
```

iii. The notes say the dataset is aligned to "start of trial (entry to linear track)" and that `trial_start`/`teleport` define trial boundaries, so the AI treated trial slicing itself as the required alignment step.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native imaging-frame resolution and does not rebin. For within-trial time it uses `1 / imaging_rate` seconds per frame, and it stores metadata `time_bin_size` as `1000 / 15.5078125` ms.

ii. 
```python
IMAGING_RATE_NOMINAL = 15.5078125  # Hz
...
imaging_rate = f['general/optophysiology/ImagingPlane/imaging_rate'][()]
...
frame_time = 1.0 / imaging_rate  # seconds per frame
...
time_bin_ms = 1000.0 / IMAGING_RATE_NOMINAL
```

iii. The notes say behavior is aligned to imaging at "~15.5 Hz (~64.5 ms/frame)" and treat the stored NWB sampling as already decoder-ready, so no temporal rebinning was considered necessary.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from trial length and imaging rate, not from stored behavior timestamps. The AI uses the number of frames in the trial and `imaging_rate` from NWB metadata.

ii. 
```python
imaging_rate = f['general/optophysiology/ImagingPlane/imaging_rate'][()]
...
n_tp = ei - si
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
```

iii. The notes justify this by saying all behavioral time series are aligned to imaging frames at ~15.5 Hz, so frame count times frame duration was treated as sufficient.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the AI constructs `0, frame_time, 2*frame_time, ...` up to the trial length. It does not subtract actual timestamps.

ii. 
```python
frame_time = 1.0 / imaging_rate  # seconds per frame
...
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
```

iii. The notes state the frame rate is effectively constant and that behavior is already aligned to imaging, which is the apparent reason the AI used synthetic evenly spaced time values.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by construction: the time vector has one sample per neural frame in the same `si:ei` trial slice.

ii. 
```python
si = trial_start_inds[t]
ei = teleport_inds[t]
n_tp = ei - si
trial_neural = neural_all[:, si:ei].copy()
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
trial_input = np.vstack([
    time_from_start.reshape(1, -1),
    np.full((1, n_tp), env_type, dtype=np.float32),
    ...
])
```

iii. The notes say neural and behavior are aligned to the same imaging frames, so the AI's justification was that matching array length within each trial was enough to guarantee alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The saved environment input is derived from the session `identifier` string via scene parsing, not from the NWB `environment` behavior time series, even though that array is loaded.

ii. 
```python
identifier = f['identifier'][()].decode()
scene = parse_scene(identifier)
...
def get_reward_zone_labels(scene, n_trials, change_trial=30):
    ...
    return labels, coords, env_per_trial
...
trial_env = env_per_trial.copy()
```

iii. The notes say "Scene parsing: handles all 26 unique scene name formats" and that `env_per_trial` from scene parsing is "more reliable for cross-env switches," which is the explicit rationale preserved in comments and notes.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI parses strings like `Env1_LocationB_to_A` or `Env1_A_to_Env2_B`, extracts environment codes from the scene name, and assumes a trial-30 switch for `_to_` sessions. The resulting per-trial environment value is then repeated across all timepoints in the trial.

ii. 
```python
if '_to_' in scene:
    parts = scene.split('_')
    to_idx = parts.index('to')
    from_zone, from_env = _parse_zone_and_env(parts[:to_idx])
    to_zone, to_env = _parse_zone_and_env(parts[to_idx+1:])
    ...
    env_per_trial[:change_trial] = from_env
    env_per_trial[change_trial:] = to_env
...
trial_input = np.vstack([
    ...,
    np.full((1, n_tp), env_type, dtype=np.float32),
    ...
])
```

iii. The notes justify this as a way to handle all scene naming patterns, especially switches, without depending on noisy or ambiguous framewise signals.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the loop index over detected trials, i.e. from the ordering induced by `trial_start_inds` and `teleport_inds`.

ii. 
```python
for t in range(n_trials):
    ...
    trial_num = np.float32(t)
```

iii. The notes map "Trial number" to "Integer per trial" and indicate this is simply the within-session trial index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transformation beyond assigning the integer trial index and repeating it across the trial's timepoints.

ii. 
```python
trial_num = np.float32(t)
...
np.full((1, n_tp), trial_num, dtype=np.float32)
```

iii. I did not find a more detailed justification than the obvious interpretation that trial number should be sequential within session.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the sparse `Reward/timestamps` time series. The AI converts reward timestamps to frame indices, builds a session-wide binary reward vector, then converts that to per-trial reward outcomes and finally lags it by one trial.

ii. 
```python
reward_timestamps = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
...
reward_frames = np.zeros(n_timepoints, dtype=np.float32)
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
    reward_frames[frame_idx] = 1.0
...
trial_rewarded = np.zeros(n_trials, dtype=np.int64)
for t in range(n_trials):
    if np.any(reward_frames[si:ei] > 0):
        trial_rewarded[t] = 1
```

iii. The notes say "Reward field: convert sparse timestamps to frame-aligned binary per trial" and map previous trial outcome to a lagged binary derived from reward.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. After computing `trial_rewarded`, the AI creates `prev_outcome` by shifting reward outcomes by one trial. Trial 0 is forced to 0. The per-trial scalar is then repeated across the trial.

ii. 
```python
prev_outcome = np.zeros(n_trials, dtype=np.int64)
for t in range(1, n_trials):
    prev_outcome[t] = trial_rewarded[t - 1]
...
np.full((1, n_tp), prev_out, dtype=np.float32)
```

iii. The notes explicitly describe this variable as "Per-trial (lagged)" and say the first trial has no previous trial, so it is encoded as omission / 0.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Distance to reward zone is derived from the framewise `position` signal plus reward-zone coordinates inferred from the session `identifier` scene string. The AI does not use the raw `reward_zone` behavior signal for this output.

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
signed_dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The notes justify scene-based reward-zone inference by saying the scene parser handles all session formats and that reward zones A/B/C have fixed coordinates taken from the reference code.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each frame, the AI computes signed distance to the current trial's reward-zone interval: negative before the zone, positive after the zone, and 0 inside the zone.

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

iii. The notes say this variable is "signed distance" to the 50 cm reward zone and treat that as the natural decoder target corresponding to the task instructions.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI hard-codes the seven requested categories with explicit comparisons rather than `np.digitize`.

ii. 
```python
bins = np.zeros(len(distance), dtype=np.int64)
bins[distance < -50] = 0
bins[(distance >= -50) & (distance < -10)] = 1
bins[(distance >= -10) & (distance < 0)] = 2
bins[distance == 0] = 3
bins[(distance > 0) & (distance <= 10)] = 4
bins[(distance > 10) & (distance <= 50)] = 5
bins[distance > 50] = 6
```

iii. The notes map this output directly to the bins specified in the instructions and report that the resulting classes were "well-distributed."

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural activity are sliced with the same `si:ei` trial indices, so distance-to-zone is aligned frame-by-frame with neural activity.

ii. 
```python
trial_neural = neural_all[:, si:ei].copy()
trial_pos = position[si:ei]
signed_dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The notes repeatedly describe all behavioral series as aligned to imaging frames, which is the AI's justification for using shared slice indices as alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavior `position` time series.

ii. 
```python
position = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
...
trial_pos = position[si:ei]
```

iii. The notes describe `position` as the animal's position on the 450 cm track and use it directly for the absolute-position output.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI takes the raw per-trial position samples and bins them by `floor(position / 90)` with clipping to `[0, 4]`.

ii. 
```python
def discretize_position(position):
    bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
    return bins
...
pos_bins = discretize_position(trial_pos)
```

iii. The notes justify this as five equal 90 cm bins spanning the 450 cm corridor, exactly as requested by the decoder specification.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is thresholded into five equal-width 90 cm bins covering the track and clipped into category labels 0 through 4.

ii. 
```python
bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
```

iii. The notes explicitly say "5 equal bins (90 cm each)" and report a roughly even output distribution, which the AI treated as a sanity check.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned by slicing `position[si:ei]` using the same trial boundaries used for neural data.

ii. 
```python
trial_neural = neural_all[:, si:ei].copy()
trial_pos = position[si:ei]
```

iii. The AI's justification is the same framewise alignment assumption used throughout the behavior variables: both streams are indexed on the same imaging frames.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavior `lick` time series.

ii. 
```python
lick = f['processing/behavior/BehavioralTimeSeries/lick/data'][:]
...
trial_lick = lick_binary[si:ei]
```

iii. The notes identify `lick` as a behavior stream already aligned to imaging and describe it as cumulative per frame before binarization.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI binarizes lick values with `> 0 -> 1`. Before trial extraction it also applies a lick-sensor QC rule: if more than 30% of a trial's frames have `lick > 2`, it zeros the entire trial instead of leaving NaNs.

ii. 
```python
lick_binary = lick.copy()
lick_binary[lick_binary > 0] = 1
...
if len(trial_lick) > 0 and (np.sum(trial_lick > 2) / len(trial_lick)) > 0.30:
    lick_binary[si:ei] = 0
...
lick_out = (trial_lick > 0).astype(np.int64)
```

iii. The notes cite the reference paper's stuck-sensor rule (>30% of frames with count > 2), but the code comment says zeros were used "for cleaner output," which is the AI's practical justification for deviating from NaNs.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned by slicing the per-frame lick vector over the same `si:ei` trial interval used for neural data.

ii. 
```python
trial_neural = neural_all[:, si:ei].copy()
trial_lick = lick_binary[si:ei]
lick_out = (trial_lick > 0).astype(np.int64)
```

iii. The notes say all behavior time series are sampled at the imaging frame rate, so no extra interpolation or resampling was used.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from the session `identifier` scene string, parsed into zone labels A/B/C and optionally switched after trial 30. It is not derived from the NWB `reward_zone` behavior variable.

ii. 
```python
identifier = f['identifier'][()].decode()
scene = parse_scene(identifier)
...
rz_labels, rz_coords, env_per_trial = get_reward_zone_labels(scene, n_trials)
...
zone_map = {'A': 0, 'B': 1, 'C': 2}
rz_loc = zone_map.get(rz_labels[t], 0)
```

iii. The notes say "Scene parsing: handles all 26 unique scene name formats" and present that parser as the mechanism for assigning reward-zone location per trial.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI parses environment/zone tokens from the scene string, assumes a switch at `change_trial=30` for `_to_` sessions, maps zone letters to fixed coordinate pairs, and then maps `A/B/C` to class labels `0/1/2`.

ii. 
```python
def get_reward_zone_labels(scene, n_trials, change_trial=30):
    ...
    labels[:change_trial] = from_zone
    labels[change_trial:] = to_zone
    coords[:change_trial] = REWARD_ZONE_DICT[from_zone]
    coords[change_trial:] = REWARD_ZONE_DICT[to_zone]
...
zone_map = {'A': 0, 'B': 1, 'C': 2}
rz_loc = zone_map.get(rz_labels[t], 0)
```

iii. The notes justify the fixed switch rule by referring to switch days and by claiming the scene names fully encode trial context.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the sparse `Reward/timestamps` time series, converted to frame-aligned rewards and then summarized per trial.

ii. 
```python
reward_timestamps = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
...
reward_frames = np.zeros(n_timepoints, dtype=np.float32)
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
    reward_frames[frame_idx] = 1.0
...
if np.any(reward_frames[si:ei] > 0):
    trial_rewarded[t] = 1
```

iii. The notes explicitly call the NWB reward field sparse and say it must be converted to a frame-aligned binary signal before trial-level labels can be assigned.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are mapped to the nearest behavioral frame with `argmin(abs(timestamp difference))`. A trial is labeled rewarded if any mapped reward frame falls between its start and end indices. The per-trial label is then repeated across all frames in that trial.

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
np.full((1, n_tp), rew_outcome, dtype=np.int64)
```

iii. The notes say this conversion was needed because `Reward` is not frame-aligned, and they use the observed omission rate (~15.7%) as a sanity check on the resulting labels.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several issues defensively: it crops neural and behavior streams to the shorter length on mismatch; heuristically rematches teleports to trial starts when the counts differ; skips sessions with fewer than two trials and trials with fewer than two samples; zero-fills NaNs in neural trial matrices; and zeroes lick-corrupted trials instead of keeping NaNs.

ii. 
```python
if n_timepoints != n_behav:
    min_len = min(n_timepoints, n_behav)
    deconv = deconv[:min_len]
    ...

if len(teleport_inds) != n_trials:
    matched_teleports = []
    for ts in trial_start_inds:
        tp_after = teleport_inds[teleport_inds > ts]
        if len(tp_after) > 0:
            matched_teleports.append(tp_after[0])

if n_tp < 2:
    continue

trial_neural = np.nan_to_num(trial_neural, nan=0.0)

if len(trial_lick) > 0 and (np.sum(trial_lick > 2) / len(trial_lick)) > 0.30:
    lick_binary[si:ei] = 0
```

iii. The notes justify cropping as a fix for off-by-one mismatches in some multi-plane sessions and describe the lick handling as a cleaner alternative for decoder output. I did not find a justification for zero-filling neural NaNs beyond convenience.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive steps are session I/O and large-array reads from NWB, concatenating multi-plane neural arrays, computing the session-wide cell filter statistics, iterating over every trial to build output matrices, and writing the very large pickle at the end.

ii. 
```python
with h5py.File(filepath, 'r') as f:
    ...
    d = f[f'processing/ophys/Deconvolved/{plane}/data'][:]
    fl = f[f'processing/ophys/Fluorescence/{plane}/data'][:]
    ne = f[f'processing/ophys/Neuropil/{plane}/data'][:]
...
for t in range(n_trials):
    ...
with open(args.output, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. `CONVERSION_NOTES.md` reports a 9.4 GB output and 762.5 s total runtime, which implies that file loading and serialization dominate wall time. The notes do not explicitly rank hotspots beyond that.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several remaining loops are straightforward vectorization candidates: the list-comprehension-style `iscell` mapping, the teleport rematching loop, the loop over reward timestamps that calls `argmin` each time, the trial-reward loop, the lick QC loop, and the per-trial packaging loop.

ii. 
```python
cell_mask_concat = np.array([iscell[concat_to_iscell[i], 0] == 1 for i in range(n_rois_concat)])
...
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
    reward_frames[frame_idx] = 1.0
...
for t in range(n_trials):
    si = trial_start_inds[t]
    ei = teleport_inds[t]
    ...
```

iii. The notes explicitly mention one vectorization choice that was made: the interneuron correlation was implemented as a vectorized z-scored dot product. I did not find explicit justification for leaving the remaining loops unvectorized.

## 13-c. What processing does the code repeat multiple times?

i. Within each session, the code reads `Deconvolved`, `Fluorescence`, and `Neuropil` even though the final saved neural signal comes only from `Deconvolved`; it also walks through all trials multiple times for reward labeling, lick QC, and final packaging.

ii. 
```python
d = f[f'processing/ophys/Deconvolved/{plane}/data'][:]
fl = f[f'processing/ophys/Fluorescence/{plane}/data'][:]
ne = f[f'processing/ophys/Neuropil/{plane}/data'][:]
...
for t in range(n_trials):
    ...
for t in range(n_trials):
    ...
for t in range(n_trials):
    ...
```

iii. The notes justify loading `Fluorescence` and `Neuropil` because they are needed for interneuron exclusion, but there is no separate justification for the repeated trial passes.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main unnecessary work is loading `Fluorescence` and `Neuropil` and computing `dff_simple` even though the saved neural output is the stored `Deconvolved` signal; loading the `environment` series but then not using it to form the environment input; and constructing `trial_input_pt` but never using it.

ii. 
```python
fluorescence = np.concatenate(flu_list, axis=1)
neuropil_data = np.concatenate(neu_list, axis=1)
...
dff_simple = (f_corrected - f_median) / np.abs(f_median)
...
environment = f['processing/behavior/BehavioralTimeSeries/environment/data'][:]
...
trial_input_pt = np.array([env_type, trial_num, prev_out], dtype=np.float32)
```

iii. The notes do not call these steps unnecessary; they appear to be incidental consequences of the chosen implementation rather than explicitly justified design choices.
