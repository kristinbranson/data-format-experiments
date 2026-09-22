# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all data by listing `/app/data/sub-*` subject directories, globbing all `.nwb` files inside them, and opening each file with `pynwb.NWBHDF5IO`. Within each file it reads both `ophys` and `behavior` processing modules directly from the NWB object.

ii. 
```python
data_dir = '/app/data'
subjects = sorted([s for s in os.listdir(data_dir) if s.startswith('sub-')])

all_nwb_files = []
for sub in subjects:
    files = sorted(glob.glob(os.path.join(data_dir, sub, '*.nwb')))
    all_nwb_files.extend(files)

with NWBHDF5IO(nwb_file, 'r') as io:
    nwb = io.read()
```

iii. The justification in `CONVERSION_NOTES.md` is that the dataset consists of 11 subject directories and 152 NWB files, with one behavior+ophys NWB per session. The trajectory also shows the agent explicitly exploring the NWB structure and then choosing direct NWB reads as the loading method.

## 1-b. How are the data split into subjects?

i. Subjects are split by `sub-*` directories and then tracked per session using `nwb.subject.subject_id`. The final `subjects` list is the order in which unique subject IDs are first encountered while iterating through the sorted NWB files.

ii. 
```python
subjects = sorted([s for s in os.listdir(data_dir) if s.startswith('sub-')])
...
subject_id = nwb.subject.subject_id
...
if subj not in unique_subjects:
    unique_subjects.append(subj)
subj_idx = unique_subjects.index(subj)
all_subject_idx.append(subj_idx)
```

iii. The notes say the subject directories are `sub-m3`, `sub-m4`, ..., `sub-m19`, matching the mice described in the paper, so the agent treated directory structure and NWB subject metadata as the authoritative subject split.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session metadata come from `nwb.session_id`, and the top-level session ordering is just the sorted order of the NWB filenames.

ii. 
```python
for sub in subjects:
    files = sorted(glob.glob(os.path.join(data_dir, sub, '*.nwb')))
    all_nwb_files.extend(files)
...
session_id = nwb.session_id
```

iii. The justification in the notes is that the dataset contains 152 NWB files and that each file has one session’s behavior and ophys data, so file granularity was taken as session granularity.

## 1-d. How are the data split into trials?

i. Trials are split from the `trial_start` and `teleport` behavioral series. The agent uses all indices where `trial_start == 1` as trial starts, then for each start takes the first later sample where `teleport == 1` as the end of that trial. If no later teleport exists, it uses the end of the recording.

ii. 
```python
tstart = bts.time_series['trial_start'].data[:].astype(np.float64)
teleport = bts.time_series['teleport'].data[:].astype(np.float64)
...
trial_start_inds = np.where(tstart == 1)[0]
teleport_inds = np.where(teleport == 1)[0]
...
trial_ends = np.zeros(n_trials, dtype=int)
for i in range(n_trials):
    later_teleports = teleport_inds[teleport_inds > trial_start_inds[i]]
    if len(later_teleports) > 0:
        trial_ends[i] = later_teleports[0]
    else:
        trial_ends[i] = len(position) - 1
```

iii. The notes explicitly state the key decision “trial boundaries: trial_start to teleport (excluding teleport/ITI period)”. The trajectory shows the agent investigating trial structure before committing to this rule.

## 1-e. How are trials filtered based on quality controls?

i. The agent does almost no trial-level QC filtering. It only drops trials with fewer than 2 frames after slicing, and later skips a session if fewer than 2 valid trials remain.

ii. 
```python
n_frames = end - start

if n_frames < 2:
    continue
...
if len(neural_trials) < 2:
    print(f"    WARNING: Less than 2 valid trials, skipping session")
    return None
```

iii. There is no detailed trial-QC rationale in the code comments. The notes frame this mainly as satisfying the decoder’s requirement that each session have at least two trials, rather than as a paper-matched quality-control decision.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` arrays are derived from raw `Fluorescence` (`F`) and `Neuropil` (`Fneu`) traces read from the NWB `ophys` module, after filtering ROIs with `iscell`.

ii. 
```python
F_plane = nwb.processing['ophys']['Fluorescence'][plane_name].data[:]
Fneu_plane = nwb.processing['ophys']['Neuropil'][plane_name].data[:]
...
F_list.append(F_plane[:, plane_iscell].T.astype(np.float64))
Fneu_list.append(Fneu_plane[:, plane_iscell].T.astype(np.float64))
...
F = np.concatenate(F_list, axis=0)
Fneu = np.concatenate(Fneu_list, axis=0)
```

iii. The notes justify this by saying the NWB `Deconvolved` data do not match the paper’s custom pipeline, so the agent chose to recompute neural activity from raw fluorescence and neuropil instead.

## 2-b. How is the `neural` data processed?

i. The agent computes `dFF` from `F` and `Fneu`, keeps that `dFF` as the neural signal, and does not deconvolve it into events for the final dataset. The processing is: neuropil subtraction with coefficient 0.7, per-trial add-back of the neuropil mean, maximin-style baseline estimation within each trial, then `(F - baseline)/|baseline|`, then Gaussian smoothing with sigma 2.

ii. 
```python
def compute_dff(F, Fneu, trial_start_inds, teleport_inds,
                neu_coef=0.7, frame_rate=15.5078125):
    f_ = F - neu_coef * Fneu
    ...
    for i in range(n_trials):
        start = start_inds[i]
        stop = stop_inds[i]
        ...
        f_[:, start:stop] = f_[:, start:stop] + neu_coef * np.nanmean(
            Fneu[:, start:stop], axis=1, keepdims=True)
        flow[:, start:stop] = nansmooth(f_[:, start:stop], 15, axis=1)
        flow[:, start:stop] = ndimage.minimum_filter1d(
            flow[:, start:stop], window_size, axis=1)
        flow[:, start:stop] = ndimage.maximum_filter1d(
            flow[:, start:stop], window_size, axis=1)
    ...
    dff[valid_mask] = (f_[valid_mask] - flow[valid_mask]) / np.abs(flow[valid_mask])
    ...
    dff[:, start:stop] = nansmooth(dff[:, start:stop], 2, axis=1)
```

iii. `CONVERSION_NOTES.md` explicitly says the key neural representation decision was to use computed `dFF`, not the NWB `Deconvolved` array, because the agent believed `dFF` was more faithful to the paper and preserved more information for decoding.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent applies two cell filters: manual Suite2p curation via `iscell[:, 0] == 1`, and then putative interneuron exclusion by dropping cells whose `dFF` has Pearson correlation greater than 0.5 with running speed.

ii. 
```python
ps = nwb.processing['ophys']['ImageSegmentation']['PlaneSegmentation']
iscell = ps['iscell'][:]
plane_idx = ps['planeIdx'][:]
...
plane_iscell = iscell[plane_mask, 0] == 1
...
for c in range(n_cells):
    dff_valid = dff[c, :n_common][valid_frames]
    if np.std(dff_valid) > 0 and np.std(speed_valid) > 0:
        corr = np.corrcoef(dff_valid, speed_valid)[0, 1]
        if not np.isnan(corr) and corr > 0.5:
            interneuron_mask[c] = False
...
dff = dff[interneuron_mask, :]
```

iii. The notes say this matches the paper’s curation: “iscell[:,0]==1 + interneuron exclusion (speed corr > 0.5)”.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start simply by slicing each trial from `start = trial_start_inds[i]` to `end = trial_ends[i]`, so timepoint 0 of each emitted neural trial is the first trial-start frame.

ii. 
```python
for i in range(n_trials):
    start = trial_start_inds[i]
    end = trial_ends[i]
    ...
    trial_neural = dff[:, start:end].astype(np.float32)
```

iii. The notes describe the temporal alignment event as “Trial start (entry to linear track)” and present trial-start slicing as the intended alignment mechanism.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent keeps the native time sampling of the NWB data and does not rebin or resample the emitted trial matrices. In metadata it records a time bin size of `1000.0 / 15.5078125` ms, about 64.5 ms.

ii. 
```python
'metadata': {
    ...
    'time_bin_size': 1000.0 / 15.5078125,
    'frame_rate': 15.5078125,
    ...
}
```

iii. The notes say “Time bin: Native frame rate (~64.5 ms = 1/15.5078125 s)” and “No speed threshold: Don’t apply 2 cm/s cutoff”, indicating the agent intended to preserve native temporal sampling rather than rebinning.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavioral timestamps, specifically `position.timestamps`.

ii. 
```python
timestamps = bts.time_series['position'].timestamps[:].astype(np.float64)
...
time_from_start = (timestamps[start:end] - timestamps[start]).astype(np.float32)
```

iii. The code contains no separate justification comment, but the notes treat behavior timestamps as already aligned to imaging frames and suitable for all time-varying trial signals.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the agent subtracts the timestamp at trial start from all timestamps in that trial slice.

ii. 
```python
time_from_start = (timestamps[start:end] - timestamps[start]).astype(np.float32)
...
input_data[0, :] = time_from_start
```

iii. No special rationale is given beyond this being the direct implementation of “time from start of trial”.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by using the same `[start:end]` frame indices as the neural trial slice.

ii. 
```python
trial_neural = dff[:, start:end].astype(np.float32)
...
time_from_start = (timestamps[start:end] - timestamps[start]).astype(np.float32)
```

iii. The notes repeatedly state that behavior is aligned to imaging frames in the NWB files, so the agent treated common frame indexing as sufficient alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The agent does not use the raw `environment` behavior series. Instead, it derives environment type from the scene name embedded in `nwb.identifier`.

ii. 
```python
identifier = nwb.identifier
scene = parse_scene_name(identifier)
...
def get_environment(scene):
    if '_to_' in scene and 'Env2' in scene.split('_to_')[1]:
        return None
    elif 'Env2' in scene:
        return 1
    else:
        return 0
```

iii. The notes say “Reward zone: Determined from scene name in NWB identifier” and also report that environment labels were verified against scene names, so the agent treated scene metadata as the authoritative source for session/trial condition identity.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The agent parses `Env1`/`Env2` from the scene string, assumes any cross-environment switch happens at trial 30, creates one environment label per trial, and broadcasts that label across all timepoints in the trial.

ii. 
```python
def get_env_per_trial(scene, n_trials, change_trial=30):
    env_labels = np.zeros(n_trials, dtype=int)
    if '_to_' in scene and 'Env2' in scene.split('_to_')[1] and 'Env1' in scene.split('_to_')[0]:
        env_labels[:change_trial] = 0
        env_labels[change_trial:] = 1
    elif 'Env2' in scene:
        env_labels[:] = 1
    else:
        env_labels[:] = 0
    return env_labels
...
env_val = float(env_per_trial[i])
...
input_data[1, :] = env_val
```

iii. The notes explicitly record “Switch after trial 30 (change_trial=30)” and present the environment mapping as a scene-name-derived per-trial label.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the loop index over the segmented trials, not from the raw `trial number` NWB data series.

ii. 
```python
for i in range(n_trials):
    ...
    trial_number = float(i)
    ...
    input_data[2, :] = trial_number
```

iii. The notes describe this as “trial index” in the variable mapping table, implying the agent wanted a simple within-session 0-indexed trial counter.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond assigning the 0-indexed trial index and broadcasting it across the whole trial.

ii. 
```python
trial_number = float(i)
...
input_data[2, :] = trial_number
```

iii. No extra rationale is given; it is treated as a straightforward session-local trial counter.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the `Reward` timestamps in the behavioral data, together with the trial start and end times.

ii. 
```python
reward_ts_obj = bts.time_series['Reward']
reward_timestamps = reward_ts_obj.timestamps[:]
...
trial_start_times = timestamps[trial_start_inds]
trial_end_times = timestamps[trial_ends]
rewarded = determine_reward_per_trial(reward_timestamps, trial_start_times, trial_end_times)
```

iii. The notes say “previous reward: From reward delivery timestamps”, so the agent intentionally based trial outcome on reward-event times rather than a framewise reward series.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The agent first marks each trial as rewarded if any reward timestamp falls between that trial’s start and end times. Then it shifts that per-trial reward vector by one trial, setting the first trial’s previous outcome to 0, and broadcasts the result across each trial.

ii. 
```python
def determine_reward_per_trial(reward_timestamps, trial_start_times, trial_end_times):
    rewarded = np.zeros(n_trials, dtype=int)
    for i in range(n_trials):
        mask = (reward_timestamps >= t_start) & (reward_timestamps <= t_end)
        if np.any(mask):
            rewarded[i] = 1
    return rewarded
...
prev_reward = np.zeros(n_trials, dtype=int)
prev_reward[1:] = rewarded[:-1]
...
input_data[3, :] = prev_outcome
```

iii. The notes explicitly state “previous_trial_outcome: 0=omission, 1=rewarded (per-trial)” and mention a sanity check verifying “trial 0 = 0, subsequent trials match previous reward”.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the `position` behavioral series plus reward-zone coordinates inferred from the session scene name, not from the raw `reward_zone` behavioral series.

ii. 
```python
position = bts.time_series['position'].data[:].astype(np.float64)
...
rz_labels, rz_coords = get_reward_zone_label_per_trial(scene, n_trials)
...
rz_start_cm = rz_coords[i, 0]
rz_end_cm = rz_coords[i, 1]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start_cm, rz_end_cm)
```

iii. The notes say “Reward zone: Determined from scene name in NWB identifier, following reference code logic” and list the canonical A/B/C coordinate ranges as the basis for this derivation.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each trial, the agent computes signed distance to the nearest edge of the current reward zone: negative before the zone, zero inside it, positive after it. It then discretizes those distances into 7 categories.

ii. 
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    distance = np.zeros_like(position)
    before_zone = position < rz_start
    in_zone = (position >= rz_start) & (position <= rz_end)
    after_zone = position > rz_end
    distance[before_zone] = position[before_zone] - rz_start
    distance[in_zone] = 0.0
    distance[after_zone] = position[after_zone] - rz_end
    return distance
...
dist_bins = discretize_distance(dist_to_rz)
```

iii. The notes include a sanity check example: `pos=201.4 in zone [200,250] -> dist=0 -> bin 3`, showing the intended signed-distance semantics.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is thresholded with explicit manual comparisons into the seven instruction-specified bins: `<-50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, `>50`.

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

iii. The notes say the agent verified this binning with example values and intended it to match the decoder task specification exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by computing it from the `position[start:end]` slice for the same trial indices used to slice neural activity.

ii. 
```python
trial_neural = dff[:, start:end].astype(np.float32)
...
trial_pos = position[start:end]
...
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start_cm, rz_end_cm)
```

iii. The agent’s notes treat behavior arrays as frame-aligned to imaging, so common trial slicing is the intended alignment mechanism.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the `position` behavioral time series.

ii. 
```python
position = bts.time_series['position'].data[:].astype(np.float64)
...
trial_pos = position[start:end]
```

iii. No special justification is given beyond position being the direct corridor-location variable.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The agent takes each trial’s position slice and discretizes each sample into one of five 90 cm bins spanning the track.

ii. 
```python
def discretize_position(position):
    bins = np.zeros(len(position), dtype=np.int64)
    bins[position < 90] = 0
    bins[(position >= 90) & (position < 180)] = 1
    bins[(position >= 180) & (position < 270)] = 2
    bins[(position >= 270) & (position < 360)] = 3
    bins[position >= 360] = 4
    return bins
...
pos_bins = discretize_position(trial_pos)
```

iii. The notes list “absolute_position: 5 bins of 90cm” in the planned variable mapping and later report position-bin sanity checks.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Using thresholds at 90, 180, 270, and 360 cm, producing bins 0 through 4.

ii. 
```python
bins[position < 90] = 0
bins[(position >= 90) & (position < 180)] = 1
bins[(position >= 180) & (position < 270)] = 2
bins[(position >= 270) & (position < 360)] = 3
bins[position >= 360] = 4
```

iii. This follows the explicit decoder specification in the task and is reflected in the notes’ variable-mapping table.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. By taking `position[start:end]` for the same trial frame range as `dff[:, start:end]`.

ii. 
```python
trial_neural = dff[:, start:end].astype(np.float32)
trial_pos = position[start:end]
```

iii. The agent’s notes assume common indexing between behavior and imaging frames.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the `lick` behavioral time series.

ii. 
```python
lick = bts.time_series['lick'].data[:].astype(np.float64)
...
trial_lick = lick[start:end]
```

iii. No separate justification is given; this is the direct lick signal from the behavior module.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick is binarized per frame by thresholding at `> 0`.

ii. 
```python
lick_binary = (trial_lick > 0).astype(np.int64)
...
output_data[3, :] = lick_binary
```

iii. The notes describe lick as “Binary (>0 -> 1)” in the variable-mapping table and later verify a lick-bin sanity check.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned by taking the same `[start:end]` slice as the neural trial.

ii. 
```python
trial_neural = dff[:, start:end].astype(np.float32)
trial_lick = lick[start:end]
```

iii. As elsewhere, the justification is the agent’s assumption that behavioral arrays are already frame-aligned to imaging.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The agent derives reward-zone location from the scene name in `nwb.identifier`, not from the framewise `reward_zone` behavioral series.

ii. 
```python
identifier = nwb.identifier
scene = parse_scene_name(identifier)
...
rz_labels, rz_coords = get_reward_zone_label_per_trial(scene, n_trials)
```

iii. The notes say reward zone was “determined from scene name in NWB identifier” and use the paper’s canonical A/B/C coordinates as the source of the labels.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene string is parsed into one or two reward-zone labels. For switch sessions, the agent assumes the switch occurs after trial 30. It then maps labels `A/B/C` to `0/1/2` and broadcasts the per-trial label across all frames of the trial.

ii. 
```python
def get_reward_zone_label_per_trial(scene, n_trials, change_trial=30):
    ...
    if len(rz_info) == 1:
        rz_labels[:] = label
        rz_coords[:] = coords
    else:
        rz_labels[:ct] = label1
        rz_labels[ct:] = label2
        rz_coords[:ct] = coords1
        rz_coords[ct:] = coords2
...
rz_label = rz_labels[i]
rz_loc = {'A': 0, 'B': 1, 'C': 2}.get(rz_label, 0)
...
output_data[4, :] = rz_loc
```

iii. The notes explicitly list “Switch after trial 30” and record a sanity check that trials 0–29 and 30 onward have the expected labels in a switch session.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the `Reward` event timestamps in the behavioral module, together with the per-trial time intervals.

ii. 
```python
reward_ts_obj = bts.time_series['Reward']
reward_timestamps = reward_ts_obj.timestamps[:]
...
rewarded = determine_reward_per_trial(reward_timestamps, trial_start_times, trial_end_times)
```

iii. The notes say “reward outcome: From reward delivery timestamps”, so the agent intentionally based this output on reward-event times.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the agent sets the reward outcome to 1 if any reward timestamp falls within that trial’s start/end times; otherwise 0. It then broadcasts that trial-level label across all frames in the trial.

ii. 
```python
def determine_reward_per_trial(reward_timestamps, trial_start_times, trial_end_times):
    rewarded = np.zeros(n_trials, dtype=int)
    for i in range(n_trials):
        mask = (reward_timestamps >= t_start) & (reward_timestamps <= t_end)
        if np.any(mask):
            rewarded[i] = 1
    return rewarded
...
reward_outcome = rewarded[i]
...
output_data[5, :] = reward_outcome
```

iii. The notes say the agent verified that “reward outcome matches reward delivery timestamps”, which is the main stated justification for this implementation.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles a few edge cases, but not many. It truncates trial slices to the shortest of neural length and behavior length, replaces `NaN` neural samples with 0 inside each kept trial, skips sessions with fewer than two valid trials, and skips sessions whose scene name cannot be parsed into a reward zone. It does not implement the reference code’s `<50`-frame short-trial filter or reward-timestamp alignment assertions.

ii. 
```python
end = min(end, dff.shape[1], len(position))
n_frames = end - start
if n_frames < 2:
    continue
...
trial_neural = np.nan_to_num(trial_neural, nan=0.0)
...
if rz_labels is None:
    print(f"    WARNING: Could not determine reward zone, skipping session")
    return None
...
if len(neural_trials) < 2:
    print(f"    WARNING: Less than 2 valid trials, skipping session")
    return None
```

iii. The notes justify these mainly as practical edge-case handling: “Handled length mismatch between neural and behavioral data”, “First trial has previous_trial_outcome = 0”, and support for sessions with unusually few trials.

## 13-a. What are the most time-consuming steps of the code?

i. The code’s most expensive steps are reading large NWB files, computing `dFF` trial-by-trial, and the per-cell interneuron correlation pass. The notes explicitly report per-session timings for load, `dFF`, interneuron filtering, and trial segmentation.

ii. 
```python
with NWBHDF5IO(nwb_file, 'r') as io:
    nwb = io.read()
...
dff = compute_dff(F, Fneu, trial_start_inds, trial_ends,
                  neu_coef=0.7, frame_rate=frame_rate)
...
for c in range(n_cells):
    dff_valid = dff[c, :n_common][valid_frames]
    ...
    corr = np.corrcoef(dff_valid, speed_valid)[0, 1]
```

iii. `CONVERSION_NOTES.md` gives explicit timing estimates, with `dFF` computation being the largest compute step after file loading.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious vectorization targets are the per-cell interneuron loop, the per-trial reward-detection loop, the repeated per-trial loop that constructs neural/input/output trial arrays, and the per-trial loops inside `compute_dff`.

ii. 
```python
for c in range(n_cells):
    ...
for i in range(n_trials):
    mask = (reward_timestamps >= t_start) & (reward_timestamps <= t_end)
...
for i in range(n_trials):
    ...
    neural_trials.append(trial_neural)
    input_trials.append(input_data)
    output_trials.append(output_data)
...
for i in range(n_trials):
    start = start_inds[i]
    stop = stop_inds[i]
    ...
```

iii. The notes mention the per-cell correlation check as a code inefficiency and say vectorization was used “where possible”, but most of these loops were left in straightforward scalar form.

## 13-c. What processing does the code repeat multiple times?

i. The code repeats trial-wise iteration several times over the same session: once to find trial ends, twice inside `compute_dff` (baseline then smoothing), once to determine reward outcome per trial, and once to build output trial objects. It also repeatedly parses/broadcasts session-level quantities into per-trial arrays.

ii. 
```python
for i in range(n_trials):
    later_teleports = teleport_inds[teleport_inds > trial_start_inds[i]]
...
for i in range(n_trials):
    start = start_inds[i]
    stop = stop_inds[i]
    ...
for i in range(n_trials):
    mask = (reward_timestamps >= t_start) & (reward_timestamps <= t_end)
...
for i in range(n_trials):
    ...
    input_data[1, :] = env_val
    input_data[2, :] = trial_number
    input_data[3, :] = prev_outcome
```

iii. The notes do not call all of these out explicitly, but they do describe the main conversion as a sequence of separate trial-segmentation and per-trial processing passes rather than a fused implementation.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code creates several temporary values that are not used downstream: `dt`, `input_time_varying`, and `input_per_trial` are computed and then ignored; the `result` dict stores `env_per_trial`, `rz_labels`, and `rewarded`, but those are not carried into the final saved dataset unless plotting is requested while the session is in memory.

ii. 
```python
dt = 1.0 / frame_rate
...
input_time_varying = time_from_start.reshape(1, -1)
input_per_trial = np.array([env_val, trial_number, prev_outcome], dtype=np.float32)
...
result = {
    'neural': neural_trials,
    'input': input_trials,
    'output': output_trials,
    ...
    'env_per_trial': env_per_trial,
    'rz_labels': rz_labels,
    'rewarded': rewarded,
}
```

iii. There is no explicit justification for these extra computations in the notes; they appear to be leftover scaffolding from development and plotting support.
