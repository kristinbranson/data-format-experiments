# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans `data/` for subject directories named `sub-*`, then scans each subject directory for every `.nwb` file. Each NWB file is treated as one session and opened directly with `h5py`, not `pynwb`. Within each file it loads behavior arrays from `processing/behavior/BehavioralTimeSeries` and neural arrays from `processing/ophys`.

ii. ```python
def get_all_nwb_files(data_dir, sample=False):
    subjects = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
    ...
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
```

```python
with h5py.File(filepath, 'r') as f:
    behav = f['processing']['behavior']['BehavioralTimeSeries']
    ophys = f['processing']['ophys']
```

iii. `CONVERSION_NOTES.md` says the dataset consists of 11 subject directories and 152 NWB sessions, and describes each file as one mouse-session recording. The notes also state the NWB files already contain the needed preprocessed fields, which is why the agent chose direct HDF5 access.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the `sub-*` directory names, with the `sub-` prefix stripped. The final dataset keeps unique subject IDs in encounter order and stores a per-session `subject_idx`.

ii. ```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
...
'subject': subj.replace('sub-', ''),
```

```python
subj = session_info['subject']
if subj not in subjects_list:
    subjects_list.append(subj)
subject_idx_list.append(subjects_list.index(subj))
```

iii. The notes explicitly identify the 11 mice from directory names and treat those directory IDs as the subject list.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file is treated as one session. Session identity is taken from the file metadata (`general/session_id`) after loading.

ii. ```python
for nwb_file in nwb_files:
    all_files.append({
        'subject': subj.replace('sub-', ''),
        'filepath': os.path.join(subj_dir, nwb_file),
        'filename': nwb_file,
    })
```

```python
session_id = f['general']['session_id'][()].decode() if isinstance(
    f['general']['session_id'][()], bytes) else str(f['general']['session_id'][()])
```

iii. `CONVERSION_NOTES.md` states “Each file contains one session for one mouse” and repeatedly treats the 152 NWB files as the session set.

## 1-d. How are the data split into trials?

i. Trials are defined by pairing every index where `trial_start > 0` with every index where `teleport > 0`, truncating to equal length, then discarding pairs where teleport is not after the start. Each kept trial uses the slice `start:end`.

ii. ```python
trial_starts = np.where(trial_start_signal > 0)[0]
teleports = np.where(teleport_signal > 0)[0]

n_trials = min(len(trial_starts), len(teleports))
trial_starts = trial_starts[:n_trials]
teleports = teleports[:n_trials]

valid = teleports > trial_starts
trial_starts = trial_starts[valid]
teleports = teleports[valid]
```

```python
start = trial_starts[t]
end = teleports[t]
trial_neural = neural_all[start:end, :].T.copy()
```

iii. The notes say the reference code uses `trial_start` to `teleport` as trial boundaries and that teleport periods should be excluded. The trajectory summary also says the agent believed trial boundaries were “trial_start to teleport indices.”

## 1-e. How are trials filtered based on quality controls?

i. The code drops only very short trials, specifically trials with fewer than 5 frames. It does not apply the reference solution’s `<50` frame filter.

ii. ```python
n_timepoints = end - start

if n_timepoints < 5:
    continue  # Skip very short trials
```

iii. I did not find an explicit justification for the `5`-frame cutoff in the notes. The notes later claim there were “no very short trials,” but the implemented threshold is still 5 frames.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Final `neural` trials come from the NWB `Deconvolved` traces after applying a neuron mask. That mask depends on `iscell`, plus an additional interneuron exclusion computed from dF/F derived from `Fluorescence`, `Neuropil`, and `speed`.

ii. ```python
deconv_data = ophys['Deconvolved']['plane0']['data'][:]
fluor_data = ophys['Fluorescence']['plane0']['data'][:]
neuropil_data = ophys['Neuropil']['plane0']['data'][:]
...
iscell = seg['iscell'][:, 0].astype(bool)
...
dff = compute_dff(fluor_data, neuropil_data, trial_starts, teleports)
is_interneuron = identify_interneurons(dff, speed, iscell)
...
neural_all = deconv_data[:, final_neuron_mask]
```

iii. The notes say the NWB files already contain precomputed deconvolved events and that the agent would “use that directly,” but also says it would compute dF/F from fluorescence and neuropil to apply interneuron filtering.

## 2-b. How is the `neural` data processed?

i. The code concatenates all imaging planes, computes dF/F only for interneuron detection, removes putative interneurons, then slices the surviving deconvolved traces into trials and transposes them to `(neurons, time)`. NaNs in the neural array are replaced with zero.

ii. ```python
for plane in planes:
    deconv_parts.append(ophys['Deconvolved'][plane]['data'][:])
...
deconv_data = np.concatenate(deconv_parts, axis=1)
```

```python
dff = compute_dff(fluor_data, neuropil_data, trial_starts, teleports)
is_interneuron = identify_interneurons(dff, speed, iscell)
...
trial_neural = neural_all[start:end, :].T.copy()
trial_neural[np.isnan(trial_neural)] = 0
```

iii. `CONVERSION_NOTES.md` says the intended pipeline was: use precomputed deconvolved events, pool planes, and additionally exclude interneurons via dF/F-speed correlation. The trajectory summary also calls dF/F and interneuron detection major implemented steps.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code first keeps only ROIs with `iscell == True`, then excludes any such neuron whose dF/F has Pearson correlation `> 0.5` with speed. No other neural QC is applied before trial extraction.

ii. ```python
iscell = seg['iscell'][:, 0].astype(bool)
...
is_interneuron = identify_interneurons(dff, speed, iscell)
...
iscell_indices = np.where(iscell)[0]
non_interneuron = ~is_interneuron
final_neuron_mask = np.zeros(n_total_rois, dtype=bool)
final_neuron_mask[iscell_indices[non_interneuron]] = True
```

iii. The notes justify this by citing the paper’s interneuron exclusion criterion (`r > 0.5`) and the `iscell` flag from Suite2p.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The code aligns neural data to trial start implicitly by defining each trial as `start:end` where `start` is the `trial_start` frame. There is no extra offset or resampling around the alignment event.

ii. ```python
start = trial_starts[t]
end = teleports[t]
trial_neural = neural_all[start:end, :].T.copy()
```

iii. The notes say temporal alignment should be to “trial start” and that splitting into trial slices is sufficient to achieve this.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use one native imaging frame per time bin, with `FRAME_PERIOD = 1 / 15.5078125` seconds. No temporal rebinning is applied.

ii. ```python
IMAGING_RATE = 15.5078125  # Hz
FRAME_PERIOD = 1.0 / IMAGING_RATE  # seconds
...
'time_bin_size': 1000.0 / IMAGING_RATE,
```

iii. The notes state the track was sampled at about 15.5 Hz and explicitly say “Time bin = 1 imaging frame.”

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The code does not use raw timestamps. It derives this input from the number of frames in the trial and the constant `FRAME_PERIOD`.

ii. ```python
FRAME_PERIOD = 1.0 / IMAGING_RATE  # seconds
...
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
```

iii. The notes say “Computed from frame timestamps relative to trial start,” but the implementation actually uses a fixed frame clock instead of NWB timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, it creates `0, 1, 2, ...` frame indices and multiplies by the fixed frame period. It does not subtract the actual first behavior timestamp of the trial.

ii. ```python
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
trial_input[0, :] = time_from_start
```

iii. I found no separate code-level justification beyond the constant imaging-rate assumption. The notes imply the agent believed a fixed frame period was acceptable because behavior and imaging were aligned.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Alignment is enforced by using the same `start:end` slice length as the neural trial. The time vector is generated with exactly `n_timepoints = end - start`, so it shares the neural trial’s frame count.

ii. ```python
n_timepoints = end - start
trial_neural = neural_all[start:end, :].T.copy()
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
trial_input = np.zeros((4, n_timepoints), dtype=np.float32)
```

iii. The notes say neural and behavioral streams were already frame aligned in the NWB data, so matching the trial slice length was treated as sufficient.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the NWB behavior variable `environment`.

ii. ```python
environment = behav['environment']['data'][:]
...
env_vals = environment[start:end]
```

iii. The notes identify `environment` as the source and describe it as `0=ENV1, 1=ENV2`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the code takes the median of nonnegative `environment` values in that trial and broadcasts that scalar across all timepoints in the input array.

ii. ```python
valid_env = env_vals[env_vals >= 0]
if len(valid_env) > 0:
    trial_env[t] = int(np.median(valid_env))
...
trial_input[1, :] = env_val
```

iii. The notes justify this by saying environment is effectively constant within a trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is not taken from the NWB `trial number` field. The code uses the loop index `t`, where trials themselves were defined from `trial_start` and `teleport`.

ii. ```python
trial_number = behav['trial number']['data'][:]
...
trial_num = float(t)
trial_input[2, :] = trial_num
```

iii. The notes say the stored `trial number` was considered unreliable relative to `trial_start`, so the sequential per-session trial index was used instead.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No extra processing is applied beyond assigning the within-session trial index and broadcasting it across the trial’s timepoints.

ii. ```python
trial_num = float(t)
trial_input[2, :] = trial_num
```

iii. The notes describe this as a simple sequential within-session trial counter.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the event-based `Reward` timestamps, aligned onto the behavior/imaging frame grid with `np.searchsorted`.

ii. ```python
reward_timestamps = behav['Reward']['timestamps'][:]
behav_timestamps = behav['position']['timestamps'][:]
...
reward_frame_indices = np.searchsorted(behav_timestamps, reward_timestamps)
reward_frame_indices = np.clip(reward_frame_indices, 0, n_samples - 1)
```

iii. The notes say previous reward outcome should be based on whether the prior trial had reward delivery, using the event timestamps.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The code first builds a per-trial `trial_rewarded` boolean by asking whether any mapped reward frame falls within each trial. It then shifts that vector by one trial so trial `t` gets the previous trial’s reward outcome; the first trial is set to 0.

ii. ```python
trial_rewarded = np.zeros(n_trials, dtype=bool)
for t in range(n_trials):
    trial_rewards = np.any((reward_frame_indices >= start) & (reward_frame_indices < end))
    trial_rewarded[t] = trial_rewards
...
prev_trial_outcome = np.zeros(n_trials, dtype=np.int64)
for t in range(1, n_trials):
    prev_trial_outcome[t] = int(trial_rewarded[t - 1])
```

iii. The notes explicitly describe previous trial outcome as a shifted binary reward indicator with trial 0 defaulting to omitted/unknown.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from `position` and an inferred per-trial reward-zone identity that is itself inferred from `reward_zone > 0` within the same trial. The code maps the mean in-zone position to the nearest of zones A/B/C, then carries the last known zone forward and backfills leading missing trials.

ii. ```python
position = behav['position']['data'][:]
reward_zone_signal = behav['reward_zone']['data'][:]
...
zone = identify_reward_zone(position, reward_zone_signal, trial_starts[t], teleports[t])
if zone is not None:
    last_known_zone = zone
trial_rz_label.append(zone if zone is not None else last_known_zone)
```

```python
if trial_rz_label[0] is None:
    for t in range(n_trials):
        if trial_rz_label[t] is not None:
            for tt in range(t):
                trial_rz_label[tt] = trial_rz_label[t]
            break
```

iii. The notes justify reward-zone inference by saying NWB lacks the scene string used in the reference repo, so the reward zone must be inferred from positions where `reward_zone > 0`. The trajectory summary also says the agent spent effort understanding reward-zone switches.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Once the reward-zone start and end coordinates are assigned, the code computes signed distance to the nearest zone edge: negative before the zone, zero inside it, positive after it.

ii. ```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    distance = np.zeros_like(position)
    before = position < rz_start
    inside = (position >= rz_start) & (position <= rz_end)
    after = position > rz_end
    distance[before] = position[before] - rz_start
    distance[inside] = 0.0
    distance[after] = position[after] - rz_end
```

iii. The notes say this was chosen to match the paper’s reward-relative geometry and the decoder bin definitions.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The signed distance is manually thresholded into 7 categories matching the instruction bins, with a dedicated equality check for exactly `0`.

ii. ```python
bins[distance < -50] = 0
bins[(distance >= -50) & (distance < -10)] = 1
bins[(distance >= -10) & (distance < 0)] = 2
bins[distance == 0] = 3
bins[(distance > 0) & (distance <= 10)] = 4
bins[(distance > 10) & (distance <= 50)] = 5
bins[distance > 50] = 6
```

iii. The notes say these bins were taken directly from the task instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It uses the same per-trial `start:end` slice as the neural data, then computes distance from the per-trial position samples within that slice.

ii. ```python
trial_neural = neural_all[start:end, :].T.copy()
trial_pos = position[start:end]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, trial_rz_start[t], trial_rz_end[t])
```

iii. The notes say the behavior streams are already frame-aligned to imaging, so common trial slicing was treated as the alignment mechanism.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavior variable `position`.

ii. ```python
position = behav['position']['data'][:]
...
trial_pos = position[start:end]
```

iii. The notes identify `position` as the direct VR corridor coordinate.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The code slices out the per-trial position trace and discretizes it by dividing by 90 cm, flooring, and clipping to `[0, 4]`.

ii. ```python
def discretize_position(position):
    bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
    return bins
...
pos_bins = discretize_position(trial_pos)
```

iii. The notes say the track is 450 cm long and therefore 5 equal bins should be 90 cm each.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is thresholded into five equal-width 90 cm bins: `[0,90)`, `[90,180)`, `[180,270)`, `[270,360)`, and `[360,450]` after clipping.

ii. ```python
TRACK_LENGTH = 450.0  # cm
...
bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
```

iii. The notes explicitly justify 90 cm bins from the 450 cm corridor length.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned by taking the same `start:end` trial slice used for the neural data.

ii. ```python
trial_neural = neural_all[start:end, :].T.copy()
trial_pos = position[start:end]
```

iii. The notes say neural and behavior arrays are already synchronized framewise in the NWB files.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavior variable `lick`.

ii. ```python
lick = behav['lick']['data'][:]
...
trial_lick = lick_binary[start:end]
```

iii. The notes identify NWB `lick` as a cumulative lick count per frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The code clips the cumulative lick signal to `[0,1]`, flags trials as lick-error trials if more than 35% of frames have raw `lick > 2`, sets those trial segments to `NaN`, and then converts `NaN` back to `0` when building the output.

ii. ```python
lick_binary = np.clip(lick, 0, 1).astype(np.float64)
...
frac_bad = np.sum(trial_lick > 2) / len(trial_lick)
if frac_bad > LICK_ERROR_FRACTION:
    lick_binary[start:end] = np.nan
...
lick_vals = np.where(np.isnan(trial_lick), 0, trial_lick).astype(np.int64)
```

iii. The notes justify this by citing the paper’s lick error threshold and saying lick should be binary. The notes also claim “diff(cumulative_lick) > 0 per frame,” but that is not what the code implements.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned by taking the same trial slice used for neural data and all other behavior outputs.

ii. ```python
trial_neural = neural_all[start:end, :].T.copy()
trial_lick = lick_binary[start:end]
```

iii. The notes say all behavior series are already aligned to imaging frames.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from `reward_zone` and `position`, using the same per-trial inference procedure described in 7-a.

ii. ```python
reward_zone_signal = behav['reward_zone']['data'][:]
position = behav['position']['data'][:]
...
zone = identify_reward_zone(position, reward_zone_signal, trial_starts[t], teleports[t])
```

iii. The notes say NWB lacks the original scene metadata, so reward-zone identity must be inferred from where `reward_zone > 0` occurs on the track.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For each trial, the code chooses the nearest zone A/B/C based on mean in-zone position, propagates the last known zone across missing trials, backfills leading missing values if needed, then maps `A/B/C` to `0/1/2`.

ii. ```python
trial_rz_label.append(zone if zone is not None else last_known_zone)
...
rz_label_to_idx = {'A': 0, 'B': 1, 'C': 2}
trial_rz_idx = np.array([rz_label_to_idx.get(lbl, 0) for lbl in trial_rz_label])
```

iii. The notes justify this as a practical replacement for missing scene strings, and say the inferred labels should remain stable across blocks.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the event-based `Reward` timestamps mapped onto the behavior/imaging frame grid.

ii. ```python
reward_timestamps = behav['Reward']['timestamps'][:]
behav_timestamps = behav['position']['timestamps'][:]
reward_frame_indices = np.searchsorted(behav_timestamps, reward_timestamps)
```

iii. The notes say reward outcome should come from reward-delivery events rather than from a per-frame behavior channel.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the code checks whether any mapped reward frame index falls within that trial’s `start:end` interval. The result is then broadcast across all timepoints in `trial_output[5, :]`.

ii. ```python
trial_rewards = np.any((reward_frame_indices >= start) & (reward_frame_indices < end))
trial_rewarded[t] = trial_rewards
...
reward_out = int(trial_rewarded[t])
trial_output[5, :] = reward_out
```

iii. The notes justify this as the natural binary per-trial reward outcome needed by the decoder.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code crops neural and behavioral arrays to the shorter shared length when they differ. It drops trials shorter than 5 frames, fills missing reward-zone labels by forward propagation and leading backfill, defaults still-missing zones to B `(200,250)`, and turns lick-error trial segments into zeros via `NaN` then `0`.

ii. ```python
if n_behav_samples != n_neural_samples:
    position = position[:n_samples]
    ...
    deconv_data = deconv_data[:n_samples, :]
```

```python
if n_timepoints < 5:
    continue
...
trial_rz_label.append(zone if zone is not None else last_known_zone)
...
trial_rz_start[t], trial_rz_end[t] = 200, 250
...
lick_vals = np.where(np.isnan(trial_lick), 0, trial_lick).astype(np.int64)
```

iii. The notes justify cropping as a small frame-mismatch fix, and justify reward-zone inference because some trials lack an in-zone signal. I did not find a separate justification for defaulting unresolved trials to zone B.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading each NWB file, computing dF/F across all neurons and timepoints, and computing the vectorized interneuron correlation filter. The script records these timings explicitly per session.

ii. ```python
t_load = time.time() - t0
...
t1 = time.time()
dff = compute_dff(fluor_data, neuropil_data, trial_starts, teleports)
t_dff = time.time() - t1
...
t1 = time.time()
is_interneuron = identify_interneurons(dff, speed, iscell)
t_int = time.time() - t1
```

iii. The notes say the full conversion took about 14.6 minutes and specifically mention optimizing dF/F and interneuron detection because they were the main bottlenecks.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized interneuron correlation, but several trial loops remain: reward assignment, reward-zone inference, lick-error detection, environment summarization, previous-outcome creation, and per-trial packing of neural/input/output arrays. These are the main remaining vectorization candidates.

ii. ```python
for t in range(n_trials):
    trial_rewards = np.any((reward_frame_indices >= start) & (reward_frame_indices < end))
...
for t in range(n_trials):
    zone = identify_reward_zone(position, reward_zone_signal, trial_starts[t], teleports[t])
...
for t in range(n_trials):
    if frac_bad > LICK_ERROR_FRACTION:
        lick_binary[start:end] = np.nan
...
for t in range(n_trials):
    start = trial_starts[t]
    end = teleports[t]
    ...
    neural_trials.append(...)
```

iii. The notes explicitly say the agent optimized `identify_interneurons()` by vectorizing Pearson correlation, implying it recognized the remaining per-trial loops as the next place efficiency could improve.

## 13-c. What processing does the code repeat multiple times?

i. The code repeatedly loops over trials to derive closely related per-trial quantities from the same boundaries: reward outcome, reward-zone label, lick QC, environment, previous outcome, and final trial packing. It also recomputes continuous distance-to-reward-zone inside `plot_processing()` even after computing discretized distance for the stored outputs.

ii. ```python
for t in range(n_trials):
    trial_rewards = ...
for t in range(n_trials):
    zone = identify_reward_zone(...)
for t in range(n_trials):
    ...
for t in range(n_trials):
    ...
```

```python
dist_to_rz = compute_distance_to_reward_zone(trial_pos, trial_rz_start[t], trial_rz_end[t])
...
dist = compute_distance_to_reward_zone(position[start:end], trial_rz_start[t], trial_rz_end[t])
```

iii. The notes emphasize iterative trial-wise processing throughout the script and separately mention optional diagnostic plotting, which redoes some calculations for visualization.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The largest unnecessary processing relative to downstream decoder use is the expensive dF/F computation and interneuron detection, since the saved dataset only uses deconvolved traces after masking. The script also loads `reward_data`, `trial_number`, `scanning`, and `plane_idx` without using them in the saved outputs, and it stores extensive timing/session metadata used only for documentation.

ii. ```python
reward_data = behav['Reward']['data'][:]
trial_number = behav['trial number']['data'][:]
scanning = behav['scanning']['data'][:]
plane_idx = seg['planeIdx'][:]
```

```python
dff = compute_dff(fluor_data, neuropil_data, trial_starts, teleports)
is_interneuron = identify_interneurons(dff, speed, iscell)
...
'session_info': all_session_infos,
```

iii. The notes justify dF/F only as an intermediate for interneuron exclusion. Nothing derived from the dF/F traces themselves is saved to `converted_data.pkl`, so that whole branch is discarded except for the neuron mask.
