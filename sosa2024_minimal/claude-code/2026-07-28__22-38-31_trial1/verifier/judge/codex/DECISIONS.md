# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all subject directories under `data/` whose names start with `sub-`, then loads every `.nwb` file in each subject directory. Each NWB file is treated as one session and is read with `pynwb`. Trial-level data are then extracted later inside `process_session()`.

ii. 
```python
def load_nwb(filepath):
    from pynwb import NWBHDF5IO
    io = NWBHDF5IO(filepath, 'r')
    nwb = io.read()
    ...

def convert_all_data(data_dir, output_path, sample_path=None):
    subjects = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
    ...
    for subj_dir in subjects:
        subj_path = os.path.join(data_dir, subj_dir)
        nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
        ...
        for nwb_file in nwb_files:
            result = process_session(nwb_file, compute_own_dff=True)
```

iii. In the trajectory, the agent explicitly noted that the data were “in NWB format with behavior+ophys files per session” and later handled all subject directories and all session files. The stated goal throughout was to match the paper/reference pipeline while iterating over the full dataset.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level directories named `sub-*`. The stored subject name for each session is also taken from `nwb.subject.subject_id`.

ii. 
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
...
subject_name = subj_dir.replace('sub-', '')
...
'subject_id': nwb.subject.subject_id if nwb.subject else None,
```

iii. The trajectory says the agent identified 11 subject directories and treated them as the mice in the dataset.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file is treated as one session.

ii. 
```python
nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
...
for nwb_file in nwb_files:
    result = process_session(nwb_file, compute_own_dff=True)
```

iii. The trajectory explicitly says each NWB file is a behavior+ophys file per session.

## 1-d. How are the data split into trials?

i. Trials are split using all positive samples of `trial_start` as trial starts and the first later positive sample of `teleport` as the trial end. The code also requires `scanning` to be on somewhere inside the interval.

ii. 
```python
def get_trial_boundaries(trial_start_signal, teleport_signal, trial_numbers, scanning):
    start_indices = np.where(trial_start_signal > 0)[0]
    end_indices = np.where(teleport_signal > 0)[0]
    ...
    for s in start_indices:
        next_ends = end_indices[end_indices > s]
        if len(next_ends) > 0:
            e = next_ends[0]
            if np.any(scanning[s:e] == 1):
                trial_starts.append(s)
                trial_ends.append(e)
```

iii. The trajectory does not give a long separate defense for this step, but the code reflects the agent’s general belief that trial alignment should be based on `trial_start` and `teleport`, with a small extra validity check from `scanning`.

## 1-e. How are trials filtered based on quality controls?

i. Trials shorter than 5 timepoints are dropped. Sessions with fewer than 3 detected trials are skipped, and sessions with fewer than 2 valid trials after filtering are skipped.

ii. 
```python
if len(trial_starts) < 3:
    print(f"    Skipping: only {len(trial_starts)} trials")
    return None
...
for t in range(len(trial_starts)):
    s, e = trial_starts[t], trial_ends[t]
    n_timepoints = e - s

    if n_timepoints < 5:
        continue
...
if valid_trial_count < 2:
    print(f"    Skipping: only {valid_trial_count} valid trials")
    return None
```

iii. The trajectory does not contain an explicit justification for the 5-sample threshold. The visible justification is pragmatic filtering for extremely short trials plus the decoder requirement that each session retain at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The converted neural signal is derived from raw `Fluorescence` and `Neuropil` traces, with `Deconvolved` also loaded but not used in the default path.

ii. 
```python
if n_planes == 1:
    fluorescence = np.array(ophys['Fluorescence']['plane0'].data[:])
    neuropil_data = np.array(ophys['Neuropil']['plane0'].data[:])
    deconvolved = np.array(ophys['Deconvolved']['plane0'].data[:])
...
F = nwb_data['fluorescence'].T
Fneu = nwb_data['neuropil'].T
deconv_nwb = nwb_data['deconvolved'].T
```

iii. In the trajectory the agent first considered using NWB `Deconvolved`, then explicitly decided to “stick with my own pipeline” based on fluorescence and neuropil to better match the paper’s described processing.

## 2-b. How is the `neural` data processed?

i. The agent computes trial-masked dF/F from fluorescence and neuropil using neuropil subtraction (`0.7`), per-trial neuropil mean add-back, maximin baseline with a 300-sample window, then 2-sample Gaussian smoothing. It then deconvolves each trial with OASIS and uses those events as `neural`.

ii. 
```python
def compute_dff_per_trial(F, Fneu, trial_starts, trial_ends, frame_rate=FRAME_RATE):
    f_ = np.full((n_neurons, n_samples), np.nan)
    f_neu_ = np.full((n_neurons, n_samples), np.nan)
    ...
    f_[:, nanmask] = f_[:, nanmask] - NEUROPIL_COEF * f_neu_[:, nanmask]
    ...
    f_[:, start:end] = f_[:, start:end] + NEUROPIL_COEF * np.nanmean(
        f_neu_[:, start:end], axis=1, keepdims=True)
    f_smooth = nansmooth(f_[:, start:end], 15, axis=1)
    baseline = ndimage.minimum_filter1d(f_smooth, window_size, axis=-1)
    baseline = ndimage.maximum_filter1d(baseline, window_size, axis=-1)
    dff[:, start:end] = (f_[:, start:end] - baseline) / np.abs(baseline)
    ...
    dff[:, start:end] = nansmooth(dff[:, start:end], 2, axis=1)
```
```python
for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
    trial_dff = dff[cell_indices, start:end]
    trial_dff_clean = np.nan_to_num(trial_dff, nan=0.0)
    events_trial = deconvolve_oasis(trial_dff_clean,
                                     frame_rate=effective_frame_rate)
    events[:, start:end] = events_trial
```

iii. This is one of the most explicit justifications in the trajectory. The agent noticed its early dF/F implementation gave too many speed-correlated cells, then said it needed to “fix it to more closely match the reference code” and “stick with my own pipeline” of neuropil subtraction, maximin baseline, smoothing, and OASIS deconvolution rather than rely on NWB `Deconvolved`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: only curated cells with `iscell[:, 0] == 1` are kept, and then putative interneurons are removed if their dF/F has correlation greater than `0.5` with running speed.

ii. 
```python
iscell = nwb_data['iscell'][:, 0].astype(bool)
...
cell_mask = filter_interneurons(dff, speed, iscell)
...
def filter_interneurons(dff, speed, iscell_mask):
    mask = np.copy(iscell_mask)
    ...
    for idx in cell_indices:
        ...
        corr = np.corrcoef(cell_data, speed_data)[0, 1]
        if corr > INTERNEURON_SPEED_CORR_THR:
            mask[idx] = False
```

iii. The trajectory shows this filter was important to the agent. It repeatedly checked the interneuron exclusion rate against the paper and treated mismatches as a sign the dF/F pipeline needed revision.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start simply by slicing the per-session events array from each trial start index to trial end index.

ii. 
```python
for t in range(len(trial_starts)):
    s, e = trial_starts[t], trial_ends[t]
    ...
    neural_trial = events[:, s:e].copy()
```

iii. The trajectory consistently describes the dataset as trial-start aligned, and the code performs no extra offset or interpolation beyond trial slicing.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use the native imaging/behavior sampling grid at about 15.5 Hz, stored as `TIME_BIN_MS = 1000 / FRAME_RATE` (about 64.5 ms). No temporal rebinning is applied.

ii. 
```python
FRAME_RATE = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE
...
'time_bin_size': TIME_BIN_MS,
```

iii. In the trajectory, the agent notes that imaging runs at approximately 15.5 Hz and keeps that native binning rather than resample the data.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavioral timestamps, specifically `position.timestamps`.

ii. 
```python
'timestamps': np.array(bts.time_series['position'].timestamps[:]),
...
time_from_start = (timestamps[s:e] - timestamps[s])
```

iii. The trajectory does not single this out, but it states that the behavioral time series share the same sampling grid, so the agent used one of those timestamp streams directly.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the first timestamp in the trial is subtracted from that trial’s timestamp vector.

ii. 
```python
time_from_start = (timestamps[s:e] - timestamps[s])
...
input_trial[0, :] = time_from_start
```

iii. No extra justification is recorded beyond straightforward temporal alignment to trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by using the same sample indices used to slice the neural events array. If neural and behavioral lengths differ by an off-by-one, both are truncated to the shorter length first.

ii. 
```python
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    F = F[:, :min_len]
    Fneu = Fneu[:, :min_len]
    ...
    nwb_data['timestamps'] = nwb_data['timestamps'][:min_len]
...
time_from_start = (timestamps[s:e] - timestamps[s])
neural_trial = events[:, s:e].copy()
```

iii. The trajectory explicitly mentions an “off by one between neural and behavioral data” and says the fix is to truncate to the shorter length.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the `environment` behavioral time series.

ii. 
```python
'environment': np.array(bts.time_series['environment'].data[:]),
...
env_type = float(environment[s])
```

iii. The trajectory treated environment as one of the standard behavioral signals already aligned to the neural samples.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The agent takes the first environment value in each trial and broadcasts it across the trial. If that value is negative, it defaults to `0`.

ii. 
```python
env_type = float(environment[s])
if env_type < 0:
    env_type = 0.0
...
input_trial[1, :] = env_type
```

iii. There is no explicit trajectory discussion of the negative-value fallback. The visible rationale is pragmatic: treat environment as a per-trial binary variable and fill unknown values with a default.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the order of the extracted trial loop, not from the stored NWB `trial number` variable.

ii. 
```python
for t in range(len(trial_starts)):
    ...
    trial_num = float(t)
```

iii. The trajectory centers trial segmentation on `trial_start` and `teleport`, and the code then uses that segmented order as the within-session trial number.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond taking the zero-based loop index and broadcasting it across the trial.

ii. 
```python
trial_num = float(t)
...
input_trial[2, :] = trial_num
```

iii. No separate justification is recorded beyond using the extracted trial order.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived indirectly from `Reward.timestamps`, trial timestamps, and the `reward_zone` signal. The previous-trial outcome is based on the agent’s per-trial `is_rewarded` variable.

ii. 
```python
def determine_trial_rewarded(position, rz_signal, reward_timestamps, timestamps,
                              trial_start, trial_end):
    reward_in_trial = np.any(
        (reward_timestamps >= t_start) & (reward_timestamps <= t_end)
    )
    rz_trial = rz_signal[trial_start:trial_end]
    entered_rz = np.any(rz_trial > 0)
    return int(reward_in_trial and entered_rz)
...
prev_outcome = float(is_rewarded[t - 1])
```

iii. The trajectory discusses reward-related consistency checks and reward-zone interpretation; the code shows that the agent chose to define a rewarded trial using both reward delivery and reward-zone entry.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The first trial is assigned `1.0` by assumption. For later trials, the value is the previous entry of `is_rewarded`, where `is_rewarded` equals 1 only if a reward timestamp falls within the previous trial and the mouse entered the reward zone.

ii. 
```python
if t == 0:
    prev_outcome = 1.0  # assume rewarded before session
else:
    prev_outcome = float(is_rewarded[t - 1])
...
input_trial[3, :] = prev_outcome
```

iii. The trajectory does not explicitly justify assuming the pre-session previous trial was rewarded. The only clear rationale available is the code’s own assumption and the broader effort to create a fully populated per-trial covariate.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the `position` time series and a per-trial reward-zone label inferred from the `reward_zone` signal.

ii. 
```python
def determine_reward_zone_label(position, rz_signal, trial_start, trial_end):
    pos_trial = position[trial_start:trial_end]
    rz_trial = rz_signal[trial_start:trial_end]
    rz_positions = pos_trial[rz_trial > 0]
    ...

pos_trial = position[s:e]
rz_label = rz_labels[t]
rz_start, rz_end = REWARD_ZONES[rz_label]
dist_to_rz = np.array([distance_to_reward_zone(p, rz_start, rz_end)
                       for p in pos_trial])
```

iii. The trajectory explicitly says the agent interpreted `reward_zone > 0` as meaning the mouse is in the reward zone, then used the positions of those samples to infer which zone was active.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The agent first infers the current trial’s reward-zone label by taking the mean position of samples where `reward_zone > 0` and assigning the nearest zone center. Missing trial labels are filled from neighboring trials. It then computes signed distance to the nearest edge of that zone and discretizes the result.

ii. 
```python
mean_rz_pos = np.mean(rz_positions)
...
for zone, (start, end) in REWARD_ZONES.items():
    zone_center = (start + end) / 2
    dist = abs(mean_rz_pos - zone_center)
```
```python
for i in range(n_trials):
    if labels[i] is None:
        for j in range(i + 1, n_trials):
            if labels[j] is not None:
                labels[i] = labels[j]
                break
        if labels[i] is None:
            for j in range(i - 1, -1, -1):
                if labels[j] is not None:
                    labels[i] = labels[j]
                    break
```
```python
def distance_to_reward_zone(position, rz_start, rz_end):
    if position < rz_start:
        return position - rz_start
    elif position > rz_end:
        return position - rz_end
    else:
        return 0.0
```

iii. The trajectory says the agent observed that `reward_zone` behaves cumulatively and that the zone could be identified from the positions where it is positive. It did not adopt the reference’s more elaborate Viterbi segmentation; the justification visible in the log is a simpler nearest-zone interpretation.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is thresholded into 7 classes using hand-written comparisons equivalent to the requested bins: `< -50`, `[-50, -10)`, `[-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, `> 50`.

ii. 
```python
def discretize_distance(distances):
    result = np.zeros(len(distances), dtype=int)
    for i, d in enumerate(distances):
        if d < -50:
            result[i] = 0
        elif d < -10:
            result[i] = 1
        elif d < 0:
            result[i] = 2
        elif d == 0:
            result[i] = 3
        elif d <= 10:
            result[i] = 4
        elif d <= 50:
            result[i] = 5
        else:
            result[i] = 6
```

iii. The trajectory and code comments both indicate the agent was trying to match the instructed output bins exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by computing it from the same `[s:e]` trial slice used for the neural events.

ii. 
```python
neural_trial = events[:, s:e].copy()
...
pos_trial = position[s:e]
...
output_trial[0, :] = dist_binned
```

iii. No extra alignment logic was added because the agent treated the behavioral and neural arrays as already sample-aligned after any length truncation.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the `position` behavioral time series.

ii. 
```python
'position': np.array(bts.time_series['position'].data[:]),
...
pos_trial = position[s:e]
```

iii. The trajectory repeatedly treats position as a directly usable behavioral stream aligned to imaging time.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The agent clips position to the `[0, 450)` corridor range and then bins it into five 90 cm bins with `floor(position / 90)`.

ii. 
```python
pos_clipped = np.clip(pos_trial, 0, TRACK_LENGTH - 0.001)
pos_binned = discretize_position(pos_clipped, n_bins=5)
```
```python
def discretize_position(positions, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins
    result = np.clip(np.floor(positions / bin_size).astype(int), 0, n_bins - 1)
    return result
```

iii. The trajectory does not discuss the clipping choice separately. The code comments show the rationale: keep position within the nominal 450 cm track before categorization.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal-width bins over the 450 cm track: 0 to 90, 90 to 180, 180 to 270, 270 to 360, and 360 to 450 cm.

ii. 
```python
def discretize_position(positions, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins  # 90 cm bins
    result = np.clip(np.floor(positions / bin_size).astype(int), 0, n_bins - 1)
```

iii. The code comments explicitly note “90 cm bins,” matching the instruction.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned by computing position bins from the same per-trial index range used for neural events.

ii. 
```python
neural_trial = events[:, s:e].copy()
pos_trial = position[s:e]
output_trial[1, :] = pos_binned
```

iii. No extra justification was recorded beyond shared indexing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the `lick` behavioral time series.

ii. 
```python
'lick': np.array(bts.time_series['lick'].data[:]),
...
lick_cumul = nwb_data['lick']
lick_trial = lick_cumul[s:e].copy()
```

iii. The trajectory treats lick as a standard already-aligned behavioral stream.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick is binarized: any positive lick value becomes 1, else 0.

ii. 
```python
lick_trial = lick_cumul[s:e].copy()
lick_binary = (lick_trial > 0).astype(float)
...
output_trial[3, :] = lick_binary.astype(int)
```

iii. The code comment says “convert cumulative to binary,” which is the agent’s justification for thresholding.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned by slicing the same `[s:e]` trial interval used for neural activity.

ii. 
```python
neural_trial = events[:, s:e].copy()
lick_trial = lick_cumul[s:e].copy()
output_trial[3, :] = lick_binary.astype(int)
```

iii. No additional alignment step is used beyond shared indices.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from `reward_zone` and `position`, using the same inferred per-trial zone label used for distance-to-zone.

ii. 
```python
rz_labels = determine_reward_zone_for_all_trials(
    position, rz_signal, trial_starts, trial_ends)
...
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_label]
```

iii. The trajectory explicitly discusses interpreting `reward_zone > 0` positions to recover zone identity.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The agent estimates a trial’s reward-zone label from the mean position of samples where `reward_zone > 0`, assigns the nearest of zones A/B/C, fills missing trials from neighboring labels, then maps `A/B/C` to `0/1/2`.

ii. 
```python
labels[i] = determine_reward_zone_label(
    position, rz_signal, trial_starts[i], trial_ends[i])
...
if labels[i] is None:
    for j in range(i + 1, n_trials):
        if labels[j] is not None:
            labels[i] = labels[j]
            break
...
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_label]
```

iii. The trajectory justifies this at a high level by saying the active reward zone can be inferred from where `reward_zone` is positive; it does not record any more formal temporal model.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward.timestamps`, the behavioral timestamps, and the `reward_zone` signal. The outcome is 1 only when the trial both contains a reward timestamp and has `reward_zone > 0` somewhere in the trial.

ii. 
```python
reward_timestamps = nwb_data['reward_timestamps']
...
reward_in_trial = np.any(
    (reward_timestamps >= t_start) & (reward_timestamps <= t_end)
)
rz_trial = rz_signal[trial_start:trial_end]
entered_rz = np.any(rz_trial > 0)
return int(reward_in_trial and entered_rz)
```

iii. The trajectory’s reward-zone discussion shows the agent was tying reward outcome to actual entry into the reward zone, not only to reward timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, reward timestamps are checked against the trial’s start/end timestamps. The output is 1 only if a reward occurred during the trial and the mouse entered the reward zone. That per-trial value is then broadcast across all timepoints of the trial.

ii. 
```python
rew = determine_trial_rewarded(
    position, rz_signal, reward_timestamps, timestamps,
    trial_starts[i], trial_ends[i])
...
rew_outcome = is_rewarded[t]
...
output_trial[5, :] = rew_outcome
```

iii. No separate trajectory message explains the additional `entered_rz` requirement; it is only visible in the code.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles several edge cases pragmatically: off-by-one neural/behavior mismatches are truncated to the shorter length; sessions with too few curated cells or trials are skipped; very short trials are skipped; missing reward-zone labels are filled from neighboring trials and ultimately fall back to zone A; negative environment values default to 0; NaNs in trial neural data are replaced with 0 before deconvolution and before saving the trial array.

ii. 
```python
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    F = F[:, :min_len]
    ...
    nwb_data['timestamps'] = nwb_data['timestamps'][:min_len]
```
```python
if n_total_cells < 5:
    return None
if len(trial_starts) < 3:
    return None
if n_timepoints < 5:
    continue
```
```python
if labels[i] is None:
    ...
if rz_label is None:
    rz_label = 'A'  # fallback
```
```python
trial_dff_clean = np.nan_to_num(trial_dff, nan=0.0)
...
neural_trial = np.nan_to_num(neural_trial, nan=0.0)
```

iii. The trajectory explicitly mentions handling multi-plane sessions, off-by-one mismatches, and decoder-format issues. The remaining fallbacks are justified only implicitly by code comments and a general effort to keep the conversion running.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive steps are repeated NWB loading, per-trial dF/F computation, per-trial OASIS deconvolution, and the nested session/trial/cell loops used during conversion.

ii. 
```python
for subj_dir in subjects:
    ...
    for nwb_file in nwb_files:
        result = process_session(nwb_file, compute_own_dff=True)
```
```python
dff = compute_dff_per_trial(F, Fneu, trial_starts, trial_ends,
                             frame_rate=effective_frame_rate)
...
for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
    events_trial = deconvolve_oasis(trial_dff_clean,
                                     frame_rate=effective_frame_rate)
```

iii. The trajectory does not explicitly benchmark these steps, but it repeatedly waits on the full conversion and revisits the dF/F and deconvolution path when debugging, which is consistent with these being the slow parts.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent leaves several loops unvectorized: the per-cell correlation loop in `filter_interneurons`, the per-trial label-filling loop for reward zones, the per-sample loops in `discretize_distance` and `discretize_speed`, and the per-trial deconvolution loop.

ii. 
```python
for idx in cell_indices:
    corr = np.corrcoef(cell_data, speed_data)[0, 1]
```
```python
for i in range(n_trials):
    ...
    for j in range(i + 1, n_trials):
```
```python
for i, d in enumerate(distances):
    ...
for i, s in enumerate(speeds):
    ...
for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
    ...
```

iii. No explicit trajectory justification is recorded. These are simply visible choices in the implemented code.

## 13-c. What processing does the code repeat multiple times?

i. Within a session, the code repeatedly loops over trials for dF/F smoothing, again for deconvolution, again for reward/outcome extraction, and again for final trial packaging. It also loads NWB `Deconvolved` data even though the default conversion recomputes events from fluorescence.

ii. 
```python
for start, end in zip(trial_starts, trial_ends):
    ...
for start, end in zip(trial_starts, trial_ends):
    ...
for i in range(len(trial_starts)):
    rew = determine_trial_rewarded(...)
...
for t in range(len(trial_starts)):
    ...
```
```python
deconv_nwb = nwb_data['deconvolved'].T  # Pre-computed deconvolved
...
if compute_own_dff:
    ...
else:
    events = deconv_nwb[cell_indices, :]
```

iii. The trajectory does not call this out explicitly. It follows directly from the session-processing structure the agent wrote.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads and crops NWB `Deconvolved` traces on the default path even though it recomputes events from fluorescence; it passes an unused `position` argument into `determine_trial_rewarded`; it computes `valid_trial_count` only to gate session inclusion; and its `--sample-only` flag does not actually reduce the amount of processing.

ii. 
```python
deconv_nwb = nwb_data['deconvolved'].T  # Pre-computed deconvolved
...
if compute_own_dff:
    ...
else:
    events = deconv_nwb[cell_indices, :]
```
```python
def determine_trial_rewarded(position, rz_signal, reward_timestamps, timestamps,
                              trial_start, trial_end):
```
```python
if args.sample_only:
    print("Creating sample dataset only...")
    data = convert_all_data(args.data_dir, args.output, args.sample_output)
else:
    data = convert_all_data(args.data_dir, args.output, args.sample_output)
```

iii. The trajectory eventually notices that `--sample-only` “doesn't actually do anything different from the full run.” The rest of the unnecessary work is visible from static inspection rather than from an explicit self-critique in the log.
