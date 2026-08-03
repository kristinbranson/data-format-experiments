# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans the top-level `data` directory for subject folders named `sub-*`, then scans each subject folder for all `.nwb` files. Each `.nwb` file is treated as one session and is loaded with `pynwb`. Inside each file, it reads the behavioral time series and ophys groups, including fluorescence, neuropil, precomputed deconvolved traces, segmentation metadata, and several behavioral variables.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
...
for subj_dir in subjects:
    subj_path = os.path.join(data_dir, subj_dir)
    nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
...
def load_nwb(filepath):
    from pynwb import NWBHDF5IO
    io = NWBHDF5IO(filepath, 'r')
    nwb = io.read()
...
    data = {
        'position': np.array(bts.time_series['position'].data[:]),
        'speed': np.array(bts.time_series['speed'].data[:]),
        'lick': np.array(bts.time_series['lick'].data[:]),
        ...
        'fluorescence': fluorescence,
        'neuropil': neuropil_data,
        'deconvolved': deconvolved,
```

iii. The justification in the code header and notes is that the converter should process all NWB sessions from the 11 switch-condition mice, with multi-plane sessions pooled across planes. The trajectory shows the agent explicitly explored the directory structure and NWB contents before implementing this loader.

## 1-b. How are the data split into subjects?

i. Subjects are split by top-level directories whose names begin with `sub-`. The stored subject list is the directory name with the `sub-` prefix removed. Each processed session is assigned a `subject_idx` pointing into that subject list.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
...
subject_name = subj_dir.replace('sub-', '')
if subject_name not in all_subjects:
    all_subjects.append(subject_name)
subj_idx = all_subjects.index(subject_name)
...
subject_idx.append(subj_idx)
```

iii. The notes say the dataset contains 11 switch-condition mice with directories such as `sub-m3`, `sub-m4`, etc., so the AI treated those directories as the subject split.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file under a subject directory is treated as one session. The AI iterates over all session files for each subject and processes them independently.

ii.
```python
for subj_dir in subjects:
    subj_path = os.path.join(data_dir, subj_dir)
    nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
...
    for nwb_file in nwb_files:
        result = process_session(nwb_file, compute_own_dff=True)
```

iii. The loader stores `nwb.session_id` and the notes describe “12-14 sessions each (152 total sessions),” which is the stated justification for mapping one NWB file to one session.

## 1-d. How are the data split into trials?

i. In the implemented code, trial starts are all indices where `trial_start > 0`, and trial ends are the next indices where `teleport > 0`. A candidate trial is kept only if `scanning` is on somewhere between start and end. This is different from the notes, which say trials run from one `trial_start` to the next or end of recording.

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

iii. The notes justify trial segmentation using `trial_start`, and the trajectory shows the agent focused on trial segmentation early. However, the written notes and the code disagree on whether the end is the next `trial_start` or the next `teleport`.

## 1-e. How are trials filtered based on quality controls?

i. The code skips a session if it has fewer than 5 curated cells, skips a session if it has fewer than 3 detected trials before conversion, skips individual trials shorter than 5 timepoints, and skips the session entirely if fewer than 2 valid trials remain.

ii.
```python
if n_total_cells < 5:
    print(f"    Skipping: only {n_total_cells} curated cells")
    return None
...
if len(trial_starts) < 3:
    print(f"    Skipping: only {len(trial_starts)} trials")
    return None
...
if n_timepoints < 5:
    continue
...
if valid_trial_count < 2:
    print(f"    Skipping: only {valid_trial_count} valid trials")
    return None
```

iii. The code itself gives no detailed rationale besides protecting decoder usability. The notes emphasize decoder-format validity and retaining at least two trials per session, but they do not justify the specific threshold of 5 timepoints.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. In the executed path, `neural` is derived from raw `Fluorescence` and `Neuropil` traces, plus the ROI curation mask `iscell`. The script also loads NWB `Deconvolved` traces, but those are only used when `compute_own_dff=False`; the main conversion calls `process_session(..., compute_own_dff=True)`.

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
...
result = process_session(nwb_file, compute_own_dff=True)
```

iii. The notes say the AI was trying to match the paper’s preprocessing pipeline by recomputing dF/F and OASIS events, rather than directly using the stored deconvolved NWB signal.

## 2-b. How is the `neural` data processed?

i. The AI recomputes per-trial dF/F from fluorescence and neuropil, using masked within-trial samples, neuropil subtraction with coefficient 0.7, a per-trial add-back of neuropil mean, a maximin baseline, Gaussian smoothing, then OASIS deconvolution. Multi-plane recordings are concatenated across planes.

ii.
```python
def compute_dff_per_trial(F, Fneu, trial_starts, trial_ends, frame_rate=FRAME_RATE):
    f_ = np.full((n_neurons, n_samples), np.nan)
    f_neu_ = np.full((n_neurons, n_samples), np.nan)
...
    f_[:, nanmask] = f_[:, nanmask] - NEUROPIL_COEF * f_neu_[:, nanmask]
...
    baseline = ndimage.minimum_filter1d(f_smooth, window_size, axis=-1)
    baseline = ndimage.maximum_filter1d(baseline, window_size, axis=-1)
    dff[:, start:end] = (f_[:, start:end] - baseline) / np.abs(baseline)
...
    dff[:, start:end] = nansmooth(dff[:, start:end], 2, axis=1)
...
events_trial = deconvolve_oasis(trial_dff_clean,
                                 frame_rate=effective_frame_rate)
```

iii. The code header and `CONVERSION_NOTES.md` explicitly justify this as “matching `preprocessing.py`,” and the trajectory shows the agent adjusting the dF/F window to 300 samples to get closer to the reference preprocessing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code first keeps ROIs marked as curated cells by `iscell[:, 0]`, then removes additional cells classified as putative interneurons if their activity is correlated with speed above 0.5.

ii.
```python
iscell = nwb_data['iscell'][:, 0].astype(bool)
...
def filter_interneurons(dff, speed, iscell_mask):
    mask = np.copy(iscell_mask)
...
    for idx in cell_indices:
        cell_data = dff[idx, valid]
        speed_data = speed[valid]
        ...
        corr = np.corrcoef(cell_data, speed_data)[0, 1]
        if corr > INTERNEURON_SPEED_CORR_THR:
            mask[idx] = False
```

iii. The notes claim this matches a paper criterion and report an observed removal rate of about 3.8%, with the trajectory showing the agent trying to reduce that rate after noticing it exceeded the paper’s mean.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start by slicing the event matrix using the trial start/end indices and setting time zero to the first timestamp in the same slice. There is no additional shifting or interpolation.

ii.
```python
for t in range(len(trial_starts)):
    s, e = trial_starts[t], trial_ends[t]
...
    neural_trial = events[:, s:e].copy()
...
    time_from_start = (timestamps[s:e] - timestamps[s])
```

iii. The notes say all data are aligned to trial start and that no additional alignment step is required beyond using common trial boundaries.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The code uses a fixed frame rate of `15.5078125 Hz`, corresponding to `64.48 ms` bins, and does not perform temporal rebinning. It assumes this stored sampling interval is already the desired decoder bin size.

ii.
```python
FRAME_RATE = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE
...
'time_bin_size': TIME_BIN_MS,
```

iii. The notes justify this with the NWB frame rate and describe the output as “~64.5 ms time bins.” The trajectory also mentions multi-plane handling and keeping the effective per-plane rate at about 15.5 Hz.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavioral `position` timestamps saved into `nwb_data['timestamps']`. The code uses those timestamps directly rather than the `trial number` timestamps used in the human reference.

ii.
```python
'timestamps': np.array(bts.time_series['position'].timestamps[:]),
...
time_from_start = (timestamps[s:e] - timestamps[s])
```

iii. The code does not justify this choice explicitly. The general assumption in the notes is that the behavior streams are already aligned and share the frame times.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the code subtracts the first timestamp in that trial from all timestamps in the trial, producing a time-from-trial-start series in seconds.

ii.
```python
time_from_start = (timestamps[s:e] - timestamps[s])
...
input_trial[0, :] = time_from_start
```

iii. The notes frame this as alignment to trial start, so subtracting the trial’s initial timestamp is the implied justification.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The input time series is aligned by using the same `[s:e]` slice as the neural data for each trial. If the full-session neural and behavioral arrays differ by one sample, both are truncated to the shorter length before trial slicing.

ii.
```python
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    F = F[:, :min_len]
    ...
    nwb_data['timestamps'] = nwb_data['timestamps'][:min_len]
...
neural_trial = events[:, s:e].copy()
time_from_start = (timestamps[s:e] - timestamps[s])
```

iii. The notes explicitly mention handling off-by-one mismatches by truncating to the minimum length, and the trajectory shows the agent investigating and then adopting that truncation rule.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the `environment` behavioral time series.

ii.
```python
'environment': np.array(bts.time_series['environment'].data[:]),
...
environment = nwb_data['environment']
...
env_type = float(environment[s])
```

iii. The notes say this variable is already a binary environment indicator for ENV1 vs ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The code reads the value at the trial start index, treats it as a per-trial scalar, and broadcasts it across all timepoints in the trial. If the value is negative, it silently replaces it with 0.

ii.
```python
env_type = float(environment[s])
if env_type < 0:
    env_type = 0.0
...
input_trial[1, :] = env_type
```

iii. The notes justify treating environment as a per-trial binary label. They do not explain the fallback for negative values.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The value written to the decoder input is not taken from the NWB `trial number` series. Instead, it is the zero-based loop index `t` after the trial boundaries have been found.

ii.
```python
for t in range(len(trial_starts)):
    s, e = trial_starts[t], trial_ends[t]
...
    trial_num = float(t)
...
    input_trial[2, :] = trial_num
```

iii. The notes say the decoder should use a within-session trial index. No separate justification is given in the code.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. There is no transformation beyond converting the loop counter to float and broadcasting it across the full trial.

ii.
```python
trial_num = float(t)
...
input_trial[2, :] = trial_num
```

iii. The implied justification is that the decoder input should be a continuous per-trial trial count.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. The implemented `previous_trial_outcome` comes from the previously computed `is_rewarded` list, and each `is_rewarded` value is derived from `Reward` timestamps plus the `reward_zone` signal within the current trial.

ii.
```python
reward_timestamps = nwb_data['reward_timestamps']
...
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

iii. The notes describe “reward outcome” as detected from `Reward` timestamps relative to trial boundaries. The additional requirement that the mouse entered the reward zone is only visible in the code.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The code first computes per-trial reward outcomes, then copies the previous trial’s value into every timepoint of the current trial. For the first trial, it hard-codes the previous outcome to `1.0`.

ii.
```python
if t == 0:
    prev_outcome = 1.0  # assume rewarded before session
else:
    prev_outcome = float(is_rewarded[t - 1])
...
input_trial[3, :] = prev_outcome
```

iii. No clear justification is given for assuming the first trial follows a rewarded trial; this appears to be an ad hoc fallback. The notes only say the variable should be binary.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from `position` and an inferred per-trial reward-zone label. That label itself is inferred from the `reward_zone` signal by looking at positions where `reward_zone > 0`.

ii.
```python
position = nwb_data['position']
rz_signal = nwb_data['reward_zone']
...
rz_labels = determine_reward_zone_for_all_trials(
    position, rz_signal, trial_starts, trial_ends)
...
dist_to_rz = np.array([distance_to_reward_zone(p, rz_start, rz_end)
                       for p in pos_trial])
```

iii. The notes explicitly say reward-zone location is determined from the `reward_zone` behavioral signal by finding positions where it is active.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The code infers a reward-zone label per trial by taking the mean position of samples where `reward_zone > 0`, matching that mean to the nearest zone center, and filling missing labels from neighboring trials. It then computes signed distance of each position sample to the chosen zone bounds, with 0 inside the zone, negative before, positive after.

ii.
```python
def determine_reward_zone_label(position, rz_signal, trial_start, trial_end):
    rz_positions = pos_trial[rz_trial > 0]
    ...
    mean_rz_pos = np.mean(rz_positions)
    for zone, (start, end) in REWARD_ZONES.items():
        zone_center = (start + end) / 2
        dist = abs(mean_rz_pos - zone_center)
...
def determine_reward_zone_for_all_trials(position, rz_signal, trial_starts, trial_ends):
    ...
    if labels[i] is None:
        for j in range(i + 1, n_trials):
            if labels[j] is not None:
                labels[i] = labels[j]
                break
...
def distance_to_reward_zone(position, rz_start, rz_end):
    if position < rz_start:
        return position - rz_start
    elif position > rz_end:
        return position - rz_end
    else:
        return 0.0
```

iii. The notes justify the zone inference by saying omission trials may need the zone inferred from neighboring trials. The more specific nearest-center heuristic is only apparent from the code.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The code manually thresholds each continuous distance sample into 7 categories corresponding to `<-50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, and `>50`.

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

iii. The thresholds directly mirror the decoder instructions, which is the apparent justification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by using the same per-trial `[s:e]` slice used for the neural data, then computing distance from the position samples in that slice.

ii.
```python
s, e = trial_starts[t], trial_ends[t]
neural_trial = events[:, s:e].copy()
...
pos_trial = position[s:e]
...
output_trial[0, :] = dist_binned
```

iii. The notes assume behavioral and neural streams are already aligned once the session has been truncated to a common length.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the `position` behavioral time series.

ii.
```python
'position': np.array(bts.time_series['position'].data[:]),
...
pos_trial = position[s:e]
```

iii. The notes describe position as extracted from the `position` time series on a 0-450 cm track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The code clips each position sample into `[0, 450)`, then discretizes into five equal-width 90 cm bins over that range by dividing by `TRACK_LENGTH / 5`.

ii.
```python
pos_clipped = np.clip(pos_trial, 0, TRACK_LENGTH - 0.001)
pos_binned = discretize_position(pos_clipped, n_bins=5)
...
def discretize_position(positions, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins
    result = np.clip(np.floor(positions / bin_size).astype(int), 0, n_bins - 1)
```

iii. The notes justify this as “5 equal-sized bins” over the 450 cm corridor.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The thresholds are implicit in the 90 cm binning: `0-90`, `90-180`, `180-270`, `270-360`, and `360-450` cm after clipping.

ii.
```python
def discretize_position(positions, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins  # 90 cm bins
    result = np.clip(np.floor(positions / bin_size).astype(int), 0, n_bins - 1)
...
'output_values': [
    ['0-90cm', '90-180cm', '180-270cm', '270-360cm', '360-450cm'],
```

iii. The justification is the instruction to use 5 equal-sized bins; the code interprets that as five bins spanning 0-450 cm.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position is aligned by slicing the same trial interval `[s:e]` used for the neural matrix.

ii.
```python
s, e = trial_starts[t], trial_ends[t]
neural_trial = events[:, s:e].copy()
pos_trial = position[s:e]
output_trial[1, :] = pos_binned
```

iii. The notes assume shared session indexing across the neural and behavioral streams after length truncation.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the `lick` behavioral time series.

ii.
```python
'lick': np.array(bts.time_series['lick'].data[:]),
...
lick_cumul = nwb_data['lick']
...
lick_trial = lick_cumul[s:e].copy()
```

iii. The notes describe the lick variable as a lick-related behavioral time series, although they also claim it is cumulative.

## 9-b. What processing is involved in computing `output` *Lick*?

i. In the code, lick is binarized as `1` whenever the raw `lick` value is greater than 0 and `0` otherwise. The notes claim the AI converted cumulative lick counts by differencing, but that is not what the implemented code does.

ii.
```python
lick_trial = lick_cumul[s:e].copy()
lick_binary = (lick_trial > 0).astype(float)
...
output_trial[3, :] = lick_binary.astype(int)
```

iii. The notes justify the target as a binary lick/no-lick decoder output, but the differencing rationale in the notes is inconsistent with the actual implementation.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned by taking the same `[s:e]` slice used for the neural data and writing the resulting binary series into the trial output.

ii.
```python
s, e = trial_starts[t], trial_ends[t]
neural_trial = events[:, s:e].copy()
lick_trial = lick_cumul[s:e].copy()
output_trial[3, :] = lick_binary.astype(int)
```

iii. The notes assume the behavioral and neural arrays are already aligned once truncated to a common session length.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from `reward_zone` and `position`, using the same zone-label inference machinery used for distance-to-zone.

ii.
```python
rz_signal = nwb_data['reward_zone']
position = nwb_data['position']
...
rz_labels = determine_reward_zone_for_all_trials(
    position, rz_signal, trial_starts, trial_ends)
...
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_label]
```

iii. The notes explicitly describe reward-zone determination from positions where `reward_zone` is active.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The code infers a categorical zone label per trial by nearest mean reward-zone position, fills missing labels from neighboring trials, falls back to `'A'` if the label is still `None`, and then maps `A/B/C` to `0/1/2`. It broadcasts the per-trial category across all timepoints in the trial output matrix.

ii.
```python
rz_labels = determine_reward_zone_for_all_trials(
    position, rz_signal, trial_starts, trial_ends)
...
rz_label = rz_labels[t]
if rz_label is None:
    rz_label = 'A'
...
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_label]
...
output_trial[4, :] = rz_loc
```

iii. The notes justify neighbor-based inference for omission trials but do not justify the final hard fallback to zone `A`.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from reward timestamps in the `Reward` time series together with the `reward_zone` signal, because the code only labels a trial as rewarded if a reward timestamp occurs during the trial and the mouse also entered the reward zone.

ii.
```python
'reward_timestamps': np.array(bts.time_series['Reward'].timestamps[:]),
...
reward_in_trial = np.any(
    (reward_timestamps >= t_start) & (reward_timestamps <= t_end)
)
rz_trial = rz_signal[trial_start:trial_end]
entered_rz = np.any(rz_trial > 0)
return int(reward_in_trial and entered_rz)
```

iii. The notes only mention reward timestamps relative to trial boundaries. The additional reward-zone-entry condition is visible only in the implementation.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the code checks whether any reward timestamp falls between the trial start and end timestamps and whether the reward-zone signal was ever active in that trial. It stores the resulting 0/1 value across the full trial output row.

ii.
```python
for i in range(len(trial_starts)):
    rew = determine_trial_rewarded(
        position, rz_signal, reward_timestamps, timestamps,
        trial_starts[i], trial_ends[i])
    is_rewarded.append(rew)
...
rew_outcome = is_rewarded[t]
...
output_trial[5, :] = rew_outcome
```

iii. The code gives no extra rationale. The notes just state the output should be binary reward/no-reward.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code contains several ad hoc fallback rules: off-by-one neural/behavior mismatches are truncated to the shorter length; missing or negative environment values are replaced with `0`; missing reward-zone labels are inferred from neighbors and then forced to `'A'` if still missing; NaNs in neural data are replaced with `0` before decoder export; and if `suite2p` is unavailable, deconvolution falls back to thresholded nonnegative dF/F.

ii.
```python
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    F = F[:, :min_len]
    ...
    nwb_data['timestamps'] = nwb_data['timestamps'][:min_len]
...
if env_type < 0:
    env_type = 0.0
...
if labels[i] is None:
    ...
if rz_label is None:
    rz_label = 'A'
...
neural_trial = np.nan_to_num(neural_trial, nan=0.0)
...
except ImportError:
    events = np.copy(dff_trial)
    events[events < 0] = 0
```

iii. The trajectory explicitly mentions the off-by-one truncation as a fix for observed mismatches. The other fallback rules are implicit in the code rather than explained in the notes.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive steps are likely NWB I/O, loading and concatenating fluorescence/neuropil/deconvolved arrays, per-trial dF/F computation, per-cell speed-correlation filtering, and per-trial OASIS deconvolution. This is substantially heavier than simply using the stored deconvolved data.

ii.
```python
nwb = io.read()
...
fluorescence = np.concatenate(fluor_planes, axis=1)
neuropil_data = np.concatenate(neuro_planes, axis=1)
...
dff = compute_dff_per_trial(F, Fneu, trial_starts, trial_ends,
                             frame_rate=effective_frame_rate)
cell_mask = filter_interneurons(dff, speed, iscell)
...
for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
    ...
    events_trial = deconvolve_oasis(trial_dff_clean,
                                     frame_rate=effective_frame_rate)
```

iii. The notes frame this expensive preprocessing as necessary to match the paper’s method. The trajectory also shows the agent spending time tuning these preprocessing steps.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several obvious Python loops could have been vectorized: the loop over curated cells in `filter_interneurons`, the loop over positions in `discretize_distance`, the loop over speeds in `discretize_speed`, the per-trial reward outcome loop, and the per-trial output-building loop. The distance and speed discretizations in particular could have used `np.digitize`.

ii.
```python
for idx in cell_indices:
    cell_data = dff[idx, valid]
    ...
for i, d in enumerate(distances):
    if d < -50:
        result[i] = 0
...
for i, s in enumerate(speeds):
    if s < 2:
        result[i] = 0
...
for i in range(len(trial_starts)):
    rew = determine_trial_rewarded(...)
...
for t in range(len(trial_starts)):
    s, e = trial_starts[t], trial_ends[t]
```

iii. The code contains no explicit justification for leaving these as Python loops.

## 13-c. What processing does the code repeat multiple times?

i. The code repeatedly scans trial boundaries several times within a session: once to compute dF/F masking, again to deconvolve trial by trial, again to infer reward outcomes, and again to build the exported per-trial arrays. It also rereads both fluorescence and precomputed deconvolved traces even though the precomputed deconvolved data are unused on the executed path.

ii.
```python
for start, end in zip(trial_starts, trial_ends):
    f_[:, start:end] = F[:, start:end]
...
for start, end in zip(trial_starts, trial_ends):
    ...
    dff[:, start:end] = nansmooth(dff[:, start:end], 2, axis=1)
...
for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
    ...
    events[:, start:end] = events_trial
...
for i in range(len(trial_starts)):
    rew = determine_trial_rewarded(...)
...
for t in range(len(trial_starts)):
    ...
```

iii. The notes do not discuss this repetition. It follows from the AI’s choice to rebuild neural preprocessing from fluorescence instead of using the stored event matrix.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code performs several computations that are not preserved in the final decoder dataset: it loads NWB deconvolved traces but discards them when `compute_own_dff=True`; it computes full dF/F arrays only to replace them with deconvolved events; it keeps `plane_idx` in the per-session return value even though the final exported `brain_region_idx` ignores plane identity; and it imports modules such as `sys` and `interpolate` that are never used.

ii.
```python
import sys
from scipy import ndimage, interpolate
...
deconv_nwb = nwb_data['deconvolved'].T
...
if compute_own_dff:
    dff = compute_dff_per_trial(...)
    ...
    events[:, start:end] = events_trial
...
planes = nwb_data['plane_idx'][cell_indices]
...
brain_region_idx.append(np.zeros(result['n_neurons'], dtype=int))
```

iii. The notes justify the extra dF/F work as trying to reproduce paper preprocessing, but from the decoder export perspective those intermediate products are discarded.
