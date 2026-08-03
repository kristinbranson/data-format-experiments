# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all subdirectories starting with `sub-` in the data directory, then finds all `.nwb` files in each subdirectory. Each NWB file is loaded using `pynwb.NWBHDF5IO`. All behavioral and neural data arrays are read from the NWB file into a dictionary. The AI loads fluorescence, neuropil, and deconvolved data plus all behavioral time series.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
...
nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
...
io = NWBHDF5IO(filepath, 'r')
nwb = io.read()
bts = nwb.processing['behavior']['BehavioralTimeSeries']
ophys = nwb.processing['ophys']
```

iii. The AI loads all NWB files from all subject directories, matching the expected 11 subjects and 152 sessions from the paper.

## 1-b. How are the data split into subjects?

i. Subjects correspond to subdirectories of the data directory starting with `sub-`. The subject name is extracted by removing the `sub-` prefix.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
...
subject_name = subj_dir.replace('sub-', '')
```

iii. The 11 subject directories match the number of mice reported in the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. All `.nwb` files within each subject directory are processed as separate sessions.

ii.
```python
nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
for nwb_file in nwb_files:
    result = process_session(nwb_file, compute_own_dff=True)
```

iii. The naming convention `sub-<id>_ses-<num>_behavior+ophys.nwb` confirms each file is a session.

## 1-d. How are the data split into trials?

i. Trial starts are identified where `trial_start > 0`. Trial ends are identified by the next `teleport > 0` signal after each start. Additionally, the AI requires that `scanning == 1` during the trial.

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
    return trial_starts, trial_ends
```

iii. The AI uses the `trial_start` and `teleport` signals from the NWB behavioral data to determine trial boundaries. It additionally filters for scanning being active.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 5 timepoints are skipped. Sessions with fewer than 3 trials total or fewer than 5 curated cells are skipped entirely. After processing, sessions with fewer than 2 valid trials are skipped.

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

iii. The AI's CONVERSION_NOTES.md doesn't discuss the short trial threshold in detail. The minimum of 5 timepoints is very permissive compared to the reference's 50.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI loads `Fluorescence`, `Neuropil`, and `Deconvolved` data from the NWB `ophys` processing module. When `compute_own_dff=True` (the default), the AI recomputes dF/F from raw fluorescence and neuropil, then deconvolves with OASIS. It does NOT use the pre-computed `Deconvolved` data.

ii.
```python
F = nwb_data['fluorescence'].T
Fneu = nwb_data['neuropil'].T
deconv_nwb = nwb_data['deconvolved'].T
...
if compute_own_dff:
    dff = compute_dff_per_trial(F, Fneu, trial_starts, trial_ends, ...)
    ...
    events_trial = deconvolve_oasis(trial_dff_clean, ...)
    events[:, start:end] = events_trial
```

iii. The AI justifies this as "matching preprocessing.py" from the reference code, implementing neuropil subtraction, maximin baseline, Gaussian smoothing, and OASIS deconvolution.

## 2-b. How is the `neural` data processed?

i. The AI implements a multi-step processing pipeline:
1. Select curated cells (`iscell[:, 0] == 1`)
2. Neuropil subtraction (coefficient = 0.7)
3. Maximin baseline with 300-sample window
4. dF/F = (F - baseline) / |baseline|
5. Smooth dF/F with 2-sample Gaussian kernel
6. Deconvolve with OASIS (tau=0.7, fs=15.5 Hz)
7. Filter interneurons (speed correlation > 0.5)
8. Multi-plane recordings are concatenated across planes

ii.
```python
def compute_dff_per_trial(F, Fneu, trial_starts, trial_ends, frame_rate=FRAME_RATE):
    f_[:, nanmask] = f_[:, nanmask] - NEUROPIL_COEF * f_neu_[:, nanmask]
    ...
    baseline = ndimage.minimum_filter1d(f_smooth, window_size, axis=-1)
    baseline = ndimage.maximum_filter1d(baseline, window_size, axis=-1)
    dff[:, start:end] = (f_[:, start:end] - baseline) / np.abs(baseline)
    ...
    dff[:, start:end] = nansmooth(dff[:, start:end], 2, axis=1)
...
def deconvolve_oasis(dff_trial, tau=TAU, frame_rate=FRAME_RATE):
    events = oasis(dff_trial, 2000, tau, frame_rate)
```

iii. The AI states this matches `preprocessing.py` from the reference code repository. The CONVERSION_NOTES document the full pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filtering steps: (1) `iscell[:, 0] == 1` from suite2p curation, and (2) interneuron filtering by removing neurons with speed correlation > 0.5.

ii.
```python
iscell = nwb_data['iscell'][:, 0].astype(bool)
...
def filter_interneurons(dff, speed, iscell_mask):
    for idx in cell_indices:
        corr = np.corrcoef(cell_data, speed_data)[0, 1]
        if corr > INTERNEURON_SPEED_CORR_THR:
            mask[idx] = False
    return mask
```

iii. The AI references the paper's criterion for interneuron filtering. The CONVERSION_NOTES acknowledge a discrepancy: ~3.8% neurons removed vs paper's 0.42%.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start by slicing the neural array using trial start and end indices. The time from trial start is computed from the timestamps.

ii.
```python
neural_trial = events[:, s:e].copy()
...
time_from_start = (timestamps[s:e] - timestamps[s])
```

iii. Since the instructions specify alignment to start of trial, slicing from the trial start index inherently provides this alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is set to ~64.48 ms (1000 / 15.5078125 Hz). No temporal rebinning is applied; the data is kept at the native imaging frame rate.

ii.
```python
FRAME_RATE = 15.5078125  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE  # ~64.5 ms
```

iii. The frame rate matches the NWB file metadata. For multi-plane recordings, the AI uses the same FRAME_RATE constant.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `timestamps` array of the `position` behavioral time series.

ii.
```python
'timestamps': np.array(bts.time_series['position'].timestamps[:]),
...
time_from_start = (timestamps[s:e] - timestamps[s])
```

iii. The timestamps provide the actual time in seconds for each sample.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp in the trial is subtracted from all timestamps in the trial to get time relative to trial start.

ii.
```python
time_from_start = (timestamps[s:e] - timestamps[s])
...
input_trial[0, :] = time_from_start
```

iii. Standard approach to compute relative time.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same time indices within each trial, so no additional alignment is needed. Length mismatches between neural and behavioral data are handled by truncating to the minimum.

ii.
```python
n_behav = len(speed)
n_neural = F.shape[1]
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    F = F[:, :min_len]
    ...
    nwb_data['timestamps'] = nwb_data['timestamps'][:min_len]
```

iii. The data in the NWB files are sampled at the same rate for neural and behavioral streams.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavioral time series in the NWB file.

ii.
```python
'environment': np.array(bts.time_series['environment'].data[:]),
...
env_type = float(environment[s])  # per trial
```

iii. The environment variable is 0 or 1, corresponding to ENV1/ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The value at the trial start index is taken as the per-trial environment type. If the value is negative, it defaults to 0 (ENV1). The value is broadcast across all timepoints.

ii.
```python
env_type = float(environment[s])  # per trial
if env_type < 0:
    env_type = 0.0  # default to ENV1 if unknown
...
input_trial[1, :] = env_type
```

iii. Environment is constant within a trial, so taking the value at trial start is sufficient.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the loop index `t` over trials within a session. It is not derived from the NWB `trial number` variable.

ii.
```python
for t in range(len(trial_starts)):
    ...
    trial_num = float(t)
    ...
    input_trial[2, :] = trial_num
```

iii. The sequential trial index within the session is used.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond assigning the loop index. The value is broadcast across all timepoints within a trial.

ii.
```python
trial_num = float(t)
input_trial[2, :] = trial_num
```

iii. Simple sequential numbering.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps and the `reward_zone` behavioral time series. A trial is considered rewarded if reward timestamps fall within the trial AND the mouse was in the reward zone.

ii.
```python
def determine_trial_rewarded(position, rz_signal, reward_timestamps, timestamps,
                              trial_start, trial_end):
    t_start = timestamps[trial_start]
    t_end = timestamps[min(trial_end, len(timestamps) - 1)]
    reward_in_trial = np.any(
        (reward_timestamps >= t_start) & (reward_timestamps <= t_end)
    )
    rz_trial = rz_signal[trial_start:trial_end]
    entered_rz = np.any(rz_trial > 0)
    return int(reward_in_trial and entered_rz)
```

iii. The AI combines both reward delivery and reward zone entry to determine if a trial was rewarded.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the previous trial's reward outcome is used. For the first trial, the value is set to 1.0 (assumes rewarded before session start).

ii.
```python
if t == 0:
    prev_outcome = 1.0  # assume rewarded before session
else:
    prev_outcome = float(is_rewarded[t - 1])
...
input_trial[3, :] = prev_outcome
```

iii. The AI's CONVERSION_NOTES don't discuss the choice of 1.0 for the first trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavioral time series and the reward zone boundaries defined in `REWARD_ZONES` dictionary. The reward zone label for each trial is determined from the `reward_zone` behavioral time series.

ii.
```python
REWARD_ZONES = {
    'A': [80, 130],
    'B': [200, 250],
    'C': [320, 370],
}
...
def distance_to_reward_zone(position, rz_start, rz_end):
    if position < rz_start:
        return position - rz_start
    elif position > rz_end:
        return position - rz_end
    else:
        return 0.0
```

iii. The reward zone boundaries are from the reference code `behavior.py`.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the signed distance from the animal's position to the nearest edge of the active reward zone is computed. Negative = before zone, 0 = inside zone, positive = past zone. This is done per-element in a loop.

ii.
```python
dist_to_rz = np.array([distance_to_reward_zone(p, rz_start, rz_end)
                       for p in pos_trial])
```

iii. Standard signed distance computation.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized using explicit if/elif comparisons into 7 bins matching the instructions.

ii.
```python
def discretize_distance(distances):
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
    return result
```

iii. The bin edges match the instruction specifications.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position data and neural data share the same time indices within each trial, so the same trial slice is used.

ii.
```python
pos_trial = position[s:e]
dist_to_rz = np.array([distance_to_reward_zone(p, rz_start, rz_end) for p in pos_trial])
```

iii. Same indexing scheme ensures alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series.

ii.
```python
'position': np.array(bts.time_series['position'].data[:]),
...
pos_trial = position[s:e]
```

iii. The `position` variable records the animal's position in cm along the VR corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 449.999] cm and then discretized into 5 bins of 90 cm each (TRACK_LENGTH / 5 = 450 / 5 = 90).

ii.
```python
def discretize_position(positions, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins  # 90 cm bins
    result = np.clip(np.floor(positions / bin_size).astype(int), 0, n_bins - 1)
    return result
...
pos_clipped = np.clip(pos_trial, 0, TRACK_LENGTH - 0.001)
pos_binned = discretize_position(pos_clipped, n_bins=5)
```

iii. The AI uses 90 cm bins over [0, 450] cm, producing bins [0-90), [90-180), [180-270), [270-360), [360-450].

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized using `np.floor(positions / 90)` clipped to [0, 4], producing 5 bins of 90 cm each over [0, 450] cm.

ii.
```python
bin_size = TRACK_LENGTH / n_bins  # 90 cm
result = np.clip(np.floor(positions / bin_size).astype(int), 0, n_bins - 1)
```

iii. The output_values label these as '0-90cm', '90-180cm', '180-270cm', '270-360cm', '360-450cm'.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data within each trial.

ii.
```python
pos_trial = position[s:e]
```

iii. Same indexing ensures alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series in the NWB file.

ii.
```python
'lick': np.array(bts.time_series['lick'].data[:]),
...
lick_trial = lick_cumul[s:e].copy()
lick_binary = (lick_trial > 0).astype(float)
```

iii. The lick variable from the NWB file.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI binarizes the lick signal: any value > 0 maps to 1, otherwise 0. The CONVERSION_NOTES describe converting "cumulative lick count to binary per-frame lick signal" using diff, but the actual code simply thresholds at > 0 without computing diff.

ii.
```python
lick_binary = (lick_trial > 0).astype(float)
...
output_trial[3, :] = lick_binary.astype(int)
```

iii. The code thresholds directly at > 0. The CONVERSION_NOTES mention computing a diff but this is not reflected in the final code.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data within each trial.

ii.
```python
lick_trial = lick_cumul[s:e].copy()
```

iii. Same indexing ensures alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` behavioral time series and the `position` time series. The mean position where `reward_zone > 0` is computed and matched to the closest reward zone center (A, B, or C).

ii.
```python
def determine_reward_zone_label(position, rz_signal, trial_start, trial_end):
    rz_positions = pos_trial[rz_trial > 0]
    mean_rz_pos = np.mean(rz_positions)
    for zone, (start, end) in REWARD_ZONES.items():
        zone_center = (start + end) / 2
        dist = abs(mean_rz_pos - zone_center)
        if dist < best_dist:
            best_dist = dist
            best_zone = zone
    return best_zone
```

iii. The AI uses mean reward zone position to match to the closest defined zone.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Two-pass approach: (1) determine zone from mean position where `reward_zone > 0` for each trial, (2) for trials where the mouse didn't enter the zone, fill from neighboring trials (first forward, then backward).

ii.
```python
def determine_reward_zone_for_all_trials(position, rz_signal, trial_starts, trial_ends):
    for i in range(n_trials):
        labels[i] = determine_reward_zone_label(...)
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
    return labels
```

iii. The AI infers missing reward zone labels from neighboring trials. The reference uses a Viterbi algorithm for more robust assignment.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` timestamps and the `reward_zone` behavioral time series.

ii.
```python
'reward_timestamps': np.array(bts.time_series['Reward'].timestamps[:]),
...
def determine_trial_rewarded(position, rz_signal, reward_timestamps, timestamps,
                              trial_start, trial_end):
    reward_in_trial = np.any(
        (reward_timestamps >= t_start) & (reward_timestamps <= t_end)
    )
    entered_rz = np.any(rz_trial > 0)
    return int(reward_in_trial and entered_rz)
```

iii. The AI requires both a reward timestamp within the trial AND the mouse entering the reward zone.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is marked as rewarded (1) if both conditions are met: (1) a reward timestamp falls within the trial's time window, and (2) the reward zone signal is active at some point during the trial. Otherwise, the outcome is 0.

ii.
```python
reward_in_trial = np.any(
    (reward_timestamps >= t_start) & (reward_timestamps <= t_end)
)
rz_trial = rz_signal[trial_start:trial_end]
entered_rz = np.any(rz_trial > 0)
return int(reward_in_trial and entered_rz)
```

iii. The dual condition is more restrictive than simply checking for reward timestamps.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Neural/behavior length mismatch**: Truncated to minimum of the two lengths.
- **Short trials**: Trials with fewer than 5 timepoints are skipped.
- **Missing reward zone**: Trials where reward zone can't be determined are filled from neighboring trials; if still None, defaults to 'A'.
- **NaN neural data**: NaN values in neural data are replaced with 0 (`np.nan_to_num`).
- **Low cell count**: Sessions with fewer than 5 curated cells are skipped.
- **Negative environment**: Defaults to ENV1 (0).

ii.
```python
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    ...
if n_timepoints < 5:
    continue
...
if rz_label is None:
    rz_label = 'A'  # fallback
...
neural_trial = np.nan_to_num(neural_trial, nan=0.0)
```

iii. These are defensive checks for edge cases in the data.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files** (I/O bound, reading large neural arrays)
2. **dF/F computation** (neuropil subtraction, baseline, smoothing for all neurons across all trials)
3. **OASIS deconvolution** per trial per session
4. **Interneuron filtering** (computing correlations for all neurons)

ii. N/A

iii. The AI's approach of recomputing dF/F from raw fluorescence adds significant processing time compared to using pre-computed deconvolved data.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
1. `distance_to_reward_zone()` iterates per-element instead of vectorized array operations
2. `discretize_distance()` and `discretize_speed()` use per-element if/elif instead of `np.digitize` or `np.searchsorted`
3. The interneuron filtering loop computes correlations one neuron at a time
4. The per-trial neural processing loop could partially be vectorized

ii.
```python
dist_to_rz = np.array([distance_to_reward_zone(p, rz_start, rz_end) for p in pos_trial])
...
for i, d in enumerate(distances):
    if d < -50:
        result[i] = 0
    ...
```

iii. List comprehensions and per-element loops are used where vectorized numpy operations would be more efficient.

## 13-c. What processing does the code repeat multiple times?

i. No explicit survey/conversion separation like the reference code. However, the full dF/F pipeline is computed from scratch for every session, including intermediate steps (fluorescence masking, smoothing, baseline) that could potentially be cached.

ii. N/A

iii. Each session is processed independently in a single pass.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes its own dF/F from raw fluorescence and then deconvolves with OASIS, when pre-computed deconvolved data is already available in the NWB files. This entire pipeline (neuropil subtraction, baseline computation, smoothing, deconvolution) is unnecessary if the pre-computed data matches the reference processing. Additionally, the interneuron filtering step removes neurons that the reference solution does not remove.

ii.
```python
if compute_own_dff:
    dff = compute_dff_per_trial(F, Fneu, trial_starts, trial_ends, ...)
    ...
    events_trial = deconvolve_oasis(trial_dff_clean, ...)
```

iii. The pre-computed `Deconvolved` data in the NWB files is the output of the same pipeline the AI is reimplementing.
