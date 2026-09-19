# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Subjects are discovered from all subdirectories of the `data` directory that start with `sub-`. All `.nwb` files within each subject directory are gathered, each corresponding to a session. Data are loaded using `pynwb.NWBHDF5IO`. All behavioral time series, fluorescence, neuropil, deconvolved data, and cell metadata are extracted.

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

iii. The agent explored the directory structure and NWB file contents. It found 11 subjects matching the paper and loaded all NWB files within each subject directory.

## 1-b. How are the data split into subjects?

i. Subjects correspond to subdirectories of `data` starting with `sub-`. The subject name is extracted by removing the `sub-` prefix.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
subject_name = subj_dir.replace('sub-', '')
```

iii. The agent noted 11 subject directories matching the paper's reported number of mice.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. All NWB files within each subject's directory are treated as separate sessions.

ii.
```python
nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
for nwb_file in nwb_files:
    result = process_session(nwb_file, compute_own_dff=True)
```

iii. The agent identified each NWB file as a separate session based on the file naming convention.

## 1-d. How are the data split into trials?

i. Trial starts are identified from the `trial_start` signal (where it is > 0). Trial ends are identified from the `teleport` signal (where it is > 0). For each start, the next teleport event after it is found and paired as the trial end.

ii.
```python
def get_trial_boundaries(trial_start_signal, teleport_signal, trial_numbers, scanning):
    start_indices = np.where(trial_start_signal > 0)[0]
    end_indices = np.where(teleport_signal > 0)[0]
    for s in start_indices:
        next_ends = end_indices[end_indices > s]
        if len(next_ends) > 0:
            e = next_ends[0]
            if np.any(scanning[s:e] == 1):
                trial_starts.append(s)
                trial_ends.append(e)
```

iii. The agent explored the behavioral signals and decided to use `trial_start` and `teleport` to define trial boundaries. It also added a scanning check (scanning == 1).

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 5 timepoints are skipped. Sessions with fewer than 3 trials or fewer than 5 curated cells are skipped entirely. Sessions with fewer than 2 valid trials after per-trial filtering are also skipped.

ii.
```python
if n_timepoints < 5:
    continue
...
if n_total_cells < 5:
    print(f"    Skipping: only {n_total_cells} curated cells")
    return None
...
if len(trial_starts) < 3:
    print(f"    Skipping: only {len(trial_starts)} trials")
    return None
...
if valid_trial_count < 2:
    print(f"    Skipping: only {valid_trial_count} valid trials")
    return None
```

iii. The agent applied multiple filtering criteria for quality. The minimum trial length of 5 is very short compared to the reference's 50.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI's code loads `Fluorescence`, `Neuropil`, and `Deconvolved` data from the NWB file. When `compute_own_dff=True` (the default), it computes dF/F from `Fluorescence` and `Neuropil`, then deconvolves. The `Deconvolved` NWB data is loaded but only used as a fallback.

ii.
```python
fluorescence = np.array(ophys['Fluorescence']['plane0'].data[:])
neuropil_data = np.array(ophys['Neuropil']['plane0'].data[:])
deconvolved = np.array(ophys['Deconvolved']['plane0'].data[:])
...
F = nwb_data['fluorescence'].T
Fneu = nwb_data['neuropil'].T
...
dff = compute_dff_per_trial(F, Fneu, trial_starts, trial_ends, frame_rate=effective_frame_rate)
```

iii. The agent recognized that the paper computes its own dF/F from raw fluorescence rather than using the NWB's precomputed deconvolved data.

## 2-b. How is the `neural` data processed?

i. The AI implemented its own dF/F pipeline: (1) mask fluorescence to within-trial data only, (2) neuropil subtraction with coefficient 0.7, (3) add back per-trial neuropil mean, (4) Gaussian smooth with sigma=15 for baseline, (5) maximin baseline with 300-sample window, (6) dF/F = (F - baseline) / |baseline|, (7) smooth dF/F with 2-sample Gaussian, (8) deconvolve with OASIS.

However, there are differences from the reference implementation: the smoothing for baseline uses `gaussian_filter1d` (1D) instead of the paper's `nansmooth` with `[0, 15]` (2D smoothing with sigma 0 on neuron axis, 15 on time). The `keep_teleports` logic is entirely absent -- the AI always restricts the baseline window to within-trial data regardless of session. The deconvolution uses `effective_frame_rate = FRAME_RATE` (the scanner rate) rather than `frame_rate/n_planes` for multi-plane sessions.

ii.
```python
def compute_dff_per_trial(F, Fneu, trial_starts, trial_ends, frame_rate=FRAME_RATE):
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
...
events_trial = deconvolve_oasis(trial_dff_clean, frame_rate=effective_frame_rate)
...
effective_frame_rate = FRAME_RATE  # already per-plane in NWB data
```

iii. The agent attempted to replicate the paper's preprocessing pipeline. It recognized neuropil subtraction, maximin baseline, and OASIS deconvolution. However, it did not copy the paper's `dff()` function verbatim and wrote its own version with subtle differences. The `keep_teleports` metadata from `teleport_metadata.py` was not incorporated. The agent reasoned that `FRAME_RATE` was "already per-plane in NWB data", which is incorrect -- the NWB stores the scanner rate.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) cells are restricted to those with `iscell[:, 0] == 1` (suite2p curated cells), (2) putative interneurons are removed based on speed correlation > 0.5 computed on the dF/F signal.

ii.
```python
iscell = nwb_data['iscell'][:, 0].astype(bool)
...
cell_mask = filter_interneurons(dff, speed, iscell)
...
def filter_interneurons(dff, speed, iscell_mask):
    for idx in cell_indices:
        corr = np.corrcoef(cell_data, speed_data)[0, 1]
        if corr > INTERNEURON_SPEED_CORR_THR:
            mask[idx] = False
```

iii. The agent recognized both filtering steps from the paper's methods: iscell curation and interneuron filtering by speed correlation > 0.5.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The data is aligned to trial start. No additional processing is needed beyond splitting into trials, since data is indexed from the trial start.

ii.
```python
neural_trial = events[:, s:e].copy()
```

iii. The instructions specify alignment to start of trial, which is the natural boundary.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data is kept at the stored recording rate with no rebinning. The time bin size is computed as `1000.0 / FRAME_RATE` where `FRAME_RATE = 15.5078125 Hz`, giving ~64.5 ms. This is the scanner rate, not the per-plane rate for multi-plane recordings.

ii.
```python
FRAME_RATE = 15.5078125  # Hz, from NWB files
TIME_BIN_MS = 1000.0 / FRAME_RATE  # ~64.5 ms
...
'time_bin_size': TIME_BIN_MS,
```

iii. The agent used the frame rate from the NWB files directly. For multi-plane sessions (m17, m18), the per-plane rate would be FRAME_RATE/n_planes (~7.75 Hz), but the stored time bin size uses the scanner rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `timestamps` of the `position` behavior time series.

ii.
```python
'timestamps': np.array(bts.time_series['position'].timestamps[:]),
...
time_from_start = (timestamps[s:e] - timestamps[s])
```

iii. The agent used position timestamps as the time reference.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The initial timestamp for the trial is subtracted to get time from trial start.

ii.
```python
time_from_start = (timestamps[s:e] - timestamps[s])
```

iii. Standard approach to compute relative time.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same time indices (same sampling rate), verified by length matching. If lengths differ, both are cropped to the minimum.

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

iii. The agent found and handled neural/behavioral length mismatches by truncation.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
'environment': np.array(bts.time_series['environment'].data[:]),
...
env_type = float(environment[s])  # per trial
```

iii. The agent identified the environment variable from the behavioral data.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The environment value at the trial start is taken as a scalar. Negative values default to 0. The value is broadcast to all timepoints.

ii.
```python
env_type = float(environment[s])  # per trial
if env_type < 0:
    env_type = 0.0  # default to ENV1 if unknown
...
input_trial[1, :] = env_type
```

iii. The agent treated it as a per-trial constant derived from the first timepoint.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the loop counter `t` over trials within the session.

ii.
```python
for t in range(len(trial_starts)):
    ...
    trial_num = float(t)
    ...
    input_trial[2, :] = trial_num
```

iii. The agent used a sequential index rather than the stored `trial number` variable.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond assigning the loop index. The value is constant across all timepoints within a trial.

ii.
```python
trial_num = float(t)
input_trial[2, :] = trial_num
```

iii. Simple sequential indexing.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps and the `reward_zone` signal. The reward outcome for each trial is determined by checking if a reward was delivered AND the mouse entered the reward zone.

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

iii. The agent used both reward delivery AND reward zone entry to determine if a trial was rewarded.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the reward outcome of the previous trial is used. For the first trial (t==0), the previous outcome is set to 1 (assumes rewarded before session).

ii.
```python
if t == 0:
    prev_outcome = 1.0  # assume rewarded before session
else:
    prev_outcome = float(is_rewarded[t - 1])
...
input_trial[3, :] = prev_outcome
```

iii. The agent chose to default to 1 for the first trial's previous outcome, which differs from the reference solution's choice of 0.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the `reward_zone` behavior time series. The reward zone label is determined per trial by computing the mean position where `reward_zone > 0` and matching to the nearest zone center.

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
```

iii. The agent used mean reward zone position matching rather than the Viterbi algorithm used by the reference. Trials without reward zone entry are filled from neighboring trials.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, signed distance from position to the reward zone boundaries is computed: negative if before the zone, positive if after, 0 if inside. Then discretized into 7 bins.

ii.
```python
def distance_to_reward_zone(position, rz_start, rz_end):
    if position < rz_start:
        return position - rz_start  # negative
    elif position > rz_end:
        return position - rz_end  # positive
    else:
        return 0.0  # inside zone
...
dist_to_rz = np.array([distance_to_reward_zone(p, rz_start, rz_end) for p in pos_trial])
```

iii. The distance computation follows the same logic as the reference.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized using an explicit if/elif chain with boundaries: < -50 -> 0, -50 to -10 -> 1, -10 to 0 -> 2, == 0 -> 3, 0 to 10 -> 4, 10 to 50 -> 5, > 50 -> 6.

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
```

iii. The bin edges match the instructions. The boundary handling differs slightly from the reference: the reference uses `np.digitize` with `[..., 0, 1e-6, ...]` to separate exactly 0 from positive values, while the AI uses `d == 0` and `d <= 10`. The bin at exactly -50 goes to bin 1 here (via `elif d < -10`) but to bin 1 in the reference as well. The boundary at exactly -10 goes to bin 2 here but to bin 2 in the reference too (since `np.digitize` with `-10` puts -10 in the [-10, 0) bin).

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same time indices as neural data within each trial.

ii.
```python
pos_trial = position[s:e]
dist_to_rz = np.array([distance_to_reward_zone(p, rz_start, rz_end) for p in pos_trial])
```

iii. Aligned by shared indexing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
'position': np.array(bts.time_series['position'].data[:]),
...
pos_trial = position[s:e]
```

iii. Direct from the VR position data.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 449.999] and then discretized using `floor(pos / 90)`, clipped to [0, 4].

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

iii. The agent used floor division for discretization rather than `np.digitize`.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal bins of 90 cm each via `floor(position / 90)`, clipped to [0, 4]: 0-90 -> 0, 90-180 -> 1, 180-270 -> 2, 270-360 -> 3, 360-450 -> 4.

ii.
```python
pos_clipped = np.clip(pos_trial, 0, TRACK_LENGTH - 0.001)
pos_binned = discretize_position(pos_clipped, n_bins=5)
```

iii. Matches the instructions' 5 equal-sized bins spanning 450 cm.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data within each trial.

ii.
```python
pos_trial = position[s:e]
```

iii. Aligned by shared indexing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
'lick': np.array(bts.time_series['lick'].data[:]),
...
lick_trial = lick_cumul[s:e].copy()
lick_binary = (lick_trial > 0).astype(float)
```

iii. The agent identified the lick variable from behavioral data.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
lick_binary = (lick_trial > 0).astype(float)
output_trial[3, :] = lick_binary.astype(int)
```

iii. Same binarization as the reference.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data within each trial.

ii.
```python
lick_trial = lick_cumul[s:e].copy()
```

iii. Aligned by shared indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` behavior time series and `position` time series. The reward zone label is determined per trial by computing the mean position where `reward_zone > 0` and matching to the closest zone center (A, B, or C). For trials without reward zone entry, labels are propagated from neighboring trials.

ii.
```python
def determine_reward_zone_label(position, rz_signal, trial_start, trial_end):
    rz_positions = pos_trial[rz_trial > 0]
    mean_rz_pos = np.mean(rz_positions)
    for zone, (start, end) in REWARD_ZONES.items():
        zone_center = (start + end) / 2
        dist = abs(mean_rz_pos - zone_center)
        ...
...
def determine_reward_zone_for_all_trials(position, rz_signal, trial_starts, trial_ends):
    # First pass: determine from direct observation
    # Second pass: fill in missing labels from neighbors
```

iii. The agent used a simpler nearest-center approach rather than the reference's Viterbi algorithm.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Mean of positions where `reward_zone > 0` is matched to nearest zone center. Missing trials are filled from forward/backward neighbors. Encoded as A=0, B=1, C=2.

ii.
```python
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_label]
output_trial[4, :] = rz_loc
```

iii. Simple encoding consistent with the instructions.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` timestamps, `reward_zone` signal, `position`, and `timestamps`.

ii.
```python
def determine_trial_rewarded(position, rz_signal, reward_timestamps, timestamps,
                              trial_start, trial_end):
    t_start = timestamps[trial_start]
    t_end = timestamps[min(trial_end, len(timestamps) - 1)]
    reward_in_trial = np.any(
        (reward_timestamps >= t_start) & (reward_timestamps <= t_end)
    )
    entered_rz = np.any(rz_trial > 0)
    return int(reward_in_trial and entered_rz)
```

iii. The agent checks both reward delivery and reward zone entry, which adds an extra condition not in the reference.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is marked rewarded (1) if both a reward timestamp falls within the trial's time range AND the mouse entered the reward zone (`reward_zone > 0`). Otherwise 0. The value is constant across all timepoints in the trial.

ii.
```python
return int(reward_in_trial and entered_rz)
...
output_trial[5, :] = rew_outcome
```

iii. The additional `entered_rz` check is redundant -- rewards are only delivered in the reward zone -- but shouldn't change the outcome.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Neural/behavior length mismatch**: Both are cropped to the minimum length.
- **Short trials**: Trials with fewer than 5 timepoints are skipped.
- **Missing reward zone labels**: Filled from neighboring trials (forward/backward propagation).
- **NaN in neural data**: Replaced with 0 for decoder input.
- **Negative environment values**: Default to 0.

ii.
```python
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    F = F[:, :min_len]
    ...
...
if n_timepoints < 5:
    continue
...
neural_trial = np.nan_to_num(neural_trial, nan=0.0)
```

iii. The agent implemented defensive handling for data mismatches discovered during development.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files** -- each file is large and loaded via `pynwb`.
2. **Computing dF/F per trial** -- involves smoothing, baseline, and filtering for every neuron.
3. **OASIS deconvolution** -- run per trial for all neurons.
4. **Saving the pickle file** -- large dataset.

ii. N/A

iii. The NWB files contain full neural recordings and behavioral data.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops:
1. `distance_to_reward_zone()` uses a Python for-loop over each position sample (line 596).
2. `discretize_distance()` uses a Python for-loop over each distance value (lines 361-376).
3. `discretize_speed()` uses a Python for-loop over each speed value (lines 398-409).
4. `filter_interneurons()` loops over cells individually (lines 214-219).

ii.
```python
dist_to_rz = np.array([distance_to_reward_zone(p, rz_start, rz_end) for p in pos_trial])
...
for i, d in enumerate(distances):
    if d < -50: ...
...
for i, s in enumerate(speeds):
    if s < 2: ...
```

iii. These could use vectorized numpy operations like `np.digitize` or vectorized comparisons.

## 13-c. What processing does the code repeat multiple times?

i. No explicit survey/conversion separation. Each NWB file is loaded once. However, the `load_nwb()` function loads all data including `Deconvolved` which is not used when `compute_own_dff=True`.

ii. N/A

iii. Loading unused data is wasteful but not a repeated computation.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads the NWB `Deconvolved` data for every session even though it's never used (when `compute_own_dff=True`). The `run_sanity_checks()` function runs after conversion but its output is only printed, not saved. The `sample_data.pkl` creation is unnecessary overhead.

ii.
```python
deconvolved = np.array(ophys['Deconvolved']['plane0'].data[:])
```

iii. Loading the deconvolved data wastes memory and time.
