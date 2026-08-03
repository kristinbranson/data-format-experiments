# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses the AllenSDK's `VisualBehaviorOphysProjectCache.from_s3_cache()` to create a cache object, then loads the experiment metadata table from a CSV file (`ophys_experiment_table.csv`). It filters this table to select active behavior, familiar image set experiments. For each experiment, it calls `cache.get_behavior_ophys_experiment(ophys_experiment_id)` which returns a `BehaviorOphysExperiment` object providing access to neural data, stimulus presentations, trials, running speed, and eye tracking. Available NWB files are discovered by listing the directory contents.

ii.
```python
cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=data_dir)
exp_table = get_experiment_table(data_dir)

# In get_experiment_table:
exp_table = pd.read_csv(
    os.path.join(data_dir, 'visual-behavior-ophys-1.1.0/project_metadata/ophys_experiment_table.csv')
)
nwb_dir = os.path.join(data_dir, 'visual-behavior-ophys-1.1.0/behavior_ophys_experiments/')
nwb_files = os.listdir(nwb_dir)
exp_ids = [int(re.search(r'(\d+)', f).group(1)) for f in nwb_files]
available = exp_table[exp_table['ophys_experiment_id'].isin(exp_ids)]
filtered = available[
    (available['behavior_type'] == 'active_behavior') &
    (available['experience_level'] == 'Familiar')
].copy()

# Per experiment:
ds = cache.get_behavior_ophys_experiment(exp_id)
```

iii. The AI's CONVERSION_NOTES.md states: "Active behavior only: Excluded passive viewing sessions... Familiar images only: Following the paper's neural analysis approach... All equipment types: Included both Scientifica single-plane and Multiscope multi-plane recordings... All cre lines: Included Slc17a7 (excitatory), Sst, and Vip cell types."

## 1-b. How are the data split into subjects (mice)?

i. The AI extracts `mouse_id` from the experiment table metadata. Unique mouse IDs are collected from all successfully processed experiments and stored as a sorted list of strings. Each session (experiment) is associated with its mouse via `subject_idx`.

ii.
```python
all_mice = sorted(set(r['exp_info']['mouse_id'] for r in experiment_results))
mouse_to_idx = {m: i for i, m in enumerate(all_mice)}
subjects = [str(m) for m in all_mice]
# Per experiment:
subject_idx_list.append(mouse_to_idx[exp_info['mouse_id']])
```

iii. The conversion output shows 38 unique mice across 110 experiments.

## 1-c. How are the data split into sessions?

i. The AI treats each **ophys_experiment_id** (imaging plane) as a separate "session" in the output data structure. This means that Multiscope sessions with multiple imaging planes (up to 7 per session) are split into multiple "sessions" with identical behavioral data but different neural populations. The 92 unique ophys_sessions become 110 "sessions" in the output.

ii.
```python
for i, (_, exp_info) in enumerate(exp_table.iterrows()):
    exp_id = int(exp_info['ophys_experiment_id'])
    # ... process each experiment as a separate session
    experiment_results.append({
        'exp_info': exp_info,
        'neural_trials': neural_trials,
        ...
    })
# Each experiment_result becomes one session in neural_all, output_all, etc.
```

iii. CONVERSION_NOTES.md states "110 sessions from 38 mice." The AI does not distinguish between ophys_session_id (the actual recording session) and ophys_experiment_id (individual imaging plane). For Multiscope data, multiple experiments share the same session, meaning behavioral data (running speed, pupil, trial outcomes, stimuli) are duplicated across these "sessions."

## 1-d. How are the data split into trials?

i. The AI uses the `ds.trials` table from the SDK and selects Go and Catch trials, excluding Aborted and Auto-rewarded trials. Each trial is defined by its `start_time` and `stop_time` from the trials table. Neural and behavioral data are sliced using ophys timestamps that fall within `[start_time, stop_time]`.

ii.
```python
trials = ds.trials
valid_trials = trials[
    ((trials['go'] == True) | (trials['catch'] == True)) &
    (trials['aborted'] == False) &
    (trials['auto_rewarded'] == False)
]
# Per trial:
t_start = trial['start_time']
t_stop = trial['stop_time']
frame_mask = (ophys_ts >= t_start) & (ophys_ts <= t_stop)
```

iii. CONVERSION_NOTES.md: "Go trials: Trials where the image changed... Catch trials: Trials where no change occurred... Excluded: Aborted trials (premature licking before change), Auto-rewarded trials (free rewards). Minimum trials: Sessions with fewer than 2 valid trials were excluded."

## 1-e. How are trials filtered based on quality controls?

i. The AI applies minimal trial-level quality filtering: (1) only Go and Catch trials are included; (2) trials with fewer than `ds_factor` ophys frames are dropped; (3) sessions with fewer than 2 usable trials after processing are skipped entirely. No additional quality filtering is applied (e.g., no engagement filtering, no exclusion based on reward rate or lick rate).

ii.
```python
if len(frame_indices) < ds_factor:
    continue
# ...
if len(neural_trials) < 2:
    return None, None, None, None, None, f"Only {len(neural_trials)} usable trials after processing"
```

iii. CONVERSION_NOTES.md: "Minimum trials: Sessions with fewer than 2 valid trials were excluded." The notes also state the high miss rate (~59%) is "consistent with including all sessions regardless of engagement level," indicating no engagement-based filtering was applied.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `ds.events`, which contains detected calcium events (deconvolved from dF/F fluorescence traces). The events are accessed via the `events` property of the `BehaviorOphysExperiment` object.

ii.
```python
events = ds.events
neural_full = np.vstack(events['events'].values).astype(np.float32)
```

iii. CONVERSION_NOTES.md: "Used detected calcium events (not raw dF/F), matching the paper: 'we used the detected calcium events as described in Garrett et al.'"

## 2-b. How is the `neural` data processed?

i. The full neural activity matrix is extracted as (n_neurons, n_total_timepoints). For each trial, the relevant time-slice is extracted. For single-plane data (~31 Hz), the neural data is downsampled by a factor of 3 via averaging to match the Multiscope rate (~10.7 Hz). The final data is stored as float32.

ii.
```python
neural_full = np.vstack(events['events'].values).astype(np.float32)
# Per trial:
neural_trial = neural_full[:, frame_indices]
# Downsample if needed:
if ds_factor > 1:
    neural_trial = downsample_by_factor(neural_trial, ds_factor)
# downsample_by_factor averages along the time axis:
return data[:, :n_new * factor].reshape(data.shape[0], n_new, factor).mean(axis=2)
```

iii. CONVERSION_NOTES.md: "Neural data: averaged over 3 frames for downsampling."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI relies entirely on the AllenSDK's pre-filtering of ROIs. No additional neuron-level quality filtering is applied (no SNR thresholds, no activity-based filtering, no cross-session matching). Sessions with 0 neurons are skipped.

ii.
```python
n_neurons = neural_full.shape[0]
if n_neurons == 0:
    return None, None, None, None, None, "No neurons"
```

iii. CONVERSION_NOTES.md: "ROIs are pre-filtered by the AllenSDK processing pipeline (see whitepaper Section F: ROI Filtering). Only valid cell bodies are included (excludes dendrites, duplicates, edge artifacts, unions)."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to ophys timestamps. Each trial's neural data is extracted by selecting ophys frames that fall within the trial's `[start_time, stop_time]` interval. The alignment event is "Trial start" per the metadata. There is no alignment to a specific within-trial event (like the change time); the trial is simply windowed by its start and stop times.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts <= t_stop)
frame_indices = np.where(frame_mask)[0]
neural_trial = neural_full[:, frame_indices]
```

iii. Metadata: `'temporal_alignment_event': 'Trial start (aligned to ophys timestamps)'`, `'off_start': 0.0` (trial starts at trial start_time), `'off_end': None` (variable trial length).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The target time bin size is 93.2 ms (~10.7 Hz), matching the Multiscope frame rate. Single-plane Scientifica data (~31 Hz, ~32.3 ms per frame) is downsampled by a factor of 3 via averaging. Multiscope data at ~10.7 Hz is kept at its native resolution (ds_factor=1).

ii.
```python
target_dt = 0.0932  # seconds
dt = np.median(np.diff(ophys_ts))
ds_factor = max(1, int(round(target_dt / dt)))
actual_dt = dt * ds_factor
```

iii. CONVERSION_NOTES.md: "All data resampled to 93.2 ms bins (~10.7 Hz), matching the Multiscope frame rate. Single-plane Scientifica data (~31 Hz, ~32.3 ms bins) downsampled by factor of 3 via averaging."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `ds.stimulus_presentations`, specifically the `image_name` column. The stimulus presentations are filtered to the `change_detection` stimulus block, and omitted stimuli are excluded when tracking image identity.

ii.
```python
stim = ds.stimulus_presentations
stim_cd = stim[stim['stimulus_block_name'].str.contains('change_detection', na=False)].copy()
stim_images = stim_cd[stim_cd['image_name'] != 'omitted']
# Per trial:
trial_stim_no_omit = trial_stim[trial_stim['image_name'] != 'omitted']
image_idx = get_image_at_timepoints(trial_stim_no_omit, trial_ophys_ts, image_to_idx)
```

iii. CONVERSION_NOTES.md: "8 categories corresponding to the 8 natural scene images... During gray screen intervals, the last shown image identity persists. Omitted stimuli retain the previous image identity."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each ophys timepoint, the code finds the most recent non-omitted stimulus presentation and assigns its image index. The code iterates through timepoints sequentially, advancing a stimulus index pointer. Images are mapped to indices 0-7 using a sorted list of 8 image names. For single-plane data, categorical downsampling uses mode (most frequent value) in each bin of 3.

ii.
```python
def get_image_at_timepoints(stim_presentations, ophys_timestamps, image_to_idx):
    image_indices = np.zeros(len(ophys_timestamps), dtype=np.int64)
    stim_starts = stim_presentations['start_time'].values
    current_image_idx = -1
    stim_idx = 0
    for t_idx, t in enumerate(ophys_timestamps):
        while stim_idx < len(stim_starts) - 1 and stim_starts[stim_idx + 1] <= t:
            stim_idx += 1
        if stim_idx < len(stim_starts) and stim_starts[stim_idx] <= t:
            img = stim_images[stim_idx]
            if img in image_to_idx and img != 'omitted':
                current_image_idx = image_to_idx[img]
        if current_image_idx >= 0:
            image_indices[t_idx] = current_image_idx
    return image_indices
```

iii. CONVERSION_NOTES.md: "At each timepoint, assigned the most recently presented image."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at each ophys timepoint within the trial window, directly aligned with the neural data timestamps. For single-plane data, it is downsampled by taking the mode over each group of 3 frames, matching the neural data downsampling.

ii.
```python
image_idx = get_image_at_timepoints(trial_stim_no_omit, trial_ophys_ts, image_to_idx)
if ds_factor > 1:
    image_idx = downsample_categorical_by_factor(image_idx, ds_factor)
```

iii. No explicit justification beyond alignment to ophys timestamps.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in `ds.stimulus_presentations`. Only actual changes (where `is_change == True`) are used.

ii.
```python
changes = stim_presentations[stim_presentations['is_change'] == True]
```

iii. CONVERSION_NOTES.md: "Only actual changes are marked (not sham changes in catch trials)."

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary signal is created: value 1 is assigned to all ophys timepoints within a 750ms window starting at each change time, and 0 otherwise. The 750ms window covers the full stimulus flash period (250ms image + 500ms gray). For single-plane data, the binary signal is downsampled by averaging, then re-binarized (any value > 0 becomes 1).

ii.
```python
def get_image_change_signal(stim_presentations, ophys_timestamps):
    change_signal = np.zeros(len(ophys_timestamps), dtype=np.float32)
    changes = stim_presentations[stim_presentations['is_change'] == True]
    for _, change in changes.iterrows():
        change_time = change['start_time']
        mask = (ophys_timestamps >= change_time) & (ophys_timestamps < change_time + 0.75)
        change_signal[mask] = 1.0
    return change_signal
```

iii. CONVERSION_NOTES.md: "Binary signal: 1 during the 750ms following a change in image identity, 0 otherwise."

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1). The instructions specify it as a "binary variable." No additional thresholding is needed. After downsampling for single-plane data, re-binarization is applied.

ii.
```python
change_signal = (change_signal > 0.0).astype(np.float32)
```

iii. Instructions specify "binary variable. Have value of 1 right after a change in image identity, otherwise 0."

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The change signal is computed directly on ophys timestamps, so it is naturally aligned with the neural data. For single-plane data, it is downsampled the same way as neural data (average then re-binarize).

ii.
```python
change_signal = get_image_change_signal(trial_stim, trial_ophys_ts)
if ds_factor > 1:
    change_signal = downsample_by_factor(change_signal, ds_factor)
    change_signal = (change_signal > 0.0).astype(np.float32)
```

iii. No specific justification beyond using ophys timestamps.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ds.running_speed`, which provides filtered running speed values at the SDK's native timestamps (~60 Hz).

ii.
```python
rs = ds.running_speed
running_at_ophys = interpolate_to_ophys(
    rs['timestamps'].values, rs['speed'].values, ophys_ts
)
```

iii. CONVERSION_NOTES.md: "Uses the filtered running speed from the SDK (10 Hz lowpass Butterworth filtered)."

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from ~60 Hz to ophys timestamps. NaN values are replaced with 0.0. For single-plane data, running speed is downsampled by averaging groups of 3. Running speed is then discretized into 5 equal-percentile bins using global percentile boundaries computed across all sessions.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    rs['timestamps'].values, rs['speed'].values, ophys_ts
)
running_at_ophys = np.nan_to_num(running_at_ophys, nan=0.0)
all_running_speeds.extend(running_at_ophys.tolist())
# Later:
running_percentiles = np.percentile(running_arr, np.linspace(0, 100, n_bins + 1)[1:-1])
running_binned = discretize_to_percentile_bins(out_raw['running'], n_bins, running_percentiles)
```

iii. CONVERSION_NOTES.md: "Running speed: linearly interpolated from ~60 Hz analog input to ophys timestamps. Discretized into 5 equal-percentile bins using global percentile boundaries."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using equal-percentile boundaries computed across all sessions globally. The boundaries are the 20th, 40th, 60th, and 80th percentile values. `np.digitize` assigns each value to a bin, clipped to [0, 4].

ii.
```python
running_percentiles = np.percentile(running_arr, np.linspace(0, 100, n_bins + 1)[1:-1])
# running_percentiles ≈ [0.013, 1.53, 17.5, 33.9]
def discretize_to_percentile_bins(values, n_bins, percentiles):
    result = np.digitize(values, percentiles)
    result = np.clip(result, 0, n_bins - 1)
    return result.astype(np.int64)
```

iii. Instructions specify "discretized into five equal percentile bins." CONVERSION_NOTES.md: "Percentile boundaries: [0.013, 1.53, 17.5, 33.9] cm/s."

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to ophys timestamps, then sliced to the trial's frame indices, matching neural data exactly. For single-plane data, downsampled by the same factor as neural data.

ii.
```python
running_trial = running_at_ophys[frame_indices]
if ds_factor > 1:
    running_trial = downsample_by_factor(running_trial, ds_factor)
```

iii. Aligned via ophys timestamps.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `ds.eye_tracking`, specifically the `pupil_width` column.

ii.
```python
et = ds.eye_tracking
pupil_vals = et['pupil_width'].values
pupil_ts = et['timestamps'].values
```

iii. CONVERSION_NOTES.md: "Uses pupil_width from eye tracking data as the diameter measure."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil width values are linearly interpolated from ~30 Hz eye tracking timestamps to ophys timestamps. NaN values (from blinks/tracking failures) are filled by nearest valid value interpolation. For single-plane data, pupil values are downsampled by averaging groups of 3. Values are then discretized into 5 equal-percentile bins using global boundaries.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_ts, pupil_vals, ophys_ts)
# NaN handling per trial:
if nan_mask.any():
    valid_indices = np.where(~nan_mask)[0]
    for j in range(len(pupil_clean)):
        if nan_mask[j]:
            dists = np.abs(valid_indices - j)
            nearest = valid_indices[np.argmin(dists)]
            pupil_clean[j] = pupil_clean[nearest]
pupil_binned = discretize_to_percentile_bins(pupil_clean, n_bins, pupil_percentiles)
```

iii. CONVERSION_NOTES.md: "Pupil diameter (width): linearly interpolated from ~30 Hz eye tracking to ophys timestamps. NaN values (blink artifacts, tracking failures) filled via nearest valid value interpolation."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into 5 bins using equal-percentile boundaries computed globally across all sessions (only non-NaN values). The boundaries are the 20th, 40th, 60th, and 80th percentiles. For trials with all-NaN pupil data, the median bin (index 2) is assigned.

ii.
```python
pupil_percentiles = np.percentile(pupil_arr, np.linspace(0, 100, n_bins + 1)[1:-1])
# pupil_percentiles ≈ [36.8, 41.2, 45.8, 53.0]
if nan_mask.all():
    pupil_binned = np.full(len(pupil_vals), 2, dtype=np.float32)  # median bin
```

iii. CONVERSION_NOTES.md: "Percentile boundaries: [36.8, 41.2, 45.8, 53.0] pixels."

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to ophys timestamps, then sliced to trial frame indices, matching neural data. For single-plane data, downsampled by the same factor.

ii.
```python
pupil_trial = pupil_at_ophys[frame_indices]
if ds_factor > 1:
    pupil_trial = downsample_by_factor(pupil_trial, ds_factor)
```

iii. Aligned via ophys timestamps.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the `ds.trials` table.

ii.
```python
def get_trial_outcome(trial):
    if trial['hit']:
        return 'hit'
    elif trial['miss']:
        return 'miss'
    elif trial['false_alarm']:
        return 'false_alarm'
    elif trial['correct_reject']:
        return 'correct_reject'
    else:
        return 'unknown'
```

iii. CONVERSION_NOTES.md: "4 categories: hit (0), miss (1), false_alarm (2), correct_reject (3)."

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The trial outcome is determined by checking boolean columns in priority order (hit > miss > false_alarm > correct_reject). The outcome string is mapped to an integer index. Although it is a static per-trial variable, it is broadcast to all timepoints in the trial (constant value across time) for format consistency.

ii.
```python
outcome = get_trial_outcome(trial)
outcome_idx = outcome_to_idx.get(outcome, 0)
# In output construction:
np.full(n_timepoints, outcome_idx, dtype=np.int64)
```

iii. CONVERSION_NOTES.md: "Static per trial (same value across all timepoints within a trial). Represented as time-varying for format consistency." The instructions specify "Trial outcome. Static per-trial."

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing data as follows:
- **Running speed NaN**: replaced with 0.0 using `np.nan_to_num`.
- **Pupil diameter NaN**: filled by nearest-neighbor interpolation within each trial. If all values in a trial are NaN, the median bin (index 2) is assigned.
- **Failed experiment loads**: caught by try/except, experiment skipped with a message.
- **Missing eye tracking data**: caught by try/except, filled with NaN array (then handled by the NaN pupil logic).
- **Short trials**: trials with fewer frames than the downsample factor are skipped.
- **Unknown trial outcomes**: mapped to index 0 ('hit') by `outcome_to_idx.get(outcome, 0)`.

ii.
```python
running_at_ophys = np.nan_to_num(running_at_ophys, nan=0.0)

# Pupil NaN:
if nan_mask.all():
    pupil_binned = np.full(len(pupil_vals), 2, dtype=np.float32)
else:
    # nearest-neighbor fill
    for j in range(len(pupil_clean)):
        if nan_mask[j]:
            dists = np.abs(valid_indices - j)
            nearest = valid_indices[np.argmin(dists)]
            pupil_clean[j] = pupil_clean[nearest]

# Failed loads:
except Exception as e:
    return None, None, None, None, None, f"Failed to load: {e}"
```

iii. CONVERSION_NOTES.md: "NaN handling: Running speed NaN filled with 0; Pupil NaN filled via nearest-neighbor interpolation."

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading each experiment via `cache.get_behavior_ophys_experiment()` which reads NWB files from disk (110 experiments).
2. The `get_image_at_timepoints()` function which uses a Python for-loop over every ophys timepoint in each trial.
3. The `get_image_change_signal()` function which iterates over change events with Python for-loop.
4. The pupil NaN filling loop which iterates over all timepoints per trial.

ii.
```python
# Slow NWB loading:
ds = load_experiment(cache, exp_id)

# Slow Python loop over timepoints:
for t_idx, t in enumerate(ophys_timestamps):
    while stim_idx < len(stim_starts) - 1 and stim_starts[stim_idx + 1] <= t:
        stim_idx += 1
```

iii. No explicit discussion of performance in CONVERSION_NOTES.md.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
1. `get_image_at_timepoints()`: The Python for-loop over timepoints could be replaced with `np.searchsorted` to find the most recent stimulus for each timepoint.
2. `get_image_change_signal()`: The Python for-loop over changes could use vectorized `np.searchsorted` or boolean array operations.
3. `downsample_categorical_by_factor()`: The per-bin mode computation loop could use `scipy.stats.mode` on reshaped array.
4. Pupil NaN nearest-neighbor filling: could use `scipy.interpolate.interp1d` with 'nearest' kind, or `pandas.Series.fillna(method='ffill').fillna(method='bfill')`.

ii.
```python
# get_image_at_timepoints - Python loop:
for t_idx, t in enumerate(ophys_timestamps):
    # ...

# downsample_categorical_by_factor - Python loop:
for i in range(n_new):
    chunk = data[i*factor:(i+1)*factor]
    values, counts = np.unique(chunk, return_counts=True)
    result[i] = values[np.argmax(counts)]

# Pupil NaN fill - Python loop:
for j in range(len(pupil_clean)):
    if nan_mask[j]:
        dists = np.abs(valid_indices - j)
        nearest = valid_indices[np.argmin(dists)]
        pupil_clean[j] = pupil_clean[nearest]
```

iii. No discussion of vectorization in CONVERSION_NOTES.md.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats:
1. **Loading stimulus presentations**: `ds.stimulus_presentations` is accessed once globally per experiment, but then per-trial filtering is done inside the trial loop.
2. **Running speed and pupil interpolation to ophys timestamps**: computed once per experiment (correctly), then sliced per trial.
3. **Global percentile computation**: running speed and pupil values are accumulated across all experiments, then percentiles are computed once. This is done correctly (not repeated).
4. **Image name discovery**: only done from first experiment, but could miss image names not present in the first experiment (unlikely since all sessions use the same image set).

ii.
```python
# Per trial, filtering stimulus presentations is repeated:
trial_stim = stim_cd[
    (stim_cd['start_time'] >= t_start - 0.5) &
    (stim_cd['start_time'] <= t_stop + 0.5)
]
```

iii. No discussion of repeated processing in CONVERSION_NOTES.md.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI produces several things that may not be used downstream:
1. **Trial outcome as time-varying**: The instructions say trial outcome is "static per-trial," but the AI broadcasts it to all timepoints, creating a (1, n_timepoints) row that is constant. This wastes memory and computation.
2. **Input array**: Empty input arrays `np.zeros((0, n_timepoints))` are created for every trial despite "No inputs for this task."
3. **Per-experiment behavioral data duplication**: For Multiscope sessions with multiple imaging planes, all behavioral data (running speed, pupil, stimuli, trials) is loaded and processed identically for each plane, duplicating work.
4. **Collecting all running/pupil values into Python lists** before computing percentiles, rather than using online/streaming statistics.

ii.
```python
# Empty inputs:
session_input.append(np.zeros((0, n_timepoints), dtype=np.float32))

# Trial outcome broadcast to time:
np.full(n_timepoints, outcome_idx, dtype=np.int64)
```

iii. No discussion of unnecessary processing in CONVERSION_NOTES.md.
