# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Data is loaded using the Allen SDK's `VisualBehaviorOphysProjectCache`. The experiment table is loaded from a CSV file, then filtered to experiments that have available NWB files, are `active_behavior` sessions, and have `Familiar` experience level. Each experiment is loaded individually via `cache.get_behavior_ophys_experiment()`.

ii.
```python
cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=data_dir)
exp_table = get_experiment_table(data_dir)
# In get_experiment_table:
exp_table = pd.read_csv(os.path.join(data_dir, 'visual-behavior-ophys-1.1.0/project_metadata/ophys_experiment_table.csv'))
available = exp_table[exp_table['ophys_experiment_id'].isin(exp_ids)]
filtered = available[
    (available['behavior_type'] == 'active_behavior') &
    (available['experience_level'] == 'Familiar')
].copy()
```

iii. The agent followed the paper's methods which state "For neural analysis we used neurons recorded during familiar image set presentations." Active behavior was selected to exclude passive viewing sessions where mice are not performing the task. The agent broadened from the paper's Multiscope-only focus to include both equipment types because restricting to Multiscope left only ~22 experiments from 1 mouse.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values from the filtered experiment table. They are sorted and assigned indices.

ii.
```python
all_mice = sorted(set(r['exp_info']['mouse_id'] for r in experiment_results))
mouse_to_idx = {m: i for i, m in enumerate(all_mice)}
subjects = [str(m) for m in all_mice]
```

iii. `mouse_id` is the SDK's unique identifier for each animal.

## 1-c. How are the data split into sessions?

i. Each individual experiment (one imaging plane) is treated as a separate "session" in the output. The agent does NOT group multiple imaging planes from the same `ophys_session_id` into a single session. Each `ophys_experiment_id` maps to one entry in the output session lists.

ii.
```python
for result in experiment_results:
    exp_info = result['exp_info']
    neural_trials = result['neural_trials']
    # ... each experiment becomes one session in the output
    neural_all.append(session_neural)
    input_all.append(session_input)
    output_all.append(session_output)
```

iii. The agent reasoned: "Each imaging plane corresponds to one experiment, and since different planes have different neurons, each experiment should be treated as a separate 'session' in the output format." This means multi-plane sessions have each plane as a separate session rather than combining neurons across planes.

## 1-d. How are the data split into trials?

i. Trials are defined using the built-in `dataset.trials` table. Go and Catch trials are included; Aborted and Auto-rewarded trials are excluded. Each trial spans from `start_time` to `stop_time` (variable length). Ophys frame indices are found using a boolean mask on timestamps.

ii.
```python
valid_trials = trials[
    ((trials['go'] == True) | (trials['catch'] == True)) &
    (trials['aborted'] == False) &
    (trials['auto_rewarded'] == False)
]
# ...
frame_mask = (ophys_ts >= t_start) & (ophys_ts <= t_stop)
frame_indices = np.where(frame_mask)[0]
```

iii. The task instructions explicitly state to include Go and Catch trials and exclude Aborted and Auto-rewarded trials.

## 1-e. How are trials filtered based on quality controls?

i. The following filters are applied: (1) Only Go and Catch trials are included. (2) Aborted and Auto-rewarded trials are excluded. (3) Trials with fewer frames than the downsample factor are skipped. (4) After processing, experiments with fewer than 2 valid trials are excluded. There is no explicit `change_time` validity check (unlike the reference).

ii.
```python
valid_trials = trials[
    ((trials['go'] == True) | (trials['catch'] == True)) &
    (trials['aborted'] == False) &
    (trials['auto_rewarded'] == False)
]
if len(frame_indices) < ds_factor:
    continue
if len(neural_trials) < 2:
    return None, ..., f"Only {len(neural_trials)} usable trials after processing"
```

iii. The agent followed the task instructions for trial type filtering. The minimum trial count ensures decoder training is feasible.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dataset.events` (detected calcium events), accessed via `ds.events['events']`. This is the deconvolved calcium event signal, NOT the raw dF/F traces.

ii.
```python
events = ds.events
neural_full = np.vstack(events['events'].values).astype(np.float32)
```

iii. The agent cited the paper: "we used the detected calcium events as described in Garrett et al." The CONVERSION_NOTES state: "Events are deconvolved from fluorescence traces using the method in Giovannucci et al. 2019, providing cleaner signals with ~200ms resolution."

## 2-b. How is the `neural` data processed?

i. Neural data is extracted per-experiment (single imaging plane). If the experiment's frame rate differs from the target rate (93.2ms, ~10.7 Hz matching multiscope), the neural data is downsampled by averaging every `ds_factor` consecutive frames. For single-plane Scientifica data (~31 Hz), `ds_factor = 3`.

ii.
```python
ds_factor = max(1, int(round(target_dt / dt)))
# ...
if ds_factor > 1:
    neural_trial = downsample_by_factor(neural_trial, ds_factor)
# downsample_by_factor averages groups of 'factor' samples:
return data[:, :n_new * factor].reshape(data.shape[0], n_new, factor).mean(axis=2)
```

iii. The agent reasoned that different equipment types have different frame rates (multiscope ~11 Hz, single-plane ~31 Hz), and the decoder format requires consistent time bins. Downsampling to the multiscope rate ensures a common temporal resolution.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied. All neurons present in `ds.events` are included. Experiments with 0 neurons are skipped.

ii.
```python
n_neurons = neural_full.shape[0]
if n_neurons == 0:
    return None, ..., "No neurons"
```

iii. No explicit reasoning was given. The agent relied on the SDK's existing quality control.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to ophys timestamps. For each trial, frames are selected where `ophys_ts >= start_time` and `ophys_ts <= stop_time`. The trial starts at `start_time` from the trials table.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts <= t_stop)
frame_indices = np.where(frame_mask)[0]
neural_trial = neural_full[:, frame_indices]
```

iii. The task instructions say "Temporally align based on ophys timestamp." The agent uses the full trial window from start to stop.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The target time bin size is 93.2 ms (~10.7 Hz), matching the multiscope frame rate. Single-plane data (~31 Hz) is downsampled by a factor of 3 via averaging. Multiscope data is kept at native rate (ds_factor = 1).

ii.
```python
target_dt = 0.0932  # seconds
ds_factor = max(1, int(round(target_dt / dt)))
if ds_factor > 1:
    neural_trial = downsample_by_factor(neural_trial, ds_factor)
```

iii. The agent reasoned: "Multiscope gives roughly 11 Hz (93ms bins) while single-plane gives 31 Hz (32ms bins). I could downsample everything to 11 Hz to work across both."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, specifically the `image_name`, `start_time`, and `end_time` columns. Only presentations from the `change_detection` stimulus block are used, and `omitted` stimuli are excluded.

ii.
```python
stim = ds.stimulus_presentations
stim_cd = stim[stim['stimulus_block_name'].str.contains('change_detection', na=False)].copy()
stim_images = stim_cd[stim_cd['image_name'] != 'omitted']
image_idx = get_image_at_timepoints(trial_stim_no_omit, trial_ophys_ts, image_to_idx)
```

iii. The agent used `stimulus_presentations` because it provides frame-by-frame image identity tracking, which is needed for the time-varying output. During gray screen intervals, the last shown image identity persists.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A mapping from image names to integer indices is built from all unique non-omitted image names discovered in the first experiment's stimulus presentations. At each ophys timepoint, the function `get_image_at_timepoints` walks through stimulus presentations to find the current or most recent image. For downsampled data, categorical downsampling uses mode within each bin.

ii.
```python
image_to_idx = {name: idx for idx, name in enumerate(all_image_names)}
# get_image_at_timepoints iterates over ophys timepoints and stimulus presentations
for t_idx, t in enumerate(ophys_timestamps):
    while stim_idx < len(stim_starts) - 1 and stim_starts[stim_idx + 1] <= t:
        stim_idx += 1
    if stim_idx < len(stim_starts) and stim_starts[stim_idx] <= t:
        img = stim_images[stim_idx]
        if img in image_to_idx and img != 'omitted':
            current_image_idx = image_to_idx[img]
# For downsampled data:
image_idx = downsample_categorical_by_factor(image_idx, ds_factor)
```

iii. The agent used stimulus_presentations for accurate frame-by-frame image tracking. During omitted stimuli, the previous image identity is maintained.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the same ophys timepoints used for neural data extraction, then downsampled by the same factor if applicable.

ii.
```python
trial_ophys_ts = ophys_ts[frame_indices]
image_idx = get_image_at_timepoints(trial_stim_no_omit, trial_ophys_ts, image_to_idx)
if ds_factor > 1:
    image_idx = downsample_categorical_by_factor(image_idx, ds_factor)
```

iii. Using the same ophys timestamps and frame indices ensures alignment with neural data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the `stimulus_presentations` table. Only presentations marked `is_change == True` are used to identify change events.

ii.
```python
changes = stim_presentations[stim_presentations['is_change'] == True]
```

iii. The `is_change` field in `stimulus_presentations` directly encodes which stimulus presentations are change events.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary signal is created: 1 at ophys timepoints within a 750ms window after each change event, 0 otherwise. The 750ms window covers the change stimulus (250ms) plus the following gray inter-stimulus interval (500ms). For downsampled data, the averaged signal is re-binarized (>0 becomes 1).

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
# After downsampling:
change_signal = (change_signal > 0.0).astype(np.float32)
```

iii. The 750ms window matches one full stimulus flash cycle (250ms on + 500ms gray).

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1). No thresholding is needed beyond the 750ms window definition. After downsampling, the signal is re-binarized (values > 0 become 1).

ii.
```python
change_signal = (change_signal > 0.0).astype(np.float32)
```

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The change signal is computed over the same trial ophys timestamps used for neural data, then downsampled by the same factor.

ii.
```python
change_signal = get_image_change_signal(trial_stim, trial_ophys_ts)
if ds_factor > 1:
    change_signal = downsample_by_factor(change_signal, ds_factor)
    change_signal = (change_signal > 0.0).astype(np.float32)
```

iii. Same alignment approach as image identity.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, which provides speed and timestamps from the running wheel encoder.

ii.
```python
rs = ds.running_speed
running_at_ophys = interpolate_to_ophys(
    rs['timestamps'].values, rs['speed'].values, ophys_ts
)
```

iii. The `running_speed` attribute is the SDK's standard interface for locomotion data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated to ophys timestamps. NaN values (from extrapolation) are replaced with 0. Values are collected across all experiments for global percentile computation. The signal is discretized into 5 bins using `np.digitize` with global percentile boundaries (20th, 40th, 60th, 80th percentiles). For single-plane data, running speed is also downsampled by averaging.

ii.
```python
running_at_ophys = interpolate_to_ophys(rs['timestamps'].values, rs['speed'].values, ophys_ts)
running_at_ophys = np.nan_to_num(running_at_ophys, nan=0.0)
all_running_speeds.extend(running_at_ophys.tolist())
# Later:
running_percentiles = np.percentile(running_arr, np.linspace(0, 100, n_bins + 1)[1:-1])
running_binned = discretize_to_percentile_bins(out_raw['running'], n_bins, running_percentiles)
```

iii. Linear interpolation preserves signal shape. Percentile-based binning ensures roughly equal class counts.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal percentile bins using global bin edges computed from all sessions. The `np.digitize` function assigns each value to a bin based on the 20th, 40th, 60th, and 80th percentile boundaries, then clipped to [0, 4].

ii.
```python
running_percentiles = np.percentile(running_arr, np.linspace(0, 100, n_bins + 1)[1:-1])
def discretize_to_percentile_bins(values, n_bins, percentiles):
    result = np.digitize(values, percentiles)
    result = np.clip(result, 0, n_bins - 1)
    return result.astype(np.int64)
```

iii. Global percentile boundaries ensure consistent categories across all sessions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to ophys timestamps before trial segmentation, then extracted using the same frame indices as neural data. Downsampled by the same factor when applicable.

ii.
```python
running_at_ophys = interpolate_to_ophys(rs['timestamps'].values, rs['speed'].values, ophys_ts)
running_trial = running_at_ophys[frame_indices]
if ds_factor > 1:
    running_trial = downsample_by_factor(running_trial, ds_factor)
```

iii. Interpolation to ophys timestamps guarantees alignment with neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`, using the `pupil_width` column and corresponding timestamps.

ii.
```python
et = ds.eye_tracking
pupil_vals = et['pupil_width'].values
pupil_ts = et['timestamps'].values
pupil_at_ophys = interpolate_to_ophys(pupil_ts, pupil_vals, ophys_ts)
```

iii. The agent used `pupil_width` as the pupil diameter measure, citing the SDK tutorial.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil width is linearly interpolated to ophys timestamps. NaN values in the interpolated signal are removed before collecting values for global percentile computation. During binning, NaN values are handled via nearest-neighbor interpolation (finding the nearest non-NaN value by index). If all values are NaN, the median bin (bin 2) is assigned. The signal is discretized into 5 percentile bins. Note: blinks are NOT filtered before interpolation (unlike the reference which removes `likely_blink` frames first).

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_ts, pupil_vals, ophys_ts)
# NaN handling during binning:
if nan_mask.any():
    valid_indices = np.where(~nan_mask)[0]
    for j in range(len(pupil_clean)):
        if nan_mask[j]:
            dists = np.abs(valid_indices - j)
            nearest = valid_indices[np.argmin(dists)]
            pupil_clean[j] = pupil_clean[nearest]
pupil_binned = discretize_to_percentile_bins(pupil_clean, n_bins, pupil_percentiles)
```

iii. The agent did not explicitly discuss blink filtering. The `interpolate_to_ophys` function removes NaN values before interpolation, but blink frames with valid (but artifact-corrupted) pupil measurements would still be included.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: discretized into 5 equal percentile bins using global percentile boundaries. All-NaN trials are assigned the median bin (2).

ii.
```python
pupil_percentiles = np.percentile(pupil_arr, np.linspace(0, 100, n_bins + 1)[1:-1])
pupil_binned = discretize_to_percentile_bins(pupil_clean, n_bins, pupil_percentiles)
```

iii. Same approach as running speed for consistency.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed: interpolated to ophys timestamps, extracted at the same frame indices, and downsampled by the same factor.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_ts, pupil_vals, ophys_ts)
pupil_trial = pupil_at_ophys[frame_indices]
if ds_factor > 1:
    pupil_trial = downsample_by_factor(pupil_trial, ds_factor)
```

iii. Same alignment as running speed.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

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

iii. These four columns are the SDK's canonical trial outcome labels for the change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3) via a fixed mapping. The outcome is constant across all time bins within a trial (broadcast to all timepoints).

ii.
```python
outcome_names = ['hit', 'miss', 'false_alarm', 'correct_reject']
outcome_to_idx = {name: idx for idx, name in enumerate(outcome_names)}
outcome_idx = outcome_to_idx.get(outcome, 0)
np.full(n_timepoints, outcome_idx, dtype=np.int64)
```

iii. The mapping order matches the `outcome_names` list. Unknown outcomes default to index 0.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Failed experiments**: If loading an experiment throws an exception, it is skipped.
- **Short trials**: Trials with fewer frames than the downsample factor are skipped.
- **Missing running speed**: NaN values are replaced with 0 before collection and binning.
- **Missing pupil data**: NaN values are filled via nearest-neighbor interpolation. All-NaN trials get median bin (2). If eye tracking fails entirely, all-NaN array is used.
- **Few trials**: Experiments with fewer than 2 valid trials are excluded.

ii.
```python
try:
    ds = load_experiment(cache, exp_id)
except Exception as e:
    return None, ..., f"Failed to load: {e}"
running_at_ophys = np.nan_to_num(running_at_ophys, nan=0.0)
try:
    et = ds.eye_tracking
except Exception:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
if len(neural_trials) < 2:
    return None, ..., f"Only {len(neural_trials)} usable trials"
```

iii. The try/except ensures a single bad experiment doesn't crash the pipeline. NaN handling prevents missing data from propagating into the output.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each experiment via `cache.get_behavior_ophys_experiment()`, which reads large NWB files containing neural and behavioral data. The `get_image_at_timepoints` function is also potentially slow as it iterates over every ophys timepoint.

ii. N/A

iii. Each experiment contains full-session data arrays. The per-timepoint iteration in `get_image_at_timepoints` could be a bottleneck for long sessions.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `get_image_at_timepoints` function (lines 119-150) iterates over every ophys timepoint in a Python for-loop to determine image identity. This could be vectorized using `np.searchsorted`. The pupil NaN nearest-neighbor fill (lines 528-535) also iterates per-timepoint and could be vectorized.

ii.
```python
# Slow per-timepoint loop:
for t_idx, t in enumerate(ophys_timestamps):
    while stim_idx < len(stim_starts) - 1 and stim_starts[stim_idx + 1] <= t:
        stim_idx += 1
    ...
# Slow NaN fill:
for j in range(len(pupil_clean)):
    if nan_mask[j]:
        dists = np.abs(valid_indices - j)
        nearest = valid_indices[np.argmin(dists)]
        pupil_clean[j] = pupil_clean[nearest]
```

iii. Both loops operate on arrays that could benefit from vectorized numpy operations.

## 9-c. What processing does the code repeat multiple times?

i. The code loads the first experiment twice: once in the main `convert_data` function to discover image names, and again during the experiment processing loop. Running speed and pupil values are collected into lists during processing and then turned into arrays for percentile computation (two-pass approach), though this is by design.

ii.
```python
# First load to discover images:
first_exp = cache.get_behavior_ophys_experiment(int(exp_table.iloc[0]['ophys_experiment_id']))
# Then loaded again in the processing loop:
ds = load_experiment(cache, exp_id)  # same experiment
```

iii. The first load is used to discover the image name set; this could be avoided by collecting image names during the main processing loop.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code collects `trial_outcomes` as a separate list per experiment result, but this is redundant since the outcomes are already stored in each trial's `output_trials` dict. The code also stores running speed NaN-filled with 0 into `all_running_speeds` for percentile computation, which means the 0-filled values affect the percentile boundaries. This is potentially problematic rather than unnecessary.

ii.
```python
trial_outcomes.append(outcome)  # collected but never used independently
```

iii. The redundant list has no significant impact on performance or correctness.
