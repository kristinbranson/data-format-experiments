# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads experiments from a CSV experiment table (`ophys_experiment_table.csv`) and discovers available NWB files on disk. It filters to experiments with `behavior_type == 'active_behavior'` and `experience_level == 'Familiar'`. Each experiment is loaded individually using `cache.get_behavior_ophys_experiment()`.

ii.
```python
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
```

iii. The AI chose to filter by `active_behavior` and `Familiar` based on the paper's methods: "For neural analysis we used neurons recorded during familiar image set presentations." It also includes both Scientifica (single-plane) and Multiscope (multi-plane) equipment types.

## 1-b. How are the data split into subjects (mice)?

i. Subjects correspond to unique `mouse_id` values from the filtered experiment table.

ii.
```python
all_mice = sorted(set(r['exp_info']['mouse_id'] for r in experiment_results))
mouse_to_idx = {m: i for i, m in enumerate(all_mice)}
subjects = [str(m) for m in all_mice]
```

iii. The `mouse_id` field is the standard identifier for each animal.

## 1-c. How are the data split into sessions?

i. Each experiment (single imaging plane) is treated as a separate session. The AI does NOT group multiple imaging planes from the same `ophys_session_id` into a single session.

ii.
```python
for i, (_, exp_info) in enumerate(exp_table.iterrows()):
    exp_id = int(exp_info['ophys_experiment_id'])
    result = process_experiment(cache, exp_info, ...)
    experiment_results.append({
        'exp_info': exp_info,
        'neural_trials': neural_trials,
        ...
    })
```

iii. The AI reasoned: "Each imaging plane corresponds to one experiment, and since different planes have different neurons, each experiment should be treated as a separate 'session' in the output format." This treats each imaging plane as its own session.

## 1-d. How are the data split into trials?

i. Trials are defined using the `ds.trials` table. Go and Catch trials are included; Aborted and Auto-rewarded trials are excluded. The trial window spans `start_time` to `stop_time` (variable length).

ii.
```python
valid_trials = trials[
    ((trials['go'] == True) | (trials['catch'] == True)) &
    (trials['aborted'] == False) &
    (trials['auto_rewarded'] == False)
]
for trial_idx, (_, trial) in enumerate(valid_trials.iterrows()):
    t_start = trial['start_time']
    t_stop = trial['stop_time']
    frame_mask = (ophys_ts >= t_start) & (ophys_ts <= t_stop)
    frame_indices = np.where(frame_mask)[0]
```

iii. The instructions specify including Go and Catch trials and excluding Aborted and Auto-rewarded. The AI uses boolean filtering on the trials table.

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded if: (a) they have fewer frames than the downsample factor, (b) after processing fewer than 2 valid trials remain per experiment. Sessions that fail to load are skipped with exception handling.

ii.
```python
if len(frame_indices) < ds_factor:
    continue
...
if len(neural_trials) < 2:
    return None, None, None, None, None, f"Only {len(neural_trials)} usable trials after processing"
```

iii. The minimum trial length check ensures that downsampling doesn't produce empty trials. The 2-trial minimum ensures decoder evaluation is possible.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from **detected calcium events** (`ds.events`), NOT from dF/F traces.

ii.
```python
events = ds.events
neural_full = np.vstack(events['events'].values).astype(np.float32)
```

iii. From CONVERSION_NOTES.md: "Used detected calcium events (not raw dF/F), matching the paper: 'we used the detected calcium events as described in Garrett et al.'" The AI chose events because the paper specifically states this is what they used for neural analysis.

## 2-b. How is the `neural` data processed?

i. Neural data is extracted per experiment (single imaging plane). For single-plane data (~31 Hz), it is downsampled by a factor of 3 via averaging to match the multiscope rate (~10.7 Hz). No additional normalization or filtering is applied.

ii.
```python
def downsample_by_factor(data, factor):
    if data.ndim == 2:
        n = data.shape[1]
        n_new = n // factor
        return data[:, :n_new * factor].reshape(data.shape[0], n_new, factor).mean(axis=2)
...
if ds_factor > 1:
    neural_trial = downsample_by_factor(neural_trial, ds_factor)
```

iii. Downsampling ensures a consistent time bin size across sessions from different equipment types. The target_dt of 0.0932s matches the multiscope frame rate.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied to neurons. All neurons present in `ds.events` are included. The Allen SDK pipeline's built-in ROI filtering is relied upon.

ii. N/A (no explicit filtering code)

iii. From CONVERSION_NOTES.md: "ROIs are pre-filtered by the AllenSDK processing pipeline (see whitepaper Section F: ROI Filtering)."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to `start_time` of each trial. Ophys frames from `start_time` to `stop_time` are extracted using a boolean mask on ophys timestamps.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts <= t_stop)
frame_indices = np.where(frame_mask)[0]
neural_trial = neural_full[:, frame_indices]
```

iii. The instructions say to "temporally align based on ophys timestamp" and segment into trials as defined in the experiment. The trial boundaries come from the SDK's trials table.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The target time bin is 93.2 ms (~10.7 Hz), matching the multiscope frame rate. Single-plane data (~31 Hz) is downsampled by factor 3 via averaging. Multiscope data is kept at its native rate.

ii.
```python
target_dt = 0.0932  # seconds
ds_factor = max(1, int(round(target_dt / dt)))
```

iii. From CONVERSION_NOTES.md: "All data resampled to 93.2 ms bins (~10.7 Hz), matching the Multiscope frame rate. Single-plane Scientifica data (~31 Hz, ~32.3 ms bins) downsampled by factor of 3 via averaging."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, specifically the `image_name` column, using start/end times and a most-recent-image tracking approach.

ii.
```python
def get_image_at_timepoints(stim_presentations, ophys_timestamps, image_to_idx):
    stim_starts = stim_presentations['start_time'].values
    stim_ends = stim_presentations['end_time'].values
    stim_images = stim_presentations['image_name'].values
    for t_idx, t in enumerate(ophys_timestamps):
        while stim_idx < len(stim_starts) - 1 and stim_starts[stim_idx + 1] <= t:
            stim_idx += 1
        if stim_idx < len(stim_starts) and stim_starts[stim_idx] <= t:
            img = stim_images[stim_idx]
            if img in image_to_idx and img != 'omitted':
                current_image_idx = image_to_idx[img]
```

iii. The AI chose to use `stimulus_presentations` rather than the trials table fields `initial_image_name`/`change_image_name`, building a frame-by-frame timeline of which image was shown.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer indices via a global sorted mapping. At each ophys timepoint, the most recently presented (non-omitted) image is assigned. During grey screens between images, the last image persists. For single-plane data, the resulting categorical array is downsampled using mode (most frequent value in each bin).

ii.
```python
image_to_idx = {name: idx for idx, name in enumerate(all_image_names)}
...
image_idx = get_image_at_timepoints(trial_stim_no_omit, trial_ophys_ts, image_to_idx)
if ds_factor > 1:
    image_idx = downsample_categorical_by_factor(image_idx, ds_factor)
```

iii. From CONVERSION_NOTES.md: "At each timepoint, assigned the most recently presented image. During gray screen intervals, the last shown image identity persists."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at each ophys frame within the trial, using the same ophys timestamps as neural data. For single-plane data, it is downsampled by the same factor as neural data.

ii.
```python
trial_ophys_ts = ophys_ts[frame_indices]
image_idx = get_image_at_timepoints(trial_stim_no_omit, trial_ophys_ts, image_to_idx)
```

iii. Both neural and image identity use the same ophys frame indices, ensuring alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `stimulus_presentations` table, specifically using the `is_change` column to identify frames where a change occurred.

ii.
```python
def get_image_change_signal(stim_presentations, ophys_timestamps):
    changes = stim_presentations[stim_presentations['is_change'] == True]
    for _, change in changes.iterrows():
        change_time = change['start_time']
        mask = (ophys_timestamps >= change_time) & (ophys_timestamps < change_time + 0.75)
        change_signal[mask] = 1.0
    return change_signal
```

iii. From CONVERSION_NOTES.md: "Binary signal: 1 during the 750ms following a change in image identity, 0 otherwise. Only actual changes are marked (not sham changes in catch trials)."

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary signal is created where 1 indicates the 750ms window after an image change (one stimulus flash + grey period). For single-plane data, the signal is downsampled by averaging and then re-binarized (> 0).

ii.
```python
mask = (ophys_timestamps >= change_time) & (ophys_timestamps < change_time + 0.75)
change_signal[mask] = 1.0
...
if ds_factor > 1:
    change_signal = downsample_by_factor(change_signal, ds_factor)
    change_signal = (change_signal > 0.0).astype(np.float32)
```

iii. The 750ms window matches one stimulus flash (250ms on) plus grey period (500ms off).

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is binary (0 or 1). The 750ms window determines when the value is 1. After downsampling, any bin with any non-zero contribution is set to 1.

ii. See 4-b above.

iii. The instructions specify "value of 1 right after a change in image identity, otherwise 0."

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is computed at ophys frame times within the trial, using the same frame indices as neural data. Downsampled by the same factor if applicable.

ii.
```python
change_signal = get_image_change_signal(trial_stim, trial_ophys_ts)
```

iii. Same alignment approach as image identity.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ds.running_speed`, using the `speed` and `timestamps` columns.

ii.
```python
rs = ds.running_speed
running_at_ophys = interpolate_to_ophys(
    rs['timestamps'].values, rs['speed'].values, ophys_ts
)
```

iii. The SDK's `running_speed` attribute provides locomotion data from the running wheel encoder.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated to ophys timestamps. NaN values are replaced with 0 before binning. The continuous values are discretized into 5 percentile-based bins using global percentile boundaries computed across all experiments. For single-plane data, running speed is downsampled by averaging before binning.

ii.
```python
running_at_ophys = interpolate_to_ophys(rs['timestamps'].values, rs['speed'].values, ophys_ts)
running_at_ophys = np.nan_to_num(running_at_ophys, nan=0.0)
...
running_percentiles = np.percentile(running_arr, np.linspace(0, 100, n_bins + 1)[1:-1])
running_binned = discretize_to_percentile_bins(out_raw['running'], n_bins, running_percentiles)
```

iii. Linear interpolation resamples to the ophys timebase. NaN-to-0 replacement handles missing data. Global percentile boundaries ensure consistent binning across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using `np.digitize` with percentile boundaries (20th, 40th, 60th, 80th percentiles). Values are clipped to range [0, 4].

ii.
```python
def discretize_to_percentile_bins(values, n_bins, percentiles):
    result = np.digitize(values, percentiles)
    result = np.clip(result, 0, n_bins - 1)
    return result.astype(np.int64)
```

iii. The instructions specify "discretized into five equal percentile bins."

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to ophys timestamps before trial segmentation, then extracted using the same frame indices as neural data.

ii.
```python
running_at_ophys = interpolate_to_ophys(rs['timestamps'].values, rs['speed'].values, ophys_ts)
...
running_trial = running_at_ophys[frame_indices]
```

iii. Interpolation to the ophys timebase guarantees alignment with neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `ds.eye_tracking`, using the `pupil_width` column.

ii.
```python
et = ds.eye_tracking
pupil_vals = et['pupil_width'].values
pupil_ts = et['timestamps'].values
pupil_at_ophys = interpolate_to_ophys(pupil_ts, pupil_vals, ophys_ts)
```

iii. From CONVERSION_NOTES.md: "Uses pupil_width from eye tracking data as the diameter measure."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter is linearly interpolated to ophys timestamps. NaN values within the interpolated signal are filled via nearest-neighbor interpolation (finding the closest valid value). The continuous values are then discretized into 5 percentile-based bins. If all values are NaN, the median bin (2) is assigned. Blink frames are NOT explicitly removed before interpolation.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_ts, pupil_vals, ophys_ts)
...
# NaN handling:
if nan_mask.any():
    valid_indices = np.where(~nan_mask)[0]
    for j in range(len(pupil_clean)):
        if nan_mask[j]:
            dists = np.abs(valid_indices - j)
            nearest = valid_indices[np.argmin(dists)]
            pupil_clean[j] = pupil_clean[nearest]
pupil_binned = discretize_to_percentile_bins(pupil_clean, n_bins, pupil_percentiles)
```

iii. The `interpolate_to_ophys` function removes NaN values before interpolation but does NOT specifically remove blink frames (via `likely_blink` flag). The nearest-neighbor NaN filling is done after interpolation to handle remaining gaps.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: `np.digitize` with global percentile boundaries, clipped to [0, 4].

ii. Same as 5-c, applied to pupil values.

iii. The instructions specify "discretized into five equal percentile bins."

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed: interpolated to ophys timestamps, then extracted using the same frame indices.

ii.
```python
pupil_trial = pupil_at_ophys[frame_indices]
```

iii. Same alignment approach as running speed.

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

i. Trial outcome strings are mapped to integer indices (hit=0, miss=1, false_alarm=2, correct_reject=3). The outcome is constant across all timepoints within a trial (replicated to fill the time dimension).

ii.
```python
outcome_names = ['hit', 'miss', 'false_alarm', 'correct_reject']
outcome_to_idx = {name: idx for idx, name in enumerate(outcome_names)}
...
outcome_idx = outcome_to_idx.get(outcome, 0)
np.full(n_timepoints, outcome_idx, dtype=np.int64)
```

iii. The instructions specify trial outcome as "static per-trial."

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Failed experiments**: Exception handling skips experiments that fail to load.
- **Short trials**: Trials with fewer frames than the downsample factor are skipped.
- **Running speed NaN**: Replaced with 0 before percentile computation and binning.
- **Pupil NaN**: After interpolation, remaining NaN values are filled via nearest-neighbor interpolation. If all values are NaN, median bin (2) is assigned. NaN values are excluded from percentile computation.
- **Few trials**: Experiments with fewer than 2 valid trials are skipped.
- **Missing eye tracking**: Falls back to all-NaN pupil values.

ii.
```python
running_at_ophys = np.nan_to_num(running_at_ophys, nan=0.0)
...
try:
    et = ds.eye_tracking
    ...
except Exception:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
...
if nan_mask.all():
    pupil_binned = np.full(len(pupil_vals), 2, dtype=np.float32)
```

iii. The AI implemented robust error handling to prevent individual experiment failures from crashing the pipeline.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each experiment via `cache.get_behavior_ophys_experiment()`, which reads large NWB files from disk. The per-timepoint loop in `get_image_at_timepoints` is also slow due to iterating over every ophys timestamp.

ii.
```python
ds = load_experiment(cache, exp_id)
...
for t_idx, t in enumerate(ophys_timestamps):  # O(n_timepoints) loop
```

iii. Data loading is I/O bound and dominates runtime.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `get_image_at_timepoints` function iterates over every ophys timepoint in a Python loop to determine which image was shown. This could be vectorized using `np.searchsorted` on stimulus start times. The nearest-neighbor NaN filling loop for pupil data could also be vectorized.

ii.
```python
for t_idx, t in enumerate(ophys_timestamps):
    while stim_idx < len(stim_starts) - 1 and stim_starts[stim_idx + 1] <= t:
        stim_idx += 1
    ...
```

iii. These loops are O(n_timepoints) per trial, but since n_timepoints per trial is relatively small (~85-270 frames), the performance impact is modest.

## 9-c. What processing does the code repeat multiple times?

i. Running speed and pupil data are collected globally across ALL ophys timepoints (not just trial timepoints) for percentile computation, then re-processed per trial for binning. The full-session behavioral signals are processed even for timepoints outside any trial.

ii.
```python
all_running_speeds.extend(running_at_ophys.tolist())  # All timepoints, not just trial ones
...
running_trial = running_at_ophys[frame_indices]  # Then re-extracted per trial
```

iii. This is slightly wasteful since percentile computation includes non-trial timepoints, but doesn't constitute repeated computation per se.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Running speed and pupil diameter percentile computation includes ALL ophys timepoints (including inter-trial intervals), not just trial timepoints. The AI also collects full-session behavioral data into lists before processing trials, adding memory overhead. Additionally, the `stimulus_presentations` table is filtered and re-queried per trial rather than being processed once per session.

ii.
```python
all_running_speeds.extend(running_at_ophys.tolist())  # Includes non-trial timepoints
...
trial_stim = stim_cd[
    (stim_cd['start_time'] >= t_start - 0.5) &
    (stim_cd['start_time'] <= t_stop + 0.5)
]  # Re-filtered per trial
```

iii. The per-trial stimulus table filtering is repeated work that could be avoided with a single pass.
