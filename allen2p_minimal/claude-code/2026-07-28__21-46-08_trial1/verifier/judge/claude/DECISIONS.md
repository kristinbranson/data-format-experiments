# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data using the Allen SDK's `VisualBehaviorOphysProjectCache.from_s3_cache`. It reads an experiment table from a CSV file, discovers available NWB files on disk, and filters to experiments that are `active_behavior` with `Familiar` experience level. Each experiment is loaded individually via `cache.get_behavior_ophys_experiment()`. This is notably different from the reference, which filters by `project_code == 'VisualBehavior'` (single-plane only) without additional behavior_type or experience_level filtering.

ii.
```python
def get_experiment_table(data_dir='data'):
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
and
```python
cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=data_dir)
```

iii. The AI justified filtering to active behavior and familiar images by citing the paper's methods: "For neural analysis we used neurons recorded during familiar image set presentations." The AI also included both single-plane (Scientifica) and multi-plane (Multiscope) data to get a larger dataset.

## 1-b. How are the data split into subjects?

i. Subjects correspond to unique `mouse_id` values from the filtered experiment table. Unique mice are sorted and assigned integer indices.

ii.
```python
all_mice = sorted(set(r['exp_info']['mouse_id'] for r in experiment_results))
mouse_to_idx = {m: i for i, m in enumerate(all_mice)}
subjects = [str(m) for m in all_mice]
```

iii. The `mouse_id` field is the SDK's unique identifier for each animal. The AI reports 38 mice.

## 1-c. How are the data split into sessions?

i. The AI treats each individual experiment (imaging plane) as a separate session. It does NOT group experiments by `ophys_session_id`. This means if a session had multiple imaging planes, each plane becomes its own "session" in the output. This differs from the reference, which groups all experiments sharing the same `ophys_session_id` into a single session, combining neurons across planes.

ii.
```python
for i, (_, exp_info) in enumerate(exp_table.iterrows()):
    exp_id = int(exp_info['ophys_experiment_id'])
    result = process_experiment(cache, exp_info, ...)
    ...
    experiment_results.append({
        'exp_info': exp_info,
        'neural_trials': neural_trials,
        ...
    })
```

iii. The AI's CONVERSION_NOTES.md states "110 experiments from 38 mice" are treated as 110 sessions. The reference groups by `ophys_session_id`, resulting in fewer sessions with more neurons per session.

## 1-d. How are the data split into trials?

i. Trials are defined using the SDK's `trials` table. The AI includes trials where `go == True` or `catch == True`, and excludes `aborted` and `auto_rewarded` trials. The trial window spans `start_time` to `stop_time` (variable length). Ophys frames are selected with an inclusive mask `(ophys_ts >= t_start) & (ophys_ts <= t_stop)`.

ii.
```python
valid_trials = trials[
    ((trials['go'] == True) | (trials['catch'] == True)) &
    (trials['aborted'] == False) &
    (trials['auto_rewarded'] == False)
]
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts <= t_stop)
frame_indices = np.where(frame_mask)[0]
```

iii. The AI followed the instructions to include Go and Catch trials and exclude Aborted and Auto-rewarded. The reference additionally filters on `change_time.notna()`.

## 1-e. How are trials filtered based on quality controls?

i. Aborted and auto-rewarded trials are excluded. Trials with fewer than `ds_factor` frames after windowing are skipped. Sessions with fewer than 2 valid trials are excluded. The AI does not filter on `change_time.notna()`.

ii.
```python
valid_trials = trials[
    ((trials['go'] == True) | (trials['catch'] == True)) &
    (trials['aborted'] == False) &
    (trials['auto_rewarded'] == False)
]
...
if len(frame_indices) < ds_factor:
    continue
...
if len(neural_trials) < 2:
    return None, ..., f"Only {len(neural_trials)} usable trials after processing"
```

iii. The AI used the instructions' explicit criteria (Go/Catch only, no Aborted/Auto-rewarded). The minimum trial count of 2 prevents degenerate sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses `events` (detected calcium events, deconvolved from dF/F) rather than `dff_traces` (raw dF/F). This is accessed via `ds.events`.

ii.
```python
events = ds.events
neural_full = np.vstack(events['events'].values).astype(np.float32)
```

iii. The AI justified this by citing the paper: "we used the detected calcium events as described in Garrett et al." The CONVERSION_NOTES state: "Events are deconvolved from fluorescence traces using the method in Giovannucci et al. 2019, providing cleaner signals with ~200ms resolution."

## 2-b. How is the `neural` data processed?

i. For single-plane experiments (~31 Hz), neural data is downsampled by a factor of 3 via averaging to match a target time bin of 93.2ms (~10.7 Hz, the multiscope rate). Multi-plane data at ~11 Hz is kept as-is. Since each experiment is treated as a separate session, there is no merging of neurons across planes.

ii.
```python
ds_factor = max(1, int(round(target_dt / dt)))
...
if ds_factor > 1:
    neural_trial = downsample_by_factor(neural_trial, ds_factor)

def downsample_by_factor(data, factor):
    if data.ndim == 2:
        n = data.shape[1]
        n_new = n // factor
        return data[:, :n_new * factor].reshape(data.shape[0], n_new, factor).mean(axis=2)
```

iii. The AI stated: "All data resampled to 93.2 ms bins (~10.7 Hz), matching the Multiscope frame rate. Single-plane Scientifica data (~31 Hz, ~32.3 ms bins) downsampled by factor of 3 via averaging."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied to neural data beyond what the SDK pipeline provides. All neurons in the `events` table are included.

ii. N/A (no explicit filtering code)

iii. The CONVERSION_NOTES state: "ROIs are pre-filtered by the AllenSDK processing pipeline (see whitepaper Section F: ROI Filtering). Only valid cell bodies are included."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (`start_time`). Ophys frames between `start_time` and `stop_time` are extracted. The metadata records `temporal_alignment_event` as "Trial start (aligned to ophys timestamps)".

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts <= t_stop)
frame_indices = np.where(frame_mask)[0]
neural_trial = neural_full[:, frame_indices]
```

iii. The alignment is to the ophys timestamps, as instructed. The trial window spans from `start_time` to `stop_time`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI applies temporal rebinning. It targets a common time bin of 93.2ms (~10.7 Hz) matching the Multiscope frame rate. Single-plane data (~31 Hz) is downsampled by a factor of 3. Multi-plane data is kept at native rate. The reference keeps the native frame rate without rebinning.

ii.
```python
target_dt = 0.0932  # seconds
ds_factor = max(1, int(round(target_dt / dt)))
...
if ds_factor > 1:
    neural_trial = downsample_by_factor(neural_trial, ds_factor)
```

iii. The AI justified this: "Common time bin size across sessions (93.2 ms, ~10.7 Hz, matching multiscope rate). Single-plane data (~31 Hz) downsampled by 3x averaging."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, using the `image_name`, `start_time`, and `end_time` columns. The AI builds a timeline of which image is being shown at each ophys timepoint. This differs from the reference, which uses `initial_image_name` and `change_image_name` from the trials table.

ii.
```python
stim = ds.stimulus_presentations
stim_cd = stim[stim['stimulus_block_name'].str.contains('change_detection', na=False)]
stim_images = stim_cd[stim_cd['image_name'] != 'omitted']
...
image_idx = get_image_at_timepoints(trial_stim_no_omit, trial_ophys_ts, image_to_idx)
```

iii. The AI used the stimulus_presentations table to track image identity at fine temporal resolution, including during gray screen intervals where the last shown image identity persists.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global sorted mapping of 8 unique image names. At each ophys timepoint, the most recently presented image is assigned. During gray screens, the previous image persists. Omitted stimuli retain the previous image identity. For single-plane data, categorical downsampling uses the mode of each bin.

ii.
```python
def get_image_at_timepoints(stim_presentations, ophys_timestamps, image_to_idx):
    image_indices = np.zeros(len(ophys_timestamps), dtype=np.int64)
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

iii. The AI chose to track image identity via the stimulus_presentations table for fine-grained temporal resolution. Global sorted mapping ensures consistency across sessions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at each ophys timepoint within the trial window, using the same `frame_indices` as the neural data. If downsampling is applied, categorical downsampling (mode) is used.

ii.
```python
image_idx = get_image_at_timepoints(trial_stim_no_omit, trial_ophys_ts, image_to_idx)
if ds_factor > 1:
    image_idx = downsample_categorical_by_factor(image_idx, ds_factor)
```

iii. Same timepoints as neural data ensure alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column in the `stimulus_presentations` table, combined with stimulus `start_time`. This differs from the reference which uses the `go` column and `change_time` from the trials table.

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

iii. The AI uses stimulus_presentations `is_change` flag to identify actual image changes. The 750ms window corresponds to one flash (250ms) plus gray interval (500ms).

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary signal is created: 1 during the 750ms window following each image change, 0 otherwise. For single-plane data, after downsampling by averaging, the signal is re-binarized (`> 0.0`).

ii.
```python
mask = (ophys_timestamps >= change_time) & (ophys_timestamps < change_time + 0.75)
change_signal[mask] = 1.0
...
if ds_factor > 1:
    change_signal = downsample_by_factor(change_signal, ds_factor)
    change_signal = (change_signal > 0.0).astype(np.float32)
```

iii. The 750ms window captures one stimulus flash plus gray period.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is binary (0/1). No additional thresholding is needed beyond the 750ms window. After downsampling, any bin with any non-zero value is set to 1.

ii.
```python
change_signal = (change_signal > 0.0).astype(np.float32)
```

iii. Simple binary variable.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The change signal is computed at each ophys timepoint within the trial window, using the same `frame_indices` as neural data. Downsampling is applied if needed.

ii.
```python
change_signal = get_image_change_signal(trial_stim, trial_ophys_ts)
if ds_factor > 1:
    change_signal = downsample_by_factor(change_signal, ds_factor)
```

iii. Same alignment as neural data via shared ophys timepoints.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `dataset.running_speed`, using the `speed` and `timestamps` columns.

ii.
```python
rs = ds.running_speed
running_at_ophys = interpolate_to_ophys(
    rs['timestamps'].values, rs['speed'].values, ophys_ts
)
```

iii. The `running_speed` attribute is the SDK's standard interface for locomotion data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated to ophys timestamps. NaN values are replaced with 0 immediately after interpolation (before collecting for percentile computation). Then it is discretized into 5 percentile-based bins using global boundaries.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    rs['timestamps'].values, rs['speed'].values, ophys_ts
)
running_at_ophys = np.nan_to_num(running_at_ophys, nan=0.0)
all_running_speeds.extend(running_at_ophys.tolist())
...
running_percentiles = np.percentile(running_arr, np.linspace(0, 100, n_bins + 1)[1:-1])
running_binned = discretize_to_percentile_bins(out_raw['running'], n_bins, running_percentiles)
```

iii. Linear interpolation resamples to ophys timebase. NaN replaced with 0. Global percentile bins ensure consistent categories. The AI collects running speeds from ALL ophys timepoints (not just within trials) for percentile computation.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal-percentile bins using `np.digitize` with 4 boundary values (20th, 40th, 60th, 80th percentiles). Values below the 20th percentile go to bin 0, above 80th to bin 4.

ii.
```python
running_percentiles = np.percentile(running_arr, np.linspace(0, 100, n_bins + 1)[1:-1])
def discretize_to_percentile_bins(values, n_bins, percentiles):
    result = np.digitize(values, percentiles)
    result = np.clip(result, 0, n_bins - 1)
    return result.astype(np.int64)
```

iii. Percentile-based binning ensures roughly equal class counts.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to ophys timestamps before trial segmentation, so it shares the same frame indices as neural data.

ii.
```python
running_at_ophys = interpolate_to_ophys(rs['timestamps'].values, rs['speed'].values, ophys_ts)
...
running_trial = running_at_ophys[frame_indices]
```

iii. Interpolation to ophys timebase guarantees alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `dataset.eye_tracking`, using the `pupil_width` column.

ii.
```python
et = ds.eye_tracking
pupil_vals = et['pupil_width'].values
pupil_ts = et['timestamps'].values
pupil_at_ophys = interpolate_to_ophys(pupil_ts, pupil_vals, ophys_ts)
```

iii. `pupil_width` is used as the diameter measure, matching the reference.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter is interpolated to ophys timestamps. Notably, the AI does NOT explicitly remove blink frames before interpolation (unlike the reference which filters on `likely_blink`). The `interpolate_to_ophys` function removes NaN values before interpolation, which partially handles blinks if they produce NaN. NaN values in the final signal are filled via nearest valid value interpolation. Then discretized into 5 percentile bins.

ii.
```python
def interpolate_to_ophys(signal_timestamps, signal_values, ophys_timestamps):
    valid = ~np.isnan(signal_values)
    f = interpolate.interp1d(
        signal_timestamps[valid], signal_values[valid],
        kind='linear', bounds_error=False, fill_value=np.nan
    )
    return f(ophys_timestamps)
...
# NaN filling during output assembly:
if nan_mask.any():
    valid_indices = np.where(~nan_mask)[0]
    for j in range(len(pupil_clean)):
        if nan_mask[j]:
            dists = np.abs(valid_indices - j)
            nearest = valid_indices[np.argmin(dists)]
            pupil_clean[j] = pupil_clean[nearest]
```

iii. The AI removes NaN values but does not use the `likely_blink` flag. Blink frames that are not NaN would be included in the interpolation, potentially introducing artifacts.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 equal-percentile bins using global boundaries. NaN values are filled via nearest-neighbor before binning. If all values are NaN, the median bin (2) is used.

ii.
```python
pupil_percentiles = np.percentile(pupil_arr, np.linspace(0, 100, n_bins + 1)[1:-1])
...
if nan_mask.all():
    pupil_binned = np.full(len(pupil_vals), 2, dtype=np.float32)
else:
    pupil_binned = discretize_to_percentile_bins(pupil_clean, n_bins, pupil_percentiles)
```

iii. Percentile bins computed globally. NaN handling uses nearest-neighbor fill (reference uses bin 0 for NaN).

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed: pupil is interpolated to ophys timestamps before trial segmentation.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_ts, pupil_vals, ophys_ts)
...
pupil_trial = pupil_at_ophys[frame_indices]
```

iii. Shared ophys timepoints guarantee alignment.

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

iii. These four columns are the SDK's canonical trial outcome labels.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3) and broadcast to all timepoints within a trial. The mapping is: hit=0, miss=1, false_alarm=2, correct_reject=3. Unknown outcomes default to index 0.

ii.
```python
outcome_names = ['hit', 'miss', 'false_alarm', 'correct_reject']
outcome_to_idx = {name: idx for idx, name in enumerate(outcome_names)}
...
outcome_idx = outcome_to_idx.get(outcome, 0)
np.full(n_timepoints, outcome_idx, dtype=np.int64)
```

iii. The mapping order matches the reference. Static per-trial but represented as time-varying for format consistency.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Failed experiment loads**: Caught by try/except, experiment is skipped.
- **Too few trials**: Sessions with < 2 valid trials are skipped.
- **Short trials**: Trials with fewer frames than `ds_factor` are skipped.
- **Running speed NaN**: Replaced with 0.0 before collecting for percentiles.
- **Pupil NaN**: Filled via nearest valid value interpolation. If all NaN, median bin (2) is assigned.
- **Missing eye tracking**: Falls back to all-NaN pupil array.

ii.
```python
try:
    ds = load_experiment(cache, exp_id)
except Exception as e:
    return None, ..., f"Failed to load: {e}"
...
running_at_ophys = np.nan_to_num(running_at_ophys, nan=0.0)
...
if nan_mask.all():
    pupil_binned = np.full(len(pupil_vals), 2, dtype=np.float32)
```

iii. Error handling ensures the pipeline doesn't crash on problematic sessions. NaN handling strategies differ from the reference (which maps NaN to bin 0).

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each experiment via `cache.get_behavior_ophys_experiment()` (I/O bound). Additionally, the `get_image_at_timepoints` function iterates over every ophys timepoint in a Python loop, which is computationally expensive.

ii.
```python
ds = load_experiment(cache, exp_id)
...
for t_idx, t in enumerate(ophys_timestamps):  # O(n_timepoints) Python loop
    while stim_idx < len(stim_starts) - 1 and stim_starts[stim_idx + 1] <= t:
        stim_idx += 1
```

iii. Data loading is I/O bound. The Python-level loop over ophys timepoints in `get_image_at_timepoints` is a performance bottleneck compared to vectorized alternatives.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `get_image_at_timepoints` function iterates over every ophys timepoint in a Python for-loop. This could be vectorized using `np.searchsorted` to find the stimulus index for each timepoint. The nearest-neighbor NaN fill loop in pupil processing could also be vectorized.

ii.
```python
for t_idx, t in enumerate(ophys_timestamps):
    while stim_idx < len(stim_starts) - 1 and stim_starts[stim_idx + 1] <= t:
        stim_idx += 1
    ...
# Also:
for j in range(len(pupil_clean)):
    if nan_mask[j]:
        dists = np.abs(valid_indices - j)
        nearest = valid_indices[np.argmin(dists)]
        pupil_clean[j] = pupil_clean[nearest]
```

iii. Both loops iterate over potentially thousands of timepoints in pure Python. Vectorized numpy operations would be significantly faster.

## 9-c. What processing does the code repeat multiple times?

i. The AI collects running speed and pupil diameter values from ALL ophys timepoints (not just trial timepoints) in the first pass, then extracts only trial timepoints again during the per-trial processing. The image discovery is done from one experiment only, rather than collecting from all.

ii.
```python
all_running_speeds.extend(running_at_ophys.tolist())  # ALL timepoints
...
running_trial = running_at_ophys[frame_indices]  # trial timepoints only
```

iii. Collecting ALL timepoints for percentile computation (including inter-trial intervals) is redundant and slightly different from the reference which computes percentiles from trial data only.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI downsamples single-plane data by a factor of 3, which is unnecessary processing not done in the reference. The AI also computes and stores running speed and pupil data for all ophys timepoints, but only trial segments are used. The `downsample_categorical_by_factor` function computes modes per bin which is more expensive than necessary.

ii.
```python
if ds_factor > 1:
    neural_trial = downsample_by_factor(neural_trial, ds_factor)
    image_idx = downsample_categorical_by_factor(image_idx, ds_factor)
    change_signal = downsample_by_factor(change_signal, ds_factor)
    running_trial = downsample_by_factor(running_trial, ds_factor)
    pupil_trial = downsample_by_factor(pupil_trial, ds_factor)
```

iii. The downsampling step processes data that the reference would keep at native resolution. The reference does not resample to a common frame rate.
