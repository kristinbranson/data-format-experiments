# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads a local experiment metadata CSV from the Allen Visual Behavior data directory, finds which `ophys_experiment_id` values have NWB files present, filters that table to `active_behavior` and `Familiar` experiments, and then loads each experiment individually through `VisualBehaviorOphysProjectCache.from_s3_cache(...).get_behavior_ophys_experiment(...)`. It does not use the SDK experiment table as the canonical list of all VisualBehavior experiments, and it does not reconstruct multi-plane sessions from shared `ophys_session_id`.

ii.
```python
exp_table = pd.read_csv(
    os.path.join(data_dir, 'visual-behavior-ophys-1.1.0/project_metadata/ophys_experiment_table.csv')
)
...
nwb_files = os.listdir(nwb_dir)
exp_ids = [int(re.search(r'(\d+)', f).group(1)) for f in nwb_files]
...
filtered = available[
    (available['behavior_type'] == 'active_behavior') &
    (available['experience_level'] == 'Familiar')
].copy()
...
cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=data_dir)
...
ds = load_experiment(cache, exp_id)
```

iii. The justification is in `CONVERSION_NOTES.md`: the AI explicitly chose active behavior only, familiar-image sessions only, and both Scientifica and Multiscope recordings. It says this follows the paper's neural analysis and gives a larger dataset, yielding 110 experiments from 38 mice.

## 1-b. How are the data split into subjects?

i. Subjects are defined by unique `mouse_id` values from the filtered experiment table. The final `subjects` list is the sorted set of mouse IDs converted to strings, and each experiment/session gets a subject index from that mapping.

ii.
```python
all_mice = sorted(set(r['exp_info']['mouse_id'] for r in experiment_results))
mouse_to_idx = {m: i for i, m in enumerate(all_mice)}
subjects = [str(m) for m in all_mice]
...
subject_idx_list.append(mouse_to_idx[exp_info['mouse_id']])
```

iii. There is no separate long justification beyond using the experiment metadata table. The AI's notes treat mice as the top-level subject grouping and report totals in terms of mice.

## 1-c. How are the data split into sessions?

i. The AI effectively treats each `ophys_experiment_id` as one session. It iterates row-by-row through the filtered experiment table and appends one top-level entry to `neural`, `input`, and `output` per processed experiment. It does not group multiple experiments belonging to the same `ophys_session_id`.

ii.
```python
for i, (_, exp_info) in enumerate(exp_table.iterrows()):
    exp_id = int(exp_info['ophys_experiment_id'])
    ...
    experiment_results.append({
        'exp_info': exp_info,
        'neural_trials': neural_trials,
        'output_trials': output_trials,
        'trial_outcomes': trial_outcomes,
        'n_neurons': n_neurons,
        'brain_region': brain_region,
    })
...
for result in experiment_results:
    ...
    neural_all.append(session_neural)
    input_all.append(session_input)
    output_all.append(session_output)
```

iii. The notes justify including both single-plane and multi-plane data, but they do not justify collapsing the dataset to experiment-level rather than reconstructing SDK sessions. That choice is mainly visible in the code structure.

## 1-d. How are the data split into trials?

i. Trials come from `ds.trials`. The AI keeps go and catch trials, excludes aborted and auto-rewarded trials, and uses the full interval from `start_time` to `stop_time`. Trial frames are selected by masking ophys timestamps that fall within that interval, so trials are variable length.

ii.
```python
trials = ds.trials
valid_trials = trials[
    ((trials['go'] == True) | (trials['catch'] == True)) &
    (trials['aborted'] == False) &
    (trials['auto_rewarded'] == False)
]
...
t_start = trial['start_time']
t_stop = trial['stop_time']
frame_mask = (ophys_ts >= t_start) & (ophys_ts <= t_stop)
frame_indices = np.where(frame_mask)[0]
```

iii. `CONVERSION_NOTES.md` says the AI intentionally included go and catch trials, excluded aborted and auto-rewarded trials, and used the task-defined trial structure from the Allen dataset.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by requiring `go` or `catch`, excluding `aborted` and `auto_rewarded`, skipping trials with too few ophys frames to survive the chosen downsampling factor, and later skipping experiments with fewer than two usable trials.

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
    return None, None, None, None, None, f"Only {len(neural_trials)} usable trials after processing"
```

iii. The AI justifies the first three filters in `CONVERSION_NOTES.md` as matching the go/catch task while excluding premature-lick and free-reward trials. It also notes a minimum-trial requirement for sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from the AllenSDK `events` table, specifically the `events` column, not from `dff_traces`.

ii.
```python
events = ds.events
neural_full = np.vstack(events['events'].values).astype(np.float32)
```

iii. The notes explicitly justify this as following the paper's statement that it used detected calcium events rather than raw dF/F traces.

## 2-b. How is the `neural` data processed?

i. The neural traces are stacked within a single experiment, sliced trial-by-trial, and optionally downsampled by averaging by a factor chosen to make the final time step about 93.2 ms. The AI does not merge multiple imaging planes into one session.

ii.
```python
dt = np.median(np.diff(ophys_ts))
ds_factor = max(1, int(round(target_dt / dt)))
...
neural_trial = neural_full[:, frame_indices]
...
if ds_factor > 1:
    neural_trial = downsample_by_factor(neural_trial, ds_factor)
```

iii. `CONVERSION_NOTES.md` says the AI wanted a common 93.2 ms bin matching Multiscope, with single-plane data downsampled 3x by averaging.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies no explicit post-load neural filtering of its own. It relies on whatever ROI filtering the AllenSDK already applied before exposing `ds.events`.

ii.
```python
events = ds.events
neural_full = np.vstack(events['events'].values).astype(np.float32)
...
if n_neurons == 0:
    return None, None, None, None, None, "No neurons"
```

iii. The notes explicitly say ROI filtering is left to the AllenSDK processing pipeline and that only valid cell bodies should remain.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are first aligned to `ophys_timestamps`, then each trial extracts the segment between `start_time` and `stop_time`. In metadata, the AI describes the alignment event as trial start.

ii.
```python
ophys_ts = ds.ophys_timestamps
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts <= t_stop)
frame_indices = np.where(frame_mask)[0]
...
'temporal_alignment_event': 'Trial start (aligned to ophys timestamps)',
```

iii. The notes emphasize that all signals are aligned to ophys timestamps and that trial boundaries come from the behavior task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI forces all data to a common 93.2 ms time bin (`~10.7 Hz`). Single-plane data are downsampled by about 3x; Multiscope-like data usually stay at factor 1.

ii.
```python
target_dt = 0.0932  # seconds
...
ds_factor = max(1, int(round(target_dt / dt)))
...
if ds_factor > 1:
    neural_trial = downsample_by_factor(neural_trial, ds_factor)
...
'time_bin_size': target_dt * 1000,
```

iii. The notes explicitly justify this as matching the Multiscope frame rate and making a common time base across equipment types.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `ds.stimulus_presentations`, using `image_name`, `start_time`, and the change-detection stimulus block. Omitted stimuli are excluded from the image identity update logic.

ii.
```python
stim = ds.stimulus_presentations
stim_cd = stim[stim['stimulus_block_name'].str.contains('change_detection', na=False)].copy()
...
stim_starts = stim_presentations['start_time'].values
stim_images = stim_presentations['image_name'].values
...
trial_stim_no_omit = trial_stim[trial_stim['image_name'] != 'omitted']
image_idx = get_image_at_timepoints(trial_stim_no_omit, trial_ophys_ts, image_to_idx)
```

iii. The notes justify this by saying image identity should be the most recently presented non-omitted image and should persist through gray periods.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI creates a global `image_to_idx` map from image names found in the first experiment, then for each ophys time point in a trial it assigns the most recently presented non-omitted image. If trials are downsampled, the categorical image label is downsampled by taking the mode across each chunk.

ii.
```python
all_image_names = sorted([n for n in stim_cd['image_name'].unique()
                          if n != 'omitted' and isinstance(n, str)])
image_to_idx = {name: idx for idx, name in enumerate(all_image_names)}
...
image_idx = get_image_at_timepoints(trial_stim_no_omit, trial_ophys_ts, image_to_idx)
...
if ds_factor > 1:
    image_idx = downsample_categorical_by_factor(image_idx, ds_factor)
```

iii. The notes explicitly state that image identity is carried forward across gray screens, omitted stimuli keep the previous image, and categories correspond to the eight natural images.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed directly on the same per-trial ophys timestamps used to slice neural data, and then downsampled in lockstep with neural data when `ds_factor > 1`.

ii.
```python
trial_ophys_ts = ophys_ts[frame_indices]
image_idx = get_image_at_timepoints(trial_stim_no_omit, trial_ophys_ts, image_to_idx)
...
if ds_factor > 1:
    neural_trial = downsample_by_factor(neural_trial, ds_factor)
    image_idx = downsample_categorical_by_factor(image_idx, ds_factor)
```

iii. The notes say all signals share the ophys timebase, so image identity is aligned frame-by-frame with neural activity.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations`, specifically the `is_change` flag and each change presentation's `start_time`.

ii.
```python
changes = stim_presentations[stim_presentations['is_change'] == True]
for _, change in changes.iterrows():
    change_time = change['start_time']
```

iii. The notes justify this as marking the interval immediately following a true image change and not sham changes.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI creates a binary vector on the ophys timebase and sets it to 1 for 750 ms after each stimulus presentation marked as a change. After downsampling it rebinarizes the result with `> 0`.

ii.
```python
change_signal = np.zeros(len(ophys_timestamps), dtype=np.float32)
...
mask = (ophys_timestamps >= change_time) & (ophys_timestamps < change_time + 0.75)
change_signal[mask] = 1.0
...
if ds_factor > 1:
    change_signal = downsample_by_factor(change_signal, ds_factor)
    change_signal = (change_signal > 0.0).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` says the 750 ms window is meant to cover one flashed image plus the following gray interval, and only actual changes should be marked.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is a binary category with values `0` and `1`. There is no additional thresholding beyond setting the 750 ms window to 1 and, after downsampling, converting any nonzero averaged value back to 1.

ii.
```python
change_signal = np.zeros(len(ophys_timestamps), dtype=np.float32)
...
change_signal[mask] = 1.0
...
change_signal = (change_signal > 0.0).astype(np.float32)
```

iii. The notes describe this output as a binary signal indicating whether the trial is in the post-change interval.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is computed on the same per-trial ophys timestamps as neural data and downsampled along with the neural trial.

ii.
```python
trial_ophys_ts = ophys_ts[frame_indices]
change_signal = get_image_change_signal(trial_stim, trial_ophys_ts)
...
if ds_factor > 1:
    neural_trial = downsample_by_factor(neural_trial, ds_factor)
    change_signal = downsample_by_factor(change_signal, ds_factor)
```

iii. The notes say all variables are aligned to ophys timestamps, so the binary change signal is frame-aligned with the neural data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `ds.running_speed`, using the `timestamps` and `speed` columns.

ii.
```python
rs = ds.running_speed
running_at_ophys = interpolate_to_ophys(
    rs['timestamps'].values, rs['speed'].values, ophys_ts
)
```

iii. The notes state that the AI uses the AllenSDK filtered running speed signal.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated to ophys timestamps, NaNs are filled with `0.0`, and if necessary the trial trace is downsampled by averaging to the common 93.2 ms bin size.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    rs['timestamps'].values, rs['speed'].values, ophys_ts
)
running_at_ophys = np.nan_to_num(running_at_ophys, nan=0.0)
...
running_trial = running_at_ophys[frame_indices]
...
if ds_factor > 1:
    running_trial = downsample_by_factor(running_trial, ds_factor)
```

iii. The notes justify interpolation to the ophys clock, averaging for downsampling, and NaN-to-zero handling for missing running values.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The AI computes global percentile boundaries across all collected running values, then bins each time point into one of five categories using `np.digitize`.

ii.
```python
running_percentiles = np.percentile(running_arr, np.linspace(0, 100, n_bins + 1)[1:-1])
...
running_binned = discretize_to_percentile_bins(
    out_raw['running'], n_bins, running_percentiles
).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` explicitly says running speed is discretized into five equal-percentile bins using global percentile boundaries.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is first interpolated onto the ophys timebase, then trial slices use the same frame indices as neural data, and then both are downsampled together if needed.

ii.
```python
running_at_ophys = interpolate_to_ophys(..., ophys_ts)
...
running_trial = running_at_ophys[frame_indices]
neural_trial = neural_full[:, frame_indices]
...
if ds_factor > 1:
    neural_trial = downsample_by_factor(neural_trial, ds_factor)
    running_trial = downsample_by_factor(running_trial, ds_factor)
```

iii. The notes justify this by using the common ophys timebase for all signals.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `ds.eye_tracking`, using the `pupil_width` and `timestamps` columns.

ii.
```python
et = ds.eye_tracking
pupil_vals = et['pupil_width'].values
pupil_ts = et['timestamps'].values
```

iii. The notes explicitly say the AI uses `pupil_width` as the pupil diameter measure.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil width is linearly interpolated to ophys timestamps. If eye tracking is unavailable, the AI fills the whole session with NaNs. Later, after trial slicing and optional averaging downsampling, NaNs are replaced by the nearest valid value within the trial; if an entire trial is NaN, it is assigned the median bin later.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_ts, pupil_vals, ophys_ts)
...
if ds_factor > 1:
    pupil_trial = downsample_by_factor(pupil_trial, ds_factor)
...
nan_mask = np.isnan(pupil_vals)
if nan_mask.all():
    pupil_binned = np.full(len(pupil_vals), 2, dtype=np.float32)
else:
    ...
    if nan_mask.any():
        valid_indices = np.where(~nan_mask)[0]
        for j in range(len(pupil_clean)):
            if nan_mask[j]:
                dists = np.abs(valid_indices - j)
                nearest = valid_indices[np.argmin(dists)]
                pupil_clean[j] = pupil_clean[nearest]
```

iii. The notes justify interpolation to the ophys clock and nearest-valid-value filling for blink artifacts or tracking failures.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The AI computes global percentile boundaries from all non-NaN pupil values and bins each trial's pupil trace into five categories after its nearest-neighbor NaN filling step.

ii.
```python
valid_pupil = pupil_at_ophys[~np.isnan(pupil_at_ophys)]
all_pupil_diameters.extend(valid_pupil.tolist())
...
pupil_percentiles = np.percentile(pupil_arr, np.linspace(0, 100, n_bins + 1)[1:-1])
...
pupil_binned = discretize_to_percentile_bins(
    pupil_clean, n_bins, pupil_percentiles
).astype(np.float32)
```

iii. The notes explicitly describe five equal-percentile bins and nearest-value filling for missing pupil samples.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil data are interpolated to the full-session ophys timebase, sliced using the same per-trial frame indices as neural data, and then downsampled together when needed.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_ts, pupil_vals, ophys_ts)
...
pupil_trial = pupil_at_ophys[frame_indices]
neural_trial = neural_full[:, frame_indices]
...
if ds_factor > 1:
    neural_trial = downsample_by_factor(neural_trial, ds_factor)
    pupil_trial = downsample_by_factor(pupil_trial, ds_factor)
```

iii. The notes say all signals share the ophys timestamp basis.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome comes from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the Allen trials table.

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
```

iii. The notes explicitly describe the four outcome categories and their mapping.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI converts the boolean trial outcome to a string label, maps that label to a fixed integer code, and then repeats that code across all time bins in the trial so the output array has a consistent `(n_output, n_timepoints)` shape.

ii.
```python
outcome_names = ['hit', 'miss', 'false_alarm', 'correct_reject']
outcome_to_idx = {name: idx for idx, name in enumerate(outcome_names)}
...
outcome = out_raw['outcome']
outcome_idx = outcome_to_idx.get(outcome, 0)
...
np.full(n_timepoints, outcome_idx, dtype=np.int64)
```

iii. The notes justify this as a static per-trial variable represented as time-varying for format consistency.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles failures by skipping experiments that cannot be loaded or processed, skips trials with too few frames after time-base harmonization, fills missing running values with zero, uses all-NaN arrays if eye tracking is absent, and later fills pupil NaNs with nearest valid values or assigns a median bin if a trial has no pupil data at all.

ii.
```python
try:
    ds = load_experiment(cache, exp_id)
except Exception as e:
    return None, None, None, None, None, f"Failed to load: {e}"
...
if len(frame_indices) < ds_factor:
    continue
...
running_at_ophys = np.nan_to_num(running_at_ophys, nan=0.0)
...
except Exception:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
...
if nan_mask.all():
    pupil_binned = np.full(len(pupil_vals), 2, dtype=np.float32)
```

iii. The notes explicitly justify NaN-to-zero for running and nearest-value interpolation for pupil failures. The rest is mostly visible from the code rather than separately argued.

## 9-a. What are the most time-consuming steps of the code?

i. The dominant costs are loading each Allen experiment through the SDK and then looping through every trial, every ophys time point for image assignment, and every change event for change-signal construction.

ii.
```python
for i, (_, exp_info) in enumerate(exp_table.iterrows()):
    ...
    result = process_experiment(...)
...
for trial_idx, (_, trial) in enumerate(valid_trials.iterrows()):
    ...
    image_idx = get_image_at_timepoints(...)
    change_signal = get_image_change_signal(...)
```

iii. There is no explicit runtime analysis in the notes beyond reporting dataset size. This characterization is inferred from the code structure and the use of SDK experiment loads.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are scalar and could be vectorized: the per-trial `iterrows()` loop, the per-timepoint loop in `get_image_at_timepoints`, the per-change loop in `get_image_change_signal`, the chunk loop in `downsample_categorical_by_factor`, and the per-sample nearest-neighbor pupil fill loop.

ii.
```python
for t_idx, t in enumerate(ophys_timestamps):
    ...
for _, change in changes.iterrows():
    ...
for i in range(n_new):
    chunk = data[i*factor:(i+1)*factor]
...
for j in range(len(pupil_clean)):
    if nan_mask[j]:
        ...
for trial_idx, (_, trial) in enumerate(valid_trials.iterrows()):
    ...
```

iii. The AI did not explicitly discuss vectorization tradeoffs in its notes. This section is reconstructed directly from the implemented code.

## 9-c. What processing does the code repeat multiple times?

i. The code repeatedly filters stimulus presentations and rebuilds per-trial image/change traces inside the trial loop, repeatedly performs nearest-neighbor searches for each missing pupil sample, and scans full-session traces first to collect global percentile statistics and then again when assembling outputs.

ii.
```python
for trial_idx, (_, trial) in enumerate(valid_trials.iterrows()):
    trial_stim = stim_cd[
        (stim_cd['start_time'] >= t_start - 0.5) &
        (stim_cd['start_time'] <= t_stop + 0.5)
    ]
    ...
    image_idx = get_image_at_timepoints(...)
    change_signal = get_image_change_signal(...)
...
all_running_speeds.extend(running_at_ophys.tolist())
all_pupil_diameters.extend(valid_pupil.tolist())
...
for result in experiment_results:
    ...
    running_binned = discretize_to_percentile_bins(...)
    pupil_binned = discretize_to_percentile_bins(...)
```

iii. The notes do not call out repeated processing. This is inferred from the implementation.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores `trial_outcomes` in `experiment_results` but never uses that stored list later, computes `stim_ends` in `get_image_at_timepoints` without using it, and collects running/pupil values from full sessions for percentile estimation even though downstream decoding uses only trial windows.

ii.
```python
stim_ends = stim_presentations['end_time'].values
...
experiment_results.append({
    'exp_info': exp_info,
    'neural_trials': neural_trials,
    'output_trials': output_trials,
    'trial_outcomes': trial_outcomes,
    'n_neurons': n_neurons,
    'brain_region': brain_region,
})
...
all_running_speeds.extend(running_at_ophys.tolist())
all_pupil_diameters.extend(valid_pupil.tolist())
```

iii. There is no explicit justification for these inefficiencies in the notes. They are visible only from the code.
