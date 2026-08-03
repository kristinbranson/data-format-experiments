# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script reads the Allen metadata CSV `ophys_experiment_table.csv`, intersects it with the locally present NWB filenames, filters to `active_behavior` and `Familiar`, and then loads each remaining `ophys_experiment_id` through `VisualBehaviorOphysProjectCache.get_behavior_ophys_experiment()`.

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
    return filtered

cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=data_dir)
...
result = process_experiment(cache, exp_info, all_image_names, image_to_idx,
                            target_dt, all_running_speeds, all_pupil_diameters)
```

iii. `CONVERSION_NOTES.md` says the agent intentionally used active-behavior familiar-image data and included both single-plane and multiscope recordings. The trajectory says it rejected a stricter familiar-multiscope-only subset because the local bundle only exposed 22 such experiments from 1 mouse.

## 1-b. How are the data split into subjects?

i. Subjects are defined by unique `mouse_id` values from the filtered experiment table. The final `subjects` list is sorted, and each loaded session stores an integer `subject_idx`.

ii.
```python
all_mice = sorted(set(r['exp_info']['mouse_id'] for r in experiment_results))
mouse_to_idx = {m: i for i, m in enumerate(all_mice)}
subjects = [str(m) for m in all_mice]
...
subject_idx_list.append(mouse_to_idx[exp_info['mouse_id']])
```

iii. This matches the Allen metadata terminology. The notes also summarize the result as 38 mice after filtering.

## 1-c. How are the data split into sessions?

i. The script treats each `ophys_experiment_id` as one output session. It does not group rows that share an `ophys_session_id`; multiplane sessions are therefore split into separate plane-level sessions.

ii.
```python
for i, (_, exp_info) in enumerate(exp_table.iterrows()):
    exp_id = int(exp_info['ophys_experiment_id'])
    ...
    experiment_results.append({
        'exp_info': exp_info,
        'neural_trials': neural_trials,
        'output_trials': output_trials,
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

iii. The trajectory explicitly justifies this by saying each imaging plane has different neurons, so each experiment should be treated as a separate “session.” That is the agent’s main structural choice.

## 1-d. How are the data split into trials?

i. Trials come directly from `ds.trials`. For each valid row, the script uses `start_time` and `stop_time` to take the ophys frames inside that interval, producing variable-length trials.

ii.
```python
trials = ds.trials
valid_trials = trials[
    ((trials['go'] == True) | (trials['catch'] == True)) &
    (trials['aborted'] == False) &
    (trials['auto_rewarded'] == False)
]
...
for trial_idx, (_, trial) in enumerate(valid_trials.iterrows()):
    t_start = trial['start_time']
    t_stop = trial['stop_time']
    frame_mask = (ophys_ts >= t_start) & (ophys_ts <= t_stop)
    frame_indices = np.where(frame_mask)[0]
```

iii. The trajectory shows the agent inspected the SDK `trials` table and decided to use the full trial window so outputs such as image identity and image change could vary within the trial.

## 1-e. How are trials filtered based on quality controls?

i. Trials are restricted to Go or Catch and filtered to exclude aborted and auto-rewarded trials. Trials with fewer than `ds_factor` frames are skipped, and experiments with fewer than 2 valid/usable trials are dropped.

ii.
```python
valid_trials = trials[
    ((trials['go'] == True) | (trials['catch'] == True)) &
    (trials['aborted'] == False) &
    (trials['auto_rewarded'] == False)
]

if len(valid_trials) < 2:
    return None, None, None, None, None, f"Only {len(valid_trials)} valid trials"
...
if len(frame_indices) < ds_factor:
    continue
...
if len(neural_trials) < 2:
    return None, None, None, None, None, f"Only {len(neural_trials)} usable trials after processing"
```

iii. `CONVERSION_NOTES.md` gives the rationale: Go and Catch are the instructed trial types, aborted trials have no valid change epoch, auto-rewarded trials are behaviorally atypical, and sessions need at least two usable trials for decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `ds.events['events']`, not from `dff_traces`.

ii.
```python
events = ds.events
neural_full = np.vstack(events['events'].values).astype(np.float32)
```

iii. The notes and trajectory both cite the paper language that neural analyses used detected calcium events rather than raw fluorescence or dF/F.

## 2-b. How is the `neural` data processed?

i. The script stacks all event traces for the experiment into a neuron-by-time matrix, slices them trial-by-trial, and for single-plane data optionally downsamples by averaging consecutive frames to match a 93.2 ms common bin.

ii.
```python
neural_full = np.vstack(events['events'].values).astype(np.float32)
...
neural_trial = neural_full[:, frame_indices]
...
if ds_factor > 1:
    neural_trial = downsample_by_factor(neural_trial, ds_factor)
...
neural_trials.append(neural_trial.astype(np.float32))
```

iii. The notes say the averaging was introduced to put single-plane and multiscope recordings on a common time base, with 93.2 ms chosen to match the multiscope rate.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit extra neuron-level filter in `convert_data.py`. The script accepts whatever ROIs are already exposed by `ds.events`, and only skips experiments with zero neurons.

ii.
```python
events = ds.events
neural_full = np.vstack(events['events'].values).astype(np.float32)
...
n_neurons = neural_full.shape[0]
if n_neurons == 0:
    return None, None, None, None, None, "No neurons"
```

iii. `CONVERSION_NOTES.md` says the agent relied on the Allen pipeline’s ROI QC rather than adding another filter in the conversion script.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to the ophys time base first, then segmented from trial `start_time` to `stop_time`. In the saved metadata, the trial alignment event is described as “Trial start (aligned to ophys timestamps).”

ii.
```python
ophys_ts = ds.ophys_timestamps
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts <= t_stop)
frame_indices = np.where(frame_mask)[0]
...
'temporal_alignment_event': 'Trial start (aligned to ophys timestamps)',
'off_start': 0.0,
'off_end': None,
```

iii. The trajectory describes this as using the imaging frame times as the common clock and then cutting out the full trial window for each trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The saved data use a common 93.2 ms bin (`~10.7 Hz`). Multiscope recordings stay at that effective rate; single-plane recordings are rebinned by a factor of 3, with averaging for continuous/event traces and mode for categorical outputs.

ii.
```python
target_dt = 0.0932  # seconds
...
ds_factor = max(1, int(round(target_dt / dt)))
...
if ds_factor > 1:
    neural_trial = downsample_by_factor(neural_trial, ds_factor)
    image_idx = downsample_categorical_by_factor(image_idx, ds_factor)
    change_signal = downsample_by_factor(change_signal, ds_factor)
    change_signal = (change_signal > 0.0).astype(np.float32)
    running_trial = downsample_by_factor(running_trial, ds_factor)
    pupil_trial = downsample_by_factor(pupil_trial, ds_factor)
...
'time_bin_size': target_dt * 1000,
```

iii. The notes and trajectory justify 93.2 ms by the measured MESO.1 frame interval and say this was done so all sessions would share one decoder time bin size.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from the `stimulus_presentations` table, specifically the `change_detection` stimulus block’s `image_name`, `start_time`, and `end_time` columns.

ii.
```python
stim = ds.stimulus_presentations
stim_cd = stim[stim['stimulus_block_name'].str.contains('change_detection', na=False)].copy()
trial_stim = stim_cd[
    (stim_cd['start_time'] >= t_start - 0.5) &
    (stim_cd['start_time'] <= t_stop + 0.5)
]
trial_stim_no_omit = trial_stim[trial_stim['image_name'] != 'omitted']
...
stim_starts = stim_presentations['start_time'].values
stim_ends = stim_presentations['end_time'].values
stim_images = stim_presentations['image_name'].values
```

iii. The agent inspected the updated AllenSDK warning about `stimulus_block_name` and intentionally restricted to the block containing `change_detection`, rather than using all stimulus rows.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to global integer IDs. Within each trial, the script walks through ophys timestamps and assigns the most recent non-omitted image; during gray periods and omissions, the previous image identity is carried forward.

ii.
```python
all_image_names = sorted([n for n in stim_cd['image_name'].unique()
                          if n != 'omitted' and isinstance(n, str)])
image_to_idx = {name: idx for idx, name in enumerate(all_image_names)}
...
for t_idx, t in enumerate(ophys_timestamps):
    while stim_idx < len(stim_starts) - 1 and stim_starts[stim_idx + 1] <= t:
        stim_idx += 1
    if stim_idx < len(stim_starts) and stim_starts[stim_idx] <= t:
        img = stim_images[stim_idx]
        if img in image_to_idx and img != 'omitted':
            current_image_idx = image_to_idx[img]
    if current_image_idx >= 0:
        image_indices[t_idx] = current_image_idx
```

iii. `CONVERSION_NOTES.md` states that the output is meant to represent the currently relevant image, with the last image persisting through the gray screen and omitted flashes.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is computed directly on each trial’s `trial_ophys_ts`, so it has one categorical value per neural time bin.

ii.
```python
trial_ophys_ts = ophys_ts[frame_indices]
...
image_idx = get_image_at_timepoints(trial_stim_no_omit, trial_ophys_ts, image_to_idx)
...
output_trial = np.stack([
    image_idx.astype(np.int64),
    ...
], axis=0)
```

iii. The code and notes both frame this as image identity aligned onto the ophys time base before saving trial outputs.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. `image_change` is derived from `stimulus_presentations['is_change']` and the corresponding stimulus `start_time` values in the change-detection block.

ii.
```python
changes = stim_presentations[stim_presentations['is_change'] == True]
for _, change in changes.iterrows():
    change_time = change['start_time']
```

iii. The notes explicitly say the agent wanted actual image changes only, not sham changes from catch trials.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The script creates a binary vector over ophys bins and sets bins to 1 for 750 ms starting at each changed image’s onset.

ii.
```python
change_signal = np.zeros(len(ophys_timestamps), dtype=np.float32)
...
mask = (ophys_timestamps >= change_time) & (ophys_timestamps < change_time + 0.75)
change_signal[mask] = 1.0
```

iii. `CONVERSION_NOTES.md` justifies 750 ms as one full flash cycle: 250 ms image plus 500 ms gray.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is binary. After optional downsampling, any averaged value greater than 0 is converted back to category 1 (`change`); otherwise it is category 0 (`no_change`).

ii.
```python
if ds_factor > 1:
    change_signal = downsample_by_factor(change_signal, ds_factor)
    change_signal = (change_signal > 0.0).astype(np.float32)
...
'output_values': [
    ...,
    ['no_change', 'change'],
    ...
]
```

iii. The trajectory shows the agent treated this as a categorical indicator rather than a continuous signal.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is generated on the same `trial_ophys_ts` vector used to slice the neural trial, so every neural time bin gets a corresponding binary change label.

ii.
```python
trial_ophys_ts = ophys_ts[frame_indices]
change_signal = get_image_change_signal(trial_stim, trial_ophys_ts)
...
output_trial = np.stack([
    image_idx.astype(np.int64),
    change_signal.astype(np.int64),
    ...
], axis=0)
```

iii. The notes describe all outputs as aligned to ophys timestamps, and this signal follows that same rule.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `ds.running_speed`, using its `timestamps` and `speed` columns.

ii.
```python
rs = ds.running_speed
running_at_ophys = interpolate_to_ophys(
    rs['timestamps'].values, rs['speed'].values, ophys_ts
)
```

iii. The agent’s background SDK exploration found that `running_speed` is the AllenSDK’s filtered running-speed table, and the notes say the script uses that filtered version.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The signal is linearly interpolated to the ophys timestamps, NaNs are replaced with 0, and values are later converted to percentile bins using global thresholds collected across the full dataset.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    rs['timestamps'].values, rs['speed'].values, ophys_ts
)
running_at_ophys = np.nan_to_num(running_at_ophys, nan=0.0)
all_running_speeds.extend(running_at_ophys.tolist())
...
running_percentiles = np.percentile(running_arr, np.linspace(0, 100, n_bins + 1)[1:-1])
...
running_binned = discretize_to_percentile_bins(
    out_raw['running'], n_bins, running_percentiles
).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` says the agent used linear interpolation to align running speed and global percentile boundaries so the discrete categories would be consistent across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is discretized into 5 equal-percentile bins using four global percentile boundaries computed from all interpolated running-speed samples across all processed experiments.

ii.
```python
n_bins = 5
running_percentiles = np.percentile(running_arr, np.linspace(0, 100, n_bins + 1)[1:-1])
...
def discretize_to_percentile_bins(values, n_bins, percentiles):
    result = np.digitize(values, percentiles)
    result = np.clip(result, 0, n_bins - 1)
    return result.astype(np.int64)
```

iii. The notes list the actual learned boundaries and state that the bins are global equal-percentile bins.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is first resampled onto the full-session ophys time base, then trialized with the same `frame_indices` used for the neural data.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    rs['timestamps'].values, rs['speed'].values, ophys_ts
)
...
running_trial = running_at_ophys[frame_indices]
```

iii. The notes say all streams are aligned to ophys timestamps, and the code does that before trial segmentation.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `ds.eye_tracking`, using `pupil_width` and `timestamps`.

ii.
```python
et = ds.eye_tracking
pupil_vals = et['pupil_width'].values
pupil_ts = et['timestamps'].values
pupil_at_ophys = interpolate_to_ophys(pupil_ts, pupil_vals, ophys_ts)
```

iii. The notes say the agent treated `pupil_width` as the diameter measure and used the Allen eye-tracking table supplied by the SDK.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The script linearly interpolates `pupil_width` onto ophys timestamps, collects all non-NaN values to compute global percentile boundaries, and then fills per-trial NaNs with the nearest valid value. Trials with all-NaN pupil values are assigned the middle bin.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_ts, pupil_vals, ophys_ts)
valid_pupil = pupil_at_ophys[~np.isnan(pupil_at_ophys)]
all_pupil_diameters.extend(valid_pupil.tolist())
...
nan_mask = np.isnan(pupil_vals)
if nan_mask.all():
    pupil_binned = np.full(len(pupil_vals), 2, dtype=np.float32)
else:
    pupil_clean = pupil_vals.copy()
    if nan_mask.any():
        valid_indices = np.where(~nan_mask)[0]
        for j in range(len(pupil_clean)):
            if nan_mask[j]:
                dists = np.abs(valid_indices - j)
                nearest = valid_indices[np.argmin(dists)]
                pupil_clean[j] = pupil_clean[nearest]
```

iii. `CONVERSION_NOTES.md` says blink/tracking-failure NaNs were filled by nearest-neighbor interpolation after alignment.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is discretized into 5 equal-percentile bins using global percentile boundaries computed from all non-NaN pupil samples gathered during the first pass.

ii.
```python
if len(pupil_arr) > 0:
    pupil_percentiles = np.percentile(pupil_arr, np.linspace(0, 100, n_bins + 1)[1:-1])
...
pupil_binned = discretize_to_percentile_bins(
    pupil_clean, n_bins, pupil_percentiles
).astype(np.float32)
```

iii. The notes report the actual percentile boundaries and say the bins were computed globally across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Like running speed, pupil is first interpolated onto `ophys_ts` and then trialized with the same `frame_indices` as the neural traces.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_ts, pupil_vals, ophys_ts)
...
pupil_trial = pupil_at_ophys[frame_indices]
```

iii. The notes describe this as full alignment of eye-tracking and neural data onto the ophys timeline.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

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

iii. The notes describe the same four outcome categories and map them to integer class IDs.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The script converts the booleans to one of four outcome strings, maps that to an integer category, and then repeats that category across every time bin in the trial even though the underlying label is static per trial.

ii.
```python
outcome = get_trial_outcome(trial)
...
outcome_idx = outcome_to_idx.get(outcome, 0)
...
output_trial = np.stack([
    ...,
    np.full(n_timepoints, outcome_idx, dtype=np.int64),
], axis=0)
```

iii. The code comments explicitly say this was done for format consistency, because the script chose to store all outputs in a common `(n_output, T)` representation.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing running-speed samples after interpolation are forced to 0. Missing pupil samples stay NaN until trial output assembly, where all-NaN trials are assigned the middle bin and partial-NaN trials are filled from the nearest valid sample. Failed experiment loads, experiments with no neurons, experiments with too few trials, and very short trials are skipped.

ii.
```python
if len(signal_timestamps) == 0 or len(signal_values) == 0:
    return np.full(len(ophys_timestamps), np.nan)
...
running_at_ophys = np.nan_to_num(running_at_ophys, nan=0.0)
...
except Exception:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
...
if nan_mask.all():
    pupil_binned = np.full(len(pupil_vals), 2, dtype=np.float32)
...
return None, None, None, None, None, f"Failed to load: {e}"
return None, None, None, None, None, "No neurons"
```

iii. The notes justify these choices as pragmatic handling for blink artifacts, tracking failures, sparse signals, and sessions that would otherwise violate decoder format requirements.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive work is loading each experiment through the SDK, interpolating running and pupil traces to ophys timestamps for every experiment, and then looping through every valid trial and every ophys timepoint to build per-trial outputs.

ii.
```python
for i, (_, exp_info) in enumerate(exp_table.iterrows()):
    result = process_experiment(...)
...
running_at_ophys = interpolate_to_ophys(...)
pupil_at_ophys = interpolate_to_ophys(...)
...
for trial_idx, (_, trial) in enumerate(valid_trials.iterrows()):
    ...
    image_idx = get_image_at_timepoints(...)
    change_signal = get_image_change_signal(...)
```

iii. The trajectory shows the agent was aware that full conversion over 110 experiments was the expensive phase and used a smaller sample mode for debugging first.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. `get_image_at_timepoints` loops over every ophys timestamp; `get_image_change_signal` loops over change rows; `downsample_categorical_by_factor` loops over bins; and the pupil NaN-filling code loops over every missing sample to find the nearest valid value.

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
        dists = np.abs(valid_indices - j)
        nearest = valid_indices[np.argmin(dists)]
```

iii. None of these loops are justified in the notes as conceptually necessary; they are straightforward imperative implementations.

## 9-c. What processing does the code repeat multiple times?

i. The script repeatedly filters stimulus presentations trial-by-trial, re-runs similar downsampling logic for multiple streams, and does a full first pass over all experiments just to gather running/pupil values before a second pass assembles final outputs.

ii.
```python
trial_stim = stim_cd[
    (stim_cd['start_time'] >= t_start - 0.5) &
    (stim_cd['start_time'] <= t_stop + 0.5)
]
...
if ds_factor > 1:
    neural_trial = downsample_by_factor(...)
    image_idx = downsample_categorical_by_factor(...)
    change_signal = downsample_by_factor(...)
    running_trial = downsample_by_factor(...)
    pupil_trial = downsample_by_factor(...)
...
all_running_speeds.extend(running_at_ophys.tolist())
all_pupil_diameters.extend(valid_pupil.tolist())
```

iii. The two-pass design is deliberate because the agent wanted global percentile thresholds, but it duplicates some work and keeps intermediate raw outputs around until the second-stage binning.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes and stores `trial_outcomes` per experiment but never uses them later; it computes `actual_dt`, `stim_ends`, and `total_neurons` without downstream use; and it expands the static trial outcome label into a full time series even though the underlying variable is trial-level.

ii.
```python
actual_dt = dt * ds_factor
...
stim_ends = stim_presentations['end_time'].values
...
trial_outcomes = []
...
trial_outcomes.append(outcome)
...
experiment_results.append({
    ...
    'trial_outcomes': trial_outcomes,
})
...
total_neurons = sum(...)
...
np.full(n_timepoints, outcome_idx, dtype=np.int64)
```

iii. These are side effects of the script’s incremental development: some values were useful for debugging or for an earlier representation choice, but they are not consumed by the final decoder pipeline.
