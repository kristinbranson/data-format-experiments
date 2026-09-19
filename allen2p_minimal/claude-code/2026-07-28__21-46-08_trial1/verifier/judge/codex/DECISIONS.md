# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads experiment metadata from the local CSV metadata table, intersects it with locally present NWB filenames, filters to `active_behavior` and `Familiar`, then loads each remaining `ophys_experiment_id` through `VisualBehaviorOphysProjectCache.from_s3_cache(...).get_behavior_ophys_experiment(...)`. It therefore works from locally available experiments rather than from the full SDK experiment table filtered only by project code.

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
ds = load_experiment(cache, exp_id)
```

iii. In trajectory step 32, the AI says familiar multiscope-only data looked too restrictive for the locally available files, so it chose "all available active behavior sessions" and then processed them experiment by experiment. Step 86 repeats that choice as "110 active behavior sessions with familiar images."

## 1-b. How are the data split into subjects?

i. Subjects are split by unique `mouse_id` values from the filtered experiment table.

ii.
```python
all_mice = sorted(set(r['exp_info']['mouse_id'] for r in experiment_results))
mouse_to_idx = {m: i for i, m in enumerate(all_mice)}
subjects = [str(m) for m in all_mice]
...
subject_idx_list.append(mouse_to_idx[exp_info['mouse_id']])
```

iii. There is no separate explicit justification beyond the code structure. The closest trajectory evidence is step 86, where the AI summarizes the output as "38 subjects."

## 1-c. How are the data split into sessions?

i. The AI treats each `ophys_experiment_id` as one output session. It does not group multiple experiments from the same `ophys_session_id`.

ii.
```python
for i, (_, exp_info) in enumerate(exp_table.iterrows()):
    exp_id = int(exp_info['ophys_experiment_id'])
    ...
    result = process_experiment(...)
    ...
    experiment_results.append({
        'exp_info': exp_info,
        'neural_trials': neural_trials,
        ...
    })
...
for result in experiment_results:
    ...
    neural_all.append(session_neural)
```

iii. Step 32 states this directly: "Each imaging plane corresponds to one experiment, and since different planes have different neurons, each experiment should be treated as a separate 'session' in the output format."

## 1-d. How are the data split into trials?

i. Trials are taken from `ds.trials`. For each valid trial, the AI uses all ophys frames whose timestamps satisfy `start_time <= t <= stop_time`, creating variable-length trial windows from trial start to trial stop.

ii.
```python
trials = ds.trials
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

iii. In step 32, the AI says it will "extract Go and Catch trials only ... pull neural activity from trial start to stop using the event traces aligned to ophys timestamps."

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only `go` or `catch` trials that are not `aborted` and not `auto_rewarded`. It skips experiments with fewer than 2 valid trials and skips individual trials with fewer than `ds_factor` ophys frames. It does not explicitly require non-null `change_time`.

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

iii. Step 32 explains the main filtering choice as "Go and Catch trials only (excluding aborted and auto-rewarded)." No explicit trajectory justification was given for omitting a `change_time.notna()` check or for the extra `len(frame_indices) < ds_factor` exclusion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `ds.events['events']`, i.e. detected calcium events, not from `dff_traces`.

ii.
```python
events = ds.events
neural_full = np.vstack(events['events'].values).astype(np.float32)
```

iii. Step 32 says it will use "event traces," and step 88 explicitly says the chosen neural signal was "`events` (detected calcium events)."

## 2-b. How is the `neural` data processed?

i. Neural traces are stacked only within each experiment, sliced into trial windows, and optionally downsampled by averaging to a common 93.2 ms bin size. The AI does not merge multiple planes from the same `ophys_session_id`.

ii.
```python
dt = np.median(np.diff(ophys_ts))
ds_factor = max(1, int(round(target_dt / dt)))
...
neural_trial = neural_full[:, frame_indices]
...
if ds_factor > 1:
    neural_trial = downsample_by_factor(neural_trial, ds_factor)
...
neural_trials.append(neural_trial.astype(np.float32))
```

iii. Step 32 says the AI wanted one common bin size because the data mix roughly 11 Hz multiscope and roughly 31 Hz single-plane experiments. Step 86 summarizes that decision as "Common time bin: 93.2 ms ... with 3x downsampling for single-plane data."

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit neuron-level quality control beyond skipping experiments that fail to load or have zero neurons.

ii.
```python
try:
    ds = load_experiment(cache, exp_id)
except Exception as e:
    return None, None, None, None, None, f"Failed to load: {e}"
...
n_neurons = neural_full.shape[0]
if n_neurons == 0:
    return None, None, None, None, None, "No neurons"
```

iii. No explicit trajectory justification was recorded for neuron QC. The code implies the AI accepted the SDK-provided events table as already curated enough for use.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to ophys timestamps within each trial window. The AI selects ophys frames between `start_time` and `stop_time`, then uses those same frame indices for neural and behavioral variables.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts <= t_stop)
frame_indices = np.where(frame_mask)[0]
neural_trial = neural_full[:, frame_indices]
trial_ophys_ts = ophys_ts[frame_indices]
```

iii. Step 32 says the plan is to "identify which ophys frames fall within the trial window" and extract all signals on that ophys time base.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI forces a common time bin size of 93.2 ms (`target_dt = 0.0932`). Single-plane experiments are downsampled by averaging over groups of frames, and categorical outputs are downsampled by per-bin mode.

ii.
```python
target_dt = 0.0932  # seconds
...
dt = np.median(np.diff(ophys_ts))
ds_factor = max(1, int(round(target_dt / dt)))
...
if ds_factor > 1:
    neural_trial = downsample_by_factor(neural_trial, ds_factor)
    image_idx = downsample_categorical_by_factor(image_idx, ds_factor)
    change_signal = downsample_by_factor(change_signal, ds_factor)
```

iii. Step 32 explicitly says the AI thought the decoder format required consistent bins across experiments and therefore chose to downsample everything to the multiscope-like rate. The module docstring and step 86 repeat that rationale.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `ds.stimulus_presentations`, specifically `stimulus_block_name`, `image_name`, `start_time`, and the omission labels. It is not derived from trial-table `initial_image_name` / `change_image_name`.

ii.
```python
stim = ds.stimulus_presentations
stim_cd = stim[stim['stimulus_block_name'].str.contains('change_detection', na=False)].copy()
trial_stim = stim_cd[
    (stim_cd['start_time'] >= t_start - 0.5) &
    (stim_cd['start_time'] <= t_stop + 0.5)
]
trial_stim_no_omit = trial_stim[trial_stim['image_name'] != 'omitted']
```

iii. Step 32 says the AI would "determine the image identity at each timepoint," and step 88 says it handled `stimulus_presentations` directly when building outputs.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI discovers image categories from the first experiment's `stimulus_presentations`, builds a global image-to-index map, then walks through each trial's stimulus timeline frame by frame. During gray periods it carries forward the most recent non-omitted image; before the first valid image it leaves the default code at 0.

ii.
```python
all_image_names = sorted([n for n in stim_cd['image_name'].unique()
                          if n != 'omitted' and isinstance(n, str)])
image_to_idx = {name: idx for idx, name in enumerate(all_image_names)}

def get_image_at_timepoints(stim_presentations, ophys_timestamps, image_to_idx):
    image_indices = np.zeros(len(ophys_timestamps), dtype=np.int64)
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
```

iii. No separate trajectory note spells out the gray-screen carry-forward rule; that rationale only appears in the helper docstring. The broader justification in step 32 is that image identity should be time-varying on the ophys frame grid.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on `trial_ophys_ts`, the same per-trial ophys timestamps used to slice the neural matrix, and then optionally downsampled in lockstep with neural data.

ii.
```python
trial_ophys_ts = ophys_ts[frame_indices]
image_idx = get_image_at_timepoints(trial_stim_no_omit, trial_ophys_ts, image_to_idx)
...
if ds_factor > 1:
    image_idx = downsample_categorical_by_factor(image_idx, ds_factor)
```

iii. Step 32 says the AI would align all outputs to the ophys timestamps so image identity and neural activity would share the same frame grid.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations['is_change']` and the corresponding `start_time` values within the change-detection block.

ii.
```python
def get_image_change_signal(stim_presentations, ophys_timestamps):
    change_signal = np.zeros(len(ophys_timestamps), dtype=np.float32)
    changes = stim_presentations[stim_presentations['is_change'] == True]
    for _, change in changes.iterrows():
        change_time = change['start_time']
```

iii. Step 32 says the AI planned to "compute the image change signal" from the stimulus stream; step 88 again references handling `stimulus_presentations`.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For every stimulus marked as a change, the AI marks all ophys frames from `change_time` through `change_time + 0.75` seconds as 1. The result is created on the frame grid before any optional downsampling.

ii.
```python
for _, change in changes.iterrows():
    change_time = change['start_time']
    mask = (ophys_timestamps >= change_time) & (ophys_timestamps < change_time + 0.75)
    change_signal[mask] = 1.0
```

iii. The code comment says this is meant to mark the change stimulus presentation, and the 0.75 s window reflects the flashed image plus gray period. The trajectory does not provide a more detailed separate justification.

## 4-c. How is `output` *Image change* thresholded into categories?

i. The variable is binary from the start: 0 for non-change frames and 1 for frames inside the change window. After downsampling, any averaged value greater than 0 is converted back to 1.

ii.
```python
change_signal = np.zeros(len(ophys_timestamps), dtype=np.float32)
...
change_signal[mask] = 1.0
...
if ds_factor > 1:
    change_signal = downsample_by_factor(change_signal, ds_factor)
    change_signal = (change_signal > 0.0).astype(np.float32)
```

iii. No explicit trajectory note discusses this thresholding beyond the general plan to make image change a binary time-varying output.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is computed on `trial_ophys_ts` using the same ophys frame selection as the neural data and is downsampled with the same factor when needed.

ii.
```python
trial_ophys_ts = ophys_ts[frame_indices]
change_signal = get_image_change_signal(trial_stim, trial_ophys_ts)
...
if ds_factor > 1:
    change_signal = downsample_by_factor(change_signal, ds_factor)
```

iii. Step 32 says all variables would be aligned to the ophys frame times.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ds.running_speed['timestamps']` and `ds.running_speed['speed']`.

ii.
```python
rs = ds.running_speed
running_at_ophys = interpolate_to_ophys(
    rs['timestamps'].values, rs['speed'].values, ophys_ts
)
```

iii. Step 32 identifies running speed as one of the behavioral variables to interpolate to the ophys frame times.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI linearly interpolates running speed onto the ophys timestamps, immediately replaces interpolation NaNs with 0.0, stores those values for global percentile estimation, and optionally downsamples them with the same averaging used for neural data.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    rs['timestamps'].values, rs['speed'].values, ophys_ts
)
running_at_ophys = np.nan_to_num(running_at_ophys, nan=0.0)
all_running_speeds.extend(running_at_ophys.tolist())
...
running_trial = running_at_ophys[frame_indices]
...
if ds_factor > 1:
    running_trial = downsample_by_factor(running_trial, ds_factor)
```

iii. Step 32 gives the high-level justification: running speed arrives at a different sampling rate and therefore should be interpolated to ophys time. No explicit trajectory note justifies the immediate NaN-to-zero replacement.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 global percentile bins. The AI collects all interpolated running values across experiments, computes the 20/40/60/80 percentile boundaries, and uses `np.digitize` to assign bin IDs 0-4.

ii.
```python
running_percentiles = np.percentile(running_arr, np.linspace(0, 100, n_bins + 1)[1:-1])

def discretize_to_percentile_bins(values, n_bins, percentiles):
    result = np.digitize(values, percentiles)
    result = np.clip(result, 0, n_bins - 1)
    return result.astype(np.int64)
...
running_binned = discretize_to_percentile_bins(
    out_raw['running'], n_bins, running_percentiles
).astype(np.float32)
```

iii. Steps 55, 57, and 59 show the AI explicitly debugging this choice to ensure five bins were actually produced, concluding that the fix was to remove an erroneous `- 1` from the digitization logic.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is first interpolated onto the full-session ophys timestamps, then each trial takes the same `frame_indices` as the neural data, and any optional downsampling is applied in parallel.

ii.
```python
running_at_ophys = interpolate_to_ophys(..., ophys_ts)
...
frame_indices = np.where(frame_mask)[0]
running_trial = running_at_ophys[frame_indices]
...
if ds_factor > 1:
    running_trial = downsample_by_factor(running_trial, ds_factor)
```

iii. Step 32 says behavioral variables sampled at different rates would be interpolated to match the ophys frame times before trial extraction.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `ds.eye_tracking['pupil_width']` and `ds.eye_tracking['timestamps']`. The AI does not use `likely_blink`.

ii.
```python
et = ds.eye_tracking
pupil_vals = et['pupil_width'].values
pupil_ts = et['timestamps'].values
pupil_at_ophys = interpolate_to_ophys(pupil_ts, pupil_vals, ophys_ts)
```

iii. The only explicit rationale is the code comment "Use pupil width as diameter (from tutorial: pupil_width)." No trajectory step separately discusses blink handling.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI linearly interpolates `pupil_width` to ophys timestamps, keeps only non-NaN values when estimating global percentiles, and optionally downsamples the per-trial traces. It does not remove blink frames before interpolation.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_ts, pupil_vals, ophys_ts)
...
valid_pupil = pupil_at_ophys[~np.isnan(pupil_at_ophys)]
all_pupil_diameters.extend(valid_pupil.tolist())
...
pupil_trial = pupil_at_ophys[frame_indices]
...
if ds_factor > 1:
    pupil_trial = downsample_by_factor(pupil_trial, ds_factor)
```

iii. Step 32 supplies the general interpolation rationale for behavioral signals, but the trajectory contains no explicit justification for omitting blink filtering.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The AI computes 5 global percentile bins from all non-NaN interpolated pupil values. Within each trial, if all pupil values are NaN it assigns the middle bin (2); otherwise it fills NaNs by copying the nearest valid sample and then digitizes the cleaned trace into bins 0-4.

ii.
```python
pupil_percentiles = np.percentile(pupil_arr, np.linspace(0, 100, n_bins + 1)[1:-1])
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
    pupil_binned = discretize_to_percentile_bins(
        pupil_clean, n_bins, pupil_percentiles
    ).astype(np.float32)
```

iii. No explicit trajectory note justifies this NaN-filling strategy. It appears to have been introduced purely to keep the decoder-facing output dense and categorical.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to full-session ophys timestamps, sliced with the same trial frame indices as the neural data, and optionally downsampled with the same factor.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_ts, pupil_vals, ophys_ts)
...
frame_indices = np.where(frame_mask)[0]
pupil_trial = pupil_at_ophys[frame_indices]
...
if ds_factor > 1:
    pupil_trial = downsample_by_factor(pupil_trial, ds_factor)
```

iii. Step 32 states that pupil and running streams would be interpolated to the ophys frame grid before trial segmentation.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

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

iii. Step 32 names the four outcome classes the AI intended to use: hits, misses, false alarms, and correct rejections.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI converts the per-trial string outcome into an integer code with a fixed mapping and then repeats that code across every time bin in the trial, even though the variable is conceptually static.

ii.
```python
outcome_names = ['hit', 'miss', 'false_alarm', 'correct_reject']
outcome_to_idx = {name: idx for idx, name in enumerate(outcome_names)}
...
outcome = out_raw['outcome']
outcome_idx = outcome_to_idx.get(outcome, 0)
...
output_trial = np.stack([
    image_idx.astype(np.int64),
    change_signal.astype(np.int64),
    running_binned.astype(np.int64),
    pupil_binned.astype(np.int64),
    np.full(n_timepoints, outcome_idx, dtype=np.int64),
], axis=0)
```

iii. The justification appears in the inline code comments: because the AI wanted a uniform `(n_output, n_timepoints)` shape, it decided to make trial outcome time-varying by repeating a constant code across the whole trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles load failures by skipping experiments, handles zero-neuron experiments by skipping them, converts missing running-speed interpolation results to 0.0 immediately, fills missing pupil samples by nearest-neighbor copying or with the median bin if the whole trial is NaN, and skips experiments/trials that are too short to satisfy its resampling scheme.

ii.
```python
try:
    ds = load_experiment(cache, exp_id)
except Exception as e:
    return None, None, None, None, None, f"Failed to load: {e}"
...
running_at_ophys = np.nan_to_num(running_at_ophys, nan=0.0)
...
if nan_mask.all():
    pupil_binned = np.full(len(pupil_vals), 2, dtype=np.float32)
else:
    ...
    pupil_clean[j] = pupil_clean[nearest]
...
if len(frame_indices) < ds_factor:
    continue
if len(neural_trials) < 2:
    return None, None, None, None, None, f"Only {len(neural_trials)} usable trials after processing"
```

iii. The trajectory mostly justifies these choices indirectly: step 32 emphasizes using what is locally available and step 47 onward shows the AI changing outputs so the decoder would accept them. There is no explicit higher-level missing-data policy in the trajectory.

## 9-a. What are the most time-consuming steps of the code?

i. The code structure implies that the most time-consuming work is repeatedly loading each experiment through the SDK and then processing every trial and every ophys frame for image identity / image change construction.

ii.
```python
for i, (_, exp_info) in enumerate(exp_table.iterrows()):
    result = process_experiment(...)

def get_image_at_timepoints(...):
    for t_idx, t in enumerate(ophys_timestamps):
        ...

for trial_idx, (_, trial) in enumerate(valid_trials.iterrows()):
    ...
```

iii. The AI did not explicitly analyze runtime in the trajectory. This section is inferred from the code and from step 68, where a full conversion over 110 experiments is treated as a substantial batch job.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorizable loops are the frame-by-frame scan in `get_image_at_timepoints`, the per-bin loop in `downsample_categorical_by_factor`, the per-trial `iterrows()` loop, and the per-sample nearest-value fill for NaN pupil points.

ii.
```python
for t_idx, t in enumerate(ophys_timestamps):
    ...

for i in range(n_new):
    chunk = data[i*factor:(i+1)*factor]
    values, counts = np.unique(chunk, return_counts=True)
    result[i] = values[np.argmax(counts)]

for trial_idx, (_, trial) in enumerate(valid_trials.iterrows()):
    ...

for j in range(len(pupil_clean)):
    if nan_mask[j]:
        dists = np.abs(valid_indices - j)
        nearest = valid_indices[np.argmin(dists)]
        pupil_clean[j] = pupil_clean[nearest]
```

iii. The AI did not discuss vectorization tradeoffs in the trajectory. These potential improvements are visible only from the implementation.

## 9-c. What processing does the code repeat multiple times?

i. The code repeatedly slices the stimulus table per trial and re-scans trial-local stimulus timelines frame by frame. It also performs a two-pass workflow where it first collects raw running/pupil values across all experiments and then loops over all trials again to bin and assemble outputs.

ii.
```python
trial_stim = stim_cd[
    (stim_cd['start_time'] >= t_start - 0.5) &
    (stim_cd['start_time'] <= t_stop + 0.5)
]
image_idx = get_image_at_timepoints(trial_stim_no_omit, trial_ophys_ts, image_to_idx)
...
all_running_speeds.extend(running_at_ophys.tolist())
all_pupil_diameters.extend(valid_pupil.tolist())
...
for result in experiment_results:
    ...
    for t_idx, (neural_trial, out_raw) in enumerate(zip(neural_trials, output_trials_raw)):
        running_binned = discretize_to_percentile_bins(...)
        ...
```

iii. No trajectory step explicitly identifies this as repeated work. It is an implementation detail visible in the code.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores several values that are not used downstream: `stim_ends`, `stim_images`, `actual_dt`, and the `trial_outcomes` list returned from `process_experiment`. It also builds `total_neurons` for the summary without using it elsewhere.

ii.
```python
stim_ends = stim_presentations['end_time'].values
stim_images = stim_presentations['image_name'].values
...
actual_dt = dt * ds_factor
...
trial_outcomes = []
...
trial_outcomes.append(outcome)
...
return neural_trials, output_trials, trial_outcomes, n_neurons, brain_region, None
...
total_neurons = sum(data['neural'][i][0].shape[0] for i in range(len(data['neural'])) if len(data['neural'][i]) > 0)
```

iii. The trajectory does not mention these as deliberate tradeoffs. They appear to be leftover or convenience computations rather than intentionally retained outputs.
