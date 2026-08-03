# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads a local experiment metadata CSV, restricts it to experiments with NWB files present on disk, then filters to `active_behavior` and `Familiar` experiments. It instantiates an Allen SDK cache with `VisualBehaviorOphysProjectCache.from_s3_cache`, but the master list of data comes from the local CSV rather than the SDK experiment table. It then processes each remaining `ophys_experiment_id` individually.

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

```python
cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=data_dir)
exp_table = get_experiment_table(data_dir)
...
for i, (_, exp_info) in enumerate(exp_table.iterrows()):
    result = process_experiment(...)
```

iii. The justification appears in `CONVERSION_NOTES.md`: the agent chose active-behavior familiar-image sessions, included all available cre lines, and explicitly expanded beyond the paper’s narrower multiscope subset because it believed the available local data were limited. In the trajectory it also says it should “use all available active behavior sessions given our limited dataset.”

## 1-b. How are the data split into subjects?

i. Subjects are split by unique `mouse_id` values from the successfully processed experiments.

ii.
```python
all_mice = sorted(set(r['exp_info']['mouse_id'] for r in experiment_results))
mouse_to_idx = {m: i for i, m in enumerate(all_mice)}
subjects = [str(m) for m in all_mice]
...
subject_idx_list.append(mouse_to_idx[exp_info['mouse_id']])
```

iii. No separate justification is given beyond using the Allen metadata fields. The implementation treats `mouse_id` as the canonical animal identifier.

## 1-c. How are the data split into sessions?

i. The agent treats each `ophys_experiment_id` as one session. It does not group multiple experiments from the same `ophys_session_id` into a combined session.

ii.
```python
for i, (_, exp_info) in enumerate(exp_table.iterrows()):
    exp_id = int(exp_info['ophys_experiment_id'])
    ...
    experiment_results.append({
        'exp_info': exp_info,
        'neural_trials': neural_trials,
        ...
    })
```

iii. The trajectory gives the rationale explicitly: “Each imaging plane corresponds to one experiment, and since different planes have different neurons, each experiment should be treated as a separate ‘session’ in the output format.”

## 1-d. How are the data split into trials?

i. Trials come from `ds.trials`. For each valid trial, the agent uses the full window from `start_time` to `stop_time` and extracts all ophys frames whose timestamps lie within that interval.

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

iii. The notes justify this using the task structure from the paper: go and catch trials from the built-in trials table, using the flashed-image trial window and excluding aborted and auto-rewarded trials.

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps only go or catch trials, excludes aborted and auto-rewarded trials, drops trials with fewer frames than the downsampling factor, and skips experiments with fewer than two valid/usable trials.

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

iii. `CONVERSION_NOTES.md` states that go and catch trials were retained, aborted and auto-rewarded trials were excluded, and sessions with fewer than two valid trials were removed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` data are derived from `ds.events['events']`, not from `dff_traces`.

ii.
```python
events = ds.events
neural_full = np.vstack(events['events'].values).astype(np.float32)
```

iii. The notes justify this by citing the paper’s use of detected calcium events rather than raw dF/F, and the trajectory repeats that the agent intentionally used event traces aligned to ophys timestamps.

## 2-b. How is the `neural` data processed?

i. Neural traces are kept per experiment, sliced per trial, and optionally downsampled by averaging groups of frames to match a common 93.2 ms time bin. There is no multi-plane merging because each experiment is treated as its own session.

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

iii. `CONVERSION_NOTES.md` says all data were resampled to 93.2 ms bins to match the multiscope frame rate, with single-plane recordings downsampled by factor 3, and that detected calcium events were chosen as a cleaner neural signal.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent applies no explicit per-neuron filtering beyond whatever the Allen SDK already supplies in `events`.

ii.
```python
events = ds.events
neural_full = np.vstack(events['events'].values).astype(np.float32)
...
if n_neurons == 0:
    return None, None, None, None, None, "No neurons"
```

iii. The notes say ROI filtering is assumed to have been handled upstream by the Allen SDK pipeline and that only valid cell bodies remain.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent aligns neural data to ophys timestamps and segments trials using trial `start_time` and `stop_time`. Within each trial, neural samples are exactly the ophys frames inside that window.

ii.
```python
ophys_ts = ds.ophys_timestamps
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts <= t_stop)
frame_indices = np.where(frame_mask)[0]
neural_trial = neural_full[:, frame_indices]
```

iii. The notes state that all signals are aligned to ophys timestamps, and the metadata describes the alignment event as trial start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data are forced onto a common 93.2 ms bin size. Single-plane recordings are downsampled by a factor determined from the native ophys `dt`; categorical outputs are mode-downsampled and continuous signals are mean-downsampled.

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
```

iii. The notes explicitly justify this as matching the multiscope frame rate and making time bins consistent across sessions.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `ds.stimulus_presentations`, specifically `image_name`, `start_time`, and the change-detection stimulus block. The agent ignores `omitted` image labels.

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
image_idx = get_image_at_timepoints(trial_stim_no_omit, trial_ophys_ts, image_to_idx)
```

iii. The notes justify this by saying image identity should track the most recently presented image during flashed-image trials, with omitted stimuli retaining the prior image.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The agent builds a global image-name-to-index map from one experiment’s change-detection stimulus table, then for each trial walks through the ophys timestamps and assigns the most recent non-omitted image identity at every frame, including gray intervals.

ii.
```python
all_image_names = sorted([n for n in stim_cd['image_name'].unique()
                          if n != 'omitted' and isinstance(n, str)])
image_to_idx = {name: idx for idx, name in enumerate(all_image_names)}
```

```python
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

iii. `CONVERSION_NOTES.md` says the agent wanted 8 categories, carried the most recently presented image through gray periods, and kept the previous image through omitted flashes.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same per-trial ophys timestamps that define the neural trial matrix, so each image label corresponds one-to-one with a neural frame.

ii.
```python
trial_ophys_ts = ophys_ts[frame_indices]
image_idx = get_image_at_timepoints(trial_stim_no_omit, trial_ophys_ts, image_to_idx)
...
output_trial = np.stack([
    image_idx.astype(np.int64),
    ...
], axis=0)
```

iii. The notes justify alignment by saying all signals are put on the ophys timebase before being packaged into trials.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `ds.stimulus_presentations`, using rows where `is_change == True` and the stimulus `start_time`.

ii.
```python
def get_image_change_signal(stim_presentations, ophys_timestamps):
    change_signal = np.zeros(len(ophys_timestamps), dtype=np.float32)
    changes = stim_presentations[stim_presentations['is_change'] == True]
    for _, change in changes.iterrows():
        change_time = change['start_time']
        mask = (ophys_timestamps >= change_time) & (ophys_timestamps < change_time + 0.75)
        change_signal[mask] = 1.0
```

iii. The notes say the change signal should mark the 750 ms after a real image change and not mark sham catch events.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The agent creates a binary time series that is 1 for 750 ms after each change stimulus presentation, then downsampled if necessary and rebinarized.

ii.
```python
mask = (ophys_timestamps >= change_time) & (ophys_timestamps < change_time + 0.75)
change_signal[mask] = 1.0
...
if ds_factor > 1:
    change_signal = downsample_by_factor(change_signal, ds_factor)
    change_signal = (change_signal > 0.0).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` ties the 750 ms window to one 250 ms flash plus the following 500 ms gray interval.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is treated as a binary variable: values remain 0 or 1, and after any averaging during downsampling the signal is thresholded with `> 0.0`.

ii.
```python
change_signal = np.zeros(len(ophys_timestamps), dtype=np.float32)
...
change_signal[mask] = 1.0
...
change_signal = (change_signal > 0.0).astype(np.float32)
```

iii. The notes describe this output as strictly binary with `['no_change', 'change']`.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The image-change signal is computed directly on the trial’s ophys timestamps and stored alongside the trial’s neural frames.

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

iii. The notes say all streams are aligned to ophys timestamps, so image change shares the neural frame index.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ds.running_speed`, using its `timestamps` and `speed` columns.

ii.
```python
rs = ds.running_speed
running_at_ophys = interpolate_to_ophys(
    rs['timestamps'].values, rs['speed'].values, ophys_ts
)
```

iii. The notes identify AllenSDK running-speed data as the source and mention the SDK’s filtered running-speed signal.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated to ophys timestamps, NaNs are immediately filled with 0, all session-wide ophys-aligned values are pooled to compute global percentile thresholds, and each trial is discretized with those thresholds.

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

iii. The notes justify linear interpolation to ophys time and five equal-percentile bins across the dataset; they also state NaN running values were filled with 0.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is split into five global percentile bins using four percentile boundaries computed from pooled running values.

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

iii. `CONVERSION_NOTES.md` explicitly says the agent used five equal-percentile bins with global boundaries.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto `ophys_ts`, then trial slices use the same `frame_indices` as the neural data, with optional shared downsampling.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    rs['timestamps'].values, rs['speed'].values, ophys_ts
)
...
running_trial = running_at_ophys[frame_indices]
...
if ds_factor > 1:
    running_trial = downsample_by_factor(running_trial, ds_factor)
```

iii. The notes say all signals are aligned to ophys timestamps before being organized into trials.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `ds.eye_tracking`, specifically `pupil_width` and `timestamps`.

ii.
```python
et = ds.eye_tracking
pupil_vals = et['pupil_width'].values
pupil_ts = et['timestamps'].values
pupil_at_ophys = interpolate_to_ophys(pupil_ts, pupil_vals, ophys_ts)
```

iii. The notes say the agent uses `pupil_width` as the pupil-diameter measure.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil width is linearly interpolated to ophys time. Non-NaN values from the full session are pooled to compute global percentile thresholds. During output assembly, NaNs inside a trial are replaced by the nearest valid value; if an entire trial is NaN, the trial is assigned the middle bin.

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
    pupil_binned = discretize_to_percentile_bins(
        pupil_clean, n_bins, pupil_percentiles
    ).astype(np.float32)
```

iii. The notes justify interpolation plus five percentile bins and state that NaN pupil values from blink/tracking failures are filled with the nearest valid value.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into five global percentile bins using four percentile boundaries from pooled pupil data.

ii.
```python
if len(pupil_arr) > 0:
    pupil_percentiles = np.percentile(pupil_arr, np.linspace(0, 100, n_bins + 1)[1:-1])
...
pupil_binned = discretize_to_percentile_bins(
    pupil_clean, n_bins, pupil_percentiles
).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` explicitly states five equal-percentile pupil bins with global boundaries.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil measurements are interpolated onto `ophys_ts`, then sliced per trial with the same frame indices as neural data and optionally downsampled by the same factor.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_ts, pupil_vals, ophys_ts)
...
pupil_trial = pupil_at_ophys[frame_indices]
...
if ds_factor > 1:
    pupil_trial = downsample_by_factor(pupil_trial, ds_factor)
```

iii. The notes say all signals share the ophys timebase before trial extraction.

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
    else:
        return 'unknown'
```

iii. The notes say the agent used the four standard task outcomes and encoded them as four categories.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The string outcome is mapped to a fixed integer code and then repeated across every time bin of the trial, even though the underlying variable is static per trial.

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

iii. The notes say trial outcome is static per trial but is represented as time-varying “for format consistency.”

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent skips experiments that fail to load or have too few trials, drops individual trials with too few frames after segmentation, fills running-speed NaNs with zero, and fills pupil NaNs with nearest valid values or with the middle category if an entire trial lacks pupil data.

ii.
```python
try:
    ds = load_experiment(cache, exp_id)
except Exception as e:
    return None, None, None, None, None, f"Failed to load: {e}"
...
if len(valid_trials) < 2:
    return None, None, None, None, None, f"Only {len(valid_trials)} valid trials"
...
if len(frame_indices) < ds_factor:
    continue
...
running_at_ophys = np.nan_to_num(running_at_ophys, nan=0.0)
...
if nan_mask.all():
    pupil_binned = np.full(len(pupil_vals), 2, dtype=np.float32)
```

iii. The notes explicitly justify zero-filling running NaNs and nearest-neighbor filling of pupil NaNs as pragmatic handling of interpolation gaps, blinks, and tracking failures.

## 9-a. What are the most time-consuming steps of the code?

i. The heaviest step is repeatedly loading full AllenSDK experiments and then iterating over all trials in each experiment. The script also does per-frame Python work for image identity and per-trial output assembly.

ii.
```python
for i, (_, exp_info) in enumerate(exp_table.iterrows()):
    result = process_experiment(...)
```

```python
ds = load_experiment(cache, exp_id)
...
for trial_idx, (_, trial) in enumerate(valid_trials.iterrows()):
    ...
    image_idx = get_image_at_timepoints(trial_stim_no_omit, trial_ophys_ts, image_to_idx)
```

iii. The agent does not explicitly discuss runtime bottlenecks, but the structure implies that SDK loading plus repeated trial/frame iteration dominates.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The most obvious vectorizable loops are the per-frame loop in `get_image_at_timepoints`, the per-change loop in `get_image_change_signal`, the chunk loop in `downsample_categorical_by_factor`, the per-trial loop over `valid_trials`, and the per-NaN search loop used to fill pupil values.

ii.
```python
for t_idx, t in enumerate(ophys_timestamps):
    ...
for _, change in changes.iterrows():
    ...
for i in range(n_new):
    ...
for trial_idx, (_, trial) in enumerate(valid_trials.iterrows()):
    ...
for j in range(len(pupil_clean)):
    if nan_mask[j]:
        dists = np.abs(valid_indices - j)
        nearest = valid_indices[np.argmin(dists)]
```

iii. There is no explicit justification in the notes; this appears to be a simplicity/readability choice rather than a deliberate optimization strategy.

## 9-c. What processing does the code repeat multiple times?

i. The code loads one experiment once to discover image names and then loads experiments again in the main processing loop. It also makes one full pass over all experiments to collect raw values and another pass over all retained trials to discretize and assemble the final outputs.

ii.
```python
first_exp = cache.get_behavior_ophys_experiment(int(exp_table.iloc[0]['ophys_experiment_id']))
...
for i, (_, exp_info) in enumerate(exp_table.iterrows()):
    result = process_experiment(...)
```

```python
all_running_speeds = []
all_pupil_diameters = []
experiment_results = []
for ...:
    result = process_experiment(...)
...
for result in experiment_results:
    ...
    for t_idx, (neural_trial, out_raw) in enumerate(zip(neural_trials, output_trials_raw)):
```

iii. The notes do not frame this as repeated work, but the first pass is used to compute global percentile bins and the second pass packages outputs after those bins are known.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes and stores several values that are not used downstream: `stim_ends`, `stim_images`, `actual_dt`, `trial_outcomes`, and `total_neurons`. It also builds trial outcome as a full-length time series even though it is static per trial, which increases output size for convenience rather than necessity.

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
total_neurons = sum(data['neural'][i][0].shape[0] for i in range(len(data['neural'])) if len(data['neural'][i]) > 0)
```

```python
output_trial = np.stack([
    image_idx.astype(np.int64),
    change_signal.astype(np.int64),
    running_binned.astype(np.int64),
    pupil_binned.astype(np.int64),
    np.full(n_timepoints, outcome_idx, dtype=np.int64),
], axis=0)
```

iii. The only explicit justification is for the repeated trial-outcome row: the notes say it was represented as time-varying “for format consistency.” The other unused quantities do not have an explicit justification.
