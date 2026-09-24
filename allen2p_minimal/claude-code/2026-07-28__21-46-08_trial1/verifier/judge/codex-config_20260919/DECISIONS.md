# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local experiment CSV and local NWB filenames, keeps experiments whose files exist, and filters them to active-behavior, Familiar sessions. It creates an AllenSDK S3-backed cache and loads every retained experiment individually. Thus “all” means all locally available active/Familiar experiments, including single- and multi-plane rigs, rather than the reference's `VisualBehavior` project selection.

ii.
```python
exp_table = pd.read_csv(os.path.join(data_dir,
    'visual-behavior-ophys-1.1.0/project_metadata/ophys_experiment_table.csv'))
exp_ids = [int(re.search(r'(\d+)', f).group(1)) for f in os.listdir(nwb_dir)]
available = exp_table[exp_table['ophys_experiment_id'].isin(exp_ids)]
filtered = available[(available['behavior_type'] == 'active_behavior') &
                     (available['experience_level'] == 'Familiar')]
cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=data_dir)
```

iii. The trajectory says the paper used Familiar Multiscope data, but the available Multiscope subset seemed too small. The agent therefore chose all available active/Familiar experiments across both rigs to obtain a larger dataset, while preserving the requested Go/Catch task.

## 1-b. How are the data split into subjects?

i. Subjects are unique sorted `mouse_id` values among successfully processed experiments, converted to strings; each experiment receives the corresponding subject index.

ii.
```python
all_mice = sorted(set(r['exp_info']['mouse_id'] for r in experiment_results))
mouse_to_idx = {m: i for i, m in enumerate(all_mice)}
subjects = [str(m) for m in all_mice]
subject_idx_list.append(mouse_to_idx[exp_info['mouse_id']])
```

iii. The agent identified `mouse_id` as the dataset's animal identifier and used it consistently after failed experiments had been removed.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id` (one imaging plane) is treated as a separate output session. Experiments sharing an `ophys_session_id` are not combined.

ii.
```python
for i, (_, exp_info) in enumerate(exp_table.iterrows()):
    result = process_experiment(cache, exp_info, ...)
    experiment_results.append({...})
...
for result in experiment_results:
    neural_all.append(session_neural)
```

iii. The trajectory explicitly reasons that different planes contain different neurons and concludes that each experiment should be a separate “session.”

## 1-d. How are the data split into trials?

i. The AllenSDK `ds.trials` table defines trials. For each retained row, ophys frames with timestamps from `start_time` through `stop_time` (inclusive) are selected, producing variable-length trials.

ii.
```python
valid_trials = trials[((trials['go'] == True) | (trials['catch'] == True)) &
                      (trials['aborted'] == False) &
                      (trials['auto_rewarded'] == False)]
frame_mask = (ophys_ts >= t_start) & (ophys_ts <= t_stop)
frame_indices = np.where(frame_mask)[0]
```

iii. The agent inspected sample trial rows and stimulus presentations, then chose the SDK's experimental trial boundaries so each trial includes its pre-change and post-change periods.

## 1-e. How are trials filtered based on quality controls?

i. Only Go or Catch trials are kept; aborted and auto-rewarded trials are removed. Trials shorter than one downsampling group are skipped, and an experiment is skipped unless at least two usable trials remain.

ii.
```python
valid_trials = trials[((trials['go'] == True) | (trials['catch'] == True)) &
                      ~trials['aborted'] & ~trials['auto_rewarded']]
if len(frame_indices) < ds_factor:
    continue
if len(neural_trials) < 2:
    return ..., f"Only {len(neural_trials)} usable trials after processing"
```

iii. The stated rationale is to follow the instruction to include Go/Catch while excluding premature-lick and free-reward trials, and to satisfy the decoder's minimum of two trials per session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from AllenSDK detected/deconvolved calcium `events`, not dF/F traces.

ii.
```python
events = ds.events
neural_full = np.vstack(events['events'].values).astype(np.float32)
```

iii. The agent interpreted the paper's statement that detected calcium events were used as the applicable processing choice and described events as cleaner than dF/F.

## 2-b. How is the `neural` data processed?

i. Event arrays are stacked by neuron within each experiment, sliced to trial frames, cast to `float32`, and—on ~31 Hz rigs—averaged in non-overlapping groups (normally three). Neurons from simultaneous planes are not stacked together.

ii.
```python
neural_trial = neural_full[:, frame_indices]
if ds_factor > 1:
    neural_trial = downsample_by_factor(neural_trial, ds_factor)
neural_trials.append(neural_trial.astype(np.float32))
```

iii. The agent wanted one common temporal resolution across single-plane (~31 Hz) and Multiscope (~11 Hz) recordings and chose averaging to retain both equipment types.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit cell-level filter is applied after reading `ds.events`; experiments with no neurons or an events-loading failure are skipped.

ii.
```python
if n_neurons == 0:
    return ..., "No neurons"
```

iii. The agent relied on AllenSDK's upstream ROI filtering, stating that invalid ROIs, duplicates, edge artifacts, and similar objects were already removed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial neural data is indexed on `ophys_timestamps` and begins at the first frame at or after trial `start_time`; frames through `stop_time` are included. Metadata calls the alignment event “Trial start.”

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts <= t_stop)
neural_trial = neural_full[:, np.where(frame_mask)[0]]
'temporal_alignment_event': 'Trial start (aligned to ophys timestamps)'
```

iii. The agent chose the native ophys clock as the common clock and the SDK trial boundaries as the natural alignment for variable-length trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The declared resolution is 93.2 ms (~10.7 Hz). A factor `round(0.0932/native_dt)` is used; ~31 Hz recordings are rebinned by averaging about three adjacent frames, while ~11 Hz recordings are retained.

ii.
```python
target_dt = 0.0932
ds_factor = max(1, int(round(target_dt / dt)))
neural_trial = downsample_by_factor(neural_trial, ds_factor)
'time_bin_size': target_dt * 1000
```

iii. The agent reasoned that the target format required common bins across sessions and selected the slower Multiscope rate as the common denominator.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from change-detection `stimulus_presentations`, principally `start_time` and `image_name`; omitted rows are excluded. The global image vocabulary is discovered from the first retained experiment.

ii.
```python
stim_cd = stim[stim['stimulus_block_name'].str.contains('change_detection', na=False)]
trial_stim_no_omit = trial_stim[trial_stim['image_name'] != 'omitted']
image_idx = get_image_at_timepoints(trial_stim_no_omit, trial_ophys_ts, image_to_idx)
```

iii. The agent wanted identity at every neural time point, including repeated flashes, gray periods, and omissions, and verified that Familiar sessions used eight images.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Sorted image names are mapped to integers. At each ophys timestamp, the most recently started non-omitted image is carried forward; gray intervals and omissions retain the last identity. On high-rate data, each three-frame group is reduced by mode.

ii.
```python
image_to_idx = {name: idx for idx, name in enumerate(all_image_names)}
while stim_idx < len(stim_starts) - 1 and stim_starts[stim_idx + 1] <= t:
    stim_idx += 1
image_indices[t_idx] = current_image_idx
image_idx = downsample_categorical_by_factor(image_idx, ds_factor)
```

iii. The justification was that image identity should persist through the 500 ms gray interval and omitted flashes, and categorical downsampling should use a mode rather than an average.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Identity is evaluated at exactly the trial ophys timestamps used to slice neural data and is downsampled in matching frame groups.

ii.
```python
trial_ophys_ts = ophys_ts[frame_indices]
image_idx = get_image_at_timepoints(..., trial_ophys_ts, ...)
```

iii. The agent used the ophys clock for all streams so output columns correspond to neural columns.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It uses `stimulus_presentations.is_change` and each change presentation's `start_time` within the trial's change-detection block.

ii.
```python
changes = stim_presentations[stim_presentations['is_change'] == True]
change_time = change['start_time']
```

iii. The agent used the presentation table to distinguish genuine image changes from catch/sham changes.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector is set to one from each actual change onset for 0.75 s. If downsampled, groups are averaged and then converted back to binary using `> 0`.

ii.
```python
mask = (ophys_timestamps >= change_time) & (ophys_timestamps < change_time + 0.75)
change_signal[mask] = 1.0
change_signal = (downsample_by_factor(change_signal, ds_factor) > 0.0).astype(np.float32)
```

iii. The 750 ms interval was justified as one 250 ms flashed image plus its following 500 ms gray interval.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Native values are already 0/1. After averaging during temporal rebinning, any group containing a positive change frame becomes category 1; all others are 0.

ii.
```python
change_signal = (change_signal > 0.0).astype(np.float32)
change_signal.astype(np.int64)
```

iii. This preserves a transient change label despite averaging and yields the required binary categorical output.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change masks are computed on the same trial ophys timestamps as neural samples and, where needed, grouped using the same downsampling factor.

ii.
```python
change_signal = get_image_change_signal(trial_stim, trial_ophys_ts)
neural_trial = downsample_by_factor(neural_trial, ds_factor)
change_signal = downsample_by_factor(change_signal, ds_factor)
```

iii. The common ophys timebase and identical grouping were intended to maintain column-level alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `ds.running_speed['timestamps']` and `ds.running_speed['speed']`.

ii.
```python
rs = ds.running_speed
running_at_ophys = interpolate_to_ophys(rs['timestamps'].values,
                                        rs['speed'].values, ophys_ts)
```

iii. The agent used the SDK's processed wheel-speed interface as the canonical locomotion signal.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Speed is linearly interpolated to the full ophys timeline, out-of-range NaNs are replaced with zero, high-rate trials are averaged in frame groups, and values are later discretized. Percentile boundaries are calculated from full-session interpolated values, including periods outside retained trials.

ii.
```python
running_at_ophys = interpolate_to_ophys(...)
running_at_ophys = np.nan_to_num(running_at_ophys, nan=0.0)
all_running_speeds.extend(running_at_ophys.tolist())
running_trial = downsample_by_factor(running_trial, ds_factor)
```

iii. The agent said interpolation aligns the ~60 Hz signal to ophys time, zero safely handles missing running samples, and averaging supports a common bin size.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four global 20th/40th/60th/80th percentile boundaries create five integer bins (0–4). The global population is all full-session speeds from processed experiments, not just retained trial samples.

ii.
```python
running_percentiles = np.percentile(running_arr,
    np.linspace(0, 100, n_bins + 1)[1:-1])
result = np.digitize(values, percentiles)
result = np.clip(result, 0, n_bins - 1)
```

iii. Equal-percentile bins were chosen to balance decoder classes globally. The trajectory shows the agent tested and corrected an initial off-by-one binning error.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. The full running trace is interpolated to `ophys_ts`; the same `frame_indices` select neural and running samples, followed by matching downsampling groups.

ii.
```python
running_trial = running_at_ophys[frame_indices]
neural_trial = neural_full[:, frame_indices]
```

iii. The agent relied on hardware-synchronized timestamps and interpolation onto the neural clock.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It comes from `ds.eye_tracking['pupil_width']` and `timestamps`. The `likely_blink` flag is not used.

ii.
```python
et = ds.eye_tracking
pupil_vals = et['pupil_width'].values
pupil_ts = et['timestamps'].values
```

iii. The agent followed an Allen tutorial's use of `pupil_width` as the pupil-diameter measure.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil width is linearly interpolated to ophys timestamps after removing NaN values (but not blink-flagged values), optionally averaged during temporal downsampling, and globally percentile-binned. Within a trial, missing samples are filled from the nearest valid sample; an all-missing trial is assigned the middle bin.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_ts, pupil_vals, ophys_ts)
pupil_trial = downsample_by_factor(pupil_trial, ds_factor)
nearest = valid_indices[np.argmin(np.abs(valid_indices - j))]
pupil_clean[j] = pupil_clean[nearest]
```

iii. The agent said interpolation aligns eye tracking to ophys time and nearest-valid filling avoids missing categorical labels. Its notes refer to blink artifacts, although the implemented code does not filter `likely_blink`.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four global percentile boundaries make five bins. Boundaries use all non-NaN full-session interpolated pupil values from processed experiments. All-missing trials receive bin 2.

ii.
```python
pupil_percentiles = np.percentile(pupil_arr,
    np.linspace(0, 100, n_bins + 1)[1:-1])
pupil_binned = discretize_to_percentile_bins(pupil_clean, n_bins, pupil_percentiles)
```

iii. As with running, the stated aim was equal-frequency, globally consistent decoder classes.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Eye tracking is interpolated to the full ophys clock, indexed with the trial's neural `frame_indices`, and downsampled with the same factor.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(..., ophys_ts)
pupil_trial = pupil_at_ophys[frame_indices]
```

iii. The agent treated eye and ophys timestamps as hardware-synchronized and used interpolation to produce one pupil value per neural column.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome comes from the trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`, checked in that order.

ii.
```python
if trial['hit']: return 'hit'
elif trial['miss']: return 'miss'
elif trial['false_alarm']: return 'false_alarm'
elif trial['correct_reject']: return 'correct_reject'
```

iii. These are the SDK's canonical, mutually exclusive outcomes for valid Go and Catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Names map to fixed codes 0–3. Although conceptually static, the selected code is repeated across all time points so it can share one output matrix with the four time-varying outputs. Unknown outcomes default to hit code 0.

ii.
```python
outcome_names = ['hit', 'miss', 'false_alarm', 'correct_reject']
outcome_idx = outcome_to_idx.get(outcome, 0)
np.full(n_timepoints, outcome_idx, dtype=np.int64)
```

iii. The agent explicitly chose repetition for format consistency because the output representation could not directly mix static and time-varying rows.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Experiment loading/events failures, no-neuron experiments, and experiments with fewer than two usable trials are skipped. Interpolation ignores NaN source values. Running NaNs become zero. Pupil read failures produce all NaNs; per-trial pupil gaps use nearest valid values and all-missing trials use bin 2. Very short trials are skipped.

ii.
```python
except Exception as e:
    return None, None, None, None, None, f"Failed to load: {e}"
running_at_ophys = np.nan_to_num(running_at_ophys, nan=0.0)
except Exception:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
```

iii. The agent prioritized completing the conversion despite isolated failures and ensuring every output remains categorical and finite. It characterized zero/nearest/median-bin replacements as safe defaults.

## 9-a. What are the most time-consuming steps of the code?

i. Loading and parsing each NWB experiment through AllenSDK is likely dominant. The conversion then materializes full event, running, and pupil timelines and loops over every trial; serialization of the large final pickle is also substantial.

ii.
```python
for _, exp_info in exp_table.iterrows():
    ds = cache.get_behavior_ophys_experiment(exp_id)
    result = process_experiment(...)
```

iii. The trajectory repeatedly uses long timeouts for experiment loads and the full conversion and describes NWB loading as the main practical cost.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-timepoint image-identity loop, categorical downsampling loop, per-trial DataFrame loop, nearest-valid pupil fill loop, and repeated stimulus DataFrame filtering could be vectorized (for example with `searchsorted`, reshaping, or bulk nearest-index computation).

ii.
```python
for t_idx, t in enumerate(ophys_timestamps): ...
for i in range(n_new): ...
for trial_idx, (_, trial) in enumerate(valid_trials.iterrows()): ...
for j in range(len(pupil_clean)): ...
```

iii. The agent did not document optimization reasoning for these loops; it focused on correctness and end-to-end validation.

## 9-c. What processing does the code repeat multiple times?

i. Every experiment—even planes from the same simultaneous session—separately loads and recomputes identical trial tables, stimulus timelines, running interpolation, pupil interpolation, image labels, and trial boundaries. Output assembly then traverses all stored trials a second time for discretization.

ii.
```python
for _, exp_info in exp_table.iterrows():
    result = process_experiment(cache, exp_info, ...)
...
for result in experiment_results:
    for neural_trial, out_raw in zip(...):
```

iii. This follows from the deliberate experiment-as-session decision. The trajectory does not acknowledge that simultaneous planes duplicate behavioral processing.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It constructs unused `actual_dt`, `stim_images`, `stim_ends`, `trial_outcomes`, and the `image_names` function argument. It collects full-session running/pupil values outside retained trials solely for thresholds, loads a first experiment separately to discover image names, and computes/prints extensive summaries not needed by the saved data.

ii.
```python
actual_dt = dt * ds_factor
stim_images = stim_cd[stim_cd['image_name'] != 'omitted']
stim_ends = stim_presentations['end_time'].values
trial_outcomes.append(outcome)
```

iii. No explicit justification is provided. Some values appear to be remnants of exploratory implementation or diagnostics rather than necessary conversion work.
