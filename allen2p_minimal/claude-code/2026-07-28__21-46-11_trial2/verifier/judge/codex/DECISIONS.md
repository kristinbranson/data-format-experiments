# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads a local metadata CSV (`ophys_experiment_table.csv`), filters it to experiments that have NWB files present locally and to `passive == False`, then opens each NWB file directly with `h5py`. It does not use the Allen SDK cache, and it does not filter to `project_code == "VisualBehavior"`; in practice this includes both `VisualBehavior` and `VisualBehaviorMultiscope` active experiments.

ii.
```python
exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
...
nwb_files = [f for f in os.listdir(NWB_DIR) if f.endswith('.nwb')]
nwb_ids = [int(f.split('_')[-1].split('.')[0]) for f in nwb_files]
...
exp_table = exp_table[exp_table['ophys_experiment_id'].isin(nwb_ids)]
exp_table = exp_table[exp_table['passive'] == False]
...
with h5py.File(nwb_path, 'r') as f:
```

iii. In the trajectory, the AI says each NWB file is one ophys experiment and that it should "filter for active behavior sessions only (passive=False)" while loading the data directly from NWB files because this "mirrors what the SDK does under the hood" (steps 31, 89, 92).

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by unique `mouse_id` values from the filtered experiment table. The output `subjects` list is `sorted(exp_table['mouse_id'].unique().astype(str))`.

ii.
```python
all_subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
...
mouse_id = str(row['mouse_id'])
...
subject_idx_list.append(subject_to_idx[mouse_id])
```

iii. The trajectory treats mice as the primary subject unit and repeatedly reports counts by mouse. Step 31 explicitly says the AI will use the experiment table to organize experiments and sessions by mouse.

## 1-c. How are the data split into sessions?

i. The AI treats each `ophys_experiment_id` / NWB file as one session. It does not group multiple experiments by shared `ophys_session_id`.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    ...
    nwb_path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{eid}.nwb')
    nwb_data = load_nwb_data(nwb_path)
    result = process_experiment(nwb_data, all_image_names)
```

iii. The clearest justification is in step 31: "Each NWB file = one ophys experiment (one imaging plane)" and "Each experiment is a 'session' for our purposes."

## 1-d. How are the data split into trials?

i. Trials are split using the NWB `intervals/trials` table. For each included trial, the AI takes all ophys frames with timestamps `>= start_time` and `< stop_time`, so trials are variable-length windows spanning the full trial.

ii.
```python
n_trials_total = len(nwb_data['trial_start_times'])
...
t_start = nwb_data['trial_start_times'][trial_idx]
t_stop = nwb_data['trial_stop_times'][trial_idx]
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
frame_indices = np.where(frame_mask)[0]
...
trial_neural = events[frame_indices, :].T
```

iii. In step 31, the AI says trials "span from start_time to stop_time and contain multiple image presentations, with the critical change occurring at change_time."

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept only if they are not aborted, not auto-rewarded, and marked as either `go` or `catch`. Trials are then skipped if they have an unknown outcome or fewer than 2 ophys frames. Sessions/experiments with fewer than 2 valid trials are dropped.

ii.
```python
def get_trial_mask(trial_idx, nwb_data):
    if nwb_data['trial_aborted'][trial_idx]:
        return False
    if nwb_data['trial_auto_rewarded'][trial_idx]:
        return False
    if not (nwb_data['trial_go'][trial_idx] or nwb_data['trial_catch'][trial_idx]):
        return False
    return True
...
outcome = get_trial_outcome(trial_idx, nwb_data)
if outcome == -1:
    continue
...
if len(frame_indices) < 2:
    continue
...
if neural_trials is None or len(neural_trials) < 2:
    ...
    continue
```

iii. The trajectory repeatedly states the intent to include "Go and Catch only, exclude Aborted and Auto-rewarded" (steps 31 and 87). There is no explicit trajectory discussion of the extra filters on unknown outcomes, very short trials, or fewer-than-2-trial sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `processing/ophys/event_detection/data` in each NWB file. It also loads dF/F traces, but those are only described as fallback/comparison and are not used in the final dataset.

ii.
```python
events = f['processing/ophys/event_detection/data'][:]
events_ts = f['processing/ophys/event_detection/timestamps'][:]
data['events'] = events
...
dff = f['processing/ophys/dff/traces/data'][:]
data['dff'] = dff
...
events = nwb_data['events']
```

iii. Step 31 says "Neural data: use events (detected calcium events) - this is what the paper uses." Step 92 repeats that using `processing/ophys/event_detection/data` "matches `experiment.events`" and is "appropriate."

## 2-b. How is the `neural` data processed?

i. The AI keeps the event traces on the ophys frame time base, filters to ROIs with `valid_roi == True`, transposes from time-by-cell to cell-by-time, and slices the resulting event matrix trial by trial. It does not merge multiple planes into one session.

ii.
```python
events = nwb_data['events']  # (n_timepoints, n_cells)
valid = nwb_data['valid_roi']
events = events[:, valid]
...
trial_neural = events[frame_indices, :].T  # (n_neurons, n_timepoints)
...
neural_trials.append(trial_neural.astype(np.float32))
```

iii. The AI justifies the choice of events in step 31 by saying the paper used "discrete calcium events rather than raw fluorescence" to avoid slow calcium-decay dynamics. Step 92 adds that SDK defaults exclude invalid ROIs and claims its `valid_roi` filtering matches that behavior.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural QC is filtering to `valid_roi` cells and skipping experiments with zero remaining neurons.

ii.
```python
valid = nwb_data['valid_roi']
events = events[:, valid]
...
if n_neurons == 0:
    return None, None, None, None, None, None
...
if n_neurons < 1:
    ...
    continue
```

iii. Step 92 explicitly cites the SDK's `exclude_invalid_rois=True` default and says the script "handles [that] by filtering on `valid_roi`."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to ophys timestamps, then segmented per trial by selecting the subset of ophys frames falling between each trial's `start_time` and `stop_time`.

ii.
```python
ophys_ts = nwb_data['ophys_timestamps']
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
frame_indices = np.where(frame_mask)[0]
trial_neural = events[frame_indices, :].T
```

iii. Step 31 says the AI will "align everything to ophys timestamps as the reference frame" and "extract neural and behavioral data between each trial's start and stop times."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The AI keeps each experiment at its native ophys frame rate, even though it recognized that some experiments are around 31 Hz and some multiscope experiments are around 11 Hz. It records a single metadata `time_bin_size` based on the median imaging rate across retained experiments.

ii.
```python
# Use calcium events as neural data (transpose to n_cells x n_timepoints)
# Events timestamps should match ophys timestamps
events = nwb_data['events']
...
imaging_rates = [s['imaging_rate_hz'] for s in session_info]
median_rate = np.median(imaging_rates)
time_bin_ms = 1000.0 / median_rate
...
'time_bin_size': time_bin_ms,
```

iii. Step 34 shows the AI noticed the mixed frame rates and debated resampling, then decided: "I'll stick with native ophys timestamps across all experiments and let the decoder handle the variable temporal resolution."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation table: `stim_start_times`, `stim_stop_times`, `stim_image_names`, and `stim_omitted`. It is not derived from the trials table's `initial_image_name` and `change_image_name`.

ii.
```python
data['stim_start_times'] = stim['start_time'][:]
data['stim_stop_times'] = stim['stop_time'][:]
data['stim_image_names'] = np.array([s.decode() if isinstance(s, bytes) else s for s in stim['image_name'][:]])
data['stim_omitted'] = stim['omitted'][:]
...
stim_starts = nwb_data['stim_start_times']
stim_stops = nwb_data['stim_stop_times']
stim_names = nwb_data['stim_image_names']
stim_omitted = nwb_data['stim_omitted']
```

iii. The trajectory does not explicitly name these raw variables, but step 31 says the AI wanted "image identity as a categorical variable across the 8 images" and step 87 says it forward-fills image identity through gray periods.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI builds a full-session per-frame image-identity series on the ophys time base by writing each non-omitted stimulus identity into the matching stimulus interval, encoding image names with a global sorted mapping, then forward-filling gray-screen periods within each trial and backfilling the start of a trial with the first seen image.

ii.
```python
image_at_ophys = np.full(len(ophys_ts), -1, dtype=int)
...
image_name_to_idx = {name: idx for idx, name in enumerate(all_image_names)}
...
mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
image_at_ophys[mask] = img_idx
...
trial_image = image_at_ophys[frame_indices]
...
for k in range(len(trial_image)):
    if trial_image[k] >= 0:
        last_img = trial_image[k]
    elif last_img >= 0:
        trial_image[k] = last_img
...
trial_image[:first_valid[0]] = trial_image[first_valid[0]]
```

iii. Step 87 explicitly summarizes this decision as "`Image identity`: Forward-filled during gray screen intervals." The broader rationale in step 31 is that image identity should be a time-varying categorical output aligned to the ophys timeline.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The AI first creates a session-long image-identity trace on the same `ophys_timestamps` as the neural data, then slices that trace with the exact same `frame_indices` used for the neural data in each trial.

ii.
```python
mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
image_at_ophys[mask] = img_idx
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
frame_indices = np.where(frame_mask)[0]
...
trial_neural = events[frame_indices, :].T
trial_image = image_at_ophys[frame_indices]
```

iii. Step 31 says the AI would "align everything to ophys timestamps as the reference frame." Step 87 again treats image identity as part of that shared ophys-aligned output representation.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table, specifically `stim_start_times`, `stim_stop_times`, and `stim_is_change`.

ii.
```python
data['stim_start_times'] = stim['start_time'][:]
data['stim_stop_times'] = stim['stop_time'][:]
data['stim_is_change'] = stim['is_change'][:]
...
stim_is_change = nwb_data['stim_is_change']
for i in range(len(stim_starts)):
    if stim_is_change[i] == 1.0:
        mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
        is_change_at_ophys[mask] = 1
```

iii. Step 31 says the AI wanted "image change as a binary signal at the change timepoint." The code implements that using stimulus-presentation change flags rather than trial-table `change_time`.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI creates a session-long binary vector on the ophys frame grid, sets it to 1 during any stimulus-presentation interval marked `is_change == 1.0`, and then slices that vector into each trial.

ii.
```python
is_change_at_ophys = np.zeros(len(ophys_ts), dtype=int)
...
if stim_is_change[i] == 1.0:
    mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
    is_change_at_ophys[mask] = 1
...
trial_change = is_change_at_ophys[frame_indices]
```

iii. Step 31 describes the desired variable as a binary signal around the change event; there is no more detailed trajectory justification for using stimulus intervals instead of a trial-table change window.

## 4-c. How is `output` *Image change* thresholded into categories?

i. There is no thresholding step beyond using the binary `stim_is_change` indicator. The categories are directly `0` and `1`, stored as `['no_change', 'change']`.

ii.
```python
is_change_at_ophys = np.zeros(len(ophys_ts), dtype=int)
...
if stim_is_change[i] == 1.0:
    ...
    is_change_at_ophys[mask] = 1
...
output_values = [
    all_image_names,
    ['no_change', 'change'],
```

iii. The trajectory consistently describes image change as a binary output (steps 31 and 87). It does not describe any additional thresholding rule.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is aligned to the neural data by constructing it on the ophys frame grid and then selecting the same `frame_indices` used for the neural trial matrix.

ii.
```python
if stim_is_change[i] == 1.0:
    mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
    is_change_at_ophys[mask] = 1
...
frame_indices = np.where(frame_mask)[0]
trial_neural = events[frame_indices, :].T
trial_change = is_change_at_ophys[frame_indices]
```

iii. Step 31 says all outputs should be aligned to ophys timestamps, so the same ophys indexing scheme is reused here.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and its matching timestamps in `processing/running/speed/timestamps`.

ii.
```python
data['running_speed'] = f['processing/running/speed/data'][:]
data['running_timestamps'] = f['processing/running/speed/timestamps'][:]
```

iii. Step 92 explicitly says this matches `experiment.running_speed.speed`.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI linearly interpolates running speed onto the ophys timestamps with `np.interp`, computes global percentile bin edges from all interpolated running samples across all retained experiments, and later digitizes each trial's running trace into 5 bins.

ii.
```python
def interpolate_to_ophys(signal, signal_ts, ophys_ts):
    return np.interp(ophys_ts, signal_ts, signal)
...
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'],
    nwb_data['running_timestamps'],
    ophys_ts
)
...
all_running_flat = np.concatenate(all_running)
running_edges = compute_percentile_bins(all_running_flat, n_bins=5)
...
running_binned = digitize_to_bins(running, running_edges)
```

iii. Step 31 says running speed comes at a different sampling rate and therefore "I'll need to interpolate [it] to align with the ophys timestamps." Step 87 says it uses "5 equal percentile bins across full dataset."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded into 5 percentile bins using `np.percentile` over all interpolated running-speed samples pooled across experiments, then assigned with `np.digitize`.

ii.
```python
def compute_percentile_bins(values, n_bins=5):
    valid = values[~np.isnan(values)]
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    return edges
...
running_edges = compute_percentile_bins(all_running_flat, n_bins=5)
...
running_binned = digitize_to_bins(running, running_edges)
...
[f'bin_{i}' for i in range(5)]
```

iii. Step 87 explicitly describes "Running speed: 5 equal percentile bins across full dataset."

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is first interpolated to the ophys time base, then sliced per trial using the same ophys `frame_indices` used for neural data.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'],
    nwb_data['running_timestamps'],
    ophys_ts
)
...
frame_indices = np.where(frame_mask)[0]
trial_neural = events[frame_indices, :].T
trial_running = running_at_ophys[frame_indices]
```

iii. Step 31 explicitly says running speed must be interpolated to ophys timestamps for alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from eye-tracking `pupil_tracking/width`, along with eye-tracking timestamps and the `likely_blink` mask.

ii.
```python
pupil = f['acquisition/EyeTracking/pupil_tracking']
data['pupil_width'] = pupil['width'][:]
...
eye_ts = f['acquisition/EyeTracking/eye_tracking/timestamps'][:]
data['eye_timestamps'] = eye_ts
data['likely_blink'] = f['acquisition/EyeTracking/likely_blink/data'][:]
```

iii. Step 92 explicitly says this matches `experiment.eye_tracking.pupil_width`, and step 87 says pupil diameter is blink-filtered.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI uses pupil width as diameter, sets blink samples to `NaN`, linearly interpolates remaining values onto the ophys timeline if there are more than 10 valid eye samples, computes global 5-bin percentile edges from all interpolated pupil samples across retained experiments, and digitizes each trial. If an experiment lacks pupil data entirely, it fills with `NaN`; if a trial has only missing pupil values at output time, it assigns the middle bin.

ii.
```python
pupil_raw = nwb_data['pupil_width'].copy()
blink_mask = nwb_data['likely_blink']
if blink_mask is not None:
    pupil_raw[blink_mask] = np.nan
...
if np.sum(valid_mask) > 10:
    pupil_at_ophys = np.interp(
        ophys_ts,
        nwb_data['eye_timestamps'][valid_mask],
        pupil_raw[valid_mask]
    )
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
...
all_pupil_flat = np.concatenate(all_pupil)
pupil_edges = compute_percentile_bins(all_pupil_flat, n_bins=5)
...
if pupil_edges is not None and not np.all(np.isnan(pupil)):
    pupil_binned = digitize_to_bins(pupil, pupil_edges)
else:
    pupil_binned = np.full(len(pupil), 2, dtype=int)
```

iii. Step 31 says pupil data must be interpolated to ophys timestamps. Step 87 says "Pupil diameter: 5 equal percentile bins, blinks excluded."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is thresholded into 5 global percentile bins using all interpolated pupil values pooled across experiments. Trials with all-NaN pupil after preprocessing are assigned bin `2` (the middle bin).

ii.
```python
pupil_edges = compute_percentile_bins(all_pupil_flat, n_bins=5)
...
if pupil_edges is not None and not np.all(np.isnan(pupil)):
    pupil_binned = digitize_to_bins(pupil, pupil_edges)
else:
    pupil_binned = np.full(len(pupil), 2, dtype=int)
...
[f'bin_{i}' for i in range(5)]
```

iii. Step 87 states the main intended rule, "5 equal percentile bins." The special handling of missing pupil data by forcing the median bin is not discussed in the trajectory.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by interpolating it to `ophys_timestamps` first, then slicing with the same per-trial `frame_indices` as the neural data.

ii.
```python
pupil_at_ophys = np.interp(
    ophys_ts,
    nwb_data['eye_timestamps'][valid_mask],
    pupil_raw[valid_mask]
)
...
frame_indices = np.where(frame_mask)[0]
trial_neural = events[frame_indices, :].T
trial_pupil = pupil_at_ophys[frame_indices]
```

iii. Step 31 says pupil data, like running speed, must be interpolated to ophys timestamps for alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
data['trial_hit'] = trials['hit'][:]
data['trial_miss'] = trials['miss'][:]
data['trial_false_alarm'] = trials['false_alarm'][:]
data['trial_correct_reject'] = trials['correct_reject'][:]
...
if nwb_data['trial_hit'][trial_idx]:
    return 0
elif nwb_data['trial_miss'][trial_idx]:
    return 1
elif nwb_data['trial_false_alarm'][trial_idx]:
    return 2
elif nwb_data['trial_correct_reject'][trial_idx]:
    return 3
```

iii. The trajectory repeatedly describes trial outcome as the static per-trial label "hit/miss/false_alarm/correct_reject" (steps 31 and 87).

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps the four boolean outcome flags to fixed integer labels `0..3` using `get_trial_outcome`, skips trials with no recognized outcome, and then repeats that integer across all time bins of the trial when building the final output matrix.

ii.
```python
def get_trial_outcome(trial_idx, nwb_data):
    if nwb_data['trial_hit'][trial_idx]:
        return 0
    elif nwb_data['trial_miss'][trial_idx]:
        return 1
    elif nwb_data['trial_false_alarm'][trial_idx]:
        return 2
    elif nwb_data['trial_correct_reject'][trial_idx]:
        return 3
    else:
        return -1
...
np.full(n_t, outcome, dtype=np.int64)
```

iii. Step 31 describes trial outcome as a "single label per trial." The trajectory does not separately discuss the choice to tile it across all time points, but that is how the code implements a static trial-level output within the decoder format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or problematic data by filtering unavailable NWB experiments out at load time, skipping aborted/auto-rewarded/non-go/non-catch trials, skipping unrecognized outcomes, skipping trials with fewer than 2 frames, skipping trials with no identifiable stimulus, filling missing pupil with all-NaN when interpolation is not possible, assigning the middle pupil bin if a trial's pupil trace is entirely NaN, and dropping experiments with fewer than 2 valid trials or zero valid neurons.

ii.
```python
exp_table = exp_table[exp_table['ophys_experiment_id'].isin(nwb_ids)]
...
if outcome == -1:
    continue
...
if len(frame_indices) < 2:
    continue
...
if len(first_valid) > 0:
    trial_image[:first_valid[0]] = trial_image[first_valid[0]]
else:
    continue
...
if np.sum(valid_mask) > 10:
    ...
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
...
pupil_binned = np.full(len(pupil), 2, dtype=int)
...
if neural_trials is None or len(neural_trials) < 2:
    ...
    continue
```

iii. The trajectory only partially justifies these behaviors: steps 31 and 87 justify excluding aborted and auto-rewarded trials, and step 87 says blinks are excluded. The extra handling for no-stimulus trials, unknown outcomes, and all-missing pupil values is implicit from the code rather than explicitly defended in the trajectory.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are repeated NWB file I/O and full-session preprocessing. Each experiment is opened once to collect image names, again in the first pass for behavioral statistics, and again in the second pass for actual trial extraction. Within those passes, the session-wide loops over stimuli and trials also add cost.

ii.
```python
for _, row in exp_table.iterrows():
    ...
    with h5py.File(nwb_path, 'r') as f:
        ...
...
for idx, (_, row) in enumerate(exp_table.iterrows()):
    ...
    nwb_data = load_nwb_data(nwb_path)
    ...
for idx, (_, row) in enumerate(exp_table.iterrows()):
    ...
    nwb_data = load_nwb_data(nwb_path)
    result = process_experiment(nwb_data, all_image_names)
```

iii. The trajectory does not discuss runtime bottlenecks explicitly. The closest justification is the AI's deliberate two-pass design in step 31 and its later summary in step 87 that emphasizes global percentile binning and full-dataset processing.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could have been vectorized or consolidated: the loop over all stimulus presentations to write `image_at_ophys`, the second loop over stimulus presentations for `is_change_at_ophys`, the per-trial forward-fill loop over `trial_image`, and the repeated `iterrows()` passes over the experiment table.

ii.
```python
for i in range(len(stim_starts)):
    ...
    image_at_ophys[mask] = img_idx
...
for i in range(len(stim_starts)):
    if stim_is_change[i] == 1.0:
        ...
        is_change_at_ophys[mask] = 1
...
for k in range(len(trial_image)):
    if trial_image[k] >= 0:
        last_img = trial_image[k]
    elif last_img >= 0:
        trial_image[k] = last_img
...
for idx, (_, row) in enumerate(exp_table.iterrows()):
```

iii. The trajectory does not mention vectorization opportunities. It presents the implementation as straightforward trial-by-trial and experiment-by-experiment processing.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats several operations: it reopens every NWB file in three separate passes, interpolates running and pupil once during the first pass and again inside `process_experiment`, and scans the stimulus table once to collect all image names and again later to build the per-frame image and change traces.

ii.
```python
# Collect all image names across all experiments
for _, row in exp_table.iterrows():
    ...
    with h5py.File(nwb_path, 'r') as f:
        ...

# First pass: collecting behavioral statistics for binning...
for idx, (_, row) in enumerate(exp_table.iterrows()):
    ...
    nwb_data = load_nwb_data(nwb_path)
    ...

# Second pass: processing experiments...
for idx, (_, row) in enumerate(exp_table.iterrows()):
    ...
    nwb_data = load_nwb_data(nwb_path)
```

iii. The trajectory does not flag this as a concern; it implicitly accepts the multi-pass design because the AI wanted global bins before final assembly (steps 31 and 87).

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads several arrays that are never used downstream (`dff`, `events_timestamps`, `cell_specimen_ids`, `pupil_area`), returns unused objects from `process_experiment` (`trial_outcomes`, `running_at_ophys`, `pupil_at_ophys`, `events`), allocates an unused `subjects_list`, and computes running/pupil bin edges using many session-level samples that are never themselves saved because only trial segments enter the final dataset.

ii.
```python
dff = f['processing/ophys/dff/traces/data'][:]
data['dff'] = dff
...
data['events_timestamps'] = events_ts
data['cell_specimen_ids'] = cst['cell_specimen_id'][:]
...
data['pupil_area'] = pupil['area'][:]
...
return neural_trials, output_trials, trial_outcomes, running_at_ophys, pupil_at_ophys, events
...
subjects_list = []
...
all_running.append(running)
...
all_pupil.append(pupil)
```

iii. The trajectory does not acknowledge this extra work. It frames the unused dF/F load as being there "for fallback / comparison" and otherwise focuses on successful validation rather than efficiency.
