# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads local NWB files directly with `h5py`, not through the AllenSDK cache. It reads `ophys_experiment_table.csv`, filters to NWB files that are physically present on disk and to a hard-coded list of active session types, then loads each experiment file in three passes: one scan for image names, one scan for running/pupil percentile statistics, and one full processing pass.

ii.
```python
exp_table = pd.read_csv(META_DIR / 'ophys_experiment_table.csv')
...
mask = (
    exp_table['ophys_experiment_id'].isin(nwb_ids) &
    exp_table['session_type'].isin(ACTIVE_SESSION_TYPES)
)
active_exps = exp_table[mask].copy()
```

```python
with h5py.File(nwb_path, 'r') as f:
    ...
    events_data = f['processing']['ophys']['event_detection']['data'][()]
    trials_grp = f['intervals']['trials']
    running_speed = f['processing']['running']['speed']['data'][()]
```

iii. In `CONVERSION_NOTES.md`, the AI says it is working from the on-disk subset and that `convert_data.py` "loads NWB files via h5py" with a "3-pass approach." The notes justify excluding passive sessions because the task is active visual behavior.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values from the metadata table, converted to strings and indexed with a global `subject_to_idx` map.

ii.
```python
subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])
```

iii. The notes explicitly say "Subject IDs: Use `mouse_id` from experiment table."

## 1-c. How are the data split into sessions?

i. The AI treats each NWB experiment file (`ophys_experiment_id`) as one output session. It does not group multiple experiments that share an `ophys_session_id`.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = NWB_DIR / f'behavior_ophys_experiment_{eid}.nwb'
    ...
    all_neural.append(session_neural)
```

iii. The notes state, "Each NWB experiment = one 'session' in output format (one imaging plane with its own neurons)." The trajectory shows the AI explicitly chose this as a key design decision.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB `intervals/trials` table. For each valid trial, the AI uses the trial's `start_time` and `stop_time`, finds all 30 Hz resampled timestamps within that interval, and treats that interval as one variable-length trial.

ii.
```python
trials = raw_data['trials']
...
for trial_idx in valid_indices:
    t_trial_start = trials['start_time'][trial_idx]
    t_trial_stop = trials['stop_time'][trial_idx]
    trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
    trial_time_indices = np.where(trial_mask)[0]
```

iii. The notes say "Trial window: Use trial `start_time` to `stop_time` from trials table. Variable length across trials."

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only Go or Catch trials, excludes aborted and auto-rewarded trials, skips trials with fewer than 3 resampled time bins, skips trials whose outcome is not one of hit/miss/false alarm/correct reject, and drops experiments with fewer than 2 remaining trials.

ii.
```python
valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
valid_indices = np.where(valid_mask)[0]
...
if len(trial_time_indices) < 3:
    continue
...
else:
    continue  # Unknown outcome, skip
...
if len(neural_trials) < 2:
    return None
```

iii. The notes justify excluding passive sessions, aborted trials, and auto-rewarded trials because the decoder target is active behavior. They also note the two-trial minimum as a dataset validity requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the NWB `processing/ophys/event_detection/data` array, filtered to ROIs where `valid_roi` is true.

ii.
```python
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][()].astype(bool)
...
events_data = f['processing']['ophys']['event_detection']['data'][()]
events_valid = events_data[:, valid_roi]
```

iii. The notes call out a deliberate decision: "Neural signal: events (not dF/F)." The justification given is that the paper analyzed discrete calcium events and that matching the paper matters more than using the more common dF/F trace.

## 2-b. How is the `neural` data processed?

i. The events are linearly interpolated from irregular ophys timestamps onto a regular 30 Hz grid, clipped to be non-negative, then sliced by trial and transposed to `(n_cells, n_timepoints)`.

ii.
```python
dt = 1.0 / target_rate
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The notes justify 30 Hz interpolation by citing the paper's use of "a consistent set of 30hz timestamps." They also note that events should remain non-negative.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural QC is `valid_roi == True`; experiments with zero valid ROIs are skipped.

ii.
```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
n_valid = valid_roi.sum()
if n_valid == 0:
    print(f"  WARNING: No valid ROIs in experiment {experiment_id}, skipping")
    return None
events_valid = events_data[:, valid_roi]
```

iii. The notes justify this by pointing to the Allen ROI classifier and saying `valid_roi` should be respected when reading raw NWB files directly.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to trial start in the sense that each trial begins at `start_time` and ends at `stop_time`, but the actual sampled bins are taken from a regular 30 Hz timeline rather than directly from `ophys_timestamps`.

ii.
```python
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
...
'temporal_alignment_event': 'Trial start time (first stimulus onset of trial)',
```

iii. The notes first considered change alignment in the trajectory, but the final notes and metadata settle on "Trial start time" as the alignment event and on `start_time` to `stop_time` trial windows.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 30 Hz grid, i.e. 33.33 ms bins. Yes, the AI rebins/resamples all streams to this grid by interpolation.

ii.
```python
TARGET_RATE_HZ = 30.0
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ
...
'time_bin_size': TIME_BIN_MS,
'target_rate_hz': TARGET_RATE_HZ,
```

iii. The notes explicitly justify this with the paper's statement about interpolating to 30 Hz and with the desire to unify Scientifica and Multiscope recordings.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentation table, specifically `stim['image_name']`, `stim['start_time']`, and `stim['omitted']`, rather than from the trial table's `initial_image_name` and `change_image_name`.

ii.
```python
stim_data = {
    'start_time': stim['start_time'][()],
    'image_name': stim['image_name'][()],
    'is_change': stim['is_change'][()].astype(bool),
    'omitted': stim['omitted'][()],
}
```

iii. The notes justify this by saying image identity should reflect the image actually on screen and that omitted flashes should keep the previous image identity.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI builds a sorted global list of non-omitted image names across all experiments. Within each trial, each time bin is assigned the most recent non-omitted stimulus identity using `searchsorted`; during gray periods and omitted flashes it carries forward the previous image.

ii.
```python
non_omitted = ~stim_data['omitted']
stim_starts = stim_data['start_time'][non_omitted]
stim_names = stim_data['image_name'][non_omitted]
insert_idx = np.searchsorted(stim_starts, timepoints, side='right') - 1
...
img_name = stim_names[insert_idx[i]]
```

```python
global_image_names = sorted(all_image_names_set)
...
img_id_global = np.array([local_to_global.get(v, 0) for v in trial_data_out['image_identity']], dtype=np.int64)
```

iii. The notes justify two special rules: "Image identity during gray screen: use the identity of the image that was just shown" and "Image identity for omitted flashes: continue with previous image identity."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is sampled at exactly the same per-trial resampled timestamps as the neural data, because it is computed from `trial_ts`, which is the same subset of `regular_ts` used for `neural_trial`.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
```

iii. The notes justify this by saying all streams are first put onto one consistent 30 Hz time base.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus presentation table's `is_change`, `start_time`, and `omitted` fields.

ii.
```python
change_mask = stim_data['is_change'] & ~stim_data['omitted']
change_starts = stim_data['start_time'][change_mask]
change_stops = stim_data['stop_time'][change_mask]
```

iii. The notes describe image change as "1 during 750ms window starting at change onset" and tie it to stimulus timing rather than to trial outcome fields.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each real, non-omitted change stimulus, the AI marks a 750 ms window beginning at the change onset as 1 and all other time bins as 0.

ii.
```python
change_signal = np.zeros(n_tp, dtype=np.int64)
for cs, ce in zip(change_starts, change_stops):
    mask = (timepoints >= cs) & (timepoints < cs + 0.75)
    change_signal[mask] = 1
```

iii. The notes justify the 750 ms window as one 250 ms flash plus the 500 ms gray interval.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is binary: `0` for no change and `1` for change.

ii.
```python
change_signal = np.zeros(n_tp, dtype=np.int64)
...
change_signal[mask] = 1
...
['no_change', 'change']
```

iii. No separate thresholding argument is documented beyond the task requirement that image change be binary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like image identity, image change is computed on the same `trial_ts` time bins used for neural data.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The notes justify this with the same "common 30 Hz grid" logic used for all time-varying outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from the NWB running module's `speed/data` and `speed/timestamps`.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. The notes treat this as the standard running wheel signal for the dataset.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated onto the regular 30 Hz grid for each experiment, then discretized into five global percentile bins computed from all raw running samples collected across all selected experiments.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
```

```python
all_running_cat = np.concatenate(all_running_values) if all_running_values else np.array([0.0])
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
...
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. The notes justify 30 Hz interpolation by paper consistency and global percentile bins by decoder balance.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five percentile bins using global bin edges from all collected running samples; `digitize_to_bins` maps NaNs to bin 0.

ii.
```python
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
...
def digitize_to_bins(values, bin_edges):
    binned = np.digitize(values, bin_edges[1:-1])
    ...
    binned[np.isnan(values)] = 0
```

iii. The notes explicitly say running speed should be "discretize[d] into 5 percentile bins" and computed globally.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is first resampled to the same regular 30 Hz time base as neural data, then trial-sliced with the same `trial_time_indices`.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
running_trial = running_resampled[trial_time_indices]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The notes justify this by saying all streams should be on one consistent time base before trial extraction.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The AI derives this output from eye tracking `pupil_tracking/area` and `timestamps`, plus the `likely_blink` mask. It does not use pupil width/diameter directly.

ii.
```python
pupil_tracking = f['acquisition']['EyeTracking']['pupil_tracking']
pupil_area = pupil_tracking['area'][()]
pupil_ts = pupil_tracking['timestamps'][()]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
```

iii. The notes explicitly say "Use pupil area as proxy for diameter."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames are set to NaN, the NaNs are linearly filled in on the pupil-area trace, the filled trace is interpolated to the 30 Hz regular grid, and the result is later discretized with global percentile bins computed from all non-blink pupil-area samples.

ii.
```python
if likely_blink is not None:
    pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
```

```python
valid_pupil = pupil_area[~np.isnan(pupil_area)]
...
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
...
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. The notes justify this by saying blinks should not define the bins and should be interpolated through before resampling.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The AI uses five global percentile bins derived from pooled non-NaN pupil-area values. NaNs are mapped to bin 0 during digitization.

ii.
```python
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
...
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. The notes mirror the running-speed decision: global five-bin percentile discretization for a categorical decoder output.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil is aligned by interpolating to the same regular 30 Hz grid used for neural data and then slicing each trial with the same indices.

ii.
```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
pupil_trial = pupil_resampled[trial_time_indices]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The notes justify this with the same common-time-base argument as the other time-varying streams.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial fields `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
if trials['hit'][trial_idx]:
    outcome = 0
elif trials['miss'][trial_idx]:
    outcome = 1
elif trials['false_alarm'][trial_idx]:
    outcome = 2
elif trials['correct_reject'][trial_idx]:
    outcome = 3
```

iii. The notes identify trial outcome as a static four-class per-trial target.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps each trial to one of four integer codes and then broadcasts that code across all time bins in the trial before stacking it with the other outputs.

ii.
```python
if trials['hit'][trial_idx]:
    outcome = 0
...
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)
```

iii. The notes justify this because trial outcome is static per-trial but the target format allows it to be represented as a constant time series.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI skips experiments with no valid ROIs or no usable stimulus table, warns and continues when pupil data are missing, fills missing pupil values by interpolation, uses NaN-filled pupil traces when the whole pupil stream is absent, maps NaNs to category 0 during discretization, skips very short trials, and skips experiments with too few remaining trials.

ii.
```python
if n_valid == 0:
    return None
...
if stim_key is None:
    return None
...
except (KeyError, Exception):
    print(f"  WARNING: No pupil tracking data in {experiment_id}")
...
if len(trial_time_indices) < 3:
    continue
...
if pupil_resampled is not None:
    pupil_trial = pupil_resampled[trial_time_indices]
else:
    pupil_trial = np.full(n_tp, np.nan)
```

iii. The notes say blink-related NaNs should be interpolated, that non-blink values should determine the percentile bins, and that sessions/trials failing minimum validity checks should be skipped rather than crash the run.

## 9-a. What are the most time-consuming steps of the code?

i. The expensive parts are repeated NWB file I/O and the main per-experiment processing pass. The code scans the same files in three passes, then loads and processes each experiment again for trial extraction.

ii.
```python
# First pass: determine global image names
for _, row in exp_table.iterrows():
    with h5py.File(nwb_path, 'r') as f:
        ...

# Pass 2: Collect running speed and pupil data
for idx, (_, row) in enumerate(exp_table.iterrows()):
    with h5py.File(nwb_path, 'r') as f:
        ...

# Pass 3: Processing experiments
for idx, (_, row) in enumerate(exp_table.iterrows()):
    raw_data = load_experiment_data(nwb_path, eid)
```

iii. The notes explicitly list "Multiple passes over NWB files" and "Sequential processing of experiments" as the main inefficiencies.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are obvious vectorization targets: the column-wise interpolation loop in `interpolate_to_regular_grid`, the per-timepoint loop in `get_image_at_timepoints`, the per-change loop in `get_image_change_at_timepoints`, and the per-trial assembly loop.

ii.
```python
for i in range(data.shape[1]):
    result[:, i] = np.interp(target_timestamps, timestamps, data[:, i])
```

```python
for i in range(n_tp):
    if insert_idx[i] >= 0:
        img_name = stim_names[insert_idx[i]]
```

iii. The notes mention sequential processing as an inefficiency but do not give a detailed vectorization analysis beyond that.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats full-file scans across three passes, recomputes local-to-global image mappings per experiment, and repeatedly calls `global_image_names.index(name)` inside loops instead of using a prebuilt dict.

ii.
```python
for _, row in exp_table.iterrows():  # pass 1
    ...
for idx, (_, row) in enumerate(exp_table.iterrows()):  # pass 2
    ...
for idx, (_, row) in enumerate(exp_table.iterrows()):  # pass 3
    ...
```

```python
for local_idx, name in enumerate(result['all_image_names']):
    if name in global_image_names:
        local_to_global[local_idx] = global_image_names.index(name)
```

iii. The notes explicitly acknowledge "Multiple passes over NWB files" as an implementation inefficiency.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does some work that is unused later: it reads `change_stops` but never uses it, allocates `trial_outcomes` but never fills or consumes it, allocates a one-element `trial_outcome` array that is not used, records `t_start_global` without using it, and imports parallel-processing utilities that are never used.

ii.
```python
change_stops = stim_data['stop_time'][change_mask]
...
trial_outcomes = []
...
trial_outcome = np.array([trial_data_out['trial_outcome']], dtype=np.int64)
...
t_start_global = time.time()
```

iii. The notes do not explicitly discuss these discarded computations; they are visible from the code itself.
