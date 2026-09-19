# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files using h5py, bypassing the Allen SDK. It reads a metadata CSV (`ophys_experiment_table.csv`) to discover experiments, filters to only downloaded NWB files, and excludes passive sessions. Each experiment is loaded individually via `load_experiment_data()` which reads neural events, running speed, pupil data, trials, and stimulus presentations from the NWB file.

ii.
```python
def get_experiment_list(sample=False):
    expt_table = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
    downloaded_ids = set()
    for f in os.listdir(EXPT_DIR):
        if f.endswith('.nwb'):
            eid = int(f.split('_')[-1].replace('.nwb', ''))
            downloaded_ids.add(eid)
    expt_table = expt_table[expt_table.ophys_experiment_id.isin(downloaded_ids)]
    expt_table = expt_table[expt_table.passive == False]
    ...

def load_experiment_data(expt_id):
    fname = os.path.join(EXPT_DIR, f'behavior_ophys_experiment_{expt_id}.nwb')
    f = h5py.File(fname, 'r')
    data['events'] = f['processing/ophys/event_detection/data'][:].T
    data['ophys_timestamps'] = f['processing/ophys/dff/traces/timestamps'][:]
    data['valid_roi'] = f['processing/ophys/image_segmentation/cell_specimen_table/valid_roi'][:]
    ...
```

iii. The AI chose to use h5py directly rather than the Allen SDK to avoid SDK overhead and improve loading speed. It filters to only active (non-passive) sessions and only experiments with downloaded NWB files.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the `subject_id` field read from each NWB file (`general/subject/subject_id`). Unique subjects are accumulated as experiments are processed.

ii.
```python
data['subject_id'] = f['general/subject/subject_id'][()]
...
subj = str(result['subject_id'])
if subj not in unique_subjects:
    unique_subjects.append(subj)
subject_idx_list.append(unique_subjects.index(subj))
```

iii. Subject IDs are extracted directly from each NWB file. The subject list is built incrementally as experiments are processed.

## 1-c. How are the data split into sessions?

i. Each experiment (imaging plane) is treated as a separate "session" in the output. The AI does NOT group multiple imaging planes from the same ophys session together. Each NWB file becomes one entry in the sessions list.

ii.
```python
for i, (_, expt_info) in enumerate(expt_table.iterrows()):
    expt_id = expt_info['ophys_experiment_id']
    ...
    result = process_experiment(expt_id, ...)
    ...
    all_neural.append(result['neural'])
```

iii. The AI treats each experiment independently. The CONVERSION_NOTES.md notes that each NWB file corresponds to one imaging plane, but the code does not merge planes from the same session.

## 1-d. How are the data split into trials?

i. Trials are defined using the `intervals/trials` data from the NWB file. Go and Catch trials are included; Aborted and Auto-rewarded trials are excluded. For each included trial, the time window from `start_time` to `stop_time` is used.

ii.
```python
trial_data = raw['trials']
include_mask = trial_data['go'] | trial_data['catch']
include_indices = np.where(include_mask)[0]
...
for trial_idx in include_indices:
    trial_start = trial_data['start_time'][trial_idx]
    trial_stop = trial_data['stop_time'][trial_idx]
```

iii. The AI uses the `go` and `catch` boolean fields to select trials, matching the instruction to include Go and Catch trials but exclude Aborted and Auto-rewarded trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by including only Go and Catch trials (excluding Aborted and Auto-rewarded). Trials with fewer than 2 common time bins are skipped. Trials where the outcome cannot be determined (returns -1) are also skipped. Sessions with fewer than 2 valid trials are excluded.

ii.
```python
include_mask = trial_data['go'] | trial_data['catch']
...
if trial_neural is None:  # < 2 time bins
    continue
...
if outcome == -1:
    continue
...
if len(neural_trials) < 2:
    return None
```

iii. The filtering follows the task instructions for trial types. Additional quality filters ensure trials have sufficient data and valid outcomes.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `event_detection/data` (detected calcium events) from the NWB file, NOT from `dff_traces` (dF/F).

ii.
```python
data['events'] = f['processing/ophys/event_detection/data'][:].T  # (n_neurons, n_timepoints)
...
events = raw['events'][valid_mask]
```

iii. The AI chose detected calcium events based on the paper stating "discrete calcium events regressed from raw fluorescence" and the CONVERSION_NOTES stating "Raw events are used for analysis (paper: 'detected calcium events')".

## 2-b. How is the `neural` data processed?

i. Neural events are filtered by `valid_roi` to exclude invalid ROIs, then resampled from native ophys timestamps to common time bins of 93.23ms (~10.73 Hz, MESO rate). Within each common bin, events from ophys frames falling in that bin are summed.

ii.
```python
valid_mask = raw['valid_roi']
events = raw['events'][valid_mask]
...
def resample_events_to_common_bins(events, ophys_ts, trial_start, trial_stop):
    common_ts = np.arange(trial_start, trial_stop, COMMON_DT)
    ...
    bin_assignments = np.digitize(trial_ophys_ts, bin_edges) - 1
    resampled = np.zeros((n_neurons, n_common), dtype=np.float32)
    for b in range(n_common):
        mask = bin_assignments == b
        if mask.any():
            resampled[:, b] = trial_events[:, mask].sum(axis=1)
    return resampled, common_ts
```

iii. The AI resamples to a common time bin because experiments use different frame rates (CAM2P ~31Hz, MESO ~10.7Hz), and the instructions state "time bins should be the same size for all trials and sessions." The MESO rate was chosen as the common rate.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered using the `valid_roi` flag from the NWB cell specimen table. Only neurons with `valid_roi=True` are included.

ii.
```python
data['valid_roi'] = f['processing/ophys/image_segmentation/cell_specimen_table/valid_roi'][:]
...
valid_mask = raw['valid_roi']
events = raw['events'][valid_mask]
```

iii. The AI uses the Allen SDK's built-in validity flag, noting it matches the SDK default of `exclude_invalid_rois=True`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start time. Events within each trial's `start_time` to `stop_time` window are resampled into common time bins starting from `trial_start`.

ii.
```python
common_ts = np.arange(trial_start, trial_stop, COMMON_DT)
...
trial_mask = (ophys_ts >= trial_start - COMMON_DT) & (ophys_ts < trial_stop + COMMON_DT)
```

iii. The instructions say to "temporally align based on ophys timestamp." The AI aligns to trial start and resamples to common bins.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI resamples all data to a common time bin of 93.23ms (~10.73 Hz), which matches the MESO imaging rate. This is a rebinning for CAM2P experiments (native ~31 Hz) and approximately native for MESO experiments.

ii.
```python
COMMON_DT = 0.09323  # seconds, ~10.73 Hz
...
common_ts = np.arange(trial_start, trial_stop, COMMON_DT)
```

iii. The AI chose this common bin size because experiments have different frame rates (CAM2P ~31Hz vs MESO ~10.7Hz) and the instructions require "time bins should be the same size for all trials and sessions."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table in the NWB file, specifically the `image_name`, `start_time`, and `stop_time` fields.

ii.
```python
stim_keys = [k for k in f['intervals'].keys() if 'Natural_Images' in k]
...
stim = f[f'intervals/{stim_key}']
for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
    stim_data[key] = stim[key][:]
```

iii. The AI uses the full stimulus presentations table to determine which image is on screen at each timepoint, rather than using the trials table's `initial_image_name`/`change_image_name`.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each timepoint, the most recent stimulus presentation is found via `np.searchsorted`. The image name is looked up and mapped to an integer index. Omitted stimuli are handled by carrying forward the last valid image identity.

ii.
```python
def get_image_at_timepoints(common_ts, stim_data, image_to_idx):
    indices = np.searchsorted(start_times, common_ts, side='right') - 1
    for t in range(n_tp):
        idx = indices[t]
        img = image_names[idx]
        if img == 'omitted':
            image_id[t] = last_valid
        elif img in image_to_idx:
            image_id[t] = image_to_idx[img]
            last_valid = image_to_idx[img]
```

iii. The AI uses the stimulus presentations timeline to create a per-timepoint image identity. Omitted images are replaced with the last shown image.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the same common time bin timestamps as the neural data, so alignment is inherent.

ii.
```python
img_id = get_image_at_timepoints(common_ts, raw['stimulus_presentations'], image_to_idx)
```

iii. Using the same `common_ts` array for both neural and output data ensures alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in the `stimulus_presentations` table, along with `start_time` and `stop_time` of each stimulus.

ii.
```python
def get_image_change_at_timepoints(common_ts, stim_data):
    is_change = stim_data.get('is_change', np.zeros(len(stim_data['start_time'])))
    change_indices = np.where(np.array(is_change) == 1)[0]
    for ci in change_indices:
        mask = (common_ts >= start_times[ci]) & (common_ts < stop_times[ci])
        change[mask] = 1
```

iii. The AI marks timepoints as "change" if they fall within the stimulus presentation window of a change stimulus.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary array is created where 1 indicates timepoints during a change stimulus presentation (from its start_time to stop_time, typically 250ms), and 0 otherwise. This applies to all change stimuli regardless of whether the trial is Go or Catch.

ii.
```python
change_indices = np.where(np.array(is_change) == 1)[0]
for ci in change_indices:
    mask = (common_ts >= start_times[ci]) & (common_ts < stop_times[ci])
    change[mask] = 1
```

iii. The AI uses the stimulus presentation duration (250ms) as the change window, rather than a 750ms window.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is a binary variable (0 or 1), no thresholding is needed.

ii. See 4-b.

iii. N/A - binary by definition.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity - computed at the same common time bin timestamps.

ii. See 4-a.

iii. Uses the same `common_ts` for alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
data['running_speed'] = f['processing/running/speed/data'][:]
data['running_timestamps'] = f['processing/running/speed/timestamps'][:]
```

iii. This is the standard running speed data from the Allen SDK pipeline.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is interpolated from its native timestamps to the common time bin timestamps, then discretized into 5 percentile-based bins. Percentiles are computed globally across all experiments (subsampled to 2000 values per experiment for efficiency). NaN values are mapped to the middle bin (bin 2).

ii.
```python
run_interp = interpolate_to_common(raw['running_speed'], raw['running_timestamps'], common_ts)
run_binned = np.digitize(run_interp, running_percentiles[1:-1])
run_binned = np.clip(run_binned, 0, 4).astype(np.int64)
run_binned[np.isnan(run_interp)] = 2
```

iii. Global percentile binning ensures consistent categories across sessions. NaN values are assigned to the middle bin (2) as a neutral default.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using global percentile edges (0th, 20th, 40th, 60th, 80th, 100th percentiles). `np.digitize` assigns each value to a bin.

ii.
```python
running_percentiles = np.percentile(all_running, np.linspace(0, 100, 6))
...
run_binned = np.digitize(run_interp, running_percentiles[1:-1])
run_binned = np.clip(run_binned, 0, 4).astype(np.int64)
```

iii. Equal percentile bins ensure roughly balanced class counts for decoding.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same common time bin timestamps as the neural data.

ii.
```python
run_interp = interpolate_to_common(raw['running_speed'], raw['running_timestamps'], common_ts)
```

iii. Using the same `common_ts` ensures alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil data is derived from `acquisition/EyeTracking/pupil_tracking/area` and its timestamps. Note: this is pupil **area**, not pupil width/diameter.

ii.
```python
data['pupil_area'] = f['acquisition/EyeTracking/pupil_tracking/area'][:]
data['pupil_timestamps'] = f['acquisition/EyeTracking/pupil_tracking/timestamps'][:]
```

iii. The AI uses pupil area from the eye tracking data. Missing pupil data (KeyError) is handled by setting `has_pupil=False`.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area is interpolated to common timestamps (NaN values are kept). It is then discretized into 5 percentile bins computed globally across all experiments (subsampled). NaN values are mapped to the middle bin (bin 2). No blink removal is performed before interpolation.

ii.
```python
if raw['has_pupil']:
    pupil_interp = interpolate_to_common(raw['pupil_area'], raw['pupil_timestamps'], common_ts)
...
pupil_binned = np.digitize(pupil_interp, pupil_percentiles[1:-1])
pupil_binned = np.clip(pupil_binned, 0, 4).astype(np.int64)
pupil_binned[np.isnan(pupil_interp)] = 2
```

iii. Same approach as running speed. No explicit blink removal is performed (NaN values in the source data serve as implicit exclusion during interpolation).

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 equal percentile bins using globally computed edges.

ii. See 5-c (same approach).

iii. Same justification as running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed - interpolated to the same common time bin timestamps.

ii. See 5-d (same approach).

iii. Same alignment mechanism.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean fields `hit`, `miss`, `correct_reject`, and `false_alarm` in the NWB trials data.

ii.
```python
def get_trial_outcome(trial_data, trial_idx):
    if trial_data['hit'][trial_idx]: return 0
    elif trial_data['miss'][trial_idx]: return 1
    elif trial_data['correct_reject'][trial_idx]: return 2
    elif trial_data['false_alarm'][trial_idx]: return 3
    else: return -1
```

iii. These four outcome types cover all non-aborted, non-auto-rewarded trials in the change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is mapped to integer codes: hit=0, miss=1, correct_reject=2, false_alarm=3. The value is broadcast as a static per-trial value across all timepoints. Trials with no matching outcome (returns -1) are excluded.

ii.
```python
outcome = get_trial_outcome(trial_data, trial_idx)
if outcome == -1:
    continue
...
trial_output = np.stack([
    ...
    np.full(n_common, outcome, dtype=np.int64)
], axis=0)
```

iii. The ordering differs from the reference (which uses hit=0, miss=1, false_alarm=2, correct_reject=3). The `output_values` list also reflects this different ordering: `['hit', 'miss', 'correct_reject', 'false_alarm']`.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Failed experiments**: Caught by try/except, the experiment is skipped with an error message.
- **Missing pupil data**: If eye tracking data is not available (KeyError), pupil values are set to NaN and assigned the middle bin (2).
- **NaN in interpolated values**: NaN running/pupil values are mapped to the middle bin (2) after interpolation.
- **Too few trials**: Experiments with fewer than 2 valid trials are skipped.
- **Trials with < 2 time bins**: Skipped when `resample_events_to_common_bins` returns None.
- **Unknown trial outcome**: Trials returning outcome -1 are skipped.

ii.
```python
try:
    result = process_experiment(...)
except Exception as e:
    print(f'  ERROR: {e}')
    continue
...
if raw['has_pupil']:
    ...
else:
    pupil_interp = np.full(n_common, np.nan)
...
run_binned[np.isnan(run_interp)] = 2
```

iii. The try/except ensures robustness. NaN-to-middle-bin mapping is a design choice to avoid extreme bin assignments for missing data.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading NWB files via h5py and the global statistics collection pass, which reads running speed, pupil, and image data from every NWB file before processing begins.

ii.
```python
def collect_global_stats(expt_table):
    for i, expt_id in enumerate(expt_ids):
        with h5py.File(fname, 'r') as f:
            run_data = f['processing/running/speed/data'][:]
            ...
```

iii. The global stats pass requires opening every NWB file once to collect running/pupil values for percentile computation.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The event resampling loop in `resample_events_to_common_bins` iterates over each time bin sequentially, which could be vectorized using sparse matrix operations or `np.add.at`. The image identity lookup loop in `get_image_at_timepoints` iterates over each timepoint.

ii.
```python
for b in range(n_common):
    mask = bin_assignments == b
    if mask.any():
        resampled[:, b] = trial_events[:, mask].sum(axis=1)
...
for t in range(n_tp):
    idx = indices[t]
    ...
```

iii. These loops are O(n_timepoints) per trial and could be vectorized, though data loading dominates overall runtime.

## 9-c. What processing does the code repeat multiple times?

i. The code performs two passes over all NWB files: once in `collect_global_stats()` to compute percentile bin edges and image names, and once in the main processing loop to extract trials. This means every NWB file is opened and partially read twice.

ii.
```python
global_stats = collect_global_stats(expt_table)  # Pass 1: read all files
...
for i, (_, expt_info) in enumerate(expt_table.iterrows()):  # Pass 2: read all files again
    result = process_experiment(expt_id, ...)
```

iii. The two-pass approach is needed because percentile bins must be computed globally before discretization. However, running/pupil data is read from each NWB file in both passes.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The global statistics pass subsamples 2000 values per experiment for percentile computation, reading and then discarding the full arrays. The code also reads stimulus presentation data (omitted flags, stop times) that may not be strictly necessary if using the trials table approach.

ii.
```python
if len(valid_run) > 2000:
    rng = np.random.RandomState(int(expt_id) % (2**31))
    valid_run = rng.choice(valid_run, 2000, replace=False)
```

iii. The subsampling is an efficiency optimization that trades exactness for speed in percentile computation.
