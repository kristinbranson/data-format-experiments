# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files using `h5py`, bypassing the Allen SDK's higher-level API. It reads an experiment table CSV from project metadata, filters to only downloaded NWB files, and excludes passive sessions. Each experiment's NWB file is loaded individually via `load_experiment_data()`.

ii.
```python
DATA_DIR = 'data/visual-behavior-ophys-1.1.0'
EXPT_DIR = os.path.join(DATA_DIR, 'behavior_ophys_experiments')
META_DIR = os.path.join(DATA_DIR, 'project_metadata')

def get_experiment_list(sample=False):
    expt_table = pd.read_csv(os.path.join(META_DIR, 'ophys_experiment_table.csv'))
    downloaded_ids = set()
    for f in os.listdir(EXPT_DIR):
        if f.endswith('.nwb'):
            eid = int(f.split('_')[-1].replace('.nwb', ''))
            downloaded_ids.add(eid)
    expt_table = expt_table[expt_table.ophys_experiment_id.isin(downloaded_ids)]
    expt_table = expt_table[expt_table.passive == False]
    return expt_table

def load_experiment_data(expt_id):
    fname = os.path.join(EXPT_DIR, f'behavior_ophys_experiment_{expt_id}.nwb')
    f = h5py.File(fname, 'r')
    # ... reads events, timestamps, running, pupil, trials, stimulus_presentations
```

iii. The AI chose to use h5py directly for efficiency, avoiding SDK overhead. It filters to only active (non-passive) sessions since passive sessions have no behavioral task. From CONVERSION_NOTES: "Exclude passive (no behavioral task)".

## 1-b. How are the data split into subjects?

i. Subjects are identified by the `subject_id` field read from each NWB file's `general/subject/subject_id` path. Unique subjects are accumulated as experiments are processed.

ii.
```python
data['subject_id'] = f['general/subject/subject_id'][()]
if isinstance(data['subject_id'], bytes):
    data['subject_id'] = data['subject_id'].decode()
# ...
subj = str(result['subject_id'])
if subj not in unique_subjects:
    unique_subjects.append(subj)
subject_idx_list.append(unique_subjects.index(subj))
```

iii. The subject ID is extracted per-experiment from the NWB metadata. This is equivalent to the reference's `mouse_id` from the experiment table.

## 1-c. How are the data split into sessions?

i. The AI treats each individual experiment (one imaging plane) as a separate session. It does NOT group multiple experiments/planes from the same ophys session together. Each NWB file becomes one session in the output.

ii.
```python
for i, (_, expt_info) in enumerate(expt_table.iterrows()):
    expt_id = expt_info['ophys_experiment_id']
    # ... processes each experiment independently as a session
    result = process_experiment(expt_id, ...)
    all_neural.append(result['neural'])
```

iii. From CONVERSION_NOTES: "Each NWB = one ophys experiment = one imaging plane". The AI processes each experiment independently rather than grouping by ophys_session_id.

## 1-d. How are the data split into trials?

i. Trials are defined using the `intervals/trials` data from the NWB file. Each trial runs from `start_time` to `stop_time`. Only Go and Catch trials are included (filtering by `go | catch`).

ii.
```python
trial_data = raw['trials']
include_mask = trial_data['go'] | trial_data['catch']
include_indices = np.where(include_mask)[0]

for trial_idx in include_indices:
    trial_start = trial_data['start_time'][trial_idx]
    trial_stop = trial_data['stop_time'][trial_idx]
    trial_neural, common_ts = resample_events_to_common_bins(
        events, ophys_ts, trial_start, trial_stop
    )
```

iii. The instructions say to include Go and Catch trials but exclude Aborted and Auto-rewarded. The AI filters directly with `go | catch` which achieves the same result since these categories are mutually exclusive.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to include only Go and Catch trials (`go | catch`). Trials with fewer than 2 common time bins are skipped. Experiments with fewer than 2 valid trials are excluded. Experiments with 0 valid neurons after ROI filtering are excluded.

ii.
```python
include_mask = trial_data['go'] | trial_data['catch']
include_indices = np.where(include_mask)[0]
if len(include_indices) < 2:
    print(f'  WARNING: Too few trials in experiment {expt_id}')
    return None
# ...
if trial_neural is None:  # fewer than 2 time bins
    continue
# ...
outcome = get_trial_outcome(trial_data, trial_idx)
if outcome == -1:
    continue
# ...
if len(neural_trials) < 2:
    return None
```

iii. The AI's `go | catch` filter is functionally equivalent to the reference's `~aborted & ~auto_rewarded & change_time.notna()` for this dataset, as Go and Catch are the non-aborted, non-auto-rewarded trial types with valid change times.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses **detected calcium events** from `processing/ophys/event_detection/data` in the NWB file, NOT dF/F traces.

ii.
```python
data['events'] = f['processing/ophys/event_detection/data'][:].T  # (n_neurons, n_timepoints)
```

iii. From CONVERSION_NOTES: "Neural data: detected calcium events (not raw dff)" and "Paper uses raw events for analysis", citing the paper's description of "discrete calcium events regressed from raw fluorescence".

## 2-b. How is the `neural` data processed?

i. The neural events are filtered by `valid_roi`, then resampled to a common time bin size (93.23 ms, matching the MESO rig frame rate). Within each common bin, events from ophys frames falling in that bin are summed.

ii.
```python
valid_mask = raw['valid_roi']
events = raw['events'][valid_mask]
# ...
def resample_events_to_common_bins(events, ophys_ts, trial_start, trial_stop):
    common_ts = np.arange(trial_start, trial_stop, COMMON_DT)
    # ...
    bin_assignments = np.digitize(trial_ophys_ts, bin_edges) - 1
    resampled = np.zeros((n_neurons, n_common), dtype=np.float32)
    for b in range(n_common):
        mask = bin_assignments == b
        if mask.any():
            resampled[:, b] = trial_events[:, mask].sum(axis=1)
    return resampled, common_ts
```

iii. From CONVERSION_NOTES: "Common time bin: 93.23ms (MESO rate) - required because 'time bins should be the same size for all trials and sessions'". The AI justified resampling because CAM2P (~31 Hz) and MESO (~10.7 Hz) rigs have different frame rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Invalid ROIs are excluded using the `valid_roi` flag from the NWB's cell specimen table.

ii.
```python
data['valid_roi'] = f['processing/ophys/image_segmentation/cell_specimen_table/valid_roi'][:]
# ...
valid_mask = raw['valid_roi']
events = raw['events'][valid_mask]
```

iii. From CONVERSION_NOTES: "Neuron curation: Exclude invalid ROIs (valid_roi=False), matching SDK default" and "SDK loads from NWB with `exclude_invalid_rois=True` by default".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start time. The common time bins are generated starting from `trial_start` to `trial_stop` with step `COMMON_DT` (93.23ms). Ophys frames falling within each common bin are summed.

ii.
```python
common_ts = np.arange(trial_start, trial_stop, COMMON_DT)
# ...
bin_edges = np.concatenate([
    [common_ts[0] - COMMON_DT/2],
    (common_ts[:-1] + common_ts[1:]) / 2,
    [common_ts[-1] + COMMON_DT/2]
])
bin_assignments = np.digitize(trial_ophys_ts, bin_edges) - 1
```

iii. The AI uses trial start/stop times from the trials table, with variable-length trials. This aligns to trial start, similar to the reference approach.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI resamples all data to a common time bin of 93.23 ms (~10.73 Hz), matching the MESO rig frame rate. This is a rebinning from the native rates (CAM2P ~31 Hz, MESO ~10.7 Hz) to a uniform rate.

ii.
```python
COMMON_DT = 0.09323  # seconds, ~10.73 Hz

def resample_events_to_common_bins(events, ophys_ts, trial_start, trial_stop):
    common_ts = np.arange(trial_start, trial_stop, COMMON_DT)
```

iii. From CONVERSION_NOTES: "Common time bin: 93.23ms (MESO rate) - required because 'time bins should be the same size for all trials and sessions'". The AI chose the MESO rate as the common bin since it's the coarser of the two rates, avoiding upsampling.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table in the NWB file, specifically the `image_name`, `start_time`, and `stop_time` fields. It uses `np.searchsorted` to find the most recent stimulus at each timepoint.

ii.
```python
stim_keys = [k for k in f['intervals'].keys() if 'Natural_Images' in k]
# ...
for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
    stim_data[key] = stim[key][:]
# ...
def get_image_at_timepoints(common_ts, stim_data, image_to_idx):
    indices = np.searchsorted(start_times, common_ts, side='right') - 1
    for t in range(n_tp):
        idx = indices[t]
        img = image_names[idx]
        if img == 'omitted':
            image_id[t] = last_valid
        elif img in image_to_idx:
            image_id[t] = image_to_idx[img]
```

iii. The AI uses stimulus_presentations rather than the trials table's `initial_image_name`/`change_image_name` to determine what image is on screen at each timepoint.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global mapping built from all unique non-omitted images across all experiments. At each timepoint, the most recent non-omitted stimulus image is assigned. Omitted stimuli inherit the previous image identity.

ii.
```python
all_images = sorted(all_images)
image_to_idx = {img: i for i, img in enumerate(all_image_names)}
# ...
def get_image_at_timepoints(common_ts, stim_data, image_to_idx):
    indices = np.searchsorted(start_times, common_ts, side='right') - 1
    last_valid = 0
    for t in range(n_tp):
        idx = indices[t]
        if idx < 0:
            image_id[t] = last_valid
        elif img == 'omitted':
            image_id[t] = last_valid
        elif img in image_to_idx:
            image_id[t] = image_to_idx[img]
            last_valid = image_to_idx[img]
```

iii. The AI handles omitted stimuli by carrying forward the last valid image. The global sorted mapping ensures consistency across sessions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the same common time bin timestamps as the neural data, using `np.searchsorted` against stimulus presentation start times.

ii.
```python
img_id = get_image_at_timepoints(common_ts, raw['stimulus_presentations'], image_to_idx)
```

iii. Both neural and image identity use the same `common_ts` timestamps, ensuring alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in `stimulus_presentations` combined with stimulus `start_time` and `stop_time`.

ii.
```python
def get_image_change_at_timepoints(common_ts, stim_data):
    is_change = stim_data.get('is_change', np.zeros(len(stim_data['start_time'])))
    start_times = stim_data['start_time']
    stop_times = stim_data['stop_time']
    change_indices = np.where(np.array(is_change) == 1)[0]
    for ci in change_indices:
        mask = (common_ts >= start_times[ci]) & (common_ts < stop_times[ci])
        change[mask] = 1
```

iii. The AI uses the stimulus_presentations' `is_change` flag rather than the trials table's `go` field and `change_time`.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary time series is created. For each stimulus presentation where `is_change == 1`, timepoints between that stimulus's `start_time` and `stop_time` are marked as 1. All other timepoints are 0.

ii. See 4-a code snippet.

iii. The window is defined by the stimulus presentation's start/stop times (250ms stimulus duration) rather than a fixed 750ms window as in the reference.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is already binary (0 or 1), no thresholding needed.

ii. N/A

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is computed at the same common time bin timestamps as the neural data.

ii.
```python
img_change = get_image_change_at_timepoints(common_ts, raw['stimulus_presentations'])
```

iii. Same `common_ts` alignment as neural data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
data['running_speed'] = f['processing/running/speed/data'][:]
data['running_timestamps'] = f['processing/running/speed/timestamps'][:]
```

iii. This is the same underlying running speed data accessed through h5py rather than the SDK.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated to the common time bin timestamps, then discretized into 5 percentile-based bins computed globally across all experiments (using a subsample of 2000 values per experiment for efficiency). NaN values are mapped to bin 2 (middle bin).

ii.
```python
run_interp = interpolate_to_common(
    raw['running_speed'], raw['running_timestamps'], common_ts
)
run_binned = np.digitize(run_interp, running_percentiles[1:-1])
run_binned = np.clip(run_binned, 0, 4).astype(np.int64)
run_binned[np.isnan(run_interp)] = 2
```

iii. From CONVERSION_NOTES: "Percentile binning: Computed across all sessions globally (sampled 2000 values per session)". NaN values are assigned to middle bin rather than bin 0.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal percentile bins using `np.digitize` with globally computed percentile edges. The bin edges are computed from `np.percentile` at [0, 20, 40, 60, 80, 100] percentiles.

ii.
```python
running_percentiles = np.percentile(all_running, np.linspace(0, 100, 6))
run_binned = np.digitize(run_interp, running_percentiles[1:-1])
run_binned = np.clip(run_binned, 0, 4).astype(np.int64)
```

iii. Global percentiles ensure consistent binning across all sessions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same common time bin timestamps as the neural data.

ii.
```python
run_interp = interpolate_to_common(
    raw['running_speed'], raw['running_timestamps'], common_ts
)
```

iii. Same `common_ts` alignment as neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil data is derived from `acquisition/EyeTracking/pupil_tracking/area` in the NWB file. The AI uses **pupil area**, not pupil width or diameter.

ii.
```python
data['pupil_area'] = f['acquisition/EyeTracking/pupil_tracking/area'][:]
data['pupil_timestamps'] = f['acquisition/EyeTracking/pupil_tracking/timestamps'][:]
```

iii. The AI chose pupil area from the NWB file's eye tracking data. This differs from the reference which uses `pupil_width`.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area is linearly interpolated to common timestamps (NaN values in source data are excluded before interpolation), then discretized into 5 percentile-based bins. NaN values after interpolation are mapped to bin 2 (middle bin). No blink removal is performed before interpolation.

ii.
```python
def interpolate_to_common(data, source_ts, common_ts):
    valid = ~np.isnan(data)
    if valid.sum() < 2:
        return np.full(len(common_ts), np.nan)
    f_interp = interpolate.interp1d(
        source_ts[valid], data[valid],
        kind='linear', bounds_error=False, fill_value=np.nan
    )
    return f_interp(common_ts)
# ...
pupil_binned = np.digitize(pupil_interp, pupil_percentiles[1:-1])
pupil_binned = np.clip(pupil_binned, 0, 4).astype(np.int64)
pupil_binned[np.isnan(pupil_interp)] = 2
```

iii. The AI removes NaN values before interpolation but does not explicitly filter out blink frames using the `likely_blink` flag. Missing pupil data experiments are filled with NaN and assigned to the middle bin.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 equal percentile bins using globally computed edges. NaN mapped to bin 2.

ii.
```python
pupil_percentiles = np.percentile(all_pupil, np.linspace(0, 100, 6))
pupil_binned = np.digitize(pupil_interp, pupil_percentiles[1:-1])
pupil_binned = np.clip(pupil_binned, 0, 4).astype(np.int64)
pupil_binned[np.isnan(pupil_interp)] = 2
```

iii. Global percentile edges and middle-bin NaN assignment.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil data is interpolated to the same common time bin timestamps as neural data.

ii.
```python
pupil_interp = interpolate_to_common(
    raw['pupil_area'], raw['pupil_timestamps'], common_ts
)
```

iii. Same `common_ts` alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `correct_reject`, and `false_alarm` in the NWB trials table.

ii.
```python
def get_trial_outcome(trial_data, trial_idx):
    if trial_data['hit'][trial_idx]: return 0
    elif trial_data['miss'][trial_idx]: return 1
    elif trial_data['correct_reject'][trial_idx]: return 2
    elif trial_data['false_alarm'][trial_idx]: return 3
    else: return -1
```

iii. Same four outcome categories as the reference but in a different order: the AI puts `correct_reject` at index 2 and `false_alarm` at index 3, while the reference puts `false_alarm` at 2 and `correct_reject` at 3.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is mapped to integer codes (0-3) and broadcast as a constant across all timepoints in the trial. Trials with no matching outcome (return -1) are skipped.

ii.
```python
outcome = get_trial_outcome(trial_data, trial_idx)
if outcome == -1:
    continue
# ...
trial_output = np.stack([
    img_id, img_change, run_binned, pupil_binned,
    np.full(n_common, outcome, dtype=np.int64)
], axis=0)
```

iii. The static per-trial outcome is replicated across all time bins. The output_values ordering is `['hit', 'miss', 'correct_reject', 'false_alarm']`.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Missing pupil data**: Experiments without eye tracking (`KeyError` on pupil path) get NaN pupil values, assigned to middle bin (bin 2).
- **NaN in running/pupil**: NaN values excluded before interpolation; remaining NaN after interpolation mapped to bin 2.
- **Failed experiments**: Wrapped in try/except, skipped with error message.
- **Short trials**: Trials with fewer than 2 common time bins return None and are skipped.
- **No valid neurons**: Experiments with 0 valid ROIs are skipped.
- **Omitted stimuli**: Image identity carries forward the last valid image.

ii.
```python
try:
    data['pupil_area'] = f['acquisition/EyeTracking/pupil_tracking/area'][:]
    data['has_pupil'] = True
except KeyError:
    data['has_pupil'] = False
    data['pupil_area'] = None
# ...
run_binned[np.isnan(run_interp)] = 2
# ...
if n_neurons == 0:
    return None
```

iii. From CONVERSION_NOTES: "Missing pupil data: Fill with NaN, assign to middle bin (bin 2)" and "Omitted stimuli: use last valid image for identity".

## 9-a. What are the most time-consuming steps of the code?

i. Loading each NWB file via `h5py` is the most time-consuming step, particularly reading the events and running speed arrays. From CONVERSION_NOTES, load time is 0.1-1.8s per experiment.

ii. N/A

iii. From CONVERSION_NOTES: "Load NWB: 0.1-1.8s per session" and "Total per session: ~2-6s".

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `resample_events_to_common_bins` function contains a Python loop over time bins (`for b in range(n_common)`) that could be vectorized using sparse matrix operations or `np.add.at`. The `get_image_at_timepoints` function has a loop over timepoints that could be partially vectorized.

ii.
```python
for b in range(n_common):
    mask = bin_assignments == b
    if mask.any():
        resampled[:, b] = trial_events[:, mask].sum(axis=1)
# ...
for t in range(n_tp):
    idx = indices[t]
    # ...
```

iii. These loops are per-trial and the number of time bins per trial is modest (~80-100), so the overhead is small relative to data loading.

## 9-c. What processing does the code repeat multiple times?

i. The code loads each NWB file twice: once during `collect_global_stats()` to compute percentile bins and image names, and again during `process_experiment()` for the actual conversion. The global stats pass reads running speed, pupil area, and image names from every NWB file.

ii.
```python
def collect_global_stats(expt_table):
    for i, expt_id in enumerate(expt_ids):
        with h5py.File(fname, 'r') as f:
            run_data = f['processing/running/speed/data'][:]
            # ... reads pupil, images
# ... later:
def process_experiment(expt_id, ...):
    raw = load_experiment_data(expt_id)  # reads everything again
```

iii. The two-pass approach is noted in the code design but represents redundant I/O.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The global statistics collection subsamples 2000 values per experiment for percentile computation, which is an approximation. The stimulus presentation data (image_name, is_change, start/stop times) is loaded in full even though only a subset is needed per trial. Additionally, the `valid_roi` mask filtering is applied but the reference solution does not filter ROIs.

ii.
```python
if len(valid_run) > 2000:
    rng = np.random.RandomState(int(expt_id) % (2**31))
    valid_run = rng.choice(valid_run, 2000, replace=False)
```

iii. The subsampling is for efficiency but introduces approximation in percentile edges compared to using all data points.
