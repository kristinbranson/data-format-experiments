# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files using h5py (bypassing the AllenSDK). It reads a metadata CSV (`ophys_experiment_table.csv`) to get the list of experiment IDs, filters to only downloaded NWB files, excludes passive sessions, then iterates over each experiment's NWB file loading neural events, ophys timestamps, valid_roi flags, running speed, pupil area, trial info, and stimulus presentations.

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
    data = {}
    data['events'] = f['processing/ophys/event_detection/data'][:].T
    data['ophys_timestamps'] = f['processing/ophys/dff/traces/timestamps'][:]
    data['valid_roi'] = f['processing/ophys/image_segmentation/cell_specimen_table/valid_roi'][:]
    # ... running speed, pupil, trials, stimulus presentations
    f.close()
    return data
```

iii. The AI justified bypassing the AllenSDK for efficiency: "Efficient NWB loading with h5py (no SDK overhead)." It traced through the SDK code to identify the correct HDF5 paths for each data stream.

## 1-b. How are the data split into subjects (mice)?

i. Each NWB file contains a subject_id. The AI reads `general/subject/subject_id` from each NWB file and builds a unique subjects list, tracking which experiment belongs to which subject via `subject_idx`.

ii.
```python
data['subject_id'] = f['general/subject/subject_id'][()]
# ...
subj = str(result['subject_id'])
if subj not in unique_subjects:
    unique_subjects.append(subj)
subject_idx_list.append(unique_subjects.index(subj))
```

iii. The metadata CSV also has `mouse_id` which is used for counting unique mice at the start, but the actual subject assignment uses the NWB-embedded subject_id. The CONVERSION_NOTES note 38 unique mice.

## 1-c. How are the data split into sessions?

i. Each ophys experiment (each NWB file / imaging plane) is treated as a separate "session" in the output. For MESO rigs with multiple imaging planes per behavioral session, each plane becomes its own session. This means multiple sessions share the same behavioral data (trials, running, pupil) but have different neural populations.

ii.
```python
for i, (_, expt_info) in enumerate(expt_table.iterrows()):
    expt_id = expt_info['ophys_experiment_id']
    result = process_experiment(expt_id, expt_info, ...)
    if result is None:
        continue
    all_neural.append(result['neural'])
    all_output.append(result['output'])
```

iii. From CONVERSION_NOTES: "202 experiments (each imaging plane = 1 session)" and "174 unique ophys sessions". The AI chose this because each experiment has different neurons even if the behavioral session is shared.

## 1-d. How are the data split into trials?

i. The AI reads trial boundaries from `intervals/trials` in the NWB file, using `start_time` and `stop_time` for each trial. Each trial is processed independently with neural events and behavioral variables extracted within the trial's time window.

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

iii. The AI used the NWB trial table directly, which defines trial boundaries via start_time and stop_time fields.

## 1-e. How are trials filtered based on quality controls?

i. The AI includes only Go and Catch trials, excluding Aborted and Auto-rewarded trials. Additionally, trials with fewer than 2 timepoints or with unknown outcome (outcome == -1) are skipped. Experiments with fewer than 2 valid trials are entirely excluded.

ii.
```python
include_mask = trial_data['go'] | trial_data['catch']
include_indices = np.where(include_mask)[0]
# ...
if trial_neural is None:
    continue  # skips trials with < 2 timepoints
# ...
outcome = get_trial_outcome(trial_data, trial_idx)
if outcome == -1:
    continue  # skips trials with unknown outcome
# ...
if len(neural_trials) < 2:
    return None  # skips experiments with too few trials
```

iii. From instructions: "Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials." The AI followed this exactly.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the detected calcium events stored at `processing/ophys/event_detection/data` in the NWB files. These are the raw event detection outputs, not dF/F or filtered events.

ii.
```python
data['events'] = f['processing/ophys/event_detection/data'][:].T  # (n_neurons, n_timepoints)
```

iii. From CONVERSION_NOTES: "Raw events are used for analysis (paper: 'detected calcium events')" and "Events filtered with half-gaussian (scale=2/31s, n_steps=20) for visualization only."

## 2-b. How is the `neural` data processed?

i. The raw events are first filtered to include only valid ROIs (valid_roi == True). Then, for each trial, events are resampled from native ophys timestamps to a common time bin (93.23 ms, the MESO frame rate). Within each common bin, ophys frames are assigned via `np.digitize` on bin edges, and events from all frames in a bin are summed.

ii.
```python
valid_mask = raw['valid_roi']
events = raw['events'][valid_mask]

def resample_events_to_common_bins(events, ophys_ts, trial_start, trial_stop):
    common_ts = np.arange(trial_start, trial_stop, COMMON_DT)
    trial_mask = (ophys_ts >= trial_start - COMMON_DT) & (ophys_ts < trial_stop + COMMON_DT)
    trial_ophys_ts = ophys_ts[trial_mask]
    trial_events = events[:, trial_mask]
    bin_edges = np.concatenate([
        [common_ts[0] - COMMON_DT/2],
        (common_ts[:-1] + common_ts[1:]) / 2,
        [common_ts[-1] + COMMON_DT/2]
    ])
    bin_assignments = np.digitize(trial_ophys_ts, bin_edges) - 1
    resampled = np.zeros((n_neurons, n_common), dtype=np.float32)
    for b in range(n_common):
        mask = bin_assignments == b
        if mask.any():
            resampled[:, b] = trial_events[:, mask].sum(axis=1)
    return resampled, common_ts
```

iii. The AI chose to sum events within bins (rather than average) to preserve total event magnitude. The MESO rate was chosen because "upsampling sparse MESO events would create artifacts."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons using the `valid_roi` flag from the NWB cell specimen table. Only neurons where `valid_roi == True` are included. No additional quality filtering (e.g., based on event statistics, SNR, or other criteria) is applied.

ii.
```python
data['valid_roi'] = f['processing/ophys/image_segmentation/cell_specimen_table/valid_roi'][:]
# ...
valid_mask = raw['valid_roi']
events = raw['events'][valid_mask]
```

iii. From CONVERSION_NOTES: "Neuron curation: Exclude invalid ROIs (valid_roi=False), matching SDK default." The SDK's `CellSpecimens.__init__` uses `exclude_invalid_rois=True` by default.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial start time. For each trial, common time bins are generated starting at `trial_start` with step `COMMON_DT` up to `trial_stop`. The ophys frames falling within these bins are summed. There is no explicit alignment to stimulus change time or other within-trial events.

ii.
```python
common_ts = np.arange(trial_start, trial_stop, COMMON_DT)
```

iii. From metadata: `'temporal_alignment_event': 'Trial start time'` and `'off_start': 0.0, 'off_end': None`. The instructions say "Temporally align based on ophys timestamp" -- the AI interpreted this as using ophys timestamps for alignment, with trials starting at the NWB-defined start_time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses a common time bin of 93.23 ms (~10.73 Hz), matching the MESO imaging rate. Yes, temporal rebinning is applied: CAM2P experiments (native ~30.94 Hz / ~32.32 ms) are downsampled by summing events within 93.23 ms bins, while MESO experiments are kept at approximately native resolution.

ii.
```python
COMMON_DT = 0.09323  # seconds, ~10.73 Hz
```

iii. The AI justified this: "MESO: multiple experiments per session (~10.73 Hz per plane); CAM2P: 1 experiment per session (~30.94 Hz)." The instructions require "Time bins should be the same size for all trials and sessions."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table in the NWB file, specifically the `image_name` and `start_time` fields from the `intervals/<stim_key>/` group (where stim_key matches 'Natural_Images').

ii.
```python
stim_keys = [k for k in f['intervals'].keys() if 'Natural_Images' in k]
# ...
for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
    vals = stim[key][:]
    stim_data[key] = vals
```

iii. The AI identified 16 unique image names across all sessions (image sets A and B combined).

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each timepoint in the common time grid, the most recent stimulus presentation is found using `np.searchsorted`. The image name is mapped to a categorical index (0-15) via a global image-to-index mapping. During omitted stimuli or before any stimulus, the last valid (non-omitted) image identity is carried forward.

ii.
```python
def get_image_at_timepoints(common_ts, stim_data, image_to_idx):
    n_tp = len(common_ts)
    image_id = np.zeros(n_tp, dtype=np.int64)
    start_times = stim_data['start_time']
    image_names = stim_data['image_name']
    indices = np.searchsorted(start_times, common_ts, side='right') - 1
    last_valid = 0
    for t in range(n_tp):
        idx = indices[t]
        if idx < 0:
            image_id[t] = last_valid
            continue
        img = image_names[idx]
        if img == 'omitted':
            image_id[t] = last_valid
        elif img in image_to_idx:
            image_id[t] = image_to_idx[img]
            last_valid = image_to_idx[img]
        else:
            image_id[t] = last_valid
    return image_id
```

iii. The AI noted: "Image identity during gray screen: Assign most recently shown non-omitted image."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the same common timestamps as the neural data (the `common_ts` array generated for each trial). The stimulus presentation `start_time` values are used to determine which image is being shown at each common timepoint via searchsorted.

ii. (See 3-b code above -- `common_ts` is the same array used for neural binning)

iii. Both neural and output data share the same time grid per trial, ensuring alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change`, `start_time`, and `stop_time` fields in the stimulus presentations table.

ii.
```python
def get_image_change_at_timepoints(common_ts, stim_data):
    is_change = stim_data.get('is_change', np.zeros(len(stim_data['start_time'])))
    start_times = stim_data['start_time']
    stop_times = stim_data['stop_time']
    change_indices = np.where(np.array(is_change) == 1)[0]
```

iii. Uses the stimulus presentations' `is_change` flag rather than computing change from sequential image names.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code finds all stimulus presentations where `is_change == 1`, then for each change presentation, marks all common timepoints between the stimulus `start_time` and `stop_time` as 1. All other timepoints are 0.

ii.
```python
def get_image_change_at_timepoints(common_ts, stim_data):
    n_tp = len(common_ts)
    change = np.zeros(n_tp, dtype=np.int64)
    is_change = stim_data.get('is_change', np.zeros(len(stim_data['start_time'])))
    start_times = stim_data['start_time']
    stop_times = stim_data['stop_time']
    change_indices = np.where(np.array(is_change) == 1)[0]
    for ci in change_indices:
        mask = (common_ts >= start_times[ci]) & (common_ts < stop_times[ci])
        change[mask] = 1
    return change
```

iii. The instructions say "Have value of 1 right after a change in image identity, otherwise 0." The AI implements it as 1 during the change stimulus presentation window (start_time to stop_time), which is ~250ms.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1), so no thresholding is needed. It is used directly as a binary categorical variable.

ii. Output values defined as: `['no_change', 'change']`

iii. The instructions specify "binary variable."

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is computed on the same `common_ts` time grid as the neural data, so alignment is automatic. Change events are marked during the stimulus presentation window that falls within each trial.

ii. (See 4-b code -- uses same `common_ts` as neural data)

iii. Temporal alignment is ensured by using the same time grid for all data streams within a trial.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
data['running_speed'] = f['processing/running/speed/data'][:]
data['running_timestamps'] = f['processing/running/speed/timestamps'][:]
```

iii. This matches the standard AllenSDK running speed data source.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is first interpolated from its native timestamps to the common time grid using linear interpolation (scipy interp1d). NaN values in the source data are excluded before interpolation. Then the interpolated values are discretized into 5 bins using globally-computed percentile boundaries.

ii.
```python
run_interp = interpolate_to_common(
    raw['running_speed'], raw['running_timestamps'], common_ts
)
run_binned = np.digitize(run_interp, running_percentiles[1:-1])
run_binned = np.clip(run_binned, 0, 4).astype(np.int64)
run_binned[np.isnan(run_interp)] = 2  # NaN -> middle bin
```

iii. Global percentiles are computed by sampling up to 2000 values per experiment across all experiments, then computing 6 percentile boundaries (0%, 20%, 40%, 60%, 80%, 100%).

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal percentile bins (quintiles) using `np.digitize` with globally-computed percentile boundaries. Bins are labeled 0-4. NaN values are assigned to the middle bin (bin 2).

ii.
```python
running_percentiles = np.percentile(all_running, np.linspace(0, 100, 6))
# Results: [-22.15, -0.004, 0.337, 14.12, 33.15, 97.87]
run_binned = np.digitize(run_interp, running_percentiles[1:-1])
run_binned = np.clip(run_binned, 0, 4).astype(np.int64)
```

iii. The instructions specify "discretized into five equal percentile bins." The AI computes these globally rather than per-session.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same `common_ts` time grid as neural data using linear interpolation, ensuring temporal alignment.

ii.
```python
def interpolate_to_common(data, source_ts, common_ts):
    f_interp = interpolate.interp1d(
        source_ts[valid], data[valid],
        kind='linear', bounds_error=False, fill_value=np.nan
    )
    return f_interp(common_ts)
```

iii. Same common time grid is used for all data streams.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/area` in the NWB file (pupil area, not diameter).

ii.
```python
data['pupil_area'] = f['acquisition/EyeTracking/pupil_tracking/area'][:]
data['pupil_timestamps'] = f['acquisition/EyeTracking/pupil_tracking/timestamps'][:]
```

iii. The AI uses pupil area rather than computing actual diameter. The NWB file has `area`, `area_raw`, `height`, `width` fields for pupil tracking. The variable name in the output is "pupil_diameter" but the underlying data is pupil area.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area is interpolated to common timestamps (same as running speed), then discretized into 5 equal percentile bins using globally-computed boundaries. For experiments without eye tracking data, all timepoints are assigned to the middle bin (bin 2).

ii.
```python
if raw['has_pupil']:
    pupil_interp = interpolate_to_common(
        raw['pupil_area'], raw['pupil_timestamps'], common_ts
    )
else:
    pupil_interp = np.full(n_common, np.nan)

if pupil_percentiles is not None:
    pupil_binned = np.digitize(pupil_interp, pupil_percentiles[1:-1])
    pupil_binned = np.clip(pupil_binned, 0, 4).astype(np.int64)
    pupil_binned[np.isnan(pupil_interp)] = 2
else:
    pupil_binned = np.full(n_common, 2, dtype=np.int64)
```

iii. Missing pupil data is handled by assigning NaN values to the middle bin (bin 2).

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 equal percentile bins computed globally across all sessions with valid pupil data. The percentile boundaries were [228.6, 4303.4, 5455.9, 6710.0, 8652.3, 107624.5].

ii.
```python
pupil_percentiles = np.percentile(all_pupil, np.linspace(0, 100, 6))
pupil_binned = np.digitize(pupil_interp, pupil_percentiles[1:-1])
```

iii. The instructions specify "discretized into five equal percentile bins."

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed: interpolated to common timestamps using scipy interp1d.

ii. (See 6-b code above)

iii. Same common time grid ensures alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial columns: `hit`, `miss`, `correct_reject`, and `false_alarm` from the NWB `intervals/trials` table.

ii.
```python
for key in ['go', 'catch', 'aborted', 'auto_rewarded', 'hit', 'miss',
            'correct_reject', 'false_alarm', 'start_time', 'stop_time',
            'change_time', 'change_image_name', 'initial_image_name', 'is_change']:
    vals = trials[key][:]
    trial_data[key] = vals
```

iii. These are standard trial outcome categories in the Visual Behavior task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is determined by checking which of the four boolean flags (hit, miss, correct_reject, false_alarm) is True for each trial. The outcome is encoded as: 0=hit, 1=miss, 2=correct_reject, 3=false_alarm. It is a static per-trial variable, replicated across all timepoints in the trial.

ii.
```python
def get_trial_outcome(trial_data, trial_idx):
    if trial_data['hit'][trial_idx]: return 0
    elif trial_data['miss'][trial_idx]: return 1
    elif trial_data['correct_reject'][trial_idx]: return 2
    elif trial_data['false_alarm'][trial_idx]: return 3
    else: return -1

# Static per trial:
trial_output = np.stack([
    img_id, img_change, run_binned, pupil_binned,
    np.full(n_common, outcome, dtype=np.int64)
], axis=0)
```

iii. Trials with no matching outcome (return -1) are excluded. The outcome distribution was: 30.2% hit, 57.2% miss, 10.8% correct_reject, 1.7% false_alarm.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several types of missing data are handled:
- **Missing pupil data**: Experiments without eye tracking have pupil set to NaN, then assigned to middle bin (bin 2).
- **NaN in running speed**: NaN values excluded during interpolation, then NaN interpolated values assigned to middle bin (bin 2).
- **Omitted stimuli**: Image identity carries forward the last valid image.
- **Missing stimulus presentations**: Experiments without stimulus presentations are skipped entirely.
- **Experiments with 0 valid neurons**: Skipped.
- **Trials with < 2 timepoints**: Skipped.
- **Unknown trial outcomes**: Trials with no matching outcome flag (outcome == -1) are skipped.

ii.
```python
# Missing pupil
if raw['has_pupil']:
    pupil_interp = interpolate_to_common(...)
else:
    pupil_interp = np.full(n_common, np.nan)
pupil_binned[np.isnan(pupil_interp)] = 2

# NaN running
run_binned[np.isnan(run_interp)] = 2

# Omitted images
if img == 'omitted':
    image_id[t] = last_valid
```

iii. The AI documented these in CONVERSION_NOTES Step 10 Check 5 (edge cases).

## 9-a. What are the most time-consuming steps of the code?

i. Based on the timing information in the output, the most time-consuming steps are:
1. **Global statistics collection**: Scanning all 202 NWB files to compute running/pupil percentiles and collect image names (iterates through all files before processing).
2. **NWB file loading**: Each file load takes 0.1-1.8 seconds.
3. **Per-trial event resampling**: The inner loop over time bins in `resample_events_to_common_bins` iterates over each common bin.

ii.
```python
# Global stats - reads every NWB file
def collect_global_stats(expt_table):
    for i, expt_id in enumerate(expt_ids):
        with h5py.File(fname, 'r') as f:
            run_data = f['processing/running/speed/data'][:]
            # ... pupil, images
```

iii. Total conversion time was ~518 seconds (8.6 min) for 202 experiments.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop that could be vectorized is in `resample_events_to_common_bins` where it loops over each common time bin to sum events:

ii.
```python
for b in range(n_common):
    mask = bin_assignments == b
    if mask.any():
        resampled[:, b] = trial_events[:, mask].sum(axis=1)
```
This could be replaced with `np.add.at` or sparse matrix operations.

Also, `get_image_at_timepoints` loops over every timepoint:
```python
for t in range(n_tp):
    idx = indices[t]
    # ...
```

iii. The AI noted "Vectorized event binning using np.digitize" in Step 6, but the inner summation loop is still sequential.

## 9-c. What processing does the code repeat multiple times?

i. The code reads each NWB file twice: once during `collect_global_stats` (to compute running/pupil percentiles and image names) and once during `process_experiment` (to extract all data for conversion). This means every NWB file's running speed, pupil area, and stimulus data are loaded twice.

ii.
```python
# First pass - collect_global_stats
for i, expt_id in enumerate(expt_ids):
    with h5py.File(fname, 'r') as f:
        run_data = f['processing/running/speed/data'][:]
        pupil_data = f['acquisition/EyeTracking/pupil_tracking/area'][:]
        imgs = f[f'intervals/{stim_keys[0]}/image_name'][:]

# Second pass - process_experiment (called per experiment)
raw = load_experiment_data(expt_id)  # loads same data again
```

iii. The AI didn't explicitly document this redundancy. The global statistics pass is needed to compute percentiles before binning, but caching intermediate data could avoid the duplicate I/O.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several processing steps produce data not used downstream:
1. The `input` field is set to empty arrays (shape `(0, n_timepoints)`) for every trial, since no decoder inputs are specified. This is technically unnecessary data.
2. The `cell_specimen_ids` are loaded from each NWB file but never used in the output.
3. The `change_time`, `change_image_name`, `initial_image_name`, and `is_change` fields are loaded from the trials table but only `is_change` is indirectly used (via stimulus presentations).

ii.
```python
# Empty inputs
all_input.append([np.zeros((0, t.shape[1]), dtype=np.float32) for t in result['neural']])

# Unused cell IDs
data['cell_specimen_ids'] = f['processing/ophys/image_segmentation/cell_specimen_table/cell_specimen_id'][:]

# Loaded but mostly unused trial fields
for key in ['go', 'catch', 'aborted', 'auto_rewarded', 'hit', 'miss',
            'correct_reject', 'false_alarm', 'start_time', 'stop_time',
            'change_time', 'change_image_name', 'initial_image_name', 'is_change']:
```

iii. The empty input arrays are created to match the expected data format structure. Loading extra trial fields is minor overhead.
