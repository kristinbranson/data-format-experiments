# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files using `h5py`, bypassing the AllenSDK. It reads a metadata CSV (`ophys_experiment_table.csv`) to discover experiments, filters to only downloaded NWB files, and excludes passive sessions. Each experiment's data (events, running speed, pupil, trials, stimulus presentations) is loaded from the NWB file structure.

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
    return expt_table

def load_experiment_data(expt_id):
    fname = os.path.join(EXPT_DIR, f'behavior_ophys_experiment_{expt_id}.nwb')
    f = h5py.File(fname, 'r')
    data['events'] = f['processing/ophys/event_detection/data'][:].T
    # ...
```

iii. The AI chose to use `h5py` directly instead of the AllenSDK to avoid SDK overhead and for faster loading. The experiment table CSV provides metadata for filtering.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `subject_id` read from each NWB file (`general/subject/subject_id`), and unique subjects are collected into a list as experiments are processed.

ii.
```python
data['subject_id'] = f['general/subject/subject_id'][()]
# ...
subj = str(result['subject_id'])
if subj not in unique_subjects:
    unique_subjects.append(subj)
subject_idx_list.append(unique_subjects.index(subj))
```

iii. Subject IDs are extracted from each NWB file and deduplicated. The ordering depends on processing order.

## 1-c. How are the data split into sessions?

i. Each experiment (NWB file / imaging plane) is treated as a separate "session" in the output. For multi-plane (MESO) imaging sessions that share the same `ophys_session_id`, each plane becomes its own session rather than being merged.

ii.
```python
for i, (_, expt_info) in enumerate(expt_table.iterrows()):
    expt_id = expt_info['ophys_experiment_id']
    result = process_experiment(expt_id, ...)
    all_neural.append(result['neural'])
```

iii. The AI recognized that MESO sessions have multiple experiments sharing behavioral data but with different neurons. It chose to treat each experiment as a separate session for simplicity, noting in CONVERSION_NOTES that "each experiment has different neurons but same trials/behavior."

## 1-d. How are the data split into trials?

i. Trials are defined using the `intervals/trials` data from the NWB file. Go and Catch trials are included; Aborted and Auto-rewarded trials are excluded. The trial window spans from `start_time` to `stop_time`. Neural data is resampled to common time bins within each trial window.

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

iii. Trial filtering uses the `go` and `catch` boolean flags directly, as specified in the instructions. The full trial window (start_time to stop_time) is used.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by requiring `go` or `catch` to be True (excluding aborted and auto-rewarded). Trials where resampled neural data is None (< 2 time bins) are skipped. Trials with an unrecognized outcome (not hit/miss/CR/FA) are skipped. Sessions with fewer than 2 valid trials are excluded.

ii.
```python
include_mask = trial_data['go'] | trial_data['catch']
include_indices = np.where(include_mask)[0]
# ...
if trial_neural is None:
    continue
# ...
outcome = get_trial_outcome(trial_data, trial_idx)
if outcome == -1:
    continue
# ...
if len(neural_trials) < 2:
    return None
```

iii. The filtering matches the instruction to include Go and Catch trials but exclude Aborted and Auto-rewarded. The minimum 2-trial threshold prevents degenerate sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `event_detection/data` (detected calcium events) from the NWB file, NOT from `dff_traces` (dF/F fluorescence traces).

ii.
```python
data['events'] = f['processing/ophys/event_detection/data'][:].T  # (n_neurons, n_timepoints)
```

iii. The AI chose events based on the paper's methods section stating "detected calcium events" and the SDK code analysis showing events are the analysis-ready signal. From CONVERSION_NOTES: "Neural data: detected calcium events (not raw dff)" and "Paper uses raw events."

## 2-b. How is the `neural` data processed?

i. Events are filtered by `valid_roi` to exclude invalid ROIs. Then for each trial, events are resampled from their native ophys timestamps to common time bins (93.23ms, matching MESO frame rate). Within each common bin, event values from ophys frames falling in that bin are summed.

ii.
```python
valid_mask = raw['valid_roi']
events = raw['events'][valid_mask]
# ...
def resample_events_to_common_bins(events, ophys_ts, trial_start, trial_stop):
    common_ts = np.arange(trial_start, trial_stop, COMMON_DT)
    bin_edges = np.concatenate([...])
    bin_assignments = np.digitize(trial_ophys_ts, bin_edges) - 1
    resampled = np.zeros((n_neurons, n_common), dtype=np.float32)
    for b in range(n_common):
        mask = bin_assignments == b
        if mask.any():
            resampled[:, b] = trial_events[:, mask].sum(axis=1)
    return resampled, common_ts
```

iii. The AI applied valid_roi filtering to match the SDK's default behavior (`exclude_invalid_rois=True`). Resampling to a common time bin was deemed necessary because CAM2P (~31 Hz) and MESO (~10.7 Hz) experiments have different frame rates, and the instructions require "time bins should be the same size for all trials and sessions."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered using the `valid_roi` flag from the NWB cell specimen table. Only neurons with `valid_roi == True` are included. Experiments with 0 valid neurons are skipped entirely.

ii.
```python
data['valid_roi'] = f['processing/ophys/image_segmentation/cell_specimen_table/valid_roi'][:]
# ...
valid_mask = raw['valid_roi']
events = raw['events'][valid_mask]
n_neurons = events.shape[0]
if n_neurons == 0:
    return None
```

iii. The `valid_roi` flag comes from the Allen pipeline's segmentation quality control. The AI noted this matches the SDK default behavior.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to `trial_start` (the trial's `start_time`). Common time bins are generated starting from `trial_start` through `trial_stop` at 93.23ms intervals. Ophys frames are assigned to the nearest common bin and summed.

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

iii. The bin edges are centered on common timestamps, with half-bin margins. The margin `trial_mask = (ophys_ts >= trial_start - COMMON_DT) & (ophys_ts < trial_stop + COMMON_DT)` allows capturing ophys frames just outside the trial boundaries, which can cause small event leakage between trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI resamples all data to a common time bin of 93.23ms (~10.73 Hz), matching the MESO imaging rate. This is applied uniformly to both CAM2P (~31 Hz) and MESO experiments. For CAM2P experiments, this means downsampling by ~3x. Events within each bin are summed.

ii.
```python
COMMON_DT = 0.09323  # seconds, ~10.73 Hz
# ...
common_ts = np.arange(trial_start, trial_stop, COMMON_DT)
```

iii. From CONVERSION_NOTES: "Common time bin: 93.23ms (MESO rate) - required because 'time bins should be the same size for all trials and sessions'." The AI chose the slower MESO rate as the common rate since upsampling sparse events would create artifacts.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table in the NWB file, specifically the `image_name`, `start_time`, and `stop_time` fields from the interval group containing "Natural_Images".

ii.
```python
stim_keys = [k for k in f['intervals'].keys() if 'Natural_Images' in k]
# ...
stim = f[f'intervals/{stim_key}']
for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
    stim_data[key] = stim[key][:]
```

iii. The AI used the stimulus_presentations table rather than the trials table fields (`initial_image_name`/`change_image_name`), reasoning that stimulus presentations provide the exact timing of each image presentation within a trial.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each time bin, the most recent non-omitted stimulus presentation is found using `np.searchsorted`. Image names are mapped to integer indices via a global sorted mapping. Omitted stimuli are handled by carrying forward the last valid image identity.

ii.
```python
def get_image_at_timepoints(common_ts, stim_data, image_to_idx):
    indices = np.searchsorted(start_times, common_ts, side='right') - 1
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

iii. The approach handles omitted stimuli (5% of presentations) by carrying forward the last valid image. The global mapping ensures consistent codes across all sessions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the same common timestamps (`common_ts`) as the resampled neural data, ensuring frame-by-frame alignment.

ii.
```python
img_id = get_image_at_timepoints(common_ts, raw['stimulus_presentations'], image_to_idx)
```

iii. Both neural data and image identity use the same `common_ts` array generated per trial.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field and `start_time`/`stop_time` of the stimulus_presentations table. It is 1 during any stimulus presentation marked as a change, and 0 otherwise.

ii.
```python
def get_image_change_at_timepoints(common_ts, stim_data):
    is_change = stim_data.get('is_change', np.zeros(len(stim_data['start_time'])))
    change_indices = np.where(np.array(is_change) == 1)[0]
    for ci in change_indices:
        mask = (common_ts >= start_times[ci]) & (common_ts < stop_times[ci])
        change[mask] = 1
    return change
```

iii. The AI uses the stimulus_presentations `is_change` flag, which marks both go and catch trial changes. The change window is the stimulus presentation duration (250ms), not a fixed 750ms window.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Binary indicator: 1 during the time window of a stimulus presentation marked as `is_change`, 0 otherwise. The window is defined by the stimulus `start_time` and `stop_time` (250ms presentation duration).

ii. See 4-a code snippet.

iii. No additional processing beyond the binary assignment.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1). No thresholding is needed.

ii. N/A

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity - computed at the same `common_ts` timestamps as neural data.

ii.
```python
img_change = get_image_change_at_timepoints(common_ts, raw['stimulus_presentations'])
```

iii. Alignment is ensured by using the same timestamp array.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
data['running_speed'] = f['processing/running/speed/data'][:]
data['running_timestamps'] = f['processing/running/speed/timestamps'][:]
```

iii. These are the standard running speed data provided in the NWB files.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the common time bins, then discretized into 5 percentile-based bins. Percentile edges are computed globally across all experiments (subsampled to 2000 values per experiment). NaN values are mapped to bin 2 (middle bin).

ii.
```python
run_interp = interpolate_to_common(raw['running_speed'], raw['running_timestamps'], common_ts)
run_binned = np.digitize(run_interp, running_percentiles[1:-1])
run_binned = np.clip(run_binned, 0, 4).astype(np.int64)
run_binned[np.isnan(run_interp)] = 2
```

iii. Global percentile-based binning ensures consistent categories across sessions. The subsampling to 2000 points per experiment was done for efficiency. NaN mapping to bin 2 (middle) was chosen as a neutral default.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using percentile edges (0th, 20th, 40th, 60th, 80th, 100th percentiles) computed globally across all experiments. `np.digitize` with the inner edges (20th through 80th) assigns each value to a bin (0-4).

ii.
```python
running_percentiles = np.percentile(all_running, np.linspace(0, 100, 6))
# ...
run_binned = np.digitize(run_interp, running_percentiles[1:-1])
run_binned = np.clip(run_binned, 0, 4).astype(np.int64)
```

iii. Equal-percentile binning ensures roughly equal class counts.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same `common_ts` timestamps used for neural data resampling.

ii.
```python
run_interp = interpolate_to_common(raw['running_speed'], raw['running_timestamps'], common_ts)
```

iii. Alignment is guaranteed by using the same timestamp array.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/area` (pupil area, not width/diameter) and its associated timestamps.

ii.
```python
data['pupil_area'] = f['acquisition/EyeTracking/pupil_tracking/area'][:]
data['pupil_timestamps'] = f['acquisition/EyeTracking/pupil_tracking/timestamps'][:]
```

iii. The AI chose pupil area rather than pupil width. Some experiments lack eye tracking data entirely, which is handled by filling with NaN.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area is linearly interpolated to common timestamps (with NaN removal before interpolation), then discretized into 5 percentile-based bins. NaN values (from missing data or interpolation boundaries) are mapped to bin 2 (middle). Blink detection is not explicitly handled - NaN values in the raw data are removed before interpolation.

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
pupil_binned[np.isnan(pupil_interp)] = 2
```

iii. The AI removes NaN values before interpolation, which implicitly handles some blink artifacts (blink frames may have NaN area). However, it does not use the `likely_blink` flag explicitly. Missing pupil data experiments are filled with NaN and assigned to bin 2.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 percentile-based bins computed globally, applied with `np.digitize`. NaN mapped to bin 2.

ii.
```python
pupil_percentiles = np.percentile(all_pupil, np.linspace(0, 100, 6))
pupil_binned = np.digitize(pupil_interp, pupil_percentiles[1:-1])
pupil_binned = np.clip(pupil_binned, 0, 4).astype(np.int64)
pupil_binned[np.isnan(pupil_interp)] = 2
```

iii. Global percentile binning for consistency. Subsampled 2000 values per experiment for efficiency.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed - interpolated to `common_ts`.

ii.
```python
pupil_interp = interpolate_to_common(raw['pupil_area'], raw['pupil_timestamps'], common_ts)
```

iii. Alignment through shared timestamps.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `correct_reject`, and `false_alarm` in the trials interval group of the NWB file.

ii.
```python
def get_trial_outcome(trial_data, trial_idx):
    if trial_data['hit'][trial_idx]: return 0
    elif trial_data['miss'][trial_idx]: return 1
    elif trial_data['correct_reject'][trial_idx]: return 2
    elif trial_data['false_alarm'][trial_idx]: return 3
    else: return -1
```

iii. These four outcomes cover all valid non-aborted, non-auto-rewarded trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0=hit, 1=miss, 2=correct_reject, 3=false_alarm). The code is replicated across all time bins in the trial as a static per-trial variable. Trials with outcome -1 (none of the four categories) are skipped.

ii.
```python
outcome = get_trial_outcome(trial_data, trial_idx)
if outcome == -1:
    continue
# ...
np.full(n_common, outcome, dtype=np.int64)
```

iii. The outcome ordering is `['hit', 'miss', 'correct_reject', 'false_alarm']` (note: correct_reject and false_alarm are swapped compared to the reference solution).

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing pupil data**: Experiments without eye tracking have pupil area filled with NaN, discretized to bin 2 (middle).
- **NaN in continuous signals**: NaN values are removed before interpolation; NaN in interpolated values maps to bin 2 for running/pupil.
- **Failed experiments**: Exceptions during processing are caught and the experiment is skipped.
- **Too few trials/neurons**: Experiments with 0 valid neurons or < 2 valid trials are skipped.
- **Omitted stimuli**: Image identity carries forward the last valid image.
- **Short trials**: Trials with < 2 time bins after resampling return None and are skipped.

ii.
```python
try:
    result = process_experiment(...)
except Exception as e:
    print(f'  ERROR: {e}')
    continue
# ...
if n_neurons == 0:
    return None
# ...
run_binned[np.isnan(run_interp)] = 2
pupil_binned[np.isnan(pupil_interp)] = 2
```

iii. The AI documented these edge cases in CONVERSION_NOTES Step 10, Check 5.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each NWB file via `h5py`, reading the full events array, running speed, pupil data, trials, and stimulus presentations. The resampling step is also significant, especially for CAM2P experiments with high frame rates.

ii. N/A

iii. From CONVERSION_NOTES: "Load NWB: 0.1-1.8s per session, Resample + process: 0.5-5s per session."

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The resampling function `resample_events_to_common_bins` contains a loop over common time bins that sums events:
```python
for b in range(n_common):
    mask = bin_assignments == b
    if mask.any():
        resampled[:, b] = trial_events[:, mask].sum(axis=1)
```
This could be vectorized using sparse matrices or `np.add.at`.

The `get_image_at_timepoints` function loops over each timepoint:
```python
for t in range(n_tp):
    # determine image at each timepoint
```
This could be vectorized using interval-based assignment.

ii. See code snippets above.

iii. The AI noted in trajectory that the loop-based resampling could be slow but chose to keep it for clarity since data loading dominates runtime.

## 9-c. What processing does the code repeat multiple times?

i. The code loads each NWB file twice: once in `collect_global_stats` to compute global percentile bins and image names, and once in `process_experiment` for the actual conversion. The global stats pass reads running speed and pupil data from every experiment just to compute percentiles.

ii.
```python
def collect_global_stats(expt_table):
    for i, expt_id in enumerate(expt_ids):
        with h5py.File(fname, 'r') as f:
            run_data = f['processing/running/speed/data'][:]
            pupil_data = f['acquisition/EyeTracking/pupil_tracking/area'][:]
            # ...

# Then later:
result = process_experiment(expt_id, ...)  # loads same NWB file again
```

iii. The two-pass approach was chosen for simplicity: first compute global stats, then process. This doubles the I/O but avoids needing to store all raw data in memory.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The stimulus presentations data (`intervals/Natural_Images_*`) is loaded in full for every experiment, including omitted stimuli information, but only the image identity and change flag are used. The full stimulus table with start/stop times for every presentation is loaded even though the trial-level alignment could be done more simply using the trials table fields (`initial_image_name`, `change_image_name`, `change_time`).

ii.
```python
stim_keys = [k for k in f['intervals'].keys() if 'Natural_Images' in k]
stim = f[f'intervals/{stim_key}']
for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
    stim_data[key] = stim[key][:]
```

iii. The reference solution uses the simpler approach of using `initial_image_name` and `change_image_name` from the trials table with `change_time` as the switch point, avoiding the need to load and process stimulus presentations entirely.
