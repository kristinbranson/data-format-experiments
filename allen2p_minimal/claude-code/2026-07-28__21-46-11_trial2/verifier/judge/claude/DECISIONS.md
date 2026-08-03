# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files using `h5py`, rather than using the Allen SDK's `VisualBehaviorOphysProjectCache`. It reads experiment metadata from a CSV table (`ophys_experiment_table.csv`), filters for available NWB files, and filters for active behavior sessions (`passive == False`). It does NOT filter by `project_code == 'VisualBehavior'`, so it includes both VisualBehavior (single-plane, ~31 Hz) and VisualBehaviorMultiscope (multi-plane, ~11 Hz) experiments.

ii.
```python
def load_experiment_table():
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    nwb_files = [f for f in os.listdir(NWB_DIR) if f.endswith('.nwb')]
    nwb_ids = [int(f.split('_')[-1].split('.')[0]) for f in nwb_files]
    exp_table = exp_table[exp_table['ophys_experiment_id'].isin(nwb_ids)]
    exp_table = exp_table[exp_table['passive'] == False]
    return exp_table
```

```python
def load_nwb_data(nwb_path):
    with h5py.File(nwb_path, 'r') as f:
        ophys_ts = f['processing/ophys/dff/traces/timestamps'][:]
        events = f['processing/ophys/event_detection/data'][:]
        dff = f['processing/ophys/dff/traces/data'][:]
        # ... loads trials, running, pupil, stimulus presentations
    return data
```

iii. From CONVERSION_NOTES.md: "Active behavior only: Filtered passive == False from experiment table (202 of 284 NWB files). All project codes included: VisualBehavior (single-plane, ~31 Hz) and VisualBehaviorMultiscope (multi-plane, ~11 Hz)." The AI chose to include all project codes rather than filtering to VisualBehavior only.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values in the experiment table, sorted and converted to strings.

ii.
```python
all_subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
```

iii. The mouse_id field is the standard identifier for each animal in the Allen dataset.

## 1-c. How are the data split into sessions?

i. Each individual experiment (imaging plane) is treated as a separate session. The AI does NOT group multiple experiments from the same `ophys_session_id` together. This means multi-plane sessions are split into separate "sessions" in the output, each with neurons from only one imaging plane.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    # ... each experiment becomes one session
    neural_all.append(final_neural)
    subject_idx_list.append(subject_to_idx[mouse_id])
```

iii. From CONVERSION_NOTES.md: "Each experiment = 1 session in the output." The AI treats each experiment (one imaging plane) as its own session rather than grouping planes from the same ophys session.

## 1-d. How are the data split into trials?

i. Trials are defined using the NWB `intervals/trials` table. Each trial spans from `start_time` to `stop_time`. Ophys frames in the interval `[start_time, stop_time)` are extracted (note: exclusive upper bound via `<`). Trials with fewer than 2 ophys frames are excluded.

ii.
```python
for trial_idx in range(n_trials_total):
    if not get_trial_mask(trial_idx, nwb_data):
        continue
    t_start = nwb_data['trial_start_times'][trial_idx]
    t_stop = nwb_data['trial_stop_times'][trial_idx]
    frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
    frame_indices = np.where(frame_mask)[0]
    if len(frame_indices) < 2:
        continue
```

iii. From CONVERSION_NOTES.md: "Trials segmented using start_time and stop_time from NWB trials table. Ophys frames within [start_time, stop_time) extracted for each trial."

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded if: (1) aborted, (2) auto-rewarded, (3) not a go or catch trial, (4) trial outcome is unknown (-1), (5) fewer than 2 ophys frames, or (6) no valid stimulus during the trial (all image indices remain -1). Sessions with fewer than 2 valid trials are skipped.

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
```
```python
outcome = get_trial_outcome(trial_idx, nwb_data)
if outcome == -1:
    continue
```
```python
if len(frame_indices) < 2:
    continue
```

iii. From CONVERSION_NOTES.md: "Included: Go trials and Catch trials. Excluded: Aborted trials (mouse licked before change) and Auto-rewarded trials (free rewards). This matches the task specification."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses **detected calcium events** (`processing/ophys/event_detection/data`) from the NWB files, NOT the dF/F traces.

ii.
```python
events = f['processing/ophys/event_detection/data'][:]
events_ts = f['processing/ophys/event_detection/timestamps'][:]
data['events'] = events  # shape: (n_timepoints, n_cells)
```
```python
# Use calcium events as neural data
events = nwb_data['events']  # (n_timepoints, n_cells)
```

iii. From CONVERSION_NOTES.md: "Used detected calcium events rather than raw dF/F traces. This matches the paper: 'We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces'."

## 2-b. How is the `neural` data processed?

i. The events data is filtered to keep only valid ROIs (using `valid_roi` flag from the cell specimen table). The data is transposed from (n_timepoints, n_cells) to (n_cells, n_timepoints) format. No other processing is applied.

ii.
```python
valid = nwb_data['valid_roi']
events = events[:, valid]
# ...
trial_neural = events[frame_indices, :].T  # (n_neurons, n_timepoints)
```

iii. From CONVERSION_NOTES.md: "Only valid ROIs included (filtered by valid_roi flag in cell specimen table)."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered using the `valid_roi` flag from the NWB cell specimen table. Only neurons marked as valid ROIs are included. Experiments with zero valid neurons are skipped entirely.

ii.
```python
data['valid_roi'] = cst['valid_roi'][:]
# ...
valid = nwb_data['valid_roi']
events = events[:, valid]
# ...
if n_neurons == 0:
    return None, None, None, None, None, None
```

iii. The AI chose to filter by `valid_roi` to exclude poorly segmented cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the ophys timestamps. For each trial, frames within `[start_time, stop_time)` are selected, giving a variable-length window aligned to trial start.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
frame_indices = np.where(frame_mask)[0]
trial_neural = events[frame_indices, :].T
```

iii. The alignment is to the trial start time, with variable-length trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The data is kept at native ophys frame rate. The time bin size is computed as `1000 / median_imaging_rate` across sessions. Because both VisualBehavior (~31 Hz) and VisualBehaviorMultiscope (~11 Hz) experiments are included, the reported time bin size depends on which project code has more sessions. The metadata reports 32.26 ms, corresponding to ~31 Hz single-plane imaging.

ii.
```python
imaging_rates = [s['imaging_rate_hz'] for s in session_info]
median_rate = np.median(imaging_rates)
time_bin_ms = 1000.0 / median_rate
```

iii. From CONVERSION_NOTES.md: "Time bin size is the inter-frame interval (~32.26 ms for single-plane at ~31 Hz, ~93.2 ms for multi-plane at ~11 Hz). Metadata time_bin_size reports median across sessions (32.26 ms)."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the **stimulus presentations table** in the NWB file (`intervals/{stim_key}`), using `start_time`, `stop_time`, and `image_name` fields. This is different from the reference which uses the trial table's `initial_image_name` and `change_image_name`.

ii.
```python
stim_starts = nwb_data['stim_start_times']
stim_stops = nwb_data['stim_stop_times']
stim_names = nwb_data['stim_image_names']
stim_omitted = nwb_data['stim_omitted']
```

iii. From CONVERSION_NOTES.md: "For each ophys timepoint, the currently displayed image is determined from stimulus presentation intervals."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each ophys timepoint, the code finds which stimulus presentation interval it falls within and assigns the corresponding image index. During grey screen periods (inter-stimulus intervals), the last shown image identity is carried forward. Omitted stimuli are skipped (treated as grey screen). Image names are mapped to sorted integer indices.

ii.
```python
for i in range(len(stim_starts)):
    if stim_omitted[i] == 1.0:
        continue
    img_name = stim_names[i]
    if img_name == 'omitted':
        continue
    img_idx = image_name_to_idx[img_name]
    mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
    image_at_ophys[mask] = img_idx

# Forward-fill grey screen periods
last_img = -1
for k in range(len(trial_image)):
    if trial_image[k] >= 0:
        last_img = trial_image[k]
    elif last_img >= 0:
        trial_image[k] = last_img
```

iii. From CONVERSION_NOTES.md: "During gray screen periods, the last shown image identity is carried forward. Omitted stimuli are treated as gray screen."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at each ophys timepoint by checking which stimulus presentation interval each frame falls within. The same ophys frame indices used for neural data are used for image identity.

ii.
```python
image_at_ophys = np.full(len(ophys_ts), -1, dtype=int)
# ... fill from stimulus presentations ...
trial_image = image_at_ophys[frame_indices]
```

iii. Alignment is inherent because image identity is mapped to ophys timestamps.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in the **stimulus presentations table**, NOT from the trial table's `change_time` and `go` columns.

ii.
```python
data['stim_is_change'] = stim['is_change'][:]
```

iii. From CONVERSION_NOTES.md: "Binary signal (0/1) marking timepoints during which a stimulus change occurred. Derived from is_change field in stimulus presentations table."

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each stimulus presentation where `is_change == 1.0`, all ophys frames within that presentation's `[start_time, stop_time)` are marked as 1. All other frames are 0.

ii.
```python
is_change_at_ophys = np.zeros(len(ophys_ts), dtype=int)
stim_is_change = nwb_data['stim_is_change']
for i in range(len(stim_starts)):
    if stim_is_change[i] == 1.0:
        mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
        is_change_at_ophys[mask] = 1
```

iii. The change indicator spans only the stimulus presentation duration (~250ms), not a 750ms window as in the reference.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1) based on whether a stimulus presentation has `is_change == 1`. No thresholding is needed.

ii. See 4-b.

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity -- computed at each ophys timepoint and extracted using the same frame indices.

ii.
```python
trial_change = is_change_at_ophys[frame_indices]
```

iii. Alignment is inherent because the change indicator is mapped to ophys timestamps.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
data['running_speed'] = f['processing/running/speed/data'][:]
data['running_timestamps'] = f['processing/running/speed/timestamps'][:]
```

iii. This is the standard running speed signal from the Allen SDK.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated to ophys timestamps using `np.interp`, then discretized into 5 equal percentile bins. Bin edges are computed globally across all sessions.

ii.
```python
def interpolate_to_ophys(signal, signal_ts, ophys_ts):
    return np.interp(ophys_ts, signal_ts, signal)

running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'],
    nwb_data['running_timestamps'],
    nwb_data['ophys_timestamps']
)
# ...
running_edges = compute_percentile_bins(all_running_flat, n_bins=5)
running_binned = digitize_to_bins(running, running_edges)
```

iii. From CONVERSION_NOTES.md: "Global percentile bins computed across ALL sessions' running speed data. 5 equal percentile bins (0-20%, 20-40%, 40-60%, 60-80%, 80-100%)."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 percentile bins using `np.digitize` with globally computed bin edges. The bin edges exclude the first and last (using `edges[1:-1]`), producing bins 0 through 4.

ii.
```python
def compute_percentile_bins(values, n_bins=5):
    valid = values[~np.isnan(values)]
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    return edges

def digitize_to_bins(values, edges):
    bins = np.digitize(values, edges[1:-1])
    return bins
```

iii. Percentile-based binning ensures roughly equal counts per bin.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to ophys timestamps before trial segmentation, then extracted using the same frame indices as neural data.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'],
    nwb_data['running_timestamps'],
    nwb_data['ophys_timestamps']
)
# ...
trial_running = running_at_ophys[frame_indices]
```

iii. Alignment is guaranteed because running speed is resampled to ophys timestamps.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `pupil_width` in the NWB eye tracking data (`acquisition/EyeTracking/pupil_tracking/width`). Blink frames (from `likely_blink`) are excluded.

ii.
```python
data['pupil_width'] = pupil['width'][:]
data['likely_blink'] = f['acquisition/EyeTracking/likely_blink/data'][:]
```

iii. From CONVERSION_NOTES.md: "Pupil width from fitted ellipse used as diameter measure. Blink periods excluded."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames are set to NaN. Valid (non-NaN) pupil values are interpolated to ophys timestamps using `np.interp`. Discretized into 5 percentile bins globally. Sessions without pupil data get assigned the median bin (bin 2).

ii.
```python
pupil_raw = nwb_data['pupil_width'].copy()
blink_mask = nwb_data['likely_blink']
if blink_mask is not None:
    pupil_raw[blink_mask] = np.nan
valid_mask = ~np.isnan(pupil_raw)
if np.sum(valid_mask) > 10:
    pupil_at_ophys = np.interp(
        ophys_ts,
        nwb_data['eye_timestamps'][valid_mask],
        pupil_raw[valid_mask]
    )
```

iii. From CONVERSION_NOTES.md: "Blink periods excluded (set to NaN before interpolation). Sessions without pupil data: assigned median bin (bin 2)."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 percentile bins using `np.digitize` with globally computed bin edges. Sessions without pupil data are assigned bin 2 (median).

ii.
```python
if pupil_edges is not None and not np.all(np.isnan(pupil)):
    pupil_binned = digitize_to_bins(pupil, pupil_edges)
else:
    pupil_binned = np.full(len(pupil), 2, dtype=int)
```

iii. Assigning the median bin for missing data is a reasonable default.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed -- pupil is interpolated to ophys timestamps and extracted using the same frame indices.

ii.
```python
pupil_at_ophys = np.interp(
    ophys_ts,
    nwb_data['eye_timestamps'][valid_mask],
    pupil_raw[valid_mask]
)
# ...
trial_pupil = pupil_at_ophys[frame_indices]
```

iii. Alignment is guaranteed by resampling to ophys timestamps.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB trials table.

ii.
```python
data['trial_hit'] = trials['hit'][:]
data['trial_miss'] = trials['miss'][:]
data['trial_false_alarm'] = trials['false_alarm'][:]
data['trial_correct_reject'] = trials['correct_reject'][:]
```
```python
def get_trial_outcome(trial_idx, nwb_data):
    if nwb_data['trial_hit'][trial_idx]:
        return 0  # hit
    elif nwb_data['trial_miss'][trial_idx]:
        return 1  # miss
    elif nwb_data['trial_false_alarm'][trial_idx]:
        return 2  # false_alarm
    elif nwb_data['trial_correct_reject'][trial_idx]:
        return 3  # correct_reject
    else:
        return -1  # unknown
```

iii. These four outcomes are the standard trial outcome labels for the change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_reject). The outcome is replicated across all timepoints as a static per-trial value. Trials with unknown outcome (-1) are excluded.

ii.
```python
np.full(n_t, outcome, dtype=np.int64),
```

iii. From CONVERSION_NOTES.md: "Static per trial (replicated across all timepoints for time-varying format)."

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing pupil data**: Sessions without pupil tracking data get all NaN pupil values, which are assigned median bin (2) during discretization.
- **Blink frames**: Set to NaN before interpolation, so blinks don't corrupt the signal.
- **Trials with few frames**: Trials with < 2 ophys frames are skipped.
- **Missing stimulus**: Trials with no valid stimulus (all image indices remain -1) are skipped.
- **No valid neurons**: Experiments with 0 valid ROIs are skipped entirely.
- **Unknown outcomes**: Trials with no matching outcome (-1) are excluded.

ii.
```python
if nwb_data['pupil_width'] is None:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
# ...
pupil_binned = np.full(len(pupil), 2, dtype=int)  # median bin fallback
# ...
if len(frame_indices) < 2:
    continue
# ...
if outcome == -1:
    continue
```

iii. From CONVERSION_NOTES.md: "Pupil data quality varies across sessions; some sessions have no usable pupil data."

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) loading each NWB file with `h5py` (I/O bound), and (2) the first pass to collect all running and pupil values for global binning, which requires loading every NWB file. The second pass reloads all NWB files again.

ii.
```python
# First pass
for idx, (_, row) in enumerate(exp_table.iterrows()):
    nwb_data = load_nwb_data(nwb_path)
    # ...
# Second pass
for idx, (_, row) in enumerate(exp_table.iterrows()):
    nwb_data = load_nwb_data(nwb_path)
```

iii. Each NWB file contains large arrays of neural, behavioral, and stimulus data.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The image identity assignment loop iterates over all stimulus presentations one at a time, applying a mask for each. This could be vectorized using `np.searchsorted`. Similarly, the forward-fill loop for grey screen periods could use `pandas.ffill` or similar.

ii.
```python
for i in range(len(stim_starts)):
    if stim_omitted[i] == 1.0:
        continue
    mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
    image_at_ophys[mask] = img_idx
```

iii. N/A

## 9-c. What processing does the code repeat multiple times?

i. The code loads each NWB file **twice**: once in the first pass (to collect running/pupil statistics for global binning) and once in the second pass (to process trials). It also collects image names in a separate preliminary pass. The interpolation of running speed and pupil data to ophys timestamps is also computed twice for each experiment.

ii.
```python
# Image name collection pass
for _, row in exp_table.iterrows():
    with h5py.File(nwb_path, 'r') as f:
        # ...

# First pass: behavioral statistics
for idx, (_, row) in enumerate(exp_table.iterrows()):
    nwb_data = load_nwb_data(nwb_path)
    running = interpolate_to_ophys(...)
    # ...

# Second pass: full processing
for idx, (_, row) in enumerate(exp_table.iterrows()):
    nwb_data = load_nwb_data(nwb_path)
    # interpolation happens again inside process_experiment
```

iii. N/A

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads dF/F traces (`processing/ophys/dff/traces/data`) in every NWB file but never uses them (only events are used for neural data). It also loads `pupil_area` but only uses `pupil_width`. The full-session `events_full` array is returned from `process_experiment` but never used by the caller.

ii.
```python
dff = f['processing/ophys/dff/traces/data'][:]
data['dff'] = dff  # loaded but never used for neural data

data['pupil_area'] = pupil['area'][:]  # loaded but never used

# In process_experiment return:
return neural_trials, output_trials, trial_outcomes, running_at_ophys, pupil_at_ophys, events_full
# events_full, running_at_ophys, pupil_at_ophys are not used after return
```

iii. N/A
