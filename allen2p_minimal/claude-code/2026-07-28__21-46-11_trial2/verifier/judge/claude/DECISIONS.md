# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files using `h5py`, rather than through the Allen SDK's `VisualBehaviorOphysProjectCache`. It reads a CSV experiment table from `project_metadata/ophys_experiment_table.csv`, filters for available NWB files and active-behavior sessions (`passive == False`), and loads each NWB file individually. It does NOT filter by `project_code == 'VisualBehavior'` — it includes both VisualBehavior (single-plane) and VisualBehaviorMultiscope (multi-plane) experiments.

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
    data = {}
    with h5py.File(nwb_path, 'r') as f:
        ophys_ts = f['processing/ophys/dff/traces/timestamps'][:]
        events = f['processing/ophys/event_detection/data'][:]
        ...
    return data
```

iii. The AI's CONVERSION_NOTES.md states: "All project codes included: VisualBehavior (single-plane, ~31 Hz) and VisualBehaviorMultiscope (multi-plane, ~11 Hz)." The AI chose to include all active behavior sessions rather than filtering to the VisualBehavior project code only. This was a deliberate choice to include more data, but deviates from the reference which filters to `project_code == 'VisualBehavior'`.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values from the experiment table, sorted and converted to strings.

ii.
```python
all_subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
```

iii. This is a straightforward mapping from the experiment table's `mouse_id` column. The approach is consistent with the reference.

## 1-c. How are the data split into sessions?

i. The AI treats each individual experiment (single imaging plane) as a separate session. This is different from the reference, which groups experiments by `ophys_session_id` to combine multiple imaging planes from the same behavioral session into one session.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    ...
    nwb_path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{eid}.nwb')
    nwb_data = load_nwb_data(nwb_path)
    result = process_experiment(nwb_data, all_image_names)
    ...
    neural_all.append(final_neural)
```

iii. The CONVERSION_NOTES.md states: "Each experiment = 1 session in the output." The AI did not group experiments by `ophys_session_id`. For the VisualBehavior (single-plane) project, each session has only one experiment, so this is equivalent. However, for VisualBehaviorMultiscope, multiple experiments from the same session are treated as separate sessions, duplicating behavioral data across planes.

## 1-d. How are the data split into trials?

i. Trials are segmented using `start_time` and `stop_time` from the NWB trials table. Aborted trials, auto-rewarded trials, and trials that are neither Go nor Catch are excluded. Trials with fewer than 2 ophys frames are also excluded. The trial window is variable-length based on the interval [start_time, stop_time).

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

# In process_experiment:
for trial_idx in range(n_trials_total):
    if not get_trial_mask(trial_idx, nwb_data):
        continue
    ...
    frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
    frame_indices = np.where(frame_mask)[0]
    if len(frame_indices) < 2:
        continue
```

iii. The AI uses Go/Catch flags explicitly rather than just excluding aborted and auto-rewarded trials. The reference also requires `change_time` to be non-NaN, which the AI does not check — instead it checks `go or catch`. The trial window boundaries are the same (start_time to stop_time), and both approaches yield variable-length trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters out: (1) aborted trials, (2) auto-rewarded trials, (3) trials that are neither Go nor Catch, (4) trials with fewer than 2 ophys frames, (5) trials with unknown outcome (outcome == -1), (6) trials with no valid stimulus (all image indices remain -1). Sessions with fewer than 2 valid trials are skipped.

ii.
```python
if not get_trial_mask(trial_idx, nwb_data):
    continue
outcome = get_trial_outcome(trial_idx, nwb_data)
if outcome == -1:
    continue
...
if len(frame_indices) < 2:
    continue
...
# Skip trial if no valid image found
if trial_image[0] < 0:
    first_valid = np.where(trial_image >= 0)[0]
    if len(first_valid) > 0:
        trial_image[:first_valid[0]] = trial_image[first_valid[0]]
    else:
        continue
...
if neural_trials is None or len(neural_trials) < 2:
    ...skipped...
```

iii. The AI applies several layers of filtering. The Go/Catch check is more explicit than the reference's approach. Filtering for valid_roi on neurons is an additional quality check not in the reference.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses **detected calcium events** (`processing/ophys/event_detection/data`) from the NWB files, NOT dF/F traces. This is a significant departure from the reference, which uses `dff_traces`.

ii.
```python
# In load_nwb_data:
events = f['processing/ophys/event_detection/data'][:]
data['events'] = events  # shape: (n_timepoints, n_cells)

# In process_experiment:
events = nwb_data['events']  # (n_timepoints, n_cells)
valid = nwb_data['valid_roi']
events = events[:, valid]
...
trial_neural = events[frame_indices, :].T  # (n_neurons, n_timepoints)
```

iii. The AI's CONVERSION_NOTES.md justifies this: "Used detected calcium events (processing/ophys/event_detection/data) rather than raw dF/F traces. This matches the paper: 'We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces.'" The AI also filters neurons by `valid_roi`, whereas the reference includes all neurons from `dff_traces`.

## 2-b. How is the `neural` data processed?

i. The AI transposes the events matrix from (n_timepoints, n_cells) to (n_cells, n_timepoints) for each trial. It filters neurons by `valid_roi` flag. No other processing is applied. Unlike the reference, it does NOT merge neurons from multiple imaging planes (since each experiment is treated as a separate session).

ii.
```python
events = nwb_data['events']  # (n_timepoints, n_cells)
valid = nwb_data['valid_roi']
events = events[:, valid]
...
trial_neural = events[frame_indices, :].T  # (n_neurons, n_timepoints)
neural_trials.append(trial_neural.astype(np.float32))
```

iii. The AI treats each experiment independently, so there is no need to merge across planes. The `valid_roi` filtering is an additional quality step compared to the reference.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons using the `valid_roi` flag from the cell specimen table in the NWB file. Neurons with `valid_roi == False` are excluded. Sessions with 0 valid neurons are skipped entirely.

ii.
```python
valid = nwb_data['valid_roi']
events = events[:, valid]
n_timepoints_total, n_neurons = events.shape
if n_neurons == 0:
    return None, None, None, None, None, None
```

iii. The reference does not apply any additional neural quality filtering beyond what the SDK provides in `dff_traces`. The AI's `valid_roi` filtering is an additional step.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to ophys timestamps. For each trial, ophys frames within [start_time, stop_time) are extracted. There is no alignment to a specific event like `change_time` — alignment is to trial start.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
frame_indices = np.where(frame_mask)[0]
...
trial_neural = events[frame_indices, :].T
```

iii. This is consistent with the reference approach of using the full trial window from start_time to stop_time. Both approaches align to ophys timestamps.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The data is kept at the native ophys frame rate. The time bin size is computed from the median imaging rate across sessions. Since the AI includes both single-plane (~31 Hz) and multi-plane (~11 Hz) experiments, the reported time bin size differs from the reference.

ii.
```python
imaging_rates = [s['imaging_rate_hz'] for s in session_info]
median_rate = np.median(imaging_rates)
time_bin_ms = 1000.0 / median_rate
```

iii. The CONVERSION_NOTES.md reports: "Time bin size is the inter-frame interval (~32.26 ms for single-plane at ~31 Hz, ~93.2 ms for multi-plane at ~11 Hz). Metadata `time_bin_size` reports median across sessions (32.26 ms)." The reference computes ~93 ms from ophys timestamps. The different time bin size results from including single-plane experiments with higher frame rates.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the **stimulus presentations table** in the NWB file (`intervals/Natural*` keys), using `start_time`, `stop_time`, and `image_name` fields. This differs from the reference, which uses `initial_image_name` and `change_image_name` from the trials table.

ii.
```python
stim_keys = [k for k in f['intervals'] if 'Natural' in k or 'natural_scene' in k.lower()]
...
data['stim_start_times'] = stim['start_time'][:]
data['stim_stop_times'] = stim['stop_time'][:]
data['stim_image_names'] = np.array([s.decode() if isinstance(s, bytes) else s for s in stim['image_name'][:]])
```

iii. The AI builds a per-timepoint image identity from the stimulus presentations, which provides exact timing of each image flash. During grey screen intervals, it carries forward the last shown image. This is a more granular approach than the reference's two-state (initial/change) approach.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each ophys timepoint, the AI checks which stimulus presentation interval it falls within and assigns the corresponding image index. During grey screen (inter-stimulus intervals), the last shown image is carried forward. Omitted stimuli are skipped. Image names are mapped to sorted integer indices globally.

ii.
```python
image_at_ophys = np.full(len(ophys_ts), -1, dtype=int)
for i in range(len(stim_starts)):
    if stim_omitted[i] == 1.0:
        continue
    img_name = stim_names[i]
    if img_name == 'omitted':
        continue
    img_idx = image_name_to_idx[img_name]
    mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
    image_at_ophys[mask] = img_idx

# Forward-fill grey screen gaps within trials
last_img = -1
for k in range(len(trial_image)):
    if trial_image[k] >= 0:
        last_img = trial_image[k]
    elif last_img >= 0:
        trial_image[k] = last_img
```

iii. The AI's approach is more detailed than the reference's simple initial/change split at change_time. Both ultimately produce time-varying image identity, but the AI's version reflects the actual stimulus presentation timing including grey screen periods.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the ophys timepoint level and indexed using the same `frame_indices` as the neural data.

ii.
```python
trial_image = image_at_ophys[frame_indices]
```

iii. Same alignment approach as neural data — both use ophys timestamps as the common timebase.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in the **stimulus presentations table**, NOT from the trials table's `change_time` and `go` columns as in the reference.

ii.
```python
stim_is_change = nwb_data['stim_is_change']
for i in range(len(stim_starts)):
    if stim_is_change[i] == 1.0:
        mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
        is_change_at_ophys[mask] = 1
```

iii. The AI uses the stimulus-level `is_change` flag, which marks the actual change stimulus presentation. This is functionally similar but derived from a different source than the reference's approach.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary array is created at the ophys timepoint level. For each stimulus presentation where `is_change == 1`, the ophys frames during that stimulus window are set to 1. All other timepoints are 0.

ii.
```python
is_change_at_ophys = np.zeros(len(ophys_ts), dtype=int)
stim_is_change = nwb_data['stim_is_change']
for i in range(len(stim_starts)):
    if stim_is_change[i] == 1.0:
        mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
        is_change_at_ophys[mask] = 1
```

iii. The reference uses a 750ms window (one flash + grey) from `change_time` for go trials only. The AI uses the actual stimulus presentation window from the stimulus table, and does not restrict to go trials only — `is_change` in the stimulus table may be True for catch trials too (though in practice this should only be True for actual image changes).

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1). No thresholding is needed.

ii. See 4-b.

iii. Same as reference — binary variable, no additional processing.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same approach as image identity — computed at ophys timepoint level and indexed by `frame_indices`.

ii.
```python
trial_change = is_change_at_ophys[frame_indices]
```

iii. Consistent alignment via ophys timestamps.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
data['running_speed'] = f['processing/running/speed/data'][:]
data['running_timestamps'] = f['processing/running/speed/timestamps'][:]
```

iii. This is the standard running speed signal, consistent with the reference's `dataset.running_speed`.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to ophys timestamps using `np.interp`. Then it is discretized into 5 equal percentile bins computed globally across all sessions.

ii.
```python
def interpolate_to_ophys(signal, signal_ts, ophys_ts):
    return np.interp(ophys_ts, signal_ts, signal)

running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts)

# Global binning
all_running_flat = np.concatenate(all_running)
running_edges = compute_percentile_bins(all_running_flat, n_bins=5)
running_binned = digitize_to_bins(running, running_edges)
```

iii. The reference uses `scipy.interp1d` with `bounds_error=False, fill_value=np.nan`, while the AI uses `np.interp` which extrapolates at boundaries rather than producing NaN. The binning approach is the same (global percentile bins).

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using global percentile edges (0th, 20th, 40th, 60th, 80th, 100th percentiles). `np.digitize` with inner edges is used to assign bin indices 0-4.

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

iii. Consistent with reference's `discretize` and `apply_discretize` functions. The reference maps NaN values to bin 0; the AI avoids NaN entirely by using `np.interp` (which extrapolates).

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to ophys timestamps before trial segmentation, then indexed by the same `frame_indices`.

ii.
```python
running_at_ophys = interpolate_to_ophys(nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts)
...
trial_running = running_at_ophys[frame_indices]
```

iii. Same alignment approach as the reference.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/width` in the NWB file, with blink detection from `acquisition/EyeTracking/likely_blink/data`.

ii.
```python
data['pupil_width'] = pupil['width'][:]
data['likely_blink'] = f['acquisition/EyeTracking/likely_blink/data'][:]
```

iii. Consistent with the reference, which uses `dataset.eye_tracking` `pupil_width` column.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames are set to NaN, then valid (non-NaN) pupil values are interpolated to ophys timestamps using `np.interp`. Sessions without pupil data get NaN. Pupil is then discretized into 5 percentile bins globally. Sessions without pupil data are assigned the median bin (bin 2).

ii.
```python
pupil_raw = nwb_data['pupil_width'].copy()
blink_mask = nwb_data['likely_blink']
if blink_mask is not None:
    pupil_raw[blink_mask] = np.nan
valid_mask = ~np.isnan(pupil_raw)
if np.sum(valid_mask) > 10:
    pupil_at_ophys = np.interp(ophys_ts, nwb_data['eye_timestamps'][valid_mask], pupil_raw[valid_mask])
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)

# Missing pupil fallback:
if pupil_edges is not None and not np.all(np.isnan(pupil)):
    pupil_binned = digitize_to_bins(pupil, pupil_edges)
else:
    pupil_binned = np.full(len(pupil), 2, dtype=int)  # median bin
```

iii. The blink exclusion approach is the same as the reference. The fallback to median bin for missing pupil data is a reasonable choice; the reference maps NaN to bin 0.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed — 5 percentile-based bins computed globally, applied with `np.digitize`.

ii. See 5-c for binning code; same functions are used.

iii. Consistent with reference.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed — interpolated to ophys timestamps, then indexed by `frame_indices`.

ii.
```python
trial_pupil = pupil_at_ophys[frame_indices]
```

iii. Consistent alignment via ophys timestamps.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean flags `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB trials table.

ii.
```python
data['trial_hit'] = trials['hit'][:]
data['trial_miss'] = trials['miss'][:]
data['trial_false_alarm'] = trials['false_alarm'][:]
data['trial_correct_reject'] = trials['correct_reject'][:]

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
```

iii. Consistent with the reference approach. The order and mapping are the same.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is mapped to an integer code (0-3) and replicated across all timepoints in the trial as a static per-trial variable.

ii.
```python
np.full(n_t, outcome, dtype=np.int64)
```

iii. Same approach as the reference.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Missing pupil data**: Sessions without pupil tracking data (`pupil_width is None`) get NaN, which is later mapped to the median bin (bin 2).
- **Blink frames**: Set to NaN before interpolation, excluded from the interpolation source.
- **Missing stimuli in trial**: If no valid stimulus is found in a trial (all image indices -1), the trial is skipped.
- **Insufficient trials**: Sessions with fewer than 2 valid trials are skipped.
- **No valid neurons**: Sessions with 0 valid ROIs are skipped.
- **Trials with < 2 frames**: Skipped.

ii.
```python
if nwb_data['pupil_width'] is None:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
...
if len(frame_indices) < 2:
    continue
...
if neural_trials is None or len(neural_trials) < 2:
    ...skipped...
```

iii. The AI handles missing data more explicitly than the reference, which wraps session processing in a try/except. The AI's approach avoids silent failures but requires more explicit checks throughout the code.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading NWB files with h5py (I/O bound), (2) The per-stimulus-presentation loop for building image identity time series, which iterates over every stimulus presentation for every timepoint.

ii.
```python
for i in range(len(stim_starts)):
    if stim_omitted[i] == 1.0:
        continue
    ...
    mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
    image_at_ophys[mask] = img_idx
```

iii. The I/O for loading NWB files is the primary bottleneck, similar to the reference's SDK loading.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The stimulus presentation loop in `process_experiment` iterates over each stimulus to build the image identity time series. This could be vectorized using `np.searchsorted` to find frame boundaries for all stimuli at once.

ii.
```python
for i in range(len(stim_starts)):
    ...
    mask = (ophys_ts >= stim_starts[i]) & (ophys_ts < stim_stops[i])
    image_at_ophys[mask] = img_idx
```

Also the forward-fill loop for grey screen gaps:
```python
for k in range(len(trial_image)):
    if trial_image[k] >= 0:
        last_img = trial_image[k]
    elif last_img >= 0:
        trial_image[k] = last_img
```

iii. Both loops are O(n) per stimulus/timepoint and could be vectorized.

## 9-c. What processing does the code repeat multiple times?

i. The code loads each NWB file **twice**: once in the first pass to collect running and pupil statistics for global percentile binning, and again in the second pass to process experiments. It also opens all NWB files a third time at the beginning to collect image names.

ii.
```python
# First: collect image names
for _, row in exp_table.iterrows():
    ...
    with h5py.File(nwb_path, 'r') as f:
        ...

# Second: collect behavioral statistics
for idx, (_, row) in enumerate(exp_table.iterrows()):
    ...
    nwb_data = load_nwb_data(nwb_path)

# Third: process experiments
for idx, (_, row) in enumerate(exp_table.iterrows()):
    ...
    nwb_data = load_nwb_data(nwb_path)
```

iii. This triples the I/O cost. The reference loads each session only once and stores the extracted trial data for later use.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads dF/F traces (`processing/ophys/dff/traces/data`) in `load_nwb_data` even though it only uses calcium events. It also loads `pupil_area` which is never used. The `events_full` return value from `process_experiment` is never used by the caller.

ii.
```python
dff = f['processing/ophys/dff/traces/data'][:]
data['dff'] = dff  # loaded but never used for neural data

data['pupil_area'] = pupil['area'][:]  # loaded but never used

# In the caller:
neural_trials, raw_output_trials, trial_outcomes, running_at_ophys, pupil_at_ophys, events_full = result
# events_full, running_at_ophys, pupil_at_ophys are never used after this
```

iii. Loading dF/F unnecessarily increases memory usage and I/O time, especially since the full dF/F array is large.
