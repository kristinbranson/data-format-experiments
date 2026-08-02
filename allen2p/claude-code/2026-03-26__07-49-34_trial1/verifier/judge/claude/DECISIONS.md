# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by reading NWB files directly via `h5py`, rather than using the Allen SDK's `VisualBehaviorOphysProjectCache`. It reads the experiment table from a CSV file in the metadata directory, then filters to experiments whose NWB files are locally available. It further filters to active (non-passive) sessions. Each NWB file is opened individually via `load_nwb_data()`.

ii.
```python
def load_experiment_table():
    exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
    nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
    downloaded_ids = set()
    for f in nwb_files:
        eid = int(os.path.basename(f).replace('behavior_ophys_experiment_', '').replace('.nwb', ''))
        downloaded_ids.add(eid)
    exp_table = exp_table[exp_table['ophys_experiment_id'].isin(downloaded_ids)].copy()
    exp_table = exp_table[~exp_table['session_type'].str.contains('passive', case=False)].copy()
    return exp_table
```

```python
def load_nwb_data(nwb_path):
    data = {}
    with h5py.File(nwb_path, 'r') as f:
        data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
        dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
        data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
        ...
    return data
```

iii. The AI chose to use h5py for direct NWB reading instead of the Allen SDK for speed and to avoid SDK overhead. The experiment table CSV provides metadata for filtering. Active-only filtering was applied based on session type names containing "passive".

## 1-b. How are the data split into subjects?

i. Subjects correspond to unique `mouse_id` values in the experiment table. A mapping from mouse_id to index is built dynamically as experiments are processed.

ii.
```python
subject_map = {}  # mouse_id -> index
...
mouse_id = result['mouse_id']
if mouse_id not in subject_map:
    subject_map[mouse_id] = len(all_subjects)
    all_subjects.append(mouse_id)
```

iii. The `mouse_id` column in the experiment table uniquely identifies each animal. Subjects are registered as encountered during processing.

## 1-c. How are the data split into sessions?

i. Each experiment (single imaging plane) is treated as a separate "session" in the output. The AI does NOT group multiple experiments by `ophys_session_id`. Each row in the experiment table with a downloaded NWB file becomes one session.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    result = process_experiment(exp_id, row, image_names_list, outcome_names, ...)
    ...
    all_neural.append(result['neural'])
```

iii. The CONVERSION_NOTES (Step 5, Decision 9) states: "Multiscope handling: Each experiment (plane) is a separate 'session' in the output, since they have different neurons but share the same behavioral data." This means multi-plane sessions from VisualBehaviorMultiscope are split into separate output sessions.

## 1-d. How are the data split into trials?

i. Trials are defined using the trials table from the NWB file. Valid trials are those that are Go or Catch, excluding Aborted and Auto-rewarded. The trial window spans from `start_time` to `stop_time` (variable length). Ophys frames within this window are extracted using a boolean mask.

ii.
```python
def get_valid_trials(trial_data):
    go = trial_data['go'].astype(bool)
    catch = trial_data['catch'].astype(bool)
    aborted = trial_data['aborted'].astype(bool)
    auto_rewarded = trial_data['auto_rewarded'].astype(bool)
    valid = (go | catch) & ~aborted & ~auto_rewarded
    return np.where(valid)[0]
```

```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
```

iii. The filtering follows the instructions: include Go and Catch trials, exclude Aborted and Auto-rewarded. The boolean mask `(ophys_ts >= t_start) & (ophys_ts < t_stop)` extracts frames within the trial window.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) must be Go or Catch, (2) must not be Aborted or Auto-rewarded, (3) must have at least 2 ophys frames, (4) sessions with fewer than 2 valid trials are excluded.

ii.
```python
valid = (go | catch) & ~aborted & ~auto_rewarded
...
if n_trial_frames < 2:
    continue
...
if len(neural_trials) < 2:
    return None
```

iii. The minimum frame count ensures trials have usable data. The minimum 2-trial threshold per session prevents degenerate sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dff_traces` (dF/F calcium fluorescence traces), read directly from the NWB file at path `processing/ophys/dff/traces/data`.

ii.
```python
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
```

iii. dF/F is the standard measure for two-photon calcium imaging. It is pre-computed in the NWB files.

## 2-b. How is the `neural` data processed?

i. No additional processing is applied. The dF/F traces are read from NWB and transposed to (n_cells, n_frames). For each trial, the subset of frames within the trial window is extracted.

ii.
```python
neural = dff[:, frame_mask].astype(np.float32)  # (n_cells, n_trial_frames)
```

iii. The dF/F traces are already processed by the Allen SDK pipeline (motion correction, neuropil subtraction, baseline normalization).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit quality filtering is applied to neurons. All cells present in the NWB file's dff_traces are included without checking `valid_roi` status.

ii. N/A (no filtering code)

iii. The CONVERSION_NOTES (Step 10, Check 3) states: "No explicit valid_roi filter... OK - all ROIs in downloaded NWB files are valid." However, this assumption may not hold if the NWB files contain ROIs that the SDK would normally exclude via `exclude_invalid_rois=True`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the ophys timestamps. Each trial's data is extracted using a boolean mask on the ophys timestamps between `start_time` and `stop_time` from the trials table. There is no alignment to a specific event like stimulus change; it is aligned to the trial start.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The instructions say "Temporally align based on ophys timestamp." The variable-length trial window from start_time to stop_time is used.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native ophys frame rate is used without any rebinning. The time bin size is computed as the median inter-frame interval across sessions (~32.3 ms for Scientifica at ~31 Hz, ~90 ms for Multiscope at ~11 Hz). Since both project codes are included, sessions have different frame rates.

ii.
```python
dt = np.median(np.diff(ophys_ts))  # time bin size
...
all_dts = [m['dt_ms'] for m in session_metadata]
median_dt = np.median(all_dts)
```

iii. No rebinning is applied. The median time bin size across all sessions is stored in metadata.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the **stimulus presentations table** in the NWB file (`intervals/Natural_Images_*_presentations`), specifically the `image_name`, `start_time`, and `stop_time` columns. It is NOT derived from the trials table's `initial_image_name`/`change_image_name`.

ii.
```python
stim_key = None
for key in f['intervals'].keys():
    if 'Natural_Images' in key or 'natural_images' in key:
        stim_key = key
        break
...
stim_data = {}
for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
    if key in stim:
        stim_data[key] = stim[key][:]
```

iii. The stimulus presentations table provides exact timing of each image flash, allowing frame-level image identity assignment including gray screen (ISI) periods.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are collected across all experiments to build a global mapping. A "gray" category is added for inter-stimulus intervals. For each trial, the stimulus presentations overlapping the trial window are mapped to ophys frames. During image flashes, the image index is assigned; during ISI periods, the gray index is assigned. Omitted stimuli are treated as gray screen.

ii.
```python
image_names_list = [GRAY_LABEL] + all_image_names
...
def build_image_identity_trace(ophys_ts, stim_data, trial_start, trial_stop, image_names_list):
    gray_idx = image_names_list.index(GRAY_LABEL)
    trace = np.full(n_frames, gray_idx, dtype=np.int64)
    for si in range(len(stim_starts)):
        name = stim_names[si]
        if name == 'omitted':
            continue
        if name in image_names_list:
            img_idx = image_names_list.index(name)
        frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
        trace[frame_mask] = img_idx
    return trace, trial_mask
```

iii. This approach faithfully represents what is on screen at each moment, including the gray inter-stimulus intervals.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed per ophys frame using the same ophys timestamps as the neural data. Stimulus presentation start/stop times are used to determine which image (or gray screen) is on screen at each frame.

ii.
```python
frame_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[frame_mask]
...
frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
trace[frame_mask] = img_idx
```

iii. Both neural and image identity share the same ophys timebase, ensuring temporal alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in the stimulus presentations table, along with `start_time` for the change onset timing.

ii.
```python
stim_starts = stim_data['start_time']
is_change = stim_data['is_change']
```

iii. The `is_change` flag in the stimulus presentations identifies which stimulus flash is the change event.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary trace is created. For each stimulus presentation with `is_change=True`, the single ophys frame at or immediately after the change onset is set to 1. All other frames are 0.

ii.
```python
def build_image_change_trace(ophys_ts, stim_data, trial_start, trial_stop):
    trace = np.zeros(n_frames, dtype=np.int64)
    for si in range(len(stim_starts)):
        if not is_change[si]:
            continue
        s_start = stim_starts[si]
        frame_idx = np.searchsorted(trial_ts, s_start)
        if frame_idx < n_frames:
            trace[frame_idx] = 1
    return trace
```

iii. Only a single frame is marked as 1 at each change onset. This is a very sparse signal.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is a binary variable (0 or 1), not thresholded. 0 = no change, 1 = change at this frame.

ii. See 4-b above.

iii. N/A - already binary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change uses the same ophys timestamps as neural data. The change frame is identified via `np.searchsorted` on the trial's ophys timestamps.

ii. See 4-b above.

iii. Same timebase as neural data ensures alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed` in the NWB file, which provides speed values and timestamps from the running wheel encoder.

ii.
```python
data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
data['running_speed'] = f['processing']['running']['speed']['data'][:]
```

iii. This is the standard running speed data path in Allen NWB files.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps (~60 Hz) to the ophys timebase. Then it is discretized into 5 percentile-based bins computed **per session**. NaN values are mapped to bin 0.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
...
running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)
...
running_binned = apply_percentile_bins(running_trial, running_edges, n_bins=5)
```

```python
def compute_session_percentile_edges(values, n_bins=5):
    valid = values[~np.isnan(values)]
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges
```

iii. Linear interpolation to ophys timebase followed by per-session percentile binning.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using percentile-based edges (0th, 20th, 40th, 60th, 80th, 100th percentiles). Edges are set to -inf and +inf at boundaries. Bin edges are computed per-session from non-NaN values. `np.digitize` maps values to bins 0-4.

ii.
```python
def apply_percentile_bins(values, edges, n_bins=5):
    valid = ~np.isnan(values)
    result = np.zeros(len(values), dtype=np.int64)
    if valid.sum() > 0:
        result[valid] = np.clip(np.digitize(values[valid], edges[1:-1]), 0, n_bins - 1)
    result[~valid] = 0
    return result
```

iii. Per-session percentile binning ensures balanced bins within each session.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to ophys timestamps for the entire session, then the trial window frames are extracted using the same boolean mask as neural data.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
...
running_trial = running_at_ophys[frame_mask]
```

iii. Same ophys timebase ensures alignment with neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `pupil_area` in the eye tracking data (`acquisition/EyeTracking/pupil_tracking/area`), with blink detection from `acquisition/EyeTracking/likely_blink`.

ii.
```python
data['pupil_area'] = pt['area']['data'][:] if 'data' in pt['area'] else pt['area'][:]
data['pupil_timestamps'] = pt['timestamps'][:]
data['likely_blink'] = et['likely_blink']['data'][:]
```

iii. The AI chose to derive pupil diameter from pupil area, rather than using a direct diameter/width measurement.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter is computed from pupil area as `2 * sqrt(area / pi)`. Blink frames are set to NaN. Then diameter is linearly interpolated to ophys timestamps, and discretized into 5 per-session percentile bins.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_diameter = np.full_like(pupil_area, np.nan)
valid_pupil = ~np.isnan(pupil_area) & (pupil_area > 0)
pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
...
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)
```

iii. Area-to-diameter conversion assumes a circular pupil. Blink removal before interpolation prevents artifacts.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 per-session percentile bins using `np.digitize`, with NaN mapped to bin 0.

ii. See 5-c (same `apply_percentile_bins` function).

iii. Per-session percentile binning adapts to each session's pupil range.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed: interpolated to ophys timestamps, then trial window extracted using same boolean mask.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
...
pupil_trial = pupil_at_ophys[frame_mask]
```

iii. Same ophys timebase ensures alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB trials table.

ii.
```python
def get_trial_outcome(trial_data, idx):
    if trial_data['hit'][idx]:
        return 'hit'
    elif trial_data['miss'][idx]:
        return 'miss'
    elif trial_data['false_alarm'][idx]:
        return 'false_alarm'
    elif trial_data['correct_reject'][idx]:
        return 'correct_reject'
    else:
        return 'unknown'
```

iii. These four outcomes are mutually exclusive for valid (non-aborted, non-auto-rewarded) trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome string is mapped to an integer index via `outcome_names.index(outcome)`. The outcome is constant across all timepoints in a trial (broadcast to all frames).

ii.
```python
outcome_names = ['hit', 'miss', 'false_alarm', 'correct_reject']
...
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
...
output_full[4] = outcome_idx  # broadcast scalar to all timepoints
```

iii. Trial outcome is a per-trial static variable, broadcast to all timepoints for consistency with the output array format (5, n_frames).

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing NWB files**: Experiments without a corresponding NWB file are skipped.
- **No cells**: Experiments with 0 cells are skipped.
- **Missing stimulus data**: Experiments without stimulus presentation data are skipped.
- **Few valid trials**: Experiments with <2 valid trials after processing are excluded.
- **Short trials**: Trials with <2 ophys frames are skipped.
- **Missing eye tracking**: If no pupil data, all pupil values are set to NaN (mapped to bin 0).
- **Blinks**: Blink frames in pupil data are set to NaN before interpolation.
- **NaN behavioral data**: NaN values from interpolation are mapped to bin 0 during discretization.

ii.
```python
if n_cells == 0:
    return None
if stim_data is None:
    return None
if n_trial_frames < 2:
    continue
if len(neural_trials) < 2:
    return None
if nwb_data['pupil_area'] is not None:
    ...
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
result[~valid] = 0  # NaN gets bin 0
```

iii. These checks ensure robust handling of incomplete or problematic data.

## 9-a. What are the most time-consuming steps of the code?

i. Loading NWB files via h5py is the most time-consuming step. Each NWB file contains large neural and behavioral data arrays. The CONVERSION_NOTES report ~1.7s per session for loading, with ~0.4s for processing.

ii. N/A

iii. I/O-bound loading dominates runtime. The AI estimated ~12.5 minutes total for full conversion.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `build_image_identity_trace` function loops over all stimulus presentations for each trial to map image identities. The `build_image_change_trace` similarly loops over presentations. The `get_all_image_names` function loops over all NWB files to collect image names. Per-trial processing in `process_experiment` iterates sequentially.

ii.
```python
for si in range(len(stim_starts)):
    ...
    frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
    trace[frame_mask] = img_idx
```

iii. The stimulus presentation loop could potentially be vectorized using `np.searchsorted` across all presentations simultaneously.

## 9-c. What processing does the code repeat multiple times?

i. The `get_all_image_names` function scans all NWB files to collect image names, then each NWB file is loaded again during the main processing loop. This means every NWB file is opened twice. Additionally, the `compute_session_percentile_edges` function is called per-session, computing bin edges independently for each session.

ii.
```python
all_image_names = get_all_image_names(exp_table)  # First pass: scan all NWBs
...
for idx, (_, row) in enumerate(exp_table.iterrows()):
    result = process_experiment(exp_id, ...)  # Second pass: load each NWB again
```

iii. Double loading of NWB files adds ~14s overhead for the image name collection pass.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores extensive per-session metadata (exp_id, session_type, cre_line, imaging_depth, dt_ms, etc.) in the output dictionary. The `plot_processing` function and its associated data retention (when `show_processing=True`) add overhead for debugging but are not needed in the final output. The `--show-processing` flag retains large arrays (ophys_ts, running_at_ophys, etc.) in memory until plotting.

ii.
```python
session_metadata.append({
    'exp_id': result['exp_id'],
    'ophys_session_id': result['ophys_session_id'],
    'session_type': result['session_type'],
    ...
})
```

iii. The metadata is useful for documentation but not used by the decoder.
