# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files using `h5py`, bypassing the AllenSDK's high-level API. It reads an experiment table CSV from `project_metadata/`, then finds downloaded NWB files in the `behavior_ophys_experiments/` directory. Only experiments with matching NWB files on disk are included. Passive sessions are filtered out by checking `session_type` for the string "passive".

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

iii. The AI justified using h5py for speed ("fast, no AllenSDK overhead") in CONVERSION_NOTES.md Step 6. The passive session filter is documented as a key decision in Step 5.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values from the experiment table. A `subject_map` dictionary maps mouse IDs to sequential indices.

ii.
```python
subject_map = {}
...
mouse_id = result['mouse_id']
if mouse_id not in subject_map:
    subject_map[mouse_id] = len(all_subjects)
    all_subjects.append(mouse_id)
```

iii. The AI uses the experiment table's `mouse_id` column, which is the standard identifier for each animal.

## 1-c. How are the data split into sessions?

i. Each experiment (single imaging plane) is treated as an independent "session" in the output. The AI does NOT group multiple imaging planes from the same ophys session together. Each NWB file becomes one entry in the session list.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    ...
    result = process_experiment(exp_id, row, image_names_list, outcome_names, ...)
    ...
    all_neural.append(result['neural'])
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 9: "Multiscope handling: Each experiment (plane) is a separate 'session' in the output, since they have different neurons but share the same behavioral data."

## 1-d. How are the data split into trials?

i. Trials are defined using the `intervals/trials` table from the NWB file. Valid trials are those that are Go or Catch, and not Aborted or Auto-rewarded. Ophys frames are extracted from `start_time` to `stop_time` using `>=` and `<` bounds, giving variable-length trials.

ii.
```python
def get_valid_trials(trial_data):
    go = trial_data['go'].astype(bool)
    catch = trial_data['catch'].astype(bool)
    aborted = trial_data['aborted'].astype(bool)
    auto_rewarded = trial_data['auto_rewarded'].astype(bool)
    valid = (go | catch) & ~aborted & ~auto_rewarded
    return np.where(valid)[0]

...
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
```

iii. The AI documented in CONVERSION_NOTES.md Step 5 that Go and Catch trials are included, Aborted and Auto-rewarded are excluded, per the instruction requirements.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) must be Go or Catch (not Aborted or Auto-rewarded), (2) must have at least 2 ophys frames, (3) sessions with fewer than 2 valid trials are discarded. There is no explicit filter for missing `change_time`.

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

iii. The AI justified these filters in CONVERSION_NOTES.md Step 5 and Step 10.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `processing/ophys/dff/traces/data` field in the NWB file, which contains pre-computed dF/F calcium fluorescence traces.

ii.
```python
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
```

iii. dF/F is noted as "pre-computed in NWB files" in CONVERSION_NOTES.md Step 1.

## 2-b. How is the `neural` data processed?

i. No additional processing is applied. The dF/F traces are read directly from NWB and transposed to (n_cells, n_frames). Per-trial slices are extracted by boolean masking on ophys timestamps.

ii.
```python
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
...
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The AI notes in CONVERSION_NOTES.md Step 1 and Step 5 that dF/F is pre-computed and no further processing is needed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit neural quality filtering is applied. All cells present in the NWB file's dF/F traces are included. The AI notes that "all ROIs in downloaded NWB files are valid" in CONVERSION_NOTES.md Step 10.

ii. N/A (no filtering code)

iii. CONVERSION_NOTES.md Step 10, Check 3: "ROI filtering: No explicit valid_roi filter... OK - all ROIs in downloaded NWB files are valid."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the ophys timestamps. Each trial's neural data spans from `start_time` to `stop_time` using boolean masking (`ophys_ts >= t_start` and `ophys_ts < t_stop`). The alignment event is trial start.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
...
neural = dff[:, frame_mask].astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 4: "Temporal alignment: Align to ophys timestamps."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. Data is kept at the native ophys frame rate. The time bin size is computed as the median inter-frame interval across sessions (~32.3 ms for Scientifica at ~31 Hz, ~93 ms for Multiscope at ~11 Hz). Sessions at different frame rates are mixed in the output.

ii.
```python
dt = np.median(np.diff(ophys_ts))
...
all_dts = [m['dt_ms'] for m in session_metadata]
median_dt = np.median(all_dts)
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 2: "Use native ophys timestamps (~31 Hz for Scientifica, ~11 Hz for Multiscope). Each session's time bin is consistent within itself."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table in the NWB file (`intervals/Natural_Images_*_presentations`), specifically the `image_name`, `start_time`, and `stop_time` fields of each presentation. A "gray" label is used for inter-stimulus intervals.

ii.
```python
def build_image_identity_trace(ophys_ts, stim_data, trial_start, trial_stop, image_names_list):
    gray_idx = image_names_list.index(GRAY_LABEL)
    trace = np.full(n_frames, gray_idx, dtype=np.int64)
    stim_starts = stim_data['start_time']
    stim_stops = stim_data['stop_time']
    stim_names = stim_data['image_name']
    for si in range(len(stim_starts)):
        ...
        frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
        trace[frame_mask] = img_idx
    return trace, trial_mask
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 5: "Image identity: Map stimulus presentations to ophys timepoints. During gray screen (ISI), use a 'gray' category."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are collected from all NWB files, sorted, and prepended with a "gray" label. Each frame within a trial is assigned to either the currently-presented image (during stimulus) or "gray" (during ISI). Omitted stimuli are treated as gray. The mapping is: gray=0, then alphabetically sorted image names.

ii.
```python
image_names_list = [GRAY_LABEL] + all_image_names
...
if name == 'omitted':
    continue
if name in image_names_list:
    img_idx = image_names_list.index(name)
```

iii. The AI justified including a "gray" category because images are only shown for 250ms of each 750ms cycle, so most frames show gray screen.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The image identity trace is built directly on the ophys timestamp grid within the trial window, using the same `frame_mask` as neural data. Each ophys frame is assigned the image visible at that time based on stimulus presentation start/stop times.

ii.
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
...
frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
trace[frame_mask] = img_idx
```

iii. Same ophys timestamp grid ensures alignment with neural data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in the stimulus presentations table. It marks single frames at stimulus change onsets.

ii.
```python
def build_image_change_trace(ophys_ts, stim_data, trial_start, trial_stop):
    ...
    is_change = stim_data['is_change']
    for si in range(len(stim_starts)):
        if not is_change[si]:
            continue
        s_start = stim_starts[si]
        frame_idx = np.searchsorted(trial_ts, s_start)
        if frame_idx < n_frames:
            trace[frame_idx] = 1
    return trace
```

iii. The AI uses the stimulus presentations `is_change` flag rather than the trials table `go` column.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary trace is created (all zeros), then for each stimulus presentation where `is_change` is True, the single ophys frame at or after the change onset is set to 1. Only one frame per change event is marked.

ii. See 4-a code snippet.

iii. No additional processing beyond the binary indicator.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is a binary variable (0 = no change, 1 = change). No thresholding is needed.

ii. Output values: `['no_change', 'change']`

iii. Per instructions: "binary variable."

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity - computed on the ophys timestamp grid within the same trial window using `np.searchsorted`.

ii. See 4-a code snippet.

iii. Same alignment approach as all other outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
data['running_speed'] = f['processing']['running']['speed']['data'][:]
```

iii. Standard running speed data from the NWB file.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps (~60 Hz) to the ophys timebase. Then it is discretized into 5 equal percentile bins computed PER SESSION (not globally across all sessions). NaN values are mapped to bin 0.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts)
...
running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)
...
running_binned = apply_percentile_bins(running_trial, running_edges, n_bins=5)
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 8: "Compute percentiles across the entire session (all valid timepoints), then apply per-trial."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Discretized into 5 bins using percentile edges computed per session. Edges are set to `-inf` and `+inf` at boundaries. `np.digitize` is used with clipping to [0, n_bins-1]. NaN values get bin 0.

ii.
```python
def compute_session_percentile_edges(values, n_bins=5):
    valid = values[~np.isnan(values)]
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges

def apply_percentile_bins(values, edges, n_bins=5):
    valid = ~np.isnan(values)
    result = np.zeros(len(values), dtype=np.int64)
    result[valid] = np.clip(np.digitize(values[valid], edges[1:-1]), 0, n_bins - 1)
    result[~valid] = 0
    return result
```

iii. Per instructions: "discretized into five equal percentile bins."

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the full session's ophys timestamps before trial segmentation, then extracted using the same boolean frame mask as neural data.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts)
...
running_trial = running_at_ophys[frame_mask]
```

iii. Same ophys timestamp alignment as neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `pupil_area` in the NWB file's `acquisition/EyeTracking/pupil_tracking/area` field. Diameter is computed from area via `2 * sqrt(area / pi)`. Blink frames (from `likely_blink`) are set to NaN before the conversion.

ii.
```python
pupil_area = nwb_data['pupil_area'].copy()
likely_blink = nwb_data['likely_blink']
pupil_area[likely_blink] = np.nan
pupil_diameter = np.full_like(pupil_area, np.nan)
valid_pupil = ~np.isnan(pupil_area) & (pupil_area > 0)
pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)
```

iii. CONVERSION_NOTES.md Step 5, Key Decision 6: "Pupil diameter: Compute from pupil area as 2*sqrt(area/pi). Set blink frames to NaN, then interpolate."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. (1) Blink frames set to NaN, (2) area converted to diameter via `2*sqrt(area/pi)`, (3) linearly interpolated to ophys timestamps, (4) discretized into 5 percentile bins per session. NaN values mapped to bin 0.

ii.
```python
pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
...
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)
```

iii. Same justification as running speed for percentile binning. The area-to-diameter conversion is documented in Step 5.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 equal percentile bins computed per session. NaN → bin 0.

ii. See 5-c code snippets (same `compute_session_percentile_edges` and `apply_percentile_bins` functions).

iii. Per instructions: "discretized into five equal percentile bins."

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed: interpolated to ophys timestamps, then extracted with same boolean frame mask as neural data.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
...
pupil_trial = pupil_at_ophys[frame_mask]
```

iii. Same alignment approach.

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

iii. Standard trial outcome categories from the change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome is determined by checking the boolean columns in priority order (hit, miss, false_alarm, correct_reject). The outcome code is broadcast to all time frames within the trial (static per-trial variable).

ii.
```python
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
...
output_full[4] = outcome_idx  # broadcast scalar to all timepoints
```

iii. The mapping order is `['hit', 'miss', 'false_alarm', 'correct_reject']`, stored as integers 0-3.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Failed experiments**: If `process_experiment` returns None (e.g., no cells, no stimulus data, or <2 valid trials), the experiment is skipped.
- **Short trials**: Trials with fewer than 2 ophys frames are skipped.
- **Missing eye tracking**: If no eye tracking data exists, pupil is set to all NaN (then bin 0).
- **Missing behavioral data**: NaN values from interpolation are mapped to bin 0.
- **Blinks**: Blink frames in pupil data are set to NaN before area-to-diameter conversion and interpolation.

ii.
```python
if result is None:
    continue
...
if n_trial_frames < 2:
    continue
...
if nwb_data['pupil_area'] is not None:
    ...
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
...
result[~valid] = 0  # NaN gets bin 0
```

iii. Documented in CONVERSION_NOTES.md Step 10, Check 5: "NaN pupil values (blinks) mapped to bin 0."

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading NWB files via h5py (~1.7s per session). Processing trials takes ~0.4s per session. Image name collection across all NWB files adds ~14s overhead at the start.

ii. N/A (timing stats from CONVERSION_NOTES.md Step 7)

iii. Total estimated time for full conversion: ~12.5 minutes for 202 experiments.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `build_image_identity_trace` function loops over all stimulus presentations for each trial. The `get_all_image_names` function opens every NWB file to collect image names. The trial processing loop is sequential per experiment.

ii.
```python
for si in range(len(stim_starts)):
    ...
    frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
    trace[frame_mask] = img_idx
```

iii. Not explicitly discussed in CONVERSION_NOTES.md. The per-stimulus loop could be vectorized using interval operations.

## 9-c. What processing does the code repeat multiple times?

i. The `get_all_image_names` function opens every NWB file to scan for image names before the main processing loop, then each NWB file is opened again during `process_experiment`. This means each NWB file is read twice.

ii.
```python
def get_all_image_names(exp_table):
    for _, row in exp_table.iterrows():
        eid = row['ophys_experiment_id']
        nwb_path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{eid}.nwb')
        with h5py.File(nwb_path, 'r') as f:
            ...
```

iii. Not explicitly discussed. The double-loading adds ~14s overhead.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores detailed session metadata (exp_id, session_type, cre_line, imaging_depth, etc.) in the `metadata['session_info']` list. The `plot_processing` function stores extra data (ophys_ts, running_at_ophys, pupil_at_ophys, etc.) when `show_processing` is enabled. The image identity trace includes a "gray" category which adds an extra category compared to just tracking the actual images.

ii. N/A

iii. Not explicitly discussed.
