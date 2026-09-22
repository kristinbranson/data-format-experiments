# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files using h5py, bypassing the Allen SDK's high-level API. It reads the experiment table from a CSV in the metadata directory, identifies downloaded NWB files by globbing the NWB directory, and loads each experiment's data fields directly from HDF5 paths.

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
    ...

def load_nwb_data(nwb_path):
    with h5py.File(nwb_path, 'r') as f:
        data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
        dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
        data['dff_traces'] = dff_raw.T
        ...
```

iii. The AI chose h5py for speed ("fast, no AllenSDK overhead") as documented in CONVERSION_NOTES.md Step 6. It reads the experiment table CSV and intersects with available NWB files to determine which experiments to process.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values from the experiment table. A subject mapping dictionary tracks mouse_id to index.

ii.
```python
mouse_id = result['mouse_id']
if mouse_id not in subject_map:
    subject_map[mouse_id] = len(all_subjects)
    all_subjects.append(mouse_id)
```

iii. The `mouse_id` field in the experiment table CSV uniquely identifies each animal. This is consistent with the SDK approach.

## 1-c. How are the data split into sessions?

i. The AI treats each experiment (each NWB file / imaging plane) as a separate "session" in the output. It iterates over experiments in the experiment table, processing each one independently.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    result = process_experiment(exp_id, row, image_names_list, outcome_names, ...)
    ...
    all_neural.append(result['neural'])
```

iii. CONVERSION_NOTES.md Step 5, Decision 9 states: "Multiscope handling: Each experiment (plane) is a separate 'session' in the output, since they have different neurons but share the same behavioral data." The AI explicitly chose to keep each imaging plane as a separate session rather than merging planes from the same ophys_session_id.

## 1-d. How are the data split into trials?

i. Trials are defined using start_time and stop_time from the trials interval in the NWB file. Valid trials are Go or Catch trials, excluding Aborted and Auto-rewarded. The trial window spans start_time to stop_time (variable length).

ii.
```python
def get_valid_trials(trial_data):
    go = trial_data['go'].astype(bool)
    catch = trial_data['catch'].astype(bool)
    aborted = trial_data['aborted'].astype(bool)
    auto_rewarded = trial_data['auto_rewarded'].astype(bool)
    valid = (go | catch) & ~aborted & ~auto_rewarded
    return np.where(valid)[0]

# In process_experiment:
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
```

iii. The AI follows the instruction to include Go and Catch trials while excluding Aborted and Auto-rewarded. The trial window uses the SDK's built-in start/stop times.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) must be Go or Catch, (2) must not be Aborted, (3) must not be Auto-rewarded, (4) must have at least 2 frames, (5) sessions with fewer than 2 valid trials are excluded. The AI does NOT explicitly filter on `change_time.notna()`.

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

iii. The AI filters using the Go/Catch flags combined with the aborted/auto-rewarded exclusions. It requires at least 2 frames per trial and 2 trials per session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the dF/F traces stored in the NWB file at `processing/ophys/dff/traces/data`.

ii.
```python
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
```

iii. dF/F is the standard measure of neural activity for two-photon calcium imaging. It is pre-computed in the NWB files.

## 2-b. How is the `neural` data processed?

i. The dF/F data is transposed from (n_frames, n_cells) to (n_cells, n_frames) and extracted per-trial as float32. No additional filtering, normalization, or processing is applied.

ii.
```python
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
...
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The AI relies on the pre-computed dF/F from the Allen pipeline. No additional processing is applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit neuron quality filtering is applied. All neurons present in the NWB file's dF/F traces are included. The AI does not filter by `valid_roi`.

ii. N/A - no filtering code.

iii. CONVERSION_NOTES.md Step 10, Check 3 states: "No explicit valid_roi filter... OK - all ROIs in downloaded NWB files are valid." The AI assumes all ROIs in the downloaded NWB files have already passed quality control.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's neural data is aligned to the trial start_time. Ophys frames from start_time to stop_time are extracted using a boolean mask on ophys timestamps.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The AI uses `>=` for start and `<` for stop, extracting all ophys frames within the trial window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. Data is kept at the native ophys frame rate. The time bin size is computed as the median inter-frame interval across sessions.

ii.
```python
dt = np.median(np.diff(ophys_ts))
...
all_dts = [m['dt_ms'] for m in session_metadata]
median_dt = np.median(all_dts)
```

iii. The ophys timestamps provide a consistent frame rate. No resampling is needed since behavioral data is interpolated to the same timebase.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table in the NWB file (`intervals/Natural_Images_*_presentations`), specifically the `image_name`, `start_time`, and `stop_time` fields. A "gray" label is used for inter-stimulus intervals.

ii.
```python
def build_image_identity_trace(ophys_ts, stim_data, trial_start, trial_stop, image_names_list):
    gray_idx = image_names_list.index(GRAY_LABEL)
    trace = np.full(n_frames, gray_idx, dtype=np.int64)
    for si in range(len(stim_starts)):
        ...
        frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
        trace[frame_mask] = img_idx
```

iii. The AI chose to use the stimulus presentations table to get frame-accurate image identity, including gray screen periods between flashes. This provides more temporal detail than using the trials table's initial/change image fields.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer indices. A sorted list of all unique image names (plus "gray") is built. The trace is initialized to "gray" and overwritten during stimulus presentation periods. Omitted stimuli are skipped (remain gray).

ii.
```python
image_names_list = [GRAY_LABEL] + all_image_names
...
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
```

iii. The AI builds a comprehensive time-varying trace that distinguishes between image presentations and gray screen periods. The global image list is sorted for deterministic ordering.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed for the same ophys frames as the neural data, using the same `frame_mask` (ophys timestamps between trial start and stop).

ii.
```python
img_trace, _ = build_image_identity_trace(ophys_ts, stim_data, t_start, t_stop, image_names_list)
```

iii. Both neural and image identity use the same ophys timestamp-based frame selection, ensuring alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in the stimulus presentations table, along with stimulus `start_time`.

ii.
```python
def build_image_change_trace(ophys_ts, stim_data, trial_start, trial_stop):
    stim_starts = stim_data['start_time']
    is_change = stim_data['is_change']
    for si in range(len(stim_starts)):
        if not is_change[si]:
            continue
        frame_idx = np.searchsorted(trial_ts, s_start)
        if frame_idx < n_frames:
            trace[frame_idx] = 1
```

iii. The AI uses the stimulus presentations' `is_change` flag to identify which stimulus is the change stimulus, then marks only the single frame at change onset.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary trace is constructed where 1 appears only at the single ophys frame at or immediately after the change stimulus onset. All other frames are 0.

ii.
```python
trace = np.zeros(n_frames, dtype=np.int64)
frame_idx = np.searchsorted(trial_ts, s_start)
if frame_idx < n_frames:
    trace[frame_idx] = 1
```

iii. The AI marks only the first frame at/after the change onset rather than a time window.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 = no change, 1 = change). No thresholding is needed.

ii.
```python
image_change_value_names = ['no_change', 'change']
```

iii. N/A - binary variable.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity - computed on the same ophys frames within the trial window.

ii. See 4-a code - uses same `ophys_ts` and trial bounds.

iii. Alignment is guaranteed by using the same ophys timestamp frame selection.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed` in the NWB file, which provides speed data and timestamps.

ii.
```python
data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
data['running_speed'] = f['processing']['running']['speed']['data'][:]
```

iii. This is the SDK's standard running speed data path in the NWB file.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the ophys timebase, then discretized into 5 percentile-based bins. Bin edges are computed **per session** (not globally).

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)
running_binned = apply_percentile_bins(running_trial, running_edges, n_bins=5)
```

iii. The AI chose per-session percentile computation rather than global. CONVERSION_NOTES.md Step 5 states: "Compute percentiles across the entire session (all valid timepoints), then apply per-trial."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using equal percentiles (0th, 20th, 40th, 60th, 80th, 100th) computed per session. Bin edges are set to -inf and +inf at the boundaries. NaN values are mapped to bin 0.

ii.
```python
def compute_session_percentile_edges(values, n_bins=5):
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges

def apply_percentile_bins(values, edges, n_bins=5):
    result[valid] = np.clip(np.digitize(values[valid], edges[1:-1]), 0, n_bins - 1)
    result[~valid] = 0
```

iii. Per-session percentile bins ensure roughly equal class counts within each session.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the full session's ophys timestamps, then extracted per-trial using the same boolean mask as neural data.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
running_trial = running_at_ophys[frame_mask]
```

iii. By interpolating to ophys timestamps first, alignment with neural data is guaranteed.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `pupil_area` (from `acquisition/EyeTracking/pupil_tracking/area`) and `likely_blink` (from `acquisition/EyeTracking/likely_blink`).

ii.
```python
data['pupil_area'] = pt['area']['data'][:] if 'data' in pt['area'] else pt['area'][:]
data['pupil_timestamps'] = pt['timestamps'][:]
data['likely_blink'] = et['likely_blink']['data'][:]
```

iii. The AI reads pupil area rather than pupil width from the NWB file.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter is computed from pupil area using the formula `diameter = 2 * sqrt(area / pi)`. Blink frames are set to NaN before the diameter computation. The result is then interpolated to ophys timestamps and discretized into 5 per-session percentile bins.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_diameter = np.full_like(pupil_area, np.nan)
valid_pupil = ~np.isnan(pupil_area) & (pupil_area > 0)
pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
```

iii. CONVERSION_NOTES.md Step 5 states: "Pupil diameter: Compute from pupil area as `2*sqrt(area/pi)`. Set blink frames to NaN, then interpolate."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 equal percentile bins computed per session, with NaN mapped to bin 0.

ii.
```python
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)
```

iii. Same approach as running speed discretization.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed - interpolated to ophys timestamps, then extracted per-trial using the same frame mask.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
pupil_trial = pupil_at_ophys[frame_mask]
```

iii. Alignment guaranteed by using the same ophys timestamp-based frame selection.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the `hit`, `miss`, `false_alarm`, and `correct_reject` fields in the trials interval of the NWB file.

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

iii. These four outcome fields are the canonical trial outcome labels for the change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes via a fixed ordered list. The outcome is constant across all timepoints in a trial (broadcast to fill the time dimension).

ii.
```python
outcome_names = ['hit', 'miss', 'false_alarm', 'correct_reject']
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
output_full[4] = outcome_idx  # broadcast scalar to all timepoints
```

iii. The mapping is deterministic and stored in the output_values field for recovery.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing NWB files**: Experiments without NWB files are skipped with a warning.
- **No cells**: Experiments with 0 cells are skipped.
- **Missing stimulus data**: Experiments without stimulus presentation data are skipped.
- **Missing eye tracking**: Pupil is set to all NaN if eye tracking is absent.
- **Blinks**: Pupil values during blinks are set to NaN.
- **NaN behavioral data**: NaN values in running speed or pupil are mapped to bin 0 during discretization.
- **Few trials**: Experiments with fewer than 2 valid trials are skipped.
- **Short trials**: Trials with fewer than 2 frames are skipped.

ii.
```python
if not os.path.exists(nwb_path):
    return None
if n_cells == 0:
    return None
if stim_data is None:
    return None
if nwb_data['pupil_area'] is not None:
    pupil_area[likely_blink] = np.nan
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
result[~valid] = 0  # NaN gets bin 0
if len(neural_trials) < 2:
    return None
```

iii. The AI handles various edge cases gracefully, skipping problematic data rather than crashing.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading NWB files via h5py (~1.7s per experiment) and collecting image names across all experiments (~14s overhead). The processing per trial is relatively fast (~0.4s per session).

ii. N/A

iii. CONVERSION_NOTES.md Step 7 documents: "Load NWB ~1.7s, Process trials ~0.4s, Image name collection adds ~14s overhead."

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `build_image_identity_trace` function loops over all stimulus presentations for each trial. The `get_all_image_names` function opens every NWB file sequentially to collect image names. The per-trial loop in `process_experiment` processes trials sequentially.

ii.
```python
for si in range(len(stim_starts)):  # loop in build_image_identity_trace
    ...
    frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
    trace[frame_mask] = img_idx
```

iii. The stimulus presentation loop could be vectorized using np.searchsorted on all start/stop times at once. The image name collection could be parallelized.

## 9-c. What processing does the code repeat multiple times?

i. The `get_all_image_names` function opens every NWB file to collect image names, and then each file is opened again during `process_experiment`. This means every NWB file is read twice: once for image names and once for processing.

ii.
```python
def get_all_image_names(exp_table):
    for _, row in exp_table.iterrows():
        with h5py.File(nwb_path, 'r') as f:
            ...  # reads each NWB file once

# Then in main loop:
for idx, (_, row) in enumerate(exp_table.iterrows()):
    result = process_experiment(...)  # reads each NWB file again
```

iii. The double-read of NWB files adds unnecessary I/O overhead.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes per-session percentile bin edges for running speed and pupil diameter. Since each session has its own edges, the bin assignments are not comparable across sessions, which may affect decoder performance. Additionally, the stimulus presentations table is parsed for image identity when the trials table already contains initial/change image info.

ii.
```python
running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
```

iii. Per-session bin edges mean bin 0 in one session may correspond to a different speed range than bin 0 in another session, potentially confusing the decoder.
