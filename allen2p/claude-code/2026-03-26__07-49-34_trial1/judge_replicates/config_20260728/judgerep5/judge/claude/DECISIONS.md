# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files using h5py, bypassing the AllenSDK API. It reads the experiment table from a CSV in the metadata directory, filters to experiments with downloaded NWB files, and then loads each NWB file individually. It also filters out passive sessions by checking if the session_type contains "passive".

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

def load_nwb_data(nwb_path):
    data = {}
    with h5py.File(nwb_path, 'r') as f:
        data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
        dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
        data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
        # ... loads running, eye tracking, trials, stimulus
    return data
```

iii. The AI chose h5py over the AllenSDK to avoid SDK overhead and for faster loading. The experiment table CSV provides metadata, while the NWB files contain the actual neural and behavioral data.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values in the experiment table. Each unique mouse_id is mapped to an index in the subjects list.

ii.
```python
subject_map = {}
# ...
mouse_id = result['mouse_id']
if mouse_id not in subject_map:
    subject_map[mouse_id] = len(all_subjects)
    all_subjects.append(mouse_id)
```

iii. The mouse_id field from the experiment table uniquely identifies each animal.

## 1-c. How are the data split into sessions?

i. Each experiment (single imaging plane) is treated as a separate "session" in the output. The AI does NOT group experiments by `ophys_session_id`. For multi-plane (Multiscope) recordings, each plane becomes its own session with its own neural data but shared behavioral data.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    result = process_experiment(exp_id, row, image_names_list, outcome_names, ...)
    # Each experiment becomes a session
    all_neural.append(result['neural'])
    all_output.append(result['output'])
```

iii. From CONVERSION_NOTES.md: "Multiscope handling: Each experiment (plane) is a separate 'session' in the output, since they have different neurons but share the same behavioral data."

## 1-d. How are the data split into trials?

i. Trials are defined using the trials table from the NWB file. Valid trials (Go + Catch, excluding Aborted and Auto-rewarded) are identified, and for each, ophys frames from `start_time` to `stop_time` are extracted, yielding variable-length trials.

ii.
```python
def get_valid_trials(trial_data):
    go = trial_data['go'].astype(bool)
    catch = trial_data['catch'].astype(bool)
    aborted = trial_data['aborted'].astype(bool)
    auto_rewarded = trial_data['auto_rewarded'].astype(bool)
    valid = (go | catch) & ~aborted & ~auto_rewarded
    return np.where(valid)[0]

# Per trial:
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
```

iii. The trials table provides pre-computed trial metadata. Go and Catch trials are included per instructions. The full trial window (start_time to stop_time) is used.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to include only Go and Catch trials, explicitly excluding Aborted and Auto-rewarded. Trials with fewer than 2 frames are skipped. Sessions with fewer than 2 valid trials are excluded.

ii.
```python
valid = (go | catch) & ~aborted & ~auto_rewarded
# ...
if n_trial_frames < 2:
    continue
# ...
if len(neural_trials) < 2:
    return None
```

iii. Per instruction: "Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the dF/F traces stored in the NWB files at `processing/ophys/dff/traces/data`.

ii.
```python
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
```

iii. dF/F is the standard calcium imaging signal, pre-computed by the Allen pipeline.

## 2-b. How is the `neural` data processed?

i. No additional processing is applied. The dF/F traces are read directly and transposed to (n_cells, n_frames) format. Per-trial slices are extracted and cast to float32.

ii.
```python
neural = dff[:, frame_mask].astype(np.float32)  # (n_cells, n_trial_frames)
```

iii. The dF/F is pre-processed by the Allen pipeline (motion correction, neuropil subtraction, baseline normalization).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit neuron quality filtering is applied. All neurons present in the NWB file's dF/F traces are included.

ii. N/A (no filtering code)

iii. From CONVERSION_NOTES.md: "No explicit valid_roi filter... all ROIs in downloaded NWB files are valid." The AI notes that the SDK default filters valid_roi=True but does not verify this when loading directly with h5py.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (start_time from the trials table). Ophys frames between start_time (inclusive) and stop_time (exclusive) are extracted using a boolean mask.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The ophys timestamps provide the temporal reference. The trial window from start_time to stop_time captures both pre-change stimulus flashes and the post-change response window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native ophys frame rate is used (~31 Hz for Scientifica, ~11 Hz for Multiscope). No rebinning is applied. The time bin size is computed as the median inter-frame interval.

ii.
```python
dt = np.median(np.diff(ophys_ts))
# ...
median_dt = np.median(all_dts)
```

iii. The ophys timestamps are already at a consistent frame rate from the microscope.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table in the NWB file (`intervals/Natural_Images_*_presentations`), specifically the `image_name`, `start_time`, and `stop_time` fields. A "gray" label is used for inter-stimulus intervals.

ii.
```python
def build_image_identity_trace(ophys_ts, stim_data, trial_start, trial_stop, image_names_list):
    gray_idx = image_names_list.index(GRAY_LABEL)
    trace = np.full(n_frames, gray_idx, dtype=np.int64)
    for si in range(len(stim_starts)):
        name = stim_names[si]
        if name == 'omitted':
            continue
        img_idx = image_names_list.index(name)
        frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
        trace[frame_mask] = img_idx
```

iii. Using the stimulus presentations table allows precise tracking of which image is on screen at each ophys frame, including gray screen periods during ISI.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each ophys frame in the trial, the image on screen is determined by checking overlap with stimulus presentation windows. Frames not covered by any stimulus presentation are labeled "gray". Image names are mapped to integer indices via a global sorted list that includes "gray" as the first entry.

ii.
```python
image_names_list = [GRAY_LABEL] + all_image_names
# ...
gray_idx = image_names_list.index(GRAY_LABEL)
trace = np.full(n_frames, gray_idx, dtype=np.int64)
for si in range(len(stim_starts)):
    # ... map stimulus frames to image index
    trace[frame_mask] = img_idx
```

iii. The global image name list ensures consistent encoding across sessions. Omitted stimuli are treated as gray screen.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at each ophys frame within the trial, using the same frame_mask as neural data. Both share the same temporal indices.

ii.
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
# Same mask used for neural data and image identity trace
```

iii. Using ophys timestamps as the common timebase ensures alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in the stimulus presentations table, which marks which stimulus presentation was the change stimulus.

ii.
```python
def build_image_change_trace(ophys_ts, stim_data, trial_start, trial_stop):
    is_change = stim_data['is_change']
    for si in range(len(stim_starts)):
        if not is_change[si]:
            continue
        frame_idx = np.searchsorted(trial_ts, s_start)
        if frame_idx < n_frames:
            trace[frame_idx] = 1
```

iii. The `is_change` field from the stimulus presentations table identifies change events.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary trace is created that is 1 only at the single first ophys frame at or after the change stimulus onset. All other frames are 0.

ii.
```python
trace = np.zeros(n_frames, dtype=np.int64)
# For each change stimulus:
frame_idx = np.searchsorted(trial_ts, s_start)
if frame_idx < n_frames:
    trace[frame_idx] = 1
```

iii. Only the onset frame is marked as 1, creating a very sparse signal.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is already binary (0 or 1). No thresholding is applied. The value is 1 at the single change onset frame, 0 elsewhere.

ii.
```python
image_change_value_names = ['no_change', 'change']
```

iii. The instructions specify "binary variable. Have value of 1 right after a change in image identity, otherwise 0."

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change uses the same trial timestamp array as neural data, ensuring alignment.

ii.
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
# Same mask for both neural and change trace
```

iii. Same ophys timebase alignment as other variables.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed` in the NWB file, which provides speed values and timestamps.

ii.
```python
data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
data['running_speed'] = f['processing']['running']['speed']['data'][:]
```

iii. The Allen pipeline provides pre-processed running speed from the running wheel encoder.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps (~60 Hz) to the ophys timebase, then discretized into 5 percentile bins computed per-session.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)
running_binned = apply_percentile_bins(running_trial, running_edges, n_bins=5)
```

iii. Linear interpolation resamples to ophys timebase. Percentile binning is done per-session.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using percentile-based edges computed from each session's data (excluding NaN). Bin edges are set to -inf and +inf at boundaries. NaN values are mapped to bin 0.

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
    result = np.clip(np.digitize(values[valid], edges[1:-1]), 0, n_bins - 1)
    result[~valid] = 0
    return result
```

iii. Per-session percentile bins ensure roughly equal class counts within each session.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to ophys timestamps for the entire session before per-trial extraction. The same frame_mask is used for neural and running data.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
# Per trial:
running_trial = running_at_ophys[frame_mask]
```

iii. Interpolation to ophys timebase before trial extraction ensures alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `pupil_area` in the eye tracking data (`acquisition/EyeTracking/pupil_tracking/area`), with blinks identified from `likely_blink`. The diameter is computed as `2 * sqrt(area / pi)`.

ii.
```python
data['pupil_area'] = pt['area']['data'][:] if 'data' in pt['area'] else pt['area'][:]
data['likely_blink'] = et['likely_blink']['data'][:]
# ...
pupil_area[likely_blink] = np.nan
pupil_diameter = np.full_like(pupil_area, np.nan)
valid_pupil = ~np.isnan(pupil_area) & (pupil_area > 0)
pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)
```

iii. From CONVERSION_NOTES.md: "Compute from pupil area as 2*sqrt(area/pi). Set blink frames to NaN, then interpolate."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames are set to NaN, then diameter is computed from area assuming a circular pupil. The resulting signal is linearly interpolated to ophys timestamps, then discretized into 5 per-session percentile bins.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)
```

iii. Blinks are removed before interpolation to avoid contamination. Per-session discretization is applied.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 percentile bins computed per-session, NaN mapped to bin 0.

ii. Same percentile binning functions as running speed.

iii. Per-session bins ensure consistent class distribution within each session.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed: interpolated to ophys timestamps, then extracted using the same frame_mask.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
pupil_trial = pupil_at_ophys[frame_mask]
```

iii. Same alignment approach as running speed.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the `hit`, `miss`, `false_alarm`, and `correct_reject` columns in the trials table.

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

iii. These four outcome columns are the canonical trial outcome labels for the change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome is mapped to an integer index (0-3) based on position in the outcome_names list. Unknown outcomes get index 0. The integer is broadcast to all timepoints in the trial.

ii.
```python
outcome_names = ['hit', 'miss', 'false_alarm', 'correct_reject']
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
output_full[4] = outcome_idx  # broadcast scalar to all timepoints
```

iii. Static per-trial output is replicated across timepoints as required by the (n_output, n_timepoints) format.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing NWB files**: Experiments without NWB files are skipped.
- **No cells**: Experiments with 0 cells are skipped.
- **No stimulus data**: Experiments without stimulus presentation data are skipped.
- **Few trials**: Experiments with < 2 valid trials are skipped.
- **Short trials**: Trials with < 2 ophys frames are skipped.
- **Missing pupil data**: If no eye tracking is available, pupil is set to NaN for all frames.
- **Blinks**: Set to NaN before interpolation. NaN values mapped to bin 0 in discretization.

ii.
```python
if n_cells == 0:
    return None
if stim_data is None:
    return None
if len(valid_trial_idx) < 2:
    return None
if n_trial_frames < 2:
    continue
if nwb_data['pupil_area'] is None:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
result[~valid] = 0  # NaN -> bin 0
```

iii. The code handles each case gracefully, skipping problematic data rather than crashing.

## 9-a. What are the most time-consuming steps of the code?

i. Loading each NWB file via h5py is the most time-consuming step. From CONVERSION_NOTES.md: load time ~1.7s per session, process time ~0.4s. Image name collection across all NWB files adds ~14s overhead.

ii. N/A

iii. The NWB files contain large arrays that must be read from disk.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `build_image_identity_trace` function loops over all stimulus presentations to map each onto ophys frames. This could potentially be vectorized using np.searchsorted for all presentations at once. The `get_all_image_names` function opens every NWB file to scan image names, which could be done during the main processing loop instead.

ii.
```python
for si in range(len(stim_starts)):
    # ... loop over stimulus presentations
    frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
    trace[frame_mask] = img_idx
```

iii. The stimulus presentation loop is relatively short (~50 presentations per trial window) so the impact is small.

## 9-c. What processing does the code repeat multiple times?

i. The `get_all_image_names` function opens every NWB file to collect image names before the main processing loop, then the main processing loop opens each NWB file again. This means every NWB file is opened twice.

ii.
```python
# First pass: collect image names
all_image_names = get_all_image_names(exp_table)

# Second pass: process each experiment
for idx, (_, row) in enumerate(exp_table.iterrows()):
    result = process_experiment(exp_id, row, ...)
```

iii. The double-open could be avoided by collecting image names during the main processing loop.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores detailed session metadata (exp_id, session_type, cre_line, imaging_depth, etc.) in the output that is not used by the decoder. The processing plots (when enabled) are visualization-only. The image name collection pass is repeated work.

ii.
```python
session_metadata.append({
    'exp_id': result['exp_id'],
    'ophys_session_id': result['ophys_session_id'],
    'session_type': result['session_type'],
    'cre_line': result['cre_line'],
    'imaging_depth': result['imaging_depth'],
    # ...
})
```

iii. The extra metadata is useful for documentation but not needed for decoder training.
