# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files using `h5py`, bypassing the AllenSDK high-level API. It reads an experiment table CSV from the metadata directory, filters to experiments that have downloaded NWB files, and processes each experiment individually by reading neural (dF/F), running speed, pupil tracking, trial, and stimulus presentation data from the NWB file structure.

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
        # ... running speed, pupil, trials, stimulus presentations
    return data
```

iii. The AI chose h5py for speed, avoiding AllenSDK overhead. It filters to downloaded NWB files and excludes passive sessions by checking `session_type` for "passive". The CONVERSION_NOTES.md documents this as "fast, no AllenSDK overhead."

## 1-b. How are the data split into subjects?

i. Subjects correspond to unique `mouse_id` values in the experiment table, mapped to indices via a dictionary.

ii.
```python
subject_map = {}  # mouse_id -> index
# ...
mouse_id = result['mouse_id']
if mouse_id not in subject_map:
    subject_map[mouse_id] = len(all_subjects)
    all_subjects.append(mouse_id)
```

iii. Each experiment's mouse_id is extracted from the experiment table row. Subjects are accumulated as new mouse IDs are encountered during processing.

## 1-c. How are the data split into sessions?

i. The AI treats each individual experiment (each imaging plane / NWB file) as a separate "session" in the output. It does NOT group multiple experiments that share the same `ophys_session_id` into a single session. Each experiment is processed independently.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    result = process_experiment(exp_id, row, image_names_list, outcome_names, ...)
    # ...
    all_neural.append(result['neural'])
    all_output.append(result['output'])
```

iii. The CONVERSION_NOTES.md states: "Each experiment (plane) is a separate 'session' in the output, since they have different neurons but share the same behavioral data." The AI made a deliberate decision to keep experiments separate rather than merging planes from the same session.

## 1-d. How are the data split into trials?

i. Trials are defined using the trials interval table in the NWB file. Valid trials are those that are Go or Catch, not aborted, and not auto-rewarded. Each trial spans from `start_time` to `stop_time` using boolean masking on ophys timestamps.

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

iii. The AI explicitly requires `go | catch` in addition to excluding aborted and auto-rewarded. The trial window uses `>=` for start and `<` for stop.

## 1-e. How are trials filtered based on quality controls?

i. Trials must be Go or Catch, not aborted, not auto-rewarded. Trials with fewer than 2 frames are skipped. Sessions with fewer than 2 valid trials are excluded. Passive sessions are excluded at the experiment table level.

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

iii. Filtering follows the instructions to include Go and Catch while excluding Aborted and Auto-rewarded. The 2-frame minimum prevents degenerate trials. Sessions need >=2 trials for decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `processing/ophys/dff/traces/data` in the NWB file, which contains pre-computed dF/F calcium fluorescence traces.

ii.
```python
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
```

iii. dF/F is pre-computed in the NWB files with neuropil correction and baseline normalization already applied. The AI transposes from (n_frames, n_cells) to (n_cells, n_frames).

## 2-b. How is the `neural` data processed?

i. No additional processing is applied beyond transposing the array and casting to float32 when extracting per-trial windows. Each experiment's neurons are kept as-is.

ii.
```python
neural = dff[:, frame_mask].astype(np.float32)  # (n_cells, n_trial_frames)
```

iii. The dF/F traces are already processed by the Allen SDK pipeline. No additional filtering, normalization, or smoothing is applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied. All neurons present in the NWB file's dF/F traces are included. The AI notes that "all ROIs in downloaded NWB files are valid" but does not explicitly check `valid_roi`.

ii. N/A (no filtering code)

iii. CONVERSION_NOTES.md states: "No explicit valid_roi filter" and "all ROIs in downloaded NWB files are valid." The AI relies on the assumption that the NWB files only contain valid ROIs.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to ophys timestamps. For each trial, frames between `start_time` and `stop_time` are extracted using a boolean mask on ophys timestamps (`>= start` and `< stop`).

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The alignment uses the ophys timestamps as the common timebase, as specified in the instructions ("Temporally align based on ophys timestamp").

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The data is kept at the native ophys frame rate. The time bin size is computed as the median of `np.diff(ophys_ts)` across sessions, reported as ~32.3 ms (~31 Hz).

ii.
```python
dt = np.median(np.diff(ophys_ts))  # time bin size
# ...
all_dts = [m['dt_ms'] for m in session_metadata]
median_dt = np.median(all_dts)
```

iii. The ophys timestamps provide a consistent frame rate. The AI uses the median dt across all sessions for the metadata `time_bin_size` field.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the **stimulus presentations table** (`intervals/Natural_Images_*_presentations`) in the NWB file, using `image_name`, `start_time`, and `stop_time` for each stimulus presentation. This provides frame-by-frame image identity including gray screen periods during the inter-stimulus interval (ISI).

ii.
```python
stim_starts = stim_data['start_time']
stim_stops = stim_data['stop_time']
stim_names = stim_data['image_name']

for si in range(len(stim_starts)):
    name = stim_names[si]
    if name == 'omitted':
        continue
    frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
    trace[frame_mask] = img_idx
```

iii. The AI chose the stimulus presentations table to get precise timing of when each image is on screen vs. gray screen. This is more granular than using the trial-level `initial_image_name`/`change_image_name`.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes. A "gray" category is included as index 0 for inter-stimulus intervals. The trace is initialized to "gray" and filled in with image indices during stimulus presentation windows. Omitted stimuli are treated as gray screen.

ii.
```python
GRAY_LABEL = 'gray'
image_names_list = [GRAY_LABEL] + all_image_names
# ...
gray_idx = image_names_list.index(GRAY_LABEL)
trace = np.full(n_frames, gray_idx, dtype=np.int64)
# Map stimulus presentations onto ophys frames
for si in range(len(stim_starts)):
    if name == 'omitted':
        continue
    if name in image_names_list:
        img_idx = image_names_list.index(name)
    frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
    trace[frame_mask] = img_idx
```

iii. The AI's approach assigns "gray" during the 500ms ISI periods, resulting in a total of 9+ categories (gray + 8 images). The reference instead uses only the 8 image names from `initial_image_name` and `change_image_name`, with no gray category.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed directly on the ophys timestamps within the trial window using the same boolean mask as neural data, ensuring frame-level alignment.

ii.
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
# trace computed on trial_ts, same as neural frame_mask
```

iii. Both neural and image identity use the same ophys timestamp mask for the trial window.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in the **stimulus presentations table**, not from the trial-level `change_time` or `go` flag.

ii.
```python
stim_starts = stim_data['start_time']
is_change = stim_data['is_change']

for si in range(len(stim_starts)):
    if not is_change[si]:
        continue
    frame_idx = np.searchsorted(trial_ts, s_start)
    if frame_idx < n_frames:
        trace[frame_idx] = 1
```

iii. The AI uses the stimulus-level `is_change` flag from the presentations table rather than trial-level `change_time` combined with the `go` flag.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary trace is created, initialized to 0. For each stimulus presentation with `is_change == True`, the single ophys frame at or after the change onset is set to 1.

ii. Same as 4-a above.

iii. Only a single frame is marked as 1 at the change onset, not a window of time. This differs from the reference which marks a 750ms window.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1). No thresholding is needed.

ii.
```python
trace = np.zeros(n_frames, dtype=np.int64)
# ...
trace[frame_idx] = 1
```

iii. N/A - already binary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same alignment as image identity - computed on the same ophys timestamps within the trial window.

ii.
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
frame_idx = np.searchsorted(trial_ts, s_start)
```

iii. Uses the same trial timestamp array as neural data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
data['running_speed'] = f['processing']['running']['speed']['data'][:]
```

iii. The running speed data from the NWB file is sampled at 60 Hz and needs to be interpolated to ophys timestamps.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native 60 Hz to the ophys timebase, then discretized into 5 percentile-based bins. Bin edges are computed **per-session** from all valid (non-NaN) timepoints within that session.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)
running_binned = apply_percentile_bins(running_trial, running_edges, n_bins=5)
```

iii. The AI computes percentile bins per-session rather than globally across all sessions. CONVERSION_NOTES.md states: "Compute percentiles across the entire session (all valid timepoints), then apply per-trial."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using percentile-based bin edges (0th, 20th, 40th, 60th, 80th, 100th percentiles). The first and last edges are set to -inf and +inf respectively. NaN values are mapped to bin 0.

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
    result = np.zeros(len(values), dtype=np.int64)
    result[valid] = np.clip(np.digitize(values[valid], edges[1:-1]), 0, n_bins - 1)
    result[~valid] = 0
    return result
```

iii. 5 equal percentile bins as specified in the instructions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the full session's ophys timestamps, then extracted per-trial using the same boolean mask as neural data.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
# Per trial:
running_trial = running_at_ophys[frame_mask]
```

iii. Same alignment approach as neural data - both indexed by the same ophys frame mask.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/area` (pupil area) and `acquisition/EyeTracking/likely_blink` in the NWB file. Diameter is computed from area.

ii.
```python
data['pupil_area'] = pt['area']['data'][:] if 'data' in pt['area'] else pt['area'][:]
data['pupil_timestamps'] = pt['timestamps'][:]
data['likely_blink'] = et['likely_blink']['data'][:]
```

iii. The AI reads pupil area rather than pupil width, then converts to diameter via a geometric formula.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames are set to NaN, then diameter is computed from area as `2 * sqrt(area / pi)`. The result is interpolated to ophys timestamps and discretized into 5 per-session percentile bins. NaN values are mapped to bin 0.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_diameter = np.full_like(pupil_area, np.nan)
valid_pupil = ~np.isnan(pupil_area) & (pupil_area > 0)
pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
```

iii. The AI converts area to diameter before interpolation, reasoning that this matches the whitepaper's description. CONVERSION_NOTES.md: "Compute from pupil area as 2*sqrt(area/pi). Set blink frames to NaN, then interpolate."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 percentile-based bins computed per-session, NaN mapped to bin 0.

ii. Same as 5-c (uses `compute_session_percentile_edges` and `apply_percentile_bins`).

iii. Same as running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the full session's ophys timestamps, then extracted per-trial using the same boolean mask as neural data.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
# Per trial:
pupil_trial = pupil_at_ophys[frame_mask]
```

iii. Same alignment as running speed and neural data.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials interval table of the NWB file.

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

iii. These four outcomes are the canonical trial outcomes for the change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes via the `outcome_names` list ordering. The outcome is constant (broadcast) across all timepoints within a trial.

ii.
```python
outcome_names = ['hit', 'miss', 'false_alarm', 'correct_reject']
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
output_full[4] = outcome_idx  # broadcast scalar to all timepoints
```

iii. The static per-trial value is broadcast to match the time-varying output shape (5, n_frames).

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing NWB files**: Experiments without downloaded NWB files are excluded.
- **Missing stimulus data**: Sessions without stimulus presentations are skipped.
- **Missing eye tracking**: If no eye tracking data, pupil values are all NaN (mapped to bin 0).
- **Blinks**: Set to NaN before interpolation, mapped to bin 0 in discretization.
- **Too few trials**: Sessions with <2 valid trials are skipped.
- **Too few frames**: Trials with <2 ophys frames are skipped.
- **NaN in interpolation**: Extrapolated values become NaN, mapped to bin 0.

ii.
```python
if stim_data is None:
    return None
if nwb_data['pupil_area'] is not None:
    # process pupil
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
if len(neural_trials) < 2:
    return None
if n_trial_frames < 2:
    continue
```

iii. The AI handles missing data by skipping or filling with safe defaults rather than crashing.

## 9-a. What are the most time-consuming steps of the code?

i. Loading NWB files via h5py is the most time-consuming step (~1.7s per session). The AI's CONVERSION_NOTES.md estimates ~340s total for NWB loading vs ~80s for trial processing.

ii. N/A

iii. NWB file I/O dominates runtime. The AI chose h5py over AllenSDK for faster loading.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `build_image_identity_trace` function loops over all stimulus presentations to map them to ophys frames. This could be vectorized using `np.searchsorted` on all stimulus start/stop times at once. The `get_all_image_names` function loops over all NWB files to collect image names.

ii.
```python
for si in range(len(stim_starts)):
    s_start = stim_starts[si]
    s_stop = stim_stops[si]
    # ...
    frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
    trace[frame_mask] = img_idx
```

iii. The stimulus loop iterates over hundreds of presentations per trial. Vectorization could speed this up, though NWB loading is the bottleneck.

## 9-c. What processing does the code repeat multiple times?

i. The `get_all_image_names` function opens every NWB file just to read image names before any processing begins. Then each NWB file is opened again during `process_experiment`. This means each NWB file is read twice.

ii.
```python
def get_all_image_names(exp_table):
    for _, row in exp_table.iterrows():
        with h5py.File(nwb_path, 'r') as f:
            # read image names
```

iii. The image name collection pass adds ~14s overhead per CONVERSION_NOTES.md.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI includes a "gray" category in image identity that represents inter-stimulus intervals. This adds an extra category that may not be necessary for downstream decoding. Additionally, for Multiscope sessions, behavioral data (running, pupil, trials) is loaded multiple times for each plane since each plane is treated as a separate session.

ii.
```python
image_names_list = [GRAY_LABEL] + all_image_names  # adds gray category
```

iii. The gray screen category increases the number of output classes for image identity decoding.
