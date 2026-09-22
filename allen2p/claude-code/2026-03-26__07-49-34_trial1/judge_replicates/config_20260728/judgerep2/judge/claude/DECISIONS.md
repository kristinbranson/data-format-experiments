# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files using h5py, rather than the AllenSDK. It reads an experiment table CSV from the metadata directory, filters to experiments that have downloaded NWB files and are active (non-passive) sessions, then loads each NWB file individually to extract neural, behavioral, and trial data.

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
        # ... running, pupil, trials, stimulus data
    return data
```

iii. The AI chose h5py for speed, bypassing AllenSDK overhead. It noted in CONVERSION_NOTES.md that all relevant data fields are accessible directly from NWB. The experiment table CSV was used as the canonical listing of experiments.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values in the experiment table. A `subject_map` dictionary maps mouse IDs to sequential indices.

ii.
```python
subject_map = {}
# ...
mouse_id = result['mouse_id']
if mouse_id not in subject_map:
    subject_map[mouse_id] = len(all_subjects)
    all_subjects.append(mouse_id)
```

iii. The `mouse_id` field is the standard identifier for each animal in the Allen dataset.

## 1-c. How are the data split into sessions?

i. Each experiment (single imaging plane / NWB file) is treated as a separate session. The AI does NOT group multiple experiments from the same ophys session together. Each row in the experiment table becomes one session in the output.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    result = process_experiment(exp_id, row, image_names_list, outcome_names, ...)
    # ... each result becomes one session
    all_neural.append(result['neural'])
```

iii. The AI noted in CONVERSION_NOTES.md (Key Decision 9): "Multiscope handling: Each experiment (plane) is a separate 'session' in the output, since they have different neurons but share the same behavioral data."

## 1-d. How are the data split into trials?

i. Trials are defined using the trials table from the NWB file. Valid trials are Go or Catch trials that are not aborted and not auto-rewarded. For each valid trial, ophys frames between `start_time` and `stop_time` are extracted, giving variable-length trials.

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

iii. The instructions specify to include Go and Catch trials and exclude Aborted and Auto-rewarded trials. The AI uses `start_time` to `stop_time` from the trials table for variable-length trial windows.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to Go+Catch, excluding Aborted and Auto-rewarded. Trials with fewer than 2 frames are skipped. Sessions with fewer than 2 valid trials after processing are excluded.

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

iii. This matches the instructions. The minimum 2-trial threshold ensures the decoder has enough data per session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dff_traces` (dF/F calcium fluorescence traces) stored in the NWB file at `processing/ophys/dff/traces/data`.

ii.
```python
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
```

iii. dF/F is the standard measure of neural activity for two-photon calcium imaging. The Allen SDK provides it pre-computed with neuropil correction and baseline normalization.

## 2-b. How is the `neural` data processed?

i. No additional processing is applied. The pre-computed dF/F traces are used directly, transposed to (n_cells, n_frames), and converted to float32 for each trial.

ii.
```python
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
# Per trial:
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The dF/F traces are already processed by the Allen SDK pipeline (motion correction, neuropil subtraction, dF/F normalization).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied to the neural data. All neurons present in the NWB file's dff_traces are included. The AI noted in CONVERSION_NOTES.md that "all ROIs in downloaded NWB files are valid" but did not explicitly verify this by checking `valid_roi` flags.

ii. No filtering code is present.

iii. The AI assumed the NWB files only contain valid ROIs. The Allen SDK by default uses `exclude_invalid_rois=True`, but since the AI reads NWB files directly via h5py, this default filtering may not apply.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's neural data is extracted by finding ophys frames between `start_time` and `stop_time` using a boolean mask. Alignment is to trial start (the first ophys frame at or after `start_time`).

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The instructions say to align based on ophys timestamps. The AI uses `>=` for start and `<` for stop, which is a standard half-open interval approach.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data is kept at the native ophys frame rate (~31 Hz for Scientifica, ~11 Hz for Multiscope). No temporal rebinning is applied. The time bin size is computed as the median inter-frame interval across sessions.

ii.
```python
dt = np.median(np.diff(ophys_ts))  # time bin size
# ...
all_dts = [m['dt_ms'] for m in session_metadata]
median_dt = np.median(all_dts)
```

iii. The ophys timestamps provide a consistent frame rate. No resampling is needed since all data streams are aligned to the same ophys timebase.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the **stimulus presentations table** in the NWB file (at `intervals/Natural_Images_*_presentations`), using `start_time`, `stop_time`, and `image_name` fields. A "gray" category is used for inter-stimulus intervals.

ii.
```python
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
```

iii. The AI used the stimulus presentations table to build a frame-by-frame trace of which image is on screen, distinguishing between image presentation periods and gray inter-stimulus intervals. Omitted stimuli are treated as gray screen.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global mapping built from all unique image names across all experiments. The mapping includes a "gray" category at index 0. The integer code varies within a trial based on when images are presented vs gray screens.

ii.
```python
all_image_names = get_all_image_names(exp_table)
image_names_list = [GRAY_LABEL] + all_image_names
# In build_image_identity_trace:
trace = np.full(n_frames, gray_idx, dtype=np.int64)
# Then fills in image periods from stimulus presentations
```

iii. The AI chose to include gray screen as a category, reasoning that during ISI (500ms of every 750ms cycle), no image is displayed. This results in 17 categories (gray + 16 unique images across image sets) rather than just 8 image categories.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed per ophys frame within the trial window. Each frame's identity is determined by checking which stimulus presentation (if any) is active at that time. Both neural and image identity use the same `frame_mask` based on ophys timestamps.

ii.
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
# Stimulus times mapped onto these same frames
frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
trace[frame_mask] = img_idx
```

iii. Alignment is guaranteed by using the same ophys timestamp array for both neural and image identity extraction.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in the **stimulus presentations table** in the NWB file, combined with `start_time` of each stimulus.

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

iii. The AI uses the stimulus presentations `is_change` flag rather than the trials table `go` field.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary trace is created where the value is 1 only at the **single frame** corresponding to the change onset. All other frames are 0.

ii.
```python
trace = np.zeros(n_frames, dtype=np.int64)
# Only sets 1 at the single frame at change onset:
frame_idx = np.searchsorted(trial_ts, s_start)
if frame_idx < n_frames:
    trace[frame_idx] = 1
```

iii. The AI marks only a single frame as "change" at the onset of the change stimulus.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1). No thresholding is needed. Categories are `['no_change', 'change']`.

ii.
```python
output_values = [
    # ...
    ['no_change', 'change'],   # image change: 0=no change, 1=change
    # ...
]
```

iii. Binary variable, no additional thresholding needed.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity — computed per ophys frame using the same trial mask. The change onset is found via `np.searchsorted` on the trial's ophys timestamps.

ii.
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
frame_idx = np.searchsorted(trial_ts, s_start)
```

iii. Same frame-level alignment as all other variables.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed` in the NWB file, which provides speed values and timestamps.

ii.
```python
data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
data['running_speed'] = f['processing']['running']['speed']['data'][:]
```

iii. This is the standard running speed data from the Allen dataset.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps (~60 Hz) to the ophys timebase, then discretized into 5 percentile-based bins computed **per session**.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)
running_binned = apply_percentile_bins(running_trial, running_edges, n_bins=5)
```

iii. Linear interpolation preserves the signal shape. Percentile-based binning ensures roughly equal class counts. Bin edges are computed **per session** rather than globally.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal percentile bins (0-20th, 20-40th, ..., 80-100th percentile) computed per session. NaN values are mapped to bin 0.

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

iii. Percentile-based ensures roughly equal class counts within each session.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to ophys timestamps for the entire session before trial segmentation, then the trial's frames are extracted using the same boolean mask as neural data.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
# Per trial:
running_trial = running_at_ophys[frame_mask]
```

iii. By interpolating upfront to ophys timestamps, alignment is guaranteed.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `pupil_area` in the eye tracking data at `acquisition/EyeTracking/pupil_tracking/area` in the NWB file. Blink frames are identified using `likely_blink`.

ii.
```python
data['pupil_area'] = pt['area']['data'][:] if 'data' in pt['area'] else pt['area'][:]
data['pupil_timestamps'] = pt['timestamps'][:]
data['likely_blink'] = et['likely_blink']['data'][:]
```

iii. The AI chose pupil_area (converting to diameter) rather than pupil_width directly.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter is computed from pupil area as `2 * sqrt(area / pi)`, treating the pupil as circular. Blink frames are set to NaN before computing diameter. The result is linearly interpolated to ophys timestamps, then discretized into 5 percentile-based bins computed **per session**.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_diameter = np.full_like(pupil_area, np.nan)
valid_pupil = ~np.isnan(pupil_area) & (pupil_area > 0)
pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
```

iii. The AI noted in CONVERSION_NOTES.md (Key Decision 6): "Compute from pupil area as 2*sqrt(area/pi). Set blink frames to NaN, then interpolate. Discretize non-NaN values."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 equal percentile bins computed per session, NaN mapped to bin 0.

ii.
```python
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)
```

iii. Per-session percentile bins.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — interpolated to ophys timestamps for the entire session, then trial frames extracted with the same boolean mask.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
# Per trial:
pupil_trial = pupil_at_ophys[frame_mask]
```

iii. Same alignment approach as running speed.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

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

iii. These four columns are the canonical trial outcome labels for the change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3) via the `outcome_names` list. The integer code is broadcast to all time bins within a trial (static per-trial variable).

ii.
```python
outcome_names = ['hit', 'miss', 'false_alarm', 'correct_reject']
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
output_full[4] = outcome_idx  # broadcast scalar to all timepoints
```

iii. The mapping order is: hit=0, miss=1, false_alarm=2, correct_reject=3.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Failed experiments**: If `process_experiment` returns None (no cells, no stimulus data, <2 valid trials), the experiment is skipped.
- **Short trials**: Trials with fewer than 2 frames are skipped.
- **Missing pupil data**: If eye tracking is not present, pupil values are set to NaN for the entire session.
- **Missing behavioral data**: NaN values from interpolation are mapped to bin 0 during discretization.
- **Missing stimulus data**: If no stimulus presentations table is found, the experiment is skipped.

ii.
```python
if nwb_data['pupil_area'] is None:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
# ...
if stim_data is None:
    return None
# ...
result[~valid] = 0  # NaN gets bin 0
```

iii. The try/except-like pattern (returning None) ensures a single bad experiment doesn't crash the pipeline.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading NWB files via h5py and scanning all files for image names. The AI noted ~1.7s per NWB file load and ~14s for image name collection overhead.

ii. N/A (timing information from CONVERSION_NOTES.md)

iii. Each NWB file contains full-session data. The h5py approach is faster than using the full AllenSDK.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `build_image_identity_trace` function loops over all stimulus presentations to map them onto ophys frames. This inner loop could be vectorized using `np.searchsorted` for all stimulus start/stop times at once. Similarly, `get_all_image_names` scans all NWB files sequentially.

ii.
```python
for si in range(len(stim_starts)):
    s_start = stim_starts[si]
    s_stop = stim_stops[si]
    # ...
    frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
    trace[frame_mask] = img_idx
```

iii. These loops are not major bottlenecks compared to I/O, but could be vectorized.

## 9-c. What processing does the code repeat multiple times?

i. Running speed and pupil diameter are interpolated to ophys timestamps for the entire session, then re-indexed per trial. The per-session bin edge computation is done once per session, which is appropriate. However, the image name collection (`get_all_image_names`) opens every NWB file just to read image names, duplicating I/O that could be combined with the main processing loop.

ii.
```python
# Separate pass for image names:
def get_all_image_names(exp_table):
    for _, row in exp_table.iterrows():
        with h5py.File(nwb_path, 'r') as f:
            # reads image names
```

iii. The two-pass approach (image names first, then processing) was a design choice for simplicity.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code collects and stores extensive session metadata (cre_line, imaging_depth, session_type, etc.) that is not used by the decoder. The `show_processing` mode stores additional large arrays (ophys_ts, running_at_ophys, etc.) that are only needed for plotting.

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

iii. This metadata is useful for documentation but not required for the decoder.
