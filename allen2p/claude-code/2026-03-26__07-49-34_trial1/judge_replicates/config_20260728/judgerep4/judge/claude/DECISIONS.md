# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads NWB files directly using h5py (bypassing the AllenSDK high-level API). It reads the experiment table CSV from the project metadata, filters to experiments whose NWB files exist on disk, and excludes passive sessions by checking for "passive" in the session_type. Each NWB file is loaded individually via `load_nwb_data()`.

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
        # ... loads running, pupil, trials, stimulus presentations
    return data
```

iii. The AI chose to read NWB files directly with h5py for speed, avoiding AllenSDK overhead. It filters to downloaded experiments and excludes passive sessions. However, it does not filter by `project_code`, so it includes both VisualBehavior (single-plane) and VisualBehaviorMultiscope (multi-plane) experiments.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mouse_id` from the experiment table. A mapping from mouse_id to index is built as experiments are processed.

ii.
```python
mouse_id = result['mouse_id']
if mouse_id not in subject_map:
    subject_map[mouse_id] = len(all_subjects)
    all_subjects.append(mouse_id)
```

iii. Uses the standard mouse_id field from the experiment table metadata.

## 1-c. How are the data split into sessions?

i. Each experiment (each NWB file / each imaging plane) is treated as a separate "session" in the output. There is no grouping by `ophys_session_id`. For VisualBehavior (single-plane) data this is equivalent to one session per recording session. For VisualBehaviorMultiscope data, multiple imaging planes from the same recording session become separate output "sessions" sharing the same behavioral data.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    result = process_experiment(exp_id, row, image_names_list, outcome_names, ...)
    if result is None:
        continue
    all_neural.append(result['neural'])
    all_output.append(result['output'])
```

iii. The AI's CONVERSION_NOTES states: "Multiscope handling: Each experiment (plane) is a separate 'session' in the output, since they have different neurons but share the same behavioral data." This means behavioral data (trials, running speed, pupil) is duplicated across planes from the same Multiscope session.

## 1-d. How are the data split into trials?

i. Trials are defined using the `intervals/trials` group in the NWB file. Each trial spans from `start_time` to `stop_time` (variable length). Valid trials are identified as Go or Catch trials that are not aborted and not auto-rewarded.

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

iii. Uses the standard trial boundaries from the NWB trials table. Go + Catch trials are the non-aborted, non-auto-rewarded trials per the task instructions.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) must be Go or Catch (excludes aborted and auto-rewarded), (2) must have at least 2 ophys frames within the trial window, (3) sessions with fewer than 2 valid trials are excluded.

ii.
```python
valid = (go | catch) & ~aborted & ~auto_rewarded

# Per trial:
if n_trial_frames < 2:
    continue

# Per session:
if len(neural_trials) < 2:
    return None
```

iii. The filtering follows the instruction to include Go and Catch trials and exclude Aborted and Auto-rewarded trials. Unlike the reference, the AI does not require `change_time` to be non-NaN.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the dF/F traces stored at `processing/ophys/dff/traces/data` in the NWB file, transposed to (n_cells, n_frames).

ii.
```python
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
```

iii. dF/F is the standard measure for two-photon calcium imaging, pre-computed in the NWB files.

## 2-b. How is the `neural` data processed?

i. No additional processing beyond extracting per-trial windows. The dF/F traces are used as-is from the NWB file. Each experiment's neurons form one session (no cross-plane merging).

ii.
```python
neural = dff[:, frame_mask].astype(np.float32)  # (n_cells, n_trial_frames)
```

iii. The dF/F is pre-computed by the Allen pipeline. No normalization, filtering, or merging is applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No quality filtering is applied to neurons. All cells present in the NWB file's dF/F traces are included, regardless of `valid_roi` status.

ii. No filtering code. All cells from `dff_traces` are used directly.

iii. The AI's CONVERSION_NOTES states: "all ROIs in downloaded NWB files are valid." However, the NWB files contain traces for all ROIs (including potentially invalid ones), whereas the AllenSDK `exclude_invalid_rois=True` default would filter them. The AI reads NWB directly and does not apply this filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to ophys timestamps. For each trial, ophys frames falling within `[start_time, stop_time)` are extracted using a boolean mask.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
neural = dff[:, frame_mask].astype(np.float32)
```

iii. Alignment to ophys timestamps as specified in the instructions. The boolean mask approach is equivalent to `np.searchsorted`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native ophys frame rate is used (~31 Hz for Scientifica, ~11 Hz for Multiscope). No temporal rebinning is applied. The time bin size is computed as the median inter-frame interval across all sessions.

ii.
```python
dt = np.median(np.diff(ophys_ts))
# ...
all_dts = [m['dt_ms'] for m in session_metadata]
median_dt = np.median(all_dts)
```

iii. Native frame rate is preserved. The median time bin is reported as ~32.3 ms in the CONVERSION_NOTES.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table (`intervals/Natural_Images_*_presentations`) in the NWB file. The `image_name`, `start_time`, and `stop_time` fields of each stimulus presentation are used.

ii.
```python
stim_key = None
for key in f['intervals'].keys():
    if 'Natural_Images' in key or 'natural_images' in key:
        stim_key = key
        break
# ...
stim_data = {}
for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
    if key in stim:
        stim_data[key] = stim[key][:]
```

iii. The AI chose to use the stimulus presentations table rather than the trials table's `initial_image_name`/`change_image_name` fields. This provides frame-level timing of when each image flash occurs and ends.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image identity is computed as a time-varying trace. During stimulus flashes, the image name is mapped to an integer index. During inter-stimulus intervals (gray screen / ISI), a "gray" category is used. Omitted stimuli are treated as gray screen. A global `image_names_list` is built by scanning all NWB files, with "gray" prepended as category 0.

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
    return trace, trial_mask

# image_names_list = [GRAY_LABEL] + all_image_names
```

iii. The AI included "gray" as a category to represent inter-stimulus intervals, arguing this captures the actual visual stimulus at each timepoint. This differs from the simpler approach of labeling all timepoints with the trial's associated image identity.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned to the same ophys timestamps as the neural data. The `build_image_identity_trace` function uses the same `trial_mask` (ophys frames within trial boundaries) to construct per-frame image identity.

ii.
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
# ... map stimulus presentations onto trial_ts frames
```

iii. Frame-level alignment ensures each ophys frame has a corresponding image identity label.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in the stimulus presentations table. It marks the first ophys frame at or after any stimulus onset where `is_change` is True.

ii.
```python
def build_image_change_trace(ophys_ts, stim_data, trial_start, trial_stop):
    trace = np.zeros(n_frames, dtype=np.int64)
    stim_starts = stim_data['start_time']
    is_change = stim_data['is_change']
    for si in range(len(stim_starts)):
        if not is_change[si]:
            continue
        s_start = stim_starts[si]
        if s_start < trial_start or s_start >= trial_stop:
            continue
        frame_idx = np.searchsorted(trial_ts, s_start)
        if frame_idx < n_frames:
            trace[frame_idx] = 1
    return trace
```

iii. The AI uses the stimulus presentations table's `is_change` flag rather than the trials table's `change_time` and `go` fields.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary trace is created where 1 marks only the single ophys frame at the onset of a change stimulus. The `is_change` flag from stimulus presentations is used, which may be True for both go trials (actual image change) and catch trials (sham change, same image repeated).

ii. See 4-a code snippet.

iii. The AI marks only a single frame at change onset, rather than a window of time. The AI's CONVERSION_NOTES describes this as "Binary, 1 at change onset frame, 0 otherwise."

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is already binary (0 = no change, 1 = change). No thresholding is applied.

ii.
```python
image_change_value_names = ['no_change', 'change']
```

iii. Binary variable requires no additional thresholding.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same alignment as image identity - the binary trace is computed over the same ophys frames within trial boundaries using `np.searchsorted`.

ii.
```python
frame_idx = np.searchsorted(trial_ts, s_start)
if frame_idx < n_frames:
    trace[frame_idx] = 1
```

iii. Aligned to ophys timestamps using the same trial frame indices.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed` in the NWB file, which provides speed values and timestamps.

ii.
```python
data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
data['running_speed'] = f['processing']['running']['speed']['data'][:]
```

iii. Standard running speed variable from the NWB file.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps (~60 Hz) to the ophys timebase, then discretized into 5 equal percentile bins. NaN values are mapped to bin 0. Percentile bins are computed **per-session** (not globally).

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts)

running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)

def compute_session_percentile_edges(values, n_bins=5):
    valid = values[~np.isnan(values)]
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges

running_binned = apply_percentile_bins(running_trial, running_edges, n_bins=5)
```

iii. The AI computes percentile bin edges per session from that session's running speed data, then applies them to each trial. This means bin boundaries differ across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 percentile bins (0-4) using per-session percentile edges. Bin edges are computed using `np.percentile` at 0, 20, 40, 60, 80, 100th percentiles of non-NaN values within each session. `np.digitize` assigns each value to a bin.

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

iii. Five equal percentile bins as specified in the instructions. Per-session computation means each session has balanced bins independently.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the ophys timebase for the entire session, then per-trial slices are extracted using the same boolean mask as the neural data.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts)
# Per trial:
running_trial = running_at_ophys[frame_mask]
```

iii. Interpolation to ophys timestamps ensures temporal alignment with neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/area` in the NWB file. The `likely_blink` flag is used to exclude blink frames.

ii.
```python
data['pupil_area'] = pt['area']['data'][:] if 'data' in pt['area'] else pt['area'][:]
data['pupil_timestamps'] = pt['timestamps'][:]
data['likely_blink'] = et['likely_blink']['data'][:]
```

iii. The AI uses `pupil_area` rather than `pupil_width`, then converts area to diameter.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames are set to NaN. Pupil diameter is computed from pupil area via `diameter = 2 * sqrt(area / pi)`, assuming a circular pupil. The diameter is then linearly interpolated to the ophys timebase and discretized into 5 per-session percentile bins.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_diameter = np.full_like(pupil_area, np.nan)
valid_pupil = ~np.isnan(pupil_area) & (pupil_area > 0)
pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
```

iii. The AI computed diameter from area rather than using the directly available `pupil_width`. Per-session percentile binning is applied.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 per-session percentile bins. NaN values mapped to bin 0.

ii. Same discretization functions as running speed (see 5-c).

iii. Five equal percentile bins as specified in the instructions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed: interpolated to ophys timebase, then per-trial slices extracted.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
# Per trial:
pupil_trial = pupil_at_ophys[frame_mask]
```

iii. Same interpolation-based alignment as running speed.

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

iii. Standard trial outcome classification from the NWB trials table.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is mapped to an integer index (0-3) based on the order `['hit', 'miss', 'false_alarm', 'correct_reject']`. The integer is broadcast to all timepoints within the trial.

ii.
```python
outcome_names = ['hit', 'miss', 'false_alarm', 'correct_reject']
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
output_full[4] = outcome_idx  # broadcast scalar to all timepoints
```

iii. Per-trial static variable broadcast across frames to maintain the (5, n_frames) output shape.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases handled: (1) Missing NWB files are skipped with a warning. (2) Experiments with no cells or no stimulus data are skipped. (3) Sessions with fewer than 2 valid trials are skipped. (4) Trials with fewer than 2 ophys frames are skipped. (5) NaN values in running speed and pupil diameter (from interpolation or blinks) are mapped to bin 0. (6) Missing eye tracking data results in all-NaN pupil values.

ii.
```python
if not os.path.exists(nwb_path):
    return None
if n_cells == 0:
    return None
if stim_data is None:
    return None
if len(valid_trial_idx) < 2:
    return None
if n_trial_frames < 2:
    continue
# NaN handling:
result[~valid] = 0  # NaN gets bin 0
if nwb_data['pupil_area'] is None:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
```

iii. Defensive checks ensure the pipeline doesn't crash on missing or incomplete data.

## 9-a. What are the most time-consuming steps of the code?

i. Loading NWB files via h5py is the dominant cost (~1.7s per file). The AI estimates ~12.5 minutes for the full 202-experiment conversion. Additionally, `get_all_image_names()` scans all NWB files upfront to collect unique image names, adding ~14s overhead.

ii. N/A (timing from CONVERSION_NOTES)

iii. I/O from reading large NWB files dominates runtime.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `build_image_identity_trace` function loops over all stimulus presentations for each trial, doing per-presentation boolean masking. This is O(trials x presentations) per session. The `build_image_change_trace` has a similar loop. The per-trial loop in `process_experiment` iterates sequentially.

ii.
```python
# In build_image_identity_trace:
for si in range(len(stim_starts)):
    # ... per-stimulus boolean mask
    frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
    trace[frame_mask] = img_idx
```

iii. The stimulus presentation loop could be vectorized using `np.searchsorted` for all presentations at once, or replaced entirely by using the trials table's simpler `initial_image_name`/`change_image_name` approach.

## 9-c. What processing does the code repeat multiple times?

i. The `get_all_image_names()` function opens and reads every NWB file just to collect image names. Then each NWB file is opened again during `process_experiment()`. This is a redundant pass over all data files.

ii.
```python
def get_all_image_names(exp_table):
    for _, row in exp_table.iterrows():
        eid = row['ophys_experiment_id']
        nwb_path = os.path.join(NWB_DIR, f'behavior_ophys_experiment_{eid}.nwb')
        with h5py.File(nwb_path, 'r') as f:
            # reads image names only
```

iii. Each NWB file is opened twice: once to collect image names, once for processing.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores per-session `running_edges` and `pupil_edges` but these are not saved to the output pickle (they are local to each `process_experiment()` call). Additionally, the `stimulus_duration_ms`, `inter_stimulus_interval_ms`, and `flash_cycle_ms` metadata fields are computed but not used by the decoder. The conversion from pupil area to diameter is unnecessary computation since `pupil_width` is directly available in the NWB file.

ii. N/A

iii. The per-session bin edges and extra metadata add minor overhead but don't affect correctness.
