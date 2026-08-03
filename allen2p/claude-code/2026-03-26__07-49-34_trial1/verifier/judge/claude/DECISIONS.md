# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files using `h5py`, bypassing the Allen SDK's Python API. It reads an experiment table CSV from the project metadata directory, filters to experiments with downloaded NWB files, and then loads each NWB file individually to extract neural, behavioral, and trial data.

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

iii. The AI chose h5py for faster loading without AllenSDK overhead. It filters to active (non-passive) sessions and only processes downloaded NWB files. The CONVERSION_NOTES.md documents this as an intentional choice for efficiency.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values from the experiment table. Each unique mouse_id gets a sequential index.

ii.
```python
mouse_id = result['mouse_id']
if mouse_id not in subject_map:
    subject_map[mouse_id] = len(all_subjects)
    all_subjects.append(mouse_id)
```

iii. Mouse IDs from the experiment table uniquely identify subjects.

## 1-c. How are the data split into sessions?

i. Each experiment (single imaging plane) is treated as a separate session. The AI does NOT group multiple imaging planes from the same ophys session together. Each NWB file = one session in the output.

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

iii. The AI's CONVERSION_NOTES.md (Decision 9) explicitly states: "Each experiment (plane) is a separate 'session' in the output, since they have different neurons but share the same behavioral data." This is a key architectural decision.

## 1-d. How are the data split into trials?

i. Trials are defined using the `intervals/trials` table from the NWB file. Valid trials are Go or Catch trials that are not aborted and not auto-rewarded. The trial window is from `start_time` to `stop_time`.

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

iii. The instructions specify including Go and Catch trials while excluding Aborted and Auto-rewarded trials. The AI uses boolean masking (`ophys_ts >= t_start & ophys_ts < t_stop`) to select frames within the trial window.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) must be Go or Catch, (2) not aborted, (3) not auto-rewarded, (4) must have at least 2 frames (`n_trial_frames < 2`), (5) sessions with fewer than 2 valid trials are discarded. Additionally, passive sessions are excluded at the experiment table level.

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

iii. This follows the instructions to include Go and Catch trials and exclude Aborted and Auto-rewarded trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the dF/F traces stored in the NWB file at `processing/ophys/dff/traces/data`.

ii.
```python
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
```

iii. dF/F is the standard measure for two-photon calcium imaging data. The AI notes it is pre-computed in the NWB files.

## 2-b. How is the `neural` data processed?

i. No additional processing is applied. The dF/F traces are transposed from (n_frames, n_cells) to (n_cells, n_frames) and converted to float32 when extracting per-trial slices.

ii.
```python
neural = dff[:, frame_mask].astype(np.float32)  # (n_cells, n_trial_frames)
```

iii. The AI notes dF/F is pre-computed with neuropil correction and baseline normalization already applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit quality filtering of neurons is performed. All cells present in the NWB file's dF/F traces are included.

ii. N/A (no filtering code)

iii. The AI's CONVERSION_NOTES.md notes: "No explicit valid_roi filter" but claims "all ROIs in downloaded NWB files are valid." The AI chose to trust the NWB file contents rather than applying the SDK's `exclude_invalid_rois` filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to `trial start_time`. Ophys frames are selected using a boolean mask `(ophys_ts >= t_start) & (ophys_ts < t_stop)`, giving a variable-length window per trial.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The full trial window from start_time to stop_time is used. Note the AI uses `< t_stop` (exclusive), while the reference uses `np.searchsorted` with default `side='left'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The native ophys frame rate is used (~31 Hz for Scientifica, ~11 Hz for Multiscope). The time bin size is the median inter-frame interval.

ii.
```python
dt = np.median(np.diff(ophys_ts))
# ...
median_dt = np.median(all_dts)
# stored as time_bin_size in metadata
```

iii. The AI preserves the native temporal resolution. Sessions from different microscopes may have different frame rates.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus presentations` table in the NWB file (at `intervals/Natural_Images_*_presentations`), specifically the `image_name`, `start_time`, and `stop_time` fields. This is fundamentally different from the reference, which uses `initial_image_name` and `change_image_name` from the trials table.

ii.
```python
def build_image_identity_trace(ophys_ts, stim_data, trial_start, trial_stop, image_names_list):
    gray_idx = image_names_list.index(GRAY_LABEL)
    trace = np.full(n_frames, gray_idx, dtype=np.int64)
    stim_starts = stim_data['start_time']
    stim_stops = stim_data['stop_time']
    stim_names = stim_data['image_name']
    for si in range(len(stim_starts)):
        # ...
        frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
        trace[frame_mask] = img_idx
    return trace, trial_mask
```

iii. The AI maps each stimulus flash onto the ophys frames using presentation start/stop times. During the inter-stimulus interval (ISI/gray screen), the trace is set to a "gray" category. This approach includes the gray screen as a distinct image category.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer indices. The mapping includes "gray" as the first category (index 0), followed by sorted unique image names. During ISI (gray screen periods), the identity is set to the gray index. During image presentations, it's set to the image's index. Omitted stimuli are treated as gray.

ii.
```python
image_names_list = [GRAY_LABEL] + all_image_names
# ...
gray_idx = image_names_list.index(GRAY_LABEL)
trace = np.full(n_frames, gray_idx, dtype=np.int64)
# Then fills in image indices during presentations
```

iii. The AI scans all NWB files to collect unique image names, adds "gray" as a category, and builds a global mapping.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed per ophys frame using the same boolean mask as the neural data. Stimulus presentation times are matched to ophys frames within the trial window.

ii.
```python
frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
trace[frame_mask] = img_idx
```

iii. Both use the same ophys timestamp indexing.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in the stimulus presentations table, NOT from the trials table `change_time` or `go` field.

ii.
```python
def build_image_change_trace(ophys_ts, stim_data, trial_start, trial_stop):
    trace = np.zeros(n_frames, dtype=np.int64)
    stim_starts = stim_data['start_time']
    is_change = stim_data['is_change']
    for si in range(len(stim_starts)):
        if not is_change[si]:
            continue
        frame_idx = np.searchsorted(trial_ts, s_start)
        if frame_idx < n_frames:
            trace[frame_idx] = 1
    return trace
```

iii. The AI uses stimulus-level `is_change` annotations from the presentations table.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary trace is created. For each stimulus presentation with `is_change=True`, the single ophys frame at or after the change onset is set to 1. All other frames are 0. This marks only a single frame per change event.

ii. See 4-a code snippet.

iii. The AI marks only one frame at the change point, unlike the reference which marks a 750ms window.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is already binary (0 or 1). No additional thresholding is applied.

ii. N/A

iii. The variable is inherently binary by definition.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Aligned using `np.searchsorted` on the trial's ophys timestamps.

ii.
```python
frame_idx = np.searchsorted(trial_ts, s_start)
```

iii. Same ophys timestamp basis as neural data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
data['running_speed'] = f['processing']['running']['speed']['data'][:]
```

iii. Standard running speed data from the running wheel encoder.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps (~60 Hz) to ophys timestamps. Then it is discretized into 5 percentile bins. Bin edges are computed per-session (not globally across all sessions).

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)
running_binned = apply_percentile_bins(running_trial, running_edges, n_bins=5)
```

iii. The AI computes percentile bin edges per-session rather than globally. This means the same running speed value could map to different bins in different sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Equal percentile bins (0th, 20th, 40th, 60th, 80th, 100th percentiles). The first and last edges are set to -inf and +inf respectively. NaN values are assigned bin 0.

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
    return result
```

iii. Percentile-based binning ensures roughly equal counts per bin. NaN handling maps missing data to bin 0.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the full session's ophys timestamps before trial segmentation, then the trial frames are selected using the same boolean mask.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
running_trial = running_at_ophys[frame_mask]
```

iii. Interpolation to the ophys timebase before trial extraction guarantees alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `pupil_area` (from `acquisition/EyeTracking/pupil_tracking/area/data`) in the NWB file, with blink frames identified via `likely_blink`.

ii.
```python
data['pupil_area'] = pt['area']['data'][:] if 'data' in pt['area'] else pt['area'][:]
data['pupil_timestamps'] = pt['timestamps'][:]
data['likely_blink'] = et['likely_blink']['data'][:]
```

iii. The AI uses pupil area and converts it to diameter, rather than using `pupil_width` directly.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter is computed from pupil area as `2 * sqrt(area / pi)`. Blink frames (from `likely_blink`) are set to NaN before the conversion. The result is then interpolated to ophys timestamps and discretized into 5 per-session percentile bins.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_diameter = np.full_like(pupil_area, np.nan)
valid_pupil = ~np.isnan(pupil_area) & (pupil_area > 0)
pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
```

iii. The AI converts area to diameter assuming a circular pupil. The reference uses `pupil_width` directly from eye tracking data, which is a different measurement.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 equal percentile bins computed per-session, NaN mapped to bin 0.

ii.
```python
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)
```

iii. Per-session percentile edges, same approach as running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed: interpolated to full-session ophys timestamps, then trial frames selected with the same boolean mask.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
pupil_trial = pupil_at_ophys[frame_mask]
```

iii. Same ophys timestamp alignment as neural and running data.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean fields `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB trial data.

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

iii. These four categories are the standard trial outcomes for the change detection task. The fallback 'unknown' handles edge cases.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome label is mapped to an integer index (0-3). The outcome is broadcast as a constant value across all timepoints in the trial.

ii.
```python
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
output_full[4] = outcome_idx  # broadcast scalar to all timepoints
```

iii. Trial outcome is static per trial, so it is repeated for all timepoints to fill the (5, n_timepoints) output array.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing NWB files**: Skipped with a warning.
- **No cells**: Experiment skipped.
- **No stimulus data**: Experiment skipped.
- **Missing eye tracking**: Pupil data set to all NaN.
- **Blink frames**: Set to NaN before diameter computation.
- **NaN behavioral values**: Mapped to bin 0 during discretization.
- **Too few trials**: Experiments with < 2 valid trials are skipped.
- **Too few frames**: Trials with < 2 frames are skipped.

ii.
```python
if nwb_data['pupil_area'] is not None:
    # process pupil
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
# ...
result[~valid] = 0  # NaN gets bin 0
# ...
if len(neural_trials) < 2:
    return None
```

iii. The AI uses defensive checks at multiple levels to handle missing or malformed data.

## 9-a. What are the most time-consuming steps of the code?

i. Loading NWB files via h5py is the most time-consuming step (~1.7s per session according to CONVERSION_NOTES.md). The estimated total load time is ~340s for all sessions. Processing trials is much faster (~0.4s per session).

ii. N/A

iii. The AI's timing data shows load time dominating processing time.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `build_image_identity_trace` function loops over all stimulus presentations to build the per-trial image identity trace. This could be vectorized using `np.searchsorted` and array operations. Similarly, `build_image_change_trace` loops over presentations.

ii.
```python
for si in range(len(stim_starts)):
    # ...
    frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
    trace[frame_mask] = img_idx
```

iii. These loops iterate over ~100-200 stimulus presentations per trial, creating boolean masks each time. Vectorization could reduce this overhead.

## 9-c. What processing does the code repeat multiple times?

i. The image name collection scans all NWB files once at the start (`get_all_image_names`), then each NWB file is loaded again during processing. This means each file's stimulus data is read twice.

ii.
```python
# First pass: collect image names
all_image_names = get_all_image_names(exp_table)
# Second pass: process each experiment
for idx, (_, row) in enumerate(exp_table.iterrows()):
    result = process_experiment(exp_id, row, ...)
```

iii. The double-read is a minor inefficiency. Image names could be collected during the main processing pass.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI includes a "gray" category in image identity, representing the inter-stimulus interval. This adds a category that may not be meaningful for decoding image identity. The downstream decoder must handle this extra category.

ii.
```python
image_names_list = [GRAY_LABEL] + all_image_names
# gray_idx used as default for all non-stimulus timepoints
```

iii. Including gray screen as a category is a design choice. It could be considered unnecessary processing since the instructions ask for "Image identity (of the image presented during the non-grey screen)".
