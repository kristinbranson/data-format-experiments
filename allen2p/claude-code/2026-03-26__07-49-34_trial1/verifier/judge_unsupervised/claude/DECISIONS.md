# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by first reading `ophys_experiment_table.csv` to get the list of experiments, then filters to only those with downloaded NWB files and active (non-passive) sessions. Each experiment's NWB file is loaded individually using h5py direct file reading (not the Allen SDK). From each NWB file, it extracts: dF/F traces, ophys timestamps, running speed, pupil tracking data (area + blinks), trial intervals, and stimulus presentations.

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
    exp_table = exp_table.sort_values('ophys_experiment_id').reset_index(drop=True)
    return exp_table

def load_nwb_data(nwb_path):
    data = {}
    with h5py.File(nwb_path, 'r') as f:
        data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
        dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
        data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
        # ... running speed, pupil, trials, stimulus ...
    return data
```

iii. The AI documented this choice in CONVERSION_NOTES.md Step 6: "Loads NWB files directly via h5py (fast, no AllenSDK overhead)". The agent chose h5py over the AllenSDK for speed and directness.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by `mouse_id` from the experiment table. A mapping dictionary tracks unique mouse IDs and assigns sequential integer indices. Each session's mouse_id is looked up to assign a `subject_idx`.

ii.
```python
mouse_id = result['mouse_id']
if mouse_id not in subject_map:
    subject_map[mouse_id] = len(all_subjects)
    all_subjects.append(mouse_id)
# ...
all_subject_idx.append(subject_map[mouse_id])
```

iii. The AI noted 38 unique mice in the downloaded subset (vs 82 in the full dataset). Subject splitting follows the experiment table's `mouse_id` field.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id` is treated as a separate session. For single-plane (VisualBehavior) data, one experiment corresponds to one imaging session. For multi-plane (VisualBehaviorMultiscope) data, multiple experiments (imaging planes) from the same physical session are treated as separate sessions. This means 202 experiments become 202 "sessions", even though only 174 unique `ophys_session_id` values exist.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    result = process_experiment(exp_id, row, image_names_list, outcome_names, ...)
    if result is None:
        continue
    all_neural.append(result['neural'])
    # ... each experiment becomes one session
```

iii. CONVERSION_NOTES.md Step 5 Decision #9: "Multiscope handling: Each experiment (plane) is a separate 'session' in the output, since they have different neurons but share the same behavioral data." This means behavioral outputs are duplicated across planes from the same physical session.

## 1-d. How are the data split into trials?

i. Trials are defined by the `start_time` and `stop_time` fields in the NWB trials interval table. For each valid trial, ophys frames falling within `[start_time, stop_time)` are extracted. Trials with fewer than 2 ophys frames are skipped.

ii.
```python
for trial_idx in valid_trial_idx:
    t_start = trial_data['start_time'][trial_idx]
    t_stop = trial_data['stop_time'][trial_idx]
    frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
    trial_ts = ophys_ts[frame_mask]
    n_trial_frames = frame_mask.sum()
    if n_trial_frames < 2:
        continue
```

iii. The AI stated in Step 5 Decision #3: "Use start_time and stop_time from trials table for Go and Catch trials only."

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to include only Go and Catch trials, excluding Aborted and Auto-rewarded trials. Valid trials must satisfy: `(go | catch) & ~aborted & ~auto_rewarded`. Additionally, sessions with fewer than 2 valid trials after processing are discarded entirely. No engagement-based filtering is applied.

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

iii. CONVERSION_NOTES.md Step 3: "Include: Go trials and Catch trials. Exclude: Aborted trials and Auto-rewarded trials." This matches the instruction requirements.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the pre-computed dF/F (delta-F-over-F) fluorescence traces stored in the NWB file at `processing/ophys/dff/traces/data`. The AI chose dF/F over the alternative deconvolved "events" signal also available in the NWB files.

ii.
```python
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
```

iii. CONVERSION_NOTES.md Step 5 Decision #1: "Use dF/F traces (not events/deconvolved). dF/F is the standard calcium imaging signal." Step 3 notes: "dF/F: Pre-computed with 600s median filter baseline, detrending with 3.33s median filter."

## 2-b. How is the `neural` data processed?

i. The dF/F traces are used directly as loaded from the NWB file with no additional processing. For each trial, the neural data is sliced to the ophys frames within the trial window and cast to float32. No normalization, smoothing, or z-scoring is applied.

ii.
```python
neural = dff[:, frame_mask].astype(np.float32)  # (n_cells, n_trial_frames)
```

iii. The AI's position was that dF/F is already preprocessed in the NWB files (600s median filter baseline, detrending with 3.33s median filter), so no additional processing is needed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does NOT explicitly filter cells based on `valid_roi` or any other quality metric. The code reads all cells present in the dF/F traces of each NWB file. The AI's justification is that the downloaded NWB files only contain valid ROIs, making explicit filtering unnecessary.

ii.
```python
# No cell filtering code exists. All cells from dff_traces are used:
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # all cells included
```

iii. CONVERSION_NOTES.md Step 10 Check 3: "ROI filtering: No explicit valid_roi filter | OK - all ROIs in downloaded NWB files are valid." The NWB files do contain a `valid_roi` field in `image_segmentation/cell_specimen_table`, but all values are True in the downloaded subset.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to ophys timestamps. For each trial, the ophys frames whose timestamps fall within `[start_time, stop_time)` of the trial are selected. The alignment event is the trial start time (not a stimulus event like change time). No fixed offset is applied; `off_start` and `off_end` are set to `None`.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
neural = dff[:, frame_mask].astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 5 Decision #4: "Align to ophys timestamps. For each trial, extract the ophys frames between trial start_time and stop_time." Metadata sets `temporal_alignment_event = 'Aligned to ophys (2-photon imaging) timestamps. Each trial spans from trial start_time to stop_time.'`

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native ophys frame rate: ~32.3ms (~31 Hz) for Scientifica single-plane sessions, and ~90ms (~11 Hz) for Multiscope sessions. No temporal rebinning is applied. The reported `time_bin_size` in metadata is the median across all sessions (~32.3ms).

ii.
```python
dt = np.median(np.diff(ophys_ts))  # time bin size per session
# No rebinning code exists
# Metadata:
'time_bin_size': median_dt,  # median across all sessions
```

iii. CONVERSION_NOTES.md Step 5 Decision #2: "Use native ophys timestamps (~31 Hz for Scientifica, ~11 Hz for Multiscope). Each session's time bin is consistent within itself." Different sessions may have different frame rates.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table in the NWB file, specifically the `image_name`, `start_time`, and `stop_time` fields from the `Natural_Images_*_presentations` interval.

ii.
```python
stim_starts = stim_data['start_time']
stim_stops = stim_data['stop_time']
stim_names = stim_data['image_name']
```

iii. CONVERSION_NOTES.md Step 5: "image_name from stimulus presentations -> output[0]: image_identity. Map to categorical integer, time-varying at ophys rate. 8 images + gray screen."

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each trial, the image identity trace is initialized to "gray" (index 0). Then, for each stimulus presentation that overlaps the trial window, the ophys frames during that presentation are assigned the corresponding image name index. Omitted stimuli are treated as gray screen. The complete category list is `['gray', 'im000', 'im031', ..., 'im106']` (17 categories: 1 gray + 16 unique images across all sessions).

ii.
```python
def build_image_identity_trace(ophys_ts, stim_data, trial_start, trial_stop, image_names_list):
    gray_idx = image_names_list.index(GRAY_LABEL)
    trace = np.full(n_frames, gray_idx, dtype=np.int64)
    for si in range(len(stim_starts)):
        name = stim_names[si]
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        if name == 'omitted':
            continue
        if name in image_names_list:
            img_idx = image_names_list.index(name)
        frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
        trace[frame_mask] = img_idx
    return trace, trial_mask
```

iii. The AI noted that images are 250ms on / 500ms off, so ~66.7% of timepoints should be gray. The measured value was 66.9%, confirming correct alignment.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is built directly on the ophys timestamps within each trial. Since both neural data and image identity use the same `frame_mask` (ophys frames within trial window), they are inherently aligned at the same temporal resolution.

ii.
```python
# Same frame_mask used for both neural and image identity:
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
neural = dff[:, frame_mask]
img_trace, _ = build_image_identity_trace(ophys_ts, stim_data, t_start, t_stop, image_names_list)
```

iii. The AI verified alignment by plotting processing visualizations and checking gray screen fraction.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` and `start_time` fields in the stimulus presentations table.

ii.
```python
stim_starts = stim_data['start_time']
is_change = stim_data['is_change']
```

iii. CONVERSION_NOTES.md Step 5: "is_change from stimulus presentations -> output[1]: image_change. Binary, 1 at change onset frame, 0 otherwise."

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary trace is created, initialized to 0. For each stimulus presentation marked as a change (`is_change == True`), the first ophys frame at or after the change onset time is set to 1 using `np.searchsorted`.

ii.
```python
def build_image_change_trace(ophys_ts, stim_data, trial_start, trial_stop):
    trace = np.zeros(n_frames, dtype=np.int64)
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

iii. The AI verified: "Image change fraction: 0.372% (expected ~0.3-0.4%)". This low fraction reflects that change events are very sparse (one frame per change).

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary: 0 (no change) or 1 (change). No thresholding is needed. Output values are `['no_change', 'change']`.

ii.
```python
output_values = [
    ...
    ['no_change', 'change'],   # image change: 0=no change, 1=change
    ...
]
```

iii. The instructions specify "binary variable. Have value of 1 right after a change in image identity, otherwise 0."

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is built on ophys timestamps within each trial, using the same frame mask as the neural data. The change event is placed at the first ophys frame at or after the stimulus onset time marked as a change.

ii.
```python
frame_idx = np.searchsorted(trial_ts, s_start)  # first frame at or after change
```

iii. Same alignment as image identity - both use ophys timestamps directly.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
data['running_speed'] = f['processing']['running']['speed']['data'][:]
```

iii. CONVERSION_NOTES.md Step 2: "Running: processing/running/speed (270K samples @ 60 Hz)."

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is first interpolated from its native 60 Hz sampling to the ophys timestamps using linear interpolation (`scipy.interpolate.interp1d`). Then it is discretized into 5 equal percentile bins using session-wide percentile edges.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)
# Per trial:
running_trial = running_at_ophys[frame_mask]
running_binned = apply_percentile_bins(running_trial, running_edges, n_bins=5)
```

iii. CONVERSION_NOTES.md Step 5 Decision #7: "Interpolate from 60 Hz to ophys timestamps using linear interpolation." Decision #8: "Compute percentiles across the entire session (all valid timepoints), then apply per-trial."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal percentile bins (0-20th, 20-40th, 40-60th, 60-80th, 80-100th percentile). Session-wide percentile edges are computed from all non-NaN values across the entire session, then applied per-trial. `np.digitize` assigns bin indices 0-4.

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

iii. The instructions say "discretized into five equal percentile bins." The AI computes percentiles at the session level.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the full session's ophys timestamps first, then sliced to trial frames using the same mask as the neural data. This ensures perfect temporal alignment.

ii.
```python
running_at_ophys = interpolate_to_ophys(nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts)
# Per trial:
running_trial = running_at_ophys[frame_mask]  # same mask as neural
```

iii. Interpolation to ophys timestamps ensures running speed is sampled at the exact same time points as the neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/area/data` (pupil area), `acquisition/EyeTracking/pupil_tracking/timestamps`, and `acquisition/EyeTracking/likely_blink/data` (blink detection).

ii.
```python
data['pupil_area'] = pt['area']['data'][:] if 'data' in pt['area'] else pt['area'][:]
data['pupil_timestamps'] = pt['timestamps'][:]
data['likely_blink'] = et['likely_blink']['data'][:]
```

iii. CONVERSION_NOTES.md Step 2: "Eye tracking: acquisition/EyeTracking/pupil_tracking (area, height, width @ 30 Hz), Blinks: acquisition/EyeTracking/likely_blink."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil processing involves: (1) setting blink-flagged frames to NaN in the pupil area, (2) computing diameter from area using `diameter = 2 * sqrt(area / pi)`, (3) interpolating to ophys timestamps using linear interpolation, and (4) discretizing into 5 percentile bins.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_diameter = np.full_like(pupil_area, np.nan)
valid_pupil = ~np.isnan(pupil_area) & (pupil_area > 0)
pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
```

iii. CONVERSION_NOTES.md Step 5 Decision #6: "Compute from pupil area as 2*sqrt(area/pi). Set blink frames to NaN, then interpolate. Discretize non-NaN values."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 equal percentile bins computed from session-wide non-NaN values. NaN values (from blinks or missing data) are mapped to bin 0 (the lowest bin).

ii.
```python
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)
# In apply_percentile_bins: result[~valid] = 0  # NaN gets bin 0
```

iii. CONVERSION_NOTES.md Step 10: "NaN pupil values (blinks) mapped to bin 0 - design choice, not bug (5 bins specified)." The AI noted that a separate blink category would add a 6th class not specified in the instructions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the full session's ophys timestamps, then sliced using the same trial frame mask as the neural data.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
# Per trial:
pupil_trial = pupil_at_ophys[frame_mask]  # same mask as neural
```

iii. Same alignment approach as running speed.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the `hit`, `miss`, `false_alarm`, and `correct_reject` boolean fields in the NWB trials interval table.

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

iii. CONVERSION_NOTES.md Step 5: "Trial outcome (hit/miss/FA/CR) -> output[4]: trial_outcome. Categorical, static per trial. 4 categories."

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. For each trial, the hit/miss/false_alarm/correct_reject fields are checked in order. The first True value determines the outcome category. The outcome is encoded as an integer index (0-3) and broadcast as a constant value across all timepoints in the trial.

ii.
```python
outcome = get_trial_outcome(trial_data, trial_idx)
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
# ...
output_full[4] = outcome_idx  # broadcast scalar to all timepoints
```

iii. The instructions specify "Trial outcome. Static per-trial." The AI broadcasts the scalar to all timepoints in the output array to maintain a consistent (5, T) shape.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several missing data scenarios are handled:
- **Missing pupil/eye tracking**: If `EyeTracking` or `pupil_tracking` is absent in the NWB file, `pupil_at_ophys` is set to all NaN, which then maps to bin 0.
- **Blinks**: Detected via `likely_blink` flag, pupil area set to NaN, which propagates through to bin 0.
- **Missing stimulus data**: Sessions without stimulus presentations are skipped entirely.
- **Few valid trials**: Sessions with fewer than 2 valid trials are discarded.
- **Few ophys frames**: Trials with fewer than 2 frames are skipped.
- **Unknown trial outcomes**: Mapped to index 0 (hit) as a fallback.
- **NaN in running speed**: Mapped to bin 0 by `apply_percentile_bins`.

ii.
```python
if nwb_data['pupil_area'] is not None:
    # ... process pupil
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)

if stim_data is None:
    print(f"  WARNING: No stimulus data in experiment {exp_id}")
    return None

if len(valid_trial_idx) < 2:
    return None

if n_trial_frames < 2:
    continue
```

iii. The AI documented handling in CONVERSION_NOTES.md Step 10 Check 5: "Sessions with very few valid trials (e.g., 39) are retained if >=2 trials."

## 9-a. What are the most time-consuming steps of the code?

i. NWB file loading is the dominant bottleneck, taking ~1.5-3.4 seconds per session (vs ~0.1-1.0s for processing). Image name collection across all NWB files adds ~14s overhead. Total conversion time is ~7-8 minutes for 202 sessions.

ii.
```python
# Timing output from conversion_full_out.txt shows:
# load=1.6s, process=0.1s (small session)
# load=3.4s, process=1.0s (large session, 504 cells)
```

iii. CONVERSION_NOTES.md Step 7: "Load NWB: ~1.7s, Process trials: ~0.4s, Total: ~3.7s per session."

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
- The stimulus-to-ophys mapping loop in `build_image_identity_trace` iterates over all stimulus presentations for each trial. This could use `np.searchsorted` and vectorized indexing.
- The `build_image_change_trace` similarly loops over presentations.
- The `get_all_image_names` function loops over all NWB files to collect image names.
- The `get_trial_outcome` function checks each field sequentially instead of using vectorized boolean operations on all trials at once.

ii.
```python
# Loop over stimulus presentations per trial:
for si in range(len(stim_starts)):
    s_start = stim_starts[si]
    s_stop = stim_stops[si]
    # ...
    frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
    trace[frame_mask] = img_idx
```

iii. The AI noted in conversion output that processing time is small relative to I/O, so vectorization would have limited impact on total runtime.

## 9-c. What processing does the code repeat multiple times?

i. Several operations are repeated:
- The ophys timestamp masking `(ophys_ts >= t_start) & (ophys_ts < t_stop)` is computed independently in `build_image_identity_trace`, `build_image_change_trace`, and the main trial loop (3 times per trial).
- Running speed and pupil interpolation to ophys timestamps is done once for the full session, which is efficient, but the frame masking to extract trial data is repeated.
- Image name collection scans all NWB files just to find unique image names, even though this information could come from the stimulus presentations already loaded during processing.

ii.
```python
# In main trial loop:
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
# In build_image_identity_trace:
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
# In build_image_change_trace:
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
```

iii. No specific justification given for the repetition.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several aspects:
- **Image name collection pass**: The code scans all NWB files upfront just to collect image names (~14s). This could be done during the main processing loop.
- **Full-session interpolation**: Running speed and pupil are interpolated to all ophys timestamps for the entire session, but only trial segments are used. The inter-trial data is discarded.
- **Metadata collection**: Extensive session metadata (cre_line, imaging_depth, etc.) is collected and stored but not used by the decoder.
- **Processing plots data**: When `--show-processing` is active, extra data (full timestamps, raw signals, edges) is stored per session for plotting.

ii.
```python
# Full-session interpolation (only trial segments used):
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
# But percentile edges need full session data, so this is partially justified
```

iii. The full-session interpolation is partially justified because session-wide percentile edges require all timepoints. However, only the valid trial segments ultimately appear in the output.
