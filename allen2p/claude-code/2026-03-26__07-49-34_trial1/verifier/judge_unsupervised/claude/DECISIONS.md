# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the ophys experiment table CSV to identify all experiments, filters to those with downloaded NWB files, filters out passive sessions, then iterates over each experiment. For each experiment, it opens the NWB file directly with h5py and extracts dF/F traces, ophys timestamps, running speed, pupil tracking, trial intervals, and stimulus presentations.

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
        # ... loads running, pupil, trials, stimulus
```

iii. The AI justified using h5py directly instead of AllenSDK for speed, noting that AllenSDK adds overhead. The AI documented that NWB files contain pre-computed dF/F, so no additional processing is needed for the neural signal itself.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `mouse_id` column from the ophys experiment table. A mapping dictionary tracks unique mouse IDs, assigning sequential indices. Each experiment's mouse_id determines which subject it belongs to.

ii.
```python
mouse_id = result['mouse_id']
if mouse_id not in subject_map:
    subject_map[mouse_id] = len(all_subjects)
    all_subjects.append(mouse_id)
# ...
mouse_id = str(exp_row['mouse_id'])
```

iii. The AI identified 38 unique mice in the downloaded subset (vs 82 in the full dataset) and documented this as a known limitation of having only a subset of data downloaded.

## 1-c. How are the data split into sessions?

i. Each ophys experiment (NWB file) is treated as a separate session. For Multiscope data, multiple experiments from the same ophys session (different imaging planes) are treated as separate sessions since they have different neurons. The AI processes 202 active experiments as 202 sessions.

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

iii. The AI noted in CONVERSION_NOTES.md: "For Multiscope sessions, multiple experiments (planes) per session share the same trials/behavior" and decided each plane should be a separate session since they have different neurons.

## 1-d. How are the data split into trials?

i. Trials are defined by the `start_time` and `stop_time` from the NWB trials interval table. For each valid trial, ophys frames between `start_time` (inclusive) and `stop_time` (exclusive) are extracted.

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
    neural = dff[:, frame_mask].astype(np.float32)
```

iii. The AI used the standard trial boundaries from the NWB file. Trials with fewer than 2 ophys frames are skipped.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to include only Go and Catch trials, excluding Aborted and Auto-rewarded trials. Additionally, trials with fewer than 2 ophys frames are excluded. Sessions with fewer than 2 valid trials after filtering are excluded entirely.

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

iii. The AI documented this matches the instructions: "Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the pre-computed dF/F (delta F over F) traces stored in the NWB file at `processing/ophys/dff/traces/data`.

ii.
```python
data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
```

iii. The AI noted: "dF/F is pre-computed in NWB files (no need to compute from scratch)" and documented that the dF/F was computed with a "600s median filter baseline, detrending with 3.33s median filter."

## 2-b. How is the `neural` data processed?

i. The neural data undergoes minimal processing: the dF/F matrix is transposed from (n_frames, n_cells) to (n_cells, n_frames), then sliced per-trial using the ophys frame mask, and cast to float32.

ii.
```python
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T  # (n_cells, n_frames)
# ...
neural = dff[:, frame_mask].astype(np.float32)  # (n_cells, n_trial_frames)
```

iii. The AI decided to use the pre-computed dF/F directly rather than computing it from raw fluorescence or using deconvolved events, as dF/F is the "standard calcium imaging signal."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not apply explicit cell-level quality filtering. It relies on the fact that the NWB files in this dataset already contain only valid ROIs (verified by checking that NWB cell counts match the ophys_cells_table.csv which only lists valid cells).

ii.
```python
# No explicit valid_roi filter in code
# All ROIs from NWB are used directly:
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T
```

iii. The AI noted in Step 10: "OK - all ROIs in downloaded NWB files are valid" and confirmed by checking that NWB cell counts match the cells table.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to ophys timestamps. For each trial, frames are selected where ophys timestamps fall within [trial_start_time, trial_stop_time). No alignment to a specific event within the trial (like stimulus change) is performed - the trial spans from start to stop.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The AI set `off_start: None` and `off_end: None` in metadata and described alignment as "Aligned to ophys (2-photon imaging) timestamps. Each trial spans from trial start_time to stop_time."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native ophys frame rate is used without rebinning. Scientifica sessions run at ~31 Hz (32.3 ms bins) and Multiscope sessions at ~11 Hz. The median time bin across sessions is reported as 32.32 ms.

ii.
```python
dt = np.median(np.diff(ophys_ts))  # time bin size
# ...
'time_bin_size': median_dt,  # in metadata
```

iii. The AI decided: "Use native ophys timestamps (~31 Hz for Scientifica, ~11 Hz for Multiscope). Each session's time bin is consistent within itself." No rebinning is applied.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations interval table in the NWB file, specifically the `image_name`, `start_time`, and `stop_time` fields from the `Natural_Images_*_presentations` interval.

ii.
```python
stim_starts = stim_data['start_time']
stim_stops = stim_data['stop_time']
stim_names = stim_data['image_name']
```

iii. The AI collected all unique image names across sessions (16 total across two image sets A and B), plus a "gray" label for inter-stimulus intervals.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A time-varying trace is built at ophys temporal resolution. Each ophys frame is assigned the index of the image being shown at that time (from stimulus presentations), or the "gray" index during inter-stimulus intervals. Omitted stimuli are treated as gray screen.

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

iii. The AI used a global image name list (gray + 16 images) shared across all sessions. Images from set A and set B are included in the same categorical variable.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at ophys timestamps directly. The same frame mask used for neural data is used for the stimulus trace, so they are inherently aligned.

ii.
```python
img_trace, _ = build_image_identity_trace(
    ophys_ts, stim_data, t_start, t_stop, image_names_list
)
# Uses same ophys_ts and same trial window as neural data
```

iii. The AI noted the trace is built "aligned to ophys timestamps for a trial window."

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` and `start_time` fields in the stimulus presentations interval table.

ii.
```python
stim_starts = stim_data['start_time']
is_change = stim_data['is_change']
```

iii. The AI used the pre-computed `is_change` flag from the stimulus presentations.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary time-varying trace is built. For each stimulus presentation flagged as `is_change=True`, the first ophys frame at or after the change onset time is set to 1. All other frames are 0.

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

iii. The AI marks only a single frame per change event, creating a very sparse signal (0.4% of frames are 1).

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 = no change, 1 = change). No thresholding is needed.

ii.
```python
['no_change', 'change'],   # image change: 0=no change, 1=change
```

iii. The instructions specify "binary variable. Have value of 1 right after a change in image identity, otherwise 0."

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The change trace is built using ophys timestamps, matching the neural data alignment. The change event is placed at the first ophys frame at or after the change onset time using `np.searchsorted`.

ii.
```python
frame_idx = np.searchsorted(trial_ts, s_start)
if frame_idx < n_frames:
    trace[frame_idx] = 1
```

iii. Same alignment strategy as image identity - built on ophys timestamps within the trial window.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
data['running_speed'] = f['processing']['running']['speed']['data'][:]
```

iii. Running speed is sampled at 60 Hz in the raw data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is interpolated from its native 60 Hz sampling to ophys timestamps using linear interpolation. Then it is discretized into 5 equal percentile bins computed across the entire session.

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

iii. The AI decided to compute percentile bins across the entire session (not per-trial) for consistent binning.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using equal percentiles (0-20th, 20-40th, 40-60th, 60-80th, 80-100th percentile). NaN values are mapped to bin 0.

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
    if valid.sum() > 0:
        result[valid] = np.clip(np.digitize(values[valid], edges[1:-1]), 0, n_bins - 1)
    result[~valid] = 0
    return result
```

iii. The AI used session-wide percentiles to ensure consistent bin boundaries within a session.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is first interpolated to all ophys timestamps for the session, then per-trial frames are extracted using the same mask as neural data.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts
)
# Per trial:
running_trial = running_at_ophys[frame_mask]
```

iii. Alignment is through interpolation to ophys timestamps, ensuring neural and running speed share the same time base.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from pupil area (`acquisition/EyeTracking/pupil_tracking/area/data`), pupil timestamps, and likely_blink flags.

ii.
```python
data['pupil_area'] = pt['area']['data'][:] if 'data' in pt['area'] else pt['area'][:]
data['pupil_timestamps'] = pt['timestamps'][:]
data['likely_blink'] = et['likely_blink']['data'][:]
```

iii. The AI noted that pupil diameter must be computed from area since only area is directly available.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Processing: (1) Set blink frames to NaN, (2) Compute diameter from area as `2 * sqrt(area / pi)`, (3) Interpolate to ophys timestamps via linear interpolation, (4) Discretize into 5 equal percentile bins across the session.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_diameter = np.full_like(pupil_area, np.nan)
valid_pupil = ~np.isnan(pupil_area) & (pupil_area > 0)
pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
```

iii. The AI documented: "Compute diameter from area: diameter = 2 * sqrt(area / pi). Set blink frames to NaN." The AI decided computing diameter from area was appropriate based on the whitepaper.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 equal percentile bins computed across the session. NaN values (blinks) are mapped to bin 0.

ii.
```python
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)
```

iii. The AI considered whether NaN should be a separate category but decided against it: "A separate blink category would add a 6th class which isn't specified."

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same alignment as running speed: interpolated to ophys timestamps, then per-trial frames extracted with the same mask.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
# Per trial:
pupil_trial = pupil_at_ophys[frame_mask]
```

iii. Alignment through interpolation to ophys timestamps.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the `hit`, `miss`, `false_alarm`, and `correct_reject` fields in the NWB trials interval table.

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

iii. These four outcome categories cover all valid (Go + Catch) trials: Go trials produce hit or miss; Catch trials produce false_alarm or correct_reject.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is a static per-trial variable. The outcome category index is determined from the boolean flags, then broadcast to all timepoints in the trial (constant value across the trial).

ii.
```python
outcome = get_trial_outcome(trial_data, trial_idx)
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
# ...
output_full[4] = outcome_idx  # broadcast scalar to all timepoints
```

iii. The AI's output format broadcasts the static trial outcome to every timepoint, creating a (5, T) output matrix where the last row is constant within each trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases of missing/problematic data are handled:
- Sessions with no cells, no stimulus data, or fewer than 2 valid trials are skipped
- Trials with fewer than 2 ophys frames are skipped
- Missing pupil tracking data: the entire session's pupil is set to NaN, which then gets mapped to bin 0
- Blink frames in pupil data: set to NaN before diameter computation
- NaN values in running/pupil: mapped to bin 0 during discretization
- Stimulus names that are bytes are decoded to UTF-8
- 'omitted' stimuli are treated as gray screen

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
# NaN handling:
result[~valid] = 0  # NaN gets bin 0
# Missing pupil:
pupil_at_ophys = np.full(len(ophys_ts), np.nan)
```

iii. The AI documented edge cases in CONVERSION_NOTES.md Step 10: "Sessions with very few valid trials (e.g., 39) are retained if >=2 trials."

## 9-a. What are the most time-consuming steps of the code?

i. NWB file loading is the dominant cost (~1.5-3.3s per session, comprising ~75% of processing time). Image name collection across all files adds ~14s overhead at startup. Total conversion time was ~9 minutes for 202 sessions.

ii.
```python
# From conversion output:
# Time: 1.8s (load=1.6s, process=0.1s)  -- typical session
# Time: 4.5s (load=3.4s, process=1.0s)  -- large session (504 cells)
```

iii. The AI reported: "Average time per session: ~3.7s" and documented load vs process times.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `build_image_identity_trace` and `build_image_change_trace` functions loop over all stimulus presentations within a trial. These inner loops could be vectorized using numpy operations. The `get_all_image_names` function reads all NWB files sequentially.

ii.
```python
# Inner loop over stimulus presentations:
for si in range(len(stim_starts)):
    s_start = stim_starts[si]
    s_stop = stim_stops[si]
    # ...
    frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
    trace[frame_mask] = img_idx
```

iii. The AI noted the processing time was already acceptable (~9 minutes for full dataset) but did not explicitly vectorize these loops.

## 9-c. What processing does the code repeat multiple times?

i. The image name collection (`get_all_image_names`) opens every NWB file just to read image names, then each file is opened again during processing. Running and pupil interpolation are done for the entire session upfront, then per-trial extraction is done from the interpolated arrays (this is efficient). The stimulus presentation loop in `build_image_identity_trace` iterates over all presentations for each trial, even though most are outside the trial window.

ii.
```python
# First pass: collect image names
def get_all_image_names(exp_table):
    for _, row in exp_table.iterrows():
        with h5py.File(nwb_path, 'r') as f:
            # read image names only

# Second pass: full processing
result = process_experiment(exp_id, ...)
```

iii. The AI acknowledged the ~14s overhead for image collection but did not optimize it by combining with the main processing pass.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores session metadata (cre_line, imaging_depth, session_type, etc.) in the output pickle that is not used by the decoder. The `show_processing` mode stores additional large arrays (ophys_ts, running_at_ophys, etc.) in the result dictionary. The code also processes all stimulus presentations per trial even though many are outside the trial window (though it does have early exit logic with `break`).

ii.
```python
session_metadata.append({
    'exp_id': result['exp_id'],
    'ophys_session_id': result['ophys_session_id'],
    'session_type': result['session_type'],
    'cre_line': result['cre_line'],
    'imaging_depth': result['imaging_depth'],
    # ... not used by decoder
})
```

iii. This extra metadata is useful for documentation but adds to file size. The converted pickle is 8.5 GB.
