# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files on disk using `h5py`, rather than using the AllenSDK's `VisualBehaviorOphysProjectCache`. It reads the experiment metadata from a CSV table (`ophys_experiment_table.csv`), determines which NWB files exist on disk, filters to active session types (OPHYS_1,3,4,6), and loads each NWB file individually. A 3-pass approach is used: (1) scan NWB files for global image names, (2) collect running/pupil data for global percentile bins, (3) full processing.

ii.
```python
DATA_DIR = Path('data/visual-behavior-ophys-1.1.0')
NWB_DIR = DATA_DIR / 'behavior_ophys_experiments'
META_DIR = DATA_DIR / 'project_metadata'

def get_experiment_list(sample=False):
    exp_table = pd.read_csv(META_DIR / 'ophys_experiment_table.csv')
    nwb_files = list(NWB_DIR.glob('*.nwb'))
    nwb_ids = set()
    for f in nwb_files:
        try:
            eid = int(f.stem.split('_')[-1])
            nwb_ids.add(eid)
        except ValueError:
            pass
    mask = (
        exp_table['ophys_experiment_id'].isin(nwb_ids) &
        exp_table['session_type'].isin(ACTIVE_SESSION_TYPES)
    )
    active_exps = exp_table[mask].copy()
    ...

def load_experiment_data(nwb_path, experiment_id):
    with h5py.File(nwb_path, 'r') as f:
        ...
```

iii. The AI chose to use h5py directly rather than the AllenSDK because the NWB files were already on disk. This bypasses the SDK's caching layer but accesses the same underlying data. The 3-pass approach was chosen to compute global statistics before the main processing pass.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values from the experiment metadata CSV table. Mouse IDs are sorted and converted to strings.

ii.
```python
subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. `mouse_id` is the standard unique identifier for each animal in the Allen dataset.

## 1-c. How are the data split into sessions?

i. Each NWB experiment (one imaging plane) is treated as a separate "session" in the output format. The AI does NOT group multiple imaging planes from the same ophys session together. This means a multi-plane recording session results in multiple output sessions, each with neurons from only one imaging plane.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = NWB_DIR / f'behavior_ophys_experiment_{eid}.nwb'
    ...
    all_neural.append(session_neural)
    all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])
    region_idx = np.full(result['n_cells'], region_to_idx[row['targeted_structure']], dtype=np.int64)
    all_brain_region_idx.append(region_idx)
```

iii. The CONVERSION_NOTES states: "Each NWB experiment = one 'session' in output format (one imaging plane with its own neurons)." This was a simplifying design choice. Each NWB file contains data for one imaging plane, so using one NWB = one session avoids the complexity of merging data across planes.

## 1-d. How are the data split into trials?

i. Trials are defined using the `intervals/trials` group from the NWB files. Each trial spans from `start_time` to `stop_time`. The AI filters to Go and Catch trials (excluding Aborted and Auto-rewarded). Trial windows are variable length.

ii.
```python
valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
valid_indices = np.where(valid_mask)[0]
...
for trial_idx in valid_indices:
    t_trial_start = trials['start_time'][trial_idx]
    t_trial_stop = trials['stop_time'][trial_idx]
    trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
    trial_time_indices = np.where(trial_mask)[0]
    if len(trial_time_indices) < 3:
        continue
```

iii. The AI uses the NWB trial structure directly, filtering to Go and Catch trials per the instructions. The variable-length windows capture the full trial including pre-change stimulus flashes and post-change response period.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) must be Go or Catch (not Aborted or Auto-rewarded), (2) must have at least 3 timepoints in the resampled window, (3) must have a recognized trial outcome (hit/miss/false_alarm/correct_reject -- trials with unknown outcome are skipped). Sessions with fewer than 2 valid trials are excluded.

ii.
```python
valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
...
if len(trial_time_indices) < 3:
    continue
...
if trials['hit'][trial_idx]:
    outcome = 0
elif trials['miss'][trial_idx]:
    outcome = 1
elif trials['false_alarm'][trial_idx]:
    outcome = 2
elif trials['correct_reject'][trial_idx]:
    outcome = 3
else:
    continue  # Unknown outcome, skip
...
if len(neural_trials) < 2:
    ...
    return None
```

iii. The filtering matches the task instructions to include Go and Catch trials while excluding Aborted and Auto-rewarded. The minimum trial length and minimum session size ensure usable data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `event_detection/data` (calcium events from FastLZeroSpikeInference), NOT from dF/F traces. Only neurons with `valid_roi=True` are included.

ii.
```python
events_data = f['processing']['ophys']['event_detection']['data'][()]
events_valid = events_data[:, valid_roi]  # (n_timepoints, n_valid_cells)
```

iii. The CONVERSION_NOTES state: "Neural signal: events (not dF/F): Paper says 'We performed our analyses on discrete calcium events.' Use raw events from event_detection." The AI chose events because the paper describes using them for analysis.

## 2-b. How is the `neural` data processed?

i. The neural events are: (1) filtered by `valid_roi`, (2) linearly interpolated from native ophys timestamps to a regular 30 Hz grid, and (3) clipped to non-negative values after interpolation. No other processing (e.g., normalization, smoothing) is applied.

ii.
```python
# Resample neural events to target rate
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
```

iii. The 30 Hz resampling is based on the paper's description: "linearly interpolating onto a consistent set of 30hz timestamps." Clipping to non-negative preserves the event interpretation (events should be >= 0).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered using the `valid_roi` boolean flag from the cell specimen table in the NWB file. Only neurons with `valid_roi=True` are included. No additional quality filtering is applied.

ii.
```python
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][()].astype(bool)
...
events_valid = events_data[:, valid_roi]
```

iii. The CONVERSION_NOTES mention: "valid_roi filtering: Only include neurons with valid_roi=True" and note that this matches the SDK's default behavior (`exclude_invalid_rois=True`).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start time. The data is first resampled to a regular 30 Hz grid spanning the full session, then trial windows are extracted by finding grid indices within each trial's `[start_time, stop_time)` interval.

ii.
```python
dt = 1.0 / target_rate  # 1/30 = 0.0333s
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
...
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The alignment uses trial start time as the reference event. The 30 Hz grid ensures consistent time bins across all sessions regardless of the native ophys frame rate.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data has a temporal resolution of ~33.33 ms (30 Hz), achieved by linearly interpolating all data streams onto a regular 30 Hz timestamp grid. This is a rebinning/resampling from the native ophys frame rates (~31 Hz for Scientifica, ~11 Hz for Multiscope).

ii.
```python
TARGET_RATE_HZ = 30.0
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ  # ~33.33 ms
...
regular_ts = np.arange(t_start, t_end, dt)
```

iii. The CONVERSION_NOTES state: "Paper: 'linearly interpolating onto a consistent set of 30hz timestamps'" and "Resample to 30 Hz: Paper interpolates to 30 Hz. Needed for consistent time bins across Scientifica (31 Hz) and Multiscope (11 Hz)."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table (`intervals/<stim_key>`) in the NWB file, specifically the `image_name`, `start_time`, `stop_time`, `is_change`, and `omitted` fields. It is NOT derived from the trial-level `initial_image_name`/`change_image_name` fields.

ii.
```python
stim = f['intervals'][stim_key]
stim_data = {
    'start_time': stim['start_time'][()],
    'stop_time': stim['stop_time'][()],
    'image_name': stim['image_name'][()],
    'is_change': stim['is_change'][()].astype(bool),
    'omitted': stim['omitted'][()],
}
```

iii. Using stimulus presentations gives finer temporal resolution than the trial-level summary, since it tracks each individual stimulus flash rather than summarizing the trial as initial/change image.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each timepoint, the most recent non-omitted stimulus onset is found via `np.searchsorted`. The image name from that stimulus is mapped to a global index. Global image names are collected across all experiments in a first pass and sorted alphabetically. During gray screen periods, the identity of the last shown image persists. For omitted flashes, the previous image continues.

ii.
```python
def get_image_at_timepoints(timepoints, stim_data, all_image_names):
    non_omitted = ~stim_data['omitted']
    stim_starts = stim_data['start_time'][non_omitted]
    stim_names = stim_data['image_name'][non_omitted]
    name_to_idx = {name: i for i, name in enumerate(all_image_names)}
    insert_idx = np.searchsorted(stim_starts, timepoints, side='right') - 1
    for i in range(n_tp):
        if insert_idx[i] >= 0:
            img_name = stim_names[insert_idx[i]]
            image_idx[i] = name_to_idx.get(img_name, 0)
        else:
            image_idx[i] = 0
    return image_idx
```

iii. The searchsorted approach efficiently finds the most recent stimulus for each timepoint. The global image name mapping ensures consistent indices across sessions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the same 30 Hz regular grid timepoints used for the neural data. The same `trial_time_indices` extract both neural and image identity data for each trial.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
```

iii. By computing image identity at the same timestamps as the neural data, alignment is guaranteed.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus presentations table, using `is_change` (boolean indicating actual image changes) and `omitted` flags, combined with `start_time` for temporal localization.

ii.
```python
def get_image_change_at_timepoints(timepoints, stim_data):
    change_mask = stim_data['is_change'] & ~stim_data['omitted']
    change_starts = stim_data['start_time'][change_mask]
    change_stops = stim_data['stop_time'][change_mask]
```

iii. `is_change` from stimulus presentations marks actual image changes (equivalent to Go trials).

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary signal is constructed: for each stimulus presentation where `is_change=True` and `omitted=False`, a 750ms window starting at the stimulus onset time is marked as 1. All other timepoints are 0.

ii.
```python
for cs, ce in zip(change_starts, change_stops):
    mask = (timepoints >= cs) & (timepoints < cs + 0.75)
    change_signal[mask] = 1
return change_signal
```

iii. The 750ms window corresponds to one image flash (250ms) plus the following gray inter-stimulus interval (500ms).

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 = no change, 1 = change). No thresholding is needed -- the signal is directly constructed as a binary indicator.

ii. See 4-b code snippet. The values are 0 or 1 by construction.

iii. N/A -- binary variable by definition.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity -- computed at the same 30 Hz regular grid timepoints used for neural data.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
```

iii. Same timestamp alignment as all other data streams.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. This is the standard running speed variable from the Allen SDK's NWB format.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps (~60 Hz) to the 30 Hz regular grid. Global percentile-based bin edges are computed from ALL running speed data across all experiments (full session data, not restricted to trial periods). Running speed is then discretized into 5 bins using these global edges.

ii.
```python
# Resampling
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])

# Percentile bin computation (Pass 2 -- full session data)
running = f['processing']['running']['speed']['data'][()]
all_running_values.append(running.astype(np.float32))
...
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)

# Discretization
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. Linear interpolation preserves the signal while resampling to the target rate. Percentile-based bins ensure roughly equal class counts.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using percentile-based edges computed from all running speed data across all sessions. `np.digitize` maps continuous values to bin indices 0-4. NaN values are mapped to bin 0.

ii.
```python
def compute_percentile_bins(values, n_bins=5):
    valid = values[~np.isnan(values)]
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-10
    return edges

def digitize_to_bins(values, bin_edges):
    n_bins = len(bin_edges) - 1
    binned = np.digitize(values, bin_edges[1:-1])
    binned = np.clip(binned, 0, n_bins - 1)
    binned[np.isnan(values)] = 0
    return binned.astype(np.int64)
```

iii. Percentile binning ensures roughly equal occupancy per bin, which benefits decoder training. The epsilon adjustment handles edge cases where percentile values are identical.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same 30 Hz regular grid as neural data before trial segmentation. The same `trial_time_indices` are used to extract both.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
running_trial = running_resampled[trial_time_indices]
```

iii. Interpolation to the same timebase ensures alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil data is derived from `acquisition/EyeTracking/pupil_tracking/area` (pupil AREA, not width or diameter) and `acquisition/EyeTracking/likely_blink/data` for blink detection.

ii.
```python
pupil_tracking = f['acquisition']['EyeTracking']['pupil_tracking']
pupil_area = pupil_tracking['area'][()]
pupil_ts = pupil_tracking['timestamps'][()]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
```

iii. The AI used pupil area as a proxy for pupil diameter. The CONVERSION_NOTES state: "Use pupil area as proxy for diameter."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames (where `likely_blink=True`) are set to NaN. Then NaN values are linearly interpolated within the timeseries (filling gaps from blinks). The clean signal is then interpolated to the 30 Hz regular grid. Global percentile bins are computed from non-blink pupil area data across all experiments (full session, not trial-restricted). The signal is then discretized into 5 bins.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)  # Fill NaN gaps by interpolation
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)

# Percentile computation (Pass 2)
pupil_area = f['acquisition']['EyeTracking']['pupil_tracking']['area'][()].astype(float)
blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
pupil_area[blink] = np.nan
valid_pupil = pupil_area[~np.isnan(pupil_area)]
...
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
```

iii. Blink removal prevents artifacts. NaN interpolation fills blink gaps before resampling. Global percentile bins ensure consistent categories.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 percentile-based bins with global edges, `np.digitize` for discretization, NaN mapped to bin 0.

ii.
```python
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. Same rationale as running speed discretization.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed -- interpolated to the 30 Hz regular grid before trial segmentation, using the same `trial_time_indices`.

ii.
```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
pupil_trial = pupil_resampled[trial_time_indices]
```

iii. Same approach as all other data streams.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB trials table.

ii.
```python
trials = {
    ...
    'hit': trials_grp['hit'][()].astype(bool),
    'miss': trials_grp['miss'][()].astype(bool),
    'false_alarm': trials_grp['false_alarm'][()].astype(bool),
    'correct_reject': trials_grp['correct_reject'][()].astype(bool),
    ...
}
```

iii. These are the standard trial outcome categories for the change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes: hit=0, miss=1, false_alarm=2, correct_reject=3. The outcome is constant across all timepoints within a trial (broadcast to match trial length). Trials with no recognized outcome are skipped.

ii.
```python
if trials['hit'][trial_idx]:
    outcome = 0
elif trials['miss'][trial_idx]:
    outcome = 1
elif trials['false_alarm'][trial_idx]:
    outcome = 2
elif trials['correct_reject'][trial_idx]:
    outcome = 3
else:
    continue  # Unknown outcome, skip
...
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
```

iii. The integer mapping follows the same order as the reference: hit=0, miss=1, false_alarm=2, correct_reject=3.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Failed experiment loading**: If `load_experiment_data` returns None (e.g., no valid ROIs, missing stimulus data), the experiment is skipped.
- **Missing pupil data**: If eye tracking is not available, pupil values are filled with NaN (then mapped to bin 0 during discretization).
- **Blink frames**: Set to NaN and linearly interpolated.
- **Short trials**: Trials with fewer than 3 timepoints after resampling are skipped.
- **Unknown trial outcomes**: Trials where none of the four outcome flags is True are skipped.
- **Few valid trials**: Experiments with fewer than 2 valid processed trials are skipped.
- **Non-negative events**: Events are clipped to >= 0 after interpolation.

ii.
```python
if raw_data is None:
    continue
...
if len(trial_time_indices) < 3:
    continue
...
else:
    continue  # Unknown outcome, skip
...
if len(neural_trials) < 2:
    return None
...
events_resampled = np.maximum(events_resampled, 0)
...
if raw_data['pupil_area'] is not None:
    ...
else:
    pupil_trial = np.full(n_tp, np.nan)
```

iii. The handling is defensive: individual failures don't crash the pipeline, and missing data is either interpolated or assigned a default value.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the 3-pass approach over NWB files. Each NWB file is opened and read multiple times: Pass 1 scans for image names, Pass 2 collects running/pupil data for percentile computation, and Pass 3 does the full processing. Data loading (h5py file I/O) dominates runtime.

ii.
```python
# Pass 1: image names
for _, row in exp_table.iterrows():
    with h5py.File(nwb_path, 'r') as f:
        ...

# Pass 2: percentile bins
for idx, (_, row) in enumerate(exp_table.iterrows()):
    with h5py.File(nwb_path, 'r') as f:
        ...

# Pass 3: main processing
for idx, (_, row) in enumerate(exp_table.iterrows()):
    raw_data = load_experiment_data(nwb_path, eid)
    ...
```

iii. The CONVERSION_NOTES estimate ~0.5s per experiment per pass, with total conversion time of ~6 minutes for 202 experiments across 3 passes.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `get_image_at_timepoints` function contains a Python for-loop over all timepoints that could be vectorized using fancy indexing. The per-trial processing loop in `process_single_experiment` iterates sequentially over trials. The `interpolate_to_regular_grid` function loops over features when data is 2D.

ii.
```python
# get_image_at_timepoints -- vectorizable loop
for i in range(n_tp):
    if insert_idx[i] >= 0:
        img_name = stim_names[insert_idx[i]]
        image_idx[i] = name_to_idx.get(img_name, 0)

# interpolate_to_regular_grid -- column-by-column loop
for i in range(data.shape[1]):
    result[:, i] = np.interp(target_timestamps, timestamps, data[:, i])
```

iii. The timepoint loop in `get_image_at_timepoints` could use numpy fancy indexing (`image_idx = stim_name_codes[insert_idx]` after mapping names to codes). The column loop in `interpolate_to_regular_grid` could use scipy's `interp1d` with axis parameter or be replaced with a single matrix operation.

## 9-c. What processing does the code repeat multiple times?

i. The code reads each NWB file up to 3 times across its 3 passes: once for image names, once for running/pupil statistics, and once for full processing. Running speed and pupil data are read in both Pass 2 (for percentile computation) and Pass 3 (for trial extraction).

ii.
```python
# Pass 1 reads image names from NWB
with h5py.File(nwb_path, 'r') as f:
    img_names = f['intervals'][k]['image_name'][()]

# Pass 2 reads running/pupil from NWB
with h5py.File(nwb_path, 'r') as f:
    running = f['processing']['running']['speed']['data'][()]
    pupil_area = f['acquisition']['EyeTracking']['pupil_tracking']['area'][()]

# Pass 3 reads everything again
raw_data = load_experiment_data(nwb_path, eid)  # reads running, pupil, events, trials, etc.
```

iii. A single-pass approach could collect image names and statistics while also processing the data, reducing I/O by ~2x.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `stim_data` (full stimulus presentations table) for each experiment, which includes all presentations including omitted flashes and non-trial periods. Only a small fraction of this data (the presentations within trial windows) is actually used. Additionally, the resampling of the entire session's neural events to 30 Hz is done for the full session duration, but only trial-period data is kept. The percentile bin computation in Pass 2 uses full-session running/pupil data (including non-trial periods), which differs from computing bins from only trial-period data.

ii.
```python
# Full session resampling -- only trial windows kept
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)  # Full session
...
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)  # Only trial kept
```

iii. The full-session resampling is wasteful since only ~60-70% of the session falls within trial windows. A more efficient approach would resample only within trial boundaries.
