# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files on disk using `h5py`, rather than through the AllenSDK's `VisualBehaviorOphysProjectCache`. It reads the experiment metadata CSV (`ophys_experiment_table.csv`) to discover experiments, filters to active session types (OPHYS_1,3,4,6), cross-references against NWB files on disk, and loads each experiment individually via `h5py.File`.

ii.
```python
DATA_DIR = Path('data/visual-behavior-ophys-1.1.0')
NWB_DIR = DATA_DIR / 'behavior_ophys_experiments'
META_DIR = DATA_DIR / 'project_metadata'

ACTIVE_SESSION_TYPES = [
    'OPHYS_1_images_A', 'OPHYS_3_images_A',
    'OPHYS_4_images_B', 'OPHYS_6_images_B',
]

def get_experiment_list(sample=False):
    exp_table = pd.read_csv(META_DIR / 'ophys_experiment_table.csv')
    nwb_files = list(NWB_DIR.glob('*.nwb'))
    nwb_ids = set()
    for f in nwb_files:
        eid = int(f.stem.split('_')[-1])
        nwb_ids.add(eid)
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

iii. The AI chose h5py for direct NWB access, which is functionally similar to the AllenSDK but bypasses the SDK abstraction layer. The AI also decided to filter to only active session types (OPHYS_1,3,4,6), excluding passive sessions (OPHYS_2,5), reasoning that passive sessions have no active behavior (no lick spout, satiated mice). The AI does not handle the `DATALIMIT_SUBSET.csv` file that the reference uses to restrict to a subset of experiments.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values from the experiment metadata table, sorted and converted to strings.

ii.
```python
subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. Same approach as reference — using `mouse_id` from the experiment table as the unique subject identifier.

## 1-c. How are the data split into sessions?

i. Each NWB experiment (imaging plane) is treated as a separate "session" in the output format. The AI does NOT group multiple experiments from the same `ophys_session_id` together. This means multi-plane sessions produce multiple output sessions, each with neurons from only one imaging plane.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = NWB_DIR / f'behavior_ophys_experiment_{eid}.nwb'
    raw_data = load_experiment_data(nwb_path, eid)
    result = process_single_experiment(raw_data, exp_meta)
    ...
    all_neural.append(session_neural)
```

iii. From CONVERSION_NOTES.md key decision #13: "Each NWB experiment = one 'session' in output format (one imaging plane with its own neurons)." The AI noted this explicitly. This means the same behavioral trials are duplicated across multiple output sessions when a recording session has multiple imaging planes.

## 1-d. How are the data split into trials?

i. Trials are defined using the trials table from the NWB file. Go and Catch trials are included; Aborted and Auto-rewarded trials are excluded. The trial window spans from `start_time` to `stop_time` (variable length). Trials with fewer than 3 timepoints (at the resampled 30 Hz rate) are skipped.

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

iii. The AI's trial filtering is similar to the reference: exclude aborted and auto-rewarded trials. The AI uses `(go | catch)` as a positive filter in addition to the negative filters, while the reference uses only the negative filters plus a `change_time.notna()` check. Trials with unknown outcomes (none of hit/miss/FA/CR) are also skipped.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) must be Go or Catch (not aborted, not auto-rewarded), (2) must have at least 3 timepoints at 30 Hz, (3) must have a known outcome (hit/miss/FA/CR), (4) experiments with fewer than 2 valid trials are skipped, (5) neurons are filtered by `valid_roi`.

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
    return None
```

iii. The AI's trial quality filtering is reasonable. The `valid_roi` neuron filtering adds a quality control not present in the reference.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `event_detection/data` (calcium events from FastLZeroSpikeInference), filtered to only neurons with `valid_roi == True`.

ii.
```python
events_data = f['processing']['ophys']['event_detection']['data'][()]
events_valid = events_data[:, valid_roi]  # (n_timepoints, n_valid_cells)
```

iii. From CONVERSION_NOTES.md key decision #1: "Neural signal: events (not dF/F): Paper says 'We performed our analyses on discrete calcium events.' Use raw events from event_detection."

## 2-b. How is the `neural` data processed?

i. Neural events are filtered by `valid_roi`, then linearly interpolated from native ophys timestamps to a regular 30 Hz grid. After interpolation, values are clipped to be non-negative.

ii.
```python
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
```

iii. From CONVERSION_NOTES.md key decision #4: "Resample to 30 Hz: Paper interpolates to 30 Hz. Needed for consistent time bins across Scientifica (31 Hz) and Multiscope (11 Hz)." The non-negativity clipping ensures events remain physically meaningful after interpolation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by the `valid_roi` boolean field from the cell specimen table. Only neurons with `valid_roi == True` are included. Experiments with zero valid ROIs are skipped entirely.

ii.
```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
cell_specimen_ids = cell_table['cell_specimen_id'][()]
n_valid = valid_roi.sum()
if n_valid == 0:
    print(f"  WARNING: No valid ROIs in experiment {experiment_id}, skipping")
    return None
events_valid = events_data[:, valid_roi]
```

iii. From CONVERSION_NOTES.md key decision #10: "valid_roi filtering: Only include neurons with valid_roi=True." This matches the AllenSDK default behavior (`exclude_invalid_rois=True`).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start time. The regular 30 Hz timestamp grid is created for the entire session, and per-trial slices are extracted by masking timestamps within `[start_time, stop_time)`.

ii.
```python
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The alignment to trial start is consistent with the reference. The use of `>=` for start and `<` for stop is a standard half-open interval convention.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is resampled to 30 Hz (time bin size = 33.33 ms). All data streams (neural events, running speed, pupil) are interpolated to a regular 30 Hz grid.

ii.
```python
TARGET_RATE_HZ = 30.0
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ  # ~33.33 ms
dt = 1.0 / target_rate
regular_ts = np.arange(t_start, t_end, dt)
```

iii. From CONVERSION_NOTES.md: "Paper: 'linearly interpolating onto a consistent set of 30hz timestamps'." The AI followed the paper's stated methodology.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table in the NWB file, specifically the `image_name`, `start_time`, `stop_time`, `omitted`, and `is_change` fields from the non-omitted stimulus entries.

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

iii. The AI chose to use the stimulus presentations table rather than the trial-level `initial_image_name`/`change_image_name` fields. This provides frame-level image identity throughout the trial.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each timepoint in the trial, the most recent non-omitted stimulus onset is found via `np.searchsorted`, and the corresponding image name is mapped to a global integer index. During gray screens and omitted flashes, the last-shown image identity persists.

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

iii. The AI's approach uses the stimulus presentations table for finer-grained image identity tracking. The global image mapping ensures consistent integer codes across all experiments.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at each resampled 30 Hz timepoint within the trial window, using the same `trial_time_indices` as the neural data.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
```

iii. Because all data streams use the same 30 Hz regular timestamp grid, alignment is guaranteed.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `stimulus_presentations` table, using the `is_change` and `omitted` fields to identify change events, and the `start_time` and `stop_time` to define the temporal window.

ii.
```python
def get_image_change_at_timepoints(timepoints, stim_data):
    change_mask = stim_data['is_change'] & ~stim_data['omitted']
    change_starts = stim_data['start_time'][change_mask]
    change_stops = stim_data['stop_time'][change_mask]
    for cs, ce in zip(change_starts, change_stops):
        mask = (timepoints >= cs) & (timepoints < cs + 0.75)
        change_signal[mask] = 1
    return change_signal
```

iii. The AI used the stimulus presentations' `is_change` flag rather than the trial-level `go` flag + `change_time`. This approach captures change events directly from the stimulus stream.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary signal is created: 1 during a 750ms window starting at each change stimulus onset, 0 otherwise. The 750ms window corresponds to one stimulus flash (250ms) plus the following grey inter-stimulus interval (500ms).

ii.
```python
for cs, ce in zip(change_starts, change_stops):
    mask = (timepoints >= cs) & (timepoints < cs + 0.75)
    change_signal[mask] = 1
```

iii. The 750ms window is consistent with the reference's approach. However, the AI marks changes for ALL change stimuli in the stimulus table (both go and catch trial changes), while the reference only marks changes for go trials.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is binary (0 or 1) and not further thresholded. It is inherently categorical.

ii. N/A - binary by construction.

iii. No thresholding needed for a binary variable.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity — computed at each 30 Hz timepoint within the trial window.

ii. See 4-a code above.

iii. Same alignment approach as all other time-varying outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. This is the standard running speed data from the Allen SDK, accessed via h5py.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the 30 Hz regular grid, then discretized into 5 equal-percentile bins computed globally across all experiments.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
...
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. The percentile-based binning ensures roughly equal class counts. Global bin edges provide consistency across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using percentile-based edges. NaN values are mapped to bin 0. The `compute_percentile_bins` function also enforces strictly increasing edges by adding epsilon.

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

iii. The percentile approach matches the reference. The epsilon adjustment for duplicate edges is a robustness measure.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same 30 Hz regular grid as neural data before trial segmentation, then the same time indices are used.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
running_trial = running_resampled[trial_time_indices]
```

iii. Same alignment as all other time-varying outputs.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/area` (pupil area, not width/diameter) and `acquisition/EyeTracking/likely_blink/data` for blink detection.

ii.
```python
pupil_area = f['acquisition']['EyeTracking']['pupil_tracking']['area'][()]
pupil_ts = pupil_tracking['timestamps'][()]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
```

iii. The AI used pupil area as a proxy for pupil diameter. The reference uses `pupil_width`.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames (where `likely_blink` is True) are set to NaN. NaN values are then linearly interpolated within the pupil time series before resampling to 30 Hz. Finally, the resampled pupil area is discretized into 5 equal-percentile bins computed globally.

ii.
```python
pupil_area = raw_data['pupil_area'].copy().astype(float)
if likely_blink is not None:
    pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
```

iii. The AI interpolates NaNs (blinks) before resampling, effectively reconstructing the pupil signal during blinks. The reference excludes blink frames and interpolates to the ophys timebase, leaving NaN for out-of-range timestamps.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 equal-percentile bins computed from non-NaN values across all experiments. NaN values are mapped to bin 0.

ii.
```python
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. Same as running speed discretization approach.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — interpolated to the 30 Hz regular grid, then the same trial time indices are used.

ii.
```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
pupil_trial = pupil_resampled[trial_time_indices]
```

iii. Same alignment as all other time-varying outputs.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB trials table.

ii.
```python
trials = {
    'hit': trials_grp['hit'][()].astype(bool),
    'miss': trials_grp['miss'][()].astype(bool),
    'false_alarm': trials_grp['false_alarm'][()].astype(bool),
    'correct_reject': trials_grp['correct_reject'][()].astype(bool),
}
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
    continue
```

iii. Same outcome mapping as the reference. Trials with no matching outcome are skipped.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_reject) and broadcast as a constant across all timepoints in the trial.

ii.
```python
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)
```

iii. Same approach as the reference — static per-trial value broadcast to all timepoints.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Failed experiments**: If `load_experiment_data` throws an exception, the experiment is skipped.
- **Missing pupil data**: If eye tracking is not available, pupil values are filled with NaN (then binned to 0).
- **Blinks in pupil data**: Set to NaN, then linearly interpolated before resampling.
- **Short trials**: Trials with fewer than 3 timepoints are skipped.
- **Unknown outcomes**: Trials without a recognized outcome are skipped.
- **Few valid trials**: Experiments with fewer than 2 valid trials are skipped.
- **No valid ROIs**: Experiments with zero valid ROIs are skipped.
- **Missing stimulus presentations**: Experiments without a stimulus table are skipped.

ii.
```python
except Exception as e:
    print(f"  ERROR loading {experiment_id}: {e}")
    return None
...
if raw_data['pupil_area'] is not None:
    ...
else:
    pupil_trial = np.full(n_tp, np.nan)
...
if len(trial_time_indices) < 3:
    continue
```

iii. The AI handles missing data robustly with multiple fallback strategies. The NaN interpolation for pupil blinks is more aggressive than the reference (which leaves NaN for missing data and maps to bin 0).

## 9-a. What are the most time-consuming steps of the code?

i. The AI performs 3 passes over the NWB files: (1) collecting global image names, (2) collecting running speed and pupil data for percentile bins, (3) full experiment processing. This means each NWB file is read 3 times, with data loading being the main bottleneck.

ii.
```python
# Pass 1: collect image names
for _, row in exp_table.iterrows():
    with h5py.File(nwb_path, 'r') as f:
        ...
# Pass 2: collect running/pupil stats
for idx, (_, row) in enumerate(exp_table.iterrows()):
    with h5py.File(nwb_path, 'r') as f:
        ...
# Pass 3: full processing
for idx, (_, row) in enumerate(exp_table.iterrows()):
    raw_data = load_experiment_data(nwb_path, eid)
    result = process_single_experiment(raw_data, exp_meta)
```

iii. The 3-pass approach is needed because global statistics (image names, bin edges) must be computed before final discretization. However, passes 1 and 2 could be combined.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `get_image_at_timepoints` function uses a Python loop over all timepoints to assign image indices, which could be fully vectorized since the `searchsorted` is already vectorized.

ii.
```python
insert_idx = np.searchsorted(stim_starts, timepoints, side='right') - 1
for i in range(n_tp):
    if insert_idx[i] >= 0:
        img_name = stim_names[insert_idx[i]]
        image_idx[i] = name_to_idx.get(img_name, 0)
```

iii. The loop body could be replaced with vectorized array indexing.

## 9-c. What processing does the code repeat multiple times?

i. The code reads NWB files 3 times (3 passes). Passes 1 and 2 each open every NWB file to extract image names and running/pupil statistics respectively, then pass 3 does the full load. The reference uses a single-pass approach by storing intermediate results.

ii. See 9-a code above.

iii. The multi-pass approach trades efficiency for simpler code structure, but results in 3x the I/O.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI resamples all data to 30 Hz via linear interpolation. For sessions already at ~31 Hz (Scientifica equipment), this is a near-identity transformation that adds computation without meaningful benefit. The reference avoids this by keeping native ophys timestamps. Additionally, the AI interpolates NaN values in pupil data (reconstructing blink periods), only to later bin the values — the reconstructed values may not be meaningful.

ii.
```python
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
```

iii. The 30 Hz resampling follows the paper's stated methodology but adds processing overhead and potential interpolation artifacts, especially for the Multiscope data at 11 Hz which gets upsampled 3x.
