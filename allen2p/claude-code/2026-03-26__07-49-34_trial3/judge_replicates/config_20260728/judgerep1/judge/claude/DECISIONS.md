# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files on disk using `h5py`, rather than using the AllenSDK's `VisualBehaviorOphysProjectCache`. It reads experiment metadata from a CSV table (`ophys_experiment_table.csv`), filters to NWB files available on disk, and then loads each experiment's NWB file individually. It filters to "active" session types (OPHYS_1, OPHYS_3, OPHYS_4, OPHYS_6) before loading.

ii.
```python
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

iii. The AI chose to use h5py directly to access NWB files, noting this reads the same data as the AllenSDK NWB reader. The AI also chose to filter to active session types (excluding passive sessions OPHYS_2, OPHYS_5) based on the reasoning that passive sessions have no active behavior (no lick spout, satiated mice) and thus no meaningful trial outcomes.

## 1-b. How are the data split into subjects?

i. Subjects are determined from unique `mouse_id` values in the experiment metadata table, sorted as strings.

ii.
```python
subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The `mouse_id` field uniquely identifies each animal.

## 1-c. How are the data split into sessions?

i. Each NWB experiment file (one imaging plane) is treated as a separate "session" in the output format. The AI does NOT group multiple imaging planes from the same ophys session together.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = NWB_DIR / f'behavior_ophys_experiment_{eid}.nwb'
    ...
    all_neural.append(session_neural)
    ...
    all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])
```

iii. The CONVERSION_NOTES.md states: "Each NWB experiment = one 'session' in output format (one imaging plane with its own neurons)." This means multi-plane sessions are split into separate output sessions rather than merged.

## 1-d. How are the data split into trials?

i. Trials are defined using the trials table within each NWB file. Go and Catch trials are included; Aborted and Auto-rewarded trials are excluded. The trial window spans from `start_time` to `stop_time`, giving variable-length trials. Trials with fewer than 3 timepoints (at the resampled 30 Hz rate) are skipped.

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

iii. Per task instructions, Go and Catch trials are included while Aborted and Auto-rewarded are excluded.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) excluding aborted trials, (2) excluding auto-rewarded trials, (3) requiring go or catch flag, (4) requiring at least 3 timepoints at 30 Hz, (5) requiring a deterministic trial outcome (hit/miss/FA/CR — trials with unknown outcome are skipped). Sessions with fewer than 2 valid trials are excluded.

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
```

iii. The filtering follows the task instructions. Trials without a recognized outcome are skipped as a safeguard.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `event_detection` data in the NWB files (calcium events computed via FastLZeroSpikeInference), NOT from dF/F traces. Only neurons with `valid_roi == True` are included.

ii.
```python
events_data = f['processing']['ophys']['event_detection']['data'][()]
events_valid = events_data[:, valid_roi]  # (n_timepoints, n_valid_cells)
```

iii. The CONVERSION_NOTES.md states: "Paper says 'We performed our analyses on discrete calcium events.' Use raw events from event_detection." The AI chose events over dF/F because the reference paper (Piet et al. 2024) uses events for their analyses.

## 2-b. How is the `neural` data processed?

i. The event traces are (1) filtered to valid_roi neurons only, (2) linearly interpolated from the native ophys timestamps to a regular 30 Hz grid, and (3) clipped to be non-negative after interpolation.

ii.
```python
events_valid = events_data[:, valid_roi]
...
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
```

iii. The 30 Hz resampling follows the paper's statement: "linearly interpolating onto a consistent set of 30hz timestamps." This ensures consistent time bins across Scientifica (~31 Hz) and Multiscope (~11 Hz) equipment.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data is filtered by `valid_roi` — only ROIs classified as valid by the Allen SDK's SVM classifier are included. No further quality filtering is applied.

ii.
```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
...
events_valid = events_data[:, valid_roi]
```

iii. The `valid_roi` filter is the SDK's default cell quality filter (SVM binary classifier). The CONVERSION_NOTES.md documents this as matching the SDK's `exclude_invalid_rois=True` default.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial start time. After resampling to 30 Hz, timepoints within the trial window (`start_time` to `stop_time`) are extracted.

ii.
```python
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
...
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The alignment is to trial start (first stimulus onset). The resampled grid provides consistent 30 Hz time resolution.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is resampled from the native ophys frame rate (~31 Hz for Scientifica, ~11 Hz for Multiscope) to a uniform 30 Hz grid via linear interpolation. The time bin size is ~33.33 ms.

ii.
```python
TARGET_RATE_HZ = 30.0
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ  # ~33.33 ms
...
dt = 1.0 / target_rate
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
```

iii. The paper states "linearly interpolating onto a consistent set of 30hz timestamps." The AI followed this convention to ensure uniform time bins across different equipment types.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table in the NWB file (specifically `image_name`, `start_time`, `stop_time`, `omitted`, `is_change` fields), NOT from the trials table's `initial_image_name`/`change_image_name`.

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
...
def get_image_at_timepoints(timepoints, stim_data, all_image_names):
    non_omitted = ~stim_data['omitted']
    stim_starts = stim_data['start_time'][non_omitted]
    stim_names = stim_data['image_name'][non_omitted]
    insert_idx = np.searchsorted(stim_starts, timepoints, side='right') - 1
    ...
```

iii. The AI used the stimulus presentations table because it provides precise timing of each image presentation, including omitted flashes and grey periods. During grey screen, the identity of the most recently presented image is used.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each timepoint, the most recent non-omitted stimulus onset is found via `searchsorted`, and its image name is mapped to a global integer index. Image names are collected globally across all experiments and sorted alphabetically. Images named 'omitted' are excluded from the global image set.

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
            image_idx[i] = 0  # Before first stimulus
    return image_idx
```

iii. Using searchsorted on stimulus presentations gives frame-accurate image identity. The global mapping ensures consistent encoding across experiments.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the same 30 Hz resampled timepoints as the neural data, using the same `trial_time_indices`.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
```

iii. Both neural and image identity data use the same resampled timestamp grid.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `stimulus_presentations` table's `is_change` and `omitted` fields, combined with `start_time` and `stop_time`.

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

iii. The AI used the stimulus presentations' `is_change` flag rather than the trials table's `go` flag. Both should identify the same change events, but from different data sources.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary signal is created: 1 during a 750ms window starting at each change stimulus onset (non-omitted, is_change=True), 0 otherwise. The 750ms window corresponds to one image presentation (250ms) plus the grey inter-stimulus interval (500ms).

ii.
```python
change_mask = stim_data['is_change'] & ~stim_data['omitted']
change_starts = stim_data['start_time'][change_mask]
for cs, ce in zip(change_starts, change_stops):
    mask = (timepoints >= cs) & (timepoints < cs + 0.75)
    change_signal[mask] = 1
```

iii. The 750ms window matches one full stimulus cycle (250ms on + 500ms grey).

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 = no change, 1 = change). No thresholding is needed.

ii. N/A — the variable is binary by construction.

iii. Per task instructions, image change is a binary variable.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity — computed at the same 30 Hz resampled timepoints.

ii.
```python
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
```

iii. Uses the same timestamp grid as neural data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. This is the SDK's standard running speed data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is (1) linearly interpolated from its native timestamps to the 30 Hz regular grid using `np.interp`, then (2) discretized into 5 equal percentile bins computed globally across all experiments.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
...
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. The percentile-based binning ensures roughly equal class counts. Bins are computed from all experiments globally. Note: the AI computes percentile bins from ALL running speed samples (entire sessions, not just trial periods), unlike the reference which computes from trial-period data only.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using global percentile edges. The `digitize_to_bins` function maps values into bins 0-4, with NaN values mapped to bin 0. Bin edges are forced to be strictly increasing by adding epsilon (1e-10) when edges are equal.

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

iii. The epsilon adjustment prevents degenerate bin edges. The clip ensures values outside the range map to the first or last bin.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is resampled to the same 30 Hz grid as neural data, then trial segments are extracted using the same time indices.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
running_trial = running_resampled[trial_time_indices]
```

iii. All data streams share the same resampled timestamp grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from the pupil tracking `area` (NOT `pupil_width`) from `acquisition/EyeTracking/pupil_tracking/area` in the NWB file. Blink frames are identified from `acquisition/EyeTracking/likely_blink/data`.

ii.
```python
pupil_area = pupil_tracking['area'][()]
pupil_ts = pupil_tracking['timestamps'][()]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
```

iii. The AI used pupil area as a proxy for pupil diameter. The CONVERSION_NOTES.md acknowledges this choice.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil area processing: (1) blink frames (likely_blink=True) are set to NaN, (2) NaN values are linearly interpolated within the pupil time series, (3) the result is resampled to 30 Hz using `np.interp`, (4) discretized into 5 percentile bins computed globally.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. The AI chose to interpolate NaN values before resampling to avoid propagating blink artifacts. Percentile bins are computed from blink-removed data across all experiments.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 equal percentile bins with global edges, NaN mapped to bin 0, epsilon adjustment for equal edges.

ii. Same `compute_percentile_bins` and `digitize_to_bins` functions as running speed.

iii. Consistent with running speed discretization approach.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — resampled to 30 Hz grid, then extracted per trial.

ii.
```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
pupil_trial = pupil_resampled[trial_time_indices]
```

iii. All data streams share the same resampled grid.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the `hit`, `miss`, `false_alarm`, and `correct_reject` boolean columns in the NWB trials table.

ii.
```python
trials = {
    ...
    'hit': trials_grp['hit'][()].astype(bool),
    'miss': trials_grp['miss'][()].astype(bool),
    'false_alarm': trials_grp['false_alarm'][()].astype(bool),
    'correct_reject': trials_grp['correct_reject'][()].astype(bool),
}
...
if trials['hit'][trial_idx]:
    outcome = 0  # hit
elif trials['miss'][trial_idx]:
    outcome = 1  # miss
elif trials['false_alarm'][trial_idx]:
    outcome = 2  # false_alarm
elif trials['correct_reject'][trial_idx]:
    outcome = 3  # correct_reject
else:
    continue  # Unknown outcome, skip
```

iii. These four columns are the SDK's canonical trial outcome labels. Trials that don't match any outcome are skipped.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_reject). The code is broadcast across all timepoints within a trial.

ii.
```python
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)
```

iii. The outcome is static per trial but represented as a constant time series for uniform output shape.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Failed experiment loads**: If loading an NWB file fails, the experiment is skipped with an error message.
- **No valid ROIs**: Experiments with no valid ROIs are skipped.
- **No stimulus presentations**: Experiments without a recognized stimulus table are skipped.
- **Missing pupil data**: If eye tracking data is absent, pupil values are filled with NaN.
- **Short trials**: Trials with fewer than 3 timepoints are skipped.
- **Unknown outcome**: Trials without a recognized outcome are skipped.
- **Few trials**: Experiments with fewer than 2 valid trials are skipped.
- **NaN in pupil**: Blink-related NaN values are interpolated before resampling.

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
    ...
    return None
```

iii. The AI handles missing data conservatively by skipping problematic experiments/trials rather than attempting to fix them.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading NWB files via h5py, which involves reading large neural and behavioral data arrays from disk. The AI makes 3 passes over the data: (1) collecting image names, (2) computing percentile bins, (3) full processing — meaning each NWB file is read multiple times.

ii.
```python
# Pass 1: image names
for _, row in exp_table.iterrows():
    with h5py.File(nwb_path, 'r') as f: ...

# Pass 2: percentile bins
for idx, (_, row) in enumerate(exp_table.iterrows()):
    with h5py.File(nwb_path, 'r') as f: ...

# Pass 3: full processing
for idx, (_, row) in enumerate(exp_table.iterrows()):
    raw_data = load_experiment_data(nwb_path, eid)
    ...
```

iii. The CONVERSION_NOTES.md acknowledges "Multiple passes over NWB files" as an inefficiency.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-timepoint loop in `get_image_at_timepoints` iterates over each timepoint to look up image identity, which could be vectorized using numpy fancy indexing. The per-column interpolation in `interpolate_to_regular_grid` loops over features when data is 2D.

ii.
```python
# Per-timepoint loop in get_image_at_timepoints
for i in range(n_tp):
    if insert_idx[i] >= 0:
        img_name = stim_names[insert_idx[i]]
        image_idx[i] = name_to_idx.get(img_name, 0)

# Per-column interpolation loop
for i in range(data.shape[1]):
    result[:, i] = np.interp(target_timestamps, timestamps, data[:, i])
```

iii. The searchsorted is already vectorized but the subsequent lookup loop is not. The per-column interpolation could be vectorized with scipy's interpolation functions.

## 9-c. What processing does the code repeat multiple times?

i. The NWB files are read 3 times: once for image names (Pass 1), once for running/pupil statistics (Pass 2), and once for full data extraction (Pass 3). The running speed and pupil data are read twice (Pass 2 for bins, Pass 3 for processing).

ii.
```python
# Pass 1: image names scan
for _, row in exp_table.iterrows():
    with h5py.File(nwb_path, 'r') as f: ...
# Pass 2: running/pupil stats
for idx, (_, row) in enumerate(exp_table.iterrows()):
    with h5py.File(nwb_path, 'r') as f: ...
# Pass 3: full processing
for idx, (_, row) in enumerate(exp_table.iterrows()):
    raw_data = load_experiment_data(nwb_path, eid)
```

iii. The CONVERSION_NOTES.md acknowledges this: "Multiple passes over NWB files" listed as a code inefficiency.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes running speed and pupil percentile bins from ALL data samples across the entire session (not just trial periods). This means non-trial periods (inter-trial intervals, spontaneous activity periods) contribute to the bin edge computation, though they are not included in the final output. The `change_stops` variable is loaded but not used (the 750ms window is hardcoded from `change_starts`). The `interpolate_nans` function is applied to pupil data before resampling — these interpolated values are then potentially re-interpolated by `np.interp`.

ii.
```python
# Bin edges from ALL running data, not just trial periods
all_running_values.append(running.astype(np.float32))
# Unused change_stops
change_stops = stim_data['stop_time'][change_mask]
```

iii. Computing bins from all data (not just trial periods) may slightly affect bin edges compared to the reference approach of computing bins only from trial-period data.
