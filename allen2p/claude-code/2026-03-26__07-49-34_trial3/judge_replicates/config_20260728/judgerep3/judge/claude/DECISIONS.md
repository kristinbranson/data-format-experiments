# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files using h5py, rather than through the AllenSDK's `VisualBehaviorOphysProjectCache`. It reads experiment metadata from a CSV file (`ophys_experiment_table.csv`), cross-references with NWB files on disk, and filters to active session types only. Data streams (events, trials, running speed, pupil, stimulus presentations) are extracted per-experiment from the NWB file's HDF5 groups.

ii.
```python
def load_experiment_data(nwb_path, experiment_id):
    try:
        with h5py.File(nwb_path, 'r') as f:
            cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
            valid_roi = cell_table['valid_roi'][()].astype(bool)
            ...
            ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][()]
            events_data = f['processing']['ophys']['event_detection']['data'][()]
            events_valid = events_data[:, valid_roi]
            ...
```

```python
def get_experiment_list(sample=False):
    exp_table = pd.read_csv(META_DIR / 'ophys_experiment_table.csv')
    nwb_files = list(NWB_DIR.glob('*.nwb'))
    ...
    mask = (
        exp_table['ophys_experiment_id'].isin(nwb_ids) &
        exp_table['session_type'].isin(ACTIVE_SESSION_TYPES)
    )
```

iii. The AI chose h5py for direct NWB access, noting it reads the same underlying data as the AllenSDK. The AI filtered to active session types (OPHYS_1,3,4,6) to exclude passive viewing sessions.

## 1-b. How are the data split into subjects?

i. Subjects are identified from unique `mouse_id` values in the experiment metadata CSV table. Subject IDs are stored as sorted string representations.

ii.
```python
subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The AI used the experiment table's `mouse_id` field as the canonical subject identifier, consistent with the AllenSDK's data model.

## 1-c. How are the data split into sessions?

i. Each NWB experiment (one imaging plane) is treated as a separate "session" in the output format. The AI does NOT group multiple imaging planes from the same ophys session together; each experiment file becomes one entry in the session lists.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = NWB_DIR / f'behavior_ophys_experiment_{eid}.nwb'
    ...
    result = process_single_experiment(raw_data, exp_meta)
    ...
    all_neural.append(session_neural)
```

iii. The AI explicitly decided: "Each NWB experiment = one 'session' in output format (one imaging plane with its own neurons)." This means multi-plane sessions are split into multiple entries rather than merged.

## 1-d. How are the data split into trials?

i. Trials are defined using the trials table from the NWB file. Valid trials are those where `go` or `catch` is True and `aborted` and `auto_rewarded` are False. Each trial spans from `start_time` to `stop_time` (variable length). A resampled 30 Hz regular time grid is used, and trial timepoints are those grid points within [start_time, stop_time).

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

iii. The AI noted per task instructions: "include Go and Catch trials, exclude Aborted and Auto-rewarded." The filter `go | catch` is redundant since non-aborted, non-auto-rewarded trials should be either go or catch, but makes the intent explicit.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) must be Go or Catch, (2) not aborted, (3) not auto-rewarded, (4) must have at least 3 timepoints after resampling, (5) must have a recognized trial outcome (hit/miss/false_alarm/correct_reject). Experiments with fewer than 2 valid trials are skipped. Trials with unknown outcomes are skipped.

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

iii. The AI followed task instructions to exclude aborted and auto-rewarded trials. The minimum 3-timepoint threshold prevents degenerate trials. The minimum 2-trial threshold per experiment prevents degenerate sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `event_detection` data in the NWB file, which contains pre-computed calcium events from FastLZeroSpikeInference, filtered by `valid_roi`.

ii.
```python
events_data = f['processing']['ophys']['event_detection']['data'][()]
events_valid = events_data[:, valid_roi]  # (n_timepoints, n_valid_cells)
```

iii. The AI noted from the paper: "We performed our analyses on discrete calcium events." This motivated using events rather than dF/F traces.

## 2-b. How is the `neural` data processed?

i. Events are resampled from the native ophys timestamps to a regular 30 Hz grid using linear interpolation. After interpolation, values are clipped to be non-negative (events should be >= 0).

ii.
```python
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
```

```python
def interpolate_to_regular_grid(timestamps, data, target_timestamps):
    if data.ndim == 1:
        return np.interp(target_timestamps, timestamps, data)
    else:
        result = np.zeros((len(target_timestamps), data.shape[1]), dtype=np.float32)
        for i in range(data.shape[1]):
            result[:, i] = np.interp(target_timestamps, timestamps, data[:, i])
        return result
```

iii. The AI noted: "Paper interpolates to 30 Hz. Needed for consistent time bins across Scientifica (31 Hz) and Multiscope (11 Hz)."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data is filtered by `valid_roi`: only cells with `valid_roi == True` in the cell specimen table are included. This is the SVM-based binary classifier output from the Allen pipeline.

ii.
```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
...
events_valid = events_data[:, valid_roi]
```

iii. The AI noted: "Cell filtering: Only automatic filter is valid_roi boolean (SVM binary classifier output). exclude_invalid_rois=True by default."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start time. The ophys timestamps are first resampled to a regular 30 Hz grid, then each trial's neural data is extracted as the grid points falling within [start_time, stop_time).

ii.
```python
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
...
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The AI set `temporal_alignment_event = 'Trial start time (first stimulus onset of trial)'` and `off_start = 0.0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is resampled to 30 Hz (33.33 ms time bins). This is a rebinning from the native ophys rate (~31 Hz for Scientifica, ~11 Hz for Multiscope) to a consistent rate.

ii.
```python
TARGET_RATE_HZ = 30.0
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ  # ~33.33 ms
...
dt = 1.0 / target_rate
regular_ts = np.arange(t_start, t_end, dt)
```

iii. The AI cited the paper: "linearly interpolating onto a consistent set of 30hz timestamps." This ensures all experiments have the same temporal resolution regardless of microscope type.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table in the NWB file, specifically the `start_time`, `image_name`, `omitted`, and `is_change` fields. The most recent non-omitted stimulus onset at each timepoint determines the current image.

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
```

iii. The AI chose to use the stimulus presentations table to determine which image is on screen at each moment, tracking through the full sequence of stimulus flashes rather than just the trial-level initial/change image from the trials table.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer indices via a global mapping built from all unique non-omitted image names across all experiments. The mapping uses sorted image names for determinism. Per-experiment local indices are remapped to global indices.

ii.
```python
global_image_names = sorted(all_image_names_set)
...
local_to_global = {}
for local_idx, name in enumerate(result['all_image_names']):
    if name in global_image_names:
        local_to_global[local_idx] = global_image_names.index(name)
img_id_global = np.array([local_to_global.get(v, 0) for v in trial_data_out['image_identity']], dtype=np.int64)
```

iii. A two-level mapping (local per-experiment, then global) is used because different experiments may have different subsets of images.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the same 30 Hz regular grid timepoints as the neural data, using the same trial time indices.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
```

iii. By computing image identity at the same grid timepoints used for neural data extraction, alignment is guaranteed.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus presentations table, using the `is_change` and `omitted` fields to identify change presentations, plus `start_time` and `stop_time` to determine timing.

ii.
```python
def get_image_change_at_timepoints(timepoints, stim_data):
    change_mask = stim_data['is_change'] & ~stim_data['omitted']
    change_starts = stim_data['start_time'][change_mask]
    change_stops = stim_data['stop_time'][change_mask]
```

iii. The AI used the stimulus-level `is_change` flag, which marks the actual change stimulus presentation. This is derived from the stimulus stream rather than the trial-level `go` flag.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary signal is computed: 1 during a 750ms window starting at each change stimulus onset, 0 otherwise. The 750ms window covers the stimulus duration (250ms) plus the following gray period (500ms).

ii.
```python
for cs, ce in zip(change_starts, change_stops):
    mask = (timepoints >= cs) & (timepoints < cs + 0.75)
    change_signal[mask] = 1
```

iii. The 750ms window matches the stimulus + gray interval described in the paper.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 = no change, 1 = change). No thresholding is needed.

ii.
```python
change_signal = np.zeros(n_tp, dtype=np.int64)
...
change_signal[mask] = 1
```

iii. The binary nature of image change makes it naturally categorical.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is computed at the same 30 Hz regular grid timepoints as the neural data, using the same trial time indices.

ii.
```python
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
```

iii. Same alignment mechanism as image identity.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from the `running/speed` data and timestamps in the NWB file.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. The AI accessed the standard running speed data stream from the NWB file.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is interpolated from its native timestamps (~60 Hz) to the 30 Hz regular grid using `np.interp`, then discretized into 5 equal percentile bins. Bin edges are computed globally across all experiments (from all running speed data, not just trial windows). NaN values are mapped to bin 0.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
all_running_cat = np.concatenate(all_running_values)
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
...
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. The AI computed percentile bins globally for consistency. The global bins are computed from ALL running speed data (including non-trial periods), not just trial-segmented data.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal percentile bins using `np.percentile` to compute edges, then `np.digitize` to assign bins. Edges are adjusted to be strictly increasing (epsilon added for duplicate edges).

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

iii. The AI used equal percentile binning as specified in the instructions ("five equal percentile bins").

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the 30 Hz regular grid before trial segmentation, then extracted at the same trial timepoints as neural data.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
running_trial = running_resampled[trial_time_indices]
```

iii. Same grid-based alignment as neural data ensures temporal consistency.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `pupil_tracking/area` in the eye tracking data, with blinks identified by `likely_blink`.

ii.
```python
pupil_area = f['acquisition']['EyeTracking']['pupil_tracking']['area'][()].astype(float)
pupil_ts = pupil_tracking['timestamps'][()]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
```

iii. The AI used pupil area (not pupil width or diameter) as the proxy for pupil diameter, noting "Use pupil area as proxy for diameter."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames are set to NaN, then NaN values are linearly interpolated in the original timebase before resampling to 30 Hz. The resampled values are then discretized into 5 equal percentile bins computed globally.

ii.
```python
pupil_area = raw_data['pupil_area'].copy().astype(float)
if likely_blink is not None:
    pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
```

```python
def interpolate_nans(arr):
    nans = np.isnan(arr)
    result = arr.copy()
    x = np.arange(len(arr))
    result[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return result
```

iii. Blinks are handled by first setting them to NaN and then interpolating, which fills blink gaps with linearly interpolated values from surrounding non-blink frames.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same percentile binning as running speed: 5 equal percentile bins computed globally from non-blink pupil area values across all experiments. NaN values map to bin 0.

ii.
```python
pupil_area[blink] = np.nan
valid_pupil = pupil_area[~np.isnan(pupil_area)]
if len(valid_pupil) > 0:
    all_pupil_values.append(valid_pupil.astype(np.float32))
...
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
...
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. The AI computed bin edges from non-blink pupil data across all experiments, ensuring blink artifacts don't distort the percentile distribution.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil area is interpolated to the same 30 Hz regular grid as neural data, then extracted at the same trial timepoints.

ii.
```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
pupil_trial = pupil_resampled[trial_time_indices]
```

iii. Same alignment mechanism as running speed and neural data.

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
}
```

iii. These are the SDK's canonical trial outcome labels.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is mapped to integer codes: hit=0, miss=1, false_alarm=2, correct_reject=3. It is static per trial but broadcast to all timepoints within the trial as a constant row.

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

iii. The integer mapping follows the standard order. Trials without a recognized outcome are skipped.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Failed experiments**: If loading or processing fails, the experiment is skipped with a warning.
- **Missing pupil data**: If eye tracking is absent, pupil is filled with NaN.
- **Blinks in pupil**: Set to NaN and linearly interpolated.
- **Short trials**: Trials with fewer than 3 timepoints are skipped.
- **Unknown trial outcomes**: Trials without hit/miss/FA/CR are skipped.
- **Few trials**: Experiments with fewer than 2 valid trials are skipped.
- **NaN in running/pupil**: Mapped to bin 0 during discretization.

ii.
```python
except Exception as e:
    print(f"  ERROR loading {experiment_id}: {e}")
    return None
...
if len(trial_time_indices) < 3:
    continue
...
else:
    continue  # Unknown outcome, skip
...
binned[np.isnan(values)] = 0
```

iii. The AI took a conservative approach: skip problematic data rather than attempting to fix it.

## 9-a. What are the most time-consuming steps of the code?

i. The code makes 3 passes over NWB files: (1) collecting global image names, (2) collecting running/pupil data for global percentile bins, (3) full processing. Each pass requires opening and reading from every NWB file, making I/O the dominant cost.

ii.
```python
# Pass 1: Collecting global image names
for _, row in exp_table.iterrows():
    with h5py.File(nwb_path, 'r') as f:
        ...

# Pass 2: Collecting running speed and pupil data for percentile bins
for idx, (_, row) in enumerate(exp_table.iterrows()):
    with h5py.File(nwb_path, 'r') as f:
        running = f['processing']['running']['speed']['data'][()]
        ...

# Pass 3: Processing experiments
for idx, (_, row) in enumerate(exp_table.iterrows()):
    raw_data = load_experiment_data(nwb_path, eid)
    ...
```

iii. The AI estimated total conversion time at ~6 minutes for 202 experiments.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `interpolate_to_regular_grid` function loops over columns (neurons) for multi-column data. The `get_image_at_timepoints` function uses a Python loop over timepoints. The local-to-global image index remapping also uses a Python list comprehension per trial.

ii.
```python
def interpolate_to_regular_grid(timestamps, data, target_timestamps):
    ...
    result = np.zeros((len(target_timestamps), data.shape[1]), dtype=np.float32)
    for i in range(data.shape[1]):
        result[:, i] = np.interp(target_timestamps, timestamps, data[:, i])
    return result
```

```python
for i in range(n_tp):
    if insert_idx[i] >= 0:
        img_name = stim_names[insert_idx[i]]
        image_idx[i] = name_to_idx.get(img_name, 0)
```

iii. The AI noted these are not bottlenecks compared to I/O, but the neuron-by-neuron interpolation loop could be vectorized.

## 9-c. What processing does the code repeat multiple times?

i. NWB files are opened and read 3 times: once for image names, once for running/pupil percentile computation, and once for full processing. Running speed and pupil data are read in both pass 2 and pass 3.

ii. See 9-a code snippets showing the 3-pass structure.

iii. The AI noted this inefficiency but prioritized correctness over optimization.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The percentile bin edges for running speed and pupil are computed from ALL running/pupil data (including non-trial periods between trials), but only trial-period data is used in the final output. The stimulus presentations table is fully loaded even though only non-omitted presentations are used.

ii.
```python
# Pass 2 collects ALL running speed, not just trial-period
running = f['processing']['running']['speed']['data'][()]
all_running_values.append(running.astype(np.float32))
```

iii. Computing percentiles from all data (including inter-trial intervals) rather than just trial-period data introduces a slight mismatch, as the distribution of running speeds during inter-trial intervals may differ from during trials.
