# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files using h5py (not the AllenSDK). It reads the ophys_experiment_table.csv metadata to identify which experiments to process, filters to only those with NWB files on disk and matching active session types, then iterates through each NWB file to extract neural events, trials, stimulus presentations, running speed, and pupil tracking data. A 3-pass approach is used: (1) collect global image names, (2) collect running/pupil data for global percentile bins, (3) process each experiment.

ii.
```python
def load_experiment_data(nwb_path, experiment_id):
    with h5py.File(nwb_path, 'r') as f:
        cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
        valid_roi = cell_table['valid_roi'][()].astype(bool)
        ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][()]
        events_data = f['processing']['ophys']['event_detection']['data'][()]
        events_valid = events_data[:, valid_roi]
        trials_grp = f['intervals']['trials']
        # ... extracts all trial fields, stimulus presentations, running speed, pupil
```

```python
def get_experiment_list(sample=False):
    exp_table = pd.read_csv(META_DIR / 'ophys_experiment_table.csv')
    nwb_files = list(NWB_DIR.glob('*.nwb'))
    mask = (
        exp_table['ophys_experiment_id'].isin(nwb_ids) &
        exp_table['session_type'].isin(ACTIVE_SESSION_TYPES)
    )
```

iii. The AI documented in CONVERSION_NOTES that it chose h5py for direct NWB access rather than the AllenSDK, reading the same data paths as the SDK would. The 3-pass approach was chosen to compute global percentile bins before per-trial processing. The AI verified that 284 NWB files were available on disk but only 202 matched active session type criteria.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified via the `mouse_id` column in `ophys_experiment_table.csv`. A sorted list of unique mouse IDs is created, and each session is mapped to its subject via `subject_to_idx`.

ii.
```python
subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
# ...
all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])
```

iii. The AI noted 38 mice were available on disk (subset of full 82 in the paper). Mouse IDs come from the experiment metadata table.

## 1-c. How are the data split into sessions?

i. Each NWB experiment file is treated as a separate "session" in the output format. This means multi-plane recordings (from MESO.1 equipment) where multiple imaging planes are recorded simultaneously during one behavioral session result in multiple "sessions" in the output — one per imaging plane/experiment. The AI selected only active session types: OPHYS_1_images_A, OPHYS_3_images_A, OPHYS_4_images_B, OPHYS_6_images_B (excluding passive: OPHYS_2, OPHYS_5).

ii.
```python
ACTIVE_SESSION_TYPES = [
    'OPHYS_1_images_A', 'OPHYS_3_images_A',
    'OPHYS_4_images_B', 'OPHYS_6_images_B',
]
```

iii. The AI documented that passive sessions (OPHYS_2, OPHYS_5) were excluded because they involve passive viewing with no lick spout and satiated mice, so no meaningful trial outcomes. The result is 202 experiments from 174 behavioral sessions. One subject (457841) has 34 "sessions" in the output, likely because it has many imaging planes across multiple sessions.

## 1-d. How are the data split into trials?

i. Trials are defined by the `intervals/trials` group in each NWB file. Each trial has a `start_time` and `stop_time`. The regular timestamp grid (at 30 Hz) is segmented using these boundaries: timepoints where `regular_ts >= start_time` and `regular_ts < stop_time` are assigned to that trial. Trials with fewer than 3 timepoints are skipped.

ii.
```python
for trial_idx in valid_indices:
    t_trial_start = trials['start_time'][trial_idx]
    t_trial_stop = trials['stop_time'][trial_idx]
    trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
    trial_time_indices = np.where(trial_mask)[0]
    if len(trial_time_indices) < 3:
        continue
```

iii. The AI used the NWB trial boundaries as-is, resulting in variable-length trials (mean ~254 timepoints = ~8.5s at 30 Hz).

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to include only Go and Catch trials, excluding Aborted and Auto-rewarded trials. Additionally, trials must have at least 3 timepoints and must have exactly one of hit/miss/false_alarm/correct_reject as their outcome. No other trial-level quality filtering is applied (e.g., no engagement filtering, no session QC).

ii.
```python
valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
# ...
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

iii. The AI's CONVERSION_NOTES document that aborted trials (premature lick before change) and auto-rewarded trials (5 free rewards at session start + after 10 consecutive misses) are excluded per the task instructions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the pre-computed event detection data (`processing/ophys/event_detection/data`) in the NWB files, filtered by the `valid_roi` boolean from the cell specimen table. These are calcium events computed via FastLZeroSpikeInference.

ii.
```python
events_data = f['processing']['ophys']['event_detection']['data'][()]
events_valid = events_data[:, valid_roi]  # (n_timepoints, n_valid_cells)
```

iii. The AI noted: "Paper says 'We performed our analyses on discrete calcium events.' Use raw events from event_detection." and "Events are PRE-COMPUTED via FastLZeroSpikeInference."

## 2-b. How is the `neural` data processed?

i. The raw calcium events (at the original ophys frame rate, ~31 Hz for Scientifica or ~11 Hz for Multiscope) are linearly interpolated onto a regular 30 Hz timestamp grid. After interpolation, values are clipped to be non-negative (events should be >= 0). The data is then segmented into per-trial windows. No additional filtering (e.g., causal half-gaussian smoothing) is applied.

ii.
```python
regular_ts = np.arange(t_start, t_end, dt)  # dt = 1/30
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
# ...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The AI justified 30 Hz resampling based on the paper: "linearly interpolating onto a consistent set of 30hz timestamps". The AI noted the SDK's optional causal half-gaussian filtering but chose not to apply it: "Optional filtering with causal half-gaussian (scale=2.0/31.0 sec, n_steps=20)."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs with `valid_roi == True` are included. This is a binary SVM classifier output from the Allen pipeline that flags valid vs. invalid ROIs. No additional quality filtering (e.g., signal-to-noise ratio, responsiveness) is applied.

ii.
```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
cell_specimen_ids = cell_table['cell_specimen_id'][()]
events_valid = events_data[:, valid_roi]
```

iii. The AI documented: "Cell filtering: Only automatic filter is valid_roi boolean (SVM binary classifier output). exclude_invalid_rois=True by default."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to ophys timestamps, then interpolated to a 30 Hz regular grid. Per-trial segments are extracted based on trial start_time and stop_time from the NWB trials table. The temporal alignment event is described as "Trial start time (first stimulus onset of trial)" with off_start=0.0 and off_end=None (variable trial length).

ii.
```python
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
# ...
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
```

iii. The instructions say "Temporally align based on ophys timestamp." The AI used ophys timestamps as the basis for the regular 30 Hz grid, then extracted trial windows from that grid.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 33.33 ms (30 Hz). Rebinning is done via linear interpolation from the original ophys timestamps (~31 Hz Scientifica, ~11 Hz Multiscope) to a regular 30 Hz grid.

ii.
```python
TARGET_RATE_HZ = 30.0
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ  # ~33.33 ms
dt = 1.0 / target_rate
regular_ts = np.arange(t_start, t_end, dt)
```

iii. The AI cited the paper: "linearly interpolating onto a consistent set of 30hz timestamps" and noted this ensures consistent time bins across different microscope types.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table (`intervals/<stim_key>/image_name`) in the NWB files, specifically from non-omitted stimulus presentations.

ii.
```python
stim = f['intervals'][stim_key]
stim_data = {
    'start_time': stim['start_time'][()],
    'image_name': stim['image_name'][()],
    'omitted': stim['omitted'][()],
}
# ...
non_omitted = ~stim_data['omitted']
stim_starts = stim_data['start_time'][non_omitted]
stim_names = stim_data['image_name'][non_omitted]
```

iii. The AI documented: "stimulus presentations table, stimulus templates (8 natural images)" and maps image names to global categorical indices across all experiments.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each timepoint, the most recent non-omitted stimulus onset is found using `np.searchsorted`. The image name from that stimulus is used as the identity. During gray screen periods (between flashes), the last shown image is used. For omitted flashes, the previous image continues. Local per-experiment image indices are mapped to a global set of 16 image names across all experiments.

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
```

iii. The AI justified: "Image identity during gray screen: Use the identity of the image that was just shown (last presented image)." and "Image identity for omitted flashes: Continue with previous image identity."

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at each timepoint of the 30 Hz regular grid (same as neural data), so it is inherently aligned — both use the same `regular_ts` timestamps within each trial window.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
```

iii. Both neural and image identity are evaluated at the same regular 30 Hz timepoints within each trial, ensuring temporal alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus presentations table, specifically from the `is_change` and `omitted` fields, along with `start_time` and `stop_time`.

ii.
```python
change_mask = stim_data['is_change'] & ~stim_data['omitted']
change_starts = stim_data['start_time'][change_mask]
change_stops = stim_data['stop_time'][change_mask]
```

iii. The AI identifies change stimuli as non-omitted presentations where `is_change == True`.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each change stimulus, a binary signal of 1 is set for all timepoints within a 750ms window starting at the change onset. All other timepoints are 0.

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

iii. The AI uses a hardcoded 750ms window (250ms stimulus + 500ms gray) rather than the actual `stop_time`. The variable `ce` (change end) is extracted but not used; instead `cs + 0.75` is used.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1), so no thresholding is needed. Value 0 = "no_change", value 1 = "change".

ii.
```python
output_values = [
    # ...
    ['no_change', 'change'],  # image change values
]
```

iii. The instructions specify "binary variable. Have value of 1 right after a change in image identity, otherwise 0."

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is computed at the same 30 Hz regular grid timepoints as neural data within each trial, using the same `trial_ts` array.

ii.
```python
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
```

iii. Same alignment approach as image identity — evaluated at shared 30 Hz grid points.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB files.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. The AI noted: "Running speed (~60 Hz)" from the NWB behavioral data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed (originally ~60 Hz) is linearly interpolated to the 30 Hz regular grid using `np.interp`. Then it is discretized into 5 percentile bins using globally-computed bin edges (from all running speed data across all experiments).

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
# ...
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. The AI computed bin edges globally in a separate pass: "Collecting running speed and pupil data for percentile bins."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal percentile bins (0-4) computed from all running speed data across all experiments. The `np.digitize` function maps values to bins based on the bin edges. Bin edges were: [-24.1, -0.004, 0.336, 14.2, 33.1, 99.9] cm/s.

ii.
```python
def compute_percentile_bins(values, n_bins=5):
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    return edges

def digitize_to_bins(values, bin_edges):
    n_bins = len(bin_edges) - 1
    binned = np.digitize(values, bin_edges[1:-1])
    binned = np.clip(binned, 0, n_bins - 1)
    return binned.astype(np.int64)
```

iii. Instructions say "discretized into five equal percentile bins." The AI computes global percentiles across all data. The resulting distribution is roughly equal: bin_0=19.9%, bin_1=20.3%, bin_2=18.4%, bin_3=20.9%, bin_4=20.5%.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same 30 Hz regular grid as neural data, then extracted at the same trial timepoints.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
running_trial = running_resampled[trial_time_indices]
```

iii. Same 30 Hz grid alignment as other data streams.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/area` (pupil area, not diameter) and `acquisition/EyeTracking/likely_blink/data` for blink detection.

ii.
```python
pupil_tracking = f['acquisition']['EyeTracking']['pupil_tracking']
pupil_area = pupil_tracking['area'][()]
pupil_ts = pupil_tracking['timestamps'][()]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
```

iii. The AI acknowledged: "Use pupil area as proxy for diameter." The NWB files contain pupil area, not diameter directly.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames (identified by `likely_blink == True`) are set to NaN. NaN values are then linearly interpolated. The cleaned pupil area is resampled to the 30 Hz regular grid via linear interpolation. Finally, it is discretized into 5 percentile bins computed globally (from all non-blink pupil data across experiments).

ii.
```python
pupil_area = raw_data['pupil_area'].copy().astype(float)
if likely_blink is not None:
    pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
# ...
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. The AI documented: "During blinks (likely_blink=True or NaN), linearly interpolate. Compute percentile bins from non-blink data."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil area is discretized into 5 equal percentile bins (0-4), with bin edges computed globally from all non-blink pupil area data. The resulting distribution is roughly equal: bin_0=22.8%, bin_1=18.5%, bin_2=19.1%, bin_3=19.0%, bin_4=20.6%.

ii.
```python
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. Same approach as running speed: global percentile bins ensuring roughly equal bin sizes.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil area is interpolated to the same 30 Hz regular grid as neural data, then extracted at the same trial timepoints.

ii.
```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
pupil_trial = pupil_resampled[trial_time_indices]
```

iii. Same alignment mechanism as other data streams.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial outcome flags in the NWB trials table: `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
trials = {
    'hit': trials_grp['hit'][()].astype(bool),
    'miss': trials_grp['miss'][()].astype(bool),
    'false_alarm': trials_grp['false_alarm'][()].astype(bool),
    'correct_reject': trials_grp['correct_reject'][()].astype(bool),
}
```

iii. Each valid (Go or Catch) trial has exactly one of these outcome flags set to True.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is encoded as a categorical integer: 0=hit, 1=miss, 2=false_alarm, 3=correct_reject. Despite being specified as "static per-trial" in the instructions, the AI broadcasts this value across all timepoints in the trial to create a (1, n_timepoints) output row that is concatenated with the time-varying outputs, resulting in shape (5, n_timepoints).

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
    continue
# ...
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)  # (5, n_tp)
```

iii. The AI documented trial outcome as "Static per-trial categorical" with 4 classes. Broadcasting to time-varying format may be required by the decoder framework.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several types of missing/problematic data are handled:
- **Missing pupil data**: If no eye tracking data exists for an experiment, pupil values are filled with NaN (which then gets binned to bin 0).
- **Blinks in pupil data**: Frames marked as `likely_blink` are set to NaN, then linearly interpolated before resampling.
- **Omitted stimuli**: Omitted flashes are skipped when determining image identity (previous image continues).
- **Negative events after interpolation**: Clipped to 0 with `np.maximum(events_resampled, 0)`.
- **Unknown trial outcomes**: Trials without a clear hit/miss/FA/CR outcome are skipped.
- **Short trials**: Trials with fewer than 3 timepoints are skipped.
- **Sessions with few valid trials**: Sessions with fewer than 2 valid trials are skipped.
- **Bytes encoding**: String fields in NWB are decoded from bytes if needed.
- **NaN in digitize_to_bins**: NaN values get assigned bin 0.

ii.
```python
# Pupil blinks
pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)

# Negative events
events_resampled = np.maximum(events_resampled, 0)

# NaN in bins
binned[np.isnan(values)] = 0

# Missing pupil
pupil_trial = np.full(n_tp, np.nan)
```

iii. The AI documented these decisions in CONVERSION_NOTES under various steps.

## 9-a. What are the most time-consuming steps of the code?

i. Based on the conversion output, each experiment takes ~0.5s total (0.25s load + 0.25s process). The full conversion of 202 experiments took ~6 minutes (358s). The most time-consuming aspects are:
1. NWB file I/O (loading from h5py) — done 3 times per file across 3 passes
2. Linear interpolation of events to 30 Hz grid
3. The second pass collecting all running speed and pupil data for global percentile computation

ii.
```python
# Pass 1: scan all files for image names
for _, row in exp_table.iterrows():
    with h5py.File(nwb_path, 'r') as f:
        # read image names

# Pass 2: scan all files for running/pupil
for idx, (_, row) in enumerate(exp_table.iterrows()):
    with h5py.File(nwb_path, 'r') as f:
        running = f['processing']['running']['speed']['data'][()]
        # read pupil data

# Pass 3: full processing
for idx, (_, row) in enumerate(exp_table.iterrows()):
    raw_data = load_experiment_data(nwb_path, eid)
```

iii. The AI estimated full conversion at ~2-3 minutes; it actually took ~6 minutes. The 3-pass approach reads each NWB file 3 times.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops could be vectorized:
1. The `get_image_at_timepoints` function uses a Python for-loop over all timepoints to map indices to image names, despite already using `np.searchsorted` for the index lookup.
2. The `get_image_change_at_timepoints` function loops over each change stimulus to create the binary signal.

ii.
```python
# Loop in get_image_at_timepoints:
for i in range(n_tp):
    if insert_idx[i] >= 0:
        img_name = stim_names[insert_idx[i]]
        image_idx[i] = name_to_idx.get(img_name, 0)

# Loop in get_image_change_at_timepoints:
for cs, ce in zip(change_starts, change_stops):
    mask = (timepoints >= cs) & (timepoints < cs + 0.75)
    change_signal[mask] = 1
```

iii. The AI noted "Vectorized interpolation using np.interp" and "Efficient searchsorted for image identity assignment" as speedups, but the Python-level loops within these functions remain.

## 9-c. What processing does the code repeat multiple times?

i. Each NWB file is opened and partially read 3 times across the 3 passes:
1. Pass 1 reads image names from stimulus presentations
2. Pass 2 reads running speed and pupil area data
3. Pass 3 does a full load of all data

This means running speed, pupil area, and image names are read twice. Additionally, `interpolate_to_regular_grid` iterates over columns of the events matrix individually rather than doing batch interpolation.

ii.
```python
# Pass 1: image names
with h5py.File(nwb_path, 'r') as f:
    img_names = f['intervals'][k]['image_name'][()]

# Pass 2: running + pupil
with h5py.File(nwb_path, 'r') as f:
    running = f['processing']['running']['speed']['data'][()]
    pupil_area = f['acquisition']['EyeTracking']['pupil_tracking']['area'][()]

# Pass 3: everything (including running + pupil again)
raw_data = load_experiment_data(nwb_path, eid)
```

iii. The AI acknowledged "Multiple passes over NWB files" as a code inefficiency but chose this approach for cleaner code organization.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several processing steps produce data that may not be fully utilized:
1. The `cell_specimen_ids` for valid ROIs are extracted but not stored in the final output.
2. The `change_time` from trials is loaded but never used (image change is computed from stimulus presentations instead).
3. The `initial_image_name` and `change_image_name` from trials are loaded but not used.
4. The `stim_data['stop_time']` (stimulus stop times) is loaded and `change_stops` is computed in `get_image_change_at_timepoints` but not actually used (a hardcoded 0.75s window is used instead).
5. The `input` field in the output is populated with empty arrays `np.zeros((0, n_tp))` for every trial — these are created and stored but carry no information.

ii.
```python
# Unused variables loaded:
'change_time': trials_grp['change_time'][()],
'initial_image_name': trials_grp['initial_image_name'][()],
'change_image_name': trials_grp['change_image_name'][()],
cell_specimen_ids = cell_table['cell_specimen_id'][()]

# Empty input arrays:
session_input.append(np.zeros((0, n_tp), dtype=np.float32))
```

iii. The AI loaded these variables for potential use in sanity checks but they are not part of the final output.
