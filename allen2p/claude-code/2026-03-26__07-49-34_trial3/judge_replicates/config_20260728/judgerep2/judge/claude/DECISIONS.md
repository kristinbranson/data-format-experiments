# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files on disk using h5py, rather than through the AllenSDK's `VisualBehaviorOphysProjectCache`. It reads experiment metadata from a CSV table (`ophys_experiment_table.csv`), determines which NWB files are available on disk, filters to active session types (OPHYS_1, 3, 4, 6), and then processes each NWB file individually. A 3-pass approach is used: (1) scan all NWB files for global image names, (2) collect running/pupil data for global percentile bins, (3) full processing of each experiment.

ii.
```python
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

iii. The AI chose h5py over the AllenSDK cache to avoid SDK overhead. It filters to active session types because passive sessions (OPHYS_2, OPHYS_5) have no lick spout and no meaningful trial outcomes.

## 1-b. How are the data split into subjects?

i. Subjects are identified by unique `mouse_id` values in the experiment table CSV. Each unique mouse_id becomes a subject.

ii.
```python
subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The `mouse_id` field is the standard unique identifier for each animal in the Allen metadata.

## 1-c. How are the data split into sessions?

i. Each NWB experiment file (single imaging plane) is treated as a separate "session" in the output. The AI does NOT group experiments by `ophys_session_id` — multi-plane sessions are kept as separate sessions, each with its own set of neurons.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = NWB_DIR / f'behavior_ophys_experiment_{eid}.nwb'
    ...
    raw_data = load_experiment_data(nwb_path, eid)
    result = process_single_experiment(raw_data, exp_meta)
    ...
    all_neural.append(session_neural)
```

iii. From CONVERSION_NOTES.md: "Each NWB experiment = one 'session' in output format (one imaging plane with its own neurons)." This means multi-plane sessions from the Mesoscope are split into multiple output sessions.

## 1-d. How are the data split into trials?

i. Trials are defined using the NWB trials table (`/intervals/trials`). For each valid trial, the time window from `start_time` to `stop_time` is used. After resampling to 30Hz, timepoints within the trial window are extracted.

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

iii. The trial window from `start_time` to `stop_time` includes both pre-change stimulus flashes and the post-change response window, giving variable-length trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to include only Go and Catch trials, excluding Aborted and Auto-rewarded trials. Additionally, trials with fewer than 3 timepoints at 30Hz are skipped. Trials where the outcome is not one of hit/miss/false_alarm/correct_reject are also skipped. Experiments with fewer than 2 valid processed trials are excluded.

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

iii. Per instructions, aborted and auto-rewarded trials are excluded. The go|catch filter is equivalent to excluding aborted/auto-rewarded since the trial types are mutually exclusive.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `event_detection/data` — the pre-computed calcium events from FastLZeroSpikeInference — NOT from dF/F traces.

ii.
```python
events_data = f['processing']['ophys']['event_detection']['data'][()]
events_valid = events_data[:, valid_roi]  # (n_timepoints, n_valid_cells)
```

iii. From CONVERSION_NOTES.md: "Paper says 'We performed our analyses on discrete calcium events.' Use raw events from event_detection." The AI chose events over dF/F because the paper used events for their analyses.

## 2-b. How is the `neural` data processed?

i. Events data is filtered by `valid_roi`, then linearly interpolated from native ophys timestamps to a regular 30Hz grid. After interpolation, values are clipped to be non-negative.

ii.
```python
events_valid = events_data[:, valid_roi]
...
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
```

iii. The 30Hz resampling matches what the paper describes: "linearly interpolating onto a consistent set of 30hz timestamps." Clipping to non-negative ensures events remain physically meaningful.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by the `valid_roi` boolean flag from the cell specimen table. Only neurons with `valid_roi == True` are included.

ii.
```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
...
events_valid = events_data[:, valid_roi]
```

iii. The `valid_roi` flag is the output of the SDK's SVM binary classifier that filters out invalid ROIs (unions, duplicates, edge/motion artifacts, etc.). This matches the SDK's default behavior of `exclude_invalid_rois=True`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start time (`start_time` from the trials table). The regular 30Hz timestamps within each trial's `[start_time, stop_time)` window are extracted.

ii.
```python
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The instructions specify "temporally align based on ophys timestamp." The AI aligns to trial start, which is consistent with using the ophys timebase.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is resampled to 30Hz (33.33ms time bins). This involves linear interpolation from the native ophys frame rate (~11Hz Multiscope, ~31Hz Scientifica) to a consistent 30Hz grid.

ii.
```python
TARGET_RATE_HZ = 30.0
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ  # ~33.33 ms
...
dt = 1.0 / target_rate
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
```

iii. From CONVERSION_NOTES.md: "Paper interpolates to 30 Hz. Needed for consistent time bins across Scientifica (31 Hz) and Multiscope (11 Hz)."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table (not the trials table). For each timepoint, the most recent non-omitted stimulus onset determines the current image.

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

iii. Using the stimulus presentations table provides precise image onset timing for every flash, including during gray screen intervals between flashes.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer indices via a global mapping built from all unique non-omitted image names across all experiments. The mapping is sorted alphabetically. During gray screen periods between flashes, the identity of the most recently shown image is used. Omitted flashes continue with the previous image identity.

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

iii. A global mapping ensures consistent integer codes across all sessions. 16 unique images are found (two sets of 8 images each, from different session types).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the same 30Hz regular timestamps as the neural data, using `np.searchsorted` to find the most recent stimulus onset for each timepoint.

ii.
```python
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
```

iii. Both neural and image identity share the same regular timestamp grid, ensuring alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus presentations table, specifically using the `is_change` flag combined with the `omitted` flag and stimulus timing.

ii.
```python
def get_image_change_at_timepoints(timepoints, stim_data):
    change_mask = stim_data['is_change'] & ~stim_data['omitted']
    change_starts = stim_data['start_time'][change_mask]
    change_stops = stim_data['stop_time'][change_mask]
```

iii. The stimulus presentations table's `is_change` flag identifies which flashes were actual image changes.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each change stimulus (is_change=True, non-omitted), timepoints within a 750ms window from the change onset are marked as 1, all others as 0.

ii.
```python
for cs, ce in zip(change_starts, change_stops):
    mask = (timepoints >= cs) & (timepoints < cs + 0.75)
    change_signal[mask] = 1
```

iii. The 750ms window corresponds to one stimulus presentation (250ms) plus the gray inter-stimulus interval (500ms).

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is binary (0 or 1). No thresholding is needed — it is directly computed as a binary indicator.

ii. See 4-b code above.

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same 30Hz regular timestamp grid as neural data.

ii.
```python
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
```

iii. Computed at the same timepoints as neural and other output variables.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. This is the standard running speed data from the Allen SDK, measured in cm/s from the running wheel encoder.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the 30Hz regular grid using `np.interp`, then discretized into 5 equal percentile bins. Bin edges are computed globally across all experiments.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
...
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. Linear interpolation resamples to 30Hz. Global percentile bins ensure consistent categories across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Discretized into 5 bins using percentile-based edges computed from all running speed data across all experiments. `np.digitize` maps values to bin indices 0-4. NaN values are mapped to bin 0.

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

iii. Percentile-based binning ensures roughly equal class counts. The epsilon handling for non-unique edges prevents edge cases with constant values.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the same 30Hz regular timestamp grid as neural data before trial segmentation, then indexed by the same trial mask.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
running_trial = running_resampled[trial_time_indices]
```

iii. Using the same regular timestamp grid guarantees temporal alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/area` — pupil AREA, not pupil width or diameter. Blinks are identified from `acquisition/EyeTracking/likely_blink/data`.

ii.
```python
pupil_tracking = f['acquisition']['EyeTracking']['pupil_tracking']
pupil_area = pupil_tracking['area'][()]
pupil_ts = pupil_tracking['timestamps'][()]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
```

iii. From CONVERSION_NOTES.md: "Use pupil area as proxy for diameter."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames (`likely_blink=True`) are set to NaN. NaN values are then linearly interpolated (filling gaps) before resampling to 30Hz. The resampled data is discretized into 5 percentile bins computed globally.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)  # fill NaN gaps
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. Blinks are removed and interpolated to avoid artifacts. Percentile bins from all data ensure consistent categories.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: 5 equal percentile bins using `compute_percentile_bins` and `digitize_to_bins`. NaN values mapped to bin 0.

ii. See 5-c code snippets — same functions are used.

iii. Same approach as running speed for consistency.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — interpolated to the 30Hz grid before trial segmentation, then indexed by the same trial mask.

ii.
```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
pupil_trial = pupil_resampled[trial_time_indices]
```

iii. Same alignment approach as all other signals.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB trials table.

ii.
```python
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

iii. These four outcomes are the canonical trial outcome categories for the change detection task. Trials not matching any category are skipped.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_reject) and broadcast as a constant across all timepoints in the trial.

ii.
```python
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)
```

iii. Trial outcome is static per-trial, so it is constant across all timepoints.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Failed experiments**: If `load_experiment_data` fails, returns None and is skipped.
- **No valid ROIs**: Experiments with no valid_roi neurons are skipped.
- **Missing pupil data**: If eye tracking is not available, pupil is filled with NaN.
- **Pupil blinks**: Set to NaN and linearly interpolated before resampling.
- **Short trials**: Trials with < 3 timepoints at 30Hz are skipped.
- **Unknown outcomes**: Trials without a recognized outcome are skipped.
- **Few trials**: Experiments with < 2 valid trials are skipped.

ii.
```python
if raw_data is None:
    continue
...
if n_valid == 0:
    return None
...
except (KeyError, Exception):
    print(f"  WARNING: No pupil tracking data in {experiment_id}")
...
pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)
...
if len(trial_time_indices) < 3:
    continue
```

iii. These handlers prevent individual data issues from crashing the pipeline.

## 9-a. What are the most time-consuming steps of the code?

i. The code makes 3 passes over all NWB files: (1) scanning for image names, (2) collecting running/pupil data for percentile bins, (3) full processing. Each pass involves opening and reading from NWB files via h5py.

ii.
```python
# Pass 1
for _, row in exp_table.iterrows():
    with h5py.File(nwb_path, 'r') as f:
        ...  # scan image names

# Pass 2
for idx, (_, row) in enumerate(exp_table.iterrows()):
    with h5py.File(nwb_path, 'r') as f:
        running = f['processing']['running']['speed']['data'][()]
        ...

# Pass 3
for idx, (_, row) in enumerate(exp_table.iterrows()):
    raw_data = load_experiment_data(nwb_path, eid)
    result = process_single_experiment(raw_data, exp_meta)
```

iii. From timing output, each experiment takes ~0.4-0.7s total. Full conversion takes ~6 minutes for 202 experiments across 3 passes.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `get_image_at_timepoints` function uses a Python loop over all timepoints to map searchsorted indices to image names:
```python
for i in range(n_tp):
    if insert_idx[i] >= 0:
        img_name = stim_names[insert_idx[i]]
        image_idx[i] = name_to_idx.get(img_name, 0)
```
This could be fully vectorized using array indexing.

ii. See above.

iii. With ~250 timepoints per trial, this loop is not a major bottleneck but could be vectorized.

## 9-c. What processing does the code repeat multiple times?

i. The code reads each NWB file 3 times across the 3 passes: once for image names, once for running/pupil statistics, and once for full processing. Pass 1 and Pass 2 could be combined, and the statistics collection in Pass 2 reads the full running speed and pupil data that is then re-read in Pass 3.

ii. See 9-a code snippets for the 3-pass structure.

iii. The multi-pass approach was likely chosen for code clarity and to compute global statistics before the main processing pass.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes percentile bins from ALL running speed and pupil data across entire sessions (including non-trial periods), but the bins are only applied to trial-segmented data. The non-trial data contributes to bin edge computation but is otherwise discarded. Additionally, the local-to-global image name mapping involves creating per-experiment image name lists that are then mapped to global indices.

ii.
```python
# Pass 2 collects ALL running speed data, not just trial periods
running = f['processing']['running']['speed']['data'][()]
all_running_values.append(running.astype(np.float32))
```

iii. Computing bins from all data vs. trial-only data may produce slightly different bin edges.
