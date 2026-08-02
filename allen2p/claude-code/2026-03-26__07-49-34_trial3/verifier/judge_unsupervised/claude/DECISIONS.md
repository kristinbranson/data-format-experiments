# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files using h5py (not through the AllenSDK API). It reads the `ophys_experiment_table.csv` metadata CSV to determine which experiments to process, filters to only active session types (OPHYS_1, OPHYS_3, OPHYS_4, OPHYS_6), and checks which NWB files exist on disk. It then iterates through each qualifying experiment, opening the NWB file via h5py and extracting neural events, trials, stimulus presentations, running speed, and pupil tracking data. A 3-pass approach is used: (1) collect global image names, (2) compute global percentile bin edges for running speed and pupil area, (3) full processing of each experiment.

ii.
```python
# From get_experiment_list():
exp_table = pd.read_csv(META_DIR / 'ophys_experiment_table.csv')
nwb_files = list(NWB_DIR.glob('*.nwb'))
# Filter to on-disk, active sessions
mask = (
    exp_table['ophys_experiment_id'].isin(nwb_ids) &
    exp_table['session_type'].isin(ACTIVE_SESSION_TYPES)
)

# From load_experiment_data():
with h5py.File(nwb_path, 'r') as f:
    cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
    ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][()]
    events_data = f['processing']['ophys']['event_detection']['data'][()]
    # ... trials, stim, running, pupil ...
```

iii. The AI chose h5py over AllenSDK because the SDK functions are primarily wrappers around h5py reads. The 3-pass approach was used to compute global statistics before per-experiment processing. The AI documented that only 284 NWB files were available on disk (a subset of the full 1,936 experiments), yielding 202 active experiments from 38 mice.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by `mouse_id` from the `ophys_experiment_table.csv` metadata. The unique mouse IDs are collected, sorted, and mapped to integer indices. Each experiment/session is associated with its mouse via the metadata table.

ii.
```python
subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
# ...
all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])
```

iii. The AI identified 38 unique subjects in the on-disk subset (vs 82 in the full dataset). Subject IDs are stored as strings in the output.

## 1-c. How are the data split into sessions?

i. Each NWB experiment file is treated as one "session" in the output format. For multi-plane (Multiscope) recordings, each imaging plane is a separate NWB file and thus a separate session. This yields 202 sessions from 174 biological sessions across 38 mice.

ii.
```python
# Each iteration of the main loop = one session
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = NWB_DIR / f'behavior_ophys_experiment_{eid}.nwb'
    # ... process single experiment ...
    all_neural.append(session_neural)
```

iii. The AI noted that "Each NWB experiment = one 'session' in output format (one imaging plane with its own neurons)." This means multi-plane sessions from Multiscope rigs result in multiple output sessions, each with its own set of neurons.

## 1-d. How are the data split into trials?

i. Trials are defined by the `intervals/trials` group in each NWB file, which provides `start_time` and `stop_time` for each trial. Neural and behavioral data are segmented using these time boundaries against the resampled 30Hz regular timestamp grid.

ii.
```python
# From process_single_experiment():
for trial_idx in valid_indices:
    t_trial_start = trials['start_time'][trial_idx]
    t_trial_stop = trials['stop_time'][trial_idx]
    trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
    trial_time_indices = np.where(trial_mask)[0]
    if len(trial_time_indices) < 3:
        continue
    trial_ts = regular_ts[trial_time_indices]
```

iii. The AI used the trial table's start/stop times directly, resulting in variable-length trials (mean ~254 timepoints = ~8.5s at 30Hz). Trials shorter than 3 timepoints are skipped. The trial window spans from trial start to trial stop, not aligned to any specific event within the trial.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to include only Go and Catch trials, excluding Aborted and Auto-rewarded trials per the task instructions. Additionally, trials with unknown outcomes (none of hit/miss/false_alarm/correct_reject) and very short trials (<3 timepoints) are skipped.

ii.
```python
valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
valid_indices = np.where(valid_mask)[0]
# ... later in the loop:
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

iii. The AI followed the instructions exactly: "Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials." The additional skip for unknown outcomes is a safety measure. The resulting Go/Catch split was 87.4%/12.6%, matching the expected 87.5%/12.5%.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the pre-computed calcium event detection signal (`event_detection/data`) in the NWB files, filtered by the `valid_roi` mask from the cell specimen table.

ii.
```python
events_data = f['processing']['ophys']['event_detection']['data'][()]
valid_roi = cell_table['valid_roi'][()].astype(bool)
events_valid = events_data[:, valid_roi]
```

iii. The AI chose events over dF/F because the reference paper states "We performed our analyses on discrete calcium events." Events are derived from FastLZeroSpikeInference with L0 regularization, which produces spike-like deconvolved signals.

## 2-b. How is the `neural` data processed?

i. The raw events (at irregular ophys timestamps, ~31Hz for Scientifica, ~11Hz for Multiscope) are linearly interpolated onto a regular 30Hz grid spanning the entire session. After interpolation, values are clipped to be non-negative (events should be >= 0). Per-trial segments are then extracted and transposed to (n_neurons, n_timepoints) format.

ii.
```python
regular_ts = np.arange(t_start, t_end, dt)  # dt = 1/30
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
# Per trial:
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The AI justified 30Hz resampling by citing the paper: "linearly interpolating onto a consistent set of 30hz timestamps." The non-negativity clipping ensures interpolation doesn't produce negative event magnitudes. No additional filtering (e.g., the SDK's optional half-Gaussian smoothing) is applied to the events.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only neurons (ROIs) with `valid_roi == True` are included. This is a binary SVM classifier output from the AllenSDK that identifies well-segmented cells vs. artifacts. No additional neuron quality filtering (e.g., by signal-to-noise ratio or responsiveness) is applied.

ii.
```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
events_valid = events_data[:, valid_roi]
n_valid = valid_roi.sum()
if n_valid == 0:
    return None  # skip experiment
```

iii. The AI documented that `valid_roi` is the only automatic filter in the AllenSDK: "Filter cells by valid_roi boolean. exclude_invalid_rois=True by default." Sessions with zero valid ROIs are skipped entirely.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the ophys timestamps (the imaging clock). The instructions say "Temporally align based on ophys timestamp," and trials are segmented by the trial start_time and stop_time from the trials table. There is no alignment to a specific within-trial event (e.g., stimulus change time); the full trial window is used.

ii.
```python
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
trial_ts = regular_ts[trial_time_indices]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The AI set `off_start=0.0` (alignment to trial start) and `off_end=None` (variable trial length). The metadata states `temporal_alignment_event = 'Trial start time (first stimulus onset of trial)'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 33.33 ms (30 Hz). Data is linearly interpolated from native sampling rates (~31Hz Scientifica, ~11Hz Multiscope) to a uniform 30Hz grid. This is resampling via interpolation, not binning/averaging.

ii.
```python
TARGET_RATE_HZ = 30.0
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ  # ~33.33 ms
dt = 1.0 / target_rate
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
```

iii. The AI cited the paper: "linearly interpolating onto a consistent set of 30hz timestamps." No temporal binning (summing/averaging over windows) is applied; only point-wise linear interpolation is used.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table in the NWB file, specifically the `image_name` field from the `intervals/<stimulus_key>` group, along with the `start_time` and `omitted` fields.

ii.
```python
stim_data = {
    'start_time': stim['start_time'][()],
    'stop_time': stim['stop_time'][()],
    'image_name': stim['image_name'][()],
    'omitted': stim['omitted'][()],
}
```

iii. The AI documented that each session uses 8 natural images, and across the full dataset there are 16 unique images (2 image sets of 8).

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each timepoint in a trial, the most recent non-omitted stimulus onset is found using `np.searchsorted`. The image name at that onset is mapped to a global categorical index. During gray screen periods and omitted flashes, the identity of the last presented image is carried forward.

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

iii. The AI decided to carry forward the last image identity during gray screens and omitted flashes rather than using a special "no image" category, reasoning this provides a cleaner representation for the decoder.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is evaluated at the same 30Hz regular timestamps used for the neural data within each trial window. The searchsorted operation finds the most recent stimulus for each neural timepoint.

ii.
```python
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
```

iii. Both neural and image identity share the same regular_ts timebase, ensuring alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus presentations table, specifically the `is_change` and `omitted` flags, along with `start_time` and `stop_time`.

ii.
```python
change_mask = stim_data['is_change'] & ~stim_data['omitted']
change_starts = stim_data['start_time'][change_mask]
change_stops = stim_data['stop_time'][change_mask]
```

iii. The AI used the stimulus-level `is_change` flag rather than the trial-level flag, allowing precise temporal marking of when changes occur.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary signal is constructed: for each change stimulus presentation (non-omitted, is_change=True), timepoints within a 750ms window starting at the change onset are marked as 1; all other timepoints are 0.

ii.
```python
def get_image_change_at_timepoints(timepoints, stim_data):
    change_signal = np.zeros(n_tp, dtype=np.int64)
    change_mask = stim_data['is_change'] & ~stim_data['omitted']
    change_starts = stim_data['start_time'][change_mask]
    for cs, ce in zip(change_starts, change_stops):
        mask = (timepoints >= cs) & (timepoints < cs + 0.75)
        change_signal[mask] = 1
    return change_signal
```

iii. The AI chose a 750ms window (250ms stimulus + 500ms gray screen = one full flash interval) to mark change events. This captures the entire image presentation interval following the change.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 = no change, 1 = change). No thresholding is needed.

ii.
```python
img_change = trial_data_out['image_change'].astype(np.int64)
# output_values: ['no_change', 'change']
```

iii. The binary representation directly follows the instruction: "Have value of 1 right after a change in image identity, otherwise 0."

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is evaluated at the same 30Hz regular timestamps as the neural data within each trial.

ii.
```python
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
```

iii. Same timebase as neural data ensures alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB files.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. Running speed is recorded at ~60Hz and measured in cm/s.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native ~60Hz timestamps to the regular 30Hz grid. Then, 5 equal percentile bin edges are computed globally across all experiments (using all running speed values from all sessions). Each timepoint's running speed is then digitized into one of 5 bins (0-4).

ii.
```python
# Resample to 30Hz
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])

# Global percentile bins (computed in Pass 2)
all_running_cat = np.concatenate(all_running_values)
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)

# Digitize
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. The AI chose global percentile binning to ensure consistent bin definitions across all sessions. The resulting bin edges were approximately: [-24.1, -0.004, 0.336, 14.2, 33.1, 99.9].

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using equal percentile boundaries computed globally. `np.digitize` assigns each value to a bin index (0-4). Values below the lowest or above the highest edge are clipped to the nearest bin.

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
    binned[np.isnan(values)] = 0
    return binned.astype(np.int64)
```

iii. The instructions specify "discretized into five equal percentile bins," which is what the code implements.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is resampled to the same 30Hz regular grid as neural data and extracted at the same trial timepoints.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
running_trial = running_resampled[trial_time_indices]
```

iii. Same timebase ensures alignment with neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/area` and `acquisition/EyeTracking/likely_blink/data` in the NWB files. Note: the raw variable is pupil **area**, not diameter, though the output is named "pupil_diameter."

ii.
```python
pupil_area = f['acquisition']['EyeTracking']['pupil_tracking']['area'][()]
pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][()]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
```

iii. The AI acknowledged using pupil area as a proxy for diameter in CONVERSION_NOTES.md: "Use pupil area as proxy for diameter."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames (identified by `likely_blink`) are set to NaN. NaN values are then linearly interpolated. The cleaned signal is resampled to 30Hz. Finally, 5 equal percentile bin edges are computed globally (from non-blink data across all sessions) and applied to digitize each timepoint.

ii.
```python
pupil_area = raw_data['pupil_area'].copy().astype(float)
if likely_blink is not None:
    pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
# Binning:
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. Blink handling involves removing blink frames and interpolating, which is a standard approach. Global percentile binning is used for consistency across sessions.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 equal percentile bins computed globally from non-blink pupil area values across all experiments. `digitize_to_bins` assigns each value to a bin index 0-4. NaN values (if any remain) default to bin 0.

ii.
```python
# Global bins computed in Pass 2:
all_pupil_cat = np.concatenate(all_pupil_values)
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
# Applied per trial:
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. Bin edges were approximately: [125.6, 4374.1, 5527.6, 6747.1, 8598.7, 323783.3].

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil area is resampled to the same 30Hz regular grid as neural data and extracted at the same trial timepoints.

ii.
```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
pupil_trial = pupil_resampled[trial_time_indices]
```

iii. Same timebase ensures alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from four boolean fields in the NWB trials table: `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
trials = {
    'hit': trials_grp['hit'][()].astype(bool),
    'miss': trials_grp['miss'][()].astype(bool),
    'false_alarm': trials_grp['false_alarm'][()].astype(bool),
    'correct_reject': trials_grp['correct_reject'][()].astype(bool),
}
```

iii. These are standard behavioral outcome categories for the Visual Behavior task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Each trial is assigned a single categorical outcome (0=hit, 1=miss, 2=false_alarm, 3=correct_reject) based on which boolean flag is True. This static value is then broadcast to all timepoints in the trial.

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

# Broadcast to all timepoints:
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)
```

iii. The instructions specify trial outcome as "Static per-trial," but the AI broadcasts it to all timepoints in the combined output array to maintain consistent (n_output, n_timepoints) shape across all output variables.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several types of missing/problematic data are handled:
- **Missing pupil data**: If no eye tracking data exists for an experiment, pupil values are filled with NaN (which defaults to bin 0 after digitization).
- **Blinks**: Identified via `likely_blink` flag, set to NaN, then linearly interpolated before resampling.
- **Negative events after interpolation**: Clipped to zero with `np.maximum(events_resampled, 0)`.
- **Unknown trial outcomes**: Trials where none of hit/miss/FA/CR is True are skipped.
- **Very short trials**: Trials with fewer than 3 timepoints are skipped.
- **Omitted stimuli**: Omitted flash presentations are excluded from image identity assignment; the previous image identity is carried forward.
- **Missing stimulus presentations**: Experiments with no stimulus presentation table are skipped.
- **Sessions with no valid ROIs**: Skipped entirely.

ii.
```python
# Missing pupil: fill NaN
pupil_trial = np.full(n_tp, np.nan)

# Blinks: NaN then interpolate
pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)

# Negative events
events_resampled = np.maximum(events_resampled, 0)

# Short trials
if len(trial_time_indices) < 3:
    continue
```

iii. The AI documented these handling strategies in CONVERSION_NOTES.md and verified via sanity checks that no NaN values remain in the final neural data.

## 9-a. What are the most time-consuming steps of the code?

i. Based on the timing output, the most time-consuming step is the main processing pass (Pass 3), which takes ~0.4-0.9 seconds per experiment. Within each experiment, loading the NWB file via h5py (~0.2-0.4s) and processing trials (~0.2-0.5s) are roughly equal. The total conversion for 202 experiments took ~358 seconds (6 minutes). The percentile data collection pass (Pass 2) also requires opening every NWB file once.

ii.
```python
# Timing per experiment:
t0 = time.time()
raw_data = load_experiment_data(nwb_path, eid)
t_load = time.time() - t0
result = process_single_experiment(raw_data, exp_meta)
t_process = time.time() - t0 - t_load
```

iii. The AI estimated ~0.5s per session and 2-3 minutes total, which was close to the actual 6 minutes.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
- The loop in `get_image_at_timepoints` that iterates over all timepoints to map local indices to image names could be fully vectorized using numpy fancy indexing.
- The loop in `get_image_change_at_timepoints` that iterates over each change event could be vectorized.
- The local-to-global image index mapping loop could use a vectorized lookup.
- The `interpolate_to_regular_grid` function loops over columns for multi-dimensional data.

ii.
```python
# get_image_at_timepoints: per-timepoint loop
for i in range(n_tp):
    if insert_idx[i] >= 0:
        img_name = stim_names[insert_idx[i]]
        image_idx[i] = name_to_idx.get(img_name, 0)

# Local-to-global mapping loop
img_id_global = np.array([local_to_global.get(v, 0) for v in trial_data_out['image_identity']], dtype=np.int64)

# Column-wise interpolation loop
for i in range(data.shape[1]):
    result[:, i] = np.interp(target_timestamps, timestamps, data[:, i])
```

iii. These loops are not major bottlenecks given the data sizes, but vectorization would improve clarity and speed for larger datasets.

## 9-c. What processing does the code repeat multiple times?

i. The code opens each NWB file 3 times across the 3 passes: once to collect image names, once to collect running/pupil statistics, and once for full processing. The running speed and pupil area data are read twice (once in Pass 2 for global stats, once in Pass 3 for per-trial extraction). Pupil blink detection and NaN handling also occurs both in Pass 2 (for computing bin edges) and Pass 3 (for per-experiment processing).

ii.
```python
# Pass 1: Open NWB to read image names
with h5py.File(nwb_path, 'r') as f:
    img_names = f['intervals'][k]['image_name'][()]

# Pass 2: Open NWB to read running/pupil
with h5py.File(nwb_path, 'r') as f:
    running = f['processing']['running']['speed']['data'][()]
    pupil_area = f['acquisition']['EyeTracking']['pupil_tracking']['area'][()]

# Pass 3: Open NWB again for full loading
raw_data = load_experiment_data(nwb_path, eid)
```

iii. The AI acknowledged this inefficiency but noted total conversion time (6 min) was well within acceptable limits.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several processing steps produce data that may not be used downstream:
- The `input` field is populated with empty (0, n_timepoints) arrays for every trial since the task specifies "No inputs." This creates overhead in storage and iteration.
- Trial outcome is broadcast to all timepoints despite being a static per-trial variable. The decoder could use a single value per trial.
- The pupil data NaN interpolation and resampling is done for all experiments even when pupil data may not be meaningfully available.
- Cell specimen IDs are extracted and filtered but not stored in the final output.

ii.
```python
# Empty input arrays for every trial
session_input.append(np.zeros((0, n_tp), dtype=np.float32))

# Trial outcome broadcast to all timepoints
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)

# Cell IDs extracted but not in output
'cell_specimen_ids': cell_specimen_ids[valid_roi],  # in raw_data but not in final dict
```

iii. These are minor inefficiencies. The empty input arrays and trial outcome broadcasting are needed for format consistency with the target data structure.
