# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files on disk using `h5py`, rather than through the AllenSDK's `VisualBehaviorOphysProjectCache`. It reads a metadata CSV (`ophys_experiment_table.csv`) to discover experiments, then filters to NWB files available on disk. It also filters to only "active" session types (OPHYS_1, OPHYS_3, OPHYS_4, OPHYS_6), excluding passive sessions (OPHYS_2, OPHYS_5). A 3-pass approach is used: (1) collect global image names, (2) compute global percentile bins, (3) process experiments.

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

iii. The AI chose to use h5py directly for faster loading, and filtered to active session types because passive sessions lack meaningful trial outcomes (mice are satiated, no lick spout). This is documented in CONVERSION_NOTES.md Step 4/5.

## 1-b. How are the data split into subjects?

i. Subjects correspond to unique `mouse_id` values from the experiment metadata table, sorted and converted to strings.

ii.
```python
subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The `mouse_id` field is the standard identifier for each animal in the Allen metadata.

## 1-c. How are the data split into sessions?

i. Each NWB experiment (one imaging plane) is treated as a separate "session" in the output format. The AI does NOT group experiments by `ophys_session_id` — each single-plane experiment is its own session.

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

iii. The AI notes in CONVERSION_NOTES.md Step 5 Key Decision 13: "Each NWB experiment = one 'session' in output format (one imaging plane with its own neurons)."

## 1-d. How are the data split into trials?

i. Trials are defined using the `intervals/trials` group from the NWB file. Each trial goes from `start_time` to `stop_time`, giving variable-length trials. Go and Catch trials are included, while Aborted and Auto-rewarded trials are excluded. The data is resampled to a regular 30 Hz grid, and trial time indices are found where `regular_ts >= start_time` and `regular_ts < stop_time`.

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

iii. The AI followed the instructions to include Go and Catch trials and exclude Aborted and Auto-rewarded trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) Must be Go or Catch (not Aborted, not Auto-rewarded), (2) Must have at least 3 timepoints after resampling, (3) Must have a known trial outcome (hit/miss/false_alarm/correct_reject — trials with unknown outcome are skipped), (4) Sessions with fewer than 2 valid trials are skipped.

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

iii. The filtering is consistent with the instructions. The AI explicitly requires `go | catch` rather than just `~aborted & ~auto_rewarded`, which is slightly more restrictive but functionally equivalent for this dataset. Trials with unknown outcomes are skipped as a safety measure.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `event_detection/data` in the NWB files — these are calcium events computed by the FastLZeroSpikeInference algorithm. Only neurons with `valid_roi == True` are included.

ii.
```python
events_data = f['processing']['ophys']['event_detection']['data'][()]
events_valid = events_data[:, valid_roi]  # (n_timepoints, n_valid_cells)
```

iii. The AI chose events over dF/F based on the paper's statement "We performed our analyses on discrete calcium events." This is documented in CONVERSION_NOTES.md Step 5 Key Decision 1.

## 2-b. How is the `neural` data processed?

i. Neural events are resampled from their native ophys timestamps to a regular 30 Hz grid via linear interpolation. After interpolation, values are clipped to be non-negative (`np.maximum(events_resampled, 0)`).

ii.
```python
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
```

iii. The 30 Hz resampling follows the paper's processing: "linearly interpolating onto a consistent set of 30hz timestamps." Non-negativity clipping ensures events remain physically meaningful.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data is filtered by the `valid_roi` flag from the cell specimen table. Only neurons where `valid_roi == True` are included. Experiments with 0 valid ROIs are skipped entirely.

ii.
```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
...
n_valid = valid_roi.sum()
if n_valid == 0:
    print(f"  WARNING: No valid ROIs in experiment {experiment_id}, skipping")
    return None
events_valid = events_data[:, valid_roi]
```

iii. The `valid_roi` flag is an SVM binary classifier output from the Allen pipeline that identifies high-quality ROIs. This is documented in CONVERSION_NOTES.md Step 3.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial start time. After resampling to a regular 30 Hz grid, the trial's timepoints are selected as those where `regular_ts >= start_time` and `regular_ts < stop_time`.

ii.
```python
regular_ts = np.arange(t_start, t_end, dt)
...
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The alignment to trial start preserves the full trial window including pre-change stimulus flashes and post-change response period.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is resampled to 30 Hz (time bin size = 33.33 ms). This is a resampling from the native ophys frame rate (~31 Hz for Scientifica, ~11 Hz for Multiscope) to a consistent 30 Hz grid.

ii.
```python
TARGET_RATE_HZ = 30.0
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ  # ~33.33 ms
...
dt = 1.0 / target_rate
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
```

iii. The paper states: "linearly interpolating onto a consistent set of 30hz timestamps." This ensures consistent time bins across different equipment types (Scientifica vs Multiscope).

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table in the NWB file, specifically the `image_name`, `start_time`, `stop_time`, `is_change`, and `omitted` fields. For each timepoint, the most recent non-omitted stimulus onset determines the current image.

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
```

iii. The AI uses the stimulus_presentations table to determine which image is shown at each timepoint, accounting for omitted flashes by continuing with the previous image identity.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global mapping built from all unique image names across all experiments (excluding 'omitted'). For each timepoint, the most recent non-omitted stimulus onset is found using `np.searchsorted`, and the corresponding image name is looked up. Local per-experiment image indices are then mapped to global indices.

ii.
```python
all_stim_images = sorted(set(
    raw_data['stim']['image_name'][raw_data['stim']['image_name'] != 'omitted']
))
...
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
...
local_to_global = {}
for local_idx, name in enumerate(result['all_image_names']):
    if name in global_image_names:
        local_to_global[local_idx] = global_image_names.index(name)
img_id_global = np.array([local_to_global.get(v, 0) for v in trial_data_out['image_identity']], dtype=np.int64)
```

iii. The two-step mapping (local then global) ensures consistent image codes across experiments, even though different experiments may use different image sets.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at the same 30 Hz regular grid timepoints as the neural data, using the same `trial_time_indices`. Both share the `regular_ts` timebase.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. Alignment is guaranteed because both neural and image identity use the same indices into the resampled timebase.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus_presentations table, specifically the `is_change` and `omitted` fields, plus `start_time` and `stop_time` of change stimuli.

ii.
```python
def get_image_change_at_timepoints(timepoints, stim_data):
    change_mask = stim_data['is_change'] & ~stim_data['omitted']
    change_starts = stim_data['start_time'][change_mask]
    change_stops = stim_data['stop_time'][change_mask]
```

iii. The AI uses `is_change` from the stimulus table rather than the `go` flag from the trials table. For go trials these are equivalent, but for catch trials `is_change` would not mark them (correct behavior since no actual image change occurs).

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary signal is computed: 1 during a 750ms window starting at the change onset time, 0 otherwise. Only non-omitted change stimuli are marked.

ii.
```python
for cs, ce in zip(change_starts, change_stops):
    mask = (timepoints >= cs) & (timepoints < cs + 0.75)
    change_signal[mask] = 1
```

iii. The 750ms window covers one stimulus flash (250ms) + gray inter-stimulus interval (500ms).

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 = no change, 1 = change), so no additional thresholding is needed.

ii. N/A (binary by construction)

iii. The binary representation directly captures the presence/absence of an image change.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity — computed at the same 30 Hz regular grid timepoints as the neural data.

ii.
```python
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
```

iii. Both neural and image change use the same `trial_ts` = `regular_ts[trial_time_indices]`.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and `processing/running/speed/timestamps` in the NWB file.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. This is the standard running speed variable from the Allen pipeline.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is resampled from its native timestamps to the regular 30 Hz grid via `np.interp`, then discretized into 5 equal-percentile bins. Bin edges are computed globally across all experiments (in a separate pass), ignoring NaN values.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
...
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. Global percentile bins ensure consistent categories across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using percentile-based edges computed globally. `np.digitize` maps continuous values to bin indices 0-4. NaN values are mapped to bin 0. Bin edges are forced to be strictly increasing by adding epsilon (1e-10) when duplicate edges occur.

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

iii. The epsilon adjustment handles the edge case of constant values across a percentile range.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is resampled to the same 30 Hz grid and indexed with the same trial time indices as the neural data.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
running_trial = running_resampled[trial_time_indices]
```

iii. Same alignment mechanism as neural data and other outputs.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/area` and `acquisition/EyeTracking/likely_blink/data` in the NWB file.

ii.
```python
pupil_area = f['acquisition']['EyeTracking']['pupil_tracking']['area'][()]
pupil_ts = f['acquisition']['EyeTracking']['pupil_tracking']['timestamps'][()]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
```

iii. The AI uses `pupil_area` (rather than `pupil_width`). The CONVERSION_NOTES.md Step 5 states: "Use pupil area as proxy for diameter."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames (`likely_blink == True`) are set to NaN, then all NaN values are linearly interpolated (filled in). The cleaned signal is then resampled to the 30 Hz grid via `np.interp`, and discretized into 5 equal-percentile bins.

ii.
```python
pupil_area = raw_data['pupil_area'].copy().astype(float)
if likely_blink is not None:
    pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. NaN interpolation before resampling prevents blink artifacts from propagating. The AI uses a custom `interpolate_nans` function that fills NaN gaps with linear interpolation.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed — discretized into 5 percentile-based bins computed globally. NaN values mapped to bin 0.

ii.
```python
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. Percentile bins ensure roughly equal class counts.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — resampled to 30 Hz grid and indexed with the same trial time indices.

ii.
```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
pupil_trial = pupil_resampled[trial_time_indices]
```

iii. Same alignment mechanism as all other data streams.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials group of the NWB file.

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

iii. These are the SDK's canonical trial outcome labels for the change detection task.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_reject) using an if-elif chain. Trials with unknown outcomes are skipped entirely. The outcome is broadcast as a constant across all timepoints in the trial.

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

iii. The mapping order is consistent with the output_values definition.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Failed experiments**: If loading or processing throws an exception, the experiment is skipped with a warning.
- **No valid ROIs**: Experiments with 0 valid ROIs are skipped.
- **No stimulus presentations**: Experiments without a stimulus table are skipped.
- **Missing pupil data**: If eye tracking is unavailable, pupil values are set to NaN for the entire session.
- **Pupil blinks**: Set to NaN, then linearly interpolated before resampling.
- **Short trials**: Trials with fewer than 3 timepoints are skipped.
- **Unknown outcomes**: Trials without a recognized outcome (hit/miss/FA/CR) are skipped.
- **Few trials**: Experiments with fewer than 2 valid trials are skipped.
- **Duplicate bin edges**: Forced strictly increasing with epsilon.

ii.
```python
except Exception as e:
    print(f"  ERROR loading {experiment_id}: {e}")
    return None
...
if n_valid == 0:
    return None
...
if stim_key is None:
    return None
...
pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)
...
if len(trial_time_indices) < 3:
    continue
...
else:
    continue  # Unknown outcome, skip
```

iii. These are reasonable safety measures for dealing with imperfect data.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each NWB file via h5py. The AI uses a 3-pass approach, which means each NWB file is opened multiple times (once for image names, once for running/pupil stats, once for full processing).

ii. N/A (architectural observation)

iii. CONVERSION_NOTES.md reports approximately 0.25s per experiment for loading.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The `get_image_at_timepoints` function contains a Python loop over all timepoints (`for i in range(n_tp)`) that could be vectorized using numpy fancy indexing after the `searchsorted` call. The `local_to_global` mapping loop could also be vectorized.

ii.
```python
for i in range(n_tp):
    if insert_idx[i] >= 0:
        img_name = stim_names[insert_idx[i]]
        image_idx[i] = name_to_idx.get(img_name, 0)
    else:
        image_idx[i] = 0
...
img_id_global = np.array([local_to_global.get(v, 0) for v in trial_data_out['image_identity']], dtype=np.int64)
```

iii. These loops are not major bottlenecks compared to I/O but represent potential efficiency gains.

## 9-c. What processing does the code repeat multiple times?

i. NWB files are opened 3 times each: once for image name collection (Pass 1), once for running/pupil statistics (Pass 2), and once for full processing (Pass 3). The running speed and pupil data are loaded twice (passes 2 and 3).

ii.
```python
# Pass 1:
for _, row in exp_table.iterrows():
    with h5py.File(nwb_path, 'r') as f:
        # scan image names
# Pass 2:
for idx, (_, row) in enumerate(exp_table.iterrows()):
    with h5py.File(nwb_path, 'r') as f:
        # collect running/pupil stats
# Pass 3:
for idx, (_, row) in enumerate(exp_table.iterrows()):
    raw_data = load_experiment_data(nwb_path, eid)
    # full processing
```

iii. The multi-pass approach was chosen to compute global statistics before processing, but it triples the I/O.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads and stores `cell_specimen_ids` which are not used in the output. The `stim_data` includes `stop_time` for each stimulus which is loaded but not meaningfully used (the 750ms window is hardcoded from `start_time`). The `interpolate_nans` function processes the entire pupil time series even though only trial segments are used.

ii.
```python
cell_specimen_ids = cell_table['cell_specimen_id'][()]
...
'cell_specimen_ids': cell_specimen_ids[valid_roi],
```

iii. These are minor overhead items that don't significantly impact runtime.
