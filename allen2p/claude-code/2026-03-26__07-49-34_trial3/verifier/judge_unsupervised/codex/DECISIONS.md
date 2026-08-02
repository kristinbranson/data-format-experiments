# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads project metadata from CSVs under `data/visual-behavior-ophys-1.1.0/project_metadata`, enumerates NWB files present on disk, keeps only experiments whose IDs appear both in the metadata table and on disk, and then opens each NWB file directly with `h5py`. The main processing path is a 3-pass scan over experiments: one pass for global image names, one for global running/pupil bin edges, and one for full extraction.

ii. ```python
exp_table = pd.read_csv(META_DIR / 'ophys_experiment_table.csv')
...
nwb_files = list(NWB_DIR.glob('*.nwb'))
...
mask = (
    exp_table['ophys_experiment_id'].isin(nwb_ids) &
    exp_table['session_type'].isin(ACTIVE_SESSION_TYPES)
)
active_exps = exp_table[mask].copy()
```

```python
with h5py.File(nwb_path, 'r') as f:
    ...
```

```python
# First pass: determine global image names
for _, row in exp_table.iterrows():
    ...

# Pass 2: Collect running speed and pupil data
for idx, (_, row) in enumerate(exp_table.iterrows()):
    ...

# Pass 3: Processing experiments
for idx, (_, row) in enumerate(exp_table.iterrows()):
    ...
```

iii. `CONVERSION_NOTES.md` says the agent chose direct NWB loading via `h5py`, using the on-disk subset only, and described the implementation as a “3-pass approach.” The trajectory also shows it intentionally moved away from AllenSDK session grouping toward per-NWB processing.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values from the filtered experiment table; the output `subjects` list is the sorted set of mouse IDs as strings.

ii. ```python
subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])
```

iii. `CONVERSION_NOTES.md` explicitly states “Subject IDs: Use `mouse_id` from experiment table.”

## 1-c. How are the data split into sessions?

i. The agent treats each NWB experiment file, identified by `ophys_experiment_id`, as one output session. It does not group multiple experiments from the same `ophys_session_id`.

ii. ```python
def get_experiment_list(sample=False):
    """Get list of active experiment IDs with metadata."""
    exp_table = pd.read_csv(META_DIR / 'ophys_experiment_table.csv')
    ...
    return active_exps
```

```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = NWB_DIR / f'behavior_ophys_experiment_{eid}.nwb'
    ...
    all_neural.append(session_neural)
```

iii. `CONVERSION_NOTES.md` says “Each NWB experiment = one ‘session’ in output format,” and trajectory step 36 states that each imaging plane should be treated as its own decoder session.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. After trial filtering, each trial is segmented using all regular-grid samples whose timestamps satisfy `start_time <= t < stop_time`.

ii. ```python
trials_grp = f['intervals']['trials']
trials = {
    'start_time': trials_grp['start_time'][()],
    'stop_time': trials_grp['stop_time'][()],
    ...
}
```

```python
t_trial_start = trials['start_time'][trial_idx]
t_trial_stop = trials['stop_time'][trial_idx]
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
```

iii. `CONVERSION_NOTES.md` says “Trial window: Use trial `start_time` to `stop_time`. Variable length across trials.”

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept only if they are Go or Catch, not aborted, and not auto-rewarded. Trials with fewer than 3 samples on the resampled grid are dropped, and experiments with fewer than 2 remaining trials are skipped.

ii. ```python
valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
valid_indices = np.where(valid_mask)[0]
...
if len(trial_time_indices) < 3:
    continue
...
if len(neural_trials) < 2:
    ...
    return None
```

iii. `CONVERSION_NOTES.md` says “Exclude aborted and auto-rewarded trials: Per task instructions,” and the trajectory repeatedly describes keeping only Go/Catch trials from active behavior sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` output is derived from NWB event-detection traces in `processing/ophys/event_detection/data`, filtered to ROIs where `valid_roi` is true.

ii. ```python
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][()].astype(bool)
...
events_data = f['processing']['ophys']['event_detection']['data'][()]
events_valid = events_data[:, valid_roi]
```

iii. `CONVERSION_NOTES.md` says “Neural signal: events (not dF/F),” and justifies that choice as matching the paper’s use of discrete calcium events. Trajectory step 36 shows the agent explicitly switching from dF/F to events for that reason.

## 2-b. How is the `neural` data processed?

i. The agent linearly interpolates event traces from native ophys timestamps onto a regular 30 Hz grid, clips negative interpolation artifacts to zero, and then slices the resampled array into trials. No plane-merging is done because each experiment is treated as one session.

ii. ```python
ophys_ts = raw_data['ophys_ts']
events = raw_data['events']
dt = 1.0 / target_rate
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
```

```python
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. `CONVERSION_NOTES.md` says “Paper uses events, not dF/F” and “Resamples all data streams to 30 Hz via linear interpolation.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural QC applied is filtering to `valid_roi == True`. Experiments with zero valid ROIs are skipped.

ii. ```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
...
if n_valid == 0:
    print(f"  WARNING: No valid ROIs in experiment {experiment_id}, skipping")
    return None
...
events_valid = events_data[:, valid_roi]
```

iii. `CONVERSION_NOTES.md` identifies `valid_roi` filtering as the key neuron curation rule and says it is needed to match SDK/reference preprocessing.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial neural data are aligned to trial start, not change time. The extracted per-trial timestamps are the regular 30 Hz samples between each trial’s `start_time` and `stop_time`.

ii. ```python
t_trial_start = trials['start_time'][trial_idx]
t_trial_stop = trials['stop_time'][trial_idx]
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
```

```python
'temporal_alignment_event': 'Trial start time (first stimulus onset of trial)',
'off_start': 0.0,
'off_end': None,
```

iii. `CONVERSION_NOTES.md` says “Alignment event: Trial start time (stimulus onset)” and the final metadata encodes the same choice.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. All streams are rebinned/resampled to a fixed 30 Hz grid, giving a time bin size of about 33.33 ms.

ii. ```python
TARGET_RATE_HZ = 30.0
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ
...
dt = 1.0 / target_rate
regular_ts = np.arange(t_start, t_end, dt)
```

iii. `CONVERSION_NOTES.md` says “Resample to 30 Hz: Paper interpolates to 30 Hz” and lists this as a key design decision.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentation table, specifically `intervals/<stimulus_table>/image_name`, after excluding omitted stimuli.

ii. ```python
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
all_stim_images = sorted(set(
    raw_data['stim']['image_name'][raw_data['stim']['image_name'] != 'omitted']
))
```

iii. `CONVERSION_NOTES.md` says image identity should come from `stimulus_presentations.image_name` and that omitted flashes should carry forward the previous image.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each resampled timepoint, the agent finds the most recent non-omitted stimulus onset and assigns that image identity. During gray periods and omitted flashes, it keeps the last shown image. It then maps per-experiment image IDs into a global image vocabulary.

ii. ```python
non_omitted = ~stim_data['omitted']
stim_starts = stim_data['start_time'][non_omitted]
stim_names = stim_data['image_name'][non_omitted]
insert_idx = np.searchsorted(stim_starts, timepoints, side='right') - 1

for i in range(n_tp):
    if insert_idx[i] >= 0:
        img_name = stim_names[insert_idx[i]]
        image_idx[i] = name_to_idx.get(img_name, 0)
```

```python
img_id_global = np.array([local_to_global.get(v, 0) for v in trial_data_out['image_identity']], dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` says “Image identity during gray screen: use the identity of the image that was just shown” and “for omitted flashes: continue with previous image identity.”

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is computed on the same per-trial 30 Hz timestamp grid as the neural data and then stacked into the trial output array with identical timepoints.

ii. ```python
trial_ts = regular_ts[trial_time_indices]
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
```

iii. The code uses the same `trial_time_indices`/`trial_ts` for neural and image identity. No separate written justification was given beyond using one shared regular grid for all streams.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus presentation table’s `is_change`, `start_time`, and `omitted` fields, not from the trial table.

ii. ```python
change_mask = stim_data['is_change'] & ~stim_data['omitted']
change_starts = stim_data['start_time'][change_mask]
change_stops = stim_data['stop_time'][change_mask]
```

iii. `CONVERSION_NOTES.md` describes image change as coming from stimulus timing and change annotations over the 750 ms image interval.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The agent generates a binary signal on the 30 Hz grid and sets it to 1 for every timepoint in the 750 ms window starting at each non-omitted change stimulus onset.

ii. ```python
for cs, ce in zip(change_starts, change_stops):
    mask = (timepoints >= cs) & (timepoints < cs + 0.75)
    change_signal[mask] = 1
```

iii. `CONVERSION_NOTES.md` says “Image change: 1 during 750ms window starting at change onset,” and later validates the expected ~7.7% occupancy.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary; the categories are `0 = no_change` and `1 = change`.

ii. ```python
change_signal = np.zeros(n_tp, dtype=np.int64)
...
output_values = [
    global_image_names,
    ['no_change', 'change'],
    ...
]
```

iii. No additional thresholding logic is used beyond binary assignment from change windows.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like the other time-varying outputs, it is computed on each trial’s shared regular 30 Hz time grid and stacked with the neural trial.

ii. ```python
trial_ts = regular_ts[trial_time_indices]
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
...
output_tv = np.stack([img_id_global, img_change, running_binned, pupil_binned], axis=0)
```

iii. The code uses the same `trial_ts` underlying the neural slices. No separate justification beyond common-grid alignment was documented.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and its timestamps.

ii. ```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. `CONVERSION_NOTES.md` maps `running/speed/data` to the running-speed output.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated to the common 30 Hz grid. Global percentile bin edges are computed from all experiments’ raw running samples, and each trial’s resampled running trace is digitized into 5 bins.

ii. ```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
```

```python
running = f['processing']['running']['speed']['data'][()]
all_running_values.append(running.astype(np.float32))
...
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
...
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. `CONVERSION_NOTES.md` says running speed should be “Interpolate[d] to 30 Hz” and “discretize[d] into 5 equal percentile bins.”

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded into 5 equal-percentile bins using global bin edges computed across all collected running-speed samples.

ii. ```python
def compute_percentile_bins(values, n_bins=5):
    valid = values[~np.isnan(values)]
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
```

```python
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. `CONVERSION_NOTES.md` explicitly says running speed should be discretized into five equal percentile bins.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. The agent resamples running speed onto the same regular 30 Hz grid used for neural data, then uses the same trial index mask to slice both.

ii. ```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
...
running_trial = running_resampled[trial_time_indices]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. No separate justification beyond the shared 30 Hz grid was written.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The agent derives this output from eye-tracking pupil area, not pupil width: `acquisition/EyeTracking/pupil_tracking/area`, plus timestamps and `likely_blink`.

ii. ```python
pupil_tracking = f['acquisition']['EyeTracking']['pupil_tracking']
pupil_area = pupil_tracking['area'][()]
pupil_ts = pupil_tracking['timestamps'][()]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
```

iii. `CONVERSION_NOTES.md` says “Use pupil area as proxy for diameter.”

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink-marked samples are set to NaN, NaNs are linearly interpolated in the 1D pupil trace, the result is resampled to the shared 30 Hz grid, and then the values are digitized into 5 global percentile bins. If pupil tracking is absent, the trial gets all-NaN pupil values until digitization.

ii. ```python
if likely_blink is not None:
    pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
```

```python
valid_pupil = pupil_area[~np.isnan(pupil_area)]
if len(valid_pupil) > 0:
    all_pupil_values.append(valid_pupil.astype(np.float32))
...
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
...
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. `CONVERSION_NOTES.md` says to handle blinks as NaNs, interpolate them, then discretize into equal-percentile bins.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is thresholded into 5 equal-percentile bins using global pupil-area bin edges, with NaNs assigned to bin 0 at digitization time.

ii. ```python
def digitize_to_bins(values, bin_edges):
    binned = np.digitize(values, bin_edges[1:-1])
    ...
    binned[np.isnan(values)] = 0
```

```python
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. `CONVERSION_NOTES.md` specifies five equal-percentile bins; NaN handling is only evident in code.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil is resampled to the common 30 Hz grid and then trial-segmented using the same per-trial mask as neural data.

ii. ```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
trial_time_indices = np.where(trial_mask)[0]
...
pupil_trial = pupil_resampled[trial_time_indices]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. No separate written justification was provided beyond the agent’s choice to align all streams on one regular grid.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table fields `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. ```python
if trials['hit'][trial_idx]:
    outcome = 0
elif trials['miss'][trial_idx]:
    outcome = 1
elif trials['false_alarm'][trial_idx]:
    outcome = 2
elif trials['correct_reject'][trial_idx]:
    outcome = 3
```

iii. `CONVERSION_NOTES.md` maps those four fields directly to the trial-outcome output.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The agent converts the four trial outcome cases into integer labels `0..3`. Unknown outcomes are skipped. In the final output array, the selected label is broadcast across every time bin in that trial.

ii. ```python
else:
    continue  # Unknown outcome, skip
...
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)
```

iii. `CONVERSION_NOTES.md` says trial outcome is a static per-trial categorical variable.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The code skips experiments that fail to load, have no valid ROIs, or end up with too few valid trials. Missing pupil data are allowed: absent pupil tracking yields all-NaN pupil traces, blink frames are converted to NaN and interpolated, and any remaining NaNs are mapped to bin 0 during discretization. Unknown trial outcomes are skipped.

ii. ```python
if n_valid == 0:
    ...
    return None
...
except Exception as e:
    print(f"  ERROR loading {experiment_id}: {e}")
    return None
```

```python
if raw_data['pupil_area'] is not None:
    ...
else:
    pupil_trial = np.full(n_tp, np.nan)
```

```python
binned[np.isnan(values)] = 0
...
else:
    continue  # Unknown outcome, skip
```

iii. `CONVERSION_NOTES.md` justifies blink interpolation and use of NaN-safe percentile binning. The rest is implied by the defensive checks in code.

## 9-a. What are the most time-consuming steps of the code?

i. The expensive parts are repeated NWB file reads and whole-session interpolation/resampling. The code scans the dataset three times, and each experiment is loaded again in the main pass before event, running, and pupil interpolation.

ii. ```python
# Pass 1: determine global image names
for _, row in exp_table.iterrows():
    with h5py.File(nwb_path, 'r') as f:
        ...

# Pass 2: collect running speed and pupil data
for idx, (_, row) in enumerate(exp_table.iterrows()):
    with h5py.File(nwb_path, 'r') as f:
        ...

# Pass 3: processing experiments
for idx, (_, row) in enumerate(exp_table.iterrows()):
    raw_data = load_experiment_data(nwb_path, eid)
```

iii. `CONVERSION_NOTES.md` explicitly calls out “Multiple passes over NWB files” and gives per-session runtime estimates dominated by load/process time.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are left scalar or repeated: per-feature interpolation in `interpolate_to_regular_grid`, per-timepoint assignment in `get_image_at_timepoints`, per-change-event marking in `get_image_change_at_timepoints`, and Python-level row iteration over the experiment table in all three passes.

ii. ```python
for i in range(data.shape[1]):
    result[:, i] = np.interp(target_timestamps, timestamps, data[:, i])
```

```python
for i in range(n_tp):
    if insert_idx[i] >= 0:
        ...
```

```python
for cs, ce in zip(change_starts, change_stops):
    mask = (timepoints >= cs) & (timepoints < cs + 0.75)
    change_signal[mask] = 1
```

iii. `CONVERSION_NOTES.md` says “Sequential processing of experiments (could parallelize)” and mentions vectorized interpolation and `searchsorted` as partial speedups, but the remaining loops are evident from code.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats full-file access across three passes and repeatedly reconstructs global mappings. It also reopens each file to collect image names and again to collect running/pupil statistics before the actual conversion pass.

ii. ```python
# Pass 1
with h5py.File(nwb_path, 'r') as f:
    ...

# Pass 2
with h5py.File(nwb_path, 'r') as f:
    ...

# Pass 3
raw_data = load_experiment_data(nwb_path, eid)
```

iii. `CONVERSION_NOTES.md` explicitly says “3-pass approach” and lists “Multiple passes over NWB files” as an implementation characteristic/inefficiency.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script imports `ProcessPoolExecutor`/`as_completed` but never uses them, reads some fields that are not used downstream (`cell_specimen_ids`, `change_stops`, `trial_outcomes` list, `TIME_BIN_MS` as a constant distinct from actual grids), and constructs a local-to-global image mapping with repeated `list.index` lookups. It also loads trial `is_change` and `change_image_name` fields without using them in output construction.

ii. ```python
from concurrent.futures import ProcessPoolExecutor, as_completed
...
cell_specimen_ids = cell_table['cell_specimen_id'][()]
...
'is_change': trials_grp['is_change'][()].astype(bool),
'change_image_name': trials_grp['change_image_name'][()],
...
change_stops = stim_data['stop_time'][change_mask]
...
trial_outcomes = []
```

iii. No explicit written justification was provided for these discarded or unused pieces; they are visible only from the final code.
