# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the Allen SDK cache. It reads `ophys_experiment_table.csv`, finds NWB files present on disk, filters to a hard-coded list of active session types, and then opens each NWB directly with `h5py`. All downstream subject/session/trial data are extracted from those local NWB files.

ii. 
```python
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
    return {
        'ophys_ts': ophys_ts,
        'events': events_valid,
        'trials': trials,
        'stim': stim_data,
        'running_speed': running_speed,
        'running_ts': running_ts,
        'pupil_area': pupil_area,
        'pupil_ts': pupil_ts,
        'likely_blink': likely_blink,
    }
```

iii. In `CONVERSION_NOTES.md` Step 4-6, the AI says the on-disk dataset is only a subset of the full release, so it chose to work directly from the local NWB files and metadata tables. It also justified filtering to active sessions because passive sessions would not support the requested trial-outcome outputs.

## 1-b. How are the data split into subjects?

i. Subjects are defined by unique `mouse_id` values from the filtered experiment table. The IDs are converted to strings and mapped to `subject_idx`.

ii. 
```python
subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI explicitly says “Subject IDs: Use `mouse_id` from experiment table.”

## 1-c. How are the data split into sessions?

i. The AI treats each NWB experiment file as a separate session in the output. It does not group experiments that share an `ophys_session_id`.

ii. 
```python
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

iii. In `CONVERSION_NOTES.md` Step 5, the AI states: “Each NWB experiment = one ‘session’ in output format (one imaging plane with its own neurons).”

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB `intervals/trials` table. For each kept trial, the AI uses the trial’s `start_time` and `stop_time`, finds all 30 Hz resampled timestamps inside that interval, and uses that window as one trial.

ii. 
```python
trials_grp = f['intervals']['trials']
trials = {
    'start_time': trials_grp['start_time'][()],
    'stop_time': trials_grp['stop_time'][()],
    'change_time': trials_grp['change_time'][()],
    ...
}
```

```python
for trial_idx in valid_indices:
    t_trial_start = trials['start_time'][trial_idx]
    t_trial_stop = trials['stop_time'][trial_idx]
    trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
    trial_time_indices = np.where(trial_mask)[0]
    ...
    neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says the trial window should be `start_time` to `stop_time` from the trials table and remain variable-length.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept only if they are Go or Catch and not `aborted` or `auto_rewarded`. Trials with fewer than 3 resampled bins are skipped, and experiments with fewer than 2 processed trials are dropped.

ii. 
```python
valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
valid_indices = np.where(valid_mask)[0]

if len(valid_indices) < 2:
    ...
```

```python
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]

if len(trial_time_indices) < 3:
    continue
...
if len(neural_trials) < 2:
    ...
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI justifies excluding aborted and auto-rewarded trials directly from the task instructions, and says only Go and Catch trials should remain.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `processing/ophys/event_detection/data` in the NWB file, after filtering cells by `valid_roi`.

ii. 
```python
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][()].astype(bool)
...
events_data = f['processing']['ophys']['event_detection']['data'][()]
events_valid = events_data[:, valid_roi]
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says it chose events rather than dF/F because the paper states that analyses were performed on “discrete calcium events.”

## 2-b. How is the `neural` data processed?

i. The event traces are linearly interpolated from the native ophys timestamps onto a regular 30 Hz grid, clipped to remain non-negative, and then sliced into trial windows. Each trial is stored as a `(n_cells, n_timepoints)` float32 matrix.

ii. 
```python
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
```

```python
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. In `CONVERSION_NOTES.md` Step 5-6, the AI says it wanted to match the paper’s 30 Hz interpolation and keep the event signal non-negative.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural QC is filtering to `valid_roi == True`. Experiments with zero valid ROIs are skipped.

ii. 
```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
...
n_valid = valid_roi.sum()
if n_valid == 0:
    print(f"  WARNING: No valid ROIs in experiment {experiment_id}, skipping")
    return None
...
events_valid = events_data[:, valid_roi]
```

iii. In `CONVERSION_NOTES.md` Step 1 and Step 5, the AI says the Allen pipeline’s key ROI curation is the `valid_roi` flag, and it chose to mirror that.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start/stop on a global 30 Hz timebase. The effective alignment event is trial start; all neural bins inside `[start_time, stop_time)` are assigned to the trial.

ii. 
```python
regular_ts = np.arange(t_start, t_end, dt)
...
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

```python
'temporal_alignment_event': 'Trial start time (first stimulus onset of trial)',
'off_start': 0.0,
'off_end': None,
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says “Alignment event: Trial start time (stimulus onset).”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 30 Hz grid, i.e. `33.33 ms` bins. Yes, the code rebins/resamples all streams onto that regular grid.

ii. 
```python
TARGET_RATE_HZ = 30.0
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ
```

```python
dt = 1.0 / target_rate
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
```

iii. The AI repeatedly justifies this in `CONVERSION_NOTES.md` Step 3-6 as matching the paper’s “30hz timestamps.”

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation table, specifically `image_name` for non-omitted presentations.

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
all_stim_images = sorted(set(
    raw_data['stim']['image_name'][raw_data['stim']['image_name'] != 'omitted']
))
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says it wanted image identity to reflect the actual presented image stream, including gray intervals and omitted flashes.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each trial timestamp, the AI finds the most recent non-omitted stimulus onset and assigns that image. During gray periods it carries forward the last shown image; omitted flashes also keep the previous identity. Local image IDs are then remapped into a global image vocabulary across experiments.

ii. 
```python
non_omitted = ~stim_data['omitted']
stim_starts = stim_data['start_time'][non_omitted]
stim_names = stim_data['image_name'][non_omitted]
...
insert_idx = np.searchsorted(stim_starts, timepoints, side='right') - 1
for i in range(n_tp):
    if insert_idx[i] >= 0:
        img_name = stim_names[insert_idx[i]]
        image_idx[i] = name_to_idx.get(img_name, 0)
```

```python
global_image_names = sorted(all_image_names_set)
...
img_id_global = np.array([local_to_global.get(v, 0) for v in trial_data_out['image_identity']], dtype=np.int64)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI explicitly says to use the last presented image during gray periods and to continue the previous identity through omitted flashes.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed at exactly the same resampled trial timestamps used for the neural data (`trial_ts` from `regular_ts`), so it is aligned bin-by-bin to each neural trial matrix.

ii. 
```python
trial_ts = regular_ts[trial_time_indices]
...
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The AI’s notes say all streams should be resampled to the same 30 Hz clock before trial extraction.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table, using `is_change` and `omitted`, rather than the trial table.

ii. 
```python
stim_data = {
    'start_time': stim['start_time'][()],
    'stop_time': stim['stop_time'][()],
    'image_name': stim['image_name'][()],
    'is_change': stim['is_change'][()].astype(bool),
    'omitted': stim['omitted'][()],
}
```

```python
change_mask = stim_data['is_change'] & ~stim_data['omitted']
change_starts = stim_data['start_time'][change_mask]
change_stops = stim_data['stop_time'][change_mask]
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says image change should come from “`trials.is_change` + stimulus timing,” and the code operationalizes that using the stimulus presentations’ change markers.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each change stimulus, the AI marks a binary 1 for 750 ms starting at the change onset, and 0 elsewhere.

ii. 
```python
change_signal = np.zeros(n_tp, dtype=np.int64)
...
for cs, ce in zip(change_starts, change_stops):
    mask = (timepoints >= cs) & (timepoints < cs + 0.75)
    change_signal[mask] = 1
```

iii. The justification in `CONVERSION_NOTES.md` Step 5 is that one image interval is 250 ms image plus 500 ms gray, so the change marker should last one 750 ms flash interval.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No continuous threshold is used. The variable is directly encoded as binary categories: `0 = no_change`, `1 = change`.

ii. 
```python
change_signal = np.zeros(n_tp, dtype=np.int64)
...
change_signal[mask] = 1
```

```python
output_values = [
    global_image_names,
    ['no_change', 'change'],
    [f'bin_{i}' for i in range(5)],
    [f'bin_{i}' for i in range(5)],
    ['hit', 'miss', 'false_alarm', 'correct_reject'],
]
```

iii. The AI’s notes treat image change as an already discrete event and do not describe any further thresholding step.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is evaluated on the same `trial_ts` vector that indexes each neural trial, so the binary change signal is aligned bin-by-bin to the resampled neural data.

ii. 
```python
trial_ts = regular_ts[trial_time_indices]
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. As with image identity, the AI’s justification is that all data streams should share the same 30 Hz grid before trial slicing.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from the NWB running module: `processing/running/speed/data` and its timestamps.

ii. 
```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI maps `running/speed/data` directly to the running-speed output.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI first gathers running-speed samples from all experiments to compute global percentile-bin edges. During trial processing it linearly interpolates running speed onto the 30 Hz grid, slices the trial window, and digitizes each bin.

ii. 
```python
all_running_values = []
...
running = f['processing']['running']['speed']['data'][()]
all_running_values.append(running.astype(np.float32))
...
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
```

```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
running_trial = running_resampled[trial_time_indices]
...
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. In `CONVERSION_NOTES.md` Step 5-6, the AI says the goal was global percentile bins and a unified 30 Hz alignment.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into five equal-percentile bins using global bin edges computed from the pooled running-speed values.

ii. 
```python
def compute_percentile_bins(values, n_bins=5):
    valid = values[~np.isnan(values)]
    ...
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
```

```python
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
...
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. The notes explicitly say running speed should be “discretized into 5 percentile bins.”

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same regular 30 Hz timestamps as the neural events, then trial slices are taken with the same `trial_time_indices`.

ii. 
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
running_trial = running_resampled[trial_time_indices]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The AI’s stated alignment rule is that all outputs should share the resampled neural timebase.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The AI uses eye-tracking pupil **area**, not pupil width/diameter. It loads `acquisition/EyeTracking/pupil_tracking/area` with timestamps and the `likely_blink` flag.

ii. 
```python
pupil_tracking = f['acquisition']['EyeTracking']['pupil_tracking']
pupil_area = pupil_tracking['area'][()]
pupil_ts = pupil_tracking['timestamps'][()]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says “Use pupil area as proxy for diameter.”

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames are set to `NaN`, missing points are linearly interpolated in the raw pupil-area trace, the result is resampled to 30 Hz, and then digitized with global percentile bins computed from pooled non-blink pupil-area values.

ii. 
```python
pupil_area = raw_data['pupil_area'].copy().astype(float)
...
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

iii. In `CONVERSION_NOTES.md` Step 5-6, the AI says it wanted to remove blink artifacts, fill them by interpolation, and then apply global percentile binning.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The output is thresholded into five equal-percentile bins using global pupil-area bin edges.

ii. 
```python
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
...
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. The notes explicitly say pupil should be “discretized into 5 percentile bins.”

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil area is aligned to neural data by resampling it to `regular_ts` first and then indexing the trial with the same `trial_time_indices` used for the neural matrix.

ii. 
```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
pupil_trial = pupil_resampled[trial_time_indices]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The AI’s alignment rationale is the same as for running speed and image variables: shared 30 Hz timestamps before trial slicing.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

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

```python
if trials['hit'][trial_idx]:
    outcome = 0
elif trials['miss'][trial_idx]:
    outcome = 1
elif trials['false_alarm'][trial_idx]:
    outcome = 2
elif trials['correct_reject'][trial_idx]:
    outcome = 3
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI maps those four trial flags directly to the trial-outcome output.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI converts the four mutually exclusive outcome flags into integer codes `0-3`, stores the code once per trial, and then broadcasts it across all time bins when building the output matrix.

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
```

```python
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)
```

iii. The notes say trial outcome should be a static per-trial variable, so the code broadcasts the same label through time.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several failure cases by skipping or filling data. Experiments are skipped if file loading fails, if there are no valid ROIs, or if no stimulus table is found. Missing pupil data are allowed; missing pupil for a whole experiment becomes all-`NaN` per trial and later bin `0`, while blink gaps inside available pupil data are interpolated. Trials with very short windows are skipped, and experiments with too few valid trials are skipped.

ii. 
```python
if n_valid == 0:
    ...
if stim_key is None:
    ...
except Exception as e:
    print(f"  ERROR loading {experiment_id}: {e}")
    return None
```

```python
if raw_data['pupil_area'] is not None:
    ...
    pupil_area = interpolate_nans(pupil_area)
    pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
else:
    pupil_trial = np.full(n_tp, np.nan)
```

```python
if len(trial_time_indices) < 3:
    continue
...
if len(neural_trials) < 2:
    ...
```

iii. In `CONVERSION_NOTES.md`, the AI says it wanted “sensible defaults” for missing data, especially blink-related pupil gaps, and wanted the pipeline to continue past bad experiments instead of crashing.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive work is repeated HDF5 I/O across three full passes over the experiment list, plus the per-experiment interpolation/resampling of neural, running, and pupil time series. The neural interpolation is especially heavy because it loops over every cell.

ii. 
```python
for _, row in exp_table.iterrows():
    ...
    with h5py.File(nwb_path, 'r') as f:
        ...
```

```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    ...
    raw_data = load_experiment_data(nwb_path, eid)
    ...
    result = process_single_experiment(raw_data, exp_meta)
```

```python
result = np.zeros((len(target_timestamps), data.shape[1]), dtype=np.float32)
for i in range(data.shape[1]):
    result[:, i] = np.interp(target_timestamps, timestamps, data[:, i])
```

iii. In `CONVERSION_NOTES.md` Step 6-7, the AI identifies file loading and processing as the main runtime contributors and estimates time separately for “Load” and “Process.”

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain scalar or Python-level: the per-cell loop in `interpolate_to_regular_grid`, the per-timepoint loop in `get_image_at_timepoints`, the per-change loop in `get_image_change_at_timepoints`, and the repeated `iterrows()` passes over the experiment table.

ii. 
```python
for i in range(data.shape[1]):
    result[:, i] = np.interp(target_timestamps, timestamps, data[:, i])
```

```python
for i in range(n_tp):
    if insert_idx[i] >= 0:
        img_name = stim_names[insert_idx[i]]
        image_idx[i] = name_to_idx.get(img_name, 0)
```

```python
for cs, ce in zip(change_starts, change_stops):
    mask = (timepoints >= cs) & (timepoints < cs + 0.75)
    change_signal[mask] = 1
```

iii. The AI’s notes mention vectorization as a goal and say `searchsorted` was used for efficiency, but the final code still leaves several vectorizable loops in place.

## 9-c. What processing does the code repeat multiple times?

i. The code rereads every experiment multiple times: once to collect global image names, once to collect running/pupil samples for bin edges, and once again to do the actual conversion. It also repeatedly performs `global_image_names.index(name)` lookups inside the experiment loop.

ii. 
```python
print("\n--- Pass 1: Collecting global image names ---")
for _, row in exp_table.iterrows():
    ...

print("\n--- Pass 2: Collecting running speed and pupil data for percentile bins ---")
for idx, (_, row) in enumerate(exp_table.iterrows()):
    ...

print("\n--- Pass 3: Processing experiments ---")
for idx, (_, row) in enumerate(exp_table.iterrows()):
    ...
```

```python
for local_idx, name in enumerate(result['all_image_names']):
    if name in global_image_names:
        local_to_global[local_idx] = global_image_names.index(name)
```

iii. In `CONVERSION_NOTES.md` Step 6, the AI itself notes “3-pass approach” and lists “Multiple passes over NWB files” as an inefficiency.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads and constructs several values that are not used later: `cell_specimen_ids`, trial-table `initial_image_name` and `change_image_name`, the local variable `change_stops`, the unused list `trial_outcomes`, the imported but unused `ProcessPoolExecutor`/`as_completed`, the unused `t_start_global`, and the temporary `trial_outcome` array created just before output assembly.

ii. 
```python
from concurrent.futures import ProcessPoolExecutor, as_completed
...
cell_specimen_ids = cell_table['cell_specimen_id'][()]
...
'initial_image_name': trials_grp['initial_image_name'][()],
'change_image_name': trials_grp['change_image_name'][()],
```

```python
change_stops = stim_data['stop_time'][change_mask]
...
trial_outcomes = []
...
t_start_global = time.time()
...
trial_outcome = np.array([trial_data_out['trial_outcome']], dtype=np.int64)
```

iii. These appear to be leftovers from exploration or earlier implementations. They are loaded or allocated but do not affect the saved dataset.
