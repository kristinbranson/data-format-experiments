# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from the local NWB files and metadata CSVs under `data/visual-behavior-ophys-1.1.0` using `pandas` and `h5py`, rather than using the Allen SDK cache. It first reads `ophys_experiment_table.csv`, keeps only NWB files present on disk, and further restricts to a hand-picked list of active session types before opening each NWB file and extracting trials, neural, running, pupil, and stimulus data.

ii.
```python
DATA_DIR = Path('data/visual-behavior-ophys-1.1.0')
NWB_DIR = DATA_DIR / 'behavior_ophys_experiments'
META_DIR = DATA_DIR / 'project_metadata'

exp_table = pd.read_csv(META_DIR / 'ophys_experiment_table.csv')
...
mask = (
    exp_table['ophys_experiment_id'].isin(nwb_ids) &
    exp_table['session_type'].isin(ACTIVE_SESSION_TYPES)
)
active_exps = exp_table[mask].copy()
...
with h5py.File(nwb_path, 'r') as f:
    ...
```

iii. In `CONVERSION_NOTES.md` Step 4 and Step 5, the AI justified this as working from the on-disk subset only and focusing on “active behavior” sessions because passive sessions were considered irrelevant for trial-outcome decoding.

## 1-b. How are the data split into subjects?

i. Subjects are split by unique `mouse_id` values from the experiment metadata table, converted to strings.

ii.
```python
subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])
```

iii. The justification in Step 5 notes is that subject IDs should come from `mouse_id` in the experiment table.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB experiment file, i.e. each `ophys_experiment_id`, as one output session. It does not group multiple experiments that share the same `ophys_session_id`.

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

iii. The trajectory explicitly says: “Each NWB file = one experiment = one imaging plane. This should be one ‘session’ in our output, since each has its own set of neurons.”

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB `intervals/trials` table. For each kept trial, the AI uses the full window from `start_time` to `stop_time`, but extracts samples on a newly created 30 Hz regular time grid rather than on native ophys timestamps.

ii.
```python
trials_grp = f['intervals']['trials']
...
t_trial_start = trials['start_time'][trial_idx]
t_trial_stop = trials['stop_time'][trial_idx]
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. In Step 5, the AI justified using `start_time` to `stop_time` as a variable-length trial window from the built-in trials table.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to `(go | catch)` and must not be `aborted` or `auto_rewarded`. Trials with fewer than 3 resampled bins are dropped, and experiments with fewer than 2 processed trials are skipped.

ii.
```python
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

iii. Step 5 notes justify excluding aborted and auto-rewarded trials per the instructions, and the code adds minimum-length and minimum-trial-count checks as extra safeguards.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from the NWB `processing/ophys/event_detection/data` array, filtered by `valid_roi`. It does not use dF/F traces.

ii.
```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
...
events_data = f['processing']['ophys']['event_detection']['data'][()]
events_valid = events_data[:, valid_roi]
```

iii. In Step 5 and the trajectory, the AI justified this by saying the paper analyzed discrete calcium events and that events are closer to spike-like activity than dF/F.

## 2-b. How is the `neural` data processed?

i. The event traces are linearly interpolated from native ophys timestamps onto a regular 30 Hz grid, then clipped to be non-negative. Per trial, the resampled event matrix is sliced and transposed to `(neurons, time)`.

ii.
```python
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. Step 5 notes justify 30 Hz interpolation as matching the paper’s analysis pipeline and note that events should remain non-negative after interpolation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only ROIs with `valid_roi == True`. If an experiment has no valid ROIs, it is skipped entirely.

ii.
```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
...
if n_valid == 0:
    print(f"  WARNING: No valid ROIs in experiment {experiment_id}, skipping")
    return None
...
events_valid = events_data[:, valid_roi]
```

iii. Step 1 and Step 5 notes justify this using the Allen ROI-quality classifier and explicitly call out `valid_roi` filtering as a curation rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial neural data are aligned to trial start: the AI defines each trial by the `start_time`/`stop_time` window and extracts regular-grid samples within that interval.

ii.
```python
t_trial_start = trials['start_time'][trial_idx]
t_trial_stop = trials['stop_time'][trial_idx]
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
```

iii. Step 5 notes say the alignment event is “Trial start time (stimulus onset)” with `off_start = 0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins all signals onto a 30 Hz regular grid, so the converted time bin is `1000/30 ≈ 33.33 ms`.

ii.
```python
TARGET_RATE_HZ = 30.0
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ
...
dt = 1.0 / target_rate
regular_ts = np.arange(t_start, t_end, dt)
```

iii. The justification in Step 3 and Step 5 notes is that the paper said signals were “linearly interpolat[ed] onto a consistent set of 30hz timestamps.”

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation table, specifically `image_name` together with each stimulus `start_time`, while dropping omitted flashes.

ii.
```python
stim_data = {
    'start_time': stim['start_time'][()],
    'stop_time': stim['stop_time'][()],
    'image_name': stim['image_name'][()],
    'is_change': stim['is_change'][()].astype(bool),
    'omitted': stim['omitted'][()],
}
...
non_omitted = ~stim_data['omitted']
stim_starts = stim_data['start_time'][non_omitted]
stim_names = stim_data['image_name'][non_omitted]
```

iii. In Step 5 notes, the AI justified this by saying image identity during gray periods should remain the most recently shown image and omitted flashes should continue the previous image.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI reconstructs image identity for every trial timepoint by finding the most recent non-omitted stimulus onset, assigns that image during gray periods, then maps per-experiment image labels to a global sorted image list collected in Pass 1.

ii.
```python
insert_idx = np.searchsorted(stim_starts, timepoints, side='right') - 1
for i in range(n_tp):
    if insert_idx[i] >= 0:
        img_name = stim_names[insert_idx[i]]
        image_idx[i] = name_to_idx.get(img_name, 0)
...
global_image_names = sorted(all_image_names_set)
...
img_id_global = np.array([local_to_global.get(v, 0) for v in trial_data_out['image_identity']], dtype=np.int64)
```

iii. The notes explicitly justify carrying forward the last shown image through gray screens and omitted flashes; the global sorted mapping is an implementation choice rather than something separately justified in the notes.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is evaluated on the same 30 Hz `trial_ts` grid used for the neural data, so it is aligned by construction after resampling.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
```

iii. The AI’s general justification is the Step 5 decision to put all streams on a common 30 Hz timeline before extracting per-trial data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table’s `is_change`, `start_time`, and `omitted` fields rather than from the trial table’s `change_time`.

ii.
```python
change_mask = stim_data['is_change'] & ~stim_data['omitted']
change_starts = stim_data['start_time'][change_mask]
change_stops = stim_data['stop_time'][change_mask]
```

iii. In Step 5 notes, the AI described this output as a binary indicator tied to stimulus timing and a single 750 ms change window.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For every detected change stimulus, the AI marks all trial timepoints from change onset through the following 750 ms as `1`; otherwise the signal is `0`.

ii.
```python
change_signal = np.zeros(n_tp, dtype=np.int64)
for cs, ce in zip(change_starts, change_stops):
    mask = (timepoints >= cs) & (timepoints < cs + 0.75)
    change_signal[mask] = 1
```

iii. The Step 5 notes justify this using the 250 ms image plus 500 ms gray interval described in the paper.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No further thresholding is applied. The AI directly produces a binary category: `0 = no_change`, `1 = change`.

ii.
```python
change_signal = np.zeros(n_tp, dtype=np.int64)
...
change_signal[mask] = 1
...
output_values = [
    global_image_names,
    ['no_change', 'change'],
    ...
]
```

iii. The Step 5 mapping notes describe image change as a binary variable, so the code simply emits binary categories.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like the neural data and image identity, image change is computed directly on the per-trial 30 Hz timestamp grid.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
...
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
...
output_tv = np.stack([img_id_global, img_change, running_binned, pupil_binned], axis=0)
```

iii. The AI’s stated alignment strategy is a shared 30 Hz timeline for all modalities before trial extraction.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from NWB `processing/running/speed/data` and its paired `timestamps`.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. No elaborate separate justification was given beyond using the dataset’s running-speed stream as the behavioral source.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI interpolates running speed onto the 30 Hz regular grid, then computes five percentile-bin edges globally in a separate pass over all experiments and digitizes each trial’s running-speed samples into those bins.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
all_running_values.append(running.astype(np.float32))
...
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
...
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. Step 5 notes justify 30 Hz interpolation for alignment and five equal-percentile bins for decoder outputs.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five categories using global percentile bin edges, with `NaN` values forced to bin `0`.

ii.
```python
def digitize_to_bins(values, bin_edges):
    binned = np.digitize(values, bin_edges[1:-1])
    binned = np.clip(binned, 0, n_bins - 1)
    binned[np.isnan(values)] = 0
    return binned.astype(np.int64)
```

iii. The notes justify this as the required discretization into five equal-percentile bins.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by resampling it to the same 30 Hz regular timestamp grid and then slicing the same `trial_time_indices` used for the neural data.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
running_trial = running_resampled[trial_time_indices]
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The AI consistently justified alignment by putting all streams on one common resampled clock before trial segmentation.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The AI derives this output from NWB eye-tracking `pupil_tracking/area` and `timestamps`, plus the `likely_blink` mask. It uses pupil area as a proxy for diameter.

ii.
```python
pupil_tracking = f['acquisition']['EyeTracking']['pupil_tracking']
pupil_area = pupil_tracking['area'][()]
pupil_ts = pupil_tracking['timestamps'][()]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
```

iii. Step 5 notes explicitly say “Use pupil area as proxy for diameter.”

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink-labeled frames are set to `NaN`, then missing points are linearly interpolated within the raw pupil-area series, then the result is resampled to the 30 Hz regular grid. Global five-bin percentile thresholds are computed in a separate pass over all valid pupil samples and used to discretize the per-trial traces.

ii.
```python
if likely_blink is not None:
    pupil_area[likely_blink] = np.nan

pupil_area = interpolate_nans(pupil_area)
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
valid_pupil = pupil_area[~np.isnan(pupil_area)]
...
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
...
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. The Step 5 notes justify blink handling and percentile discretization; the specific choice to interpolate raw-area NaNs before resampling is implemented in code and summarized in Step 6 notes.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The AI uses five global percentile bins over pupil area, again with `NaN` values mapped to bin `0`.

ii.
```python
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
...
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. The justification is the same as for running speed: the task required a five-way categorical output.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil values are aligned by resampling them onto the same 30 Hz regular grid and then slicing by the same per-trial index set used for neural data.

ii.
```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
pupil_trial = pupil_resampled[trial_time_indices]
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The AI justified this with the same “shared 30 Hz timeline” argument used for the other streams.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

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
```

iii. Step 5 notes identify these four canonical trial outcomes as the intended output classes.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps the four outcome booleans directly to integer class IDs 0-3, then broadcasts the chosen class across all time bins in the trial.

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
...
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)
```

iii. Step 5 notes justify trial outcome as a static per-trial categorical variable.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several data issues by skipping broken experiments, skipping experiments with no valid ROIs or too few valid trials, dropping very short resampled trials, warning and filling missing pupil data with `NaN` that later digitize to bin `0`, and interpolating blink-induced pupil gaps. If all pupil values are `NaN`, it replaces them with zeros during interpolation.

ii.
```python
if n_valid == 0:
    ...
    return None
...
except Exception as e:
    print(f"  ERROR loading {experiment_id}: {e}")
    return None
...
if len(trial_time_indices) < 3:
    continue
...
if raw_data['pupil_area'] is not None:
    ...
else:
    pupil_trial = np.full(n_tp, np.nan)
...
if nans.all():
    return np.zeros_like(arr)
```

iii. The notes explicitly justify blink interpolation and skipping unusable experiments; the other guardrails are visible in the code and are only lightly discussed in the notes.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming work is repeated full-dataset NWB I/O across three passes: one pass to scan image names, one to collect running and pupil values for global bins, and one to fully load and process each experiment. The per-experiment 30 Hz interpolation also adds cost.

ii.
```python
# Pass 1
for _, row in exp_table.iterrows():
    with h5py.File(nwb_path, 'r') as f:
        ...

# Pass 2
for idx, (_, row) in enumerate(exp_table.iterrows()):
    with h5py.File(nwb_path, 'r') as f:
        ...

# Pass 3
for idx, (_, row) in enumerate(exp_table.iterrows()):
    raw_data = load_experiment_data(nwb_path, eid)
    result = process_single_experiment(raw_data, exp_meta)
```

iii. Step 6 notes explicitly list “Multiple passes over NWB files” and sequential processing as inefficiencies.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization targets are the per-feature loop in `interpolate_to_regular_grid`, the per-timepoint loop in `get_image_at_timepoints`, and the per-change loop in `get_image_change_at_timepoints`. The outer experiment loop could also have been parallelized, as hinted by the unused process-pool imports.

ii.
```python
for i in range(data.shape[1]):
    result[:, i] = np.interp(target_timestamps, timestamps, data[:, i])
...
for i in range(n_tp):
    if insert_idx[i] >= 0:
        ...
...
for cs, ce in zip(change_starts, change_stops):
    mask = (timepoints >= cs) & (timepoints < cs + 0.75)
    change_signal[mask] = 1
```

iii. The Step 6 notes mention sequential processing as an inefficiency; the more specific vectorization opportunities are evident from the code itself rather than separately documented.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats dataset traversal three times, reopening every NWB file each pass. It also recomputes per-experiment local-to-global image mappings during assembly.

ii.
```python
print("\n--- Pass 1: Collecting global image names ---")
...
print("\n--- Pass 2: Collecting running speed and pupil data for percentile bins ---")
...
print("\n--- Pass 3: Processing experiments ---")
...
for local_idx, name in enumerate(result['all_image_names']):
    if name in global_image_names:
        local_to_global[local_idx] = global_image_names.index(name)
```

iii. Step 6 notes explicitly call out the multi-pass design; the repeated `index` lookups are an additional code-level inefficiency.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads and stores some data that are never used downstream, including `cell_specimen_ids`, `valid_roi` in the returned raw-data dict, `change_stops` inside image-change computation, and a one-element `trial_outcome` array that is immediately replaced by broadcast output. It also imports several unused modules for parallelism and OS access.

ii.
```python
return {
    'experiment_id': experiment_id,
    'valid_roi': valid_roi,
    'cell_specimen_ids': cell_specimen_ids[valid_roi],
    ...
}
...
change_stops = stim_data['stop_time'][change_mask]
for cs, ce in zip(change_starts, change_stops):
    ...
...
trial_outcome = np.array([trial_data_out['trial_outcome']], dtype=np.int64)
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
```

iii. There is no explicit separate justification for these; they appear to be byproducts of implementation rather than deliberate downstream requirements.
