# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not use the Allen SDK cache path used by the human reference. Instead, it read the metadata CSV `ophys_experiment_table.csv`, discovered which NWB files were physically present on disk, filtered to a hard-coded set of active session types, and then opened each NWB file directly with `h5py`. It made three passes over the experiment list: one to collect image names, one to collect running/pupil values for global binning, and one to load and process each experiment into trials.

ii.
```python
DATA_DIR = Path('data/visual-behavior-ophys-1.1.0')
NWB_DIR = DATA_DIR / 'behavior_ophys_experiments'
META_DIR = DATA_DIR / 'project_metadata'

def get_experiment_list(sample=False):
    exp_table = pd.read_csv(META_DIR / 'ophys_experiment_table.csv')
    nwb_files = list(NWB_DIR.glob('*.nwb'))
    ...
    mask = (
        exp_table['ophys_experiment_id'].isin(nwb_ids) &
        exp_table['session_type'].isin(ACTIVE_SESSION_TYPES)
    )
    active_exps = exp_table[mask].copy()

def load_experiment_data(nwb_path, experiment_id):
    with h5py.File(nwb_path, 'r') as f:
        ...
```

iii. In `CONVERSION_NOTES.md` Step 4-6, the AI justified this as working from the on-disk subset only, excluding passive sessions, and matching the paper more directly by reading NWB contents with `h5py` instead of going through the SDK cache.

## 1-b. How are the data split into subjects?

i. Subjects are split by unique `mouse_id` values from the experiment metadata table, converted to strings and indexed globally.

ii.
```python
subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI explicitly says “Subject IDs: Use `mouse_id` from experiment table.”

## 1-c. How are the data split into sessions?

i. The AI treated each NWB experiment file, i.e. each imaging plane / `ophys_experiment_id`, as one output “session.” It did not group multiple experiments from the same `ophys_session_id`.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    ...
    result = process_single_experiment(raw_data, exp_meta)
    ...
    all_neural.append(session_neural)
    all_input.append(session_input)
    all_output.append(session_output)
```

iii. In `CONVERSION_NOTES.md` Step 5 and trajectory step 36, the AI justified this by saying each NWB file has its own neurons, so “Each NWB experiment = one ‘session’ in output format.”

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB `intervals/trials` table. For each kept trial, the AI used `start_time` and `stop_time` to select all samples from a 30 Hz regularized time grid that fall inside the trial.

ii.
```python
trials_grp = f['intervals']['trials']
trials = {
    'start_time': trials_grp['start_time'][()],
    'stop_time': trials_grp['stop_time'][()],
    ...
}
...
for trial_idx in valid_indices:
    t_trial_start = trials['start_time'][trial_idx]
    t_trial_stop = trials['stop_time'][trial_idx]
    trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
    trial_time_indices = np.where(trial_mask)[0]
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says “Trial window: Use `trial start_time` to `stop_time` from trials table. Variable length across trials.”

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to `(go OR catch) AND not aborted AND not auto_rewarded`. Trials with fewer than 3 samples on the resampled grid are dropped, and experiments with fewer than 2 processed trials are skipped entirely. There is no explicit `change_time.notna()` filter.

ii.
```python
valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
valid_indices = np.where(valid_mask)[0]
...
if len(trial_time_indices) < 3:
    continue
...
if len(neural_trials) < 2:
    print(f"  WARNING: Only {len(neural_trials)} processed trials in experiment {raw_data['experiment_id']}, skipping")
    return None
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI justifies this from the task instructions: include Go and Catch trials, exclude Aborted and Auto-rewarded trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derived `neural` from the NWB event-detection matrix `processing/ophys/event_detection/data`, filtered by the ROI-validity mask `valid_roi`.

ii.
```python
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][()].astype(bool)
...
events_data = f['processing']['ophys']['event_detection']['data'][()]
events_valid = events_data[:, valid_roi]
```

iii. In `CONVERSION_NOTES.md` Step 5 and trajectory step 36, the AI says it chose calcium events rather than dF/F because the paper analyzed “discrete calcium events.”

## 2-b. How is the `neural` data processed?

i. Neural events are resampled from native ophys timestamps onto a regular 30 Hz grid with linear interpolation, negative interpolation artifacts are clipped to zero, and then per-trial neuron-by-time matrices are cut out from the resampled session array.

ii.
```python
TARGET_RATE_HZ = 30.0
...
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. In `CONVERSION_NOTES.md` Step 5-6, the AI justifies this by citing the paper’s “30 Hz” interpolation and by wanting one consistent time bin size across single-plane and mesoscope recordings.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies the `valid_roi` mask and skips experiments with zero valid ROIs. It does not perform additional neuron-level filtering after that.

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

iii. In `CONVERSION_NOTES.md` Step 1 and Step 5, the AI states that `valid_roi` is the SDK-equivalent automatic cell-quality filter and should be respected.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The final code aligns each trial to trial start. The session is first put on a regular 30 Hz timestamp grid, then per-trial neural data are the samples with timestamps `>= start_time` and `< stop_time`. Metadata says the alignment event is “Trial start time (first stimulus onset of trial).”

ii.
```python
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
...
'metadata': {
    ...
    'temporal_alignment_event': 'Trial start time (first stimulus onset of trial)',
    'off_start': 0.0,
    'off_end': None,
}
```

iii. `CONVERSION_NOTES.md` Step 5 says the alignment event is trial start time, not change time. The trajectory shows the agent considered change-time alignment earlier but did not keep that design.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins/resamples everything to a uniform 30 Hz grid, i.e. `33.33 ms` bins.

ii.
```python
TARGET_RATE_HZ = 30.0
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ
...
'metadata': {
    ...
    'time_bin_size': TIME_BIN_MS,
    'target_rate_hz': TARGET_RATE_HZ,
}
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI explicitly justifies 30 Hz resampling from the paper text and from wanting a common bin size across rigs.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The AI derived image identity from the stimulus-presentation table rather than from trial-level image-name columns. Specifically it used stimulus `image_name`, `start_time`, and `omitted`.

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
stim_starts = stim_data['start_time'][non_omitted]
stim_names = stim_data['image_name'][non_omitted]
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says image identity should come from `stimulus_presentations.image_name` and should persist through gray periods and omitted flashes.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI built per-experiment sorted image vocabularies and a global sorted image vocabulary. At each trial timepoint, it looked up the most recent non-omitted stimulus onset and assigned that image index; gray-screen periods and omitted flashes inherited the previous image.

ii.
```python
def get_image_at_timepoints(timepoints, stim_data, all_image_names):
    non_omitted = ~stim_data['omitted']
    stim_starts = stim_data['start_time'][non_omitted]
    stim_names = stim_data['image_name'][non_omitted]
    ...
    insert_idx = np.searchsorted(stim_starts, timepoints, side='right') - 1
    for i in range(n_tp):
        if insert_idx[i] >= 0:
            img_name = stim_names[insert_idx[i]]
            image_idx[i] = name_to_idx.get(img_name, 0)

global_image_names = sorted(all_image_names_set)
...
img_id_global = np.array([local_to_global.get(v, 0) for v in trial_data_out['image_identity']], dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` Step 5 lists two explicit edge-case choices: keep the last image during gray screen and continue the previous image identity across omitted flashes.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is evaluated on the exact same 30 Hz `trial_ts` values used to cut neural trials, so it is aligned sample-by-sample with each trial’s neural matrix.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
```

iii. The AI’s stated justification in `CONVERSION_NOTES.md` Step 6 is that all streams are resampled to the same 30 Hz time base before trial slicing.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The AI derived image change from the stimulus-presentation table’s `is_change`, `start_time`, and `omitted` fields, not from trial-level `change_time` and `go`.

ii.
```python
change_mask = stim_data['is_change'] & ~stim_data['omitted']
change_starts = stim_data['start_time'][change_mask]
change_stops = stim_data['stop_time'][change_mask]
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI describes image change as coming from `trials.is_change + stimulus timing`, but the code actually operationalizes that via the stimulus-presentation table.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI made `image_change` a binary time series that is 1 for a 750 ms window after each non-omitted change stimulus onset and 0 otherwise.

ii.
```python
def get_image_change_at_timepoints(timepoints, stim_data):
    change_signal = np.zeros(n_tp, dtype=np.int64)
    ...
    for cs, ce in zip(change_starts, change_stops):
        mask = (timepoints >= cs) & (timepoints < cs + 0.75)
        change_signal[mask] = 1
```

iii. In `CONVERSION_NOTES.md` Step 6 and trajectory step 95, the AI justifies the 750 ms window as one image flash plus the following gray interval.

## 4-c. How is `output` *Image change* thresholded into categories?

i. There is no additional thresholding step: the output is directly constructed as a binary categorical variable with values `0` and `1`.

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

iii. The AI treats image change as already categorical in `CONVERSION_NOTES.md` Step 5.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The change signal is computed on the same `trial_ts` sample times used for the neural data, so it is aligned pointwise to the resampled neural matrix.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
```

iii. The AI’s justification is the same shared 30 Hz timebase used for all streams before trial segmentation.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes directly from the NWB running-speed acquisition stream: `processing/running/speed/data` and its `timestamps`.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI maps `running/speed/data` to the running-speed decoder output.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI first pooled raw full-session running traces across experiments to compute global 5-bin percentile edges. Within each experiment it linearly interpolated running speed to the 30 Hz regular grid, then digitized each trial sample into those global percentile bins.

ii.
```python
all_running_values.append(running.astype(np.float32))
...
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
...
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. `CONVERSION_NOTES.md` Step 5-6 says running speed should be interpolated to 30 Hz and discretized into five equal percentile bins globally.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five equal-percentile bins computed globally across all experiments’ running-speed values.

ii.
```python
def compute_percentile_bins(values, n_bins=5):
    valid = values[~np.isnan(values)]
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
    ...

running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
...
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. The AI repeatedly states in `CONVERSION_NOTES.md` that running should use five equal percentile bins.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is first interpolated to the shared 30 Hz regular grid, then trial slices use the same `trial_time_indices` as the neural data.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
trial_time_indices = np.where(trial_mask)[0]
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
running_trial = running_resampled[trial_time_indices]
```

iii. The AI’s justification is the same as for image identity and change: all streams are put on one common 30 Hz timeline.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Although the output is named “pupil diameter,” the AI actually derived it from the eye-tracking pupil `area` signal and its timestamps, plus the `likely_blink` mask.

ii.
```python
pupil_tracking = f['acquisition']['EyeTracking']['pupil_tracking']
pupil_area = pupil_tracking['area'][()]
pupil_ts = pupil_tracking['timestamps'][()]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI explicitly says “Use pupil area as proxy for diameter.”

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI marked blink frames as NaN, linearly interpolated across NaNs in the native pupil-area trace, resampled the result to the 30 Hz regular grid, computed global percentile bin edges from all non-NaN full-session pupil-area values, and digitized the trial samples. If pupil tracking was missing, it filled the whole trial with NaNs, which later became bin 0.

ii.
```python
if likely_blink is not None:
    pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
valid_pupil = pupil_area[~np.isnan(pupil_area)]
if len(valid_pupil) > 0:
    all_pupil_values.append(valid_pupil.astype(np.float32))
...
if pupil_resampled is not None:
    pupil_trial = pupil_resampled[trial_time_indices]
else:
    pupil_trial = np.full(n_tp, np.nan)
...
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. `CONVERSION_NOTES.md` Step 5-6 says blinks should be interpolated out and pupil should be discretized globally into five bins.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The AI thresholded pupil values into five global percentile bins.

ii.
```python
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
...
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
...
output_values = [
    ...,
    [f'bin_{i}' for i in range(5)],
    ['hit', 'miss', 'false_alarm', 'correct_reject'],
]
```

iii. This matches the AI’s Step 5 plan for “five equal percentile bins.”

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil area is resampled to the shared 30 Hz regular grid, then trial slices use the same timestamp indices as the neural data.

ii.
```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
trial_time_indices = np.where(trial_mask)[0]
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
pupil_trial = pupil_resampled[trial_time_indices]
```

iii. The AI’s general alignment rationale is the common 30 Hz timeline for all streams.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

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
...
if trials['hit'][trial_idx]:
    outcome = 0
elif trials['miss'][trial_idx]:
    outcome = 1
elif trials['false_alarm'][trial_idx]:
    outcome = 2
elif trials['correct_reject'][trial_idx]:
    outcome = 3
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI maps these four trial flags directly to the outcome output.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI encoded the four outcome classes as integers `0..3` and then broadcast the single per-trial label across all time bins in that trial when assembling the output matrix.

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
...
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)
```

iii. `CONVERSION_NOTES.md` Step 5 says trial outcome is a static per-trial categorical variable; the code represents that as a constant row over time.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI added several ad hoc failure and missing-data behaviors:
- experiments with no valid ROI or no stimulus table are skipped;
- any load exception causes the experiment to be skipped;
- missing pupil data yields all-NaN pupil trials, which are later binned to 0;
- blink-related NaNs are interpolated across;
- very short trials (`<3` samples after resampling) are dropped;
- trials with no recognized outcome are dropped.

ii.
```python
if n_valid == 0:
    ...
    return None
...
if stim_key is None:
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
else:
    pupil_trial = np.full(n_tp, np.nan)
...
else:
    continue  # Unknown outcome, skip
```

iii. In `CONVERSION_NOTES.md` the AI frames these as pragmatic handling of sparse calcium events, missing pupil tracking, and blink artifacts.

## 9-a. What are the most time-consuming steps of the code?

i. The slowest parts are repeated NWB file I/O and per-experiment loading/processing. The code opens every file once for image discovery, once for running/pupil percentile collection, and once again for the full data extraction, including the large events matrix.

ii.
```python
# First pass: determine global image names
for _, row in exp_table.iterrows():
    with h5py.File(nwb_path, 'r') as f:
        ...

# Pass 2: Collect running speed and pupil data
for idx, (_, row) in enumerate(exp_table.iterrows()):
    with h5py.File(nwb_path, 'r') as f:
        ...

# Pass 3: Processing experiments
for idx, (_, row) in enumerate(exp_table.iterrows()):
    raw_data = load_experiment_data(nwb_path, eid)
    result = process_single_experiment(raw_data, exp_meta)
```

iii. `CONVERSION_NOTES.md` Step 6 explicitly calls out sequential processing and multiple passes over NWB files as the main efficiency issue.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain scalar or repeated where vectorization was possible:
- `interpolate_to_regular_grid()` loops over features one column at a time;
- `get_image_at_timepoints()` loops over each sample after `searchsorted`;
- `get_image_change_at_timepoints()` loops over each change interval;
- the code performs three separate `iterrows()` passes over the experiment table.

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
...
for _, row in exp_table.iterrows():
    ...
```

iii. The AI’s notes mention vectorized interpolation and `searchsorted`, but the final code still leaves these loops and repeated table scans in place.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats file-level scanning three times. It separately re-reads every NWB file to collect image names, then to collect running/pupil distributions, then again to load the same experiments for actual conversion. It also rebuilds local-to-global image maps for every experiment.

ii.
```python
print("\n--- Pass 1: Collecting global image names ---")
for _, row in exp_table.iterrows():
    with h5py.File(nwb_path, 'r') as f:
        ...

print("\n--- Pass 2: Collecting running speed and pupil data for percentile bins ---")
for idx, (_, row) in enumerate(exp_table.iterrows()):
    with h5py.File(nwb_path, 'r') as f:
        ...

print("\n--- Pass 3: Processing experiments ---")
for idx, (_, row) in enumerate(exp_table.iterrows()):
    raw_data = load_experiment_data(nwb_path, eid)
    ...
```

iii. `CONVERSION_NOTES.md` Step 6 explicitly says the implementation uses a “3-pass approach.”

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does some extra work that is not used downstream:
- it imports `ProcessPoolExecutor` / `as_completed` but never uses them;
- it reads `cell_specimen_id`, `n_trials`, `change_stops`, and creates `trial_outcomes` / `trial_outcome` temporaries that are unused;
- it computes percentile bins from full-session running/pupil values including periods outside the kept trial windows;
- it stores and decodes some metadata fields that do not feed the final output arrays.

ii.
```python
from concurrent.futures import ProcessPoolExecutor, as_completed
...
cell_specimen_ids = cell_table['cell_specimen_id'][()]
...
n_trials = len(trials_grp['start_time'][()])
...
change_stops = stim_data['stop_time'][change_mask]
...
trial_outcomes = []
...
trial_outcome = np.array([trial_data_out['trial_outcome']], dtype=np.int64)
```

iii. The AI never described these as intentional. They appear to be leftovers from development or convenience choices rather than required processing.
