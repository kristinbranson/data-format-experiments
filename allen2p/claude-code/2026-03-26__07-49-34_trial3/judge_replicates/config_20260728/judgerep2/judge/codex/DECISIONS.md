# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads metadata from the on-disk `ophys_experiment_table.csv`, filters it to NWB files that physically exist and to a hard-coded set of active session types, and then opens each NWB file directly with `h5py`. It does not use the Allen SDK cache or `get_ophys_experiment_table()` / `get_behavior_ophys_experiment()`.

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
...
with h5py.File(nwb_path, 'r') as f:
    ...
```

iii. In `CONVERSION_NOTES.md`, the AI says it has only a partial on-disk subset and therefore works directly from the local NWB files and metadata tables. It also states that passive sessions should be excluded and that using `h5py` gives access to the same underlying NWB content as the SDK.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined by unique `mouse_id` values from the experiment metadata table.

ii. 
```python
subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])
```

iii. The notes explicitly say “Subject IDs: Use `mouse_id` from experiment table,” treating `mouse_id` as the canonical mouse identifier.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB experiment file (`ophys_experiment_id`) as one output session. It does not group multiple experiments from the same `ophys_session_id` into one session.

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

iii. In the notes and trajectory, the AI explicitly justifies this by saying each NWB file is one imaging plane with its own neurons, so each experiment should be one “session” in the decoder output.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB `intervals/trials` table. For each valid trial, the AI uses the trial’s `start_time` and `stop_time` to cut a variable-length slice from a regular 30 Hz timeline spanning the full experiment.

ii. 
```python
trials_grp = f['intervals']['trials']
...
for trial_idx in valid_indices:
    t_trial_start = trials['start_time'][trial_idx]
    t_trial_stop = trials['stop_time'][trial_idx]
    trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
    trial_time_indices = np.where(trial_mask)[0]
```

iii. The notes say the “Trial window” is `start_time` to `stop_time` from the trials table and that trials should remain variable length.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to `(go OR catch) AND NOT aborted AND NOT auto_rewarded`. Trials with fewer than 3 resampled time bins are skipped, and entire experiments are skipped if fewer than 2 valid trials remain. Trials with no recognized outcome are also skipped.

ii. 
```python
valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
valid_indices = np.where(valid_mask)[0]
...
if len(valid_indices) < 2:
    ...
if len(trial_time_indices) < 3:
    continue
...
else:
    continue  # Unknown outcome, skip
```

iii. The notes justify excluding aborted and auto-rewarded trials because the task instructions asked for Go and Catch only. They also require at least 2 trials per experiment for decoder usability.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `processing/ophys/event_detection/data`, after applying the ROI-quality mask `valid_roi`.

ii. 
```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
...
events_data = f['processing']['ophys']['event_detection']['data'][()]
events_valid = events_data[:, valid_roi]
```

iii. The notes state that the paper analyzed discrete calcium events rather than dF/F, so the AI chose event-detection output to match that interpretation of the paper.

## 2-b. How is the `neural` data processed?

i. The AI linearly interpolates the event matrix from native ophys timestamps onto a regular 30 Hz grid for the whole experiment, clips any negative interpolation artifacts to zero, and then transposes each trial slice to `(n_neurons, n_timepoints)`.

ii. 
```python
dt = 1.0 / target_rate
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The notes justify this with “Paper analysis: neural data interpolated to 30 Hz” and with the desire to put Scientifica and Mesoscope recordings on a common timebase.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells are filtered using the NWB `valid_roi` flag. If an experiment has zero valid ROIs, it is dropped.

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

iii. The notes identify `valid_roi` as the Allen QC flag produced by an ROI classifier and say only `valid_roi=True` cells should be included.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to trial start in the final implementation: for each trial, the AI selects the regular-grid samples whose timestamps fall between `start_time` and `stop_time`. The metadata also names the alignment event as trial start.

ii. 
```python
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
...
'temporal_alignment_event': 'Trial start time (first stimulus onset of trial)',
'off_start': 0.0,
'off_end': None,
```

iii. The notes explicitly say “Alignment event: Trial start time (stimulus onset)” and “Trial window: `start_time` to `stop_time`.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 30 Hz grid, i.e. `1000 / 30 = 33.33 ms` per bin. Yes, temporal resampling is applied to the neural data before trial extraction.

ii. 
```python
TARGET_RATE_HZ = 30.0
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ
...
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
```

iii. The AI repeatedly cites the paper’s statement about interpolation to a consistent 30 Hz timestamp grid as the reason for this choice.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. `output` image identity is derived from the stimulus-presentation table, specifically `stimulus_presentations.image_name` after excluding omitted flashes.

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

iii. The notes say image identity should come from stimulus presentations, should persist through gray periods as the most recently shown image, and should ignore omitted flashes.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI builds a sorted per-experiment image list, maps image names to integer indices, and for each timepoint assigns the most recent non-omitted image onset using `searchsorted`. Later it remaps those local integer codes to a global image-code list gathered across experiments.

ii. 
```python
all_stim_images = sorted(set(
    raw_data['stim']['image_name'][raw_data['stim']['image_name'] != 'omitted']
))
...
name_to_idx = {name: i for i, name in enumerate(all_image_names)}
insert_idx = np.searchsorted(stim_starts, timepoints, side='right') - 1
for i in range(n_tp):
    if insert_idx[i] >= 0:
        img_name = stim_names[insert_idx[i]]
        image_idx[i] = name_to_idx.get(img_name, 0)
...
img_id_global = np.array([local_to_global.get(v, 0) for v in trial_data_out['image_identity']], dtype=np.int64)
```

iii. The notes justify this as preserving image identity during gray gaps and omitted flashes by carrying forward the most recent real image.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same `trial_ts` regular-grid timestamps used to slice the neural data, so it is aligned timepoint-by-timepoint with each neural trial matrix.

ii. 
```python
trial_ts = regular_ts[trial_time_indices]
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
```

iii. The notes say all streams are resampled to the same 30 Hz timebase before trial segmentation.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. `output` image change is derived from the stimulus-presentation table, using `stimulus_presentations.is_change`, `start_time`, and `omitted`.

ii. 
```python
change_mask = stim_data['is_change'] & ~stim_data['omitted']
change_starts = stim_data['start_time'][change_mask]
change_stops = stim_data['stop_time'][change_mask]
```

iii. The notes describe image change as coming from stimulus timing plus whether the flash is a real change, not from trial outcome labels.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For every non-omitted change flash, the AI sets a binary signal to 1 from the change onset through the next 0.75 s, corresponding to one image-on plus gray interval.

ii. 
```python
change_signal = np.zeros(n_tp, dtype=np.int64)
...
for cs, ce in zip(change_starts, change_stops):
    mask = (timepoints >= cs) & (timepoints < cs + 0.75)
    change_signal[mask] = 1
```

iii. The notes justify the 750 ms duration from the task structure: 250 ms stimulus plus 500 ms gray.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is directly represented as a binary categorical signal: `0 = no_change`, `1 = change`.

ii. 
```python
output_values = [
    global_image_names,
    ['no_change', 'change'],
    [f'bin_{i}' for i in range(5)],
    [f'bin_{i}' for i in range(5)],
    ['hit', 'miss', 'false_alarm', 'correct_reject'],
]
```

iii. The AI treats image change as inherently binary, so no additional thresholding beyond constructing the 0/1 signal is needed.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like image identity, the change signal is computed on `trial_ts`, which is the same 30 Hz timeline used for the neural trial slice.

ii. 
```python
trial_ts = regular_ts[trial_time_indices]
...
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
...
output_tv = np.stack([img_id_global, img_change, running_binned, pupil_binned], axis=0)
```

iii. The notes say all behavioral and stimulus streams are aligned by resampling to the same regular timestamp grid.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and its paired `timestamps`.

ii. 
```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. The notes identify this as the standard running-wheel signal in the dataset.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI first collects running-speed samples from all included experiments to compute global percentile bin edges. Per experiment, it then linearly interpolates running speed to the 30 Hz regular grid and discretizes each trial’s samples into 5 bins.

ii. 
```python
running = f['processing']['running']['speed']['data'][()]
all_running_values.append(running.astype(np.float32))
...
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
...
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. The notes justify this with the requirement for 5 equal-percentile bins and with the decision to put all modalities on a common 30 Hz timebase.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded into 5 equal-percentile bins computed globally over the collected running-speed values.

ii. 
```python
def compute_percentile_bins(values, n_bins=5):
    valid = values[~np.isnan(values)]
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(valid, percentiles)
...
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
```

iii. The notes explicitly say running speed should be “discretize[d] into 5 percentile bins.”

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same `regular_ts` grid as the neural event data, and each trial uses the same `trial_time_indices` to slice both modalities.

ii. 
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
running_trial = running_resampled[trial_time_indices]
```

iii. The notes repeatedly state that all streams are aligned on the same 30 Hz grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The AI derives this output from eye-tracking `pupil_tracking/area` and `likely_blink`, not from a width or diameter field.

ii. 
```python
pupil_tracking = f['acquisition']['EyeTracking']['pupil_tracking']
pupil_area = pupil_tracking['area'][()]
pupil_ts = pupil_tracking['timestamps'][()]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
```

iii. The notes explicitly say “Use pupil area as proxy for diameter.”

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI marks blink frames as `NaN`, fills missing values within the pupil-area trace by 1D linear interpolation, resamples the result to the 30 Hz grid, and discretizes the values into 5 global percentile bins.

ii. 
```python
if likely_blink is not None:
    pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
pupil_area = f['acquisition']['EyeTracking']['pupil_tracking']['area'][()].astype(float)
blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
pupil_area[blink] = np.nan
valid_pupil = pupil_area[~np.isnan(pupil_area)]
...
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
```

iii. The notes justify this as blink handling plus the same 30 Hz / percentile-binning strategy used for other outputs.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is thresholded into 5 equal-percentile bins computed globally over valid pupil-area values.

ii. 
```python
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
...
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
```

iii. The notes say pupil should be discretized into 5 percentile bins after blink handling.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil is resampled to the same `regular_ts` grid used for neural events and then sliced with the same per-trial indices.

ii. 
```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
...
pupil_trial = pupil_resampled[trial_time_indices]
```

iii. The notes frame pupil alignment exactly the same way as running-speed alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the trial-table boolean flags `hit`, `miss`, `false_alarm`, and `correct_reject`.

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

iii. The notes identify these four columns as the intended outcome classes for valid Go/Catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps the mutually exclusive outcome booleans to integer codes `0..3`, stores that code per trial, and then broadcasts it across every time bin in the output matrix for that trial.

ii. 
```python
output_trials.append({
    ...
    'trial_outcome': outcome,
})
...
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)
```

iii. The notes describe trial outcome as a static per-trial categorical variable.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases by skipping problematic experiments or trials and by interpolating / defaulting missing behavioral values. Specifically: experiments with no valid ROIs or no stimulus table are skipped; missing pupil data yields all-NaN pupil trials; blink-contaminated pupil samples are interpolated; trials with too few bins or no recognized outcome are skipped; NaNs sent to `digitize_to_bins` become category 0.

ii. 
```python
if n_valid == 0:
    ...
if stim_key is None:
    ...
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

iii. The notes justify this as keeping the pipeline robust on a partial, messy on-disk subset while still producing decoder-ready categorical outputs.

## 9-a. What are the most time-consuming steps of the code?

i. The script’s most time-consuming work is repeated full-file NWB I/O across three passes, plus whole-experiment interpolation/resampling during the main processing pass.

ii. 
```python
# First pass: determine global image names
for _, row in exp_table.iterrows():
    with h5py.File(nwb_path, 'r') as f:
        ...

# Pass 2: Collecting running speed and pupil data
for idx, (_, row) in enumerate(exp_table.iterrows()):
    with h5py.File(nwb_path, 'r') as f:
        ...

# Pass 3: Processing experiments
for idx, (_, row) in enumerate(exp_table.iterrows()):
    raw_data = load_experiment_data(nwb_path, eid)
    result = process_single_experiment(raw_data, exp_meta)
```

iii. The notes explicitly describe a 3-pass pipeline and list “multiple passes over NWB files” as a code inefficiency.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities are the per-feature interpolation loop in `interpolate_to_regular_grid`, the per-timepoint loop in `get_image_at_timepoints`, the per-change loop in `get_image_change_at_timepoints`, and several Python-level list comprehensions / trial loops in the assembly phase.

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

iii. The notes mention some vectorization already added (`np.interp`, `searchsorted`) but also acknowledge the code still processes experiments sequentially and uses repeated passes.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats work by reopening every NWB file in three separate passes: one for global image names, one for global running/pupil bin statistics, and one for the actual conversion.

ii. 
```python
print("\n--- Pass 1: Collecting global image names ---")
for _, row in exp_table.iterrows():
    with h5py.File(nwb_path, 'r') as f:
        ...
...
print("\n--- Pass 2: Collecting running speed and pupil data for percentile bins ---")
for idx, (_, row) in enumerate(exp_table.iterrows()):
    with h5py.File(nwb_path, 'r') as f:
        ...
...
print("\n--- Pass 3: Processing experiments ---")
for idx, (_, row) in enumerate(exp_table.iterrows()):
    raw_data = load_experiment_data(nwb_path, eid)
```

iii. This repeated processing is also called out in the notes under “Code inefficiencies identified.”

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script does some extra work that is not needed in the saved decoder dataset: it imports parallel-processing utilities that are never used, reads and stores `cell_specimen_ids` but never uses them, computes `change_stops` but never uses it, allocates `trial_outcome` and `trial_outcomes` temporaries that are unused, and performs two whole-dataset pre-scans whose intermediate arrays are thrown away after global mappings / bin edges are built.

ii. 
```python
from concurrent.futures import ProcessPoolExecutor, as_completed
...
'cell_specimen_ids': cell_specimen_ids[valid_roi],
...
change_stops = stim_data['stop_time'][change_mask]
...
trial_outcomes = []
...
trial_outcome = np.array([trial_data_out['trial_outcome']], dtype=np.int64)
...
del all_running_values, all_pupil_values, all_running_cat, all_pupil_cat
```

iii. The notes do not frame these as correctness problems, but they do acknowledge “multiple passes over NWB files” and sequential processing as inefficiencies.
