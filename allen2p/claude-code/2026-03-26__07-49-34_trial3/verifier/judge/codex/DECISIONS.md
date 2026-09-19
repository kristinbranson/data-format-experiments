# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the AllenSDK cache path used by the reference solution. It reads `ophys_experiment_table.csv`, keeps only experiments that have NWB files on disk and whose `session_type` is one of four active session types, then opens each NWB file directly with `h5py` and extracts neural, trial, stimulus, running, and pupil streams.

ii. ```python
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
    events_data = f['processing']['ophys']['event_detection']['data'][()]
    trials_grp = f['intervals']['trials']
```

iii. In `CONVERSION_NOTES.md`, the AI justified this as operating on the on-disk subset only, excluding passive sessions, and reading the same NWB contents as the SDK while avoiding extra SDK machinery.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values from the experiment metadata table, converted to strings and stored globally.

ii. ```python
subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
'mouse_id': str(exp_meta['mouse_id']),
```

iii. The AI explicitly states in `CONVERSION_NOTES.md` that subject IDs should come from `mouse_id` in the experiment table.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB experiment as a separate session in the output. It does not group multiple experiments that share an `ophys_session_id`.

ii. ```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = NWB_DIR / f'behavior_ophys_experiment_{eid}.nwb'
    ...
    all_neural.append(session_neural)
    all_input.append(session_input)
    all_output.append(session_output)
```

iii. The AI documented this directly: “Each NWB experiment = one ‘session’ in output format (one imaging plane with its own neurons).”

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB `intervals/trials` table. For each valid trial, the AI uses the trial’s `start_time` and `stop_time` and extracts all regular-grid 30 Hz samples that fall inside that interval.

ii. ```python
valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
...
t_trial_start = trials['start_time'][trial_idx]
t_trial_stop = trials['stop_time'][trial_idx]
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
```

iii. In the notes, the AI says the trial window should be `start_time` to `stop_time` and remain variable-length across trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only go or catch trials, excludes aborted and auto-rewarded trials, drops experiments with fewer than 2 valid trials, and skips trials with fewer than 3 time bins or unknown outcome labels.

ii. ```python
valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
valid_indices = np.where(valid_mask)[0]
...
if len(valid_indices) < 2:
    return None
...
if len(trial_time_indices) < 3:
    continue
...
else:
    continue  # Unknown outcome, skip
```

iii. The stated justification is that the task explicitly requires go and catch trials only and excludes aborted and auto-rewarded trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from precomputed event traces in the NWB file, specifically `processing/ophys/event_detection/data`, after applying the `valid_roi` mask.

ii. ```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
...
events_data = f['processing']['ophys']['event_detection']['data'][()]
events_valid = events_data[:, valid_roi]
```

iii. The notes justify this by citing the paper’s use of “discrete calcium events” and choosing events instead of dF/F.

## 2-b. How is the `neural` data processed?

i. The AI linearly interpolates the event traces from native ophys timestamps onto a regular 30 Hz grid and clips any negative interpolated values to zero. It does not merge multiple planes within a session because it treats each experiment as a session.

ii. ```python
dt = 1.0 / target_rate
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The notes say this was chosen to match the paper’s 30 Hz interpolation step and to keep a uniform bin size across single-plane and multiscope recordings.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural QC filter is `valid_roi == True`; experiments with zero valid ROIs are skipped.

ii. ```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
n_valid = valid_roi.sum()
if n_valid == 0:
    print(f"  WARNING: No valid ROIs in experiment {experiment_id}, skipping")
    return None
...
events_valid = events_data[:, valid_roi]
```

iii. The AI’s notes say `valid_roi` is the SDK’s automatic cell-quality flag and should be the only neuron-level curation step.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start. The AI first builds a session-wide 30 Hz grid, then slices each trial from `start_time` to `stop_time`; metadata describe the alignment event as the first stimulus onset of the trial.

ii. ```python
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
...
'temporal_alignment_event': 'Trial start time (first stimulus onset of trial)',
'off_start': 0.0,
'off_end': None,
```

iii. The notes explicitly changed the alignment decision to “Trial start time (stimulus onset)” with a variable trial end.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a regular 30 Hz grid, i.e. about 33.33 ms bins. Yes, temporal rebinning/resampling is applied to the neural data.

ii. ```python
TARGET_RATE_HZ = 30.0
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ
...
'time_bin_size': TIME_BIN_MS,
```

iii. The AI justified this as matching the paper’s “linearly interpolating onto a consistent set of 30hz timestamps.”

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The code derives image identity from the stimulus-presentation table’s `image_name`, `start_time`, and `omitted` fields, not from the trial table’s `initial_image_name` and `change_image_name`.

ii. ```python
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

iii. The notes justify this as preserving the currently displayed image through gray periods and omitted flashes, using the most recent non-omitted stimulus identity.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI builds session-local image indices from the most recent non-omitted stimulus onset at each timepoint, then remaps those local indices into one global sorted image list shared across all experiments.

ii. ```python
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

iii. The AI says image identity during gray should be the last shown image and omitted flashes should inherit the prior image; it also wanted a deterministic global category mapping.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is evaluated on exactly the same 30 Hz trial timestamps used for neural data.

ii. ```python
trial_ts = regular_ts[trial_time_indices]
...
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The justification is implicit in the code and notes: all outputs are first put onto the same regular timebase before trial extraction.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The implemented code derives image change from the stimulus-presentation table’s `is_change`, `start_time`, and `omitted` fields.

ii. ```python
change_mask = stim_data['is_change'] & ~stim_data['omitted']
change_starts = stim_data['start_time'][change_mask]
change_stops = stim_data['stop_time'][change_mask]
```

iii. In the notes, the AI describes the same idea at a higher level as using change markers plus stimulus timing to mark the change interval.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each non-omitted change stimulus, the AI marks a 750 ms window beginning at the change onset as `1`; all other timepoints stay `0`.

ii. ```python
change_signal = np.zeros(n_tp, dtype=np.int64)
for cs, ce in zip(change_starts, change_stops):
    mask = (timepoints >= cs) & (timepoints < cs + 0.75)
    change_signal[mask] = 1
```

iii. The notes justify this as one full flash interval: 250 ms stimulus plus 500 ms gray.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is a binary categorical variable with values `0/1`, later labeled as `no_change` and `change`.

ii. ```python
img_change = trial_data_out['image_change'].astype(np.int64)
...
['no_change', 'change']
```

iii. The AI treats this as a naturally binary output and does not apply any extra thresholding beyond the 750 ms change window.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is computed on the same per-trial 30 Hz timestamps as the neural data and then stacked into the output tensor for that trial.

ii. ```python
trial_ts = regular_ts[trial_time_indices]
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
...
output_tv = np.stack([img_id_global, img_change, running_binned, pupil_binned], axis=0)
```

iii. The code and notes both assume all outputs should be evaluated on the same common temporal grid as neural activity.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from the NWB running stream’s `data` and `timestamps`.

ii. ```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. The AI treats this as the standard locomotion signal for the task.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes global percentile bin edges from raw session-wide running values, interpolates running speed onto the 30 Hz grid with `np.interp`, then digitizes each per-trial trace into 5 bins.

ii. ```python
all_running_values.append(running.astype(np.float32))
...
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
...
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. The notes justify the interpolation by the 30 Hz target timebase and the percentile bins by the decoder instruction requiring five equal-percentile categories.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is discretized into five global percentile bins, with NaNs assigned to bin 0 by `digitize_to_bins`.

ii. ```python
def digitize_to_bins(values, bin_edges):
    binned = np.digitize(values, bin_edges[1:-1])
    ...
    binned[np.isnan(values)] = 0
...
[f'bin_{i}' for i in range(5)]
```

iii. The AI repeatedly states in the notes that running speed should be “discretized into 5 percentile bins.”

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same session-wide 30 Hz grid as neural events and then sliced by the same trial indices.

ii. ```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
running_trial = running_resampled[trial_time_indices]
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The notes frame this as putting all streams on a common 30 Hz timebase before trial extraction.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The AI actually uses pupil **area**, not pupil width/diameter, from `acquisition/EyeTracking/pupil_tracking/area`, together with `timestamps` and `likely_blink`.

ii. ```python
pupil_tracking = f['acquisition']['EyeTracking']['pupil_tracking']
pupil_area = pupil_tracking['area'][()]
pupil_ts = pupil_tracking['timestamps'][()]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
```

iii. The AI’s notes explicitly justify this as using “pupil area as proxy for diameter.”

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink-marked pupil samples are set to NaN, NaNs are linearly interpolated within the raw pupil-area trace, the result is resampled onto the 30 Hz grid, and finally the per-trial values are binned with global percentile edges computed from non-blink session-wide pupil-area samples.

ii. ```python
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

iii. The notes justify this by handling blink artifacts before resampling and by following the five-bin decoder requirement.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is discretized into five global percentile bins, labeled `bin_0` through `bin_4`, with any NaNs mapped to bin 0.

ii. ```python
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
...
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
...
[f'bin_{i}' for i in range(5)]
```

iii. The AI treats pupil the same way as running speed: global percentile discretization to meet the output specification.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil area is resampled onto the same 30 Hz regular grid as neural events and then cut into trials using the same trial indices.

ii. ```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
pupil_trial = pupil_resampled[trial_time_indices]
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The justification is the same common-grid alignment strategy used for all continuous streams.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

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

iii. The notes describe trial outcome as a 4-class categorical variable from these standard trial annotations.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps the four mutually exclusive booleans to integer codes 0-3 and broadcasts the chosen code across all time bins of the trial.

ii. ```python
if trials['hit'][trial_idx]:
    outcome = 0
...
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)
```

iii. The notes justify this as a static per-trial decoder output represented in the same `(n_output, n_timepoints)` format as the time-varying outputs.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI skips unreadable experiments, experiments with no valid ROIs, experiments with too few valid trials, and trials with too few bins or no recognized outcome. Missing pupil data causes NaN-filled trial traces; blink-related NaNs are interpolated before resampling; NaNs that remain after discretization are put in bin 0.

ii. ```python
except Exception as e:
    print(f"  ERROR loading {experiment_id}: {e}")
    return None
...
if n_valid == 0:
    return None
...
if len(valid_indices) < 2:
    return None
...
if len(trial_time_indices) < 3:
    continue
...
else:
    pupil_trial = np.full(n_tp, np.nan)
...
binned[np.isnan(values)] = 0
```

iii. The notes describe this as pragmatic fault tolerance so one bad experiment or missing stream does not crash the entire conversion.

## 9-a. What are the most time-consuming steps of the code?

i. The AI’s implementation is dominated by repeated full-file HDF5 reads and by trial processing after resampling. The code makes three dataset-wide passes: one to collect image names, one to collect running/pupil values for bin edges, and one to do the full conversion.

ii. ```python
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
```

iii. In the notes, the AI explicitly calls the script a “3-pass approach” and flags sequential experiment processing as a speed limitation.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Two obvious loops remain unvectorized: interpolation across cells in `interpolate_to_regular_grid`, and per-timepoint image assignment in `get_image_at_timepoints`.

ii. ```python
for i in range(data.shape[1]):
    result[:, i] = np.interp(target_timestamps, timestamps, data[:, i])
...
for i in range(n_tp):
    if insert_idx[i] >= 0:
        img_name = stim_names[insert_idx[i]]
        image_idx[i] = name_to_idx.get(img_name, 0)
```

iii. The AI notes mention efficiency concerns and partial vectorization, but these loops were left in place in the final implementation.

## 9-c. What processing does the code repeat multiple times?

i. The code reopens every NWB file multiple times across the three global passes. It also reconstructs local-to-global image mappings per experiment and repeatedly uses `global_image_names.index(name)` inside a loop.

ii. ```python
for _, row in exp_table.iterrows():
    with h5py.File(nwb_path, 'r') as f:
        ...
for idx, (_, row) in enumerate(exp_table.iterrows()):
    with h5py.File(nwb_path, 'r') as f:
        ...
for idx, (_, row) in enumerate(exp_table.iterrows()):
    raw_data = load_experiment_data(nwb_path, eid)
...
for local_idx, name in enumerate(result['all_image_names']):
    if name in global_image_names:
        local_to_global[local_idx] = global_image_names.index(name)
```

iii. The AI itself documented the multi-pass design and identified “multiple passes over NWB files” as a code inefficiency.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads or computes several things that are not used downstream: `cell_specimen_ids`, `change_stops`, `trial_outcomes`, a one-element `trial_outcome` array, and imported parallelism utilities that are never used. It also computes percentile edges from all raw session values, including time outside kept trials, even though downstream outputs only use trial-restricted slices.

ii. ```python
from concurrent.futures import ProcessPoolExecutor, as_completed
...
cell_specimen_ids = cell_table['cell_specimen_id'][()]
...
change_stops = stim_data['stop_time'][change_mask]
...
trial_outcomes = []
...
trial_outcome = np.array([trial_data_out['trial_outcome']], dtype=np.int64)
```

iii. The notes acknowledge extra passes and unused intermediate work as tradeoffs made for simplicity rather than for minimal downstream computation.
