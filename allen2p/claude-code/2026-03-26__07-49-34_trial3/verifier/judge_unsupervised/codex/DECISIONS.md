# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads `ophys_experiment_table.csv`, intersects it with the NWB files present on disk, keeps only the active session types in `ACTIVE_SESSION_TYPES`, and then opens each selected NWB with `h5py`. From each file it loads the ROI table, ophys timestamps, event traces, trial table, one stimulus-presentation table, running speed, and pupil tracking.

ii. ```python
DATA_DIR = Path('data/visual-behavior-ophys-1.1.0')
NWB_DIR = DATA_DIR / 'behavior_ophys_experiments'
META_DIR = DATA_DIR / 'project_metadata'

exp_table = pd.read_csv(META_DIR / 'ophys_experiment_table.csv')
mask = (
    exp_table['ophys_experiment_id'].isin(nwb_ids) &
    exp_table['session_type'].isin(ACTIVE_SESSION_TYPES)
)

with h5py.File(nwb_path, 'r') as f:
    cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
    ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][()]
    events_data = f['processing']['ophys']['event_detection']['data'][()]
    trials_grp = f['intervals']['trials']
    running_speed = f['processing']['running']['speed']['data'][()]
    pupil_tracking = f['acquisition']['EyeTracking']['pupil_tracking']
```

iii. In `CONVERSION_NOTES.md`, Steps 1, 5, and 6 say the AllenSDK already loads these NWB-backed objects, that `h5py` should mirror the SDK reader, and that only active behavior sessions should be kept.

## 1-b. How are the data split into subjects?

i. Subjects are defined by `mouse_id` from the experiment metadata table. The code makes a sorted unique subject list and stores one `subject_idx` per output session.

ii. ```python
subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])
```

iii. The notes explicitly say “Subject IDs: Use `mouse_id` from experiment table.”

## 1-c. How are the data split into sessions?

i. The agent treats each NWB `ophys_experiment_id` as one session in the output. It appends one entry to `neural`, `input`, `output`, `subject_idx`, and `brain_region_idx` for each processed experiment.

ii. ```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    ...
    result = process_single_experiment(raw_data, exp_meta)
    ...
    all_neural.append(session_neural)
    all_input.append(session_input)
    all_output.append(session_output)
    all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])
```

iii. The notes say “Each NWB experiment = one ‘session’ in output format,” justified as one imaging plane with its own neurons.

## 1-d. How are the data split into trials?

i. For each kept experiment, the agent uses the Allen trials table. Each trial spans `start_time` to `stop_time`, and the regular 30 Hz ophys-aligned grid is sliced within those bounds.

ii. ```python
for trial_idx in valid_indices:
    t_trial_start = trials['start_time'][trial_idx]
    t_trial_stop = trials['stop_time'][trial_idx]
    trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
    trial_time_indices = np.where(trial_mask)[0]
    neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The notes say “Trial window: Use `trial start_time` to `stop_time` from trials table. Variable length across trials.”

## 1-e. How are trials filtered based on quality controls?

i. The main trial filter keeps Go or Catch trials and drops Aborted and Auto-rewarded trials. Trials with fewer than 3 time bins are skipped, unknown-outcome trials are skipped, and experiments with fewer than 2 kept trials are skipped entirely.

ii. ```python
valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
valid_indices = np.where(valid_mask)[0]
...
if len(trial_time_indices) < 3:
    continue
...
else:
    continue  # Unknown outcome, skip
...
if len(neural_trials) < 2:
    return None
```

iii. The task instructions required Go and Catch only, excluding Aborted and Auto-rewarded. The notes also mention the decoder requirement that each session needs at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `processing/ophys/event_detection/data`, together with `processing/ophys/dff/traces/timestamps` for time alignment, after filtering cells by `valid_roi`.

ii. ```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
ophys_ts = f['processing']['ophys']['dff']['traces']['timestamps'][()]
events_data = f['processing']['ophys']['event_detection']['data'][()]
events_valid = events_data[:, valid_roi]
```

iii. The notes say the paper analyzed “discrete calcium events,” not raw fluorescence or recomputed dF/F.

## 2-b. How is the `neural` data processed?

i. The agent uses the precomputed event traces directly, linearly interpolates them from native ophys timestamps onto a regular 30 Hz grid, clips negative interpolation artifacts to zero, and then slices them trial-by-trial.

ii. ```python
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The notes cite the paper’s “linearly interpolating onto a consistent set of 30hz timestamps” and say event traces are already inferred in the NWB files.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neuron-level QC filter is `valid_roi == True`. Experiments with zero valid ROIs are skipped.

ii. ```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
n_valid = valid_roi.sum()
if n_valid == 0:
    return None
events_valid = events_data[:, valid_roi]
```

iii. The notes say the SDK’s default curation is to exclude invalid ROIs, and that no extra trial/session QC is automatically applied by the SDK.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent aligns all streams on the ophys clock, then defines trials relative to trial start. Metadata describes the alignment event as “Trial start time (first stimulus onset of trial).”

ii. ```python
regular_ts = np.arange(t_start, t_end, dt)
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
...
'temporal_alignment_event': 'Trial start time (first stimulus onset of trial)',
'off_start': 0.0,
'off_end': None,
```

iii. In the notes, the agent explicitly chose trial start as the alignment event because it fit variable-length trials more naturally than change-time alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Everything is resampled to 30 Hz, so each bin is about 33.33 ms. This is done by linear interpolation onto a regular timestamp grid.

ii. ```python
TARGET_RATE_HZ = 30.0
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ
dt = 1.0 / target_rate
regular_ts = np.arange(t_start, t_end, dt)
```

iii. The notes repeatedly justify this with the paper’s statement that analyses used 30 Hz interpolated timestamps.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity comes from the stimulus-presentation table’s `image_name` field. The code also uses `start_time` and `omitted` to decide which identity should be active at each ophys time bin.

ii. ```python
stim_data = {
    'start_time': stim['start_time'][()],
    'stop_time': stim['stop_time'][()],
    'image_name': stim['image_name'][()],
    'is_change': stim['is_change'][()].astype(bool),
    'omitted': stim['omitted'][()],
}
```

iii. The mapping table in the notes says `stimulus_presentations.image_name` is the source for `output[0]: image_identity`.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The code drops omitted flashes, sorts the remaining image names, and for each neural time bin uses the most recent non-omitted stimulus onset. During gray periods and omitted flashes it carries forward the previous image identity. Finally it remaps per-experiment image indices into a global image list.

ii. ```python
non_omitted = ~stim_data['omitted']
stim_starts = stim_data['start_time'][non_omitted]
stim_names = stim_data['image_name'][non_omitted]
insert_idx = np.searchsorted(stim_starts, timepoints, side='right') - 1
...
# During gray screen periods, returns the last shown image.
# For omitted flashes, continues with the previous image.
...
img_id_global = np.array([local_to_global.get(v, 0) for v in trial_data_out['image_identity']], dtype=np.int64)
```

iii. The notes justify this as preserving the image interval context through the 500 ms gray screen and omitted flashes.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed directly on the per-trial regular ophys timestamps used for `neural`, so it has the same number of bins as the neural matrix for each trial.

ii. ```python
trial_ts = regular_ts[trial_time_indices]
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
output_tv = np.stack([img_id_global, img_change, running_binned, pupil_binned], axis=0)
```

iii. The notes say all outputs were made time-varying “per ophys frame” after resampling to 30 Hz.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table’s `is_change`, `start_time`, `stop_time`, and `omitted` fields.

ii. ```python
change_mask = stim_data['is_change'] & ~stim_data['omitted']
change_starts = stim_data['start_time'][change_mask]
change_stops = stim_data['stop_time'][change_mask]
```

iii. The notes describe this as using “stimulus timing” plus the dataset’s change flags.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code builds a binary signal and marks every 30 Hz bin from each non-omitted change onset through the following 0.75 s image interval as `1`.

ii. ```python
change_signal = np.zeros(n_tp, dtype=np.int64)
for cs, ce in zip(change_starts, change_stops):
    mask = (timepoints >= cs) & (timepoints < cs + 0.75)
    change_signal[mask] = 1
```

iii. The notes justify the 0.75 s window with the task structure: 250 ms image plus 500 ms gray.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is treated as a binary categorical variable with values `0 = no_change` and `1 = change`.

ii. ```python
img_change = trial_data_out['image_change'].astype(np.int64)
...
output_values = [
    global_image_names,
    ['no_change', 'change'],
```

iii. The task instructions asked for a binary time-varying output, so the notes document a two-category mapping.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The change signal is evaluated on the same per-trial `trial_ts` bins used to extract neural activity.

ii. ```python
trial_ts = regular_ts[trial_time_indices]
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The notes say all streams were aligned on the common 30 Hz ophys time base before trial slicing.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `processing/running/speed/data` and its associated timestamps.

ii. ```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. The notes identify AllenSDK’s running-speed object as the reference source and describe it as cm/s samples at about 60 Hz.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The code concatenates running traces across experiments to compute five global percentile bins, resamples each experiment’s running trace to the regular 30 Hz grid, then slices and stores per-trial values before discretization.

ii. ```python
all_running_values.append(running.astype(np.float32))
...
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
...
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
running_trial = running_resampled[trial_time_indices]
```

iii. The notes say running speed should be discretized into five equal-percentile bins and aligned to the 30 Hz grid.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The agent thresholds running speed with `np.percentile`-based global bin edges, then digitizes each time point into categories `bin_0` through `bin_4`.

ii. ```python
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
...
[f'bin_{i}' for i in range(5)]
```

iii. The notes explicitly say “five equal percentile bins.”

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same regular ophys-aligned 30 Hz grid and then sliced with the same trial indices as the neural data.

ii. ```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
running_trial = running_resampled[trial_time_indices]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The notes describe this as temporal alignment on ophys timestamps followed by per-trial segmentation.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The agent derives “pupil diameter” from `EyeTracking/pupil_tracking/area`, together with `timestamps` and `likely_blink`.

ii. ```python
pupil_tracking = f['acquisition']['EyeTracking']['pupil_tracking']
pupil_area = pupil_tracking['area'][()]
pupil_ts = pupil_tracking['timestamps'][()]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
```

iii. The notes explicitly say “Use pupil area as proxy for diameter.”

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code marks blink frames as missing, linearly interpolates over NaNs, resamples to the regular 30 Hz grid, collects global percentile edges, and then digitizes per-trial values.

ii. ```python
if likely_blink is not None:
    pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
valid_pupil = pupil_area[~np.isnan(pupil_area)]
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
```

iii. The notes justify the missing-data handling with blink artifacts and say percentile bins should be computed from non-blink values.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The agent uses five global percentile bins and labels them `bin_0` through `bin_4`.

ii. ```python
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
...
[f'bin_{i}' for i in range(5)]
```

iii. The notes use the same five-bin percentile scheme as running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil values are interpolated onto the regular ophys-aligned 30 Hz grid and then sliced with the same trial mask as the neural data.

ii. ```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
pupil_trial = pupil_resampled[trial_time_indices]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The notes say all outputs were aligned to ophys timestamps before per-trial extraction.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. ```python
'hit': trials_grp['hit'][()].astype(bool),
'miss': trials_grp['miss'][()].astype(bool),
'false_alarm': trials_grp['false_alarm'][()].astype(bool),
'correct_reject': trials_grp['correct_reject'][()].astype(bool),
```

iii. The notes map these directly to the four decoder classes for valid Go/Catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code checks the four mutually exclusive outcome flags in priority order, maps them to integers 0-3, skips trials with no recognized outcome, and then broadcasts the static label across all time bins in the stored output array.

ii. ```python
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
...
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
```

iii. The notes describe trial outcome as a static per-trial four-class categorical variable.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The code decodes byte strings, casts float-like boolean fields, skips files with no valid ROIs or no stimulus table, interpolates NaNs in pupil traces, fills fully missing pupil traces with zeros through `interpolate_nans`, assigns NaN binned values to bin 0, clips negative resampled event values to zero, and skips malformed trials or sessions that become too small.

ii. ```python
if isinstance(trials['initial_image_name'][0], bytes):
    trials['initial_image_name'] = np.array([x.decode() for x in trials['initial_image_name']])
...
if stim_data['omitted'].dtype == float:
    stim_data['omitted'] = stim_data['omitted'].astype(bool)
...
if n_valid == 0:
    return None
...
if nans.all():
    return np.zeros_like(arr)
...
binned[np.isnan(values)] = 0
...
events_resampled = np.maximum(events_resampled, 0)
```

iii. The notes say blink-contaminated pupil samples were interpolated, negative event values came only from interpolation, and the agent wanted robust handling of sparse or partially missing streams.

## 9-a. What are the most time-consuming steps of the code?

i. The heaviest work is reopening all NWB files three times, reading large HDF5 arrays, interpolating full-session neural and behavioral traces, and then slicing them trial-by-trial.

ii. ```python
# Pass 1: Collecting global image names
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
```

iii. Step 6 of the notes explicitly calls out the multi-pass file traversal and session processing runtime.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-feature interpolation loop in `interpolate_to_regular_grid`, the per-timepoint loop in `get_image_at_timepoints`, the per-change loop in `get_image_change_at_timepoints`, and the repeated `global_image_names.index(name)` lookup could all be replaced with more vectorized or precomputed forms.

ii. ```python
for i in range(data.shape[1]):
    result[:, i] = np.interp(target_timestamps, timestamps, data[:, i])

for i in range(n_tp):
    if insert_idx[i] >= 0:
        ...

for cs, ce in zip(change_starts, change_stops):
    mask = (timepoints >= cs) & (timepoints < cs + 0.75)
    change_signal[mask] = 1

for local_idx, name in enumerate(result['all_image_names']):
    local_to_global[local_idx] = global_image_names.index(name)
```

iii. The notes mention vectorized interpolation and `searchsorted` as partial optimizations, but several Python loops remain.

## 9-c. What processing does the code repeat multiple times?

i. It rescans interval groups to find the natural-image table in both Pass 1 and the real load step, rereads every NWB multiple times across three passes, and recomputes per-experiment image-name mappings repeatedly.

ii. ```python
for k in f['intervals'].keys():
    if k != 'trials' and 'spontaneous' not in k.lower() and 'movie' not in k.lower():
        ...

for _, row in exp_table.iterrows():   # pass 1
...
for idx, (_, row) in enumerate(exp_table.iterrows()):   # pass 2
...
for idx, (_, row) in enumerate(exp_table.iterrows()):   # pass 3
```

iii. The notes explicitly describe the script as a “3-pass approach.”

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script imports but never uses parallel-processing utilities, creates some unused locals (`trial_outcomes`, `trial_outcome`, `change_stops`, `t_start_global`, `n_trials`), and expands the static trial outcome into a full-length per-time-bin vector even though it is logically one value per trial.

ii. ```python
from concurrent.futures import ProcessPoolExecutor, as_completed
...
n_trials = len(trials_grp['start_time'][()])
...
change_stops = stim_data['stop_time'][change_mask]
...
trial_outcomes = []
...
trial_outcome = np.array([trial_data_out['trial_outcome']], dtype=np.int64)
...
t_start_global = time.time()
...
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
```

iii. These are code-side conveniences or leftovers rather than part of the reference processing itself; the notes also mention abandoned plans for parallelization that never made it into the final implementation.
