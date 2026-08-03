# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads data directly from local NWB files and metadata CSVs, not through the AllenSDK cache. It reads `ophys_experiment_table.csv`, finds experiment IDs that have NWB files on disk, filters to four active session types, then opens each experiment NWB with `h5py` and extracts neural, trial, stimulus, running, and pupil streams.

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
    events_data = f['processing']['ophys']['event_detection']['data'][()]
    trials_grp = f['intervals']['trials']
    stim = f['intervals'][stim_key]
```

iii. In `CONVERSION_NOTES.md`, the agent justified this as matching the on-disk subset and avoiding dependence on the AllenSDK cache. It also decided to exclude passive sessions and to work from the local NWB structure it had inspected.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values from the experiment metadata table after filtering to on-disk active experiments. A session’s `subject_idx` is assigned from this mapping.

ii. 
```python
subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])
```

iii. The notes state that subject IDs should come from `mouse_id` in the experiment table.

## 1-c. How are the data split into sessions?

i. The agent treats each NWB experiment file as one output session. It does not group multiple experiments that share an `ophys_session_id`.

ii. 
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    nwb_path = NWB_DIR / f'behavior_ophys_experiment_{eid}.nwb'
    ...
    all_neural.append(session_neural)
```

iii. In the trajectory and notes, the agent explicitly reasoned that each NWB experiment is one imaging plane with its own neurons, so it should become one decoder “session”.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. For each valid trial, the agent takes all regularly resampled timestamps from `start_time` to `stop_time`, so trials are variable-length windows covering the full trial.

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

iii. The notes say the intended trial window was `start_time` to `stop_time`, with variable duration across trials.

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps trials where `go` or `catch` is true and excludes `aborted` and `auto_rewarded`. It also drops experiments with fewer than 2 such trials, skips trials with fewer than 3 resampled time bins, and skips trials whose outcome is not one of hit/miss/false_alarm/correct_reject.

ii. 
```python
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

iii. The notes justify excluding aborted and auto-rewarded trials per instructions, and keeping only Go/Catch trials. The trajectory also mentions passive sessions being excluded because they are not active behavior sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `processing/ophys/event_detection/data` in the NWB file, filtered to ROIs where `valid_roi` is true.

ii. 
```python
cell_table = f['processing']['ophys']['image_segmentation']['cell_specimen_table']
valid_roi = cell_table['valid_roi'][()].astype(bool)
...
events_data = f['processing']['ophys']['event_detection']['data'][()]
events_valid = events_data[:, valid_roi]
```

iii. The notes and trajectory say the agent chose deconvolved calcium events instead of dF/F because it believed the paper analyzed discrete calcium events and that these were closer to spike-like activity.

## 2-b. How is the `neural` data processed?

i. The agent resamples event traces from native ophys timestamps to a uniform 30 Hz grid by linear interpolation, clips negative interpolated values to zero, then transposes trial slices into `(n_cells, n_timepoints)`.

ii. 
```python
regular_ts = np.arange(t_start, t_end, dt)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. `CONVERSION_NOTES.md` says the agent wanted to match the paper’s “30 Hz” analysis timebase, and to use raw events rather than SDK-filtered events.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural QC is filtering to `valid_roi == True`. Experiments with zero valid ROIs are skipped.

ii. 
```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
...
n_valid = valid_roi.sum()
if n_valid == 0:
    return None
...
events_valid = events_data[:, valid_roi]
```

iii. The notes identify `valid_roi` as the SDK/whitepaper ROI-quality flag and state that only valid ROIs should be included.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start: each trial contains resampled neural activity from `start_time` to `stop_time`, and metadata labels the alignment event as trial start time.

ii. 
```python
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
...
'temporal_alignment_event': 'Trial start time (first stimulus onset of trial)',
'off_start': 0.0,
```

iii. The notes list “alignment event: trial start time (stimulus onset)” as a key decision.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses fixed 30 Hz bins (`33.33 ms`). Yes: all neural, running, and pupil streams are interpolated onto this regular grid.

ii. 
```python
TARGET_RATE_HZ = 30.0
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ
...
regular_ts = np.arange(t_start, t_end, dt)
...
'time_bin_size': TIME_BIN_MS,
```

iii. The notes explicitly justify this with the paper phrase about “linearly interpolating onto a consistent set of 30hz timestamps.”

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table, specifically stimulus `image_name`, `start_time`, and `omitted` status, rather than from per-trial `initial_image_name` / `change_image_name`.

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

iii. The notes justify this by saying image identity should reflect the actual most recent non-omitted image on screen and persist through gray periods and omitted flashes.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The agent builds a global image-name vocabulary, then for each trial timepoint it assigns the most recent non-omitted stimulus identity using `searchsorted`. During gray periods and omitted flashes, it carries forward the previous image identity. Before the first stimulus in a trial, it falls back to category 0.

ii. 
```python
global_image_names = sorted(all_image_names_set)
...
insert_idx = np.searchsorted(stim_starts, timepoints, side='right') - 1
for i in range(n_tp):
    if insert_idx[i] >= 0:
        img_name = stim_names[insert_idx[i]]
        image_idx[i] = name_to_idx.get(img_name, 0)
    else:
        image_idx[i] = 0
```

iii. The notes call out two explicit decisions: keep the previous image during gray screen periods, and keep the previous image for omitted flashes.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is computed on the exact same per-trial resampled timestamps (`trial_ts`) used for the neural data, so each neural bin has one image-identity label.

ii. 
```python
trial_ts = regular_ts[trial_time_indices]
...
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
```

iii. The agent’s justification was to align all decoded outputs to the same 30 Hz timebase as neural activity.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus presentations table using `is_change`, `start_time`, and `omitted`, not from trial-level `change_time` / `go`.

ii. 
```python
change_mask = stim_data['is_change'] & ~stim_data['omitted']
change_starts = stim_data['start_time'][change_mask]
change_stops = stim_data['stop_time'][change_mask]
```

iii. The notes say image change should mark the 750 ms change interval following an actual change stimulus.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each change stimulus in the session, the agent marks bins from change onset through 750 ms later as 1 and all other bins as 0.

ii. 
```python
change_signal = np.zeros(n_tp, dtype=np.int64)
for cs, ce in zip(change_starts, change_stops):
    mask = (timepoints >= cs) & (timepoints < cs + 0.75)
    change_signal[mask] = 1
```

iii. The notes justify the 750 ms window as one 250 ms image flash plus the 500 ms gray interval.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is directly represented as a binary categorical output: `0 = no_change`, `1 = change`.

ii. 
```python
img_change = trial_data_out['image_change'].astype(np.int64)
...
['no_change', 'change']
```

iii. The notes describe this as a binary variable active only during the change interval.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is computed on `trial_ts`, the same resampled timestamps used to slice the neural trial, so it is binwise aligned to neural activity.

ii. 
```python
trial_ts = regular_ts[trial_time_indices]
...
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The agent’s 30 Hz common-timebase decision was the stated justification.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from NWB datasets `processing/running/speed/data` and `processing/running/speed/timestamps`.

ii. 
```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. The notes identify running speed as the standard locomotion signal to decode.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the 30 Hz regular grid, and later discretized using five global percentile bins computed across all experiments’ raw running samples.

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

iii. The notes justify linear interpolation plus global percentile bins so categories are consistent across the dataset.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is discretized into five equal-percentile bins using global bin edges. NaNs are assigned to bin 0.

ii. 
```python
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
...
binned = np.digitize(values, bin_edges[1:-1])
...
binned[np.isnan(values)] = 0
```

iii. The notes say the target format requires categorical outputs and that five percentile bins were chosen per instructions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is first resampled onto the same regular 30 Hz grid as neural data, then per-trial slices are taken using the same `trial_time_indices`.

ii. 
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
...
running_trial = running_resampled[trial_time_indices]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The notes repeatedly describe a shared 30 Hz timebase for all modalities.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The agent derives this output from NWB eye-tracking `pupil_tracking/area` and `timestamps`, together with `likely_blink`.

ii. 
```python
pupil_tracking = f['acquisition']['EyeTracking']['pupil_tracking']
pupil_area = pupil_tracking['area'][()]
pupil_ts = pupil_tracking['timestamps'][()]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
```

iii. The notes explicitly say the agent used pupil area “as proxy for diameter”.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink frames are set to NaN, NaN gaps are linearly interpolated in the native pupil trace, the result is resampled to 30 Hz, and then values are discretized using five global percentile bins. If no pupil stream exists, the trial is filled with NaNs until binning.

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

iii. The notes justify blink handling as necessary cleanup, and say percentile bins should be computed from non-blink data only.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is discretized into five equal-percentile bins using global bin edges, with NaNs assigned to bin 0.

ii. 
```python
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
...
binned[np.isnan(values)] = 0
```

iii. The notes describe the same five-bin percentile discretization used for running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil values are resampled to the same 30 Hz grid as neural data, then trial slices use the same `trial_time_indices`.

ii. 
```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
pupil_trial = pupil_resampled[trial_time_indices]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The shared-`regular_ts` design is the agent’s stated alignment rationale.

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

iii. The notes identify these four trial-outcome fields as the target categories.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The agent maps the outcome to integer codes `0..3` and then broadcasts the chosen code across all time bins in the trial when assembling the final output tensor.

ii. 
```python
outcome_broadcast = np.full((1, n_tp), trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)
```

iii. The notes describe trial outcome as a static per-trial categorical output.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent skips failed loads, experiments with no valid ROIs, experiments with too few valid trials, very short trials, and trials with unknown outcomes. Missing pupil streams become all-NaN arrays; blink-corrupted pupil samples are interpolated; NaNs at discretization time are assigned to bin 0. Constant-valued percentile edges are forced to be strictly increasing with small epsilons.

ii. 
```python
if n_valid == 0:
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
binned[np.isnan(values)] = 0
...
if edges[i] <= edges[i-1]:
    edges[i] = edges[i-1] + 1e-10
```

iii. The notes justify these choices as defensive handling for blinks, missing pupil data, and edge cases that would otherwise break downstream decoding.

## 9-a. What are the most time-consuming steps of the code?

i. The code is dominated by repeated file I/O and session processing: it scans all NWB files once for image names, again for running/pupil percentile statistics, and a third time for full conversion. Within each experiment, interpolation and per-trial slicing are the main processing costs.

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

iii. `CONVERSION_NOTES.md` explicitly describes a “3-pass approach” and separately notes sequential experiment processing as an inefficiency.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still contains several scalar or Python-level loops that could be vectorized: per-feature interpolation in `interpolate_to_regular_grid`, per-timepoint image assignment in `get_image_at_timepoints`, per-change masking in `get_image_change_at_timepoints`, and repeated `global_image_names.index(name)` lookups while remapping local image codes.

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
for local_idx, name in enumerate(result['all_image_names']):
    if name in global_image_names:
        local_to_global[local_idx] = global_image_names.index(name)
```

iii. The notes mention sequential experiment processing and multiple passes as efficiency issues, and say vectorized interpolation/searchsorted were already partly used.

## 9-c. What processing does the code repeat multiple times?

i. The agent repeats file opening and scanning across three full passes over the experiment list: one to collect image names, one to collect running/pupil values for percentile bins, and one to actually convert experiments. It also recomputes image-change masks trial by trial from full-session change lists.

ii. 
```python
print("\n--- Pass 1: Collecting global image names ---")
...
print("\n--- Pass 2: Collecting running speed and pupil data for percentile bins ---")
...
print("\n--- Pass 3: Processing experiments ---")
```

iii. The notes explicitly call out the “3-pass approach” and list “multiple passes over NWB files” as an identified inefficiency.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code performs some work that is not used downstream: `change_stops` is computed but unused, `trial_outcome = np.array([..])` is created but then ignored, `ProcessPoolExecutor`/`as_completed` are imported but unused, and the code builds per-experiment local image vocabularies only to remap them again to global codes later.

ii. 
```python
from concurrent.futures import ProcessPoolExecutor, as_completed
...
change_stops = stim_data['stop_time'][change_mask]
...
trial_outcome = np.array([trial_data_out['trial_outcome']], dtype=np.int64)
...
all_stim_images = sorted(set(
    raw_data['stim']['image_name'][raw_data['stim']['image_name'] != 'omitted']
))
...
local_to_global[local_idx] = global_image_names.index(name)
```

iii. The notes do not call all of these out explicitly, but they do acknowledge avoidable inefficiency and repeated processing. The discarded intermediate `trial_outcome` array and unused imports are apparent from the code itself.
