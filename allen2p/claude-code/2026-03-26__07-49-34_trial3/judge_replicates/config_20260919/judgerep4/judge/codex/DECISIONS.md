# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the local experiment metadata CSV, finds NWB files physically present, restricts them to four active session types, and loads each selected NWB directly with `h5py`. It makes three passes: image-name collection, behavior-value collection for bin edges, and full experiment processing.

ii.
```python
exp_table = pd.read_csv(META_DIR / 'ophys_experiment_table.csv')
nwb_files = list(NWB_DIR.glob('*.nwb'))
mask = (exp_table['ophys_experiment_id'].isin(nwb_ids) &
        exp_table['session_type'].isin(ACTIVE_SESSION_TYPES))
active_exps = exp_table[mask].copy()
with h5py.File(nwb_path, 'r') as f:
    ...
```

iii. The notes say the disk contains a partial release and passive OPHYS_2/5 recordings lack meaningful behavioral outcomes, so only the 202 on-disk active experiments are selected. Direct NWB loading was chosen after inspecting AllenSDK's NWB accessors.

## 1-b. How are the data split into subjects?

i. Subjects are unique metadata `mouse_id` values, converted to strings and sorted; every output experiment receives the corresponding subject index.

ii.
```python
subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])
```

iii. The notes identify `mouse_id` as the subject identifier and report 38 mice in the on-disk subset.

## 1-c. How are the data split into sessions?

i. Each selected `ophys_experiment_id`/NWB file is treated as one output session. Simultaneously recorded imaging planes sharing an `ophys_session_id` are not combined.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    ...
    all_neural.append(session_neural)
```

iii. The mapping plan explicitly states: “Each NWB experiment = one ‘session’ in output format (one imaging plane with its own neurons).”

## 1-d. How are the data split into trials?

i. The built-in NWB trials table defines trials. For every retained trial, samples on the regular grid satisfying `start_time <= t < stop_time` are selected, producing variable-length trials.

ii.
```python
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The notes justify using the trials table's full start-to-stop interval and describe alignment to trial start, with variable end offsets.

## 1-e. How are trials filtered based on quality controls?

i. It retains Go or Catch trials, excludes aborted and auto-rewarded trials, skips trials with fewer than three resampled points or without a recognized outcome, and drops experiments with fewer than two retained trials.

ii.
```python
valid_mask = (trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']
if len(trial_time_indices) < 3:
    continue
...
else:
    continue  # Unknown outcome, skip
if len(neural_trials) < 2:
    return None
```

iii. Excluding aborted and auto-rewarded trials follows the task directly; the notes say valid trials have exactly one of the four standard outcomes.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the NWB `processing/ophys/event_detection/data` calcium-event array, filtered using `cell_specimen_table/valid_roi`.

ii.
```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
events_data = f['processing']['ophys']['event_detection']['data'][()]
events_valid = events_data[:, valid_roi]
```

iii. The notes state that the paper analyzed discrete FastLZeroSpikeInference calcium events rather than dF/F, and that the SDK normally excludes invalid ROIs.

## 2-b. How is the `neural` data processed?

i. Valid-ROI event traces are linearly interpolated from native ophys timestamps to a regular 30 Hz grid, clipped nonnegative, sliced by trial, transposed to neuron-by-time, and cast to float32.

ii.
```python
regular_ts = np.arange(t_start, t_end, 1.0 / target_rate)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. It cites the paper's statement that signals were linearly interpolated to consistent 30 Hz timestamps and notes the need to unify ~31 Hz and ~11 Hz equipment.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only cells whose `valid_roi` flag is true are included; experiments with no valid ROIs are skipped. No further cell-level filtering is done.

ii.
```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
if valid_roi.sum() == 0:
    return None
events_valid = events_data[:, valid_roi]
```

iii. The notes describe `valid_roi` as the Allen SVM-based cell-quality filter and identify it as the SDK's default curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to trial `start_time`; all regular-grid samples until (but excluding) `stop_time` are included. Metadata records trial start as the alignment event, `off_start=0`, and a variable end.

ii.
```python
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
...
'temporal_alignment_event': 'Trial start time (first stimulus onset of trial)',
'off_start': 0.0,
'off_end': None,
```

iii. The notes say the full trial permits time-varying stimulus and behavior outputs while preserving the task-defined trial structure.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output is 30 Hz (33.333 ms bins). All native neural and behavioral streams are interpolated onto a new regular grid; this is resampling rather than aggregation into count bins.

ii.
```python
TARGET_RATE_HZ = 30.0
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ
regular_ts = np.arange(t_start, t_end, dt)
```

iii. The notes cite the paper's 30 Hz interpolation and argue that a common grid is needed across microscope types.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus-presentation table's `start_time`, `image_name`, and `omitted`, rather than from the trial table's initial/change image fields.

ii.
```python
stim_data = {'start_time': stim['start_time'][()],
             'image_name': stim['image_name'][()],
             'omitted': stim['omitted'][()]}
non_omitted = ~stim_data['omitted']
```

iii. The notes say this supports the actual flashed-image sequence, carrying the previous image across gray screens and omitted flashes.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each time point, `searchsorted` finds the most recent non-omitted stimulus onset. Its image gets a session-local sorted code, which is then remapped to a global sorted image-name vocabulary.

ii.
```python
insert_idx = np.searchsorted(stim_starts, timepoints, side='right') - 1
image_idx[i] = name_to_idx.get(stim_names[insert_idx[i]], 0)
...
img_id_global = np.array([local_to_global.get(v, 0)
                          for v in trial_data_out['image_identity']])
```

iii. The mapping plan explicitly chooses “last shown image” during gray and continuity over omissions; a global vocabulary provides consistent output codes.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is evaluated directly at the same per-trial 30 Hz timestamps used to slice neural events, so its length and sample positions match neural time points.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
```

iii. The notes identify interpolation and timestamp lookup on the common 30 Hz timebase as the alignment mechanism.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change comes from stimulus-presentation `is_change`, `omitted`, and presentation `start_time` (the loaded stop times are not actually used in the computation).

ii.
```python
change_mask = stim_data['is_change'] & ~stim_data['omitted']
change_starts = stim_data['start_time'][change_mask]
change_stops = stim_data['stop_time'][change_mask]
```

iii. The notes describe this as marking the first flash after an actual change and excluding omissions.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector is created and each actual change onset marks a fixed 750 ms interval, representing the 250 ms image plus subsequent 500 ms gray period.

ii.
```python
for cs, ce in zip(change_starts, change_stops):
    mask = (timepoints >= cs) & (timepoints < cs + 0.75)
    change_signal[mask] = 1
```

iii. The 750 ms duration is justified from the task's flash cycle and the notes' interpretation of “right after” a change.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is directly binary: 1 inside an actual-change 750 ms window and 0 otherwise. There is no numeric thresholding stage.

ii.
```python
change_signal = np.zeros(n_tp, dtype=np.int64)
change_signal[mask] = 1
```

iii. The requested output is binary, so the code supplies labels `no_change` and `change`.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change windows are tested at the same `trial_ts` samples used for neural slicing.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
```

iii. The shared 30 Hz timestamps guarantee equal-length framewise arrays.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses the NWB running-speed data and timestamps.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. The notes identify this as the SDK's standard running speed in cm/s.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated to 30 Hz. Five global percentile edges are computed in a separate pass from every raw running sample in every selected experiment, then trial values are digitized.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
running_binned = digitize_to_bins(trial_data_out['running_speed'], running_bin_edges)
```

iii. The notes say interpolation provides common alignment and global percentile bins provide the requested five approximately balanced classes.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Global 0th, 20th, 40th, 60th, 80th, and 100th percentiles define five bins; repeated edges are nudged upward and NaNs map to class 0.

ii.
```python
edges = np.percentile(valid, np.linspace(0, 100, n_bins + 1))
if edges[i] <= edges[i-1]:
    edges[i] = edges[i-1] + 1e-10
binned = np.digitize(values, bin_edges[1:-1])
binned[np.isnan(values)] = 0
```

iii. This implements “five equal percentile bins” and guards constant signals and missing values.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is interpolated onto the exact full-session 30 Hz grid used for neural events, then sliced with the same trial indices.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'], raw_data['running_speed'])
running_trial = running_resampled[trial_time_indices]
```

iii. The notes cite synchronized acquisition clocks and a shared regular grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Despite the requested/output name, the AI uses `EyeTracking/pupil_tracking/area`, its timestamps, and `likely_blink`, not pupil width/diameter.

ii.
```python
pupil_area = pupil_tracking['area'][()]
pupil_ts = pupil_tracking['timestamps'][()]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()].astype(bool)
```

iii. The notes explicitly call pupil area a proxy for pupil diameter.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink samples are changed to NaN, internal NaNs are linearly filled (with endpoint propagation through `np.interp`), the result is resampled to 30 Hz, and five global percentile bins are computed from all non-blink raw pupil-area samples.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
```

iii. The notes say removing blink artifacts before interpolation prevents them from contaminating the decoded behavior.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It uses the same global five-percentile-bin helper as running speed; missing values, including an entirely absent pupil stream for an experiment, become category 0.

ii.
```python
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'], pupil_bin_edges)
binned[np.isnan(values)] = 0
```

iii. The stated goal is approximately balanced global classes while maintaining a valid categorical array when pupil data are absent.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Cleaned pupil area is interpolated to the same regular timestamps and sliced by the same trial indices as neural activity.

ii.
```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
pupil_trial = pupil_resampled[trial_time_indices]
```

iii. The notes say the eye clock is synchronized and common-grid resampling establishes samplewise alignment.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome comes from the trial table's boolean `hit`, `miss`, `false_alarm`, and `correct_reject` fields.

ii.
```python
if trials['hit'][trial_idx]: outcome = 0
elif trials['miss'][trial_idx]: outcome = 1
elif trials['false_alarm'][trial_idx]: outcome = 2
elif trials['correct_reject'][trial_idx]: outcome = 3
```

iii. The notes say these fields are mutually exclusive for valid Go/Catch trials and are the canonical outcomes.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The four booleans map in fixed order to codes 0–3; unrecognized trials are skipped. Although conceptually static, the code broadcasts the code across every trial time point.

ii.
```python
outcome_broadcast = np.full((1, n_tp),
                            trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)
```

iii. Broadcasting is justified by the common `(n_output, n_timepoints)` representation for mixed static and time-varying outputs.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. File/open errors and missing required structures return `None` and skip experiments; no-valid-ROI, missing-stimulus, too-few-trial, and unknown-outcome cases are also skipped. Missing pupil data becomes all NaN then bin 0. Blink/NaN pupil points are interpolated, missing-data percentile calculations ignore NaNs, repeated bin edges are adjusted, and data before/after interpolation support inherit endpoint values through `np.interp`.

ii.
```python
except Exception as e:
    print(f"  ERROR loading {experiment_id}: {e}")
    return None
...
if raw_data['pupil_area'] is None:
    pupil_trial = np.full(n_tp, np.nan)
...
binned[np.isnan(values)] = 0
```

iii. The notes describe blink interpolation and bin-0 fallback as pragmatic ways to avoid invalid decoder arrays; warnings expose skipped or absent data.

## 9-a. What are the most time-consuming steps of the code?

i. Repeated NWB I/O and full-session neural interpolation are the main expensive operations. The same files are opened in three passes, and per-neuron event interpolation processes large arrays.

ii.
```python
# Pass 1 ... with h5py.File(nwb_path, 'r')
# Pass 2 ... with h5py.File(nwb_path, 'r')
# Pass 3 ... raw_data = load_experiment_data(nwb_path, eid)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
```

iii. The notes explicitly identify sequential experiment processing and multiple NWB passes as inefficiencies, while measured runs separate load and processing times.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-feature interpolation loop and per-time-point image assignment loop could be vectorized; global/local image remapping and repeated list searches could also use array lookup tables. Experiment processing could be parallelized (the imported executor is unused), though that is parallelism rather than vectorization.

ii.
```python
for i in range(data.shape[1]):
    result[:, i] = np.interp(...)
for i in range(n_tp):
    image_idx[i] = name_to_idx.get(...)
global_image_names.index(name)
```

iii. The notes claim searchsorted is efficient but acknowledge sequential experiment processing. They do not discuss the remaining scalar assignment/remapping loops.

## 9-c. What processing does the code repeat multiple times?

i. Every NWB is opened once to gather image names, again to gather full-session running/pupil values, and again to load/process the experiment. Image vocabularies and local-to-global mappings are also rebuilt per experiment.

ii.
```python
print("--- Pass 1: Collecting global image names ---")
print("--- Pass 2: Collecting running speed and pupil data ... ---")
print("--- Pass 3: Processing experiments ---")
```

iii. The notes explicitly list “multiple passes over NWB files” as an inefficiency, accepted for straightforward global mappings and thresholds.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads several values that do not affect the saved data: `cell_specimen_ids`, trial `is_change` and initial/change names, stimulus stop times in the change routine, and `trial_outcomes`. It also computes unused locals (`ce`, `trial_outcome`) and imports multiprocessing utilities without using them. Most materially, it scans and bins behavior from all full-session samples even though only retained trial samples enter the decoder.

ii.
```python
cell_specimen_ids = cell_table['cell_specimen_id'][()]
change_stops = stim_data['stop_time'][change_mask]
for cs, ce in zip(change_starts, change_stops):  # ce unused
trial_outcomes = []
trial_outcome = np.array([trial_data_out['trial_outcome']], dtype=np.int64)
```

iii. The notes do not identify these discarded computations; they focus instead on multiple passes and sequential execution.
