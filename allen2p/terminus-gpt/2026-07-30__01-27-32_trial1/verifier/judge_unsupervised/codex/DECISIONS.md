# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script reads the experiment-level metadata table, builds one NWB path per `ophys_experiment_id`, and loads each file with `BehaviorOphysExperiment.from_nwb_path`. It makes one global pass in `collect_global_info()` to gather image labels and percentile-bin edges, then a second pass in `process_experiment()` to build the converted session/trial structures.

ii. ```python
def load_experiment_table():
    return pd.read_csv(EXPT_TABLE)

def get_nwb_path(exp_id: int) -> Path:
    return EXPT_DIR / f'behavior_ophys_experiment_{int(exp_id)}.nwb'

def load_experiment(exp_id: int) -> BehaviorOphysExperiment:
    return BehaviorOphysExperiment.from_nwb_path(str(get_nwb_path(exp_id)))

chosen_ids, image_names, image_to_idx, run_edges, pupil_edges = collect_global_info(exp_table)
for i, exp_id in enumerate(chosen_ids, 1):
    sess = process_experiment(exp_id, row_lookup[exp_id], run_edges, pupil_edges, image_to_idx, blank_idx)
```

iii. In Step 4/5 notes, the agent said the SDK is centered on `BehaviorOphysExperiment` objects and treated each experiment as the fundamental recording unit to load.

## 1-b. How are the data split into subjects?

i. Subjects are defined from the metadata-table `mouse_id` field for each experiment. After all converted sessions are collected, the script creates a sorted unique `subjects` list and maps each converted session to `subject_idx`.

ii. ```python
region = str(meta_row.get('targeted_structure', 'unknown'))
subject = str(meta_row.get('mouse_id', 'unknown'))

subjects = sorted({s['subject'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.asarray([subject_to_idx[s['subject']] for s in sessions], dtype=np.int64),
```

iii. Step 5 notes explicitly map `metadata.mouse_id` to `subjects` and `subject_idx`, with session order following converted experiment order.

## 1-c. How are the data split into sessions?

i. The script uses each `ophys_experiment_id` as one converted session, not each `ophys_session_id`. Every loaded experiment becomes one session entry in `neural`, `input`, `output`, and `brain_region_idx`.

ii. ```python
for _, row in exp_table.iterrows():
    exp_id = int(row['ophys_experiment_id'])
    ...
    chosen_ids.append(exp_id)

'neural': [s['neural'] for s in sessions],
'input': [s['input'] for s in sessions],
'output': [s['output'] for s in sessions],
```

iii. Step 4/5 notes justify this as: one experiment has one coherent neuron set and one brain-region assignment, while one ophys session may contain multiple experiments/planes.

## 1-d. How are the data split into trials?

i. Within each experiment, the script uses the AllenSDK `trials` table. Each trial spans `trial['start_time']` to `trial['stop_time']`, and the script slices all ophys frames whose `ophys_timestamps` fall inside that interval. Each retained trial becomes one element of the per-session lists.

ii. ```python
trials = exp.trials.copy()
...
for _, tr in trials.iterrows():
    start = float(tr['start_time'])
    stop = float(tr['stop_time'])
    m = (ophys_timestamps >= start) & (ophys_timestamps < stop)
    if m.sum() < 2:
        continue
    sample_times = ophys_timestamps[m]
    trial_neural = neural[:, m].astype(np.float32)
```

iii. Step 4/5 notes say the agent resolved the temporal unit as behaviorally defined trials, with time-varying outputs assigned inside each trial on the ophys time base.

## 1-e. How are trials filtered based on quality controls?

i. The script keeps only `go` or `catch` trials, excludes `aborted` and `auto_rewarded` trials, and drops any trial with fewer than 2 ophys samples.

ii. ```python
keep = trials['go'].fillna(False) | trials['catch'].fillna(False)
keep &= ~trials['aborted'].fillna(False)
keep &= ~trials['auto_rewarded'].fillna(False)
trials = trials[keep].copy()
...
if m.sum() < 2:
    continue
```

iii. Step 1/3/4 notes repeatedly cite the AllenSDK trial logic and the task instructions: include go and catch, exclude aborted and auto-rewarded, and keep at least two trials/timepoints for decoder compatibility.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The script prefers `exp.dff_traces` and falls back to `exp.events`. Once a signal table is selected, it extracts the first available array-like column in priority order `events`, `filtered_events`, `dff`.

ii. ```python
def choose_signal(exp):
    dff = exp.dff_traces.copy()
    if isinstance(dff, pd.DataFrame) and len(dff) > 0:
        return 'dff', dff
    events = exp.events.copy()
    return 'events', events

def extract_neural_matrix(signal_df, signal_kind):
    preferred = ['events', 'filtered_events', 'dff']
    ...
```

iii. The trajectory shows the agent originally used events, then switched to dF/F after sample verification/training because an events-based run produced an all-zero neural trial warning and worse decoder performance.

## 2-b. How is the `neural` data processed?

i. The selected per-cell trace arrays are converted to `float32`, truncated to the minimum shared length across neurons, stacked into a neuron-by-time matrix, then truncated again to match the available `ophys_timestamps`. Per-trial neural data are simple time slices of that session matrix; there is no extra normalization, rebinning, smoothing, or baseline subtraction in this script.

ii. ```python
arrays = [np.asarray(v, dtype=np.float32) for v in signal_df[col].values]
n_t = min(len(a) for a in arrays)
return np.stack([a[:n_t] for a in arrays], axis=0)

neural = extract_neural_matrix(signal_df, signal_kind)
ophys_timestamps = np.asarray(exp.ophys_timestamps, dtype=float)
n_t = min(neural.shape[1], len(ophys_timestamps))
neural = neural[:, :n_t]
ophys_timestamps = ophys_timestamps[:n_t]
...
trial_neural = neural[:, m].astype(np.float32)
```

iii. The notes justify this mainly as using SDK-provided traces already aligned to ophys timestamps; the main explicit later justification was the switch from events to dF/F because dF/F behaved better in sample validation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script does not apply its own explicit neuron QC mask. Instead, it relies on AllenSDK’s `BehaviorOphysExperiment.from_nwb_path()` defaults, which load `CellSpecimens` with `exclude_invalid_rois=True`, and then it keeps whatever cells remain in `exp.dff_traces`/`exp.events`.

ii. ```python
def load_experiment(exp_id: int) -> BehaviorOphysExperiment:
    return BehaviorOphysExperiment.from_nwb_path(str(get_nwb_path(exp_id)))

signal_kind, signal_df = choose_signal(exp)
neural = extract_neural_matrix(signal_df, signal_kind)
```

iii. Step 1/3 notes mention AllenSDK cell-specimen curation and `valid_roi`, but the agent never added extra QC beyond trusting the SDK-filtered experiment object.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The script aligns neural data to the experiment’s `ophys_timestamps` and then slices those timestamps within each trial’s start/stop interval. It does not realign to image change or trial start as a new zero; the common frame axis is simply the subset of ophys frames inside that trial.

ii. ```python
ophys_timestamps = np.asarray(exp.ophys_timestamps, dtype=float)
...
m = (ophys_timestamps >= start) & (ophys_timestamps < stop)
sample_times = ophys_timestamps[m]
trial_neural = neural[:, m].astype(np.float32)
...
'temporal_alignment_event': 'ophys timestamps within each behaviorally defined trial',
```

iii. Step 4/5 notes explicitly say “Align everything on ophys timestamps,” because the instructions required temporal alignment on the ophys time base.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted time bin is the native ophys frame interval, estimated as the median difference between consecutive `ophys_timestamps` and reported in milliseconds. No temporal rebinning is applied.

ii. ```python
'dt_ms': float(np.median(np.diff(ophys_timestamps)) * 1000.0),
...
time_bin_size = float(np.median([s['dt_ms'] for s in sessions])) if sessions else np.nan
...
'time_bin_size': time_bin_size,
```

iii. Step 3 notes record the native ophys timestamps as the intended neural time base, rather than the 750 ms image-interval abstraction used in some behavioral analyses.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `stimulus_presentations.image_name`.

ii. ```python
image_vals = [image_to_idx.get(str(x), blank_idx) for x in stim_non_omitted['image_name'].astype(str).values]
image_identity = interval_assign(
    sample_times,
    stim_non_omitted['start_time'].values,
    stim_non_omitted['stop_time'].values,
    image_vals,
    default=blank_idx,
)
```

iii. Step 5 notes map `stimulus_presentations.image_name` directly to the image-identity output.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The script sorts presentations by `start_time`, synthesizes a `stop_time` if needed, removes omitted flashes, adds a `blank` category for timepoints with no assigned image, maps image names to integer labels, and fills every ophys sample in the selected presentation intervals with that label. Crucially, `ensure_stim_stop_time()` is called with `use_full_interval=True`, so it extends each image’s interval to the next image onset rather than using only the flashed-image duration.

ii. ```python
def ensure_stim_stop_time(stim, use_full_interval=True):
    ...
    default_interval = float(np.nanmedian(diffs)) if len(diffs) else 0.75
    if use_full_interval:
        next_starts = np.r_[starts[1:], starts[-1] + default_interval]
        stim['stop_time'] = next_starts
        return stim

stim = ensure_stim_stop_time(stim_df)
if 'omitted' in stim.columns:
    stim_non_omitted = stim[~stim['omitted'].fillna(False)].copy()
...
image_identity = interval_assign(..., default=blank_idx)
```

iii. Step 7 notes say the agent changed this after seeing “too many unknown image_identity bins,” and “fixed” it by assigning image labels over full image-presentation intervals rather than only flash durations.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. For each trial, image identity is assigned on the same `sample_times` array used for the neural slice. The script uses interval membership on the ophys timestamps within that trial, so output timepoints and neural timepoints are one-to-one.

ii. ```python
sample_times = ophys_timestamps[m]
...
image_identity = interval_assign(sample_times, ..., default=blank_idx)
...
return np.stack([image_identity, image_change, run_bins, pupil_bins, trial_outcome], axis=0)
```

iii. The Step 4/5 notes justify all time-varying outputs being resampled or assigned onto the ophys timestamp grid within each trial.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations.is_change`, using the presentation `start_time` of each marked change.

ii. ```python
image_change = np.zeros(sample_times.shape, dtype=np.int64)
if 'is_change' in stim_non_omitted.columns:
    for ct in stim_non_omitted.loc[stim_non_omitted['is_change'].fillna(False), 'start_time'].values:
        idx = np.searchsorted(sample_times, ct, side='left')
        if idx < len(image_change):
            image_change[idx] = 1
```

iii. Step 5 notes explicitly map `stimulus_presentations.is_change` and image-identity transitions to the binary image-change output.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The script initializes a zero vector, finds all stimulus presentations with `is_change=True`, and sets the first ophys sample at or after each change `start_time` to 1.

ii. ```python
image_change = np.zeros(sample_times.shape, dtype=np.int64)
...
idx = np.searchsorted(sample_times, ct, side='left')
if idx < len(image_change):
    image_change[idx] = 1
```

iii. The justification in Step 5 is simply to make change a time-varying binary series aligned to ophys samples.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is not thresholded from a continuous value; it is directly represented as a binary categorical series with classes `0=no_change` and `1=change`.

ii. ```python
'output_values': [
    image_names,
    ['no_change', 'change'],
    ...
]
```

iii. This follows the decoder specification from the instructions, which asked for a binary variable that is 1 right after a change and 0 otherwise.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change labels are placed directly on the per-trial `sample_times` ophys grid, using `searchsorted` to find the first neural frame at or after a change time.

ii. ```python
sample_times = ophys_timestamps[m]
...
idx = np.searchsorted(sample_times, ct, side='left')
if idx < len(image_change):
    image_change[idx] = 1
```

iii. Step 4/5 notes say all streams are aligned to ophys timestamps within each behaviorally defined trial.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `exp.running_speed`, specifically its timestamp and speed columns.

ii. ```python
def get_running_df(exp):
    run_df = exp.running_speed.copy()
    ...
    return run_df.rename(columns={tcol: 'timestamps', scol: 'speed'})[['timestamps', 'speed']]
```

iii. Step 5 notes map `BehaviorOphysExperiment.running_speed` directly to the running-speed output and rely on the SDK’s processed running-speed stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The script renames columns as needed, takes the nearest running-speed sample for each ophys timestamp, and then discretizes those values into percentile bins. It does not compute speed from wheel signals itself.

ii. ```python
run_vals = nearest_assign(sample_times, run_df['timestamps'].values, run_df['speed'].values)
run_bins = digitize(run_vals, run_edges)
```

iii. The notes justify using the SDK-provided running-speed series and only adding the discretization required by the decoder spec.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The script computes 4 percentile cut points from all collected running-speed samples and uses `np.digitize` to form 5 bins labeled `bin_0` through `bin_4`.

ii. ```python
run_edges = make_percentile_bins(np.concatenate(run_vals) if run_vals else np.array([0.0]))
...
out = np.digitize(vals, edges, right=False).astype(np.int64)
...
[f'bin_{i}' for i in range(5)],
```

iii. Step 5 notes explicitly say running speed should be discretized into five equal-percentile bins, with global cut points computed from valid retained samples.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by nearest-neighbor assignment from running timestamps onto the per-trial ophys `sample_times`.

ii. ```python
def nearest_assign(sample_times, source_times, source_values):
    ...
    return source_values[idx]

run_vals = nearest_assign(sample_times, run_df['timestamps'].values, run_df['speed'].values)
```

iii. Step 4/5 notes justify putting all time-varying outputs onto the ophys timestamp grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The script chooses the first available “pupil” column from the eye-tracking table using the priority list `pupil_diameter`, `pupil_area`, `pupil_width`, `pupil_radius`. On this dataset, that means it uses `pupil_area`.

ii. ```python
def pick_pupil_column(eye_tracking):
    candidates = ['pupil_diameter', 'pupil_area', 'pupil_width', 'pupil_radius']
    for c in candidates:
        if c in eye_tracking.columns:
            return c
```

iii. Step 5 notes only say the agent needed to “identify exact pupil-diameter column name from experiment object.” The implemented fallback to area/width/radius is the code’s actual decision.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The selected pupil column is cleaned by replacing infinities with NaN and dropping missing rows, then nearest-neighbor assigned to ophys timestamps and digitized into percentile bins. If no usable pupil column or no valid rows exist, the script fills the whole trial with zeros.

ii. ```python
pupil_col = pick_pupil_column(eye_df)
if pupil_col is None:
    pupil_bins = np.zeros(sample_times.shape, dtype=np.int64)
else:
    eye_valid = eye_df[['timestamps', pupil_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(eye_valid) == 0:
        pupil_bins = np.zeros(sample_times.shape, dtype=np.int64)
    else:
        pupil_vals = nearest_assign(sample_times, eye_valid['timestamps'].values, eye_valid[pupil_col].values)
        pupil_bins = digitize(pupil_vals, pupil_edges)
```

iii. The notes justify only the general plan: align pupil to ophys timestamps and discretize to five percentile bins. There is no explicit justification in the notes for substituting `pupil_area` for diameter.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The script computes 4 percentile cut points from all collected pupil values and uses `np.digitize` to create 5 global bins labeled `bin_0` through `bin_4`.

ii. ```python
pupil_edges = make_percentile_bins(np.concatenate(pupil_vals) if pupil_vals else np.array([0.0]))
...
pupil_bins = digitize(pupil_vals, pupil_edges)
...
[f'bin_{i}' for i in range(5)],
```

iii. Step 5 notes explicitly say pupil should be discretized into five equal-percentile bins.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. The chosen pupil signal is aligned by nearest-neighbor assignment onto the per-trial ophys `sample_times`.

ii. ```python
pupil_vals = nearest_assign(sample_times, eye_valid['timestamps'].values, eye_valid[pupil_col].values)
pupil_bins = digitize(pupil_vals, pupil_edges)
```

iii. Step 4/5 notes justify using ophys timestamps as the common alignment grid for all time-varying streams.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the AllenSDK `trials` table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. ```python
if bool(trial_row.get('hit', False)):
    outcome = 0
elif bool(trial_row.get('miss', False)):
    outcome = 1
elif bool(trial_row.get('false_alarm', False)):
    outcome = 2
elif bool(trial_row.get('correct_reject', False)):
    outcome = 3
```

iii. Step 1/5 notes identify those four outcome fields as the intended trial-outcome labels after excluding aborted and auto-rewarded trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The script converts the boolean outcome flags to integer class IDs in a fixed order and then repeats the resulting label across every ophys sample in the trial, making the output time-varying but constant within a trial.

ii. ```python
trial_outcome = np.full(sample_times.shape, outcome, dtype=np.int64)
...
return np.stack([image_identity, image_change, run_bins, pupil_bins, trial_outcome], axis=0)
```

iii. Step 5 notes describe trial outcome as a static per-trial label derived from the SDK trial table; the implementation makes it decoder-compatible by repeating it across timepoints.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles missingness and inconsistencies mostly by silent fallback: it skips missing NWB files, truncates neural traces and ophys timestamps to the shortest common length, infers stimulus `stop_time` if absent, drops NaN/inf pupil samples before alignment, fills missing pupil streams with bin 0, assigns `blank` for unassigned image identity bins, and defaults any otherwise-unclassified outcome to class 0.

ii. ```python
if not nwb_path.exists():
    continue
...
n_t = min(neural.shape[1], len(ophys_timestamps))
neural = neural[:, :n_t]
ophys_timestamps = ophys_timestamps[:n_t]
...
if vals.size == 0:
    return np.array([0, 1, 2, 3], dtype=float)
...
out[~np.isfinite(vals)] = 0
...
image_identity = interval_assign(..., default=blank_idx)
...
if pupil_col is None:
    pupil_bins = np.zeros(sample_times.shape, dtype=np.int64)
...
else:
    outcome = 0
```

iii. The notes mention these as pragmatic fixes for sample verification issues, especially adding an explicit `blank` class and extending image intervals after seeing invalid or “unknown” image bins.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive operations are repeated NWB loading through `BehaviorOphysExperiment.from_nwb_path` and the double pass over experiments: `collect_global_info()` loads every experiment once for labels/bin edges, and `process_experiment()` loads them again for actual conversion.

ii. ```python
def load_experiment(exp_id: int) -> BehaviorOphysExperiment:
    return BehaviorOphysExperiment.from_nwb_path(str(get_nwb_path(exp_id)))

for _, row in exp_table.iterrows():
    ...
    exp = load_experiment(exp_id)
    ...

for i, exp_id in enumerate(chosen_ids, 1):
    sess = process_experiment(exp_id, row_lookup[exp_id], ...)
```

iii. The Step 6/33/37 notes explicitly call out repeated experiment loading during global percentile/image collection as the main runtime bottleneck.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The main non-vectorized loops are: iterating experiments row-by-row, iterating trials with `iterrows()`, iterating stimulus intervals in `interval_assign()`, iterating change times in `build_trial_output()`, and building neural arrays via a Python list comprehension over cells.

ii. ```python
for _, row in exp_table.iterrows():
    ...
for _, tr in trials.iterrows():
    ...
for s, e, v in zip(np.asarray(starts, float), np.asarray(stops, float), np.asarray(values, np.int64)):
    m = (sample_times >= s) & (sample_times < e)
    out[m] = v
...
for ct in stim_non_omitted.loc[stim_non_omitted['is_change'].fillna(False), 'start_time'].values:
    ...
arrays = [np.asarray(v, dtype=np.float32) for v in signal_df[col].values]
```

iii. Step 6 notes specifically flag interval assignment as a loop that could be improved; the rest follow directly from the code structure.

## 9-c. What processing does the code repeat multiple times?

i. It recomputes stimulus stop times for both whole experiments and per-trial subsets, re-picks the pupil column for every trial, reloads experiments twice, and re-slices/copies stimulus tables per trial.

ii. ```python
stim = ensure_stim_stop_time(exp.stimulus_presentations.copy())
...
trial_stim = stim[(stim['start_time'] < stop) & (stim['stop_time'] > start)].copy()
trial_output = build_trial_output(tr, sample_times, trial_stim, run_df, eye_df, ...)

def build_trial_output(...):
    stim = ensure_stim_stop_time(stim_df)
    ...
    pupil_col = pick_pupil_column(eye_df)
```

iii. The notes identify the repeated experiment loads and per-trial stimulus interval work as key inefficiencies.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script builds empty `input` arrays for every trial even though decoder inputs are defined as none, computes `signal_kind` and `exp_id` for each session but does not store them in the final output, creates a placeholder `brain_region_idx` zero vector only to overwrite it later with a full vector of the session’s region index, and preserves a `--show-processing` flag without implementing any plotting logic.

ii. ```python
trial_input = np.zeros((0, len(sample_times)), dtype=np.float32)
sess_input.append(trial_input)
...
return {
    'exp_id': int(exp_id),
    ...
    'signal_kind': signal_kind,
    ...
    'brain_region_idx': brain_region_idx,
}
...
'brain_region_idx': [np.full((len(s['brain_region_idx']),), region_to_idx[s['region']], dtype=np.int64) for s in sessions],
```

iii. Step 6 notes already acknowledge placeholder or deferred pieces such as `--show-processing`; the rest are visible from the script’s final data assembly.
