# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by globbing all NWB files from the `behavior_ophys_experiments` directory. Each NWB file is loaded individually using `pynwb.NWBHDF5IO` and then wrapped via `BehaviorOphysExperiment.from_nwb()`. There is no project code filtering; all NWB files in the directory are processed. A two-pass approach is used: first `collect_global_info()` loads a subset of files to estimate global bin edges and image names, then the main loop loads all files to build sessions.

ii.
```python
def list_session_files(sample=False):
    files = sorted(NWB_DIR.glob('*.nwb'))
    return files[:2] if sample else files

def load_experiment(nwb_path):
    io = NWBHDF5IO(str(nwb_path), 'r', load_namespaces=True)
    nwbfile = io.read()
    exp = BehaviorOphysExperiment.from_nwb(nwbfile)
    exp._nwb_io = io
    return exp
```

iii. The AI chose to load NWB files directly via `pynwb` rather than using the `VisualBehaviorOphysProjectCache` S3 cache API. This was justified by the fact that the data was already available locally as NWB files and the cache API might try to fetch from S3. The AI noted in CONVERSION_NOTES.md that the "local data are a subset of the full release."

## 1-b. How are the data split into subjects?

i. Subjects are identified from each session's metadata `mouse_id` field. Unique mouse IDs are collected across all loaded sessions and sorted to create the subjects list.

ii.
```python
subjects = sorted({s['mouse_id'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The `mouse_id` from experiment metadata is the standard identifier for each animal. Sorting ensures deterministic ordering.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as a separate session. There is no grouping of experiments by `ophys_session_id`. Since each NWB file corresponds to one imaging plane in one recording session, and the VisualBehavior project uses single-plane imaging, each file effectively represents one session.

ii.
```python
files = sorted(NWB_DIR.glob('*.nwb'))
for i, p in enumerate(files, 1):
    exp = load_experiment(p)
    sess = build_session(exp, image_to_idx, run_edges, pupil_edges, outcome_to_idx)
    sessions.append(sess)
```

iii. The AI treats each NWB file as a session. For the VisualBehavior project (single-plane), this is functionally equivalent to grouping by `ophys_session_id`, since each session has exactly one imaging plane and thus one NWB file.

## 1-d. How are the data split into trials?

i. Trials are defined using the experiment's `trials` table, filtered to include only Go and Catch trials (excluding aborted and auto-rewarded trials). Trial windows span from `start_time` to `stop_time`. Ophys frames are selected where `ts >= start` and `ts < stop`.

ii.
```python
def valid_trials_table(trials):
    df = trials.copy()
    for col in ['aborted', 'auto_rewarded', 'auto_rewarded_trial', 'is_auto_rewarded']:
        if col in df.columns:
            df = df[~df[col].fillna(False)]
    keep = None
    if 'go' in df.columns and 'catch' in df.columns:
        keep = df['go'].fillna(False) | df['catch'].fillna(False)
    elif 'trial_type' in df.columns:
        keep = df['trial_type'].astype(str).str.lower().isin(['go', 'catch'])
    if keep is not None:
        df = df[keep]
    return df.reset_index(drop=True)
```
```python
idx = np.flatnonzero((ts >= start) & (ts < stop))
if idx.size < 2:
    continue
```

iii. The AI filters using both aborted/auto_rewarded exclusion and explicit go/catch inclusion. This is a belt-and-suspenders approach that ensures only valid trial types are included. The minimum trial size of 2 frames prevents degenerate trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) excluding aborted trials, (2) excluding auto-rewarded trials (checking multiple column name variants), (3) requiring the trial to be go or catch type, (4) requiring at least 2 ophys frames in the trial window. There is no explicit filter on `change_time` validity.

ii.
```python
for col in ['aborted', 'auto_rewarded', 'auto_rewarded_trial', 'is_auto_rewarded']:
    if col in df.columns:
        df = df[~df[col].fillna(False)]
if 'go' in df.columns and 'catch' in df.columns:
    keep = df['go'].fillna(False) | df['catch'].fillna(False)
...
if idx.size < 2:
    continue
```

iii. The AI's filtering approach is robust to different column naming conventions in the trials table. The explicit go/catch filter is redundant with aborted/auto-rewarded exclusion but adds an extra safety layer.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dff_traces` (dF/F calcium fluorescence traces). The code accesses `exp.dff_traces` and extracts the `dff` column.

ii.
```python
def get_cell_matrix(exp):
    dff = exp.dff_traces.copy()
    if isinstance(dff, pd.DataFrame) and 'dff' in dff.columns:
        cell_ids = np.asarray(dff.index)
        traces = [np.asarray(v, dtype=np.float32) for v in dff['dff'].values]
        mat = np.stack(traces, axis=0)
        return mat, cell_ids, 'dff'
```

iii. The CONVERSION_NOTES.md notes that the paper used "detected calcium events" for neural analyses, but the AI chose to use dF/F traces because events were not readily accessible from the local NWB files. The notes acknowledge this as a fallback decision.

## 2-b. How is the `neural` data processed?

i. No additional processing is applied beyond reading the dF/F traces and stacking them into a matrix. Each neuron's trace is extracted as a float32 array and stacked into shape (n_neurons, T). Per-trial slicing extracts the relevant time window.

ii.
```python
traces = [np.asarray(v, dtype=np.float32) for v in dff['dff'].values]
mat = np.stack(traces, axis=0)
...
trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
```

iii. The dF/F traces are pre-computed by the Allen SDK pipeline, so no additional normalization is needed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No quality filtering is applied to neurons. All neurons present in the `dff_traces` table are included.

ii. N/A (no filtering code)

iii. The AI relies on the Allen SDK's own quality control during cell segmentation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial window defined by `start_time` and `stop_time`. Ophys frames falling within `[start_time, stop_time)` are selected using boolean indexing.

ii.
```python
ts = np.asarray(exp.ophys_timestamps, dtype=float)
...
idx = np.flatnonzero((ts >= start) & (ts < stop))
trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
```

iii. The ophys timestamps serve as the master clock. Using `>=` for start and `<` for stop provides consistent frame selection.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. Data is kept at the native ophys frame rate (~11 Hz). The time bin size is computed from the median inter-frame interval of the first session's ophys timestamps.

ii.
```python
dt_ms = float(np.median(np.diff(load_experiment(files[0]).ophys_timestamps)) * 1000.0)
```

iii. The native ophys frame rate provides sufficient temporal resolution for the decoder task. No resampling is needed since all data streams are aligned to the ophys timebase.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, specifically the `image_name` column. The AI iterates over stimulus presentations that overlap with each trial window and assigns the image name to the corresponding ophys frames. Periods between stimuli are labeled as 'blank'.

ii.
```python
stim = exp.stimulus_presentations.copy().sort_values('start_time')
stim_start = np.asarray(stim['start_time'], dtype=float)
stim_end = stim_start + 0.75
...
image_series = np.full(len(trial_ts), image_to_idx['blank'], dtype=np.int64)
...
for _, srow in stim_trial.iterrows():
    name = str(srow['image_name']) if 'image_name' in srow.index else 'blank'
    image_series[smask] = image_to_idx.get(name, image_to_idx['blank'])
```

iii. The AI uses the stimulus_presentations table to get precise timing of when each image is displayed, including grey/blank periods and omissions. This provides a more fine-grained representation than just using trial-level image names.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names from stimulus_presentations are mapped to integer codes via a global sorted mapping. The mapping includes 'blank' and 'omitted' as categories in addition to the natural image names. Each 750ms stimulus window is labeled with its image name, and gaps between stimuli default to 'blank'.

ii.
```python
image_names = set(['blank', 'omitted'])
...
names = stim['image_name'].dropna().astype(str).unique().tolist()
image_names.update(names)
image_values = sorted(image_names)
image_to_idx = {name: i for i, name in enumerate(image_values)}
```

iii. Including 'blank' and 'omitted' as categories captures the full stimulus sequence. The global mapping ensures consistent encoding across sessions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. For each trial, stimulus presentations overlapping with the trial window are identified. Each stimulus's time interval is intersected with the trial timestamps, and the corresponding ophys frames are labeled with the image name. Frames not covered by any stimulus are labeled 'blank'.

ii.
```python
stim_trial = stim[(stim['start_time'] < stop) & (stim['_end_time'] > start)]
for _, srow in stim_trial.iterrows():
    s0 = max(start, float(srow['start_time']))
    s1 = min(stop, float(srow['_end_time']))
    smask = (trial_ts >= s0) & (trial_ts < s1)
    image_series[smask] = image_to_idx.get(name, image_to_idx['blank'])
```

iii. The alignment uses the same ophys timestamps as the neural data, ensuring frame-level synchronization.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from transitions in the `image_name` across consecutive stimulus presentations within each trial. When the current stimulus has a different name than the previous one, the first frame of the new stimulus is marked as a change.

ii.
```python
change_series = np.zeros(len(trial_ts), dtype=np.int64)
prev_name = None
for _, srow in stim_trial.iterrows():
    name = str(srow['image_name']) if 'image_name' in srow.index else 'blank'
    if prev_name is not None and name != prev_name:
        hit = np.flatnonzero(smask)
        if hit.size:
            change_series[hit[0]] = 1
    prev_name = name
```

iii. The AI detects image changes by comparing consecutive stimulus presentations, marking transitions as change events.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The processing is a simple comparison of consecutive image names. When a transition is detected, only the first frame of the new stimulus is marked as 1, rather than a time window. The change detection applies to ALL image transitions, not just go-trial changes.

ii. See 4-a code snippet.

iii. This approach captures all stimulus transitions, not just the behavioral "change" event in go trials.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1), so no thresholding is applied. Category values are ['no_change', 'change'].

ii.
```python
'output_values': [..., ['no_change', 'change'], ...]
```

iii. Binary encoding directly represents the presence or absence of a change.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is computed per ophys frame using the same trial timestamps as the neural data. The change is marked at the first frame of the new stimulus presentation.

ii. See 4-a code snippet. The `smask` and `trial_ts` arrays use the same ophys timestamps as the neural data.

iii. Same frame-level alignment as image identity and neural data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `exp.running_speed`, which provides speed and timestamps from the running wheel encoder.

ii.
```python
def get_running_series(run_df):
    cols = list(run_df.columns)
    tcol = 'timestamps' if 'timestamps' in cols else cols[0]
    vcol = 'speed' if 'speed' in cols else cols[1]
    return np.asarray(run_df[tcol], dtype=float), np.asarray(run_df[vcol], dtype=float)
```

iii. The `running_speed` attribute is the SDK's standard interface for locomotion data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is interpolated to ophys timestamps using `np.interp`, then discretized into 5 bins using pre-computed percentile-based bin edges. Bin edges are estimated from the first 8 NWB files (not all files) as an efficiency optimization.

ii.
```python
run_interp = np.interp(trial_ts, run_t, run_v).astype(np.float32)
run_bins = np.digitize(run_interp, run_edges[1:-1], right=False).astype(np.int64)
...
# In collect_global_info:
probe_files = files[:min(8, len(files))]
...
run_edges = np.quantile(run_all, [0, .2, .4, .6, .8, 1])
```

iii. The AI used `np.interp` for linear interpolation and `np.quantile` for percentile-based binning. Estimating bin edges from a subset of files was an efficiency optimization to avoid loading all files twice.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal-percentile bins. The bin edges are computed using quantiles [0, 0.2, 0.4, 0.6, 0.8, 1.0] from the first 8 files. `np.digitize` maps values to bin indices.

ii.
```python
run_edges = np.quantile(run_all, [0, .2, .4, .6, .8, 1])
run_bins = np.digitize(run_interp, run_edges[1:-1], right=False).astype(np.int64)
```

iii. Percentile-based bins aim to produce equal class counts, suitable for balanced decoding.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated from its native timestamps to the trial's ophys timestamps using `np.interp`. The same trial timestamps are used for both neural and running speed data.

ii.
```python
run_interp = np.interp(trial_ts, run_t, run_v).astype(np.float32)
```

iii. By interpolating to ophys timestamps, alignment with neural data is guaranteed.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `exp.eye_tracking`. The code searches for appropriate column names in priority order: `pupil_diameter`, `pupil_area`, `pupil_width`, `pupil_height`, `pupil_radius`, `pupil_size`.

ii.
```python
def pick_pupil_series(eye_tracking):
    candidate_cols = ['pupil_diameter', 'pupil_area', 'pupil_width', 'pupil_height', 'pupil_radius', 'pupil_size']
    value_col = next((c for c in candidate_cols if c in cols), None)
    ...
    return np.asarray(eye_tracking[time_col], dtype=float), np.asarray(eye_tracking[value_col], dtype=float), value_col
```

iii. The flexible column selection handles potential variations in the eye tracking data format across sessions.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil values are interpolated to ophys timestamps using `np.interp`, then discretized into 5 percentile-based bins. Only finite values are used for interpolation. Bin edges are estimated from the first 8 files. Non-finite pupil timestamps/values are filtered before interpolation.

ii.
```python
if pupil_t is not None and pupil_v is not None:
    valid = np.isfinite(pupil_t) & np.isfinite(pupil_v)
    if valid.sum() >= 2:
        pupil_interp = np.interp(trial_ts, pupil_t[valid], pupil_v[valid]).astype(np.float32)
        pupil_bins = np.digitize(pupil_interp, pupil_edges[1:-1], right=False).astype(np.int64)
    else:
        pupil_bins = np.zeros(len(trial_ts), dtype=np.int64)
else:
    pupil_bins = np.zeros(len(trial_ts), dtype=np.int64)
```

iii. Using `np.isfinite` filters out NaN and inf values but does NOT explicitly filter blink frames using the `likely_blink` column. Missing pupil data defaults to bin 0.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 equal-percentile bins using `np.quantile` and `np.digitize`. Bin edges estimated from first 8 files.

ii.
```python
pupil_edges = np.quantile(pupil_all, [0, .2, .4, .6, .8, 1])
pupil_bins = np.digitize(pupil_interp, pupil_edges[1:-1], right=False).astype(np.int64)
```

iii. Percentile-based binning for balanced class distributions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the trial's ophys timestamps using `np.interp`, same as running speed. The same trial timestamps are used for neural and pupil data.

ii.
```python
pupil_interp = np.interp(trial_ts, pupil_t[valid], pupil_v[valid]).astype(np.float32)
```

iii. Same frame-level alignment as running speed and neural data.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table. If none match, the outcome is 'other'.

ii.
```python
def infer_trial_outcome(row):
    for key in ['hit', 'miss', 'false_alarm', 'correct_reject']:
        if key in row.index and bool(row[key]):
            return key
    return 'other'
```

iii. These four columns are the SDK's canonical trial outcome labels. The 'other' fallback handles edge cases.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes via a fixed mapping that includes 5 categories: ['hit', 'miss', 'false_alarm', 'correct_reject', 'other']. The integer code is constant across all time bins within a trial (static per-trial output replicated across frames).

ii.
```python
TRIAL_OUTCOME_VALUES = ['hit', 'miss', 'false_alarm', 'correct_reject', 'other']
outcome_to_idx = {name: i for i, name in enumerate(TRIAL_OUTCOME_VALUES)}
...
outcome = infer_trial_outcome(tr)
outcome_idx = outcome_to_idx.get(outcome, outcome_to_idx['other'])
outcome_series = np.full(len(trial_ts), outcome_idx, dtype=np.int64)
```

iii. Including 'other' as a fifth category covers any trials that don't match the four standard outcomes.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: If eye tracking data is unavailable or has fewer than 2 valid points, pupil bins default to 0 for the entire trial.
- **Non-finite values**: `np.isfinite` is used to filter pupil timestamps and values before interpolation.
- **Short trials**: Trials with fewer than 2 ophys frames are skipped.
- **Flexible column names**: The code checks multiple potential column names for pupil data, running speed, and trial timing.
- **Missing image names**: Defaults to 'blank' if `image_name` column is missing from a stimulus row.

ii.
```python
if pupil_t is not None and pupil_v is not None:
    valid = np.isfinite(pupil_t) & np.isfinite(pupil_v)
    if valid.sum() >= 2:
        ...
    else:
        pupil_bins = np.zeros(len(trial_ts), dtype=np.int64)
else:
    pupil_bins = np.zeros(len(trial_ts), dtype=np.int64)
...
if idx.size < 2:
    continue
```

iii. The approach is defensive, with fallbacks for various missing data scenarios, ensuring the pipeline doesn't crash on edge cases.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each NWB file via `load_experiment()`, which involves parsing the NWB file with pynwb and constructing the `BehaviorOphysExperiment` object. The code also loads files twice: once in `collect_global_info()` for the first 8 files, and again in the main loop for all files.

ii.
```python
exp = load_experiment(p)  # Called for each NWB file
```

iii. NWB file I/O dominates runtime. Each file contains the full neural and behavioral data for one experiment.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `build_session` iterates over each trial sequentially, performing interpolation and stimulus matching per trial. The stimulus matching loop (iterating over `stim_trial` rows) could potentially be vectorized using interval-based operations.

ii.
```python
for _, tr in trials.iterrows():
    ...
    for _, srow in stim_trial.iterrows():
        ...
```

iii. The nested loops (trials x stimulus presentations) are the main candidates for vectorization, though the variable-length nature of trials makes full vectorization complex.

## 9-c. What processing does the code repeat multiple times?

i. The code loads NWB files twice: `collect_global_info()` loads the first 8 files to estimate bin edges and image names, then the main loop loads all 284 files again (including those same 8). Additionally, the first file is loaded a third time at the end to compute `dt_ms`.

ii.
```python
# First pass (subset):
probe_files = files[:min(8, len(files))]
for p in probe_files:
    exp = load_experiment(p)
    ...
# Second pass (all):
for i, p in enumerate(files, 1):
    exp = load_experiment(p)
    ...
# Third load (single file):
dt_ms = float(np.median(np.diff(load_experiment(files[0]).ophys_timestamps)) * 1000.0)
```

iii. This redundant loading is a significant efficiency issue, especially since NWB file loading is the bottleneck.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code collects `cell_ids`, `neural_source`, `image_names_seen`, and `trial_outcomes_seen` per session, which are used only for logging/printing and not included in the final pickle output. The `stimulus_presentations` table is fully loaded and processed for image identity, but the detailed stimulus timing information is not retained.

ii.
```python
return {
    ...
    'cell_ids': cell_ids,
    'neural_source': neural_source,
    'image_names_seen': sorted(set(image_names_seen)),
    'trial_outcomes_seen': sorted(set(trial_outcomes_seen)),
    ...
}
```

iii. These extra fields are useful for debugging but add processing overhead without contributing to the final output.
