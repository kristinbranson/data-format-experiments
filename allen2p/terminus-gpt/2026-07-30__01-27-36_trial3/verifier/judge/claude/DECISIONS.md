# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by discovering NWB files in the local `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/` directory and loading each NWB file individually using `pynwb` and `BehaviorOphysExperiment.from_nwb()`. Each NWB file corresponds to one ophys experiment (one imaging plane). The AI does not use the Allen SDK's `VisualBehaviorOphysProjectCache` to fetch data; instead it reads files directly from disk.

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

iii. The AI chose to load NWB files directly because the data was available locally on disk. The CONVERSION_NOTES state: "Raw dataset is in `data/visual-behavior-ophys-1.1.0/`" and "conversion must operate on the locally available NWB files only."

## 1-b. How are the data split into subjects?

i. Subjects are identified by extracting `mouse_id` from each experiment's metadata and collecting unique values. Subject assignment is done after all sessions are processed.

ii.
```python
meta = exp.metadata if isinstance(exp.metadata, dict) else {}
mouse_id = str(meta.get('mouse_id', 'unknown'))
...
subjects = sorted({s['mouse_id'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. Each experiment's metadata contains the mouse_id field, which uniquely identifies the animal.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as a separate session. There is no grouping of experiments by `ophys_session_id` — each imaging plane (experiment) becomes its own session in the output.

ii.
```python
files = sorted(NWB_DIR.glob('*.nwb'))
...
for i, p in enumerate(files, 1):
    exp = load_experiment(p)
    sess = build_session(exp, image_to_idx, run_edges, pupil_edges, outcome_to_idx)
    sessions.append(sess)
```

iii. The AI treated each NWB file as a session. The CONVERSION_NOTES state: "one file per available ophys experiment."

## 1-d. How are the data split into trials?

i. Trials are defined using the experiment's built-in `trials` table. The AI filters to keep only Go and Catch trials, excluding aborted and auto-rewarded trials. Trial boundaries are defined by `start_time` and `stop_time` from the trials table. Ophys frames within `[start_time, stop_time)` are extracted for each trial.

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

Trial extraction in `build_session`:
```python
for _, tr in trials.iterrows():
    start = float(tr['start_time'])
    stop = float(tr['stop_time']) if 'stop_time' in tr.index else float(tr['end_time'])
    idx = np.flatnonzero((ts >= start) & (ts < stop))
    if idx.size < 2:
        continue
```

iii. The AI followed the instruction to include Go and Catch trials and exclude Aborted and Auto-rewarded trials. The filtering is robust, checking multiple possible column names for auto-rewarded status.

## 1-e. How are trials filtered based on quality controls?

i. Aborted and auto-rewarded trials are excluded. Additionally, trials with fewer than 2 ophys frames are skipped. The AI also explicitly filters to only include trials marked as `go` or `catch`. No filtering based on `change_time` validity.

ii.
```python
for col in ['aborted', 'auto_rewarded', 'auto_rewarded_trial', 'is_auto_rewarded']:
    if col in df.columns:
        df = df[~df[col].fillna(False)]
...
if idx.size < 2:
    continue
```

iii. The filtering follows the task instructions to include only Go and Catch trials. The minimum 2-frame threshold prevents degenerate trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dff_traces` (dF/F calcium fluorescence traces) from each experiment.

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

iii. The CONVERSION_NOTES mention "Prefer detected calcium events to match paper; verify exact access pattern in SDK/local NWB. If unavailable, use dF/F traces with clear documentation." The AI ultimately used dF/F traces after encountering difficulties with events.

## 2-b. How is the `neural` data processed?

i. No additional processing is applied to the neural data beyond extracting the dF/F traces and slicing them into trial windows. Each neuron's trace is cast to float32.

ii.
```python
mat = np.stack(traces, axis=0)
...
trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
```

iii. The dF/F traces are pre-processed by the Allen SDK pipeline (motion correction, neuropil subtraction, dF/F normalization).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied to the neural data. All neurons in the SDK's `dff_traces` are included.

ii. N/A — no filtering code present.

iii. The Allen SDK pipeline already applies quality control during cell segmentation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the ophys timestamps within the trial window (`start_time` to `stop_time`). Frames where `ts >= start_time` and `ts < stop_time` are selected.

ii.
```python
ts = np.asarray(exp.ophys_timestamps, dtype=float)
...
idx = np.flatnonzero((ts >= start) & (ts < stop))
trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
```

iii. The instructions say to "Temporally align based on ophys timestamp." The AI aligns all data streams to the ophys timebase within trial boundaries.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the native ophys frame rate (~11 Hz, approximately 93ms bins). No temporal rebinning is applied. The time bin size is computed from the median inter-frame interval.

ii.
```python
dt_ms = float(np.median(np.diff(load_experiment(files[0]).ophys_timestamps)) * 1000.0)
```

iii. No rebinning was needed since all data streams are aligned to the same ophys timebase.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, specifically the `image_name` column and the `start_time` of each stimulus presentation. The AI uses 750ms presentation intervals to determine which image is on screen at each ophys timepoint.

ii.
```python
stim = exp.stimulus_presentations.copy().sort_values('start_time')
stim_start = np.asarray(stim['start_time'], dtype=float)
stim_end = stim_start + 0.75
...
for _, srow in stim_trial.iterrows():
    name = str(srow['image_name']) if 'image_name' in srow.index else 'blank'
    image_series[smask] = image_to_idx.get(name, image_to_idx['blank'])
```

iii. The AI used the stimulus_presentations table rather than the trials table's `initial_image_name`/`change_image_name` fields, to get finer-grained timing of when each image is on screen.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global mapping. The mapping includes 'blank' and 'omitted' as categories in addition to the actual image names. During each trial, ophys timepoints are labeled with the image code corresponding to the currently presented stimulus based on the 750ms presentation intervals. Timepoints outside any presentation get the 'blank' code.

ii.
```python
image_names = set(['blank', 'omitted'])
...
image_values = sorted(image_names)
image_to_idx = {name: i for i, name in enumerate(image_values)}
...
image_series = np.full(len(trial_ts), image_to_idx['blank'], dtype=np.int64)
for _, srow in stim_trial.iterrows():
    s0 = max(start, float(srow['start_time']))
    s1 = min(stop, float(srow['_end_time']))
    smask = (trial_ts >= s0) & (trial_ts < s1)
    name = str(srow['image_name']) if 'image_name' in srow.index else 'blank'
    image_series[smask] = image_to_idx.get(name, image_to_idx['blank'])
```

iii. The AI included 'blank' and 'omitted' as image categories to represent grey-screen periods and omitted stimuli, following the paper's treatment of omitted images.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed per ophys frame within the trial window, using the same index array as the neural data. Each frame is labeled based on which stimulus presentation interval it falls within.

ii. See 3-a code above — `trial_ts = ts[idx]` uses the same indices as `trial_neural = neural_mat[:, idx]`.

iii. Both neural and image identity use the same ophys frame indices, ensuring alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `stimulus_presentations` table by detecting transitions in `image_name` between consecutive stimulus presentations within a trial.

ii.
```python
prev_name = None
for _, srow in stim_trial.iterrows():
    ...
    if prev_name is not None and name != prev_name:
        hit = np.flatnonzero(smask)
        if hit.size:
            change_series[hit[0]] = 1
    prev_name = name
```

iii. The AI detects image changes by comparing consecutive stimulus names rather than using the trials table's `change_time` field.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary time series is created. For each stimulus presentation, if the image name differs from the previous stimulus's name, the first frame of that presentation is marked as 1. Otherwise, the series remains 0.

ii. See 4-a code above.

iii. This approach detects all image transitions within a trial, not just the primary "change" event.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is binary (0 or 1), with no thresholding — it is inherently categorical.

ii. `change_series = np.zeros(len(trial_ts), dtype=np.int64)` and `change_series[hit[0]] = 1`.

iii. The binary encoding directly represents the two categories (no_change, change).

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is computed per ophys frame within the trial window, using the same time indices as the neural data.

ii. The change is marked at the first frame where the new stimulus presentation begins.

iii. Same frame-level alignment as neural data and image identity.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `exp.running_speed`, using the `speed` and `timestamps` columns.

ii.
```python
run_t, run_v = get_running_series(exp.running_speed.copy())
```

iii. The `running_speed` attribute is the SDK's standard interface for locomotion data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated from its native timestamps to the trial's ophys timepoints using `np.interp`, then discretized into 5 bins using pre-computed percentile-based bin edges. Bin edges are estimated from the first 8 sessions rather than all sessions.

ii.
```python
# Bin edge computation (from subset):
probe_files = files[:min(8, len(files))]
for p in probe_files:
    exp = load_experiment(p)
    rt, rv = get_running_series(exp.running_speed.copy())
    run_vals.append(rv[np.isfinite(rv)])
run_all = np.concatenate(run_vals)
run_edges = np.quantile(run_all, [0, .2, .4, .6, .8, 1])

# Per-trial interpolation and binning:
run_interp = np.interp(trial_ts, run_t, run_v).astype(np.float32)
run_bins = np.digitize(run_interp, run_edges[1:-1], right=False).astype(np.int64)
```

iii. The CONVERSION_NOTES mention "Efficiency optimization: estimate bin edges from a small subset to avoid loading all NWB files twice." The AI chose to compute percentile edges from only 8 sessions as a performance optimization.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using `np.quantile` at [0, 0.2, 0.4, 0.6, 0.8, 1.0] percentiles (quintiles). `np.digitize` is used to assign each value to a bin.

ii.
```python
run_edges = np.quantile(run_all, [0, .2, .4, .6, .8, 1])
run_bins = np.digitize(run_interp, run_edges[1:-1], right=False).astype(np.int64)
```

iii. This implements "five equal percentile bins" as specified in the instructions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the trial's ophys timepoints using `np.interp`, so it shares the same temporal indices as the neural data.

ii.
```python
run_interp = np.interp(trial_ts, run_t, run_v).astype(np.float32)
```

iii. Linear interpolation to the ophys timebase ensures alignment with the neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `exp.eye_tracking`. The AI uses `pick_pupil_series()` which searches for columns in priority order: `pupil_diameter`, `pupil_area`, `pupil_width`, `pupil_height`, `pupil_radius`, `pupil_size`.

ii.
```python
def pick_pupil_series(eye_tracking):
    candidate_cols = ['pupil_diameter', 'pupil_area', 'pupil_width', 'pupil_height', 'pupil_radius', 'pupil_size']
    value_col = next((c for c in candidate_cols if c in cols), None)
    ...
    return np.asarray(eye_tracking[time_col], dtype=float), np.asarray(eye_tracking[value_col], dtype=float), value_col
```

iii. The AI wrote a flexible column selector. The actual column used depends on what's available in the data (likely `pupil_width` since that's what the SDK provides).

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter is linearly interpolated to ophys timepoints using `np.interp`, then discretized into 5 bins using percentile-based edges computed from the first 8 sessions. Notably, blink frames are NOT filtered before interpolation — only `np.isfinite` checks are applied when computing bin edges.

ii.
```python
if pupil_t is not None and pupil_v is not None:
    valid = np.isfinite(pupil_t) & np.isfinite(pupil_v)
    if valid.sum() >= 2:
        pupil_interp = np.interp(trial_ts, pupil_t[valid], pupil_v[valid]).astype(np.float32)
        pupil_bins = np.digitize(pupil_interp, pupil_edges[1:-1], right=False).astype(np.int64)
```

iii. The AI filters NaN/Inf values but does not specifically filter blink frames using the `likely_blink` column.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed — 5 bins using quintile percentile edges and `np.digitize`.

ii.
```python
pupil_edges = np.quantile(pupil_all, [0, .2, .4, .6, .8, 1])
pupil_bins = np.digitize(pupil_interp, pupil_edges[1:-1], right=False).astype(np.int64)
```

iii. Five equal percentile bins as specified in the instructions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — interpolated to the trial's ophys timepoints.

ii.
```python
pupil_interp = np.interp(trial_ts, pupil_t[valid], pupil_v[valid]).astype(np.float32)
```

iii. Linear interpolation ensures temporal alignment with neural data.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
def infer_trial_outcome(row):
    for key in ['hit', 'miss', 'false_alarm', 'correct_reject']:
        if key in row.index and bool(row[key]):
            return key
    return 'other'
```

iii. These four columns are the SDK's canonical trial outcome labels. The AI also includes an 'other' fallback category.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes using a fixed mapping that includes 5 categories: hit, miss, false_alarm, correct_reject, and other. The integer code is constant across all time bins within a trial (static per-trial, but broadcast to time-varying format).

ii.
```python
TRIAL_OUTCOME_VALUES = ['hit', 'miss', 'false_alarm', 'correct_reject', 'other']
outcome_to_idx = {name: i for i, name in enumerate(TRIAL_OUTCOME_VALUES)}
...
outcome_idx = outcome_to_idx.get(outcome, outcome_to_idx['other'])
outcome_series = np.full(len(trial_ts), outcome_idx, dtype=np.int64)
```

iii. The outcome is replicated to a time-varying format to match the output matrix shape.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: If `eye_tracking` fails or has no valid data, pupil bins default to all zeros.
- **Missing pupil data**: If fewer than 2 valid pupil samples exist, pupil bins are all zeros.
- **Trial too short**: Trials with fewer than 2 ophys frames are skipped.
- **Missing image names**: Default to 'blank' if `image_name` is not in the stimulus row.

ii.
```python
try:
    eye = exp.eye_tracking.copy()
except Exception:
    eye = None
...
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

iii. The code uses defensive checks and fallback values throughout to prevent crashes from missing data.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each NWB file via `load_experiment()`, which opens and parses the full NWB file. The AI also loads all files twice — once in `collect_global_info()` for the first 8 files to compute bin edges, and again in the main loop to process all files.

ii.
```python
def collect_global_info(files):
    probe_files = files[:min(8, len(files))]
    for p in probe_files:
        exp = load_experiment(p)
        ...
...
for i, p in enumerate(files, 1):
    exp = load_experiment(p)
```

iii. NWB file loading is I/O bound and involves parsing large HDF5 files.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-stimulus-presentation loop within `build_session` iterates over each stimulus row to assign image identity and detect changes. This could potentially be vectorized using array operations on the stimulus presentation times and ophys timestamps.

ii.
```python
for _, srow in stim_trial.iterrows():
    s0 = max(start, float(srow['start_time']))
    s1 = min(stop, float(srow['_end_time']))
    smask = (trial_ts >= s0) & (trial_ts < s1)
    ...
```

iii. The loop is not a major bottleneck compared to NWB loading, but it could be vectorized using `np.searchsorted` on stimulus times.

## 9-c. What processing does the code repeat multiple times?

i. The first 8 NWB files are loaded twice: once in `collect_global_info()` to estimate bin edges, and once in the main processing loop. This doubles the I/O cost for those files.

ii.
```python
def collect_global_info(files):
    probe_files = files[:min(8, len(files))]
    for p in probe_files:
        exp = load_experiment(p)
        ...
```

iii. The AI chose this approach for efficiency — estimating bin edges from a subset avoids loading all files twice. However, it still loads the subset files twice.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores `cell_ids`, `neural_source`, `image_names_seen`, and `trial_outcomes_seen` per session, but these are not included in the final output dictionary. Additionally, loading the full time bin size by re-opening `files[0]` at the end is redundant since timestamps are already available from the processed sessions.

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
...
dt_ms = float(np.median(np.diff(load_experiment(files[0]).ophys_timestamps)) * 1000.0)
```

iii. These extra fields are used for logging/printing during processing but are not part of the final saved dictionary.
