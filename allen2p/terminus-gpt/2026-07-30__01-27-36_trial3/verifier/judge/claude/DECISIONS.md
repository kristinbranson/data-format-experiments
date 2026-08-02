# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by directly reading NWB files from the local filesystem using `pynwb` and `BehaviorOphysExperiment.from_nwb()`. It lists all `.nwb` files in the `behavior_ophys_experiments` directory and processes each one individually. Global info (image names, bin edges) is estimated from a subset of 8 files first, then all files are processed in a second pass.

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

iii. The AI chose to load NWB files directly rather than using the S3 cache API because the data was available locally. The CONVERSION_NOTES state: "Convert only locally available NWB experiment files, not the full project metadata release, because only those sessions are actually present."

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `mouse_id` field in each experiment's metadata dictionary. Unique mouse IDs are collected across all sessions after processing.

ii.
```python
meta = exp.metadata if isinstance(exp.metadata, dict) else {}
mouse_id = str(meta.get('mouse_id', 'unknown'))
...
subjects = sorted({s['mouse_id'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The AI extracts mouse_id from individual experiment metadata rather than from a project-level experiment table. This yields 38 subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file (one per experiment/imaging plane) is treated as a separate session. There is no grouping of multiple imaging planes by `ophys_session_id`. This results in 284 sessions (one per NWB file).

ii.
```python
files = sorted(NWB_DIR.glob('*.nwb'))
...
for i, p in enumerate(files, 1):
    exp = load_experiment(p)
    sess = build_session(exp, image_to_idx, run_edges, pupil_edges, outcome_to_idx)
    sessions.append(sess)
```

iii. The AI treats each experiment file as a session because each NWB file contains one imaging plane. The CONVERSION_NOTES note that the local dataset has 284 NWB files. The AI did not group experiments by `ophys_session_id` to merge multi-plane sessions.

## 1-d. How are the data split into trials?

i. Trials are defined using the experiment's built-in trials table. Aborted and auto-rewarded trials are excluded using multiple possible column names. Go and catch trials are retained. Trial boundaries use `start_time` and `stop_time` from the trials table, producing variable-length trials.

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

iii. The AI's trial filtering is more defensive, checking multiple column name variants for auto_rewarded and using go/catch flags to additionally ensure only go and catch trials are included. This follows the instruction to include Go and Catch trials and exclude Aborted and Auto-rewarded.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 ophys frames are skipped. No other quality control filtering is applied beyond the trial type filtering described in 1-d.

ii.
```python
idx = np.flatnonzero((ts >= start) & (ts < stop))
if idx.size < 2:
    continue
```

iii. The AI applies minimal quality filtering: only skipping trials with insufficient data points. No minimum trial count per session is enforced (unlike the reference which requires >= 2 trials per session).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `dff_traces` (dF/F calcium fluorescence traces) from each experiment, accessed via `exp.dff_traces`.

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

iii. The CONVERSION_NOTES state the AI initially planned to use detected calcium events to match the paper, but fell back to dF/F: "Prefer detected calcium events to match paper; use dF/F only if events are inaccessible/impractical." The final code uses dF/F.

## 2-b. How is the `neural` data processed?

i. No additional processing is applied beyond converting dF/F traces to float32 arrays. Each experiment's neurons are stacked into a matrix of shape (n_neurons, T).

ii.
```python
traces = [np.asarray(v, dtype=np.float32) for v in dff['dff'].values]
mat = np.stack(traces, axis=0)
...
trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
```

iii. The AI does not apply any normalization, smoothing, or additional filtering. The dF/F traces from the Allen SDK pipeline are used as-is.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No quality filtering is applied to individual neurons. All neurons present in the dF/F traces are included.

ii. N/A (no filtering code)

iii. The AI relies on the Allen SDK's built-in quality control for cell segmentation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial window defined by `start_time` and `stop_time` from the trials table. Ophys frames within this window are selected using boolean indexing on `ophys_timestamps`.

ii.
```python
idx = np.flatnonzero((ts >= start) & (ts < stop))
trial_ts = ts[idx]
trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
```

iii. The AI uses `>=` for start and `<` for stop to select ophys frames. This differs from the reference which uses `np.searchsorted` for both boundaries. The `< stop` boundary means the frame exactly at `stop_time` is excluded.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The native ophys frame rate (~11 Hz, ~93 ms per frame) is preserved. The time bin size is computed from the median inter-frame interval of the first session's ophys timestamps.

ii.
```python
dt_ms = float(np.median(np.diff(load_experiment(files[0]).ophys_timestamps)) * 1000.0)
```

iii. The AI keeps the native temporal resolution, consistent with the reference approach.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `stimulus_presentations` table, specifically the `image_name` column and the `start_time` of each stimulus presentation. Each presentation is treated as a 750ms interval.

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

iii. The AI uses the stimulus_presentations table to construct a frame-by-frame image identity, labeling each ophys frame based on which 750ms stimulus interval it falls within. This includes 'blank' and 'omitted' categories (18 total image values).

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are collected globally across all sessions (including 'blank' and 'omitted'), sorted, and mapped to integer indices. For each trial, each ophys frame is labeled with the image shown during the 750ms stimulus interval containing that frame. Default is 'blank'.

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
    image_series[smask] = image_to_idx.get(name, image_to_idx['blank'])
```

iii. The AI maps each ophys frame to the stimulus presentation interval it falls in. This produces 18 categories (16 natural images + blank + omitted). The reference uses only `initial_image_name` and `change_image_name` from the trials table (8 images).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is computed on the same ophys frame indices as the neural data, within the trial window. Both use the same `trial_ts` array.

ii.
```python
trial_ts = ts[idx]
trial_neural = neural_mat[:, idx]
image_series = np.full(len(trial_ts), image_to_idx['blank'], dtype=np.int64)
```

iii. Alignment is guaranteed by using the same ophys timestamp indices for both neural and image identity data.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `stimulus_presentations` table by comparing consecutive `image_name` values. When a stimulus has a different name from the previous one, the first frame of that stimulus is marked as a change.

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

iii. The AI detects changes based on transitions in image name in the stimulus_presentations table, marking only the first frame of the new stimulus.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary series is created with 1 at the first frame when the stimulus image name differs from the previous stimulus, and 0 everywhere else. This marks only a single frame per change event.

ii. See 4-a code snippet.

iii. The change detection is based on consecutive stimulus presentation comparisons, not on the `change_time` field from the trials table. This marks only one frame (not a 750ms window as in the reference).

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is a binary variable (0 or 1). No thresholding is needed; 0 = no_change, 1 = change.

ii.
```python
change_series = np.zeros(len(trial_ts), dtype=np.int64)
...
change_series[hit[0]] = 1
```

iii. The output_values for this dimension are `['no_change', 'change']`.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is computed on the same ophys timestamps as the neural data, within the trial window.

ii. Same as 3-c — uses the same `trial_ts` / `idx` array.

iii. Alignment is consistent with neural data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `exp.running_speed`, using the `speed` and `timestamps` columns.

ii.
```python
run_t, run_v = get_running_series(exp.running_speed.copy())

def get_running_series(run_df):
    cols = list(run_df.columns)
    tcol = 'timestamps' if 'timestamps' in cols else cols[0]
    vcol = 'speed' if 'speed' in cols else cols[1]
    return np.asarray(run_df[tcol], dtype=float), np.asarray(run_df[vcol], dtype=float)
```

iii. The AI uses the standard running_speed accessor from the Allen SDK.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is interpolated to ophys timestamps using `np.interp`, then discretized into 5 bins using percentile-based edges computed from a subset of 8 sessions.

ii.
```python
run_interp = np.interp(trial_ts, run_t, run_v).astype(np.float32)
run_bins = np.digitize(run_interp, run_edges[1:-1], right=False).astype(np.int64)
...
run_edges = np.quantile(run_all, [0, .2, .4, .6, .8, 1])
```

iii. The AI uses `np.interp` which does not produce NaN for out-of-range values (it extrapolates with edge values), unlike the reference which uses `scipy.interpolate.interp1d` with `fill_value=np.nan`. Bin edges are computed from only 8 sessions instead of all sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins (0-4) using `np.digitize` with percentile-based bin edges at quantiles [0, 0.2, 0.4, 0.6, 0.8, 1.0].

ii.
```python
run_edges = np.quantile(run_all, [0, .2, .4, .6, .8, 1])
run_bins = np.digitize(run_interp, run_edges[1:-1], right=False).astype(np.int64)
```

iii. The verification output shows unbalanced bins: `bin_0 (0.150), bin_1 (0.096), bin_2 (0.104), bin_3 (0.166), bin_4 (0.484)`. This imbalance is likely because bin edges were estimated from only 8 sessions rather than all 284.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated per-trial to the trial's ophys timestamps, so it shares the same time indices as the neural data.

ii.
```python
run_interp = np.interp(trial_ts, run_t, run_v).astype(np.float32)
```

iii. Alignment is achieved by interpolating running speed to the ophys timestamps for each trial.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `exp.eye_tracking`. The AI uses a priority list of column names: `pupil_diameter`, `pupil_area`, `pupil_width`, `pupil_height`, `pupil_radius`, `pupil_size`.

ii.
```python
def pick_pupil_series(eye_tracking):
    candidate_cols = ['pupil_diameter', 'pupil_area', 'pupil_width', 'pupil_height', 'pupil_radius', 'pupil_size']
    value_col = next((c for c in candidate_cols if c in cols), None)
    ...
    return np.asarray(eye_tracking[time_col], dtype=float), np.asarray(eye_tracking[value_col], dtype=float), value_col
```

iii. The AI prioritizes `pupil_diameter` over `pupil_width`. The actual column used depends on which is available in the data. The reference explicitly uses `pupil_width`.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil values are interpolated to ophys timestamps using `np.interp` on finite values only, then discretized into 5 bins. No explicit blink removal is performed (unlike the reference which filters `likely_blink` frames).

ii.
```python
if pupil_t is not None and pupil_v is not None:
    valid = np.isfinite(pupil_t) & np.isfinite(pupil_v)
    if valid.sum() >= 2:
        pupil_interp = np.interp(trial_ts, pupil_t[valid], pupil_v[valid]).astype(np.float32)
        pupil_bins = np.digitize(pupil_interp, pupil_edges[1:-1], right=False).astype(np.int64)
    else:
        pupil_bins = np.zeros(len(trial_ts), dtype=np.int64)
```

iii. The AI filters for finite values before interpolation but does not explicitly remove blink frames using the `likely_blink` column. Bin edges are estimated from a subset of sessions.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is discretized into 5 bins using `np.digitize` with percentile-based edges.

ii.
```python
pupil_edges = np.quantile(pupil_all, [0, .2, .4, .6, .8, 1])
pupil_bins = np.digitize(pupil_interp, pupil_edges[1:-1], right=False).astype(np.int64)
```

iii. The verification output shows highly unbalanced pupil bins: `bin_0 (0.835), bin_1 (0.112), bin_2 (0.027), bin_3 (0.017), bin_4 (0.008)`. This extreme imbalance suggests the bin edges from the 8-session subset are not representative of the full dataset.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated per-trial to the trial's ophys timestamps, sharing the same time indices as the neural data.

ii.
```python
pupil_interp = np.interp(trial_ts, pupil_t[valid], pupil_v[valid]).astype(np.float32)
```

iii. Alignment is achieved through interpolation to ophys timestamps.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, `correct_reject` in the trials table, with an additional `other` fallback category.

ii.
```python
TRIAL_OUTCOME_VALUES = ['hit', 'miss', 'false_alarm', 'correct_reject', 'other']

def infer_trial_outcome(row):
    for key in ['hit', 'miss', 'false_alarm', 'correct_reject']:
        if key in row.index and bool(row[key]):
            return key
    return 'other'
```

iii. The AI includes 5 outcome categories (including 'other'), while the reference uses only 4.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is mapped to an integer code (0-4) and broadcast to all time frames within the trial as a constant value.

ii.
```python
outcome = infer_trial_outcome(tr)
outcome_idx = outcome_to_idx.get(outcome, outcome_to_idx['other'])
outcome_series = np.full(len(trial_ts), outcome_idx, dtype=np.int64)
```

iii. The outcome is static per trial but represented as a time-varying series (same value at every frame), matching the reference approach.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: If eye_tracking raises an exception or is None/empty, pupil bins default to 0.
- **Insufficient pupil data**: If fewer than 2 finite pupil values, pupil bins default to 0.
- **Short trials**: Trials with fewer than 2 ophys frames are skipped.
- **Column name variants**: The code checks multiple possible column names for auto_rewarded and pupil data.

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
```

iii. The AI takes a defensive approach with try/except blocks and column name fallbacks. However, it does not handle failed sessions with a try/except around the main per-session loop (unlike the reference).

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each NWB file via `load_experiment()` and constructing the `BehaviorOphysExperiment` object. Each file takes ~3-4 seconds. Additionally, the global info pass (`collect_global_info`) loads 8 files redundantly.

ii.
```python
exp = load_experiment(p)  # ~3-4s per file
sess = build_session(exp, ...)
```

iii. From the conversion output, each session takes about 3-5 seconds, totaling ~17 minutes for 284 files.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-stimulus-presentation loop inside `build_session` iterates over each stimulus row to assign image identity and detect changes. This could be vectorized using `np.searchsorted` or similar array operations.

ii.
```python
for _, srow in stim_trial.iterrows():
    s0 = max(start, float(srow['start_time']))
    s1 = min(stop, float(srow['_end_time']))
    smask = (trial_ts >= s0) & (trial_ts < s1)
    ...
```

iii. The per-row iteration over stimulus presentations within each trial is potentially slow for long trials with many stimulus presentations.

## 9-c. What processing does the code repeat multiple times?

i. The code loads NWB files twice: once in `collect_global_info` (first 8 files) to estimate bin edges and image names, and again in the main processing loop for all 284 files. The first 8 files are loaded and processed redundantly.

ii.
```python
# First pass: load 8 files for global info
probe_files = files[:min(8, len(files))]
for p in probe_files:
    exp = load_experiment(p)
    ...

# Second pass: load all files
for i, p in enumerate(files, 1):
    exp = load_experiment(p)
    ...
```

iii. This redundant loading adds ~30 seconds of unnecessary I/O.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `cell_ids`, `neural_source`, `image_names_seen`, and `trial_outcomes_seen` per session, but only `cell_ids` and `targeted_structure` and `mouse_id` are used in the final output assembly. The intermediate lists `image_names_seen` and `trial_outcomes_seen` are not used after being collected.

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

iii. These extra fields consume memory and processing time but are not referenced in the final data assembly.
