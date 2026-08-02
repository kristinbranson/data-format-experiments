# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads data by globbing all local NWB files under `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/`, then opens each file with `pynwb.NWBHDF5IO` and constructs an AllenSDK `BehaviorOphysExperiment` from the NWB object. It does not use `VisualBehaviorOphysProjectCache` or the experiment table.

ii.
```python
DATA_ROOT = Path('data/visual-behavior-ophys-1.1.0')
NWB_DIR = DATA_ROOT / 'behavior_ophys_experiments'

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

iii. In `CONVERSION_NOTES.md`, the agent explicitly decided to "convert only locally available NWB experiment files" because the local folder was a subset of the larger project release. The trajectory also shows it intentionally avoided the larger project metadata and treated the NWB files as the available source of truth.

## 1-b. How are the data split into subjects?

i. Subjects are split by the `mouse_id` stored in each loaded experiment's metadata. After all files are processed, unique mouse IDs are sorted and indexed.

ii.
```python
meta = exp.metadata if isinstance(exp.metadata, dict) else {}
mouse_id = str(meta.get('mouse_id', 'unknown'))
...
subjects = sorted({s['mouse_id'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
'subject_idx': np.array([subject_to_idx[s['mouse_id']] for s in sessions], dtype=np.int64),
```

iii. The notes map `metadata mouse_id` to `subjects / subject_idx`. No stronger justification was given beyond using NWB metadata for subject identity.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The agent does not group multiple experiments by `ophys_session_id`.

ii.
```python
def list_session_files(sample=False):
    files = sorted(NWB_DIR.glob('*.nwb'))
    return files[:2] if sample else files

for i, p in enumerate(files, 1):
    exp = load_experiment(p)
    sess = build_session(exp, image_to_idx, run_edges, pupil_edges, outcome_to_idx)
    sessions.append(sess)
```

iii. The notes say "Session source: Convert only locally available NWB experiment files" and repeatedly describe the local NWB files as the conversion unit. The trajectory shows the agent was aware this differed from the larger metadata release.

## 1-d. How are the data split into trials?

i. Trials are taken from `exp.trials`. For each retained trial, the agent uses the trial's `start_time` and `stop_time`/`end_time`, finds all ophys timestamps in that interval, and makes one variable-length trial slice.

ii.
```python
trials = valid_trials_table(exp.trials)
...
for _, tr in trials.iterrows():
    start = float(tr['start_time'])
    stop = float(tr['stop_time']) if 'stop_time' in tr.index else float(tr['end_time'])
    idx = np.flatnonzero((ts >= start) & (ts < stop))
    if idx.size < 2:
        continue
    trial_ts = ts[idx]
    trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
```

iii. The notes say "Trial segmentation: Use experiment-defined trial intervals from the trial table." The trajectory shows the agent intentionally reconciled the paper's image-level analyses with the instruction to segment by experiment-defined trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by dropping rows where any of `aborted`, `auto_rewarded`, `auto_rewarded_trial`, or `is_auto_rewarded` are true, and then keeping only `go`/`catch` trials if those columns exist. Trials with fewer than 2 ophys frames are skipped.

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
...
if idx.size < 2:
    continue
```

iii. The notes explicitly say to include Go/Catch and exclude Aborted/Auto-rewarded. No additional justification was given for omitting the reference solution's `change_time.notna()` filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural data is derived only from `exp.dff_traces['dff']`.

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

iii. The notes initially preferred events to match the paper, but later explicitly record that sample verification improved "after switching neural data to dF/F." The trajectory explains this change as a response to event sparsity and poor decoder performance.

## 2-b. How is the `neural` data processed?

i. Neural processing is minimal: each cell's dF/F trace is cast to `float32`, stacked into a neuron-by-time matrix, and then sliced into trial windows. There is no extra normalization or deconvolution in the script.

ii.
```python
traces = [np.asarray(v, dtype=np.float32) for v in dff['dff'].values]
mat = np.stack(traces, axis=0)
...
trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
```

iii. The trajectory says the agent switched from events to dF/F because events were "extremely sparse" and generated many all-zero trials, while dF/F worked better with the provided decoder.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit neural quality-control filter is applied. All cells present in `dff_traces` are kept.

ii.
```python
def get_cell_matrix(exp):
    dff = exp.dff_traces.copy()
    if isinstance(dff, pd.DataFrame) and 'dff' in dff.columns:
        ...
        return mat, cell_ids, 'dff'
```

iii. The notes do not document any cell-level QC beyond using what the NWB/AllenSDK exposes.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data is aligned by the ophys timestamp grid within each experiment-defined trial window. The script selects all ophys timestamps satisfying `start_time <= ts < stop_time`; it does not realign each trial to a common zero time like `trial_start` or `change_time`.

ii.
```python
ts = np.asarray(exp.ophys_timestamps, dtype=float)
...
start = float(tr['start_time'])
stop = float(tr['stop_time']) if 'stop_time' in tr.index else float(tr['end_time'])
idx = np.flatnonzero((ts >= start) & (ts < stop))
trial_ts = ts[idx]
trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
```

iii. The notes say "Temporal alignment: Use `ophys_timestamps` as the master clock" and the metadata says `"temporal_alignment_event": "ophys timestamps within experiment-defined trial window"`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data keeps the native ophys frame resolution. The time bin size is estimated as the median difference between the first session's ophys timestamps, converted to milliseconds. No temporal rebinning is applied to neural activity.

ii.
```python
dt_ms = float(np.median(np.diff(load_experiment(files[0]).ophys_timestamps)) * 1000.0)
...
'time_bin_size': dt_ms,
```

iii. The notes repeatedly describe `ophys_timestamps` as the master time base and do not mention any neural resampling.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `exp.stimulus_presentations['image_name']`, not from trial-table `initial_image_name` / `change_image_name`.

ii.
```python
stim = exp.stimulus_presentations.copy().sort_values('start_time')
...
name = str(srow['image_name']) if 'image_name' in srow.index else 'blank'
image_series[smask] = image_to_idx.get(name, image_to_idx['blank'])
```

iii. The trajectory says the agent changed its representation after debugging poor accuracy: it decided to label image identity over 750 ms image-presentation intervals based on the methods text rather than only over short flash epochs.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The agent creates a global image vocabulary from only the first 8 files plus special classes `blank` and `omitted`. Within each trial, it initializes all time bins to `blank`, extends each stimulus to a 750 ms interval (`start_time + 0.75`), and fills bins in that interval with the corresponding image code.

ii.
```python
image_names = set(['blank', 'omitted'])
probe_files = files[:min(8, len(files))]
...
image_values = sorted(image_names)
image_to_idx = {name: i for i, name in enumerate(image_values)}
...
stim_start = np.asarray(stim['start_time'], dtype=float)
stim_end = stim_start + 0.75
...
image_series = np.full(len(trial_ts), image_to_idx['blank'], dtype=np.int64)
...
image_series[smask] = image_to_idx.get(name, image_to_idx['blank'])
```

iii. `CONVERSION_NOTES.md` says verification passed after "labeling image identity over 750 ms image-presentation intervals." The trajectory justifies this as a fix for image identity being dominated by `blank`.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned directly on the same per-trial ophys timestamps as neural activity. For each stimulus overlapping the trial, the code marks the corresponding `trial_ts` bins.

ii.
```python
trial_ts = ts[idx]
...
stim_trial = stim[(stim['start_time'] < stop) & (stim['_end_time'] > start)]
for _, srow in stim_trial.iterrows():
    s0 = max(start, float(srow['start_time']))
    s1 = min(stop, float(srow['_end_time']))
    smask = (trial_ts >= s0) & (trial_ts < s1)
    image_series[smask] = image_to_idx.get(name, image_to_idx['blank'])
```

iii. The notes state that all time-varying streams are aligned to `ophys_timestamps`.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived implicitly from successive rows of `stimulus_presentations['image_name']` within each trial. It does not use `trials.change_time`.

ii.
```python
change_series = np.zeros(len(trial_ts), dtype=np.int64)
...
if prev_name is not None and name != prev_name:
    hit = np.flatnonzero(smask)
    if hit.size:
        change_series[hit[0]] = 1
prev_name = name
```

iii. The trajectory shows the agent wanted image-level labeling tied to 750 ms image-presentation intervals, so it detected image changes from successive stimulus identities rather than from the trial table.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code initializes a binary zero vector and sets a single time bin to 1 at the first ophys sample of any stimulus interval whose `image_name` differs from the previous interval's name.

ii.
```python
change_series = np.zeros(len(trial_ts), dtype=np.int64)
...
if prev_name is not None and name != prev_name:
    hit = np.flatnonzero(smask)
    if hit.size:
        change_series[hit[0]] = 1
```

iii. No separate written justification is given beyond the broader 750 ms stimulus-interval rationale in the notes and trajectory.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is treated as a binary categorical variable with values `no_change` and `change`.

ii.
```python
'output_values': [
    image_values,
    ['no_change', 'change'],
    [f'bin_{i}' for i in range(5)],
    [f'bin_{i}' for i in range(5)],
    TRIAL_OUTCOME_VALUES
],
```

iii. This follows directly from the decoder task; no extra justification was recorded.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The binary change series is generated on the same `trial_ts` axis used for neural slices, so it is frame-aligned to neural activity.

ii.
```python
trial_ts = ts[idx]
trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
...
smask = (trial_ts >= s0) & (trial_ts < s1)
...
output = np.vstack([image_series, change_series, run_bins, pupil_bins, outcome_series]).astype(np.int64)
```

iii. The notes say all time-varying outputs are aligned on the ophys clock.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `exp.running_speed`, using timestamp and speed columns.

ii.
```python
def get_running_series(run_df):
    cols = list(run_df.columns)
    tcol = 'timestamps' if 'timestamps' in cols else cols[0]
    vcol = 'speed' if 'speed' in cols else cols[1]
    return np.asarray(run_df[tcol], dtype=float), np.asarray(run_df[vcol], dtype=float)
...
run_t, run_v = get_running_series(exp.running_speed.copy())
```

iii. The notes map AllenSDK running-speed output directly to the decoder target.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The agent samples running-speed values from a small probe subset of files to estimate global bin edges, then linearly interpolates running speed onto each trial's ophys timestamps and discretizes it with `np.digitize`.

ii.
```python
probe_files = files[:min(8, len(files))]
...
run_vals.append(rv[np.isfinite(rv)])
...
run_edges = np.quantile(run_all, [0, .2, .4, .6, .8, 1])
...
run_interp = np.interp(trial_ts, run_t, run_v).astype(np.float32)
run_bins = np.digitize(run_interp, run_edges[1:-1], right=False).astype(np.int64)
```

iii. The only explicit justification is the code comment: "Efficiency optimization: estimate bin edges from a small subset to avoid loading all NWB files twice."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five bins using 0/20/40/60/80/100% quantiles of the pooled probe-file sample.

ii.
```python
run_edges = np.quantile(run_all, [0, .2, .4, .6, .8, 1])
...
run_bins = np.digitize(run_interp, run_edges[1:-1], right=False).astype(np.int64)
...
[f'bin_{i}' for i in range(5)]
```

iii. The notes say continuous outputs should use five equal-percentile bins, but the final script approximates those percentiles from the probe subset for speed.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated directly onto each trial's ophys timestamps (`trial_ts`) and therefore shares the neural trial axis exactly.

ii.
```python
trial_ts = ts[idx]
trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
...
run_interp = np.interp(trial_ts, run_t, run_v).astype(np.float32)
run_bins = np.digitize(run_interp, run_edges[1:-1], right=False).astype(np.int64)
```

iii. The notes identify `ophys_timestamps` as the common time base for all outputs.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `exp.eye_tracking`, but the script flexibly chooses the first available pupil-like numeric column from `pupil_diameter`, `pupil_area`, `pupil_width`, `pupil_height`, `pupil_radius`, or `pupil_size`.

ii.
```python
def pick_pupil_series(eye_tracking):
    ...
    candidate_cols = ['pupil_diameter', 'pupil_area', 'pupil_width', 'pupil_height', 'pupil_radius', 'pupil_size']
    value_col = next((c for c in candidate_cols if c in cols), None)
    if value_col is None:
        value_col = next((c for c in cols if 'pupil' in c.lower() and np.issubdtype(eye_tracking[c].dtype, np.number)), None)
```

iii. The notes planned to use "pupil diameter / eye tracking pupil area-equivalent" if needed. No narrower justification was given in the final script.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The agent estimates pupil bin edges from the probe subset, then linearly interpolates the chosen pupil signal onto each trial's ophys timestamps if at least two finite samples exist. Otherwise it fills the entire trial with zeros.

ii.
```python
pupil_vals.append(pv[np.isfinite(pv)])
...
pupil_edges = np.quantile(pupil_all, [0, .2, .4, .6, .8, 1])
...
valid = np.isfinite(pupil_t) & np.isfinite(pupil_v)
if valid.sum() >= 2:
    pupil_interp = np.interp(trial_ts, pupil_t[valid], pupil_v[valid]).astype(np.float32)
    pupil_bins = np.digitize(pupil_interp, pupil_edges[1:-1], right=False).astype(np.int64)
else:
    pupil_bins = np.zeros(len(trial_ts), dtype=np.int64)
```

iii. The notes say missing data should be handled carefully, but the final implementation uses a generic fallback-to-zero strategy rather than the earlier plan to remove blink frames.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil diameter is thresholded into five quantile bins derived from the pooled pupil values in the probe subset.

ii.
```python
pupil_edges = np.quantile(pupil_all, [0, .2, .4, .6, .8, 1])
...
pupil_bins = np.digitize(pupil_interp, pupil_edges[1:-1], right=False).astype(np.int64)
...
[f'bin_{i}' for i in range(5)]
```

iii. The notes planned percentile bins globally; the final script approximates that with the probe subset for efficiency.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. The pupil signal is interpolated onto `trial_ts`, the same ophys timestamps used for the per-trial neural slice.

ii.
```python
trial_ts = ts[idx]
trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
...
pupil_interp = np.interp(trial_ts, pupil_t[valid], pupil_v[valid]).astype(np.float32)
```

iii. The notes use the same common-clock rationale as for running speed.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
def infer_trial_outcome(row):
    for key in ['hit', 'miss', 'false_alarm', 'correct_reject']:
        if key in row.index and bool(row[key]):
            return key
    return 'other'
```

iii. The notes say trial outcomes should come from the trial table's Go/Catch annotations.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The first true outcome flag is mapped to an integer code, with fallback class `other`. That code is then repeated across every time bin of the trial.

ii.
```python
outcome_to_idx = {name: i for i, name in enumerate(TRIAL_OUTCOME_VALUES)}
...
outcome = infer_trial_outcome(tr)
outcome_idx = outcome_to_idx.get(outcome, outcome_to_idx['other'])
outcome_series = np.full(len(trial_ts), outcome_idx, dtype=np.int64)
...
output = np.vstack([image_series, change_series, run_bins, pupil_bins, outcome_series]).astype(np.int64)
```

iii. The notes say trial outcome should be static per trial, and the agent implemented that as a time-constant row.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The script uses permissive fallbacks: missing pupil data returns `(None, None, None)` and later all-zero bins; missing `stop_time` falls back to `end_time`; missing pupil values are ignored unless there are fewer than two finite samples; absent `image_name` falls back to `blank`; too-short trials are skipped; unknown metadata fall back to `'unknown'`.

ii.
```python
if eye_tracking is None or len(eye_tracking) == 0:
    return None, None, None
...
if time_col is None or value_col is None:
    return None, None, None
...
stop = float(tr['stop_time']) if 'stop_time' in tr.index else float(tr['end_time'])
...
if idx.size < 2:
    continue
...
name = str(srow['image_name']) if 'image_name' in srow.index else 'blank'
...
else:
    pupil_bins = np.zeros(len(trial_ts), dtype=np.int64)
...
targeted_structure = str(meta.get('targeted_structure', 'unknown'))
mouse_id = str(meta.get('mouse_id', 'unknown'))
```

iii. The notes and trajectory frame these choices as robustness measures needed to handle inconsistent local NWB contents without crashing conversion.

## 9-a. What are the most time-consuming steps of the code?

i. The expensive steps are repeated NWB file loading and per-session AllenSDK object construction, plus full-session trial/stimulus interpolation across 284 files. The script also does an extra probe pass to compute discretization metadata and another load to estimate `time_bin_size`.

ii.
```python
for p in probe_files:
    exp = load_experiment(p)
    ...
for i, p in enumerate(files, 1):
    exp = load_experiment(p)
    sess = build_session(exp, image_to_idx, run_edges, pupil_edges, outcome_to_idx)
...
dt_ms = float(np.median(np.diff(load_experiment(files[0]).ophys_timestamps)) * 1000.0)
```

iii. The trajectory explicitly identified `collect_global_info(files)` as slow because it loads files before the main processing loop, and the code comment calls the probe pass an "Efficiency optimization".

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The code keeps several Python loops that could be vectorized or precomputed: over probe files, over trials within each session, and over stimulus intervals within each trial. Image-label assignment especially uses a nested trial/stimulus loop with boolean masks per interval.

ii.
```python
for p in probe_files:
    ...
for _, tr in trials.iterrows():
    ...
    for _, srow in stim_trial.iterrows():
        smask = (trial_ts >= s0) & (trial_ts < s1)
        image_series[smask] = image_to_idx.get(name, image_to_idx['blank'])
```

iii. There is no explicit note defending these loops. The only efficiency-related justification is the separate probe subset introduced to avoid loading all files twice.

## 9-c. What processing does the code repeat multiple times?

i. The script repeats experiment loading: once for each probe file in `collect_global_info`, again for every file in the main conversion loop, and once more for the first file when estimating `time_bin_size`.

ii.
```python
for p in probe_files:
    exp = load_experiment(p)
...
for i, p in enumerate(files, 1):
    exp = load_experiment(p)
...
dt_ms = float(np.median(np.diff(load_experiment(files[0]).ophys_timestamps)) * 1000.0)
```

iii. The trajectory shows the agent knowingly accepted some repeated work after deciding a probe subset was a speed/accuracy compromise.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes and stores several session-level values that are not used in the final dataset: `stim_start`, `pupil_col`, `image_names_seen`, `trial_outcomes_seen`, and `neural_source`. It also constructs per-session dictionaries and then discards much of that information during final assembly.

ii.
```python
stim_start = np.asarray(stim['start_time'], dtype=float)
...
pupil_t, pupil_v, pupil_col = pick_pupil_series(eye)
...
image_names_seen = []
trial_outcomes_seen = []
...
'neural_source': neural_source,
'image_names_seen': sorted(set(image_names_seen)),
'trial_outcomes_seen': sorted(set(trial_outcomes_seen)),
```

iii. No explicit justification is given. These appear to be leftover bookkeeping and debugging aids from the agent's iterative development process.
