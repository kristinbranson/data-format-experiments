# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all available local NWB files by globbing `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/*.nwb`. Each file is opened with `pynwb`, converted to an AllenSDK `BehaviorOphysExperiment`, and then processed one-by-one. It does not use `VisualBehaviorOphysProjectCache` or the project experiment table to discover the full VisualBehavior dataset.

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

files = list_session_files(sample=sample)
for i, p in enumerate(files, 1):
    exp = load_experiment(p)
    sess = build_session(exp, image_to_idx, run_edges, pupil_edges, outcome_to_idx)
```

iii. The justification is explicit in `CONVERSION_NOTES.md`: the agent decided that the local `behavior_ophys_experiments/` folder is a subset of the larger release and that conversion should operate on the locally available NWB files only. The trajectory repeats that decision and treats file-by-file NWB loading as the canonical input source.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from each loaded experiment's metadata. After all files are processed, the AI forms `subjects` as the sorted set of session-level `mouse_id` values and builds `subject_idx` from that mapping.

ii. 
```python
meta = exp.metadata if isinstance(exp.metadata, dict) else {}
mouse_id = str(meta.get('mouse_id', 'unknown'))

subjects = sorted({s['mouse_id'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.array([subject_to_idx[s['mouse_id']] for s in sessions], dtype=np.int64),
```

iii. The notes say the mapping for `subjects / subject_idx` should come from metadata `mouse_id`, and the final code follows that plan.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The AI does not group multiple experiments by `ophys_session_id`; instead, file order defines session order in the output.

ii. 
```python
def list_session_files(sample=False):
    files = sorted(NWB_DIR.glob('*.nwb'))
    return files[:2] if sample else files

sessions = []
for i, p in enumerate(files, 1):
    exp = load_experiment(p)
    sess = build_session(exp, image_to_idx, run_edges, pupil_edges, outcome_to_idx)
    sessions.append(sess)
```

iii. `CONVERSION_NOTES.md` Step 2 describes the local data as “one file per available ophys experiment,” and the trajectory consistently uses file-level processing rather than reconstructing multi-plane sessions from metadata.

## 1-d. How are the data split into trials?

i. Trials come from `exp.trials`. The AI filters that table with `valid_trials_table`, then for each remaining row takes all ophys timestamps satisfying `start_time <= ts < stop_time`. Each such slice becomes one trial.

ii. 
```python
def valid_trials_table(trials):
    df = trials.copy()
    ...
    return df.reset_index(drop=True)

trials = valid_trials_table(exp.trials)

for _, tr in trials.iterrows():
    start = float(tr['start_time'])
    stop = float(tr['stop_time']) if 'stop_time' in tr.index else float(tr['end_time'])
    idx = np.flatnonzero((ts >= start) & (ts < stop))
    if idx.size < 2:
        continue
```

iii. The notes and trajectory say trial segmentation should use experiment-defined trial intervals from the SDK trial table. The code implements that directly.

## 1-e. How are trials filtered based on quality controls?

i. The AI removes trials marked `aborted`, `auto_rewarded`, `auto_rewarded_trial`, or `is_auto_rewarded`, and then keeps only `go` or `catch` trials if those columns exist. It also drops trials with fewer than two ophys frames. It does not explicitly require non-null `change_time`.

ii. 
```python
for col in ['aborted', 'auto_rewarded', 'auto_rewarded_trial', 'is_auto_rewarded']:
    if col in df.columns:
        df = df[~df[col].fillna(False)]
if 'go' in df.columns and 'catch' in df.columns:
    keep = df['go'].fillna(False) | df['catch'].fillna(False)
...
idx = np.flatnonzero((ts >= start) & (ts < stop))
if idx.size < 2:
    continue
```

iii. The stated justification in the notes is to include Go and Catch trials and exclude Aborted and Auto-rewarded trials per the task instructions. The extra column-name fallbacks suggest a robustness-oriented choice for schema variation. No separate justification is given for omitting a `change_time` validity check.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural signal is derived from `exp.dff_traces['dff']`.

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

iii. The trajectory shows the agent originally considered events, then switched to dF/F after observing that event traces were extremely sparse in this local subset. `CONVERSION_NOTES.md` Step 7 also says verification passed “after switching neural data to dF/F.”

## 2-b. How is the `neural` data processed?

i. The AI stacks all per-cell dF/F traces from one NWB experiment into a dense `neurons x time` matrix, then slices that matrix into trial windows. There is no additional normalization, denoising, or multi-plane merging.

ii. 
```python
traces = [np.asarray(v, dtype=np.float32) for v in dff['dff'].values]
mat = np.stack(traces, axis=0)
...
trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
```

iii. The documented justification is pragmatic rather than paper-faithful: the trajectory says the agent forced dF/F because events were too sparse for decoder training, and the notes explicitly mention the switch from events to dF/F during debugging.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level quality filtering is applied in the conversion script. Every cell present in `dff_traces` is used.

ii. 
```python
if isinstance(dff, pd.DataFrame) and 'dff' in dff.columns:
    cell_ids = np.asarray(dff.index)
    traces = [np.asarray(v, dtype=np.float32) for v in dff['dff'].values]
    mat = np.stack(traces, axis=0)
```

iii. No explicit neural QC rule is implemented. The notes only say to rely on the data exposed by `BehaviorOphysExperiment` unless additional reference filtering is confirmed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to the ophys timestamps inside each trial window. For a given trial, the AI takes all ophys frames between `start_time` and `stop_time`, so trial arrays begin at trial start rather than at `change_time`.

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

iii. The notes repeatedly state that `ophys_timestamps` are the master clock. The metadata field `temporal_alignment_event` was set to “ophys timestamps within experiment-defined trial window,” which matches that design.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native ophys frame rate and records `time_bin_size` as the median difference between consecutive ophys timestamps from the first file. No temporal rebinning is applied to neural data.

ii. 
```python
dt_ms = float(np.median(np.diff(load_experiment(files[0]).ophys_timestamps)) * 1000.0)
...
'time_bin_size': dt_ms,
```

iii. The notes and code both treat ophys timestamps as the canonical time base, so keeping the native resolution is the intended behavior.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `exp.stimulus_presentations['image_name']`, not from trial-table fields like `initial_image_name` or `change_image_name`.

ii. 
```python
stim = exp.stimulus_presentations.copy().sort_values('start_time')
...
name = str(srow['image_name']) if 'image_name' in srow.index else 'blank'
image_series[smask] = image_to_idx.get(name, image_to_idx['blank'])
```

iii. The justification is documented in the notes and trajectory: the agent wanted image labels based on stimulus presentation intervals on the ophys clock, and later expanded those intervals to 750 ms to match the methods text more closely.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI initializes every time bin in a trial to a `blank` class, then overwrites bins covered by overlapping stimulus-presentation intervals with the corresponding `image_name`. Stimulus intervals are extended to `start_time + 0.75`, and a global mapping from image names to integer codes is built using names seen in a probe subset plus reserved `blank` and `omitted` classes.

ii. 
```python
image_names = set(['blank', 'omitted'])
...
stim_start = np.asarray(stim['start_time'], dtype=float)
stim_end = stim_start + 0.75
stim['_end_time'] = stim_end
...
image_series = np.full(len(trial_ts), image_to_idx['blank'], dtype=np.int64)
...
image_series[smask] = image_to_idx.get(name, image_to_idx['blank'])
...
image_values = sorted(image_names)
image_to_idx = {name: i for i, name in enumerate(image_values)}
```

iii. The explicit comment in `collect_global_info` says the agent estimated global information from a small probe subset for efficiency. The trajectory also says the 750 ms interval expansion was introduced to reduce blank dominance and better reflect the methods description of image-presentation intervals.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned frame-by-frame on the same per-trial ophys timestamps used for the neural slices. For each stimulus interval overlapping the trial, a boolean mask on `trial_ts` is used to write image labels.

ii. 
```python
trial_ts = ts[idx]
trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
...
smask = (trial_ts >= s0) & (trial_ts < s1)
image_series[smask] = image_to_idx.get(name, image_to_idx['blank'])
```

iii. The justification is the same master-clock decision: `ophys_timestamps` are used for all time-varying signals.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from transitions in consecutive rows of `stimulus_presentations['image_name']` within a trial, not from the trial table's `change_time` and `go` fields.

ii. 
```python
stim_trial = stim[(stim['start_time'] < stop) & (stim['_end_time'] > start)]
prev_name = None
for _, srow in stim_trial.iterrows():
    ...
    name = str(srow['image_name']) if 'image_name' in srow.index else 'blank'
    if prev_name is not None and name != prev_name:
        hit = np.flatnonzero(smask)
        if hit.size:
            change_series[hit[0]] = 1
    prev_name = name
```

iii. No separate written justification is given beyond the general choice to derive stimulus labels from `stimulus_presentations` on the ophys clock.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The AI starts with an all-zero vector and sets a single frame to `1` at the first frame of any overlapping stimulus interval whose `image_name` differs from the previous overlapping stimulus interval.

ii. 
```python
change_series = np.zeros(len(trial_ts), dtype=np.int64)
...
if prev_name is not None and name != prev_name:
    hit = np.flatnonzero(smask)
    if hit.size:
        change_series[hit[0]] = 1
```

iii. The code implies a “mark the transition point” interpretation of change. The trajectory frames this as an image-identity transition signal rather than a `go`-trial change window.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is represented as a binary categorical variable with values `0 = no_change` and `1 = change`. There is no additional thresholding step.

ii. 
```python
change_series = np.zeros(len(trial_ts), dtype=np.int64)
...
'output_values': [image_values, ['no_change', 'change'], ...]
```

iii. The justification is implicit in the decoder target itself: image change is treated as a binary output.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is aligned on the same `trial_ts` grid as the neural data and image-identity output, using per-trial boolean masks derived from ophys timestamps.

ii. 
```python
trial_ts = ts[idx]
trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
...
smask = (trial_ts >= s0) & (trial_ts < s1)
...
change_series[hit[0]] = 1
```

iii. The notes consistently justify time-varying outputs by direct alignment to `ophys_timestamps`.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `exp.running_speed`, using one timestamp column and one value column selected by `get_running_series`.

ii. 
```python
def get_running_series(run_df):
    cols = list(run_df.columns)
    tcol = 'timestamps' if 'timestamps' in cols else cols[0]
    vcol = 'speed' if 'speed' in cols else cols[1]
    return np.asarray(run_df[tcol], dtype=float), np.asarray(run_df[vcol], dtype=float)
```

iii. This follows the notes’ plan to use the SDK running-speed stream and align it to ophys timestamps.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The AI collects running values from only the first up to eight files to estimate global bin edges, computes 20/40/60/80% quantiles with `np.quantile`, then linearly interpolates each trial’s running speed onto `trial_ts` and discretizes the interpolated values with `np.digitize`.

ii. 
```python
probe_files = files[:min(8, len(files))]
...
run_all = np.concatenate(run_vals) if run_vals else np.array([0.0])
run_edges = np.quantile(run_all, [0, .2, .4, .6, .8, 1])
...
run_interp = np.interp(trial_ts, run_t, run_v).astype(np.float32)
run_bins = np.digitize(run_interp, run_edges[1:-1], right=False).astype(np.int64)
```

iii. The code comment gives the justification directly: “Efficiency optimization: estimate bin edges from a small subset to avoid loading all NWB files twice.”

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is discretized into five bins defined by the 0/20/40/60/80/100% quantiles of the pooled running-speed values from the probe subset.

ii. 
```python
run_edges = np.quantile(run_all, [0, .2, .4, .6, .8, 1])
...
run_bins = np.digitize(run_interp, run_edges[1:-1], right=False).astype(np.int64)
...
'output_values': [..., [f'bin_{i}' for i in range(5)], ...]
```

iii. The justification is again the efficiency optimization in `collect_global_info`.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated directly onto the same per-trial ophys timestamps `trial_ts` used for neural slices.

ii. 
```python
trial_ts = ts[idx]
trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
...
run_interp = np.interp(trial_ts, run_t, run_v).astype(np.float32)
```

iii. The notes explicitly say `ophys_timestamps` are the master alignment clock for all streams.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `exp.eye_tracking`, but the AI chooses the first available pupil-like numeric column from a candidate list rather than fixing the source to `pupil_width`.

ii. 
```python
candidate_cols = ['pupil_diameter', 'pupil_area', 'pupil_width', 'pupil_height', 'pupil_radius', 'pupil_size']
value_col = next((c for c in candidate_cols if c in cols), None)
if value_col is None:
    value_col = next((c for c in cols if 'pupil' in c.lower() and np.issubdtype(eye_tracking[c].dtype, np.number)), None)
```

iii. No explicit written justification is given beyond schema robustness. The code clearly prefers “whatever pupil-like column exists” over a reference-specific choice.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The AI gathers finite pupil values from up to eight probe files to estimate quantile bin edges. For each trial, if at least two finite pupil samples exist, it linearly interpolates that pupil signal onto `trial_ts`; otherwise it fills the whole trial with zeros. It does not remove `likely_blink` frames.

ii. 
```python
probe_files = files[:min(8, len(files))]
...
if pv is not None:
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

iii. The only explicit justification is the same efficiency comment for subset-based binning. The fallback-to-zero behavior is an implicit robustness choice for missing or unusable eye-tracking data.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It is discretized into five bins from the pooled 0/20/40/60/80/100% quantiles of finite pupil values from the probe subset.

ii. 
```python
pupil_edges = np.quantile(pupil_all, [0, .2, .4, .6, .8, 1])
...
pupil_bins = np.digitize(pupil_interp, pupil_edges[1:-1], right=False).astype(np.int64)
...
'output_values': [..., [f'bin_{i}' for i in range(5)], ...]
```

iii. The justification is the efficiency optimization documented in `collect_global_info`.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. As with running speed, pupil is aligned by interpolation onto the per-trial ophys timestamps `trial_ts`.

ii. 
```python
trial_ts = ts[idx]
...
pupil_interp = np.interp(trial_ts, pupil_t[valid], pupil_v[valid]).astype(np.float32)
```

iii. The code follows the notes’ master-clock decision to align all time-varying streams to ophys timestamps.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table fields `hit`, `miss`, `false_alarm`, and `correct_reject`, with a fallback category `other`.

ii. 
```python
TRIAL_OUTCOME_VALUES = ['hit', 'miss', 'false_alarm', 'correct_reject', 'other']

def infer_trial_outcome(row):
    for key in ['hit', 'miss', 'false_alarm', 'correct_reject']:
        if key in row.index and bool(row[key]):
            return key
    return 'other'
```

iii. No detailed written justification is given. The fallback `other` appears to be a defensive coding choice for unexpected trial rows.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The AI maps the inferred string outcome to an integer code and repeats that code across every frame in the trial.

ii. 
```python
outcome_to_idx = {name: i for i, name in enumerate(TRIAL_OUTCOME_VALUES)}
...
outcome = infer_trial_outcome(tr)
outcome_idx = outcome_to_idx.get(outcome, outcome_to_idx['other'])
outcome_series = np.full(len(trial_ts), outcome_idx, dtype=np.int64)
```

iii. The justification is implicit in the target format: the decoder expects categorical outputs, and the agent chose to make the static per-trial label time-aligned by repeating it across the trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The script is robust to a range of schema and missing-data issues. It accepts several auto-reward column names, falls back from `stop_time` to `end_time`, chooses the first available pupil-like column, returns `(None, None, None)` if eye tracking is missing, and fills pupil bins with zeros when there are too few valid samples. For trials, it skips any with fewer than two ophys frames. It does not explicitly catch session-level load failures in the final code.

ii. 
```python
for col in ['aborted', 'auto_rewarded', 'auto_rewarded_trial', 'is_auto_rewarded']:
    if col in df.columns:
        df = df[~df[col].fillna(False)]
...
stop = float(tr['stop_time']) if 'stop_time' in tr.index else float(tr['end_time'])
...
if eye_tracking is None or len(eye_tracking) == 0:
    return None, None, None
...
if valid.sum() >= 2:
    ...
else:
    pupil_bins = np.zeros(len(trial_ts), dtype=np.int64)
...
if idx.size < 2:
    continue
```

iii. The notes and trajectory show a repeated concern with handling the “local subset” and schema variation. The code reflects that through multiple fallbacks, but the final script no longer has the broader `try/except` session-skip behavior that earlier trajectory versions discussed.

## 9-a. What are the most time-consuming steps of the code?

i. The expensive steps are repeated NWB loading with `load_experiment`, building `BehaviorOphysExperiment` objects, and then the per-session/per-trial output construction in `build_session`.

ii. 
```python
for p in probe_files:
    exp = load_experiment(p)
    ...

for i, p in enumerate(files, 1):
    exp = load_experiment(p)
    sess = build_session(exp, image_to_idx, run_edges, pupil_edges, outcome_to_idx)
```

iii. The notes discuss runtime and the code contains an explicit optimization comment about avoiding loading all NWB files twice, which implies that file I/O and session construction were identified as the bottleneck.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The main non-vectorized loops are the session loop, the per-trial loop inside each session, and the nested loop over overlapping stimulus rows inside each trial. The repeated boolean-mask construction for every stimulus interval is especially vectorizable.

ii. 
```python
for i, p in enumerate(files, 1):
    ...

for _, tr in trials.iterrows():
    ...
    for _, srow in stim_trial.iterrows():
        smask = (trial_ts >= s0) & (trial_ts < s1)
```

iii. No explicit justification is given for leaving these loops as-is. The only explicit performance decision the AI documented was subset-based quantile estimation.

## 9-c. What processing does the code repeat multiple times?

i. The AI repeats data loading and some per-session work. It loads a probe subset in `collect_global_info`, then loads every file again for full conversion, and then re-loads the first file a third time just to compute `time_bin_size`. It also repeatedly filters `stim` per trial.

ii. 
```python
for p in probe_files:
    exp = load_experiment(p)
    ...

for i, p in enumerate(files, 1):
    exp = load_experiment(p)
    ...

dt_ms = float(np.median(np.diff(load_experiment(files[0]).ophys_timestamps)) * 1000.0)
...
stim_trial = stim[(stim['start_time'] < stop) & (stim['_end_time'] > start)]
```

iii. The subset pass is explicitly justified by an efficiency comment, but it still repeats loading work relative to a single-pass design. The reload of the first file for `dt_ms` has no separate justification.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several values are computed but not used downstream: `stim_start`, `pupil_col`, `image_names_seen`, `trial_outcomes_seen`, and `neural_source` are collected or returned but never affect the saved outputs. The command-line flag `--show-processing` is parsed but unused.

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

iii. No explicit justification is documented. These look like remnants of intermediate debugging or bookkeeping rather than parts of the final converted representation.
