# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all locally available `.nwb` files from `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/`, opens each file with `pynwb`, and constructs an AllenSDK `BehaviorOphysExperiment` from the NWB object. It does not use `VisualBehaviorOphysProjectCache` or the experiment table as the primary loader.

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

iii. In `CONVERSION_NOTES.md`, the agent explicitly justified converting "only locally available NWB experiment files" because the metadata tables described a larger release than the files actually present locally. The trajectory also shows an early decision to avoid fetching absent experiments and to operate directly on the local subset.

## 1-b. How are the data split into subjects?

i. Subjects are split using `mouse_id` read from each experiment's metadata. After all sessions are processed, unique mouse IDs are collected into a sorted subject list, and each session is assigned an index into that list.

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

iii. The notes describe mapping `metadata mouse_id` to `subjects / subject_idx`, and the trajectory shows the agent treating `mouse_id` as the canonical subject identifier exposed by AllenSDK metadata.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file is treated as one session. The code does not group multiple experiments by `ophys_session_id`; instead, each loaded experiment object becomes one output session entry.

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

iii. `CONVERSION_NOTES.md` says the raw data are "one file per available ophys experiment" and that conversion should "operate on the locally available NWB files only." The agent therefore operationalized "session" as one locally available NWB experiment file.

## 1-d. How are the data split into trials?

i. Trials are split using the AllenSDK `exp.trials` table. For each kept trial, the code uses `start_time` and `stop_time` (or `end_time` if needed) to select the ophys timestamps within that interval and slices all per-trial signals with that mask.

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

iii. The notes state "Trial segmentation: Use experiment-defined trial intervals from the trial table" and the trajectory repeatedly describes segmentation by experiment-defined trials while aligning everything to `ophys_timestamps`.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by excluding rows marked as aborted or auto-rewarded across several possible column names, then keeping only go/catch trials when the relevant columns are present. Trials with fewer than 2 ophys frames are skipped. The code does not explicitly require non-null `change_time`, and it does not explicitly drop sessions with fewer than 2 valid trials.

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
idx = np.flatnonzero((ts >= start) & (ts < stop))
if idx.size < 2:
    continue
```

iii. The notes say "include Go and Catch trials, exclude Aborted and Auto-rewarded trials." The trajectory also shows the agent explicitly reconciling the task instruction with the trial table fields and choosing a tolerant implementation that handles several possible auto-reward column names.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural data are derived from `exp.dff_traces`, specifically the `dff` column of the DataFrame returned by AllenSDK.

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

iii. The notes document that the agent initially preferred detected events to match the paper, but the trajectory shows that after debugging sparsity and poor decoder behavior it switched to dF/F traces and recorded that switch in the notes.

## 2-b. How is the `neural` data processed?

i. The code converts each cell's dF/F trace to `float32`, stacks cells into a neuron-by-time matrix, and then slices that session-level matrix into per-trial matrices using the trial timestamp mask. No extra normalization, denoising, or temporal rebinning is applied.

ii.
```python
traces = [np.asarray(v, dtype=np.float32) for v in dff['dff'].values]
mat = np.stack(traces, axis=0)
...
trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
session_neural.append(trial_neural)
```

iii. The notes describe the implemented script as using AllenSDK `BehaviorOphysExperiment` with ophys-timestamp alignment and per-trial slicing, and the trajectory shows the switch from events to dF/F was motivated by sample verification and decoder performance.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neural quality filtering is applied in the conversion script. All cells present in `exp.dff_traces` are kept.

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

iii. In the notes, neuron-quality rules were left unresolved and no later step added an explicit filter. The trajectory likewise contains no later neuron curation pass beyond choosing dF/F instead of events.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned to the ophys time base by selecting frames whose `ophys_timestamps` fall within each trial's start and stop times. The code keeps the original ophys frame positions rather than re-centering time around a within-trial event.

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

iii. The notes repeatedly state that `ophys_timestamps` are the master clock and that all streams should be aligned to the ophys time base, which is the alignment strategy the code implements.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native ophys frame spacing, estimated as the median difference between consecutive `ophys_timestamps` in the first file. No temporal rebinning is applied.

ii.
```python
dt_ms = float(np.median(np.diff(load_experiment(files[0]).ophys_timestamps)) * 1000.0)
...
'time_bin_size': dt_ms,
```

iii. The notes say the agent would use `ophys_timestamps` as the master clock. The trajectory shows the agent debugging representation issues without adding any extra temporal aggregation step.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `exp.stimulus_presentations`, especially the `image_name` and `start_time` columns. The code defines each stimulus interval as `start_time + 0.75` seconds and uses that table rather than the trials table's `initial_image_name` / `change_image_name`.

ii.
```python
stim = exp.stimulus_presentations.copy().sort_values('start_time')
stim_start = np.asarray(stim['start_time'], dtype=float)
stim_end = stim_start + 0.75
stim = stim.copy()
stim['_end_time'] = stim_end
...
name = str(srow['image_name']) if 'image_name' in srow.index else 'blank'
image_series[smask] = image_to_idx.get(name, image_to_idx['blank'])
```

iii. The notes explicitly planned to derive image identity from `stimulus_presentations.image_name` over image-presentation intervals, and the trajectory shows the agent changing to 750 ms image intervals after discovering that the 250 ms flash duration alone caused poor image decoding.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The code builds a global image vocabulary from a probe subset of up to 8 files, adds `blank` and `omitted`, maps names to integer codes, initializes each trial as `blank`, and then fills in codes for overlapping 750 ms stimulus intervals inside the trial.

ii.
```python
def collect_global_info(files):
    image_names = set(['blank', 'omitted'])
    ...
    probe_files = files[:min(8, len(files))]
    for p in probe_files:
        exp = load_experiment(p)
        stim = exp.stimulus_presentations.copy()
        if 'image_name' in stim.columns:
            names = stim['image_name'].dropna().astype(str).unique().tolist()
            image_names.update(names)
    ...
    image_values = sorted(image_names)
    image_to_idx = {name: i for i, name in enumerate(image_values)}

image_series = np.full(len(trial_ts), image_to_idx['blank'], dtype=np.int64)
...
image_series[smask] = image_to_idx.get(name, image_to_idx['blank'])
```

iii. The notes justify a "global" categorical mapping and the trajectory shows an explicit speed optimization: only probe a small subset up front rather than loading every NWB file twice. The same debugging episode motivated the use of 750 ms intervals and the `blank` background class.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is aligned frame-by-frame on the same per-trial `trial_ts` array used to slice neural activity. For each stimulus interval overlapping a trial, the code marks the ophys frames whose timestamps fall inside that interval.

ii.
```python
idx = np.flatnonzero((ts >= start) & (ts < stop))
trial_ts = ts[idx]
...
stim_trial = stim[(stim['start_time'] < stop) & (stim['_end_time'] > start)]
for _, srow in stim_trial.iterrows():
    s0 = max(start, float(srow['start_time']))
    s1 = min(stop, float(srow['_end_time']))
    smask = (trial_ts >= s0) & (trial_ts < s1)
    image_series[smask] = image_to_idx.get(name, image_to_idx['blank'])
```

iii. The notes say all time-varying streams should be aligned to `ophys_timestamps`, and the trajectory shows the agent debugging the image labeling specifically by checking how stimulus intervals overlapped the ophys frame grid.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from changes in consecutive `stimulus_presentations.image_name` values within a trial, not from `trials.change_time`. The code compares each stimulus name to the immediately previous stimulus name seen in that trial.

ii.
```python
change_series = np.zeros(len(trial_ts), dtype=np.int64)
stim_trial = stim[(stim['start_time'] < stop) & (stim['_end_time'] > start)]
prev_name = None
for _, srow in stim_trial.iterrows():
    ...
    name = str(srow['image_name']) if 'image_name' in srow.index else 'blank'
    ...
    if prev_name is not None and name != prev_name:
        hit = np.flatnonzero(smask)
        if hit.size:
            change_series[hit[0]] = 1
    prev_name = name
```

iii. The notes planned to derive image change from image identity transitions, and the trajectory shows the agent pivoting toward stimulus-interval labeling after inspecting the stimulus table and the paper's 750 ms interval description.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code starts with an all-zero vector and sets a `1` at the first ophys frame of a stimulus interval whenever that interval's image name differs from the previous interval's image name within the same trial.

ii.
```python
change_series = np.zeros(len(trial_ts), dtype=np.int64)
...
if prev_name is not None and name != prev_name:
    hit = np.flatnonzero(smask)
    if hit.size:
        change_series[hit[0]] = 1
```

iii. The notes frame image change as a time-varying binary output derived from image transitions. The trajectory indicates the agent wanted an event-like signal tied to the start of a changed image interval.

## 4-c. How is `output` *Image change* thresholded into categories?

i. There is no numeric thresholding step. The output is directly converted into two categories, `no_change` and `change`, by keeping the vector binary: 0 everywhere except the marked frame(s) at the start of changed stimulus intervals.

ii.
```python
change_series = np.zeros(len(trial_ts), dtype=np.int64)
...
change_series[hit[0]] = 1
...
'output_values': [image_values, ['no_change', 'change'], [f'bin_{i}' for i in range(5)], [f'bin_{i}' for i in range(5)], TRIAL_OUTCOME_VALUES],
```

iii. The notes explicitly planned a binary time-varying image-change signal. The trajectory does not show any later threshold refinement beyond this binary event coding.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is aligned on the same per-trial ophys timestamps as the neural data. The code computes `smask` on `trial_ts`, then writes change labels into the matching frames of `change_series`.

ii.
```python
trial_ts = ts[idx]
trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
...
smask = (trial_ts >= s0) & (trial_ts < s1)
...
change_series[hit[0]] = 1
```

iii. The notes say all outputs should be aligned on the ophys clock, and the trajectory's image-label debugging centered on fixing the per-frame alignment against `trial_ts`.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `exp.running_speed`, using one column for timestamps and one for speed. The code will fall back to the first two DataFrame columns if the standard names are absent.

ii.
```python
def get_running_series(run_df):
    cols = list(run_df.columns)
    tcol = 'timestamps' if 'timestamps' in cols else cols[0]
    vcol = 'speed' if 'speed' in cols else cols[1]
    return np.asarray(run_df[tcol], dtype=float), np.asarray(run_df[vcol], dtype=float)
```

iii. The notes map `running_speed` directly to a time-varying decoder output and the trajectory shows the agent implementing defensive column selection to tolerate AllenSDK table differences.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The code gathers running values from up to 8 probe files to estimate 5-bin quantile edges, then linearly interpolates the running speed onto each trial's ophys timestamps with `np.interp` and discretizes each frame into those bins.

ii.
```python
probe_files = files[:min(8, len(files))]
...
rt, rv = get_running_series(exp.running_speed.copy())
run_vals.append(rv[np.isfinite(rv)])
...
run_edges = np.quantile(run_all, [0, .2, .4, .6, .8, 1])
...
run_interp = np.interp(trial_ts, run_t, run_v).astype(np.float32)
run_bins = np.digitize(run_interp, run_edges[1:-1], right=False).astype(np.int64)
```

iii. In the notes, the agent planned percentile binning over pooled valid samples. The trajectory later records an explicit speed optimization: estimate bin edges from a small subset rather than loading all sessions twice.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is thresholded into five quantile bins using edges at 0%, 20%, 40%, 60%, 80%, and 100% of the pooled probe-session distribution.

ii.
```python
run_edges = np.quantile(run_all, [0, .2, .4, .6, .8, 1])
...
run_bins = np.digitize(run_interp, run_edges[1:-1], right=False).astype(np.int64)
...
[f'bin_{i}' for i in range(5)]
```

iii. The notes explicitly say running speed should be "discretize[d] into 5 equal-percentile bins." The only extra justification in the trajectory is runtime: estimate those percentiles from a smaller subset to reduce up-front cost.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned directly onto the per-trial ophys timestamps with linear interpolation. The neural data and running labels then use the same frame mask for each trial.

ii.
```python
idx = np.flatnonzero((ts >= start) & (ts < stop))
trial_ts = ts[idx]
trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
...
run_interp = np.interp(trial_ts, run_t, run_v).astype(np.float32)
run_bins = np.digitize(run_interp, run_edges[1:-1], right=False).astype(np.int64)
```

iii. The notes identify `ophys_timestamps` as the master clock and specifically map `running_speed` to an ophys-aligned time-varying output.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `exp.eye_tracking`, but the code does not hard-code one field. It chooses the first available column among `pupil_diameter`, `pupil_area`, `pupil_width`, `pupil_height`, `pupil_radius`, or `pupil_size`, or another numeric pupil-like column if needed.

ii.
```python
def pick_pupil_series(eye_tracking):
    ...
    candidate_cols = ['pupil_diameter', 'pupil_area', 'pupil_width', 'pupil_height', 'pupil_radius', 'pupil_size']
    value_col = next((c for c in candidate_cols if c in cols), None)
    if value_col is None:
        value_col = next((c for c in cols if 'pupil' in c.lower() and np.issubdtype(eye_tracking[c].dtype, np.number)), None)
    ...
    return np.asarray(eye_tracking[time_col], dtype=float), np.asarray(eye_tracking[value_col], dtype=float), value_col
```

iii. The notes planned to use "pupil diameter / eye tracking pupil area-equivalent" and the trajectory shows this was a deliberate robustness choice because the agent was unsure which pupil column would be consistently available.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code collects finite pupil samples from up to 8 probe files to estimate quantile edges, then linearly interpolates the selected pupil signal onto each trial's ophys timestamps. If too few valid pupil samples are available, the trial is filled with zeros. No blink filtering is applied.

ii.
```python
probe_files = files[:min(8, len(files))]
...
eye = exp.eye_tracking.copy()
pt, pv, _ = pick_pupil_series(eye)
if pv is not None:
    pupil_vals.append(pv[np.isfinite(pv)])
...
pupil_edges = np.quantile(pupil_all, [0, .2, .4, .6, .8, 1])
...
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

iii. The notes planned alignment to `ophys_timestamps` and 5-bin percentile discretization over pooled valid samples. The trajectory shows the same subset-based speed optimization used for running speed.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Pupil values are thresholded into five quantile bins using pooled finite values from the probe subset and `np.digitize`.

ii.
```python
pupil_edges = np.quantile(pupil_all, [0, .2, .4, .6, .8, 1])
...
pupil_bins = np.digitize(pupil_interp, pupil_edges[1:-1], right=False).astype(np.int64)
...
[f'bin_{i}' for i in range(5)]
```

iii. The notes say pupil should be discretized into "5 equal-percentile bins." The trajectory indicates the agent kept that strategy but approximated the pooled distribution from a small subset for speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil data are aligned to neural data by interpolating the selected pupil signal onto each trial's ophys timestamps and then writing the discretized values into the same trial-length frame grid as the neural data.

ii.
```python
trial_ts = ts[idx]
trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
...
pupil_interp = np.interp(trial_ts, pupil_t[valid], pupil_v[valid]).astype(np.float32)
pupil_bins = np.digitize(pupil_interp, pupil_edges[1:-1], right=False).astype(np.int64)
```

iii. The notes consistently treat `ophys_timestamps` as the common alignment clock for all behavioral and neural signals.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the trial-table boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject`. If none are true, the code assigns `other`.

ii.
```python
TRIAL_OUTCOME_VALUES = ['hit', 'miss', 'false_alarm', 'correct_reject', 'other']

def infer_trial_outcome(row):
    for key in ['hit', 'miss', 'false_alarm', 'correct_reject']:
        if key in row.index and bool(row[key]):
            return key
    return 'other'
```

iii. The notes planned to use the trial outcome fields from the trials table and the trajectory shows the fallback `other` category was kept as a defensive edge-case handler.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The inferred outcome label is mapped to an integer code using `outcome_to_idx`, and that code is repeated across every ophys frame in the trial so that outcome is represented as a static per-trial output in time-series form.

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

iii. The notes say trial outcome should be "static per trial," and the trajectory shows the agent encoding that by broadcasting the per-trial category over the trial's full time axis.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles missing or inconsistent fields defensively. It accepts multiple possible trial and auto-reward column names, falls back to generic running/pupil column choices, fills missing or unusable pupil traces with zeros, defaults unknown image names to `blank`, and skips trials with fewer than 2 frames. It does not use a try/except to skip failed sessions, and `np.interp` will extrapolate endpoint values rather than producing NaNs.

ii.
```python
for col in ['aborted', 'auto_rewarded', 'auto_rewarded_trial', 'is_auto_rewarded']:
    if col in df.columns:
        df = df[~df[col].fillna(False)]
...
tcol = 'timestamps' if 'timestamps' in cols else cols[0]
vcol = 'speed' if 'speed' in cols else cols[1]
...
value_col = next((c for c in candidate_cols if c in cols), None)
...
image_series[smask] = image_to_idx.get(name, image_to_idx['blank'])
...
if valid.sum() >= 2:
    pupil_interp = np.interp(...)
else:
    pupil_bins = np.zeros(len(trial_ts), dtype=np.int64)
...
if idx.size < 2:
    continue
```

iii. The notes and trajectory both show the agent emphasizing robustness to dataset/schema variation. The strongest explicit justification in the trajectory is that the local data and metadata were not perfectly matched, so the agent favored tolerant loaders and fallbacks.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming work is repeatedly opening NWB files and constructing `BehaviorOphysExperiment` objects, then iterating through all trials and stimulus intervals per session. The code also incurs a large final serialization cost when writing the full pickle.

ii.
```python
for p in probe_files:
    exp = load_experiment(p)
...
for i, p in enumerate(files, 1):
    exp = load_experiment(p)
    sess = build_session(exp, image_to_idx, run_edges, pupil_edges, outcome_to_idx)
...
with open(args.outpicklefile, 'wb') as f:
    pickle.dump(data, f)
```

iii. The trajectory explicitly discusses startup cost from `collect_global_info(files)` and later records a 14 GB output file and a long save/finalization step, so the agent clearly understood I/O and session loading as the main runtime bottlenecks.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The code contains several Python-level loops that could have been vectorized or pre-indexed: the per-trial loop over `trials.iterrows()`, the nested loop over overlapping stimulus presentations inside each trial, and the probe-file loop used to estimate global bin edges.

ii.
```python
for _, tr in trials.iterrows():
    ...
    for _, srow in stim_trial.iterrows():
        ...

for p in probe_files:
    exp = load_experiment(p)
```

iii. The notes leave the "Code inefficiencies identified" section blank, but the trajectory repeatedly focuses on runtime, especially the up-front probe pass and the session-by-session loop. That provides indirect evidence that the agent knew these loops were the places where efficiency was being traded against simplicity.

## 9-c. What processing does the code repeat multiple times?

i. The code repeats session loading at least twice for some files: once in `collect_global_info()` for up to 8 probe files, again in the main conversion loop, and once more for the first file when estimating `time_bin_size`. It also repeatedly filters the stimulus table per trial.

ii.
```python
probe_files = files[:min(8, len(files))]
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

iii. The trajectory directly mentions this as a speed issue: the agent changed `collect_global_info` so it would not load every session twice, but it still accepted a smaller amount of repeated loading as a runtime compromise.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores several pieces of information that are not used in the final saved dataset: `image_names_seen`, `trial_outcomes_seen`, `neural_source`, and `cell_ids` inside each session dict; the chosen `pupil_col`; and the `--show-processing` argument, which is parsed but never used. It also defaults unknown images to `blank`, which can silently discard information.

ii.
```python
ap.add_argument('--show-processing', action='store_true', help='Save processing plots for up to 2 sessions')
...
pupil_t, pupil_v, pupil_col = pick_pupil_series(eye)
...
image_names_seen = []
trial_outcomes_seen = []
...
return {
    'neural': session_neural,
    'input': session_input,
    'output': session_output,
    'cell_ids': cell_ids,
    'neural_source': neural_source,
    'image_names_seen': sorted(set(image_names_seen)),
    'trial_outcomes_seen': sorted(set(trial_outcomes_seen)),
    'targeted_structure': targeted_structure,
    'mouse_id': mouse_id,
}
```

iii. There is no explicit written justification for these extra computations. The trajectory suggests they were mostly debugging or bookkeeping conveniences added while iterating on the conversion.
