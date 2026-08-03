# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script assumes all source data are local NWB files under a hard-coded dataset root, enumerates every `*.nwb` file in sorted order, and loads each file into an AllenSDK `BehaviorOphysExperiment` with `NWBHDF5IO` and `BehaviorOphysExperiment.from_nwb`. In `--sample` mode it truncates to the first two files.

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

iii. `CONVERSION_NOTES.md` Step 2 says the raw data are NWB files, one per ophys experiment, and Step 5 says the conversion should use only the locally available NWB sessions. The trajectory shows the agent explicitly resolved a metadata-versus-local-subset discrepancy by deciding to convert the local NWB files only.

## 1-b. How are the data split into subjects?

i. Subjects are identified per session from `exp.metadata['mouse_id']`. After all sessions are built, the script creates a sorted unique subject list and a per-session `subject_idx` array.

ii.
```python
meta = exp.metadata if isinstance(exp.metadata, dict) else {}
mouse_id = str(meta.get('mouse_id', 'unknown'))

subjects = sorted({s['mouse_id'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.array(
    [subject_to_idx[s['mouse_id']] for s in sessions], dtype=np.int64
),
```

iii. `CONVERSION_NOTES.md` Step 5 maps metadata `mouse_id` to `subjects / subject_idx`. The trajectory does not show a competing subject-splitting scheme.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The script iterates over the sorted NWB files, builds one session dictionary per file, and appends it to `sessions`.

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

iii. `CONVERSION_NOTES.md` Step 2 says the main data are one NWB file per ophys experiment, and Step 5 says session order should follow the converted NWB session list.

## 1-d. How are the data split into trials?

i. Within each session, the script uses the AllenSDK `exp.trials` table, filters it with `valid_trials_table`, then slices all time-varying streams between each trial’s `start_time` and `stop_time` on the ophys timestamp axis. Trials with fewer than 2 ophys samples are dropped.

ii.
```python
trials = valid_trials_table(exp.trials)

for _, tr in trials.iterrows():
    start = float(tr['start_time'])
    stop = float(tr['stop_time']) if 'stop_time' in tr.index else float(tr['end_time'])
    idx = np.flatnonzero((ts >= start) & (ts < stop))
    if idx.size < 2:
        continue
    trial_ts = ts[idx]
    trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
```

iii. The task instructions explicitly require segmentation into experiment-defined trials. `CONVERSION_NOTES.md` Step 5 records the same decision: use the trial table and segment on the common ophys clock.

## 1-e. How are trials filtered based on quality controls?

i. The script excludes aborted and auto-rewarded trials, then keeps only Go or Catch trials. It also discards any kept trial that contains fewer than 2 ophys timestamps.

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

iii. This follows the task instructions exactly. `CONVERSION_NOTES.md` Step 3 and Step 5 both state “include Go and Catch, exclude Aborted and Auto-rewarded.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The delivered script derives `neural` only from `exp.dff_traces['dff']`. It does not use the SDK `events` table.

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

iii. The trajectory shows this was an intentional late-stage change. In Step 29 the agent says it would “switch the neural representation to dF/F for the decoder task,” even though Step 3 of `CONVERSION_NOTES.md` had already noted that the paper used detected calcium events.

## 2-b. How is the `neural` data processed?

i. The dF/F traces are copied from the AllenSDK table, cast to `float32`, stacked into a neuron-by-time matrix, and then trial-sliced by ophys timestamp. There is no event detection, deconvolution, temporal smoothing, or rebinning in `convert_data.py`.

ii.
```python
traces = [np.asarray(v, dtype=np.float32) for v in dff['dff'].values]
mat = np.stack(traces, axis=0)
...
trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
```

iii. Step 29 of the trajectory justifies this as a pragmatic decoder choice: event traces were “extremely sparse,” dF/F was denser, and sample decoder accuracy improved after switching.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script does not implement its own neuron QC beyond what AllenSDK does when constructing `BehaviorOphysExperiment.from_nwb`. Because it calls the SDK with default arguments, ROI filtering is delegated to AllenSDK’s default `exclude_invalid_rois=True` behavior.

ii.
```python
def load_experiment(nwb_path):
    io = NWBHDF5IO(str(nwb_path), 'r', load_namespaces=True)
    nwbfile = io.read()
    exp = BehaviorOphysExperiment.from_nwb(nwbfile)
    exp._nwb_io = io
    return exp
```

iii. `CONVERSION_NOTES.md` Step 1 focuses on AllenSDK accessors rather than custom QC code, and the trajectory never adds any explicit neuron filter. The QC choice was effectively “use SDK defaults.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to the ophys clock, not to a separate discrete event such as change onset. For each trial, the script selects all ophys frames whose timestamps fall between the trial’s `start_time` and `stop_time`.

ii.
```python
ts = np.asarray(exp.ophys_timestamps, dtype=float)
...
idx = np.flatnonzero((ts >= start) & (ts < stop))
trial_ts = ts[idx]
trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
```

iii. `CONVERSION_NOTES.md` Step 5 says “use `ophys_timestamps` as the master clock,” which mirrors the instruction “Temporally align based on ophys timestamp.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use the native ophys frame spacing. The script estimates one global `time_bin_size` from the median timestamp difference of the first session and does not rebin the per-trial matrices.

ii.
```python
dt_ms = float(np.median(np.diff(load_experiment(files[0]).ophys_timestamps)) * 1000.0)
...
'metadata': {
    'time_bin_size': dt_ms,
```

iii. The trajectory and notes consistently describe alignment on the native ophys timestamps. No rebinning step was added anywhere in the code.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the session’s `stimulus_presentations` table, specifically the `image_name` column. The script also injects special `blank` and `omitted` categories into the global label vocabulary.

ii.
```python
image_names = set(['blank', 'omitted'])
...
stim = exp.stimulus_presentations.copy().sort_values('start_time')
...
name = str(srow['image_name']) if 'image_name' in srow.index else 'blank'
image_series[smask] = image_to_idx.get(name, image_to_idx['blank'])
```

iii. `CONVERSION_NOTES.md` Step 5 maps `stimulus_presentations.image_name` to `output[0] image identity` and notes that a background/blank class might be needed for full time-series consistency.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The script creates a global categorical image vocabulary, initializes each trial to `blank`, then overwrites time bins using overlapping stimulus rows. Crucially, it ignores the NWB `end_time` and instead defines every image’s effective interval as `start_time + 0.75` seconds.

ii.
```python
image_values = sorted(image_names)
image_to_idx = {name: i for i, name in enumerate(image_values)}
...
stim = exp.stimulus_presentations.copy().sort_values('start_time')
stim_start = np.asarray(stim['start_time'], dtype=float)
stim_end = stim_start + 0.75
stim = stim.copy()
stim['_end_time'] = stim_end
...
image_series = np.full(len(trial_ts), image_to_idx['blank'], dtype=np.int64)
```

iii. The trajectory shows the reason clearly. In Step 29 the agent says the raw `stimulus_presentations` flashes last about 0.25 s, causing `blank` to dominate, so it expanded image labels to 750 ms “to better match the methods.”

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image labels are aligned directly onto each trial’s ophys timestamps. For every stimulus row overlapping a trial, the script computes a boolean mask on `trial_ts` and writes the corresponding image category into those ophys bins.

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

iii. `CONVERSION_NOTES.md` Step 5 says all time-varying outputs should be aligned to the common ophys clock, and the trajectory repeats that plan.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The delivered script does not use the raw `is_change` field from `stimulus_presentations` or `trials`. Instead, it derives image change from successive `image_name` values encountered within each trial.

ii.
```python
change_series = np.zeros(len(trial_ts), dtype=np.int64)
...
name = str(srow['image_name']) if 'image_name' in srow.index else 'blank'
...
if prev_name is not None and name != prev_name:
    hit = np.flatnonzero(smask)
    if hit.size:
        change_series[hit[0]] = 1
prev_name = name
```

iii. `CONVERSION_NOTES.md` Step 5 proposed deriving image change from image-identity transitions. The trajectory does not show the agent reconsidering this after discovering that the SDK already exposes `is_change`.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Image change is initialized as all zeros. While iterating over trial-overlapping stimulus intervals, the script compares each current image name to the previous image name and sets exactly the first ophys bin of the changed interval to 1.

ii.
```python
change_series = np.zeros(len(trial_ts), dtype=np.int64)
...
if prev_name is not None and name != prev_name:
    hit = np.flatnonzero(smask)
    if hit.size:
        change_series[hit[0]] = 1
```

iii. The trajectory frames this as “preserving change markers” after stretching stimulus intervals to 750 ms. No separate change-detection logic was added.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No continuous threshold is used. The script emits a binary category: `0` for `no_change`, `1` for `change`.

ii.
```python
change_series = np.zeros(len(trial_ts), dtype=np.int64)
...
'output_values': [
    image_values,
    ['no_change', 'change'],
    [f'bin_{i}' for i in range(5)],
```

iii. This follows the task instruction that image change should be binary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change labels live on the same per-trial ophys timestamp grid as the neural data. The script sets the change flag at the first ophys sample whose timestamp falls inside the changed image interval.

ii.
```python
trial_ts = ts[idx]
...
smask = (trial_ts >= s0) & (trial_ts < s1)
...
hit = np.flatnonzero(smask)
if hit.size:
    change_series[hit[0]] = 1
```

iii. This is the same alignment mechanism used for image identity and the other time-varying outputs.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is taken from `exp.running_speed`, specifically its timestamp and speed columns.

ii.
```python
def get_running_series(run_df):
    cols = list(run_df.columns)
    tcol = 'timestamps' if 'timestamps' in cols else cols[0]
    vcol = 'speed' if 'speed' in cols else cols[1]
    return np.asarray(run_df[tcol], dtype=float), np.asarray(run_df[vcol], dtype=float)
```

iii. `CONVERSION_NOTES.md` Step 1 identifies `BehaviorOphysExperiment.running_speed` as the relevant aligned time series, and Step 5 maps it to the running-speed output.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The script pools finite running-speed values from only the first 8 sessions, computes global quintile edges, then linearly interpolates running speed onto each trial’s ophys timestamps and bins the interpolated values with those edges.

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

iii. The comment in `collect_global_info` calls this an “Efficiency optimization” to avoid loading all NWB files twice. `CONVERSION_NOTES.md` Step 5 had originally planned to compute percentile bins over pooled valid samples, but the final code uses only a subset.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins using empirical 0/20/40/60/80/100% quantiles estimated from the subset collected in `collect_global_info`, then assigned with `np.digitize` to categories `bin_0` through `bin_4`.

ii.
```python
run_edges = np.quantile(run_all, [0, .2, .4, .6, .8, 1])
...
run_bins = np.digitize(run_interp, run_edges[1:-1], right=False).astype(np.int64)
...
[f'bin_{i}' for i in range(5)]
```

iii. The agent’s notes describe the intended rule as “five equal-percentile bins.” The delivered implementation approximates that rule from the first 8 sessions for speed.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by interpolation from running timestamps onto each trial’s ophys timestamps, so every ophys bin gets one running-speed bin label.

ii.
```python
run_t, run_v = get_running_series(exp.running_speed.copy())
...
run_interp = np.interp(trial_ts, run_t, run_v).astype(np.float32)
run_bins = np.digitize(run_interp, run_edges[1:-1], right=False).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly planned “align to ophys timestamps” for running speed, and the code implements exactly that.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The script searches the eye-tracking table for a pupil-like numeric column. In practice, because the AllenSDK table exposes `pupil_area`, `pupil_width`, and `pupil_height` but not `pupil_diameter`, the script usually uses `pupil_area`.

ii.
```python
candidate_cols = [
    'pupil_diameter', 'pupil_area', 'pupil_width',
    'pupil_height', 'pupil_radius', 'pupil_size'
]
value_col = next((c for c in candidate_cols if c in cols), None)
...
return np.asarray(eye_tracking[time_col], dtype=float), \
       np.asarray(eye_tracking[value_col], dtype=float), value_col
```

iii. `CONVERSION_NOTES.md` Step 5 already softened this from “diameter” to “pupil diameter / eye tracking pupil area-equivalent,” showing the agent knew the raw field might not actually be diameter.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The chosen pupil-like series is not converted to diameter. The script simply interpolates the chosen numeric series onto the trial ophys timestamps and then bins it. Missing or non-finite samples are ignored when interpolation is possible.

ii.
```python
pupil_t, pupil_v, pupil_col = pick_pupil_series(eye)
...
valid = np.isfinite(pupil_t) & np.isfinite(pupil_v)
if valid.sum() >= 2:
    pupil_interp = np.interp(trial_ts, pupil_t[valid], pupil_v[valid]).astype(np.float32)
    pupil_bins = np.digitize(pupil_interp, pupil_edges[1:-1], right=False).astype(np.int64)
```

iii. The trajectory never shows a diameter reconstruction step. The justification is purely robustness: choose whichever pupil-like numeric field exists and proceed.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The script pools finite values of the chosen pupil-like series from the first 8 sessions, computes 5 quantile bins, and assigns each interpolated sample to `bin_0` through `bin_4`.

ii.
```python
pupil_all = np.concatenate(pupil_vals) if pupil_vals else np.array([0.0])
pupil_edges = np.quantile(pupil_all, [0, .2, .4, .6, .8, 1])
...
pupil_bins = np.digitize(pupil_interp, pupil_edges[1:-1], right=False).astype(np.int64)
...
[f'bin_{i}' for i in range(5)]
```

iii. This mirrors the running-speed discretization. The notes say the target should be five equal-percentile bins, but the final code estimates the thresholds from a subset and from a proxy field.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil values are aligned by interpolation from eye-tracking timestamps onto the per-trial ophys timestamps. If there are fewer than 2 finite pupil samples, the whole trial is filled with zeros.

ii.
```python
valid = np.isfinite(pupil_t) & np.isfinite(pupil_v)
if valid.sum() >= 2:
    pupil_interp = np.interp(trial_ts, pupil_t[valid], pupil_v[valid]).astype(np.float32)
    pupil_bins = np.digitize(pupil_interp, pupil_edges[1:-1], right=False).astype(np.int64)
else:
    pupil_bins = np.zeros(len(trial_ts), dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` Step 5 planned the same alignment strategy: align pupil data to ophys timestamps, then discretize.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
def infer_trial_outcome(row):
    for key in ['hit', 'miss', 'false_alarm', 'correct_reject']:
        if key in row.index and bool(row[key]):
            return key
    return 'other'
```

iii. `CONVERSION_NOTES.md` Step 5 says trial outcome should come from the trial table’s outcome fields, and the code follows that plan.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. For each filtered trial, the script picks the first true outcome among `hit`, `miss`, `false_alarm`, and `correct_reject`, maps it to a category index, and repeats that same category across all time bins of the trial.

ii.
```python
outcome = infer_trial_outcome(tr)
outcome_idx = outcome_to_idx.get(outcome, outcome_to_idx['other'])
outcome_series = np.full(len(trial_ts), outcome_idx, dtype=np.int64)
...
output = np.vstack([image_series, change_series, run_bins, pupil_bins, outcome_series])
```

iii. `CONVERSION_NOTES.md` Step 5 says trial outcome should be “static per trial,” and the trajectory never deviates from that.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles missingness with fallbacks rather than explicit masking. Missing pupil streams return `None` and become all-zero bins; too-few ophys bins cause the trial to be skipped; unknown image names fall back to `blank`; unknown outcomes fall back to `other`; flexible column-name checks are used for trial and pupil fields.

ii.
```python
if eye_tracking is None or len(eye_tracking) == 0:
    return None, None, None
...
if idx.size < 2:
    continue
...
name = str(srow['image_name']) if 'image_name' in srow.index else 'blank'
...
else:
    pupil_bins = np.zeros(len(trial_ts), dtype=np.int64)
...
return 'other'
```

iii. The trajectory emphasizes robustness and decoder pass-through rather than preserving missingness as a separate state. The main explicit justification is “handle missing data carefully” from the planning notes; the final implementation chooses zero-filled fallbacks.

## 9-a. What are the most time-consuming steps of the code?

i. The main runtime cost is repeatedly opening NWB files and constructing `BehaviorOphysExperiment` objects, followed by per-trial stimulus iteration and finally serializing the huge pickle. The code also does an extra pass over up to 8 sessions in `collect_global_info`.

ii.
```python
for p in probe_files:
    exp = load_experiment(p)
    ...

for i, p in enumerate(files, 1):
    exp = load_experiment(p)
    sess = build_session(exp, image_to_idx, run_edges, pupil_edges, outcome_to_idx)
    sessions.append(sess)

with open(args.outpicklefile, 'wb') as f:
    pickle.dump(data, f)
```

iii. `conversion_full_out.txt` and the trajectory show the conversion spending most of its time session-by-session in NWB loading and session building, then another long write at the end (`saved converted_data.pkl in 1887.28s`).

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The biggest vectorization opportunities are the nested Python loops over trials and then over overlapping stimulus rows inside each trial, plus the repeated list-building for dF/F traces and the per-session global-info pass.

ii.
```python
traces = [np.asarray(v, dtype=np.float32) for v in dff['dff'].values]
...
for _, tr in trials.iterrows():
    ...
    for _, srow in stim_trial.iterrows():
        ...
        image_series[smask] = image_to_idx.get(name, image_to_idx['blank'])
```

iii. `CONVERSION_NOTES.md` Step 6 has placeholders for “Code inefficiencies identified,” and the trajectory repeatedly comments on performance. The delivered code remains mostly loop-based.

## 9-c. What processing does the code repeat multiple times?

i. It reloads NWB sessions in `collect_global_info` and then loads them again in the main conversion pass. It also reopens the first session once more just to estimate `time_bin_size`.

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

iii. The trajectory explicitly mentions this as an efficiency issue and describes the subset-based quantile pass as an optimization to avoid loading all files twice. Even after that change, repeated loading remains.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several values are computed but not used in the final dataset: `stim_start`, `pupil_col`, `image_names_seen`, and `trial_outcomes_seen`. The script also stores zero-width `input` arrays even though the decoder task specifies no inputs.

ii.
```python
stim_start = np.asarray(stim['start_time'], dtype=float)
...
pupil_t, pupil_v, pupil_col = pick_pupil_series(eye)

session_neural, session_input, session_output = [], [], []
image_names_seen = []
trial_outcomes_seen = []
...
image_names_seen.append(name)
...
trial_outcomes_seen.append(outcome)
...
session_input.append(np.zeros((0, len(trial_ts)), dtype=np.float32))
```

iii. These are side effects of writing a general conversion script quickly. The trajectory focuses on passing validation and improving decoder accuracy, not minimizing discarded bookkeeping.
