# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by iterating over all NWB files in the `behavior_ophys_experiments` directory. Each NWB file is opened with `pynwb.NWBHDF5IO` and then wrapped via `BehaviorOphysExperiment.from_nwb()` from the AllenSDK. A two-pass approach is used: first, a subset of files (up to 8) is loaded to compute global statistics (image names, running speed and pupil percentile bin edges), then all files are loaded again for full processing.

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

iii. The AI identified the AllenSDK `BehaviorOphysExperiment` as the canonical way to load the data, consistent with the reference code and SDK documentation. It iterates over locally available NWB files (284 files). Loading is done per-experiment rather than using the project metadata CSVs as a master list.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by extracting `mouse_id` from each experiment's metadata dictionary. Unique mouse IDs are collected across all sessions and sorted to form the `subjects` list. A `subject_idx` array maps each session to its subject.

ii.
```python
subjects = sorted({s['mouse_id'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
# ...
'subject_idx': np.array([subject_to_idx[s['mouse_id']] for s in sessions], dtype=np.int64),
```
Where `mouse_id` comes from:
```python
meta = exp.metadata if isinstance(exp.metadata, dict) else {}
mouse_id = str(meta.get('mouse_id', 'unknown'))
```

iii. The AI used the experiment metadata embedded in each NWB file to determine subject identity, which is the standard approach for this dataset. 38 unique subjects were found across 284 sessions.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one ophys experiment, and the AI treats each file as a separate session. Sessions are processed sequentially and stored as lists in the output dictionary.

ii.
```python
files = list_session_files(sample=sample)
sessions = []
for i, p in enumerate(files, 1):
    exp = load_experiment(p)
    sess = build_session(exp, image_to_idx, run_edges, pupil_edges, outcome_to_idx)
    sessions.append(sess)
```

iii. The AI treats each NWB experiment file as one session. This is consistent with the dataset structure where each file represents one ophys experiment.

## 1-d. How are the data split into trials?

i. Trials are extracted from the experiment's `trials` table (via `exp.trials`). Trial boundaries are defined by `start_time` and `stop_time` (with `end_time` as fallback). For each trial, ophys timestamps falling within `[start_time, stop_time)` define the trial's timepoints.

ii.
```python
trials = valid_trials_table(exp.trials)
# ...
for _, tr in trials.iterrows():
    start = float(tr['start_time'])
    stop = float(tr['stop_time']) if 'stop_time' in tr.index else float(tr['end_time'])
    idx = np.flatnonzero((ts >= start) & (ts < stop))
    if idx.size < 2:
        continue
```

iii. The AI uses the experiment-defined trial intervals from the trials table, as specified in the instructions ("Segment each recording session into individual trials based on how they are defined in the experiment").

## 1-e. How are trials filtered based on quality controls?

i. Trials labeled as `aborted` or `auto_rewarded` (or variants) are excluded. Only `Go` and `Catch` trials are kept. Additionally, trials with fewer than 2 ophys timepoints are silently skipped.

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

iii. The instructions explicitly require including Go and Catch trials and excluding Aborted and Auto-rewarded trials. The AI checks multiple possible column names for robustness.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `dff_traces` (delta F/F fluorescence traces) from the `BehaviorOphysExperiment` object.

ii.
```python
def get_cell_matrix(exp):
    dff = exp.dff_traces.copy()
    if isinstance(dff, pd.DataFrame) and 'dff' in dff.columns:
        cell_ids = np.asarray(dff.index)
        traces = [np.asarray(v, dtype=np.float32) for v in dff['dff'].values]
        mat = np.stack(traces, axis=0)
        return mat, cell_ids, 'dff'
    raise RuntimeError('Could not access dff traces')
```

iii. The paper states "For all analysis of neural data we used the detected calcium events." The AI initially tried using events but switched to dF/F because events were extremely sparse (~99.75% zeros), making decoder training ineffective. The agent justified this by noting the decoder worked better with dense time-series data.

## 2-b. How is the `neural` data processed?

i. The dF/F traces are extracted as a full `(n_neurons, n_total_timepoints)` matrix, then sliced per trial using ophys timestamp indices. The data is cast to float32. No additional processing (normalization, filtering, smoothing) is applied.

ii.
```python
traces = [np.asarray(v, dtype=np.float32) for v in dff['dff'].values]
mat = np.stack(traces, axis=0)
# ...
trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
```

iii. The AI chose minimal processing — just extracting the pre-computed dF/F from the SDK. No additional filtering or normalization was applied, which is reasonable since the dF/F computation is already done in the processing pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality filtering is applied. All neurons available from `dff_traces` are included. The only filtering is at the trial level (trials with fewer than 2 timepoints are skipped).

ii. There is no neuron filtering code. All cell IDs from `dff.index` are used:
```python
cell_ids = np.asarray(dff.index)
```

iii. The AI does not document a reason for skipping neuron quality filtering. The AllenSDK `BehaviorOphysExperiment` has an `exclude_invalid_rois` parameter that defaults to True, which may perform some filtering at load time, but the AI does not explicitly verify or leverage this.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned using ophys timestamps. For each trial, the ophys timestamps falling within `[trial_start, trial_stop)` are identified, and the corresponding columns of the neural matrix are extracted.

ii.
```python
ts = np.asarray(exp.ophys_timestamps, dtype=float)
# ...
idx = np.flatnonzero((ts >= start) & (ts < stop))
trial_ts = ts[idx]
trial_neural = neural_mat[:, idx].astype(np.float32, copy=False)
```

iii. The instructions say "Temporally align based on ophys timestamp." The AI aligns to the trial's time window as defined by the trials table, using the common ophys timestamp array. There is no fixed offset alignment to a specific event within each trial (like stimulus onset); it simply uses the trial boundaries.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native ophys frame rate (~31 Hz, approximately 32.3 ms per frame). No temporal rebinning is applied.

ii.
```python
dt_ms = float(np.median(np.diff(load_experiment(files[0]).ophys_timestamps)) * 1000.0)
# stored in metadata:
'time_bin_size': dt_ms,
```

iii. The AI preserves the native ophys temporal resolution without rebinning. This is consistent with the instruction to "Temporally align based on ophys timestamp" — the ophys timestamps define the natural time base.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `image_name` column of `stimulus_presentations` from the `BehaviorOphysExperiment`.

ii.
```python
stim = exp.stimulus_presentations.copy().sort_values('start_time')
# ...
name = str(srow['image_name']) if 'image_name' in srow.index else 'blank'
image_series[smask] = image_to_idx.get(name, image_to_idx['blank'])
```

iii. The `stimulus_presentations` table contains the identity of each image shown. The AI maps each unique image name to a categorical index.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The AI first collects all unique image names across sessions (from a subset of 8 files for efficiency, plus 'blank' and 'omitted' as defaults). Each image name is mapped to an integer index. For each trial, the stimulus presentations overlapping the trial window are identified, and each ophys timepoint is labeled with the image index of the overlapping stimulus. Timepoints not covered by any stimulus presentation default to 'blank'. Image presentation intervals are assumed to be 750 ms (`stim_end = stim_start + 0.75`).

ii.
```python
stim_start = np.asarray(stim['start_time'], dtype=float)
stim_end = stim_start + 0.75
# ...
image_series = np.full(len(trial_ts), image_to_idx['blank'], dtype=np.int64)
stim_trial = stim[(stim['start_time'] < stop) & (stim['_end_time'] > start)]
for _, srow in stim_trial.iterrows():
    s0 = max(start, float(srow['start_time']))
    s1 = min(stop, float(srow['_end_time']))
    smask = (trial_ts >= s0) & (trial_ts < s1)
    name = str(srow['image_name']) if 'image_name' in srow.index else 'blank'
    image_series[smask] = image_to_idx.get(name, image_to_idx['blank'])
```

iii. The AI uses 750 ms intervals consistent with the methods text describing 750 ms image presentation intervals. Initially used actual flash duration (~250 ms) but switched to 750 ms after finding that the shorter duration made ~65% of timepoints "blank," destroying decoder performance.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Both are on the same ophys timestamp grid. For each ophys timepoint within a trial, the image identity is determined by which stimulus presentation interval contains that timepoint.

ii. Same code as 3-b — `trial_ts` is the set of ophys timestamps for the trial, and `smask` identifies which ophys timepoints fall within each stimulus presentation window.

iii. Alignment is inherent since both neural data and image identity labels are indexed by the same ophys timestamps within each trial.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `image_name` column of `stimulus_presentations` by tracking transitions between consecutive stimulus presentations within each trial.

ii.
```python
prev_name = None
for _, srow in stim_trial.iterrows():
    # ...
    name = str(srow['image_name']) if 'image_name' in srow.index else 'blank'
    if prev_name is not None and name != prev_name:
        hit = np.flatnonzero(smask)
        if hit.size:
            change_series[hit[0]] = 1
    prev_name = name
```

iii. The AI computes image changes by detecting when consecutive stimulus presentations have different image names, rather than using the pre-computed `is_change` column available in `stimulus_presentations`.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary time series is initialized to zeros. The stimulus presentations within each trial are iterated in time order. When two consecutive presentations have different `image_name` values, the first ophys timepoint of the new stimulus is marked with 1.

ii. Same code as 4-a.

iii. The AI's approach detects any transition in image name, not specifically the task-defined "change" event. This could include transitions from omitted images or between repeated images and gray screens that aren't true "changes" in the behavioral task sense.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1), so no additional thresholding is needed.

ii.
```python
change_series = np.zeros(len(trial_ts), dtype=np.int64)
# ...
change_series[hit[0]] = 1
```

iii. The instructions specify "binary variable. Have value of 1 right after a change in image identity, otherwise 0."

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The change event is placed at the first ophys timepoint of the new stimulus presentation, using the same ophys timestamp grid as the neural data.

ii. Same alignment mechanism as image identity — both use the trial's ophys timestamps.

iii. Aligned by construction since change events are placed on ophys timepoints.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `exp.running_speed`, which provides a DataFrame with timestamps and speed values.

ii.
```python
def get_running_series(run_df):
    cols = list(run_df.columns)
    tcol = 'timestamps' if 'timestamps' in cols else cols[0]
    vcol = 'speed' if 'speed' in cols else cols[1]
    return np.asarray(run_df[tcol], dtype=float), np.asarray(run_df[vcol], dtype=float)
```

iii. The AllenSDK's `running_speed` property provides pre-processed running speed data aligned to the session clock.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is interpolated onto ophys timestamps using `np.interp`, then discretized into 5 bins using percentile-based bin edges computed from a subset of sessions.

ii.
```python
run_interp = np.interp(trial_ts, run_t, run_v).astype(np.float32)
run_bins = np.digitize(run_interp, run_edges[1:-1], right=False).astype(np.int64)
```
Bin edges computed from first 8 sessions:
```python
run_all = np.concatenate(run_vals) if run_vals else np.array([0.0])
run_edges = np.quantile(run_all, [0, .2, .4, .6, .8, 1])
```

iii. The instructions specify "discretized into five equal percentile bins." The AI computes quantiles at 0%, 20%, 40%, 60%, 80%, 100%, which defines 5 bins.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed values are mapped to bins 0-4 using `np.digitize` with the inner edges (20th, 40th, 60th, 80th percentiles). Values below the 20th percentile go to bin 0, values at or above the 80th percentile go to bin 4.

ii. Same as 5-b.

iii. The 5 equal percentile bins are labeled as `['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']`.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same ophys timestamps used for neural data, so alignment is automatic.

ii.
```python
run_interp = np.interp(trial_ts, run_t, run_v).astype(np.float32)
```

iii. Interpolation to ophys timestamps ensures temporal alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from the `eye_tracking` data, specifically the `pupil_area` column (or similar pupil-related column found by the `pick_pupil_series` function).

ii.
```python
def pick_pupil_series(eye_tracking):
    # ...
    candidate_cols = ['pupil_diameter', 'pupil_area', 'pupil_width', 'pupil_height', 'pupil_radius', 'pupil_size']
    value_col = next((c for c in candidate_cols if c in cols), None)
    # ...
    return np.asarray(eye_tracking[time_col], dtype=float), np.asarray(eye_tracking[value_col], dtype=float), value_col
```

iii. The AI searches for pupil-related columns in priority order. Since the NWB files contain `pupil_area` in the eye tracking data, this is likely the column used. Note that `pupil_area` is not the same as `pupil_diameter` — the instructions specify "pupil diameter" but the code may use pupil area instead.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The pupil signal is interpolated onto ophys timestamps (filtering out non-finite values), then discretized into 5 percentile bins using the same approach as running speed. If eye tracking is unavailable or has fewer than 2 valid data points, pupil bins default to zeros.

ii.
```python
valid = np.isfinite(pupil_t) & np.isfinite(pupil_v)
if valid.sum() >= 2:
    pupil_interp = np.interp(trial_ts, pupil_t[valid], pupil_v[valid]).astype(np.float32)
    pupil_bins = np.digitize(pupil_interp, pupil_edges[1:-1], right=False).astype(np.int64)
else:
    pupil_bins = np.zeros(len(trial_ts), dtype=np.int64)
```

iii. The AI handles missing pupil data by defaulting to bin 0, which would bias the bin distribution.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same 5-bin percentile approach as running speed: quintile edges from a subset of sessions, applied via `np.digitize`.

ii. Same as 6-b.

iii. Bins are labeled `['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']`.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolated onto ophys timestamps, same as running speed.

ii. `pupil_interp = np.interp(trial_ts, pupil_t[valid], pupil_v[valid])`

iii. Alignment through interpolation to common ophys timestamp grid.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from boolean columns in the trials table: `hit`, `miss`, `false_alarm`, `correct_reject`.

ii.
```python
def infer_trial_outcome(row):
    for key in ['hit', 'miss', 'false_alarm', 'correct_reject']:
        if key in row.index and bool(row[key]):
            return key
    return 'other'
```

iii. The trial table in the NWB data contains boolean flags for each outcome type. The AI checks each flag in priority order and returns the first True value.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The trial outcome is determined per trial, then broadcast to all timepoints in the trial as a constant time-varying series.

ii.
```python
outcome = infer_trial_outcome(tr)
outcome_idx = outcome_to_idx.get(outcome, outcome_to_idx['other'])
outcome_series = np.full(len(trial_ts), outcome_idx, dtype=np.int64)
```

iii. The instructions specify trial outcome as "Static per-trial." However, the AI represents it as a time-varying series (repeated constant), which is equivalent since every timepoint has the same value. The target format uses `(n_output, n_timepoints)` for all outputs in a stacked array, so this representation works with the data structure.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several missing data scenarios are handled:
- Missing eye tracking: defaults pupil bins to zeros (bin 0)
- Missing stimulus stop_time: uses `end_time` or assumes 750 ms duration
- Short trials (< 2 ophys timepoints): silently skipped
- NaN values in running/pupil: filtered with `np.isfinite` before computing percentiles
- Unknown image names: mapped to 'blank' index
- Missing trial outcome flags: classified as 'other'

ii.
```python
# Missing eye tracking
except Exception:
    eye = None
# ...
if pupil_t is not None and pupil_v is not None:
    valid = np.isfinite(pupil_t) & np.isfinite(pupil_v)
    # ...
else:
    pupil_bins = np.zeros(len(trial_ts), dtype=np.int64)
```

iii. The AI handles missing data reasonably but the choice to default missing pupil data to bin 0 rather than a separate "missing" category could introduce bias.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each NWB file via `load_experiment()` (3-20 seconds per session). The total conversion for 284 sessions took ~31.5 minutes. Each NWB file is loaded at least once for processing, and the first 8 files are loaded an additional time for global statistics computation.

ii.
```python
exp = load_experiment(p)  # 3-20 seconds per session
```

iii. NWB file I/O dominates processing time. The agent noted the 15-minute target was exceeded but did not further optimize.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Two inner loops could be vectorized:
1. The loop over stimulus presentations within each trial (`for _, srow in stim_trial.iterrows()`) that assigns image identity and detects changes.
2. The loop over trials within each session (`for _, tr in trials.iterrows()`) uses `iterrows()` which is slow for pandas DataFrames.

ii.
```python
for _, srow in stim_trial.iterrows():  # Inner loop over stimulus presentations
    # ...
for _, tr in trials.iterrows():  # Outer loop over trials
    # ...
```

iii. The stimulus presentation assignment could be vectorized using `np.searchsorted` on the sorted stimulus start times. The trial loop could potentially use bulk array operations for extracting neural data slices.

## 9-c. What processing does the code repeat multiple times?

i. NWB file loading is repeated: the first 8 files are loaded in `collect_global_info()` to compute percentile bin edges, then loaded again in the main processing loop. This doubles the I/O cost for those sessions.

ii.
```python
def collect_global_info(files):
    probe_files = files[:min(8, len(files))]
    for p in probe_files:
        exp = load_experiment(p)  # First load
        # ...
# ...
for i, p in enumerate(files, 1):
    exp = load_experiment(p)  # Second load for first 8 files
```

iii. The agent optimized from loading ALL files twice (original version loaded all files for global stats) to only loading 8 files twice.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several pieces of data are computed but not used in the final output:
1. `image_names_seen` and `trial_outcomes_seen` per session — computed and stored in intermediate dict but not included in final pickle.
2. `cell_ids` are stored per session but not in the final data dictionary (only used for counting neurons).
3. `neural_source` is tracked per session but only stored as a metadata string, not used in processing.

ii.
```python
'image_names_seen': sorted(set(image_names_seen)),  # Not in final output
'trial_outcomes_seen': sorted(set(trial_outcomes_seen)),  # Not in final output
'cell_ids': cell_ids,  # Used only for brain_region_idx sizing
'neural_source': neural_source,  # Only used in print/metadata
```

iii. These are minor inefficiencies — the main unnecessary processing is the duplicate file loading for global statistics described in 9-c.
