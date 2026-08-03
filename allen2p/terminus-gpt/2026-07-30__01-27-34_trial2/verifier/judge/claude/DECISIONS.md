# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by discovering NWB files under `data/visual-behavior-ophys-1.1.0` using `rglob('*.nwb')`, then loading each NWB file directly via `BehaviorOphysExperiment.from_nwb_path()`. Each NWB file corresponds to one ophys experiment (one imaging plane in one session).

ii.
```python
def list_nwb_files():
    files = sorted(DATASET_ROOT.rglob('*.nwb'))
    if not files:
        raise FileNotFoundError(f'No NWB files found under {DATASET_ROOT}')
    return files

# In process_experiment:
exp = BehaviorOphysExperiment.from_nwb_path(str(nwb_path))
```

iii. The AI chose to load NWB files directly rather than using the SDK's `VisualBehaviorOphysProjectCache`. This bypasses the cache/project-level filtering but still accesses the same underlying data structures. The AI noted in CONVERSION_NOTES.md Step 1 that `BehaviorOphysExperiment.from_nwb` is the primary loading function.

## 1-b. How are the data split into subjects?

i. Subjects are determined from the `mouse_id` field in each experiment's metadata. Unique subjects are collected across all processed sessions.

ii.
```python
subjects = sorted({s['subject'] for s in processed_sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}

# In process_experiment:
session = {
    'subject': str(meta['mouse_id']),
    ...
}
```

iii. The AI uses the same `mouse_id` identifier as the reference. The sorted unique set ensures deterministic ordering.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file (one ophys experiment / one imaging plane) as a separate session. This means a single ophys session with multiple imaging planes becomes multiple "sessions" in the output.

ii.
```python
def process_experiment(nwb_path):
    exp = BehaviorOphysExperiment.from_nwb_path(str(nwb_path))
    # ... processes entire experiment as one session
    session = {
        'session_id': int(meta['ophys_experiment_id']),
        ...
    }
```

iii. The AI documented this decision in CONVERSION_NOTES.md Step 4: "Treat each ophys experiment NWB as one decoder session because neural populations are experiment-specific." This differs from the reference which groups multiple experiments by `ophys_session_id`.

## 1-d. How are the data split into trials?

i. The AI defines trials as individual **stimulus presentation intervals** (~750ms each), NOT as full SDK trials (which span from trial start_time to stop_time, ~8s). Each non-omitted, non-gray stimulus presentation within a valid (Go/Catch, non-aborted, non-auto-rewarded) SDK trial becomes one "trial" in the converted data.

ii.
```python
stim = exp.stimulus_presentations.copy().sort_values('start_time')
stim = stim[stim['trials_id'].notna()].copy()
stim = stim[stim['trials_id'].isin(valid_trial_ids)].copy()
if 'active' in stim.columns:
    stim = stim[stim['active'].fillna(False)].copy()
if 'omitted' in stim.columns:
    stim = stim[~stim['omitted'].fillna(False)].copy()
stim = stim[stim['image_name'].notna()].copy()

for stim_id, parent_id, start, stop, img_name, is_change in zip(
    stim_ids, stim_trial_ids, stim_start, stim_end, stim_image_name, stim_is_change
):
    left = np.searchsorted(ophys_t, start, side='left')
    right = np.searchsorted(ophys_t, stop, side='left')
    if right - left < 2:
        continue
```

iii. The AI justified this in CONVERSION_NOTES.md Step 7: "Revised conversion uses image-presentation intervals as trials, eliminating invalid labels and matching the paper's image-by-image analysis." The paper does mention 750ms image presentation intervals for analysis, which the AI interpreted as the trial unit.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in two stages: (1) SDK trials are filtered to keep Go/Catch and exclude Aborted/Auto-rewarded using `valid_trials_df()`, then (2) stimulus presentations are filtered to keep only those belonging to valid trials, with active presentations, non-omitted, with valid image names and start/end times. Presentations with fewer than 2 ophys frames are skipped. Sessions with fewer than 2 valid trials are excluded.

ii.
```python
def valid_trials_df(trials):
    trials = trials.copy()
    keep = (trials['go'].fillna(False) | trials['catch'].fillna(False))
    keep &= ~trials['aborted'].fillna(False)
    keep &= ~trials['auto_rewarded'].fillna(False)
    trials = trials.loc[keep].copy()
    trials['trial_outcome_idx'] = trials.apply(get_trial_outcome, axis=1)
    trials = trials.loc[trials['trial_outcome_idx'].notna()].copy()
    return trials
```

iii. The trial filtering logic for Go/Catch inclusion and Aborted/Auto-rewarded exclusion matches the instructions. The additional stimulus presentation filtering (active, non-omitted, valid image name) is a consequence of the AI's decision to use stimulus presentations as trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses `events` (detected calcium events) from each experiment, accessed via `exp.events`. This is extracted using `extract_events_matrix()` which stacks per-cell event traces.

ii.
```python
def extract_events_matrix(events_df):
    event_col = 'events'
    arrs = [np.asarray(x, dtype=np.float32) for x in events_df[event_col].values]
    return np.stack(arrs, axis=0)

# In process_experiment:
neural_full = extract_events_matrix(exp.events)
```

iii. The AI documented this in CONVERSION_NOTES.md Step 4: "methods.txt says neural analyses used detected calcium events" and Step 5: "Neural signal = detected events: methods.txt explicitly states neural analyses used detected calcium events, so use events rather than dF/F."

## 2-b. How is the `neural` data processed?

i. The neural data (event traces) are stacked into a (n_neurons, T) matrix for the full session, then sliced per trial using ophys timestamp indices. Data is cast to float32.

ii.
```python
neural_full = extract_events_matrix(exp.events)
# Per trial:
neural = neural_full[:, left:right].astype(np.float32, copy=False)
```

iii. No additional processing (normalization, smoothing, etc.) is applied. The AI relies on the SDK's pre-computed event detection.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied to neurons. All neurons present in the experiment's events table are included.

ii. N/A - no filtering code.

iii. The AI did not implement any neuron quality filtering beyond what the SDK provides by default.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to ophys timestamps. For each stimulus presentation trial, the ophys frames between the presentation's `start_time` and `end_time` are extracted using `np.searchsorted`.

ii.
```python
left = np.searchsorted(ophys_t, start, side='left')
right = np.searchsorted(ophys_t, stop, side='left')
trial_t = ophys_t[left:right]
neural = neural_full[:, left:right].astype(np.float32, copy=False)
```

iii. The AI uses the stimulus presentation start/end times (not the SDK trial start/stop times) because it defined trials as individual stimulus presentations.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native ophys frame rate (~11 Hz, ~93ms per frame) is preserved. No temporal rebinning is applied. The `time_bin_size` metadata field is set to `None`.

ii.
```python
'metadata': {
    'time_bin_size': None,
    ...
}
```

iii. The AI keeps the native ophys temporal resolution. However, setting `time_bin_size` to `None` rather than computing it from the ophys timestamps is a metadata issue.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `image_name` column of the `stimulus_presentations` table. Since each trial is a single stimulus presentation, there is one image per trial.

ii.
```python
stim_image_name = stim['image_name'].astype(str).to_numpy()
# Per trial:
session['trials'].append({
    'image_name': img_name,
    ...
})
```

iii. Because trials are defined as individual stimulus presentations, image identity is static within each trial (a single image name per presentation).

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global sorted mapping built across all sessions. Each trial gets a constant image identity value across all its timepoints.

ii.
```python
image_names = sorted({tr['image_name'] for s in processed_sessions for tr in s['trials']})
image_to_idx = {name: i for i, name in enumerate(image_names)}
# Per trial:
np.full(T, image_to_idx[tr['image_name']], dtype=np.int64),
```

iii. The global sorted mapping ensures consistent integer codes across sessions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is a constant value repeated across all ophys frames in the stimulus presentation interval, sharing the same temporal indices as the neural data.

ii.
```python
np.full(T, image_to_idx[tr['image_name']], dtype=np.int64),
```

iii. Since each trial is a single stimulus presentation, the image identity is trivially aligned - it's the same value at every timepoint.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column of the `stimulus_presentations` table.

ii.
```python
if 'is_change' in stim.columns:
    stim_is_change = stim['is_change'].fillna(False).astype(bool).to_numpy()
else:
    stim_is_change = np.zeros(len(stim), dtype=bool)
# Per trial:
'image_change': np.full(right - left, int(is_change), dtype=np.int64),
```

iii. The AI uses the SDK's pre-computed `is_change` flag from stimulus presentations.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The `is_change` boolean from stimulus_presentations is converted to an integer (0 or 1) and filled across all timepoints of the trial. Since each trial is one stimulus presentation, this is a constant value per trial.

ii.
```python
'image_change': np.full(right - left, int(is_change), dtype=np.int64),
```

iii. No additional processing beyond type conversion.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is already binary (0/1) from the `is_change` flag. No thresholding is applied.

ii. See 4-b above.

iii. The binary nature comes directly from the stimulus presentation metadata.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is a constant value across all ophys frames in the stimulus presentation interval, using the same temporal indices as neural data.

ii.
```python
'image_change': np.full(right - left, int(is_change), dtype=np.int64),
```

iii. Same alignment approach as image identity - trivially aligned since each trial is one presentation.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `exp.running_speed`, using the `timestamps` and `speed` columns.

ii.
```python
run_t = exp.running_speed['timestamps'].to_numpy(dtype=np.float64)
run_v = exp.running_speed['speed'].to_numpy(dtype=np.float32)
```

iii. Same source as the reference solution.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is interpolated to ophys timestamps using `np.interp`, then discretized into 5 percentile-based bins computed globally across all sessions. NaN values outside the interpolation range are filled with the median bin value of finite data.

ii.
```python
def interp_to_ophys(src_t, src_v, dst_t):
    return np.interp(dst_t, src_t, src_v, left=np.nan, right=np.nan).astype(np.float32)

def compute_bin_edges(values, n_bins=5):
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(values, qs)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    return edges

def digitize_with_edges(values, edges):
    out = np.digitize(values, edges[1:-1], right=False).astype(np.int64)
    bad = ~np.isfinite(values)
    if np.any(bad):
        finite = np.where(np.isfinite(values))[0]
        fill = int(np.median(out[finite])) if finite.size else 0
        out[bad] = fill
    out = np.clip(out, 0, len(edges) - 2)
    return out
```

iii. The bin edge computation adds a small epsilon (1e-6) to handle duplicate quantile edges. NaN filling uses the median bin value rather than a fixed value.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 equal percentile bins using global bin edges computed across all sessions. The `np.quantile` function with `np.linspace(0, 1, 6)` determines bin boundaries.

ii. See 5-b above.

iii. Same general approach as the reference (percentile-based binning), with minor implementation differences in edge handling and NaN fill strategy.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to ophys timestamps within each trial using `np.interp`, ensuring alignment with neural data at the same time points.

ii.
```python
run_aligned = interp_to_ophys(run_t, run_v, trial_t)
```

iii. The interpolation approach is similar to the reference's use of `scipy.interpolate.interp1d`.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `exp.eye_tracking`. The AI uses `pupil_area` preferentially (computing diameter as `2*sqrt(area/pi)`), falling back to `sqrt(pupil_width * pupil_height)` if area is unavailable.

ii.
```python
def pupil_diameter_series(eye_tracking_df):
    et = eye_tracking_df.copy()
    blink = et['likely_blink'].fillna(False).to_numpy(dtype=bool)
    if 'pupil_area' in et.columns:
        area = et['pupil_area'].to_numpy(dtype=np.float64)
        diam = 2.0 * np.sqrt(area / math.pi)
    elif 'pupil_width' in et.columns and 'pupil_height' in et.columns:
        diam = np.sqrt(et['pupil_width'].to_numpy(dtype=np.float64) * et['pupil_height'].to_numpy(dtype=np.float64))
    else:
        raise KeyError('No pupil area/width-height columns available')
    diam[blink] = np.nan
    return et['timestamps'].to_numpy(dtype=np.float64), diam.astype(np.float32)
```

iii. The AI documented in CONVERSION_NOTES.md Step 5: "Pupil variable choice: derive diameter from pupil area if no direct diameter column exists; mask likely blinks and missing values before binning/alignment."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter is computed from pupil area, blink frames are set to NaN, then the signal is interpolated to ophys timestamps using `np.interp`. Discretization into 5 percentile bins follows, with NaN values filled using the median bin value.

ii. See 6-a and 5-b above for the relevant code.

iii. Blink exclusion is done by setting blink frames to NaN before interpolation, similar to the reference's approach of filtering out blink rows before interpolation.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 equal percentile bins with global bin edges across all sessions.

ii. Same `compute_bin_edges` and `digitize_with_edges` functions as running speed.

iii. Consistent discretization approach across both continuous variables.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to ophys timestamps within each trial, same approach as running speed.

ii.
```python
pupil_aligned = interp_to_ophys(pupil_t, pupil_v, trial_t)
```

iii. Same alignment mechanism as running speed.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the `hit`, `miss`, `false_alarm`, and `correct_reject` columns in the SDK trials table.

ii.
```python
def get_trial_outcome(row):
    if bool(row.get('hit', False)):
        return 0
    if bool(row.get('miss', False)):
        return 1
    if bool(row.get('false_alarm', False)):
        return 2
    if bool(row.get('correct_reject', False)):
        return 3
    return None
```

iii. Same source variables as the reference. The mapping order matches the reference's `TRIAL_OUTCOMES = ['hit', 'miss', 'false_alarm', 'correct_reject']`.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is determined from the first matching boolean flag in the trials table. The outcome is mapped to an integer code (0-3) and stored as a static per-trial value, broadcast across all timepoints. Each stimulus presentation inherits the outcome of its parent SDK trial.

ii.
```python
trial_outcome_map = trials['trial_outcome_idx'].to_dict()
# Per stimulus presentation trial:
outcome = int(trial_outcome_map[parent_id])
# In assemble_dataset:
np.full(T, tr['trial_outcome'], dtype=np.int64),
```

iii. Since each stimulus presentation belongs to one SDK trial, the outcome is inherited from the parent trial's outcome. Trials where no outcome flag is True are excluded (return None).

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Failed sessions**: If `process_experiment` encounters an error, the session is skipped (in parallel mode, errors would propagate but sessions with <2 trials are filtered).
- **Short trials**: Stimulus presentations with fewer than 2 ophys frames (`right - left < 2`) are skipped.
- **Missing behavioral data**: NaN values from interpolation are filled with the median bin value of finite data during discretization.
- **Duplicate bin edges**: If quantile edges are equal, a small epsilon (1e-6) is added.
- **Few trials**: Sessions with fewer than 2 valid trials are excluded.
- **Missing stimulus metadata**: Presentations with missing `image_name`, `start_time`, or `end_time` are filtered out.

ii.
```python
if right - left < 2:
    continue

# In digitize_with_edges:
bad = ~np.isfinite(values)
if np.any(bad):
    finite = np.where(np.isfinite(values))[0]
    fill = int(np.median(out[finite])) if finite.size else 0
    out[bad] = fill

# In compute_bin_edges:
for i in range(1, len(edges)):
    if edges[i] <= edges[i - 1]:
        edges[i] = edges[i - 1] + 1e-6
```

iii. The AI handles edge cases robustly with fallback values and filtering.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each NWB file via `BehaviorOphysExperiment.from_nwb_path()`, which reads and parses large HDF5/NWB data. The AI estimated ~22 seconds per session.

ii.
```python
exp = BehaviorOphysExperiment.from_nwb_path(str(nwb_path))
```

iii. The AI noted this in CONVERSION_NOTES.md Step 7 with an estimated total time of ~104 minutes for 284 sessions. The AI implemented multiprocessing to parallelize this.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-stimulus-presentation loop in `process_experiment` iterates over each presentation sequentially, performing `np.searchsorted` and array slicing. This could potentially be vectorized.

ii.
```python
for stim_id, parent_id, start, stop, img_name, is_change in zip(
    stim_ids, stim_trial_ids, stim_start, stim_end, stim_image_name, stim_is_change
):
    left = np.searchsorted(ophys_t, start, side='left')
    right = np.searchsorted(ophys_t, stop, side='left')
    ...
```

iii. The loop is simple and readable. Data loading dominates runtime, so vectorizing this loop would yield minimal improvement.

## 9-c. What processing does the code repeat multiple times?

i. Running speed and pupil diameter raw values are stored per-trial (`running_raw`, `pupil_raw`) AND collected separately in session-level lists (`running_raw_all`, `pupil_raw_all`). This duplicates the behavioral data in memory.

ii.
```python
session['running_raw_all'].append(run_aligned)
session['pupil_raw_all'].append(pupil_aligned)
session['trials'].append({
    ...
    'running_raw': run_aligned,
    'pupil_raw': pupil_aligned,
    ...
})
```

iii. The duplication is for convenience - `running_raw_all` is used for global bin edge computation, while per-trial values are used in assembly. This is a minor inefficiency.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `interp_to_ophys` function performs sorting (`np.argsort`) and finite-value filtering on every call, even though the source timestamps are already sorted and mostly finite. The `build_image_labels_for_trial` function is defined but not used in the final code path. Processing plots are generated even if not strictly needed.

ii.
```python
def interp_to_ophys(src_t, src_v, dst_t):
    good = np.isfinite(src_t) & np.isfinite(src_v)
    ...
    order = np.argsort(src_t)
    src_t = src_t[order]
    src_v = src_v[order]
    return np.interp(dst_t, src_t, src_v, left=np.nan, right=np.nan).astype(np.float32)
```

iii. The sorting and filtering are defensive but unnecessary for well-formed SDK data. The `build_image_labels_for_trial` function appears to be from an earlier iteration of the code that used SDK trials rather than stimulus presentations.
