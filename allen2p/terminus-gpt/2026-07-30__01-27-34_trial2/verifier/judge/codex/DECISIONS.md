# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all data by recursively finding every local NWB file under `data/visual-behavior-ophys-1.1.0`, optionally truncating to the first two files in sample mode, and then loading each file with `BehaviorOphysExperiment.from_nwb_path(...)`. It does not use the Allen SDK project cache or the experiment table to discover sessions.

ii. 
```python
def list_nwb_files():
    files = sorted(DATASET_ROOT.rglob('*.nwb'))
    if not files:
        raise FileNotFoundError(f'No NWB files found under {DATASET_ROOT}')
    return files


def choose_files(sample=False):
    files = list_nwb_files()
    return files[:2] if sample else files
```

```python
def process_experiment(nwb_path):
    t0 = time.time()
    exp = BehaviorOphysExperiment.from_nwb_path(str(nwb_path))
```

iii. In `CONVERSION_NOTES.md` Step 4, the AI says the actual dataset is stored as per-experiment NWB files and resolves to use `BehaviorOphysExperiment` NWB loading. The notes then explicitly treat each NWB as the natural unit to process.

## 1-b. How are the data split into subjects?

i. Subjects are taken from `exp.metadata['mouse_id']`. After all sessions are processed, unique subject IDs are collected and sorted, and `subject_idx` maps each kept session to that subject list.

ii. 
```python
session = {
    'session_id': int(meta['ophys_experiment_id']),
    'subject': str(meta['mouse_id']),
    'region': str(meta['targeted_structure']),
```

```python
subjects = sorted({s['subject'] for s in processed_sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
'subject_idx': np.array([subject_to_idx[s['subject']] for s in processed_sessions], dtype=np.int64),
```

iii. Step 5 of `CONVERSION_NOTES.md` maps `metadata['mouse_id']` directly onto `subjects` and `subject_idx`.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file, i.e. each `ophys_experiment_id`, as one decoder session. It does not group multiple experiments that share an `ophys_session_id`.

ii. 
```python
for i, f in enumerate(files, start=1):
    print(f'[{i}/{len(files)}] loading {f}')
    sess = process_experiment(f)
```

```python
session = {
    'session_id': int(meta['ophys_experiment_id']),
    'subject': str(meta['mouse_id']),
    'region': str(meta['targeted_structure']),
```

iii. Step 4 and Step 5 in `CONVERSION_NOTES.md` say “Treat each ophys experiment NWB as one decoder session because neural populations are experiment-specific,” and list “Session unit = ophys experiment” as a key decision.

## 1-d. How are the data split into trials?

i. The final code does not use the SDK trial start/stop windows as decoder trials. Instead, it filters `stimulus_presentations` to presentations belonging to valid parent trials, and each kept image-presentation interval becomes one decoder trial.

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
stim = stim[stim['start_time'].notna() & stim['end_time'].notna()].copy()
```

```python
for stim_id, parent_id, start, stop, img_name, is_change in zip(
    stim_ids, stim_trial_ids, stim_start, stim_end, stim_image_name, stim_is_change
):
    left = np.searchsorted(ophys_t, start, side='left')
    right = np.searchsorted(ophys_t, stop, side='left')
    if right - left < 2:
        continue
    ...
    session['trials'].append({
        'trial_id': int(stim_id) if isinstance(stim_id, (int, np.integer)) else len(session['trials']),
        'start_time': float(start),
        'stop_time': float(stop),
```

iii. The trajectory shows this was a deliberate revision. In steps 20-21, the AI explains that full-trial segmentation produced invalid `image_identity = -1` during gray periods, so it switched to image-presentation intervals to keep image labels valid and to “match the paper’s image-by-image analysis.”

## 1-e. How are trials filtered based on quality controls?

i. Parent SDK trials are first filtered to `go` or `catch`, excluding `aborted` and `auto_rewarded`, and keeping only rows with a recognized outcome label. Then stimulus presentations are filtered to those belonging to those valid parent trials, marked `active` if that column exists, not omitted, with non-null `image_name`, `start_time`, and `end_time`. Image-presentation intervals with fewer than 2 ophys frames are skipped, and sessions with fewer than 2 kept trials are dropped.

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

```python
stim = stim[stim['trials_id'].notna()].copy()
stim = stim[stim['trials_id'].isin(valid_trial_ids)].copy()
if 'active' in stim.columns:
    stim = stim[stim['active'].fillna(False)].copy()
if 'omitted' in stim.columns:
    stim = stim[~stim['omitted'].fillna(False)].copy()
stim = stim[stim['image_name'].notna()].copy()
stim = stim[stim['start_time'].notna() & stim['end_time'].notna()].copy()
...
if right - left < 2:
    continue
```

```python
if len(sess['trials']) >= 2:
    processed.append(sess)
...
else:
    print(f'skipping session {sess["session_id"]}: <2 valid trials')
```

iii. The notes repeatedly cite the instruction to include Go and Catch trials while excluding Aborted and Auto-rewarded trials. The later trajectory justifies the additional `stimulus_presentations` filtering as necessary to avoid unlabeled gray/omitted periods and keep only non-grey image intervals.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the `events` column of `exp.events`, i.e. detected calcium event traces from the SDK object.

ii. 
```python
def extract_events_matrix(events_df):
    event_col = 'events'
    if event_col not in events_df.columns:
        raise KeyError(f'Expected {event_col} column in events dataframe')
    arrs = [np.asarray(x, dtype=np.float32) for x in events_df[event_col].values]
    ...
    return np.stack(arrs, axis=0)
```

```python
neural_full = extract_events_matrix(exp.events)
```

iii. Step 4 of `CONVERSION_NOTES.md` explicitly resolves the dF/F vs events discrepancy in favor of detected calcium events because `methods.txt` says the paper’s neural analyses used detected calcium events.

## 2-b. How is the `neural` data processed?

i. The AI stacks the event traces for all ROIs in one experiment into a `(neurons, time)` matrix, converts to `float32`, and then slices that matrix by ophys-frame indices for each image-presentation interval. There is no additional normalization or denoising in the script.

ii. 
```python
arrs = [np.asarray(x, dtype=np.float32) for x in events_df[event_col].values]
...
return np.stack(arrs, axis=0)
```

```python
left = np.searchsorted(ophys_t, start, side='left')
right = np.searchsorted(ophys_t, stop, side='left')
...
neural = neural_full[:, left:right].astype(np.float32, copy=False)
```

iii. Step 5 in the notes says the primary neural signal is the SDK’s detected calcium events, and that the decoder should use direct slicing on native ophys timestamps rather than further signal processing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit neuron-level quality filter is applied in the final script. The code uses whatever ROIs are present in `exp.events` and does not inspect `cell_specimen_table`, valid-ROI flags, or other QC metadata.

ii. 
```python
neural_full = extract_events_matrix(exp.events)
trials = valid_trials_df(exp.trials)
```

iii. The notes discuss relying on SDK-provided valid cells/ROIs and do not specify any extra neuron curation. The final code reflects that by applying no additional neuron filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to each kept stimulus presentation by finding the ophys frames between that presentation’s `start_time` and `end_time`. Each decoder trial therefore covers one image-presentation interval on the native ophys clock.

ii. 
```python
left = np.searchsorted(ophys_t, start, side='left')
right = np.searchsorted(ophys_t, stop, side='left')
if right - left < 2:
    continue
trial_t = ophys_t[left:right]
neural = neural_full[:, left:right].astype(np.float32, copy=False)
```

iii. In trajectory step 21, the AI says it changed the alignment window from full SDK trials to image-presentation intervals to remove invalid image labels during gray periods and to match the paper’s image-by-image framing.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the native ophys frame rate. No temporal rebinning is applied; each image-presentation interval simply contains however many native ophys frames fall in that 750 ms window, typically 7-8 frames. The metadata do not record a numeric `time_bin_size`.

ii. 
```python
trial_t = ophys_t[left:right]
neural = neural_full[:, left:right].astype(np.float32, copy=False)
```

```python
'metadata': {
    ...
    'time_bin_size': None,
```

iii. Step 7 of `CONVERSION_NOTES.md` says the revised conversion “showed aligned neural/activity/output traces over 7-8 ophys-frame image windows,” and trajectory step 23 says those 7-8 frames match a 750 ms image presentation at about 10 Hz.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. `image_identity` is derived from `stimulus_presentations.image_name` after filtering to non-omitted presentations inside valid parent trials.

ii. 
```python
stim = exp.stimulus_presentations.copy().sort_values('start_time')
...
stim = stim[~stim['omitted'].fillna(False)].copy()
stim = stim[stim['image_name'].notna()].copy()
...
stim_image_name = stim['image_name'].astype(str).to_numpy()
```

iii. The notes’ Step 5 mapping says `stimulus_presentations.image_name` should supply the non-grey image identity, and the later trajectory says the AI switched to image-presentation trials precisely so image identity would always correspond to a non-grey screen.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Each kept image presentation is treated as having a single image identity. During assembly, the code collects all distinct image names across processed sessions, sorts them, maps them to integer codes, and fills the whole trial with the corresponding code.

ii. 
```python
image_names = sorted({tr['image_name'] for s in processed_sessions for tr in s['trials']})
image_to_idx = {name: i for i, name in enumerate(image_names)}
```

```python
out = np.vstack([
    np.full(T, image_to_idx[tr['image_name']], dtype=np.int64),
    tr['image_change'],
    run_bin,
    pupil_bin,
    np.full(T, tr['trial_outcome'], dtype=np.int64),
]).astype(np.int64)
```

iii. Trajectory step 21 explicitly says that after redefining each image presentation as a decoder trial, `image_identity` should be “constant across the segment.”

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. `image_identity` is aligned by broadcasting a constant image code over exactly the same ophys-frame slice used for the neural data in that image-presentation interval.

ii. 
```python
trial_t = ophys_t[left:right]
neural = neural_full[:, left:right].astype(np.float32, copy=False)
```

```python
T = tr['neural'].shape[1]
out = np.vstack([
    np.full(T, image_to_idx[tr['image_name']], dtype=np.int64),
```

iii. The AI’s trajectory explains that it wanted all image labels to be valid for every ophys frame kept in a trial, which is why it aligned both neural data and image labels to the same stimulus-presentation window.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. `image_change` is derived from `stimulus_presentations.is_change` for each kept image-presentation interval.

ii. 
```python
if 'is_change' in stim.columns:
    stim_is_change = stim['is_change'].fillna(False).astype(bool).to_numpy()
else:
    stim_is_change = np.zeros(len(stim), dtype=bool)
```

```python
'image_change': np.full(right - left, int(is_change), dtype=np.int64),
```

iii. Step 5 of `CONVERSION_NOTES.md` maps `stimulus_presentations.is_change` directly onto the image-change output, and trajectory step 21 says `image_change` should be taken “from the presentation row.”

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code does not compute image change from `change_time`. It simply converts the boolean `is_change` value on each stimulus-presentation row into a constant binary array spanning that whole image-presentation trial.

ii. 
```python
'image_change': np.full(right - left, int(is_change), dtype=np.int64),
```

```python
out = np.vstack([
    np.full(T, image_to_idx[tr['image_name']], dtype=np.int64),
    tr['image_change'],
```

iii. In step 21 of the trajectory, the AI explicitly states that with image-presentation trials, “image_change [is] constant from the presentation row.”

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is treated as a binary categorical variable with values `0/1`, named `no_change` and `change`.

ii. 
```python
IMAGE_CHANGE_VALUES = ['no_change', 'change']
```

```python
'output_values': [
    image_values,
    IMAGE_CHANGE_VALUES,
    RUN_BIN_VALUES,
    PUPIL_BIN_VALUES,
    TRIAL_OUTCOME_VALUES,
],
```

iii. The notes frame image change as a binary output from `stimulus_presentations.is_change`; the final code uses exactly two category labels.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. `image_change` is aligned by assigning a constant 0/1 label over the same ophys-frame interval used to slice the neural data for that image presentation.

ii. 
```python
trial_t = ophys_t[left:right]
neural = neural_full[:, left:right].astype(np.float32, copy=False)
...
'image_change': np.full(right - left, int(is_change), dtype=np.int64),
```

iii. The AI’s step-21 rationale was that once each decoder trial is an image-presentation interval, image-change alignment is naturally frame-for-frame with the neural slice from that same interval.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is taken from `exp.running_speed['timestamps']` and `exp.running_speed['speed']`.

ii. 
```python
run_t = exp.running_speed['timestamps'].to_numpy(dtype=np.float64)
run_v = exp.running_speed['speed'].to_numpy(dtype=np.float32)
```

iii. Step 5 in the notes maps `running_speed.speed` plus `running_speed.timestamps` directly onto the running-speed output.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The code linearly interpolates running speed onto each trial’s ophys timestamps using `np.interp`, after dropping non-finite source samples and sorting by time. Then it concatenates all kept running-speed samples across sessions, computes global 5-quantile edges, and digitizes each trial’s aligned running-speed values into those bins.

ii. 
```python
def interp_to_ophys(src_t, src_v, dst_t):
    ...
    good = np.isfinite(src_t) & np.isfinite(src_v)
    if good.sum() < 2:
        return np.full(dst_t.shape, np.nan, dtype=np.float32)
    ...
    return np.interp(dst_t, src_t, src_v, left=np.nan, right=np.nan).astype(np.float32)
```

```python
run_aligned = interp_to_ophys(run_t, run_v, trial_t)
...
run_edges = compute_bin_edges(np.concatenate(all_run))
...
run_bin = digitize_with_edges(tr['running_raw'], run_edges)
```

iii. Step 5 of `CONVERSION_NOTES.md` says running speed should be aligned to ophys timestamps and discretized into five equal-frequency bins globally across the included data.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The AI uses five global quantile bins. If quantiles collapse, it nudges later edges upward by `1e-6`. During digitization, non-finite values are filled with the median already-assigned bin within that trial (or 0 if everything is missing), then clipped to the valid bin range.

ii. 
```python
def compute_bin_edges(values, n_bins=5):
    values = values[np.isfinite(values)]
    ...
    edges = np.quantile(values, qs)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    return edges
```

```python
def digitize_with_edges(values, edges):
    ...
    out = np.digitize(values, edges[1:-1], right=False).astype(np.int64)
    bad = ~np.isfinite(values)
    if np.any(bad):
        finite = np.where(np.isfinite(values))[0]
        fill = int(np.median(out[finite])) if finite.size else 0
        out[bad] = fill
    out = np.clip(out, 0, len(edges) - 2)
    return out
```

iii. The notes justify global percentile bins for comparability across sessions. The code’s specific NaN handling is not described in the notes, but it is clearly an intentional robustness choice in the implementation.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated directly onto the ophys timestamps inside each kept image-presentation interval, so the aligned running-speed array has the same length and frame times as the neural slice.

ii. 
```python
trial_t = ophys_t[left:right]
neural = neural_full[:, left:right].astype(np.float32, copy=False)
run_aligned = interp_to_ophys(run_t, run_v, trial_t)
```

iii. The notes’ key decision is to use ophys timestamps as the common temporal basis for neural and behavioral variables.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `exp.eye_tracking`. The code prefers `pupil_area`, converting it to diameter as `2 * sqrt(area / pi)`. If `pupil_area` is absent, it falls back to a geometric width-height proxy from `pupil_width` and `pupil_height`. It also uses `likely_blink`.

ii. 
```python
def pupil_diameter_series(eye_tracking_df):
    et = eye_tracking_df.copy()
    blink = et['likely_blink'].fillna(False).to_numpy(dtype=bool) if 'likely_blink' in et.columns else np.zeros(len(et), dtype=bool)
    if 'pupil_area' in et.columns:
        area = et['pupil_area'].to_numpy(dtype=np.float64)
        diam = 2.0 * np.sqrt(area / math.pi)
    elif 'pupil_width' in et.columns and 'pupil_height' in et.columns:
        diam = np.sqrt(et['pupil_width'].to_numpy(dtype=np.float64) * et['pupil_height'].to_numpy(dtype=np.float64))
    else:
        raise KeyError('No pupil area/width-height columns available')
```

iii. Step 5 in the notes says the AI intentionally wanted a pupil-diameter proxy and would “prefer geometric diameter from area” when available, rather than using a raw width value directly.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code derives a diameter-like quantity from eye-tracking geometry, masks blink frames to `NaN`, interpolates the resulting series onto each trial’s ophys timestamps, and then discretizes globally into five quantile bins.

ii. 
```python
diam[blink] = np.nan
return et['timestamps'].to_numpy(dtype=np.float64), diam.astype(np.float32)
```

```python
pupil_t, pupil_v = pupil_diameter_series(exp.eye_tracking)
...
pupil_aligned = interp_to_ophys(pupil_t, pupil_v, trial_t)
...
pupil_edges = compute_bin_edges(np.concatenate(all_pupil))
...
pupil_bin = digitize_with_edges(tr['pupil_raw'], pupil_edges)
```

iii. The notes justify this as matching the requested “pupil diameter” variable while still using the SDK’s blink annotations and the common ophys time base.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The thresholding is the same as for running speed: five global quantile bins, duplicate edges nudged upward, and non-finite aligned samples replaced by the trial’s median valid bin (or 0 if nothing is valid).

ii. 
```python
def compute_bin_edges(values, n_bins=5):
    ...
    edges = np.quantile(values, qs)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
```

```python
def digitize_with_edges(values, edges):
    ...
    if np.any(bad):
        finite = np.where(np.isfinite(values))[0]
        fill = int(np.median(out[finite])) if finite.size else 0
        out[bad] = fill
```

iii. The notes explain the global percentile-binning choice; the particular missing-value fill rule is only evident from the code.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is aligned by interpolating the eye-tracking series directly onto the same per-trial ophys timestamps used for the neural slice.

ii. 
```python
trial_t = ophys_t[left:right]
neural = neural_full[:, left:right].astype(np.float32, copy=False)
pupil_aligned = interp_to_ophys(pupil_t, pupil_v, trial_t)
```

iii. Step 5 in the notes says all outputs should be aligned to native ophys timestamps within each trial.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject` in `exp.trials`, after restricting to kept Go/Catch parent trials.

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

```python
trials['trial_outcome_idx'] = trials.apply(get_trial_outcome, axis=1)
trials = trials.loc[trials['trial_outcome_idx'].notna()].copy()
```

iii. Step 5 of the notes maps those four canonical SDK outcome flags onto the trial-outcome output.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome flags are converted to integer labels 0-3, stored on the parent trial, and then repeated across every ophys frame of each image-presentation trial that belongs to that parent trial.

ii. 
```python
trial_outcome_map = trials['trial_outcome_idx'].to_dict()
...
outcome = int(trial_outcome_map[parent_id])
```

```python
out = np.vstack([
    np.full(T, image_to_idx[tr['image_name']], dtype=np.int64),
    tr['image_change'],
    run_bin,
    pupil_bin,
    np.full(T, tr['trial_outcome'], dtype=np.int64),
]).astype(np.int64)
```

iii. The notes describe trial outcome as a static per-trial category and explicitly say a per-trial constant representation is acceptable.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The final code handles several kinds of imperfect data: missing booleans are treated as `False`; blink frames are masked to `NaN`; interpolation with fewer than two finite source points returns all-`NaN`; non-finite running/pupil values are binned by filling with the trial’s median valid bin; stimulus rows with missing `trials_id`, `image_name`, `start_time`, or `end_time` are dropped; image-presentation intervals with fewer than 2 frames are skipped; and sessions with fewer than 2 kept trials are skipped. The final script does not have a session-level `try/except` around loading.

ii. 
```python
keep = (trials['go'].fillna(False) | trials['catch'].fillna(False))
keep &= ~trials['aborted'].fillna(False)
keep &= ~trials['auto_rewarded'].fillna(False)
```

```python
if good.sum() < 2:
    return np.full(dst_t.shape, np.nan, dtype=np.float32)
```

```python
diam[blink] = np.nan
```

```python
bad = ~np.isfinite(values)
if np.any(bad):
    finite = np.where(np.isfinite(values))[0]
    fill = int(np.median(out[finite])) if finite.size else 0
    out[bad] = fill
```

```python
stim = stim[stim['trials_id'].notna()].copy()
...
stim = stim[stim['start_time'].notna() & stim['end_time'].notna()].copy()
...
if right - left < 2:
    continue
```

iii. The notes and trajectory explicitly discuss NA-safe boolean handling, blink masking, and eliminating unlabeled gray/omitted periods. `CONVERSION_NOTES.md` earlier mentions skipping failed sessions, but that behavior is not present in the final script.

## 9-a. What are the most time-consuming steps of the code?

i. The expensive parts are loading each NWB into a `BehaviorOphysExperiment` and then iterating through the large number of stimulus-presentation rows to slice neural and behavioral data. The final code even adds multiprocessing for the full run because this pass is slow.

ii. 
```python
exp = BehaviorOphysExperiment.from_nwb_path(str(nwb_path))
```

```python
for stim_id, parent_id, start, stop, img_name, is_change in zip(
    stim_ids, stim_trial_ids, stim_start, stim_end, stim_image_name, stim_is_change
):
    ...
```

```python
use_parallel = (not sample) and len(files) > 4
if use_parallel:
    n_workers = min(max((os.cpu_count() or 2) // 2, 2), 8)
    with Pool(processes=n_workers) as pool:
```

iii. Step 6/7 of the notes and trajectory step 26 say the first full-version bottleneck was the combination of per-experiment loading and iterating over many stimulus presentations, which drove a runtime estimate of roughly 104 minutes before optimization.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The main candidate is the Python loop over stimulus presentations in `process_experiment`. The AI’s own trajectory also identifies earlier pandas row iteration as an inefficiency and says it sped things up by precomputing NumPy arrays and iterating over zipped arrays instead, but the final code still has a per-presentation Python loop.

ii. 
```python
stim_ids = stim.index.to_numpy()
stim_trial_ids = stim['trials_id'].to_numpy()
stim_start = stim['start_time'].to_numpy(dtype=np.float64)
stim_end = stim['end_time'].to_numpy(dtype=np.float64)
stim_image_name = stim['image_name'].astype(str).to_numpy()
```

```python
for stim_id, parent_id, start, stop, img_name, is_change in zip(
    stim_ids, stim_trial_ids, stim_start, stim_end, stim_image_name, stim_is_change
):
```

iii. Trajectory step 26 explicitly says the code should reduce “Python/DataFrame row iteration overhead in `process_experiment`” by using NumPy arrays instead of `iterrows()`. That explains both what the AI thought was slow and what it changed.

## 9-c. What processing does the code repeat multiple times?

i. There is not much repeated heavy processing. The AI intentionally avoids a second raw-data load by collecting raw aligned running/pupil arrays during the first pass and computing global bin edges from those saved arrays. Minor repeated work remains, such as scanning the NWB tree twice (`choose_files()` and the log message) and digitizing running/pupil only after storing the raw aligned versions.

ii. 
```python
def choose_files(sample=False):
    files = list_nwb_files()
    return files[:2] if sample else files
```

```python
files = choose_files(sample=sample)
print(f'found {len(list_nwb_files())} nwb files; processing {len(files)}')
```

```python
session['running_raw_all'].append(run_aligned)
session['pupil_raw_all'].append(pupil_aligned)
...
run_edges = compute_bin_edges(np.concatenate(all_run))
pupil_edges = compute_bin_edges(np.concatenate(all_pupil))
...
run_bin = digitize_with_edges(tr['running_raw'], run_edges)
pupil_bin = digitize_with_edges(tr['pupil_raw'], pupil_edges)
```

iii. Step 6 in `CONVERSION_NOTES.md` says “one-pass global bin edge collection” was added specifically to avoid reloading the raw data a second time.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script carries a fair amount of intermediate state that is not written to the final dataset: full `meta`, per-trial `start_time`, `stop_time`, `ophys_timestamps`, `go`, `catch`, and the raw aligned `running_raw` and `pupil_raw` arrays. It also keeps `running_raw_all` and `pupil_raw_all` only to estimate global bin edges, defines an unused helper `build_image_labels_for_trial`, and defines an unused `PROJECT_METADATA` path.

ii. 
```python
PROJECT_METADATA = DATASET_ROOT / 'project_metadata'
```

```python
def build_image_labels_for_trial(trial_timestamps, stim_df, image_to_idx):
    ...
```

```python
session = {
    'session_id': int(meta['ophys_experiment_id']),
    'subject': str(meta['mouse_id']),
    'region': str(meta['targeted_structure']),
    'meta': dict(meta),
    'trials': [],
    'running_raw_all': [],
    'pupil_raw_all': [],
}
```

```python
session['trials'].append({
    'trial_id': ...,
    'start_time': float(start),
    'stop_time': float(stop),
    'ophys_timestamps': trial_t.astype(np.float32),
    'neural': neural,
    'image_name': img_name,
    'image_change': np.full(right - left, int(is_change), dtype=np.int64),
    'running_raw': run_aligned,
    'pupil_raw': pupil_aligned,
    'trial_outcome': outcome,
    'go': bool(trial_go_map[parent_id]),
    'catch': bool(trial_catch_map[parent_id]),
})
```

iii. The notes mention keeping raw aligned arrays for plotting and for one-pass bin-edge computation. Those intermediates are useful for conversion, but they are discarded from the final output and so represent extra work/storage compared with the downstream pickle.
