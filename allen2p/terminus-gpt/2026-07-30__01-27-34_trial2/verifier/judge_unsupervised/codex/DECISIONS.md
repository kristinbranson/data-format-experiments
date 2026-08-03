# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script hard-codes a dataset root, recursively finds every `.nwb` file under that root, and loads each file with `BehaviorOphysExperiment.from_nwb_path`. In sample mode it only keeps the first two NWBs.

ii. 
```python
DATASET_ROOT = Path('data/visual-behavior-ophys-1.1.0')

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

iii. The agent’s notes say it would “use `BehaviorOphysExperiment` per NWB/ophys experiment,” because the AllenSDK object already exposes aligned trials, stimulus, neural, running, and eye-tracking tables. The trajectory shows it deliberately chose direct NWB loading instead of building through a project cache.

## 1-b. How are the data split into subjects?

i. Subjects are defined by `metadata['mouse_id']`. The code collects one subject string per processed session, builds the sorted unique subject list, and stores a per-session `subject_idx`.

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
'subjects': subjects,
'subject_idx': np.array([subject_to_idx[s['subject']] for s in processed_sessions], dtype=np.int64),
```

iii. In the planning notes the agent wrote “`metadata['mouse_id']` -> subjects / `subject_idx`” and “one subject per experiment session.”

## 1-c. How are the data split into sessions?

i. Each NWB file / `ophys_experiment_id` is treated as one decoder session. The code does not merge multiple experiments that share an `ophys_session_id`.

ii.
```python
def process_experiment(nwb_path):
    ...
    session = {
        'session_id': int(meta['ophys_experiment_id']),
        'subject': str(meta['mouse_id']),
        'region': str(meta['targeted_structure']),
```

```python
for i, f in enumerate(files, start=1):
    print(f'[{i}/{len(files)}] loading {f}')
    sess = process_experiment(f)
    if len(sess['trials']) >= 2:
        processed.append(sess)
```

iii. The notes explicitly justify this as “Session unit = ophys experiment” because each experiment has its own neural population and aligned behavior tables.

## 1-d. How are the data split into trials?

i. The final code uses image presentations, not SDK trial rows, as decoder trials. It filters `stimulus_presentations`, then every kept presentation becomes one trial spanning that presentation’s `start_time:end_time` on the ophys clock.

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
    trial_t = ophys_t[left:right]
```

iii. The trajectory shows this was a deliberate revision. After sample verification exposed `image_identity == -1` during grey periods, the agent changed from full `trials.start_time:stop_time` windows to image-presentation windows so every trial would have a valid image label and better match the paper’s “image-by-image” analysis.

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering is two-stage. First, `valid_trials_df` keeps only Go or Catch trials, removes `aborted` and `auto_rewarded`, and drops trials with no mapped outcome. Second, the code only keeps stimulus-presentation segments linked to those valid trials, requires active/non-omitted presentations with valid image/timing fields, and skips segments with fewer than 2 ophys frames.

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
stim = stim[stim['trials_id'].isin(valid_trial_ids)].copy()
if 'active' in stim.columns:
    stim = stim[stim['active'].fillna(False)].copy()
if 'omitted' in stim.columns:
    stim = stim[~stim['omitted'].fillna(False)].copy()
...
if right - left < 2:
    continue
```

iii. The Go/Catch, aborted, and auto-rewarded filters come directly from the task instructions and the agent’s notes. The extra stimulus filters were justified in the trajectory as needed to exclude grey / unlabeled intervals and ensure valid image labels.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `exp.events`, specifically the `events` column in the AllenSDK events table.

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
ophys_t = np.asarray(exp.ophys_timestamps, dtype=np.float64)
neural_full = extract_events_matrix(exp.events)
```

iii. The notes repeatedly justify this by citing the methods text: “For all analysis of neural data we used the detected calcium events.” The trajectory shows the agent explicitly resolved an events-vs-dF/F discrepancy in favor of events.

## 2-b. How is the `neural` data processed?

i. The code stacks each ROI’s events trace into a session-wide neuron-by-time matrix, casts to `float32`, and then slices columns corresponding to each kept stimulus presentation. There is no extra denoising, normalization, smoothing, or temporal rebinning.

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

iii. The notes say the goal was to “use detected calcium `events` as neural signal” and “use direct slicing on native ophys timestamps,” preserving the SDK-provided signal rather than recomputing anything.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit neuron / ROI quality-control filter in the final script. Every row in `exp.events` is included.

ii.
```python
arrs = [np.asarray(x, dtype=np.float32) for x in events_df[event_col].values]
...
return np.stack(arrs, axis=0)
```

iii. The notes mention SDK cell/ROI metadata as a possible curation source, but the final code never uses `cell_specimen_table` or any ROI-validity flag. The implemented decision is effectively to trust the event table as provided.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to native ophys timestamps and then restricted to each stimulus-presentation interval. In other words, the “trial” starts at presentation `start_time` and ends at presentation `end_time` on the ophys clock.

ii.
```python
left = np.searchsorted(ophys_t, start, side='left')
right = np.searchsorted(ophys_t, stop, side='left')
...
trial_t = ophys_t[left:right]
neural = neural_full[:, left:right].astype(np.float32, copy=False)
```

```python
'metadata': {
    ...
    'temporal_alignment_event': 'native ophys timestamps within each trial defined by SDK trial start/stop times',
    'off_start': 0.0,
    'off_end': None,
```

iii. The task required alignment on ophys timestamps. The trajectory shows the agent later tightened the window to each image presentation because full trial windows created unlabeled grey periods.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The code keeps the native ophys frame times. It does not rebin or resample neural data to a coarser temporal grid. However, it leaves `metadata['time_bin_size']` as `None` instead of computing the frame interval.

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

iii. The notes state “Temporal basis = ophys timestamps” and do not mention any rebinning step. The agent appears to have intended the native frame rate to stand in for the time bin size.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity comes from `stimulus_presentations['image_name']`.

ii.
```python
stim_image_name = stim['image_name'].astype(str).to_numpy()
```

```python
image_names = sorted({tr['image_name'] for s in processed_sessions for tr in s['trials']})
image_to_idx = {name: i for i, name in enumerate(image_names)}
```

iii. The notes say “`stimulus_presentations.image_name` -> output[0] image identity” because the task asks for “the image presented during the non-grey screen.”

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The code collects all unique image names globally, assigns each name an integer code, and then fills every time point of a trial with that trial’s single image code. Because each trial is one stimulus presentation, image identity is constant within trial.

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

iii. In the trajectory the agent says the image-presentation trialing was chosen specifically so `image_identity` would always be valid and naturally constant within each segment.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is aligned by using the same ophys-timestamp slice as the neural trial and repeating the image code across all ophys frames in that slice.

ii.
```python
trial_t = ophys_t[left:right]
neural = neural_full[:, left:right].astype(np.float32, copy=False)
...
out = np.vstack([
    np.full(T, image_to_idx[tr['image_name']], dtype=np.int64),
```

iii. The agent’s justification was that all outputs should be represented on the ophys clock, and that using the presentation interval as the trial window removes any unlabeled grey-screen frames.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from `stimulus_presentations['is_change']`.

ii.
```python
if 'is_change' in stim.columns:
    stim_is_change = stim['is_change'].fillna(False).astype(bool).to_numpy()
else:
    stim_is_change = np.zeros(len(stim), dtype=bool)
```

iii. The notes map “`stimulus_presentations.is_change` -> output[1] image change.”

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code converts `is_change` to `0/1` and fills the whole image-presentation trial with that constant value.

ii.
```python
session['trials'].append({
    ...
    'image_change': np.full(right - left, int(is_change), dtype=np.int64),
```

iii. This follows from the agent’s shift to image-presentation trials: once a trial is exactly one presentation interval, `is_change` becomes a constant label for that whole segment.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is binary with categories `no_change` and `change`.

ii.
```python
IMAGE_CHANGE_VALUES = ['no_change', 'change']
```

```python
'output_values': [
    image_values,
    IMAGE_CHANGE_VALUES,
```

iii. This matches the task statement that image change should be binary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. It is aligned by repeating the presentation’s `is_change` flag across the same ophys frames used for the neural slice.

ii.
```python
trial_t = ophys_t[left:right]
neural = neural_full[:, left:right].astype(np.float32, copy=False)
...
'image_change': np.full(right - left, int(is_change), dtype=np.int64),
```

iii. The agent’s rationale was the same as for image identity: keep all labels on the ophys clock and restrict each trial to one presentation interval.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `exp.running_speed['timestamps']` and `exp.running_speed['speed']`.

ii.
```python
run_t = exp.running_speed['timestamps'].to_numpy(dtype=np.float64)
run_v = exp.running_speed['speed'].to_numpy(dtype=np.float32)
```

iii. The notes explicitly map `running_speed.speed` plus `running_speed.timestamps` to the decoder output.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The raw running-speed trace is linearly interpolated onto each trial’s ophys timestamps. The code stores the continuous aligned trace first, then later discretizes it.

ii.
```python
def interp_to_ophys(src_t, src_v, dst_t):
    ...
    return np.interp(dst_t, src_t, src_v, left=np.nan, right=np.nan).astype(np.float32)
```

```python
run_aligned = interp_to_ophys(run_t, run_v, trial_t)
...
session['running_raw_all'].append(run_aligned)
```

iii. The agent’s notes say “align based on ophys timestamps” and “compute percentile bin edges over all valid timepoints in processed sessions.”

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The code computes 5 equal-frequency global percentile bins over all aligned running-speed values from all processed sessions, then digitizes each trial into those bins.

ii.
```python
def compute_bin_edges(values, n_bins=5):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    ...
    edges = np.quantile(values, qs)
```

```python
run_edges = compute_bin_edges(np.concatenate(all_run))
...
run_bin = digitize_with_edges(tr['running_raw'], run_edges)
```

iii. The notes justify global binning as making categories comparable across sessions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Continuous running speed is first interpolated to the same `trial_t` ophys timestamps used for the neural slice, then binned pointwise.

ii.
```python
trial_t = ophys_t[left:right]
neural = neural_full[:, left:right].astype(np.float32, copy=False)
run_aligned = interp_to_ophys(run_t, run_v, trial_t)
```

```python
run_bin = digitize_with_edges(tr['running_raw'], run_edges)
```

iii. The agent consistently justified this as the task-required “align based on ophys timestamp.”

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil output is derived from eye-tracking timestamps plus either `pupil_area` or, if that is absent, `pupil_width` and `pupil_height`. It also uses `likely_blink` to mask bad samples.

ii.
```python
blink = et['likely_blink'].fillna(False).to_numpy(dtype=bool) if 'likely_blink' in et.columns else np.zeros(len(et), dtype=bool)
if 'pupil_area' in et.columns:
    area = et['pupil_area'].to_numpy(dtype=np.float64)
    diam = 2.0 * np.sqrt(area / math.pi)
elif 'pupil_width' in et.columns and 'pupil_height' in et.columns:
    diam = np.sqrt(et['pupil_width'].to_numpy(dtype=np.float64) * et['pupil_height'].to_numpy(dtype=np.float64))
```

iii. The Step 5 notes say the agent wanted a “pupil diameter proxy,” preferring a diameter derived from area when possible and otherwise falling back to width/height.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The code converts area to diameter with `2*sqrt(area/pi)` or uses the geometric mean of width and height, masks likely blinks as `NaN`, and linearly interpolates the result to ophys timestamps.

ii.
```python
diam[blink] = np.nan
return et['timestamps'].to_numpy(dtype=np.float64), diam.astype(np.float32)
```

```python
pupil_t, pupil_v = pupil_diameter_series(exp.eye_tracking)
...
pupil_aligned = interp_to_ophys(pupil_t, pupil_v, trial_t)
```

iii. The justification in the notes is that the task asks for pupil diameter, so the code should turn the available pupil measurement into a diameter-like quantity and mask blinks before alignment/binning.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Like running speed, pupil is discretized into 5 global percentile bins computed across all processed sessions.

ii.
```python
pupil_edges = compute_bin_edges(np.concatenate(all_pupil))
...
pupil_bin = digitize_with_edges(tr['pupil_raw'], pupil_edges)
```

iii. The notes use the same justification as running speed: global equal-percentile bins create comparable categorical outputs across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. The pupil trace is interpolated to the same ophys timestamps used for the neural slice and then binned pointwise.

ii.
```python
pupil_t, pupil_v = pupil_diameter_series(exp.eye_tracking)
...
pupil_aligned = interp_to_ophys(pupil_t, pupil_v, trial_t)
```

```python
pupil_bin = digitize_with_edges(tr['pupil_raw'], pupil_edges)
```

iii. The notes and trajectory justify this as part of the general “all outputs live on the ophys clock” decision.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in `exp.trials`.

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

iii. The notes map the trial-outcome flags in the SDK trial table directly to the static decoder output.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code maps the mutually exclusive outcome booleans to integer class IDs, drops trials that do not map to one of the four categories, and then repeats the class ID across all frames of a kept trial.

ii.
```python
trials['trial_outcome_idx'] = trials.apply(get_trial_outcome, axis=1)
trials = trials.loc[trials['trial_outcome_idx'].notna()].copy()
```

```python
out = np.vstack([
    ...
    np.full(T, tr['trial_outcome'], dtype=np.int64),
]).astype(np.int64)
```

iii. The justification is directly from the task: trial outcome should be a static per-trial categorical output.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The code uses several ad hoc safeguards: `fillna(False)` or explicit `pd.isna` checks for booleans, skips missing image/timing fields, masks blinks in pupil data, returns all-`NaN` aligned traces when there are fewer than 2 valid source samples, ignores non-finite values when computing bin edges, and fills non-finite discretized values with the median finite bin.

ii.
```python
omitted_val = srow.get('omitted', False)
omitted = False if pd.isna(omitted_val) else bool(omitted_val)
...
image_change[mask] = int(False if pd.isna(is_change_val) else bool(is_change_val))
```

```python
good = np.isfinite(src_t) & np.isfinite(src_v)
if good.sum() < 2:
    return np.full(dst_t.shape, np.nan, dtype=np.float32)
```

```python
values = values[np.isfinite(values)]
...
bad = ~np.isfinite(values)
if np.any(bad):
    finite = np.where(np.isfinite(values))[0]
    fill = int(np.median(out[finite])) if finite.size else 0
    out[bad] = fill
```

iii. The trajectory shows at least one explicit bug fix here: the first version crashed on `pd.NA` in `omitted` / `is_change`, and the agent patched the code to handle missing booleans safely. The notes also say the script should “handle missing data appropriately” with sensible defaults.

## 9-a. What are the most time-consuming steps of the code?

i. The most expensive work is loading each NWB into a `BehaviorOphysExperiment`, extracting per-session arrays, and iterating over every kept stimulus presentation to slice neural and behavioral data. This was slow enough that the agent added multiprocessing for full runs.

ii.
```python
def process_experiment(nwb_path):
    t0 = time.time()
    exp = BehaviorOphysExperiment.from_nwb_path(str(nwb_path))
    ...
    for stim_id, parent_id, start, stop, img_name, is_change in zip(
        stim_ids, stim_trial_ids, stim_start, stim_end, stim_image_name, stim_is_change
    ):
```

```python
use_parallel = (not sample) and len(files) > 4
if use_parallel:
    n_workers = min(max((os.cpu_count() or 2) // 2, 2), 8)
    with Pool(processes=n_workers) as pool:
        for i, sess in enumerate(pool.imap_unordered(process_experiment, files), start=1):
```

iii. The notes and trajectory explicitly identify serial session loading plus per-stimulus iteration as the main bottlenecks, and document the later multiprocessing optimization.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still has several Python-level loops that could be vectorized: `trials.apply(get_trial_outcome, axis=1)`, the per-stimulus `for ... zip(...)` loop in `process_experiment`, and the per-trial assembly loop that digitizes and stacks outputs one trial at a time.

ii.
```python
trials['trial_outcome_idx'] = trials.apply(get_trial_outcome, axis=1)
```

```python
for stim_id, parent_id, start, stop, img_name, is_change in zip(
    stim_ids, stim_trial_ids, stim_start, stim_end, stim_image_name, stim_is_change
):
```

```python
for tr in s['trials']:
    T = tr['neural'].shape[1]
    run_bin = digitize_with_edges(tr['running_raw'], run_edges)
    pupil_bin = digitize_with_edges(tr['pupil_raw'], pupil_edges)
    out = np.vstack([
```

iii. The agent itself notes that the implementation “iterates over stimulus presentations per trial” and originally targeted that loop during optimization before adding multiprocessing.

## 9-c. What processing does the code repeat multiple times?

i. The script repeats file discovery (`list_nwb_files()` is called in both `choose_files()` and again for the status print), stores running and pupil traces twice per session (once inside each trial and again in `running_raw_all` / `pupil_raw_all` for global binning), and digitizes running/pupil separately for every trial after already caching all continuous aligned values.

ii.
```python
def choose_files(sample=False):
    files = list_nwb_files()
    return files[:2] if sample else files
...
files = choose_files(sample=sample)
print(f'found {len(list_nwb_files())} nwb files; processing {len(files)}')
```

```python
'running_raw_all': [],
'pupil_raw_all': [],
...
session['running_raw_all'].append(run_aligned)
session['pupil_raw_all'].append(pupil_aligned)
...
'running_raw': run_aligned,
'pupil_raw': pupil_aligned,
```

iii. The notes mention a “one-pass global bin edge collection,” but the final code still duplicates some bookkeeping to make that one pass possible.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes and stores several intermediate fields that never reach `converted_data.pkl`: `trial_id`, `start_time`, `stop_time`, `ophys_timestamps`, raw running/pupil traces, and the `go`/`catch` flags inside each session dictionary. It also leaves behind an unused helper, `build_image_labels_for_trial`, from the earlier full-trial labeling approach.

ii.
```python
def build_image_labels_for_trial(trial_timestamps, stim_df, image_to_idx):
    image_idx = np.full(trial_timestamps.shape, -1, dtype=np.int64)
    image_change = np.zeros(trial_timestamps.shape, dtype=np.int64)
    ...
    return image_idx, image_change
```

```python
session['trials'].append({
    'trial_id': int(stim_id) if isinstance(stim_id, (int, np.integer)) else len(session['trials']),
    'start_time': float(start),
    'stop_time': float(stop),
    'ophys_timestamps': trial_t.astype(np.float32),
    ...
    'running_raw': run_aligned,
    'pupil_raw': pupil_aligned,
    ...
    'go': bool(trial_go_map[parent_id]),
    'catch': bool(trial_catch_map[parent_id]),
})
```

iii. These fields were useful for debugging, plotting, and computing global bins, but downstream analyses only consume the final `neural`, `input`, `output`, subject, and region structures.
