# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads NWB files directly from the local data directory using `BehaviorOphysExperiment.from_nwb_path()`. It discovers all NWB files by recursively globbing `data/visual-behavior-ophys-1.1.0/**/*.nwb`. In `--sample` mode, only the first 2 NWB files are processed; in `--full` mode, all are processed. For full runs with >4 files, multiprocessing is used.

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

def process_experiment(nwb_path):
    exp = BehaviorOphysExperiment.from_nwb_path(str(nwb_path))
    # ...
```

iii. The AI chose to load NWB files directly rather than using the SDK cache API. The CONVERSION_NOTES.md documents that the dataset is stored under `data/visual-behavior-ophys-1.1.0` with NWB files per experiment.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `mouse_id` field in each experiment's metadata. Unique subjects are collected across all processed sessions.

ii.
```python
subjects = sorted({s['subject'] for s in processed_sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
# In process_experiment:
session = {
    'subject': str(meta['mouse_id']),
    # ...
}
```

iii. The `mouse_id` from experiment metadata uniquely identifies each animal.

## 1-c. How are the data split into sessions?

i. Each NWB file (ophys experiment) is treated as a separate decoder session. Unlike the human reference which groups experiments by `ophys_session_id` and merges multiple imaging planes, the AI treats each experiment independently. Each session therefore contains neurons from only one imaging plane.

ii.
```python
def process_experiment(nwb_path):
    exp = BehaviorOphysExperiment.from_nwb_path(str(nwb_path))
    # ...
    session = {
        'session_id': int(meta['ophys_experiment_id']),
        'subject': str(meta['mouse_id']),
        'region': str(meta['targeted_structure']),
        # ...
    }
```

iii. The CONVERSION_NOTES.md states: "Treat each ophys experiment NWB as one decoder session because neural populations are experiment-specific." Each NWB file corresponds to one imaging plane in one session.

## 1-d. How are the data split into trials?

i. The AI uses individual stimulus presentations (approximately 750ms image flashes) as trials, not the full behavioral trials from the SDK's trials table. It first filters trials to go/catch, non-aborted, non-auto-rewarded, then iterates over stimulus presentations belonging to those valid trials. Each stimulus presentation becomes one "trial" in the output.

ii.
```python
def valid_trials_df(trials):
    keep = (trials['go'].fillna(False) | trials['catch'].fillna(False))
    keep &= ~trials['aborted'].fillna(False)
    keep &= ~trials['auto_rewarded'].fillna(False)
    trials = trials.loc[keep].copy()
    trials['trial_outcome_idx'] = trials.apply(get_trial_outcome, axis=1)
    trials = trials.loc[trials['trial_outcome_idx'].notna()].copy()
    return trials

# In process_experiment:
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

iii. The AI's CONVERSION_NOTES.md discusses using stimulus presentations as the trial segmentation unit, noting that "Decoder analyses in the paper were image-by-image, suggesting stimulus-aligned segmentation is the natural unit for trialing." This matches the paper's analysis approach of 750ms image presentation intervals.

## 1-e. How are trials filtered based on quality controls?

i. The AI first filters SDK trials to go/catch, excluding aborted and auto-rewarded. Then it further filters stimulus presentations to those that are active, non-omitted, have a valid image_name, and belong to valid trials. Presentations with fewer than 2 ophys frames are skipped. Sessions with fewer than 2 valid presentations are excluded.

ii.
```python
# Trial-level filtering:
keep = (trials['go'].fillna(False) | trials['catch'].fillna(False))
keep &= ~trials['aborted'].fillna(False)
keep &= ~trials['auto_rewarded'].fillna(False)

# Stimulus presentation filtering:
stim = stim[stim['trials_id'].isin(valid_trial_ids)].copy()
if 'active' in stim.columns:
    stim = stim[stim['active'].fillna(False)].copy()
if 'omitted' in stim.columns:
    stim = stim[~stim['omitted'].fillna(False)].copy()
stim = stim[stim['image_name'].notna()].copy()
stim = stim[stim['start_time'].notna() & stim['end_time'].notna()].copy()

# Frame count check:
if right - left < 2:
    continue

# Session-level check:
if len(sess['trials']) >= 2:
    processed.append(sess)
```

iii. Aborted and auto-rewarded trials are excluded per instructions. Omitted image presentations are filtered out because they contain no visual stimulus. Active-only filtering ensures only task-engaged presentations are included.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `events` (detected calcium events), accessed via `exp.events`. This is a deconvolved/detected event trace rather than the raw dF/F fluorescence.

ii.
```python
def extract_events_matrix(events_df):
    event_col = 'events'
    arrs = [np.asarray(x, dtype=np.float32) for x in events_df[event_col].values]
    return np.stack(arrs, axis=0)

# In process_experiment:
neural_full = extract_events_matrix(exp.events)
```

iii. The CONVERSION_NOTES.md states: "Neural signal = detected events: methods.txt explicitly states neural analyses used detected calcium events, so use events rather than dF/F." This matches the paper's methodology.

## 2-b. How is the `neural` data processed?

i. Events traces are extracted per cell, stacked into a (n_neurons, T) matrix, and sliced per stimulus presentation. No additional processing (filtering, normalization) is applied.

ii.
```python
neural_full = extract_events_matrix(exp.events)  # (n_neurons, T)
# Per trial:
neural = neural_full[:, left:right].astype(np.float32, copy=False)
```

iii. The detected calcium events are already processed by the Allen SDK pipeline (deconvolution from fluorescence traces).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional quality filtering is applied to neurons. All cells present in the experiment's events data are included.

ii. N/A - no filtering code exists.

iii. The SDK already applies ROI filtering and quality control during preprocessing.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's neural data is aligned to the stimulus presentation start time. The ophys frames between the presentation's `start_time` and `end_time` are extracted using `np.searchsorted`.

ii.
```python
left = np.searchsorted(ophys_t, start, side='left')
right = np.searchsorted(ophys_t, stop, side='left')
trial_t = ophys_t[left:right]
neural = neural_full[:, left:right].astype(np.float32, copy=False)
```

iii. Since each "trial" is a stimulus presentation, the alignment is to the image onset time. The temporal window is short (~750ms, typically 8-9 ophys frames at ~11 Hz).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The data is kept at the native ophys frame rate (~11 Hz). The `time_bin_size` in metadata is set to `None`.

ii.
```python
'metadata': {
    'time_bin_size': None,
    # ...
}
```

iii. The ophys timestamps are at a consistent frame rate. No resampling is applied.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `image_name` column of the `stimulus_presentations` table. Since each "trial" is a single stimulus presentation, the image identity is a constant value per trial (not time-varying).

ii.
```python
stim_image_name = stim['image_name'].astype(str).to_numpy()
# Per trial in the loop:
session['trials'].append({
    'image_name': img_name,
    # ...
})
# In assemble_dataset:
np.full(T, image_to_idx[tr['image_name']], dtype=np.int64),
```

iii. The AI uses the stimulus_presentations table directly. Since each trial is one image flash, image identity is constant within the trial.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global sorted mapping built from all unique image names across all sessions.

ii.
```python
image_names = sorted({tr['image_name'] for s in processed_sessions for tr in s['trials']})
image_to_idx = {name: i for i, name in enumerate(image_names)}
# Per trial:
np.full(T, image_to_idx[tr['image_name']], dtype=np.int64),
```

iii. A global sorted mapping ensures consistent integer codes across sessions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is broadcast as a constant across all ophys frames within the stimulus presentation window. It uses the same frame indices as the neural data.

ii.
```python
np.full(T, image_to_idx[tr['image_name']], dtype=np.int64),
```

iii. Since each trial is a single stimulus presentation with one image, alignment is trivial -- the image is constant throughout the trial window.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column of the `stimulus_presentations` table. It is a constant binary value per trial: 1 if the presentation is a change, 0 otherwise.

ii.
```python
if 'is_change' in stim.columns:
    stim_is_change = stim['is_change'].fillna(False).astype(bool).to_numpy()
else:
    stim_is_change = np.zeros(len(stim), dtype=bool)

# Per trial:
'image_change': np.full(right - left, int(is_change), dtype=np.int64),
```

iii. The `is_change` flag from stimulus_presentations indicates whether this image flash is the change image in its parent trial.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The boolean `is_change` is cast to integer (0 or 1) and broadcast as a constant across all frames in the presentation.

ii.
```python
'image_change': np.full(right - left, int(is_change), dtype=np.int64),
```

iii. No additional processing beyond type conversion.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary (0 or 1). No thresholding is needed.

ii. See 4-a.

iii. N/A

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Image change is broadcast as a constant across the same ophys frame window as the neural data.

ii. See 4-a.

iii. Same frame-level window as neural data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `exp.running_speed`, using the `timestamps` and `speed` columns.

ii.
```python
run_t = exp.running_speed['timestamps'].to_numpy(dtype=np.float64)
run_v = exp.running_speed['speed'].to_numpy(dtype=np.float32)
```

iii. The `running_speed` attribute is the SDK's standard interface for locomotion data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is interpolated to the ophys timebase within each trial using `np.interp` (via the `interp_to_ophys` helper), then discretized into 5 percentile-based bins computed globally across all sessions. NaN values are filled with the median bin value.

ii.
```python
def interp_to_ophys(src_t, src_v, dst_t):
    good = np.isfinite(src_t) & np.isfinite(src_v)
    src_t = src_t[good]
    src_v = src_v[good]
    order = np.argsort(src_t)
    src_t = src_t[order]
    src_v = src_v[order]
    return np.interp(dst_t, src_t, src_v, left=np.nan, right=np.nan).astype(np.float32)

# Per trial:
run_aligned = interp_to_ophys(run_t, run_v, trial_t)

# Global binning:
run_edges = compute_bin_edges(np.concatenate(all_run))
run_bin = digitize_with_edges(tr['running_raw'], run_edges)
```

iii. Linear interpolation preserves the signal shape. Global percentile-based binning ensures balanced class counts.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 percentile bins using `np.quantile` for bin edges, then `np.digitize`. Duplicate bin edges are deduplicated by adding 1e-6. NaN values are filled with the median of the other digitized values (or 0 if all NaN). Values are clipped to valid range.

ii.
```python
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

iii. Percentile-based binning with deduplication ensures non-degenerate bin edges.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to the ophys timestamps within each trial window, using the same frame indices as the neural data.

ii.
```python
run_aligned = interp_to_ophys(run_t, run_v, trial_t)
```

iii. Both neural and running data are aligned to the same ophys frame window.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `exp.eye_tracking`. The AI preferentially uses `pupil_area` (computing diameter as `2*sqrt(area/pi)`), falling back to the geometric mean of `pupil_width` and `pupil_height` (`sqrt(width*height)`). Blink frames (where `likely_blink` is True) are set to NaN.

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

iii. The AI derives actual diameter from pupil area for a more principled measurement. Blink frames are masked to NaN before interpolation.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Same approach as running speed: interpolate to ophys timebase within each trial, then discretize into 5 global percentile bins. NaN values filled with median bin.

ii.
```python
pupil_t, pupil_v = pupil_diameter_series(exp.eye_tracking)
pupil_aligned = interp_to_ophys(pupil_t, pupil_v, trial_t)
# Global binning:
pupil_edges = compute_bin_edges(np.concatenate(all_pupil))
pupil_bin = digitize_with_edges(tr['pupil_raw'], pupil_edges)
```

iii. Same processing pipeline as running speed.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same discretization approach as running speed: 5 percentile bins, NaN filled with median bin, values clipped.

ii. See 5-c (same `compute_bin_edges` and `digitize_with_edges` functions).

iii. Same as running speed.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same approach as running speed -- interpolated to ophys timebase within each trial window.

ii.
```python
pupil_aligned = interp_to_ophys(pupil_t, pupil_v, trial_t)
```

iii. Same frame-level alignment as running speed and neural data.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table. Each stimulus presentation inherits the outcome of its parent trial.

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

# Per trial:
outcome = int(trial_outcome_map[parent_id])
```

iii. These four outcome categories are the SDK's canonical trial outcome labels. The mapping order is hit=0, miss=1, false_alarm=2, correct_reject=3.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3) and broadcast as a constant across all frames within each stimulus presentation.

ii.
```python
np.full(T, tr['trial_outcome'], dtype=np.int64),
```

iii. Since all stimulus presentations within the same behavioral trial share the same outcome, each presentation gets the parent trial's outcome code.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- Failed experiments: if `process_experiment` throws an exception, output indicates error but processing continues for other files.
- Short presentations: if a stimulus presentation has fewer than 2 ophys frames, it is skipped.
- Missing behavioral data: NaN values from interpolation are filled with the median of finite digitized values (or 0 if all NaN).
- Few trials: sessions with fewer than 2 valid presentations are excluded.
- Missing columns: code checks for existence of `is_change`, `active`, `omitted` columns before using them.
- Invalid NaN source data: `interp_to_ophys` filters out non-finite source values before interpolation.

ii.
```python
if right - left < 2:
    continue

# NaN handling in digitize_with_edges:
bad = ~np.isfinite(values)
if np.any(bad):
    finite = np.where(np.isfinite(values))[0]
    fill = int(np.median(out[finite])) if finite.size else 0
    out[bad] = fill

# Column existence checks:
if 'active' in stim.columns:
    stim = stim[stim['active'].fillna(False)].copy()
```

iii. The code is defensive about missing columns and data, using fillna and existence checks.

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each NWB file via `BehaviorOphysExperiment.from_nwb_path()`, which reads large neural and behavioral data from disk. The AI uses multiprocessing for full runs to parallelize this I/O-bound step.

ii.
```python
use_parallel = (not sample) and len(files) > 4
if use_parallel:
    n_workers = min(max((os.cpu_count() or 2) // 2, 2), 8)
    with Pool(processes=n_workers) as pool:
        for i, sess in enumerate(pool.imap_unordered(process_experiment, files), start=1):
```

iii. Each NWB file contains full-session event traces, running speed, eye tracking, and stimulus presentations.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-stimulus-presentation loop in `process_experiment` iterates over each valid presentation sequentially, performing `np.searchsorted` and `interp_to_ophys` per presentation. The `interp_to_ophys` call per presentation is redundant since running speed and pupil could be interpolated once for the whole session and then sliced. The `build_image_labels_for_trial` function (defined but not used in the final code path) iterates over stimulus presentations with a Python loop.

ii.
```python
for stim_id, parent_id, start, stop, img_name, is_change in zip(
    stim_ids, stim_trial_ids, stim_start, stim_end, stim_image_name, stim_is_change
):
    # ...
    run_aligned = interp_to_ophys(run_t, run_v, trial_t)
    pupil_aligned = interp_to_ophys(pupil_t, pupil_v, trial_t)
```

iii. Interpolating running speed and pupil diameter once for the whole session and then slicing per presentation would be more efficient.

## 9-c. What processing does the code repeat multiple times?

i. Running speed and pupil diameter interpolation is repeated for every stimulus presentation within a session, when it could be done once per session and sliced. The `interp_to_ophys` function sorts the source data each time, which is redundant since the source data is the same for all presentations in a session.

ii.
```python
# Called once per stimulus presentation:
run_aligned = interp_to_ophys(run_t, run_v, trial_t)
pupil_aligned = interp_to_ophys(pupil_t, pupil_v, trial_t)
```

iii. This repeated interpolation is a performance inefficiency but doesn't affect correctness.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores `running_raw_all` and `pupil_raw_all` lists in each session dict alongside the per-trial data. These are only used to compute global bin edges and are redundant with the per-trial `running_raw` and `pupil_raw` arrays. The `build_image_labels_for_trial` function is defined but not used in the main processing path. The `make_processing_plot` function is only used in `--show-processing` mode.

ii.
```python
session['running_raw_all'].append(run_aligned)
session['pupil_raw_all'].append(pupil_aligned)
# These are separate from per-trial data:
session['trials'].append({
    'running_raw': run_aligned,
    # ...
})
```

iii. The `running_raw_all` and `pupil_raw_all` lists duplicate data already stored in the per-trial dicts.
