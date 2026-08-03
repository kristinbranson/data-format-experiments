# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by discovering all NWB files under `data/visual-behavior-ophys-1.1.0` using recursive glob, then loading each NWB file individually via the AllenSDK's `BehaviorOphysExperiment.from_nwb_path()`. Each NWB file corresponds to one ophys experiment. When running in `--full` mode with >4 files, multiprocessing is used with up to 8 workers via `Pool.imap_unordered`. Each experiment is processed independently in `process_experiment()`, extracting neural events, trials, stimulus presentations, running speed, and eye tracking data.

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

# In process_experiment:
exp = BehaviorOphysExperiment.from_nwb_path(str(nwb_path))
```

iii. The agent identified the AllenSDK as the appropriate loader during Step 1 (Reference Code Exploration) and found that `BehaviorOphysExperiment` is the primary class that assembles all data streams (neural, behavioral, stimulus) from NWB files. The decision to use `from_nwb_path` was driven by the data being stored as local NWB files in the data directory.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `mouse_id` field from each experiment's metadata. After all experiments are processed, unique mouse IDs are collected and sorted. A `subject_idx` array maps each session to its corresponding subject index.

ii.
```python
# In process_experiment:
session = {
    'subject': str(meta['mouse_id']),
    ...
}

# In assemble_dataset:
subjects = sorted({s['subject'] for s in processed_sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
data['subject_idx'] = np.array([subject_to_idx[s['subject']] for s in processed_sessions], dtype=np.int64)
```

iii. The agent noted in CONVERSION_NOTES.md Step 4 that each ophys experiment has one subject, and the metadata `mouse_id` field provides the mapping. The final dataset contains 38 unique subjects.

## 1-c. How are the data split into sessions?

i. Each ophys experiment NWB file is treated as one session. The session is identified by `ophys_experiment_id` from the experiment metadata. Sessions with fewer than 2 valid trials are excluded.

ii.
```python
# In process_experiment:
session = {
    'session_id': int(meta['ophys_experiment_id']),
    ...
}

# In main:
if len(sess['trials']) >= 2:
    processed.append(sess)

# In assemble_dataset:
if len(sess_neural) >= 2:
    data['neural'].append(sess_neural)
```

iii. The agent documented in CONVERSION_NOTES.md Step 4 that "each ophys experiment NWB file is treated as one decoder session because neural populations are experiment-specific." The instructions require at least 2 trials per session for decoder evaluation. The final dataset contains 284 sessions.

## 1-d. How are the data split into trials?

i. **This is a critical decision.** The AI initially segmented trials using the SDK's `trials` table (start/stop times), which defines trials as spanning from one change/catch event to the next. However, this caused most timepoints to fall during gray screen periods with no valid image identity label. The AI then redefined "trials" as individual **stimulus presentations** — each ~750ms image flash becomes a separate trial with ~7-8 ophys timepoints. Each presentation must belong to a valid Go/Catch parent trial, be active, non-omitted, and have a valid image name.

ii.
```python
# Filtering stimulus presentations:
stim = exp.stimulus_presentations.copy().sort_values('start_time')
stim = stim[stim['trials_id'].notna()].copy()
stim = stim[stim['trials_id'].isin(valid_trial_ids)].copy()
if 'active' in stim.columns:
    stim = stim[stim['active'].fillna(False)].copy()
if 'omitted' in stim.columns:
    stim = stim[~stim['omitted'].fillna(False)].copy()
stim = stim[stim['image_name'].notna()].copy()

# Each stimulus presentation becomes a trial:
for stim_id, parent_id, start, stop, img_name, is_change in zip(...):
    left = np.searchsorted(ophys_t, start, side='left')
    right = np.searchsorted(ophys_t, stop, side='left')
    if right - left < 2:
        continue
    trial_t = ophys_t[left:right]
    neural = neural_full[:, left:right]
```

iii. The agent's trajectory (Steps 20-22) shows this was a deliberate pivot. Initially the SDK trials were used, but 67% of timepoints had no valid image label (gray screens), causing -1 values that crashed the decoder training. The agent reasoned that per-image-presentation trials align with the paper's "image-by-image" analysis and ensure all output labels are valid. The final dataset has 963,758 "trials" (individual image presentations).

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in two stages: (1) The SDK trials table is filtered to include only Go and Catch trials, excluding Aborted and Auto-rewarded trials. (2) Stimulus presentations are filtered to include only those belonging to valid trials, that are active, non-omitted, and have valid image names and timing. Presentations with fewer than 2 ophys timepoints are also excluded.

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

iii. The agent followed the task instructions directly: "Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials." Additional filtering on stimulus presentations was added to ensure valid labels. Trials without a determinable outcome (hit/miss/false_alarm/correct_reject) are also excluded.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the detected calcium events accessed via `exp.events`, specifically the `'events'` column of the events DataFrame. This is the deconvolved/detected event trace for each ROI.

ii.
```python
def extract_events_matrix(events_df):
    event_col = 'events'
    arrs = [np.asarray(x, dtype=np.float32) for x in events_df[event_col].values]
    return np.stack(arrs, axis=0)

# In process_experiment:
neural_full = extract_events_matrix(exp.events)
```

iii. The agent documented in CONVERSION_NOTES.md Step 3: "For all analysis of neural data we used the detected calcium events..." (from methods.txt). The decision to use events rather than dF/F traces is consistent with the reference paper.

## 2-b. How is the `neural` data processed?

i. The neural data undergoes minimal processing. The full events matrix (all neurons x all timepoints) is extracted once per experiment. For each trial (stimulus presentation), the relevant columns are sliced using `np.searchsorted` on ophys timestamps. The data is stored as float32.

ii.
```python
neural_full = extract_events_matrix(exp.events)
# Per trial:
left = np.searchsorted(ophys_t, start, side='left')
right = np.searchsorted(ophys_t, stop, side='left')
neural = neural_full[:, left:right].astype(np.float32, copy=False)
```

iii. The agent chose to use the raw detected events without further processing (no smoothing, no normalization, no rebinning). This is consistent with using the SDK-provided events traces directly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit neuron-level quality filtering is applied. All neurons (ROIs) present in the NWB file's events table are included. The agent relies on the AllenSDK's built-in cell selection (valid ROIs already filtered by the SDK when creating the experiment object).

ii.
```python
# No neuron filtering code exists in convert_data.py
# All neurons from exp.events are used:
neural_full = extract_events_matrix(exp.events)
```

iii. The agent's CONVERSION_NOTES.md states under neuron curation: "Use SDK-provided valid cell/ROI tables and whichever neural signal the reference analysis used." The agent implicitly trusts the SDK to provide only valid cells. No additional filtering based on SNR, reliability, or other metrics is performed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to ophys timestamps. For each stimulus presentation, the ophys timestamps falling within the presentation's `[start_time, end_time)` window are identified using `np.searchsorted`. The neural data columns at those indices are extracted.

ii.
```python
ophys_t = np.asarray(exp.ophys_timestamps, dtype=np.float64)
# Per trial:
left = np.searchsorted(ophys_t, start, side='left')
right = np.searchsorted(ophys_t, stop, side='left')
trial_t = ophys_t[left:right]
neural = neural_full[:, left:right]
```

iii. The instructions state "Temporally align based on ophys timestamp." The agent uses the native ophys timestamps as the temporal basis and slices all data streams to match. The metadata records `temporal_alignment_event` as "native ophys timestamps within each trial defined by SDK trial start/stop times."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native ophys frame rate (~10.7 Hz, approximately 93ms per frame). No temporal rebinning is applied. The metadata field `time_bin_size` is set to `None`.

ii.
```python
'time_bin_size': None,
```

iii. The agent chose to use native ophys timestamps without rebinning. Each trial has ~7-8 timepoints (ophys frames within the ~750ms stimulus presentation window). The `time_bin_size` is left as `None` rather than being set to the actual frame interval. The instructions state "Time bins should be the same size for all trials and sessions" — the ophys frame rate is approximately constant but not exact, so no explicit rebinning is done.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `image_name` column of the `stimulus_presentations` table. Each image name (e.g., 'im000', 'im031') is mapped to a categorical integer index.

ii.
```python
stim_image_name = stim['image_name'].astype(str).to_numpy()

# In assemble_dataset:
image_names = sorted({tr['image_name'] for s in processed_sessions for tr in s['trials']})
image_to_idx = {name: i for i, name in enumerate(image_names)}
# Per trial:
np.full(T, image_to_idx[tr['image_name']], dtype=np.int64),
```

iii. Since each "trial" is a single stimulus presentation, the image identity is constant throughout the trial. The 16 unique images are assigned integer indices 0-15 sorted alphabetically.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The image name string is converted to a categorical integer index. Since trials are defined as individual image presentations, each trial has a single constant image identity across all timepoints. The mapping is built globally across all sessions.

ii.
```python
# Per trial output:
np.full(T, image_to_idx[tr['image_name']], dtype=np.int64),
```

iii. Omitted stimuli and presentations without valid image names are filtered out before trial construction, so image identity is always valid (no -1 or NaN values).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Since each trial corresponds to exactly one image presentation period, the image identity is constant across all timepoints in the trial. It is represented as a time-varying vector filled with the same value, matching the neural data's temporal dimension.

ii.
```python
np.full(T, image_to_idx[tr['image_name']], dtype=np.int64),
```

iii. By redefining trials as individual presentations, alignment is trivially exact — the entire trial window is one image.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` column of the `stimulus_presentations` table.

ii.
```python
if 'is_change' in stim.columns:
    stim_is_change = stim['is_change'].fillna(False).astype(bool).to_numpy()
else:
    stim_is_change = np.zeros(len(stim), dtype=bool)
```

iii. The `is_change` flag is set by the AllenSDK to indicate whether a stimulus presentation represents a change from the previous image.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The `is_change` boolean from the stimulus presentation row is converted to a binary integer (0 or 1) and broadcast to all timepoints in the trial. NaN values are filled with False.

ii.
```python
'image_change': np.full(right - left, int(is_change), dtype=np.int64),
```

iii. Since each trial is one presentation, image_change is constant across the trial's timepoints.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is inherently binary. No thresholding is needed — it directly takes values 0 (no_change) or 1 (change).

ii.
```python
IMAGE_CHANGE_VALUES = ['no_change', 'change']
```

iii. The distribution in the full dataset is 92.3% no_change, 7.7% change.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity — since each trial is one presentation, the image_change value is constant across all timepoints in the trial, matching neural data dimensions.

ii.
```python
'image_change': np.full(right - left, int(is_change), dtype=np.int64),
```

iii. Alignment is trivial due to the per-presentation trial definition.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `exp.running_speed`, specifically the `'speed'` column with corresponding `'timestamps'`.

ii.
```python
run_t = exp.running_speed['timestamps'].to_numpy(dtype=np.float64)
run_v = exp.running_speed['speed'].to_numpy(dtype=np.float32)
```

iii. The agent used the SDK's filtered running speed signal.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed values are interpolated to ophys timestamps using `np.interp`. The interpolated values are then discretized into 5 equal percentile bins. Bin edges are computed globally across all sessions using `np.quantile` with quantiles [0, 0.2, 0.4, 0.6, 0.8, 1.0]. Non-finite values are filled with the median bin.

ii.
```python
def interp_to_ophys(src_t, src_v, dst_t):
    good = np.isfinite(src_t) & np.isfinite(src_v)
    return np.interp(dst_t, src_t, src_v, left=np.nan, right=np.nan).astype(np.float32)

run_aligned = interp_to_ophys(run_t, run_v, trial_t)

# Global bin computation:
run_edges = compute_bin_edges(np.concatenate(all_run))
run_bin = digitize_with_edges(tr['running_raw'], run_edges)
```

iii. The instructions specify "discretized into five equal percentile bins." The agent computes percentile edges globally and applies them per-trial.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins (indices 0-4) using equal percentile (quintile) edges computed across all valid timepoints in the dataset. `np.digitize` assigns each value to a bin. Values are clipped to [0, 4].

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
    out = np.clip(out, 0, len(edges) - 2)
    return out
```

iii. The bin edge computation ensures monotonically increasing edges by adding epsilon when quantiles collide. The final distribution is approximately uniform (20% per bin).

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed (which has its own timestamps) is linearly interpolated to the ophys timestamps within each trial using `np.interp`. Values outside the running speed's time range are set to NaN and then handled during binning (filled with median bin).

ii.
```python
run_aligned = interp_to_ophys(run_t, run_v, trial_t)
```

iii. The agent interpolates to ophys timestamps as required by the instructions.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `exp.eye_tracking`, preferring `pupil_area` (converted to diameter via `2*sqrt(area/pi)`) or falling back to geometric mean of `pupil_width` and `pupil_height`.

ii.
```python
def pupil_diameter_series(eye_tracking_df):
    if 'pupil_area' in et.columns:
        area = et['pupil_area'].to_numpy(dtype=np.float64)
        diam = 2.0 * np.sqrt(area / math.pi)
    elif 'pupil_width' in et.columns and 'pupil_height' in et.columns:
        diam = np.sqrt(et['pupil_width'].to_numpy(dtype=np.float64) * et['pupil_height'].to_numpy(dtype=np.float64))
```

iii. The agent's CONVERSION_NOTES.md Step 5 states: "derive diameter from pupil area if no direct diameter column exists." The conversion from area to diameter assumes a circular pupil.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter is computed from pupil area, with blink frames masked as NaN (using the `likely_blink` column). The diameter is then interpolated to ophys timestamps using `np.interp`, and discretized into 5 equal percentile bins using globally computed edges.

ii.
```python
blink = et['likely_blink'].fillna(False).to_numpy(dtype=bool)
diam[blink] = np.nan
pupil_aligned = interp_to_ophys(pupil_t, pupil_v, trial_t)
pupil_bin = digitize_with_edges(tr['pupil_raw'], pupil_edges)
```

iii. The agent masks blinks before computing bin edges and interpolating, preventing blink artifacts from affecting the binning distribution.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed — 5 equal percentile bins computed globally across all valid (non-blink, finite) pupil diameter values.

ii.
```python
pupil_edges = compute_bin_edges(np.concatenate(all_pupil))
pupil_bin = digitize_with_edges(tr['pupil_raw'], pupil_edges)
```

iii. The distribution is approximately uniform (~20% per bin), with slight deviation in bin_0 (21%) due to blink-masked NaN values being filled with the median bin.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter (on eye tracking timestamps) is linearly interpolated to ophys timestamps within each trial, same approach as running speed.

ii.
```python
pupil_aligned = interp_to_ophys(pupil_t, pupil_v, trial_t)
```

iii. Interpolation to ophys timestamps ensures alignment with the neural data temporal basis.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the SDK's trials table.

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

TRIAL_OUTCOME_VALUES = ['hit', 'miss', 'false_alarm', 'correct_reject']
```

iii. The four outcomes correspond to: hit (correct lick on Go trial), miss (no lick on Go trial), false_alarm (lick on Catch trial), correct_reject (no lick on Catch trial).

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcome is determined by checking the boolean outcome columns in priority order (hit > miss > false_alarm > correct_reject). Rows where none of these are True are excluded. The outcome is stored as a static per-trial value but represented as a time-varying constant in the output matrix (same value at every timepoint).

ii.
```python
# Per trial output:
np.full(T, tr['trial_outcome'], dtype=np.int64),
```

iii. The outcome from the parent SDK trial is propagated to all stimulus presentations belonging to that trial. The distribution in the full dataset is: hit (18.7%), miss (68.7%), false_alarm (1.1%), correct_reject (11.5%).

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several data quality issues are handled:
- **Missing pupil data / blinks**: Frames with `likely_blink=True` are set to NaN before interpolation. NaN values after binning are filled with the median bin value.
- **Missing running speed**: `np.interp` with `left=np.nan, right=np.nan` handles out-of-range timestamps. NaN values are filled with median bin during digitization.
- **Missing trial outcomes**: Trials where no outcome flag is True return `None` and are excluded.
- **Short trials**: Stimulus presentations with fewer than 2 ophys timepoints are skipped.
- **Missing image names / omitted stimuli**: Filtered out before trial construction.
- **All-zero neural data**: Many trials have all-zero neural activity (expected for sparse calcium events). These are kept in the dataset.

ii.
```python
# NaN handling in digitize_with_edges:
bad = ~np.isfinite(values)
if np.any(bad):
    finite = np.where(np.isfinite(values))[0]
    fill = int(np.median(out[finite])) if finite.size else 0
    out[bad] = fill

# In interp_to_ophys:
if good.sum() < 2:
    return np.full(dst_t.shape, np.nan, dtype=np.float32)
```

iii. The agent's approach is generally conservative — missing data is either excluded or filled with a neutral value (median bin).

## 9-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading each NWB file via `BehaviorOphysExperiment.from_nwb_path()`. The agent's sample timing showed ~22 seconds per session. For the full 284-session dataset, the agent used multiprocessing (8 workers) to parallelize this.

ii.
```python
use_parallel = (not sample) and len(files) > 4
if use_parallel:
    n_workers = min(max((os.cpu_count() or 2) // 2, 2), 8)
    with Pool(processes=n_workers) as pool:
        for i, sess in enumerate(pool.imap_unordered(process_experiment, files), start=1):
```

iii. The trajectory shows the agent initially estimated 104 minutes for full serial conversion at 22s/session, prompting the addition of multiprocessing.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop that could be vectorized is the per-stimulus-presentation iteration in `process_experiment()`. The code iterates over each stimulus presentation with a Python for-loop, performing `np.searchsorted`, array slicing, and `interp_to_ophys` for each presentation individually. These operations (especially the many small interpolations) could potentially be batched.

ii.
```python
for stim_id, parent_id, start, stop, img_name, is_change in zip(
    stim_ids, stim_trial_ids, stim_start, stim_end, stim_image_name, stim_is_change
):
    left = np.searchsorted(ophys_t, start, side='left')
    right = np.searchsorted(ophys_t, stop, side='left')
    ...
```

iii. The agent noted in CONVERSION_NOTES.md Step 6 that the code "iterates over stimulus presentations per trial" as an identified inefficiency but did not vectorize it, instead using multiprocessing as the primary speedup.

## 9-c. What processing does the code repeat multiple times?

i. The `interp_to_ophys` function is called separately for running speed and pupil diameter for every single stimulus presentation (~3,400 presentations per session), even though the interpolation base (running speed and pupil timestamps/values) is the same across all presentations in a session. A single interpolation to all ophys timestamps followed by slicing would be more efficient.

ii.
```python
# Called per presentation, not per session:
run_aligned = interp_to_ophys(run_t, run_v, trial_t)
pupil_aligned = interp_to_ophys(pupil_t, pupil_v, trial_t)
```

iii. The agent did not optimize this repeated computation. The running speed and pupil signals could be interpolated once to all ophys timestamps, then sliced per trial.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores `running_raw_all` and `pupil_raw_all` lists in the session dictionary solely for computing global bin edges, but these raw values are not included in the final dataset. The `build_image_labels_for_trial` function is defined but never called in the final version of the code (vestigial from the earlier trial-definition approach). The `make_processing_plot` function performs plotting that is only used with `--show-processing` flag.

ii.
```python
# Stored but only used for bin edge computation:
session['running_raw_all'].append(run_aligned)
session['pupil_raw_all'].append(pupil_aligned)

# Defined but never called:
def build_image_labels_for_trial(trial_timestamps, stim_df, image_to_idx):
    ...
```

iii. The `build_image_labels_for_trial` function is dead code from the earlier approach where trials were SDK-defined and image labels needed to be computed per-timepoint within longer trial windows. After switching to per-presentation trials, this function became unnecessary but was not removed.
