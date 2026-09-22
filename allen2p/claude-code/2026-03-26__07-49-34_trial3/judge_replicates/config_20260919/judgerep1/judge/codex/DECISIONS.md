# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the project experiment CSV, intersects it with NWB files present on disk, restricts to four active session types, and opens each selected NWB directly with `h5py`. It makes three passes: image-name discovery, behavioral percentile statistics, and full experiment loading.

ii.
```python
exp_table = pd.read_csv(META_DIR / 'ophys_experiment_table.csv')
nwb_files = list(NWB_DIR.glob('*.nwb'))
mask = (exp_table['ophys_experiment_id'].isin(nwb_ids) &
        exp_table['session_type'].isin(ACTIVE_SESSION_TYPES))
active_exps = exp_table[mask].copy()
...
with h5py.File(nwb_path, 'r') as f:
    ...
```

iii. The notes justify direct NWB loading as matching the available on-disk subset and exclude passive OPHYS_2/5 sessions because they lack meaningful active-task outcomes. The three-pass design establishes global categorical mappings and percentile edges.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values in the filtered experiment table; sorted string IDs are mapped to integer indices.

ii.
```python
subjects = sorted(exp_table['mouse_id'].unique().astype(str))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
all_subject_idx.append(subject_to_idx[str(row['mouse_id'])])
```

iii. The agent states that `mouse_id` is the subject identifier and reports 38 mice in the on-disk subset.

## 1-c. How are the data split into sessions?

i. Each selected `ophys_experiment_id`/NWB file is emitted as one output session. Simultaneously recorded planes sharing an `ophys_session_id` are not combined.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    eid = row['ophys_experiment_id']
    ...
    all_neural.append(session_neural)
```

iii. The notes explicitly decide that “Each NWB experiment = one ‘session’ in output format (one imaging plane with its own neurons).”

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. For each accepted trial, timestamps on the regular 30-Hz grid satisfying `start_time <= t < stop_time` define a variable-length trial.

ii.
```python
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
trial_time_indices = np.where(trial_mask)[0]
if len(trial_time_indices) < 3:
    continue
```

iii. The agent chose the full trial-table start-to-stop window so it contains pre-change and post-change activity and supports time-varying outputs.

## 1-e. How are trials filtered based on quality controls?

i. It retains Go or Catch trials, excludes aborted and auto-rewarded trials, skips windows shorter than three bins and trials with no recognized outcome, and drops experiments with fewer than two processed trials.

ii.
```python
valid_mask = ((trials['go'] | trials['catch']) &
              ~trials['aborted'] & ~trials['auto_rewarded'])
...
else:
    continue  # Unknown outcome, skip
...
if len(neural_trials) < 2:
    return None
```

iii. This is justified directly by the requested Go/Catch inclusion and aborted/auto-rewarded exclusion, plus the decoder’s minimum-two-trials requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the NWB event-detection matrix, restricted using the cell table’s `valid_roi` flag.

ii.
```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
events_data = f['processing']['ophys']['event_detection']['data'][()]
events_valid = events_data[:, valid_roi]
```

iii. The agent cites the paper’s statement that analyses used discrete calcium events and identifies these as FastLZeroSpikeInference outputs.

## 2-b. How is the `neural` data processed?

i. Valid event traces are linearly interpolated from native ophys timestamps to a regular 30-Hz grid, clipped nonnegative, sliced by trial, transposed to neuron-by-time, and cast to float32.

ii.
```python
regular_ts = np.arange(t_start, t_end, 1.0 / target_rate)
events_resampled = interpolate_to_regular_grid(ophys_ts, events, regular_ts)
events_resampled = np.maximum(events_resampled, 0)
neural_trial = events_resampled[trial_time_indices, :].T.astype(np.float32)
```

iii. The notes say the paper linearly interpolated neural data onto consistent 30-Hz timestamps, especially to reconcile approximately 31-Hz and 11-Hz systems.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only cells marked `valid_roi=True` are retained; experiments with no valid ROIs are skipped. Interpolated negative event values are also clipped to zero.

ii.
```python
valid_roi = cell_table['valid_roi'][()].astype(bool)
if valid_roi.sum() == 0:
    return None
events_valid = events_data[:, valid_roi]
```

iii. The agent describes `valid_roi` as the Allen SVM-based ROI quality-control result and considers events intrinsically nonnegative.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples are segmented relative to trial start, from `start_time` through (but excluding) `stop_time`; metadata calls the alignment event trial start.

ii.
```python
trial_mask = (regular_ts >= t_trial_start) & (regular_ts < t_trial_stop)
...
'temporal_alignment_event': 'Trial start time (first stimulus onset of trial)',
'off_start': 0.0,
```

iii. The full window was selected to preserve both sides of the image change and variable trial duration.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output is a regular 30-Hz grid (33.33 ms). Neural events and behaviors are linearly interpolated onto it; this is resampling, not aggregation into count bins.

ii.
```python
TARGET_RATE_HZ = 30.0
TIME_BIN_MS = 1000.0 / TARGET_RATE_HZ
regular_ts = np.arange(t_start, t_end, dt)
```

iii. The stated reason is to reproduce the paper’s 30-Hz interpolation and standardize differing microscope frame rates.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from `image_name`, `start_time`, and `omitted` in the selected NWB stimulus-presentation interval table.

ii.
```python
stim_data = {'start_time': stim['start_time'][()],
             'image_name': stim['image_name'][()],
             'omitted': stim['omitted'][()]}
```

iii. The agent uses actual stimulus presentations to create a frame-level image label and excludes the literal omitted category.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For every target time, it finds the most recent non-omitted stimulus onset with `searchsorted`, carries that image through gray and omitted flashes, maps local sorted image names to a global sorted vocabulary, and uses 0 before any presentation.

ii.
```python
non_omitted = ~stim_data['omitted']
insert_idx = np.searchsorted(stim_starts, timepoints, side='right') - 1
image_idx[i] = name_to_idx.get(stim_names[insert_idx[i]], 0)
...
img_id_global = np.array([local_to_global.get(v, 0) for v in
                          trial_data_out['image_identity']])
```

iii. The notes justify holding the last image identity during gray periods and omissions and using one global deterministic codebook.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is evaluated at the same per-trial `regular_ts` samples used to index the resampled neural matrix, so lengths and timestamps coincide.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
image_idx = get_image_at_timepoints(trial_ts, raw_data['stim'], all_stim_images)
```

iii. The common regular grid is the agent’s alignment mechanism.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It uses stimulus-presentation `is_change`, `omitted`, and `start_time` (the loaded `stop_time` is not used in the final indicator).

ii.
```python
change_mask = stim_data['is_change'] & ~stim_data['omitted']
change_starts = stim_data['start_time'][change_mask]
```

iii. The rationale is to label actual, non-omitted change presentations rather than infer them solely from trial type.

## 4-b. What processing is involved in computing `output` *Image change*?

i. It initializes zeros and sets samples in each 750-ms interval beginning at a change presentation to one.

ii.
```python
for cs, ce in zip(change_starts, change_stops):
    mask = (timepoints >= cs) & (timepoints < cs + 0.75)
    change_signal[mask] = 1
```

iii. The 750 ms represents a 250-ms image flash plus its 500-ms gray interval.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is directly binary: 1 inside a qualifying 750-ms change window and 0 otherwise; no continuous threshold is estimated.

ii.
```python
change_signal = np.zeros(n_tp, dtype=np.int64)
change_signal[mask] = 1
```

iii. The requested output is binary, and the stimulus metadata already supplies the change decision.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The binary indicator is evaluated on `trial_ts`, the same 30-Hz samples as each neural trial.

ii.
```python
trial_ts = regular_ts[trial_time_indices]
change_signal = get_image_change_at_timepoints(trial_ts, raw_data['stim'])
```

iii. Shared target timestamps guarantee sample-wise alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses the NWB running speed `data` and corresponding `timestamps`.

ii.
```python
running_speed = f['processing']['running']['speed']['data'][()]
running_ts = f['processing']['running']['speed']['timestamps'][()]
```

iii. This is documented as the SDK/NWB standard wheel-speed stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. It linearly interpolates speed to 30 Hz, computes five global percentile bins from all raw full-session running samples in a preliminary pass, then digitizes trial samples. Duplicate edges are nudged upward and NaNs map to bin 0.

ii.
```python
running_resampled = np.interp(regular_ts, raw_data['running_ts'],
                              raw_data['running_speed'])
running_bin_edges = compute_percentile_bins(all_running_cat, n_bins=5)
running_binned = digitize_to_bins(trial_data_out['running_speed'],
                                  running_bin_edges)
```

iii. Global percentiles were chosen for approximately balanced decoder classes and consistent category meanings across experiments.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The 0th, 20th, 40th, 60th, 80th, and 100th percentiles form five bins; `np.digitize` returns labels 0–4.

ii.
```python
percentiles = np.linspace(0, 100, n_bins + 1)
edges = np.percentile(valid, percentiles)
binned = np.digitize(values, bin_edges[1:-1])
```

iii. This directly implements the requested five equal-percentile categories.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Full-session speed is interpolated onto the identical regular 30-Hz timestamps, then sliced using the same trial indices as neural data.

ii.
```python
running_resampled = np.interp(regular_ts, ...)
running_trial = running_resampled[trial_time_indices]
```

iii. The common timestamp grid is justified by hardware synchronization of data streams.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Despite naming the output pupil diameter, the code uses EyeTracking `pupil_tracking/area`, its timestamps, and the `likely_blink` flag.

ii.
```python
pupil_area = pupil_tracking['area'][()]
pupil_ts = pupil_tracking['timestamps'][()]
likely_blink = f['acquisition']['EyeTracking']['likely_blink']['data'][()]
```

iii. The notes explicitly call pupil area a proxy for diameter and use the blink flag for artifact handling.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink samples are set to NaN, linearly filled over sample index (all-NaN becomes zeros), then resampled to 30 Hz. Five percentile edges are computed globally from raw, non-blink full-session areas and applied to trial samples. Missing pupil streams produce all-NaN trials and hence category 0.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_area = interpolate_nans(pupil_area)
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
...
pupil_bin_edges = compute_percentile_bins(all_pupil_cat, n_bins=5)
```

iii. The agent says interpolation prevents blink artifacts and global percentiles balance categories consistently.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It uses the same five global percentile categories (labels 0–4) as running speed, with NaN assigned to 0.

ii.
```python
pupil_binned = digitize_to_bins(trial_data_out['pupil_area'],
                                pupil_bin_edges)
binned[np.isnan(values)] = 0
```

iii. This follows the instruction to discretize pupil size into five equal percentile bins.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Cleaned pupil area is interpolated to the same regular 30-Hz timestamps and sliced by the neural trial indices.

ii.
```python
pupil_resampled = np.interp(regular_ts, pupil_ts, pupil_area)
pupil_trial = pupil_resampled[trial_time_indices]
```

iii. Alignment relies on synchronized clocks and the shared target grid.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It comes from the trials table’s mutually exclusive `hit`, `miss`, `false_alarm`, and `correct_reject` booleans.

ii.
```python
if trials['hit'][trial_idx]: outcome = 0
elif trials['miss'][trial_idx]: outcome = 1
elif trials['false_alarm'][trial_idx]: outcome = 2
elif trials['correct_reject'][trial_idx]: outcome = 3
```

iii. These are described as the canonical outcomes for accepted Go and Catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The four outcomes are assigned fixed codes 0–3. Trials without one are skipped, and the selected code is broadcast across all time bins in the combined output matrix.

ii.
```python
outcome_broadcast = np.full((1, n_tp),
                            trial_data_out['trial_outcome'], dtype=np.int64)
output_combined = np.concatenate([output_tv, outcome_broadcast], axis=0)
```

iii. Broadcasting accommodates the target format’s single rectangular output array while retaining a static per-trial label.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. File/field loading is wrapped in exception handlers and bad experiments are skipped. No-valid-ROI or too-few-trial experiments are skipped. Missing pupil data becomes NaN and category 0; blink/NaN pupil samples are interpolated (all missing becomes zero). Duplicate percentile edges are made strictly increasing. Unknown trial outcomes and extremely short trials are skipped. `np.interp` uses endpoint values outside source ranges.

ii.
```python
except Exception as e:
    print(f"  ERROR loading {experiment_id}: {e}")
    return None
...
if nans.all(): return np.zeros_like(arr)
...
if edges[i] <= edges[i-1]:
    edges[i] = edges[i-1] + 1e-10
```

iii. The rationale is to keep one malformed file from aborting conversion and to guarantee discrete, finite decoder labels, though some fallbacks conflate missingness with the lowest category.

## 9-a. What are the most time-consuming steps of the code?

i. Repeated NWB I/O across three passes and per-cell interpolation of the full event matrix are the main costs. Saving the resulting 8.3-GB pickle is also material.

ii.
```python
for _, row in exp_table.iterrows():
    with h5py.File(nwb_path, 'r') as f: ...  # Passes 1 and 2
...
raw_data = load_experiment_data(nwb_path, eid)  # Pass 3
for i in range(data.shape[1]):
    result[:, i] = np.interp(...)
```

iii. The notes identify sequential experiment processing and multiple passes over NWB files as inefficiencies; the full run took about six minutes.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The loop assigning image labels can be replaced by direct indexed assignment from the already vectorized `insert_idx`; local-to-global image remapping can use a lookup array; percentile gathering and some per-trial work could be batched. Neural interpolation remains a feature loop because `np.interp` is one-dimensional, though a vectorized interpolation routine could replace it.

ii.
```python
for i in range(n_tp):
    if insert_idx[i] >= 0:
        image_idx[i] = name_to_idx.get(stim_names[insert_idx[i]], 0)
...
img_id_global = np.array([local_to_global.get(v, 0) for v in
                          trial_data_out['image_identity']])
```

iii. The agent claims `searchsorted` and interpolation are efficient, but its notes still acknowledge sequential processing; it does not specifically justify the remaining Python label loops.

## 9-c. What processing does the code repeat multiple times?

i. Every NWB is opened once to collect image names, again to collect full-session running/pupil statistics, and again for conversion. Stimulus-table discovery/decoding is repeated in passes 1 and 3. Local image mappings are rebuilt and `global_image_names.index(name)` is repeatedly searched per experiment.

ii.
```python
# Pass 1 ... with h5py.File(nwb_path, 'r')
# Pass 2 ... with h5py.File(nwb_path, 'r')
# Pass 3 ... load_experiment_data(nwb_path, eid)
```

iii. The notes explicitly list multiple NWB passes as an identified inefficiency, accepted to establish global mappings and bins before assembly.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads unused fields including `cell_specimen_ids`, trial `change_time`/`is_change`/some trial-type fields, and stimulus `stop_time`; creates an unused scalar `trial_outcome` array; imports unused concurrency tools; and computes/retains `raw_data` for plots even when plotting is disabled. Most importantly, percentile passes process full-session behavior outside accepted trials even though those samples are only used to set thresholds.

ii.
```python
cell_specimen_ids = cell_table['cell_specimen_id'][()]
change_stops = stim_data['stop_time'][change_mask]  # ce is never used
trial_outcome = np.array([trial_data_out['trial_outcome']], dtype=np.int64)
from concurrent.futures import ProcessPoolExecutor, as_completed
```

iii. These appear to be implementation leftovers or diagnostics. The agent documents multiple passes as inefficient but does not explicitly discuss these discarded values.
