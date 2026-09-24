# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively discovers every local NWB file under `data/visual-behavior-ophys-1.1.0` (or the first two in sample mode), loads each with `BehaviorOphysExperiment.from_nwb_path`, and processes full mode in a multiprocessing pool when there are more than four files.

ii.
```python
files = sorted(DATASET_ROOT.rglob('*.nwb'))
exp = BehaviorOphysExperiment.from_nwb_path(str(nwb_path))
with Pool(processes=n_workers) as pool:
    for i, sess in enumerate(pool.imap_unordered(process_experiment, files), start=1):
```

iii. The notes say NWB/AllenSDK objects provide the experiment, behavior, ophys, and aligned timestamp tables. Local NWBs were chosen because those are the supplied data; multiprocessing was later added because SDK/NWB loading dominated runtime.

## 1-b. How are the data split into subjects?

i. A subject is the experiment metadata `mouse_id`. Unique string IDs are sorted globally and each retained experiment/session receives an index into that list.

ii.
```python
'subject': str(meta['mouse_id']),
subjects = sorted({s['subject'] for s in processed_sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The agent identified `mouse_id` as the dataset's animal identifier and reported 38 subjects among the supplied ophys experiments.

## 1-c. How are the data split into sessions?

i. Each ophys-experiment NWB is treated as one decoder session, identified by `ophys_experiment_id`; planes are not merged into an `ophys_session_id` session.

ii.
```python
session = {
    'session_id': int(meta['ophys_experiment_id']),
    'subject': str(meta['mouse_id']),
    'region': str(meta['targeted_structure']),
```

iii. The notes justify this because each experiment NWB contains one experiment-specific neural population and aligned behavior tables, which fits the decoder's session semantics.

## 1-d. How are the data split into trials?

i. After identifying valid behavioral trials, the agent uses each eligible non-omitted active stimulus presentation belonging to one of those trials as a decoder trial. It slices from that presentation's `start_time` to `end_time`, yielding about 7–8 native ophys frames rather than one long behavioral-trial window.

ii.
```python
stim = stim[stim['trials_id'].isin(valid_trial_ids)].copy()
for stim_id, parent_id, start, stop, img_name, is_change in zip(...):
    left = np.searchsorted(ophys_t, start, side='left')
    right = np.searchsorted(ophys_t, stop, side='left')
    if right - left < 2:
        continue
```

iii. The initial full-behavioral-trial design left image identity undefined during gray periods and failed decoder training. The agent switched to image-presentation intervals because the paper analyzes 750 ms image intervals and this gives valid image labels throughout each decoder trial.

## 1-e. How are trials filtered based on quality controls?

i. Parent trials must be Go or Catch, not aborted or auto-rewarded, and must have one of the four recognized outcomes. Presentations must have a parent trial, belong to a valid parent, be active when that column exists, be non-omitted, have an image name and finite start/end times; slices shorter than two frames and sessions with fewer than two resulting trials are discarded.

ii.
```python
keep = (trials['go'].fillna(False) | trials['catch'].fillna(False))
keep &= ~trials['aborted'].fillna(False)
keep &= ~trials['auto_rewarded'].fillna(False)
trials = trials.loc[trials['trial_outcome_idx'].notna()].copy()
stim = stim[~stim['omitted'].fillna(False)].copy()
if right - left < 2:
    continue
```

iii. The Go/Catch inclusion and aborted/auto-reward exclusion directly follow the task. Active, non-omitted presentations ensure the output really is a presented non-gray image; the minimum-trial rule satisfies decoder validation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the AllenSDK `exp.events` table's per-cell `events` arrays (detected/deconvolved calcium events), not dF/F.

ii.
```python
neural_full = extract_events_matrix(exp.events)
arrs = [np.asarray(x, dtype=np.float32) for x in events_df['events'].values]
return np.stack(arrs, axis=0)
```

iii. The methods explicitly state that detected calcium events were used for neural analyses, so the agent judged events more reference-consistent than the also-available dF/F traces.

## 2-b. How is the `neural` data processed?

i. Per-cell event vectors are converted to float32, checked to have equal lengths, stacked neuron-by-time, then directly sliced at presentation boundaries. No normalization, smoothing, or additional event processing is applied.

ii.
```python
lengths = {a.shape[0] for a in arrs}
if len(lengths) != 1:
    raise ValueError(...)
neural_full = np.stack(arrs, axis=0)
neural = neural_full[:, left:right].astype(np.float32, copy=False)
```

iii. The agent relies on the SDK's already processed detected events and preserves native samples, using float32 to reduce the very large output's memory footprint.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no extra neuron-level filtering. The cells exposed in `exp.events` are used, with only a trace-length consistency check; experiments or short presentation slices can be excluded upstream.

ii.
```python
if event_col not in events_df.columns:
    raise KeyError(...)
if len(lengths) != 1:
    raise ValueError(...)
```

iii. The notes state that SDK cell/event tables embody valid ROI selection and no further paper-supported neuron filter was identified.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Presentation start/end times are located on `exp.ophys_timestamps` with left-sided `searchsorted`; the identical native timestamp slice defines neural and behavioral output samples. Alignment is therefore to the onset of each non-gray image presentation.

ii.
```python
left = np.searchsorted(ophys_t, start, side='left')
right = np.searchsorted(ophys_t, stop, side='left')
trial_t = ophys_t[left:right]
neural = neural_full[:, left:right]
```

iii. The task explicitly requests ophys-timestamp alignment, while stimulus onset is the natural event for the paper's image-by-image analysis.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native ophys frames are retained (roughly 7–8 frames per 750 ms presentation); no temporal rebinning or resampling of neural events is done. The output metadata leaves `time_bin_size` as `None` rather than recording the median native frame interval.

ii.
```python
'time_bin_size': None,
trial_t = ophys_t[left:right]
neural = neural_full[:, left:right]
```

iii. The agent chose native timing to meet the ophys-timestamp requirement. It distinguished the paper's 750 ms analysis interval (the trial window) from the finer native neural samples.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from `exp.stimulus_presentations['image_name']` for each retained presentation.

ii.
```python
stim_image_name = stim['image_name'].astype(str).to_numpy()
'image_name': img_name,
```

iii. The notes identify stimulus presentations, rather than trial initial/change columns, as the authoritative time-specific source for the non-gray image.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Omitted and missing identities are removed. Remaining image-name strings are sorted globally, mapped to integer categories, and repeated across every ophys frame of that presentation.

ii.
```python
image_names = sorted({tr['image_name'] for s in processed_sessions for tr in s['trials']})
image_to_idx = {name: i for i, name in enumerate(image_names)}
np.full(T, image_to_idx[tr['image_name']], dtype=np.int64)
```

iii. A global deterministic mapping keeps category meanings consistent between experiments and eliminates the invalid `-1` labels found in the abandoned full-trial version.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Each decoder trial is exactly one presentation interval; its identity is constant for `T = neural.shape[1]` samples.

ii.
```python
T = tr['neural'].shape[1]
np.full(T, image_to_idx[tr['image_name']], dtype=np.int64)
```

iii. Presentation-defined slicing makes the categorical vector exactly the same length and interval as the neural matrix.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It is derived from the retained stimulus presentation's `is_change` column (or all false if the column is absent).

ii.
```python
stim_is_change = stim['is_change'].fillna(False).astype(bool).to_numpy()
'image_change': np.full(right - left, int(is_change), dtype=np.int64),
```

iii. The agent treats the presentation table's explicit change annotation as the most direct indicator of an image change.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Missing values become false, values are cast to Boolean/integer, and the result is repeated over the presentation's frames.

ii.
```python
stim['is_change'].fillna(False).astype(bool)
np.full(right - left, int(is_change), dtype=np.int64)
```

iii. No derived timing calculation is needed because the SDK already marks change presentations.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numeric threshold is applied; the Boolean `is_change` flag is encoded directly as 0 (`no_change`) or 1 (`change`).

ii.
```python
IMAGE_CHANGE_VALUES = ['no_change', 'change']
int(is_change)
```

iii. The source is already binary, so direct categorical encoding is sufficient.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The presentation's change label is repeated for precisely the number of neural frames in that presentation slice.

ii.
```python
'image_change': np.full(right - left, int(is_change), dtype=np.int64)
```

iii. Both label and neural data use the same `left:right` ophys interval.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from `exp.running_speed['speed']` and its `timestamps`.

ii.
```python
run_t = exp.running_speed['timestamps'].to_numpy(dtype=np.float64)
run_v = exp.running_speed['speed'].to_numpy(dtype=np.float32)
```

iii. This is the AllenSDK's processed running-wheel speed stream and timestamp base.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Finite source pairs are sorted, linearly interpolated onto each trial's ophys timestamps, collected across retained trials, and discretized with global five-quantile edges. Non-increasing duplicate edges are nudged upward.

ii.
```python
run_aligned = interp_to_ophys(run_t, run_v, trial_t)
edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
if edges[i] <= edges[i - 1]:
    edges[i] = edges[i - 1] + 1e-6
```

iii. Ophys interpolation guarantees sample alignment; global percentile bins create comparable, approximately balanced decoder classes across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Values are assigned to five categories by the four interior global quantile edges using `np.digitize`. Nonfinite aligned samples are filled with the median category among finite samples in that trial (or 0 if none), then clipped to 0–4.

ii.
```python
out = np.digitize(values, edges[1:-1], right=False).astype(np.int64)
fill = int(np.median(out[finite])) if finite.size else 0
out[bad] = fill
out = np.clip(out, 0, len(edges) - 2)
```

iii. The five equal-percentile bins are required by the task. Median-class imputation avoids invalid categorical targets for missing samples.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is interpolated directly to `trial_t = ophys_t[left:right]`, so its vector and neural slice share every sample timestamp.

ii.
```python
trial_t = ophys_t[left:right]
run_aligned = interp_to_ophys(run_t, run_v, trial_t)
```

iii. This follows the instruction to use the ophys clock as the common basis.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses `exp.eye_tracking`: preferentially `pupil_area`, converted to equivalent circular diameter; otherwise geometric mean diameter from `pupil_width * pupil_height`. `timestamps` and `likely_blink` are also used.

ii.
```python
if 'pupil_area' in et.columns:
    diam = 2.0 * np.sqrt(area / math.pi)
elif 'pupil_width' in et.columns and 'pupil_height' in et.columns:
    diam = np.sqrt(et['pupil_width'] * et['pupil_height'])
diam[blink] = np.nan
```

iii. The notes explicitly chose an area-derived diameter proxy when available and mask likely blinks to prevent artifacts.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Diameter is derived as above, blink samples become NaN, finite source samples are linearly interpolated to trial ophys timestamps, and all retained values define global five-quantile edges with duplicate-edge repair.

ii.
```python
pupil_t, pupil_v = pupil_diameter_series(exp.eye_tracking)
pupil_aligned = interp_to_ophys(pupil_t, pupil_v, trial_t)
pupil_edges = compute_bin_edges(np.concatenate(all_pupil))
```

iii. This produces a diameter-like measure, suppresses blink corruption, and yields globally comparable balanced classes.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It uses the same five-bin global quantile digitization and per-trial median-category missing-value fill as running speed.

ii.
```python
pupil_bin = digitize_with_edges(tr['pupil_raw'], pupil_edges)
```

iii. Five equal percentile bins are required; the common helper enforces valid 0–4 targets.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. The pupil series is interpolated at the exact ophys timestamps used by the neural presentation slice.

ii.
```python
pupil_aligned = interp_to_ophys(pupil_t, pupil_v, trial_t)
```

iii. The shared timestamp vector gives framewise alignment despite the eye tracker's native sampling clock.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It is derived from the parent behavioral trial's mutually exclusive `hit`, `miss`, `false_alarm`, and `correct_reject` flags.

ii.
```python
if bool(row.get('hit', False)): return 0
if bool(row.get('miss', False)): return 1
if bool(row.get('false_alarm', False)): return 2
if bool(row.get('correct_reject', False)): return 3
```

iii. These are the SDK's canonical outcomes for valid Go/Catch trials; rows without one are excluded.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The first true flag is mapped to a fixed 0–3 category, inherited by every image-presentation trial from its parent trial, and repeated across that presentation's neural frames.

ii.
```python
outcome = int(trial_outcome_map[parent_id])
np.full(T, tr['trial_outcome'], dtype=np.int64)
```

iii. Repeating the static label makes it compatible with the decoder's time-series output matrix while preserving the parent trial's outcome.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Trial flags use `fillna(False)`; incomplete outcome trials and malformed/missing presentation metadata are filtered. Interpolation ignores nonfinite source pairs and returns NaNs with fewer than two; digitization replaces nonfinite aligned values by the trial's median bin (or 0). Blink pupil samples are masked. Missing pupil measurement columns, missing event columns, inconsistent event lengths, absent NWBs, or no usable sessions raise explicit errors. Duplicate quantile edges are repaired.

ii.
```python
good = np.isfinite(src_t) & np.isfinite(src_v)
if good.sum() < 2:
    return np.full(dst_t.shape, np.nan, dtype=np.float32)
out[bad] = int(np.median(out[finite])) if finite.size else 0
raise KeyError('No pupil area/width-height columns available')
```

iii. The agent encountered pandas missing booleans and invalid `-1` image labels during development, patched both, and used explicit failure for structural corruption rather than silently producing mis-shaped neural data.

## 9-a. What are the most time-consuming steps of the code?

i. Loading and constructing each AllenSDK experiment from a large NWB is dominant; full-data assembly and writing the 4.7 GB pickle are also substantial. The agent parallelizes independent experiment loads over up to eight workers.

ii.
```python
exp = BehaviorOphysExperiment.from_nwb_path(str(nwb_path))
n_workers = min(max((os.cpu_count() or 2) // 2, 2), 8)
pool.imap_unordered(process_experiment, files)
pickle.dump(data, f)
```

iii. Serial sample timing was about 20–22 seconds per experiment, implying roughly 100 minutes; this prompted multiprocessing after smaller loop optimizations had little effect.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The Python loop over retained stimulus presentations (including repeated `searchsorted`, interpolation, allocation, and dictionary creation), row-wise trial outcome `apply`, quantile-edge repair loop, and final per-session/per-trial assembly loops could be partly vectorized or batched. The former `iterrows` presentation loop was already improved to zipped NumPy arrays.

ii.
```python
trials['trial_outcome_idx'] = trials.apply(get_trial_outcome, axis=1)
for stim_id, parent_id, start, stop, img_name, is_change in zip(...):
for s in processed_sessions:
    for tr in s['trials']:
```

iii. The notes acknowledge per-presentation iteration, but profiling-by-timing suggested SDK/NWB loading dominated, so multiprocessing offered the larger improvement.

## 9-c. What processing does the code repeat multiple times?

i. For every presentation, it independently interpolates running and pupil streams, performs timestamp searches, and creates constant label arrays. Adjacent presentations from the same experiment repeatedly sort/filter the same source timestamp arrays inside `interp_to_ophys`. It also calls `list_nwb_files()` once to select files and again only to print the count.

ii.
```python
src_t = src_t[good]
order = np.argsort(src_t)
src_t = src_t[order]
run_aligned = interp_to_ophys(run_t, run_v, trial_t)
pupil_aligned = interp_to_ophys(pupil_t, pupil_v, trial_t)
print(f'found {len(list_nwb_files())} nwb files; processing {len(files)}')
```

iii. The agent emphasized avoiding a second raw-data loading pass for binning, but did not call out these smaller repeated computations.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It stores per-trial `trial_id`, start/stop times, ophys timestamps, Go/Catch flags, full experiment metadata, and session ID during processing, but final assembly uses essentially only neural data, image/change labels, raw behaviors, outcome, subject, and region. Optional plotting also reads timestamps, but otherwise these fields are discarded. The unused `build_image_labels_for_trial` function remains from the abandoned full-trial approach.

ii.
```python
'meta': dict(meta),
'trial_id': int(stim_id),
'start_time': float(start),
'stop_time': float(stop),
'go': bool(trial_go_map[parent_id]),
'catch': bool(trial_catch_map[parent_id]),
```

iii. These fields helped debugging, plots, and provenance during development, but the final decoder dataset and training do not consume them.
