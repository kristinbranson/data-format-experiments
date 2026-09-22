# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the project experiment-table CSV, finds locally downloaded NWB files, keeps only matching experiments, excludes session types containing `passive`, and reads each remaining NWB directly with `h5py`. Thus “all” means all 202 locally downloaded active experiments, including both single-plane and multiscope project data.

ii.
```python
exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))
nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
exp_table = exp_table[exp_table['ophys_experiment_id'].isin(downloaded_ids)].copy()
exp_table = exp_table[~exp_table['session_type'].str.contains('passive', case=False)].copy()
...
with h5py.File(nwb_path, 'r') as f:
```

iii. The notes justify direct HDF5 access as faster than AllenSDK, limit processing to the downloaded subset, and exclude passive sessions because the requested task is active Visual Behavior. They explicitly report 202 active experiments, 38 mice, and 284 downloaded NWBs.

## 1-b. How are the data split into subjects?

i. Subjects are unique metadata `mouse_id` values encountered among successfully processed experiments. String mouse IDs are mapped to integer indices.

ii.
```python
mouse_id = str(exp_row['mouse_id'])
if mouse_id not in subject_map:
    subject_map[mouse_id] = len(all_subjects)
    all_subjects.append(mouse_id)
all_subject_idx.append(subject_map[mouse_id])
```

iii. The notes identify `mouse_id` as the subject mapping and sanity-check the resulting 38 downloaded mice.

## 1-c. How are the data split into sessions?

i. Every `ophys_experiment_id` (one imaging plane) becomes a separate output “session,” even when several experiments share one `ophys_session_id`.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    result = process_experiment(exp_id, row, image_names_list, outcome_names,
                                show_processing=args.show_processing)
    all_neural.append(result['neural'])
```

iii. The agent explicitly decided that multiscope planes should be separate output sessions because they contain different neurons while sharing behavior. Its notes distinguish 202 experiments from 174 unique biological sessions.

## 1-d. How are the data split into trials?

i. The NWB trials table defines each trial. For every valid trial, samples with `start_time <= ophys_timestamp < stop_time` are selected, producing variable-length trials.

ii.
```python
valid_trial_idx = get_valid_trials(trial_data)
...
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The notes say the built-in trial boundaries are the experiment-defined Go/Catch trials and preserve the full pre- and post-change interval.

## 1-e. How are trials filtered based on quality controls?

i. Only `(go OR catch) AND NOT aborted AND NOT auto_rewarded` trials are retained. Experiments with fewer than two such trials, or fewer than two processed trials, are skipped; individual trials with fewer than two ophys frames are skipped.

ii.
```python
valid = (go | catch) & ~aborted & ~auto_rewarded
...
if len(valid_trial_idx) < 2:
    return None
...
if n_trial_frames < 2:
    continue
```

iii. This is justified directly by the task’s Go/Catch inclusion and aborted/auto-rewarded exclusion rule. The two-trial minimum is required by decoder validation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the precomputed NWB dF/F trace dataset and its ophys timestamps.

ii.
```python
data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T
```

iii. The agent chose dF/F rather than deconvolved events because dF/F is the standard calcium-imaging signal and is already computed by the Allen pipeline.

## 2-b. How is the `neural` data processed?

i. The stored frame-by-cell dF/F matrix is transposed to cell-by-frame, cast to `float32`, and sliced per trial. No filtering, normalization, event inference, or plane merging is performed.

ii.
```python
data['dff_traces'] = dff_raw.T
...
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The notes rely on the NWB’s precomputed dF/F pipeline and state that no recomputation is needed. Separate-plane treatment follows the agent’s experiment-as-session decision.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit ROI-quality filtering. An experiment with zero cells is dropped; otherwise every column stored in the NWB dF/F dataset is used.

ii.
```python
n_cells, n_frames = dff.shape
if n_cells == 0:
    return None
```

iii. The notes say AllenSDK normally uses `exclude_invalid_rois=True` and assert that all ROIs in the downloaded NWBs are valid, so no extra filter was applied.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples are aligned on the ophys clock and segmented from trial start through (but excluding) trial stop. There is no fixed change-centered window.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
trial_ts = ophys_ts[frame_mask]
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The agent says using the same ophys timestamps for every stream guarantees alignment and retains the full trial’s changing stimulus and behavior.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. Each experiment stays at its native ophys cadence (about 32.3 ms for Scientifica or about 91 ms for multiscope), although the dataset metadata reports only the median experiment `dt` (32.32 ms).

ii.
```python
dt = np.median(np.diff(ophys_ts))
...
all_dts = [m['dt_ms'] for m in session_metadata]
median_dt = np.median(all_dts)
```

iii. The notes justify native sampling as preserving the acquired data and explicitly acknowledge both ~31 Hz and ~11 Hz recordings. They did not reconcile this with the requested common bin size across sessions.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from stimulus-presentation `start_time`, `stop_time`, and `image_name`, aligned using ophys timestamps; `omitted` is treated as gray.

ii.
```python
for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
    ...
stim_starts = stim_data['start_time']
stim_stops = stim_data['stop_time']
stim_names = stim_data['image_name']
```

iii. The agent chose the presentation table because it represents actual image-on and gray intervals, rather than inferring identity only from trial-level initial/change names.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global sorted list of image names is collected by scanning all experiments. `gray` is prepended. Each trial is initialized to gray, then frames falling within non-omitted stimulus presentations receive the presentation’s categorical index.

ii.
```python
image_names_list = [GRAY_LABEL] + all_image_names
trace = np.full(n_frames, gray_idx, dtype=np.int64)
...
frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
trace[frame_mask] = img_idx
```

iii. The notes justify gray as the inter-stimulus category and report a 66.9% gray fraction matching the 500/750 ms duty cycle.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Presentation intervals are evaluated at the exact trial ophys timestamps used to slice neural data, yielding one identity label per neural frame.

ii.
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
```

iii. The agent reports spot-checking image labels against raw NWB stimulus onset times and finding exact agreement.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It is derived from stimulus-presentation `is_change` and `start_time`, not trial-level `go` and `change_time`.

ii.
```python
stim_starts = stim_data['start_time']
is_change = stim_data['is_change']
```

iii. The agent considered the presentation table the direct source of actual image transitions.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code creates zeros and, for each `is_change` presentation inside the trial, sets only the first ophys frame at or after onset to one.

ii.
```python
trace = np.zeros(n_frames, dtype=np.int64)
...
frame_idx = np.searchsorted(trial_ts, s_start)
if frame_idx < n_frames:
    trace[frame_idx] = 1
```

iii. The notes describe the requested “right after” change indicator as an onset impulse and report the resulting sparse 0.372% positive fraction.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary: `0` is `no_change` and `1` is `change`; no numerical threshold is applied.

ii.
```python
output_values = [..., ['no_change', 'change'], ...]
trace = np.zeros(n_frames, dtype=np.int64)
trace[frame_idx] = 1
```

iii. The raw `is_change` flag is boolean, so the agent viewed further thresholding as unnecessary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change onset is mapped to the first trial ophys timestamp at or after the stimulus start, on the same frame grid as neural data.

ii.
```python
trial_ts = ophys_ts[trial_mask]
frame_idx = np.searchsorted(trial_ts, s_start)
```

iii. The agent’s validation notes say image-change impulses were checked against actual stimulus onsets.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses NWB running `speed/data` and `speed/timestamps`.

ii.
```python
data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
data['running_speed'] = f['processing']['running']['speed']['data'][:]
```

iii. The notes identify this as the standard wheel-derived running stream sampled near 60 Hz.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Speed is linearly interpolated over the whole experiment onto ophys timestamps. Five percentile edges are computed separately for each experiment from all session-wide interpolated values, then applied to trial slices.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts)
running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)
running_binned = apply_percentile_bins(running_trial, running_edges, n_bins=5)
```

iii. The agent says interpolation synchronizes the 60 Hz stream with ophys, while per-session percentiles make five equal-frequency categories within each recording.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Non-NaN session-wide values define 0th, 20th, 40th, 60th, 80th, and 100th percentile edges. `np.digitize` assigns labels 0–4; NaNs are assigned 0.

ii.
```python
edges = np.percentile(valid, percentiles)
edges[0] = -np.inf
edges[-1] = np.inf
result[valid] = np.clip(np.digitize(values[valid], edges[1:-1]), 0, n_bins - 1)
result[~valid] = 0
```

iii. Equal percentile bins were chosen to balance decoder classes; bin 0 was used as the missing-data fallback.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. The full running stream is interpolated to all ophys timestamps, then the same trial frame mask used for neural data selects running labels.

ii.
```python
running_at_ophys = interpolate_to_ophys(..., ophys_ts)
running_trial = running_at_ophys[frame_mask]
```

iii. The notes cite hardware synchronization and spot-check exact reproduction of the computed bins.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses eye-tracking pupil `area`, eye timestamps, and `likely_blink`; it does not use the NWB/SDK `pupil_width` variable.

ii.
```python
data['pupil_area'] = pt['area']['data'][:] if 'data' in pt['area'] else pt['area'][:]
data['pupil_timestamps'] = pt['timestamps'][:]
data['likely_blink'] = et['likely_blink']['data'][:]
```

iii. The notes cite the whitepaper’s ellipse/area data and choose an area-derived equivalent circular diameter.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink samples are set to NaN; positive pupil areas are transformed as `2*sqrt(area/pi)`; this signal is linearly interpolated to ophys timestamps. Five experiment-wide percentile bins are then computed and applied.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
```

iii. The agent says blink removal prevents artifacts and the circular-area formula supplies diameter. It chose session percentile bins for balanced classes.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The same per-experiment quintile procedure as running is used; non-NaNs map to 0–4 and NaNs map to 0.

ii.
```python
pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)
result[~valid] = 0
```

iii. The notes call NaN-to-bin-0 a deliberate design choice and otherwise aim for five equal percentile groups.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. The derived pupil series is interpolated to the full ophys clock, then indexed with the neural trial mask.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
pupil_trial = pupil_at_ophys[frame_mask]
```

iii. The agent says common ophys timestamps provide alignment and reports a raw-NWB spot check of the pupil bins.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome is selected from trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
if trial_data['hit'][idx]: return 'hit'
elif trial_data['miss'][idx]: return 'miss'
elif trial_data['false_alarm'][idx]: return 'false_alarm'
elif trial_data['correct_reject'][idx]: return 'correct_reject'
```

iii. These are the canonical mutually exclusive outcomes for valid Go and Catch trials; the notes validate their aggregate rates and raw labels.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The name is converted to its index in `[hit, miss, false_alarm, correct_reject]` and broadcast across every frame of the trial. Unknown outcomes fall back to code 0.

ii.
```python
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
output_full[4] = outcome_idx
```

iii. Broadcasting makes the static target fit the single `(n_outputs, time)` array expected by the decoder.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing NWBs, cells, stimulus tables, or adequate trials cause an experiment to be skipped. Missing pupil produces an all-NaN signal. Interpolation outside support produces NaN, and both behavioral binners encode NaN as category 0. Unrecognized images remain gray; unreadable files during image-name scanning emit a warning and processing continues.

ii.
```python
if not os.path.exists(nwb_path): return None
if stim_data is None: return None
...
pupil_at_ophys = np.full(len(ophys_ts), np.nan)
...
result[~valid] = 0
...
except Exception as e:
    print(f"  WARNING: Could not read images from {eid}: {e}")
```

iii. The agent favors retaining usable experiments and assigning a valid discrete label to missing behavioral samples so validation/training can proceed.

## 9-a. What are the most time-consuming steps of the code?

i. Reading the large dF/F and behavior arrays from every NWB dominates; trial processing is next. The preliminary image-name pass reopens every NWB but was measured at only about 14 seconds. The notes estimate roughly 1.7 s load and 0.4 s trial processing per experiment.

ii.
```python
nwb_data = load_nwb_data(nwb_path)
t_load = time.time() - t0
...
all_image_names = get_all_image_names(exp_table)
```

iii. These timings and a roughly 12.5-minute projected full conversion are recorded in the sample-validation notes.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The image-name file scan, per-experiment processing loop, per-trial loop, and especially the full stimulus-presentation loops inside both trace builders are Python loops. Presentation-to-frame mapping and change-onset placement could be vectorized/searchsorted once per session, then sliced per trial.

ii.
```python
for _, row in exp_table.iterrows():
...
for trial_idx in valid_trial_idx:
...
for si in range(len(stim_starts)):
```

iii. The agent did not discuss vectorization in its notes; it emphasized that loading dominates and accepted these readable loops.

## 9-c. What processing does the code repeat multiple times?

i. Every NWB is opened once to collect image names and again for conversion. Within an experiment, every trial separately scans all stimulus presentations once for identity and again for changes, and recomputes its trial mask in both helpers in addition to `process_experiment`.

ii.
```python
all_image_names = get_all_image_names(exp_table)
...
nwb_data = load_nwb_data(nwb_path)
...
img_trace, _ = build_image_identity_trace(...)
change_trace = build_image_change_trace(...)
```

iii. No justification is documented. The extra image scan supports a global deterministic category list, but the repeated within-trial scans are an implementation convenience.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `cell_roi_ids` and several trial/stimulus fields are read but never used. `output_tv` and `output_static` are constructed and then discarded before `output_full` is built. The returned trial mask from the identity helper is ignored. Optional plotting retains large full-session arrays only when requested.

ii.
```python
data['cell_roi_ids'] = seg[key]['id'][:]
...
output_tv = np.stack([img_trace, change_trace, running_binned, pupil_binned], axis=0)
output_static = np.array([outcome_idx], dtype=np.int64)
output_full = np.zeros((5, n_trial_frames), dtype=np.int64)
```

iii. The comments show that `output_tv`/`output_static` are remnants of resolving mixed static/time-varying output formatting. No downstream need for the other unused reads is documented.
