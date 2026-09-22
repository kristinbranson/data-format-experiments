# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local `ophys_experiment_table.csv`, enumerates locally downloaded NWB files, retains table rows whose experiment IDs have files, excludes session types containing `passive`, and sorts experiments. It then opens every retained NWB directly with `h5py` and reads ophys timestamps/dF/F, running, eye tracking, trials, and stimulus presentations. Thus “all” means all locally downloaded active experiments, including both project codes, rather than the reference's SDK-selected `VisualBehavior` sessions.

ii.
```python
nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))
exp_table = exp_table[exp_table['ophys_experiment_id'].isin(downloaded_ids)].copy()
exp_table = exp_table[~exp_table['session_type'].str.contains('passive', case=False)].copy()
...
with h5py.File(nwb_path, 'r') as f:
    data['ophys_timestamps'] = f['processing']['ophys']['dff']['traces']['timestamps'][:]
    dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
```

iii. The notes say direct HDF5 access is fast, only a downloaded subset exists, and active sessions alone are appropriate for the behavioral decoder. They report 284 downloaded files and 202 retained active experiments.

## 1-b. How are the data split into subjects?

i. Each processed experiment's metadata `mouse_id` is converted to a string and mapped to an insertion-ordered unique subject index.

ii.
```python
mouse_id = str(exp_row['mouse_id'])
if mouse_id not in subject_map:
    subject_map[mouse_id] = len(all_subjects)
    all_subjects.append(mouse_id)
all_subject_idx.append(subject_map[mouse_id])
```

iii. The agent treats `mouse_id` as the dataset's unique animal identifier and reports 38 mice in the downloaded subset.

## 1-c. How are the data split into sessions?

i. Every retained `ophys_experiment_id` (one imaging plane) becomes a separate output “session.” Experiments sharing an `ophys_session_id` are not combined.

ii.
```python
for idx, (_, row) in enumerate(exp_table.iterrows()):
    exp_id = row['ophys_experiment_id']
    result = process_experiment(exp_id, row, image_names_list, outcome_names,
                                show_processing=args.show_processing)
    ...
    all_neural.append(result['neural'])
```

iii. The notes explicitly justify this as allowing each plane, with its distinct neurons, to be a separate session even when multiscope planes share behavior. This differs from the reference, which groups planes by `ophys_session_id`.

## 1-d. How are the data split into trials?

i. Trial indices come from the NWB trials interval. For every valid trial, frames satisfying `start_time <= ophys_timestamp < stop_time` are selected, producing variable-length trial arrays.

ii.
```python
for trial_idx in valid_trial_idx:
    t_start = trial_data['start_time'][trial_idx]
    t_stop = trial_data['stop_time'][trial_idx]
    frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
    neural = dff[:, frame_mask].astype(np.float32)
```

iii. The agent says the SDK/NWB trial boundaries are the experimental definition and that full trial windows retain pre-change flashes and post-change behavior.

## 1-e. How are trials filtered based on quality controls?

i. It keeps trials marked go or catch and rejects aborted or auto-rewarded trials. Experiments with fewer than two initially valid trials, trials with fewer than two ophys frames, and experiments with fewer than two trials after processing are excluded. It also rejects experiments with no cells or stimulus table.

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

iii. The notes cite the task's explicit Go/Catch inclusion and aborted/auto-rewarded exclusion rules, plus the decoder's two-trial minimum.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is taken from the NWB `processing/ophys/dff/traces/data` dataset and transposed from frame-by-cell to cell-by-frame.

ii.
```python
dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]
data['dff_traces'] = dff_raw.T
```

iii. The agent chose precomputed dF/F as the standard calcium-imaging signal rather than deconvolved events.

## 2-b. How is the `neural` data processed?

i. Apart from transposition, trial slicing, and conversion to `float32`, no additional normalization, filtering, deconvolution, or temporal resampling is done. Unlike the reference, planes from the same session are not vertically stacked.

ii.
```python
dff = nwb_data['dff_traces']
...
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The notes state dF/F was already produced by the Allen pipeline and native ophys sampling should be preserved.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent does not explicitly consult `valid_roi` or apply cell-level QC; it includes every column present in the NWB dF/F matrix. It only drops an experiment if the matrix has zero cells.

ii.
```python
n_cells, n_frames = dff.shape
if n_cells == 0:
    return None
```

iii. The notes recognize `valid_roi` filtering in the SDK and appear to assume the NWB dF/F contents already reflect that curation, but the direct-HDF5 loader does not verify or reproduce it.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Ophys timestamps are the master timebase. Each neural trial begins at the first frame at or after trial `start_time` and ends before `stop_time`; alignment is therefore to trial start, not to image-change onset.

ii.
```python
frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)
trial_ts = ophys_ts[frame_mask]
neural = dff[:, frame_mask].astype(np.float32)
```

iii. The agent describes this as alignment to ophys timestamps across the complete trial, consistent with retaining time-varying stimulus and behavior.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Each experiment stays at its native ophys cadence (about 32.3 ms for Scientifica and about 91 ms for multiscope); no rebinning is applied. Metadata stores the median of experiment-level median frame intervals, about 32.32 ms for this subset, even though sessions do not all share that bin size.

ii.
```python
dt = np.median(np.diff(ophys_ts))
...
median_dt = np.median([m['dt_ms'] for m in session_metadata])
```

iii. The notes justify native sampling because timestamps are internally consistent, while also acknowledging mixed ~31 Hz and ~11 Hz recordings.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from stimulus-presentation `start_time`, `stop_time`, and `image_name`, evaluated on each trial's ophys timestamps. `omitted` is treated as gray.

ii.
```python
for key in ['start_time', 'stop_time', 'image_name', 'is_change', 'omitted']:
    ...
stim_starts = stim_data['start_time']
stim_stops = stim_data['stop_time']
stim_names = stim_data['image_name']
```

iii. The notes say stimulus presentations provide the actual flashed-image timing and that gray/ISI must be represented explicitly.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global category list is built by scanning retained NWBs, excluding `omitted`, sorting names, and prepending `gray`. Each trial trace starts as gray and is overwritten with an image code for frames inside each non-omitted presentation.

ii.
```python
image_names_list = [GRAY_LABEL] + all_image_names
trace = np.full(n_frames, gray_idx, dtype=np.int64)
...
frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
trace[frame_mask] = img_idx
```

iii. The agent reports checking plots and a 66.9% gray fraction against the expected 500/750 ms ISI fraction.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image labels are assigned directly to the exact `trial_ts` selected from the ophys timestamps, yielding one label per neural frame.

ii.
```python
trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)
trial_ts = ophys_ts[trial_mask]
...
frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)
```

iii. The agent relies on hardware-synchronized timestamps and verified equal neural/output lengths.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from stimulus-presentation `is_change` and `start_time`, rather than the trials-table `go` and `change_time` used by the reference.

ii.
```python
stim_starts = stim_data['start_time']
is_change = stim_data['is_change']
for si in range(len(stim_starts)):
    if not is_change[si]:
        continue
```

iii. The notes characterize `is_change` as the direct event annotation and intend a sparse onset indicator.

## 4-b. What processing is involved in computing `output` *Image change*?

i. It initializes zeros, finds every true change presentation within the trial, locates the first ophys frame at or after its onset, and marks only that frame as one.

ii.
```python
trace = np.zeros(n_frames, dtype=np.int64)
frame_idx = np.searchsorted(trial_ts, s_start)
if frame_idx < n_frames:
    trace[frame_idx] = 1
```

iii. The agent interprets “right after a change” as the onset frame and notes the resulting severe but intentional class imbalance.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numerical threshold is estimated. The boolean raw `is_change` annotation becomes category 1 at its onset frame; all other frames are category 0, named `no_change` and `change`.

ii.
```python
output_values = [..., ['no_change', 'change'], ...]
trace = np.zeros(n_frames, dtype=np.int64)
...
trace[frame_idx] = 1
```

iii. The underlying annotation is already binary, so the agent considered further thresholding unnecessary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change onset is mapped by `searchsorted` onto the same per-trial ophys timestamps used for neural frames.

ii.
```python
trial_ts = ophys_ts[trial_mask]
frame_idx = np.searchsorted(trial_ts, s_start)
```

iii. The agent's plots reportedly confirmed that change impulses coincide with the relevant stimulus onset.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses NWB `processing/running/speed/data` and its timestamps.

ii.
```python
data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]
data['running_speed'] = f['processing']['running']['speed']['data'][:]
```

iii. This is the dataset's processed running-speed stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Full-session running speed is linearly interpolated to ophys timestamps. Five percentile edges are computed over that experiment's entire interpolated session, then applied to valid trial samples. No extra low-pass filter is applied by the conversion.

ii.
```python
running_at_ophys = interpolate_to_ophys(
    nwb_data['running_speed'], nwb_data['running_timestamps'], ophys_ts)
running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)
running_binned = apply_percentile_bins(running_trial, running_edges, n_bins=5)
```

iii. The agent chose session-wide percentiles to create roughly equal bins within each experiment while retaining consistent thresholds across that experiment's trials.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Non-NaN values are assigned with 20th, 40th, 60th, and 80th percentile cut points into integer bins 0–4. End edges are set to infinities; NaNs are forced to bin 0.

ii.
```python
edges = np.percentile(valid, np.linspace(0, 100, n_bins + 1))
edges[0] = -np.inf
edges[-1] = np.inf
result[valid] = np.clip(np.digitize(values[valid], edges[1:-1]), 0, n_bins - 1)
result[~valid] = 0
```

iii. The five equal-percentile bins are required by the task; bin 0 is used as the agent's fallback for missing samples.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. The entire running trace is linearly interpolated onto the full ophys clock, and the neural trial's identical Boolean frame mask selects running values.

ii.
```python
running_at_ophys = interpolate_to_ophys(..., ophys_ts)
...
running_trial = running_at_ophys[frame_mask]
```

iii. The notes cite common hardware synchronization and direct ophys-time interpolation.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses pupil ellipse `area`, eye-tracking timestamps, and `likely_blink` from the NWB acquisition group. This differs from the reference's direct `pupil_width` measure.

ii.
```python
data['pupil_area'] = pt['area']['data'][:] if 'data' in pt['area'] else pt['area'][:]
data['pupil_timestamps'] = pt['timestamps'][:]
data['likely_blink'] = et['likely_blink']['data'][:]
```

iii. The agent reasons that an equivalent-circle diameter can be calculated from ellipse area and that blink flags should invalidate contaminated frames.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink samples are set to NaN, positive areas are converted using `2*sqrt(area/pi)`, and the resulting array is passed directly to linear interpolation onto ophys timestamps. Five session-wide percentile bins are then applied. Because NaNs remain in the interpolation input, interpolation can propagate missingness around blink intervals rather than interpolate across removed blink samples.

ii.
```python
pupil_area[likely_blink] = np.nan
pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
```

iii. The notes describe area-to-diameter conversion, blink invalidation, ophys interpolation, and per-session percentiles as the planned processing.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It uses the same per-experiment 20/40/60/80 percentile procedure as running, with integer bins 0–4 and NaNs assigned to 0.

ii.
```python
pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)
pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)
```

iii. The agent cites the requested five equal percentile bins and documents NaN-to-bin-0 as a design choice.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Derived pupil diameter is linearly interpolated onto all ophys timestamps and then sliced with the same trial frame mask as neural data.

ii.
```python
pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)
...
pupil_trial = pupil_at_ophys[frame_mask]
```

iii. The agent relies on synchronized acquisition clocks and reports checking matching temporal dimensions.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It checks the trials-table boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in that priority order.

ii.
```python
if trial_data['hit'][idx]:
    return 'hit'
elif trial_data['miss'][idx]:
    return 'miss'
elif trial_data['false_alarm'][idx]:
    return 'false_alarm'
elif trial_data['correct_reject'][idx]:
    return 'correct_reject'
```

iii. The agent treats these as the canonical mutually exclusive outcomes for retained Go/Catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The selected name is converted to its index in `[hit, miss, false_alarm, correct_reject]` (unknown falls back to 0), then broadcast over every timepoint as the fifth output row.

ii.
```python
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
output_full[4] = outcome_idx
```

iii. Broadcasting makes the static per-trial label compatible with the single `(5, T)` output array expected by the validator.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing NWBs are excluded by the initial file intersection; unreadable files during image scanning generate warnings; experiments missing files, cells, stimulus data, or enough trials return `None`. Missing pupil data becomes all NaN. Interpolation outside source ranges and invalid/blink pupil samples yield NaNs, which are encoded as bin 0. Unknown outcomes also become class 0. Most exceptions in full experiment processing are not caught and would terminate conversion.

ii.
```python
if not os.path.exists(nwb_path):
    return None
...
else:
    pupil_at_ophys = np.full(len(ophys_ts), np.nan)
...
result[~valid] = 0
outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0
```

iii. The agent presents bin 0 as a safe discrete fallback and uses warnings/skips to preserve a valid dataset when individual experiments lack required content.

## 9-a. What are the most time-consuming steps of the code?

i. Loading each very large NWB/dF/F array dominates per-experiment runtime; serialization of the 8.5 GB pickle is also substantial. The code records loading and processing durations separately and scans every NWB once beforehand for image names.

ii.
```python
nwb_data = load_nwb_data(nwb_path)
t_load = time.time() - t0
...
with open(args.output, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. The notes estimate about 1.7 seconds loading versus 0.4 seconds processing per sample experiment and about 14 seconds for the image-name scan.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The loop over all stimulus presentations for every trial is the main avoidable nested loop; it repeatedly tests presentations far outside the trial. Trial masks and percentile bin application are already vectorized. Image-name scanning and per-trial processing could also be consolidated/indexed, though variable trial lengths still require some iteration.

ii.
```python
for trial_idx in valid_trial_idx:
    ...
    for si in range(len(stim_starts)):
        if s_stop < trial_start:
            continue
        if s_start >= trial_stop:
            break
```

iii. The agent did not discuss vectorization in its notes; it prioritized direct HDF5 loading and reported processing as much cheaper than I/O.

## 9-c. What processing does the code repeat multiple times?

i. Each NWB is opened once in `get_all_image_names` and again in `process_experiment`. Experiments sharing an ophys session independently reload and recompute identical trials, stimulus mappings, running, pupil, and bin edges. Within an experiment, `build_image_identity_trace` and `build_image_change_trace` separately rebuild the same trial mask and scan stimulus presentations for every trial.

ii.
```python
all_image_names = get_all_image_names(exp_table)
...
result = process_experiment(exp_id, row, image_names_list, outcome_names, ...)
```

iii. The image scan was intended to make a stable global category list. The repeated work caused by treating planes separately follows from the agent's stated multiscope-session decision, but was not identified as a cost.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `cell_roi_ids` is loaded but never used. `output_tv` and `output_static` are constructed and then discarded in favor of `output_full`. Trial timestamps are materialized in `process_experiment` mainly for a length check, while helper functions recompute them. With `--show-processing`, large raw arrays and tables are retained only for diagnostic plots. The full pre-scan of image names also reads every NWB before normal processing.

ii.
```python
data['cell_roi_ids'] = seg[key]['id'][:]
...
output_tv = np.stack([img_trace, change_trace, running_binned, pupil_binned], axis=0)
output_static = np.array([outcome_idx], dtype=np.int64)
output_full = np.zeros((5, n_trial_frames), dtype=np.int64)
```

iii. The agent justified optional retained arrays as visualization support and the pre-scan as category discovery, but did not mention the dead ROI/output intermediates.
