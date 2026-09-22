# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Reads the project metadata CSV directly, discovers locally downloaded NWB files, keeps only rows whose experiment IDs have files, excludes session types containing `passive`, and opens each remaining experiment independently with `h5py`. Thus “all” means the 202 active downloaded experiments, not every experiment in the metadata/cache.

ii.
```python
`nwb_files = glob.glob(os.path.join(NWB_DIR, '*.nwb'))`
`exp_table = exp_table[exp_table['ophys_experiment_id'].isin(downloaded_ids)]`
`exp_table = exp_table[~exp_table['session_type'].str.contains('passive', case=False)]`
`with h5py.File(nwb_path, 'r') as f:`
```

iii. The notes say direct `h5py` loading is faster than AllenSDK, the downloaded subset has 38 mice, and passive sessions were intentionally excluded because the decoder concerns active behavior.

## 1-b. How are the data split into subjects?

i. Subjects are unique metadata `mouse_id` values encountered among successfully processed experiments. A dictionary maps their string form to indices.

ii.
```python
`mouse_id = str(exp_row['mouse_id'])`
`if mouse_id not in subject_map:`
`    subject_map[mouse_id] = len(all_subjects)`
`    all_subjects.append(mouse_id)`
```

iii. The notes identify `mouse_id` as the subject identifier and report 38 downloaded mice.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id` (one imaging plane) becomes a separate output “session”; experiments sharing an `ophys_session_id` are not combined.

ii.
```python
`for idx, (_, row) in enumerate(exp_table.iterrows()):`
`    exp_id = row['ophys_experiment_id']`
`    result = process_experiment(exp_id, row, ...)`
`    all_neural.append(result['neural'])`
```

iii. The mapping plan explicitly states that each multiscope experiment/plane will be a separate output session because planes have different neurons, even though they share behavior.

## 1-d. How are the data split into trials?

i. Uses NWB `intervals/trials`; for each valid trial it selects ophys frames in the half-open interval `[start_time, stop_time)`.

ii.
```python
`valid_trial_idx = get_valid_trials(trial_data)`
`frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)`
`neural = dff[:, frame_mask].astype(np.float32)`
```

iii. The notes say trials follow the experiment’s trial table and retain the full variable-length trial window.

## 1-e. How are trials filtered based on quality controls?

i. Keeps `(go | catch) & ~aborted & ~auto_rewarded`, skips trials with fewer than two ophys frames, and drops an experiment if fewer than two processed trials remain. It also excludes all passive experiments.

ii.
```python
`valid = (go | catch) & ~aborted & ~auto_rewarded`
`if n_trial_frames < 2: continue`
`if len(neural_trials) < 2: return None`
```

iii. This directly follows the requested Go/Catch inclusion and Aborted/Auto-rewarded exclusion; the two-trial minimum satisfies decoder validation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Uses the precomputed NWB dF/F trace dataset at `processing/ophys/dff/traces/data`, transposed from time-by-cell to cell-by-time.

ii.
```python
`dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]`
`data['dff_traces'] = dff_raw.T`
```

iii. The notes choose dF/F rather than deconvolved events because it is the standard precomputed calcium signal.

## 2-b. How is the `neural` data processed?

i. No numerical transformation beyond using precomputed dF/F, transposition, trial slicing, and `float32` conversion. Planes from one behavioral session remain separate.

ii.
```python
`dff = nwb_data['dff_traces']`
`neural = dff[:, frame_mask].astype(np.float32)`
```

iii. The notes state dF/F already contains the Allen preprocessing and therefore need not be recomputed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit cell/ROI filter is applied. Empty-cell experiments are skipped.

ii.
```python
`if n_cells == 0:`
`    return None`
```

iii. The notes acknowledge AllenSDK normally applies `exclude_invalid_rois=True`, then assert all ROIs in the downloaded NWBs are valid and report a cell-table count match.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All streams are placed on ophys timestamps; neural samples are selected from trial start through, but excluding, trial stop. The metadata describes alignment to ophys timestamps/trial bounds.

ii.
```python
`frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)`
`trial_ts = ophys_ts[frame_mask]`
```

iii. The stated rationale is that the trial table defines the trial and ophys timestamps are the required master clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is done. Each experiment retains native sampling (~31 Hz Scientifica or ~11 Hz Multiscope); metadata stores the median per-experiment interval across output sessions (reported 32.32 ms), even though individual sessions can differ.

ii.
```python
`dt = np.median(np.diff(ophys_ts))`
`median_dt = np.median([m['dt_ms'] for m in session_metadata])`
```

iii. The notes deliberately retain native ophys timestamps and claim within-session consistency; they recognize both 31 Hz and 11 Hz sources.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Derived from stimulus-presentation `start_time`, `stop_time`, and `image_name`, aligned against ophys timestamps; trial bounds come from the trials table.

ii.
```python
`stim_starts = stim_data['start_time']`
`stim_stops = stim_data['stop_time']`
`stim_names = stim_data['image_name']`
```

iii. The notes prefer the presentation table because it captures actual flashes and gray inter-stimulus periods.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Builds a global sorted image-name vocabulary, prepends a `gray` class, initializes every trial frame to gray, then assigns the image code during each non-omitted presentation overlap.

ii.
```python
`image_names_list = [GRAY_LABEL] + all_image_names`
`trace = np.full(n_frames, gray_idx, dtype=np.int64)`
`trace[frame_mask] = img_idx`
```

iii. The agent says this represents 250-ms images and 500-ms gray periods faithfully; omitted presentations remain gray.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Creates `trial_ts` with the same half-open trial mask as neural data and labels frames whose timestamps fall in each presentation’s `[start, stop)` interval.

ii.
```python
`trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)`
`frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)`
```

iii. The notes report spot-checking exact equality against raw NWB presentations and the expected ~66.7% gray fraction.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Derived from stimulus-presentation `is_change` and `start_time`.

ii.
```python
`stim_starts = stim_data['start_time']`
`is_change = stim_data['is_change']`
```

iii. The notes use the presentation-level flag as the direct source of actual image changes.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Initializes zeros, loops over presentations flagged as changes, and marks only the first ophys frame at or after each change onset.

ii.
```python
`frame_idx = np.searchsorted(trial_ts, s_start)`
`if frame_idx < n_frames: trace[frame_idx] = 1`
```

iii. The rationale is that the instruction asks for 1 right after a change and 0 otherwise; notes report a sparse ~0.372% positive fraction.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary: `0 = no_change`, `1 = change`; no numeric threshold is applied.

ii.
```python
`output_values[1] = ['no_change', 'change']`
`trace = np.zeros(n_frames, dtype=np.int64)`
```

iii. The underlying NWB `is_change` boolean supplies the category directly.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change onset is mapped into the same `trial_ts` array used for trial neural frames using `searchsorted`.

ii.
```python
`trial_ts = ophys_ts[trial_mask]`
`frame_idx = np.searchsorted(trial_ts, s_start)`
```

iii. The agent reports checking alignment to raw stimulus onset and image-identity transitions.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Uses NWB `processing/running/speed/data` and its timestamps.

ii.
```python
`data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]`
`data['running_speed'] = f['processing']['running']['speed']['data'][:]`
```

iii. The notes identify this as the standard running-wheel speed stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linearly interpolates the whole running trace onto ophys timestamps, computes five percentile bins over all non-NaN timepoints in that experiment (including outside retained trials), then applies those edges per trial. NaNs become bin 0.

ii.
```python
`running_at_ophys = interpolate_to_ophys(..., ophys_ts)`
`running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)`
`running_binned = apply_percentile_bins(running_trial, running_edges, n_bins=5)`
```

iii. The agent chose session-wide percentiles to make five equal bins within each recording and linear interpolation for synchronized clocks.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Thresholds are the experiment-wide 0,20,40,60,80,100 percentiles; interior four edges are passed to `digitize`, clipped to 0–4.

ii.
```python
`edges = np.percentile(valid, np.linspace(0, 100, n_bins + 1))`
`result[valid] = np.clip(np.digitize(values[valid], edges[1:-1]), 0, n_bins - 1)`
```

iii. The stated goal is five equal percentile bins; missing values are conservatively assigned category 0.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Interpolates speed to the full ophys clock first, then indexes speed and dF/F with the identical trial frame mask.

ii.
```python
`running_at_ophys = interpolate_to_ophys(..., ophys_ts)`
`running_trial = running_at_ophys[frame_mask]`
```

iii. The notes cite hardware synchronization and raw-data spot checks.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Uses EyeTracking pupil `area`, pupil timestamps, and `likely_blink`; it does not use the raw `pupil_width` variable used by the reference.

ii.
```python
`data['pupil_area'] = pt['area']['data'][:]`
`data['likely_blink'] = et['likely_blink']['data'][:]`
```

iii. The notes interpret pupil area as a circle and use blink flags to reject artifacts.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Sets blink areas to NaN, converts positive area to equivalent-circle diameter `2*sqrt(area/pi)`, linearly interpolates that array to ophys timestamps, computes experiment-wide percentile edges, and maps remaining NaNs to bin 0.

ii.
```python
`pupil_area[likely_blink] = np.nan`
`pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)`
`pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)`
```

iii. The agent cites the whitepaper formula and says blink-to-NaN handling avoids artifacts.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Uses experiment-wide 20th-percentile boundaries over interpolated non-NaN samples, digitized into categories 0–4; NaNs are category 0.

ii.
```python
`pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)`
`pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)`
```

iii. The choice mirrors the agent’s running-speed scheme and is intended to balance categories within each experiment.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolates pupil diameter onto the complete ophys clock, then uses the same trial frame mask as dF/F.

ii.
```python
`pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)`
`pupil_trial = pupil_at_ophys[frame_mask]`
```

iii. The notes cite synchronized acquisition and a raw-NWB equality spot check.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Uses the trial-table boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject`, checked in that order.

ii.
```python
`if trial_data['hit'][idx]: return 'hit'`
`elif trial_data['miss'][idx]: return 'miss'`
`elif trial_data['false_alarm'][idx]: return 'false_alarm'`
```

iii. The agent treats these as the canonical mutually exclusive outcomes.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Maps the outcome name to its fixed index and broadcasts the scalar across all trial timepoints in the final `(5,T)` output.

ii.
```python
`outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0`
`output_full[4] = outcome_idx`
```

iii. The comments recognize the outcome is static but repeat it because the target stores a single rectangular output array; the reference does likewise.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing files are prefiltered; absent stimulus data, zero-cell experiments, and experiments/trials with too little data are skipped. Missing pupil data becomes all-NaN then bin 0; interpolation outside bounds and other NaNs also become bin 0. Unknown outcome falls back to class 0. File-read failures during image-name scanning warn and continue, but the main experiment loop has no general exception handler.

ii.
```python
`if stim_data is None: return None`
`pupil_at_ophys = np.full(len(ophys_ts), np.nan)`
`result[~valid] = 0`
`outcome_idx = ... else 0`
```

iii. The notes call NaN-to-bin-0 a documented design choice and retain any experiment with at least two valid trials.

## 9-a. What are the most time-consuming steps of the code?

i. Reading large NWB arrays is dominant (`load_nwb_data`), followed by trial processing; reported estimates were ~1.7 s load and ~0.4 s processing per experiment. The full conversion also writes an 8.5-GB pickle.

ii.
```python
`nwb_data = load_nwb_data(nwb_path)`
`t_load = time.time() - t0`
`t_process = time.time() - t0 - t_load`
```

iii. The notes use measured timings and report ~9–12.5 minutes for the full run.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The valid-trial loop, image-presentation loop inside every trial, change-presentation loop, downloaded-file ID loop, image-name scan, and experiment loop are serial. Most notably, every trial scans all session presentations twice; overlap candidates and frame indices could be found once with sorted searches/vectorized interval mapping.

ii.
```python
`for trial_idx in valid_trial_idx:`
`    ... build_image_identity_trace(...)`
`for si in range(len(stim_starts)):`
```

iii. The agent emphasized direct h5py speed but did not document or implement vectorization/parallelism for these loops.

## 9-c. What processing does the code repeat multiple times?

i. Every NWB is opened once by `get_all_image_names` and again for conversion. Within each trial it recomputes the trial mask in the main loop and both trace builders, and independently scans presentations for identity and change.

ii.
```python
`all_image_names = get_all_image_names(exp_table)`
`nwb_data = load_nwb_data(nwb_path)`
`trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)`
```

iii. The notes mention the image scan’s ~14-s overhead but accept it to create a global vocabulary.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It constructs `output_tv` and `output_static` but never uses either, creates `trial_ts` in the main loop without using it, loads `cell_roi_ids` but never consumes them, and stores timing/debug fields that are only transient. It also scans whole stimulus tables repeatedly when only overlapping rows matter.

ii.
```python
`output_tv = np.stack([...])`
`output_static = np.array([outcome_idx], dtype=np.int64)`
`trial_ts = ophys_ts[frame_mask]`
`data['cell_roi_ids'] = seg[key]['id'][:]`
```

iii. No explicit justification is given; these are remnants of development/debugging and incur avoidable memory/CPU work.


