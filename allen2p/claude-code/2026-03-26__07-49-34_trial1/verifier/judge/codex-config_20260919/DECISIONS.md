# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local ophys experiment CSV, discovers downloaded NWB files, keeps downloaded non-passive experiments, and loads each NWB directly with `h5py`. It processes 202 active experiments rather than reconstructing sessions from all planes.

ii. `exp_table = pd.read_csv(...'ophys_experiment_table.csv')`; `nwb_files = glob.glob(...'*.nwb')`; `exp_table = exp_table[exp_table['ophys_experiment_id'].isin(downloaded_ids)]`; `exp_table = exp_table[~exp_table['session_type'].str.contains('passive', case=False)]`; `with h5py.File(nwb_path, 'r') as f:`

iii. The notes say direct HDF5 loading is fast and equivalent to the SDK, that the available data are a downloaded subset, and that only active sessions should be used.

## 1-b. How are the data split into subjects?

i. Subjects are unique `mouse_id` values encountered among successfully processed experiments, mapped to integer indices.

ii. `mouse_id = str(exp_row['mouse_id'])`; `subject_map[mouse_id] = len(all_subjects)`; `all_subject_idx.append(subject_map[mouse_id])`

iii. The notes identify `mouse_id` as the subject identifier and report 38 downloaded subjects.

## 1-c. How are the data split into sessions?

i. Every `ophys_experiment_id` (one imaging plane) becomes a separate output session; experiments sharing an `ophys_session_id` are not merged.

ii. `for idx, (_, row) in enumerate(exp_table.iterrows()):`; `result = process_experiment(exp_id, row, ...)`; `all_neural.append(result['neural'])`

iii. The notes explicitly decide that each multiscope plane is a separate session because planes contain different neurons while sharing behavior.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. For each valid row, ophys frames satisfying `start_time <= timestamp < stop_time` form a variable-length trial.

ii. `valid_trial_idx = get_valid_trials(trial_data)`; `frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)`; `neural = dff[:, frame_mask].astype(np.float32)`

iii. The agent says this follows the experiment’s built-in trial definitions and retains the complete pre- and post-change trial.

## 1-e. How are trials filtered based on quality controls?

i. It retains Go or Catch trials and excludes aborted and auto-rewarded trials. Experiments with fewer than two valid/processed trials, trials with fewer than two frames, experiments with no cells, and experiments lacking stimulus data are skipped.

ii. `valid = (go | catch) & ~aborted & ~auto_rewarded`; `if n_trial_frames < 2: continue`; `if len(neural_trials) < 2: return None`

iii. The notes tie the four trial flags directly to the instructions and describe the two-trial minimum as required by decoder validation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is the NWB precomputed dF/F trace matrix at `processing/ophys/dff/traces/data`, transposed to neuron by time.

ii. `dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]`; `data['dff_traces'] = dff_raw.T`

iii. The notes choose dF/F rather than deconvolved events because it is the standard precomputed calcium signal.

## 2-b. How is the `neural` data processed?

i. Apart from transposition, float32 conversion, and trial slicing, no normalization, filtering, resampling, or merging of planes is performed.

ii. `neural = dff[:, frame_mask].astype(np.float32)`

iii. The agent states that dF/F is already motion/neuropil/baseline processed in the NWB and therefore need not be recomputed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It rejects experiments with zero cells but does not explicitly apply `valid_roi`; it assumes NWB dF/F contains valid ROIs.

ii. `if n_cells == 0: ... return None`

iii. The notes acknowledge SDK `exclude_invalid_rois=True`, then claim all ROIs in the downloaded NWBs are valid and that neuron counts match the cell table.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples are aligned on the ophys clock and cut from each trial’s start time through (but excluding) stop time; there is no fixed event-centered window.

ii. `frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)`; `neural = dff[:, frame_mask]`

iii. The notes say all streams are hardware synchronized and that the required temporal basis is the ophys timestamp.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native ophys sampling is retained, with no rebinning. Per-experiment median frame intervals are recorded, but one dataset-level median (32.32 ms in the full output) is stored as `time_bin_size`, even though multiscope experiments are about 11 Hz.

ii. `dt = np.median(np.diff(ophys_ts))`; `median_dt = np.median(all_dts)`; `'time_bin_size': median_dt`

iii. The notes explicitly choose native rates (~31 or ~11 Hz) and say each session is internally consistent.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from stimulus-presentation `start_time`, `stop_time`, and `image_name`, plus ophys timestamps; omitted presentations are treated as gray.

ii. `stim_starts = stim_data['start_time']`; `stim_stops = stim_data['stop_time']`; `stim_names = stim_data['image_name']`

iii. The notes prefer the presentation table because it describes the actual flashed image timing.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global sorted image vocabulary is scanned from all experiments, prefixed with `gray`. Every trial frame defaults to gray and is replaced by an image code only during a non-omitted presentation interval.

ii. `image_names_list = [GRAY_LABEL] + all_image_names`; `trace = np.full(n_frames, gray_idx, dtype=np.int64)`; `trace[frame_mask] = img_idx`

iii. The notes explain that image identity is requested for non-gray presentations and validate an expected roughly two-thirds gray fraction.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Presentation intervals are evaluated at the exact `trial_ts` selected from the same ophys timestamps used for neural slicing.

ii. `trial_ts = ophys_ts[trial_mask]`; `frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)`

iii. The agent reports spot-checking the resulting trace against raw NWB stimulus onsets.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It uses stimulus-presentation `is_change` and `start_time`, evaluated on trial ophys timestamps.

ii. `stim_starts = stim_data['start_time']`; `is_change = stim_data['is_change']`

iii. The notes describe `is_change` as the direct event source.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero trace is created; each true change presentation within the trial sets exactly the first ophys frame at or after onset to one.

ii. `frame_idx = np.searchsorted(trial_ts, s_start)`; `trace[frame_idx] = 1`

iii. The justification is that the requested output says “right after” a change, interpreted as an onset impulse.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numeric threshold is applied: the NWB boolean `is_change` directly produces categories 0 (`no_change`) and 1 (`change`).

ii. `if not is_change[si]: continue`; `output_values[1] = ['no_change', 'change']`

iii. The agent treats the raw boolean annotation as authoritative.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change onset is mapped by `searchsorted` to the first trial ophys timestamp at or after the presentation start, on the same time vector as neural data.

ii. `frame_idx = np.searchsorted(trial_ts, s_start)`

iii. The notes report checking that change impulses coincide with image transitions.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses NWB `processing/running/speed/data` and its timestamps.

ii. `data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]`; `data['running_speed'] = f['processing']['running']['speed']['data'][:]`

iii. The notes identify this as the SDK-standard running stream sampled near 60 Hz.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Speed is linearly interpolated over the full ophys timebase. Five percentile edges are then computed separately for each experiment from all session timepoints, including time outside retained trials.

ii. `running_at_ophys = interpolate_to_ophys(..., ophys_ts)`; `running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)`

iii. The agent says interpolation preserves alignment and session-wide percentiles yield five equal-frequency levels.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Experiment-specific 0,20,40,60,80,100 percentile edges define bins 0–4. NaNs are assigned bin 0.

ii. `edges = np.percentile(valid, percentiles)`; `result[valid] = np.clip(np.digitize(values[valid], edges[1:-1]), 0, n_bins - 1)`; `result[~valid] = 0`

iii. The notes interpret “five equal percentile bins” per session and document NaN-to-lowest-bin as a design choice.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is interpolated to every ophys timestamp first, then sliced with the identical trial frame mask as neural activity.

ii. `running_trial = running_at_ophys[frame_mask]`

iii. Hardware synchronization and the common ophys timebase are cited as ensuring alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses pupil ellipse area and timestamps from `acquisition/EyeTracking/pupil_tracking`, along with `likely_blink`.

ii. `data['pupil_area'] = pt['area']['data'][:]`; `data['pupil_timestamps'] = pt['timestamps'][:]`; `data['likely_blink'] = et['likely_blink']['data'][:]`

iii. The notes say area is available in the local NWBs and can be converted to an equivalent circular diameter.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Blink samples become NaN; positive area is converted by `2*sqrt(area/pi)`, then linearly interpolated directly (with internal NaNs still present) to ophys timestamps. Experiment-wide percentile edges are computed afterward.

ii. `pupil_area[likely_blink] = np.nan`; `pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)`; `pupil_at_ophys = interpolate_to_ophys(...)`

iii. The notes cite the whitepaper’s ellipse/area processing and blink exclusion.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Non-NaN experiment-wide diameter values define five percentile bins; values in retained trials are digitized to 0–4 and NaNs become 0.

ii. `pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)`; `pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)`

iii. The justification mirrors running speed: balanced categorical targets, with missing values forced into the lowest class.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Converted pupil diameter is interpolated to the full ophys timestamp vector and then sliced by the same trial frame mask as neural activity.

ii. `pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)`; `pupil_trial = pupil_at_ophys[frame_mask]`

iii. The agent relies on hardware-synchronized clocks and reports raw-data spot checks.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It is derived in priority order from trial booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. `if trial_data['hit'][idx]: return 'hit'`; `elif trial_data['miss'][idx]: return 'miss'`; `elif trial_data['false_alarm'][idx]: return 'false_alarm'`

iii. The notes identify these as the canonical mutually exclusive outcomes for valid trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome name is mapped to its position in a fixed four-class list and broadcast across all frames of the trial; an unknown outcome falls back to code 0.

ii. `outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0`; `output_full[4] = outcome_idx`

iii. Broadcasting was chosen because the output container is a single output-by-time array, despite the target being static per trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing NWBs, unreadable image scans, no-cell experiments, missing stimulus tables, too-few-trial experiments, and too-short trials are skipped with warnings. Missing pupil streams become all-NaN; interpolation extrapolation and blink-related NaNs ultimately map to category 0. Unknown outcomes also map to 0.

ii. `if not os.path.exists(nwb_path): return None`; `except Exception as e: print(...)`; `pupil_at_ophys = np.full(len(ophys_ts), np.nan)`; `result[~valid] = 0`

iii. The notes characterize these choices as robust edge-case handling and explicitly acknowledge that missing pupil samples share the lowest bin.

## 9-a. What are the most time-consuming steps of the code?

i. Reading large dF/F arrays from 202 NWBs dominates per-experiment time; conversion also makes an extra scan through every NWB to collect image names. Plotting, when requested, adds work.

ii. `dff_raw = ...['data'][:]`; `all_image_names = get_all_image_names(exp_table)`; `result = process_experiment(...)`

iii. The notes measure about 1.7 seconds loading versus 0.4 seconds processing per experiment and estimate roughly 12.5 minutes total.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The loop over every stimulus presentation for every trial, repeated image-list `.index` lookups, the loop over trial rows, and the preliminary per-file image scan could be reduced with indexed/vectorized interval mapping and precomputed dictionaries.

ii. `for si in range(len(stim_starts)):`; `img_idx = image_names_list.index(name)`; `for trial_idx in valid_trial_idx:`

iii. The agent provides no explicit vectorization rationale; it emphasizes that file I/O is the measured bottleneck.

## 9-c. What processing does the code repeat multiple times?

i. Each NWB is opened once in `get_all_image_names` and again for full processing. Within an experiment, trial masks and overlapping stimulus scans are recomputed separately for image identity and change, and percentile binning is separately repeated per plane even when planes share one behavioral session.

ii. `all_image_names = get_all_image_names(exp_table)` followed by `nwb_data = load_nwb_data(nwb_path)`; separate calls to `build_image_identity_trace(...)` and `build_image_change_trace(...)`

iii. The notes mention the image scan’s 14-second overhead but do not flag the other repetition.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `load_nwb_data` reads cell ROI IDs that are never used. It computes unused `output_tv` and `output_static` arrays before rebuilding `output_full`; `discretize_percentile` is dead code. Timing/metadata and optional plotting state do not affect decoding.

ii. `data['cell_roi_ids'] = seg[key]['id'][:]`; `output_tv = np.stack(...)`; `output_static = np.array(...)`; `output_full = np.zeros(...)`

iii. No justification is supplied; these appear to be development remnants, while optional plotting was retained for validation.
