# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Reads the local metadata CSV, inventories locally downloaded NWB files, keeps downloaded non-passive experiments, then opens every selected NWB directly with `h5py`. It does not use the Allen SDK cache, does not filter `project_code == 'VisualBehavior'`, and therefore processes both single-plane and downloaded multiscope experiments.

ii.
```python
`exp_table = pd.read_csv(os.path.join(METADATA_DIR, 'ophys_experiment_table.csv'))`
`exp_table = exp_table[exp_table['ophys_experiment_id'].isin(downloaded_ids)]`
`exp_table = exp_table[~exp_table['session_type'].str.contains('passive', case=False)]`
`with h5py.File(nwb_path, 'r') as f:`
```

iii. The notes call direct HDF5 loading faster than AllenSDK, say only the downloaded 38-mouse subset is available, and deliberately exclude passive sessions. They report 202 active experiments.

## 1-b. How are the data split into subjects?

i. Uses the experiment-table `mouse_id`, converted to a string, and builds a first-seen global subject map; every retained experiment receives that mouse’s index.

ii.
```python
`mouse_id = str(exp_row['mouse_id'])`
`if mouse_id not in subject_map:`
`    subject_map[mouse_id] = len(all_subjects)`
```

iii. The notes identify `mouse_id` as the subject identifier and cross-check 38 unique downloaded mice.

## 1-c. How are the data split into sessions?

i. Treats each `ophys_experiment_id` (one imaging plane) as an independent output session. It records `ophys_session_id` only as metadata and never groups experiments sharing that ID.

ii.
```python
`for idx, (_, row) in enumerate(exp_table.iterrows()):`
`    exp_id = row['ophys_experiment_id']`
`    result = process_experiment(exp_id, row, ...)`
`    all_neural.append(result['neural'])`
```

iii. The planning notes explicitly choose each experiment/plane as a separate session for multiscope recordings because planes have different neurons but shared behavior.

## 1-d. How are the data split into trials?

i. Uses the NWB `intervals/trials` table and slices each selected experiment over `[start_time, stop_time)` using an ophys timestamp mask, producing variable-length trials.

ii.
```python
`t_start = trial_data['start_time'][trial_idx]`
`t_stop = trial_data['stop_time'][trial_idx]`
`frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)`
```

iii. The notes say this follows the experiment’s built-in trial definition and keeps the complete behavioral trial.

## 1-e. How are trials filtered based on quality controls?

i. Keeps only `(go OR catch) AND NOT aborted AND NOT auto_rewarded`; skips experiments with fewer than two selected trials, trials with fewer than two ophys frames, and experiments left with fewer than two processed trials.

ii.
```python
`valid = (go | catch) & ~aborted & ~auto_rewarded`
`if len(valid_trial_idx) < 2: return None`
`if n_trial_frames < 2: continue`
```

iii. The filtering is justified directly by the decoder instructions. The two-trial rule is required by validation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Uses the precomputed dF/F array at `processing/ophys/dff/traces/data`, transposed from frame-by-cell to cell-by-frame.

ii.
```python
`dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]`
`data['dff_traces'] = dff_raw.T`
```

iii. The notes prefer dF/F over deconvolved events because it is a standard calcium-imaging signal already produced by the Allen pipeline.

## 2-b. How is the `neural` data processed?

i. Per experiment, transposes the stored dF/F matrix, casts each trial slice to float32, and applies no normalization, filtering, or plane merging.

ii.
```python
`dff = nwb_data['dff_traces']`
`neural = dff[:, frame_mask].astype(np.float32)`
```

iii. The agent says the NWB dF/F is already motion-corrected, neuropil-corrected, baselined, and detrended, so further processing is unnecessary.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only rejects an experiment when `n_cells == 0`; it performs no explicit `valid_roi` lookup or cell-level QC.

ii.
```python
`if n_cells == 0:`
`    return None`
```

iii. The notes assert all ROIs stored in the downloaded NWB dF/F dataset are valid and cite the SDK’s default `exclude_invalid_rois=True`, although the script bypasses that SDK path.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Uses ophys timestamps as the master clock and takes frames at or after trial start and strictly before trial stop. Thus time zero is trial start, not image-change onset.

ii.
```python
`frame_mask = (ophys_ts >= t_start) & (ophys_ts < t_stop)`
`neural = dff[:, frame_mask]`
```

iii. The notes state that the full trial preserves pre-change flashes and post-change behavior and fulfills alignment to the ophys clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Does no temporal rebinning. Each experiment remains at native ophys cadence (~32.3 ms Scientifica or ~90 ms multiscope), while one dataset-level `time_bin_size` is reported as the median session dt.

ii.
```python
`dt = np.median(np.diff(ophys_ts))`
`median_dt = np.median(all_dts)`
`'time_bin_size': median_dt`
```

iii. The notes explicitly retain native rates and acknowledge both ~31 Hz and ~11 Hz recordings, claiming consistency within a session.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Derives it from stimulus-presentation `start_time`, `stop_time`, and `image_name`, not trial `initial_image_name`/`change_image_name`; omitted and inter-stimulus periods become a synthetic `gray` class.

ii.
```python
`stim_starts = stim_data['start_time']`
`stim_stops = stim_data['stop_time']`
`stim_names = stim_data['image_name']`
`trace = np.full(n_frames, gray_idx)`
```

iii. The notes argue that the requested identity is the image actually on the non-gray screen and validate the expected ~66.7% gray fraction.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Scans all files for unique non-omitted names, prepends `gray`, assigns category indices, initializes every trial frame to gray, and overwrites frames falling inside each non-omitted presentation.

ii.
```python
`image_names_list = [GRAY_LABEL] + all_image_names`
`frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)`
`trace[frame_mask] = img_idx`
```

iii. The global list gives stable categories across experiments; omitted flashes are treated as gray.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Builds identity on exactly the same ophys timestamps and `[trial_start, trial_stop)` mask as neural activity; presentation intervals determine per-frame labels.

ii.
```python
`trial_ts = ophys_ts[trial_mask]`
`frame_mask = (trial_ts >= s_start) & (trial_ts < s_stop)`
```

iii. The agent reports spot-checking identity traces against raw NWB stimulus onsets.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Uses stimulus-presentation `is_change` and `start_time` fields. It does not use trial `go` and `change_time` directly.

ii.
```python
`stim_starts = stim_data['start_time']`
`is_change = stim_data['is_change']`
```

iii. The notes describe `is_change` as the authoritative presentation-level change flag.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Starts with zeros and places a one only at the first ophys frame at or after every flagged change presentation within the trial.

ii.
```python
`frame_idx = np.searchsorted(trial_ts, s_start)`
`trace[frame_idx] = 1`
```

iii. The agent intended a transient onset event and cites the instruction’s wording, “right after a change.”

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary: category 1 iff a stimulus presentation has truthy `is_change` and the frame is the first ophys sample after its onset; all other frames are 0.

ii.
```python
`if not is_change[si]: continue`
`trace = np.zeros(n_frames, dtype=np.int64)`
```

iii. The categories are documented as `['no_change', 'change']`; no numeric threshold beyond the boolean NWB flag is applied.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Uses the same trial ophys timestamps as neural activity and `searchsorted` to select the first frame at/after onset.

ii.
```python
`trial_ts = ophys_ts[trial_mask]`
`frame_idx = np.searchsorted(trial_ts, s_start)`
```

iii. The notes say hardware synchronization plus the shared ophys timebase provides alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Reads running `speed/data` and its `timestamps` from the NWB processing group.

ii.
```python
`data['running_timestamps'] = f['processing']['running']['speed']['timestamps'][:]`
`data['running_speed'] = f['processing']['running']['speed']['data'][:]`
```

iii. The notes identify this as the standard wheel-derived running stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linearly interpolates the entire running stream onto ophys timestamps, computes quintile edges from all non-NaN ophys-aligned frames of that experiment, then bins trial samples; it does not apply the paper’s noted 10-Hz low-pass itself.

ii.
```python
`running_at_ophys = interpolate_to_ophys(...)`
`running_edges = compute_session_percentile_edges(running_at_ophys, n_bins=5)`
```

iii. The notes justify interpolation for clock alignment and session-wide quintiles for balanced labels.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Uses per-experiment 0/20/40/60/80/100 percentiles, digitizes against the four inner edges, clips to 0–4, and maps NaN to bin 0.

ii.
```python
`edges = np.percentile(valid, percentiles)`
`result[valid] = np.clip(np.digitize(values[valid], edges[1:-1]), 0, n_bins - 1)`
```

iii. The agent calls these five equal percentile bins and treats bin 0 as a conservative missing-data fallback.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Interpolates onto full-session `ophys_ts` first, then applies the identical trial frame mask used for dF/F.

ii.
```python
`running_at_ophys = interpolate_to_ophys(..., ophys_ts)`
`running_trial = running_at_ophys[frame_mask]`
```

iii. The justification is hardware-synchronized timestamps and a common ophys sampling grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Reads pupil ellipse `area`, pupil timestamps, and `likely_blink`; derives an equivalent circular diameter from area rather than using the SDK `pupil_width` column.

ii.
```python
`data['pupil_area'] = pt['area']['data'][:]`
`data['likely_blink'] = et['likely_blink']['data'][:]`
```

iii. The notes cite the geometric conversion `2*sqrt(area/pi)` and the paper’s blink flags.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Sets blink-area samples to NaN, converts positive area to equivalent diameter, linearly interpolates that array to ophys timestamps, computes per-experiment quintile edges, and bins trials. Because NaNs are passed into `interp1d` rather than removed, gaps may propagate.

ii.
```python
`pupil_area[likely_blink] = np.nan`
`pupil_diameter[valid_pupil] = 2.0 * np.sqrt(pupil_area[valid_pupil] / np.pi)`
`pupil_at_ophys = interpolate_to_ophys(...)`
```

iii. The notes say blink removal prevents artifacts and that diameter follows the whitepaper.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Uses per-experiment quintiles of non-NaN ophys-aligned diameter, digitizes to categories 0–4, and assigns missing samples to 0.

ii.
```python
`pupil_edges = compute_session_percentile_edges(pupil_at_ophys, n_bins=5)`
`pupil_binned = apply_percentile_bins(pupil_trial, pupil_edges, n_bins=5)`
```

iii. The agent chose equal percentile bins to balance classes and explicitly documents NaN-to-zero.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolates pupil diameter to the full ophys clock and extracts it with the same trial frame mask as dF/F.

ii.
```python
`pupil_at_ophys = interpolate_to_ophys(pupil_diameter, pupil_ts, ophys_ts)`
`pupil_trial = pupil_at_ophys[frame_mask]`
```

iii. The notes rely on hardware synchronization and report a raw-NWB spot check.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Reads the four boolean trial columns `hit`, `miss`, `false_alarm`, and `correct_reject` in priority order.

ii.
```python
`if trial_data['hit'][idx]: return 'hit'`
`elif trial_data['miss'][idx]: return 'miss'`
```

iii. The notes call these the canonical mutually exclusive outcomes for retained go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Maps the outcome string into the fixed four-category list and broadcasts its integer index over all timepoints in that trial. Unknown outcomes fall back to category 0 (`hit`).

ii.
```python
`outcome_idx = outcome_names.index(outcome) if outcome in outcome_names else 0`
`output_full[4] = outcome_idx`
```

iii. Broadcasting was chosen because one array must contain both time-varying outputs and a static trial label.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Skips missing NWBs, unreadable files during image-name scanning, empty-cell experiments, missing stimulus tables, too-short trials/sessions, and missing pupil streams. Interpolation out-of-range values and missing pupil become NaN, later bin 0. It does not wrap full experiment processing in a general exception handler.

ii.
```python
`if not os.path.exists(nwb_path): return None`
`if nwb_data['pupil_area'] is None: pupil_at_ophys = np.full(len(ophys_ts), np.nan)`
`result[~valid] = 0`
```

iii. The notes describe NaN-to-zero as a design choice and say edge cases passed validation.

## 9-a. What are the most time-consuming steps of the code?

i. Large NWB reads, especially full-session dF/F, dominate; the full run also stores large per-trial copies and serializes an ~8.5-GB pickle. The separate all-file image scan adds about 14 seconds.

ii.
```python
`dff_raw = f['processing']['ophys']['dff']['traces']['data'][:]`
`pickle.dump(data, f, protocol=4)`
```

iii. The timing notes estimate ~1.7 s load and ~0.4 s processing per experiment, ~12.5 minutes total.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The loops over every stimulus presentation in both identity and change builders repeatedly test overlap; identity lookup uses linear `list.index`. Trial loops and repeated boolean masks could also be batched or bounded with `searchsorted`.

ii.
```python
`for si in range(len(stim_starts)):`
`    ...
`    img_idx = image_names_list.index(name)`
```

iii. The agent prioritized readable code and reported I/O as the larger bottleneck; it did not document vectorization opportunities explicitly.

## 9-c. What processing does the code repeat multiple times?

i. Each NWB is opened once to collect image names and again for full processing. Within every trial, both output builders reconstruct the same trial mask/timestamps and rescan the full presentation table; main also constructs that trial mask separately.

ii.
```python
`all_image_names = get_all_image_names(exp_table)`
`nwb_data = load_nwb_data(nwb_path)`
`trial_mask = (ophys_ts >= trial_start) & (ophys_ts < trial_stop)`
```

iii. The notes acknowledge the image-name scan’s overhead but accept it to establish global category names before conversion.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes unused `output_tv`, `output_static`, `n_frames`, and includes an unused `discretize_percentile` helper. In normal mode it also loads cell ROI IDs that are never consumed. Plot-only diagnostic data are retained only when requested.

ii.
```python
`output_tv = np.stack([...])`
`output_static = np.array([outcome_idx], dtype=np.int64)`
`# output_full is then rebuilt separately`
```

iii. The trajectory comments show uncertainty about mixed static/time-varying output layout; the intermediate arrays remained after choosing a broadcast `(5,T)` representation.


