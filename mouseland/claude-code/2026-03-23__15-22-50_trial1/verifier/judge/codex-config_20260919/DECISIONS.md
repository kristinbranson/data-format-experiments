# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the master behavior index, deduplicates physical recordings, then loads each session's behavior dictionary, per-plane `spks`, and retinotopy `iarea`. Full mode processes all 89 unique sessions.

ii. `exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()`; `spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0)`; `return dtrans['iarea']`

iii. The notes say this follows the reference loaders, gives 89 recordings from 19 mice, and treats repeated experiment-index entries as the same physical recording.

## 1-b. How are the data split into subjects?

i. Subjects are taken from `mname`; the final sorted unique names are indexed once per session.

ii. `all_subjects = sorted(set(r['subject'] for r in session_results))`; `subject_idx.append(subject_to_idx[result['subject']])`

iii. The notes identify 19 subjects directly from the experiment index.

## 1-c. How are the data split into sessions?

i. A session key is mouse, date, and block. Duplicate appearances under experiment types are retained only on their first occurrence.

ii. `key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"`; `if key not in session_map: session_map[key] = (exp_type, ndb)`

iii. The agent states that duplicate index entries are alternate experiment/stimulus mappings of the same physical recording.

## 1-d. How are the data split into trials?

i. For each behavior trial, rounded `StartFr` and `GrayFr` define a contiguous half-open slice from corridor entry to grey-space entry.

ii. `start_frs = np.round(beh['StartFr']).astype(int)`; `gray_frs = np.round(beh['GrayFr']).astype(int)`; `trial_neural = spk[:, sfr:gfr].astype(np.float16)`

iii. The notes justify this as capturing the full 4 m textured corridor and aligning trials to corridor entry.

## 1-e. How are trials filtered based on quality controls?

i. A trial is skipped only when its rounded bounds are invalid/outside available frames or it contains fewer than two frames. No long-trial or movement filter is applied.

ii. `if sfr < 0 or gfr > n_frames or sfr >= gfr: ... continue`; `if n_trial_frames < 2: ... continue`

iii. The notes explicitly choose “all trials included,” regard very long stopped trials as valid, and report one invalid trial skipped.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from the per-plane `spks` arrays; `iarea` supplies the parallel neuron region labels.

ii. `spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0)`; `iarea = load_retino(...)`

iii. The agent identified `spks` as already-deconvolved fluorescence and `iarea` as retinotopic area assignment.

## 2-b. How is the `neural` data processed?

i. Planes are concatenated, trial columns are sliced, and values are cast to float16; there is no fluorescence normalization, deconvolution, temporal interpolation, or padding.

ii. `trial_neural = spk[:, sfr:gfr].astype(np.float16)`

iii. The notes say the files already contain Suite2p-deconvolved traces and float16 reduces the very large output size.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron is removed. `iarea` values outside the four named visual regions are retained under an `other` category.

ii. `BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'other']`; `region_idx = np.full(len(iarea), 4, dtype=int)`

iii. The agent interpreted the reference as using all Suite2p-detected neurons with no post hoc curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each variable-length trial begins at rounded corridor-entry `StartFr` and ends before rounded `GrayFr`; no common-length padding is used.

ii. `sfr = start_frs[trial_idx]`; `gfr = gray_frs[trial_idx]`; `spk[:, sfr:gfr]`

iii. The notes identify corridor entry as the required alignment event and preserve the full textured-corridor traversal.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One imaging frame is one bin, with no rebinning. Per-session median timestamp spacing is computed and the dataset metadata uses its cross-session median (about 314.7 ms).

ii. `dt_sec = np.nanmedian(np.diff(ft)) * 24 * 3600`; `'time_bin_size': float(median_dt * 1000)`

iii. The notes report the native approximately 3.17 Hz acquisition and no need for temporal resampling.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundFr`, trial frame indices, and the session median interval calculated from `ft` timestamps.

ii. `sound_frs = beh['SoundFr']`; `dt_sec = np.nanmedian(np.diff(ft)) * 24 * 3600`

iii. The notes describe the mapping as `(SoundFr - frame_idx) * dt_sec`.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The difference between fractional sound frame and each integer frame is multiplied by the median seconds per frame; values are positive before and negative after the cue.

ii. `time_to_sound = (sound_frs[trial_idx] - frame_indices) * dt_sec`

iii. This sign convention is documented as literal “time to” the cue.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated for the exact integer range `sfr:gfr`, giving one value for every neural column.

ii. `frame_indices = np.arange(sfr, gfr, dtype=np.float64)`; `trial_neural = spk[:, sfr:gfr]`

iii. The agent says all trial streams use the same frame window.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from `mname` and the calendar date string `datexp` in the master session index.

ii. `date = datetime.strptime(datexp, '%Y_%m_%d')`; `mouse_sessions[mname].append((sess_key, date))`

iii. The notes call it days since each mouse's first recording.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Dates are sorted per mouse and the elapsed number of calendar days from the first recording is broadcast over every frame of the trial.

ii. `day_map[sess_key] = (date - first_date).days`; `np.full(n_trial_frames, day_val, dtype=np.float32)`

iii. The agent intentionally defines training day as calendar-day offset and reports a 0–92 range.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It uses rounded `StartFr`, each frame index, and the median interval derived from `ft`.

ii. `start_frs = np.round(beh['StartFr']).astype(int)`; `dt_sec = np.nanmedian(np.diff(ft)) * 24 * 3600`

iii. The notes specify `(frame_idx - StartFr) * dt_sec`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The zero-based offset from the rounded start frame is multiplied by seconds per frame, so every retained trial begins exactly at zero.

ii. `time_since_start = (frame_indices - sfr) * dt_sec`

iii. The agent treats frame rounding and a constant per-session interval as sufficient temporal precision.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It uses the same `sfr:gfr` frame indices as the neural slice.

ii. `frame_indices = np.arange(sfr, gfr, dtype=np.float64)`; `spk[:, sfr:gfr]`

iii. The notes state frame-number alignment is shared by all streams.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes directly from the trial-level `isRew` flag.

ii. `is_rew = beh['isRew']`; `rew_val = np.float32(1.0 if is_rew[trial_idx] else 0.0)`

iii. The notes map rewarded corridor to 1 and unrewarded corridor to 0.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean is converted to 0/1 float32 and broadcast across the trial.

ii. `np.full(n_trial_frames, rew_val, dtype=np.float32)`

iii. Broadcasting supplies the required `(4, T)` input representation.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived directly from each trial's `WallName`.

ii. `wall_names = beh['WallName']`; `stim_name = wall_names[trial_idx]`

iii. The agent chose `WallName` rather than masked/experiment-specific stimulus IDs.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Every distinct wall-name string across processed sessions is sorted and assigned its own category index (15 full-data categories), then broadcast across trial frames.

ii. `stim_names = sorted(all_stim_names)`; `out_data[0, :] = stim_to_idx[stim_name]`

iii. The notes explicitly preserve variants such as `circle1`, `leaf1`, and swaps rather than collapsing them to four base textures.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from session-level `LickFr`; `LickTrind` is read but not used.

ii. `lick_frs = beh['LickFr']`; `lick_trinds = beh['LickTrind']`

iii. The agent notes that frame-range selection is equivalent to using the trial index for this purpose.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frames are rounded to nearest integers. A retained bin is 1 if one or more rounded licks land in it, otherwise 0.

ii. `lick_frs_int = np.round(lick_frs).astype(int)`; `lick_binary[frame_offsets] = 1.0`

iii. The notes justify a binary “any lick in frame” time series.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Only rounded lick frames in `[sfr, gfr)` are selected and converted to offsets into the same-length neural trial.

ii. `mask = (lick_frs_int >= start_fr) & (lick_frs_int < end_fr)`; `frame_offsets = lick_frs_int[mask] - start_fr`

iii. The agent relies on `LickFr` already indexing imaging frames.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It comes from frame-level `ft_Pos` values in decimeters.

ii. `ft_pos = beh['ft_Pos'][:n_frames]`; `trial_pos = ft_pos[sfr:gfr]`

iii. The notes identify 0–40 dm as the 4 m textured corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Positions are digitized against five evenly spaced edges from 0 to 40 dm, shifted to zero-based category IDs, and clipped to 0–3.

ii. `bin_edges = np.linspace(0, CORRIDOR_LENGTH_DM, n_bins + 1)`; `binned = np.clip(np.digitize(pos, bin_edges) - 1, 0, n_bins - 1)`

iii. This implements the requested four equal 1 m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are 0, 10, 20, 30, and 40 dm, producing categories 0–1 m, 1–2 m, 2–3 m, and 3–4 m; out-of-range values are clipped.

ii. `bin_edges = np.linspace(0, 40, 5)`; `np.digitize(pos, bin_edges) - 1`

iii. The notes state these are four equal 10 dm bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is sliced with the same `sfr:gfr` boundaries as neural data before binning.

ii. `trial_pos = ft_pos[sfr:gfr]`; `trial_neural = spk[:, sfr:gfr]`

iii. Both streams have one sample per imaging frame.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from frame-level `ft_RunSpeed` values at frames marked by `ft_CorrSpc`.

ii. `speeds = beh['ft_RunSpeed'][:n_frames]`; `all_speeds.append(speeds[corr_mask])`

iii. The agent uses direct measured speed and the corridor mask for threshold estimation.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Three percentile thresholds are calculated once over all corridor speeds in the selected dataset; every trial's speeds are digitized using those global thresholds.

ii. `quartiles = np.percentile(valid, [25, 50, 75])`; `speed_binned = discretize_speed(trial_speed, speed_quartiles)`

iii. The notes say dataset-wide thresholds create consistent speed categories across sessions.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize(..., right=True)` applies the global 25th, 50th, and 75th percentile values and clips results to 0–3. Ties, especially zero speed, mean realized bins are not exactly 25% each.

ii. `binned = np.digitize(speed, quartiles, right=True)`; `binned = np.clip(binned, 0, 3)`

iii. The agent changed to `right=True` after observing a large zero-speed tie and reports a 30.2/19.8/24.9/25.1% distribution.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Frame-level speed is sliced with the exact `sfr:gfr` neural window and then categorized.

ii. `trial_speed = ft_run_speed[sfr:gfr]`; `trial_neural = spk[:, sfr:gfr]`

iii. Speed and neural data share the imaging-frame grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Behavior and neural streams are truncated to their shorter frame count; malformed trial ranges are skipped; empty lick lists are accepted; nonfinite speeds are excluded from percentile estimation; and neuron/retinotopy mismatch raises an assertion.

ii. `n_frames = min(n_frames_neural, n_frames_beh)`; `if sfr < 0 or gfr > n_frames or sfr >= gfr: continue`; `valid = all_speeds[np.isfinite(all_speeds)]`

iii. The notes describe a clean dataset, report one skipped invalid trial, and use assertions/sanity checks to expose structural mismatch.

## 12-a. What are the most time-consuming steps of the code?

i. Loading/concatenating huge spike files, copying them into trial arrays, retaining the 148 GB result, and pickling it dominate. The full run took about 33 minutes.

ii. `spk = np.concatenate(...)`; `trial_neural = spk[:, sfr:gfr].astype(np.float16)`; `pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)`

iii. The notes estimate roughly 15 seconds per session and several minutes for the final save.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial construction and the later nested stimulus-fill pass could be partly vectorized or combined, though variable trial lengths limit full vectorization. Region mapping also uses repeated boolean assignments.

ii. `for trial_idx in range(ntrials):`; `for result in session_results: for i, (out_data, stim_name) in enumerate(result['output']):`

iii. The agent provides no explicit vectorization justification; its notes emphasize I/O and memory as the practical bottlenecks.

## 12-c. What processing does the code repeat multiple times?

i. It calls `get_unique_sessions()` twice, loads every behavior file once while collecting global speeds and again while processing sessions, and traverses outputs a second time to replace stimulus placeholders.

ii. `session_map = get_unique_sessions()`; `all_session_map = get_unique_sessions()`; `beh = load_behavior_for_session(...)` appears in both collection and session processing.

iii. The repeat passes support full-map day calculation, global speed thresholds, and a mapping that is only known after collecting every stimulus name.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In normal full conversion little computed data is gratuitous, but `LickTrind` is assigned and unused, timing variables are created only for logging, and global raw speeds are discarded after percentile calculation. Optional processing plots add work only when requested.

ii. `lick_trinds = beh['LickTrind']`; `t0 = time.time()`; `all_speeds = np.concatenate(all_speeds)`

iii. The notes justify speed collection for discretization and plotting for visual validation; they do not justify the unused `LickTrind` assignment.
