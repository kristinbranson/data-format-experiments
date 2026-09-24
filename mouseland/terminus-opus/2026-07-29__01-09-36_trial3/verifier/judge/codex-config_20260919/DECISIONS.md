# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads `Imaging_Exp_info.npy` as a master index, deduplicates session IDs, loads each session's plane-wise `spks`, retinotopy `iarea`, and searches its possible `Beh_<type>.npy` files/keys. Full mode processes all 89 unique sessions.

ii. `exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()`; `spk = np.concatenate(spk_data['spks'], axis=0)`; `beh = load_beh(sid, meta['beh_files'], meta['beh_keys'])`.

iii. The notes say this reproduces the reference loaders, handles stimulus-suffix behavior keys, and matches 89 recordings in 19 mice. Behavior files are nevertheless reopened per lookup rather than cached by file.

## 1-b. How are the data split into subjects?

i. Subject is `mname`; unique names are sorted, and each session gets the corresponding integer `subject_idx`.

ii. `subjects_list = sorted(subjects_set)`; `session_subject_map.append(subject_to_idx[meta['mname']])`.

iii. The notes report 19 mouse names, matching the paper.

## 1-c. How are the data split into sessions?

i. A session ID is mouse, date, and block. Duplicate appearances under experiment types are merged into one metadata record.

ii. `sid = f"{db['mname']}_{db['datexp']}_{db['blk']}"`; `if sid not in session_meta: session_meta[sid] = {...}`.

iii. The agent explains that 142 experiment-index entries reduce to 89 unique spike-file sessions.

## 1-d. How are the data split into trials?

i. For each integer `trial_idx` in `ntrials`, every frame whose `ft_trInd` equals that index is used. The loaded `ft_CorrSpc` flag is not applied, so pre-corridor/gray-space frames assigned to the trial are included.

ii. `trial_mask = (ft_trInd == trial_idx)`; `frame_indices = np.where(trial_mask)[0]`.

iii. The notes call this “per-trial via ft_trInd” and say it matches the reference approach; they do not justify omitting `ft_CorrSpc`.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than two indexed frames are skipped, including after the neural-frame bounds check. No stalled/abnormally long trial filter is applied.

ii. `if len(frame_indices) < 2: skipped += 1; continue`.

iii. The notes explicitly choose “Include all trials. Skip trials with < 2 frames,” without examining extreme stopping trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from the plane arrays under `spks` in each session neural file; `iarea` from retinotopy supplies neuron-region selection and labels.

ii. `spk_data = np.load(spk_path, allow_pickle=True).item()`; `spk = np.concatenate(spk_data['spks'], axis=0)`; `return dtrans['iarea']`.

iii. The agent identified these as already deconvolved Suite2p fluorescence traces, so no raw fluorescence conversion is needed.

## 2-b. How is the `neural` data processed?

i. Planes are concatenated, neurons are filtered/subsampled, trial columns are selected, and values are stored as float32. There is no normalization, smoothing, padding, dF/F calculation, or temporal resampling.

ii. `trial_neural = spk[:, frame_indices].astype(np.float32)`.

iii. The notes cite the methods statement that analyses use deconvolved traces and justify the 2,000-neuron cap by the downstream PCA/SVD.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons with `iarea` -1 or 7 are removed. If more than 2,000 remain, a fixed-seed, approximately region-proportional random sample is retained; the result can be 1,999 because proportional counts are rounded.

ii. `valid_mask = (iarea != -1) & (iarea != 7)`; `sampled = rng.choice(region_local_idx, size=n_sample, replace=False)`; `MAX_NEURONS = 2000`.

iii. Visual-area filtering is said to match reference code. Subsampling is justified by decoder PCA to 100 components and SVD cost, and made reproducible with seed 42.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent labels corridor entry as alignment and computes time relative to `StartFr`, but stores all `ft_trInd` frames rather than restricting each trial to corridor entry through the textured corridor. Trial arrays remain variable length.

ii. `frame_indices = np.where(ft_trInd == trial_idx)[0]`; `time_since_start = (frame_indices.astype(np.float64) - trial_start) / FS`.

iii. The notes state “Align to corridor entry” and “Use ft_trInd for frame-to-trial mapping,” treating relative time as sufficient alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native imaging frames are retained at 3.17 Hz, recorded as about 315.46 ms per bin; no rebinning is applied.

ii. `FS = 3.17`; `TIME_BIN_MS = 1000.0 / FS`.

iii. The notes identify the native frame rate and explicitly say “No resampling.”

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundFr`, the integer/global frame indices selected by `ft_trInd`, and the constant frame rate. Raw `ft` timestamps are not used.

ii. `trial_sound_fr = sound_fr[trial_idx]`; `time_to_sound = (trial_sound_fr - frame_indices.astype(np.float64)) / FS`.

iii. The mapping plan describes `SoundFr - frame_idx` divided by `FS` as seconds.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The signed frame difference is divided by 3.17, yielding positive values before the cue and negative values afterward; it is cast to float32.

ii. `trial_input[0, :] = time_to_sound.astype(np.float32)`.

iii. The notes validated this formula on one trial but did not account for measured, slightly irregular `ft` timestamps or fractional-event interpolation.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at exactly the same `frame_indices` used to slice neural columns, so its array length and columns align with the agent's neural trial.

ii. `trial_neural = spk[:, frame_indices]`; `time_to_sound = (trial_sound_fr - frame_indices.astype(np.float64)) / FS`.

iii. The sanity check reports matching lengths and a hand-computed example.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It uses the index entry's `days` field when present, otherwise `sess#`, otherwise zero; it is not derived by ordering unique dates per mouse.

ii. `day_key = 'days' if 'days' in db else 'sess#'`; `day_val = db.get(day_key, 0)`.

iii. The agent treats experiment-index metadata as the direct source and reports converted values from 0 to 15.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The selected metadata value is cast to float and broadcast over every frame of the trial.

ii. `day_of_training = float(meta['day_of_training'])`; `trial_input[1, :] = day_of_training`.

iii. The notes call it a direct per-trial value, broadcast in the time-varying input matrix.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It uses per-trial `StartFr`, selected global frame indices, and `FS`; it does not use `ft` timestamps.

ii. `trial_start = start_fr[trial_idx]`; `time_since_start = (frame_indices.astype(np.float64) - trial_start) / FS`.

iii. The mapping plan documents `frame_idx - StartFr` divided by frame rate.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The signed frame difference is converted to nominal seconds and float32. Frames before `StartFr` can consequently be negative.

ii. `trial_input[2, :] = time_since_start.astype(np.float32)`.

iii. The notes validated the arithmetic on one trial and describe the result as time varying in seconds.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It uses the same `frame_indices` as the neural slice, giving column-wise alignment within the agent's broader trial window.

ii. `trial_neural = spk[:, frame_indices]`; `time_since_start = (frame_indices.astype(np.float64) - trial_start) / FS`.

iii. The agent cites frame-trial mapping and a numerical sanity check.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes directly from the per-trial `isRew` array.

ii. `is_rew = beh['isRew']`; `trial_input[3, :] = float(is_rew[trial_idx])`.

iii. The notes describe a Boolean-to-0/1 direct mapping.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The Boolean/numeric value is converted to float and broadcast over all trial frames.

ii. `trial_input[3, :] = float(is_rew[trial_idx])`.

iii. A sanity check confirms a rewarded trial becomes 1.0.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It starts from `WallName` and `UniqWalls`; if a behavioral `stim_id` array exists, IDs are mapped through a seven-name table, otherwise wall names are used. A dataset-wide sorted set of resulting names defines category IDs.

ii. `stim_name = wall_to_stim.get(wall_name[trial_idx], wall_name[trial_idx])`; `sorted_stim = sorted(all_stim_names)`.

iii. The notes say stimulus IDs are used when available and wall names otherwise, including special handling for swap sessions.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each distinct derived wall/crop/swap name is encoded separately and broadcast through its trial. Full conversion therefore has 12 categories rather than the four base textures.

ii. `stim_idx = stim_to_idx[stim_name]`; `output_arr[0, :] = stim_idx`.

iii. The agent reports 12 categories as a successful consistency result, overlooking the requested broad category examples and reference's four-texture collapse.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It uses `LickFr` event frames and `LickTrind` trial assignments.

ii. `trial_lick_mask = (lick_trind == trial_idx)`; `trial_lick_frames = lick_fr[trial_lick_mask]`.

iii. The notes identify these fields and describe binary licking per frame.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Every lick frame is rounded to the nearest integer, searched within the trial's frames, and sets that bin to one; multiple licks in one frame remain one.

ii. `lf_int = int(np.round(lf))`; `pos = np.searchsorted(frame_indices, lf_int)`; `lick_binary[pos] = 1`.

iii. The notes report 36 events becoming 24 positive frames and regard this as correct aggregation.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Rounded global lick frames are matched to positions in the exact frame-index array used for neural columns.

ii. `pos = np.searchsorted(frame_indices, lf_int)`; `trial_neural = spk[:, frame_indices]`.

iii. The agent says `ft_*` and lick frame variables are frame aligned.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from frame-aligned `ft_Pos` at the selected trial frames.

ii. `trial_pos = ft_Pos[frame_indices]`.

iii. The notes state the position scale is 10 VR units per meter.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Values are clipped to [0, 39.999], divided by 10, floored, cast to integer, and clipped to category 0–3.

ii. `pos_clipped = np.clip(trial_pos, 0, 39.999)`; `pos_bin = np.floor(pos_clipped / 10.0).astype(np.int64)`.

iii. The agent intended four equal 1 m bins and chose to force gray-space positions into bin 3.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are 0, 10, 20, 30, and 40 VR units, labeled 0–1 m, 1–2 m, 2–3 m, and 3–4 m. Values at/above 40 are clipped into the last category.

ii. `pos_bin = np.clip(pos_bin, 0, 3)`; `pos_bin_names = ['0-1m', '1-2m', '2-3m', '3-4m']`.

iii. This is justified directly by the requested four equal-length bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is indexed by the same `frame_indices` as neural data, though those indices include non-texture portions of each trial.

ii. `trial_pos = ft_Pos[frame_indices]`; `trial_neural = spk[:, frame_indices]`.

iii. The notes describe behavioral `ft_*` variables as already frame aligned.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed`.

ii. `speeds = beh['ft_RunSpeed']`; `trial_speed = ft_RunSpeed[frame_indices]`.

iii. The mapping plan calls this a direct frame-aligned behavioral source.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Before session processing, the agent concatenates every behavior frame from all selected sessions and computes three global value percentiles. Each trial speed is assigned with `np.digitize`.

ii. `all_speeds = np.concatenate(all_speeds)`; `quartiles = np.percentile(all_speeds, [25, 50, 75])`; `speed_bin = np.digitize(trial_speed, speed_quartiles)`.

iii. The notes explicitly choose global quartiles including all frames, aiming for dataset-wide 25% bins.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three global numeric percentile thresholds define Q1–Q4; ties, especially the large mass at zero, are not divided, so the output bins are not equal-sized.

ii. `speed_bin = np.clip(speed_bin, 0, 3)`; `speed_bin_names = ['Q1', 'Q2', 'Q3', 'Q4']`.

iii. The agent assumes ordinary percentile boundaries satisfy “each corresponding to 25% of the data”; it does not discuss ties.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is sliced by exactly the neural frame indices and then categorized.

ii. `trial_speed = ft_RunSpeed[frame_indices]`; `trial_neural = spk[:, frame_indices]`.

iii. The agent regards `ft_RunSpeed` as natively frame aligned.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural/behavior length is limited to their minimum; trial indices are bounds-checked; trials under two frames are skipped; alternate behavior keys are tried; neuron-count mismatch raises an assertion. There is no per-session exception recovery in the full loop.

ii. `n_frames = min(n_frames_spk, n_frames_beh)`; `frame_indices = frame_indices[frame_indices < n_frames_spk]`; `assert len(iarea) == n_neurons_raw`.

iii. The notes cite mismatched frame counts and suffix keys as resolved edge cases and report no verifier warnings.

## 12-a. What are the most time-consuming steps of the code?

i. Loading and concatenating the very large neural files dominates; full conversion took about 18 minutes. Global speed collection also reloads behavior before processing.

ii. `spk_data = np.load(spk_path, allow_pickle=True).item()`; `spk = np.concatenate(spk_data['spks'], axis=0)`.

iii. The notes estimate neural loading at 1–30 seconds per session and trial processing under one second.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The code rescans `ft_trInd` once for every trial, loops over lick events even though frame flags can be assigned vectorially, and performs proportional neuron sampling region by region. Per-trial assembly still requires a loop because arrays are variable length.

ii. `for trial_idx in range(ntrials): trial_mask = (ft_trInd == trial_idx)`; `for lf in trial_lick_frames:`.

iii. The agent provides no efficiency discussion of these loops; its notes only say trial processing is fast.

## 12-c. What processing does the code repeat multiple times?

i. Behavior data are loaded once while collecting global speed values and again during each session; moreover, `load_beh` reloads a shared behavior file for every session lookup. Trial masks repeatedly scan the full frame vector.

ii. `beh = load_beh(sid, ...)` appears in both `compute_running_speed_quartiles` and `process_session`; `beh_all = np.load(beh_path, allow_pickle=True).item()` occurs inside `load_beh`.

iii. The notes do not acknowledge this repetition.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `ft_CorrSpc`, `selected_idx`, `show_processing` in `process_session`, and some metadata fields are computed/loaded but unused there. Raw outputs temporarily carry `-1` stimulus placeholders plus names and are rewritten in a second pass. Optional plotting computes concatenations used only for diagnostic figures.

ii. `ft_CorrSpc = beh['ft_CorrSpc'][:n_frames]`; `spk, iarea_filtered, selected_idx = filter_and_subsample_neurons(...)`; `trial_output[0, :] = -1`.

iii. The notes do not identify discarded work; plotting is intentionally optional diagnostic processing.
