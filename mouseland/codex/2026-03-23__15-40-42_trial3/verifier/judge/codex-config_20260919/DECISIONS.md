# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads `Imaging_Exp_info.npy` as the master index, groups its entries by unique `<mouse>_<date>_<block>` recording ID, chooses one validated canonical behavior entry for each recording, and then loads that behavior, its per-session `spks` file, and retinotopy `iarea`. Full mode processes all 89 unique recordings.

ii. `exp_info = load_exp_info()`; `per_rec[rec_id].append((exp_type, db))`; `beh = load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]`; `spk_obj = np.load(ROOT / "data" / "spk" / f"{sess.rec_id}_neural_data.npy", allow_pickle=True).item()`; `iarea = np.asarray(ret["iarea"])`.

iii. The notes say the index has 142 references but 89 unique imaging recordings, matching the paper, and that duplicate behavior references were validated before one was selected. Behavior-only cohorts without neural files were excluded.

## 1-b. How are the data split into subjects?

i. Subject is the `mname` field. A first-seen unique subject list is built from the sorted sessions, and each session gets an integer `subject_idx`.

ii. `subject=first_db["mname"]`; `subject_to_idx[sess.subject] = len(subjects)`; `"subject_idx": np.array([subject_to_idx[s.subject] for s in sessions], dtype=np.int64)`.

iii. The agent states that the result contains 19 mice and 89 sessions, matching the paper.

## 1-c. How are the data split into sessions?

i. A session is a unique mouse/date/block recording. Duplicate experiment-group references are merged; the canonical behavior entry is the candidate maximizing non-NaN `stim_id`, number of wall names, and key after agreement checks.

ii. `rec_id = f"{db['mname']}_{db['datexp']}_{db['blk']}"`; `_, exp_type, key, _ = max(candidates)`.

iii. The notes justify deduplication by the 89-recording paper count and say duplicated entries agree on conversion-relevant fields; `WallName` is preferred over unreliable `stim_id`.

## 1-d. How are the data split into trials?

i. For each `trial` from `0` to `ntrials-1`, the agent selects frame indices bearing that trial ID and satisfying both corridor and movement masks. Thus a trial consists only of running frames inside the textured corridor and can contain temporal gaps where the animal stopped.

ii. `valid = finite_trial & ft_corr & ft_move`; `frame_idx = np.flatnonzero(valid & (ft_trial_int == trial))`.

iii. The agent cites the paper's running-only analyses and argues that corridor-only frames preserve the requested behavioral alignment while excluding gray space.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level outlier filter is used. Every declared trial must have at least one running corridor frame; otherwise conversion raises an error. All 38,110 trials survived in the full artifact.

ii. `if len(frame_idx) == 0: raise ValueError(f"Trial {trial} has no retained running corridor frames")`; `masks.append(frame_idx)`.

iii. The notes report exact preservation of all 38,110 raw trials. They justify filtering frames by running status, but do not justify retaining extremely long stopped trials as a trial-level choice.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from the per-plane arrays under `spks`, while neuron selection and region labels use retinotopy `iarea`. Behavior fields are additionally used to select task-responsive neurons.

ii. `spk_chunks = list(spk_obj["spks"])`; `spk = np.concatenate(spk_chunks, axis=0)`; `areas = utils.neu_area_ID(iarea)`.

iii. The notes identify `spks` as saved deconvolved fluorescence and `iarea` as the region assignment, so no raw fluorescence conversion is needed.

## 2-b. How is the `neural` data processed?

i. The agent does not deconvolve, normalize, interpolate, or temporally rebin the saved traces. It selects neuron rows, slices the retained frame indices, concatenates selected rows across imaging chunks, and casts each trial to `float16`.

ii. `trial_chunks.append(chunk[local_rows][:, frame_idx])`; `neural_trial = np.concatenate(trial_chunks, axis=0).astype(np.float16, copy=False)`.

iii. The notes say all paper analyses use the already-deconvolved traces; frame alignment is retained for decoder outputs, and float16 controls the output size.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The exported neurons are a union of stimulus-selective mHV cells and reward-prediction aHV cells. mHV selection uses extreme 5% d-prime tails on running odd-parity corridor frames, with a 128-cell fallback; aHV selection uses late-versus-early cue d-prime at least 0.3 and nonnegative stimulus selectivity. Other areas and most cells are discarded.

ii. `mhv_mask = percentile_union_mask(stim_dp, corr_neu & areas["mHV"], 95, 5)`; `local_keep = (reward_dp >= 0.3) & (stim_dp[ahv_idx] >= 0)`; `keep_mask = mhv_mask | ahv_mask`.

iii. The agent explicitly calls this a tractability compromise modeled on task-specific paper analyses: 102,541 of 4,691,034 neurons are exported so the decoder can concatenate the data in memory.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial begins at corridor entry conceptually, but only frames satisfying corridor and movement masks are exported. Native timestamps remain, so time covariates expose gaps caused by removed stationary frames. Trials have variable length and no padding.

ii. `valid = finite_trial & ft_corr & ft_move`; `neural_trial = ...[:, frame_idx]`; metadata sets `"temporal_alignment_event": "corridor entry (trial start)"` and `"off_start": 0.0`.

iii. The agent says trial-start alignment is preserved through `Trial_start_time` and that running-only filtering matches the paper.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native imaging frames are retained without temporal rebinning. The metadata uses the median observed frame interval across sessions, approximately 314.69 ms.

ii. `"time_bin_size": float(np.median([np.median(np.diff(...['ft'])) * 86400.0 for s in sessions]) * 1000.0)`.

iii. The notes identify the reference rate as 3.17 Hz and explain that native frame alignment is needed for time-resolved behavioral outputs.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from trial-level `SoundTime` and frame-level `ft` timestamps.

ii. `sound_time = np.asarray(beh["SoundTime"], dtype=float)`; `current_ft = ft[frame_idx]`.

iii. The mapping notes say these timestamps are in days and provide frame-aligned cue timing.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For every retained frame, the frame time is subtracted from cue time and converted from days to seconds, so values are positive before and negative after the cue.

ii. `time_to_cue = ((sound_time[trial_idx] - current_ft) * 86400.0).astype(np.float32)`.

iii. The notes explicitly describe this signed-seconds convention and report raw-data `np.allclose` checks.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It uses `ft` at exactly the same `frame_idx` columns used to slice neural activity.

ii. `current_ft = ft[frame_idx]`; `chunk[local_rows][:, frame_idx]`.

iii. The agent reports hand-checking the exported inputs and neural slices against the same raw frame masks.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from each subject's recording date strings in `Imaging_Exp_info.npy`.

ii. `first_date = min(parse_date(s.date_str) for s in sess_list)`; `parse_date(sess.date_str)`.

iii. The notes call it a calendar-day offset from the mouse's first unique imaging recording.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The agent computes one-based elapsed calendar days: first recording is day 1 and later values are date difference plus one (up to 93), then broadcasts the value over every retained frame.

ii. `day_map[subject][sess.rec_id] = float((parse_date(sess.date_str) - first_date).days + 1)`; `day_of_training = np.full(len(frame_idx), day_value, dtype=np.float32)`.

iii. The agent says actual recording chronology is a continuous per-trial covariate; it does not justify choosing elapsed calendar days rather than ordinal training-session count.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It uses frame timestamps `ft` and per-trial `Trial_start_time`.

ii. `trial_start = np.asarray(beh["Trial_start_time"], dtype=float)`; `current_ft = ft[frame_idx]`.

iii. The notes identify `Trial_start_time` as corridor-entry timing.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Trial-start time is subtracted from each retained frame timestamp and converted from days to seconds.

ii. `time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)`.

iii. The agent says using actual timestamps preserves elapsed time even after non-running frames are removed.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is calculated for the exact `frame_idx` used for neural columns.

ii. `current_ft = ft[frame_idx]`; `chunk[local_rows][:, frame_idx]`.

iii. The notes report independent alignment checks against raw data.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes from the trial-level `isRew` flag.

ii. `is_rew = np.asarray(beh["isRew"], dtype=float)`.

iii. The notes say this correctly captures rewarded, unrewarded, and mixed sessions.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The 0/1 trial value is broadcast unchanged across retained frames as float32.

ii. `reward_available = np.full(len(frame_idx), is_rew[trial_idx], dtype=np.float32)`.

iii. Broadcasting gives every trial a consistent feature-by-time input matrix.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived directly from per-trial `WallName` strings.

ii. `wall_name = as_str_array(beh["WallName"])`; `category_to_idx[str(wall_name[trial_idx])]`.

iii. The agent chose `WallName` because `stim_id` is NaN or masked in some duplicate/swap entries.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. All 15 distinct wall-name variants are sorted globally and encoded as separate integer classes, then broadcast across trial frames. Crop and swap variants are not collapsed to four base textures.

ii. `categories = sorted({str(v) ... for v in ...["WallName"]})`; `stim_code = np.full(len(frame_idx), category_to_idx[str(wall_name[trial_idx])], dtype=np.int16)`.

iii. The notes say preserving all naturalistic, grating, and swap strings avoids ambiguity and report 15 categories in the full data.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from lick frame numbers `LickFr` and their trial assignments `LickTrind`.

ii. `lick_fr = np.asarray(beh["LickFr"], dtype=float).astype(int)`; `lick_tr = np.asarray(beh["LickTrind"], dtype=float).astype(int)`.

iii. The notes identify both fields as the direct time/trial representation of lick events.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frames are truncated to integers, restricted to the current trial using `LickTrind`, and each retained frame is marked 1 if present in that set, otherwise 0.

ii. `lick_trial_frames = lick_fr[lick_tr == trial_idx]`; `licking = np.isin(frame_idx, lick_trial_frames).astype(np.int16)`.

iii. The agent describes a binary, frame-aligned output and reports exact reconstruction checks.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Membership is evaluated on the same retained raw `frame_idx` used for neural columns.

ii. `np.isin(frame_idx, lick_trial_frames)` and `chunk[local_rows][:, frame_idx]`.

iii. The sample plots and raw checks were reported to show lick/neural alignment.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It comes from frame-level `ft_Pos`, in decimeters.

ii. `ft_pos = np.asarray(beh["ft_Pos"], dtype=float)`.

iii. The notes relate 0–40 dm to the four-meter textured corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Values at retained frames are clipped to `[0, 39.999]`, divided into ten-decimeter intervals, and converted to integer labels 0–3.

ii. `pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)`; `pos_bin = np.clip((pos // 10.0).astype(np.int16), 0, 3)`.

iii. The notes call this the requested four equal one-meter bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are 10, 20, and 30 dm, yielding `[0,10)`, `[10,20)`, `[20,30)`, and `[30,40]`, labeled `0-1m` through `3-4m`.

ii. `"output_values": [..., ["0-1m", "1-2m", "2-3m", "3-4m"], ...]` and `(pos // 10.0)`.

iii. This directly follows the task's four one-meter category requirement.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is indexed by the same retained `frame_idx` used for neural activity.

ii. `ft_pos[frame_idx]`; `chunk[local_rows][:, frame_idx]`.

iii. The agent reports exact raw-data checks and near-equal position-bin occupancy.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from frame-level `ft_RunSpeed` at all retained running corridor frames.

ii. `np.asarray(beh["ft_RunSpeed"], dtype=float)[np.concatenate(trial_masks)]`.

iii. The notes identify this as the direct aligned running-speed stream.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A behavior-only first pass pools retained speed values from every selected session and computes three global quantile thresholds. Each trial's raw speeds are digitized with those thresholds.

ii. `speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)`; `np.digitize(values, edges, right=False)`.

iii. The agent says global quartiles satisfy the request for bins containing 25% of retained data and avoid loading neural files in the first pass.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Categories 0–3 (`q1`–`q4`) are determined by the three global 25th, 50th, and 75th percentile values; ties are not rank-split.

ii. `return np.clip(np.digitize(values, edges, right=False), 0, 3).astype(np.int16)`.

iii. Full-data notes report an approximately/exactly 25% occupancy for each global category.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Raw speed is indexed at the same `frame_idx` used for neural columns before digitization.

ii. `speed_bin = speed_to_bin(ft_speed[frame_idx], speed_edges)`.

iii. Frame alignment is part of the agent's raw reconstruction checks.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Duplicate behavior references are cross-validated and conversion aborts if they disagree; nonfinite trial indices are excluded; absent running frames cause an exception; d-prime NaN/Inf values become zero; selection has 128-cell fallbacks. The code does not truncate behavior arrays to neural length or gracefully drop empty trials/sessions.

ii. `np.allclose(..., equal_nan=True)`; `finite_trial = np.isfinite(ft_trial)`; `np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)`; `if len(frame_idx) == 0: raise ValueError(...)`.

iii. The notes emphasize duplicate validation and successful full-data checks, but do not document general missing-data recovery; they characterize the completed full conversion as clean.

## 12-a. What are the most time-consuming steps of the code?

i. Loading and concatenating very large spike files, computing per-session selectivity over them, and position-interpolating aHV activity for rewarded sessions dominate. The full run took 1,013.7 seconds.

ii. `spk = np.concatenate(spk_chunks, axis=0)`; `stim_dp = dprime(spk[:, stim_pos_fr], spk[:, stim_neg_fr])`; `interp_spk = utils.get_interpPos_spk(...)`.

iii. The notes estimate rewarded sessions as slower and say restricting interpolation to aHV and selecting neurons before export were the principal speed/memory compromises.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. `compute_trial_masks` scans the entire frame vector once per trial; trial export loops over every trial and imaging chunk. The neuron region-name/index construction also uses a Python list comprehension. These could be grouped/indexed more directly, though variable trial lengths still require final per-trial objects.

ii. `for trial in range(int(beh["ntrials"])): frame_idx = np.flatnonzero(valid & (ft_trial_int == trial))`; `for trial_idx, frame_idx in enumerate(trial_masks):`; `for chunk, local_rows in zip(spk_chunks, chunk_local_rows):`.

iii. The notes identify expensive interpolation but do not explicitly discuss these vectorization opportunities.

## 12-c. What processing does the code repeat multiple times?

i. Behavior files are repeatedly loaded in canonical selection, sample selection, global metadata, category enumeration, metadata time-bin calculation, and conversion. Trial masks are recomputed in the metadata pass and conversion. Selected spike rows are first part of a full concatenation for selection and later gathered again chunk-by-chunk for export.

ii. Repeated calls include `load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]` and `trial_masks = compute_trial_masks(beh)` in both `build_global_metadata` and `convert_dataset`.

iii. The notes praise the behavior-only first pass for avoiding neural loads but do not acknowledge the repeated behavior loading/masking.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `build_global_metadata` computes `dts` and `category_values` but never returns or uses them. A full neuron-by-frame `spk` concatenation is created solely to select indices, then discarded before selected rows are gathered again. Optional plots compute extra summaries only when requested.

ii. `dts.append(...)`; `category_values.update(...)`; `return subjects, subject_to_idx, day_map, speed_edges`; `spk = np.concatenate(spk_chunks, axis=0)`.

iii. The agent does not document the unused local variables. It justifies the full activity processing used for selection as reference-style neuron curation, although those intermediate arrays are not exported.
