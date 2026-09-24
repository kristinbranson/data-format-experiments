# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globs every matching NWB file one directory below `/app/data`, numerically sorts the parsed mouse/session IDs, and converts each file with `h5py`.

ii. `paths = sort_session_paths(glob.glob(str(DATA_ROOT / "sub-*" / "sub-*_behavior+ophys.nwb")))` and `with h5py.File(path, "r") as f:`

iii. The trajectory says it inspected the NWB schema and chose direct NWB loading; it later reported all 152 sessions were retained.

## 1-b. How are the data split into subjects?

i. The subject ID is read from each NWB file; unique retained IDs are naturally sorted and each session gets an index into that list.

ii. `subject = f["general/subject/subject_id"][()].decode()` and `subjects = sorted({session["subject"] for session in session_results}, key=lambda s: int(s[1:]))`

iii. The agent favored authoritative NWB metadata and filename-based deterministic ordering.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session and produces one element in each top-level session list; sessions with fewer than two usable trials are skipped.

ii. `converted = convert_session(path)` and `if len(converted["neural"]) < 2: ... continue`

iii. It identified filenames and NWB `session_id` as session boundaries and enforced the decoder's two-trial minimum.

## 1-d. How are the data split into trials?

i. Positive `trial_start` samples are paired, in order, with the next positive `teleport`; slices are `[start, stop)`, excluding teleport.

ii. `starts = np.flatnonzero(trial_start_signal > 0)`; `teleports = np.flatnonzero(teleport_signal > 0)`; `segments.append((int(start), int(stop)))`

iii. The NWB has no populated trials table, so the trajectory explicitly chose frame-aligned `trial_start` and `teleport` streams.

## 1-e. How are trials filtered based on quality controls?

i. A trial is removed when more than 35% of its raw lick samples exceed 2; trials too short to form one 8-frame bin are also removed, and sessions with fewer than two survivors are removed.

ii. `lick_error = bool(np.mean(lick_counts[start:stop] > 2.0) > LICK_ERROR_THRESHOLD)` and `if info["lick_error"]: ... continue`

iii. The agent described this as lick-sensor-error exclusion and used bin viability/session viability as practical decoder checks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Final neural values come from the NWB `processing/ophys/Deconvolved` series. `Fluorescence` and `Neuropil` are used only to identify putative interneurons.

ii. `deconvolved = load_roi_response_matrix(f, "processing/ophys/Deconvolved", curated_cell_idx, plane_idx_all).T`

iii. The trajectory considered reproducing processing, then trusted the released deconvolved stream as the activity signal while retaining paper-inspired filtering.

## 2-b. How is the `neural` data processed?

i. Plane-specific matrices are pooled in segmentation-table order, restricted to curated non-interneurons, averaged over nonoverlapping 8-frame bins, nonfinite values become zero, and values are stored as `float16`.

ii. `neural = bin_2d_mean(deconvolved[:, start:stop_trimmed], BIN_FRAMES)` and `neural = np.nan_to_num(neural, ...).astype(np.float16)`

iii. Multipane pooling fixed an observed indexing bug. Eight-frame bins were chosen because the agent judged native 15.5 Hz data too large to serialize/train directly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It keeps `iscell[:,0] > 0.5`, then removes cells whose trial-wise recomputed dF/F has speed correlation above 0.5.

ii. `curated_cell_idx = np.flatnonzero(iscell[:, 0] > 0.5)` and `keep_cells = ~is_interneuron`

iii. The trajectory identifies `iscell` as Suite2p/manual curation and the correlation threshold as the paper's putative-interneuron rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural frames are sliced from the `trial_start` index through the frame before `teleport`; bins begin at that same start and leftover terminal frames are discarded.

ii. `stop_trimmed = start + n_bins * BIN_FRAMES` and `neural = bin_2d_mean(deconvolved[:, start:stop_trimmed], BIN_FRAMES)`

iii. The agent states that this preserves trial-start alignment across frame-aligned NWB streams.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Eight native frames at 15.5078125 Hz are averaged per bin: 0.515868 s or about 515.9 ms. Yes, substantial temporal rebinning is applied.

ii. `BIN_FRAMES = 8`; `TIME_BIN_SIZE_S = BIN_FRAMES / FRAME_RATE_HZ`

iii. It was an explicit tractability choice for pickle size and decoder training, not processing specified by the paper.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the `position/timestamps` frame timestamps and the selected trial-start frame.

ii. `frame_times = np.asarray(behavior["position/timestamps"][:], dtype=np.float64)`

iii. Position timestamps were treated as the shared frame clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Every eighth timestamp starting at trial start has the start timestamp subtracted.

ii. `time_from_start = (frame_times[start:stop_trimmed:BIN_FRAMES] - frame_times[start]).astype(np.float32)`

iii. This makes bin times start at zero while matching the rebinned matrices.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Timestamp subsampling and neural binning use identical trial bounds and one output per 8-frame block; time labels represent each block's first frame.

ii. `frame_times[start:stop_trimmed:BIN_FRAMES]` alongside `deconvolved[:, start:stop_trimmed]`

iii. The shared NWB frame indices were assumed already aligned.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Primarily the frame-aligned `environment/data`; the scene encoded in NWB `identifier` supplies an expected fallback and mismatch check.

ii. `environment = np.asarray(behavior["environment/data"][:], dtype=np.float32)` and `zone_labels, expected_envs = expected_trial_labels(scene, ...)`

iii. The trajectory specifically investigated scene identifiers to recover schedule metadata robustly.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Valid nonnegative samples in a trial are reduced by rounded median; if absent, scene-derived expected environment is used; the scalar is repeated across bins.

ii. `env_value = int(np.rint(np.nanmedian(env_slice)))` and `np.full(n_bins, info["environment"], ...)`

iii. Median/fallback handles malformed or missing frame samples while maintaining a trial-level variable.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It comes from the frame-aligned `trial number/data`, with the enumerated segment index as fallback.

ii. `trial_number_signal = np.asarray(behavior["trial number/data"][:], dtype=np.float32)`

iii. The agent used the released behavioral stream where available.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. Valid nonnegative values are reduced by rounded median, otherwise `trial_idx` is used; the result is repeated across time bins.

ii. `trial_number = int(np.rint(np.nanmedian(trialnum_slice)))` and `np.full(n_bins, info["trial_number"], ...)`

iii. This robustly turns a frame-wise constant field into a per-trial covariate.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `Reward/timestamps` compared with the position timestamp interval of each segmented trial.

ii. `reward_outcome = int(np.any((reward_timestamps >= frame_times[start]) & (reward_timestamps < frame_times[stop])))`

iii. Reward events have their own timestamps, so interval membership was used.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Current outcomes are calculated for every unfiltered segment; trial zero gets 0, otherwise the immediately preceding original trial's outcome is repeated over all bins.

ii. `prev_outcome = reward_outcomes[trial_idx - 1] if trial_idx > 0 else 0`

iii. This preserves “previous trial” even when that previous trial is later excluded for lick error.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses binned `position/data` and zone A/B/C inferred from the scene name and a hard-coded switch at trial 30, with fixed bounds A 80–130, B 200–250, C 320–370 cm.

ii. `zone_labels, expected_envs = expected_trial_labels(scene, len(trial_segments))` and `zone_bounds = REWARD_ZONES[info["zone_label"]]`

iii. The trajectory says scene identifiers encode schedules, so it chose explicit labels rather than noisy zone-entry heuristics.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Raw position is averaged per 8 frames and clipped to 0–450 cm. Distance is negative to the leading boundary, zero inside, and positive past the trailing boundary.

ii. `pos_binned = bin_1d_mean(...)`; `distance[before] = position_cm[before] - start_cm`; `distance[after] = position_cm[after] - end_cm`

iii. This implements signed distance to any point in the zone after the agent's temporal rebinning.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit Boolean masks implement seven requested classes at -50, -10, 0, 10, and 50 cm, with exact zero isolated.

ii. `out[distance_cm == 0.0] = 3` and `out[(distance_cm > 0.0) & (distance_cm <= 10.0)] = 4`

iii. The boundaries directly follow the decoder instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data use the same start/trimmed-stop indices and both yield one value per 8-frame block.

ii. `neural = bin_2d_mean(...)` and `pos_binned = bin_1d_mean(...)`

iii. The streams are frame-aligned in NWB and processed with identical bins.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It comes from `processing/behavior/BehavioralTimeSeries/position/data`.

ii. `position = np.asarray(behavior["position/data"][:], dtype=np.float32)`

iii. This is the direct corridor position measurement.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is averaged within 8-frame bins, clipped to [0,450], then categorized.

ii. `pos_binned = bin_1d_mean(...)` and `pos_binned = np.clip(pos_binned, 0.0, TRACK_LENGTH_CM)`

iii. Averaging aligns it to rebinned neural data; clipping handles small out-of-track errors.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Explicit masks create five bins: <90, [90,180), [180,270), [270,360), and >=360 cm.

ii. `out[(position_cm >= 270.0) & (position_cm < 360.0)] = 3`; `out[position_cm >= 360.0] = 4`

iii. These are five 90-cm bins over the stated 450-cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is averaged over exactly the same 8-frame trial blocks as neural activity.

ii. `position[start:stop_trimmed]` and `deconvolved[:, start:stop_trimmed]`

iii. Shared frame indices provide alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from frame-aligned `lick/data` counts.

ii. `lick_counts = np.asarray(behavior["lick/data"][:], dtype=np.float32)`

iii. The trajectory identified this as the released lick stream.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Trials with pervasive values >2 are dropped. In retained trials values >1 are capped at 1, each 8-frame block is max-pooled, then thresholded above zero.

ii. `lick_trial[lick_trial > 1.0] = 1.0`; `lick_binned = bin_1d_max(...)`; `lick_binned = (lick_binned > 0).astype(np.uint8)`

iii. Max pooling preserves whether any lick occurred during a coarse bin; exclusion addresses sensor-error trials.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It uses the same trial boundaries and 8-frame blocks as neural data, with max rather than mean aggregation.

ii. `lick_counts[start:stop_trimmed]` and `deconvolved[:, start:stop_trimmed]`

iii. Shared frames ensure binwise alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the scene name in NWB `identifier` and trial index, not from the frame-wise `reward_zone` series.

ii. `scene = f["identifier"][()].decode().split("/")[-1]` and `expected_trial_labels(scene, len(trial_segments))`

iii. The agent found identifiers such as `Env1_LocationC_to_A` and chose explicit schedule parsing.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Regex parsing extracts pre/post zones. `_to_` scenes switch at hard-coded `SWITCH_TRIAL = 30`; A/B/C become 0/1/2 and are repeated over bins.

ii. `zones = [info["zone_before"]] * pre_n + [info["zone_after"]] * post_n` and `ord(info["zone_label"]) - ord("A")`

iii. This reflects the scene naming convention and the agent's inferred fixed switch schedule.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It derives from `Reward/timestamps` and per-frame `position/timestamps` trial bounds.

ii. `reward_timestamps = np.asarray(behavior["Reward/timestamps"][:], dtype=np.float64)`

iii. Separate reward-event timestamps require temporal interval matching.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Outcome is 1 if any reward timestamp lies in `[trial-start time, teleport time)`, otherwise 0, repeated across the trial's bins.

ii. `np.any((reward_timestamps >= frame_times[start]) & (reward_timestamps < frame_times[stop]))`

iii. This implements binary rewarded/omitted outcome per trial.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing environment/trial-number samples use schedule/index fallbacks; neural/speed NaN or infinity becomes zero; position is clipped; dF/F filtering ignores NaNs; incomplete bins are discarded; unpaired final starts are ignored; bad-lick trials and under-two-trial sessions are skipped.

ii. `np.nan_to_num(neural, nan=0.0, posinf=0.0, neginf=0.0)`; `if teleport_ptr >= len(teleports): break`; `if env_slice.size: ... else: env_value = expected_envs[trial_idx]`

iii. These are defensive choices made while inspecting release irregularities and satisfying verifier constraints.

## 13-a. What are the most time-consuming steps of the code?

i. Reading three large ROI matrices per session, recomputing trial-wise dF/F for every curated cell, filtering, 8-frame conversion, and serializing are dominant; full conversion took several minutes in the trajectory.

ii. `load_roi_response_matrix(...)` is called for fluorescence, neuropil, and deconvolved data; `find_putative_interneurons(...)` loops through trials.

iii. Runtime updates singled out full conversion and multipane assembly; binning was introduced to keep serialization/training tractable.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial pairing, per-trial dF/F accumulation, trial metadata construction, retained-trial conversion, session conversion, and per-plane loading are Python loops. Some session-wide masks/statistics and fixed-state operations could be vectorized, though variable trial lengths limit full vectorization.

ii. `for start, stop in trial_segments:`; `for trial_idx, info in enumerate(trial_info):`; `for plane_id in plane_ids:`

iii. The trajectory provides no explicit vectorization analysis; correctness and manageable memory were prioritized.

## 13-c. What processing does the code repeat multiple times?

i. Every trial is traversed once for interneuron statistics, once for reward/environment/trial metadata, and once for final arrays. Trial slices and validity handling are therefore revisited, and three similarly structured ROI datasets are independently loaded.

ii. The three loops beginning `for start, stop in trial_segments`, `for trial_idx, (start, stop)`, and `for trial_idx, info` repeat trial access.

iii. The separation keeps cell filtering, trial metadata, and output construction modular and avoids retaining more intermediates.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes `expected_envs` mostly for fallback/mismatch metadata, `speed_corr` and numerous counters only for metadata, and full dF/F solely to discard interneurons; dF/F itself is discarded. It also loads raw `reward_zone` nowhere, despite being available.

ii. `is_interneuron, speed_corr = find_putative_interneurons(...)`; `environment_mismatch_trials += 1`; metadata stores counters/ranges.

iii. These diagnostics document curation, but the decoder consumes neither the dF/F intermediate nor most session metadata.
