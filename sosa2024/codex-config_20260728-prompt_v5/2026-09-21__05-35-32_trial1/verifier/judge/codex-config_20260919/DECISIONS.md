# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively enumerates every matching NWB file under `/app/data`, sorts them, and in full mode processes all 152 files. It reads NWB datasets directly with `h5py` and imports the paper repository's session metadata.

ii. `all_files = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))`; `with h5py.File(nwb_path, "r") as h5:`

iii. The notes justify this as using all 11 mice and all 152 released sessions, with NWB treated as the authoritative serialized dataset.

## 1-b. How are the data split into subjects?

i. Subject IDs are read from each NWB file, unique IDs are sorted, and each session receives an index into that list.

ii. `subject_id = h5["general/subject/subject_id"][()].decode()`; `subjects = sorted({sess["subject_id"] for sess in converted_sessions})`

iii. The notes say NWB IDs such as `m3` are canonical and unambiguous.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session; sorted files become separate top-level session entries.

ii. `for file_idx, nwb_path in enumerate(selected_files, start=1):`; `"neural": [sess["neural_trials"] for sess in converted_sessions]`

iii. The agent observed one file per subject/day and 152 files, consistent with the study schedule.

## 1-d. How are the data split into trials?

i. Trial starts are positive `trial_start` samples during scanning. Teleport samples during scanning are candidate ends. Each start is paired with the next later teleport; exported slices include the start and exclude the teleport frame.

ii. `start_idx = np.where((trial_start > 0) & (scanning > 0))[0]`; `end_idx = np.where((teleport > 0) & (scanning > 0))[0]`; `trial_slice = slice(start, end)`

iii. The notes say explicit start/teleport events match the reference trial definition and exclude tunnel/corrupted teleport samples.

## 1-e. How are trials filtered based on quality controls?

i. Incomplete unpaired trials and empty trials are omitted. Entire trials are also removed when more than 30% of their frames have raw lick count greater than 2; sessions with fewer than two retained trials fail.

ii. `bad[i] = np.mean(trial_lick > 2) > LICK_ERROR_THRESHOLD`; `if bad_lick: continue`; `if pos_trial.size == 0: continue`

iii. The agent cites the manuscript lick-sensor rule and reports exactly 81 removals (0.663%), close to the paper's 81 (~0.65%).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes directly from each plane's NWB `processing/ophys/Deconvolved` dataset, restricted by `iscell` and pooled across planes.

ii. `deconv_root = h5["processing/ophys/Deconvolved"]`; `plane_data = deconv_root[f"plane{int(plane)}"]["data"][:min_frames, local_cols]`

iii. The notes claim the stored deconvolved signal most directly matches event-like activity used by the manuscript and avoids reimplementing dF/F.

## 2-b. How is the `neural` data processed?

i. Stored deconvolved samples are selected by plane/ROI, converted to `float16`, pooled into a time-by-cell array, trial-sliced, and transposed to neuron-by-time. No dF/F, smoothing, baseline estimation, or new OASIS deconvolution is performed.

ii. `deconv[:, kept_positions] = plane_data.astype(np.float16)`; `neural_trial = np.ascontiguousarray(deconv[trial_slice, :].T, dtype=np.float16)`

iii. The agent chose direct NWB activity for tractability and manuscript relevance, and used `float16` to halve the roughly 5 GB output.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs with `iscell[:, 0] > 0.5` are retained. The contemplated speed-correlation interneuron filter is not implemented.

ii. `keep_mask = iscell[:, 0] > 0.5`; `keep_indices = np.flatnonzero(keep_mask)`

iii. The notes identify `iscell` as manual Suite2p curation. They acknowledge the paper's additional interneuron exclusion but leave it unapplied.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It is aligned to trial start by using the same `[start:end)` frame slice for neural and behavioral arrays; the first exported sample is the `trial_start` frame.

ii. `trial_slice = slice(start, end)`; `neural_trial = ... deconv[trial_slice, :].T`

iii. The notes state that all NWB streams are already imaging-frame aligned, so common slicing is sufficient.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The reported resolution is `1/15.5078125 s = 64.4839 ms`. No temporal aggregation/interpolation is performed; behavior timestamps are the master grid even for multi-plane sessions.

ii. `COMMON_FRAME_DT_S = 1.0 / COMMON_FRAME_RATE_HZ`; `TIME_BIN_SIZE_MS = COMMON_FRAME_DT_S * 1000.0`

iii. The agent found the behavior timestamps share this effective rate and intentionally ignored inconsistent 31 Hz series metadata for two-plane recordings.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the `position` behavioral series timestamps and the paired trial-start index.

ii. `frame_timestamps = beh_root["position"]["timestamps"][()].astype(np.float64)`

iii. The notes select imaging-frame-aligned behavior timestamps as the canonical clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at trial start is subtracted from every timestamp in the trial.

ii. `t_trial = frame_timestamps[trial_slice] - frame_timestamps[start]`

iii. This directly creates seconds elapsed from the required alignment event.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The timestamp and neural arrays use the identical trial slice and are checked to have equal time dimensions.

ii. `if neural_trial.shape[1] != input_trial.shape[1] ...: raise RuntimeError(...)`

iii. The agent verified behavioral and neural streams are frame synchronized.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from `processing/behavior/BehavioralTimeSeries/environment/data`.

ii. `environment = beh_root["environment"]["data"][()].astype(np.float32)`

iii. The notes identify this stream as binary ENV1/ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Negative values are discarded, the trial median is rounded to an integer, and that value is repeated across the trial.

ii. `env_trial = int(np.rint(np.median(env_trial_vals)))`; `repeated_row(env_trial, T, np.float32)`

iii. The agent observed environment is constant per trial and wanted uniform `(d, T)` inputs.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It uses the raw `trial number` behavioral series at the trial-start frame.

ii. `trial_number = beh_root["trial number"]["data"][()].astype(np.int32)`; `trial_num = int(trial_number[start])`

iii. The notes describe this as a session-local trial number and report the expected 0–99 range.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The start-frame value is repeated across all trial frames without renumbering after filtering.

ii. `repeated_row(trial_num, T, np.float32)`

iii. Repetition provides a uniform time-varying matrix representation.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from NWB `Reward` event timestamps and the preceding paired trial's timestamp interval.

ii. `reward_event_ts = beh_root["Reward"]["timestamps"][()].astype(np.float64)`

iii. The notes use reward delivery as the binary rewarded/omitted indicator.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Each trial is marked rewarded if any event is in `[start_ts, end_ts)`. The current trial receives the immediately preceding original trial's mark, or 0 for trial zero, repeated across time. A removed bad-lick previous trial still supplies the history.

ii. `reward_outcomes_all[i] = int(np.any((reward_event_ts >= start_ts) & (reward_event_ts < end_ts)))`; `prev_reward = ... if trial_idx > 0 else 0`

iii. This implements the requested omitted=0/rewarded=1 history.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw position plus reward-zone A/B/C inferred from the session scene in `sessions_dict.py` and a hard-coded switch at trial 30.

ii. `reward_labels_all = scene_reward_labels(session_meta.scene, len(trial_pairs))`; `zone_start, zone_end = REWARD_ZONE_COORDS[reward_label]`

iii. The agent preferred the paper repository's scene metadata because the framewise reward-zone stream was interpreted as occupancy, not a zone label.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position minus the near zone edge is negative before the zone, zero inside it, and position minus the far edge is positive after it.

ii. `dist[before] = position_cm[before] - zone_start`; `dist[after] = position_cm[after] - zone_end`

iii. The notes describe this as distance to the nearest point in the active reward interval.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is mapped to seven categories at -50, -10, 0, +10, and +50 cm, with exact zero receiving class 3.

ii. `out[distance_cm < -50] = 0`; `out[distance_cm == 0] = 3`; `out[(distance_cm > 10) & (distance_cm <= 50)] = 5`

iii. The thresholds are stated to follow the decoder specification exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural activity are sliced with the same start-inclusive, teleport-exclusive indices and checked for equal length.

ii. `pos_trial = position[trial_slice]`; `neural_trial = ... deconv[trial_slice, :].T`

iii. The agent relies on common imaging-frame synchronization.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It comes directly from the behavioral `position/data` series.

ii. `position = beh_root["position"]["data"][()].astype(np.float32)`

iii. The notes identify it as position on the 450 cm virtual track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Per-frame positions are directly discretized; no smoothing or clipping is applied.

ii. `pos_bin = discretize_absolute_position(pos_trial)`

iii. Five 90 cm bins implement the requested equal partition.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Classes are `<90`, `[90,180)`, `[180,270)`, `[270,360]`, and `>360` cm.

ii. `out[(position_cm >= 270) & (position_cm <= 360)] = 3`; `out[position_cm > 360] = 4`

iii. These boundaries follow the literal instructions, including assigning exactly 360 cm to class 3.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The same trial slice is applied to position and neural data.

ii. `pos_trial = position[trial_slice]`; `neural_trial = ... deconv[trial_slice, :].T`

iii. Equal-length checks validate alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavioral `lick/data` series.

ii. `lick = beh_root["lick"]["data"][()].astype(np.float32)`

iii. The agent notes the raw stream contains counts that can exceed one.

## 9-b. What processing is involved in computing `output` *Lick*?

i. After bad-lick trials are removed, every positive count is converted to 1 and all others to 0.

ii. `lick_bin = (lick[trial_slice] > 0).astype(np.int16)`

iii. Binarization is required by the output specification and mirrors reference analysis behavior.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural arrays use the same per-trial frame slice.

ii. `lick[trial_slice]`; `deconv[trial_slice, :]`

iii. The common behavioral/imaging frame grid supplies alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from subject/day session metadata (`scene`) imported from the reference repository, plus trial ordinal for switch sessions.

ii. `session_meta = sessions_dict[(gcamp, exp_day)]`; `scene_reward_labels(session_meta.scene, len(trial_pairs))`

iii. The agent considered scene identity and the trial-30 switch rule more authoritative than framewise zone occupancy.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Fixed scenes yield one A/B/C label for all trials; transition scenes switch from source to destination after 30 trials. Labels map A/B/C to 0/1/2 and are repeated in time.

ii. `return [src] * min(switch_trial, n_trials) + [dst] * max(0, n_trials - switch_trial)`; `RZ_CODE = {"A": 0, "B": 1, "C": 2}`

iii. This mirrors the experimental design and reference session metadata.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It comes from the NWB `Reward` event timestamps and each trial's timestamp bounds.

ii. `reward_event_ts = beh_root["Reward"]["timestamps"][()].astype(np.float64)`

iii. Reward delivery is treated as the authoritative rewarded outcome.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is 1 if any reward timestamp falls from trial start (inclusive) to teleport time (exclusive), otherwise 0; the scalar is repeated across time.

ii. `int(np.any((reward_event_ts >= start_ts) & (reward_event_ts < end_ts)))`; `repeated_row(reward_outcome, T, np.int16)`

iii. This directly implements the requested per-trial binary output.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. All streams are truncated to the shortest behavior/neural plane length. Unpaired trailing trials and empty trials are dropped; invalid environment, NaNs, unexpected frame rate, dimension mismatches, or fewer than two trials raise errors. Bad lick-sensor trials are removed.

ii. `min_frames = min([frame_timestamps.shape[0], *plane_frame_counts])`; `if np.isnan(...).any(): raise RuntimeError(...)`

iii. The notes cite one dangling partial trial and say explicit paired events prevent it entering the export; strict validation avoids silently exporting malformed data.

## 13-a. What are the most time-consuming steps of the code?

i. Reading large per-plane deconvolved matrices and serializing the roughly 4.82 GB pickle dominate. The full run took about 107 seconds.

ii. `plane_data = deconv_root[...]["data"][:min_frames, local_cols]`; `pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)`

iii. The notes estimated billions of neural values and identify NWB I/O and pickle serialization as the main costs.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial-event pairing, bad-lick detection, reward-outcome assignment, per-trial conversion, and some plane/ROI indexing are Python loops. Session-wide reward and discretization operations could be vectorized before trial splitting, though ragged outputs still require slicing.

ii. `for start in start_idx:`; `for i, (start, end) in enumerate(trial_pairs):`; `for trial_idx, (...) in enumerate(...):`

iii. The notes accept session-at-a-time and per-trial processing as a memory-conscious compromise for variable-length trials.

## 13-c. What processing does the code repeat multiple times?

i. Each trial interval is traversed once to detect lick faults, again to compute reward outcomes, and again to construct arrays. Optional plotting traverses representative outputs again. Repeated constant rows are materialized for every trial.

ii. `detect_bad_lick_trials(lick, trial_pairs)`; `for i, (start, end) in enumerate(trial_pairs):`; `for trial_idx, ... in enumerate(...)`

iii. The agent prioritized clear staged checks; unlike the reference, it does not perform a separate all-file survey pass before conversion.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads the raw `trial number` array only for one value per trial; it builds summary fields and temporary kept-label/outcome/number lists, with `kept_trial_numbers` never subsequently used. When requested, processing plots and their intermediate statistics do not affect the dataset.

ii. `kept_trial_numbers.append(trial_num)`; `if make_plot: make_processing_plot(...)`

iii. Plotting was deliberately optional for visual sanity checks; the notes otherwise emphasize bounded-memory conversion rather than extra analysis.
