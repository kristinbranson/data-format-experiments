# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively enumerates every matching NWB file, sorts by numeric mouse/session, and opens each directly with `h5py`; each file becomes one session.

ii. `for path in sorted(data_dir.glob("sub-*/sub-*_behavior+ophys.nwb"), key=subject_session_key):` and `with h5py.File(path, "r") as f:`

iii. The trajectory says it inspected the data layout, confirmed 11 mice and 152 sessions, and chose direct HDF5 access after finding that the export stores aligned streams in processing modules and has no NWB trials table.

## 1-b. How are the data split into subjects?

i. Subject identity is read from each NWB file, then unique IDs are collected in first-session order and mapped to `subject_idx`.

ii. `subject = decode_scalar(f["general/subject/subject_id"][()])` and `subject_to_idx[subject] = len(subjects)`

iii. The agent checked subject metadata and manuscript-level counts; it reported 11 converted mice.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session; files are ordered by mouse number and session number parsed from their paths.

ii. `session = int(path.stem.split("_ses-")[-1].split("_")[0])` and `session_records.append(build_session(path))`

iii. The trajectory identifies the NWB files as session recordings and validates the resulting 152-session count.

## 1-d. How are the data split into trials?

i. Starts are positive `trial_start` samples. For each start, the end is one past the last position in `[0, 450.5]` before the next start, excluding teleport-zone samples.

ii. `starts = np.flatnonzero(trial_start > 0)` and `valid = np.flatnonzero((trial_pos >= 0.0) & (trial_pos <= 450.5)); end = start + valid[-1] + 1`

iii. The agent reasoned that trials should run from `trial_start` through the last corridor frame, matching its reading of the 450-cm lap and excluding teleport frames.

## 1-e. How are trials filtered based on quality controls?

i. No per-trial quality filter is applied. Empty segments are dropped while finding boundaries, and a session must contain at least two trials.

ii. `if len(valid) == 0: continue` and `if ntrials < 2: raise ValueError(...)`

iii. The trajectory discusses keeping corridor frames and checking trial statistics, but gives no justification for omitting the reference's short-trial filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Final neural data comes from NWB `processing/ophys/Deconvolved`. `Fluorescence` and `Neuropil` are used only to decide which putative interneurons to exclude.

ii. `events = f["processing/ophys/Deconvolved"]` and `fluorescence = f["processing/ophys/Fluorescence"]`

iii. The agent concluded that the exported deconvolved traces were already the paper's frame-aligned events and documented that choice explicitly.

## 2-b. How is the `neural` data processed?

i. Stored deconvolved traces are truncated to a common length, manually curated, filtered by an independently reconstructed dF/F interneuron mask, concatenated across planes, transposed, and trial-sliced. They are not recomputed from F/Fneu.

ii. `plane_data = np.asarray(events[plane_name]["data"][:common_length, curated_mask], dtype=np.float32)` and `neural = events[start:end].T.astype(np.float32)`

iii. The agent believed the NWB deconvolution was suitable as the decoder signal. It reconstructed paper-like dF/F only for the speed-correlation screen to mirror the extra cell curation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It keeps `iscell` ROIs, then excludes cells whose reconstructed per-trial dF/F has correlation greater than 0.5 with speed.

ii. `iscell = np.asarray(seg["iscell"])[:, 0].astype(bool)` and `keep_mask = ~is_int` where `all_is_int.append(corr > 0.5)`

iii. The agent cites the Methods and `dayData.py` for manual curation and the `corr(dff, speed) > 0.5` putative-interneuron rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples are sliced from the detected `trial_start`; no resampling or offset is applied.

ii. `for trial_idx, (start, end) in enumerate(zip(starts, ends)):` and `neural = events[start:end].T`

iii. The agent states that behavior and ophys are already frame-aligned in the NWB and calls the alignment event trial start/entry onto the corridor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. Metadata uses the median behavior timestamp interval across sessions, about 64.48 ms.

ii. `dt_seconds = float(np.median(np.diff(position_ts)))` and `"time_bin_size_ms": float(np.median(dt_all) * 1000.0)`

iii. The agent checked that this is approximately 15.5 Hz per plane and consistent with the paper.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from `position/timestamps`.

ii. `position_ts = np.asarray(beh["position/timestamps"], dtype=np.float64)`

iii. The trajectory identifies the behavior timestamps as frame-aligned with neural samples.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp in each trial is subtracted from all timestamps in that trial and the result is cast to float32.

ii. `time_trial = (position_ts[start:end] - position_ts[start]).astype(np.float32)`

iii. This implements elapsed time from the requested trial-start alignment.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Exactly the same `[start:end]` frame slice is used for timestamps and neural data.

ii. `time_trial = ... position_ts[start:end] ...` and `neural = events[start:end].T`

iii. The agent relied on the NWB's frame alignment and also truncated all dense streams to a shared length.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The emitted value is derived from the session scene string in the NWB `identifier`, not from the raw `environment` series; that series is used only as a sanity check.

ii. `scene = identifier.split("/")[-1]` and `env_by_trial, ... = scene_schedule(scene, ntrials)`

iii. The agent says this reproduces the schedule logic in the original behavior code and verified zero mismatches against the behavior stream.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Scene names are regex-parsed; Env1/Env2 map to 0/1, and switch scenes change after trial 30. The selected scalar is repeated across a trial.

ii. `ENV_TO_IDX = {"Env1": 0, "Env2": 1}` and `np.full(time_trial.shape[0], env_by_trial[trial_idx], dtype=np.float32)`

iii. The agent justified the hard-coded switch as matching the manuscript/code schedule and validated it against observed environment values.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is the zero-based index of each detected trial, indirectly based on `trial_start` and position segmentation.

ii. `for trial_idx, (start, end) in enumerate(zip(starts, ends)):`

iii. The agent describes this as zero-indexed within-session numbering matching the aligned stream.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index is repeated at every time point of that trial.

ii. `np.full(time_trial.shape[0], trial_idx, dtype=np.float32)`

iii. No further processing was considered necessary for a per-trial variable.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It ultimately comes from sparse `Reward/timestamps` and `position/timestamps` used to label each current trial.

ii. `reward_ts = np.asarray(beh["Reward/timestamps"], dtype=np.float64)`

iii. The agent chose sparse reward timestamps as the direct record of delivery events.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A trial is rewarded if any reward timestamp falls inclusively between its first and last timestamps. The prior label is repeated through the current trial; trial zero receives 0.

ii. `reward_outcome.append(int(np.any((reward_ts >= position_ts[start]) & (reward_ts <= position_ts[end - 1] + 1e-9))))` and `prev_reward = reward_outcome[trial_idx - 1] if trial_idx > 0 else 0`

iii. This directly implements omitted=0/rewarded=1 and the agent explicitly documents the first-trial convention.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw position plus zone identity parsed from the NWB identifier/scene. The raw `reward_zone` stream is only a validation check.

ii. `pos_trial = np.clip(position[start:end], 0.0, 450.0)` and `zone_start, zone_end = zone_bounds[trial_idx]`

iii. The agent believed scene parsing exactly reproduced fixed and trial-30 switch schedules; observed zone occupancy agreed whenever present.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position is clipped to the track. Distance is negative to the zone's leading edge before it, zero inside, and positive from its trailing edge after it.

ii. `dist[before] = position_cm[before] - zone_start; dist[after] = position_cm[after] - zone_end`

iii. The agent says this is signed distance to the nearest point of the active 50-cm zone, matching the task definition.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It explicitly assigns seven requested bins at -50, -10, 0, 10, and 50 cm, keeping exact zero separate.

ii. `bins[(dist >= -10.0) & (dist < 0.0)] = 2; bins[dist == 0.0] = 3; bins[(dist > 0.0) & (dist <= 10.0)] = 4`

iii. The boundaries were taken directly from the decoder instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data use the same `[start:end]` samples.

ii. `pos_trial = ... position[start:end]` and `neural = events[start:end].T`

iii. The agent relies on frame alignment in the NWB.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It comes from the behavior `position/data` series.

ii. `position = np.asarray(beh["position/data"], dtype=np.float32)`

iii. The agent treats this as the corridor coordinate in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Per-trial position is clipped to `[0,450]`, divided by 90, floored, and capped at class 4.

ii. `clipped = np.clip(position_cm, 0.0, 450.0); bins = np.floor(clipped / 90.0).astype(np.int64); bins[bins > 4] = 4`

iii. The agent chose clipping to absorb small out-of-range samples and create five equal corridor bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five 90-cm bins are produced: values below 90 are 0, then 1--3, with 360 cm and above in 4.

ii. `bins = np.floor(clipped / 90.0).astype(np.int64)`

iii. This follows the requested five equal-sized bins over a 450-cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The same trial slice is used for position and neural events.

ii. `pos_trial = np.clip(position[start:end], ...)` and `neural = events[start:end].T`

iii. The NWB streams were judged already aligned.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from behavior `lick/data`.

ii. `lick = np.asarray(beh["lick/data"], dtype=np.float32)`

iii. The agent identifies this as the framewise lick-count stream.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Any positive raw lick value becomes 1; all others become 0.

ii. `lick_trial = (lick[start:end] > 0).astype(np.int64)`

iii. Binarization is required by the decoder specification.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural arrays use the same start/end indices.

ii. `lick[start:end]` and `events[start:end].T`

iii. The agent relies on stored frame alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The output is derived from the session scene string and trial index; `reward_zone/data` and position only check the inferred schedule.

ii. `scene = identifier.split("/")[-1]` and `zone_idx = np.array([ZONE_TO_IDX[z] for z in zone_labels])`

iii. The agent says scene-name schedules mirror the reference behavior code and achieved perfect agreement with observed occupancy where available.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Regex rules assign A/B/C (0/1/2), with fixed scenes constant and switch scenes changing at trial 30; the label is repeated through the trial.

ii. `split = min(change_trial, ntrials)` and `np.full(time_trial.shape[0], zone_idx[trial_idx], dtype=np.int64)`

iii. The hard-coded switch was justified from manuscript/task conventions and sanity checks.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It comes from sparse behavior `Reward/timestamps`, compared with position timestamps. The data occupy output row 5, but the metadata accidentally labels index 5 `reward_zone_location` and appends `reward_outcome` as a seventh name.

ii. `reward_ts = np.asarray(beh["Reward/timestamps"], dtype=np.float64)`

iii. The agent regarded actual reward-event times as the appropriate outcome source.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. It tests whether any reward event lies within each trial's timestamp interval and repeats the resulting binary label across the trial.

ii. `reward_trial = reward_outcome[trial_idx]` and `np.full(time_trial.shape[0], reward_trial, dtype=np.int64)`; the malformed metadata is `[..., "reward_zone_location", "reward_zone_location", "reward_outcome"]` for a six-row output.

iii. This matches a per-trial omitted/rewarded target; the agent also checked the omission fraction against the paper.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. All dense behavior, fluorescence, neuropil, and deconvolved streams are cropped to their minimum common length. Empty trials are skipped, missing observed zone occupancy is tolerated because labels come from scenes, zero denominators in correlations yield zero, and invalid scenes/mask mismatches/no-neuron sessions raise errors.

ii. `common_length = min(dense_lengths)`; `if len(valid) == 0: continue`; and `np.divide(..., out=np.zeros_like(...), where=denom > 0)`

iii. The agent describes these as alignment and defensive checks found during exploration; it records the number of clipped samples in metadata.

## 13-a. What are the most time-consuming steps of the code?

i. The code does not explicitly profile itself. By structure and trajectory, the full 152-session pass, repeated large-array I/O, per-trial dF/F filtering for interneuron detection, writing the 9.3-GB pickle, and later decoder training are the expensive stages.

ii. `is_int = compute_interneuron_mask(...)`, `events = load_curated_events(...)`, and `pickle.dump(data, f)`

iii. The trajectory calls full conversion and decoder training “heavy/expensive” and avoided rerunning the completed conversion.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial-boundary discovery, per-trial environment/zone/reward summaries, per-trial dF/F baseline construction, and final trial assembly are Python loops. Variable trial lengths make final assembly natural, but masks and schedule labels could be more vectorized.

ii. `for start, end in zip(starts, ends):` and `for trial_idx, (start, end) in enumerate(zip(starts, ends)):`

iii. The agent did not explicitly discuss vectorization; its code already vectorizes cell correlations and categorical transforms within each trial.

## 13-c. What processing does the code repeat multiple times?

i. It traverses trials once to build masks/dF/F, again to calculate validation summaries and reward outcomes, and again to construct outputs. It also reads Deconvolved, Fluorescence, and Neuropil lengths before loading those arrays later.

ii. Repeated forms include `for start, end in zip(starts, ends):` in `compute_interneuron_mask` and another loop in `build_session`.

iii. No explicit justification is stated; separation keeps curation, sanity checks, and serialization logically distinct.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes behavior-derived environment medians, inferred observed zones, extensive counters/session summaries, and a full reconstructed dF/F array used only for an interneuron mask. It also builds optional sample data and prints broad summaries; these are not decoder features.

ii. `behavior_env_by_trial.append(...)`, `parsed_zone_matches += ...`, and `dff = np.full_like(f_roi, np.nan, dtype=np.float32)`

iii. The trajectory says these calculations are manuscript-level alignment, curation, and sanity checks; they are validation aids rather than downstream decoder inputs.
