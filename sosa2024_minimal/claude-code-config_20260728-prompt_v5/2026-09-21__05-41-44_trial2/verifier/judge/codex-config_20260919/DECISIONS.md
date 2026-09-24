# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers sorted `sub-*` directories and every `.nwb` beneath them, opens each file with `h5py`, but skips identifiers its scene parser does not recognize as an analysis session.

ii. `subjects_dirs = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('sub-') ...])`; `nwb_files = sorted([f for f in os.listdir(subj_path) if f.endswith('.nwb')])`; `with h5py.File(nwb_path, 'r') as f:`

iii. The trajectory says all 11 available switch-task mice should be used and that scene identifiers distinguish usable sessions from training sessions.

## 1-b. How are the data split into subjects?

i. Each `sub-<mouse>` directory defines a subject; the prefix is removed and a subject is registered when its first valid session is retained.

ii. `subject_name = subj_dir.replace('sub-', '')`; `subject_to_idx[subject_name] = len(subjects)`

iii. The agent observed that directory names map directly to GCAMP mouse identities and found 11 relevant subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session; retained session results are appended as one element of each top-level session list.

ii. `for nwb_file in nwb_files: ... result = load_and_process_session(...)`; `all_neural.append(result['neural'])`

iii. The trajectory treats each file/scene as a session and reports 152 retained sessions.

## 1-d. How are the data split into trials?

i. Starts are frames where `trial_start > 0.5`; each end is the first later frame with `teleport > 0.5`, included via `+1`. Missing teleports fall back to the next start or end of data. Samples are then restricted to positions 0 through 455 cm.

ii. `start_frames = np.where(trial_start_signal > 0.5)[0]`; `ef = future_teleports[0] + 1`; `on_track = (trial_pos >= 0) & (trial_pos <= TRACK_LENGTH + 5)`

iii. The agent reasoned that trial start and teleport implement the paper's lap definition, and added positional trimming to remove teleport/inter-trial samples.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than five on-track samples are dropped; sessions with fewer than two trials, fewer than two valid trials, or no good cells are dropped. Training/unparsed sessions are also skipped.

ii. `if len(on_track_idx) < 5: continue`; `if len(neural_trials) < 2: return None`

iii. The trajectory describes short-trial filtering as an edge-case safeguard and retained very long, slow trials as genuine behavior.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Final neural matrices come directly from NWB `processing/ophys/Deconvolved/plane*/data`. `Fluorescence` and `Neuropil` are separately used only to calculate dF/F for interneuron detection.

ii. `deconv0 = ophys['Deconvolved']['plane0']['data'][:]`; `neural_data = deconvolved_cells[:, good_cells]`

iii. The agent believed the stored deconvolution was the primary analysis signal, while acknowledging uncertainty about whether it represented the paper's custom processing.

## 2-b. How is the `neural` data processed?

i. Planes are concatenated, arrays are truncated to a shared minimum length, `iscell` and interneuron masks are applied, trials are indexed, transposed to neuron-by-time, converted to float32, and nonfinite values replaced by zero. The stored signal itself is not recomputed.

ii. `deconvolved = np.concatenate([deconv0, deconv1], axis=1)`; `trial_neural = neural_data[on_track_idx, :].T`; `np.nan_to_num(...)`

iii. The trajectory cites pooling planes per the paper and truncation as a practical fix for one-sample length mismatches.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs must pass NWB `iscell`; putative interneurons are excluded when paper-like dF/F has Pearson correlation with speed above 0.5.

ii. `cell_mask = iscell == 1`; `r, _ = pearsonr(cell_dff, speed_valid)`; `good_cells = ~is_interneuron`

iii. The agent explicitly followed the paper's manual cell curation and speed-correlation interneuron criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial neural data begins at the first on-track sample within the detected trial-start/teleport interval; no interpolation or explicit resampling is done.

ii. `on_track_idx = extract_on_track_indices(position, ts, te)`; `trial_neural = neural_data[on_track_idx, :].T`

iii. The agent regarded splitting at `trial_start` as trial-start alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native sampling is retained with no rebinning. Metadata is the median adjacent difference of emitted trial timestamps, about 64.5 ms.

ii. `dt = np.diff(trial_inp[0, :])`; `median_dt_ms = np.median(all_dt) * 1000`

iii. The agent intended to preserve the native approximately 15.5 Hz imaging resolution.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It derives from `position/timestamps` at retained on-track indices.

ii. `beh_timestamps = beh['position']['timestamps'][:]`

iii. Timestamps were chosen as the direct timebase shared with behavior and neural rows.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first retained timestamp is subtracted from every retained timestamp, yielding seconds starting at zero.

ii. `trial_times = beh_timestamps[on_track_idx] - beh_timestamps[on_track_idx[0]]`

iii. The agent wanted a continuous trial-relative time input.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Exactly the same `on_track_idx` selects both timestamps and neural rows, so lengths and sample correspondence match.

ii. `neural_data[on_track_idx, :]`; `beh_timestamps[on_track_idx]`

iii. The trajectory says mixed inputs were broadcast into a common `(4, time)` representation.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Environment is inferred from the NWB identifier/scene string, not the raw behavioral `environment` series.

ii. `scene = parse_scene_from_identifier(identifier)`; `env_map = {'Env1': 0, 'Env2': 1}`

iii. The agent found scene identifiers reliable and used them to parse stable, within-environment, and cross-environment sessions.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Regex parsing maps Env1/Env2 to 0/1; switch sessions change at zero-based trial index 30. The scalar is broadcast across time.

ii. `if session_info['is_switch'] and t >= CHANGE_TRIAL`; `inp[1, :] = float(trial_env)`

iii. The trajectory assumed the first 30 trials precede the switch and elected scene parsing over the available signal.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is taken from behavioral `trial number` at each detected start.

ii. `tid = int(trial_number[sf])`; `trial_num = trial_ids[i]`

iii. The agent described this as the session-level trial index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. Negative IDs are rejected; the integer ID is cast to float and broadcast over all trial samples.

ii. `if tid < 0: continue`; `inp[2, :] = float(trial_num)`

iii. Broadcasting was chosen to keep all four inputs in one two-dimensional array.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It derives from NWB Reward event timestamps compared with the preceding detected trial's timestamp interval.

ii. `reward_timestamps = beh['Reward']['timestamps'][:]`; `prev_outcome = is_rewarded[i - 1]`

iii. The agent interpreted actual reward delivery—not omission cause—as the requested outcome.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A trial is rewarded if any reward timestamp falls between its first and last timestamp. Trial zero gets 0; later trials inherit the immediately preceding unfiltered trial's flag, broadcast across time.

ii. `np.any((reward_timestamps >= t_start_time) & (reward_timestamps <= t_end_time))`; `if i == 0: prev_outcome = 0`

iii. The first trial has no predecessor, so the agent deliberately used zero.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw position and reward-zone A/B/C inferred from the scene and fixed switch trial; zone boundaries are hard-coded.

ii. `REWARD_ZONES = {'A': (80.0, 130.0), ...}`; `trial_pos = position[on_track_idx]`

iii. The agent chose identifiers because they directly encode experimental zones.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position is clipped to 0–450 cm. Distance is negative to the near edge before a zone, zero inside it, and positive from the far edge after it.

ii. `position - rz_start`; `0.0  # inside zone`; `position - rz_end`

iii. This implements signed distance to any point in the active zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven explicit masks implement `<-50`, `[-50,-10)`, `[-10,0)`, zero, `(0,10]`, `(10,50]`, and `>50`.

ii. `bins[(distance >= -50) & (distance < -10)] = 1`; `bins[distance == 0] = 3`; `bins[distance > 50] = 6`

iii. The thresholds directly follow the decoder specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data use identical `on_track_idx`; category arrays therefore have one value per neural time bin.

ii. `trial_pos = position[on_track_idx]`; `out[0, :] = discretize_distance_to_reward(trial_pos, rz)`

iii. No separate alignment was considered necessary.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It derives from behavioral `position/data`.

ii. `position = beh['position']['data'][:]`

iii. Position is the direct corridor coordinate.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Retained samples are clipped to 0–450 cm before discretization.

ii. `trial_pos = np.clip(trial_pos, 0, TRACK_LENGTH)`

iii. The agent used clipping to handle small positional noise beyond corridor bounds.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Floor division by 90 cm creates five bins and clipping forces labels into 0–4.

ii. `np.clip(np.floor(position / bin_size).astype(np.int64), 0, 4)`

iii. This follows the specified five equal track bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The same retained indices select position and neural data.

ii. `trial_pos = position[on_track_idx]`; `trial_neural = neural_data[on_track_idx, :].T`

iii. The shared frame indexing was treated as sufficient alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It derives from behavioral `lick/data` at retained indices.

ii. `lick = beh['lick']['data'][:]`; `trial_lick = lick[on_track_idx]`

iii. The agent used the provided per-frame lick count.

## 9-b. What processing is involved in computing `output` *Lick*?

i. If more than 35% of trial frames have count above 2, the entire trial lick trace is zeroed; otherwise positive counts become 1 and all others 0.

ii. `frac_high = np.mean(lick_data > 2)`; `trial_lick = np.zeros_like(trial_lick)`; `(trial_lick > 0).astype(np.int64)`

iii. The trajectory chose the repository's 0.35 threshold over the paper's rounded 30% description, interpreting high counts as sensor failure.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural samples share `on_track_idx` without timestamp validation or interpolation.

ii. `trial_lick = lick[on_track_idx]`; `out[3, :] = trial_lick_binary`

iii. The agent assumed NWB behavioral arrays were frame-aligned.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is inferred from the identifier scene string, rather than raw `reward_zone` observations.

ii. `session_info = get_session_info(scene)`; `zone_label = zones[i]`

iii. The agent found the identifier's location labels easier and more explicit than decoding numeric raw values.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Regex extracts A/B/C; switch sessions change at trial index 30; A/B/C map to 0/1/2 and are broadcast over time.

ii. `REWARD_ZONE_LABELS = {'A': 0, 'B': 1, 'C': 2}`; `out[4, :] = REWARD_ZONE_LABELS[zone_label]`

iii. The trajectory assumes stable sessions retain one zone and switch scenes transition after 30 trials.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It derives from Reward event timestamps and behavioral position timestamps defining trial time bounds.

ii. `reward_timestamps = beh['Reward']['timestamps'][:]`; `t_start_time = beh_timestamps[ts]`

iii. Actual event presence was selected because it captures rewarded versus omitted/nonrewarded outcomes.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Any reward event inside the inclusive trial time interval produces 1, otherwise 0; the result is broadcast over all retained samples.

ii. `is_rewarded[i] = 1`; `out[5, :] = is_rewarded[i]`

iii. The agent expected roughly 85% rewarded trials and verified that rate empirically.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. All streams are truncated to their shortest common length; absent terminal teleports use fallbacks; nonfinite neural values become zero; very short trials and unusable sessions are skipped. A 455 cm buffer and later clipping address position noise.

ii. `n_time = min(...)`; `np.nan_to_num(...)`; `if len(on_track_idx) < 5: continue`

iii. The trajectory specifically encountered one-sample multimodal mismatches and judged common-length truncation safest.

## 13-a. What are the most time-consuming steps of the code?

i. Loading full NWB arrays, per-trial maximin dF/F filtering, cell-by-cell Pearson correlations, per-trial reward searches, and final serialization dominate runtime.

ii. `minimum_filter1d(... size=BASELINE_WINDOW ...)`; `for i in range(n_cells):`; `for i, (ts, te) in enumerate(...)`

iii. The trajectory does not explicitly profile runtime; this is inferred from the implemented operations and full 152-session run.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Cell correlations, reward-event assignment, per-trial zone/environment construction, trial boundary searches, and accumulation of all timestamp differences could be vectorized or use search-sorted indexing.

ii. `for sf in start_frames:`; `for i in range(n_cells):`; `for t in range(n_trials):`; `all_dt.extend(dt.tolist())`

iii. No vectorization justification appears in the trajectory; clarity and incremental debugging appear to have driven the loop-based implementation.

## 13-c. What processing does the code repeat multiple times?

i. Trial boundaries are traversed for dF/F, reward detection, and output extraction; arrays are repeatedly indexed per trial; timestamps are scanned again after conversion to calculate bin size.

ii. `for t_start, t_end in zip(...)`; `for i, (ts, te) in enumerate(...)`; `for i in range(n_trials):`

iii. The trajectory does not justify the repetition; it arose from separate processing stages.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes dF/F solely for interneuron detection, reads unused `plane_idx`, stores unused `valid_trial_indices`, counts unused `n_cells_raw`, and builds session metadata beyond decoder needs. It also loads fluorescence/neuropil although final neural values use stored Deconvolved data.

ii. `plane_idx = ...`; `valid_trial_indices.append(i)`; `n_cells_raw = int(np.sum(cell_mask))`

iii. dF/F was intentionally retained for the paper's interneuron QC; the other discarded intermediates are not discussed in the trajectory.
