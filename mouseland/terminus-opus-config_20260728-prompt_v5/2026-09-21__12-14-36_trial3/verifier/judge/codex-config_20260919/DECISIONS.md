# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads `beh/Imaging_Exp_info.npy` as the master index, deduplicates recording keys, caches each `Beh_<experiment>.npy`, and loads each recording's plane-wise spike file and retinotopy file.

ii. `exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()`; `data = np.load(fn, allow_pickle=True).item()`; `return np.concatenate(data['spks'], 0)`

iii. Its notes say there are 23 experiment types but 89 unique recordings, and that behavioral files repeat recordings; caching avoids repeated behavior-file reads.

## 1-b. How are the data split into subjects?

i. `mname` is the subject identifier. Subjects are added in first-session encounter order and every retained session receives the corresponding integer index.

ii. `mname = result['mname']`; `if mname not in subject_to_idx: subject_to_idx[mname] = len(subjects)`

iii. The agent reports that this recovers the paper's 19 mice.

## 1-c. How are the data split into sessions?

i. A session key is mouse, date, and block. Duplicate appearances under experiment types are removed by a `seen` set, producing 89 sessions.

ii. `key = f"{s['mname']}_{s['datexp']}_{s['blk']}"`; `if key not in seen: ... recordings.append((key, s, exp_type))`

iii. The notes state that the same recording occurs under multiple experiment types and its behavioral content is identical.

## 1-d. How are the data split into trials?

i. It loops over `range(ntrials)` and selects frames whose `ft_trInd` equals that trial, which are in `ft_CorrSpc`, and additionally satisfy `ft_move > 0`. Thus a converted trial contains only moving corridor frames and may be temporally discontinuous.

ii. `trial_mask = (ft_trInd == trial) & corridor_mask & running_mask`; `trial_frames = np.where(trial_mask)[0]`

iii. It says running-only frames match the paper and corridor-only frames restrict the data to the textured 4 m region.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than two retained moving-corridor frames are dropped; sessions with fewer than two retained trials are dropped. It has no long-trial/outlier filter.

ii. `if len(trial_frames) < 2: continue`; `if len(neural_trials) < 2: return None`

iii. The explicit rationale is decoder compatibility (at least two trials per session); the notes do not discuss long stopped trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from each plane's `spks` array in the session neural file. `iarea` is loaded separately for region labels.

ii. `return np.concatenate(data['spks'], 0)`; `return np.load(fn, allow_pickle=True)['iarea']`

iii. The agent identifies `spks` as already-deconvolved Suite2p fluorescence, so no dF/F calculation is needed.

## 2-b. How is the `neural` data processed?

i. Imaging planes are concatenated, moving-corridor columns for each trial are selected, and the result is copied as float32. No temporal smoothing, interpolation, or padding is applied.

ii. `neural = spk[:, trial_frames].astype(np.float32)`

iii. It claims the running-frame filter matches the reference paper and says float32 is retained for decoder compatibility.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are removed. Known `iarea` codes are mapped to four regions, while unknown codes and missing retinotopy are assigned an `unassigned` fifth region. No d-prime filter is used.

ii. `region_idx = np.full(len(iarea), 4, dtype=np.int64)`; `brain_region_idx = ... if iarea is not None else np.full(n_neurons, 4, dtype=np.int64)`

iii. The notes say all neurons should be included because d-prime selection was analysis-specific and the decoder performs PCA internally.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are described as aligned to corridor entry, but the stored columns begin at the first retained moving corridor frame and omit every stationary frame. Variable lengths are retained, with `off_start=0` and `off_end=None`.

ii. `trial_frames = np.where((ft_trInd == trial) & corridor_mask & running_mask)[0]`; `'temporal_alignment_event': 'Trial start (corridor entry)'`

iii. The agent equates corridor/running selection with trial-start alignment and cites the paper's running-only analysis.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native imaging frames are used without rebinning. Session bin duration is the median `ft` difference in seconds and metadata stores the median across sessions, about 315 ms.

ii. `dt = float(np.median(np.diff(ft)) * 86400)`; `median_dt_ms = float(np.median(dt_values) * 1000)`

iii. The README explicitly says native ~3.18 Hz resolution and no temporal rebinning.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from trial `SoundFr`, retained imaging-frame indices, and the median time interval computed from `ft`.

ii. `sound_fr = beh['SoundFr']`; `time_to_cue = ((trial_frames - sound_fr[trial]) * dt).astype(np.float32)`

iii. The mapping table describes it as `SoundFr - frame` in seconds, although the implemented subtraction is the reverse.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Frame-index displacement from the cue is multiplied by the session median frame duration. The implemented value is negative before the cue and positive afterward, i.e. time since cue rather than time to cue.

ii. `time_to_cue = ((trial_frames - sound_fr[trial]) * dt).astype(np.float32)`

iii. Documentation calls the variable time to cue but also states “negative = before cue,” revealing the sign convention the code intentionally used.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at exactly the `trial_frames` used to select neural columns, so array columns align one-to-one (including the same omitted stationary frames).

ii. `neural = spk[:, trial_frames]`; `time_to_cue = ((trial_frames - sound_fr[trial]) * dt)`

iii. The agent's sanity checks reconstruct converted values using the same retained frame indices.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It uses the index entry's `days` field when present, otherwise `sess#`, otherwise zero.

ii. `if 'days' in db_info: return float(db_info['days'])`; `if 'sess#' in db_info: return float(db_info['sess#'])`

iii. The notes describe this as `sess#/days` and interpret 0 as before learning and 1+ as after.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The selected scalar is converted to float and repeated across all retained frames of the trial; no chronological per-mouse count is calculated.

ii. `day_arr = np.full(n_t, training_day, dtype=np.float32)`

iii. The agent treats the raw fields as the training-day label and performs no further derivation.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It uses retained frame indices and median `dt`; it does not use raw `StartFr`. The first moving corridor frame is treated as start.

ii. `time_since_start = ((trial_frames - trial_frames[0]) * dt).astype(np.float32)`

iii. The notes label this “frame - start,” but do not acknowledge replacement of true `StartFr` by the first retained frame.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The first retained frame index is subtracted and the difference is multiplied by median seconds per frame, making the first stored value exactly zero while preserving gaps caused by removed stationary frames.

ii. `((trial_frames - trial_frames[0]) * dt).astype(np.float32)`

iii. It presents this as elapsed time since corridor entry.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Values are computed for the same retained frame indices as neural columns, giving one input value per neural column.

ii. `neural = spk[:, trial_frames]`; `time_since_start = ((trial_frames - trial_frames[0]) * dt)`

iii. The agent consistently uses the shared frame selection for all time-varying streams.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes directly from the per-trial `isRew` field.

ii. `is_rew = beh['isRew']`

iii. The mapping notes identify `isRew` as the binary source.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The trial value is cast to float and broadcast over every retained time point.

ii. `reward_arr = np.full(n_t, float(is_rew[trial]), dtype=np.float32)`

iii. It is documented as 1 for rewarded corridor and 0 otherwise.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from per-trial `WallName`.

ii. `wall_name = beh['WallName']`; `stim_idx = STIM_TO_IDX.get(str(wall_name[trial]), 0)`

iii. The notes identify `WallName` as the source.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Fifteen distinct raw wall names are sorted, globally indexed, and broadcast over the trial. Unknown names silently become category 0. Variants are not collapsed into circle/leaf/rock/wood.

ii. `ALL_STIMULI = sorted([...])`; `stim_arr = np.full(n_t, stim_idx, dtype=np.int64)`

iii. The agent explicitly chose “15 stimulus categories mapped globally,” evidently treating each crop/swap name as a distinct visual category.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It uses `LickFr` and `LickTrind`.

ii. `lick_fr = beh['LickFr']`; `lick_trind = beh['LickTrind']`

iii. The notes say licking is converted from lick frames to binary values.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Licks are grouped by trial. For each lick it finds the nearest retained moving frame and marks it only when the distance is less than one frame; all other retained bins are zero.

ii. `closest = np.argmin(np.abs(trial_frames - lf))`; `if diffs[closest] < 1.0: lick_binary[closest] = 1`

iii. The intended result is a binary time-varying output; no separate rationale for nearest-frame handling is documented.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The binary vector has one element for each retained neural frame, and licks are mapped by proximity to those frame indices.

ii. `lick_binary = np.zeros(n_t, dtype=np.int64)`; `out = np.stack([stim_arr, lick_binary, ...])`

iii. The shared `n_t` and `trial_frames` grid is used to maintain shape alignment.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from framewise `ft_Pos` at the retained trial frames.

ii. `pos = ft_Pos[trial_frames]`

iii. The notes describe `ft_Pos / 10` as the position mapping.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is divided by 10, floored to an integer, and clipped to category 0–3.

ii. `pos_binned = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)`

iii. The agent interprets the source units as decimeters and the requested bins as 1 m each.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are 0, 10, 20, 30, and 40 source units, corresponding to 0–1, 1–2, 2–3, and 3–4 m; clipping absorbs out-of-range values.

ii. `np.clip(np.floor(pos / 10.0), 0, 3)`

iii. It chose equal physical-length bins exactly as requested.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is indexed by the identical `trial_frames` used for neural columns.

ii. `neural = spk[:, trial_frames]`; `pos = ft_Pos[trial_frames]`

iii. Its sanity check directly verifies converted position against source positions at those frames.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from framewise `ft_RunSpeed`; `ft_CorrSpc` and `ft_move` also determine which speeds enter global threshold calculation and conversion.

ii. `corridor_speeds = beh['ft_RunSpeed'][:n][running_corridor]`; `speed = ft_RunSpeed[trial_frames]`

iii. The notes say quartiles are computed on running corridor frames so the source population matches retained data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. All retained running-corridor speeds across all recordings are concatenated, three percentile value thresholds are computed, and session/trial speeds are digitized with those global thresholds.

ii. `bins = np.percentile(all_speeds, [25, 50, 75])`; `speed_binned = np.clip(np.digitize(speed, speed_bins), 0, 3)`

iii. The agent revised this computation after observing badly imbalanced bins, aiming for roughly 25% of retained data in each category.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Categories are defined by global 25th, 50th, and 75th percentile speed values and `np.digitize`; ties at a threshold go into the higher bin.

ii. `np.percentile(all_speeds, [25, 50, 75])`; `np.digitize(speed, speed_bins)`

iii. It chose data-wide quartile boundaries to satisfy “each corresponding to 25% of the data.”

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speeds are selected at exactly the neural `trial_frames`, then categorized.

ii. `neural = spk[:, trial_frames]`; `speed = ft_RunSpeed[trial_frames]`

iii. The same retained-frame grid is used for every time-varying stream.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. It truncates behavior to `min(neural frames, len(ft))`, safely handles a missing behavior lookup by skipping the session, assigns missing retinotopy to `unassigned`, skips short trials/sessions, and maps unknown stimuli to category 0. It suppresses all warnings globally.

ii. `n_use = min(n_frames, len(beh['ft']))`; `if beh is None: return None`; `STIM_TO_IDX.get(..., 0)`

iii. The notes highlight the behavior/neural off-by-one truncation and call it an edge case handled safely; they report no validation warnings.

## 12-a. What are the most time-consuming steps of the code?

i. Loading and concatenating very large spike files, copying trial slices into float32, holding/saving the resulting 161.65 GB pickle, and full decoder training dominate. Speed collection scans all behavioral recordings but is comparatively small.

ii. `spk = load_spk(...)`; `neural = spk[:, trial_frames].astype(np.float32)`; `pickle.dump(data, f, protocol=4)`

iii. The script explicitly times spike loading and per-session processing; the notes report a 161.65 GB output and a full training run.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-trial full-array masking repeatedly scans `ft_trInd`; lick grouping and per-lick nearest-frame searches are Python loops; session assembly is necessarily coarse-grained but could be streamed.

ii. `for trial in range(ntrials): trial_mask = (ft_trInd == trial) ...`; `for li in range(len(lick_fr))`; `for lf in lick_by_trial[trial]`

iii. The code calls itself efficient and precomputes lick lookups/masks, but its documentation does not identify remaining vectorization opportunities.

## 12-c. What processing does the code repeat multiple times?

i. It scans each behavior dataset once for global speed thresholds and again during conversion; per trial it scans the entire session trial-index array; spike data are copied once per trial. Behavior caches are periodically cleared, so shared behavior files can also be reloaded.

ii. `speed_bins = compute_running_speed_quartiles(all_recordings)` followed by `process_session(...)`; `if (i + 1) % 10 == 0: _beh_cache.clear()`

iii. The agent justifies the first pass as necessary for global balanced speed thresholds and caching as a memory-control measure.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It retains unassigned neurons and float32 precision, greatly enlarging the pickle; computes/stores detailed timing and session metadata not used by training; and optionally builds processing plots. Global speed computation also processes all recordings even in sample mode.

ii. `BRAIN_REGION_NAMES = [..., 'unassigned']`; `.astype(np.float32)`; `speed_bins = compute_running_speed_quartiles(all_recordings)` before the sample branch; `if show_processing: create_processing_plots(data)`

iii. It justifies all-neuron retention by PCA and float32 by decoder compatibility, while plots are presented as validation aids rather than decoder inputs.
