# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively globs every `*_behavior+ophys.nwb` below `/app/data`, naturally sorts the 152 files, and opens each with `h5py`.

ii. `paths = sorted(data_dir.glob("sub-*/*_behavior+ophys.nwb"), key=natural_key)` and `with h5py.File(path, "r") as nwb:`

iii. The trajectory says the full artifact contains all 152 available imaging sessions and that direct HDF5 access was chosen after inspecting the NWB layout.

## 1-b. How are the data split into subjects?

i. Subject IDs are parsed from each path (`sub-m<N>`), unique IDs are numerically sorted, and every session receives the corresponding index.

ii. `subject_names = [f"m{mouse}" for mouse in sorted({natural_key(path)[0] for path in paths})]` and `subject_idx.append(subject_lookup[info["subject"]])`

iii. The agent identified 11 mice and used numeric rather than lexical ordering.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session and becomes one element of each top-level session list.

ii. `for path in paths: neural_session, input_session, output_session, info = convert_session(path)`

iii. The trajectory explicitly describes conversion across all 152 imaging sessions.

## 1-d. How are the data split into trials?

i. Trial starts are positive `trial_start` samples and stops are positive `teleport` samples; each emitted trial is `[start, stop)`, including the start marker and excluding teleport.

ii. `starts = np.flatnonzero(read_behavior(behavior, "trial_start") > 0)`; `stops = np.flatnonzero(read_behavior(behavior, "teleport") > 0)`; `trial_slice = slice(int(start), int(stop))`

iii. The agent states that trials span the trial-start marker through the sample before teleport.

## 1-e. How are trials filtered based on quality controls?

i. A trial is dropped when more than 30% of its frames have raw lick values greater than 2. Sessions with fewer than two retained trials are rejected. No speed filter is applied.

ii. `if np.mean(trial_lick > 2) > LICK_ERROR_FRACTION: ... continue` and `if len(neural_session) < 2: raise ValueError(...)`

iii. The agent calls this the paper-defined corrupt-lick rule, reports exactly 81 removed trials, and retains stopped frames because low speed is a requested output class.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `processing/ophys/Fluorescence` and `Neuropil`, restricted by `ImageSegmentation/PlaneSegmentation/iscell`; stored `Deconvolved` is deliberately not used.

ii. `fluorescence = nwb["processing/ophys/Fluorescence"]`; `neuropil = nwb["processing/ophys/Neuropil"]`; `iscell = np.asarray(segmentation["iscell"][:, 0]) > 0`

iii. The agent found the NWB `Deconvolved` signal was Suite2P-scale and not the paper's trial-wise pipeline.

## 2-b. How is the `neural` data processed?

i. Per trial and plane, the code subtracts `0.7*Fneu`, restores the trial neuropil mean, applies sigma-15 smoothing, 300-sample min/max baseline filters, calculates dF/F, and smooths it with sigma 2. It saves that dF/F directly and does not perform the paper/reference OASIS deconvolution.

ii. `corrected = f - NEUROPIL_COEFFICIENT * fneu`; `baseline = maximum_filter1d(minimum_filter1d(gaussian_filter1d(corrected, 15, axis=1), 300, axis=1), 300, axis=1)`; `dff = gaussian_filter1d((corrected - baseline) / denominator, 2, axis=1)`

iii. The trajectory claims this reconstructs the paper's neural signal, but its earlier plan and the reference require activity extraction/deconvolution after dF/F; that final operation is absent.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only manually curated `iscell` ROIs are processed, then cells with Pearson correlation of dF/F and speed greater than 0.5 are removed session-wide.

ii. `cell_indices = np.flatnonzero(iscell)`; `keep_neurons = speed_correlations <= INTERNEURON_R_THRESHOLD`; `neural_trials = [trial[keep_neurons] for trial in neural_trials]`

iii. The agent cites manual ROI curation and the paper's putative-interneuron criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples are sliced from the trial-start index, making its first column time zero, through but not including teleport.

ii. `dff = paper_dff_trial(..., int(start), int(stop))`

iii. The agent says synchronized behavior/neural frame indexing makes additional alignment unnecessary.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native 15.5078125 Hz sampling is retained (64.4839 ms bins); no rebinning or resampling occurs.

ii. `TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ` and `"time_bin_size": TIME_BIN_MS`

iii. The trajectory explicitly says native synchronized 64.48 ms frames are retained.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the `position` time-series timestamps and `trial_start` indices.

ii. `timestamps = np.asarray(behavior["position"]["timestamps"], dtype=np.float64)`

iii. The agent relied on the common synchronized behavior clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at the start marker is subtracted from every timestamp in that trial.

ii. `time_from_start = timestamps[trial_slice] - timestamps[start]`

iii. This directly implements trial-relative elapsed seconds.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The same `[start, stop)` indices are used for timestamps and neural data, and a sampling-interval check is enforced.

ii. `inputs = np.vstack([time_from_start, ...])` and `if not np.isclose(np.median(frame_intervals), 1.0 / FRAME_RATE_HZ, ...): raise ValueError(...)`

iii. The agent considered the streams already synchronized frame-by-frame.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavior `environment` stream.

ii. `environment = read_behavior(behavior, "environment")`

iii. The stream directly represents ENV1/ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Negative values are ignored, the median valid value is rounded, and that scalar is repeated across the trial.

ii. `env = float(np.rint(np.median(env_values[env_values >= 0])))`; `np.full(dff.shape[1], env)`

iii. The decision treats environment as a per-trial input and defensively handles invalid labels.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is read from the behavior `trial number` stream and cross-checked against the zero-based marker-loop index.

ii. `trial_number_stream = read_behavior(behavior, "trial number")`

iii. The agent uses the recorded variable while validating the expected sequential numbering.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The within-trial median is rounded, checked equal to the loop index, and repeated across every frame.

ii. `source_trial_number = float(np.rint(np.median(trial_number_stream[trial_slice])))`; `np.full(dff.shape[1], source_trial_number)`

iii. This ensures the raw stream and detected boundaries agree.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It derives from `Reward.timestamps`, `position.timestamps`, and trial start/stop boundaries.

ii. `reward_timestamps = np.asarray(behavior["Reward"]["timestamps"], dtype=np.float64)`

iii. Reward events are the direct evidence of trial outcome.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Each trial is rewarded if any reward timestamp lies within its inclusive timestamp interval. Outcomes are shifted by one trial; the first is set to zero and the scalar is repeated in time.

ii. `previous_outcomes = np.r_[0, outcomes[:-1]]`; `np.full(dff.shape[1], previous_outcomes[trial])`

iii. The first pre-imaging outcome is unavailable, so the agent documents zero as its convention.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses the behavior `position` stream plus active zone labels inferred from reward delivery timestamps/positions and fixed A/B/C boundaries.

ii. `ZONE_BOUNDS = np.asarray([[80.,130.],[200.,250.],[320.,370.]])`; `reward_positions = np.interp(block_rewards, timestamps, position)`

iii. The agent infers one zone per pre/post-trial-30 block so omission trials inherit a label.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Distance is signed to the nearest zone edge: negative before, zero inside, positive after.

ii. `distance = np.where(position < zone_start, position-zone_start, np.where(position > zone_stop, position-zone_stop, 0.0))`

iii. The metadata explicitly documents this edge-distance definition.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit masks implement the seven requested classes at -50, -10, 0, 10, and 50 cm, with the whole zone assigned class 3.

ii. `result[distance < -50.0] = 0`; `result[(distance >= -50.0) & (distance < -10.0)] = 1`; `result[distance > 50.0] = 6`

iii. The thresholds follow the task specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data use the identical `[start, stop)` slice.

ii. `trial_position = position[trial_slice]` and `distance_classes(trial_position, int(zones[trial]))`

iii. The synchronized frame streams require no interpolation here.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavior `position` stream.

ii. `position = read_behavior(behavior, "position")`

iii. Position already uses corridor centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Apart from trial slicing, it is directly discretized without smoothing or normalization.

ii. `np.digitize(trial_position, [90.0, 180.0, 270.0, 360.0])`

iii. This preserves the raw spatial coordinate.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.digitize` creates five classes at 90, 180, 270, and 360 cm.

ii. `np.digitize(trial_position, [90.0, 180.0, 270.0, 360.0]).astype(np.int8)`

iii. These are five equal 90-cm bins across the 450-cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is sliced with the same trial indices and consequently has the same number of columns.

ii. `trial_position = position[trial_slice]`

iii. Behavior and fluorescence are stored on the synchronized frame grid.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from the behavior `lick` stream.

ii. `lick = read_behavior(behavior, "lick")`

iii. The raw stream is the direct lick measurement.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Retained trials are binarized as raw lick greater than zero; heavily corrupted trials are removed first.

ii. `(trial_lick > 0).astype(np.int8)`

iii. Binarization implements the required no/yes output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Licks use the identical `[start, stop)` trial slice.

ii. `trial_lick = lick[trial_slice]`

iii. The synchronized behavior grid makes samplewise alignment direct.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is inferred from `Reward.timestamps`, `position` and its timestamps, plus known A/B/C centers.

ii. `median_position = float(np.median(reward_positions))`; `label = int(np.argmin(np.abs(ZONE_CENTERS - median_position)))`

iii. Actual delivery locations identify the active zone, including switch direction.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For trials 0–29 and 30 onward separately, reward positions are interpolated at event times, their median is mapped to the nearest zone center, and the label is repeated over every trial/frame in that block.

ii. `boundaries = [(0, min(SWITCH_TRIAL,n_trials)), ...]`; `labels[first:last] = label`; `np.full(dff.shape[1], zones[trial])`

iii. The agent says block inference is robust and supplies labels for omissions.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It comes from reward event timestamps compared with behavior timestamps and trial bounds.

ii. `reward_timestamps = np.asarray(behavior["Reward"]["timestamps"], dtype=np.float64)`

iii. Delivery events directly encode rewarded versus omitted trials.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is 1 if any event time is between its start and stop timestamps (inclusive), otherwise 0; the value is repeated across time.

ii. `outcomes[trial] = np.any((reward_timestamps >= timestamps[start]) & (reward_timestamps <= timestamps[stop]))`; `np.full(dff.shape[1], outcomes[trial])`

iii. This produces the requested per-trial binary label.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code fails loudly on missing files/labels, malformed start-stop pairs, unexpected sampling, plane labels, trial numbers, or blocks without rewards; guards zero dF/F denominators; filters corrupt-lick trials; and validates at least two usable trials. It does not crop neural/behavior length mismatches like the reference.

ii. `if len(starts) != len(stops) or np.any(stops <= starts): raise ValueError(...)`; `denominator[denominator == 0] = np.finfo(np.float32).eps`

iii. The trajectory emphasizes catching the multi-plane edge case before partial output and validating the completed artifact without warnings.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive work is reading fluorescence/neuropil from 152 large HDF5 files, per-trial Gaussian/min/max filtering, correlation accumulation, and writing the 9.52-GB pickle.

ii. `dff = paper_dff_trial(...)` inside the session trial loop; `pickle.dump(converted, stream, protocol=pickle.HIGHEST_PROTOCOL)`

iii. The trajectory reports the full conversion and subsequent decoder run as long-running operations.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Reward-outcome trial iteration and portions of per-trial assembly could be vectorized, although variable trial lengths make final list construction natural. Plane and session loops are structurally appropriate.

ii. `for trial, (start, stop) in enumerate(zip(starts, stops)):` appears in both `reward_outcomes` and conversion.

iii. No explicit trajectory justification discusses vectorization; the implementation favors streaming to control memory.

## 13-c. What processing does the code repeat multiple times?

i. Trial boundaries are traversed once for reward outcomes and again for conversion; reward timestamp masks are rebuilt per trial; natural-key parsing is repeated during sorting and subject extraction. Per-trial baseline filtering necessarily repeats for every lap.

ii. `reward_outcomes(...)` loops over `zip(starts, stops)`, followed by another `for trial, (start, stop) ...`.

iii. The agent prioritizes memory-bounded streaming and sufficient statistics rather than session-wide duplicate arrays.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes and stores diagnostic block medians, speed correlations (only their threshold mask is retained), and detailed session metadata that the decoder does not consume. Unlike the reference survey, it does not reread every file solely for a preliminary survey.

ii. `block_medians.append(median_position)`; `speed_correlations = correlations_from_sums(...)`; `"median_reward_position_cm_by_block": block_reward_medians`

iii. These diagnostics support validation and provenance, even though decoder training discards them.
