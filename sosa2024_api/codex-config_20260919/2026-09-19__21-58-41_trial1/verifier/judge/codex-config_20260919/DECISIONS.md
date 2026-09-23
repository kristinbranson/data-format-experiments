# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively discovers every NWB one directory below `/app/data`, sorts the paths, and opens each file with `pynwb.NWBHDF5IO`. Full mode processes all 152 files; sample mode is optional.

ii. `files = sorted(DATA_ROOT.glob("sub-*/*.nwb"))` and `with NWBHDF5IO(str(path), "r", load_namespaces=True) as io: nwb = io.read()`.

iii. The notes justify this from an all-file audit: 11 subject directories, 152 assets, and successful `pynwb` opening of every file, while explicitly honoring the no-`h5py` requirement.

## 1-b. How are the data split into subjects?

i. Subject IDs are read from `nwb.subject.subject_id`; the top-level subject list is derived from parent directory names, sorted, and mapped to each session through `subject_idx`.

ii. `subject = nwb.subject.subject_id`; `subjects = sorted({p.parent.name.removeprefix("sub-") for p in files})`; `subject_idx.append(subject_lookup[result["info"]["subject"]])`.

iii. The agent found 11 `sub-*` directories and checked this against the 11-mouse cohort.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session and produces one element of each session-level list.

ii. `for session_i, path in enumerate(files): result, diagnostics = process_session(path, ...)`; `neural.append(result["neural"])`.

iii. The notes report 152 unique source files/session records and reconcile the per-subject counts with the experiment days.

## 1-d. How are the data split into trials?

i. Trial starts are positive `trial_start` samples and stops are positive `teleport` samples. The agent validates equal counts and strict alternation, then slices half-open intervals `[start, stop)`, excluding teleport.

ii. `starts = np.flatnonzero(np.asarray(behavior["trial_start"].data[:]) > 0)`; `stops = np.flatnonzero(np.asarray(behavior["teleport"].data[:]) > 0)`; `events[:, start:stop]`.

iii. It inspected all files, found 12,216 paired events, and argues that `[start, teleport)` matches the reference’s track interval and avoids a trailing false `trial number` fragment.

## 1-e. How are trials filtered based on quality controls?

i. A trial is dropped if more than 30% of its frames have raw cumulative lick count greater than 2. No duration filter is applied; all remaining trials are retained and each session must retain at least two.

ii. `bad_lick = np.asarray([np.mean(lick[a:z] > 2) > LICK_ERROR_FRACTION for a, z in zip(starts, stops)])`; `if bad_lick[trial_i]: continue`.

iii. The notes link this to the paper’s stuck-lick-sensor QC and report exactly 81 exclusions. Long trials are retained because variable duration is expected.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. It derives neural activity from the NWB `Fluorescence` and `Neuropil` ROI response series, not the stored `Deconvolved` series, and uses the `ImageSegmentation` `iscell` column for curation.

ii. `f_series = ophys["Fluorescence"].roi_response_series[plane]`; `fn_series = ophys["Neuropil"].roi_response_series[plane]`; `iscell = iscell_table[:, 0] > 0`.

iii. The agent determined that stored deconvolution is Suite2p output rather than the paper’s analyzed signal, so it recomputes the manuscript pipeline from F and Fneu.

## 2-b. How is the `neural` data processed?

i. Per plane and trial it subtracts `0.7*Fneu`, restores the trial mean neuropil, computes a sigma-15/300-sample min-then-max baseline, forms dF/F, smooths with sigma 2, and OASIS-deconvolves at 15.5078125 Hz with tau 0.7. Planes are concatenated across neurons. Unlike the human reference, baseline/deconvolution never span teleport intervals on designated sessions.

ii. `corrected = fluorescence[:, start:stop] - NEUROPIL_COEF * fneu`; `baseline = ndimage.maximum_filter1d(ndimage.minimum_filter1d(smooth, 300, axis=-1), 300, axis=-1)`; `dcnv.oasis(dff[:, start:stop], ..., tau=OASIS_TAU_S, fs=FRAME_RATE)`.

iii. The notes cite the paper’s dF/F parameters and explain why recomputation is necessary, but do not acknowledge the reference solution’s `teleport_sessions` exception.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It keeps manually curated `iscell` ROIs, then removes cells with non-finite dF/F-speed correlation or correlation greater than 0.5.

ii. `local_cells = np.flatnonzero(iscell[roi_indices])`; `keep = np.isfinite(speed_corr) & (speed_corr <= INTERNEURON_R_THRESHOLD)`.

iii. This is justified as matching manual Suite2p curation and the paper’s putative-interneuron exclusion; 402 cells were excluded by correlation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural rows are assumed already synchronized to behavioral imaging frames and are sliced at the same `trial_start` index, so column zero is the alignment event.

ii. `neural_trials.append(np.ascontiguousarray(events[:, start:stop], dtype=np.float32))`.

iii. The agent audited identical row clocks, checked plots and raw-to-converted trials, and found no frame shift.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native 15.5078125 Hz samples are retained: 64.483627 ms per bin, with no rebinning or resampling.

ii. `FRAME_RATE = 15.5078125`; `TIME_BIN_MS = 1000.0 / FRAME_RATE`; `"time_bin_size": TIME_BIN_MS`.

iii. Behavior timestamps and per-plane cadence were audited; dual-plane scanner metadata was correctly interpreted as interleaved acquisition rather than doubled per-plane temporal resolution.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It uses timestamps from the raw `position` BehavioralTimeSeries.

ii. `timestamps = np.asarray(behavior["position"].timestamps[:], dtype=np.float64)`.

iii. Position timestamps were found to be the common behavior/imaging-volume clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp of each trial is subtracted from every timestamp in that trial and the result is stored as float32.

ii. `timestamps[start:stop] - timestamps[start]`.

iii. This directly implements time relative to trial start; checks confirmed zero at the first sample and constant 64.483627-ms increments.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The timestamp slice and neural slice use identical `[start:stop)` indices and length `T`.

ii. `T = stop - start`; `input_trial = np.vstack((timestamps[start:stop] - timestamps[start], ...))`; `events[:, start:stop]`.

iii. Session validation enforces equal neural/input/output lengths, and independent comparisons found no offset.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the `environment` BehavioralTimeSeries.

ii. `environment = np.asarray(behavior["environment"].data[:], dtype=np.float64)`.

iii. The agent identified this as synchronized ENV1/ENV2 identity.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Negative values are discarded when determining the trial’s unique value; exactly one value must remain, it is rounded to integer, validated as 0 or 1, and repeated across the trial.

ii. `env_values = env_values[env_values >= 0]`; `env = int(round(float(env_values[0])))`; `np.full(T, env)`.

iii. Constant environment per retained trial is explicitly asserted and full-data class fractions were checked.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It uses the zero-based ordinal of paired `trial_start`/`teleport` events, not the raw `trial number` series.

ii. `for trial_i, (start, stop) in enumerate(zip(starts, stops))`; `np.full(T, trial_i)`.

iii. The raw series had a known trailing false value; event ordinals were judged more reliable and preserve gaps when QC removes trials.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transformation beyond repeating the zero-based source ordinal across all timepoints.

ii. `np.full(T, trial_i)`.

iii. The notes explicitly describe zero-based source trial number and verify retained trials preserve original ordinals.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from sparse timestamps in the `Reward` BehavioralTimeSeries and the previous source trial’s timestamp interval.

ii. `reward_times = np.asarray(behavior["Reward"].timestamps[:], dtype=np.float64)`; `outcomes = np.asarray([np.any((reward_times >= timestamps[a]) & (reward_times < timestamps[z])) ...])`.

iii. Sparse reward events were identified as the authoritative delivery record; reward rate was reconciled with the paper.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Each raw trial is rewarded if any event lies in `[start time, stop time)`. Trial zero gets 0; later trials receive the immediately preceding raw trial’s outcome, even if that preceding trial is excluded by lick QC, repeated over time.

ii. `previous_outcome = int(outcomes[trial_i - 1]) if trial_i > 0 else 0`; `np.full(T, previous_outcome)`.

iii. This prevents QC from changing the experimental meaning of “previous trial” and implements omitted/rewarded binary coding.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw `position` plus the active reward-zone label parsed from the NWB scene/identifier. Zone coordinates are A 80–130, B 200–250, and C 320–370 cm; switching sessions change after source trial 30.

ii. `initial_zone, switched_zone = parse_scene_zones(scene)`; `zone_label = initial_zone if switched_zone is None or trial_i < 30 else switched_zone`; `REWARD_ZONES[zone_label]`.

iii. The agent traced the coordinates and switch rule to the paper code, validated every scene string, and checked all switch sessions.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is position minus the near edge before a zone, zero inside it, and position minus the far edge after it; this continuous value is then categorized.

ii. `distance = np.where(position < start, position - start, np.where(position > stop, position - stop, 0.0))`.

iii. This is justified as distance to the nearest point in the active 50-cm interval, matching the requested exact zero within the reward zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit masks create seven classes: `<-50`, `[-50,-10)`, `[-10,0)`, exactly 0, `(0,10]`, `(10,50]`, and `>50`.

ii. `classes[(distance > 0) & (distance <= 10)] = 4`; `classes[(distance > 10) & (distance <= 50)] = 5`.

iii. The agent says the masks implement the requested bins and validates that no value remains unclassified. Its +10/+50 boundary convention differs from the human code’s `np.digitize` convention.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural activity are sliced with the same `[start:stop)` indices.

ii. `pos = position[start:stop]`; `events[:, start:stop]`.

iii. Equal lengths, raw comparisons, and processing plots were used to verify alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It comes directly from `behavior["position"].data`.

ii. `position = np.asarray(behavior["position"].data[:], dtype=np.float64)`; `pos = position[start:stop]`.

iii. The NWB description and observed 0–450-cm progression identify this as corridor position.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No continuous transformation is applied; each per-trial position sample is directly categorized.

ii. `position_classes(pos)`.

iii. The agent retains native synchronized positions because the requested decoder is temporal, not spatially rebinned.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five explicit masks use boundaries 90, 180, 270, and 360 cm. The agent assigns exactly 180 to class 1, exactly 270 to class 2, and exactly 360 to class 3.

ii. `classes[(position >= 90) & (position <= 180)] = 1`; `classes[(position > 180) & (position <= 270)] = 2`.

iii. It interprets the textual ranges as including their named upper endpoint. This differs at exact internal boundaries from the human `np.digitize` implementation.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Both use the same trial indices with no interpolation.

ii. `pos = position[start:stop]`; `neural_trials.append(... events[:, start:stop])`.

iii. Native behavioral and imaging rows were audited as synchronized and validation enforces matching lengths.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the raw `lick` BehavioralTimeSeries.

ii. `lick = np.asarray(behavior["lick"].data[:], dtype=np.float64)`.

iii. Inspection showed that it contains cumulative lick counts per imaging frame rather than an already binary flag.

## 9-b. What processing is involved in computing `output` *Lick*?

i. After bad-sensor trial exclusion, any positive count is 1 and all others are 0.

ii. `(lick[start:stop] > 0).astype(np.int8)`.

iii. This matches the requested binary output and the reference practice of capping positive counts.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural arrays are sliced by the same event indices.

ii. `(lick[start:stop] > 0)` and `events[:, start:stop]`.

iii. The common imaging-frame clock and plots/checks support direct index alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the NWB identifier’s scene string and the source trial ordinal, not the raw `reward_zone` stream.

ii. `scene = nwb.identifier.rstrip("/").split("/")[-1]`; `initial_zone, switched_zone = parse_scene_zones(scene)`.

iii. The agent found scene metadata to encode the experimental zone schedule exactly and validated parsing across all sessions.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene names are regex-parsed for initial/optional switched zones; the switched zone applies from trial index 30, labels A/B/C map to 0/1/2, and the value is repeated over the trial.

ii. `re.search(r"Location([ABC])(?:_to_([ABC]))?$", scene)`; `ZONE_CODES = {"A": 0, "B": 1, "C": 2}`; `np.full(T, ZONE_CODES[zone_label])`.

iii. This follows the paper’s known switch schedule and avoids noisy/missing frame-level reward-zone events.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It uses sparse timestamps in `behavior["Reward"]` plus trial start/stop timestamps.

ii. `reward_times = np.asarray(behavior["Reward"].timestamps[:], dtype=np.float64)`.

iii. Reward deliveries are sparse events and were checked against the expected approximately 85% rate.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is 1 if any reward timestamp lies in its half-open temporal interval, otherwise 0; this scalar is repeated over all its samples.

ii. `np.any((reward_times >= timestamps[a]) & (reward_times < timestamps[z]))`; `np.full(T, outcomes[trial_i])`.

iii. The half-open interval matches trial slicing and prevents out-of-trial reward events from being assigned.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code fails loudly on inconsistent event counts, series lengths, clocks, environments, nonfinite values, or neural under-runs. It safely ignores ten extra neural tail rows by loading only `:n_behavior`, avoids a false trial-number fragment by event pairing, excludes bad-lick trials, and rejects nonfinite-correlation neurons. It does not impute missing data or crop a neural under-run.

ii. `if f_series.data.shape[0] < n_behavior ...: raise ValueError`; `f_series.data[:n_behavior, local_cells]`; `if not np.allclose(dt, 1.0 / FRAME_RATE, ...): raise ValueError`.

iii. The notes document an all-session edge audit and independent finite/shape checks, explaining each known anomaly and its treatment.

## 13-a. What are the most time-consuming steps of the code?

i. Reading large F/Fneu arrays, per-trial maximin filtering and OASIS across all curated cells, retaining the full dataset in memory, and serializing the roughly 8.9-GiB pickle dominate. Optional plotting adds work only for two sessions.

ii. `reference_dff(...)`; `reference_events(...)`; `pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)`.

iii. The agent measured 350.08 seconds total and identified raw fluorescence/neuropil plus OASIS as the necessary heavy path.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial loops in `reference_dff`, `reference_events`, `trace_processing`, outcome/QC construction, and trial packaging could partly be replaced with masks or whole-session precomputations. Variable trial boundaries and required per-trial baselines/deconvolution limit useful vectorization; neuron correlations already are vectorized.

ii. `for start, stop in zip(starts, stops): ...`; `for trial_i, (start, stop) in enumerate(zip(starts, stops)): ...`.

iii. The notes emphasize vectorization across neurons and correlations, sequential planes for memory control, and the necessity of per-trial operations.

## 13-c. What processing does the code repeat multiple times?

i. It iterates over trial boundaries separately for dF/F, valid-mask construction, OASIS, outcomes, bad-lick QC, and final packaging. With diagnostics it recomputes correction/baseline for one trace and plotting again slices/derives trial views. Dataset statistics make another full pass over outputs.

ii. Repeated constructs include `for start, stop in zip(starts, stops)` in `reference_dff`, `reference_events`, and `process_plane`, plus `for trial_i, ...` in `process_session`.

iii. The agent justifies diagnostics as optional validation and reports that normal full conversion avoids collecting them.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In normal full mode, signed continuous distances are computed although only classes are saved; detailed per-plane diagnostics dictionaries (including all speed correlations) are created then discarded; `kept_ordinals` and several diagnostics arrays are built even when plots are disabled. Per-session `gc.collect()` also adds overhead. With `--show-processing`, trace intermediates and figures are intentionally diagnostic and not decoder inputs.

ii. `signed_distance, dist_class = distance_classes(...)`; `diagnostics = {"speed_corr": speed_corr, ...}`; `kept_ordinals.append(trial_i)`; `del diagnostics; gc.collect()`.

iii. These costs support auditing/plotting, but the agent’s notes do not explicitly identify them as discarded; they are small relative to neural processing except for optional figures.
