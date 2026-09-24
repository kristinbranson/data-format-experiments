# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates locally present NWB files, intersects their IDs with the AllenSDK experiment table, excludes passive experiments, and loads each file directly as a `BehaviorOphysExperiment`.

ii. `for path in sorted(nwb_root.glob("behavior_ophys_experiment_*.nwb")):`; `experiments = experiments.loc[experiments.index.intersection(local_paths.keys())]`; `experiments = experiments.loc[~experiments["passive"]]`; `dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])`.

iii. The trajectory says the local cache was incomplete, so it deliberately switched to “available NWB files only.” It also chose active sessions and direct local loading to avoid attempting unavailable release files.

## 1-b. How are the data split into subjects?

i. Subjects are unique string-valued `mouse_id`s encountered among successfully retained experiments, with a lookup used to create per-session `subject_idx`.

ii. `subject = str(row["mouse_id"])`; `subject_to_idx[subject] = len(subjects)`; `subject_idx.append(subject_to_idx[subject])`.

iii. The AI treated `mouse_id` as the SDK animal identifier and verified subject/session counts in the saved artifact.

## 1-c. How are the data split into sessions?

i. Each NWB/`ophys_experiment_id` (one imaging plane) becomes a separate decoder session; experiments sharing an `ophys_session_id` are not merged.

ii. `for candidate_number, (experiment_id, row) in enumerate(experiments.iterrows(), start=1):`; `neural_sessions.append(neural_trials)`; metadata separately records `ophys_session_id`.

iii. The trajectory explicitly claims experiment-plane sessions are the “right unit for multiscope data” because one behavior session may have multiple planes. This differs from the reference grouping of planes by `ophys_session_id`.

## 1-d. How are the data split into trials?

i. The SDK `trials` table defines trials. Each retained trial spans `start_time` to `stop_time`, sampled at centers of fixed 100 ms bins by default.

ii. `target_times = make_target_times(start_time, stop_time, bin_size_s)` where `centers = edges + (0.5 * bin_size_s)`.

iii. The trajectory identified the SDK trial table as the behavior-defined source and chose full trial windows so time-varying outputs could be aligned throughout each trial.

## 1-e. How are trials filtered based on quality controls?

i. It retains go or catch trials and removes aborted and auto-rewarded trials. Trials with fewer than two target bins, missing behavior interpolation, or no recognized outcome are skipped; sessions with fewer than two kept trials are excluded.

ii. `trial_mask = ((trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"]))`; `if outcome_idx is None: continue`; `if len(neural_trials) < 2: ... continue`.

iii. The trajectory states that `go`/`catch` are the defined trial types and that aborted/auto-rewarded labels permit the requested exclusions. The two-trial rule follows decoder requirements.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the AllenSDK `events` table’s discrete calcium event traces.

ii. `events_df = dataset.events`; `neural_full = np.vstack(events_df["events"].to_numpy()).astype(np.float32)`.

iii. The trajectory investigated dF/F versus events and ultimately described the chosen signal as “AllenSDK discrete calcium events,” believing this matched the paper’s neural signal.

## 2-b. How is the `neural` data processed?

i. Event traces are stacked neuron-by-time, then nearest native ophys samples are selected at fixed 100 ms target-bin centers; no averaging, normalization, or smoothing is performed.

ii. `nearest = nearest_indices(ophys_timestamps, target_times)`; `neural_trial = neural_full[:, nearest].astype(np.float32, copy=False)`.

iii. The AI chose a common temporal grid to satisfy its interpretation of the same-bin-size requirement and used nearest sampling to retain event values.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It relies on AllenSDK’s valid-ROI filtering, rejects experiments with no events, but applies no additional cell-level filter.

ii. `if len(events_df) == 0: ... continue`; otherwise all rows of `events_df["events"]` are stacked.

iii. The trajectory explicitly confirmed that the SDK excludes invalid ROIs and regarded that as the appropriate baseline QC.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to SDK trial start, with target times from `start_time` to `stop_time`; nearest ophys frames supply neural values.

ii. `start_time = float(trial_row["start_time"])`; `target_times = make_target_times(start_time, stop_time, bin_size_s)`; `nearest_indices(ophys_timestamps, target_times)`.

iii. The metadata and final trajectory call this “Allen trial start alignment”; ophys timestamps are used as required.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The default resolution is 100 ms. A new regular grid is made, but neural values are nearest-sampled rather than aggregated within bins.

ii. `TIME_BIN_MS_DEFAULT = 100.0`; `bin_size_s = time_bin_ms / 1000.0`; metadata stores `"time_bin_size": float(time_bin_ms)`.

iii. The AI intentionally chose a “common 100 ms bin size” for decoder compatibility, despite native ophys frames already being approximately 90 ms apart.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It derives identity from `stimulus_presentations` rows in the `change_detection` block, using `trials_id`, `image_name`, `start_time`, `end_time`, and `omitted`.

ii. `trial_stim = change_detection.loc[change_detection["trials_id"] == trial_id]`; conditions use `stim_row.image_name`, `stim_row.start_time`, `stim_row.end_time`, and `stim_row.omitted`.

iii. The trajectory observed that stimulus rows are already linked to trials and provide exact flashed-image intervals.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Every bin starts as category 0 (`gray`). Non-omitted image intervals receive globally assigned integer image codes; omitted periods remain gray.

ii. `image_labels = np.full(..., image_to_idx[GRAY_LABEL])`; `if bool(stim_row.omitted) ...: continue`; `image_to_idx[stim_row.image_name] = len(image_values)`.

iii. The AI wanted identity only during non-gray flashes, consistent with the requested “image presented during the non-grey screen,” and used the presentation table rather than a single pre/post label.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Stimulus interval masks are evaluated at the same `target_times` used to nearest-sample neural activity.

ii. `mask = (target_times >= float(stim_row.start_time)) & (target_times < float(stim_row.end_time))`; `image_labels[mask] = image_idx`.

iii. The trajectory selected timestamp-based alignment because stimulus presentations contain synchronized start/end times.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change comes from the `is_change` field of each linked change-detection stimulus-presentation row, along with its start/end times.

ii. `if bool(stim_row.is_change): change_labels[mask] = 1`.

iii. The trajectory specifically notes that the changed stimulus row is marked by `is_change`.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector is created and bins falling inside a changed image’s presentation interval are set to one; omitted presentations are skipped.

ii. `change_labels = np.zeros(..., dtype=np.int16)` followed by `change_labels[mask] = 1`.

iii. The AI interpreted “right after a change” as the changed image flash itself, using the SDK’s exact interval.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is inherently binary: 0 outside a changed-image interval and 1 inside it; no numeric threshold is learned.

ii. `change_labels = np.zeros(...)`; `change_labels[mask] = 1`; output values are `["no_change", "change"]`.

iii. This directly implements the required binary output.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The same target-time mask used for image identity defines change labels, so its columns correspond one-to-one with neural columns.

ii. `mask = (target_times >= ...start_time) & (target_times < ...end_time)`; both arrays have `target_times.shape[0]` columns.

iii. The AI relied on shared synchronized timestamps for all time-varying signals.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses `dataset.running_speed["timestamps"]` and `dataset.running_speed["speed"]`.

ii. `running_df = dataset.running_speed`; later `interp_signal(running_df["timestamps"], running_df["speed"], target_times)`.

iii. The trajectory treated the SDK processed running-speed stream as the standard source.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Finite values are sorted/deduplicated and linearly interpolated directly to each trial’s target times, with endpoint extrapolation held constant. Global 20/40/60/80 percentiles are then computed.

ii. `interp = np.interp(target_times, times, values, left=values[0], right=values[-1])`; `running_edges = np.percentile(running_values, [20, 40, 60, 80])`.

iii. The AI chose interpolation for hardware-synchronized streams and global quintiles for consistent, approximately balanced classes.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Values are digitized at four global percentile cutoffs into integer categories 0–4.

ii. `np.digitize(values, edges, right=False).astype(np.int16)`; values are named `bin_0` through `bin_4`.

iii. Five equal-percentile bins were explicitly required; global edges preserve category meanings across sessions.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is linearly interpolated to the identical trial `target_times` used for neural nearest sampling.

ii. `running_trial = interp_signal(..., target_times)` and `neural_trial = neural_full[:, nearest]` where `nearest` is computed from those target times.

iii. The AI justified interpolation based on synchronized clocks and matching output/neural time axes.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses eye-tracking `pupil_area` and `timestamps`, converting area to equivalent circular diameter.

ii. `pupil_diameter = 2.0 * np.sqrt(np.asarray(eye_df["pupil_area"]) / np.pi)`.

iii. The trajectory inspected SDK circular-area processing and chose an equivalent diameter derived from area.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Equivalent diameter is calculated, finite samples are linearly interpolated to trial target times with held endpoints, and global quintile edges are computed. Blink flags are not used explicitly.

ii. `pupil_trial = interp_signal(eye_df["timestamps"].to_numpy(...), pupil_diameter, target_times)`; `pupil_edges = np.percentile(pupil_values, [20, 40, 60, 80])`.

iii. The AI considered area-derived diameter a standard geometric conversion and used the same alignment/discretization design as running speed.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It digitizes diameter at four global percentile edges into classes 0–4.

ii. `pupil_bins = remap_to_bins(trial["pupil"], pupil_edges)`.

iii. This implements the requested five equal-percentile categories consistently across experiments.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated to the same trial target times used to select neural samples.

ii. `pupil_trial = interp_signal(..., target_times)`.

iii. The AI used timestamp interpolation because eye and ophys clocks are synchronized.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It derives outcome from trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii. `if bool(trial_row["hit"]): return 0` ... through `correct_reject`.

iii. The AI recognized these as the SDK’s canonical mutually exclusive outcomes for retained trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Outcomes map to fixed codes 0–3 in hit/miss/false-alarm/correct-reject order; unrecognized trials are dropped, and the code is repeated across every time bin.

ii. `outcome = np.full(t, outcome_idx, dtype=np.int16)`; output values list `['hit', 'miss', 'false_alarm', 'correct_reject']`.

iii. Repetition makes the static per-trial label compatible with the common 2D output representation.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Load failures and experiments lacking timestamps, events, running, pupil, or enough trials are logged and skipped. Interpolation removes nonfinite source samples and holds boundary values; short/invalid trials and unknown outcomes are skipped. No-valid-pupil sessions are excluded.

ii. Examples include `except Exception as exc: ... continue`, `valid = np.isfinite(source_times) & np.isfinite(source_values)`, and `if len(neural_trials) < 2: ... continue`.

iii. The trajectory emphasizes robustness to the partial local cache and records exclusion reasons so a bad experiment does not abort conversion.

## 9-a. What are the most time-consuming steps of the code?

i. Repeated NWB parsing/loading and extraction of full events, behavior, eye, stimulus, and trial tables dominate; decoder training was also lengthy but is outside conversion.

ii. `dataset = BehaviorOphysExperiment.from_nwb_path(row["local_path"])` inside the experiment loop.

iii. The trajectory repeatedly waited on NWB/cache parsing and described these loads as the practical bottleneck.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The experiment loop is I/O-bound, but trial iteration, per-trial stimulus-row iteration, final per-trial binning, and Python image-map growth could be partly vectorized/grouped.

ii. `for trial_id, trial_row in trials.iterrows():`; `for stim_row in trial_stim.itertuples():`; nested loops over `output_temp_sessions` and trials.

iii. The trajectory favored clear explicit alignment loops and did not claim vectorization; most optimization effort went to successful loading and validation.

## 9-c. What processing does the code repeat multiple times?

i. Running timestamp/speed arrays are converted to NumPy for every trial, each trial filters the full stimulus table by `trials_id`, and output trials are traversed again after global percentile computation.

ii. Inside the trial loop: `running_df["timestamps"].to_numpy(...)`, `running_df["speed"].to_numpy(...)`, and `change_detection.loc[change_detection["trials_id"] == trial_id]`.

iii. This repetition follows the two-stage design: retain continuous trial values first, then apply dataset-global thresholds after all sessions are known.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It performs a one-point running interpolation only to test signal availability, stores zero-row input arrays for every trial, records extensive metadata, and accumulates `native_dt_ms` entries even for experiments later excluded. Some sorting/copying is also redundant.

ii. `running_interp_source = interp_signal(..., np.array([ophys_timestamps[0]]))`; `input_trials.append(np.zeros((0, target_times.shape[0]), ...))`; repeated `.copy()`/`.sort_values()`.

iii. The availability probe was a defensive check, empty inputs satisfy the decoder format, and metadata was retained for provenance rather than decoder features.
