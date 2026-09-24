# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates every `.nwb` file in sorted subject directories, opens each once with `h5py`, converts it, and accumulates retained sessions before pickling the dataset.

ii. `for subject in sorted(os.listdir(data_root)):` / `if filename.endswith(".nwb"):` / `with h5py.File(path, "r") as f:`

iii. The trajectory says NWB is the authoritative source and reports finding 174 files; sorting provides deterministic ordering. It chose direct HDF5 access for speed and schema inspection.

## 1-b. How are the data split into subjects?

i. Subject identity is the parent directory name (for example `sub-440956`). Unique names are sorted and each session receives an integer index.

ii. `return os.path.basename(os.path.dirname(path))` and `subjects = sorted({session["subject"] for session in session_results})`

iii. The agent reasoned that the DANDI directory layout already groups sessions by subject and verified 28 subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session; the filename stem is its session ID and file order determines output order.

ii. `session_id = get_session_id_from_path(path)` / `result = convert_session(path)`

iii. The trajectory identifies the dataset as one NWB file per recording session and excludes only the file with no classifier-good units.

## 1-d. How are the data split into trials?

i. Behavioral trials come from `intervals/trials`, but only trials matching the first good unit's ragged `obs_intervals` are retained. Interval start/stop pairs rounded to four decimals map ephys trials back to behavioral rows. Go cues are then selected by requiring exactly one within each retained trial.

ii. `trial_lookup = {(round(float(start), 4), round(float(stop), 4)): idx ...}` / `trial_indices.append(trial_lookup[key])` / `go_times = assert_one_event_per_trial(...)`

iii. The agent found more behavioral than ephys-covered trials and used `obs_intervals` to prevent behavior-only trials becoming fabricated all-zero neural trials.

## 1-e. How are trials filtered based on quality controls?

i. It keeps all trials represented in the first good unit's `obs_intervals`; it does not remove `free_water` trials and does not enforce two surviving trials. Sessions with no good units are removed.

ii. `first_unit_obs = obs_intervals[obs_start:obs_stop]` / `trial_indices = np.asarray(trial_indices, dtype=np.int64)`

iii. The trajectory explicitly says it kept all ephys-covered trials to preserve ignore, early-lick, and photostimulation conditions. It believed `obs_intervals` eliminated behavior-only zero-neural trials, although validation still found 2,452 silent trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from ragged `units/spike_times` and `spike_times_index`, filtered by `units/classification`, and aligned using trial start/stop and go-cue timestamps.

ii. `spike_times_flat=np.asarray(f["units/spike_times"][:], ...)` / `spike_times_index=np.asarray(f["units/spike_times_index"][:], ...)`

iii. The agent identified spike timestamps as the raw neural representation and classification as the paper's classifier-based QC field.

## 2-b. How is the `neural` data processed?

i. For each good unit, spikes are assigned to behavioral trials, made relative to each go cue, restricted to the four-second window, accumulated into 50 ms counts, transposed to neuron-by-time, cast to `float16`, and multiplied by 20 to obtain Hz.

ii. `np.add.at(counts[:, :, good_unit_col], (valid_trial_idx, bin_idx), 1)` / `spike_counts[trial_idx].T.astype(np.float16) * (1.0 / BIN_SIZE_S)`

iii. The trajectory states that nonoverlapping spike counts divided by bin width reproduce firing rates without smoothing or normalization; `float16` reduced the multi-gigabyte output.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose `classification` equals `"good"` are retained; a session with zero such units is skipped. No individual metric thresholds or `unit_quality` filter are applied.

ii. `good_unit_mask = classifications == "good"` / `if n_good_units == 0: return None`

iii. The agent linked this label to the Chen et al. spike-sorting classifier and noted that excluding one unclassified session yields 173 sessions and 69,453 units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes are placed on the common session clock, associated with trials, then converted to seconds relative to that trial's go cue; only offsets from -2.5 to +1.5 seconds are retained.

ii. `rel_spikes = spikes[valid] - go_times[valid_trial_idx]` / `in_window = (rel_spikes >= WINDOW_START_S) & (rel_spikes < WINDOW_END_S)`

iii. The agent reasoned that NWB spikes and events share an absolute clock, so subtracting the go timestamp provides direct alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is 50 ms, with 80 nonoverlapping bins over [-2.5, 1.5). Raw spike timestamps are histogrammed into these bins; no later rebinning, smoothing, or sliding window is used.

ii. `BIN_SIZE_S = 0.05` / `BIN_EDGES_S = np.arange(...)`

iii. This directly follows the decoder instructions and was recorded as 50 ms in metadata.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It derives tone onset from `BehavioralEvents/sample_start_times`, using behavioral trial boundaries and go cues to select the last sample event before each go cue.

ii. `sample_start_times = last_event_before_per_trial(sample_events, trial_start, go_times, "sample_start_times")`

iii. The agent recognized repeated sample events after early licks and therefore chose the last sample start preceding go, rather than assuming one sample event per trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Sample onset is expressed relative to go; each bin-center offset relative to go then subtracts that sample-relative offset, producing elapsed seconds since tone onset.

ii. `sample_rel = sample_start_times - go_times` / `time_from_tone = BIN_CENTERS_S[None, :] - sample_rel[:, None]`

iii. The trajectory describes this as a continuous time-varying regressor at bin centers, as requested.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It uses the same 80 go-relative 50 ms bin centers as neural activity, so input column k describes the center of neural bin k.

ii. `BIN_CENTERS_S = BIN_EDGES_S[:-1] + (BIN_SIZE_S / 2.0)`

iii. The agent deliberately used bin centers to give each binned neural sample one corresponding elapsed-time value.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses per-trial `photostim_onset` and `photostim_duration`, plus trial start and go-cue times.

ii. `photostim_onset = ...["photostim_onset"]...` / `photostim_duration = ...["photostim_duration"]...`

iii. The agent found these explicit trial-table fields and treated `N/A`, empty, or null values as no stimulation.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Onset is interpreted relative to trial start, duration defines offset, and every 50 ms bin overlapping the half-open stimulation interval is assigned 1; all others are 0.

ii. `rel_on = (trial_start[trial_idx] + onset) - go_times[trial_idx]` / `overlap = (bin_left < rel_off) & (bin_right > rel_on)`

iii. The agent chose overlap rather than center sampling so any bin containing stimulation is marked on.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute stimulation onset is shifted by the same trial go cue and intersected with the same go-relative bin edges used for spike counts.

ii. `bin_left = BIN_EDGES_S[:-1][None, :]` / `bin_right = BIN_EDGES_S[1:][None, :]`

iii. The common go-relative grid ensures a one-to-one correspondence between photostimulation and neural bins.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from `left_lick_times`, `right_lick_times`, trial bounds, and go cues; `trial_instruction` is a fallback when no lick is found.

ii. `choice, choice_sources = compute_choice_labels(... instructions=..., left_lick_times=..., right_lick_times=...)`

iii. The trajectory says the actual choice was not stored explicitly, so it preferred the first post-go lick, then any trial lick, then instructed side.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The earliest post-go lick determines left=0 or right=1. If absent, the earliest lick anywhere in the trial is used; if still absent, the instructed side is assigned. The result is repeated over all bins. No no-lick category exists.

ii. `choice[trial_idx] = 0 if left_first < right_first else 1` / `choice[trial_idx] = 0 if instructions[trial_idx] == "left" else 1`

iii. The agent used fallbacks to avoid missing labels and reported 12,662 instruction-fallback trials. It considered two classes consistent with its chosen `OUTPUT_VALUES`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the retained rows of `intervals/trials/outcome`.

ii. `outcomes_raw = decode_array(trial_group["outcome"][:])[trial_indices]`

iii. The field already contains the requested categories, so the agent did not infer outcome.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings map to ignore=0, miss=1, hit=2 and each per-trial label is repeated across all 80 time bins.

ii. `outcome_map = {"ignore": 0, "miss": 1, "hit": 2}` / `np.full(..., outcome[trial_idx], dtype=np.int8)`

iii. The mapping follows the requested category order; repetition lets all outputs share a time-indexed matrix.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from `intervals/trials/early_lick` for retained trial rows.

ii. `early_raw = decode_array(trial_group["early_lick"][:])[trial_indices]`

iii. The agent used the explicit behavioral flag rather than reconstructing early licks from event times.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `"no early"` maps to 0 and `"early"` to 1, repeated across 80 bins.

ii. `early_map = {"no early": 0, "early": 1}` / `np.full(..., early[trial_idx], dtype=np.int8)`

iii. This implements the requested no/yes categorical output and the common output shape.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses column 1 of `Camera0_side_TongueTracking/data` and that series' timestamps. It reads but does not use the likelihood column.

ii. `tongue_data = np.asarray(tongue_group["data"][:], ...)` / `tongue_y_raw = tongue_data[:, 1]`

iii. The agent identified the side-camera tongue stream as the relevant source, but assumed every reported y coordinate was valid.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. It computes 40th/60th percentiles over all raw frame-level y values in the session. For each trial/bin it takes the final frame before the bin's right edge, even when the bin has no sample (previous-frame fallback), then discretizes that scalar.

ii. `q40 = float(np.percentile(tongue_y_raw, 40))` / `sample_idx = np.clip(right_idx - 1, 0, len(timestamps) - 1)`

iii. The trajectory claims last-sample alignment matches repository marker processing and says fallback avoids NaNs; it logged 149,144 fallback bins.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Values below q40 become 0, values from q40 through q60 become 1, and values above q60 become 2. There is no fourth `not visible` category.

ii. `tongue_disc = np.ones_like(y_binned, dtype=np.int8)` / `tongue_disc[y_binned < q40] = 0` / `tongue_disc[y_binned > q60] = 2`

iii. The agent followed the three percentile bands literally, but did not account for tracking confidence or the task's explicit `not visible` value.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera samples are located using absolute go-cue-plus-bin-edge times. Each output bin uses the last camera sample before its right edge, falling back to a preceding sample if none lies inside.

ii. `trial_edges = go_times[:, None] + BIN_EDGES_S[None, :]` / `right_idx = np.searchsorted(timestamps, trial_edges[:, 1:], side="left")`

iii. The agent reasoned camera and ephys share the NWB clock and used a causal previous-frame rule rather than interpolation.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions with no good units are skipped; unmatched or malformed trial/event relationships raise errors; missing photostim values mean off; missing tongue frames are filled from the previous frame. No-confidence tongue detections are not treated as missing, and free-water all-zero trials remain.

ii. `if n_good_units == 0: return None` / `raise ValueError(...)` / `if onset is None or duration is None: continue` / `sample_idx = np.clip(right_idx - 1, ...)`

iii. The trajectory emphasizes strict schema checks and retaining valid decoder examples, while preferring imputation for camera sampling gaps. Validation warnings were accepted as real sparsity.

## 10-a. What are the most time-consuming steps of the code?

i. The full run is dominated by reading large spike/camera arrays, per-unit spike assignment/binning, retaining the complete multi-gigabyte dataset in memory, and writing the 5.8 GB pickle. Full conversion took long enough to require repeated monitoring.

ii. `for unit_idx, unit_end in enumerate(spike_times_index):` / `write_pickle(args.full_out, full_data)`

iii. The trajectory attributes runtime growth to later sessions having many more units and reports a healthy CPU-bound conversion followed by a large pickle write.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-trial choice extraction, photostimulation conversion, output-list construction, and interval-to-trial lookup could be vectorized. The per-unit spike loop is partly vectorized within each unit but remains a Python loop because spike trains are ragged.

ii. `for trial_idx in range(len(trial_start)):` / `for unit_idx, unit_end in enumerate(spike_times_index):` / `for trial_idx in range(n_trials):`

iii. The agent intentionally vectorized bin operations inside loops and regarded ragged units and trial-specific events as clearer to handle iteratively; no explicit optimization discussion beyond memory/runtime monitoring was given.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly scans lick arrays separately for every trial, repeatedly decodes scalar photostim values, and builds per-trial matrices in Python. It also builds both full and sample datasets from already converted session structures, duplicating dataset assembly and sample serialization.

ii. `left_lick_times[(left_lick_times >= start) & ...]` / `full_data, full_summary = build_dataset(session_results)` / `sample_data, sample_summary = build_dataset(sample_results)`

iii. The trajectory frames the sample artifact as useful validation and accepts the repeated lightweight assembly; it does not explicitly justify repeated lick-array scans.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes extensive counters, summaries, fallback counts, sample data, and metadata statistics not needed by downstream decoding. It also reads `n_behavior_trials`, `is_good_trials` dimensions, and left/right lick streams chiefly for checks or its nonreference choice derivation.

ii. `choice_sources = Counter()` / `summary = {...}` / `sample_data, sample_summary = build_dataset(sample_results)`

iii. The trajectory used these diagnostics to validate conversion decisions and decoder behavior, so they aided development even though most do not affect `converted_data.pkl` analyses.
