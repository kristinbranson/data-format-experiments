# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads `bwm_release.csv`, groups its probe rows by session `eid`, constructs each local cache path from lab/subject/date/session number, and directly loads parquet/NumPy files. Missing sessions/files are skipped. It does not use ONE or restrict sessions with `DATALIMIT_SUBSET.csv`.

ii. `release = pd.read_csv(RELEASE_CSV, index_col=0)`; `sessions = release.groupby("eid", sort=False).agg(...)`; `return DATA_ROOT / row["lab"] / "Subjects" / row["subject"] / row["date"] / f"{int(row['session_number']):03d}"`

iii. The trajectory says the agent inspected local coverage and chose “a thinner converter against the cached ALF files.” Its notes justify this as using all 459 sessions in the provided release table and resolving paths directly from the local ONE cache.

## 1-b. How are the data split into subjects?

i. Subject names come from the release CSV. Sessions retain `row["subject"]`; unique subjects are accumulated in first-encounter order, and `subject_idx` maps every retained session to that list.

ii. `"subject": "first"`; `if rec["subject"] not in subject_to_idx: ...`; `"subject_idx": np.array([subject_to_idx[rec["subject"]] for rec in records], dtype=np.int16)`

iii. The trajectory treats the release metadata as authoritative and reports its subject/session totals as sanity checks.

## 1-c. How are the data split into sessions?

i. Probe-level release rows are grouped by `eid`; probe names are collected into a sorted tuple, producing one record per session. All probes in that session are later merged.

ii. `release.groupby("eid", sort=False).agg({... "probe_name": lambda x: tuple(sorted(x))}).reset_index()`

iii. The agent identified `eid` as the session identifier and explicitly aimed to merge probes belonging to the same session.

## 1-d. How are the data split into trials?

i. The trials parquet already has one row per trial. Stimulus-centered intervals are formed per row, continuous streams are interpolated per interval, and spikes are binned per interval.

ii. `stim_on = trials[ALIGN_EVENT].to_numpy(dtype=np.float32)`; `intervals = np.c_[stim_on + WINDOW[0], stim_on + WINDOW[1]]`; `for lo, hi, start in zip(start_idx, end_idx, starts, strict=True):`

iii. The trajectory follows the reference trial table and selected a common stimulus-aligned two-second trial window.

## 1-e. How are trials filtered based on quality controls?

i. Trials require reaction time 0.08–2 s, duration from go cue to feedback at most 10 s, nonzero choice, and nonmissing required fields. They also need wheel and whisker coverage, no NaNs in those traces, and at least one spike among retained neurons. Sessions need at least two surviving trials.

ii. `mask &= rt >= 0.08`; `mask &= rt <= 2.0`; `mask &= trial_len <= 10.0`; `mask &= trials["choice"].to_numpy() != 0`; `final_mask = trial_mask & wheel_valid & whisker_valid & neural_valid`

iii. The agent said the first conditions match `load_trials_and_mask`; it added the behavior-coverage checks for valid alignment and explicitly added the nonzero-spike rule “to avoid avoidable validator warnings.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural arrays derive from `spikes.times.npy` and `spikes.clusters.npy`. Cluster metrics, cluster channels, and channel atlas IDs determine unit QC and region labels.

ii. `spikes_times = np.load(probe_base / "spikes.times.npy")`; `spikes_clusters = np.load(probe_base / "spikes.clusters.npy")`; `metrics = pd.read_parquet(... "clusters.metrics.pqt")`

iii. The trajectory inspected the reference spike-loading utilities and notes that it loads those five spike/cluster/channel resources per probe.

## 2-b. How is the `neural` data processed?

i. Good-unit spikes from every probe are remapped into one session-wide unit index, sorted by time, and counted into 100 20-ms bins per trial. Counts are stored as `float16`; they are not divided by bin width into firing rates.

ii. `flat = spike_clusters[lo:hi][valid] * NBINS + bins[valid]`; `counts = np.bincount(flat, minlength=n_neurons * NBINS).reshape(n_neurons, NBINS)`; `trials.append(counts.astype(np.float16))`

iii. The agent justified 2-s/20-ms binning from the methods and probe merging because probes in a session form one population. It did not state a reason for retaining counts instead of the reference’s Hz rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters with metrics label at least 1 are retained. Unlike the human solution, atlas `void` units are not removed; the notes explicitly preserve `root`, `void`, `x`, and `y`. Trials with no retained-unit spikes are additionally removed.

ii. `good_mask = metrics["label"].to_numpy(dtype=float) >= 1.0`; `neural_valid = np.array([np.any(trial > 0) for trial in neural_trials], dtype=bool)`

iii. The agent reproduced the paper’s 75,708 label-passing units and argued that region filtering belonged to per-region analysis, so preserving all QC-passed units was more faithful for pooled decoding.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial uses `[stimOn_times - 0.5, stimOn_times + 1.5)`. Spikes are sliced on the shared session clock and their bin is computed relative to the interval start.

ii. `ALIGN_EVENT = "stimOn_times"`; `intervals = np.c_[stim_on + WINDOW[0], stim_on + WINDOW[1]]`; `times = spike_times[lo:hi] - start`

iii. The agent followed the user-required common stimulus alignment even though it noted that the original methods used first-movement alignment for some dynamic targets.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 20 ms, with 100 bins across two seconds. Raw spikes are binned once; there is no later neural resampling or rebinning.

ii. `BIN_SIZE_S = 0.02`; `NBINS = int(round((WINDOW[1] - WINDOW[0]) / BIN_SIZE_S))`

iii. The methods describe two-second trials divided into 20-ms bins, which the agent explicitly followed.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is defined from the chosen window and bin size relative to `stimOn_times`, rather than measured from another raw stream.

ii. `time_input = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)).astype(np.float32)`

iii. The agent selected `stimOn_times` because the task explicitly requires stimulus alignment.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code creates 100 bin-end timestamps from -0.48 through 1.50 seconds and repeats the same vector for every trial.

ii. `WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)`; `np.vstack([time_input, ...])`

iii. The notes explicitly describe these as “bin end times.” No justification is given for using ends rather than the human reference’s bin centers.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It has the same 100 positions as neural bins but labels each interval by its right edge. Neural spike counts cover intervals beginning at -0.5 s, so the timestamp is not the bin center used by the reference.

ii. `bins = np.floor(times / BIN_SIZE_S)` versus `time_input = WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1)`

iii. The agent regarded shared stimulus alignment and equal length as sufficient and documented the endpoint convention in metadata.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the full, unfiltered `probabilityLeft` sequence; a probability change denotes a new block.

ii. `trial_number = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy(dtype=np.float32))`

iii. The agent followed the reference concept that `probabilityLeft` is constant within a block and computed numbering before filtering.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Numbering starts at 1, increments while consecutive probabilities match, and resets to 1 on a change. The scalar is broadcast across 100 bins.

ii. `trial_num = np.ones(...)`; `curr += 1.0`; `curr = 1.0`; `np.full(NBINS, trial_number[i], dtype=np.float32)`

iii. The notes explicitly chose one-based numbering; they give no reason for differing from the human reference’s zero-based position.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes directly from `trials["choice"]` after no-choice trials are removed.

ii. `choice = map_choice(trials["choice"].to_numpy(dtype=np.float32)[selected_idx])`

iii. The agent used the IBL trials table and considered the choice column authoritative.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The agent maps raw -1 to category 0 (“left”) and raw +1 to category 1 (“right”), then repeats it over all bins. This is reversed relative to the human reference.

ii. `out[np.isclose(values, -1.0)] = 0`; `out[np.isclose(values, 1.0)] = 1`; `np.full((1, NBINS), choice, dtype=np.int8)`

iii. Its notes assert that IBL `-1` means left and `+1` means right; that assertion drove the mapping.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from `trials["probabilityLeft"]` for each selected trial.

ii. `prior = map_probability_left(trials["probabilityLeft"].to_numpy(dtype=np.float32)[selected_idx])`

iii. The agent followed the requested raw field and category mapping.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values are rounded to one decimal and mapped 0.2→0, 0.5→1, 0.8→2, then repeated over time.

ii. `rounded = np.round(values.astype(np.float64), 1)`; `out[np.isclose(rounded, 0.2)] = 0`; `...0.5... = 1`; `...0.8... = 2`

iii. This mapping is explicitly required by the task; rounding guards against floating-point representation.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It derives from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii. `timestamps = np.load(... "_ibl_wheel.timestamps.npy")`; `position = np.load(... "_ibl_wheel.position.npy")`

iii. The agent inspected the IBL wheel implementation and followed the same source streams as the reference.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Position is interpolated to 1 kHz, filtered/differentiated with a 20-Hz, order-8 velocity filter, absolute-valued, and interpolated to trial timestamps.

ii. `interp_pos, interp_t = interpolate_position(..., freq=WHEEL_FS)`; `velocity, _ = velocity_filtered(... corner_frequency=20, order=8)`; `return interp_t, np.abs(velocity)`

iii. The agent says this is the same IBL wheel preprocessing path used by the provided code.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Two thresholds are global 1/3 and 2/3 quantiles pooled over every included session and time bin; `np.digitize` produces low/medium/high.

ii. `wheel_thresholds = np.quantile(np.concatenate(wheel_pool), [1 / 3, 2 / 3])`; `np.digitize(values, thresholds, right=False)`

iii. The agent argued global tertiles preserve rank, prevent session-specific label drift, and balance classes. The human reference instead uses per-session percentiles.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel is interpolated at 100 trial-relative points from -0.48 through 1.50 s. These are bin ends, whereas the reference samples bin centers.

ii. `x_interp = (WINDOW[0] + BIN_SIZE_S * np.arange(1, NBINS + 1))`; `interp = np.interp(x_interp, rel_t, y)`

iii. The agent used one common stimulus-aligned two-second grid for every variable, but chose endpoints without explaining the 10-ms offset from reference centers.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses left-camera ROI motion energy and camera times when possible, falling back to right-camera equivalents. Extra leading timestamps are trimmed if timestamps outnumber values.

ii. `times = np.load(... f"*_ibl_{view}Camera.times.npy")`; `values = np.load(... f"{view}Camera.ROIMotionEnergy.npy")`; `times = times[-values.shape[0]:]`

iii. The agent says left preference/right fallback matches provided code and materially retains six sessions.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released values receive no filtering or normalization. They are interpolated onto the trial-relative endpoint grid and later discretized.

ii. `whisker_trials, whisker_valid = interpolate_behavior_into_trials(whisker_times, whisker_me, intervals)`

iii. The notes state that the released motion-energy trace is used and that the camera fallback/length reconciliation follows reference behavior.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Global pooled 1/3 and 2/3 quantiles define three categories across the complete converted dataset.

ii. `whisker_thresholds = np.quantile(np.concatenate(whisker_pool), [1 / 3, 2 / 3])`; `digitize_tertiles(whisker, whisker_thresholds)`

iii. The same global-rank, stable-label, class-balance rationale as wheel speed was used; this differs from reference per-session thresholds.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Motion energy is interpolated relative to stimulus onset at bin ends (-0.48 to 1.50), not the reference bin centers.

ii. `rel_t = t - beg`; `interp = np.interp(x_interp, rel_t, y).astype(np.float32)`

iii. The agent aimed for a unified stimulus-aligned grid for neural and behavioral variables.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. File matching chooses the lexicographically latest revision. Missing required files skip a session; left-camera failure triggers right-camera fallback; camera length mismatch is trimmed only when times are longer; invalid/NaN/uncovered trials are dropped; broad per-session exceptions are logged and skipped.

ii. `return matches[-1]`; `except FileNotFoundError: skip_reasons["missing_required_file"] += 1`; `except Exception as exc: ...`; `if np.isnan(y).any(): outputs.append(None)`

iii. The agent prioritized completing a valid local-cache conversion, documented 20 missing-file sessions and one alignment failure, and used fallback/skip behavior to keep processing robust.

## 10-a. What are the most time-consuming steps of the code?

i. The likely dominant work is reading large spike arrays for every probe, followed by per-trial spike binning and 1-kHz wheel interpolation/filtering. The agent ran sessions sequentially.

ii. `spikes_times = np.load(...)`; `for lo, hi, start in zip(...)`; `for idx, row in enumerate(sessions.to_dict("records"), start=1):`

iii. The trajectory shows full conversion taking sustained runtime and repeatedly waits on it, but the agent did not provide a formal profile.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial spike-binning loop, behavior interpolation loop, input construction, output construction, and block-number loop are vectorization candidates. Session/probe loops naturally involve heterogeneous files and arrays.

ii. `for lo, hi, start in zip(...)`; `for i, (ib, ie, beg, end) in enumerate(...)`; `for i in range(1, prob_left.shape[0]):`

iii. The trajectory emphasizes correctness and inspection over optimization; it does not justify these loops or analyze vectorization.

## 10-c. What processing does the code repeat multiple times?

i. It scans/bin-processes every trial before applying the already-known reference trial mask, interpolates wheel and whisker in separate passes through the same helper, and later loops over trials again to build inputs and outputs.

ii. `neural_trials = bin_spikes(... intervals ...)`; `wheel_trials, ... = interpolate_behavior_into_trials(...)`; `whisker_trials, ... = interpolate_behavior_into_trials(...)`; `selected_idx = np.flatnonzero(final_mask)`

iii. No explicit justification was recorded. The structure keeps stream-specific operations simple but repeats trial traversal and performs work later discarded.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Neural and behavioral traces are computed for trials already rejected by `trial_mask`; they are discarded only after `final_mask`. Extensive session metadata and conversion summaries are also stored although the decoder does not consume them. Continuous behavior is retained in intermediate records until categorical outputs are built.

ii. `final_mask = trial_mask & wheel_valid & whisker_valid & neural_valid`; `selected_idx = np.flatnonzero(final_mask)`; `"session_info": [...]`

iii. The agent valued provenance, debugging summaries, and validator cleanliness; it did not discuss the extra computation or metadata size as an efficiency tradeoff.
