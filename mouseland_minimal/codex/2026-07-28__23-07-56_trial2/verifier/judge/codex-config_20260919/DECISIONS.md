# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates every `*_neural_data.npy` spike file (89 physical sessions), loads every `Beh_*.npy` behavior dictionary into views keyed by the session name after removing `_swap1/_swap2`, merges duplicate behavior views after equality checks, and loads the corresponding retinotopy file per session.

ii. `spk_session_keys = sorted([path.name.replace("_neural_data.npy", "") for path in SPK_ROOT.glob("*_neural_data.npy")], key=natural_key)`; `for beh_path in sorted(BEH_ROOT.glob("Beh_*.npy")):`; `spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]`; `area_codes = np.load(ret_path, allow_pickle=True)["iarea"]`

iii. The trajectory says the 99 behavior views contain duplicates/swap views but the spike release contains the paper's 89 physical recordings; it therefore reconstructs sessions from spike filenames and verifies duplicated behavior annotations agree.

## 1-b. How are the data split into subjects?

i. Subject is parsed as the first underscore-separated component of each session key; sorted unique subjects form `subjects`, and every session receives its lookup index.

ii. `subject, year, month, day, block = session_key.split("_")`; `subjects = sorted({info["subject"] for info in parsed.values()})`; `subject_idx.append(subject_to_id[parsed[session_key]["subject"]])`

iii. The agent regarded the physical session filename as authoritative and reported 19 subjects across 89 sessions.

## 1-c. How are the data split into sessions?

i. Each spike filename defines one mouse/date/block session. Behavior keys differing only by `_swap1` or `_swap2` are collapsed to that physical session, with fields checked for agreement.

ii. `base_key = strip_swap_suffix(view_key)`; `return re.sub(r"_swap[12]$", "", session_key)`; `behavior_by_session = {key: merge_behavior_views(key, behavior_views[key]) for key in spk_session_keys}`

iii. The agent justified this using the paper's 89 recordings versus 99 behavior views and sought to avoid double-counting duplicate analysis views.

## 1-d. How are the data split into trials?

i. It loops over `ntrials`; a trial contains frames whose rounded `ft_trInd` equals the trial index and which additionally satisfy `ft_move > 0` and `ft_CorrSpc`. Those frames are then collapsed/interpolated to four corridor-position centers.

ii. `return np.where((ft_trial_idx == trial_idx) & move & in_corridor)[0]`; `for trial_idx in range(int(beh["ntrials"])):`; `target_positions=POSITION_BIN_CENTERS`

iii. The agent cited the paper's running-period restriction and the requested four 1 m position categories, interpreting these as grounds for a four-sample spatial representation.

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped if they have no moving in-corridor frames or do not reach the last bin center (35 dm). It raises if a session retains fewer than two trials. It does not apply the reference's global 99th-percentile long-trial filter.

ii. `if len(frame_idx) == 0: ... continue`; `if np.nanmax(trial_positions) < POSITION_BIN_CENTERS[-1]: ... continue`; `if len(session_neural) < 2: raise ValueError(...)`

iii. The trajectory says unusable running trials should be dropped; the completed run reported zero such drops, so all 38,110 trials were retained.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from each plane's `spks` in the session neural file, concatenated across planes. `iarea` supplies neuron region labels; behavior position/movement fields determine selection and interpolation.

ii. `spk = np.concatenate([plane.astype(np.float32, copy=False) for plane in spk_planes], axis=0)`; `area_codes = ret["iarea"]`

iii. The agent states these are the released deconvolved traces and the retinotopy grouping used by the reference analysis.

## 2-b. How is the `neural` data processed?

i. It concatenates planes, selects at most 512 neurons, and linearly interpolates each neuron's moving-frame activity by corridor position to centers 5, 15, 25, and 35 dm, averaging duplicate positions first. Results are stored as float16.

ii. `selected_neurons, region_idx = pick_neurons_by_variance(...)`; `neural_interp = interpolate_features(... target_positions=POSITION_BIN_CENTERS).astype(np.float16)`

iii. The agent called four spatial samples a decoder-facing interpretation of the four requested position bins and used float16/capping to make training tractable.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons outside V1/mHV/lHV/aHV are removed. If more than 512 remain, high-variance neurons over moving corridor frames are chosen with an initial 128-per-region quota and remainder fill.

ii. `valid_indices = np.array(sorted(i for indices in region_to_indices.values() for i in indices))`; `variances = np.var(spk_valid, axis=1)`; `per_region_quota = max_neurons // len(BRAIN_REGIONS)`

iii. Visual-area filtering was claimed to match the paper. The 512 cap was explicitly documented as a deliberate, format-driven deviation for decoder memory/trainability.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Metadata names corridor entry/trial start, but neural samples are aligned by spatial position and interpolated to four fixed corridor centers; there is no fixed time window from entry.

ii. `interpolate_features(positions=trial_positions, values=spk_selected[:, frame_idx], target_positions=POSITION_BIN_CENTERS)`; `"temporal_alignment_event": "corridor entry (trial start)"`

iii. The agent reasoned that ordered spatial bins begin at corridor entry and satisfy the requested alignment, while noting this deviated from raw-frame reference paths.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Each trial has four spatial samples, not uniform temporal bins. Nonetheless metadata reports 1666.67 ms from 1 m divided by a nominal 0.60 m/s. Extensive position interpolation/rebinning is applied; the actual median imaging frame period is separately recorded as about 0.3147 s.

ii. `"time_bin_size": float((1.0 / 0.60) * 1000.0)`; `"binning_scheme": "4 spatial bins of 1 m each ..."`

iii. The agent inferred a nominal time per spatial bin to satisfy metadata, based on a nominal VR speed, rather than retaining the native 3.17 Hz bins.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundTime` and frame timestamps `ft`, with position used for later interpolation.

ii. `sound_time = float(np.asarray(beh["SoundTime"])[trial_idx])`; `time_to_sound = ((sound_time - ft[frame_idx]) * SECONDS_PER_DAY)`

iii. The agent identified the times as MATLAB-day units and converted their difference to seconds.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Sound time minus each retained frame time is converted from days to seconds, then linearly interpolated by position to four centers.

ii. `interpolate_features(trial_positions, time_to_sound, POSITION_BIN_CENTERS)[0]`

iii. This preserves the sign convention (positive before cue, negative after) while fitting the chosen four-bin representation.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The timing values and neural rows use the same original trial frames and the same position-interpolation targets, yielding four aligned columns.

ii. Both calls use `positions=trial_positions` and `target_positions=POSITION_BIN_CENTERS`.

iii. The agent considered shared interpolation sufficient to maintain samplewise alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the date embedded in the session filename and the earliest recorded date for that subject.

ii. `date_obj = datetime.strptime(date_str, "%Y_%m_%d").date()`; `(info["date"] - first_day[info["subject"]]).days`

iii. The trajectory treats calendar days since a subject's first included recording as training day.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The earliest session date per mouse is assigned day 0; later dates use elapsed calendar days. That scalar is repeated across all four samples of every trial in the session.

ii. `np.full(4, subject_day_value, dtype=np.float32)`

iii. The agent chose elapsed calendar days as a continuous measure; it did not justify why gaps without recordings should count as training days.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It uses per-trial `Trial_start_time` and frame timestamps `ft`, plus position for interpolation.

ii. `trial_start_time = float(np.asarray(beh["Trial_start_time"])[trial_idx])`; `time_since_start = ((ft[frame_idx] - trial_start_time) * SECONDS_PER_DAY)`

iii. The agent identified corridor entry/trial start as the requested zero point.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Frame time minus trial-start time is converted from days to seconds and interpolated by position to four centers.

ii. `interpolate_features(trial_positions, time_since_start, POSITION_BIN_CENTERS)[0]`

iii. The stated aim was to preserve continuous elapsed time within the spatial representation.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It shares the neural data's retained frames and four position targets rather than native time bins.

ii. Both are passed through `interpolate_features(..., POSITION_BIN_CENTERS)`.

iii. The agent justified alignment by identical interpolation coordinates.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes directly from per-trial `isRew`.

ii. `float(np.asarray(beh["isRew"])[trial_idx])`

iii. The raw flag already denotes whether the corridor is rewarded.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The flag is cast to float and repeated over the trial's four samples.

ii. `np.full(4, float(np.asarray(beh["isRew"])[trial_idx]), dtype=np.float32)`

iii. Repetition makes the per-trial input conform to the common input matrix shape.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It comes from per-trial `WallName`; categories are the naturally sorted union of every literal wall name in all sessions.

ii. `names.update(map(str, np.asarray(beh["WallName"]).tolist()))`; `stim_name = str(np.asarray(beh["WallName"])[trial_idx])`

iii. The agent followed the trajectory version of the task that gave examples such as `circle1` and `leaf2`, retaining 15 detailed wall identities.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each literal name is mapped to an integer catalog index and repeated across four samples. Crop and swap variants are not collapsed to four texture families.

ii. `stimulus_to_id = {name: idx for idx, name in enumerate(stimulus_values)}`; `np.full(4, stimulus_to_id[stim_name], dtype=np.int16)`

iii. The agent interpreted “category” as each raw stimulus identity, supported by its prompt examples and validation of 15 classes.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It uses lick trial indices `LickTrind` and lick positions `LickPos`, not lick frame indices/times.

ii. `lick_trials = np.rint(np.asarray(beh["LickTrind"])).astype(np.int64)`; `lick_positions = np.asarray(beh["LickPos"], dtype=np.float32)`

iii. This was selected to make licking categorical within each spatial bin.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial and each 1 m interval, the bin is 1 if any lick position falls inside it; licks at or beyond 40 dm are forced into the last bin.

ii. `int(np.any((lick_pos_trial >= POSITION_BIN_EDGES[bin_idx]) & (lick_pos_trial < POSITION_BIN_EDGES[bin_idx + 1])))`; `lick_trial[-1] = 1`

iii. The agent wanted a binary, time-varying-looking output on its four spatial samples.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks are aggregated into the same four spatial intervals whose centers hold interpolated neural samples. This is spatial, not frame/time, alignment.

ii. `POSITION_BIN_EDGES = [0,10,20,30,40]` and neural `POSITION_BIN_CENTERS = [5,15,25,35]`.

iii. The agent regarded corresponding corridor bins as aligned.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Raw `ft_Pos` selects/interpolates frames and determines whether a trial reaches 35 dm; the saved output itself is a constructed `[0,1,2,3]` sequence.

ii. `ft_pos = np.asarray(beh["ft_Pos"][:n_frames], dtype=np.float32)`; `"position_bin": np.arange(4, dtype=np.int16)`

iii. Four ordered one-meter bins were taken directly from the decoder specification.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Every retained trial is resampled at the four bin centers and is assigned all four category labels in order, independent of the exact observed frames.

ii. `target_positions=POSITION_BIN_CENTERS`; `np.arange(4, dtype=np.int16)`

iii. The agent aimed for exactly four equally represented position classes.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The intended boundaries are 0, 10, 20, 30, 40 dm, represented by centers 5, 15, 25, 35 dm; output IDs 0–3 correspond to 0–1 m through 3–4 m.

ii. `POSITION_BIN_EDGES = np.array([0.,10.,20.,30.,40.])`; `output_values ... ["0_to_1m", ..., "3_to_4m"]`

iii. These are the four equal-length bins explicitly requested.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Neural activity is interpolated to each category's center, and the same column is assigned that center's bin ID.

ii. `neural_interp ... target_positions=POSITION_BIN_CENTERS`; `"position_bin": np.arange(4)`

iii. The shared ordered center grid is the agent's alignment mechanism.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from framewise `ft_RunSpeed` at retained moving in-corridor frames, with `ft_Pos` used for interpolation.

ii. `ft_speed = np.asarray(beh["ft_RunSpeed"][:n_frames], dtype=np.float32)`; `speed_interp = interpolate_features(positions=trial_positions, values=ft_speed[frame_idx], ...)`

iii. The agent chose the released behavioral speed stream.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speed is interpolated by position to four centers. All interpolated speeds across all sessions/trials are pooled to compute global 25th/50th/75th percentiles, then `digitize` assigns IDs 0–3.

ii. `q25, q50, q75 = np.quantile(speed_values, [0.25, 0.50, 0.75])`; `np.digitize(..., speed_edges[1:-1], right=False)`

iii. Pooling globally was intended to make each category correspond to roughly 25% of the converted samples.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Global thresholds are approximately 16.592, 28.701, and 43.168 in the raw speed units; values equal to a threshold enter the higher bin.

ii. `speed_edges = np.array([-np.inf, q25, q50, q75, np.inf])`; `right=False`

iii. The agent followed the explicit quartile requirement.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Both speed and neural activity are independently interpolated from the same retained frames/positions to the same four centers.

ii. Both `interpolate_features` calls use `trial_positions` and `POSITION_BIN_CENTERS`.

iii. The shared spatial grid was considered aligned.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. NaN trial indices are marked invalid; duplicate positions are averaged; duplicate behavior views are compared and disagreement raises; missing behavior raises; empty/incomplete trials are skipped; absent visual-area neurons are excluded; sessions with fewer than two trials raise. There is no general imputation.

ii. `valid = ~np.isnan(ft_trial_idx_raw)`; `np.unique(positions, return_inverse=True)`; `raise ValueError("Behavior views disagree...")`; `raise KeyError(f"Missing behavior entries...")`

iii. The agent emphasized fail-fast consistency checks and deterministic handling rather than silently merging contradictory data.

## 12-a. What are the most time-consuming steps of the code?

i. Loading multi-gigabyte spike files, concatenating/casting all planes, variance calculation for neuron selection, and per-trial/per-neuron position interpolation dominate. The full conversion visibly stalled on a 7.9 GB session.

ii. `spk = np.concatenate([...], axis=0)`; `spk_valid = spk[valid_indices][:, running_corridor_mask]`; nested calls to `interpolate_features`.

iii. Trajectory updates explicitly identify large spike files and CPU-bound session processing as the bottleneck.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-neuron area mapping, each row's duplicate-position aggregation/interpolation, per-trial frame-mask reconstruction, four-bin licking checks, and output finalization could be batched or vectorized.

ii. `for neuron_idx, area_code in enumerate(area_codes)`; `for row in range(values.shape[0])`; `for trial_idx in range(int(beh["ntrials"]))`; list comprehension over `range(4)`.

iii. The trajectory does not discuss vectorization; it accepted the stable long run rather than revising working code.

## 12-c. What processing does the code repeat multiple times?

i. `trial_frame_indices` repeatedly converts the entire session's `ft_trInd`, `ft_move`, and `ft_CorrSpc` arrays once per trial. `interpolate_features` repeatedly sorts/collapses identical trial positions for neural, speed, and two timing variables. Sample creation deep-copies the already built full data.

ii. `ft_trial_idx_raw = np.asarray(beh["ft_trInd"][:n_frames], ...)` occurs inside the trial helper; four separate `interpolate_features(...)` calls occur per trial.

iii. No explicit justification was recorded beyond implementation clarity and completing a stable conversion.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads and equality-checks many duplicate behavior fields, computes metadata/statistics and source-view lists unused by decoder training, builds a sample even with `--full-only`, and initially materializes all visual neurons before discarding most under the 512 cap. Continuous speed is retained only until global thresholds are computed.

ii. `check_fields = [...]`; `sample = make_sample_dataset(...)`; `spk_valid = spk[valid_indices][:, running_corridor_mask]`; `"running_speed_continuous": speed_interp`

iii. These steps were used for sanity checks, reproducibility, sample artifacts, and global speed binning, though they do not enter the final decoder features directly.
