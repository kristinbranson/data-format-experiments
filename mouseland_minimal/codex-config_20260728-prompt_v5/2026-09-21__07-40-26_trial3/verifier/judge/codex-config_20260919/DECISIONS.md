# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. It loads `Imaging_Exp_info.npy` and every `Beh_<exp_type>.npy` into a canonical behavior map, lists every spike-file session, and loads that session’s `spks` plus its retinotopy `iarea`. Duplicate behavior entries are checked for equality and represented once.

ii. ```python
`exp_info = np.load(EXP_INFO_PATH, allow_pickle=True).item()`
`beh = np.load(behavior_path, allow_pickle=True).item()`
`spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]`
```

iii. The trajectory says it inspected the paper code/data layout, handled swap-suffixed behavior keys, and deduplicated to 89 raw recordings.

## 1-b. How are the data split into subjects?

i. Mouse IDs are parsed from spike session keys, sorted uniquely into `subjects`, and mapped to one `subject_idx` per retained session.

ii. ```python
`subjects = sorted({parse_base_key(key)[0] for key in session_keys})`
`subject_idx = np.array([subject_to_idx[parse_base_key(key)[0]] for key in session_keys])`
```

iii. The trajectory reports a full pass over all 89 sessions and deterministic session deduplication.

## 1-c. How are the data split into sessions?

i. Each `mouse_YYYY_MM_DD_block` spike filename defines a session. Behavior records with optional stimulus suffixes are canonicalized by the same base key; duplicate experiment-type listings are equality-checked and collapsed.

ii. ```python
`session_keys.append(filename[:-len("_neural_data.npy")])`
`if base_key not in canonical: canonical[base_key] = {...}`
```

iii. The agent explicitly identified swap-key edge cases and says it deduplicated analysis metadata to 89 imaging recordings.

## 1-d. How are the data split into trials?

i. For each declared trial number, it selects only frames whose `ft_trInd` equals that trial and which satisfy the global retained-frame mask `ft_CorrSpc & (ft_move > 0)`. Trials may have variable length.

ii. ```python
`retained_mask = ft_corr & (ft_move > 0)`
`frame_idx = np.flatnonzero((ft_trind == trial) & retained_mask)`
```

iii. The agent believed matching the paper required retaining running frames inside the corridor and building each trial on that grid.

## 1-e. How are trials filtered based on quality controls?

i. Trials with no running corridor frames are skipped; sessions with fewer than two remaining trials are skipped. It does not apply the reference’s whole-trial 99th-percentile duration filter.

ii. ```python
`if len(frame_idx) == 0: continue`
`if len(neural_trials) < 2: return None`
```

iii. The trajectory justifies removing stationary frames as paper-matched and excluding stationary reward-collection periods.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from per-plane `spks`; `iarea` supplies retinotopic region labels used to choose neurons.

ii. ```python
`spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]`
`return build_region_index(retino["iarea"])`
```

iii. The agent inspected representative neural and retinotopy files and identified deconvolved traces and visual-area mappings.

## 2-b. How is the `neural` data processed?

i. It scores neurons using corridor variance and corridor-versus-gray means, selects up to 128 per region, concatenates selected rows across planes, slices retained trial frames, and casts to float16.

ii. ```python
`session_matrix = build_selected_session_matrix(spk_planes, selected_global)`
`neural = session_matrix[:, frame_idx].astype(np.float16, copy=False)`
```

iii. The agent cited tractability of the 405-GB raw source and chose deterministic curation capped at 512 neurons/session.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It drops neurons outside V1/mHV/lHV/aHV, prefers neurons with corridor mean above gray mean and finite corridor variance, ranks by variance, caps each region at 128, and backfills with nonresponsive region neurons if needed.

ii. ```python
`responsive = corridor_mean > gray_mean`
`region_selected = sort_take_desc(resp_scores, resp_indices, max_per_region)`
```

iii. The trajectory calls visual-cortex restriction paper-consistent and the cap necessary for a usable artifact.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It slices frame-aligned traces by each trial’s running corridor frames. Thus sequences are associated with corridor entry but can start after entry when the animal is not moving, with no padding or resampling.

ii. ```python
`frame_idx = np.flatnonzero((ft_trind == trial) & retained_mask)`
`neural = session_matrix[:, frame_idx]`
```

iii. The agent measured lag of the first retained sample and judged the running-only grid compatible with trial-start alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied; native imaging frames are retained. However metadata computes `TIME_BIN_MS` by treating about 0.3147 seconds as days, yielding about 27,190,000 ms rather than about 315 ms.

ii. ```python
`TIME_BIN_MS = 1000.0 * 24.0 * 3600.0 * 0.31469352543354034`
`"time_bin_size": float(TIME_BIN_MS)`
```

iii. The agent measured native intervals, chose to preserve the frame grid, and regarded it as roughly 0.315 s.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It uses per-trial `SoundTime` and per-frame timestamps `ft`.

ii. ```python
`sound_time = np.asarray(behavior["SoundTime"], dtype=np.float64)`
`ft_time = np.asarray(behavior["ft"][:n_frames])`
```

iii. The agent established that behavior arrays are imaging-frame aligned.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. It subtracts each retained frame’s MATLAB-day timestamp from the trial sound timestamp and converts days to seconds; values are positive before and negative after the cue.

ii. ```python
`times_to_sound = ((sound_time[trial] - ft_time[frame_idx]) * 24.0 * 3600.0).astype(np.float32)`
```

iii. The trajectory supports direct timestamp alignment rather than guessing from cue position.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It indexes `ft` with exactly the same `frame_idx` used for neural columns.

ii. ```python
`ft_time[frame_idx]`
`session_matrix[:, frame_idx]`
```

iii. The agent says all derived variables were built on the same retained imaging-frame grid.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It parses mouse and calendar date from every spike session filename.

ii. ```python
`date_from_base_key(session_key)`
`by_mouse[mouse].append(session_key)`
```

iii. The agent inspected session ordering before implementation.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. It calculates calendar-day difference from that mouse’s earliest imaging date, then broadcasts the value over each trial’s retained frames.

ii. ```python
`training_day[session_key] = float((date_from_base_key(session_key) - first_day[mouse]).days)`
`np.full(len(frame_idx), day_of_training)`
```

iii. Metadata explicitly defines this as calendar days since the first imaging session.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It uses per-trial `Trial_start_time` and frame timestamps `ft`.

ii. ```python
`trial_start_time = np.asarray(behavior["Trial_start_time"], dtype=np.float64)`
```

iii. The agent inspected a trial around start/cue/end markers before writing the slicer.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. It subtracts trial-start timestamp from each retained frame timestamp and converts MATLAB days to seconds.

ii. ```python
`times_since_start = ((ft_time[frame_idx] - trial_start_time[trial]) * 24.0 * 3600.0).astype(np.float32)`
```

iii. The decision uses the raw timing fields directly.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed on the identical retained `frame_idx`; because stationary frames are removed, it need not begin at zero or advance uniformly.

ii. ```python
`ft_time[frame_idx]`
`session_matrix[:, frame_idx]`
```

iii. The agent accepted a small first-sample lag after applying the paper-inspired running filter.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the trial-level boolean `isRew`.

ii. ```python
`is_rew = np.asarray(behavior["isRew"], dtype=bool)`
```

iii. No separate trajectory rationale was given beyond deriving required variables from behavior fields.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean is converted to float and broadcast across all retained frames of the trial.

ii. ```python
`reward_available = np.full(len(frame_idx), float(is_rew[trial]), dtype=np.float32)`
```

iii. This implements the requested per-trial binary input on the common temporal shape.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It uses each trial’s `WallName`.

ii. ```python
`wall_names = np.asarray(behavior["WallName"])`
```

iii. The agent enumerated wall names to choose a consistent label space.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. It strips `_swap1/_swap2` and trailing digits, maps circle/leaf/rock/wood to 0–3, and broadcasts the category across trial frames.

ii. ```python
`category = re.sub(r"_swap[12]$", "", str(wall_name))`
`category = re.sub(r"\d+$", "", category)`
`stim_out = np.full(len(frame_idx), stim_category, dtype=np.int8)`
```

iii. The agent chose broad corridor categories rather than exact wall identities to avoid mixing label schemes.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It uses `LickTrind`, `LickFr`, and `LickPos`.

ii. ```python
`lick_trial_index = np.asarray(behavior["LickTrind"])`
`lick_frame_idx = np.asarray(behavior["LickFr"])`
`lick_pos = np.asarray(behavior["LickPos"])`
```

iii. The agent identified lick events during behavior-schema inspection.

## 8-b. What processing is involved in computing `output` *Licking*?

i. It selects licks for the trial and within 0–texture length, assigns each to the nearest retained frame if within 0.5 frame, and sets unique assigned frames to 1.

ii. ```python
`valid = ... & (lick_pos >= 0.0) & (lick_pos < texture_length)`
`lick_binary[np.unique(nearest[nearest_dist <= 0.5])] = 1`
```

iii. The common retained grid forced event reassignment because stationary frames had been removed.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick frame numbers are mapped to nearest entries of the same retained frame-index array used for neural columns.

ii. ```python
`nearest_retained_licks(retained_frame_idx=frame_idx, ...)`
```

iii. The agent intended every stream to share the running-corridor frame grid.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It uses per-frame `ft_Pos` and `Texture_Length` (default 40 decimeters).

ii. ```python
`ft_pos = np.asarray(behavior["ft_Pos"][:n_frames])`
`texture_length = float(behavior.get("Texture_Length", 40.0))`
```

iii. The agent followed the per-frame behavior schema.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Positions are clipped into the texture range, digitized using four equal edges, and cast to int8.

ii. ```python
`pos = np.clip(..., 0.0, float(texture_length) - 1e-6)`
`bin_edges = np.linspace(0.0, float(texture_length), 5)`
```

iii. This directly implements four equal-length spatial categories.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are 10, 20, and 30 decimeters for the usual 40-dm corridor, producing categories 0–3 for 0–1, 1–2, 2–3, and 3–4 m.

ii. ```python
`np.digitize(pos, bin_edges[1:-1], right=False)`
```

iii. The output labels explicitly name the four 1-m bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. It indexes `ft_Pos` with the same retained `frame_idx` used for neural columns.

ii. ```python
`position_out = make_position_bins(ft_pos[frame_idx], texture_length)`
```

iii. The agent built outputs on the same frame grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses per-frame `ft_RunSpeed`, with `ft_CorrSpc` and `ft_move` defining which speeds contribute to thresholds.

ii. ```python
`mask = sess["ft_CorrSpc"] & (sess["ft_move"] > 0)`
`sess["ft_RunSpeed"][mask]`
```

iii. The agent interpreted the paper as retaining only running corridor frames.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. It pools speeds from retained frames across all selected sessions, computes global 25th/50th/75th value quantiles, and digitizes each trial’s speeds.

ii. ```python
`q25, q50, q75 = np.quantile(speed_values, [0.25, 0.5, 0.75])`
`make_speed_bins(ft_speed[frame_idx], speed_edges)`
```

iii. The agent preserved exact cut points in metadata and used generic quartile labels.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Values below/above the three global quantile edges are assigned categories 0–3 using `np.digitize`; ties stay together, so categories need not contain exactly 25% each.

ii. ```python
`np.clip(np.digitize(speed, speed_edges, right=False), 0, 3)`
```

iii. The intent was four bins corresponding to global data quartiles.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. It indexes `ft_RunSpeed` with the same retained `frame_idx` used for neural columns.

ii. ```python
`speed_out = make_speed_bins(ft_speed[frame_idx], speed_edges)`
```

iii. The agent built all streams on the common retained grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. It validates duplicated behavior entries and retinotopy length, errors on missing behavior/unexpected wall names/no selected neurons, skips empty trials and sessions with fewer than two trials, defaults missing `Texture_Length` to 40, and filters nonfinite lick events/variance. It does not catch per-session exceptions.

ii. ```python
`if missing_behavior: raise ValueError(...)`
`if len(frame_idx) == 0: continue`
`texture_length = float(behavior.get("Texture_Length", 40.0))`
```

iii. The trajectory reports a full schema pass and debug/full validation to catch inconsistencies.

## 12-a. What are the most time-consuming steps of the code?

i. Reading the 405-GB spike directory, then computing per-neuron means/variances over large corridor/gray slices and training/validation are dominant.

ii. ```python
`corridor_data = plane_valid[:, corridor_mask]`
`corridor_mean = corridor_data.mean(axis=1)`
`corridor_var = corridor_data.var(axis=1)`
```

iii. The trajectory repeatedly identifies multi-gigabyte raw-file I/O and neuron scoring as the expensive full-pass work.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial `flatnonzero` rescans all session frames once per trial; per-region selection and Python membership-based fallback can also be vectorized.

ii. ```python
`for trial in range(n_trials): frame_idx = np.flatnonzero((ft_trind == trial) & retained_mask)`
`fill_mask = np.array([idx not in already for idx in all_indices])`
```

iii. No explicit trajectory justification was given; the agent prioritized correctness and bounded memory.

## 12-c. What processing does the code repeat multiple times?

i. It loads all behavior files into memory, repeatedly slices the same frame arrays per trial, scans behavior once for global speed edges and again during conversion, and explicitly invokes garbage collection per session.

ii. ```python
`speed_edges = compute_speed_edges(session_keys, canonical_behavior)`
`for trial in range(n_trials): ...`
`gc.collect()`
```

iii. The trajectory emphasizes one spike-file pass, but the cheaper behavior and trial-index work is repeated.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It creates and stores unused `behavior_files`, computes fallback scores even where responsive neurons fill caps, loads/checks duplicate behavior fields, and stores extensive source metadata not needed by the decoder. More importantly, costly neuron scoring exists only to support its non-reference cap.

ii. ```python
`behavior_files[exp_type] = beh`
`fallback_by_region[region].append((global_rows, region_scores))`
```

iii. The agent considered the extra checks and scoring worthwhile for deterministic deduplication, tractability, and validation, although downstream decoding does not consume the intermediates.


