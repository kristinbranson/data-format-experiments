# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads `Imaging_Exp_info.npy`, but only iterates the seven supervised/rewarded groups in `SUP_GROUP_PRIORITY`. It canonicalizes duplicate base recording IDs, then loads each chosen session's behavior, spike file, and retinotopy file. Thus it loads 28 rewarded-task recordings, not all 89 unique recordings in the dataset.

ii.
```python
for group_name in SUP_GROUP_PRIORITY:
    for entry in exp_info[group_name]:
        base_session_id = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
```
```python
beh = load_behavior(session["behavior_group"])[session["behavior_key"]]
spk = utils.load_spk(..., root=SPK_DIR)
iarea = np.load(os.path.join(RETINO_DIR, f"{session['subject']}_{session['date']}_trans.npz"))["iarea"]
```

iii. The trajectory says the agent narrowed scope to the rewarded task cohort because reward availability was meaningful there and the size was tractable; it explicitly chose “28 unique rewarded-task recordings.”

## 1-b. How are the data split into subjects?

i. Subjects are taken from each catalog entry's `mname`; unique names are sorted, and each retained session receives the corresponding index. This yields only the five mice in the selected rewarded cohort.

ii.
```python
"subject": entry["mname"],
subjects = sorted({session["subject"] for session in catalog})
subject_to_index = {subject: idx for idx, subject in enumerate(subjects)}
subject_idx.append(subject_to_index[session["subject"]])
```

iii. The agent recognized that the complete metadata contains 19 mice, but deliberately limited conversion to the rewarded cohort for decoder tractability and relevance of reward availability.

## 1-c. How are the data split into sessions?

i. A base session is mouse + date + block. Duplicate entries across selected supervised groups are collected, then one canonical entry is chosen according to a hard-coded group priority. Sessions are sorted by subject, date, and block.

ii.
```python
base_session_id = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
session_candidates[base_session_id].append(...)
canonical = candidates[0]
catalog.sort(key=lambda item: (item["subject"], parse_date(item["date"]), item["block"]))
```

iii. The trajectory notes that behavior files contain figure-specific aliases and `test3` may duplicate a full session as swap views; the agent therefore targeted unique base recordings, but only within its selected groups.

## 1-d. How are the data split into trials?

i. The behavior field `ntrials` defines trial count. Rather than selecting each trial's native imaging frames, the agent uses `spk_pos_interp` with cumulative position and reshapes every trial to 60 spatial bins, then keeps the first 40 bins (the 4 m corridor). Every retained trial therefore has exactly 40 columns.

ii.
```python
ntrials = int(beh["ntrials"])
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR]
for trial_idx in range(ntrials):
    session_neural.append(interp_spk[:, trial_idx, :])
```

iii. The agent believed the published `utils.spk_pos_interp` running-only, position-interpolation path was the processing to mirror and chose a fixed spatial representation that the decoder accepted.

## 1-e. How are trials filtered based on quality controls?

i. Trials are not filtered. Every index from `0` to `ntrials - 1` is emitted. Missing familiar-corridor frames or a non-finite sound position aborts the session/conversion instead of dropping a bad trial.

ii.
```python
for trial_idx in range(ntrials):
    ...
    if not np.isfinite(cue_position):
        raise ValueError(...)
```

iii. No trial-quality rationale appears in the trajectory. The agent focused quality control on rewarded-cohort selection, running-only frames, and neuron selection.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values originate from plane-concatenated deconvolved traces loaded by `utils.load_spk`. Retinotopy `iarea` assigns regions. Selection additionally depends on `ft_move`, `ft_CorrSpc`, `ft_WallID`, `UniqWalls`, and `stim_id`, from which familiar rewarded-versus-unrewarded d-prime is calculated.

ii.
```python
spk = utils.load_spk(...)
iarea = np.load(...)["iarea"]
dp = 2.0 * (stim1_mean - stim2_mean) / (stim1_std + stim2_std + 1e-12)
selected_neurons = select_neurons(dp, region_idx_all, max_neurons=max_neurons)
```

iii. The agent traced the paper's loader, interpolation, retinotopy, and d-prime functions and chose the paper's familiar rewarded-versus-unrewarded selectivity logic to keep the export trainable.

## 2-b. How is the `neural` data processed?

i. Selected traces are restricted to moving frames (`ft_move > 0`), interpolated against cumulative position into 10 cm spatial bins, cropped to the first 4 m/40 bins, cast to float16, and stored per trial.

ii.
```python
move_position = beh["ft_PosCum"][:nframes][vr_move]
move_spk = spk[selected_neurons][:, vr_move]
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
interp_spk = interp_spk[:, :, :40].astype(np.float16)
```

iii. The agent states that it was keeping the paper's “running-only and position-interpolation pipeline”; float16 and the neuron cap were used to make conversion and training tractable.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Outside-visual-cortex neurons are excluded. Visual neurons with `abs(d-prime) >= 0.3` are ranked; up to 128 per each of V1, mHV, lHV, and aHV are selected first, then remaining slots are filled by highest absolute d-prime. If fewer than 512 pass threshold, below-threshold visual neurons are used as fallback. At most 512 neurons are kept.

ii.
```python
candidate_mask = (region_idx != REGION_TO_INDEX["outside_visual_cortex"]) & (np.abs(dp) >= DP_THRESHOLD)
for region_name in AREA_SELECTION_ORDER:
    ... order[:MIN_AREA_NEURONS]
...
fallback_candidates = np.flatnonzero(region_idx != REGION_TO_INDEX["outside_visual_cortex"])
```

iii. The trajectory says raw recordings were too large for dense export; balanced stimulus-selective sampling was adopted for decoder tractability, and a small training run was used to justify the 512-neuron cap.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Arrays are described as beginning at corridor entry, but alignment is spatial rather than temporal: each trial is represented at corridor positions 0–4 m in 40 10-cm bins. It is not aligned by selecting native timestamps relative to `StartFr`.

ii.
```python
interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR]
"temporal_alignment_event": "corridor entry / trial start",
"off_start": 0.0,
```

iii. The agent interpreted the paper's spatial interpolation as the appropriate corridor-entry-aligned representation and confirmed that the validator accepted the fixed shapes.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The code reports 166.67 ms, obtained by treating each 0.1 m spatial bin as 0.1/0.6 seconds. Substantial resampling is applied: native ~315 ms imaging frames are interpolated to fixed spatial bins. The reported “time” resolution is an assumed constant-speed conversion, not the actual timing of samples.

ii.
```python
POSITION_STEP_M = 0.1
VR_SPEED_M_PER_S = 0.6
TIME_BIN_SIZE_MS = POSITION_STEP_M / VR_SPEED_M_PER_S * 1000.0
```

iii. The trajectory attributes the choice to the paper's position-interpolation pipeline. It does not justify converting spatial bins to time with a fixed 0.6 m/s beyond encoding that constant.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundPos`, not `SoundFr` and frame timestamps.

ii.
```python
sound_positions = np.asarray(beh["SoundPos"], dtype=np.float32)
cue_position = float(sound_positions[trial_idx])
```

iii. The agent inspected early task sessions for absent cues and chose the rewarded cohort where the sound-position field was usable.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Cue position is divided by 6 (the assumed VR speed in position units per second), then the constructed elapsed-time vector is subtracted. This produces positive values before and negative values after the cue.

ii.
```python
time_since_start = position_units / 6.0
cue_position / 6.0 - time_since_start
```

iii. The implied justification is consistency with the fixed spatial grid and assumed 0.6 m/s traversal, not the recorded frame times.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Both have 40 columns on the same constructed spatial-bin grid, so column alignment is exact within the agent's representation.

ii.
```python
input_trial = np.vstack([cue_position / 6.0 - time_since_start, ...])
session_neural.append(interp_spk[:, trial_idx, :])
```

iii. The agent selected a common fixed corridor grid for neural and task variables and verified matching shapes with the decoder validator.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the subject name and the `datexp` dates of retained sessions.

ii.
```python
session_date = parse_date(session["date"])
first_date = parse_date(subject_sessions[0]["date"])
session["training_day_index"] = float((session_date - first_date).days)
```

iii. The agent wanted a per-mouse training progression and retained both elapsed calendar day and session rank in metadata.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Sessions are grouped by mouse and date-sorted. The feature is calendar days elapsed since that mouse's first retained rewarded session, then repeated across all 40 bins of each trial. It is not the ordinal count of recorded training sessions.

ii.
```python
session["training_day_index"] = float((session_date - first_date).days)
np.full(40, session["training_day_index"], dtype=np.float32)
```

iii. No explicit rationale for calendar-day difference instead of recorded-session rank is given. The agent's metadata also calculates `training_day_rank`, but does not use it as the decoder input.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is not directly derived from a raw timing variable. It is constructed from the output spatial-bin indices and the fixed assumed speed.

ii.
```python
position_units = np.arange(N_POSITION_BINS_CORRIDOR, dtype=np.float32)
time_since_start = position_units / 6.0
```

iii. The agent's spatially normalized representation motivated constructing time from corridor position rather than using `StartFr` and `ft`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Bin numbers 0–39 are divided by 6, yielding 0 to 6.5 seconds at equal increments, identical for every trial irrespective of actual running speed or pauses.

ii.
```python
time_since_start = position_units / 6.0
```

iii. The only justification is the hard-coded assumed VR speed and fixed 10 cm bins.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. The constructed values share the 40-column spatial grid with interpolated neural activity.

ii.
```python
input_trial = np.vstack([..., time_since_start, ...])
session_neural.append(interp_spk[:, trial_idx, :])
```

iii. Fixed-size shared arrays passed the agent's structural checks, which it treated as confirmation of alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes from the per-trial raw field `isRew`.

ii.
```python
trial_is_rewarded = np.asarray(beh["isRew"], dtype=bool)
```

iii. The trajectory says meaningful reward availability was a major reason for restricting the export to the rewarded task cohort.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The Boolean trial value is cast to float and repeated across all 40 bins.

ii.
```python
np.full(N_POSITION_BINS_CORRIDOR, float(trial_is_rewarded[trial_idx]), dtype=np.float32)
```

iii. It is a per-trial decoder input, so the agent broadcasts it across the trial representation.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from each trial's `WallName`; `UniqWalls` and `stim_id` are separately used for neuron selection.

ii.
```python
trial_wall_names = np.asarray(beh["WallName"]).astype(str)
trial_stimulus = str(trial_wall_names[trial_idx])
```

iii. The agent regarded wall names as direct stimulus labels and gathered their union across selected sessions.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Every distinct wall name is naturally sorted and assigned its own integer category, then repeated over 40 bins. Variants such as `circle1`, `circle2`, and swaps are not collapsed to the four base textures.

ii.
```python
stimulus_names = collect_stimulus_names(catalog)
stimulus_to_index = {name: idx for idx, name in enumerate(stimulus_names)}
np.full(40, stimulus_to_index[trial_stimulus], dtype=np.int16)
```

iii. The original trajectory's user prompt gave examples such as `circle1` and `leaf2`, so the agent preserved exact wall identities. This differs from the later evaluation reference, which groups them into four broad textures.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from lick positions `LickPos` and lick trial indices `LickTrind`, rather than lick frame indices.

ii.
```python
lick_positions = np.asarray(beh["LickPos"], dtype=np.float32)
lick_trial_index = np.asarray(beh["LickTrind"], dtype=np.int64)
```

iii. The spatially interpolated representation led the agent to use spatial lick fields.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, lick positions in [0, 40) are floored to integer 10-cm bins and those bins are set to one; all other bins are zero. Multiple licks in one bin remain binary.

ii.
```python
indices = np.floor(valid_positions).astype(int)
lick_bins[np.clip(indices, 0, nbins - 1)] = 1
```

iii. Binary spatial binning matches the agent's 40-bin representation and the requested binary licking output.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks and interpolated neural activity use corresponding 10-cm corridor bins for the same trial.

ii.
```python
lick_bins = bin_licks(lick_positions[lick_trial_index == trial_idx], 40)
session_neural.append(interp_spk[:, trial_idx, :])
```

iii. The agent viewed their shared spatial grid and equal shapes as alignment.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position labels are constructed from the known 40-bin interpolated corridor grid. Raw `ft_PosCum` is used to interpolate neural data onto that grid, but raw `ft_Pos` values are not used for output labels.

ii.
```python
position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
move_position = beh["ft_PosCum"][:nframes][vr_move]
```

iii. Once the agent chose uniform 10-cm spatial interpolation, corridor position became implicit in the column index.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. A fixed vector of ten zeros, ten ones, ten twos, and ten threes is reused for every trial.

ii.
```python
position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
```

iii. Ten 10-cm bins form each requested 1-m category.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Spatial bins 0–9, 10–19, 20–29, and 30–39 are assigned categories 0–3, corresponding to 0–1 m, 1–2 m, 2–3 m, and 3–4 m.

ii.
```python
position_values = [f"{start}-{start + 1}m" for start in range(4)]
position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
```

iii. This directly implements the four equal 1-m bins requested.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Each position category labels the matching column of the neural data after spatial interpolation.

ii.
```python
interp_spk = interp_spk[:, :, :40]
output_trial = np.vstack([..., position_indices, ...])
```

iii. Alignment is intrinsic to the shared spatial-bin index.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from frame-level `ft_RunSpeed`, restricted to frames where `ft_move > 0`.

ii.
```python
move_speed = np.asarray(beh["ft_RunSpeed"][:nframes][vr_move], dtype=np.float32)[None, :]
```

iii. The agent identified running speed as directly available and processed it consistently with the running-only neural stream.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speed is spatially interpolated with `spk_pos_interp`, cropped to 40 bins, pooled across all selected sessions/trials, and discretized using three global quantiles.

ii.
```python
interp_speed = interpolate_running_signal(move_speed, move_position, corridor_length, ntrials)[0]
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75])
output_trial[3, :] = np.digitize(speed_trial, speed_edges, right=False)
```

iii. The agent used global quartiles to satisfy the requirement that each speed class correspond to 25% of the data.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three global value thresholds at the 25th, 50th, and 75th percentiles define categories 0–3 via `np.digitize`. Ties can prevent exactly equal class sizes.

ii.
```python
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75])
np.digitize(speed_trial, speed_edges, right=False)
```

iii. The agent chose literal global quartile thresholds; it did not discuss the reference's per-session stable rank split for zero-speed ties.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed and neural traces are independently passed through the same cumulative-position interpolation and cropped to the same trial and 40 columns.

ii.
```python
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
interp_speed = interpolate_running_signal(move_speed, move_position, corridor_length, ntrials)[0]
```

iii. The agent intentionally reused the paper utility and shared positional coordinate to maintain alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code slices behavior streams to neural `nframes`, ignores lick positions outside [0, 40), and uses a small denominator epsilon in d-prime. However, missing familiar-corridor frames, unexpected corridor length, or non-finite sound position raise `ValueError` and can terminate the full conversion; no trial/session recovery is implemented.

ii.
```python
nframes = spk.shape[1]
valid_positions = lick_positions[(lick_positions >= 0) & (lick_positions < nbins)]
dp = ... / (... + 1e-12)
if not np.isfinite(cue_position):
    raise ValueError(...)
```

iii. The trajectory reports that no selected session failed interpolation or selection. The agent relied on checks and aborts rather than documenting a general missing-data policy.

## 12-a. What are the most time-consuming steps of the code?

i. Loading very large spike files and running position interpolation are the dominant costs. The trajectory specifically identifies interpolation as dominant during the full 28-session conversion; conversion slowed markedly on large sessions.

ii.
```python
spk = utils.load_spk(...)
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
```

iii. The agent monitored the long full conversion and stated that interpolation was the dominant cost. Earlier, it also described the raw recordings as too large for a naive export.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop builds arrays, repeatedly masks all lick events, and appends slices even though most input/output rows can be built for all trials at once. The later nested loops assigning speed categories could also operate on stacked session arrays.

ii.
```python
for trial_idx in range(ntrials):
    lick_bins = bin_licks(lick_positions[lick_trial_index == trial_idx], 40)
    ...
for session_outputs, session_speeds in zip(output_sessions, speed_sessions):
    for output_trial, speed_trial in zip(session_outputs, session_speeds):
```

iii. The trajectory does not discuss vectorizing these loops; its optimization effort focused on limiting sessions and neurons.

## 12-c. What processing does the code repeat multiple times?

i. Behavior files are reloaded in `collect_stimulus_names` and again per session in conversion, including repeated loads of the same group file. Spatial interpolation is separately invoked for neural activity and speed. Speed arrays are stored first and traversed again after global edges are known.

ii.
```python
beh = load_behavior(session["behavior_group"])[session["behavior_key"]]  # collect_stimulus_names
beh = load_behavior(session["behavior_group"])[session["behavior_key"]]  # convert loop
interp_spk = interpolate_running_signal(...)
interp_speed = interpolate_running_signal(...)
```

iii. The agent did not justify these repetitions. It prioritized a straightforward, verifiable conversion path over minimizing repeated work.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes `training_day_rank` but uses calendar-day index as the decoder input. It records `outside_visual_cortex` as a brain-region category even though those neurons are never selected. It also creates a sample dataset and extensive counters/metadata that do not affect the full downstream decoder.

ii.
```python
session["training_day_rank"] = int(index)
GROUPED_REGIONS = ["V1", "mHV", "lHV", "aHV", "outside_visual_cortex"]
sample_data = make_sample_dataset(data, ...)
```

iii. The sample and counters were used for the agent's validation and documentation workflow, while the unused rank appears to have been retained as metadata for interpretability.
