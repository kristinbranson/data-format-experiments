# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent does not load the full dataset. It first reads `Imaging_Exp_info.npy`, but then restricts processing to the supervised `sup_*` groups listed in `SUP_GROUP_PRIORITY`. For each retained session it loads one behavior dictionary from the matching `Beh_<group>.npy`, one spike matrix via `utils.load_spk(...)`, and one retinotopy file from `retinotopy`.

ii.
```python
SUP_GROUP_PRIORITY = [
    "sup_train1_before_learning",
    "sup_train1_after_learning",
    "sup_test1",
    "sup_train2_before_learning",
    "sup_train2_after_learning",
    "sup_test2",
    "sup_test3",
]
```
```python
def load_exp_info():
    return np.load(EXP_INFO_PATH, allow_pickle=True).item()
```
```python
def load_behavior(group_name):
    path = os.path.join(BEH_DIR, f"Beh_{group_name}.npy")
    return np.load(path, allow_pickle=True).item()
```
```python
spk = utils.load_spk(
    {"mname": session["subject"], "datexp": session["date"], "blk": session["block"]},
    root=SPK_DIR,
)
iarea = np.load(
    os.path.join(RETINO_DIR, f"{session['subject']}_{session['date']}_trans.npz"),
    allow_pickle=True,
)["iarea"]
```

iii. The trajectory shows this was intentional. Step 63 says the agent "narrowed the likely conversion scope to the rewarded task cohort," step 68 says it would use "the 28 unique rewarded-task recordings," and step 170 repeats that it exported only the rewarded task sessions.

## 1-b. How are the data split into subjects?

i. Subjects are defined by `mname` from the supervised-session metadata. The final subject list is the sorted unique mouse names from the retained sessions, so only 5 mice remain.

ii.
```python
catalog.append(
    {
        "base_session_id": base_session_id,
        "behavior_group": canonical["group_name"],
        "behavior_key": canonical["behavior_key"],
        "subject": entry["mname"],
        "date": entry["datexp"],
        "block": entry["blk"],
        "source_groups": [item["group_name"] for item in candidates],
    }
)
```
```python
subjects = sorted({session["subject"] for session in catalog})
subject_to_index = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. Step 57 of the trajectory measured the retained cohort and found 28 unique supervised sessions; step 58 explicitly reports that this subset contains 5 mice. Step 170 repeats the same 5-subject cohort.

## 1-c. How are the data split into sessions?

i. A session is identified by `mouse_date_block` (`base_session_id`). The agent deduplicates repeated supervised entries by that base id and chooses one canonical behavior source using the fixed `SUP_GROUP_PRIORITY` order.

ii.
```python
base_session_id = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
behavior_key = base_session_id
if "stimtype" in entry:
    behavior_key = f"{behavior_key}_{entry['stimtype']}"
session_candidates[base_session_id].append(
    {
        "group_name": group_name,
        "behavior_key": behavior_key,
        "entry": entry,
    }
)
```
```python
candidates = sorted(
    session_candidates[base_session_id],
    key=lambda item: SUP_GROUP_PRIORITY.index(item["group_name"]),
)
canonical = candidates[0]
```

iii. Step 65 checks how duplicate supervised rows map onto canonical sessions, step 67 reports "unique sessions 28," and step 68 says the converter will use those 28 rewarded-task recordings.

## 1-d. How are the data split into trials?

i. Trials are not split from framewise trial labels such as `ft_trInd`. Instead, the code takes `beh["ntrials"]` as the trial count, uses `utils.spk_pos_interp(..., new_shape=[ntrials, 0])` to produce a trial-by-position representation, and then iterates `for trial_idx in range(ntrials)`.

ii.
```python
ntrials = int(beh["ntrials"])
```
```python
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR].astype(np.float16, copy=False)
```
```python
for trial_idx in range(ntrials):
    ...
    session_neural.append(np.ascontiguousarray(interp_spk[:, trial_idx, :]))
```

iii. The trajectory justification is the general one in steps 68, 69, 70, and 170: the agent decided to keep the paper's "running-only and position-interpolation pipeline" and to represent each trial on that interpolated corridor grid.

## 1-e. How are trials filtered based on quality controls?

i. The agent does not explicitly filter trials by length or by empty imaged windows. It keeps every `trial_idx` from `0` to `ntrials - 1`. The only trial-level check is that `SoundPos[trial_idx]` must be finite; otherwise the whole session errors. Lick positions outside the corridor bins are silently dropped.

ii.
```python
for trial_idx in range(ntrials):
    ...
    cue_position = float(sound_positions[trial_idx])
    if not np.isfinite(cue_position):
        raise ValueError(f"{session['base_session_id']}: non-finite SoundPos on trial {trial_idx}")
```
```python
def bin_licks(lick_positions, nbins):
    lick_bins = np.zeros(nbins, dtype=np.int16)
    ...
    valid_positions = lick_positions[(lick_positions >= 0) & (lick_positions < nbins)]
```

iii. No explicit trial-filtering rationale is given beyond the global shift to rewarded-task, interpolated trials. The trajectory steps 63, 68, 69, and 170 justify the cohort restriction and interpolated representation, but do not describe any reference-style trial QC.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The exported neural data comes from the spike arrays loaded by `utils.load_spk(...)`, with retinotopy `iarea` used for region labels and neuron selection.

ii.
```python
spk = utils.load_spk(
    {"mname": session["subject"], "datexp": session["date"], "blk": session["block"]},
    root=SPK_DIR,
)
iarea = np.load(
    os.path.join(RETINO_DIR, f"{session['subject']}_{session['date']}_trans.npz"),
    allow_pickle=True,
)["iarea"]
region_idx_all = grouped_region_indices(iarea)
```

iii. The trajectory repeatedly treats the paper spike file and retinotopy as the neural sources. Steps 68, 69, and 170 say the converter would reuse the paper's visual-cortex grouping and selective-neuron logic.

## 2-b. How is the `neural` data processed?

i. The neural data is heavily processed. The code restricts to running frames, computes a d-prime score between rewarded and unrewarded familiar corridors, selects up to 512 visual-cortex neurons, interpolates their activity by accumulated corridor position, crops the first 40 position bins, and stores each trial as a fixed `(512, 40)` or smaller-by-neuron matrix in `float16`.

ii.
```python
vr_move = np.asarray(beh["ft_move"][:nframes] > 0)
...
stim1_mean = np.nanmean(spk[:, stim1_frames], axis=1)
stim2_mean = np.nanmean(spk[:, stim2_frames], axis=1)
...
dp = 2.0 * (stim1_mean - stim2_mean) / (stim1_std + stim2_std + 1e-12)
selected_neurons = select_neurons(dp, region_idx_all, max_neurons=max_neurons)
```
```python
move_position = np.asarray(beh["ft_PosCum"][:nframes][vr_move], dtype=np.float64)
move_spk = np.asarray(spk[selected_neurons][:, vr_move], dtype=np.float32)
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR].astype(np.float16, copy=False)
```

iii. This is one of the clearest design choices in the trajectory. Step 54 says exporting every raw neuron looked infeasible, step 68 says the converter will use the paper's "running-only and position-interpolation pipeline" and a balanced set of selective neurons, step 69 says the remaining design problem was tractability, and step 170 summarizes the same 512-neuron interpolated export.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered to grouped visual-cortex regions and then further curated by d-prime magnitude. The code prefers neurons with `|d'| >= 0.3`, tries to balance across `V1`, `mHV`, `lHV`, and `aHV`, and caps each session at 512 neurons. Neurons outside those four grouped regions are excluded.

ii.
```python
candidate_mask = (region_idx != REGION_TO_INDEX["outside_visual_cortex"]) & (np.abs(dp) >= DP_THRESHOLD)
```
```python
for region_name in AREA_SELECTION_ORDER:
    region_code = REGION_TO_INDEX[region_name]
    region_candidates = np.flatnonzero(candidate_mask & (region_idx == region_code))
    ...
```
```python
selected = np.asarray(selected[:max_neurons], dtype=np.int32)
selected = selected[np.argsort(abs_dp[selected])[::-1]]
return np.sort(selected)
```

iii. Steps 68, 69, and 170 explicitly justify this as a tractability choice: use the paper's d-prime selectivity logic and grouped visual areas to curate a fixed-size, balanced neuron set that the decoder can train on.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent says trials are aligned to corridor entry / trial start, but the actual per-trial representation is a fixed 40-bin interpolated corridor trajectory rather than the original framewise trial window. Neural activity is therefore aligned to a standardized spatial trajectory from the start of the 4 m corridor.

ii.
```python
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR].astype(np.float16, copy=False)
```
```python
"temporal_alignment_event": "corridor entry / trial start",
"off_start": 0.0,
"off_end": float(N_POSITION_BINS_CORRIDOR / 6.0),
```

iii. Step 68 says the converter would keep the paper's position-interpolation pipeline while aligning trials to corridor entry, and step 170 repeats that it aligned trials to corridor entry but kept the 4 m corridor as 40 bins.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent sets a derived bin size of `166.666...` ms by assuming each 10 cm corridor bin corresponds to `0.1 / 0.6` seconds. Yes: the data is rebinned by spatial interpolation onto 40 fixed bins per trial.

ii.
```python
POSITION_STEP_M = 0.1
VR_SPEED_M_PER_S = 0.6
TIME_BIN_SIZE_MS = POSITION_STEP_M / VR_SPEED_M_PER_S * 1000.0
```
```python
"time_bin_size": TIME_BIN_SIZE_MS,
```

iii. Step 68 and step 170 justify the use of the paper's position-interpolation pipeline and a fixed 40-bin corridor representation; the explicit 166.7 ms interpretation follows from that fixed-speed assumption in the code.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundPos`, not `SoundFr` and `ft`. The code converts cue position to time by dividing position by a fixed speed.

ii.
```python
sound_positions = np.asarray(beh["SoundPos"], dtype=np.float32)
...
cue_position = float(sound_positions[trial_idx])
```
```python
cue_position / 6.0 - time_since_start
```

iii. The trajectory does not discuss this variable separately. The closest justification is the general one in steps 68, 69, and 170: after deciding to use the running-only position-interpolation pipeline, the agent expressed trial timing on that same fixed corridor grid.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the code treats `SoundPos` as a cue location along the corridor, converts it to seconds by dividing by `6.0`, and subtracts the per-bin `time_since_start` vector. This yields positive values before the cue and negative values after the cue on the 40-bin grid.

ii.
```python
time_since_start = position_units / 6.0
```
```python
input_trial = np.vstack(
    [
        cue_position / 6.0 - time_since_start,
        ...
    ]
).astype(np.float32, copy=False)
```

iii. No separate variable-level reasoning is recorded. The justification visible in steps 68 and 170 is that the agent wanted all streams on the same interpolated corridor representation.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned by construction to the same 40 interpolated corridor bins used for the neural data in that trial.

ii.
```python
session_neural.append(np.ascontiguousarray(interp_spk[:, trial_idx, :]))
```
```python
input_trial = np.vstack(
    [
        cue_position / 6.0 - time_since_start,
        ...
    ]
).astype(np.float32, copy=False)
```

iii. Steps 68 and 170 justify using a single interpolated trial grid for neural and behavioral variables.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from each session's subject id and recording date in the session catalog. The code computes both a date difference from the subject's first retained supervised session and a rank order across that subject's retained supervised sessions.

ii.
```python
subject_sessions.sort(key=lambda item: parse_date(item["date"]))
first_date = parse_date(subject_sessions[0]["date"])
for index, session in enumerate(subject_sessions):
    session_date = parse_date(session["date"])
    session["training_day_index"] = float((session_date - first_date).days)
    session["training_day_rank"] = int(index)
```

iii. The trajectory does not discuss this variable by itself. The retained rationale is the same rewarded-task cohort restriction from steps 63, 68, and 170; the code then measures training day within that restricted cohort.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The exported input uses `training_day_index`, which is elapsed calendar days since the mouse's first retained supervised session, not session count. That scalar is repeated across all 40 bins of every trial in the session.

ii.
```python
np.full(N_POSITION_BINS_CORRIDOR, session["training_day_index"], dtype=np.float32)
```

iii. No explicit justification for choosing elapsed days over session count appears in the trajectory. The general justification is only that the conversion is built around the supervised rewarded-session catalog.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is not derived from `StartFr` and frame timestamps. Instead it is derived from the fixed corridor-bin grid: `position_units = 0..39` and the fixed conversion factor `1 / 6.0` seconds per 10 cm.

ii.
```python
position_units = np.arange(N_POSITION_BINS_CORRIDOR, dtype=np.float32)
time_since_start = position_units / 6.0
```

iii. The trajectory does not discuss this variable separately. Steps 68 and 170 show the general decision to use a 40-bin corridor representation rather than framewise timing.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The code creates a fixed vector from `0` to `39 / 6` seconds and reuses it for every trial, assuming constant progression through the corridor bins.

ii.
```python
time_since_start = position_units / 6.0
```
```python
input_trial = np.vstack(
    [
        cue_position / 6.0 - time_since_start,
        np.full(N_POSITION_BINS_CORRIDOR, session["training_day_index"], dtype=np.float32),
        time_since_start,
        ...
    ]
).astype(np.float32, copy=False)
```

iii. No dedicated explanation is recorded in the trajectory; this follows from the fixed-bin interpolated representation justified in steps 68, 69, and 170.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned exactly to the same 40 interpolated corridor bins used for the neural data of each trial.

ii.
```python
session_neural.append(np.ascontiguousarray(interp_spk[:, trial_idx, :]))
session_input.append(np.ascontiguousarray(input_trial))
```

iii. The trajectory justification is the same single-grid alignment choice described in steps 68 and 170.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from `isRew` at the trial level.

ii.
```python
trial_is_rewarded = np.asarray(beh["isRew"], dtype=bool)
```

iii. Step 63 says the agent restricted the conversion to the rewarded-task cohort because `reward availability` is meaningful there, and step 170 repeats that rationale.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No additional transformation is applied beyond casting the trial-level boolean to float and repeating it across the 40 bins of the trial.

ii.
```python
np.full(
    N_POSITION_BINS_CORRIDOR,
    float(trial_is_rewarded[trial_idx]),
    dtype=np.float32,
)
```

iii. The trajectory justification is again cohort-level rather than variable-level: steps 63 and 170 say the supervised rewarded cohort was chosen so this input would vary meaningfully.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName` at the trial level, while the list of possible categories is collected from `UniqWalls` across retained sessions.

ii.
```python
trial_wall_names = np.asarray(beh["WallName"]).astype(str)
...
trial_stimulus = str(trial_wall_names[trial_idx])
```
```python
stimulus_names.update(map(str, beh["UniqWalls"]))
```

iii. The trajectory does not justify this variable separately. Step 67 explicitly tallies the retained stimulus names from the supervised cohort, and step 170 reports those fine-grained stimulus categories in the final export.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The code does not collapse textures into four broad categories. Instead it enumerates all retained raw wall names, maps each name to an index, and repeats that index across the 40 bins of the trial.

ii.
```python
stimulus_names = collect_stimulus_names(catalog)
stimulus_to_index = {name: idx for idx, name in enumerate(stimulus_names)}
```
```python
output_trial = np.vstack(
    [
        np.full(
            N_POSITION_BINS_CORRIDOR,
            stimulus_to_index[trial_stimulus],
            dtype=np.int16,
        ),
        ...
    ]
)
```

iii. Step 67 shows the agent explicitly gathering the retained stimulus set, and step 170 reports the final dataset as containing the fine-grained categories such as `circle1`, `leaf1_swap1`, and `wood5`.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickPos` and `LickTrind`, not from `LickFr`.

ii.
```python
lick_positions = np.asarray(beh["LickPos"], dtype=np.float32)
lick_trial_index = np.asarray(beh["LickTrind"], dtype=np.int64)
```

iii. The trajectory does not call out licking separately. The nearest justification is the same one in steps 68 and 170: behavioral outputs were put onto the same interpolated corridor-bin representation as the neural data.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the code selects lick positions with the matching `LickTrind`, floors each lick position to a corridor bin, clips to the 40-bin corridor, and marks each occupied bin as `1`.

ii.
```python
def bin_licks(lick_positions, nbins):
    lick_bins = np.zeros(nbins, dtype=np.int16)
    ...
    indices = np.floor(valid_positions).astype(int)
    lick_bins[np.clip(indices, 0, nbins - 1)] = 1
    return lick_bins
```
```python
lick_bins = bin_licks(lick_positions[lick_trial_index == trial_idx], N_POSITION_BINS_CORRIDOR)
```

iii. No explicit variable-specific reasoning is recorded in the trajectory; this is another consequence of the corridor-bin representation justified in steps 68 and 170.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned to the neural data through the same 40 interpolated corridor bins used for the trial's neural matrix.

ii.
```python
session_neural.append(np.ascontiguousarray(interp_spk[:, trial_idx, :]))
...
output_trial = np.vstack(
    [
        ...,
        lick_bins,
        ...
    ]
)
```

iii. Steps 68 and 170 justify using one shared interpolated corridor grid for both neural and behavioral variables.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. The exported categorical position output is not taken directly from a raw per-frame position variable. After the neural data is interpolated across a 40-bin corridor grid, position is represented by fixed bin indices `0,1,2,3` repeated in blocks of 10 bins.

ii.
```python
N_POSITION_BINS_CORRIDOR = 40
position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
```
```python
output_trial = np.vstack(
    [
        ...,
        position_indices,
        ...
    ]
)
```

iii. The trajectory justification is again the fixed corridor-grid design stated in steps 68 and 170.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The code hard-codes four 1 m bins over the 40-bin interpolated corridor. No per-frame `ft_Pos` values are used at export time.

ii.
```python
position_values = [f"{start}-{start + 1}m" for start in range(4)]
position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
```

iii. The trajectory justification is the same choice to keep the paper's corridor interpolation while exporting only the first 4 m as 40 bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The thresholding is deterministic: bins `0-9` are category 0, `10-19` category 1, `20-29` category 2, and `30-39` category 3.

ii.
```python
position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
```

iii. No additional justification is recorded beyond the fixed 40-bin corridor window described in steps 68 and 170.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. It is aligned by sharing the same 40 interpolated corridor bins as the neural data for each trial.

ii.
```python
session_neural.append(np.ascontiguousarray(interp_spk[:, trial_idx, :]))
...
output_trial = np.vstack(
    [
        ...,
        position_indices,
        ...
    ]
)
```

iii. The trajectory justification is the shared corridor-grid representation from steps 68 and 170.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed`, after first restricting to running frames and interpolating by accumulated corridor position.

ii.
```python
move_speed = np.asarray(beh["ft_RunSpeed"][:nframes][vr_move], dtype=np.float32)[None, :]
interp_speed = interpolate_running_signal(move_speed, move_position, corridor_length, ntrials)[0]
```

iii. The trajectory justification is the same running-only, position-interpolation plan stated in steps 68, 69, and 170.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is interpolated onto the 40 corridor bins for each trial, concatenated across all retained sessions, and then discretized using global quartile edges from the full converted dataset.

ii.
```python
interp_speed = interpolate_running_signal(move_speed, move_position, corridor_length, ntrials)[0]
interp_speed = interp_speed[:, :N_POSITION_BINS_CORRIDOR].astype(np.float32, copy=False)
all_speed_values.append(interp_speed.reshape(-1))
```
```python
speed_values = np.concatenate(all_speed_values)
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
```

iii. No separate speed-specific reasoning appears in the trajectory. The recorded justification is the general decision to use the interpolated corridor representation and then make the exported decoder task trainable.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The thresholding uses the three global numeric quartile cutoffs in `speed_edges`, then `np.digitize(...)` assigns category `0-3`.

ii.
```python
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
```
```python
output_trial[3, :] = np.digitize(speed_trial, speed_edges, right=False).astype(np.int16)
```

iii. No explicit justification is recorded beyond the desire to export a quartile-binned running-speed variable on the shared corridor grid.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned to the neural data through the same interpolated 40-bin corridor grid used for the trial neural matrices.

ii.
```python
session_neural.append(np.ascontiguousarray(interp_spk[:, trial_idx, :]))
session_speed.append(np.ascontiguousarray(interp_speed[trial_idx]))
...
output_trial[3, :] = np.digitize(speed_trial, speed_edges, right=False).astype(np.int16)
```

iii. Steps 68 and 170 justify using one interpolated corridor grid for all streams.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Minor issues are handled only partially. The code truncates several behavior arrays to the imaged frame count `nframes`, drops lick positions that fall outside the 40 corridor bins, and rejects sessions with missing familiar corridor frames or a non-finite cue position by raising an exception. There is no reference-style handling of empty or overlong trials.

ii.
```python
nframes = spk.shape[1]
vr_move = np.asarray(beh["ft_move"][:nframes] > 0)
corridor_frames = np.asarray(beh["ft_CorrSpc"][:nframes], dtype=bool)
```
```python
if stim1_frames.sum() == 0 or stim2_frames.sum() == 0:
    raise ValueError(f"{session['base_session_id']}: missing familiar corridor frames for d' selection")
```
```python
if not np.isfinite(cue_position):
    raise ValueError(f"{session['base_session_id']}: non-finite SoundPos on trial {trial_idx}")
```

iii. The trajectory does not present this as a separate data-cleaning policy. The only explicit rationale is step 54 and step 69, where the agent focuses on tractability and selective-neuron export rather than on reference-style data QC.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading the large spike files for each session and then running the extra per-session interpolation and d-prime neuron-selection pipeline. The behavior files are also reloaded repeatedly.

ii.
```python
beh = load_behavior(session["behavior_group"])[session["behavior_key"]]
...
spk = utils.load_spk(
    {"mname": session["subject"], "datexp": session["date"], "blk": session["block"]},
    root=SPK_DIR,
)
```
```python
stim1_mean = np.nanmean(spk[:, stim1_frames], axis=1)
...
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
```

iii. The trajectory repeatedly treats conversion as expensive because of neural-data scale and extra processing. Step 54 says memory rules out exporting every raw neuron, step 69 says tractability is the main design issue, and steps 91, 108, 131, and 158 describe the full conversion and training runs as the expensive stages.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop that builds `input_trial`, `output_trial`, and appends trial arrays could have been vectorized more aggressively. The per-trial `bin_licks(...)` work and the second pass that writes running-speed categories are also obvious vectorization targets.

ii.
```python
for trial_idx in range(ntrials):
    ...
    lick_bins = bin_licks(lick_positions[lick_trial_index == trial_idx], N_POSITION_BINS_CORRIDOR)
    input_trial = np.vstack([...]).astype(np.float32, copy=False)
    output_trial = np.vstack([...])
    ...
```
```python
for session_outputs, session_speeds in zip(output_sessions, speed_sessions):
    for output_trial, speed_trial in zip(session_outputs, session_speeds):
        output_trial[3, :] = np.digitize(speed_trial, speed_edges, right=False).astype(np.int16)
```

iii. No explicit vectorization discussion appears in the trajectory. This looks like an implementation consequence rather than a separately justified design choice.

## 12-c. What processing does the code repeat multiple times?

i. The code reloads behavior files session-by-session even though multiple sessions can share a source file, and it computes/uses some dataset-wide summaries in a second pass after building all trials. It also makes one full pass over the catalog just to collect stimulus names, then another full pass to actually convert the sessions.

ii.
```python
def collect_stimulus_names(catalog):
    stimulus_names = set()
    for session in catalog:
        beh = load_behavior(session["behavior_group"])[session["behavior_key"]]
        stimulus_names.update(map(str, beh["UniqWalls"]))
    return sorted(stimulus_names, key=natural_sort_key)
```
```python
for session in catalog:
    ...
    beh = load_behavior(session["behavior_group"])[session["behavior_key"]]
```
```python
for session_outputs, session_speeds in zip(output_sessions, speed_sessions):
    for output_trial, speed_trial in zip(session_outputs, session_speeds):
        output_trial[3, :] = np.digitize(speed_trial, speed_edges, right=False).astype(np.int16)
```

iii. The trajectory does not explicitly justify these repeated passes. They appear to be incidental consequences of the implementation.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores several quantities that are not needed by downstream decoder training: `training_day_rank`, `source_groups`, session-wide neuron-count summaries, and temporary `speed_sessions` used only for the later second pass. It also builds a separate `sample_data.pkl`, which is not part of the required full-dataset artifact.

ii.
```python
session["training_day_index"] = float((session_date - first_date).days)
session["training_day_rank"] = int(index)
```
```python
"source_groups": [item["group_name"] for item in candidates],
```
```python
speed_sessions = []
selected_neuron_counts = []
total_neuron_counts = []
stimulus_counts = defaultdict(int)
```

iii. No explicit justification for these extras is recorded in the trajectory, aside from the general desire to create notes, sample exports, and summary statistics for validation.
