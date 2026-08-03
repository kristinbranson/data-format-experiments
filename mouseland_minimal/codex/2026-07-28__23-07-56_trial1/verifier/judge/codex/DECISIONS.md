# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not load the full dataset. It first reads `Imaging_Exp_info.npy`, then restricts itself to the supervised task-session groups listed in `SUP_GROUP_PRIORITY`. For those sessions it loads behavior from `Beh_<group>.npy`, neural traces through `utils.load_spk(...)`, and retinotopy from the corresponding `*_trans.npz` file.

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
beh = load_behavior(session["behavior_group"])[session["behavior_key"]]
spk = utils.load_spk(
    {"mname": session["subject"], "datexp": session["date"], "blk": session["block"]},
    root=SPK_DIR,
)
iarea = np.load(
    os.path.join(RETINO_DIR, f"{session['subject']}_{session['date']}_trans.npz"),
    allow_pickle=True,
)["iarea"]
```

iii. The justification in `CONVERSION_NOTES.md` is that only rewarded task sessions should be exported because the decoder input `reward_availability` should remain meaningful; the notes say unsupervised and naive cohorts would make that variable collapse.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are taken from `entry["mname"]` while building the session catalog. After filtering to supervised sessions, the converter forms `subjects` as the sorted unique mouse names and stores `subject_idx` from that filtered set.

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

iii. The notes explicitly state this filtered export yields 5 mice, because the AI chose the rewarded task cohort only.

## 1-c. How are the data split into sessions?

i. A session is defined as `mouse_date_block` (`base_session_id`). The AI only considers sessions appearing in the supervised priority list, gathers all candidate rows for each `base_session_id`, and keeps one canonical row per session according to `SUP_GROUP_PRIORITY`.

ii. 
```python
base_session_id = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
```
```python
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

iii. `CONVERSION_NOTES.md` says duplicate supervised-analysis rows were deduplicated by recording id and that the priority order chooses the canonical representation.

## 1-d. How are the data split into trials?

i. Trials are not defined from frame-level trial windows. The AI reads `ntrials` from behavior, interpolates neural and speed signals into per-trial corridor bins, and then iterates over `range(ntrials)` to build one trial per trial index. All trials from each selected session are retained.

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

iii. The justification in the notes is that the export should follow the paper’s running-only spatial interpolation and then use the first 40 bins of the 4 m corridor.

## 1-e. How are trials filtered based on quality controls?

i. The converter applies no explicit per-trial quality-control filter. It keeps every `trial_idx` from `0` to `ntrials - 1`; failures are handled at the session level instead, for example by raising if cue position is non-finite or if the session lacks familiar rewarded/unrewarded corridor frames needed for d-prime selection.

ii. 
```python
for trial_idx in range(ntrials):
    cue_position = float(sound_positions[trial_idx])
    if not np.isfinite(cue_position):
        raise ValueError(f"{session['base_session_id']}: non-finite SoundPos on trial {trial_idx}")
```
```python
if stim1_frames.sum() == 0 or stim2_frames.sum() == 0:
    raise ValueError(f"{session['base_session_id']}: missing familiar corridor frames for d' selection")
```

iii. No explicit per-trial filtering justification is given beyond the broader decision to keep all rewarded-task trials on the interpolated corridor grid.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the session spike traces returned by `utils.load_spk(...)`, which reads the deconvolved imaging data, and retinotopy labels come from `iarea` in the matching `*_trans.npz` file.

ii. 
```python
spk = utils.load_spk(
    {"mname": session["subject"], "datexp": session["date"], "blk": session["block"]},
    root=SPK_DIR,
)
```
```python
iarea = np.load(
    os.path.join(RETINO_DIR, f"{session['subject']}_{session['date']}_trans.npz"),
    allow_pickle=True,
)["iarea"]
```

iii. `CONVERSION_NOTES.md` says the converter uses the deconvolved Suite2p traces and retinotopy assignments from the provided data sources.

## 2-b. How is the `neural` data processed?

i. The AI first restricts to running frames, computes a d-prime score per neuron from familiar rewarded versus familiar unrewarded corridor frames, selects up to 512 visual-cortex neurons, then spatially interpolates the selected traces into 60 corridor bins and truncates to the first 40 bins. The final stored trials are contiguous `float16` arrays.

ii. 
```python
vr_move = np.asarray(beh["ft_move"][:nframes] > 0)
valid_corridor_frames = vr_move & corridor_frames
```
```python
dp = 2.0 * (stim1_mean - stim2_mean) / (stim1_std + stim2_std + 1e-12)
selected_neurons = select_neurons(dp, region_idx_all, max_neurons=max_neurons)
```
```python
move_spk = np.asarray(spk[selected_neurons][:, vr_move], dtype=np.float32)
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR].astype(np.float16, copy=False)
```

iii. The notes justify this as making the dataset trainable while “keeping aligned with the paper’s analysis logic,” specifically reusing running-only interpolation and a d-prime threshold used elsewhere in the paper code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are assigned to grouped retinotopic regions and any neuron outside `V1`, `mHV`, `lHV`, or `aHV` is excluded from candidate selection. Among the remaining neurons, the AI prefers those with `|d'| >= 0.3`, balances selection across the four regions, and caps each session at 512 neurons.

ii. 
```python
GROUPED_REGIONS = ["V1", "mHV", "lHV", "aHV", "outside_visual_cortex"]
DP_THRESHOLD = 0.3
MAX_NEURONS = 512
```
```python
candidate_mask = (region_idx != REGION_TO_INDEX["outside_visual_cortex"]) & (np.abs(dp) >= DP_THRESHOLD)
```
```python
for region_name in AREA_SELECTION_ORDER:
    region_code = REGION_TO_INDEX[region_name]
    region_candidates = np.flatnonzero(candidate_mask & (region_idx == region_code))
```

iii. `CONVERSION_NOTES.md` gives the full rationale: retain grouped visual-cortex neurons, prefer neurons above the paper’s selective-neuron d-prime threshold, and cap to 512 for tractability with the provided decoder.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI says trials are aligned to corridor entry / trial start, but the actual per-trial representation is a spatially interpolated corridor grid rather than original imaging frames. Neural traces are expressed on the first 40 bins of the 4 m corridor.

ii. 
```python
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR].astype(np.float16, copy=False)
```
```python
"temporal_alignment_event": "corridor entry / trial start",
```

iii. The notes justify this as matching corridor entry alignment while also using the paper’s running-only spatial interpolation and the requested 4 x 1 m position output.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 40 bins of 10 cm each and treats them as 166.6667 ms bins by assuming a fixed VR speed of 60 cm/s. This is effectively a resampling from frame time into a spatial grid with an implied time axis.

ii. 
```python
POSITION_STEP_M = 0.1
VR_SPEED_M_PER_S = 0.6
TIME_BIN_SIZE_MS = POSITION_STEP_M / VR_SPEED_M_PER_S * 1000.0
```
```python
"time_bin_size": TIME_BIN_SIZE_MS,
```

iii. `CONVERSION_NOTES.md` states that the reference interpolation grid is 10 cm bins and that the implied time bin is computed from the fixed VR speed.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The AI derives this variable from `SoundPos`, not from frame times. It converts sound-cue position in the corridor into an implied time by dividing by the constant VR speed.

ii. 
```python
sound_positions = np.asarray(beh["SoundPos"], dtype=np.float32)
```
```python
cue_position = float(sound_positions[trial_idx])
```

iii. The notes describe this input as “signed cue time minus current bin time” on the 40-bin grid, implying the position-to-time conversion via constant corridor speed.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the AI converts cue position to seconds with `cue_position / 6.0` and subtracts the per-bin `time_since_start` vector. The result is a dense time-varying signal over 40 corridor bins.

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

iii. The notes justify this as a signed “cue time minus current bin time” representation.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned by using the same 40-bin interpolated corridor grid as the neural data. Each trial’s cue-time input and neural activity share identical bin count and ordering.

ii. 
```python
session_neural.append(np.ascontiguousarray(interp_spk[:, trial_idx, :]))
session_input.append(np.ascontiguousarray(input_trial))
```

iii. The notes repeatedly justify alignment through the shared 10 cm running-only corridor grid.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. This input is derived from the session metadata fields `subject` and `date` created from `mname` and `datexp` in the experiment index.

ii. 
```python
session["subject"] = entry["mname"]
session["date"] = entry["datexp"]
```
```python
session_date = parse_date(session["date"])
```

iii. The notes say this variable is “elapsed days since the first rewarded task imaging session for that mouse.”

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Sessions are grouped by mouse, sorted by calendar date, and each session gets both a rank and an elapsed-day value from the first session. The exported input uses `training_day_index`, which is actual elapsed days, broadcast across all 40 bins of each trial.

ii. 
```python
first_date = parse_date(subject_sessions[0]["date"])
for index, session in enumerate(subject_sessions):
    session_date = parse_date(session["date"])
    session["training_day_index"] = float((session_date - first_date).days)
    session["training_day_rank"] = int(index)
```
```python
np.full(N_POSITION_BINS_CORRIDOR, session["training_day_index"], dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` explicitly justifies this as elapsed days since the first rewarded task imaging session for that mouse.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. The AI does not derive this from frame timestamps or `StartFr`. Instead it constructs a fixed vector from corridor-bin index and the assumed VR speed, so the effective ingredients are the 40-bin corridor grid plus the 60 cm/s constant.

ii. 
```python
position_units = np.arange(N_POSITION_BINS_CORRIDOR, dtype=np.float32)
time_since_start = position_units / 6.0
```

iii. The notes describe the exported value as “0 to 6.5 s on the 40-bin grid.”

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The AI creates a single session-independent vector `0, 1/6, 2/6, ...` seconds for the 40 corridor bins and reuses it for every trial. No raw per-trial timing field is consulted.

ii. 
```python
time_since_start = position_units / 6.0
```
```python
input_trial = np.vstack(
    [
        ...,
        time_since_start,
        ...
    ]
).astype(np.float32, copy=False)
```

iii. The justification in the notes is the fixed-speed, 10 cm corridor-bin representation.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned by construction to the same 40-bin spatially interpolated corridor grid used for neural data.

ii. 
```python
session_neural.append(np.ascontiguousarray(interp_spk[:, trial_idx, :]))
session_input.append(np.ascontiguousarray(input_trial))
```

iii. The notes justify this through the shared interpolated corridor-bin representation.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Reward availability is derived directly from the per-trial behavior field `isRew`.

ii. 
```python
trial_is_rewarded = np.asarray(beh["isRew"], dtype=bool)
```

iii. The notes summarize this as “1 for rewarded familiar corridors, 0 otherwise.”

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The AI converts `isRew` to float and broadcasts the per-trial value across all 40 bins of the trial input matrix.

ii. 
```python
np.full(
    N_POSITION_BINS_CORRIDOR,
    float(trial_is_rewarded[trial_idx]),
    dtype=np.float32,
),
```

iii. The justification is only implicit: the target format expects time-varying inputs aligned to the neural bins, so the per-trial reward flag is repeated across the trial.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The AI derives this output from the trial-level `WallName` strings and from the set of unique wall names collected across the selected sessions.

ii. 
```python
trial_wall_names = np.asarray(beh["WallName"]).astype(str)
```
```python
stimulus_names.update(map(str, beh["UniqWalls"]))
```

iii. The notes justify this by listing the actual stimulus categories present in the rewarded-task export.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI keeps each distinct wall-name string as its own category rather than collapsing to four base textures. It builds `stimulus_to_index` from sorted unique names and broadcasts the resulting category index across all 40 bins of each trial.

ii. 
```python
stimulus_names = collect_stimulus_names(catalog)
stimulus_to_index = {name: idx for idx, name in enumerate(stimulus_names)}
```
```python
np.full(
    N_POSITION_BINS_CORRIDOR,
    stimulus_to_index[trial_stimulus],
    dtype=np.int16,
),
```

iii. `CONVERSION_NOTES.md` explicitly lists 14 stimulus categories present and treats those wall-name variants as the exported classes.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `LickPos` and `LickTrind`, not from lick frame numbers. The AI groups lick positions by trial and bins them into corridor bins.

ii. 
```python
lick_positions = np.asarray(beh["LickPos"], dtype=np.float32)
lick_trial_index = np.asarray(beh["LickTrind"], dtype=np.int64)
```

iii. The notes explicitly say licking is “binary per 10 cm bin, derived from `LickPos` within the corridor.”

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the AI selects the lick positions whose `LickTrind` matches that trial, drops positions outside `[0, nbins)`, floors the remaining positions to integer bins, and marks those bins as 1. The result is a 40-bin binary vector.

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

iii. The notes justify this as a binary per-bin licking output on the same 10 cm corridor grid as the neural data.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. It is aligned by trial index and by the same 40-bin interpolated corridor grid used for neural data. Both licking and neural activity are stored on identical per-trial bin axes.

ii. 
```python
session_neural.append(np.ascontiguousarray(interp_spk[:, trial_idx, :]))
session_output.append(np.ascontiguousarray(output_trial))
```

iii. The notes justify this by describing the full export as a shared 10 cm corridor-bin representation.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. The AI does not use raw per-frame `ft_Pos` directly for the exported category values. Instead it constructs a fixed 40-bin corridor template and treats those bins themselves as position.

ii. 
```python
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

iii. The notes justify this as using the first 40 bins of the 4 m corridor and then collapsing them into four 1 m categories.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The output is a fixed vector of ten bins labeled 0, then ten labeled 1, then ten labeled 2, then ten labeled 3. This means every trial shares the same position-category trajectory.

ii. 
```python
position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
```

iii. The notes explicitly describe this as “four 1 m bins across the 4 m corridor.”

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Category thresholds are implicit in the fixed template: bins 0-9 are category 0, bins 10-19 category 1, bins 20-29 category 2, and bins 30-39 category 3.

ii. 
```python
position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
position_values = [f"{start}-{start + 1}m" for start in range(4)]
```

iii. The notes justify these as the requested four equal 1 m bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. It is aligned by sharing the same 40-bin corridor axis as the interpolated neural data.

ii. 
```python
session_neural.append(np.ascontiguousarray(interp_spk[:, trial_idx, :]))
session_output.append(np.ascontiguousarray(output_trial))
```

iii. The notes justify all time-varying variables as living on the same 10 cm running-only corridor grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ft_RunSpeed` on running frames only.

ii. 
```python
move_speed = np.asarray(beh["ft_RunSpeed"][:nframes][vr_move], dtype=np.float32)[None, :]
```

iii. The notes say the converter reuses running-only frames and position interpolation from the paper code.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI first interpolates running speed onto the 60-bin corridor grid, truncates to the first 40 bins, concatenates all interpolated speed bins across all selected sessions, computes global quartile edges, and later digitizes each trial’s speed bins with those global thresholds.

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

iii. `CONVERSION_NOTES.md` explicitly says running-speed bins are quartiles computed over all interpolated corridor time bins in the full export.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Thresholds are the three global quantile edges of the concatenated interpolated speed values. `np.digitize(..., right=False)` converts each speed bin into categories 0-3.

ii. 
```python
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
```
```python
output_trial[3, :] = np.digitize(speed_trial, speed_edges, right=False).astype(np.int16)
```

iii. The notes justify this as producing quartile-based running-speed bins on the exported dataset.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned to neural activity through the same interpolated 40-bin corridor grid.

ii. 
```python
session_speed.append(np.ascontiguousarray(interp_speed[trial_idx]))
...
output_trial[3, :] = np.digitize(speed_trial, speed_edges, right=False).astype(np.int16)
```

iii. The notes justify this as part of the shared running-only position-interpolated representation.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code truncates frame-level behavior arrays to `nframes = spk.shape[1]`, ignores lick positions outside the corridor bins, and raises hard errors for non-finite `SoundPos` values or missing familiar rewarded/unrewarded corridor frames needed for neuron selection. It also relies on boolean slicing rather than explicit repair for many issues.

ii. 
```python
nframes = spk.shape[1]
vr_move = np.asarray(beh["ft_move"][:nframes] > 0)
corridor_frames = np.asarray(beh["ft_CorrSpc"][:nframes], dtype=bool)
wall_ids = np.asarray(beh["ft_WallID"][:nframes])
```
```python
valid_positions = lick_positions[(lick_positions >= 0) & (lick_positions < nbins)]
```
```python
if stim1_frames.sum() == 0 or stim2_frames.sum() == 0:
    raise ValueError(...)
if not np.isfinite(cue_position):
    raise ValueError(...)
```

iii. There is no dedicated data-cleaning discussion in the notes beyond the assumption that the selected task cohort and paper-style interpolation are clean enough; the explicit justifications are session-selection and neuron-selection based.

## 12-a. What are the most time-consuming steps of the code?

i. The expensive steps are loading the large spike arrays, computing per-neuron d-prime statistics over many frames, interpolating spikes and speed onto corridor bins, and then iterating through every trial to assemble outputs.

ii. 
```python
spk = utils.load_spk(
    {"mname": session["subject"], "datexp": session["date"], "blk": session["block"]},
    root=SPK_DIR,
)
```
```python
stim1_mean = np.nanmean(spk[:, stim1_frames], axis=1)
stim2_mean = np.nanmean(spk[:, stim2_frames], axis=1)
...
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
```
```python
for trial_idx in range(ntrials):
    ...
```

iii. The notes justify these extra costs as necessary to keep the dataset tractable for decoder training while following analysis logic from the paper.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities are the per-trial loop that repeatedly constructs `input_trial` and `output_trial`, the repeated `lick_trial_index == trial_idx` mask inside that loop, and the region-balancing loop in `select_neurons`.

ii. 
```python
for trial_idx in range(ntrials):
    ...
    lick_bins = bin_licks(lick_positions[lick_trial_index == trial_idx], N_POSITION_BINS_CORRIDOR)
```
```python
for region_name in AREA_SELECTION_ORDER:
    region_code = REGION_TO_INDEX[region_name]
    region_candidates = np.flatnonzero(candidate_mask & (region_idx == region_code))
```

iii. No explicit efficiency justification is given beyond prioritizing a working conversion pipeline.

## 12-c. What processing does the code repeat multiple times?

i. The code reloads behavior during `collect_stimulus_names(...)` and again during conversion, interpolates spikes and speed in two separate calls over the same running-position axis, and repeatedly materializes per-trial lick masks inside the trial loop.

ii. 
```python
for session in catalog:
    beh = load_behavior(session["behavior_group"])[session["behavior_key"]]
    stimulus_names.update(map(str, beh["UniqWalls"]))
```
```python
beh = load_behavior(session["behavior_group"])[session["behavior_key"]]
```
```python
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
interp_speed = interpolate_running_signal(move_speed, move_position, corridor_length, ntrials)[0]
```

iii. No explicit justification is given for the repeated work; it appears to be a straightforward implementation choice.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores `training_day_rank` but exports `training_day_index` instead; it accumulates `speed_sessions` only so it can digitize outputs later and then discards the continuous speeds; and it loads behavior once just to gather stimulus names before the main conversion pass. It also stores extensive metadata such as `source_groups` and per-session stimulus counts that the downstream decoder never uses.

ii. 
```python
session["training_day_index"] = float((session_date - first_date).days)
session["training_day_rank"] = int(index)
```
```python
speed_sessions.append(session_speed)
...
for session_outputs, session_speeds in zip(output_sessions, speed_sessions):
    for output_trial, speed_trial in zip(session_outputs, session_speeds):
        output_trial[3, :] = np.digitize(speed_trial, speed_edges, right=False).astype(np.int16)
```
```python
sample["metadata"]["session_info"] = sample["metadata"]["session_info"][: len(keep_sessions)]
```

iii. The notes justify some of the extra processing as validation or tractability support, but not as requirements of the downstream decoder itself.
