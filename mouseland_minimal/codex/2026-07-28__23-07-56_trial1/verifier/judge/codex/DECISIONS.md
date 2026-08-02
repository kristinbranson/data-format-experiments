# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The converter starts from `data/beh/Imaging_Exp_info.npy`, but it does not load all imaging cohorts. It hard-codes seven rewarded `sup_*` groups, builds a catalog from those rows only, then for each kept session loads one behavior dict entry from `Beh_<group>.npy`, one neural file through `utils.load_spk(...)`, and one retinotopy file from `data/retinotopy`.

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

def load_exp_info():
    return np.load(EXP_INFO_PATH, allow_pickle=True).item()

def load_behavior(group_name):
    path = os.path.join(BEH_DIR, f"Beh_{group_name}.npy")
    return np.load(path, allow_pickle=True).item()

for group_name in SUP_GROUP_PRIORITY:
    for entry in exp_info[group_name]:
        ...

beh = load_behavior(session["behavior_group"])[session["behavior_key"]]
spk = utils.load_spk(..., root=SPK_DIR)
iarea = np.load(
    os.path.join(RETINO_DIR, f"{session['subject']}_{session['date']}_trans.npz"),
    allow_pickle=True,
)["iarea"]
```

iii. `CONVERSION_NOTES.md` says the agent intentionally restricted the export to the rewarded task cohort because it wanted a meaningful `reward_availability` input. Trajectory step 170 repeats that the final dataset contains 28 rewarded task sessions from 5 mice.

## 1-b. How are the data split into subjects?

i. Subjects are identified directly from `entry["mname"]`. The converter collects unique mouse names from the session catalog, sorts them, and stores a per-session integer index into that list.

ii. 
```python
catalog.append(
    {
        ...
        "subject": entry["mname"],
        ...
    }
)

subjects = sorted({session["subject"] for session in catalog})
subject_to_index = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx.append(subject_to_index[session["subject"]])
```

iii. No separate justification is recorded beyond using the mouse identifier already present in the metadata.

## 1-c. How are the data split into sessions?

i. Sessions are defined by `mouse_date_block` (`base_session_id`). If the same base recording appears in more than one rewarded `sup_*` metadata group, the converter deduplicates it and keeps the earliest group according to `SUP_GROUP_PRIORITY`, while preserving the list of source groups in metadata.

ii. 
```python
base_session_id = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
behavior_key = base_session_id
if "stimtype" in entry:
    behavior_key = f"{behavior_key}_{entry['stimtype']}"
session_candidates[base_session_id].append(...)

for base_session_id in sorted(session_candidates):
    candidates = sorted(
        session_candidates[base_session_id],
        key=lambda item: SUP_GROUP_PRIORITY.index(item["group_name"]),
    )
    canonical = candidates[0]
    ...
    catalog.append(
        {
            "base_session_id": base_session_id,
            "behavior_group": canonical["group_name"],
            "behavior_key": canonical["behavior_key"],
            ...
            "source_groups": [item["group_name"] for item in candidates],
        }
    )
```

iii. `CONVERSION_NOTES.md` says the raw rewarded metadata had 33 supervised rows but only 28 unique task recordings, so the agent deduplicated by recording id.

## 1-d. How are the data split into trials?

i. Trials come from `beh["ntrials"]`. The code loops `for trial_idx in range(ntrials)` and builds one neural matrix, one input matrix, and one output matrix per trial by slicing the interpolated session arrays at that trial index.

ii. 
```python
ntrials = int(beh["ntrials"])
...
for trial_idx in range(ntrials):
    ...
    session_neural.append(np.ascontiguousarray(interp_spk[:, trial_idx, :]))
    session_input.append(np.ascontiguousarray(input_trial))
    session_output.append(np.ascontiguousarray(output_trial))
```

iii. No special justification was recorded; this follows the trial structure in each behavior dictionary.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-rejection pass. All trials in `0..ntrials-1` are exported. The only per-trial hard check is that `SoundPos` must be finite; otherwise conversion aborts. Within trials, non-running frames were already excluded upstream during interpolation, and invalid lick positions are ignored by `bin_licks`.

ii. 
```python
for trial_idx in range(ntrials):
    cue_position = float(sound_positions[trial_idx])
    if not np.isfinite(cue_position):
        raise ValueError(f"{session['base_session_id']}: non-finite SoundPos on trial {trial_idx}")

    lick_bins = bin_licks(lick_positions[lick_trial_index == trial_idx], N_POSITION_BINS_CORRIDOR)
```

iii. The notes emphasize running-only analysis and validation checks, but they do not describe any trial-dropping rule beyond format/sanity failures.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The exported neural matrices are ultimately derived from the deconvolved spike-trace arrays loaded by `utils.load_spk(...)` from `*_neural_data.npy`. The converter also uses `beh["ft_move"]`, `beh["ft_CorrSpc"]`, `beh["ft_WallID"]`, `beh["stim_id"]`, `beh["UniqWalls"]`, `beh["ft_PosCum"]`, and retinotopy `iarea` to decide which neurons and frames to keep and how to align them.

ii. 
```python
spk = utils.load_spk(
    {"mname": session["subject"], "datexp": session["date"], "blk": session["block"]},
    root=SPK_DIR,
)
iarea = np.load(... )["iarea"]

vr_move = np.asarray(beh["ft_move"][:nframes] > 0)
corridor_frames = np.asarray(beh["ft_CorrSpc"][:nframes], dtype=bool)
wall_ids = np.asarray(beh["ft_WallID"][:nframes])
stim_names = np.asarray(beh["UniqWalls"]).astype(str)
stim_id = np.asarray(beh["stim_id"], dtype=float)
move_position = np.asarray(beh["ft_PosCum"][:nframes][vr_move], dtype=np.float64)
```

iii. `CONVERSION_NOTES.md` says the converter reuses the behavior metadata, deconvolved Suite2p traces, retinotopy assignments, and running-only frames from the paper code.

## 2-b. How is the `neural` data processed?

i. The code concatenates the raw deconvolved traces via `utils.load_spk(...)`, restricts them to running frames (`ft_move > 0`), interpolates them onto trial-by-position bins using `utils.spk_pos_interp(...)`, keeps only the first 40 bins of the 60-bin trial path, and stores the result as `float16`.

ii. 
```python
move_spk = np.asarray(spk[selected_neurons][:, vr_move], dtype=np.float32)
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR].astype(np.float16, copy=False)
...
session_neural.append(np.ascontiguousarray(interp_spk[:, trial_idx, :]))
```

iii. The notes say the converter keeps the paper's running-only spatial interpolation and then exports only the 4 m corridor window because the decoder task is aligned to corridor entry and predicts 4 corridor-position bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The converter applies a strong extra curation step that is not just basic QC: it excludes all neurons outside grouped visual cortex, computes familiar rewarded-versus-familiar unrewarded `d'` on running corridor frames, prefers neurons with `|d'| >= 0.3`, balances sampling across `V1`, `mHV`, `lHV`, and `aHV`, and caps each session at 512 neurons.

ii. 
```python
candidate_mask = (region_idx != REGION_TO_INDEX["outside_visual_cortex"]) & (np.abs(dp) >= DP_THRESHOLD)
...
for region_name in AREA_SELECTION_ORDER:
    ...
    for neuron_idx in order[:MIN_AREA_NEURONS]:
        selected.append(int(neuron_idx))
...
if len(selected) < max_neurons:
    remaining_candidates = np.flatnonzero(candidate_mask)
    ...
if len(selected) < max_neurons:
    fallback_candidates = np.flatnonzero(region_idx != REGION_TO_INDEX["outside_visual_cortex"])
```

iii. `CONVERSION_NOTES.md` explicitly justifies this as a tractability choice: the raw recordings contain tens of thousands of neurons, so the agent restricted the export to visual-cortex neurons with paper-style `|d'| >= 0.3` selectivity and a 512-neuron cap so the provided decoder could train.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to corridor entry / trial start indirectly through the interpolation grid. The code interpolates running-only activity into `ntrials x 60` position bins over cumulative position, then keeps bins `0:40`, so each trial starts at corridor entry and covers the 4 m corridor.

ii. 
```python
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR].astype(np.float16, copy=False)
...
"temporal_alignment_event": "corridor entry / trial start",
"off_start": 0.0,
"off_end": float(N_POSITION_BINS_CORRIDOR / 6.0),
```

iii. Both the notes and metadata say the alignment event is `corridor entry / trial start`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The exported grid is 10 cm per bin. The code converts this to time using a fixed VR speed of 0.6 m/s, so the implied time bin is 166.6667 ms. It does not do frame-based temporal rebinning; instead it uses position-based interpolation and then interprets those bins as time bins.

ii. 
```python
POSITION_STEP_M = 0.1
VR_SPEED_M_PER_S = 0.6
TIME_BIN_SIZE_MS = POSITION_STEP_M / VR_SPEED_M_PER_S * 1000.0
...
"time_bin_size": TIME_BIN_SIZE_MS,
```

iii. `CONVERSION_NOTES.md` explicitly gives the same calculation: 10 cm / 60 cm s^-1 = 166.6667 ms, and says the representation stays on the paper's spatial interpolation grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from trial-level `beh["SoundPos"]` plus the synthetic trial time grid `time_since_start`, which itself comes from the fixed 10 cm corridor bins.

ii. 
```python
position_units = np.arange(N_POSITION_BINS_CORRIDOR, dtype=np.float32)
time_since_start = position_units / 6.0
...
sound_positions = np.asarray(beh["SoundPos"], dtype=np.float32)
...
cue_position = float(sound_positions[trial_idx])
```

iii. The notes describe this variable as "signed cue time minus current bin time" and justify it by keeping the paper's position-based interpolation while meeting the decoder's need for a time-varying cue signal.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the sound position is converted to seconds by dividing by 6 (because the corridor advances at 6 bins/s), and then the current bin time is subtracted. The result is a signed time-to-cue trace that is positive before the cue and negative after it.

ii. 
```python
input_trial = np.vstack(
    [
        cue_position / 6.0 - time_since_start,
        ...
    ]
).astype(np.float32, copy=False)
```

iii. `CONVERSION_NOTES.md` says `time_to_sound_cue_s` is "signed cue time minus current bin time."

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is sampled on the exact same 40-bin grid as `session_neural`, because the cue-time trace is built from `time_since_start`, which is also the grid used to interpret the interpolated neural bins.

ii. 
```python
time_since_start = position_units / 6.0
...
session_neural.append(np.ascontiguousarray(interp_spk[:, trial_idx, :]))
session_input.append(np.ascontiguousarray(input_trial))
```

iii. The notes say all exported decoder variables use the 40-bin corridor grid aligned to corridor entry.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is not derived from a behavior field. The code derives it from the session metadata date string `entry["datexp"]`, grouped by subject `entry["mname"]`, after building the rewarded-session catalog.

ii. 
```python
def parse_date(date_str):
    return datetime.strptime(date_str, "%Y_%m_%d")

for session in catalog:
    by_subject[session["subject"]].append(session)
...
session_date = parse_date(session["date"])
session["training_day_index"] = float((session_date - first_date).days)
```

iii. `CONVERSION_NOTES.md` says `day_of_training` is "elapsed days since the first rewarded task imaging session for that mouse."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, sessions are sorted by date, the earliest kept rewarded imaging date is treated as day 0, and each later session gets the integer day difference from that first date. That scalar is then repeated across all 40 time bins of every trial in the session.

ii. 
```python
subject_sessions.sort(key=lambda item: parse_date(item["date"]))
first_date = parse_date(subject_sessions[0]["date"])
for index, session in enumerate(subject_sessions):
    session_date = parse_date(session["date"])
    session["training_day_index"] = float((session_date - first_date).days)
...
np.full(N_POSITION_BINS_CORRIDOR, session["training_day_index"], dtype=np.float32)
```

iii. The notes explicitly justify the variable as days since the first rewarded-task imaging session, not true behavioral training day.

## 4-c. What variables in the raw data is `input` *Environment type* derived from?

i. It is not derived at all, because the converter does not create an environment-type input.

ii. 
```python
"input_names": [
    "time_to_sound_cue_s",
    "day_of_training",
    "time_since_trial_start_s",
    "reward_availability",
],
```

iii. No explicit justification is recorded beyond focusing on the four decoder inputs listed in the task and notes.

## 4-d. What processing is involved in computing `input` *Environment type*?

i. None. The converter omits this variable entirely.

ii. 
```python
"input_names": [
    "time_to_sound_cue_s",
    "day_of_training",
    "time_since_trial_start_s",
    "reward_availability",
],
```

iii. No separate justification was recorded.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from the converter's own 40-bin corridor grid, not directly from `Trial_start_time` or `StartFr`. The code uses bin indices `0..39` and converts them to seconds assuming the constant VR speed used in the experiment.

ii. 
```python
position_units = np.arange(N_POSITION_BINS_CORRIDOR, dtype=np.float32)
time_since_start = position_units / 6.0
```

iii. The notes describe `time_since_trial_start_s` as running from 0 to 6.5 s on the 40-bin corridor grid.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The code divides each 10 cm bin index by 6 bins/s, producing a fixed trace `[0, 1/6, 2/6, ..., 39/6]` seconds for every trial.

ii. 
```python
position_units = np.arange(N_POSITION_BINS_CORRIDOR, dtype=np.float32)
time_since_start = position_units / 6.0
...
np.vstack(
    [
        ...,
        time_since_start,
        ...,
    ]
)
```

iii. `CONVERSION_NOTES.md` gives the same interpretation: 0 to 6.5 s over the 40-bin grid.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is exactly co-registered to the neural data because both use the same 40 interpolated corridor bins starting at trial onset.

ii. 
```python
session_neural.append(np.ascontiguousarray(interp_spk[:, trial_idx, :]))
session_input.append(np.ascontiguousarray(input_trial))
```

iii. The notes and metadata both state that all exported variables are aligned to corridor entry / trial start.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes directly from the per-trial boolean `beh["isRew"]`.

ii. 
```python
trial_is_rewarded = np.asarray(beh["isRew"], dtype=bool)
...
float(trial_is_rewarded[trial_idx])
```

iii. `CONVERSION_NOTES.md` says `reward_availability` is 1 for rewarded familiar corridors and 0 otherwise.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. For each trial, the boolean `isRew` value is cast to float and repeated across all 40 bins, making it a time-varying constant within that trial.

ii. 
```python
np.full(
    N_POSITION_BINS_CORRIDOR,
    float(trial_is_rewarded[trial_idx]),
    dtype=np.float32,
)
```

iii. The notes justify the rewarded-task-only export by saying this input should remain meaningful and variable across trials.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from the per-trial stimulus name `beh["WallName"]`, with the session-global vocabulary built from `beh["UniqWalls"]` across all kept sessions.

ii. 
```python
stimulus_names = set()
for session in catalog:
    beh = load_behavior(session["behavior_group"])[session["behavior_key"]]
    stimulus_names.update(map(str, beh["UniqWalls"]))
...
trial_wall_names = np.asarray(beh["WallName"]).astype(str)
...
stimulus_to_index[trial_stimulus]
```

iii. `CONVERSION_NOTES.md` lists the exported stimulus categories and says `visual_stimulus_category` is constant within each trial.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The code natural-sorts all stimulus names, maps each name to an integer category id, and then fills every time bin of a trial with that same category index.

ii. 
```python
stimulus_names = collect_stimulus_names(catalog)
stimulus_to_index = {name: idx for idx, name in enumerate(stimulus_names)}
...
np.full(
    N_POSITION_BINS_CORRIDOR,
    stimulus_to_index[trial_stimulus],
    dtype=np.int16,
)
```

iii. The notes say this output is constant across time within each trial.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `beh["LickPos"]` and `beh["LickTrind"]`.

ii. 
```python
lick_positions = np.asarray(beh["LickPos"], dtype=np.float32)
lick_trial_index = np.asarray(beh["LickTrind"], dtype=np.int64)
...
lick_bins = bin_licks(lick_positions[lick_trial_index == trial_idx], N_POSITION_BINS_CORRIDOR)
```

iii. The notes say licking is exported as a binary per-10-cm-bin signal derived from `LickPos` within the corridor.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The helper `bin_licks(...)` keeps lick positions in `[0, 40)`, floors each position to an integer decimeter bin, clips the index range, and writes `1` if at least one lick lands in that bin; otherwise the bin stays `0`.

ii. 
```python
def bin_licks(lick_positions, nbins):
    lick_bins = np.zeros(nbins, dtype=np.int16)
    ...
    valid_positions = lick_positions[(lick_positions >= 0) & (lick_positions < nbins)]
    ...
    indices = np.floor(valid_positions).astype(int)
    lick_bins[np.clip(indices, 0, nbins - 1)] = 1
    return lick_bins
```

iii. `CONVERSION_NOTES.md` says licking is binary per 10 cm bin.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The binary lick vector is built on the same 40 corridor bins as the neural trial matrix, so alignment is by trial-start-referenced corridor position/time rather than by raw lick timestamps.

ii. 
```python
output_trial = np.vstack(
    [
        ...,
        lick_bins,
        ...,
    ]
)
...
session_neural.append(np.ascontiguousarray(interp_spk[:, trial_idx, :]))
session_output.append(np.ascontiguousarray(output_trial))
```

iii. The notes justify all output variables as living on the same corridor-entry-aligned 40-bin grid.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from the interpolated trial grid itself. The raw position signal used to define that grid is `beh["ft_PosCum"]` on running frames, but the exported category comes from the bin index on the truncated 40-bin corridor window.

ii. 
```python
move_position = np.asarray(beh["ft_PosCum"][:nframes][vr_move], dtype=np.float64)
...
position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
```

iii. The notes say the export uses the paper's 10 cm interpolation grid and then groups the 4 m corridor into four 1 m output bins.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The code precomputes a 40-element vector containing ten `0`s, ten `1`s, ten `2`s, and ten `3`s, then copies that same vector into every trial.

ii. 
```python
position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
...
output_trial = np.vstack(
    [
        ...,
        position_indices,
        ...,
    ]
)
```

iii. `CONVERSION_NOTES.md` says `corridor_position_bin` is "four 1 m bins across the 4 m corridor."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholding is fixed and deterministic: the first 10 decimeter bins map to category 0, the next 10 to category 1, then 2, then 3.

ii. 
```python
position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
position_values = [f"{start}-{start + 1}m" for start in range(4)]
```

iii. The notes explicitly justify this as the requested 4 x 1 m discretization.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. It is perfectly aligned because it is just the categorical version of the same 40-bin interpolated corridor axis used for neural activity.

ii. 
```python
interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR]
...
output_trial = np.vstack(
    [
        ...,
        position_indices,
        ...,
    ]
)
```

iii. The notes say the representation stays on the corridor-entry-aligned interpolation grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from frame-level `beh["ft_RunSpeed"]`, restricted to running frames.

ii. 
```python
vr_move = np.asarray(beh["ft_move"][:nframes] > 0)
move_speed = np.asarray(beh["ft_RunSpeed"][:nframes][vr_move], dtype=np.float32)[None, :]
```

iii. The notes say running-speed bins are computed from the interpolated corridor time bins in the full export.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speed is interpolated onto the same trial-by-position grid as neural activity using `utils.spk_pos_interp(...)`, truncated to the first 40 bins, stored temporarily as continuous values, and pooled across all sessions/trials to estimate global quartile edges.

ii. 
```python
interp_speed = interpolate_running_signal(move_speed, move_position, corridor_length, ntrials)[0]
interp_speed = interp_speed[:, :N_POSITION_BINS_CORRIDOR].astype(np.float32, copy=False)
all_speed_values.append(interp_speed.reshape(-1))
...
speed_values = np.concatenate(all_speed_values)
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` says speed quartiles are computed over all interpolated corridor bins in the full export.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. After collecting all continuous speed values, the converter computes the 25th, 50th, and 75th percentiles and uses `np.digitize(...)` to assign each time bin to categories 0-3.

ii. 
```python
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
...
output_trial[3, :] = np.digitize(speed_trial, speed_edges, right=False).astype(np.int16)
```

iii. The notes justify this with the decoder specification that running speed should be discretized into four bins containing 25% of the data each.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. It is aligned by using the same running-only interpolation over `ft_PosCum` and the same 40-bin truncation as neural activity.

ii. 
```python
interp_speed = interpolate_running_signal(move_speed, move_position, corridor_length, ntrials)[0]
interp_speed = interp_speed[:, :N_POSITION_BINS_CORRIDOR].astype(np.float32, copy=False)
...
session_speed.append(np.ascontiguousarray(interp_speed[trial_idx]))
```

iii. The notes say all exported variables share the corridor-entry-aligned 40-bin grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code uses a mixture of tolerance, fallback, and hard failure. `np.nanmean`/`np.nanstd` tolerate NaNs in d-prime computation; `bin_licks` silently returns all-zero bins for empty or out-of-range lick sets; neuron selection falls back to lower-priority candidates if too few neurons exceed the d-prime threshold; but a session aborts if corridor length is not 60, if familiar rewarded/unrewarded frames are missing, or if any trial has non-finite `SoundPos`.

ii. 
```python
if corridor_length != N_POSITION_BINS_TOTAL:
    raise ValueError(...)
...
if stim1_frames.sum() == 0 or stim2_frames.sum() == 0:
    raise ValueError(...)
...
stim1_mean = np.nanmean(spk[:, stim1_frames], axis=1)
stim1_std = np.nanstd(spk[:, stim1_frames], axis=1)
...
if not np.isfinite(cue_position):
    raise ValueError(...)
```

iii. The notes emphasize sanity checks and validation, but they do not document a broader missing-data policy beyond these guardrails.

## 12-a. What are the most time-consuming steps of the code?

i. The expensive steps are loading the very large `spk` files, computing d-prime over all neurons and many frames, interpolating neural activity and speed with `utils.spk_pos_interp(...)`, and then materializing per-trial arrays in Python loops.

ii. 
```python
spk = utils.load_spk(..., root=SPK_DIR)
...
stim1_mean = np.nanmean(spk[:, stim1_frames], axis=1)
stim2_mean = np.nanmean(spk[:, stim2_frames], axis=1)
...
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
interp_speed = interpolate_running_signal(move_speed, move_position, corridor_length, ntrials)[0]
...
for trial_idx in range(ntrials):
    ...
```

iii. The notes only justify this indirectly by saying the raw sessions contain 47k-89k neurons and that tractability was a major concern.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop that builds `input_trial`, `output_trial`, `lick_bins`, and appends arrays could be vectorized. The region-balancing loop in `select_neurons(...)` is also inherently Python-heavy. Most importantly, the reference `utils.spk_pos_interp(...)` interpolates one neuron at a time in Python.

ii. 
```python
for region_name in AREA_SELECTION_ORDER:
    ...
    for neuron_idx in order[:MIN_AREA_NEURONS]:
        selected.append(int(neuron_idx))
...
for trial_idx in range(ntrials):
    ...
    lick_bins = bin_licks(...)
    input_trial = np.vstack(...)
    output_trial = np.vstack(...)
```

iii. No explicit justification for these loops was recorded.

## 12-c. What processing does the code repeat multiple times?

i. It loads behavior once in `collect_stimulus_names(...)` and again during conversion, computes interpolation twice per session (once for neural activity and once for speed), and stores continuous speed arrays only to revisit them later for quartile binning.

ii. 
```python
for session in catalog:
    beh = load_behavior(session["behavior_group"])[session["behavior_key"]]
    stimulus_names.update(map(str, beh["UniqWalls"]))
...
beh = load_behavior(session["behavior_group"])[session["behavior_key"]]
...
interp_spk = interpolate_running_signal(...)
...
interp_speed = interpolate_running_signal(...)[0]
...
all_speed_values.append(interp_speed.reshape(-1))
```

iii. No explicit justification was recorded beyond simplicity.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest discarded work is the temporary storage of continuous `session_speed` and `all_speed_values`: they are only used to compute speed quartiles and then the continuous traces are thrown away from the final saved dataset. The code also retains extra session metadata such as `source_groups`, full stimulus counters, and original neuron-count ranges that are not used by downstream decoder training.

ii. 
```python
speed_sessions = []
all_speed_values = []
...
session_speed.append(np.ascontiguousarray(interp_speed[trial_idx]))
...
speed_values = np.concatenate(all_speed_values)
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
...
for session_outputs, session_speeds in zip(output_sessions, speed_sessions):
    for output_trial, speed_trial in zip(session_outputs, session_speeds):
        output_trial[3, :] = np.digitize(speed_trial, speed_edges, right=False).astype(np.int16)
```

iii. No explicit justification was recorded; this appears to be an implementation convenience for computing global quartiles.
