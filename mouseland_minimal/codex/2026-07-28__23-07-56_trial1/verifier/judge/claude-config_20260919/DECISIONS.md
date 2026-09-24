# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the same three data folders as the reference (`data/beh`, `data/spk`, `data/retinotopy`) and uses `beh/Imaging_Exp_info.npy` as the master index. It re-uses the paper's own loader `code/utils.load_spk` for the deconvolved traces and loads the retinotopy `*_trans.npz` per session for `iarea`. Crucially, it **does not load all the data**: it iterates only over the seven supervised (task-mouse) experiment groups (`SUP_GROUP_PRIORITY`), deliberately excluding the unsupervised, naive and grating cohorts. This yields 28 sessions / 5 mice / 11,528 trials, versus the 89 recordings in 19 mice described in the Methods (the reference solution converts all 89 / 19 / 37,728 trials). Behavior is loaded lazily by group name inside the session loop (`load_behavior(...)` is called once per session, plus once more per session in `collect_stimulus_names`), so each `Beh_*.npy` file is read many times.

ii.
```python
SUP_GROUP_PRIORITY = [
    "sup_train1_before_learning", "sup_train1_after_learning", "sup_test1",
    "sup_train2_before_learning", "sup_train2_after_learning", "sup_test2", "sup_test3",
]

def load_exp_info():
    return np.load(EXP_INFO_PATH, allow_pickle=True).item()

def load_behavior(group_name):
    path = os.path.join(BEH_DIR, f"Beh_{group_name}.npy")
    return np.load(path, allow_pickle=True).item()

def build_session_catalog(exp_info):
    session_candidates = defaultdict(list)
    for group_name in SUP_GROUP_PRIORITY:
        for entry in exp_info[group_name]:
            base_session_id = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
            ...
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

iii. From `CONVERSION_NOTES.md` and step 123/133 of the trajectory: *"I restricted the export to the rewarded task cohort because the decoder input specification requires a meaningful per-trial `reward_availability` variable. The unsupervised and naive cohorts do not have rewarded corridors, so including them would collapse that input and make the converted dataset inconsistent with the requested decoder task."* It also cites tractability ("the memory profile rules out exporting every raw neuron across every trial").

## 1-b. How are the data split into subjects (mice)?

i. The subject is taken directly from `entry['mname']` in the master index and carried on every catalog record. `subjects` is the sorted set of unique mouse names and `subject_idx` is each session's index into that list. Sessions are sorted by `(subject, date, block)`, so all sessions of a mouse are contiguous. Result: 5 subjects (`TX108, TX109, TX60, TX61, VR2`) — a consequence of the cohort restriction in 1-a, not of the splitting mechanism.

ii.
```python
catalog.append({..., "subject": entry["mname"], "date": entry["datexp"], "block": entry["blk"], ...})
...
subjects = sorted({session["subject"] for session in catalog})
subject_to_index = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx.append(subject_to_index[session["subject"]])
...
"subject_idx": np.asarray(subject_idx, dtype=np.int16),
```

iii. Not explicitly argued; the index already names the mouse, so no derivation is needed. The AI verified against the paper ("89 unique base recordings ... in 19 mice") in step 35 before restricting scope.

## 1-c. How are the data split into sessions?

i. A session is one recording = `mname_datexp_blk`. Because the same recording appears under several experiment types (33 supervised rows → 28 unique recordings), candidates are grouped by `base_session_id` and only the first group in a fixed priority order is kept; the `stimtype` suffix (e.g. `_swap1`) is used only to build the behavior dictionary key, not to create extra sessions. The kept group names are recorded in `session_info['source_groups']`.

ii.
```python
for group_name in SUP_GROUP_PRIORITY:
    for entry in exp_info[group_name]:
        base_session_id = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
        behavior_key = base_session_id
        if "stimtype" in entry:
            behavior_key = f"{behavior_key}_{entry['stimtype']}"
        session_candidates[base_session_id].append({...})

for base_session_id in sorted(session_candidates):
    candidates = sorted(session_candidates[base_session_id],
                        key=lambda item: SUP_GROUP_PRIORITY.index(item["group_name"]))
    canonical = candidates[0]
```

iii. Step 67 and 83: *"there are 89 unique base recordings, but 142 analysis rows because some sessions are reused in multiple figure-specific subsets and some `test3` sessions are split into `swap1` and `swap2` views. The converter should target unique recordings, not duplicate analysis rows."* It directly compared the `swap1`/`swap2` dictionaries (step 44) before deciding they are views of one recording.

## 1-d. How are the data split into trials?

i. Trials are not derived from the frame-level trial label `ft_trInd`. Instead the AI uses the published notebook's approach: `utils.spk_pos_interp` resamples every stream against cumulative VR position `ft_PosCum / Corridor_Length`, on a grid `np.arange(0, ntrials, 1/60)`. Because each corridor+grey cycle is exactly 60 dm, bin `t*60 + k` is trial `t`, position `k` dm. The trial count is `beh['ntrials']` and every trial is the **first 40 of the 60 bins**, i.e. the 4 m textured corridor, discarding the 2 m grey space. Only frames where the VR was moving (`ft_move > 0`) enter the interpolation, so stationary periods are removed. Every trial therefore has exactly 40 bins.

ii.
```python
nframes = spk.shape[1]
vr_move = np.asarray(beh["ft_move"][:nframes] > 0)
move_position = np.asarray(beh["ft_PosCum"][:nframes][vr_move], dtype=np.float64)
move_spk = np.asarray(spk[selected_neurons][:, vr_move], dtype=np.float32)
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR].astype(np.float16, copy=False)
...
for trial_idx in range(ntrials):
    ...
    session_neural.append(np.ascontiguousarray(interp_spk[:, trial_idx, :]))
```
```python
def interpolate_running_signal(signal_2d, accumulated_position, corridor_length, ntrials):
    return utils.spk_pos_interp(raw_spk=signal_2d, accum_pos=accumulated_position,
                                corridorLen=corridor_length, new_shape=[ntrials, 0])
```

iii. Step 55/133: the notebook cell it found states *"get interpolational neural activity (neurons * trials * positions) — bins of position: 60, each bin is 1 decimeter, total length of corridor is 6 meters"*, and the Methods say *"We only considered timepoints during running for analysis"*. The AI concluded it should *"keep the paper's running-only position interpolation, align trials to corridor entry, and keep the 4 m corridor window as 40 bins"*.

## 1-e. How are trials filtered based on quality controls?

i. **No trial-level quality control at all.** Every one of `beh['ntrials']` trials in every retained session is exported. There is no check that a trial was actually imaged, no trial-length outlier rule, and no minimum-trials-per-session rule. Session-level guards exist but abort the run rather than skip a session (`raise ValueError` on unexpected corridor length, on non-finite `SoundPos`, or on a missing familiar-corridor contrast).

ii.
```python
for trial_idx in range(ntrials):
    trial_stimulus = str(trial_wall_names[trial_idx])
    ...
    session_neural.append(np.ascontiguousarray(interp_spk[:, trial_idx, :]))
```
```python
if corridor_length != N_POSITION_BINS_TOTAL:
    raise ValueError(...)
if not np.isfinite(cue_position):
    raise ValueError(f"{session['base_session_id']}: non-finite SoundPos on trial {trial_idx}")
if stim1_frames.sum() == 0 or stim2_frames.sum() == 0:
    raise ValueError(f"{session['base_session_id']}: missing familiar corridor frames for d' selection")
```

iii. Never argued. Implicitly, the position-interpolated representation removes the motivation for the reference's length filter (a mouse that stops contributes no extra bins), and the AI's sanity checks (`CONVERSION_NOTES.md`) only verify shapes: *"Every exported trial has: 512 neurons, 40 time bins, 4 inputs, 4 outputs."*

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spks` from `spk/<session>_neural_data.npy`, concatenated across imaging planes by the paper's own `utils.load_spk`, plus `iarea` from `retinotopy/<mouse>_<date>_trans.npz` for the region label. Frame masking uses `ft_move` and `ft_CorrSpc`; the resampling axis is `ft_PosCum`.

ii.
```python
spk = utils.load_spk({"mname": ..., "datexp": ..., "blk": ...}, root=SPK_DIR)
iarea = np.load(os.path.join(RETINO_DIR, f"{session['subject']}_{session['date']}_trans.npz"),
                allow_pickle=True)["iarea"]
region_idx_all = grouped_region_indices(iarea)
```
```python
def grouped_region_indices(iarea):
    region_idx = np.full(len(iarea), REGION_TO_INDEX["outside_visual_cortex"], dtype=np.int16)
    region_idx[np.asarray(iarea == 8)] = REGION_TO_INDEX["V1"]
    region_idx[np.asarray(np.isin(iarea, [0, 1, 2, 9]))] = REGION_TO_INDEX["mHV"]
    region_idx[np.asarray(np.isin(iarea, [5, 6]))] = REGION_TO_INDEX["lHV"]
    region_idx[np.asarray(np.isin(iarea, [3, 4]))] = REGION_TO_INDEX["aHV"]
    return region_idx
```

iii. `CONVERSION_NOTES.md`: *"Uses the deconvolved Suite2p traces in `data/spk/*_neural_data.npy` ... Uses the same grouped visual areas as the paper code: V1, mHV, lHV, aHV"* — the area codes were copied from `utils.neu_area_ID`.

## 2-b. How is the `neural` data processed?

i. Three operations: (1) restrict to frames where the VR was moving; (2) linearly interpolate each selected neuron's deconvolved trace onto the cumulative-position grid (60 bins/trial) with `utils.spk_pos_interp`, then keep the first 40 bins; (3) cast to `float16`. No normalisation, z-scoring or smoothing beyond the interpolation itself. The traces are **not** kept at native frame resolution: a 4 m corridor traversal is ~21 imaging frames at 3.17 Hz but is written out as 40 bins, so the data are up-sampled ≈2× by linear interpolation.

ii.
```python
move_spk = np.asarray(spk[selected_neurons][:, vr_move], dtype=np.float32)
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR].astype(np.float16, copy=False)
```

iii. `CONVERSION_NOTES.md`: *"Uses running-only frames: `beh['ft_move'][:nframes] > 0`. Uses accumulated-position interpolation exactly in the style of `utils.spk_pos_interp(...)`"*; *"Reference interpolation grid: 60 bins across the 6 m trial path, as in the notebook and `utils.get_interpPos_spk(...)`. Exported analysis window: first 40 bins only, corresponding to the 4 m corridor."*

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. (1) Retinotopic area: only neurons in V1/mHV/lHV/aHV are eligible (matching the reference). (2) A **selectivity filter and a hard cap**: for each session the AI computes d′ between the familiar rewarded and familiar unrewarded corridor (running, in-corridor frames), keeps only neurons with `|d'| >= 0.3`, and selects at most 512 per session — 128 top-|d′| neurons per area, then the next highest-|d′| neurons, with a fallback that ignores the d′ threshold if fewer than 512 qualify. Every exported session has exactly 512 neurons, out of 47,785–89,577 recorded (≈1% kept; the reference keeps all 4.1 M visual-cortex neurons, 46,128 per session on average).

ii.
```python
DP_THRESHOLD = 0.3
MAX_NEURONS = 512
MIN_AREA_NEURONS = MAX_NEURONS // len(AREA_SELECTION_ORDER)
...
stim1_frames = (wall_ids == familiar_rewarded) & valid_corridor_frames
stim2_frames = (wall_ids == familiar_unrewarded) & valid_corridor_frames
stim1_mean = np.nanmean(spk[:, stim1_frames], axis=1)
stim2_mean = np.nanmean(spk[:, stim2_frames], axis=1)
stim1_std = np.nanstd(spk[:, stim1_frames], axis=1)
stim2_std = np.nanstd(spk[:, stim2_frames], axis=1)
dp = 2.0 * (stim1_mean - stim2_mean) / (stim1_std + stim2_std + 1e-12)
selected_neurons = select_neurons(dp, region_idx_all, max_neurons=max_neurons)
```
```python
def select_neurons(dp, region_idx, max_neurons):
    candidate_mask = (region_idx != REGION_TO_INDEX["outside_visual_cortex"]) & (np.abs(dp) >= DP_THRESHOLD)
    for region_name in AREA_SELECTION_ORDER:
        region_candidates = np.flatnonzero(candidate_mask & (region_idx == region_code))
        order = region_candidates[np.argsort(abs_dp[region_candidates])[::-1]]
        for neuron_idx in order[:MIN_AREA_NEURONS]:
            selected.append(int(neuron_idx))
    ...
```

iii. `CONVERSION_NOTES.md`: *"The raw task recordings contain 47,785 to 89,577 neurons per session. Exporting every ROI trial-by-trial would not be trainable by the provided decoder. To keep the export aligned with the paper's analysis logic while making the benchmark tractable ... Prefer neurons with `|d'| >= 0.3`, matching the paper's selective-neuron threshold used throughout the figure code; cap each session at 512 neurons; fill the 512 slots with a balanced top-|d'| sample across V1, mHV, lHV and aHV."* Trajectory step 54: *"The memory profile rules out exporting every raw neuron across every trial into this decoder format."*

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to corridor entry. Because the interpolation grid is `np.arange(0, ntrials, 1/60)` in units of cumulative position / 60 dm, bin 0 of trial *t* is exactly the frame at which cumulative position crosses `t * 60` dm, i.e. corridor entry (I verified `PosCum` at `StartFr` ≈ 60·trial to within ~0.3 dm on `TX109_2023_03_27_1`). All trials have the same fixed window: 40 bins from entry to the 4 m mark; `off_start = 0.0`, `off_end = 6.667` s. No padding is needed because the window is defined in position, not time.

ii.
```python
linPos = np.arange(0, new_shape[0], 1/new_shape[1])      # utils.spk_pos_interp
...
interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR]
...
"temporal_alignment_event": "corridor entry / trial start",
"off_start": 0.0,
"off_end": float(N_POSITION_BINS_CORRIDOR / 6.0),
```

iii. `CONVERSION_NOTES.md`: *"Alignment event: corridor entry / trial start ... Exported analysis window: first 40 bins only, corresponding to the 4 m corridor. This keeps the paper's running-only spatial interpolation while matching the requested decoder alignment and the requested 4 × 1 m corridor-position output bins."*

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Declared `time_bin_size = 166.667 ms`, obtained by dividing the 10 cm spatial bin by the fixed VR speed of 60 cm/s. Rebinning is applied: the data are resampled from native imaging frames (3.17 Hz → 315 ms) onto the 10 cm position grid. Since a 4 m traversal takes 6.67 s of VR time ≈ 21 frames but is written as 40 bins, this is an ≈2× **up-sampling**: adjacent bins are linear interpolations between the same pair of imaging frames. It is also "VR time", not wall-clock time — every stationary period is excised, so the real elapsed time spanned by a bin varies.

ii.
```python
POSITION_STEP_M = 0.1
VR_SPEED_M_PER_S = 0.6
TIME_BIN_SIZE_MS = POSITION_STEP_M / VR_SPEED_M_PER_S * 1000.0   # 166.667 ms
...
"time_bin_size": TIME_BIN_SIZE_MS,
"corridor_length_m": 4.0, "bin_size_m": POSITION_STEP_M, "vr_speed_m_per_s": VR_SPEED_M_PER_S,
```

iii. `CONVERSION_NOTES.md`: *"Bin width: 10 cm. Implied time bin: 10 cm / 60 cm s⁻¹ = 0.1666667 s = 166.6667 ms."* The justification is the Methods statement that *"the virtual corridors always moved at a constant speed (60 cm s⁻¹) as long as mice kept running faster than the threshold"*, so a fixed distance corresponds to a fixed VR time. The AI did not comment on the resulting up-sampling relative to the 3.17 Hz imaging rate.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. `beh['SoundPos']` — the corridor position (in decimetres) at which the sound cue was played on each trial — together with the bin's own position index. `SoundFr` / `ft` (the frame-based route used by the reference) are not used.

ii.
```python
sound_positions = np.asarray(beh["SoundPos"], dtype=np.float32)
...
cue_position = float(sound_positions[trial_idx])
if not np.isfinite(cue_position):
    raise ValueError(...)
```

iii. Implicit in the design: since the whole dataset lives on a position axis, the cue is naturally located by its position. The AI checked cue presence per corridor type in step 124 (*"I'm checking the early VR2 sessions that the notes flag as 'no cue in non-reward corridor', because that directly affects the time to sound cue input"*).

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The cue position is converted to VR seconds by dividing by 6 (1 dm / 0.6 m s⁻¹ = 1/6 s), and the bin's own VR time is subtracted, giving a signed value that is positive before the cue and negative after — the same sign convention as the reference. Range in the exported data: −6.31 to +6.51 s.

ii.
```python
position_units = np.arange(N_POSITION_BINS_CORRIDOR, dtype=np.float32)
time_since_start = position_units / 6.0
...
input_trial = np.vstack([
    cue_position / 6.0 - time_since_start,
    ...
])
```

iii. `CONVERSION_NOTES.md`: *"`time_to_sound_cue_s`: signed cue time minus current bin time"*, and a sanity check: *"The cue-time input spans both positive and negative values, confirming that bins occur before and after the cue."*

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated on exactly the same 40-bin position grid as the neural matrix for the same trial, so bin *k* of the input corresponds to bin *k* of `neural`. Shapes are `(4, 40)` vs `(512, 40)`.

ii.
```python
input_trial = np.vstack([cue_position / 6.0 - time_since_start, ...]).astype(np.float32, copy=False)
session_input.append(np.ascontiguousarray(input_trial))
session_neural.append(np.ascontiguousarray(interp_spk[:, trial_idx, :]))
```

iii. Not separately argued; all streams share the single position grid by construction.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. `entry['datexp']` from the master index, parsed as a calendar date.

ii.
```python
def parse_date(date_str):
    return datetime.strptime(date_str, "%Y_%m_%d")
```

iii. Not separately argued; the date string is the only training-time information in the index.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse the sessions are sorted by date and the value is the **number of calendar days elapsed since that mouse's first exported session** (`training_day_index`, 0–73 in the export). An ordinal rank (`training_day_rank`) is also computed but is only stored in `session_info`, not used as the input. The scalar is broadcast across the trial's 40 bins.

ii.
```python
def attach_session_days(catalog):
    by_subject = defaultdict(list)
    for session in catalog:
        by_subject[session["subject"]].append(session)
    for subject_sessions in by_subject.values():
        subject_sessions.sort(key=lambda item: parse_date(item["date"]))
        first_date = parse_date(subject_sessions[0]["date"])
        for index, session in enumerate(subject_sessions):
            session_date = parse_date(session["date"])
            session["training_day_index"] = float((session_date - first_date).days)
            session["training_day_rank"] = int(index)
```
```python
np.full(N_POSITION_BINS_CORRIDOR, session["training_day_index"], dtype=np.float32),
```

iii. `CONVERSION_NOTES.md`: *"`day_of_training`: elapsed days since the first rewarded task imaging session for that mouse."*

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Nothing from the behaviour file: it is the bin index of the position grid divided by 6, i.e. a fixed ramp implied by `ft_PosCum` and the constant VR speed. `StartFr` and `ft` (the reference's source) are not read.

ii.
```python
position_units = np.arange(N_POSITION_BINS_CORRIDOR, dtype=np.float32)
time_since_start = position_units / 6.0
```

iii. `CONVERSION_NOTES.md`: *"`time_since_trial_start_s`: 0 to 6.5 s on the 40-bin grid"* — a direct consequence of the choice to express everything in VR time on a position axis.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. `k/6` seconds for bin `k`, i.e. 0 → 6.5 s in 0.1667 s steps. The vector is computed once and reused for every trial of every session, so it is identical across all 11,528 trials and carries no across-trial information; it is also an exact affine function of the *Position in corridor* output.

ii.
```python
time_since_start = position_units / 6.0
...
input_trial = np.vstack([
    cue_position / 6.0 - time_since_start,
    np.full(N_POSITION_BINS_CORRIDOR, session["training_day_index"], dtype=np.float32),
    time_since_start,
    np.full(N_POSITION_BINS_CORRIDOR, float(trial_is_rewarded[trial_idx]), dtype=np.float32),
]).astype(np.float32, copy=False)
```

iii. As above: the AI treats VR time and position as interchangeable because the VR advances at a fixed 60 cm/s whenever the mouse runs, and non-running frames are excluded from the whole dataset.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same 40-bin grid as the neural data; bin 0 is corridor entry so the value starts at 0.

ii.
```python
session_input.append(np.ascontiguousarray(input_trial))     # (4, 40)
session_neural.append(np.ascontiguousarray(interp_spk[:, trial_idx, :]))   # (512, 40)
```

iii. Not separately argued.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. `beh['isRew']`, the per-trial flag marking trials run in the rewarded corridor — the same variable the reference uses.

ii.
```python
trial_is_rewarded = np.asarray(beh["isRew"], dtype=bool)
```

iii. `CONVERSION_NOTES.md`: *"`reward_availability`: 1 for rewarded familiar corridors, 0 otherwise."*

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast to float and broadcast across the trial's 40 bins. No other processing. Because only task mice are exported, the variable actually varies (0 and 1 both present).

ii.
```python
np.full(N_POSITION_BINS_CORRIDOR, float(trial_is_rewarded[trial_idx]), dtype=np.float32),
```

iii. This variable is the AI's stated reason for restricting the export to the task cohort (see 1-a): *"The decoder input requires reward availability per trial. Unsupervised and naive cohorts do not have rewarded corridors."* It also verified *"Reward availability varies between 0 and 1 across the dataset."*

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. `beh['WallName']` for the per-trial label, and `beh['UniqWalls']` (pooled over all exported sessions) to build the global category list.

ii.
```python
def collect_stimulus_names(catalog):
    stimulus_names = set()
    for session in catalog:
        beh = load_behavior(session["behavior_group"])[session["behavior_key"]]
        stimulus_names.update(map(str, beh["UniqWalls"]))
    return sorted(stimulus_names, key=natural_sort_key)
...
trial_wall_names = np.asarray(beh["WallName"]).astype(str)
trial_stimulus = str(trial_wall_names[trial_idx])
```

iii. Not separately argued; the AI's instructions gave the example categories as *"circle1, leaf2, etc."*, i.e. the raw wall names.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. No pooling: the 14 distinct wall names that occur across the 28 exported sessions become 14 categories (`circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap2, wood2, wood5`), naturally sorted, and the trial's index into that list is broadcast across its 40 bins. The reference instead maps the 15 names onto 4 base textures (circle/leaf/rock/wood). Individual sessions contain 2–5 of the categories, and several categories are rare globally (`circle3` 0.7%, `rock2` 0.5%).

ii.
```python
stimulus_to_index = {name: idx for idx, name in enumerate(stimulus_names)}
...
output_trial = np.vstack([
    np.full(N_POSITION_BINS_CORRIDOR, stimulus_to_index[trial_stimulus], dtype=np.int16),
    lick_bins, position_indices, np.zeros(N_POSITION_BINS_CORRIDOR, dtype=np.int16),
])
```

iii. `CONVERSION_NOTES.md`: *"`visual_stimulus_category`: constant across time within each trial"*, with the full 14-name list given under "Stimulus categories present". The AI's task instruction literally requested per-trial categories *"e.g. circle1, leaf2"*.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. `beh['LickPos']` (position of each lick, in decimetres) together with `beh['LickTrind']` (the trial each lick belongs to). The reference instead uses `LickFr`, the imaging frame of each lick.

ii.
```python
lick_positions = np.asarray(beh["LickPos"], dtype=np.float32)
lick_trial_index = np.asarray(beh["LickTrind"], dtype=np.int64)
...
lick_bins = bin_licks(lick_positions[lick_trial_index == trial_idx], N_POSITION_BINS_CORRIDOR)
```

iii. Consistent with the position-indexed representation; `utils.get_lick_raster` in the paper code likewise works from `LickPos`/`LickTrind`.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A bin is 1 if at least one lick of that trial falls in it. Lick positions outside `[0, 40)` (i.e. licks in the grey space) are discarded, and the remaining positions are floored to a 10 cm bin. Result: 6.4% of bins are licks (reference: 4.1% of frames).

ii.
```python
def bin_licks(lick_positions, nbins):
    lick_bins = np.zeros(nbins, dtype=np.int16)
    if lick_positions.size == 0:
        return lick_bins
    valid_positions = lick_positions[(lick_positions >= 0) & (lick_positions < nbins)]
    if valid_positions.size == 0:
        return lick_bins
    indices = np.floor(valid_positions).astype(int)
    lick_bins[np.clip(indices, 0, nbins - 1)] = 1
    return lick_bins
```

iii. `CONVERSION_NOTES.md`: *"`licking`: binary per 10 cm bin, derived from `LickPos` within the corridor."*

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Via position: a lick is placed in the same 10 cm bin whose neural value was interpolated at that position, so the streams share one index. Note that licks emitted while the mouse was stationary (e.g. collecting reward) are still assigned to a bin, whereas the neural samples for those stationary frames were dropped before interpolation — the label comes from the interval between the two imaging frames that bracket that position.

ii.
```python
lick_bins = bin_licks(lick_positions[lick_trial_index == trial_idx], N_POSITION_BINS_CORRIDOR)
output_trial = np.vstack([..., lick_bins, ...])
session_output.append(np.ascontiguousarray(output_trial))
```

iii. Not separately argued beyond the shared position grid.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Indirectly from `beh['ft_PosCum']`: position is the interpolation grid itself, so the output is the bin index. `beh['ft_Pos']` (used by the reference) is never read.

ii.
```python
position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
```

iii. Implicit: after position interpolation, each bin's corridor position is known exactly by construction.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. None beyond construction: the identical constant vector `[0]*10 + [1]*10 + [2]*10 + [3]*10` is emitted for every trial in every session. It is therefore exactly determined by the bin index and, consequently, is an exact function of the *Time since trial start* decoder **input** — the decoder can recover it perfectly without using any neural activity (validation balanced accuracy 0.923, vs 0.310 for the reference).

ii.
```python
position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
...
output_trial = np.vstack([
    np.full(N_POSITION_BINS_CORRIDOR, stimulus_to_index[trial_stimulus], dtype=np.int16),
    lick_bins,
    position_indices,
    np.zeros(N_POSITION_BINS_CORRIDOR, dtype=np.int16),
])
```

iii. `CONVERSION_NOTES.md`: *"`corridor_position_bin`: four 1 m bins across the 4 m corridor"* and, as a sanity check, *"Position bins are exactly balanced at 25% each by construction."* The degeneracy relative to the time input is not discussed.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Ten consecutive 10 cm bins per category, giving four 1 m bins (0–1, 1–2, 2–3, 3–4 m), exactly as the task specifies, and exactly 25% of bins each.

ii.
```python
N_POSITION_BINS_CORRIDOR = 40
position_values = [f"{start}-{start + 1}m" for start in range(4)]
position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
```

iii. *"This keeps the paper's running-only spatial interpolation while matching ... the requested 4 × 1 m corridor-position output bins."*

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Perfectly by construction — the neural bin and the position label are the same grid index.

ii.
```python
interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR]   # bin k == position k dm
...
output_trial = np.vstack([..., position_indices, ...])
```

iii. Not separately argued.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. `beh['ft_RunSpeed']`, the per-frame running speed — the same variable as the reference.

ii.
```python
move_speed = np.asarray(beh["ft_RunSpeed"][:nframes][vr_move], dtype=np.float32)[None, :]
```

iii. `CONVERSION_NOTES.md`: *"`running_speed_bin`: quartiles computed over all interpolated corridor time bins in the full export."*

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speed is restricted to running frames (`ft_move > 0`) and pushed through the *same* `spk_pos_interp` call as the neural data, so each trial gets 60 position-interpolated speed values of which the first 40 are kept. Stationary (zero-speed) frames are therefore absent from the distribution entirely, and each value is a position-weighted interpolation rather than a raw sample. All per-trial speed vectors are accumulated so that global quantiles can be taken after the session loop.

ii.
```python
interp_speed = interpolate_running_signal(move_speed, move_position, corridor_length, ntrials)[0]
interp_speed = interp_speed[:, :N_POSITION_BINS_CORRIDOR].astype(np.float32, copy=False)
all_speed_values.append(interp_speed.reshape(-1))
```

iii. Implicit: treating speed like every other stream keeps it on the same grid. Restriction to running frames follows the Methods (*"We only considered timepoints during running for analysis"*).

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three quartile edges are computed with `np.quantile` over **all** interpolated speed values in the whole export, and every bin is assigned with `np.digitize`. Edges: 15.07, 24.91, 38.00. Globally this gives exactly 25% per bin; per session it does not — e.g. `TX109_2023_03_27_1` is 85.2% / 14.2% / 0.6% / 0.0% (one class empty) and `VR2_2021_04_11_1` is 3.6% / 6.3% / 16.4% / 73.7%. The reference instead ranks within each session, guaranteeing 25% per session.

ii.
```python
speed_values = np.concatenate(all_speed_values)
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
for session_outputs, session_speeds in zip(output_sessions, speed_sessions):
    for output_trial, speed_trial in zip(session_outputs, session_speeds):
        output_trial[3, :] = np.digitize(speed_trial, speed_edges, right=False).astype(np.int16)
```

iii. `CONVERSION_NOTES.md`: *"Running-speed bins are exactly balanced on the full dataset by global quartile construction"*, matching the task requirement *"4 bins, each corresponding to 25% of the data"*.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Same position grid — the speed trace is interpolated with the identical call and the identical `accum_pos` used for the neural traces, so bin *k* of the speed label and bin *k* of the neural matrix come from the same interpolation.

ii.
```python
interp_spk   = interpolate_running_signal(move_spk,   move_position, corridor_length, ntrials)
interp_speed = interpolate_running_signal(move_speed, move_position, corridor_length, ntrials)[0]
```

iii. Not separately argued.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Partly handled. Every behaviour stream is truncated to the number of imaged frames (`[:nframes]`), which is the main real defect in this dataset. Licks outside the 4 m corridor are dropped. `np.nanmean` / `np.nanstd` are used for d′ so NaN frames do not poison the statistic, and a small epsilon guards the d′ denominator. Three conditions `raise` (unexpected corridor length, non-finite `SoundPos`, missing familiar-corridor frames); because there is no per-session `try/except`, any of these would abort the whole multi-hour conversion rather than skip one session. Nothing checks that a trial index actually falls inside the imaged period: `utils.interp_value` uses `fill_value='extrapolate'`, so a trailing trial that was never imaged would be silently filled with extrapolated activity rather than dropped (the reference explicitly drops such a trial). There is also no `n_trials >= 2` / `n_units > 0` guard, though all 28 exported sessions satisfy both.

ii.
```python
nframes = spk.shape[1]
vr_move = np.asarray(beh["ft_move"][:nframes] > 0)
corridor_frames = np.asarray(beh["ft_CorrSpc"][:nframes], dtype=bool)
wall_ids = np.asarray(beh["ft_WallID"][:nframes])
```
```python
dp = 2.0 * (stim1_mean - stim2_mean) / (stim1_std + stim2_std + 1e-12)
```
```python
valid_positions = lick_positions[(lick_positions >= 0) & (lick_positions < nbins)]
```

iii. Not argued explicitly; the truncation pattern is copied from `utils.Get_dprime_selective_neuron` (`nfr = spk.shape[1]`, `beh[...][:nfr]`).

## 12-a. What are the most time-consuming steps of the code?

i. (1) Reading the per-session spike files — 1.5–20 GB each, hundreds of GB overall; the run took roughly 2.5 hours and the AI itself had to check whether a "stall" at session 15 was a hang. (2) The d′ computation, which materialises four full boolean-indexed copies of the (50k–90k) × (10k–50k) spike matrix. (3) `utils.spk_pos_interp`, a Python loop that builds a `scipy.interpolate.interp1d` object per neuron (512 per session). (4) Re-reading `Beh_*.npy` files, twice per session.

ii.
```python
spk = utils.load_spk({...}, root=SPK_DIR)
```
```python
stim1_mean = np.nanmean(spk[:, stim1_frames], axis=1)
...
stim2_std = np.nanstd(spk[:, stim2_frames], axis=1)
```
```python
for s in range(raw_spk.shape[0]):            # utils.spk_pos_interp
    spk_resh.append(np.reshape(interp_value(raw_spk[s, :], accum_pos/corridorLen, linPos), ...))
```

iii. Step 133/185: *"the interpolation step is the dominant cost"*; the AI added `--limit-sessions` specifically so it could test the pipeline without paying the full I/O cost.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Three. (1) `select_neurons` appends neuron indices one at a time in Python loops (and maintains a parallel `set`); this is pure index bookkeeping that `np.argsort`/`np.isin` would do in one pass. (2) The per-neuron loop inside `utils.spk_pos_interp`: since all neurons share one `accum_pos → linPos` mapping, a single `np.interp`-style computation (or one `interp1d` on a 2-D `y` with `axis=1`) would replace 512 separate interpolator objects. (3) The final speed-binning double loop over sessions and trials, which could be one `np.digitize` on the concatenated array. Only (2) is material relative to I/O.

ii.
```python
for neuron_idx in order[:MIN_AREA_NEURONS]:
    selected.append(int(neuron_idx))
    selected_set.add(int(neuron_idx))
```
```python
for session_outputs, session_speeds in zip(output_sessions, speed_sessions):
    for output_trial, speed_trial in zip(session_outputs, session_speeds):
        output_trial[3, :] = np.digitize(speed_trial, speed_edges, right=False).astype(np.int16)
```

iii. Not discussed by the AI.

## 12-c. What processing does the code repeat multiple times?

i. (1) Behaviour files are re-read from disk for every session: once in `collect_stimulus_names` and again in the main loop, with no caching, so a `Beh_*.npy` holding *n* sessions is unpickled 2*n* times. The reference groups records by behaviour file and reads each exactly once. (2) `spk[:, stim1_frames]` and `spk[:, stim2_frames]` are each materialised twice (once for the mean, once for the std) — four large copies where two would do. (3) The retinotopy `_trans.npz` is re-read per session even when a mouse has several sessions on the same date/registration.

ii.
```python
def collect_stimulus_names(catalog):
    for session in catalog:
        beh = load_behavior(session["behavior_group"])[session["behavior_key"]]   # full file re-read
        stimulus_names.update(map(str, beh["UniqWalls"]))
...
for session_idx, session in enumerate(catalog, start=1):
    beh = load_behavior(session["behavior_group"])[session["behavior_key"]]       # read again
```
```python
stim1_mean = np.nanmean(spk[:, stim1_frames], axis=1)
stim1_std  = np.nanstd(spk[:, stim1_frames], axis=1)
```

iii. Not discussed.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) `spk_pos_interp` produces 60 position bins per trial and the last 20 (the grey space) are immediately thrown away — a third of the most expensive computation, for both the neural and the speed streams. (2) In `select_neurons` the final `np.argsort(abs_dp[selected])[::-1]` ordering is discarded on the very next line by `np.sort(selected)`. (3) `training_day_rank` is computed for every session but only stored in metadata, never used as an input. (4) `speed_sessions` keeps a second full copy of every trial's interpolated speed alive until the end of the run purely to re-bin it later, roughly doubling the peak behavioural memory. (5) The whole `make_sample_dataset` path writes an extra `sample_data.pkl` (plus the `tmp_two_sessions*.pkl` debugging artefacts left in `/app`) that no downstream analysis consumes. (6) `stimulus_counts` / `session_stimulus_counter` bookkeeping is purely informational.

ii.
```python
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR]      # 60 computed, 40 kept
```
```python
selected = selected[np.argsort(abs_dp[selected])[::-1]]
return np.sort(selected)                                      # previous line's order discarded
```
```python
session_speed.append(np.ascontiguousarray(interp_speed[trial_idx]))
...
speed_values = np.concatenate(all_speed_values)
```

iii. Not discussed. The 60-bin grid is inherited from `utils.spk_pos_interp`, whose `new_shape=[ntrials, 0]` argument forces `bins = corridorLen = 60`; the AI chose to slice afterwards rather than interpolate 40 bins directly.
