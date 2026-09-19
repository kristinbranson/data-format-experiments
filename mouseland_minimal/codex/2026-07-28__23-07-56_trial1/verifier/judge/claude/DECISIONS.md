# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads `Imaging_Exp_info.npy` as the master index, then iterates only over the supervised experiment groups (`sup_*`). For each session, it loads the behavior file (`Beh_<group>.npy`), the spike data via `utils.load_spk()` from the reference code, and the retinotopy file (`*_trans.npz`) for neuron area assignments. It also uses `utils.spk_pos_interp()` from the reference code for spatial interpolation.

ii.
```python
exp_info = np.load(EXP_INFO_PATH, allow_pickle=True).item()
# ...
SUP_GROUP_PRIORITY = [
    "sup_train1_before_learning",
    "sup_train1_after_learning",
    "sup_test1",
    "sup_train2_before_learning",
    "sup_train2_after_learning",
    "sup_test2",
    "sup_test3",
]
# ...
for group_name in SUP_GROUP_PRIORITY:
    for entry in exp_info[group_name]:
        base_session_id = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
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

iii. The agent examined the experiment groups in the index and decided to only include the `sup_*` (supervised/rewarded) groups, reasoning that the decoder input requires a meaningful `reward_availability` variable and unsupervised/naive cohorts do not have rewarded corridors. The agent used `utils.load_spk()` from the reference code rather than loading the spike files directly.

## 1-b. How are the data split into subjects?

i. The mouse name is taken from `entry['mname']` in the session catalog. Only 5 mice from the supervised cohort are included (TX60, TX61, TX108, TX109, VR2), yielding 28 sessions.

ii.
```python
subjects = sorted({session["subject"] for session in catalog})
subject_to_index = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The agent restricted to supervised-task mice because the decoder input specification requires reward_availability. The unsupervised and naive cohorts were excluded entirely.

## 1-c. How are the data split into sessions?

i. A session is one mouse on one date in one block. When a recording appears under multiple `sup_*` groups, the agent deduplicates by selecting the one with highest priority in `SUP_GROUP_PRIORITY`. This yields 28 unique sessions.

ii.
```python
session_candidates = defaultdict(list)
for group_name in SUP_GROUP_PRIORITY:
    for entry in exp_info[group_name]:
        base_session_id = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
        session_candidates[base_session_id].append(...)
# ...
candidates = sorted(
    session_candidates[base_session_id],
    key=lambda item: SUP_GROUP_PRIORITY.index(item["group_name"]),
)
canonical = candidates[0]
```

iii. The agent recognized that recordings can appear under multiple experiment type groups and used a priority-based deduplication to select the canonical behavior dict for each recording.

## 1-d. How are the data split into trials?

i. Trials are taken directly from `beh['ntrials']`. All trials in each session are included without filtering. However, within each trial, neural data is spatially interpolated onto 40 position bins (the 4m corridor) using only running-only frames (`ft_move > 0`), yielding fixed-length trials of 40 bins each.

ii.
```python
ntrials = int(beh["ntrials"])
# ...
for trial_idx in range(ntrials):
    # ...
    session_neural.append(np.ascontiguousarray(interp_spk[:, trial_idx, :]))
```

iii. The agent followed the reference code's approach of using `spk_pos_interp()` to map variable-length trials onto a uniform spatial grid, making all trials the same length (40 bins).

## 1-e. How are trials filtered based on quality controls?

i. No trial-level filtering is applied. All `ntrials` trials from each session are included. Only running-only frames are used during spatial interpolation (frames where `ft_move > 0`), which removes stationary periods.

ii.
```python
vr_move = np.asarray(beh["ft_move"][:nframes] > 0)
corridor_frames = np.asarray(beh["ft_CorrSpc"][:nframes], dtype=bool)
valid_corridor_frames = vr_move & corridor_frames
```

iii. The agent relied on the spatial interpolation pipeline to handle trial quality implicitly -- stationary frames are excluded from the interpolation by using only running frames.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From the deconvolved calcium traces loaded via `utils.load_spk()`, which reads `spk/<session_id>_neural_data.npy` and concatenates planes. The visual area comes from `iarea` in the retinotopy file. d-prime is computed between familiar rewarded vs unrewarded corridor activity during running.

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
```

iii. The agent used the reference code's `utils.load_spk()` function to load spike data, which performs the same concatenation of imaging planes.

## 2-b. How is the `neural` data processed?

i. Neural data undergoes three processing steps: (1) neuron selection based on d-prime (|d'| >= 0.3) with a cap of 512 neurons per session, balanced across visual areas; (2) only running frames are kept (`ft_move > 0`); (3) spatial interpolation via `utils.spk_pos_interp()` maps running frames onto 40 uniform position bins across the 4m corridor. The result is stored as float16.

ii.
```python
selected_neurons = select_neurons(dp, region_idx_all, max_neurons=max_neurons)
# ...
move_spk = np.asarray(spk[selected_neurons][:, vr_move], dtype=np.float32)
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR].astype(np.float16, copy=False)
```

iii. The agent justified d-prime filtering as matching the paper's analysis pipeline (d' threshold of 0.3 found in `data_process_script.ipynb`). The 512 neuron cap was motivated by memory constraints and decoder trainability, as raw sessions had 20,000-90,000 neurons. The spatial interpolation followed the reference code's `spk_pos_interp()` function.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) only neurons in V1, mHV, lHV, or aHV visual areas are eligible; (2) neurons must pass a d-prime selectivity threshold of |d'| >= 0.3 between familiar rewarded and unrewarded corridor activity. Additionally, at most 512 neurons are kept per session, balanced across the four areas (128 minimum per area, then remaining slots by highest |d'|).

ii.
```python
DP_THRESHOLD = 0.3
MAX_NEURONS = 512
# ...
candidate_mask = (region_idx != REGION_TO_INDEX["outside_visual_cortex"]) & (np.abs(dp) >= DP_THRESHOLD)
# ...
for region_name in AREA_SELECTION_ORDER:
    region_code = REGION_TO_INDEX[region_name]
    region_candidates = np.flatnonzero(candidate_mask & (region_idx == region_code))
    order = region_candidates[np.argsort(abs_dp[region_candidates])[::-1]]
    for neuron_idx in order[:MIN_AREA_NEURONS]:
        selected.append(int(neuron_idx))
```

iii. The agent noted this approach matches the paper's neuron selection methodology and was necessary because exporting all 20,000-90,000 neurons per session was impractical for the decoder.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to corridor entry (trial start). Neural data is spatially interpolated onto position bins using `utils.spk_pos_interp()`, which maps the accumulated position during running onto a uniform grid. The first bin corresponds to corridor entry (position 0). All trials have exactly 40 bins (the 4m corridor at 10cm resolution).

ii.
```python
move_position = np.asarray(beh["ft_PosCum"][:nframes][vr_move], dtype=np.float64)
move_spk = np.asarray(spk[selected_neurons][:, vr_move], dtype=np.float32)
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR].astype(np.float16, copy=False)
```

iii. The agent followed the reference code's approach of spatial interpolation, aligning to corridor entry by using position rather than time as the primary axis.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is binned spatially, not temporally. Each bin is 0.1m (10cm), and the implied time bin size is 166.67ms (0.1m / 0.6 m/s VR speed). This is different from the raw imaging frame rate of 3.17 Hz (315ms). The spatial interpolation via `spk_pos_interp()` effectively rebins the data from temporal frames to spatial position bins.

ii.
```python
POSITION_STEP_M = 0.1
VR_SPEED_M_PER_S = 0.6
TIME_BIN_SIZE_MS = POSITION_STEP_M / VR_SPEED_M_PER_S * 1000.0  # = 166.67 ms
```

iii. The agent derived the time bin size from the spatial bin size and VR speed, following the spatial interpolation approach in the reference code.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundPos` (the position at which the sound cue was played, in position-bin units) and the deterministic position-to-time mapping based on VR speed.

ii.
```python
cue_position = float(sound_positions[trial_idx])
# ...
input_trial = np.vstack([
    cue_position / 6.0 - time_since_start,
    # ...
])
```

iii. The agent used position-based cue timing rather than frame-based (`SoundFr`), consistent with its spatial interpolation approach.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The cue position (from `SoundPos`) is converted to time by dividing by 6.0 (the VR speed in bins/second). The time-to-cue at each spatial bin is `cue_position / 6.0 - time_since_start`, giving positive values before the cue and negative after.

ii.
```python
time_since_start = position_units / 6.0  # position_units = np.arange(40)
# ...
cue_position / 6.0 - time_since_start
```

iii. The agent computed time-to-cue as the difference between the cue's implied time and the current bin's implied time, both derived from the spatial grid.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Both are on the same 40-bin spatial grid. The time-to-cue array has exactly 40 elements matching the 40 neural columns.

ii.
```python
input_trial = np.vstack([
    cue_position / 6.0 - time_since_start,
    # ...  # shape: (4, 40)
])
```

iii. Alignment is implicit through the shared spatial grid.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the session dates (`entry['datexp']`), parsed into datetime objects.

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
```

iii. The agent parsed the date strings from the session catalog to compute calendar days since the first session.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The day of training is computed as elapsed calendar days since the first session for that mouse (e.g., if first session is day 0 and the next is 3 days later, it's 3.0). This is broadcast as a constant across all 40 bins of each trial.

ii.
```python
session["training_day_index"] = float((session_date - first_date).days)
# ...
np.full(N_POSITION_BINS_CORRIDOR, session["training_day_index"], dtype=np.float32)
```

iii. The agent used calendar days rather than ordinal session count, providing a more precise measure of training progression.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From the deterministic spatial bin index, converted to time using the VR speed (0.6 m/s).

ii.
```python
position_units = np.arange(N_POSITION_BINS_CORRIDOR, dtype=np.float32)
time_since_start = position_units / 6.0
```

iii. Because the data is spatially interpolated, time since trial start is a deterministic function of position bin index.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Each position bin index (0-39) is divided by 6.0 (VR speed in bins/sec = 0.6 m/s / 0.1 m/bin = 6 bins/s), giving a monotonic ramp from 0 to ~6.5 seconds. This is the same for every trial.

ii.
```python
time_since_start = position_units / 6.0  # [0.0, 0.167, 0.333, ..., 6.5]
```

iii. The agent derived time from the spatial grid using the known VR speed.

## 5-c. How is `input` *Time since trial start* aligned with the neural data?

i. Both share the same 40-bin spatial grid, so alignment is implicit.

ii.
```python
input_trial = np.vstack([
    cue_position / 6.0 - time_since_start,
    np.full(N_POSITION_BINS_CORRIDOR, session["training_day_index"], dtype=np.float32),
    time_since_start,
    # ...
])
```

iii. Alignment is through the shared spatial position grid.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `beh['isRew']`, which marks whether each trial is in a rewarded corridor.

ii.
```python
trial_is_rewarded = np.asarray(beh["isRew"], dtype=bool)
# ...
np.full(N_POSITION_BINS_CORRIDOR, float(trial_is_rewarded[trial_idx]), dtype=np.float32)
```

iii. Directly taken from the behavior data.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean is cast to float and broadcast across all 40 bins of the trial. No further processing.

ii.
```python
np.full(N_POSITION_BINS_CORRIDOR, float(trial_is_rewarded[trial_idx]), dtype=np.float32)
```

iii. No processing needed beyond type conversion.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `beh['WallName']`, which gives the texture name for each trial.

ii.
```python
trial_wall_names = np.asarray(beh["WallName"]).astype(str)
trial_stimulus = str(trial_wall_names[trial_idx])
```

iii. The agent used WallName as the stimulus identifier for each trial.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The wall name is mapped to an integer index into the sorted list of all 14 unique stimulus names that appear across the dataset. The agent does NOT group textures into 4 base categories (circle/leaf/rock/wood) -- instead it keeps all 14 individual names (circle1, circle2, circle3, leaf1, leaf2, leaf3, leaf1_swap1, leaf1_swap2, rock1, rock2, wood1, wood2, wood5, wood1_swap1, wood1_swap2). The index is broadcast across all 40 bins of each trial.

ii.
```python
stimulus_names = collect_stimulus_names(catalog)  # sorted list of 14 names
stimulus_to_index = {name: idx for idx, name in enumerate(stimulus_names)}
# ...
np.full(N_POSITION_BINS_CORRIDOR, stimulus_to_index[trial_stimulus], dtype=np.int16)
```

iii. The agent collected all unique stimulus names and indexed them individually rather than grouping into base texture categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `beh['LickPos']` (lick positions in position-bin units) and `beh['LickTrind']` (trial index of each lick).

ii.
```python
lick_positions = np.asarray(beh["LickPos"], dtype=np.float32)
lick_trial_index = np.asarray(beh["LickTrind"], dtype=np.int64)
```

iii. The agent used position-based lick data rather than frame-based (`LickFr`), consistent with its spatial interpolation approach.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, lick positions belonging to that trial (identified by `LickTrind`) are binned into the 40-bin spatial grid. A bin is 1 if at least one lick falls in it, 0 otherwise. Lick positions outside the corridor (0-39) are discarded.

ii.
```python
def bin_licks(lick_positions, nbins):
    lick_bins = np.zeros(nbins, dtype=np.int16)
    valid_positions = lick_positions[(lick_positions >= 0) & (lick_positions < nbins)]
    indices = np.floor(valid_positions).astype(int)
    lick_bins[np.clip(indices, 0, nbins - 1)] = 1
    return lick_bins
# ...
lick_bins = bin_licks(lick_positions[lick_trial_index == trial_idx], N_POSITION_BINS_CORRIDOR)
```

iii. The agent binned licks by spatial position to match the spatial grid of the neural data.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Both are on the same 40-bin spatial position grid. Lick positions are mapped to the same bins as the neural data.

ii.
```python
lick_bins = bin_licks(lick_positions[lick_trial_index == trial_idx], N_POSITION_BINS_CORRIDOR)
```

iii. Alignment is through the shared spatial grid.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From the deterministic spatial bin index. Since the data is already on a uniform spatial grid, position is simply the bin index mapped to 4 one-meter bins.

ii.
```python
position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
```

iii. Position is deterministic from the spatial grid structure.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Each group of 10 spatial bins (each 0.1m) maps to one 1-meter position bin. Bins 0-9 -> 0 (0-1m), 10-19 -> 1 (1-2m), 20-29 -> 2 (2-3m), 30-39 -> 3 (3-4m). This is the same for every trial.

ii.
```python
position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
# [0,0,0,0,0,0,0,0,0,0, 1,1,..., 2,2,..., 3,3,...]
```

iii. The position encoding is a deterministic consequence of the spatial binning scheme.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four equal-length 1-meter bins, each covering 10 spatial position bins (0.1m each). The categorization is deterministic from the grid structure.

ii.
```python
position_values = [f"{start}-{start + 1}m" for start in range(4)]
# ['0-1m', '1-2m', '2-3m', '3-4m']
position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
```

iii. The discretization follows the instructions' specification of 4 equal-length 1-m bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Both share the same 40-bin spatial grid, so alignment is implicit and exact.

ii.
```python
output_trial = np.vstack([
    # ...
    position_indices,
    # ...
])
```

iii. Alignment is through the shared spatial grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `beh['ft_RunSpeed']`, the running speed at each imaging frame. Only running frames (`ft_move > 0`) are used.

ii.
```python
move_speed = np.asarray(beh["ft_RunSpeed"][:nframes][vr_move], dtype=np.float32)[None, :]
interp_speed = interpolate_running_signal(move_speed, move_position, corridor_length, ntrials)[0]
interp_speed = interp_speed[:, :N_POSITION_BINS_CORRIDOR].astype(np.float32, copy=False)
```

iii. Running speed is extracted from the behavior data and spatially interpolated.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed undergoes spatial interpolation (same as neural data) via `utils.spk_pos_interp()`, mapping it onto the 40-bin spatial grid. Then global quartile edges are computed across all trials in the full dataset using `np.quantile` at [0.25, 0.5, 0.75], and each bin is assigned to a quartile via `np.digitize`.

ii.
```python
interp_speed = interpolate_running_signal(move_speed, move_position, corridor_length, ntrials)[0]
# ...
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
# ...
output_trial[3, :] = np.digitize(speed_trial, speed_edges, right=False).astype(np.int16)
```

iii. The agent computed global quartiles to ensure each bin contains exactly 25% of all speed observations across the entire dataset.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four quartile-based bins using global thresholds. Bin edges are computed using `np.quantile` at [25%, 50%, 75%] across all interpolated speed values from all sessions, then `np.digitize` assigns each speed value to a bin (0-3).

ii.
```python
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
output_trial[3, :] = np.digitize(speed_trial, speed_edges, right=False).astype(np.int16)
```

iii. The agent used global quartile edges rather than per-session quartiles, ensuring consistent bin boundaries across sessions.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is spatially interpolated using the same `spk_pos_interp()` function and same spatial grid as the neural data, so they are aligned on the 40-bin position grid.

ii.
```python
move_speed = np.asarray(beh["ft_RunSpeed"][:nframes][vr_move], dtype=np.float32)[None, :]
interp_speed = interpolate_running_signal(move_speed, move_position, corridor_length, ntrials)[0]
```

iii. Both neural and speed data go through the same spatial interpolation pipeline.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent validates that corridor length is exactly 60 bins, that `SoundPos` is finite for every trial, and that familiar rewarded/unrewarded corridor frames exist for d-prime computation. Licks outside the corridor (position < 0 or >= 40) are discarded. Behavior streams are clipped to the number of imaged frames (`nframes = spk.shape[1]`). Memory is managed with explicit `del` and `gc.collect()`.

ii.
```python
if corridor_length != N_POSITION_BINS_TOTAL:
    raise ValueError(...)
if not np.isfinite(cue_position):
    raise ValueError(...)
if stim1_frames.sum() == 0 or stim2_frames.sum() == 0:
    raise ValueError(...)
# ...
valid_positions = lick_positions[(lick_positions >= 0) & (lick_positions < nbins)]
```

iii. The agent added explicit validation checks and raises errors for unexpected conditions rather than silently handling them.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the spike files (large numpy arrays) and performing spatial interpolation via `utils.spk_pos_interp()`. The agent also computes d-prime for neuron selection, which requires loading and processing the full spike matrix before selecting neurons.

ii.
```python
spk = utils.load_spk(
    {"mname": session["subject"], "datexp": session["date"], "blk": session["block"]},
    root=SPK_DIR,
)
# ...
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
```

iii. The spike files are the largest data files and dominate I/O time, similar to the reference solution.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop that constructs input and output arrays could potentially be vectorized, since `SoundPos`, `WallName`, and `isRew` are per-trial and could be broadcast. The lick binning per trial is also done in a loop.

ii.
```python
for trial_idx in range(ntrials):
    trial_stimulus = str(trial_wall_names[trial_idx])
    # ...
    lick_bins = bin_licks(lick_positions[lick_trial_index == trial_idx], N_POSITION_BINS_CORRIDOR)
    input_trial = np.vstack([...])
    output_trial = np.vstack([...])
```

iii. The per-trial loop is straightforward but not performance-critical compared to I/O and interpolation.

## 12-c. What processing does the code repeat multiple times?

i. The code loads behavior data twice for stimulus name collection: once in `collect_stimulus_names()` to gather all unique stimulus names, and again in the main conversion loop. Each behavior file is loaded separately rather than cached.

ii.
```python
def collect_stimulus_names(catalog):
    for session in catalog:
        beh = load_behavior(session["behavior_group"])[session["behavior_key"]]
        stimulus_names.update(map(str, beh["UniqWalls"]))
# ...
# Later, in the main loop:
beh = load_behavior(session["behavior_group"])[session["behavior_key"]]
```

iii. The duplicate loading is a minor inefficiency since behavior files are much smaller than spike files.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The agent computes `training_day_rank` (ordinal session index) in addition to `training_day_index` (calendar days), but only `training_day_index` is used as the decoder input. The grey-space portion of the interpolated data (bins 40-59) is computed by `spk_pos_interp()` but immediately discarded. The sample dataset creation is also unnecessary for the main conversion.

ii.
```python
session["training_day_rank"] = int(index)  # computed but not used as input
# ...
interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR]  # discard bins 40-59
```

iii. The grey-space interpolation is a consequence of using the reference code's `spk_pos_interp()` which interpolates the full 60-bin corridor.
