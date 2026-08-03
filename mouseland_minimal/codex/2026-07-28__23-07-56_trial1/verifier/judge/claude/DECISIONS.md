# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three directories: `beh/` for behavior, `spk/` for deconvolved calcium traces, and `retinotopy/` for visual area assignments. It reads `Imaging_Exp_info.npy` as the master index, but only iterates over the `sup_*` (supervised/rewarded) experiment groups, explicitly excluding unsupervised and naive cohorts. For each session, it loads the behavior dict from `Beh_<group>.npy`, the spike data via `utils.load_spk()`, and the retinotopy from `*_trans.npz`. It also imports and uses `utils.py` from the reference code directory for spike loading and spatial interpolation.

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

exp_info = load_exp_info()
catalog = attach_session_days(build_session_catalog(exp_info))

# Only iterates over SUP_GROUP_PRIORITY keys:
for group_name in SUP_GROUP_PRIORITY:
    for entry in exp_info[group_name]:
        ...
```

iii. The AI justified restricting to supervised groups because "the decoder input requires reward availability per trial. Unsupervised and naive cohorts do not have rewarded corridors."

## 1-b. How are the data split into subjects?

i. Subjects are extracted from `entry['mname']` in the catalog. Unique subjects are sorted and mapped to indices. This yields 5 mice (only the supervised cohort).

ii.
```python
subjects = sorted({session["subject"] for session in catalog})
subject_to_index = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. Subject splitting follows directly from the metadata. The restriction to 5 mice is a consequence of only including supervised sessions.

## 1-c. How are the data split into sessions?

i. A session is identified by `mname_datexp_blk`. Duplicates across experiment types are deduplicated by keeping the first match according to `SUP_GROUP_PRIORITY` ordering. This yields 28 unique sessions.

ii.
```python
session_candidates = defaultdict(list)
for group_name in SUP_GROUP_PRIORITY:
    for entry in exp_info[group_name]:
        base_session_id = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
        ...
        session_candidates[base_session_id].append(...)

# Keep canonical (first priority) entry per session
for base_session_id in sorted(session_candidates):
    candidates = sorted(
        session_candidates[base_session_id],
        key=lambda item: SUP_GROUP_PRIORITY.index(item["group_name"]),
    )
    canonical = candidates[0]
```

iii. The deduplication ensures each physical recording is processed once, prioritized by experiment group.

## 1-d. How are the data split into trials?

i. Trials are taken directly from `beh['ntrials']`. All trials are processed; no trials are dropped. For each trial, neural data is spatially interpolated using `utils.spk_pos_interp()` across the corridor, producing a fixed-size representation per trial.

ii.
```python
ntrials = int(beh["ntrials"])
...
for trial_idx in range(ntrials):
    ...
    session_neural.append(np.ascontiguousarray(interp_spk[:, trial_idx, :]))
```

iii. All trials are included since the spatial interpolation produces a fixed-length representation for every trial.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. All `ntrials` trials are kept for every session. The only filtering is at the neuron level (d' threshold and neuron cap).

ii.
```python
for trial_idx in range(ntrials):
    # No filtering condition - all trials processed
    ...
```

iii. The AI did not document any trial filtering rationale.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the `*_neural_data.npy` files (loaded via `utils.load_spk()`), plus `iarea` from the retinotopy files. Additionally, `ft_move`, `ft_CorrSpc`, and `ft_PosCum` are used to select running-only corridor frames and spatially interpolate the neural data.

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

iii. The AI uses the same raw spike data as the reference, loaded through the reference code's own `utils.load_spk()` function.

## 2-b. How is the `neural` data processed?

i. The neural data undergoes substantial processing: (1) running-only frames are selected via `ft_move > 0`, (2) the data is spatially interpolated onto a 60-bin position grid using `utils.spk_pos_interp()`, (3) only the first 40 bins (the 4m corridor) are kept, (4) cast to float16. This converts the data from a temporal representation to a spatial (position-based) representation.

ii.
```python
vr_move = np.asarray(beh["ft_move"][:nframes] > 0)
move_spk = np.asarray(spk[selected_neurons][:, vr_move], dtype=np.float32)
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR].astype(np.float16, copy=False)
```

iii. The AI followed the reference paper's spatial interpolation approach used in the paper's analysis code, converting temporal neural traces to position-binned activity.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Extensive neuron filtering is applied: (1) neurons outside the 4 visual areas (V1, mHV, lHV, aHV) are excluded, (2) d-prime is computed between familiar rewarded and familiar unrewarded corridors, (3) neurons with |d'| < 0.3 are excluded, (4) a maximum of 512 neurons per session is enforced, with balanced selection across areas.

ii.
```python
DP_THRESHOLD = 0.3
MAX_NEURONS = 512

candidate_mask = (region_idx != REGION_TO_INDEX["outside_visual_cortex"]) & (np.abs(dp) >= DP_THRESHOLD)

# Balanced selection across V1, mHV, lHV, aHV
for region_name in AREA_SELECTION_ORDER:
    region_code = REGION_TO_INDEX[region_name]
    region_candidates = np.flatnonzero(candidate_mask & (region_idx == region_code))
    order = region_candidates[np.argsort(abs_dp[region_candidates])[::-1]]
    for neuron_idx in order[:MIN_AREA_NEURONS]:
        selected.append(int(neuron_idx))
```

iii. The AI noted: "Exporting every ROI trial-by-trial would not be trainable by the provided decoder" and chose to follow "the paper's selective-neuron threshold used throughout the figure code."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry / trial start. However, rather than using temporal alignment (frame-based), the AI uses spatial alignment via position-based interpolation. Each trial is represented as 40 position bins (10cm each) across the 4m corridor, with neural activity interpolated to each position.

ii.
```python
move_position = np.asarray(beh["ft_PosCum"][:nframes][vr_move], dtype=np.float64)
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR]
```

iii. The AI stated: "Trials are aligned to corridor entry, use running-only frames, and neural activity is interpolated onto 10 cm corridor bins across the 4 m corridor."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses spatial bins rather than temporal bins. Each bin is 10cm of corridor position, which at the VR speed of 0.6 m/s corresponds to ~166.67ms. This is a spatial rebinning, not a temporal one. The original temporal resolution of ~315ms (3.17 Hz imaging) is lost.

ii.
```python
POSITION_STEP_M = 0.1
VR_SPEED_M_PER_S = 0.6
TIME_BIN_SIZE_MS = POSITION_STEP_M / VR_SPEED_M_PER_S * 1000.0  # = 166.6667 ms
```

iii. The AI chose to follow the paper's spatial interpolation grid, which changes the temporal resolution from the imaging frame rate to a position-derived equivalent.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundPos`, the position at which the sound cue was played in each trial. The time grid is derived from the position bins (position / 6.0 seconds).

ii.
```python
cue_position = float(sound_positions[trial_idx])
# ...
input_trial = np.vstack([
    cue_position / 6.0 - time_since_start,
    ...
])
```

iii. The AI uses `SoundPos` (position-based) rather than `SoundFr` (frame-based) because the representation is spatial.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The cue position is divided by 6.0 to convert to time (at 0.6 m/s VR speed, position in decimeters / 6 = time in seconds). The time to cue is then `cue_time - current_bin_time`, so it's positive before the cue and negative after.

ii.
```python
cue_position / 6.0 - time_since_start
```
where `time_since_start = position_units / 6.0` and `position_units = np.arange(N_POSITION_BINS_CORRIDOR)` (i.e., 0 to 39 in 10cm units).

iii. The conversion from position to time assumes a constant VR speed of 0.6 m/s.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on the same 40-bin position grid as the neural data, so alignment is by position bin index.

ii.
```python
time_since_start = position_units / 6.0
# ...
cue_position / 6.0 - time_since_start
```

iii. All variables share the same spatial grid.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the session date string `entry['datexp']`, parsed as a calendar date. Two values are computed: `training_day_index` (calendar days elapsed since the mouse's first session) and `training_day_rank` (ordinal session count). The `training_day_index` (calendar days) is used in the exported data.

ii.
```python
def attach_session_days(catalog):
    for subject_sessions in by_subject.values():
        subject_sessions.sort(key=lambda item: parse_date(item["date"]))
        first_date = parse_date(subject_sessions[0]["date"])
        for index, session in enumerate(subject_sessions):
            session_date = parse_date(session["date"])
            session["training_day_index"] = float((session_date - first_date).days)
            session["training_day_rank"] = int(index)
```

iii. The AI chose calendar days elapsed rather than session rank.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The first session date for each mouse is found, and the number of calendar days between each session date and the first date is computed. This value is broadcast across all 40 position bins of each trial. Values range from 0 to 73.

ii.
```python
session["training_day_index"] = float((session_date - first_date).days)
# ...
np.full(N_POSITION_BINS_CORRIDOR, session["training_day_index"], dtype=np.float32)
```

iii. The AI noted this gives elapsed calendar days, which can result in large gaps between sessions.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from the position grid itself. Since each bin is 10cm at 0.6 m/s VR speed, the time at each bin is `bin_index * 0.1m / 0.6 m/s`.

ii.
```python
position_units = np.arange(N_POSITION_BINS_CORRIDOR, dtype=np.float32)
time_since_start = position_units / 6.0
```

iii. The AI derives time from the spatial grid assuming constant VR speed, rather than using actual frame timestamps.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. A fixed array `[0, 1/6, 2/6, ..., 39/6]` seconds is used for every trial. This is deterministic from the position grid and does not vary between trials.

ii.
```python
time_since_start = position_units / 6.0  # 0.0 to 6.5 seconds
```

iii. Since the representation is spatial, the time is derived from position rather than measured.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It shares the same 40-bin position grid as the neural data, aligned by bin index.

ii.
```python
time_since_start = position_units / 6.0
```

iii. Same spatial grid alignment as all other variables.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `beh['isRew']`, which marks whether each trial is in a rewarded corridor.

ii.
```python
trial_is_rewarded = np.asarray(beh["isRew"], dtype=bool)
# ...
np.full(N_POSITION_BINS_CORRIDOR, float(trial_is_rewarded[trial_idx]), dtype=np.float32)
```

iii. Direct use of the reward flag from the behavior data.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew` value is cast to float (0.0 or 1.0) and broadcast across all 40 position bins. No additional processing.

ii.
```python
np.full(N_POSITION_BINS_CORRIDOR, float(trial_is_rewarded[trial_idx]), dtype=np.float32)
```

iii. Straightforward conversion from boolean to float.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `beh['WallName']`, which names the texture on the walls of each trial's corridor. Also uses `beh['UniqWalls']` to collect all unique stimulus names across the dataset.

ii.
```python
stimulus_names = collect_stimulus_names(catalog)  # sorted unique wall names
stimulus_to_index = {name: idx for idx, name in enumerate(stimulus_names)}
# ...
trial_stimulus = str(trial_wall_names[trial_idx])
np.full(N_POSITION_BINS_CORRIDOR, stimulus_to_index[trial_stimulus], dtype=np.int16)
```

iii. The AI uses individual stimulus names as categories.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each unique wall name (e.g., `circle1`, `leaf2`, `wood1_swap2`) is treated as a separate category, yielding 14 categories. This contrasts with the reference which groups them into 4 base textures (circle, leaf, rock, wood). The category index is broadcast across all 40 position bins.

ii.
```python
stimulus_names = collect_stimulus_names(catalog)
# Returns: ['circle1', 'circle2', 'circle3', 'leaf1', 'leaf1_swap1', 'leaf1_swap2',
#           'leaf2', 'leaf3', 'rock1', 'rock2', 'wood1', 'wood1_swap2', 'wood2', 'wood5']
stimulus_to_index = {name: idx for idx, name in enumerate(stimulus_names)}
```

iii. The AI treats each unique wall name as its own category rather than grouping by base texture.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `beh['LickPos']` (lick positions in corridor) and `beh['LickTrind']` (trial index of each lick), rather than `beh['LickFr']` (lick frame numbers).

ii.
```python
lick_positions = np.asarray(beh["LickPos"], dtype=np.float32)
lick_trial_index = np.asarray(beh["LickTrind"], dtype=np.int64)
```

iii. Since the representation is spatial, lick positions are used instead of lick frames.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, licks belonging to that trial (identified by `LickTrind`) are binned into the 40 position bins. A bin is 1 if at least one lick falls in it, 0 otherwise. Only licks within the corridor range (0 to 40 decimeters) are included.

ii.
```python
def bin_licks(lick_positions, nbins):
    lick_bins = np.zeros(nbins, dtype=np.int16)
    valid_positions = lick_positions[(lick_positions >= 0) & (lick_positions < nbins)]
    indices = np.floor(valid_positions).astype(int)
    lick_bins[np.clip(indices, 0, nbins - 1)] = 1
    return lick_bins

lick_bins = bin_licks(lick_positions[lick_trial_index == trial_idx], N_POSITION_BINS_CORRIDOR)
```

iii. Position-based lick binning matches the spatial representation.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is binned into the same 40 position bins as the neural data, so alignment is by position bin index.

ii.
```python
lick_bins = bin_licks(lick_positions[lick_trial_index == trial_idx], N_POSITION_BINS_CORRIDOR)
```

iii. Same spatial grid as neural data.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is deterministic from the spatial grid itself. Since the data is spatially interpolated onto 40 bins of 10cm each, the position bin is directly determined by the bin index.

ii.
```python
position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
```

iii. Each of the four 1m position bins spans 10 of the 40 spatial bins.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The 40 position bins are mapped to 4 one-meter bins by integer division: bins 0-9 -> category 0, bins 10-19 -> category 1, etc. This is identical for every trial since the spatial grid is fixed.

ii.
```python
position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
# [0,0,0,0,0,0,0,0,0,0, 1,1,..., 2,2,..., 3,3,...]
```

iii. Deterministic from the spatial grid.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four equal-length 1m bins: 0-1m, 1-2m, 2-3m, 3-4m. Each spans exactly 10 of the 40 position bins.

ii.
```python
position_values = [f"{start}-{start + 1}m" for start in range(4)]
position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
```

iii. Matches the instruction's requirement for 4 equal-length 1m bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Same 40-bin spatial grid as neural data.

ii.
```python
position_indices = np.repeat(np.arange(4, dtype=np.int16), 10)
```

iii. Deterministic from the grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `beh['ft_RunSpeed']`, the running speed at each imaging frame. Only running frames (`ft_move > 0`) are used, and the speed is spatially interpolated using `utils.spk_pos_interp()`.

ii.
```python
move_speed = np.asarray(beh["ft_RunSpeed"][:nframes][vr_move], dtype=np.float32)[None, :]
interp_speed = interpolate_running_signal(move_speed, move_position, corridor_length, ntrials)[0]
interp_speed = interp_speed[:, :N_POSITION_BINS_CORRIDOR]
```

iii. Speed is interpolated onto the same spatial grid as neural data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is spatially interpolated, then discretized into 4 quartile bins using global quantiles computed over all interpolated speed values across all sessions. The quartile edges are computed once after all sessions are processed.

ii.
```python
speed_values = np.concatenate(all_speed_values)
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)

for session_outputs, session_speeds in zip(output_sessions, speed_sessions):
    for output_trial, speed_trial in zip(session_outputs, session_speeds):
        output_trial[3, :] = np.digitize(speed_trial, speed_edges, right=False)
```

iii. Global quartiles ensure each bin contains 25% of all data across the dataset.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Global quartile edges are computed from all interpolated speed values: [15.07, 24.91, 38.00]. Values are discretized using `np.digitize`, giving bins 0-3 (speed_q1 through speed_q4).

ii.
```python
speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75])
output_trial[3, :] = np.digitize(speed_trial, speed_edges, right=False)
```

iii. The AI explicitly computed and reported the quartile edges.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is spatially interpolated onto the same 40-bin grid as neural data, using the same `utils.spk_pos_interp()` function.

ii.
```python
interp_speed = interpolate_running_signal(move_speed, move_position, corridor_length, ntrials)[0]
interp_speed = interp_speed[:, :N_POSITION_BINS_CORRIDOR]
```

iii. Same spatial interpolation and grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI checks for non-finite `SoundPos` values and raises an error if found. It clips lick positions to valid ranges. It clips behavior arrays to `nframes` (number of neural frames). No trials are dropped; all trials from `ntrials` are kept.

ii.
```python
if not np.isfinite(cue_position):
    raise ValueError(...)
valid_positions = lick_positions[(lick_positions >= 0) & (lick_positions < nbins)]
nframes = spk.shape[1]
vr_move = np.asarray(beh["ft_move"][:nframes] > 0)
```

iii. The AI validates data integrity but does not implement graceful handling of edge cases like empty trials.

## 12-a. What are the most time-consuming steps of the code?

i. Reading the spike files (each is several GB) and running the spatial interpolation via `utils.spk_pos_interp()` are the most time-consuming steps.

ii.
```python
spk = utils.load_spk(...)
interp_spk = interpolate_running_signal(move_spk, move_position, corridor_length, ntrials)
```

iii. The I/O cost of loading large neural data files dominates runtime.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop that assembles input/output arrays could potentially be vectorized, as most operations are simple array constructions. The lick binning per trial could also be vectorized.

ii.
```python
for trial_idx in range(ntrials):
    ...
    lick_bins = bin_licks(lick_positions[lick_trial_index == trial_idx], N_POSITION_BINS_CORRIDOR)
    input_trial = np.vstack([...])
    output_trial = np.vstack([...])
```

iii. The per-trial loop is not the bottleneck compared to I/O and interpolation.

## 12-c. What processing does the code repeat multiple times?

i. The `collect_stimulus_names()` function loads every behavior file once just to collect stimulus names, and then each behavior file is loaded again during the main conversion loop.

ii.
```python
def collect_stimulus_names(catalog):
    for session in catalog:
        beh = load_behavior(session["behavior_group"])[session["behavior_key"]]
        stimulus_names.update(map(str, beh["UniqWalls"]))
```

iii. This double-loading of behavior files is unnecessary since stimulus names could be collected during the main processing loop.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI includes `outside_visual_cortex` as a brain region in the output (5 regions instead of 4), but no neurons are assigned to it (0 count). The `training_day_rank` is computed but not used (only `training_day_index` is exported). The `collect_stimulus_names` pre-pass is redundant. The spatial interpolation processes all 60 position bins but only the first 40 are kept.

ii.
```python
GROUPED_REGIONS = ["V1", "mHV", "lHV", "aHV", "outside_visual_cortex"]
# ...
session["training_day_rank"] = int(index)  # computed but not used in export
# ...
interp_spk = interp_spk[:, :, :N_POSITION_BINS_CORRIDOR]  # discard bins 40-59
```

iii. The extra brain region and unused computations add minor overhead and clutter.
