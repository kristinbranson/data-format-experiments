# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by scanning the `spk/` directory for `*_neural_data.npy` files, rather than reading the master index `Imaging_Exp_info.npy`. It loads all `Beh_*.npy` files from `beh/`, merges behavior views for each session (handling swap suffixes), and loads retinotopy files from `retinotopy/`. Spike files and retinotopy are loaded per session.

ii.
```python
spk_session_keys = sorted(
    [path.name.replace("_neural_data.npy", "") for path in SPK_ROOT.glob("*_neural_data.npy")],
    key=natural_key,
)
behavior_views = load_behavior_views()
# In load_behavior_views:
for beh_path in sorted(BEH_ROOT.glob("Beh_*.npy")):
    beh_dict = np.load(beh_path, allow_pickle=True).item()
# Per session:
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
ret = np.load(ret_path, allow_pickle=True)
area_codes = ret["iarea"]
```

iii. The AI chose to discover sessions from spike file names rather than the index file. It also implemented an elaborate behavior view merging system to handle sessions that appear in multiple behavior files, including swap sessions. The CONVERSION_NOTES.md states: "The converter rebuilds sessions at the 89-recording level by stripping `_swap1` and `_swap2` suffixes, merging all behavior views that point to the same physical recording."

## 1-b. How are the data split into subjects?

i. The subject is extracted from the session key by splitting on underscores. Subjects are the sorted unique subject names across all sessions.

ii.
```python
def parse_session_key(session_key: str):
    subject, year, month, day, block = session_key.split("_")
    return {"subject": subject, ...}

subjects = sorted({info["subject"] for info in parsed.values()})
subject_to_id = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The session key encodes the mouse name as the first field, so parsing it gives the subject directly.

## 1-c. How are the data split into sessions?

i. A session is identified by the spike file name, which encodes mouse, date, and block. The AI discovers all 89 sessions from the `spk/` directory. Behavior is merged from multiple `Beh_*.npy` files per session.

ii.
```python
spk_session_keys = sorted(
    [path.name.replace("_neural_data.npy", "") for path in SPK_ROOT.glob("*_neural_data.npy")],
    key=natural_key,
)
```

iii. Using spike file names guarantees one entry per physical recording. The behavior merging handles duplicate appearances across experiment types.

## 1-d. How are the data split into trials?

i. Trials are iterated from 0 to `ntrials`. For each trial, frames are selected where `ft_trInd == trial_idx` AND `ft_move > 0` AND `ft_CorrSpc == True`. Trials with no qualifying frames or whose maximum position does not reach 35 dm are dropped. The remaining frames are then interpolated onto 4 spatial position bins (at 5, 15, 25, 35 dm).

ii.
```python
def trial_frame_indices(beh, n_frames, trial_idx):
    ft_trial_idx = ...
    move = np.asarray(beh["ft_move"][:n_frames]) > 0
    in_corridor = np.asarray(beh["ft_CorrSpc"][:n_frames]).astype(bool)
    return np.where((ft_trial_idx == trial_idx) & move & in_corridor)[0]

# Additional filtering:
if np.nanmax(trial_positions) < POSITION_BIN_CENTERS[-1]:
    continue
```

iii. The CONVERSION_NOTES.md states: "Only frames satisfying `ft_move > 0` and `ft_CorrSpc == True` are used. This follows the paper's statement that analyses only considered timepoints during running."

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped if (1) they have no frames after the movement and corridor-space filter, or (2) the maximum position of qualifying frames does not reach 35 dm (the center of the last spatial bin). Sessions with fewer than 2 usable trials raise an error.

ii.
```python
if len(frame_idx) == 0:
    trial_stats["empty_running_trials"] += 1
    continue
if np.nanmax(trial_positions) < POSITION_BIN_CENTERS[-1]:
    trial_stats["empty_running_trials"] += 1
    continue
if len(session_neural) < 2:
    raise ValueError(...)
```

iii. The AI's CONVERSION_NOTES.md states: "Trials without usable running corridor frames are dropped."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_key>_neural_data.npy` (deconvolved calcium traces, one array per imaging plane) and `iarea` in `retinotopy/<mouse>_<date>_trans.npz` for brain region assignment.

ii.
```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
spk = np.concatenate([plane.astype(np.float32, copy=False) for plane in spk_planes], axis=0)
area_codes = ret["iarea"]
```

iii. Same source variables as the reference.

## 2-b. How is the `neural` data processed?

i. The AI applies two major processing steps not in the reference: (1) Neurons are capped at 512 per session using variance-based ranking with per-region balancing. (2) Neural activity is spatially interpolated onto 4 position bin centers (5, 15, 25, 35 dm) rather than kept at the native temporal resolution. The result is stored as float16.

ii.
```python
# Neuron capping:
selected_neurons, region_idx = pick_neurons_by_variance(
    spk=spk, area_codes=area_codes,
    running_corridor_mask=running_corridor_mask,
    max_neurons=max_neurons,
)
# Spatial interpolation:
neural_interp = interpolate_features(
    positions=trial_positions,
    values=spk_selected[:, frame_idx],
    target_positions=POSITION_BIN_CENTERS,
).astype(np.float16)
```

iii. The CONVERSION_NOTES.md states: "A deterministic neuron cap is applied per session after visual-cortex filtering so the expanded trial structure remains trainable with the provided decoder harness."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are first filtered to keep only those in visual cortex regions (V1, mHV, lHV, aHV) using the same area codes as the reference. Then, a second filter caps neurons at 512 per session by selecting the highest-variance neurons during running corridor frames, with per-region quota balancing.

ii.
```python
def pick_neurons_by_variance(spk, area_codes, running_corridor_mask, max_neurons):
    # First: keep only visual cortex
    for neuron_idx, area_code in enumerate(area_codes):
        region_name = area_code_to_region_name(area_code)
        if region_name is not None:
            region_to_indices[region_name].append(neuron_idx)
    # Then: cap by variance
    if len(valid_indices) <= max_neurons:
        return valid_indices, region_idx
    spk_valid = spk[valid_indices][:, running_corridor_mask]
    variances = np.var(spk_valid, axis=1)
    per_region_quota = max_neurons // len(BRAIN_REGIONS)
    ...
```

iii. The AI justified the neuron cap as needed for the decoder harness to be trainable. The reference keeps all visual cortex neurons (up to ~89,577 per session).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry. However, instead of aligning temporally from corridor entry, the AI aligns spatially by interpolating neural activity onto 4 spatial position bins (0-1m, 1-2m, 2-3m, 3-4m). This produces trials with exactly 4 "time points" that correspond to spatial positions rather than temporal frames.

ii.
```python
neural_interp = interpolate_features(
    positions=trial_positions,
    values=spk_selected[:, frame_idx],
    target_positions=POSITION_BIN_CENTERS,  # [5, 15, 25, 35] dm
).astype(np.float16)
```

iii. The CONVERSION_NOTES.md states: "The decoder task explicitly requires 4 equal 1 m position bins, so each trial is represented by 4 ordered samples corresponding to corridor positions 0-1 m, 1-2 m, 2-3 m, and 3-4 m."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI does not use temporal bins at all. Instead, it uses 4 spatial bins. The `time_bin_size` in metadata is set to `(1.0 / 0.60) * 1000.0 = 1666.67 ms`, derived from the nominal VR running speed of 60 cm/s over 1 m bins. No temporal rebinning is applied because the data is resampled spatially rather than temporally.

ii.
```python
"time_bin_size": float((1.0 / 0.60) * 1000.0),  # ~1666.67 ms
POSITION_BIN_CENTERS = np.array([5.0, 15.0, 25.0, 35.0], dtype=np.float32)
```

iii. The AI chose to interpret the "4 equal-length, 1-m-long spatial bins" instruction as defining the temporal axis, converting each spatial bin into a time duration based on nominal running speed.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime` (the MATLAB datenum of the sound cue) and `ft` (the frame timestamps), both in MATLAB datenum format.

ii.
```python
sound_time = float(np.asarray(beh["SoundTime"])[trial_idx])
time_to_sound = ((sound_time - ft[frame_idx]) * SECONDS_PER_DAY).astype(np.float32)
```

iii. The AI uses `SoundTime` directly rather than `SoundFr` (frame number) which the reference uses.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The time to sound cue is computed as `(SoundTime - ft[frame_idx]) * SECONDS_PER_DAY`, giving positive values before the cue and negative after. It is then spatially interpolated onto the 4 position bin centers.

ii.
```python
time_to_sound = ((sound_time - ft[frame_idx]) * SECONDS_PER_DAY).astype(np.float32)
input_trial = np.vstack([
    interpolate_features(trial_positions, time_to_sound, POSITION_BIN_CENTERS)[0],
    ...
])
```

iii. The sign convention (positive before cue) matches the reference's `cue[trial] - time`.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is interpolated onto the same 4 spatial position bins as the neural data, using the `interpolate_features` function.

ii.
```python
interpolate_features(trial_positions, time_to_sound, POSITION_BIN_CENTERS)[0]
```

iii. All variables share the same spatial interpolation scheme.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the date encoded in the session key, parsed into a `datetime.date` object.

ii.
```python
def compute_subject_day_index(session_keys):
    parsed = {key: parse_session_key(key) for key in session_keys}
    first_day = {}
    for key, info in parsed.items():
        if subj not in first_day or info["date"] < first_day[subj]:
            first_day[subj] = info["date"]
    day_index = {}
    for key, info in parsed.items():
        day_index[key] = float((info["date"] - first_day[info["subject"]]).days)
    return day_index, parsed
```

iii. The AI uses actual calendar day differences from the first recording date for each subject.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each subject, the earliest recording date is found. The day of training for each session is the number of calendar days between that session's date and the subject's first date. This is broadcast across all 4 spatial bins.

ii.
```python
day_index[key] = float((info["date"] - first_day[info["subject"]]).days)
# Per trial:
np.full(4, subject_day_value, dtype=np.float32)
```

iii. This differs from the reference, which counts session index (0, 1, 2, ...) regardless of gaps between recording dates. If a mouse was recorded on day 1, day 5, and day 10, the AI would give values 0, 4, 9 while the reference gives 0, 1, 2.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time` (MATLAB datenum of trial start) and `ft` (frame timestamps), both in MATLAB datenum format.

ii.
```python
trial_start_time = float(np.asarray(beh["Trial_start_time"])[trial_idx])
time_since_start = ((ft[frame_idx] - trial_start_time) * SECONDS_PER_DAY).astype(np.float32)
```

iii. The AI uses `Trial_start_time` directly rather than `StartFr` (frame number) which the reference uses.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The time since trial start is computed as `(ft[frame_idx] - Trial_start_time) * SECONDS_PER_DAY`, giving seconds since trial start. It is then spatially interpolated onto the 4 position bin centers.

ii.
```python
time_since_start = ((ft[frame_idx] - trial_start_time) * SECONDS_PER_DAY).astype(np.float32)
interpolate_features(trial_positions, time_since_start, POSITION_BIN_CENTERS)[0]
```

iii. The sign convention (positive after start) matches the reference's `time - start[trial]`.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is interpolated onto the same 4 spatial position bins as the neural data.

ii.
```python
interpolate_features(trial_positions, time_since_start, POSITION_BIN_CENTERS)[0]
```

iii. Same spatial interpolation as all other variables.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks rewarded corridor trials.

ii.
```python
np.full(4, float(np.asarray(beh["isRew"])[trial_idx]), dtype=np.float32)
```

iii. Same source as the reference.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The value is converted to float and broadcast across all 4 spatial bins. No additional processing.

ii.
```python
np.full(4, float(np.asarray(beh["isRew"])[trial_idx]), dtype=np.float32)
```

iii. Straightforward conversion, same as reference.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, which names the wall texture of each trial's corridor.

ii.
```python
stim_name = str(np.asarray(beh["WallName"])[trial_idx])
output_trial_partial = {
    "visual_stimulus_category": np.full(4, stimulus_to_id[stim_name], dtype=np.int16),
    ...
}
```

iii. Same source variable as the reference.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI uses each individual `WallName` (e.g., circle1, circle2, leaf1, leaf2, etc.) as a separate category, resulting in 15 distinct categories. The reference groups them into 4 broad texture types (circle, leaf, rock, wood). The category index is broadcast across all 4 spatial bins.

ii.
```python
def stimulus_catalog(behavior_by_session):
    names = set()
    for beh in behavior_by_session.values():
        names.update(map(str, np.asarray(beh["WallName"]).tolist()))
    return sorted(names, key=natural_key)

stimulus_to_id = {name: idx for idx, name in enumerate(stimulus_values)}
# output_values:
"output_values": [stimulus_values, ...]  # 15 individual names
```

iii. The instruction says "Visual stimulus category. e.g. circle1, leaf2, etc." which the AI interpreted as each individual texture being its own category. The reference groups them into 4 broad categories following the paper's analysis.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickTrind` (trial index of each lick) and `LickPos` (position in corridor where each lick occurred).

ii.
```python
lick_trials = np.rint(np.asarray(beh["LickTrind"])).astype(np.int64)
lick_positions = np.asarray(beh["LickPos"], dtype=np.float32)
lick_pos_trial = lick_positions[lick_trials == trial_idx]
```

iii. The AI uses position-based lick data to match its spatial binning scheme, whereas the reference uses `LickFr` (frame-based).

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, lick positions are filtered by trial index. For each of the 4 spatial bins, licking is 1 if any lick position falls within that bin's edges, 0 otherwise. Licks at or beyond the last bin edge (40 dm) are assigned to the last bin.

ii.
```python
lick_trial = np.array(
    [int(np.any((lick_pos_trial >= POSITION_BIN_EDGES[bin_idx]) &
                (lick_pos_trial < POSITION_BIN_EDGES[bin_idx + 1])))
     for bin_idx in range(4)],
    dtype=np.int16,
)
if len(lick_pos_trial) and np.any(lick_pos_trial >= POSITION_BIN_EDGES[-1]):
    lick_trial[-1] = 1
```

iii. This is a spatial binning of licking, different from the reference's temporal per-frame binary flag.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is binned into the same 4 spatial bins as the neural data, using lick position rather than lick frame.

ii.
```python
lick_trial = np.array(
    [int(np.any((lick_pos_trial >= POSITION_BIN_EDGES[bin_idx]) &
                (lick_pos_trial < POSITION_BIN_EDGES[bin_idx + 1])))
     for bin_idx in range(4)],
)
```

iii. All variables share the spatial binning alignment.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is not derived from raw data in the traditional sense. Since the AI uses 4 spatial position bins as the data's axis, the position output is trivially the bin index (0, 1, 2, 3).

ii.
```python
"position_bin": np.arange(4, dtype=np.int16),
```

iii. In the AI's spatial binning scheme, position is the axis itself, not a derived variable.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. No processing is needed. The position output is simply `np.arange(4)` for every trial, since each "time point" corresponds to a spatial bin.

ii.
```python
"position_bin": np.arange(4, dtype=np.int16),
```

iii. This makes position a trivially decodable variable (always [0, 1, 2, 3]) with no variance across trials.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. No thresholding is applied. The spatial bins (0-1m, 1-2m, 2-3m, 3-4m) are inherently the position categories, represented as indices 0-3.

ii.
```python
"position_bin": np.arange(4, dtype=np.int16),
"output_values": [..., ["0_to_1m", "1_to_2m", "2_to_3m", "3_to_4m"], ...]
```

iii. The reference discretizes `ft_Pos // 10` and clips to [0, 3], producing time-varying position within each temporal frame.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is inherently aligned since the spatial bins define the axis of all data streams.

ii.
```python
"position_bin": np.arange(4, dtype=np.int16),
```

iii. No alignment step is needed in the spatial binning scheme.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:n_frames], dtype=np.float32)
speed_interp = interpolate_features(
    positions=trial_positions,
    values=ft_speed[frame_idx],
    target_positions=POSITION_BIN_CENTERS,
)[0].astype(np.float32)
```

iii. Same source variable as the reference.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is first spatially interpolated onto the 4 position bin centers per trial. Then, continuous speed values from all sessions are collected and global quantiles (25th, 50th, 75th percentiles) are computed. These quantiles define bin edges for discretization using `np.digitize`.

ii.
```python
# Global quantile computation:
speed_values = np.concatenate(all_speed_values).astype(np.float32)
q25, q50, q75 = np.quantile(speed_values, [0.25, 0.50, 0.75])
speed_edges = np.array([-np.inf, q25, q50, q75, np.inf])

# Per-trial discretization:
speed_bins = np.digitize(trial["running_speed_continuous"], speed_edges[1:-1], right=False)
```

iii. The reference uses per-session rank-based quartiles, which guarantee exactly 25% of frames in each bin per session. The AI uses global quantile edges, which don't guarantee equal bin counts per session.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Continuous speed values are discretized using `np.digitize` with global quantile edges (q25, q50, q75). This produces 4 bins (0-3).

ii.
```python
speed_bins = np.digitize(trial["running_speed_continuous"], speed_edges[1:-1], right=False).astype(np.int16)
```

iii. The reference uses per-session rank-based quartiles ensuring exact 25% splits per session.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is spatially interpolated onto the same 4 position bin centers as the neural data.

ii.
```python
speed_interp = interpolate_features(
    positions=trial_positions,
    values=ft_speed[frame_idx],
    target_positions=POSITION_BIN_CENTERS,
)[0]
```

iii. Same spatial interpolation as all other variables.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI cuts behavior streams to the number of imaged frames (`n_frames = spk.shape[1]`). Trials with no running corridor frames are dropped. Trials that don't reach the last spatial bin center (35 dm) are dropped. Sessions with fewer than 2 usable trials raise an error. The AI also validates that behavior views agree across files and merges them.

ii.
```python
n_frames = spk.shape[1]
ft = np.asarray(beh["ft"][:n_frames], dtype=np.float64)
ft_pos = np.asarray(beh["ft_Pos"][:n_frames], dtype=np.float32)
# Trial filtering:
if len(frame_idx) == 0: continue
if np.nanmax(trial_positions) < POSITION_BIN_CENTERS[-1]: continue
```

iii. The behavior view merging and validation is more elaborate than the reference, which simply uses the first occurrence of each session.

## 12-a. What are the most time-consuming steps of the code?

i. Reading the spike files from disk (89 files totaling hundreds of GB) and the spatial interpolation/variance computation for neuron selection.

ii.
```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
spk = np.concatenate([plane.astype(np.float32, copy=False) for plane in spk_planes], axis=0)
# Variance computation:
spk_valid = spk[valid_indices][:, running_corridor_mask]
variances = np.var(spk_valid, axis=1)
```

iii. The I/O cost dominates, same as the reference. The variance computation is additional overhead not present in the reference.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The `interpolate_features` function loops over rows for bincount and interp operations. The `pick_neurons_by_variance` function loops over neurons for area code checking. The `trial_frame_indices` function scans the full frame index per trial.

ii.
```python
# Row-by-row interpolation:
for row in range(collapsed.shape[0]):
    collapsed[row] = np.bincount(inverse, weights=values[row], ...) / counts
for row in range(collapsed.shape[0]):
    interp[row] = np.interp(target_positions, unique_pos, collapsed[row])
# Per-neuron area check:
for neuron_idx, area_code in enumerate(area_codes):
    region_name = area_code_to_region_name(area_code)
```

iii. The row-by-row interpolation in `interpolate_features` is called for every trial of every session (neural, speed, time_to_sound, time_since_start), making it a significant candidate for vectorization.

## 12-c. What processing does the code repeat multiple times?

i. The `interpolate_features` function is called multiple times per trial with the same position data but different values (neural, time_to_sound, time_since_start, speed). The position sorting and unique computation is repeated each time. Also, `area_code_to_region_name` is called per-neuron multiple times during selection.

ii.
```python
# Called 4 times per trial with same positions:
neural_interp = interpolate_features(positions=trial_positions, values=spk_selected[:, frame_idx], ...)
interpolate_features(trial_positions, time_to_sound, POSITION_BIN_CENTERS)
interpolate_features(trial_positions, time_since_start, POSITION_BIN_CENTERS)
speed_interp = interpolate_features(positions=trial_positions, values=ft_speed[frame_idx], ...)
```

iii. The position-related computations (argsort, unique, inverse) within `interpolate_features` are redundant across calls for the same trial.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The variance-based neuron selection computes variances for all visual cortex neurons across running corridor frames, but this information is not stored in the output. The behavior view merging and validation (checking field agreement across views) is elaborate but the merged result is equivalent to just picking one view. The `source_behavior_views` metadata is stored but unlikely to be used downstream.

ii.
```python
# Variance computation for neuron selection:
spk_valid = spk[valid_indices][:, running_corridor_mask]
variances = np.var(spk_valid, axis=1)
# Elaborate view merging:
merged["source_behavior_views"] = [...]
```

iii. The neuron selection itself is unnecessary processing since the reference keeps all visual cortex neurons.
