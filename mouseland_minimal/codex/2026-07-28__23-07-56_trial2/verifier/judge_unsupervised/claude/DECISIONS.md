# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads three categories of data files: (1) behavior files from `data/beh/Beh_*.npy`, each containing dictionaries keyed by session identifiers, (2) spike data from `data/spk/*_neural_data.npy`, each containing a `spks` list of imaging planes, and (3) retinotopy files from `data/retinotopy/*_trans.npz` containing brain region labels (`iarea`). Session keys are extracted from the spike file names. Behavior views are loaded and matched to sessions by key, with `_swap1`/`_swap2` suffixes stripped to merge duplicate views of the same physical recording.

ii.
```python
def load_behavior_views():
    views = defaultdict(list)
    for beh_path in sorted(BEH_ROOT.glob("Beh_*.npy")):
        beh_dict = np.load(beh_path, allow_pickle=True).item()
        for view_key, view in beh_dict.items():
            base_key = strip_swap_suffix(view_key)
            views[base_key].append(...)
    return views

# Spike data loaded per session:
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
spk = np.concatenate([plane.astype(np.float32, copy=False) for plane in spk_planes], axis=0)
ret = np.load(ret_path, allow_pickle=True)
area_codes = ret["iarea"]
```

iii. The AI documented in CONVERSION_NOTES.md that 89 spike files correspond to 89 physical recordings (matching the paper's Methods section), while behavior exports contain 99 keys due to `swap1`/`swap2` duplicates. Swap views are merged and verified for consistency before use.

## 1-b. How are the data split into subjects (mice)?

i. Subject identity is parsed from the session key string (e.g., "DR10_2022_07_12_1" yields subject "DR10"). A sorted list of unique subjects is maintained, and each session is assigned a `subject_idx` mapping to this list.

ii.
```python
def parse_session_key(session_key: str):
    subject, year, month, day, block = session_key.split("_")
    ...
    return {"subject": subject, "date": date_obj, ...}

subjects = sorted({info["subject"] for info in parsed.values()})
subject_to_id = {subject: idx for idx, subject in enumerate(subjects)}
subject_idx.append(subject_to_id[parsed[session_key]["subject"]])
```

iii. The AI identified 19 unique subjects across 89 sessions, consistent with the paper's description.

## 1-c. How are the data split into sessions?

i. Each `*_neural_data.npy` file in the spike directory defines one session. The AI uses the file names (sorted with natural key ordering) as the canonical session list. Behavior data is matched to each session key.

ii.
```python
spk_session_keys = sorted(
    [path.name.replace("_neural_data.npy", "") for path in SPK_ROOT.glob("*_neural_data.npy")],
    key=natural_key,
)
```

iii. The AI found 89 session files, matching the paper's "89 recordings" count.

## 1-d. How are the data split into trials?

i. Within each session, trial identity is determined by the `ft_trInd` field in the behavior data, which provides the trial index for each neural imaging frame. The AI iterates from trial 0 to `ntrials-1`, selecting frames where `ft_trInd == trial_idx`.

ii.
```python
def trial_frame_indices(beh, n_frames, trial_idx):
    ft_trial_idx_raw = np.asarray(beh["ft_trInd"][:n_frames], dtype=np.float64)
    valid = ~np.isnan(ft_trial_idx_raw)
    ft_trial_idx = np.full(n_frames, -1, dtype=np.int64)
    ft_trial_idx[valid] = np.rint(ft_trial_idx_raw[valid]).astype(np.int64)
    move = np.asarray(beh["ft_move"][:n_frames]) > 0
    in_corridor = np.asarray(beh["ft_CorrSpc"][:n_frames]).astype(bool)
    return np.where((ft_trial_idx == trial_idx) & move & in_corridor)[0]

for trial_idx in range(int(beh["ntrials"])):
    frame_idx = trial_frame_indices(beh, n_frames, trial_idx)
```

iii. The AI's approach follows the reference code's use of `ft_trInd` for frame-to-trial assignment, combined with running and corridor masks.

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped if (1) they have zero valid running-in-corridor frames, or (2) the maximum position within the trial doesn't reach the last spatial bin center (35 dm). Sessions with fewer than 2 usable trials raise an error.

ii.
```python
frame_idx = trial_frame_indices(beh, n_frames, trial_idx)
if len(frame_idx) == 0:
    trial_stats["empty_running_trials"] += 1
    continue

trial_positions = ft_pos[frame_idx]
if np.nanmax(trial_positions) < POSITION_BIN_CENTERS[-1]:  # 35.0
    trial_stats["empty_running_trials"] += 1
    continue
```

iii. The AI's conversion notes state "Trials without usable running corridor frames are dropped." The output shows 0 empty running trials were actually dropped across all 89 sessions (raw and kept trial counts match).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `spks` field in `*_neural_data.npy` files, which contains Suite2p deconvolved fluorescence traces organized by imaging plane.

ii.
```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
spk = np.concatenate([plane.astype(np.float32, copy=False) for plane in spk_planes], axis=0)
```

iii. The AI's CONVERSION_NOTES.md states: "Neural activity comes from the deconvolved traces stored in `*_neural_data.npy`." This matches the reference code's `load_spk` function which also concatenates across planes.

## 2-b. How is the `neural` data processed?

i. After concatenating across planes, the AI: (1) filters to visual cortex neurons only using retinotopy area codes, (2) applies a variance-ranked cap of 512 neurons per session with per-region balancing, and (3) interpolates the selected neurons' activity onto 4 spatial position bin centers (5, 15, 25, 35 dm) using the running-in-corridor frame positions. The result is stored as float16.

ii.
```python
selected_neurons, region_idx = pick_neurons_by_variance(
    spk=spk, area_codes=area_codes,
    running_corridor_mask=running_corridor_mask, max_neurons=max_neurons,
)
spk_selected = spk[selected_neurons]

neural_interp = interpolate_features(
    positions=trial_positions,
    values=spk_selected[:, frame_idx],
    target_positions=POSITION_BIN_CENTERS,  # [5.0, 15.0, 25.0, 35.0]
).astype(np.float16)
```

iii. The AI documented this as: "Visual-cortex neurons only (V1, mHV, lHV, aHV), then deterministic variance-ranked cap at 512 neurons per session with per-region balancing." The 4-bin interpolation was justified as matching the decoder task's requirement for 4 spatial bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons in two stages: (1) only neurons mapped to known visual cortex regions (V1, mHV, lHV, aHV) via retinotopy area codes are retained; neurons with NaN or unrecognized area codes are excluded. (2) If more than 512 visual cortex neurons remain, they are ranked by variance during running-in-corridor frames, with per-region quotas (512/4 = 128 per region) filled by highest-variance neurons.

ii.
```python
def pick_neurons_by_variance(spk, area_codes, running_corridor_mask, max_neurons):
    region_to_indices = {region: [] for region in BRAIN_REGIONS}
    for neuron_idx, area_code in enumerate(area_codes):
        region_name = area_code_to_region_name(area_code)
        if region_name is not None:
            region_to_indices[region_name].append(neuron_idx)
    ...
    per_region_quota = max_neurons // len(BRAIN_REGIONS)
    ...
```

iii. The AI noted in CONVERSION_NOTES.md: "Neurons outside these visual-cortex groups are excluded, consistent with the reference analysis code paths that remove neurons outside visual cortex." The 512-neuron cap was described as a "format-driven deviation" to keep the decoder trainable.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The instructions specify alignment to "trial start (corridor entry)." The AI aligns by spatial position within the corridor rather than by time from corridor entry. Each trial's neural data is interpolated onto 4 position bin centers (5, 15, 25, 35 dm into the 40 dm corridor), using only running-in-corridor frames.

ii.
```python
POSITION_BIN_CENTERS = np.array([5.0, 15.0, 25.0, 35.0], dtype=np.float32)
neural_interp = interpolate_features(
    positions=trial_positions,
    values=spk_selected[:, frame_idx],
    target_positions=POSITION_BIN_CENTERS,
).astype(np.float16)
```

iii. The AI justified this by noting the reference code uses position interpolation (`spk_pos_interp`), and the decoder task requires 4 spatial bins. The metadata says `temporal_alignment_event: "corridor entry (trial start)"`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 4 spatial position bins (not temporal bins). Each trial has exactly 4 "timepoints" corresponding to 0-1m, 1-2m, 2-3m, and 3-4m corridor segments. The reported `time_bin_size` is `(1.0/0.60)*1000.0 ≈ 1666.7 ms`, calculated as the time to traverse 1 meter at 60 cm/s nominal speed. No temporal rebinning is applied; instead, position-based interpolation replaces time-based binning.

ii.
```python
"time_bin_size": float((1.0 / 0.60) * 1000.0),  # ~1666.7 ms
```

iii. The AI's conversion notes state: "4 spatial bins of 1 m each inside the 4 m corridor; moving frames only." The median frame period of the raw imaging data is ~0.315 s.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from two variables: `SoundTime` (per-trial sound event time in MATLAB serial date format) and `ft` (per-frame timestamps in the same format).

ii.
```python
sound_time = float(np.asarray(beh["SoundTime"])[trial_idx])
time_to_sound = ((sound_time - ft[frame_idx]) * SECONDS_PER_DAY).astype(np.float32)
```

iii. The AI computes the difference between sound time and frame time, then converts from MATLAB date fractions to seconds by multiplying by 86400.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, `time_to_sound = (SoundTime - ft) * 86400` gives seconds until the sound cue for each frame. This is then interpolated onto the 4 spatial position bin centers.

ii.
```python
time_to_sound = ((sound_time - ft[frame_idx]) * SECONDS_PER_DAY).astype(np.float32)
input_trial = np.vstack([
    interpolate_features(trial_positions, time_to_sound, POSITION_BIN_CENTERS)[0],
    ...
])
```

iii. Positive values mean the sound hasn't occurred yet; negative values mean it has passed. This gives a continuous time-varying signal.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Time to sound cue is interpolated onto the same 4 spatial position bin centers as the neural data, using the same running-in-corridor frame positions. This ensures temporal alignment through shared spatial registration.

ii.
```python
interpolate_features(trial_positions, time_to_sound, POSITION_BIN_CENTERS)[0]
```

iii. Same spatial interpolation scheme as neural data ensures alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from the session key's date component (e.g., "2022_07_12"). The date is parsed, and for each subject, the number of calendar days since that subject's first recorded session is computed.

ii.
```python
def compute_subject_day_index(session_keys: list[str]):
    parsed = {key: parse_session_key(key) for key in session_keys}
    first_day = {}
    for key, info in parsed.items():
        subj = info["subject"]
        if subj not in first_day or info["date"] < first_day[subj]:
            first_day[subj] = info["date"]
    day_index = {}
    for key, info in parsed.items():
        day_index[key] = float((info["date"] - first_day[info["subject"]]).days)
    return day_index, parsed
```

iii. The AI computes day-of-training as days since each subject's first session, yielding 0 for the first session. Values range from 0 to 92.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Day of training is a per-session scalar (same value for all trials and timepoints within a session). It is broadcast to all 4 spatial bins per trial as a constant vector.

ii.
```python
np.full(4, subject_day_value, dtype=np.float32),
```

iii. No further processing beyond the date subtraction. The value is constant within a session.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from `Trial_start_time` (per-trial start timestamp) and `ft` (per-frame timestamps), both in MATLAB serial date format.

ii.
```python
trial_start_time = float(np.asarray(beh["Trial_start_time"])[trial_idx])
time_since_start = ((ft[frame_idx] - trial_start_time) * SECONDS_PER_DAY).astype(np.float32)
```

iii. Computed as `(frame_time - trial_start_time) * 86400` to convert to seconds.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The difference between each frame's timestamp and the trial start timestamp is converted to seconds. This is then interpolated onto the 4 spatial position bin centers.

ii.
```python
time_since_start = ((ft[frame_idx] - trial_start_time) * SECONDS_PER_DAY).astype(np.float32)
interpolate_features(trial_positions, time_since_start, POSITION_BIN_CENTERS)[0]
```

iii. Values range from ~0.5 to 1764.1 seconds across the dataset.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same spatial interpolation as neural data - interpolated onto the 4 position bin centers using running-in-corridor frame positions.

ii.
```python
interpolate_features(trial_positions, time_since_start, POSITION_BIN_CENTERS)[0]
```

iii. Aligned through shared spatial registration with neural data.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from the `isRew` field in the behavior data, which is a per-trial binary indicator.

ii.
```python
np.full(4, float(np.asarray(beh["isRew"])[trial_idx]), dtype=np.float32)
```

iii. Directly uses the existing `isRew` field from the behavior structure.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The per-trial `isRew` value (0 or 1) is broadcast to all 4 spatial bins as a constant value. No additional processing.

ii.
```python
np.full(4, float(np.asarray(beh["isRew"])[trial_idx]), dtype=np.float32)
```

iii. Simple per-trial constant, broadcast to match the spatial bin structure.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from the `WallName` field in the behavior data, which contains the name of the visual texture displayed in each trial's corridor.

ii.
```python
stim_name = str(np.asarray(beh["WallName"])[trial_idx])
```

iii. The AI catalogs all unique wall names across all sessions.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. All unique `WallName` values across sessions are collected and sorted (natural sort order). Each name is assigned a numeric ID (0-14). The stimulus ID is then broadcast to all 4 spatial bins per trial. 15 categories total: circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5.

ii.
```python
def stimulus_catalog(behavior_by_session):
    names = set()
    for beh in behavior_by_session.values():
        names.update(map(str, np.asarray(beh["WallName"]).tolist()))
    return sorted(names, key=natural_key)

stimulus_to_id = {name: idx for idx, name in enumerate(stimulus_values)}
output_trial_partial = {
    "visual_stimulus_category": np.full(4, stimulus_to_id[stim_name], dtype=np.int16),
    ...
}
```

iii. The AI uses all unique wall names as categories. The reference code's `get_cat_id` function maps stimuli to 4 categories based on reward status, but the AI preserves the full stimulus identity. The instructions say "e.g. circle1, leaf2, etc." suggesting fine-grained categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `LickTrind` (trial index for each lick event) and `LickPos` (position in corridor where each lick occurred).

ii.
```python
lick_trials = np.rint(np.asarray(beh["LickTrind"])).astype(np.int64)
lick_positions = np.asarray(beh["LickPos"], dtype=np.float32)
```

iii. Uses position-stamped lick events rather than frame-level lick indicators.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, lick events are filtered by trial index. Then for each of the 4 position bins, the AI checks whether any lick position falls within that bin's boundaries ([0,10), [10,20), [20,30), [30,40] dm). The result is a binary vector of length 4.

ii.
```python
lick_pos_trial = lick_positions[lick_trials == trial_idx]
lick_trial = np.array([
    int(np.any((lick_pos_trial >= POSITION_BIN_EDGES[bin_idx]) &
               (lick_pos_trial < POSITION_BIN_EDGES[bin_idx + 1])))
    for bin_idx in range(4)
], dtype=np.int16)
if len(lick_pos_trial) and np.any(lick_pos_trial >= POSITION_BIN_EDGES[-1]):
    lick_trial[-1] = 1
```

iii. Licks at position >= 40 are included in the last bin. Binary: 1 if any lick in that spatial bin, 0 otherwise.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned by position: each spatial bin's lick indicator corresponds to the same spatial bin as the neural data. Since both neural data and licking are mapped to the same 4 position bins, they are inherently aligned.

ii. The bin edges `[0, 10, 20, 30, 40]` correspond to the position bin centers `[5, 15, 25, 35]` used for neural interpolation.

iii. Spatial alignment through shared position binning.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is not derived from any raw data variable in a meaningful sense. Since the AI uses 4 spatial position bins as its trial representation, the position output is deterministically `[0, 1, 2, 3]` for every trial.

ii.
```python
"position_bin": np.arange(4, dtype=np.int16),
```

iii. The position is the index of the spatial bin itself.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. No processing is involved. The output is simply `np.arange(4)` for every trial, representing the 4 spatial bins (0-1m, 1-2m, 2-3m, 3-4m).

ii.
```python
"position_bin": np.arange(4, dtype=np.int16),
```

iii. This is a deterministic consequence of the spatial binning design.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. No thresholding is needed because the 4 position bins are the fundamental unit of the trial representation. The 4 categories correspond to 0-1m, 1-2m, 2-3m, 3-4m.

ii.
```python
POSITION_BIN_EDGES = np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
# output_values: ["0_to_1m", "1_to_2m", "2_to_3m", "3_to_4m"]
```

iii. The spatial bin boundaries define the 4 equal-length 1-meter segments.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is trivially aligned because it IS the spatial bin index, and the neural data is also organized by the same spatial bins. Each "timepoint" in the trial IS a position bin.

ii.
```python
"position_bin": np.arange(4, dtype=np.int16),
```

iii. Perfect alignment by construction since both share the same spatial binning.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Derived from `ft_RunSpeed` in the behavior data, which provides running speed for each neural imaging frame.

ii.
```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:n_frames], dtype=np.float32)
```

iii. Uses the pre-computed frame-level running speed from the behavior export.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is interpolated from running-in-corridor frame positions onto the 4 spatial position bin centers, yielding one speed value per bin per trial.

ii.
```python
speed_interp = interpolate_features(
    positions=trial_positions,
    values=ft_speed[frame_idx],
    target_positions=POSITION_BIN_CENTERS,
)[0].astype(np.float32)
```

iii. Position-based interpolation averages speeds at similar positions within each trial.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Global quartiles (25th, 50th, 75th percentiles) are computed across all speed values from all sessions. These three edges define 4 bins. `np.digitize` assigns each speed value to a bin (0-3). The quartile edges are 16.592, 28.701, 43.168 cm/s.

ii.
```python
speed_values = np.concatenate(all_speed_values).astype(np.float32)
q25, q50, q75 = np.quantile(speed_values, [0.25, 0.50, 0.75]).astype(np.float32)
speed_edges = np.array([-np.inf, q25, q50, q75, np.inf], dtype=np.float32)

def finalize_outputs(session_output_partial, speed_edges):
    ...
    speed_bins = np.digitize(trial["running_speed_continuous"], speed_edges[1:-1], right=False).astype(np.int16)
```

iii. The AI uses global quartiles across all data, ensuring each bin contains approximately 25% of the data. Confirmed by output statistics: each speed quartile accounts for ~25% of data.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same 4 spatial position bin centers as the neural data, ensuring alignment through shared spatial registration.

ii.
```python
speed_interp = interpolate_features(
    positions=trial_positions,
    values=ft_speed[frame_idx],
    target_positions=POSITION_BIN_CENTERS,
)[0].astype(np.float32)
```

iii. Same `interpolate_features` function and target positions as neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) NaN values in `ft_trInd` are handled by marking those frames as invalid (trial index = -1). (2) Trials with zero valid running-in-corridor frames are skipped. (3) Trials that don't reach the last position bin center are skipped. (4) Multiple behavior views for the same session are merged and verified for consistency; mismatches raise errors. (5) `stim_id` fields with NaN values are counted during view selection but don't cause failures.

ii.
```python
ft_trial_idx_raw = np.asarray(beh["ft_trInd"][:n_frames], dtype=np.float64)
valid = ~np.isnan(ft_trial_idx_raw)
ft_trial_idx = np.full(n_frames, -1, dtype=np.int64)
ft_trial_idx[valid] = np.rint(ft_trial_idx_raw[valid]).astype(np.int64)
```

iii. The AI's approach is conservative: questionable data is excluded rather than imputed. In practice, 0 trials were dropped across all 89 sessions.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading each session's spike data (files range from tens to hundreds of MB, containing 20k-90k neurons). (2) The position-based interpolation (`interpolate_features`) called for each trial for neural data, speed, time-to-sound, and time-since-start. (3) The variance computation for neuron selection, which requires accessing all running-corridor frames.

ii.
```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
spk = np.concatenate([plane.astype(np.float32, copy=False) for plane in spk_planes], axis=0)
```

iii. The AI processes sessions sequentially to manage memory (spike data can total ~405 GB). Each session's spike data is explicitly freed after processing via `del spk; gc.collect()`.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized: (1) The trial-level loop processes each trial independently with `interpolate_features` calls that could be batched. (2) The `interpolate_features` function itself loops over rows for `np.bincount` and `np.interp` operations. (3) The lick binary computation uses a Python list comprehension over 4 bins. (4) The neuron region mapping iterates over each neuron individually.

ii.
```python
# Per-trial loop (line 304):
for trial_idx in range(int(beh["ntrials"])):
    ...

# Per-row interpolation loop (line 237):
for row in range(values.shape[0]):
    collapsed[row] = np.bincount(inverse, weights=values[row], minlength=len(unique_pos)) / counts

# Lick comprehension (line 342):
lick_trial = np.array([
    int(np.any((lick_pos_trial >= POSITION_BIN_EDGES[bin_idx]) & ...))
    for bin_idx in range(4)
], dtype=np.int16)
```

iii. The trial-level loop is the most impactful target. Since trials within a session share the same neuron set, batched interpolation could avoid redundant setup.

## 12-c. What processing does the code repeat multiple times?

i. (1) `interpolate_features` is called 4 times per trial (neural, time_to_sound, time_since_start, speed), each independently sorting positions and computing unique-position collapse. The position sorting/deduplication could be computed once per trial. (2) The `trial_frame_indices` function recomputes the frame-level trial mask for each trial, when a single pass building a trial-to-frames mapping would be more efficient. (3) Frame-level arrays (`ft_pos`, `ft_speed`, etc.) are sliced by frame indices repeatedly.

ii.
```python
# Called 4 times per trial with same positions:
interpolate_features(trial_positions, time_to_sound, POSITION_BIN_CENTERS)
interpolate_features(trial_positions, time_since_start, POSITION_BIN_CENTERS)
interpolate_features(trial_positions, ft_speed[frame_idx], POSITION_BIN_CENTERS)
interpolate_features(trial_positions, spk_selected[:, frame_idx], POSITION_BIN_CENTERS)
```

iii. The position sorting and unique-position computation in `interpolate_features` is identical for all 4 calls within a trial.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The `merge_behavior_views` function verifies consistency across many behavior fields (e.g., `ft_PosCum`, `LickTime`) that are never used in the conversion. (2) Frame periods are computed across all sessions to store a median value in metadata, but this is informational only. (3) The `running_corridor_mask` used for variance-based neuron selection is computed session-wide, but only a subset of frames (per-trial) is used for the actual data. (4) The full `spk` array is loaded and concatenated even though only a subset of neurons (up to 512) is ultimately used.

ii.
```python
# Computing frame periods for metadata only:
frame_periods = []
for key in spk_session_keys:
    ft = np.asarray(behavior_by_session[key]["ft"], dtype=np.float64)
    if len(ft) > 1:
        frame_periods.append(float(np.median(np.diff(ft) * SECONDS_PER_DAY)))
```

iii. The metadata fields and consistency checks add robustness but don't contribute to the converted data used by the decoder.
