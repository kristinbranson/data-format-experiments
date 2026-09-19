# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates session keys from the spike files in `data/spk/` (glob `*_neural_data.npy`), then loads all behavior files from `data/beh/Beh_*.npy` and merges multiple behavior "views" (including swap variants) for each session. Retinotopy files are loaded per-session from `data/retinotopy/`. The AI does not use `Imaging_Exp_info.npy` as a master index; instead it discovers sessions from the spike file names.

ii.
```python
spk_session_keys = sorted(
    [path.name.replace("_neural_data.npy", "") for path in SPK_ROOT.glob("*_neural_data.npy")],
    key=natural_key,
)
behavior_views = load_behavior_views()
behavior_by_session = {
    key: merge_behavior_views(key, behavior_views[key]) for key in spk_session_keys
}
```

iii. The agent reasoned: "I've confirmed the raw assets are in `data/spk`, `data/retinotopy`, and behavior arrays under `data/beh`." It surveyed the reference code's loading pattern and discovered that swap entries were duplicate views of the same physical session, choosing to merge them rather than use the experiment info index.

## 1-b. How are the data split into subjects?

i. Subjects are extracted by parsing session keys (`{subject}_{year}_{month}_{day}_{block}`). Unique subjects are sorted alphabetically and mapped to indices.

ii.
```python
def parse_session_key(session_key: str):
    subject, year, month, day, block = session_key.split("_")
    ...
subjects = sorted({info["subject"] for info in parsed.values()})
subject_to_id = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The agent parsed the session key format from the filenames directly. 19 unique subjects were identified.

## 1-c. How are the data split into sessions?

i. Each spike file corresponds to one session. The AI identifies 89 sessions from the spike file names. Swap behavior variants are merged into a single canonical behavior record per session, with consistency checks on shared fields.

ii.
```python
spk_session_keys = sorted(
    [path.name.replace("_neural_data.npy", "") for path in SPK_ROOT.glob("*_neural_data.npy")],
    key=natural_key,
)
```
```python
def merge_behavior_views(base_key, session_views):
    ...
    canonical = max(session_views, key=sort_key)
    ...
```

iii. The agent noted: "the behavior exports contain 99 keys, but the paper reports 89 recordings because ten swap1/swap2 entries are duplicated views of the same physical session."

## 1-d. How are the data split into trials?

i. For each session, the AI iterates over `range(beh['ntrials'])` and extracts frame indices where the trial index matches AND the mouse is moving (`ft_move > 0`) AND the mouse is in the corridor (`ft_CorrSpc`). Data from these frames is then spatially interpolated into 4 position bins of 1 m each.

ii.
```python
def trial_frame_indices(beh, n_frames, trial_idx):
    ...
    move = np.asarray(beh["ft_move"][:n_frames]) > 0
    in_corridor = np.asarray(beh["ft_CorrSpc"][:n_frames]).astype(bool)
    return np.where((ft_trial_idx == trial_idx) & move & in_corridor)[0]
```

iii. The agent observed: "using all frames gives pathological trial durations because mice sometimes pause for hundreds of frames, while ft_move > 0 inside the corridor yields the stable 20-35 frame traversals the paper analyzes."

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped if they have zero usable running-corridor frames, or if the mouse never reached the last spatial bin center (position 35 out of 40). Sessions with fewer than 2 usable trials raise an error. No percentile-based length filtering is applied.

ii.
```python
frame_idx = trial_frame_indices(beh, n_frames, trial_idx)
if len(frame_idx) == 0:
    trial_stats["empty_running_trials"] += 1
    continue

trial_positions = ft_pos[frame_idx]
if np.nanmax(trial_positions) < POSITION_BIN_CENTERS[-1]:
    trial_stats["empty_running_trials"] += 1
    continue
```

iii. The agent's filtering logic is driven by the spatial binning approach: if the mouse didn't reach the last bin center (35 dm), interpolation to that bin would be extrapolation, so the trial is dropped.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the spike files (list of arrays per imaging plane, concatenated), and `iarea` from the retinotopy files for brain region assignment.

ii.
```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
spk = np.concatenate([plane.astype(np.float32, copy=False) for plane in spk_planes], axis=0)
area_codes = np.load(ret_path, allow_pickle=True)["iarea"]
```

iii. The agent traced the reference code's `load_spk()` function to confirm the loading approach.

## 2-b. How is the `neural` data processed?

i. The neural data undergoes two major processing steps beyond the reference: (1) only moving frames (`ft_move > 0`) are used, and (2) the frame-level neural data is spatially interpolated onto 4 position bin centers (5, 15, 25, 35 dm) using linear interpolation. Duplicate positions are averaged before interpolation. The result is stored as float16.

ii.
```python
neural_interp = interpolate_features(
    positions=trial_positions,
    values=spk_selected[:, frame_idx],
    target_positions=POSITION_BIN_CENTERS,
).astype(np.float16)
```

iii. The agent reasoned: "The decoder task here requires exactly 4 spatial bins. The exported trials therefore use those 4 bins directly." And: "averaging within the four required 1 m bins."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) only neurons in the four visual areas (V1, mHV, lHV, aHV) are kept, matching the reference; (2) if more than 512 visual-cortex neurons remain, a variance-based selection caps at 512 neurons with per-region balancing.

ii.
```python
def pick_neurons_by_variance(spk, area_codes, running_corridor_mask, max_neurons):
    ...
    per_region_quota = max_neurons // len(BRAIN_REGIONS)
    for region_name in BRAIN_REGIONS:
        ...
        order = sorted(candidates.tolist(), key=lambda idx: (-variance_map[idx], idx))
        selected.extend(order[:min(per_region_quota, len(order))])
    ...
```

iii. The agent noted: "a literal trial-by-trial export of all 20k-90k neurons per recording would be on the order of 100 GB...which is not practical for the decoder harness."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The instructions specify alignment to corridor entry (trial start). The AI's data is spatially binned into 4 position bins rather than temporally aligned. The alignment is implicitly to corridor entry since position 0 corresponds to corridor entry, but the data is in spatial rather than temporal coordinates.

ii.
```python
neural_interp = interpolate_features(
    positions=trial_positions,
    values=spk_selected[:, frame_idx],
    target_positions=POSITION_BIN_CENTERS,
)
```

iii. The agent described the alignment as "corridor entry (trial start)" in the metadata, and `off_start: 0.0`, but the actual representation is spatial bins, not temporal bins from corridor entry.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI reports a `time_bin_size` of ~1667 ms, computed as `(1.0 / 0.60) * 1000.0` -- interpreting each 1 m spatial bin as taking 1/0.6 seconds at a nominal 60 cm/s VR speed. This is not a true temporal bin size; it's a spatial-to-temporal conversion. Each trial has exactly 4 "time" bins (the 4 spatial bins).

ii.
```python
"time_bin_size": float((1.0 / 0.60) * 1000.0),
"off_end": float(4.0 / 0.60),
"binning_scheme": "4 spatial bins of 1 m each inside the 4 m corridor; moving frames only",
```

iii. The agent chose spatial binning, converting the temporal bin size to a nominal time equivalent using the VR speed.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime` (the timestamp of the sound cue for each trial) and `ft` (the timestamp of each imaging frame).

ii.
```python
sound_time = float(np.asarray(beh["SoundTime"])[trial_idx])
time_to_sound = ((sound_time - ft[frame_idx]) * SECONDS_PER_DAY).astype(np.float32)
```

iii. The agent uses `SoundTime` (a timestamp) rather than `SoundFr` (a frame number). Both encode the same event but in different units.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The time-to-sound is computed per frame as `(SoundTime - ft[frame]) * SECONDS_PER_DAY`, then spatially interpolated onto the 4 bin centers. The result is positive before the sound and negative after, consistent with "time TO sound."

ii.
```python
time_to_sound = ((sound_time - ft[frame_idx]) * SECONDS_PER_DAY).astype(np.float32)
input_trial = np.vstack([
    interpolate_features(trial_positions, time_to_sound, POSITION_BIN_CENTERS)[0],
    ...
])
```

iii. The agent computes a time difference converted from day-fractions to seconds.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The same moving-corridor frames are used to compute both neural and input data, which are then spatially interpolated onto the same 4 bin centers.

ii.
```python
frame_idx = trial_frame_indices(beh, n_frames, trial_idx)
# Same frame_idx used for both neural and input interpolation
```

iii. Alignment is via the shared spatial interpolation grid.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the date component of the session key (parsed as `{subject}_{year}_{month}_{day}_{block}`), and the earliest recording date per subject.

ii.
```python
def compute_subject_day_index(session_keys):
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

iii. The agent parses dates from session keys to compute calendar days since first recording.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The day of training is the number of calendar days between the session date and the subject's first recording date. This differs from the reference, which counts the ordinal session index (0, 1, 2...). Calendar days can skip values if the mouse was not recorded on consecutive days. The value is broadcast across all 4 spatial bins.

ii.
```python
day_index[key] = float((info["date"] - first_day[info["subject"]]).days)
...
np.full(4, subject_day_value, dtype=np.float32)
```

iii. The agent chose calendar days as a natural measure of training progression.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time` (the timestamp of trial start) and `ft` (frame timestamps).

ii.
```python
trial_start_time = float(np.asarray(beh["Trial_start_time"])[trial_idx])
time_since_start = ((ft[frame_idx] - trial_start_time) * SECONDS_PER_DAY).astype(np.float32)
```

iii. The agent uses `Trial_start_time` rather than `StartFr`. Both encode the same event.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Time since trial start is computed per frame as `(ft[frame] - Trial_start_time) * SECONDS_PER_DAY`, then spatially interpolated to 4 bin centers. The result is positive (time elapsed since start).

ii.
```python
time_since_start = ((ft[frame_idx] - trial_start_time) * SECONDS_PER_DAY).astype(np.float32)
...
interpolate_features(trial_positions, time_since_start, POSITION_BIN_CENTERS)[0]
```

iii. Simple time difference converted to seconds, then spatially interpolated.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. The same moving-corridor frames and spatial interpolation grid are used for both neural and input data.

ii. Same as 3-c.

iii. Alignment is via shared spatial interpolation.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, the per-trial reward availability flag.

ii.
```python
np.full(4, float(np.asarray(beh["isRew"])[trial_idx]), dtype=np.float32)
```

iii. Directly read from the behavior data.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The binary flag is broadcast across all 4 spatial bins as a constant per-trial value. No additional processing.

ii.
```python
np.full(4, float(np.asarray(beh["isRew"])[trial_idx]), dtype=np.float32)
```

iii. No processing needed.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, the texture name for each trial.

ii.
```python
stim_name = str(np.asarray(beh["WallName"])[trial_idx])
output_trial_partial = {
    "visual_stimulus_category": np.full(4, stimulus_to_id[stim_name], dtype=np.int16),
    ...
}
```

iii. The agent reads the wall name from the behavior data.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI uses the 15 individual texture names (e.g., circle1, circle2, leaf1, leaf1_swap1, etc.) as separate categories, rather than grouping them into 4 base textures (circle, leaf, rock, wood) as the reference does. The global stimulus catalog is built from all sessions and sorted. The value is broadcast across 4 spatial bins.

ii.
```python
def stimulus_catalog(behavior_by_session):
    names = set()
    for beh in behavior_by_session.values():
        names.update(map(str, np.asarray(beh["WallName"]).tolist()))
    return sorted(names, key=natural_key)
```
Output values:
```python
"output_values": [
    stimulus_values,  # 15 individual texture names
    ...
]
```

iii. The agent did not implement the grouping from texture variants to base categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickTrind` (trial index of each lick) and `LickPos` (position in corridor of each lick).

ii.
```python
lick_trials = np.rint(np.asarray(beh["LickTrind"])).astype(np.int64)
lick_positions = np.asarray(beh["LickPos"], dtype=np.float32)
```

iii. The agent uses spatial position of licks rather than frame-based lick times.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, lick positions are checked against the 4 spatial bin edges `[0, 10, 20, 30, 40]`. If any lick falls within a bin's range, that bin gets value 1, else 0. Licks beyond the final edge are assigned to the last bin.

ii.
```python
lick_pos_trial = lick_positions[lick_trials == trial_idx]
lick_trial = np.array(
    [int(np.any((lick_pos_trial >= POSITION_BIN_EDGES[bin_idx]) &
                 (lick_pos_trial < POSITION_BIN_EDGES[bin_idx + 1])))
     for bin_idx in range(4)],
    dtype=np.int16,
)
if len(lick_pos_trial) and np.any(lick_pos_trial >= POSITION_BIN_EDGES[-1]):
    lick_trial[-1] = 1
```

iii. The agent bins licks spatially rather than temporally, consistent with the spatial binning approach.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Both licking and neural data are represented in the same 4 spatial bins, so they are aligned by spatial position.

ii. See 8-b above -- both use the same position bin grid.

iii. Alignment is via shared spatial bins.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. The position output is trivially `np.arange(4)` -- the bin index itself. Since data is spatially binned, the position is inherent in the bin structure.

ii.
```python
"position_bin": np.arange(4, dtype=np.int16),
```

iii. In the spatial binning scheme, position is definitionally the bin index.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. No processing: the value is simply the bin index 0, 1, 2, 3 for each of the 4 spatial bins. This is a trivially deterministic output -- always [0, 1, 2, 3] for every trial.

ii.
```python
"position_bin": np.arange(4, dtype=np.int16),
```

iii. Position is the identity of the spatial bin.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. No thresholding is needed since the spatial bins directly correspond to position categories. Each bin is one of the four 1 m segments.

ii.
```python
"position_bin": np.arange(4, dtype=np.int16),
```

iii. The spatial binning inherently discretizes position.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Both share the same 4-bin spatial grid, so alignment is inherent.

ii. Same spatial bin structure for all variables.

iii. Alignment is by construction.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
ft_speed = np.asarray(beh["ft_RunSpeed"][:n_frames], dtype=np.float32)
```

iii. Directly from the behavior data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is first spatially interpolated from frame positions to the 4 bin centers. Then, across ALL sessions globally, the 25th, 50th, and 75th percentiles of speed values are computed. Speed is discretized into 4 bins using `np.digitize` with these global percentile edges.

ii.
```python
speed_interp = interpolate_features(
    positions=trial_positions,
    values=ft_speed[frame_idx],
    target_positions=POSITION_BIN_CENTERS,
)[0].astype(np.float32)
...
speed_values = np.concatenate(all_speed_values).astype(np.float32)
q25, q50, q75 = np.quantile(speed_values, [0.25, 0.50, 0.75])
speed_edges = np.array([-np.inf, q25, q50, q75, np.inf])
...
speed_bins = np.digitize(trial["running_speed_continuous"], speed_edges[1:-1], right=False)
```

iii. The agent chose global percentile thresholds rather than per-session rank-based quartiles.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Global 25th, 50th, and 75th percentile thresholds are used to bin speed into 4 categories. This uses value-based thresholds rather than rank-based quartiles.

ii. See 10-b.

iii. The agent computed global percentile edges across all sessions.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed values are spatially interpolated onto the same 4 bin centers as neural data.

ii.
```python
speed_interp = interpolate_features(
    positions=trial_positions,
    values=ft_speed[frame_idx],
    target_positions=POSITION_BIN_CENTERS,
)[0]
```

iii. Alignment via shared spatial interpolation grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several missing data cases are handled: (1) `ft_trInd` can contain NaN outside behavior coverage -- these are set to -1 before trial matching; (2) `stim_id` has NaN in swap sessions -- the canonical view with fewest NaNs is selected; (3) trials with no running frames are dropped; (4) trials where the mouse doesn't reach the last bin center are dropped; (5) sessions with fewer than 2 usable trials raise an error.

ii.
```python
ft_trial_idx_raw = np.asarray(beh["ft_trInd"][:n_frames], dtype=np.float64)
valid = ~np.isnan(ft_trial_idx_raw)
ft_trial_idx = np.full(n_frames, -1, dtype=np.int64)
ft_trial_idx[valid] = np.rint(ft_trial_idx_raw[valid]).astype(np.int64)
```

iii. The agent noted: "ft_trInd has NaN outside behavior coverage, so I'm making that conversion explicit."

## 12-a. What are the most time-consuming steps of the code?

i. Reading the spike files (405 GB total). The agent noted individual files like `TX108_2023_03_25_1_neural_data.npy` at ~7.9 GB caused slowdowns. Full conversion took ~30+ minutes.

ii.
```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
spk = np.concatenate([plane.astype(np.float32, copy=False) for plane in spk_planes], axis=0)
```

iii. The agent noted: "The slowdown is explained: TX108_2023_03_25_1_neural_data.npy is about 7.9 GB."

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The `interpolate_features` function loops over rows (neurons) for both the collapse and interpolation steps. The `pick_neurons_by_variance` function iterates over neurons to build region indices. The `area_code_to_region_name` function is called per-neuron in a loop.

ii.
```python
for row in range(values.shape[0]):
    collapsed[row] = np.bincount(inverse, weights=values[row], minlength=len(unique_pos)) / counts
for row in range(collapsed.shape[0]):
    interp[row] = np.interp(target_positions, unique_pos, collapsed[row])
```
```python
for neuron_idx, area_code in enumerate(area_codes):
    region_name = area_code_to_region_name(area_code)
```

iii. These loops are not the bottleneck compared to I/O.

## 12-c. What processing does the code repeat multiple times?

i. The behavior views are loaded and merged for all sessions upfront, but `area_code_to_region_name` is called repeatedly per-neuron both in `pick_neurons_by_variance` and when building `region_idx`. The `interpolate_features` function is called separately for neural data, speed, time_to_sound, and time_since_start for each trial.

ii.
```python
# Called multiple times per session
interpolate_features(positions=trial_positions, values=spk_selected[:, frame_idx], ...)
interpolate_features(positions=trial_positions, values=ft_speed[frame_idx], ...)
interpolate_features(trial_positions, time_to_sound, ...)
interpolate_features(trial_positions, time_since_start, ...)
```

iii. Each call to `interpolate_features` re-sorts and re-collapses the same positions.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The neuron variance computation across all running-corridor frames is performed for every session even when the session has fewer than 512 visual-cortex neurons (and no subsampling is needed). The `interpolate_features` function computes position-collapsed averages that could be simplified for 1D signals. The behavior view merge checks many fields that are always consistent. The sample dataset creation copies the full dataset.

ii.
```python
spk_valid = spk[valid_indices][:, running_corridor_mask]
variances = np.var(spk_valid, axis=1)
# Even when len(valid_indices) <= max_neurons, variance is computed before the check
```

iii. The variance computation on potentially 50K+ neurons across all corridor frames is expensive even when not needed for selection.
