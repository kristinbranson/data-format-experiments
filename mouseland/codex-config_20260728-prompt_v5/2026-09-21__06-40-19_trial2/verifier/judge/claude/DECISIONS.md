# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three directories: `beh/` (behavior), `spk/` (neural), and `retinotopy/` (visual area assignments). It first loads the master index `Imaging_Exp_info.npy`, then builds a canonical list of 89 unique sessions by collapsing duplicate behavior keys across experiment types. Behavior files are loaded via an LRU-cached function per experiment type. Neural data and retinotopy are loaded per session during conversion.

ii.
```python
def load_experiment_index() -> dict[str, list[dict[str, Any]]]:
    return np.load(BEH_ROOT / "Imaging_Exp_info.npy", allow_pickle=True).item()

@lru_cache(maxsize=2)
def load_behavior_file(exp_type: str) -> dict[str, dict[str, Any]]:
    return np.load(BEH_ROOT / f"Beh_{exp_type}.npy", allow_pickle=True).item()

spk_parts = np.load(session.spk_path, allow_pickle=True).item()["spks"]
retino = np.load(session.retino_path, allow_pickle=True)
```

iii. The AI documented in CONVERSION_NOTES.md that there are 142 experiment-index entries, 99 unique behavior-session keys, and 89 unique raw neural sessions. It canonicalizes to 89 sessions matching the paper's stated count.

## 1-b. How are the data split into subjects?

i. The subject (mouse) name is extracted from the `mname` field in the experiment index entries. Unique subjects are collected from all sessions and sorted alphabetically.

ii.
```python
subjects = sorted({session.subject for session in sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The AI identifies 19 unique mice matching the paper's count of "89 recordings in 19 mice."

## 1-c. How are the data split into sessions?

i. A session is defined by the canonical raw ID `<mname>_<datexp>_<blk>`. The AI collapses duplicate behavior keys (same recording listed under multiple experiment types or with different `stimtype` suffixes) into a single canonical session, yielding 89 sessions.

ii.
```python
def canonical_raw_id(entry: dict[str, Any]) -> str:
    return f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"

# In build_canonical_sessions:
rec = per_raw.setdefault(raw_id, {...})
rec["aliases"].add(beh_key)
rec["exp_types"].add(exp_type)
```

iii. The AI verified that duplicate behavior keys for the same raw session contain identical core arrays, so collapsing is safe. Sessions appearing under multiple experiment types are not duplicated.

## 1-d. How are the data split into trials?

i. Trials are split using `ft_trInd` (the trial index for each frame), `ft_CorrSpc` (corridor space mask), and `ft_move > 0` (running mask). For each trial, the retained frames are those where all three conditions are true. Trials with zero retained frames are dropped. No trial length filtering is applied.

ii.
```python
def trial_frame_groups(beh: dict[str, Any]) -> list[np.ndarray]:
    ft_tr = np.asarray(beh["ft_trInd"])
    ft_corr = np.asarray(beh["ft_CorrSpc"], dtype=bool)
    ft_move = np.asarray(beh["ft_move"]) > 0
    ...
    keep = finite & ft_corr & ft_move
    frame_idx = np.flatnonzero(keep)
    trial_ids = tr_int[keep]
    ...
```

iii. The AI justifies using `ft_move > 0` by citing the paper: "We only considered timepoints during running for analysis" and the reference code's use of `ft_CorrSpc & (ft_move > 0)` in `Get_dprime_selective_neuron`. However, the reference solution does NOT use `ft_move > 0`.

## 1-e. How are trials filtered based on quality controls?

i. Trials with zero retained corridor-running frames are dropped. Sessions with fewer than 2 remaining trials are skipped. No trial-length-based filtering (e.g., percentile cutoff) is applied.

ii.
```python
# In convert_session:
for trial, frames in enumerate(groups):
    if frames.size == 0:
        continue
    ...

# In build_dataset:
if len(neural_trials) < 2:
    print(f"[convert] skipping {session.raw_id}: only {len(neural_trials)} converted trial(s)")
    continue
```

iii. The AI does not implement any trial length outlier filtering. The reference solution drops trials longer than the 99th percentile of trial lengths across the entire dataset to remove extremely long trials where mice stopped.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy`, which contains a list of arrays (one per imaging plane). The visual area of each neuron comes from `iarea` in the retinotopy files.

ii.
```python
spk_parts = np.load(session.spk_path, allow_pickle=True).item()["spks"]
retino = np.load(session.retino_path, allow_pickle=True)
iarea = np.asarray(retino["iarea"], dtype=np.float32)
```

iii. The AI correctly identifies the deconvolved traces as the neural signal, matching the paper's statement: "All our analyses were based on deconvolved fluorescence traces."

## 2-b. How is the `neural` data processed?

i. The neural data from the three imaging planes is concatenated per trial (not pre-concatenated for the whole session). The result is stored as float32. Trials are variable length.

ii.
```python
neural_trial = np.concatenate([part[:, frames] for part in spk_parts], axis=0).astype(np.float32, copy=False)
```

iii. The AI avoids full-session concatenation for memory efficiency, slicing trial-by-trial from the raw plane arrays. This is functionally equivalent to concatenating then slicing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps ALL neurons, including those outside the four main visual areas. Neurons with `iarea == 7` are labeled "unassigned_7" and those with `iarea == -1` are labeled "outside_visual". This results in 6 brain regions and retains all 4,691,034 neurons.

ii.
```python
BRAIN_REGIONS = ["V1", "mHV", "lHV", "aHV", "unassigned_7", "outside_visual"]

def map_iarea_to_region_idx(iarea: np.ndarray) -> np.ndarray:
    out = np.empty(iarea.shape[0], dtype=np.int16)
    out.fill(-1)
    out[iarea == 8] = 0  # V1
    out[np.isin(iarea, [0, 1, 2, 9])] = 1  # mHV
    out[np.isin(iarea, [5, 6])] = 2  # lHV
    out[np.isin(iarea, [3, 4])] = 3  # aHV
    out[iarea == 7] = 4  # unassigned_7
    out[iarea == -1] = 5  # outside_visual
    ...
```

iii. The AI justified keeping all neurons to "preserve paper-level neuron counts and source information." However, the reference solution drops neurons outside V1, mHV, lHV, and aHV, keeping only 4,105,393 of 4,691,034 neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry (trial start). Each trial starts at its first retained corridor-running frame and ends at its last. Trials are variable length with no padding.

ii.
```python
for trial, frames in enumerate(groups):
    if frames.size == 0:
        continue
    neural_trial = np.concatenate([part[:, frames] for part in spk_parts], axis=0).astype(np.float32, copy=False)
    session_neural.append(neural_trial)
```

iii. The alignment to corridor entry is consistent with the instructions. However, the AI's running filter means the first frame may not actually be corridor entry if the mouse wasn't running at entry.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The native imaging frame rate of 3.17 Hz is preserved, giving a time bin size of ~315.46 ms.

ii.
```python
FRAME_RATE_HZ = 3.17
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
```

iii. The imaging frame is the finest temporal resolution available, matching the reference approach.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime` (the absolute timestamp of the sound cue for each trial) and `ft` (the absolute timestamp of each imaging frame).

ii.
```python
sound_time = np.asarray(beh["SoundTime"], dtype=np.float64)
ft = np.asarray(beh["ft"], dtype=np.float64)
time_to_cue = ((sound_time[trial] - frame_times) * 86400.0).astype(np.float32)
```

iii. The reference solution uses `SoundFr` (frame number of sound cue) interpolated onto the frame time axis instead. Both approaches should give equivalent results if timestamps are consistent, but `SoundFr` is more directly in frame coordinates.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The time to sound cue is computed as `(SoundTime[trial] - ft[frame]) * 86400.0` for each retained frame, giving the time difference in seconds. Positive values mean the cue hasn't happened yet; negative means it has.

ii.
```python
frame_times = ft[frames]
time_to_cue = ((sound_time[trial] - frame_times) * 86400.0).astype(np.float32)
```

iii. The sign convention (positive before cue, negative after) matches the reference solution's `cue[trial] - time`.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the timestamps of the same retained frames used for the neural data of that trial.

ii.
```python
frame_times = ft[frames]
time_to_cue = ((sound_time[trial] - frame_times) * 86400.0).astype(np.float32)
```

iii. The alignment is inherent because all data streams use the same frame indices.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the `dateexp` field of each session, parsed as a calendar date.

ii.
```python
def compute_day_offsets(sessions: list[SessionInfo]) -> dict[str, float]:
    subject_first_date = {}
    for session in sessions:
        subject_first_date[session.subject] = min(
            subject_first_date.get(session.subject, session.date_obj),
            session.date_obj,
        )
    return {
        session.raw_id: float((session.date_obj - subject_first_date[session.subject]).days)
        for session in sessions
    }
```

iii. The AI uses the `dateexp` field parsed into datetime objects to compute calendar day offsets.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The day of training is the number of calendar days since the subject's first recording date. For each subject, the earliest `dateexp` is found, and the offset in days is computed for each session. This is a per-trial constant broadcast across all time bins.

ii.
```python
return {
    session.raw_id: float((session.date_obj - subject_first_date[session.subject]).days)
    for session in sessions
}
# ...
day_value = np.float32(day_offsets[session.raw_id])
day_of_training = np.full(frames.size, day_value, dtype=np.float32)
```

iii. The reference solution instead counts session ordinals (0, 1, 2, ...) per mouse, incrementing by 1 for each session regardless of actual calendar gaps. The AI's calendar-day approach can produce larger values (up to 92 days) versus the reference's max of 7.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time` (the absolute timestamp of each trial start) and `ft` (the absolute timestamp of each imaging frame).

ii.
```python
trial_start_time = np.asarray(beh["Trial_start_time"], dtype=np.float64)
time_since_start = ((frame_times - trial_start_time[trial]) * 86400.0).astype(np.float32)
```

iii. The reference solution uses `StartFr` (the frame number at corridor entry) interpolated onto the frame time axis instead.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The time since trial start is computed as `(ft[frame] - Trial_start_time[trial]) * 86400.0` for each retained frame, in seconds.

ii.
```python
time_since_start = ((frame_times - trial_start_time[trial]) * 86400.0).astype(np.float32)
```

iii. The sign convention (positive after start) matches the reference.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed from the timestamps of the same retained frames used for the neural data.

ii.
```python
frame_times = ft[frames]
time_since_start = ((frame_times - trial_start_time[trial]) * 86400.0).astype(np.float32)
```

iii. Alignment is inherent because all streams use the same frame indices.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks whether each trial is in the rewarded corridor.

ii.
```python
is_rew = np.asarray(beh["isRew"], dtype=bool)
reward_availability = np.full(frames.size, float(is_rew[trial]), dtype=np.float32)
```

iii. This matches the reference solution.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew` is converted to a float (0.0 or 1.0) and broadcast across all time bins of the trial.

ii.
```python
reward_availability = np.full(frames.size, float(is_rew[trial]), dtype=np.float32)
```

iii. No additional processing needed. Matches the reference.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, which names the texture on the corridor walls for each trial.

ii.
```python
wall_name = np.asarray(beh["WallName"])
stimulus_idx = stim_to_idx[str(wall_name[trial])]
```

iii. Same source variable as the reference.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI preserves the raw wall names as individual categories. The global stimulus vocabulary is gathered across all sessions, sorted, and each name gets a unique integer index. This yields 15 categories (circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5).

ii.
```python
def gather_stimulus_vocabulary(sessions: list[SessionInfo]) -> list[str]:
    vocab = set()
    for session in sessions:
        beh = get_behavior(session)
        vocab.update(map(str, np.unique(np.asarray(beh["WallName"]))))
    return sorted(vocab)

stim_to_idx = {stim: idx for idx, stim in enumerate(stim_vocab)}
stimulus_idx = stim_to_idx[str(wall_name[trial])]
stimulus_out = np.full(frames.size, stimulus_idx, dtype=np.int16)
```

iii. The reference solution groups the 15 raw names into 4 broad categories (circle, leaf, rock, wood) using a lookup table. The AI's choice to use 15 categories means the decoder has a much harder classification task and doesn't match the instruction's example "e.g. circle, leaf, etc."

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the frame number of each lick in the session.

ii.
```python
lick_frames = np.asarray(beh["LickFr"])
```

iii. Same source variable as the reference.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A binary array is created with 1 at each frame where a lick occurred. The AI handles edge cases: checks for finite values, clips to valid frame range, and uses `np.unique` to avoid double-counting.

ii.
```python
def build_lick_binary(beh: dict[str, Any], nframes: int) -> np.ndarray:
    lick_binary = np.zeros(nframes, dtype=np.int8)
    lick_frames = np.asarray(beh["LickFr"])
    if lick_frames.size == 0:
        return lick_binary
    finite = np.isfinite(lick_frames)
    lick_idx = lick_frames[finite].astype(np.int64, copy=False)
    lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nframes)]
    if lick_idx.size:
        lick_binary[np.unique(lick_idx)] = 1
    return lick_binary
```

iii. The logic is functionally equivalent to the reference's approach. However, the AI uses `nframes=ft.shape[0]` (behavior length) rather than `spikes.shape[1]` (neural length), which could include frames beyond what was imaged.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick binary is indexed at the same retained frame indices used for neural data.

ii.
```python
lick_out = lick_binary[frames].astype(np.int16, copy=False)
```

iii. Alignment is inherent because all streams use the same frame indices.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position in the corridor at each imaging frame, in decimeters (0-40 for the texture corridor, up to 60 through gray space).

ii.
```python
ft_pos = np.asarray(beh["ft_Pos"], dtype=np.float32)
```

iii. Same source variable as the reference.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is clipped to [0, 39.999999], divided by 10 (converting decimeters to meters), floored, and clipped to [0, 3], yielding 4 bins of 1 m each.

ii.
```python
def position_to_bin(pos: np.ndarray) -> np.ndarray:
    bins = np.floor(np.clip(pos, 0.0, 39.999999) / 10.0).astype(np.int16)
    return np.clip(bins, 0, 3)
```

iii. Functionally equivalent to the reference's `np.clip(beh['ft_Pos'][:n_frames] // 10, 0, 3)`.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four equal 1-meter bins: 0-1 m, 1-2 m, 2-3 m, 3-4 m, determined by integer division by 10 decimeters.

ii.
```python
bins = np.floor(np.clip(pos, 0.0, 39.999999) / 10.0).astype(np.int16)
```

iii. Matches the reference approach and the instruction's requirement for "4 equal-length, 1-m-long spatial bins."

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is indexed at the same retained frame indices used for neural data.

ii.
```python
pos_out = position_to_bin(ft_pos[frames])
```

iii. Alignment is inherent because all streams use the same frame indices.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
ft_run_speed = np.asarray(beh["ft_RunSpeed"], dtype=np.float32)
```

iii. Same source variable as the reference.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes global quantile edges at 25%, 50%, 75% across ALL retained corridor-running frames from all sessions, then uses `np.digitize` to assign each frame to one of 4 bins.

ii.
```python
def stabilized_quantile_edges(values: np.ndarray) -> np.ndarray:
    edges = np.quantile(values, [0.25, 0.50, 0.75]).astype(np.float32)
    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf, dtype=np.float32)
    return edges

def speed_to_bin(speed: np.ndarray, speed_edges: np.ndarray) -> np.ndarray:
    return np.digitize(speed, speed_edges, right=False).astype(np.int16)
```

iii. The reference solution uses per-session rank-based quartiles instead. The reference computes quartiles within each session's kept frames. The AI's global approach means bins are consistent across sessions but may not reflect per-session distributions. Additionally, using `np.digitize` with value-based edges rather than rank-based assignment means bins may not each contain exactly 25% of data when there are many tied values (e.g., many zero-speed frames).

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four quartile-based bins using global value thresholds from `np.quantile`. The stabilization step ensures monotonically increasing edges.

ii.
```python
edges = np.quantile(values, [0.25, 0.50, 0.75]).astype(np.float32)
```

iii. The reference uses rank-based quartiles per session, which guarantees exactly 25% of frames in each bin regardless of tied values. The AI's value-based approach with global edges may produce unequal bin sizes when speeds have many ties.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is indexed at the same retained frame indices used for neural data.

ii.
```python
speed_out = speed_to_bin(ft_run_speed[frames], speed_edges)
```

iii. Alignment is inherent because all streams use the same frame indices.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: non-finite values in `ft_trInd` and `LickFr` are filtered out. Lick frames beyond valid range are clipped. Trials with zero retained frames are dropped. Sessions with fewer than 2 trials are skipped.

ii.
```python
# NaN handling in ft_trInd
finite = np.isfinite(ft_tr)
tr_int = np.full(ft_tr.shape, -1, dtype=np.int32)
tr_int[finite] = ft_tr[finite].astype(np.int32)

# Lick frame validation
finite = np.isfinite(lick_frames)
lick_idx = lick_frames[finite].astype(np.int64, copy=False)
lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nframes)]
```

iii. The AI does NOT clip behavior arrays to the neural frame count (spikes.shape[1]). The reference does this clipping: `n_frames = spikes.shape[1]` and then `beh['ft'][:n_frames]`. The AI uses `ft.shape[0]` for the lick binary, which could include behavior frames beyond what was imaged.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the neural spike files, which are very large (each session has tens of thousands of neurons x tens of thousands of frames). The AI also does a behavior-only pre-pass to compute global speed quartile edges, which adds overhead.

ii.
```python
spk_parts = np.load(session.spk_path, allow_pickle=True).item()["spks"]
# Also the speed-scan pass:
def gather_speed_edges(sessions: list[SessionInfo]) -> np.ndarray:
    ...
```

iii. The AI estimated ~17-20 min for full conversion. The neural I/O dominates.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The `trial_frame_groups` function uses a vectorized approach with `np.split` rather than per-trial scanning, which is more efficient than the reference's per-trial approach. However, the `gather_speed_edges` function loops through all sessions' trial groups individually when computing speed samples.

ii.
```python
# Vectorized trial grouping:
changes = np.flatnonzero(np.diff(trial_ids)) + 1
split_frames = np.split(frame_idx, changes)

# Could be more vectorized:
for frames in groups:
    if frames.size == 0:
        continue
    vals = ft_run_speed[frames]
```

iii. The trial grouping is well vectorized. The speed gathering pass iterates per-trial per-session but this is relatively minor compared to I/O.

## 12-c. What processing does the code repeat multiple times?

i. The AI's code calls `trial_frame_groups(beh)` at least twice for each session when `--show-processing` is used: once in `convert_session` and again in `plot_processing_summary`. The `gather_speed_edges` function also computes trial frame groups for every session before the main conversion loop does the same. Additionally, `gather_stimulus_vocabulary` loads behavior for every session to collect wall names, then the main loop loads it again.

ii.
```python
# In gather_speed_edges:
groups = trial_frame_groups(beh)

# In convert_session:
groups = trial_frame_groups(beh)

# In plot_processing_summary:
groups = trial_frame_groups(beh)
```

iii. The behavior file caching (LRU cache with maxsize=2) mitigates repeated file I/O but the recomputation of trial groups is redundant.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI includes neurons from `iarea == 7` ("unassigned_7") and `iarea == -1` ("outside_visual") brain regions. These neurons are outside the visual areas studied in the paper and likely add noise without signal for the decoder. The reference solution drops these neurons. Additionally, the AI's 15-category stimulus output is likely suboptimal for the decoder task compared to the 4-category grouping.

ii.
```python
out[iarea == 7] = 4  # unassigned_7
out[iarea == -1] = 5  # outside_visual
```

iii. Keeping non-visual-area neurons inflates memory usage substantially and may degrade decoder performance by including irrelevant neural activity.
