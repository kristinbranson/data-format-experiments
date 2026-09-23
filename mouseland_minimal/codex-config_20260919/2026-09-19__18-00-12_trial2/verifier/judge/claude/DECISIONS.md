# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads `Imaging_Exp_info.npy` as the master index, iterates over experiment types, and builds a source map of unique session IDs to their behavior file and key. It then loads each behavior file once via `load_behaviors()`, extracting a compact subset of fields. Neural data (`spks`) and retinotopy (`iarea`) are loaded per-session during conversion. It also cross-checks that every session in the experiment table has a matching spike file.

ii.
```python
exp_info = np.load(data_root / "beh" / "Imaging_Exp_info.npy", allow_pickle=True).item()
sources = make_source_map(exp_info, data_root / "spk")
behaviors = load_behaviors(data_root, sources)
```
```python
raw_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
iarea = np.load(data_root / "retinotopy" / retino_name)["iarea"]
```

iii. The agent stated: "The reference code confirms the critical mask exactly: neural samples are used only while the VR is moving and the animal is inside the 4 m textured corridor." The agent explored the data structure, file naming, and cross-referenced behavior and spike files before implementing.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from the `mname` field of each record in the experiment info. A sorted list of unique subject names is built, and `subject_to_id` maps each to an index. `subject_idx` is populated per session.

ii.
```python
subjects = sorted({sources[sid]["record"]["mname"] for sid in session_ids})
subject_to_id = {name: idx for idx, name in enumerate(subjects)}
subject_idx.append(subject_to_id[str(record["mname"])])
```

iii. The agent used the `mname` field directly from the experiment info records.

## 1-c. How are the data split into sessions?

i. A session is a unique combination of mouse name, date, and block, forming the session ID. The `make_source_map` function deduplicates sessions that appear under multiple experiment types by keeping only the first occurrence via `setdefault`.

ii.
```python
def session_id(record: dict) -> str:
    return f"{record['mname']}_{record['datexp']}_{record['blk']}"

sources.setdefault(sid, {...})
```

iii. The agent noted that repeated behavior copies differ only in analysis labels, so the first occurrence is used as the canonical copy.

## 1-d. How are the data split into trials?

i. Trials are identified by `ft_trInd`, which labels each imaging frame with its trial number. The AI creates a `valid_frame_mask` requiring `ft_CorrSpc` (in textured corridor) AND `ft_move > 0` (VR is moving). Unique trial IDs from valid frames define the kept trials.

ii.
```python
def valid_frame_mask(beh, nframes):
    return (
        valid_trial
        & beh["ft_CorrSpc"][:nframes].astype(bool)
        & (beh["ft_move"][:nframes] > 0)
    )
kept_trials = np.unique(trial_ids)
```

iii. The agent stated: "neural samples are used only while the VR is moving and the animal is inside the 4 m textured corridor" based on analyzing the paper's `Get_dprime_selective_neuron` function in `code/utils.py`.

## 1-e. How are trials filtered based on quality controls?

i. The AI does NOT apply any trial-length-based filtering. Trials are kept as long as they have valid frames (in corridor + moving). A session must have at least 2 usable trials. There is no outlier removal for long trials.

ii.
```python
kept_trials = np.unique(trial_ids)
# ...
if len(session_neural) < 2:
    raise RuntimeError(f"Fewer than two usable trials in {sid}")
```

iii. The agent did not mention any trial-length filtering in its reasoning. It relied on the `ft_move > 0` filter to exclude stationary frames within trials, rather than dropping entire long trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the session-level `_neural_data.npy` files (deconvolved calcium traces), concatenated across imaging planes. Neuron area assignments come from `iarea` in the retinotopy `.npz` files.

ii.
```python
raw_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
iarea = np.load(data_root / "retinotopy" / retino_name)["iarea"]
```

iii. The agent confirmed these are "Suite2p non-negative deconvolved fluorescence traces."

## 2-b. How is the `neural` data processed?

i. The AI filters neurons by visual area (keeping V1, medial, anterior, lateral), then selects only valid frame columns (corridor + moving). Neural data is selected plane-by-plane to avoid concatenating the full unfiltered recording first, which saves memory.

ii.
```python
for plane in raw_planes:
    local_keep = np.flatnonzero(keep_neuron[offset : offset + plane.shape[0]])
    selected_planes.append(plane[np.ix_(local_keep, frame_indices)])
    offset += plane.shape[0]
selected_neural = np.concatenate(selected_planes, axis=0)
```
```python
trial_neural = np.ascontiguousarray(selected_neural[:, columns])
```

iii. The agent noted this avoids "first concatenating the full (often multi-gigabyte) unfiltered recording."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are kept only if their `iarea` falls in one of the four paper-defined visual area groups: V1 (code 8), medial (0,1,2,9), anterior (3,4), lateral (5,6). Neurons with area code -1 or 7 are excluded.

ii.
```python
AREA_CODES = {
    "V1": (8,),
    "medial": (0, 1, 2, 9),
    "anterior": (3, 4),
    "lateral": (5, 6),
}
def area_labels(iarea):
    region = np.full(len(iarea), -1, dtype=np.int8)
    for idx, name in enumerate(names):
        region[np.isin(iarea, AREA_CODES[name])] = idx
    keep = region >= 0
    return keep, region[keep].astype(np.int64)
```

iii. The agent referenced the paper's `neu_area_ID` function for the area groupings.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry (trial start). Each trial's neural data starts at the first valid frame of that trial (in corridor and moving) and ends at the last valid frame. Trials vary in length. Within a trial, non-moving frames are excluded, creating potential temporal gaps.

ii.
```python
columns = np.flatnonzero(trial_ids == trial)
frames = frame_indices[columns]
trial_neural = np.ascontiguousarray(selected_neural[:, columns])
```

iii. The agent stated alignment is "entry into the 4 m textured corridor."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The imaging frame rate of 3.17 Hz is the temporal resolution, giving ~315.5 ms time bins. However, because the `ft_move > 0` filter removes non-moving frames, some frames within a trial may be dropped, meaning the time series may have gaps (not uniformly spaced).

ii.
```python
FRAME_RATE_HZ = 3.17
"time_bin_size": 1000.0 / FRAME_RATE_HZ,
```

iii. The agent set the time bin size to 1000/3.17 ms.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime` (the timestamp of the sound cue for each trial) and `ft` (the timestamp of each imaging frame).

ii.
```python
time_to_sound = (
    (beh["SoundTime"][trial] - times) * SECONDS_PER_DAY
).astype(np.float32)
```
where `times = beh["ft"][frames]`.

iii. The agent used the recorded timestamps directly rather than frame-number-based interpolation.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The difference between `SoundTime[trial]` and each frame's timestamp `ft[frame]` is computed, multiplied by 86400 to convert from MATLAB datenum (days) to seconds. The sign convention is positive before the cue and negative after (sound time minus current time).

ii.
```python
time_to_sound = (
    (beh["SoundTime"][trial] - times) * SECONDS_PER_DAY
).astype(np.float32)
```

iii. The agent stated this follows the "time_to_sound" sign convention.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the same `frames` array used to select neural columns for that trial, so the timestamps correspond exactly to the neural time bins.

ii.
```python
frames = frame_indices[columns]
times = beh["ft"][frames]
time_to_sound = ((beh["SoundTime"][trial] - times) * SECONDS_PER_DAY).astype(np.float32)
```

iii. All time-varying variables use the same frame indices as the neural data.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the `datexp` field in each session's experiment record, which is parsed as a calendar date. The elapsed calendar days from each subject's earliest imaging session is computed.

ii.
```python
def elapsed_days_by_session(session_ids, sources):
    for sid in session_ids:
        record = sources[sid]["record"]
        date = datetime.strptime(record["datexp"], "%Y_%m_%d").date()
        subject = str(record["mname"])
        parsed[sid] = (subject, date)
        first_date[subject] = min(date, first_date.get(subject, date))
    return {
        sid: float((date - first_date[subject]).days)
        for sid, (subject, date) in parsed.items()
    }
```

iii. The agent noted: "Dates are available for every recording, so elapsed calendar day is the only uniform, non-imputed continuous measure across supervised, unsupervised, grating, and naive mice."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Each session's date is parsed from `datexp`. Per subject, the earliest date is found. The day of training for each session is the difference in calendar days from that earliest date. This is a float value broadcast across all time bins of a trial.

ii.
```python
trial_input[1] = day_by_session[sid]
```

iii. The agent chose elapsed calendar days rather than counting recording sessions (0, 1, 2...).

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time` (the timestamp of trial start) and `ft` (the timestamp of each frame).

ii.
```python
time_from_start = (
    (times - beh["Trial_start_time"][trial]) * SECONDS_PER_DAY
).astype(np.float32)
```

iii. The agent used the recorded trial start timestamp.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The difference between each frame's timestamp and `Trial_start_time[trial]` is computed, multiplied by 86400 to convert from days to seconds. The sign convention is positive after trial start.

ii.
```python
time_from_start = (
    (times - beh["Trial_start_time"][trial]) * SECONDS_PER_DAY
).astype(np.float32)
```

iii. Simple timestamp subtraction and unit conversion.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed from the same `frames` used for the neural data of that trial.

ii.
```python
frames = frame_indices[columns]
times = beh["ft"][frames]
time_from_start = ((times - beh["Trial_start_time"][trial]) * SECONDS_PER_DAY).astype(np.float32)
```

iii. All variables share the same frame indices.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which indicates whether each trial is in the rewarded corridor.

ii.
```python
trial_input[3] = float(beh["isRew"][trial])
```

iii. Directly from the behavior data.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast to float; 1.0 if rewarded, 0.0 if not. Broadcast as a constant across all time bins of the trial.

ii.
```python
trial_input[3] = float(beh["isRew"][trial])
```

iii. No further processing needed.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, which names the wall texture for each trial.

ii.
```python
stimulus_values = visual_vocabulary(behaviors)
stimulus_to_id = {name: idx for idx, name in enumerate(stimulus_values)}
trial_output[0] = stimulus_to_id[str(beh["WallName"][trial])]
```

iii. The agent used `WallName` for each trial.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI collects ALL unique `WallName` values across the dataset (15 unique names like `circle1`, `circle2`, `leaf1`, `leaf1_swap1`, etc.) and assigns each a unique integer ID. It does NOT group them into the 4 base texture categories (circle, leaf, rock, wood) as the reference does.

ii.
```python
def visual_vocabulary(behaviors):
    return sorted({str(wall) for beh in behaviors.values() for wall in beh["WallName"]})
# Results in 15 categories instead of 4
stimulus_to_id = {name: idx for idx, name in enumerate(stimulus_values)}
trial_output[0] = stimulus_to_id[str(beh["WallName"][trial])]
```

iii. The agent did not explicitly discuss grouping textures into base categories. It treated each unique wall name as a separate stimulus.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the frame numbers at which licks occurred.

ii.
```python
lick_frame = beh["LickFr"]
lick_frame = lick_frame[np.isfinite(lick_frame)].astype(np.int64)
trial_output[1] = np.isin(frames, lick_frame).astype(np.int64)
```

iii. The agent used `LickFr` directly.

## 8-b. What processing is involved in computing `output` *Licking*?

i. `LickFr` values are filtered for finite values and cast to int. For each trial, a binary vector is created: 1 if the frame index is in the lick frame set, 0 otherwise.

ii.
```python
lick_frame = beh["LickFr"]
lick_frame = lick_frame[np.isfinite(lick_frame)].astype(np.int64)
trial_output[1] = np.isin(frames, lick_frame).astype(np.int64)
```

iii. Binary encoding of lick presence per frame.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick check uses the same `frames` array (frame indices) as the neural data for that trial.

ii.
```python
frames = frame_indices[columns]
trial_output[1] = np.isin(frames, lick_frame).astype(np.int64)
```

iii. Aligned by shared frame indices.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position in the corridor at each imaging frame (in decimeters), and `Texture_Length` (the length of the textured corridor in decimeters).

ii.
```python
position_bin = np.floor(
    beh["ft_Pos"][frames] / (beh["texture_length_dm"] / 4.0)
).astype(np.int64)
position_bin = np.clip(position_bin, 0, 3)
```

iii. The agent used `ft_Pos` and `Texture_Length` from the behavior data.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimeters is divided by `Texture_Length/4` (= 10 dm = 1 m per bin), floored, and clipped to [0, 3], giving 4 equal 1-m bins.

ii.
```python
position_bin = np.floor(
    beh["ft_Pos"][frames] / (beh["texture_length_dm"] / 4.0)
).astype(np.int64)
position_bin = np.clip(position_bin, 0, 3)
```

iii. The agent used the recorded texture length rather than hardcoding 10 dm as the bin width.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four equal-length bins of 1 m each: 0-1m, 1-2m, 2-3m, 3-4m.

ii.
```python
"output_values": [
    ...
    ["0-1 m", "1-2 m", "2-3 m", "3-4 m"],
    ...
]
```

iii. Matches the instruction requirement of 4 equal-length 1-m bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is read from `ft_Pos` at the same `frames` indices used for neural data.

ii.
```python
frames = frame_indices[columns]
position_bin = np.floor(beh["ft_Pos"][frames] / (beh["texture_length_dm"] / 4.0)).astype(np.int64)
```

iii. Aligned by shared frame indices.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
speed_bin = np.digitize(
    beh["ft_RunSpeed"][frames], speed_edges, right=False
).astype(np.int64)
```

iii. The agent used `ft_RunSpeed` directly.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes dataset-wide quartile edges over all valid (corridor + moving) frames using `np.quantile` at [0.25, 0.50, 0.75], then uses `np.digitize` to bin each frame's speed. This produces 4 bins based on global speed thresholds.

ii.
```python
def global_speed_edges(behaviors):
    speeds = []
    for beh in behaviors.values():
        mask = valid_frame_mask(beh, len(beh["ft"]))
        speeds.append(beh["ft_RunSpeed"][: len(mask)][mask].astype(np.float32))
    all_speeds = np.concatenate(speeds)
    edges = np.quantile(all_speeds, (0.25, 0.50, 0.75))
    return edges.astype(np.float64)

speed_bin = np.digitize(beh["ft_RunSpeed"][frames], speed_edges, right=False).astype(np.int64)
```

iii. The agent computed global quartile edges and verified they are distinct.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four bins based on dataset-wide quartile thresholds: lowest 25%, 25-50%, 50-75%, highest 25%.

ii.
```python
["lowest 25%", "25-50%", "50-75%", "highest 25%"]
```

iii. Global quartile edges applied via `np.digitize`.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is read from `ft_RunSpeed` at the same `frames` indices as the neural data.

ii.
```python
speed_bin = np.digitize(beh["ft_RunSpeed"][frames], speed_edges, right=False).astype(np.int64)
```

iii. Aligned by shared frame indices.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: (1) behavior arrays are truncated to the minimum neural frame count across planes; (2) `LickFr` values are filtered for `np.isfinite` before use; (3) `ft_trInd` is validated for finite values and valid range; (4) sessions with fewer than 2 usable trials raise an error. The behavior data is compacted to only needed fields with explicit copies.

ii.
```python
nframes = min(plane.shape[1] for plane in raw_planes)
lick_frame = lick_frame[np.isfinite(lick_frame)].astype(np.int64)
valid_trial = np.isfinite(trial) & (trial >= 0) & (trial < beh["ntrials"])
if len(session_neural) < 2:
    raise RuntimeError(f"Fewer than two usable trials in {sid}")
```

iii. The agent added explicit validation checks beyond the reference.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the neural spike files (each session's `_neural_data.npy` file can be multi-gigabyte), and performing the plane-by-plane neuron and frame selection.

ii.
```python
raw_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
for plane in raw_planes:
    selected_planes.append(plane[np.ix_(local_keep, frame_indices)])
```

iii. The agent's session-by-session processing dominated by I/O of neural data files.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop that iterates over `kept_trials` and extracts columns from the pre-selected neural array could potentially be avoided by grouping operations. The `visual_vocabulary` function iterates over all behaviors to collect unique wall names.

ii.
```python
for trial in kept_trials:
    columns = np.flatnonzero(trial_ids == trial)
    # ... per-trial extraction
```

iii. The per-trial extraction loop scans `trial_ids` once per trial rather than grouping all trials in one pass.

## 12-c. What processing does the code repeat multiple times?

i. The `valid_frame_mask` is computed twice for each session: once in `global_speed_edges` (over all sessions) and once during conversion. The behavior loading happens twice as well (once for speed edges, once for conversion), though the AI mitigates this by loading behaviors into a dict first.

ii.
```python
# In global_speed_edges:
mask = valid_frame_mask(beh, len(beh["ft"]))
# In convert, per session:
frame_mask = valid_frame_mask(beh, nframes)
```

iii. The speed edge computation requires a pass over all sessions before the main conversion loop.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores extensive metadata (session_info with reward_mode, source_experiment_type, training_day, n_valid_frames, n_neurons_raw, etc.) that is not used by the decoder. It also validates neuron/behavior consistency with explicit error checks that only run once.

ii.
```python
session_info.append({
    "session_id": sid, "subject": str(record["mname"]),
    "date": str(record["datexp"]), "block": str(record["blk"]),
    "source_experiment_type": source["experiment_type"],
    "reward_mode": beh["reward_mode"],
    "training_day": day_by_session[sid],
    "n_trials": len(session_neural),
    "n_valid_frames": len(frame_indices),
    "n_neurons_raw": raw_neuron_count,
    "n_neurons_retained": int(keep_neuron.sum()),
})
```

iii. The extra metadata is harmless but not required by the decoder.
