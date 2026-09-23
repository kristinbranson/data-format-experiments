# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Data is loaded from three subdirectories under `data/`: `beh/` for behavior, `spk/` for deconvolved calcium traces, and `retinotopy/` for visual area assignments. The master index `beh/Imaging_Exp_info.npy` is loaded first, listing every recording grouped by experiment type. Each behavior file `Beh_<exp_type>.npy` contains data for multiple sessions keyed by recording ID. Spike files are loaded per-session from `spk/`, and retinotopy files from `retinotopy/`. The agent discovered 89 unique recordings represented 142 times because the same recording is reused across paper analyses.

ii.
```python
exp_info = np.load(
    data_root / "beh" / "Imaging_Exp_info.npy", allow_pickle=True
).item()
# ...
behavior_file = data_root / "beh" / f"Beh_{experiment_type}.npy"
# ...
spk_path = DATA_ROOT / "spk" / f"{desc['recording_id']}_neural_data.npy"
spk_obj = np.load(spk_path, allow_pickle=True).item()
# ...
ret = np.load(DATA_ROOT / "retinotopy" / f"{desc['entry']['mname']}_{desc['entry']['datexp']}_trans.npz")
```

iii. The agent examined the experiment index structure and identified that recordings are duplicated across experiment types. It loads behavior per-session via `load_behavior()`, neural data per-session from spike files, and retinotopy per-session from retinotopy files.

## 1-b. How are the data split into subjects?

i. The mouse name is taken from `entry['mname']` in the experiment info entries. The unique sorted mouse names form the `subjects` list, and `subject_idx` maps each session to its position in that list. There are 19 unique subjects.

ii.
```python
subjects = sorted({str(x["entry"]["mname"]) for x in sessions})
subject_to_id = {name: i for i, name in enumerate(subjects)}
# ...
subject_idx.append(subject_to_id[str(desc["entry"]["mname"])])
```

iii. The subject identity comes directly from the experiment index field `mname`.

## 1-c. How are the data split into sessions?

i. A session is one unique recording identified by `(mouse_name, date, imaging_block)`, which forms the recording ID. When the same recording appears under multiple experiment types in the index, only the first occurrence is kept (via an `OrderedDict` keyed by recording ID). This yields 89 unique sessions.

ii.
```python
def _recording_id(entry: dict) -> str:
    return f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"

sessions: OrderedDict[str, dict] = OrderedDict()
for experiment_type, entries in exp_info.items():
    for entry in entries:
        rid = _recording_id(entry)
        if rid not in sessions:
            sessions[rid] = { ... }
```

iii. The agent identified that experiment info contains duplicate recordings and used an OrderedDict to deduplicate by recording ID, keeping only the first occurrence.

## 1-d. How are the data split into trials?

i. Trials are split using the behavior data's `ft_trInd` (trial index per frame) and `ntrials` count. Within each trial, only frames passing the paper's frame mask are retained: `(ft_move > 0) & ft_CorrSpc`. This keeps only frames where the animal is actively moving through the textured corridor. Trials with zero retained frames are dropped.

ii.
```python
def retained_frame_mask(beh: dict, nframes: int) -> np.ndarray:
    return (
        np.asarray(beh["ft_move"][:nframes]) > 0
    ) & np.asarray(beh["ft_CorrSpc"][:nframes], dtype=bool)

# In convert_session:
for trial in range(ntrials):
    frames = np.flatnonzero(valid & (trial_stamp == trial))
    if len(frames) == 0:
        dropped_empty += 1
        continue
```

iii. The agent chose to apply the paper's analysis frame mask (`ft_move > 0` and `ft_CorrSpc`), reasoning that this matches the paper's published analysis code for figure creation.

## 1-e. How are trials filtered based on quality controls?

i. Trials are only filtered by having zero retained frames after the running/corridor mask. Unlike the reference solution, the AI does NOT apply a trial length percentile filter to remove extremely long trials. A minimum of 2 trials per session is enforced.

ii.
```python
for trial in range(ntrials):
    frames = np.flatnonzero(valid & (trial_stamp == trial))
    if len(frames) == 0:
        dropped_empty += 1
        continue
# ...
if len(neural) < 2:
    raise ValueError(f"{desc['recording_id']} retained fewer than two trials")
```

iii. The agent relied on the `ft_move` filter to handle stationary trials rather than explicitly filtering by trial length. Since `ft_move > 0` removes frames where the animal is not moving, extremely long stationary periods are implicitly handled by removing their frames (though the trial itself may still be retained with only the moving frames).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the spike files (`spk/<session_id>_neural_data.npy`), which contains deconvolved calcium traces organized by imaging plane. Visual area assignments come from `iarea` in the retinotopy files.

ii.
```python
spk_obj = np.load(spk_path, allow_pickle=True).item()
planes = spk_obj["spks"]
# ...
ret = np.load(DATA_ROOT / "retinotopy" / f"{desc['entry']['mname']}_{desc['entry']['datexp']}_trans.npz")
area_ids = np.asarray(ret["iarea"]).astype(np.int16, copy=False)
```

iii. The agent identified that the data contains Suite2p deconvolved fluorescence traces (spks) organized by imaging plane, consistent with the paper's methods.

## 2-b. How is the `neural` data processed?

i. The planes are loaded individually and filtered by visual area before being assembled into a single neurons-by-time matrix. Neural data is stored as float32 (not float16 as in the reference). Only frames passing the `ft_move & ft_CorrSpc` mask are included.

ii.
```python
neural = np.empty((n_neurons, T), dtype=np.float32)
row = 0
for plane, keep in zip(planes, plane_keep):
    n = len(keep)
    neural[row:row + n] = plane[np.ix_(keep, frames)]
    row += n
```

iii. The agent processes neural data plane-by-plane, applying per-plane neuron selection indices directly to avoid unnecessary concatenation and filtering.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by visual area using the retinotopy data. A neuron is kept only if its `iarea` value maps to one of four regions: V1 (area 8), mHV (areas 0,1,2,9), lHV (areas 5,6), aHV (areas 3,4). Area IDs -1 and 7 are excluded.

ii.
```python
AREA_ID_TO_REGION = {
    8: 0,                         # V1
    0: 1, 1: 1, 2: 1, 9: 1,     # medial higher visual areas
    5: 2, 6: 2,                  # lateral higher visual areas
    3: 3, 4: 3,                  # anterior higher visual areas
}
# ...
keep = np.isin(ids, np.fromiter(AREA_ID_TO_REGION, dtype=np.int16))
```

iii. The agent adopted the exact area grouping from `neu_area_ID()` in the paper repository.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to corridor entry (trial start). Each trial's frames are those where `ft_trInd == trial` AND the retained frame mask is True. The data starts at the first qualifying frame and ends at the last, with non-qualifying frames in between removed.

ii.
```python
frames = np.flatnonzero(valid & (trial_stamp == trial))
# ...
neural[row:row + n] = plane[np.ix_(keep, frames)]
```

iii. The agent aligned to trial start using frame indices, matching the instruction's specification. The `ft_move` filter means non-moving frames within a trial may be excluded, creating gaps in the temporal sequence.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The imaging frames are the time bins. The agent uses a nominal rate of 3.0 Hz (giving ~333.3 ms bins), compared to the reference's 3.17 Hz (~315 ms).

ii.
```python
"time_bin_size": 1000.0 / 3.0,
```

iii. The agent noted imaging was "nominally approximately 3 Hz in the paper" and used that for the metadata time bin size.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime` (a MATLAB datenum per trial) and `ft` (frame timestamps as MATLAB datenums). NOT from `SoundFr` (frame index) as in the reference.

ii.
```python
to_cue_s = (float(beh["SoundTime"][trial]) - frame_times[frames]) * 86_400.0
```

iii. The agent used the absolute timestamp `SoundTime` rather than the frame-based `SoundFr`, computing the time difference in days and converting to seconds.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, `SoundTime` (datenum) minus each frame's timestamp is computed and converted to seconds by multiplying by 86400. The sign convention is positive before the cue and negative after, matching "time TO sound cue."

ii.
```python
to_cue_s = (float(beh["SoundTime"][trial]) - frame_times[frames]) * 86_400.0
inputs[0] = to_cue_s
```

iii. The agent chose to use the absolute timestamp rather than interpolating from frame indices, reasoning that timestamps provide the ground truth timing.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the same `frames` array used to extract neural data, so alignment is automatic.

ii.
```python
frames = np.flatnonzero(valid & (trial_stamp == trial))
# Same frames used for neural and inputs
to_cue_s = (float(beh["SoundTime"][trial]) - frame_times[frames]) * 86_400.0
```

iii. All data streams share the same frame indices within each trial.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From `sess#` in the experiment info entry (when available and finite), with a fallback based on experiment type for missing values.

ii.
```python
def session_day(desc: dict) -> float:
    value = desc["entry"].get("sess#")
    if value is not None and np.isfinite(value):
        return float(value)
    typ = desc["experiment_type"]
    return 1.0 if "after" in typ else 0.0
```

iii. The agent used the paper's own session number field. For missing values (some train2-after records), it inferred stage 0 or 1 based on experiment type to avoid inventing calendar-day estimates.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The `sess#` value is used directly as a float. For missing values, the experiment type determines 0.0 (before) or 1.0 (after). The value is per-trial, broadcast across all frames.

ii.
```python
day = session_day(desc)
# ...
inputs[1] = day
```

iii. The agent documented: "Treating the two paper stages as 0/1 is consistent with train1 and avoids an invented calendar-day estimate (recording dates are not training start dates)."

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time` (a MATLAB datenum per trial) and `ft` (frame timestamps). NOT from `StartFr` (frame index) as in the reference.

ii.
```python
elapsed_s = (frame_times[frames] - float(beh["Trial_start_time"][trial])) * 86_400.0
```

iii. The agent used absolute timestamps for trial start rather than frame-based interpolation.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Each frame's timestamp minus `Trial_start_time` is computed and converted to seconds (multiply by 86400). The result is time-varying, starting near zero.

ii.
```python
elapsed_s = (frame_times[frames] - float(beh["Trial_start_time"][trial])) * 86_400.0
inputs[2] = elapsed_s
```

iii. Direct timestamp arithmetic in MATLAB datenum units, converted to seconds.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Uses the same `frames` array as the neural data.

ii.
```python
frames = np.flatnonzero(valid & (trial_stamp == trial))
elapsed_s = (frame_times[frames] - float(beh["Trial_start_time"][trial])) * 86_400.0
```

iii. Alignment is automatic because all data streams use the same frame indices.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, a per-trial indicator in the behavior data.

ii.
```python
inputs[3] = float(bool(beh["isRew"][trial]))
```

iii. The agent used the direct reward indicator from the behavior data.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The value is cast to bool then to float (0.0 or 1.0), and broadcast across all frames of the trial.

ii.
```python
inputs[3] = float(bool(beh["isRew"][trial]))
```

iii. No processing needed beyond type conversion.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, which names the wall texture for each trial.

ii.
```python
category = visual_category(beh["WallName"][trial])
outputs[0] = category_to_id[category]
```

iii. The agent used `WallName` as the source, noting that `TrialStim` is masked in swap sessions.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The wall name is collapsed to its base category using a regex that extracts only the leading alphabetic characters (e.g., "circle1" -> "circle", "leaf1_swap1" -> "leaf"). Categories are sorted alphabetically and assigned integer indices. The value is per-trial, broadcast across all frames.

ii.
```python
def visual_category(name: str) -> str:
    match = re.match(r"[A-Za-z]+", str(name))
    return match.group(0).lower()

categories = set()
for desc in sessions:
    beh = load_behavior(desc)
    categories.update(visual_category(x) for x in beh["WallName"])
category_names = sorted(categories)
category_to_id = {name: i for i, name in enumerate(category_names)}
```

iii. The agent used regex-based extraction rather than a hardcoded lookup table, which dynamically discovers categories from the data.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (frame indices of licks) and `LickTrind` (trial index of each lick).

ii.
```python
lick_frame = np.rint(np.asarray(beh["LickFr"], dtype=np.float64)).astype(np.int64)
lick_trial = np.asarray(beh["LickTrind"], dtype=np.float64)
```

iii. The agent used both `LickFr` and `LickTrind` to assign licks to their respective trials.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frames are rounded to the nearest integer frame. For each trial, licks assigned to that trial (via `LickTrind`) are checked against the retained frames. A frame is marked 1 if it contains a lick, 0 otherwise. If a lick falls on a frame removed by the running mask, it is discarded.

ii.
```python
trial_licks = lick_frame[lick_trial == trial]
outputs[1] = np.isin(frames, trial_licks).astype(np.int16)
```

iii. The agent used `np.isin` to match lick frames against retained frames, which ensures licks on non-retained frames are dropped.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking uses the same `frames` array as the neural data. A frame is 1 if its index appears in `LickFr` for that trial.

ii.
```python
frames = np.flatnonzero(valid & (trial_stamp == trial))
outputs[1] = np.isin(frames, trial_licks).astype(np.int16)
```

iii. Alignment is inherent because lick frames are matched to the exact retained frames.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position in the corridor at each imaging frame, in decimeters.

ii.
```python
frame_pos = np.asarray(beh["ft_Pos"][:nframes], dtype=np.float64)
outputs[2] = np.clip(np.floor(frame_pos[frames] / 10.0), 0, 3).astype(np.int16)
```

iii. The agent identified that positions are in decimeters from the behavior data.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimeters is divided by 10 (converting to meters), floored, and clipped to [0, 3] to produce four 1-meter bins.

ii.
```python
outputs[2] = np.clip(np.floor(frame_pos[frames] / 10.0), 0, 3).astype(np.int16)
```

iii. Straightforward binning matching the instruction requirement for 4 equal-length 1-m spatial bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four bins of 1 m each: [0-1 m), [1-2 m), [2-3 m), [3-4 m]. The floor division and clip operations produce integer indices 0-3.

ii.
```python
["0-1 m", "1-2 m", "2-3 m", "3-4 m"]
outputs[2] = np.clip(np.floor(frame_pos[frames] / 10.0), 0, 3).astype(np.int16)
```

iii. The bin boundaries are deterministic and match the instruction specification.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Uses the same `frames` array as the neural data.

ii.
```python
outputs[2] = np.clip(np.floor(frame_pos[frames] / 10.0), 0, 3).astype(np.int16)
```

iii. Alignment is automatic through shared frame indices.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
frame_speed = np.asarray(beh["ft_RunSpeed"][:nframes], dtype=np.float64)
```

iii. Direct use of the running speed variable from the behavior data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speed is binned using **global quartile edges** computed across ALL retained frames from ALL sessions. The edges are computed as the 25th, 50th, and 75th percentiles of all running speeds from retained frames (after `ft_move & ft_CorrSpc` filtering). `np.digitize` is then used to assign each speed value to a bin.

ii.
```python
def compute_speed_edges(sessions):
    chunks = []
    for desc in sessions:
        beh = load_behavior(desc)
        nframes = len(beh["ft"]) - 1
        valid = retained_frame_mask(beh, nframes)
        chunks.append(np.asarray(beh["ft_RunSpeed"][:nframes][valid], dtype=np.float32))
    speeds = np.concatenate(chunks)
    edges = np.quantile(speeds, [0.25, 0.50, 0.75])
    return edges, stats

# Applied per trial:
outputs[3] = np.digitize(frame_speed[frames], speed_edges, right=False).astype(np.int16)
```

iii. The agent computed global quartiles over all retained decoder samples, reasoning this ensures each bin holds approximately 25% of the data globally.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four bins defined by global quartile edges: Q1 (slowest), Q2, Q3, Q4 (fastest). Values are assigned using `np.digitize` with the three edge values.

ii.
```python
["Q1 (slowest)", "Q2", "Q3", "Q4 (fastest)"]
outputs[3] = np.digitize(frame_speed[frames], speed_edges, right=False).astype(np.int16)
```

iii. The quartile-based approach ensures approximately equal bin sizes globally.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Uses the same `frames` array as the neural data.

ii.
```python
outputs[3] = np.digitize(frame_speed[frames], speed_edges, right=False).astype(np.int16)
```

iii. Alignment is automatic through shared frame indices.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several handling strategies: (1) Behavior arrays are trimmed to the number of imaged neural frames (`nframes`). (2) The agent uses `len(beh["ft"]) - 1` to estimate nframes in the speed pass, noting the behavior clock has one terminal interpolation sample beyond Suite2p arrays. (3) Trials with zero retained frames are dropped with counter tracking. (4) Sessions with fewer than 2 retained trials raise an error. (5) Missing `sess#` values are handled with a stage-based fallback. (6) Lick frames are rounded to nearest integer.

ii.
```python
nframes = min(a.shape[1] for a in planes)
planes = [a[:, :nframes] for a in planes]
# ...
if len(frames) == 0:
    dropped_empty += 1
    continue
# ...
lick_frame = np.rint(np.asarray(beh["LickFr"], dtype=np.float64)).astype(np.int64)
```

iii. The agent documented handling for various edge cases in comments and metadata fields.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the spike files, which total hundreds of GB. The agent noted this and optimized the speed computation pass to avoid loading neural files unnecessarily.

ii.
```python
spk_obj = np.load(spk_path, allow_pickle=True).item()
```

iii. The agent explicitly noted: "Avoid opening the hundreds of GB of neural files merely to discard that one sample here."

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session()` processes each trial individually including plane-by-plane neural data extraction. The `visual_category()` function is called per trial per session in two separate passes (once for category discovery, once during conversion). The speed edge computation loads all behavior files once for speed computation, then they are reloaded during conversion.

ii.
```python
for trial in range(ntrials):
    frames = np.flatnonzero(valid & (trial_stamp == trial))
    # ... per-trial processing
    for plane, keep in zip(planes, plane_keep):
        neural[row:row + n] = plane[np.ix_(keep, frames)]
```

iii. No explicit discussion of vectorization opportunities in the trajectory.

## 12-c. What processing does the code repeat multiple times?

i. Behavior files are loaded multiple times: once for category discovery, once for speed edge computation, and once during conversion. Each session's behavior is loaded up to 3 times total.

ii.
```python
# First pass - category discovery:
for desc in sessions:
    beh = load_behavior(desc)
    categories.update(visual_category(x) for x in beh["WallName"])

# Second pass - speed edges:
def compute_speed_edges(sessions):
    for desc in sessions:
        beh = load_behavior(desc)
        ...

# Third pass - conversion:
for desc in sessions:
    beh = load_behavior(desc)  # inside convert_session
```

iii. The multiple passes were designed to compute global statistics before the main conversion pass, at the cost of repeated I/O.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes frame interval statistics (`observed_median_frame_interval_ms`, `observed_frame_interval_iqr_ms`) during the speed computation pass. These are stored in metadata but are not used by downstream analysis. The code also stores extensive per-session metadata (`session_info`) with many fields that may not be consumed downstream.

ii.
```python
dt = np.diff(np.asarray(beh["ft"][:nframes], dtype=np.float64)) * 86_400_000.0
frame_dts_ms.append(dt[(dt > 100) & (dt < 1000)])
stats = {
    "speed_quartile_edges_cm_s": edges.tolist(),
    "observed_median_frame_interval_ms": float(np.median(dt_all)),
    "observed_frame_interval_iqr_ms": np.quantile(dt_all, [0.25, 0.75]).tolist(),
}
```

iii. The additional metadata provides diagnostic information but adds computational overhead during the speed pass.
