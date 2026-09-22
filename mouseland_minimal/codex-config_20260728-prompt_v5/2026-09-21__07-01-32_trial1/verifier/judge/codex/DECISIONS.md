# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use `Imaging_Exp_info.npy` as the master session list. Instead, it scans every `Beh_*.npy` file to build a behavior-candidate index keyed by a raw session id, scores duplicate behavior entries heuristically, then iterates over every spike file in `spk/` as the authoritative session list. For each spike file it loads one selected behavior entry and one retinotopy file; neural trial matrices are later stored as memmap-backed sidecars rather than directly materialized inside the pickle.

ii.
```python
def build_behavior_index():
    candidates = defaultdict(list)
    behavior_files = sorted(
        fp for fp in BEH_DIR.glob("Beh_*.npy")
        if fp.name not in {"Imaging_Exp_info.npy", "example_bef_and_aft_learning_behavior.npy"}
    )
```
```python
spk_files = sorted(SPK_DIR.glob("*_neural_data.npy"))
for spk_file in spk_files:
    raw_key = spk_file.name.replace("_neural_data.npy", "")
    beh_fname, beh_key = select_behavior_candidate(raw_key, behavior_index[raw_key])
    beh = np.load(BEH_DIR / beh_fname, allow_pickle=True).item()[beh_key]
```

iii. In the trajectory, the agent said it found duplicated analysis entries in `Imaging_Exp_info.npy` and wanted to deduplicate by raw recording key instead (steps 31 and 74). It also justified the memmap-sidecar design as a tractability choice to avoid very large RAM use and a huge pickle (steps 80, 83, 96, and 221).

## 1-b. How are the data split into subjects (mice)?

i. Subjects are derived from the first token of each raw session key, e.g. `TX123_2023_12_18_1 -> TX123`. The final `subjects` list is the sorted unique set of those names, and `subject_idx` maps each session to its subject.

ii.
```python
def parse_session_key(raw_key: str):
    parts = raw_key.split("_")
    subject = parts[0]
    datestr = "_".join(parts[1:4])
    blk = parts[4]
    return subject, datestr, blk
```
```python
subjects = sorted({plan.subject for plan in session_plans})
subject_to_index = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. There is no separate justification beyond using the raw session naming convention consistently. The trajectory treats mouse identity as encoded in the session key from the outset.

## 1-c. How are the data split into sessions?

i. A session is defined by one spike file name of the form `<mouse>_<date>_<block>_neural_data.npy`. Behavior keys with `_swap1` or `_swap2` are collapsed back to the corresponding raw session id for matching, so there is one converted session per spike file.

ii.
```python
def raw_session_key_from_behavior_key(key: str) -> str:
    if key.endswith("_swap1") or key.endswith("_swap2"):
        return key.rsplit("_", 1)[0]
    return key
```
```python
spk_files = sorted(SPK_DIR.glob("*_neural_data.npy"))
for spk_file in spk_files:
    raw_key = spk_file.name.replace("_neural_data.npy", "")
```

iii. The agent explicitly noted that `swap1/swap2` sessions should deduplicate to the full raw key including block, not a shorter date-only key (step 74). Earlier it also noted that the master index had repeated analysis entries for the same underlying recording (step 31).

## 1-d. How are the data split into trials?

i. Trials are defined by looping over `range(ntrials)` and taking all imaging frames whose `ft_trInd` equals the trial id and whose `ft_CorrSpc` flag is true. This keeps only frames inside the corridor texture, and each trial remains variable length.

ii.
```python
for tr in range(ntrials):
    idx = np.flatnonzero((ft_tr == tr) & corr)
```
```python
TrialPlan(
    trial_index=tr,
    frame_idx=idx.astype(np.int32),
    ...
)
```

iii. The trajectory repeatedly says the agent wanted corridor-entry alignment and corridor-only frames, not a fixed padded window (steps 14, 26, 67, and 221).

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped if they have fewer than `5` corridor frames, corridor duration above `40` s, or a within-trial maximum frame gap above `1` s. A session is rejected if fewer than two trials survive. This is different from the human reference’s zero-frame and 99th-percentile-length rule.

ii.
```python
TRIAL_DURATION_MAX_S = 40.0
MAX_FRAME_GAP_S = 1.0
MIN_FRAMES_PER_TRIAL = 5
```
```python
if idx.size < MIN_FRAMES_PER_TRIAL:
    continue
duration_s = float((gray_time[tr] - trial_start[tr]) * 24.0 * 3600.0)
if idx.size > 1:
    max_gap_s = float(np.max(np.diff(ft[idx]) * 24.0 * 3600.0))
if duration_s > TRIAL_DURATION_MAX_S or max_gap_s > MAX_FRAME_GAP_S:
    continue
```

iii. The agent justified this as filtering “pathological” pause-heavy trials rather than padding them, after inspecting extreme outliers and long stopped trials (steps 38, 46, 50, 67, and 221).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from `spks` inside `spk/<session>_neural_data.npy`. Brain-region labels come from `iarea` inside the corresponding retinotopy `.npz`.

ii.
```python
spk_obj = np.load(SPK_DIR / f"{plan.raw_key}_neural_data.npy", allow_pickle=True).item()
spk_chunks = spk_obj["spks"]
```
```python
retino = np.load(RETINO_DIR / f"{raw_key.rsplit('_', 1)[0]}_trans.npz", allow_pickle=True)
region_idx = region_index_from_iarea(retino["iarea"])
```

iii. The trajectory shows the agent inspected both raw spike files and retinotopy files before coding, and treated them as the authoritative neural inputs (steps 16, 17, 26, and 74).

## 2-b. How is the `neural` data processed?

i. The AI does not further denoise or normalize the traces. It copies all kept trial frames into a per-session float16 memmap laid out as time-by-neuron, then exposes each trial as a `MappedTrialArray` transposed back to neuron-by-time.

ii.
```python
base = np.memmap(out_path, dtype=np.float16, mode="w+", shape=base_shape)
for chunk in spk_chunks:
    ...
    block = chunk[:, keep_idx[c0:c1]].T
    base[c0:c1, col_start:col_start + n_chunk] = block
```
```python
arr = base[row_start:row_stop, :].T.view(MappedTrialArray)
```

iii. The trajectory says the agent wanted to keep raw imaging-frame traces and avoid extra resampling, but needed a memmap-backed representation because keeping all neurons and frames in RAM was too large (steps 57, 80, 83, 96, and 221).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are not filtered out. Every neuron is assigned to `V1`, `mHV`, `aHV`, `lHV`, or `unknown`, and all of them are kept, including `unknown`.

ii.
```python
REGION_NAMES = ["V1", "mHV", "aHV", "lHV", "unknown"]
```
```python
def region_index_from_iarea(iarea: np.ndarray) -> np.ndarray:
    out = np.full(iarea.shape, REGION_TO_INDEX["unknown"], dtype=np.int16)
    ...
    return out
```
```python
if total_neurons != len(plan.region_idx):
    raise RuntimeError(...)
```

iii. The trajectory shows the agent explicitly considered whether to use the paper’s four pooled areas or finer labels, and ultimately reported keeping `V1`, `mHV`, `aHV`, `lHV`, and `unknown` (steps 74 and 221).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to corridor entry by taking the frame indices for each trial’s corridor period and slicing the neural data on exactly those indices. Trials start at their own first corridor frame and remain variable length.

ii.
```python
idx = np.flatnonzero((ft_tr == tr) & corr)
```
```python
neural_trials.append(
    MappedTrialArray(memmap_path, base_shape, np.dtype(np.float16).str, tp.row_start, tp.row_stop)
)
```

iii. The agent repeatedly justified corridor-entry alignment and variable-length trials in the trajectory (steps 26, 67, and 221).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin is the native imaging frame. The script estimates the bin width from the median `ft` frame interval across sessions, stores that median in milliseconds, and does not apply any temporal rebinning.

ii.
```python
dt = np.diff(ft) * 24.0 * 3600.0
median_dt_s = float(np.median(dt)) if dt.size else np.nan
```
```python
median_time_bin_ms = 1000.0 * float(np.median(np.asarray(time_bin_values)))
```

iii. The trajectory says the agent confirmed the imaging time base from `ft` at about 3.18 Hz and wanted to avoid inventing extra resampling (steps 26 and 67).

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The converted `time_to_sound_cue` input is derived from per-frame timestamps `ft` and per-trial cue times `SoundTime`.

ii.
```python
ft = np.asarray(beh["ft"][:nfr], dtype=np.float64)
sound_time = np.asarray(beh["SoundTime"], dtype=np.float64)
```

iii. The trajectory shows the agent investigated cue availability and wanted a deterministic cue-time representation without extra interpolation if direct times were already present (step 90).

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, it subtracts each frame time from that trial’s `SoundTime` and converts the result from days to seconds. It does not interpolate `SoundFr`.

ii.
```python
t_frame = ft[idx]
((sound_time[tr] - t_frame) * 24.0 * 3600.0).astype(np.float32)
```

iii. The trajectory indicates the agent checked for missing cue values before finalizing this input, then used the direct session time stamps (step 90).

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated on the same per-trial frame index array `idx` used for the neural trial, so it has the same number of time bins as the neural data.

ii.
```python
t_frame = ft[idx]
neural_trials.append(...)
input_trial = np.vstack([
    ((sound_time[tr] - t_frame) * 24.0 * 3600.0).astype(np.float32),
    ...
])
```

iii. The trajectory consistently says all converted streams should be aligned on the imaging-frame grid and sliced with the same trial windows (steps 20, 26, and 221).

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The value is derived from the subject name and calendar date parsed from the raw session key, which itself comes from the spike filename.

ii.
```python
subject, _, block = parse_session_key(raw_key)
date = session_date(raw_key)
```
```python
first_date = {
    subject: min(plan.date for plan in session_plans if plan.subject == subject)
    for subject in subjects
}
```

iii. There is no explicit separate justification in the trajectory beyond using session dates extracted from the raw keys.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The script computes actual elapsed calendar days since the first recording date for each subject, not ordinal training-session count. That value is then broadcast across all time bins of each trial.

ii.
```python
for plan in session_plans:
    plan.day_of_training = float((plan.date - first_date[plan.subject]).days)
```
```python
np.full((T,), plan.day_of_training, dtype=np.float32)
```

iii. The trajectory does not contain a detailed defense of this choice; it appears to be an inference from the parsed calendar date in the session id.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. The converted `time_since_trial_start` input is derived from per-frame timestamps `ft` and per-trial start times `Trial_start_time`.

ii.
```python
ft = np.asarray(beh["ft"][:nfr], dtype=np.float64)
trial_start = np.asarray(beh["Trial_start_time"], dtype=np.float64)
```

iii. The trajectory shows the agent was trying to stay on the raw imaging time base rather than derive a new one from fixed windows (steps 26 and 67).

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each trial, it subtracts `Trial_start_time[tr]` from each frame’s timestamp and converts the difference from days to seconds.

ii.
```python
t_frame = ft[idx]
((t_frame - trial_start[tr]) * 24.0 * 3600.0).astype(np.float32)
```

iii. No additional justification was recorded beyond aligning everything to the imaging-frame time base.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed from the same `idx` frame subset used for the neural trial, so each trial’s neural and `time_since_trial_start` arrays have matching lengths.

ii.
```python
t_frame = ft[idx]
neural_trials.append(...)
input_trial = np.vstack([
    ...,
    ((t_frame - trial_start[tr]) * 24.0 * 3600.0).astype(np.float32),
    ...
])
```

iii. The trajectory repeatedly frames alignment as shared imaging-frame indexing across streams (steps 20, 26, and 221).

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial `isRew` flag.

ii.
```python
is_rew = np.asarray(beh["isRew"], dtype=bool)
```

iii. The trajectory does not discuss this choice in detail; it follows directly from the behavior schema inspection.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No transformation beyond boolean-to-float conversion and trial-wise broadcasting is applied. Every time bin in a trial gets the same reward-availability value.

ii.
```python
np.full((T,), float(is_rew[tr]), dtype=np.float32)
```

iii. No separate justification was recorded. The agent treated reward availability as a straightforward per-trial input.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName` for each trial.

ii.
```python
def stimulus_series_for_trial(beh, tr: int, T: int):
    stim_idx = canonical_stimulus_index(str(beh["WallName"][tr]))
    return np.full((T,), stim_idx, dtype=np.int16)
```

iii. The trajectory notes that the agent investigated `TrialStim` versus `WallName` and the special handling of swap sessions before deciding on its mapping scheme (steps 31, 42, 62, 63, and 74).

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The script uses an 8-category stimulus scheme: `circle1`, `circle2`, `circle3`, `leaf1`, `leaf2`, `leaf3`, `leaf1_swap1`, and `leaf1_swap2`. Raw `rock*` walls are remapped onto `circle*`, and raw `wood*` walls are remapped onto `leaf*`, so the output categories are neither the raw 15 names nor the 4 coarse texture groups from the reference.

ii.
```python
STIMULUS_CATEGORIES = [
    "circle1", "circle2", "circle3",
    "leaf1", "leaf2", "leaf3",
    "leaf1_swap1", "leaf1_swap2",
]
```
```python
WALL_TO_CANONICAL = {
    "rock1": "circle1",
    "rock2": "circle2",
    "wood1": "leaf1",
    "wood2": "leaf2",
    "wood5": "leaf3",
    ...
}
```

iii. The trajectory shows the agent spent substantial time inspecting `TrialStim`, `WallName`, placeholder labels, and swap-session naming, then settled on this canonicalization (steps 31, 42, 62, 63, 65, 66, and 221).

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr` and `LickTrind`.

ii.
```python
lick_frames = np.asarray(np.rint(beh["LickFr"]).astype(np.int64))
lick_trials = np.asarray(np.rint(beh["LickTrind"]).astype(np.int64))
```

iii. The trajectory shows the agent examined fractional lick-frame indexing and trial assignment before choosing this representation (steps 87, 88, and 89).

## 8-b. What processing is involved in computing `output` *Licking*?

i. The script rounds each lick’s fractional frame index to the nearest imaging frame, restricts licks to the current trial using `LickTrind`, clips any out-of-range frame to `[0, nfr - 1]`, maps kept lick frames into the current trial’s relative indices, and sets those positions to `1`.

ii.
```python
lick = np.zeros((frame_idx.size,), dtype=np.int16)
lick_frames = np.asarray(np.rint(beh["LickFr"]).astype(np.int64))
lick_trials = np.asarray(np.rint(beh["LickTrind"]).astype(np.int64))
sel = lick_trials == int(tr)
lick_frames = np.clip(lick_frames[sel], 0, nfr - 1)
...
lick[np.unique(rel)] = 1
```

iii. The agent explicitly checked whether nearest-frame assignment was accurate enough for a binary frame raster and then implemented rounding-based assignment (steps 87, 88, and 89).

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks are mapped onto the same imaging-frame grid and then restricted to the same per-trial `idx` frame subset as the neural data.

ii.
```python
frame_to_rel[frame_idx] = np.arange(frame_idx.size, dtype=np.int32)
rel = frame_to_rel[lick_frames]
```
```python
neural_trials.append(...)
lick = lick_series_for_trial(beh, tr, idx, nfr)
```

iii. The trajectory’s stated goal was alignment on the imaging-frame grid, which is exactly how the code handles licking (steps 20, 26, and 221).

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from per-frame corridor position `ft_Pos`.

ii.
```python
pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
```

iii. No separate justification was recorded; the position stream comes directly from the behavior file.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The per-frame position values are clipped to the corridor texture range `[0, 39.999]` and then converted into meter-scale bins by dividing by `10` and taking the floor.

ii.
```python
def position_series_for_trial(pos_trial: np.ndarray):
    pos_clipped = np.clip(pos_trial, 0.0, 39.999)
    return np.floor(pos_clipped / 10.0).astype(np.int16)
```

iii. The trajectory summary at the end reports that trials are corridor-aligned and restricted to the 4 m corridor, which is consistent with this position processing (step 221).

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The thresholds are fixed spatial cut points at 10, 20, and 30 decimeters, yielding four categories: `0_to_1m`, `1_to_2m`, `2_to_3m`, and `3_to_4m`.

ii.
```python
"output_values": [
    STIMULUS_CATEGORIES,
    ["no_lick", "lick"],
    ["0_to_1m", "1_to_2m", "2_to_3m", "3_to_4m"],
    ["q1", "q2", "q3", "q4"],
],
```
```python
return np.floor(pos_clipped / 10.0).astype(np.int16)
```

iii. No separate justification was recorded beyond following the decoder task’s request for four equal-length 1 m bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is sampled at `pos[idx]`, using the same trial frame indices as the neural data.

ii.
```python
idx = tp.frame_idx
...
pos_bins = position_series_for_trial(pos[idx])
```

iii. The trajectory repeatedly described the conversion as imaging-frame aligned across modalities (steps 20, 26, and 221).

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from per-frame running speed `ft_RunSpeed`.

ii.
```python
speed = np.clip(np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32), 0.0, None)
```

iii. No separate justification was recorded beyond inspecting the behavior schema and deciding to keep the imaging-frame time base.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. During planning, the script collects speed samples from all kept trial frames across all sessions, computes global 25th/50th/75th percentile thresholds, and later bins each trial’s speeds against those fixed global thresholds with `np.digitize`.

ii.
```python
speed_values.extend(speed_values_curr)
...
all_speed = np.concatenate(speed_values).astype(np.float32)
speed_quantiles = np.quantile(all_speed, [0.25, 0.50, 0.75]).astype(np.float32)
```
```python
def speed_series_for_trial(speed_trial: np.ndarray, speed_quantiles: np.ndarray):
    return np.digitize(speed_trial, speed_quantiles, right=True).astype(np.int16)
```

iii. The trajectory does not provide a detailed defense of global quantile binning; it appears to have been chosen to make the bins approximately balanced across the whole converted dataset.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The thresholds are the dataset-wide empirical quartiles of kept running-speed values. The code stores them in metadata and assigns categories `q1` to `q4` with `np.digitize`.

ii.
```python
"speed_bin_quantiles_cm_per_s": speed_quantiles.astype(float).tolist(),
```
```python
return np.digitize(speed_trial, speed_quantiles, right=True).astype(np.int16)
```

iii. No separate justification beyond the implicit “25% of the data” interpretation appears in the trajectory.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is sampled at the same per-trial frame indices `idx` as the neural data.

ii.
```python
idx = tp.frame_idx
...
speed_bins = speed_series_for_trial(speed[idx], speed_quantiles)
```

iii. The trajectory consistently describes all outputs as frame-aligned to the neural data (steps 20, 26, and 221).

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The script explicitly handles small behavior/spike frame-count mismatches by recomputing trial plans against the spike-frame length when the mismatch is at most 2 frames, clips lick frames into valid bounds, and raises hard errors for larger mismatches or sessions with fewer than two surviving trials. It does not add explicit imputation for missing cue times; `SoundTime` is used directly.

ii.
```python
if spk_nfr != plan.nframes_behavior:
    if abs(spk_nfr - plan.nframes_behavior) > 2:
        raise RuntimeError(...)
    trial_plans, median_dt_s, _ = compute_trial_plans_from_behavior(beh, spk_nfr)
    ...
```
```python
lick_frames = np.clip(lick_frames[sel], 0, nfr - 1)
```

iii. The trajectory explicitly mentions discovering off-by-one/off-by-two behavior-versus-spike mismatches and refactoring the code to trim against the actual spike length (step 107). It also mentions checking cue availability before finalizing the inputs (step 90).

## 12-a. What are the most time-consuming steps of the code?

i. The dominant cost is neural-data I/O: reading every large spike file, copying selected frames into memmap sidecars, and later scanning those memmaps during verification. The planning pass also rescans all behavior files to build candidate indices and collect speed statistics.

ii.
```python
spk_obj = np.load(SPK_DIR / f"{plan.raw_key}_neural_data.npy", allow_pickle=True).item()
...
base = np.memmap(out_path, dtype=np.float16, mode="w+", shape=base_shape)
for chunk in spk_chunks:
    for c0 in range(0, keep_idx.size, time_block):
        block = chunk[:, keep_idx[c0:c1]].T
        base[c0:c1, col_start:col_start + n_chunk] = block
```

iii. The trajectory repeatedly emphasizes dataset size, large spike files, and verification over 108 GB of neural sidecars as the expensive part of the run (steps 57, 80, 120, 177, 188, and 221).

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The code repeatedly scans the full frame arrays once per trial in `compute_trial_plans_from_behavior`, and it also performs nested Python loops over spike chunks and 4096-frame blocks while writing memmaps. Those are the clearest vectorization targets.

ii.
```python
for tr in range(ntrials):
    idx = np.flatnonzero((ft_tr == tr) & corr)
```
```python
for chunk in spk_chunks:
    for c0 in range(0, keep_idx.size, time_block):
        c1 = min(c0 + time_block, keep_idx.size)
        block = chunk[:, keep_idx[c0:c1]].T
```

iii. The trajectory does not present this as an optimization target explicitly, but it repeatedly frames the conversion as I/O-heavy and built around chunked writes, which explains why these loops were left in place (steps 80, 83, and 96).

## 12-c. What processing does the code repeat multiple times?

i. The code re-loads behavior data several times: once to build the behavior-candidate index, again while building session plans, and again during final assembly. It can also recompute trial plans a second time in `write_session_neural_memmap` if the spike frame count differs from the behavior frame count.

ii.
```python
beh = np.load(fp, allow_pickle=True).item()
```
```python
beh = np.load(BEH_DIR / beh_fname, allow_pickle=True).item()[beh_key]
```
```python
def load_behavior(plan: SessionPlan):
    beh = np.load(BEH_DIR / plan.behavior_file, allow_pickle=True).item()
    return beh[plan.behavior_key]
```
```python
trial_plans, median_dt_s, _ = compute_trial_plans_from_behavior(beh, spk_nfr)
```

iii. The trajectory only partially justifies this repetition. It explicitly says the behavior-candidate indexing was a memory-sensitive planning step and the frame-mismatch recomputation was added to handle real data edge cases (steps 98 and 107).

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes behavior-candidate scoring statistics (`uniq_walls`, `placeholder_count`, etc.), builds `experiment_types`, and stores extensive session metadata and filter metadata that the decoder does not use. Those computations are useful for the converter’s bookkeeping, but they are not consumed in downstream decoding.

ii.
```python
candidates[raw_key].append(
    {
        "file": fp.name,
        "key": key,
        "uniq_walls": len(np.unique(dat["WallName"])),
        "n_finite_stim_id": int(np.isfinite(np.asarray(dat["stim_id"], dtype=float)).sum()),
        "placeholder_count": int(np.sum(np.asarray(dat["TrialStim"]) == "stimulus_of_trial")),
        "has_swap_suffix": int(key.endswith("_swap1") or key.endswith("_swap2")),
    }
)
```
```python
def build_experiment_type_index():
    info = np.load(BEH_DIR / "Imaging_Exp_info.npy", allow_pickle=True).item()
    ...
```
```python
"trial_filter": {
    "max_corridor_duration_s": TRIAL_DURATION_MAX_S,
    "max_frame_gap_s": MAX_FRAME_GAP_S,
    "min_frames_per_trial": MIN_FRAMES_PER_TRIAL,
},
"session_info": [],
```

iii. The trajectory justifies much of this as defensive bookkeeping and auditability around ambiguous behavior files and a large dataset, not as something required by the downstream decoder (steps 31, 74, 96, and 221).
