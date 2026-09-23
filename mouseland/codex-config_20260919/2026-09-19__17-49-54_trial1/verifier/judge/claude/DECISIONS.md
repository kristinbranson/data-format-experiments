# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads `Imaging_Exp_info.npy` as a master index, then iterates over behavior files (`Beh_<exp_type>.npy`) keyed by experiment type, and for each session loads the spike file (`<sid>_neural_data.npy`) and retinotopy file (`<mouse>_<date>_trans.npz`). It builds a "catalog" of 89 unique sessions by deduplicating across experiment types. It also merges complementary stimulus ID mappings from different experiment-type views of the same session via `UniqWalls`/`stim_id`.

ii.
```python
exp_info = np.load(BEH / "Imaging_Exp_info.npy", allow_pickle=True).item()
# ...
for exp_type, records in exp_info.items():
    for rec in records:
        sid = f"{rec['mname']}_{rec['datexp']}_{rec['blk']}"
        record_meta[sid].append({"experiment_type": exp_type, **plain(rec)})
# ...
for exp_type in exp_info:
    behavior = np.load(BEH / f"Beh_{exp_type}.npy", allow_pickle=True).item()
    for key, beh in behavior.items():
        sid = sid_from_key(key)
        if sid not in entries:
            entries[sid] = { ... }
```

```python
obj = np.load(spk_path(sid), allow_pickle=True).item()
planes = obj["spks"]
# ...
with np.load(ret_path(sid), allow_pickle=True) as ret:
    iarea = np.asarray(ret["iarea"])
```

iii. The AI iterates through all experiment types to build a deduplicated catalog of 89 unique physical sessions, merging stimulus ID mappings from complementary behavior views. The approach is documented in CONVERSION_NOTES Step 4 and Step 5.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the mouse name (first component of the session ID). The catalog records each entry's subject, and unique subjects are collected and sorted. `subject_idx` maps each session to its subject.

ii.
```python
subjects = sorted({entry["subject"] for entry in catalog})
subject_lookup = {name: i for i, name in enumerate(subjects)}
# ...
subject_idx.append(subject_lookup[entry["subject"]])
```

iii. The mouse name is directly available from the session ID and experiment index.

## 1-c. How are the data split into sessions?

i. A session corresponds to one unique neural file, identified by `<mouse>_<date>_<block>`. The AI deduplicates across experiment types that reuse the same physical recording, keeping 89 unique sessions.

ii.
```python
sid = f"{rec['mname']}_{rec['datexp']}_{rec['blk']}"
# ...
if sid not in entries:
    entries[sid] = { ... }
```

iii. The AI documents that "a session is a neural file, not a paper-analysis label" and that 89 unique sessions are expected from the data.

## 1-d. How are the data split into trials?

i. Trials are defined by iterating over `range(beh['ntrials'])`. For each trial, valid frames are identified by: `ft_trInd == trial`, `ft_CorrSpc` (in textured corridor), AND `ft_move > 0` (mouse is running). Trials with no valid frames or with unmapped stimulus are dropped.

ii.
```python
valid = (
    np.isfinite(ft_trial)
    & np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool)
    & (np.asarray(beh["ft_move"][:nfr]) > 0)
)
# ...
for tr in range(int(beh["ntrials"])):
    category = wall_map.get(str(beh["WallName"][tr]))
    if category is None:
        dropped["unmapped_stimulus"] += 1
        continue
    frames = np.flatnonzero(valid & (ft_trial == tr))
    if not len(frames):
        dropped["no_valid_running_corridor_frames"] += 1
        continue
```

iii. The AI justifies the `ft_move > 0` filter by citing the reference code's selectivity computation which uses `ft_CorrSpc & ft_move > 0`, and states that "the paper's valid neural regime is active running."

## 1-e. How are trials filtered based on quality controls?

i. The AI filters out trials with (1) unmapped canonical stimulus ID (309 trials) and (2) no valid running-corridor frames. No trial length filtering is applied.

ii.
```python
category = wall_map.get(str(beh["WallName"][tr]))
if category is None:
    dropped["unmapped_stimulus"] += 1
    continue
frames = np.flatnonzero(valid & (ft_trial == tr))
if not len(frames):
    dropped["no_valid_running_corridor_frames"] += 1
    continue
```

iii. The AI documents that 309 trials are excluded for missing canonical stimulus IDs and states "Do not impose selectivity-analysis train/test splits. Exclude only trials lacking a canonical output label or enough valid aligned corridor samples."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the neural data files (`<sid>_neural_data.npy`), which contains a list of float32 neuron-by-frame matrices (one per imaging plane). The visual area of each neuron comes from `iarea` in the retinotopy files.

ii.
```python
obj = np.load(spk_path(sid), allow_pickle=True).item()
planes = obj["spks"]
# ...
with np.load(ret_path(sid), allow_pickle=True) as ret:
    iarea = np.asarray(ret["iarea"])
```

iii. The AI correctly identifies `spks` as deconvolved fluorescence traces produced by Suite2p.

## 2-b. How is the `neural` data processed?

i. The neural data is not further processed (no normalization, z-scoring, or dF/F). Plane arrays are not concatenated into a single array; instead trial arrays are filled plane-by-plane. Data is kept as float32.

ii.
```python
n = np.empty((nneurons, T), dtype=np.float32)
row = 0
for plane in planes:
    nr = plane.shape[0]
    n[row:row + nr] = plane[:, frames]
    row += nr
```

iii. The AI notes "No dF/F, z-scoring, selectivity filtering, or temporal/spatial averaging" and "Raw supplied deconvolved values match the core paper analysis."

## 2-c. How is the `neural` data filtered based on quality controls?

i. ALL neurons are retained, including those outside the four visual areas (iarea=-1 or 7). These are assigned to a 5th brain region called "unmapped/non-visual". This differs from the reference which drops neurons outside V1/mHV/lHV/aHV.

ii.
```python
REGIONS = ["V1", "mHV", "lHV", "aHV", "unmapped/non-visual"]
# ...
def region_idx(iarea: np.ndarray) -> np.ndarray:
    out = np.full(len(iarea), 4, dtype=np.int8)
    out[iarea == 8] = 0
    out[np.isin(iarea, [0, 1, 2, 9])] = 1
    out[np.isin(iarea, [5, 6])] = 2
    out[np.isin(iarea, [3, 4])] = 3
    return out
```

iii. The AI justifies retaining all neurons: "Preserve all recorded cells to avoid losing neural data. Use exactly the four reference groups and assign -1/7 to a fifth `unmapped/non-visual` group; downstream users can reproduce the paper mask by excluding that group."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Aligned to corridor entry (trial start). Only frames satisfying `ft_trInd == trial & ft_CorrSpc & ft_move > 0` are included. Trials have variable length (no padding or truncation).

ii.
```python
frames = np.flatnonzero(valid & (ft_trial == tr))
# ...
n[row:row + nr] = plane[:, frames]
```

iii. The AI states temporal alignment is corridor entry, and documents `off_start: 0.0` and `off_end: None`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning. The native imaging frame rate is used (~315 ms per frame at ~3.17 Hz). The AI computes the actual median inter-frame interval from timestamps rather than using a constant.

ii.
```python
dt = np.diff(np.asarray(beh["ft"][:nfr], dtype=np.float64)) * 86_400_000
intervals.append(dt[np.isfinite(dt) & (dt > 0)])
# ...
nominal_ms = float(np.median(np.concatenate(intervals)))
```

iii. The AI reports the nominal imaging interval as 314.8 ms and stores this in metadata.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime` (the timestamp of the sound cue for each trial) and `ft` (the timestamp of every imaging frame).

ii.
```python
x[0] = (float(beh["SoundTime"][tr]) - sample_time) * 86_400
```
where `sample_time = ft[frames]`.

iii. The AI uses `SoundTime` (a per-trial timestamp in MATLAB datenum format) rather than `SoundFr` (the frame number of the cue).

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The cue time minus the frame time, converted from days to seconds by multiplying by 86400. Positive means cue is in the future.

ii.
```python
sample_time = ft[frames]
x[0] = (float(beh["SoundTime"][tr]) - sample_time) * 86_400
```

iii. The AI documents: "Signed seconds to cue: positive before cue, zero at cue, negative after cue."

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Computed from the same frame timestamps (`ft[frames]`) used for the neural data of that trial, so alignment is inherent.

ii.
```python
sample_time = ft[frames]
x[0] = (float(beh["SoundTime"][tr]) - sample_time) * 86_400
```

iii. All data streams are aligned by using the same set of valid frame indices.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the date component of each session ID (`datexp`), parsed into a datetime.

ii.
```python
def sid_date(sid: str) -> datetime:
    return datetime.strptime("_".join(sid.split("_")[1:4]), "%Y_%m_%d")
# ...
first_date = {
    subject: min(
        sid_date(entry["session_id"])
        for entry in catalog if entry["subject"] == subject
    )
    for subject in subjects
}
# ...
day = float((sid_date(entry["session_id"]) - first_date[entry["subject"]]).days)
```

iii. The AI uses elapsed calendar days from each subject's first imaging session as a complete continuous measure.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each subject, the earliest session date is found. The day of training for each session is the number of calendar days elapsed since that earliest date. This is a float value tiled across all frames in a trial.

ii.
```python
day = float((sid_date(entry["session_id"]) - first_date[entry["subject"]]).days)
# ...
x[1] = day
```

iii. The AI justifies this as "the only complete continuous day measure" since `sess#` and `days` fields are not universally available. This differs from the reference which counts the ordinal session number (0, 1, 2, ...).

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time` (per-trial timestamp) and `ft` (per-frame timestamps).

ii.
```python
x[2] = (sample_time - float(beh["Trial_start_time"][tr])) * 86_400
```

iii. The AI uses `Trial_start_time` directly rather than `StartFr` (frame number of corridor entry).

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The frame time minus the trial start time, converted from days to seconds. Positive after trial start.

ii.
```python
x[2] = (sample_time - float(beh["Trial_start_time"][tr])) * 86_400
```

iii. Simple time difference computation.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed from the same frame timestamps used for neural data extraction.

ii.
```python
sample_time = ft[frames]
x[2] = (sample_time - float(beh["Trial_start_time"][tr])) * 86_400
```

iii. Alignment is inherent through shared frame indices.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks whether a trial is in the rewarded corridor.

ii.
```python
x[3] = float(bool(beh["isRew"][tr]))
```

iii. Direct use of the per-trial reward flag.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast to float (0.0 or 1.0), tiled across all frames of the trial.

ii.
```python
x[3] = float(bool(beh["isRew"][tr]))
```

iii. No additional processing needed.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName` (per-trial wall texture name) mapped via merged `UniqWalls`/`stim_id` mappings from behavior files to canonical stimulus IDs (0-6).

ii.
```python
for wall, stim in zip(beh["UniqWalls"], beh["stim_id"]):
    if not np.isfinite(stim):
        continue
    wall, stim = str(wall), int(stim)
    mappings[sid][wall] = stim
# ...
category = wall_map.get(str(beh["WallName"][tr]))
```

iii. The AI merges stimulus ID mappings from all experiment-type views to maximize coverage.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI uses 7 individual stimulus categories: circle1, circle2, leaf1, leaf2, leaf3, leaf1_swap1, leaf1_swap2, based on the `stim_id` values from the data. Trials with unmapped (NaN) stim_id are excluded (309 trials). The category is tiled across frames.

ii.
```python
STIMULI = [
    "circle1", "circle2", "leaf1", "leaf2", "leaf3",
    "leaf1_swap1", "leaf1_swap2",
]
OUTPUT_VALUES = [
    STIMULI,
    ...
]
# ...
y[0] = category
```

iii. The AI chose to use 7 fine-grained stimulus categories based on the reference code's canonical IDs rather than grouping into 4 broad texture categories (circle, leaf, rock, wood).

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the imaging frame numbers of lick events.

ii.
```python
lick = np.zeros(nfr, dtype=np.int8)
lick_frames = np.asarray(beh["LickFr"], dtype=np.float64)
lick_frames = lick_frames[np.isfinite(lick_frames)].astype(np.int64)
lick_frames = lick_frames[(lick_frames >= 0) & (lick_frames < nfr)]
lick[np.unique(lick_frames)] = 1
```

iii. The AI uses `LickFr` directly, converting frame numbers to a binary per-frame indicator.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frame numbers are filtered for finite values and valid range, then converted to a binary indicator (1 = lick in that frame, 0 = no lick). Multiple licks in one frame still result in 1.

ii.
```python
lick_frames = np.asarray(beh["LickFr"], dtype=np.float64)
lick_frames = lick_frames[np.isfinite(lick_frames)].astype(np.int64)
lick_frames = lick_frames[(lick_frames >= 0) & (lick_frames < nfr)]
lick[np.unique(lick_frames)] = 1
```

iii. The AI handles edge cases (non-finite values, out-of-range frames) more carefully than the reference.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` indexes imaging frames, so the lick binary is on the same grid as neural data. The same valid frame indices are used.

ii.
```python
y[1] = lick[frames]
```

iii. Alignment through shared frame indices.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position in the corridor at each imaging frame, in decimeters.

ii.
```python
pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float64)
# ...
y[2] = np.floor(pos[frames] / 10).astype(np.int8)
```

iii. Direct use of the per-frame position variable.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimeters is divided by 10 (converting to meters) and floored to get the bin index (0-3).

ii.
```python
y[2] = np.floor(pos[frames] / 10).astype(np.int8)
```

iii. The 4 m corridor is divided into 4 equal 1-m bins as requested.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. `floor(pos / 10)` gives bins 0 (0-1m), 1 (1-2m), 2 (2-3m), 3 (3-4m). The AI validates that no values fall outside 0-3.

ii.
```python
y[2] = np.floor(pos[frames] / 10).astype(np.int8)
if np.any((y[2] < 0) | (y[2] > 3)):
    raise ValueError(f"Position outside 0-4 m: {sid} trial {tr}")
```

iii. The corridor mask (`ft_CorrSpc`) ensures position is within the textured 0-4m portion.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` gives one position per imaging frame. The same frame indices are used for neural and position data.

ii.
```python
y[2] = np.floor(pos[frames] / 10).astype(np.int8)
```

iii. Alignment through shared frame indices.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float64)
# ...
y[3] = np.searchsorted(speed_cuts, speed[frames], side="right").astype(np.int8)
```

iii. Direct use of the per-frame running speed variable.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speed is discretized into 4 quartiles using GLOBAL thresholds computed across all valid frames of all sessions. The thresholds are computed as quantiles at 0.25, 0.5, 0.75 of the pooled speed distribution, then `searchsorted` assigns each frame to a quartile bin.

ii.
```python
# Global computation:
speed = np.concatenate(speeds)
cuts = np.quantile(speed, [0.25, 0.5, 0.75])
# Per-trial:
y[3] = np.searchsorted(speed_cuts, speed[frames], side="right").astype(np.int8)
```

iii. The AI computes global speed quartiles so that the 25% bins are consistent across sessions.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three threshold values at the 25th, 50th, and 75th percentiles of the global speed distribution. `searchsorted(..., side='right')` assigns speeds to bins 0-3.

ii.
```python
cuts = np.quantile(speed, [0.25, 0.5, 0.75])
classes = np.searchsorted(cuts, speed, side="right")
```

iii. Global thresholds ensure consistent meaning across sessions. The AI notes that the global quartile bins have counts differing by at most 1.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` gives one speed per imaging frame. Same frame indices are used.

ii.
```python
y[3] = np.searchsorted(speed_cuts, speed[frames], side="right").astype(np.int8)
```

iii. Alignment through shared frame indices.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: (1) behavior can extend past neural data, so streams are truncated to the shorter of neural/behavior frame counts; (2) non-finite `LickFr` values are filtered out; (3) non-finite `ft_trInd` values are handled; (4) trials with unmapped stimulus IDs are dropped; (5) trials with no valid running-corridor frames are dropped; (6) non-finite neural or input values raise errors.

ii.
```python
nfr = min(neural_nfr, len(beh["ft"]))
# ...
lick_frames = lick_frames[np.isfinite(lick_frames)].astype(np.int64)
# ...
np.isfinite(ft_trial)
# ...
if not np.isfinite(n).all():
    raise ValueError(...)
```

iii. The AI documents the systematic 1-3 frame offset between neural and behavior data and handles it by truncation, consistent with the reference approach.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the neural spike files (89 files totaling ~412 GB). The AI reports total conversion time of ~770s for all sessions.

ii.
```python
obj = np.load(spk_path(sid), allow_pickle=True).item()
```

iii. The AI implemented optimizations including file-size-based frame count inference to avoid loading neural files during the prepass.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The plane-by-plane trial construction loop iterates over imaging planes to fill the trial array. This could potentially be vectorized by pre-concatenating planes, though the AI chose this approach to avoid the memory cost of a full concatenated intermediate.

ii.
```python
for plane in planes:
    nr = plane.shape[0]
    n[row:row + nr] = plane[:, frames]
    row += nr
```

iii. The AI explicitly chose this design for memory efficiency over speed.

## 12-c. What processing does the code repeat multiple times?

i. The behavior files are loaded twice: once in `build_catalog()` to merge stimulus mappings and deduplicate sessions, and once in `compute_global_stats()` for the speed quartile prepass, and a third time during actual conversion. The global stats prepass also recomputes frame validity masks that are recomputed during conversion.

ii.
```python
# In build_catalog():
behavior = np.load(BEH / f"Beh_{exp_type}.npy", allow_pickle=True).item()
# In compute_global_stats():
behavior = np.load(BEH / f"Beh_{exp_type}.npy", allow_pickle=True).item()
# In convert():
current_behavior = np.load(BEH / f"Beh_{current_exp}.npy", allow_pickle=True).item()
```

iii. The behavior files are relatively small compared to neural data, so the redundancy has minimal impact.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes extensive metadata and validation statistics that are stored in the pickle but not used by the decoder. The `compute_global_stats` function also computes frame interval statistics and terminal frame offset distributions that are informational only.

ii.
```python
stats[f"terminal_frame_offset_{len(beh['ft']) - nfr}"] += 1
# ...
"full_dataset_prepass": raw_stats,
```

iii. This is for validation and documentation purposes rather than downstream analysis.
