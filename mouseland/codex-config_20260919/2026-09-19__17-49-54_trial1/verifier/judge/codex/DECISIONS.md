# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the master session index from `data/beh/Imaging_Exp_info.npy`, builds a deduplicated catalog of 89 physical sessions, reads each `Beh_<exp_type>.npy` behavior dictionary to recover per-session metadata and stimulus mappings, infers neural matrix shapes from retinotopy files plus spike-file size during a prepass, and later loads each session's actual spike file and retinotopy file during conversion.

ii. 
```python
exp_info = np.load(BEH / "Imaging_Exp_info.npy", allow_pickle=True).item()
```
```python
behavior = np.load(BEH / f"Beh_{exp_type}.npy", allow_pickle=True).item()
```
```python
with np.load(ret_path(sid), allow_pickle=True) as ret:
    nneurons = len(ret["iarea"])
payload = spk_path(sid).stat().st_size - OBJECT_NPY_OVERHEAD
```
```python
obj = np.load(spk_path(sid), allow_pickle=True).item()
with np.load(ret_path(sid), allow_pickle=True) as ret:
    iarea = np.asarray(ret["iarea"])
```

iii. In `CONVERSION_NOTES.md`, the agent says the dataset has repeated paper labels for the same physical recording, so it first deduplicates to 89 neural-file keys. It also justifies the behavior-only prepass as an efficiency optimization to avoid loading 412 GB of spike data merely to infer frame counts and speed quartiles.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the mouse prefix of the session id. The agent stores `entry["subject"] = sid.split("_")[0]`, then builds a sorted unique `subjects` list and a `subject_idx` lookup for converted sessions.

ii. 
```python
entries[sid] = {
    "session_id": sid,
    "subject": sid.split("_")[0],
    ...
}
```
```python
subjects = sorted({entry["subject"] for entry in catalog})
subject_lookup = {name: i for i, name in enumerate(subjects)}
```

iii. The notes describe the authoritative physical-recording session id as `<mouse>_<date>_<block>`, so the mouse prefix is the natural subject split. The agent reports 19 unique subjects after deduplication.

## 1-c. How are the data split into sessions?

i. A session is one unique neural recording keyed by `<mname>_<datexp>_<blk>`. The agent merges repeated paper-analysis labels that point to the same physical recording and keeps one session entry per unique session id.

ii. 
```python
sid = f"{rec['mname']}_{rec['datexp']}_{rec['blk']}"
record_meta[sid].append({"experiment_type": exp_type, **plain(rec)})
```
```python
if len(catalog) != 89 or len({e["session_id"] for e in catalog}) != 89:
    raise AssertionError(f"Expected 89 unique sessions, got {len(catalog)}")
```

iii. In the notes, the agent explicitly resolves the discrepancy between 142 experiment-label records and 89 physical recordings by treating one neural file as one session and using duplicate labels only to recover complementary metadata.

## 1-d. How are the data split into trials?

i. Trials are defined by `beh["ntrials"]` and `ft_trInd`, but the agent does not keep all corridor frames from each trial. Instead, for each trial it keeps only frames with a finite trial id, inside the textured corridor, and with `ft_move > 0`; those retained frame indices become the per-trial window.

ii. 
```python
valid = (
    np.isfinite(ft_trial)
    & np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool)
    & (np.asarray(beh["ft_move"][:nfr]) > 0)
)
```
```python
for tr in range(int(beh["ntrials"])):
    ...
    frames = np.flatnonzero(valid & (ft_trial == tr))
```

iii. The notes justify this as matching the paper's "active running in the 0-4 m textured corridor" regime rather than the broader traversal used by the human reference. The agent treats gray-space frames, stopped frames, and unlabeled-stimulus trials as outside the decoder observation window.

## 1-e. How are trials filtered based on quality controls?

i. A trial is dropped if its wall texture cannot be mapped to one of the agent's seven stimulus classes or if no frames survive the `finite trial id & ft_CorrSpc & ft_move>0` mask. Sessions with fewer than two surviving trials cause an error and are not retained.

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
```python
if len(neural) < 2:
    raise ValueError(f"Fewer than two converted trials: {sid}")
```

iii. The notes say there is no global trial deletion in the paper, but the agent adds decoder-specific exclusions for trials lacking "enough valid aligned corridor samples" and for trials without a mapped canonical stimulus id. It does not use the human reference's 99th-percentile trial-length filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` arrays come from the `spks` list inside each session's `<sid>_neural_data.npy`. Brain-region labels come from retinotopy `iarea`.

ii. 
```python
obj = np.load(spk_path(sid), allow_pickle=True).item()
planes = obj["spks"]
```
```python
with np.load(ret_path(sid), allow_pickle=True) as ret:
    iarea = np.asarray(ret["iarea"])
```

iii. The notes explicitly conclude that `spks` are already Suite2p deconvolved traces and that `iarea` is the region source of truth, so no raw fluorescence reconstruction is needed.

## 2-b. How is the `neural` data processed?

i. The agent keeps native deconvolved float32 values, does not compute dF/F, z-score, or temporally rebin them, and fills each trial matrix plane-by-plane instead of concatenating a full-session spike matrix first.

ii. 
```python
if any(a.ndim != 2 or a.dtype != np.float32 for a in planes):
    raise ValueError(f"Expected a list of float32 matrices in {sid}")
```
```python
n = np.empty((nneurons, T), dtype=np.float32)
row = 0
for plane in planes:
    nr = plane.shape[0]
    n[row:row + nr] = plane[:, frames]
    row += nr
```

iii. The notes justify this on two grounds: the paper's core analyses use the supplied deconvolved traces directly, and avoiding full-session concatenation reduces peak memory. The agent deliberately retains float32 rather than downcasting.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent does not remove neurons by retinotopy area. It keeps all neurons and maps them into five region labels: `V1`, `mHV`, `lHV`, `aHV`, and `unmapped/non-visual`.

ii. 
```python
def region_idx(iarea: np.ndarray) -> np.ndarray:
    out = np.full(len(iarea), 4, dtype=np.int8)
    out[iarea == 8] = 0
    out[np.isin(iarea, [0, 1, 2, 9])] = 1
    out[np.isin(iarea, [5, 6])] = 2
    out[np.isin(iarea, [3, 4])] = 3
    return out
```

iii. In Step 4 and Step 5 of the notes, the agent argues that the paper's load path keeps all Suite2p cells and that excluding `iarea=-1,7` is only needed for region-specific analyses. It therefore treats visual-area grouping as metadata rather than a neuron-quality filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent says the alignment event is corridor entry / trial start, but the actual retained neural samples are the trial's active-running textured-corridor frames. Thus trial-relative time is referenced to start, but stopped samples between trial start and later running frames may be absent.

ii. 
```python
valid = (
    np.isfinite(ft_trial)
    & np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool)
    & (np.asarray(beh["ft_move"][:nfr]) > 0)
)
```
```python
frames = np.flatnonzero(valid & (ft_trial == tr))
...
n[row:row + nr] = plane[:, frames]
```
```python
"temporal_alignment_event": (
    "trial start / entry into the 4-m textured corridor"
),
```

iii. The notes state that trial alignment is intentionally corridor entry, but also that "native active-running imaging samples in the textured 0-4 m corridor" are the kept samples and that explicit time inputs preserve gaps caused by excluding stopped frames.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent keeps native imaging samples without temporal rebinning. It records the nominal time bin as the median frame interval computed from timestamps across the full dataset, about 314.8 ms.

ii. 
```python
dt = np.diff(np.asarray(beh["ft"][:nfr], dtype=np.float64)) * 86_400_000
intervals.append(dt[np.isfinite(dt) & (dt > 0)])
...
nominal_ms = float(np.median(np.concatenate(intervals)))
```
```python
"time_bin_size": nominal_ms,
"time_bin_size_units": "ms (nominal median native imaging interval)",
```

iii. The notes say the paper's core selectivity uses original frames, so no temporal averaging is needed. The agent also notes ordinary timestamp jitter and therefore stores a nominal median interval rather than a hard-coded frame rate.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The agent derives time to cue from session frame timestamps `ft` and per-trial absolute cue timestamps `SoundTime`.

ii. 
```python
ft = np.asarray(beh["ft"][:nfr], dtype=np.float64)
...
x[0] = (float(beh["SoundTime"][tr]) - sample_time) * 86_400
```

iii. The notes explicitly choose `SoundTime` over `SoundFr`, arguing that the decoder needs a continuous time variable and that datenum timestamps provide that directly without frame interpolation.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained sample in a trial, the agent subtracts the sample's absolute frame time from the trial's absolute cue time and converts days to seconds. Positive values mean the cue is still in the future; negative values mean the cue has already occurred.

ii. 
```python
sample_time = ft[frames]
...
x[0] = (float(beh["SoundTime"][tr]) - sample_time) * 86_400
```
```python
"cue_time_sign_convention": (
    "positive before cue, zero at cue, negative after cue"
),
```

iii. Step 5 of the notes says this sign convention is the literal interpretation of "time to cue" and that continuous time is preferred over binary onset coding for this requested decoder input.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on exactly the same retained frame indices as the per-trial neural matrix.

ii. 
```python
frames = np.flatnonzero(valid & (ft_trial == tr))
sample_time = ft[frames]
...
n[row:row + nr] = plane[:, frames]
x[0] = (float(beh["SoundTime"][tr]) - sample_time) * 86_400
```

iii. The agent repeatedly states in the notes that all converted streams are frame-aligned and that the sanity checks verified `np.allclose` agreement between source files and converted arrays on selected trials.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The agent derives day of training from each session's calendar date, parsed from the session id, relative to that subject's earliest indexed imaging session.

ii. 
```python
def sid_date(sid: str) -> datetime:
    return datetime.strptime("_".join(sid.split("_")[1:4]), "%Y_%m_%d")
```
```python
first_date = {
    subject: min(
        sid_date(entry["session_id"])
        for entry in catalog if entry["subject"] == subject
    )
    for subject in subjects
}
```

iii. The notes reject using the paper's partial `days` or `sess#` fields because they are incomplete across session types. The agent presents elapsed calendar days as the only complete continuous training-day proxy.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The agent computes an integer number of elapsed calendar days from the subject's first indexed session, casts it to float, and broadcasts that same value across all retained samples of the trial.

ii. 
```python
day = float((sid_date(entry["session_id"]) - first_date[entry["subject"]]).days)
```
```python
x[1] = day
```
```python
"day_of_training_definition": (
    "elapsed calendar days from each subject's earliest indexed "
    "imaging session"
),
```

iii. The notes frame this as a complete per-session continuous measure that avoids fabricating missing values for sessions where explicit training-day annotations are absent.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. The agent derives time since trial start from absolute imaging timestamps `ft` and the per-trial absolute start timestamp `Trial_start_time`.

ii. 
```python
ft = np.asarray(beh["ft"][:nfr], dtype=np.float64)
...
x[2] = (sample_time - float(beh["Trial_start_time"][tr])) * 86_400
```

iii. The notes describe trial start as corridor entry and use the absolute timestamp field rather than interpolating `StartFr` onto a relative time axis.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each retained sample, the agent subtracts the absolute trial-start timestamp from the sample's absolute frame timestamp and converts from days to seconds.

ii. 
```python
sample_time = ft[frames]
...
x[2] = (sample_time - float(beh["Trial_start_time"][tr])) * 86_400
```

iii. The notes justify this as preserving exact wall-clock gaps even when stopped frames are excluded from the kept neural samples.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is evaluated on the same retained frame indices as the neural data for that trial.

ii. 
```python
frames = np.flatnonzero(valid & (ft_trial == tr))
sample_time = ft[frames]
...
n[row:row + nr] = plane[:, frames]
x[2] = (sample_time - float(beh["Trial_start_time"][tr])) * 86_400
```

iii. The notes state that explicit time inputs are what preserve the trial-start reference after stopped frames are removed from the trial window.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Reward availability is derived directly from `isRew`.

ii. 
```python
x[3] = float(bool(beh["isRew"][tr]))
```

iii. The notes interpret `isRew` as corridor-level reward availability, not reward delivery, and use it unchanged.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The agent converts the per-trial reward flag to binary `0/1` and broadcasts it across all retained samples in the trial.

ii. 
```python
x[3] = float(bool(beh["isRew"][tr]))
```

iii. The notes say no additional processing is needed because unrewarded cohorts legitimately have zero reward-available trials.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The agent derives stimulus labels from a merged mapping between `UniqWalls` and `stim_id`, then applies that mapping to each trial's `WallName`. It does not use the wall names alone as the category definition.

ii. 
```python
for wall, stim in zip(beh["UniqWalls"], beh["stim_id"]):
    if not np.isfinite(stim):
        continue
    wall, stim = str(wall), int(stim)
    ...
    mappings[sid][wall] = stim
```
```python
category = wall_map.get(str(beh["WallName"][tr]))
```

iii. The notes argue that reused paper labels provide complementary `stim_id` views for the same physical session, so the agent merges them to recover a seven-class canonical label set. Trials whose wall name still has no finite `stim_id` are dropped rather than inferred.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The agent defines seven stimulus categories (`circle1`, `circle2`, `leaf1`, `leaf2`, `leaf3`, `leaf1_swap1`, `leaf1_swap2`), looks up the mapped integer class for each trial, and broadcasts that class across all retained samples of the trial.

ii. 
```python
STIMULI = [
    "circle1", "circle2", "leaf1", "leaf2", "leaf3",
    "leaf1_swap1", "leaf1_swap2",
]
```
```python
y = np.empty((4, T), dtype=np.int8)
y[0] = category
```

iii. In the notes, the agent explicitly rejects collapsing walls to the broader circle/leaf/rock/wood families used by the human reference. It views the canonical `stim_id` labels as the safer "source of truth" and excludes 309 trials with no mapped label.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `LickFr`.

ii. 
```python
lick = np.zeros(nfr, dtype=np.int8)
lick_frames = np.asarray(beh["LickFr"], dtype=np.float64)
lick_frames = lick_frames[np.isfinite(lick_frames)].astype(np.int64)
lick_frames = lick_frames[(lick_frames >= 0) & (lick_frames < nfr)]
lick[np.unique(lick_frames)] = 1
```

iii. The notes describe `LickFr` as already indexed to imaging frames, so the conversion only needs to convert it into a binary per-frame flag.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The agent filters out non-finite and out-of-range lick-frame values, truncates remaining frame indices to integers, deduplicates repeated licks within the same frame, and marks those frames as `1`.

ii. 
```python
lick_frames = np.asarray(beh["LickFr"], dtype=np.float64)
lick_frames = lick_frames[np.isfinite(lick_frames)].astype(np.int64)
lick_frames = lick_frames[(lick_frames >= 0) & (lick_frames < nfr)]
lick[np.unique(lick_frames)] = 1
```

iii. The notes justify this as a defensive version of the reference's frame-binning of licks: multiple licks in one imaging frame remain a single positive binary label.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The licking trace is sampled on the same retained frame indices as the neural data for each trial.

ii. 
```python
y[1] = lick[frames]
```

iii. The notes say the imaging-frame indexing of `LickFr` already places licking on the neural frame grid, so selecting the same trial `frames` aligns them.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is derived from `ft_Pos`.

ii. 
```python
pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float64)
...
y[2] = np.floor(pos[frames] / 10).astype(np.int8)
```

iii. The notes identify `ft_Pos` as the per-frame decimeter position and use it directly on the retained imaging samples.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The agent divides decimeter positions by 10, floors to integer meter bins, and stores the result as categorical indices 0-3.

ii. 
```python
y[2] = np.floor(pos[frames] / 10).astype(np.int8)
if np.any((y[2] < 0) | (y[2] > 3)):
    raise ValueError(f"Position outside 0-4 m: {sid} trial {tr}")
```

iii. The notes justify this by noting that the retained samples already satisfy the corridor mask, so valid positions should lie entirely within the 0-4 m textured corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The four categories are 1-meter bins over the 4-meter textured corridor: `[0,1)`, `[1,2)`, `[2,3)`, and `[3,4)` meters, encoded as integers 0-3.

ii. 
```python
OUTPUT_VALUES = [
    STIMULI,
    ["not_licking", "licking"],
    ["0-1_m", "1-2_m", "2-3_m", "3-4_m"],
    ["Q1_slowest", "Q2", "Q3", "Q4_fastest"],
]
```
```python
y[2] = np.floor(pos[frames] / 10).astype(np.int8)
```

iii. The notes explicitly say the task asks for four equal 1-m bins, so the conversion uses the decimeter grid to realize those categories directly.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is indexed on the same retained frame indices as the neural data for each trial.

ii. 
```python
y[2] = np.floor(pos[frames] / 10).astype(np.int8)
```

iii. The notes treat `ft_Pos` as already frame-aligned to imaging data, so alignment is achieved by using the identical `frames` array.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running-speed categories are derived from `ft_RunSpeed`.

ii. 
```python
speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float64)
```

iii. The notes identify `ft_RunSpeed` as the per-frame running-speed source and use no alternate derived signal.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The agent performs a full-dataset prepass, collects running speeds from all frames that satisfy its validity mask, computes global quartile thresholds, and then bins each retained sample by threshold with `searchsorted(..., side="right")`.

ii. 
```python
valid = (
    labeled
    & np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool)
    & (np.asarray(beh["ft_move"][:nfr]) > 0)
)
speeds.append(np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float64)[valid])
...
cuts = np.quantile(speed, [0.25, 0.5, 0.75])
```
```python
y[3] = np.searchsorted(speed_cuts, speed[frames], side="right").astype(np.int8)
```

iii. The notes defend this as matching the decoder task wording "25% of the data" globally. They also report the resulting thresholds and class counts from the prepass.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The thresholds are the three global quartile cutpoints of the retained speed samples across the full dataset, and each sample is assigned to quartile 0-3 by threshold comparison.

ii. 
```python
cuts = np.quantile(speed, [0.25, 0.5, 0.75])
```
```python
y[3] = np.searchsorted(speed_cuts, speed[frames], side="right").astype(np.int8)
```

iii. The notes record the thresholds as approximately `12.3986`, `25.2848`, and `40.7527` cm/s and emphasize that the bins are globally balanced rather than session-balanced.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running-speed labels are sampled on the same retained frame indices as the neural data.

ii. 
```python
y[3] = np.searchsorted(speed_cuts, speed[frames], side="right").astype(np.int8)
```

iii. The notes state that `ft_RunSpeed` is an imaging-frame stream, so selecting the identical `frames` indices aligns it to the neural matrix.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent truncates all frame-wise behavior arrays to the common neural/behavior prefix, filters non-finite trial ids and lick timestamps, discards out-of-range lick frames, drops trials with no retained valid samples or no mapped stimulus label, and raises errors on non-finite neural/input values or out-of-range position labels.

ii. 
```python
nfr = min(neural_nfr, len(beh["ft"]))
```
```python
valid = (
    np.isfinite(ft_trial)
    & np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool)
    & (np.asarray(beh["ft_move"][:nfr]) > 0)
)
```
```python
lick_frames = lick_frames[np.isfinite(lick_frames)].astype(np.int64)
lick_frames = lick_frames[(lick_frames >= 0) & (lick_frames < nfr)]
```

iii. The notes describe the source as generally clean but still add defensive checks because of frame-count mismatches, occasional non-finite values, and genuinely unmapped stimulus labels. The agent treats those cases as grounds for exclusion rather than imputation.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading large spike files session by session, copying their per-trial slices into output arrays, and writing the final enormous pickle. The agent added a behavior-only prepass precisely because full neural loading is so expensive.

ii. 
```python
obj = np.load(spk_path(sid), allow_pickle=True).item()
planes = obj["spks"]
```
```python
for plane in planes:
    nr = plane.shape[0]
    n[row:row + nr] = plane[:, frames]
    row += nr
```
```python
with outpath.open("wb") as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. In Step 6, Step 7, and Step 9 of the notes, the agent repeatedly says that naive spike loading would dominate memory/I/O cost and reports per-session payload/timing plus a 151.82 s pickle-write time for the full dataset.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still loops over every trial and, inside each kept trial, loops over every imaging plane to copy neural slices. The per-trial frame search is also repeated instead of grouping frames by trial in one pass.

ii. 
```python
for tr in range(int(beh["ntrials"])):
    ...
    frames = np.flatnonzero(valid & (ft_trial == tr))
```
```python
for plane in planes:
    nr = plane.shape[0]
    n[row:row + nr] = plane[:, frames]
    row += nr
```

iii. The notes emphasize memory reduction over full vectorization. They explicitly avoid whole-session concatenation, so the remaining trial/plane loops are a deliberate tradeoff rather than an oversight.

## 12-c. What processing does the code repeat multiple times?

i. The agent rereads behavior files across several phases: once to build the deduplicated catalog and stimulus mappings, once again in the full-dataset prepass for speed quartiles and timing statistics, and once more during actual conversion. It also reparses session dates and regenerates some bookkeeping in multiple passes.

ii. 
```python
for exp_type in exp_info:
    behavior = np.load(BEH / f"Beh_{exp_type}.npy", allow_pickle=True).item()
```
```python
for exp_type, entries in grouped(catalog).items():
    behavior = np.load(BEH / f"Beh_{exp_type}.npy", allow_pickle=True).item()
```
```python
if current_exp != entry["source_experiment"]:
    current_behavior = np.load(
        BEH / f"Beh_{current_exp}.npy", allow_pickle=True
    ).item()
```

iii. The notes acknowledge some of this repetition and justify it as cheaper than repeatedly loading spike files; the behavior-only prepass is presented as an intentional speed/memory tradeoff.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The conversion computes extra prepass statistics and metadata that are not needed by the downstream decoder itself, including speed min/max, class counts, terminal frame-offset counts, nominal interval estimates, and extensive per-session provenance fields.

ii. 
```python
stats_out = {
    **dict(stats),
    "speed_min": float(speed.min()),
    "speed_max": float(speed.max()),
    "speed_class_counts": np.bincount(classes, minlength=4).tolist(),
}
```
```python
"full_dataset_prepass": raw_stats,
"session_info": session_info,
```

iii. The notes present these as auditability and validation aids rather than decoder inputs. They support the agent's internal checks but are not used by `train_decoder.py` for learning.
