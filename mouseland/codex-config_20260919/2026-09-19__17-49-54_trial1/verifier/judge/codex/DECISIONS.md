# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent builds a catalog from `beh/Imaging_Exp_info.npy`, then opens every `Beh_<experiment>.npy` to deduplicate 89 physical session IDs and merge stimulus mappings. During conversion it loads behavior by experiment group, each session's object-NPY `spks`, and its retinotopy `iarea`. A prepass infers neural shapes from retinotopy length and neural-file byte size.

ii.
```python
exp_info = np.load(BEH / "Imaging_Exp_info.npy", allow_pickle=True).item()
behavior = np.load(BEH / f"Beh_{exp_type}.npy", allow_pickle=True).item()
obj = np.load(spk_path(sid), allow_pickle=True).item()
with np.load(ret_path(sid), allow_pickle=True) as ret:
    iarea = np.asarray(ret["iarea"])
```

iii. The notes say physical neural recordings, rather than repeated paper-analysis labels, are the conversion unit. Grouped behavior loading and file-size shape inference avoid repeated behavior reads and an extra 412 GB neural prepass.

## 1-b. How are the data split into subjects?

i. Subject is the mouse-name prefix of the physical session ID. The full subject vocabulary is sorted, and every converted session receives its integer index.

ii.
```python
"subject": sid.split("_")[0]
subjects = sorted({entry["subject"] for entry in catalog})
subject_idx.append(subject_lookup[entry["subject"]])
```

iii. The mouse name is explicitly encoded in the source record/session ID; the full run yields 19 subjects.

## 1-c. How are the data split into sessions?

i. A physical session is `mouse_date_block`, obtained from behavior keys/index records. Duplicate appearances under paper experiment labels are merged, producing exactly 89 sessions; the first encountered behavior view supplies the primary source while all views contribute stimulus mappings.

ii.
```python
sid = f"{rec['mname']}_{rec['datexp']}_{rec['blk']}"
sid = sid_from_key(key)
if sid not in entries:
    entries[sid] = {...}
if len(catalog) != 89:
    raise AssertionError(...)
```

iii. The notes justify deduplication because 142 paper-label records refer to only 89 physical neural/retinotopy recordings.

## 1-d. How are the data split into trials?

i. The code loops over `ntrials`; a trial's samples are frames whose finite `ft_trInd` equals the trial index and also pass corridor and movement masks. Trial matrices remain variable length.

ii.
```python
for tr in range(int(beh["ntrials"])):
    frames = np.flatnonzero(valid & (ft_trial == tr))
```

iii. The agent states that frame-indexed behavior is the canonical neural alignment and that the paper's analyses use active samples in the textured 0–4 m corridor.

## 1-e. How are trials filtered based on quality controls?

i. Trials lacking a merged canonical stimulus ID or any finite, moving, textured-corridor frame are dropped. Sessions with fewer than two converted trials fail. No trial-length outlier filter is used.

ii.
```python
if category is None:
    dropped["unmapped_stimulus"] += 1; continue
if not len(frames):
    dropped["no_valid_running_corridor_frames"] += 1; continue
if len(neural) < 2:
    raise ValueError(...)
```

iii. The notes call paper train/test and selectivity filters analysis-specific, decline to guess missing labels, and report 309 label-based exclusions and no trials without valid running frames.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from every plane in the session neural object's `spks` list. Per-neuron region indices come from retinotopy `iarea`.

ii.
```python
planes = obj["spks"]
with np.load(ret_path(sid), allow_pickle=True) as ret:
    iarea = np.asarray(ret["iarea"])
```

iii. The source already contains Suite2p deconvolved traces, so the agent treats these as the neural signal directly.

## 2-b. How is the `neural` data processed?

i. For each trial, plane rows are copied in source order at selected frames into a float32 neuron-by-time array. There is no dF/F recomputation, normalization, averaging, padding, or resampling.

ii.
```python
n = np.empty((nneurons, T), dtype=np.float32)
for plane in planes:
    n[row:row + nr] = plane[:, frames]
```

iii. The notes say supplied traces are already deconvolved and native sampling avoids unnecessary signal loss; plane-wise filling avoids a full-session concatenation copy.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are removed. `iarea` values are assigned to V1, mHV, lHV, aHV, or a fifth `unmapped/non-visual` category. Neural time samples are filtered to finite trial IDs, `ft_CorrSpc`, and `ft_move > 0`.

ii.
```python
out = np.full(len(iarea), 4, dtype=np.int8)
out[iarea == 8] = 0
...
valid = np.isfinite(ft_trial) & beh_corridor & (beh_move > 0)
```

iii. The agent argues there is no global Suite2p quality filter in reference loading and retaining all cells preserves paper population counts; unmapped cells remain explicitly identifiable.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned conceptually to corridor entry, but retain only each trial's moving textured-corridor frames. Thus arrays begin at the first retained running sample and can contain real-time gaps where stationary frames were removed; they are variable length and not padded.

ii.
```python
frames = np.flatnonzero(valid & (ft_trial == tr))
n[row:row + nr] = plane[:, frames]
"temporal_alignment_event": "trial start / entry into the 4-m textured corridor"
```

iii. The notes say explicit timestamp-derived inputs preserve gaps and that the downstream decoder supports variable trial lengths.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native imaging samples are retained without rebinning. Metadata uses the median positive raw frame interval over the full dataset, approximately 314.804 ms.

ii.
```python
dt = np.diff(np.asarray(beh["ft"][:nfr])) * 86_400_000
nominal_ms = float(np.median(np.concatenate(intervals)))
"time_bin_size": nominal_ms
```

iii. The agent chose the finest available temporal grid and notes that removed stopped frames make the retained samples non-contiguous despite a nominal bin size.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundTime` and frame timestamps `ft`.

ii.
```python
sample_time = ft[frames]
x[0] = (float(beh["SoundTime"][tr]) - sample_time) * 86_400
```

iii. The notes prefer exact datenum timestamps for temporal alignment.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Cue timestamp minus sample timestamp is converted from MATLAB days to seconds, making values positive before and negative after the cue.

ii.
```python
x[0] = (float(beh["SoundTime"][tr]) - sample_time) * 86_400
```

iii. The sign convention is explicitly documented in metadata and was visually/independently audited.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It uses `ft[frames]` for exactly the same frame indices copied into the neural matrix.

ii.
```python
sample_time = ft[frames]
n[row:row + nr] = plane[:, frames]
```

iii. The agent reports raw-source spot checks and zero-crossing plots confirming common-frame alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the date embedded in each session ID and the earliest indexed imaging date for that subject.

ii.
```python
first_date = {subject: min(sid_date(entry["session_id"]) ...)}
day = float((sid_date(entry["session_id"]) - first_date[entry["subject"]]).days)
```

iii. The notes describe this as a calendar-day proxy because no universal paper training-day field exists.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The elapsed integer calendar days from the subject's first imaging session are converted to float and broadcast over every retained trial sample.

ii.
```python
x[1] = day
```

iii. The definition is documented in metadata; the full range is reported as 0–92 days.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from per-trial `Trial_start_time` and frame timestamps `ft`.

ii.
```python
x[2] = (sample_time - float(beh["Trial_start_time"][tr])) * 86_400
```

iii. Exact source timestamps were selected to preserve real elapsed time through excluded pauses.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Trial-start timestamp is subtracted from each retained frame timestamp and MATLAB days are converted to seconds.

ii.
```python
x[2] = (sample_time - float(beh["Trial_start_time"][tr])) * 86_400
```

iii. The notes verify nonnegative, monotonic trial time and explain large values as genuine pauses.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is calculated at the identical `frames` indices as the neural columns.

ii.
```python
sample_time = ft[frames]
n[row:row + nr] = plane[:, frames]
```

iii. The common global imaging-frame index is the asserted alignment authority.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes from per-trial `isRew`.

ii.
```python
x[3] = float(bool(beh["isRew"][tr]))
```

iii. The notes identify `isRew` as the direct rewarded-corridor flag.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The value is coerced to Boolean/float 0 or 1 and broadcast across the trial.

ii.
```python
x[3] = float(bool(beh["isRew"][tr]))
```

iii. This preserves the raw per-trial task context without further transformation.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Categories are built by merging every behavior view's `UniqWalls`/`stim_id` mapping for a physical session and applying it to each trial's `WallName`.

ii.
```python
for wall, stim in zip(beh["UniqWalls"], beh["stim_id"]):
    mappings[sid][str(wall)] = int(stim)
category = wall_map.get(str(beh["WallName"][tr]))
```

iii. The agent says merged views recover complementary authoritative labels without guessing NaN IDs.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Finite source IDs are encoded as seven categories (`circle1`, `circle2`, `leaf1`, `leaf2`, `leaf3`, `leaf1_swap1`, `leaf1_swap2`) and broadcast over the trial. Unmapped trials are excluded.

ii.
```python
STIMULI = ["circle1", "circle2", "leaf1", "leaf2", "leaf3",
           "leaf1_swap1", "leaf1_swap2"]
y[0] = category
```

iii. The notes call these paper-canonical IDs and reject inferring labels for 309 genuinely unlabeled trials.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It comes from the session's `LickFr` event frame numbers.

ii.
```python
lick_frames = np.asarray(beh["LickFr"], dtype=np.float64)
```

iii. The agent follows the reference's frame-index event representation.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Finite lick frame numbers are truncated to integers, bounds-checked, uniqued, and marked 1 in an otherwise-zero frame vector.

ii.
```python
lick_frames = lick_frames[np.isfinite(lick_frames)].astype(np.int64)
lick_frames = lick_frames[(lick_frames >= 0) & (lick_frames < nfr)]
lick[np.unique(lick_frames)] = 1
```

iii. This matches the reference integer-frame convention while safely handling invalid or repeated events.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The frame-level lick vector is indexed by the same retained frames as neural activity.

ii.
```python
y[1] = lick[frames]
n[row:row + nr] = plane[:, frames]
```

iii. Independent raw reconstructions reportedly matched exactly.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It comes from frame-wise `ft_Pos`, expressed in decimeters.

ii.
```python
pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float64)
```

iii. The textured-corridor mask restricts eligible samples to the task's 0–4 m region.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Raw position is read at retained native frames and divided into requested 1 m categories; no spatial interpolation is performed.

ii.
```python
y[2] = np.floor(pos[frames] / 10).astype(np.int8)
```

iii. The notes say paper spatial interpolation is inappropriate for a temporally aligned decoder, so native frames are preserved.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Floor division by 10 dm yields classes 0–3, corresponding to 0–1, 1–2, 2–3, and 3–4 m; out-of-range results raise an error.

ii.
```python
y[2] = np.floor(pos[frames] / 10).astype(np.int8)
if np.any((y[2] < 0) | (y[2] > 3)):
    raise ValueError(...)
```

iii. This directly implements four equal-length, one-meter bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is sampled at the same global frame indices as the neural columns.

ii.
```python
pos[frames]
plane[:, frames]
```

iii. The notes report source-level exact output comparisons and review a small number of source acquisition-boundary wraps without modifying alignment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from frame-wise `ft_RunSpeed` over all samples eligible for conversion.

ii.
```python
speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float64)
speeds.append(speed[valid])
```

iii. The raw speed stream is already imaging-frame aligned.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A full-dataset prepass concatenates eligible speeds and computes three global value quantiles. Each retained sample is classified with those thresholds.

ii.
```python
cuts = np.quantile(speed, [0.25, 0.5, 0.75])
y[3] = np.searchsorted(speed_cuts, speed[frames], side="right")
```

iii. The agent interprets “each corresponding to 25% of the data” globally and reports nearly exact global balance.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Thresholds are the global 25th, 50th, and 75th percentiles (12.3986, 25.2848, 40.7527 cm/s); `side="right"` assigns ties at a cut to the upper class.

ii.
```python
cuts = np.quantile(speed, [0.25, 0.5, 0.75])
np.searchsorted(speed_cuts, speed[frames], side="right")
```

iii. The thresholds and class counts are stored in metadata for reproducibility.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is selected at precisely the neural trial's `frames` indices.

ii.
```python
y[3] = np.searchsorted(speed_cuts, speed[frames], side="right")
n[row:row + nr] = plane[:, frames]
```

iii. The common imaging-frame grid is the alignment basis.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural and behavior streams are truncated to their common length; finite/in-range checks protect trial IDs, licks, timestamps, labels, inputs, and neural values. Missing stimulus mappings cause trial exclusion, inconsistent shapes/labels raise errors, and repeated physical sessions are deduplicated.

ii.
```python
nfr = min(neural_nfr, len(beh["ft"]))
valid = np.isfinite(ft_trial) & ...
if not np.isfinite(n).all(): raise ValueError(...)
if category is None: continue
```

iii. The notes document systematic 1–3 terminal-frame mismatches, refuse to invent missing categories, and emphasize assertions and independent raw spot checks.

## 12-a. What are the most time-consuming steps of the code?

i. Loading roughly 412 GB of neural object files, copying selected plane data into per-trial arrays, serializing the 172.6 GB pickle, and the full-dataset decoder training dominate runtime.

ii.
```python
obj = np.load(spk_path(sid), allow_pickle=True).item()
n[row:row + nr] = plane[:, frames]
pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes report about 769 s conversion plus 152 s pickle writing and identify neural I/O/copying as the primary cost.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial scan `valid & (ft_trial == tr)` rescans every session frame for every trial; frames could be grouped once by trial. Plane copying necessarily remains chunked but could also use precomputed trial-index groups.

ii.
```python
for tr in range(int(beh["ntrials"])):
    frames = np.flatnonzero(valid & (ft_trial == tr))
    for plane in planes:
        n[row:row + nr] = plane[:, frames]
```

iii. The agent focused its optimization discussion on avoiding much larger I/O and concatenation costs; it did not explicitly propose vectorizing this smaller loop.

## 12-c. What processing does the code repeat multiple times?

i. Behavior files are read once while building the catalog, again in the global-statistics prepass, and again during conversion. Eligible masks, stimulus categories, and speed data are recomputed in the prepass and conversion. Trial masks rescan frame arrays once per trial.

ii.
```python
behavior = np.load(BEH / f"Beh_{exp_type}.npy", ...).item()  # catalog/prepass
current_behavior = np.load(BEH / f"Beh_{current_exp}.npy", ...).item()  # conversion
```

iii. The notes accept repeated behavior processing to keep memory bounded while avoiding the vastly more expensive repeated neural load; behavior is reused within each experiment group.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It constructs extensive session provenance, raw prepass statistics, five-region mappings (including unmapped cells), and optionally ten-panel plots; the trainer chiefly consumes neural/input/output arrays and basic labels. `infer_shape` file-layout work and merged record metadata are validation/optimization aids rather than decoder features.

ii.
```python
"experiment_records": entry["records"],
"full_dataset_prepass": raw_stats,
if show_plot:
    processing_plot(...)
```

iii. The agent intentionally retains these for auditability, source reconstruction, sanity checks, and performance diagnostics, even though they do not improve decoder fitting directly.
