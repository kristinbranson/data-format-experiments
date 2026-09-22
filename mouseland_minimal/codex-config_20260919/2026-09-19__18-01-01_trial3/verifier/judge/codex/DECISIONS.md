# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent uses `Imaging_Exp_info.npy` as the inventory, deduplicates references into physical recordings identified by mouse/date/block, loads each behavior file at most once, and loads the corresponding retinotopy and spike files per recording. It processes all 89 unique recordings.

ii.
```python
exp_info = np.load(data_root / "beh" / "Imaging_Exp_info.npy", allow_pickle=True).item()
pid = physical_id(row)
raw = np.load(data_root / "beh" / f"Beh_{exp_type}.npy", allow_pickle=True).item()
raw_neural = np.load(spk_path, allow_pickle=True).item()
iarea = np.load(retino_path, allow_pickle=True)["iarea"]
```

iii. The trajectory says there are 89 physical recordings but 142 metadata references because recordings are reused by paper analyses; each recording should therefore appear once. Behavior files are cached/released to control memory.

## 1-b. How are the data split into subjects?

i. A subject is the `mname` component of each physical recording. Subjects are accumulated in first-seen session order, and each session receives its index in that list.

ii.
```python
mouse, date, block = pid
if mouse not in subjects:
    subjects.append(mouse)
subject_idx.append(subjects.index(mouse))
```

iii. The trajectory identified 19 mice and treated the explicit mouse name as the subject identifier.

## 1-c. How are the data split into sessions?

i. A session is the unique tuple `(mname, datexp, blk)`. Multiple analysis-group references to that tuple are merged, preserving the first inventory occurrence; exactly 89 sessions are required.

ii.
```python
def physical_id(row):
    return row["mname"], row["datexp"], str(row["blk"])
if pid not in references:
    references[pid] = []
    sessions.append(pid)
references[pid].append((exp_type, row))
```

iii. The agent reasoned that duplicated metadata entries share one neural recording and should not become duplicate sessions.

## 1-d. How are the data split into trials?

i. The agent trusts `ntrials` and creates one trial for each integer index. Rather than selecting that trial’s recorded frames, it creates 40 spatial targets at cumulative positions `trial*60 + 0..39`, representing the 4 m textured corridor.

ii.
```python
targets = (np.arange(behavior["ntrials"])[:, None] * FULL_TRIAL_BINS
           + np.arange(CORRIDOR_BINS)[None, :]).ravel()
session_neural = [cube[trial] for trial in range(behavior["ntrials"])]
```

iii. The trajectory states that the paper’s running-only 0.1 m interpolation was chosen to match released processing and exclude the 2 m gray interval.

## 1-e. How are trials filtered based on quality controls?

i. No trials are filtered. Every declared trial is retained, including unusually long/stationary trials, because interpolation always emits 40 samples.

ii.
```python
cube = np.empty((behavior["ntrials"], nneurons, CORRIDOR_BINS), dtype=np.float32)
for trial in range(behavior["ntrials"]):
    ...
```

iii. The agent explicitly said “all trials” and that no arbitrary trial subsampling was performed; the trajectory reports 38,110 retained trials and no quality exceptions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from each plane’s `spks` array. `ft_move` and `ft_PosCum` supply the samples and coordinate used for interpolation; `iarea` supplies neuron inclusion and region labels.

ii.
```python
planes = raw_neural["spks"]
moving = behavior["ft_move"][:nframes] > 0
x = behavior["ft_PosCum"][:nframes][moving]
keep, regions = area_indices(iarea)
```

iii. The agent identified `spks` as the already deconvolved Suite2p signal and selected the authors’ position-processing variables to reproduce `spk_pos_interp`.

## 2-b. How is the `neural` data processed?

i. Only moving frames are used. For each neuron, deconvolved activity is linearly interpolated/extrapolated over cumulative VR position onto 40 positions per trial. Processing is chunked across retained neurons and stored as float32.

ii.
```python
y = plane[ids][:, moving]
values = y[:, lo] * (1.0 - weight) + y[:, hi] * weight
cube[:, dest_offset:dest_offset + count, :] = values.reshape(
    count, behavior["ntrials"], CORRIDOR_BINS).transpose(1, 0, 2)
```

iii. The trajectory says this matches the paper helper numerically and preserves the authors’ running-only linear interpolation, while producing a fixed aligned grid.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons outside the four visual-cortex groupings are removed. `iarea` labels are mapped exactly to V1, mHV, lHV, and aHV; labels -1 and 7 are excluded. No further neuron subsampling is done.

ii.
```python
region[iarea == 8] = 0
region[np.isin(iarea, [0, 1, 2, 9])] = 1
region[np.isin(iarea, [5, 6])] = 2
region[np.isin(iarea, [3, 4])] = 3
keep = region >= 0
```

iii. The agent says released neurons were already Suite2p-selected and that no paper-supported additional subsampling exists.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to textured-corridor entry by making its first spatial target position 0 and retaining positions 0–3.9 m. Every trial has 40 bins and an asserted end offset of 40/6 seconds.

ii.
```python
targets = trial_index * FULL_TRIAL_BINS + np.arange(CORRIDOR_BINS)
"temporal_alignment_event": "entry into the 4 m textured corridor (trial start)",
"off_start": 0.0,
"off_end": CORRIDOR_BINS / VR_SPEED_DM_S,
```

iii. The agent treated fixed-speed VR distance as time and argued this gives uniform samples aligned to entry while excluding gray space.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Metadata declares 166.67 ms (`1000/6`) per sample. The source imaging is 3.17 Hz, but neural activity is resampled spatially to 0.1 m bins and those bins are interpreted as temporal samples using 6 dm/s.

ii.
```python
VR_SPEED_DM_S = 6.0
CORRIDOR_BINS = 40
"time_bin_size": 1000.0 / VR_SPEED_DM_S,
```

iii. The trajectory states that 0.1 m at a claimed fixed 0.6 m/s equals 166.67 ms and directly matches paper preprocessing.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundDelPos` and the synthetic spatial positions 0–39, not from frame timestamps or `SoundFr`.

ii.
```python
cue_dm = float(np.mod(behavior["SoundDelPos"][trial], FULL_TRIAL_BINS))
positions = np.arange(CORRIDOR_BINS, dtype=np.float32)
```

iii. The agent chose the spatial cue field because every converted sample lies on the same spatial grid.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Cue position is reduced modulo the 60-bin full-trial coordinate, the current position is subtracted, and the distance is divided by 6 dm/s. Values are positive before and negative after the cue.

ii.
```python
cue_dm = float(np.mod(behavior["SoundDelPos"][trial], FULL_TRIAL_BINS))
inp[0] = (cue_dm - positions) / VR_SPEED_DM_S
```

iii. The justification is the assumed constant 0.6 m/s VR advance, allowing distance-to-cue to stand in for elapsed time-to-cue.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Both have 40 entries indexed by the identical interpolated corridor positions.

ii.
```python
inp = np.empty((4, CORRIDOR_BINS), dtype=np.float32)
cube = np.empty((behavior["ntrials"], nneurons, CORRIDOR_BINS), dtype=np.float32)
```

iii. The agent considered all streams aligned by the common 0.1 m spatial samples.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It uses metadata fields `days` when present, otherwise the minimum `sess#` among references to the physical recording; absent annotations become 0.

ii.
```python
explicit = [row["days"] for _, row in references if "days" in row]
sessions = [row["sess#"] for _, row in references if "sess#" in row]
return float(min(sessions)) if sessions else 0.0
```

iii. The agent said explicit release annotations should be preferred, and the minimum avoids relabeling duplicated pre-exposure references as later sessions.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The selected scalar is cast to float and broadcast across all 40 samples in every trial of that session.

ii.
```python
day = day_value(references[pid])
inp[1] = day
```

iii. The trajectory describes this as a per-trial/session covariate using explicit annotations where available.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is not derived from recorded timestamps. It uses the synthetic positions `0..39` and the constant `VR_SPEED_DM_S=6`.

ii.
```python
positions = np.arange(CORRIDOR_BINS, dtype=np.float32)
elapsed_s = positions / VR_SPEED_DM_S
```

iii. The agent inferred time from spatial displacement under fixed VR speed.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Each 0.1 m position index is divided by 6 dm/s, yielding 0 through 6.5 seconds, and copied to each trial.

ii.
```python
elapsed_s = positions / VR_SPEED_DM_S
inp[2] = elapsed_s
```

iii. The claimed constant corridor speed is the sole conversion justification.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Entry corresponds to spatial bin zero; each later value shares the corresponding interpolated neural position bin.

ii.
```python
inp[2] = elapsed_s
session_neural = [cube[trial] for trial in range(behavior["ntrials"])]
```

iii. The common 40-bin spatial axis is treated as a temporal axis.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes directly from per-trial `isRew`.

ii.
```python
"isRew": np.asarray(d["isRew"], dtype=bool)
inp[3] = float(behavior["isRew"][trial])
```

iii. The agent interprets this as whether the current corridor makes reward available.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The Boolean is converted to float 0/1 and broadcast across all 40 trial samples.

ii.
```python
inp[3] = float(behavior["isRew"][trial])
```

iii. No transformation beyond categorical encoding and broadcasting was considered necessary.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from each trial’s `WallName`.

ii.
```python
"WallName": np.asarray(d["WallName"]),
out[0] = visual_category(behavior["WallName"][trial])
```

iii. The trajectory notes that `WallName` remains usable in swap sessions and all observed exemplar names map cleanly.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Lowercased names are collapsed by prefix into circle, leaf, rock, or brick; raw `wood` is labeled brick. The class index is broadcast across all trial samples.

ii.
```python
if name.startswith("circle"): return 0
if name.startswith("leaf"): return 1
if name.startswith("rock"): return 2
if name.startswith("wood") or name.startswith("brick"): return 3
out[0] = visual_category(behavior["WallName"][trial])
```

iii. The agent states exemplar suffixes and spatial swaps should collapse to four texture families and calls the paper’s raw wood family “brick.”

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It uses `LickTrind` for trial identity and `LickPos` for spatial position, rather than frame-indexed `LickFr`.

ii.
```python
"LickTrind": np.asarray(d["LickTrind"]).astype(np.int64),
"LickPos": np.asarray(d["LickPos"]),
```

iii. The agent chose spatial lick fields to match the resampled 0.1 m neural grid.

## 8-b. What processing is involved in computing `output` *Licking*?

i. `LickPos` is floored to an integer spatial bin. Valid trial/bin pairs set a binary 40-bin matrix to one; multiple licks in a bin remain one.

ii.
```python
lick_bin = np.floor(behavior["LickPos"]).astype(np.int64)
valid = (behavior["LickTrind"] >= 0) & (lick_bin >= 0) & (lick_bin < CORRIDOR_BINS)
lick[behavior["LickTrind"][valid], lick_bin[valid]] = 1
```

iii. Metadata explicitly defines a sample as one when one or more licks occurred in its 0.1 m bin.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks and neural activity use the same trial index and 40 corridor-position bins.

ii.
```python
out[1] = lick[trial]
neural.append(session_neural)
```

iii. The agent regarded spatial binning as direct alignment to the interpolated neural product.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is generated from the known output grid itself (`np.arange(40)`), not read from `ft_Pos`.

ii.
```python
position_class = (np.arange(CORRIDOR_BINS) // 10).astype(np.int16)
```

iii. Since neural samples were placed at fixed 0.1 m targets, their corridor position is known by construction.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The 40 spatial samples are divided into four contiguous groups of ten samples, corresponding to four 1 m intervals.

ii.
```python
position_class = (np.arange(CORRIDOR_BINS) // 10).astype(np.int16)
out[2] = position_class
```

iii. This directly implements the requested four equal 1 m corridor categories.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Integer division by 10 gives class 0 for bins 0–9, class 1 for 10–19, class 2 for 20–29, and class 3 for 30–39.

ii.
```python
position_class = (np.arange(CORRIDOR_BINS) // 10).astype(np.int16)
```

iii. Each source spatial unit is one decimeter, so ten consecutive bins equal one meter.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. It is indexed by the same fixed spatial target axis used to interpolate neural activity, identically for every trial.

ii.
```python
targets = trial_offset + np.arange(CORRIDOR_BINS)
out[2] = position_class
```

iii. Position is exact by construction on the spatially resampled grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses `run_pos`, truncated to the first 40 spatial bins of every trial.

ii.
```python
"run_pos": np.asarray(d["run_pos"], dtype=np.float32)[:, :CORRIDOR_BINS],
```

iii. The trajectory selected the paper’s already position-binned running trajectory to accompany spatially interpolated neural activity.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. All retained `run_pos` values across all 89 sessions are concatenated, global 25th/50th/75th percentiles are computed, and each value is assigned with `np.digitize`.

ii.
```python
all_speeds = np.concatenate([behaviors[pid]["run_pos"].ravel() for pid in sessions])
speed_edges = np.quantile(all_speeds, [0.25, 0.50, 0.75]).astype(np.float32)
out[3] = np.digitize(behavior["run_pos"][trial], speed_edges)
```

iii. The agent aimed for four globally balanced quartile classes; verification reportedly showed exactly 25% per class.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three global quantile values are used as threshold edges. `np.digitize` maps speeds below/among/above those edges to classes 0–3.

ii.
```python
speed_edges = np.quantile(all_speeds, [0.25, 0.50, 0.75])
np.digitize(behavior["run_pos"][trial], speed_edges)
```

iii. The requested bins each correspond to 25% of the full converted data, so global quantiles were chosen.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Both `run_pos[trial]` and neural activity have 40 position-indexed samples, and are paired by sample index.

ii.
```python
out[3] = np.digitize(behavior["run_pos"][trial], speed_edges)
session_neural = [cube[trial] for trial in range(behavior["ntrials"])]
```

iii. The agent uses the paper’s position-binned behavior on the same nominal 0.1 m grid as the interpolated neural activity.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code performs strict consistency checks and aborts on missing behavior records, inconsistent neural frame counts, short behavior streams, non-monotonic moving-position coordinates, neuron/retinotopy mismatches, non-finite speed edges, or unexpected session count. Invalid lick indices/positions are silently excluded. There is no repair or trial-level fallback.

ii.
```python
if missing: raise KeyError(f"No behavior record found for {missing}")
if np.any(np.diff(x) <= 0): raise ValueError(...)
valid = (behavior["LickTrind"] >= 0) & ... & (lick_bin < CORRIDOR_BINS)
```

iii. The trajectory reports that no data-quality exceptions occurred and verification passed, so the agent favored fail-fast validation over imputation.

## 12-a. What are the most time-consuming steps of the code?

i. Loading hundreds of GB of spike arrays, interpolating all retained neurons across all trials, and serializing/reloading the 273.77 GB pickle dominate. Decoder verification and SVD setup also require repeated full-data scans.

ii.
```python
raw_neural = np.load(spk_path, allow_pickle=True).item()
values = y[:, lo] * (1.0 - weight) + y[:, hi] * weight
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory records a ~274 GB build, long serialization/reload phases, and repeated decoder scans before SVD/training.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Most heavy interpolation is already vectorized across targets and neuron chunks. The per-trial covariate loop, per-plane/chunk loops, linear subject lookup, and conversion of the cube into a list of trial views remain; the trial loop could be vectorized into session tensors before list conversion.

ii.
```python
for plane in planes:
    for start in range(0, len(plane_keep), chunk_neurons):
        ...
for trial in range(behavior["ntrials"]):
    ...
```

iii. The agent specifically emphasized vectorizing interpolation across neurons and chunking for memory; it did not discuss the smaller Python loops as bottlenecks.

## 12-c. What processing does the code repeat multiple times?

i. Per-trial arrays are allocated and constant elapsed-time/position vectors are copied repeatedly; `visual_category` is re-run per trial. Garbage collection is invoked after behavior files and every session. Downstream verification and training then reload and rescan the huge pickle multiple times.

ii.
```python
for trial in range(behavior["ntrials"]):
    inp = np.empty((4, CORRIDOR_BINS), dtype=np.float32)
    out = np.empty((4, CORRIDOR_BINS), dtype=np.int16)
gc.collect()
```

iii. The trajectory recognized repeated full-payload passes in the supplied decoder but did not identify avoidable repetition inside conversion.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `Corridor_Length` and `Texture_Length` are extracted but never used. The complete `references` lists and analysis-group sets are retained mainly for metadata/day selection. More importantly, the conversion materializes a massive fixed spatial neural cube and duplicates constant per-trial covariates across 40 samples; the spatial resampling itself is unnecessary for the reference frame-aligned decoder task.

ii.
```python
"Corridor_Length": float(d["Corridor_Length"]),
"Texture_Length": float(d["Texture_Length"]),
inp[1] = day
out[0] = visual_category(behavior["WallName"][trial])
```

iii. The agent considered the interpolation essential paper processing and accepted the ~274 GB result, so it did not characterize these operations as unnecessary.
