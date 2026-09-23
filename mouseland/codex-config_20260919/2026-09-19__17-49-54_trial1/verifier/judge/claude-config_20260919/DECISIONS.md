# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything is read from `/app/data`, from the three subfolders `beh/` (behavior), `spk/` (deconvolved traces) and `retinotopy/` (visual area per neuron). `beh/Imaging_Exp_info.npy` is opened first and gives the 23 experiment types; the 23 `Beh_<exp_type>.npy` files are then read one at a time and **every key** in each is turned into a catalog entry. A session id is the first five underscore-separated fields of the behavior key (`mouse_year_month_day_block`); the first behavior view of an id wins, later ones only contribute stimulus labels. The catalog is asserted to contain exactly 89 unique sessions. Per session, the spike file `spk/<sid>_neural_data.npy` and the retinotopy `retinotopy/<mouse>_<date>_trans.npz` are read at conversion time. There is an extra behavior-only prepass (`compute_global_stats`) that re-reads all behavior files and infers the neural frame count from the *file size* of the spike file rather than loading it, so the 412 GB of traces is read only once.

ii.
```python
exp_info = np.load(BEH / "Imaging_Exp_info.npy", allow_pickle=True).item()
...
for exp_type in exp_info:
    behavior = np.load(BEH / f"Beh_{exp_type}.npy", allow_pickle=True).item()
    for key, beh in behavior.items():
        sid = sid_from_key(key)
        if sid not in entries:
            entries[sid] = {...}
```
```python
def sid_from_key(key: str) -> str:
    parts = key.split("_")
    return "_".join(parts[:5])
```
```python
obj = np.load(spk_path(sid), allow_pickle=True).item()
planes = obj["spks"]
with np.load(ret_path(sid), allow_pickle=True) as ret:
    iarea = np.asarray(ret["iarea"])
```
```python
def infer_shape(sid: str) -> tuple[int, int]:
    """Infer exact neuron/frame counts without loading a multi-GB neural object."""
    with np.load(ret_path(sid), allow_pickle=True) as ret:
        nneurons = len(ret["iarea"])
    payload = spk_path(sid).stat().st_size - OBJECT_NPY_OVERHEAD
    divisor = nneurons * np.dtype(np.float32).itemsize
    return int(nneurons), int(payload // divisor)
```

iii. From CONVERSION_NOTES Step 4/5: "One target session per physical neural-file key. Do not duplicate recordings reused under several analysis labels or the two swap views." The file-size trick is justified as a speedup: "the invariant scalar-NPY payload layout [lets it] infer neural frame counts from file size during the behavior-only prepass", avoiding "loading all 412 GB just to obtain frame counts"; the constant was "validated over all 89 scalar object-NPY neural files" and a `% divisor != 0` check aborts if the layout assumption ever fails.

## 1-b. How are the data split into subjects?

i. The subject is the first field of the session id (the mouse name). `subjects` is the sorted set of unique names (19 mice) and `subject_idx` is each session's index into that list. Sessions per subject range 1–8.

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
...
subject_idx.append(subject_lookup[entry["subject"]])
```

iii. From Step 2/5: the index and the file names already name the mouse, so no split has to be derived; "Sorted unique mouse IDs and integer session mapping. 19 subjects." This reproduces the paper's "89 recordings in 19 mice".

## 1-c. How are the data split into sessions?

i. A session is one physical recording = one mouse / one date / one block, i.e. one `spk/*_neural_data.npy` file. `Imaging_Exp_info.npy` lists 142 label records for only 89 physical recordings (a recording is re-listed under several analysis labels, and swap recordings appear under two `_swap1`/`_swap2` behavior keys). The converter deduplicates by session id, keeping the first behavior view it sees as `source_behavior_key`, and asserts the result is exactly 89. The other views are not discarded silently — their `UniqWalls`/`stim_id` pairs are merged into one per-session wall→stimulus map.

ii.
```python
if sid not in entries:
    entries[sid] = {... "source_experiment": exp_type, "source_behavior_key": key ...}
if exp_type not in entries[sid]["experiment_labels"]:
    entries[sid]["experiment_labels"].append(exp_type)
for wall, stim in zip(beh["UniqWalls"], beh["stim_id"]):
    if not np.isfinite(stim):
        continue
    wall, stim = str(wall), int(stim)
    old = mappings[sid].get(wall)
    if old is not None and old != stim:
        raise ValueError(f"Conflicting stimulus ID: {sid} {wall}")
    mappings[sid][wall] = stim
...
if len(catalog) != 89 or len({e["session_id"] for e in catalog}) != 89:
    raise AssertionError(f"Expected 89 unique sessions, got {len(catalog)}")
```

iii. Step 4: "Experiment index contains 142 label records / 89 neural files and 89 retinotopy files / paper says 89 recordings in 19 mice → One target session per physical neural-file key." Step 5 Key Decision 1: "A session is a neural file, not a paper-analysis label. Reused labels are merged only to recover canonical stimulus IDs." I verified that the two swap views of a session carry byte-identical behavior apart from which swap texture has a finite `stim_id`, so merging the labels is safe and the choice of view is immaterial.

## 1-d. How are the data split into trials?

i. Trials are the ones the data declares (`ntrials`, with every imaging frame labelled by `ft_trInd`). Each trial keeps the frames that satisfy **three** conditions: a finite trial index, `ft_CorrSpc` (inside the 0–4 m textured corridor), **and `ft_move > 0` (the animal/VR is actually moving)**. Trials are variable length and nothing is padded. The third condition is the substantive departure from the reference, which keeps every corridor frame of a trial. I measured its effect on the raw data: it removes 551,591 of 1,373,170 corridor frames (40.2%); 52.0% of trials end up with at least one internal gap and 24.1% have a gap longer than 10 frames (>3 s), so for half of all trials the columns of the neural matrix are not consecutive imaging frames even though the metadata declares a single 314.8 ms bin size.

ii.
```python
ft_trial = np.asarray(beh["ft_trInd"][:nfr])
valid = (
    np.isfinite(ft_trial)
    & np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool)
    & (np.asarray(beh["ft_move"][:nfr]) > 0)
)
...
for tr in range(int(beh["ntrials"])):
    ...
    frames = np.flatnonzero(valid & (ft_trial == tr))
```

iii. Step 4: "Reference `ft_CorrSpc & (ft_move>0)` for selectivity ... Only running timepoints in 0–4 m are analyzed → Eligible decoder samples are textured-corridor, running samples. Pauses/reward collection and 2-m gray intervals are invalid for this task." Step 5 Key Decision 4: "This exactly matches the paper's neural validity regime and makes the four requested 1-m position classes exhaustive." Step 7 acknowledges the consequence: "Long wall-clock gaps occur when a task mouse pauses — the intervening stopped frames are intentionally excluded per the paper, while the explicit time inputs preserve the gap."

## 1-e. How are trials filtered based on quality controls?

i. Three filters, applied per trial: (1) the trial's `WallName` must have a finite canonical `stim_id` in the merged per-session map — 309 trials of 38,110 fail and are dropped; (2) the trial must retain at least one valid running-corridor frame — zero trials failed; (3) a session must yield ≥2 trials, otherwise the script raises (no session failed). No trial-length outlier filter is applied; the `ft_move>0` frame filter already caps trial length at 178 retained frames. 37,801 trials survive.

ii.
```python
for tr in range(int(beh["ntrials"])):
    category = wall_map.get(str(beh["WallName"][tr]))
    if category is None:
        dropped["unmapped_stimulus"] += 1
        continue
    frames = np.flatnonzero(valid & (ft_trial == tr))
    if not len(frames):
        dropped["no_valid_running_corridor_frames"] += 1
        continue
...
if len(neural) < 2:
    raise ValueError(f"Fewer than two converted trials: {sid}")
```

iii. Step 4: "No global acquisition-level trial deletion [is described] ... Do not impose selectivity-analysis train/test splits. Exclude only trials lacking a canonical output label or enough valid aligned corridor samples." Step 5: "exclude the 390 [later corrected to 309] trials whose stimulus remains genuinely unlabeled; do not infer a paper category for NaN IDs." Step 9 records the difference from the raw 38,110 as an "Explained task-required difference".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy`, a list of one `(neurons, frames)` float32 array per imaging plane, taken in list order (the same order `load_spk` concatenates in). The per-neuron visual area comes from `iarea` in `retinotopy/<mouse>_<date>_trans.npz`, and its length is asserted to equal the total neuron count.

ii.
```python
obj = np.load(spk_path(sid), allow_pickle=True).item()
if set(obj) != {"spks"} or not obj["spks"]:
    raise ValueError(f"Unexpected neural object in {sid}")
planes = obj["spks"]
nneurons = sum(a.shape[0] for a in planes)
with np.load(ret_path(sid), allow_pickle=True) as ret:
    iarea = np.asarray(ret["iarea"])
if len(iarea) != nneurons:
    raise AssertionError(f"Retinotopy/neural mismatch in {sid}")
```

iii. Step 1/4: "`load_spk` ... takes its `spks` list, and concatenates planes/cell groups along the neuron axis"; "Use supplied `spks` without dF/F or additional deconvolution"; "Retinotopy length is a safe pre-load count and concatenated `spks` is authoritative neural order."

## 2-b. How is the `neural` data processed?

i. Not processed at all: no dF/F, no deconvolution, no z-scoring, no smoothing, no temporal binning. The retained columns of each plane are copied into a pre-allocated `(n_neurons, T)` **float32** array in plane order, and a finiteness check is run. Unlike the reference (which stores float16) the source dtype is preserved, which together with retaining every neuron produces a 172.6 GB pickle.

ii.
```python
T = len(frames)
n = np.empty((nneurons, T), dtype=np.float32)
row = 0
for plane in planes:
    nr = plane.shape[0]
    n[row:row + nr] = plane[:, frames]
    row += nr
if not np.isfinite(n).all():
    raise ValueError(f"Non-finite neural values: {sid} trial {tr}")
```

iii. Step 5 Key Decision 6: "Raw supplied deconvolved values match the core paper analysis and allow the decoder's learned session projection to choose scaling. Analysis-specific z-scoring used only for selected figure panels is not generalized." Step 6: "Naively calling the reference `load_spk` would concatenate the complete multi-plane recording and then copy it again into trials ... Trial arrays are allocated once and filled plane-by-plane without a full concatenated intermediate; source and target remain float32."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron is ever dropped. All 4,691,034 Suite2p cells are kept. The reference retinotopy grouping is applied only as a *label*: `iarea==8` → V1, `{0,1,2,9}` → mHV, `{5,6}` → lHV, `{3,4}` → aHV, and everything else (`-1`, `7`; 585,641 cells) → a fifth region `unmapped/non-visual`. The human reference instead drops those 585,641 cells and keeps only the four visual areas.

ii.
```python
REGIONS = ["V1", "mHV", "lHV", "aHV", "unmapped/non-visual"]

def region_idx(iarea: np.ndarray) -> np.ndarray:
    out = np.full(len(iarea), 4, dtype=np.int8)
    out[iarea == 8] = 0
    out[np.isin(iarea, [0, 1, 2, 9])] = 1
    out[np.isin(iarea, [5, 6])] = 2
    out[np.isin(iarea, [3, 4])] = 3
    return out
```

iii. Step 4: "Recorded-neuron range counts the full Suite2p population ... Preserve all recorded cells to avoid losing neural data. Use exactly the four reference groups and assign -1/7 to a fifth `unmapped/non-visual` group; downstream users can reproduce the paper mask by excluding that group." Step 3: "Suite2p cell classification was already applied upstream. The paper does not describe an additional global cell-quality threshold." Keeping all cells reproduces the paper's reported 20,547–89,577 neurons/recording exactly.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry (trial start). Each trial simply consists of its own retained frames in acquisition order, so it begins at the first *running* corridor frame of that traversal and ends where the traversal ends; lengths vary (11–178 retained frames, mean 22.25). Nothing is cut to a common window and nothing is padded; `off_start` is declared 0.0 and `off_end` `None`. I checked how far the first retained frame sits from the recorded `Trial_start_time`: median 0.18 s, 99th percentile 0.35 s, i.e. within about one imaging frame, so requiring `ft_move>0` almost never truncates the start of a traversal (rare exception: 28.8 s max).

ii.
```python
frames = np.flatnonzero(valid & (ft_trial == tr))
...
n[row:row + nr] = plane[:, frames]
...
"temporal_alignment_event": ("trial start / entry into the 4-m textured corridor"),
"off_start": 0.0,
"off_end": None,
```

iii. Step 5 Key Decision 5: "Retaining every valid native frame avoids padding, NaNs, or truncating slow/paused trials. The temporal alignment remains corridor entry through the explicit time-since-start input." The format only requires a common bin *size*, and the decoder reads each trial's length from its own array.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning, resampling or spatial interpolation. The imaging frames are the bins. The bin size written to metadata is measured from the data: the median positive `diff(ft)` over every session, 314.804 ms (~3.18 Hz), matching the reference's 1000/3.17 = 315.5 ms.

ii.
```python
dt = np.diff(np.asarray(beh["ft"][:nfr], dtype=np.float64)) * 86_400_000
intervals.append(dt[np.isfinite(dt) & (dt > 0)])
...
nominal_ms = float(np.median(np.concatenate(intervals)))
...
"time_bin_size": nominal_ms,
"time_bin_size_units": "ms (nominal median native imaging interval)",
```

iii. Step 5 Key Decision 3: "The paper explicitly uses original deconvolved frames for core selectivity. No temporal averaging is needed; trial lengths may vary and the supplied decoder supports this. Metadata will call this the nominal acquisition bin and include observed interval percentiles because timestamps have ordinary acquisition jitter." Step 4 notes the paper's 0.1-m spatial interpolation is inappropriate here because "the requested trial-start temporal alignment overrides spatial resampling."

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime`, the MATLAB datenum timestamp of the sound cue on each trial, and `ft`, the datenum timestamp of every imaging frame. (The reference instead interpolates the fractional cue *frame* `SoundFr` onto the frame-time axis; I checked both on a session and they agree to 0.0 s, so the two sources are numerically identical.)

ii.
```python
sample_time = ft[frames]
x[0] = (float(beh["SoundTime"][tr]) - sample_time) * 86_400
```

iii. Step 5 mapping table: "`SoundTime`, `ft` → `input[0]`: signed seconds **to** cue: `(SoundTime[trial] - ft[frame]) * 86400` ... Continuous as explicitly required, rather than binary cue onset." Step 3 notes the cue "is present in all imaging trial types."

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. A datenum difference converted to seconds (`× 86400`), evaluated at every retained frame of the trial, stored float32. Sign convention: positive before the cue, zero at the cue, negative after — the literal "time *to* cue", same as the reference. Values are asserted finite. Because stopped frames are removed but the timestamps are real, the range runs to [-1763.3, 723.5] s, versus [-72.2, 73.4] s for the reference, which keeps the pauses as frames instead.

ii.
```python
x = np.empty((4, T), dtype=np.float32)
x[0] = (float(beh["SoundTime"][tr]) - sample_time) * 86_400
...
if not np.isfinite(x).all():
    raise ValueError(f"Non-finite input values: {sid} trial {tr}")
```
```python
"cue_time_sign_convention": ("positive before cue, zero at cue, negative after cue"),
```

iii. Step 5 Key Decision 7: "Positive means cue is in the future, zero at cue, negative after; this is the literal 'time to cue' interpretation and is documented in metadata."

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated on `ft[frames]` — exactly the same frame indices used to slice the neural columns of that trial — so it is sample-for-sample aligned by construction, and the trial's `T` is checked to match.

ii.
```python
sample_time = ft[frames]
...
for n, x, y in zip(neural, inputs, outputs):
    if n.shape[1] != x.shape[1] or n.shape[1] != y.shape[1]:
        raise AssertionError(f"Time dimension mismatch: {sid}")
```

iii. All streams in this dataset are indexed by global imaging frame, so using one `frames` index vector for every stream guarantees alignment. The `--show-processing` audit plots the cue-time zero crossing against trial time to confirm it visually.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the recording date embedded in the session id (`datexp`, i.e. `mouse_YYYY_MM_DD_blk`), together with the subject, parsed with `datetime.strptime`.

ii.
```python
def sid_date(sid: str) -> datetime:
    return datetime.strptime("_".join(sid.split("_")[1:4]), "%Y_%m_%d")

first_date = {
    subject: min(sid_date(entry["session_id"])
                 for entry in catalog if entry["subject"] == subject)
    for subject in subjects
}
```

iii. Step 4: "Code mostly uses analysis labels, `sess#`, and occasional `days` ... `days` exists only for some train-2 endpoints and `sess#` is ordinal ... Use elapsed calendar days from each subject's first indexed imaging session as the only complete continuous day measure ... This avoids fabricating missing training-duration values." The original `sess#`/`days`/experiment labels are still stored per session in `metadata.session_info` for auditability.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The elapsed number of **calendar days** between a session's date and the earliest indexed session of the same mouse, computed over the whole 89-session catalog (so it is identical in `--sample` and `--full` runs), cast to float and broadcast across every bin of every trial of that session. Range [0, 92]. The human reference instead counts *recorded sessions* per mouse, giving an ordinal 0–7.

ii.
```python
day = float((sid_date(entry["session_id"]) - first_date[entry["subject"]]).days)
...
x[1] = day
```

iii. Step 5 Key Decision 8: "Calendar-day elapsed time is complete for naive/task/unsupervised/grating sessions, unlike `days` and `sess#`. It is per-session/per-trial and does not invent exact unrecorded training duration."

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time`, the datenum timestamp of corridor entry for each trial, and `ft`. (The reference interpolates the fractional `StartFr` onto the frame-time axis; I verified the two agree to 0.0 s.)

ii.
```python
x[2] = (sample_time - float(beh["Trial_start_time"][tr])) * 86_400
```

iii. Step 5 mapping table: "`Trial_start_time`, `ft` → `input[2]`: Seconds since corridor entry ... Temporal alignment is corridor entry; minor interpolation offsets are preserved rather than rounded."

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. A datenum difference converted to seconds, per retained frame, float32, positive after entry (and never negative, since only in-corridor frames are kept). The global minimum is 0.0 s; the maximum is 1765.2 s, again because retained frames can straddle long pauses that were themselves removed.

ii.
```python
sample_time = ft[frames]
x[2] = (sample_time - float(beh["Trial_start_time"][tr])) * 86_400
```

iii. Same as 5-a; Step 10's edge-case audit explicitly asserts "nonnegative trial time" and "constant cue time relative to trial start within each trial."

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same `frames` index vector as the neural columns, so it is aligned sample-for-sample; the per-trial `T` equality check covers it.

ii.
```python
frames = np.flatnonzero(valid & (ft_trial == tr))
sample_time = ft[frames]
n[row:row + nr] = plane[:, frames]
```

iii. Every stream is indexed by global imaging frame, so one index vector aligns them all.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, the per-trial boolean marking trials run in the rewarded corridor.

ii.
```python
x[3] = float(bool(beh["isRew"][tr]))
```

iii. Step 5 mapping table: "`isRew` → `input[3]`: Trial value 1 for rewarded corridor, 0 otherwise, tiled across frames. This is reward **availability**, not whether reward was delivered."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. None beyond a bool→float cast and broadcast across the trial's bins. It is 0 for all trials of the naive and unsupervised mice (the raw data has 4,336 rewarded trials, 11.4% of 38,110, concentrated in 28 sessions).

ii.
```python
x[3] = float(bool(beh["isRew"][tr]))
```
```python
"has_reward": bool(np.any(beh["isRew"])),
```

iii. No processing is needed; the flag is already per trial. The `has_reward` flag is used only to make `--sample` pick one rewarded and one unrewarded session "so all binary task variables and lick classes are exercised" (Step 5 Key Decision 9).

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From the trial's `WallName`, mapped through a per-session `UniqWalls → stim_id` table that is merged across every behavior view of that session. `stim_id` is the reference notebook's canonical id (`0:circle1, 1:circle2, 2:leaf1, 3:leaf2, 4:leaf3, 5:leaf1_swap1, 6:leaf1_swap2`), so the label is the paper's *stimulus role code* rather than the literal texture. The human reference derives the label from `WallName` alone via a hard-coded texture table.

ii.
```python
for wall, stim in zip(beh["UniqWalls"], beh["stim_id"]):
    if not np.isfinite(stim):
        continue
    mappings[sid][str(wall)] = int(stim)
...
category = wall_map.get(str(beh["WallName"][tr]))
if category is None:
    dropped["unmapped_stimulus"] += 1
    continue
...
y[0] = category
```
```python
STIMULI = ["circle1", "circle2", "leaf1", "leaf2", "leaf3",
           "leaf1_swap1", "leaf1_swap2"]
```

iii. Step 4: "Code defines canonical IDs 0-6 ... Different physical texture pairs are described generically as circle/leaf roles → Merge all label views per physical session, map by canonical `stim_id`, and exclude the 390 trials whose stimulus remains genuinely unlabeled; do not infer a paper category for NaN IDs." Step 2 also records the countervailing observation the agent chose not to act on: "`stim_id` assigns paper-level category IDs but differs between paper contrasts for reused sessions; literal `WallName` remains the acquisition-level source of truth."

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The wall name of each trial is replaced by its session's canonical `stim_id` (0–6) and broadcast across the trial's bins as int8; trials whose wall has no finite id anywhere are dropped (309 trials). Frame-fraction distribution is [0.319, 0.060, 0.335, 0.170, 0.059, 0.027, 0.029]. Because `stim_id` is a role, the classes are not physically homogeneous — checking the raw mapping across all 89 sessions, class 0 is `circle1` in 62 sessions, `rock1` in 19 and `leaf1` in 8; class 2 is `leaf1` in 62 sessions, `wood1` in 19, `circle1` in 6 and `rock1` in 2; class 3 mixes `leaf2`/`wood2`/`circle2`; class 4 mixes `leaf3`/`wood5`/`rock1`/`circle3`. So the same physical texture (`leaf1`) carries two different labels depending on the session, and physically different textures share a label, while the `output_values` strings still name the classes "circle1", "leaf1", etc. The reference instead collapses the 15 wall names to their four base textures (circle/leaf/rock/wood), which are physically consistent across sessions.

ii.
```python
OUTPUT_VALUES = [
    STIMULI,
    ["not_licking", "licking"],
    ["0-1_m", "1-2_m", "2-3_m", "3-4_m"],
    ["Q1_slowest", "Q2", "Q3", "Q4_fastest"],
]
...
y = np.empty((4, T), dtype=np.int8)
y[0] = category
```
```python
"stimulus_id_definition": dict(enumerate(STIMULI)),
```

iii. Step 5: "Canonical class 0-6, tiled across frames. Merge complementary paper-label views for the same physical recording ... do not guess labels." The agent's stated rationale is fidelity to the reference code's canonical id list; it did observe rock/wood sessions carrying `stim_id` 0/2 but treated the ids as the paper's category definition.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the (fractional) imaging-frame number of every lick in the session.

ii.
```python
lick = np.zeros(nfr, dtype=np.int8)
lick_frames = np.asarray(beh["LickFr"], dtype=np.float64)
lick_frames = lick_frames[np.isfinite(lick_frames)].astype(np.int64)
lick_frames = lick_frames[(lick_frames >= 0) & (lick_frames < nfr)]
lick[np.unique(lick_frames)] = 1
```

iii. Step 5 mapping table: "`LickFr` → `output[1]`: Binary per retained imaging frame; event frames use `LickFr.astype(int)` exactly as cue/first-lick reference functions do. Multiple events in a frame remain 1." This is the same truncation convention as the reference.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A session-length binary vector is built once: a frame is 1 if at least one lick falls in it, 0 otherwise; fractional frame numbers are truncated; non-finite entries and entries outside `[0, nfr)` are discarded (the behavior can run past the imaging). The trial's values are then the vector indexed by its retained frames, stored int8. Resulting distribution 0.963 / 0.037, matching the reference's 0.959 / 0.041.

ii.
```python
lick[np.unique(lick_frames)] = 1
...
y[1] = lick[frames]
```

iii. Step 3/5: licking is only recorded in the reward sessions (61 non-rewarded sessions have no licks at all), and the reference's own alignment functions truncate `LickFr` to integer frames, so no further processing is appropriate.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` indexes imaging frames, so the flag is already on the neural grid; the trial takes it with the same `frames` vector used for the neural columns, giving identical `T`.

ii.
```python
n[row:row + nr] = plane[:, frames]
y[1] = lick[frames]
```

iii. All streams share the global imaging-frame index. The audit plot overlays the lick binary on trial time for a rewarded session.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the within-corridor position at each imaging frame, in decimetres (0–40 across the texture, continuing to 60 through the grey space), truncated to the imaged frames.

ii.
```python
pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float64)
...
y[2] = np.floor(pos[frames] / 10).astype(np.int8)
```

iii. Step 2: "Position is recorded in decimeters: `Texture_Length=40` (4 m corridor) and `Corridor_Length=60`." Step 5: "`ft_Pos` + `ft_CorrSpc` → `output[2]`."

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Decimetres → metres by integer division by 10, giving the four 1-m classes; the result is validated to lie in 0–3 rather than clipped (the corridor mask already guarantees `ft_Pos < 40`). Resulting distribution [0.250, 0.249, 0.250, 0.252], essentially identical to the reference's [0.254, 0.242, 0.247, 0.257].

ii.
```python
y[2] = np.floor(pos[frames] / 10).astype(np.int8)
if np.any((y[2] < 0) | (y[2] > 3)):
    raise ValueError(f"Position outside 0-4 m: {sid} trial {tr}")
```

iii. Step 5: "Four equal 1-m bins: 0-1, 1-2, 2-3, 3-4 m", following the Decoder Task specification and the paper's 0–4 m textured-corridor convention.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Fixed, equal-width 1-m thresholds at 0/1/2/3/4 m — i.e. `floor(ft_Pos/10)` — exactly as the task specifies ("4 equal-length, 1-m-long spatial bins"). Named `0-1_m`, `1-2_m`, `2-3_m`, `3-4_m`. No data-driven thresholds are used.

ii.
```python
OUTPUT_VALUES = [..., ["0-1_m", "1-2_m", "2-3_m", "3-4_m"], ...]
y[2] = np.floor(pos[frames] / 10).astype(np.int8)
```

iii. Because only textured-corridor frames are retained, the four fixed bins are exhaustive and near-uniformly occupied, which the agent used as a sanity check ("position classes are near-balanced as expected from constant-speed VR traversal").

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` gives one value per imaging frame; the trial takes it with the same `frames` vector as the neural columns.

ii.
```python
n[row:row + nr] = plane[:, frames]
y[2] = np.floor(pos[frames] / 10).astype(np.int8)
```

iii. Shared frame indexing. The audit found and explained 21 trials (of 37,801) with a single 3→0 bin transition at an acquisition boundary, where `ft_Pos` has already wrapped while `ft_trInd` still assigns the frame to the previous trial — the agent chose to keep the reference's trial assignment rather than reassign the frame.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed (cm/s) at each imaging frame, truncated to the imaged frames.

ii.
```python
speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float64)
...
speeds.append(np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float64)[valid])
```

iii. Step 5 mapping table: "`ft_RunSpeed` → `output[3]`: Global quartile thresholds over every retained frame."

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A prepass (`compute_global_stats`) collects `ft_RunSpeed` over exactly the frames that will be converted — every session, valid trial index, `ft_CorrSpc`, `ft_move>0`, mapped stimulus — and takes the 25/50/75% quantiles of that pooled distribution: 12.3986 / 25.2848 / 40.7527 cm/s. Each frame's class is then `searchsorted(cuts, speed, 'right')`. The thresholds are computed over the whole dataset even in `--sample` mode, so sample and full runs agree. Resulting distribution is [0.250, 0.250, 0.250, 0.250] (counts 203877/203876/203876/203877). The reference instead ranks within each session, giving quartiles that are exact per session.

ii.
```python
speed = np.concatenate(speeds)
cuts = np.quantile(speed, [0.25, 0.5, 0.75])
classes = np.searchsorted(cuts, speed, side="right")
```
```python
y[3] = np.searchsorted(speed_cuts, speed[frames], side="right").astype(np.int8)
```

iii. Step 5 mapping table gives the planned thresholds and notes "counts differ by at most one". The global pooling is what makes the four classes each hold 25% "of the data" as the Decoder Task asks. Because stationary frames are excluded upstream, the large mass of exactly-zero speeds that the reference had to break by rank ordering is not present, so value thresholds give an essentially exact split.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three global value thresholds (12.40, 25.28, 40.75 cm/s) applied identically to every session and trial, `side='right'`, labelled `Q1_slowest … Q4_fastest`; the thresholds are also written to metadata. This is a dataset-wide fixed threshold rather than the reference's per-session rank split.

ii.
```python
"running_speed_units": "cm/s",
"running_speed_quartile_thresholds": cuts.tolist(),
...
OUTPUT_VALUES = [..., ["Q1_slowest", "Q2", "Q3", "Q4_fastest"]]
```

iii. Step 10: the converted speed distribution is asserted to be exactly [0.25, 0.25, 0.25, 0.25] with "speed class counts differing by at most one", which is the literal requirement "4 bins, each corresponding to 25% of the data".

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is one value per imaging frame; the trial takes it with the same `frames` vector as the neural columns.

ii.
```python
n[row:row + nr] = plane[:, frames]
y[3] = np.searchsorted(speed_cuts, speed[frames], side="right").astype(np.int8)
```

iii. Shared frame indexing; the audit plot draws the raw speed trace, the three thresholds and the assigned class against trial time.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several guards. (1) The behavior runs 1–3 frames past the imaging in all 89 sessions (59/23/7), so every frame-wise stream and the neural array are cut to `nfr = min(neural_frames, len(ft))`, and the prepass asserts the neural stream is never the longer one. (2) Frames with non-finite `ft_trInd` (inter-trial frames) are excluded. (3) Lick frames that are non-finite, negative, or past `nfr` are dropped. (4) Non-finite `stim_id` entries are skipped, and trials whose wall never gets an id are dropped. (5) Non-finite neural or input values, out-of-range position, retinotopy/neuron count mismatch, unexpected `.npy` layout, conflicting stimulus ids, plane frame mismatch and a session with <2 trials all raise. Note these raise out of `convert()` uncaught, so any one bad session would abort the whole run — the reference instead catches per session and skips; in practice no session triggered it.

ii.
```python
nfr = min(neural_nfr, len(beh["ft"]))
```
```python
if nfr > len(beh["ft"]):
    raise AssertionError(f"Neural frames exceed behavior frames: {sid}")
```
```python
lick_frames = lick_frames[np.isfinite(lick_frames)].astype(np.int64)
lick_frames = lick_frames[(lick_frames >= 0) & (lick_frames < nfr)]
```
```python
ft_trial = np.asarray(beh["ft_trInd"][:nfr])
valid = np.isfinite(ft_trial) & ... 
```

iii. Step 4: "Neural files are systematically 1-3 frames shorter ... Truncate every frame-wise behavior vector and neural stream to their common minimum length before trial extraction", which is exactly what the reference code's `beh[...][:nfr]` does. Step 6/10: shape, dtype, finiteness and class-range assertions are deliberate — "Validate data shapes and types at each step".

## 12-a. What are the most time-consuming steps of the code?

i. Reading and copying the 412 GB of spike files: 769.2 s for the 89 sessions (2–8 s each, scaling with file size), then 151.8 s to pickle the 172.6 GB result — 921.0 s total. Everything else (catalog build + full-dataset behavior prepass) is about 5 s. The agent measured this by printing per-session elapsed/ETA.

ii.
```python
obj = np.load(spk_path(sid), allow_pickle=True).item()
...
print(f"  trials={len(n)}, neurons={n[0].shape[0]}, samples=..., "
      f"session={elapsed:.2f}s, elapsed={so_far:.2f}s, ETA={eta:.2f}s", flush=True)
```

iii. Step 6/7: the I/O is irreducible, so the agent attacked the avoidable parts — "the invariant scalar-NPY payload layout [lets it] infer neural frame counts from file size during the behavior-only prepass", avoiding a second full read of the traces, and trial arrays are filled plane-by-plane to avoid a whole-session concatenated copy.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Two remain. (1) The per-trial `np.flatnonzero(valid & (ft_trial == tr))` rescans the whole session frame vector once per trial — O(n_trials × n_frames), where one `argsort`/`np.split` grouping pass would do; the same loop exists in the human reference. (2) The inner `for plane in planes` copy per trial performs `n_trials × n_planes` fancy-index copies instead of one per plane per session. Neither was identified in CONVERSION_NOTES; both are negligible beside the disk I/O.

ii.
```python
for tr in range(int(beh["ntrials"])):
    ...
    frames = np.flatnonzero(valid & (ft_trial == tr))
    ...
    for plane in planes:
        nr = plane.shape[0]
        n[row:row + nr] = plane[:, frames]
        row += nr
```

iii. The agent's documented efficiency work targeted memory and I/O instead ("Avoids a second full-session concatenated neural copy and lowers peak memory/I/O"), which is where the runtime actually is.

## 12-c. What processing does the code repeat multiple times?

i. Every behavior file is loaded **three** times: once in `build_catalog` (to enumerate sessions and merge stimulus maps), once in `compute_global_stats` (for the speed quartiles and frame-interval statistics), and once in the conversion loop. The retinotopy `.npz` is opened twice per session (`infer_shape` in the prepass, then `convert_session`). The validity mask (`finite trial index & ft_CorrSpc & ft_move>0`) and the wall→category lookup are built twice, once in each of the last two passes. The human reference reads its behavior files twice (trial-length prepass + conversion) and its retinotopy once.

ii.
```python
for exp_type in exp_info:                                   # build_catalog
    behavior = np.load(BEH / f"Beh_{exp_type}.npy", allow_pickle=True).item()
```
```python
for exp_type, entries in grouped(catalog).items():          # compute_global_stats
    behavior = np.load(BEH / f"Beh_{exp_type}.npy", allow_pickle=True).item()
```
```python
current_behavior = np.load(BEH / f"Beh_{current_exp}.npy", allow_pickle=True).item()  # convert
```

iii. The agent documented only the conversion-loop reuse ("behavior files are loaded once per source experiment group ... avoids repeated 0.1-0.4 GB behavior loads") and did not note the two prepasses; the whole prepass costs ~5 s of 921 s, so the repetition is real but immaterial.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items, one of them costly. (1) Neural data is kept in float32 and every one of the 4,691,034 cells is retained, including the 585,641 `unmapped/non-visual` ones the paper excludes — producing a 172.6 GB pickle (vs. the reference's float16, visual-areas-only file). That volume is not used by the decoder, which projects each session onto 100 PCs, and it exhausted GPU memory so that the full training had to fall back to CPU. (2) `compute_global_stats` accumulates every inter-frame interval across the dataset purely to set one metadata number (`nominal_ms`). (3) `has_reward` is computed for all 89 sessions but used only to choose the two `--sample` sessions. (4) The full-dataset prepass runs even in `--sample` mode (deliberate, so quartiles match). (5) `metadata.session_info` carries the full `experiment_records` dicts. (6) In `--show-processing` mode the audit re-concatenates the raw planes for the plotted trial and runs a full `np.allclose` against the converted matrix.

ii.
```python
n = np.empty((nneurons, T), dtype=np.float32)
```
```python
out = np.full(len(iarea), 4, dtype=np.int8)   # nothing is ever dropped
```
```python
dt = np.diff(np.asarray(beh["ft"][:nfr], dtype=np.float64)) * 86_400_000
intervals.append(dt[np.isfinite(dt) & (dt > 0)])
...
nominal_ms = float(np.median(np.concatenate(intervals)))
```
```python
raw = np.concatenate([plane[:, keep] for plane in planes], axis=0)
diff = np.abs(raw - n)
```

iii. Step 5 Key Decisions 2 and 10 justify the size: "This preserves the source and paper-reported population sizes. Region-specific paper filtering remains reproducible from `brain_region_idx`", and "source and target remain float32" was chosen to avoid any precision loss. The README acknowledges the consequence: "The full pickle requires substantially more than 173 GB of available RAM when loaded." The audit-plot recomputation is intentional — it is the visual proof that the neural slice is bit-identical to the source.
