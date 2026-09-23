# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the three released directories under `/app/data`: `beh/` (behavior), `spk/` (deconvolved Suite2p traces) and `retinotopy/` (visual-area label per neuron). `beh/Imaging_Exp_info.npy` is the master index; it is grouped by *paper analysis type* (23 types, 142 entries), and the AI collapses it to the 89 unique *physical* recordings, keyed by `(mname, datexp, blk)`, while keeping the list of analysis groups that reference each recording. Each `Beh_<exp_type>.npy` is opened at most once and only for recordings not yet resolved; only the ~11 fields actually needed are retained (`extract_behavior`), and the multi-GB raw dict is released with `del`/`gc.collect()`. Spikes and retinotopy are then read once per session inside the main conversion loop.

ii.
```python
exp_info = np.load(data_root / "beh" / "Imaging_Exp_info.npy", allow_pickle=True).item()
for exp_type, rows in exp_info.items():
    for row in rows:
        pid = physical_id(row)                      # (mname, datexp, str(blk))
        if pid not in references:
            references[pid] = []
            sessions.append(pid)
        references[pid].append((exp_type, row))
...
for exp_type, rows in exp_info.items():
    unresolved = [row for row in rows if physical_id(row) not in behaviors]
    if not unresolved:
        continue
    raw = np.load(data_root / "beh" / f"Beh_{exp_type}.npy", allow_pickle=True).item()
    for row in rows:
        ...
        key = behavior_key(row)                     # id + "_" + stimtype for swap sessions
        if key in raw:
            behaviors[pid] = extract_behavior(raw[key])
    del raw; gc.collect()
```
```python
retino_path = data_root / "retinotopy" / f"{mouse}_{date}_trans.npz"
iarea = np.load(retino_path, allow_pickle=True)["iarea"]
spk_path = data_root / "spk" / f"{mouse}_{date}_{block}_neural_data.npy"
raw_neural = np.load(spk_path, allow_pickle=True).item()
planes = raw_neural["spks"]
```

iii. From the trajectory (step 15): *"there are 89 unique recordings from 19 mice; the 142 metadata references intentionally reuse recordings for different paper analyses. I'll represent each physical recording once, merging split 'swap' behavior blocks that share one neural file."* The AI verified empirically that the `swap1`/`swap2` behaviour records of a recording are byte-identical in every trial/frame field and differ only in the paper's stimulus-ID lookup, so loading one of them loses nothing. `extract_behavior` was introduced explicitly to avoid holding whole behaviour files in RAM.

## 1-b. How are the data split into subjects (mice)?

i. The subject is the `mname` field of the index entry, i.e. the first element of the physical id. `subjects` is built in first-encounter order and `subject_idx` records each session's index into it. Result: 19 mice, 89 sessions, with the same sessions-per-subject counts as the reference.

ii.
```python
def physical_id(row: dict) -> tuple[str, str, str]:
    return row["mname"], row["datexp"], str(row["blk"])
...
mouse, date, block = pid
if mouse not in subjects:
    subjects.append(mouse)
subject_idx.append(subjects.index(mouse))
```

iii. Not discussed at length; the index names the mouse directly, so no derivation is needed. The AI checked the count against the paper ("89 recordings in 19 mice") in step 15.

## 1-c. How are the data split into sessions?

i. A session is one physical recording = `(mname, datexp, blk)`. Because the index lists the same recording under several analysis types (142 references for 89 recordings), the AI de-duplicates on that triple, keeps first-encounter order, and asserts that exactly 89 sessions were found. The behaviour key adds `_<stimtype>` when the entry has one (swap sessions). All analysis groups that referenced a recording are stored in `metadata['session_info']`.

ii.
```python
pid = physical_id(row)
if pid not in references:
    references[pid] = []
    sessions.append(pid)
references[pid].append((exp_type, row))
...
if len(sessions) != 89:
    raise ValueError(f"Expected 89 unique imaging recordings, found {len(sessions)}")
```

iii. Trajectory step 15 and step 52: the duplicated references are a metadata artefact of the paper's analysis grouping, and the paired swap1/swap2 records "share a full behavioral/neural recording, [so] the converter correctly keeps that recording once while retaining both swap stimuli as separate trial labels within it."

## 1-d. How are the data split into trials?

i. Trials are the `ntrials` corridor traversals declared by the behaviour file. Rather than using the per-frame trial labels (`ft_trInd`), the AI uses the VR geometry: cumulative VR position advances exactly 60 dm per trial (40 dm of textured corridor + 20 dm of grey space), so trial *t* occupies cumulative positions `[60t, 60t+40)` and is resampled at 40 fixed 0.1 m targets. Every trial of every session is kept: 38,110 trials, all exactly 40 samples long. The 2 m grey interval is excluded.

ii.
```python
FULL_TRIAL_BINS = 60
CORRIDOR_BINS = 40
...
targets = (
    np.arange(behavior["ntrials"], dtype=np.float64)[:, None] * FULL_TRIAL_BINS
    + np.arange(CORRIDOR_BINS, dtype=np.float64)[None, :]
).ravel()
...
cube = np.empty((behavior["ntrials"], nneurons, CORRIDOR_BINS), dtype=np.float32)
session_neural = [cube[trial] for trial in range(behavior["ntrials"])]
```

iii. Step 19: *"I'll use the authors' running-only linear interpolation onto 0.1 m bins, retaining the first 4 m (40 samples) of each trial… This directly matches their released processing while satisfying the four 1 m position classes and avoiding gray-space samples that have no corridor-position label."* This mirrors `code/data_process_script.ipynb` cell 9, which calls `get_interpPos_spk(..., n_bins=60, lengths=Corridor_Length)`; the AI keeps the first 40 of those 60 bins.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality control is applied. All 38,110 trials of all 89 sessions are written. The only guards are structural assertions that abort the whole run: equal frame counts across imaging planes, `len(ft_move) >= nframes`, neuron count matching the retinotopy, and strict monotonicity of the running-only cumulative position. Sessions are likewise never dropped (the smallest has 84 trials, so the "≥2 trials per session" requirement is met everywhere).

ii.
```python
if len(behavior["ft_move"]) < nframes:
    raise ValueError("Behavior frame stream is shorter than neural activity")
if len(x) < 2 or np.any(np.diff(x) <= 0):
    raise ValueError("Running-only cumulative VR position is not strictly increasing")
if sum(len(p) for p in planes) != len(iarea):
    raise ValueError(f"Neuron/retinotopy mismatch for {pid}")
```

iii. The module docstring states the intent: *"No arbitrary neuron or trial subsampling is performed."* Trajectory step 15: *"there is no paper-supported extra neuron subsampling."* The paper's own pipeline interpolates all `ntrials` without trial curation, and because every trial is resampled to a fixed 40 spatial samples, the reference solution's motivation for dropping over-long traversals (animals that stop for minutes) does not produce over-long trials here. Step 37/57: the AI monitored per-session trial counts and reported "no data-quality exceptions".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spks` from `spk/<mouse>_<date>_<blk>_neural_data.npy` — a list of (neurons × frames) deconvolved arrays, one per imaging plane — together with `iarea` from `retinotopy/<mouse>_<date>_trans.npz` for the area label, and `ft_move` / `ft_PosCum` from the behaviour file, which define the frames used and the resampling axis.

ii.
```python
raw_neural = np.load(spk_path, allow_pickle=True).item()
planes = raw_neural["spks"]
iarea = np.load(retino_path, allow_pickle=True)["iarea"]
...
moving = behavior["ft_move"][:nframes] > 0
x = np.asarray(behavior["ft_PosCum"][:nframes][moving], dtype=np.float64)
```

iii. Metadata records the provenance: `"source_neural_signal": "Suite2p non-negative deconvolved fluorescence (tau=0.75 s)"`, matching the Methods statement that "All our analyses were based on deconvolved fluorescence traces". Planes are never concatenated into one array; the AI walks them in order and tracks a `source_offset` so the plane order still lines up with `iarea`.

## 2-b. How is the `neural` data processed?

i. Each retained neuron's trace is restricted to frames where the VR advanced (`ft_move > 0`) and then **linearly interpolated (with extrapolation) over cumulative VR position** onto the 0.1 m target grid, producing a (trials × neurons × 40) float32 cube per session; per-trial views of that cube are stored. No z-scoring, smoothing, or averaging. The arithmetic is done in float64 and cast once to float32. Work is chunked over neurons (default 256) to bound peak memory. The resulting dataset is ~274 GB.

ii.
```python
hi = np.searchsorted(x, targets, side="left")
hi = np.clip(hi, 1, len(x) - 1)
lo = hi - 1
weight = (targets - x[lo]) / (x[hi] - x[lo])
...
y = plane[ids][:, moving]
# Linear interpolation/extrapolation, identical to the paper helper's
# scipy interp1d call but vectorized across neurons.
values = y[:, lo] * (1.0 - weight) + y[:, hi] * weight
cube[:, dest_offset:dest_offset + count, :] = values.reshape(
    count, behavior["ntrials"], CORRIDOR_BINS).transpose(1, 0, 2)
```

iii. Step 10: the notebook *"confirms the key convention: 3.17 Hz deconvolved Suite2p activity, trials defined by corridor entry, and analyses restricted to frames where the VR advances (`ft_move > 0`)"*. Step 22/25: before launching the build the AI numerically checked its vectorised interpolation against `scipy.interpolate.interp1d(..., fill_value='extrapolate')` on real data and found agreement to float32 precision. float32 (rather than float16) was chosen to match that precision check; no size reduction was attempted because "the supplied decoder is explicitly engineered for hundreds-of-GB datasets" (step 15).

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron filter is the visual-area assignment, taken verbatim from `code/utils.py::neu_area_ID`: `iarea == 8` → V1, `{0,1,2,9}` → mHV, `{5,6}` → lHV, `{3,4}` → aHV; `iarea == -1` and `iarea == 7` (outside visual cortex) are dropped. This keeps 4,105,393 of 4,691,034 released cells — numerically identical to the reference solution (V1 1,833,035 / mHV 1,108,860 / lHV 495,318 / aHV 668,180). No further curation (the release is already Suite2p cell-classifier curated).

ii.
```python
def area_indices(iarea):
    region = np.full(len(iarea), -1, dtype=np.int8)
    region[iarea == 8] = 0
    region[np.isin(iarea, [0, 1, 2, 9])] = 1
    region[np.isin(iarea, [5, 6])] = 2
    region[np.isin(iarea, [3, 4])] = 3
    keep = region >= 0  # equivalently excludes iarea -1 and 7 (outside visual cortex)
    return keep, region[keep]
```

iii. Metadata: *"Retain released Suite2p cells assigned to visual cortex; iarea -1 and 7 excluded, and remaining iarea labels grouped exactly as code/utils.py neu_area_ID."* The AI inspected `areas.npz` and the `iarea` histogram over all retinotopy files (step 17) before fixing the grouping, and explicitly rejected any additional subsampling (step 15).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every trial starts at corridor entry: sample 0 of each trial is cumulative position `60t`, i.e. the first 0.1 m of the textured corridor, and sample 39 is the last 0.1 m of the 4 m corridor. All trials therefore have identical length (40) and identical nominal extent. Metadata declares `temporal_alignment_event = "entry into the 4 m textured corridor (trial start)"`, `off_start = 0.0`, `off_end = 40/6 = 6.667 s`.

ii.
```python
targets = (np.arange(behavior["ntrials"])[:, None] * FULL_TRIAL_BINS
           + np.arange(CORRIDOR_BINS)[None, :]).ravel()
...
"temporal_alignment_event": "entry into the 4 m textured corridor (trial start)",
"off_start": 0.0,
"off_end": CORRIDOR_BINS / VR_SPEED_DM_S,
```

iii. Step 19: because the VR "advances at a fixed 0.6 m/s, these are also uniform 166.67 ms samples aligned to corridor entry" — i.e. the AI argues that distance-from-entry and (running) time-from-entry are proportional, so a fixed 40-sample window is simultaneously a fixed spatial and a fixed temporal window relative to the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The raw acquisition is 3.17 Hz (315 ms frames). The AI does **not** rebin in time; it resamples in cumulative VR position at 0.1 m steps and then *declares* each sample to be 166.67 ms, on the grounds that the VR advances at a constant 60 cm s⁻¹ whenever the mouse runs. Stationary frames (`ft_move == 0`) are excluded entirely, so an interval of real time of arbitrary length can sit between two consecutive stored samples. `metadata['time_bin_size'] = 1000/6 = 166.67 ms`, `spatial_bin_size_m = 0.1`.

ii.
```python
VR_SPEED_DM_S = 6.0  # 60 cm/s, expressed in the behavior file's decimeters.
...
moving = behavior["ft_move"][:nframes] > 0
...
"time_bin_size": 1000.0 / VR_SPEED_DM_S,
"resampling": ("Linear interpolation over cumulative VR position using only ft_move>0 "
               "frames, matching code/utils.py spk_pos_interp; first 40 of the "
               "paper's 60 0.1-m bins retained (4-m corridor, gray space excluded)."),
```

iii. Step 10 and step 19: the AI first considered "the raw frame-aligned fields rather than blindly using the paper's separate 0.1 m spatial interpolation product" because "this decoder is explicitly time-aligned", then settled on the spatial product, reasoning that the fixed VR speed makes the 0.1 m bins equivalent to uniform 166.67 ms bins, that this "directly matches their released processing", and that it "avoid[s] gray-space samples that have no corridor-position label". The Methods sentence "We only considered timepoints during running for analysis" is the basis for dropping stationary frames.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. `SoundDelPos` — the cumulative VR position of the (delay-shifted) sound cue for each trial — reduced modulo 60 dm to a within-corridor position. The 0.1 m sample grid supplies the second term.

ii.
```python
cue_dm = float(np.mod(behavior["SoundDelPos"][trial], FULL_TRIAL_BINS))
```

iii. Not discussed explicitly in the trajectory beyond the general principle that everything is expressed on the VR position axis. `SoundDelPos` is cumulative across the session, hence the `mod 60`; the AI verified the covariate shapes and value ranges on a real session in step 23 and checked the plotted trial in step 90 ("elapsed time and cue countdown are aligned").

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The within-corridor cue position (dm) minus the sample's position (dm) is divided by the nominal VR speed of 6 dm s⁻¹, giving seconds of VR (running) time until the cue: positive before the cue, negative after — the same sign convention as the reference. The observed range over the dataset is −6.31 s to +7.43 s (the reference, which uses true frame timestamps, spans −72 s to +73 s).

ii.
```python
positions = np.arange(CORRIDOR_BINS, dtype=np.float32)
inp[0] = (cue_dm - positions) / VR_SPEED_DM_S
```

iii. Same justification as 2-e: within the running-only VR-position representation, elapsed time is exactly distance / 60 cm s⁻¹, so a positional offset converts to seconds by a constant factor.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is defined directly on the same 40-sample 0.1 m grid used to resample the neural data, so element *k* of the input corresponds to column *k* of the neural matrix by construction. Every trial's input is `(4, 40)`, matching the neural `(n_neurons, 40)`.

ii.
```python
inp = np.empty((4, CORRIDOR_BINS), dtype=np.float32)
inp[0] = (cue_dm - positions) / VR_SPEED_DM_S
```

iii. Alignment is automatic because all streams are expressed in cumulative VR position; the AI's sample plot check (step 90) confirmed the cue countdown crosses zero where expected.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The annotations in `Imaging_Exp_info.npy`: the `days` field when an entry has one (present only for the `*_train2_after_learning` groups), otherwise the `sess#` field, minimised over all analysis groups that reference the recording. The recording date is **not** used.

ii.
```python
def day_value(references):
    explicit = [row["days"] for _, row in references if "days" in row]
    if explicit:
        return float(explicit[0])
    sessions = [row["sess#"] for _, row in references if "sess#" in row]
    return float(min(sessions)) if sessions else 0.0
```

iii. Docstring: *"Some later training recordings have a `days` field; the remaining before/after and test recordings use `sess#`. A physical recording can occur in several paper-analysis groups, so the minimum session annotation avoids relabeling a pre-exposure recording as a later analysis session."* Step 63: *"Explicit `days` annotations are being used for the later training recordings where provided; elsewhere the release's session annotation supplies the per-trial training-day input."*

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The scalar is broadcast across all 40 samples of every trial of the session. No per-mouse ordering or re-indexing is done, so the values mix two different quantities: `sess#` is an ordinal within a paper analysis group (almost always 0, 1, 2 or 3), while `days` is a true day count (6–15). Across the dataset the input ranges 0–15; within 11 of the 19 mice the value is **not** monotonically increasing with recording date (e.g. TX119 in date order: 1, 1, 0, 1, 1, 10, 1, 1; TX108: 1, 0, 1, 1, 6, 1, 1).

ii.
```python
day = day_value(references[pid])
...
inp[1] = day
```

iii. See 4-a: the AI's stated rationale is that the release's own annotations should be preferred to a derived quantity, and that taking the minimum avoids "relabeling a pre-exposure recording as a later analysis session". The trajectory contains no check that the resulting sequence is ordered in time (the AI did print a per-subject chronological table of `sess#`/`days` in step 18, but did not act on the inconsistency).

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. No raw variable. It is the sample index on the 0.1 m grid divided by the nominal VR speed, i.e. it is a deterministic function of the position bin and is identical in every trial of every session (0, 0.167, …, 6.5 s).

ii.
```python
positions = np.arange(CORRIDOR_BINS, dtype=np.float32)
elapsed_s = positions / VR_SPEED_DM_S
...
inp[2] = elapsed_s
```

iii. Follows from 2-e: since sample *k* is 0.1·k m into the corridor and the VR advances at 60 cm s⁻¹ while the mouse runs, k/6 s of running time has elapsed since corridor entry. Real (wall-clock) time since entry, which includes stops, is not represented anywhere in the converted data.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. A single `arange(40)/6` ramp is computed once per session and copied into row 2 of every trial's input array; it spans 0–6.5 s. Because it is an affine function of the position bin index, it also determines the *Position in corridor* output exactly (`position_class = bin // 10 = floor(elapsed_s * 6 / 10)`), and the decoder concatenates the inputs with the projected neural data before the classification heads, so this output is recoverable from the inputs alone.

ii.
```python
elapsed_s = positions / VR_SPEED_DM_S
position_class = (np.arange(CORRIDOR_BINS) // 10).astype(np.int16)
...
inp[2] = elapsed_s
out[2] = position_class
```

iii. As above; the AI treated "these are also uniform 166.67 ms samples aligned to corridor entry" (step 19) as sufficient, and its sample-plot check (step 90) reports "elapsed time and cue countdown are aligned, corridor class changes exactly every meter" without flagging that the two are the same variable.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. By construction — it is defined on the identical 40-sample grid as the neural columns, one value per neural column.

ii.
```python
inp = np.empty((4, CORRIDOR_BINS), dtype=np.float32)
inp[2] = elapsed_s
```

iii. All streams share the cumulative-VR-position axis, so no explicit alignment step is required.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. `isRew`, the per-trial boolean flag marking trials run in the rewarded corridor.

ii.
```python
"isRew": np.asarray(d["isRew"], dtype=bool),
...
inp[3] = float(behavior["isRew"][trial])
```

iii. Direct read of the released flag; no justification needed or given.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast to 0.0/1.0 float and broadcast across the trial's 40 samples. Observed range over the dataset is 0–1, as expected (it is false for all unsupervised/naive recordings).

ii.
```python
inp[3] = float(behavior["isRew"][trial])
```

iii. None given; none needed.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. `WallName`, the per-trial name of the wall texture (15 distinct names across the dataset, including exemplars `circle1/2/3`, `leaf1/2/3`, `rock1/2`, `wood1/2/5` and the spatial shuffles `leaf1_swap1/2`, `wood1_swap1/2`).

ii.
```python
"WallName": np.asarray(d["WallName"]),
...
out[0] = visual_category(behavior["WallName"][trial])
```

iii. Step 47: *"naive recordings with all exemplar types; their names map cleanly into the four requested categories (circle, leaf, rock, brick/'wood')."* `WallName` is present and unmasked in all sessions including the swap sessions, where the paper's `TrialStim`/`stim_id` lookup is group-specific.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Exemplar suffixes and swap suffixes are collapsed by lower-cased prefix into four classes — circle (0), leaf (1), rock (2), brick (3) — and an unrecognised name raises. The class is constant within a trial and broadcast over the 40 samples. Class fractions over the dataset: 0.305 / 0.466 / 0.086 / 0.143 (reference: 0.312 / 0.481 / 0.082 / 0.125; the small differences follow from the different trial set and sample weighting).

ii.
```python
def visual_category(name: str) -> int:
    name = str(name).lower()
    if name.startswith("circle"):
        return 0
    if name.startswith("leaf"):
        return 1
    if name.startswith("rock"):
        return 2
    # The raw files call the brick-texture family "wood".
    if name.startswith("wood") or name.startswith("brick"):
        return 3
    raise ValueError(f"Unrecognized visual stimulus name: {name!r}")
```

iii. Metadata: *"Exemplar suffixes and spatial swaps collapsed to circle, leaf, rock, or brick; raw 'wood' names are the paper's brick-texture family."* The paper names the four texture photographs circle, leaf, rock and brick, so the AI renames `wood` → `brick` in `output_values` while keeping the mapping.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. `LickPos` (the within-corridor VR position of each lick, in dm) and `LickTrind` (the trial index of each lick).

ii.
```python
"LickTrind": np.asarray(d["LickTrind"]).astype(np.int64),
"LickPos": np.asarray(d["LickPos"]),
```

iii. These are the position-referenced lick records, which is what the position-resampled representation needs; the frame-referenced `LickFr` would have required the (discarded) frame axis.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A (trials × 40) binary matrix: a sample is 1 if at least one lick fell inside its 0.1 m bin. Lick positions are floored to an integer dm bin; licks outside `[0, 40)` (i.e. in the grey space) and licks whose trial index is out of range are discarded. Repeated licks inside one 0.1 m bin collapse to a single 1, and because the mouse's licking bouts during stops all map to the single bin where it stopped, the overall positive fraction is 1.94 % (reference, on a per-imaging-frame axis: 4.14 %).

ii.
```python
lick = np.zeros((behavior["ntrials"], CORRIDOR_BINS), dtype=np.int16)
lick_bin = np.floor(behavior["LickPos"]).astype(np.int64)
valid = ((behavior["LickTrind"] >= 0)
         & (behavior["LickTrind"] < behavior["ntrials"])
         & (lick_bin >= 0) & (lick_bin < CORRIDOR_BINS))
lick[behavior["LickTrind"][valid], lick_bin[valid]] = 1
...
out[1] = lick[trial]
```

iii. Metadata: *"A spatial sample is 1 if one or more licks occurred in its 0.1-m bin."* This is the positional analogue of the reference's per-frame rule.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks are placed in the same 0.1 m corridor bins that index the neural columns, so row 1 of the output array lines up with the neural matrix sample for sample.

ii.
```python
out = np.empty((4, CORRIDOR_BINS), dtype=np.int16)
out[1] = lick[trial]
```

iii. Position is the common axis for all streams; no separate alignment step.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. None. Position is the resampling axis itself, so the label is the sample index: `np.arange(40) // 10`. `ft_Pos` (the released per-frame corridor position) is never read.

ii.
```python
position_class = (np.arange(CORRIDOR_BINS) // 10).astype(np.int16)
...
out[2] = position_class
```

iii. Implied by 2-e/1-d: since each trial is resampled at exactly 40 known corridor positions, the position of each sample is known exactly and needs no measurement. Step 19 gives this as a positive reason for the spatial representation: it "satisf[ies] the four 1 m position classes and avoid[s] gray-space samples that have no corridor-position label."

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Integer division of the 0.1 m sample index by 10. The resulting vector `[0]*10 + [1]*10 + [2]*10 + [3]*10` is byte-identical in all 38,110 trials, giving exactly 25.000 % per class, and is an exact function of the `time_since_trial_start` input.

ii.
```python
position_class = (np.arange(CORRIDOR_BINS) // 10).astype(np.int16)
out[2] = position_class
```

iii. See 9-a. The AI's verification step recorded "corridor position bin: 0.25/0.25/0.25/0.25" and its plot check noted "corridor class changes exactly every meter" (step 90), which it read as confirmation rather than as a degeneracy.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four equal 1 m bins — 0–1, 1–2, 2–3 and 3–4 m — exactly as the decoder task specifies, obtained as `bin // 10` over the 40 decimetre samples. `output_values[2] = ['0-1 m', '1-2 m', '2-3 m', '3-4 m']`.

ii.
```python
"output_values": [
    CATEGORY_NAMES,
    ["not licking", "licking"],
    ["0-1 m", "1-2 m", "2-3 m", "3-4 m"],
    ["0-25%", "25-50%", "50-75%", "75-100%"],
],
```

iii. Directly from the instruction "Position in corridor discretized into 4 equal-length, 1-m-long spatial bins". The same rule as the reference's `ft_Pos // 10`.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Trivially: the position label *is* the index of the neural column, so alignment is exact by construction.

ii.
```python
cube = np.empty((behavior["ntrials"], nneurons, CORRIDOR_BINS), dtype=np.float32)
...
out[2] = position_class
```

iii. Common position axis for all streams.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. `run_pos`, the released (trials × 60) matrix of "running speed interpolated into trials * positions" (documented in `code/data_process_script.ipynb`), truncated to the first 40 corridor bins. The per-frame `ft_RunSpeed` used by the reference is not read.

ii.
```python
"run_pos": np.asarray(d["run_pos"], dtype=np.float32)[:, :CORRIDOR_BINS],
...
out[3] = np.digitize(behavior["run_pos"][trial], speed_edges).astype(np.int16)
```

iii. `run_pos` is the authors' own position-binned running speed, i.e. it is already on exactly the grid the AI resamples the neural data onto, so no re-derivation from the frame axis is needed.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. None beyond the truncation to 40 bins and the discretisation. Raw values are used as released, including negative speeds (backward running) and large positive outliers (values up to ~550 in one session); no clipping, smoothing, or NaN handling is applied.

ii.
```python
all_speeds = np.concatenate([behaviors[pid]["run_pos"].ravel() for pid in sessions])
speed_edges = np.quantile(all_speeds, [0.25, 0.50, 0.75]).astype(np.float32)
if not np.all(np.isfinite(speed_edges)):
    raise ValueError("Non-finite running-speed quartiles")
```

iii. The AI profiled the pooled speed distribution (min / quartiles / max / NaN / Inf counts) in step 20 before fixing the scheme, and asserts finiteness of the resulting edges.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Into quartiles with **global** edges: the quartiles of `run_pos[:, :40]` pooled over all 89 sessions (16.58, 28.72, 43.29), applied with `np.digitize` to every trial of every session. Globally this gives exactly 25 % per class (confirmed in `verification_stats.json`), but the split is not balanced within a session or within a mouse. `output_values[3]` labels the classes "0-25%" … "75-100%".

ii.
```python
speed_edges = np.quantile(all_speeds, [0.25, 0.50, 0.75]).astype(np.float32)
...
out[3] = np.digitize(behavior["run_pos"][trial], speed_edges).astype(np.int16)
...
"running_speed_quartile_edges": speed_edges.tolist(),
```

iii. The instruction is "Running speed discretized into 4 bins, each corresponding to 25% of the data"; the AI reads "the data" as the whole dataset and verifies the outcome ("global speed classes at precisely 25% each", step 87). Computing the edges before the main loop also required only the small `run_pos` arrays to be held, not the neural data.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `run_pos` is indexed by (trial, 0.1 m position bin), the same index pair as the resampled neural cube, so row 3 of the output matches the neural columns exactly.

ii.
```python
"run_pos": np.asarray(d["run_pos"], dtype=np.float32)[:, :CORRIDOR_BINS],
...
out[3] = np.digitize(behavior["run_pos"][trial], speed_edges).astype(np.int16)
```

iii. Common position axis; the AI's plot check (step 90) confirmed "speed varies independently" across samples.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The behaviour stream is truncated to the imaged frames (`[:nframes]`) for the two frame-indexed fields used, `ft_move` and `ft_PosCum`. Everything else is handled by assertion rather than repair: unequal plane frame counts, a behaviour stream shorter than the neural one, a neuron/retinotopy count mismatch, a non-monotonic running-only position, a neuron-accounting mismatch, an unrecognised wall name, a missing behaviour record and non-finite speed quartiles all raise and abort the entire conversion. There is no per-session `try/except`, so one bad session would lose the whole (multi-hour, 274 GB) run. Out-of-range lick records are silently dropped. Per-trial fields (`run_pos`, `WallName`, `isRew`, `SoundDelPos`) are *not* restricted to trials that were actually imaged, and positions beyond the imaged range are silently extrapolated by `interpolation_lookup` rather than dropped; in practice the sessions checked have behaviour ending within one frame of the imaging, so no large extrapolation occurs. A NaN in `run_pos` would be mapped silently to the top speed class by `np.digitize` (no NaNs are present in the sessions inspected).

ii.
```python
moving = behavior["ft_move"][:nframes] > 0
x = np.asarray(behavior["ft_PosCum"][:nframes][moving], dtype=np.float64)
...
missing = [pid for pid in sessions if pid not in behaviors]
if missing:
    raise KeyError(f"No behavior record found for {missing}")
...
valid = ((behavior["LickTrind"] >= 0) & (behavior["LickTrind"] < behavior["ntrials"])
         & (lick_bin >= 0) & (lick_bin < CORRIDOR_BINS))
```

iii. The AI's stance in the trajectory is that the release is clean and that anything unexpected should be surfaced loudly: it reports at each checkpoint that "no frame-length, retinotopy, or stimulus-label inconsistencies have appeared" (step 28) and "no data-quality exceptions" (step 57). The atomic `.tmp` + `os.replace` write is the one explicit failure-tolerance measure ("the temporary file will be renamed to `converted_data.pkl` only after the full write completes", step 68).

## 12-a. What are the most time-consuming steps of the code?

i. (1) Reading the 405 GB of `spk/*_neural_data.npy` files — 89 unpickled dicts of 4–8 GB each — which dominates everything else; (2) the per-session interpolation, which touches every retained neuron × every moving frame (a gather of `n_neurons × n_trials × 40` values plus two multiplies); (3) pickling the ~274 GB result, and (4) the prologue that loads all 23 behaviour files (6.6 GB) once to compute the global speed quartiles. Downstream, the size itself is the dominant cost: the verifier needed a full 274 GB read, and the decoder run spent its entire wall-clock budget in repeated passes over the payload before OOM-ing on the 23.7 GB GPU.

ii.
```python
raw_neural = np.load(spk_path, allow_pickle=True).item()
...
values = y[:, lo] * (1.0 - weight) + y[:, hi] * weight
...
with open(tmp_path, "wb", buffering=16 * 1024 * 1024) as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The AI was aware of the scale, sized it in advance ("the expected neural payload is about 274 GB", step 25), chunked the neuron axis to keep peak RAM bounded, and used a buffered atomic write. It accepted the cost deliberately: "the supplied decoder is explicitly engineered for hundreds-of-GB datasets" (step 15).

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `trial_covariates` builds a fresh `(4, 40)` input and `(4, 40)` output array per trial, three of whose eight rows are the *same* constant vectors in every trial (`elapsed_s`, `position_class`) or a broadcast scalar (`day`, `isRew`, stimulus class); the whole thing could be built as two `(ntrials, 4, 40)` arrays with vectorised `np.mod`, `np.digitize` and a vectorised name→class lookup. `visual_category` re-parses a string per trial (38,110 `str.lower()`/`startswith` chains) where a dict over the ~15 unique names would do. `subjects.index(mouse)` is a linear scan per session. `load_inventory` iterates over all of `exp_info` twice. The neuron chunk loop is genuinely needed for memory, but `plane[ids]` materialises a fancy-indexed copy of the full frame axis per chunk before the `[:, moving]` selection, so it copies far more than it uses.

ii.
```python
for trial in range(behavior["ntrials"]):
    cue_dm = float(np.mod(behavior["SoundDelPos"][trial], FULL_TRIAL_BINS))
    inp = np.empty((4, CORRIDOR_BINS), dtype=np.float32)
    inp[0] = (cue_dm - positions) / VR_SPEED_DM_S
    inp[1] = day
    inp[2] = elapsed_s
    inp[3] = float(behavior["isRew"][trial])
```
```python
y = plane[ids][:, moving]
```

iii. Not discussed; these loops are negligible next to the 405 GB of I/O, which is presumably why they were left alone.

## 12-c. What processing does the code repeat multiple times?

i. `positions`, `elapsed_s` and `position_class` are rebuilt on every session (and the latter two are then copied into all 38,110 trial arrays, ~6 MB of duplicated constants). `exp_info` is walked twice in `load_inventory` (once to build the inventory, once to resolve behaviour files). Every behaviour file is loaded in full and discarded in the prologue only to harvest `run_pos` for the quartile edges, and `run_pos` for all sessions is then kept in memory for the whole run. `np.mod` is applied per trial rather than once per session. Most consequentially, the per-trial neural arrays are 89 views into one big cube, so the pickle stores each trial separately and any downstream reader pays for the full materialisation again.

ii.
```python
positions = np.arange(CORRIDOR_BINS, dtype=np.float32)
elapsed_s = positions / VR_SPEED_DM_S
position_class = (np.arange(CORRIDOR_BINS) // 10).astype(np.int16)
```
```python
all_speeds = np.concatenate([behaviors[pid]["run_pos"].ravel() for pid in sessions])
```

iii. The AI explicitly checked (step 22/23) that pickling a list of views of one cube writes only the payload once and round-trips correctly, so it considered the view-based storage safe. `extract_behavior` exists precisely so that the behaviour dicts kept between the prologue and the main loop are small.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The big one is precision and size: the neural cube is written as float32 at full neuron count (274 GB), while the supplied decoder immediately reduces each session to a 2,000-dimensional random projection and then 100 principal components — so almost all of the stored precision and dimensionality is thrown away. float16 (as in the reference) would have halved it with no effect on the PCs. The consequence was not hypothetical: the decoder run reached the training loop, exceeded the 23.7 GB GPU, and its CPU retry was abandoned, so no decoder accuracy was ever obtained. Three of the four input rows and two of the four output rows are constants replicated across 40 samples and 38,110 trials, which the decoder's own broadcasting path would have supplied from 1-D arrays. `Corridor_Length` and `Texture_Length` are extracted from every behaviour record and never used (60 and 40 are hardcoded instead), and `FRAME_RATE_HZ` survives only as a metadata string. The `metadata['session_info']` block recomputes `sorted({...})` per session for bookkeeping that nothing downstream reads.

ii.
```python
cube = np.empty((behavior["ntrials"], nneurons, CORRIDOR_BINS), dtype=np.float32)
```
```python
"Corridor_Length": float(d["Corridor_Length"]),
"Texture_Length": float(d["Texture_Length"]),
```
```python
inp[1] = day
inp[3] = float(behavior["isRew"][trial])
out[0] = visual_category(behavior["WallName"][trial])
```

iii. The AI justified the size as intentional ("This is intentionally a large conversion (about 274 GB). No arbitrary neuron or trial subsampling is performed.") and float32 as matching the precision at which it had validated the interpolation against scipy. When the decoder OOM-ed it concluded: "A full 200-epoch CPU retry over 274 GB would be disproportionate and does not test formatting further, so I'm stopping that retry" (step 142).
