# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the three subdirectories of `/app/data`: `beh/` (behavior), `spk/` (deconvolved traces, one file per session) and `retinotopy/` (per-cell visual area). `beh/Imaging_Exp_info.npy` is treated as the master index: it holds 23 experiment-group keys with 142 descriptors that resolve to 89 unique physical recordings. Each descriptor yields a session id `<mname>_<datexp>_<blk>` and a behavior key (with a `_<stimtype>` suffix for swap sessions, plus a prefix-match fallback). Records are grouped by experiment type so each `Beh_<group>.npy` is loaded once for all of its sessions; `spk/<sid>_neural_data.npy` and `retinotopy/<mname>_<datexp>_trans.npz` are read once per session. The behavior files are actually read **twice** over the run: once in a behavior-only "prepass" (shape validation + global running-speed quartile edges) and once in the conversion pass.

ii.
```python
exp_info = np.load(BEH_ROOT / "Imaging_Exp_info.npy", allow_pickle=True).item()
...
sid = session_id(db)                      # f"{db['mname']}_{db['datexp']}_{db['blk']}"
behavior_key = sid
if "stimtype" in db:
    behavior_key = f"{sid}_{db['stimtype']}"
```
```python
for group, group_records in records_by_group(records).items():
    behavior_dict = np.load(BEH_ROOT / f"Beh_{group}.npy", allow_pickle=True).item()
    for record in group_records:
        beh = get_behavior(behavior_dict, record)
        frames_by_trial = selected_frames_by_trial(beh, sid)
        activity, region_idx, iarea = load_filtered_neural(record)
```
```python
ret_path = RET_ROOT / f"{db['mname']}_{db['datexp']}_trans.npz"
with np.load(ret_path, allow_pickle=True) as ret:
    iarea = np.asarray(ret["iarea"])
spk_path = SPK_ROOT / f"{sid}_neural_data.npy"
raw = np.load(spk_path, allow_pickle=True).item()
```

iii. From CONVERSION_NOTES Step 1/2/4: the reference `utils.load_exp_beh`, `load_spk` and `load_retino` use exactly these three sources, and `code/README.md` points at `data_process_script.ipynb` as the loading workflow. The AI notes that the 142 index entries refer to only 89 physical recordings "because the same session can serve multiple analyses", and that the three `Unsupervised_pretraining_behavior/` files are behavior-only cohorts with no neural file, so they are excluded. Behavior files are loaded per group and released to bound memory ("Behavior files are loaded once per relevant group in each pass").

## 1-b. How are the data split into subjects (mice)?

i. The subject is the `mname` field of the experiment-index descriptor. The subject list is the sorted set of unique `mname` values across the records being converted, and `subject_idx` is each session's index into that list. This yields 19 subjects for the full run (2 for `--sample`).

ii.
```python
subjects = sorted({r["db"]["mname"] for r in records})
subject_lookup = {name: i for i, name in enumerate(subjects)}
...
subject_idx.append(subject_lookup[record["db"]["mname"]])
...
"subjects": subjects,
"subject_idx": np.asarray(subject_idx, dtype=np.int16),
```

iii. CONVERSION_NOTES Step 2 records "Subjects | 19: DR10, DR15, LZ13, ... VR2" with exact per-mouse session counts, and Step 3 quotes the paper: "We performed 89 recordings in 19 mice". The mapping is described as a "deterministic subject list and per-session integer lookup"; sample mode deliberately lists only the subjects present in its two sessions.

## 1-c. How are the data split into sessions?

i. A session is one physical recording = mouse + date + block (`<mname>_<datexp>_<blk>`), which is also the name of the spike file. The AI walks the experiment index in order and keeps a recording the first time it is seen, skipping later appearances under other experiment groups, and hard-asserts that exactly 89 records result.

ii.
```python
records, seen = [], set()
for group, descriptors in exp_info.items():
    for db0 in descriptors:
        db = dict(db0); sid = session_id(db)
        if sid in seen:
            continue
        seen.add(sid)
        ...
        records.append({...})
if len(records) != 89:
    raise AssertionError(f"Expected 89 physical sessions, found {len(records)}")
```

iii. Step 4 "Discrepancies Found": "Experiment index has 142 entries across 23 analysis groups" vs "89 physical neural files"; resolution: "Convert each mouse/date/block physical recording exactly once (89 sessions); do not duplicate recordings merely because they support multiple paper analyses." For swap sessions the AI verified "the two behavior keys have identical timestamps, trial labels, and frame alignment; only `stim_id` marks one swap or the other", so only one physical session is emitted.

## 1-d. How are the data split into trials?

i. Trials are the native trials declared by the behavior (`ntrials`, indexed by `ft_trInd`). For each trial the retained frames are the imaging frames that simultaneously satisfy: (a) `ft_trInd == trial` (finite), (b) `ft_CorrSpc` (inside the 4 m textured corridor), and (c) `ft_move > 0` (VR advancing, i.e. the mouse is running). Trials keep their own variable length; nothing is padded or truncated to a common window. Because condition (c) is applied *within* a trial, the retained frames of a trial are generally **not contiguous** — stationary periods are excised from the middle of trials (≈28.5% of in-corridor frames are dropped dataset-wide; 821,579 of ~1.15M in-corridor frames are kept).

ii.
```python
def selected_frames_by_trial(beh: dict, sid: str) -> list[np.ndarray]:
    """Apply the paper's moving, textured-corridor frame mask."""
    trial_id = np.asarray(beh["ft_trInd"])
    valid = (
        np.isfinite(trial_id)
        & np.asarray(beh["ft_CorrSpc"], dtype=bool)
        & (np.asarray(beh["ft_move"]) > 0)
    )
    selected = np.flatnonzero(valid)
    selected_trial = trial_id[selected].astype(np.int64)
    ntrials = int(beh["ntrials"])
    frames = [selected[selected_trial == trial] for trial in range(ntrials)]
```

iii. Step 4/Step 5: "Use textured-corridor frames and the reference running criterion (`ft_CorrSpc & (ft_move > 0)`) for neural/behavior samples. This also makes four 1-m position bins meaningful and prevents water-consumption pauses from dominating. Retain actual frame timestamps as time inputs so skipped stationary intervals remain explicit." The AI points to `Get_dprime_selective_neuron`, `Get_coding_direction` and `Get_sort_spk` in `utils.py`, which all build `corr_fr = beh['ft_CorrSpc'][:nfr] & (beh['ft_move'][:nfr] > 0)`. It also justifies using the mask rather than the fractional `StartFr`/`GrayFr` frame numbers: "Select frames by `ft_trInd == trial`, `ft_CorrSpc`, and movement rather than rounding fractional endpoints. This avoids off-by-one gray frames".

## 1-e. How are trials filtered based on quality controls?

i. No trial is dropped. All 38,110 native trials are converted. The only trial-level gate is a hard assertion that every trial retains at least two selected frames — if any trial had fewer, the script would abort rather than drop it. There is no trial-length outlier filter, no session-level minimum-trial check, and no drop of trials whose corridor was not imaged (instead the script asserts that every selected frame index is within the neural array). The ft_move mask indirectly shortens "parked animal" trials (max retained length 178 frames), but the corresponding wall-clock inputs survive: `time_since_trial_start` reaches 1765 s and `time_to_sound_cue` spans [-1763.3, 723.5] s.

ii.
```python
lengths = np.asarray([len(x) for x in frames])
if np.any(lengths < 2):
    bad = np.flatnonzero(lengths < 2).tolist()
    raise ValueError(f"{sid}: trials with fewer than two valid frames: {bad}")
```
```python
max_selected = max(int(x[-1]) for x in frames_by_trial)
if max_selected >= nframes_neural:
    raise ValueError(
        f"{sid}: selected frame {max_selected} exceeds neural length {nframes_neural}"
    )
```

iii. Step 5 Key Decision 6: "**Trial curation**: Keep every imaging trial because all have valid aligned running/corridor frames and all sessions have at least 84 trials." Step 3 Curation Steps: "No whole-session/trial exclusion rule is reported for the general dataset." Step 10 Check 1 addresses the extreme time values directly: "Extreme elapsed/cue times are genuine trials with long pauses; because stationary observations are removed but exact timestamps are preserved, these values are expected and documented rather than silently compressed."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `spks` entry of `spk/<session_id>_neural_data.npy` — a list of one (n_cells_plane, n_frames) float32 array per imaging plane — concatenated (effectively) along the neuron axis in released plane order. Per-neuron brain area comes from `iarea` in `retinotopy/<mname>_<datexp>_trans.npz`. The script validates that the file contains only the key `spks`, that all planes share a frame count, and that the total cell count equals `len(iarea)`.

ii.
```python
raw = np.load(spk_path, allow_pickle=True).item()
if set(raw) != {"spks"} or not isinstance(raw["spks"], list):
    raise ValueError(f"{sid}: unexpected neural file structure")
planes = raw["spks"]
nframes_set = {int(x.shape[1]) for x in planes}
raw_neurons = sum(int(x.shape[0]) for x in planes)
if raw_neurons != len(iarea):
    raise ValueError(f"{sid}: spks has {raw_neurons} cells but retinotopy has {len(iarea)}")
```

iii. Step 1: "Neural files contain a list named `spks`; reference loading concatenates that list across planes. These are the released processed calcium-event/deconvolved activity values used directly by all analyses." This mirrors `utils.load_spk`. Step 4: "Retinotopy exactly matches cell count/order".

## 2-b. How is the `neural` data processed?

i. No processing at all: no ΔF/F, no deconvolution, no smoothing, no z-scoring, no temporal rebinning, no spatial interpolation. Each trial is the column slice of the (area-filtered) session matrix at that trial's selected frame indices, stored as a C-contiguous **float32** copy. Trials keep their own length. A finiteness check is run over the session matrix and over every trial array.

ii.
```python
activity = np.empty((int(keep.sum()), nframes), dtype=np.float32)
raw_offset = kept_offset = 0
for plane in planes:
    nplane = int(plane.shape[0])
    plane_keep = keep[raw_offset : raw_offset + nplane]
    nkeep = int(plane_keep.sum())
    activity[kept_offset : kept_offset + nkeep] = plane[plane_keep]
    raw_offset += nplane; kept_offset += nkeep
```
```python
# Explicit C-order copy lets the full session matrix be released.
ntrial = np.array(activity[:, frames], dtype=np.float32, order="C", copy=True)
```

iii. Step 5 Key Decision 4: "**Neural values**: Preserve float32 deconvolved activity without normalization. Reference d-prime uses raw deconvolved frames; decoder code learns per-session projections and does not require conversion-time z-scoring." Step 4: "Use released `spks` directly; do not calculate ΔF/F or deconvolve again" (paper methods: Suite2p neuropil correction + non-negative deconvolution with 0.75 s decay already applied). The plane-by-plane fill is an explicit memory optimization to avoid "one full all-cell concatenated copy per session".

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron filter is the visual-area assignment. Raw retinotopy codes are mapped V1←{8}, mHV←{0,1,2,9}, lHV←{5,6}, aHV←{3,4}; codes −1 and 7 are dropped as outside visual cortex, and any other unmapped code raises. This keeps 4,105,393 of 4,691,034 neuron-session units. No selectivity (d′) threshold, no activity threshold, and no additional quality filter is applied.

ii.
```python
region = np.full(len(iarea), -1, dtype=np.int16)
region[iarea == 8] = 0
region[np.isin(iarea, [0, 1, 2, 9])] = 1
region[np.isin(iarea, [5, 6])] = 2
region[np.isin(iarea, [3, 4])] = 3
keep = region >= 0
unexpected = np.unique(iarea[(~keep) & (~np.isin(iarea, [-1, 7]))])
if len(unexpected):
    raise ValueError(f"{sid}: unmapped retinotopy codes {unexpected.tolist()}")
```

iii. Step 4: "Suite2p classification is the quality curation. For target regional neural matrices, exclude raw codes −1 and 7 because reference code explicitly labels them outside visual cortex and no valid target brain-region name exists; retain the 4,105,393 cells in V1/mHV/lHV/aHV." Step 5 Key Decision 5 adds: "No stimulus-selectivity filtering is applied: d-prime thresholds are analysis-specific and would leak the requested visual output into neuron selection." The mapping reproduces `utils.neu_area_ID`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to trial start = corridor entry, implemented implicitly by the frame mask: the first retained column of a trial is the first moving, in-texture frame of that trial, and the trial runs to the last such frame. Trials are variable-length (T from 11 to 178, mean 22.25); nothing is padded or cut to a common window. Metadata records `temporal_alignment_event = "trial start (entry into the 4-m textured corridor)"`, `off_start = 0.0`, `off_end = None`. The time inputs carry the exact elapsed seconds, so within-trial gaps created by the `ft_move` mask are explicit in the input rather than implicit in the sample spacing.

ii.
```python
frames = [selected[selected_trial == trial] for trial in range(ntrials)]
...
ntrial = np.array(activity[:, frames], dtype=np.float32, order="C", copy=True)
ft = np.asarray(beh["ft"])[frames]
time_since_start = (ft - float(beh["Trial_start_time"][trial])) * 86400.0
```
```python
"temporal_alignment_event": "trial start (entry into the 4-m textured corridor)",
"off_start": 0.0,
"off_end": None,
"timepoints_may_have_gaps": True,
```

iii. Step 3: "Trials are naturally aligned to corridor entry via `Trial_start_time` / `StartFr`, matching the requested alignment event." Step 5 Key Decision 9: "`off_start=0.0` because conceptual trial start is corridor entry; `off_end=None` because elapsed end time varies. Document moving-frame selection/gaps..." Step 4 adds that the mask is used instead of rounding the fractional `StartFr`/`GrayFr` to avoid off-by-one boundary frames.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging frame is the bin: 3.17 Hz → `time_bin_size = 1000/3.17 = 315.4574` ms. No rebinning, resampling, smoothing or spatial (position) interpolation is applied. Important caveat that the AI documents itself: because stationary frames are removed inside trials, consecutive stored bins are **not always 315 ms apart**; the metadata flags `"timepoints_may_have_gaps": True` and the actual elapsed time is carried in the `time_since_trial_start` / `time_to_sound_cue` input rows (which reach 1765 s within a single trial).

ii.
```python
IMAGING_RATE_HZ = 3.17
TIME_BIN_MS = 1000.0 / IMAGING_RATE_HZ
...
"time_bin_size": float(TIME_BIN_MS),
"sampling_rate_hz": float(IMAGING_RATE_HZ),
"frame_selection": "ft_trInd == trial AND ft_CorrSpc AND ft_move > 0",
"stationary_frames_removed": True,
"timepoints_may_have_gaps": True,
```

iii. Step 5 Key Decision 2: "**Temporal representation**: Preserve original imaging-frame activity rather than spatially interpolating to 60 bins. The target requires a common time-bin duration and temporal cue/lick variables; spatial interpolation would replace time with position. The nominal bin is `1000/3.17 = 315.4574 ms`, while actual elapsed times are carried explicitly because excluded stationary frames create gaps." Step 3 sources the rate from the processing notebook ("Calcium signal recording frame rate: fs = 3.17Hz").

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. `SoundTime[trial]` (the MATLAB datenum timestamp of the cue on that trial) and `ft` (the datenum timestamp of every imaging frame), evaluated at the trial's selected frames.

ii.
```python
ft = np.asarray(beh["ft"])[frames]
time_to_cue = (float(beh["SoundTime"][trial]) - ft) * 86400.0
```

iii. Step 5 mapping table: source "`SoundTime`, `ft`" → `input[0]`, "Frame alignment variables documented in processing notebook; cue alignment in `spk_2_cue`". The AI chose the timestamp rather than the fractional frame index `SoundFr`; the two are numerically equivalent (interpolating `SoundFr` onto the `ft` axis reproduces `SoundTime` to <1e-12 s).

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Subtract the frame timestamp from the cue timestamp and convert datenum days to seconds (×86400). The result is signed: positive before the cue, zero at the cue, negative after. It is a continuous time-varying row, not a binary onset pulse. Values are checked for finiteness (a NaN cue would abort).

ii.
```python
time_to_cue = (float(beh["SoundTime"][trial]) - ft) * 86400.0
inp = np.vstack([time_to_cue, np.full(T, record["day_value"]),
                 time_since_start, np.full(T, reward_available[trial])]).astype(np.float32)
if not np.all(np.isfinite(inp)) or not np.all(np.isfinite(ntrial)):
    raise ValueError(f"{sid} trial {trial}: NaN/Inf after conversion")
```

iii. Step 5 mapping: "`(SoundTime[trial] - ft[selected_frames]) * 86400`, signed seconds to cue; float32 ... Positive before cue, zero at cue, negative after cue. Decoder Task explicitly requests continuous time-to-cue, so this is not replaced with a binary onset pulse."

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from `ft` indexed by the very same `frames` array used to slice the neural columns of that trial, so it is element-for-element aligned with the neural matrix and has the same T. The script asserts the time dimensions of neural, input and output match for every trial.

ii.
```python
ft = np.asarray(beh["ft"])[frames]
...
if ntrial.shape[1] != inp.shape[1] or inp.shape[1] != out.shape[1]:
    raise AssertionError(f"{sid} trial {trial}: time dimension mismatch")
```

iii. Step 1: "Reference temporal alignment uses behavior variables already sampled/aligned to neural frames (`ft*`, `StartFr`, `EndFr`, `SoundFr`, `LickFr`, etc.)." Every stream in this dataset shares the imaging-frame grid, so indexing by the same frame array is the alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the experiment-index descriptor of the session: the `days` field when present, otherwise the `sess#` field. Whichever is used is recorded per session as `day_source_field`. Crucially, the descriptor used is the *first* one encountered during deduplication, i.e. whichever experiment group happens to list that recording first.

ii.
```python
if "days" in db:
    day_value = float(db["days"]);  day_source = "days"
elif "sess#" in db:
    day_value = float(db["sess#"]); day_source = "sess#"
else:
    raise ValueError(f"{sid}: neither 'days' nor 'sess#' is available")
```

iii. Step 4: "Use `days` when present, otherwise `sess#`, from the primary experiment descriptor. This is the only released continuous per-session training-stage/day field; store source group/value in metadata." Step 5 mapping: "`days` has priority because it is explicit; `sess#` is the released fallback."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The scalar is simply broadcast across all T bins of every trial of the session as input row 1. No ordering by date, no per-mouse re-indexing, no normalization. Across the 89 sessions the resulting values are 0, 1, 2, 3, 6, 7, 8, 9, 10, 12, 13, 15 — but they are dominated by the constant 1 (most mice have 4–6 sessions that all receive the value 1) and they are not monotonic in date within a mouse (e.g. TX119: 2023_12_12→1, 2023_12_13→1, 2023_12_14→0, 2023_12_23→1, 2024_01_06→10). The two source fields also have different meanings and scales: `sess#` is a session ordinal within an experiment phase (0/1/2), `days` is an actual day count (6–15); they are mixed into one continuous variable.

ii.
```python
inp = np.vstack([
    time_to_cue,
    np.full(T, record["day_value"]),   # broadcast per-trial scalar
    time_since_start,
    np.full(T, reward_available[trial]),
]).astype(np.float32, copy=False)
```
```python
"day_of_training": float(record["day_value"]),
"day_source_field": record["day_source"],
```

iii. Step 4 acknowledges the ambiguity: "Descriptors primarily use numeric `sess#`; later train2-after sessions use `days` or a day-like `sess#` ... Paper describes before/after stages and approximately 2-week phases but does not list every session day." The AI treats the released field as authoritative rather than deriving an order from the recording dates, and stores the provenance in metadata.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. `Trial_start_time[trial]` (the datenum timestamp of corridor entry for that trial) and `ft` (the datenum timestamp of every imaging frame), at the trial's selected frames.

ii.
```python
ft = np.asarray(beh["ft"])[frames]
time_since_start = (ft - float(beh["Trial_start_time"][trial])) * 86400.0
```

iii. Step 5 mapping: "`Trial_start_time`, `ft` → `input[2]` ... Frame alignment documented in notebook". As with the cue, the timestamp is used rather than the fractional `StartFr`; interpolating `StartFr` onto `ft` reproduces `Trial_start_time` to <1e-12 s.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Frame timestamp minus trial-start timestamp, converted from days to seconds (×86400). Positive after entry, ≈0 at the first retained frame. Continuous, time-varying, float32. Because stationary frames are omitted but real timestamps are kept, the value jumps across a pause; the per-session maxima therefore run up to 1765 s.

ii.
```python
time_since_start = (ft - float(beh["Trial_start_time"][trial])) * 86400.0
```

iii. Step 5 mapping: "seconds since corridor entry ... Preserves actual elapsed time across omitted stationary frames." Step 10: extreme values are "genuine trials with long pauses ... expected and documented rather than silently compressed."

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same mechanism as 3-c: `ft` indexed by the identical `frames` array used for the neural columns, so it is bin-for-bin aligned and shares T. Verified by the per-trial shape assertion.

ii.
```python
ntrial = np.array(activity[:, frames], dtype=np.float32, order="C", copy=True)
ft = np.asarray(beh["ft"])[frames]
time_since_start = (ft - float(beh["Trial_start_time"][trial])) * 86400.0
```

iii. All behavioral and neural streams in this dataset live on the imaging-frame grid, so sharing the frame index array is the alignment (Step 1 notes).

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. `WallName` (the texture of each trial's corridor) together with `isRew`. `isRew` is used only to *identify* which wall is the rewarded one; the availability flag is then `WallName == rewarded_wall` for every trial. Sessions where no reward was ever delivered have no rewarded wall and get all zeros.

ii.
```python
def rewarded_wall(beh: dict, sid: str) -> str | None:
    walls = np.asarray(beh["WallName"])
    reward_walls = np.unique(walls[np.asarray(beh["isRew"], dtype=bool)])
    if len(reward_walls) > 1:
        raise ValueError(f"{sid}: multiple rewarded wall names {reward_walls.tolist()}")
    return None if len(reward_walls) == 0 else str(reward_walls[0])
```
```python
reward_available = (
    np.zeros(len(walls), dtype=np.float32) if rw_wall is None
    else (walls == rw_wall).astype(np.float32)
)
```

iii. Step 4 "Reward label" row: "Active sessions have 1,037 delivered rewards but 1,147 trials in the rewarded corridor; passive sessions match 3,299/3,299 ... Derive the single rewarded `WallName` from any `isRew=True` trial, then mark every trial with that wall as availability=1 ... Result: 4,446 availability-positive trials, not 4,336 delivered-reward trials." The AI points to `utils.get_cat_id`, which identifies the rewarded stimulus the same way (`rewStim = WallName[isRew][0]`), and to the Decoder Task wording "1 if in rewarded corridor".

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The per-trial 0/1 flag is broadcast across the trial's T bins as input row 3, as float32. A guard raises if `isRew` marks more than one distinct wall in a session.

ii.
```python
np.full(T, reward_available[trial])
```
```python
"reward_availability_definition": (
    "All trials whose WallName equals the unique wall identified by any "
    "isRew=True trial; not merely trials where reward was delivered."
),
```

iii. Step 5 mapping: "Encodes reward *availability*, not delivery. No-reward sessions are all 0." Step 10 Check 6 (edge cases) adds: "active missed-reward trials remain availability-positive."

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The per-trial `WallName` string. The 15 native labels are circle1/2/3, leaf1/2/3, leaf1_swap1/2, rock1/2, wood1/2/5, wood1_swap1/2. `stim_id`/`UniqWalls` are deliberately not used.

ii.
```python
walls = np.asarray(beh["WallName"])
...
stim = stimulus_class(str(walls[trial]))
```

iii. Step 4 "Stimulus naming": "Code canonicalizes stimuli differently for each analysis using `stim_id`; figures call natural categories circle/leaf even for some rock/brick mice ... Native `WallName` has 15 exemplar labels across four image families". The AI keeps `WallName` because it is unambiguous per trial, including in swap sessions.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Prefix matching collapses exemplar number and swap suffix to four texture families: `circle*`→0, `leaf*`→1, `rock*`→2, `wood*`→3. An unrecognized name raises. The per-trial class index is broadcast across all T bins as int16 output row 0. Output values are `["circle", "leaf", "rock", "wood"]`.

ii.
```python
STIMULUS_PREFIX_TO_CLASS = {"circle": 0, "leaf": 1, "rock": 2, "wood": 3}

def stimulus_class(wall_name: str) -> int:
    for prefix, value in STIMULUS_PREFIX_TO_CLASS.items():
        if str(wall_name).startswith(prefix):
            return value
    raise ValueError(f"Unrecognized WallName {wall_name!r}")
...
np.full(T, stim, dtype=np.int16)
```

iii. Step 4: "Decoder target says visual stimulus *category*, so collapse suffix/exemplar/swap labels to four raw families: circle, leaf, rock, wood (wood is the native-data name corresponding to the paper's brick family). Preserve native `WallName` mappings in metadata." Step 5 mapping: "Exemplar number and swap do not change visual category."

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. `LickFr` (the fractional imaging-frame number of every lick) together with `LickTrind` (the trial each lick belongs to). Non-finite entries in either array are dropped.

ii.
```python
lick_fr = np.asarray(beh["LickFr"])
lick_tr = np.asarray(beh["LickTrind"])
finite_lick = np.isfinite(lick_fr) & np.isfinite(lick_tr)
lick_fr_int = lick_fr[finite_lick].astype(np.int64)
lick_tr_int = lick_tr[finite_lick].astype(np.int64)
```

iii. Step 5 mapping: "`LickFr`, `LickTrind` → `output[1]` ... `spk_2_firstLick`, `spk_2_cue` cast `LickFr.astype(int)`". The AI adopts the reference's truncation convention and uses `LickTrind` to scope licks to their own trial.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The fractional lick frame is truncated to the integer frame it lands in. For each trial, the licks with `LickTrind == trial` are matched against the trial's selected frames; a bin is 1 if at least one such lick falls in it, else 0 (int16). Multiple licks in one frame still give 1. Licks that land on excluded (stationary or gray-space) frames are not represented, since those bins are not in the converted data.

ii.
```python
trial_lick_frames = lick_fr_int[lick_tr_int == trial]
lick = np.isin(frames, trial_lick_frames).astype(np.int16)
```

iii. Step 5 mapping: "1 where at least one lick from that trial lands on selected frame, else 0; int16 ... Binary even if multiple licks occupy one frame. Licks during excluded stationary/gray frames are outside the converted interval." Resulting dataset-wide rate: 0.963 no-lick / 0.037 lick.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` is already an imaging-frame index, so the flag is on the neural grid by construction; `np.isin(frames, ...)` is evaluated over exactly the frame array used to slice the neural columns, giving the same T and bin-for-bin correspondence.

ii.
```python
ntrial = np.array(activity[:, frames], dtype=np.float32, order="C", copy=True)
lick = np.isin(frames, trial_lick_frames).astype(np.int16)
```

iii. Step 1: the reference's frame-aligned behavior variables (`LickFr` among them) are already sampled on the imaging grid, so indexing by the shared frame array is the alignment.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. `ft_Pos`, the per-imaging-frame VR position in source units (decimeters): 0–40 across the 4 m texture, 40–60 through the 2 m grey space. Cumulative position (`ft_PosCum`) is explicitly rejected.

ii.
```python
position = np.floor(np.asarray(beh["ft_Pos"])[frames] / 10.0)
```
```python
positions = np.asarray(beh["ft_Pos"])[selected]
if not np.all(np.isfinite(positions)) or np.any((positions < 0) | (positions >= 40)):
    raise ValueError(f"{sid}: selected positions are outside [0, 40)")
```

iii. Step 5 mapping: "`get_interpPos_spk` establishes 10 source units = 1 m ... Direct `ft_Pos` avoids cumulative-position offsets observed in 21 boundary samples." Step 2 verified "60 source position units = 6 m; texture 40 units = 4 m; gray 20 units = 2 m in all 89 sessions".

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Divide by 10 (source units per metre) and take the floor, giving an integer metre index; stored int16 as output row 2 with value names `["0-1 m", "1-2 m", "2-3 m", "3-4 m"]`. Because only `ft_CorrSpc` frames are kept, the input to this step is always in [0, 40), which the script asserts.

ii.
```python
position = np.floor(np.asarray(beh["ft_Pos"])[frames] / 10.0)
position = np.clip(position, 0, 3).astype(np.int16)
```
```python
OUTPUT_VALUES[2] = ["0-1 m", "1-2 m", "2-3 m", "3-4 m"]
"position_source_units_per_m": 10.0,
```

iii. Step 5 mapping: "Four equal physical bins: 0–1, 1–2, 2–3, 3–4 m." Step 4 notes the moving/corridor mask "also makes four 1-m position bins meaningful".

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Fixed physical thresholds at 1, 2 and 3 m (i.e. source units 10, 20, 30) — the four equal-length 1-m bins the Decoder Task asks for. Not data-driven. `np.clip(..., 0, 3)` is a defensive guard against a boundary value of exactly 40. The realised distribution is [0.250, 0.249, 0.250, 0.252].

ii.
```python
position = np.clip(np.floor(np.asarray(beh["ft_Pos"])[frames] / 10.0), 0, 3).astype(np.int16)
```

iii. Step 5 mapping: "`floor(position_source_units / 10)` for selected 0–40 source-unit frames; clip defensively to 0–3". The `--show-processing` plot draws the histogram of position with the 1/2/3 m edges overlaid to demonstrate the four equal bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` has one value per imaging frame, so it is already on the neural grid; it is indexed with the same `frames` array as the neural columns, so it shares T bin-for-bin.

ii.
```python
ntrial = np.array(activity[:, frames], dtype=np.float32, order="C", copy=True)
position = np.floor(np.asarray(beh["ft_Pos"])[frames] / 10.0)
```

iii. Frame-indexed behavior streams are aligned to imaging frames by construction (Step 1). The processing plot panel "Corridor-entry alignment" plots position-vs-elapsed-time for the first five trials to show monotone 0→4 m traversals.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. `ft_RunSpeed`, the per-imaging-frame running speed in native units, at the selected frames. Finiteness is asserted during frame selection.

ii.
```python
speed = np.asarray(beh["ft_RunSpeed"])[frames]
```
```python
speed = np.asarray(beh["ft_RunSpeed"])[selected]
if not np.all(np.isfinite(speed)):
    raise ValueError(f"{sid}: selected running speed contains NaN/Inf")
```

iii. Step 5 mapping: "Paper running-speed section uses imaging-frame interpolation; notebook defines `ft_RunSpeed`." The released stream is already interpolated to imaging frames, so nothing further is needed.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A behavior-only prepass concatenates `ft_RunSpeed` over **every selected timepoint of every session being converted** and computes the 25th/50th/75th percentiles once. These three global edges are then applied to every session. Edges for the full run: 12.4224, 25.3526, 40.8546 native units. The edges are stored in metadata. A guard raises if the three edges are not strictly increasing.

ii.
```python
speeds = np.concatenate(speed_chunks).astype(np.float64, copy=False)
thresholds = np.quantile(speeds, [0.25, 0.50, 0.75])
if not np.all(np.diff(thresholds) > 0):
    raise ValueError(f"Non-unique running-speed quartile thresholds: {thresholds}")
```
```python
"running_speed_quartile_edges": [float(x) for x in speed_edges],
"running_speed_units": "native ft_RunSpeed units",
```

iii. Step 5 Key Decision 8: "**Speed binning**: Use global selected-observation quartiles, not per-session quartiles, so labels have a consistent meaning for the shared decoder and each class represents approximately 25% of full data." Note that removing `ft_move == 0` frames first means essentially no ties at zero speed remain (≈0.03% of selected frames vs ≈13% of all in-corridor frames), which is why a value-threshold quantile works here.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize(speed, speed_edges, right=False)` → class 0–3, stored int16 as output row 3, value names `["0-25%", "25-50%", "50-75%", "75-100%"]`. Globally the classes are exactly balanced ([0.250, 0.250, 0.250, 0.250] to within 2e-6), but because the edges are global while running statistics differ strongly between sessions, individual sessions are very skewed — e.g. one session is [0.956, 0.042, 0.002, 0.000] and another [0.066, 0.089, 0.194, 0.651].

ii.
```python
speed_bin = np.digitize(speed, speed_edges, right=False).astype(np.int16)
```

iii. Step 5 Key Decision 8 (above) and Step 9 consistency table: "Speed-bin distribution ... quartiles [0.250,0.250,0.250,0.250] | exact". The `--show-processing` plot overlays the three edges on the session's pooled speed histogram.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is one value per imaging frame, indexed with the same `frames` array as the neural columns, so it is bin-for-bin aligned with the same T.

ii.
```python
ntrial = np.array(activity[:, frames], dtype=np.float32, order="C", copy=True)
speed = np.asarray(beh["ft_RunSpeed"])[frames]
speed_bin = np.digitize(speed, speed_edges, right=False).astype(np.int16)
```

iii. Frame-aligned behavior streams share the imaging grid (Step 1), so the shared frame index is the alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The strategy is mostly **assert-and-abort** rather than repair:
- `validate_behavior_shapes` checks that 15 trial-level fields have length `ntrials`, 6 frame-level fields have length `len(ft)`, `run_pos` is `(ntrials, 60)`, and the corridor geometry is 60/40/20.
- Frames with non-finite `ft_trInd` (the un-assigned frames at the start of a session) are excluded by `np.isfinite(trial_id)`.
- Licks with non-finite `LickFr`/`LickTrind` are dropped.
- The behavior arrays are **not** truncated to the neural frame count; instead the script asserts that the largest selected frame index is inside the neural array.
- Guards raise on: <2 frames in any trial, positions outside [0, 40), non-finite speed, planes with different frame counts, `spks` cell count ≠ `iarea` length, unmapped retinotopy codes, a session with no named-area cell, >1 rewarded wall, NaN/Inf in any converted neural or input array, and a time-dimension mismatch between neural/input/output.
- A missing behavior key falls back to a unique prefix match on the session id.
None of these fired on the full run.

ii.
```python
valid = (np.isfinite(trial_id)
         & np.asarray(beh["ft_CorrSpc"], dtype=bool)
         & (np.asarray(beh["ft_move"]) > 0))
```
```python
finite_lick = np.isfinite(lick_fr) & np.isfinite(lick_tr)
```
```python
max_selected = max(int(x[-1]) for x in frames_by_trial)
if max_selected >= nframes_neural:
    raise ValueError(f"{sid}: selected frame {max_selected} exceeds neural length {nframes_neural}")
```
```python
matches = [k for k in behavior_dict if k.startswith(record["session_id"])]
if len(matches) != 1:
    raise KeyError(f"{record['session_id']}: expected one behavior key, got {matches}")
```

iii. Step 10 Check 6: "Verified fractional `StartFr`/`GrayFr` are not rounded (mask used instead); cumulative-position offsets are avoided by direct `ft_Pos`; active missed-reward trials remain availability-positive; duplicate swap keys produce one physical session; neural arrays one frame shorter than some behavior arrays never lose a selected frame; ... every session has at least two trials and every trial at least 11 selected frames." Step 2 reports "All audited trial fields had length `ntrials`, all frame fields had length `ft`" — the AI treats the dataset as clean and prefers loud failure to silent repair.

## 12-a. What are the most time-consuming steps of the code?

i. Neural file I/O and serialization dominate. The full run took 1,517 s total: 1,394 s of in-memory conversion (reading 405 GB of `spk` files, one session at a time, and copying the retained cells) and 123 s writing the 141.1 GiB pickle. Per-session times printed in the log range from 1.4 s to 52 s and track file size, not trial count. The behavior prepass over all 89 sessions took only 1.44 s. Two non-I/O costs that are non-trivial at this scale are the `np.all(np.isfinite(...))` scan of every retained session matrix and the per-trial `np.isfinite` scan of every neural trial array — together these touch the full ~141 GB of activity roughly twice.

ii.
```python
raw = np.load(spk_path, allow_pickle=True).item()     # 1.5-10 GB per session
...
activity[kept_offset : kept_offset + nkeep] = plane[plane_keep]
...
if not np.all(np.isfinite(activity)):
    raise ValueError(f"{sid}: retained neural activity contains NaN/Inf")
```
```python
with open(args.outpicklefile, "wb", buffering=16 * 1024 * 1024) as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Step 9/10: "This exceeded the sample-derived guideline estimate because cumulative multi-GB neural reads and retained float32 copies scaled less favorably ... no semantics-preserving way exists to avoid reading source activity or writing target activity." Step 10: "the artifact requires reading 405 GB of neural sources and writing 141 GB of exact float32 trial data. The script already avoids full raw concatenation and loads one session at a time."

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI's own position is that nothing meaningful remains ("no avoidable full-matrix copy/I/O remains"), and the per-trial work is already "direct vectorized indexing/stacking". Three loops are nonetheless not vectorized:
- `selected_frames_by_trial` runs one full boolean scan of the selected-frame array per trial (`selected_trial == trial`), i.e. O(ntrials × n_selected) per session, where one `np.argsort`/`np.bincount`+`np.split` pass would do. With up to 789 trials per session this is the largest pure-Python/NumPy cost outside I/O.
- The prepass appends one `np.asarray(beh["ft_RunSpeed"])[idx]` chunk per trial (38,110 small slices and array wrappers) rather than gathering once per session.
- Inside the per-trial loop, `np.asarray(beh["ft"])`, `np.asarray(beh["ft_Pos"])` and `np.asarray(beh["ft_RunSpeed"])` are re-wrapped on every trial instead of being hoisted out of the loop.
All are negligible next to the 405 GB of reads, which is why they do not change the runtime picture.

ii.
```python
frames = [selected[selected_trial == trial] for trial in range(ntrials)]
```
```python
for idx in frames:
    speed_chunks.append(np.asarray(beh["ft_RunSpeed"])[idx])
    trial_lengths.append(len(idx))
```
```python
for trial, frames in enumerate(frames_by_trial):
    ...
    position = np.floor(np.asarray(beh["ft_Pos"])[frames] / 10.0)
    speed = np.asarray(beh["ft_RunSpeed"])[frames]
```

iii. Step 6: "trials use direct vectorized indexing/stacking"; Step 10: "no mismatch ... not due to redundant computation". The AI did not itemize the per-trial mask scan as a vectorizable loop.

## 12-c. What processing does the code repeat multiple times?

i. Every behavior file is loaded and fully re-processed twice — once in `speed_quartile_prepass` and once in `convert`. Within each pass this means `validate_behavior_shapes` and `selected_frames_by_trial` (the O(ntrials × n_selected) mask scan, plus the position/speed validation) are executed twice for all 89 sessions, and `rewarded_wall` logic is computed twice (once for the prepass reward statistics, once for the actual input). The frames computed in the prepass are discarded rather than cached. This is a deliberate design choice: the global speed quartile edges must exist before any neural I/O begins, so the first pass is behavior-only and cheap (1.44 s for all 89 sessions).

ii.
```python
def speed_quartile_prepass(records):
    for group, group_records in records_by_group(records).items():
        behavior_dict = np.load(BEH_ROOT / f"Beh_{group}.npy", allow_pickle=True).item()
        for record in group_records:
            beh = get_behavior(behavior_dict, record)
            validate_behavior_shapes(beh, sid)
            frames = selected_frames_by_trial(beh, sid)     # first computation, discarded
```
```python
def convert(args):
    speed_edges, prepass = speed_quartile_prepass(records)
    for group, group_records in records_by_group(records).items():
        behavior_dict = np.load(BEH_ROOT / f"Beh_{group}.npy", allow_pickle=True).item()
        for record in group_records:
            frames_by_trial = selected_frames_by_trial(beh, sid)   # recomputed
```

iii. Step 5 Key Decision 10 / Step 6: "Behavior prepass is small and computes global speed thresholds before neural I/O", "Behavior files are loaded once per relevant group in **each pass**; speed thresholds are computed before neural loading." The AI justifies the two-pass structure by the need for global quartiles and by keeping the prepass free of neural I/O; it does not otherwise flag the duplicated mask computation.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations exist only for validation or metadata and never reach the decoder:
- `np.all(np.isfinite(activity))` per session plus `np.all(np.isfinite(ntrial))` per trial — a full second pass over ~141 GB of activity that the format verifier also performs.
- `validate_behavior_shapes` checks fields the conversion never reads (`run_pos`, `RewTime`, `RewPos`, `Trial_end_time`, `Gray_space_time`, `SoundPos`, `RewardFr`, `GrayFr`, `EndFr`, `trInd`).
- The prepass computes `reward_deliveries`, `reward_availability_trials`, trial-length statistics and speed min/max solely for the `behavior_prepass` metadata block.
- `load_filtered_neural` returns the raw `iarea` array (and the `Counter` over it) only so `make_processing_plot` can draw a raw-area bar chart; it is discarded in `--full` mode.
- `frames_by_trial` computed in the prepass (see 12-c).
- The trial-level `[-1]` maximum-frame check and the neural/input/output shape assertions are pure guards.
The AI reports none of this as waste; it presents all of it as required verification.

ii.
```python
if not np.all(np.isfinite(activity)):
    raise ValueError(f"{sid}: retained neural activity contains NaN/Inf")
...
if not np.all(np.isfinite(inp)) or not np.all(np.isfinite(ntrial)):
    raise ValueError(f"{sid} trial {trial}: NaN/Inf after conversion")
```
```python
if tuple(np.asarray(beh["run_pos"]).shape) != (ntrials, 60):
    raise ValueError(f"{sid}: unexpected run_pos shape ...")
```
```python
rewarded_deliveries += int(is_rewarded.sum())
...
rewarded_availability += int(np.sum(np.asarray(beh["WallName"]) == reward_walls[0]))
```

iii. The instructions asked to "Validate data shapes and types at each step" and "Include sanity checks", and Step 5's Planned Sanity Checks list structural/curation/statistics checks; Step 10 Check 4 uses exactly these prepass counts ("4,336 reward deliveries, and 4,446 availability-positive trials") as the cross-check against the independently recomputed originals. The AI's stated position (Step 10) is that "no mismatch was found ... not due to redundant computation".
