# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads `beh/Imaging_Exp_info.npy` as the master index, deduplicates physical recordings by `<mouse>_<date>_<block>`, groups sessions by experiment group so each `Beh_<group>.npy` file is read once, and then loads one spike file and one retinotopy file per session.

ii. 
```python
exp_info = np.load(BEH_ROOT / "Imaging_Exp_info.npy", allow_pickle=True).item()
...
for group, group_records in records_by_group(records).items():
    behavior_dict = np.load(
        BEH_ROOT / f"Beh_{group}.npy", allow_pickle=True
    ).item()
    for record in group_records:
        beh = get_behavior(behavior_dict, record)
        activity, region_idx, iarea = load_filtered_neural(record)
```

iii. In `CONVERSION_NOTES.md`, the agent says the dataset should be converted from the 89 unique physical imaging recordings, not the 142 experiment-index appearances, and that behavior files should be read once per group while neural and retinotopy files are read per session.

## 1-b. How are the data split into subjects?

i. Subjects are taken from `db['mname']` in the experiment index, and sessions are assigned to subjects by building a sorted unique subject list and per-session lookup.

ii.
```python
subjects = sorted({r["db"]["mname"] for r in records})
subject_lookup = {name: i for i, name in enumerate(subjects)}
...
subject_idx.append(subject_lookup[record["db"]["mname"]])
```

iii. The notes describe mouse name as the session’s subject identifier and state that the final dataset contains 19 imaging mice.

## 1-c. How are the data split into sessions?

i. A session is defined as one physical recording with id `<mname>_<datexp>_<blk>`. Duplicate appearances of the same physical recording across experiment groups are removed with a `seen` set.

ii.
```python
def session_id(db: dict) -> str:
    return f"{db['mname']}_{db['datexp']}_{db['blk']}"
...
sid = session_id(db)
if sid in seen:
    continue
seen.add(sid)
```

iii. The notes explicitly justify converting each physical recording exactly once because the experiment index contains repeated analysis-group entries for the same neural recording.

## 1-d. How are the data split into trials?

i. Trials are defined by `ft_trInd`, but the agent keeps only frame indices that are simultaneously assigned to the trial, inside the textured corridor, and marked as moving. Each trial is therefore a subset of imaging frames rather than the full contiguous corridor traversal.

ii.
```python
trial_id = np.asarray(beh["ft_trInd"])
valid = (
    np.isfinite(trial_id)
    & np.asarray(beh["ft_CorrSpc"], dtype=bool)
    & (np.asarray(beh["ft_move"]) > 0)
)
selected = np.flatnonzero(valid)
selected_trial = trial_id[selected].astype(np.int64)
frames = [selected[selected_trial == trial] for trial in range(ntrials)]
```

iii. The notes argue that the “reference running criterion” `ft_CorrSpc & (ft_move > 0)` should be used because the paper’s main neural analyses focus on moving textured-corridor frames and because stationary reward-consumption pauses would otherwise dominate the decoder data.

## 1-e. How are trials filtered based on quality controls?

i. The agent does not apply the reference 99th-percentile long-trial filter. Instead, it requires every trial to have at least two retained moving corridor frames and raises an error if any trial violates that condition. In practice it keeps all imaging trials that satisfy the moving-frame mask.

ii.
```python
lengths = np.asarray([len(x) for x in frames])
if np.any(lengths < 2):
    bad = np.flatnonzero(lengths < 2).tolist()
    raise ValueError(f"{sid}: trials with fewer than two valid frames: {bad}")
```

iii. In the notes, the agent says all imaging trials had valid aligned running/corridor frames and therefore “all valid trials” were retained, with behavior-only sessions excluded only because they lack neural input.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data come from the `spks` list in each session’s spike file, and neuron-region assignments come from `iarea` in the session’s retinotopy file.

ii.
```python
with np.load(ret_path, allow_pickle=True) as ret:
    iarea = np.asarray(ret["iarea"])
...
raw = np.load(spk_path, allow_pickle=True).item()
planes = raw["spks"]
```

iii. The notes state that the released `spks` are the Suite2p deconvolved traces used by the paper and that retinotopy provides the raw area codes needed for region curation.

## 2-b. How is the `neural` data processed?

i. The agent keeps the released deconvolved activity values without re-normalization, concatenates the retained cells across planes into a dense matrix, slices per-trial frame windows, and stores the per-trial matrices as `float32`.

ii.
```python
activity = np.empty((int(keep.sum()), nframes), dtype=np.float32)
...
activity[kept_offset : kept_offset + nkeep] = plane[plane_keep]
...
ntrial = np.array(activity[:, frames], dtype=np.float32, order="C", copy=True)
```

iii. The notes justify this by saying the paper’s analyses already use the released deconvolved traces directly, so no new dF/F, smoothing, interpolation, or z-scoring should be added.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered only by named visual-region membership. Raw retinotopy codes are mapped into `V1`, `mHV`, `lHV`, and `aHV`; cells outside those sets are dropped.

ii.
```python
region = np.full(len(iarea), -1, dtype=np.int16)
region[iarea == 8] = 0
region[np.isin(iarea, [0, 1, 2, 9])] = 1
region[np.isin(iarea, [5, 6])] = 2
region[np.isin(iarea, [3, 4])] = 3
keep = region >= 0
```

iii. The notes say Suite2p cell classification was already done upstream and that paper-specific selectivity filters should not be reused as generic quality control for this decoder dataset.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry in the sense that the retained frames are grouped by trial, but the agent keeps only moving frames inside the textured corridor. The neural data are therefore aligned to trial start while omitting stationary parts of the traversal.

ii.
```python
frames_by_trial = selected_frames_by_trial(beh, sid)
...
ntrial = np.array(activity[:, frames], dtype=np.float32, order="C", copy=True)
```

iii. The notes explicitly say the alignment event is trial start / corridor entry, but also defend the choice to remove stationary frames because it matches the paper’s movement-restricted neural analyses.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal bin is the native imaging frame at 3.17 Hz, and no temporal rebinning or resampling is applied.

ii.
```python
IMAGING_RATE_HZ = 3.17
TIME_BIN_MS = 1000.0 / IMAGING_RATE_HZ
...
"time_bin_size": float(TIME_BIN_MS),
```

iii. The notes say the conversion preserves native imaging-frame activity and that adding temporal interpolation or resampling would depart from the paper’s released data.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from trial-level `SoundTime` and frame-level `ft`.

ii.
```python
ft = np.asarray(beh["ft"])[frames]
time_to_cue = (float(beh["SoundTime"][trial]) - ft) * 86400.0
```

iii. The notes justify using exact timestamps rather than reconstructing cue time from `SoundFr`, describing it as the direct aligned timing source.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame, the agent subtracts the frame timestamp from the trial’s cue timestamp and converts MATLAB-day units to seconds.

ii.
```python
time_to_cue = (float(beh["SoundTime"][trial]) - ft) * 86400.0
...
inp = np.vstack(
    [
        time_to_cue,
        np.full(T, record["day_value"]),
        time_since_start,
        np.full(T, reward_available[trial]),
    ]
).astype(np.float32, copy=False)
```

iii. The notes say the task asks for a continuous signed time-to-cue input, so the cue should be represented as elapsed time rather than as a binary event pulse.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the same retained frame timestamps that index the neural columns for that trial.

ii.
```python
ntrial = np.array(activity[:, frames], dtype=np.float32, order="C", copy=True)
ft = np.asarray(beh["ft"])[frames]
time_to_cue = (float(beh["SoundTime"][trial]) - ft) * 86400.0
```

iii. The notes repeatedly say all streams are aligned on the native imaging-frame grid and that exact frame timestamps are carried through the conversion.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The agent derives it from the experiment-descriptor metadata fields `days` when present, otherwise `sess#`, both read from `Imaging_Exp_info.npy`.

ii.
```python
if "days" in db:
    day_value = float(db["days"])
    day_source = "days"
elif "sess#" in db:
    day_value = float(db["sess#"])
    day_source = "sess#"
```

iii. The notes say this is the only released continuous per-session training-day or stage value available in the experiment descriptors, so it should be used directly and recorded with its source field.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. No cross-session counting is done. The chosen session-level numeric field is stored as `day_value` and then broadcast across all retained frames of each trial.

ii.
```python
np.full(T, record["day_value"])
```

iii. The notes describe this as preserving the released training-stage metadata rather than inferring a day index from session ordering.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from trial-level `Trial_start_time` and frame-level `ft`.

ii.
```python
ft = np.asarray(beh["ft"])[frames]
time_since_start = (ft - float(beh["Trial_start_time"][trial])) * 86400.0
```

iii. The notes say the trial-start event is corridor entry and that exact timestamps should be preserved directly.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each retained frame, the frame timestamp is subtracted from the trial’s start timestamp and converted from days to seconds, yielding elapsed time since corridor entry.

ii.
```python
time_since_start = (ft - float(beh["Trial_start_time"][trial])) * 86400.0
```

iii. The notes say this preserves true elapsed time even though stationary frames were removed from the sampled sequence.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It uses the same retained frame array that is used to slice the neural matrix.

ii.
```python
ntrial = np.array(activity[:, frames], dtype=np.float32, order="C", copy=True)
ft = np.asarray(beh["ft"])[frames]
time_since_start = (ft - float(beh["Trial_start_time"][trial])) * 86400.0
```

iii. The notes describe the dataset as keeping native imaging-frame alignment across neural, input, and output streams.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from `isRew` together with `WallName`: `isRew` is used to identify the rewarded wall identity, and then all trials with that wall are marked reward-available.

ii.
```python
def rewarded_wall(beh: dict, sid: str) -> str | None:
    walls = np.asarray(beh["WallName"])
    reward_walls = np.unique(walls[np.asarray(beh["isRew"], dtype=bool)])
    ...
rw_wall = rewarded_wall(beh, sid)
walls = np.asarray(beh["WallName"])
reward_available = (
    np.zeros(len(walls), dtype=np.float32)
    if rw_wall is None
    else (walls == rw_wall).astype(np.float32)
)
```

iii. The notes explicitly distinguish delivered reward from reward availability and justify marking every trial in the rewarded corridor as available, even when active sessions omit delivery on trials without a lick.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The agent identifies the unique rewarded wall from rewarded trials, then broadcasts a trial-level binary flag to every retained frame of that trial. Sessions with no rewarded wall become all zeros.

ii.
```python
reward_available = (
    np.zeros(len(walls), dtype=np.float32)
    if rw_wall is None
    else (walls == rw_wall).astype(np.float32)
)
...
np.full(T, reward_available[trial])
```

iii. The notes justify this as matching the task definition of availability rather than actual water delivery.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`.

ii.
```python
walls = np.asarray(beh["WallName"])
stim = stimulus_class(str(walls[trial]))
```

iii. The notes say `WallName` is the unmasked native trial label and is therefore preferable for constructing the requested visual category.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The wall-name prefix is mapped to one of four categories, `circle`, `leaf`, `rock`, or `wood`, and the category code is repeated across all retained frames of the trial.

ii.
```python
STIMULUS_PREFIX_TO_CLASS = {"circle": 0, "leaf": 1, "rock": 2, "wood": 3}
...
def stimulus_class(wall_name: str) -> int:
    for prefix, value in STIMULUS_PREFIX_TO_CLASS.items():
        if str(wall_name).startswith(prefix):
            return value
...
np.full(T, stim, dtype=np.int16)
```

iii. The notes justify collapsing exemplar numbers and swap suffixes because the decoder target asks for stimulus category rather than exemplar identity.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr` and `LickTrind`.

ii.
```python
lick_fr = np.asarray(beh["LickFr"])
lick_tr = np.asarray(beh["LickTrind"])
finite_lick = np.isfinite(lick_fr) & np.isfinite(lick_tr)
lick_fr_int = lick_fr[finite_lick].astype(np.int64)
lick_tr_int = lick_tr[finite_lick].astype(np.int64)
```

iii. The notes describe this as the paper’s frame-based lick representation, with `LickTrind` used to ensure licks are assigned to the correct trial.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frame indices are truncated to integers, licks are filtered to the current trial, and the output is a binary vector indicating whether any lick fell on each retained frame.

ii.
```python
trial_lick_frames = lick_fr_int[lick_tr_int == trial]
lick = np.isin(frames, trial_lick_frames).astype(np.int16)
```

iii. The notes say this follows the reference-style `LickFr.astype(int)` convention and yields a binary per-frame licking target.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The licking output is computed on the same retained frame indices used to slice the neural data for each trial.

ii.
```python
ntrial = np.array(activity[:, frames], dtype=np.float32, order="C", copy=True)
...
lick = np.isin(frames, trial_lick_frames).astype(np.int16)
```

iii. The notes state that all converted streams remain on the same native imaging-frame grid.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from frame-level `ft_Pos`.

ii.
```python
position = np.floor(np.asarray(beh["ft_Pos"])[frames] / 10.0)
```

iii. The notes say `ft_Pos` already gives imaging-frame-aligned corridor position in the source units used by the dataset.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The position values are divided by 10 source units per meter, floored to integer bins, and clipped to the range 0–3.

ii.
```python
position = np.floor(np.asarray(beh["ft_Pos"])[frames] / 10.0)
position = np.clip(position, 0, 3).astype(np.int16)
```

iii. The notes justify this as the direct conversion from the source 0–40 textured corridor units into four equal 1 m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The categories are fixed meter bins: 0–1 m, 1–2 m, 2–3 m, and 3–4 m.

ii.
```python
OUTPUT_VALUES = [
    ["circle", "leaf", "rock", "wood"],
    ["not licking", "licking"],
    ["0-1 m", "1-2 m", "2-3 m", "3-4 m"],
    ["0-25%", "25-50%", "50-75%", "75-100%"],
]
...
position = np.floor(np.asarray(beh["ft_Pos"])[frames] / 10.0)
position = np.clip(position, 0, 3).astype(np.int16)
```

iii. The notes explicitly say the source data use 10 position units per meter and that four equal physical bins should be used.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is taken from the same retained frame indices used for the neural trial matrix.

ii.
```python
ntrial = np.array(activity[:, frames], dtype=np.float32, order="C", copy=True)
position = np.floor(np.asarray(beh["ft_Pos"])[frames] / 10.0)
```

iii. The notes say `ft_Pos` is already frame-aligned and should be sampled on the same native frame grid as the neural data.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from frame-level `ft_RunSpeed`.

ii.
```python
speed_chunks.append(np.asarray(beh["ft_RunSpeed"])[idx])
...
speed = np.asarray(beh["ft_RunSpeed"])[frames]
```

iii. The notes say the paper and notebook already define running speed on the imaging-frame-aligned behavioral stream.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The agent performs a behavior-only prepass across the converted session set, concatenates all retained running-speed samples, computes global 25th/50th/75th percentile thresholds, and then bins each retained frame’s speed with those thresholds.

ii.
```python
speeds = np.concatenate(speed_chunks).astype(np.float64, copy=False)
thresholds = np.quantile(speeds, [0.25, 0.50, 0.75])
...
speed = np.asarray(beh["ft_RunSpeed"])[frames]
speed_bin = np.digitize(speed, speed_edges, right=False).astype(np.int16)
```

iii. The notes justify global quartiles by saying the decoder’s speed classes should have a consistent meaning across sessions and should represent approximately 25% of the converted data overall.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded with three global quantile edges, producing four categories labeled `0-25%`, `25-50%`, `50-75%`, and `75-100%`.

ii.
```python
OUTPUT_VALUES = [
    ["circle", "leaf", "rock", "wood"],
    ["not licking", "licking"],
    ["0-1 m", "1-2 m", "2-3 m", "3-4 m"],
    ["0-25%", "25-50%", "50-75%", "75-100%"],
]
...
speed_bin = np.digitize(speed, speed_edges, right=False).astype(np.int16)
```

iii. The notes explicitly say speed binning should use global selected-observation quartiles rather than per-session quartiles.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is sampled on the same retained frame indices used for the neural trial matrices.

ii.
```python
ntrial = np.array(activity[:, frames], dtype=np.float32, order="C", copy=True)
speed = np.asarray(beh["ft_RunSpeed"])[frames]
speed_bin = np.digitize(speed, speed_edges, right=False).astype(np.int16)
```

iii. The notes say speed should stay on the native imaging-frame grid and use the same selected frames as the neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent validates trial-level and frame-level array lengths, checks corridor geometry, rejects non-finite selected positions and speeds, ignores non-finite lick entries, falls back to a prefix match if the exact behavior key is missing, and raises errors if selected frames exceed the neural recording length or if a trial has fewer than two retained frames.

ii.
```python
if key in behavior_dict:
    return behavior_dict[key]
matches = [k for k in behavior_dict if k.startswith(record["session_id"])]
...
finite_lick = np.isfinite(lick_fr) & np.isfinite(lick_tr)
...
if max_selected >= nframes_neural:
    raise ValueError(
        f"{sid}: selected frame {max_selected} exceeds neural length {nframes_neural}"
    )
```

iii. The notes present the dataset as very clean and describe these checks as defensive validation rather than heavy imputation or repair.

## 12-a. What are the most time-consuming steps of the code?

i. The code is designed around the expectation that full-session neural I/O dominates runtime. The agent also added a behavior-only prepass for running-speed thresholds, but the notes describe neural loading and session conversion as the main cost.

ii.
```python
raw = np.load(spk_path, allow_pickle=True).item()
planes = raw["spks"]
...
speed_edges, prepass = speed_quartile_prepass(records)
```

iii. In `CONVERSION_NOTES.md`, the agent says the expensive part is loading and converting the multi-gigabyte neural sessions and that the behavior prepass is comparatively cheap.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still uses Python loops over sessions, over trials within each session, and over imaging planes when filling the retained neural matrix. Trial-frame grouping also uses a list comprehension over every trial.

ii.
```python
frames = [selected[selected_trial == trial] for trial in range(ntrials)]
...
for plane in planes:
    ...
for trial, frames in enumerate(frames_by_trial):
    ...
```

iii. The notes emphasize memory and I/O optimizations rather than eliminating these loops, implicitly treating them as acceptable relative to the size of the neural files.

## 12-c. What processing does the code repeat multiple times?

i. The behavior data are processed in two passes: once in `speed_quartile_prepass` to validate sessions and compute global speed thresholds, and again during the full conversion loop. Within the conversion loop, raw behavioral arrays are repeatedly wrapped with `np.asarray(...)` inside per-trial code.

ii.
```python
speed_edges, prepass = speed_quartile_prepass(records)
...
for group, group_records in records_by_group(records).items():
    behavior_dict = np.load(
        BEH_ROOT / f"Beh_{group}.npy", allow_pickle=True
    ).item()
```

iii. The notes justify the extra prepass as the cost of getting global speed quartiles before neural conversion starts.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code performs extra validation and audit work that is not needed by the final decoder arrays, including shape/geometry checks, prepass summary statistics, raw `iarea` retention for diagnostics, timing/ETA reporting, and optional processing plots.

ii.
```python
speed_edges, prepass = speed_quartile_prepass(records)
...
activity, region_idx, iarea = load_filtered_neural(record)
...
if args.show_processing and plot_count < 2:
    plot_path = make_processing_plot(...)
```

iii. The notes frame these as deliberate sanity checks and audit artifacts rather than required downstream data processing.
