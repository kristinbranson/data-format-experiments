# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the master session index from `data/beh/Imaging_Exp_info.npy`, deduplicates recordings by `(mouse, date, block)`, and then loads behavior from the matching `Beh_<experiment_type>.npy` file, spikes from `data/spk/<recording_id>_neural_data.npy`, and retinotopy from `data/retinotopy/<mouse>_<date>_trans.npz`. Unlike the reference, it reloads behavior per session and also makes separate full-dataset behavior passes for category discovery and speed-bin computation.

ii. ```python
exp_info = np.load(
    data_root / "beh" / "Imaging_Exp_info.npy", allow_pickle=True
).item()
```
```python
records = np.load(desc["behavior_file"], allow_pickle=True).item()
return records[_behavior_key(desc["entry"], records)]
```
```python
spk_obj = np.load(spk_path, allow_pickle=True).item()
ret = np.load(
    DATA_ROOT / "retinotopy" /
    f"{desc['entry']['mname']}_{desc['entry']['datexp']}_trans.npz"
)
```

iii. The clearest explicit rationale is in trajectory step 21: the agent says it found 89 unique recordings reused across paper analyses and therefore would deduplicate by recording ID while retaining the paper’s visual-cortex mask and frame rule. The trajectory does not separately justify the repeated behavior-file passes.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are taken from `entry["mname"]`. The final `subjects` list is the sorted unique mouse names, and each session gets `subject_idx` from that lookup.

ii. ```python
subjects = sorted({str(x["entry"]["mname"]) for x in sessions})
subject_to_id = {name: i for i, name in enumerate(subjects)}
...
subject_idx.append(subject_to_id[str(desc["entry"]["mname"])])
```

iii. The trajectory does not contain a separate justification beyond the session-discovery pass. This choice follows directly from the master index structure the agent inspected before step 21.

## 1-c. How are the data split into sessions?

i. A session is defined as one physical recording identified by `mname`, `datexp`, and `blk`. The agent builds `recording_id = "<mouse>_<date>_<block>"` and keeps only the first occurrence when the same recording appears under multiple experiment types.

ii. ```python
def _recording_id(entry: dict) -> str:
    return f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
```
```python
rid = _recording_id(entry)
if rid not in sessions:
    sessions[rid] = {
        "recording_id": rid,
        "experiment_type": experiment_type,
        "entry": entry,
        "behavior_file": behavior_file,
    }
```

iii. Step 21 explicitly states that the experiment index reuses the same recording for different analyses, so the converter will deduplicate by recording ID.

## 1-d. How are the data split into trials?

i. Trials are split by `ft_trInd`, but only on frames that also satisfy the agent’s `retained_frame_mask`, which requires both `ft_move > 0` and `ft_CorrSpc`. This means a trial is not stored as all corridor frames from entry to exit/gray space; it is stored only as moving textured-corridor frames for that trial.

ii. ```python
def retained_frame_mask(beh: dict, nframes: int) -> np.ndarray:
    return (
        np.asarray(beh["ft_move"][:nframes]) > 0
    ) & np.asarray(beh["ft_CorrSpc"][:nframes], dtype=bool)
```
```python
for trial in range(ntrials):
    frames = np.flatnonzero(valid & (trial_stamp == trial))
    if len(frames) == 0:
        dropped_empty += 1
        continue
```

iii. Step 21 says the agent intended to retain the paper’s “running-within-textured-corridor frame rule.” Step 10 shows its reasoning tension: it recognized the decoder is trial-start/time aligned, but still chose to keep the moving-frame rule from the source analyses.

## 1-e. How are trials filtered based on quality controls?

i. The only trial-level filter in the agent code is dropping trials that have zero retained frames after the moving-textured-corridor mask. It does not implement the reference solution’s long-trial outlier removal.

ii. ```python
for trial in range(ntrials):
    frames = np.flatnonzero(valid & (trial_stamp == trial))
    if len(frames) == 0:
        dropped_empty += 1
        continue
```

iii. Step 21 explicitly says the agent would “preserve all trials,” and step 46 reports that it was preserving every nonempty source trial after the paper-defined frame mask.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity comes from the Suite2p deconvolved `spks` arrays in each session’s spike file. The neuron-to-region labels come from `iarea` in the matching retinotopy file.

ii. ```python
spk_obj = np.load(spk_path, allow_pickle=True).item()
planes = spk_obj["spks"]
```
```python
ret = np.load(
    DATA_ROOT / "retinotopy" /
    f"{desc['entry']['mname']}_{desc['entry']['datexp']}_trans.npz"
)
area_ids = np.asarray(ret["iarea"]).astype(np.int16, copy=False)
```

iii. Step 21 states that the converter would retain the paper’s visual-cortex area mask. The trajectory otherwise treats the neural source arrays as direct inputs rather than something to derive indirectly.

## 2-b. How is the `neural` data processed?

i. The agent keeps the deconvolved traces on the native imaging-frame grid, trims all planes to the minimum plane length in the session, filters neurons by retinotopy, and copies selected frame columns plane by plane into a per-trial matrix. It stores the resulting per-trial neural arrays as `float32`.

ii. ```python
planes = spk_obj["spks"]
nframes = min(a.shape[1] for a in planes)
planes = [a[:, :nframes] for a in planes]
```
```python
neural = np.empty((n_neurons, T), dtype=np.float32)
for plane, keep in zip(planes, plane_keep):
    n = len(keep)
    neural[row:row + n] = plane[np.ix_(keep, frames)]
```

iii. The trajectory does not give a separate free-text justification for `float32` storage or plane-wise copying. Step 27 does say the artifact intentionally preserves “single-neuron, single-frame activity,” which matches the no-rebinning/no-averaging choice.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered only by retinotopy. The agent keeps neurons whose `iarea` maps to V1, mHV, lHV, or aHV and drops unlabeled/out-of-area neurons.

ii. ```python
AREA_ID_TO_REGION = {
    8: 0,
    0: 1, 1: 1, 2: 1, 9: 1,
    5: 2, 6: 2,
    3: 3, 4: 3,
}
```
```python
keep = np.isin(ids, np.fromiter(AREA_ID_TO_REGION, dtype=np.int16))
local = np.flatnonzero(keep)
```

iii. Step 21 says the converter will retain the paper’s visual-cortex area masks, and steps 32 and 38 repeat that it is excluding only the retinotopy labels treated as outside or unassigned visual cortex.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The metadata says trials are aligned to corridor entry, but the actual neural arrays begin at the first retained frame satisfying `ft_move > 0` and `ft_CorrSpc` within each trial. Because stationary corridor frames are removed, the stored neural data are aligned to a movement-filtered subset of the trial rather than all frames from corridor entry onward.

ii. ```python
valid = retained_frame_mask(beh, nframes)
trial_stamp = np.asarray(beh["ft_trInd"][:nframes])
...
frames = np.flatnonzero(valid & (trial_stamp == trial))
```
```python
"temporal_alignment_event": "entry into the 4 m textured corridor",
"off_start": 0.0,
"off_end": None,
```

iii. Step 10 shows the agent knew the decoder task was trial-start/time aligned, but step 21 says it would still keep the paper’s running-within-textured-corridor frame rule.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. Each column is one original imaging frame, but the metadata hard-codes the nominal bin size as `1000 / 3.0 = 333.33 ms` rather than the paper/reference value `1000 / 3.17`.

ii. ```python
# Imaging was nominally approximately 3 Hz in the paper. Exact frame
# timestamps drive event variables; observed timing is also recorded.
"time_bin_size": 1000.0 / 3.0,
```
```python
frame_times = np.asarray(beh["ft"][:nframes], dtype=np.float64)
```

iii. The trajectory does not give a separate defense of the `3.0 Hz` metadata. Step 27 and step 68 emphasize preserving single-frame activity, which supports the “no rebinning” part of the decision.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The agent derives this input from `SoundTime` for each trial and the per-frame timestamps `ft`. It does not use `SoundFr` plus interpolation as in the reference.

ii. ```python
frame_times = np.asarray(beh["ft"][:nframes], dtype=np.float64)
...
to_cue_s = (float(beh["SoundTime"][trial]) - frame_times[frames]) * 86_400.0
```

iii. Step 10 says the agent wanted the conversion to remain “time-based” and avoid turning position interpolation into time, which explains why it preferred stored absolute timestamps over frame-index interpolation.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame in a trial, the agent subtracts the frame timestamp from the trial’s `SoundTime` and converts MATLAB days to seconds. The signal is therefore continuous and time-varying, positive before the cue and negative after it.

ii. ```python
to_cue_s = (float(beh["SoundTime"][trial]) - frame_times[frames]) * 86_400.0
...
inputs[0] = to_cue_s
```

iii. The justification again comes from step 10: the agent explicitly wanted to preserve a time-based representation rather than reconstruct timing indirectly from positional information.

## 3-c. How is `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated on exactly the same retained frame indices `frames` used to build each trial’s neural matrix, so it matches the neural arrays one column for one column. That alignment is internal to the agent’s movement-filtered trial definition.

ii. ```python
frames = np.flatnonzero(valid & (trial_stamp == trial))
...
to_cue_s = (float(beh["SoundTime"][trial]) - frame_times[frames]) * 86_400.0
...
neural[row:row + n] = plane[np.ix_(keep, frames)]
```

iii. The trajectory does not single this variable out, but step 10’s “time-based” reasoning and step 21’s retained-frame rule together explain why the cue timing is aligned on the same filtered frame subset.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The agent derives day-of-training from `entry["sess#"]` in the master experiment index. If `sess#` is missing, it falls back to `experiment_type`, treating `"after"` sessions as day `1.0` and other sessions as day `0.0`.

ii. ```python
def session_day(desc: dict) -> float:
    value = desc["entry"].get("sess#")
    if value is not None and np.isfinite(value):
        return float(value)
    typ = desc["experiment_type"]
    return 1.0 if "after" in typ else 0.0
```

iii. The trajectory shows that the agent inspected the `sess#` field in the experiment index during dataset exploration. It does not contain a separate free-text justification for this rule; the rationale is only documented in the code comments around `session_day()`.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The value is a single per-session scalar from `sess#` or the `before/after` fallback. That scalar is then broadcast across all time bins of every trial in the session.

ii. ```python
day = session_day(desc)
...
inputs[1] = day
```

iii. No explicit trajectory message justifies the broadcast or the `0/1` fallback. The only explanation is implicit: the code comments say the agent wanted to avoid inventing calendar-day estimates when `sess#` is missing.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. The agent derives this input from `Trial_start_time` and `ft`. It does not use `StartFr` plus interpolation as the reference does.

ii. ```python
frame_times = np.asarray(beh["ft"][:nframes], dtype=np.float64)
...
elapsed_s = (frame_times[frames] - float(beh["Trial_start_time"][trial])) * 86_400.0
```

iii. Step 10 gives the rationale: the agent preferred explicit stored timestamps so the decoder inputs stayed on a time axis rather than being reconstructed from position-like surrogates.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each retained frame, the agent subtracts the trial’s `Trial_start_time` from the frame timestamp and converts MATLAB days to seconds. The result is a continuous per-frame elapsed-time trace.

ii. ```python
elapsed_s = (frame_times[frames] - float(beh["Trial_start_time"][trial])) * 86_400.0
...
inputs[2] = elapsed_s
```

iii. As with time-to-cue, the only explicit justification is step 10’s statement that the conversion should remain time-based.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed on the same retained frame indices `frames` used for the neural columns of that trial. Internally it is exactly aligned to the neural data the agent stores, but that shared frame set already excludes stationary corridor frames.

ii. ```python
frames = np.flatnonzero(valid & (trial_stamp == trial))
elapsed_s = (frame_times[frames] - float(beh["Trial_start_time"][trial])) * 86_400.0
...
neural[row:row + n] = plane[np.ix_(keep, frames)]
```

iii. The trajectory justification is the same combination as above: step 10 favors time-based signals, and step 21 fixes the retained frame set to moving textured-corridor frames.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Reward availability is derived directly from the per-trial `isRew` flag.

ii. ```python
inputs[3] = float(bool(beh["isRew"][trial]))
```

iii. The trajectory does not discuss this variable separately. It is treated as a direct trial-level field from behavior.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The agent converts `isRew[trial]` to a Boolean/float and broadcasts that same value across all time bins in the trial.

ii. ```python
inputs = np.empty((4, T), dtype=np.float32)
...
inputs[3] = float(bool(beh["isRew"][trial]))
```

iii. No explicit trajectory justification is given. This is a straightforward per-trial broadcast choice.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The output is derived from `WallName` for each trial.

ii. ```python
category = visual_category(beh["WallName"][trial])
outputs[0] = category_to_id[category]
```

iii. The trajectory does not provide a separate free-text rationale. During exploration the agent inspected the behavior fields and later used `WallName` consistently in the converter.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The agent collapses each `WallName` to an alphabetic prefix such as `circle`, `leaf`, `rock`, or `wood` using a regex, discovers all categories present across sessions, sorts them, maps them to integer IDs, and broadcasts the trial’s category across all retained time bins.

ii. ```python
def visual_category(name: str) -> str:
    match = re.match(r"[A-Za-z]+", str(name))
    if match is None:
        raise ValueError(f"Cannot derive visual category from {name!r}")
    return match.group(0).lower()
```
```python
categories.update(visual_category(x) for x in beh["WallName"])
category_names = sorted(categories)
category_to_id = {name: i for i, name in enumerate(category_names)}
```

iii. The trajectory does not justify the regex rule explicitly. The choice appears to come from inspection of the `WallName` patterns and from the goal of collapsing crop/version suffixes into broad categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `LickFr` and also filtered by `LickTrind` so the agent only considers licks belonging to the current trial.

ii. ```python
lick_frame = np.rint(np.asarray(beh["LickFr"], dtype=np.float64)).astype(np.int64)
lick_trial = np.asarray(beh["LickTrind"], dtype=np.float64)
...
trial_licks = lick_frame[lick_trial == trial]
```

iii. The trajectory includes exploration of `LickFr` and related fields, but no separate free-text justification. The code comments later explain the intended alignment rule.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The agent rounds each fractional `LickFr` to the nearest imaging frame, selects only licks assigned to the current trial, and marks retained trial frames whose indices appear in that trial’s lick-frame set. If a lick lands on a frame removed by the moving-frame mask, it is effectively discarded.

ii. ```python
lick_frame = np.rint(np.asarray(beh["LickFr"], dtype=np.float64)).astype(np.int64)
...
trial_licks = lick_frame[lick_trial == trial]
outputs[1] = np.isin(frames, trial_licks).astype(np.int16)
```

iii. The trajectory does not give a standalone justification for rounding rather than truncating. The only explicit rationale is in the code comment that licks are assigned to the nearest original imaging frame.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned by asking whether each retained neural frame index in `frames` is present in the rounded lick-frame set for that trial. This produces a lick vector with the same length as the neural trial, but it removes licks on non-retained stationary corridor frames.

ii. ```python
frames = np.flatnonzero(valid & (trial_stamp == trial))
trial_licks = lick_frame[lick_trial == trial]
outputs[1] = np.isin(frames, trial_licks).astype(np.int16)
```

iii. The trajectory justification is indirect: step 21 fixes the frame mask, and the code comment says licks on removed frames are removed as well.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is derived from `ft_Pos`, the per-frame corridor position values.

ii. ```python
frame_pos = np.asarray(beh["ft_Pos"][:nframes], dtype=np.float64)
```
```python
outputs[2] = np.clip(np.floor(frame_pos[frames] / 10.0), 0, 3).astype(np.int16)
```

iii. The trajectory does not discuss this variable in a separate message. The code follows the documented decimetre-to-metre-bin conversion.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The agent divides `ft_Pos` by 10, floors it, and clips the result into four integer bins `0..3`, corresponding to four 1 m spatial bins.

ii. ```python
outputs[2] = np.clip(np.floor(frame_pos[frames] / 10.0), 0, 3).astype(np.int16)
```

iii. The trajectory does not contain a separate rationale. The implementation reflects the task requirement of four equal-length 1 m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The categories are thresholded by fixed spatial boundaries at 0, 1, 2, 3, and 4 meters, implemented as decimetre thresholds every 10 units and clipped into bins 0 to 3.

ii. ```python
"output_values": [
    category_names,
    ["not_licking", "licking"],
    ["0-1 m", "1-2 m", "2-3 m", "3-4 m"],
    ["Q1 (slowest)", "Q2", "Q3", "Q4 (fastest)"],
],
```
```python
outputs[2] = np.clip(np.floor(frame_pos[frames] / 10.0), 0, 3).astype(np.int16)
```

iii. No explicit trajectory justification is given beyond satisfying the decoder-task discretization requirement.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. The agent samples `ft_Pos` on the same retained frame indices `frames` used for the neural arrays, so the position output is column-aligned with neural data. As elsewhere, that means alignment is to the movement-filtered trial subset rather than all corridor frames.

ii. ```python
frames = np.flatnonzero(valid & (trial_stamp == trial))
...
outputs[2] = np.clip(np.floor(frame_pos[frames] / 10.0), 0, 3).astype(np.int16)
```

iii. The trajectory does not single out position alignment, but step 21’s retained-frame rule determines it.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ft_RunSpeed`, the per-frame running-speed trace.

ii. ```python
frame_speed = np.asarray(beh["ft_RunSpeed"][:nframes], dtype=np.float64)
```
```python
chunks.append(np.asarray(beh["ft_RunSpeed"][:nframes][valid], dtype=np.float32))
```

iii. Step 21 explicitly says the agent would compute speed quartiles globally over the retained frames.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The agent performs a first pass over all sessions, collects `ft_RunSpeed` values from frames that satisfy `ft_move > 0` and `ft_CorrSpc`, computes global 25th/50th/75th percentile edges with `np.quantile`, and then bins each retained frame’s speed against those global thresholds with `np.digitize`.

ii. ```python
valid = retained_frame_mask(beh, nframes)
chunks.append(np.asarray(beh["ft_RunSpeed"][:nframes][valid], dtype=np.float32))
...
edges = np.quantile(speeds, [0.25, 0.50, 0.75])
```
```python
outputs[3] = np.digitize(frame_speed[frames], speed_edges, right=False).astype(np.int16)
```

iii. Step 21 explicitly justifies this: “Speed quartiles will be computed globally over exactly those retained frames.” Step 27 repeats the resulting global edges during the dry run.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded with three global numeric cut points, yielding four bins labeled Q1 to Q4. The thresholds are dataset-level percentile values, not per-session rank quartiles.

ii. ```python
edges = np.quantile(speeds, [0.25, 0.50, 0.75])
```
```python
outputs[3] = np.digitize(frame_speed[frames], speed_edges, right=False).astype(np.int16)
```

iii. Step 21 provides the explicit rationale for global quartiles over retained decoder samples.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. The speed category is computed for the same retained frame indices `frames` used in the neural trial, so it is column-aligned with the stored neural arrays. Like the other time-varying streams, it inherits the movement-filtered trial window.

ii. ```python
frames = np.flatnonzero(valid & (trial_stamp == trial))
...
outputs[3] = np.digitize(frame_speed[frames], speed_edges, right=False).astype(np.int16)
```

iii. The trajectory justification is again step 21’s retained-frame rule plus the explicit global speed-binning rule.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent trims all spike planes to the shortest plane length in the session, assumes one extra terminal behavior timestamp in `ft` during the speed pre-pass, drops trials with zero retained frames, and falls back from missing `sess#` values to a stage-based `0/1` rule. Licks outside the retained frame set are implicitly ignored by the `np.isin` alignment.

ii. ```python
nframes = min(a.shape[1] for a in planes)
planes = [a[:, :nframes] for a in planes]
```
```python
nframes = len(beh["ft"]) - 1
valid = retained_frame_mask(beh, nframes)
```
```python
if len(frames) == 0:
    dropped_empty += 1
    continue
```

iii. The trajectory does not present a unified “data mistakes” policy. Step 10 and step 21 explain the frame-selection philosophy; the missing-`sess#` and terminal-sample handling are justified only by code comments.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive work is reading the large per-session spike files and copying selected neurons/frames into per-trial arrays. The agent also adds a full-dataset speed pre-pass and a separate full-dataset category-discovery pass over behavior files.

ii. ```python
spk_obj = np.load(spk_path, allow_pickle=True).item()
planes = spk_obj["spks"]
```
```python
for i, desc in enumerate(sessions):
    beh = load_behavior(desc)
    ...
    chunks.append(np.asarray(beh["ft_RunSpeed"][:nframes][valid], dtype=np.float32))
```

iii. The trajectory emphasizes the artifact’s size and long full-conversion runtime in steps 27, 32, 53, and 62. It does not separately analyze the cost centers, but the repeated session passes are visible in the code.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop recomputes `np.flatnonzero(valid & (trial_stamp == trial))` once for each trial, and the neural extraction copies one plane at a time for every trial. The separate category and speed passes over all sessions also repeat scalar Python-level loops that could be consolidated.

ii. ```python
for trial in range(ntrials):
    frames = np.flatnonzero(valid & (trial_stamp == trial))
```
```python
for plane, keep in zip(planes, plane_keep):
    n = len(keep)
    neural[row:row + n] = plane[np.ix_(keep, frames)]
```

iii. The trajectory does not discuss vectorization opportunities. This is an inference from the implementation the agent chose.

## 12-c. What processing does the code repeat multiple times?

i. It loads behavior repeatedly: once for category discovery, once for global speed-edge estimation, and again inside `convert_session` for the actual conversion. It also recomputes the retained-frame mask independently in the speed pass and in the session-conversion pass.

ii. ```python
for desc in sessions:
    beh = load_behavior(desc)
    categories.update(visual_category(x) for x in beh["WallName"])
```
```python
speed_edges, timing_stats = compute_speed_edges(sessions)
```
```python
def convert_session(...):
    beh = load_behavior(desc)
    ...
    valid = retained_frame_mask(beh, nframes)
```

iii. The trajectory does not justify these repeated passes. They are a consequence of how the agent decomposed the converter.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes timing summary statistics for metadata, discovers stimulus categories in a separate pre-pass, and stores extensive `session_info` metadata that the downstream decoder does not use. It also keeps neural trials in `float32`, which substantially inflates the serialized artifact without helping the downstream categorical decoding task.

ii. ```python
dt_all = np.concatenate(frame_dts_ms)
stats = {
    "speed_quartile_edges_cm_s": edges.tolist(),
    "observed_median_frame_interval_ms": float(np.median(dt_all)),
    "observed_frame_interval_iqr_ms": np.quantile(dt_all, [0.25, 0.75]).tolist(),
}
```
```python
"session_info": session_info,
```
```python
neural = np.empty((n_neurons, T), dtype=np.float32)
```

iii. The trajectory does not defend these extras as necessary for the decoder. The only explicit trajectory theme is preserving fine-grained activity, not minimizing work or output size.
