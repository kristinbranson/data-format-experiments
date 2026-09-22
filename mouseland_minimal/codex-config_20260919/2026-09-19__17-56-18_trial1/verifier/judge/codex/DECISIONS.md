# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads `Imaging_Exp_info.npy`, deduplicates physical recordings by mouse/date/block, loads the corresponding behavior dictionary, each session's plane-wise spike file, and its retinotopy file. It makes two behavior passes (category discovery and speed thresholds) before converting all sessions.

ii.
```python
exp_info = np.load(data_root / "beh" / "Imaging_Exp_info.npy", allow_pickle=True).item()
for experiment_type, entries in exp_info.items():
    ...
spk_obj = np.load(spk_path, allow_pickle=True).item()
ret = np.load(DATA_ROOT / "retinotopy" / f"{desc['entry']['mname']}_{desc['entry']['datexp']}_trans.npz")
```

iii. The trajectory says a mouse/date/block is a physical recording and that duplicate experiment-table entries must not duplicate sessions; it also identifies behavior, `spks`, and `iarea` as the required sources.

## 1-b. How are the data split into subjects?

i. Subjects are sorted unique `mname` values, and every retained session receives the corresponding integer `subject_idx`.

ii.
```python
subjects = sorted({str(x["entry"]["mname"]) for x in sessions})
subject_idx.append(subject_to_id[str(desc["entry"]["mname"])])
```

iii. The AI relied on the master index's explicit mouse identifier and verified that the result contains 19 mice.

## 1-c. How are the data split into sessions?

i. A session is the first occurrence of each unique `mname_datexp_blk`, retained in repository iteration order. A swap-stimulus suffix is used only to resolve the behavior key.

ii.
```python
rid = _recording_id(entry)
if rid not in sessions:
    sessions[rid] = {...}
```

iii. The trajectory notes that the experiment table repeats recordings for multiple analyses, so deduplication represents each physical recording once.

## 1-d. How are the data split into trials?

i. For trial numbers `0..ntrials-1`, frames are selected where `ft_trInd == trial`, `ft_CorrSpc` is true, and `ft_move > 0`. Thus stationary frames inside a trial are removed, potentially making a trial temporally discontinuous.

ii.
```python
valid = retained_frame_mask(beh, nframes)
frames = np.flatnonzero(valid & (trial_stamp == trial))
```

iii. The AI interpreted the paper statement that only running timepoints were analyzed and the repository's `fr_valid = VRmove & isCorridor` as the governing frame selection.

## 1-e. How are trials filtered based on quality controls?

i. Only trials with no retained moving-corridor frames are dropped. There is no long-trial/outlier filter; sessions with fewer than two retained trials cause an error.

ii.
```python
if len(frames) == 0:
    dropped_empty += 1
    continue
if len(neural) < 2:
    raise ValueError(...)
```

iii. The trajectory focused on matching the paper's running-frame mask and reported all 38,110 source trials survived; it did not identify or justify a long-trial quality-control rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from plane-wise Suite2p `spks`; `iarea` supplies the retinotopic label used to select and label neurons.

ii.
```python
planes = spk_obj["spks"]
area_ids = np.asarray(ret["iarea"])
```

iii. The AI identified `spks` as the paper's non-negative deconvolved fluorescence signal and `iarea` as matching plane-concatenated neuron order.

## 2-b. How is the `neural` data processed?

i. Planes are truncated to their common minimum frame count. Selected rows from each plane are copied in plane order for each trial and stored as `float32`; there is no further signal transform, padding, or resampling.

ii.
```python
nframes = min(a.shape[1] for a in planes)
planes = [a[:, :nframes] for a in planes]
neural = np.empty((n_neurons, T), dtype=np.float32)
neural[row:row + n] = plane[np.ix_(keep, frames)]
```

iii. The AI reasoned that deconvolution had already been performed and designed the plane-wise copy to avoid a very large full concatenation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only neurons whose `iarea` maps to V1, mHV, lHV, or aHV are retained; unassigned and area-7 neurons are excluded. Neural timepoints also inherit the moving-corridor frame mask.

ii.
```python
keep = np.isin(ids, np.fromiter(AREA_ID_TO_REGION, dtype=np.int16))
valid = retained_frame_mask(beh, nframes)
```

iii. The AI says the area grouping exactly follows `neu_area_ID()` and the frame mask follows the paper analysis code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each variable-length trial begins with its first retained moving frame in the textured corridor, treated as corridor entry, and ends at its last such frame. No common window or padding is applied.

ii.
```python
frames = np.flatnonzero(valid & (trial_stamp == trial))
neural[row:row + n] = plane[np.ix_(keep, frames)]
```

iii. The AI viewed `ft_CorrSpc` as corridor alignment and set metadata to “entry into the 4 m textured corridor”; the running mask was chosen to reproduce paper analyses.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One original imaging frame is one bin; no temporal rebinning occurs. Metadata reports `1000/3 = 333.33 ms`, while observed timestamps are only recorded as diagnostic statistics.

ii.
```python
"time_bin_size": 1000.0 / 3.0,
```

iii. The AI described imaging as nominally approximately 3 Hz and kept the native frame clock.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived directly from per-trial `SoundTime` and per-frame MATLAB datenum `ft`.

ii.
```python
to_cue_s = (float(beh["SoundTime"][trial]) - frame_times[frames]) * 86_400.0
```

iii. The AI inspected the behavior fields and chose the direct timestamps as the most precise common clock.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Frame time is subtracted from cue time and converted from days to seconds, so values are positive before and negative after the cue.

ii.
```python
inputs[0] = to_cue_s
```

iii. The trajectory explicitly reasons that direct timestamps avoid unnecessary interpolation from frame indices.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at exactly the `frames` used as neural columns.

ii.
```python
frame_times = np.asarray(beh["ft"][:nframes])
to_cue_s = ... frame_times[frames]
```

iii. The AI states all behavioral events are aligned to the original imaging-frame clock.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It uses `sess#` from the experiment-index entry; if missing, it substitutes 1 for experiment types containing “after” and 0 otherwise.

ii.
```python
value = desc["entry"].get("sess#")
return 1.0 if "after" in typ else 0.0
```

iii. The AI preferred the repository's explicit session/day field and said the fallback avoids inventing calendar days when the recording date is not training start.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The scalar is cast to float and broadcast across every frame of the trial; the fallback collapses missing before/after stages to 0/1.

ii.
```python
day = session_day(desc)
inputs[1] = day
```

iii. The rationale is that `sess#` is intended to encode session/day and stage is the only known information for missing values.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `Trial_start_time[trial]` and the retained per-frame `ft` timestamps.

ii.
```python
elapsed_s = (frame_times[frames] - float(beh["Trial_start_time"][trial])) * 86_400.0
```

iii. The AI selected direct timestamps on the imaging clock.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Trial-start datenum is subtracted from each frame datenum and converted to seconds.

ii.
```python
inputs[2] = elapsed_s
```

iii. The trajectory considers this a direct, precision-preserving time difference.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Values are computed only at the exact retained neural frame indices.

ii.
```python
elapsed_s = (frame_times[frames] - ...) * 86_400.0
```

iii. The AI's stated alignment principle is to use the original imaging-frame clock for every stream.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes from the per-trial `isRew` field.

ii.
```python
inputs[3] = float(bool(beh["isRew"][trial]))
```

iii. The AI treated `isRew` as the direct rewarded-corridor indicator requested by the task.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The value is converted to Boolean/float and broadcast over all retained frames of the trial.

ii.
```python
inputs[3] = float(bool(beh["isRew"][trial]))
```

iii. No additional processing was considered necessary for a per-trial binary variable.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from each trial's `WallName`.

ii.
```python
category = visual_category(beh["WallName"][trial])
```

iii. The AI chose `WallName` because it directly names the corridor texture, including swap/crop variants.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The leading alphabetic substring is lowercased, globally sorted into category names, mapped to a zero-based ID, and broadcast over the trial.

ii.
```python
match = re.match(r"[A-Za-z]+", str(name))
return match.group(0).lower()
outputs[0] = category_to_id[category]
```

iii. The AI intended to collapse crop/version and swap suffixes while retaining the four base texture categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It uses `LickFr` (lick frame positions) and `LickTrind` (lick trial assignment).

ii.
```python
lick_frame = np.rint(np.asarray(beh["LickFr"])).astype(np.int64)
lick_trial = np.asarray(beh["LickTrind"])
```

iii. The AI identified these as the event frame and trial fields needed to avoid assigning a lick to the wrong trial.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Fractional lick frames are rounded to the nearest integer frame; a retained frame is 1 if its index occurs among that trial's lick frames, else 0.

ii.
```python
trial_licks = lick_frame[lick_trial == trial]
outputs[1] = np.isin(frames, trial_licks).astype(np.int16)
```

iii. The AI says nearest-frame rounding represents the event on the imaging grid and that licks on filtered-out frames should also be removed.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick indices are compared with the same retained `frames` used for neural columns.

ii.
```python
outputs[1] = np.isin(frames, trial_licks).astype(np.int16)
```

iii. The AI explicitly documents “nearest original imaging frame”; events on discarded stationary frames are discarded.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It comes from per-frame `ft_Pos`.

ii.
```python
frame_pos = np.asarray(beh["ft_Pos"][:nframes])
```

iii. The AI found this to be the position stream already sampled on imaging frames.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Positions in decimeters are divided into 10-dm (1-m) intervals, floored, clipped to 0–3, and cast to integers.

ii.
```python
outputs[2] = np.clip(np.floor(frame_pos[frames] / 10.0), 0, 3).astype(np.int16)
```

iii. The AI recognized the source unit as decimeters and followed the requested four equal 1-m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are 10, 20, and 30 dm, producing `[0,10)`, `[10,20)`, `[20,30)`, and `[30,∞)` after clipping.

ii.
```python
np.clip(np.floor(frame_pos[frames] / 10.0), 0, 3)
```

iii. This directly implements four equal-length spatial categories over the 4-m textured corridor.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is indexed by the identical retained neural `frames`.

ii.
```python
outputs[2] = ... frame_pos[frames] ...
```

iii. The AI notes `ft_Pos` is already on the original imaging-frame grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from `ft_RunSpeed` at retained frames.

ii.
```python
frame_speed = np.asarray(beh["ft_RunSpeed"][:nframes])
```

iii. The AI treated this as the direct imaging-frame speed stream.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A preliminary pass pools speeds from every moving-corridor frame across all sessions, calculates global 25th/50th/75th value quantiles, then digitizes each trial using those three edges.

ii.
```python
speeds = np.concatenate(chunks)
edges = np.quantile(speeds, [0.25, 0.50, 0.75])
outputs[3] = np.digitize(frame_speed[frames], speed_edges, right=False)
```

iii. The AI interpreted “each corresponding to 25% of the data” globally and wanted consistent numeric thresholds across sessions.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The categories are defined by the three global quantile values, with equality assigned to the upper bin (`right=False`). Ties are not rank-split.

ii.
```python
np.digitize(frame_speed[frames], speed_edges, right=False).astype(np.int16)
```

iii. The trajectory reports the resulting global bins as essentially 25% each after filtering.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed labels are computed from `frame_speed[frames]`, the exact neural-column indices.

ii.
```python
outputs[3] = np.digitize(frame_speed[frames], speed_edges, right=False)
```

iii. The AI consistently aligns frame-wise streams on the original imaging indices.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Plane arrays are cut to their common minimum length; behavior streams are sliced to that length. Retinotopy/neuron count mismatches and sessions with fewer than two trials raise errors. Missing `sess#` gets a before/after fallback, behavior keys fall back from suffixed to base IDs, and empty trials are dropped.

ii.
```python
nframes = min(a.shape[1] for a in planes)
if len(area_ids) != expected: raise ValueError(...)
if base not in behavior: raise KeyError(...)
```

iii. The AI observed that processed behavior commonly has one terminal sample beyond neural arrays and deliberately trims it; other safeguards were added to fail loudly instead of silently corrupting alignment.

## 12-a. What are the most time-consuming steps of the code?

i. Loading hundreds of gigabytes of spike arrays, copying every retained neuron/frame into trial arrays, serializing the 142-GiB pickle, and the extra all-session behavior pass for global speed edges dominate.

ii.
```python
spk_obj = np.load(spk_path, allow_pickle=True).item()
neural[row:row + n] = plane[np.ix_(keep, frames)]
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory repeatedly calls neural I/O and full conversion expensive and reports the final file size; its optimization avoids concatenating all planes first.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial full-mask scan, per-plane trial copy, area-ID generator, category discovery, and per-session speed collection could be grouped/vectorized. The per-trial output computations could also operate once on all retained frames before slicing.

ii.
```python
for trial in range(ntrials):
    frames = np.flatnonzero(valid & (trial_stamp == trial))
for plane, keep in zip(planes, plane_keep):
```

iii. The AI did not explicitly discuss these vectorization opportunities; it prioritized bounded memory and direct plane-wise copying.

## 12-c. What processing does the code repeat multiple times?

i. Behavior files are repeatedly loaded per session during category discovery, speed-edge computation, and conversion. `retained_frame_mask` is calculated in the speed and conversion passes, and every trial scans full-length masks.

ii.
```python
for desc in sessions: beh = load_behavior(desc)  # categories
speed_edges, timing_stats = compute_speed_edges(sessions)
beh = load_behavior(desc)                        # conversion
```

iii. The trajectory accepts these passes to avoid holding behavior data and neural data together excessively; it does not explicitly justify repeated disk loads.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes frame-interval diagnostic distributions, detailed `session_info`, and observed timing statistics that the decoder does not use. It also performs separate category and global-speed passes and invokes `gc.collect()` every session; these support metadata or memory management rather than decoding.

ii.
```python
frame_dts_ms.append(dt[(dt > 100) & (dt < 1000)])
gc.collect()
"observed_frame_interval_iqr_ms": ...
```

iii. The AI wanted auditable metadata and explicit memory reclamation for the huge dataset; the trajectory does not claim these are needed by downstream training.
