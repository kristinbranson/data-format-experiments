# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all 89 neural session files with `glob`, uses their basenames as physical session IDs, loads every `Beh_*.npy`, groups all matching behavior views by physical ID after stripping `_swap1/_swap2`, and loads each session's spike and retinotopy files during conversion.

ii.
```python
glob.glob(os.path.join(ROOT, "spk", "*_neural_data.npy"))
for path in sorted(glob.glob(os.path.join(ROOT, "beh", "Beh_*.npy"))):
    obj = np.load(path, allow_pickle=True).item()
obj = np.load(path, allow_pickle=True).item()
with np.load(rp) as r:
    region_idx = map_regions(r["iarea"])
```

iii. The notes say this captures 89 unique physical recordings while merging duplicate analysis views. Loading one neural session at a time avoids holding the 434 GB source collection in memory.

## 1-b. How are the data split into subjects?

i. The mouse prefix of each physical session ID identifies the subject. Unique prefixes are sorted, and each session receives the corresponding `subject_idx`.

ii.
```python
subjects = sorted({sid.split("_")[0] for sid in selected})
subject_map = {v: i for i, v in enumerate(subjects)}
"subject_idx": np.asarray([subject_map[s.split("_")[0]] for s in selected])
```

iii. The notes report 19 mice and 89 sessions, matching independently aggregated source metadata.

## 1-c. How are the data split into sessions?

i. A session is the unique `<mouse>_<YYYY>_<MM>_<DD>_<block>` ID named by each neural file. Behavior keys with swap suffixes are treated as duplicate analysis views of the same physical session and merged.

ii.
```python
def physical_id(key):
    return re.sub(r"_swap[12]$", "", key)
views[sid].append((group, key, beh))
```

iii. The AI justified this as preventing duplicate conversion of the same recording while retaining semantic labels exposed in different behavior views.

## 1-d. How are the data split into trials?

i. Trial windows are variable length and run from `ceil(StartFr)` inclusive to `ceil(GrayFr)` exclusive, clipped to the neural frame count. Neural, input, and output arrays are sliced using the same bounds.

ii.
```python
starts = np.ceil(np.asarray(beh["StartFr"], dtype=np.float64)).astype(np.int64)
ends = np.ceil(np.asarray(beh["GrayFr"], dtype=np.float64)).astype(np.int64)
for i, (a, z) in enumerate(zip(starts, ends)):
    nt = np.concatenate([x[:, a:z] for x in parts], axis=0)
```

iii. The notes say spot checks showed floor would include a previous-trial frame and that these bounds reproduce 1,375,142 retained corridor frames.

## 1-e. How are trials filtered based on quality controls?

i. No long or stalled trials are filtered. All positive corridor windows are retained; invalid nonpositive windows cause an exception rather than being dropped.

ii.
```python
if np.any(ends <= starts):
    raise ValueError(f"Invalid trial windows: {bad[:20].tolist()}")
for i, (a, z) in enumerate(zip(starts, ends)):
```

iii. The AI inspected rare long trials and concluded they were genuine stalled traversals with consistent trial IDs, and found no explicit paper rule supporting exclusion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from all three `spks` arrays in each `*_neural_data.npy`. `iarea` from the session retinotopy NPZ supplies region labels.

ii.
```python
parts = obj["spks"]
nt = np.concatenate([x[:, a:z] for x in parts], axis=0)
region_idx = map_regions(r["iarea"])
```

iii. The AI states this preserves the reference loader's exact component and neuron order.

## 2-b. How is the `neural` data processed?

i. Deconvolved traces are sliced by trial, concatenated across planes, and stored as float32. No normalization, dF/F calculation, smoothing, or temporal resampling is applied.

ii.
```python
nt = np.concatenate([x[:, a:z] for x in parts], axis=0).astype(np.float32, copy=False)
```

iii. The notes identify `spks` as already-deconvolved Suite2p traces and say per-trial concatenation is mathematically equivalent to full-session concatenation while avoiding a multi-GB temporary copy.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are removed. Retinotopy codes outside V1/mHV/lHV/aHV are assigned an explicit `unassigned` fifth region.

ii.
```python
out = np.full(a.shape, 4, dtype=np.int16)
REGIONS = ["V1", "mHV", "lHV", "aHV", "unassigned"]
```

iii. The AI reasoned that the released rows were already Suite2p cell-classified and that no additional SNR/activity or figure-specific area selection should be invented.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each variable-length trial begins at the first included frame at or after `StartFr` (corridor entry) and ends before `ceil(GrayFr)`. Nothing is padded.

ii.
```python
"temporal_alignment_event": "visual corridor entry (StartFr; first included frame is ceil(StartFr))"
"trial_window": "[ceil(StartFr), ceil(GrayFr)); visual 4-m corridor only; variable duration"
```

iii. The AI chose native trial durations because the decoder supports variable lengths and reported direct raw-data alignment checks.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native imaging frames are retained without rebinning. A per-session median interval is computed from `ft`; metadata stores the median across sessions, approximately 315 ms.

ii.
```python
dt = float(np.median(np.diff(beh["ft"]) * 86400.0))
"time_bin_size": float(np.median(dt_values))
```

iii. The AI says temporal resampling is unnecessary because neural and behavioral values already share the imaging-frame grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundFr`, the retained integer frame indices, and the session median frame interval derived from `ft`.

ii.
```python
sound = np.asarray(beh["SoundFr"], dtype=np.float64)
frames = np.arange(a, z, dtype=np.float64)
inp[0] = (sound[i] - frames) * dt
```

iii. The AI treated behavior events as fractional imaging-frame indices and converted their offsets to seconds.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The current frame index is subtracted from `SoundFr` and multiplied by median seconds per frame, producing positive values before and negative values after the cue.

ii.
```python
inp[0] = (sound[i] - frames) * dt
```

iii. The notes describe this as signed cue-relative timing crossing zero at the event.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated for exactly the integer frame indices `[a, z)` used to slice neural data, so it has the same number and order of bins.

ii.
```python
frames = np.arange(a, z, dtype=np.float64)
nt = np.concatenate([x[:, a:z] for x in parts], axis=0)
inp[0] = (sound[i] - frames) * dt
```

iii. The AI reports independent raw input checks and no temporal offset in diagnostic plots.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the calendar date embedded in every session ID, relative to that mouse's earliest recording among all 89 sessions.

ii.
```python
d = datetime.strptime(date, "%Y_%m_%d").date()
return {sid: float((dates[sid] - first[sid.split("_")[0]]).days) for sid in session_ids}
```

iii. The AI says metadata `sess#` was inconsistent, so dates were the robust source.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Calendar-day differences are computed per mouse and the scalar session value is broadcast across every frame of every trial.

ii.
```python
inp[1] = day_value
```

iii. The notes explicitly define this as calendar days since the subject's first included recording; observed values can skip days.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It comes from fractional `StartFr`, retained integer frame indices, and the median interval calculated from `ft`.

ii.
```python
inp[2] = (frames - float(beh["StartFr"][i])) * dt
```

iii. The AI identifies corridor entry as trial start.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The fractional start frame is subtracted from each included frame and multiplied by seconds per frame. Thus the first included value is near, but generally above, zero.

ii.
```python
inp[2] = (frames - float(beh["StartFr"][i])) * dt
```

iii. This preserves the fractional event location rather than forcing the first stored bin to exactly zero.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It uses the same `[a, z)` frame sequence as the neural slice.

ii.
```python
frames = np.arange(a, z, dtype=np.float64)
nt = np.concatenate([x[:, a:z] for x in parts], axis=0)
```

iii. The AI's diagnostics showed a monotonically increasing time trace aligned to corridor entry.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is read directly from per-trial `isRew`.

ii.
```python
reward = np.asarray(beh["isRew"], dtype=np.int16)
```

iii. The notes treat it as the source's rewarded-corridor indicator.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The 0/1 trial value is broadcast over all retained frames.

ii.
```python
inp[3] = reward[i]
```

iii. No transformation beyond dtype conversion and broadcasting is applied.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It primarily uses concrete `TrialStim` labels merged across duplicate behavior views, falling back to `WallName` when all views contain the placeholder.

ii.
```python
concrete = {str(b["TrialStim"][i]) for _, _, b in session_views
            if str(b["TrialStim"][i]) != "stimulus_of_trial"}
label = next(iter(concrete)) if concrete else next(iter(wall))
```

iii. The AI merged views to recover labels masked in some analysis views and rejects conflicts or unknown labels.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Eight specific variants (`circle1-3`, `leaf1-3`, `leaf1_swap1-2`) are assigned IDs 0–7 and broadcast across a trial; variants are not collapsed to four base textures.

ii.
```python
STIM_TO_ID = {v: i for i, v in enumerate(STIM_VALUES)}
out[0] = STIM_TO_ID[str(stimuli[i])]
```

iii. The AI regarded the eight merged semantic labels as the global class schema and reported all eight in the full dataset.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from session-level fractional `LickFr` frame numbers.

ii.
```python
lick_frames = np.asarray(beh["LickFr"], dtype=np.float64)
lick_idx = np.floor(lick_frames).astype(np.int64)
```

iii. The notes state that `LickFr` directly indexes the imaging-frame grid.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frames are floored, restricted to valid neural frames, uniqued, and rasterized as binary presence; multiple licks in one frame remain 1.

ii.
```python
lick_session = np.zeros(nfr, dtype=np.int16)
lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nfr)]
lick_session[np.unique(lick_idx)] = 1
```

iii. The AI chose deterministic binary frame presence, as required by the task.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The session raster is sliced with exactly the same `[a:z]` bounds as neural data.

ii.
```python
out[1] = lick_session[a:z]
```

iii. Direct raw-output checks reportedly matched.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from frame-wise `ft_Pos` over retained corridor frames.

ii.
```python
pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
```

iii. The AI interpreted the 0–40 source range as the four-metre visual corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Frame-wise positions are digitized into four integer categories and clipped to 0–3.

ii.
```python
out[2] = np.clip(np.digitize(pos[a:z], [10.0, 20.0, 30.0]), 0, 3)
```

iii. The notes say thresholds 10/20/30 implement four physical one-metre bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Source positions below 10, 10–20, 20–30, and at least 30 map to categories 0–3, labeled 0–1 m through 3–4 m.

ii.
```python
["0-1 m", "1-2 m", "2-3 m", "3-4 m"]
np.digitize(pos[a:z], [10.0, 20.0, 30.0])
```

iii. The choice follows the requested four equal-length spatial bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos[a:z]` uses the same retained imaging-frame indices as the neural trial.

ii.
```python
nt = np.concatenate([x[:, a:z] for x in parts], axis=0)
out[2] = np.clip(np.digitize(pos[a:z], [10.0, 20.0, 30.0]), 0, 3)
```

iii. The AI audited trial IDs and monotonic position traces in raw and converted trials.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from frame-wise `ft_RunSpeed` over all selected corridor frames.

ii.
```python
speed = np.asarray(b["ft_RunSpeed"][:nfr], dtype=np.float64)
chunks.extend(speed[a:z] for a, z in zip(starts, ends))
```

iii. The source stream is already on the imaging-frame grid.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Three global numerical quantiles are computed across all retained frames, then each session's values are digitized into 0–3.

ii.
```python
q = np.quantile(values, [0.25, 0.5, 0.75])
out[3] = np.clip(np.digitize(speed[a:z], speed_q), 0, 3)
```

iii. The AI used deterministic value thresholds. It explicitly declined arbitrary/random tie breaking when many speeds equal zero.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Full-data global thresholds are approximately 0, 8.3812, and 30.1894. Because `np.digitize` places values equal to zero above the first boundary, the bottom classes are imbalanced (about 9.8%/40.2%/25%/25%).

ii.
```python
speed_q, n_speed, speed_range = compute_speed_thresholds(...)
np.digitize(speed[a:z], speed_q)
```

iii. The notes argue deterministic value bins are preferable to splitting tied zero values arbitrarily.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is sliced over the same `[a:z]` frame interval as neural data.

ii.
```python
out[3] = np.clip(np.digitize(speed[a:z], speed_q), 0, 3)
```

iii. Raw-output spot checks reportedly passed exactly under the chosen thresholds.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Duplicate behavior views are checked for compatible dimensions and physical streams; missing views, conflicts, unknown labels, invalid windows, malformed IDs, nonfinite speeds, and region/neuron count mismatches raise errors. Trial bounds are clipped to neural length, and out-of-range licks are discarded.

ii.
```python
starts = np.clip(starts, 0, nfr); ends = np.clip(ends, 0, nfr)
lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nfr)]
if len(region_idx) != nneu:
    raise ValueError(...)
```

iii. The AI emphasized strict validation and reported no unresolved data or verifier warnings.

## 12-a. What are the most time-consuming steps of the code?

i. Reading hundreds of GB of object NPY neural files, creating the very large per-trial arrays, and serializing the 296 GB pickle dominate runtime.

ii.
```python
obj = np.load(path, allow_pickle=True).item()
nt = np.concatenate([x[:, a:z] for x in parts], axis=0)
pickle.dump(data, f, protocol=4)
```

iii. The notes report 1,609.8 seconds for full conversion and 434.1 seconds for serialization.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The Python loop over trials, per-trial concatenation across three planes, `merged_stimuli`'s trial/view loops, and speed-chunk construction could be reduced or grouped, although variable trial lengths still require final per-trial objects.

ii.
```python
for i, (a, z) in enumerate(zip(starts, ends)):
    nt = np.concatenate([x[:, a:z] for x in parts], axis=0)
for i in range(n):
    concrete = {... for _, _, b in session_views ...}
```

iii. The AI instead vectorized frame arithmetic and digitization, judging source I/O and unavoidable output copies to dominate.

## 12-c. What processing does the code repeat multiple times?

i. `choose_behavior` and `trial_bounds` are repeated during global speed-threshold collection and session conversion. Behavior files are loaded once in `load_behavior_views` and again in `metadata_by_session`; the sample mode also reads neural files in a shape scan and again for conversion.

ii.
```python
b = choose_behavior(views[sid]); starts, ends = trial_bounds(b, nfr)
beh = choose_behavior(session_views); starts, ends = trial_bounds(beh, nfr)
info = np.load(os.path.join(ROOT, "beh", "Imaging_Exp_info.npy"), ...)
```

iii. The AI explicitly avoids the neural pre-scan in full mode to prevent reading 434 GB twice; the remaining repetitions support validation or metadata collection.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Full conversion creates `session_stats`, source dtype placeholders, extensive metadata, and repeated validation scans that the decoder does not use. It also retains unassigned neurons, eight fine stimulus labels, and extremely long stalled-trial frames that the reference conversion excludes or coarsens.

ii.
```python
session_stats.append(st)
st["experiment_views"] = meta_views.get(sid, [])
validate_local(data)
REGIONS = ["V1", "mHV", "lHV", "aHV", "unassigned"]
```

iii. The AI viewed the metadata and checks as useful provenance/validation, and deliberately retained all released traces and stalled runs rather than treating them as unnecessary.
