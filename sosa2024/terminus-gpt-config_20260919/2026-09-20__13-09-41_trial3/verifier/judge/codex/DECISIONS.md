# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively globs every NWB under `/app/data`, naturally sorts the paths, and processes each file as one session with `h5py`. Full mode uses all 152 files; sample mode intentionally uses only two.

ii.
```python
files=sorted(glob.glob(DATA_ROOT+'/**/*.nwb',recursive=True),key=natural_key)
if args.sample: files=files[:2]
for i,f in enumerate(files):
    ns,xs,ys,info=process_session(f, ...)
```

iii. The notes justify this from an inventory of the released tree: 152 NWBs, 11 mice, and 12,217 source trial IDs. They state that all 12,216 complete start/teleport trials were accounted for before quality filtering.

## 1-b. How are the data split into subjects?

i. Subject identity is read from each NWB's `general/subject/subject_id`. A first-seen unique subject list is built, and every session receives its index into that list.

ii.
```python
subject = h['general/subject/subject_id'][()].decode()
if info['subject'] not in subjects: subjects.append(info['subject'])
subject_idx.append(subjects.index(info['subject']))
```

iii. The agent says this produces the expected 11 mice and avoids relying only on path parsing.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session and contributes one element to each top-level session list.

ii.
```python
ns,xs,ys,info=process_session(f, ...)
neural.append(ns); inputs.append(xs); outputs.append(ys)
```

iii. This follows the released `sub-*/...ses-*.nwb` organization; the notes report 152 sessions.

## 1-d. How are the data split into trials?

i. Positive `trial_start` samples are starts. Each is paired with the sole positive `teleport` sample after it and before the next start; the interval is start-inclusive and teleport-exclusive. Trial-number transitions are deliberately not used as boundaries.

ii.
```python
starts = np.flatnonzero(start_flags > 0)
teleports = np.flatnonzero(teleport_flags > 0)
for j, start in enumerate(starts):
    next_start = starts[j+1] if j+1 < len(starts) else len(start_flags)
    candidates = teleports[(teleports > start) & (teleports < next_start)]
    if len(candidates) != 1: raise ValueError(...)
    trials.append((int(start), int(candidates[0])))
```

iii. Exploration found trial IDs cross teleport boundaries, whereas all 12,216 starts pair uniquely this way. Teleports are invalid track activity in the reference processing.

## 1-e. How are trials filtered based on quality controls?

i. A trial is discarded if more than 30% of its raw lick samples exceed 2. Sessions with fewer than two retained trials fail. No short-trial filter is used.

ii.
```python
if np.mean(lick_raw > 2) > 0.30:
    bad_lick += 1
    continue
...
if len(neural_trials) < 2: raise ValueError(...)
```

iii. The agent identifies this as the paper's lick-sensor corruption rule. Because the requested categorical lick target cannot encode NaN, it excludes the whole trial; exactly 81 trials were removed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. It comes directly from each plane in `processing/ophys/Deconvolved`, restricted using the segmentation table's `iscell` flag.

ii.
```python
dec = h['processing/ophys/Deconvolved']
arrays = [np.asarray(dec[p]['data'][:], dtype=np.float32) for p in plane_names]
iscell = np.asarray(seg['iscell'][:, 0]) > 0
return full[:, iscell], plane_idx[iscell], plane_names
```

iii. The agent argues the released NWBs are already reference-processed and frame-aligned, and that time-domain population/GLM analyses use deconvolved events.

## 2-b. How is the `neural` data processed?

i. Plane matrices are naturally sorted and concatenated along ROIs, `iscell` is applied, an allowed single trailing unmatched row is trimmed, and trial slices are transposed to neuron-by-time float32 arrays. It does not recompute dF/F, smooth, deconvolve, spatially bin, or speed-mask.

ii.
```python
full = np.concatenate(arrays, axis=1)
...
if neural_excess_frames: neural_all = neural_all[:ntime]
neu = np.ascontiguousarray(neural_all[q, :].T, dtype=np.float32)
```

iii. The notes say recomputation would be redundant and that native time-domain events preserve the requested stationary and framewise targets.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only Suite2p ROIs with `iscell[:,0] > 0` are retained. Putative interneurons, place-cell subsets, and reward-relative subsets are not removed.

ii.
```python
iscell = np.asarray(seg['iscell'][:, 0]) > 0
return full[:, iscell], plane_idx[iscell], plane_names
```

iii. The agent considers `iscell` the general imaging-quality criterion and regards interneuron/place-cell/reward-relative selection as analysis-specific.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural rows share the behavior frame grid. Each trial begins at its `trial_start` sample, which becomes column zero.

ii.
```python
q = slice(start, stop)
neu = np.ascontiguousarray(neural_all[q, :].T, dtype=np.float32)
```

iii. The notes report independent raw-data spot checks and matching time dimensions for neural, inputs, and outputs.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native median timestamp spacing is retained: about 0.0644836 s (64.4836 ms, 15.5078 Hz). No temporal rebinning is applied.

ii.
```python
dt = float(np.median(np.diff(timestamps)))
if not np.isclose(dt, DT_REFERENCE, ...): raise ValueError(...)
metadata=dict(time_bin_size=float(np.median(dts)*1000), ...)
```

iii. The agent explains that 62 Hz attributes in two-plane files describe aggregate scanning; each plane and behavior already have one row per ~15.5 Hz sample.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/position/timestamps` and trial-start indices.

ii.
```python
timestamps = np.asarray(b['position/timestamps'][:], dtype=np.float64)
(timestamps[q]-timestamps[start]).astype(np.float32)
```

iii. Position timestamps were selected as the common frame-aligned behavior grid.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at the trial start is subtracted from every timestamp in the trial and the result is cast to float32.

ii.
```python
(timestamps[q]-timestamps[start]).astype(np.float32)
```

iii. This makes the first retained sample zero and expresses elapsed time in seconds.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The identical slice `q` is applied to timestamps and neural rows, and shapes are asserted equal.

ii.
```python
inp = np.vstack([...])
neu = np.ascontiguousarray(neural_all[q, :].T, dtype=np.float32)
if neu.shape[1] != n or inp.shape != (4, n): raise AssertionError(...)
```

iii. The NWB streams were already synchronized to imaging frames.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the framewise behavior `environment/data` stream.

ii.
```python
streams = {n: np.asarray(b[n+'/data'][:]) for n in names}
env_values = np.unique(streams['environment'][q])
```

iii. The notes found values 0/1 and treated this stream as authoritative even where identifier naming was unusual.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Negative sentinels are removed, the remaining value must be one constant binary value, and it is repeated across the trial.

ii.
```python
env_values = env_values[env_values >= 0]
if len(env_values) != 1 or env_values[0] not in (0, 1): raise ValueError(...)
np.full(n, env, dtype=np.float32)
```

iii. This enforces the requested per-trial ENV1/ENV2 variable.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It uses the behavior `trial number/data` value at the start sample.

ii.
```python
trial_id = int(round(float(streams['trial number'][start])))
```

iii. The agent reports source IDs are contiguous and preserves the original zero-based source value.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The start value is rounded, converted to integer, then repeated as float32 across the trial.

ii.
```python
np.full(n, trial_id, dtype=np.float32)
```

iii. Repetition gives all inputs the required `(4, time)` representation.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from sparse `Reward/timestamps`, behavior timestamps, and the preceding chronological complete trial's boundaries.

ii.
```python
reward_t = np.asarray(b['Reward/timestamps'][:], dtype=float)
outcomes = np.asarray([np.any((reward_t >= timestamps[a]) & (reward_t < timestamps[z]))
                       for a, z in trials], dtype=np.int64)
```

iii. Sparse delivered rewards, rather than `autoreward`, directly encode rewarded versus omitted outcomes.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Each complete source trial is labeled by whether any reward timestamp lies in `[start, teleport)`. The first trial gets 0; later trials use `outcomes[j-1]`, even if that preceding trial is later excluded for bad licking.

ii.
```python
prev_outcome = int(outcomes[j-1]) if j > 0 else 0
np.full(n, prev_outcome, dtype=np.float32)
```

iii. This preserves the immediately preceding experimental trial rather than the preceding retained trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses framewise `position/data` and fixed A/B/C zone coordinates. The zone label is parsed from the NWB identifier's scene schedule and chronological trial index.

ii.
```python
scene, labels = parse_scene(identifier)
zone = zone_for_trial(labels, j)
zstart, zstop = ZONE_COORDS[zone]
pos = streams['position'][q].astype(np.float32)
```

iii. The agent says this matches reference `get_reward_zones(change_trial=30)` and fixed coordinates A=80–130, B=200–250, C=320–370 cm.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Distance is position minus the near edge before a zone, zero inside it, and position minus the far edge after it.

ii.
```python
distance = np.where(position < start, position-start,
                    np.where(position > stop, position-stop, 0.0))
```

iii. This implements signed distance to any location in the 50-cm reward interval.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit Boolean masks produce the seven requested classes with exact boundary handling.

ii.
```python
out[distance < -50] = 0
out[(distance >= -50) & (distance < -10)] = 1
out[(distance >= -10) & (distance < 0)] = 2
out[distance == 0] = 3
out[(distance > 0) & (distance <= 10)] = 4
out[(distance > 10) & (distance <= 50)] = 5
out[distance > 50] = 6
```

iii. The notes state these masks follow the instruction wording exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data use the same start-to-teleport slice, producing the same number of columns.

ii.
```python
pos = streams['position'][q].astype(np.float32)
neu = np.ascontiguousarray(neural_all[q, :].T, dtype=np.float32)
```

iii. Shape assertions and independent spot checks validate the alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It comes directly from framewise behavior `position/data`.

ii.
```python
pos = streams['position'][q].astype(np.float32)
```

iii. The stream is the virtual corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The trial slice is discretized directly; no smoothing, clipping, or spatial rebinning precedes categorization.

ii.
```python
posclass = np.digitize(pos, [90, 180, 270, 360], right=False).astype(np.int64)
```

iii. The agent preserves the native framewise series and lets end bins absorb small out-of-track values.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.digitize` with edges 90, 180, 270, and 360 creates five 90-cm classes; exact edges enter the higher class.

ii.
```python
np.digitize(pos, [90, 180, 270, 360], right=False)
```

iii. Five equal bins span the instructed 450-cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position and neural rows are sliced with the identical `q` interval.

ii.
```python
q = slice(start, stop)
pos = streams['position'][q]
neu = neural_all[q, :].T
```

iii. The common NWB frame grid and trial-shape checks ensure alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from framewise behavior `lick/data`.

ii.
```python
lick_raw = streams['lick'][q]
```

iii. Exploration showed this is a cumulative/count-like signal ranging above one, so it must be binarized.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Corrupt trials are excluded first; on retained trials, any positive value becomes 1 and all others become 0.

ii.
```python
if np.mean(lick_raw > 2) > 0.30: continue
lickclass = (lick_raw > 0).astype(np.int64)
```

iii. This uses the paper's corruption criterion and requested binary semantics.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural data are indexed by the same frame slice.

ii.
```python
lick_raw = streams['lick'][q]
neu = neural_all[q, :].T
```

iii. Both were already aligned to imaging frames in the NWB.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the final scene component of the NWB `identifier` and chronological complete-trial index, not from the framewise `reward_zone` stream.

ii.
```python
identifier = h['identifier'][()].decode()
scene, labels = parse_scene(identifier)
zone = zone_for_trial(labels, j)
```

iii. The agent found `reward_zone/data` contains transient codes 0–8 rather than per-trial A/B/C identity, while the identifier encodes the schedule.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. A/B/C labels are regex-parsed. A one-label schedule stays fixed; a two-label schedule switches from the first to second at chronological trial index 30. A/B/C map to 0/1/2 and are repeated over time.

ii.
```python
labels = re.findall(r'(?:Location)?([ABC])', scene)
return labels[0] if len(labels) == 1 or chronological_index < 30 else labels[1]
np.full(n, ZONE_INDEX[zone], dtype=np.int64)
```

iii. This is justified as matching reference `get_reward_zones(..., change_trial=30)` and is unaffected by excluded lick trials.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from sparse `Reward/timestamps`, behavior timestamps, and each complete trial's start/teleport bounds.

ii.
```python
reward_t = np.asarray(b['Reward/timestamps'][:], dtype=float)
np.any((reward_t >= timestamps[a]) & (reward_t < timestamps[z]))
```

iii. These timestamps represent delivered reward events directly.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Any reward in `[start, teleport)` yields 1, otherwise 0; the value is repeated for every frame in the trial.

ii.
```python
outcomes = np.asarray([...], dtype=np.int64)
np.full(n, outcomes[j], dtype=np.int64)
```

iii. This realizes the requested rewarded/omitted per-trial category.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code mainly fails loudly on unexpected structure or values. It allows and trims only a known one-row neural excess in ten files, rejects nonfinite neural/input values, rejects behavior length/rate anomalies and malformed trials, excludes corrupt-lick trials, and ignores one incomplete source trial lacking a start pulse.

ii.
```python
if neural_excess_frames not in (0, 1): raise ValueError(...)
if neural_excess_frames: neural_all = neural_all[:ntime]
if not (np.isfinite(neu).all() and np.isfinite(inp).all()): raise ValueError(...)
```

iii. The notes fully reconcile 12,217 IDs as 12,216 complete trials plus one incomplete edge ID, and document the extra neural rows as trailing samples after behavior.

## 13-a. What are the most time-consuming steps of the code?

i. Reading and concatenating large deconvolved plane matrices for 152 NWBs, materializing all per-trial arrays, and serializing the roughly dataset-sized pickle dominate. Optional plotting and downstream decoder training are separate from conversion.

ii.
```python
arrays = [np.asarray(dec[p]['data'][:], dtype=np.float32) for p in plane_names]
full = np.concatenate(arrays, axis=1)
with open(args.outpicklefile,'wb') as f: pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
```

iii. The full run log and notes emphasize the 87-GB source and report per-session and total elapsed times.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Reward outcome calculation loops over trials; `process_session` then loops again over every trial to validate, allocate, categorize, and append arrays. Some class calculations could be performed session-wide before slicing, though variable trial validity and lengths make complete vectorization awkward.

ii.
```python
return np.asarray([np.any((reward_t >= timestamps[a]) & (reward_t < timestamps[z]))
                   for a, z in trials], dtype=np.int64)
for j, (start, stop) in enumerate(trials):
    ...
```

iii. The agent does not explicitly discuss these loops, but its implementation favors clear per-trial validation and variable-length output construction.

## 13-c. What processing does the code repeat multiple times?

i. It repeatedly allocates constant per-trial rows for environment, trial ID, previous outcome, zone, and outcome. Reward timestamps are scanned once per trial, and classification functions run separately per trial. Unlike the human solution, the shipped converter has no separate survey pass that reloads every NWB.

ii.
```python
np.full(n, env, dtype=np.float32)
np.full(n, trial_id, dtype=np.float32)
np.full(n, prev_outcome, dtype=np.float32)
np.full(n, ZONE_INDEX[zone], dtype=np.int64)
np.full(n, outcomes[j], dtype=np.int64)
```

iii. Repetition is largely required by the uniform time-series target format; the notes describe extensive exploration separately, but it is not repeated by `convert_data.py` during a conversion.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes continuous distance minima/maxima and detailed per-trial/session provenance used only as metadata; it also retains curated plane indices in metadata while all neurons receive the same dorsal-CA1 region label. Optional plots are diagnostic and not decoder inputs.

ii.
```python
kept_info.append(dict(..., distance_min=float(distance.min()),
                      distance_max=float(distance.max())))
...
curated_plane_idx=curated_planes
```

iii. The agent justifies these as auditability and sanity-check information. They are small relative to the neural arrays but are not consumed by decoder training.
