# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively enumerates every subject directory under `/app/data`, collects every `.nwb` file, sorts files by mouse number and session number, and loads each file directly with `h5py`. Sessions are converted in a multiprocessing pool and cached temporarily; failed or unusable sessions are skipped.

ii.
```python
for sub in sorted(os.listdir(DATA_ROOT)):
    d = os.path.join(DATA_ROOT, sub)
    if os.path.isdir(d):
        files += [os.path.join(d, fn) for fn in sorted(os.listdir(d)) if fn.endswith('.nwb')]
files.sort(key=lambda p: (int(re.search(r'sub-m(\d+)', p).group(1)),
                          int(re.search(r'ses-(\d+)', p).group(1))))
...
with h5py.File(path, 'r') as f:
```

iii. The trajectory says this covers the full DANDI switch cohort (152 sessions, 11 mice). The agent used direct HDF5 access for speed and parallelized/cached session conversion because the full dataset is large.

## 1-b. How are the data split into subjects?

i. Subjects are parsed from file paths as `m<number>`, numerically sorted, and each converted session gets a `subject_idx` pointing into that list. The subject ID is also read from the NWB metadata.

ii.
```python
subject = f['general/subject/subject_id'][()].decode()
subjects = sorted({re.search(r'sub-(m\d+)', p).group(1) for p in files},
                  key=lambda s: int(s[1:]))
subject_idx.append(subjects.index(res['subject']))
```

iii. The agent treated the directory/file naming and NWB subject metadata as consistent identifiers and reported 11 mice, matching the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session. One session-level entry is appended to `neural`, `input`, `output`, and `brain_region_idx` for each successfully converted file.

ii.
```python
for path in files:
    ...
    data['neural'].append(res['neural'])
    data['input'].append(res['input'])
    data['output'].append(res['output'])
```

iii. The filenames contain `ses-<number>`, and the agent described the resulting 152 files as 152 sessions.

## 1-d. How are the data split into trials?

i. A trial is a lap bounded by a positive `trial_start` sample and a positive `teleport` sample. For exported arrays it uses `[trial_start-1, teleport-1)`, intended to reproduce the one-based indexing inside the paper’s `dff()` implementation.

ii.
```python
trial_starts = np.flatnonzero(beh['trial_start/data'][:] > 0)
teleports = np.flatnonzero(beh['teleport/data'][:] > 0)
...
a, b = int(trial_starts[i]) - 1, int(teleports[i]) - 1
neu = np.ascontiguousarray(events[:, a:b], dtype=np.float32)
```

iii. The trajectory says laps are the paper’s natural trial unit and calls this slice “exactly” the reference `dff()` slice, excluding the interpolated teleport frame.

## 1-e. How are trials filtered based on quality controls?

i. The first imaged lap is always dropped because its preceding outcome is unknown. A lap is also dropped if more than 30% of its frames have raw lick values above 2, if it has fewer than two sliced frames, or if its neural slice is nonfinite. Sessions with fewer than two surviving trials are dropped.

ii.
```python
lick_error[i] = np.mean(lick[a:b] > 2) > LICK_ERROR_THRESH
...
if i == 0: continue
if lick_error[i]: continue
if b - a < 2: continue
if not np.all(np.isfinite(neu)): continue
...
if len(neural_trials) < 2: return None
```

iii. The agent found that the 30% lick-artifact rule flags exactly 81 laps, matching the Methods, and dropped these because lick is a required output. It preferred dropping each first lap rather than inventing an unknown previous-trial label.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is recomputed from each plane’s raw suite2p `Fluorescence` (F) and `Neuropil` (Fneu), restricted by `iscell`; the stored NWB `Deconvolved` series is not used.

ii.
```python
F = np.concatenate([f['processing/ophys/Fluorescence'][p]['data'][:nframes, :]
                    for p in plane_names], axis=1).T
Fneu = np.concatenate([f['processing/ophys/Neuropil'][p]['data'][:nframes, :]
                       for p in plane_names], axis=1).T
iscell = f['processing/ophys/ImageSegmentation/PlaneSegmentation/iscell'][:, 0] > 0
```

iii. The agent reasoned that the NWB `Deconvolved` field is suite2p’s raw-F result, whereas the paper analyzes events produced by its own `preprocessing.dff` pipeline.

## 2-b. How is the `neural` data processed?

i. Planes are pooled. After neuropil subtraction (`F - 0.7*Fneu`), trial-mean neuropil is added back. A maximin baseline is obtained with Gaussian sigma 15, a 300-sample minimum filter and then maximum filter. The code computes `(F-baseline)/abs(baseline)`, smooths dF/F with Gaussian sigma 2, and OASIS-deconvolves it with tau 0.7 at 15.5078125 Hz. Selected sessions include teleport intervals in baseline segments according to a hard-coded mouse/day table.

ii.
```python
f_ -= NEU_COEF * fneu_
f_[:, a:b] += NEU_COEF * np.nanmean(fneu_[:, a:b], axis=1, keepdims=True)
seg = ndi.gaussian_filter1d(f_[:, a:b], BASELINE_SMOOTH_SIG, axis=1)
seg = ndi.minimum_filter1d(seg, BASELINE_WIN, axis=-1)
seg = ndi.maximum_filter1d(seg, BASELINE_WIN, axis=-1)
dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
events[:, a:b] = dcnv.oasis(..., 2000, TAU, FRAME_RATE)
```

iii. The trajectory explicitly ties every parameter to the Methods/reference repository and notes that `keep_teleports` follows `teleport_metadata.py` so the baseline reflects whether the laser remained on between laps.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only suite2p/manual `iscell` ROIs are retained, then cells with Pearson correlation greater than 0.5 between dF/F and speed are removed as putative interneurons.

ii.
```python
F = np.ascontiguousarray(F[iscell], dtype=np.float64)
...
speed_corr = (dv @ spc) / denom
is_int = np.nan_to_num(speed_corr, nan=0.0) > INT_R_THRESH
events = events[~is_int]
```

iii. The agent cites the paper’s manual curation and `dayData` threshold, and checked that the removed fraction closely matches the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials start at the lap-start slice boundary and therefore are aligned to trial start without resampling. All behavior uses the identical `[a:b]` slice.

ii.
```python
a, b = int(trial_starts[i]) - 1, int(teleports[i]) - 1
neu = np.ascontiguousarray(events[:, a:b], dtype=np.float32)
t_rel = (ts[a:b] - ts[a]).astype(np.float32)
```

iii. The agent states that alignment follows directly from splitting the already frame-aligned streams at trial start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native per-plane imaging resolution is retained: `1000/15.5078125 = 64.484... ms`. No temporal rebinning or resampling is applied.

ii.
```python
FRAME_RATE = 15.5078125
...
'time_bin_size': 1000.0 / FRAME_RATE,
```

iii. The agent reports a constant 64.484 ms bin and says preserving native bins maintains exact behavioral/neural alignment.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the `position` behavioral time-series timestamps.

ii.
```python
ts = beh['position/timestamps'][:]
t_rel = (ts[a:b] - ts[a]).astype(np.float32)
```

iii. The agent treated the behavioral timestamps as the common imaging-frame clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp in the retained trial slice is subtracted from every timestamp; the result is cast to float32.

ii.
```python
t_rel = (ts[a:b] - ts[a]).astype(np.float32)
inp[0] = t_rel
```

iii. This directly represents seconds elapsed since the aligned trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses precisely the same `[a:b]` frame slice and thus has the same length as neural data.

ii.
```python
neu = events[:, a:b]
T = b - a
t_rel = ts[a:b] - ts[a]
inp = np.empty((4, T), dtype=np.float32)
```

iii. The agent relied on the NWB behavioral and imaging arrays already sharing frame indices.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavioral `environment/data` time series.

ii.
```python
env_ts = beh['environment/data'][:]
env_vals = np.unique(env_ts[a:b])
environment[i] = int(env_vals[0])
```

iii. The agent verified each trial has exactly one environment value.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Unique values are checked per lap, the sole value is converted to an integer, and that scalar is repeated across every timepoint of the trial.

ii.
```python
assert len(env_vals) == 1
environment[i] = int(env_vals[0])
...
inp[1] = environment[i]
```

iii. The variable is per-trial and the observed raw coding already represents the two environments.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the zero-based loop index over paired `trial_start`/`teleport` boundaries, not from a stored trial-number time series.

ii.
```python
for i in range(ntrials):
    ...
    inp[2] = i
```

iii. The agent preserves the true within-session ordinal even after filtering, so exported trials begin at ordinal 1 because ordinal 0 is removed.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transformation beyond assigning the integer loop index to every timepoint.

ii.
```python
inp[2] = i
```

iii. The trajectory identifies this as a per-lap contextual variable.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from behavioral `Reward/timestamps`, mapped onto the `position/timestamps` frame clock, then summarized within the preceding trial boundaries.

ii.
```python
reward_ts = beh['Reward/timestamps'][:]
rew_frames = np.searchsorted(ts, reward_ts)
rewarded[i] = int(np.any((rew_frames >= a) & (rew_frames < b)))
```

iii. The agent notes that the Reward data values are all zero, so event timestamps are the usable indication of delivery.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Each recorded trial is marked rewarded if any mapped reward frame falls between its start and teleport. For retained trial `i`, `rewarded[i-1]` is repeated over all timepoints. Trial 0 is dropped because no recorded predecessor exists.

ii.
```python
if i == 0:
    continue
...
inp[3] = rewarded[i - 1]
```

iii. The agent preferred removing the unknown case rather than labeling the unobserved warm-up lap as omitted.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses behavioral `position/data` plus reward-zone identity inferred from the NWB identifier/scene name and a fixed switch after trial 30. Fixed zone extents are A=80–130, B=200–250, C=320–370 cm.

ii.
```python
scene = f['identifier'][()].decode().rstrip('/').split('/')[-1]
zone_labels = scene_reward_zones(scene, ntrials)
zstart, zstop = REWARD_ZONES[zone_labels[i]]
p = pos[a:b]
```

iii. The agent traced this rule to `behavior.get_reward_zones` and says it validated all 12,216 rewarded laps against observed zone entry positions with zero mismatches.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is position minus the near edge before the zone, zero inside it, and position minus the far edge after it; this continuous intermediate is immediately categorized.

ii.
```python
d = np.zeros_like(pos)
d[pos < zstart] = pos[pos < zstart] - zstart
d[pos > zstop] = pos[pos > zstop] - zstop
```

iii. This implements distance to the nearest point in the active zone and the requested sign convention.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit masks create seven categories: `<-50`, `[-50,-10)`, `[-10,0)`, exactly/inside-zone zero, `(0,10]`, `(10,50]`, and `>50`.

ii.
```python
out = np.full(pos.shape, 3, dtype=np.int64)
out[d < -50.0] = 0
out[(d >= -50.0) & (d < -10.0)] = 1
out[(d >= -10.0) & (d < 0.0)] = 2
out[(d > 0.0) & (d <= 10.0)] = 4
out[(d > 10.0) & (d <= 50.0)] = 5
out[d > 50.0] = 6
```

iii. The masks were chosen to implement the instruction’s stated thresholds directly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position is sliced with the identical `[a:b]` indices as events and categorized frame by frame.

ii.
```python
neu = events[:, a:b]
p = pos[a:b]
out[0] = reward_zone_distance_bin(p, zstart, zstop)
```

iii. No interpolation is needed because both streams share imaging-frame indexing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It comes directly from behavioral `position/data`.

ii.
```python
pos = beh['position/data'][:]
...
p = pos[a:b]
```

iii. The raw variable is already position in centimeters along the corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The trial slice is directly discretized; no smoothing, clipping, or resampling is done.

ii.
```python
out[1] = np.digitize(p, POSITION_EDGES)
```

iii. The instruction requests categorical bins spanning the 450 cm track, so direct binning suffices.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `np.digitize` with `[90,180,270,360]` produces five bins numbered 0–4.

ii.
```python
POSITION_EDGES = [90.0, 180.0, 270.0, 360.0]
out[1] = np.digitize(p, POSITION_EDGES)
```

iii. These are five equal 90 cm sections of a 450 cm corridor.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position and neural arrays use the same trial-frame slice.

ii.
```python
neu = events[:, a:b]
p = pos[a:b]
```

iii. The agent considered the streams natively frame-aligned.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from behavioral `lick/data`.

ii.
```python
lick = beh['lick/data'][:]
```

iii. This is the dataset’s frame-level lick sensor stream.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Positive values become 1 and all others become 0. Trials with the paper’s stuck-sensor artifact are removed before this conversion.

ii.
```python
lick_error[i] = np.mean(lick[a:b] > 2) > LICK_ERROR_THRESH
...
out[3] = (lick[a:b] > 0).astype(np.int64)
```

iii. The required decoder output is binary; artifact laps are removed because their lick labels are unreliable.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is indexed with the same `[a:b]` trial slice as neural activity.

ii.
```python
neu = events[:, a:b]
out[3] = (lick[a:b] > 0).astype(np.int64)
```

iii. The streams share imaging-frame indices.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the NWB identifier’s scene name, the lap index, and fixed zone definitions/switch point—not from the raw `reward_zone` behavior series.

ii.
```python
scene = f['identifier'][()].decode().rstrip('/').split('/')[-1]
zone_labels = scene_reward_zones(scene, ntrials)
```

iii. The agent says this matches the repository’s `behavior.get_reward_zones` and was empirically validated against rewarded laps.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Regexes parse constant-zone or switching scene names. Switching scenes use the first label for trials 0–29 and the second thereafter; A/B/C is converted to 0/1/2 and repeated over the trial.

ii.
```python
return [m.group(1)] * SWITCH_TRIAL + [m.group(2)] * (ntrials - SWITCH_TRIAL)
...
out[4] = ZONE_LABELS.index(zone_labels[i])
```

iii. The switch-after-30 rule comes from the paper code, and the agent found no mismatch on observable rewarded laps.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/timestamps` and the common behavioral `position/timestamps` clock.

ii.
```python
reward_ts = beh['Reward/timestamps'][:]
rew_frames = np.searchsorted(ts, reward_ts)
```

iii. The agent used timestamps because the Reward data values are unusable (all zero).

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward times are mapped to insertion-frame indices with `searchsorted`; a lap is 1 if any index lies in its `[trial_start, teleport)` interval and 0 otherwise. The scalar is repeated over the exported trial.

ii.
```python
rewarded[i] = int(np.any((rew_frames >= a) & (rew_frames < b)))
...
out[5] = rewarded[i]
```

iii. This produces the requested per-trial binary outcome and yielded an 84.7% reward rate, consistent with approximately 15% omission.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code asserts equal fluorescence-frame and timestamp lengths and equal counts of starts/teleports; asserts a single environment and positive scanning per retained slice; drops nonfinite neural trials, sub-two-frame trials, zero-neuron sessions, and sessions with fewer than two trials. Worker exceptions are caught and cause a logged session skip. It does not crop mismatched streams or verify reward timestamp error.

ii.
```python
assert F.shape[1] == nframes and len(trial_starts) == len(teleports)
...
if not np.all(np.isfinite(neu)): continue
...
except Exception:
    return path, None, traceback.format_exc()
```

iii. The agent emphasized successful whole-dataset verification (zero validator errors/warnings), while using assertions and skips to prevent malformed data entering the output.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive work is reading large fluorescence/neuropil arrays, Gaussian/minimum/maximum filtering and OASIS deconvolution for every session, then serializing the roughly 9.5 GB result. The implementation uses multiprocessing and per-session caches.

ii.
```python
with ctx.Pool(args.workers) as pool:
    for ... in pool.imap_unordered(_worker, files):
...
pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory shows timed session tests, an eight-worker full run, and caching, reflecting the agent’s concern with expensive full-dataset preprocessing.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial-level reward/environment/artifact calculation and trial assembly remain Python loops; reward detection repeatedly scans all reward-frame indices. Segment loops are needed for variable trial windows, though some per-session labels and masks could be vectorized.

ii.
```python
for i, (a, b) in enumerate(zip(trial_starts, teleports)):
    rewarded[i] = int(np.any((rew_frames >= a) & (rew_frames < b)))
...
for i in range(ntrials):
```

iii. The trajectory does not explicitly justify these loops; variable-length trials and segment-specific filtering make them straightforward, while multiprocessing targets the larger session-level cost.

## 13-c. What processing does the code repeat multiple times?

i. It traverses trial/segment boundaries multiple times: to copy F/Fneu, compute/add neuropil means and baselines, smooth/deconvolve, calculate trial metadata, and finally assemble arrays. It also reloads each cached session to build the final object.

ii.
```python
for a, b in segments: ...
for a, b in segments: ...
for a, b in segments: ...
for i, (a, b) in enumerate(...): ...
for i in range(ntrials): ...
```

iii. The segment passes mirror distinct stages of the reference dF/F procedure; caching was used so completed sessions need not be recomputed after failures.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Full dF/F is retained long enough to compute interneuron correlations and then deleted; raw F/Fneu and centered correlation arrays are likewise discarded. Metadata/statistics and caches support validation but not decoder features. The code also processes first laps neurally before later dropping them.

ii.
```python
dff, events = compute_dff_and_events(...)
...
events = events[keep_cells]
del dff, dv
...
if i == 0:
    continue
```

iii. The discarded dF/F is necessary for the paper’s interneuron filter, and whole-session processing is required for matching its baseline pipeline; the trajectory does not identify further avoidable work.
