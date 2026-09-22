# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent creates an AllenSDK `VisualBehaviorOphysProjectCache` from the local cache, gets the ophys experiment table, intersects it with experiment IDs inferred from locally present NWB filenames, filters to active `project_code == "VisualBehavior"` experiments, and loads each selected experiment through `get_behavior_ophys_experiment`. It parallelizes experiment loading with a process pool.

ii.
```python
def get_cache():
    global _CACHE
    if _CACHE is None:
        _CACHE = bpc.VisualBehaviorOphysProjectCache.from_local_cache(cache_dir=CACHE_DIR)
    return _CACHE

ids = local_experiment_ids()
sel = et[(et.index.isin(ids)) &
         (et.project_code == PROJECT_CODE) &
         (~et.passive)].copy()
...
ds = get_cache().get_behavior_ophys_experiment(oeid)
```

iii. The notes say this is the named VisualBehavior variant and the only project completely present locally. Passive sessions lack task outcomes. The agent emphasizes that data contents are accessed through AllenSDK, not by opening NWBs directly, and uses 16 workers because NWB loading/decompression dominates runtime.

## 1-b. How are the data split into subjects?

i. Subjects are unique mouse IDs from retained sessions, sorted as strings; each session receives the corresponding index.

ii.
```python
subjects = sorted({s['mouse_id'] for s in sessions})
subject_idx = np.array([subjects.index(s['mouse_id']) for s in sessions], dtype=np.int64)
```

iii. The agent uses the SDK mouse identifier as the animal identity and verifies subject/session mappings in its sanity checks.

## 1-c. How are the data split into sessions?

i. One retained `BehaviorOphysExperiment`/ophys experiment is treated as one session. Experiments with no usable eye tracking or fewer than two usable trials are dropped.

ii.
```python
oeids = list(sel.index.values)
...
sessions = [r for r in results if not r['skip'] and r['n_trials_kept'] >= 2]
...
for s in sessions:
    neural.append(sess_neural)
```

iii. The notes justify this by stating that the selected `VisualBehavior` project is single-plane, so experiment, imaging plane, NWB, and ophys session coincide. They contrast this with excluded multiscope data, which would require merging planes on different clocks.

## 1-d. How are the data split into trials?

i. Trials are rows of `ds.trials` labeled go or catch. Each trial is the half-open interval `[start_time, stop_time)` mapped onto ophys frames. Trials with fewer than two frames are skipped.

ii.
```python
sel = trials[(trials.go.astype(bool)) | (trials.catch.astype(bool))]
...
i0, i1 = frame_slice(ophys_ts, tr.start_time, tr.stop_time)
if i1 - i0 < 2:
    continue
neural.append(activity[:, i0:i1].copy())
```

iii. The agent says this follows the experiment's own trial definition and preserves the variable pre-change duration and fixed post-change period rather than imposing a new window.

## 1-e. How are trials filtered based on quality controls?

i. Selecting `go | catch` excludes aborted and auto-rewarded trials; assertions verify that fact and that every selected trial has exactly one canonical outcome. Trials without at least two covered ophys frames are removed, and sessions need at least two retained trials.

ii.
```python
assert not sel.aborted.any()
assert not sel.auto_rewarded.any()
outcome_flags = sel[OUTCOME_NAMES].values.astype(bool)
assert (outcome_flags.sum(axis=1) == 1).all()
...
if i1 - i0 < 2:
    continue
```

iii. The notes cite AllenSDK trial logic making go/catch mutually exclusive with aborted/auto-rewarded status and report that all selected trials in the full data had coverage.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from the SDK's per-cell detrended dF/F traces, `ds.dff_traces['dff']`.

ii.
```python
ev = ds.dff_traces if NEURAL_SIGNAL == 'dff' else ds.events
activity = np.vstack(ev[NEURAL_SIGNAL].values).astype(np.float32)
```

iii. The agent argues that dF/F is an endpoint of the Allen preprocessing pipeline and is much less sparse at native 31 Hz than detected events. Its empirical decoder comparison favored dF/F for every output.

## 2-b. How is the `neural` data processed?

i. Per-cell traces are vertically stacked, converted to float32, checked against the ophys timestamp length, and sliced by trial. No normalization, smoothing, event detection, or temporal aggregation is added.

ii.
```python
activity = np.vstack(ev[NEURAL_SIGNAL].values).astype(np.float32)
assert activity.shape[1] == len(ophys_ts)
...
neural.append(activity[:, i0:i1].copy())
```

iii. The notes state that motion correction, ROI processing, demixing, neuropil subtraction, dF/F computation, and detrending were already done upstream by Allen.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Empty-cell experiments are skipped, but otherwise every cell exposed in `dff_traces` is retained; no additional cell-level filtering is applied.

ii.
```python
if len(ev) == 0:
    return {'oeid': oeid, 'skip': 'no cells'}
```

iii. The agent reports that Allen's upstream ROI filtering has already been applied and that released `valid_roi` values are all true, so further filtering would be redundant.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples are aligned to trial start by finding the first ophys timestamps at or after trial start and stop. The full trial interval is retained.

ii.
```python
def frame_slice(ophys_ts, t_start, t_stop):
    i0 = int(np.searchsorted(ophys_ts, t_start, side='left'))
    i1 = int(np.searchsorted(ophys_ts, t_stop, side='left'))
    return i0, i1
```

iii. The agent describes `trials.start_time` as the alignment event and uses the common synchronized clock for all streams.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Data remain at native single-plane ophys resolution, about 32.319 ms (30.94/31 Hz). No temporal rebinning is applied; the reported bin size is the mean of session median frame intervals.

ii.
```python
'dt': float(np.median(np.diff(ophys_ts))),
...
'time_bin_size': float(np.mean(dt_all) * 1000.0),
```

iii. The task requests ophys-timestamp alignment, and the notes say frame intervals vary only by about 0.01 ms across retained sessions.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity comes from `ds.stimulus_presentations`: the change-detection block's `image_name`, `start_time`, `end_time`, and `omitted` fields.

ii.
```python
block = stim[stim.stimulus_block_name.str.contains('change_detection', na=False)]
shown = block[~block.omitted.astype(bool)]
image_names = sorted(set(shown.image_name.unique()))
starts = shown.start_time.values
ends = shown.end_time.values
```

iii. The agent chose the presentation table because image identity is truly time-varying across flashes and grey intervals, including omissions.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A whole-session vector is initialized as grey code 0. For each non-omitted flash, frames in `[start_time, end_time)` receive a session-local image code. Local codes are later remapped to a globally sorted vocabulary consisting of grey plus all 16 images.

ii.
```python
img_local = np.zeros(n, dtype=np.int16)
...
for a, b, bc, c, ch in zip(i0, i1, i1c, codes, changes):
    img_local[a:b] = c
...
out[0] = lut[s['image_local'][k]]
```

iii. The notes interpret the requested identity as the image actually on screen: grey is a distinct class during the 500 ms interval and during omitted flashes. A global mapping makes labels comparable across image sets and sessions.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Presentation boundaries are converted to indices on `ophys_timestamps`, and the resulting session vector is sliced using the exact same trial indices as neural activity.

ii.
```python
i0 = np.searchsorted(ophys_ts, starts, side='left')
i1 = np.searchsorted(ophys_ts, ends, side='left')
...
img_tr.append(img_local[i0:i1].copy())
```

iii. The agent's frame-by-frame independent checks reportedly found complete agreement with the SDK presentation table.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change comes from `stimulus_presentations.is_change` plus each presentation's start/end times in the change-detection block.

ii.
```python
changes = shown.is_change.values.astype(bool)
...
if ch:
    is_change[a:bc] = 1
```

iii. The agent uses actual change presentations, so sham changes on catch trials remain zero and auto-reward changes do not enter retained trial windows.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero-valued session vector is set to one during the 250 ms changed-image flash. The alternative 750 ms presentation-interval mode exists only as a debug option; the shipped data use `CHANGE_WINDOW = 'flash'`.

ii.
```python
CHANGE_WINDOW = 'flash'
...
chg_ends = ends
...
if ch:
    is_change[a:bc] = 1
```

iii. The agent says 250 ms is the period in which the changed identity is visibly on screen and reports that it decoded slightly better than a 750 ms alternative.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is intrinsically binary: 1 on frames in a changed, non-omitted flash and 0 otherwise. No numerical threshold is estimated.

ii.
```python
is_change = np.zeros(n, dtype=np.uint8)
...
is_change[a:bc] = 1
```

iii. The two declared values are `no_change` and `change`; catch trials remain all zero.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Changed-flash boundaries are mapped onto ophys timestamps and then sliced with the neural trial boundaries.

ii.
```python
i1c = np.searchsorted(ophys_ts, chg_ends, side='left')
...
chg_tr.append(is_change_ts[i0:i1].copy())
```

iii. The notes report 7–9 positive frames per go trial, consistent with 250 ms at 31 Hz, exactly one positive episode per go trial, and none per catch trial.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ds.running_speed.timestamps` and `ds.running_speed.speed`.

ii.
```python
rs = ds.running_speed
t = rs.timestamps.values.astype(np.float64)
v = rs.speed.values.astype(np.float64)
```

iii. The agent uses Allen's processed wheel-speed stream in cm/s rather than recomputing speed from encoder signals.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Nonfinite speed samples are removed, values are linearly interpolated to all ophys timestamps with constant endpoint extrapolation, and then converted to global quintile labels.

ii.
```python
good = np.isfinite(v)
if not good.all():
    t, v = t[good], v[good]
return np.interp(ophys_ts, t, v).astype(np.float32)
```

iii. The notes say all clocks are hardware synchronized; global bins keep category meanings consistent across sessions and yield equal dataset-level proportions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four global 20th/40th/60th/80th percentile edges over all retained trial timepoints create five categories via `np.digitize`.

ii.
```python
qs = np.arange(1, nbins) / nbins
return np.quantile(values, qs)
...
return np.digitize(x, edges, right=False).astype(np.int64)
```

iii. This directly implements “five equal percentile bins,” with global rather than session-specific cutoffs.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running is interpolated onto the complete ophys timebase before segmentation and sliced with the same `[i0:i1]` bounds as neural activity.

ii.
```python
running = resample_running(ds, ophys_ts)
...
run_tr.append(running[i0:i1].copy())
```

iii. The agent reports visual and numerical checks showing no temporal shift.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `ds.eye_tracking.timestamps` and `pupil_area`; finite positive area implicitly excludes blink/missing frames.

ii.
```python
t = eye.timestamps.values.astype(np.float64)
area = eye.pupil_area.values.astype(np.float64)
good = np.isfinite(area) & (area > 0)
```

iii. The agent notes that pupil area is NaN exactly at likely-blink frames and uses the geometric equivalent diameter rather than `pupil_width`.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Positive pupil area is transformed to equivalent circular diameter `2*sqrt(area/pi)`. Linear interpolation bridges blink gaps and resamples to ophys timestamps, with endpoint values held constant; global quintile binning follows.

ii.
```python
diam = 2.0 * np.sqrt(area[good] / np.pi)
return np.interp(ophys_ts, t[good], diam).astype(np.float32)
```

iii. The agent says interpolation preserves trial length and alignment. It also notes the monotonic area-to-diameter transform leaves percentile membership unchanged.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four global percentile edges over all retained pupil samples define five equal-frequency categories.

ii.
```python
all_pup = np.concatenate([np.concatenate(s['pupil']) for s in sessions])
pup_edges = quantile_edges(all_pup)
...
out[3] = digitize(s['pupil'][k], pup_edges)
```

iii. Global cutoffs make the output labels comparable across sessions and produced 20% in each class in the full conversion.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Clean pupil samples are interpolated directly onto `ophys_timestamps`, then pupil and neural activity are sliced with the same trial bounds.

ii.
```python
pupil = resample_pupil(ds, ophys_ts)
...
pup_tr.append(pupil[i0:i1].copy())
```

iii. The notes cite the synchronized hardware clock and report independent reconstruction checks.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome comes from the four boolean trial-table columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
OUTCOME_NAMES = ['hit', 'miss', 'false_alarm', 'correct_reject']
outcome_flags = sel[OUTCOME_NAMES].values.astype(bool)
```

iii. These are the canonical mutually exclusive outcomes for retained go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The true column's position becomes class 0–3 via `argmax`; an assertion requires exactly one true flag. The static class is broadcast across all frames of the trial.

ii.
```python
assert (outcome_flags.sum(axis=1) == 1).all()
outcomes = np.argmax(outcome_flags, axis=1).astype(np.int64)
...
out[4] = s['outcome'][k]
```

iii. Broadcasting fits the required single `(n_outputs, T)` output array while preserving the per-trial meaning.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Nonfinite running samples are removed before interpolation; invalid pupil samples/blinks are interpolated; experiments with absent or unusable eye tracking or no cells are skipped; too-short trials and sessions with fewer than two trials are skipped. Worker exceptions are collected with tracebacks and cause the whole conversion to exit rather than silently losing a session. Assertions guard dimensions, outcomes, quantiles, and final value ranges.

ii.
```python
if good.sum() < 2:
    return None
...
if pupil is None:
    return {'oeid': oeid, 'skip': 'no eye tracking'}
...
if errors:
    sys.exit(1)
```

iii. The agent documents all three no-eye-tracking sessions dropped from the full run and favors maintaining aligned trial lengths by interpolation instead of dropping individual frames.

## 9-a. What are the most time-consuming steps of the code?

i. Loading/decompressing each experiment through AllenSDK is the dominant conversion cost; writing the 8.73 GB pickle is the next notable cost. Full-run measurements were 58.7 s for read/extract and 7.9 s to write.

ii.
```python
with Pool(min(args.workers, len(oeids))) as pool:
    for i, info in enumerate(pool.imap(extract_session_safe, oeids, chunksize=1)):
        results.append(info)
...
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes identify NWB decompression as the bottleneck and report that 16-process parallelism reduced wall time substantially.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The flash assignment loop and per-trial slicing/assembly loops remain Python loops. Flash boundary lookup is already vectorized with bulk `searchsorted`; variable-length trial arrays make complete vectorization awkward. Subject/region `.index` lookups could also be replaced by dictionaries, though their cost is negligible.

ii.
```python
for a, b, bc, c, ch in zip(i0, i1, i1c, codes, changes):
    img_local[a:b] = c
    if ch:
        is_change[a:bc] = 1
...
for k in range(s['n_trials_kept']):
```

iii. The agent specifically says it avoided an expensive trials-by-flashes loop by constructing each whole-session stimulus vector once, then slicing it per trial.

## 9-c. What processing does the code repeat multiple times?

i. Trial data are traversed once to concatenate continuous running/pupil values for global quantiles and again to digitize and assemble final outputs. With `--show-processing`, up to two sessions are reloaded and raw neural/stimulus/behavior streams are reconstructed for independent plots. `run_summary` then traverses the assembled outputs again for validation statistics.

ii.
```python
all_run = np.concatenate([np.concatenate(s['running']) for s in sessions])
...
out[2] = digitize(s['running'][k], run_edges)
...
ds = get_cache().get_behavior_ophys_experiment(oeid)  # show_processing
```

iii. The repeated plot-path loading is deliberate independent verification; the ordinary full run does not enable it. Re-traversal is needed because global quantile edges cannot be known until all sessions have been extracted.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It stores extensive per-session metadata and cell specimen IDs not used by the decoder, initially constructs full-session stimulus/behavior arrays before retaining trial slices, and keeps continuous running/pupil arrays until final binning. Optional plots reread and process data that do not enter the pickle. The final trial outcome is also broadcast over every frame, increasing storage, though this accommodates the required output representation.

ii.
```python
session_info.append({
    ...
    'cell_specimen_ids': s['cell_specimen_ids'],
})
...
out[4] = s['outcome'][k]
```

iii. The notes frame the whole-session vectors as an efficiency improvement over repeatedly searching presentations, the continuous arrays as necessary for global quantiles, metadata as provenance, and optional plotting as validation rather than decoder input.
