# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent creates an AllenSDK `VisualBehaviorOphysProjectCache` for `/app/data`, intersects the SDK experiment table with locally cached NWB asset IDs, retains only `behavior_type == "active_behavior"`, and loads each selected experiment through `get_behavior_ophys_experiment`. Full mode uses four worker processes, each with its own cache object.

ii.
```python
cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=str(CACHE_DIR))
table = active_experiment_table(cache)
...
obj = cache.get_behavior_ophys_experiment(int(experiment_id))
```

iii. The notes justify SDK-only access as required, active-only selection because passive experiments lack meaningful trial outcomes, and cache-file intersection to avoid requesting unavailable assets. Parallel workers were chosen to reduce SDK loading time.

## 1-b. How are the data split into subjects?

i. Subjects are the sorted unique string-valued `mouse_id` fields among retained experiment records; each experiment gets the corresponding subject index.

ii.
```python
subjects = sorted({s["mouse_id"] for s in sessions})
subject_map = {x: i for i, x in enumerate(subjects)}
"subject_idx": np.asarray([subject_map[s["mouse_id"]] for s in sessions], dtype=np.int64),
```

iii. The notes identify `mouse_id` as the animal identifier and report that 38 subjects matched the active local cache.

## 1-c. How are the data split into sessions?

i. Each retained `ophys_experiment_id` (one imaging plane) is treated as a separate target-format session. `ophys_session_id` is retained only as metadata; experiments sharing it are not merged.

ii.
```python
for i, (experiment_id, meta) in enumerate(table.iterrows(), start=1):
    session, exclusions = convert_experiment(cache, int(experiment_id), meta, ...)
...
neural.append(sn); inputs.append(si); outputs.append(so)
```

iii. The notes explicitly state “One target session per experiment/plane,” allowing a single region and a common frame stream per output session. This differs from the human reconstruction of an ophys session from all experiments with the same `ophys_session_id`.

## 1-d. How are the data split into trials?

i. Trials come from `obj.trials`. For each eligible row, the agent uses its complete `start_time`–`stop_time` interval, retaining only complete 100 ms bins, so trials have variable duration.

ii.
```python
for trial_id, row in eligible.iterrows():
    start, stop = float(row.start_time), float(row.stop_time)
    nbins = int(np.floor((stop - start) / DT + 1e-9))
    edges = start + np.arange(nbins + 1, dtype=float) * DT
```

iii. The notes say native trial boundaries preserve the pre-change and post-change periods, while complete bins provide a common temporal resolution.

## 1-e. How are trials filtered based on quality controls?

i. The agent requires go or catch, non-aborted, non-auto-rewarded, exactly one standard outcome, finite ordered bounds, at least two 100 ms bins, full running and pupil interpolation coverage, and go/catch consistency with stimulus `is_change`. Experiments need at least two retained trials and one cell; sessions without processed pupil data are dropped.

ii.
```python
mask = contingent & non_aborted & non_auto & one_outcome & finite_bounds & positive
...
if run is None or pupil is None:
    dropped_coverage += 1
    continue
if (is_go and nchanges != 1) or (is_catch and nchanges != 0):
    dropped_stim_consistency += 1
    continue
```

iii. The explicit exclusions implement the requested go/catch and aborted/auto-reward rules. Additional checks avoid fabricated outcomes, extrapolation, and inconsistent stimulus records; all exclusions are logged.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from the SDK `obj.events` table’s `events` arrays (unfiltered FastLZero inferred calcium-event magnitudes), indexed by SDK-valid cell specimen rows, plus `ophys_timestamps` for binning.

ii.
```python
timestamps = np.asarray(obj.ophys_timestamps, dtype=float)
events, cell_ids = stack_event_traces(obj.events, len(timestamps))
```

iii. The notes argue that inferred events match the paper’s neural representation and that event magnitudes are sparse but valid. The human reference instead uses `dff_traces.dff`.

## 2-b. How is the `neural` data processed?

i. Event traces are stacked cell-by-time and summed within each half-open 100 ms trial bin using timestamp searches and a trial-local cumulative sum. Data are stored as float32; planes are not combined.

ii.
```python
indices = np.searchsorted(timestamps, edges, side="left")
np.cumsum(local, axis=1, dtype=np.float32, out=cumulative[:, 1:])
return cumulative[:, rel[1:]] - cumulative[:, rel[:-1]]
```

iii. The notes justify sums as preserving event magnitude/count-like activity while creating a common rate near the slow multiscope frame period.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No activity- or SNR-based cell filtering is added. The agent trusts the cells exposed by the SDK events table, but rejects malformed shapes, non-finite event values, and experiments with zero cells.

ii.
```python
if traces.shape != (len(cell_ids), nframes):
    raise ValueError(...)
if not np.isfinite(traces).all():
    raise ValueError("non-finite inferred event value")
```

iii. The notes say the SDK has already selected valid ROIs and that filtering by activity would bias the decoder input and is not requested by the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural bins begin at each trial’s `start_time` and continue to the last complete 100 ms bin before `stop_time`; bin membership is determined from synchronized ophys timestamps.

ii.
```python
edges = start + np.arange(nbins + 1, dtype=float) * DT
neural = bin_events(events, timestamps, edges)
```

iii. The notes describe alignment to variable-length trial start/boundaries on the common synchronized clock, rather than a fixed window around the change event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is fixed at 100 ms, and native event samples are explicitly rebinned by summation.

ii.
```python
DT = 0.100
...
"time_bin_size": DT * 1000.0,
```

iii. The notes justify 100 ms as close to the slower ~11 Hz acquisition, common across recordings, and still fine enough for the 250 ms display and 400 ms decoding windows. The human reference keeps native ophys frames without rebinning.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from active change-detection rows of `obj.stimulus_presentations`: `image_name`, `start_time`, `end_time`, and `omitted`.

ii.
```python
stim = active_stimulus_table(obj.stimulus_presentations)
image_name = stim["image_name"].fillna("").astype(str).to_numpy()
shown = ... & (~omitted[rows]) & (image_name[rows] != "")
```

iii. The notes say stimulus intervals represent what was actually on screen and permit explicit labeling of gray and omitted periods. The human reference uses trial `initial_image_name`, `change_image_name`, and `change_time`.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. At each 100 ms bin center, the active overlapping non-omitted presentation’s name is assigned; otherwise the label is `gray`. A deterministic global vocabulary (`gray` then sorted names) maps labels to integer codes.

ii.
```python
labels = np.full(len(centers), "gray", dtype=object)
labels[shown] = image_name[rows[shown]]
...
image_values = ["gray"] + sorted({...})
```

iii. The notes justify a global mapping and treating inter-stimulus/omitted periods as genuine gray, rather than carrying forward the prior identity.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Stimulus identity is sampled at the centers of the exact 100 ms bins used to sum neural activity, producing the same `T`.

ii.
```python
centers = edges[:-1] + DT / 2.0
image_labels, change, nchanges = label_stimuli(trial_stim, centers, edges)
```

iii. The notes state that bin centers consistently define categorical and continuous labels on the same synchronized clock as neural bin edges.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change comes from active `stimulus_presentations.is_change` together with presentation `start_time` and `end_time`; trial go/catch flags are used as a consistency check.

ii.
```python
is_change = stim["is_change"].fillna(False).astype(bool).to_numpy()
change_rows = np.flatnonzero(is_change & (starts < edges[-1]) & (ends > edges[0]))
```

iii. The notes prefer the display-lag-corrected presentation row carrying `is_change` over independently using trial `change_time`.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary vector starts at zero and is set to one for bin centers within the complete on-screen interval of every `is_change` presentation, normally about 250 ms. Catch trials remain zero.

ii.
```python
for row in change_rows:
    changes[(centers >= starts[row]) & (centers < ends[row])] = 1
```

iii. The notes report that a one-bin impulse decoded poorly and was changed to the displayed changed-image interval because “right after” the identity change should not be an undersampled mathematical impulse. The human reference marks 750 ms (flash plus gray).

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is intrinsically binary: zero outside changed-image presentations and one inside them; there is no continuous thresholding.

ii.
```python
changes = np.zeros(len(centers), dtype=np.int16)
changes[condition] = 1
```

iii. The notes define output values as `no_change` and `change`; go/catch consistency is asserted.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The same 100 ms bin centers used for image identity and behavioral interpolation determine the binary change vector, so its length and clock match the binned neural array.

ii.
```python
image_labels, change, nchanges = label_stimuli(trial_stim, centers, edges)
records.append({"neural": neural, "change": change, ...})
```

iii. The notes say the changed-image interval overlays the corresponding neural bins on processing plots and independent checks.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from `obj.running_speed['timestamps']` and `obj.running_speed['speed']`.

ii.
```python
running = obj.running_speed
run = interpolate_finite(running["timestamps"].to_numpy(), running["speed"].to_numpy(), centers)
```

iii. The notes identify this as the SDK-filtered wheel speed in cm/s.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Finite samples are sorted, duplicate timestamps removed, and speed is linearly interpolated to trial bin centers without extrapolation. Pooled retained values are then categorized by global quintiles.

ii.
```python
return np.interp(query, times, values).astype(np.float32)
...
edges = np.quantile(joined[np.isfinite(joined)], [0.2, 0.4, 0.6, 0.8])
```

iii. Global percentiles were chosen to yield balanced and consistent decoder classes across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four pooled thresholds at the 20th, 40th, 60th, and 80th percentiles yield five integer categories 0–4; values equal to an edge enter the higher bin.

ii.
```python
def discretize(values, edges):
    return np.searchsorted(edges, values, side="right").astype(np.int16)
```

iii. The notes report exactly equal pooled quintile frequencies and record the thresholds in metadata.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to each neural bin center on the synchronized clock before trial assembly.

ii.
```python
centers = edges[:-1] + DT / 2.0
run = interpolate_finite(..., centers)
```

iii. The notes and plots state that aligned values overlap the SDK samples and share the neural bin count.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It comes from `obj.eye_tracking['pupil_area']` and eye-tracking timestamps. Equivalent circular diameter is computed from processed area.

ii.
```python
pupil_area = eye["pupil_area"].to_numpy(float)
pupil_diameter = 2.0 * np.sqrt(np.maximum(pupil_area, 0.0) / np.pi)
```

iii. The notes say processed pupil area already masks blink/outlier samples upstream and that equivalent diameter is preferable to raw corrupted values. The human reference instead uses `pupil_width` after filtering `likely_blink`.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Processed area is converted to equivalent diameter, finite samples are linearly interpolated to bin centers without extrapolation, and all retained values are pooled for quintile discretization. Short internal missing intervals are bridged.

ii.
```python
pupil = interpolate_finite(eye["timestamps"].to_numpy(), pupil_diameter, centers)
pupil_edges = percentile_edges([r["pupil"] for s in sessions for r in s["records"]])
```

iii. The notes justify interpolation across brief masked blinks while dropping trials outside eye-data coverage.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four global pooled percentile thresholds (20/40/60/80%) create five integer classes 0–4, with equality assigned upward.

ii.
```python
discretize(r["pupil"], pupil_edges)
```

iii. This mirrors running-speed quintiles, balances classes globally, and stores thresholds in metadata.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Equivalent diameter is interpolated at the same 100 ms bin centers as every other time-varying output.

ii.
```python
pupil = interpolate_finite(eye["timestamps"].to_numpy(), pupil_diameter, centers)
```

iii. Hardware-synchronized timestamps and a shared center grid are the stated alignment basis; uncovered trials are dropped instead of extrapolated.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It is derived from the four boolean trial columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
OUTCOME_COLUMNS = ["hit", "miss", "false_alarm", "correct_reject"]
flags = np.asarray([bool(row[x]) for x in OUTCOME_COLUMNS])
```

iii. The notes call these the canonical mutually exclusive contingent outcomes and reject trials without exactly one.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The true flag’s position maps to code 0–3, and the static code is repeated over every time bin so it can share a `(5,T)` output matrix.

ii.
```python
code = int(np.flatnonzero(flags)[0])
outcome = np.full(T, r["outcome"], dtype=np.int16)
```

iii. The notes explain that repetition is a formatting choice for combining static and time-varying requested outputs.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Non-finite behavioral samples are omitted before interpolation; duplicate timestamps are removed; interpolation is forbidden outside coverage; uncovered/short/inconsistent trials and no-eye or under-two-trial experiments are dropped and counted. Malformed/non-finite neural traces raise an error. Per-experiment fatal errors are logged while conversion continues.

ii.
```python
good = np.isfinite(times) & np.isfinite(values)
...
if len(times) < 2 or query[0] < times[0] or query[-1] > times[-1]:
    return None
...
except Exception as exc:
    excluded[int(experiment_id)] = {"fatal_error": repr(exc)}
```

iii. The notes emphasize not extrapolating or inventing pupil/outcome values, retaining a full exclusion ledger, and interpolating only short internal gaps. Three experiments without processed pupil data were excluded.

## 9-a. What are the most time-consuming steps of the code?

i. SDK/NWB object construction and loading full event arrays dominate; stacking full-session events, per-trial aggregation, final assembly, and writing a 2.74 GB pickle are secondary costs.

ii.
```python
obj = cache.get_behavior_ophys_experiment(int(experiment_id))
events, cell_ids = stack_event_traces(obj.events, len(timestamps))
```

iii. The notes measured full conversion at 348 seconds and used four experiment workers, estimating loading/processing as the bottleneck.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The Python loop over trials, the loop over change-presentation rows, the per-trial image-label-to-code generator, and assembly loops could potentially be batched/vectorized. Within a trial, the expensive neuron-by-bin work already uses `searchsorted`, cumulative sums, NumPy interpolation, and boolean masks.

ii.
```python
for trial_id, row in eligible.iterrows():
...
for row in change_rows:
...
image = np.fromiter((image_map[str(x)] for x in r["image_labels"]), ...)
```

iii. The notes explicitly say vectorized binning/interpolation avoids neuron/bin Python loops and that SDK loading dominates, making more complex cross-trial vectorization a lower priority.

## 9-c. What processing does the code repeat multiple times?

i. Each worker reconstructs an SDK cache object; behavior arrays are converted to NumPy inside every trial; output assembly traverses every trial a second time after conversion; and image strings are mapped element by element after already being labeled. These repetitions support process safety and the two-pass need for global quantile/class mappings.

ii.
```python
cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=str(CACHE_DIR))
...
running["timestamps"].to_numpy()
...
for s in sessions:
    for r in s["records"]:
```

iii. The notes acknowledge independent cache/session construction for safe multiprocessing and retain continuous values until pooled global edges can be computed; thus much of the repetition is intentional.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In normal full conversion, little computed data are wholly discarded beyond temporary full-session arrays and bookkeeping. Trial `start_time`/`stop_time`, IDs, exclusions, and cell IDs are not decoder features but are preserved as metadata. With `--show-processing`, full diagnostic mean-event/running/pupil streams are retained only to create plots and then are not included in the output.

ii.
```python
"diagnostic": {...} if keep_diagnostic else None,
...
if show_processing:
    for s in sessions[:2]:
        plot_processing(s, run_edges, pupil_edges)
```

iii. The notes justify diagnostics as sanity checks and disable them in the full parallel path. Temporary full arrays are released with `del` and `gc.collect()` to control memory.
