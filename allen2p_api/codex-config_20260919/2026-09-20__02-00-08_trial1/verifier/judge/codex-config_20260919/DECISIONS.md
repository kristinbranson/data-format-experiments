# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent constructs an AllenSDK `VisualBehaviorOphysProjectCache`, obtains the experiment table, and selects locally present, active experiments whose project code is exactly `VisualBehavior`. It discovers locally available IDs by matching the table against cached NWB filenames, then loads every selected experiment with `get_behavior_ophys_experiment`, in up to 16 worker processes.

ii.
```python
return cls.from_s3_cache(cache_dir=DATA_DIR)
...
file_ids = {int(path.stem.rsplit("_", 1)[1]) for path in
            (DATA_DIR / "visual-behavior-ophys-1.1.0" /
             "behavior_ophys_experiments").glob("*.nwb")}
keep = (table.index.isin(file_ids) & table["project_code"].eq("VisualBehavior")
        & ~table["passive"].astype(bool))
...
exp = cache.get_behavior_ophys_experiment(int(experiment_id))
```

iii. The notes say this uses the SDK for detailed access, avoids fetching absent cache entries, excludes passive replay sessions, and parallelizes expensive SDK trace decompression. The agent emphasizes that it never opens NWB contents directly.

## 1-b. How are the data split into subjects?

i. Subjects are the sorted unique string-valued `mouse_id` metadata across retained experiments; each experiment/session receives the corresponding index.

ii.
```python
"mouse_id": str(meta["mouse_id"]),
...
subjects = sorted({s["mouse_id"] for s in raw_sessions})
subject_map = {name: i for i, name in enumerate(subjects)}
```

iii. The notes identify SDK mouse ID as the animal identifier and validate the resulting 37 local mice against the supplied cache metadata.

## 1-c. How are the data split into sessions?

i. Each retained `ophys_experiment_id` is emitted as one decoder session; `ophys_session_id` is retained only as metadata. Experiments are sorted by experiment ID.

ii.
```python
ids = local_experiment_ids(cache)
...
raw_sessions.append(result)
raw_sessions.sort(key=lambda x: x["experiment_id"])
...
neural.append(s["neural"])
```

iii. The agent states that the exact single-plane `VisualBehavior` project has a one-to-one experiment/session relationship in this local release, so experiment-level loading produces the intended sessions.

## 1-d. How are the data split into trials?

i. The SDK `trials` table defines variable-length trials. For each eligible row, the agent creates a 30 Hz half-open grid from `start_time` through (but not including) `stop_time`, concatenates all grids for processing, then splits arrays back using stored lengths.

ii.
```python
def _trial_grid(start, stop):
    n = int(np.ceil((stop - start) * FS - 1e-10))
    grid = start + np.arange(max(0, n), dtype=np.float64) * DT
    return grid[grid < stop]
...
grids = [_trial_grid(float(r.start_time), float(r.stop_time))
         for r in trials.itertuples()]
```

iii. The notes justify full SDK trial intervals to preserve pre-change and response periods and a half-open interval to prevent duplicated boundary samples.

## 1-e. How are trials filtered based on quality controls?

i. Only go or catch trials are retained; aborted and auto-rewarded trials are removed. Outcomes must be exactly one of four mutually exclusive flags, grids must be nonempty, and an experiment must have at least two eligible trials. Passive experiments are filtered earlier.

ii.
```python
eligible = ((trials["go"] | trials["catch"])
            & ~trials["aborted"] & ~trials["auto_rewarded"])
...
if len(trials) < 2: raise ValueError(...)
if not np.all(outcome_bool.sum(axis=1) == 1): raise ValueError(...)
```

iii. The agent cites the explicit task requirement for go/catch and exclusion of aborted/auto-rewarded trials; it also treats exclusivity, nonempty grids, and minimum trial count as defensive validation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `exp.events`: the SDK's raw L0 calcium-event magnitude for each valid cell, with cell IDs taken from that table's index.

ii.
```python
event_table = exp.events
cell_ids = event_table.index.to_numpy(copy=True)
events = np.vstack(event_table["events"].to_numpy()).astype(np.float32)
```

iii. The notes say the paper analyzed event magnitudes and that SDK event/cell tables already reflect valid-ROI quality control; the agent therefore chose events rather than recomputing fluorescence or dF/F.

## 2-b. How is the `neural` data processed?

i. Every cell's event trace is linearly interpolated from `ophys_timestamps` onto the concatenated 30 Hz trial grids, converted to float32, and split into neuron-by-time trial matrices. A shared vectorized timestamp-bracketing routine avoids one interpolation search per neuron.

ii.
```python
neural_all = interp_rows_shared_time(ophys_t, events, all_t)
neural = [neural_all[:, offsets[i]:offsets[i + 1]]
          for i in range(len(lengths))]
```

iii. The agent says the paper interpolated calcium events to 30 Hz and reports verifying the optimized interpolation against independent `np.interp` to within about `4.77e-7`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no additional neuron-level filtering beyond the cells exposed in the SDK event table. Sessions require at least one cell through internal validation; all-zero event trials are retained.

ii.
```python
cell_ids = event_table.index.to_numpy(copy=True)
...
assert len(data["brain_region_idx"][si]) == ncell and ncell > 0
```

iii. The agent argues SDK valid ROIs already embody the whitepaper QC. It independently confirmed that all-zero trials are genuine sparse-event trials and avoided activity-based filtering that would bias the sample.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples are placed on a trial-relative grid whose zero is `trials.start_time`; absolute grid times drive interpolation against synchronized ophys timestamps. Each trial ends at its SDK stop time.

ii.
```python
grids = [_trial_grid(float(r.start_time), float(r.stop_time))
         for r in trials.itertuples()]
...
"temporal_alignment_event": "trial start (AllenSDK trials.start_time)",
```

iii. The notes say this honors ophys-timestamp alignment while representing the complete trial and that independent reconstruction found no offset or boundary leakage.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is exactly 30 Hz, or 33.333 ms. The code performs linear temporal resampling from the native ophys frame times; it does not aggregate/count events into bins.

ii.
```python
FS = 30.0
DT = 1.0 / FS
...
"time_bin_size": 1000.0 / FS,
"neural_resampling": "linear interpolation from synchronized ophys timestamps",
```

iii. The agent attributes 30 Hz interpolation to the paper's methods and distinguishes it from spike/event counting.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from task-block rows of `exp.stimulus_presentations`, specifically `start_time`, `end_time`, `image_name`, and `omitted`. Unfilled time bins retain the explicit gray class.

ii.
```python
stim = exp.stimulus_presentations
task = stim[stim["stimulus_block_name"].str.contains("change_detection", na=False)]
...
if hi > lo and not bool(row.omitted):
    name = str(row.image_name)
    image_all[idx[valid]] = IMAGE_TO_INT[name]
```

iii. The agent says presentation intervals are the direct source for the flashed image/gray cadence and reports a non-gray fraction consistent with 250 ms image plus 500 ms gray.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A fixed global mapping assigns 0 to gray and 1–16 to the A/B natural images. Each 30 Hz point within a non-omitted task presentation gets that image's code; other points remain gray.

ii.
```python
IMAGE_VALUES = ["gray", "im000", ..., "im106"]
IMAGE_TO_INT = {name: idx for idx, name in enumerate(IMAGE_VALUES)}
image_all = np.zeros(len(all_t), dtype=np.int16)
```

iii. The fixed mapping ensures cross-session consistency and explicitly represents the requested non-gray image identity together with inter-presentation gray periods.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Stimulus interval membership is evaluated directly at the same concatenated 30 Hz timestamps (`all_t`) used to interpolate neural events, then split by the same lengths.

ii.
```python
lo = int(np.searchsorted(all_t, start, side="left"))
hi = int(np.searchsorted(all_t, end, side="left"))
valid = (all_t[idx] >= start) & (all_t[idx] < end)
...
image_trials = split_vector(s["image"], lengths)
```

iii. The agent notes that common target timestamps guarantee framewise alignment and that the explicit interval condition prevents labels in excluded inter-trial gaps from leaking into retained trials.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change uses `is_change`, `omitted`, and the start/end times of task rows in `stimulus_presentations`.

ii.
```python
if hi > lo and not bool(row.omitted):
    ...
    if bool(row.is_change):
        change_all[idx[valid]] = 1
```

iii. The notes explain that `is_change` directly identifies a changed-image presentation; catch/sham and omitted presentations therefore stay zero.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero-initialized binary vector is set to one throughout each actual changed-image presentation (about 250 ms), not merely at a single sample.

ii.
```python
change_all = np.zeros(len(all_t), dtype=np.int16)
...
change_all[idx[valid]] = 1
```

iii. After a one-bin target decoded poorly, the agent revised it to the full changed-image flash, arguing this is a sampling-robust operationalization of “right after” and improves decoding.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary: 0 is `no_change`, and 1 is `change`; no numeric thresholding is applied.

ii.
```python
change_all = np.zeros(len(all_t), dtype=np.int16)
...
["no_change", "change"],
```

iii. The source `is_change` flag is boolean, so the agent considered further thresholding unnecessary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change labels are assigned by stimulus intervals on exactly the same `all_t` grid as neural activity and split using identical trial boundaries.

ii.
```python
valid = (all_t[idx] >= start) & (all_t[idx] < end)
change_all[idx[valid]] = 1
...
change_trials = split_vector(s["change"], lengths)
```

iii. The notes cite shared target timestamps and visual/independent checks that each change label coincides with the changed-image flash.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from the SDK `exp.running_speed` table's `timestamps` and `speed` columns.

ii.
```python
running = exp.running_speed
run_all = finite_interp(running["timestamps"], running["speed"], all_t,
                        "running speed")
```

iii. The agent describes this as the SDK's reference-filtered wheel speed in cm/s.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Finite samples are stably sorted, duplicate timestamps removed, and linearly interpolated to the 30 Hz grid. Four cohort-wide quantiles (20/40/60/80%) are computed, and `searchsorted(..., side="right")` assigns classes 0–4.

ii.
```python
run_all = finite_interp(..., all_t, "running speed")
...
edges = np.quantile(joined, [0.2, 0.4, 0.6, 0.8])
run_bin = np.searchsorted(run_edges, s["running"], side="right").astype(np.int16)
```

iii. The agent justifies interpolation by synchronized clocks and global quintiles by the requirement for five equal percentile bins and consistent categories across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four global cohort thresholds at the 20th, 40th, 60th, and 80th percentiles form five bins; repeated/non-increasing edges are rejected.

ii.
```python
edges = np.quantile(joined, [0.2, 0.4, 0.6, 0.8])
if np.any(np.diff(edges) <= 0): raise ValueError(...)
np.searchsorted(run_edges, s["running"], side="right")
```

iii. The notes report each resulting class contains 20% of all retained time samples to rounding.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is interpolated directly onto the same `all_t` target timestamps used for neural traces, then split with the same trial lengths.

ii.
```python
neural_all = interp_rows_shared_time(ophys_t, events, all_t)
run_all = finite_interp(..., all_t, "running speed")
```

iii. The agent says hardware-synchronized timestamps and a common target grid provide direct temporal alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter comes from `exp.eye_tracking`: timestamps plus both `pupil_width` and `pupil_height`. It is defined as twice the larger ellipse half-axis.

ii.
```python
pupil_diameter = 2.0 * np.maximum(
    eye["pupil_width"].to_numpy(dtype=np.float64),
    eye["pupil_height"].to_numpy(dtype=np.float64),
)
```

iii. The agent cites the SDK/whitepaper interpretation of width and height as ellipse half-axes and diameter as the full major axis. It says cleaned columns contain NaNs for likely blinks.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The major-axis diameter is computed, nonfinite samples (including blink/outlier gaps) are discarded, remaining points are sorted/deduplicated and linearly interpolated to 30 Hz. Global 20/40/60/80% quantiles then produce five classes.

ii.
```python
good = np.isfinite(source_t) & np.isfinite(source_x)
...
return np.interp(target_t, t, x).astype(np.float32)
...
pupil_bin = np.searchsorted(pupil_edges, s["pupil"], side="right")
```

iii. The agent says this avoids blink contamination, follows the paper's interpolation, and yields balanced cohort-wide categories.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four global thresholds at pupil-diameter percentiles 20, 40, 60, and 80 define integer classes 0–4; degenerate edges are rejected.

ii.
```python
pupil_edges = percentile_edges([s["pupil"] for s in raw_sessions], "pupil")
pupil_bin = np.searchsorted(pupil_edges, s["pupil"], side="right").astype(np.int16)
```

iii. The notes report the five pupil classes are globally equal in frequency to within one sample.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Cleaned pupil diameter is interpolated directly to the same 30 Hz `all_t` timestamps as neural data and split using the same stored trial lengths.

ii.
```python
pupil_all = finite_interp(eye["timestamps"], pupil_diameter, all_t,
                          "pupil diameter")
```

iii. The agent cites synchronized SDK clocks and independent trial-level reconstruction checks.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It is derived from the trials table's boolean `hit`, `miss`, `false_alarm`, and `correct_reject` columns.

ii.
```python
OUTCOME_COLUMNS = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_bool = trials[OUTCOME_COLUMNS].to_numpy(dtype=bool)
```

iii. The agent calls these the SDK's canonical mutually exclusive outcomes and explicitly verifies exclusivity.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. `argmax` maps the four exclusive booleans to codes 0–3, then the selected code is repeated across every time point of that trial.

ii.
```python
outcomes = outcome_bool.argmax(axis=1).astype(np.int16)
...
np.full(int(length), s["outcome"][i], dtype=np.int16)
```

iii. The static repetition satisfies the common `(outputs, time)` representation while preserving a per-trial target; the notes verify aggregate codes exactly against the SDK counts.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Running/pupil interpolation drops nonfinite and duplicate-time samples and requires two unique finite points. Empty eye tables or insufficient required behavioral samples cause the experiment to be explicitly excluded and logged. Other extraction errors abort the conversion. Empty trial grids, too few trials, nonexclusive outcomes, length mismatches, unknown images, nonfinite final arrays, and degenerate percentile edges raise errors. Interpolation uses endpoint values outside sampled support.

ii.
```python
good = np.isfinite(source_t) & np.isfinite(source_x)
if good.sum() < 2: raise ValueError(...)
...
except MissingRequiredData as exc:
    exclusions.append({"experiment_id": int(eid), "reason": str(exc)})
    continue
except Exception as exc:
    raise RuntimeError(f"experiment {eid} failed") from exc
```

iii. The agent argues that excluding three sessions with empty eye tables is preferable to fabricating required pupil labels, while retaining and documenting genuine all-zero neural trials. It uses assertions and independent sanity checks rather than silently repairing structural inconsistencies.

## 9-a. What are the most time-consuming steps of the code?

i. SDK construction and compressed event-trace reads/decompression dominate. Finalizing, validating, and serializing the 7.6 GiB pickle are secondary costs.

ii.
```python
with ProcessPoolExecutor(max_workers=workers) as pool:
    futures = {pool.submit(extract_session, eid): eid for eid in ids}
```

iii. Timed optimization in the notes found decompression, rather than interpolation, became the full-cache bottleneck; 16 independent workers reduced the full conversion to about 73 seconds.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent already vectorized interpolation across neurons and concatenated all trial grids. Remaining loops over stimulus presentations, trial splitting/assembly, sessions, and validation could be further batched, though variable trial lengths make some looping natural.

ii.
```python
for row in task.itertuples(): ...
for i, length in enumerate(lengths): ...
for si in range(ns): ...
```

iii. The notes specifically identify per-neuron/per-trial interpolation as avoidable repetition and say shared bracketing removed it. They regard SDK I/O/decompression as more important than the remaining lightweight loops.

## 9-c. What processing does the code repeat multiple times?

i. Each worker reconstructs its own cache object; output construction performs several separate `split_vector` passes with repeated cumulative-offset calculation; validation traverses all trial arrays again; optional plotting splits running and pupil again. Stimulus rows are also searched against the concatenated timeline one by one.

ii.
```python
cache = make_cache()  # inside extract_session, once per worker task
...
run_trials = split_vector(run_bin, lengths)
pupil_trials = split_vector(pupil_bin, lengths)
image_trials = split_vector(s["image"], lengths)
change_trials = split_vector(s["change"], lengths)
```

iii. The notes focus on repetition that was removed (one timestamp search per neuron/trial). They accept remaining passes as modest relative to decompression and useful for validation/readability.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several extracted intermediates are retained only transiently or for metadata/diagnostics: trial IDs, starts/stops, continuous running and pupil after discretization, cell IDs beyond their count, detailed counters, and timing. Optional processing plots are diagnostic and not consumed by training. There is otherwise little heavy discarded processing because fluorescence, masks, and images are not loaded.

ii.
```python
result = {"trial_ids": ..., "trial_starts": ..., "trial_stops": ...,
          "running": run_all, "pupil": pupil_all, ...}
...
data = finalize(raw_sessions, run_edges, pupil_edges, exclusions=exclusions)
```

iii. The agent explicitly says it avoids loading ROI masks, fluorescence, and image data, keeps only needed output-ready intermediates, and uses optional plots/metadata for auditability and sanity checking.
