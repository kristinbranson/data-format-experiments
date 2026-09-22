# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local experiment metadata CSV, intersects it with locally present `behavior_ophys_experiment_*.nwb` files, excludes passive experiments, and opens each retained NWB directly with `h5py`. It makes two complete passes over the files: one for global bin edges and one for conversion.

ii.
```python
exp_table = pd.read_csv(table_path)
available_files = {int(path.stem.split("_")[-1]): path
                   for path in sorted(experiment_dir.glob("behavior_ophys_experiment_*.nwb"))}
exp_table = exp_table[exp_table["ophys_experiment_id"].isin(available_files)].copy()
exp_table = exp_table[~exp_table["passive"]].copy()
with h5py.File(session.filepath, "r") as f:
    ...
```

iii. The notes justify direct HDF5 access because the local AllenSDK/NWB stack could not instantiate the files, and say the converter uses the same processed NWB sources. It intentionally limits “all data” to the locally available active subset.

## 1-b. How are the data split into subjects?

i. Subjects are the sorted unique `mouse_id` strings among retained experiment files; each converted session gets the corresponding index.

ii.
```python
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
subject_idx[idx - 1] = subject_to_idx[session.mouse_id]
```

iii. The agent treats the metadata table's mouse ID as the canonical animal identifier and reports 38 mice in the local subset.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id` (one imaging plane/NWB) is treated as a separate decoder session, even when multiple files share an `ophys_session_id`.

ii.
```python
for row in exp_table.itertuples(index=False):
    sessions.append(SessionMeta(ophys_experiment_id=int(row.ophys_experiment_id), ...))
...
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(session= session, ...)
```

iii. The notes explicitly justify this as “AllenSDK experiment granularity” and because each plane has its own neuron set. This differs from reconstructing a simultaneous session by grouping planes with the same `ophys_session_id`.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. Each retained row is segmented from its `start_time` through `stop_time` on a newly constructed 30 Hz grid, so trial lengths vary.

ii.
```python
trials = read_interval_table(f["intervals"]["trials"], columns)
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
```

iii. The agent says this follows the experiment's contingent-trial definitions and uses bin centers strictly inside `[start_time, stop_time)` to avoid boundary errors.

## 1-e. How are trials filtered based on quality controls?

i. It retains go or catch trials and excludes aborted and auto-rewarded trials. Empty/invalid windows are skipped, and sessions with fewer than two usable trials are rejected. Passive sessions and sessions lacking eye tracking are also excluded upstream; all-zero neural trials are retained.

ii.
```python
trials = trials[(trials["go"] | trials["catch"]) &
                (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
if centers.size == 0:
    continue
if len(neural_trials) < 2:
    raise RuntimeError(...)
```

iii. The notes cite the requested go/catch and aborted/auto-reward filters. They retain zero-event trials after confirming that the source event arrays are genuinely zero.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are the NWB `processing/ophys/event_detection/data` event magnitudes and their timestamps, not dF/F traces.

ii.
```python
event_group = f["processing"]["ophys"]["event_detection"]
timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
events = np.asarray(event_group["data"][:], dtype=np.float32)
```

iii. The agent describes these as processed neural event magnitudes and validates them directly against the raw NWB event-detection arrays.

## 2-b. How is the `neural` data processed?

i. Time-by-cell event arrays are linearly interpolated to each trial's 30 Hz bin centers and transposed to neuron-by-time. No planes are stacked because each plane is a separate session.

ii.
```python
idx_hi = np.searchsorted(src_time, dst_time, side="left")
interp = src_value[idx_lo] * (1.0 - w[:, None]) + src_value[idx_hi] * w[:, None]
return interp.T.astype(np.float32, copy=False)
```

iii. The notes argue a common 30 Hz grid reconciles mixed native ophys rates with 30 Hz behavior and eye streams and satisfies the common-bin decoder requirement.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit `valid_roi` mask or activity filter is applied. Every row represented in the file's cell specimen table/event matrix is kept; zero-event trials are not removed.

ii.
```python
n_cells = len(cell_table["cell_specimen_id"])
return np.full(n_cells, region_index, dtype=np.int64)
```

iii. The notes report that all 29,168 locally listed cells were valid in the included files and that zero trials reflected source sparsity rather than conversion failure.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples are aligned to absolute ophys timestamps and evaluated at 30 Hz centers measured from each trial's `start_time`; metadata names “trial start” as the alignment event.

ii.
```python
centers = start_time + (np.arange(n_bins) + 0.5) * DT
neural_trial = linear_resample_matrix(ophys_time, events, centers)
"temporal_alignment_event": "trial start"
```

iii. The agent says absolute-time interpolation provides common alignment and reports independent raw reconstructions matching selected converted trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is fixed at 1/30 s (33.333 ms). Neural, running, and pupil series are linearly resampled to this grid.

ii.
```python
DT = 1.0 / 30.0
TIME_BIN_MS = DT * 1000.0
"time_bin_size": TIME_BIN_MS
```

iii. The agent chose 30 Hz as a common grid across streams and viewed event-triggered interpolation as consistent with the source methods.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from active change-detection stimulus-presentation interval tables: `image_name`, `omitted`, `start_time`, `stop_time`, and `trials_id`.

ii.
```python
columns = ["start_time", "stop_time", "image_name", "omitted", "is_change",
           "trials_id", "stimulus_block_name", "active", "duration"]
```

iii. The agent chose the presentation table because it records the actual flash/gray timing rather than only trial-level initial/change names.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global sorted image vocabulary is built with `gray` forced to code 0. Each trial starts gray; presentation bins receive the presented image code, while omitted presentations remain gray.

ii.
```python
image_values = ["gray"] + [x for x in sorted(image_names) if x != "gray"]
image_identity = np.full(centers.shape[0], image_value_to_idx["gray"])
image_identity[mask] = (image_value_to_idx["gray"] if row.omitted
                        else image_value_to_idx[str(row.image_name)])
```

iii. The notes emphasize the task wording “during the non-grey screen” and represent gray/omissions explicitly.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. The same 30 Hz centers used for neural interpolation are masked by each presentation's half-open `[start_time, stop_time)` interval.

ii.
```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. The agent's raw-data checks reconstructed both signals on the identical trial grid and found exact matches.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It uses the stimulus-presentation `is_change` flag plus presentation start/stop times and trial association.

ii.
```python
trial_presentations = presentations[presentations["trials_id"] == int(trial["id"])]
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The agent regards the processed presentation table as the authoritative record of real image changes.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero trace is initialized and bins inside an `is_change` presentation itself are set to one; the following gray interval is not marked.

ii.
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The notes describe this as a sparse change indicator derived directly from stimulus presentations (about 2.5% positive bins).

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numeric threshold is estimated: the raw boolean `is_change` becomes category 1 during its presentation, and all other bins are category 0.

ii.
```python
"output_values": [..., ["no_change", "change"], ...]
```

iii. The source flag is already categorical, so the agent applies a direct boolean encoding.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Presentation intervals are evaluated on the same 30 Hz centers used for neural interpolation.

ii.
```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
image_change[mask] = 1
```

iii. The agent cites shared absolute timestamps and exact raw reconstruction checks.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It reads timestamps and speed from `processing/running/speed` in each NWB.

ii.
```python
running_group = f["processing"]["running"]["speed"]
timestamps = np.asarray(running_group["timestamps"][:])
speed = np.asarray(running_group["data"][:])
```

iii. This is the NWB's processed wheel-speed stream and corresponds to the SDK running-speed interface.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Speed is linearly interpolated to trial centers. A first pass pools all included trial samples, computes global quintile edges, and a second pass digitizes values.

ii.
```python
running_trial = np.interp(centers, running_time, running_speed)
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
running_bin = digitize_with_edges(running_trial, running_edges)
```

iii. The agent says global quantiles yield balanced, consistent decoder classes across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five global percentile bins are produced using 0,20,40,60,80,100% edges; values are clipped to the edge range and assigned with right-sided `searchsorted`.

ii.
```python
edges = np.quantile(values, np.linspace(0.0, 1.0, 6))
bins = np.searchsorted(edges[1:-1], np.clip(values, edges[0], edges[-1]), side="right")
```

iii. The five equal-percentile requirement is implemented globally; duplicate edges are nudged monotonically by machine epsilon.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed and neural activity are independently interpolated at the exact same trial-center timestamps.

ii.
```python
neural_trial = linear_resample_matrix(ophys_time, events, centers)
running_trial = linear_resample_vector(running_time, running_speed, centers)
```

iii. The agent relies on hardware-synchronized timestamps and a shared destination grid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It reads pupil-tracking `width` and `height` plus eye-tracking timestamps from the NWB acquisition group, then defines diameter as twice the larger dimension.

ii.
```python
width = np.asarray(eye_group["pupil_tracking"]["width"][:])
height = np.asarray(eye_group["pupil_tracking"]["height"][:])
diameter = 2.0 * np.maximum(width, height)
```

iii. The agent calls this pupil diameter and excludes whole sessions without `EyeTracking`; it does not cite a paper rationale for the max-axis formula.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. NaNs in the derived diameter are interpolated over eye time, the result is linearly resampled to trial centers, and global quintile edges are computed in pass one and applied in pass two. Blink flags are not used.

ii.
```python
diameter = fill_nan_by_time(timestamps, diameter)
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
```

iii. The agent frames interpolation as robust missing-value handling and validates globally balanced bins, but its notes do not justify omitting blink rejection.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The filled/resampled pupil values are divided into five global percentile bins using the same clipping and `searchsorted` procedure as running speed.

ii.
```python
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. This directly implements five equal-percentile classes across the retained dataset.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil and neural data are interpolated from their synchronized source clocks to identical 30 Hz trial centers.

ii.
```python
neural_trial = linear_resample_matrix(ophys_time, events, centers)
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. The agent cites common absolute timestamps and raw-vs-converted spot checks.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It uses the trial table's `hit`, `miss`, `false_alarm`, and `correct_reject` boolean fields.

ii.
```python
if trial_row["hit"]: return 0
if trial_row["miss"]: return 1
if trial_row["false_alarm"]: return 2
if trial_row["correct_reject"]: return 3
```

iii. These are documented as mutually exclusive canonical outcomes for valid contingent trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The four flags map to codes 0–3 in the declared output-value order, and the selected code is repeated across every time bin of the trial. Missing outcomes raise an error.

ii.
```python
outcome_idx = trial_outcome_index(trial)
outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)
```

iii. The agent makes the static trial label time-shaped for compatibility with the decoder output matrix.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Invalid trial times are skipped; pupil NaNs are interpolated (all-NaN raises); missing eye tracking or missing presentation groups cause sessions to be skipped in pass one; duplicate quantile edges are nudged; interpolation outside source ranges uses endpoint values through `np.interp`. Unexpected errors in pass two are not caught.

ii.
```python
if not np.isfinite(start_time) or stop_time <= start_time: return np.asarray([])
values[~finite] = np.interp(time_axis[~finite], time_axis[finite], values[finite])
except KeyError as exc:
    print(f"... skip ... {exc}")
```

iii. The notes say missing required pupil data warrants session exclusion and source-zero event trials should remain. They emphasize exact raw-data cross-checks rather than broad exception suppression.

## 9-a. What are the most time-consuming steps of the code?

i. Repeated NWB I/O, per-trial interpolation (especially the time-by-neuron event matrices), global concatenation/quantiles, and optional plotting dominate. The full run took about 396 seconds.

ii.
```python
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The agent's logs time both passes per session; its documentation identifies loading and reconstruction checks as major work, while the two-pass design necessarily rereads files.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial iteration in both passes, presentation iteration inside every trial, string decoding, and plotting-neuron loops could be reduced or vectorized. In particular, presentation selection repeatedly scans the full table.

ii.
```python
for trial in trials.itertuples(index=False): ...
for trial_idx, trial in trials.iterrows():
    trial_presentations = presentations[presentations["trials_id"] == int(trial["id"])]
    for row in trial_presentations.itertuples(index=False): ...
```

iii. The agent prioritizes readable per-trial construction; no explicit optimization justification beyond successful runtime is given.

## 9-c. What processing does the code repeat multiple times?

i. Every retained NWB is opened twice. Trial tables, running data, pupil data, trial grids, and behavioral interpolation are recomputed in both passes; the second pass additionally loads neural and presentation data.

ii.
```python
# pass 1
running_trial = linear_resample_vector(...)
pupil_trial = linear_resample_vector(...)
# pass 2
running_trial = linear_resample_vector(...)
pupil_trial = linear_resample_vector(...)
```

iii. The agent uses pass one to obtain global class edges before constructing final categorical outputs; this avoids retaining all continuous data in memory but trades memory for repeated work.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Continuous running and pupil trial arrays from pass one are retained only long enough to compute quantile edges, then discarded and recomputed. Optional diagnostic plots compute extra windows, percentiles, histograms, and figures that do not enter the pickle. Several metadata/table columns are loaded but do not affect outputs.

ii.
```python
running_values.append(running_trial)
pupil_values.append(pupil_trial)
...
if show_processing and not plotted:
    make_processing_plot(...)
```

iii. The first-pass temporary data are required by the chosen global-binning architecture, while plots are explicitly diagnostic sanity checks and only run on request.
