# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local experiment metadata CSV, inventories locally present `behavior_ophys_experiment_*.nwb` files, retains metadata rows with a local file, excludes passive experiments, and opens each NWB directly with `h5py`. It makes two passes: one for global bin statistics/QC and one for conversion.

ii.
```python
exp_table = pd.read_csv(table_path)
available_files = {int(path.stem.split("_")[-1]): path for path in sorted(experiment_dir.glob("behavior_ophys_experiment_*.nwb"))}
exp_table = exp_table[exp_table["ophys_experiment_id"].isin(available_files)].copy()
exp_table = exp_table[~exp_table["passive"]].copy()
with h5py.File(session.filepath, "r") as f:
```

iii. The notes say direct HDF5 access was chosen because the installed AllenSDK/NWB stack could not instantiate these NWBs. Passive data were excluded as outside active task performance, and two passes were used to establish dataset-wide quantiles.

## 1-b. How are the data split into subjects?

i. Subjects are unique string-valued `mouse_id`s among retained experiments; each converted experiment receives the corresponding index.

ii.
```python
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
subject_idx[idx - 1] = subject_to_idx[session.mouse_id]
```

iii. The agent identifies `mouse_id` as the metadata table’s animal identifier and checks subject/session metadata against that table.

## 1-c. How are the data split into sessions?

i. Each locally available active `ophys_experiment_id` (one NWB/imaging plane) is treated as a separate converted session, sorted by experiment ID. It does not merge experiments sharing an `ophys_session_id`.

ii.
```python
for row in exp_table.itertuples(index=False):
    sessions.append(SessionMeta(ophys_experiment_id=int(row.ophys_experiment_id), ...))
...
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(session, ...)
```

iii. The notes justify this as matching `BehaviorOphysExperiment` granularity and yielding one plane/region per session.

## 1-d. How are the data split into trials?

i. Trials come from `intervals/trials`; retained trial start/stop times define a variable-duration grid of 30 Hz bin centers. Each stream is sampled on that grid.

ii.
```python
trials = get_trial_table(f)
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
n_bins = max(1, int(np.ceil((stop_time - start_time) / DT)))
centers = start_time + (np.arange(n_bins) + 0.5) * DT
```

iii. The processed NWB trial table was regarded as the authoritative Allen-processed definition, avoiding reimplementation of trial logic.

## 1-e. How are trials filtered based on quality controls?

i. Only go or catch trials are kept; aborted and auto-rewarded trials are removed. Sessions with fewer than two retained/usable trials, passive sessions, sessions without task presentations, and sessions without required eye tracking are excluded.

ii.
```python
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
if len(trials) < 2: continue
except KeyError as exc: ...
```

iii. This follows the requested contingent-trial filter; extra session exclusions were justified by decoder validity and the need for pupil output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are raw detected calcium-event magnitudes and their timestamps from NWB `processing/ophys/event_detection`, not dF/F.

ii.
```python
event_group = f["processing"]["ophys"]["event_detection"]
timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
events = np.asarray(event_group["data"][:], dtype=np.float32)
```

iii. The notes cite the paper’s use of extracted calcium events and explicitly reject visualization-only filtered events and recomputed dF/F.

## 2-b. How is the `neural` data processed?

i. The time-by-cell event matrix is linearly interpolated onto each trial’s 30 Hz bin centers and transposed to cells by time. No normalization or smoothing is added.

ii.
```python
idx_hi = np.searchsorted(src_time, dst_time, side="left")
interp = src_value[idx_lo] * (1.0 - w[:, None]) + src_value[idx_hi] * w[:, None]
return interp.T.astype(np.float32, copy=False)
```

iii. A shared 30 Hz grid was chosen because local rigs have differing native frame rates and the paper describes 30 Hz interpolation for event-triggered analysis.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No new per-neuron filter is applied. The code uses the cells/events already present in the processed NWB and derives the region-index length from `cell_specimen_table`.

ii.
```python
cell_table = f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
n_cells = len(cell_table["cell_specimen_id"])
```

iii. The notes state that upstream Allen processing already removes invalid, duplicate, union, motion-corrupted, dendritic, and otherwise poor ROIs.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute ophys timestamps are interpolated at bin centers spanning trial start to trial stop. Metadata calls the alignment event “trial start,” with offset 0 and variable end.

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
"temporal_alignment_event": "trial start", "off_start": 0.0, "off_end": None
```

iii. The agent says using absolute time before segmentation preserves synchronization among streams.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 1/30 s, or 33.333 ms. All streams, including neural events, are linearly resampled; this is interpolation rather than count-preserving aggregation.

ii.
```python
DT = 1.0 / 30.0
TIME_BIN_MS = DT * 1000.0
"time_bin_size": TIME_BIN_MS
```

iii. The common grid was intended to satisfy the same-bin-size requirement across mixed native acquisition rates.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It comes from task stimulus-presentation interval tables: `image_name`, `omitted`, start/stop time, stimulus block, and trial ID.

ii.
```python
columns = ["start_time", "stop_time", "image_name", "omitted", "is_change", "trials_id", "stimulus_block_name"]
trial_presentations = presentations[presentations["trials_id"] == int(trial["id"])]
```

iii. The agent chose presentations rather than only initial/change trial fields to represent the actual flashed stimulus timeline.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global sorted category map is built. Every bin defaults to `gray`; bins inside non-omitted presentation intervals receive that image’s code, while omissions remain gray.

ii.
```python
image_identity = np.full(centers.shape[0], image_value_to_idx["gray"], dtype=np.int64)
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
image_identity[mask] = image_value_to_idx["gray" if row.omitted else str(row.image_name)]
```

iii. The notes explicitly represent gray/ISI/omission periods because only non-gray periods have an image identity.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Presentation intervals are evaluated at the exact same 30 Hz bin centers used for neural interpolation.

ii.
```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. Shared absolute-time bin centers guarantee equal length and temporal correspondence.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It is derived from the stimulus-presentation `is_change` flag and presentation start/stop times.

ii.
```python
if bool(row.is_change):
    image_change[mask] = 1
```

iii. Trial `change_time` and image fields were used as conceptual sanity checks; the presentation table was selected for frame-level timing.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector is created, then bins within an `is_change` presentation interval are assigned one.

ii.
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
if bool(row.is_change): image_change[mask] = 1
```

iii. This marks the actual change-image flash rather than the whole post-change trial.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already boolean and is encoded directly as 0=`no_change`, 1=`change`; no numeric threshold is estimated.

ii.
```python
"output_values": [..., ["no_change", "change"], ...]
```

iii. The NWB `is_change` annotation supplies the categorical threshold.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The change interval is masked on the same trial bin centers used for neural data.

ii.
```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
image_change[mask] = 1
```

iii. The diagnostic plots were used to visually verify alignment to the change flash.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses NWB `processing/running/speed` data and timestamps.

ii.
```python
running_group = f["processing"]["running"]["speed"]
timestamps = np.asarray(running_group["timestamps"][:])
speed = np.asarray(running_group["data"][:])
```

iii. The notes identify this as Allen’s processed, filtered running speed in cm/s.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Speed is linearly interpolated to trial centers, then discretized using global quintile edges collected in pass 1.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
running_bin = digitize_with_edges(running_trial, running_edges)
```

iii. Global rather than per-session quantiles provide consistent category meanings and balanced classes.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five dataset-wide equal-percentile bins are used; values are clipped to the observed edge range and assigned codes 0–4.

ii.
```python
edges = np.quantile(values, np.linspace(0.0, 1.0, nbins + 1))
bins = np.searchsorted(edges[1:-1], np.clip(values, edges[0], edges[-1]), side="right")
```

iii. This directly implements the requested five equal percentile bins.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running is interpolated at the same `centers` as neural activity.

ii.
```python
neural_trial = linear_resample_matrix(ophys_time, events, centers)
running_trial = linear_resample_vector(running_time, running_speed, centers)
```

iii. Absolute hardware-synchronized timestamps and the shared grid are the stated basis for alignment.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses eye-tracking timestamps plus pupil ellipse width and height from `acquisition/EyeTracking`.

ii.
```python
width = np.asarray(eye_group["pupil_tracking"]["width"][:])
height = np.asarray(eye_group["pupil_tracking"]["height"][:])
diameter = 2.0 * np.maximum(width, height)
```

iii. The agent interprets diameter as twice the larger pupil ellipse radius/axis and says blink-related missing samples are interpolated.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Diameter is computed as `2*max(width,height)`, NaNs are filled by within-session time interpolation, it is interpolated to trial centers, then globally quintile-binned.

ii.
```python
diameter = fill_nan_by_time(timestamps, 2.0 * np.maximum(width, height))
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. The notes claim this follows blink filtering and avoids losing otherwise usable sessions/samples, although the code does not read a blink flag.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Five global equal-percentile bins are computed from all finite retained trial samples, using the same edge/digitization functions as running.

ii.
```python
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. Global quintiles make the output consistent and approximately balanced.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is linearly interpolated to the same trial centers as neural activity.

ii.
```python
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. Alignment is based on absolute synchronized timestamps and verified in diagnostic plots.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It comes from mutually exclusive trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
if bool(trial_row["hit"]): return 0
if bool(trial_row["miss"]): return 1
if bool(trial_row["false_alarm"]): return 2
if bool(trial_row["correct_reject"]): return 3
```

iii. These are treated as the canonical Allen outcome labels; unlabeled retained trials raise an error.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The four labels map to codes 0–3 in fixed order, and the selected code is repeated across every time bin of the trial.

ii.
```python
OUTCOME_NAMES = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)
```

iii. Repeating the static label produces a uniform `(n_output,T)` representation.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing eye-tracking sessions and sessions lacking task presentations are skipped; pupil NaNs are time-interpolated (including endpoint extrapolation by nearest finite value); invalid trial bounds are skipped; duplicate quantile edges are separated by epsilon; an invalid outcome or all-NaN pupil series raises an error.

ii.
```python
if "EyeTracking" not in f["acquisition"]: raise KeyError(...)
values[~finite] = np.interp(time_axis[~finite], time_axis[finite], values[finite])
if centers.size == 0: continue
edges = np.maximum.accumulate(edges + np.arange(edges.size) * eps)
```

iii. The strategy is to repair local missing pupil samples but exclude sessions missing an entire required stream, while refusing ambiguous outcome labels.

## 9-a. What are the most time-consuming steps of the code?

i. Reading full NWB event matrices and processing every file twice dominate runtime; decoder training is also expensive but outside conversion.

ii.
```python
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
for session in kept_sessions: convert_session(...)
events = np.asarray(event_group["data"][:], dtype=np.float32)
```

iii. The notes explicitly call conversion I/O-heavy and note that two-pass global binning reads each NWB twice.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial iteration, stimulus-presentation iteration within each trial, string decoding, metadata row construction, and pass-1 per-trial resampling remain Python loops. Neural interpolation itself is vectorized across cells.

ii.
```python
for trial_idx, trial in trials.iterrows():
    for row in trial_presentations.itertuples(index=False):
for value in values: ...
```

iii. The notes emphasize the implemented vectorized `searchsorted`/broadcast interpolation as the major speedup; remaining variable-length trial operations favor clarity.

## 9-c. What processing does the code repeat multiple times?

i. Files, trial tables, running, and pupil data are read in both passes; trial grids and behavior interpolation are recomputed. Pass 2 additionally reads neural and presentation data.

ii.
```python
# pass 1
running_time, running_speed = get_running_data(f)
pupil_time, pupil_diameter = get_pupil_data(f)
# pass 2
running_time, running_speed = get_running_data(f)
pupil_time, pupil_diameter = get_pupil_data(f)
```

iii. Repetition is the deliberate memory-saving cost of global discretization without retaining full-session arrays.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Pass-1 resampled running and pupil trial arrays are used only to derive bin edges and then discarded. With `--show-processing`, raw windows and plots are diagnostic-only. Metadata fields are read beyond what conversion ultimately stores, and the main run reads presentation data once in each pass although pass 1 primarily needs image names.

ii.
```python
running_values.append(running_trial)
pupil_values.append(pupil_trial)
running_all = np.concatenate(running_values)
...
make_processing_plot(...)
```

iii. The notes characterize the first pass and optional plots as necessary global-statistics/validation work rather than decoder inputs, accepting extra I/O for bounded memory and sanity checking.
