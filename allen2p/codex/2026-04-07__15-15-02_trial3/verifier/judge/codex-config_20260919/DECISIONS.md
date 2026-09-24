# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the local experiment metadata CSV and enumerates locally present `behavior_ophys_experiment_*.nwb` files. It retains metadata rows with a local file, removes passive experiments, sorts by experiment ID, and reads each NWB directly with `h5py`. It therefore processes the locally supplied active subset, not every experiment listed in the release or data fetched through the AllenSDK cache.

ii.
```python
exp_table = pd.read_csv(table_path)
available_files = {
    int(path.stem.split("_")[-1]): path
    for path in sorted(experiment_dir.glob("behavior_ophys_experiment_*.nwb"))
}
exp_table = exp_table[exp_table["ophys_experiment_id"].isin(available_files)].copy()
exp_table = exp_table[~exp_table["passive"]].copy()
```

iii. The notes say direct HDF5 access was necessary because the installed AllenSDK/NWB stack could not instantiate these files. They justify active-only data as the task-performing subset: passive sessions make behavioral outcomes degenerate. The trajectory explicitly records this scope decision.

## 1-b. How are the data split into subjects?

i. Subjects are unique string-valued `mouse_id`s among retained experiment files. They are sorted globally, mapped to integer indices, and each converted experiment gets the corresponding index.

ii.
```python
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
subject_idx[idx - 1] = subject_to_idx[session.mouse_id]
```

iii. The AI identifies `mouse_id` as the release metadata's animal identifier and verifies subject/session metadata against the experiment table.

## 1-c. How are the data split into sessions?

i. Each retained `ophys_experiment_id` NWB is treated as a separate converted session. Experiments are not grouped by common `ophys_session_id`, so simultaneous imaging planes from one behavioral session remain separate entries.

ii.
```python
for row in exp_table.itertuples(index=False):
    sessions.append(SessionMeta(
        ophys_experiment_id=int(row.ophys_experiment_id),
        ophys_session_id=int(row.ophys_session_id),
        filepath=available_files[int(row.ophys_experiment_id)],
        ...
    ))
```

iii. The notes justify this as matching the `BehaviorOphysExperiment`/local-NWB granularity and yielding one plane and one targeted structure per entry.

## 1-d. How are the data split into trials?

i. Trials come from the processed NWB `intervals/trials` table. For every retained row, the AI builds a variable-length 30 Hz grid of bin centers spanning `start_time` to `stop_time`; all streams are sampled onto this grid.

ii.
```python
trials = read_interval_table(f["intervals"]["trials"], columns)
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
```

iii. The AI calls the processed trials table the authoritative Allen-processed trial definition and uses full trial bounds to retain pre-change and post-change activity.

## 1-e. How are trials filtered based on quality controls?

i. It keeps only GO or CATCH trials and explicitly excludes aborted and auto-rewarded trials. Invalid/empty time windows are skipped, sessions with fewer than two kept trials are removed, passive sessions are excluded earlier, and sessions without required eye tracking are skipped.

ii.
```python
trials = trials[(trials["go"] | trials["catch"]) &
                (~trials["aborted"]) &
                (~trials["auto_rewarded"])].copy()
if centers.size == 0:
    continue
if len(trials) < 2:
    continue
```

iii. GO/CATCH inclusion and aborted/auto-reward exclusion follow the task literally. The notes justify excluding passive and eye-tracking-missing sessions because valid outcome and pupil targets are required.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are raw detected calcium-event magnitudes from `processing/ophys/event_detection/data`, with their `timestamps`; dF/F is not used.

ii.
```python
event_group = f["processing"]["ophys"]["event_detection"]
timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
events = np.asarray(event_group["data"][:], dtype=np.float32)
```

iii. The AI says the paper's neural analyses use extracted calcium events and that raw event magnitudes best reproduce that processing; it avoids visualization-only filtered events.

## 2-b. How is the `neural` data processed?

i. The time-by-cell event matrix is linearly interpolated onto each trial's common 30 Hz bin centers using vectorized `searchsorted` and broadcasting, then transposed to neuron-by-time `float32`. No normalization or smoothing is added.

ii.
```python
idx_hi = np.searchsorted(src_time, dst_time, side="left")
interp = src_value[idx_lo] * (1.0 - w[:, None]) + src_value[idx_hi] * w[:, None]
return interp.T.astype(np.float32, copy=False)
```

iii. The common grid was chosen because native rates vary across rigs and the paper describes 30 Hz interpolation for event-triggered analyses. The implementation vectorizes across neurons for speed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level filtering is performed in the converter. It uses the event matrix and cell-specimen count already stored in the processed NWB, relying on upstream Allen ROI curation. All-zero event trials are retained.

ii.
```python
cell_table = f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
n_cells = len(cell_table["cell_specimen_id"])
return np.full(n_cells, region_index, dtype=np.int64)
```

iii. The notes identify Allen's invalid-ROI, duplicate/union, motion-edge, dendrite, and low-quality ROI removal as upstream processing. Raw checks confirmed warned all-zero trials existed in the source rather than being conversion errors.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute ophys/event timestamps are interpolated to bin centers beginning half a bin after trial `start_time` and ending before `stop_time`. Thus time zero is trial start, and every output uses the identical centers.

ii.
```python
centers = start_time + (np.arange(n_bins, dtype=np.float64) + 0.5) * DT
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. The AI describes this as alignment on absolute ophys time followed by trial cutting, which keeps all streams synchronized and permits variable trial duration.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output resolution is 1/30 s, or 33.333 ms. Neural and behavioral streams are linearly resampled from their native timestamp grids; stimulus categories are assigned by interval membership.

ii.
```python
DT = 1.0 / 30.0
TIME_BIN_MS = DT * 1000.0
```

iii. The AI chose 30 Hz to impose the required common bin size across mixed native acquisition rates and cites the paper's 30 Hz event-triggered interpolation.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from change-detection stimulus-presentation interval tables: `image_name`, `omitted`, `start_time`, `stop_time`, and `trials_id`.

ii.
```python
columns = ["start_time", "stop_time", "image_name", "omitted",
           "is_change", "trials_id", "stimulus_block_name", ...]
trial_presentations = presentations[presentations["trials_id"] == int(trial["id"])]
```

iii. The AI prefers presentation records over trial-level initial/change names because they preserve every flash, gray interval, and omission.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Global image names are sorted and encoded. Each trial starts as `gray`; bins within a non-omitted presentation interval receive that image's code, while omitted presentations and inter-stimulus intervals remain `gray`.

ii.
```python
image_identity = np.full(centers.shape[0], image_value_to_idx["gray"], dtype=np.int64)
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
if bool(row.omitted) or str(row.image_name) == "omitted":
    image_identity[mask] = image_value_to_idx["gray"]
else:
    image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. The notes say gray/omission periods must be explicit because trial time includes them, while the requested identity is the image shown during non-gray periods.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Presentation start/stop intervals are evaluated at the same 30 Hz bin centers used for neural interpolation, producing exactly one identity code per neural time bin.

ii.
```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. The AI reports direct raw-NWB spot checks of stimulus traces and neural timing, plus diagnostic plots, as alignment validation.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It is derived from `is_change`, `start_time`, and `stop_time` in the task stimulus-presentation rows associated through `trials_id`.

ii.
```python
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The AI uses the SDK-derived presentation-level change flag and notes that trial `change_time` and image fields are available for sanity checking.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero-filled integer trace is created, and bins lying inside any presentation marked `is_change` are set to one. No smoothing or extension into the following gray interval is applied.

ii.
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The notes characterize this as a change-image-flash indicator based directly on Allen's presentation semantics.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary: `0` for `no_change` and `1` during a change presentation. No numerical threshold is estimated.

ii.
```python
"output_values": [image_values, ["no_change", "change"], ...]
```

iii. The source `is_change` flag supplies the categorical threshold, so additional thresholding was unnecessary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The same trial bin centers define both the interpolated neural samples and interval mask for the change flash.

ii.
```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
image_change[mask] = 1
```

iii. The AI validated this visually and with independent raw-file equality checks.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from `processing/running/speed/data` and its timestamps.

ii.
```python
running_group = f["processing"]["running"]["speed"]
timestamps = np.asarray(running_group["timestamps"][:], dtype=np.float64)
speed = np.asarray(running_group["data"][:], dtype=np.float64)
```

iii. The AI treats this as the SDK-processed, filtered running speed in cm/s rather than recomputing wheel kinematics.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Speed is linearly interpolated onto every trial's 30 Hz centers. A first pass pools all retained trial samples to compute global quantile edges; a second pass digitizes values.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
running_bin = digitize_with_edges(running_trial, running_edges)
```

iii. Global rather than per-session edges give consistent class meaning and approximately equal class frequency across the dataset.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. It is divided into five global equal-percentile bins. Duplicate quantile edges are made monotonically distinct by tiny epsilon increments, and values are clipped to the observed global range before digitization.

ii.
```python
edges = np.quantile(values, np.linspace(0.0, 1.0, nbins + 1))
edges = np.maximum.accumulate(edges + np.arange(edges.size) * eps)
bins = np.searchsorted(edges[1:-1], clipped, side="right")
```

iii. This implements the requested five equal-percentile categories robustly when repeated values produce tied edges.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running timestamps are interpolated directly to the same absolute 30 Hz centers as neural events.

ii.
```python
neural_trial = linear_resample_matrix(ophys_time, events, centers)
running_trial = linear_resample_vector(running_time, running_speed, centers)
```

iii. The notes cite hardware-synchronized timestamps and independent interpolation spot checks.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses `acquisition/EyeTracking/eye_tracking/timestamps` and the `width` and `height` arrays under `pupil_tracking`.

ii.
```python
timestamps = np.asarray(eye_group["eye_tracking"]["timestamps"][:], dtype=np.float64)
width = np.asarray(eye_group["pupil_tracking"]["width"][:], dtype=np.float64)
height = np.asarray(eye_group["pupil_tracking"]["height"][:], dtype=np.float64)
```

iii. The AI explicitly investigated pupil geometry because the target says diameter rather than merely width.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Diameter is computed as twice the larger of pupil width and height. NaNs are filled by within-session temporal interpolation, the result is resampled to trial centers, and global five-quantile edges are computed in pass one.

ii.
```python
diameter = 2.0 * np.maximum(width, height)
diameter = fill_nan_by_time(timestamps, diameter)
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. The notes claim this represents pupil geometry after upstream blink filtering; sessions entirely lacking EyeTracking are excluded.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It uses five global equal-percentile bins with the same tie handling, clipping, and digitization as running speed.

ii.
```python
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. The global edges maintain consistent labels and balanced frequencies across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. The filled pupil series is linearly interpolated from eye timestamps to the identical 30 Hz trial centers used for neural activity.

ii.
```python
neural_trial = linear_resample_matrix(ophys_time, events, centers)
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. The AI cites synchronized absolute timestamps, diagnostic plots, and direct raw-file spot checks.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome is derived from the mutually exclusive boolean trial columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
if bool(trial_row["hit"]): return 0
if bool(trial_row["miss"]): return 1
if bool(trial_row["false_alarm"]): return 2
if bool(trial_row["correct_reject"]): return 3
raise ValueError("Trial has no valid outcome label")
```

iii. These are the canonical Allen task outcomes. The AI deliberately raises on a kept trial without one rather than silently inventing a label.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The four outcomes map to codes 0–3 in fixed order and the selected code is repeated over every time bin of the trial.

ii.
```python
OUTCOME_NAMES = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)
```

iii. Repeating the static category produces a uniform `(n_output, T)` representation while preserving its per-trial meaning.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Invalid trial bounds yield no centers and are skipped. Pupil NaNs are interpolated through time; an all-NaN pupil series raises. Sessions missing EyeTracking are skipped in pass one. Sessions with fewer than two trials or no task presentations are removed. Quantile ties are stabilized, and missing outcome labels raise. The code does not broadly swallow all errors.

ii.
```python
if finite.sum() == 0:
    raise ValueError("All values are NaN")
values[~finite] = np.interp(time_axis[~finite], time_axis[finite], values[finite])
except KeyError as exc:
    print(f"... skip {session.ophys_experiment_id}: {exc}")
```

iii. The AI chose interpolation for modest blink-related missingness and exclusion when a required stream is absent. It retained genuine source all-zero neural trials after independent checks.

## 9-a. What are the most time-consuming steps of the code?

i. Disk I/O and reading full NWB event matrices dominate. The two-pass design reads each retained file twice; neural matrix resampling is the main numerical step.

ii.
```python
with h5py.File(session.filepath, "r") as f:
    ...
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
neural_trials, output_trials, brain_region_idx = convert_session(...)
```

iii. The notes explicitly identify NWB event-matrix I/O and the repeated pass as the main costs.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Metadata construction, pass-one per-trial behavioral resampling, pass-two per-trial conversion, and per-presentation mask assignment remain Python loops. Neural interpolation across cells is already vectorized; variable trial lengths make complete trial-loop vectorization less direct.

ii.
```python
for trial_idx, trial in trials.iterrows():
    ...
    for row in trial_presentations.itertuples(index=False):
        mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
```

iii. The AI specifically notes its vectorized `searchsorted`/broadcasting interpolation avoided a much more expensive per-neuron loop.

## 9-c. What processing does the code repeat multiple times?

i. Each usable NWB is opened in pass one to filter sessions and collect behavioral quantiles/image names, then reopened in pass two to reread trials, presentations, running, pupil, and convert outputs. Trial grids and behavioral interpolation are recomputed in both passes.

ii.
```python
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
...
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(...)
```

iii. The notes acknowledge this cost as the tradeoff for global binning without retaining every session's full raw arrays in memory.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Pass one constructs trial grids and interpolates running and pupil only to aggregate quantile statistics; those arrays are discarded and recomputed. It also gathers presentation/image information before pass two. Optional diagnostic plotting performs extra raw-window extraction and plotting, but only when requested. Metadata fields such as some `SessionMeta` attributes are carried without affecting conversion.

ii.
```python
running_values.append(running_trial)
pupil_values.append(pupil_trial)
...
running_all = np.concatenate(running_values)
pupil_all = np.concatenate(pupil_values)
```

iii. The AI considers the first pass necessary for global bins but recognizes its duplicated I/O/processing. Optional plots are justified as alignment validation and are off by default.
