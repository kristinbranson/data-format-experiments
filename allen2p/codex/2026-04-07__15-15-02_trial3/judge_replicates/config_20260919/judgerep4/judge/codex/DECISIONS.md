# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local experiment metadata CSV, inventories locally present `behavior_ophys_experiment_*.nwb` files, retains metadata rows with a local file, removes passive experiments, and reads the NWBs directly with `h5py`. It makes two passes over retained files: one for global behavioral quantiles/session eligibility and one for conversion.

ii.
```python
exp_table = pd.read_csv(table_path)
available_files = {
    int(path.stem.split("_")[-1]): path
    for path in sorted(experiment_dir.glob("behavior_ophys_experiment_*.nwb"))
}
exp_table = exp_table[exp_table["ophys_experiment_id"].isin(available_files)].copy()
exp_table = exp_table[~exp_table["passive"]].copy()
...
with h5py.File(session.filepath, "r") as f:
```

iii. The notes say direct `h5py` access was chosen because the local AllenSDK/NWB stack could not instantiate these files. The agent intended to process the complete locally available active subset while avoiding network fetches.

## 1-b. How are the data split into subjects?

i. Subjects are sorted unique `mouse_id` strings among sessions retained after the first pass; each converted experiment receives the corresponding subject index.

ii.
```python
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx[idx - 1] = subject_to_idx[session.mouse_id]
```

iii. The agent treats the experiment-table mouse identifier as the canonical animal identifier and reports that its 38 converted subjects exactly match the local active, eye-tracking subset.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id` NWB (one imaging plane) is treated as an independent output session. The code records but does not group files by `ophys_session_id` or `behavior_session_id`.

ii.
```python
for row in exp_table.itertuples(index=False):
    sessions.append(SessionMeta(
        ophys_experiment_id=int(row.ophys_experiment_id),
        ophys_session_id=int(row.ophys_session_id),
        ...
        filepath=available_files[int(row.ophys_experiment_id)],
    ))
```

iii. The notes explicitly justify this as AllenSDK “experiment granularity,” because each plane has its own neuron set, even when several experiments share an ophys or behavior session.

## 1-d. How are the data split into trials?

i. Trials come from `intervals/trials`. For every retained trial, the agent constructs bin centers spanning `start_time` through (but not beyond) `stop_time`, producing variable-length trials.

ii.
```python
trials = read_interval_table(f["intervals"]["trials"], columns)
...
n_bins = max(1, int(np.ceil((stop_time - start_time) / DT)))
centers = start_time + (np.arange(n_bins, dtype=np.float64) + 0.5) * DT
valid = centers < (stop_time + 1e-9)
```

iii. The agent chose full behavioral trial windows so pre-change flashes and the response period are both represented, while bin centers avoid including a sample beyond the trial end.

## 1-e. How are trials filtered based on quality controls?

i. Only go or catch trials are kept; aborted and auto-rewarded trials are removed. Invalid/empty time windows are skipped, sessions with fewer than two kept trials are excluded, passive experiments are excluded, and sessions lacking required eye tracking or task presentations are skipped. It does not explicitly require non-null `change_time`.

ii.
```python
trials = trials[(trials["go"] | trials["catch"]) &
                (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
...
if centers.size == 0:
    continue
...
if len(neural_trials) < 2:
    raise RuntimeError(...)
```

iii. The notes call this the contingent-trial filter and report exact agreement with their raw-NWB recount of 51,075 local trials. Eye tracking is required because pupil diameter is a mandatory decoder output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the NWB processed `event_detection` timestamps and event-magnitude matrix, rather than dF/F traces.

ii.
```python
event_group = f["processing"]["ophys"]["event_detection"]
timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
events = np.asarray(event_group["data"][:], dtype=np.float32)
```

iii. The notes describe these as processed ophys event magnitudes and regard all-zero trial warnings as genuine sparse source detections, not conversion failures.

## 2-b. How is the `neural` data processed?

i. The time-by-cell event matrix is linearly interpolated onto each trial’s 30 Hz centers and transposed to neuron-by-time. Planes are not combined.

ii.
```python
idx_hi = np.searchsorted(src_time, dst_time, side="left")
...
interp = src_value[idx_lo] * (1.0 - w[:, None]) + src_value[idx_hi] * w[:, None]
return interp.T.astype(np.float32, copy=False)
```

iii. The agent chose a common 30 Hz grid to give neural, running, pupil, and stimulus outputs identical time axes, and vectorized interpolation to avoid a per-neuron loop.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit cell-level mask is applied. Every cell represented in `event_detection/data` and the `cell_specimen_table` is retained; all-zero trials are retained.

ii.
```python
n_cells = len(cell_table["cell_specimen_id"])
return np.full(n_cells, region_index, dtype=np.int64)
```

iii. The notes state that all cells listed in the included NWBs were already valid ROIs and that raw spot checks showed all-zero event trials truly existed in the source.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial time zero is `trial.start_time`. Absolute 30 Hz bin centers between trial start and stop are used to interpolate the event data; metadata names the alignment event “trial start.”

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
...
"temporal_alignment_event": "trial start",
```

iii. The agent says absolute synchronized timestamps preserve cross-stream alignment and its independently reconstructed spot checks matched converted arrays.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output uses fixed 1/30-second bins (33.333 ms). Neural events are linearly resampled from their native ophys timestamps to that grid; this is interpolation/resampling, not aggregation into native bins.

ii.
```python
DT = 1.0 / 30.0
TIME_BIN_MS = DT * 1000.0
...
"time_bin_size": TIME_BIN_MS,
"sampling_grid_hz": 30.0,
```

iii. The stated rationale is a shared resolution across mixed native ophys rates and approximately 30 Hz behavior/eye streams.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from change-detection stimulus-presentation interval tables: `image_name`, `omitted`, `start_time`, `stop_time`, and `trials_id`.

ii.
```python
columns = ["start_time", "stop_time", "image_name", "omitted",
           "is_change", "trials_id", ...]
...
trial_presentations = presentations[presentations["trials_id"] == int(trial["id"])].copy()
```

iii. The agent chose presentation-level records to reconstruct the actual flash/gray sequence, including omissions, rather than infer only a pre/post identity from trial metadata.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global sorted category map is built, with `gray` forced first. Each trial begins as gray; bins inside a non-omitted presentation receive its image code, while omitted presentations remain gray.

ii.
```python
image_identity = np.full(centers.shape[0], image_value_to_idx["gray"], dtype=np.int64)
...
if bool(row.omitted) or str(row.image_name) == "omitted":
    image_identity[mask] = image_value_to_idx["gray"]
else:
    image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. The notes describe this as faithfully representing 250 ms image flashes alternating with 500 ms gray periods, with 17 global categories (gray plus 16 images).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Presentation intervals are evaluated on the exact same 30 Hz `centers` used for neural interpolation.

ii.
```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. The agent’s rationale is that common absolute timestamps and a common destination grid guarantee binwise alignment.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change comes from the stimulus-presentation `is_change` flag and each presentation’s start/stop interval.

ii.
```python
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The agent used the processed presentation table as the direct source of actual change flashes.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector is initialized; bins falling within a presentation marked `is_change` are set to one.

ii.
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
...
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The notes say the resulting sparse positives correspond to the change flash and report a roughly 2.5% positive fraction.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numeric threshold is estimated: the raw boolean `is_change` flag maps directly to categorical 0/1 (`no_change`, `change`).

ii.
```python
"output_values": [image_values, ["no_change", "change"], ...]
```

iii. Direct boolean coding avoids an arbitrary threshold and follows the processed stimulus annotation.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The change-presentation interval is masked on the same trial centers used for neural interpolation.

ii.
```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
if bool(row.is_change):
    image_change[mask] = 1
```

iii. The agent says diagnostic plots and raw reconstructions showed the indicator aligned to the change flash.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is read from NWB `processing/running/speed`, using its `timestamps` and `data` arrays.

ii.
```python
running_group = f["processing"]["running"]["speed"]
timestamps = np.asarray(running_group["timestamps"][:], dtype=np.float64)
speed = np.asarray(running_group["data"][:], dtype=np.float64)
```

iii. The agent treats this processed NWB stream as the canonical wheel-speed output.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Speed is linearly interpolated to trial centers. A first pass pools all included trial samples, computes five global quantile bins, and the second pass digitizes clipped values.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
...
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
...
running_bin = digitize_with_edges(running_trial, running_edges)
```

iii. Global quantiles provide consistent labels across sessions and approximately balanced decoder classes.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five dataset-wide percentile bins are formed at 0, 20, 40, 60, 80, and 100%; labels are `q1`–`q5`. Repeated edges are nudged monotonically and values are clipped to the observed range.

ii.
```python
probs = np.linspace(0.0, 1.0, nbins + 1)
edges = np.quantile(values, probs)
...
bins = np.searchsorted(edges[1:-1], clipped, side="right")
```

iii. This directly implements the requested five equal-percentile categories and handles tied quantiles defensively.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running timestamps are interpolated directly onto the same absolute trial centers as neural events.

ii.
```python
neural_trial = linear_resample_matrix(ophys_time, events, centers)
running_trial = linear_resample_vector(running_time, running_speed, centers)
```

iii. The notes rely on hardware-synchronized timestamps and report exact independent reconstruction checks.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil data uses eye-tracking timestamps plus pupil-tracking `width` and `height`; “diameter” is defined as twice the larger dimension.

ii.
```python
width = np.asarray(eye_group["pupil_tracking"]["width"][:], dtype=np.float64)
height = np.asarray(eye_group["pupil_tracking"]["height"][:], dtype=np.float64)
diameter = 2.0 * np.maximum(width, height)
```

iii. The agent interpreted width and height as radii/axes and used the larger full axis as pupil diameter.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Missing diameter samples are linearly filled over eye time, the result is interpolated to trial centers, global quintile edges are computed in pass one, and values are digitized in pass two. Blink flags are not used.

ii.
```python
diameter = fill_nan_by_time(timestamps, diameter)
...
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. The notes emphasize smooth common-grid alignment and global class balance; sessions with no eye-tracking acquisition are excluded.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. It uses five global percentile bins, labeled `q1`–`q5`, with the same quantile/digitization procedure as running speed.

ii.
```python
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
...
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. This implements equal-percentile discretization across all retained trial time points.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Filled eye measurements are linearly interpolated to the common 30 Hz trial centers used for neural activity.

ii.
```python
neural_trial = linear_resample_matrix(ophys_time, events, centers)
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. The agent cites synchronized absolute timestamps and successful raw-data spot checks.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome is derived from the trial table’s mutually exclusive `hit`, `miss`, `false_alarm`, and `correct_reject` boolean columns.

ii.
```python
if bool(trial_row["hit"]): return 0
if bool(trial_row["miss"]): return 1
if bool(trial_row["false_alarm"]): return 2
if bool(trial_row["correct_reject"]): return 3
raise ValueError("Trial has no valid outcome label")
```

iii. The agent uses the dataset’s canonical outcome flags and intentionally fails if a retained trial has no valid label.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The four flags map to codes 0–3 in the stated order, and the selected code is repeated at every time bin of the trial.

ii.
```python
outcome_idx = trial_outcome_index(trial)
outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)
```

iii. Repetition makes the static per-trial target compatible with the common time-varying output matrix; the notes report exact agreement with raw outcome distributions.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing pupil samples are interpolated in time; a wholly missing pupil trace raises an error. Sessions missing EyeTracking or required NWB groups are skipped during pass one via `KeyError`. Empty trial windows are skipped; sessions with fewer than two trials or no task presentations are removed. Quantile calculations ignore non-finite pooled values. All-zero neural trials are retained. Output directories are created automatically.

ii.
```python
if finite.sum() == 0:
    raise ValueError("All values are NaN")
values[~finite] = np.interp(...)
...
except KeyError as exc:
    print(f"... skip ...: {exc}")
```

iii. The notes justify retaining source-valid zero-event trials, while excluding sessions that cannot supply the required pupil output. They report raw reconstructions to distinguish source anomalies from conversion bugs.

## 9-a. What are the most time-consuming steps of the code?

i. NWB I/O—especially reading the full event matrix—and the second-pass session conversion dominate. The full two-pass run reads each retained file twice.

ii.
```python
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
...
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(...)
```

iii. The notes estimate about 0.36 s/session for pass one and 1.86 s/session for pass two, explicitly identifying event-matrix disk reads as I/O-heavy.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Remaining candidates are the per-session, per-trial, per-presentation, byte-string decoding, and first-pass trial interpolation loops. Neural interpolation itself is already vectorized across cells.

ii.
```python
for trial_idx, trial in trials.iterrows():
    ...
    for row in trial_presentations.itertuples(index=False):
        mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
```

iii. The notes highlight the implemented vectorized `searchsorted`/broadcast neural interpolation as the major speedup; variable trial lengths make full trial-loop vectorization less straightforward.

## 9-c. What processing does the code repeat multiple times?

i. Each NWB is opened in both passes; trials, presentations, running, and pupil are read/reconstructed twice, and behavioral streams are interpolated once for quantile collection and again for final output.

ii.
```python
# pass 1
running_trial = linear_resample_vector(...)
pupil_trial = linear_resample_vector(...)
...
# pass 2
running_trial = linear_resample_vector(...)
pupil_trial = linear_resample_vector(...)
```

iii. The agent accepts this repetition to stream data without retaining the entire dataset in memory while still computing global percentile edges.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Pass-one trialwise interpolated running/pupil arrays are discarded after global edges are calculated and recomputed later. Trial-table initial/change image names are loaded but unused. Optional diagnostic plots compute padded raw windows and summaries not stored in the dataset. Metadata such as experience level and session type is loaded but not used in conversion decisions.

ii.
```python
running_values.append(running_trial)
pupil_values.append(pupil_trial)
...
return running_edges, pupil_edges, kept_sessions, image_names
```

iii. The discarded pass-one arrays are a deliberate memory/performance tradeoff; optional plotting exists only for validation and is disabled unless requested.
