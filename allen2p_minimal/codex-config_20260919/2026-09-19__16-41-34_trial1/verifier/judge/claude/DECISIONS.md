# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files using `h5py` rather than the AllenSDK high-level API. Experiment metadata is read from a CSV file (`ophys_experiment_table.csv`). NWB files are discovered by globbing `behavior_ophys_experiments/behavior_ophys_experiment_*.nwb`. Only experiments present in the local NWB files (intersected with the metadata table) are retained. Passive sessions are excluded upfront.

ii.
```python
files = {
    _experiment_id(path): path
    for path in sorted(experiment_dir.glob("behavior_ophys_experiment_*.nwb"))
}
table = pd.read_csv(metadata_path)
table = table[table["ophys_experiment_id"].isin(files)].copy()
table = table[~table["passive"].astype(bool)].copy()
```

iii. The agent initially tried using the AllenSDK's `BehaviorOphysExperiment.from_nwb()` API but encountered compatibility errors. After exploring the NWB file structure with h5py, the agent chose direct h5py access for full control and to avoid SDK compatibility issues. The metadata CSV was used to determine session groupings and filtering criteria.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mouse_id` from the experiment metadata table. Unique mouse IDs are collected from all sessions that pass filtering.

ii.
```python
subjects = sorted({str(int(g.iloc[0]["mouse_id"])) for _, g in grouped})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. Mouse IDs from the metadata table uniquely identify each animal in the dataset.

## 1-c. How are the data split into sessions?

i. Sessions are identified by `ophys_session_id`. Experiments sharing the same session ID (multi-plane recordings) are grouped together. Sessions are processed in sorted order of session ID.

ii.
```python
for session_id, experiments in table.groupby("ophys_session_id", sort=True):
    ...
    grouped.append((int(session_id), experiments.copy()))
```

iii. The agent stated: "Experiments sharing an ophys_session_id are simultaneous planes and are concatenated as neurons in one recording session."

## 1-d. How are the data split into trials?

i. Trials are defined by the stimulus presentations table in the NWB file. For each eligible trial ID (from `intervals/trials`), active stimulus presentations carrying that trial's `trials_id` are collected. Each trial becomes a variable-length sequence of 750 ms presentation intervals (typically 10-17 presentations per trial). This differs from the reference approach which uses the full trial window from `start_time` to `stop_time`.

ii.
```python
def _eligible_trial_ids(nwb):
    trials = nwb["intervals/trials"]
    keep = (
        (trials["go"][:] | trials["catch"][:])
        & ~trials["aborted"][:]
        & ~trials["auto_rewarded"][:]
    )
    trial_ids = trials["id"][:][keep].astype(np.int64)
    ...

def _session_presentations(nwb, trial_ids):
    presentations = _presentation_group(nwb)
    source_trial_ids = presentations["trials_id"][:]
    active = presentations["active"][:].astype(bool)
    ...
    for trial_id in trial_ids:
        idx = np.flatnonzero((source_trial_ids == trial_id) & active)
        ...
```

iii. The agent chose this approach because: "The reference paper's actual analysis unit is the 750 ms image-presentation interval (250 ms image plus 500 ms gray), and omissions are treated as their own 750 ms intervals."

## 1-e. How are trials filtered based on quality controls?

i. Go and Catch trials are retained; aborted and auto-rewarded trials are excluded. An assertion checks that every retained trial has exactly one of the four outcomes (hit, miss, false_alarm, correct_reject). Passive sessions are excluded entirely. Sessions without pupil tracking data or with fewer than 2 finite pupil samples are excluded. Sessions with fewer than 2 eligible trials are excluded.

ii.
```python
keep = (
    (trials["go"][:] | trials["catch"][:])
    & ~trials["aborted"][:]
    & ~trials["auto_rewarded"][:]
)
...
if pupil_path not in nwb:
    excluded.append({"reason": "missing pupil tracking required by decoder output"})
    continue
if np.isfinite(nwb[pupil_path][:]).sum() < 2:
    excluded.append({"reason": "insufficient finite pupil diameter samples"})
    continue
if len(trial_ids) < 2:
    excluded.append({"reason": "fewer than two eligible Go/Catch trials"})
    continue
```

iii. The agent followed the task instructions to include Go and Catch trials only and exclude aborted/auto-rewarded. Sessions without pupil data were excluded since pupil diameter is a required decoder output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from **inferred calcium events** (`processing/ophys/event_detection/data` in the NWB files), NOT from dF/F traces. This is a key difference from the reference solution which uses `dff_traces`.

ii.
```python
event_detection = nwb["processing/ophys/event_detection"]
timestamps = event_detection["timestamps"][:]
aggregated = _interval_sums(timestamps, event_detection["data"], starts)
```

iii. The agent justified this based on the paper: "A key modeling choice is emerging from the references: use the SDK-provided inferred calcium events -- not raw dF/F." The paper states: "We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces, thus removing the slow decay dynamics of the calcium indicator GCaMP6f."

## 2-b. How is the `neural` data processed?

i. Event detection magnitudes are **summed** within each 750 ms presentation interval using a prefix-sum approach. Planes from the same session are concatenated along the neuron axis.

ii.
```python
def _interval_sums(timestamps, values, starts):
    first = int(np.searchsorted(timestamps, starts[0], side="left"))
    last = int(np.searchsorted(timestamps, starts[-1] + BIN_SECONDS, side="left"))
    event_data = np.empty((last - first, values.shape[1]), dtype=np.float32)
    values.read_direct(event_data, source_sel=np.s_[first:last, :])
    prefix = np.empty((len(event_data) + 1, event_data.shape[1]), dtype=np.float32)
    prefix[0] = 0.0
    np.cumsum(event_data, axis=0, dtype=np.float32, out=prefix[1:])
    left = np.searchsorted(timestamps, starts, side="left") - first
    right = np.searchsorted(timestamps, starts + BIN_SECONDS, side="left") - first
    return prefix[right] - prefix[left]
```

iii. The metadata records: "AllenSDK inferred calcium-event magnitude, summed per interval." Summing events within the 750 ms window aggregates neural activity to the paper's analysis unit.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional filtering of neurons is applied. All cells present in the event detection data are included.

ii. N/A (no filtering code)

iii. The agent relied on the Allen pipeline's existing quality control. NWB files contain only cells that passed the pipeline's ROI filtering steps.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to each stimulus presentation's `start_time` from the presentations table. Event magnitudes are summed within the 750 ms window starting at each presentation. The ophys timestamps from the event detection data are used as the reference clock.

ii.
```python
reference_ts = nwb["processing/ophys/event_detection/timestamps"][:]
aggregated = _interval_sums(timestamps, event_detection["data"], starts)
```

iii. The agent chose presentation-level alignment based on the paper's analysis unit (750 ms image-presentation intervals).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is **750 ms**, corresponding to the task's image-presentation interval. This involves explicit temporal rebinning from the native ophys frame rate (~11 Hz, ~93 ms) to 750 ms bins. The reference solution keeps native resolution.

ii.
```python
BIN_SECONDS = 0.750
...
"time_bin_size": BIN_SECONDS * 1000.0,  # 750.0 ms
```

iii. The agent justified: "The reference paper's actual analysis unit is the 750 ms image-presentation interval... I'm adopting that native task cadence... This both matches the published processing and avoids inventing a higher-resolution label during the gray portion."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the `image_name` field in the stimulus presentations table within the NWB file. Each presentation interval is labeled with the image shown during that presentation.

ii.
```python
names_source = np.asarray(_decode_strings(presentations["image_name"][:]), dtype=object)
...
image_codes = np.asarray([image_to_code[name] for name in names], dtype=np.int16)
```

iii. The presentations table directly provides the image name for each stimulus presentation, making it the natural source when using presentation-level binning.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are mapped to integer codes via a global mapping. Real images are sorted alphabetically and assigned codes 0-N, with "omitted" appended as the last category. The mapping is consistent across all sessions.

ii.
```python
real_images = sorted(name for name in image_names if name != "omitted")
image_values = real_images + (["omitted"] if "omitted" in image_names else [])
image_to_code = {name: idx for idx, name in enumerate(image_values)}
```

iii. The agent stated: "Omissions are labeled as their own image-identity category and occupy the same 750 ms interval used in the reference analysis." Natural image identifiers are stable strings (imXXX).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is inherently aligned because both neural data and image labels are indexed by the same presentation intervals. Each row in the aggregated data corresponds to one 750 ms presentation.

ii.
```python
session_output.append(np.vstack([
    image_codes[rows],
    changes[rows],
    running_codes[rows],
    pupil_codes[rows],
    np.full(timepoints, outcome, dtype=np.int16),
]).astype(np.int16, copy=False))
```

iii. Since all data streams are aggregated to the same presentation intervals, alignment is guaranteed by construction.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the `is_change` field in the stimulus presentations table.

ii.
```python
changes_source = presentations["is_change"][:].astype(bool)
...
changes = np.concatenate(changes).astype(np.int16)
```

iii. The `is_change` flag is pre-computed in the NWB file and marks presentations where the image identity changed.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The boolean `is_change` field is cast to int16 (0 or 1). No further processing is needed since each presentation interval naturally has a single change/no-change label.

ii.
```python
changes_source = presentations["is_change"][:].astype(bool)
changes = np.concatenate(changes).astype(np.int16)
```

iii. The pre-computed `is_change` field directly provides the binary indicator needed.

## 4-c. How is `output` *Image change* thresholded into categories?

i. The variable is already binary (0 = no change, 1 = change). No thresholding is needed.

ii.
```python
"output_values": [
    ...
    ["no_change", "change"],
    ...
]
```

iii. Binary variable requires no discretization.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same as image identity - aligned through the shared presentation-interval indexing.

ii. See 3-c.

iii. All outputs use the same presentation-level time bins as the neural data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed` in the NWB file (timestamps and data arrays).

ii.
```python
running_ts = nwb["processing/running/speed/timestamps"][:]
running_raw = nwb["processing/running/speed/data"][:]
```

iii. The running speed data from the NWB file is the standard processed running wheel signal.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is (1) linearly interpolated from its native timestamps to ophys timestamps (only finite values used), (2) averaged within each 750 ms presentation interval using prefix sums, then (3) discretized into 5 bins using **within-session** quintile boundaries (20th, 40th, 60th, 80th percentiles).

ii.
```python
running_at_ophys = _interpolate_finite(running_ts, running_raw, reference_ts)
running_values = _interval_means(reference_ts, running_at_ophys, starts)
running_codes, running_edges = _quintile_codes(running_values)

def _quintile_codes(values):
    edges = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
    codes = np.searchsorted(edges, values, side="right").astype(np.int16)
    return codes, edges
```

iii. The agent implemented within-session quintiles to ensure equal bin counts. The per-session approach means bin edges vary across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Within-session 20th, 40th, 60th, and 80th percentile boundaries create 5 bins (0-4). Bins are assigned via `np.searchsorted` with `side="right"`.

ii. See 5-b code.

iii. The task instructions specified "five equal percentile bins." The agent interpreted this as within-session quintiles.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated to ophys timestamps, then averaged per 750 ms presentation interval using the same interval boundaries as the neural data.

ii.
```python
running_at_ophys = _interpolate_finite(running_ts, running_raw, reference_ts)
running_values = _interval_means(reference_ts, running_at_ophys, starts)
```

iii. By first aligning to ophys timestamps and then aggregating to presentation intervals, running speed shares the same time basis as the neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/width` in the NWB file.

ii.
```python
pupil = nwb["acquisition/EyeTracking/pupil_tracking"]
pupil_at_ophys = _interpolate_finite(
    pupil["timestamps"][:], pupil["width"][:], reference_ts
)
```

iii. The agent used pupil width as a proxy for diameter. Sessions missing this field were excluded.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Same approach as running speed: linearly interpolated (over finite/non-blink values) to ophys timestamps, averaged per 750 ms interval, discretized into 5 within-session quintile bins.

ii.
```python
pupil_at_ophys = _interpolate_finite(
    pupil["timestamps"][:], pupil["width"][:], reference_ts
)
pupil_values = _interval_means(reference_ts, pupil_at_ophys, starts)
pupil_codes, pupil_edges = _quintile_codes(pupil_values)
```

iii. Same reasoning as running speed. Blink frames (NaN) are handled by `_interpolate_finite` which filters to finite values before interpolation.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same as running speed: within-session quintile boundaries (20th, 40th, 60th, 80th percentiles) create 5 bins.

ii. See 6-b code.

iii. Same as running speed discretization.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed: interpolated to ophys timestamps, averaged per 750 ms interval.

ii. See 6-b code.

iii. Shared interval boundaries guarantee alignment with neural data.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the NWB trials table.

ii.
```python
OUTCOME_COLUMNS = ("hit", "miss", "false_alarm", "correct_reject")
outcomes = np.column_stack([trials[name][:][keep] for name in OUTCOME_COLUMNS])
if not np.all(outcomes.sum(axis=1) == 1):
    raise RuntimeError("Every retained trial must have exactly one trial outcome")
return trial_ids, outcomes.argmax(axis=1).astype(np.int16)
```

iii. The four columns are mutually exclusive for non-aborted, non-auto-rewarded trials. The assertion verifies exactly one outcome per trial.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The four outcome columns are stacked, and `argmax` converts to integer codes (0=hit, 1=miss, 2=false_alarm, 3=correct_reject). The code is broadcast as a constant across all time bins within the trial.

ii.
```python
outcomes.argmax(axis=1).astype(np.int16)
...
np.full(timepoints, outcome, dtype=np.int16)
```

iii. Trial outcome is a per-trial static variable, so it is constant across all presentation intervals within a trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Missing pupil tracking**: Sessions are excluded entirely with reason logged.
- **Insufficient pupil data**: Sessions with fewer than 2 finite pupil samples excluded.
- **Blink frames**: NaN values in pupil data are filtered before interpolation via `_interpolate_finite`, which uses only finite samples.
- **Running speed NaN/Inf**: Similarly handled by `_interpolate_finite`.
- **Zero neural activity**: Retained (1,677 trials with no detected events were noted but kept).
- **Sessions with <2 trials**: Excluded during initial filtering.

ii.
```python
def _interpolate_finite(source_timestamps, source_values, target_timestamps):
    finite = np.isfinite(source_timestamps) & np.isfinite(source_values)
    if finite.sum() < 2:
        raise RuntimeError("Behavioral stream has fewer than two finite samples")
    return np.interp(target_timestamps, source_timestamps[finite], source_values[finite])
```

iii. The agent explicitly excluded sessions that couldn't provide all required decoder outputs, rather than filling in missing values.

## 9-a. What are the most time-consuming steps of the code?

i. Loading NWB files via h5py is the most time-consuming step. Each experiment file contains full-session neural, running, and eye tracking data. The code opens NWB files multiple times (once for session filtering, once or more for data extraction).

ii. N/A

iii. I/O-bound data loading dominates runtime.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial assembly loop (iterating over `row_groups` and `trial_outcomes` with `zip`) could potentially be vectorized using advanced indexing, though each trial has a different number of time points making this nontrivial.

ii.
```python
for rows, outcome in zip(row_groups, trial_outcomes):
    timepoints = len(rows)
    session_neural.append(np.ascontiguousarray(activity[rows].T, dtype=np.float32))
    ...
```

iii. The loop is not the bottleneck; NWB I/O is.

## 9-c. What processing does the code repeat multiple times?

i. The representative NWB file for each session is opened at least twice: once during the filtering phase (to check for pupil data, count eligible trials, and collect image names) and once during the main data extraction phase. Trial IDs are computed in both phases.

ii.
```python
# First pass (filtering):
with h5py.File(representative, "r") as nwb:
    trial_ids, _ = _eligible_trial_ids(nwb)
    _, _, names, _ = _session_presentations(nwb, trial_ids)

# Second pass (extraction):
with h5py.File(representative, "r") as nwb:
    trial_ids, trial_outcomes = _eligible_trial_ids(nwb)
    starts, row_groups, names, changes = _session_presentations(nwb, trial_ids)
```

iii. The two-pass approach was used to first filter sessions and build global mappings before processing data.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code collects and stores extensive metadata (session_info with per-session bin edges, excluded session lists, etc.) that is not used by the decoder. The running and pupil quintile edges are stored per-session in metadata but not needed downstream.

ii.
```python
session_info.append({
    ...
    "running_quintile_edges_cm_per_s": running_edges.tolist(),
    "pupil_diameter_quintile_edges_pixels": pupil_edges.tolist(),
})
```

iii. This metadata is useful for debugging and interpretability but is not consumed by the decoder training code.
