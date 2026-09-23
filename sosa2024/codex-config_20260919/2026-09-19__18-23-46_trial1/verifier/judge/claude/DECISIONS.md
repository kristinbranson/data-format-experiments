# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all NWB files via `glob.glob("sub-m*/*.nwb")` under the data root `/app/data`. Files are sorted by numeric subject and session identifiers parsed from filenames. Each NWB file is opened with `h5py` (not `pynwb`) and behavioral time series are read from `processing/behavior/BehavioralTimeSeries`, neural data from `processing/ophys/Deconvolved`. All discovered files are processed.

ii.
```python
def discover_files(sample: bool) -> list[Path]:
    files = [Path(p) for p in glob.glob(str(DATA_ROOT / "sub-m*" / "*.nwb"))]
    files.sort(key=natural_key)
    ...
    return files

def read_series(nwb: h5py.File, name: str) -> tuple[np.ndarray, np.ndarray]:
    group = nwb[f"{BEHAVIOR}/{name}"]
    return group["data"][()], group["timestamps"][()]
```

iii. The AI verifies that all 152 NWB files (11 subjects, up to 14 sessions each, m11 starting at session 03) are discovered. The number matches the paper's description.

## 1-b. How are the data split into subjects?

i. Subjects are parsed from NWB filenames using the regex `sub-m(\d+)` and de-duplicated into a sorted list.

ii.
```python
subjects = sorted({f"m{natural_key(p)[0]}" for p in files}, key=lambda x: int(x[1:]))
subject_lookup = {subject: i for i, subject in enumerate(subjects)}
```

iii. This yields 11 subjects matching the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Session numbers are parsed from filenames via `ses-(\d+)`.

ii.
```python
def natural_key(path: str | Path) -> tuple[int, int]:
    match = re.search(r"sub-m(\d+)_ses-(\d+)", str(path))
    return int(match.group(1)), int(match.group(2))
```

iii. The one-NWB-per-session structure is consistent with the data organization.

## 1-d. How are the data split into trials?

i. Trial starts are identified as frames where `trial_start > 0`. Trial ends are identified as frames where `teleport > 0`. The AI uses `np.flatnonzero(teleport > 0)` to find all frames where teleport is positive, rather than detecting rising edges of the teleport signal.

ii.
```python
starts = np.flatnonzero(trial_start > 0).astype(np.int64)
stops = np.flatnonzero(teleport > 0).astype(np.int64)
if starts.size != stops.size or np.any(stops <= starts):
    raise ValueError(f"{session_id}: invalid trial start/teleport pairs")
```

iii. The AI validates that starts and stops are paired and ordered. This works because teleport appears to be a single-frame pulse in the data. Trial data is sliced as `[start:stop)`, excluding the teleport frame.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered using the paper's lick-sensor artifact criterion: trials where >30% of on-track frames have a cumulative lick count > 2 are removed. This removes exactly 81 trials across all 11 mice, matching the paper's reported number.

ii.
```python
lick_artifact_fraction = np.array(
    [np.mean(lick[s:e] > 2) for s, e in zip(starts, stops)], dtype=np.float64
)
keep = lick_artifact_fraction <= 0.30
```

iii. The AI cites the paper's Methods section which describes this criterion and the 81-trial count. The AI does not apply a short-trial filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses the `Deconvolved` data stored in the NWB file under `processing/ophys/Deconvolved/plane*/data`. The AI believes this is the author-computed deconvolved events from the paper's pipeline.

ii.
```python
def load_curated_events(nwb: h5py.File, n_behavior_frames: int) -> ...:
    plane_groups = nwb[f"{OPHYS}/Deconvolved"]
    ...
    for plane_name in plane_names:
        dataset = plane_groups[f"{plane_name}/data"]
        dense = dataset[:n_behavior_frames, :]
        selected = np.asarray(dense[:, local_mask], dtype=np.float32)
        ...
```

iii. The AI's CONVERSION_NOTES state: "Native data are already author-processed ... so dF/F does not need to be recomputed." However, the reference solution identifies that the NWB `Deconvolved` field is suite2p's own deconvolution of raw fluorescence, NOT the paper's custom dF/F + OASIS deconvolution pipeline. The paper computes its own events from raw `Fluorescence` and `Neuropil` traces using neuropil subtraction, maximin baseline, Gaussian smoothing, and OASIS deconvolution with specific parameters.

## 2-b. How is the `neural` data processed?

i. The AI applies no neural processing beyond reading the stored Deconvolved values and subsetting to curated cells. There is no dF/F computation, no neuropil subtraction, no baseline estimation, no smoothing, and no OASIS deconvolution.

ii.
```python
dense = dataset[:n_behavior_frames, :]
selected = np.asarray(dense[:, local_mask], dtype=np.float32)
pooled[:, cursor : cursor + n_selected] = selected
```

iii. The AI states that the data is "already author-processed" and doesn't need recomputation. The reference solution, by contrast, copies the paper's entire `dff()` function and applies the full processing pipeline: neuropil subtraction (0.7 coefficient), per-trial maximin baseline with 300-sample window, dF/F computation, 2-sample Gaussian smoothing, and OASIS deconvolution (tau=0.7).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only the manual Suite2p `iscell` curation is applied. Cells with `iscell[:,0] > 0.5` are kept; others are discarded. Planes are concatenated. The AI explicitly does NOT apply the paper's putative interneuron filter (speed correlation > 0.5).

ii.
```python
segmentation = nwb[f"{OPHYS}/ImageSegmentation/PlaneSegmentation"]
iscell = segmentation["iscell"][:, 0] > 0.5
...
selected = np.asarray(dense[:, local_mask], dtype=np.float32)
```

iii. The AI argues that "excluding neurons based on correlation with the speed output would directly select against a requested prediction" and that the filter is "analysis-specific." The reference applies the interneuron filter because it is described in the paper's Methods.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start by slicing `[start:stop)` using the same indices as behavioral data. No additional temporal shifting is needed since neural and behavioral data share the same time axis.

ii.
```python
neural = np.ascontiguousarray(neural_by_time[s:e, :].T, dtype=np.float32)
```

iii. Timestamps are verified to be uniformly spaced and consistent across all behavioral streams.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native temporal resolution of ~64.48 ms (~15.5 Hz) is preserved. No rebinning is applied.

ii.
```python
EXPECTED_DT_S = 1.0 / 15.5078125
...
"time_bin_size": EXPECTED_DT_S * 1000.0,
```

iii. The AI validates that all timestamps have the expected uniform spacing.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `timestamps` array of the behavioral time series (all behavior streams share the same timestamps).

ii.
```python
position, timestamps = read_series(nwb, "position")
...
np.asarray(timestamps[s:e] - timestamps[s], dtype=np.float32),
```

iii. Timestamps are verified to be uniform and aligned across all behavioral streams.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp of the trial start frame is subtracted from all frame timestamps within the trial.

ii.
```python
np.asarray(timestamps[s:e] - timestamps[s], dtype=np.float32),
```

iii. Straightforward subtraction.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data are already aligned on the same time axis (same frame indices), so no additional alignment is needed.

ii. Both use the same `[s:e)` slice indices.

iii. Verified by timestamp consistency checks.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavioral time series.

ii.
```python
environment, env_time = read_series(nwb, "environment")
...
np.full(n_time, trial_environments[trial], dtype=np.float32),
```

iii. Environment is validated to have exactly one unique valid value (0 or 1) per trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The unique valid environment value within each trial is extracted and replicated across all timepoints.

ii.
```python
for s, e in zip(starts, stops):
    valid = np.unique(environment[s:e][environment[s:e] >= 0])
    if valid.size != 1 or valid[0] not in (0, 1):
        raise ValueError(...)
    trial_environments.append(int(valid[0]))
```

iii. The validation ensures environment is constant and binary within each trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the within-session trial index (loop counter), derived from the enumeration of trials.

ii.
```python
for trial, (s, e) in enumerate(zip(starts, stops)):
    ...
    np.full(n_time, trial, dtype=np.float32),
```

iii. This is the zero-based sequential index of trials within the session. Note that the AI uses the original trial index even after filtering (skipped trials retain their original index).

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index is used directly as the trial number, replicated across all timepoints. No transformation is applied.

ii.
```python
np.full(n_time, trial, dtype=np.float32),
```

iii. The trial number preserves the original within-session ordering.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the sparse `Reward/timestamps` in the NWB behavioral time series. Reward event timestamps are mapped to the nearest behavioral frame using `nearest_timestamp_indices`.

ii.
```python
reward_times = nwb[f"{BEHAVIOR}/Reward/timestamps"][()]
reward_indices = nearest_timestamp_indices(timestamps, reward_times)
outcomes = np.array(
    [np.any((reward_indices >= s) & (reward_indices < e)) for s, e in zip(starts, stops)],
    dtype=np.int64,
)
```

iii. Reward delivery events are aligned with a tolerance of half a time bin.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the previous trial's outcome (from the `outcomes` array) is used. For the first trial, the value is set to 0. The value is constant across all timepoints within a trial.

ii.
```python
previous_outcome = int(outcomes[trial - 1]) if trial > 0 else 0
...
np.full(n_time, previous_outcome, dtype=np.float32),
```

iii. The AI correctly uses the preceding trial's outcome (before any trial filtering), consistent with the temporal ordering of the experiment.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavioral time series and the inferred reward zone label for each trial. The zone label determines which zone boundaries (A: 80-130, B: 200-250, C: 320-370 cm) to use.

ii.
```python
ZONE_BOUNDS = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}
...
label = zone_labels[trial]
z0, z1 = ZONE_BOUNDS[label]
trial_position = np.asarray(position[s:e], dtype=np.float32)
distance = distance_to_interval(trial_position, z0, z1)
```

iii. Zone bounds match the paper's defined reward zones.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed from position to the nearest edge of the reward zone interval. Distance is negative before the zone, zero inside, and positive after.

ii.
```python
def distance_to_interval(position, start, end):
    distance = np.zeros(position.shape, dtype=np.float32)
    before = position < start
    after = position > end
    distance[before] = position[before] - start
    distance[after] = position[after] - end
    return distance
```

iii. This follows the instruction's definition of signed distance to the reward zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous distance is discretized into 7 bins using manual comparison operators. The boundaries are at -50, -10, 0, 10, and 50 cm. The default value is 3 (distance == 0).

ii.
```python
def discretize_distance(distance):
    out = np.full(distance.shape, 3, dtype=np.int64)
    out[distance < -50] = 0
    out[(distance >= -50) & (distance <= -10)] = 1
    out[(distance > -10) & (distance < 0)] = 2
    out[(distance > 0) & (distance <= 10)] = 4
    out[(distance > 10) & (distance <= 50)] = 5
    out[distance > 50] = 6
    return out
```

iii. The boundary handling differs slightly from the reference, which uses `np.digitize`. At exact boundary values (e.g., -10, 10, 50), the AI assigns to the bin including the boundary, while the reference assigns to the adjacent bin (e.g., -10 goes to bin 1 in AI, bin 2 in reference).

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indices as neural data within each trial. No additional alignment needed.

ii.
```python
trial_position = np.asarray(position[s:e], dtype=np.float32)
distance = distance_to_interval(trial_position, z0, z1)
```

iii. Both neural and behavioral data use the same `[s:e)` indexing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series.

ii.
```python
position, timestamps = read_series(nwb, "position")
...
trial_position = np.asarray(position[s:e], dtype=np.float32)
```

iii. Position directly records the animal's location in the virtual corridor in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice and discretizing.

ii.
```python
discretize_position(trial_position),
```

iii. Raw position values are used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 equal-sized 90 cm bins spanning the 450 cm track using manual comparison operators.

ii.
```python
def discretize_position(position):
    out = np.zeros(position.shape, dtype=np.int64)
    out[(position >= 90) & (position < 180)] = 1
    out[(position >= 180) & (position < 270)] = 2
    out[(position >= 270) & (position <= 360)] = 3
    out[position > 360] = 4
    return out
```

iii. The boundary handling at 360 cm differs from the reference: AI assigns 360 to bin 3, reference assigns to bin 4.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices as neural data. No additional alignment needed.

ii. Same `[s:e)` indexing.

iii. Verified by timestamp checks.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series.

ii.
```python
lick, lick_time = read_series(nwb, "lick")
...
(lick[s:e] > 0).astype(np.int64),
```

iii. The lick variable records cumulative within-frame lick counts.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0. Trials with lick sensor artifacts are entirely removed before this step.

ii.
```python
(lick[s:e] > 0).astype(np.int64),
```

iii. The instructions specify binary output. Cumulative counts can exceed 1, so thresholding at > 0 converts to binary.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices as neural data. No additional alignment needed.

ii. Same `[s:e)` indexing.

iii. Verified by timestamp checks.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from `position` and `reward_zone` behavioral time series. Position values where `reward_zone > 0` reveal the zone identity. The AI uses a trial-30 switch structure (from the paper) combined with median position within the zone-entry period to infer zone labels.

ii.
```python
def classify_observed_zone(values):
    median_position = float(np.median(values))
    centers = {label: np.mean(bounds) for label, bounds in ZONE_BOUNDS.items()}
    return min(centers, key=lambda label: abs(median_position - centers[label]))

def infer_zone_labels(position, reward_zone, starts, stops):
    observed = [
        classify_observed_zone(position[s:e][reward_zone[s:e] > 0])
        for s, e in zip(starts, stops)
    ]
    labels = [None] * len(starts)
    for lo, hi in ((0, min(30, len(starts))), (min(30, len(starts)), len(starts))):
        segment_observed = [observed[i] for i in range(lo, hi) if observed[i] is not None]
        label, _ = Counter(segment_observed).most_common(1)[0]
        labels[lo:hi] = [label] * (hi - lo)
    ...
```

iii. The AI uses the paper's trial-30 switch structure to segment trials into two blocks, then assigns the most common observed zone label to each block. The reference uses a Viterbi algorithm instead.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Trials are split into two segments (0-29 and 30+), following the paper's trial-30 switch schedule. Within each segment, the most common observed zone label (from reward_zone > 0 positions) is assigned to all trials. Zone labels A/B/C are mapped to integers 0/1/2.

ii.
```python
np.full(n_time, ZONE_TO_INT[label], dtype=np.int64),
```

iii. The AI validates that no observed zone contradicts the inferred schedule.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the sparse `Reward/timestamps` in the NWB behavioral data.

ii.
```python
reward_times = nwb[f"{BEHAVIOR}/Reward/timestamps"][()]
reward_indices = nearest_timestamp_indices(timestamps, reward_times)
outcomes = np.array(
    [np.any((reward_indices >= s) & (reward_indices < e)) for s, e in zip(starts, stops)],
    dtype=np.int64,
)
```

iii. Reward timestamps are aligned to nearest behavioral frames with tolerance checking.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the output is 1 if any reward delivery event falls within the trial's time range, 0 otherwise. The value is constant across all timepoints in the trial.

ii.
```python
np.full(n_time, outcomes[trial], dtype=np.int64),
```

iii. Binary per-trial output matching the instructions.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Behavioral timestamp alignment**: All behavioral streams are validated to have identical timestamps. Non-uniform or unexpected intervals raise an error.
- **Neural/behavior length mismatch**: Neural data is truncated to match behavior frame count if longer.
- **Lick artifact trials**: Removed entirely using the paper's >30% criterion.
- **Missing reward zone observations**: On unrewarded trials where `reward_zone` is never active, `classify_observed_zone` returns None, and the zone label is inferred from neighboring trials via the block majority-vote approach.
- **Reward timestamp alignment**: Validated to be within half a time bin of the nearest behavioral frame.
- **Sessions with < 2 valid trials**: Raise an error (though none occur in practice).

ii.
```python
if dataset.shape[0] < n_behavior_frames:
    raise ValueError(f"{plane_name}: neural stream is shorter than behavior")
...
if not np.allclose(timestamps, other) for other in aligned_times):
    raise ValueError(f"{session_id}: behavior timestamps are not aligned")
```

iii. The AI uses strict validation (raising errors) rather than silent corrections for most cases.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Reading dense neural data from HDF5** (I/O bound, especially for large dual-plane sessions)
2. **Full dataset conversion loop** across all 152 sessions
3. **Pickle serialization** of the ~9 GiB output

ii. N/A

iii. The AI reports total conversion time of ~56 s plus ~8 s for writing.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates over each trial sequentially. Some operations (discretization, distance computation) could theoretically be applied to full session arrays before splitting, though variable trial lengths make this awkward. The `infer_zone_labels` function's list comprehension iterates over trials for zone classification.

ii. N/A

iii. The per-trial loop is the natural structure given variable-length trials.

## 13-c. What processing does the code repeat multiple times?

i. The code reads each NWB file only once per session, which is efficient. However, the `nearest_timestamp_indices` function computes distances redundantly for left and right candidates. Zone observation is computed for all trials even though only the modal label per segment is used.

ii. N/A

iii. The AI's design avoids the double-loading pattern seen in the reference's survey + conversion steps.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores extensive per-session metadata and statistics dictionaries that are not strictly necessary for the decoder. The `plot_processing` function (when enabled) computes and renders visualizations. The `lick_artifact_fraction` is computed for all trials but only used for filtering.

ii. N/A

iii. Metadata supports documentation and debugging but is not used by the decoder itself.
