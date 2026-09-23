# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively discovers every `sub-m*/*.nwb` file, numerically sorts it by mouse and session, and opens each file directly with `h5py`. Full mode includes all 152 discovered sessions; sample mode deliberately selects two.

ii.
```python
files = [Path(p) for p in glob.glob(str(DATA_ROOT / "sub-m*" / "*.nwb"))]
files.sort(key=natural_key)
...
with h5py.File(path, "r") as nwb:
```

iii. The notes justify direct NWB loading from the supplied DANDI export, report 11 mice and 152 files, and say this avoids loading data not needed for conversion.

## 1-b. How are the data split into subjects?

i. Subject IDs are parsed from filenames with a regular expression, made unique, naturally sorted, and mapped to each session through `subject_idx`.

ii.
```python
match = re.search(r"sub-m(\d+)_ses-(\d+)", str(path))
...
subjects = sorted({f"m{natural_key(p)[0]}" for p in files}, key=lambda x: int(x[1:]))
subject_lookup = {subject: i for i, subject in enumerate(subjects)}
subject_idx.append(subject_lookup[info["subject"]])
```

iii. The notes report that the resulting 11 IDs agree with the supplied cohort and NWB metadata.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session and produces one entry in each session-level list.

ii.
```python
for index, path in enumerate(files):
    session_neural, session_input, session_output, info, stats = convert_session(path, ...)
    neural.append(session_neural)
```

iii. The filename/session convention and reference repository both support one NWB per imaging session.

## 1-d. How are the data split into trials?

i. Trial starts are nonzero `trial_start` frames and ends are nonzero `teleport` frames. Each trial is sliced as zero-based `[start, teleport)`, excluding teleport/ITI.

ii.
```python
starts = np.flatnonzero(trial_start > 0).astype(np.int64)
stops = np.flatnonzero(teleport > 0).astype(np.int64)
...
neural = np.ascontiguousarray(neural_by_time[s:e, :].T, dtype=np.float32)
```

iii. The agent checked that starts and teleports pair and that these boundaries agree with trial-number transitions. It explains that the old pickle code's `-1` correction does not apply to native zero-based NWB indices.

## 1-e. How are trials filtered based on quality controls?

i. A whole trial is removed when more than 30% of its on-track frames have raw cumulative lick count greater than 2. Rewarded, omitted, slow, and variable-length trials otherwise remain.

ii.
```python
lick_artifact_fraction = np.array(
    [np.mean(lick[s:e] > 2) for s, e in zip(starts, stops)], dtype=np.float64
)
keep = lick_artifact_fraction <= 0.30
```

iii. The notes identify this as the paper's lick-sensor artifact criterion and report exactly 81 removed trials. The agent chose whole-trial rejection because lick is a required output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `processing/ophys/Deconvolved/plane*/data`, with ROI selection derived from `ImageSegmentation/PlaneSegmentation/iscell` and `planeIdx`.

ii.
```python
segmentation = nwb[f"{OPHYS}/ImageSegmentation/PlaneSegmentation"]
iscell = segmentation["iscell"][:, 0] > 0.5
plane_idx = segmentation["planeIdx"][()].astype(np.int64)
plane_groups = nwb[f"{OPHYS}/Deconvolved"]
```

iii. The agent believed these were author-computed OASIS events exported from the processed session and therefore should not be deconvolved again.

## 2-b. How is the `neural` data processed?

i. For every plane, the dense deconvolved matrix is read once, manually curated columns are selected, planes are concatenated in a preallocated time-by-cell array, and each trial is transposed to cell-by-time float32. No dF/F calculation, smoothing, deconvolution, or temporal resampling is done by the converter.

ii.
```python
dense = dataset[:n_behavior_frames, :]
selected = np.asarray(dense[:, local_mask], dtype=np.float32)
pooled[:, cursor : cursor + n_selected] = selected
...
neural = np.ascontiguousarray(neural_by_time[s:e, :].T, dtype=np.float32)
```

iii. The notes say the stored events already reflect author preprocessing, so recomputation would be an incompatible second deconvolution. Dense once-per-plane reads were chosen for speed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent keeps only ROIs whose manual Suite2p `iscell[:,0]` flag exceeds 0.5. It does not apply the paper/reference speed-correlation putative-interneuron exclusion or a place-cell filter.

ii.
```python
iscell = segmentation["iscell"][:, 0] > 0.5
local_mask = iscell[plane_idx == plane]
selected = np.asarray(dense[:, local_mask], dtype=np.float32)
```

iii. It argues that place-cell filtering is analysis-specific and that filtering neurons correlated with speed would bias a decoder explicitly asked to predict speed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural frames are sliced at the same frame indices as behavior, beginning at the `trial_start` frame, so time zero is the first retained neural column.

ii.
```python
for trial, (s, e) in enumerate(zip(starts, stops)):
    neural = np.ascontiguousarray(neural_by_time[s:e, :].T, dtype=np.float32)
```

iii. The notes report exact raw-data audits and processing plots ruling out a one-frame shift.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native synchronized frames are retained at 64.483627 ms (15.5078125 Hz); no rebinning or resampling is applied.

ii.
```python
EXPECTED_DT_S = 1.0 / 15.5078125
...
"time_bin_size": EXPECTED_DT_S * 1000.0,
```

iii. Preserving native bins retains timing, stopped periods, speed, and licks without fabricating samples.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the timestamps attached to the raw `position` behavioral series, after checking all other dense behavioral timestamps match.

ii.
```python
position, timestamps = read_series(nwb, "position")
...
np.asarray(timestamps[s:e] - timestamps[s], dtype=np.float32)
```

iii. The dense behavioral streams are synchronized to the imaging frames, so any checked dense series supplies the same clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp in each trial is subtracted from every timestamp in that trial and the result is stored as float32.

ii.
```python
np.asarray(timestamps[s:e] - timestamps[s], dtype=np.float32)
```

iii. This directly implements time from the alignment event, with each trial beginning at zero.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses the identical `[s:e)` frame slice and is checked to have the same `n_time` as neural.

ii.
```python
if neural.shape[1] != n_time or inputs.shape != (4, n_time):
    raise ValueError(...)
```

iii. Shared synchronized indices and explicit shape validation provide alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the `environment` behavioral time series.

ii.
```python
environment, env_time = read_series(nwb, "environment")
valid = np.unique(environment[s:e][environment[s:e] >= 0])
```

iii. Inspection found exactly one valid binary environment label per trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Negative out-of-acquisition values are ignored, the code requires one value in `{0,1}`, and that value is repeated across the trial.

ii.
```python
if valid.size != 1 or valid[0] not in (0, 1):
    raise ValueError(...)
np.full(n_time, trial_environments[trial], dtype=np.float32)
```

iii. This validates the expected ENV1/ENV2 trial context and gives every time bin an aligned input.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the zero-based enumeration of paired `trial_start`/`teleport` intervals, not from the raw `trial number` values.

ii.
```python
for trial, (s, e) in enumerate(zip(starts, stops)):
```

iii. The notes say this retains the original within-session trial index even after QC rejection.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index is repeated as a float32 constant over all time bins; rejected trials do not renumber later trials.

ii.
```python
np.full(n_time, trial, dtype=np.float32)
```

iii. This is the requested continuous per-trial covariate and preserves experimental chronology.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from sparse `Reward/timestamps`, the common behavior timestamps, and the immediately preceding raw trial's boundaries.

ii.
```python
reward_times = nwb[f"{BEHAVIOR}/Reward/timestamps"][()]
reward_indices = nearest_timestamp_indices(timestamps, reward_times)
outcomes = np.array([np.any((reward_indices >= s) & (reward_indices < e)) ...])
```

iii. Sparse Reward records actual delivery, unlike reward-zone entry, and nearest-frame alignment is constrained to half a frame.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Each trial outcome is binary presence of at least one reward event in `[s,e)`. Trial 0 gets 0; otherwise the preceding native trial's outcome is repeated, even if that preceding trial was filtered from the output dataset.

ii.
```python
previous_outcome = int(outcomes[trial - 1]) if trial > 0 else 0
np.full(n_time, previous_outcome, dtype=np.float32)
```

iii. This matches omitted=0/rewarded=1 and preserves the meaning of “previous trial.”

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from raw `position`, raw `reward_zone` entry signals, trial boundaries, the trial-30 schedule, and fixed A/B/C intervals.

ii.
```python
zone_labels, observed_zones, contradictions = infer_zone_labels(position, reward_zone, starts, stops)
z0, z1 = ZONE_BOUNDS[label]
distance = distance_to_interval(trial_position, z0, z1)
```

iii. The paper/code define A/B/C bounds and a switch after trial 30. The agent infers each segment's modal observed zone and reports zero contradictions.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position before the interval is expressed relative to its start, position after it relative to its end, and every position inside the interval is exactly zero.

ii.
```python
distance[before] = position[before] - start
distance[after] = position[after] - end
```

iii. The agent interprets “distance to any location in the reward zone” as signed distance to the nearest point of the active interval.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit masks assign seven requested classes: `<-50`, `[-50,-10]`, `(-10,0)`, exactly zero, `(0,10]`, `(10,50]`, and `>50`.

ii.
```python
out[distance < -50] = 0
out[(distance >= -50) & (distance <= -10)] = 1
out[(distance > -10) & (distance < 0)] = 2
out[(distance > 0) & (distance <= 10)] = 4
```

iii. Synthetic boundary tests were documented; class 3 remains the initialized value only for zero.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data use the same `[s:e)` synchronized frame interval, and output shape is checked against neural length.

ii.
```python
trial_position = np.asarray(position[s:e], dtype=np.float32)
...
if ... outputs.shape != (6, n_time):
    raise ValueError(...)
```

iii. Independent raw-data audits reportedly matched the complete output row exactly.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the raw `position` behavioral series.

ii.
```python
position, timestamps = read_series(nwb, "position")
trial_position = np.asarray(position[s:e], dtype=np.float32)
```

iii. Position records centimeters along the 450-cm corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The per-trial synchronized position slice is passed directly to categorical thresholding; there is no interpolation or spatial averaging.

ii.
```python
discretize_position(trial_position)
```

iii. Native samples are retained because the target decoder requires time-varying labels.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Explicit masks create five 90-cm classes: `<90`, `[90,180)`, `[180,270)`, `[270,360]`, and `>360`.

ii.
```python
out[(position >= 90) & (position < 180)] = 1
out[(position >= 180) & (position < 270)] = 2
out[(position >= 270) & (position <= 360)] = 3
out[position > 360] = 4
```

iii. This follows the textual thresholds in the task and was boundary-tested.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is sliced with the identical `[s:e)` frame indices and included in the checked `6 × n_time` output matrix.

ii.
```python
trial_position = np.asarray(position[s:e], dtype=np.float32)
outputs = np.vstack((..., discretize_position(trial_position), ...))
```

iii. Shared frame timestamps and exact audit comparisons justify the alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the raw dense `lick` behavioral series.

ii.
```python
lick, lick_time = read_series(nwb, "lick")
```

iii. The notes identify it as cumulative within-frame lick counts produced by the reference synchronization.

## 9-b. What processing is involved in computing `output` *Lick*?

i. After lick-artifact trial rejection, every value greater than zero becomes 1 and all others become 0.

ii.
```python
(lick[s:e] > 0).astype(np.int64)
```

iii. This matches the requested binary output and the paper helper's clipping/binarization behavior.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick timestamps are first asserted equal to position timestamps and then the same `[s:e)` slice is used.

ii.
```python
if any(not np.allclose(timestamps, other) for other in aligned_times):
    raise ValueError(...)
...
(lick[s:e] > 0).astype(np.int64)
```

iii. The synchronized frame axis and output-shape check enforce alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is inferred from positions at which raw `reward_zone` is positive, paired trial boundaries, and the known pre/post-trial-30 schedule.

ii.
```python
classify_observed_zone(position[s:e][reward_zone[s:e] > 0])
```

iii. Omission trials lack direct zone observations, so neighboring rewarded trials and the known schedule provide their context.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Observed positions are classified by the nearest A/B/C center. The modal observation is selected separately for trials 0–29 and 30 onward, contradictions raise an error, labels map A/B/C to 0/1/2, and the value is repeated over time.

ii.
```python
label, _ = Counter(segment_observed).most_common(1)[0]
labels[lo:hi] = [label] * (hi - lo)
...
np.full(n_time, ZONE_TO_INT[label], dtype=np.int64)
```

iii. This follows the documented switch structure and was empirically contradiction-free.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from sparse raw `Reward/timestamps`, dense behavior timestamps, and trial boundaries.

ii.
```python
reward_times = nwb[f"{BEHAVIOR}/Reward/timestamps"][()]
reward_indices = nearest_timestamp_indices(timestamps, reward_times)
```

iii. The sparse Reward series represents delivery and therefore distinguishes omissions.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward events are mapped to the nearest behavior frame within half a bin. Presence of one or more events in a trial yields 1, otherwise 0; this value is repeated across the trial.

ii.
```python
outcomes = np.array(
    [np.any((reward_indices >= s) & (reward_indices < e)) for s, e in zip(starts, stops)],
    dtype=np.int64,
)
...
np.full(n_time, outcomes[trial], dtype=np.int64)
```

iii. Multiple deliveries remain a binary rewarded outcome, as required.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The converter mostly fails loudly on malformed timestamps, trial pairs, zone schedules, nonfinite values, or insufficient trials. Neural streams may be longer than behavior and are safely truncated to behavior length, but shorter streams raise an error. Missing zone observations on individual omission trials are filled from the inferred segment schedule.

ii.
```python
if dataset.shape[0] < n_behavior_frames:
    raise ValueError(...)
dense = dataset[:n_behavior_frames, :]
...
if contradictions:
    raise ValueError(...)
```

iii. The notes document ten files with one extra terminal neural row outside all trial intervals and extensive exact audits; no imputation of neural samples is attempted.

## 13-a. What are the most time-consuming steps of the code?

i. Reading dense per-plane neural HDF5 datasets, copying/transposing them into thousands of trial arrays, serializing the roughly 9-GiB pickle, and downstream decoder training are the major costs.

ii.
```python
dense = dataset[:n_behavior_frames, :]
...
pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes time full conversion at 55.73 s plus 8.41 s for writing and explain that HDF5 chunk decompression dominates naive access patterns.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining per-trial loops for outcomes, lick QC, environment validation, zone observations, and construction of variable-length arrays could partly be vectorized over full-session frames, though final ragged trial packaging necessarily remains iterative.

ii.
```python
outcomes = np.array([np.any((reward_indices >= s) & (reward_indices < e)) for s, e in zip(starts, stops)])
for trial, (s, e) in enumerate(zip(starts, stops)):
```

iii. The agent emphasizes that expensive neural I/O and ROI selection are already vectorized/preallocated; ragged trial loops are natural and relatively cheap.

## 13-c. What processing does the code repeat multiple times?

i. Per trial it repeatedly allocates constant input/output rows and applies slicing; zone observations and later converted data also revisit the same trial slices. Across separate executions, validation/audit reloads source or pickle data, but the conversion itself reads each behavioral stream and neural plane only once per session.

ii.
```python
np.full(n_time, trial_environments[trial], dtype=np.float32)
np.full(n_time, trial, dtype=np.float32)
np.full(n_time, outcomes[trial], dtype=np.int64)
```

iii. The notes explicitly designed the implementation to avoid the reference solution's separate all-file survey and second conversion read.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In ordinary conversion it computes diagnostic `observed_zones`, `plane_info`, extensive aggregate statistics, and metadata; `observed_zones` itself is not consumed after contradiction checking. With `--show-processing`, it also builds plots. Dense raw uncurated ROI columns are read and then discarded after applying `iscell`, an I/O tradeoff for much faster HDF5 access.

ii.
```python
zone_labels, observed_zones, contradictions = infer_zone_labels(...)
dense = dataset[:n_behavior_frames, :]
selected = np.asarray(dense[:, local_mask], dtype=np.float32)
del dense, selected
```

iii. The agent considers the dense read an intentional performance optimization and the diagnostics useful validation, even though they are not decoder features.
