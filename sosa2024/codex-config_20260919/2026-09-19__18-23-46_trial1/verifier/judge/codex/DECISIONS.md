# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files under `/app/data/sub-m*/` with `glob`, sorts them by parsed subject/session number, and treats each file as one session. Within each file it reads only the behavior series it needs plus the deconvolved ophys arrays and segmentation metadata.

ii.
```python
def discover_files(sample: bool) -> list[Path]:
    files = [Path(p) for p in glob.glob(str(DATA_ROOT / "sub-m*" / "*.nwb"))]
    files.sort(key=natural_key)
    ...
    return files

with h5py.File(path, "r") as nwb:
    position, timestamps = read_series(nwb, "position")
    speed, speed_time = read_series(nwb, "speed")
    ...
    neural_by_time, n_raw_rois, plane_info = load_curated_events(nwb, len(timestamps))
```

iii. In `CONVERSION_NOTES.md` Step 2, the AI says the dataset is “152 NWB 2.8 files in `data/sub-m*/sub-m*_ses-*_behavior+ophys.nwb`” and in Step 5 it commits to “one NWB=session.”

## 1-b. How are the data split into subjects (mice)?

i. Subjects are inferred from the NWB filenames. The subject number parsed from `sub-m<id>_ses-<id>` is used to build the global `subjects` list and the per-session `subject_idx`.

ii.
```python
def natural_key(path: str | Path) -> tuple[int, int]:
    match = re.search(r"sub-m(\d+)_ses-(\d+)", str(path))
    return int(match.group(1)), int(match.group(2))

subjects = sorted({f"m{natural_key(p)[0]}" for p in files}, key=lambda x: int(x[1:]))
subject_lookup = {subject: i for i, subject in enumerate(subjects)}
...
subject_idx.append(subject_lookup[info["subject"]])
```

iii. Step 2 of `CONVERSION_NOTES.md` lists 11 mice and Step 5 says subject IDs come from NWB metadata/file naming and are ordered naturally.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as exactly one session. Session identity is parsed from the filename and preserved in session metadata.

ii.
```python
subject_number, session_number = natural_key(path)
session_id = f"m{subject_number}_ses-{session_number:02d}"
...
info = {
    "session_id": session_id,
    "source_file": str(path.relative_to(DATA_ROOT)),
    "subject": f"m{subject_number}",
    "session_number": session_number,
    ...
}
```

iii. Step 2 says the supplied dataset contains one NWB per session, and Step 5 explicitly records “each target session is one NWB.”

## 1-d. How are the data split into trials?

i. Trials are defined by pairing every positive `trial_start` frame with the corresponding positive `teleport` frame, then slicing each stream on the zero-based interval `[start, stop)`.

ii.
```python
starts = np.flatnonzero(trial_start > 0).astype(np.int64)
stops = np.flatnonzero(teleport > 0).astype(np.int64)
if starts.size != stops.size or np.any(stops <= starts):
    raise ValueError(f"{session_id}: invalid trial start/teleport pairs")
...
neural = np.ascontiguousarray(neural_by_time[s:e, :].T, dtype=np.float32)
```

iii. In Step 4, the AI justifies zero-based `[trial_start, teleport)` slicing as the correct interpretation for the NWB export and explicitly rejects applying the paper code’s historical `-1` offset to NWB data.

## 1-e. How are trials filtered based on quality controls?

i. The AI removes whole trials only when they meet the paper’s lick-artifact criterion: more than 30% of on-track frames have cumulative lick count `> 2`. It also errors out if a session would end up with fewer than two retained trials.

ii.
```python
lick_artifact_fraction = np.array(
    [np.mean(lick[s:e] > 2) for s, e in zip(starts, stops)], dtype=np.float64
)
keep = lick_artifact_fraction <= 0.30
...
for trial, (s, e) in enumerate(zip(starts, stops)):
    if not keep[trial]:
        continue
...
if len(neural_trials) < 2:
    raise ValueError(f"{session_id}: fewer than two valid trials")
```

iii. Step 4 resolves a code/paper discrepancy in favor of the Methods text and says “Use >30%. The supplied data reproduce the paper’s 81 rejected trials exactly.” Step 5 also says lick-artifact rejection is whole-trial because lick is a required output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` from the NWB `processing/ophys/Deconvolved/plane*/data` arrays plus `ImageSegmentation/PlaneSegmentation/{iscell, planeIdx}` to choose curated cells and pool planes.

ii.
```python
segmentation = nwb[f"{OPHYS}/ImageSegmentation/PlaneSegmentation"]
iscell = segmentation["iscell"][:, 0] > 0.5
plane_idx = segmentation["planeIdx"][()].astype(np.int64)
plane_groups = nwb[f"{OPHYS}/Deconvolved"]
...
dataset = plane_groups[f"{plane_name}/data"]
```

iii. Step 1 says the “native data are already author-processed `multi_anim_sess` pickles, so dF/F does not need to be recomputed,” and Step 5 maps `processing/ophys/Deconvolved/plane*/data` directly to `neural`.

## 2-b. How is the `neural` data processed?

i. The AI does not recompute dF/F or deconvolution. It truncates each plane to the behavior length, applies the curated-cell mask, concatenates planes, and later slices the pooled time-by-cell matrix into per-trial cell-by-time matrices.

ii.
```python
dense = dataset[:n_behavior_frames, :]
selected = np.asarray(dense[:, local_mask], dtype=np.float32)
pooled[:, cursor : cursor + n_selected] = selected
...
neural = np.ascontiguousarray(neural_by_time[s:e, :].T, dtype=np.float32)
```

iii. The explicit justification in Step 1 is that the NWB already supplies “author-computed OASIS-deconvolved calcium events,” so the converter should use those directly instead of “an incompatible second deconvolution.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron-level quality filter is Suite2p manual curation via `iscell[:, 0] > 0.5`. No extra interneuron/speed-correlation filter is applied.

ii.
```python
segmentation = nwb[f"{OPHYS}/ImageSegmentation/PlaneSegmentation"]
iscell = segmentation["iscell"][:, 0] > 0.5
...
local_mask = iscell[plane_idx == plane]
selected = np.asarray(dense[:, local_mask], dtype=np.float32)
```

iii. Step 3 and Step 10 justify this by arguing that the paper’s putative-interneuron exclusion was analysis-specific and would be inappropriate for a general neural-to-behavior decoder, especially because speed is itself a requested output.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to trial start by slicing each trial on `[trial_start, teleport)`. No additional resampling or offset correction is applied.

ii.
```python
starts = np.flatnonzero(trial_start > 0).astype(np.int64)
...
for trial, (s, e) in enumerate(zip(starts, stops)):
    ...
    neural = np.ascontiguousarray(neural_by_time[s:e, :].T, dtype=np.float32)
```

iii. Step 5 calls this “align each trial at its track-entry frame,” and metadata record the alignment event as “start of trial (entry onto the 450-cm virtual track).”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native frame rate, using a constant bin size of `1 / 15.5078125` seconds (`64.483627... ms`). It does not rebin or resample the neural stream.

ii.
```python
EXPECTED_DT_S = 1.0 / 15.5078125
...
dt = np.diff(timestamps)
if not np.allclose(dt, EXPECTED_DT_S, rtol=1e-6, atol=1e-9):
    raise ValueError(f"{session_id}: nonuniform or unexpected sample interval")
...
"time_bin_size": EXPECTED_DT_S * 1000.0,
```

iii. Step 4 says to “trust synchronized behavioral timestamps/per-plane description: common bin size 64.483627 ms,” and Step 5 lists native temporal bins as a key decision.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the timestamps attached to the `position` behavior series, which the AI treats as the shared behavior frame times for the session.

ii.
```python
position, timestamps = read_series(nwb, "position")
...
inputs = np.vstack(
    (
        np.asarray(timestamps[s:e] - timestamps[s], dtype=np.float32),
        ...
    )
)
```

iii. Step 2 notes that the behavior series share the same synchronized timestamps; the AI therefore uses one shared time base rather than a specific dedicated “trial number” timestamp vector.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each retained trial, the AI subtracts the first timestamp in the trial so time starts at zero and then increases in native frame steps.

ii.
```python
np.asarray(timestamps[s:e] - timestamps[s], dtype=np.float32)
```

iii. Step 5 maps this variable as “`timestamp[start:stop] - timestamp[start]`, seconds” and Step 10 reports that all converted trials start at time 0 and have strictly increasing time.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time trace is built from the same `[s:e]` frame interval used to slice neural activity, so it is sample-aligned without interpolation.

ii.
```python
neural = np.ascontiguousarray(neural_by_time[s:e, :].T, dtype=np.float32)
inputs = np.vstack(
    (
        np.asarray(timestamps[s:e] - timestamps[s], dtype=np.float32),
        ...
    )
)
```

iii. Step 4 says “all behavioral labels use the same timestamps without resampling,” and Step 10 says exact trial checks ruled out any one-frame shift.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes directly from the `environment` behavior time series.

ii.
```python
environment, env_time = read_series(nwb, "environment")
...
valid = np.unique(environment[s:e][environment[s:e] >= 0])
```

iii. Step 5 maps `environment` to `input[1]` and interprets it as the paper’s binary morph/context variable.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI checks that each trial has exactly one valid binary environment label (`0` or `1`) and then repeats that constant value across all time bins in the trial.

ii.
```python
for s, e in zip(starts, stops):
    valid = np.unique(environment[s:e][environment[s:e] >= 0])
    if valid.size != 1 or valid[0] not in (0, 1):
        raise ValueError(f"{session_id}: trial lacks one binary environment label")
    trial_environments.append(int(valid[0]))
...
np.full(n_time, trial_environments[trial], dtype=np.float32)
```

iii. Step 2 says environment is “exactly one valid value (0 or 1) per trial,” and Step 5 says to use that unique value as a repeated per-trial context.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the trial enumeration induced by the paired `trial_start` and `teleport` boundaries, not from the stored NWB `trial number` values.

ii.
```python
starts = np.flatnonzero(trial_start > 0).astype(np.int64)
stops = np.flatnonzero(teleport > 0).astype(np.int64)
...
np.full(n_time, trial, dtype=np.float32)
```

iii. Step 5 maps trial number as the within-session trial index and says the NWB `trial number` agrees with this enumeration, but the converter uses the explicit boundary-derived index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transformation is applied beyond taking the loop index for the current trial and repeating it across all time bins in that trial.

ii.
```python
np.full(n_time, trial, dtype=np.float32)
```

iii. Step 5 records this as a “zero-based trial index, repeated across time.”

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the sparse reward-delivery timestamps in `processing/behavior/BehavioralTimeSeries/Reward/timestamps`, after mapping those timestamps to the nearest behavior frames and reducing them to per-trial reward outcomes.

ii.
```python
reward_times = nwb[f"{BEHAVIOR}/Reward/timestamps"][()]
reward_indices = nearest_timestamp_indices(timestamps, reward_times)
outcomes = np.array(
    [np.any((reward_indices >= s) & (reward_indices < e)) for s, e in zip(starts, stops)],
    dtype=np.int64,
)
```

iii. Step 4 and Step 5 explicitly choose sparse reward delivery, rather than reward-zone entry, as the source of reward outcome.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI computes current-trial outcomes first, then assigns each trial the previous trial’s binary outcome; the first trial gets 0 by convention. The chosen value is repeated across the trial’s time bins.

ii.
```python
previous_outcome = int(outcomes[trial - 1]) if trial > 0 else 0
...
np.full(n_time, previous_outcome, dtype=np.float32)
```

iii. Step 5 says this input must not leak current-trial outcome and explicitly sets “first trial previous outcome = 0.”

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the continuous `position` trace together with an inferred per-trial reward-zone label obtained from `reward_zone` observations within each trial and the known trial-30 pre/post switch structure.

ii.
```python
zone_labels, observed_zones, contradictions = infer_zone_labels(
    position, reward_zone, starts, stops
)
...
trial_position = np.asarray(position[s:e], dtype=np.float32)
label = zone_labels[trial]
z0, z1 = ZONE_BOUNDS[label]
distance = distance_to_interval(trial_position, z0, z1)
```

iii. Step 4 says explicit scene labels are absent in NWB, so zone identity is inferred from observed reward-zone-entry positions and the documented switch-after-trial-30 rule.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The AI converts position to signed distance from the active reward-zone interval: negative before the interval, zero inside it, and positive after it.

ii.
```python
def distance_to_interval(position: np.ndarray, start: float, end: float) -> np.ndarray:
    distance = np.zeros(position.shape, dtype=np.float32)
    before = position < start
    after = position > end
    distance[before] = position[before] - start
    distance[after] = position[after] - end
    return distance
```

iii. Step 3 and Step 5 justify this as “distance to any location in the 50-cm active reward-zone interval,” explicitly distinguishing it from the paper’s circular reward-relative coordinate.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous signed distance is mapped into seven classes with explicit comparisons: `< -50`, `[-50, -10]`, `(-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50`.

ii.
```python
def discretize_distance(distance: np.ndarray) -> np.ndarray:
    out = np.full(distance.shape, 3, dtype=np.int64)
    out[distance < -50] = 0
    out[(distance >= -50) & (distance <= -10)] = 1
    out[(distance > -10) & (distance < 0)] = 2
    out[(distance > 0) & (distance <= 10)] = 4
    out[(distance > 10) & (distance <= 50)] = 5
    out[distance > 50] = 6
    return out
```

iii. Step 5 spells out this boundary convention explicitly, including that class 1 includes `-50` and `-10`, class 3 is exactly zero, and class 5 includes `10` and `50`.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Distance is computed from `position[s:e]` inside the same per-trial slice used for `neural`, then stacked into the output matrix for that trial.

ii.
```python
trial_position = np.asarray(position[s:e], dtype=np.float32)
distance = distance_to_interval(trial_position, z0, z1)
...
outputs = np.vstack(
    (
        discretize_distance(distance),
        ...
    )
)
```

iii. Step 10 states that independent raw-trial checks found exact neural/input/output trial alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is taken directly from the `position` behavior time series.

ii.
```python
position, timestamps = read_series(nwb, "position")
...
trial_position = np.asarray(position[s:e], dtype=np.float32)
```

iii. Step 5 maps `position` directly to the absolute-position output, with only task-required discretization.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI slices the trial’s continuous position values and discretizes them into five bins over the 450-cm track.

ii.
```python
def discretize_position(position: np.ndarray) -> np.ndarray:
    out = np.zeros(position.shape, dtype=np.int64)
    out[(position >= 90) & (position < 180)] = 1
    out[(position >= 180) & (position < 270)] = 2
    out[(position >= 270) & (position <= 360)] = 3
    out[position > 360] = 4
    return out
```

iii. Step 5 describes this as the requested five equal 90-cm bins and says no spatial averaging or paper-style 10-cm binning should be used for this task.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is thresholded with five fixed bins: `<90`, `[90,180)`, `[180,270)`, `[270,360]`, and `>360`.

ii.
```python
out = np.zeros(position.shape, dtype=np.int64)
out[(position >= 90) & (position < 180)] = 1
out[(position >= 180) & (position < 270)] = 2
out[(position >= 270) & (position <= 360)] = 3
out[position > 360] = 4
```

iii. Step 5 explicitly notes the boundary convention: “360 is assigned to bin 3; >360 to bin 4.”

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same trial slice `[s:e]` as the neural data, so position and neural activity share frame-by-frame alignment.

ii.
```python
neural = np.ascontiguousarray(neural_by_time[s:e, :].T, dtype=np.float32)
trial_position = np.asarray(position[s:e], dtype=np.float32)
...
discretize_position(trial_position)
```

iii. Step 10 reports that raw-data spot checks showed the reconstructed output matrices exactly matched trial-aligned slices from the NWB files.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the `lick` behavior time series.

ii.
```python
lick, lick_time = read_series(nwb, "lick")
...
(lick[s:e] > 0).astype(np.int64)
```

iii. Step 2 notes that lick is stored as cumulative counts per imaging frame, and Step 5 maps that stream to the binary lick output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. After trial-level lick-artifact rejection, the AI binarizes lick framewise with `lick > 0`.

ii.
```python
keep = lick_artifact_fraction <= 0.30
...
(lick[s:e] > 0).astype(np.int64)
```

iii. Step 1 notes that the reference code binarizes cumulative lick counts, and Step 5 says time-varying binary lick should be kept because lick is a required decoder output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The lick output is built from the same `[s:e]` indices as the neural trial slice, so it is aligned at the native frame resolution.

ii.
```python
neural = np.ascontiguousarray(neural_by_time[s:e, :].T, dtype=np.float32)
...
(lick[s:e] > 0).astype(np.int64)
```

iii. Step 10 says exact `np.allclose` checks on raw trials confirmed the lick output was synchronized with the neural time axis.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from the `reward_zone` time series and the positions where that signal is active, together with the known pre/post trial-30 schedule.

ii.
```python
observed = [
    classify_observed_zone(position[s:e][reward_zone[s:e] > 0])
    for s, e in zip(starts, stops)
]
...
np.full(n_time, ZONE_TO_INT[label], dtype=np.int64)
```

iii. Step 4 says zone identity is not stored explicitly for omission trials, so the AI infers the modal label separately before and after the switch point and fills omissions from that schedule.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI classifies observed reward-zone-entry positions to A/B/C, chooses the modal label separately for trials `0:30` and `30:end`, checks for contradictions, then repeats the resulting per-trial zone label across all time bins in the trial.

ii.
```python
for lo, hi in ((0, min(30, len(starts))), (min(30, len(starts)), len(starts))):
    segment_observed = [observed[i] for i in range(lo, hi) if observed[i] is not None]
    label, _ = Counter(segment_observed).most_common(1)[0]
    labels[lo:hi] = [label] * (hi - lo)
...
np.full(n_time, ZONE_TO_INT[label], dtype=np.int64)
```

iii. Step 5 identifies this as a key decision and defends it as an empirical, contradiction-free reconstruction of the schedule without hard-coding per-mouse sequences.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from sparse reward-delivery timestamps in `BehavioralTimeSeries/Reward/timestamps`.

ii.
```python
reward_times = nwb[f"{BEHAVIOR}/Reward/timestamps"][()]
reward_indices = nearest_timestamp_indices(timestamps, reward_times)
```

iii. Step 4 says sparse `Reward` is the actual delivery event, whereas reward-zone entry can occur on omission trials.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are snapped to the nearest behavior frame with a half-bin tolerance check, then each trial is marked rewarded if any mapped reward frame falls within `[start, stop)`. The resulting binary label is repeated across the whole trial.

ii.
```python
reward_indices = nearest_timestamp_indices(timestamps, reward_times)
outcomes = np.array(
    [np.any((reward_indices >= s) & (reward_indices < e)) for s, e in zip(starts, stops)],
    dtype=np.int64,
)
...
np.full(n_time, outcomes[trial], dtype=np.int64)
```

iii. Step 5 says this “exactly preserves 84.66% rewarded trials,” and Step 12 explains why the label stays per-trial rather than switching only after delivery.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles some issues defensively and rejects others. It truncates neural planes to the behavior length when neural arrays are one sample longer, drops lick-artifact trials, fills omission-trial zone labels by pre/post-switch majority, and validates timestamp alignment, trial pairing, environment uniqueness, and shape consistency with hard errors.

ii.
```python
if dataset.shape[0] < n_behavior_frames:
    raise ValueError(f"{plane_name}: neural stream is shorter than behavior")
dense = dataset[:n_behavior_frames, :]
...
if any(not np.allclose(timestamps, other) for other in aligned_times):
    raise ValueError(f"{session_id}: behavior timestamps are not aligned")
...
keep = lick_artifact_fraction <= 0.30
...
if len(neural_trials) < 2:
    raise ValueError(f"{session_id}: fewer than two valid trials")
```

iii. Step 2 notes the real data issue of ten dual-plane sessions having one extra final neural row, and Step 4 says this is “safely ignored by slicing to behavioral trial endpoints.” The notes do not claim more general imputation beyond the zone-schedule fill and trial rejection rules.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive steps are dense HDF5 reads of the deconvolved planes, per-session trial assembly/copying into the output structure, optional plotting, and serializing the large pickle.

ii.
```python
dense = dataset[:n_behavior_frames, :]
selected = np.asarray(dense[:, local_mask], dtype=np.float32)
...
for trial, (s, e) in enumerate(zip(starts, stops)):
    ...
    neural_trials.append(neural)
...
with args.outpicklefile.open("wb") as stream:
    pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Step 6 explicitly identifies per-plane HDF5 access patterns as the main bottleneck and says the code was written to avoid thousands of small point selections.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still has Python-level loops over planes, sessions, and trials. The per-trial lick-artifact calculation, environment validation, and trial materialization could in principle be further vectorized, but variable trial lengths make full vectorization awkward.

ii.
```python
for plane_name in plane_names:
    ...
for s, e in zip(starts, stops):
    valid = np.unique(environment[s:e][environment[s:e] >= 0])
...
for trial, (s, e) in enumerate(zip(starts, stops)):
    ...
```

iii. Step 6 says the AI already vectorized the behavior transforms where practical and avoided slower HDF5 access patterns, but it kept the per-trial loop as the natural structure for variable-length laps.

## 13-c. What processing does the code repeat multiple times?

i. The code repeats some trial-wise slicing work: it scans all trials once to compute lick-artifact fractions, again to validate environments, and again to build neural/input/output matrices. In `--show-processing` mode it also recomputes discretizations for plotting.

ii.
```python
lick_artifact_fraction = np.array(
    [np.mean(lick[s:e] > 2) for s, e in zip(starts, stops)], dtype=np.float64
)
...
for s, e in zip(starts, stops):
    valid = np.unique(environment[s:e][environment[s:e] >= 0])
...
for trial, (s, e) in enumerate(zip(starts, stops)):
    ...
```

iii. The notes do not defend this repetition in detail, but Step 6 emphasizes that the AI avoided much larger repeated work such as rereading every plane many times or running a separate survey pass over all files.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The converter performs extra diagnostic and bookkeeping work that is not used by downstream decoder training: optional `plot_processing` figures, collection of `plane_info`/`stats`/`session_info`, printing conversion summaries, and returning `observed_zones` from `infer_zone_labels` even though only `zone_labels` and `contradictions` matter afterward.

ii.
```python
zone_labels, observed_zones, contradictions = infer_zone_labels(
    position, reward_zone, starts, stops
)
...
if show_processing:
    plot_processing(...)
...
info = {
    ...
    "planes": plane_info,
    ...
}
stats = {
    ...
}
```

iii. Step 6 and later review steps justify this extra work as validation and documentation support rather than core training input, and Step 13 says the README/notes/plots were part of the required deliverables.
