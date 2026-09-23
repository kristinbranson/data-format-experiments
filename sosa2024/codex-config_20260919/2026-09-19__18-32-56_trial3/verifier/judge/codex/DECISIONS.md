# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers every `.nwb` file under `/app/data/sub-*`, sorts them by numeric mouse ID and session ID parsed from the filename, and processes each file as one session. It reads the NWB/HDF5 structure directly with `h5py` instead of using `pynwb`.

ii.
```python
def discover_files(sample: bool) -> list[Path]:
    files = sorted(DATA_ROOT.glob("sub-*/*.nwb"), key=natural_file_key)
    if not files:
        raise FileNotFoundError(f"No NWB files found under {DATA_ROOT}")
    if not sample:
        return files
```

```python
with h5py.File(path, "r") as nwb:
    behavior = nwb[BEHAVIOR_ROOT]
```

iii. In `CONVERSION_NOTES.md` Step 2 and Step 5, the AI says the supplied data are one NWB file per mouse/day session and that the repository has no native NWB loader, so direct HDF5 reads are a format adaptation while preserving the same processed streams.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the discovered filenames/directories and then re-read from NWB metadata (`general/subject/subject_id`). The final dataset stores one unique sorted list of subject IDs and a per-session `subject_idx`.

ii.
```python
subjects = sorted(
    {f"m{natural_file_key(path)[0]}" for path in files},
    key=lambda value: int(value[1:]),
)
subject_lookup = {subject: i for i, subject in enumerate(subjects)}
```

```python
subject = decode_text(nwb["general/subject/subject_id"])
```

iii. In `CONVERSION_NOTES.md` Step 2, the AI notes that the local dataset has 11 `sub-<mouse>` directories and treats each mouse as one subject.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session order is the natural file order by mouse ID and session number, and the NWB `general/session_id` field is stored in session metadata.

ii.
```python
def natural_file_key(path: Path) -> tuple[int, int]:
    match = re.search(r"sub-m(\d+)_ses-(\d+)", path.name)
    ...
    return int(match.group(1)), int(match.group(2))
```

```python
session_id = decode_text(nwb["general/session_id"])
session_tag = f"{subject}_ses-{session_id}"
```

iii. In `CONVERSION_NOTES.md` Steps 2 and 5, the AI states that each provided mouse/day NWB is a session and that all 152 such sessions should be converted.

## 1-d. How are the data split into trials?

i. Trials are split with explicit frame indices from dense behavioral streams: starts are all samples where `trial_start > 0`, and stops are all samples where `teleport > 0`. Converted trials use the half-open interval `[start:stop)`, so the teleport sample itself is excluded.

ii.
```python
def trial_bounds(group: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    starts = np.flatnonzero(group["trial_start/data"][:] > 0)
    stops = np.flatnonzero(group["teleport/data"][:] > 0)
    ...
    return starts, stops
```

```python
for raw_trial, (start, stop) in enumerate(zip(starts, stops)):
    ...
    trial_position = position[start:stop]
    ...
    neural = activity[start:stop].T.copy()
```

iii. In `CONVERSION_NOTES.md` Steps 4 and 5, the AI says NWB contains explicit binary `trial_start` and `teleport` samples and argues that `[trial_start_index:teleport_index)` is the correct native NWB alignment, unlike the one-sample legacy offset in the paper’s older pipeline objects.

## 1-e. How are trials filtered based on quality controls?

i. The AI excludes whole trials when more than 30% of frames in the trial have cumulative `lick > 2`, calling these lick-sensor-fault trials. It does not apply a separate minimum-trial-length filter.

ii.
```python
bad_trials = np.array(
    [np.mean(lick[a:b] > 2) > LICK_FAULT_FRACTION for a, b in zip(starts, stops)],
    dtype=bool,
)
...
for raw_trial, (start, stop) in enumerate(zip(starts, stops)):
    if bad_trials[raw_trial]:
        continue
```

iii. In `CONVERSION_NOTES.md` Steps 4 and 5, the AI says the paper’s authoritative lick QC is `>30%` of frames with lick count `>2`, and because the target format has no missing-label mask for only the lick output, it chose to exclude those trials entirely.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The converted `neural` arrays come from the NWB `processing/ophys/Deconvolved/plane*/data` datasets after cell filtering and plane pooling. Raw `Fluorescence` and `Neuropil` are only used transiently to recreate the interneuron-quality-control mask.

ii.
```python
def load_deconvolved(nwb: h5py.File, keep_global: np.ndarray, n_behavior: int) -> np.ndarray:
    ...
    deconvolved = nwb[f"{OPHYS_ROOT}/Deconvolved"]
    for plane_name in sorted(deconvolved):
        group = deconvolved[plane_name]
        ...
        chunks.append(group["data"][:n_behavior, local].astype(np.float32, copy=False))
```

```python
fluorescence = nwb[f"{OPHYS_ROOT}/Fluorescence"]
neuropil = nwb[f"{OPHYS_ROOT}/Neuropil"]
```

iii. In `CONVERSION_NOTES.md` Step 1 and Step 5, the AI argues that the provided NWBs already expose the author-produced event traces used by the manuscript decoder, so it should load those directly rather than recomputing them for output.

## 2-b. How is the `neural` data processed?

i. The AI keeps the native deconvolved event stream on the native imaging time grid. It trims each plane to the behavior-aligned length, filters cells, concatenates planes in global ROI order, and slices each trial as `[start:stop)` without further smoothing or rebinning. A separate transient dF/F recreation is used only for interneuron QC, not for the exported `neural` arrays.

ii.
```python
activity = load_deconvolved(nwb, keep, n_behavior)
...
neural = activity[start:stop].T.copy()
```

```python
corrected = f - 0.7 * f_neu + 0.7 * f_neu.mean(axis=1, keepdims=True)
baseline = ndimage.gaussian_filter1d(corrected, 15, axis=1)
baseline = ndimage.minimum_filter1d(baseline, 300, axis=1)
baseline = ndimage.maximum_filter1d(baseline, 300, axis=1)
with np.errstate(divide="ignore", invalid="ignore"):
    dff = (corrected - baseline) / np.abs(baseline)
dff = ndimage.gaussian_filter1d(dff, 2, axis=1)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI explicitly chose “author-provided OASIS-deconvolved calcium events” as the exported neural representation and said recomputing deconvolution would only introduce numerical differences. The transient dF/F recreation is justified there as being needed only for the paper’s interneuron filter.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies two neuron filters before exporting `neural`: keep only manually curated Suite2p cells from `iscell[:,0]`, and then exclude cells whose reconstructed dF/F has Pearson correlation `> 0.5` with running speed.

ii.
```python
segmentation = nwb[f"{OPHYS_ROOT}/ImageSegmentation/PlaneSegmentation"]
iscell = segmentation["iscell"][:, 0].astype(bool)
keep, speed_correlations = compute_interneuron_mask(
    nwb, iscell, speed, starts, stops
)
```

```python
keep[ids[corr > INTERNEURON_R_THRESHOLD]] = False
```

iii. In `CONVERSION_NOTES.md` Steps 4 and 5, the AI says these are the paper’s two relevant neuron-curation rules: manual Suite2p cell curation and exclusion of putative interneurons with dF/F-speed correlation above `0.5`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to the first imaging frame where `trial_start > 0`. The first neural sample in each converted trial is the first sample after trial entry, and the last included sample is immediately before teleport.

ii.
```python
relative_time = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
...
neural = activity[start:stop].T.copy()
```

iii. In `CONVERSION_NOTES.md` Steps 4 and 5, the AI says the requested alignment event is the explicit NWB trial-start frame and that the teleport sample should be excluded because it is position-ambiguous.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use the native aligned frame grid at `1 / 15.5078125` seconds per sample, stored as `64.4836272 ms` in metadata. No temporal rebinning or interpolation is applied during conversion.

ii.
```python
DT_SECONDS = 1.0 / 15.5078125
...
"time_bin_size": DT_SECONDS * 1000.0,
```

```python
if not np.allclose(np.diff(timestamps), DT_SECONDS, rtol=0, atol=1e-10):
    raise ValueError(f"Nonuniform or unexpected timestamps in {path}")
```

iii. In `CONVERSION_NOTES.md` Steps 2, 4, and 5, the AI says the NWB behavior streams are already synchronized to the imaging grid and that no extra resampling should be introduced.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the dense synchronized behavior timestamps, specifically `position/timestamps`.

ii.
```python
timestamps = behavior["position/timestamps"][:]
...
relative_time = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI describes this variable as “dense behavior timestamps” with the timestamp at trial start subtracted.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the AI slices the timestamps over `[start:stop)` and subtracts the first timestamp so that each trial starts at `0.0` seconds.

ii.
```python
relative_time = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
...
decoder_input = np.vstack(
    [
        relative_time,
        ...
    ]
).astype(np.float32, copy=False)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says the target should be aligned so that `input[0, 0] == 0` for every trial.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by using exactly the same `[start:stop)` frame indices used for the neural slice, so input time and neural activity share the same per-frame samples.

ii.
```python
relative_time = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
...
neural = activity[start:stop].T.copy()
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says both neural and dense behavior streams already lie on the same imaging frame grid, so alignment is achieved by common slicing rather than by interpolation.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the dense synchronized `environment/data` behavioral variable.

ii.
```python
environment = behavior["environment/data"][:]
...
env_values = np.unique(environment[start:stop])
```

iii. In `CONVERSION_NOTES.md` Steps 2 and 5, the AI says this stream is already synchronized and uses `0=Env1`, `1=Env2`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial, the AI checks that environment is constant and binary, then broadcasts that single value across all timepoints in the trial.

ii.
```python
if env_values.size != 1 or env_values[0] not in (0, 1):
    raise ValueError(f"Invalid environment in {session_tag} trial {raw_trial}: {env_values}")
...
np.full(T, env_values[0], dtype=np.float32)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says environment is a per-trial binary context variable that should be broadcast into the per-timepoint input array.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the synchronized dense `trial number/data` stream, not from the loop index over retained trials.

ii.
```python
trial_number = behavior["trial number/data"][:]
...
trial_values = np.unique(trial_number[start:stop])
```

```python
np.full(T, trial_values[0], dtype=np.float32)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says it chose the “native zero-based trial number” and wanted to preserve original numbering across excluded lick-fault trials.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The AI verifies there is exactly one unique trial-number value inside each trial interval, then broadcasts that constant value across timepoints.

ii.
```python
if trial_values.size != 1:
    raise ValueError(f"Multiple trial IDs in {session_tag} trial {raw_trial}: {trial_values}")
...
np.full(T, trial_values[0], dtype=np.float32)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI justifies this as preserving the original experimental lap IDs rather than reindexing after trial exclusions.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from sparse reward delivery timestamps (`Reward/timestamps`) together with dense `reward_zone/data` evidence, summarized trial-by-trial by `reward_outcomes(...)`, and then shifted by one trial.

ii.
```python
reward_zone_signal = behavior["reward_zone/data"][:]
reward_timestamps = behavior["Reward/timestamps"][:]
...
outcomes = reward_outcomes(
    timestamps, reward_timestamps, reward_zone_signal, starts, stops
)
previous_outcomes = np.r_[0, outcomes[:-1]].astype(np.int8)
```

iii. In `CONVERSION_NOTES.md` Steps 4 and 5, the AI says it wanted to match the paper’s `get_trial_types` logic for reward outcome rather than using reward timestamps alone.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. First, the AI computes each raw trial’s reward outcome as “reward delivered during `[start, stop)` and reward-zone evidence present in that same interval.” Then it shifts that vector by one trial, sets the first trial’s previous outcome to `0`, and broadcasts the result across each current trial.

ii.
```python
def reward_outcomes(...):
    ...
    delivered = np.any(
        (reward_timestamps >= timestamps[start])
        & (reward_timestamps < timestamps[stop])
    )
    zone_evidence = np.any(reward_zone_signal[start:stop] > 0)
    out[i] = int(delivered and zone_evidence)
```

```python
previous_outcomes = np.r_[0, outcomes[:-1]].astype(np.int8)
...
np.full(T, previous_outcomes[raw_trial], dtype=np.float32)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI explicitly says prior outcome should refer to the preceding raw experimental trial, even if that earlier trial is later excluded for lick QC.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from dense `position/data` and a per-trial reward-zone label parsed from the session `identifier` string, using fixed zone bounds `A=80–130`, `B=200–250`, `C=320–370`.

ii.
```python
scene = decode_text(nwb["identifier"]).rsplit("/", 1)[-1]
...
labels = scene_zone_labels(scene, starts.size)
...
trial_position = position[start:stop]
label = labels[raw_trial]
zone_start, zone_stop = ZONE_BOUNDS[label]
```

iii. In `CONVERSION_NOTES.md` Steps 2, 4, and 5, the AI says session identifiers encode fixed or switched reward-zone protocols and that all switch sessions change after trial 30, so these metadata are sufficient to assign the per-trial reward-zone location.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the AI computes signed linear distance to the reward-zone interval: negative before the zone, exactly `0` anywhere inside the zone, and positive after the zone. That continuous signed distance is then discretized.

ii.
```python
signed_distance = np.where(
    trial_position < zone_start,
    trial_position - zone_start,
    np.where(trial_position > zone_stop, trial_position - zone_stop, 0.0),
)
```

```python
discretize_distance(signed_distance)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says the requested output is not the paper’s circular reward-relative position, so it defines the requested linear distance as zero throughout the whole reward-zone interval.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is discretized into seven bins with exact comparisons implementing the requested boundaries: `<-50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, `>50`.

ii.
```python
def discretize_distance(distance: np.ndarray) -> np.ndarray:
    out = np.empty(distance.shape, dtype=np.int8)
    out[distance < -50] = 0
    out[(distance >= -50) & (distance < -10)] = 1
    out[(distance >= -10) & (distance < 0)] = 2
    out[distance == 0] = 3
    out[(distance > 0) & (distance <= 10)] = 4
    out[(distance > 10) & (distance <= 50)] = 5
    out[distance > 50] = 6
    return out
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says these exact inequalities were chosen to match the task specification. It also added `verify_discretization_edges()` to check the edge cases explicitly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by computing distance from the same `position[start:stop]` samples used to define each neural trial, so there is one distance label per neural frame.

ii.
```python
trial_position = position[start:stop]
...
decoder_output = np.vstack(
    [
        discretize_distance(signed_distance),
        ...
    ]
).astype(np.int8, copy=False)
neural = activity[start:stop].T.copy()
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says all requested outputs should live on the same native frame grid as neural activity.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the dense synchronized `position/data` behavioral stream.

ii.
```python
position = behavior["position/data"][:]
...
trial_position = position[start:stop]
```

iii. In `CONVERSION_NOTES.md` Steps 2 and 5, the AI treats position as the native synchronized animal position in centimeters along the 450-cm corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI slices the per-trial position trace and discretizes it into five 90-cm bins spanning the track.

ii.
```python
decoder_output = np.vstack(
    [
        discretize_distance(signed_distance),
        discretize_position(trial_position),
        ...
    ]
).astype(np.int8, copy=False)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says this is a task-driven discretization of the native position signal rather than a paper-derived 10-cm spatial binning.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is discretized into five bins with boundaries `<90`, `[90,180)`, `[180,270)`, `[270,360]`, `>360`.

ii.
```python
def discretize_position(position: np.ndarray) -> np.ndarray:
    out = np.empty(position.shape, dtype=np.int8)
    out[position < 90] = 0
    out[(position >= 90) & (position < 180)] = 1
    out[(position >= 180) & (position < 270)] = 2
    out[(position >= 270) & (position <= 360)] = 3
    out[position > 360] = 4
    return out
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says it followed the task’s five equal-size bins over the 450-cm track and made the endpoint conventions explicit.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned sample-for-sample with neural activity by using the same `[start:stop)` trial slice.

ii.
```python
trial_position = position[start:stop]
...
neural = activity[start:stop].T.copy()
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says no extra alignment is needed because position is already synchronized to imaging frames.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the dense synchronized `lick/data` behavioral stream.

ii.
```python
lick = behavior["lick/data"][:]
...
trial_lick = lick[start:stop]
```

iii. In `CONVERSION_NOTES.md` Steps 2 and 5, the AI says the stored lick values are cumulative per imaging frame and should be binarized for the requested decoder output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI binarizes the lick trace at each timepoint: `lick > 0` becomes `1`, otherwise `0`.

ii.
```python
decoder_output = np.vstack(
    [
        ...,
        (trial_lick > 0).astype(np.int8),
        ...
    ]
).astype(np.int8, copy=False)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI cites the paper’s behavior processing and says the requested output is binary no/yes, not cumulative lick count.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The lick labels share the same `[start:stop)` frame indices as the neural data, so they are aligned one sample per neural frame.

ii.
```python
trial_lick = lick[start:stop]
...
neural = activity[start:stop].T.copy()
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says lick is already on the common imaging grid and should be sliced on that same grid.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the session `identifier` string, parsed into reward-zone labels A/B/C with trial 30 as the switch point in switching sessions.

ii.
```python
scene = decode_text(nwb["identifier"]).rsplit("/", 1)[-1]
...
labels = scene_zone_labels(scene, starts.size)
```

iii. In `CONVERSION_NOTES.md` Steps 2, 4, and 5, the AI says session identifiers explicitly encode the reward-zone protocol and that all switch sessions in the provided data change after 30 trials.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI parses one label for fixed sessions or two labels for switch sessions, assigns the first label to the first 30 trials and the second label to later trials, maps A/B/C to `0/1/2`, and broadcasts that class across all timepoints in the trial.

ii.
```python
def scene_zone_labels(scene: str, n_trials: int) -> np.ndarray:
    labels = re.findall(r"(?:Location)?([ABC])", scene)
    if len(labels) == 1:
        return np.full(n_trials, labels[0], dtype="<U1")
    if len(labels) == 2:
        ...
        out = np.full(n_trials, labels[1], dtype="<U1")
        out[:30] = labels[0]
        return out
```

```python
np.full(T, ZONE_TO_CLASS[label], dtype=np.int8)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says this follows the code/paper reward-zone semantics while using the cleaner session metadata instead of inferring labels from noisy `reward_zone` samples.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from sparse reward delivery timestamps (`Reward/timestamps`) together with dense `reward_zone/data` evidence inside each trial.

ii.
```python
reward_zone_signal = behavior["reward_zone/data"][:]
reward_timestamps = behavior["Reward/timestamps"][:]
...
outcomes = reward_outcomes(
    timestamps, reward_timestamps, reward_zone_signal, starts, stops
)
```

iii. In `CONVERSION_NOTES.md` Steps 4 and 5, the AI says it wanted to match the paper’s trial-outcome semantics: reward delivery is only considered a rewarded trial if there is also reward-zone evidence during that interval.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the AI marks the outcome as `1` if any reward timestamp falls in `[trial start, teleport)` and the trial has any positive reward-zone signal; otherwise it marks `0`. That per-trial value is then broadcast across all timepoints in the trial.

ii.
```python
def reward_outcomes(...):
    ...
    delivered = np.any(
        (reward_timestamps >= timestamps[start])
        & (reward_timestamps < timestamps[stop])
    )
    zone_evidence = np.any(reward_zone_signal[start:stop] > 0)
    out[i] = int(delivered and zone_evidence)
```

```python
np.full(T, outcomes[raw_trial], dtype=np.int8)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI explicitly cites `get_trial_types` as the intended reference logic for current-trial reward outcome.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly uses fail-fast validation rather than repair. It checks that all dense behavioral streams have the same length, that timestamps are uniform, that environment and trial number are single-valued within each trial, and that scanning is always `1` inside retained trials. It also trims deconvolved planes to the behavior-aligned length by reading only `[:n_behavior]`. It does not implement special handling for missing reward-zone labels because it does not infer labels from the noisy `reward_zone` stream.

ii.
```python
if set(dense_lengths.values()) != {n_behavior}:
    raise ValueError(f"Dense behavior length mismatch in {path}: {dense_lengths}")
if not np.allclose(np.diff(timestamps), DT_SECONDS, rtol=0, atol=1e-10):
    raise ValueError(f"Nonuniform or unexpected timestamps in {path}")
```

```python
if not np.all(scanning[start:stop] == 1):
    raise ValueError(f"Non-scanning sample inside {session_tag} trial {raw_trial}")
if env_values.size != 1 or env_values[0] not in (0, 1):
    raise ValueError(f"Invalid environment in {session_tag} trial {raw_trial}: {env_values}")
if trial_values.size != 1:
    raise ValueError(f"Multiple trial IDs in {session_tag} trial {raw_trial}: {trial_values}")
```

```python
chunks.append(group["data"][:n_behavior, local].astype(np.float32, copy=False))
```

iii. In `CONVERSION_NOTES.md` Steps 4, 5, and 10, the AI says the supplied NWBs were internally consistent, so it preferred assertions and integrity checks over interpolation or gap-filling.

## 13-a. What are the most time-consuming steps of the code?

i. The AI says the most expensive step is recreating paper-style dF/F from raw fluorescence and neuropil only to compute the interneuron mask. It also notes that materializing and serializing the final variable-length trial lists is expensive.

ii.
```python
def compute_interneuron_mask(
    nwb: h5py.File,
    iscell: np.ndarray,
    speed: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    block_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    ...
    for plane_name in sorted(fluorescence):
        ...
        for block_start in range(0, local_cells.size, block_size):
            ...
            for start, stop in qc_segments:
                ...
```

```python
with args.outpicklefile.open("wb") as stream:
    pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. In `CONVERSION_NOTES.md` Step 6 and Step 9, the AI explicitly calls out the exact-paper interneuron filter and full-pickle serialization as the main runtime costs.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized cell-wise QC in blocks, but the remaining obvious loops are the per-trial conversion loop in `convert_session(...)` and the nested block/trial loops inside `compute_interneuron_mask(...)`.

ii.
```python
for raw_trial, (start, stop) in enumerate(zip(starts, stops)):
    ...
    decoder_input = np.vstack(...)
    decoder_output = np.vstack(...)
    neural = activity[start:stop].T.copy()
```

```python
for block_start in range(0, local_cells.size, block_size):
    ...
    for start, stop in qc_segments:
        ...
```

iii. In `CONVERSION_NOTES.md` Step 6, the AI says the exact paper QC is expensive and that it reduced the cost with blockwise sufficient statistics, implying that the remaining loops are largely a consequence of per-trial windows and variable-length outputs.

## 13-c. What processing does the code repeat multiple times?

i. The code reads raw fluorescence and neuropil to reconstruct dF/F for interneuron QC, and then separately reads the already deconvolved event data for the actual exported neural signal. It also revisits the same trial bounds for multiple variables during session conversion.

ii.
```python
fluorescence = nwb[f"{OPHYS_ROOT}/Fluorescence"]
neuropil = nwb[f"{OPHYS_ROOT}/Neuropil"]
...
deconvolved = nwb[f"{OPHYS_ROOT}/Deconvolved"]
```

```python
for raw_trial, (start, stop) in enumerate(zip(starts, stops)):
    ...
    trial_position = position[start:stop]
    trial_speed = speed[start:stop]
    trial_lick = lick[start:stop]
    ...
    neural = activity[start:stop].T.copy()
```

iii. In `CONVERSION_NOTES.md` Step 6, the AI says the exact paper interneuron filter forces it to touch raw fluorescence/neuropil even though the final output exports the already deconvolved events.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI recomputes a paper-style dF/F trace only to decide which cells to keep, then discards that dF/F entirely. It also computes and stores speed-correlation summaries for QC/metadata rather than for decoder inputs/outputs.

ii.
```python
corrected = f - 0.7 * f_neu + 0.7 * f_neu.mean(axis=1, keepdims=True)
baseline = ndimage.gaussian_filter1d(corrected, 15, axis=1)
baseline = ndimage.minimum_filter1d(baseline, 300, axis=1)
baseline = ndimage.maximum_filter1d(baseline, 300, axis=1)
with np.errstate(divide="ignore", invalid="ignore"):
    dff = (corrected - baseline) / np.abs(baseline)
dff = ndimage.gaussian_filter1d(dff, 2, axis=1)
```

```python
"max_dff_speed_correlation": float(np.nanmax(speed_correlations)),
```

iii. In `CONVERSION_NOTES.md` Step 6, the AI explicitly says raw fluorescence/neuropil are used only for interneuron QC and that the exported dataset retains the deconvolved events instead.
