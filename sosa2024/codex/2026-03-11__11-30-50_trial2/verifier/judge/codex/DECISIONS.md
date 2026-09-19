# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent scans `data/sub-*` for every `*.nwb` session file, opens each file directly with `h5py`, and reads the needed behavior and ophys datasets into a `SessionArrays` dataclass. It does not use `pynwb`.

ii.
```python
def list_session_files() -> list[str]:
    files = sorted(glob.glob(os.path.join("data", "sub-*", "sub-*_behavior+ophys.nwb")))
    if not files:
        raise FileNotFoundError("No NWB files found under data/sub-*/")
    return files
```

```python
def load_session_arrays(path: str) -> SessionArrays:
    with h5py.File(path, "r") as f:
        beh = f["processing/behavior/BehavioralTimeSeries"]
        neural_group = f["processing/ophys/Deconvolved/plane0"]
        ...
```

iii. In `CONVERSION_NOTES.md`, the agent says it used all 152 NWB sessions from the 11 switch-task mice and used direct `h5py` reads as a speedup over full NWB object materialization.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified session-by-session from `general/subject/subject_id`. A unique-subject list is built in the order sessions are processed, and `subject_idx` records the per-session subject index.

ii.
```python
subject = decode_if_bytes(f["general/subject/subject_id"][()])
...
if arrays.subject not in subject_to_idx:
    subject_to_idx[arrays.subject] = len(all_subjects)
    all_subjects.append(arrays.subject)
...
data["subject_idx"].append(subject_to_idx[arrays.subject])
```

iii. The notes say the archive contains one NWB file per subject-session under `data/sub-<mouse>/...`, and that the shared archive corresponds to the 11-mouse switch cohort.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The script reads `general/session_id` and constructs a session label such as `m11_ses-08`.

ii.
```python
session_id = decode_if_bytes(f["general/session_id"][()])
session_label = f"{subject}_ses-{session_id}"
...
for session_idx, path in enumerate(files):
    arrays = load_session_arrays(path)
```

iii. The notes explicitly state that `data/` contains one NWB file per subject-session and report 152 such files.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from the framewise `trial_start` and `teleport` behavior signals. Each positive `trial_start` index is paired with the corresponding positive `teleport` index, and the trial slice is `[start, stop + 1)`.

ii.
```python
def build_trial_slices(trial_start_signal: np.ndarray, teleport_signal: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(trial_start_signal > 0.5)
    teleports = np.flatnonzero(teleport_signal > 0.5)
    ...
    for start, stop in zip(starts[:n], teleports[:n]):
        if stop < start:
            continue
        slices.append((int(start), int(stop) + 1))
```

iii. In the notes, the agent says this matches the paper/reference code better than relying on `trial number` changes alone.

## 1-e. How are trials filtered based on quality controls?

i. The agent does not remove short trials. Instead, it drops any trial whose lick trace looks artifactual: if more than 35% of frames have `lick > 2`, the trial is discarded. It also errors out if fewer than two valid trials remain.

ii.
```python
LICK_ARTIFACT_FRACTION = 0.35
...
lick_trial = arrays.lick[start:stop_exclusive]
if lick_trial.size and np.mean(lick_trial > 2) > LICK_ARTIFACT_FRACTION:
    lick_artifact[trial_id] = True
...
if lick_artifact[trial_id]:
    continue
...
if len(neural_trials) < 2:
    raise RuntimeError(f"{arrays.session_label}: fewer than 2 valid trials after filtering.")
```

iii. The notes justify this as matching the paper/reference licking QC in `glmUtils.get_timeseries_data`, and the trajectory shows the agent explicitly monitored aggregate trial removal against the paper’s reported lick-QC rate.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The agent derives `neural` from the stored NWB dataset `processing/ophys/Deconvolved/plane0/data`, plus the ROI region `Deconvolved/plane0/rois` and the segmentation-table `iscell` flag for curation.

ii.
```python
neural_group = f["processing/ophys/Deconvolved/plane0"]
seg = f["processing/ophys/ImageSegmentation/PlaneSegmentation"]
...
neural = neural_group["data"][()].astype(np.float32, copy=False)
roi_ids = neural_group["rois"][()].astype(np.int64, copy=False)
iscell = seg["iscell"][()][roi_ids, 0] > 0.5
neural = neural[:, iscell]
```

iii. The notes say the agent chose NWB `Deconvolved` as the paper-equivalent neural stream and treated the archive as already exposing the deconvolved activity needed for decoding.

## 2-b. How is the `neural` data processed?

i. Processing is minimal: crop neural and behavior to the common time range, keep curated ROIs, optionally rebin 31 Hz sessions down to a common 15.5078125 Hz grid by summing adjacent neural bins, transpose to `(neurons, time)`, and split into trials. The script does not recompute dF/F or deconvolution.

ii.
```python
t_neural = neural.shape[0]
t_behavior = beh["position/data"].shape[0]
t_common = min(t_neural, t_behavior)
...
neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)
...
if factor == 2:
    neural_trial = rebin_2d_sum_time_last(neural_trial, factor)
```

iii. The notes justify this as using the deconvolved stream already present in the archive and standardizing all sessions to a common decoder timebase.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only implemented neural QC is ROI-region-aware `iscell` filtering. The script does not implement the paper’s additional speed-correlation putative-interneuron exclusion.

ii.
```python
roi_ids = neural_group["rois"][()].astype(np.int64, copy=False)
iscell = seg["iscell"][()][roi_ids, 0] > 0.5
neural = neural[:, iscell]
roi_ids = roi_ids[iscell]
```

iii. The notes say this avoids overcounting ROIs in multipane sessions and acknowledges that the exact dF/F-speed-correlation interneuron filter was not re-run.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to `trial_start`. The script uses `trial_start`/`teleport` to form each trial slice and then takes the same slice from the neural matrix.

ii.
```python
trial_slices = build_trial_slices(arrays.trial_start, arrays.teleport)
...
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)
```

iii. The notes explicitly say temporal alignment should use `trial_start_inds` as the start and `teleport_inds` as the end of each trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a common time bin of `1 / 15.5078125` s (`~64.483 ms`). Sessions already at that rate are left alone; sessions at `31.015625 Hz` are rebinned by a factor of 2.

ii.
```python
TARGET_RATE_HZ = 15.5078125
TARGET_DT_S = 1.0 / TARGET_RATE_HZ
...
if math.isclose(arrays.rate_hz, TARGET_RATE_HZ):
    factor = 1
elif math.isclose(arrays.rate_hz, 2.0 * TARGET_RATE_HZ):
    factor = 2
...
neural_trial = rebin_2d_sum_time_last(neural_trial, factor)
```

iii. The notes justify this as enforcing a common output bin size and matching the paper’s effective per-plane rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is not derived from a raw timestamp variable. Instead, it is reconstructed from the number of output bins in the trial and the fixed target time step `TARGET_DT_S`.

ii.
```python
TARGET_DT_S = 1.0 / TARGET_RATE_HZ
...
t_bins = neural_trial.shape[1]
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
```

iii. The notes say the agent aligned all trials to the first frame of the trial and standardized sessions to a common `15.5078125 Hz` bin size.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the script creates `0, dt, 2*dt, ...` using the rebinned trial length and fixed `dt = 1 / 15.5078125`.

ii.
```python
t_bins = neural_trial.shape[1]
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
```

iii. The justification in the notes is that all sessions were placed on one common sampling grid, so elapsed trial time could be reconstructed from bin count alone.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by construction: after neural slicing and any rebinning, the script creates exactly one time value per neural bin.

ii.
```python
if factor == 2:
    neural_trial = rebin_2d_sum_time_last(neural_trial, factor)
...
t_bins = neural_trial.shape[1]
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
input_trial = np.stack(
    [time_from_start, env_series, trial_series, prev_reward_series],
    axis=0,
)
```

iii. The notes describe the pipeline as frame-aligned and say all trials are aligned to the first frame of the trial.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/environment/data`.

ii.
```python
environment=beh["environment/data"][()][:t_common].astype(np.float32, copy=False),
...
env = arrays.environment[start:stop_exclusive]
```

iii. The notes map this field directly to environment type and state that valid values are `0` and `1`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial, the script removes negative sentinel values, takes the modal valid environment value, and repeats that constant value across all bins in the trial.

ii.
```python
env = arrays.environment[start:stop_exclusive]
env = env[env >= 0]
trial_env[trial_id] = mode_int(env, default=0)
...
env_series = np.full((t_bins,), float(trial_env[trial_id]), dtype=np.float32)
```

iii. The notes justify this as a per-trial decoder input and note that the NWB environment variable behaves as a binary trial-state variable.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the reconstructed trial order itself. The loop index over `trial_slices` becomes the within-session trial number.

ii.
```python
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    ...
    trial_series = np.full((t_bins,), float(trial_id), dtype=np.float32)
```

iii. The notes say this keeps the within-session trial number 0-indexed and matches the code convention used elsewhere.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No additional transformation beyond assigning the sequential trial index and repeating it across all bins of that trial.

ii.
```python
trial_series = np.full((t_bins,), float(trial_id), dtype=np.float32)
```

iii. The notes explicitly say to keep the within-session trial number 0-indexed.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from reward events in `Reward/timestamps`, combined with `trial number/data` to assign each reward event to a trial.

ii.
```python
reward_times=beh["Reward/timestamps"][()].astype(np.float64, copy=False),
trial_number=beh["trial number/data"][()][:t_common].astype(np.float32, copy=False),
...
reward_frame_idx = reward_frames_for_session(arrays)
rewarded_trial_ids = set(int(t) for t in arrays.trial_number[reward_frame_idx] if t >= 0)
```

iii. The notes describe this as “trial reward outcome shifted by one trial” and justify it as the user-requested previous-outcome decoder input.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. First, the script marks which trials were rewarded. Then it shifts that trial-level array by one trial, prepending `0` for the first trial, and repeats the previous-trial reward label across the current trial’s bins.

ii.
```python
trial_rewarded = np.array(
    [1 if trial_id in rewarded_trial_ids else 0 for trial_id in range(len(trial_slices))],
    dtype=np.int64,
)
trial_prev_rewarded = np.concatenate([[0], trial_rewarded[:-1]]).astype(np.float32, copy=False)
...
prev_reward_series = np.full((t_bins,), float(trial_prev_rewarded[trial_id]), dtype=np.float32)
```

iii. The notes say the first trial gets `0` and later trials receive the previous trial’s rewarded-vs-omitted label.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from position plus an inferred active reward-zone label. That zone label is inferred from `reward_zone/data`, `position/data`, reward-event timestamps, and framewise `trial number/data`; `environment` is also used when filling missing labels across blocks.

ii.
```python
trial_zone, observed_zone_position = infer_trial_zones(
    positions=np.clip(arrays.position, 0.0, TRACK_LENGTH_CM),
    reward_zone_signal=arrays.reward_zone_signal,
    trial_slices=trial_slices,
    trial_rewarded=trial_rewarded,
    reward_frame_idx=reward_frame_idx,
    trial_by_frame=arrays.trial_number.astype(np.int64, copy=False),
    trial_env=trial_env,
)
...
dist = compute_distance_to_zone(position, zone_idx)
```

iii. The notes say the active reward zone should be inferred from reward-zone-active positions and reward events, then filled on omission trials within stable task blocks.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each trial, once the active zone is known, the script computes signed distance to the nearest reward-zone edge: negative before the zone, `0` inside the zone, positive after the zone.

ii.
```python
def compute_distance_to_zone(position_cm: np.ndarray, zone_idx: int) -> np.ndarray:
    zone_start, zone_end = ZONE_BOUNDS[zone_idx]
    distance = np.zeros(position_cm.shape, dtype=np.float32)
    before = position_cm < zone_start
    after = position_cm > zone_end
    distance[before] = position_cm[before] - zone_start
    distance[after] = position_cm[after] - zone_end
    return distance
```

iii. The notes map this directly to “signed distance to nearest point in active reward zone.”

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous distance is discretized into 7 categories matching the task bins.

ii.
```python
def discretize_distance(distance_cm: np.ndarray) -> np.ndarray:
    out = np.zeros(distance_cm.shape, dtype=np.int64)
    out[distance_cm < -50.0] = 0
    out[(distance_cm >= -50.0) & (distance_cm < -10.0)] = 1
    out[(distance_cm >= -10.0) & (distance_cm < 0.0)] = 2
    out[distance_cm == 0.0] = 3
    out[(distance_cm > 0.0) & (distance_cm <= 10.0)] = 4
    out[(distance_cm > 10.0) & (distance_cm <= 50.0)] = 5
    out[distance_cm > 50.0] = 6
    return out
```

iii. The notes explicitly say this output is discretized to the 7 task-specified bins.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by taking the same per-trial slices as neural data and, when needed, rebinned behavior arrays before computing distance bins.

ii.
```python
neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
...
if factor == 2:
    position = rebin_1d_mean(position, factor)
...
dist = compute_distance_to_zone(position, zone_idx)
dist_bin = discretize_distance(dist)
```

iii. The notes repeatedly describe the conversion as frame-aligned and trial-aligned to `trial_start`.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/position/data`.

ii.
```python
position=beh["position/data"][()][:t_common].astype(np.float32, copy=False),
...
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
```

iii. The notes map this directly from the behavior position trace on the 450 cm track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The script clips position to the valid track interval `[0, 450]` cm, optionally rebins by mean for 31 Hz sessions, and then discretizes.

ii.
```python
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
...
if factor == 2:
    position = rebin_1d_mean(position, factor)
...
pos_bin = discretize_position(position)
```

iii. The notes justify this as keeping converted data on the physical 450 cm corridor while harmonizing sessions to a common rate.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is divided into 5 equal-width bins across the 450 cm track by clipping to `<450` and using integer division by `90`.

ii.
```python
def discretize_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, 0.0, TRACK_LENGTH_CM - 1e-6)
    return np.minimum((clipped / (TRACK_LENGTH_CM / 5.0)).astype(np.int64), 4)
```

iii. The notes say this matches the decoder requirement of five equal bins rather than the paper’s 45 spatial bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The same trial slices and rebinning factor used for neural data are applied to position before discretization.

ii.
```python
neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
if factor == 2:
    position = rebin_1d_mean(position, factor)
```

iii. The notes say the pipeline is based on frame-aligned neural and behavior streams from the NWB export.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/lick/data`.

ii.
```python
lick=beh["lick/data"][()][:t_common].astype(np.float32, copy=False),
...
lick_binary = (arrays.lick[start:stop_exclusive] > 0).astype(np.int64, copy=False)
```

iii. The notes map lick directly from the behavior timeseries and note that rebinned 31 Hz sessions use logical OR.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The script binarizes lick as `lick > 0`. For rebinned 31 Hz sessions, it performs a logical OR across each pair of frames.

ii.
```python
lick_binary = (arrays.lick[start:stop_exclusive] > 0).astype(np.int64, copy=False)
...
if factor == 2:
    lick_binary = rebin_1d_any(lick_binary, factor)
```

iii. The notes justify this as the appropriate binary decoder output and say it follows the paper’s lick cleanup logic.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned by using the same trial slices as the neural trace and, when needed, the same factor-2 rebinning.

ii.
```python
lick_binary = (arrays.lick[start:stop_exclusive] > 0).astype(np.int64, copy=False)
if factor == 2:
    lick_binary = rebin_1d_any(lick_binary, factor)
...
output_trial = np.stack(
    [dist_bin, pos_bin, speed_bin, lick_binary, zone_bin, reward_bin],
    axis=0,
)
```

iii. The notes describe the behavior streams as already frame-aligned to imaging.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is inferred from `reward_zone/data` and `position/data`, with reward-event timestamps and framewise `trial number` used as fallbacks when `reward_zone` is absent on a trial.

ii.
```python
def infer_trial_zones(
    positions: np.ndarray,
    reward_zone_signal: np.ndarray,
    trial_slices: list[tuple[int, int]],
    trial_rewarded: np.ndarray,
    reward_frame_idx: np.ndarray,
    trial_by_frame: np.ndarray,
    trial_env: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
```

```python
mask = reward_zone_signal[start:stop_exclusive] > 0
if np.any(mask):
    zone_pos = float(np.nanmedian(positions[start:stop_exclusive][mask]))
elif trial_id in reward_pos_by_trial:
    zone_pos = reward_pos_by_trial[trial_id]
```

iii. The notes say reward-zone labels should be inferred from reward-zone-active positions and reward events, especially on omission trials.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The script converts the observed reward-zone position in each trial to the nearest of the three fixed zone centers. Missing labels are filled heuristically: if only one zone is observed in the session, all missing trials get that zone; if two zones are observed, missing labels are filled separately before and after the hard-coded switch trial 30, with optional overwrite by pre/post block when multiple environments appear.

ii.
```python
if np.isfinite(zone_pos):
    observed_position[trial_id] = zone_pos
    observed[trial_id] = int(np.argmin(np.abs(ZONE_CENTERS - zone_pos)))
...
split = min(SWITCH_TRIAL_INDEX, n_trials)
pre_majority = trial_majority_zone(observed, 0, split, fallback=int(unique_observed[0]))
post_majority = trial_majority_zone(observed, split, n_trials, fallback=post_fallback)
filled[:split][filled[:split] < 0] = pre_majority
filled[split:][filled[split:] < 0] = post_majority
```

iii. The notes justify this with the task structure: reward switches occur after 30 trials and omission trials can inherit the active zone from a stable block.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from reward-event timestamps in `Reward/timestamps`, mapped to trials through the framewise `trial number` series.

ii.
```python
reward_times=beh["Reward/timestamps"][()].astype(np.float64, copy=False),
trial_number=beh["trial number/data"][()][:t_common].astype(np.float32, copy=False),
...
rewarded_trial_ids = set(int(t) for t in arrays.trial_number[reward_frame_idx] if t >= 0)
```

iii. The notes define reward outcome as trial-level rewarded vs omitted and use reward events to determine it.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are first mapped to frame indices with `searchsorted`; the corresponding `trial number` labels define a set of rewarded trials. Every bin in a rewarded trial gets `1`, otherwise `0`.

ii.
```python
def reward_frames_for_session(arrays: SessionArrays) -> np.ndarray:
    if arrays.reward_times.size == 0:
        return np.empty((0,), dtype=np.int64)
    idx = np.searchsorted(arrays.timestamps, arrays.reward_times, side="left")
    return np.clip(idx, 0, arrays.timestamps.shape[0] - 1).astype(np.int64, copy=False)
```

```python
trial_rewarded = np.array(
    [1 if trial_id in rewarded_trial_ids else 0 for trial_id in range(len(trial_slices))],
    dtype=np.int64,
)
reward_bin = np.full((t_bins,), int(trial_rewarded[trial_id]), dtype=np.int64)
```

iii. The notes say this output is “rewarded if any `Reward` event timestamp falls within the trial.”

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The script crops neural and behavior to the common minimum length, removes negative sentinel values when summarizing environment, clips reward timestamps to the valid frame range, fills missing reward-zone labels within stable session blocks, and raises runtime errors when key structures are missing or inconsistent. It also drops trials with lick artifacts and aborts sessions with fewer than two remaining trials.

ii.
```python
t_common = min(t_neural, t_behavior)
...
env = env[env >= 0]
...
idx = np.searchsorted(arrays.timestamps, arrays.reward_times, side="left")
return np.clip(idx, 0, arrays.timestamps.shape[0] - 1).astype(np.int64, copy=False)
...
if unique_observed.size == 0:
    raise RuntimeError("Could not infer any reward-zone labels from the session.")
...
if len(neural_trials) < 2:
    raise RuntimeError(f"{arrays.session_label}: fewer than 2 valid trials after filtering.")
```

iii. The notes say 10 sessions had a one-sample neural/behavior mismatch and justify cropping to the common minimum; they also describe omission-trial reward-zone filling as required by the task structure.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive parts are full-session HDF5 reads, per-session materialization of the large neural response matrices, per-trial conversion loops, and optional plotting. The agent also tracked per-session timing and total runtime.

ii.
```python
with h5py.File(path, "r") as f:
    ...
    neural = neural_group["data"][()].astype(np.float32, copy=False)
```

```python
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    ...
```

```python
session_info = {
    ...
    "conversion_seconds": time.perf_counter() - t0,
}
```

iii. In the notes, the agent explicitly says full-session NWB reads dominate cost, and that plotting adds overhead and should stay off for full conversion.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious vectorization targets are the per-trial loop in `convert_session`, the reward-frame loop in `infer_trial_zones`, and the repeated per-trial extraction of environment and lick QC. The current code uses Python loops for all of these.

ii.
```python
for frame_idx in reward_frame_idx:
    ...
```

```python
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    env = arrays.environment[start:stop_exclusive]
    ...
    lick_trial = arrays.lick[start:stop_exclusive]
```

iii. The notes only mention “vectorized trial construction” in a broad sense; they do not claim these loops were fully optimized.

## 13-c. What processing does the code repeat multiple times?

i. The code repeats slice-and-rebin logic for behavior streams across neural conversion and plotting, and it repeatedly scans per-trial slices for environment, lick QC, and reward-zone inference. Across separate runs, sample mode and full mode also reread sessions from disk.

ii.
```python
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    ...
    position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
    speed = np.clip(arrays.speed[start:stop_exclusive], 0.0, None)
    lick_binary = (arrays.lick[start:stop_exclusive] > 0).astype(np.int64, copy=False)
```

```python
example_position=np.clip(arrays.position[trial_slices[example_idx][0] : trial_slices[example_idx][1]], 0.0, TRACK_LENGTH_CM)
...
example_speed=np.clip(arrays.speed[trial_slices[example_idx][0] : trial_slices[example_idx][1]], 0.0, None)
...
example_lick=(arrays.lick[trial_slices[example_idx][0] : trial_slices[example_idx][1]] > 0).astype(np.int64)
```

iii. The notes justify plotting as debug-only overhead and emphasize keeping it disabled for full conversion; they do not otherwise present repeated processing as a deliberate design choice.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes and stores debug/diagnostic information that the downstream decoder does not need: plotting summaries, `observed_zone_positions_cm`, conversion timing, raw/kept/removed trial counts, and original-rate metadata. In `--show-processing` mode it also recomputes rebinned example traces solely for visualization.

ii.
```python
session_info = {
    "original_rate_hz": arrays.rate_hz,
    "rebin_factor": factor,
    "n_trials_raw": len(trial_slices),
    "n_trials_kept": len(neural_trials),
    "n_trials_lick_artifact_removed": int(lick_artifact.sum()),
    "observed_zone_positions_cm": observed_zone_position.tolist(),
    "conversion_seconds": time.perf_counter() - t0,
}
```

```python
if show_processing:
    ...
    plot_processing_summary(...)
```

iii. The notes explicitly characterize the plots as inspection artifacts and overhead for review rather than part of the decoder-ready dataset itself.
