# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all NWB files by globbing `data/sub-*/sub-*_behavior+ophys.nwb`, sorting that list, and iterating once per file. Each file is opened directly with `h5py`, and the needed behavior and neural arrays are copied into a `SessionArrays` record before trial conversion.

ii.
```python
def list_session_files() -> list[str]:
    files = sorted(glob.glob(os.path.join("data", "sub-*", "sub-*_behavior+ophys.nwb")))
    if not files:
        raise FileNotFoundError("No NWB files found under data/sub-*/")
    return files

...

for session_idx, path in enumerate(files):
    arrays = load_session_arrays(path)
```

```python
def load_session_arrays(path: str) -> SessionArrays:
    with h5py.File(path, "r") as f:
        ...
        return SessionArrays(
            ...
            position=beh["position/data"][()][:t_common].astype(np.float32, copy=False),
            speed=beh["speed/data"][()][:t_common].astype(np.float32, copy=False),
            ...
            neural=neural[:t_common],
        )
```

iii. In `CONVERSION_NOTES.md`, the agent says it will "Use all 152 NWB sessions from the 11 switch-task mice" and that direct `h5py` reads were chosen as a speedup over materializing the full NWB object graph.

## 1-b. How are the data split into subjects?

i. Subjects are taken from each file’s NWB metadata field `general/subject/subject_id`, not from directory names. A `subject_to_idx` map is built as sessions are processed, and unique subjects are appended in first-seen order.

ii.
```python
subject = decode_if_bytes(f["general/subject/subject_id"][()])

...

if arrays.subject not in subject_to_idx:
    subject_to_idx[arrays.subject] = len(all_subjects)
    all_subjects.append(arrays.subject)

...

data["subjects"] = all_subjects
data["subject_idx"].append(subject_to_idx[arrays.subject])
```

iii. The notes say the archive corresponds to the 11-mouse switch cohort and that all sessions are processed; using the NWB metadata was the agent’s way to avoid relying on path parsing.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The session label is read from NWB metadata and combined with the subject id for metadata/reporting.

ii.
```python
for session_idx, path in enumerate(files):
    arrays = load_session_arrays(path)
    ...
    data["neural"].append(neural_trials)
    data["input"].append(input_trials)
    data["output"].append(output_trials)
```

```python
session_id = decode_if_bytes(f["general/session_id"][()])
session_label = f"{subject}_ses-{session_id}"
```

iii. The notes explicitly say "Use all 152 NWB sessions" and treat one subject-session NWB file as one converted session.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from framewise `trial_start` and `teleport` signals. Every frame where `trial_start > 0.5` is taken as a start, every frame where `teleport > 0.5` is taken as an end marker, and each trial slice is `[start, stop + 1)`.

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

```python
trial_slices = build_trial_slices(arrays.trial_start, arrays.teleport)
```

iii. The notes say this matches the reference trial definition better than relying on stored trial-number changes and cite `trial_start` plus `teleport` as the paper-consistent boundary representation.

## 1-e. How are trials filtered based on quality controls?

i. The agent does not remove short trials. Instead, it drops trials flagged as lick-artifact trials and then requires at least two valid trials to keep a session.

ii.
```python
LICK_ARTIFACT_FRACTION = 0.35

...

lick_trial = arrays.lick[start:stop_exclusive]
if lick_trial.size and np.mean(lick_trial > 2) > LICK_ARTIFACT_FRACTION:
    lick_artifact[trial_id] = True

...

for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    if lick_artifact[trial_id]:
        continue

...

if len(neural_trials) < 2:
    raise RuntimeError(f"{arrays.session_label}: fewer than 2 valid trials after filtering.")
```

iii. The notes justify this with the paper’s licking QC, citing `glmUtils.get_timeseries_data` and reporting `69 / 12,216` removed trials as close to the paper’s `~0.65%` lick-artifact removal rate.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `processing/ophys/Deconvolved/plane0/data`, with ROI inclusion determined by `processing/ophys/Deconvolved/plane0/rois` and `processing/ophys/ImageSegmentation/PlaneSegmentation/iscell`.

ii.
```python
neural_group = f["processing/ophys/Deconvolved/plane0"]
seg = f["processing/ophys/ImageSegmentation/PlaneSegmentation"]

neural = neural_group["data"][()].astype(np.float32, copy=False)
roi_ids = neural_group["rois"][()].astype(np.int64, copy=False)
iscell = seg["iscell"][()][roi_ids, 0] > 0.5
neural = neural[:, iscell]
```

iii. The notes say the decoder-relevant stream should be the NWB `Deconvolved` signal because the paper decodes deconvolved activity (`events`), and they document the archive limitation that only the provided response matrix is available directly.

## 2-b. How is the `neural` data processed?

i. The agent crops neural/behavior streams to a common frame count, filters neurons by ROI-aware `iscell`, splits by trials, transposes each trial to `(neurons, time)`, and rebins 31 Hz sessions to a common 15.5078125 Hz output rate by summing adjacent neural bins.

ii.
```python
t_neural = neural.shape[0]
t_behavior = beh["position/data"].shape[0]
t_common = min(t_neural, t_behavior)

...

neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)

if factor == 2:
    neural_trial = rebin_2d_sum_time_last(neural_trial, factor)
```

```python
def rebin_2d_sum_time_last(x: np.ndarray, factor: int) -> np.ndarray:
    ...
    full = x[:, : n_full * factor].reshape(x.shape[0], n_full, factor).sum(axis=2)
```

iii. The notes say the script should "Standardize all sessions to a common `15.5078125 Hz` bin size" and that using the archive’s deconvolved stream is the closest paper-equivalent signal available.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered only with the ROI-referenced `iscell[:, 0] > 0.5` mask. The notes discuss the paper’s additional dF/F-speed-correlation interneuron exclusion, but the script does not implement that filter.

ii.
```python
roi_ids = neural_group["rois"][()].astype(np.int64, copy=False)
iscell = seg["iscell"][()][roi_ids, 0] > 0.5
neural = neural[:, iscell]
roi_ids = roi_ids[iscell]
```

iii. The notes say this ROI-region-aware `iscell` filtering is "critical for multipane sessions" and later explicitly document that the exact speed-correlation interneuron filter was not re-run from the NWB export.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start by slicing each trial from the reconstructed `trial_start`/`teleport` boundaries. No extra offset is applied; the metadata also records `temporal_alignment_event = "trial_start"` and `off_start = 0.0`.

ii.
```python
trial_slices = build_trial_slices(arrays.trial_start, arrays.teleport)

...

neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)
```

```python
"metadata": {
    ...
    "temporal_alignment_event": "trial_start",
    "off_start": 0.0,
```

iii. The notes repeatedly say trial alignment should use `trial_start` as the start and `teleport` as the end, matching the task instruction to align to start of trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted output always uses a common time bin size of `1 / 15.5078125` s (`~64.48 ms`). Sessions already at that rate are kept as-is; sessions at `31.015625 Hz` are rebinned by a factor of 2.

ii.
```python
TARGET_RATE_HZ = 15.5078125
TARGET_DT_S = 1.0 / TARGET_RATE_HZ

...

if math.isclose(arrays.rate_hz, TARGET_RATE_HZ):
    factor = 1
elif math.isclose(arrays.rate_hz, 2.0 * TARGET_RATE_HZ):
    factor = 2
```

```python
"time_bin_size": 1000.0 * TARGET_DT_S,
```

iii. The notes justify this as a "common output bin size" so all sessions share one decoder timebase and say this matches the paper’s effective per-plane sampling rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. In the final script, this input is not taken from a raw timestamp variable at trial level. It is synthesized from the output bin index and the fixed target bin size after trial slicing/rebinning.

ii.
```python
t_bins = neural_trial.shape[1]
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
```

iii. The notes describe this as "Reconstructed trial-relative elapsed time" and justify it by the decision to standardize everything to a common 15.5078125 Hz timebase.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the agent creates `0, dt, 2*dt, ...` where `dt = 1 / 15.5078125` s and the number of bins equals the rebinned neural trial length.

ii.
```python
t_bins = neural_trial.shape[1]
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
```

iii. The notes say all trials should be aligned to the first frame of the trial on the common output timebase.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by construction: the number of generated time bins is set from `neural_trial.shape[1]`, after any trial slicing and factor-2 rebinning, so the time vector has exactly one entry per neural bin.

ii.
```python
neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)
...
t_bins = neural_trial.shape[1]
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
```

iii. The notes frame this as trial-aligned elapsed time on a common neural/behavior output grid rather than as preserved raw behavioral timestamps.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/environment/data`.

ii.
```python
environment=beh["environment/data"][()][:t_common].astype(np.float32, copy=False),

...

env = arrays.environment[start:stop_exclusive]
```

iii. The notes map this source directly to `input[1]` and say it corresponds to binary `ENV1` vs `ENV2`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The agent filters out invalid negative values, takes the modal environment value within each trial, and repeats that single trial-level label across all time bins in the trial.

ii.
```python
env = arrays.environment[start:stop_exclusive]
env = env[env >= 0]
trial_env[trial_id] = mode_int(env, default=0)

...

env_series = np.full((t_bins,), float(trial_env[trial_id]), dtype=np.float32)
```

iii. The notes justify this as a per-trial decoder input and explicitly say to "take modal valid environment value (`0` or `1`) and repeat across bins."

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from reconstructed trial boundaries and the loop index `trial_id`; it is not copied from the NWB `trial number` series for the output input field.

ii.
```python
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    ...
    trial_series = np.full((t_bins,), float(trial_id), dtype=np.float32)
```

iii. The notes say to keep the within-session trial number 0-indexed and describe it as the trial index after reconstructing trials from `trial_start` and `teleport`.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No further processing is applied beyond filling a constant vector with the within-session trial index for every time bin of that trial.

ii.
```python
trial_series = np.full((t_bins,), float(trial_id), dtype=np.float32)
```

iii. The notes explicitly say "Keep native 0-indexing" and repeat the per-trial label across bins.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `Reward/timestamps` plus the framewise NWB `trial number` stream, after first assigning reward events to frame indices.

ii.
```python
reward_times=beh["Reward/timestamps"][()].astype(np.float64, copy=False),
trial_number=beh["trial number/data"][()][:t_common].astype(np.float32, copy=False),
```

```python
reward_frame_idx = reward_frames_for_session(arrays)
rewarded_trial_ids = set(int(t) for t in arrays.trial_number[reward_frame_idx] if t >= 0)
```

iii. The notes say previous outcome should come from trial reward outcome shifted by one trial and cite reward events assigned to trials as the source.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. First, reward timestamps are mapped onto frame indices with `np.searchsorted`. Then each trial is labeled rewarded if its trial id appears among the reward frames. Finally that binary vector is shifted by one trial, with the first trial forced to 0, and repeated across bins.

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
trial_prev_rewarded = np.concatenate([[0], trial_rewarded[:-1]]).astype(np.float32, copy=False)
...
prev_reward_series = np.full((t_bins,), float(trial_prev_rewarded[trial_id]), dtype=np.float32)
```

iii. The notes say this is a user-requested decoder input, implemented as previous trial rewarded (`1`) vs omitted (`0`).

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from `position/data` plus an inferred per-trial reward-zone label. That label is inferred using `reward_zone/data`, reward timestamps, the framewise `trial number`, and trial-level environment information.

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
```

```python
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
zone_idx = int(trial_zone[trial_id])
dist = compute_distance_to_zone(position, zone_idx)
```

iii. The notes say the active reward zone should be inferred from reward-zone-active positions and reward events, then omission trials should be filled within stable blocks using switch timing as a sanity anchor.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Once the trial’s reward-zone category is known, the code computes signed distance to the nearest edge of that zone: negative before the zone, zero inside it, positive after it. The continuous distances are then discretized.

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

```python
dist = compute_distance_to_zone(position, zone_idx)
dist_bin = discretize_distance(dist)
```

iii. The notes explicitly describe the signed-distance rule `pos < start -> pos-start`, `inside -> 0`, `pos > end -> pos-end`, following the paper’s reward-zone geometry.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The agent uses seven hard-coded bins matching the task specification.

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

iii. The notes say the discretization should follow the decoder task bins exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by using the same per-trial slices as `neural`, and for 31 Hz sessions position is rebinned with the same factor used for neural data before distance is computed.

ii.
```python
neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)

if factor == 2:
    neural_trial = rebin_2d_sum_time_last(neural_trial, factor)
    position = rebin_1d_mean(position, factor)
```

iii. The notes justify this by the decision to standardize all sessions onto one common output timebase.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/position/data`.

ii.
```python
position=beh["position/data"][()][:t_common].astype(np.float32, copy=False),
...
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
```

iii. The notes map `position/data` directly to `output[1]`.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The agent clips positions to `[0, 450)` cm and then converts them into five equal-width bins by dividing the 450 cm corridor into 90 cm segments.

ii.
```python
def discretize_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, 0.0, TRACK_LENGTH_CM - 1e-6)
    return np.minimum((clipped / (TRACK_LENGTH_CM / 5.0)).astype(np.int64), 4)
```

```python
pos_bin = discretize_position(position)
```

iii. The notes justify this from the paper’s 450 cm track length and the task instruction to use 5 equal bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is thresholded into five bins of width 90 cm each across the clipped 0 to 450 cm corridor.

ii.
```python
clipped = np.clip(position_cm, 0.0, TRACK_LENGTH_CM - 1e-6)
return np.minimum((clipped / (TRACK_LENGTH_CM / 5.0)).astype(np.int64), 4)
```

iii. The notes say the decoder spec, not the paper’s original 45-bin analyses, should control this discretization.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position uses the same trial slices as neural data, and in rebinned sessions the agent averages pairs of position samples so the position vector stays one-to-one with neural bins.

ii.
```python
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)

if factor == 2:
    position = rebin_1d_mean(position, factor)
```

iii. The notes explicitly treat this as part of harmonizing all time-varying variables to the common neural decoder timebase.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/lick/data`.

ii.
```python
lick=beh["lick/data"][()][:t_common].astype(np.float32, copy=False),
...
lick_binary = (arrays.lick[start:stop_exclusive] > 0).astype(np.int64, copy=False)
```

iii. The notes map `lick/data` directly to binary lick output, with paper-style lick-artifact filtering applied at the trial level first.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The raw lick signal is binarized with `> 0`. For rebinned 31 Hz sessions, adjacent lick bins are combined with logical OR so any lick in the pair becomes 1.

ii.
```python
lick_binary = (arrays.lick[start:stop_exclusive] > 0).astype(np.int64, copy=False)

if factor == 2:
    lick_binary = rebin_1d_any(lick_binary, factor)
```

```python
def rebin_1d_any(x: np.ndarray, factor: int) -> np.ndarray:
    ...
    out.append(np.any(x[: n_full * factor].reshape(n_full, factor) > 0, axis=1))
```

iii. The notes say to "Convert to binary per bin (`lick > 0`)" and, for rebinned sessions, use logical OR within each 15.5 Hz bin.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned by the same trial slices as neural data, with identical factor-2 temporal rebinned alignment for 31 Hz sessions.

ii.
```python
lick_binary = (arrays.lick[start:stop_exclusive] > 0).astype(np.int64, copy=False)

if factor == 2:
    neural_trial = rebin_2d_sum_time_last(neural_trial, factor)
    lick_binary = rebin_1d_any(lick_binary, factor)
```

iii. The notes justify this by the common-bin harmonization strategy and paper-inspired lick QC.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is inferred from `reward_zone/data`, `position/data`, `Reward/timestamps`, `trial number/data`, and trial-level environment labels.

ii.
```python
trial_zone, observed_zone_position = infer_trial_zones(
    positions=np.clip(arrays.position, 0.0, TRACK_LENGTH_CM),
    reward_zone_signal=arrays.reward_zone_signal,
    ...
    reward_frame_idx=reward_frame_idx,
    trial_by_frame=arrays.trial_number.astype(np.int64, copy=False),
    trial_env=trial_env,
)
```

iii. The notes say missing reward-zone labels are common on omission trials, so the agent should infer the active zone from reward-zone-active positions and reward events, then fill within stable blocks.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For each trial, the code takes the median position where `reward_zone_signal > 0`; if none exists, it falls back to the position at the reward frame; that position is mapped to the nearest zone center. Missing labels are then filled either globally, or as pre/post blocks split at trial 30, with an extra environment-based stabilization rule.

ii.
```python
mask = reward_zone_signal[start:stop_exclusive] > 0
if np.any(mask):
    zone_pos = float(np.nanmedian(positions[start:stop_exclusive][mask]))
elif trial_id in reward_pos_by_trial:
    zone_pos = reward_pos_by_trial[trial_id]
else:
    zone_pos = np.nan

if np.isfinite(zone_pos):
    observed[trial_id] = int(np.argmin(np.abs(ZONE_CENTERS - zone_pos)))
```

```python
split = min(SWITCH_TRIAL_INDEX, n_trials)
pre_majority = trial_majority_zone(observed, 0, split, fallback=int(unique_observed[0]))
post_majority = trial_majority_zone(observed, split, n_trials, fallback=post_fallback)
filled[:split][filled[:split] < 0] = pre_majority
filled[split:][filled[split:] < 0] = post_majority
```

iii. The notes explicitly justify this with the paper’s switch timing ("Each switch occurred after 30 trials") and the assumption that one active zone persists within each stable block.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from `Reward/timestamps` together with `trial number/data` and trial boundaries, after assigning reward events to frame indices.

ii.
```python
reward_times=beh["Reward/timestamps"][()].astype(np.float64, copy=False),
trial_number=beh["trial number/data"][()][:t_common].astype(np.float32, copy=False),
```

```python
reward_frame_idx = reward_frames_for_session(arrays)
rewarded_trial_ids = set(int(t) for t in arrays.trial_number[reward_frame_idx] if t >= 0)
```

iii. The notes say reward outcome should be "Rewarded if any `Reward` event timestamp falls within the trial."

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are mapped to frame indices with `searchsorted`, each reward frame is converted to a trial id using the NWB `trial number` stream, each trial is marked rewarded if it contains at least one reward frame, and that binary label is repeated across all bins in the trial.

ii.
```python
idx = np.searchsorted(arrays.timestamps, arrays.reward_times, side="left")
...
rewarded_trial_ids = set(int(t) for t in arrays.trial_number[reward_frame_idx] if t >= 0)

trial_rewarded = np.array(
    [1 if trial_id in rewarded_trial_ids else 0 for trial_id in range(len(trial_slices))],
    dtype=np.int64,
)
...
reward_bin = np.full((t_bins,), int(trial_rewarded[trial_id]), dtype=np.int64)
```

iii. The notes tie this directly to the paper’s rewarded-vs-omitted trial structure and use the omission fraction as a sanity check.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles several edge cases: it crops neural and behavior arrays to a common frame count, ignores invalid negative environment values, supports sessions with no reward events, infers missing trial reward-zone labels using reward-zone signal/reward fallback plus block filling, skips lick-artifact trials, and raises hard errors if trial markers are missing, no zone can be inferred, the sampling rate is unexpected, or fewer than two trials survive.

ii.
```python
t_common = min(t_neural, t_behavior)
...
timestamps=timestamps[:t_common],
...
neural=neural[:t_common],
```

```python
env = env[env >= 0]
...
if arrays.reward_times.size == 0:
    return np.empty((0,), dtype=np.int64)
...
if unique_observed.size == 0:
    raise RuntimeError("Could not infer any reward-zone labels from the session.")
...
if lick_artifact[trial_id]:
    continue
```

iii. The notes justify these as archive quirks discovered during exploration: 1-sample neural/behavior mismatches, omission trials with missing reward-zone cues, and lick sensor artifacts close to the paper’s QC fraction.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive parts are session I/O and full-array materialization from NWB/HDF5, then the per-session conversion loop over trials; optional plotting adds extra cost but is only used for a few sessions.

ii.
```python
with h5py.File(path, "r") as f:
    ...
    neural = neural_group["data"][()].astype(np.float32, copy=False)
```

```python
for session_idx, path in enumerate(files):
    arrays = load_session_arrays(path)
    ...
    neural_trials, input_trials, output_trials, brain_region_idx, session_info = convert_session(
        arrays=arrays,
        show_processing=args.show_processing and session_idx < 2,
    )
```

iii. The notes explicitly list "Full-session NWB reads" and "Processing plots" as the main overheads, and report `~95 s` for full conversion of all 152 sessions.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The most obvious non-vectorized parts are the loops over reward frames and trials in `infer_trial_zones`, the loop that computes per-trial environment / lick-artifact flags, and the main per-trial loop that slices, rebins, and stacks outputs.

ii.
```python
for frame_idx in reward_frame_idx:
    ...

for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    ...
```

```python
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    if lick_artifact[trial_id]:
        continue
    ...
    input_trial = np.stack(...)
    output_trial = np.stack(...)
```

iii. The notes say the code already uses vectorized HDF5 reads and simple vectorized rebinning, but per-trial variable-length slicing remains the natural bottleneck.

## 13-c. What processing does the code repeat multiple times?

i. The script avoids the reference solution’s separate survey pass, but it still repeats some work: in sample mode it first opens many files in `infer_sample_files()` just to inspect environment/rate, then reopens chosen files for actual conversion; within each converted session it separately reads multiple behavior datasets from HDF5 and repeatedly clips/rebins per-trial variables.

ii.
```python
def infer_sample_files(files: list[str]) -> list[str]:
    metadata = []
    for path in files:
        with h5py.File(path, "r") as f:
            env = f["processing/behavior/BehavioralTimeSeries/environment/data"][()]
            rate = float(f["processing/ophys/Deconvolved/plane0/starting_time"].attrs["rate"])
```

```python
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
speed = np.clip(arrays.speed[start:stop_exclusive], 0.0, None)
...
if factor == 2:
    position = rebin_1d_mean(position, factor)
    speed = rebin_1d_mean(speed, factor)
```

iii. The notes emphasize that the agent deliberately removed the separate survey step from the final conversion script as an efficiency improvement, so repetition is limited compared with the human reference.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Some arrays and intermediate values are loaded or computed but not used to build the final decoder tensors: `scanning` and `roi_ids` are stored in `SessionArrays` but never used later; `observed_zone_position` is computed only for metadata; and optional plotting/summary work is purely diagnostic.

ii.
```python
scanning=beh["scanning/data"][()][:t_common].astype(np.float32, copy=False),
...
roi_ids=roi_ids,
```

```python
trial_zone, observed_zone_position = infer_trial_zones(...)

session_info = {
    ...
    "observed_zone_positions_cm": observed_zone_position.tolist(),
```

```python
if show_processing:
    plot_processing_summary(...)
```

iii. The notes themselves call the plots "sample/debug mode" artifacts and describe several metadata/diagnostic checks that are not required for downstream decoder training.
