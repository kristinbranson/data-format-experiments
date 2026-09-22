# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script enumerates every NWB file matching `sub-*/sub-*_behavior+ophys.nwb` under `/app/data`, sorts them, and opens each file with `NWBHDF5IO`. All session data are loaded session-by-session inside `_load_session`, and each session contributes its trials to the final dataset.

ii. ```python
session_paths = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))
...
with NWBHDF5IO(str(session_path), "r", load_namespaces=True) as io:
    nwb = io.read()
```

iii. The trajectory says the agent first inspected the NWB release and then planned to "read every NWB session" (step 51). The final summary also states that the converter reads every session from the NWB synchronized frame stream (step 150).

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the parent directory name of each NWB file, with the `sub-` prefix removed. A new subject is added the first time one of its session files is seen.

ii. ```python
subject = session_path.parent.name.replace("sub-", "")
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
```

iii. The agent treated the DANDI-style subject folders as the natural subject split while scanning the data layout (steps 5, 7, 51).

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The sorted file list defines session order in the output lists.

ii. ```python
session_paths = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))
...
for session_idx, session_path in enumerate(session_paths, start=1):
    ...
```

iii. The trajectory repeatedly refers to "every NWB session" and validates the dataset in terms of the number of session files processed (steps 51, 67, 95, 150).

## 1-d. How are the data split into trials?

i. Trials are defined from `trial_start` to `teleport`, using all indices where `trial_start > 0` and `teleport > 0`. The code trims one unmatched leading or trailing event if present, pairs the resulting start/stop indices, and keeps frames `[start:stop)` for each trial.

ii. ```python
def _prepare_trial_bounds(trial_start, teleport):
    starts = np.flatnonzero(trial_start > 0)
    teleports = np.flatnonzero(teleport > 0)
    ...
    for start, stop in zip(starts, teleports):
        if stop > start:
            bounds.append((int(start), int(stop)))
```

iii. The agent states explicitly that it had "pinned down the right trial slicing rule": pair `trial_start` and `teleport` and keep frames from start up to but excluding teleport, matching the repo's on-track slicing (step 39). After later finding one-frame mismatches, it truncated all streams to a shared minimum length before building those bounds (step 85).

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit per-trial duration or quality filter. The code only keeps `(start, stop)` pairs with `stop > start`, requires at least two valid trials in a session, and otherwise retains all paired trials.

ii. ```python
for start, stop in zip(starts, teleports):
    if stop > start:
        bounds.append((int(start), int(stop)))
...
if len(bounds) < 2:
    raise RuntimeError(f"{session_path.name}: fewer than two valid trials...")
```

iii. The trajectory does not mention a short-trial or behavioral-quality exclusion. The agent focused on making trial pairing valid enough for the decoder validator rather than reproducing the reference solution's explicit trial-length filter (steps 51, 113, 150).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural arrays come from the NWB `Deconvolved` interface, plus the `iscell` and `planeIdx` columns in `PlaneSegmentation` to curate ROIs and pool planes. The script does not derive neural data from raw `Fluorescence` and `Neuropil` traces.

ii. ```python
seg = nwb.processing["ophys"].data_interfaces["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]
deconv_iface = nwb.processing["ophys"].data_interfaces["Deconvolved"]
...
iscell_all = np.asarray(seg["iscell"].data[:])[:, 0].astype(bool)
plane_idx_all = np.asarray(seg["planeIdx"].data[:], dtype=np.int64)
```

iii. The agent explicitly compared using the provided deconvolved traces against recomputing `dF/F` from fluorescence, then chose the released deconvolved signal for the final converter (step 32). Its final summary says the neural signal is the released deconvolved traces filtered to curated `iscell` ROIs (step 150).

## 2-b. How is the `neural` data processed?

i. For each plane, the code selects curated `iscell` ROIs, concatenates the per-plane deconvolved matrices across planes, truncates all streams to the shared minimum frame count, applies a speed-correlation filter, and then slices each trial and stores it as a neuron-by-time matrix in `float16`.

ii. ```python
for series_name, rr in sorted(deconv_iface.roi_response_series.items(), ...):
    ...
    deconv_parts.append(np.asarray(rr.data[:, :], dtype=np.float32)[:, keep_plane])
...
deconv = np.concatenate(deconv_parts, axis=1)
...
non_interneuron = _speed_corr_filter(deconv, speed, bounds)
...
neural_trial = np.ascontiguousarray(deconv[start:stop].T, dtype=np.float16)
```

iii. The trajectory says the multi-plane sessions should be handled by concatenating `plane0` and `plane1`, matching the paper's pooled-plane analysis (step 80). The final summary frames this as preserving curated ROI filtering and on-track slicing while deriving decoder inputs from the synchronized frame stream (step 150).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural QC consists of two filters: keep only `iscell` ROIs, then remove neurons whose deconvolved activity has correlation greater than `0.5` with running speed on on-track frames. In practice the run that generated the saved pickle excluded zero neurons by this second filter.

ii. ```python
iscell_all = np.asarray(seg["iscell"].data[:])[:, 0].astype(bool)
...
non_interneuron = _speed_corr_filter(deconv, speed, bounds)
deconv = deconv[:, non_interneuron]
plane_idx = plane_idx[non_interneuron]
```

iii. The agent investigated whether it should reproduce the paper's extra interneuron screen and described its final approach as a "practical approximation" based on speed correlation of the provided deconvolved traces rather than fully recomputed `dF/F` (steps 24, 32, 150).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned by trial slicing itself. Each trial starts at the `trial_start` frame, so the neural trial matrices begin at trial start and end immediately before the paired `teleport` frame.

ii. ```python
bounds = _prepare_trial_bounds(trial_start, teleport)
...
neural_trial = np.ascontiguousarray(deconv[start:stop].T, dtype=np.float16)
```

iii. The agent states that the correct alignment event is trial start and that frames should be kept from `trial_start` inclusive to `teleport` exclusive (step 39).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The code assumes a fixed frame rate of `15.5078125 Hz`, so each bin is `1000 / 15.5078125` ms (about `64.48 ms`). No temporal rebinning or resampling is applied; the saved trials use the native per-frame samples from the NWB deconvolved series.

ii. ```python
FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
...
"time_bin_size": TIME_BIN_MS,
```

iii. The agent treated the dataset as a synchronized frame stream and kept the raw framewise sampling for decoding rather than resampling it (steps 51 and 150).

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavior timestamps attached to the `position` time series. The code slices those timestamps per trial and then subtracts the first sample time.

ii. ```python
timestamps = np.asarray(beh["position"].timestamps[:], dtype=np.float64)
...
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The agent consistently described the NWB behavior streams as already synchronized framewise, so it chose one of those shared timestamp arrays directly rather than reconstructing time separately (steps 51, 85, 150).

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Within each trial, the code subtracts the first timestamp from every timestamp in that trial. The result is a continuous time-from-trial-start vector.

ii. ```python
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. There is no separate explicit justification in the trajectory beyond the decision to use the synchronized frame timestamps and align trials by `trial_start` (steps 39, 51).

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time vector is built from the same `[start:stop)` frame slice used for neural activity, after all continuous streams are truncated to the shared minimum frame count. That gives one time value per neural frame within each trial.

ii. ```python
n_frames = min(..., deconv.shape[0])
...
timestamps = timestamps[:n_frames]
...
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
neural_trial = np.ascontiguousarray(deconv[start:stop].T, dtype=np.float16)
```

iii. When the agent hit a one-frame behavior/ophys mismatch, it explicitly decided to truncate every continuous stream to the shared minimum length before building trials, citing the repo's own one-frame correction logic (step 85).

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the raw `environment` behavior time series.

ii. ```python
environment = np.asarray(beh["environment"].data[:], dtype=np.float32)
```

iii. The agent inspected whether environment switches could happen within a session, then chose the `environment` series as the authoritative source (steps 53 to 55).

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the code drops negative values, takes the mode of the remaining environment values, and then repeats that single environment code across all timepoints in the trial.

ii. ```python
env_vals = environment[start:stop]
env_vals = env_vals[env_vals >= 0]
env = float(_safe_mode(env_vals.astype(int), fallback=0))
...
np.full(T, env, dtype=np.float32)
```

iii. The trajectory shows the agent checked that the day-8 environment switch is across trials, not within a trial, and therefore treated environment as a per-trial label while preserving within-session switches (steps 53 to 55, 150).

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is taken from the raw `trial number` behavior series at the trial's first frame. If that value is negative, the code falls back to the loop index for the trial.

ii. ```python
trial_number = np.asarray(beh["trial number"].data[:], dtype=np.float32)
...
tnum = float(trial_number[start]) if trial_number[start] >= 0 else float(trial_idx)
```

iii. The trajectory does not contain a dedicated justification for preferring the NWB `trial number` field. The choice is implicit in the agent's broader plan to derive labels directly from the synchronized frame stream (steps 51, 150).

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transformation is applied beyond choosing one scalar trial number per trial and repeating it across all timepoints of that trial.

ii. ```python
tnum = float(trial_number[start]) if trial_number[start] >= 0 else float(trial_idx)
...
np.full(T, tnum, dtype=np.float32)
```

iii. The trajectory does not give a separate argument here; the implementation simply treats trial number as a per-trial constant input (steps 51, 150).

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the reward event timestamps in the `Reward` behavior series, combined with the trial bounds defined from `trial_start` and `teleport`.

ii. ```python
reward_timestamps = np.asarray(beh["Reward"].timestamps[:], dtype=np.float64)
...
reward_outcome_by_trial[trial_idx] = np.uint8(
    np.any((reward_timestamps >= trial_start_t) & (reward_timestamps < trial_end_t))
)
```

iii. The agent's plan explicitly says reward outcome should come from the sparse reward timestamps rather than a dense per-frame signal (step 51).

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The code first computes a current-trial reward outcome for every trial using reward timestamps. Then, for each trial, it sets `previous_trial_outcome` to the previous trial's reward flag, using `0` for the first trial. The value is repeated across all timepoints in the trial.

ii. ```python
reward_outcome_by_trial = np.zeros(len(bounds), dtype=np.uint8)
for trial_idx, (start, stop) in enumerate(bounds):
    ...
    reward_outcome_by_trial[trial_idx] = np.uint8(np.any(...))
...
prev_reward = float(reward_outcome_by_trial[trial_idx - 1]) if trial_idx > 0 else 0.0
np.full(T, prev_reward, dtype=np.float32)
```

iii. This follows the agent's stated plan to derive reward outcome from sparse reward timestamps and then build trialwise decoder labels from those outcomes (step 51).

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The distance output is derived from the raw `position` time series together with a per-trial reward-zone identity inferred from the raw `reward_zone` series. The reward-zone identity is inferred from positions where `reward_zone > 0`.

ii. ```python
position = np.asarray(beh["position"].data[:], dtype=np.float32)
reward_zone = np.asarray(beh["reward_zone"].data[:], dtype=np.float32)
...
rz_pos = position[start:stop][reward_zone[start:stop] > 0]
if rz_pos.size:
    raw_zone[trial_idx] = int(np.argmin(np.abs(ZONE_CENTERS_CM - np.median(rz_pos))))
```

iii. The agent examined the noisy and sometimes-missing `reward_zone` events, then decided to infer reward-zone block identity from reward-zone event positions with gap filling for trials that missed the event (steps 46, 49, 51, 150).

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. First, the code infers a zone label per trial. Then, for each frame, it computes signed distance to that zone's nearest edge: negative before the zone, `0` inside the zone, positive after the zone. Finally it converts those distances to categorical bins.

ii. ```python
def _bin_distance_to_reward_zone(position_cm, zone_idx):
    start = ZONE_STARTS_CM[zone_idx]
    end = ZONE_ENDS_CM[zone_idx]
    dist = np.zeros_like(position_cm, dtype=np.float32)
    before = position_cm < start
    after = position_cm > end
    dist[before] = position_cm[before] - start
    dist[after] = position_cm[after] - end
```

iii. The trajectory says the reward-zone block identity should respect the paper's session structure, and the final summary says trial labels are inferred from reward-zone event positions with interpolation-like gap filling when events are missing (steps 51, 150).

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The code uses seven hard-coded categories matching the task instructions: `< -50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, and `> 50` cm.

ii. ```python
bins[dist < -50.0] = 0
bins[(dist >= -50.0) & (dist < -10.0)] = 1
bins[(dist >= -10.0) & (dist < 0.0)] = 2
bins[dist == 0.0] = 3
bins[(dist > 0.0) & (dist <= 10.0)] = 4
bins[(dist > 10.0) & (dist <= 50.0)] = 5
bins[dist > 50.0] = 6
```

iii. There is no separate discussion in the trajectory beyond implementing the decoder task's requested discretization when the converter was written (step 60).

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Distance-to-zone is computed from `position[start:stop]` for the same frame slice used to build the neural trial matrix, so it is aligned frame-by-frame with neural activity.

ii. ```python
pos = np.clip(position[start:stop], 0.0, TRACK_LENGTH_CM).astype(np.float32, copy=False)
...
neural_trial = np.ascontiguousarray(deconv[start:stop].T, dtype=np.float16)
output_trial = np.vstack([
    _bin_distance_to_reward_zone(pos, zone_idx),
    ...
])
```

iii. The agent's stated slicing rule was to use the shared frame indices from `trial_start` to `teleport`, so behavioral and neural trial signals stay aligned without extra interpolation (step 39).

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the raw `position` behavior time series.

ii. ```python
position = np.asarray(beh["position"].data[:], dtype=np.float32)
...
pos = np.clip(position[start:stop], 0.0, TRACK_LENGTH_CM).astype(np.float32, copy=False)
```

iii. The agent inspected track position ranges before implementation and then used the per-frame position stream as the source for position outputs (step 42).

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Before binning, the code clips position values to the nominal track range `[0, 450]` cm. It then assigns each clipped frame to one of five position bins.

ii. ```python
pos = np.clip(position[start:stop], 0.0, TRACK_LENGTH_CM).astype(np.float32, copy=False)
...
_bin_absolute_position(pos)
```

iii. The trajectory shows the agent sampled sessions to look for position values outside the track and reported no serious issues in that sample (step 42). The clipping in the final code appears to be a defensive simplification rather than a separately justified processing step.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The clipped position is discretized into five bins: `<90`, `[90,180)`, `[180,270)`, `[270,360)`, and `>=360` cm.

ii. ```python
def _bin_absolute_position(position_cm):
    bins[position_cm < 90.0] = 0
    bins[(position_cm >= 90.0) & (position_cm < 180.0)] = 1
    bins[(position_cm >= 180.0) & (position_cm < 270.0)] = 2
    bins[(position_cm >= 270.0) & (position_cm < 360.0)] = 3
    bins[position_cm >= 360.0] = 4
```

iii. This matches the five equal-width track bins requested by the decoder task; the trajectory does not add extra justification beyond implementing that requirement.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Absolute position uses the same per-trial `[start:stop)` frame slice as the neural data, so it is aligned one sample per neural frame.

ii. ```python
pos = np.clip(position[start:stop], 0.0, TRACK_LENGTH_CM).astype(np.float32, copy=False)
neural_trial = np.ascontiguousarray(deconv[start:stop].T, dtype=np.float16)
```

iii. The agent's overall alignment rule was to derive decoder labels directly from the same synchronized frame stream used to slice the neural trials (steps 39, 150).

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the raw `lick` behavior time series.

ii. ```python
lick = np.asarray(beh["lick"].data[:], dtype=np.float32)
```

iii. The agent used the per-frame behavior streams in the NWB file directly for decoder labels once it had confirmed the streams were synchronized (steps 20, 51).

## 9-b. What processing is involved in computing `output` *Lick*?

i. The code binarizes lick values by thresholding at `> 0`.

ii. ```python
lick_bin = (lick[start:stop] > 0).astype(np.uint8, copy=False)
```

iii. No separate trajectory discussion was recorded for lick; the implementation follows the requested binary output format.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is sliced with the same `[start:stop)` frame indices used for the neural data and other behavior outputs.

ii. ```python
lick_bin = (lick[start:stop] > 0).astype(np.uint8, copy=False)
neural_trial = np.ascontiguousarray(deconv[start:stop].T, dtype=np.float16)
```

iii. The trajectory treats the behavior streams as already synchronized to the neural frame stream after session-level truncation (step 85).

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from the raw `reward_zone` series together with `position`, because the code infers zone identity from where reward-zone-positive samples fall along the track.

ii. ```python
rz_pos = position[start:stop][reward_zone[start:stop] > 0]
if rz_pos.size:
    raw_zone[trial_idx] = int(np.argmin(np.abs(ZONE_CENTERS_CM - np.median(rz_pos))))
```

iii. The agent explicitly planned to derive reward-zone block identity from reward-zone event positions, with gap filling for trials where the event was missing (step 51).

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The code infers a raw zone for each trial from the median reward-zone position, maps that to the nearest of three fixed zone centers, then fills missing or noisy trials using a hard split at trial 30: pre-switch trials get the pre-switch mode, post-switch trials get the post-switch mode. If only one side has valid detections, it uses nearest-neighbor forward/backward filling.

ii. ```python
if rz_pos.size:
    raw_zone[trial_idx] = int(np.argmin(np.abs(ZONE_CENTERS_CM - np.median(rz_pos))))
...
pre_valid = raw_zone[:SWITCH_SPLIT_TRIAL][raw_zone[:SWITCH_SPLIT_TRIAL] >= 0]
post_valid = raw_zone[SWITCH_SPLIT_TRIAL:][raw_zone[SWITCH_SPLIT_TRIAL:] >= 0]
...
inferred[:SWITCH_SPLIT_TRIAL] = _safe_mode(pre_valid)
inferred[SWITCH_SPLIT_TRIAL:] = _safe_mode(post_valid, fallback=int(inferred[SWITCH_SPLIT_TRIAL - 1]))
```

iii. The agent justified this by appealing to the paper's 30-trial switch structure and by noting that many trials were missing explicit reward-zone events, so some gap-filling heuristic was needed (steps 46, 49, 51, 55, 150).

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from the `Reward` event timestamps together with the trial start/end timestamps for each trial.

ii. ```python
reward_timestamps = np.asarray(beh["Reward"].timestamps[:], dtype=np.float64)
...
reward_outcome_by_trial[trial_idx] = np.uint8(
    np.any((reward_timestamps >= trial_start_t) & (reward_timestamps < trial_end_t))
)
```

iii. The agent's plan explicitly names reward outcome as a label built from sparse reward timestamps rather than a dense analog trace (step 51).

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the code marks reward outcome as `1` if any reward timestamp falls within that trial's `[start_time, end_time)` interval, and `0` otherwise. It then repeats that scalar across all timepoints in the trial.

ii. ```python
for trial_idx, (start, stop) in enumerate(bounds):
    trial_start_t = timestamps[start]
    trial_end_t = timestamps[stop]
    reward_outcome_by_trial[trial_idx] = np.uint8(np.any((reward_timestamps >= trial_start_t) & (reward_timestamps < trial_end_t)))
...
np.full(T, reward_outcome, dtype=np.uint8)
```

iii. The trajectory describes this exactly at the planning level: reward outcome should come from sparse reward timestamps and become a trialwise decoder label (step 51).

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles several edge cases by silent truncation or fallback heuristics. It truncates every continuous stream to the minimum shared frame count; drops one unmatched leading teleport or trailing trial start if counts differ by one; discards invalid `(start, stop)` pairs where `stop <= start`; fills missing reward-zone trials by blockwise mode or nearest-neighbor propagation; falls back to environment `0` if a trial has no nonnegative environment values; uses the loop index if `trial number` is negative at trial start; and raises errors if no valid reward-zone trial exists, if fewer than two trials remain, or if no neurons survive filtering.

ii. ```python
n_frames = min(..., deconv.shape[0])
...
if teleports.size == starts.size + 1 and teleports[0] < starts[0]:
    teleports = teleports[1:]
if starts.size == teleports.size + 1 and starts[-1] > teleports[-1]:
    starts = starts[:-1]
...
if not np.any(valid):
    raise RuntimeError("Could not infer reward zone...")
```

iii. The agent explicitly added the shared-minimum truncation after encountering one-frame mismatches and described reward-zone gap filling as a response to missed reward-zone events (steps 46, 85, 150).

## 13-a. What are the most time-consuming steps of the code?

i. The dominant costs are loading all NWB sessions, reading/concatenating the large deconvolved matrices, writing the 4.6 GB pickle, and then running the downstream validator/training pass. Within conversion, the work is mostly I/O-bound session reads.

ii. ```python
with NWBHDF5IO(str(session_path), "r", load_namespaces=True) as io:
    nwb = io.read()
...
with OUTPUT_PATH.open("wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory repeatedly describes the conversion as still being in the session-read phase and mostly I/O-bound, and later notes the 4.6 GB pickle size before validation (steps 67, 73, 95, 113).

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious non-vectorized pieces are the per-trial `reward_outcome_by_trial` loop, the per-trial assembly loop that slices and stacks every input/output array, the reward-zone inference loop over trials, and the nearest-neighbor fill loop for missing reward-zone labels. These are all written in Python loops over trials.

ii. ```python
for trial_idx, (start, stop) in enumerate(bounds):
    trial_start_t = timestamps[start]
    ...
for trial_idx, (start, stop) in enumerate(bounds):
    pos = np.clip(position[start:stop], 0.0, TRACK_LENGTH_CM)
    ...
for idx in range(first_valid + 1, len(inferred)):
    if inferred[idx] < 0:
        inferred[idx] = inferred[idx - 1]
```

iii. The trajectory does not contain an explicit efficiency discussion here; this follows directly from inspection of the final code structure.

## 13-c. What processing does the code repeat multiple times?

i. The code makes multiple passes over the same trial bounds within each session: one pass to infer reward-zone identity, one pass to compute trialwise reward outcome, and another pass to build neural/input/output trial arrays. It also recomputes per-trial environment modes once for the actual input tensors and again when populating `session_info` metadata. After conversion it scans outputs again in `print_summary` to count classes.

ii. ```python
zone_by_trial = _infer_reward_zone_by_trial(position, reward_zone, bounds)
...
for trial_idx, (start, stop) in enumerate(bounds):
    reward_outcome_by_trial[trial_idx] = ...
...
for trial_idx, (start, stop) in enumerate(bounds):
    ...
    input_trials.append(input_trial)
    output_trials.append(output_trial)
```

iii. The trajectory does not justify these repeated passes explicitly. They are simply a consequence of the implementation style used in the final converter.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code preserves `plane_idx` long enough to filter it, but then discards plane distinctions by writing `brain_region_idx` as all zeros because every neuron is labeled CA1. It also computes and stores `session_info` metadata such as per-session reward-zone labels and environment sets, and later computes summary counters in `print_summary`; those values are not used by the decoder itself.

ii. ```python
plane_idx = np.concatenate(plane_parts)
...
data["brain_region_idx"].append(np.zeros(plane_idx.shape[0], dtype=np.int64))
...
session_info = {
    ...
    "reward_zone_labels": [ZONE_LABELS[idx] for idx in zone_by_trial.tolist()],
    "environments": sorted({...}),
}
```

iii. The trajectory emphasizes successful validation and training rather than any downstream use of these metadata fields, so they appear to be convenience bookkeeping rather than decoder-critical outputs (steps 113, 150).
