# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates all NWB files under `/app/data/sub-*/sub-*_behavior+ophys.nwb`, sorts them, and processes each file as one session. Each file is opened with `h5py`, selected behavior streams are read into memory, all deconvolved ophys plane series are read and concatenated, and later trial slices are extracted from those session-level arrays.

ii.
```python
files = sorted(Path("/app/data").glob("sub-*/sub-*_behavior+ophys.nwb"))
...
for session_i, path in enumerate(files):
    session_raw = load_session_raw(path)
```

```python
def load_session_raw(path):
    with h5py.File(path, "r") as f:
        ...
        for name in [
            "environment", "position", "speed", "lick", "reward_zone",
            "scanning", "trial number", "trial_start", "teleport", "autoreward",
        ]:
            behavior[name] = f[f"{BEHAVIOR_TS_PATH}/{name}/data"][:]
        ...
        plane_series = sorted(f[f"{OPHYS_TS_PATH}/Deconvolved"].keys())
```

iii. In `CONVERSION_NOTES.md`, the AI says the dataset contains one NWB file per subject-session, and that direct `h5py` access was chosen for speed and lower overhead than building full `pynwb` objects. It also states sessions are processed sequentially to avoid unnecessary memory use.

## 1-b. How are the data split into subjects?

i. Subjects are split by NWB file grouping under `sub-*` directories, but the canonical subject label comes from each file’s `general/subject/subject_id`. Unique subjects are collected in order of first encountered session.

ii.
```python
subject = read_str(f["general/subject/subject_id"])
...
if subject not in subject_lookup:
    subject_lookup[subject] = len(subject_lookup)
    data["subjects"].append(subject)
subject_idx.append(subject_lookup[subject])
```

iii. The notes state the NWB bundle has subject folders such as `sub-m3`, `sub-m11`, etc., and that subject/session counts from these files match the expected 11 mice.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The session identifier is also read from NWB metadata, but the session unit of processing is the file itself.

ii.
```python
files = sorted(Path("/app/data").glob("sub-*/sub-*_behavior+ophys.nwb"))
...
session_id = read_str(f["general/session_id"])
...
for session_i, path in enumerate(files):
    session_raw = load_session_raw(path)
```

iii. The notes explicitly say there is one NWB file per subject-session and report 152 NWB session files in the release.

## 1-d. How are the data split into trials?

i. Trials are defined by pairing each positive `trial_start` sample with the next positive `teleport` sample. Per-trial arrays use the half-open slice `[start:stop)`, so the on-track portion is kept and the teleport/tunnel period is excluded.

ii.
```python
def build_trial_bounds(behavior):
    starts = np.flatnonzero(behavior["trial_start"] > 0)
    teleports = np.flatnonzero(behavior["teleport"] > 0)
    bounds = []
    tele_ptr = 0
    for start in starts:
        while tele_ptr < len(teleports) and teleports[tele_ptr] <= start:
            tele_ptr += 1
        if tele_ptr >= len(teleports):
            break
        stop = teleports[tele_ptr]
        tele_ptr += 1
        if stop <= start:
            continue
        bounds.append((int(start), int(stop)))
```

```python
for raw_trial_idx, (start, stop) in enumerate(bounds):
    pos = behavior["position"][start:stop]
    ...
    neural_trial = neural[start:stop].T.astype(np.float32, copy=False)
```

iii. The notes say the reference pipeline uses `trial_start_inds` and `teleport_inds`, and that `trial number` includes tunnel fragments, so `trial_start` to `teleport` was treated as authoritative.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops trials with zero position samples and drops trials where more than 35% of samples have `lick > 2`, which it treats as lick-sensor artifacts. After trial extraction, any session with fewer than two remaining trials is skipped entirely. It does not apply the human reference’s explicit `<50`-timepoint trial filter.

ii.
```python
if pos.size == 0:
    dropped["empty"] += 1
    continue

bad_lick = np.mean(lick_raw > 2.0) > LICK_QC_FRACTION
if bad_lick:
    dropped["lick_qc"] += 1
    continue
```

```python
if len(neural_trials) < 2:
    print(f"Skipping {path.name}: only {len(neural_trials)} valid trials after filtering.")
    continue
```

iii. The notes justify this with the paper’s lick-artifact curation and state that because lick is a decoder output, artifact trials should be excluded rather than preserved.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural signal is taken directly from the NWB `processing/ophys/Deconvolved/<plane>/data` arrays, restricted to curated ROIs via `iscell` and mapped through each response series’ ROI index.

ii.
```python
iscell = f[f"{OPHYS_TS_PATH}/ImageSegmentation/PlaneSegmentation/iscell"][:]
plane_series = sorted(f[f"{OPHYS_TS_PATH}/Deconvolved"].keys())
...
response_group = f[f"{OPHYS_TS_PATH}/Deconvolved/{series_name}"]
rois = response_group["rois"][:]
response_iscell = iscell[rois, 0].astype(np.int64) == 1
response_data = response_group["data"][:common_len, :]
neural_planes.append(response_data[:, response_iscell].astype(np.float32, copy=False))
```

iii. In the notes, the AI argues that the NWB export already contains the deconvolved activity used by the decoder analyses, so recomputing dF/F and deconvolution would introduce avoidable mismatch.

## 2-b. How is the `neural` data processed?

i. Neural preprocessing is minimal: trim all streams to a common session length, keep curated cells, concatenate all deconvolved plane series across planes, cast to `float32`, and then slice per trial and transpose to `(neurons, time)`. No dF/F recomputation, smoothing, or OASIS deconvolution is applied in this script.

ii.
```python
neural_len = min(neural_lens)
behavior_len = min(len(v) for v in behavior.values())
common_len = min(neural_len, behavior_len)
...
response_data = response_group["data"][:common_len, :]
neural_planes.append(response_data[:, response_iscell].astype(np.float32, copy=False))
...
neural = np.concatenate(neural_planes, axis=1)
...
neural_trial = neural[start:stop].T.astype(np.float32, copy=False)
```

iii. The justification in the notes is that the deconvolved traces are already frame-aligned and task-ready, so the AI preserved them instead of reproducing the full paper preprocessing pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron-level quality filter is `iscell[:, 0] == 1` applied within each response series’ ROI region. No additional interneuron or activity-quality filter is applied.

ii.
```python
rois = response_group["rois"][:]
response_iscell = iscell[rois, 0].astype(np.int64) == 1
response_data = response_group["data"][:common_len, :]
neural_planes.append(response_data[:, response_iscell].astype(np.float32, copy=False))
```

iii. The notes say manual Suite2p curation is the applicable filter from the release and that no further automatic neuron filter was added.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to trial start implicitly by slicing exactly the same `[start:stop)` trial window that is defined from `trial_start` to `teleport`.

ii.
```python
bounds, starts, teleports = build_trial_bounds(behavior)
...
for raw_trial_idx, (start, stop) in enumerate(bounds):
    neural_trial = neural[start:stop].T.astype(np.float32, copy=False)
```

iii. The notes say the requested alignment event is trial start, so once trials are cut using `trial_start`, no extra realignment step is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is taken from the median spacing of the behavior timestamps, about 1/15.5 s per sample, and stored in metadata in milliseconds. No temporal rebinning or resampling is applied.

ii.
```python
dt_behavior = float(np.median(np.diff(behavior["timestamps"][: min(common_len, 1000)])))
...
"time_bin_size": dt_sec * 1000.0,
```

iii. The notes justify using behavior timestamps because multi-plane sessions expose a misleading 31 Hz ophys `rate`, while the effective per-plane sampling grid in the paper is still about 15.5 Hz.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavior timestamps copied from `processing/behavior/BehavioralTimeSeries/position/timestamps`.

ii.
```python
behavior["timestamps"] = f[f"{BEHAVIOR_TS_PATH}/position/timestamps"][:]
...
time_raw = behavior["timestamps"][start:stop].astype(np.float32, copy=False)
```

iii. The notes describe the behavior streams as already frame-aligned to imaging samples, so the behavior timestamp grid was treated as the session clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the first timestamp of that trial is subtracted from all timestamps in the slice to obtain elapsed time from trial start.

ii.
```python
time_raw = behavior["timestamps"][start:stop].astype(np.float32, copy=False)
time_from_start = (time_raw - time_raw[0]).astype(np.float32, copy=False)
...
input_trial = np.vstack([time_from_start, ...])
```

iii. The justification is implicit in the code and notes: the decoder input requested elapsed time from the start of each trial.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time vector is taken from the exact same trial slice used for neural data, after session-level trimming to a shared `common_len`.

ii.
```python
common_len = min(neural_len, behavior_len)
...
time_raw = behavior["timestamps"][start:stop].astype(np.float32, copy=False)
neural_trial = neural[start:stop].T.astype(np.float32, copy=False)
```

iii. The notes say behavior and neural streams are already synchronized framewise in the NWB export, and trimming resolves the few one-sample mismatches.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the framewise `environment` behavior time series.

ii.
```python
behavior[name] = f[f"{BEHAVIOR_TS_PATH}/{name}/data"][:]
...
env_raw = behavior["environment"][start:stop]
```

iii. The notes say NWB `environment` already encodes the within-session ENV1/ENV2 identity.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial, negative values are ignored, the modal nonnegative environment value is selected, and that value is repeated across all timepoints of the trial.

ii.
```python
def trial_environment_value(environment_slice):
    valid = environment_slice[environment_slice >= 0]
    ...
    values, counts = np.unique(valid, return_counts=True)
    return float(values[np.argmax(counts)])
```

```python
env_value = trial_environment_value(env_raw)
...
np.full(time_from_start.shape, env_value, dtype=np.float32)
```

iii. The notes justify this as a per-trial contextual variable; the AI states the environment is effectively constant within a trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is taken from the stored `trial number` stream at the trial’s start index.

ii.
```python
trial_label = int(behavior["trial number"][start])
...
np.full(time_from_start.shape, float(trial_label), dtype=np.float32)
```

iii. The notes acknowledge that `trial number` contains tunnel fragments if used session-wide, but the AI considered the value at each true `trial_start` to be a safe per-trial label.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The selected scalar trial label is converted to float and repeated across every timepoint in the trial.

ii.
```python
trial_label = int(behavior["trial number"][start])
...
np.full(time_from_start.shape, float(trial_label), dtype=np.float32)
```

iii. The notes describe this as one of the per-trial contextual variables repeated over time for consistent tensor shapes.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the sparse `Reward` event timestamps, which are aligned to the behavior timestamp grid to create per-trial reward outcomes.

ii.
```python
reward_timestamps = f[f"{BEHAVIOR_TS_PATH}/Reward/timestamps"][:]
...
reward_frame_idx = np.searchsorted(behavior["timestamps"], reward_timestamps, side="left")
...
reward_outcomes_raw, reward_event_groups = trial_reward_outcomes(
    bounds, session_raw["reward_frame_idx"]
)
```

iii. The notes say reward delivery is sparse in NWB rather than already framewise, so it must be reconstructed by timestamp alignment.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The per-trial reward outcome array is shifted by one trial; the first trial is set to 0, and each resulting scalar is repeated across the trial.

ii.
```python
previous_outcomes = np.zeros(len(bounds), dtype=np.int64)
if len(bounds) > 1:
    previous_outcomes[1:] = reward_outcomes_raw[:-1]
...
previous_outcome = int(previous_outcomes[raw_trial_idx])
...
np.full(time_from_start.shape, float(previous_outcome), dtype=np.float32)
```

iii. The notes describe this as the direct implementation of “previous trial outcome” after reconstructing current-trial outcomes from sparse reward events.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from framewise `position` plus a per-trial reward-zone identity inferred from the session `scene` string. The scene is parsed into a reward-zone sequence, and switch sessions change zone after trial 30.

ii.
```python
def parse_scene_reward_sequence(scene):
    match = re.search(r"Location([ABC])_to_([ABC])$", scene)
    ...
    match = re.search(r"Env[12]_([ABC])_to_Env[12]_([ABC])$", scene)
```

```python
def reward_labels_for_trials(scene, n_trials):
    seq = parse_scene_reward_sequence(scene)
    if len(seq) == 1:
        return [seq[0]] * n_trials
    split = min(SWITCH_TRIAL_INDEX, n_trials)
    return [seq[0]] * split + [seq[1]] * max(0, n_trials - split)
```

```python
zone_label = reward_labels[raw_trial_idx]
zone_start, zone_end = REWARD_ZONE_COORDS[zone_label]
distance = signed_distance_to_zone(pos, zone_start, zone_end)
```

iii. The notes explicitly justify using scene metadata instead of the raw `reward_zone` stream because the latter marks occupancy/entry, while the session scene names match the reference switch structure and reward-zone identity.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each sample, signed distance is computed relative to the nearest edge of the current reward zone: negative before the zone, zero inside, positive after the zone.

ii.
```python
def signed_distance_to_zone(position_cm, zone_start, zone_end):
    dist = np.zeros_like(position_cm, dtype=np.float32)
    before = position_cm < zone_start
    after = position_cm > zone_end
    dist[before] = position_cm[before] - zone_start
    dist[after] = position_cm[after] - zone_end
    return dist
```

iii. The notes tie this directly to the task specification and the paper’s fixed reward-zone coordinates.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous signed distance is thresholded into 7 categories with explicit hard-coded boundaries matching the task spec.

ii.
```python
def discretize_distance(distance_cm):
    out = np.full(distance_cm.shape, 6, dtype=np.int64)
    out[distance_cm < -50.0] = 0
    out[(distance_cm >= -50.0) & (distance_cm < -10.0)] = 1
    out[(distance_cm >= -10.0) & (distance_cm < 0.0)] = 2
    out[distance_cm == 0.0] = 3
    out[(distance_cm > 0.0) & (distance_cm < 10.0)] = 4
    out[(distance_cm >= 10.0) & (distance_cm <= 50.0)] = 5
    out[distance_cm > 50.0] = 6
```

iii. The notes say this discretization was chosen to match the decoder-task instructions exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data use the same `[start:stop)` trial indices, so the distance output is aligned simply by computing it from the same trial slice.

ii.
```python
for raw_trial_idx, (start, stop) in enumerate(bounds):
    pos = behavior["position"][start:stop].astype(np.float32, copy=False)
    ...
    neural_trial = neural[start:stop].T.astype(np.float32, copy=False)
    distance = signed_distance_to_zone(pos, zone_start, zone_end)
```

iii. The notes repeatedly state that the NWB release already contains frame-aligned neural and behavioral streams.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the framewise `position` behavior series.

ii.
```python
pos = behavior["position"][start:stop].astype(np.float32, copy=False)
...
discretize_absolute_position(pos)
```

iii. The notes identify `position` as one of the already aligned framewise behavior variables carried into the conversion.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI uses the raw per-trial position slice without smoothing or normalization and discretizes it into 5 corridor bins.

ii.
```python
def discretize_absolute_position(position_cm):
    out = np.zeros(position_cm.shape, dtype=np.int64)
    out[(position_cm >= 90.0) & (position_cm < 180.0)] = 1
    out[(position_cm >= 180.0) & (position_cm < 270.0)] = 2
    out[(position_cm >= 270.0) & (position_cm <= 360.0)] = 3
    out[position_cm > 360.0] = 4
    return out
```

iii. The justification is the task requirement of 5 equal bins across a 450 cm track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is thresholded at 90, 180, 270, and 360 cm into 5 categories.

ii.
```python
out[(position_cm >= 90.0) & (position_cm < 180.0)] = 1
out[(position_cm >= 180.0) & (position_cm < 270.0)] = 2
out[(position_cm >= 270.0) & (position_cm <= 360.0)] = 3
out[position_cm > 360.0] = 4
```

iii. The AI’s notes say this is the direct implementation of the instruction’s “5 equal-sized bins spanning the 450 cm track.”

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The position output is computed from the same per-trial slice used for neural data, so no extra alignment step is applied.

ii.
```python
pos = behavior["position"][start:stop].astype(np.float32, copy=False)
neural_trial = neural[start:stop].T.astype(np.float32, copy=False)
```

iii. The notes treat the behavioral frame grid as already aligned to the neural samples.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the framewise `lick` behavior series.

ii.
```python
lick_raw = behavior["lick"][start:stop].astype(np.float32, copy=False)
```

iii. The notes identify lick as one of the aligned behavioral streams in the NWB export.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The raw lick counts are binarized so that any positive value becomes 1 and 0 stays 0.

ii.
```python
lick_binary = (lick_raw > 0.0).astype(np.int64, copy=False)
...
output_trial = np.vstack([
    ...,
    lick_binary,
    ...
])
```

iii. The notes justify this because the decoder target is binary lick/no-lick.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is taken from the same trial slice as neural activity, after any session-level trimming to a common length.

ii.
```python
lick_raw = behavior["lick"][start:stop].astype(np.float32, copy=False)
neural_trial = neural[start:stop].T.astype(np.float32, copy=False)
```

iii. The notes say behavior and neural are already synchronized framewise, with only a few one-sample mismatches trimmed away.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from the session `scene` metadata string rather than from the framewise `reward_zone` series. The scene is parsed into A/B/C labels and, when needed, a first-zone then second-zone sequence.

ii.
```python
scene = read_str(f["identifier"]).split("/")[-1]
...
def parse_scene_reward_sequence(scene):
    match = re.search(r"Location([ABC])_to_([ABC])$", scene)
    ...
def reward_labels_for_trials(scene, n_trials):
    ...
```

iii. The notes explicitly argue that scene metadata provides the intended zone identity, while `reward_zone` in NWB is an occupancy/event-like signal and not the canonical A/B/C label source.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene string is parsed into zone labels, switch days change after trial 30, each zone label is mapped `A->0`, `B->1`, `C->2`, and the per-trial scalar is repeated across the full trial.

ii.
```python
reward_labels = reward_labels_for_trials(session_raw["scene"], len(bounds))
...
np.full(pos.shape, {"A": 0, "B": 1, "C": 2}[zone_label], dtype=np.int64)
```

iii. The notes justify the trial-30 split from the paper’s switch-day structure and the reference code’s scene-based reward-zone logic.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from sparse `Reward` timestamps aligned to the behavior timestamp grid.

ii.
```python
reward_timestamps = f[f"{BEHAVIOR_TS_PATH}/Reward/timestamps"][:]
reward_frame_idx = np.searchsorted(behavior["timestamps"], reward_timestamps, side="left")
...
reward_outcomes_raw, reward_event_groups = trial_reward_outcomes(
    bounds, session_raw["reward_frame_idx"]
)
```

iii. The notes explain that NWB stores reward delivery as sparse events, so it must be converted into per-trial labels.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps after the last kept behavior sample are dropped, the remaining events are converted to frame indices, and a trial is labeled rewarded if any aligned reward event falls within `[start, stop)`. The scalar label is then repeated across the trial.

ii.
```python
reward_valid = reward_timestamps <= behavior["timestamps"][-1]
reward_timestamps = reward_timestamps[reward_valid]
...
def trial_reward_outcomes(bounds, reward_frame_idx):
    outcomes = np.zeros(len(bounds), dtype=np.int64)
    for i, (start, stop) in enumerate(bounds):
        has_reward = np.any((reward_frame_idx >= start) & (reward_frame_idx < stop))
        outcomes[i] = int(has_reward)
```

```python
reward_outcome = int(reward_outcomes_raw[raw_trial_idx])
...
np.full(pos.shape, reward_outcome, dtype=np.int64)
```

iii. The notes justify this as the direct per-trial rewarded/omitted definition used in the paper’s task.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases defensively: neural and behavior streams are trimmed to a shared `common_len`; reward events after the last behavior timestamp are dropped and frame indices are clipped into range; starts without a later teleport are ignored; zero-length trials and lick-artifact trials are dropped; sessions with fewer than two valid trials are skipped. It does not attempt interpolation or imputation.

ii.
```python
common_len = min(neural_len, behavior_len)
...
for key in behavior:
    behavior[key] = behavior[key][:common_len]
```

```python
reward_valid = reward_timestamps <= behavior["timestamps"][-1]
reward_timestamps = reward_timestamps[reward_valid]
reward_frame_idx = np.searchsorted(behavior["timestamps"], reward_timestamps, side="left")
reward_frame_idx = np.clip(reward_frame_idx, 0, common_len - 1)
```

```python
if tele_ptr >= len(teleports):
    break
...
if pos.size == 0:
    dropped["empty"] += 1
    continue
```

iii. The notes cite concrete release-specific issues: some multi-plane sessions are one sample longer on the neural side, and one session contains a trailing `trial number` fragment with no `trial_start`, so the code was written to trim and ignore incomplete fragments rather than repair them heuristically.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive work is reading each NWB/HDF5 session, loading the full curated deconvolved neural matrices, slicing all trial arrays, and serializing the final large pickle. Optional plotting also adds overhead when enabled.

ii.
```python
for session_i, path in enumerate(files):
    session_raw = load_session_raw(path)
    neural_trials, input_trials, output_trials, session_info, plot_payload = make_trial_arrays(...)
```

```python
with outpath.open("wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes explicitly discuss runtime and say direct `h5py` loading and sequential per-session processing were chosen because file I/O and large-array handling dominate cost.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities are the Python trial loops in `build_trial_bounds`, `trial_reward_outcomes`, and especially `make_trial_arrays`, where distance, discretization, and constant-array construction are repeated trial by trial.

ii.
```python
for start in starts:
    ...
```

```python
for i, (start, stop) in enumerate(bounds):
    has_reward = np.any((reward_frame_idx >= start) & (reward_frame_idx < stop))
```

```python
for raw_trial_idx, (start, stop) in enumerate(bounds):
    pos = behavior["position"][start:stop]
    ...
    output_trial = np.vstack([...])
```

iii. The code structure reflects variable-length trials, but many per-trial computations could in principle be done session-wide before splitting.

## 13-c. What processing does the code repeat multiple times?

i. The code avoids a separate survey pass, but it still repeats some work: within `trial_reward_outcomes` the same reward-in-trial mask is computed twice per trial (`np.any(...)` and `np.flatnonzero(...)`); constant per-trial arrays are freshly allocated for each feature; and if plotting is enabled, trial data are traversed again for visualization.

ii.
```python
has_reward = np.any((reward_frame_idx >= start) & (reward_frame_idx < stop))
outcomes[i] = int(has_reward)
reward_trials.append(np.flatnonzero((reward_frame_idx >= start) & (reward_frame_idx < stop)))
```

```python
input_trial = np.vstack([
    time_from_start,
    np.full(time_from_start.shape, env_value, dtype=np.float32),
    np.full(time_from_start.shape, float(trial_label), dtype=np.float32),
    np.full(time_from_start.shape, float(previous_outcome), dtype=np.float32),
])
```

iii. The notes emphasize that the AI intentionally avoided the larger repeated work of loading every file twice; the remaining repetition is local within the per-session conversion.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads several fields that are not used to build the final decoder inputs/outputs, including `reward_zone`, `scanning`, `autoreward`, `reward_amounts`, and some metadata-only quantities such as `ophys_rate_attr_hz` and `plane_idx`. It also constructs `trial_debug`/plot payloads when plotting is requested, which are not part of the saved dataset.

ii.
```python
for name in [
    "environment", "position", "speed", "lick", "reward_zone",
    "scanning", "trial number", "trial_start", "teleport", "autoreward",
]:
    behavior[name] = f[f"{BEHAVIOR_TS_PATH}/{name}/data"][:]
reward_amounts = f[f"{BEHAVIOR_TS_PATH}/Reward/data"][:]
```

```python
plot_payload = None
if show_processing:
    plot_payload = {
        "session_key": ...,
        "trial_debug": trial_debug,
        "neural_trials": neural_trials,
        "input_trials": input_trials,
        "output_trials": output_trials,
    }
```

iii. The notes frame these as acceptable overhead for validation and diagnostics, but they are not necessary for the minimal downstream decoder dataset.
