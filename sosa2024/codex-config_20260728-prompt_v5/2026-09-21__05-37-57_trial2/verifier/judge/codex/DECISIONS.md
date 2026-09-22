# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script loads all data by globbing every NWB file under `/app/data/sub-*` and then converting each file as one session. It does not use a cached survey file or any per-subject manifest.

ii.
```python
def main() -> None:
    args = parse_args()
    data_root = Path("/app/data")
    files = sorted(data_root.glob("sub-*/sub-*_behavior+ophys.nwb"))
    ...
    dataset = build_dataset(files, show_processing=args.show_processing)
```

```python
for session_idx, file_path in enumerate(files):
    neural_trials, input_trials, output_trials, plane_per_cell, meta, stats = convert_session(file_path)
```

iii. In `CONVERSION_NOTES.md`, the agent says the NWB dataset has one folder per subject and one NWB file per session, so globbing all matching files should capture the full 152-session dataset.

## 1-b. How are the data split into subjects?

i. Subjects are split by NWB file parent directory / filename convention. The subject id is taken from the `sub-<mouse>` directory name, and unique subjects are accumulated in first-seen session order.

ii.
```python
def parse_identifier(identifier: str, file_path: Path) -> SessionMeta:
    ...
    subject = file_path.parent.name.replace("sub-", "")
```

```python
if meta.subject not in subject_to_idx:
    subject_to_idx[meta.subject] = len(subjects)
    subjects.append(meta.subject)
subject_idx.append(subject_to_idx[meta.subject])
```

iii. The notes say `/app/data` is organized as one folder per subject and that this matches the 11-mouse cohort described in the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session metadata are parsed from the file stem and filename session number.

ii.
```python
session_match = re.search(r"_ses-(\d+)_", file_path.name)
session_id = session_match.group(1) if session_match else "unknown"
...
session_name=file_path.stem.replace("_behavior+ophys", "")
```

```python
for session_idx, file_path in enumerate(files):
    neural_trials, input_trials, output_trials, plane_per_cell, meta, stats = convert_session(file_path)
```

iii. The notes explicitly state that `/app/data` contains one NWB file per session and that filenames follow `sub-<mouse>_ses-<NN>_behavior+ophys.nwb`.

## 1-d. How are the data split into trials?

i. Trials are defined from the `trial_start` and `teleport` behavior pulse streams. The script takes every positive `trial_start` sample as a start, then pairs it with the next positive `teleport` sample after that start. Trial slices are `[start, stop)`.

ii.
```python
def pair_trial_bounds(trial_start: np.ndarray, teleport: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(trial_start > 0)
    stops = np.flatnonzero(teleport > 0)
    bounds: list[tuple[int, int]] = []
    stop_idx = 0
    for start in starts:
        while stop_idx < len(stops) and stops[stop_idx] <= start:
            stop_idx += 1
        if stop_idx >= len(stops):
            break
        stop = int(stops[stop_idx])
        bounds.append((int(start), stop))
        stop_idx += 1
    return bounds
```

```python
trial_bounds = pair_trial_bounds(trial_start, teleport)
...
for trial_idx, (start, stop) in enumerate(trial_bounds):
```

iii. The notes say the agent checked the one session with a `trial number` mismatch and concluded the extra labeled segment was only a tunnel/post-teleport fragment, so the pulse streams were treated as authoritative.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit minimum-length trial filter. Trials are dropped only if `stop <= start`, if a bound runs past the neural array, or if the trial contains no nonnegative environment samples. Sessions are rejected if fewer than 2 usable trials remain.

ii.
```python
for trial_idx, (start, stop) in enumerate(trial_bounds):
    if stop <= start:
        continue
    if stop > deconv.shape[0]:
        raise ValueError(f"{meta.session_name}: trial stop exceeds neural data length")

    env_trial = environment[start:stop]
    env_valid = env_trial[env_trial >= 0]
    if env_valid.size == 0:
        continue
```

```python
if len(neural_trials) < 2:
    raise ValueError(f"{meta.session_name}: fewer than 2 usable trials after conversion")
```

iii. In the notes, the agent emphasizes excluding pre-sync sentinel samples by working only within pulse-defined trial windows, but it does not justify a separate short-trial exclusion rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are taken directly from `processing/ophys/Deconvolved`, with cell selection from `ImageSegmentation/PlaneSegmentation` via `iscell` and `planeIdx`.

ii.
```python
seg = nwb.processing["ophys"]["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]
iscell = np.asarray(seg["iscell"].data[:] if hasattr(seg["iscell"], "data") else seg["iscell"][:])
plane_idx = np.asarray(seg["planeIdx"].data[:] if hasattr(seg["planeIdx"], "data") else seg["planeIdx"][:]).astype(int)
keep = iscell[:, 0] == 1

deconv_mod = nwb.processing["ophys"]["Deconvolved"]
```

iii. The notes state that the agent deliberately chose stored deconvolved calcium events because it believed this matched the reference decoder notebook’s use of `events`.

## 2-b. How is the `neural` data processed?

i. The script sorts plane names, subsets to curated ROIs, concatenates planes, trims a single trailing neural frame if behavior is shorter by exactly one frame, slices the trial window, replaces NaN/inf with zero, transposes to `(neurons, time)`, and casts to `float16`. It does not recompute dF/F or deconvolution.

ii.
```python
plane_names = sorted(deconv_mod.roi_response_series.keys(), key=get_plane_number)
...
plane_data = np.asarray(rs.data[:, plane_keep_local], dtype=np.float32)
arrays.append(plane_data)
...
deconv = np.concatenate(arrays, axis=1)
```

```python
frame_delta = deconv.shape[0] - frame_times.shape[0]
if frame_delta == 1:
    deconv = deconv[: frame_times.shape[0], :]
...
neural_trial = np.nan_to_num(deconv[start:stop, :].T, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16, copy=False)
```

iii. The notes justify this as using the aligned NWB event representation directly and as a storage optimization (`float16`) that keeps the output pickle smaller.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural filtering is limited to keeping Suite2p/manual curated ROIs where `iscell[:,0] == 1`. No speed-correlation interneuron exclusion is applied.

ii.
```python
iscell = np.asarray(seg["iscell"].data[:] if hasattr(seg["iscell"], "data") else seg["iscell"][:])
...
keep = iscell[:, 0] == 1
```

iii. In Step 10 of the notes, the agent explicitly says it chose not to apply the later speed-correlation interneuron exclusion because the decoder task here is session-wide and the converted counts/verification looked internally consistent.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start by slicing each trial as `[start, stop)` using the `trial_start` pulse-defined bounds. No further temporal shift is applied.

ii.
```python
trial_bounds = pair_trial_bounds(trial_start, teleport)
...
neural_trial = np.nan_to_num(deconv[start:stop, :].T, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16, copy=False)
```

iii. The notes say pulse-defined windows are the NWB analogue of the reference code’s `trial_start_inds` / `teleport_inds`, so starting each slice at `trial_start` was taken as the requested alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data keep the native frame-wise sampling at about `1 / 15.5078125 s` per sample (`~64.48 ms`). No rebinning or resampling is applied.

ii.
```python
EXPECTED_FRAME_DT_S = 1.0 / 15.5078125
...
frame_dt_s = float(np.median(np.diff(frame_times)))
if not np.isclose(frame_dt_s, EXPECTED_FRAME_DT_S, atol=1e-9, rtol=0):
    raise ValueError(...)
```

```python
if time_bin_ms is None:
    time_bin_ms = float(stats["frame_dt_s"]) * 1000.0
```

iii. The notes say the behavior-frame timestamps are the authoritative common time base and that no extra temporal resampling is needed.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavior frame timestamps, specifically `position.timestamps`, which the script treats as the common aligned frame clock.

ii.
```python
frame_times = np.asarray(beh.time_series["position"].timestamps[:], dtype=np.float64)
...
trial_time_s = (frame_times[start:stop] - frame_times[start]).astype(np.float32, copy=False)
```

iii. The notes justify this by saying the behavior streams share a common frame grid and that these timestamps are the authoritative time base.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the timestamp at the trial start frame is subtracted from every frame timestamp in that trial.

ii.
```python
trial_time_s = (frame_times[start:stop] - frame_times[start]).astype(np.float32, copy=False)
```

iii. The notes describe this as the direct framewise time-from-trial-start input required by the decoder.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses the same trial slice indices as the neural data, so each time sample lines up one-for-one with the neural frame in the same `[start, stop)` window.

ii.
```python
neural_trial = np.nan_to_num(deconv[start:stop, :].T, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16, copy=False)
trial_time_s = (frame_times[start:stop] - frame_times[start]).astype(np.float32, copy=False)
```

iii. The notes say neural and behavior are already frame-aligned in NWB; the only special case handled is trimming one trailing neural frame in some multi-plane sessions.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavior `environment` time series.

ii.
```python
environment = np.asarray(beh.time_series["environment"].data[:], dtype=np.float32)
...
env_trial = environment[start:stop]
```

iii. The notes say the valid environment codes observed in the NWB files are `0` and `1`, matching the two environments.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial, the script drops negative sentinel values, takes the integer mode of the remaining environment values, and then broadcasts that single code across all timepoints in the trial.

ii.
```python
def mode_int(values: np.ndarray) -> int:
    uniq, counts = np.unique(values.astype(int), return_counts=True)
    return int(uniq[np.argmax(counts)])
```

```python
env_valid = env_trial[env_trial >= 0]
...
env_code = float(mode_int(env_valid))
...
np.full(trial_time_s.shape, env_code, dtype=np.float32)
```

iii. The notes say the agent wanted to use the aligned environment stream rather than infer environment only from the scene name, while also making the per-trial input constant over time.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number comes from the behavior `trial number` time series, but the trial boundaries themselves come from `trial_start` and `teleport`.

ii.
```python
trial_start = np.asarray(beh.time_series["trial_start"].data[:], dtype=np.float32)
teleport = np.asarray(beh.time_series["teleport"].data[:], dtype=np.float32)
trial_number = np.asarray(beh.time_series["trial number"].data[:], dtype=np.float32)
```

```python
trial_num_native = int(round(float(trial_number[start])))
```

iii. The notes say the pulse streams are authoritative for trial structure, but the agent still uses the native `trial number` label at each trial start when it is valid.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The script reads the `trial number` value at the trial start sample, rounds it to an integer, falls back to the loop index if the value is negative, and broadcasts that number across the trial.

ii.
```python
trial_num_native = int(round(float(trial_number[start])))
if trial_num_native < 0:
    trial_num_native = trial_idx
...
np.full(trial_time_s.shape, float(trial_num_native), dtype=np.float32)
```

iii. The notes justify the fallback as protection against pre-sync sentinel values while keeping the native completed-trial numbering when present.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from reward delivery timestamps in the behavior `Reward` series, rasterized onto the frame grid and summarized per trial.

ii.
```python
reward_times = np.asarray(beh.time_series["Reward"].timestamps[:], dtype=np.float64)
reward_frames = rasterize_reward_events(frame_times, reward_times)
```

```python
reward_outcomes = trial_reward_outcomes(trial_bounds, reward_frames)
```

iii. The notes say reward outcome is rebuilt from raw reward events because NWB stores reward as timestamped events rather than as an already frame-aligned binary series.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The script first computes a binary reward outcome for each trial (`any reward frame in that trial`). It then shifts that vector by one trial, filling the first trial with 0, and repeats the value across every timepoint of the current trial.

ii.
```python
def trial_reward_outcomes(bounds: list[tuple[int, int]], reward_frames: np.ndarray) -> np.ndarray:
    outcomes = np.zeros(len(bounds), dtype=np.int8)
    for idx, (start, stop) in enumerate(bounds):
        outcomes[idx] = int(np.any(reward_frames[start:stop] > 0))
    return outcomes
```

```python
prev_reward_outcomes = np.zeros_like(reward_outcomes)
if reward_outcomes.size > 1:
    prev_reward_outcomes[1:] = reward_outcomes[:-1]
...
np.full(trial_time_s.shape, float(prev_reward_outcomes[trial_idx]), dtype=np.float32)
```

iii. The notes describe this as the decoder-required previous-trial binary label, with the first trial conventionally set to 0.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the behavior `position` time series plus a reward-zone label inferred from the session `identifier` scene string, mapped onto fixed reward-zone coordinate intervals `A/B/C`.

ii.
```python
position = np.asarray(beh.time_series["position"].data[:], dtype=np.float32)
...
reward_labels = scene_to_reward_labels(meta.scene, len(trial_bounds))
...
rz_label = reward_labels[trial_idx]
rz_start, rz_stop = REWARD_ZONE_COORDS_CM[rz_label]
rz_dist = signed_distance_to_interval(pos_trial, rz_start, rz_stop)
```

iii. The notes say the NWB export does not store categorical `A/B/C` labels directly, so the agent reconstructed them from `nwb.identifier` using the scene logic from the paper’s code.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each trial frame, the script computes signed distance to the active reward-zone interval: negative before the zone, zero inside it, positive after it. That continuous value is then discretized.

ii.
```python
def signed_distance_to_interval(position_cm: np.ndarray, start_cm: float, stop_cm: float) -> np.ndarray:
    out = np.zeros_like(position_cm, dtype=np.float32)
    below = position_cm < start_cm
    above = position_cm > stop_cm
    out[below] = position_cm[below] - start_cm
    out[above] = position_cm[above] - stop_cm
    return out
```

iii. In the notes, the agent says it intentionally used distance to the full reward-zone interval, not just the zone start, so all in-zone samples become exactly zero distance.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The signed distance is thresholded into seven categories with explicit comparisons implementing the bins `<-50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, and `>50`.

ii.
```python
def discretize_reward_distance(distance_cm: np.ndarray) -> np.ndarray:
    out = np.full(distance_cm.shape, 6, dtype=np.int8)
    out[distance_cm < -50.0] = 0
    out[(distance_cm >= -50.0) & (distance_cm < -10.0)] = 1
    out[(distance_cm >= -10.0) & (distance_cm < 0.0)] = 2
    out[distance_cm == 0.0] = 3
    out[(distance_cm > 0.0) & (distance_cm <= 10.0)] = 4
    out[(distance_cm > 10.0) & (distance_cm <= 50.0)] = 5
    out[distance_cm > 50.0] = 6
    return out
```

iii. The notes record these exact bin rules as the intended output convention for reward-distance decoding.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by slicing `position[start:stop]` over the same trial bounds used for `deconv[start:stop, :]`.

ii.
```python
neural_trial = np.nan_to_num(deconv[start:stop, :].T, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16, copy=False)
pos_trial = np.nan_to_num(position[start:stop], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
```

iii. The notes say behavior and neural streams are already aligned on the same frame grid in NWB, so using the same `[start, stop)` window is sufficient.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavior `position` time series.

ii.
```python
position = np.asarray(beh.time_series["position"].data[:], dtype=np.float32)
...
pos_trial = np.nan_to_num(position[start:stop], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
```

iii. The notes treat `position` as the aligned per-frame VR corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The per-trial `position` slice is NaN/inf-cleaned with `np.nan_to_num` and then discretized into five bins. There is no additional smoothing or interpolation.

ii.
```python
pos_trial = np.nan_to_num(position[start:stop], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
...
discretize_position(pos_trial)
```

iii. The notes frame this as a direct discretization of the aligned behavior stream into the task-required position classes.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The script uses five bins implemented as `<90`, `[90,180)`, `[180,270)`, `[270,360)`, and `>=360`.

ii.
```python
def discretize_position(position_cm: np.ndarray) -> np.ndarray:
    out = np.full(position_cm.shape, 4, dtype=np.int8)
    out[position_cm < 90.0] = 0
    out[(position_cm >= 90.0) & (position_cm < 180.0)] = 1
    out[(position_cm >= 180.0) & (position_cm < 270.0)] = 2
    out[(position_cm >= 270.0) & (position_cm < 360.0)] = 3
    out[position_cm >= 360.0] = 4
    return out
```

iii. The notes record those same five 90 cm-wide bins as the chosen absolute-position output convention.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same trial bounds and therefore the same sample indices as the neural trial.

ii.
```python
neural_trial = np.nan_to_num(deconv[start:stop, :].T, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16, copy=False)
pos_trial = np.nan_to_num(position[start:stop], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
```

iii. The notes repeatedly state that the NWB behavior streams are already synchronized to imaging frames.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavior `lick` time series.

ii.
```python
lick_counts = np.asarray(beh.time_series["lick"].data[:], dtype=np.float32)
...
lick_binary, bad_lick = correct_lick_trial(lick_counts[start:stop])
```

iii. The notes describe the raw lick channel as a per-frame count-like stream that is later converted to a binary decoder target.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The script first applies a trial-level artifact rule: if more than 35% of frames in the trial have lick counts greater than 2, the whole trial is zeroed out. It then binarizes the remaining lick counts as `count > 0`.

ii.
```python
LICK_SENSOR_COUNT_THRESHOLD = 2.0
LICK_SENSOR_BAD_FRAC = 0.35
```

```python
def correct_lick_trial(lick_counts: np.ndarray) -> tuple[np.ndarray, bool]:
    corrected = lick_counts.copy()
    bad_trial = float(np.mean(corrected > LICK_SENSOR_COUNT_THRESHOLD)) > LICK_SENSOR_BAD_FRAC
    if bad_trial:
        corrected[:] = 0.0
    corrected = (corrected > 0).astype(np.int8)
    return corrected, bad_trial
```

iii. The notes say this was an intentional “reference heuristic” for obvious lick-sensor failures and that the agent spot-checked corrected trials in Step 12.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick output is aligned by slicing the same `[start, stop)` trial window used for the neural data.

ii.
```python
lick_binary, bad_lick = correct_lick_trial(lick_counts[start:stop])
...
neural_trial = np.nan_to_num(deconv[start:stop, :].T, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16, copy=False)
```

iii. The notes state that lick, position, and other behavior streams are already frame-aligned to imaging in NWB.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from session metadata, specifically the `nwb.identifier` scene string parsed into `meta.scene`, plus the trial count within the session.

ii.
```python
meta = parse_identifier(nwb.identifier, file_path)
...
reward_labels = scene_to_reward_labels(meta.scene, len(trial_bounds))
reward_codes = np.array([reward_label_to_code(label) for label in reward_labels], dtype=np.int8)
```

iii. The notes say the NWB export does not provide direct categorical `A/B/C` zone labels, so the agent reconstructs them from the scene name using the reference session logic.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Fixed-zone scenes map every trial to one label. Switch scenes map the first 30 trials to the pre-switch zone and the remaining trials to the post-switch zone. The label is converted to `0/1/2` and then repeated across all frames of the trial.

ii.
```python
def scene_to_reward_labels(scene: str, ntrials: int, change_trial: int = SCENE_SWITCH_TRIAL) -> list[str]:
    if scene in {"Env1_LocationA", "Env2_LocationA", "Env3_LocationA"}:
        return ["A"] * ntrials
    ...
    if "A_to" in scene and scene.endswith("B"):
        return ["A"] * first_n + ["B"] * second_n
    ...
```

```python
np.full(trial_time_s.shape, reward_codes[trial_idx], dtype=np.int8)
```

iii. The notes justify this as matching the paper’s switch schedule and as simpler than inferring zone identity from the noisy `reward_zone` behavior stream.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from reward delivery timestamps in the behavior `Reward` series.

ii.
```python
reward_times = np.asarray(beh.time_series["Reward"].timestamps[:], dtype=np.float64)
reward_frames = rasterize_reward_events(frame_times, reward_times)
```

iii. The notes describe `Reward` as an event series that must be rasterized back onto the aligned frame grid before trial-level outcomes can be computed.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are rasterized to frame indices with `np.searchsorted`, then each trial gets a binary `any reward in [start, stop)` label, which is repeated across all frames in the trial.

ii.
```python
def rasterize_reward_events(frame_times: np.ndarray, reward_times: np.ndarray) -> np.ndarray:
    reward_frames = np.zeros(frame_times.shape[0], dtype=np.int8)
    ...
    reward_indices = np.searchsorted(frame_times, reward_times)
    ...
    reward_frames[reward_indices] = 1
    return reward_frames
```

```python
reward_outcomes = trial_reward_outcomes(trial_bounds, reward_frames)
...
np.full(trial_time_s.shape, reward_outcomes[trial_idx], dtype=np.int8)
```

iii. The notes justify this as the decoder-required per-trial rewarded/omitted target and report spot checks on rewarded and omission trials.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles several edge cases defensively:
- If neural data are exactly one frame longer than behavior, it trims the final neural frame; any other mismatch raises an error.
- If a trial has invalid bounds or no nonnegative environment values, it is skipped.
- NaN and infinite values in neural, position, and speed slices are replaced with zero.
- Obvious lick-sensor artifact trials are zeroed by `correct_lick_trial`.
- Reward timestamps must align exactly to frame timestamps or the script raises an error.
- Sessions with fewer than 2 usable trials raise an error.

ii.
```python
frame_delta = deconv.shape[0] - frame_times.shape[0]
if frame_delta == 1:
    deconv = deconv[: frame_times.shape[0], :]
elif frame_delta != 0:
    raise ValueError(...)
```

```python
if env_valid.size == 0:
    continue
...
neural_trial = np.nan_to_num(...)
pos_trial = np.nan_to_num(...)
speed_trial = np.nan_to_num(...)
```

iii. These behaviors are justified in Steps 10 and 12 of the notes, where the agent says it found a recurring one-frame multi-plane mismatch and some lick-sensor artifact trials during validation.

## 13-a. What are the most time-consuming steps of the code?

i. The likely slowest steps are reading NWB sessions, loading/concatenating large deconvolved arrays, iterating through all trials in `convert_session`, writing the multi-gigabyte pickle, and optionally reopening sessions to make processing plots.

ii.
```python
with NWBHDF5IO(str(file_path), "r", load_namespaces=True) as io:
    nwb = io.read()
```

```python
for trial_idx, (start, stop) in enumerate(trial_bounds):
    ...
```

```python
with out_path.open("wb") as f:
    pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes explicitly call out large-session conversion, file size, and the cost of rereading sessions for plotting as the main runtime drivers.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The main per-trial construction loop in `convert_session`, the per-trial reward summarization loop in `trial_reward_outcomes`, and the plot-only lick-correction loop could all be vectorized or restructured if needed. The plane concatenation loop could also be batched more aggressively.

ii.
```python
for idx, (start, stop) in enumerate(bounds):
    outcomes[idx] = int(np.any(reward_frames[start:stop] > 0))
```

```python
for trial_idx, (start, stop) in enumerate(trial_bounds):
    ...
```

```python
for start, stop in pair_trial_bounds(trial_start, teleport):
    corrected_full_lick[start:stop], _ = correct_lick_trial(lick_counts[start:stop])
```

iii. The notes mention that trial-wise organization was chosen for simplicity, but they also acknowledge some avoidable repeated work in plot generation.

## 13-c. What processing does the code repeat multiple times?

i. The code repeats several pieces of work:
- It reopens each NWB file when `--show-processing` is enabled.
- It recomputes `pair_trial_bounds(...)` and `scene_to_reward_labels(...)` multiple times inside the plotting path.
- It recomputes lick correction over the full session for plotting even after per-trial lick outputs already exist.

ii.
```python
if show_processing and session_idx < 2:
    with NWBHDF5IO(str(file_path), "r", load_namespaces=True) as io:
        nwb = io.read()
        ...
```

```python
trial_bounds=pair_trial_bounds(trial_start, teleport),
reward_zone_labels=scene_to_reward_labels(meta.scene, len(pair_trial_bounds(trial_start, teleport))),
reward_zone_codes=np.array([reward_label_to_code(label) for label in scene_to_reward_labels(meta.scene, len(pair_trial_bounds(trial_start, teleport)))], dtype=np.int8),
```

iii. The notes describe plot rereads as a known inefficiency and intentionally limit plotting to at most two sessions.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does a few pieces of work that are not used by the downstream decoder:
- `plane_per_cell` is computed and returned from `convert_session`, but `build_dataset` discards it and writes all-zero `brain_region_idx` instead.
- `full_lick` is concatenated but never used.
- `trial_reward` is assigned inside plotting but never used.
- Optional plotting and some rich metadata are generated for inspection only, not for decoder training.

ii.
```python
deconv, plane_per_cell, n_planes = load_curated_deconvolved(nwb)
...
return neural_trials, input_trials, output_trials, plane_per_cell, meta, stats
```

```python
plane_all.append(session_brain_region_idx(neural_trials[0].shape[0]))
```

```python
full_lick = np.concatenate([trial[3] for trial in output_trials]).astype(np.int8)
...
trial_reward = output_trials[0][5, 0:1]
```

iii. The notes justify some of this as inspection/debug support, but the final dataset consumed by the decoder does not use these intermediate values.
