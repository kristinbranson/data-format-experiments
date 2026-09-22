# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script globbed every `sub-*/sub-*_behavior+ophys.nwb` file under `/app/data`, sorted them, and processed each file as one session. Within each session it read behavior and ophys arrays directly from the NWB HDF5 structure with `h5py`, rather than using `pynwb`.

ii. 
```python
all_files = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))
selected_files = all_files if args.mode == "full" else select_sample_files(all_files)
...
with h5py.File(nwb_path, "r") as h5:
    beh_root = h5["processing/behavior/BehavioralTimeSeries"]
    ...
    deconv_root = h5["processing/ophys/Deconvolved"]
```

iii. In `CONVERSION_NOTES.md` Steps 4-6, the AI says the NWB release should be treated as the authoritative serialized dataset, and that direct `h5py` access was chosen for speed and bounded memory use during full-dataset conversion.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from each NWB file’s `general/subject/subject_id`. The exported `subjects` list is the sorted set of subject IDs seen across converted sessions, and `subject_idx` maps each session back to that list.

ii. 
```python
with h5py.File(nwb_path, "r") as h5:
    subject_id = h5["general/subject/subject_id"][()].decode()
...
subjects = sorted({sess["subject_id"] for sess in converted_sessions})
subject_lookup = {subject: i for i, subject in enumerate(subjects)}
...
"subject_idx": np.asarray(
    [subject_lookup[sess["subject_id"]] for sess in converted_sessions],
    dtype=np.int16,
),
```

iii. The notes say the NWB subject IDs (`m3`, `m4`, etc.) are the canonical mouse identifiers present in the shared release, so the AI used them directly rather than inferring subjects from directory names only.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session-level metadata are looked up by `(GCAMP animal, experiment day)` using `sessions_dict.py`, and the converted dataset keeps one top-level session entry per input file.

ii. 
```python
def load_sessions_dict() -> dict[tuple[str, int], SessionMeta]:
    ...

for file_idx, nwb_path in enumerate(selected_files, start=1):
    converted, summary = convert_session(nwb_path, sessions_dict, do_plot)
    converted_sessions.append(converted)
    session_summaries.append(summary)
```

iii. In Step 5 of the notes, the AI states that the NWB release contains one `behavior+ophys.nwb` file per subject-session and that all 152 files should be kept as candidate sessions.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from framewise `trial_start` and `teleport` pulses, restricted to samples where `scanning > 0`. For each detected trial, the exported trial window is `start:end`, so the teleport frame itself is excluded.

ii. 
```python
def pair_trial_events(trial_start: np.ndarray, teleport: np.ndarray, scanning: np.ndarray) -> list[tuple[int, int]]:
    start_idx = np.where((trial_start > 0) & (scanning > 0))[0]
    end_idx = np.where((teleport > 0) & (scanning > 0))[0]
    ...
    pairs.append((int(start), int(end)))

...
trial_pairs = pair_trial_events(trial_start, teleport, scanning)
...
trial_slice = slice(start, end)  # exclude teleport frame itself
```

iii. The notes say the reference code defines trials from explicit start/end event streams and that trial-aligned matrices should include linear-track frames only, not the teleport frame or negative-position tunnel frames.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops trials flagged as lick-sensor failures, defined as trials where more than 30% of samples have cumulative lick count `> 2`. It also skips empty trials and raises an error if a session ends up with fewer than 2 valid trials. Unlike the human reference solution, it does not apply a `< 50` timepoint minimum-trial-length filter.

ii. 
```python
def detect_bad_lick_trials(lick: np.ndarray, trial_pairs: list[tuple[int, int]]) -> np.ndarray:
    ...
    bad[i] = np.mean(trial_lick > 2) > LICK_ERROR_THRESHOLD

...
for trial_idx, ((start, end), bad_lick, reward_label) in enumerate(
    zip(trial_pairs, bad_lick_trials, reward_labels_all, strict=True)
):
    if bad_lick:
        continue
    ...
    if pos_trial.size == 0:
        continue

if len(neural_trials) < 2:
    raise RuntimeError(f"{session_tag}: fewer than 2 valid trials after filtering")
```

iii. In Steps 5, 9, and 10 of the notes, the AI says it intentionally matched the manuscript’s lick-QC rule because `lick` is a decoder target, and it highlights that this reproduces the paper’s reported removal of 81 bad-lick trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The exported `neural` matrices are taken directly from the NWB `processing/ophys/Deconvolved/plane*` datasets after ROI filtering and pooling across planes. The AI does not recompute the paper’s event signal from raw fluorescence and neuropil.

ii. 
```python
seg = h5["processing/ophys/ImageSegmentation/PlaneSegmentation"]
iscell = seg["iscell"][()]
plane_idx = seg["planeIdx"][()].astype(np.int16)
...
deconv_root = h5["processing/ophys/Deconvolved"]
...
plane_data = deconv_root[f"plane{int(plane)}"]["data"][:min_frames, local_cols]
deconv[:, kept_positions] = plane_data.astype(np.float16)
```

iii. In Steps 4-6 and 10 of the notes, the AI argues that NWB `Deconvolved` is the closest available representation to the paper’s event-like signal and avoids reimplementing dF/F and OASIS deconvolution.

## 2-b. How is the `neural` data processed?

i. Processing is limited to trimming all streams to a common frame count, filtering ROIs by `iscell`, pooling planes into one neuron axis, slicing trials, transposing to `(n_neurons, T)`, and casting to `float16`. The script does not perform neuropil subtraction, baseline estimation, dF/F computation, smoothing, or deconvolution.

ii. 
```python
min_frames = min([frame_timestamps.shape[0], *plane_frame_counts])
...
keep_mask = iscell[:, 0] > 0.5
...
deconv = np.empty((min_frames, keep_indices.size), dtype=np.float16)
...
neural_trial = np.ascontiguousarray(deconv[trial_slice, :].T, dtype=np.float16)
```

iii. The notes describe this as a pragmatic choice to keep the full export tractable and to use the stored deconvolved signal rather than reconstructing the manuscript pipeline from `Fluorescence` and `Neuropil`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural QC filter in the code is `iscell[:,0] > 0.5`, i.e. keeping suite2p-curated ROIs. The AI does not implement the additional putative-interneuron exclusion used in the human reference solution.

ii. 
```python
iscell = seg["iscell"][()]
plane_idx = seg["planeIdx"][()].astype(np.int16)
keep_mask = iscell[:, 0] > 0.5
keep_indices = np.flatnonzero(keep_mask)
```

iii. In Steps 5 and 10, the AI says it considered adding the paper’s speed-correlation interneuron filter, but omitted it after checking that the NWB release’s upper-tail cell counts could not plausibly be explained by that small exclusion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to the trial start event simply by slicing each trial from the `trial_start` frame onward. The first sample in each exported neural trial is therefore the `trial_start` frame.

ii. 
```python
trial_pairs = pair_trial_events(trial_start, teleport, scanning)
...
trial_slice = slice(start, end)  # exclude teleport frame itself
...
neural_trial = np.ascontiguousarray(deconv[trial_slice, :].T, dtype=np.float16)
```

iii. The notes explicitly say the decoder export should be temporally aligned to trial start and that no extra offsetting is needed once trials are reconstructed from the aligned framewise streams.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI assumes a common imaging-frame spacing of `1 / 15.5078125 s` and uses the behavior timestamps as the master temporal grid. It does not apply any explicit temporal rebinning or downsampling in the conversion code.

ii. 
```python
COMMON_FRAME_RATE_HZ = 15.5078125
COMMON_FRAME_DT_S = 1.0 / COMMON_FRAME_RATE_HZ
TIME_BIN_SIZE_MS = COMMON_FRAME_DT_S * 1000.0
...
frame_dt = median_step_seconds(frame_timestamps)
if not np.isclose(frame_dt, COMMON_FRAME_DT_S, atol=1e-6):
    raise RuntimeError(f"{session_tag}: unexpected frame dt {frame_dt}")
```

iii. In Steps 4, 5, and 10, the AI says the behavior timestamps are already on the effective per-plane imaging grid for all sessions, so the NWB `rate` attribute was ignored and no resampling was needed.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the behavior timestamps attached to the `position` time series. The AI treats those timestamps as the session’s master clock.

ii. 
```python
frame_timestamps = beh_root["position"]["timestamps"][()].astype(np.float64)
...
t_trial = frame_timestamps[trial_slice] - frame_timestamps[start]
```

iii. The notes say all framewise behavior streams are synchronized to imaging and that one imaging frame should be the canonical time grid, so any of the aligned behavior timestamp arrays would have worked.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the code subtracts the trial’s first timestamp from every timestamp in the trial slice.

ii. 
```python
t_trial = frame_timestamps[trial_slice] - frame_timestamps[start]
...
input_trial = np.ascontiguousarray(
    np.vstack(
        [
            t_trial.astype(np.float32),
            ...
        ]
    ),
    dtype=np.float32,
)
```

iii. The notes describe this as the natural trial-aligned continuous time variable required by the decoder task.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is sliced on exactly the same `trial_slice` as the neural data, so its timepoints are one-to-one with the neural frames in each trial.

ii. 
```python
trial_slice = slice(start, end)
...
t_trial = frame_timestamps[trial_slice] - frame_timestamps[start]
...
neural_trial = np.ascontiguousarray(deconv[trial_slice, :].T, dtype=np.float16)
```

iii. The AI repeatedly states in the notes that behavior is already frame-synchronized to imaging in the NWB data, so shared slicing is sufficient for alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the framewise `environment` behavior stream.

ii. 
```python
environment = beh_root["environment"]["data"][()].astype(np.float32)
...
env_trial_vals = environment[trial_slice]
```

iii. The notes identify `environment` as the synchronized NWB equivalent of the paper code’s binary environment/morph variable.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The code removes any negative values inside the slice, takes the median environment code across the trial, rounds it to an integer, and repeats that scalar across all time bins in the trial.

ii. 
```python
env_trial_vals = environment[trial_slice]
env_trial_vals = env_trial_vals[env_trial_vals >= 0]
...
env_trial = int(np.rint(np.median(env_trial_vals)))
...
repeated_row(env_trial, T, np.float32),
```

iii. The notes say environment identity is constant within a trial and should be exported as a per-trial decoder input repeated across time for a uniform `(d, T)` format.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is taken from the framewise `trial number` behavior stream at the trial’s start frame.

ii. 
```python
trial_number = beh_root["trial number"]["data"][()].astype(np.int32)
...
trial_num = int(trial_number[start])
```

iii. The notes say the NWB trial-number stream looked usable after pairing explicit trial boundaries, so the AI kept the stored session-local trial IDs instead of recomputing them from the loop index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The selected scalar trial number is repeated across every timepoint in the trial.

ii. 
```python
trial_num = int(trial_number[start])
...
repeated_row(trial_num, T, np.float32),
```

iii. The notes describe per-trial decoder variables as being repeated across time so that every trial has a uniform matrix-shaped input representation.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the sparse `Reward` event timestamps, combined with the reconstructed trial start and end times. The code first computes a current-trial reward outcome for every trial, then shifts that vector by one trial.

ii. 
```python
reward_event_ts = beh_root["Reward"]["timestamps"][()].astype(np.float64)
...
for i, (start, end) in enumerate(trial_pairs):
    start_ts = frame_timestamps[start]
    end_ts = frame_timestamps[end]
    reward_outcomes_all[i] = int(np.any((reward_event_ts >= start_ts) & (reward_event_ts < end_ts)))
...
prev_reward = int(reward_outcomes_all[trial_idx - 1]) if trial_idx > 0 else 0
```

iii. The notes say reward outcome should be computed from actual reward events, not from trial metadata, and then shifted to meet the decoder’s “previous trial outcome” definition.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The code marks each trial as rewarded if any reward event timestamp falls in that trial’s `[start_ts, end_ts)` window. It then assigns trial 0 a previous-outcome value of 0 and repeats the previous trial’s reward outcome across all time bins of each later trial.

ii. 
```python
reward_outcomes_all[i] = int(np.any((reward_event_ts >= start_ts) & (reward_event_ts < end_ts)))
...
prev_reward = int(reward_outcomes_all[trial_idx - 1]) if trial_idx > 0 else 0
...
repeated_row(prev_reward, T, np.float32),
```

iii. In the notes, the AI says this matches the task’s binary omitted-versus-rewarded definition while preserving trial-level semantics in a time-varying export format.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the framewise `position` stream and a reward-zone interval inferred from session metadata (`scene` in `sessions_dict.py`), not from the framewise NWB `reward_zone` occupancy signal.

ii. 
```python
session_meta = sessions_dict[(gcamp, exp_day)]
...
reward_labels_all = scene_reward_labels(session_meta.scene, len(trial_pairs))
...
pos_trial = position[trial_slice]
zone_start, zone_end = REWARD_ZONE_COORDS[reward_label]
reward_dist = reward_distance_to_zone(pos_trial, zone_start, zone_end)
```

iii. The notes explicitly justify this as following the paper code’s scene-based reward-zone logic more directly than trying to infer A/B/C from the noisy framewise `reward_zone` occupancy values.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the code computes signed distance to the nearest point in the active reward-zone interval: negative before the zone, zero inside it, positive after it.

ii. 
```python
def reward_distance_to_zone(position_cm: np.ndarray, zone_start: float, zone_end: float) -> np.ndarray:
    before = position_cm < zone_start
    after = position_cm > zone_end
    dist = np.zeros(position_cm.shape, dtype=np.float32)
    dist[before] = position_cm[before] - zone_start
    dist[after] = position_cm[after] - zone_end
    return dist
```

iii. The notes say this implements “distance to any location in the reward zone” while preserving the paper’s 50 cm reward-zone geometry.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The signed distance is discretized into 7 categories using the instructed cut points: `< -50`, `[-50, -10)`, `[-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50`.

ii. 
```python
def discretize_reward_distance(distance_cm: np.ndarray) -> np.ndarray:
    out = np.full(distance_cm.shape, 6, dtype=np.int16)
    out[distance_cm < -50] = 0
    out[(distance_cm >= -50) & (distance_cm < -10)] = 1
    out[(distance_cm >= -10) & (distance_cm < 0)] = 2
    out[distance_cm == 0] = 3
    out[(distance_cm > 0) & (distance_cm <= 10)] = 4
    out[(distance_cm > 10) & (distance_cm <= 50)] = 5
    return out
```

iii. The notes describe this as a direct implementation of the task specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The distance is computed from `position[trial_slice]`, using the same per-trial slice as the neural matrix, so the distance category at each column corresponds to the same frame as the neural activity.

ii. 
```python
trial_slice = slice(start, end)
pos_trial = position[trial_slice]
...
reward_dist_bin = discretize_reward_distance(reward_dist)
...
neural_trial = np.ascontiguousarray(deconv[trial_slice, :].T, dtype=np.float16)
```

iii. The notes say all framewise variables share the same imaging-aligned clock, so common slicing provides the alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the framewise `position` behavior stream.

ii. 
```python
position = beh_root["position"]["data"][()].astype(np.float32)
...
pos_trial = position[trial_slice]
```

iii. The notes identify the synchronized `position` stream as the direct virtual-track coordinate in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The code slices trial-local position values and discretizes them into 5 coarse track-position bins spanning the 450 cm corridor.

ii. 
```python
pos_trial = position[trial_slice]
pos_bin = discretize_absolute_position(pos_trial)
```

iii. The notes say the task asks for categorical absolute position rather than the manuscript’s reward-relative position, so the AI used a direct 5-bin discretization over the track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The code uses thresholds at 90, 180, 270, and 360 cm. Values `< 90` are bin 0; `[90,180)` bin 1; `[180,270)` bin 2; `[270,360]` bin 3; and `> 360` bin 4.

ii. 
```python
def discretize_absolute_position(position_cm: np.ndarray) -> np.ndarray:
    out = np.zeros(position_cm.shape, dtype=np.int16)
    out[(position_cm >= 90) & (position_cm < 180)] = 1
    out[(position_cm >= 180) & (position_cm < 270)] = 2
    out[(position_cm >= 270) & (position_cm <= 360)] = 3
    out[position_cm > 360] = 4
    return out
```

iii. The notes treat this as the direct implementation of the user’s requested 5 equal-width bins over a 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is computed from `position[trial_slice]`, so each position category is aligned frame-for-frame with the neural trial matrix.

ii. 
```python
trial_slice = slice(start, end)
pos_trial = position[trial_slice]
...
neural_trial = np.ascontiguousarray(deconv[trial_slice, :].T, dtype=np.float16)
```

iii. The notes say the behavior arrays are already synchronized to imaging frames in the NWB release.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the framewise `lick` behavior stream.

ii. 
```python
lick = beh_root["lick"]["data"][()].astype(np.float32)
...
lick_bin = (lick[trial_slice] > 0).astype(np.int16)
```

iii. The notes say NWB `lick` is the synchronized cumulative-per-frame lick signal used for lick QC and binarization.

## 9-b. What processing is involved in computing `output` *Lick*?

i. First, some whole trials are removed by the lick-sensor QC rule. For retained trials, the code binarizes framewise lick values with the threshold `> 0`.

ii. 
```python
bad_lick_trials = detect_bad_lick_trials(lick, trial_pairs)
...
if bad_lick:
    continue
...
lick_bin = (lick[trial_slice] > 0).astype(np.int16)
```

iii. The notes say this mirrors the manuscript’s lick-fault handling and then converts the raw cumulative lick counts into the binary output requested by the decoder task.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is taken from `lick[trial_slice]`, so the lick value at each exported timepoint corresponds to the same frame as the neural activity.

ii. 
```python
trial_slice = slice(start, end)
...
lick_bin = (lick[trial_slice] > 0).astype(np.int16)
...
neural_trial = np.ascontiguousarray(deconv[trial_slice, :].T, dtype=np.float16)
```

iii. The notes justify this by saying all framewise behavior streams are already synchronized to the imaging frames.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from session metadata: subject ID and experiment day identify a `scene` entry in `sessions_dict.py`, and the scene name is converted into per-trial reward-zone labels A/B/C.

ii. 
```python
gcamp = subject_to_gcamp(subject_id)
session_meta = sessions_dict[(gcamp, exp_day)]
...
reward_labels_all = scene_reward_labels(session_meta.scene, len(trial_pairs))
```

iii. In Step 5 of the notes, the AI explicitly says reward-zone identity should come from the paper’s session metadata logic rather than from framewise `reward_zone` occupancy counts.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The code maps scene names like `Env1_LocationA` to fixed A/B/C labels, and maps switch scenes like `A_to_B` to pre-switch and post-switch labels with the switch fixed at trial 30. The resulting A/B/C code is then repeated across all time bins within each trial.

ii. 
```python
def scene_reward_labels(scene: str, n_trials: int, switch_trial: int = 30) -> list[str]:
    if scene in {"Env1_LocationA", "Env2_LocationA", "Env3_LocationA"}:
        return ["A"] * n_trials
    ...
    for src, dst in transitions:
        if f"{src}_to" in scene and scene.endswith(dst):
            return [src] * min(switch_trial, n_trials) + [dst] * max(0, n_trials - switch_trial)

...
repeated_row(RZ_CODE[reward_label], T, np.int16),
```

iii. The notes say this was chosen because the paper code defines reward zones from scene identity and a known switch rule, and because the user task asked for reward-zone location as a trial-level decoder output.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the sparse `Reward` event timestamps and the reconstructed trial windows.

ii. 
```python
reward_event_ts = beh_root["Reward"]["timestamps"][()].astype(np.float64)
...
reward_outcomes_all[i] = int(np.any((reward_event_ts >= start_ts) & (reward_event_ts < end_ts)))
```

iii. The notes say current-trial reward outcome should be based on actual reward events falling inside each trial.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is labeled rewarded if any reward event timestamp falls between its start time and teleport time. That binary value is then repeated across all time bins in the trial.

ii. 
```python
for i, (start, end) in enumerate(trial_pairs):
    start_ts = frame_timestamps[start]
    end_ts = frame_timestamps[end]
    reward_outcomes_all[i] = int(np.any((reward_event_ts >= start_ts) & (reward_event_ts < end_ts)))
...
repeated_row(reward_outcome, T, np.int16),
```

iii. The notes say this preserves the trial-level semantics of reward delivery while fitting the decoder’s uniform matrix format.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several anomalies defensively. It crops every behavior stream and each plane’s deconvolved activity to the minimum common frame count, excludes incomplete trailing trials implicitly by pairing starts to subsequent teleports, removes empty trials and lick-fault trials, errors on missing environment values or shape mismatches, and errors if NaNs remain after conversion.

ii. 
```python
min_frames = min([frame_timestamps.shape[0], *plane_frame_counts])
frame_timestamps = frame_timestamps[:min_frames]
position = position[:min_frames]
...
trial_pairs = pair_trial_events(trial_start, teleport, scanning)
...
if pos_trial.size == 0:
    continue
...
if env_trial_vals.size == 0:
    raise RuntimeError(f"{session_tag}: no valid environment values in trial {trial_idx}")
...
if neural_trial.shape[1] != input_trial.shape[1] or neural_trial.shape[1] != output_trial.shape[1]:
    raise RuntimeError(f"{session_tag}: time dimension mismatch in trial {trial_idx}")
if np.isnan(neural_trial).any() or np.isnan(input_trial).any() or np.isnan(output_trial).any():
    raise RuntimeError(f"{session_tag}: NaN values detected after conversion")
```

iii. The notes mention an NWB edge case with a dangling partial trial in `sub-m11_ses-03`, mixed frame-count anomalies across streams, and the need to keep the full export format-valid even if that means dropping malformed data rather than imputing it.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive parts are reading large NWB arrays, assembling the pooled deconvolved matrix across planes and ROIs, iterating over every trial to build per-trial arrays, optional plotting, and serializing the multi-gigabyte pickle.

ii. 
```python
with h5py.File(nwb_path, "r") as h5:
    ...
    plane_data = deconv_root[f"plane{int(plane)}"]["data"][:min_frames, local_cols]
    deconv[:, kept_positions] = plane_data.astype(np.float16)
...
for trial_idx, ((start, end), bad_lick, reward_label) in enumerate(
    zip(trial_pairs, bad_lick_trials, reward_labels_all, strict=True)
):
    ...
with args.outpicklefile.open("wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. In Steps 6, 7, and 9, the AI calls out large-file I/O and full-dataset pickle writing as the main runtime and memory bottlenecks.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious vectorization opportunities are the per-trial bad-lick loop, the per-trial reward-outcome loop, and the main per-trial conversion loop that repeatedly slices session arrays and constructs repeated-row inputs and outputs. Some of these remain loop-based because trial lengths vary.

ii. 
```python
for i, (start, end) in enumerate(trial_pairs):
    trial_lick = lick[start:end]
    ...

for i, (start, end) in enumerate(trial_pairs):
    start_ts = frame_timestamps[start]
    end_ts = frame_timestamps[end]
    reward_outcomes_all[i] = int(np.any((reward_event_ts >= start_ts) & (reward_event_ts < end_ts)))

for trial_idx, ((start, end), bad_lick, reward_label) in enumerate(
    zip(trial_pairs, bad_lick_trials, reward_labels_all, strict=True)
):
    ...
```

iii. The notes emphasize bounded memory and simple session-by-session processing over aggressive vectorization, especially because trials are variable length.

## 13-c. What processing does the code repeat multiple times?

i. Within one session, the code walks over the same `trial_pairs` three separate times: once for lick QC, once for reward-outcome assignment, and once for actual trial extraction and feature construction. It also computes repeated-row versions of several trial-level variables separately for each trial.

ii. 
```python
bad_lick_trials = detect_bad_lick_trials(lick, trial_pairs)
...
for i, (start, end) in enumerate(trial_pairs):
    ...
    reward_outcomes_all[i] = ...
...
for trial_idx, ((start, end), bad_lick, reward_label) in enumerate(
    zip(trial_pairs, bad_lick_trials, reward_labels_all, strict=True)
):
    ...
    repeated_row(env_trial, T, np.float32),
    repeated_row(trial_num, T, np.float32),
    repeated_row(prev_reward, T, np.float32),
```

iii. The notes frame this as a tradeoff for clarity and low peak memory use rather than a fully fused session-processing pipeline.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script performs some work that is useful for QA or documentation but not for decoder training itself: optional plotting, detailed per-session summaries stored in metadata, and bookkeeping lists like `kept_reward_labels`, `kept_reward_outcomes`, and `kept_trial_numbers` that are only used for plots or summaries, not for the exported decoder tensors.

ii. 
```python
kept_reward_labels: list[str] = []
kept_reward_outcomes: list[int] = []
kept_trial_numbers: list[int] = []
...
if make_plot:
    make_processing_plot(...)
...
summary = {
    "session_tag": session_tag,
    ...
    "elapsed_s": time.perf_counter() - start_time,
}
```

iii. The notes say these extras were kept to support sanity checks and human review of the conversion, even though the decoder only needs the final `neural`, `input`, and `output` arrays.
