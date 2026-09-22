# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files from the `/app/data` directory by globbing for `sub-*/sub-*_behavior+ophys.nwb`. Files are read using `h5py` directly (not `pynwb`). Each NWB file corresponds to one session. Additionally, the AI loads session metadata from the reference code's `sessions_dict.py` to recover experiment-day scene information for reward zone assignment.

ii.
```python
all_files = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))
# ...
with h5py.File(nwb_path, "r") as h5:
    subject_id = h5["general/subject/subject_id"][()].decode()
    exp_day = int(h5["general/session_id"][()].decode())
    # ...
    beh_root = h5["processing/behavior/BehavioralTimeSeries"]
    frame_timestamps = beh_root["position"]["timestamps"][()].astype(np.float64)
    position = beh_root["position"]["data"][()].astype(np.float32)
    # ... loads all behavior and neural streams
```

iii. The AI documented in CONVERSION_NOTES.md that there are 152 NWB files across 11 subjects, matching the paper's description. Using `h5py` instead of `pynwb` was chosen for performance and direct access to HDF5 structure.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the NWB file's `general/subject/subject_id` field, which gives IDs like `m3`, `m4`, etc. The unique sorted set of subject IDs forms the `subjects` list.

ii.
```python
subjects = sorted({sess["subject_id"] for sess in converted_sessions})
subject_lookup = {subject: i for i, subject in enumerate(subjects)}
# ...
"subject_idx": np.asarray([subject_lookup[sess["subject_id"]] for sess in converted_sessions], dtype=np.int16),
```

iii. Subject IDs are taken from each NWB file's metadata, ensuring they match the canonical identifiers in the dataset.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. All 152 NWB files (10 mice x 14 sessions + m11 x 12 sessions) are processed as individual sessions.

ii.
```python
all_files = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))
# ...
for file_idx, nwb_path in enumerate(selected_files, start=1):
    converted, summary = convert_session(nwb_path, sessions_dict, do_plot)
```

iii. The one-file-per-session structure matches the paper's description of one imaging session per day.

## 1-d. How are the data split into trials?

i. Trials are defined by pairing `trial_start` and `teleport` pulse events. The AI also requires `scanning > 0` for both trial start and teleport events. A pointer-based pairing algorithm matches each start with its next valid end.

ii.
```python
def pair_trial_events(trial_start, teleport, scanning):
    start_idx = np.where((trial_start > 0) & (scanning > 0))[0]
    end_idx = np.where((teleport > 0) & (scanning > 0))[0]
    pairs = []
    end_ptr = 0
    for start in start_idx:
        while end_ptr < len(end_idx) and end_idx[end_ptr] <= start:
            end_ptr += 1
        if end_ptr >= len(end_idx):
            break
        end = end_idx[end_ptr]
        if end > start:
            pairs.append((int(start), int(end)))
        end_ptr += 1
    return pairs
```

iii. The AI explained this matches the reference code's use of explicit start/end indices, and the `scanning > 0` filter ensures only valid imaging periods are included.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on lick-sensor fault detection. A trial is removed if >30% of its imaging frames have a cumulative lick count > 2. This matches the paper's described lick QC procedure. No minimum-timepoint filter is applied.

ii.
```python
def detect_bad_lick_trials(lick, trial_pairs):
    bad = np.zeros(len(trial_pairs), dtype=bool)
    for i, (start, end) in enumerate(trial_pairs):
        trial_lick = lick[start:end]
        if trial_lick.size == 0:
            bad[i] = True
            continue
        bad[i] = np.mean(trial_lick > 2) > LICK_ERROR_THRESHOLD  # LICK_ERROR_THRESHOLD = 0.30
    return bad
```

iii. The AI documented that this recovered exactly 81 removed trials matching the paper's reported "81 out of 12,376 trials removed across 11 switch mice", confirming consistency with the paper's QC logic.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the NWB's `processing/ophys/Deconvolved/plane*/data` arrays -- the pre-stored deconvolved activity computed by suite2p. It is NOT derived from the raw Fluorescence and Neuropil traces.

ii.
```python
deconv_root = h5["processing/ophys/Deconvolved"]
# ...
deconv = np.empty((min_frames, keep_indices.size), dtype=np.float16)
for plane in np.unique(kept_plane_idx):
    kept_positions = np.where(kept_plane_idx == plane)[0]
    local_cols = local_index[keep_indices[kept_positions]]
    plane_data = deconv_root[f"plane{int(plane)}"]["data"][:min_frames, local_cols]
    deconv[:, kept_positions] = plane_data.astype(np.float16)
```

iii. The AI justified this in CONVERSION_NOTES.md Step 4: "Prefer NWB Deconvolved as the neural signal for export, because it matches the paper's event-like activity and avoids reimplementing dF/F from scratch." The AI believed the NWB Deconvolved was the same signal the paper analyzes.

## 2-b. How is the `neural` data processed?

i. No additional processing is applied to the neural data beyond reading the pre-stored deconvolved values, filtering to `iscell` ROIs, pooling across planes by `planeIdx`, and casting to `float16`. The AI does NOT recompute dF/F, does NOT apply neuropil correction, and does NOT perform OASIS deconvolution.

ii.
```python
deconv = np.empty((min_frames, keep_indices.size), dtype=np.float16)
# ... reads and pools planes ...
neural_trial = np.ascontiguousarray(deconv[trial_slice, :].T, dtype=np.float16)
```

iii. The AI's rationale was that the NWB Deconvolved should already be the paper's processed signal. However, CONVERSION_NOTES.md Step 1 noted that "Code computes dff from fluorescence, then deconvolves to events" and the NWB stores separate Fluorescence, Neuropil, and Deconvolved streams.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only the `iscell` ROI curation filter is applied (`iscell[:,0] > 0.5`). No putative interneuron exclusion based on speed correlation is performed.

ii.
```python
seg = h5["processing/ophys/ImageSegmentation/PlaneSegmentation"]
iscell = seg["iscell"][()]
keep_mask = iscell[:, 0] > 0.5
keep_indices = np.flatnonzero(keep_mask)
```

iii. The AI noted in CONVERSION_NOTES.md that the paper describes an additional interneuron exclusion step (Pearson correlation of dF/F with speed > 0.5 removes ~0.42% of cells). The AI checked one session and found 0 cells exceeded the threshold on the NWB Deconvolved signal, so it retained all `iscell` ROIs. However, this check was done on the wrong signal -- the paper's interneuron test operates on dF/F, not on deconvolved activity.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start by slicing frames from the `trial_start` index to the frame before `teleport`. No additional temporal shifting is needed since the trial start IS the alignment event.

ii.
```python
trial_slice = slice(start, end)  # exclude teleport frame itself
# ...
neural_trial = np.ascontiguousarray(deconv[trial_slice, :].T, dtype=np.float16)
```

iii. The instructions specify "Temporally align based on start of the trial," which requires no additional alignment beyond splitting data at trial boundaries.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native imaging frame rate: ~64.48 ms (1/15.5078125 Hz). No rebinning is applied. The AI hardcodes `COMMON_FRAME_RATE_HZ = 15.5078125` and validates that each session's median timestamp step matches this value.

ii.
```python
COMMON_FRAME_RATE_HZ = 15.5078125
COMMON_FRAME_DT_S = 1.0 / COMMON_FRAME_RATE_HZ
TIME_BIN_SIZE_MS = COMMON_FRAME_DT_S * 1000.0
# ...
frame_dt = median_step_seconds(frame_timestamps)
if not np.isclose(frame_dt, COMMON_FRAME_DT_S, atol=1e-6):
    raise RuntimeError(f"{session_tag}: unexpected frame dt {frame_dt}")
```

iii. The paper states "0.0645 s imaging frame samples" and "~15.5 Hz," consistent with this rate. No rebinning is needed since behavioral data is already synchronized to the imaging frame rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the behavior timestamps array (`processing/behavior/BehavioralTimeSeries/position/timestamps`).

ii.
```python
frame_timestamps = beh_root["position"]["timestamps"][()].astype(np.float64)
# ...
t_trial = frame_timestamps[trial_slice] - frame_timestamps[start]
```

iii. The behavior timestamps are the same for all behavior time series (verified to be at the imaging frame rate).

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The trial-start timestamp is subtracted from each frame's timestamp within the trial.

ii.
```python
t_trial = frame_timestamps[trial_slice] - frame_timestamps[start]
```

iii. Simple subtraction to get time relative to trial onset.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same frame indices (both are sampled at imaging frame rate), so they are inherently aligned by using the same `trial_slice`.

ii.
```python
trial_slice = slice(start, end)
# ...
t_trial = frame_timestamps[trial_slice] - frame_timestamps[start]
neural_trial = np.ascontiguousarray(deconv[trial_slice, :].T, dtype=np.float16)
```

iii. Both use the same slice indices, ensuring temporal alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from `processing/behavior/BehavioralTimeSeries/environment/data`.

ii.
```python
environment = beh_root["environment"]["data"][()].astype(np.float32)
# ...
env_trial_vals = environment[trial_slice]
env_trial_vals = env_trial_vals[env_trial_vals >= 0]
env_trial = int(np.rint(np.median(env_trial_vals)))
```

iii. The environment variable encodes ENV1=0 and ENV2=1.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI filters out negative values (pre-synchronization markers), then takes the rounded median of the remaining values within the trial as a single per-trial value, which is repeated across all timepoints.

ii.
```python
env_trial_vals = environment[trial_slice]
env_trial_vals = env_trial_vals[env_trial_vals >= 0]
env_trial = int(np.rint(np.median(env_trial_vals)))
# ...
repeated_row(env_trial, T, np.float32),
```

iii. Since environment is constant within a trial, taking the median is a robust way to extract the value even if there are edge-case frames.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `processing/behavior/BehavioralTimeSeries/trial number/data` field in the NWB file, evaluated at the trial start frame.

ii.
```python
trial_number = beh_root["trial number"]["data"][()].astype(np.int32)
# ...
trial_num = int(trial_number[start])
```

iii. The AI uses the stored trial number from the NWB data rather than computing a sequential index from the loop.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial number value at the trial start frame is extracted and used as a constant value repeated across all timepoints in the trial.

ii.
```python
trial_num = int(trial_number[start])
# ...
repeated_row(trial_num, T, np.float32),
```

iii. No additional processing beyond reading the stored value.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps and per-trial reward outcome computation. Reward events are detected by checking if any reward timestamp falls within a trial's time window.

ii.
```python
reward_event_ts = beh_root["Reward"]["timestamps"][()].astype(np.float64)
# ...
reward_outcomes_all = np.zeros(len(trial_pairs), dtype=np.int16)
for i, (start, end) in enumerate(trial_pairs):
    start_ts = frame_timestamps[start]
    end_ts = frame_timestamps[end]
    reward_outcomes_all[i] = int(np.any((reward_event_ts >= start_ts) & (reward_event_ts < end_ts)))
```

iii. Reward events have separate timestamps from the behavior frame rate, so timestamp-range comparison is used.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the previous trial's reward outcome (from the `reward_outcomes_all` array computed over ALL trials including bad-lick ones) is used. For the first trial, the value is 0.

ii.
```python
prev_reward = int(reward_outcomes_all[trial_idx - 1]) if trial_idx > 0 else 0
```

iii. Using the previous trial index from ALL trials (not just kept trials) is correct because the animal experienced that previous trial regardless of whether it was later filtered out.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone coordinates. Reward zone identity per trial comes from the reference code's `sessions_dict.py` scene metadata and the `scene_reward_labels()` function, which maps scene names to A/B/C labels including switch-trial logic.

ii.
```python
def scene_reward_labels(scene, n_trials, switch_trial=30):
    if scene in {"Env1_LocationA", "Env2_LocationA", "Env3_LocationA"}:
        return ["A"] * n_trials
    # ... handles switch sessions ...
    for src, dst in transitions:
        if f"{src}_to" in scene and scene.endswith(dst):
            return [src] * min(switch_trial, n_trials) + [dst] * max(0, n_trials - switch_trial)
# ...
reward_labels_all = scene_reward_labels(session_meta.scene, len(trial_pairs))
```

iii. The AI uses the reference code's session metadata to determine reward zone labels, which is the same approach the paper's code uses internally via `behavior.get_reward_zones`.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, signed distance from position to the nearest edge of the assigned reward zone is computed. Distance is 0 when inside the zone, negative when before it, positive when past it.

ii.
```python
def reward_distance_to_zone(position_cm, zone_start, zone_end):
    before = position_cm < zone_start
    after = position_cm > zone_end
    dist = np.zeros(position_cm.shape, dtype=np.float32)
    dist[before] = position_cm[before] - zone_start
    dist[after] = position_cm[after] - zone_end
    return dist
```

iii. This matches the standard signed-distance-to-interval computation.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit conditional assignment rather than `np.digitize`.

ii.
```python
def discretize_reward_distance(distance_cm):
    out = np.full(distance_cm.shape, 6, dtype=np.int16)
    out[distance_cm < -50] = 0
    out[(distance_cm >= -50) & (distance_cm < -10)] = 1
    out[(distance_cm >= -10) & (distance_cm < 0)] = 2
    out[distance_cm == 0] = 3
    out[(distance_cm > 0) & (distance_cm <= 10)] = 4
    out[(distance_cm > 10) & (distance_cm <= 50)] = 5
    return out
```

iii. The bin boundaries match the instructions. Minor boundary handling differences from the reference at exactly 10, 50, etc. are inconsequential.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indices (`trial_slice`) are used for both position (to compute distance) and neural data, ensuring alignment.

ii.
```python
trial_slice = slice(start, end)
pos_trial = position[trial_slice]
# ...
neural_trial = np.ascontiguousarray(deconv[trial_slice, :].T, dtype=np.float16)
```

iii. Both are indexed by the same slice.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from `processing/behavior/BehavioralTimeSeries/position/data`.

ii.
```python
position = beh_root["position"]["data"][()].astype(np.float32)
# ...
pos_trial = position[trial_slice]
```

iii. The position variable records the animal's position along the VR corridor in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice and discretizing.

ii.
```python
pos_bin = discretize_absolute_position(pos_trial)
```

iii. Raw position values are used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 equal-sized bins (90 cm each) spanning the 450 cm track.

ii.
```python
def discretize_absolute_position(position_cm):
    out = np.zeros(position_cm.shape, dtype=np.int16)
    out[(position_cm >= 90) & (position_cm < 180)] = 1
    out[(position_cm >= 180) & (position_cm < 270)] = 2
    out[(position_cm >= 270) & (position_cm <= 360)] = 3
    out[position_cm > 360] = 4
    return out
```

iii. Five 90 cm bins match the instruction for "5 equal-sized bins spanning the 450 cm track."

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame-based slicing ensures alignment.

ii. `pos_trial = position[trial_slice]` uses the same slice as neural data.

iii. Verified by shared indexing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from `processing/behavior/BehavioralTimeSeries/lick/data`.

ii.
```python
lick = beh_root["lick"]["data"][()].astype(np.float32)
```

iii. The lick variable records per-frame lick counts.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
lick_bin = (lick[trial_slice] > 0).astype(np.int16)
```

iii. The instructions specify binary output (0 = no, 1 = yes). The raw lick values are cumulative per-frame counts, so thresholding at >0 converts to binary.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame-based slicing.

ii. `lick[trial_slice]` uses the same indices as neural data.

iii. Aligned by shared indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. NOT derived from the NWB's framewise `reward_zone` behavior variable. Instead derived from the reference code's `sessions_dict.py` metadata, which maps each (animal, experiment_day) pair to a scene name. The scene name encodes the reward zone location(s) and switch transitions.

ii.
```python
def load_sessions_dict():
    # imports sessions_dict.py from reference code
    for collection_name in ("single_plane", "multi_plane"):
        collection = getattr(module, collection_name)
        for animal, entries in collection.items():
            for entry in entries:
                meta = SessionMeta(...)
                mapping[(animal, meta.exp_day)] = meta
    return mapping
# ...
session_meta = sessions_dict[(gcamp, exp_day)]
reward_labels_all = scene_reward_labels(session_meta.scene, len(trial_pairs))
```

iii. The AI noted in CONVERSION_NOTES.md that "the paper/code define zone A/B/C from scene identity and switch timing; the framewise reward_zone signal indicates occupancy, not location label."

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene names are parsed to determine reward zone labels. For non-switch sessions, all trials get the same label (e.g., "A"). For switch sessions, the first 30 trials get the pre-switch label and remaining trials get the post-switch label. Labels are mapped to codes: A=0, B=1, C=2.

ii.
```python
def scene_reward_labels(scene, n_trials, switch_trial=30):
    if scene in {"Env1_LocationA", ...}: return ["A"] * n_trials
    # ...
    for src, dst in transitions:
        if f"{src}_to" in scene and scene.endswith(dst):
            return [src] * min(switch_trial, n_trials) + [dst] * max(0, n_trials - switch_trial)
# ...
output: repeated_row(RZ_CODE[reward_label], T, np.int16),
```

iii. This matches the reference code's `behavior.get_reward_zones` logic.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` timestamps in the NWB file.

ii.
```python
reward_event_ts = beh_root["Reward"]["timestamps"][()].astype(np.float64)
# ...
reward_outcomes_all[i] = int(np.any((reward_event_ts >= start_ts) & (reward_event_ts < end_ts)))
```

iii. Reward events have their own timestamps separate from the frame-rate behavior streams.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward event timestamp falls within the trial's time window (from trial_start to teleport). If yes, output is 1; otherwise 0. The value is constant across all timepoints in the trial.

ii.
```python
for i, (start, end) in enumerate(trial_pairs):
    start_ts = frame_timestamps[start]
    end_ts = frame_timestamps[end]
    reward_outcomes_all[i] = int(np.any((reward_event_ts >= start_ts) & (reward_event_ts < end_ts)))
# ...
repeated_row(reward_outcome, T, np.int16),
```

iii. The per-trial binary reward outcome matches the instruction's "0 = no, 1 = yes."

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Neural/behavior length mismatch**: Data is truncated to the minimum common frame count across all plane data and behavior timestamps.
- **Bad lick trials**: Trials with suspected lick-sensor faults are removed entirely.
- **Empty trials**: Trials with zero frames are skipped.
- **NaN checks**: An assertion verifies no NaN values exist in the final converted trial data.
- **Trailing partial trials**: The pointer-based pairing algorithm naturally excludes unpaired trial_start events.
- **Fewer than 2 valid trials**: Sessions with fewer than 2 valid trials raise an error.

ii.
```python
min_frames = min([frame_timestamps.shape[0], *plane_frame_counts])
frame_timestamps = frame_timestamps[:min_frames]
# ...
if bad_lick: continue
# ...
if pos_trial.size == 0: continue
# ...
if np.isnan(neural_trial).any() or ...: raise RuntimeError(...)
if len(neural_trials) < 2: raise RuntimeError(...)
```

iii. These are defensive checks found during data exploration.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files with h5py and reading large deconvolved neural arrays
2. Saving the large pickle file (~4.82 GB)
3. The full conversion completed in ~107 seconds for 152 sessions.

ii. N/A

iii. The NWB files contain full neural recordings, making I/O the bottleneck.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` (iterating over trial_pairs) could potentially be vectorized for some operations (e.g., reward outcome computation, discretization), but variable trial lengths make this awkward. The bad-lick detection loop iterates over all trials sequentially.

ii. N/A

iii. The per-trial loop is the natural structure given variable-length trials.

## 13-c. What processing does the code repeat multiple times?

i. The AI's code does NOT repeat processing -- it loads each NWB file once and processes it in a single pass. This is more efficient than the reference solution which has a separate survey step that loads all files first.

ii. N/A

iii. The AI optimized for single-pass processing.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `sessions_dict.py` from the reference code repository using `importlib`, which loads more metadata than strictly needed. Also, the code imports but doesn't use `speed` filtering (the 2 cm/s threshold) for place-cell analyses, since speed is a decoder output rather than a filter criterion.

ii. N/A

iii. These are minor inefficiencies that don't significantly impact performance.
