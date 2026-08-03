# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files stored under `data/sub-*/sub-*_behavior+ophys.nwb`. It discovers all session files using a glob pattern, then iterates over each file, opening it with `h5py` to extract behavioral time series from `processing/behavior/BehavioralTimeSeries` and neural data from `processing/ophys/Deconvolved/plane0`. Each NWB file represents one session for one subject.

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
        subject = decode_if_bytes(f["general/subject/subject_id"][()])
        session_id = decode_if_bytes(f["general/session_id"][()])
        beh = f["processing/behavior/BehavioralTimeSeries"]
        neural_group = f["processing/ophys/Deconvolved/plane0"]
        ...
```

iii. The AI justified using direct `h5py` reads rather than the full NWB API for performance: "Used direct h5py reads instead of heavier NWB object materialization." This approach loads all 152 NWB files from the 11-mouse switch cohort.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `subject_id` field in each NWB file's `general/subject` group. A mapping from subject name to index is built incrementally as sessions are processed. All 11 mice (m3, m4, m7, m11-m15, m17-m19) are included.

ii.
```python
subject = decode_if_bytes(f["general/subject/subject_id"][()])
...
if arrays.subject not in subject_to_idx:
    subject_to_idx[arrays.subject] = len(all_subjects)
    all_subjects.append(arrays.subject)
data["subject_idx"].append(subject_to_idx[arrays.subject])
```

iii. The AI documented: "Treat the shared NWB archive as the 11-mouse switch cohort only. Fixed-condition mice are not present in data/."

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Files are sorted alphabetically by path, which groups sessions by subject and orders them by session number. Each session is processed independently and appended to the output lists.

ii.
```python
for session_idx, path in enumerate(files):
    arrays = load_session_arrays(path)
    neural_trials, input_trials, output_trials, brain_region_idx, session_info = convert_session(
        arrays=arrays,
        show_processing=args.show_processing and session_idx < 2,
    )
    data["neural"].append(neural_trials)
    data["input"].append(input_trials)
    data["output"].append(output_trials)
```

iii. The AI noted there are 152 sessions total (11 mice x 14 days minus 2 for m11 which starts on day 3), consistent with the paper.

## 1-d. How are the data split into trials?

i. Trial boundaries are reconstructed from the binary `trial_start` and `teleport` signals in the NWB behavioral time series. Trial start frames are where `trial_start > 0.5`, and trial end frames are where `teleport > 0.5`. Each trial spans from its start frame to its teleport frame (inclusive, i.e., `stop + 1` for Python slicing).

ii.
```python
def build_trial_slices(trial_start_signal: np.ndarray, teleport_signal: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(trial_start_signal > 0.5)
    teleports = np.flatnonzero(teleport_signal > 0.5)
    ...
    n = min(starts.size, teleports.size)
    slices: list[tuple[int, int]] = []
    for start, stop in zip(starts[:n], teleports[:n]):
        if stop < start:
            continue
        slices.append((int(start), int(stop) + 1))
```

iii. The AI documented: "Reconstruct trial start/end indices from trial_start and teleport signals in NWB. This matches the reference trial definition more closely than relying only on trial-number changes."

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on lick artifact detection only. A trial is excluded if more than 35% of its imaging frames have a cumulative lick count > 2. This threshold (0.35) matches the reference code's `glmUtils.get_timeseries_data`. The AI does NOT apply speed filtering (excluding timepoints with speed < 2 cm/s) which the reference code does.

ii.
```python
LICK_ARTIFACT_FRACTION = 0.35

lick_trial = arrays.lick[start:stop_exclusive]
if lick_trial.size and np.mean(lick_trial > 2) > LICK_ARTIFACT_FRACTION:
    lick_artifact[trial_id] = True
...
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    if lick_artifact[trial_id]:
        continue
```

iii. The AI noted: "Matches glmUtils.get_timeseries_data in the reference code" and reported 69/12,216 trials removed (0.56%), close to the paper's 81/12,376 (0.65%). The paper text mentions ">30%" threshold but the reference code uses 0.35; the AI followed the code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `processing/ophys/Deconvolved/plane0/data` in the NWB files, which contains pre-computed deconvolved calcium activity (events). This is filtered by `iscell` from `processing/ophys/ImageSegmentation/PlaneSegmentation`.

ii.
```python
neural_group = f["processing/ophys/Deconvolved/plane0"]
neural = neural_group["data"][()].astype(np.float32, copy=False)
roi_ids = neural_group["rois"][()].astype(np.int64, copy=False)
iscell = seg["iscell"][()][roi_ids, 0] > 0.5
neural = neural[:, iscell]
```

iii. The AI justified: "Use NWB Deconvolved as the neural signal: The paper's decoder and SI analyses operate on deconvolved activity after dF/F, and this stream is already provided in the archive."

## 2-b. How is the `neural` data processed?

i. The neural data undergoes the following processing: (1) Filter to curated cells using `iscell` on ROI indices referenced by the response matrix. (2) Crop to the common minimum length of neural and behavioral frames. (3) For 31 Hz sessions (m17, m18), rebin by factor 2 (summing pairs of frames) to achieve the common 15.5078125 Hz rate. (4) Extract per-trial slices and transpose to (neurons, time).

ii.
```python
# Cell filtering
iscell = seg["iscell"][()][roi_ids, 0] > 0.5
neural = neural[:, iscell]

# Common length
t_common = min(t_neural, t_behavior)

# Per-trial extraction and rebinning
neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)
if factor == 2:
    neural_trial = rebin_2d_sum_time_last(neural_trial, factor)
```

iii. The AI noted: "Restricted neural loading to ROI indices actually referenced by the response matrix and then to curated iscell ROIs." The rebinning uses summation for neural data (consistent with deconvolved event counts).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data is filtered at the cell level by the suite2p `iscell` flag (manual curation). Only ROIs referenced by the `Deconvolved/plane0/rois` index are considered, and among those, only cells with `iscell[:, 0] > 0.5` are kept. The AI did NOT implement the additional putative interneuron exclusion based on dF/F-speed correlation > 0.5 described in the paper.

ii.
```python
roi_ids = neural_group["rois"][()].astype(np.int64, copy=False)
iscell = seg["iscell"][()][roi_ids, 0] > 0.5
neural = neural[:, iscell]
roi_ids = roi_ids[iscell]
```

iii. The AI acknowledged the missing interneuron filter: "exact dF/F-speed-correlation interneuron exclusion was not re-run because the archive exposes deconvolved responses but not the exact reference dF/F stream." The AI noted this affected only 0.42 +/- 0.85% of cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start. Each trial's neural data begins at the `trial_start` frame and ends at the `teleport` frame (inclusive). The first time bin of each trial corresponds to the trial start frame, with `off_start = 0.0`.

ii.
```python
neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)
```
Where `start` comes from `np.flatnonzero(trial_start_signal > 0.5)`.

iii. Metadata records `temporal_alignment_event: "trial_start"` and `off_start: 0.0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The target temporal resolution is 1/15.5078125 Hz = ~64.5 ms. Sessions recorded at 15.5078125 Hz are used at native resolution. Sessions recorded at 31.015625 Hz (m17, m18) are rebinned by a factor of 2 to achieve the common rate.

ii.
```python
TARGET_RATE_HZ = 15.5078125
TARGET_DT_S = 1.0 / TARGET_RATE_HZ

if math.isclose(arrays.rate_hz, TARGET_RATE_HZ):
    factor = 1
elif math.isclose(arrays.rate_hz, 2.0 * TARGET_RATE_HZ):
    factor = 2
```

iii. The AI noted: "Standardize all sessions to a common 15.5078125 Hz bin size: This is the dominant dataset rate and the paper's effective per-plane sampling rate."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Time from start of trial is computed synthetically from the bin index and the target sampling rate. It is not read directly from any raw data variable.

ii.
```python
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
```

iii. This creates a time vector starting at 0.0 with increments of `1/15.5078125` seconds per bin, representing elapsed time since trial start.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Simple arithmetic: bin index multiplied by the time bin duration (1/15.5078125 s). No additional processing.

ii.
```python
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
```

iii. Straightforward computation aligned with the trial-start alignment event.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It shares the same number of time bins (`t_bins = neural_trial.shape[1]`), so each time value corresponds to exactly one neural time bin. The first bin is time 0 (trial start).

ii.
```python
t_bins = neural_trial.shape[1]
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
```

iii. Alignment is guaranteed by construction since both share the same number of bins.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from `processing/behavior/BehavioralTimeSeries/environment/data` in the NWB file.

ii.
```python
environment=beh["environment/data"][()][:t_common].astype(np.float32, copy=False),
```

iii. The AI documented that environment values in the NWB are -1, 0, or 1, where -1 indicates invalid periods.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the modal valid environment value (excluding negative values) is computed and broadcast across all time bins. Environment 0 maps to ENV1, environment 1 maps to ENV2.

ii.
```python
env = arrays.environment[start:stop_exclusive]
env = env[env >= 0]
trial_env[trial_id] = mode_int(env, default=0)
...
env_series = np.full((t_bins,), float(trial_env[trial_id]), dtype=np.float32)
```

iii. The AI noted: "Per trial, take modal valid environment value (0 or 1) and repeat across bins."

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The trial number is the within-session trial index (0-indexed), derived from the trial's position in the reconstructed trial list, not from the NWB `trial number` field.

ii.
```python
trial_series = np.full((t_bins,), float(trial_id), dtype=np.float32)
```

iii. The AI noted: "Keep native 0-indexing to match code conventions." Note: `trial_id` is the index in the `trial_slices` list, not the NWB `trial number` value. However, after lick-artifact filtering, the trial_id still reflects the original position in the session, not the filtered index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial index from the `trial_slices` enumeration is broadcast as a constant value across all time bins of that trial. This gives 0-indexed sequential trial numbers within each session. Note: lick-artifact-filtered trials retain their original index (e.g., if trial 5 is removed, the sequence has a gap).

ii.
```python
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    if lick_artifact[trial_id]:
        continue
    ...
    trial_series = np.full((t_bins,), float(trial_id), dtype=np.float32)
```

iii. The AI stated: "Per trial, use within-session trial index and repeat across bins."

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from `processing/behavior/BehavioralTimeSeries/Reward/timestamps` (reward event times) and `trial number` (to assign rewards to trials).

ii.
```python
reward_frame_idx = reward_frames_for_session(arrays)
rewarded_trial_ids = set(int(t) for t in arrays.trial_number[reward_frame_idx] if t >= 0)

trial_rewarded = np.array(
    [1 if trial_id in rewarded_trial_ids else 0 for trial_id in range(len(trial_slices))],
    dtype=np.int64,
)
trial_prev_rewarded = np.concatenate([[0], trial_rewarded[:-1]]).astype(np.float32, copy=False)
```

iii. The AI noted: "Previous trial rewarded (1) vs omitted (0), repeated across current-trial bins; first trial gets 0."

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Reward event timestamps are mapped to frame indices via `np.searchsorted`. Each frame's trial number determines which trial received the reward. The reward outcome array is shifted by one position: `trial_prev_rewarded = [0] + trial_rewarded[:-1]`. The first trial of each session defaults to 0 (no previous trial). This is broadcast across all time bins.

ii.
```python
def reward_frames_for_session(arrays: SessionArrays) -> np.ndarray:
    idx = np.searchsorted(arrays.timestamps, arrays.reward_times, side="left")
    return np.clip(idx, 0, arrays.timestamps.shape[0] - 1).astype(np.int64, copy=False)

trial_prev_rewarded = np.concatenate([[0], trial_rewarded[:-1]]).astype(np.float32, copy=False)
prev_reward_series = np.full((t_bins,), float(trial_prev_rewarded[trial_id]), dtype=np.float32)
```

iii. The AI documented this as "User-requested decoder input, not a paper decoder variable."

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position/data` (animal position on track) and the inferred reward zone location for each trial.

ii.
```python
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
zone_idx = int(trial_zone[trial_id])
dist = compute_distance_to_zone(position, zone_idx)
```

iii. Reward zone locations (A, B, C) have fixed bounds defined as constants matching the paper.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from the animal's position to the nearest edge of the active reward zone: negative if before the zone, 0 inside the zone, positive if past the zone. Position is clipped to [0, 450] cm.

ii.
```python
ZONE_BOUNDS = {
    0: (80.0, 130.0),   # A
    1: (200.0, 250.0),  # B
    2: (320.0, 370.0),  # C
}

def compute_distance_to_zone(position_cm: np.ndarray, zone_idx: int) -> np.ndarray:
    zone_start, zone_end = ZONE_BOUNDS[zone_idx]
    distance = np.zeros(position_cm.shape, dtype=np.float32)
    before = position_cm < zone_start
    after = position_cm > zone_end
    distance[before] = position_cm[before] - zone_start
    distance[after] = position_cm[after] - zone_end
    return distance
```

iii. The AI noted: "For zone [start, end]: pos < start -> pos-start; start <= pos <= end -> 0; pos > end -> pos-end."

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins matching the instruction specification exactly.

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

iii. Bins match the instruction: `< -50`, `-50 to -10`, `-10 to < 0`, `0`, `> 0 to +10`, `+10 to +50`, `> +50`.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position is extracted for the same trial slice (start:stop_exclusive) as neural data, so they share the same time bins. For 31 Hz sessions, position is rebinned by averaging pairs of frames.

ii.
```python
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
if factor == 2:
    position = rebin_1d_mean(position, factor)
```

iii. Alignment is guaranteed by using the same frame indices and rebinning factor.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from `position/data` in the NWB behavioral time series.

ii.
```python
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
pos_bin = discretize_position(position)
```

iii. Position is the animal's location on the 450 cm virtual track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450) cm and then discretized into 5 equal-sized bins of 90 cm each.

ii.
```python
def discretize_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, 0.0, TRACK_LENGTH_CM - 1e-6)
    return np.minimum((clipped / (TRACK_LENGTH_CM / 5.0)).astype(np.int64), 4)
```

iii. The AI noted: "Decoder spec requires 5 bins, not the paper's 45 bins." Each bin spans 90 cm (0-90, 90-180, 180-270, 270-360, 360-450).

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is divided by 90 cm (450/5) and floored to get bin index 0-4.

ii.
```python
return np.minimum((clipped / (TRACK_LENGTH_CM / 5.0)).astype(np.int64), 4)
```

iii. 5 equal bins as specified in the instructions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same as distance to reward zone: position is extracted from the same trial slice and rebinned identically.

ii.
```python
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
if factor == 2:
    position = rebin_1d_mean(position, factor)
```

iii. Alignment guaranteed by shared frame indices.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from `lick/data` in the NWB behavioral time series.

ii.
```python
lick=beh["lick/data"][()][:t_common].astype(np.float32, copy=False),
```

iii. The lick signal represents cumulative lick counts per frame from the NWB.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick is binarized: any value > 0 becomes 1 (lick present), otherwise 0. For 31 Hz sessions, lick is rebinned using logical OR (any lick in the 2-frame window counts as a lick).

ii.
```python
lick_binary = (arrays.lick[start:stop_exclusive] > 0).astype(np.int64, copy=False)
if factor == 2:
    lick_binary = rebin_1d_any(lick_binary, factor)
```

iii. The AI noted: "Convert to binary per bin (lick > 0) after lick-artifact handling. For rebinned 31 Hz sessions, use logical OR within each 15.5 Hz bin."

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices as neural data within each trial, with identical rebinning for 31 Hz sessions.

ii. Same trial slice extraction as neural and position data.

iii. Alignment guaranteed by construction.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from `reward_zone/data` (binary signal indicating when the animal is in the reward zone), `position/data`, and `Reward/timestamps`. These are combined to infer which of the three zones (A, B, C) is active on each trial.

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

iii. The AI used the `reward_zone` signal and reward event positions to identify which of the three fixed zones (A=80-130cm, B=200-250cm, C=320-370cm) is active, then filled gaps using block structure and the switch trial index (trial 30).

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For each trial: (1) Check if the `reward_zone` signal is active; if so, compute the median position during activation and map to the nearest zone center. (2) If no reward_zone signal, use the reward event position. (3) For trials with no zone information, fill using block structure: pre-switch majority zone for trials 0-29, post-switch majority for trials 30+. The zone index (0=A, 1=B, 2=C) is broadcast across all time bins.

ii.
```python
def infer_trial_zones(...):
    for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
        mask = reward_zone_signal[start:stop_exclusive] > 0
        if np.any(mask):
            zone_pos = float(np.nanmedian(positions[start:stop_exclusive][mask]))
        elif trial_id in reward_pos_by_trial:
            zone_pos = reward_pos_by_trial[trial_id]
        ...
        observed[trial_id] = int(np.argmin(np.abs(ZONE_CENTERS - zone_pos)))
    ...
    split = min(SWITCH_TRIAL_INDEX, n_trials)
    pre_majority = trial_majority_zone(observed, 0, split, ...)
    post_majority = trial_majority_zone(observed, split, n_trials, ...)
```

iii. The AI documented: "Infer reward-zone location from rewarded trials and fill omission trials within stable blocks."

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from `Reward/timestamps` in the NWB file. Reward events are mapped to trial numbers to determine which trials were rewarded.

ii.
```python
reward_frame_idx = reward_frames_for_session(arrays)
rewarded_trial_ids = set(int(t) for t in arrays.trial_number[reward_frame_idx] if t >= 0)
trial_rewarded = np.array(
    [1 if trial_id in rewarded_trial_ids else 0 for trial_id in range(len(trial_slices))],
    dtype=np.int64,
)
```

iii. Rewarded if any reward event timestamp maps to a frame within that trial's trial number.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are converted to frame indices using `np.searchsorted`. The trial number at each reward frame determines which trial received the reward. Binary outcome (0=omitted, 1=rewarded) is broadcast across all time bins of the trial.

ii.
```python
def reward_frames_for_session(arrays: SessionArrays) -> np.ndarray:
    idx = np.searchsorted(arrays.timestamps, arrays.reward_times, side="left")
    return np.clip(idx, 0, arrays.timestamps.shape[0] - 1).astype(np.int64, copy=False)

reward_bin = np.full((t_bins,), int(trial_rewarded[trial_id]), dtype=np.int64)
```

iii. The AI verified: "raw rewarded/omission = [0.153, 0.847]" matching the paper's ~15% omission rate.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled: (1) Neural/behavior frame count mismatches (up to 1 frame) are resolved by cropping to the common minimum. (2) Invalid environment values (-1) are excluded when computing the per-trial mode. (3) Trials where teleport < start are skipped. (4) Missing reward zone information on omission trials is filled using block structure. (5) Sessions with fewer than 2 valid trials after filtering raise an error.

ii.
```python
t_common = min(t_neural, t_behavior)  # Handle 1-frame mismatches

env = env[env >= 0]  # Exclude invalid environment values

if stop < start:  # Skip invalid trial boundaries
    continue

if len(neural_trials) < 2:  # Require minimum trials
    raise RuntimeError(...)
```

iii. The AI documented: "handled 10 sessions with a 1-sample neural/behavior length mismatch by cropping to the common minimum length before trial parsing."

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading neural data from NWB files (materializing the full deconvolved response matrix). Full conversion of 152 sessions completed in ~95 seconds (~0.63s/session average). Sessions with more neurons (e.g., m12 with ~1780 neurons) take longer (~0.7s).

ii. From output: `Processed 152 sessions in 95.11s`

iii. The AI estimated ~1.2-1.35s per session for sample runs and ~3.5 min total, but achieved faster times in practice.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates over all trials sequentially to construct neural, input, and output arrays. The lick artifact detection loop and reward zone inference loop also iterate per-trial. These could potentially be vectorized using numpy operations on the full session arrays with trial boundary indexing.

ii.
```python
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    if lick_artifact[trial_id]:
        continue
    neural_trial = arrays.neural[start:stop_exclusive].T...
    position = np.clip(arrays.position[start:stop_exclusive], ...)
    ...
```

iii. The AI noted these as acceptable given the fast overall runtime.

## 13-c. What processing does the code repeat multiple times?

i. Position clipping (`np.clip(position, 0.0, TRACK_LENGTH_CM)`) is done twice: once in `infer_trial_zones` (on the full session) and again per trial in the main loop. The reward frame computation is done once and reused, which is efficient.

ii.
```python
# In infer_trial_zones call:
positions=np.clip(arrays.position, 0.0, TRACK_LENGTH_CM),
# In trial loop:
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
```

iii. Minor duplication; the overhead is negligible.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes speed discretization as an output variable, but the reference paper excludes timepoints with speed < 2 cm/s from spatial analyses. Including all speed bins (including < 2 cm/s) means the code keeps timepoints that the reference analysis would discard. Additionally, the `observed_zone_position` array from `infer_trial_zones` is stored in session metadata but not used in the output data directly.

ii.
```python
# Speed is discretized but not filtered:
speed_bin = discretize_speed(speed)
# Observed zone positions stored in metadata:
"observed_zone_positions_cm": observed_zone_position.tolist(),
```

iii. Speed processing is required by the decoder task specification (which requests speed as an output), even though the reference code filters out low-speed timepoints.
