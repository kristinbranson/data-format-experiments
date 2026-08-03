# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all NWB files by globbing `data/sub-*/sub-*_behavior+ophys.nwb`. Each NWB file is opened with `h5py` (not `pynwb`) and behavioral + neural data are read from `processing/behavior/BehavioralTimeSeries` and `processing/ophys/Deconvolved/plane0`. Only a single plane (`plane0`) is read from the ophys data, regardless of whether the session has multiple imaging planes.

ii.
```python
def list_session_files() -> list[str]:
    files = sorted(glob.glob(os.path.join("data", "sub-*", "sub-*_behavior+ophys.nwb")))
    if not files:
        raise FileNotFoundError("No NWB files found under data/sub-*/")
    return files

def load_session_arrays(path: str) -> SessionArrays:
    with h5py.File(path, "r") as f:
        subject = decode_if_bytes(f["general/subject/subject_id"][()])
        ...
        beh = f["processing/behavior/BehavioralTimeSeries"]
        neural_group = f["processing/ophys/Deconvolved/plane0"]
        seg = f["processing/ophys/ImageSegmentation/PlaneSegmentation"]
        ...
```

iii. The AI chose `h5py` for direct, faster reads compared to `pynwb`. It discovered all NWB files by sorted glob matching. The AI justified reading only `plane0` by claiming "The archive exposes only the plane-0 response series for these files" (CONVERSION_NOTES Step 4).

## 1-b. How are the data split into subjects?

i. Subjects are identified from NWB metadata (`general/subject/subject_id`) as each file is loaded. A running `subject_to_idx` dictionary maps subject names to indices.

ii.
```python
if arrays.subject not in subject_to_idx:
    subject_to_idx[arrays.subject] = len(all_subjects)
    all_subjects.append(arrays.subject)
```

iii. Subject identity is extracted from NWB metadata rather than parsing directory names. The AI verified 11 subjects matching the paper's switch-task cohort.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are identified by sorted glob of all NWB files across all subject directories.

ii.
```python
files = sorted(glob.glob(os.path.join("data", "sub-*", "sub-*_behavior+ophys.nwb")))
```

iii. The NWB archive contains one file per session. The AI processes 152 sessions total (11 mice x 14 days - 2 for m11).

## 1-d. How are the data split into trials?

i. Trial boundaries are determined from the `trial_start` and `teleport` binary signals. Trial starts are frames where `trial_start > 0.5`. Trial ends are frames where `teleport > 0.5`. Starts and teleport frames are paired sequentially via `zip`, skipping pairs where `stop < start`. The trial slice includes the teleport frame (stop + 1 is the exclusive end).

ii.
```python
def build_trial_slices(trial_start_signal, teleport_signal):
    starts = np.flatnonzero(trial_start_signal > 0.5)
    teleports = np.flatnonzero(teleport_signal > 0.5)
    n = min(starts.size, teleports.size)
    slices = []
    for start, stop in zip(starts[:n], teleports[:n]):
        if stop < start:
            continue
        slices.append((int(start), int(stop) + 1))
    if len(slices) < 2:
        raise RuntimeError("Need at least two trials in a session.")
    return slices
```

iii. The AI used the `trial_start` and `teleport` signals to reconstruct trial boundaries, consistent with the reference code's use of `trial_start_inds` and `teleport_inds`. The AI notes that `trial number` and `trial_start` do not always agree, so `trial_start`/`teleport` are preferred.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on **lick artifacts**: if more than 35% of imaging frames in a trial have lick counts > 2, the trial is removed. There is no minimum-timepoint filter for short trials.

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

iii. The AI implemented the paper's lick-artifact QC rule: "trials with erroneous lick detection when >30% of imaging frames in the trial have cumulative lick count > 2" were removed. The AI used a 35% threshold (CONVERSION_NOTES reference `glmUtils.get_timeseries_data`). 69 out of 12,216 trials were removed (~0.56%), close to the paper's reported 81/12,376 (~0.65%).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `processing/ophys/Deconvolved/plane0/data` in the NWB files.

ii.
```python
neural_group = f["processing/ophys/Deconvolved/plane0"]
neural = neural_group["data"][()].astype(np.float32, copy=False)
```

iii. The paper's decoder uses deconvolved calcium activity (`events`), which corresponds to the `Deconvolved` processing module in the NWB files (CONVERSION_NOTES Steps 1, 4).

## 2-b. How is the `neural` data processed?

i. For single-plane sessions (~15.5 Hz), no processing beyond ROI filtering. For multi-plane sessions (~31 Hz), neural data is rebinned by a factor of 2 (pair-wise summation) to achieve the common ~15.5 Hz rate.

ii.
```python
if factor == 2:
    neural_trial = rebin_2d_sum_time_last(neural_trial, factor)

def rebin_2d_sum_time_last(x, factor):
    t = x.shape[1]
    n_full = t // factor
    parts = []
    if n_full:
        full = x[:, : n_full * factor].reshape(x.shape[0], n_full, factor).sum(axis=2)
        parts.append(full)
    if t % factor:
        parts.append(x[:, n_full * factor :].sum(axis=1, keepdims=True))
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)
```

iii. The AI standardized all sessions to a common 15.5 Hz rate so all sessions share the same decoder timebase, as required by the format specification. For 31 Hz sessions (m17, m18), pairwise summation across adjacent time bins was applied to the neural data.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by the `iscell` column from `ImageSegmentation/PlaneSegmentation`. Only ROI indices referenced by `Deconvolved/plane0/rois` are considered, and among those, only cells with `iscell[:,0] > 0.5` are kept.

ii.
```python
seg = f["processing/ophys/ImageSegmentation/PlaneSegmentation"]
roi_ids = neural_group["rois"][()].astype(np.int64, copy=False)
iscell = seg["iscell"][()][roi_ids, 0] > 0.5
neural = neural[:, iscell]
roi_ids = roi_ids[iscell]
```

iii. The AI noted that the ROI-region-aware filtering is critical for multi-plane sessions to avoid overcounting ROIs. The curated neuron counts (155-1780 per session, 118,493 total) fall within the paper's reported range (155-2172).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start by slicing the neural array at the trial boundaries (trial_start to teleport). No additional temporal shifting is needed since the neural and behavioral data share the same timebase.

ii.
```python
neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)
```

iii. The instructions specify alignment to "start of the trial", which corresponds to the `trial_start` signal. Since neural and behavioral data are already frame-aligned in the NWB, slicing at trial boundaries achieves the alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The target temporal resolution is ~15.5 Hz (64.45 ms per bin). Sessions recorded at ~31 Hz (m17, m18) are rebinned by a factor of 2 using pairwise summation for neural data and pairwise mean for behavioral variables. Single-plane sessions at ~15.5 Hz are kept at their native resolution.

ii.
```python
TARGET_RATE_HZ = 15.5078125
TARGET_DT_S = 1.0 / TARGET_RATE_HZ

if math.isclose(arrays.rate_hz, TARGET_RATE_HZ):
    factor = 1
elif math.isclose(arrays.rate_hz, 2.0 * TARGET_RATE_HZ):
    factor = 2
```

iii. The paper states "sampled at ~15.5 Hz" and multi-plane sessions are "~15.5 Hz per plane". The AI chose to standardize to the common rate to satisfy the format requirement that "time bins should be the same size for all trials and sessions."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. This input is constructed from the bin index and the target sampling rate, not from raw timestamps.

ii.
```python
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
```

iii. The AI chose to construct time from the known constant sampling rate rather than using raw timestamps. This ensures perfectly uniform time steps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, an array of evenly-spaced time values is created: `[0, dt, 2*dt, ...]` where `dt = 1/15.5078125 s`.

ii.
```python
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
```

iii. This gives time from trial start in seconds, starting at 0.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Time is constructed to have the same number of bins as the neural data for the trial (`t_bins = neural_trial.shape[1]`), so alignment is guaranteed by construction.

ii.
```python
t_bins = neural_trial.shape[1]
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
```

iii. The time array length matches neural data length by using the same trial bin count.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from `processing/behavior/BehavioralTimeSeries/environment/data`.

ii.
```python
environment=beh["environment/data"][()][:t_common].astype(np.float32, copy=False),
```

iii. The `environment` signal directly records the environment type (0 or 1).

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The modal (most frequent) non-negative environment value within each trial is computed and used as a constant value for the entire trial.

ii.
```python
def mode_int(values, default=0):
    values = values.astype(np.int64, copy=False)
    values = values[values >= 0]
    if values.size == 0:
        return default
    return int(np.bincount(values).argmax())

for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    env = arrays.environment[start:stop_exclusive]
    env = env[env >= 0]
    trial_env[trial_id] = mode_int(env, default=0)
...
env_series = np.full((t_bins,), float(trial_env[trial_id]), dtype=np.float32)
```

iii. The environment value of -1 appears outside valid trial/imaging periods, so filtering to non-negative values and taking the mode gives the correct per-trial environment.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The trial number is the within-session trial index (0-indexed), derived from the loop counter over trial slices.

ii.
```python
trial_series = np.full((t_bins,), float(trial_id), dtype=np.float32)
```

iii. The within-session trial index corresponds to the sequential ordering of trials.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond assigning the loop index. The value is constant across all timepoints within a trial. Note: the trial index counts ALL trials including those later removed by lick-artifact filtering, so the trial numbers may not be contiguous in the final dataset.

ii.
```python
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    ...
    trial_series = np.full((t_bins,), float(trial_id), dtype=np.float32)
```

iii. The trial number is simply the sequential index of the trial within the session.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` event timestamps and the `trial number` behavioral time series. Reward events are mapped to trial IDs using the `trial_number` signal at the reward frame indices.

ii.
```python
reward_times=beh["Reward/timestamps"][()].astype(np.float64, copy=False),
...
def reward_frames_for_session(arrays):
    idx = np.searchsorted(arrays.timestamps, arrays.reward_times, side="left")
    return np.clip(idx, 0, arrays.timestamps.shape[0] - 1).astype(np.int64, copy=False)

rewarded_trial_ids = set(int(t) for t in arrays.trial_number[reward_frame_idx] if t >= 0)
trial_rewarded = np.array(
    [1 if trial_id in rewarded_trial_ids else 0 for trial_id in range(len(trial_slices))],
    dtype=np.int64,
)
trial_prev_rewarded = np.concatenate([[0], trial_rewarded[:-1]]).astype(np.float32, copy=False)
```

iii. The AI determines which trials were rewarded by checking whether any reward event falls within a frame whose `trial_number` matches the trial index, then shifts by one to get the previous trial's outcome.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Reward event timestamps are mapped to frame indices via `searchsorted`. The `trial_number` signal at those frames identifies which trial received a reward. A boolean array of per-trial reward status is shifted by one position (first trial gets 0).

ii.
```python
trial_prev_rewarded = np.concatenate([[0], trial_rewarded[:-1]]).astype(np.float32, copy=False)
...
prev_reward_series = np.full((t_bins,), float(trial_prev_rewarded[trial_id]), dtype=np.float32)
```

iii. The first trial's previous outcome is set to 0 (no prior trial). Subsequent trials inherit the reward status of the preceding trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavioral time series and the inferred reward zone location for each trial. The reward zone is inferred from the `reward_zone` behavioral signal, `position`, and reward event locations using a heuristic combining median position during reward-zone-active periods and closest zone center.

ii.
```python
position=beh["position/data"][()][:t_common].astype(np.float32, copy=False),
reward_zone_signal=beh["reward_zone/data"][()][:t_common].astype(np.float32, copy=False),

def infer_trial_zones(positions, reward_zone_signal, trial_slices, ...):
    for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
        mask = reward_zone_signal[start:stop_exclusive] > 0
        if np.any(mask):
            zone_pos = float(np.nanmedian(positions[start:stop_exclusive][mask]))
        ...
        if np.isfinite(zone_pos):
            observed[trial_id] = int(np.argmin(np.abs(ZONE_CENTERS - zone_pos)))
    ...

ZONE_BOUNDS = {
    0: (80.0, 130.0),   # A
    1: (200.0, 250.0),  # B
    2: (320.0, 370.0),  # C
}
```

iii. The paper defines reward zones A, B, C at specific position ranges. The AI infers which zone is active per trial using median position in reward-zone-active periods, then assigns the closest of the three known zones.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the signed distance from the animal's position to the nearest edge of the reward zone is computed. Distance is 0 when inside the zone, negative when before the zone, positive when past it. Position is clipped to [0, 450] before computing distance.

ii.
```python
def compute_distance_to_zone(position_cm, zone_idx):
    zone_start, zone_end = ZONE_BOUNDS[zone_idx]
    distance = np.zeros(position_cm.shape, dtype=np.float32)
    before = position_cm < zone_start
    after = position_cm > zone_end
    distance[before] = position_cm[before] - zone_start
    distance[after] = position_cm[after] - zone_end
    return distance
```

iii. Matches the paper's concept of distance relative to the reward zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous distance is discretized into 7 bins using explicit conditional logic:
- 0: < -50 cm
- 1: >= -50 and < -10 cm
- 2: >= -10 and < 0 cm
- 3: == 0 cm
- 4: > 0 and <= 10 cm
- 5: > 10 and <= 50 cm
- 6: > 50 cm

ii.
```python
def discretize_distance(distance_cm):
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

iii. The bin edges follow the instruction specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position is sliced at the same trial boundaries as neural data, so alignment is inherent. For 31 Hz sessions, position is rebinned (pairwise mean) with the same factor as neural data.

ii.
```python
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
if factor == 2:
    position = rebin_1d_mean(position, factor)
dist = compute_distance_to_zone(position, zone_idx)
```

iii. Both neural and behavioral data use the same trial slicing and rebinning factor.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from `processing/behavior/BehavioralTimeSeries/position/data`.

ii.
```python
position=beh["position/data"][()][:t_common].astype(np.float32, copy=False),
```

iii. The `position` variable directly records the animal's position in the VR corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450] cm, then discretized into 5 equal bins of 90 cm each by dividing by 90 and flooring.

ii.
```python
def discretize_position(position_cm):
    clipped = np.clip(position_cm, 0.0, TRACK_LENGTH_CM - 1e-6)
    return np.minimum((clipped / (TRACK_LENGTH_CM / 5.0)).astype(np.int64), 4)
```

iii. Track length is 450 cm, so 450/5 = 90 cm per bin gives equal-sized bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. 5 bins of 90 cm: [0, 90), [90, 180), [180, 270), [270, 360), [360, 450].

ii.
```python
clipped = np.clip(position_cm, 0.0, TRACK_LENGTH_CM - 1e-6)
return np.minimum((clipped / (TRACK_LENGTH_CM / 5.0)).astype(np.int64), 4)
```

iii. The AI interprets "5 equal-sized bins" as 450/5 = 90 cm each, covering [0, 450].

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same trial slicing and rebinning as neural data ensures alignment.

ii.
```python
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
if factor == 2:
    position = rebin_1d_mean(position, factor)
pos_bin = discretize_position(position)
```

iii. Verified by using the same trial boundaries and rebinning factor.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from `processing/behavior/BehavioralTimeSeries/lick/data`.

ii.
```python
lick=beh["lick/data"][()][:t_common].astype(np.float32, copy=False),
```

iii. The `lick` variable records lick events at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value mapped to 1, otherwise 0. For 31 Hz sessions, lick is rebinned using logical OR (any lick in a 2-frame window counts as lick).

ii.
```python
lick_binary = (arrays.lick[start:stop_exclusive] > 0).astype(np.int64, copy=False)
if factor == 2:
    lick_binary = rebin_1d_any(lick_binary, factor)
```

iii. Binary output per the instructions. Logical OR rebinning preserves lick events during downsampling.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same trial slicing and rebinning as neural data.

ii. Same slicing with `arrays.lick[start:stop_exclusive]` and same rebinning factor.

iii. Alignment is by construction.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` behavioral signal, `position`, and `Reward` event timestamps. The AI uses `infer_trial_zones()` which combines multiple signals: median position when `reward_zone > 0`, reward event positions, and a majority-vote heuristic with a switch-trial cutoff at trial 30.

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

iii. The paper defines zones A, B, C at specific ranges and switches occur after trial 30. The AI uses multiple signals to robustly infer the active zone.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For each trial: (1) if `reward_zone > 0` for any frame, take median position during those frames and assign closest zone center; (2) else if a reward event occurred, use the reward position; (3) else mark as unknown. Unknown trials are filled using a majority-vote heuristic: the pre-trial-30 majority zone fills unknown pre-30 trials, and post-30 majority fills post-30 trials. If only one unique zone is observed, all unknowns get that zone. Encoded as 0=A, 1=B, 2=C.

ii.
```python
def infer_trial_zones(positions, reward_zone_signal, trial_slices, ...):
    for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
        mask = reward_zone_signal[start:stop_exclusive] > 0
        if np.any(mask):
            zone_pos = float(np.nanmedian(positions[start:stop_exclusive][mask]))
        elif trial_id in reward_pos_by_trial:
            zone_pos = reward_pos_by_trial[trial_id]
        else:
            zone_pos = np.nan
        if np.isfinite(zone_pos):
            observed[trial_id] = int(np.argmin(np.abs(ZONE_CENTERS - zone_pos)))

    split = min(SWITCH_TRIAL_INDEX, n_trials)
    pre_majority = trial_majority_zone(observed, 0, split, ...)
    post_majority = trial_majority_zone(observed, split, n_trials, ...)
    filled[:split][filled[:split] < 0] = pre_majority
    filled[split:][filled[split:] < 0] = post_majority
```

iii. The switch-trial heuristic at trial 30 matches the paper's statement that "each switch occurred after 30 trials."

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from `Reward` event timestamps and the `trial_number` behavioral signal.

ii.
```python
reward_times=beh["Reward/timestamps"][()].astype(np.float64, copy=False),
...
rewarded_trial_ids = set(int(t) for t in arrays.trial_number[reward_frame_idx] if t >= 0)
trial_rewarded = np.array(
    [1 if trial_id in rewarded_trial_ids else 0 for trial_id in range(len(trial_slices))],
    dtype=np.int64,
)
```

iii. The AI uses `trial_number` at reward-event frames to determine which trials received rewards.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward event timestamps are mapped to frame indices via `searchsorted`. The `trial_number` signal at those frames identifies which trial was rewarded. Per-trial output is 0 (omitted) or 1 (rewarded), repeated across all timepoints.

ii.
```python
reward_bin = np.full((t_bins,), int(trial_rewarded[trial_id]), dtype=np.int64)
```

iii. Binary per-trial output matching the instruction specification.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Neural/behavior length mismatch**: Data is cropped to `t_common = min(t_neural, t_behavior)`.
- **Missing reward zone data**: Trials where `reward_zone` is never active get zone positions inferred from reward events or filled by majority vote.
- **Negative environment values**: Filtered out (set to -1 outside valid periods) before computing modal environment.
- **Lick artifacts**: Trials with >35% high-lick frames are removed entirely.
- **Sessions with < 2 valid trials**: Raise a runtime error.

ii.
```python
t_common = min(t_neural, t_behavior)
...
if lick_trial.size and np.mean(lick_trial > 2) > LICK_ARTIFACT_FRACTION:
    lick_artifact[trial_id] = True
...
if len(neural_trials) < 2:
    raise RuntimeError(f"{arrays.session_label}: fewer than 2 valid trials after filtering.")
```

iii. The AI implemented defensive checks found during data exploration, including the paper's lick-artifact QC.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files** via `h5py` — I/O bound, reading large neural data matrices
2. **Trial-by-trial conversion** — iterating over trials within each session for rebinning, discretization, and stacking

ii. N/A

iii. The AI reports full conversion took ~95 seconds for 152 sessions (~0.6s per session), with `h5py` being faster than `pynwb`.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates over each trial sequentially. Operations like `discretize_distance`, `discretize_position`, `discretize_speed` could be applied to the full session arrays before splitting into trials, but variable trial lengths make this awkward without padding/masking.

ii. N/A

iii. The per-trial loop is the natural structure given variable-length trials.

## 13-c. What processing does the code repeat multiple times?

i. Each NWB file is loaded only once, unlike the reference which has separate survey and conversion passes. However, the `infer_sample_files` function reads metadata from all files when in sample mode before selecting 2 sessions, adding a small overhead.

ii.
```python
def infer_sample_files(files):
    for path in files:
        with h5py.File(path, "r") as f:
            env = f["processing/behavior/BehavioralTimeSeries/environment/data"][()]
            rate = float(f["processing/ophys/Deconvolved/plane0/starting_time"].attrs["rate"])
```

iii. The `infer_sample_files` function is only called in `--sample` mode and reads minimal metadata.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Position values are clipped to [0, 450] before computing distance-to-reward-zone and absolute position. Speed values are clipped to [0, inf]. These clipping operations modify raw values that might have been informative (e.g., negative positions indicating the mouse is in the pre-track area).

ii.
```python
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
speed = np.clip(arrays.speed[start:stop_exclusive], 0.0, None)
```

iii. The clipping is a design choice, not strictly unnecessary, but it discards information about positions outside the [0, 450] track range.
