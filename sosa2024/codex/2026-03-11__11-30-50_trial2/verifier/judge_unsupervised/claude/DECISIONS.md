# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all NWB files matching the pattern `data/sub-*/sub-*_behavior+ophys.nwb` using glob, sorted alphabetically. Each NWB file represents one session for one subject. Files are loaded sequentially using h5py to read HDF5 groups directly, extracting behavioral timeseries from `processing/behavior/BehavioralTimeSeries` and neural data from `processing/ophys/Deconvolved/plane0`. The full dataset comprises 152 NWB files across 11 subjects.

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
        session_id = decode_if_bytes(f["general/session_id"][()])
        beh = f["processing/behavior/BehavioralTimeSeries"]
        neural_group = f["processing/ophys/Deconvolved/plane0"]
        seg = f["processing/ophys/ImageSegmentation/PlaneSegmentation"]
        ...
```

iii. The AI justified using direct h5py reads (rather than heavier NWB object materialization) for performance. It identified that each NWB file contains a single session with aligned behavioral and neural streams. All 152 sessions (11 mice x 14 days, minus 2 missing days for m11) are processed.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `general/subject/subject_id` field within each NWB file. A running dictionary maps subject names to indices. Each session is assigned a subject index, and unique subjects are collected into a list.

ii.
```python
if arrays.subject not in subject_to_idx:
    subject_to_idx[arrays.subject] = len(all_subjects)
    all_subjects.append(arrays.subject)

data["subject_idx"].append(subject_to_idx[arrays.subject])
```

iii. The AI noted in CONVERSION_NOTES that the archive contains 11 switch-task mice (m3, m4, m7, m11-m15, m17-m19), consistent with the paper's report of "n = 11 mice" for the switch task.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are processed in the order returned by the sorted glob of NWB file paths. Each session becomes one entry in the `neural`, `input`, and `output` lists.

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

iii. The AI identified 152 NWB files matching the expected count (11 * 14 - 2 for m11's missing days 1-2). This is documented in CONVERSION_NOTES Step 4.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from the binary `trial_start` and `teleport` signals within each session. Rising edges in `trial_start` mark trial onsets, and rising edges in `teleport` mark trial ends (inclusive). Each trial is a contiguous slice `(start_frame, stop_frame_exclusive)`.

ii.
```python
def build_trial_slices(trial_start_signal: np.ndarray, teleport_signal: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(trial_start_signal > 0.5)
    teleports = np.flatnonzero(teleport_signal > 0.5)
    n = min(starts.size, teleports.size)
    slices: list[tuple[int, int]] = []
    for start, stop in zip(starts[:n], teleports[:n]):
        if stop < start:
            continue
        slices.append((int(start), int(stop) + 1))
    if len(slices) < 2:
        raise RuntimeError("Need at least two trials in a session.")
    return slices
```

iii. The AI justified this approach by referencing the paper's use of `trial_start_inds` and `teleport_inds` for trial boundaries. The `+1` on the teleport index makes the slice exclusive, capturing the full trial including the teleport frame.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on lick sensor artifacts. If more than 35% of frames in a trial have a lick value > 2 (indicating erroneous cumulative lick counts), the trial is excluded. This is the only trial-level quality filter applied.

ii.
```python
LICK_ARTIFACT_FRACTION = 0.35

for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    lick_trial = arrays.lick[start:stop_exclusive]
    if lick_trial.size and np.mean(lick_trial > 2) > LICK_ARTIFACT_FRACTION:
        lick_artifact[trial_id] = True

# Later in the trial loop:
if lick_artifact[trial_id]:
    continue
```

iii. The AI referenced `glmUtils.get_timeseries_data` in the reference code, which suppresses lick sensor artifacts when cumulative licks are implausibly high. The threshold of 0.35 (35%) is stated to match the reference code. The paper reports 81/12,376 trials removed (~0.65%); the AI's code removed 69/12,216 (~0.56%), close to the paper value.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `processing/ophys/Deconvolved/plane0/data` in each NWB file, which contains the deconvolved calcium activity (events) after dF/F computation and OASIS deconvolution. The ROI indices come from `processing/ophys/Deconvolved/plane0/rois`.

ii.
```python
neural = neural_group["data"][()].astype(np.float32, copy=False)
roi_ids = neural_group["rois"][()].astype(np.int64, copy=False)
```

iii. The AI justified using `Deconvolved` rather than raw `Fluorescence` because the paper's decoder and SI analyses operate on deconvolved activity (`events`). The NWB archive already provides the deconvolved stream, so re-computing dF/F from raw fluorescence was unnecessary.

## 2-b. How is the `neural` data processed?

i. The neural data is: (1) filtered to keep only curated cells via `iscell`, (2) truncated to match behavioral frame count, (3) transposed from (time, neurons) to (neurons, time), (4) sliced per-trial, and (5) rebinned from 31 Hz to 15.5 Hz for multipane sessions by summing pairs of consecutive frames.

ii.
```python
# Cell filtering at load time:
iscell = seg["iscell"][()][roi_ids, 0] > 0.5
neural = neural[:, iscell]

# Per-trial extraction and rebinning:
neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)
if factor == 2:
    neural_trial = rebin_2d_sum_time_last(neural_trial, factor)

def rebin_2d_sum_time_last(x: np.ndarray, factor: int) -> np.ndarray:
    # Sum within bins for integration
    full = x[:, : n_full * factor].reshape(x.shape[0], n_full, factor).sum(axis=2)
```

iii. The AI documented that deconvolved activity is the paper-equivalent neural signal and that rebinning uses summation (integration) to preserve total event counts when downsampling from 31 Hz to 15.5 Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered using the suite2p `iscell` flag, applied only to ROI indices referenced by the response matrix (`Deconvolved/plane0/rois`). Only cells with `iscell[:, 0] > 0.5` are kept. The paper's additional putative interneuron filter (dF/F-speed correlation > 0.5) was NOT implemented.

ii.
```python
roi_ids = neural_group["rois"][()].astype(np.int64, copy=False)
iscell = seg["iscell"][()][roi_ids, 0] > 0.5
neural = neural[:, iscell]
roi_ids = roi_ids[iscell]
```

iii. The AI documented that the exact dF/F-speed-correlation interneuron filter could not be re-run because the NWB archive does not expose the exact reference dF/F timeseries. Since neuron counts already fall within the paper's reported range (155-1780 vs paper's 155-2172), this omission was accepted. This is documented in CONVERSION_NOTES Steps 10 and 4.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start. Each trial's neural data begins at the `trial_start` frame index and ends at the `teleport` frame index (inclusive). Time bin 0 corresponds to the first frame of the trial.

ii.
```python
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)
```

iii. The instructions specify "Temporally align based on start of the trial." The AI uses the `trial_start` signal edges to define trial onset, matching the paper's `trial_start_inds`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The target temporal resolution is 1/15.5078125 Hz = ~64.48 ms per bin. Sessions natively at 15.5078125 Hz require no rebinning (factor=1). Sessions at 31.015625 Hz (m17, m18 multipane mice) are rebinned by a factor of 2 to achieve the common rate.

ii.
```python
TARGET_RATE_HZ = 15.5078125
TARGET_DT_S = 1.0 / TARGET_RATE_HZ

if math.isclose(arrays.rate_hz, TARGET_RATE_HZ):
    factor = 1
elif math.isclose(arrays.rate_hz, 2.0 * TARGET_RATE_HZ):
    factor = 2
```

iii. The AI noted that ~15.5 Hz is the dominant sampling rate and the paper's effective per-plane rate. The metadata records `time_bin_size = 1000.0 * TARGET_DT_S` (~64.48 ms).

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. This input is not derived from any raw data variable. It is computed synthetically from the frame index within each trial and the known sampling rate.

ii.
```python
t_bins = neural_trial.shape[1]
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
```

iii. The AI creates a monotonically increasing time vector starting at 0.0 for each trial, with increments of `TARGET_DT_S` (~0.0645 s) per bin. This represents elapsed time since trial onset.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. A simple arange multiplied by the time bin duration. No complex processing is involved.

ii.
```python
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
```

iii. Straightforward computation; time_from_start[0] = 0.0 for every trial.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time vector has the same number of bins as the neural data for each trial (`t_bins = neural_trial.shape[1]`), so they are inherently aligned frame-by-frame.

ii.
```python
t_bins = neural_trial.shape[1]
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
```

iii. Both are derived from the same trial slice boundaries and rebinning factor, ensuring exact alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/environment/data` in the NWB file. This contains per-frame environment labels (0 or 1, with -1 for invalid periods).

ii.
```python
environment=beh["environment/data"][()][:t_common].astype(np.float32, copy=False),
```

iii. The AI identified that environment values are 0 (ENV1) or 1 (ENV2), with -1 outside valid trial/imaging periods.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Per trial, the modal (most common) valid environment value is computed, excluding -1 values. This single value is then repeated across all time bins of the trial.

ii.
```python
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    env = arrays.environment[start:stop_exclusive]
    env = env[env >= 0]
    trial_env[trial_id] = mode_int(env, default=0)

# In trial loop:
env_series = np.full((t_bins,), float(trial_env[trial_id]), dtype=np.float32)
```

iii. The AI treats environment as a per-trial variable (matching the instruction "per trial") and uses the mode to handle any noise or transition frames within a trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the enumeration index of the reconstructed trial slices, NOT directly from the NWB `trial number/data` field. The NWB trial number field is loaded but is used indirectly (for reward assignment).

ii.
```python
trial_series = np.full((t_bins,), float(trial_id), dtype=np.float32)
```

Where `trial_id` comes from `for trial_id, (start, stop_exclusive) in enumerate(trial_slices)`.

iii. The AI uses the 0-indexed position in the reconstructed trial list. Since the trial slices are built from trial_start/teleport signals, this should closely match the NWB trial number values. The AI documented keeping "within-session trial number 0-indexed" to match raw NWB values and `glmUtils.get_timeseries_data`.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial index from the enumerated trial slices is cast to float and repeated across all time bins of the trial.

ii.
```python
trial_series = np.full((t_bins,), float(trial_id), dtype=np.float32)
```

iii. No additional processing. The value is a per-trial constant repeated across frames.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/Reward/timestamps` and `trial number/data`. Reward timestamps are matched to frames via `searchsorted`, and trial-level reward outcomes are shifted by one trial.

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

iii. The first trial of each session gets a default value of 0 (no previous reward). The shifted outcome array captures the reward state of the preceding trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. (1) Reward event timestamps are converted to frame indices using `searchsorted`. (2) Each trial is marked rewarded if any reward event maps to a frame within that trial's NWB trial number. (3) The reward array is shifted by one position, with 0 prepended for the first trial. (4) The value is repeated across all time bins.

ii.
```python
def reward_frames_for_session(arrays: SessionArrays) -> np.ndarray:
    idx = np.searchsorted(arrays.timestamps, arrays.reward_times, side="left")
    return np.clip(idx, 0, arrays.timestamps.shape[0] - 1).astype(np.int64, copy=False)

trial_prev_rewarded = np.concatenate([[0], trial_rewarded[:-1]]).astype(np.float32, copy=False)
prev_reward_series = np.full((t_bins,), float(trial_prev_rewarded[trial_id]), dtype=np.float32)
```

iii. This creates a binary per-trial input indicating whether the previous trial was rewarded (1) or omitted (0).

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from `position/data` (position on the track in cm) and the inferred reward zone location for each trial. Reward zone inference uses `reward_zone/data`, `Reward/timestamps`, and position data.

ii.
```python
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
zone_idx = int(trial_zone[trial_id])
dist = compute_distance_to_zone(position, zone_idx)
```

iii. The AI combines position with the per-trial reward zone to compute signed distance to the active zone boundaries.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The signed distance is computed as: negative if before zone start, 0 if inside zone, positive if after zone end. Zone boundaries are hardcoded: A=[80,130], B=[200,250], C=[320,370] cm.

ii.
```python
ZONE_BOUNDS = {0: (80.0, 130.0), 1: (200.0, 250.0), 2: (320.0, 370.0)}

def compute_distance_to_zone(position_cm: np.ndarray, zone_idx: int) -> np.ndarray:
    zone_start, zone_end = ZONE_BOUNDS[zone_idx]
    distance = np.zeros(position_cm.shape, dtype=np.float32)
    before = position_cm < zone_start
    after = position_cm > zone_end
    distance[before] = position_cm[before] - zone_start
    distance[after] = position_cm[after] - zone_end
    return distance
```

iii. The zone bounds match the paper: "zone A, 80-130 cm; zone B, 200-250 cm; zone C, 320-370 cm".

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Continuous distance is discretized into 7 bins following the instructions exactly.

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

iii. The bin edges match the instructions: <-50, -50 to -10, -10 to <0, 0, >0 to +10, +10 to +50, >+50 cm.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The position data used to compute distance is extracted from the same frame indices as the neural data (same trial slice boundaries). For rebinned sessions, position is rebinned by averaging pairs of frames.

ii.
```python
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
if factor == 2:
    position = rebin_1d_mean(position, factor)
dist = compute_distance_to_zone(position, zone_idx)
dist_bin = discretize_distance(dist)
```

iii. Frame-by-frame correspondence between neural and behavioral data ensures alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/position/data` in the NWB file (position in cm along the 450 cm track).

ii.
```python
position=beh["position/data"][()][:t_common].astype(np.float32, copy=False),
```

iii. Position is one of the primary behavioral timeseries.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450] cm and then discretized into 5 equal-sized bins (each 90 cm wide).

ii.
```python
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)

def discretize_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, 0.0, TRACK_LENGTH_CM - 1e-6)
    return np.minimum((clipped / (TRACK_LENGTH_CM / 5.0)).astype(np.int64), 4)
```

iii. The 5 bins of 90 cm each match the instruction "discretized into 5 equal-sized bins" over a 450 cm track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is divided by 90 cm (= 450/5) and truncated to integer, capped at 4. This produces bins: 0=[0,90), 1=[90,180), 2=[180,270), 3=[270,360), 4=[360,450].

ii.
```python
return np.minimum((clipped / (TRACK_LENGTH_CM / 5.0)).astype(np.int64), 4)
```

iii. Straightforward equal-width binning as specified in the instructions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same as distance-to-reward-zone: position is extracted from the same trial frame slice as neural data, and rebinned identically for 31 Hz sessions.

ii.
```python
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
if factor == 2:
    position = rebin_1d_mean(position, factor)
pos_bin = discretize_position(position)
```

iii. Frame-aligned by construction.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/lick/data` in the NWB file, which contains cumulative lick sensor counts per frame.

ii.
```python
lick=beh["lick/data"][()][:t_common].astype(np.float32, copy=False),
```

iii. The AI identified lick as a cumulative sensor count that needs binarization.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The raw lick signal is binarized: any frame with lick > 0 is marked as 1, otherwise 0. For rebinned sessions, a logical OR is applied within each pair of frames.

ii.
```python
lick_binary = (arrays.lick[start:stop_exclusive] > 0).astype(np.int64, copy=False)
if factor == 2:
    lick_binary = rebin_1d_any(lick_binary, factor)

def rebin_1d_any(x: np.ndarray, factor: int) -> np.ndarray:
    out.append(np.any(x[: n_full * factor].reshape(n_full, factor) > 0, axis=1))
```

iii. The AI uses `lick > 0` as the binarization threshold, producing a binary output (0=no lick, 1=lick detected).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick data is extracted from the same trial frame slice as neural data. For 31 Hz sessions, it is rebinned using logical OR to match the neural temporal resolution.

ii.
```python
lick_binary = (arrays.lick[start:stop_exclusive] > 0).astype(np.int64, copy=False)
if factor == 2:
    lick_binary = rebin_1d_any(lick_binary, factor)
```

iii. Frame-aligned by construction; OR preserves lick events during downsampling.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from multiple sources: (1) `reward_zone/data` - a per-frame signal indicating when the animal is in the reward zone, (2) `position/data` - to locate the position when in the zone, (3) `Reward/timestamps` - to identify reward positions for trials where the zone signal is absent.

ii.
```python
def infer_trial_zones(positions, reward_zone_signal, trial_slices, trial_rewarded,
                      reward_frame_idx, trial_by_frame, trial_env):
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
```

iii. The AI uses a multi-stage inference strategy: first from the reward zone signal, then from reward event positions, then by block majority filling for omission trials.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. (1) For each trial, check if the reward_zone signal is active; if so, take the median position when active and assign to nearest zone center. (2) For unrewarded trials without a zone signal, use the position at the reward event. (3) Fill remaining unobserved trials by majority vote within pre-switch (trials 0-29) and post-switch (trials 30+) blocks. The result is a per-trial categorical value (0=A, 1=B, 2=C) repeated across time bins.

ii.
```python
ZONE_CENTERS = np.array([(lo + hi) / 2.0 for lo, hi in ZONE_BOUNDS.values()])
# ... inference logic ...
split = min(SWITCH_TRIAL_INDEX, n_trials)
pre_majority = trial_majority_zone(observed, 0, split, fallback=int(unique_observed[0]))
post_majority = trial_majority_zone(observed, split, n_trials, fallback=post_fallback)
filled[:split][filled[:split] < 0] = pre_majority
filled[split:][filled[split:] < 0] = post_majority

zone_bin = np.full((t_bins,), zone_idx, dtype=np.int64)
```

iii. The switch at trial 30 matches the paper's "Each switch occurred after 30 trials." The block-filling strategy handles omission trials that lack direct zone observations.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/Reward/timestamps` (sparse reward event times) and `trial number/data` (to assign rewards to trials).

ii.
```python
reward_times=beh["Reward/timestamps"][()].astype(np.float64, copy=False),

reward_frame_idx = reward_frames_for_session(arrays)
rewarded_trial_ids = set(int(t) for t in arrays.trial_number[reward_frame_idx] if t >= 0)
trial_rewarded = np.array(
    [1 if trial_id in rewarded_trial_ids else 0 for trial_id in range(len(trial_slices))],
    dtype=np.int64,
)
```

iii. A trial is rewarded (1) if any reward event timestamp falls within it, otherwise omitted (0).

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. (1) Reward timestamps are converted to frame indices via `searchsorted`. (2) The NWB trial number at each reward frame identifies which trial was rewarded. (3) A binary array is constructed: 1 if the trial ID appears in the rewarded set, 0 otherwise. (4) The per-trial value is repeated across all time bins.

ii.
```python
def reward_frames_for_session(arrays: SessionArrays) -> np.ndarray:
    idx = np.searchsorted(arrays.timestamps, arrays.reward_times, side="left")
    return np.clip(idx, 0, arrays.timestamps.shape[0] - 1).astype(np.int64, copy=False)

reward_bin = np.full((t_bins,), int(trial_rewarded[trial_id]), dtype=np.int64)
```

iii. The resulting distribution (~84% rewarded, ~16% omitted) matches the paper's ~85%/15% reward omission rate.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several data quality issues are handled: (1) Neural/behavioral frame count mismatches (up to 1 frame) are resolved by truncating to the shorter of the two. (2) Invalid environment values (-1) are excluded when computing per-trial mode. (3) Missing reward zone observations (on omission trials) are filled by block majority vote. (4) Trials with lick sensor artifacts are excluded. (5) Sessions where `trial_start` or `teleport` edges don't pair up correctly (teleport before start) are skipped with `if stop < start: continue`.

ii.
```python
t_common = min(t_neural, t_behavior)
# ... truncate both to t_common ...

env = env[env >= 0]  # exclude invalid values
trial_env[trial_id] = mode_int(env, default=0)

# In build_trial_slices:
if stop < start:
    continue
```

iii. The AI documented handling 10 sessions with 1-sample neural/behavior length mismatch, and the block-filling strategy for missing zone labels in CONVERSION_NOTES Step 10.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading NWB files via h5py, which involves reading the full neural response matrix (`Deconvolved/plane0/data`) into memory for each session. The full conversion completes in ~95 seconds for 152 sessions (~0.63 s/session).

ii.
```python
neural = neural_group["data"][()].astype(np.float32, copy=False)
```

iii. The AI documented that full-session NWB reads materialize the neural response matrix once per session as the primary bottleneck. Processing plots add overhead but are disabled for full conversion.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` (lines 469-513) iterates over each trial sequentially, performing slicing, rebinning, and discretization. Some of these operations (e.g., environment mode computation, lick artifact detection) could be vectorized across trials.

ii.
```python
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    # Per-trial processing: slicing, clipping, rebinning, discretization
    neural_trial = arrays.neural[start:stop_exclusive].T
    position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
    ...
```

iii. The AI noted vectorized operations within trials (e.g., `np.clip`, `np.bincount`, discretization functions) but the outer trial loop remains sequential. Since trials have variable length, full vectorization would require padding.

## 13-c. What processing does the code repeat multiple times?

i. Position clipping (`np.clip(..., 0.0, TRACK_LENGTH_CM)`) is done both in the main trial loop and again in the plotting code for `show_processing` mode. The plotting code re-extracts and re-processes the example trial data from raw arrays rather than reusing already-processed trial data.

ii.
```python
# In main trial loop:
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)

# In plotting (lines 539-544):
example_position=np.clip(arrays.position[trial_slices[example_idx][0]:trial_slices[example_idx][1]], 0.0, TRACK_LENGTH_CM)
```

iii. The duplication is only in the plotting path, which is not used during full conversion.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The `speed` output is computed and discretized, but the reference paper masks out samples with speed < 2 cm/s rather than including speed as a decoded variable. However, the instructions explicitly request speed as a decoder output, so this is necessary for the task. (2) The `observed_zone_position` array tracks zone observation positions per trial for metadata/debugging but is not used in the output data structure. (3) Per-trial session metadata (conversion_seconds, etc.) adds overhead for documentation but doesn't affect the decoder data.

ii.
```python
# observed_zone_position stored in session_info but not in output:
session_info = {
    ...
    "observed_zone_positions_cm": observed_zone_position.tolist(),
    ...
}
```

iii. The AI prioritized comprehensive metadata for debugging and verification purposes.
