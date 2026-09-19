# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All NWB files are discovered via a sorted glob pattern `data/sub-*/sub-*_behavior+ophys.nwb`. Each file is loaded using `h5py.File` (not pynwb). All behavioral and neural data arrays are read in a single pass per session through the `load_session_arrays` function.

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
        ...
        neural = neural_group["data"][()].astype(np.float32, copy=False)
```

iii. From CONVERSION_NOTES Step 5: "Use all 152 NWB sessions from the 11 switch-task mice." The glob pattern finds all NWB files in all sub-* directories. h5py was chosen over pynwb for faster, more direct file access.

## 1-b. How are the data split into subjects?

i. Subject identity is read from the NWB metadata field `general/subject/subject_id`. A subject-to-index mapping is built as sessions are processed, accumulating unique subject IDs.

ii.
```python
subject = decode_if_bytes(f["general/subject/subject_id"][()])
...
if arrays.subject not in subject_to_idx:
    subject_to_idx[arrays.subject] = len(all_subjects)
    all_subjects.append(arrays.subject)
```

iii. Each NWB file embeds its subject ID. The 11 unique subjects match the paper's switch-task cohort.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session label is constructed from subject + session_id extracted from the NWB metadata.

ii.
```python
session_id = decode_if_bytes(f["general/session_id"][()])
session_label = f"{subject}_ses-{session_id}"
```

iii. From CONVERSION_NOTES: "152 NWB files = 11 * 14 - 2", consistent with the paper's 14 imaging days per mouse minus m11's missing first two days.

## 1-d. How are the data split into trials?

i. Trial boundaries are reconstructed from the `trial_start` and `teleport` behavior time series. All frames where `trial_start > 0.5` define trial onsets, and all frames where `teleport > 0.5` define trial ends. Starts and teleports are paired elementwise. The trial slice includes the teleport frame: `(start, stop + 1)`.

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
    return slices
```

iii. From CONVERSION_NOTES Step 5: "Reconstruct trials from `trial_start` and `teleport`: This matches the reference trial definition more closely than relying only on trial-number changes."

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on lick artifacts: if more than 35% of frames in a trial have lick count > 2, the trial is removed. This matches the paper's lick-artifact removal procedure described in the Methods.

ii.
```python
LICK_ARTIFACT_FRACTION = 0.35  # Matches glmUtils.get_timeseries_data in the reference code.
...
lick_trial = arrays.lick[start:stop_exclusive]
if lick_trial.size and np.mean(lick_trial > 2) > LICK_ARTIFACT_FRACTION:
    lick_artifact[trial_id] = True
...
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    if lick_artifact[trial_id]:
        continue
```

iii. From CONVERSION_NOTES: "Lick-artifact trial removal in converted data: 69 / 12,216 = 0.56%, close to the paper's reported ~0.65%." The threshold of 0.35 matches `glmUtils.get_timeseries_data` in the reference code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `processing/ophys/Deconvolved/plane0/data` in the NWB file. This is suite2p's pre-computed deconvolved calcium activity.

ii.
```python
neural_group = f["processing/ophys/Deconvolved/plane0"]
neural = neural_group["data"][()].astype(np.float32, copy=False)
```

iii. From CONVERSION_NOTES Step 5: "Use NWB `Deconvolved` as the neural signal: The paper's decoder and SI analyses operate on deconvolved activity after dF/F, and this stream is already provided in the archive."

## 2-b. How is the `neural` data processed?

i. The raw Deconvolved data is loaded, filtered by `iscell`, and for 31 Hz sessions (m17, m18) rebinned by factor 2 to the common 15.5078125 Hz rate by summing pairs of adjacent time bins. No additional signal processing (dF/F, smoothing, OASIS) is applied, as the NWB Deconvolved stream is used directly.

ii.
```python
neural = neural_group["data"][()].astype(np.float32, copy=False)
roi_ids = neural_group["rois"][()].astype(np.int64, copy=False)
iscell = seg["iscell"][()][roi_ids, 0] > 0.5
neural = neural[:, iscell]
...
if factor == 2:
    neural_trial = rebin_2d_sum_time_last(neural_trial, factor)
```

iii. From CONVERSION_NOTES Step 10: "equivalent archive representation of aligned behavior + neural data." The AI viewed the NWB Deconvolved as a suitable proxy for the paper's events.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by the `iscell` flag from suite2p's `PlaneSegmentation` table, restricted to ROI indices referenced by the `Deconvolved/plane0/rois` array. No additional putative interneuron filtering (dF/F-speed correlation) is performed.

ii.
```python
roi_ids = neural_group["rois"][()].astype(np.int64, copy=False)
iscell = seg["iscell"][()][roi_ids, 0] > 0.5
neural = neural[:, iscell]
```

iii. From CONVERSION_NOTES Step 10: "Exact dF/F-speed-correlation interneuron filter: documented as an archive-level limitation. The shared NWB files provide the deconvolved response stream used for decoding, but not the exact reference dF/F timeseries needed to reproduce that curation step bit-for-bit."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial start by slicing the session-level neural array at trial boundaries defined by `trial_start` and `teleport` signals. Since neural and behavior data share the same time axis in the NWB, no additional alignment is needed.

ii.
```python
neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)
```

iii. From CONVERSION_NOTES: "Trial alignment should use `trial_start_inds` as the start and `teleport_inds` as the end of each trial."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The target temporal resolution is 15.5078125 Hz (~64.48 ms per bin). Sessions recorded at 15.5 Hz are kept as-is. Sessions recorded at ~31 Hz (mice m17, m18 with two-plane interleaved imaging) are rebinned by factor 2: neural data is summed, behavioral data is averaged (position, speed) or OR-ed (lick).

ii.
```python
TARGET_RATE_HZ = 15.5078125
...
if math.isclose(arrays.rate_hz, TARGET_RATE_HZ):
    factor = 1
elif math.isclose(arrays.rate_hz, 2.0 * TARGET_RATE_HZ):
    factor = 2
...
if factor == 2:
    neural_trial = rebin_2d_sum_time_last(neural_trial, factor)
    position = rebin_1d_mean(position, factor)
    speed = rebin_1d_mean(speed, factor)
    lick_binary = rebin_1d_any(lick_binary, factor)
```

iii. From CONVERSION_NOTES Step 5: "Standardize all sessions to a common 15.5078125 Hz bin size: This is the dominant dataset rate and the paper's effective per-plane sampling rate."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the bin index within the trial and the target sampling rate (`TARGET_DT_S = 1/15.5078125`). Not derived from raw timestamps.

ii.
```python
t_bins = neural_trial.shape[1]
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
```

iii. After rebinning to a consistent rate, using bin_index * dt is equivalent to using actual timestamps but ensures perfect consistency with the reported time bin size.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Multiply each bin's index by the time step `TARGET_DT_S = 1/15.5078125 s`. This produces a time series starting at 0 and incrementing uniformly.

ii.
```python
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
```

iii. Straightforward computation from bin index.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both are derived from the same trial slice length (`t_bins = neural_trial.shape[1]`), so they are aligned by construction.

ii.
```python
t_bins = neural_trial.shape[1]
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
```

iii. Using the neural trial's bin count ensures exact alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series in the NWB.

ii.
```python
environment=beh["environment/data"][()][:t_common].astype(np.float32, copy=False),
...
env = arrays.environment[start:stop_exclusive]
env = env[env >= 0]
trial_env[trial_id] = mode_int(env, default=0)
```

iii. The environment variable is 0 or 1, corresponding to ENV1 and ENV2 in the paper.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the modal (most common) valid environment value (excluding negative values) is computed and used as a constant per-trial input replicated across all time bins.

ii.
```python
trial_env[trial_id] = mode_int(env, default=0)
...
env_series = np.full((t_bins,), float(trial_env[trial_id]), dtype=np.float32)
```

iii. Taking the mode handles any edge-case negative values that appear outside valid trial periods.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the trial loop index (sequential within-session trial counter, 0-indexed).

ii.
```python
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    ...
    trial_series = np.full((t_bins,), float(trial_id), dtype=np.float32)
```

iii. From CONVERSION_NOTES Step 5: "Keep within-session trial number 0-indexed: This matches raw NWB values and `glmUtils.get_timeseries_data`."

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index is used directly as a constant per-trial value, replicated across all time bins. No additional processing.

ii.
```python
trial_series = np.full((t_bins,), float(trial_id), dtype=np.float32)
```

iii. Straightforward sequential indexing.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps. Reward event timestamps are mapped to frame indices using `searchsorted`, and the `trial number` signal determines which trial each reward belongs to.

ii.
```python
reward_times=beh["Reward/timestamps"][()].astype(np.float64, copy=False),
...
reward_frame_idx = reward_frames_for_session(arrays)
rewarded_trial_ids = set(int(t) for t in arrays.trial_number[reward_frame_idx] if t >= 0)
trial_rewarded = np.array(
    [1 if trial_id in rewarded_trial_ids else 0 for trial_id in range(len(trial_slices))],
    dtype=np.int64,
)
```

iii. The Reward time series has separate timestamps from behavior. The `trial_number` signal at the reward frame identifies the trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A per-trial binary reward array is constructed. The previous trial's outcome is obtained by shifting this array by one position, with the first trial defaulting to 0.

ii.
```python
trial_prev_rewarded = np.concatenate([[0], trial_rewarded[:-1]]).astype(np.float32, copy=False)
...
prev_reward_series = np.full((t_bins,), float(trial_prev_rewarded[trial_id]), dtype=np.float32)
```

iii. From the instructions: "Previous trial outcome (binary, omitted = 0, rewarded = 1, per trial)."

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the inferred reward zone location for each trial. The reward zone is inferred from the `reward_zone` signal and reward event positions using a majority-vote and block-filling approach based on the known switch trial index (trial 30).

ii.
```python
trial_zone, observed_zone_position = infer_trial_zones(
    positions=np.clip(arrays.position, 0.0, TRACK_LENGTH_CM),
    reward_zone_signal=arrays.reward_zone_signal,
    trial_slices=trial_slices,
    ...)
...
zone_idx = int(trial_zone[trial_id])
dist = compute_distance_to_zone(position, zone_idx)
```

iii. From CONVERSION_NOTES Step 5: "Infer reward-zone location from rewarded trials and fill omission trials within stable blocks."

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the signed distance from the animal's position to the nearest edge of the reward zone is computed. Distance is 0 inside the zone, negative before it, positive after it.

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

iii. This matches the paper's concept of reward-relative position.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit Boolean conditions matching the instruction-specified bin edges: <-50, -50 to -10, -10 to <0, 0, >0 to 10, 10 to 50, >50.

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

iii. The bin definitions match the instruction specifications exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position is sliced from the same trial boundaries as neural data, optionally rebinned with the same factor, so alignment is by construction.

ii.
```python
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
...
if factor == 2:
    position = rebin_1d_mean(position, factor)
...
dist = compute_distance_to_zone(position, zone_idx)
```

iii. Both neural and behavioral signals share the same time axis and trial boundaries.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position=beh["position/data"][()][:t_common].astype(np.float32, copy=False),
...
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
```

iii. The `position` variable records the animal's position on the VR corridor in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450] cm, then discretized into 5 equal bins of 90 cm each by dividing by 90 and truncating to integer.

ii.
```python
def discretize_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, 0.0, TRACK_LENGTH_CM - 1e-6)
    return np.minimum((clipped / (TRACK_LENGTH_CM / 5.0)).astype(np.int64), 4)
```

iii. The 450 cm track divided into 5 bins gives 90 cm per bin, matching the instructions.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is clipped to [0, 449.9999] and divided by 90 to get bin indices 0-4. The `np.minimum(..., 4)` caps any edge values.

ii.
```python
clipped = np.clip(position_cm, 0.0, TRACK_LENGTH_CM - 1e-6)
return np.minimum((clipped / (TRACK_LENGTH_CM / 5.0)).astype(np.int64), 4)
```

iii. This produces 5 equal-width bins as specified: 0-90, 90-180, 180-270, 270-360, 360-450 cm.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same trial slicing and optional rebinning as neural data ensures alignment.

ii.
```python
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
if factor == 2:
    position = rebin_1d_mean(position, factor)
```

iii. Shared time axis and rebinning factor.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick=beh["lick/data"][()][:t_common].astype(np.float32, copy=False),
```

iii. The `lick` variable records lick events at each timepoint.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0. For rebinned sessions, logical OR is applied across pairs of frames.

ii.
```python
lick_binary = (arrays.lick[start:stop_exclusive] > 0).astype(np.int64, copy=False)
...
if factor == 2:
    lick_binary = rebin_1d_any(lick_binary, factor)
```

iii. The instructions specify binary output (no/yes). The OR-rebinning preserves lick events across bin boundaries.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same trial slicing and rebinning factor ensures alignment.

ii.
```python
lick_binary = (arrays.lick[start:stop_exclusive] > 0).astype(np.int64, copy=False)
```

iii. Shared time axis.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` behavior time series and `position` time series. The reward zone signal indicates when the animal is in the active zone. Reward event positions provide additional evidence.

ii.
```python
def infer_trial_zones(...):
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

iii. The zone position is matched to the nearest zone center among A (105 cm), B (225 cm), C (345 cm).

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For each trial, the median position when reward_zone > 0 identifies the zone. Trials with no reward_zone signal (typically omission trials) are filled using a majority-vote block-filling approach: trials before the switch (trial 30) get the pre-switch majority zone, and trials after the switch get the post-switch majority zone.

ii.
```python
split = min(SWITCH_TRIAL_INDEX, n_trials)
pre_majority = trial_majority_zone(observed, 0, split, fallback=int(unique_observed[0]))
post_majority = trial_majority_zone(observed, split, n_trials, fallback=post_fallback)
filled[:split][filled[:split] < 0] = pre_majority
filled[split:][filled[split:] < 0] = post_majority
```

iii. The paper describes reward zone switches occurring after trial 30 on switch days. Using this known switch point enables accurate filling of missing zone labels.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps and the `trial number` signal. Reward event timestamps are mapped to frame indices, and the trial_number at each reward frame identifies the rewarded trial.

ii.
```python
reward_frame_idx = reward_frames_for_session(arrays)
rewarded_trial_ids = set(int(t) for t in arrays.trial_number[reward_frame_idx] if t >= 0)
trial_rewarded = np.array(
    [1 if trial_id in rewarded_trial_ids else 0 for trial_id in range(len(trial_slices))],
    dtype=np.int64,
)
```

iii. A trial is rewarded if any reward event timestamp falls within it.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Binary per-trial output: 1 if the trial contains a reward event, 0 otherwise. The value is constant across all timepoints in the trial.

ii.
```python
reward_bin = np.full((t_bins,), int(trial_rewarded[trial_id]), dtype=np.int64)
```

iii. The instructions specify binary reward outcome (omitted = 0, rewarded = 1).

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- **Neural/behavior length mismatch**: If neural and behavior arrays have different lengths, both are cropped to the common minimum (`t_common = min(t_neural, t_behavior)`).
- **Lick artifacts**: Trials with >35% lick artifact frames are removed.
- **Missing reward zone data**: Omission trials without reward_zone signal are filled via block-majority-vote based on the switch trial index.
- **Invalid environment values**: Negative values are excluded when computing the trial's modal environment.
- **Invalid trial pairings**: Trial slices where stop < start are skipped.
- **Sessions with too few valid trials**: RuntimeError raised if fewer than 2 trials remain after filtering.

ii.
```python
t_common = min(t_neural, t_behavior)
...
if stop < start:
    continue
...
if len(neural_trials) < 2:
    raise RuntimeError(f"{arrays.session_label}: fewer than 2 valid trials after filtering.")
```

iii. From CONVERSION_NOTES Step 10: "handled 10 sessions with a 1-sample neural/behavior length mismatch by cropping to the common minimum length before trial parsing."

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files** via h5py and reading large neural arrays from disk.
2. **Full conversion loop** iterating over all 152 sessions.

The full conversion completed in ~95 seconds for 152 sessions (~0.6 s/session).

ii. N/A

iii. From conversion_full_out.txt: individual session conversion times range from 0.2 to 0.7 seconds, with larger sessions (more neurons) taking longer.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates over each trial sequentially. Some operations (discretization, distance computation) could be applied to full session arrays before splitting into trials. The `infer_trial_zones` function loops over trials to compute observed positions.

ii. N/A

iii. The per-trial loop is a natural structure for variable-length trials and the overall conversion is already fast (~0.6 s/session).

## 13-c. What processing does the code repeat multiple times?

i. The code is single-pass: each NWB file is loaded once, and all processing happens in that pass. There is no separate survey step that would reload files. The `infer_sample_files` function does read environment and rate from all files when in sample mode, but this is a lightweight check.

ii. N/A

iii. The single-pass design avoids redundant I/O.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `observed_zone_position` for each trial (used only for metadata/logging). The `session_info` dict stores per-session statistics that are recorded in metadata but not used by the decoder. The `scanning` and `trial_number` arrays are loaded but `scanning` is not used as an output (it's always 1).

ii.
```python
scanning=beh["scanning/data"][()][:t_common].astype(np.float32, copy=False),
...
"observed_zone_positions_cm": observed_zone_position.tolist(),
```

iii. These are minor overheads useful for debugging and documentation but not strictly necessary for the decoder.
