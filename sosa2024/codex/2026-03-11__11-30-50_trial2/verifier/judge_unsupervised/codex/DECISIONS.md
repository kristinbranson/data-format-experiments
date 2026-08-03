# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script discovers every NWB file under `data/sub-*/`, then iterates one file per session. Each file is opened with `h5py`, the needed behavior and ophys arrays are read into a `SessionArrays` object, and the per-session outputs are appended to the global `data` dictionary.

ii. ```python
def list_session_files() -> list[str]:
    files = sorted(glob.glob(os.path.join("data", "sub-*", "sub-*_behavior+ophys.nwb")))
    if not files:
        raise FileNotFoundError("No NWB files found under data/sub-*/")
    return files

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

iii. In `CONVERSION_NOTES.md` Step 5, the agent says the key decision is to use all 152 NWB sessions from the 11 switch-task mice. The trajectory also says the shared dataset is already in NWB and mostly frame-aligned, so the agent treated direct NWB loading as the archive-equivalent way to access the full dataset.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by NWB file metadata. The script reads `general/subject/subject_id`, uses the first occurrence of each subject to build `subjects`, and stores one `subject_idx` per session.

ii. ```python
subject = decode_if_bytes(f["general/subject/subject_id"][()])

if arrays.subject not in subject_to_idx:
    subject_to_idx[arrays.subject] = len(all_subjects)
    all_subjects.append(arrays.subject)

data["subject_idx"].append(subject_to_idx[arrays.subject])
data["subjects"] = all_subjects
```

iii. `CONVERSION_NOTES.md` Step 2 documents 11 subject directories and Step 9 says the converted dataset should contain those same 11 mice. The split is therefore driven by NWB subject IDs rather than by any behavioral variable.

## 1-c. How are the data split into sessions?

i. Sessions are split one-to-one by NWB files. Each file is treated as one session, and the script forms a session label from subject ID plus NWB `session_id`.

ii. ```python
session_id = decode_if_bytes(f["general/session_id"][()])
session_label = f"{subject}_ses-{session_id}"

for session_idx, path in enumerate(files):
    arrays = load_session_arrays(path)
```

iii. In `CONVERSION_NOTES.md` Step 2, the agent notes the archive contains one NWB file per subject-session and inventories 152 files. Step 4 resolves session counts by matching the paper’s 14-day schedule with the missing first two imaging days for `m11`.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from binary framewise `trial_start` and `teleport` signals. The script pairs start frames with teleport frames, keeps slices from `start` through `teleport + 1`, and requires at least two valid trials per session.

ii. ```python
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

iii. The notes repeatedly justify this with the reference convention `trial_start_inds` and `teleport_inds`. Step 4 explicitly says NWB trial boundaries were reconstructed from `trial_start` and `teleport` to match the paper’s lap definition.

## 1-e. How are trials filtered based on quality controls?

i. The script removes only lick-artifact trials. A trial is flagged if more than 35% of its frames have `lick > 2`, and flagged trials are skipped entirely. Sessions with fewer than two remaining trials error out.

ii. ```python
LICK_ARTIFACT_FRACTION = 0.35

lick_artifact = np.zeros((len(trial_slices),), dtype=bool)
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    lick_trial = arrays.lick[start:stop_exclusive]
    if lick_trial.size and np.mean(lick_trial > 2) > LICK_ARTIFACT_FRACTION:
        lick_artifact[trial_id] = True

for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    if lick_artifact[trial_id]:
        continue

if len(neural_trials) < 2:
    raise RuntimeError(f"{arrays.session_label}: fewer than 2 valid trials after filtering.")
```

iii. `CONVERSION_NOTES.md` Step 5 says this threshold was chosen to match `glmUtils.get_timeseries_data` and Step 9 highlights the sanity check that 69 removed trials is close to the paper’s reported lick-QC removal rate.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The script derives `neural` directly from `processing/ophys/Deconvolved/plane0/data`, then uses `processing/ophys/Deconvolved/plane0/rois` and `processing/ophys/ImageSegmentation/PlaneSegmentation/iscell` to keep only curated ROIs referenced by the response matrix.

ii. ```python
neural_group = f["processing/ophys/Deconvolved/plane0"]
seg = f["processing/ophys/ImageSegmentation/PlaneSegmentation"]

neural = neural_group["data"][()].astype(np.float32, copy=False)
roi_ids = neural_group["rois"][()].astype(np.int64, copy=False)
iscell = seg["iscell"][()][roi_ids, 0] > 0.5
neural = neural[:, iscell]
```

iii. In Step 5 of the notes, the agent states the neural decoder signal should be the paper-equivalent deconvolved activity and explicitly chooses NWB `Deconvolved` instead of recomputing dF/F plus deconvolution from fluorescence and neuropil.

## 2-b. How is the `neural` data processed?

i. The script minimally processes neural data: crop behavior and neural streams to a common length, filter ROIs by `iscell`, transpose each trial to `(neurons, time)`, and rebin only 31.015625 Hz sessions by summing adjacent samples in time.

ii. ```python
t_common = min(t_neural, t_behavior)
neural=neural[:t_common]

neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)

if factor == 2:
    neural_trial = rebin_2d_sum_time_last(neural_trial, factor)

def rebin_2d_sum_time_last(x: np.ndarray, factor: int) -> np.ndarray:
    full = x[:, : n_full * factor].reshape(x.shape[0], n_full, factor).sum(axis=2)
```

iii. The notes justify this as using the archive’s already deconvolved signal plus a harmonization step to a common 15.5078125 Hz timebase. The agent explicitly documents that it did not rerun the reference dF/F and OASIS pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural QC in the script is limited to the Suite2p `iscell` mask restricted to the ROI ids actually referenced by `Deconvolved/plane0/rois`. The script does not implement the paper’s extra speed-correlation-based interneuron exclusion, speed masking, or NaN-based sample filtering.

ii. ```python
roi_ids = neural_group["rois"][()].astype(np.int64, copy=False)
iscell = seg["iscell"][()][roi_ids, 0] > 0.5
neural = neural[:, iscell]
```

iii. `CONVERSION_NOTES.md` Step 10 says the exact dF/F-speed-correlation interneuron filter was not rerun and was treated as an archive-level limitation. Earlier in Step 5, the agent had planned to exclude those cells if feasible, but the final code does not do so.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to `trial_start`. For each reconstructed `(start, stop_exclusive)` pair, the script slices neural frames beginning at the trial-start frame, keeps frames through the teleport endpoint, and then defines time zero at the first bin in that slice.

ii. ```python
trial_slices = build_trial_slices(arrays.trial_start, arrays.teleport)

for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)
    t_bins = neural_trial.shape[1]
    time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
```

iii. The task instructions say to align on start of trial, and the notes say the reference code uses `trial_start_inds` and `teleport_inds`. The agent cites that as the main reason for this alignment choice.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a common 15.5078125 Hz sampling rate, i.e. `1 / 15.5078125` s per bin, stored as milliseconds in metadata. Sessions already at that rate are left unchanged; 31.015625 Hz sessions are rebinned by a factor of 2.

ii. ```python
TARGET_RATE_HZ = 15.5078125
TARGET_DT_S = 1.0 / TARGET_RATE_HZ

if math.isclose(arrays.rate_hz, TARGET_RATE_HZ):
    factor = 1
elif math.isclose(arrays.rate_hz, 2.0 * TARGET_RATE_HZ):
    factor = 2

"time_bin_size": 1000.0 * TARGET_DT_S,
```

iii. Step 4 of the notes says the archive has both 15.5078125 Hz and 31.015625 Hz sessions, while the paper describes an effective per-plane rate of about 15.5 Hz. Step 5 therefore commits to a common 15.5078125 Hz output bin size.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is not taken from an explicit raw time-from-start variable. The script derives it from the reconstructed trial slice length and the fixed target sampling interval, after any rebinning. Raw timestamps are used only to place reward events onto frames, not to compute this input directly.

ii. ```python
timestamps = beh["position/timestamps"][()].astype(np.float64, copy=False)

t_bins = neural_trial.shape[1]
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
```

iii. In Step 5 of the notes, the agent describes this variable as “reconstructed trial-relative elapsed time” and says the trials should be aligned to the first frame of the trial.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. After trial slicing and optional factor-2 rebinning, the script creates a zero-based evenly spaced vector with one entry per neural bin: `0, dt, 2*dt, ...`.

ii. ```python
if factor == 2:
    neural_trial = rebin_2d_sum_time_last(neural_trial, factor)

t_bins = neural_trial.shape[1]
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
```

iii. The notes justify this as the simplest way to represent elapsed time at the common decoder timebase once all trials are trial-start aligned.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned exactly bin-for-bin with the neural trial array. The number of time entries is `neural_trial.shape[1]`, so every neural time bin has one matching elapsed-time value.

ii. ```python
t_bins = neural_trial.shape[1]
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)

input_trial = np.stack(
    [time_from_start, env_series, trial_series, prev_reward_series],
    axis=0,
)
```

iii. This follows the task instruction to align the decoder variables to the start of each trial. The notes say all trial-wise inputs were repeated or generated at the same frame-aligned timebase as the neural data.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/environment/data`.

ii. ```python
environment=beh["environment/data"][()][:t_common].astype(np.float32, copy=False)

env = arrays.environment[start:stop_exclusive]
env = env[env >= 0]
trial_env[trial_id] = mode_int(env, default=0)
```

iii. In Step 5 of the notes, the agent maps this raw variable directly to the decoder input and says valid values are the binary environment labels for `ENV1` and `ENV2`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The script removes negative invalid samples, takes the modal nonnegative environment value within each trial, converts that to a per-trial integer, and repeats it across all time bins of the trial.

ii. ```python
env = arrays.environment[start:stop_exclusive]
env = env[env >= 0]
trial_env[trial_id] = mode_int(env, default=0)

env_series = np.full((t_bins,), float(trial_env[trial_id]), dtype=np.float32)
```

iii. Step 5 of the notes explicitly says the environment should be a per-trial modal value repeated across bins. The justification is consistency with `get_trial_types` and the task’s request for a per-trial environment label.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. In the final code it is effectively derived from the order of reconstructed trial slices, not directly from the raw `trial number/data` stream. The raw trial-number signal is loaded and used for reward assignment and reward-zone inference, but the decoder input itself comes from `enumerate(trial_slices)`.

ii. ```python
trial_number=beh["trial number/data"][()][:t_common].astype(np.float32, copy=False)

for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    ...
    trial_series = np.full((t_bins,), float(trial_id), dtype=np.float32)
```

iii. The notes say the agent wanted a within-session 0-indexed trial number matching `glmUtils.get_timeseries_data`. That is the rationale for using the reconstructed trial index instead of copying the raw framewise `trial number` values directly.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The processing is just to take the reconstructed per-session trial index, treat it as a scalar for that trial, and repeat it across all bins in the trial.

ii. ```python
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    ...
    trial_series = np.full((t_bins,), float(trial_id), dtype=np.float32)
```

iii. Step 5 of the notes says “Keep native 0-indexing to match code conventions,” which is the stated justification for this representation.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from reward timestamps mapped onto trial identities. Concretely, the script uses `processing/behavior/BehavioralTimeSeries/Reward/timestamps`, the behavior timestamps used to place those rewards on frames, and the framewise `trial number/data`.

ii. ```python
reward_times=beh["Reward/timestamps"][()].astype(np.float64, copy=False)

def reward_frames_for_session(arrays: SessionArrays) -> np.ndarray:
    idx = np.searchsorted(arrays.timestamps, arrays.reward_times, side="left")
    return np.clip(idx, 0, arrays.timestamps.shape[0] - 1).astype(np.int64, copy=False)

rewarded_trial_ids = set(int(t) for t in arrays.trial_number[reward_frame_idx] if t >= 0)
```

iii. Step 5 in the notes maps reward events to trial outcomes and then defines previous-trial outcome as a one-trial shift of that derived outcome.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The script first marks each trial as rewarded if any reward timestamp lands in that trial, otherwise omitted. It then shifts that per-trial binary vector back by one trial, inserting `0` for the first trial, and repeats the previous-outcome value across all bins of the current trial.

ii. ```python
trial_rewarded = np.array(
    [1 if trial_id in rewarded_trial_ids else 0 for trial_id in range(len(trial_slices))],
    dtype=np.int64,
)
trial_prev_rewarded = np.concatenate([[0], trial_rewarded[:-1]]).astype(np.float32, copy=False)
prev_reward_series = np.full((t_bins,), float(trial_prev_rewarded[trial_id]), dtype=np.float32)
```

iii. The notes describe this as a user-requested decoder input rather than a paper decoder variable, with the first trial defaulting to omitted because there is no prior trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from trial position plus an inferred active reward-zone label. Position comes from `processing/behavior/BehavioralTimeSeries/position/data`; zone identity is inferred from `reward_zone/data`, reward timestamps, framewise trial numbers, and the hard-coded zone geometry.

ii. ```python
position=beh["position/data"][()][:t_common].astype(np.float32, copy=False)
reward_zone_signal=beh["reward_zone/data"][()][:t_common].astype(np.float32, copy=False)

trial_zone, observed_zone_position = infer_trial_zones(
    positions=np.clip(arrays.position, 0.0, TRACK_LENGTH_CM),
    reward_zone_signal=arrays.reward_zone_signal,
    reward_frame_idx=reward_frame_idx,
    trial_by_frame=arrays.trial_number.astype(np.int64, copy=False),
    trial_env=trial_env,
)
```

iii. Step 5 of the notes says reward-zone location had to be inferred from the NWB export because omission trials often lacked direct zone occupancy. That inferred zone is then used for reward-relative outputs such as distance-to-zone.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each trial, the script clips position into `[0, 450]`, uses the trial’s inferred zone bounds, and computes signed distance to the nearest point in that zone: negative before the zone, zero inside it, positive after it.

ii. ```python
def compute_distance_to_zone(position_cm: np.ndarray, zone_idx: int) -> np.ndarray:
    zone_start, zone_end = ZONE_BOUNDS[zone_idx]
    distance = np.zeros(position_cm.shape, dtype=np.float32)
    before = position_cm < zone_start
    after = position_cm > zone_end
    distance[before] = position_cm[before] - zone_start
    distance[after] = position_cm[after] - zone_end
    return distance

dist = compute_distance_to_zone(position, zone_idx)
```

iii. The notes state this choice was made to satisfy the decoder-task definition “distance to any location in the reward zone,” while still using the paper’s A/B/C reward-zone boundaries.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is thresholded into the seven requested bins with exact boundaries at `-50`, `-10`, `0`, `10`, and `50` cm.

ii. ```python
out[distance_cm < -50.0] = 0
out[(distance_cm >= -50.0) & (distance_cm < -10.0)] = 1
out[(distance_cm >= -10.0) & (distance_cm < 0.0)] = 2
out[distance_cm == 0.0] = 3
out[(distance_cm > 0.0) & (distance_cm <= 10.0)] = 4
out[(distance_cm > 10.0) & (distance_cm <= 50.0)] = 5
out[distance_cm > 50.0] = 6
```

iii. This is taken directly from the task instructions, and the notes say the discretization was intentionally matched to the decoder specification rather than to the paper’s 10 cm spatial bins.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from the same per-trial position slice used for each neural trial, after any rebinning. The resulting distance bin vector has exactly the same number of time bins as `neural_trial`.

ii. ```python
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
if factor == 2:
    position = rebin_1d_mean(position, factor)

t_bins = neural_trial.shape[1]
dist = compute_distance_to_zone(position, zone_idx)
dist_bin = discretize_distance(dist)
```

iii. Step 10 of the notes says the agent manually spot-checked output alignment against raw NWB trials with `np.allclose`, including this reward-relative distance output.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/position/data`.

ii. ```python
position=beh["position/data"][()][:t_common].astype(np.float32, copy=False)
...
pos_bin = discretize_position(position)
```

iii. The notes map absolute position directly from the NWB position trace and note that the 450 cm corridor length comes from the paper.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The script clips position to the track range, optionally averages adjacent pairs for 31 Hz sessions, and then converts the continuous position into five equal-width bins over the 450 cm corridor.

ii. ```python
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
if factor == 2:
    position = rebin_1d_mean(position, factor)

def discretize_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, 0.0, TRACK_LENGTH_CM - 1e-6)
    return np.minimum((clipped / (TRACK_LENGTH_CM / 5.0)).astype(np.int64), 4)
```

iii. The notes justify the five-bin discretization as a required departure from the paper’s 45 spatial bins because the task explicitly asks for 5 equal-sized bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The 450 cm track is divided into five equal 90 cm bins, labeled `0` through `4`.

ii. ```python
return np.minimum((clipped / (TRACK_LENGTH_CM / 5.0)).astype(np.int64), 4)
```

iii. This follows the decoder-task instruction exactly and is described that way in the notes.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is built from the same per-trial, trial-start-aligned position slice as the neural data and, after optional rebinning, has one value per neural time bin.

ii. ```python
neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
if factor == 2:
    position = rebin_1d_mean(position, factor)
pos_bin = discretize_position(position)
```

iii. The notes say all behavioral outputs were aligned to the same frame-aligned trial representation as the neural data, with separate raw-vs-converted sanity checks for example trials.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/lick/data`.

ii. ```python
lick=beh["lick/data"][()][:t_common].astype(np.float32, copy=False)
...
lick_binary = (arrays.lick[start:stop_exclusive] > 0).astype(np.int64, copy=False)
```

iii. Step 5 of the notes maps the lick output directly from the framewise lick trace, with separate handling for artifact trials and rebinned sessions.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The script thresholds each frame to a binary `lick > 0` value. If the session is at 31 Hz, it rebins pairs of frames with a logical OR, so the output is 1 whenever either frame in the merged bin had a lick.

ii. ```python
lick_binary = (arrays.lick[start:stop_exclusive] > 0).astype(np.int64, copy=False)

if factor == 2:
    lick_binary = rebin_1d_any(lick_binary, factor)

def rebin_1d_any(x: np.ndarray, factor: int) -> np.ndarray:
    out.append(np.any(x[: n_full * factor].reshape(n_full, factor) > 0, axis=1))
```

iii. The notes justify this with the task’s binary lick target and say the rebinned case should use logical OR within each common 15.5 Hz bin.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The lick trace is sliced on the same trial boundaries as the neural data and, after any factor-2 rebinning, has the same number of bins as the neural trial array.

ii. ```python
lick_binary = (arrays.lick[start:stop_exclusive] > 0).astype(np.int64, copy=False)
if factor == 2:
    lick_binary = rebin_1d_any(lick_binary, factor)

output_trial = np.stack(
    [dist_bin, pos_bin, speed_bin, lick_binary, zone_bin, reward_bin],
    axis=0,
)
```

iii. The notes say the sample plots and raw-vs-converted spot checks were used to confirm that lick timing was aligned correctly relative to trial time and other behavioral streams.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is not copied from a single raw categorical field. The script infers reward-zone location from `reward_zone/data`, `position/data`, reward timestamps, framewise trial numbers, and the hard-coded A/B/C zone centers and bounds.

ii. ```python
ZONE_BOUNDS = {
    0: (80.0, 130.0),
    1: (200.0, 250.0),
    2: (320.0, 370.0),
}

mask = reward_zone_signal[start:stop_exclusive] > 0
if np.any(mask):
    zone_pos = float(np.nanmedian(positions[start:stop_exclusive][mask]))
elif trial_id in reward_pos_by_trial:
    zone_pos = reward_pos_by_trial[trial_id]
```

iii. The trajectory says this was the main unresolved mapping issue in Step 5, and `CONVERSION_NOTES.md` says the agent decided to infer zone identity because omission trials often lacked direct zone labels in the NWB export.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The script estimates a reward-zone position per trial from `reward_zone_signal` or reward delivery position, maps it to the nearest zone center, then fills missing trials by block structure: all trials before trial 30 get the pre-switch majority zone and trials from 30 onward get the post-switch majority zone. The final zone label is repeated across all time bins in the trial.

ii. ```python
observed[trial_id] = int(np.argmin(np.abs(ZONE_CENTERS - zone_pos)))

split = min(SWITCH_TRIAL_INDEX, n_trials)
pre_majority = trial_majority_zone(observed, 0, split, fallback=int(unique_observed[0]))
post_majority = trial_majority_zone(observed, split, n_trials, fallback=post_fallback)
filled[:split][filled[:split] < 0] = pre_majority
filled[split:][filled[split:] < 0] = post_majority

zone_bin = np.full((t_bins,), zone_idx, dtype=np.int64)
```

iii. Step 5 of the notes explicitly says the agent would “infer reward-zone location from rewarded trials and fill omission trials within stable blocks,” using the paper’s switch timing as a sanity anchor.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/timestamps`, behavior timestamps, and framewise trial numbering used to assign each reward event to a trial.

ii. ```python
reward_times=beh["Reward/timestamps"][()].astype(np.float64, copy=False)

reward_frame_idx = reward_frames_for_session(arrays)
rewarded_trial_ids = set(int(t) for t in arrays.trial_number[reward_frame_idx] if t >= 0)
```

iii. The notes map reward outcome directly from reward-event timestamps and describe the result as a binary rewarded-versus-omitted per-trial label.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is labeled rewarded if any mapped reward event falls in that trial and omitted otherwise. That scalar is then repeated across every time bin of the trial.

ii. ```python
trial_rewarded = np.array(
    [1 if trial_id in rewarded_trial_ids else 0 for trial_id in range(len(trial_slices))],
    dtype=np.int64,
)
reward_bin = np.full((t_bins,), int(trial_rewarded[trial_id]), dtype=np.int64)
```

iii. Step 5 of the notes says this reproduces the rewarded-versus-omitted condition required by the decoder task, and Step 9 checks that the resulting omission fraction stays close to the paper’s reported ~15%.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles small inconsistencies with silent repairs and fallbacks: it crops neural and behavior streams to the shared minimum length, discards negative invalid environment samples, clips reward-frame indices into range, fills missing reward-zone labels by fallback logic, averages leftover frames in partial rebinning windows, skips malformed trial pairs where teleport precedes start, and errors only when a session ends up with too few valid trials or no inferable reward zone.

ii. ```python
t_common = min(t_neural, t_behavior)

env = env[env >= 0]

return np.clip(idx, 0, arrays.timestamps.shape[0] - 1).astype(np.int64, copy=False)

if stop < start:
    continue

if unique_observed.size == 1:
    filled[filled < 0] = int(unique_observed[0])

if x.shape[0] % factor:
    out.append(np.array([x[n_full * factor :].mean()], dtype=np.float32))
```

iii. Step 10 of the notes says the agent intentionally handled one-sample behavior/neural mismatches by cropping to the common minimum and treated unresolved reward-zone and trial-boundary edge cases with documented fallbacks rather than dropping whole sessions.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive parts are reading each session’s full deconvolved matrix from disk, filtering ROIs, and then looping over every trial to build neural/input/output arrays. Plot generation is also expensive when enabled, but only used for up to two sessions.

ii. ```python
neural = neural_group["data"][()].astype(np.float32, copy=False)

for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    ...
    neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)
    ...
    neural_trials.append(neural_trial.astype(np.float32, copy=False))

plot_processing_summary(...)
```

iii. Step 6 and Step 7 of the notes explicitly identify full-session NWB reads and per-session conversion as the runtime bottlenecks, with plotting called out as optional overhead for debugging mode.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several trial-level loops remain scalar Python loops: reward-zone inference across trials, per-trial environment and lick-artifact detection, and the main trial-conversion loop that slices and builds every trial individually.

ii. ```python
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    mask = reward_zone_signal[start:stop_exclusive] > 0
    ...

for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    env = arrays.environment[start:stop_exclusive]
    ...

for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    if lick_artifact[trial_id]:
        continue
    neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)
```

iii. The notes mention that direct `h5py` loading and simple rebinning were chosen speedups, but they do not claim these per-trial loops were eliminated. These are the clearest remaining vectorization opportunities.

## 13-c. What processing does the code repeat multiple times?

i. The code repeatedly slices trials and clips continuous variables, and it repeats similar per-trial logic both for actual conversion and for debug plotting. It also repeatedly fills per-bin constant arrays for environment, trial number, previous reward, reward zone, and reward outcome.

ii. ```python
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
speed = np.clip(arrays.speed[start:stop_exclusive], 0.0, None)

env_series = np.full((t_bins,), float(trial_env[trial_id]), dtype=np.float32)
trial_series = np.full((t_bins,), float(trial_id), dtype=np.float32)
prev_reward_series = np.full((t_bins,), float(trial_prev_rewarded[trial_id]), dtype=np.float32)
zone_bin = np.full((t_bins,), zone_idx, dtype=np.int64)
reward_bin = np.full((t_bins,), int(trial_rewarded[trial_id]), dtype=np.int64)

example_position=np.clip(arrays.position[trial_slices[example_idx][0] : trial_slices[example_idx][1]], 0.0, TRACK_LENGTH_CM)
```

iii. Step 6 notes that plotting adds overhead, and the script structure shows the same trial slice being revisited for conversion and optional visualization rather than being cached once.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads and stores some data that are not used downstream by the decoder: `scanning` is read but never used, `roi_ids` are retained only inside `SessionArrays`, `observed_zone_positions_cm` are stored only in metadata, and debug plot generation does not affect the final dataset. More broadly, the code constructs repeated per-bin versions of trial-level labels even for outputs that are constant within a trial.

ii. ```python
scanning=beh["scanning/data"][()][:t_common].astype(np.float32, copy=False),
roi_ids=roi_ids,

session_info = {
    ...
    "observed_zone_positions_cm": observed_zone_position.tolist(),
}

if show_processing:
    plot_processing_summary(...)
```

iii. The notes themselves call plotting overhead out as nonessential for full conversion. The rest of these extra loads and metadata fields appear to be kept only for bookkeeping or auditability rather than for the decoder’s actual training inputs.
