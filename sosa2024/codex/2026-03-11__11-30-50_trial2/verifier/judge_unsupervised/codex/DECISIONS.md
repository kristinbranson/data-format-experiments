# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads every NWB file matching `data/sub-*/sub-*_behavior+ophys.nwb`, then opens each file with `h5py` and reads the aligned behavior streams plus `processing/ophys/Deconvolved/plane0`.

ii. ```python
def list_session_files() -> list[str]:
    files = sorted(glob.glob(os.path.join("data", "sub-*", "sub-*_behavior+ophys.nwb")))

with h5py.File(path, "r") as f:
    beh = f["processing/behavior/BehavioralTimeSeries"]
    neural_group = f["processing/ophys/Deconvolved/plane0"]
```

iii. In `CONVERSION_NOTES.md`, Step 5 says to "Use all 152 NWB sessions from the 11 switch-task mice" and Step 6 says the script uses direct `h5py` loading instead of materializing full NWB objects.

## 1-b. How are the data split into subjects?

i. Subjects are split by the NWB field `general/subject/subject_id`. The script builds a unique `subjects` list and a per-session `subject_idx`.

ii. ```python
subject = decode_if_bytes(f["general/subject/subject_id"][()])

if arrays.subject not in subject_to_idx:
    subject_to_idx[arrays.subject] = len(all_subjects)
    all_subjects.append(arrays.subject)
data["subject_idx"].append(subject_to_idx[arrays.subject])
```

iii. Step 2 and Step 9 of `CONVERSION_NOTES.md` treat the dataset as 11 mice and track sessions by mouse/day, so this is the agent's explicit subject split.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The session label is built from the subject id and `general/session_id`.

ii. ```python
session_id = decode_if_bytes(f["general/session_id"][()])
session_label = f"{subject}_ses-{session_id}"

for session_idx, path in enumerate(files):
    arrays = load_session_arrays(path)
```

iii. The notes describe the archive as "one NWB file per subject-session" and the converter iterates file-by-file, so file boundaries define sessions.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from the binary `trial_start` and `teleport` signals. Each trial is the inclusive slice from a start pulse to the matching teleport pulse.

ii. ```python
starts = np.flatnonzero(trial_start_signal > 0.5)
teleports = np.flatnonzero(teleport_signal > 0.5)
for start, stop in zip(starts[:n], teleports[:n]):
    if stop < start:
        continue
    slices.append((int(start), int(stop) + 1))
```

iii. Step 5 calls this out directly: "Reconstruct trials from `trial_start` and `teleport`", to mirror the reference use of `trial_start_inds` and `teleport_inds`.

## 1-e. How are trials filtered based on quality controls?

i. The only explicit trial-level QC is dropping "lick artifact" trials. A trial is removed if more than `0.35` of its lick samples exceed `2`.

ii. ```python
LICK_ARTIFACT_FRACTION = 0.35

lick_trial = arrays.lick[start:stop_exclusive]
if lick_trial.size and np.mean(lick_trial > 2) > LICK_ARTIFACT_FRACTION:
    lick_artifact[trial_id] = True

if lick_artifact[trial_id]:
    continue
```

iii. Step 5 and Step 10 justify this as matching the paper/reference licking QC. The trajectory also notes trial loss from this rule was monitored closely because it was the main QC likely to change totals.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The agent derives `neural` from `processing/ophys/Deconvolved/plane0/data`, with ROI selection from `Deconvolved/plane0/rois` and `ImageSegmentation/PlaneSegmentation/iscell`.

ii. ```python
neural_group = f["processing/ophys/Deconvolved/plane0"]
seg = f["processing/ophys/ImageSegmentation/PlaneSegmentation"]

neural = neural_group["data"][()].astype(np.float32, copy=False)
roi_ids = neural_group["rois"][()].astype(np.int64, copy=False)
iscell = seg["iscell"][()][roi_ids, 0] > 0.5
```

iii. Step 4 and Step 5 explicitly say the agent chose to use the NWB `Deconvolved` stream as the paper-equivalent neural signal and to filter via ROI-region-aware `iscell`.

## 2-b. How is the `neural` data processed?

i. Processing is limited to ROI filtering, cropping neural/behavior to a common frame count, transposing each trial to `(neurons, time)`, and summing adjacent frames for 31 Hz sessions to reach a common 15.5 Hz rate. It does not recompute dF/F or deconvolution.

ii. ```python
t_common = min(t_neural, t_behavior)
neural = neural[:, iscell]

neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)
if factor == 2:
    neural_trial = rebin_2d_sum_time_last(neural_trial, factor)
```

iii. Step 6 says the script implements ROI-aware cell filtering and `31.015625 Hz` to `15.5078125 Hz` rebinning. Step 10 says exact dF/F-speed-correlation filtering was not re-run because the archive already exposes deconvolved responses.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered only by `iscell[:,0] > 0.5` applied to the ROI ids referenced by the deconvolved matrix. No additional speed-correlation interneuron exclusion or NaN-based sample masking is implemented.

ii. ```python
roi_ids = neural_group["rois"][()].astype(np.int64, copy=False)
iscell = seg["iscell"][()][roi_ids, 0] > 0.5
neural = neural[:, iscell]
```

iii. Step 5 planned an additional speed-correlation filter, but Step 10 says it was not applied because the exact reference dF/F stream was unavailable from the archive.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each neural trial starts at `trial_start` and ends at `teleport`, so all neural trials are aligned to start of trial as requested.

ii. ```python
trial_slices = build_trial_slices(arrays.trial_start, arrays.teleport)
neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)
```

iii. The task instructions required trial-start alignment, and Step 5 says trial parsing was chosen specifically to match the reference trial boundary convention.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a common time bin of `1 / 15.5078125` s, about `64.48 ms`. Sessions sampled at `31.015625 Hz` are rebinned by a factor of 2; native 15.5 Hz sessions are left unchanged.

ii. ```python
TARGET_RATE_HZ = 15.5078125
TARGET_DT_S = 1.0 / TARGET_RATE_HZ

if math.isclose(arrays.rate_hz, TARGET_RATE_HZ):
    factor = 1
elif math.isclose(arrays.rate_hz, 2.0 * TARGET_RATE_HZ):
    factor = 2
```

iii. Step 4 and Step 5 justify this as harmonizing the archive's two frame rates to the paper's effective per-plane sampling rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from reconstructed trial length plus the fixed target bin size, not directly from raw timestamps at each sample. Trial boundaries come from `trial_start`/`teleport`.

ii. ```python
trial_slices = build_trial_slices(arrays.trial_start, arrays.teleport)
t_bins = neural_trial.shape[1]
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
```

iii. Step 5 says this input should be "seconds since trial start for each bin", aligned to the first frame of the trial.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The script creates a monotonic per-bin vector using `np.arange(t_bins) * TARGET_DT_S` after any neural rebinning.

ii. ```python
t_bins = neural_trial.shape[1]
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
```

iii. The notes describe this as reconstructed elapsed time rather than copied timestamps.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is generated with exactly `neural_trial.shape[1]` bins for each trial, so it is one-to-one with the final neural time axis.

ii. ```python
t_bins = neural_trial.shape[1]
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
input_trial = np.stack([time_from_start, env_series, trial_series, prev_reward_series], axis=0)
```

iii. The agent's Step 10 sanity checks explicitly compared hand-reconstructed inputs against converted trials and reported `np.allclose(...) == True`.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/environment/data`.

ii. ```python
environment=beh["environment/data"][()][:t_common].astype(np.float32, copy=False)
```

iii. Step 5 maps `environment/data` to the binary ENV1/ENV2 decoder input.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial, negative values are discarded, the modal remaining value is taken as the trial environment, and that scalar is repeated across all bins in the trial.

ii. ```python
env = arrays.environment[start:stop_exclusive]
env = env[env >= 0]
trial_env[trial_id] = mode_int(env, default=0)
env_series = np.full((t_bins,), float(trial_env[trial_id]), dtype=np.float32)
```

iii. Step 5 says the agent planned to take the modal valid environment value per trial and repeat it across bins.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. In code, it is derived from the enumerated converted trial index `trial_id`, not from the stored `trial number` stream. The raw `trial number` array is only used for reward assignment and zone inference.

ii. ```python
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    ...
    trial_series = np.full((t_bins,), float(trial_id), dtype=np.float32)
```

iii. The notes planned to use within-session trial index and keep it 0-indexed, matching the reference code convention.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The script uses the loop index from the reconstructed trial list and repeats that scalar over all bins in the trial.

ii. ```python
trial_series = np.full((t_bins,), float(trial_id), dtype=np.float32)
```

iii. Step 5 explicitly says "Keep within-session trial number 0-indexed."

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from reward event timestamps (`Reward/timestamps`) mapped onto frame indices and then onto trial ids using the raw `trial number` stream.

ii. ```python
reward_times=beh["Reward/timestamps"][()].astype(np.float64, copy=False)
reward_frame_idx = reward_frames_for_session(arrays)
rewarded_trial_ids = set(int(t) for t in arrays.trial_number[reward_frame_idx] if t >= 0)
```

iii. Step 5 describes this input as a shifted version of trial reward outcome, with first trial set to 0.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. First the current-trial reward outcome is computed as binary rewarded/omitted, then it is shifted by one trial. The first trial is hard-coded to 0.

ii. ```python
trial_rewarded = np.array(
    [1 if trial_id in rewarded_trial_ids else 0 for trial_id in range(len(trial_slices))],
    dtype=np.int64,
)
trial_prev_rewarded = np.concatenate([[0], trial_rewarded[:-1]]).astype(np.float32, copy=False)
```

iii. The notes say this variable is user-requested rather than taken directly from the paper decoder, so the agent derived it from reward outcome.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from trial-wise position (`position/data`) and the inferred active reward zone for that trial, which itself comes from `reward_zone/data`, reward timestamps, trial numbers, and environment structure.

ii. ```python
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
trial_zone, observed_zone_position = infer_trial_zones(...)
dist = compute_distance_to_zone(position, zone_idx)
```

iii. Step 5 maps this output to position plus inferred reward-zone bounds, following the task's reward-relative geometry.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position is clipped to the 450 cm track, then converted to signed distance from the active zone: negative before zone start, zero inside the zone, positive after the zone end.

ii. ```python
def compute_distance_to_zone(position_cm: np.ndarray, zone_idx: int) -> np.ndarray:
    zone_start, zone_end = ZONE_BOUNDS[zone_idx]
    distance = np.zeros(position_cm.shape, dtype=np.float32)
    before = position_cm < zone_start
    after = position_cm > zone_end
    distance[before] = position_cm[before] - zone_start
    distance[after] = position_cm[after] - zone_end
```

iii. Step 5 states this was chosen to satisfy the decoder spec: signed distance to the nearest point in the active reward zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is discretized into 7 bins using the exact thresholds from the task instructions.

ii. ```python
out[distance_cm < -50.0] = 0
out[(distance_cm >= -50.0) & (distance_cm < -10.0)] = 1
out[(distance_cm >= -10.0) & (distance_cm < 0.0)] = 2
out[distance_cm == 0.0] = 3
out[(distance_cm > 0.0) & (distance_cm <= 10.0)] = 4
out[(distance_cm > 10.0) & (distance_cm <= 50.0)] = 5
out[distance_cm > 50.0] = 6
```

iii. The code follows the user-specified bin edges exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from the same per-trial position vector used for that trial's neural slice, after the same optional factor-2 rebinning.

ii. ```python
if factor == 2:
    position = rebin_1d_mean(position, factor)
...
dist = compute_distance_to_zone(position, zone_idx)
dist_bin = discretize_distance(dist)
```

iii. Step 7 and Step 10 say sample and full sanity checks found the output aligned correctly with neural trials.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/position/data`.

ii. ```python
position=beh["position/data"][()][:t_common].astype(np.float32, copy=False)
```

iii. Step 5 maps `position/data` directly to this output.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to the 450 cm corridor and then discretized into five equal-width bins.

ii. ```python
clipped = np.clip(position_cm, 0.0, TRACK_LENGTH_CM - 1e-6)
return np.minimum((clipped / (TRACK_LENGTH_CM / 5.0)).astype(np.int64), 4)
```

iii. The notes acknowledge this is a task-specific change from the paper's finer 45-bin spatial analyses.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The thresholding is uniform: 5 bins spanning the 450 cm track, so about 90 cm per bin.

ii. ```python
return np.minimum((clipped / (TRACK_LENGTH_CM / 5.0)).astype(np.int64), 4)
```

iii. Step 5 states this is driven by the decoder spec rather than the paper's original 10 cm bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is derived from the same trial slice as the neural data and rebinned with the same factor when the session rate is 31 Hz.

ii. ```python
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
if factor == 2:
    position = rebin_1d_mean(position, factor)
pos_bin = discretize_position(position)
```

iii. The agent's Step 10 raw-vs-converted spot check included absolute-position bins and reported an exact match.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/lick/data`.

ii. ```python
lick=beh["lick/data"][()][:t_common].astype(np.float32, copy=False)
```

iii. Step 5 maps `lick/data` directly to the lick output after QC.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The lick channel is thresholded to binary with `lick > 0`. For 31 Hz sessions it is rebinned with logical OR across each pair of frames. Trials failing the lick-artifact QC are dropped before output creation.

ii. ```python
lick_binary = (arrays.lick[start:stop_exclusive] > 0).astype(np.int64, copy=False)
if factor == 2:
    lick_binary = rebin_1d_any(lick_binary, factor)
```

iii. Step 5 says to convert licks to binary per bin and use logical OR when rebinned.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The lick output uses the same reconstructed trial boundaries as neural and the same optional rebin factor, so it has the same number of time bins as each neural trial.

ii. ```python
lick_binary = (arrays.lick[start:stop_exclusive] > 0).astype(np.int64, copy=False)
if factor == 2:
    lick_binary = rebin_1d_any(lick_binary, factor)
output_trial = np.stack([dist_bin, pos_bin, speed_bin, lick_binary, zone_bin, reward_bin], axis=0)
```

iii. The notes report output sanity checks on lick bins against raw NWB trials.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is inferred from `reward_zone/data`, `position/data`, `Reward/timestamps`, `trial number/data`, and trial-wise environment structure.

ii. ```python
trial_zone, observed_zone_position = infer_trial_zones(
    positions=np.clip(arrays.position, 0.0, TRACK_LENGTH_CM),
    reward_zone_signal=arrays.reward_zone_signal,
    reward_frame_idx=reward_frame_idx,
    trial_by_frame=arrays.trial_number.astype(np.int64, copy=False),
    trial_env=trial_env,
)
```

iii. Step 5 says the agent planned to infer reward-zone labels from rewarded trials and fill omission trials within stable blocks because missing labels in the NWB export were concentrated on omission trials.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For each trial, the script estimates reward-zone position from active `reward_zone` samples or reward-event positions, maps that position to the nearest zone center A/B/C, and fills missing trials by majority vote before and after the fixed switch trial `30`. It then repeats the inferred zone label across all bins of the trial.

ii. ```python
if np.any(mask):
    zone_pos = float(np.nanmedian(positions[start:stop_exclusive][mask]))
elif trial_id in reward_pos_by_trial:
    zone_pos = reward_pos_by_trial[trial_id]
...
split = min(SWITCH_TRIAL_INDEX, n_trials)
filled[:split][filled[:split] < 0] = pre_majority
filled[split:][filled[split:] < 0] = post_majority
zone_bin = np.full((t_bins,), zone_idx, dtype=np.int64)
```

iii. This is one of the agent's explicit key decisions in Step 5 and is defended there as a structure-constrained imputation for omission trials.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/timestamps`, mapped to frame indices using behavior timestamps, then mapped to trial ids using the raw `trial number` stream.

ii. ```python
idx = np.searchsorted(arrays.timestamps, arrays.reward_times, side="left")
rewarded_trial_ids = set(int(t) for t in arrays.trial_number[reward_frame_idx] if t >= 0)
```

iii. Step 5 maps this variable to reward events assigned to trials, with rewarded if any reward event falls within that trial.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The script marks each reconstructed trial as rewarded or omitted, then repeats that binary label across every time bin in the trial.

ii. ```python
trial_rewarded = np.array(
    [1 if trial_id in rewarded_trial_ids else 0 for trial_id in range(len(trial_slices))],
    dtype=np.int64,
)
reward_bin = np.full((t_bins,), int(trial_rewarded[trial_id]), dtype=np.int64)
```

iii. The notes say per-trial outputs are repeated across bins to keep a consistent `(n_output, n_timepoints)` structure.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The script crops neural and behavior streams to a common minimum length, ignores negative environment samples, skips malformed start/stop pairs, raises errors when no valid trials or reward-zone labels can be inferred, and fills missing reward-zone labels using trial-block heuristics. It does not attempt more elaborate interpolation.

ii. ```python
t_common = min(t_neural, t_behavior)
env = env[env >= 0]
if stop < start:
    continue
if unique_observed.size == 0:
    raise RuntimeError("Could not infer any reward-zone labels from the session.")
filled[:split][filled[:split] < 0] = pre_majority
```

iii. Step 4 and Step 10 document the 1-sample mismatch cropping and the reward-zone filling strategy as the main archive edge-case handling decisions.

## 13-a. What are the most time-consuming steps of the code?

i. The agent identified full-session NWB reads and per-session conversion as the main cost, with optional plotting adding substantial overhead during debug/sample runs.

ii. ```python
with h5py.File(path, "r") as f:
    ...
if show_processing:
    plot_processing_summary(...)
```

iii. Step 6 and Step 7 explicitly say full-session reads dominate core conversion time and plotting should remain off for full conversion.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loops for environment/lick QC, reward-zone inference, and trial construction could all have been more heavily vectorized. As written, the script loops over trials multiple times.

ii. ```python
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    ...

for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    if lick_artifact[trial_id]:
        continue
    ...
```

iii. Step 6 notes that only "simple factor-2 rebinning" and direct `h5py` loading were used as speedups; the rest of the code remains trial-loop oriented.

## 13-c. What processing does the code repeat multiple times?

i. It repeatedly slices trials from the same arrays, separately loops once for trial metadata/QC and again for data construction, and recomputes some rebinned example arrays for plotting even after conversion is complete.

ii. ```python
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    ...
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    ...
if show_processing:
    example_position = ... if factor == 1 else rebin_1d_mean(...)
```

iii. The implementation notes mention plotting overhead and repeated downstream work as known inefficiencies.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads `scanning` but never uses it, retains `roi_ids` only for immediate filtering, computes `observed_zone_position` mainly for metadata/debugging, and optionally generates processing plots that are not used by the downstream decoder.

ii. ```python
scanning=beh["scanning/data"][()][:t_common].astype(np.float32, copy=False),
roi_ids=roi_ids,
observed_zone_positions_cm": observed_zone_position.tolist(),
if show_processing:
    plot_processing_summary(...)
```

iii. Step 6 and Step 9 treat plots and some session-info fields as validation aids rather than decoder-required outputs.
