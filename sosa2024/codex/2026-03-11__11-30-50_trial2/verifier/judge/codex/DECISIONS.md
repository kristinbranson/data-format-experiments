# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all `.nwb` files matching `data/sub-*/sub-*_behavior+ophys.nwb`, then opens each file directly with `h5py`. It processes one file per session and builds the session/trial outputs from arrays loaded out of each NWB file.

ii.
```python
def list_session_files() -> list[str]:
    files = sorted(glob.glob(os.path.join("data", "sub-*", "sub-*_behavior+ophys.nwb")))
    if not files:
        raise FileNotFoundError("No NWB files found under data/sub-*/")
    return files

def load_session_arrays(path: str) -> SessionArrays:
    with h5py.File(path, "r") as f:
        ...
```

iii. In `CONVERSION_NOTES.md`, the AI says the archive contains one NWB file per subject-session and that the shared NWB export should be treated as the source of truth. The trajectory also shows it deliberately switched to “direct NWB introspection” after finding the layout differed from its initial expectation.

## 1-b. How are the data split into subjects?

i. Subjects are split by the `sub-*` directory / filename pattern and by the subject id stored in each NWB file. During conversion, sessions are assigned a subject index the first time each subject string is seen.

ii.
```python
files = sorted(glob.glob(os.path.join("data", "sub-*", "sub-*_behavior+ophys.nwb")))

subject = decode_if_bytes(f["general/subject/subject_id"][()])

if arrays.subject not in subject_to_idx:
    subject_to_idx[arrays.subject] = len(all_subjects)
    all_subjects.append(arrays.subject)
```

iii. The notes say the dataset inventory contains 11 subject directories named `sub-m*`, matching the paper’s switch-task cohort. The AI therefore treated subject identity from the NWB metadata and the directory structure as consistent.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The session label is taken from `general/session_id` inside the file.

ii.
```python
session_id = decode_if_bytes(f["general/session_id"][()])
session_label = f"{subject}_ses-{session_id}"

for session_idx, path in enumerate(files):
    arrays = load_session_arrays(path)
```

iii. In the notes, the AI states the archive has one NWB file per subject-session and lists 152 files total. It therefore uses file-level splitting as session-level splitting.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from framewise `trial_start` and `teleport` signals. The AI takes every frame where `trial_start > 0.5` as a start, every frame where `teleport > 0.5` as an end marker, then zips those two sequences and uses `[start, stop + 1)` for each trial.

ii.
```python
def build_trial_slices(trial_start_signal: np.ndarray, teleport_signal: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(trial_start_signal > 0.5)
    teleports = np.flatnonzero(teleport_signal > 0.5)
    ...
    for start, stop in zip(starts[:n], teleports[:n]):
        if stop < start:
            continue
        slices.append((int(start), int(stop) + 1))
```

iii. `CONVERSION_NOTES.md` says the AI chose to reconstruct trials from `trial_start` and `teleport` because that “matches the reference trial definition more closely” than relying on `trial number`. The trajectory explicitly says it wanted this mapping to be “explicit, not guessed.”

## 1-e. How are trials filtered based on quality controls?

i. The AI removes trials flagged as lick-sensor artifacts. A trial is dropped when more than 35% of its frames have `lick > 2`. It does not implement the reference solution’s short-trial filter.

ii.
```python
LICK_ARTIFACT_FRACTION = 0.35

lick_artifact = np.zeros((len(trial_slices),), dtype=bool)
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    lick_trial = arrays.lick[start:stop_exclusive]
    if lick_trial.size and np.mean(lick_trial > 2) > LICK_ARTIFACT_FRACTION:
        lick_artifact[trial_id] = True

for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    if lick_artifact[trial_id]:
        continue
```

iii. The notes cite the paper’s lick-QC rule and say the AI planned to use the reported `81 / 12,376` removed-trial statistic as a sanity anchor. The trajectory also says it specifically wanted to check whether the lick-artifact rule “overshoot[s] the paper’s QC rate.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` output is derived from `processing/ophys/Deconvolved/plane0/data`, with ROI ids from `processing/ophys/Deconvolved/plane0/rois` and cell curation from `processing/ophys/ImageSegmentation/PlaneSegmentation/iscell`.

ii.
```python
neural_group = f["processing/ophys/Deconvolved/plane0"]
seg = f["processing/ophys/ImageSegmentation/PlaneSegmentation"]

neural = neural_group["data"][()].astype(np.float32, copy=False)
roi_ids = neural_group["rois"][()].astype(np.int64, copy=False)
iscell = seg["iscell"][()][roi_ids, 0] > 0.5
neural = neural[:, iscell]
```

iii. The notes say the paper decoder uses deconvolved activity and that the shared NWB archive exposes `Deconvolved`, `Fluorescence`, and `Neuropil`, but not the exact reference `dF/F` stream. The chosen decision was to use the NWB deconvolved matrix as the “paper-equivalent neural signal.”

## 2-b. How is the `neural` data processed?

i. The AI keeps curated deconvolved activity, crops it to the common neural/behavior length, transposes each trial to `(neurons, time)`, and rebins 31.015625 Hz sessions to a common 15.5078125 Hz timebase by summing adjacent neural frames.

ii.
```python
t_common = min(t_neural, t_behavior)
...
neural=neural[:t_common],

neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)
...
if factor == 2:
    neural_trial = rebin_2d_sum_time_last(neural_trial, factor)

def rebin_2d_sum_time_last(x: np.ndarray, factor: int) -> np.ndarray:
    ...
    full = x[:, : n_full * factor].reshape(x.shape[0], n_full, factor).sum(axis=2)
```

iii. The notes say the archive contains both 15.5078125 Hz and 31.015625 Hz sessions, and the AI decided to “standardize all sessions to a common 15.5078125 Hz bin size.” It also documented that the multipane files expose only `plane0`, and chose to work with the provided response matrix rather than reconstruct the full paper pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered with the Suite2p-style `iscell` mask, using only the ROI indices referenced by the deconvolved response matrix.

ii.
```python
roi_ids = neural_group["rois"][()].astype(np.int64, copy=False)
iscell = seg["iscell"][()][roi_ids, 0] > 0.5
neural = neural[:, iscell]
roi_ids = roi_ids[iscell]
```

iii. The notes say this choice avoids overcounting ROIs in multipane segmentation tables and matches the paper’s reliance on manual Suite2p curation. The AI explicitly documented the ROI-region mismatch as a “nontrivial data-format issue.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start by slicing each trial from the reconstructed `trial_start` index through the corresponding `teleport` index, then treating the first bin of each slice as time zero.

ii.
```python
trial_slices = build_trial_slices(arrays.trial_start, arrays.teleport)
...
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)
```

iii. The notes say “trial alignment should use `trial_start_inds` as the start and `teleport_inds` as the end of each trial.” The metadata it writes also sets `temporal_alignment_event` to `trial_start`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a common 15.5078125 Hz timebase, i.e. `1 / 15.5078125` s per bin (about 64.48 ms). Sessions already at that rate are left unchanged; sessions at 31.015625 Hz are rebinned by a factor of 2.

ii.
```python
TARGET_RATE_HZ = 15.5078125
TARGET_DT_S = 1.0 / TARGET_RATE_HZ

if math.isclose(arrays.rate_hz, TARGET_RATE_HZ):
    factor = 1
elif math.isclose(arrays.rate_hz, 2.0 * TARGET_RATE_HZ):
    factor = 2

"time_bin_size": 1000.0 * TARGET_DT_S,
```

iii. `CONVERSION_NOTES.md` says the archive contains both 15.5 Hz and 31 Hz recordings and that a common output bin size was chosen so all sessions share one decoder timebase. The notes frame this as matching the paper’s effective per-plane sampling rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is not derived from a raw time series after trial slicing. Instead, the AI derives it from the number of bins in the extracted neural trial together with the fixed target sampling interval `TARGET_DT_S`.

ii.
```python
TARGET_DT_S = 1.0 / TARGET_RATE_HZ
...
t_bins = neural_trial.shape[1]
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
```

iii. The notes justify a common 15.5078125 Hz output grid across sessions. Once that decision was made, the AI treated elapsed time as an evenly spaced synthetic series rather than using per-frame behavior timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The AI creates a regularly spaced vector starting at 0 and increasing by one fixed bin duration for each timepoint in the trial.

ii.
```python
t_bins = neural_trial.shape[1]
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
```

iii. The notes emphasize a standardized common-bin representation, especially after rebinning 31 Hz sessions. This makes trial time depend only on bin count, not on the stored behavioral timestamps.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by construction: the elapsed-time vector has exactly one entry per neural bin in the trial, starting at the first bin of the neural slice after any optional rebinning.

ii.
```python
neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)
...
if factor == 2:
    neural_trial = rebin_2d_sum_time_last(neural_trial, factor)
...
t_bins = neural_trial.shape[1]
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
```

iii. The notes repeatedly describe the output as a “trial-aligned” dataset on a common timebase. The AI’s sanity checks in the notes also focused on trial-wise raw-vs-converted alignment after rebinning.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavior time series `processing/behavior/BehavioralTimeSeries/environment/data`.

ii.
```python
environment=beh["environment/data"][()][:t_common].astype(np.float32, copy=False),
...
env = arrays.environment[start:stop_exclusive]
env = env[env >= 0]
trial_env[trial_id] = mode_int(env, default=0)
```

iii. The notes say the environment channel takes values `-1, 0, 1`, with `-1` outside valid periods, and they map `0 -> ENV1`, `1 -> ENV2`.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the AI removes invalid negative samples, takes the modal valid environment value, and repeats that single value across all bins in the trial.

ii.
```python
env = arrays.environment[start:stop_exclusive]
env = env[env >= 0]
trial_env[trial_id] = mode_int(env, default=0)
...
env_series = np.full((t_bins,), float(trial_env[trial_id]), dtype=np.float32)
```

iii. The notes describe environment as a per-trial variable and say the modal valid value should be repeated across bins. This was justified as a robust way to ignore `-1` outside valid trial/imaging periods.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the order of the reconstructed trial slices, i.e. the loop index `trial_id`, not from the stored framewise `trial number` series.

ii.
```python
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    ...
    trial_series = np.full((t_bins,), float(trial_id), dtype=np.float32)
```

iii. The notes say the AI chose within-session 0-indexed trial ids and preferred reconstructing trials from `trial_start` / `teleport` rather than depending on the raw `trial number` stream for boundaries.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No additional transformation is applied beyond assigning the per-session sequential trial index and repeating it across all bins of that trial.

ii.
```python
trial_series = np.full((t_bins,), float(trial_id), dtype=np.float32)
```

iii. The notes explicitly say “Keep within-session trial number 0-indexed” and describe it as a repeated per-trial context variable.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from reward timestamps in `Reward/timestamps`, mapped onto trial ids using the framewise `trial number` array at the nearest reward frame.

ii.
```python
reward_times=beh["Reward/timestamps"][()].astype(np.float64, copy=False),
trial_number=beh["trial number/data"][()][:t_common].astype(np.float32, copy=False),

reward_frame_idx = reward_frames_for_session(arrays)
rewarded_trial_ids = set(int(t) for t in arrays.trial_number[reward_frame_idx] if t >= 0)
trial_rewarded = np.array(
    [1 if trial_id in rewarded_trial_ids else 0 for trial_id in range(len(trial_slices))]
)
trial_prev_rewarded = np.concatenate([[0], trial_rewarded[:-1]])
```

iii. The notes say previous reward outcome should come from trial reward outcome shifted by one trial, with the first trial set to 0. They also cite the omission-rate sanity check from the paper as support for this parsing.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI first infers each trial’s reward outcome, then shifts that vector by one trial. Trial 0 gets 0, and every later trial gets the previous trial’s rewarded/omitted label repeated across bins.

ii.
```python
trial_rewarded = np.array(
    [1 if trial_id in rewarded_trial_ids else 0 for trial_id in range(len(trial_slices))],
    dtype=np.int64,
)
trial_prev_rewarded = np.concatenate([[0], trial_rewarded[:-1]]).astype(np.float32, copy=False)
...
prev_reward_series = np.full((t_bins,), float(trial_prev_rewarded[trial_id]), dtype=np.float32)
```

iii. The notes say this variable is task-specified rather than taken directly from the paper’s decoder inputs, and should be represented as a repeated per-trial binary context variable.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from position plus an inferred per-trial reward-zone identity. The reward-zone identity is inferred from `reward_zone` occupancy frames when present, or from reward-event positions otherwise, with missing trials filled using block structure around trial 30.

ii.
```python
position=beh["position/data"][()][:t_common].astype(np.float32, copy=False),
reward_zone_signal=beh["reward_zone/data"][()][:t_common].astype(np.float32, copy=False),
reward_times=beh["Reward/timestamps"][()].astype(np.float64, copy=False),
trial_number=beh["trial number/data"][()][:t_common].astype(np.float32, copy=False),

mask = reward_zone_signal[start:stop_exclusive] > 0
if np.any(mask):
    zone_pos = float(np.nanmedian(positions[start:stop_exclusive][mask]))
elif trial_id in reward_pos_by_trial:
    zone_pos = reward_pos_by_trial[trial_id]
...
dist = compute_distance_to_zone(position, zone_idx)
```

iii. The notes say “the key inferential step is reward-zone location,” because zone occupancy is often missing on omission trials. The trajectory says the AI intentionally chose to “infer it from rewarded trials and fill within stable session blocks.”

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Once a trial’s active reward-zone label is inferred, the AI computes signed distance to the nearest edge of that zone: negative before the zone, 0 inside it, positive after it.

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

iii. The notes cite the paper’s A/B/C reward-zone bounds and describe signed reward-relative distance as the intended geometry for the derived output.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The signed distance is thresholded into seven categories using the task-specified cutoffs: `<-50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, `>50`.

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
```

iii. The notes say the output binning follows the decoder task specification directly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The AI derives distance from the same per-trial time slices used for neural data, then optionally rebins position alongside neural activity for 31 Hz sessions before computing the distance and bin labels.

ii.
```python
neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
...
if factor == 2:
    neural_trial = rebin_2d_sum_time_last(neural_trial, factor)
    position = rebin_1d_mean(position, factor)
...
dist = compute_distance_to_zone(position, zone_idx)
```

iii. The notes frame this as trial-aligned processing on a common timebase, with raw-vs-converted spot checks after rebinning.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/position/data`.

ii.
```python
position=beh["position/data"][()][:t_common].astype(np.float32, copy=False),
...
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
```

iii. The notes say the track is 450 cm long and treat this channel as the animal’s absolute location along the corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI clips position to the corridor range `[0, 450]`, optionally averages adjacent samples for 31 Hz sessions, then discretizes the resulting value.

ii.
```python
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
...
if factor == 2:
    position = rebin_1d_mean(position, factor)
...
pos_bin = discretize_position(position)
```

iii. The notes justify a common output timebase and emphasize a literal 450 cm corridor. That leads to clipping and equal-width corridor bins rather than using raw out-of-corridor values.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The AI uses 5 equal-width bins over the 450 cm corridor, i.e. 90 cm per bin, after clipping position into `[0, 450)`.

ii.
```python
def discretize_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, 0.0, TRACK_LENGTH_CM - 1e-6)
    return np.minimum((clipped / (TRACK_LENGTH_CM / 5.0)).astype(np.int64), 4)
```

iii. In the notes, the AI repeatedly cites the paper’s “450 cm linear track” and the instruction’s “5 equal-sized bins,” which is the apparent basis for this implementation.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position uses the same trial slices as neural data, and for 31 Hz sessions it is rebinned in lockstep with the neural trials before categorization.

ii.
```python
neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
...
if factor == 2:
    neural_trial = rebin_2d_sum_time_last(neural_trial, factor)
    position = rebin_1d_mean(position, factor)
```

iii. The notes describe the dataset as a common-bin, trial-aligned representation and mention explicit raw-vs-converted position spot checks.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/lick/data`.

ii.
```python
lick=beh["lick/data"][()][:t_common].astype(np.float32, copy=False),
...
lick_binary = (arrays.lick[start:stop_exclusive] > 0).astype(np.int64, copy=False)
```

iii. The notes describe lick as a binary-like framewise behavior channel and cite the paper’s lick-artifact handling as the main QC issue around it.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI binarizes lick as `lick > 0`, and for 31 Hz sessions it rebins by logical OR across adjacent samples. Separate from the output coding itself, it may drop whole trials that exceed the lick-artifact threshold.

ii.
```python
lick_binary = (arrays.lick[start:stop_exclusive] > 0).astype(np.int64, copy=False)
...
if factor == 2:
    lick_binary = rebin_1d_any(lick_binary, factor)

def rebin_1d_any(x: np.ndarray, factor: int) -> np.ndarray:
    ...
    out.append(np.any(x[: n_full * factor].reshape(n_full, factor) > 0, axis=1))
```

iii. The notes say licking should be binary in the converted data and that logical OR is the right rebinning rule for event-like outputs.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is sliced with the same trial boundaries as neural data and rebinned to the same number of bins whenever neural activity is rebinned.

ii.
```python
neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)
lick_binary = (arrays.lick[start:stop_exclusive] > 0).astype(np.int64, copy=False)
...
if factor == 2:
    neural_trial = rebin_2d_sum_time_last(neural_trial, factor)
    lick_binary = rebin_1d_any(lick_binary, factor)
```

iii. The notes describe this as part of the common-bin alignment strategy and mention manual spot checks of lick timing against converted trials.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is inferred from `reward_zone` occupancy frames, position, reward timestamps, trial ids, and session block structure.

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

iii. The notes explicitly call reward-zone inference the main nontrivial mapping decision, because the raw NWB export does not reliably provide a clean per-trial zone label on omission trials.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI infers a zone per trial from median reward-zone-active position when available, otherwise from reward-event position, maps that position to the nearest zone center, then fills missing trials by assuming stable blocks and a switch near trial 30.

ii.
```python
if np.any(mask):
    zone_pos = float(np.nanmedian(positions[start:stop_exclusive][mask]))
elif trial_id in reward_pos_by_trial:
    zone_pos = reward_pos_by_trial[trial_id]
...
observed[trial_id] = int(np.argmin(np.abs(ZONE_CENTERS - zone_pos)))
...
split = min(SWITCH_TRIAL_INDEX, n_trials)
pre_majority = trial_majority_zone(observed, 0, split, fallback=int(unique_observed[0]))
post_majority = trial_majority_zone(observed, split, n_trials, fallback=post_fallback)
filled[:split][filled[:split] < 0] = pre_majority
filled[split:][filled[split:] < 0] = post_majority
```

iii. The trajectory says the AI chose to “fill within stable session blocks rather than treating missing as a separate zone.” The notes tie this to the paper’s switch-after-30-trials structure.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from `Reward/timestamps`, mapped to frames and then to trials using the framewise `trial number` signal.

ii.
```python
reward_times=beh["Reward/timestamps"][()].astype(np.float64, copy=False),
...
reward_frame_idx = reward_frames_for_session(arrays)
rewarded_trial_ids = set(int(t) for t in arrays.trial_number[reward_frame_idx] if t >= 0)
```

iii. The notes say reward outcome should be inferred from reward events, and they use the overall rewarded/omission fraction as a sanity check against the paper.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are assigned to the nearest frame with `searchsorted`, then a trial is marked rewarded if any mapped reward event lands in that trial. The resulting binary label is repeated across all bins of the trial.

ii.
```python
def reward_frames_for_session(arrays: SessionArrays) -> np.ndarray:
    if arrays.reward_times.size == 0:
        return np.empty((0,), dtype=np.int64)
    idx = np.searchsorted(arrays.timestamps, arrays.reward_times, side="left")
    return np.clip(idx, 0, arrays.timestamps.shape[0] - 1).astype(np.int64, copy=False)

trial_rewarded = np.array(
    [1 if trial_id in rewarded_trial_ids else 0 for trial_id in range(len(trial_slices))],
    dtype=np.int64,
)
reward_bin = np.full((t_bins,), int(trial_rewarded[trial_id]), dtype=np.int64)
```

iii. The notes describe reward outcome as a per-trial categorical target and justify it with the paper’s omission schedule.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several issues defensively: it crops neural and behavioral streams to a common length, ignores invalid negative environment samples, fills missing reward-zone labels using rewarded trials and block structure, clips positions/speeds into plausible ranges, skips lick-artifact trials, and raises errors for sessions with unsupported sampling rates or too few remaining trials.

ii.
```python
t_common = min(t_neural, t_behavior)
...
env = env[env >= 0]
...
if unique_observed.size == 0:
    raise RuntimeError("Could not infer any reward-zone labels from the session.")
...
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
speed = np.clip(arrays.speed[start:stop_exclusive], 0.0, None)
...
if lick_artifact[trial_id]:
    continue
...
else:
    raise RuntimeError(f"Unexpected sampling rate {arrays.rate_hz} in {arrays.session_label}")
```

iii. The notes explicitly document 10 sessions with a 1-sample neural/behavior mismatch and say they were cropped to the common minimum. They also describe missing reward-zone labels as concentrated on omission trials and justify filling them from stable task blocks.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming work is reading every NWB file, loading the large neural/behavior arrays from each file, converting every session trial-by-trial, and optionally generating plots. In sample mode, `infer_sample_files` also scans every file once before conversion.

ii.
```python
for path in files:
    with h5py.File(path, "r") as f:
        ...

for session_idx, path in enumerate(files):
    arrays = load_session_arrays(path)
    ...
    neural_trials, input_trials, output_trials, brain_region_idx, session_info = convert_session(...)
```

iii. The notes explicitly say the main heavy steps are loading each NWB file and the full conversion pass, and later record per-session conversion times in metadata.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The code repeatedly loops over trials to compute per-trial environment, lick QC, reward-zone inference, and final neural/input/output construction. Some of that could be vectorized, though the variable trial lengths make it awkward.

ii.
```python
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    ...
    trial_env[trial_id] = mode_int(env, default=0)
    ...

for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    if lick_artifact[trial_id]:
        continue
    ...
    input_trial = np.stack(...)
    output_trial = np.stack(...)
```

iii. The notes mention “vectorized trial construction” as an optimization that was added, which implies the remaining trial loops were still recognized as the main unavoidable sequential structure.

## 13-c. What processing does the code repeat multiple times?

i. The code re-opens NWB files multiple times in sample mode and again during the full conversion pass. It also repeatedly slices and rebins trial-level arrays separately for neural, position, speed, and lick. Reward-zone inference and trial metadata are computed before the final conversion loop and then reused inside that loop.

ii.
```python
def infer_sample_files(files: list[str]) -> list[str]:
    for path in files:
        with h5py.File(path, "r") as f:
            ...

for session_idx, path in enumerate(files):
    arrays = load_session_arrays(path)
    ...
    for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
        ...
        if factor == 2:
            neural_trial = rebin_2d_sum_time_last(neural_trial, factor)
            position = rebin_1d_mean(position, factor)
            speed = rebin_1d_mean(speed, factor)
            lick_binary = rebin_1d_any(lick_binary, factor)
```

iii. The notes describe repeated file reads as the main unavoidable overhead and mention separate raw-vs-converted verification passes after conversion.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores some diagnostics that are not required for the converted dataset itself: sample-mode file inference metadata, plotting artifacts, timing metadata, and `observed_zone_positions_cm` in session metadata. It also loads fields such as `scanning` and `roi_ids` that are not used in the final decoder arrays.

ii.
```python
metadata.append((path, rate, env_valid))
...
scanning=beh["scanning/data"][()][:t_common].astype(np.float32, copy=False),
roi_ids=roi_ids,
...
"observed_zone_positions_cm": observed_zone_position.tolist(),
"conversion_seconds": time.perf_counter() - t0,
...
if show_processing:
    plot_processing_summary(...)
```

iii. The notes emphasize extensive sanity checks and diagnostic plots. Those decisions explain why the script carries extra metadata and plotting work beyond the minimal conversion path.
