# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. It glob-sorts every matching NWB file under `data/sub-*` (152 in the full run), opens each directly with `h5py`, and materializes the selected neural and behavioral arrays. Full mode is the default; sample mode chooses two files.

ii.
```python
files = sorted(glob.glob(os.path.join("data", "sub-*", "sub-*_behavior+ophys.nwb")))
...
with h5py.File(path, "r") as f:
    beh = f["processing/behavior/BehavioralTimeSeries"]
```

iii. The notes justify direct `h5py` as faster than constructing the full NWB object graph and state that all 152 archive sessions should be used.

## 1-b. How are the data split into subjects?

i. Subject IDs are read from each NWB file. A first-seen lookup builds `subjects`, and each session receives the corresponding `subject_idx`.

ii.
```python
subject = decode_if_bytes(f["general/subject/subject_id"][()])
...
if arrays.subject not in subject_to_idx:
    subject_to_idx[arrays.subject] = len(all_subjects)
    all_subjects.append(arrays.subject)
```

iii. The notes identify the archive as 11 switch-task mice and treat the NWB subject metadata as authoritative.

## 1-c. How are the data split into sessions?

i. Each discovered NWB file becomes one session; its session label combines the embedded subject and session ID.

ii.
```python
session_id = decode_if_bytes(f["general/session_id"][()])
session_label = f"{subject}_ses-{session_id}"
...
data["neural"].append(neural_trials)
```

iii. The agent found 152 files and reconciled this with 11 mice, 14 days, and two missing early m11 sessions.

## 1-d. How are the data split into trials?

i. Starts are every positive `trial_start` sample and ends are every positive `teleport` sample. Starts and teleports are truncated to the shorter count, paired positionally, and the teleport sample is included (`stop + 1`). Invalid reversed pairs are skipped.

ii.
```python
starts = np.flatnonzero(trial_start_signal > 0.5)
teleports = np.flatnonzero(teleport_signal > 0.5)
n = min(starts.size, teleports.size)
for start, stop in zip(starts[:n], teleports[:n]):
    if stop < start: continue
    slices.append((int(start), int(stop) + 1))
```

iii. The notes say this reconstructs the paper’s lap definition more closely than trial-number changes.

## 1-e. How are trials filtered based on quality controls?

i. It drops trials when more than 35% of samples have raw lick values above 2, and rejects a session if fewer than two trials remain. It does not apply the reference converter’s under-50-timepoint filter.

ii.
```python
if lick_trial.size and np.mean(lick_trial > 2) > LICK_ARTIFACT_FRACTION:
    lick_artifact[trial_id] = True
...
if lick_artifact[trial_id]:
    continue
```

iii. The agent cites lick-artifact QC from the paper/reference path and reports 69 of 12,216 trials removed, close to the paper count, though its notes themselves describe a >30% criterion while code uses 35%.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. It uses only `processing/ophys/Deconvolved/plane0/data`, plus that response series’ ROI IDs and `PlaneSegmentation/iscell` for selection. It does not derive events from raw `Fluorescence` and `Neuropil`.

ii.
```python
neural_group = f["processing/ophys/Deconvolved/plane0"]
neural = neural_group["data"][()].astype(np.float32, copy=False)
roi_ids = neural_group["rois"][()]
```

iii. The agent argued that the stored Deconvolved stream was paper-equivalent calcium events and documented the absence of an exact stored dF/F stream as a limitation.

## 2-b. How is the `neural` data processed?

i. The response matrix is restricted to referenced `iscell` ROIs, cropped to the common neural/behavior length, transposed per trial, and—in 31.015625 Hz sessions—adjacent pairs are summed. No neuropil correction, maximin dF/F, smoothing, OASIS reconstruction, or multi-plane pooling is performed.

ii.
```python
iscell = seg["iscell"][()][roi_ids, 0] > 0.5
neural = neural[:, iscell]
...
neural_trial = arrays.neural[start:stop_exclusive].T
if factor == 2:
    neural_trial = rebin_2d_sum_time_last(neural_trial, factor)
```

iii. The notes prioritize a decoder-ready common timebase and speed, claiming the archive response is the closest supplied equivalent to paper events.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs are filtered by `iscell[:,0] > 0.5` after restricting to IDs referenced by the response series. The paper’s dF/F–speed correlation >0.5 putative-interneuron exclusion is not implemented.

ii.
```python
iscell = seg["iscell"][()][roi_ids, 0] > 0.5
neural = neural[:, iscell]
```

iii. The notes acknowledge the omitted interneuron filter, claiming exact paper dF/F was unavailable and curated neuron counts were plausible.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each neural matrix is sliced beginning at the detected trial-start frame, so column zero is aligned to trial start. No interpolation or timestamp-based offset correction is done.

ii.
```python
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    neural_trial = arrays.neural[start:stop_exclusive].T
```

iii. The agent’s validation compared raw slices with converted trials and reported no visible temporal misalignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. All output is declared at 15.5078125 Hz (64.483 ms). Native-rate sessions are unchanged; 31.015625 Hz sessions are rebinned by factor two, summing neural samples, averaging position/speed, and OR-ing licks.

ii.
```python
TARGET_RATE_HZ = 15.5078125
TARGET_DT_S = 1.0 / TARGET_RATE_HZ
...
elif math.isclose(arrays.rate_hz, 2.0 * TARGET_RATE_HZ): factor = 2
```

iii. The notes justify one decoder timebase as the paper’s effective per-plane rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is synthesized from the number of converted bins and the fixed target sampling interval, rather than calculated from raw timestamps.

ii.
```python
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
```

iii. The agent viewed trial-relative elapsed time as reconstructible after alignment and common-rate conversion.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. It creates a zero-based arithmetic sequence at 1/15.5078125 seconds per bin.

ii.
```python
time_from_start = np.arange(t_bins, dtype=np.float32) * TARGET_DT_S
```

iii. This enforces an exact common grid and makes the first trial bin time zero.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The sequence length is taken directly from the processed neural trial, so it has one value per neural column and starts at neural column zero.

ii.
```python
t_bins = neural_trial.shape[1]
time_from_start = np.arange(t_bins, dtype=np.float32) * TARGET_DT_S
```

iii. The agent’s spot checks treated raw slicing/rebinning and equal array lengths as sufficient alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from `processing/behavior/BehavioralTimeSeries/environment/data`.

ii.
```python
environment=beh["environment/data"][()][:t_common].astype(np.float32, copy=False)
```

iii. The notes map raw 0/1 to ENV1/ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial it removes negative values, takes the integer mode (default 0 if none), then repeats that value at every converted time bin.

ii.
```python
env = arrays.environment[start:stop_exclusive]
env = env[env >= 0]
trial_env[trial_id] = mode_int(env, default=0)
env_series = np.full((t_bins,), float(trial_env[trial_id]))
```

iii. The agent reasoned environment is a per-trial constant and modal aggregation is robust to invalid samples.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The saved input uses the zero-based enumeration of reconstructed trial slices, not the raw trial-number values (raw values are used for reward assignment).

ii.
```python
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    trial_series = np.full((t_bins,), float(trial_id))
```

iii. The notes say zero-based within-session numbering matches raw conventions and reference GLM trial IDs.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index is repeated across all time bins in the trial; filtering a trial does not renumber later retained trials.

ii.
```python
trial_series = np.full((t_bins,), float(trial_id), dtype=np.float32)
```

iii. The agent deliberately retained the original within-session index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It derives reward frames from `Reward/timestamps`, maps them onto position timestamps with `searchsorted`, reads raw `trial number` at those frames, and constructs current-trial reward flags before shifting them.

ii.
```python
idx = np.searchsorted(arrays.timestamps, arrays.reward_times, side="left")
rewarded_trial_ids = set(int(t) for t in arrays.trial_number[reward_frame_idx] if t >= 0)
```

iii. The notes specify reward events assigned to trials and a one-trial shift.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A trial is rewarded when its enumerated ID is in the rewarded-ID set. The binary vector is shifted right; the first trial is 0, and each value is repeated through the current trial.

ii.
```python
trial_rewarded = np.array([1 if trial_id in rewarded_trial_ids else 0 for trial_id in range(len(trial_slices))])
trial_prev_rewarded = np.concatenate([[0], trial_rewarded[:-1]])
```

iii. This directly implements omitted=0/rewarded=1 for the previous trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses clipped raw position and an inferred per-trial reward-zone label. Labels come from positions where `reward_zone > 0`, falling back to position at a reward event, then nearest one of the A/B/C centers.

ii.
```python
mask = reward_zone_signal[start:stop_exclusive] > 0
zone_pos = float(np.nanmedian(positions[start:stop_exclusive][mask]))
...
observed[trial_id] = int(np.argmin(np.abs(ZONE_CENTERS - zone_pos)))
```

iii. The agent used known zone geometry and the task’s stable blocks to fill omission trials.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For the inferred active zone, distance is position minus the near boundary before the zone, zero inside, and position minus the far boundary after it.

ii.
```python
before = position_cm < zone_start
after = position_cm > zone_end
distance[before] = position_cm[before] - zone_start
distance[after] = position_cm[after] - zone_end
```

iii. The notes state this matches the requested signed distance to the nearest point in the active reward zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit masks implement seven classes: <-50, [-50,-10), [-10,0), exactly 0, (0,10], (10,50], and >50 cm.

ii.
```python
out[distance_cm < -50.0] = 0
out[(distance_cm >= -50.0) & (distance_cm < -10.0)] = 1
...
out[distance_cm > 50.0] = 6
```

iii. The agent states these boundaries implement the decoder specification exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position is sliced with the same trial bounds as neural and factor-two averaged when neural is factor-two summed; distance/categories are then computed per resulting bin.

ii.
```python
position = arrays.position[start:stop_exclusive]
...
position = rebin_1d_mean(position, factor)
dist = compute_distance_to_zone(position, zone_idx)
```

iii. The agent’s raw-frame spot checks and plots were used to justify alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It comes from `processing/behavior/BehavioralTimeSeries/position/data`.

ii.
```python
position=beh["position/data"][()][:t_common].astype(np.float32, copy=False)
```

iii. The agent treats this field as corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Values are clipped to [0,450]; factor-two sessions average adjacent samples before categorization.

ii.
```python
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
position = rebin_1d_mean(position, factor)
```

iii. Clipping was intended to absorb small out-of-track artifacts.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. After clipping, integer division by 90 cm creates five classes, capped at 4. Thus exact internal boundaries (90,180,270,360) enter the upper bin.

ii.
```python
clipped = np.clip(position_cm, 0.0, TRACK_LENGTH_CM - 1e-6)
return np.minimum((clipped / 90.0).astype(np.int64), 4)
```

iii. The agent cites five equal 90-cm bins over the 450-cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the identical trial slice and matching factor-two aggregation, yielding one position class per neural bin.

ii.
```python
neural_trial = arrays.neural[start:stop_exclusive].T
position = arrays.position[start:stop_exclusive]
...
position = rebin_1d_mean(position, factor)
```

iii. The agent validated converted samples against raw frames.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from `processing/behavior/BehavioralTimeSeries/lick/data`.

ii.
```python
lick=beh["lick/data"][()][:t_common].astype(np.float32, copy=False)
```

iii. The notes identify this as the raw lick channel.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Raw values greater than zero become 1. In factor-two sessions, either positive source sample makes the rebinned output 1. Trials failing the separate lick-artifact criterion are removed.

ii.
```python
lick_binary = (arrays.lick[start:stop_exclusive] > 0).astype(np.int64)
...
lick_binary = rebin_1d_any(lick_binary, factor)
```

iii. The binary conversion follows the requested output; logical OR was chosen to preserve transient events during rebinning.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is sliced with the neural trial and, where needed, OR-rebinned over the same adjacent pairs used for neural summation.

ii.
```python
neural_trial = arrays.neural[start:stop_exclusive].T
lick_binary = (arrays.lick[start:stop_exclusive] > 0)
...
rebin_1d_any(lick_binary, factor)
```

iii. Plots and spot checks were cited as showing aligned lick transients.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It uses `reward_zone`, `position`, reward timestamps, raw trial number, and environment/task-block context.

ii.
```python
trial_zone, observed_zone_position = infer_trial_zones(positions=..., reward_zone_signal=..., reward_frame_idx=..., trial_by_frame=..., trial_env=...)
```

iii. The agent considered omission-trial labels missing and inferred them from observed zones plus stable task blocks.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Observed zone positions are mapped to the nearest A/B/C center. Missing values use the sole observed zone or pre/post trial-30 majorities; when environments change, all trials before/after 30 are forcibly assigned those majorities. The class is repeated over time.

ii.
```python
split = min(SWITCH_TRIAL_INDEX, n_trials)
pre_majority = trial_majority_zone(observed, 0, split, ...)
post_majority = trial_majority_zone(observed, split, n_trials, ...)
...
zone_bin = np.full((t_bins,), zone_idx)
```

iii. The notes use the known switch near trial 30 and roughly balanced zone classes as sanity anchors.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/timestamps`, position timestamps, and raw framewise `trial number`.

ii.
```python
reward_times=beh["Reward/timestamps"][()]
idx = np.searchsorted(arrays.timestamps, arrays.reward_times, side="left")
... arrays.trial_number[reward_frame_idx]
```

iii. The agent treats existence of a reward event assigned to a trial as rewarded.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are mapped to the following/equal behavior sample and clipped to valid bounds. Raw trial IDs at those frames form a set; each enumerated trial is 1 if present, then repeated over all bins.

ii.
```python
idx = np.searchsorted(arrays.timestamps, arrays.reward_times, side="left")
return np.clip(idx, 0, arrays.timestamps.shape[0] - 1)
...
reward_bin = np.full((t_bins,), int(trial_rewarded[trial_id]))
```

iii. The omission fraction was compared with the paper’s ~15% as a sanity check.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural and behavior streams are silently cropped to their minimum length; unequal marker counts are truncated to the smaller count; reversed trial pairs are skipped; missing environment defaults to 0; missing zones are filled by stable-block heuristics; indices are clipped; position/speed are clipped; unexpected rates, no zones, >2 observed zones, or too few trials raise errors.

ii.
```python
t_common = min(t_neural, t_behavior)
...
n = min(starts.size, teleports.size)
...
return np.clip(idx, 0, arrays.timestamps.shape[0] - 1)
```

iii. The agent calls these defensive archive-handling choices and relies on full-run statistics and spot checks, but several corrections are silent rather than explicitly validated.

## 13-a. What are the most time-consuming steps of the code?

i. Materializing each full NWB neural matrix is the main conversion cost; optional plotting adds overhead, and serial processing plus final pickle writing also cost time.

ii.
```python
neural = neural_group["data"][()].astype(np.float32, copy=False)
...
for session_idx, path in enumerate(files):
    arrays = load_session_arrays(path)
```

iii. The notes explicitly identify full-session NWB reads as the main inefficiency and keep processing plots off for the full conversion.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The Python loops over sessions and trials, reward-frame dictionary construction, zone observation/filling, trial QC, and trial reward membership could partly be vectorized. Variable trial lengths make full trial vectorization less straightforward.

ii.
```python
for frame_idx in reward_frame_idx: ...
for trial_id, (start, stop_exclusive) in enumerate(trial_slices): ...
for session_idx, path in enumerate(files): ...
```

iii. The notes emphasize already-vectorized array transforms but do not provide a detailed loop audit.

## 13-c. What processing does the code repeat multiple times?

i. Per trial, it repeatedly slices the same channels; for optional plots it recomputes clipping/rebinning for the example trial. Trial loops also separately compute environment/lick QC, zone evidence, and final arrays.

ii.
```python
for trial_id, (start, stop_exclusive) in enumerate(trial_slices): ...
...
example_position=np.clip(arrays.position[trial_slices[example_idx][0]:...])
```

iii. The notes accept plotting overhead because it is disabled in full mode and favor simple readable conversion logic.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads/stores `scanning` and ROI IDs although conversion never uses them after loading; `trial_rewarded` is passed to `infer_trial_zones` but unused; observed zone positions and extensive session metadata are retained only for audit; optional plots recompute example channels.

ii.
```python
scanning=beh["scanning/data"][()][:t_common]
...
trial_rewarded=trial_rewarded,  # parameter unused in infer_trial_zones
...
roi_ids=roi_ids
```

iii. The agent does not explicitly call these discarded operations out; it prioritizes traceability and diagnostics.


