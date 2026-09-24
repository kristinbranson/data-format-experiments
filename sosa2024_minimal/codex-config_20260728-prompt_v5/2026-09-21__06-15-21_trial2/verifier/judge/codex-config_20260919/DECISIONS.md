# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively discovers every matching NWB session under `/app/data`, sorts the paths, and opens each with `pynwb`. Each file becomes one output session.

ii.
```python
session_paths = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))
with NWBHDF5IO(str(session_path), "r", load_namespaces=True) as io:
    nwb = io.read()
```

iii. The trajectory says the release contains the 11 main-cohort mice and 152 sessions and that the agent intended to read every NWB session. It chose NWB because the files contain synchronized behavior and ophys streams.

## 1-b. How are the data split into subjects?

i. Subject IDs are parsed from each session's parent directory (`sub-...`). A first-seen lookup creates `subjects`, and `subject_idx` maps each session to that list.

ii.
```python
subject = session_path.parent.name.replace("sub-", "")
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
data["subject_idx"].append(subject_to_idx[subject])
```

iii. The agent confirmed that the directory layout represented the 11 switch-task mice and treated that naming convention as the subject identifier.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session, in lexicographically sorted path order.

ii.
```python
for session_idx, session_path in enumerate(session_paths, start=1):
    neural_trials, input_trials, output_trials, plane_idx, session_info = _load_session(session_path)
    data["neural"].append(neural_trials)
```

iii. The trajectory explicitly describes the dataset as NWB data across mice and days and processes all 152 files as sessions.

## 1-d. How are the data split into trials?

i. Nonzero `trial_start` and `teleport` samples are paired in order. An unmatched initial teleport or terminal start is removed, remaining pairs are truncated to the shorter count, invalid pairs are dropped, and each trial is `[start, stop)`.

ii.
```python
starts = np.flatnonzero(trial_start > 0)
teleports = np.flatnonzero(teleport > 0)
...
for start, stop in zip(starts, teleports):
    if stop > start:
        bounds.append((int(start), int(stop)))
```

iii. The agent states that start-inclusive/teleport-exclusive slicing matches the repository's `keep_teleports=False` path and checked the whole dataset for boundary edge cases.

## 1-e. How are trials filtered based on quality controls?

i. There is no minimum-duration trial filter. Only pairs with `stop > start` survive, and a session is rejected if fewer than two such trials remain.

ii.
```python
if stop > start:
    bounds.append((int(start), int(stop)))
...
if len(bounds) < 2:
    raise RuntimeError(...)
```

iii. The trajectory mentions checking edge cases but gives no scientific justification for omitting the reference's short-trial exclusion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes directly from the NWB `Deconvolved` ROI response series, after applying `iscell`; it is not recomputed from raw fluorescence and neuropil.

ii.
```python
deconv_iface = nwb.processing["ophys"].data_interfaces["Deconvolved"]
deconv_parts.append(np.asarray(rr.data[:, :], dtype=np.float32)[:, keep_plane])
```

iii. The agent reasoned that the NWB already contained curated ophys and chose the released deconvolved traces. It benchmarked this option against recomputing dF/F on a sample, then adopted the direct signal for practicality.

## 2-b. How is the `neural` data processed?

i. Per-plane deconvolved matrices are filtered by the corresponding `iscell` slice, concatenated across planes, truncated to the shared behavior/ophys length, speed-correlation filtered, trial-sliced, transposed to neuron-by-time, and cast to float16. No dF/F baseline or OASIS calculation is performed by this script.

ii.
```python
deconv = np.concatenate(deconv_parts, axis=1)
deconv = deconv[:n_frames]
deconv = deconv[:, non_interneuron]
neural_trial = np.ascontiguousarray(deconv[start:stop].T, dtype=np.float16)
```

iii. The agent wanted to pool split planes like the paper and used float16 to make the all-session pickle tractable. Its metadata calls the interneuron calculation a “practical approximation”; it did not claim to reproduce the paper's fluorescence preprocessing exactly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs must have `iscell=True`. The code additionally excludes cells whose provided deconvolved activity has correlation greater than 0.5 with speed over on-track frames. Multi-plane masks are applied per plane.

ii.
```python
keep_plane = iscell_all[plane_mask]
...
non_interneuron = _speed_corr_filter(deconv, speed, bounds)
deconv = deconv[:, non_interneuron]
```

iii. The agent inspected `iscell`, identified the paper's extra interneuron screen, and benchmarked whether to reproduce it. It used deconvolved activity rather than the paper/reference dF/F, explicitly describing this as an approximation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start simply by slicing the synchronized frame matrix from the `trial_start` index; the first retained frame is time zero.

ii.
```python
neural_trial = np.ascontiguousarray(deconv[start:stop].T, dtype=np.float16)
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The agent determined that neural and behavior data share a synchronized frame stream, so common indices preserve alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The declared resolution is `1000 / 15.5078125 = 64.48387 ms`. No temporal rebinning or resampling is applied.

ii.
```python
FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
```

iii. The agent preserves the released synchronized frames. It does not derive the rate per file, but selected the common per-plane rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the timestamps attached to the behavior `position` series and the trial boundary indices.

ii.
```python
timestamps = np.asarray(beh["position"].timestamps[:], dtype=np.float64)
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. The agent regarded the behavior streams as synchronized and used a timestamp-bearing series directly.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp in each trial is subtracted from every timestamp in that trial, then values are cast to float32.

ii.
```python
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. This makes the trial-start frame exactly zero, as required by the requested alignment.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Time and neural arrays use the identical `[start:stop]` frame slice. All continuous streams are first truncated to a common minimum length.

ii.
```python
n_frames = min(..., deconv.shape[0])
timestamps = timestamps[:n_frames]
deconv = deconv[:n_frames]
```

iii. The agent encountered a one-frame mismatch and described common-minimum truncation as consistent with the repository's one-frame correction logic.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from the framewise behavior `environment` series.

ii.
```python
environment = np.asarray(beh["environment"].data[:], dtype=np.float32)
env_vals = environment[start:stop]
```

iii. The agent explicitly checked for within-session environment transitions so they would be retained.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Negative values are discarded, the integer mode is taken per trial with fallback 0, and that scalar is repeated over all trial frames.

ii.
```python
env_vals = env_vals[env_vals >= 0]
env = float(_safe_mode(env_vals.astype(int), fallback=0))
np.full(T, env, dtype=np.float32)
```

iii. The agent considered environment a per-trial label and used the mode to tolerate anomalous samples while preserving within-session switches.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It normally uses the framewise `trial number` behavior series at the trial start; if negative, it falls back to the zero-based loop index.

ii.
```python
trial_number = np.asarray(beh["trial number"].data[:], dtype=np.float32)
tnum = float(trial_number[start]) if trial_number[start] >= 0 else float(trial_idx)
```

iii. No explicit rationale for preferring the stored field appears in the trajectory.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The chosen scalar is repeated at every time point in the trial.

ii.
```python
np.full(T, tnum, dtype=np.float32)
```

iii. This implements the requested per-trial variable, with a fallback for invalid stored values.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from timestamps of the sparse behavior `Reward` series and the current/previous trial's timestamp interval.

ii.
```python
reward_timestamps = np.asarray(beh["Reward"].timestamps[:], dtype=np.float64)
reward_outcome_by_trial[trial_idx] = np.uint8(np.any(
    (reward_timestamps >= trial_start_t) & (reward_timestamps < trial_end_t)))
```

iii. The agent chose sparse reward timestamps because they directly encode reward delivery.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Each trial is marked rewarded if any reward timestamp lies in its half-open time interval. The next trial receives that value; the first trial receives 0. The scalar is repeated across frames.

ii.
```python
prev_reward = float(reward_outcome_by_trial[trial_idx - 1]) if trial_idx > 0 else 0.0
np.full(T, prev_reward, dtype=np.float32)
```

iii. This directly follows the requested omitted/rewarded binary definition.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from framewise `position` and a per-trial zone inferred using positions where framewise `reward_zone > 0`, compared with hard-coded A/B/C centers.

ii.
```python
rz_pos = position[start:stop][reward_zone[start:stop] > 0]
raw_zone[trial_idx] = int(np.argmin(np.abs(ZONE_CENTERS_CM - np.median(rz_pos))))
```

iii. The agent states that zone labels are inferred from reward-zone event positions and that missed entry events are gap-filled using session block structure.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position is clipped to 0–450 cm. Signed distance is position minus the near edge before a zone, zero inside it, and position minus the far edge after it.

ii.
```python
pos = np.clip(position[start:stop], 0.0, TRACK_LENGTH_CM)
dist[before] = position_cm[before] - start
dist[after] = position_cm[after] - end
```

iii. This represents signed distance to any point in the active reward-zone interval, matching the semantic task definition.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven explicit Boolean masks implement the instructed cutoffs, with exact zero assigned class 3.

ii.
```python
bins[dist < -50.0] = 0
bins[(dist >= -50.0) & (dist < -10.0)] = 1
bins[(dist >= -10.0) & (dist < 0.0)] = 2
bins[dist == 0.0] = 3
bins[(dist > 0.0) & (dist <= 10.0)] = 4
bins[(dist > 10.0) & (dist <= 50.0)] = 5
bins[dist > 50.0] = 6
```

iii. The boundaries are a direct implementation of the decoder instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data are sliced with identical trial bounds, yielding one distance class per neural frame.

ii.
```python
pos = position[start:stop]
neural_trial = deconv[start:stop].T
```

iii. The agent relied on the NWB synchronized frame stream and common-length truncation.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It comes from the behavior `position` series.

ii.
```python
position = np.asarray(beh["position"].data[:], dtype=np.float32)
pos = np.clip(position[start:stop], 0.0, TRACK_LENGTH_CM)
```

iii. Position is the direct virtual-corridor coordinate.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Per-trial position is clipped to [0, 450] cm and passed to the categorical binning function.

ii.
```python
pos = np.clip(position[start:stop], 0.0, TRACK_LENGTH_CM)
_bin_absolute_position(pos)
```

iii. The agent imposed physical track limits; the trajectory does not provide a separate justification for clipping.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five bins use thresholds at 90, 180, 270, and 360 cm; boundaries enter the upper bin.

ii.
```python
bins[position_cm < 90.0] = 0
bins[(position_cm >= 90.0) & (position_cm < 180.0)] = 1
...
bins[position_cm >= 360.0] = 4
```

iii. These are five equal-width divisions of the instructed 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position and neural activity use the same `[start:stop]` frame interval after common-length truncation.

ii.
```python
pos = position[start:stop]
neural_trial = deconv[start:stop].T
```

iii. The synchronized NWB frame stream makes additional interpolation unnecessary.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavior `lick` series.

ii.
```python
lick = np.asarray(beh["lick"].data[:], dtype=np.float32)
```

iii. The agent used the released synchronized behavioral field directly.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Every positive value becomes 1 and every other value becomes 0; the result is uint8.

ii.
```python
lick_bin = (lick[start:stop] > 0).astype(np.uint8, copy=False)
```

iii. This implements the requested no/yes output despite raw counts or amplitudes.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural streams use identical trial indices after common-minimum truncation.

ii.
```python
lick_bin = (lick[start:stop] > 0)
neural_trial = deconv[start:stop].T
```

iii. The agent treats all behavior series as synchronized to ophys frames.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It uses `reward_zone`, `position`, trial bounds, and hard-coded zone intervals/centers.

ii.
```python
ZONE_CENTERS_CM = (ZONE_STARTS_CM + ZONE_ENDS_CM) / 2.0
rz_pos = position[start:stop][reward_zone[start:stop] > 0]
```

iii. The agent inferred block identity from the observed positions of zone events because some trials miss an entry event.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The median event position is mapped to the nearest zone center. If all detected trials agree, that zone fills the session. Otherwise, modes before and after trial 30 fill two blocks. A forward/edge fill is used if one side lacks detections. The final 0/1/2 label is repeated across trial frames.

ii.
```python
inferred[:SWITCH_SPLIT_TRIAL] = _safe_mode(pre_valid)
inferred[SWITCH_SPLIT_TRIAL:] = _safe_mode(post_valid, fallback=int(inferred[SWITCH_SPLIT_TRIAL - 1]))
...
np.full(T, zone_idx, dtype=np.uint8)
```

iii. The trajectory says this respects the paper's 30-trial switch structure and fills trials whose entry event was missed by interpolation.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from timestamps of the behavior `Reward` series and position-series timestamps defining trial intervals.

ii.
```python
reward_timestamps = np.asarray(beh["Reward"].timestamps[:], dtype=np.float64)
trial_start_t = timestamps[start]
trial_end_t = timestamps[stop]
```

iii. Sparse reward timestamps directly indicate delivered reward events.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is 1 if any reward timestamp lies in `[trial_start_t, trial_end_t)`, otherwise 0. This scalar is repeated at all trial time points.

ii.
```python
reward_outcome_by_trial[trial_idx] = np.uint8(np.any(
    (reward_timestamps >= trial_start_t) & (reward_timestamps < trial_end_t)))
np.full(T, reward_outcome, dtype=np.uint8)
```

iii. This is the direct binary reward/no-reward interpretation required by the instructions.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Continuous streams are truncated to their shared minimum length. Unmatched leading/trailing trial events are removed and invalid bounds dropped. Missing zone events are filled by block modes or nearest prior/edge labels. Negative environment values are ignored with fallback 0; negative trial number falls back to loop index; NaNs are replaced with zero only inside the speed-correlation computation. Sessions with fewer than two trials or no remaining neurons raise errors.

ii.
```python
n_frames = min(..., deconv.shape[0])
...
if teleports.size == starts.size + 1 and teleports[0] < starts[0]:
    teleports = teleports[1:]
...
x = np.nan_to_num(x.astype(np.float32), nan=0.0)
```

iii. The trajectory records two concrete repairs: per-plane ROI metadata handling and a one-frame behavior/ophys mismatch. The agent used shared-minimum truncation because it viewed this as consistent with repository correction logic.

## 13-a. What are the most time-consuming steps of the code?

i. Reading full deconvolved matrices from 152 NWB files, computing per-cell speed correlations, duplicating session neural matrices into per-trial arrays, serializing the 4.6 GB pickle, and downstream PCA/training dominate runtime.

ii.
```python
deconv_parts.append(np.asarray(rr.data[:, :], dtype=np.float32)[:, keep_plane])
for session_idx, session_path in enumerate(session_paths, start=1):
    ...
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory explicitly calls the conversion I/O-bound, notes that each session requires the full deconvolved matrix, and reports the 4.6 GB output and lengthy decoder training.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The loops marking the on-track mask, pairing trial bounds, inferring zone per trial, computing reward outcomes, constructing per-trial arrays, and forward-filling zone labels could partly be vectorized. Trial construction remains naturally loop-based because lengths vary.

ii.
```python
for start, stop in bounds:
    track_mask[start:stop] = True
for trial_idx, (start, stop) in enumerate(bounds):
    ...
```

iii. The trajectory provides no explicit vectorization discussion. The agent did vectorize the expensive correlation across neurons, leaving mostly variable-length trial loops.

## 13-c. What processing does the code repeat multiple times?

i. Trial bounds are traversed repeatedly for the track mask, zone inference, reward outcomes, trial assembly, and metadata environment summaries. Each trial output also repeats per-trial scalar values across every frame.

ii.
```python
for start, stop in bounds: ...
for trial_idx, (start, stop) in enumerate(bounds): ...
for trial_idx, (start, stop) in enumerate(bounds): ...
```

iii. No rationale is recorded. The repeated passes keep the code modular and are small relative to NWB I/O and neural-array copying.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `plane_idx` and its per-neuron values are built but ultimately converted to an all-zero CA1 `brain_region_idx`; `env_counter.update()` is a no-op; several metadata summaries are computed only for reporting. Neural data are also copied/cast per trial, increasing conversion work and storage.

ii.
```python
plane_parts.append(np.full(int(np.sum(keep_plane)), plane_num, dtype=np.int64))
data["brain_region_idx"].append(np.zeros(plane_idx.shape[0], dtype=np.int64))
...
env_counter.update()
```

iii. The trajectory emphasizes pooled-plane analysis and validation summaries, explaining why plane handling and reporting exist, but it gives no justification for retaining then discarding plane identity or for the no-op environment counter.
