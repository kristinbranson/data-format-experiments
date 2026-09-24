# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. It recursively selects every matching NWB session with `sorted(Path('/app/data').glob('sub-*/sub-*_behavior+ophys.nwb'))` (unless `--sample`), opens each with `NWBHDF5IO`, and reads behavior and ophys arrays.

ii.
```python
files = sorted(data_root.glob("sub-*/sub-*_behavior+ophys.nwb"))
with NWBHDF5IO(str(file_path), "r", load_namespaces=True) as io:
    nwb = io.read()
```

iii. The notes report 11 subjects, 152 sessions, and 12,216 pulse-defined trials, matching the raw audit.

## 1-b. How are the data split into subjects?

i. The parent directory (`sub-m3`, etc.) determines the subject; first occurrence builds `subjects`, and every session receives an index in `subject_idx`.

ii.
```python
subject = file_path.parent.name.replace("sub-", "")
if meta.subject not in subject_to_idx:
    subject_to_idx[meta.subject] = len(subjects)
    subjects.append(meta.subject)
subject_idx.append(subject_to_idx[meta.subject])
```

iii. The DANDI-style directory organization has one directory per mouse; the reported 11 mice match the data audit.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session and becomes one outer-list entry in `neural`, `input`, and `output`.

ii.
```python
for session_idx, file_path in enumerate(files):
    neural_trials, input_trials, output_trials, ... = convert_session(file_path)
    neural_all.append(neural_trials)
```

iii. The filename/session metadata and NWB organization identify one recording session per file.

## 1-d. How are the data split into trials?

i. Starts are every positive `trial_start` sample. Each is paired in order with the next positive `teleport` sample; slices are `[start, stop)`. Native trial-number transitions do not define bounds.

ii.
```python
starts = np.flatnonzero(trial_start > 0)
stops = np.flatnonzero(teleport > 0)
...
bounds.append((int(start), stop))
...
deconv[start:stop, :]
```

iii. The agent found an extra tunnel-only native trial-number segment in one session and therefore treated start/teleport pulses as authoritative, analogous to the paper’s trial-start/teleport indices.

## 1-e. How are trials filtered based on quality controls?

i. No minimum-duration filter is applied. Incomplete/unpairable terminal starts are omitted; empty windows, windows without a valid environment, and invalid bounds are skipped or rejected. Sessions must retain at least two trials.

ii.
```python
if stop <= start: continue
if env_valid.size == 0: continue
if len(neural_trials) < 2: raise ValueError(...)
```

iii. The notes say completed long laps were deliberately preserved and all 12,216 pulse-defined trials survived; sentinel samples are excluded by trial windows.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. It comes directly from `processing/ophys/Deconvolved` ROI response series, plus `ImageSegmentation` fields `iscell` and `planeIdx` for selection/order.

ii.
```python
deconv_mod = nwb.processing["ophys"]["Deconvolved"]
seg = nwb.processing["ophys"]["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]
```

iii. The agent reasoned that the paper decoder used an events representation and treated the NWB `Deconvolved` array as that representation.

## 2-b. How is the `neural` data processed?

i. Per-plane deconvolved arrays are filtered to curated cells, concatenated across planes, sliced by trial, transposed to neuron×time, non-finite values are replaced by zero, and values are stored as float16. No dF/F recomputation or new deconvolution is done.

ii.
```python
deconv = np.concatenate(arrays, axis=1)
neural_trial = np.nan_to_num(deconv[start:stop, :].T, ...).astype(np.float16, copy=False)
```

iii. The notes justify direct use as matching the decoder notebook’s `events`; float16 was chosen to halve storage, with exact round-trip spot checks.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs with `iscell[:,0] == 1` are kept. There is no speed-correlation/putative-interneuron exclusion.

ii.
```python
keep = iscell[:, 0] == 1
plane_keep_mask = keep & (plane_idx == plane_num)
```

iii. The agent identifies `iscell` as Suite2p/manual curation and reports 138,678 retained cells. It did not document or implement the paper’s r>0.5 interneuron filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial start is column/time zero: the neural array is sliced beginning at the `trial_start` pulse and ending before teleport. No further shift is applied.

ii.
```python
for trial_idx, (start, stop) in enumerate(trial_bounds):
    neural_trial = deconv[start:stop, :].T
```

iii. The NWB behavior and imaging streams are already frame-aligned; spot checks found no temporal shift.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The native median timestamp interval is enforced as 1/15.5078125 s, and metadata records about 64.484 ms.

ii.
```python
frame_dt_s = float(np.median(np.diff(frame_times)))
time_bin_ms = float(stats["frame_dt_s"]) * 1000.0
```

iii. The paper and repository describe behavior/neural samples at the imaging-frame rate (~15.5 Hz).

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It uses timestamps attached to the raw `position` behavior series and the timestamp at the paired trial-start frame.

ii.
```python
frame_times = np.asarray(beh.time_series["position"].timestamps[:], dtype=np.float64)
```

iii. The agent found the behavioral streams aligned on imaging frames and selected position timestamps as the common time base.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp in each trial is subtracted from every timestamp in that trial and the result is cast to float32.

ii.
```python
trial_time_s = (frame_times[start:stop] - frame_times[start]).astype(np.float32)
```

iii. This directly expresses elapsed seconds from the required trial-start event.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The exact same `[start:stop)` frame slice determines both elapsed time and neural columns, so their lengths and columns correspond one-to-one.

ii.
```python
neural_trial = deconv[start:stop, :].T
trial_time_s = frame_times[start:stop] - frame_times[start]
```

iii. The NWB export already aligns behavior to imaging frames; the full-data spot check confirmed the timestamp input against raw data.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the frame-aligned behavior series `environment`.

ii.
```python
environment = np.asarray(beh.time_series["environment"].data[:], dtype=np.float32)
```

iii. The notes explicitly prefer the aligned stream over scene prefixes because environments can change within a session.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Negative sentinel values are removed, the modal integer environment in the trial is selected, and that value is repeated across all trial frames.

ii.
```python
env_valid = env_trial[env_trial >= 0]
env_code = float(mode_int(env_valid))
np.full(trial_time_s.shape, env_code, dtype=np.float32)
```

iii. The stream is expected to be constant within a valid trial; mode makes the per-trial value robust to isolated anomalies.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It uses the raw behavior `trial number` value at the trial-start frame; the pulse-pair loop index is only a fallback for a negative label.

ii.
```python
trial_number = np.asarray(beh.time_series["trial number"].data[:], dtype=np.float32)
trial_num_native = int(round(float(trial_number[start])))
```

iii. The agent wanted native zero-indexed completed-trial labels while avoiding use of trial-number transitions for segmentation.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The start-frame value is rounded to an integer, replaced with the zero-based loop index if negative, converted to float, and broadcast across time.

ii.
```python
if trial_num_native < 0:
    trial_num_native = trial_idx
np.full(trial_time_s.shape, float(trial_num_native), dtype=np.float32)
```

iii. The notes state this preserves the reference code’s zero-indexing and switch-at-trial-30 convention.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `Reward.timestamps`, rasterized against position timestamps, then summarized within the previous pulse-defined trial.

ii.
```python
reward_times = np.asarray(beh.time_series["Reward"].timestamps[:], dtype=np.float64)
reward_frames = rasterize_reward_events(frame_times, reward_times)
```

iii. The NWB exports rewards as timestamped events rather than the reference’s framewise reward series.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Each current trial outcome is `any(reward_frames[start:stop])`; the outcome vector is shifted by one, with the first trial set to zero, then broadcast per frame.

ii.
```python
prev_reward_outcomes = np.zeros_like(reward_outcomes)
prev_reward_outcomes[1:] = reward_outcomes[:-1]
```

iii. This follows the requested binary rewarded/omitted definition; zero is the documented no-history convention.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw `position` and an active reward-zone label inferred from the NWB identifier’s scene string plus the fixed 30-trial switch schedule; A/B/C map to fixed intervals.

ii.
```python
rz_label = reward_labels[trial_idx]
rz_start, rz_stop = REWARD_ZONE_COORDS_CM[rz_label]
```

iii. The agent followed the repository’s scene-based reward-zone logic because the NWB lacks a direct categorical A/B/C field.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is zero inside the active interval, position−lower-bound before it, and position−upper-bound after it.

ii.
```python
out[below] = position_cm[below] - start_cm
out[above] = position_cm[above] - stop_cm
```

iii. This interprets “distance to any location in the reward zone” literally and matches the paper’s reward-zone intervals.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit Boolean masks produce the seven requested classes: <-50, [-50,-10), [-10,0), exactly 0, (0,10], (10,50], and >50.

ii.
```python
out[distance_cm < -50.0] = 0
...
out[distance_cm == 0.0] = 3
...
out[distance_cm > 50.0] = 6
```

iii. The boundaries are a direct transcription of the decoder specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data use the same trial `[start:stop)` frames; the derived distance therefore has one value per neural column.

ii.
```python
pos_trial = position[start:stop]
neural_trial = deconv[start:stop, :].T
```

iii. The raw-to-converted trial spot check confirmed identical frame alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the raw frame-aligned behavior series `position`.

ii.
```python
position = np.asarray(beh.time_series["position"].data[:], dtype=np.float32)
pos_trial = position[start:stop]
```

iii. The NWB position stream records corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The per-trial slice has NaN/+inf/−inf replaced by zero, then it is categorized; no spatial smoothing or resampling is applied.

ii.
```python
pos_trial = np.nan_to_num(position[start:stop], nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The notes regard the aligned raw position as already suitable for task-required discretization.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Boolean masks implement five bins: <90, [90,180), [180,270), [270,360), and >=360 cm.

ii.
```python
out[position_cm < 90.0] = 0
...
out[position_cm >= 360.0] = 4
```

iii. These are five equal 90-cm bins spanning the 450-cm track, with end bins absorbing out-of-range samples.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position and neural activity are sliced with the same `[start:stop)` indices.

ii.
```python
pos_trial = position[start:stop]
neural_trial = deconv[start:stop, :].T
```

iii. The agent relies on the NWB’s existing behavior/imaging-frame alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the raw framewise `lick` behavior series.

ii.
```python
lick_counts = np.asarray(beh.time_series["lick"].data[:], dtype=np.float32)
```

iii. The reference exploration described this as a count-like stream requiring binary conversion and artifact cleanup.

## 9-b. What processing is involved in computing `output` *Lick*?

i. If more than 35% of a trial’s samples exceed count 2, the full trial is treated as a sensor artifact and zeroed; otherwise any positive count becomes 1.

ii.
```python
bad_trial = float(np.mean(corrected > 2.0)) > 0.35
if bad_trial: corrected[:] = 0.0
corrected = (corrected > 0).astype(np.int8)
```

iii. The agent says this is the reference heuristic and validated two artifact-flagged trials against the intended all-zero result.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The lick series is sliced on the same `[start:stop)` trial window as neural activity before cleanup/binarization.

ii.
```python
lick_binary, bad_lick = correct_lick_trial(lick_counts[start:stop])
```

iii. Existing NWB frame alignment and common slicing provide one lick label per neural time bin.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is inferred from the raw NWB `identifier` scene component and trial ordinal, rather than from the raw `reward_zone` behavior stream.

ii.
```python
meta = parse_identifier(nwb.identifier, file_path)
reward_labels = scene_to_reward_labels(meta.scene, len(trial_bounds))
```

iii. The agent concluded categorical A/B/C is omitted from NWB and reconstructed it using the repository’s scene naming logic.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Fixed-location scenes map all trials to A/B/C; switch scenes map the first 30 trials to the source and subsequent trials to the destination; labels become 0/1/2 and are broadcast across time.

ii.
```python
first_n = min(change_trial, ntrials)
...
return ["A"] * first_n + ["B"] * second_n
...
np.full(trial_time_s.shape, reward_codes[trial_idx], dtype=np.int8)
```

iii. The paper states switches occur after 30 trials, and the reference repository derives zone identity from scene metadata.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the event timestamps of the raw `Reward` behavior time series and common behavior-frame timestamps.

ii.
```python
reward_times = np.asarray(beh.time_series["Reward"].timestamps[:], dtype=np.float64)
```

iii. Reward deliveries are event-style in NWB; auto-rewards are included because they are deliveries.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Reward timestamps are placed on exactly matching frame timestamps with `searchsorted`; a trial is 1 if any reward frame lies in `[start,stop)`, otherwise 0, and the result is broadcast.

ii.
```python
reward_indices = np.searchsorted(frame_times, reward_times)
...
outcomes[idx] = int(np.any(reward_frames[start:stop] > 0))
```

iii. The agent found exact timestamp equality in this dataset and verified both rewarded and omitted examples.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Exactly one extra neural frame is trimmed (seen in 10 multi-plane sessions); any larger neural/behavior mismatch errors. Non-finite neural/position/speed values become zero, invalid-environment trials are skipped, reward timestamps must match exactly, and sessions need at least two usable trials.

ii.
```python
if frame_delta == 1: deconv = deconv[:frame_times.shape[0], :]
elif frame_delta != 0: raise ValueError(...)
np.nan_to_num(...)
```

iii. The one-frame mismatch was investigated as a synchronization edge case and revalidated. The agent otherwise favors explicit failures over silent broad cropping.

## 13-a. What are the most time-consuming steps of the code?

i. Reading all large NWB arrays across 152 sessions, constructing/storing per-trial arrays, serializing the full pickle, and—when enabled—reopening files and rendering diagnostic plots dominate. The code records per-session/runtime timings.

ii.
```python
session_t0 = time.perf_counter()
...
pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes benchmark conversion as under one second even for a large session without plotting; full I/O and downstream training are the practical costs.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial outcome loop, per-trial conversion/broadcast loop, trial-bound pairing loop, and per-plane extraction loop could partly be vectorized. Variable-length trials make full vectorization awkward.

ii.
```python
for idx, (start, stop) in enumerate(bounds): ...
for trial_idx, (start, stop) in enumerate(trial_bounds): ...
for plane_name in plane_names: ...
```

iii. The implementation favors clear variable-length trial construction; most heavy numerical operations inside each trial are already NumPy-vectorized.

## 13-c. What processing does the code repeat multiple times?

i. With `--show-processing`, selected NWBs are reopened and behavior arrays reread; trial bounds and scene labels are recomputed several times in one plotting call. Trial-level constants are also materialized across every frame.

ii.
```python
with NWBHDF5IO(str(file_path), "r", ...) as io: ...
pair_trial_bounds(trial_start, teleport)  # repeated in plotting arguments
```

iii. Plotting is optional and limited to two sessions; repeated constants satisfy the uniform channel×time format.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `plane_per_cell` and `full_lick` are computed but not used meaningfully (`plane_per_cell` is replaced by all-zero CA1 indices; `full_lick` is never consumed). Plot-only rereads/figures are discarded from the dataset. Float16 conversion also discards precision.

ii.
```python
full_lick = np.concatenate([trial[3] for trial in output_trials])
plane_all.append(session_brain_region_idx(neural_trials[0].shape[0]))
```

iii. The agent intentionally collapses all neurons to CA1 and uses float16 for storage; the unused `full_lick` appears accidental, while plots are optional diagnostics.


