# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively discovers every NWB file under `/app/data/sub-*`, numerically sorts them by mouse and session, and opens each file with PyNWB. Full mode processes all 152 files sequentially.

ii.
```python
files = sorted(DATA_ROOT.glob("sub-*/*.nwb"), key=natural_session_key)
with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
    nwb = io.read()
```

iii. The notes justify this from the observed one-file-per-session layout (11 mice, 152 files), the mandatory PyNWB constraint, and memory-efficient sequential processing of the 92-GB source.

## 1-b. How are the data split into subjects?

i. Subject identity is read from `nwb.subject.subject_id`; unique IDs are naturally sorted, and each session receives an index into that list.

ii.
```python
subject_id = str(nwb.subject.subject_id)
subjects = sorted(set(subject_ids), key=lambda value: int(re.search(r"\d+", value).group()))
subject_idx = np.asarray([subject_lookup[subject] for subject in subject_ids])
```

iii. The agent checked that this yields the expected 11 mice and agrees with file/directory identities.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session and produces one element in each top-level session list.

ii.
```python
for session_idx, path in enumerate(files):
    converted, info = process_session(path, show_processing=plot_this)
    neural.append(converted["neural"])
```

iii. Session filenames contain `ses-<day>` and the release contains 152 session files, consistent with the paper metadata.

## 1-d. How are the data split into trials?

i. Trials are half-open intervals from each positive `trial_start` sample through, but excluding, its paired positive `teleport` sample.

ii.
```python
starts = np.flatnonzero(np.asarray(behavior["trial_start"].data[:]) > 0)
stops = np.flatnonzero(np.asarray(behavior["teleport"].data[:]) > 0)
for raw_idx, (start, stop) in enumerate(zip(starts, stops)):
    trial_position = position[start:stop]
```

iii. The notes explain that this is the NWB equivalent of the legacy one-based `start-1:stop-1` convention; audited trials start near 0 cm, end near 450 cm, and exclude teleport artifacts.

## 1-e. How are trials filtered based on quality controls?

i. A trial is removed when more than 30% of its samples have lick values greater than 2. A retained trial must contain at least two frames; otherwise conversion raises rather than silently dropping it. Low-speed samples are retained.

ii.
```python
corrupt_lick = np.asarray([
    np.mean(lick[start:stop] > 2) > LICK_ERROR_FRACTION
    for start, stop in zip(starts, stops)
])
if corrupt_lick[raw_idx]:
    continue
```

iii. The agent attributes this to the paper's lick-sensor-error rule and reports exactly 81 exclusions. It retains low-speed frames because speed class 0 and lick are required decoder outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is recomputed from the NWB `Fluorescence` and `Neuropil` ROI response series, restricted first to manually curated `iscell` ROIs; the NWB `Deconvolved` export is not used.

ii.
```python
fluorescence_series = ophys["Fluorescence"].roi_response_series
neuropil_series = ophys["Neuropil"].roi_response_series
iscell = np.asarray(segmentation["iscell"].data[:])[:, 0] > 0
```

iii. The agent found that the exported `Deconvolved` field is Suite2p output, whereas the paper analyzes its own trial-wise dF/F and OASIS events.

## 2-b. How is the `neural` data processed?

i. Per plane and trial, it subtracts `0.7*Fneu`, restores `0.7` times the trial mean neuropil, applies Gaussian(15)-minimum(300)-maximum(300) maximin baseline estimation, computes `(F-baseline)/abs(baseline)`, Gaussian-smooths with sigma 2, and OASIS-deconvolves with tau 0.7 at 15.5078125 Hz. Retained planes are concatenated.

ii.
```python
corrected = f_trial - NEUROPIL_COEF * n_trial
corrected += NEUROPIL_COEF * np.mean(n_trial, axis=0, keepdims=True)
baseline = gaussian_filter1d(corrected, 15, axis=0)
baseline = minimum_filter1d(baseline, 300, axis=0)
baseline = maximum_filter1d(baseline, 300, axis=0)
trial_dff = gaussian_filter1d((corrected - baseline) / np.abs(baseline), 2, axis=0)
events[start:stop] = dcnv.oasis(dff[start:stop].T, 2000, OASIS_TAU_S, EFFECTIVE_FS_HZ).T
```

iii. The notes cite the paper's preprocessing functions and parameters and report independent full-trial numerical checks. The agent deliberately processes every baseline within the lap and does not implement the reference's session-dependent `keep_teleports` behavior.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It retains manually curated `iscell[:,0] > 0` ROIs, then excludes cells whose trial-sample dF/F has Pearson correlation with speed greater than 0.5.

ii.
```python
local_keep = np.flatnonzero(iscell[roi_ids])
correlations = speed_correlations(dff, speed, starts, stops)
keep = np.isfinite(correlations) & (correlations <= 0.5)
plane_events.append(events[:, keep])
```

iii. These are identified as the paper's manual cell curation and putative-interneuron criterion; 402 of 138,678 manual session-neurons were excluded.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural rows are already frame-synchronous with behavior. Each trial slices events with the same `[trial_start, teleport)` indices, so column zero aligns to trial start.

ii.
```python
neural = np.concatenate([events[start:stop].T for events in plane_events], axis=0)
```

iii. The agent validated equal neural/input/output lengths and audited position at the boundaries; no interpolation or offset was needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native synchronized frames are kept at 15.5078125 Hz, or 64.483627204 ms per bin. No rebinning or resampling occurs.

ii.
```python
EFFECTIVE_FS_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / EFFECTIVE_FS_HZ
"time_bin_size": float(TIME_BIN_MS)
```

iii. The notes state that behavior and imaging share this sampling grid and that preserving it matches the reference processing.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It comes from the timestamp array attached to the raw `position` behavior series and the detected trial start index.

ii.
```python
timestamps = np.asarray(behavior["position"].timestamps[:], dtype=np.float64)
time_from_start = timestamps[start:stop] - timestamps[start]
```

iii. The agent chose explicit timestamps rather than assuming perfectly uniform samples.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp of each trial is subtracted and the result is stored as float32.

ii.
```python
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. This makes every retained trial begin at exactly zero seconds.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses the identical `[start:stop)` slice as neural activity, followed by an explicit equality-of-length assertion.

ii.
```python
if neural.shape[1] != inputs.shape[1] or inputs.shape[1] != outputs.shape[1]:
    raise AssertionError("Neural/input/output time axes differ")
```

iii. Frame synchronization and independent spot checks are the stated justification.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived directly from the `environment` behavioral time series.

ii.
```python
environment = np.asarray(behavior["environment"].data[:], dtype=np.float32)
env_values = np.unique(environment[start:stop])
```

iii. Inspection showed values 0/1 corresponding to ENV1/ENV2 and constant within a trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Negative values are discarded for validation; exactly one valid value in `{0,1}` is required and repeated across the trial.

ii.
```python
env_values = env_values[env_values >= 0]
np.full(stop - start, env_values[0], dtype=np.float32)
```

iii. Strict validation ensures the requested binary per-trial input.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is read from the `trial number` behavior series at the trial start and checked against the zero-based raw trial index.

ii.
```python
trial_number = np.asarray(behavior["trial number"].data[:], dtype=np.float32)
source_trial_number = trial_number[start]
if not np.isclose(source_trial_number, raw_idx): raise ValueError(...)
```

iii. The agent reports that the native stream is zero-based and agrees with raw chronological trial order.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The validated scalar is repeated across all timepoints; numbering is not renumbered after quality exclusions.

ii.
```python
np.full(stop - start, source_trial_number, dtype=np.float32)
```

iii. Preserving raw chronology also preserves correct previous-trial semantics after an excluded trial.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It derives outcomes from sparse `Reward.timestamps`, behavior timestamps, and raw trial boundaries, then takes the preceding raw trial's outcome.

ii.
```python
outcomes = reward_outcomes(np.asarray(behavior["Reward"].timestamps[:]), timestamps, starts, stops)
previous_outcome = outcomes[raw_idx - 1] if raw_idx > 0 else 0
```

iii. Sparse timestamp search is justified as efficient and maintains chronology even when a corrupt trial is omitted from the converted data.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Reward events are counted by `searchsorted` within each half-open trial timestamp interval; the previous binary value is repeated across the current trial, with trial zero set to 0.

ii.
```python
left = np.searchsorted(reward_times, timestamps[starts], side="left")
right = np.searchsorted(reward_times, timestamps[stops], side="left")
return (right > left).astype(np.int8)
```

iii. This directly implements omitted=0/rewarded=1 and avoids expanding sparse rewards into a full frame array.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw `position` and a per-trial zone label parsed from `nwb.identifier` scene metadata. Fixed scenes end in `LocationA/B/C`; switch scenes use the left zone for the first 30 trials and the right zone thereafter.

ii.
```python
scene = nwb.identifier.rsplit("/", 1)[-1]
zone_labels_raw = parse_zone_schedule(scene, len(starts))
zone_start, zone_stop = ZONE_BOUNDS[zone_label]
```

iii. The agent states that this follows the paper's `get_reward_zones` scene mapping and known trial-30 switch, avoiding noisy `reward_zone` measurements.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is position minus the nearest zone edge: negative before, zero inside the closed interval, and positive after.

ii.
```python
signed_distance = np.where(
    trial_position < zone_start, trial_position - zone_start,
    np.where(trial_position > zone_stop, trial_position - zone_stop, 0.0))
```

iii. The definition matches the requested distance to any point in the active reward zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit masks implement seven requested categories, including an exact-zero class and inclusive 10/50-cm upper edges.

ii.
```python
out[distance < -50.0] = 0
out[(distance >= -50.0) & (distance < -10.0)] = 1
out[(distance >= -10.0) & (distance < 0.0)] = 2
out[distance == 0.0] = 3
out[(distance > 0.0) & (distance <= 10.0)] = 4
out[(distance > 10.0) & (distance <= 50.0)] = 5
```

iii. Synthetic equality-boundary assertions test the specified inequalities.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural events use the same frame indices and slice.

ii.
```python
trial_position = position[start:stop]
neural = np.concatenate([events[start:stop].T for events in plane_events], axis=0)
```

iii. Equal-axis assertions and raw-data spot checks support alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the raw `position` behavior series.

ii.
```python
position = np.asarray(behavior["position"].data[:], dtype=np.float32)
trial_position = position[start:stop]
```

iii. Position is already in centimeters along the 450-cm corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No transformation precedes categorical thresholding; the per-trial slice is passed to `position_classes`.

ii.
```python
position_classes(trial_position)
```

iii. The requested decoder needs categorical rather than continuous position.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Explicit masks form five 90-cm bins: `<90`, `[90,180)`, `[180,270)`, `[270,360]`, and `>360`.

ii.
```python
out[position < 90.0] = 0
out[(position >= 90.0) & (position < 180.0)] = 1
out[(position >= 180.0) & (position < 270.0)] = 2
out[(position >= 270.0) & (position <= 360.0)] = 3
```

iii. These are five equal-width bins over 450 cm and exact edges are unit-checked.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position and neural use identical trial boundaries and frame indices.

ii.
```python
trial_position = position[start:stop]
neural = np.concatenate([events[start:stop].T for events in plane_events], axis=0)
```

iii. The source streams are synchronized and converted dimensions are asserted equal.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the frame-synchronous `lick` behavior time series.

ii.
```python
lick = np.asarray(behavior["lick"].data[:], dtype=np.float32)
trial_lick = lick[start:stop]
```

iii. This is the release's direct lick measurement.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Any positive raw value becomes 1 and all other values become 0.

ii.
```python
(trial_lick > 0).astype(np.int8)
```

iii. This implements the requested binary no/yes output; heavily corrupted trials are filtered beforehand.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and events share the same `[start:stop)` frame slice.

ii.
```python
trial_lick = lick[start:stop]
```

iii. Source synchronization and length assertions are the justification.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Zone identity is derived from scene text in `nwb.identifier` plus raw trial index, not from the raw `reward_zone` time series.

ii.
```python
scene = nwb.identifier.rsplit("/", 1)[-1]
zone_labels_raw = parse_zone_schedule(scene, len(starts))
```

iii. The agent says this is the deterministic mapping used by the paper and is less noisy than inferring identity from sampled zone signals.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Fixed scenes map all trials to their suffix zone; switch scenes change after 30 trials. A/B/C map to 0/1/2 and are repeated over each trial.

ii.
```python
labels = [left_match.group(1)] * min(30, n_trials)
labels.extend([right_match.group(1)] * max(0, n_trials - 30))
np.full(stop - start, ZONE_TO_CLASS[zone_label], dtype=np.int8)
```

iii. The known experimental schedule motivates this deterministic assignment.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It comes from sparse timestamps in the raw `Reward` time series and trial timestamp boundaries.

ii.
```python
np.asarray(behavior["Reward"].timestamps[:])
outcomes = reward_outcomes(..., timestamps, starts, stops)
```

iii. These timestamps directly represent delivered rewards.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Sorted reward timestamps are searched within each trial interval; presence becomes 1, absence 0, and the scalar is repeated across trial frames.

ii.
```python
return (right > left).astype(np.int8)
np.full(stop - start, outcomes[raw_idx], dtype=np.int8)
```

iii. This implements the requested per-trial binary label efficiently.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Most inconsistencies cause explicit errors: unmatched boundaries, behavior lengths, invalid environments/trial numbers, nonfinite data, or unexpected neural lengths. A specifically audited one-extra-terminal-ophys-row defect in ten sessions is tolerated by truncating only that post-trial row. Corrupt lick trials are removed.

ii.
```python
extra_tail = n_ophys_frames - len(speed)
if extra_tail > 1:
    raise ValueError(...)
fluorescence = np.asarray(f_series.data[:len(speed), local_keep], dtype=np.float32)
```

iii. The first full run exposed the +1 issue; the agent audited all occurrences and confirmed that the extra row follows the final teleport. Strict failure elsewhere avoids silently masking unknown corruption.

## 13-a. What are the most time-consuming steps of the code?

i. Reading large fluorescence/neuropil arrays, trial-wise maximin/OASIS processing, retaining the large converted matrices, and serializing the 8.86-GiB pickle dominate. The full conversion took about 301 seconds; pickle writing took 8.43 seconds.

ii.
```python
dff, events, trace = trial_dff_and_events(fluorescence, neuropil, starts, stops, ...)
pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes estimate cost from neuron-by-frame workload and report measured per-session and full-run timings.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial loop for fluorescence preprocessing/deconvolution and output construction could partly be replaced by masked full-session operations, and corrupt-lick detection is a Python comprehension. Variable trial lengths and trial-specific baselines make complete vectorization awkward. Speed correlation is already vectorized over neurons.

ii.
```python
for trial_idx, (start, stop) in enumerate(zip(starts, stops)):
    ...
for raw_idx, (start, stop) in enumerate(zip(starts, stops)):
    ...
```

iii. The agent chose one-session/one-plane processing to limit memory and reports that vectorized neuron operations already kept runtime below the optimization threshold.

## 13-c. What processing does the code repeat multiple times?

i. Trial intervals are traversed repeatedly for dF/F, OASIS, speed-correlation sufficient statistics, corrupt-lick checks, and final trial assembly. Each plane independently reads and processes its fluorescence/neuropil data.

ii.
```python
for trial_idx, (start, stop) in enumerate(zip(starts, stops)): ...
for start, stop in zip(starts, stops): events[start:stop] = dcnv.oasis(...)
for start, stop in zip(starts, stops): ...  # speed correlations
```

iii. This repetition separates memory-heavy stages and preserves trial-local baselines; the agent favored bounded memory over fusing all passes.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Full-session dF/F is allocated only to filter interneurons and is then discarded; raw F/Fneu and plotting traces are also discarded after processing. When processing plots are requested, copies of position/speed and diagnostic figures do not enter the final pickle. The per-session conversion times and extensive metadata are not decoder features.

ii.
```python
dff, events, trace = trial_dff_and_events(...)
correlations = speed_correlations(dff, speed, starts, stops)
del fluorescence, neuropil, dff, events
```

iii. dF/F is necessary for the paper's speed-correlation filter even though only OASIS events are saved; diagnostics and metadata support validation rather than training.
