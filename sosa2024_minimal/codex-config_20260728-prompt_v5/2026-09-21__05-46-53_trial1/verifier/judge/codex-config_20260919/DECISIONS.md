# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively globbed every matching NWB session under `/app/data`, sorted the paths, and opened each with PyNWB. It first reopened every file to build the unique-subject list, then opened each again for conversion. Sessions with fewer than two retained trials would be omitted.

ii.
```python
session_paths = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))
with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
    nwb = io.read()
```

iii. The trajectory says the agent found 152 sessions across 11 mice and chose direct NWB conversion because imaging and behavior streams appeared frame-aligned. It validated the resulting pickle.

## 1-b. How are the data split into subjects?

i. Subject IDs come from `nwb.subject.subject_id`; first occurrence determines the index in `subjects`.

ii.
```python
subject = io.read().subject.subject_id
if subject not in subjects:
    subjects.append(subject)
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The agent inspected all NWBs and reported 11 unique mice. It preferred NWB metadata over parsing directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session and produces one element in each session-level list.

ii.
```python
for path in session_paths:
    session = convert_session(path, subject_to_idx)
    data["neural"].append(session["neural"])
```

iii. The trajectory explicitly treats the 152 NWB files as imaging sessions and preserves their session IDs in metadata.

## 1-d. How are the data split into trials?

i. Trial starts are positive samples of `trial_start`; ends are positive samples of `teleport`. Each slice is start-inclusive and teleport-exclusive.

ii.
```python
trial_start = np.flatnonzero(np.asarray(behavior["trial_start"].data[:]) > 0)
teleport = np.flatnonzero(np.asarray(behavior["teleport"].data[:]) > 0)
neural_trial = deconv[start:stop].T
```

iii. The agent checked that counts matched in all sessions and concluded the valid lap was `trial_start` inclusive through `teleport` exclusive.

## 1-e. How are trials filtered based on quality controls?

i. A trial is dropped if more than 35% of its raw frames have lick values greater than 2. Sessions with fewer than two remaining trials are dropped. No short-trial filter is used.

ii.
```python
if np.mean(trial_lick > 2) > LICK_ERROR_THRESHOLD:
    drop_trial[trial_idx] = True
if session["n_trials_kept"] < 2:
    continue
```

iii. The trajectory says the agent investigated the repository's lick-error rule and chose to drop 69 corrupted laps rather than fabricate lick labels.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from an ophys interface whose name contains `Deconvolved`, restricted using the ImageSegmentation `iscell` field. Raw fluorescence and neuropil are not used.

ii.
```python
deconv_name = next(name for name in ophys_interfaces.keys() if "Deconvolved" in name)
plane_data = np.asarray(rrs.data[:], dtype=np.float32)
parts.append(plane_data[:, keep_cells[roi_indices]])
```

iii. The agent believed the NWBs were already in a useful aligned form and that the paper decoder operated on deconvolved events, so it used the stored signal directly.

## 2-b. How is the `neural` data processed?

i. Curated stored deconvolved arrays are concatenated across sorted planes, sliced by trial, transposed to neuron-by-time, and summed within consecutive 16-frame bins. The agent does not recompute neuropil correction, dF/F, smoothing, or OASIS events.

ii.
```python
return np.concatenate(parts, axis=1), keep_cells
neural_binned[:, bin_idx] = neural_trial[:, sl].sum(axis=1, dtype=np.float32)
```

iii. The agent chose direct NWB conversion and later introduced 16-frame bins because native-resolution full-data decoder training was judged impractical.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only Suite2p ROIs with `iscell > 0` are retained. There is no speed-correlation filter for putative interneurons.

ii.
```python
keep = iscell[:, 0].astype(float) > 0
parts.append(plane_data[:, keep_cells[roi_indices]])
```

iii. The agent inspected the ROI table, saw that deconvolved matrices included non-cell ROIs, and selected `iscell[:,0] == 1` to match the repository's session objects.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples start at the `trial_start` index; the first aggregate bin begins there, so trials are aligned to trial start.

ii.
```python
neural_trial = deconv[start:stop].T
start = bin_idx * BIN_FRAMES
```

iii. The agent found behavior and deconvolved signals had equal frame counts and treated them as already aligned.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Sixteen native frames are aggregated per bin, giving metadata resolution about 1.0317 s. The final partial bin may contain fewer frames, although metadata still reports the fixed nominal size.

ii.
```python
BIN_FRAMES = 16
nbins = math.ceil(nframes / BIN_FRAMES)
"time_bin_size": float(BIN_FRAMES * 1000.0 * 0.06448362720402656)
```

iii. The agent explicitly traded temporal fidelity for tractable end-to-end decoder training at full dataset scale.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the timestamps attached to the raw `position` behavioral series.

ii.
```python
timestamps = np.asarray(behavior["position"].timestamps[:], dtype=np.float64)
```

iii. The agent regarded behavioral streams as sharing imaging-frame timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first trial timestamp is subtracted, and after rebinning the left-edge time of each 16-frame bin is retained.

ii.
```python
time_trial_s = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
time_binned[bin_idx] = time_trial_s[start]
```

iii. The agent changed an earlier bin-mean implementation to left edges so every aligned trial begins exactly at zero.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Time and neural activity use identical raw trial slices and identical 16-frame bin boundaries; each time value is the bin's left edge while neural is the sum over that bin.

ii.
```python
sl = slice(start, stop)
neural_binned[:, bin_idx] = neural_trial[:, sl].sum(axis=1)
time_binned[bin_idx] = time_trial_s[start]
```

iii. The agent relied on the NWB's frame alignment and kept all modalities in the same aggregation loop.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Primarily the raw `environment` behavior series; if no nonnegative sample exists, it falls back to the environment parsed from the NWB scene identifier.

ii.
```python
env = np.asarray(behavior["environment"].data[:], dtype=np.float32)
env_value = float(np.round(np.nanmedian(env_trial))) if len(env_trial) else float(parse_scene(scene)[0][-1] == "2")
```

iii. The trajectory says exact per-trial labels and robust handling of scene strings were checked before implementation.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Negative samples are discarded, the remaining trial median is rounded, and that scalar is repeated over all converted bins.

ii.
```python
env_trial = env_trial[env_trial >= 0]
np.full(ntime, env_value, dtype=np.float32)
```

iii. The agent treated environment as a per-trial context and added scene fallback for schema/data imperfections.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is primarily derived from the raw `trial number` behavior series, with the zero-based loop index as fallback.

ii.
```python
trial_number = np.asarray(behavior["trial number"].data[:], dtype=np.float32)
trialnum_value = ... if len(trialnum_trial) else float(trial_idx)
```

iii. The agent noted that the NWB already contained per-frame trial number and used it directly.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. Negative values are discarded, the median is rounded, and the resulting scalar is repeated across converted bins.

ii.
```python
trialnum_trial = trialnum_trial[trialnum_trial >= 0]
trialnum_value = float(np.round(np.nanmedian(trialnum_trial)))
```

iii. No specific rationale beyond robustly turning a nominally constant per-frame field into a trial scalar is recorded.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from raw `Reward` event timestamps, the position timestamps, and the preceding raw trial boundaries.

ii.
```python
reward_timestamps = np.asarray(behavior["Reward"].timestamps[:], dtype=np.float64)
prev_reward_outcomes[1:] = reward_outcomes[:-1]
```

iii. The agent investigated whether unrewarded laps were true omissions and used actual reward delivery to define outcome.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A raw trial is rewarded if any reward timestamp lies in its half-open time interval. Outcomes are shifted by one, with trial zero set to 0, then repeated across bins. Dropped trials do not alter the shift: “previous” means previous raw trial.

ii.
```python
reward_outcomes[trial_idx] = int(len(trial_reward_ts) > 0)
prev_reward_outcomes[1:] = reward_outcomes[:-1]
```

iii. This implements the binary omitted/rewarded input using observed reward events.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw position, raw `reward_zone`, and the scene identifier. Zone identity is the nearest canonical zone to the median position where `reward_zone > 0`, or a scene/switch-trial fallback.

ii.
```python
rz_mask = reward_zone[start:stop] > 0
zone_labels.append(zone_from_position(float(np.nanmedian(position[start:stop][rz_mask]))))
```

iii. The agent chose observed zone detection plus scene fallback after checking unique scenes and repository reward-zone logic.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position is averaged within each 16-frame bin. Signed distance is zero inside the assigned zone, position minus the start before it, and position minus the end after it; it is then categorized.

ii.
```python
pos_binned[bin_idx] = np.mean(pos_trial_cm[sl], dtype=np.float64)
dist[before] = position_cm[before] - start_cm
dist[after] = position_cm[after] - end_cm
```

iii. The agent used canonical A/B/C bounds and kept distance relative to the nearest zone edge.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It uses seven explicit categories at -50, -10, 0, 10, and 50 cm, with exactly zero isolated.

ii.
```python
out[distance_cm < -50.0] = 0
out[(distance_cm >= -10.0) & (distance_cm < 0.0)] = 2
out[distance_cm == 0.0] = 3
out[distance_cm > 50.0] = 6
```

iii. These thresholds were taken directly from the decoder instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural activity share trial slices and bin boundaries, but distance is computed from mean position while neural events are summed.

ii.
```python
neural_binned[:, bin_idx] = neural_trial[:, sl].sum(axis=1)
pos_binned[bin_idx] = np.mean(pos_trial_cm[sl])
```

iii. The agent stated all NWB modalities were already frame-aligned and aggregated them together.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from the raw `position` behavior time series.

ii.
```python
position = np.asarray(behavior["position"].data[:], dtype=np.float32)
```

iii. Position was one of the frame-aligned streams the agent identified in the NWB.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Raw position is averaged within each 16-frame bin and then discretized.

ii.
```python
pos_binned[bin_idx] = np.mean(pos_trial_cm[sl], dtype=np.float64)
"position_bin": discretize_position(pos_binned)
```

iii. Aggregation was introduced to make full-data decoder training tractable.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five bins are assigned using thresholds 90, 180, 270, and 360 cm.

ii.
```python
out[(position_cm >= 90.0) & (position_cm < 180.0)] = 1
out[(position_cm >= 270.0) & (position_cm < 360.0)] = 3
out[position_cm >= 360.0] = 4
```

iii. The agent used five equal 90 cm divisions of the 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position averages and neural sums use the same raw frame slice for each 16-frame bin.

ii.
```python
sl = slice(start, stop)
neural_binned[:, bin_idx] = neural_trial[:, sl].sum(axis=1)
pos_binned[bin_idx] = np.mean(pos_trial_cm[sl])
```

iii. The agent relied on existing frame alignment and joint aggregation.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the raw `lick` behavior time series.

ii.
```python
lick = np.asarray(behavior["lick"].data[:], dtype=np.float32)
```

iii. The agent identified lick as an imaging-rate-aligned NWB stream.

## 9-b. What processing is involved in computing `output` *Lick*?

i. A converted bin is 1 if any raw lick value in its 16 frames is positive; otherwise 0. Trials with pervasive values above 2 are removed first.

ii.
```python
lick_binned[bin_idx] = int(np.any(lick_trial[sl] > 0))
if np.mean(trial_lick > 2) > LICK_ERROR_THRESHOLD:
    drop_trial[trial_idx] = True
```

iii. The binary rule follows the task; dropping heavily corrupted trials was chosen instead of inventing labels.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick presence and neural event sums are calculated over the same 16-frame slices.

ii.
```python
neural_binned[:, bin_idx] = neural_trial[:, sl].sum(axis=1)
lick_binned[bin_idx] = int(np.any(lick_trial[sl] > 0))
```

iii. Joint binning preserves coarse bin-level alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from raw `reward_zone` and position, with the scene identifier as fallback.

ii.
```python
rz_mask = reward_zone[start:stop] > 0
zone_labels.append(zone_from_position(float(np.nanmedian(position[start:stop][rz_mask]))))
```

iii. The agent aimed to use observed active-zone samples while supporting omissions through scene parsing.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The median active-zone position is assigned to the nearest canonical zone center. If absent, scene strings prescribe the zone, with transitions assumed at trial 30. A/B/C become 0/1/2 and are repeated over bins.

ii.
```python
split = min(SWITCH_TRIAL, ntrials)
reward_zone_value = {"A": 0, "B": 1, "C": 2}[zone_labels[trial_idx]]
```

iii. The agent enumerated unique scenes and added a deterministic fallback before conversion.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward` event timestamps and position-series timestamps defining trial intervals.

ii.
```python
reward_timestamps = np.asarray(behavior["Reward"].timestamps[:], dtype=np.float64)
trial_reward_ts = reward_timestamps[(reward_timestamps >= timestamps[start]) & (reward_timestamps < timestamps[stop])]
```

iii. The agent used delivered reward rather than zone activity after examining omission semantics.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Outcome is 1 if at least one event timestamp is in the trial, otherwise 0, and is repeated over all converted bins.

ii.
```python
reward_outcomes[trial_idx] = int(len(trial_reward_ts) > 0)
np.full(ntime, reward_outcome_value, dtype=np.int16)
```

iii. This directly implements the requested per-trial binary outcome.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code raises on unmatched trial starts/teleports, missing deconvolved series, or ROI-column mismatch. It falls back to scene labels when reward-zone samples are missing and to scene/loop values when environment/trial-number samples are invalid. It handles multipane ROI mappings, drops lick-corrupted trials, and drops sessions with fewer than two trials. It does not crop neural/behavior length mismatches, filter short trials, or validate timestamps.

ii.
```python
if len(trial_start) != len(teleport):
    raise ValueError(...)
zone_labels.append(expected_zone_labels[trial_idx])
```

iii. The trajectory records a multipane schema fix and describes the fallbacks as robustness to small schema differences.

## 13-a. What are the most time-consuming steps of the code?

i. Reading all 152 large NWBs is dominant; every file is opened once to identify subjects and again to convert. Conversion also streams neural arrays, loops over trials/bins, and serializes the full pickle.

ii.
```python
for path in session_paths:
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        subject = io.read().subject.subject_id
for path in session_paths:
    session = convert_session(path, subject_to_idx)
```

iii. During execution the agent explicitly reported that most conversion time was spent streaming 152 NWBs and building trial lists.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-bin loop in `bin_trial` could be vectorized for complete 16-frame blocks via reshape/reductions, leaving only the partial tail. Reward timestamp assignment could use `searchsorted`; per-trial reward/zone/error summaries could also be grouped more efficiently. Variable trial lengths still make the outer trial loop reasonable.

ii.
```python
for bin_idx in range(nbins):
    ...
for trial_idx, (start, stop) in enumerate(zip(trial_start, teleport)):
    ...
```

iii. The agent did not discuss vectorization; it prioritized correctness, multipane robustness, and training tractability.

## 13-c. What processing does the code repeat multiple times?

i. Every NWB is opened twice. Each trial is also traversed once for reward/zone/error labels and again for conversion. `parse_scene(scene)` can be repeated for fallback environment values.

ii.
```python
for trial_idx, (start, stop) in enumerate(zip(trial_start, teleport)):
    ... # labels
for trial_idx, (start, stop) in enumerate(zip(trial_start, teleport)):
    ... # conversion
```

iii. No explicit justification was given; the two-pass trial structure lets previous outcomes and drop flags be computed before output construction.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `expected_trial_labels` builds `env_by_trial`, but the caller discards it. `TRACK_LENGTH_CM` is unused. Raw `speed` is loaded, averaged, and discretized because speed is an output, so it is not discarded. Metadata counts and scene parsing are retained in `session_info`.

ii.
```python
env_by_trial = [envs[0]] * ntrials
_, expected_zone_labels = expected_trial_labels(scene, ntrials)
TRACK_LENGTH_CM = 450.0
```

iii. The trajectory gives no justification for the unused environment-label list or constant; they are minor implementation leftovers.
