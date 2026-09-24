# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The converter recursively glob-sorts every session NWB under `data/sub-*`, opens each with `h5py`, reads the behavior arrays and plane-wise ophys arrays, and processes every file unless `--sample` limits the run to two files.

ii.
```python
files = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))
with h5py.File(path, "r") as f:
    behavior_group = f["processing/behavior/BehavioralTimeSeries"]
    ophys_group = f["processing/ophys"]
```

iii. The notes justify direct HDF5 access as an efficient way to read all 152 released NWB sessions and describe the release as already frame-aligned processed data.

## 1-b. How are the data split into subjects?

i. Subject identity is parsed from each file's parent directory. Unique IDs are sorted, and each processed session receives the corresponding `subject_idx`.

ii.
```python
subject = path.parent.name.replace("sub-", "")
subjects = sorted({sess["subject"] for sess in processed_sessions})
subject_idx = np.asarray([subject_to_idx[sess["subject"]] for sess in processed_sessions])
```

iii. The agent observed 11 `sub-<mouse>` directories and reports that this matches the paper's 11-mouse switch cohort.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session; path sorting gives stable subject/session ordering. Sessions with fewer than two retained trials are skipped.

ii.
```python
for idx, path in enumerate(nwb_files, start=1):
    session, examples = process_session(path)
    if len(session["neural"]) < 2:
        continue
    processed_sessions.append(session)
```

iii. The notes identify one file per imaging day and reconcile 152 files with 14 days for ten mice plus 12 days for m11.

## 1-d. How are the data split into trials?

i. Positive `trial_start` samples are paired in order with the next positive `teleport` sample; the interval is start-inclusive and teleport-exclusive. Incomplete leading/trailing trials are omitted.

ii.
```python
starts = np.flatnonzero(trial_start > 0)
teleports = np.flatnonzero(teleport > 0)
...
bounds.append((int(start), int(stop)))
trial_slice = slice(start, stop)
```

iii. The agent says these continuous markers replace a missing NWB trials table and match the reference trial-start/teleport epoch semantics.

## 1-e. How are trials filtered based on quality controls?

i. Trials shorter than five frames are dropped; trials with lick-sensor artifacts (`>35%` of frames having lick count `>2`) are dropped entirely; after removing invalid behavior/neural frames, trials with fewer than five frames are dropped. Sessions must retain at least two trials.

ii.
```python
MIN_TRIAL_FRAMES = 5
if stop - start < MIN_TRIAL_FRAMES: continue
if meta["lick_error"]: continue
frame_mask = valid_behavior_frames[start:stop] & valid_neural_frames[start:stop]
if np.count_nonzero(frame_mask) < MIN_TRIAL_FRAMES: continue
```

iii. The notes prefer the released code's 35% lick threshold over the Methods' 30%, drop bad-lick trials to avoid NaNs, and retain only complete/usable trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes directly from each plane of `processing/ophys/Deconvolved`, restricted using `iscell`; `Fluorescence` and `Neuropil` are not used.

ii.
```python
accepted_mask = np.asarray(iscell[:, 0]) == 1
plane_data = ophys_group[f"Deconvolved/plane{plane}/data"][:, accepted_local_idx]
```

iii. The agent asserts that released NWB `Deconvolved` is the equivalent of the paper's `sess.timeseries['events']`, so recomputing dF/F would introduce divergence.

## 2-b. How is the `neural` data processed?

i. Accepted plane matrices are assembled into pooled ROI order, cast to `float32`, sliced with the valid-frame mask, and transposed from time-by-ROI to neuron-by-time. No dF/F, smoothing, baseline estimation, or new deconvolution is performed.

ii.
```python
deconv[:, dest_cols] = plane_data
neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. The agent considers the stored signal already processed and frame-aligned, and notes that pooling planes matches the paper's main analyses.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs are retained only when `iscell[:,0] == 1`; frames must be finite across every retained neuron. The code does not exclude cells whose dF/F correlates with speed above 0.5.

ii.
```python
accepted_mask = np.asarray(iscell[:, 0]) == 1
valid_neural_frames = np.all(np.isfinite(deconv), axis=1)
```

iii. The notes recognize the paper's additional interneuron criterion but choose only the explicit curated mask available in the NWB.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples begin at the `trial_start` index and end before teleport; invalid frames are removed jointly from neural and behavioral streams.

ii.
```python
trial_slice = slice(start, stop)
neural_trial = deconv[trial_slice][frame_mask].T
```

iii. The agent states that the NWB streams already share a sample grid, so trial slicing supplies trial-start alignment without resampling.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The original approximately 64.5 ms frame grid is retained; metadata uses the median timestamp difference across sessions.

ii.
```python
"time_bin_size_ms": float(np.median(np.diff(timestamps)) * 1000.0)
median_bin_ms = float(np.median([...]))
```

iii. The notes use behavior timestamps rather than misleading 31 Hz multi-plane rate attributes and report an effective ~15.5 Hz grid.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/position/timestamps` and trial-start indices.

ii.
```python
timestamps = behavior_group["position/timestamps"][()].astype(np.float64)
```

iii. The agent regards position timestamps as the reliable shared time base, especially for multi-plane sessions.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at the original trial-start frame is subtracted from every retained timestamp in the trial and the result is cast to `float32`.

ii.
```python
time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
time_trial.astype(np.float32)
```

iii. This directly expresses elapsed seconds relative to the required alignment event.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The same `slice(start, stop)` and `frame_mask` are applied to timestamps and neural rows before transposition.

ii.
```python
time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
neural_trial = deconv[trial_slice][frame_mask].T
```

iii. The agent validated raw-to-converted time vectors and relies on the NWB's shared frame-aligned sample grid.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from the behavior `environment/data` series.

ii.
```python
environment = behavior_group["environment/data"][()].astype(np.float32)
env_bin = env_to_binary(environment[start:stop])
```

iii. The notes map this series to the paper's binary morph/environment identity.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Nonfinite and negative sentinel values are removed, the median is rounded into a binary value, and that per-trial value is repeated over all retained frames.

ii.
```python
values = values[np.isfinite(values)]
values = values[values >= 0]
return int(np.round(np.median(values)) > 0.5)
```

iii. The agent chose a robust per-trial summary because environment should be constant within a trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the behavior `trial number/data` series within each marker-defined trial.

ii.
```python
trial_number = behavior_group["trial number/data"][()].astype(np.float32)
trial_num = modal_trial_number(trial_number[start:stop])
```

iii. The notes say the raw per-trial integer is used to preserve actual task numbering and switch semantics.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. Values are filtered to finite nonnegative integers, rounded, reduced to the modal value, and repeated across trial frames.

ii.
```python
values = np.rint(values).astype(np.int64)
counts = np.bincount(values)
trial_num = int(np.argmax(counts))
```

iii. The mode is intended to tolerate sentinel or noisy samples while yielding one stable trial label.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is indirectly derived from sparse `Reward/timestamps`, trial timestamps, and `reward_zone/data`, through the computed outcome for trial number minus one.

ii.
```python
reward_by_trial_number[trial_num] = reward_outcome
prev_outcome = reward_by_trial_number.get(trial_num - 1, 0)
```

iii. The agent maps reward delivery plus zone entry to the paper/reference reward semantics and uses zero when no preceding within-session trial exists.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Outcomes are first computed for all complete trials. For each retained trial, the dictionary entry keyed by `trial_num - 1` is read, defaulting to zero, then repeated over time.

ii.
```python
prev_outcome = reward_by_trial_number.get(trial_num - 1, 0)
np.full(time_trial.shape, prev_outcome, dtype=np.float32)
```

iii. Precomputing outcomes ensures a dropped lick-artifact trial can still supply the following trial's history.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses behavior `position/data` and trial-specific A/B/C zone bounds inferred from the NWB identifier's scene string and a trial-30 switch rule.

ii.
```python
scene = identifier.split("/")[-1]
zone_label = zone_for_trial(scene_info, trial_num)
zone_coords = ZONE_TO_COORDS_CM[zone_label]
```

iii. The agent argues that `reward_zone` is an entry-event series rather than full spatial extent, while scene metadata encodes the actual condition and known A/B/C bounds.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position minus the near zone edge is used before the zone, zero inside it, and position minus the far edge after it.

ii.
```python
distance[before] = position_cm[before] - zone_start
distance[after] = position_cm[after] - zone_end
```

iii. This implements signed distance to any location within the reward zone, consistent with the requested reward-relative coordinate.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit Boolean masks create the seven requested classes at -50, -10, 0, 10, and 50 cm, with exact zero isolated.

ii.
```python
out[distance_cm < -50] = 0
out[(distance_cm >= -10) & (distance_cm < 0)] = 2
out[distance_cm == 0] = 3
out[distance_cm > 50] = 6
```

iii. The agent follows the decoder specification exactly and raises if any sample remains unclassified.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position is selected using the identical trial slice and valid-frame mask as neural data before distance calculation.

ii.
```python
position_trial = position[trial_slice][frame_mask]
distance_trial = signed_distance_to_zone(position_trial, zone_start, zone_end)
```

iii. Shared frame indexing is the asserted alignment mechanism.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It comes directly from behavior `position/data`.

ii.
```python
position = behavior_group["position/data"][()].astype(np.float32)
position_trial = position[trial_slice][frame_mask]
```

iii. The notes identify this as the released frame-aligned corridor position.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Positions are clipped to `[0, 450)` and divided by 90 cm, then floored to an integer class.

ii.
```python
clipped = np.clip(position_cm, 0.0, np.nextafter(450.0, 0.0))
bins = np.floor(clipped / 90.0).astype(np.int16)
```

iii. Clipping prevents sentinel/out-of-track values and guarantees one of five classes.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal-width 90 cm bins are used: class 0 for `[0,90)`, through class 4 for `[360,450]` after clipping.

ii.
```python
bins = np.floor(clipped / 90.0).astype(np.int16)
bins[bins > 4] = 4
```

iii. This directly implements the requested five equal-sized track bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position and neural use the same trial slice and valid-frame mask.

ii.
```python
position_trial = position[trial_slice][frame_mask]
neural_trial = deconv[trial_slice][frame_mask].T
```

iii. The agent reports raw-to-converted spot checks passing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from behavior `lick/data`.

ii.
```python
lick = behavior_group["lick/data"][()].astype(np.float32)
lick_trial = lick[trial_slice][frame_mask]
```

iii. The notes describe this as cumulative lick count per imaging frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Trials first undergo the 35% lick-sensor-error rule; retained lick samples are binarized as `lick > 0`.

ii.
```python
return bool(np.mean(lick_segment > 2) > LICK_ERROR_FRACTION)
(lick_trial > 0).astype(np.int16)
```

iii. Binarization matches the requested no/yes output; the agent adopted the released code's 35% artifact criterion.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural samples share the same trial slice and frame mask.

ii.
```python
lick_trial = lick[trial_slice][frame_mask]
neural_trial = deconv[trial_slice][frame_mask].T
```

iii. The agent relies on the already aligned NWB time series.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It comes from the NWB `identifier` scene name plus modal raw trial number; the scene encodes initial/final zones and environments.

ii.
```python
scene_info = parse_scene(scene)
zone_label = zone_for_trial(scene_info, trial_num)
```

iii. The agent says scene metadata is more semantically appropriate than sparse `reward_zone` entries for assigning A/B/C.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Regexes parse single, within-environment switch, and cross-environment switch scenes. On switch sessions, trial numbers below 30 use the first zone and numbers 30+ use the second; A/B/C map to 0/1/2 and are repeated over time.

ii.
```python
if scene_info.has_switch and trial_number >= SWITCH_TRIAL:
    return scene_info.after_zone
np.full(time_trial.shape, ZONE_TO_CODE[meta["zone_label"]], dtype=np.int16)
```

iii. The paper says switches occurred after 30 trials, and counterbalanced scene strings encode both conditions.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It uses sparse behavior `Reward/timestamps`, position timestamps defining the trial interval, and `reward_zone/data` to require a zone-entry event.

ii.
```python
reward_timestamps = behavior_group["Reward/timestamps"][()].astype(np.float64)
reward_zone = behavior_group["reward_zone/data"][()].astype(np.float32)
```

iii. The agent says this matches `behavior.get_trial_types` semantics rather than labeling from reward timestamps alone.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. `searchsorted` checks for at least one reward timestamp in `[t_start,t_stop)`, this is ANDed with any positive reward-zone sample, and the binary result is repeated across frames.

ii.
```python
has_reward = right > left
has_rzone_entry = np.any(reward_zone_segment > 0)
return int(has_reward and has_rzone_entry)
```

iii. The notes report an 84.2% reward rate, close to the paper's ~85%, and raw spot checks of rewarded/omitted trials.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Only complete marker-paired trials are kept. Sentinel/invalid samples are identified using finiteness, position `>-100`, and finite neural activity, then removed synchronously across all streams. Trials with fewer than five remaining frames are dropped. Invalid environment/trial-number values are ignored when finding robust per-trial labels.

ii.
```python
valid_behavior_frames = np.isfinite(position) & ... & (position > -100.0)
valid_neural_frames = np.all(np.isfinite(deconv), axis=1)
frame_mask = valid_behavior_frames[start:stop] & valid_neural_frames[start:stop]
```

iii. The agent chose not to fabricate incomplete boundaries or retain NaNs because the validator forbids NaNs; the clipped first m11 day-3 trial is explicitly omitted.

## 13-a. What are the most time-consuming steps of the code?

i. Reading/assembling large plane-wise neural matrices for all 152 NWBs and serializing the approximately 9 GB pickle dominate. Optional plotting and decoder training are separate costs.

ii.
```python
plane_data = ophys_group[f"Deconvolved/plane{plane}/data"][:, accepted_local_idx]
pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes report 3.36 minutes for conversion and emphasize the size of the neural arrays/output file.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The loops pairing starts/teleports, iterating trial metadata twice, and assembling planes could potentially be partially vectorized. Variable trial lengths make the per-trial construction naturally list-based.

ii.
```python
for start in starts: ...
for start, stop in trial_bounds: ...
for meta in trial_meta: ...
for plane in planes: ...
```

iii. The agent did not explicitly discuss vectorization, but the implementation already vectorizes calculations within each trial.

## 13-c. What processing does the code repeat multiple times?

i. It makes two passes over every session's trials: first to compute metadata/outcomes, then to filter and build arrays. It also computes common masks and fills constant per-trial vectors repeatedly; optional plots retain a few copied examples.

ii.
```python
for start, stop in trial_bounds:
    trial_meta.append(...)
for meta in trial_meta:
    neural_trials.append(...)
```

iii. The first pass is intentional so previous outcomes remain available even when an intervening trial is later dropped.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It builds up to three rich diagnostic example dictionaries per session even when `--show-processing` is false, returns them redundantly, and computes session summaries/ETA metadata not used by decoder training. It also calculates optional plotting fields such as reward-time offsets.

ii.
```python
if len(kept_examples) < 3:
    kept_examples.append({...})
return ({..., "examples": kept_examples}, kept_examples)
```

iii. These objects support diagnostics and documentation, but the decoder consumes only the saved dataset fields; plotting occurs only when explicitly requested.
