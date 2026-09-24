# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively enumerates every matching NWB file under `/app/data/sub-*`, sorts the paths, and processes one file (one session) at a time. It uses `h5py` to read behavior, reward events, ROI metadata, and every deconvolved imaging-plane series. Full mode is the default; sample mode selects two files.

ii.
```python
files = sorted(Path("/app/data").glob("sub-*/sub-*_behavior+ophys.nwb"))
for session_i, path in enumerate(files):
    session_raw = load_session_raw(path)
```
```python
with h5py.File(path, "r") as f:
    ...
    plane_series = sorted(f[f"{OPHYS_TS_PATH}/Deconvolved"].keys())
```

iii. The notes say direct HDF5 access was chosen for speed and lower overhead, while sequential per-session processing avoids loading the whole dataset into memory. The full run found 152 sessions and 11 subjects.

## 1-b. How are the data split into subjects?

i. The subject ID is read from each NWB file. A first-seen lookup builds `subjects`, and `subject_idx` maps each retained session to that list.

ii.
```python
subject = read_str(f["general/subject/subject_id"])
...
if subject not in subject_lookup:
    subject_lookup[subject] = len(subject_lookup)
    data["subjects"].append(subject)
subject_idx.append(subject_lookup[subject])
```

iii. The agent justified this using authoritative NWB session metadata and verified the resulting 11 subjects against the paper/data bundle.

## 1-c. How are the data split into sessions?

i. Each NWB file is one output session. Sessions with fewer than two retained trials are skipped; otherwise their trial lists are appended together and their subject index and neuron-region index are appended in the same order.

ii.
```python
neural_trials, input_trials, output_trials, session_info, plot_payload = make_trial_arrays(session_raw, ...)
if len(neural_trials) < 2:
    continue
data["neural"].append(neural_trials)
data["input"].append(input_trials)
data["output"].append(output_trials)
```

iii. The notes identify one NWB per subject-session and report that all 152 sessions survived conversion.

## 1-d. How are the data split into trials?

i. Trial starts are positive `trial_start` samples. For each start, the code pairs the next positive `teleport` sample and uses the half-open slice `[start, stop)`. Unpaired trailing fragments are ignored.

ii.
```python
starts = np.flatnonzero(behavior["trial_start"] > 0)
teleports = np.flatnonzero(behavior["teleport"] > 0)
...
bounds.append((int(start), int(stop)))
...
pos = behavior["position"][start:stop]
neural_trial = neural[start:stop].T
```

iii. The agent says this matches the reference trial-start/teleport structure, excludes tunnel/teleport samples, and safely ignores a known trailing label-only fragment.

## 1-e. How are trials filtered based on quality controls?

i. Empty trials and trials where more than 35% of samples have raw lick values greater than 2 are removed. Sessions with fewer than two remaining trials are removed. There is no minimum-duration filter.

ii.
```python
if pos.size == 0:
    dropped["empty"] += 1
    continue
bad_lick = np.mean(lick_raw > 2.0) > LICK_QC_FRACTION
if bad_lick:
    dropped["lick_qc"] += 1
    continue
```

iii. The notes connect the lick rule to the reference decoder’s sensor-error heuristic because lick is a target, and report 69 excluded trials. The two-trial check enforces the output-format requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes directly from all response series under NWB `processing/ophys/Deconvolved`, restricted by the Suite2p `iscell` flag. It is not recomputed from `Fluorescence` and `Neuropil`.

ii.
```python
plane_series = sorted(f[f"{OPHYS_TS_PATH}/Deconvolved"].keys())
response_iscell = iscell[rois, 0].astype(np.int64) == 1
response_data = response_group["data"][:common_len, :]
neural_planes.append(response_data[:, response_iscell].astype(np.float32, copy=False))
```

iii. The agent believed the stored deconvolved array matched the paper decoder’s event signal and argued that recomputation would add avoidable mismatch.

## 2-b. How is the `neural` data processed?

i. Each plane is cropped to a common neural/behavior length, manually curated ROIs are retained, planes are concatenated along the neuron axis, and each trial slice is transposed to neuron-by-time `float32`. No dF/F baseline computation, neuropil subtraction, smoothing, or new OASIS deconvolution is performed.

ii.
```python
common_len = min(neural_len, behavior_len)
neural = np.concatenate(neural_planes, axis=1)
...
neural_trial = neural[start:stop].T.astype(np.float32, copy=False)
```

iii. The notes say direct use preserves the paper’s decoder signal, handles multi-plane files, and reduces computation and I/O.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs with `iscell[:, 0] == 1` are retained, applied through each response series’ ROI references. The code does not remove putative interneurons based on correlation with running speed.

ii.
```python
response_plane_idx = plane_idx_all[rois]
response_iscell = iscell[rois, 0].astype(np.int64) == 1
neural_planes.append(response_data[:, response_iscell])
```

iii. The agent identified `iscell` as upstream manual Suite2p curation and deliberately chose all curated neurons for the broader user-specified decoder.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is by slicing neural samples from the same `trial_start` index used for behavior, so each trial begins at time zero and ends immediately before teleport.

ii.
```python
for raw_trial_idx, (start, stop) in enumerate(bounds):
    neural_trial = neural[start:stop].T.astype(np.float32, copy=False)
```

iii. The notes state that NWB neural and behavior arrays are already frame-aligned; raw-vs-converted spot checks matched exactly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The native behavior timestamp grid is preserved, with the dataset metadata set to the median session interval, approximately 64.48 ms (~15.51 Hz).

ii.
```python
dt_behavior = float(np.median(np.diff(behavior["timestamps"][: min(common_len, 1000)])))
...
"time_bin_size": dt_sec * 1000.0,
```

iii. The agent used behavior timestamps because the 31 Hz attribute in multi-plane sessions is a scanner rate rather than the effective per-plane sample rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the timestamps attached to the raw `position` behavior series.

ii.
```python
behavior["timestamps"] = f[f"{BEHAVIOR_TS_PATH}/position/timestamps"][:]
time_raw = behavior["timestamps"][start:stop]
```

iii. The agent treats these as the reliable aligned behavior-frame timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp in each trial slice is subtracted from every timestamp and the result is stored as `float32`.

ii.
```python
time_from_start = (time_raw - time_raw[0]).astype(np.float32, copy=False)
```

iii. This directly implements elapsed time from the requested alignment event.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The timestamp and neural arrays are first cropped to a common session length and then sliced with identical `[start:stop]` bounds, yielding one time value per neural column.

ii.
```python
common_len = min(neural_len, behavior_len)
...
time_raw = behavior["timestamps"][start:stop]
neural_trial = neural[start:stop].T
```

iii. The agent reports exact spot-check agreement and notes that ten one-sample mismatches were resolved by common-length trimming.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from the NWB behavior `environment` series within each trial.

ii.
```python
behavior["environment"] = f[f"{BEHAVIOR_TS_PATH}/environment/data"][:]
env_raw = behavior["environment"][start:stop]
```

iii. The notes identify raw values 0 and 1 with ENV1 and ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Negative values are discarded, the modal remaining value is selected, and that scalar is repeated across all trial timepoints.

ii.
```python
valid = environment_slice[environment_slice >= 0]
values, counts = np.unique(valid, return_counts=True)
env_value = float(values[np.argmax(counts)])
...
np.full(time_from_start.shape, env_value, dtype=np.float32)
```

iii. The agent considered environment a per-trial context and repetition ensures consistent feature-by-time matrices.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is read from the raw behavior `trial number` value at the trial-start sample.

ii.
```python
trial_label = int(behavior["trial number"][start])
```

iii. The notes call this the actual per-session 0-based trial index and distinguish complete trials from label-only tunnel fragments.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The scalar raw label is converted to float and repeated over the trial’s time axis.

ii.
```python
np.full(time_from_start.shape, float(trial_label), dtype=np.float32)
```

iii. The agent repeats all per-trial variables to keep a uniform `(features, timepoints)` representation.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from sparse NWB `Reward` timestamps, aligned to position timestamps, and from the preceding raw trial bounds.

ii.
```python
reward_timestamps = f[f"{BEHAVIOR_TS_PATH}/Reward/timestamps"][:]
reward_frame_idx = np.searchsorted(behavior["timestamps"], reward_timestamps, side="left")
previous_outcomes[1:] = reward_outcomes_raw[:-1]
```

iii. The notes say this reconstructs rewarded versus omission trials using the paper’s sparse reward events.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A trial is rewarded if any mapped reward index lies in `[start, stop)`. The resulting raw-trial outcome vector is shifted by one; the first trial is zero. The value is repeated across the current trial, even if an intervening trial is later removed by lick QC.

ii.
```python
has_reward = np.any((reward_frame_idx >= start) & (reward_frame_idx < stop))
outcomes[i] = int(has_reward)
...
previous_outcomes = np.zeros(len(bounds), dtype=np.int64)
previous_outcomes[1:] = reward_outcomes_raw[:-1]
```

iii. The agent followed the requested binary omitted/rewarded definition and correctly interpreted “previous” as the preceding raw experimental trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw `position` plus a reward-zone label parsed from the NWB identifier/scene. Fixed intervals are A=80–130, B=200–250, and C=320–370 cm; switch scenes change label at trial index 30.

ii.
```python
scene = read_str(f["identifier"]).split("/")[-1]
reward_labels = reward_labels_for_trials(session_raw["scene"], len(bounds))
zone_start, zone_end = REWARD_ZONE_COORDS[zone_label]
distance = signed_distance_to_zone(pos, zone_start, zone_end)
```

iii. The agent says the framewise `reward_zone` stream represents occupancy rather than identity, whereas scene metadata and reference behavior code encode the intended A/B/C sequence.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Distance is zero inside the interval, position minus zone start before it, and position minus zone end after it; this continuous result is then discretized.

ii.
```python
dist = np.zeros_like(position_cm, dtype=np.float32)
dist[before] = position_cm[before] - zone_start
dist[after] = position_cm[after] - zone_end
```

iii. This implements signed distance to the nearest location in the current reward zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven explicit masks implement the requested boundaries. Notably +10 belongs to class 5, while the task text says class 4 is “+10 cm” and class 5 begins at “+10 cm,” an ambiguous shared boundary.

ii.
```python
out[distance_cm < -50.0] = 0
out[(distance_cm >= -50.0) & (distance_cm < -10.0)] = 1
out[(distance_cm >= -10.0) & (distance_cm < 0.0)] = 2
out[distance_cm == 0.0] = 3
out[(distance_cm > 0.0) & (distance_cm < 10.0)] = 4
out[(distance_cm >= 10.0) & (distance_cm <= 50.0)] = 5
out[distance_cm > 50.0] = 6
```

iii. The agent’s intent was to reproduce the seven task-specified classes exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural activity use the same common-length frame grid and identical trial bounds.

ii.
```python
pos = behavior["position"][start:stop]
neural_trial = neural[start:stop].T
```

iii. The notes report independent raw-data spot checks of output alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It comes directly from NWB behavior `position`.

ii.
```python
pos = behavior["position"][start:stop].astype(np.float32, copy=False)
```

iii. The raw stream is already corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The trial slice is discretized directly; no interpolation or smoothing is applied.

ii.
```python
discretize_absolute_position(pos)
```

iii. The agent preserves the aligned framewise samples and applies only required categorical conversion.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five 90-cm bins are implemented. Exactly 360 cm is assigned class 3; negative values remain class 0.

ii.
```python
out[(position_cm >= 90.0) & (position_cm < 180.0)] = 1
out[(position_cm >= 180.0) & (position_cm < 270.0)] = 2
out[(position_cm >= 270.0) & (position_cm <= 360.0)] = 3
out[position_cm > 360.0] = 4
```

iii. This follows the task’s five equal divisions of the 450-cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The same `[start:stop]` indices are applied to the position and neural matrices after common-length cropping.

ii.
```python
pos = behavior["position"][start:stop]
neural_trial = neural[start:stop].T
```

iii. The agent validated converted slices against the corresponding raw NWB samples.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the framewise NWB behavior `lick` series.

ii.
```python
lick_raw = behavior["lick"][start:stop].astype(np.float32, copy=False)
```

iii. The notes identify this as the aligned raw lick count/sensor stream.

## 9-b. What processing is involved in computing `output` *Lick*?

i. After trial-level artifact filtering, all positive raw values are mapped to 1 and all others to 0.

ii.
```python
bad_lick = np.mean(lick_raw > 2.0) > LICK_QC_FRACTION
...
lick_binary = (lick_raw > 0.0).astype(np.int64, copy=False)
```

iii. The requested output is binary, and the notes say invalid sensor-heavy trials should not be treated as valid lick targets.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural activity are cropped to the same session extent and sliced with the same trial bounds.

ii.
```python
lick_raw = behavior["lick"][start:stop]
neural_trial = neural[start:stop].T
```

iii. The agent reports exact raw-versus-converted checks.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone identity is derived from the scene encoded in the NWB `identifier`, not from the framewise `reward_zone` stream. Trial order selects the pre/post-switch scene label.

ii.
```python
scene = read_str(f["identifier"]).split("/")[-1]
seq = parse_scene_reward_sequence(scene)
return [seq[0]] * split + [seq[1]] * max(0, n_trials - split)
```

iii. The agent argues scene metadata is the intended identity source and raw `reward_zone` values are occupancy/entry signals.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Regexes parse one or two A/B/C labels. Single-zone sessions use one label throughout; switch sessions use the first label for up to 30 trials and the second thereafter. Labels map A/B/C to 0/1/2 and are repeated over time.

ii.
```python
split = min(SWITCH_TRIAL_INDEX, n_trials)
...
np.full(pos.shape, {"A": 0, "B": 1, "C": 2}[zone_label], dtype=np.int64)
```

iii. The notes cite the reference code’s default trial-30 reward switch and verified an approximately balanced full-data label distribution.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the sparse NWB behavior `Reward` timestamps (reward amounts are loaded but not used for the binary result).

ii.
```python
reward_timestamps = f[f"{BEHAVIOR_TS_PATH}/Reward/timestamps"][:]
reward_amounts = f[f"{BEHAVIOR_TS_PATH}/Reward/data"][:]
```

iii. The agent interprets presence of a reward delivery as rewarded and absence as omission.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Out-of-range reward events are discarded; remaining timestamps are mapped with left `searchsorted` and clipped. A trial is 1 if any mapped index lies in its half-open bounds, otherwise 0; the scalar is repeated across time.

ii.
```python
reward_valid = reward_timestamps <= behavior["timestamps"][-1]
reward_frame_idx = np.searchsorted(behavior["timestamps"], reward_timestamps, side="left")
has_reward = np.any((reward_frame_idx >= start) & (reward_frame_idx < stop))
```

iii. The notes say this matches rewarded/omission logic and obtained the expected ~15% omission fraction.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural planes and behavior are cropped to their common minimum length; late reward events are dropped and mapped indices clipped; unmatched trial markers and empty trials are ignored; missing valid environment values raise an error; sessions below two valid trials are skipped. Ten one-sample mismatches were trimmed. There is no imputation.

ii.
```python
common_len = min(neural_len, behavior_len)
...
reward_valid = reward_timestamps <= behavior["timestamps"][-1]
...
if tele_ptr >= len(teleports):
    break
...
if valid.size == 0:
    raise ValueError("No valid environment values within trial slice.")
```

iii. The agent documented these as defensive handling of released-data edge cases and validated that trimming removed format warnings without changing trial alignment.

## 13-a. What are the most time-consuming steps of the code?

i. Reading and concatenating the large deconvolved imaging arrays dominates conversion; full pickle serialization and optional plotting are additional costs. The per-session timing log enables diagnosis. The full conversion reportedly took 53.79 seconds, while decoder training is separate and much slower.

ii.
```python
response_data = response_group["data"][:common_len, :]
neural = np.concatenate(neural_planes, axis=1)
...
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes emphasize raw HDF5 slicing and curated-trace-only loading as speedups over NWB object construction.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-bound pairing loop, reward-outcome loop, and per-trial construction loop could partly be vectorized, though variable trial lengths make final list construction naturally iterative. Reward membership currently scans all reward indices once per trial.

ii.
```python
for start in starts:
    ...
for i, (start, stop) in enumerate(bounds):
    has_reward = np.any((reward_frame_idx >= start) & (reward_frame_idx < stop))
...
for raw_trial_idx, (start, stop) in enumerate(bounds):
```

iii. The agent did not explicitly document vectorization candidates, but chose in-memory slicing and sequential sessions as the principal efficiency improvements.

## 13-c. What processing does the code repeat multiple times?

i. Each trial separately allocates repeated constant rows and applies discretizers. Reward indices are compared against every trial twice in effect: once for outcomes and once to build debug event groups. Plot mode retains and revisits trial arrays. There is no separate full-dataset survey pass in this script.

ii.
```python
has_reward = np.any((reward_frame_idx >= start) & (reward_frame_idx < stop))
reward_trials.append(np.flatnonzero((reward_frame_idx >= start) & (reward_frame_idx < stop)))
...
np.full(time_from_start.shape, env_value, dtype=np.float32)
```

iii. The notes stress avoiding repeated NWB loads; most remaining repetition supports variable-length output construction or optional diagnostics.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `reward_amounts`, per-neuron plane indices, several loaded behavior streams (`reward_zone`, `scanning`, `autoreward`), and much of `trial_debug` do not affect the saved decoder arrays. Debug arrays are built even when plots are disabled, although only their length indirectly matters nowhere.

ii.
```python
reward_amounts = f[f"{BEHAVIOR_TS_PATH}/Reward/data"][:]
...
for name in [..., "reward_zone", "scanning", ..., "autoreward"]:
    behavior[name] = ...
...
trial_debug.append({"distance_cm": distance, "position_cm": pos, ...})
```

iii. The agent did not identify these as unnecessary; it viewed diagnostic information and plane metadata as useful for validation and provenance.
