# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script glob-loads every `.nwb` file under `data/`, optionally truncates to the first two files in `--sample` mode, and processes each file as one session via `load_session(...)`. Within `load_session`, it reads behavior streams from `processing/behavior/BehavioralTimeSeries` and neural data from `processing/ophys/Deconvolved`.

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))
if args.sample:
    files = files[:2]

for i, f in enumerate(files):
    neural_trials, input_trials, output_trials, info = load_session(
        f, reward_zone_value_map, show_processing=args.show_processing and i < 2
    )
```

iii. The notes justify this as matching the NWB session organization seen in Step 2, where each file was treated as one session and the relevant behavior and ophys groups were identified.

## 1-b. How are the data split into subjects?

i. Subjects are split by the NWB metadata field `general/subject/subject_id`. A unique `subject_names` list is built in encounter order, and each kept session stores its integer `subject_idx`.

ii.
```python
subject = dec(h['general/subject/subject_id'][()])
...
if info['subject'] not in subject_names:
    subject_names.append(info['subject'])
subject_idx.append(subject_names.index(info['subject']))
```

iii. This follows the dataset exploration notes, which counted subjects from NWB metadata and reported 11 mice.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as exactly one session. The top-level lists `neural`, `input`, `output`, and `brain_region_idx` each receive one element per successfully converted NWB file.

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))
...
sessions_neural.append(neural_trials)
sessions_input.append(input_trials)
sessions_output.append(output_trials)
brain_region_idx.append(info['brain_region_idx'])
```

iii. The notes repeatedly describe the dataset as “NWB session files” and summarize counts at the file/session level, so the AI clearly chose file = session.

## 1-d. How are the data split into trials?

i. Trials are split by the `trial number/data` stream. The script takes unique nonnegative trial IDs, then for each trial keeps only samples where `scanning > 0`; those indices define the per-trial window for neural and behavioral arrays.

ii.
```python
trial_num = np.array(beh['trial number/data'], dtype=np.int64)
scanning = np.array(beh['scanning/data'], dtype=np.float32)
trial_ids = np.unique(trial_num)
trial_ids = trial_ids[trial_ids >= 0]

for tr in trial_ids:
    idx = np.where((trial_num == tr) & (scanning > 0))[0]
    if len(idx) < 2:
        continue
```

iii. Step 5 in the notes says trial segmentation should use `trial_start` / `trial number`, but the implemented code actually uses `trial number` plus `scanning > 0`. No deeper justification was recorded beyond treating `scanning` as a valid-data mask.

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering is minimal. A trial is dropped if it has fewer than two `scanning > 0` samples or if its `reward_zone` values are all NaN. A whole session is dropped only if fewer than two trials survive.

ii.
```python
idx = np.where((trial_num == tr) & (scanning > 0))[0]
if len(idx) < 2:
    continue
...
rz_vals = rz_trial[~np.isnan(rz_trial)]
if len(rz_vals) == 0:
    continue
...
if len(neural_trials) < 2:
    print(f'  skipping {f} because <2 valid trials')
    continue
```

iii. The notes mention expected curation rules from the paper/code, but the implemented justification is mostly pragmatic: retain trials with usable sampled data and enough trials to satisfy the decoder format.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the NWB ophys deconvolved calcium traces in `processing/ophys/Deconvolved/plane*/data`.

ii.
```python
plane_names = sorted(h['processing/ophys/Deconvolved'].keys())
for plane in plane_names:
    arr = np.array(h[f'processing/ophys/Deconvolved/{plane}/data'], dtype=np.float32)
    deconv_planes.append(arr)
```

iii. This is explicitly justified in the notes: the methods excerpt mentions “deconvolved activity matrices,” and Step 4 resolves the neural signal choice in favor of deconvolved activity.

## 2-b. How is the `neural` data processed?

i. The script concatenates all deconvolved planes across neurons, slices each trial by time index, transposes each trial from `(time, neurons)` to `(neurons, time)`, and stores it as `float16`.

ii.
```python
neural_full = np.concatenate(deconv_planes, axis=1)
...
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
```

iii. The notes justify using deconvolved activity and mention an efficiency pass that concatenated planes once per session and later compacted neural arrays to `float16` to reduce the huge pickle size.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is effectively not filtered for neural quality. The script includes every ROI present in the deconvolved matrices, with no use of `iscell`, no cell-quality threshold, and no place-cell/TR/RR subset restriction.

ii.
```python
for plane in plane_names:
    arr = np.array(h[f'processing/ophys/Deconvolved/{plane}/data'], dtype=np.float32)
    deconv_planes.append(arr)
neural_full = np.concatenate(deconv_planes, axis=1)
...
brain_region_idx = np.zeros(neural_full.shape[1], dtype=np.int16)
```

iii. The notes show the AI considered filtering with `iscell` in Step 5, but no final explicit justification for omitting that filter was recorded; the implementation simply keeps all deconvolved ROIs.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. In code, neural data are aligned to the first sample in the kept `scanning > 0` segment of each trial, not to the `trial_start` event stream. Trial-relative time zero is `qts[0]`.

ii.
```python
idx = np.where((trial_num == tr) & (scanning > 0))[0]
qts = pos_t[q_idx]
trial_t0 = qts[0]
rel_t = (qts - trial_t0).astype(np.float32)
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
```

iii. This conflicts with the Step 5 notes, which said trials would be aligned to `trial_start` “as required.” The justification preserved in notes is therefore stronger than the code actually delivered.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data keep the native behavior/imaging sample spacing, with metadata `time_bin_size` set to the median `np.diff(position_timestamps)` across sessions. No temporal rebinning is implemented.

ii.
```python
dt = float(np.median(np.diff(pos_t)))
...
dts.append(info['dt_s'])
...
'time_bin_size': float(np.median(dts) * 1000.0) if dts else None,
```

iii. Step 5 says the AI intended to use a common bin size based on the native sampling interval. Step 6 also notes that the code uses “nearest native samples rather than explicit rebinning.”

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. In the implementation, this input is derived from `position/timestamps` for samples inside the `trial number` and `scanning` mask. The `trial_start` stream is read but not used.

ii.
```python
pos_t = np.array(beh['position/timestamps'], dtype=np.float64)
trial_num = np.array(beh['trial number/data'], dtype=np.int64)
trial_start = np.array(beh['trial_start/data'], dtype=np.float32)
...
qts = pos_t[q_idx]
rel_t = (qts - trial_t0).astype(np.float32)
```

iii. The notes claim this variable should come from “behavior timestamps relative to trial start,” but the code never uses `trial_start` despite reading it.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The processing is just subtraction of the first kept timestamp in the trial: `rel_t = qts - qts[0]`. This produces seconds since the first `scanning > 0` sample, not since the explicit `trial_start` pulse.

ii.
```python
qts = pos_t[q_idx]
trial_t0 = qts[0]
rel_t = (qts - trial_t0).astype(np.float32)
```

iii. Step 5 says the AI intended true start-of-trial alignment, but there is no code implementing that plan; the only visible justification is convenience and matching the kept sample indices.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by construction to exactly the same per-trial sample indices as `neural`: both are sliced by `q_idx`, and `rel_t` has one value per neural timepoint.

ii.
```python
q_idx = idx
qts = pos_t[q_idx]
rel_t = (qts - trial_t0).astype(np.float32)
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
inp = np.vstack([rel_t, ...]).astype(np.float32)
```

iii. The notes emphasize joint trial alignment of all streams. Even though the event choice is questionable, the time input and neural data are at least sample-aligned to each other.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The script reads the `environment/data` stream, but the actual stored decoder input is derived from the session `identifier` string via `parse_env_from_identifier(...)`. The inferred `env_binary` from the raw stream is unused.

ii.
```python
env = np.array(beh['environment/data'], dtype=np.float32)
...
env_binary = infer_env_binary(identifier, env)
env_from_identifier = parse_env_from_identifier(identifier)
...
np.full(len(q_idx), float(env_from_identifier), dtype=np.float32),
```

iii. Step 5 says the AI meant to use the `environment` stream, but later implementation silently switched to identifier parsing. No explicit justification for ignoring the stream was recorded.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The stored environment input is a single constant per session/trial, set to `1` if `"Env2"` appears anywhere in the identifier and `0` otherwise, then broadcast across all timepoints in the trial.

ii.
```python
def parse_env_from_identifier(identifier):
    if 'Env2' in identifier:
        return 1
    return 0
...
np.full(len(q_idx), float(env_from_identifier), dtype=np.float32),
```

iii. The notes justify environment as a binary trial-level variable, but they explicitly mapped it from the `environment` stream. The code-level decision to parse the identifier was not separately defended.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived directly from `processing/behavior/BehavioralTimeSeries/trial number/data`.

ii.
```python
trial_num = np.array(beh['trial number/data'], dtype=np.int64)
...
np.full(len(q_idx), float(tr), dtype=np.float32),
```

iii. This matches the Step 5 mapping table, which explicitly assigns `trial number` to the decoder input.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial ID `tr` is taken from the unique values of the `trial number` stream and then broadcast as a constant across every timepoint in the corresponding trial.

ii.
```python
for tr in trial_ids:
    ...
    inp = np.vstack([
        rel_t,
        ...,
        np.full(len(q_idx), float(tr), dtype=np.float32),
        ...
    ]).astype(np.float32)
```

iii. The notes justify this as using the within-session trial index as contextual decoder input.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from the `Reward/data` event stream and its timestamps, using the prior trial's computed reward outcome.

ii.
```python
reward_event = np.array(beh['Reward/data'], dtype=np.float32)
reward_event_t = np.array(beh['Reward/timestamps'], dtype=np.float64)
...
reward_in_trial = reward_event[(reward_event_t >= t_lo) & (reward_event_t <= t_hi)]
outcome = int(np.any(reward_in_trial > 0))
...
np.full(len(q_idx), float(prev_outcome), dtype=np.float32),
```

iii. Step 5 explicitly says previous trial outcome should come from reward events, with the first trial defaulting to 0.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the code marks the current outcome as rewarded if any positive reward event falls inside the trial's kept time span, then stores the previous trial's outcome as a constant input for the current trial. The first trial starts with `prev_outcome = 0`.

ii.
```python
prev_outcome = 0
...
t_lo, t_hi = qts[0], qts[-1]
reward_in_trial = reward_event[(reward_event_t >= t_lo) & (reward_event_t <= t_hi)]
outcome = int(np.any(reward_in_trial > 0))
...
prev_outcome = outcome
```

iii. The notes justify exactly this recurrence and later record a sanity check showing reward-outcome labels matched raw event-derived labels on checked sessions.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from `position/data` and `reward_zone/data`. The code first infers a reward-zone “center” for each raw reward-zone code, then subtracts that center from trial position.

ii.
```python
pos = np.array(beh['position/data'], dtype=np.float32)
rz = np.array(beh['reward_zone/data'], dtype=np.float32)
...
reward_zone_centers[zval] = float(np.nanmedian(pos[mask & (speed > -np.inf)]))
...
dist = pos_trial - rz_center
```

iii. Step 5 says distance should come from position relative to reward-zone location, and the code tries to operationalize that using the raw `reward_zone` stream.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The script computes a per-session dictionary `reward_zone_centers` by taking the median position for each unique raw `reward_zone` value. For each trial it takes the modal raw `reward_zone` value, looks up the corresponding median position, and subtracts that scalar from every trial position sample.

ii.
```python
reward_zone_centers = {}
for zval in np.unique(rz[~np.isnan(rz)]):
    mask = rz == zval
    reward_zone_centers[zval] = float(np.nanmedian(pos[mask & (speed > -np.inf)]))
...
rz_mode = float(Counter(rz_vals.tolist()).most_common(1)[0][0])
rz_center = reward_zone_centers.get(rz_mode, float(np.nanmedian(pos_trial)))
dist = pos_trial - rz_center
```

iii. The notes justify distance as a reward-relative variable, but they do not explicitly justify this median-position heuristic for interpreting the raw `reward_zone` codes.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is discretized into 7 ordered bins by `discretize_distance(...)`, with thresholds at `-50`, `-10`, `0`, `10`, and `50` cm.

ii.
```python
def discretize_distance(x):
    out[x < -50] = 0
    out[(x >= -50) & (x <= -10)] = 1
    out[(x > -10) & (x < 0)] = 2
    out[x == 0] = 3
    out[(x > 0) & (x <= 10)] = 4
    out[(x > 10) & (x <= 50)] = 5
    out[x > 50] = 6
```

iii. This follows the task specification almost verbatim, so no separate justification beyond compliance is needed.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Distance is computed from `pos_trial = pos[q_idx]`, so it uses the same sample indices and trial boundaries as the neural matrix.

ii.
```python
q_idx = idx
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
pos_trial = pos[q_idx]
...
dist = pos_trial - rz_center
dist_bin = discretize_distance(dist)
```

iii. The notes repeatedly say all decoder variables should be trial-aligned and sample-aligned; this part of the implementation follows that plan.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is derived directly from the `position/data` behavior stream.

ii.
```python
pos = np.array(beh['position/data'], dtype=np.float32)
...
pos_trial = pos[q_idx]
```

iii. The Step 5 mapping table explicitly assigns absolute position to the raw `position` variable.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The script computes a session-wide minimum and maximum over all position samples, then bins each trial's position samples relative to those global session limits.

ii.
```python
abs_pos_global_min = float(np.nanmin(pos))
abs_pos_global_max = float(np.nanmax(pos))
...
abs_pos_bin = discretize_abs_position(pos_trial, abs_pos_global_min, abs_pos_global_max)
```

iii. The notes justify absolute position as “discretize corridor into 5 equal bins,” but they do not justify using observed session min/max, which also include off-corridor values such as `-500`.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `discretize_abs_position(...)` draws 6 equally spaced edges between the session's minimum and maximum position and then uses `np.digitize` to assign bins `0` through `4`.

ii.
```python
def discretize_abs_position(pos, pmin, pmax):
    edges = np.linspace(pmin, pmax, 6)
    bins = np.digitize(pos, edges[1:-1], right=False)
    bins = np.clip(bins, 0, 4)
    return bins.astype(np.int64)
```

iii. The only explicit justification is matching the decoder task's requirement for 5 equal-sized bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Absolute position uses `pos[q_idx]`, so it is sampled on the same per-trial indices as the neural data.

ii.
```python
q_idx = idx
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
pos_trial = pos[q_idx]
abs_pos_bin = discretize_abs_position(pos_trial, abs_pos_global_min, abs_pos_global_max)
```

iii. This is consistent with the AI's general plan to keep all trial variables on a shared sample grid.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick is derived from `processing/behavior/BehavioralTimeSeries/lick/data`.

ii.
```python
lick = np.array(beh['lick/data'], dtype=np.float32)
...
lick_trial = (lick[q_idx] > 0).astype(np.int64)
```

iii. Step 5 maps `lick` directly from the behavior stream and treats it as a time-varying binary output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The lick stream is thresholded to binary: any value greater than zero becomes `1`, otherwise `0`.

ii.
```python
lick_trial = (lick[q_idx] > 0).astype(np.int64)
```

iii. The notes justify this as matching the decoder's required binary lick output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick uses the same `q_idx` sample indices as the neural trial, so it is timepoint-aligned to the neural matrix.

ii.
```python
q_idx = idx
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
lick_trial = (lick[q_idx] > 0).astype(np.int64)
```

iii. The justification is the same shared sample-grid design used throughout the script.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. In the final code, reward-zone location is derived from the NWB `identifier` string, not from `reward_zone/data`. The raw `reward_zone` stream is only used for distance-to-zone.

ii.
```python
identifier = dec(h['identifier'][()])
...
session_reward_location = parse_location_from_identifier(identifier)
...
np.full(len(q_idx), rz_loc, dtype=np.int64),
```

iii. The notes explicitly record that the AI initially misread the `reward_zone` stream and then “fixed” reward-zone location by deriving A/B/C from the session identifier instead.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The identifier is parsed with simple substring checks: if it contains `LocationA`, return `0`; else `LocationB` gives `1`; else `LocationC` gives `2`; otherwise default to `0`. That single session-level label is then broadcast across every timepoint of every trial.

ii.
```python
def parse_location_from_identifier(identifier):
    if 'LocationA' in identifier:
        return 0
    if 'LocationB' in identifier:
        return 1
    if 'LocationC' in identifier:
        return 2
    return 0
...
rz_loc = session_reward_location
```

iii. The justification in Step 10/12 is that session identifiers seemed to match expected sample-session labels better than the time-varying `reward_zone` stream. No more nuanced handling of switch sessions was recorded.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from the `Reward/data` event series and `Reward/timestamps`.

ii.
```python
reward_event = np.array(beh['Reward/data'], dtype=np.float32)
reward_event_t = np.array(beh['Reward/timestamps'], dtype=np.float64)
...
reward_in_trial = reward_event[(reward_event_t >= t_lo) & (reward_event_t <= t_hi)]
```

iii. Step 5 explicitly says per-trial reward outcome should come from reward delivery versus omission, derived from reward events.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the code checks whether any positive reward event falls between the first and last kept trial timestamps. If so, outcome is `1`; otherwise `0`. That value is broadcast over all timepoints in the trial.

ii.
```python
t_lo, t_hi = qts[0], qts[-1]
reward_in_trial = reward_event[(reward_event_t >= t_lo) & (reward_event_t <= t_hi)]
outcome = int(np.any(reward_in_trial > 0))
...
np.full(len(q_idx), outcome, dtype=np.int64),
```

iii. The notes justify this approach and later claim a raw-data sanity check showed these labels matched event-derived outcomes on checked sessions.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or awkward data are handled with a mix of silent skipping and silent defaults. Trials with too few valid samples or all-NaN reward-zone values are skipped; NaNs are masked out in `infer_env_binary(...)` and reward-zone-center estimation; identifier parsing defaults unknown labels to `0`; and if a trial reward-zone code lacks a precomputed center, the code falls back to the trial's median position.

ii.
```python
if len(idx) < 2:
    continue
...
rz_vals = rz_trial[~np.isnan(rz_trial)]
if len(rz_vals) == 0:
    continue
...
return 0
...
rz_center = reward_zone_centers.get(rz_mode, float(np.nanmedian(pos_trial)))
```

iii. Step 6 says the script should “handle missing data appropriately,” but the recorded justification is mostly practical rather than reference-based; no detailed missing-data policy from the reference workflow was documented.

## 13-a. What are the most time-consuming steps of the code?

i. The likely bottlenecks are repeated file I/O over all NWB files, full-session loading of large deconvolved matrices, concatenating planes, and the per-trial extraction loop over every trial in every session.

ii.
```python
for f in files:
    with h5py.File(f, 'r') as h:
        all_rz.append(np.array(...))
...
for plane in plane_names:
    arr = np.array(h[f'processing/ophys/Deconvolved/{plane}/data'], dtype=np.float32)
...
for tr in trial_ids:
    idx = np.where((trial_num == tr) & (scanning > 0))[0]
```

iii. Step 6 and later notes explicitly mention timing, large pickle size, and the cost of loading full session arrays into memory.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop is the main remaining nonvectorized loop, especially repeated `np.where(...)`, `Counter(...)`, and repeated per-trial broadcasts. The reward-zone-center loop over unique `zval` codes could also be vectorized or precomputed more directly.

ii.
```python
for zval in np.unique(rz[~np.isnan(rz)]):
    mask = rz == zval
    reward_zone_centers[zval] = float(np.nanmedian(pos[mask & (speed > -np.inf)]))

for tr in trial_ids:
    idx = np.where((trial_num == tr) & (scanning > 0))[0]
    ...
    rz_mode = float(Counter(rz_vals.tolist()).most_common(1)[0][0])
```

iii. The notes already flag vectorization as a goal and acknowledge only partial speedups were added.

## 13-c. What processing does the code repeat multiple times?

i. It makes a full first pass over every file just to collect `reward_zone` values for `infer_reward_zone_map(...)`, then a second pass to do the real conversion. Within sessions it also repeatedly builds constant trial-wise arrays, repeatedly searches trial indices, and repeatedly computes quantities from the same trial mask.

ii.
```python
all_rz = []
for f in files:
    with h5py.File(f, 'r') as h:
        all_rz.append(np.array(...))
reward_zone_value_map = infer_reward_zone_map(np.concatenate(all_rz))

for i, f in enumerate(files):
    neural_trials, input_trials, output_trials, info = load_session(...)
```

iii. Step 6 calls out inefficiency from loading full session arrays and from using native-sample extraction without a more consolidated resampling path.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations are dead or effectively unused: `infer_reward_zone_map(...)` is computed and passed into `load_session(...)` but never used there; `nearest_sample(...)` is defined but never called; `env_binary` and `env_trial` are computed but not used in the final inputs; and `trial_start` is read but ignored. The script also reads `reward_zone_value_map`-related data even though reward-zone location ultimately comes from the identifier.

ii.
```python
def nearest_sample(values, ts, qts):
    ...

reward_zone_value_map = infer_reward_zone_map(np.concatenate(all_rz))
...
def load_session(path, reward_zone_value_map, show_processing=False):
    ...
    trial_start = np.array(beh['trial_start/data'], dtype=np.float32)
    env_binary = infer_env_binary(identifier, env)
    ...
    env_trial = env_binary[q_idx]
```

iii. The notes partly acknowledge this drift: Step 5 planned to use the environment and trial-start streams directly, but later debugging shifted some outputs to identifier parsing without cleaning up the now-unused computations.
