# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI recursively finds every `.nwb` file under `data/` and loads each session directly with `h5py`. Inside each file it reads the behavioral arrays and all deconvolved ophys plane matrices into memory before splitting into trials.

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))
...
with h5py.File(path, 'r') as h:
    identifier = dec(h['identifier'][()])
    subject = dec(h['general/subject/subject_id'][()])
    beh = h['processing/behavior/BehavioralTimeSeries']
    ...
    plane_names = sorted(h['processing/ophys/Deconvolved'].keys())
    for plane in plane_names:
        arr = np.array(h[f'processing/ophys/Deconvolved/{plane}/data'], dtype=np.float32)
        deconv_planes.append(arr)
```

iii. The notes justify this mainly as a direct NWB/HDF5 conversion path; there is no detailed defense of using `h5py` instead of `pynwb`. Step 6 notes also emphasize loading full session arrays once and concatenating planes for speed.

## 1-b. How are the data split into subjects?

i. Subjects are taken from each NWB file’s `general/subject/subject_id` field. A unique `subject_names` list is built in file iteration order, and each kept session gets a `subject_idx` into that list.

ii.
```python
with h5py.File(path, 'r') as h:
    subject = dec(h['general/subject/subject_id'][()])
...
if info['subject'] not in subject_names:
    subject_names.append(info['subject'])
subject_idx.append(subject_names.index(info['subject']))
```

iii. There is no explicit justification beyond using the NWB metadata field as the canonical subject identifier.

## 1-c. How are the data split into sessions?

i. Each discovered `.nwb` file is treated as one session. The final dataset stores one session-level list entry per kept file.

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))
...
for i, f in enumerate(files):
    ...
    sessions_neural.append(neural_trials)
    sessions_input.append(input_trials)
    sessions_output.append(output_trials)
```

iii. The notes and trajectory treat one NWB file as one session and do not describe any additional within-file session splitting.

## 1-d. How are the data split into trials?

i. The final code groups samples by the stored `trial number` stream and further masks them to samples with `scanning > 0`. It does not use `trial_start` or `teleport` to define trial boundaries, even though the notes earlier discussed trial-start alignment.

ii.
```python
trial_num = np.array(beh['trial number/data'], dtype=np.int64)
trial_start = np.array(beh['trial_start/data'], dtype=np.float32)
scanning = np.array(beh['scanning/data'], dtype=np.float32)
...
trial_ids = np.unique(trial_num)
trial_ids = trial_ids[trial_ids >= 0]
...
for tr in trial_ids:
    idx = np.where((trial_num == tr) & (scanning > 0))[0]
    if len(idx) < 2:
        continue
```

iii. Step 5 notes said trials would be aligned to trial start using `trial_start` and `trial number`, but the implemented code instead uses `trial number` plus `scanning`. No final justification is given for dropping the `teleport`-based boundary logic.

## 1-e. How are trials filtered based on quality controls?

i. Trials are skipped if they have fewer than 2 kept samples after applying `(trial_num == tr) & (scanning > 0)`. Trials are also skipped if the per-trial `reward_zone` values are all NaN. Entire sessions are skipped if fewer than 2 valid trials remain.

ii.
```python
for tr in trial_ids:
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

iii. The session-level `<2 trials` rule is justified by the decoder format requirement. The code gives no explicit justification for the `<2 samples` or all-NaN reward-zone trial filters.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from every plane under `processing/ophys/Deconvolved/*/data`.

ii.
```python
plane_names = sorted(h['processing/ophys/Deconvolved'].keys())
for plane in plane_names:
    arr = np.array(h[f'processing/ophys/Deconvolved/{plane}/data'], dtype=np.float32)
    deconv_planes.append(arr)
neural_full = np.concatenate(deconv_planes, axis=1)
```

iii. Step 5 explicitly says to use deconvolved calcium activity rather than fluorescence or neuropil because that is the processed neural signal used in the paper/code.

## 2-b. How is the `neural` data processed?

i. The code concatenates deconvolved data from all planes across the neuron axis, slices each trial by the selected time indices, transposes each trial from `(time, neurons)` to `(neurons, time)`, and stores it as `float16`.

ii.
```python
for plane in plane_names:
    arr = np.array(h[f'processing/ophys/Deconvolved/{plane}/data'], dtype=np.float32)
    deconv_planes.append(arr)
neural_full = np.concatenate(deconv_planes, axis=1)
...
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
```

iii. Step 6 notes justify plane concatenation as a speed optimization. The `float16` cast is not explicitly justified except indirectly by later notes about reducing pickle size.

## 2-c. How is the `neural` data filtered based on quality controls?

i. In the final code, it is not filtered by ROI quality metrics at all. All deconvolved columns from all planes are kept.

ii.
```python
for plane in plane_names:
    arr = np.array(h[f'processing/ophys/Deconvolved/{plane}/data'], dtype=np.float32)
    deconv_planes.append(arr)
neural_full = np.concatenate(deconv_planes, axis=1)
```

iii. This conflicts with Step 5 notes, which said cell inclusion would likely use `iscell`. There is no final justification for omitting ROI filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial’s neural data are aligned to the first kept sample in that trial-number/scanning segment. The code uses the same `q_idx` indices for neural and behavior and defines relative time from the first timestamp in that segment, rather than explicitly aligning to the `trial_start` event.

ii.
```python
idx = np.where((trial_num == tr) & (scanning > 0))[0]
q_idx = idx
qts = pos_t[q_idx]
trial_t0 = qts[0]
rel_t = (qts - trial_t0).astype(np.float32)
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
```

iii. The notes repeatedly say the intent was trial-start alignment, but the implemented alignment is actually “first retained sample within each trial-number block.” No explicit justification is given for this difference.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the native behavioral sampling interval estimated from `position/timestamps`. No temporal rebinning is applied.

ii.
```python
pos_t = np.array(beh['position/timestamps'], dtype=np.float64)
dt = float(np.median(np.diff(pos_t)))
...
'time_bin_size': float(np.median(dts) * 1000.0) if dts else None,
```

iii. Step 5 says to use a common time bin size based on the native behavior/imaging sampling interval. Step 6 further notes that the initial script used native samples rather than explicit rebinning.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from `position/timestamps`, not from `trial number.timestamps`.

ii.
```python
pos_t = np.array(beh['position/timestamps'], dtype=np.float64)
...
qts = pos_t[q_idx]
trial_t0 = qts[0]
rel_t = (qts - trial_t0).astype(np.float32)
```

iii. The notes only say “behavior timestamps relative to trial start”; they do not explain why `position` timestamps were chosen over other behavior timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each kept trial segment, the first timestamp is subtracted from all timestamps in that segment.

ii.
```python
qts = pos_t[q_idx]
trial_t0 = qts[0]
rel_t = (qts - trial_t0).astype(np.float32)
...
inp = np.vstack([
    rel_t,
    ...
]).astype(np.float32)
```

iii. The notes explicitly say this variable should be “behavior timestamps relative to trial start,” and the code implements that by zeroing the first kept sample.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by using the same sample indices `q_idx` for both the neural matrix and the timestamp vector. There is no explicit cross-stream timestamp verification.

ii.
```python
q_idx = idx
qts = pos_t[q_idx]
rel_t = (qts - trial_t0).astype(np.float32)
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
```

iii. The code assumes the behavioral sample indices and deconvolved frame indices are already aligned. The notes do not document any stronger alignment check in the final implementation.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The final stored `input` uses the NWB `identifier` string, parsed for `Env2` vs not-`Env2`. The code also reads the `environment` stream and computes `env_binary`, but does not actually use it in the stored input.

ii.
```python
env = np.array(beh['environment/data'], dtype=np.float32)
...
env_binary = infer_env_binary(identifier, env)
env_from_identifier = parse_env_from_identifier(identifier)
...
inp = np.vstack([
    rel_t,
    np.full(len(q_idx), float(env_from_identifier), dtype=np.float32),
    ...
]).astype(np.float32)
```

iii. Step 5 notes say environment should be mapped from “task structure in identifiers + behavior stream” and was likely constant within trial. The later implementation collapses this to identifier-only without further justification.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The code converts the session identifier to a binary value, then broadcasts that constant over all timepoints in the trial.

ii.
```python
def parse_env_from_identifier(identifier):
    if 'Env2' in identifier:
        return 1
    return 0
...
np.full(len(q_idx), float(env_from_identifier), dtype=np.float32)
```

iii. The notes imply the environment is effectively constant within a trial/session, which is the main apparent justification for replacing the recorded stream with a parsed constant.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived directly from the stored `trial number` stream; each unique nonnegative `trial number` becomes one trial label.

ii.
```python
trial_num = np.array(beh['trial number/data'], dtype=np.int64)
trial_ids = np.unique(trial_num)
trial_ids = trial_ids[trial_ids >= 0]
...
np.full(len(q_idx), float(tr), dtype=np.float32)
```

iii. Step 5 notes said “use trial index within session,” but the final code uses the raw `trial number` value itself. No final rationale is given for preferring the stored value.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The code broadcasts the raw trial ID as a constant over all timepoints in the trial.

ii.
```python
inp = np.vstack([
    rel_t,
    np.full(len(q_idx), float(env_from_identifier), dtype=np.float32),
    np.full(len(q_idx), float(tr), dtype=np.float32),
    np.full(len(q_idx), float(prev_outcome), dtype=np.float32),
]).astype(np.float32)
```

iii. No further processing is documented. The value is simply copied into every time bin of the trial.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the `Reward` behavior time series and its timestamps.

ii.
```python
reward_event = np.array(beh['Reward/data'], dtype=np.float32)
reward_event_t = np.array(beh['Reward/timestamps'], dtype=np.float64)
...
reward_in_trial = reward_event[(reward_event_t >= t_lo) & (reward_event_t <= t_hi)]
outcome = int(np.any(reward_in_trial > 0))
```

iii. Step 5 explicitly says previous trial outcome should come from reward events, with omitted mapped to 0 and rewarded to 1.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial-number block, the code computes the current trial’s reward outcome from reward events falling between that trial’s first and last timestamps. It then stores the previous block’s outcome as a constant covariate for the next block, with the first trial set to 0.

ii.
```python
prev_outcome = 0
...
t_lo, t_hi = qts[0], qts[-1]
reward_in_trial = reward_event[(reward_event_t >= t_lo) & (reward_event_t <= t_hi)]
outcome = int(np.any(reward_in_trial > 0))
...
np.full(len(q_idx), float(prev_outcome), dtype=np.float32),
...
prev_outcome = outcome
```

iii. Step 5 justifies this as the natural binary previous-trial covariate, with trial 0 defaulting to 0.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the `position` stream and from the `reward_zone` stream. The code estimates a center position for each distinct raw reward-zone value by taking the median position where that value occurs, then uses the current trial’s modal reward-zone value to choose the center.

ii.
```python
pos = np.array(beh['position/data'], dtype=np.float32)
rz = np.array(beh['reward_zone/data'], dtype=np.float32)
...
reward_zone_centers = {}
for zval in np.unique(rz[~np.isnan(rz)]):
    mask = rz == zval
    if np.any(mask):
        reward_zone_centers[zval] = float(np.nanmedian(pos[mask & (speed > -np.inf)]))
...
rz_vals = rz_trial[~np.isnan(rz_trial)]
rz_mode = float(Counter(rz_vals.tolist()).most_common(1)[0][0])
rz_center = reward_zone_centers.get(rz_mode, float(np.nanmedian(pos_trial)))
```

iii. The notes say distance should be computed from position and reward-zone location, but the final code does not use the survey/Viterbi zone labeling described in the human reference. There is no explicit final justification for switching to a center-estimation approach.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The code subtracts the inferred reward-zone center from position, then discretizes the signed center-relative distance with a custom thresholding function.

ii.
```python
dist = pos_trial - rz_center
dist_bin = discretize_distance(dist)
...
def discretize_distance(x):
    out = np.zeros_like(x, dtype=np.int64)
    out[x < -50] = 0
    out[(x >= -50) & (x <= -10)] = 1
    out[(x > -10) & (x < 0)] = 2
    out[x == 0] = 3
    out[(x > 0) & (x <= 10)] = 4
    out[(x > 10) & (x <= 50)] = 5
    out[x > 50] = 6
```

iii. The only explicit justification is that the output should be reward-relative and use the task’s bin scheme. The code does not justify using zone center instead of distance to the nearest zone edge.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is thresholded by `discretize_distance` into 7 integer bins with explicit comparison operators. The boundary handling is custom rather than `np.digitize`.

ii.
```python
def discretize_distance(x):
    out = np.zeros_like(x, dtype=np.int64)
    out[x < -50] = 0
    out[(x >= -50) & (x <= -10)] = 1
    out[(x > -10) & (x < 0)] = 2
    out[x == 0] = 3
    out[(x > 0) & (x <= 10)] = 4
    out[(x > 10) & (x <= 50)] = 5
    out[x > 50] = 6
    return out
```

iii. Step 5 says to discretize into the 7 task-specified bins. There is no explicit explanation of the chosen inclusive/exclusive edge conventions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by using the same `q_idx` indices used for the neural trial matrix and then storing the binned distance time series with the same length as the neural time axis.

ii.
```python
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
pos_trial = pos[q_idx]
...
dist = pos_trial - rz_center
dist_bin = discretize_distance(dist)
```

iii. The code assumes sample-wise alignment from shared indexing and does not add any additional synchronization step.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the `position` behavior stream.

ii.
```python
pos = np.array(beh['position/data'], dtype=np.float32)
...
pos_trial = pos[q_idx]
```

iii. No extra justification is given beyond position being the raw corridor coordinate to decode.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The code computes the session-wide minimum and maximum position values, divides that range into 5 equal-width bins, and discretizes each trial’s position values against those edges.

ii.
```python
abs_pos_global_min = float(np.nanmin(pos))
abs_pos_global_max = float(np.nanmax(pos))
...
abs_pos_bin = discretize_abs_position(pos_trial, abs_pos_global_min, abs_pos_global_max)
...
def discretize_abs_position(pos, pmin, pmax):
    edges = np.linspace(pmin, pmax, 6)
    bins = np.digitize(pos, edges[1:-1], right=False)
    bins = np.clip(bins, 0, 4)
```

iii. Step 5 says to discretize the corridor into 5 equal bins. The implemented justification appears to be taking “equal-sized bins” literally from the observed session range.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is thresholded with 5 equal-width bins defined by `np.linspace(session_min, session_max, 6)` and then `np.digitize`.

ii.
```python
def discretize_abs_position(pos, pmin, pmax):
    edges = np.linspace(pmin, pmax, 6)
    bins = np.digitize(pos, edges[1:-1], right=False)
    bins = np.clip(bins, 0, 4)
    return bins.astype(np.int64)
```

iii. The notes only justify “5 equal bins”; they do not discuss whether the bins should be globally fixed across sessions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned by slicing `position` with the same `q_idx` indices used for the neural trial matrix.

ii.
```python
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
pos_trial = pos[q_idx]
abs_pos_bin = discretize_abs_position(pos_trial, abs_pos_global_min, abs_pos_global_max)
```

iii. As elsewhere, the code relies on shared sample indexing rather than explicit timestamp checks.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the `lick` behavior stream.

ii.
```python
lick = np.array(beh['lick/data'], dtype=np.float32)
...
lick_trial = (lick[q_idx] > 0).astype(np.int64)
```

iii. No special justification is given; this is the obvious raw stream for lick events.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The code binarizes the raw lick values by thresholding at `> 0`.

ii.
```python
lick_trial = (lick[q_idx] > 0).astype(np.int64)
...
out = np.vstack([
    ...,
    lick_trial,
    ...
]).astype(np.int16)
```

iii. Step 5 says lick should be binary 0/1, which is the apparent justification for this threshold.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is aligned by slicing the lick stream with the same `q_idx` indices used for neural.

ii.
```python
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
lick_trial = (lick[q_idx] > 0).astype(np.int64)
```

iii. The code assumes the behavior and deconvolved arrays are already sample-aligned.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. In the final code, the stored reward-zone location comes from the NWB `identifier` string (`LocationA`, `LocationB`, `LocationC`), not from the `reward_zone` behavior stream. The raw `reward_zone` stream is only used indirectly for the distance-to-zone computation.

ii.
```python
identifier = dec(h['identifier'][()])
...
session_reward_location = parse_location_from_identifier(identifier)
...
np.full(len(q_idx), rz_loc, dtype=np.int64),
```

iii. The trajectory explicitly says the agent “fixed” reward-zone location by deriving A/B/C from the session identifier rather than the time-varying `reward_zone` stream.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The identifier substring is mapped to `0/1/2` and then broadcast as a trial-constant output for every timepoint in the trial.

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
...
np.full(len(q_idx), rz_loc, dtype=np.int64)
```

iii. The main justification recorded in the notes/trajectory is that sample-session reward-zone labels then varied “as expected from session identifiers.” No trialwise inference step remains in the final code.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the `Reward` behavior time series and its timestamps.

ii.
```python
reward_event = np.array(beh['Reward/data'], dtype=np.float32)
reward_event_t = np.array(beh['Reward/timestamps'], dtype=np.float64)
...
reward_in_trial = reward_event[(reward_event_t >= t_lo) & (reward_event_t <= t_hi)]
outcome = int(np.any(reward_in_trial > 0))
```

iii. Step 5 explicitly says reward outcome should be derived from reward delivery events within each trial.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial-number/scanning segment, the code marks the output as 1 if any reward event with positive value falls between the first and last timestamps of that segment, otherwise 0. The result is then broadcast across the trial’s timepoints.

ii.
```python
t_lo, t_hi = qts[0], qts[-1]
reward_in_trial = reward_event[(reward_event_t >= t_lo) & (reward_event_t <= t_hi)]
outcome = int(np.any(reward_in_trial > 0))
...
np.full(len(q_idx), outcome, dtype=np.int64)
```

iii. The notes justify this as the required per-trial rewarded-vs-omitted label. Later trajectory analysis argued the labels matched raw reward events and that decoder weakness was due to class imbalance, not conversion error.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code has limited handling. It skips trials with too few kept samples, skips trials whose reward-zone values are all NaN, and skips sessions with fewer than 2 valid trials. It does not implement the stronger neural/behavior length checks or timestamp assertions described in the human reference.

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

iii. The only explicit justification in the notes is the decoder requirement that each session have at least two trials. Other missing-data behavior is largely implicit.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive parts are reading large NWB arrays from disk, concatenating all deconvolved planes for every session, and then looping over every trial to allocate per-trial neural/input/output arrays. There is also an extra full-dataset prepass that reads every file’s `reward_zone` array before conversion.

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))
...
all_rz = []
for f in files:
    with h5py.File(f, 'r') as h:
        all_rz.append(np.array(h['processing/behavior/BehavioralTimeSeries/reward_zone/data'], dtype=np.float32))
...
for plane in plane_names:
    arr = np.array(h[f'processing/ophys/Deconvolved/{plane}/data'], dtype=np.float32)
    deconv_planes.append(arr)
...
for tr in trial_ids:
    ...
```

iii. Step 6 notes explicitly mention that the script “loads full session arrays into memory,” and the sample runtime estimate is presented per session.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop, the per-file reward-zone prepass, and the per-session reward-zone-center construction loop are the main places the AI left in scalar/Python-loop form. Much of the discretization could have been vectorized at the full-session level before trial splitting.

ii.
```python
for f in files:
    with h5py.File(f, 'r') as h:
        all_rz.append(...)
...
for zval in np.unique(rz[~np.isnan(rz)]):
    mask = rz == zval
    if np.any(mask):
        reward_zone_centers[zval] = ...
...
for tr in trial_ids:
    idx = np.where((trial_num == tr) & (scanning > 0))[0]
    ...
```

iii. Step 6 notes only mention avoiding per-neuron loops. There is no evidence that the agent considered vectorizing the trial-level postprocessing beyond that.

## 13-c. What processing does the code repeat multiple times?

i. The code first scans every session file to collect all `reward_zone` arrays for `infer_reward_zone_map`, then opens every session again for the actual conversion. Within conversion, it also repeatedly constructs trial-level constant arrays with `np.full`.

ii.
```python
all_rz = []
for f in files:
    with h5py.File(f, 'r') as h:
        all_rz.append(np.array(h['processing/behavior/BehavioralTimeSeries/reward_zone/data'], dtype=np.float32))
reward_zone_value_map = infer_reward_zone_map(np.concatenate(all_rz))
...
for i, f in enumerate(files):
    ...
    neural_trials, input_trials, output_trials, info = load_session(...)
```

iii. The notes emphasize direct session-by-session conversion and do not justify this extra full-data prepass, especially because the resulting map is not used later.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computed values are unused: `reward_zone_value_map`, the `nearest_sample` helper, the loaded `trial_start` array, the computed `env_binary` and `env_trial`, and the `reward_zone_value_map` argument passed into `load_session`. The code also computes reward-zone centers from raw `reward_zone` values even though the separate reward-zone-location output ignores that stream and comes from the identifier.

ii.
```python
def nearest_sample(values, ts, qts):
    ...

def load_session(path, reward_zone_value_map, show_processing=False):
    ...
    trial_start = np.array(beh['trial_start/data'], dtype=np.float32)
    ...
    env_binary = infer_env_binary(identifier, env)
    ...
    env_trial = env_binary[q_idx]
    ...

all_rz = []
for f in files:
    ...
reward_zone_value_map = infer_reward_zone_map(np.concatenate(all_rz))
...
neural_trials, input_trials, output_trials, info = load_session(
    f, reward_zone_value_map, show_processing=args.show_processing and i < 2
)
```

iii. Step 6 notes mention only broad speedups and do not acknowledge these discarded computations. These appear to be leftovers from earlier versions of the approach.
