# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads every `.nwb` file under `data/` with `Path('data').rglob('*.nwb')`, then opens each file with `h5py.File`. Each file is treated as one session and all trial data are later extracted from that file.

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))

for i, f in enumerate(files):
    print(f'[{i+1}/{len(files)}] loading {f}', flush=True)
    neural_trials, input_trials, output_trials, info = load_session(
        f, reward_zone_value_map, show_processing=args.show_processing and i < 2
    )
```

```python
with h5py.File(path, 'r') as h:
    identifier = dec(h['identifier'][()])
    subject = dec(h['general/subject/subject_id'][()])
```

iii. `CONVERSION_NOTES.md` Step 2 says the dataset is organized as NWB session files under subject subdirectories, and Step 6 says the script uses NWB deconvolved activity with session-wise loading.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are not taken from directory names in the final code. Instead, each session contributes the NWB field `general/subject/subject_id`, and unique values are accumulated in encounter order.

ii.
```python
subject = dec(h['general/subject/subject_id'][()])
...
if info['subject'] not in subject_names:
    subject_names.append(info['subject'])
subject_idx.append(subject_names.index(info['subject']))
```

iii. The notes say sessions are grouped by subject subdirectories, but the final code uses the subject id stored inside each NWB file, which is a direct metadata source.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file is treated as one session.

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))
...
for i, f in enumerate(files):
    neural_trials, input_trials, output_trials, info = load_session(f, ...)
```

iii. This matches the notes’ description that the dataset consists of NWB session files under subject folders.

## 1-d. How are the data split into trials?

i. Trials are defined by unique nonnegative values in the `trial number` behavior stream, then restricted to samples where `scanning > 0`. The loaded `trial_start` stream is not actually used.

ii.
```python
trial_num = np.array(beh['trial number/data'], dtype=np.int64)
trial_start = np.array(beh['trial_start/data'], dtype=np.float32)
scanning = np.array(beh['scanning/data'], dtype=np.float32)
...
trial_ids = np.unique(trial_num)
trial_ids = trial_ids[trial_ids >= 0]

for tr in trial_ids:
    idx = np.where((trial_num == tr) & (scanning > 0))[0]
    if len(idx) < 2:
        continue
```

iii. `CONVERSION_NOTES.md` Step 5 says temporal alignment would use `trial_start` / `trial number`, but the shipped code simplified this to trial-number segmentation plus `scanning > 0`.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops trials with fewer than 2 valid samples and trials whose `reward_zone` values are all NaN. After session conversion, it also drops sessions with fewer than 2 remaining trials.

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
```

```python
if len(neural_trials) < 2:
    print(f'  skipping {f} because <2 valid trials')
    continue
```

iii. No detailed QC rationale is documented beyond general notes about excluding invalid data periods and ensuring at least two trials per session for decoder use.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` data are taken directly from `processing/ophys/Deconvolved/<plane>/data` in the NWB files.

ii.
```python
deconv_planes = []
plane_names = sorted(h['processing/ophys/Deconvolved'].keys())
for plane in plane_names:
    arr = np.array(h[f'processing/ophys/Deconvolved/{plane}/data'], dtype=np.float32)
    deconv_planes.append(arr)
neural_full = np.concatenate(deconv_planes, axis=1)
```

iii. `CONVERSION_NOTES.md` Step 3 and Step 5 explicitly state that the AI chose the NWB `Deconvolved` matrices because the methods mention deconvolved activity matrices.

## 2-b. How is the `neural` data processed?

i. Processing is minimal: the AI concatenates all imaging planes across the neuron axis, slices each trial, transposes to `(neurons, time)`, and casts to `float16`. It does not recompute dF/F or deconvolution.

ii.
```python
neural_full = np.concatenate(deconv_planes, axis=1)
...
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
...
neural_trials.append(neural_trial)
```

iii. The notes justify this by claiming the NWB deconvolved activity is the processed signal used in the paper; Step 10 and Step 12 also mention float16 compression as a storage optimization.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit neuron-quality filtering in the final code. It does not apply `iscell`, interneuron exclusion, or any cell-level curation. The only indirect filtering is that neural samples are restricted to `scanning > 0` timepoints through the trial index.

ii.
```python
for plane in plane_names:
    arr = np.array(h[f'processing/ophys/Deconvolved/{plane}/data'], dtype=np.float32)
    deconv_planes.append(arr)
...
idx = np.where((trial_num == tr) & (scanning > 0))[0]
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
```

iii. `CONVERSION_NOTES.md` Step 4 says cell filtering was still unresolved and “likely” to use `iscell`, but that plan was not implemented in the shipped script.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned implicitly to the first kept sample of each `trial number` segment after applying `scanning > 0`. There is no separate realignment to `trial_start`.

ii.
```python
for tr in trial_ids:
    idx = np.where((trial_num == tr) & (scanning > 0))[0]
    ...
    qts = pos_t[q_idx]
    trial_t0 = qts[0]
    rel_t = (qts - trial_t0).astype(np.float32)
    neural_trial = neural_full[q_idx, :].T.astype(np.float16)
```

iii. The notes say trials are “trial-start aligned,” but the final code operationalizes that as using the first retained sample in each trial-number segment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native sampling interval. It estimates session `dt` from the median spacing of `position` timestamps and applies no temporal rebinning.

ii.
```python
pos_t = np.array(beh['position/timestamps'], dtype=np.float64)
dt = float(np.median(np.diff(pos_t)))
...
'time_bin_size': float(np.median(dts) * 1000.0) if dts else None,
```

iii. `CONVERSION_NOTES.md` Step 5 says the plan was to use the native behavior/imaging sampling interval and avoid explicit rebinning.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the `position` timestamp stream, `position/timestamps`.

ii.
```python
pos_t = np.array(beh['position/timestamps'], dtype=np.float64)
...
qts = pos_t[q_idx]
trial_t0 = qts[0]
rel_t = (qts - trial_t0).astype(np.float32)
```

iii. The notes only say behavior timestamps are used to represent time-from-trial-start; they do not defend choosing `position` timestamps specifically over another behavior timestamp field.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the first kept timestamp is subtracted from all timestamps in that trial.

ii.
```python
qts = pos_t[q_idx]
trial_t0 = qts[0]
rel_t = (qts - trial_t0).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 says this input should be “timestamps relative to trial start.”

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned by using the same per-trial sample indices `q_idx` for both neural and behavioral arrays.

ii.
```python
q_idx = idx
qts = pos_t[q_idx]
rel_t = (qts - trial_t0).astype(np.float32)
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
```

iii. Step 5 in the notes says all streams would be put on the same trial-aligned bins; the final implementation achieves this only by shared indexing, not by explicit validation.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The final input is derived from the NWB `identifier` string, not from the `environment` time series. The code does read and infer a binary environment stream, but it is not used for the saved input.

ii.
```python
env = np.array(beh['environment/data'], dtype=np.float32)
...
env_binary = infer_env_binary(identifier, env)
env_from_identifier = parse_env_from_identifier(identifier)
...
np.full(len(q_idx), float(env_from_identifier), dtype=np.float32),
```

iii. `CONVERSION_NOTES.md` Step 5 says environment would come from the behavior stream plus session identifiers and would likely be constant within trial; the final code collapses this to identifier parsing.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The effective processing is string parsing: sessions whose identifier contains `Env2` are assigned 1, otherwise 0, and that value is broadcast across the whole trial.

ii.
```python
def parse_env_from_identifier(identifier):
    if 'Env2' in identifier:
        return 1
    return 0
...
np.full(len(q_idx), float(env_from_identifier), dtype=np.float32),
```

iii. The notes justify environment as a per-trial constant. They do not explicitly justify preferring identifier parsing over the behavior stream in the final implementation.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the raw `trial number` behavior stream. The unique trial id `tr` becomes the trial-number input.

ii.
```python
trial_num = np.array(beh['trial number/data'], dtype=np.int64)
...
trial_ids = np.unique(trial_num)
trial_ids = trial_ids[trial_ids >= 0]
...
np.full(len(q_idx), float(tr), dtype=np.float32),
```

iii. `CONVERSION_NOTES.md` Step 5 says trial number would use the `trial number` source variable and be a per-trial quantity.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No extra transform is applied beyond selecting the raw trial id and broadcasting it over all timepoints in the trial.

ii.
```python
np.full(len(q_idx), float(tr), dtype=np.float32),
```

iii. No more specific justification is documented.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the `Reward` event stream: both `Reward/data` and `Reward/timestamps`.

ii.
```python
reward_event = np.array(beh['Reward/data'], dtype=np.float32)
reward_event_t = np.array(beh['Reward/timestamps'], dtype=np.float64)
...
reward_in_trial = reward_event[(reward_event_t >= t_lo) & (reward_event_t <= t_hi)]
outcome = int(np.any(reward_in_trial > 0))
```

iii. Step 5 in the notes says previous trial outcome should be derived from reward events and shifted by one trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI first computes the current trial’s binary reward outcome as whether any reward event fell within that trial’s time interval. It then stores a running `prev_outcome`, initializes it to 0 for the first trial, and broadcasts that previous value across the next trial’s timepoints.

ii.
```python
prev_outcome = 0
...
reward_in_trial = reward_event[(reward_event_t >= t_lo) & (reward_event_t <= t_hi)]
outcome = int(np.any(reward_in_trial > 0))
...
np.full(len(q_idx), float(prev_outcome), dtype=np.float32),
...
prev_outcome = outcome
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly states that the first trial defaults to 0 and later trials use the prior trial’s reward outcome.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from `position` and the numeric `reward_zone` behavior stream. The code infers a center position for each numeric reward-zone value by taking the session-wide median position where that value occurs.

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
```

iii. Step 5 in the notes says distance-to-zone would come from position plus reward-zone information; later notes mention that reward-zone location was initially misread and then partly corrected for the separate reward-location output.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each trial, the modal numeric `reward_zone` value is selected, mapped to its inferred center position, and the output is computed as `position - center`. This is then discretized; it is a signed distance to the inferred center, not distance to the nearest edge.

ii.
```python
rz_vals = rz_trial[~np.isnan(rz_trial)]
rz_mode = float(Counter(rz_vals.tolist()).most_common(1)[0][0])
rz_center = reward_zone_centers.get(rz_mode, float(np.nanmedian(pos_trial)))
dist = pos_trial - rz_center
dist_bin = discretize_distance(dist)
```

iii. The notes say the goal was “position relative to reward zone,” but they do not document the specific center-based calculation used in the final script.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI uses seven hard-coded bins implemented by comparisons, with labels corresponding to `< -50`, `-50 to -10`, `-10 to <0`, `0`, `>0 to 10`, `10 to 50`, and `>50`.

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

iii. `CONVERSION_NOTES.md` Step 5 says the discretization should follow the task specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by computing position-derived quantities on the same per-trial indices `q_idx` used to slice the neural matrix.

ii.
```python
q_idx = idx
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
pos_trial = pos[q_idx]
...
dist = pos_trial - rz_center
```

iii. The notes’ general alignment plan was to use shared trial-aligned sample indices for all streams.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the `position` behavior stream.

ii.
```python
pos = np.array(beh['position/data'], dtype=np.float32)
...
pos_trial = pos[q_idx]
```

iii. `CONVERSION_NOTES.md` Step 5 maps `position` to the absolute-position output.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI discretizes position into five equal-width bins spanning the observed session-wide minimum and maximum position values, not the fixed 0-450 cm corridor.

ii.
```python
abs_pos_global_min = float(np.nanmin(pos))
abs_pos_global_max = float(np.nanmax(pos))
...
abs_pos_bin = discretize_abs_position(pos_trial, abs_pos_global_min, abs_pos_global_max)
```

```python
def discretize_abs_position(pos, pmin, pmax):
    edges = np.linspace(pmin, pmax, 6)
    bins = np.digitize(pos, edges[1:-1], right=False)
    bins = np.clip(bins, 0, 4)
    return bins.astype(np.int64)
```

iii. Step 5 of the notes says absolute position should be discretized into five bins, but the final implementation chooses data-driven session bounds rather than the fixed track bounds in the task.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Thresholds are session-specific `np.linspace(pmin, pmax, 6)` edges, clipped into five categories.

ii.
```python
edges = np.linspace(pmin, pmax, 6)
bins = np.digitize(pos, edges[1:-1], right=False)
bins = np.clip(bins, 0, 4)
```

iii. No separate justification beyond the generic Step 5 discretization plan is documented.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is aligned by slicing position with the same trial indices as the neural data.

ii.
```python
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
pos_trial = pos[q_idx]
abs_pos_bin = discretize_abs_position(pos_trial, abs_pos_global_min, abs_pos_global_max)
```

iii. This follows the notes’ shared-index alignment strategy.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the `lick` behavior stream.

ii.
```python
lick = np.array(beh['lick/data'], dtype=np.float32)
...
lick_trial = (lick[q_idx] > 0).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` Step 5 maps the `lick` stream directly to the lick output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick values are binarized: `> 0` becomes 1, otherwise 0.

ii.
```python
lick_trial = (lick[q_idx] > 0).astype(np.int64)
```

iii. Step 5 in the notes says lick should be a binary time-varying output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It uses the same per-trial sample indices as the neural data.

ii.
```python
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
lick_trial = (lick[q_idx] > 0).astype(np.int64)
```

iii. No extra alignment step is documented beyond shared indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. In the final code it is derived from the NWB `identifier` string by parsing `LocationA`, `LocationB`, or `LocationC`.

ii.
```python
identifier = dec(h['identifier'][()])
...
session_reward_location = parse_location_from_identifier(identifier)
...
np.full(len(q_idx), rz_loc, dtype=np.int64),
```

iii. `CONVERSION_NOTES.md` Step 10 and Step 12 explicitly say reward-zone location was initially misread from the behavior stream and then “fixed by deriving A/B/C from session identifier.”

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The identifier is string-matched to A/B/C, converted to 0/1/2, and broadcast across all timepoints in each trial.

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
np.full(len(q_idx), rz_loc, dtype=np.int64),
```

iii. The explicit justification in the notes is that this replaced an earlier, incorrect use of the time-varying `reward_zone` stream for location labels.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/data` and `Reward/timestamps`.

ii.
```python
reward_event = np.array(beh['Reward/data'], dtype=np.float32)
reward_event_t = np.array(beh['Reward/timestamps'], dtype=np.float64)
...
reward_in_trial = reward_event[(reward_event_t >= t_lo) & (reward_event_t <= t_hi)]
```

iii. `CONVERSION_NOTES.md` Step 5 and Step 12 both describe reward outcome as coming from reward events in each trial.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is labeled rewarded if any positive reward event falls between the first and last timestamps kept for that trial. The scalar label is then broadcast across all trial timepoints.

ii.
```python
t_lo, t_hi = qts[0], qts[-1]
reward_in_trial = reward_event[(reward_event_t >= t_lo) & (reward_event_t <= t_hi)]
outcome = int(np.any(reward_in_trial > 0))
...
np.full(len(q_idx), outcome, dtype=np.int64),
```

iii. Step 12 says the AI checked this logic against raw NWB reward events and concluded the labels matched exactly for sampled sessions.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Handling is limited. The code skips very short trial segments (`len(idx) < 2`), skips trials with no non-NaN `reward_zone` values, skips sessions with fewer than 2 valid trials, defaults unknown location/env parsing to 0, and falls back to the trial median position if a reward-zone center is missing. It does not implement explicit neural/behavior length reconciliation or timestamp-consistency assertions.

ii.
```python
if len(idx) < 2:
    continue
...
rz_vals = rz_trial[~np.isnan(rz_trial)]
if len(rz_vals) == 0:
    continue
...
rz_center = reward_zone_centers.get(rz_mode, float(np.nanmedian(pos_trial)))
```

```python
if len(neural_trials) < 2:
    print(f'  skipping {f} because <2 valid trials')
    continue
```

iii. The notes mostly frame this as excluding invalid periods and keeping the decoder-compatible minimum of two trials per session; they do not document a broader missing-data policy.

## 13-a. What are the most time-consuming steps of the code?

i. The likely dominant costs are reading all NWB files, loading large deconvolved matrices, concatenating planes per session, iterating over all trials, and writing the large pickle. There is also a full-dataset prepass over `reward_zone` arrays before session conversion.

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))
...
for f in files:
    with h5py.File(f, 'r') as h:
        all_rz.append(np.array(h['processing/behavior/BehavioralTimeSeries/reward_zone/data'], dtype=np.float32))
...
for i, f in enumerate(files):
    neural_trials, input_trials, output_trials, info = load_session(f, ...)
```

iii. `CONVERSION_NOTES.md` Step 6 mentions full-session array loading as the main inefficiency and notes that the code avoids per-neuron loops.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining loop is the per-trial loop inside `load_session`, which repeatedly slices arrays and discretizes outputs one trial at a time. The reward-zone-center computation over unique values is another small loop. Several per-trial broadcasts and discretizations could be precomputed session-wide before splitting.

ii.
```python
for tr in trial_ids:
    idx = np.where((trial_num == tr) & (scanning > 0))[0]
    ...
    dist_bin = discretize_distance(dist)
    abs_pos_bin = discretize_abs_position(pos_trial, abs_pos_global_min, abs_pos_global_max)
    speed_bin = discretize_speed(speed_trial)
```

iii. Step 6 in the notes explicitly says the code still loads full session arrays into memory and uses native samples rather than more optimized rebinned/vectorized processing.

## 13-c. What processing does the code repeat multiple times?

i. The code makes one pass over all files to collect `reward_zone` streams into `all_rz`, then reopens all files again for real conversion. Inside each session it also computes `env_binary` and `env_trial`, but the final saved input actually uses `env_from_identifier` instead.

ii.
```python
all_rz = []
for f in files:
    with h5py.File(f, 'r') as h:
        all_rz.append(np.array(h['processing/behavior/BehavioralTimeSeries/reward_zone/data'], dtype=np.float32))
reward_zone_value_map = infer_reward_zone_map(np.concatenate(all_rz))
```

```python
env_binary = infer_env_binary(identifier, env)
...
env_trial = env_binary[q_idx]
...
np.full(len(q_idx), float(env_from_identifier), dtype=np.float32),
```

iii. The notes discuss a pre-conversion survey step in the abstract, but in the final script the clearest repeated work is the all-files reward-zone prepass plus unused environment inference.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations are unused in the final dataset: `reward_zone_value_map` is built and passed into `load_session` but never used; `infer_env_binary` and `env_trial` are computed but ignored; `trial_start` is loaded but unused; and `nearest_sample` is defined but never called.

ii.
```python
def nearest_sample(values, ts, qts):
    ...
```

```python
trial_start = np.array(beh['trial_start/data'], dtype=np.float32)
...
env_binary = infer_env_binary(identifier, env)
...
env_trial = env_binary[q_idx]
```

```python
reward_zone_value_map = infer_reward_zone_map(np.concatenate(all_rz))
...
def load_session(path, reward_zone_value_map, show_processing=False):
```

iii. No explicit justification is given for keeping these unused paths; they appear to be remnants of earlier versions described in the notes and trajectory.
