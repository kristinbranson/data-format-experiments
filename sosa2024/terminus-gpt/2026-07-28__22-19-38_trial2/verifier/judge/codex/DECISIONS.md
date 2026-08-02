# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads every `.nwb` file under `data/` with `Path('data').rglob('*.nwb')`, then opens each file directly with `h5py.File`. Within each file it reads behavior arrays from `processing/behavior/BehavioralTimeSeries` and deconvolved neural arrays from `processing/ophys/Deconvolved`.

ii. ```python
files = sorted(Path('data').rglob('*.nwb'))

for i, f in enumerate(files):
    neural_trials, input_trials, output_trials, info = load_session(
        f, reward_zone_value_map, show_processing=args.show_processing and i < 2
    )
```

```python
with h5py.File(path, 'r') as h:
    beh = h['processing/behavior/BehavioralTimeSeries']
    ...
    plane_names = sorted(h['processing/ophys/Deconvolved'].keys())
```

iii. The notes say the data are organized as NWB session files under subject subdirectories and that behavior and ophys live at those NWB paths. Step 6 also says the script was written around direct NWB loading with deconvolved activity. No explicit justification was given for preferring `h5py` over `pynwb`.

## 1-b. How are the data split into subjects?

i. Subjects are not inferred from directory names. The AI reads `general/subject/subject_id` from each NWB file and builds `subjects` in first-seen file order.

ii. ```python
subject = dec(h['general/subject/subject_id'][()])
...
if info['subject'] not in subject_names:
    subject_names.append(info['subject'])
subject_idx.append(subject_names.index(info['subject']))
```

iii. The notes say the dataset contains 11 subjects and is organized by subject subdirectories, but the final code uses the NWB subject field instead. No separate justification beyond using the NWB metadata is recorded.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The top-level session lists in `neural`, `input`, and `output` are populated once per file.

ii. ```python
files = sorted(Path('data').rglob('*.nwb'))
...
for i, f in enumerate(files):
    ...
    sessions_neural.append(neural_trials)
    sessions_input.append(input_trials)
    sessions_output.append(output_trials)
```

iii. The notes describe the data as “NWB session files” and report 152 sessions from the provided data, consistent with one session per file.

## 1-d. How are the data split into trials?

i. Trials are defined by unique nonnegative values in the `trial number` stream, then restricted to samples where `scanning > 0`. The code reads `trial_start` but does not use it.

ii. ```python
trial_num = np.array(beh['trial number/data'], dtype=np.int64)
trial_start = np.array(beh['trial_start/data'], dtype=np.float32)
scanning = np.array(beh['scanning/data'], dtype=np.float32)
...
trial_ids = np.unique(trial_num)
trial_ids = trial_ids[trial_ids >= 0]
...
for tr in trial_ids:
    idx = np.where((trial_num == tr) & (scanning > 0))[0]
```

iii. Step 5 says the plan was to segment using the `trial_start` / `trial number` streams, but Step 6 describes the implemented script as using “trial-number segmentation.” That matches the final code.

## 1-e. How are trials filtered based on quality controls?

i. A trial is dropped if it has fewer than 2 `scanning > 0` samples or if its `reward_zone` values are all NaN. After session conversion, the whole session is dropped if fewer than 2 valid trials remain.

ii. ```python
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

iii. The notes emphasize excluding invalid periods via `scanning` and ensuring at least two trials per session for decoder evaluation. No explicit justification was recorded for the `<2 samples` threshold or skipping trials with missing reward-zone values.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data come from every plane’s `processing/ophys/Deconvolved/<plane>/data` array.

ii. ```python
plane_names = sorted(h['processing/ophys/Deconvolved'].keys())
for plane in plane_names:
    arr = np.array(h[f'processing/ophys/Deconvolved/{plane}/data'], dtype=np.float32)
    deconv_planes.append(arr)
neural_full = np.concatenate(deconv_planes, axis=1)
```

iii. Step 4 and Step 5 explicitly justify using deconvolved calcium activity because both the paper/methods and the NWB files indicate that this is the processed neural signal used in the analyses.

## 2-b. How is the `neural` data processed?

i. Plane matrices are concatenated across neurons, then per-trial slices are taken and transposed from `(time, neurons)` to `(neurons, time)`. The trial matrices are downcast to `float16` before storage.

ii. ```python
for plane in plane_names:
    arr = np.array(h[f'processing/ophys/Deconvolved/{plane}/data'], dtype=np.float32)
    deconv_planes.append(arr)
neural_full = np.concatenate(deconv_planes, axis=1)
...
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
```

iii. Step 6 says the implementation “concatenates planes once per session and avoids per-neuron loops,” and Step 10/12 says the dtype compression was added to reduce the full pickle size from roughly 29G to 15G.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The final code does not apply any neuron-level quality filter. It includes all columns from every deconvolved plane and ignores `iscell`.

ii. ```python
for plane in plane_names:
    arr = np.array(h[f'processing/ophys/Deconvolved/{plane}/data'], dtype=np.float32)
    deconv_planes.append(arr)
neural_full = np.concatenate(deconv_planes, axis=1)
```

iii. Step 5 proposed likely filtering with `iscell`, but the implemented script does not do that. No explicit justification for dropping the planned filter appears in the notes.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned implicitly by slicing the same `trial_num`/`scanning` indices as behavior and then resetting time within each slice so the first retained sample is time zero.

ii. ```python
idx = np.where((trial_num == tr) & (scanning > 0))[0]
qts = pos_t[q_idx]
trial_t0 = qts[0]
rel_t = (qts - trial_t0).astype(np.float32)
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
```

iii. Step 5 says the intended temporal alignment event was trial start. Step 6 then describes “trial-number segmentation” and “trial-start alignment,” implying the AI considered the first retained sample from each `trial number` block to be the trial start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data keep the native sample spacing from `position/timestamps`; `metadata['time_bin_size']` is the median session `dt` in milliseconds. No explicit temporal rebinning is performed.

ii. ```python
dt = float(np.median(np.diff(pos_t)))
...
dts.append(info['dt_s'])
...
'time_bin_size': float(np.median(dts) * 1000.0) if dts else None,
```

iii. Step 6 notes that the initial version “uses nearest native samples rather than explicit rebinning,” and the final code indeed leaves the native sampling unchanged.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from `position/timestamps`, restricted to each trial’s retained indices.

ii. ```python
pos_t = np.array(beh['position/timestamps'], dtype=np.float64)
...
qts = pos_t[q_idx]
rel_t = (qts - trial_t0).astype(np.float32)
```

iii. The notes broadly describe using behavior timestamps relative to trial start. The final code specifically uses position timestamps; no extra justification for choosing them over another behavior timestamp stream is recorded.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp in the retained trial slice is subtracted from each timestamp to make the trial start equal to zero.

ii. ```python
qts = pos_t[q_idx]
trial_t0 = qts[0]
rel_t = (qts - trial_t0).astype(np.float32)
```

iii. Step 5 states the goal was to align all trials to trial start as required by the task. This subtraction is the implemented alignment.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The same index array `q_idx` is used to slice both the behavior timestamps and the neural matrix, so the time input and neural data have the same retained samples within each trial.

ii. ```python
q_idx = idx
qts = pos_t[q_idx]
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
```

iii. Step 5 says the plan was to “resample behavior and neural data onto the same trial-aligned bins,” but the final code instead assumes native index alignment and uses shared indices directly.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The final `input` variable is derived from the NWB `identifier` string, not from the `environment` behavior stream. The `environment` stream is read and an `env_binary` array is computed, but that array is not used in the exported inputs.

ii. ```python
identifier = dec(h['identifier'][()])
env = np.array(beh['environment/data'], dtype=np.float32)
...
env_binary = infer_env_binary(identifier, env)
env_from_identifier = parse_env_from_identifier(identifier)
...
np.full(len(q_idx), float(env_from_identifier), dtype=np.float32),
```

iii. Step 5 planned to map `environment` to the decoder input, but Step 10/12 record a later correction for reward-zone labels based on session identifiers. No explicit written justification was given for making environment depend on the identifier instead of the behavior stream.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The code parses the session `identifier`: if it contains `Env2`, the environment is `1`; otherwise it is `0`. That constant is then broadcast across the whole trial.

ii. ```python
def parse_env_from_identifier(identifier):
    if 'Env2' in identifier:
        return 1
    return 0
...
np.full(len(q_idx), float(env_from_identifier), dtype=np.float32),
```

iii. The only recorded rationale is indirect: the notes describe “task structure in identifiers + behavior stream” as the intended source. The final implementation resolved that to identifier-only.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the raw `trial number` stream. The code groups samples by each unique nonnegative `trial_num` value and then stores that raw value as the trial-number input.

ii. ```python
trial_num = np.array(beh['trial number/data'], dtype=np.int64)
trial_ids = np.unique(trial_num)
trial_ids = trial_ids[trial_ids >= 0]
...
np.full(len(q_idx), float(tr), dtype=np.float32),
```

iii. Step 5 explicitly maps `trial number` to the decoder input and says to use the behavior stream. The final code follows that mapping literally.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. There is no extra processing beyond broadcasting the selected `trial number` value across all retained timepoints of the trial.

ii. ```python
inp = np.vstack([
    rel_t,
    np.full(len(q_idx), float(env_from_identifier), dtype=np.float32),
    np.full(len(q_idx), float(tr), dtype=np.float32),
    np.full(len(q_idx), float(prev_outcome), dtype=np.float32),
]).astype(np.float32)
```

iii. The notes describe this field simply as “continuous per trial/timepoint” and do not record any more elaborate transformation.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the `Reward/data` and `Reward/timestamps` streams, combined with each trial’s start and end times inferred from the retained sample indices.

ii. ```python
reward_event = np.array(beh['Reward/data'], dtype=np.float32)
reward_event_t = np.array(beh['Reward/timestamps'], dtype=np.float64)
...
t_lo, t_hi = qts[0], qts[-1]
reward_in_trial = reward_event[(reward_event_t >= t_lo) & (reward_event_t <= t_hi)]
outcome = int(np.any(reward_in_trial > 0))
```

iii. Step 5 states that previous trial outcome should be derived from reward events and that trial 0 should default to 0.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The script keeps a running `prev_outcome` variable. Trial 0 gets 0; after each trial, `prev_outcome` is updated from that trial’s reward outcome and then broadcast across the next trial.

ii. ```python
prev_outcome = 0
...
inp = np.vstack([
    rel_t,
    np.full(len(q_idx), float(env_from_identifier), dtype=np.float32),
    np.full(len(q_idx), float(tr), dtype=np.float32),
    np.full(len(q_idx), float(prev_outcome), dtype=np.float32),
]).astype(np.float32)
...
prev_outcome = outcome
```

iii. Step 5 explicitly says previous outcome comes from the prior trial’s reward outcome, with first trial defaulting to 0.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from the `position` stream plus the `reward_zone` stream. The code first computes a session-level center position for each distinct `reward_zone` value, then uses the modal `reward_zone` value within each trial to choose that center.

ii. ```python
pos = np.array(beh['position/data'], dtype=np.float32)
rz = np.array(beh['reward_zone/data'], dtype=np.float32)
...
for zval in np.unique(rz[~np.isnan(rz)]):
    mask = rz == zval
    if np.any(mask):
        reward_zone_centers[zval] = float(np.nanmedian(pos[mask & (speed > -np.inf)]))
...
rz_vals = rz_trial[~np.isnan(rz_trial)]
rz_mode = float(Counter(rz_vals.tolist()).most_common(1)[0][0])
rz_center = reward_zone_centers.get(rz_mode, float(np.nanmedian(pos_trial)))
```

iii. Step 5 says the AI intended to compute reward-relative position from `position` and reward-zone information. Later notes say reward-zone location was initially misread from the `reward_zone` stream and then corrected only for the separate `reward_zone_location` output; there is no later correction for this distance computation.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The code subtracts the inferred reward-zone center from each position sample, producing a signed distance from the center, then discretizes that quantity.

ii. ```python
rz_center = reward_zone_centers.get(rz_mode, float(np.nanmedian(pos_trial)))
dist = pos_trial - rz_center
dist_bin = discretize_distance(dist)
```

iii. The notes justify reward-relative coding as an important task variable, but no explicit written justification is given for using zone centers rather than zone edges.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI uses hard-coded thresholds in `discretize_distance`: `< -50`, `[-50, -10]`, `(-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50`.

ii. ```python
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

iii. The notes say the outputs should follow the task’s requested discretization bins. The final implementation is the AI’s direct thresholded version of that spec.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural activity are sliced with the same `q_idx` array for each trial, so the distance labels are aligned sample-for-sample with the retained neural samples.

ii. ```python
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
pos_trial = pos[q_idx]
...
dist = pos_trial - rz_center
```

iii. Step 5’s intended alignment strategy was shared trial-aligned bins across behavior and neural data. The implemented code realizes that by using the same retained indices.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the `position` behavior stream.

ii. ```python
pos = np.array(beh['position/data'], dtype=np.float32)
...
pos_trial = pos[q_idx]
```

iii. Step 5 maps absolute `position` directly to this decoder output.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The per-trial position slice is passed into `discretize_abs_position`, which bins it using session-specific min and max position values.

ii. ```python
abs_pos_global_min = float(np.nanmin(pos))
abs_pos_global_max = float(np.nanmax(pos))
...
abs_pos_bin = discretize_abs_position(pos_trial, abs_pos_global_min, abs_pos_global_max)
```

iii. Step 5 says the variable should be discretized into five equal bins. The AI implemented that using equal-width bins over the observed session range.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The code computes five equal-width bins from the session’s observed global minimum and maximum position using `np.linspace(pmin, pmax, 6)`, then digitizes into bin labels `0` to `4`.

ii. ```python
def discretize_abs_position(pos, pmin, pmax):
    edges = np.linspace(pmin, pmax, 6)
    bins = np.digitize(pos, edges[1:-1], right=False)
    bins = np.clip(bins, 0, 4)
    return bins.astype(np.int64)
```

iii. Step 7 notes that in the sample data `absolute_position_bin` only occupied bins 2-4, and says this may reflect partial occupancy of the corridor. That note is consistent with the observed-range binning choice.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position and neural activity use the same `q_idx` trial slices, so absolute-position labels are aligned sample-for-sample to neural activity.

ii. ```python
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
pos_trial = pos[q_idx]
abs_pos_bin = discretize_abs_position(pos_trial, abs_pos_global_min, abs_pos_global_max)
```

iii. The notes describe shared trial-aligned behavior and neural bins; the final code implements that by index sharing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived directly from the `lick` behavior stream.

ii. ```python
lick = np.array(beh['lick/data'], dtype=np.float32)
...
lick_trial = (lick[q_idx] > 0).astype(np.int64)
```

iii. Step 5 maps `lick` directly to a binary decoder output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The lick values are thresholded at `> 0` and converted to integer 0/1 labels.

ii. ```python
lick_trial = (lick[q_idx] > 0).astype(np.int64)
...
out = np.vstack([
    dist_bin,
    abs_pos_bin,
    speed_bin,
    lick_trial,
    ...
]).astype(np.int16)
```

iii. Step 5 states that lick should be binary 0/1 and time-varying.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural activity are sliced with the same `q_idx` indices for each trial.

ii. ```python
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
lick_trial = (lick[q_idx] > 0).astype(np.int64)
```

iii. The notes consistently describe neural and behavioral outputs as using the same trial-aligned samples.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The final exported reward-zone-location output is derived from the NWB `identifier` string, specifically from whether it contains `LocationA`, `LocationB`, or `LocationC`.

ii. ```python
def parse_location_from_identifier(identifier):
    if 'LocationA' in identifier:
        return 0
    if 'LocationB' in identifier:
        return 1
    if 'LocationC' in identifier:
        return 2
    return 0
...
session_reward_location = parse_location_from_identifier(identifier)
...
np.full(len(q_idx), rz_loc, dtype=np.int64),
```

iii. Step 10 and Step 12 explicitly say the AI initially misread reward-zone location from the time-varying `reward_zone` stream and then “fixed” it by deriving A/B/C from the session identifier.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The location substring in the session identifier is converted to `0/1/2` for A/B/C and then broadcast across all timepoints in the trial.

ii. ```python
session_reward_location = parse_location_from_identifier(identifier)
...
rz_loc = session_reward_location
...
np.full(len(q_idx), rz_loc, dtype=np.int64),
```

iii. The recorded justification is the Step 10/12 note that using the identifier fixed an earlier misinterpretation of the `reward_zone` stream.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward/data` and `Reward/timestamps`, restricted to the time window of each retained trial.

ii. ```python
reward_event = np.array(beh['Reward/data'], dtype=np.float32)
reward_event_t = np.array(beh['Reward/timestamps'], dtype=np.float64)
...
t_lo, t_hi = qts[0], qts[-1]
reward_in_trial = reward_event[(reward_event_t >= t_lo) & (reward_event_t <= t_hi)]
outcome = int(np.any(reward_in_trial > 0))
```

iii. Step 5 says reward outcome should come from reward events, with omitted trials mapped to 0 and rewarded trials to 1. Step 12 later says the AI checked these labels against raw NWB reward events and found exact agreement for inspected sessions.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the code checks whether any positive reward event timestamp falls between the first and last retained timestamps of that trial. The result is converted to `0/1` and broadcast across the trial.

ii. ```python
t_lo, t_hi = qts[0], qts[-1]
reward_in_trial = reward_event[(reward_event_t >= t_lo) & (reward_event_t <= t_hi)]
outcome = int(np.any(reward_in_trial > 0))
...
np.full(len(q_idx), outcome, dtype=np.int64),
```

iii. Step 5 justifies this as the required per-trial rewarded-versus-omitted label. Step 12 adds that the labels matched raw reward-event-derived outcomes exactly in the checked sessions.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles a few cases by skipping data rather than repairing them:
- Trials with fewer than 2 retained `scanning > 0` samples are dropped.
- Trials with no non-NaN `reward_zone` values are dropped.
- Sessions with fewer than 2 remaining trials are dropped.
- If `reward_zone` center lookup fails, the median trial position is used as a fallback center.

It does not implement explicit neural/behavior length reconciliation or timestamp-alignment assertions.

ii. ```python
if len(idx) < 2:
    continue
...
rz_vals = rz_trial[~np.isnan(rz_trial)]
if len(rz_vals) == 0:
    continue
...
rz_center = reward_zone_centers.get(rz_mode, float(np.nanmedian(pos_trial)))
...
if len(neural_trials) < 2:
    print(f'  skipping {f} because <2 valid trials')
    continue
```

iii. The notes repeatedly mention checking for valid data periods via `scanning`, keeping at least two trials per session, and handling the earlier reward-zone interpretation issue. No more systematic missing-data policy is documented.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive steps in the implemented code are:
- Reading every NWB file once to collect all `reward_zone` arrays for `infer_reward_zone_map`.
- Reading every NWB file again during `load_session`.
- Loading and concatenating all deconvolved plane arrays per session.
- The per-trial extraction loop over each session.
- Serializing the final large pickle.

ii. ```python
all_rz = []
for f in files:
    with h5py.File(f, 'r') as h:
        all_rz.append(np.array(h['processing/behavior/BehavioralTimeSeries/reward_zone/data'], dtype=np.float32))
reward_zone_value_map = infer_reward_zone_map(np.concatenate(all_rz))
```

```python
for plane in plane_names:
    arr = np.array(h[f'processing/ophys/Deconvolved/{plane}/data'], dtype=np.float32)
    deconv_planes.append(arr)
...
for tr in trial_ids:
    idx = np.where((trial_num == tr) & (scanning > 0))[0]
```

iii. Step 6 explicitly notes loading full session arrays into memory as an efficiency issue, and the trajectory/notes emphasize pickle size reduction as an important practical concern.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious vectorization candidates are:
- The per-trial loop over `trial_ids`, which repeatedly recomputes masks and stacks arrays.
- The loop over unique reward-zone values used to build `reward_zone_centers`.
- The pass over files used only to collect `all_rz`.
- The plane-concatenation loop is mostly I/O-bound but is still a Python loop.

ii. ```python
for zval in np.unique(rz[~np.isnan(rz)]):
    mask = rz == zval
    if np.any(mask):
        reward_zone_centers[zval] = float(np.nanmedian(pos[mask & (speed > -np.inf)]))
...
for tr in trial_ids:
    idx = np.where((trial_num == tr) & (scanning > 0))[0]
```

iii. Step 6 says the AI tried to avoid per-neuron loops and identified full-array loading as the main inefficiency, but it did not document any deeper vectorization plan for these remaining loops.

## 13-c. What processing does the code repeat multiple times?

i. The code makes a full first pass over all files to collect `reward_zone` values for `infer_reward_zone_map`, then a second pass to do the actual conversion. Within `load_session`, it also computes `env_binary` and reward-zone centers even though only part of that work is used downstream.

ii. ```python
for f in files:
    with h5py.File(f, 'r') as h:
        all_rz.append(np.array(h['processing/behavior/BehavioralTimeSeries/reward_zone/data'], dtype=np.float32))
reward_zone_value_map = infer_reward_zone_map(np.concatenate(all_rz))
...
for i, f in enumerate(files):
    neural_trials, input_trials, output_trials, info = load_session(
        f, reward_zone_value_map, show_processing=args.show_processing and i < 2
    )
```

iii. The notes mention separate survey-like exploration and then full conversion as part of the workflow. In the final script, the explicit repeated processing that remains is the all-files reward-zone scan plus the full conversion pass.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several pieces of work are unused in the final exported dataset:
- `infer_reward_zone_map(...)` and the first pass over `all_rz`; the resulting `reward_zone_value_map` is passed into `load_session` but never used there.
- `env_binary` and `env_trial`; they are computed from the `environment` stream but not used in the exported inputs.
- `trial_start`; it is read but unused.
- `nearest_sample`; it is defined but never called.

ii. ```python
def nearest_sample(values, ts, qts):
    ...

def load_session(path, reward_zone_value_map, show_processing=False):
    ...
    env_binary = infer_env_binary(identifier, env)
    env_from_identifier = parse_env_from_identifier(identifier)
    ...
    env_trial = env_binary[q_idx]
```

```python
all_rz = []
for f in files:
    with h5py.File(f, 'r') as h:
        all_rz.append(np.array(h['processing/behavior/BehavioralTimeSeries/reward_zone/data'], dtype=np.float32))
reward_zone_value_map = infer_reward_zone_map(np.concatenate(all_rz))
```

iii. The notes only explicitly acknowledge one inefficiency: loading full arrays and using native samples. The unused `reward_zone_value_map`, unused environment-derived arrays, and dead helper are not documented there; they are visible only in the final code.
