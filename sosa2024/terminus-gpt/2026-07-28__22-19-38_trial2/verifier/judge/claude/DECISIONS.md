# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `h5py` to directly read NWB files (rather than `pynwb`). It finds all `.nwb` files via `Path('data').rglob('*.nwb')` and iterates over them. For sampling mode, only the first 2 files are used. Before loading sessions, a preliminary pass reads all `reward_zone` data to build a global reward zone value map.

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))
if args.sample:
    files = files[:2]

all_rz = []
for f in files:
    with h5py.File(f, 'r') as h:
        all_rz.append(np.array(h['processing/behavior/BehavioralTimeSeries/reward_zone/data'], dtype=np.float32))
reward_zone_value_map = infer_reward_zone_map(np.concatenate(all_rz))

for i, f in enumerate(files):
    neural_trials, input_trials, output_trials, info = load_session(
        f, reward_zone_value_map, ...
    )
```

iii. The AI chose h5py for direct HDF5 access rather than pynwb. The glob pattern `*.nwb` finds all NWB files across subject subdirectories. The preliminary pass collects reward zone values for mapping.

## 1-b. How are the data split into subjects?

i. Subjects are identified by reading `general/subject/subject_id` from each NWB file. Subject names are accumulated dynamically as sessions are processed.

ii.
```python
subject = dec(h['general/subject/subject_id'][()])
...
if info['subject'] not in subject_names:
    subject_names.append(info['subject'])
subject_idx.append(subject_names.index(info['subject']))
```

iii. Subject identity is extracted from the NWB file metadata rather than from directory names. This is functionally equivalent but relies on internal NWB metadata.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session, same as the reference approach.

ii.
```python
for i, f in enumerate(files):
    neural_trials, input_trials, output_trials, info = load_session(f, ...)
    sessions_neural.append(neural_trials)
```

iii. The one-file-per-session structure is consistent with the data organization.

## 1-d. How are the data split into trials?

i. Trials are segmented using the `trial number` behavior stream. Unique non-negative trial IDs are extracted, and data points matching each trial ID (with `scanning > 0`) form each trial.

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

iii. The AI uses the `trial number` stream combined with `scanning > 0` to define trial boundaries. The reference uses `trial_start` and `teleport` signals to find trial start/end indices.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered if: (1) fewer than 2 valid time points after applying `scanning > 0`, (2) no non-NaN reward zone values exist in the trial. Sessions with fewer than 2 valid trials are also skipped.

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

iii. The minimum trial length threshold (2) is much lower than the reference (50). Additionally, the `scanning > 0` filter restricts to time points when scanning was active.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `processing/ophys/Deconvolved/*/data` — the deconvolved calcium activity across all imaging planes.

ii.
```python
deconv_planes = []
plane_names = sorted(h['processing/ophys/Deconvolved'].keys())
for plane in plane_names:
    arr = np.array(h[f'processing/ophys/Deconvolved/{plane}/data'], dtype=np.float32)
    deconv_planes.append(arr)
neural_full = np.concatenate(deconv_planes, axis=1)
```

iii. Same source as reference — deconvolved activity.

## 2-b. How is the `neural` data processed?

i. Deconvolved data from all planes is concatenated along the neuron axis. The data is cast to float32 during loading, and later to float16 when stored per-trial. No additional processing (normalization, smoothing) is applied.

ii.
```python
neural_full = np.concatenate(deconv_planes, axis=1)
...
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
```

iii. The float16 cast reduces memory but may lose precision. No other processing beyond concatenation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No `iscell` filtering is applied.** All ROIs from the deconvolved data are included, regardless of whether they are classified as cells. The AI simply concatenates all planes without checking `iscell`.

ii.
```python
for plane in plane_names:
    arr = np.array(h[f'processing/ophys/Deconvolved/{plane}/data'], dtype=np.float32)
    deconv_planes.append(arr)
neural_full = np.concatenate(deconv_planes, axis=1)
```

iii. The CONVERSION_NOTES.md mentions that cell filtering was considered (Step 4 discusses `iscell`), but the final code does not implement it. The reference code filters using `iscell`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is indexed by the same trial indices as behavior data (`q_idx`), which are determined by `trial_num == tr` and `scanning > 0`. This aligns neural and behavioral data to the start of each trial.

ii.
```python
q_idx = idx  # indices where trial_num == tr and scanning > 0
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
```

iii. The alignment relies on neural and behavioral data sharing the same time index in the NWB file.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The native sampling rate is preserved, computed as the median of inter-sample intervals: `dt = float(np.median(np.diff(pos_t)))`. The time bin size is stored in metadata in ms.

ii.
```python
dt = float(np.median(np.diff(pos_t)))
...
'time_bin_size': float(np.median(dts) * 1000.0) if dts else None,
```

iii. Same approach as reference — native rate with no rebinning.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position/timestamps` array (`pos_t`), which provides timestamps for each behavior sample.

ii.
```python
pos_t = np.array(beh['position/timestamps'], dtype=np.float64)
...
qts = pos_t[q_idx]
trial_t0 = qts[0]
rel_t = (qts - trial_t0).astype(np.float32)
```

iii. The reference uses `trial number` timestamps but these should be identical.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp of the trial is subtracted from all timestamps in the trial.

ii.
```python
trial_t0 = qts[0]
rel_t = (qts - trial_t0).astype(np.float32)
```

iii. Standard approach matching the reference.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Same indices (`q_idx`) are used for both neural and behavioral data, so they are inherently aligned.

ii.
```python
q_idx = idx
qts = pos_t[q_idx]
neural_trial = neural_full[q_idx, :].T
```

iii. Both share the same sample indices.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the session identifier string (e.g., containing "Env2"), NOT from the `environment` behavior stream. The `infer_env_binary` function exists but is overridden by `env_from_identifier`.

ii.
```python
def parse_env_from_identifier(identifier):
    if 'Env2' in identifier:
        return 1
    return 0
...
env_from_identifier = parse_env_from_identifier(identifier)
...
inp = np.vstack([
    ...
    np.full(len(q_idx), float(env_from_identifier), dtype=np.float32),
    ...
])
```

iii. The AI parses environment type from the NWB identifier string rather than from the per-timepoint `environment` data stream. This makes environment constant per session.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. String matching on the session identifier: if "Env2" is found, environment = 1, otherwise 0. The value is constant for the entire session (all trials, all timepoints).

ii.
```python
def parse_env_from_identifier(identifier):
    if 'Env2' in identifier:
        return 1
    return 0
```

iii. This is a simpler but less flexible approach than using the behavior stream.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number` behavior stream in the NWB file. The raw trial number value `tr` is used directly.

ii.
```python
trial_num = np.array(beh['trial number/data'], dtype=np.int64)
...
for tr in trial_ids:
    ...
    np.full(len(q_idx), float(tr), dtype=np.float32),
```

iii. The AI uses the NWB `trial number` values, which are the stored trial IDs. The reference uses a 0-indexed loop counter instead.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The raw trial number from the NWB stream is broadcast as a constant across all timepoints in the trial. No renumbering or offsetting.

ii.
```python
np.full(len(q_idx), float(tr), dtype=np.float32),
```

iii. The values are the original NWB trial numbers, not reindexed.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from `Reward/data` and `Reward/timestamps` behavior streams.

ii.
```python
reward_event = np.array(beh['Reward/data'], dtype=np.float32)
reward_event_t = np.array(beh['Reward/timestamps'], dtype=np.float64)
```

iii. Same source as reference.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A running variable `prev_outcome` tracks the reward outcome of the previous trial. For the first trial, it is 0. For subsequent trials, it is the outcome of the immediately preceding trial. The outcome is determined by checking if any reward event with value > 0 occurred during the trial's time range.

ii.
```python
prev_outcome = 0
...
for tr in trial_ids:
    ...
    t_lo, t_hi = qts[0], qts[-1]
    reward_in_trial = reward_event[(reward_event_t >= t_lo) & (reward_event_t <= t_hi)]
    outcome = int(np.any(reward_in_trial > 0))
    ...
    np.full(len(q_idx), float(prev_outcome), dtype=np.float32),
    ...
    prev_outcome = outcome
```

iii. Same logic as reference. Note that if a trial is skipped (filtered out), `prev_outcome` is NOT updated for that trial, which differs from the reference where all trials update the reward tracking even if filtered.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` and `reward_zone` behavior streams. The reward zone center is computed as the median position when the mouse is in the reward zone.

ii.
```python
reward_zone_centers = {}
for zval in np.unique(rz[~np.isnan(rz)]):
    mask = rz == zval
    if np.any(mask):
        reward_zone_centers[zval] = float(np.nanmedian(pos[mask & (speed > -np.inf)]))
...
rz_center = reward_zone_centers.get(rz_mode, float(np.nanmedian(pos_trial)))
dist = pos_trial - rz_center
```

iii. The AI computes distance to the center of the reward zone (median position when in zone), while the reference computes signed distance to the nearest boundary of a fixed reward zone range.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each trial, the mode of the `reward_zone` values is found. The reward zone center is looked up from `reward_zone_centers`. Then distance = position - center. This produces a simple signed offset from center.

ii.
```python
rz_mode = float(Counter(rz_vals.tolist()).most_common(1)[0][0])
rz_center = reward_zone_centers.get(rz_mode, float(np.nanmedian(pos_trial)))
dist = pos_trial - rz_center
```

iii. The reference computes distance as signed distance to the nearest edge of the zone boundaries (0 when inside the zone). The AI computes distance to center (never 0 unless exactly at center). This is a fundamentally different interpretation.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous distance is discretized into 7 bins matching the instruction specification.

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

iii. The bin boundaries match the instructions. However, since distance-to-center is used (rather than distance-to-edge), bin 3 (`== 0`) would almost never be hit, and the distribution across bins would differ from the reference.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same trial indices as neural data (`q_idx`).

ii.
```python
pos_trial = pos[q_idx]
dist = pos_trial - rz_center
```

iii. Consistent with the shared indexing approach.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior stream.

ii.
```python
pos = np.array(beh['position/data'], dtype=np.float32)
...
pos_trial = pos[q_idx]
```

iii. Same source as reference.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extraction and discretization.

ii.
```python
abs_pos_bin = discretize_abs_position(pos_trial, abs_pos_global_min, abs_pos_global_max)
```

iii. Same as reference in concept.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The corridor is divided into 5 bins using `np.linspace(pmin, pmax, 6)` where `pmin` and `pmax` are the global min and max of position across the entire session. This creates data-driven bin edges.

ii.
```python
def discretize_abs_position(pos, pmin, pmax):
    edges = np.linspace(pmin, pmax, 6)
    bins = np.digitize(pos, edges[1:-1], right=False)
    bins = np.clip(bins, 0, 4)
    return bins.astype(np.int64)
```

iii. The reference uses fixed bin edges `[-inf, 50, 150, 250, 350, inf]`. The AI's approach uses data-driven edges which vary per session (based on global min/max of position in that session). This means bin boundaries differ across sessions and from the reference.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same trial indices as neural data (`q_idx`).

ii.
```python
pos_trial = pos[q_idx]
```

iii. Consistent shared indexing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior stream.

ii.
```python
lick = np.array(beh['lick/data'], dtype=np.float32)
...
lick_trial = (lick[q_idx] > 0).astype(np.int64)
```

iii. Same source as reference.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any lick value > 0 maps to 1, otherwise 0.

ii.
```python
lick_trial = (lick[q_idx] > 0).astype(np.int64)
```

iii. Same logic as reference.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same trial indices as neural data (`q_idx`).

ii.
```python
lick_trial = (lick[q_idx] > 0).astype(np.int64)
```

iii. Consistent shared indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB session identifier string (e.g., "LocationA", "LocationB", "LocationC"). The `reward_zone` behavior stream is read but only used for the distance computation, not for determining the A/B/C label.

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
session_reward_location = parse_location_from_identifier(identifier)
...
np.full(len(q_idx), rz_loc, dtype=np.int64),  # rz_loc = session_reward_location
```

iii. The AI assigns a single reward zone location per session based on the identifier string. The reference determines per-trial reward zone labels using the Viterbi algorithm on position and reward_zone data, because reward zones can change within a session.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Simple string matching on the identifier. The reward zone is constant for the entire session.

ii.
```python
session_reward_location = parse_location_from_identifier(identifier)
...
np.full(len(q_idx), rz_loc, dtype=np.int64),
```

iii. This assumes reward zone doesn't change within a session. The reference handles within-session changes via Viterbi segmentation.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from `Reward/data` and `Reward/timestamps` behavior streams.

ii.
```python
reward_event = np.array(beh['Reward/data'], dtype=np.float32)
reward_event_t = np.array(beh['Reward/timestamps'], dtype=np.float64)
```

iii. Same source as reference.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, reward events within the trial's time range are identified. If any reward event value > 0, the trial outcome is 1 (rewarded), otherwise 0.

ii.
```python
t_lo, t_hi = qts[0], qts[-1]
reward_in_trial = reward_event[(reward_event_t >= t_lo) & (reward_event_t <= t_hi)]
outcome = int(np.any(reward_in_trial > 0))
```

iii. Similar logic to reference. Reference uses index-based lookup while AI uses timestamp-based filtering.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Short trials**: Trials with fewer than 2 timepoints (after scanning filter) are skipped.
- **Missing reward zone**: Trials where all `reward_zone` values are NaN are skipped.
- **Sessions with few trials**: Sessions with fewer than 2 valid trials are skipped.
- **No explicit neural/behavior length mismatch handling**: The code assumes neural and behavior arrays share indexing.

ii.
```python
if len(idx) < 2:
    continue
...
if len(rz_vals) == 0:
    continue
...
if len(neural_trials) < 2:
    print(f'  skipping {f} because <2 valid trials')
    continue
```

iii. The minimum timepoints threshold (2) is much lower than the reference (50). The reference also explicitly handles neural/behavior length mismatches with cropping.

## 13-a. What are the most time-consuming steps of the code?

i. Loading NWB files is the dominant cost. Each file requires reading large neural arrays from disk. The preliminary pass to read all reward zone data adds overhead.

ii.
```python
# preliminary pass
for f in files:
    with h5py.File(f, 'r') as h:
        all_rz.append(np.array(h['processing/behavior/BehavioralTimeSeries/reward_zone/data'], dtype=np.float32))
# main pass
for i, f in enumerate(files):
    neural_trials, input_trials, output_trials, info = load_session(f, ...)
```

iii. h5py may be slightly faster than pynwb for raw data access.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop within `load_session` iterates over each trial sequentially. Operations like discretization and distance computation are already vectorized within each trial but the trial loop itself is sequential.

ii.
```python
for tr in trial_ids:
    idx = np.where((trial_num == tr) & (scanning > 0))[0]
    ...
```

iii. Variable-length trials make full vectorization impractical.

## 13-c. What processing does the code repeat multiple times?

i. The code reads `reward_zone` data twice: once in the preliminary pass to build the global reward zone map, and again within `load_session`. Position data is also processed multiple times (for reward zone center computation and for distance computation).

ii.
```python
# First pass
all_rz.append(np.array(h['processing/behavior/BehavioralTimeSeries/reward_zone/data'], ...))
# Second pass in load_session
rz = np.array(beh['reward_zone/data'], dtype=np.float32)
```

iii. The preliminary pass is needed for the global reward zone value map before session processing begins.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `infer_env_binary` function is defined and called (`env_binary = infer_env_binary(identifier, env)`) but its result is never used — the code uses `env_from_identifier` instead. The `reward_zone_value_map` parameter is passed to `load_session` but never used inside the function.

ii.
```python
env_binary = infer_env_binary(identifier, env)  # computed but unused
env_from_identifier = parse_env_from_identifier(identifier)  # this is what's actually used
...
def load_session(path, reward_zone_value_map, ...):  # reward_zone_value_map never referenced
```

iii. Dead code from earlier iterations that was not cleaned up.
