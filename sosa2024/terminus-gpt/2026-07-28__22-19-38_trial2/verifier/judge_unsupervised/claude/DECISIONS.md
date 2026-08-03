# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files found by globbing `data/**/*.nwb`. Each NWB file is opened with `h5py` and behavioral time series are read from `processing/behavior/BehavioralTimeSeries`, while neural data is read from `processing/ophys/Deconvolved`. Before processing sessions, a preliminary pass reads the `reward_zone` stream from all files to build a global reward-zone value map.

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
```

```python
with h5py.File(path, 'r') as h:
    identifier = dec(h['identifier'][()])
    subject = dec(h['general/subject/subject_id'][()])
    beh = h['processing/behavior/BehavioralTimeSeries']
    pos = np.array(beh['position/data'], dtype=np.float32)
    pos_t = np.array(beh['position/timestamps'], dtype=np.float64)
    speed = np.array(beh['speed/data'], dtype=np.float32)
    lick = np.array(beh['lick/data'], dtype=np.float32)
    # ... (loads env, rz, trial_num, trial_start, scanning, reward events)
```

iii. The agent explored the NWB file structure in Steps 1-2 and identified the relevant behavioral and neural data paths. The approach loads every NWB file sequentially.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by reading the `general/subject/subject_id` field from each NWB file. A running list of unique subject names is maintained, and each session is mapped to its subject via `subject_idx`.

ii.
```python
subject = dec(h['general/subject/subject_id'][()])
# ...
if info['subject'] not in subject_names:
    subject_names.append(info['subject'])
subject_idx.append(subject_names.index(info['subject']))
```

iii. The agent uses the subject ID embedded in each NWB file. The final dataset has 11 subjects (m3, m4, m7, m11-m15, m17-m19).

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. All 152 NWB files are loaded as separate sessions.

ii.
```python
for i, f in enumerate(files):
    neural_trials, input_trials, output_trials, info = load_session(f, reward_zone_value_map, ...)
    if len(neural_trials) < 2:
        print(f'  skipping {f} because <2 valid trials')
        continue
    sessions_neural.append(neural_trials)
```

iii. The agent determined from data exploration that each NWB file is a separate session. Sessions with fewer than 2 valid trials are skipped (none were skipped in practice).

## 1-d. How are the data split into trials?

i. Trials are identified using the `trial number` behavioral time series. Unique non-negative trial IDs are extracted, and for each trial, timepoints are selected where `trial_num == tr` AND `scanning > 0`.

ii.
```python
trial_ids = np.unique(trial_num)
trial_ids = trial_ids[trial_ids >= 0]

for tr in trial_ids:
    idx = np.where((trial_num == tr) & (scanning > 0))[0]
    if len(idx) < 2:
        continue
```

iii. The agent recognized that trial numbers are stored as a continuous time series and that `scanning > 0` marks valid imaging periods. Negative trial numbers (pre-scanning) are excluded.

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded if they have fewer than 2 valid timepoints (where `scanning > 0` and `trial_num == tr`). Trials where all reward_zone values are NaN are also skipped. No other trial-level filtering is applied (e.g., no filtering by lick behavior, running speed, or other criteria from the reference code).

ii.
```python
if len(idx) < 2:
    continue
# ...
rz_vals = rz_trial[~np.isnan(rz_trial)]
if len(rz_vals) == 0:
    continue
```

iii. The agent applied minimal trial filtering. The CONVERSION_NOTES mention that analysis-specific filtering (e.g., significance criteria for remapping) was intentionally not applied to the decoder dataset.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `processing/ophys/Deconvolved/plane0/data` (deconvolved calcium activity). All planes are concatenated, though in practice only `plane0` exists in all files.

ii.
```python
deconv_planes = []
plane_names = sorted(h['processing/ophys/Deconvolved'].keys())
for plane in plane_names:
    arr = np.array(h[f'processing/ophys/Deconvolved/{plane}/data'], dtype=np.float32)
    deconv_planes.append(arr)
neural_full = np.concatenate(deconv_planes, axis=1)
```

iii. The agent chose deconvolved activity because "both methods text and NWB contents indicate this is the processed neural signal used in analyses."

## 2-b. How is the `neural` data processed?

i. The deconvolved data is loaded as float32, concatenated across planes, then for each trial the relevant timepoints are extracted and the matrix is transposed from (time, neurons) to (neurons, time). The final arrays are stored as float16 to reduce file size.

ii.
```python
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
```

iii. The agent did not apply any additional processing (no smoothing, no normalization, no delta F/F computation). The float16 conversion was done to reduce the pickle file from ~29GB to ~15GB.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neuron filtering is applied.** All ROIs from the deconvolved matrices are included, regardless of the `iscell` classification. The NWB files contain an `iscell` field in `ImageSegmentation/PlaneSegmentation` that classifies ROIs as cells vs non-cells, but this was not used. The dataset contains 312,110 total ROIs across all sessions, whereas only ~138,678 (44%) are classified as cells by Suite2p.

ii.
```python
# No iscell filtering anywhere in the code
neural_full = np.concatenate(deconv_planes, axis=1)  # uses ALL ROIs
brain_region_idx = np.zeros(neural_full.shape[1], dtype=np.int16)
```

iii. The agent explicitly considered iscell filtering but chose not to apply it, stating: "that changes dataset semantics and should be justified carefully; dtype optimization is the safer first move." The CONVERSION_NOTES acknowledge this as a deferred decision.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the start of each trial. The first valid timepoint (where `scanning > 0` and `trial_num == tr`) serves as the trial start. Neural data is directly indexed from the full deconvolved matrix using these timepoint indices.

ii.
```python
idx = np.where((trial_num == tr) & (scanning > 0))[0]
q_idx = idx
qts = pos_t[q_idx]
trial_t0 = qts[0]
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
```

iii. The agent used the behavioral timestamps to identify trial boundaries and directly sliced the neural data. Since behavioral and neural data share the same sampling rate (~15.5 Hz), no interpolation or resampling was needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging sampling rate (~15.5 Hz, dt ~64.5 ms) is used directly. No temporal rebinning is applied. The time bin size is stored in metadata as `float(np.median(dts) * 1000.0)`.

ii.
```python
dt = float(np.median(np.diff(pos_t)))
# ...
'time_bin_size': float(np.median(dts) * 1000.0) if dts else None,
```

iii. The agent noted "no explicit rebinning beyond native samples" and chose to keep the native frame rate. This is reasonable given the instruction to use the same processing as described in the reference.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the behavioral `position/timestamps` array. For each trial, the timestamps of valid timepoints are extracted, and the relative time is computed by subtracting the first timestamp.

ii.
```python
pos_t = np.array(beh['position/timestamps'], dtype=np.float64)
# ...
qts = pos_t[q_idx]
trial_t0 = qts[0]
rel_t = (qts - trial_t0).astype(np.float32)
```

iii. The agent used behavioral timestamps rather than computing time from frame indices and sampling rate, which gives slightly more accurate timing.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first valid timestamp of each trial is subtracted from all timestamps in that trial, giving time relative to trial start in seconds. Cast to float32.

ii.
```python
rel_t = (qts - trial_t0).astype(np.float32)
```

iii. Straightforward subtraction. No additional processing.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both use the same timepoint indices (`q_idx`), so they are inherently aligned. Each timepoint in the neural array corresponds to the same timepoint in the time input.

ii.
```python
q_idx = idx  # same indices used for neural_trial and rel_t
neural_trial = neural_full[q_idx, :].T
rel_t = (qts - trial_t0).astype(np.float32)
inp = np.vstack([rel_t, ...])
```

iii. Same frame indices guarantee alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Environment type is derived from the session `identifier` string (e.g., containing "Env1" or "Env2"), NOT from the `environment` behavioral time series. The `environment` stream in the NWB files only contains values -1 (pre-scanning) and 0 (during scanning), making it uninformative for distinguishing environments.

ii.
```python
def parse_env_from_identifier(identifier):
    if 'Env2' in identifier:
        return 1
    return 0

env_from_identifier = parse_env_from_identifier(identifier)
# Used as:
np.full(len(q_idx), float(env_from_identifier), dtype=np.float32)
```

iii. The agent initially tried using the `environment` behavior stream but found it only contains -1/0 values. The identifier-based approach was adopted "for robustness."

## 4-b. What processing is involved in computing `input` *Environment type*?

i. A simple string check of the session identifier for "Env2". If found, environment=1 (ENV2), otherwise environment=0 (ENV1). This value is constant for the entire session and broadcast across all timepoints.

ii.
```python
np.full(len(q_idx), float(env_from_identifier), dtype=np.float32)
```

iii. No complex processing; just string parsing.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number` behavioral time series. The unique trial IDs from this stream are used directly.

ii.
```python
trial_num = np.array(beh['trial number/data'], dtype=np.int64)
trial_ids = np.unique(trial_num)
trial_ids = trial_ids[trial_ids >= 0]
# ...
for tr in trial_ids:
    # ...
    np.full(len(q_idx), float(tr), dtype=np.float32)
```

iii. The raw trial number from the data is used directly as the input variable.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial number is used as-is from the `trial number` stream (values 0 to ~99). It is broadcast as a constant across all timepoints within each trial.

ii.
```python
np.full(len(q_idx), float(tr), dtype=np.float32)
```

iii. No transformation; the raw trial number is used directly.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` event time series. For each trial, reward events within the trial's time window are checked to determine if a reward was delivered.

ii.
```python
reward_event = np.array(beh['Reward/data'], dtype=np.float32)
reward_event_t = np.array(beh['Reward/timestamps'], dtype=np.float64)
# ...
t_lo, t_hi = qts[0], qts[-1]
reward_in_trial = reward_event[(reward_event_t >= t_lo) & (reward_event_t <= t_hi)]
outcome = int(np.any(reward_in_trial > 0))
```

iii. The agent computes reward outcome for each trial, then uses the previous trial's outcome as the current trial's input.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A running variable `prev_outcome` is initialized to 0 at the start of each session. After processing each trial, `prev_outcome` is updated to the current trial's outcome. The current trial uses the previous trial's outcome as its input.

ii.
```python
prev_outcome = 0
for tr in trial_ids:
    # ... compute outcome for current trial
    inp = np.vstack([..., np.full(len(q_idx), float(prev_outcome), dtype=np.float32)])
    # ...
    prev_outcome = outcome
```

iii. First trial of each session defaults to 0 (no previous reward). This is a reasonable default.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` and `reward_zone` behavioral time series. The reward zone center is computed from the median position at timepoints where the `reward_zone` stream has a specific value.

ii.
```python
pos = np.array(beh['position/data'], dtype=np.float32)
rz = np.array(beh['reward_zone/data'], dtype=np.float32)
```

iii. The agent used both position and reward_zone streams to compute distance.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each session, the code computes reward zone "centers" by finding the median position for each unique `reward_zone` value. Per trial, the most common `reward_zone` value (mode) is used to look up the center, and distance is computed as `position - center`.

**Critical issue**: The `reward_zone` stream values are 0-6, where 0 means "NOT in reward zone" (most timepoints) and 1-6 mean "IN the reward zone." Since rz=0 is the dominant value in nearly every trial, the mode is almost always 0, and `reward_zone_centers[0]` is the median position across the whole corridor (~200 cm), NOT the actual reward zone center (~320-350 cm). This leads to systematically incorrect distance computations.

ii.
```python
reward_zone_centers = {}
for zval in np.unique(rz[~np.isnan(rz)]):
    mask = rz == zval
    if np.any(mask):
        reward_zone_centers[zval] = float(np.nanmedian(pos[mask & (speed > -np.inf)]))

# Per trial:
rz_mode = float(Counter(rz_vals.tolist()).most_common(1)[0][0])
rz_center = reward_zone_centers.get(rz_mode, float(np.nanmedian(pos_trial)))
dist = pos_trial - rz_center
```

iii. The agent attempted to infer reward zone position from the data streams but misinterpreted the `reward_zone` values. The mode-based approach selects rz=0 (not-in-reward-zone) for most trials, giving an incorrect center position.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins matching the instruction specification:

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

iii. The bin edges match the instructions exactly: <-50, -50 to -10, -10 to <0, 0, >0 to 10, 10 to 50, >50.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Uses the same timepoint indices (`q_idx`) as neural data, so alignment is inherent.

ii.
```python
pos_trial = pos[q_idx]
dist = pos_trial - rz_center
dist_bin = discretize_distance(dist)
```

iii. Same frame-level alignment as all other variables.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from `position` behavioral time series.

ii.
```python
pos = np.array(beh['position/data'], dtype=np.float32)
pos_trial = pos[q_idx]
```

iii. Direct use of the position stream.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is discretized into 5 equal-sized bins using the global min/max position across the **entire** position stream (including non-scanning periods). The global min is -500 (artifact from teleport/non-scanning) and max is ~450, giving bin edges at approximately [-500, -310, -120, 70, 260, 450]. Since valid trial positions only range from ~-50 to ~450, only bins 2, 3, and 4 are ever populated.

ii.
```python
abs_pos_global_min = float(np.nanmin(pos))  # -500 (includes non-scanning!)
abs_pos_global_max = float(np.nanmax(pos))  # ~450

def discretize_abs_position(pos, pmin, pmax):
    edges = np.linspace(pmin, pmax, 6)
    bins = np.digitize(pos, edges[1:-1], right=False)
    bins = np.clip(bins, 0, 4)
    return bins.astype(np.int64)
```

iii. The agent acknowledged the issue during validation ("absolute_position_bin occupies bins 2-4 in this sample") but decided it "may reflect partial occupancy of the full corridor." This is incorrect -- the position range used includes non-valid values (-500 from non-scanning periods), causing only 3 of 5 bins to be used.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. 5 equal-sized bins based on global min/max. Due to the bug above, effective bin edges are approximately [-500, -310, -120, 70, 260, 450], and valid data only falls in bins 2-4.

ii.
```python
edges = np.linspace(pmin, pmax, 6)  # pmin=-500, pmax=~450
bins = np.digitize(pos, edges[1:-1], right=False)
bins = np.clip(bins, 0, 4)
```

iii. The verification output confirms: `absolute_position_bin: {bin2 (0.356), bin3 (0.351), bin4 (0.293)}`. Bins 0 and 1 are never used.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same timepoint indices as neural data.

ii.
```python
abs_pos_bin = discretize_abs_position(pos_trial, abs_pos_global_min, abs_pos_global_max)
```

iii. Aligned via shared `q_idx` indexing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series.

ii.
```python
lick = np.array(beh['lick/data'], dtype=np.float32)
```

iii. Direct use of the lick stream.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The raw lick values (0-6, representing lick counts per frame) are binarized: any value > 0 becomes 1 (yes), 0 stays 0 (no).

ii.
```python
lick_trial = (lick[q_idx] > 0).astype(np.int64)
```

iii. Simple binarization matching the instruction specification (0 = no, 1 = yes).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same timepoint indices as neural data.

ii.
```python
lick_trial = (lick[q_idx] > 0).astype(np.int64)
```

iii. Aligned via shared `q_idx` indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the session `identifier` string (e.g., "LocationA", "LocationB", "LocationC"). NOT from the `reward_zone` behavioral stream.

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

session_reward_location = parse_location_from_identifier(identifier)
rz_loc = session_reward_location
```

iii. The agent initially tried using the `reward_zone` stream but found the values (0-6) don't directly encode A/B/C locations. Switched to identifier parsing.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. String matching on the session identifier to map LocationA/B/C to 0/1/2. This value is constant per session and broadcast across all timepoints in every trial.

ii.
```python
np.full(len(q_idx), rz_loc, dtype=np.int64)
```

iii. Per-trial (constant within session) variable. Default to 0 (LocationA) if no location is found in the identifier.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` event time series (same as used for previous trial outcome input).

ii.
```python
reward_event = np.array(beh['Reward/data'], dtype=np.float32)
reward_event_t = np.array(beh['Reward/timestamps'], dtype=np.float64)
```

iii. Event-based reward delivery data with timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the code checks whether any reward events (with value > 0) fall within the trial's time window. If yes, outcome = 1; otherwise, outcome = 0. This is broadcast across all timepoints.

ii.
```python
t_lo, t_hi = qts[0], qts[-1]
reward_in_trial = reward_event[(reward_event_t >= t_lo) & (reward_event_t <= t_hi)]
outcome = int(np.any(reward_in_trial > 0))
np.full(len(q_idx), outcome, dtype=np.int64)
```

iii. The reward event values are all 0.004 (likely mL of reward), so any value > 0 indicates reward delivery. Class balance is ~84.4% rewarded, ~15.6% omitted.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles several edge cases:
- Trials with fewer than 2 valid scanning timepoints are skipped
- Trials where all reward_zone values are NaN are skipped
- NaN values in reward_zone are filtered when computing mode
- Sessions with fewer than 2 valid trials would be skipped (though none are in practice)
- Position -500 during non-scanning periods is NOT explicitly handled (it leaks into global min/max)

ii.
```python
if len(idx) < 2:
    continue
rz_vals = rz_trial[~np.isnan(rz_trial)]
if len(rz_vals) == 0:
    continue
```

iii. The agent applied basic data quality checks but did not handle all edge cases. The position -500 artifact during non-scanning periods affects the absolute position binning.

## 13-a. What are the most time-consuming steps of the code?

i. Loading and reading the NWB files is the main bottleneck. Each session takes 0.1-0.7 seconds to load (depending on file size). Full conversion of 152 sessions takes approximately 1-2 minutes. The preliminary pass to read all reward_zone data adds overhead.

ii.
```python
all_rz = []
for f in files:
    with h5py.File(f, 'r') as h:
        all_rz.append(np.array(h['processing/behavior/BehavioralTimeSeries/reward_zone/data'], dtype=np.float32))
```

iii. The agent noted load times per session and estimated full conversion time at ~1-2 minutes, which is reasonable.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-level loop within each session iterates over trials sequentially. Some operations (e.g., computing reward zone mode, discretizing arrays) are already vectorized within each trial but the outer loop over trials could potentially be replaced with more vectorized operations using advanced indexing.

ii.
```python
for tr in trial_ids:
    idx = np.where((trial_num == tr) & (scanning > 0))[0]
    # ... per-trial processing
```

iii. The per-trial loop is the main non-vectorized structure, but since trial boundaries are irregular, full vectorization would be complex and the current runtime is acceptable.

## 13-c. What processing does the code repeat multiple times?

i. The reward_zone data is read twice: once in the preliminary pass to build the global value map, and once during session loading. The `infer_env_binary` function is called but its result is never used (the identifier-based environment is used instead).

ii.
```python
# First pass:
all_rz.append(np.array(h['processing/behavior/BehavioralTimeSeries/reward_zone/data'], ...))
# Second pass (in load_session):
rz = np.array(beh['reward_zone/data'], dtype=np.float32)

# Unused:
env_binary = infer_env_binary(identifier, env)  # computed but not used
```

iii. The preliminary reward_zone pass is a design choice for building a global mapping. The unused `infer_env_binary` call is dead code left from an earlier approach.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations are performed but not used:
1. `infer_env_binary()` result is computed but unused (identifier-based env is used instead)
2. `reward_zone_value_map` is passed to `load_session` but only used indirectly; the actual center computation uses per-session reward_zone_centers
3. The `speed > -np.inf` condition in reward_zone_centers computation is always true and serves no purpose

ii.
```python
env_binary = infer_env_binary(identifier, env)  # unused
reward_zone_value_map = infer_reward_zone_map(np.concatenate(all_rz))  # passed but barely used
reward_zone_centers[zval] = float(np.nanmedian(pos[mask & (speed > -np.inf)]))  # speed > -inf is trivially true
```

iii. These are artifacts of iterative development where the agent changed approaches without cleaning up old code.
