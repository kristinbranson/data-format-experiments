# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All NWB files are found by recursively globbing `data/**/*.nwb`. Each NWB file corresponds to one session. Files are loaded using `h5py` (not `pynwb`). Behavior streams are read from `processing/behavior/BehavioralTimeSeries`, and neural data from `processing/ophys/Deconvolved`. A first pass reads all `reward_zone` data across sessions to build a global reward zone value map, then a second pass loads each session fully.

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))
...
for i, f in enumerate(files):
    neural_trials, input_trials, output_trials, info = load_session(
        f, reward_zone_value_map, show_processing=args.show_processing and i < 2
    )
```

Loading within `load_session`:
```python
with h5py.File(path, 'r') as h:
    identifier = dec(h['identifier'][()])
    subject = dec(h['general/subject/subject_id'][()])
    beh = h['processing/behavior/BehavioralTimeSeries']
    pos = np.array(beh['position/data'], dtype=np.float32)
    ...
    deconv_planes = []
    plane_names = sorted(h['processing/ophys/Deconvolved'].keys())
    for plane in plane_names:
        arr = np.array(h[f'processing/ophys/Deconvolved/{plane}/data'], dtype=np.float32)
        deconv_planes.append(arr)
    neural_full = np.concatenate(deconv_planes, axis=1)
```

iii. The AI uses `h5py` for direct HDF5 access rather than `pynwb`. It globs all NWB files recursively under `data/`. The CONVERSION_NOTES.md documents that 152 sessions across 11 subjects were found, matching the expected data. Two passes are made: one to build a reward zone value map, then one to load and convert.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `general/subject/subject_id` field within each NWB file. Subject names are collected dynamically as sessions are loaded.

ii.
```python
subject = dec(h['general/subject/subject_id'][()])
...
if info['subject'] not in subject_names:
    subject_names.append(info['subject'])
subject_idx.append(subject_names.index(info['subject']))
```

iii. The subject ID is extracted directly from each NWB file's metadata. The order depends on the order files are loaded (sorted glob). 11 subjects are found matching the expected count.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. All NWB files found by globbing are treated as separate sessions.

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))
...
for i, f in enumerate(files):
    ...
    sessions_neural.append(neural_trials)
```

iii. The one-file-per-session mapping is standard for this dataset. 152 sessions total were found.

## 1-d. How are the data split into trials?

i. Trials are split using the `trial number` behavior stream. Unique non-negative trial numbers are extracted, and for each trial, timepoints where `trial_num == tr` and `scanning > 0` are selected.

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

iii. The AI chose to use the `trial number` behavior stream rather than `trial_start`/`teleport` variables. Trials with fewer than 2 valid timepoints are skipped. The `scanning > 0` filter excludes non-scanning periods.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by two criteria: (1) must have at least 2 timepoints where `trial_num == tr` and `scanning > 0`, and (2) must have at least one non-NaN reward zone value within the trial.

ii.
```python
idx = np.where((trial_num == tr) & (scanning > 0))[0]
if len(idx) < 2:
    continue
...
rz_vals = rz_trial[~np.isnan(rz_trial)]
if len(rz_vals) == 0:
    continue
```

Additionally, sessions with fewer than 2 valid trials are skipped:
```python
if len(neural_trials) < 2:
    print(f'  skipping {f} because <2 valid trials')
    continue
```

iii. The minimum of 2 timepoints is a basic quality control. The reward zone filter ensures each trial has identifiable reward zone data. The session filter ensures decoder training is possible.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `Deconvolved` field under `processing/ophys`.

ii.
```python
deconv_planes = []
plane_names = sorted(h['processing/ophys/Deconvolved'].keys())
for plane in plane_names:
    arr = np.array(h[f'processing/ophys/Deconvolved/{plane}/data'], dtype=np.float32)
    deconv_planes.append(arr)
neural_full = np.concatenate(deconv_planes, axis=1)
```

iii. Deconvolved calcium activity is the standard processed neural signal for this type of two-photon imaging data. The CONVERSION_NOTES.md confirms the paper uses deconvolved activity.

## 2-b. How is the `neural` data processed?

i. Deconvolved data from multiple imaging planes are concatenated along the neuron axis. No further processing (e.g., normalization, smoothing) is applied. The final neural arrays are stored as float16 to reduce file size.

ii.
```python
neural_full = np.concatenate(deconv_planes, axis=1)
...
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
```

iii. The AI noted in CONVERSION_NOTES.md that the deconvolved data are already processed. The float16 conversion was done to reduce the pickle file size from ~29GB to ~15GB.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does NOT filter neurons based on the `iscell` variable. All ROIs from the deconvolved data are included, including those not classified as cells.

ii.
```python
for plane in plane_names:
    arr = np.array(h[f'processing/ophys/Deconvolved/{plane}/data'], dtype=np.float32)
    deconv_planes.append(arr)
neural_full = np.concatenate(deconv_planes, axis=1)
```

iii. No justification is provided for omitting iscell filtering. The CONVERSION_NOTES.md mentions iscell in Step 4 ("likely start from valid `iscell` ROIs") but the final code does not implement this filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start by indexing the neural array with the same timepoint indices used for behavior. Since neural and behavior data share the same timepoints in the NWB file, extracting matching indices effectively aligns them.

ii.
```python
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
```

Where `q_idx` are the indices where `trial_num == tr` and `scanning > 0`.

iii. The alignment relies on neural and behavior data being stored at the same sampling rate with shared indexing in the NWB file.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is kept at the native sampling rate. The time bin size is computed as the median inter-sample interval. No temporal rebinning is applied.

ii.
```python
dt = float(np.median(np.diff(pos_t)))
...
'time_bin_size': float(np.median(dts) * 1000.0) if dts else None,
```

iii. The AI does not account for multi-plane scanning when computing the time bin. The raw `dt` from behavior timestamps is used. The reference computes `nplanes/plane_data.rate*1000` to get the effective bin size accounting for multi-plane interleaving.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position` timestamps (`pos_t`).

ii.
```python
pos_t = np.array(beh['position/timestamps'], dtype=np.float64)
...
qts = pos_t[q_idx]
trial_t0 = qts[0]
rel_t = (qts - trial_t0).astype(np.float32)
```

iii. Position timestamps are used as the common time base. These are identical to other behavior stream timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp of the trial is subtracted from all timestamps in the trial.

ii.
```python
trial_t0 = qts[0]
rel_t = (qts - trial_t0).astype(np.float32)
```

iii. Standard approach for computing time relative to trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data use the same indices (`q_idx`), so they are inherently aligned.

ii.
```python
q_idx = idx  # same indices used for neural and behavior
neural_trial = neural_full[q_idx, :].T.astype(np.float16)
...
rel_t = (qts - trial_t0).astype(np.float32)
```

iii. The shared indexing ensures temporal alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Environment type is derived from the session `identifier` string, NOT from the `environment` behavior stream.

ii.
```python
def parse_env_from_identifier(identifier):
    if 'Env2' in identifier:
        return 1
    return 0
...
env_from_identifier = parse_env_from_identifier(identifier)
...
np.full(len(q_idx), float(env_from_identifier), dtype=np.float32),
```

iii. The AI parses the identifier for 'Env2' to determine environment type, making it constant per session. There is also an `infer_env_binary` function defined but the code uses `env_from_identifier` for the actual input construction.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Binary classification: if the session identifier contains 'Env2', the value is 1; otherwise 0. The value is constant for all timepoints in all trials within the session.

ii.
```python
np.full(len(q_idx), float(env_from_identifier), dtype=np.float32),
```

iii. This assumes environment doesn't change within a session.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number` behavior stream values. The raw trial number from the NWB file is used directly.

ii.
```python
trial_num = np.array(beh['trial number/data'], dtype=np.int64)
trial_ids = np.unique(trial_num)
trial_ids = trial_ids[trial_ids >= 0]
for tr in trial_ids:
    ...
    np.full(len(q_idx), float(tr), dtype=np.float32),
```

iii. The raw trial number values from the data stream are used rather than a 0-indexed loop counter.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The raw trial number `tr` is broadcast as a constant across all timepoints in the trial. No transformation is applied.

ii.
```python
np.full(len(q_idx), float(tr), dtype=np.float32),
```

iii. The value is taken directly from the NWB trial number stream.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior time series. Reward event data and timestamps are loaded. The outcome of each trial is determined by whether any positive reward events occurred within the trial's time window.

ii.
```python
reward_event = np.array(beh['Reward/data'], dtype=np.float32)
reward_event_t = np.array(beh['Reward/timestamps'], dtype=np.float64)
...
t_lo, t_hi = qts[0], qts[-1]
reward_in_trial = reward_event[(reward_event_t >= t_lo) & (reward_event_t <= t_hi)]
outcome = int(np.any(reward_in_trial > 0))
```

iii. Reward events with their own timestamps are matched to trial time boundaries.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The `prev_outcome` variable tracks the reward outcome of the previous trial. It is initialized to 0 for the first trial and updated after each trial. The value is constant across all timepoints within a trial.

ii.
```python
prev_outcome = 0
for tr in trial_ids:
    ...
    inp = np.vstack([
        ...
        np.full(len(q_idx), float(prev_outcome), dtype=np.float32),
    ])
    ...
    outcome = int(np.any(reward_in_trial > 0))
    ...
    prev_outcome = outcome
```

iii. The first trial gets 0. Subsequent trials get the outcome of the previous trial in the loop. Note: if a trial is skipped (e.g., due to <2 timepoints), `prev_outcome` is NOT updated, so the next kept trial gets the outcome of the last kept trial rather than the actual previous trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior stream and the `reward_zone` behavior stream. The reward zone center is computed as the median position when the reward zone value equals the trial's mode reward zone value.

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

iii. The AI computes distance from a single center point (median position when in the reward zone), rather than from the zone boundaries.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each trial, the mode reward zone value is determined, and the corresponding center point is looked up. The distance is `position - center`, giving a signed scalar distance from the center. This is then discretized.

ii.
```python
rz_vals = rz_trial[~np.isnan(rz_trial)]
rz_mode = float(Counter(rz_vals.tolist()).most_common(1)[0][0])
rz_center = reward_zone_centers.get(rz_mode, float(np.nanmedian(pos_trial)))
dist = pos_trial - rz_center
dist_bin = discretize_distance(dist)
```

iii. This differs from the reference approach of computing distance to zone BOUNDARIES (where distance is 0 inside the zone). The AI's approach gives distance from a point, so the "0 cm" bin would only contain the exact center point, not the full zone interior.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized using a manual function with explicit boundary conditions into 7 bins.

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

iii. The bin edges follow the instructions. Note that the handling of boundary values (e.g., -10 goes in bin 1 vs bin 2) differs slightly from the reference's `np.digitize` approach.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same indices (`q_idx`) are used for both neural and position data, ensuring alignment.

ii.
```python
pos_trial = pos[q_idx]
...
dist = pos_trial - rz_center
```

iii. Shared indexing ensures alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
pos = np.array(beh['position/data'], dtype=np.float32)
...
pos_trial = pos[q_idx]
```

iii. Direct use of the position variable.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is discretized into 5 bins using `np.linspace` between the session's global min and max position values.

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

iii. The bin edges are computed per-session based on the observed position range. This differs from the reference which uses fixed bin edges across all sessions.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. 5 equal-width bins are computed using `np.linspace(pmin, pmax, 6)` where pmin and pmax are the session's position extremes. The inner edges are used with `np.digitize`.

ii. Same as 8-b.

iii. The verification output shows only bins 2, 3, 4 are ever assigned (`absolute_position_bin: {bin2 (0.356), bin3 (0.351), bin4 (0.293)}`), meaning only 3 of 5 bins are used across the entire dataset. This suggests the per-session min/max approach does not properly span the intended corridor range.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same indices used for neural and position data.

ii. `pos_trial = pos[q_idx]`

iii. Shared indexing ensures alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = np.array(beh['lick/data'], dtype=np.float32)
...
lick_trial = (lick[q_idx] > 0).astype(np.int64)
```

iii. Direct use of the lick variable.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
lick_trial = (lick[q_idx] > 0).astype(np.int64)
```

iii. Matches the instruction for binary lick output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same indices used for neural and lick data.

ii. `lick[q_idx]`

iii. Shared indexing ensures alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the session `identifier` string, NOT from the per-trial `reward_zone` behavior stream. The identifier is parsed for 'LocationA', 'LocationB', or 'LocationC'.

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
np.full(len(q_idx), rz_loc, dtype=np.int64),  # where rz_loc = session_reward_location
```

iii. The AI derives reward zone location from the session identifier, making it constant for all trials within a session.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Simple string matching on the session identifier. The result is a single integer (0, 1, or 2) that is constant for all trials and all timepoints in the session.

ii.
```python
rz_loc = session_reward_location
...
np.full(len(q_idx), rz_loc, dtype=np.int64),
```

iii. This approach assumes the reward zone location does not change across trials within a session. The reference solution uses a Viterbi algorithm to assign per-trial reward zone labels based on position data.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavior time series data and timestamps.

ii.
```python
reward_event = np.array(beh['Reward/data'], dtype=np.float32)
reward_event_t = np.array(beh['Reward/timestamps'], dtype=np.float64)
...
t_lo, t_hi = qts[0], qts[-1]
reward_in_trial = reward_event[(reward_event_t >= t_lo) & (reward_event_t <= t_hi)]
outcome = int(np.any(reward_in_trial > 0))
```

iii. Reward events are matched to trial time boundaries using their own timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, reward events falling within the trial's time window are checked. If any positive reward event exists, the outcome is 1 (rewarded), otherwise 0. The value is constant across all timepoints in the trial.

ii.
```python
t_lo, t_hi = qts[0], qts[-1]
reward_in_trial = reward_event[(reward_event_t >= t_lo) & (reward_event_t <= t_hi)]
outcome = int(np.any(reward_in_trial > 0))
...
np.full(len(q_idx), outcome, dtype=np.int64),
```

iii. Straightforward binary classification of rewarded vs non-rewarded trials.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Short trials**: Trials with fewer than 2 valid timepoints (after scanning filter) are skipped.
- **Missing reward zone data**: Trials where all reward zone values are NaN are skipped entirely.
- **Sessions with too few trials**: Sessions with fewer than 2 valid trials are skipped.
- **NaN handling**: `np.nanmin`/`np.nanmax` used for position range computation.

ii.
```python
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

iii. The handling is defensive but does not address neural/behavior length mismatches (which the reference handles explicitly by cropping to the minimum).

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files via h5py and reading large neural arrays
2. The initial reward zone value map pass (reads all NWB files)
3. The main conversion pass (reads all NWB files again)

ii. N/A

iii. CONVERSION_NOTES.md reports ~0.6s per session for sample conversion.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `load_session` iterates over each trial sequentially. The reward event matching (`(reward_event_t >= t_lo) & (reward_event_t <= t_hi)`) is done per-trial but could potentially be vectorized across trials. The `Counter` call for reward zone mode is per-trial.

ii. N/A

iii. The per-trial loop is natural given variable-length trials.

## 13-c. What processing does the code repeat multiple times?

i. The code reads all NWB files twice: once for the reward zone value map (initial pass over all files to read reward_zone data) and once for the full conversion. The reward zone data and position data are read in both passes.

ii.
```python
# First pass
for f in files:
    with h5py.File(f, 'r') as h:
        all_rz.append(np.array(h['processing/behavior/BehavioralTimeSeries/reward_zone/data'], ...))

# Second pass
for i, f in enumerate(files):
    neural_trials, input_trials, output_trials, info = load_session(f, ...)
```

iii. The first pass is needed to build the global reward zone value map before conversion begins.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `infer_env_binary` function is defined and called (`env_binary = infer_env_binary(identifier, env)`) but its output is never used in the final input construction. Instead, `env_from_identifier` is used. The `reward_zone_value_map` is computed in the first pass but is passed to `load_session` without being used within it. The `reward_zone_centers` computation is done per-session but is used in a suboptimal way (center-based distance rather than boundary-based).

ii.
```python
env_binary = infer_env_binary(identifier, env)  # computed but not used in output
...
reward_zone_value_map = infer_reward_zone_map(np.concatenate(all_rz))  # passed but not used
```

iii. These suggest the code went through iterations where approaches were changed but unused code was left in.
