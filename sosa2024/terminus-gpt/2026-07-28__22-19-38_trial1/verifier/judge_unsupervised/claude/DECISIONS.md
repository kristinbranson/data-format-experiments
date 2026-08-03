# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files in the `data/` directory. It discovers all files matching `data/sub-*/*.nwb` using glob, iterates through them one by one with `NWBHDF5IO`, and extracts behavioral time series from `processing['behavior']['BehavioralTimeSeries']` and neural data from `processing['ophys']['Deconvolved']`. Each file represents one session.

ii.
```python
def get_files(sample=False):
    files = sorted(Path('data').glob('sub-*/*.nwb'))
    return files[:2] if sample else files

def process_file(fpath):
    with NWBHDF5IO(str(fpath), 'r', load_namespaces=True) as io:
        nwb = io.read()
        ...
        bts = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series
        deconv = nwb.processing['ophys'].data_interfaces['Deconvolved'].roi_response_series['plane0']
```

iii. The AI documented in CONVERSION_NOTES.md Step 2 that data are NWB files organized by subject directories, with 152 sessions across 11 subjects. In the trajectory, the AI explored the NWB file structure and identified the behavioral and ophys interfaces.

## 1-b. How are the data split into subjects (mice)?

i. Subject identity is extracted from the NWB file's `subject.subject_id` field, falling back to the parent directory name. After processing all sessions, unique subjects are collected and sorted to create the `subjects` list and `subject_idx` mapping.

ii.
```python
subj = getattr(nwb.subject, 'subject_id', fpath.parent.name)
# ...
subjects = sorted({s['subject'] for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The AI noted in Step 2 that there are 11 subjects, consistent with the data directory structure (`sub-m3`, `sub-m4`, `sub-m7`, `sub-m11` through `sub-m19`).

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are processed independently in `process_file()`, and only sessions with at least 2 valid trials are included in the final dataset.

ii.
```python
for i, f in enumerate(files, 1):
    sess = process_file(f)
    if len(sess['neural']) >= 2:
        sessions.append(sess)
```

iii. The AI documented 152 sessions in the full dataset across 11 subjects, consistent with the NWB file count.

## 1-d. How are the data split into trials?

i. Trials are segmented using the `trial_start` behavioral time series. The AI finds indices where `trial_start > 0` (impulse markers), then defines each trial as spanning from one trial_start index to the next (or to the end of the recording for the last trial).

ii.
```python
def trial_bounds_from_trial_start(trial_start):
    starts = np.flatnonzero(trial_start > 0)
    bounds = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(trial_start)
        if e > s:
            bounds.append((s, e))
    return bounds
```

iii. The AI noted in Step 4 that `nwb.trials` is empty, so trial boundaries must be derived from the `trial_start` behavioral time series. The representative session had 80 trial_start impulses matching expected trial counts.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies three trial-level filters: (1) trials shorter than 2 timepoints are skipped, (2) trials where all trial numbers are negative (baseline period) are skipped, and (3) trials where all positions are <= -100 (sentinel values) are skipped. No neuron-level filtering is applied.

ii.
```python
for ti, (s, e) in enumerate(bounds):
    if e - s < 2:
        continue
    if np.nanmax(trial_num[s:e]) < 0:
        continue
    if np.all(position[s:e] <= -100):
        continue
```

iii. The AI documented in Step 5 Key Decision 9 that pre-task baseline samples with trial number -1 or sentinel position values should be excluded. However, the AI did not implement neuron-level quality filtering (e.g., spatial information criteria or place cell selection) described in the reference paper's methods.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the deconvolved calcium imaging traces stored at `processing['ophys']['Deconvolved'].roi_response_series['plane0']` in the NWB files.

ii.
```python
deconv = nwb.processing['ophys'].data_interfaces['Deconvolved'].roi_response_series['plane0']
neural = np.asarray(deconv.data[:], dtype=np.float32)
```

iii. The AI justified this in Step 5 Key Decision 1: "Use deconvolved activity because both methods and NWB organization indicate this is the primary processed neural variable for remapping/GLM analyses."

## 2-b. How is the `neural` data processed?

i. The neural data is loaded as float32, then sliced per trial and transposed from (time, neurons) to (neurons, time) format. No additional processing (normalization, smoothing, filtering) is applied.

ii.
```python
neural = np.asarray(deconv.data[:], dtype=np.float32)
# ...
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. The AI noted in CONVERSION_NOTES.md that the reference paper uses min-max normalization for some analyses and Gaussian smoothing for others, but chose not to apply these transformations for the decoder task, keeping the raw deconvolved traces.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level filtering is applied. All neurons in the deconvolved data are included.

ii.
```python
n_neurons = neural.shape[1]
region_idx = np.zeros(n_neurons, dtype=np.int64)
```

iii. The AI documented neuron curation rules from the paper (spatial information significance, FDE > 0.15) in CONVERSION_NOTES.md Step 3, but did not implement any of these filters, stating the converted dataset has 260,091 total neurons matching the raw data count.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start. Each trial's neural data begins at the `trial_start` index and extends to the next trial start (or end of recording). The alignment event is the first timepoint of each trial.

ii.
```python
trial_neural = neural[s:e, :].T.astype(np.float32)
# s is the trial_start index
```

iii. Documented as Key Decision 2: "Align trials to `trial_start`, matching the decoder task requirement." The metadata records `temporal_alignment_event: 'trial_start'` and `off_start: 0.0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native sampling rate of the deconvolved traces (~15.5 Hz, ~64.5 ms bins) is used directly. No temporal rebinning is applied.

ii.
```python
rate = float(deconv.rate) if getattr(deconv, 'rate', None) is not None else float(1.0 / np.median(np.diff(timestamps)))
# ...
'time_bin_size': float(1000.0 / sessions[0]['rate']) if sessions else None,
```

iii. Key Decision 4: "Use the native shared sampling grid (~15.5 Hz) because behavior and deconvolved traces appear synchronized on the same timestamps."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the deconvolved trace timestamps. The trial start time is subtracted from each timepoint's timestamp within the trial.

ii.
```python
timestamps = np.asarray(deconv.timestamps[:]) if deconv.timestamps is not None else np.arange(neural.shape[0]) / float(deconv.rate)
# ...
rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
```

iii. The AI used deconvolved ophys timestamps as the time base, computing relative time by subtracting the first timestamp in each trial.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Simple subtraction: `timestamps[s:e] - timestamps[s]`, giving time from 0 at trial start in seconds.

ii.
```python
rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
```

iii. No additional processing beyond subtraction.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time values use the same indices as the neural data (same `s:e` slice), so they are inherently aligned.

ii.
```python
rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
inp = np.vstack([rel_time, ...])
```

iii. Both neural and input data use the same trial boundary indices, ensuring one-to-one alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavioral time series in the NWB file.

ii.
```python
env = np.asarray(behavior['environment'][0]).ravel()
# ...
env_valid = env[s:e][env[s:e] >= 0]
env_label = int(np.round(np.median(env_valid))) if len(env_valid) else 0
```

iii. The AI documented in Step 5 that environment codes 0 and 1 correspond to ENV1 and ENV2, with baseline coded as -1.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, valid environment values (>= 0) within the trial window are extracted, and the median is taken (rounded to integer). This per-trial scalar is broadcast across all timepoints in the trial.

ii.
```python
env_valid = env[s:e][env[s:e] >= 0]
env_label = int(np.round(np.median(env_valid))) if len(env_valid) else 0
# ...
np.full(e - s, env_label, dtype=np.float32),
```

iii. The AI treated environment as a per-trial binary variable, consistent with the decoder task specification.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number` behavioral time series in the NWB file.

ii.
```python
trial_num = np.asarray(behavior['trial number'][0]).ravel()
# ...
tr_valid = trial_num[s:e][trial_num[s:e] >= 0]
tr_label = float(np.median(tr_valid)) if len(tr_valid) else float(ti)
```

iii. The AI used the NWB `trial number` variable directly.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. For each trial, valid trial number values (>= 0) are extracted from the trial window, and the median is taken. This per-trial scalar is broadcast across all timepoints. If no valid values exist, the loop index `ti` is used as fallback.

ii.
```python
tr_valid = trial_num[s:e][trial_num[s:e] >= 0]
tr_label = float(np.median(tr_valid)) if len(tr_valid) else float(ti)
np.full(e - s, tr_label, dtype=np.float32),
```

iii. The trial number is used directly from the NWB data as a continuous per-trial variable.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps in the NWB file. For each trial, the AI determines whether the previous trial was rewarded by checking the `reward_outcomes` list accumulated during processing.

ii.
```python
reward_ts = np.asarray(behavior['Reward'][1]).ravel() if behavior['Reward'][1] is not None else np.array([])
# ...
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
reward_outcomes.append(rew)
# ...
prev_rew = reward_outcomes[-2] if len(reward_outcomes) >= 2 else 0
```

iii. The AI documented Key Decision 7: "Mark a trial rewarded if any `Reward` event timestamp falls within that trial." For the first trial, previous outcome defaults to 0.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the AI checks if any reward event timestamp falls within the trial's time window. The outcome of the previous trial in the processing order is then used as the `previous_trial_outcome` input. First trial defaults to 0.

ii.
```python
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
reward_outcomes.append(rew)
prev_rew = reward_outcomes[-2] if len(reward_outcomes) >= 2 else 0
np.full(e - s, prev_rew, dtype=np.float32),
```

iii. The previous trial outcome is binary (0=omitted, 1=rewarded), broadcast across all timepoints.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from two NWB variables: `position` and `reward_zone`. The reward zone identity is determined per trial from `reward_zone` values, mapped to a physical center location, and the distance is computed as `position - center`.

ii.
```python
position = np.asarray(behavior['position'][0]).ravel()
reward_zone = np.asarray(behavior['reward_zone'][0]).ravel()
zone_centers = reward_zone_centers(position, reward_zone)
# ...
d = position[s:e] - zc
```

iii. The AI documented in Step 5 that reward zone codes 1-6 collapse to three physical locations (A/B/C) along the corridor.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The AI first computes global reward zone centers by taking the median position where each reward zone code is active. Then per trial, it identifies the reward zone code (most common nonzero code), maps it to A/B/C using position thresholds (< 150 = A, < 260 = B, else C), and uses hardcoded zone centers (A=90, B=205, C=325) to compute signed distance.

ii.
```python
def reward_zone_centers(position, reward_zone):
    centers = {}
    for code in sorted(c for c in np.unique(reward_zone) if c > 0):
        pos = position[reward_zone == code]
        centers[int(code)] = float(np.median(pos))
    return centers

def collapse_zone_position_to_abc(pos):
    if pos < 150: return 0
    if pos < 260: return 1
    return 2

# In process_file:
zone_center_lookup = {0: 90.0, 1: 205.0, 2: 325.0}
zc = zone_center_lookup[zone_label]
d = position[s:e] - zc
```

iii. The AI noted the reward zone centers are hardcoded at 90, 205, 325 cm based on observed data.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The signed distance is discretized into 7 bins using the thresholds specified in the instructions: < -50 (0), -50 to -10 (1), -10 to 0 (2), exactly 0 (3), 0 to +10 (4), +10 to +50 (5), > +50 (6).

ii.
```python
def distance_bin(d):
    if d < -50: return 0
    if d < -10: return 1
    if d < 0: return 2
    if d == 0: return 3
    if d <= 10: return 4
    if d <= 50: return 5
    return 6
```

iii. The thresholds match the decoder task specification. Note: bin 3 (exactly 0) is virtually never populated since continuous position values rarely equal the exact center.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The distance is computed using the same trial indices `s:e` as the neural data, so they share the same timepoints.

ii.
```python
d = position[s:e] - zc
out_dist = np.array([distance_bin(float(x)) for x in d], dtype=np.int64)
```

iii. Alignment is inherent from using the same index slice.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series in the NWB file.

ii.
```python
position = np.asarray(behavior['position'][0]).ravel()
out_pos = pos_bins(position[s:e])
```

iii. Directly from the NWB position variable.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position values are clipped to [0, 450) and digitized into 5 equal-width bins using `np.linspace(0, 450, 6)` as bin edges.

ii.
```python
def pos_bins(pos, lo=0.0, hi=450.0):
    edges = np.linspace(lo, hi, 6)
    out = np.digitize(np.clip(pos, lo, hi - 1e-6), edges[1:-1], right=False)
    return out.astype(np.int64)
```

iii. The corridor length of 450 cm is consistent with the reference paper's description.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. 5 equal-sized bins: [0, 90), [90, 180), [180, 270), [270, 360), [360, 450).

ii.
```python
edges = np.linspace(lo, hi, 6)  # [0, 90, 180, 270, 360, 450]
out = np.digitize(np.clip(pos, lo, hi - 1e-6), edges[1:-1], right=False)
```

iii. Consistent with the instruction to discretize into 5 equal-sized bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same trial index slice `s:e` is used, ensuring one-to-one alignment with neural timepoints.

ii.
```python
out_pos = pos_bins(position[s:e])
```

iii. Alignment is inherent from shared indexing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series in the NWB file.

ii.
```python
lick = np.asarray(behavior['lick'][0]).ravel()
out_lick = (lick[s:e] > 0).astype(np.int64)
```

iii. The AI noted that lick values in the NWB range from 0-6 and are binarized.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick values are binarized: any value > 0 becomes 1, otherwise 0.

ii.
```python
out_lick = (lick[s:e] > 0).astype(np.int64)
```

iii. Key Decision 6: "Convert lick values >0 to 1 for decoder output."

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same trial index slice `s:e` used for both, ensuring alignment.

ii.
```python
out_lick = (lick[s:e] > 0).astype(np.int64)
```

iii. Inherent alignment from shared indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from `reward_zone` and `position` behavioral time series. The reward zone code active during each trial is mapped to a physical location (A/B/C).

ii.
```python
rz_vals = reward_zone[s:e]
rz_nz = rz_vals[rz_vals > 0]
if len(rz_nz):
    code = int(np.bincount(rz_nz.astype(int)).argmax())
    center = zone_centers.get(code, np.nan)
elif rew:
    ridx = np.searchsorted(timestamps, reward_ts[...][0])
    center = float(position[ridx])
else:
    center = np.nanmedian(position[s:e])
zone_label = collapse_zone_position_to_abc(center if np.isfinite(center) else 225.0)
```

iii. The AI used the most common nonzero reward zone code, with fallback to reward event position or median position.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The most common nonzero `reward_zone` code within the trial is found. Its median position is looked up. That position is classified as A (< 150 cm), B (150-260 cm), or C (>= 260 cm). Output is 0/1/2 per trial, broadcast across timepoints.

ii.
```python
def collapse_zone_position_to_abc(pos):
    if pos < 150: return 0
    if pos < 260: return 1
    return 2

out_rz = np.full(e - s, zone_label, dtype=np.int64)
```

iii. The A/B/C classification is based on position thresholds derived from the AI's data exploration.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` timestamps in the NWB file, checked against the trial's time window.

ii.
```python
reward_ts = np.asarray(behavior['Reward'][1]).ravel()
# ...
t0 = timestamps[s]
t1 = timestamps[e - 1]
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
```

iii. Key Decision 7: "Mark a trial rewarded if any `Reward` event timestamp falls within that trial."

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Binary determination: if any reward event timestamp falls within the trial's time window [t0, t1], the trial is rewarded (1), otherwise not (0). The per-trial scalar is broadcast across timepoints.

ii.
```python
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
out_rew = np.full(e - s, rew, dtype=np.int64)
```

iii. Straightforward binary classification per trial.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: (1) trials with no valid environment codes default to 0, (2) trials with no valid trial numbers use the loop index, (3) trials with no nonzero reward zone codes fall back to reward event position or median position, (4) non-finite zone centers default to 225.0 cm (mid-corridor), (5) first trial's previous outcome defaults to 0.

ii.
```python
env_label = int(np.round(np.median(env_valid))) if len(env_valid) else 0
tr_label = float(np.median(tr_valid)) if len(tr_valid) else float(ti)
zone_label = collapse_zone_position_to_abc(center if np.isfinite(center) else 225.0)
prev_rew = reward_outcomes[-2] if len(reward_outcomes) >= 2 else 0
```

iii. The AI documented some edge case handling in the CONVERSION_NOTES.md but did not extensively discuss missing data scenarios.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is reading NWB files with `NWBHDF5IO`. Each file takes ~0.7-1.1 seconds to process, with larger-neuron sessions (e.g., 3934 neurons) taking longer. The full conversion of 152 sessions takes ~200 seconds. The pickle serialization at the end is also slow for the 25GB output file.

ii.
```python
with NWBHDF5IO(str(fpath), 'r', load_namespaces=True) as io:
    nwb = io.read()
    # ... all processing within the NWB context
```

iii. The AI noted conversion time of ~0.7 s/session in Step 7 and the full run completed in ~201 seconds.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The `distance_bin` and `speed_bin` functions are applied element-wise using Python list comprehensions, which could be vectorized using `np.digitize` or vectorized conditions.

ii.
```python
out_dist = np.array([distance_bin(float(x)) for x in d], dtype=np.int64)
out_speed = np.array([speed_bin(float(max(0.0, x))) for x in speed[s:e]], dtype=np.int64)
```

iii. The AI identified this but did not optimize, as the total runtime was already within acceptable bounds (~3 minutes for full conversion).

## 13-c. What processing does the code repeat multiple times?

i. The `reward_zone_centers` function computes global zone centers once per session, which is appropriate. However, the code reads all behavioral variables upfront even if some trials are filtered out. The `collapse_zone_position_to_abc` and hardcoded `zone_center_lookup` are redundant -- the zone centers from data are computed but then overridden by hardcoded values.

ii.
```python
zone_centers = reward_zone_centers(position, reward_zone)  # computed but not used for distance
zone_center_lookup = {0: 90.0, 1: 205.0, 2: 325.0}  # hardcoded values used instead
```

iii. The `zone_centers` computation is partially wasted since the final distance calculation uses the hardcoded lookup.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `speed_bin` output is computed (output index 2) but the instructions include speed as a decoder output. The `autoreward` and `scanning` behavioral time series are loaded but never used in the conversion logic (they are loaded in the behavior dictionary but not referenced).

ii.
```python
for k in ['trial_start', 'trial number', 'environment', 'reward_zone', 'teleport', 'Reward', 'lick', 'speed', 'position', 'autoreward', 'scanning']:
    behavior[k] = _ts_data(bts[k])
```

iii. Loading `autoreward` and `scanning` adds minor overhead but doesn't affect correctness. The `teleport` variable is also loaded but not used for trial boundary detection (only `trial_start` is used).
