# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads NWB files using `pynwb`. It finds all `.nwb` files under `data/sub-*/` using `pathlib.Path.glob('sub-*/*.nwb')` sorted by path. Each file is opened with `NWBHDF5IO` and processed in a single pass per file. The behavior data comes from `processing['behavior']['BehavioralTimeSeries']` and neural data from `processing['ophys']['Deconvolved']`.

ii.
```python
def get_files(sample=False):
    files = sorted(Path('data').glob('sub-*/*.nwb'))
    return files[:2] if sample else files

# In process_file:
with NWBHDF5IO(str(fpath), 'r', load_namespaces=True) as io:
    nwb = io.read()
    bts = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series
    deconv = nwb.processing['ophys'].data_interfaces['Deconvolved'].roi_response_series['plane0']
```

iii. The AI identified the NWB directory structure and used pynwb to load data. In the trajectory (Step 87), the AI noted that neural data is in `processing['ophys']['Deconvolved']` with `plane0`.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the NWB file's `subject.subject_id` attribute, falling back to the parent directory name. Unique subjects are collected across all sessions.

ii.
```python
subj = getattr(nwb.subject, 'subject_id', fpath.parent.name)
# later:
subjects = sorted({s['subject'] for s in sessions})
```

iii. The AI extracted subject identity from the NWB metadata. The CONVERSION_NOTES confirm 11 subjects were found, matching the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are identified by the `session_id` attribute of the NWB file.

ii.
```python
sess_id = nwb.session_id
```

iii. The CONVERSION_NOTES document 152 session files across 11 subjects.

## 1-d. How are the data split into trials?

i. Trials are segmented using only the `trial_start` behavior time series. Each trial starts at a `trial_start > 0` index and ends at the next `trial_start` index (or end of recording for the last trial). The `teleport` signal is NOT used for trial end detection.

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

iii. The AI noted in CONVERSION_NOTES Step 4 that trial boundaries come from `trial_start` events. The AI did not use the `teleport` signal that the reference uses to define trial ends.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by three criteria: (1) fewer than 2 timepoints, (2) `trial_num` all negative, and (3) position all <= -100 cm. Sessions with fewer than 2 valid trials are excluded.

ii.
```python
for ti, (s, e) in enumerate(bounds):
    if e - s < 2:
        continue
    if np.nanmax(trial_num[s:e]) < 0:
        continue
    if np.all(position[s:e] <= -100):
        continue
# later:
if len(sess['neural']) >= 2:
    sessions.append(sess)
```

iii. The AI applied multiple heuristic filters. The reference uses a minimum of 50 timepoints instead of 2.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `Deconvolved` ophys processing module, specifically only `plane0`.

ii.
```python
deconv = nwb.processing['ophys'].data_interfaces['Deconvolved'].roi_response_series['plane0']
neural = np.asarray(deconv.data[:], dtype=np.float32)
```

iii. The AI noted in CONVERSION_NOTES that deconvolved activity is the primary neural signal, consistent with the paper's methods.

## 2-b. How is the `neural` data processed?

i. The deconvolved data from `plane0` is loaded directly and cast to float32. No additional processing (no filtering, no normalization, no multi-plane concatenation) is applied.

ii.
```python
neural = np.asarray(deconv.data[:], dtype=np.float32)
# ...
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. The AI chose to use the raw deconvolved data directly. No iscell filtering or multi-plane handling was applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality filtering is applied. The AI does NOT use the `iscell` variable to filter neurons. All ROIs from plane0 are included regardless of their classification.

ii.
```python
deconv = nwb.processing['ophys'].data_interfaces['Deconvolved'].roi_response_series['plane0']
neural = np.asarray(deconv.data[:], dtype=np.float32)
# No iscell filtering
```

iii. The AI's CONVERSION_NOTES mention neuron curation rules from the paper (SI significance, FDE > 0.15) but these were not implemented in the code. The `iscell` filter used in the reference is absent.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The data is aligned to trial start by slicing the neural array from trial start index to trial end index.

ii.
```python
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. The instructions specify alignment to trial start, which the AI implements by indexing from the trial start boundary.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is derived from the rate of the `plane0` deconvolved data as `1000.0 / rate`. No temporal rebinning is applied. The AI does NOT account for multi-plane imaging (the reference divides by rate and multiplies by nplanes).

ii.
```python
rate = float(deconv.rate) if getattr(deconv, 'rate', None) is not None else float(1.0 / np.median(np.diff(timestamps)))
# In metadata:
'time_bin_size': float(1000.0 / sessions[0]['rate']) if sessions else None,
```

iii. The AI uses the native sampling rate without rebinning. However, the time bin calculation differs from the reference which accounts for multi-plane scanning: `nplanes/plane_data.rate*1000`.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `timestamps` of the deconvolved neural data (or constructed from rate if timestamps are absent).

ii.
```python
timestamps = np.asarray(deconv.timestamps[:]) if deconv.timestamps is not None else np.arange(neural.shape[0]) / float(deconv.rate)
# ...
rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
```

iii. The AI uses neural timestamps rather than behavior timestamps. Since both should be on the same time grid, this is functionally equivalent.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The trial start timestamp is subtracted from all timestamps within the trial.

ii.
```python
rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
```

iii. Standard approach matching the reference.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both neural and behavior data use the same timestamp indices (both indexed by `s:e`), so they are inherently aligned.

ii.
```python
trial_neural = neural[s:e, :].T.astype(np.float32)
rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
```

iii. The AI's CONVERSION_NOTES note that behavior and deconvolved traces share the same timestamps.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
env = np.asarray(behavior['environment'][0]).ravel()
# ...
env_valid = env[s:e][env[s:e] >= 0]
env_label = int(np.round(np.median(env_valid))) if len(env_valid) else 0
```

iii. The AI identified the `environment` variable as containing the ENV1/ENV2 labels.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI computes the median of valid (>= 0) environment values within each trial and rounds to get a single per-trial label. This is broadcast as a constant across all timepoints. The reference simply uses the raw per-timepoint environment values directly.

ii.
```python
env_valid = env[s:e][env[s:e] >= 0]
env_label = int(np.round(np.median(env_valid))) if len(env_valid) else 0
inp = np.vstack([
    ...
    np.full(e - s, env_label, dtype=np.float32),
    ...
])
```

iii. The AI excluded baseline (-1) values and took the median. Since environment is constant within a trial, this should produce the same result in most cases.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number` behavior time series in the NWB file, with fallback to the loop index.

ii.
```python
trial_num = np.asarray(behavior['trial number'][0]).ravel()
# ...
tr_valid = trial_num[s:e][trial_num[s:e] >= 0]
tr_label = float(np.median(tr_valid)) if len(tr_valid) else float(ti)
```

iii. The AI used the stored trial number variable, unlike the reference which uses the loop counter (0, 1, 2, ...).

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The AI takes the median of valid (>= 0) `trial number` values within each trial. Falls back to the loop index `ti` if no valid values exist.

ii.
```python
tr_valid = trial_num[s:e][trial_num[s:e] >= 0]
tr_label = float(np.median(tr_valid)) if len(tr_valid) else float(ti)
```

iii. The reference uses the simple loop counter. The AI's approach uses the NWB-stored trial number which may differ.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps.

ii.
```python
reward_ts = np.asarray(behavior['Reward'][1]).ravel() if behavior['Reward'][1] is not None else np.array([])
# ...
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
reward_outcomes.append(rew)
```

iii. The AI uses reward event timestamps to determine per-trial reward outcome.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the reward outcome is computed as whether any reward timestamp falls within the trial's time range. The previous trial outcome is then taken from the reward_outcomes list (index -2, since the current trial's outcome has already been appended). For the first trial, defaults to 0.

ii.
```python
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
reward_outcomes.append(rew)
# ...
prev_rew = reward_outcomes[-2] if len(reward_outcomes) >= 2 else 0
```

iii. This is functionally similar to the reference, though the reference uses a pre-computed `isreward` array indexed by trial boundaries and looks at the actual previous trial (including skipped trials), while the AI only considers previously processed (non-skipped) trials.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the `reward_zone` behavior time series. Reward zone centers are computed from the median position where `reward_zone > 0` for each code, then collapsed to A/B/C based on position thresholds.

ii.
```python
def reward_zone_centers(position, reward_zone):
    centers = {}
    for code in sorted(c for c in np.unique(reward_zone) if c > 0):
        pos = position[reward_zone == code]
        pos = pos[np.isfinite(pos)]
        if len(pos):
            centers[int(code)] = float(np.median(pos))
    return centers

def collapse_zone_position_to_abc(pos):
    if pos < 150:
        return 0  # A
    if pos < 260:
        return 1  # B
    return 2  # C
```

iii. The AI computed zone centers from the data rather than using fixed zone ranges from the reference code.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Distance is computed as `position - zone_center` where zone_center is a fixed lookup (90.0, 205.0, 325.0). This is distance to the center of the zone, NOT distance to the nearest edge as in the reference.

ii.
```python
zone_center_lookup = {0: 90.0, 1: 205.0, 2: 325.0}
zc = zone_center_lookup[zone_label]
d = position[s:e] - zc
out_dist = np.array([distance_bin(float(x)) for x in d], dtype=np.int64)
```

iii. The reference computes signed distance to the nearest edge of the reward zone range (e.g., [80, 130] for zone A), with 0 when inside the zone. The AI computes distance to a single center point, which will never be exactly 0 and will produce different bin assignments.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Uses a per-element function `distance_bin` with if/else thresholds matching the instruction specification.

ii.
```python
def distance_bin(d):
    if d < -50: return 0
    if d < -10: return 1
    if d < 0:   return 2
    if d == 0:  return 3
    if d <= 10: return 4
    if d <= 50: return 5
    return 6
```

iii. The bin boundaries match the instructions. However, because the distance computation differs (center-based vs edge-based), the bin assignments will differ. Also, `d == 0` will essentially never be true for center-based distance, so bin 3 will be nearly empty.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position data uses the same indices `s:e` as the neural data, so alignment is inherent.

ii.
```python
d = position[s:e] - zc
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. Same indexing ensures alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = np.asarray(behavior['position'][0]).ravel()
out_pos = pos_bins(position[s:e])
```

iii. Standard extraction of position data.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450) and digitized into 5 bins using `np.linspace(0, 450, 6)` which gives edges [0, 90, 180, 270, 360, 450].

ii.
```python
def pos_bins(pos, lo=0.0, hi=450.0):
    edges = np.linspace(lo, hi, 6)
    out = np.digitize(np.clip(pos, lo, hi - 1e-6), edges[1:-1], right=False)
    return out.astype(np.int64)
```

iii. The bin edges [0, 90, 180, 270, 360, 450] differ significantly from the reference [-inf, 50, 150, 250, 350, inf]. The reference uses 100 cm wide bins centered differently.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five bins with edges at [0, 90, 180, 270, 360, 450]. Position is clipped to the [0, 450) range before binning.

ii.
```python
edges = np.linspace(lo, hi, 6)  # [0, 90, 180, 270, 360, 450]
out = np.digitize(np.clip(pos, lo, hi - 1e-6), edges[1:-1], right=False)
```

iii. The reference uses bins [-inf, 50, 150, 250, 350, inf] which handle the full range of positions including values below 0 and above 450 without clipping. The AI's bins are 90 cm wide, while the reference's bins are 100 cm wide.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same trial indices `s:e` ensure alignment with neural data.

ii.
```python
out_pos = pos_bins(position[s:e])
trial_neural = neural[s:e, :].T.astype(np.float32)
```

iii. Same indexing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = np.asarray(behavior['lick'][0]).ravel()
out_lick = (lick[s:e] > 0).astype(np.int64)
```

iii. Standard extraction.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value maps to 1, else 0.

ii.
```python
out_lick = (lick[s:e] > 0).astype(np.int64)
```

iii. Matches the reference approach and instruction specification.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same trial indices `s:e` ensure alignment.

ii.
```python
out_lick = (lick[s:e] > 0).astype(np.int64)
```

iii. Same indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` behavior time series and `position`. The raw reward_zone codes (1-6) are mapped to physical positions via median position, then collapsed to A/B/C using position thresholds.

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

iii. The AI maps reward zone codes to physical positions and then to A/B/C labels using position thresholds. The reference uses a Viterbi algorithm on the start position when reward_zone > 0 to assign zone labels.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For each trial: (1) find the most common reward_zone code in the trial, (2) look up its median position from the session-wide zone_centers, (3) collapse to A/B/C based on position thresholds (< 150 = A, < 260 = B, else C). Fallback logic handles trials with no reward zone activity or no reward.

ii. See 10-a code snippets.

iii. The reference uses a more sophisticated Viterbi approach that encourages temporal consistency in zone assignments. The AI's approach is simpler and per-trial.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps.

ii.
```python
reward_ts = np.asarray(behavior['Reward'][1]).ravel()
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
out_rew = np.full(e - s, rew, dtype=np.int64)
```

iii. Standard approach using reward event timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, checks whether any reward timestamp falls within the trial's time range [t0, t1 + epsilon]. The outcome is broadcast as a constant across all timepoints.

ii.
```python
t0 = timestamps[s]
t1 = timestamps[e - 1]
rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
out_rew = np.full(e - s, rew, dtype=np.int64)
```

iii. Similar to the reference which maps reward timestamps to indices and checks `np.any(isreward[idx])`.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Short trials**: Trials with fewer than 2 timepoints are skipped.
- **Invalid trial numbers**: Trials where all `trial_num` values are negative are skipped.
- **Invalid positions**: Trials where all positions are <= -100 are skipped.
- **Missing reward zone**: Falls back to reward position or median position if no reward_zone values are present.
- **Missing environment**: Falls back to 0 if no valid environment values.
- **Session with too few trials**: Sessions with fewer than 2 valid trials are excluded.

ii.
```python
if e - s < 2: continue
if np.nanmax(trial_num[s:e]) < 0: continue
if np.all(position[s:e] <= -100): continue
# ...
if len(sess['neural']) >= 2:
    sessions.append(sess)
```

iii. The AI applied several heuristic filters. The reference uses a single filter (< 50 timepoints) and handles neural/behavior length mismatches by cropping.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading and reading NWB files with `NWBHDF5IO`. Each file requires reading large neural data arrays. The AI processes all 152 files sequentially.

ii.
```python
with NWBHDF5IO(str(fpath), 'r', load_namespaces=True) as io:
    nwb = io.read()
```

iii. The AI's CONVERSION_NOTES estimate ~0.7 s/session for a total of ~2-3 minutes for 152 sessions.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The `distance_bin` and `speed_bin` functions use per-element Python loops via list comprehensions, which could be vectorized with `np.digitize`.

ii.
```python
out_dist = np.array([distance_bin(float(x)) for x in d], dtype=np.int64)
out_speed = np.array([speed_bin(float(max(0.0, x))) for x in speed[s:e]], dtype=np.int64)
```

iii. The reference uses `np.digitize` for both distance and speed binning, which is vectorized and faster.

## 13-c. What processing does the code repeat multiple times?

i. The code does not repeat processing across multiple passes. Unlike the reference which has a separate survey step that loads all NWB files before conversion, the AI loads each file only once.

ii. N/A

iii. The AI's single-pass approach is more efficient than the reference's two-pass (survey + convert) approach.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads all behavior time series including `autoreward` and `scanning` which are not used in the output. The `infer_region` function reads ImageSegmentation metadata that could be hard-coded as 'CA1'.

ii.
```python
for k in ['trial_start', 'trial number', 'environment', 'reward_zone', 'teleport', 'Reward', 'lick', 'speed', 'position', 'autoreward', 'scanning']:
    behavior[k] = _ts_data(bts[k])
```

iii. Loading unused variables adds minor overhead.
