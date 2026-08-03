# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files found by globbing `data/**/*.nwb`. Each NWB file corresponds to one session for one subject and is opened with `h5py`. Behavioral time series, neural (DfOverF or Fluorescence) data, and ROI metadata are extracted from standard NWB paths within each file.

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))
# ...
for fp in files:
    print('Converting', fp, flush=True)
    sess = convert_session(fp, show_processing=args.show_processing)
```

Inside `convert_session`:
```python
with h5py.File(path, 'r') as f:
    subj = decode_scalar(f['general/subject/subject_id'][()])
    sess = decode_scalar(f['general/session_id'][()])
    b = load_behavior_series(f)
    neural, neural_t, neural_base, neural_key = load_neural_series(f)
    n_rois, regions = load_n_rois_and_regions(f)
```

iii. The AI identified that data are stored as individual NWB files per subject/session under `data/sub-*/`. It chose to iterate over all NWB files, loading each one independently. This approach is documented in CONVERSION_NOTES Steps 2 and 5: "Data are organized as NWB files under `data/sub-*/sub-*_ses-*_behavior+ophys.nwb`."

## 1-b. How are the data split into subjects (mice)?

i. Subject identity is extracted from each NWB file's metadata field `general/subject/subject_id`. After processing all sessions, unique subjects are collected and sorted, and a `subject_idx` array maps each session to a subject index.

ii.
```python
subj = decode_scalar(f['general/subject/subject_id'][()])
# ...
subjects = sorted(set(s['subject'] for s in sessions))
subj_to_idx = {s: i for i, s in enumerate(subjects)}
data = {
    'subjects': subjects,
    'subject_idx': np.asarray([subj_to_idx[s['subject']] for s in sessions], dtype=np.int64),
}
```

iii. The AI noted 11 subjects in CONVERSION_NOTES Step 2, consistent with the paper's report of 11 mice.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session identity is extracted from `general/session_id`. Sessions with fewer than 2 valid trials after processing are skipped. Result: 152 sessions.

ii.
```python
sess = convert_session(fp, show_processing=args.show_processing)
if sess['n_trials'] >= 2:
    sessions.append(sess)
else:
    print('Skipping session with <2 trials after processing:', fp, flush=True)
```

iii. CONVERSION_NOTES Step 9 documents "152 NWB sessions total". The instruction reference requires at least 2 trials per session for decoder evaluation.

## 1-d. How are the data split into trials?

i. Trials are inferred from the behavioral time series. The `infer_trial_bounds` function detects rising edges of the `trial_start` signal and uses `trial number` to identify valid trials (trial_num >= 0). Each trial spans from one trial_start rising edge to the next.

ii.
```python
def infer_trial_bounds(b):
    trial_num = np.asarray(b['trial number'])
    trial_start = np.asarray(b['trial_start'])
    valid = trial_num >= 0
    starts = rising_edges(trial_start, 0.5)
    starts = starts[valid[starts]]
    if len(starts) == 0:
        changes = np.where(np.diff(trial_num) > 0)[0] + 1
        starts = changes[valid[changes]]
    trial_ids = trial_num[starts].astype(int)
    bounds = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(trial_num)
        if e - s > 1:
            bounds.append((int(trial_ids[i]), int(s), int(e)))
    return bounds
```

iii. The AI noted in CONVERSION_NOTES Step 2 that "Trial structure is not stored in a standard NWB `intervals/trials` table in the inspected files, so trial segmentation likely must be reconstructed from behavioral time series variables." The rising-edge approach on `trial_start` is consistent with how the reference code reconstructs trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by two criteria: (1) `trial_num >= 0` (valid trial numbers only), and (2) the trial must have at least 2 neural timepoints within the trial window (`np.sum(nmask) < 2`) and at least 2 behavior samples (`(e - s) < 2`). There is no additional quality filtering based on licking behavior, speed, or other behavioral criteria mentioned in the reference code.

ii.
```python
valid = trial_num >= 0
starts = starts[valid[starts]]
# ...
if np.sum(nmask) < 2 or (e - s) < 2:
    continue
```

iii. The AI does not document any specific trial quality filtering beyond the minimum sample requirements. The reference code's trial filtering (e.g., based on behavioral performance, omission trials, etc.) is not applied here.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the NWB `processing/ophys/DfOverF` group (preferred) or `processing/ophys/Fluorescence` as fallback. The `data` array and `timestamps` array are loaded.

ii.
```python
def load_neural_series(f):
    for base in ['processing/ophys/DfOverF', 'processing/ophys/Fluorescence']:
        if base in f:
            grp = f[base]
            for k in grp.keys():
                g = grp[k]
                if 'data' in g:
                    data = np.asarray(g['data'][:])
                    ts = np.asarray(g['timestamps'][:]) if 'timestamps' in g else None
                    return data, ts, base, k
    raise RuntimeError('No DfOverF or Fluorescence dataset found')
```

iii. The AI noted in CONVERSION_NOTES Step 5: "Prefer NWB `DfOverF` if present to match reference processed activity." This is consistent with the reference code using `dff` (dF/F) as the default activity key.

## 2-b. How is the `neural` data processed?

i. The raw DfOverF data is loaded and oriented to (neurons, time). It is then sliced per-trial based on the temporal overlap with behavioral timestamps. The data is cast to float32. No additional processing (smoothing, z-scoring, baseline subtraction, etc.) is applied.

ii.
```python
if neural.shape[0] == n_rois:
    neural_nt = neural
elif neural.shape[1] == n_rois:
    neural_nt = neural.T
else:
    neural_nt = neural if neural.shape[0] < neural.shape[1] else neural.T
# ...
neural_trial = neural_nt[:, nmask].astype(np.float32)
```

iii. The AI reasoned that since the NWB already contains pre-computed DfOverF, no further normalization was needed. This is reasonable since the reference `dff` function would have been applied before NWB export.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality filtering is applied. All ROIs from the NWB file are included regardless of cell type or recording quality. The reference code contains `is_putative_interneuron` and `get_cell_classes` functions for cell curation, but these were not implemented.

ii.
```python
# No filtering code - all neurons are included:
brain_region_idx = np.zeros(neural_nt.shape[0], dtype=np.int64)
```

iii. CONVERSION_NOTES Step 4 states: "For conversion, start from all valid neural ROIs unless reference code indicates a required exclusion; verify later against methods/code." The AI chose not to implement the reference code's cell curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start. For each trial, the behavioral timestamps at the trial start and end are used to create a temporal mask on the neural timestamps. All neural frames within [t0, t1] (the behavioral time window of the trial) are included.

ii.
```python
t0 = bt[s]
t1 = bt[e - 1]
nmask = (neural_t >= t0) & (neural_t <= t1)
nt = neural_t[nmask]
neural_trial = neural_nt[:, nmask].astype(np.float32)
```

iii. The instructions specify "Temporally align based on start of the trial." The AI uses trial_start behavioral timestamps as the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The native imaging frame rate (~15.5 Hz, ~64.5 ms per frame) is preserved. The `time_bin_size` metadata is set to `NaN`.

ii.
```python
'metadata': {
    'time_bin_size': float(np.nan),
    # ...
}
```

iii. The AI preserves the native temporal resolution of the calcium imaging data. The reference paper reports ~15.5 Hz imaging. Setting `time_bin_size` to NaN rather than computing the actual value (~64.5 ms) is a gap.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the neural timestamps (`neural_t`) and the behavioral timestamp at trial start (`bt[s]`).

ii.
```python
nt = neural_t[nmask]
t0 = bt[s]
time_from_start = (nt - t0).astype(np.float32)
```

iii. The AI uses the behavior timestamp at the trial start index as the reference time, then computes elapsed seconds from the neural timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Simple subtraction: each neural timestamp minus the trial start time. Result is in seconds (float32).

ii.
```python
time_from_start = (nt - t0).astype(np.float32)
```

iii. No additional processing beyond the subtraction.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Perfectly aligned by construction since both use the same neural timestamp array (`nt`).

ii.
```python
nt = neural_t[nmask]
time_from_start = (nt - t0).astype(np.float32)
```

iii. One timepoint per neural frame.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavioral time series in the NWB file.

ii.
```python
env = map_environment(np.asarray(b['environment']))
```

iii. CONVERSION_NOTES Step 5: "Convert to binary ENV1 vs ENV2 per trial."

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The `map_environment` function maps environment values to binary (0/1). It finds unique non-NaN values >= 0 (excluding -1), takes the lowest and highest, and maps the highest to 1 and everything else to 0. Within each trial, the median environment value is taken and broadcast across all timepoints.

ii.
```python
def map_environment(x):
    vals = np.unique(x[np.isfinite(x)])
    vals = [v for v in vals if v >= 0 or v == -1 or v == 1]
    uniq = sorted(set(v for v in vals if v != -1))
    if len(uniq) >= 2:
        lo, hi = uniq[0], uniq[-1]
        return np.where(x == hi, 1, 0)
    return (x > 0).astype(np.int64)
# ...
env_trial = np.full(nt.shape, int(np.round(np.nanmedian(env[sl]))), dtype=np.float32)
```

iii. The AI converts the environment to a per-trial constant binary value, consistent with the instruction's specification of "binary, ENV1 vs ENV2, per trial."

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number` behavioral time series. Specifically, the trial ID from the `infer_trial_bounds` function is used (the value of `trial_num` at each trial start).

ii.
```python
trial_num = np.asarray(b['trial number']).astype(int)
# In infer_trial_bounds:
trial_ids = trial_num[starts].astype(int)
# In convert_session:
trialnum_trial = np.full(nt.shape, tid, dtype=np.float32)
```

iii. The trial number is the native trial numbering from the NWB behavioral data, broadcast as a constant across all timepoints within the trial.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The raw trial number value at each trial start is used directly (as a float32 constant per trial). No renumbering or normalization is applied.

ii.
```python
trialnum_trial = np.full(nt.shape, tid, dtype=np.float32)
```

iii. Straightforward broadcast of the trial ID.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the reward outcome of the preceding trial. Reward outcomes are computed per-trial using the `Reward` event timestamps from the NWB behavioral time series.

ii.
```python
reward_event_timestamps = b.get('Reward__timestamps', np.asarray([]))
reward_outcomes = infer_reward_outcomes_from_events(trials, bt, reward_event_timestamps)
```

iii. CONVERSION_NOTES Step 7: "Reward outcome inference was corrected to use NWB `BehavioralTimeSeries/Reward` event timestamps rather than lick heuristics."

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, check if any reward event timestamps fall within the trial's time window. The previous trial's outcome (0=omitted, 1=rewarded) is stored and carried forward. The first trial uses `prev_out = 0`.

ii.
```python
def infer_reward_outcomes_from_events(trials, behavior_timestamps, reward_event_timestamps):
    outcomes = []
    for _, s, e in trials:
        t0 = behavior_timestamps[s]
        t1 = behavior_timestamps[e - 1]
        rewarded = int(np.any((reward_event_timestamps >= t0) & (reward_event_timestamps <= t1)))
        outcomes.append(rewarded)
    return np.asarray(outcomes, dtype=np.int64)
# ...
prev_out = 0
for i, (tid, s, e) in enumerate(trials):
    # ...
    prev_out_trial = np.full(nt.shape, prev_out, dtype=np.float32)
    # ...
    prev_out = int(reward_outcomes[i])
```

iii. The AI initially used a lick-based heuristic but corrected to use the NWB `Reward` event stream.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` (current position on track) and `reward_zone` behavioral time series. The reward zone center for each trial is inferred from positions where `reward_zone > 0`.

ii.
```python
pos = np.asarray(b['position'])
rz_raw = np.asarray(b['reward_zone'])
trial_reward_pos, reward_loc_centers = infer_trial_reward_positions(pos, rz_raw, trials)
```

iii. The AI computes per-trial reward positions from the overlap of position and reward zone signals.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each trial, positions where `reward_zone > 0` are used to compute the median position as the reward center. These per-trial centers are clustered into 3 canonical locations using KMeans. Distance is then `position - reward_center`.

ii.
```python
def infer_trial_reward_positions(pos, rz_signal, trials):
    trial_reward_pos = {}
    for tid, s, e in trials:
        sl = slice(s, e)
        nz = rz_signal[sl] > 0
        trial_reward_pos[tid] = float(np.nanmedian(pos[sl][nz])) if np.any(nz) else np.nan
    vals = np.array([v for v in trial_reward_pos.values() if np.isfinite(v)])
    # ...
    km = KMeans(n_clusters=k, random_state=0, n_init=20).fit(vals.reshape(-1, 1))
    centers = np.sort(km.cluster_centers_.ravel())
    return trial_reward_pos, centers
```

iii. The AI uses KMeans to identify 3 canonical reward zone positions, consistent with the task having 3 reward locations (A, B, C).

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins based on signed distance from reward zone center: <-50, -50 to -10, -10 to 0, 0, 0 to 10, 10 to 50, >50 cm.

ii.
```python
def discretize_dist_to_reward(pos, reward_center):
    dist = pos - reward_center
    out = np.full(dist.shape, 0, dtype=np.int64)
    out[(dist >= -50) & (dist < -10)] = 1
    out[(dist >= -10) & (dist < 0)] = 2
    out[np.isclose(dist, 0)] = 3
    out[(dist > 0) & (dist <= 10)] = 4
    out[(dist > 10) & (dist <= 50)] = 5
    out[dist > 50] = 6
    out[np.isnan(dist)] = 0
    return out, dist
```

iii. Matches the instruction specification exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Behavioral position is resampled to neural timepoints using `np.searchsorted` to find the nearest behavioral sample for each neural timestamp.

ii.
```python
bidx = np.searchsorted(bt, nt, side='left')
bidx = np.clip(bidx, 0, len(bt) - 1)
# ...
dist_bins, _ = discretize_dist_to_reward(pos[bidx], reward_center)
```

iii. The AI uses nearest-neighbor interpolation from behavioral to neural timestamps.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series.

ii.
```python
pos = np.asarray(b['position'])
```

iii. Direct use of the NWB position variable.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is discretized into 5 equal-sized bins. The bin edges are computed from the min/max of valid positions (where `trial_num >= 0` and `pos > -400`).

ii.
```python
def discretize_position(pos, valid_mask, n_bins=5):
    valid_pos = pos[valid_mask]
    lo, hi = np.nanmin(valid_pos), np.nanmax(valid_pos)
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(pos, edges[1:-1], right=False), 0, n_bins - 1)
    return idx, edges
```

iii. The instruction specifies "discretized into 5 equal-sized bins." The AI computes this from the data range. The reference paper describes the track as 0-450 cm.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. 5 equal bins determined by `np.linspace(lo, hi, 6)` edges from the valid position range, using `np.digitize`.

ii. Same as 8-b above.

iii. Consistent with instruction specification of 5 equal-sized bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same as distance to reward zone: behavioral position is resampled to neural timepoints using `np.searchsorted`.

ii.
```python
abs_pos_bins[bidx]
```

iii. Nearest-neighbor alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series in the NWB file.

ii.
```python
lick = np.asarray(b['lick'])
```

iii. Direct use of the NWB lick variable.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized with a threshold of 0.5: values > 0.5 are mapped to 1 (lick present), otherwise 0.

ii.
```python
(lick[bidx] > 0.5).astype(np.int64)
```

iii. Consistent with instruction: "0 = no, 1 = yes."

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same nearest-neighbor resampling via `np.searchsorted` as other behavioral variables.

ii.
```python
lick[bidx]  # bidx from np.searchsorted(bt, nt, side='left')
```

iii. Aligned to neural timestamps.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from `position` and `reward_zone` behavioral time series. Per-trial reward zone positions are inferred and then clustered into 3 canonical locations using KMeans.

ii.
```python
trial_reward_pos, reward_loc_centers = infer_trial_reward_positions(pos, rz_raw, trials)
trial_reward_loc = assign_reward_location_labels(trial_reward_pos, reward_loc_centers)
# ...
np.full(nt.shape, int(trial_reward_loc.get(tid, 0)), dtype=np.int64)
```

iii. The AI infers reward zone locations dynamically via clustering rather than using known fixed positions from the reference code. CONVERSION_NOTES: "Reward-zone per-trial location is inferred from positions of nonzero `reward_zone` samples and clustered into three canonical locations."

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For each trial, the median position where `reward_zone > 0` is computed. These values are clustered into k=3 groups via KMeans. Each trial is assigned to the nearest cluster center (0=A, 1=B, 2=C). Trials with no reward zone signal default to label 0.

ii.
```python
def assign_reward_location_labels(trial_reward_pos, centers):
    labels = {}
    finite_centers = [c for c in centers if np.isfinite(c)]
    for tid, rp in trial_reward_pos.items():
        if not np.isfinite(rp) or len(finite_centers) == 0:
            labels[tid] = 0
        else:
            labels[tid] = int(np.argmin(np.abs(np.asarray(finite_centers) - rp)))
    return labels
```

iii. Reasonable approach, though the reference code likely has explicit reward zone positions rather than needing clustering.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` event timestamps in the NWB behavioral time series.

ii.
```python
reward_event_timestamps = b.get('Reward__timestamps', np.asarray([]))
reward_outcomes = infer_reward_outcomes_from_events(trials, bt, reward_event_timestamps)
```

iii. CONVERSION_NOTES: "Reward outcome inference was corrected to use NWB `BehavioralTimeSeries/Reward` event timestamps."

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, if any reward event timestamp falls within the trial's behavioral time window [t0, t1], the trial is marked as rewarded (1), otherwise omitted (0).

ii.
```python
def infer_reward_outcomes_from_events(trials, behavior_timestamps, reward_event_timestamps):
    outcomes = []
    for _, s, e in trials:
        t0 = behavior_timestamps[s]
        t1 = behavior_timestamps[e - 1]
        rewarded = int(np.any((reward_event_timestamps >= t0) & (reward_event_timestamps <= t1)))
        outcomes.append(rewarded)
    return np.asarray(outcomes, dtype=np.int64)
```

iii. This produces a binary per-trial outcome (0/1) as specified in the instructions.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- Missing neural timestamps: if no timestamps exist, a linear interpolation is generated from behavioral timestamps.
- Missing reward zone signals: if no reward zone signal is detected within a trial, `np.nan` is used and later defaults to label 0.
- Missing reward event timestamps: defaults to an empty array, meaning no trials are rewarded.
- Trials with too few samples (< 2 neural or behavioral) are skipped.
- NaN values in distance computations are mapped to bin 0 (< -50 cm).
- Invalid trial numbers (< 0) are excluded.

ii.
```python
if neural_t is None:
    neural_t = np.linspace(bt[0], bt[-1], neural_nt.shape[1])
# ...
trial_reward_pos[tid] = float(np.nanmedian(pos[sl][nz])) if np.any(nz) else np.nan
# ...
reward_event_timestamps = b.get('Reward__timestamps', np.asarray([]))
# ...
out[np.isnan(dist)] = 0
```

iii. The AI documented some of these in CONVERSION_NOTES Step 10: handling was iteratively improved during development.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) loading the NWB files from disk (each file can be very large with full neural data), and (2) the KMeans clustering for reward zone location inference per session. The per-trial loop within `convert_session` involves array indexing operations that scale with data size.

ii.
```python
# Large data load:
data = np.asarray(g['data'][:])  # loads entire neural array into memory
# KMeans clustering:
km = KMeans(n_clusters=k, random_state=0, n_init=20).fit(vals.reshape(-1, 1))
```

iii. No timing information is printed despite the instructions requesting it in Step 7.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates sequentially over all trials. The `infer_trial_bounds` loop over starts could potentially be vectorized. The `infer_reward_outcomes_from_events` loop iterates over trials checking reward timestamps.

ii.
```python
for i, (tid, s, e) in enumerate(trials):
    # Per-trial processing: slicing, searchsorted, discretization
```

```python
def infer_reward_outcomes_from_events(trials, behavior_timestamps, reward_event_timestamps):
    outcomes = []
    for _, s, e in trials:
        # ...
```

iii. These loops are relatively straightforward and could be partially vectorized using numpy broadcasting.

## 13-c. What processing does the code repeat multiple times?

i. The `discretize_position` function computes position bin edges from the global valid data, but position binning is applied within the trial loop via `abs_pos_bins[bidx]` (precomputed outside the loop, which is efficient). The KMeans clustering for reward zone locations is only done once per session. No major redundant processing is apparent.

ii. The position discretization is done once outside the trial loop:
```python
abs_pos_bins, pos_edges = discretize_position(pos, valid_mask, n_bins=5)
speed_bins = discretize_speed(speed)
```

iii. The code is reasonably structured to avoid repeated computation.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `discretize_dist_to_reward` function returns both the binned values and the raw distance, but only the binned values are used. The `pos_edges` from `discretize_position` are returned but never used downstream. The `zone_centers` are stored in the session result dict but not in the final output. The `--show-processing` plots are generated for debugging but not used in the final output.

ii.
```python
dist_bins, _ = discretize_dist_to_reward(pos[bidx], reward_center)  # _ is discarded
abs_pos_bins, pos_edges = discretize_position(pos, valid_mask, n_bins=5)  # pos_edges unused
# In session result:
'zone_centers': reward_loc_centers.tolist()  # not in final data dict
```

iii. These are minor inefficiencies; the discarded data is cheap to compute.
