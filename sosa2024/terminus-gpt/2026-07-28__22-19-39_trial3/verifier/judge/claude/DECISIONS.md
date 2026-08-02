# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by globbing all `.nwb` files under `data/` using `Path('data').rglob('*.nwb')`. Each NWB file is opened with `h5py.File` (not `pynwb`). Behavior variables are read from `processing/behavior/BehavioralTimeSeries` and neural data from `processing/ophys/DfOverF` or `processing/ophys/Fluorescence`.

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))
...
def convert_session(path, show_processing=False):
    with h5py.File(path, 'r') as f:
        subj = decode_scalar(f['general/subject/subject_id'][()])
        sess = decode_scalar(f['general/session_id'][()])
        b = load_behavior_series(f)
        neural, neural_t, neural_base, neural_key = load_neural_series(f)
        n_rois, regions = load_n_rois_and_regions(f)
```

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

iii. The AI's CONVERSION_NOTES.md states it used NWB files as the data source (Step 2/5), explored the NWB structure with h5py, and chose DfOverF as the neural activity source based on its planning in Step 5: "Use processed imaging activity rather than raw fluorescence when available: Reference code includes dff processing, so NWB DfOverF should be preferred if present."

## 1-b. How are the data split into subjects?

i. Subjects are identified from each session's NWB metadata field `general/subject/subject_id`. After all sessions are converted, unique subjects are sorted and indexed.

ii.
```python
subj = decode_scalar(f['general/subject/subject_id'][()])
...
subjects = sorted(set(s['subject'] for s in sessions))
subj_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The AI reads subject identity from the NWB file metadata rather than parsing directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session ID is extracted from the NWB metadata field `general/session_id`.

ii.
```python
sess = decode_scalar(f['general/session_id'][()])
...
for fp in files:
    print('Converting', fp, flush=True)
    sess = convert_session(fp, show_processing=args.show_processing)
    if sess['n_trials'] >= 2:
        sessions.append(sess)
```

iii. The AI infers session identity from the NWB metadata directly rather than parsing filenames.

## 1-d. How are the data split into trials?

i. Trials are identified using `infer_trial_bounds()`, which finds rising edges of the `trial_start` signal (filtered to where `trial number >= 0`). Trial end is defined as the start of the next trial (or end of recording), not by the teleport signal. Trials with fewer than 2 timepoints are skipped.

ii.
```python
def infer_trial_bounds(b):
    trial_num = np.asarray(b['trial number'])
    trial_start = np.asarray(b['trial_start'])
    valid = trial_num >= 0
    starts = rising_edges(trial_start, 0.5)
    starts = starts[valid[starts]]
    ...
    trial_ids = trial_num[starts].astype(int)
    bounds = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(trial_num)
        if e - s > 1:
            bounds.append((int(trial_ids[i]), int(s), int(e)))
    return bounds
```

iii. The AI's CONVERSION_NOTES (Step 4) notes that trial structure must be reconstructed from behavioral time series variables in NWB. It uses rising edges of `trial_start` combined with `trial number >= 0` for validity.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 neural or behavior timepoints are skipped. Sessions with fewer than 2 remaining trials are also excluded.

ii.
```python
if np.sum(nmask) < 2 or (e - s) < 2:
    continue
...
if sess['n_trials'] >= 2:
    sessions.append(sess)
```

iii. The AI uses minimal filtering -- only excluding extremely short trials and sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `processing/ophys/DfOverF` (preferred) or `processing/ophys/Fluorescence` as a fallback. This is NOT the `Deconvolved` data.

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
                    ...
                    return data, ts, base, k
    raise RuntimeError('No DfOverF or Fluorescence dataset found')
```

iii. The AI's CONVERSION_NOTES Step 5 states: "Use processed imaging activity rather than raw fluorescence when available: Reference code includes dff processing, so NWB DfOverF should be preferred if present." The AI did not consider the `Deconvolved` data stream.

## 2-b. How is the `neural` data processed?

i. The neural data is loaded as-is from the NWB file. Orientation is determined by comparing dimensions to the number of ROIs from `ImageSegmentation`. No further processing (e.g., combining planes, iscell filtering) is performed.

ii.
```python
if n_rois is not None:
    if neural.shape[0] == n_rois:
        neural_nt = neural
    elif neural.shape[1] == n_rois:
        neural_nt = neural.T
    else:
        neural_nt = neural if neural.shape[0] < neural.shape[1] else neural.T
else:
    neural_nt = neural if neural.shape[0] < neural.shape[1] else neural.T
```

iii. The AI simply orients the matrix and does not perform any plane-combining or additional processing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No filtering by `iscell` or any other quality metric is applied. All ROIs from the loaded neural series are included.

ii. No filtering code exists; the full neural matrix is used directly:
```python
neural_trial = neural_nt[:, nmask].astype(np.float32)
```

iii. The AI's CONVERSION_NOTES Step 4 notes: "Cell curation: Code contains cell classification/interneuron logic... For conversion, start from all valid neural ROIs unless reference code indicates a required exclusion." The AI chose not to apply iscell filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. For each trial, the AI finds neural timepoints whose timestamps fall within the trial's behavior timestamps (t0 to t1). Neural data is then indexed using a boolean mask on neural timestamps.

ii.
```python
t0 = bt[s]
t1 = bt[e - 1]
nmask = (neural_t >= t0) & (neural_t <= t1)
...
neural_trial = neural_nt[:, nmask].astype(np.float32)
```

iii. The AI uses timestamp-based alignment rather than assuming neural and behavior share the same sample indices.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The `time_bin_size` in metadata is set to `NaN`. The data is kept at whatever resolution the loaded neural data has.

ii.
```python
'time_bin_size': float(np.nan),
```

iii. The AI's CONVERSION_NOTES does not discuss temporal rebinning. The time_bin_size is left as NaN because it was not computed from the neural data rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the neural timestamps (`neural_t`) after masking to the trial interval. The start time `t0` comes from the behavior timestamps at trial start.

ii.
```python
nt = neural_t[nmask]
...
time_from_start = (nt - t0).astype(np.float32)
```

iii. The AI uses neural timestamps rather than behavior timestamps for computing time from trial start.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The trial start time (`t0 = bt[s]`) is subtracted from each neural timestamp within the trial.

ii.
```python
time_from_start = (nt - t0).astype(np.float32)
```

iii. Straightforward subtraction of trial start time.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The time-from-start values are computed from the same neural timestamps used for the neural data, so they are inherently aligned.

ii.
```python
nt = neural_t[nmask]
time_from_start = (nt - t0).astype(np.float32)
```

iii. Same timestamps used for both.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
env = map_environment(np.asarray(b['environment']))
```

iii. The AI identified the `environment` variable in the NWB behavior data.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The `map_environment()` function maps environment values to binary 0/1. It finds unique non-NaN values, excludes -1, and maps the highest value to 1 and others to 0. A per-trial median is then used to get a single environment value per trial.

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
...
env_trial = np.full(nt.shape, int(np.round(np.nanmedian(env[sl]))), dtype=np.float32)
```

iii. The AI processed the environment variable through a mapping function and then took the per-trial median.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the NWB `trial number` variable. The trial ID (`tid`) is read from `trial number` at the trial start index.

ii.
```python
trial_ids = trial_num[starts].astype(int)
...
trialnum_trial = np.full(nt.shape, tid, dtype=np.float32)
```

iii. The AI uses the NWB-stored trial number rather than a sequential loop counter.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial number from the NWB file is broadcast as a constant across all timepoints in the trial.

ii.
```python
trialnum_trial = np.full(nt.shape, tid, dtype=np.float32)
```

iii. No additional processing beyond broadcasting.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps (`Reward__timestamps`). Reward outcomes are inferred per-trial by checking whether any reward event timestamp falls within the trial's time range.

ii.
```python
reward_event_timestamps = b.get('Reward__timestamps', np.asarray([]))
reward_outcomes = infer_reward_outcomes_from_events(trials, bt, reward_event_timestamps)
```

iii. The AI's CONVERSION_NOTES Step 7 notes: "Reward outcome inference was corrected to use NWB BehavioralTimeSeries/Reward event timestamps rather than lick heuristics."

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The previous trial's reward outcome is used as the current trial's previous trial outcome. For the first trial, it defaults to 0. The value is constant across all timepoints in the trial.

ii.
```python
prev_out = 0
for i, (tid, s, e) in enumerate(trials):
    ...
    prev_out_trial = np.full(nt.shape, prev_out, dtype=np.float32)
    ...
    prev_out = int(reward_outcomes[i])
```

iii. The AI tracks `prev_out` as a running variable updated after each trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` and reward zone center. The reward zone center is the median position when `reward_zone > 0`, clustered across trials using KMeans with k=3.

ii.
```python
def infer_trial_reward_positions(pos, rz_signal, trials):
    trial_reward_pos = {}
    for tid, s, e in trials:
        sl = slice(s, e)
        nz = rz_signal[sl] > 0
        trial_reward_pos[tid] = float(np.nanmedian(pos[sl][nz])) if np.any(nz) else np.nan
    vals = np.array([v for v in trial_reward_pos.values() if np.isfinite(v)])
    ...
    km = KMeans(n_clusters=k, random_state=0, n_init=20).fit(vals.reshape(-1, 1))
    centers = np.sort(km.cluster_centers_.ravel())
    return trial_reward_pos, centers
```

iii. The AI determined reward zone positions empirically from the data using KMeans clustering of median positions during reward zone occupancy.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Distance is computed as `position - reward_center`, where `reward_center` is a single point (the median position in the reward zone for that trial). This is NOT distance to the nearest edge of a reward zone range.

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

iii. The AI used a single reward center point rather than a reward zone range for distance computation.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using manual boolean masks with boundaries at -50, -10, 0, 10, 50. The bin for "0 cm" uses `np.isclose(dist, 0)` rather than checking if the animal is inside the zone range.

ii.
```python
out[(dist >= -50) & (dist < -10)] = 1
out[(dist >= -10) & (dist < 0)] = 2
out[np.isclose(dist, 0)] = 3
out[(dist > 0) & (dist <= 10)] = 4
out[(dist > 10) & (dist <= 50)] = 5
out[dist > 50] = 6
```

iii. The bin boundaries roughly match the instructions but the "0 cm" bin only catches exact zero (via `np.isclose`), not a range of positions inside the reward zone.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Behavior data (position) is aligned to neural timestamps using `np.searchsorted` to find the nearest behavior index for each neural timepoint.

ii.
```python
bidx = np.searchsorted(bt, nt, side='left')
bidx = np.clip(bidx, 0, len(bt) - 1)
...
dist_bins, _ = discretize_dist_to_reward(pos[bidx], reward_center)
```

iii. The AI uses searchsorted-based resampling from behavior to neural timestamps.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
pos = np.asarray(b['position'])
...
abs_pos_bins, pos_edges = discretize_position(pos, valid_mask, n_bins=5)
```

iii. Position data comes directly from the NWB behavior stream.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is discretized into 5 equal-sized bins where bin edges are computed from the data range using `np.linspace(lo, hi, n_bins + 1)` with `lo` and `hi` from valid position values.

ii.
```python
def discretize_position(pos, valid_mask, n_bins=5):
    valid_pos = pos[valid_mask]
    lo, hi = np.nanmin(valid_pos), np.nanmax(valid_pos)
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(pos, edges[1:-1], right=False), 0, n_bins - 1)
    return idx, edges
```

iii. The AI chose data-driven bin edges rather than fixed bin edges.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. 5 equal-sized bins from the min to max of valid position values. Bin edges are session-specific (data-driven).

ii. Same as 8-b above.

iii. The AI uses `np.linspace` to create 5 equal-width bins spanning the data range.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same searchsorted-based alignment as other behavioral variables.

ii.
```python
bidx = np.searchsorted(bt, nt, side='left')
bidx = np.clip(bidx, 0, len(bt) - 1)
...
abs_pos_bins[bidx]
```

iii. Position indices are looked up at the neural timepoints via searchsorted.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = np.asarray(b['lick'])
```

iii. Direct from the NWB behavior stream.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized with threshold > 0.5.

ii.
```python
(lick[bidx] > 0.5).astype(np.int64)
```

iii. The AI used 0.5 as threshold rather than 0.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same searchsorted-based alignment as other behavioral variables.

ii.
```python
(lick[bidx] > 0.5).astype(np.int64)
```

iii. Lick values at behavior indices closest to neural timestamps.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from `reward_zone` and `position` behavior time series. The median position when `reward_zone > 0` is computed per trial, then KMeans (k=3) clusters these into 3 canonical locations. Each trial is assigned to its nearest cluster center.

ii.
```python
trial_reward_pos, reward_loc_centers = infer_trial_reward_positions(pos, rz_raw, trials)
trial_reward_loc = assign_reward_location_labels(trial_reward_pos, reward_loc_centers)
```

iii. The AI empirically inferred reward zone locations from data rather than using known zone boundaries.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. KMeans clustering of per-trial median reward-zone positions, then nearest-center assignment.

ii.
```python
km = KMeans(n_clusters=k, random_state=0, n_init=20).fit(vals.reshape(-1, 1))
centers = np.sort(km.cluster_centers_.ravel())
...
def assign_reward_location_labels(trial_reward_pos, centers):
    labels = {}
    for tid, rp in trial_reward_pos.items():
        if not np.isfinite(rp) or len(finite_centers) == 0:
            labels[tid] = 0
        else:
            labels[tid] = int(np.argmin(np.abs(np.asarray(finite_centers) - rp)))
    return labels
```

iii. The AI chose KMeans over the reference's Viterbi algorithm approach with known zone ranges. Trials with no reward zone signal default to label 0.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from `Reward` event timestamps in the behavior time series.

ii.
```python
reward_event_timestamps = b.get('Reward__timestamps', np.asarray([]))
reward_outcomes = infer_reward_outcomes_from_events(trials, bt, reward_event_timestamps)
```

iii. The AI uses reward event timestamps to determine per-trial reward outcomes.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, checks whether any reward event timestamp falls within the trial's time range. Returns 1 if rewarded, 0 otherwise.

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

iii. The AI uses time-range matching rather than index-based reward detection.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Extremely short trials**: Trials with < 2 timepoints (neural or behavior) are skipped.
- **Sessions with < 2 trials**: Entire sessions are excluded.
- **Missing reward zone data**: Trials with no nonzero reward_zone signal get NaN positions; `assign_reward_location_labels` defaults these to label 0.
- **Missing reward events**: If no `Reward__timestamps` exist, defaults to empty array (all trials unrewarded).
- **Neural timestamp fallback**: If no neural timestamps exist, they are synthesized via `np.linspace`.

ii.
```python
if np.sum(nmask) < 2 or (e - s) < 2:
    continue
...
if sess['n_trials'] >= 2:
    sessions.append(sess)
...
reward_event_timestamps = b.get('Reward__timestamps', np.asarray([]))
...
if neural_t is None:
    neural_t = np.linspace(bt[0], bt[-1], neural_nt.shape[1])
```

iii. The AI uses defensive defaults throughout. No explicit neural/behavior length mismatch handling (unlike reference which crops to minimum).

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading all NWB files, as each file contains large neural recordings. The code loads each file once per session (via `h5py.File`), reading full neural and behavior arrays.

ii. N/A

iii. The AI's CONVERSION_NOTES note sample conversion was "fast (seconds per 2 sessions)" and "full dataset likely manageable in minutes."

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates over each trial sequentially, computing neural masking, behavioral indexing, and output discretization. Some operations (e.g., `discretize_dist_to_reward`, `discretize_speed`, `discretize_position`) are already applied to full arrays before the loop.

ii. N/A

iii. Pre-computing discretization on full arrays before the trial loop is a form of vectorization.

## 13-c. What processing does the code repeat multiple times?

i. Each NWB file is loaded only once per session during conversion (unlike the reference which loads each file twice -- once in survey and once in conversion). However, the `--sample` mode opens files just to check environment values before selecting sessions.

ii.
```python
for fp in files:
    with h5py.File(fp, 'r') as f:
        env = f['processing/behavior/BehavioralTimeSeries']['environment']['data'][:]
```

iii. The sample-selection environment check is lightweight compared to full conversion.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `choose_zone_centers` function is defined but never called. The `Counter` import is unused. The speed discretization is computed for all timepoints including invalid periods, though only valid trial slices are used.

ii.
```python
from collections import Counter  # unused
...
def choose_zone_centers(pos, rz_code):  # defined but never called
```

iii. Minor unused code artifacts.
