# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all `.nwb` files by recursively globbing `data/**/*.nwb` using `pathlib.Path.rglob`. Each NWB file is opened with `h5py` (not `pynwb`). All NWB files found are processed; for `--sample` mode, it selects two sessions with diverse environment types.

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))
...
for fp in files:
    print('Converting', fp, flush=True)
    sess = convert_session(fp, show_processing=args.show_processing)
```

Loading within `convert_session`:
```python
with h5py.File(path, 'r') as f:
    subj = decode_scalar(f['general/subject/subject_id'][()])
    sess = decode_scalar(f['general/session_id'][()])
    b = load_behavior_series(f)
    neural, neural_t, neural_base, neural_key = load_neural_series(f)
    n_rois, regions = load_n_rois_and_regions(f)
```

iii. The AI uses `h5py` for direct HDF5 access rather than `pynwb`. It loads all NWB files found under `data/` recursively. The CONVERSION_NOTES.md mentions exploring the NWB structure and finding 152 sessions across 11 subjects.

## 1-b. How are the data split into subjects?

i. Subjects are determined from the `general/subject/subject_id` field within each NWB file. After processing all sessions, unique subjects are collected and sorted.

ii.
```python
with h5py.File(path, 'r') as f:
    subj = decode_scalar(f['general/subject/subject_id'][()])
...
subjects = sorted(set(s['subject'] for s in sessions))
subj_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The AI extracts subject identity from the NWB metadata rather than parsing directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Session identity is extracted from the `general/session_id` field.

ii.
```python
sess = decode_scalar(f['general/session_id'][()])
```

iii. The one-file-per-session structure is standard for NWB datasets.

## 1-d. How are the data split into trials?

i. Trial boundaries are inferred using the `infer_trial_bounds` function, which finds rising edges of the `trial_start` signal. Trial ends are defined as the start of the next trial (or the end of the recording for the last trial). The function also requires `trial_num >= 0` for the start to be valid.

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

iii. The AI uses rising edges of `trial_start` validated against `trial_num >= 0`. Trial end is defined as the next trial's start, not the teleport signal.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 neural timepoints or fewer than 2 behavior timepoints are skipped. Sessions with fewer than 2 valid trials are excluded entirely.

ii.
```python
if np.sum(nmask) < 2 or (e - s) < 2:
    continue
...
if sess['n_trials'] >= 2:
    sessions.append(sess)
else:
    print('Skipping session with <2 trials after processing:', fp, flush=True)
```

iii. The AI uses minimal trial filtering (at least 2 timepoints), compared to the reference which filters trials with < 50 timepoints.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `DfOverF` or `Fluorescence` processing groups in the NWB file. The `load_neural_series` function looks for these in order of preference.

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

iii. The AI chose DfOverF/Fluorescence as the neural data source. The CONVERSION_NOTES.md Step 5 mapping table states "Prefer NWB `DfOverF` if present to match reference processed activity." However, the paper's decoder analysis uses deconvolved activity.

## 2-b. How is the `neural` data processed?

i. The neural data is oriented to (neurons x time) format and cast to float32. No further processing (e.g., deconvolution, filtering) is applied.

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
...
neural_trial = neural_nt[:, nmask].astype(np.float32)
```

iii. The AI assumed DfOverF data was already in a usable form.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron filtering is applied. The AI does not filter by `iscell` or any other quality metric. All ROIs from the neural data source are included.

ii.
```python
# No filtering code present - all neurons from load_neural_series are used directly
neural_nt = neural if neural.shape[0] < neural.shape[1] else neural.T
```

iii. The CONVERSION_NOTES.md Step 4 states "For conversion, start from all valid neural ROIs unless reference code indicates a required exclusion; verify later against methods/code." The AI did not implement `iscell` filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start by finding neural timestamps that fall within the behavior trial boundaries (`t0` to `t1`). Behavior indices are then mapped to neural timepoints using `searchsorted`.

ii.
```python
t0 = bt[s]
t1 = bt[e - 1]
nmask = (neural_t >= t0) & (neural_t <= t1)
...
nt = neural_t[nmask]
bidx = np.searchsorted(bt, nt, side='left')
bidx = np.clip(bidx, 0, len(bt) - 1)
neural_trial = neural_nt[:, nmask].astype(np.float32)
```

iii. This approach uses timestamp-based matching between neural and behavior data streams, which handles cases where the two streams have different sampling rates.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is set to `np.nan` in metadata. No temporal rebinning is applied; data is kept at whatever rate the neural data source provides.

ii.
```python
'time_bin_size': float(np.nan),
```

iii. The AI did not compute the actual time bin size from the data. The reference computes it from multi-plane recording rates.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the neural timestamps within the trial. The neural timestamps (`neural_t`) are used, masked to the trial boundaries.

ii.
```python
nt = neural_t[nmask]
...
time_from_start = (nt - t0).astype(np.float32)
```

iii. The AI uses neural timestamps rather than behavior timestamps for computing time from trial start.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp of the trial (`t0`, which is the behavior timestamp at trial start) is subtracted from each neural timestamp within the trial.

ii.
```python
t0 = bt[s]
...
time_from_start = (nt - t0).astype(np.float32)
```

iii. Standard approach of subtracting trial start time.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Time from start is computed directly from the neural timestamps, so it is inherently aligned.

ii.
```python
nt = neural_t[nmask]
time_from_start = (nt - t0).astype(np.float32)
```

iii. Since the time variable is derived from the same neural timestamps, alignment is automatic.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
env = map_environment(np.asarray(b['environment']))
...
env_trial = np.full(nt.shape, int(np.round(np.nanmedian(env[sl]))), dtype=np.float32)
```

iii. The environment variable is read from the NWB behavior data.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The `map_environment` function maps the raw environment values to binary 0/1. It finds unique finite non-negative values and maps the highest to 1, everything else to 0. Within a trial, the median value is used (to handle potential noise).

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

iii. The AI applies a remapping function that could change the semantics if the raw environment values are already 0/1 (as the reference suggests they are). Taking the median per trial adds robustness but shouldn't be needed if the value is constant per trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the `trial number` behavior time series. The trial ID (`tid`) stored at the trial start index is used.

ii.
```python
trial_ids = trial_num[starts].astype(int)
...
trialnum_trial = np.full(nt.shape, tid, dtype=np.float32)
```

iii. The AI uses the NWB `trial number` variable value at trial start, not a sequential loop counter.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial number from the NWB file at the trial start index is broadcast across all timepoints in the trial. No additional processing.

ii.
```python
trialnum_trial = np.full(nt.shape, tid, dtype=np.float32)
```

iii. The value is constant within a trial.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps. Reward event timestamps are compared to behavior timestamps to determine whether each trial was rewarded.

ii.
```python
reward_event_timestamps = b.get('Reward__timestamps', np.asarray([]))
reward_outcomes = infer_reward_outcomes_from_events(trials, bt, reward_event_timestamps)
```

iii. The AI uses reward event timestamps to infer outcomes per trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the code checks if any reward event timestamp falls within the trial's behavior timestamp range. The previous trial's outcome becomes the current trial's `previous_trial_outcome` input. For the first trial, it defaults to 0.

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
...
prev_out = 0
for i, (tid, s, e) in enumerate(trials):
    ...
    prev_out_trial = np.full(nt.shape, prev_out, dtype=np.float32)
    ...
    prev_out = int(reward_outcomes[i])
```

iii. The AI uses a running variable `prev_out` that starts at 0 and is updated after each trial. This correctly gives the previous trial's outcome.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the per-trial reward zone location. The reward zone location is determined by computing the median position when `reward_zone > 0` for each trial, then clustering these medians into 3 groups using KMeans.

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
...
reward_center = float(trial_reward_pos.get(tid, np.nan))
dist_bins, _ = discretize_dist_to_reward(pos[bidx], reward_center)
```

iii. The AI uses a data-driven approach (KMeans) to find reward zone centers rather than using known zone boundaries from the literature.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Distance is computed as `position - reward_center`, where `reward_center` is the median position within the reward zone for that trial. This gives a simple signed distance from a single point, not from a zone boundary.

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

iii. This differs from the reference which computes signed distance to the nearest edge of the reward zone range (0 inside the zone, negative before, positive after).

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using conditional assignments matching the specified bin edges: `< -50`, `-50 to -10`, `-10 to 0`, `0`, `0 to 10`, `10 to 50`, `> 50`.

ii.
```python
out[(dist >= -50) & (dist < -10)] = 1
out[(dist >= -10) & (dist < 0)] = 2
out[np.isclose(dist, 0)] = 3
out[(dist > 0) & (dist <= 10)] = 4
out[(dist > 10) & (dist <= 50)] = 5
out[dist > 50] = 6
```

iii. The bin edges match the instruction specification, but the underlying distance computation (from center point vs. zone edges) changes the semantics.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Behavior positions are indexed using `bidx` (from `searchsorted` of neural timestamps against behavior timestamps), ensuring the position values align with the neural timepoints.

ii.
```python
bidx = np.searchsorted(bt, nt, side='left')
bidx = np.clip(bidx, 0, len(bt) - 1)
...
dist_bins, _ = discretize_dist_to_reward(pos[bidx], reward_center)
```

iii. The timestamp-based alignment maps neural timepoints to their nearest behavior sample.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
pos = np.asarray(b['position'])
...
abs_pos_bins, pos_edges = discretize_position(pos, valid_mask, n_bins=5)
```

iii. Position data is directly available in the NWB behavior streams.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is discretized into 5 equal-sized bins using data-driven edges: `np.linspace(lo, hi, n_bins + 1)` where `lo` and `hi` are the min/max of valid positions. Then `np.digitize` and clipping to [0, n_bins-1].

ii.
```python
def discretize_position(pos, valid_mask, n_bins=5):
    valid_pos = pos[valid_mask]
    lo, hi = np.nanmin(valid_pos), np.nanmax(valid_pos)
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(pos, edges[1:-1], right=False), 0, n_bins - 1)
    return idx, edges
```

iii. The AI uses data-driven bin edges based on the observed position range, rather than fixed edges.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins using `np.digitize` with data-driven edges from `np.linspace(min_pos, max_pos, 6)`.

ii.
```python
edges = np.linspace(lo, hi, n_bins + 1)
idx = np.clip(np.digitize(pos, edges[1:-1], right=False), 0, n_bins - 1)
```

iii. The reference uses fixed bins `[-inf, 50, 150, 250, 350, inf]` (100 cm wide bins), while the AI computes edges from data range.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position is aligned using the `bidx` array from `searchsorted`, mapping neural timestamps to behavior indices.

ii.
```python
abs_pos_bins[bidx]
```

iii. Same timestamp-based alignment as other behavior variables.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = np.asarray(b['lick'])
```

iii. Lick data is directly available in the NWB behavior streams.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized using a threshold of 0.5: values > 0.5 are mapped to 1, otherwise 0.

ii.
```python
(lick[bidx] > 0.5).astype(np.int64),
```

iii. The reference uses `> 0` as the threshold. Using `> 0.5` could miss small lick values between 0 and 0.5.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick values are aligned using the `bidx` array from `searchsorted`.

ii.
```python
lick[bidx]
```

iii. Same timestamp-based alignment approach.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` behavior time series and `position` behavior time series. The AI computes the median position when `reward_zone > 0` for each trial, then uses KMeans to cluster these medians into 3 groups.

ii.
```python
trial_reward_pos, reward_loc_centers = infer_trial_reward_positions(pos, rz_raw, trials)
trial_reward_loc = assign_reward_location_labels(trial_reward_pos, reward_loc_centers)
```

iii. The AI uses an unsupervised clustering approach rather than known zone boundaries.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. After clustering reward positions into 3 groups with KMeans, each trial is assigned to the nearest cluster center. The label is the cluster index (0, 1, 2), which does NOT correspond to the known A/B/C zone mapping.

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

iii. The AI does not use known reward zone boundaries. The KMeans approach may produce different zone assignments than the Viterbi approach using known boundaries. Also, trials where `reward_zone` is never active default to label 0 (zone A), which may not be correct.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavior time series timestamps.

ii.
```python
reward_event_timestamps = b.get('Reward__timestamps', np.asarray([]))
reward_outcomes = infer_reward_outcomes_from_events(trials, bt, reward_event_timestamps)
```

iii. Reward events are identified by their timestamps in the NWB data.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, checks if any reward event timestamp falls within the trial's time range. Binary output: 1 if rewarded, 0 if not.

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

iii. This is a reasonable approach for determining per-trial reward outcome.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- Trials with < 2 neural or behavior timepoints are skipped.
- Sessions with < 2 valid trials are excluded.
- If no reward zone activity is detected in a trial, the reward position defaults to NaN and the label defaults to 0.
- Neural data orientation is inferred based on shape comparison with ROI count.
- If neural timestamps are missing, they are synthesized using `np.linspace`.

ii.
```python
if np.sum(nmask) < 2 or (e - s) < 2:
    continue
...
if sess['n_trials'] >= 2:
    sessions.append(sess)
...
trial_reward_pos[tid] = float(np.nanmedian(pos[sl][nz])) if np.any(nz) else np.nan
...
if not np.isfinite(rp) or len(finite_centers) == 0:
    labels[tid] = 0
```

iii. The AI implements defensive checks but does not explicitly handle neural/behavior length mismatches (the reference crops to the minimum of the two).

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files with `h5py` and reading neural data arrays.
2. The sample selection in `--sample` mode opens multiple NWB files to check environment values.
3. Full conversion iterates over all NWB files.

ii. N/A

iii. NWB files are large and I/O-bound.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` iterates over trials sequentially. The `infer_trial_reward_positions` function loops over trials. The `infer_reward_outcomes_from_events` function loops over trials. These could potentially be vectorized with array operations.

ii. N/A

iii. Variable-length trials make full vectorization challenging.

## 13-c. What processing does the code repeat multiple times?

i. In `--sample` mode, the code opens NWB files to check environment values, then opens them again for conversion. The `discretize_position` function computes bin edges over all valid positions, then applies binning - this is done once per session.

ii. N/A

iii. The double file access in sample mode is a minor inefficiency.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `load_n_rois_and_regions` function is called to determine ROI count and region labels, but the region information is not used (all neurons are assigned to CA1). The `zone_centers` are stored in the session result dict but not included in the final output. The `neural_source` field is also stored but not included in the final data dict.

ii.
```python
n_rois, regions = load_n_rois_and_regions(f)
...
'zone_centers': reward_loc_centers.tolist() if hasattr(reward_loc_centers, 'tolist') else reward_loc_centers,
'neural_source': f'{neural_base}/{neural_key}',
```

iii. These are minor inefficiencies that don't significantly impact runtime.
