# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script recursively finds every `.nwb` file under `data/`, opens each one with `h5py`, loads all behavior time series from `processing/behavior/BehavioralTimeSeries`, and loads a single neural series from the first available group among `processing/ophys/DfOverF` and `processing/ophys/Fluorescence`. It then converts each file as one session. This means it does load all session files, but it does not load all optical physiology series within a session because it returns the first series it finds.

ii.
```python
def load_behavior_series(f):
    grp = f['processing/behavior/BehavioralTimeSeries']
    out = {}
    for k in grp.keys():
        g = grp[k]
        if 'data' in g:
            out[k] = np.asarray(g['data'][:])
        if 'timestamps' in g:
            out[k + '__timestamps'] = np.asarray(g['timestamps'][:])
    return out

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

files = sorted(Path('data').rglob('*.nwb'))
for fp in files:
    sess = convert_session(fp, show_processing=args.show_processing)
```

iii. In `CONVERSION_NOTES.md`, the agent justified NWB as the source of truth and said it would use NWB behavior time series as the canonical alignment source and "processed imaging activity rather than raw fluorescence when available." The README likewise describes the dataset as a direct NWB conversion.

## 1-b. How are the data split into subjects?

i. Subjects are determined per session from `general/subject/subject_id`, then the final `subjects` list is the sorted set of subject IDs observed across converted sessions.

ii.
```python
with h5py.File(path, 'r') as f:
    subj = decode_scalar(f['general/subject/subject_id'][()])

subjects = sorted(set(s['subject'] for s in sessions))
subj_to_idx = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.asarray([subj_to_idx[s['subject']] for s in sessions], dtype=np.int64),
```

iii. The notes say the NWB files are organized one file per subject/session under `data/sub-*`, and the agent treated the NWB metadata as authoritative for subject identity.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file is treated as one session. Session identity is read from `general/session_id`.

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))

with h5py.File(path, 'r') as f:
    sess = decode_scalar(f['general/session_id'][()])
```

iii. The notes explicitly state that data are organized as `data/sub-*/sub-*_ses-*_behavior+ophys.nwb` and that each file corresponds to one subject/session.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from rising edges of `trial_start`, filtered to samples where `trial number >= 0`. Each trial ends at the next detected `trial_start`, or at end-of-recording for the final trial. If no `trial_start` edges are found, the fallback is positive changes in `trial number`. The code does not use `teleport` to terminate trials.

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

iii. In the notes, the agent said standard NWB trial intervals were absent and that trials therefore had to be reconstructed from behavioral time series, with `trial_start` as the main signal.

## 1-e. How are trials filtered based on quality controls?

i. Trials are only discarded if they have fewer than 2 neural samples or fewer than 2 behavioral samples after alignment. Sessions are only discarded if fewer than 2 trials remain. No explicit 50-sample minimum, reward-based filtering, or teleport-based curation is applied.

ii.
```python
for i, (tid, s, e) in enumerate(trials):
    sl = slice(s, e)
    t0 = bt[s]
    t1 = bt[e - 1]
    nmask = (neural_t >= t0) & (neural_t <= t1)
    if np.sum(nmask) < 2 or (e - s) < 2:
        continue

...
if sess['n_trials'] >= 2:
    sessions.append(sess)
else:
    print('Skipping session with <2 trials after processing:', fp, flush=True)
```

iii. The notes emphasize maintaining at least two trials per session for decoder evaluation. No stronger trial-quality rule is documented.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The script derives `neural` from the first available ophys dataset among `processing/ophys/DfOverF` and `processing/ophys/Fluorescence`. It does not use `Neuropil`, `Deconvolved`, or combine multiple planes.

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
```

iii. The main justification in `CONVERSION_NOTES.md` is "Use processed imaging activity rather than raw fluorescence when available: Reference code includes `dff` processing, so NWB `DfOverF` should be preferred if present."

## 2-b. How is the `neural` data processed?

i. Neural processing is minimal. The selected matrix is oriented to `neurons x time`, cast to `float32` per trial, and sliced by trial-aligned neural timestamps. There is no dF/F recomputation, neuropil subtraction, deconvolution, smoothing, or multi-plane concatenation.

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

iii. The notes justify this as a simplification: use existing NWB processed activity when available and preserve time alignment required for the decoder.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is effectively no neural quality-control filtering beyond shape/orientation checks and trial-length requirements. The code does not apply `iscell`, does not remove putative interneurons, and does not filter low-quality neurons.

ii.
```python
def load_n_rois_and_regions(f):
    n_rois = None
    regions = None
    if 'processing/ophys/ImageSegmentation' in f:
        for k in f['processing/ophys/ImageSegmentation'].keys():
            ps = f['processing/ophys/ImageSegmentation'][k]
            if 'id' in ps:
                n_rois = len(ps['id'])
            ...

...
brain_region_idx = np.zeros(neural_nt.shape[0], dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` originally framed this as "start from all valid neural ROIs unless reference code indicates a required exclusion." The final script never added the paper's neuron-censoring steps.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to `trial_start` by taking the behavioral timestamp at the trial start (`t0`) and selecting neural samples whose timestamps fall within that trial window. The neural time axis is then represented relative to `t0`.

ii.
```python
for i, (tid, s, e) in enumerate(trials):
    t0 = bt[s]
    t1 = bt[e - 1]
    nmask = (neural_t >= t0) & (neural_t <= t1)
    ...
    nt = neural_t[nmask]
    neural_trial = neural_nt[:, nmask].astype(np.float32)
    time_from_start = (nt - t0).astype(np.float32)
```

iii. The README says "Trials are aligned to `trial_start`." The notes also say the decoder requires trial-start alignment and that NWB behavior timestamps should be the canonical alignment source.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No explicit temporal rebinning is applied. The code keeps the native sample count of the chosen neural series. If neural timestamps are missing, it synthesizes a linearly spaced neural timestamp vector spanning the behavior timestamps. The script never computes a definitive bin size for metadata and stores `NaN` for `metadata['time_bin_size']`.

ii.
```python
if neural_t is None:
    if bt is None:
        raise RuntimeError('No timestamps for neural or behavior')
    neural_t = np.linspace(bt[0], bt[-1], neural_nt.shape[1])

...
'metadata': {
    ...
    'time_bin_size': float(np.nan),
    ...
}
```

iii. The notes say trials should be represented in time bins aligned to trial start and that DfOverF should be used "if present," but they do not justify leaving the final time bin size unspecified.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the neural timestamp vector `nt` and the behavior timestamp at trial start `bt[s]`. When the neural series lacks timestamps, `nt` is synthesized from the behavior timestamps.

ii.
```python
bt = b.get('position__timestamps', None)
if bt is None:
    any_ts = [v for k, v in b.items() if k.endswith('__timestamps')]
    bt = any_ts[0]

...
if neural_t is None:
    neural_t = np.linspace(bt[0], bt[-1], neural_nt.shape[1])

...
time_from_start = (nt - t0).astype(np.float32)
```

iii. The notes say NWB behavior timestamps are the canonical alignment source and that all decoder variables should be represented on the aligned trial time base.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the code subtracts the trial-start time `t0` from every neural timestamp in that trial.

ii.
```python
t0 = bt[s]
...
nt = neural_t[nmask]
time_from_start = (nt - t0).astype(np.float32)
```

iii. No elaborate justification is given beyond the requirement to align trials to their start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is defined directly on the same neural samples used for each trial, so it is exactly aligned to the per-trial neural matrix.

ii.
```python
nt = neural_t[nmask]
neural_trial = neural_nt[:, nmask].astype(np.float32)
time_from_start = (nt - t0).astype(np.float32)
inp = np.vstack([time_from_start, env_trial, trialnum_trial, prev_out_trial])
```

iii. The notes say the decoder representation should be time-aligned trial tensors; the implementation uses neural timepoints as that common grid.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavior time series `environment`.

ii.
```python
env = map_environment(np.asarray(b['environment']))
```

iii. The notes map `processing/behavior/BehavioralTimeSeries/environment` to the decoder's binary environment variable.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The code first maps raw environment codes to binary values with `map_environment`, handling possible `-1` values and arbitrary low/high codes. Then, within each trial, it takes the median environment value and broadcasts that constant across the neural timepoints in the trial.

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

iii. The notes explicitly say environment should be converted to binary ENV1/ENV2 and broadcast within each trial.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the stored `trial number` behavior series. The code records the `trial number` value at each detected trial start and uses that as the trial ID.

ii.
```python
trial_num = np.asarray(b['trial number']).astype(int)
...
trial_ids = trial_num[starts].astype(int)
...
trialnum_trial = np.full(nt.shape, tid, dtype=np.float32)
```

iii. The notes say "Use native trial numbering after reconstructing valid trials."

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The chosen trial ID `tid` is broadcast as a constant across all neural timepoints in that trial.

ii.
```python
for i, (tid, s, e) in enumerate(trials):
    ...
    trialnum_trial = np.full(nt.shape, tid, dtype=np.float32)
```

iii. The justification in the notes is that trial-level covariates can be broadcast across time bins for decoder compatibility.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the `Reward` event timestamps together with the inferred trial windows based on behavior timestamps.

ii.
```python
reward_event_timestamps = b.get('Reward__timestamps', np.asarray([]))
reward_outcomes = infer_reward_outcomes_from_events(trials, bt, reward_event_timestamps)
```

iii. The README states that reward outcomes are derived from `BehavioralTimeSeries/Reward` event timestamps, and the notes say previous outcome must be derived only after reward outcome is recovered per trial.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The code first computes a binary reward outcome for every trial. It then initializes `prev_out = 0` and, for each trial, broadcasts the previous trial's reward outcome across the current trial. The first trial therefore receives 0.

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

iii. The notes say previous-trial outcome should be inferred from rewarded versus omitted trials and broadcast within each trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from `position` and `reward_zone`. The code uses `reward_zone > 0` within each trial to find position samples belonging to the reward zone, takes the median of those positions, and uses that single per-trial position as the reward-zone center.

ii.
```python
pos = np.asarray(b['position'])
rz_raw = np.asarray(b['reward_zone'])

def infer_trial_reward_positions(pos, rz_signal, trials):
    trial_reward_pos = {}
    for tid, s, e in trials:
        sl = slice(s, e)
        nz = rz_signal[sl] > 0
        trial_reward_pos[tid] = float(np.nanmedian(pos[sl][nz])) if np.any(nz) else np.nan
    ...

reward_center = float(trial_reward_pos.get(tid, np.nan))
```

iii. The notes say reward-zone location is inferred "from positions of nonzero `reward_zone` samples and clustered into three canonical locations."

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The code computes signed distance from current position to a single inferred reward-zone center, not to the nearest point in a reward-zone interval. Negative values mean before that center and positive values mean after it.

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

iii. The notes justify this with the idea that reward-zone positions can be inferred from behavior and then converted into canonical locations, but they do not discuss using zone edges rather than a center.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is thresholded with explicit inequalities implementing the requested 7 bins: `< -50`, `[-50, -10)`, `[-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50`. `NaN` distances are mapped to category 0.

ii.
```python
out = np.full(dist.shape, 0, dtype=np.int64)
out[(dist >= -50) & (dist < -10)] = 1
out[(dist >= -10) & (dist < 0)] = 2
out[np.isclose(dist, 0)] = 3
out[(dist > 0) & (dist <= 10)] = 4
out[(dist > 10) & (dist <= 50)] = 5
out[dist > 50] = 6
out[np.isnan(dist)] = 0
```

iii. The justification is implicit: these thresholds match the decoder specification in the task instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Behavior is aligned to neural timepoints by taking each neural timestamp in the trial and mapping it to the nearest behavior index using `np.searchsorted`. Distance-to-zone is then computed from behavior `position[bidx]` on those neural-aligned samples.

ii.
```python
nt = neural_t[nmask]
bidx = np.searchsorted(bt, nt, side='left')
bidx = np.clip(bidx, 0, len(bt) - 1)

reward_center = float(trial_reward_pos.get(tid, np.nan))
dist_bins, _ = discretize_dist_to_reward(pos[bidx], reward_center)
```

iii. The notes say behavior timestamps are the canonical alignment source, but the final code uses neural timepoints as the grid and samples behavior onto that grid.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from the behavior time series `position`.

ii.
```python
pos = np.asarray(b['position'])
...
abs_pos_bins, pos_edges = discretize_position(pos, valid_mask, n_bins=5)
```

iii. The notes map `position` directly to the absolute-position decoder output.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The code computes five equal-width bins between the session-specific minimum and maximum "valid" position values, where validity is defined by `trial number >= 0` and `position > -400`. It then digitizes all positions into those session-specific edges.

ii.
```python
valid_mask = (trial_num >= 0) & (pos > -400)

def discretize_position(pos, valid_mask, n_bins=5):
    valid_pos = pos[valid_mask]
    lo, hi = np.nanmin(valid_pos), np.nanmax(valid_pos)
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(pos, edges[1:-1], right=False), 0, n_bins - 1)
    return idx, edges
```

iii. The notes say absolute position should be "5 equal-sized bins" and that teleport/out-of-track periods should be excluded if needed; the code operationalizes that by deriving session-specific edges from valid samples.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is thresholded by the session-specific equally spaced edges returned by `np.linspace(lo, hi, 6)`, not by fixed 90 cm bins.

ii.
```python
edges = np.linspace(lo, hi, n_bins + 1)
idx = np.clip(np.digitize(pos, edges[1:-1], right=False), 0, n_bins - 1)
```

iii. No explicit extra justification is given beyond the notes' statement that the output should be discretized into equal-sized bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Behavior positions are sampled onto the neural timestamps within each trial using `bidx = np.searchsorted(bt, nt, side='left')`, and `abs_pos_bins[bidx]` is emitted on that neural grid.

ii.
```python
nt = neural_t[nmask]
bidx = np.searchsorted(bt, nt, side='left')
bidx = np.clip(bidx, 0, len(bt) - 1)

out = np.vstack([
    dist_bins,
    abs_pos_bins[bidx],
    ...
])
```

iii. The notes justify using trial-aligned time tensors; the final code uses neural timepoints as the common time base.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavior time series `lick`.

ii.
```python
lick = np.asarray(b['lick'])
```

iii. The notes directly map `lick` to the binary lick decoder output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The lick trace is binarized at `> 0.5`, yielding 0 for no lick and 1 for lick.

ii.
```python
(lick[bidx] > 0.5).astype(np.int64)
```

iii. The task requires a binary lick output, and the notes describe lick as a time-varying 0/1 series.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The lick output is sampled from the behavior stream at behavior indices matched to each neural timestamp in the trial.

ii.
```python
nt = neural_t[nmask]
bidx = np.searchsorted(bt, nt, side='left')
bidx = np.clip(bidx, 0, len(bt) - 1)

out = np.vstack([
    ...
    (lick[bidx] > 0.5).astype(np.int64),
    ...
])
```

iii. As with other time-varying behavior outputs, the notes favor synchronized trial-aligned tensors.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from `reward_zone` and `position`. The code looks at reward-zone-positive samples within each trial, takes the median position of those samples, clusters those per-trial medians, and then assigns each trial to the nearest cluster center.

ii.
```python
def infer_trial_reward_positions(pos, rz_signal, trials):
    ...
    trial_reward_pos[tid] = float(np.nanmedian(pos[sl][nz])) if np.any(nz) else np.nan
    ...
    km = KMeans(n_clusters=k, random_state=0, n_init=20).fit(vals.reshape(-1, 1))
    centers = np.sort(km.cluster_centers_.ravel())

def assign_reward_location_labels(trial_reward_pos, centers):
    ...
    labels[tid] = int(np.argmin(np.abs(np.asarray(finite_centers) - rp)))
```

iii. The README says reward-zone location is inferred from per-trial reward-zone position dynamics, and the notes say those positions are clustered into three canonical locations.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Per trial, the code computes a single reward-zone position summary, runs session-level `KMeans` with up to 3 clusters on the set of finite trial summaries, sorts the cluster centers by position, and assigns each trial the index of the nearest center. Missing trials default to label 0.

ii.
```python
vals = np.array([v for v in trial_reward_pos.values() if np.isfinite(v)])
if len(vals) == 0:
    centers = np.array([np.nan, np.nan, np.nan])
else:
    k = min(3, len(np.unique(np.round(vals, 3))))
    km = KMeans(n_clusters=k, random_state=0, n_init=20).fit(vals.reshape(-1, 1))
    centers = np.sort(km.cluster_centers_.ravel())

...
if not np.isfinite(rp) or len(finite_centers) == 0:
    labels[tid] = 0
else:
    labels[tid] = int(np.argmin(np.abs(np.asarray(finite_centers) - rp)))
```

iii. The notes say this was intended to recover the three canonical reward locations from the observed position dynamics rather than from a fixed hand-coded mapping.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the `Reward` event timestamps and the inferred trial windows from behavior timestamps.

ii.
```python
reward_event_timestamps = b.get('Reward__timestamps', np.asarray([]))
reward_outcomes = infer_reward_outcomes_from_events(trials, bt, reward_event_timestamps)
```

iii. The README explicitly states that reward outcomes are derived from NWB `BehavioralTimeSeries/Reward` event timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the code checks whether any reward-event timestamp falls within the trial interval `[t0, t1]`. The result is a binary per-trial value, which is broadcast across that trial's neural timepoints.

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
np.full(nt.shape, int(reward_outcomes[i]), dtype=np.int64)
```

iii. The notes say rewarded versus omitted trials must be recovered from NWB reward signals and then emitted as a per-trial variable.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles several cases heuristically rather than with strict validation. If `position__timestamps` are missing it uses the first timestamp array it can find; if neural timestamps are missing it synthesizes them with `np.linspace`; if a trial has fewer than 2 samples it is dropped; if reward-zone evidence is missing the reward center becomes `NaN`, the reward-zone label defaults to 0, and distance-to-zone bins default to 0; if reward timestamps are missing the script treats all trials as unrewarded.

ii.
```python
bt = b.get('position__timestamps', None)
if bt is None:
    any_ts = [v for k, v in b.items() if k.endswith('__timestamps')]
    bt = any_ts[0]

...
if neural_t is None:
    if bt is None:
        raise RuntimeError('No timestamps for neural or behavior')
    neural_t = np.linspace(bt[0], bt[-1], neural_nt.shape[1])

...
if np.sum(nmask) < 2 or (e - s) < 2:
    continue

...
if not np.isfinite(rp) or len(finite_centers) == 0:
    labels[tid] = 0

...
out[np.isnan(dist)] = 0
```

iii. The notes mention excluding invalid periods and recovering variables from available synchronized streams, but the final code does not document these fallback rules explicitly; they are inferred from the implementation.

## 13-a. What are the most time-consuming steps of the code?

i. The dominant costs are reading large NWB/HDF5 arrays for every session, per-session reward-zone clustering with `KMeans`, trial-by-trial neural/behavior alignment and slicing, optional per-session plotting, and serializing the final pickle.

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))
for fp in files:
    sess = convert_session(fp, show_processing=args.show_processing)

...
km = KMeans(n_clusters=k, random_state=0, n_init=20).fit(vals.reshape(-1, 1))

...
for i, (tid, s, e) in enumerate(trials):
    ...
```

iii. `CONVERSION_NOTES.md` says the full conversion is "manageable in minutes," which implies the agent expected file I/O and per-session conversion to dominate run time.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` could be partially vectorized: reward outcomes, previous outcomes, and behavior-to-neural alignment are recomputed trial by trial. `infer_trial_reward_positions` and `assign_reward_location_labels` also loop in Python over trials.

ii.
```python
for _, s, e in trials:
    ...

for tid, s, e in trials:
    sl = slice(s, e)
    nz = rz_signal[sl] > 0
    trial_reward_pos[tid] = float(np.nanmedian(pos[sl][nz])) if np.any(nz) else np.nan

for i, (tid, s, e) in enumerate(trials):
    ...
```

iii. The notes do not give an explicit efficiency analysis. This is a code-level observation from the final implementation.

## 13-c. What processing does the code repeat multiple times?

i. In `--sample` mode, the script first opens many files to inspect environment values and then reopens the chosen files for conversion. Within conversion, behavior timestamps are loaded once into a dictionary and then repeatedly searched with `np.searchsorted` for every trial. The script also computes session-wide position bins before immediately subsampling them trial by trial.

ii.
```python
if args.sample:
    for fp in files:
        with h5py.File(fp, 'r') as f:
            env = f['processing/behavior/BehavioralTimeSeries']['environment']['data'][:]
            ...

for fp in files:
    sess = convert_session(fp, show_processing=args.show_processing)

...
for i, (tid, s, e) in enumerate(trials):
    ...
    bidx = np.searchsorted(bt, nt, side='left')
```

iii. The notes describe the sample-selection logic as deliberately choosing "diverse env0/env1 sessions," which explains the extra file pass in sample mode.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code contains several unused or discarded pieces of work: `choose_zone_centers` is defined but never used; `teleport`, `trial_start`, `regions`, `pos_edges`, and the imported `Counter` are unused; the returned per-session `zone_centers` and `neural_source` are not kept in the final saved dataset; optional processing plots are generated only for inspection and not used downstream.

ii.
```python
from collections import Counter

def choose_zone_centers(pos, rz_code):
    centers = {}
    ...
    return centers

...
teleport = np.asarray(b['teleport'])
trial_start = np.asarray(b['trial_start'])
...
abs_pos_bins, pos_edges = discretize_position(pos, valid_mask, n_bins=5)

...
return {
    ...
    'zone_centers': reward_loc_centers.tolist() if hasattr(reward_loc_centers, 'tolist') else reward_loc_centers,
    'neural_source': f'{neural_base}/{neural_key}',
}
```

iii. No explicit justification is given for these leftovers. They appear to be remnants of exploratory development and optional debugging support.
