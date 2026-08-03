# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads every `.nwb` file under `data/` with `Path('data').rglob('*.nwb')`, then opens each file with `h5py`. Within each session file, it loads behavior from `processing/behavior/BehavioralTimeSeries` and neural data from the first available dataset under `processing/ophys/DfOverF` or `processing/ophys/Fluorescence`.

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))
```

```python
with h5py.File(path, 'r') as f:
    subj = decode_scalar(f['general/subject/subject_id'][()])
    sess = decode_scalar(f['general/session_id'][()])
    b = load_behavior_series(f)
    neural, neural_t, neural_base, neural_key = load_neural_series(f)
```

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
```

iii. In `CONVERSION_NOTES.md`, the AI states that the dataset is organized as one NWB file per subject/session and that NWB should be treated as the source of truth. In the trajectory it explicitly notes that the NWB files contain the needed behavior and ophys streams, so it decided to load everything directly from those files rather than through the reference package abstractions.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the NWB metadata field `general/subject/subject_id`. After all sessions are converted, the unique subject IDs are sorted and stored in `subjects`, and each session gets a `subject_idx`.

ii.
```python
with h5py.File(path, 'r') as f:
    subj = decode_scalar(f['general/subject/subject_id'][()])
```

```python
subjects = sorted(set(s['subject'] for s in sessions))
subj_to_idx = {s: i for i, s in enumerate(subjects)}
...
'subjects': subjects,
'subject_idx': np.asarray([subj_to_idx[s['subject']] for s in sessions], dtype=np.int64),
```

iii. The notes say each NWB file corresponds to one subject/session and exposes subject metadata, so the AI used the file metadata rather than directory names to define subject identity.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The converter loops over all `.nwb` files, runs `convert_session` once per file, and appends the result as one session if at least two trials survive processing.

ii.
```python
for fp in files:
    print('Converting', fp, flush=True)
    sess = convert_session(fp, show_processing=args.show_processing)
    if sess['n_trials'] >= 2:
        sessions.append(sess)
```

```python
with h5py.File(path, 'r') as f:
    sess = decode_scalar(f['general/session_id'][()])
```

iii. The notes describe the data as “one subject/session per NWB file,” and the trajectory repeatedly refers to picking or converting “sessions” by selecting NWB files. That is the AI’s operational definition of a session.

## 1-d. How are the data split into trials?

i. Trials are inferred from the behavior time series. The AI primarily uses rising edges of `trial_start`; if none are found, it falls back to positive changes in `trial number`. Trial ends are defined by the next inferred start, or the end of the recording for the last trial. The `teleport` variable is loaded but not used to define trial ends.

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

iii. In the trajectory, the AI wrote that “trial segmentation can be reconstructed from `trial_start` and/or changes in `trial number`.” It also noted that the NWB files do not expose a standard trials table, so it reconstructed trial structure from behavior streams.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept if they contain at least two neural bins and at least two behavior samples under the inferred trial window. Sessions are kept if at least two trials survive. There is no stricter trial-length filter.

ii.
```python
nmask = (neural_t >= t0) & (neural_t <= t1)
if np.sum(nmask) < 2 or (e - s) < 2:
    continue
```

```python
if sess['n_trials'] >= 2:
    sessions.append(sess)
else:
    print('Skipping session with <2 trials after processing:', fp, flush=True)
```

iii. I did not find an explicit justification in `CONVERSION_NOTES.md` for this very permissive filter. The trajectory only mentions satisfying the decoder’s “at least two trials per session” requirement and getting sample/full validation to pass.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from the first available NWB dataset under `processing/ophys/DfOverF` or, if absent, `processing/ophys/Fluorescence`. It does not use the `Deconvolved` series from the NWB file.

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

iii. In the notes, the AI planned to “prefer NWB `DfOverF` if present to match reference processed activity.” The trajectory shows that it explicitly checked for `DfOverF` and `Fluorescence` availability before implementing the converter.

## 2-b. How is the `neural` data processed?

i. The AI does minimal processing. It ensures the neural array is 2D, heuristically orients it to neurons by time using the ROI count when available, and then slices neural data by trial time windows. It does not compute deconvolution, concatenate multi-plane `Deconvolved` series, or otherwise transform the activity.

ii.
```python
if neural.ndim != 2:
    raise RuntimeError(f'Unexpected neural shape {neural.shape}')
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

```python
neural_trial = neural_nt[:, nmask].astype(np.float32)
```

iii. The notes say the AI wanted to “use processed imaging activity rather than raw fluorescence when available.” The trajectory also shows it considered `DfOverF` to be an acceptable processed signal for the decoder and did not pursue additional neural preprocessing once validation passed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is effectively no neural quality filtering. The converter reads ROI count metadata but does not apply `iscell` or any other ROI-level quality criterion before constructing the neural matrices.

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
            if 'location' in ps:
                try:
                    loc = ps['location'][:]
                    regions = [decode_scalar(x) for x in loc]
                except Exception:
                    pass
            break
    return n_rois, regions
```

```python
brain_region_idx = np.zeros(neural_nt.shape[0], dtype=np.int64)
```

iii. In `CONVERSION_NOTES.md`, the AI explicitly wrote: “For conversion, start from all valid neural ROIs unless reference code indicates a required exclusion.” I did not find any later step where it added cell-level filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start by taking each inferred trial window, converting it to start and end behavior timestamps (`t0`, `t1`), and selecting neural timestamps within that interval. Time from trial start is then computed relative to `t0`.

ii.
```python
t0 = bt[s]
t1 = bt[e - 1]
nmask = (neural_t >= t0) & (neural_t <= t1)
...
neural_trial = neural_nt[:, nmask].astype(np.float32)
time_from_start = (nt - t0).astype(np.float32)
```

iii. The notes and trajectory both state that the decoder task required temporal alignment to trial start. The AI’s explicit design choice was to use behavior timestamps as the trial reference and project neural data into those trial windows.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI does not rebin neural data. It uses the native neural timestamps if present; otherwise it linearly interpolates a timestamp vector across the behavior time span. It does not compute or store a definite time bin size, and `metadata['time_bin_size']` is left as `NaN`.

ii.
```python
if neural_t is None:
    if bt is None:
        raise RuntimeError('No timestamps for neural or behavior')
    neural_t = np.linspace(bt[0], bt[-1], neural_nt.shape[1])
```

```python
'metadata': {
    ...
    'time_bin_size': float(np.nan),
    'temporal_alignment_event': 'trial_start',
```

iii. In the trajectory, the AI acknowledged early that `time_bin_size` was still provisional and remained `NaN`. Its justification was essentially pragmatic: keep native sample timing and proceed if decoder validation succeeded.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. This input is derived from neural timestamps for the bins included in a trial (`nt`) and the behavior timestamp at the inferred trial start (`bt[s]`).

ii.
```python
bt = b.get('position__timestamps', None)
...
t0 = bt[s]
...
nt = neural_t[nmask]
time_from_start = (nt - t0).astype(np.float32)
```

iii. The AI’s notes say behavior time series share a common time base and that the decoder requires trial-start alignment. The implemented choice was to represent elapsed time on the neural bins themselves so that this input would match the neural matrix shape exactly.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each neural bin in a trial, the AI subtracts the trial-start timestamp `t0` from that bin’s timestamp.

ii.
```python
time_from_start = (nt - t0).astype(np.float32)
```

iii. No elaborate justification was given beyond the notes’ statement that trials should be “aligned to trial start” and time should be a continuous trial-relative variable.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is defined directly on the neural bins. After selecting the neural timestamps `nt` within a trial, the AI computes elapsed time on those same timestamps and stacks them into the input matrix for that trial.

ii.
```python
nt = neural_t[nmask]
...
inp = np.vstack([time_from_start, env_trial, trialnum_trial, prev_out_trial])
```

iii. The trajectory shows that the AI wanted this variable to be exactly aligned with the neural data rather than computed on the behavior clock and then copied over.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The environment input comes from the behavior time series `environment`.

ii.
```python
env = map_environment(np.asarray(b['environment']))
```

iii. The notes and trajectory explicitly identify `environment` as one of the canonical behavior variables available in `BehavioralTimeSeries` and map it to the decoder’s environment input.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI maps the raw environment values to binary 0/1 using `map_environment`, then takes the median environment value over the trial and broadcasts that constant value across the neural bins in the trial.

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
```

```python
env_trial = np.full(nt.shape, int(np.round(np.nanmedian(env[sl]))), dtype=np.float32)
```

iii. In the notes, the AI planned to “convert to binary ENV1 vs ENV2 per trial.” In the trajectory it also observed that sessions could be all environment 0, all environment 1, or mixed, so it chose a per-trial summary.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is taken from the raw `trial number` behavior series at the inferred trial starts. `infer_trial_bounds` stores that value as `tid`, and the converter broadcasts `tid` across the neural bins for the trial.

ii.
```python
trial_num = np.asarray(b['trial number']).astype(int)
...
trial_ids = trial_num[starts].astype(int)
...
for i, (tid, s, e) in enumerate(trials):
    ...
    trialnum_trial = np.full(nt.shape, tid, dtype=np.float32)
```

iii. The notes originally proposed using the native trial numbering after reconstructing valid trials. I did not find a later justification that revisited the mismatch between `trial_start` and `trial number` described in the reference solution.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. There is no additional processing beyond extracting `tid` and repeating it across all time bins in the trial.

ii.
```python
trialnum_trial = np.full(nt.shape, tid, dtype=np.float32)
```

iii. The notes frame trial number as a per-trial covariate that can be broadcast across time bins for decoder compatibility.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from the event timestamps of the `Reward` behavior series. First the AI infers a binary reward outcome for each trial by checking whether any `Reward` event timestamp falls inside that trial’s time window; then it shifts that value by one trial.

ii.
```python
reward_event_timestamps = b.get('Reward__timestamps', np.asarray([]))
reward_outcomes = infer_reward_outcomes_from_events(trials, bt, reward_event_timestamps)
```

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

iii. The trajectory shows a clear justification: the AI first used a lick-based heuristic, found that it made reward outcome degenerate, then discovered `processing/behavior/BehavioralTimeSeries/Reward` and switched to using reward event timestamps because session reward counts matched the existence of omission trials.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI stores `prev_out = 0` before the first trial. After each trial, it updates `prev_out` to that trial’s inferred reward outcome, and for the next trial it broadcasts `prev_out` across all time bins.

ii.
```python
prev_out = 0
for i, (tid, s, e) in enumerate(trials):
    ...
    prev_out_trial = np.full(nt.shape, prev_out, dtype=np.float32)
    ...
    prev_out = int(reward_outcomes[i])
```

iii. The notes state that previous trial outcome should be a trial-level covariate, and the trajectory says the `Reward` event stream made it possible to make `previous_trial_outcome` “meaningful” rather than constant.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The AI derives this output from `position` and `reward_zone`. It infers one reward-related position per trial from the median position where `reward_zone > 0`, then uses that inferred per-trial reward position as the reference for distance.

ii.
```python
pos = np.asarray(b['position'])
rz_raw = np.asarray(b['reward_zone'])
...
trial_reward_pos, reward_loc_centers = infer_trial_reward_positions(pos, rz_raw, trials)
```

```python
def infer_trial_reward_positions(pos, rz_signal, trials):
    trial_reward_pos = {}
    for tid, s, e in trials:
        sl = slice(s, e)
        nz = rz_signal[sl] > 0
        trial_reward_pos[tid] = float(np.nanmedian(pos[sl][nz])) if np.any(nz) else np.nan
```

iii. The trajectory shows the reasoning in detail. The AI discovered that `reward_zone` was not a simple A/B/C label, observed that nonzero `reward_zone` values occurred at reward-related positions, and decided to infer reward location from those positions instead.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The AI computes a signed distance from current position to a single inferred reward-center position for that trial, not to the edges of a reward zone interval. Distances are then thresholded by the instruction-specified cutoffs around that center.

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

```python
reward_center = float(trial_reward_pos.get(tid, np.nan))
dist_bins, _ = discretize_dist_to_reward(pos[bidx], reward_center)
```

iii. The AI’s justification in the trajectory was pragmatic: once it concluded that `reward_zone` was a time-varying code rather than a zone ID, it simplified the problem to estimating a per-trial reward position and measuring distance relative to that position.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI uses the instructed category boundaries `-50`, `-10`, `0`, `10`, and `50`, but it applies them to distance from an inferred reward center, not to distance from the nearest edge of the reward zone. Category `3` is assigned only when `dist` is numerically close to zero.

ii.
```python
out = np.full(dist.shape, 0, dtype=np.int64)
out[(dist >= -50) & (dist < -10)] = 1
out[(dist >= -10) & (dist < 0)] = 2
out[np.isclose(dist, 0)] = 3
out[(dist > 0) & (dist <= 10)] = 4
out[(dist > 10) & (dist <= 50)] = 5
out[dist > 50] = 6
```

iii. I did not find a more specific justification than the trajectory’s choice to reduce reward-zone geometry to a per-trial center estimate.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The AI first selects neural timestamps for the trial, then maps those neural timestamps back onto the behavior clock with `np.searchsorted(bt, nt)`. It uses the resulting behavior indices `bidx` to sample `position` on the neural time grid.

ii.
```python
nt = neural_t[nmask]
bidx = np.searchsorted(bt, nt, side='left')
bidx = np.clip(bidx, 0, len(bt) - 1)
...
dist_bins, _ = discretize_dist_to_reward(pos[bidx], reward_center)
```

iii. The notes emphasize synchronized behavior streams and temporal alignment to trial start. The AI’s implementation justification was to make all outputs live on the neural bins rather than on the original behavior samples.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position is derived from the `position` behavior time series.

ii.
```python
pos = np.asarray(b['position'])
```

```python
out = np.vstack([
    dist_bins,
    abs_pos_bins[bidx],
    speed_bins[bidx],
```

iii. The notes and trajectory consistently identify `position` as a directly available behavior stream that should drive both absolute position and reward-relative outputs.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI computes five equal-width bins from the session’s valid position range using `np.linspace(lo, hi, 6)`, digitizes the full session position trace into those bins, and then samples the binned position at neural-aligned behavior indices.

ii.
```python
def discretize_position(pos, valid_mask, n_bins=5):
    valid_pos = pos[valid_mask]
    lo, hi = np.nanmin(valid_pos), np.nanmax(valid_pos)
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(pos, edges[1:-1], right=False), 0, n_bins - 1)
    return idx, edges
```

```python
abs_pos_bins, pos_edges = discretize_position(pos, valid_mask, n_bins=5)
```

iii. The notes map position to “5 equal bins” and say invalid out-of-track periods should be excluded. That appears to be why the AI used `valid_mask = (trial_num >= 0) & (pos > -400)` before defining the bin edges.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The thresholds are session-specific equal-width bins spanning the valid observed position range, not fixed global corridor edges.

ii.
```python
edges = np.linspace(lo, hi, n_bins + 1)
idx = np.clip(np.digitize(pos, edges[1:-1], right=False), 0, n_bins - 1)
```

iii. The AI’s justification, as stated in the notes, was to meet the instruction “5 equal-sized bins” while excluding clearly invalid off-track position values.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The full-session binned position trace is indexed by `bidx`, the behavior samples nearest to each neural bin in the trial.

ii.
```python
nt = neural_t[nmask]
bidx = np.searchsorted(bt, nt, side='left')
...
abs_pos_bins[bidx]
```

iii. The implementation follows the AI’s general alignment strategy of representing all decoded outputs on the neural bins.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick is derived from the `lick` behavior time series.

ii.
```python
lick = np.asarray(b['lick'])
```

```python
(lick[bidx] > 0.5).astype(np.int64)
```

iii. The notes identify `lick` as a canonical `BehavioralTimeSeries` variable and map it directly to the binary lick output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI thresholds the lick signal at `> 0.5` and converts the result to an integer 0/1 time series.

ii.
```python
(lick[bidx] > 0.5).astype(np.int64)
```

iii. I did not find a detailed justification for the `0.5` threshold. The trajectory only states that lick should be binary and that the behavior streams appear synchronized.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The AI samples `lick` at the neural-aligned behavior indices `bidx`, so the binary lick output has one value per neural bin.

ii.
```python
nt = neural_t[nmask]
bidx = np.searchsorted(bt, nt, side='left')
...
(lick[bidx] > 0.5).astype(np.int64)
```

iii. This follows the same timestamp-based behavior-to-neural alignment strategy used for position, speed, and distance-to-reward outputs.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward zone location is derived from `position` and `reward_zone`. The AI infers one reward-related position per trial from samples where `reward_zone > 0`, then clusters those positions into up to three session-specific centers.

ii.
```python
trial_reward_pos, reward_loc_centers = infer_trial_reward_positions(pos, rz_raw, trials)
trial_reward_loc = assign_reward_location_labels(trial_reward_pos, reward_loc_centers)
```

```python
def infer_trial_reward_positions(pos, rz_signal, trials):
    ...
    nz = rz_signal[sl] > 0
    trial_reward_pos[tid] = float(np.nanmedian(pos[sl][nz])) if np.any(nz) else np.nan
    ...
    km = KMeans(n_clusters=k, random_state=0, n_init=20).fit(vals.reshape(-1, 1))
    centers = np.sort(km.cluster_centers_.ravel())
```

iii. The trajectory contains the main justification. The AI found that raw `reward_zone` values were not direct A/B/C labels and decided that reward location had to be inferred from the positions where the reward-zone signal was active.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For each trial, the AI computes the median position of samples with `reward_zone > 0`. Across the session, it clusters the finite per-trial positions with `KMeans` into up to three sorted centers. Each trial is then labeled by the nearest center, and that label is broadcast across the neural bins of the trial.

ii.
```python
vals = np.array([v for v in trial_reward_pos.values() if np.isfinite(v)])
if len(vals) == 0:
    centers = np.array([np.nan, np.nan, np.nan])
else:
    k = min(3, len(np.unique(np.round(vals, 3))))
    km = KMeans(n_clusters=k, random_state=0, n_init=20).fit(vals.reshape(-1, 1))
    centers = np.sort(km.cluster_centers_.ravel())
```

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

```python
np.full(nt.shape, int(trial_reward_loc.get(tid, 0)), dtype=np.int64)
```

iii. The trajectory says the AI adopted this approach after observing clear switches in the inferred trial reward positions and deciding that clustering those positions was a workable proxy for A/B/C reward location.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from the event timestamps of the `Reward` behavior series.

ii.
```python
reward_event_timestamps = b.get('Reward__timestamps', np.asarray([]))
reward_outcomes = infer_reward_outcomes_from_events(trials, bt, reward_event_timestamps)
```

iii. The trajectory explicitly states that the `Reward` time series was a “key breakthrough” because it gave a direct reward-delivery event stream and explained the omission-trial counts.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The AI checks whether any `Reward` event timestamp falls between the behavior timestamps for the start and end of each inferred trial. The resulting 0/1 value is then broadcast across the neural bins of that trial.

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

```python
np.full(nt.shape, int(reward_outcomes[i]), dtype=np.int64)
```

iii. The AI justified this change in the trajectory by noting that a session could have 80 trials but only 74 reward events, which strongly suggested omission trials were exactly the trials without a `Reward` event.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI uses several fallback heuristics rather than strict checks. If neural timestamps are missing, it invents them with `np.linspace` over the behavior time span. If position timestamps are missing, it falls back to any available behavior timestamp array. If per-trial reward-zone activity is missing, the reward position becomes `NaN`, reward-zone label defaults to `0`, and distance-to-reward bins default mostly to `0`. Trials with fewer than two samples are dropped; sessions with fewer than two surviving trials are dropped.

ii.
```python
bt = b.get('position__timestamps', None)
if bt is None:
    any_ts = [v for k, v in b.items() if k.endswith('__timestamps')]
    bt = any_ts[0]
```

```python
if neural_t is None:
    if bt is None:
        raise RuntimeError('No timestamps for neural or behavior')
    neural_t = np.linspace(bt[0], bt[-1], neural_nt.shape[1])
```

```python
trial_reward_pos[tid] = float(np.nanmedian(pos[sl][nz])) if np.any(nz) else np.nan
...
if not np.isfinite(rp) or len(finite_centers) == 0:
    labels[tid] = 0
```

```python
if np.sum(nmask) < 2 or (e - s) < 2:
    continue
```

iii. I did not find a single consolidated justification in the notes. The trajectory shows a pattern of adding heuristics when a direct mapping was unclear, especially for missing timestamps and ambiguous reward-zone semantics, with decoder validation used as the main backstop.

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive work is reading every NWB file with `h5py`, materializing the behavior and neural arrays for each session, computing per-trial neural/behavior alignment, and, for sample mode, scanning files to find environment-diverse examples. Reward-location clustering with `KMeans` also adds per-session cost, but the dominant cost is file I/O plus full-array loading.

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))
...
with h5py.File(path, 'r') as f:
    ...
    b = load_behavior_series(f)
    neural, neural_t, neural_base, neural_key = load_neural_series(f)
```

```python
for i, (tid, s, e) in enumerate(trials):
    ...
    nmask = (neural_t >= t0) & (neural_t <= t1)
    ...
    bidx = np.searchsorted(bt, nt, side='left')
```

iii. There is no explicit performance discussion in `CONVERSION_NOTES.md` beyond “full dataset likely manageable in minutes.” This answer is mostly inferred from the code structure and from the trajectory’s repeated waiting during the full-conversion pass.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The largest vectorization opportunity is the per-trial loop in `convert_session`, which repeatedly constructs `nmask`, `bidx`, and stacked inputs/outputs. Reward outcome inference also loops trial by trial, and reward-position inference iterates through all trials to compute medians. Those operations could be batched if a ragged or padded representation were acceptable.

ii.
```python
for i, (tid, s, e) in enumerate(trials):
    sl = slice(s, e)
    t0 = bt[s]
    t1 = bt[e - 1]
    nmask = (neural_t >= t0) & (neural_t <= t1)
    ...
    inp = np.vstack([time_from_start, env_trial, trialnum_trial, prev_out_trial])
    ...
    out = np.vstack([
        dist_bins,
        abs_pos_bins[bidx],
        speed_bins[bidx],
        (lick[bidx] > 0.5).astype(np.int64),
        np.full(nt.shape, int(trial_reward_loc.get(tid, 0)), dtype=np.int64),
        np.full(nt.shape, int(reward_outcomes[i]), dtype=np.int64),
    ]).astype(np.int64)
```

```python
for tid, s, e in trials:
    sl = slice(s, e)
    nz = rz_signal[sl] > 0
    trial_reward_pos[tid] = float(np.nanmedian(pos[sl][nz])) if np.any(nz) else np.nan
```

iii. I did not find an explicit AI justification about vectorization tradeoffs. The closest thing in the trajectory is the AI’s bias toward getting a working end-to-end converter and using decoder verification as the main validation criterion.

## 13-c. What processing does the code repeat multiple times?

i. In `--sample` mode, the converter first scans files just to inspect environment values and choose two sessions, then reopens those same files for actual conversion. Within a session, it also derives reward-related trial structure multiple times in separate passes: once to infer trial bounds, once to infer reward positions, once to infer reward outcomes, and again during the main trial loop to build the tensors.

ii.
```python
if args.sample:
    ...
    for fp in files:
        with h5py.File(fp, 'r') as f:
            env = f['processing/behavior/BehavioralTimeSeries']['environment']['data'][:]
            tn = f['processing/behavior/BehavioralTimeSeries']['trial number']['data'][:]
```

```python
trials = infer_trial_bounds(b)
trial_reward_pos, reward_loc_centers = infer_trial_reward_positions(pos, rz_raw, trials)
trial_reward_loc = assign_reward_location_labels(trial_reward_pos, reward_loc_centers)
reward_outcomes = infer_reward_outcomes_from_events(trials, bt, reward_event_timestamps)
...
for i, (tid, s, e) in enumerate(trials):
```

iii. There is no explicit performance justification for these repeated passes. The trajectory suggests the AI was iterating on correctness and sample diversity rather than optimizing the implementation.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads `teleport`, `trial_start`, ROI-region metadata, and session ID metadata that are not included in the final tensors. It also computes `pos_edges` and stores `zone_centers`, though those are not used by downstream decoder training. Optional `show_processing` plots are also diagnostic only. In addition, the helper `choose_zone_centers` is defined but never used.

ii.
```python
teleport = np.asarray(b['teleport'])
trial_start = np.asarray(b['trial_start'])
...
n_rois, regions = load_n_rois_and_regions(f)
...
abs_pos_bins, pos_edges = discretize_position(pos, valid_mask, n_bins=5)
```

```python
return {
    'subject': str(subj),
    'session': str(sess),
    ...
    'zone_centers': reward_loc_centers.tolist() if hasattr(reward_loc_centers, 'tolist') else reward_loc_centers,
    'neural_source': f'{neural_base}/{neural_key}',
}
```

```python
def choose_zone_centers(pos, rz_code):
    centers = {}
    for z in sorted(set(rz_code[rz_code >= 0].tolist())):
        mask = rz_code == z
        if np.any(mask):
            centers[int(z)] = float(np.nanmedian(pos[mask]))
    return centers
```

iii. I did not find an explicit justification for these extra computations. They appear to be artifacts of exploratory development and optional diagnostics rather than deliberate downstream requirements.
