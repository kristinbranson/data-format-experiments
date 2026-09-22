# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent scans `/app/data` recursively for every `.nwb` file, treats each file as one session, and opens each file with `h5py`. Within each file it pulls neural data from `processing/ophys/Deconvolved/plane0/data`, behavior timestamps from `position/timestamps`, and a fixed set of behavior arrays from `processing/behavior/BehavioralTimeSeries`.

ii.
```python
def find_nwb_files():
    return sorted(Path('/app/data').rglob('*.nwb'))

def convert_session(fpath):
    with h5py.File(fpath, 'r') as f:
        subj = read_scalar(f, 'general/subject/subject_id')
        sess_id = read_scalar(f, 'general/session_id', fpath.stem)
        neural = np.asarray(f['processing/ophys/Deconvolved/plane0/data'], dtype=np.float32)
        fluor = np.asarray(f['processing/ophys/Fluorescence/plane0/data'], dtype=np.float32)
        ts = np.asarray(f['processing/behavior/BehavioralTimeSeries/position/timestamps'], dtype=np.float64)
        base = 'processing/behavior/BehavioralTimeSeries'
        beh = {k: np.asarray(f[f'{base}/{k}/data']) for k in ['environment', 'reward_zone', 'teleport', 'lick', 'speed', 'position', 'trial number', 'trial_start', 'scanning']}
```

iii. The notes say the data are NWB files under `/app/data`, and the agent chose the NWB time base directly for decoder conversion. No stronger justification for using `h5py` rather than `pynwb` was recorded.

## 1-b. How are the data split into subjects?

i. Subjects are identified from each NWB file’s `general/subject/subject_id`. The output `subjects` list is built in first-seen order while iterating over files, and each session stores an index into that list.

ii.
```python
subj = read_scalar(f, 'general/subject/subject_id')
...
if subj not in subject_to_idx:
    subject_to_idx[subj] = len(subjects)
    subjects.append(subj)
...
subject_idx.append(subject_to_idx[subj])
```

iii. The notes explicitly mention standard NWB subject metadata and 11 unique subjects in the dataset.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file is treated as one session. Session metadata are read from `general/session_id`, with the filename stem as fallback.

ii.
```python
def convert_session(fpath):
    with h5py.File(fpath, 'r') as f:
        subj = read_scalar(f, 'general/subject/subject_id')
        sess_id = read_scalar(f, 'general/session_id', fpath.stem)
...
for i, fpath in enumerate(files):
    subj, sess_id, n_neurons, n_list, in_list, out_list = convert_session(fpath)
```

iii. The notes state that the dataset consists of NWB session files and report 152 sessions total.

## 1-d. How are the data split into trials?

i. Trial starts are rising edges of `trial_start`; trial ends are rising edges of `teleport`. A candidate interval is kept only if it contains at least one sample with nonnegative `trial number`.

ii.
```python
def get_trial_bounds(trial_start, teleport, trial_number):
    starts = np.where(np.diff(trial_start.astype(int), prepend=0) > 0)[0]
    ends = np.where(np.diff(teleport.astype(int), prepend=0) > 0)[0]
    bounds = []
    for s in starts:
        e_candidates = ends[ends > s]
        if e_candidates.size == 0:
            continue
        e = int(e_candidates[0])
        if np.any(trial_number[s:e] >= 0):
            bounds.append((int(s), int(e)))
    return bounds
```

iii. The notes and README both say trials are defined from trial start to teleport and exclude off-trial samples marked by `trial number = -1`.

## 1-e. How are trials filtered based on quality controls?

i. Within each trial window, the agent keeps only samples where both `trial number >= 0` and `scanning > 0`. It discards a trial if fewer than 2 valid samples remain, and discards an entire session if fewer than 2 valid trials remain.

ii.
```python
valid = (beh['trial number'][s:e] >= 0) & (beh['scanning'][s:e] > 0)
if valid.sum() < 2:
    continue
...
if len(n_list) < 2:
    print(f'skipping {fpath.name}: fewer than 2 valid trials after filtering')
    continue
```

iii. The recorded justification is structural rather than scientific: the README says active scanning samples are required, and the trajectory cites the decoder-format requirement that each session must retain at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The converted neural data come directly from the NWB `processing/ophys/Deconvolved/plane0/data`. `Fluorescence/plane0/data` is also loaded, but only for a shape check.

ii.
```python
neural = np.asarray(f['processing/ophys/Deconvolved/plane0/data'], dtype=np.float32)
fluor = np.asarray(f['processing/ophys/Fluorescence/plane0/data'], dtype=np.float32)
...
assert neural.shape == fluor.shape
```

iii. The notes explicitly justify this by saying the Methods mention deconvolved activity as the model response matrix, so the agent preferred `Deconvolved` rather than recomputing dF/F and events.

## 2-b. How is the `neural` data processed?

i. There is no additional neural preprocessing beyond trimming all streams to a common session length, selecting valid in-trial indices, and transposing each trial to `(neurons, time)`.

ii.
```python
lengths = [neural.shape[0], ts.shape[0]] + [v.shape[0] for v in beh.values()]
common_len = min(lengths)
...
valid = (beh['trial number'][s:e] >= 0) & (beh['scanning'][s:e] > 0)
idx = np.where(valid)[0] + s
...
neu = neural[idx].T.astype(np.float32)
```

iii. The notes say the agent intentionally used the NWB deconvolved signal directly and “uses vectorized slicing and avoids per-neuron loops.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent does not apply cell-level curation such as `iscell` filtering or interneuron exclusion. Neural samples are only filtered indirectly through the trial/sample mask (`trial number >= 0` and `scanning > 0`).

ii.
```python
valid = (beh['trial number'][s:e] >= 0) & (beh['scanning'][s:e] > 0)
...
neu = neural[idx].T.astype(np.float32)
```

iii. No explicit cell-quality justification was recorded. The notes instead emphasize speed and using the deconvolved NWB signal as-is.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start by slicing each trial window from `trial_start` to `teleport`, then re-indexing the kept samples within that trial.

ii.
```python
bounds = get_trial_bounds(beh['trial_start'], beh['teleport'], beh['trial number'])
...
valid = (beh['trial number'][s:e] >= 0) & (beh['scanning'][s:e] > 0)
idx = np.where(valid)[0] + s
neu = neural[idx].T.astype(np.float32)
```

iii. The notes explicitly say to “use the NWB time base directly” and define each trial from trial start to teleport.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent does not rebin or resample. It keeps the native sample spacing of the stored arrays, but does not compute or store the actual bin size; metadata set `time_bin_size` to `None`.

ii.
```python
'metadata': {
    ...
    'time_bin_size': None,
    'temporal_alignment_event': 'trial start',
    ...
}
```

iii. The justification in the notes is only that the agent chose the NWB time base directly; it did not document an exact sampling interval.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/position/timestamps`.

ii.
```python
ts = np.asarray(f['processing/behavior/BehavioralTimeSeries/position/timestamps'], dtype=np.float64)
...
t0 = ts[idx[0]]
t_rel = (ts[idx] - t0).astype(np.float32)
```

iii. The notes say the agent used the NWB time base directly. No further justification for choosing `position/timestamps` rather than another behavior timestamp stream was recorded.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the timestamp of the first retained sample is subtracted from all retained timestamps.

ii.
```python
t0 = ts[idx[0]]
t_rel = (ts[idx] - t0).astype(np.float32)
...
inp = np.vstack([
    t_rel,
    ...
]).astype(np.float32)
```

iii. This follows directly from the decoder requirement to represent time from trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It uses the exact same retained sample indices `idx` that are used for neural slicing, so time and neural matrices have matching columns.

ii.
```python
idx = np.where(valid)[0] + s
t_rel = (ts[idx] - t0).astype(np.float32)
...
neu = neural[idx].T.astype(np.float32)
```

iii. The notes say the agent used a shared NWB time base for aligned continuous streams.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the `environment` behavior time series. For each trial the agent records the first valid nonnegative in-trial value.

ii.
```python
env_vals = beh['environment'][s:e][valid]
env_per_trial.append(int(first_valid(env_vals)) if env_vals.size else -1)
```

iii. The notes say the raw NWB `environment` stream is the source for the binary ENV1/ENV2 decoder input.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The code computes a session-local mapping from unique nonnegative environment codes to `{0, 1}`, but never applies it. The actual input uses the first valid in-trial environment value, broadcast across the trial, with negative values replaced by `0`.

ii.
```python
def compute_env_mapping(env_trials):
    vals = sorted({int(v) for v in env_trials if v >= 0})
    return {v: i for i, v in enumerate(vals[:2])}
...
env_map = compute_env_mapping(env_per_trial)
...
np.full_like(t_rel, float(env_per_trial[i] if env_per_trial[i] >= 0 else 0), dtype=np.float32)
```

iii. The notes say the plan was to map valid environment codes to binary values after excluding `-1` off-trial samples. No explicit explanation was given for why the computed `env_map` was left unused.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the `trial number` behavior time series by taking the first valid in-trial value for each trial.

ii.
```python
tn_vals = beh['trial number'][s:e][valid]
trial_nums.append(float(first_valid(tn_vals)) if tn_vals.size else len(trial_nums))
```

iii. The notes say the agent would likely use the observed trial number stored during valid trial samples.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The selected per-trial value is broadcast across all retained time bins of that trial. If a trial has no valid stored value, the fallback is the running trial count.

ii.
```python
trial_nums.append(float(first_valid(tn_vals)) if tn_vals.size else len(trial_nums))
...
np.full_like(t_rel, trial_nums[i], dtype=np.float32),
```

iii. The notes present this as a per-trial continuous input rather than a time-varying signal.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the NWB `Reward` series, using `Reward/data` and, when needed, `Reward/timestamps`.

ii.
```python
reward = np.asarray(f[f'{base}/Reward/data']) if f'{base}/Reward/data' in f else None
reward_ts = np.asarray(f[f'{base}/Reward/timestamps']) if f'{base}/Reward/timestamps' in f else None
...
if reward is not None and reward_ts is not None and reward.shape[0] != ts.shape[0]:
    t_start = ts[s]
    t_end = ts[e-1] if e-1 < len(ts) else ts[-1]
    m_rew = (reward_ts >= t_start) & (reward_ts <= t_end)
    reward_outcomes.append(int(np.any(reward[m_rew] > 0)))
elif reward is not None:
    reward_outcomes.append(int(np.any(reward[s:e][valid] > 0)))
```

iii. The notes and README explicitly say reward outcome comes from `Reward` and, when necessary, is aligned using its separate timestamps.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The agent first computes a binary reward outcome for each trial (`any reward > 0` in that trial’s window). The input for trial `i` is then the previous trial’s outcome, with the first trial forced to `0`.

ii.
```python
reward_outcomes.append(int(np.any(reward[m_rew] > 0)))
...
np.full_like(t_rel, reward_outcomes[i-1] if i > 0 else 0, dtype=np.float32),
```

iii. The notes state that the intended rule was “shift per-trial reward outcome by one trial; first trial default 0.”

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from trial position plus an inferred per-trial reward-zone center. That center is estimated from positions where `reward_zone > 0`; the raw `reward_zone` values are not used as final labels directly.

ii.
```python
pos_vals = beh['position'][s:e][valid]
rz_vals = beh['reward_zone'][s:e][valid]
m = rz_vals > 0
rz_centers.append(float(np.nanmedian(pos_vals[m])) if np.any(m) else None)
...
pos = beh['position'][idx].astype(np.float32)
rz_center = rz_centers[i] if i < len(rz_centers) else None
```

iii. The notes explicitly call this reward-zone inference “provisional” and say the agent inferred A/B/C categories from reward-zone-associated positions.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each trial, the agent computes `dist = position - inferred_reward_zone_center`, so the value is signed relative to the center rather than to the nearest reward-zone edge. If no reward-zone center can be inferred, it falls back to the trial median position.

ii.
```python
if rz_center is None or not np.isfinite(rz_center):
    rz_center = float(np.nanmedian(pos))
...
dist = pos - float(rz_center)
```

iii. The notes justify this only indirectly, by saying reward-zone identity was inferred from positions with `reward_zone > 0`.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous signed distance is discretized into 7 bins: `< -50`, `[-50, -10]`, `(-10, 0)`, `0`, `(0, 10]`, `(10, 50]`, and `> 50`.

ii.
```python
def discretize_distance(d):
    out = np.full(d.shape, -1, dtype=np.int64)
    out[d < -50] = 0
    out[(d >= -50) & (d <= -10)] = 1
    out[(d > -10) & (d < 0)] = 2
    out[np.isclose(d, 0)] = 3
    out[(d > 0) & (d <= 10)] = 4
    out[(d > 10) & (d <= 50)] = 5
    out[d > 50] = 6
```

iii. The bin edges match the task specification; no additional justification beyond that was recorded.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from `position[idx]` on the same retained sample indices used for neural, input time, speed, and lick.

ii.
```python
idx = np.where(valid)[0] + s
pos = beh['position'][idx].astype(np.float32)
...
dist = pos - float(rz_center)
...
neu = neural[idx].T.astype(np.float32)
```

iii. The notes repeatedly state that the agent used the NWB time base directly for all aligned streams.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from the `position` behavior time series.

ii.
```python
pos = beh['position'][idx].astype(np.float32)
```

iii. The notes identify `position` as the source variable for track location.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The agent clips position to `[0, 450]` cm and then discretizes it into five bins.

ii.
```python
discretize_position(np.clip(pos, 0, TRACK_LEN))
```

iii. The notes say the main track representation uses a 450 cm corridor; no explicit rationale for clipping was recorded.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Position is thresholded into five bins: `<90`, `[90,180)`, `[180,270)`, `[270,360)`, and `>=360`.

ii.
```python
def discretize_position(pos):
    out = np.full(pos.shape, -1, dtype=np.int64)
    out[pos < 90] = 0
    out[(pos >= 90) & (pos < 180)] = 1
    out[(pos >= 180) & (pos < 270)] = 2
    out[(pos >= 270) & (pos < 360)] = 3
    out[pos >= 360] = 4
```

iii. The notes tie this to the 450 cm track length required by the task.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is sliced on the same retained indices `idx` used for the per-trial neural data.

ii.
```python
idx = np.where(valid)[0] + s
pos = beh['position'][idx].astype(np.float32)
neu = neural[idx].T.astype(np.float32)
```

iii. The notes say the continuous behavior and neural streams are aligned on the shared NWB sample axis.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the `lick` behavior time series.

ii.
```python
lick = (beh['lick'][idx] > 0).astype(np.int64)
```

iii. The notes identify the raw NWB lick stream as the source and describe binarization as task-driven.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The lick signal is binarized as `1` when `lick > 0` and `0` otherwise.

ii.
```python
lick = (beh['lick'][idx] > 0).astype(np.int64)
...
out = np.vstack([
    ...
    lick,
    ...
]).astype(np.int64)
```

iii. The notes and README say the decoder requires binary lick output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It uses the same retained sample indices `idx` as the neural trial slice.

ii.
```python
idx = np.where(valid)[0] + s
lick = (beh['lick'][idx] > 0).astype(np.int64)
neu = neural[idx].T.astype(np.float32)
```

iii. The notes say all trial-wise streams are aligned on the NWB time base.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived jointly from `reward_zone` and `position`, via inferred per-trial reward-zone centers.

ii.
```python
pos_vals = beh['position'][s:e][valid]
rz_vals = beh['reward_zone'][s:e][valid]
m = rz_vals > 0
rz_centers.append(float(np.nanmedian(pos_vals[m])) if np.any(m) else None)
```

iii. The notes say raw `reward_zone` is not a direct A/B/C label, so the agent inferred categories from reward-zone-associated positions.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The agent clusters inferred reward-zone centers within a session, merges nearby centers if they differ by at most 30 cm, assigns sorted clusters to categories `0,1,2` (A/B/C), and then labels each trial by the nearest discovered center. If no mapping exists, it defaults to `0`.

ii.
```python
def compute_rz_mapping_from_centers(centers):
    vals = sorted({round(float(v), 1) for v in centers if v is not None and np.isfinite(v)})
    if not vals:
        return {}
    merged = []
    for v in vals:
        if not merged or abs(v - merged[-1]) > 30:
            merged.append(v)
    return {v: min(i, 2) for i, v in enumerate(merged[:3])}
...
if rz_map:
    nearest = min(rz_map.keys(), key=lambda x: abs(x - rz_center))
    rz_cat = rz_map[nearest]
else:
    rz_cat = 0
```

iii. The notes call this solution provisional and document that reward-zone location was initially wrong before this position-based inference was added.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the NWB `Reward` series, using either direct indexing or reward timestamps depending on shape compatibility.

ii.
```python
reward = np.asarray(f[f'{base}/Reward/data']) if f'{base}/Reward/data' in f else None
reward_ts = np.asarray(f[f'{base}/Reward/timestamps']) if f'{base}/Reward/timestamps' in f else None
```

iii. The notes and README explicitly say reward outcome is derived from `BehavioralTimeSeries/Reward`.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, reward outcome is `1` if any reward sample in that trial window is positive, else `0`. It is then broadcast across the full trial.

ii.
```python
if reward is not None and reward_ts is not None and reward.shape[0] != ts.shape[0]:
    ...
    reward_outcomes.append(int(np.any(reward[m_rew] > 0)))
elif reward is not None:
    reward_outcomes.append(int(np.any(reward[s:e][valid] > 0)))
...
np.full(t_rel.shape, reward_outcomes[i], dtype=np.int64),
```

iii. The notes say the intended rule was “any reward > 0” within the trial.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles several cases pragmatically: session stream length mismatches are trimmed to the common minimum; missing `Reward` causes reward outcome to default to `0`; missing or non-finite reward-zone center falls back to the trial median position; negative or absent environment values fall back to `0`; trials with fewer than 2 retained samples are dropped; sessions with fewer than 2 valid trials are skipped.

ii.
```python
common_len = min(lengths)
if len(set(lengths)) != 1:
    print(f'length mismatch in {fpath.name}: lengths={lengths}, trimming to {common_len}')
...
if reward is not None and reward_ts is not None and reward.shape[0] != ts.shape[0]:
    ...
elif reward is not None:
    ...
else:
    reward_outcomes.append(0)
...
if rz_center is None or not np.isfinite(rz_center):
    rz_center = float(np.nanmedian(pos))
...
np.full_like(t_rel, float(env_per_trial[i] if env_per_trial[i] >= 0 else 0), dtype=np.float32)
...
if valid.sum() < 2:
    continue
```

iii. The trajectory explicitly records the length-trimming fix after a full conversion failure, and the README repeats the length-mismatch and reward-timestamp handling. The reward-zone fallback is described in the notes as provisional.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive parts are reading large NWB arrays for each session and iterating over all sessions/trials during conversion. The code loads full session arrays into memory, then loops over every trial at least twice before assembling outputs.

ii.
```python
files = find_nwb_files()
...
for i, fpath in enumerate(files):
    subj, sess_id, n_neurons, n_list, in_list, out_list = convert_session(fpath)
```

```python
for s, e in bounds:
    ...
for i, (s, e) in enumerate(bounds):
    ...
for i, (s, e) in enumerate(bounds):
    ...
```

iii. The notes explicitly mention that the converter “loads full session arrays into memory” and that this is the main efficiency concern.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The two per-trial preprocessing passes over `bounds` and the final per-trial assembly loop could be collapsed or partially vectorized. The nearest-center label assignment also does a Python `min(..., key=...)` for every trial.

ii.
```python
for s, e in bounds:
    ...
for i, (s, e) in enumerate(bounds):
    ...
for i, (s, e) in enumerate(bounds):
    ...
    nearest = min(rz_map.keys(), key=lambda x: abs(x - rz_center))
```

iii. The notes say the agent already “uses vectorized slicing and avoids per-neuron loops,” which implies the remaining trial loops were accepted as the main non-vectorized structure.

## 13-c. What processing does the code repeat multiple times?

i. The code makes three passes over the same trial bounds in each session: one to extract per-trial environment/trial-number/reward outcomes, one to estimate reward-zone centers, and one to build neural/input/output trial arrays. It also loads fluorescence even though the downstream data use only deconvolved activity.

ii.
```python
for s, e in bounds:
    ...
for i, (s, e) in enumerate(bounds):
    ...
for i, (s, e) in enumerate(bounds):
    ...
neural = np.asarray(f['processing/ophys/Deconvolved/plane0/data'], dtype=np.float32)
fluor = np.asarray(f['processing/ophys/Fluorescence/plane0/data'], dtype=np.float32)
```

iii. No explicit justification was recorded for the repeated trial passes. The only relevant note is that loading full arrays into memory was considered acceptable.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several values are computed or loaded but never used downstream: `fluor` is loaded only for `assert neural.shape == fluor.shape`; `rz_series` is read and unused; `rz_per_trial` is collected and unused; `env_map` is computed and unused; `reward_zone_label_from_position()` is defined but unused; `n_time` is assigned but unused.

ii.
```python
fluor = np.asarray(f['processing/ophys/Fluorescence/plane0/data'], dtype=np.float32)
...
assert neural.shape == fluor.shape
...
rz_per_trial = []
...
rz_per_trial.append(int(first_valid(rz_vals)) if rz_vals.size else -1)
...
env_map = compute_env_mapping(env_per_trial)
...
rz_series = beh['reward_zone'][idx].astype(np.float32)
...
def reward_zone_label_from_position(rz_val):
    # provisional mapping by sorted unique in-trial zone codes to A/B/C
    return rz_val
```

iii. No explicit justification was recorded for these leftovers. The notes instead describe some of the reward-zone logic as provisional, which is consistent with partially unused scaffolding remaining in the file.
