# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads every `.nwb` file under `data/` with `Path('data').rglob('*.nwb')`, opens each file with `h5py.File`, and converts each file as one session. Trial data are then derived inside `convert_session`.

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))
...
for fp in files:
    print('Converting', fp, flush=True)
    sess = convert_session(fp, show_processing=args.show_processing)
```

iii. In `CONVERSION_NOTES.md`, the AI says the NWB files are the source of truth and that the dataset is organized as `data/sub-*/sub-*_ses-*_behavior+ophys.nwb`, so it chose to iterate over all NWB files directly.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are taken from the NWB metadata field `general/subject/subject_id`, then deduplicated and sorted after session conversion.

ii.
```python
with h5py.File(path, 'r') as f:
    subj = decode_scalar(f['general/subject/subject_id'][()])
...
subjects = sorted(set(s['subject'] for s in sessions))
subj_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The notes state that each NWB file contains subject metadata and that the release has 11 mice, so the AI used the file metadata rather than directory names as the canonical subject identity.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as a single session. The converted dataset stores one top-level session entry per file that survives basic trial-count filtering.

ii.
```python
for fp in files:
    sess = convert_session(fp, show_processing=args.show_processing)
    if sess['n_trials'] >= 2:
        sessions.append(sess)
```

iii. `CONVERSION_NOTES.md` explicitly says the data are organized as one NWB file per subject/session and that each file contains both behavior and ophys streams, which is the justification the AI used.

## 1-d. How are the data split into trials?

i. Trials are defined primarily by rising edges in `trial_start`. Trial ends are set to the next detected trial start, with a fallback to changes in `trial number` if no `trial_start` rising edges are found. The AI does not use `teleport` to terminate trials.

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
    ...
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(trial_num)
        if e - s > 1:
            bounds.append((int(trial_ids[i]), int(s), int(e)))
```

iii. In the trajectory and notes, the AI says trial structure had to be reconstructed from behavior time series and that `trial_start` and `trial number` were enough to support trial reconstruction.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept if both the behavior segment and the neural segment have at least 2 samples. Sessions are kept only if they retain at least 2 trials. There is no explicit minimum-length filter like the human reference’s 50-timepoint threshold.

ii.
```python
for i, (tid, s, e) in enumerate(trials):
    ...
    nmask = (neural_t >= t0) & (neural_t <= t1)
    if np.sum(nmask) < 2 or (e - s) < 2:
        continue
...
if sess['n_trials'] >= 2:
    sessions.append(sess)
else:
    print('Skipping session with <2 trials after processing:', fp, flush=True)
```

iii. The notes only justify a coarse validity rule: “There needs to be at least two trials within each session” and that pre-task or invalid behavior periods should be excluded. No separate rationale for a stronger trial-quality threshold was documented.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` from the first dataset it finds under `processing/ophys/DfOverF`, or falls back to `processing/ophys/Fluorescence` if `DfOverF` is absent.

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

iii. The notes say “Use processed imaging activity rather than raw fluorescence when available” and explicitly mention preferring NWB `DfOverF` because the reference code contains `dff` processing.

## 2-b. How is the `neural` data processed?

i. The AI performs only light processing: it loads one 2D neural array, infers whether it is `neurons x time` or `time x neurons`, transposes if needed, casts trial slices to `float32`, and uses timestamps if present. It does not compute deconvolution or concatenate multiple planes.

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
...
neural_trial = neural_nt[:, nmask].astype(np.float32)
```

iii. The notes justify using already processed imaging activity and matching raw NWB traces with `np.allclose()`. No additional neural preprocessing rationale was documented beyond using “processed imaging activity.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not apply explicit neuron quality filtering. It loads all ROIs from the chosen neural dataset and assigns every neuron to CA1.

ii.
```python
brain_region_idx = np.zeros(neural_nt.shape[0], dtype=np.int64)
...
'brain_regions': ['CA1'],
'brain_region_idx': [s['brain_region_idx'] for s in sessions],
```

iii. The notes say cell curation existed in the reference code but that conversion would “start from all valid neural ROIs unless reference code indicates a required exclusion.” No `iscell`-style filter was implemented.

## 2-d. How is the per-trial `neural` data aligned to the event described in the instructions?

i. For each trial, the AI uses the behavior timestamp at the trial start as the alignment event, then keeps neural samples whose timestamps fall between the trial’s start and end times.

ii.
```python
t0 = bt[s]
t1 = bt[e - 1]
nmask = (neural_t >= t0) & (neural_t <= t1)
...
neural_trial = neural_nt[:, nmask].astype(np.float32)
time_from_start = (nt - t0).astype(np.float32)
```

iii. The notes explicitly say “Represent trials in time bins aligned to trial start,” because the decoder instructions required temporal alignment to trial start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI does not rebin the data. It uses the native neural timestamps if present; otherwise it creates evenly spaced neural timestamps spanning the behavior time range. It leaves `metadata['time_bin_size']` as `NaN`.

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
```

iii. No explicit numeric time-bin justification was recorded. The closest note is that the AI wanted to preserve synchronized NWB time series rather than add extra resampling.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from behavior timestamps, preferring `position__timestamps` and otherwise using the first available behavior timestamp stream.

ii.
```python
bt = b.get('position__timestamps', None)
if bt is None:
    any_ts = [v for k, v in b.items() if k.endswith('__timestamps')]
    bt = any_ts[0]
```

iii. The notes say NWB behavior time series are the canonical alignment source and that the behavior streams share a common time base, which is the AI’s justification for using one behavior timestamp stream as the master clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The AI subtracts the trial-start timestamp from each neural sample timestamp in that trial.

ii.
```python
time_from_start = (nt - t0).astype(np.float32)
```

iii. The notes say trials are aligned to trial start, so time-from-start is simply the aligned neural time axis.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The AI computes the time-from-start values directly from the neural timestamps `nt`, so the time input is exactly on the neural time grid.

ii.
```python
nt = neural_t[nmask]
...
time_from_start = (nt - t0).astype(np.float32)
inp = np.vstack([time_from_start, env_trial, trialnum_trial, prev_out_trial])
```

iii. The notes justify using NWB behavior as the canonical alignment source and then sampling all outputs/inputs onto the neural timeline.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the behavior variable `environment`.

ii.
```python
env = map_environment(np.asarray(b['environment']))
```

iii. The mapping plan in `CONVERSION_NOTES.md` explicitly maps `processing/behavior/BehavioralTimeSeries/environment` to the decoder input “environment type.”

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI first binarizes the raw environment codes with `map_environment`, then reduces each trial to a single per-trial environment value using the rounded median and broadcasts it across neural time bins.

ii.
```python
def map_environment(x):
    vals = np.unique(x[np.isfinite(x)])
    ...
    if len(uniq) >= 2:
        lo, hi = uniq[0], uniq[-1]
        return np.where(x == hi, 1, 0)
    return (x > 0).astype(np.int64)
...
env_trial = np.full(nt.shape, int(np.round(np.nanmedian(env[sl]))), dtype=np.float32)
```

iii. The notes say the decoder input should be binary ENV1 vs ENV2 and that per-trial covariates can be broadcast across trial time bins.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The AI uses the `trial number` values sampled at detected trial starts to define a trial ID `tid`, then broadcasts that ID within the trial.

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

iii. The notes originally planned to use the native `trial number` variable after reconstructing valid trials, and the trajectory says `trial_start` plus `trial number` looked sufficient for trial reconstruction.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No extra transformation is applied beyond making it constant within each trial on the neural time grid.

ii.
```python
trialnum_trial = np.full(nt.shape, tid, dtype=np.float32)
inp = np.vstack([time_from_start, env_trial, trialnum_trial, prev_out_trial])
```

iii. The notes say per-trial covariates can be broadcast across trial time bins for decoder compatibility.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the reward-event timestamps stored in `BehavioralTimeSeries/Reward`.

ii.
```python
reward_event_timestamps = b.get('Reward__timestamps', np.asarray([]))
reward_outcomes = infer_reward_outcomes_from_events(trials, bt, reward_event_timestamps)
```

iii. The notes explicitly say the initial lick-based reward inference was wrong and was corrected to use NWB `BehavioralTimeSeries/Reward` event timestamps.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The AI first computes a per-trial `reward_outcomes` array, then uses a running `prev_out` variable so each trial receives the previous trial’s outcome, with the first trial set to 0.

ii.
```python
prev_out = 0
for i, (tid, s, e) in enumerate(trials):
    ...
    prev_out_trial = np.full(nt.shape, prev_out, dtype=np.float32)
    ...
    prev_out = int(reward_outcomes[i])
```

iii. The notes say previous-trial outcome should be inferred only after reward outcome is derived per trial, then broadcast within the current trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from `position` plus a per-trial reward location inferred from nonzero samples of the `reward_zone` behavior signal.

ii.
```python
pos = np.asarray(b['position'])
rz_raw = np.asarray(b['reward_zone'])
...
trial_reward_pos, reward_loc_centers = infer_trial_reward_positions(pos, rz_raw, trials)
...
reward_center = float(trial_reward_pos.get(tid, np.nan))
dist_bins, _ = discretize_dist_to_reward(pos[bidx], reward_center)
```

iii. The notes say the raw `reward_zone` code was not directly usable, so the AI “resolved” this by inferring per-trial reward positions from nonzero `reward_zone` samples and clustering them into canonical locations.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The AI computes distance from the animal’s position to a single inferred reward-zone center for that trial, not to the nearest edge of a reward-zone interval.

ii.
```python
def discretize_dist_to_reward(pos, reward_center):
    dist = pos - reward_center
    out = np.full(dist.shape, 0, dtype=np.int64)
    out[(dist >= -50) & (dist < -10)] = 1
    out[(dist >= -10) & (dist < 0)] = 2
    out[np.isclose(dist, 0)] = 3
    ...
    return out, dist
```

iii. The notes justify this only indirectly: the AI concluded that per-trial reward positions inferred from the nonzero `reward_zone` signal were stable enough to use for downstream reward-relative outputs.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI thresholds the center-relative continuous distance into 7 bins matching the decoder specification: `<-50`, `-50 to -10`, `-10 to <0`, `0`, `>0 to 10`, `10 to 50`, `>50`.

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

iii. The instructions themselves supplied these categories, and the AI preserved those cut points once it had chosen its continuous distance definition.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The AI samples behavior-derived position at neural timestamps within each trial using `searchsorted`, then computes the reward-distance output on that neural time grid.

ii.
```python
nt = neural_t[nmask]
bidx = np.searchsorted(bt, nt, side='left')
bidx = np.clip(bidx, 0, len(bt) - 1)
...
dist_bins, _ = discretize_dist_to_reward(pos[bidx], reward_center)
```

iii. The notes say all decoder variables should be aligned to the neural timeline after using NWB behavior timestamps as the canonical reference.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from the behavior variable `position`.

ii.
```python
pos = np.asarray(b['position'])
...
abs_pos_bins, pos_edges = discretize_position(pos, valid_mask, n_bins=5)
```

iii. The notes map the behavior `position` series directly to the decoder output “absolute corridor position.”

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI computes a session-wide valid position range from the data, divides that range into 5 equal-width bins with `linspace`, digitizes the whole session, and then samples those bins at neural timestamps.

ii.
```python
def discretize_position(pos, valid_mask, n_bins=5):
    valid_pos = pos[valid_mask]
    lo, hi = np.nanmin(valid_pos), np.nanmax(valid_pos)
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(pos, edges[1:-1], right=False), 0, n_bins - 1)
    return idx, edges
...
out = np.vstack([
    dist_bins,
    abs_pos_bins[bidx],
```

iii. The notes say absolute position should be in 5 equal bins and that invalid out-of-track values such as `position = -500` should be excluded when computing the usable range.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is thresholded into 5 equal-width bins defined separately for each session from that session’s valid observed position range.

ii.
```python
lo, hi = np.nanmin(valid_pos), np.nanmax(valid_pos)
edges = np.linspace(lo, hi, n_bins + 1)
idx = np.clip(np.digitize(pos, edges[1:-1], right=False), 0, n_bins - 1)
```

iii. The AI’s stated rationale in the notes is the decoder instruction “5 equal-sized bins,” interpreted using the empirical valid position range in each session.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position bins are first computed on the behavior stream, then reindexed onto neural timestamps with `searchsorted` inside each trial.

ii.
```python
bidx = np.searchsorted(bt, nt, side='left')
bidx = np.clip(bidx, 0, len(bt) - 1)
...
abs_pos_bins[bidx]
```

iii. The notes justify this with the same alignment rule used for other behavior outputs: behavior provides the canonical timestamps, but final arrays should live on the neural time grid.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavior variable `lick`.

ii.
```python
lick = np.asarray(b['lick'])
```

iii. The mapping plan in the notes explicitly maps `lick` to a binary time-varying decoder output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI samples lick values at neural timestamps and binarizes them with a threshold of `> 0.5`.

ii.
```python
out = np.vstack([
    ...
    (lick[bidx] > 0.5).astype(np.int64),
    ...
]).astype(np.int64)
```

iii. The notes say lick should be a binary 0/1 time series, so the AI thresholded the sampled lick values to enforce binary categories.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned by sampling the behavior lick stream at neural timestamps within each trial.

ii.
```python
nt = neural_t[nmask]
bidx = np.searchsorted(bt, nt, side='left')
...
(lick[bidx] > 0.5).astype(np.int64)
```

iii. The notes say the decoder representation should use the neural timeline, so lick is projected from behavior timestamps onto neural timestamps.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from `reward_zone` and `position`, using nonzero `reward_zone` samples to locate the active zone in position space.

ii.
```python
def infer_trial_reward_positions(pos, rz_signal, trials):
    trial_reward_pos = {}
    for tid, s, e in trials:
        sl = slice(s, e)
        nz = rz_signal[sl] > 0
        trial_reward_pos[tid] = float(np.nanmedian(pos[sl][nz])) if np.any(nz) else np.nan
```

iii. The notes explicitly say the raw `reward_zone` code was not used directly and instead per-trial reward positions were inferred from positions of nonzero `reward_zone` samples.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI infers one reward position per trial, clusters all finite trial reward positions in a session into up to 3 centers with `KMeans`, then assigns each trial to the nearest center and broadcasts that label across the trial.

ii.
```python
vals = np.array([v for v in trial_reward_pos.values() if np.isfinite(v)])
...
km = KMeans(n_clusters=k, random_state=0, n_init=20).fit(vals.reshape(-1, 1))
centers = np.sort(km.cluster_centers_.ravel())
...
labels[tid] = int(np.argmin(np.abs(np.asarray(finite_centers) - rp)))
...
np.full(nt.shape, int(trial_reward_loc.get(tid, 0)), dtype=np.int64),
```

iii. The notes state this was a deliberate fix after the AI found that the raw `reward_zone` code did not directly match A/B/C. It therefore clustered inferred reward positions into three canonical locations.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `BehavioralTimeSeries/Reward` event timestamps.

ii.
```python
reward_event_timestamps = b.get('Reward__timestamps', np.asarray([]))
reward_outcomes = infer_reward_outcomes_from_events(trials, bt, reward_event_timestamps)
```

iii. The notes explicitly record that reward outcome was switched from a failed lick heuristic to NWB `Reward` event timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the AI checks whether any reward event timestamp falls between the trial’s start and end times, converts that to a binary label, and broadcasts it across the trial.

ii.
```python
def infer_reward_outcomes_from_events(trials, behavior_timestamps, reward_event_timestamps):
    outcomes = []
    for _, s, e in trials:
        t0 = behavior_timestamps[s]
        t1 = behavior_timestamps[e - 1]
        rewarded = int(np.any((reward_event_timestamps >= t0) & (reward_event_timestamps <= t1)))
        outcomes.append(rewarded)
...
np.full(nt.shape, int(reward_outcomes[i]), dtype=np.int64),
```

iii. The notes justify this as the correct fix for omission-vs-reward detection after the original heuristic marked all trials as rewarded.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI uses several fallbacks rather than strict validation: it falls back to the first available behavior timestamp stream if `position__timestamps` is missing, synthesizes neural timestamps with `linspace` if missing, assigns `NaN` reward positions a default zone label `0` and distance bin `0`, clips `searchsorted` indices into valid bounds, and drops extremely short trials or sessions with too few retained trials.

ii.
```python
bt = b.get('position__timestamps', None)
if bt is None:
    any_ts = [v for k, v in b.items() if k.endswith('__timestamps')]
    bt = any_ts[0]
...
if neural_t is None:
    ...
    neural_t = np.linspace(bt[0], bt[-1], neural_nt.shape[1])
...
if not np.isfinite(rp) or len(finite_centers) == 0:
    labels[tid] = 0
...
out[np.isnan(dist)] = 0
...
bidx = np.clip(bidx, 0, len(bt) - 1)
...
if np.sum(nmask) < 2 or (e - s) < 2:
    continue
```

iii. The notes frame these as practical fixes needed because the released NWB data did not directly expose the same structures as the original paper code, and they emphasize “use NWB as source of truth while matching processing logic as closely as possible.”

## 13-a. What are the most time-consuming steps of the code?

i. The likely bottlenecks are repeatedly opening large NWB files with `h5py`, reading full behavior/neural arrays for each session, performing per-trial neural/behavior alignment inside `convert_session`, and clustering reward-zone positions with `KMeans` for every session.

ii.
```python
with h5py.File(path, 'r') as f:
    ...
    neural, neural_t, neural_base, neural_key = load_neural_series(f)
...
for i, (tid, s, e) in enumerate(trials):
    ...
    bidx = np.searchsorted(bt, nt, side='left')
...
km = KMeans(n_clusters=k, random_state=0, n_init=20).fit(vals.reshape(-1, 1))
```

iii. The notes repeatedly describe the dataset as large NWB sessions with many ROIs and say full conversion should still be “manageable in minutes,” implying file I/O dominates runtime.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_session` could be reduced by precomputing more behavior-derived outputs on the full session timeline and only slicing once. The per-trial reward outcome loop and the per-trial reward-position loop are also vectorization candidates.

ii.
```python
for tid, s, e in trials:
    ...
for _, s, e in trials:
    t0 = behavior_timestamps[s]
    t1 = behavior_timestamps[e - 1]
    rewarded = int(np.any((reward_event_timestamps >= t0) & (reward_event_timestamps <= t1)))
...
for tid, s, e in trials:
    sl = slice(s, e)
    nz = rz_signal[sl] > 0
```

iii. No explicit optimization rationale was documented beyond the notes’ placeholder “Code inefficiencies identified,” so this is inferred from the implemented loop structure.

## 13-c. What processing does the code repeat multiple times?

i. In `--sample` mode, the AI scans NWB files once to find an env0-only and env1-only session and then opens the selected files again for conversion. Within a session, it repeatedly performs `searchsorted` and trial-wise broadcasting for each trial.

ii.
```python
if args.sample:
    for fp in files:
        with h5py.File(fp, 'r') as f:
            env = f['processing/behavior/BehavioralTimeSeries']['environment']['data'][:]
            tn = f['processing/behavior/BehavioralTimeSeries']['trial number']['data'][:]
...
for i, (tid, s, e) in enumerate(trials):
    ...
    bidx = np.searchsorted(bt, nt, side='left')
```

iii. The notes explicitly say the sample-selection logic was revised after the first sample was not diverse enough in environment, which explains the extra pre-pass in `--sample` mode.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script does some unused work: `teleport`, `trial_start`, `regions`, `pos_edges`, and `chosen` are read but not used downstream; `choose_zone_centers` is defined but never called; `Counter` is imported but unused; and optional plotting code produces figures not consumed by the converted dataset.

ii.
```python
from collections import Counter
...
def choose_zone_centers(pos, rz_code):
    ...
teleport = np.asarray(b['teleport'])
trial_start = np.asarray(b['trial_start'])
...
abs_pos_bins, pos_edges = discretize_position(pos, valid_mask, n_bins=5)
...
chosen = []
```

iii. No explicit justification was recorded for these discarded computations; they appear to be leftovers from exploratory analysis and optional diagnostics noted in `CONVERSION_NOTES.md`.
