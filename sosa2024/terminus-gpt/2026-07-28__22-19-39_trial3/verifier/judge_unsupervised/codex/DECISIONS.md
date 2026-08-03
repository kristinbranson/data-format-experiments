# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads every `.nwb` file under `data/` with `Path('data').rglob('*.nwb')`. Each file is treated as one session; inside each file it reads behavior from `processing/behavior/BehavioralTimeSeries` and neural data from `processing/ophys/DfOverF` or, if absent, `processing/ophys/Fluorescence`.

ii. ```python
files = sorted(Path('data').rglob('*.nwb'))

with h5py.File(path, 'r') as f:
    b = load_behavior_series(f)
    neural, neural_t, neural_base, neural_key = load_neural_series(f)
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the agent said it would use NWB behavior time series as the canonical source and NWB ophys traces as the neural source. The trajectory shows it settled on direct NWB loading after concluding the mounted release was NWB-per-session rather than the paper code’s original session objects.

## 1-b. How are the data split into subjects?

i. Subjects are identified from each file’s `general/subject/subject_id`, then unique subject names are sorted into `subjects`; `subject_idx` maps each kept session to that sorted list.

ii. ```python
subj = decode_scalar(f['general/subject/subject_id'][()])
subjects = sorted(set(s['subject'] for s in sessions))
subj_to_idx = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.asarray([subj_to_idx[s['subject']] for s in sessions], dtype=np.int64),
```

iii. The notes describe one NWB file per subject/session and report 11 mice. The trajectory shows the agent relied on file metadata, not directory names, for the final subject split.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. `convert_session()` returns one session dict, and `main()` appends it if at least two processed trials remain.

ii. ```python
for fp in files:
    sess = convert_session(fp, show_processing=args.show_processing)
    if sess['n_trials'] >= 2:
        sessions.append(sess)
```

iii. Step 2 notes say the data are organized as `data/sub-*/sub-*_ses-*_behavior+ophys.nwb` and that each file corresponds to one subject/session.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from the behavior streams. The agent uses rising edges of `trial_start`; if none are found, it falls back to positive changes in `trial number`. Trial ends are defined by the next detected start, not by `teleport`.

ii. ```python
starts = rising_edges(trial_start, 0.5)
starts = starts[valid[starts]]
if len(starts) == 0:
    changes = np.where(np.diff(trial_num) > 0)[0] + 1
    starts = changes[valid[changes]]

for i, s in enumerate(starts):
    e = starts[i + 1] if i + 1 < len(starts) else len(trial_num)
    if e - s > 1:
        bounds.append((int(trial_ids[i]), int(s), int(e)))
```

iii. Step 5 notes say the agent planned to reconstruct trials from `trial_start` and/or `trial number` because it did not find a standard trial table in NWB. The trajectory explicitly records this as the chosen workaround.

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps only trial starts whose `trial number >= 0`, discards trial bounds shorter than 2 behavior samples, skips trials with fewer than 2 neural bins or fewer than 2 behavior bins inside the span, and drops sessions with fewer than 2 surviving trials.

ii. ```python
valid = trial_num >= 0
starts = starts[valid[starts]]
...
if e - s > 1:
    bounds.append((int(trial_ids[i]), int(s), int(e)))
...
if np.sum(nmask) < 2 or (e - s) < 2:
    continue
...
if sess['n_trials'] >= 2:
    sessions.append(sess)
```

iii. In Step 5 notes the agent wrote that pre-task periods such as `trial number = -1` and out-of-track periods like `position = -500` should be excluded. The final code only implements the `trial number >= 0` and minimum-length checks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes from the first dataset found under `processing/ophys/DfOverF/*/data`; if that is absent, it falls back to `processing/ophys/Fluorescence/*/data`.

ii. ```python
for base in ['processing/ophys/DfOverF', 'processing/ophys/Fluorescence']:
    if base in f:
        grp = f[base]
        for k in grp.keys():
            g = grp[k]
            if 'data' in g:
                data = np.asarray(g['data'][:])
```

iii. Step 5 notes say “Prefer NWB `DfOverF` if present to match reference processed activity.” The trajectory shows this was a deliberate choice, even though the dataset inspection mainly reported fluorescence-like groups.

## 2-b. How is the `neural` data processed?

i. The agent only reorients the matrix to neuron-by-time, optionally synthesizes timestamps with `linspace` if missing, slices by trial, and casts each trial to `float32`. It does not deconvolve or normalize in the conversion script.

ii. ```python
if neural.shape[0] == n_rois:
    neural_nt = neural
elif neural.shape[1] == n_rois:
    neural_nt = neural.T
...
if neural_t is None:
    neural_t = np.linspace(bt[0], bt[-1], neural_nt.shape[1])
...
neural_trial = neural_nt[:, nmask].astype(np.float32)
```

iii. The notes frame this as “use processed imaging activity rather than raw fluorescence when available,” but the trajectory after script creation acknowledges the implementation was still provisional and mostly an alignment/orientation pass.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is effectively no neuron-level QC filter in the final script. All ROIs in the chosen neural matrix are retained; the only filtering is indirect trial/session removal.

ii. ```python
brain_region_idx = np.zeros(neural_nt.shape[0], dtype=np.int64)
return {
    ...
    'brain_region_idx': brain_region_idx,
```

iii. Step 4 and Step 5 notes explicitly say the agent would “start from all valid neural ROIs unless reference code indicates a required exclusion,” and it never added any `iscell`, interneuron, or similar filter afterward.

## 2-d. How is the per-trial `neural` data aligned to the event described in the instructions?

i. For each reconstructed trial, the agent finds neural bins whose timestamps fall between behavior times `t0=bt[s]` and `t1=bt[e-1]`. Time zero is the trial start, and every neural sample inside that window is retained.

ii. ```python
t0 = bt[s]
t1 = bt[e - 1]
nmask = (neural_t >= t0) & (neural_t <= t1)
nt = neural_t[nmask]
neural_trial = neural_nt[:, nmask].astype(np.float32)
time_from_start = (nt - t0).astype(np.float32)
```

iii. Step 5 notes say trials should be “represented in time bins aligned to trial start,” because the decoder task explicitly required trial-start alignment even though the paper analyses were mainly position-binned.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The script keeps the native neural sample spacing for each session/trial, but it does not compute or store the actual bin size and leaves `metadata['time_bin_size']` as `NaN`.

ii. ```python
if neural_t is None:
    neural_t = np.linspace(bt[0], bt[-1], neural_nt.shape[1])
...
'metadata': {
    ...
    'time_bin_size': float(np.nan),
```

iii. In the trajectory after Step 6, the agent explicitly noted that `time_bin_size` was still `NaN` and that the script was provisional in this respect.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from behavior timestamps at the detected trial start (`bt[s]`) and the per-bin neural timestamps `nt`. `bt` is taken from `position__timestamps` if available, otherwise from any behavior timestamp stream.

ii. ```python
bt = b.get('position__timestamps', None)
if bt is None:
    any_ts = [v for k, v in b.items() if k.endswith('__timestamps')]
    bt = any_ts[0]
...
t0 = bt[s]
time_from_start = (nt - t0).astype(np.float32)
```

iii. The notes say NWB behavior time series should be the canonical alignment source, and the trajectory shows the agent used trial-start behavior timestamps to anchor continuous within-trial time.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each retained neural sample inside a trial, the agent subtracts the trial-start timestamp and stores the result as a continuous float row in the input matrix.

ii. ```python
nt = neural_t[nmask]
time_from_start = (nt - t0).astype(np.float32)
inp = np.vstack([time_from_start, env_trial, trialnum_trial, prev_out_trial])
```

iii. This follows the Step 5 mapping entry “time from start of trial: continuous time in seconds for each bin.”

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is defined directly on the retained neural timestamps `nt`, so it has exactly one value per neural time bin in each trial.

ii. ```python
nt = neural_t[nmask]
neural_trial = neural_nt[:, nmask].astype(np.float32)
time_from_start = (nt - t0).astype(np.float32)
```

iii. The agent’s plan in Step 5 was to broadcast all covariates at the trial’s time-bin resolution; here the time row is computed on the neural bins themselves, so no extra interpolation is used.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/environment`.

ii. ```python
env = map_environment(np.asarray(b['environment']))
```

iii. Step 5 notes map `processing/behavior/BehavioralTimeSeries/environment` directly to decoder input “environment type.”

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The environment stream is converted to a binary 0/1 code by `map_environment()`. Then, within each trial, the median environment value is taken and broadcast across all neural time bins in that trial.

ii. ```python
def map_environment(x):
    vals = np.unique(x[np.isfinite(x)])
    vals = [v for v in vals if v >= 0 or v == -1 or v == 1]
    uniq = sorted(set(v for v in vals if v != -1))
    if len(uniq) >= 2:
        lo, hi = uniq[0], uniq[-1]
        return np.where(x == hi, 1, 0)
    return (x > 0).astype(np.int64)

env_trial = np.full(nt.shape, int(np.round(np.nanmedian(env[sl]))), dtype=np.float32)
```

iii. In Step 5 the agent wrote that native values such as `±1` might need mapping to `0/1` consistently and that per-trial covariates should be broadcast across time bins.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/trial number`, specifically the integer trial id attached to each detected trial start.

ii. ```python
trial_num = np.asarray(b['trial number']).astype(int)
...
trial_ids = trial_num[starts].astype(int)
```

iii. Step 5 notes map `trial number` directly to the decoder input “trial number.”

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The detected start sample’s trial id `tid` is used as the per-trial scalar, and that value is broadcast to every neural time bin in the trial.

ii. ```python
for i, (tid, s, e) in enumerate(trials):
    ...
    trialnum_trial = np.full(nt.shape, tid, dtype=np.float32)
```

iii. The notes say trial number should be a per-trial scalar broadcast across bins. The agent did not add any cross-session offset in the final script.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the per-trial `reward_outcomes` array, which itself comes from `processing/behavior/BehavioralTimeSeries/Reward` event timestamps compared with trial boundaries.

ii. ```python
reward_event_timestamps = b.get('Reward__timestamps', np.asarray([]))
reward_outcomes = infer_reward_outcomes_from_events(trials, bt, reward_event_timestamps)
...
prev_out_trial = np.full(nt.shape, prev_out, dtype=np.float32)
```

iii. The trajectory shows a late correction: the agent first used a lick-based heuristic, then in Steps 66-67 switched to the `Reward` event stream after discovering it in NWB and noting that omission trials aligned with missing reward events.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The agent computes a binary reward outcome for each trial, initializes `prev_out = 0`, then for each kept trial broadcasts the previous kept trial’s outcome across the current trial’s time bins and updates `prev_out` afterward.

ii. ```python
prev_out = 0
for i, (tid, s, e) in enumerate(trials):
    ...
    prev_out_trial = np.full(nt.shape, prev_out, dtype=np.float32)
    ...
    prev_out = int(reward_outcomes[i])
```

iii. Step 5 notes say previous-trial outcome must be derived after reward outcome is inferred, then encoded `0/1` and broadcast within the current trial.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from behavioral `position` and `reward_zone`. The agent infers one reward-center position per trial from the positions where `reward_zone > 0`, then uses current position relative to that center.

ii. ```python
pos = np.asarray(b['position'])
rz_raw = np.asarray(b['reward_zone'])
trial_reward_pos, reward_loc_centers = infer_trial_reward_positions(pos, rz_raw, trials)
...
reward_center = float(trial_reward_pos.get(tid, np.nan))
dist_bins, _ = discretize_dist_to_reward(pos[bidx], reward_center)
```

iii. Step 5 notes say the variable should come from `position` and `reward_zone`, with reward-zone code mapped to spatial reward locations. Steps 63-64 in the trajectory show the specific switch from raw `reward_zone` codes to per-trial inferred positions.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each trial, the agent finds the median reward-zone position among samples with `reward_zone > 0`, treats that as a scalar reward center, computes `distance = position - reward_center` at behavior samples nearest the neural bins, and then bins that signed distance.

ii. ```python
def infer_trial_reward_positions(pos, rz_signal, trials):
    ...
    nz = rz_signal[sl] > 0
    trial_reward_pos[tid] = float(np.nanmedian(pos[sl][nz])) if np.any(nz) else np.nan

def discretize_dist_to_reward(pos, reward_center):
    dist = pos - reward_center
```

iii. The trajectory explicitly justifies this as a fix after concluding raw `reward_zone` values were not directly A/B/C labels. The notes describe it as “positions of nonzero reward-zone samples clustered into three canonical locations.”

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. It is discretized into the seven categories required by the task: `<-50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, `>50`, with `NaN` also mapped to category 0.

ii. ```python
out = np.full(dist.shape, 0, dtype=np.int64)
out[(dist >= -50) & (dist < -10)] = 1
out[(dist >= -10) & (dist < 0)] = 2
out[np.isclose(dist, 0)] = 3
out[(dist > 0) & (dist <= 10)] = 4
out[(dist > 10) & (dist <= 50)] = 5
out[dist > 50] = 6
out[np.isnan(dist)] = 0
```

iii. The agent copied these category boundaries directly from the decoder specification in the instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Neural timestamps `nt` are matched to behavior indices `bidx = searchsorted(bt, nt)`, and the position samples at those indices are used to compute distance categories with one output value per neural bin.

ii. ```python
bidx = np.searchsorted(bt, nt, side='left')
bidx = np.clip(bidx, 0, len(bt) - 1)
dist_bins, _ = discretize_dist_to_reward(pos[bidx], reward_center)
```

iii. The notes say behavior should be aligned to the time-binned neural representation required by the decoder; the agent implemented that by nearest-forward timestamp lookup.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from behavioral `position`.

ii. ```python
pos = np.asarray(b['position'])
abs_pos_bins, pos_edges = discretize_position(pos, valid_mask, n_bins=5)
```

iii. Step 5 maps `position` directly to the “absolute corridor position” output.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The agent computes session-specific minimum and maximum valid positions, splits that range into five equal-width bins with `linspace`, digitizes all positions, and later samples those digitized values at behavior indices aligned to neural bins.

ii. ```python
def discretize_position(pos, valid_mask, n_bins=5):
    valid_pos = pos[valid_mask]
    lo, hi = np.nanmin(valid_pos), np.nanmax(valid_pos)
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(pos, edges[1:-1], right=False), 0, n_bins - 1)
```

iii. The notes say absolute position should be discretized into five equal bins and that out-of-track periods should be excluded when defining the valid range.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is thresholded by equal-width binning between the session’s valid minimum and maximum position, producing categories `0..4`.

ii. ```python
edges = np.linspace(lo, hi, n_bins + 1)
idx = np.clip(np.digitize(pos, edges[1:-1], right=False), 0, n_bins - 1)
```

iii. The agent treated the task’s “5 equal-sized bins” as session-relative equal-width bins rather than fixed global corridor edges.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. After precomputing position bins on the behavior stream, the agent indexes them at `bidx`, the behavior samples aligned to each neural timestamp.

ii. ```python
bidx = np.searchsorted(bt, nt, side='left')
...
out = np.vstack([
    dist_bins,
    abs_pos_bins[bidx],
```

iii. This follows the same alignment strategy the agent used for all behavior-derived time-varying outputs.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from behavioral `lick`.

ii. ```python
lick = np.asarray(b['lick'])
```

iii. Step 5 maps `lick` directly to the binary lick output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The lick signal is thresholded to binary by checking `lick > 0.5` at the behavior samples aligned to each neural bin.

ii. ```python
(lick[bidx] > 0.5).astype(np.int64)
```

iii. The notes say the output should be a binary time series. The agent kept the processing minimal because the raw values were already lick counts / pulses.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is sampled from `lick[bidx]`, where `bidx` is the behavior index nearest each neural timestamp in the trial.

ii. ```python
bidx = np.searchsorted(bt, nt, side='left')
...
(lick[bidx] > 0.5).astype(np.int64)
```

iii. The agent’s Step 5 plan was to align behavior to the neural time bins rather than resample neural data to behavior bins.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from behavioral `reward_zone` together with behavioral `position`. The raw `reward_zone` code is not used directly as A/B/C.

ii. ```python
pos = np.asarray(b['position'])
rz_raw = np.asarray(b['reward_zone'])
trial_reward_pos, reward_loc_centers = infer_trial_reward_positions(pos, rz_raw, trials)
trial_reward_loc = assign_reward_location_labels(trial_reward_pos, reward_loc_centers)
```

iii. Steps 63-64 in the trajectory say the agent concluded raw `reward_zone` values were sparse time-varying codes rather than direct per-trial labels, so it switched to a position-based derivation.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The agent computes one median reward-zone position per trial from nonzero `reward_zone` samples, clusters all trial medians in a session into up to three centers with `KMeans`, labels each trial by nearest center as `0/1/2`, and broadcasts that label across the trial’s neural bins.

ii. ```python
km = KMeans(n_clusters=k, random_state=0, n_init=20).fit(vals.reshape(-1, 1))
centers = np.sort(km.cluster_centers_.ravel())
...
labels[tid] = int(np.argmin(np.abs(np.asarray(finite_centers) - rp)))
...
np.full(nt.shape, int(trial_reward_loc.get(tid, 0)), dtype=np.int64)
```

iii. The notes say this replaced an earlier incorrect direct mapping from raw `reward_zone` codes. The trajectory records that the change was motivated by observing clear positional clusters around switched reward locations.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/Reward` event timestamps.

ii. ```python
reward_event_timestamps = b.get('Reward__timestamps', np.asarray([]))
reward_outcomes = infer_reward_outcomes_from_events(trials, bt, reward_event_timestamps)
```

iii. Step 7 notes explicitly state that reward outcome inference was corrected to use NWB `BehavioralTimeSeries/Reward` event timestamps rather than lick heuristics.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the agent checks whether any reward event timestamp falls between the trial’s start and end behavior timestamps. The result is encoded `0/1` and broadcast across all neural bins in that trial.

ii. ```python
def infer_reward_outcomes_from_events(trials, behavior_timestamps, reward_event_timestamps):
    ...
    rewarded = int(np.any((reward_event_timestamps >= t0) & (reward_event_timestamps <= t1)))
...
np.full(nt.shape, int(reward_outcomes[i]), dtype=np.int64),
```

iii. In the trajectory, the agent called discovery of the `Reward` event stream a “key breakthrough” because 80 trials versus 74 reward events strongly suggested omission trials were exactly the no-event trials.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles several edge cases heuristically: if `position__timestamps` are missing it uses any available behavior timestamp stream; if neural timestamps are missing it synthesizes them with `linspace`; if no reward samples exist in a trial it stores `NaN` and later defaults reward-zone label or distance category; if reward timestamps are missing it uses an empty array; and it clips behavior indices after `searchsorted`.

ii. ```python
bt = b.get('position__timestamps', None)
if bt is None:
    any_ts = [v for k, v in b.items() if k.endswith('__timestamps')]
    bt = any_ts[0]
...
if neural_t is None:
    neural_t = np.linspace(bt[0], bt[-1], neural_nt.shape[1])
...
labels[tid] = 0
...
reward_event_timestamps = b.get('Reward__timestamps', np.asarray([]))
...
bidx = np.clip(bidx, 0, len(bt) - 1)
```

iii. The notes and trajectory show the agent repeatedly treated inconsistencies as something to patch around rather than fail on. It called the final script “provisional” after Step 6 because several such fallback paths remained.

## 13-a. What are the most time-consuming steps of the code?

i. The expensive parts are opening and reading all 152 NWB files, loading large neural matrices from HDF5, looping over every trial in every session, and running per-session `KMeans` on inferred reward positions. Writing the full pickle is also substantial because the output file is very large.

ii. ```python
files = sorted(Path('data').rglob('*.nwb'))
for fp in files:
    sess = convert_session(fp, show_processing=args.show_processing)
...
for i, (tid, s, e) in enumerate(trials):
    ...
km = KMeans(n_clusters=k, random_state=0, n_init=20).fit(vals.reshape(-1, 1))
```

iii. The runtime notes say full conversion was “manageable in minutes,” implying file I/O and the session/trial loops dominated rather than any heavy numerical model.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-bound reconstruction loop, per-trial reward-position loop, reward-outcome loop, and the main per-trial conversion loop could all be vectorized or batched more aggressively. The repeated `searchsorted` inside the trial loop is another obvious target.

ii. ```python
for i, s in enumerate(starts):
    ...
for tid, s, e in trials:
    ...
for _, s, e in trials:
    ...
for i, (tid, s, e) in enumerate(trials):
```

iii. The agent did not add explicit speedups beyond basic NumPy operations, and the notes leave the “Code speedups added” section effectively empty.

## 13-c. What processing does the code repeat multiple times?

i. It repeatedly opens NWB files for sample selection and full conversion, repeatedly scans behavior timestamps per trial, and separately recomputes trialwise reward statistics for reward location and reward outcome. It also re-runs the whole pipeline for sample and full datasets.

ii. ```python
if args.sample:
    for fp in files:
        with h5py.File(fp, 'r') as f:
            env = ...
...
trial_reward_pos, reward_loc_centers = infer_trial_reward_positions(...)
reward_outcomes = infer_reward_outcomes_from_events(...)
...
python3 -u convert_data.py sample_data.pkl --sample
python3 -u convert_data.py converted_data.pkl --full
```

iii. The trajectory shows several exploratory passes over the same NWB signals before the final script stabilized, especially for reward-zone and reward-outcome logic.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes or loads several values that are not used by the decoder dataset: `teleport` is loaded but not used in final trial boundaries, `valid_mask` only affects position bin edge estimation, ROI `regions` are loaded but discarded in favor of all-zero `brain_region_idx`, and `zone_centers` / `neural_source` are returned from `convert_session()` but not written into the saved dataset.

ii. ```python
teleport = np.asarray(b['teleport'])
valid_mask = (trial_num >= 0) & (pos > -400)
n_rois, regions = load_n_rois_and_regions(f)
...
return {
    ...
    'zone_centers': reward_loc_centers.tolist() if hasattr(reward_loc_centers, 'tolist') else reward_loc_centers,
    'neural_source': f'{neural_base}/{neural_key}',
}
```

iii. The notes discuss several of these quantities during exploratory validation, but the final exported `data` dict does not keep them, so that work is effectively discarded for downstream decoding.
