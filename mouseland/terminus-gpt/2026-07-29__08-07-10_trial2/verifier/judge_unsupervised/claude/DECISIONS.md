# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Behavioral data is loaded from multiple `.npy` files in `data/beh/` via `load_all_behavior()`, which iterates over `Beh_*.npy` files and merges all session-keyed dictionaries into one `beh_by_session` dict. Neural data is loaded per-session from `data/spk/<session>_neural_data.npy` via `load_spk_session()`. Retinotopy data is loaded per-session from `data/retinotopy/` via `load_retino_for_session()`.

ii.
```python
def load_all_behavior():
    beh_by_session = {}
    beh_source = {}
    for bf in sorted(BEH_DIR.glob('Beh_*.npy')):
        obj = np.load(bf, allow_pickle=True).item()
        for k, v in obj.items():
            beh_by_session[k] = v
            beh_source[k] = bf.name
    return beh_by_session, beh_source

def load_spk_session(session_base):
    fn = SPK_DIR / f'{session_base}_neural_data.npy'
    obj = np.load(fn, allow_pickle=True).item()
    spk = np.concatenate([x for x in obj['spks']], axis=0).astype(np.float32)
    return spk
```

iii. The agent documented in CONVERSION_NOTES.md Step 1 that `load_spk` in the reference code concatenates all arrays in `spks` along the neuron axis, and the agent replicates this logic. The behavioral loading matches the reference `load_exp_beh` pattern of loading `Beh_*.npy` files.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by parsing the first component of the session name (e.g., `DR10` from `DR10_2022_07_12_1`). A running `subj_to_idx` dict tracks unique subjects as they appear during session processing.

ii.
```python
subj = sess.split('_')[0]
if subj not in subj_to_idx:
    subj_to_idx[subj] = len(subjects)
    subjects.append(subj)
```

iii. The agent noted that 19 subjects exist in the raw data but only 14 appear in the converted output, because the others lack matching neural data files or are filtered out.

## 1-c. How are the data split into sessions?

i. Sessions are selected from the behavior data keys that have a matching neural data file (via `base_session_name` mapping `_swap1/_swap2` suffixes to the base session). Sessions where `TrialStim` contains the placeholder `'stimulus_of_trial'` are excluded. Each valid session becomes one entry in the output lists.

ii.
```python
selected = []
for sess in sorted(beh_by_session.keys()):
    base = base_session_name(sess)
    if base not in spk_sessions:
        continue
    ts = np.asarray(beh_by_session[sess].get('TrialStim', [])).astype(str)
    if ts.size and 'stimulus_of_trial' in set(ts.tolist()):
        continue
    selected.append(sess)
```

iii. The agent documented the `_swap1/_swap2` session handling in CONVERSION_NOTES Steps 4-5, noting that swap sessions reuse the same neural recording with different stimulus subsets.

## 1-d. How are the data split into trials?

i. Within each session, trials are identified from the frame-wise `ft_trInd` variable. Unique trial indices are extracted from valid (non-NaN) frames. Each unique trial index that falls within `[0, ntrials_declared)` and has at least 2 frames becomes a separate trial.

ii.
```python
ft_trInd = np.asarray(beh['ft_trInd'], dtype=float)[:nfr]
valid = np.isfinite(ft_trInd)
tr_idx = ft_trInd[valid].astype(int)
uniq_trials = np.unique(tr_idx)
ntrials_declared = int(np.asarray(beh['ntrials']).item())

for tr in uniq_trials:
    if tr < 0 or tr >= ntrials_declared:
        continue
    mask = valid.copy()
    mask[valid] = tr_idx == tr
    if mask.sum() < 2:
        continue
```

iii. The agent used the frame-wise trial index to segment data per trial, which matches the reference code's trial structure approach.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) trial index must be in range `[0, ntrials_declared)`, (2) trial must have at least 2 valid frames, (3) sessions must have at least 2 valid trials total to be included.

ii.
```python
if tr < 0 or tr >= ntrials_declared:
    continue
if mask.sum() < 2:
    continue
# ...
if len(nt) < 2:
    print(f'skipping {sess}: fewer than 2 valid trials')
    continue
```

iii. The agent did not implement any additional trial quality filtering beyond these basic checks. The reference code and paper do not describe explicit trial exclusion criteria beyond what is implied by valid data availability. No neuron-level quality filtering is applied either.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `spks` key within the per-session neural data `.npy` files in `data/spk/`. Each file contains a list of arrays (one per imaging plane), which are concatenated along the neuron axis.

ii.
```python
def load_spk_session(session_base):
    fn = SPK_DIR / f'{session_base}_neural_data.npy'
    obj = np.load(fn, allow_pickle=True).item()
    spk = np.concatenate([x for x in obj['spks']], axis=0).astype(np.float32)
    return spk
```

iii. The agent identified in CONVERSION_NOTES Step 1 that these are deconvolved calcium traces from Suite2p processing (not extracellular spikes despite the `spks` name).

## 2-b. How is the `neural` data processed?

i. The raw neural data (neurons x frames) is segmented by trial using frame-wise trial indices. For each trial, the relevant frames are extracted, then resampled to exactly 60 time bins using index-based nearest-neighbor interpolation (`resample_matrix_time`). Data is stored as float16 for compactness.

ii.
```python
nmat_raw = spk[:, mask].astype(np.float32)
nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)

def resample_matrix_time(mat, n_bins=60):
    mat = np.asarray(mat)
    if mat.ndim == 1:
        mat = mat[None, :]
    t = mat.shape[1]
    if t == n_bins:
        return mat.astype(np.float32, copy=False)
    if t == 1:
        return np.repeat(mat, n_bins, axis=1).astype(np.float32, copy=False)
    idx = np.round(np.linspace(0, t - 1, n_bins)).astype(int)
    return mat[:, idx].astype(np.float32, copy=False)
```

iii. The agent noted in CONVERSION_NOTES Step 5 that the reference code uses position-based interpolation (`spk_pos_interp` / `get_interpPos_spk`) to align neural activity by cumulative position with 60 bins. However, the agent's actual implementation uses time-based frame segmentation with resampling to 60 bins, not position-based interpolation. The agent's metadata notes this discrepancy: "reference code uses position-based interpolation of deconvolved traces."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality filtering is applied. All neurons from the `spks` arrays are included after concatenation. No filtering based on firing rate, signal quality, or other neuron-level metrics.

ii. The neural data is simply loaded and concatenated:
```python
spk = np.concatenate([x for x in obj['spks']], axis=0).astype(np.float32)
```

iii. The agent noted that Suite2p cell classification is the primary neuron inclusion criterion from the reference, and no additional explicit exclusion rules were found.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data is aligned to trial start by selecting frames whose `ft_trInd` equals the trial index. The first frame of each trial is the alignment point (trial start / corridor entry). The data is then resampled to 60 equal time bins spanning the trial duration.

ii.
```python
mask = valid.copy()
mask[valid] = tr_idx == tr
nmat_raw = spk[:, mask].astype(np.float32)
nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
```

iii. The instructions specify "Temporally aligned based on trial start (corridor entry)." The agent uses frame-wise trial indices to segment each trial, with bin 0 corresponding to the first frame of the trial (corridor entry). This differs from the reference code which uses position-based alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is set to `None` in the metadata. Each trial is resampled to exactly 60 bins, but these bins correspond to equally-spaced indices across the variable-length trial frames, not fixed-duration time bins. No fixed temporal resolution exists since trial durations vary.

ii.
```python
'time_bin_size': None,
```
And resampling:
```python
idx = np.round(np.linspace(0, t - 1, n_bins)).astype(int)
return mat[:, idx].astype(np.float32, copy=False)
```

iii. The agent acknowledged this is a variable-rate resampling. The reference code uses 60 position bins (not time bins), so there is no fixed time bin size in either case.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from two raw variables: `SoundTime` (per-trial sound cue time in MATLAB serial date format) and `ft` (frame-wise timestamps in MATLAB serial date format).

ii.
```python
ft = matlab_days_to_seconds(np.asarray(beh['ft'], dtype=float)[:nfr])
soundtime = matlab_days_to_seconds(np.asarray(beh['SoundTime'], dtype=float))
cue_rel = soundtime[tr] - tvec  # tvec = ft[mask]
```

iii. The agent identified SoundTime as the per-trial sound cue timestamp and ft as the frame-wise time array.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Both `SoundTime[tr]` and `ft` frame times are converted from MATLAB serial days to seconds (multiply by 86400). The difference `SoundTime[tr] - frame_time` is computed, giving positive values before the cue and negative values after.

ii.
```python
def matlab_days_to_seconds(x):
    return np.asarray(x, dtype=float) * 24.0 * 3600.0

cue_rel = soundtime[tr] - tvec
```

iii. The agent computed time to cue as a signed difference in seconds. This gives a time-varying signal that counts down to the cue.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The cue-relative time is computed at each frame within the trial, then resampled to 60 bins using the same `resample_matrix_time` function as the neural data.

ii.
```python
inp_raw = np.vstack([
    cue_rel.astype(np.float32),
    ...
])
inp = resample_matrix_time(inp_raw, n_bins=60).astype(np.float32)
```

iii. The same resampling scheme ensures temporal alignment between inputs and neural data within each trial.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from the session name, which encodes the date as `MOUSE_YYYY_MM_DD_BLK`. The year, month, and day are parsed from the session name.

ii.
```python
def session_day_value(session_name):
    parts = session_name.split('_')
    y, m, d = map(int, parts[1:4])
    return y * 10000 + m * 100 + d
```

iii. The agent chose to represent day of training as a YYYYMMDD integer (e.g., 20220712). This is a per-trial scalar repeated across all 60 time bins.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The date components from the session name are combined into a single integer `y * 10000 + m * 100 + d`. This is stored as a float and broadcast to all time bins for the trial.

ii.
```python
day_val = float(session_day_value(base))
day_arr = np.full(mask.sum(), float(day_val), dtype=np.float32)
```

iii. The agent chose YYYYMMDD encoding rather than an ordinal day count or days-since-first-session. This means the numerical values range from 20210320 to 20240116, which are large numbers but monotonically increasing with calendar date.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from `ft` (frame timestamps), specifically the difference between each frame time and the first frame time of the trial.

ii.
```python
tvec = ft[mask]
time_since_start = tvec - tvec[0]
```

iii. The agent uses the frame times within the trial to compute elapsed time from trial start.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Frame times are converted from MATLAB serial days to seconds, then the first frame time within the trial is subtracted to get time since trial start. Values start at 0 and increase.

ii.
```python
ft = matlab_days_to_seconds(np.asarray(beh['ft'], dtype=float)[:nfr])
tvec = ft[mask]
time_since_start = tvec - tvec[0]
```

iii. This is a straightforward computation giving seconds since the trial began.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Aligned via the same `resample_matrix_time` resampling to 60 bins, matching the neural data alignment.

ii.
```python
inp_raw = np.vstack([
    ...
    time_since_start.astype(np.float32),
    ...
])
inp = resample_matrix_time(inp_raw, n_bins=60).astype(np.float32)
```

iii. Same resampling as all other time-varying inputs and neural data.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from `isRew`, a per-trial boolean array in the behavior data indicating whether the trial is in a rewarded corridor.

ii.
```python
isrew = np.asarray(beh['isRew']).astype(bool)
reward_available = np.full(mask.sum(), 1.0 if isrew[tr] else 0.0, dtype=np.float32)
```

iii. The agent used `isRew` directly, which indicates the reward contingency of the corridor (not whether reward was actually delivered). This matches the instruction to report 1 if in rewarded corridor, 0 if not.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. A simple binary lookup: if `isRew[trial_index]` is True, the entire trial gets value 1.0; otherwise 0.0. This value is constant across all time bins within a trial.

ii.
```python
reward_available = np.full(mask.sum(), 1.0 if isrew[tr] else 0.0, dtype=np.float32)
```

iii. No complex processing. The agent noted in CONVERSION_NOTES Step 5 that reward availability should come from corridor identity, not reward delivery, because unsupervised sessions can include cue without reward delivery.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from `TrialStim`, a per-trial string array in the behavior data indicating the visual stimulus presented (e.g., 'circle1', 'leaf2').

ii.
```python
trialstim = np.asarray(beh['TrialStim']).astype(str)
stim_idx = stim_to_idx[str(trialstim[tr])]
stim_arr = np.full(mask.sum(), stim_idx, dtype=np.uint8)
```

iii. The agent built a global mapping from stimulus names to integer indices across all selected sessions via `build_global_mappings()`.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. A global sorted list of unique stimulus names is built across all sessions. Each trial's stimulus is mapped to its index in this list. The resulting category is constant across all 60 time bins. The categories are: `['circle1', 'circle2', 'leaf1', 'leaf1_swap1', 'leaf1_swap2', 'leaf2', 'leaf3']` (7 categories).

ii.
```python
def build_global_mappings(beh_by_session, selected_sessions):
    stim_names = set()
    for sess in selected_sessions:
        beh = beh_by_session[sess]
        if 'TrialStim' in beh:
            vals = [str(x) for x in np.unique(beh['TrialStim']) if str(x) != 'stimulus_of_trial']
            stim_names.update(vals)
    stim_names = sorted(stim_names)
    stim_to_idx = {s: i for i, s in enumerate(stim_names)}
    return stim_names, stim_to_idx
```

iii. The agent chose to use the raw `TrialStim` labels directly rather than the reference code's `get_cat_id` function which maps stimuli to 4 categories (0-3) based on reward contingency.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `LickTime` (timestamps of lick events) and `LickTrind` (trial index for each lick event), combined with `ft` (frame times) for temporal alignment.

ii.
```python
lick = np.zeros(mask.sum(), dtype=np.uint8)
if len(beh['LickTrind']) > 0:
    lick_times = np.asarray(beh['LickTime'])[np.asarray(beh['LickTrind']) == tr]
    if len(lick_times) > 0:
        inds = np.searchsorted(tvec, lick_times, side='left')
        inds = inds[(inds >= 0) & (inds < len(tvec))]
        lick[inds] = 1
```

iii. The agent used the raw lick event times and trial assignments to construct a binary lick raster.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, lick events are filtered to those matching the trial index. Lick times (in MATLAB serial date format, NOT converted to seconds here) are matched to frame times via `searchsorted`. Matched frames are set to 1 (licking), all others to 0 (not licking). The result is resampled to 60 bins.

ii.
```python
lick_times = np.asarray(beh['LickTime'])[np.asarray(beh['LickTrind']) == tr]
inds = np.searchsorted(tvec, lick_times, side='left')
inds = inds[(inds >= 0) & (inds < len(tvec))]
lick[inds] = 1
```

iii. The agent creates a per-frame binary lick indicator, which is then resampled. Note: `LickTime` values are NOT converted to seconds via `matlab_days_to_seconds`, but `tvec` IS converted to seconds. This is a potential bug if `LickTime` is in MATLAB serial days format - `searchsorted` would fail to match if units differ.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick binary array is resampled from frame resolution to 60 bins using `resample_labels_1d`, which uses nearest-neighbor index-based resampling.

ii.
```python
out_raw = np.vstack([stim_arr, lick, pos_bin.astype(np.uint8), speed_bin.astype(np.uint8)])
out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60) for i in range(out_raw.shape[0])]).astype(np.uint8)
```

iii. Same resampling scheme as all other trial-level data.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Derived from `ft_Pos`, the frame-wise VR position of the animal in the corridor.

ii.
```python
ft_pos = np.asarray(beh['ft_Pos'], dtype=float)[:nfr]
pos = ft_pos[mask]
pos_bin = discretize_position(pos, corridor_length)
```

iii. The agent used `ft_Pos` (frame-wise position, not cumulative position).

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The raw frame-wise position is clipped to `[0, corridor_length)` and divided into 4 equal bins using floor division.

ii.
```python
def discretize_position(pos, corridor_length):
    pos = np.asarray(pos, dtype=float)
    bins = np.floor(np.clip(pos, 0, corridor_length - 1e-8) / (corridor_length / 4.0)).astype(int)
    return np.clip(bins, 0, 3)
```

iii. Position is discretized into 4 equal segments of the corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The corridor length (default 40.0) is divided into 4 equal segments. Position values are floor-divided by `corridor_length/4` and clipped to [0, 3]. Bin 0 = [0, corridor_length/4), bin 1 = [corridor_length/4, corridor_length/2), etc.

ii.
```python
bins = np.floor(np.clip(pos, 0, corridor_length - 1e-8) / (corridor_length / 4.0)).astype(int)
return np.clip(bins, 0, 3)
```

iii. This matches the instruction for "4 equal-length, 1-m-long spatial bins" assuming corridor_length in VR units corresponds to a 4m physical corridor.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position bins are computed at frame resolution and then resampled to 60 bins using `resample_labels_1d`.

ii.
```python
pos_bin = discretize_position(pos, corridor_length)
out_raw = np.vstack([stim_arr, lick, pos_bin.astype(np.uint8), speed_bin.astype(np.uint8)])
out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60) for i in range(out_raw.shape[0])]).astype(np.uint8)
```

iii. Same resampling as neural data.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Derived from `ft_RunSpeed`, the frame-wise running speed of the animal.

ii.
```python
ft_speed = np.asarray(beh['ft_RunSpeed'], dtype=float)[:nfr]
speed = ft_speed[mask]
speed_bin = discretize_speed(speed, speed_edges)
```

iii. The agent used the frame-wise running speed variable.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Global speed quartile edges are computed across all selected sessions from raw `ft_RunSpeed` values (excluding non-finite values). These edges define the boundaries for 4 bins (q1-q4).

ii.
```python
def compute_speed_edges(beh_by_session, selected_sessions):
    vals = []
    for sess in selected_sessions:
        x = np.asarray(beh_by_session[sess]['ft_RunSpeed'], dtype=float)
        x = x[np.isfinite(x)]
        if x.size:
            vals.append(x)
    allv = np.concatenate(vals) if vals else np.array([0.0, 1.0])
    edges = np.quantile(allv, [0.25, 0.5, 0.75])
    return edges.astype(float)
```

iii. The agent computed quartile edges from the raw speed data across all sessions. The resulting edges were `[0.0, 7.91, 27.79]`.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Speed is discretized using the quartile edges: bin 0 if speed <= edge[0], bin 1 if edge[0] < speed <= edge[1], bin 2 if edge[1] < speed <= edge[2], bin 3 if speed > edge[2].

ii.
```python
def discretize_speed(speed, edges):
    speed = np.asarray(speed, dtype=float)
    out = np.zeros(speed.shape, dtype=int)
    out[speed > edges[0]] = 1
    out[speed > edges[1]] = 2
    out[speed > edges[2]] = 3
    return out
```

iii. The agent computed global quartile edges from the raw data and applied them uniformly. The resulting bin distribution in the converted data is uneven (q1=15.1%, q2=15.7%, q3=31.3%, q4=38.0%) because edges are computed on raw frame-level data but applied after resampling to 60 bins per trial.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed bins are computed at frame resolution and resampled to 60 bins using `resample_labels_1d`.

ii.
```python
speed_bin = discretize_speed(speed, speed_edges)
out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60) for i in range(out_raw.shape[0])]).astype(np.uint8)
```

iii. Same resampling as all other variables.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles several edge cases: (1) Non-finite values in `ft_trInd` are excluded via `np.isfinite` check. (2) Frame lengths are harmonized by taking the minimum across all frame-wise variables and neural data (`nfr = min(frame_lengths)`). (3) Trials with fewer than 2 frames are skipped. (4) Sessions with fewer than 2 valid trials are skipped. (5) Missing retinotopy data results in all neurons labeled as 'unknown'. (6) If retinotopy array length doesn't match neuron count, it is truncated or padded with 'unknown'.

ii.
```python
frame_lengths = [len(np.asarray(beh[k])) for k in frame_keys if k in beh]
frame_lengths.append(spk.shape[1])
nfr = min(frame_lengths)

# Missing retinotopy
if ret is None or 'iarea' not in ret:
    return np.zeros(n_neurons, dtype=np.int64), ['unknown']

# Length mismatch padding
if len(idx) != n_neurons:
    m = min(len(idx), n_neurons)
    idx = idx[:m]
    if m < n_neurons:
        pad = np.full(n_neurons - m, lab_to_idx.get('unknown', 0), dtype=np.int64)
        idx = np.concatenate([idx, pad])
```

iii. The agent documented in CONVERSION_NOTES Step 10 that the initial converted dataset was 334 GB and needed to be compacted. The agent used truncation to handle length mismatches between behavioral and neural arrays.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is `extract_session`, which takes 13-39 seconds per session (based on conversion_full_out.txt). This involves loading the large neural data file (up to ~90k neurons x ~30k frames), segmenting it by trial, and resampling. Total conversion time for 67 sessions was approximately 30 minutes.

ii.
```python
spk = load_spk_session(base)  # Load full neural matrix
# ... per-trial loop
nmat_raw = spk[:, mask].astype(np.float32)  # Extract trial frames
nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)  # Resample
```

iii. The agent noted the initial dataset was 334 GB before switching to the 60-bin representation, which reduced it to 176 GB.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The main trial-level loop in `extract_session` iterates over each trial sequentially, extracting frames and building arrays. The `resample_matrix_time` and `resample_labels_1d` are called per-trial. These could potentially be vectorized if trials were processed as a batch.

ii.
```python
for tr in uniq_trials:
    # ... per-trial frame extraction, resampling, input/output construction
    nmat_raw = spk[:, mask].astype(np.float32)
    nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
```

iii. The per-trial loop is the main bottleneck. Vectorization would require handling variable-length trials differently.

## 12-c. What processing does the code repeat multiple times?

i. The full neural data matrix (`spk`) is loaded once per session, but frame masking and extraction are repeated for each trial. The speed edge computation iterates over all sessions' speed data. The `build_global_mappings` function iterates over all sessions to collect stimulus names.

ii.
```python
# Speed edges computed over all sessions
def compute_speed_edges(beh_by_session, selected_sessions):
    vals = []
    for sess in selected_sessions:
        x = np.asarray(beh_by_session[sess]['ft_RunSpeed'], dtype=float)
        ...
```

iii. The stimulus mapping and speed edge computation are performed once before the main loop, which is efficient. No redundant recomputation was identified.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `show_processing` plots (position vs time, speed vs time, example neural/output visualizations) when the flag is set, but these are only visual aids and not used in the dataset. The `session_info` metadata contains timing and frame count details that are not used by the decoder. The code also stores neural data as float16 which loses precision but reduces file size.

ii.
```python
info = {
    'session_name': session_name,
    'base_session': base,
    'n_frames_common': int(nfr),
    'n_trials_kept': len(neural_trials),
    'elapsed_sec': time.time() - t0,
}
```

iii. The session_info metadata is stored but not used by the decoder. The processing plot generation is only invoked when `--show-processing` is passed. No major unnecessary processing was identified beyond these ancillary outputs.
