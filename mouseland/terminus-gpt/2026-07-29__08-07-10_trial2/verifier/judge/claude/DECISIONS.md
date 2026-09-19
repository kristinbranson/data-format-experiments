# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all behavior data by globbing `Beh_*.npy` files from `data/beh/` and merging all session keys into a single dictionary. Neural data is loaded per session from `data/spk/<session_base>_neural_data.npy`. Retinotopy data is loaded from `data/retinotopy/`. Session selection is determined by matching behavior session keys to available spike files via `base_session_name()`, which strips `_swap1`/`_swap2` suffixes. Sessions where `TrialStim` contains `'stimulus_of_trial'` are excluded, which filters out sessions where the stimulus label is masked. This approach does NOT use the `Imaging_Exp_info.npy` master index.

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

# Session selection:
for sess in sorted(beh_by_session.keys()):
    base = base_session_name(sess)
    if base not in spk_sessions:
        continue
    ts = np.asarray(beh_by_session[sess].get('TrialStim', [])).astype(str)
    if ts.size and 'stimulus_of_trial' in set(ts.tolist()):
        continue
    selected.append(sess)
```

iii. The AI chose to iterate over all behavior file keys rather than using the `Imaging_Exp_info.npy` master index. It filters sessions based on `TrialStim` content. The CONVERSION_NOTES mention matching behavior sessions to neural sessions by stripping swap suffixes.

## 1-b. How are the data split into subjects?

i. The subject (mouse) name is extracted from the first component of the session name by splitting on `_`. Subjects are accumulated as sessions are processed.

ii.
```python
subj = sess.split('_')[0]
if subj not in subj_to_idx:
    subj_to_idx[subj] = len(subjects)
    subjects.append(subj)
```

iii. The session naming convention encodes the mouse name as the first underscore-delimited field.

## 1-c. How are the data split into sessions?

i. Each behavior session key in the merged dictionary that has a matching spike file is treated as a separate session. Swap sessions (`_swap1`, `_swap2`) are included as separate sessions even though they share the same neural recording as their base session. This results in 67 sessions across 14 subjects.

ii.
```python
def base_session_name(session):
    if session.endswith('_swap1') or session.endswith('_swap2'):
        return session.rsplit('_', 1)[0]
    return session

# In extract_session:
base = base_session_name(session_name)
spk = load_spk_session(base)
```

iii. The AI recognized that swap sessions share neural data (loading from the base session's spike file), but treated each behavior variant as its own session entry.

## 1-d. How are the data split into trials?

i. Trials are split by finding unique values in `ft_trInd` (the frame-wise trial index). Only trials with indices >= 0 and < `ntrials` declared in the behavior data are kept. Trials with fewer than 2 frames are skipped. All frames assigned to a trial by `ft_trInd` are included, regardless of whether they are inside the corridor texture or the gray space.

ii.
```python
ft_trInd = np.asarray(beh['ft_trInd'], dtype=float)[:nfr]
valid = np.isfinite(ft_trInd)
tr_idx = ft_trInd[valid].astype(int)
uniq_trials = np.unique(tr_idx)

for tr in uniq_trials:
    if tr < 0 or tr >= ntrials_declared:
        continue
    mask = valid.copy()
    mask[valid] = tr_idx == tr
    if mask.sum() < 2:
        continue
```

iii. The AI uses the trial index directly from the behavior data without filtering to corridor space only.

## 1-e. How are trials filtered based on quality controls?

i. Only trials with fewer than 2 valid frames are dropped. There is no filtering based on trial length (no removal of extremely long trials where the mouse stopped).

ii.
```python
if mask.sum() < 2:
    continue
```

iii. No explicit justification for the lack of outlier trial filtering was provided. Sessions with fewer than 2 valid trials are skipped entirely.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_base>_neural_data.npy`, which is a list of neurons-by-frames arrays per imaging plane, concatenated along the neuron axis. The visual area comes from `iarea` in the retinotopy files but is only used for labeling, not filtering.

ii.
```python
def load_spk_session(session_base):
    fn = SPK_DIR / f'{session_base}_neural_data.npy'
    obj = np.load(fn, allow_pickle=True).item()
    spk = np.concatenate([x for x in obj['spks']], axis=0).astype(np.float32)
    return spk
```

iii. The AI notes these are deconvolved calcium traces from Suite2p processing.

## 2-b. How is the `neural` data processed?

i. The neural data is first cast to float32, then each trial's data is resampled to a fixed 60 time bins using linear index interpolation (`resample_matrix_time`), then cast to float16 for storage.

ii.
```python
def resample_matrix_time(mat, n_bins=60):
    ...
    idx = np.round(np.linspace(0, t - 1, n_bins)).astype(int)
    return mat[:, idx].astype(np.float32, copy=False)

nmat_raw = spk[:, mask].astype(np.float32)
nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
```

iii. The CONVERSION_NOTES mention that the 60-bin representation was adopted to match the reference code's 60-position-bin processing and to reduce file size from an initial 334 GB dataset.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron filtering is applied. All neurons from all imaging planes are kept, including those outside the four visual areas used in the paper. The verification output shows 2,868,960 neurons labeled "unknown" out of ~3.9M total.

ii.
```python
# No filtering in load_spk_session - all neurons kept
spk = np.concatenate([x for x in obj['spks']], axis=0).astype(np.float32)
```

iii. The AI's `build_brain_region_idx` function labels neurons by area but does not filter based on area assignment.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is extracted for frames belonging to each trial (as identified by `ft_trInd`), starting from the first frame of the trial. This includes frames from before corridor entry (gray space frames). All trials are then resampled to 60 bins.

ii.
```python
mask = valid.copy()
mask[valid] = tr_idx == tr
nmat_raw = spk[:, mask].astype(np.float32)
nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
```

iii. The alignment is implicitly to the first frame of each trial, which may include pre-corridor and gray-space frames rather than strictly corridor entry.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The metadata sets `time_bin_size: None`. All trials are resampled to exactly 60 time bins regardless of original trial length. The effective bin size varies per trial since different trials span different durations.

ii.
```python
'time_bin_size': None,
```
```python
nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
```

iii. The CONVERSION_NOTES justify the 60-bin approach as matching the reference code's 60-position-bin interpolation. However, the reference code's 60-bin representation is a position-based interpolation for specific analyses, not a temporal representation.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime`, the absolute MATLAB datenum of the sound cue for each trial, and `ft`, the frame timestamps.

ii.
```python
soundtime = matlab_days_to_seconds(np.asarray(beh['SoundTime'], dtype=float))
cue_rel = soundtime[tr] - tvec
```

iii. The AI used `SoundTime` (absolute timestamps) rather than `SoundFr` (frame indices).

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Both `SoundTime` and frame times `ft` are converted from MATLAB datenums (days) to seconds. The input is computed as `SoundTime[trial] - frame_time`, giving positive values before the cue and negative after, matching the "time to" semantics.

ii.
```python
def matlab_days_to_seconds(x):
    return np.asarray(x, dtype=float) * 24.0 * 3600.0

soundtime = matlab_days_to_seconds(np.asarray(beh['SoundTime'], dtype=float))
ft = matlab_days_to_seconds(np.asarray(beh['ft'], dtype=float)[:nfr])
cue_rel = soundtime[tr] - tvec
```

iii. The sign convention (positive before cue, negative after) matches the "time to sound cue" semantics.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The cue time is computed for the same frames selected for the trial's neural data, then resampled to 60 bins alongside the neural data.

ii.
```python
inp_raw = np.vstack([cue_rel.astype(np.float32), ...])
inp = resample_matrix_time(inp_raw, n_bins=60).astype(np.float32)
```

iii. Same resampling as neural data ensures alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the session name, specifically the date portion (year, month, day), converted to an integer of the form YYYYMMDD.

ii.
```python
def session_day_value(session_name):
    parts = session_name.split('_')
    y, m, d = map(int, parts[1:4])
    return y * 10000 + m * 100 + d

day_val = float(session_day_value(base))
```

iii. The AI derived the day value from the date encoded in the session name.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The date is encoded as a single integer YYYYMMDD (e.g., 20220712 for July 12, 2022). This value is broadcast across all time bins of every trial in the session. No ordinal counting or per-mouse normalization is applied.

ii.
```python
day_val = float(session_day_value(base))
day_arr = np.full(mask.sum(), float(day_val), dtype=np.float32)
```

iii. No explicit justification for using the raw date integer rather than an ordinal day count.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `ft`, the frame timestamps (MATLAB datenums), for the frames belonging to the trial.

ii.
```python
ft = matlab_days_to_seconds(np.asarray(beh['ft'], dtype=float)[:nfr])
tvec = ft[mask]
time_since_start = tvec - tvec[0]
```

iii. The trial start is taken as the timestamp of the first frame in the trial.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Frame timestamps are converted from MATLAB datenums to seconds. The input is computed as `frame_time - first_frame_time`, starting at 0 for the first frame of the trial.

ii.
```python
time_since_start = tvec - tvec[0]
```

iii. Simple time difference from the first frame.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed from the same frame mask as the neural data, then resampled to 60 bins.

ii.
```python
inp_raw = np.vstack([..., time_since_start.astype(np.float32), ...])
inp = resample_matrix_time(inp_raw, n_bins=60).astype(np.float32)
```

iii. Same resampling as neural data.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks whether each trial is in a rewarded corridor.

ii.
```python
isrew = np.asarray(beh['isRew']).astype(bool)
reward_available = np.full(mask.sum(), 1.0 if isrew[tr] else 0.0, dtype=np.float32)
```

iii. Direct use of the reward indicator variable.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew` is converted to a float (0.0 or 1.0) and broadcast across all time bins of the trial.

ii.
```python
reward_available = np.full(mask.sum(), 1.0 if isrew[tr] else 0.0, dtype=np.float32)
```

iii. No additional processing needed.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `TrialStim`, which names the stimulus of each trial.

ii.
```python
trialstim = np.asarray(beh['TrialStim']).astype(str)
stim_idx = stim_to_idx[str(trialstim[tr])]
```

iii. The AI uses `TrialStim` rather than `WallName`.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. A global mapping of unique `TrialStim` values to indices is built across all selected sessions (excluding sessions with masked `'stimulus_of_trial'` labels). This yields 7 categories: `['circle1', 'circle2', 'leaf1', 'leaf1_swap1', 'leaf1_swap2', 'leaf2', 'leaf3']`. The stimulus index is broadcast across all 60 bins of the trial.

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

iii. The AI kept individual texture sub-variants as separate categories rather than grouping them into the 4 broad texture categories (circle, leaf, rock, wood) as specified in the instructions.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickTrind` (trial index of each lick) and `LickTime` (absolute timestamp of each lick).

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

iii. The AI uses `LickTime`/`LickTrind` rather than `LickFr` (which directly gives frame indices).

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, lick events matching the trial index are found via `LickTrind`. Their absolute timestamps (`LickTime`, in MATLAB datenum seconds) are mapped to frame indices via `searchsorted` against the frame time vector. A binary array is created: 1 if a lick falls in that frame, 0 otherwise. The result is then resampled to 60 bins.

ii.
```python
lick_times = np.asarray(beh['LickTime'])[np.asarray(beh['LickTrind']) == tr]
inds = np.searchsorted(tvec, lick_times, side='left')
inds = inds[(inds >= 0) & (inds < len(tvec))]
lick[inds] = 1
```

iii. The searchsorted approach introduces potential temporal misalignment compared to using `LickFr` directly.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick events are aligned to the same frame time vector as the neural data, then resampled to 60 bins alongside it.

ii.
```python
out_raw = np.vstack([stim_arr, lick, pos_bin.astype(np.uint8), speed_bin.astype(np.uint8)])
out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60) for i in range(out_raw.shape[0])]).astype(np.uint8)
```

iii. Same 60-bin resampling as all other variables.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the frame-wise position in the corridor, and `Corridor_Length` (defaulting to 40.0).

ii.
```python
ft_pos = np.asarray(beh['ft_Pos'], dtype=float)[:nfr]
corridor_length = float(beh.get('Corridor_Length', 40.0))
```

iii. Direct use of the position variable from behavior data.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is clipped to `[0, corridor_length)` and divided into 4 equal-length bins of `corridor_length / 4` each.

ii.
```python
def discretize_position(pos, corridor_length):
    pos = np.asarray(pos, dtype=float)
    bins = np.floor(np.clip(pos, 0, corridor_length - 1e-8) / (corridor_length / 4.0)).astype(int)
    return np.clip(bins, 0, 3)
```

iii. With `corridor_length=40` (decimeters), each bin is 10 decimeters = 1 meter, matching the instruction's 4 equal 1-m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position is discretized using `floor(pos / (corridor_length/4))` into bins 0-3. With default corridor length of 40 (decimeters), this gives 4 bins of 10 decimeters each.

ii.
```python
bins = np.floor(np.clip(pos, 0, corridor_length - 1e-8) / (corridor_length / 4.0)).astype(int)
return np.clip(bins, 0, 3)
```

iii. Equivalent to the reference's `ft_Pos // 10` approach.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is taken from the same frame indices as the neural data, then resampled to 60 bins.

ii.
```python
pos = ft_pos[mask]
pos_bin = discretize_position(pos, corridor_length)
out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60) for i in range(out_raw.shape[0])]).astype(np.uint8)
```

iii. Same 60-bin resampling as all other data streams.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the frame-wise running speed.

ii.
```python
ft_speed = np.asarray(beh['ft_RunSpeed'], dtype=float)[:nfr]
```

iii. Direct use of running speed variable.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speed is discretized into 4 bins using global quantile edges computed across ALL selected sessions. The edges are the 25th, 50th, and 75th percentiles of all `ft_RunSpeed` values from all selected sessions.

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

def discretize_speed(speed, edges):
    speed = np.asarray(speed, dtype=float)
    out = np.zeros(speed.shape, dtype=int)
    out[speed > edges[0]] = 1
    out[speed > edges[1]] = 2
    out[speed > edges[2]] = 3
    return out
```

iii. The output shows speed edges `[0.0, 7.91, 27.79]`, and the resulting distribution is not balanced: `{q1: 0.151, q2: 0.157, q3: 0.313, q4: 0.380}`. This is because the global quantiles include all frames (including gray space/inter-trial), and the threshold-based discretization doesn't handle ties at edges correctly.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Speed values are compared against 3 global quantile edges using `>` comparisons. Values <= edge[0] get bin 0, values in (edge[0], edge[1]] get bin 1, etc.

ii.
```python
out[speed > edges[0]] = 1
out[speed > edges[1]] = 2
out[speed > edges[2]] = 3
```

iii. The distribution is highly uneven (15/16/31/38%) because the global quantiles are computed over all frames including non-trial frames, and the threshold approach doesn't handle the many frames at zero speed.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is taken from the same frame indices as neural data, then resampled to 60 bins.

ii.
```python
speed = ft_speed[mask]
speed_bin = discretize_speed(speed, speed_edges)
```

iii. Same 60-bin resampling as all other data streams.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI takes the minimum of all frame-level array lengths (including spike data) as the common frame count, truncating longer arrays. NaN values in `ft_trInd` are treated as invalid. Lick events outside the valid frame range are silently dropped. Sessions where `TrialStim` contains `'stimulus_of_trial'` are entirely excluded. Sessions with fewer than 2 valid trials are skipped.

ii.
```python
frame_lengths = [len(np.asarray(beh[k])) for k in frame_keys if k in beh]
frame_lengths.append(spk.shape[1])
nfr = min(frame_lengths)

valid = np.isfinite(ft_trInd)
```

iii. The AI's approach to taking the minimum frame count across multiple arrays is reasonable for handling length mismatches. However, including `BefCueFr` and `AftCueFr` in the min calculation is incorrect as these are per-trial arrays, not per-frame arrays, potentially truncating the data.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the neural data files (each is large), and the per-session resampling of neural matrices to 60 bins. The full conversion takes ~1800 seconds (67 sessions, averaging ~27 seconds each).

ii.
```python
spk = load_spk_session(base)  # loads and concatenates all planes
nmat = resample_matrix_time(nmat_raw, n_bins=60)  # resamples per trial
```

iii. The conversion log shows individual sessions taking 10-63 seconds, dominated by I/O.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop that extracts frames, computes inputs/outputs, and resamples could potentially be vectorized for the resampling step. The lick detection loop per trial (searching through `LickTrind` for each trial) could be vectorized with a groupby operation.

ii.
```python
for tr in uniq_trials:
    ...
    lick_times = np.asarray(beh['LickTime'])[np.asarray(beh['LickTrind']) == tr]
```

iii. N/A

## 12-c. What processing does the code repeat multiple times?

i. The `load_spk_session` function is called once per behavior session key, but for swap sessions (e.g., `TX108_2023_04_07_1_swap1` and `TX108_2023_04_07_1_swap2`), the same neural file is loaded multiple times since they share the same base session. Additionally, `load_retino_for_session` is called separately from `extract_session`, potentially loading the retinotopy file twice per session.

ii.
```python
# In extract_session:
spk = load_spk_session(base)
# In build_brain_region_idx:
ret = load_retino_for_session(base)
```

iii. No caching mechanism is implemented for neural data across swap sessions.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `get_cat_id` and `neu_area_ID` functions are defined but `get_cat_id` is never called. The `build_brain_region_idx` function computes area labels for all neurons but does not use them to filter neurons, resulting in 2.87M "unknown" neurons that add noise and storage overhead. The resampling to 60 bins discards temporal resolution information that could be useful.

ii.
```python
def get_cat_id(WallName, isRew):
    # defined but never called in main conversion flow
    ...

def neu_area_ID(iarea):
    # defined but never called
    ...
```

iii. These utility functions were copied from the reference code but not integrated into the conversion pipeline.
