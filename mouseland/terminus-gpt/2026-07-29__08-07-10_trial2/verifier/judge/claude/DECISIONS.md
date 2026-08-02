# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads behavioral data by iterating over all `Beh_*.npy` files in `data/beh/`, calling `np.load(..., allow_pickle=True).item()` on each, and merging all session-keyed dictionaries into a single flat `beh_by_session` dict. Neural (spike) data is loaded per-session from `data/spk/<session>_neural_data.npy` by concatenating all arrays in the `spks` list along the neuron axis. Retinotopy data for brain region mapping is loaded per-session from `data/retinotopy/`.

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

iii. The AI documented in CONVERSION_NOTES.md that it identified `load_exp_beh` and `load_spk` as key reference functions for loading behavior and neural data respectively, and replicated their logic. The concatenation of `spks` entries matches the reference code's `load_spk`.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by parsing the first underscore-delimited part of each session name (e.g., "DR10" from "DR10_2022_07_12_1"). A unique subject list is built incrementally as sessions are processed.

ii.
```python
subj = sess.split('_')[0]
if subj not in subj_to_idx:
    subj_to_idx[subj] = len(subjects)
    subjects.append(subj)
```

iii. The AI noted 19 unique subjects in the raw data but the final converted dataset contains only 14 subjects. The reduction comes from session filtering (see 1-e).

## 1-c. How are the data split into sessions?

i. Sessions are defined by the keys in the behavioral data dictionaries (e.g., "DR10_2022_07_12_1"). Sessions with `_swap1` or `_swap2` suffixes are mapped to the unsuffixed base session for neural data loading (since the same neural recording was used with different stimulus subsets). Each behavior session key that has a matching neural data file is treated as a separate session in the output.

ii.
```python
def base_session_name(session):
    if session.endswith('_swap1') or session.endswith('_swap2'):
        return session.rsplit('_', 1)[0]
    return session

for sess in sorted(beh_by_session.keys()):
    base = base_session_name(sess)
    if base not in spk_sessions:
        continue
    # ... filtering ...
    selected.append(sess)
```

iii. The AI documented the need to handle `_swap1/_swap2` behavior sessions that reuse the same neural recording, mapping them to the base session name for spike data loading.

## 1-d. How are the data split into trials?

i. Trials are identified using the frame-wise trial index variable `ft_trInd` from the behavior data. The unique integer values in this array define individual trials. Each trial consists of the frames where `ft_trInd` equals that trial's index.

ii.
```python
ft_trInd = np.asarray(beh['ft_trInd'], dtype=float)[:nfr]
valid = np.isfinite(ft_trInd)
tr_idx = ft_trInd[valid].astype(int)
uniq_trials = np.unique(tr_idx)
# ...
for tr in uniq_trials:
    if tr < 0 or tr >= ntrials_declared:
        continue
    mask = valid.copy()
    mask[valid] = tr_idx == tr
```

iii. The AI used frame-wise trial indices to segment data into trials, which is consistent with the frame-level structure of the behavior data.

## 1-e. How are trials filtered based on quality controls?

i. Sessions are filtered if `TrialStim` contains the literal string `'stimulus_of_trial'` (likely a placeholder or header value). Sessions with fewer than 2 valid trials after processing are also skipped. Individual trials are skipped if their index is negative or exceeds `ntrials`, or if they have fewer than 2 frames. No other quality filtering (e.g., based on running, licking, or neural quality) is applied.

ii.
```python
ts = np.asarray(beh_by_session[sess].get('TrialStim', [])).astype(str)
if ts.size and 'stimulus_of_trial' in set(ts.tolist()):
    continue
# ...
if tr < 0 or tr >= ntrials_declared:
    continue
if mask.sum() < 2:
    continue
# ...
if len(nt) < 2:
    print(f'skipping {sess}: fewer than 2 valid trials')
    continue
```

iii. The AI noted the need to filter sessions without valid stimulus identities. There is no explicit neuron quality filtering or trial quality filtering beyond minimal frame count requirements.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the `spks` key in each session's neural data file (`<session>_neural_data.npy`). `spks` is a list of 2D arrays (neurons x frames), one per imaging plane.

ii.
```python
obj = np.load(fn, allow_pickle=True).item()
spk = np.concatenate([x for x in obj['spks']], axis=0).astype(np.float32)
```

iii. The AI documented that the reference code's `load_spk` function concatenates all entries in `spks` along the neuron axis, and replicated this approach. The data represents deconvolved fluorescence traces from Suite2p processing.

## 2-b. How is the `neural` data processed?

i. The neural data is: (1) concatenated across imaging planes, (2) truncated to the common frame count shared between neural and behavioral streams, (3) segmented by trial using `ft_trInd`, and (4) resampled to a fixed 60-bin representation using nearest-neighbor index-based interpolation. The data is then stored as float16.

ii.
```python
spk = spk[:, :nfr]
# ...
nmat_raw = spk[:, mask].astype(np.float32)
nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)

def resample_matrix_time(mat, n_bins=60):
    # ...
    idx = np.round(np.linspace(0, t - 1, n_bins)).astype(int)
    return mat[:, idx].astype(np.float32, copy=False)
```

iii. The AI noted in CONVERSION_NOTES.md that the reference code uses position-based interpolation (`get_interpPos_spk`, `spk_pos_interp`) to create a 60-bin representation. However, the AI's implementation uses a simpler time-based index resampling rather than position-based interpolation. This is a significant deviation from the reference code's approach.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality filtering is applied. All neurons from all imaging planes that pass Suite2p's cell classification (already done in preprocessing) are included.

ii. There is no filtering code - the concatenated `spks` array is used as-is.

iii. The AI noted that Suite2p cell classification is part of preprocessing. No additional inclusion/exclusion criteria are applied in the conversion script.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The instructions specify alignment to "trial start (corridor entry)." The AI aligns neural data to trial start by extracting frames where `ft_trInd == trial_index`, starting from the first frame of each trial. The data is then resampled to 60 fixed time bins spanning the trial duration.

ii.
```python
mask = valid.copy()
mask[valid] = tr_idx == tr
nmat_raw = spk[:, mask].astype(np.float32)
nmat = resample_matrix_time(nmat_raw, n_bins=60)
```

iii. The AI set `temporal_alignment_event` to `'trial start / corridor entry'` in metadata and `off_start` to 0.0. However, the reference code uses position-based interpolation (`spk_pos_interp`) to align neural data by accumulated corridor position rather than by time, creating a spatial representation where each bin corresponds to a fixed position in the corridor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data has a fixed 60-bin representation per trial. The AI sets `time_bin_size` to `None` in the metadata. The 60 bins are created by nearest-neighbor resampling of the variable-length frame-wise data. Since trials have different durations, the effective time per bin varies across trials.

ii.
```python
'time_bin_size': None,
# ...
def resample_matrix_time(mat, n_bins=60):
    idx = np.round(np.linspace(0, t - 1, n_bins)).astype(int)
    return mat[:, idx].astype(np.float32, copy=False)
```

iii. The AI noted that it uses the reference code's 60-bin convention, but the reference code uses 60 *position* bins (each corresponding to a fixed spatial interval in the corridor), not 60 time bins. The AI's time-based resampling produces a fundamentally different representation.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from `SoundTime` (per-trial sound cue time in MATLAB datenum format) and `ft` (frame-wise timestamps in MATLAB datenum format).

ii.
```python
soundtime = matlab_days_to_seconds(np.asarray(beh['SoundTime'], dtype=float))
ft = matlab_days_to_seconds(np.asarray(beh['ft'], dtype=float)[:nfr])
# ...
cue_rel = soundtime[tr] - tvec
```

iii. The AI maps `SoundTime` and frame timestamps to a relative time-to-cue variable.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Both `SoundTime` and `ft` are converted from MATLAB datenum (days) to seconds by multiplying by 86400. Then for each trial, `cue_rel = soundtime[tr] - tvec` computes the time remaining until the sound cue at each frame (positive before cue, negative after).

ii.
```python
def matlab_days_to_seconds(x):
    return np.asarray(x, dtype=float) * 24.0 * 3600.0

cue_rel = soundtime[tr] - tvec
```

iii. The resulting values are in seconds. The verification output shows ranges like `[-1767.7, 723.5]`, suggesting some trials have very long durations (up to ~30 minutes), which is unusual for virtual corridor trials and may indicate issues with trial segmentation.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The time-to-cue vector is computed at the same frame-wise resolution as the neural data and then resampled to 60 bins using the same `resample_matrix_time` function.

ii.
```python
inp_raw = np.vstack([
    cue_rel.astype(np.float32),
    # ...
])
inp = resample_matrix_time(inp_raw, n_bins=60).astype(np.float32)
```

iii. This ensures temporal alignment between inputs and neural data within each trial.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from the session name, which encodes the date as `<subject>_<year>_<month>_<day>_<block>`.

ii.
```python
def session_day_value(session_name):
    parts = session_name.split('_')
    y, m, d = map(int, parts[1:4])
    return y * 10000 + m * 100 + d
```

iii. The AI represents the day as a YYYYMMDD integer (e.g., 20220712), not as an ordinal "day of training" counting from the first session.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The session date is parsed from the session name and encoded as `year * 10000 + month * 100 + day`. This scalar value is broadcast to all 60 time bins as a constant per-trial input.

ii.
```python
day_val = float(session_day_value(base))
day_arr = np.full(mask.sum(), float(day_val), dtype=np.float32)
```

iii. This representation preserves date ordering but does not represent days as ordinal training days. The values are very large numbers (e.g., 20220712.0) that may not be ideal for a neural decoder but are monotonically related to actual training days.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The AI does NOT include "Environment type" as a separate decoder input. The concept of environment type (e.g., naive, unsupervised, task training phases) is not explicitly captured. The closest variable is `reward_availability`, which captures whether the corridor is rewarded.

ii. No relevant code - this input is not implemented.

iii. The decoder task instructions do not list "Environment type" as a required input, so its omission is consistent with the task specification.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Not applicable - Environment type is not included as a decoder input.

ii. No relevant code.

iii. See 4-a above.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from `ft` (frame timestamps in MATLAB datenum format), using the first frame of each trial as the reference.

ii.
```python
tvec = ft[mask]
time_since_start = tvec - tvec[0]
```

iii. This gives elapsed time in seconds since the first frame of the trial.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Frame timestamps are converted from MATLAB datenum to seconds, then the trial's first frame time is subtracted to produce a zero-based elapsed time vector.

ii.
```python
ft = matlab_days_to_seconds(np.asarray(beh['ft'], dtype=float)[:nfr])
tvec = ft[mask]
time_since_start = tvec - tvec[0]
```

iii. The values always start at 0 and increase monotonically within each trial. Maximum values per session reach 28-1769 seconds, again suggesting some trials may be very long.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same as other inputs: computed at frame resolution and resampled to 60 bins via `resample_matrix_time`.

ii.
```python
inp_raw = np.vstack([
    # ...
    time_since_start.astype(np.float32),
    # ...
])
inp = resample_matrix_time(inp_raw, n_bins=60).astype(np.float32)
```

iii. Alignment is consistent with neural data through shared resampling.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from `isRew`, a per-trial boolean array indicating whether the trial is in a rewarded corridor.

ii.
```python
isrew = np.asarray(beh['isRew']).astype(bool)
reward_available = np.full(mask.sum(), 1.0 if isrew[tr] else 0.0, dtype=np.float32)
```

iii. This directly uses the per-trial reward flag from the behavior data.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The `isRew` boolean is converted to a float (1.0 or 0.0) and broadcast to all frames/time bins in the trial as a constant per-trial value.

ii.
```python
reward_available = np.full(mask.sum(), 1.0 if isrew[tr] else 0.0, dtype=np.float32)
```

iii. This matches the decoder task specification: "1 if in rewarded corridor, 0 if not, discrete, per-trial."

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from `TrialStim`, a per-trial string array indicating the stimulus name (e.g., "circle1", "leaf1").

ii.
```python
trialstim = np.asarray(beh['TrialStim']).astype(str)
stim_idx = stim_to_idx[str(trialstim[tr])]
stim_arr = np.full(mask.sum(), stim_idx, dtype=np.uint8)
```

iii. A global mapping from stimulus names to integer indices is built across all selected sessions.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. All unique stimulus names across selected sessions are sorted alphabetically and assigned integer indices. Each trial is labeled with its stimulus index, broadcast as a constant across all time bins. The output_values list is `['circle1', 'circle2', 'leaf1', 'leaf1_swap1', 'leaf1_swap2', 'leaf2', 'leaf3']` (7 categories).

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

iii. The reference code uses `get_cat_id` which maps stimuli to a 4-category system based on reward status (rewarded stim = category 2, its pair = category 3, non-rewarded = category 0, its pair = category 1). The AI instead uses raw stimulus names directly, producing 7 categories instead of the reference's 4.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `LickTime` (timestamps of individual lick events) and `LickTrind` (trial index for each lick event).

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

iii. Individual lick events are mapped to their nearest frame, producing a binary time series.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick events are filtered to the current trial by matching `LickTrind`. Lick timestamps are converted to frame indices using `np.searchsorted` against the trial's time vector. Frames with lick events are set to 1, all others to 0. The result is resampled to 60 bins using `resample_labels_1d`.

ii.
```python
lick_times = np.asarray(beh['LickTime'])[np.asarray(beh['LickTrind']) == tr]
inds = np.searchsorted(tvec, lick_times, side='left')
inds = inds[(inds >= 0) & (inds < len(tvec))]
lick[inds] = 1
# Later:
out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60) for i in range(out_raw.shape[0])])
```

iii. The verification output shows licking is extremely sparse: 99.6% no_lick, 0.4% lick across the dataset. Many sessions have 0% licking. This extreme sparsity is concerning and may indicate issues with how lick events are mapped to frames, or that many sessions are from unsupervised/naive mice that don't lick.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick binary vector is constructed at frame resolution (same timebase as neural data) and resampled to 60 bins using nearest-neighbor index interpolation.

ii.
```python
out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60) for i in range(out_raw.shape[0])])
```

iii. Alignment is consistent with the neural data resampling approach.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Derived from `ft_Pos`, the frame-wise position in the virtual corridor.

ii.
```python
ft_pos = np.asarray(beh['ft_Pos'], dtype=float)[:nfr]
pos = ft_pos[mask]
pos_bin = discretize_position(pos, corridor_length)
```

iii. Uses frame-level position data aligned to the same frame indices as neural data.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Raw position values are discretized into 4 bins of equal spatial extent.

ii.
```python
def discretize_position(pos, corridor_length):
    pos = np.asarray(pos, dtype=float)
    bins = np.floor(np.clip(pos, 0, corridor_length - 1e-8) / (corridor_length / 4.0)).astype(int)
    return np.clip(bins, 0, 3)
```

iii. The corridor length defaults to 40.0 (from `beh.get('Corridor_Length', 40.0)`), so each bin spans 10 position units.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position is divided into 4 equal-width bins: [0, corridor_length/4), [corridor_length/4, corridor_length/2), [corridor_length/2, 3*corridor_length/4), [3*corridor_length/4, corridor_length). With corridor_length=40, bins are [0,10), [10,20), [20,30), [30,40).

ii.
```python
bins = np.floor(np.clip(pos, 0, corridor_length - 1e-8) / (corridor_length / 4.0)).astype(int)
return np.clip(bins, 0, 3)
```

iii. The instructions specify "4 equal-length, 1-m-long spatial bins" for a 4m corridor. If position is encoded in decimeters (0-40 for a 4m corridor), the AI's approach of dividing into 4 bins of 10 units each (= 1 meter each) is correct.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is extracted at frame resolution and discretized, then resampled to 60 bins using `resample_labels_1d` (nearest-neighbor index interpolation).

ii.
```python
pos_bin = discretize_position(pos, corridor_length)
out_raw = np.vstack([stim_arr, lick, pos_bin.astype(np.uint8), speed_bin.astype(np.uint8)])
out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60) for i in range(out_raw.shape[0])])
```

iii. Same resampling as neural data ensures alignment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Derived from `ft_RunSpeed`, the frame-wise running speed.

ii.
```python
ft_speed = np.asarray(beh['ft_RunSpeed'], dtype=float)[:nfr]
speed = ft_speed[mask]
speed_bin = discretize_speed(speed, speed_edges)
```

iii. Uses frame-level running speed data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is discretized into 4 bins using dataset-wide quartile edges computed from all selected sessions' `ft_RunSpeed` values.

ii.
```python
def compute_speed_edges(beh_by_session, selected_sessions):
    vals = []
    for sess in selected_sessions:
        x = np.asarray(beh_by_session[sess]['ft_RunSpeed'], dtype=float)
        x = x[np.isfinite(x)]
        if x.size:
            vals.append(x)
    allv = np.concatenate(vals)
    edges = np.quantile(allv, [0.25, 0.5, 0.75])
    return edges.astype(float)
```

iii. Speed edges are `[0.0, 7.91, 27.79]` (from conversion output), meaning bin boundaries are at 25th, 50th, and 75th percentiles of the pooled speed data.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four bins defined by quartile edges: bin 0 (<=Q25), bin 1 (Q25-Q50), bin 2 (Q50-Q75), bin 3 (>Q75).

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

iii. The instructions specify "4 bins, each corresponding to 25% of the data", which matches the quartile-based approach. However, the actual distribution in the output is imbalanced (15.1%, 15.7%, 31.3%, 38.0%), suggesting the quartile computation may include non-trial data or there's a mismatch between frame-level and trial-level statistics.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is extracted at frame resolution, discretized, and resampled to 60 bins using `resample_labels_1d`.

ii. Same resampling approach as position - see 9-d.

iii. Alignment is consistent with neural data through shared resampling.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: (1) NaN/non-finite values in `ft_trInd` are masked out using `np.isfinite`. (2) The common frame count across neural and behavioral streams is computed as the minimum of all array lengths, truncating to the shortest. (3) Non-finite speed values are excluded from quartile computation. (4) Brain region index arrays shorter than the neuron count are padded with a default region. (5) Empty `LickTrind` arrays are handled by checking length before processing.

ii.
```python
frame_lengths = [len(np.asarray(beh[k])) for k in frame_keys if k in beh]
frame_lengths.append(spk.shape[1])
nfr = min(frame_lengths)
# ...
valid = np.isfinite(ft_trInd)
# ...
if len(idx) != n_neurons:
    m = min(len(idx), n_neurons)
    idx = idx[:m]
    if m < n_neurons:
        pad = np.full(n_neurons - m, lab_to_idx.get('unknown', 0), dtype=np.int64)
        idx = np.concatenate([idx, pad])
```

iii. The AI's approach to truncating to the minimum frame count is reasonable for handling length mismatches. The padding of brain region indices with 'unknown' labels when the retinotopy array doesn't match the neuron count is a pragmatic solution. The verification output shows 2,868,960 neurons labeled "unknown", which is the largest brain region category, suggesting many sessions have incomplete retinotopy data.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading and processing each session's neural data. Each session takes 13-40 seconds (from conversion output), with the main bottleneck being loading large spike arrays and the trial-by-trial extraction loop. Total conversion time for 67 sessions is approximately 30 minutes.

ii.
```python
spk = load_spk_session(base)  # loads ~50-90K neurons x ~20-30K frames
# ...
for tr in uniq_trials:  # loop over 200-700 trials per session
    nmat_raw = spk[:, mask].astype(np.float32)
    nmat = resample_matrix_time(nmat_raw, n_bins=60)
```

iii. The AI noted timing information for each session but did not optimize the main processing loop. The per-trial numpy fancy indexing on large matrices is a significant cost.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The main trial-extraction loop iterates over each trial individually, applying a boolean mask to the full spike matrix for each trial. This could potentially be vectorized by pre-computing all trial masks and using batch operations. The `resample_labels_1d` function is called separately for each output dimension, which could be vectorized.

ii.
```python
for tr in uniq_trials:
    mask = valid.copy()
    mask[valid] = tr_idx == tr
    nmat_raw = spk[:, mask].astype(np.float32)
    # ... (repeated for each trial)
```

iii. The loop creates separate boolean masks and performs fancy indexing for each trial, which is inherently sequential.

## 12-c. What processing does the code repeat multiple times?

i. The `resample_matrix_time` function is called separately for neural data and inputs, and `resample_labels_1d` is called for each output dimension. The `valid.copy()` operation is repeated for each trial within a session. The full `spk` matrix is loaded even though only subsets are used per trial.

ii.
```python
nmat = resample_matrix_time(nmat_raw, n_bins=60)
inp = resample_matrix_time(inp_raw, n_bins=60)
out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60) for i in range(out_raw.shape[0])])
```

iii. While each call processes different data, the resampling index computation (`np.linspace(0, t-1, n_bins)`) is recalculated for every trial and every data stream separately, even though the indices could be pre-computed for trials of the same length.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `show_processing` plots for up to 2 sessions when the flag is enabled, which involves additional matplotlib rendering. The `beh_source` dictionary tracking which file each session came from is computed but not used in the final output. The `neu_area_ID` function (defined in the code) is never called - a simpler mapping is used instead. The global computation of speed quartile edges includes data from non-trial periods (frames between trials), which may not match the per-trial statistics the decoder uses.

ii.
```python
def neu_area_ID(iarea):  # defined but never called
    # ...

beh_source = {}  # computed but not used in output
for bf in sorted(BEH_DIR.glob('Beh_*.npy')):
    # ...
    beh_source[k] = bf.name
```

iii. The `neu_area_ID` function was copied from the reference code but replaced with inline brain region mapping in `build_brain_region_idx`. The `beh_source` dict was useful during development for debugging but is not included in the output.
