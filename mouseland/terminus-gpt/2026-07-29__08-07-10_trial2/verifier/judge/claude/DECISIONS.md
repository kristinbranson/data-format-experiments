# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all behavior data by globbing `Beh_*.npy` files from `data/beh/` and flattening all session keys into a single dictionary `beh_by_session`. Spike data is loaded per session from `data/spk/<session_id>_neural_data.npy`. Retinotopy data is loaded per session from `data/retinotopy/`. The AI does NOT use `Imaging_Exp_info.npy` as a master index; instead, it discovers sessions by iterating over all behavior file keys and matching them to spike files by base session name.

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
```

```python
def load_spk_session(session_base):
    fn = SPK_DIR / f'{session_base}_neural_data.npy'
    obj = np.load(fn, allow_pickle=True).item()
    spk = np.concatenate([x for x in obj['spks']], axis=0).astype(np.float32)
    return spk
```

iii. The AI's trajectory shows it discovered the behavior files through directory exploration in Step 2, noting 23 behavior `.npy` files. It chose to iterate over all behavior file keys rather than using `Imaging_Exp_info.npy` as an index. The CONVERSION_NOTES document the file structure discovery.

## 1-b. How are the data split into subjects?

i. Subjects are determined by parsing the first element of the session name (split by `_`). A dictionary `subj_to_idx` maps subject names to indices. Sessions are matched to subjects by their prefix.

ii.
```python
subj = sess.split('_')[0]
if subj not in subj_to_idx:
    subj_to_idx[subj] = len(subjects)
    subjects.append(subj)
```

iii. The AI noted 19 subjects observed in the data during Step 2 exploration. The subject parsing is straightforward since session IDs start with the mouse name.

## 1-c. How are the data split into sessions?

i. Sessions are identified by keys in the behavior dictionaries. The AI includes swap sessions (`_swap1`, `_swap2`) as separate sessions, and filters out sessions whose `TrialStim` contains the placeholder `stimulus_of_trial`. After filtering, 67 sessions remain (vs. 89 in the reference). Each behavior key that has a matching base spike file is treated as a session.

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

iii. The trajectory shows the AI initially selected 99 sessions, discovered the `stimulus_of_trial` placeholder issue, and filtered down to 67. The AI explicitly decided to keep swap sessions as separate entries.

## 1-d. How are the data split into trials?

i. Trials are identified using `ft_trInd` (frame-wise trial index). The AI finds unique trial indices from valid (non-NaN) frames, filters to trials within `[0, ntrials_declared)`, and requires at least 2 frames per trial. Unlike the reference, the AI does NOT filter frames to corridor space (`ft_CorrSpc`).

ii.
```python
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

iii. The trajectory notes that `ft_trInd` begins with NaNs before trial 0, so valid (non-NaN) frames are used for trial membership. The AI does not mention `ft_CorrSpc` filtering.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies minimal filtering: trials with fewer than 2 valid frames are skipped, and sessions with fewer than 2 valid trials are skipped. No other quality filtering is applied, consistent with the reference approach.

ii.
```python
if mask.sum() < 2:
    continue
```
```python
if len(nt) < 2:
    print(f'skipping {sess}: fewer than 2 valid trials')
    continue
```

iii. The CONVERSION_NOTES mention that trial curation rules need confirmation from code/consistency checks but no explicit trial exclusion criteria were documented beyond the minimum frame/trial count.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_base>_neural_data.npy`, concatenated across planes along the neuron axis. This matches the reference.

ii.
```python
def load_spk_session(session_base):
    fn = SPK_DIR / f'{session_base}_neural_data.npy'
    obj = np.load(fn, allow_pickle=True).item()
    spk = np.concatenate([x for x in obj['spks']], axis=0).astype(np.float32)
    return spk
```

iii. CONVERSION_NOTES Step 4 documents that `load_spk` concatenates all arrays in `spks` along axis 0, and the AI confirmed this produces a neuron x frame matrix.

## 2-b. How is the `neural` data processed?

i. Neural data frames for each trial are extracted using `ft_trInd` masking (without `ft_CorrSpc` filtering), then resampled to a fixed 60 time bins using index-based nearest-neighbor resampling, and stored as float16.

ii.
```python
nmat_raw = spk[:, mask].astype(np.float32)
```
```python
nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
```
```python
def resample_matrix_time(mat, n_bins=60):
    ...
    idx = np.round(np.linspace(0, t - 1, n_bins)).astype(int)
    return mat[:, idx].astype(np.float32, copy=False)
```

iii. The trajectory documents the decision to use 60-bin resampling, motivated by the reference code's 60-bin position-based representation and the need to reduce file size from an initial 334 GB output.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does NOT filter neurons by brain region. All neurons from the concatenated `spks` array are kept. Brain region labels are assigned for metadata purposes but are not used for filtering. This contrasts with the reference, which drops neurons outside V1, mHV, lHV, and aHV.

ii.
```python
spk = np.concatenate([x for x in obj['spks']], axis=0).astype(np.float32)
return spk  # no filtering applied
```

```python
def build_brain_region_idx(session_name, n_neurons):
    # assigns labels but does not filter
    ...
    return idx, uniq
```

iii. CONVERSION_NOTES Step 1 mentions `neu_area_ID` for area mapping, and Step 4 notes that region labels should be preserved, but no neuron filtering based on area is documented or implemented.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial data is aligned to trial start by extracting frames where `ft_trInd == trial` (valid, non-NaN frames). The first frame of each trial serves as the alignment point. All frames belonging to the trial (including grey space) are included, not just corridor-space frames. The data is then resampled to 60 bins.

ii.
```python
mask = valid.copy()
mask[valid] = tr_idx == tr
...
nmat_raw = spk[:, mask].astype(np.float32)
tvec = ft[mask]
```

iii. The trajectory notes that the alignment event is corridor entry / trial start, and the AI uses frame-wise trial indices for segmentation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI resamples all trials to a fixed 60 time bins using nearest-neighbor index-based resampling. The `time_bin_size` in metadata is set to `None`. The reference uses native frame rate (3.17 Hz, ~315 ms bins) with 32 frames per trial and no resampling.

ii.
```python
nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
```
```python
'time_bin_size': None,
```

iii. The trajectory documents the decision to use 60 bins as motivated by the reference code's 60-bin position-based representation (`get_interpPos_spk(..., n_bins=60)`). However, the reference's 60 bins are for position interpolation in downstream analysis, not for the raw frame-based representation used in conversion.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime` (MATLAB datenum of the sound cue per trial) and `ft` (frame timestamps), both converted from MATLAB datenums to seconds. The reference uses `SoundFr` (frame number of cue) instead.

ii.
```python
soundtime = matlab_days_to_seconds(np.asarray(beh['SoundTime'], dtype=float))
```
```python
cue_rel = soundtime[tr] - tvec
```

iii. The AI's variable mapping in CONVERSION_NOTES Step 5 lists `SoundTime` or `SoundFr` as potential sources, and the code uses `SoundTime`.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. `SoundTime` is converted from MATLAB datenum to seconds (multiply by 86400). Frame times `ft` are similarly converted. The input is computed as `SoundTime[trial] - tvec` (cue time minus current frame time), giving positive values before cue and negative after. This is then resampled to 60 bins.

ii.
```python
soundtime = matlab_days_to_seconds(np.asarray(beh['SoundTime'], dtype=float))
...
ft = matlab_days_to_seconds(np.asarray(beh['ft'], dtype=float)[:nfr])
...
cue_rel = soundtime[tr] - tvec
```

iii. The sign convention (positive before cue) matches the instruction's "time to sound cue" semantics. However, using absolute MATLAB datenums for `SoundTime` and session-start-relative frame times could introduce an offset issue since `ft` is not re-referenced to the same absolute time base as `SoundTime`.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It uses the same frame mask (`tvec = ft[mask]`) as the neural data, then both are resampled to 60 bins.

ii.
```python
tvec = ft[mask]
...
cue_rel = soundtime[tr] - tvec
...
inp = resample_matrix_time(inp_raw, n_bins=60).astype(np.float32)
```

iii. Alignment is through shared frame indices.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the session name's date components (year, month, day), converted to an integer `YYYYMMDD` format.

ii.
```python
def session_day_value(session_name):
    parts = session_name.split('_')
    y, m, d = map(int, parts[1:4])
    return y * 10000 + m * 100 + d
```

iii. The AI chose to encode the date as a large integer rather than counting training days per mouse. This differs significantly from the reference, which counts sequential recording sessions per mouse (0, 1, 2, ...).

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The date is parsed from the session name and encoded as `year*10000 + month*100 + day`, giving values like `20220712`. This is broadcast across all 60 bins of each trial. The reference instead counts the ordinal session number per mouse (0-indexed).

ii.
```python
day_val = float(session_day_value(base))
day_arr = np.full(mask.sum(), float(day_val), dtype=np.float32)
```

iii. No justification for this encoding choice is documented in the CONVERSION_NOTES or trajectory.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `ft` (frame timestamps converted from MATLAB datenums to seconds) for frames belonging to the trial.

ii.
```python
ft = matlab_days_to_seconds(np.asarray(beh['ft'], dtype=float)[:nfr])
...
tvec = ft[mask]
time_since_start = tvec - tvec[0]
```

iii. The AI derives trial start time as the first frame's timestamp within the trial, rather than using `StartFr` as in the reference.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Frame times are converted to seconds and the first frame's time is subtracted, giving time since the first frame of the trial. This is then resampled to 60 bins. The reference uses `StartFr` interpolated onto the frame time axis for sub-frame precision.

ii.
```python
time_since_start = tvec - tvec[0]
```

iii. Using `tvec[0]` gives time since the first imaging frame of the trial rather than the precise corridor entry time. This may introduce a small offset but is functionally similar.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same frame mask as neural data, then resampled to 60 bins.

ii.
```python
tvec = ft[mask]
time_since_start = tvec - tvec[0]
inp_raw = np.vstack([..., time_since_start.astype(np.float32), ...])
inp = resample_matrix_time(inp_raw, n_bins=60).astype(np.float32)
```

iii. Alignment through shared frame indices.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks whether each trial is in a rewarded corridor.

ii.
```python
isrew = np.asarray(beh['isRew']).astype(bool)
...
reward_available = np.full(mask.sum(), 1.0 if isrew[tr] else 0.0, dtype=np.float32)
```

iii. This matches the reference approach.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. `isRew` is cast to boolean and used to create a per-trial constant array (1.0 if rewarded, 0.0 if not), broadcast across all time bins. This matches the reference.

ii.
```python
reward_available = np.full(mask.sum(), 1.0 if isrew[tr] else 0.0, dtype=np.float32)
```

iii. No special processing needed, consistent with both solutions.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `TrialStim`, the stimulus identity assigned to each trial. The reference uses `WallName` instead, mapped to 4 base texture categories.

ii.
```python
trialstim = np.asarray(beh['TrialStim']).astype(str)
...
stim_idx = stim_to_idx[str(trialstim[tr])]
```

iii. The CONVERSION_NOTES mention `TrialStim` and `stim_id` as sources. The AI discovered that some sessions have `TrialStim` set to the placeholder `stimulus_of_trial` and excluded those sessions entirely.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI uses `TrialStim` values directly as category labels (e.g., `circle1`, `circle2`, `leaf1`, `leaf1_swap1`, etc.) without grouping them into base textures. The global mapping collects all unique `TrialStim` values across selected sessions and assigns integer indices. This results in many more categories (potentially 10+) than the reference's 4 categories (circle, leaf, rock, wood).

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

iii. The trajectory shows the AI's `stim_names` included entries like `circle1`, `circle2`, `leaf1`, `leaf1_swap1`, `leaf1_swap2`, etc. The reference groups these into 4 base categories using the `TEXTURE` mapping.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickTime` and `LickTrind` (lick timestamps and trial indices). The reference uses `LickFr` (lick frame numbers) instead.

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

iii. The AI uses `searchsorted` to find the nearest frame for each lick time, which is functionally similar to the reference's approach of using frame indices directly.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, lick times are filtered to those belonging to the trial (via `LickTrind`), then mapped to frame indices using `searchsorted` against frame timestamps. A binary array is created (0 = no lick, 1 = lick). The result is resampled to 60 bins. The output values are `['no_lick', 'lick']` (2 categories), whereas the reference has `['no lick', 'lick', 'none']` (3 categories, with 'none' for padding).

ii.
```python
lick = np.zeros(mask.sum(), dtype=np.uint8)
...
lick[inds] = 1
...
out_raw = np.vstack([stim_arr, lick, pos_bin.astype(np.uint8), speed_bin.astype(np.uint8)])
out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60) for i in range(out_raw.shape[0])]).astype(np.uint8)
```

iii. Since the AI uses 60-bin resampling instead of padding, there is no need for a 'none' padding category.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking uses the same frame mask as neural data, then both are resampled to 60 bins.

ii.
```python
tvec = ft[mask]
...
inds = np.searchsorted(tvec, lick_times, side='left')
```

iii. Alignment through shared frame indices.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position at each imaging frame.

ii.
```python
ft_pos = np.asarray(beh['ft_Pos'], dtype=float)[:nfr]
...
pos = ft_pos[mask]
```

iii. Same source variable as reference.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is discretized using `corridor_length` (read from the behavior data, defaulting to 40.0) divided into 4 equal bins. The reference uses a simpler `ft_Pos // 10` clipped to 0-3.

ii.
```python
def discretize_position(pos, corridor_length):
    pos = np.asarray(pos, dtype=float)
    bins = np.floor(np.clip(pos, 0, corridor_length - 1e-8) / (corridor_length / 4.0)).astype(int)
    return np.clip(bins, 0, 3)
```

iii. Both approaches should produce equivalent results when `corridor_length=40` (in decimeters), since `40/4 = 10`, matching `// 10`.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position is divided into 4 equal-length bins of `corridor_length / 4` each, clipped to [0, 3]. Output values are labeled `['bin0', 'bin1', 'bin2', 'bin3']`. The reference labels them `['0-1 m', '1-2 m', '2-3 m', '3-4 m', 'none']`.

ii.
```python
bins = np.floor(np.clip(pos, 0, corridor_length - 1e-8) / (corridor_length / 4.0)).astype(int)
return np.clip(bins, 0, 3)
```

iii. The thresholding logic is equivalent to the reference. The labels differ cosmetically.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Same frame mask as neural data, then resampled to 60 bins.

ii.
```python
pos = ft_pos[mask]
pos_bin = discretize_position(pos, corridor_length)
...
out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60) for i in range(out_raw.shape[0])]).astype(np.uint8)
```

iii. Alignment through shared frame indices.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
ft_speed = np.asarray(beh['ft_RunSpeed'], dtype=float)[:nfr]
...
speed = ft_speed[mask]
```

iii. Same source variable as reference.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speed is discretized using global quantile edges (25th, 50th, 75th percentiles) computed across ALL selected sessions' `ft_RunSpeed` values. The reference computes quartiles per session using rank-based ordering on the kept frames only.

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
```python
def discretize_speed(speed, edges):
    speed = np.asarray(speed, dtype=float)
    out = np.zeros(speed.shape, dtype=int)
    out[speed > edges[0]] = 1
    out[speed > edges[1]] = 2
    out[speed > edges[2]] = 3
    return out
```

iii. The trajectory mentions quartile binning. However, the AI computes global quantile edges rather than per-session rank-based quartiles. Also, the AI computes edges on ALL frames (not just kept trial frames), and uses strict inequality (`>`), which will not produce equal-sized bins when many values are at the edge boundaries (e.g., zero speed).

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Speed is thresholded using 3 global quantile edges into 4 bins (0-3). Output values are `['q1', 'q2', 'q3', 'q4']`. The reference uses per-session rank-based quartiles with labels `['lowest', 'low', 'high', 'highest', 'none']`.

ii.
```python
out[speed > edges[0]] = 1
out[speed > edges[1]] = 2
out[speed > edges[2]] = 3
```

iii. The instruction says "4 bins, each corresponding to 25% of the data." The AI's global edge approach may not achieve exactly 25% per bin within each session, especially with tied values.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Same frame mask as neural data, then resampled to 60 bins.

ii.
```python
speed = ft_speed[mask]
speed_bin = discretize_speed(speed, speed_edges)
```

iii. Alignment through shared frame indices.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles frame count mismatches by taking the minimum across behavioral frame arrays and neural frame count. NaN values in `ft_trInd` are treated as invalid frames. Trials with fewer than 2 frames are skipped. Sessions with fewer than 2 valid trials are skipped. The AI also handles missing retinotopy files by assigning 'unknown' labels. Sessions with placeholder `TrialStim` values are excluded entirely.

ii.
```python
frame_lengths = [len(np.asarray(beh[k])) for k in frame_keys if k in beh]
frame_lengths.append(spk.shape[1])
nfr = min(frame_lengths)
```
```python
if mask.sum() < 2:
    continue
```

iii. CONVERSION_NOTES Step 10 mentions that the behavior can run past imaging and streams are trimmed accordingly.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the spike files is the dominant cost, as these are large numpy files (the full dataset has ~89 large files). The AI's trajectory documents repeated issues with runtime, initially producing a 334 GB output that took very long to save/load, leading to optimization iterations.

ii.
```python
spk = np.concatenate([x for x in obj['spks']], axis=0).astype(np.float32)
```

iii. The trajectory extensively documents performance issues and optimization iterations.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `extract_session` processes each trial individually, building masks and extracting frames one at a time. The resampling functions operate per-trial as well. These could potentially be vectorized.

ii.
```python
for tr in uniq_trials:
    ...
    mask = valid.copy()
    mask[valid] = tr_idx == tr
    ...
```

iii. The trajectory documents the AI optimizing resampling from interpolation-based to index-based, but the per-trial loop structure remains.

## 12-c. What processing does the code repeat multiple times?

i. The retinotopy file is loaded twice per session: once in `extract_session` (indirectly via `load_spk_session` which doesn't load it) and once in `build_brain_region_idx`. The spike file is only loaded once since `build_brain_region_idx` loads the retinotopy separately. The behavior files are all loaded upfront into memory at the start.

ii.
```python
# In main loop:
nt, it, ot, info, n_neu = extract_session(...)
bri, br_names = build_brain_region_idx(sess, n_neu)
```

iii. Loading all behavior files at once avoids repeated I/O but uses more memory.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes global speed edges across all sessions upfront, requiring loading all behavior data into memory before any session processing begins. The AI also loads ALL neurons (no area filtering), meaning neurons outside the visual areas contribute to a much larger file size despite likely not being useful for the analysis described in the paper. The 60-bin resampling is also unnecessary processing since the reference uses native frame resolution.

ii.
```python
def compute_speed_edges(beh_by_session, selected_sessions):
    vals = []
    for sess in selected_sessions:
        x = np.asarray(beh_by_session[sess]['ft_RunSpeed'], dtype=float)
        ...
```

iii. The trajectory notes the file size issue (334 GB initially, 176 GB after optimization), partly caused by keeping all neurons without filtering.
