# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads every `Beh_*.npy` into one dictionary, discovers spike-session names from `spk`, and keeps behavior keys whose base name has a spike file. Swap suffixes are stripped, placeholder `TrialStim` sessions are excluded, and each selected session is processed separately. This produced 67 sessions rather than the reference's 89 unique recordings.

ii.
```python
for bf in sorted(BEH_DIR.glob('Beh_*.npy')):
    obj = np.load(bf, allow_pickle=True).item()
    for k, v in obj.items():
        beh_by_session[k] = v
...
if base not in spk_sessions:
    continue
if ts.size and 'stimulus_of_trial' in set(ts.tolist()):
    continue
selected.append(sess)
```

iii. The notes justify selecting only sessions with paired neural and behavioral data and explicitly mapping swap behavior keys to unsuffixed neural recordings. They report 67 sessions, 27,279 trials, and 14 subjects after conversion.

## 1-b. How are the data split into subjects?

i. Subject IDs are parsed as the prefix before the first underscore in each selected session. Subjects are added in selected-session order and each retained session receives the corresponding index.

ii.
```python
subj = sess.split('_')[0]
if subj not in subj_to_idx:
    subj_to_idx[subj] = len(subjects)
    subjects.append(subj)
subject_idx.append(subj_to_idx[subj])
```

iii. The notes state that the subject prefix of the session ID is the mouse ID. No justification is given for the resulting loss of five of the 19 mice.

## 1-c. How are the data split into sessions?

i. Each surviving behavior dictionary key is treated as a session. `_swap1` and `_swap2` keys use the same unsuffixed spike recording, and there is no master-index deduplication.

ii.
```python
def base_session_name(session):
    if session.endswith('_swap1') or session.endswith('_swap2'):
        return session.rsplit('_', 1)[0]
...
for sess in sorted(beh_by_session.keys()):
```

iii. The agent believed swap behavior sessions should be mapped explicitly to a shared unsuffixed neural session. Its notes acknowledge 99 behavior sessions, 89 spike sessions, and only 76 literal overlaps.

## 1-d. How are the data split into trials?

i. Trials are the unique finite values of `ft_trInd`. For each value the code selects every common frame with that trial index, without requiring `ft_CorrSpc`, then index-resamples the entire selection to 60 points.

ii.
```python
valid = np.isfinite(ft_trInd)
tr_idx = ft_trInd[valid].astype(int)
uniq_trials = np.unique(tr_idx)
...
mask = valid.copy()
mask[valid] = tr_idx == tr
nmat = resample_matrix_time(nmat_raw, n_bins=60)
```

iii. The notes say trial boundaries should use trial start/corridor entry and initially contemplated frame-wise segments, but later chose a compact fixed 60-bin representation because the original full artifact was 334 GB.

## 1-e. How are trials filtered based on quality controls?

i. A trial is rejected only if its index is out of the declared range or it has fewer than two common frames. A session is rejected if fewer than two trials remain. There is no long-trial/outlier filter.

ii.
```python
if tr < 0 or tr >= ntrials_declared:
    continue
...
if mask.sum() < 2:
    continue
...
if len(nt) < 2:
    continue
```

iii. The agent gives no trial-quality rationale beyond format requirements. Its notes emphasize successful decoder verification, not stopped-animal or anomalously long-trial checks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity comes from every plane in the raw `spks` list in `<session>_neural_data.npy`, concatenated along the neuron axis.

ii.
```python
obj = np.load(fn, allow_pickle=True).item()
spk = np.concatenate([x for x in obj['spks']], axis=0).astype(np.float32)
```

iii. The notes correctly identify these as already deconvolved calcium activity and cite the reference loader's plane concatenation.

## 2-b. How is the `neural` data processed?

i. The concatenated traces are truncated to the shortest relevant frame stream, selected trial by trial, nearest-index resampled to exactly 60 columns, and stored as `float16`.

ii.
```python
nfr = min(frame_lengths)
spk = spk[:, :nfr]
...
idx = np.round(np.linspace(0, t - 1, n_bins)).astype(int)
return mat[:, idx].astype(np.float32, copy=False)
...
nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
```

iii. The agent says fixed 60-bin trials and reduced precision reduced the full pickle from 334 GB to 176 GB and were thought to match the reference's position-interpolation representation more closely.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not neuron-filtered. Retinotopy is used only to attach labels; unknown areas remain, and mismatched label counts are truncated or padded.

ii.
```python
labels = ['V1' if x == 0 else ... else 'unknown' for x in iarea]
...
if len(idx) != n_neurons:
    idx = idx[:m]
    ... idx = np.concatenate([idx, pad])
```

iii. The notes planned to preserve retinotopy labels and assumed Suite2p had already curated cells. They did not document the implemented decision to retain all area codes or the non-reference area mapping.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. A trial starts at its first frame carrying that `ft_trInd` value, not explicitly at corridor entry, and its full tagged interval is warped to 60 samples. Metadata nevertheless calls the alignment event trial start/corridor entry.

ii.
```python
mask[valid] = tr_idx == tr
nmat_raw = spk[:, mask].astype(np.float32)
nmat = resample_matrix_time(nmat_raw, n_bins=60)
...
'temporal_alignment_event': 'trial start / corridor entry'
```

iii. The notes intended trial-start/corridor-entry alignment, but the size-driven 60-bin revision did not implement an explicit `StartFr` or `ft_CorrSpc` boundary.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Every trial has 60 index-selected bins regardless of duration, so physical bin duration varies both within the rounded selection and across trials. The metadata records no bin size (`None`).

ii.
```python
idx = np.round(np.linspace(0, t - 1, n_bins)).astype(int)
...
'time_bin_size': None,
```

iii. The agent describes this as a compact 60-bin representation motivated by storage. It does not justify the loss of the original fixed 3.17 Hz/approximately 315 ms temporal resolution.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundTime` and frame timestamps `ft`, both treated as MATLAB-day values and converted to seconds.

ii.
```python
ft = matlab_days_to_seconds(np.asarray(beh['ft'], dtype=float)[:nfr])
soundtime = matlab_days_to_seconds(np.asarray(beh['SoundTime'], dtype=float))
```

iii. The notes planned to use `SoundTime` or `SoundFr` relative to trial time; no preference between them was justified.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The code subtracts each selected frame time from the trial's cue time, yielding positive values before and negative values after the cue, then index-resamples to 60 bins.

ii.
```python
cue_rel = soundtime[tr] - tvec
...
inp = resample_matrix_time(inp_raw, n_bins=60).astype(np.float32)
```

iii. The mapping notes specify a continuous, time-varying cue-time difference, consistent with the requested input.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It uses the same trial mask and the same 60-point resampling indices as neural activity because both arrays initially have the same number of columns.

ii.
```python
tvec = ft[mask]
nmat = resample_matrix_time(nmat_raw, n_bins=60)
inp = resample_matrix_time(inp_raw, n_bins=60)
```

iii. The agent's stated principle was that all decoder streams must use a common trial axis.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived directly from the year, month, and day embedded in the session filename, not from the within-mouse order of sessions.

ii.
```python
parts = session_name.split('_')
y, m, d = map(int, parts[1:4])
return y * 10000 + m * 100 + d
```

iii. The notes said session date/experiment ordering would be converted to a continuous per-trial value, but did not justify using a YYYYMMDD number as training day.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The YYYYMMDD integer is cast to float, repeated at every original frame of the trial, and resampled to 60 identical values.

ii.
```python
day_val = float(session_day_value(base))
day_arr = np.full(mask.sum(), float(day_val), dtype=np.float32)
```

iii. The agent only states that the input should be continuous and repeated over time; it does not address scale or within-subject training-day counting.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `ft` and the first frame selected by `ft_trInd`; the loaded `Trial_start_time` is not used.

ii.
```python
tvec = ft[mask]
time_since_start = tvec - tvec[0]
```

iii. The planning notes mention `Trial_start_time`, `Trial_end_time`, and frame indices, but the final code silently uses the first tagged frame.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The first selected time is subtracted from every selected time, then the resulting elapsed-time vector is nearest-index resampled to 60 points.

ii.
```python
time_since_start = tvec - tvec[0]
...
inp = resample_matrix_time(inp_raw, n_bins=60)
```

iii. The notes describe a continuous variable starting at zero and following the common trial axis.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It uses the same trial-frame mask and 60 resampling indices as the neural matrix.

ii.
```python
nmat_raw = spk[:, mask]
tvec = ft[mask]
...
nmat = resample_matrix_time(nmat_raw, n_bins=60)
inp = resample_matrix_time(inp_raw, n_bins=60)
```

iii. The agent justifies a shared axis for neural, input, and output arrays.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes from the per-trial boolean `isRew`.

ii.
```python
isrew = np.asarray(beh['isRew']).astype(bool)
```

iii. The notes explicitly distinguish rewarded-corridor identity from actual reward delivery and select `isRew` for that reason.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean is converted to 0.0 or 1.0 and repeated across the trial; resampling leaves it constant.

ii.
```python
reward_available = np.full(mask.sum(), 1.0 if isrew[tr] else 0.0,
                           dtype=np.float32)
```

iii. The agent considered this a per-trial binary context input and broadcast it to satisfy the time-by-variable format.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It comes from `TrialStim`, with a global category list built from the selected sessions.

ii.
```python
trialstim = np.asarray(beh['TrialStim']).astype(str)
...
stim_names = sorted(stim_names)
stim_idx = stim_to_idx[str(trialstim[tr])]
```

iii. The notes mention `TrialStim` and/or canonical stimulus identity, and aimed to preserve categorical stimuli including swap variants.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Unique `TrialStim` strings are sorted and assigned integer indices. The per-trial index is repeated across every bin. No mapping collapses crop/swap names to the four base textures.

ii.
```python
stim_to_idx = {s: i for i, s in enumerate(stim_names)}
...
stim_arr = np.full(mask.sum(), stim_idx, dtype=np.uint8)
```

iii. The agent describes visual stimulus as categorical, but gives no justification for departing from the broad four-texture categories used by the reference.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickTime` events restricted by `LickTrind`, rather than directly from `LickFr`.

ii.
```python
lick_times = np.asarray(beh['LickTime'])[np.asarray(beh['LickTrind']) == tr]
```

iii. The notes list `LickTime`, `LickFr`, and `LickTrind` as possible sources and choose binary time-varying lick occurrence.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each lick, `searchsorted` finds the first trial frame time at or after it and marks that bin 1. However, `ft` was converted from MATLAB days to seconds while `LickTime` was not, so the two arrays are in incompatible units.

ii.
```python
lick = np.zeros(mask.sum(), dtype=np.uint8)
inds = np.searchsorted(tvec, lick_times, side='left')
inds = inds[(inds >= 0) & (inds < len(tvec))]
lick[inds] = 1
```

iii. The agent says lick events should become a binary series, but does not discuss the necessary timestamp conversion or collision behavior.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The constructed trial-length lick vector is nearest-index resampled to 60 values, nominally matching neural indices. The timestamp unit error makes the event placement itself invalid.

ii.
```python
out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60)
                 for i in range(out_raw.shape[0])])
```

iii. The notes claim all streams share the frame-based trial axis and report high decoder accuracy, but do not validate individual lick timestamps.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It comes from frame-wise `ft_Pos` and the session's `Corridor_Length` (default 40).

ii.
```python
ft_pos = np.asarray(beh['ft_Pos'], dtype=float)[:nfr]
corridor_length = float(beh.get('Corridor_Length', 40.0))
```

iii. The notes identify `ft_Pos` as the requested frame-level position source.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Values are clipped to the corridor range, divided by one quarter of corridor length, floored, and cast to integer categories before index resampling.

ii.
```python
bins = np.floor(np.clip(pos, 0, corridor_length - 1e-8) /
                (corridor_length / 4.0)).astype(int)
return np.clip(bins, 0, 3)
```

iii. The agent intended four equal 1 m bins, assuming the source's 40 units represent a 4 m corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. With the usual 40-unit corridor, thresholds are 10, 20, and 30 source units, producing categories 0–3; out-of-range values are clipped.

ii.
```python
pos_bin = discretize_position(pos, corridor_length)
```

iii. This directly implements the requested four equal spatial bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is selected with the same `ft_trInd` mask and then sampled at the same 60 relative indices as neural data. Because the mask is not restricted to `ft_CorrSpc`, non-corridor positions can be clipped into endpoint categories.

ii.
```python
pos = ft_pos[mask]
...
resample_labels_1d(out_raw[i], n_bins=60)
```

iii. The agent's rationale is common frame-wise alignment followed by a fixed compact representation.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from frame-wise `ft_RunSpeed` values.

ii.
```python
ft_speed = np.asarray(beh['ft_RunSpeed'], dtype=float)[:nfr]
```

iii. The notes correctly identify this as the direct running-speed stream.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Three global quantiles are computed from all finite `ft_RunSpeed` values in every selected behavior session, including frames not retained in converted trials. Each selected trial's speed is thresholded and then resampled to 60 labels.

ii.
```python
allv = np.concatenate(vals)
edges = np.quantile(allv, [0.25, 0.5, 0.75])
...
speed_bin = discretize_speed(speed, speed_edges)
```

iii. The notes planned quartiles “over dataset” and thus selected global edges, without addressing large ties at zero or discarded frames.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Category starts at zero and increases for strict `>` comparisons against each global quartile edge. Tied values at an edge stay in the lower bin, so bins need not each contain 25% of samples.

ii.
```python
out[speed > edges[0]] = 1
out[speed > edges[1]] = 2
out[speed > edges[2]] = 3
```

iii. The agent interprets “each corresponding to 25% of the data” as value-quantile thresholds rather than rank-based equal-count bins.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed uses the same trial mask as neural data, is discretized before resampling, and is sampled at the same 60 relative indices.

ii.
```python
speed = ft_speed[mask]
...
out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60) ...])
```

iii. The agent states that frame-wise signals should be put on the common decoder trial axis.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Streams are truncated to their minimum available length; nonfinite trial indices, invalid trial IDs, and trials shorter than two frames are dropped. Missing retinotopy becomes `unknown`, and neuron/area length mismatches are truncated or padded. There is no explicit finite-value handling for most other streams, and behavior keys loaded later can overwrite earlier duplicate keys.

ii.
```python
nfr = min(frame_lengths)
valid = np.isfinite(ft_trInd)
...
if ret is None or 'iarea' not in ret:
    return np.zeros(n_neurons, dtype=np.int64), ['unknown']
```

iii. The notes discuss paired-session filtering and successful verification, but do not document most fallback behavior or test it against known data errors.

## 12-a. What are the most time-consuming steps of the code?

i. Loading and concatenating very large spike files, copying each trial's neuron-by-frame slice, resampling it, and serializing the resulting 176 GB nested pickle dominate. Global speed collection also concatenates every behavior speed value.

ii.
```python
spk = np.concatenate([x for x in obj['spks']], axis=0).astype(np.float32)
nmat_raw = spk[:, mask].astype(np.float32)
...
pickle.dump(data, f)
```

iii. The notes identify data size as the main problem: the first artifact was 334 GB and the 60-bin/float16 revision reduced it to 176 GB.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop repeatedly builds a full-session Boolean mask, slices all neurons, and resamples four label rows separately. Brain-region label construction and local/global remapping also use Python loops/list searches. Trial extraction could be grouped once by contiguous/indexed trial frames.

ii.
```python
for tr in uniq_trials:
    mask = valid.copy()
    mask[valid] = tr_idx == tr
...
out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60)
                 for i in range(out_raw.shape[0])])
```

iii. The notes leave the “loops that could be vectorized” field blank and do not discuss these costs.

## 12-c. What processing does the code repeat multiple times?

i. A base spike file is loaded and concatenated again for every selected behavior key, including swap keys sharing the same base. Each trial rescans the full session trial-index vector. The same resampling index is independently recomputed for neural, input, and each output row.

ii.
```python
for i, sess in enumerate(selected):
    ... extract_session(sess, ...)
...
idx = np.round(np.linspace(0, t - 1, n_bins)).astype(int)
```

iii. The agent does not acknowledge repeated processing in the notes; its development efficiency sections remain placeholders.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `beh_source` but never uses it, loads/copies every retinotopy array although only `iarea` is needed, computes unused `trialstart`, defines unused `neu_area_ID` and `get_cat_id`, imports `defaultdict`, and creates plots only when requested. It also casts full spike matrices to float32 even when already float32, before later casting trials to float16.

ii.
```python
beh_by_session, beh_source = load_all_behavior()
trialstart = matlab_days_to_seconds(np.asarray(beh['Trial_start_time'], dtype=float))
return {k: r[k] for k in r.files}
```

iii. The notes do not identify discarded work. The plotting option is intentional diagnostics, but the other values/functions have no downstream consumer.
