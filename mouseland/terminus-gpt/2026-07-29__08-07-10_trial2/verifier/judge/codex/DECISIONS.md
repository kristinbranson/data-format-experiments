# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script loads every `Beh_*.npy` file under `data/beh`, merges their per-session dict entries into one `beh_by_session` mapping, and loads neural data lazily per session from `data/spk/<base_session>_neural_data.npy`. Only behavior sessions whose base name has a matching spike file are processed.

ii.
```python
def load_all_behavior():
    beh_by_session = {}
    for bf in sorted(BEH_DIR.glob('Beh_*.npy')):
        obj = np.load(bf, allow_pickle=True).item()
        for k, v in obj.items():
            beh_by_session[k] = v
    return beh_by_session, beh_source

def load_spk_session(session_base):
    fn = SPK_DIR / f'{session_base}_neural_data.npy'
    obj = np.load(fn, allow_pickle=True).item()
    spk = np.concatenate([x for x in obj['spks']], axis=0).astype(np.float32)
    return spk
```

iii. In `CONVERSION_NOTES.md`, the agent said decoder data should use only sessions with paired behavior and neural data, and in trajectory step 78 it justified a frame-wise behavior/neural loading path because behavior frame arrays and neural frame counts almost matched.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the prefix before the first underscore in the session name, e.g. `DR10` from `DR10_2022_07_12_1`.

ii.
```python
subj = sess.split('_')[0]
if subj not in subj_to_idx:
    subj_to_idx[subj] = len(subjects)
    subjects.append(subj)
subject_idx.append(subj_to_idx[subj])
```

iii. The notes state that subject identity comes from the session naming convention and that 19 subjects were observed in the data exploration step.

## 1-c. How are the data split into sessions?

i. Each selected behavior session becomes one output session. `_swap1` and `_swap2` behavior sessions are treated as distinct sessions but mapped to the same unsuffixed neural recording file.

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
    selected.append(sess)
```

iii. `CONVERSION_NOTES.md` calls out `_swap1/_swap2` as a special case and says those behavior sessions likely reuse the same neural recording.

## 1-d. How are the data split into trials?

i. Trials are defined from frame-wise `ft_trInd`. The script trims all frame-aligned arrays to the shared minimum frame count, finds finite frame labels, then extracts one trial per unique integer trial index.

ii.
```python
frame_lengths.append(spk.shape[1])
nfr = min(frame_lengths)

spk = spk[:, :nfr]
ft_trInd = np.asarray(beh['ft_trInd'], dtype=float)[:nfr]
valid = np.isfinite(ft_trInd)
tr_idx = ft_trInd[valid].astype(int)
uniq_trials = np.unique(tr_idx)

for tr in uniq_trials:
    mask = valid.copy()
    mask[valid] = tr_idx == tr
```

iii. In trajectory steps 76 and 78, the agent said `ft_trInd` was the most reliable trial segmentation source because `StartFr`/`EndFr` were float-valued and there were NaN pre-trial frames.

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering is minimal. The script skips out-of-range trial indices, keeps only trials with at least 2 frames, skips behavior sessions whose `TrialStim` contains the placeholder `stimulus_of_trial`, and skips whole sessions with fewer than 2 valid trials.

ii.
```python
if ts.size and 'stimulus_of_trial' in set(ts.tolist()):
    continue

if tr < 0 or tr >= ntrials_declared:
    continue
if mask.sum() < 2:
    continue

if len(nt) < 2:
    print(f'skipping {sess}: fewer than 2 valid trials')
    continue
```

iii. The notes admit that exact trial exclusion criteria “still need confirmation from code/consistency checks”; no stronger QC rule was implemented later.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes only from the `spks` field in each per-session neural `.npy` file.

ii.
```python
obj = np.load(fn, allow_pickle=True).item()
spk = np.concatenate([x for x in obj['spks']], axis=0).astype(np.float32)
```

iii. The notes say this matches reference `load_spk`, and trajectory step 72 explicitly says the reference code concatenates the arrays in `spks`.

## 2-b. How is the `neural` data processed?

i. The script concatenates all `spks` arrays across axis 0, trims frames to the shortest common behavior/neural length, segments by `ft_trInd`, and then resamples each trial to exactly 60 bins by nearest-index selection. It does not use the reference position-interpolation functions.

ii.
```python
spk = np.concatenate([x for x in obj['spks']], axis=0).astype(np.float32)
spk = spk[:, :nfr]
nmat_raw = spk[:, mask].astype(np.float32)
nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
```

iii. The notes repeatedly mention that the reference code uses position-based interpolation with 60 bins, but in trajectory step 76 the agent decided to use the frame-wise trial timebase instead because it seemed easier to align to decoder inputs and outputs.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit neuron filtering is applied in `convert_data.py`. All rows of concatenated `spks` are kept.

ii.
```python
spk = np.concatenate([x for x in obj['spks']], axis=0).astype(np.float32)
...
return neural_trials, input_trials, output_trials, info, spk.shape[0]
```

iii. The notes rely on upstream Suite2p preprocessing and cell classification mentioned in the methods, but the conversion script itself performs no additional neuron QC.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Per-trial neural data are aligned to the first kept frame of each `ft_trInd` trial, which the script treats as trial start / corridor entry.

ii.
```python
tvec = ft[mask]
time_since_start = tvec - tvec[0]
...
'temporal_alignment_event': 'trial start / corridor entry',
'off_start': 0.0,
```

iii. The notes say decoder trials should be aligned to trial start / corridor entry, but the implementation uses the first valid frame in each `ft_trInd` segment rather than a direct `Trial_start_time` or `StartFr` anchor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Every trial is forced to 60 bins. The script performs nearest-index resampling, but it does not define a true time-bin size in milliseconds and stores `metadata['time_bin_size'] = None`.

ii.
```python
def resample_matrix_time(mat, n_bins=60):
    ...
    idx = np.round(np.linspace(0, t - 1, n_bins)).astype(int)
    return mat[:, idx].astype(np.float32, copy=False)

'time_bin_size': None,
```

iii. The notes frame 60 bins as being consistent with the reference code’s 60-bin representation, even though the reference functions shown in trajectory step 72 are position bins, not time bins.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundTime` and frame-wise `ft`.

ii.
```python
ft = matlab_days_to_seconds(np.asarray(beh['ft'], dtype=float)[:nfr])
soundtime = matlab_days_to_seconds(np.asarray(beh['SoundTime'], dtype=float))
cue_rel = soundtime[tr] - tvec
```

iii. In trajectory step 90, the agent explicitly fixed a unit bug by converting MATLAB datenums to seconds before computing cue-relative time.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Both `ft` and `SoundTime` are converted from MATLAB day units to seconds, trial frames are selected, and the current time is subtracted from the trial’s cue time.

ii.
```python
def matlab_days_to_seconds(x):
    return np.asarray(x, dtype=float) * 24.0 * 3600.0

cue_rel = soundtime[tr] - tvec
```

iii. The trajectory shows the first version mistakenly left these in day units; step 90 documents the correction and the reason.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on the same trial frame mask as the neural matrix and then resampled to the same 60 bins as the neural data.

ii.
```python
nmat_raw = spk[:, mask].astype(np.float32)
tvec = ft[mask]
cue_rel = soundtime[tr] - tvec
inp_raw = np.vstack([cue_rel.astype(np.float32), ...])
inp = resample_matrix_time(inp_raw, n_bins=60).astype(np.float32)
```

iii. The agent justified using the common frame-wise trial mask rather than the reference interpolation path in trajectory steps 76 and 78.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the session name string only.

ii.
```python
def session_day_value(session_name):
    parts = session_name.split('_')
    y, m, d = map(int, parts[1:4])
    return y * 10000 + m * 100 + d
```

iii. `CONVERSION_NOTES.md` planned to derive day from session date / experiment ordering. The final code only uses the date encoded in the session id.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The script parses year, month, and day from the session name, converts them to a numeric `YYYYMMDD`, casts to float, and repeats that scalar across all time bins in the trial.

ii.
```python
day_val = float(session_day_value(base))
day_arr = np.full(mask.sum(), float(day_val), dtype=np.float32)
```

iii. The trajectory notes mention relative day ordering as a pending issue, but the final implementation did not convert dates into an ordinal training-day index.

## 4-c. What variables in the raw data is `input` *Environment type* derived from?

i. It is not represented as a decoder input at all in the final dataset.

ii.
```python
'input_names': ['time_to_sound_cue', 'day_of_training',
                'time_since_trial_start', 'reward_availability'],
```

iii. The task instructions did not require environment type in the decoder inputs, so the agent omitted it.

## 4-d. What processing is involved in computing `input` *Environment type*?

i. None. No environment-type variable is computed or stored.

ii.
```python
'input_names': ['time_to_sound_cue', 'day_of_training',
                'time_since_trial_start', 'reward_availability'],
```

iii. There is no discussion in the notes of an implemented environment-type feature.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from frame-wise `ft` after trial segmentation.

ii.
```python
tvec = ft[mask]
time_since_start = tvec - tvec[0]
```

iii. The trajectory shows the agent chose the first frame of each `ft_trInd` trial as the effective start time.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The script converts `ft` to seconds, extracts trial frames, subtracts the first frame time from every frame in the trial, and resamples the result to 60 bins.

ii.
```python
ft = matlab_days_to_seconds(np.asarray(beh['ft'], dtype=float)[:nfr])
time_since_start = tvec - tvec[0]
inp_raw = np.vstack([... time_since_start.astype(np.float32), ...])
inp = resample_matrix_time(inp_raw, n_bins=60).astype(np.float32)
```

iii. Step 90 in the trajectory explains that this variable originally collapsed toward zero until the agent fixed the day-to-seconds conversion.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is built on the same per-trial frame mask as the neural slice and then resampled using the same 60-bin nearest-index procedure.

ii.
```python
nmat_raw = spk[:, mask].astype(np.float32)
tvec = ft[mask]
time_since_start = tvec - tvec[0]
nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
inp = resample_matrix_time(inp_raw, n_bins=60).astype(np.float32)
```

iii. The agent’s justification was internal alignment consistency with its frame-based trial representation.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived directly from per-trial `isRew`.

ii.
```python
isrew = np.asarray(beh['isRew']).astype(bool)
reward_available = np.full(mask.sum(), 1.0 if isrew[tr] else 0.0, dtype=np.float32)
```

iii. The notes explicitly say reward availability should come from corridor identity (`isRew`), not actual reward delivery.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean rewarded-corridor flag for the trial is cast to `0.0/1.0` and repeated across all frames/bins in that trial.

ii.
```python
reward_available = np.full(mask.sum(), 1.0 if isrew[tr] else 0.0, dtype=np.float32)
```

iii. The agent justified this in `CONVERSION_NOTES.md` because unsupervised sessions may still have cues without delivered reward.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `TrialStim`.

ii.
```python
trialstim = np.asarray(beh['TrialStim']).astype(str)
stim_idx = stim_to_idx[str(trialstim[tr])]
```

iii. The notes say the goal was to preserve categories like `circle1`, `leaf2`, and swap variants as categorical decoder outputs.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Unique `TrialStim` strings are collected globally across selected sessions, mapped to integer ids, then each trial gets a constant label repeated across all time bins.

ii.
```python
vals = [str(x) for x in np.unique(beh['TrialStim']) if str(x) != 'stimulus_of_trial']
stim_names = sorted(stim_names)
stim_to_idx = {s: i for i, s in enumerate(stim_names)}
...
stim_arr = np.full(mask.sum(), stim_idx, dtype=np.uint8)
```

iii. The agent chose this instead of using `get_cat_id`, because the decoder task asked for explicit stimulus categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickTime` and `LickTrind`.

ii.
```python
lick = np.zeros(mask.sum(), dtype=np.uint8)
if len(beh['LickTrind']) > 0:
    lick_times = np.asarray(beh['LickTime'])[np.asarray(beh['LickTrind']) == tr]
```

iii. The notes say licking should be represented as a binary time-varying series per trial.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The script selects lick timestamps for the current trial, uses `np.searchsorted` against the trial time vector, and marks those frame indices as `1`. However, it never converts `LickTime` from MATLAB days to seconds even though `tvec` was converted, so the alignment procedure is unit-inconsistent.

ii.
```python
tvec = ft[mask]
...
lick_times = np.asarray(beh['LickTime'])[np.asarray(beh['LickTrind']) == tr]
inds = np.searchsorted(tvec, lick_times, side='left')
lick[inds] = 1
```

iii. The trajectory never records a corresponding `LickTime` unit fix. Raw data inspection shows `LickTime`, `ft`, and `SoundTime` are all stored as MATLAB day values, so leaving `LickTime` unconverted is a real implementation bug.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Intended alignment is via the same trial frame vector `tvec` used for neural slicing, followed by 60-bin resampling. Because `LickTime` is left in the wrong units, this alignment is not trustworthy.

ii.
```python
tvec = ft[mask]
inds = np.searchsorted(tvec, lick_times, side='left')
...
out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60)
                 for i in range(out_raw.shape[0])]).astype(np.uint8)
```

iii. The agent justified a common frame-wise alignment path, but the code does not correctly implement it for lick timestamps.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from frame-wise `ft_Pos`.

ii.
```python
ft_pos = np.asarray(beh['ft_Pos'], dtype=float)[:nfr]
pos = ft_pos[mask]
```

iii. The notes map `ft_Pos` to the required position-bin output.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Trial-frame positions are discretized into 4 equal corridor segments using `Corridor_Length`, then resampled to 60 bins.

ii.
```python
corridor_length = float(beh.get('Corridor_Length', 40.0))
pos_bin = discretize_position(pos, corridor_length)
...
out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60) ...])
```

iii. The notes describe this as satisfying the decoder requirement to predict 4 spatial bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The code divides the full corridor length by 4, floors the normalized position, and clips the result to integers `0..3`.

ii.
```python
def discretize_position(pos, corridor_length):
    bins = np.floor(np.clip(pos, 0, corridor_length - 1e-8)
                    / (corridor_length / 4.0)).astype(int)
    return np.clip(bins, 0, 3)
```

iii. No more elaborate thresholding rule is discussed in the notes.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. It uses the same trial frame mask as the neural data and is resampled to the same 60-bin trial representation.

ii.
```python
nmat_raw = spk[:, mask].astype(np.float32)
pos = ft_pos[mask]
pos_bin = discretize_position(pos, corridor_length)
out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60) ...])
```

iii. This follows the agent’s frame-wise alignment choice rather than the reference position-interpolation functions.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from frame-wise `ft_RunSpeed`.

ii.
```python
ft_speed = np.asarray(beh['ft_RunSpeed'], dtype=float)[:nfr]
speed = ft_speed[mask]
```

iii. The notes map `ft_RunSpeed` directly to the required speed-bin output.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The script computes global quartile edges across all selected sessions from all finite `ft_RunSpeed` samples, then bins each trial’s frame-wise speed values against those edges and resamples to 60 bins.

ii.
```python
edges = np.quantile(allv, [0.25, 0.5, 0.75])
...
speed_bin = discretize_speed(speed, speed_edges)
```

iii. This was motivated by the decoder instruction that running speed should be discretized into 4 bins containing 25% of the data each.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Values `<= q25` are bin 0; values above successive quartiles are promoted to bins 1, 2, and 3.

ii.
```python
def discretize_speed(speed, edges):
    out = np.zeros(speed.shape, dtype=int)
    out[speed > edges[0]] = 1
    out[speed > edges[1]] = 2
    out[speed > edges[2]] = 3
    return out
```

iii. No alternative speed-threshold rule appears in the notes.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. It is extracted on the same trial frame mask as the neural slice and then resampled to the same 60 bins.

ii.
```python
speed = ft_speed[mask]
speed_bin = discretize_speed(speed, speed_edges)
out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60) ...])
```

iii. The justification is the same as for other time-varying variables: one common per-trial frame axis, then fixed-length resampling.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles mismatched frame counts by trimming to the common minimum, ignores non-finite `ft_trInd` frames, skips too-short trials, skips placeholder stimulus sessions, and assigns `unknown` brain regions when retinotopy is missing or the retinotopy vector length mismatches the neuron count.

ii.
```python
nfr = min(frame_lengths)
valid = np.isfinite(ft_trInd)
if mask.sum() < 2:
    continue
if ts.size and 'stimulus_of_trial' in set(ts.tolist()):
    continue
if ret is None or 'iarea' not in ret:
    return np.zeros(n_neurons, dtype=np.int64), ['unknown']
if len(idx) != n_neurons:
    ...
```

iii. The trajectory step 78 explicitly mentions the need to trim a small 3-frame mismatch between behavior and neural arrays.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive work is loading and concatenating huge `spks` matrices, then looping through all trials and repeatedly resampling large neuron-by-time trial matrices to 60 bins.

ii.
```python
spk = np.concatenate([x for x in obj['spks']], axis=0).astype(np.float32)
for tr in uniq_trials:
    ...
    nmat_raw = spk[:, mask].astype(np.float32)
    nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
```

iii. The trajectory repeatedly mentions very large session matrices and sample/full conversions that took long enough to motivate CPU/GPU memory discussion.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial extraction loop and the per-output-row label resampling loop could both be vectorized or batched more aggressively.

ii.
```python
for tr in uniq_trials:
    ...
out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60)
                 for i in range(out_raw.shape[0])]).astype(np.uint8)
```

iii. `CONVERSION_NOTES.md` leaves the “Code speedups added” section effectively blank, so the final script keeps the obvious Python-level loops.

## 12-c. What processing does the code repeat multiple times?

i. It repeatedly constructs boolean trial masks, slices the same frame-aligned arrays, and applies the same 60-bin resampling procedure separately to neural, input, and output arrays for every trial.

ii.
```python
mask = valid.copy()
mask[valid] = tr_idx == tr
...
nmat = resample_matrix_time(nmat_raw, n_bins=60).astype(np.float16)
inp = resample_matrix_time(inp_raw, n_bins=60).astype(np.float32)
out = np.vstack([resample_labels_1d(out_raw[i], n_bins=60) ...])
```

iii. The implementation favors simple repeated per-trial logic over reuse of precomputed trial index groups or a single joint resampling pass.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It imports plotting machinery and optionally generates processing figures, computes `trialstart` but never uses it, collects `kept_trial_ids` but never saves them, and defines helper functions like `neu_area_ID` and `get_cat_id` that are not used in the final conversion path.

ii.
```python
trialstart = matlab_days_to_seconds(np.asarray(beh['Trial_start_time'], dtype=float))
...
kept_trial_ids = []
...
kept_trial_ids.append(int(tr))
```

iii. The trajectory shows `trialstart` was part of the original plan, but the final script aligns to `tvec[0]` instead and leaves `trialstart` unused.
