# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads `beh/Imaging_Exp_info.npy` as a master index, deduplicates physical recordings by `mname_datexp_blk`, then for every session searches experiment types for its behavior dictionary, loads and concatenates all `spks` planes, and loads the matching retinotopy `iarea` array. Full mode processes all 89 unique sessions.

ii.
```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
all_sessions = get_all_unique_sessions(exp_info)
beh, exp_type, db_entry = get_session_beh(key, exp_info)
spk = load_spk(mname, datexp, blk)
iarea = load_retino(mname, datexp)
```

iii. The notes say this reproduces the paper's 89 recordings/19 mice and the reference loaders. The first matching experiment type is used because duplicate index entries describe the same physical recording and behavior.

## 1-b. How are the data split into subjects?

i. Subject identity is `mname`. Subjects are assigned indices in first-encounter order among successfully processed sessions, and each appended session gets the corresponding `subject_idx`.

ii.
```python
mname = result['mname']
if mname not in subjects_seen:
    subjects_seen[mname] = len(subjects_seen)
subject_idx_list.append(subjects_seen[mname])
subjects = list(subjects_seen.keys())
```

iii. The notes identify 19 unique mouse IDs and report that the converted full dataset contains all 19.

## 1-c. How are the data split into sessions?

i. A session is the unique combination of mouse, date, and block. Duplicate listings across experiment types are removed by a `seen` set; each retained session becomes one outer-list element.

ii.
```python
key = f"{s['mname']}_{s['datexp']}_{s['blk']}"
if key not in seen:
    seen.add(key)
    sessions.append({'key': key, ...})
```

iii. The agent observed 142 index entries but only 89 physical recordings and documented that duplicate experiment-type entries contain identical behavior.

## 1-d. How are the data split into trials?

i. The code loops over `range(beh['ntrials'])` and selects frame indices whose `ft_trInd` equals that zero-based trial, are in `ft_CorrSpc`, and have `ft_move > 0`. Each resulting frame set yields one variable-length trial.

ii.
```python
for trial_idx in range(ntrials):
    trial_mask = (ft_trInd == trial_idx) & ft_CorrSpc & (ft_move > RUNNING_THRESHOLD)
    frame_indices = np.where(trial_mask)[0]
```

iii. The agent interpreted the paper's “only considered timepoints during running” statement and reference analysis masks as requiring running-only, texture-corridor frames.

## 1-e. How are trials filtered based on quality controls?

i. A trial is dropped when fewer than two frames remain after corridor/running filtering, or when its wall name is unknown. A whole session is dropped if fewer than two trials remain. There is no long-trial/outlier filter.

ii.
```python
if len(frame_indices) < 2:
    continue
if stim_cat is None:
    continue
if len(neural_trials) < 2:
    return None
```

iii. The stated rationale is to retain only analyzable running data and satisfy the decoder's minimum of two trials per session. The notes state that no explicit general-analysis trial exclusion was found in the source code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the per-plane `spks` arrays in each session's neural file. Retinotopy `iarea` supplies the neuron inclusion mask and region labels.

ii.
```python
dat = np.load(fn, allow_pickle=True).item()
spk = np.concatenate(dat['spks'], axis=0)
iarea = load_retino(mname, datexp)
```

iii. The agent notes that `spks` are already Suite2p-deconvolved fluorescence traces used by the paper, so raw fluorescence or dF/F is unnecessary.

## 2-b. How is the `neural` data processed?

i. Imaging planes are concatenated along neurons, retained neurons are selected, and columns at each trial's selected native frames are copied to float32. There is no normalization, interpolation, padding, or deconvolution.

ii.
```python
spk = np.concatenate(dat['spks'], axis=0)
spk = spk[neuron_mask]
neural = spk[:, frame_indices].astype(np.float32)
```

iii. The notes justify direct use because the supplied signal is already deconvolved and behavior is already mapped to imaging frames.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons with `iarea == -1` or `iarea == 7` are excluded; all other known visual-area codes are retained. If spike and retinotopy neuron counts disagree, both are silently truncated to the smaller count.

ii.
```python
EXCLUDED_IAREA = {-1, 7}
for exc in EXCLUDED_IAREA:
    mask &= (iarea != exc)
if len(iarea) != n_neurons_raw:
    min_n = min(n_neurons_raw, len(iarea))
    spk = spk[:min_n]
    iarea = iarea[:min_n]
```

iii. The agent says this follows the reference visual-cortex exclusion and that the authors already performed Suite2p cell curation, so no firing-rate/SNR filter is added.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials contain selected corridor/running frames in their original order. The agent labels the first retained running frame as trial start and stores variable-length arrays; it does not use `StartFr` to establish corridor-entry time.

ii.
```python
frame_indices = np.where(trial_mask)[0]
neural = spk[:, frame_indices].astype(np.float32)
```

iii. Metadata calls the event “Trial start (corridor entry, first running frame in texture corridor).” The notes treat running-only filtering as reference-consistent and accept variable trial length.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Each native imaging frame is one time bin; metadata records 315 ms (3.178 Hz). No temporal rebinning or resampling is applied, although removed non-running frames can create gaps between adjacent stored columns.

ii.
```python
'time_bin_size': 315.0,
'frame_rate_hz': 3.178,
```

iii. The agent says native imaging frames match the reference resolution and behavior is already on the same frame grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundFr` and session frame timestamps `ft`.

ii.
```python
SoundFr = beh['SoundFr']
ft = beh['ft'][:nfr]
```

iii. The agent chose actual timestamps rather than assuming a fixed frame interval.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. `SoundFr` is linearly interpolated on the frame timestamp axis, converted from days to seconds, and each retained frame time is subtracted. Values are positive before the cue and negative after it.

ii.
```python
frame_times = ft[frame_indices] * 86400
sound_time_s = np.interp(sound_fr, np.arange(nfr), ft[:nfr]) * 86400
time_to_sound = sound_time_s - frame_times
```

iii. The notes explicitly verify the sign convention and say a spot-check against source timestamps matched exactly.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at exactly the same `frame_indices` used for neural columns and placed in the first row of the trial input matrix.

ii.
```python
frame_times = ft[frame_indices] * 86400
neural = spk[:, frame_indices].astype(np.float32)
```

iii. The agent states that all streams are aligned by imaging-frame number.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from `mname` and the calendar date string `datexp` in the session index.

ii.
```python
date = datetime.strptime(s['datexp'], '%Y_%m_%d')
mouse_dates[mname].append((date, s['key']))
```

iii. The notes describe the feature as days since each mouse's first recorded session.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, dates are sorted and the integer calendar-day difference from its earliest date is calculated, then broadcast over every retained timepoint of each trial.

ii.
```python
first_date = dates[0][0]
day_map[key] = (date - first_date).days
np.full((1, n_tp), day_of_training, dtype=np.float32)
```

iii. The agent considered elapsed days a natural interpretation of “day of training” and reports independent date spot-checks.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `ft` at the retained `frame_indices`; it does not use raw `StartFr`.

ii.
```python
frame_times = ft[frame_indices] * 86400
time_since_start = frame_times - frame_times[0]
```

iii. The plan describes time from the first retained trial frame, which it equates with corridor entry.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. MATLAB-day timestamps are converted to seconds and the timestamp of the first retained running frame is subtracted, so every stored trial begins at exactly zero.

ii.
```python
frame_times = ft[frame_indices] * 86400
time_since_start = frame_times - frame_times[0]
```

iii. The agent prefers actual timestamps and identifies the first valid frame as the alignment origin.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. One value is computed for each selected neural frame using the same indices, including any real-time gaps caused by dropping stopped frames.

ii.
```python
neural = spk[:, frame_indices].astype(np.float32)
time_since_start = frame_times - frame_times[0]
```

iii. The notes say all data streams use the same frame-number alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is taken directly from per-trial `isRew`.

ii.
```python
isRew = beh['isRew']
```

iii. The notes identify `isRew` as the rewarded-corridor flag.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The trial's boolean/numeric flag is cast to float and broadcast over all retained trial frames.

ii.
```python
np.full((1, n_tp), float(isRew[trial_idx]), dtype=np.float32)
```

iii. The agent treats it as a binary per-trial variable; broadcasting makes the input matrix rectangular.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from each trial's `WallName`.

ii.
```python
WallName = beh['WallName']
wall_name = str(WallName[trial_idx])
```

iii. The agent notes that actual data names “wood,” despite prose mentioning “brick,” and uses the data labels.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Fifteen wall variants/crops/swaps are hard-mapped to four broad categories (`circle`, `leaf`, `rock`, `wood`), converted to indices 0–3, and broadcast across the trial. Unknown names cause trial removal.

ii.
```python
stim_cat = STIM_CATEGORY_MAP.get(wall_name, None)
stim_cat_idx = STIM_CATEGORIES.index(stim_cat)
np.full((1, n_tp), stim_cat_idx, dtype=np.int64)
```

iii. The notes interpret “e.g. circle, leaf” as requesting broad texture categories rather than individual exemplars.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr` and `LickTrind`.

ii.
```python
LickFr = beh['LickFr']
LickTrind = beh['LickTrind']
trial_lick_frs = LickFr[LickTrind == trial_idx]
```

iii. The notes identify these as lick frame numbers and trial assignments.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A zero vector over retained frames is created. Every lick assigned to the trial is mapped to the nearest retained frame using `searchsorted` plus a Python loop, and that bin is set to one; multiple licks collapse to one.

ii.
```python
lick_bin_idx = np.searchsorted(fi_sorted, trial_lick_frs)
for li in range(len(trial_lick_frs)):
    ...
    lick_binary[idx] = 1.0
```

iii. The agent says the desired output is binary and time-varying, and describes assigning each lick to its nearest frame.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks are snapped to the nearest member of the filtered `frame_indices`, so the result has one value per stored neural column. A lick on a removed stopped frame may therefore move to another retained timepoint, including a trial endpoint.

ii.
```python
fi_sorted = frame_indices.astype(float)
lick_bin_idx = np.searchsorted(fi_sorted, trial_lick_frs)
```

iii. The rationale is that `LickFr` is expressed in neural-frame coordinates and nearest-frame assignment aligns it to the retained grid.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from framewise `ft_Pos` at the selected trial indices.

ii.
```python
positions = ft_Pos[frame_indices]
```

iii. The notes state positions are in decimeters and the target covers the 4 m textured corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Positions are clipped to `[0, 39.999]` dm and discretized with 10, 20, and 30 dm internal edges, then clipped to category indices 0–3.

ii.
```python
positions = np.clip(positions, 0, CORRIDOR_LENGTH_DM - 0.001)
pos_bin = np.digitize(positions, POSITION_BIN_EDGES_DM[1:])
pos_bin = np.clip(pos_bin, 0, N_POSITION_BINS - 1)
```

iii. The agent follows the requested four equal 1 m bins and excludes grey-space frames via `ft_CorrSpc`.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Categories are `[0,10)`, `[10,20)`, `[20,30)`, and `[30,40]` decimeters, labeled `0-1m` through `3-4m`.

ii.
```python
POSITION_BIN_EDGES_DM = np.array([0, 10, 20, 30, 40])
np.digitize(positions, POSITION_BIN_EDGES_DM[1:])
```

iii. These are exactly four equal-length spatial bins requested by the task.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is indexed by the same selected frame indices as neural activity.

ii.
```python
positions = ft_Pos[frame_indices]
neural = spk[:, frame_indices].astype(np.float32)
```

iii. The agent states framewise behavior and neural data share the imaging-frame grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from framewise `ft_RunSpeed`; `ft_CorrSpc` and `ft_move` determine the population used to estimate thresholds and the frames included in trials.

ii.
```python
speeds = ft_RunSpeed[mask]
speeds = ft_RunSpeed[frame_indices]
```

iii. The notes identify `ft_RunSpeed` as the direct behavioral speed measurement.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Before conversion, the code gathers corridor/running speeds from all sessions, randomly subsampling each session to at most 10,000 values with the same seed, concatenates them, and calculates global percentile edges. Per-trial speeds are digitized using the three internal edges.

ii.
```python
if len(speeds) > 10000:
    rng = np.random.default_rng(42)
    speeds = rng.choice(speeds, 10000, replace=False)
edges = np.percentile(all_speeds, [0, 25, 50, 75, 100])
```

iii. The agent says global thresholds provide consistent bins across sessions and are intended to put 25% of data in each bin.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize` uses approximate global 25th, 50th, and 75th percentile value thresholds, producing indices 0–3. Ties are not rank-split, and equal per-session subsampling means the thresholds are not exact quartiles of all converted frames.

ii.
```python
speed_bin = np.digitize(speeds, speed_edges[1:-1])
speed_bin = np.clip(speed_bin, 0, N_SPEED_BINS - 1)
```

iii. The notes call these “quartile bins (computed across all data),” though the implementation approximates that distribution through capped sampling.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed values are selected with the identical `frame_indices` used for neural columns and digitized one-for-one.

ii.
```python
speeds = ft_RunSpeed[frame_indices]
neural = spk[:, frame_indices].astype(np.float32)
```

iii. The agent relies on the shared frame grid for alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural and behavior streams are truncated to their common frame count. Missing behavior skips a session; mismatched neuron/retinotopy lengths are truncated to their minimum; too-short/unknown-stimulus trials and sessions with fewer than two trials are skipped. Warnings are printed rather than raising errors.

ii.
```python
nfr = min(n_frames, nfr_beh)
if beh is None: return None
min_n = min(n_neurons_raw, len(iarea))
if len(frame_indices) < 2: continue
```

iii. The agent aimed for robust full-dataset completion and documented validation showing no NaN/Inf and expected counts. It did not document a principled neuron-mismatch repair beyond truncation.

## 12-a. What are the most time-consuming steps of the code?

i. Loading/concatenating the very large spike files, copying filtered neurons into trial arrays, holding/saving the 151.58 GB pickle, and the extra all-session behavior pass for speed thresholds dominate. Optional plotting and decoder training are outside core conversion or separately invoked.

ii.
```python
spk = np.concatenate(dat['spks'], axis=0)
neural = spk[:, frame_indices].astype(np.float32)
pickle.dump(data, f, protocol=4)
```

iii. The script records separate load/filter/process/session timings, while the notes emphasize dataset size and report the final file size; they do not provide a dedicated efficiency analysis.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-neuron brain-region lookup, the per-lick nearest-frame loop, repeated per-trial full-array masks, and speed-label loop could be vectorized. The trial loop itself is still useful for variable-length output construction.

ii.
```python
for i, ia in enumerate(filtered_iarea): ...
for trial_idx in range(ntrials):
    trial_mask = (ft_trInd == trial_idx) & ...
for li in range(len(trial_lick_frs)): ...
```

iii. The agent labels licking “vectorized,” but retains a loop for nearest-neighbor resolution; no explicit justification of these loops appears in the notes.

## 12-c. What processing does the code repeat multiple times?

i. `get_session_beh` repeatedly scans the entire experiment index and reloads behavior files once per session during speed-edge computation, then again during session conversion, and again for optional plotting. Frame masks are also rebuilt independently for every trial.

ii.
```python
for s in sessions:
    beh, _, _ = get_session_beh(s['key'], exp_info)
...
beh, exp_type, db_entry = get_session_beh(key, exp_info)
```

iii. The agent did not discuss this repeated work; the notes merely describe the loader and processing stages.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes unused `frame_dt_s`, `reward_avail`, `valid_trial_indices`, `all_subjects`, `nfr_beh` in plotting, and several timing/metadata intermediates. It also creates float32 trial copies and diagnostic plots (when requested) that the decoder does not require.

ii.
```python
frame_dt_s = float(np.median(np.diff(ft)) * 86400)
reward_avail = np.full(1, float(isRew[trial_idx]), dtype=np.float32)
valid_trial_indices.append(trial_idx)
all_subjects = []
```

iii. These are remnants of development, diagnostics, or documentation. The agent's notes do not identify them as unnecessary.
