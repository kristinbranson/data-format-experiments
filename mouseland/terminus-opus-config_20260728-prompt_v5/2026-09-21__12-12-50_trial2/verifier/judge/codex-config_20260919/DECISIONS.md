# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads `beh/Imaging_Exp_info.npy` as a master index, deduplicates recordings by mouse/date/block, loads and concatenates all imaging planes from one spike file per recording, loads one retinotopy file per recording date, and finds the session in a `Beh_<experiment type>.npy` file. Behavioral dictionaries are cached globally. Full mode processes all 89 unique session keys.

ii.
```python
exp_info = load_exp_info()
sessions = get_unique_sessions(exp_info)
session_keys = sorted(sessions.keys())
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([nspk for nspk in spk_data['spks']], 0)
dtrans = np.load(ret_path, allow_pickle=True)
```

iii. The notes say this matches the paper's 89 recordings and 19 mice and the reference loading functions. Caching was justified as avoiding repeated loads of large behavior files.

## 1-b. How are the data split into subjects?

i. Subject identity is the `mname` field. Subjects are accumulated in first-session order, and every session receives the corresponding integer `subject_idx`.

ii.
```python
mname = sess['mname']
if mname not in subject_map:
    subject_map[mname] = len(unique_subjects)
    unique_subjects.append(mname)
subject_idx_list.append(subject_map[mname])
```

iii. The notes validate 19 unique mice, matching the paper.

## 1-c. How are the data split into sessions?

i. A session is uniquely identified by `mname_datexp_blk`. Duplicate listings under experiment types are merged into one dictionary record; all associated experiment types are retained to locate behavior.

ii.
```python
key = f"{s['mname']}_{s['datexp']}_{s['blk']}"
if key not in sessions:
    sessions[key] = {'mname': s['mname'], 'datexp': s['datexp'],
                     'blk': s['blk'], 'key': key, 'exp_types': []}
sessions[key]['exp_types'].append(exp_type)
```

iii. The agent states that 89 spike files produce 89 unique sessions and treats this as matching the paper.

## 1-d. How are the data split into trials?

i. The agent uses `beh['ntrials']`, then position-interpolates the session-wide moving-frame neural stream onto `n_trials * 60` regularly spaced normalized positions and reshapes it into one fixed 60-bin array per trial. Each trial includes the 4 m corridor and 2 m grey space.

ii.
```python
lin_pos = np.arange(0, n_trials, 1.0 / n_bins)
src_pos = accum_pos / corridor_len
interp_spk = interp_spk.reshape(n_neurons, n_trials, n_bins)
neural_trials = [interp_spk[:, t, :].astype(np.float32) for t in range(n_trials)]
```

iii. The notes say that 60 position bins match the paper's reference analysis and that including grey space is consistent with a 4 m corridor plus 2 m grey interval.

## 1-e. How are trials filtered based on quality controls?

i. Trials are not filtered. All `ntrials` are emitted. Only source frames with `ft_move > 0` are used for position interpolation.

ii.
```python
n_trials = beh['ntrials']
vr_moving = ft_move > 0
interp_spk = position_interpolate_spk(spk[:, vr_moving], ft_pos_cum[vr_moving],
                                      corridor_len, n_trials, N_POS_BINS)
```

iii. The notes interpret the paper's “only considered timepoints during running” as the applicable curation and report all 38,110 trials; they do not discuss empty or extreme-duration trial rejection.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from each plane in the spike file's `spks` list. `iarea` from the retinotopy file determines which neurons are retained and their brain-region labels.

ii.
```python
spk = np.concatenate([nspk for nspk in spk_data['spks']], 0)
iarea = load_retino(mname, datexp)
valid_mask, region_idx = get_brain_region_idx(iarea)
spk = spk[valid_mask]
```

iii. The notes identify `spks` as Suite2p deconvolved fluorescence and say no further deconvolution is necessary.

## 2-b. How is the `neural` data processed?

i. After area filtering and truncating behavior streams to the spike-frame count, the code discards nonmoving source frames and linearly interpolates each neuron's deconvolved trace against cumulative position onto 60 positions per trial. Output is float32.

ii.
```python
ft_pos_cum = beh['ft_PosCum'][:n_frames]
vr_moving = ft_move > 0
interp_spk[i + s] = np.interp(lin_pos, src_pos, batch[s])
neural_trials = [interp_spk[:, t, :].astype(np.float32) for t in range(n_trials)]
```

iii. The agent says position interpolation, moving-frame selection, and 60 bins match `spk_pos_interp` and the paper's analysis code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are retained only if `iarea` maps to V1, mHV, lHV, or aHV; unmapped areas such as -1 and 7 are excluded. Source timepoints are additionally restricted to `ft_move > 0` before interpolation.

ii.
```python
region_idx = np.full(n_neurons, -1, dtype=int)
for r_idx, (region, areas) in enumerate(AREA_MAPPING.items()):
    for area_val in areas:
        region_idx[iarea == area_val] = r_idx
valid_mask = region_idx >= 0
```

iii. The notes explicitly cite the paper/reference area mapping and running-only analysis as the curation rules.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It is not directly time-aligned using corridor-entry frame timestamps. Instead, cumulative position divided by corridor length defines trial phase; every trial is interpolated to positions 0–59, with bin 0 treated as trial start/corridor entry.

ii.
```python
lin_pos = np.arange(0, n_trials, 1.0 / n_bins)
src_pos = accum_pos / corridor_len
interp_spk[i + s] = np.interp(lin_pos, src_pos, batch[s])
```

iii. The notes call this temporal alignment “position interpolation using cumulative position” and assert that it matches the reference code, while metadata labels the event as corridor entry.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent reports 166.67 ms per bin, derived from a 0.1 m spatial bin and an assumed 0.6 m/s VR speed. Neural activity is resampled from native imaging frames to 60 position bins per trial.

ii.
```python
BIN_SIZE_M = TOTAL_LENGTH_M / N_POS_BINS
TIME_PER_BIN = BIN_SIZE_M / VR_SPEED
TIME_BIN_MS = TIME_PER_BIN * 1000
'time_bin_size': TIME_BIN_MS,
```

iii. The notes reason that the VR “always moved” at 60 cm/s and therefore assign a time duration to each spatial bin.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundPos` and the generated centers of the 60 position bins.

ii.
```python
sound_pos = beh['SoundPos']
positions = np.arange(n_bins) + 0.5
dist = sound_pos[:, np.newaxis] - positions[np.newaxis, :]
```

iii. The notes describe `SoundPos` as the direct source and convert position distance to time.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Sound position minus each bin center is multiplied by 0.1 m and divided by the fixed 0.6 m/s speed. Values are positive before and negative after the cue.

ii.
```python
time_to_cue = (dist * BIN_SIZE_M / VR_SPEED).astype(np.float32)
```

iii. The mapping plan says this produces a continuous, time-varying value in seconds.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Both have exactly 60 generated position bins, so array index aligns them spatially. This is not alignment through the native frame timestamps.

ii.
```python
time_to_cue = make_time_to_sound_cue(sound_pos, N_POS_BINS)
inp = np.stack([time_to_cue[t], ...], axis=0)
```

iii. The agent treats the shared position grid as alignment and reports fixed-size verified arrays.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It comes from the first index entry's `sess#` field, stored as `sess_num` when unique sessions are assembled.

ii.
```python
'sess_num': s.get('sess#', 0),
day_of_training = np.full(n_trials, sess_meta['sess_num'], dtype=np.float32)
```

iii. The notes call this a direct mapping from experiment information.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. No ordering or recomputation is performed. The session-level number is copied to every trial and broadcast across all 60 bins.

ii.
```python
np.full(N_POS_BINS, day_of_training[t], dtype=np.float32)
```

iii. The README interprets the value as session number (for example, 0 before and 1 after learning).

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is generated solely from the spatial-bin index plus constants for bin size and VR speed; no raw start frame or timestamp is used.

ii.
```python
time_per_bin = BIN_SIZE_M / VR_SPEED
times = (np.arange(n_bins) * time_per_bin).astype(np.float32)
```

iii. The mapping plan describes position-bin index times time per bin.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The same 0-to-9.833-second sequence at 1/6-second intervals is tiled over every trial.

ii.
```python
return np.tile(times, (n_trials, 1))
```

iii. The justification relies on constant 60 cm/s VR movement and 0.1 m spatial bins.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It shares the generated 60-position-bin index with interpolated neural data, with index zero declared to be trial start.

ii.
```python
time_since_start = make_time_since_trial_start(n_trials, N_POS_BINS)
neural_trials = [interp_spk[:, t, :] for t in range(n_trials)]
```

iii. The metadata declares corridor entry as the alignment event and a fixed 10-second endpoint.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is taken directly from per-trial `isRew`.

ii.
```python
reward_avail = beh['isRew'].astype(np.float32)
```

iii. The notes describe a direct Boolean-to-float mapping.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The float value is broadcast unchanged across the 60 bins of its trial.

ii.
```python
np.full(N_POS_BINS, reward_avail[t], dtype=np.float32)
```

iii. It is documented as a discrete per-trial input, with no substantive transformation.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It uses per-trial `WallName` and each session's `UniqWalls`, then collects all `UniqWalls` values across the dataset for global labels.

ii.
```python
wall_name = beh['WallName']
uniq_walls = beh['UniqWalls']
categories = np.array([category_names.index(wn) for wn in wall_name])
```

iii. The notes say wall names supply 15 global stimulus categories.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each exact wall variant gets a local category, is remapped to the sorted global list of 15 names, and is broadcast over the trial. Variants are not collapsed into four base textures.

ii.
```python
global_cat = all_stim_names.index(local_name)
result['output'][t][0, :] = global_cat
```

iii. The agent deliberately describes “15 stimulus categories” and regards all unique wall names as the requested categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It comes from lick event positions `LickPos` and trial indices `LickTrind`.

ii.
```python
lick_raster = make_lick_raster(beh['LickPos'], beh['LickTrind'],
                               n_trials, N_POS_BINS, corridor_len)
```

iii. The notes describe a binary lick raster at position bins.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each event, its position is located among 60 equal edges, clipped to the valid range, and that trial/bin is set to one; multiple licks in a bin remain one.

ii.
```python
bin_idx = np.searchsorted(bin_edges, pos, side='right') - 1
bin_idx = np.clip(bin_idx, 0, n_bins - 1)
lick_raster[tr, bin_idx] = 1.0
```

iii. The agent wanted a binary time-varying output and reports expected sparsity, including no licks in unsupervised animals.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks and neural activity are independently placed on the same 60-bin position grid and aligned by trial and position-bin index, not native imaging frame.

ii.
```python
lick_raster = make_lick_raster(..., N_POS_BINS, corridor_len)
out = np.stack([..., lick_raster[t].astype(np.int64), ...], axis=0)
```

iii. The notes treat shared position binning as matching the reference position-based processing.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is generated from output-bin index and constants, rather than read from framewise `ft_Pos`.

ii.
```python
pos_output = make_position_output(n_trials, N_POS_BINS)
for b in range(n_bins):
```

iii. The notes justify deterministic position labels because neural data were interpolated to a regular position grid.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. A fixed category vector is constructed for every trial: ten bins each for categories 0, 1, and 2, then category 3 for bins 30–59, including all grey-space bins.

ii.
```python
elif b < 30:
    pos_output[:, b] = 2
else:
    pos_output[:, b] = 3
```

iii. The agent acknowledges that bin 3 occupies 50% because grey space is included and calls this inherent to its trial definition.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position bins 0–9, 10–19, 20–29, and 30–59 map to categories 0–3, nominally representing four 1 m corridor segments; grey space is clipped into category 3.

ii.
```python
if b < 10: ... = 0
elif b < 20: ... = 1
elif b < 30: ... = 2
else: ... = 3
```

iii. The stated goal is four equal 1 m corridor categories, with grey assigned to the last category because no fifth output value exists.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Category labels are indexed on the same regular 60-position grid used to interpolate neural activity.

ii.
```python
pos_output = make_position_output(n_trials, N_POS_BINS)
neural_trials = [interp_spk[:, t, :] for t in range(n_trials)]
```

iii. The notes consider position-grid identity sufficient alignment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from the precomputed trial-by-position matrix `run_pos`.

ii.
```python
run_pos = beh['run_pos']
speed_output = make_speed_output(run_pos, speed_bin_edges)
```

iii. The mapping plan selects `run_pos` because it is already shaped `(n_trials, 60)`.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The code gathers flattened `run_pos` values across behavior dictionaries, removes nonfinite and nonpositive values, computes global 25th/50th/75th percentiles, and uses `np.digitize` on every value.

ii.
```python
valid = np.isfinite(all_speeds) & (all_speeds > 0)
quartiles = np.percentile(all_speeds[valid], [25, 50, 75])
speed_output = np.digitize(run_pos, speed_bin_edges[1:-1]).astype(np.float32)
```

iii. The notes say global quartiles provide Q1–Q4 with 25% each, while also noting negative/backward values are handled by binning.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three global positive-speed percentile thresholds define categories 0–3. Zero and negative values fall in category 0; ties are not split by rank.

ii.
```python
bin_edges = np.array([0, quartiles[0], quartiles[1], quartiles[2], np.inf])
np.digitize(run_pos, speed_bin_edges[1:-1])
```

iii. The intended justification is four dataset-wide quartiles, although filtering nonpositive values means the final complete dataset is not divided into four equal groups.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `run_pos` already has 60 position samples per trial and is stacked with the corresponding 60-bin interpolated neural trial.

ii.
```python
run_pos = beh['run_pos']  # (n_trials, 60)
out = np.stack([..., speed_output[t].astype(np.int64)], axis=0)
```

iii. The agent's position-grid strategy is intended to make all streams share indices.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Framewise movement and cumulative position are truncated to the neural frame count. Invalid speed values are excluded only while estimating thresholds; out-of-range lick trials are skipped and lick positions are clipped. Retinotopy mismatch triggers an assertion, missing behavior raises an exception, and the main session loop has no recovery handler. `np.interp` extrapolates endpoint values over uncovered target positions.

ii.
```python
ft_move = beh['ft_move'][:n_frames]
if tr < 0 or tr >= n_trials:
    continue
bin_idx = np.clip(bin_idx, 0, n_bins - 1)
assert len(iarea) == n_neurons_total
```

iii. The notes report no unresolved discrepancies and describe negative speeds as handled by quartile binning; they do not document broader missing-data repair.

## 12-a. What are the most time-consuming steps of the code?

i. Loading very large spike arrays and, especially, interpolating every neuron separately over every session dominate conversion. The notes report about 50 seconds for two sessions, estimate about 40 minutes for 89, and identify interpolation timing per session.

ii.
```python
spk_data = np.load(spk_path, allow_pickle=True).item()
for i in range(0, n_neurons, batch_size):
    for s in range(batch.shape[0]):
        interp_spk[i + s] = np.interp(lin_pos, src_pos, batch[s])
```

iii. The agent says batched `np.interp` was a fourfold optimization and prints interpolation duration, indicating it viewed this as the main compute bottleneck.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The nested per-neuron interpolation loop is the largest candidate. The per-lick loop, 60-bin position loop, per-trial input/output construction, stimulus remapping, and repeated per-area equality loop are also vectorizable.

ii.
```python
for s in range(batch.shape[0]):
    interp_spk[i + s] = np.interp(lin_pos, src_pos, batch[s])
for i in range(len(lick_pos)):
for b in range(n_bins):
for t in range(n_trials):
```

iii. The notes only explicitly justify batching and `np.interp` as memory-efficient/fast; they do not discuss remaining vectorization opportunities.

## 12-c. What processing does the code repeat multiple times?

i. Behavior files are read in separate full-dataset passes to collect stimulus names, compute speed thresholds, and later process sessions. `load_beh_for_session` also reconstructs all unique sessions on every cache miss. Trial arrays are traversed again to remap local stimulus codes globally.

ii.
```python
all_stim_names = collect_all_stim_names(exp_info)
speed_bin_edges = compute_global_speed_quartiles(exp_info)
sessions = get_unique_sessions(exp_info)  # inside load_beh_for_session
for t in range(result['n_trials']):
```

iii. The notes claim each helper loads each behavior file once for its own pass and emphasize caching, but do not acknowledge the multiple passes across helpers.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It builds an unused local `beh_cache`, an unused `subjects_list`, and stores unused session metadata fields. In optional plotting mode it computes figures that do not affect conversion. More materially relative to the requested corridor task, it interpolates and stores 20 grey-space bins per trial, then collapses them into the final corridor-position category.

ii.
```python
subjects_list = []
beh_cache = {}
'exptype': s.get('exptype', ''),
'rewType': s.get('rewType', ''),
if show_processing and session_idx < 2:
    plot_processing(...)
```

iii. The notes regard grey-space inclusion as intentional and plotting as visual validation; they do not identify the unused local variables or metadata.
