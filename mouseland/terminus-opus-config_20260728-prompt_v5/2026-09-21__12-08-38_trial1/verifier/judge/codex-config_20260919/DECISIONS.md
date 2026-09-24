# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads `beh/Imaging_Exp_info.npy` as the master index, deduplicates recordings by mouse/date/block, caches `Beh_<exp_type>.npy` behavior dictionaries, and loads each session's spike and retinotopy files. It processes all 89 unique sessions unless `--sample` is used.

ii.
```python
info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
               allow_pickle=True).item()
key = f"{db['mname']}_{db['datexp']}_{db['blk']}"
spk = load_spk(db)
retino = load_retino(db)
```

iii. The notes say this matches the reference loaders and reproduces the paper's 89 recordings and 19 mice. Behavior files are cached to avoid repeated reads.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `db['mname']`. A first-seen ordered dictionary assigns each mouse an integer, and one `subject_idx` is appended per retained session.

ii.
```python
mname = db['mname']
if mname not in subjects_set:
    subjects_set[mname] = len(subjects_set)
subject_idx_list.append(subjects_set[mname])
subjects = list(subjects_set.keys())
```

iii. The notes report 19 unique mouse names, matching the paper.

## 1-c. How are the data split into sessions?

i. A session is uniquely keyed by mouse, experiment date, and block. Duplicate appearances under experiment types are discarded, keeping the first occurrence.

ii.
```python
key = f"{db['mname']}_{db['datexp']}_{db['blk']}"
if key not in seen_keys:
    seen_keys.add(key)
    all_sessions.append((exp_type, db, key))
```

iii. The AI observed 89 unique spike/retinotopy session files and says this agrees with the paper.

## 1-d. How are the data split into trials?

i. Each trial is sliced from the floored, clipped `StartFr` (inclusive) to floored, clipped `EndFr` (exclusive). Thus the window includes the 4 m textured corridor and subsequent gray space rather than selecting frames labeled as inside the textured corridor.

ii.
```python
start_fr_int = np.floor(start_fr).astype(int)
end_fr_int = np.floor(end_fr).astype(int)
s_fr = start_fr_int[tr]
e_fr = end_fr_int[tr]
neural = spk[:, s_fr:e_fr].astype(np.float16)
```

iii. The notes explicitly choose “StartFr to EndFr (including gray space)” and retain all frames to preserve temporal continuity.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than two frames after clipped StartFr–EndFr slicing are skipped. No running-time, extreme-duration, or other quality filter is applied; very long trials are retained.

ii.
```python
n_trial_frames = e_fr - s_fr
if n_trial_frames < 2:
    skipped_trials += 1
    continue
```

iii. The notes state that all frames and even trials up to 5,621 frames are intentionally included for temporal continuity.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from the `spks` arrays for every imaging plane in `<mouse>_<date>_<block>_neural_data.npy`; `iarea` from the matching retinotopy file supplies region labels.

ii.
```python
spk = np.concatenate([nspk for nspk in
    np.load(spk_path, allow_pickle=True).item()['spks']], 0)
iarea = retino['iarea']
```

iii. The notes identify these as Suite2p deconvolved fluorescence traces and say concatenating planes matches `utils.load_spk`.

## 2-b. How is the `neural` data processed?

i. Plane arrays are concatenated, optionally truncated on a neuron-count mismatch, randomly subsampled to at most 5,000 neurons with seed 42, sliced by trial, and cast to float16. There is no dF/F calculation, smoothing, or temporal interpolation.

ii.
```python
if n_total > MAX_NEURONS:
    rng = np.random.RandomState(42)
    neuron_idx = np.sort(rng.choice(n_total, MAX_NEURONS, replace=False))
    spk = spk[neuron_idx]
neural = spk[:, s_fr:e_fr].astype(np.float16)
```

iii. The AI says the source already contains deconvolved traces. It justifies 5,000-cell subsampling by file size and the decoder's later PCA to 100 components, and float16 as a 2× storage reduction.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code does not exclude cells outside V1/mHV/lHV/aHV. It labels them `unassigned`, then randomly retains 5,000 cells from the entire population when necessary. If spike and retinotopy cell counts disagree, both are truncated to the shorter length.

ii.
```python
regions.append(area_map.get(a_int, 'unassigned'))
brain_regions = ['V1', 'mHV', 'lHV', 'aHV', 'unassigned']
min_n = min(n_neu_spk, n_neu_ret)
spk = spk[:min_n]
```

iii. The notes describe no explicit neuron curation beyond Suite2p, and regard inclusion of `unassigned` as consistent, while documenting subsampling as a resource compromise.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each neural trial begins at floored `StartFr`, treated as corridor entry/trial start, and runs to `EndFr`. Fractional starts are lost in the neural slice, although time-since-start retains the original fractional value.

ii.
```python
s_fr = start_fr_int[tr]
e_fr = end_fr_int[tr]
neural = spk[:, s_fr:e_fr].astype(np.float16)
```

iii. The metadata names “Trial start (corridor entry)” and the notes choose the entire StartFr–EndFr interval to maintain continuity.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One imaging frame is one bin at 3.17 Hz, reported as about 315.46 ms. No rebinning or resampling is applied.

ii.
```python
FS = 3.17
TIME_BIN_MS = 1000.0 / FS
'time_bin_size': TIME_BIN_MS,
```

iii. The notes identify 3.17 Hz from the reference notebook and retain the native frame grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from the current absolute frame indices and the per-trial `SoundFr` value, using the constant frame rate; raw `ft` timestamps are not used.

ii.
```python
frame_indices = np.arange(s_fr, e_fr, dtype=np.float64)
time_to_cue = (frame_indices - sound_fr[tr]) / FS
```

iii. The mapping table describes it as `(frame - SoundFr) / fs` in seconds.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The code subtracts `SoundFr` from each frame index and divides by 3.17. This yields negative values before the cue and positive values after it, despite the name “time to cue.”

ii.
```python
time_to_cue = (frame_indices - sound_fr[tr]) / FS
```

iii. The inline comment and notes explicitly specify “negative before, positive after.”

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated over the identical integer frame range `s_fr:e_fr`, so it has one value per neural column.

ii.
```python
frame_indices = np.arange(s_fr, e_fr, dtype=np.float64)
neural = spk[:, s_fr:e_fr].astype(np.float16)
```

iii. The AI's sanity checks say this value matched its manual frame-based calculation.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is taken directly from session metadata field `sess#`, falling back to `days`, then to zero.

ii.
```python
if 'sess#' in db:
    return float(db['sess#'])
elif 'days' in db:
    return float(db['days'])
return 0.0
```

iii. The planning table describes `sess# or days` as the direct source.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The selected metadata value is converted to float and broadcast across every frame of the trial; the code does not derive an ordinal training-day index from chronological sessions.

ii.
```python
day_of_training = get_day_of_training(db)
day_input = np.full(n_trial_frames, day_of_training, dtype=np.float64)
```

iii. The AI considered the metadata value already suitable as a per-trial input.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It uses the absolute frame indices, per-trial `StartFr`, and the fixed 3.17 Hz rate, not `ft` timestamps.

ii.
```python
time_since_start = (frame_indices - start_fr[tr]) / FS
```

iii. The mapping plan specifies `(frame - StartFr) / fs`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The original (possibly fractional) `StartFr` is subtracted from each retained integer frame and the difference is converted to seconds. Because the neural start was floored, the first value can be slightly negative.

ii.
```python
time_since_start = (frame_indices - start_fr[tr]) / FS
```

iii. The notes say it is a continuous seconds-valued time series.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It uses the same `s_fr:e_fr` frame range and therefore has exactly one sample per neural column.

ii.
```python
frame_indices = np.arange(s_fr, e_fr, dtype=np.float64)
neural = spk[:, s_fr:e_fr].astype(np.float16)
```

iii. Alignment is frame-index based throughout the conversion.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes directly from the trial-level `isRew` array.

ii.
```python
is_rew = np.array(beh['isRew']).astype(float)
```

iii. The mapping table identifies `isRew` as a Boolean converted to float.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The trial flag is broadcast unchanged across all trial frames and later stored as float32 with the other inputs.

ii.
```python
reward_input = np.full(n_trial_frames, is_rew[tr], dtype=np.float64)
inputs = np.stack([... reward_input], axis=0).astype(np.float32)
```

iii. The AI treats reward availability as a per-trial constant.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Per-trial labels come from `WallName`; the universe of labels is built from every behavior session's `UniqWalls`.

ii.
```python
for w in beh_cache[exp_type][beh_key]['UniqWalls']:
    all_stim.add(str(w))
wall_name = np.array(beh['WallName'])
```

iii. The notes describe `WallName` as the source and report 15 unique stimuli.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Fifteen raw wall names are sorted and assigned indices; the trial's exact wall variant is broadcast across time. Unknown names silently map to category 0. Variants are not collapsed into the four base categories.

ii.
```python
stim_list = sorted(all_stim)
stim_to_idx = {s: i for i, s in enumerate(stim_list)}
stim_idx = stim_to_idx.get(stim_name, 0)
stim_output = np.full(n_trial_frames, stim_idx, dtype=np.int64)
```

iii. The notes intentionally call this a 15-category task and use its 1/15 chance level in decoder evaluation.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived jointly from session lick frame values `LickFr` and their trial indices `LickTrind`.

ii.
```python
lick_fr = np.array(beh['LickFr']).astype(float)
lick_trind = np.array(beh['LickTrind']).astype(int)
```

iii. The notes map these fields to a binary per-frame signal.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Licks assigned to the current trial are rounded to an integer global frame, shifted by the trial start, range-checked, and marked as 1; all other bins are 0. Multiple licks in a frame remain binary.

ii.
```python
trial_lick_mask = lick_trind == tr
trial_lick_frames = np.round(lick_fr[trial_lick_mask]).astype(int) - s_fr
lick_binary[trial_lick_frames[valid]] = 1
```

iii. The notes state that licking is binary and time-varying, and sanity-check its presence in supervised sessions.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Global lick frames are converted into offsets relative to the same floored `s_fr` used for neural slicing, creating a vector of identical length.

ii.
```python
lick_binary = np.zeros(n_trial_frames, dtype=np.int64)
trial_lick_frames = trial_lick_frames - s_fr
```

iii. The AI uses the common imaging-frame grid for alignment.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from the framewise position field `ft_Pos`, truncated to the available neural frame count.

ii.
```python
ft_pos = np.array(beh['ft_Pos'])[:nfr]
pos = ft_pos[s_fr:e_fr]
```

iii. The notes identify `ft_Pos` as decimeter-valued corridor position.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is divided by 10 decimeters with floor division, clipped to 0–3, and cast to an integer category.

ii.
```python
bins = np.clip(position_dm // POS_BIN_SIZE_DM, 0,
               N_POS_BINS - 1).astype(np.int64)
```

iii. The AI intended four 1 m bins over the 4 m texture.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Categories are 0–<10, 10–<20, 20–<30, and at least 30 dm. Because trials extend through gray space, all positions from 4–6 m are clipped into the fourth category.

ii.
```python
POS_BIN_SIZE_DM = TEXTURE_LENGTH_DM / N_POS_BINS
bins = np.clip(position_dm // POS_BIN_SIZE_DM, 0, 3).astype(np.int64)
```

iii. The notes explicitly accept gray space in bin 3, despite observing that this makes the final class 48.6% of the data.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. The already frame-aligned `ft_Pos` vector is sliced with the same `s_fr:e_fr` boundaries as neural data.

ii.
```python
neural = spk[:, s_fr:e_fr].astype(np.float16)
pos = ft_pos[s_fr:e_fr]
```

iii. The notes report a manual position-bin sanity check.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from framewise `ft_RunSpeed`, both to calculate global boundaries and to label each trial.

ii.
```python
speeds = np.array(beh['ft_RunSpeed'])
ft_run_speed = np.array(beh['ft_RunSpeed'])[:nfr]
```

iii. The mapping table identifies the raw speed stream directly.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Three percentile values are computed once over concatenated full behavior streams from all deduplicated sessions. Each retained speed is passed through `np.digitize` against those global thresholds.

ii.
```python
quartiles = np.percentile(all_speeds, [25, 50, 75])
bins = np.digitize(speed, quartiles)
```

iii. The notes choose global quartiles across all frames and all sessions so labels share fixed value boundaries.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize` assigns labels 0–3 using the global 25th, 50th, and 75th percentile values. Tied speeds, especially zeros, stay together, so the resulting classes do not each contain 25% of retained data.

ii.
```python
def discretize_speed(speed, quartiles):
    bins = np.digitize(speed, quartiles)
    return bins.astype(np.int64)
```

iii. The AI acknowledges one class reaches 40.9% because many frames share near-zero speed, but still considers the percentile-threshold approach correct.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. The framewise speed vector is truncated to neural length and sliced with exactly the neural trial boundaries before discretization.

ii.
```python
speed = ft_run_speed[s_fr:e_fr]
speed_binned = discretize_speed(speed, speed_quartiles)
```

iii. Alignment is by common frame index.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Behavior streams are truncated to neural length; start/end indices are clipped; too-short trials and missing behavior keys are skipped. Spike/retinotopy count mismatches are truncated to the shorter count. Unknown areas become `unassigned`, unknown stimuli default to class 0, and warnings are globally suppressed.

ii.
```python
warnings.filterwarnings('ignore')
start_fr_int = np.clip(start_fr_int, 0, nfr - 1)
end_fr_int = np.clip(end_fr_int, 0, nfr)
min_n = min(n_neu_spk, n_neu_ret)
stim_idx = stim_to_idx.get(stim_name, 0)
```

iii. The notes mention negative StartFr clipping and describe the converted data as passing format and manual sanity checks.

## 12-a. What are the most time-consuming steps of the code?

i. Loading and concatenating very large session spike files, trial slicing/copying those arrays into float16, retaining the whole converted dataset in memory, and serializing the roughly 20 GB pickle dominate. Global behavior scans are smaller but add work.

ii.
```python
spk = np.concatenate([nspk for nspk in
    np.load(spk_path, allow_pickle=True).item()['spks']], 0)
neural = spk[:, s_fr:e_fr].astype(np.float16)
pickle.dump(data, f, protocol=4)
```

iii. The notes measured about 6.6 seconds per sample session and estimated roughly 9.8 minutes overall; neuron subsampling and float16 were chosen mainly to limit storage and decoder cost.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Area-code mapping loops neuron-by-neuron in Python and could use array masks/lookups. Trial processing must produce ragged arrays but several per-trial constants and lick assignments could be prepared vectorially. Stimulus collection repeatedly walks sessions/walls.

ii.
```python
for a in iarea:
    regions.append(area_map.get(int(a), 'unassigned'))
for tr in range(ntrials):
    ...
```

iii. The AI did not document vectorization opportunities; it focused on I/O and memory controls.

## 12-c. What processing does the code repeat multiple times?

i. It scans experiment metadata/behavior repeatedly: once to build sessions, once to collect stimuli, once to calculate speed quartiles, and again during conversion. Behavior files are cached within each pass but loaded again across passes. Trial arrays are separately sliced for neural, position, and speed.

ii.
```python
for exp_type, sessions in info.items():  # session list
...
for exp_type, sessions in info.items():  # stimuli
...
quartiles = compute_global_speed_quartiles(info)  # another scan
```

iii. The notes do not call this repetition out; they emphasize behavior caching during processing.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads and concatenates every neuron before randomly retaining only 5,000; computes labels for gray-space frames that should not be part of the 4 m texture trial under the reference; loads all behavior to discover stimuli and again for speed thresholds; and computes several summary statistics solely for logging. Optional plotting also converts trial data back to float32 and builds figures not used by conversion.

ii.
```python
spk = load_spk(db)
spk = spk[neuron_idx]
all_speeds = np.concatenate(all_speeds)
total_neurons = sum(br.shape[0] for br in brain_region_idx_all)
```

iii. The AI regards neuron loading/subsampling, global speed computation, summaries, and optional plots as useful validation or resource-management work, rather than documenting them as discarded downstream work.
