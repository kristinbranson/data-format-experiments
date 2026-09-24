# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads `beh/Imaging_Exp_info.npy` as the master index, deduplicates recordings by mouse/date/block, finds a behavior entry through its experiment-type files, concatenates all neural planes from the session spike file, and reads the matching retinotopy file. Behavior files are reloaded by `find_behavior_for_session` for each use rather than grouped and cached.

ii.
```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
beh_data = np.load(beh_file, allow_pickle=True).item()
data = np.load(path, allow_pickle=True).item()
spk = np.concatenate(data['spks'], axis=0)
```

iii. The notes say this follows the reference loaders and reconciles 142 index entries with 89 unique recordings.

## 1-b. How are the data split into subjects?

i. Sessions are grouped by the `mname` field. Final subjects are sorted unique mouse names, and each session gets the corresponding integer index.

ii.
```python
all_subjects = sorted(set(r['mname'] for r in session_results))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
subject_idx.append(subject_to_idx[result['mname']])
```

iii. The notes identify 19 mice and treat the mouse identifier in the master index as authoritative.

## 1-c. How are the data split into sessions?

i. A unique session is `(mname, datexp, blk)`. Duplicate appearances under experiment types are merged; behavior loading prefers an entry without `stimtype`, then falls back to the suffixed key.

ii.
```python
key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
if key not in sessions:
    sessions[key] = {'mname': ndb['mname'], 'datexp': ndb['datexp'], 'blk': ndb['blk'], ...}
```

iii. The AI justified this with the paper's 89 recordings and the presence of the same physical session in multiple experiment types.

## 1-d. How are the data split into trials?

i. Trials are enumerated from `0` to `ntrials-1`. A trial contains only frames labeled with that trial, in the texture corridor, and moving (`ft_move > 0`); this can remove internal stopped frames and make the retained sequence temporally non-contiguous.

ii.
```python
frame_mask = trial_mask & corr_mask & move_mask
return np.where(frame_mask)[0]
for t in range(ntrials):
    frame_idx = extract_trial_frames(beh, t, n_neural_frames)
```

iii. The notes cite the paper's “only considered timepoints during running” language and the fact that the requested position bins cover the 4 m texture region.

## 1-e. How are trials filtered based on quality controls?

i. A trial is skipped if fewer than two moving corridor frames remain. A whole session is skipped if fewer than two such trials remain. There is no long-trial/outlier filter.

ii.
```python
if len(frame_idx) < 2:
    continue
...
if len(neural_trials) < 2:
    return None
```

iii. The notes say the reference has no explicit trial filtering; the retained-frame rule is presented as matching its running-frame analysis and the decoder's minimum-session requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from the per-plane `spks` arrays in each `_neural_data.npy`; `iarea` from retinotopy supplies neuron-region assignments and filtering.

ii.
```python
spk = np.concatenate(data['spks'], axis=0)
iarea = load_retinotopy(mname, datexp)
spk_filtered = spk[neuron_mask]
```

iii. The AI states that these are already Suite2p deconvolved fluorescence traces, so no raw fluorescence conversion is needed.

## 2-b. How is the `neural` data processed?

i. Planes are concatenated, neurons are filtered by retinotopy, and selected trial-frame columns are copied to float32. No normalization, deconvolution, interpolation, padding, or temporal resampling is applied.

ii.
```python
spk = np.concatenate(data['spks'], axis=0)
neural = spk_filtered[:, frame_idx].astype(np.float32)
```

iii. The notes cite the methods' use of already-deconvolved fluorescence and retain native imaging frames for decoder alignment.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons with `iarea == -1` or `iarea == 7` are removed; all other area codes are mapped to V1, mHV, lHV, or aHV. If neural and retinotopy counts differ, both are silently truncated to the smaller count.

ii.
```python
mask = (iarea != -1) & (iarea != 7)
min_n = min(len(iarea), n_neurons_total)
iarea = iarea[:min_n]
spk = spk[:min_n]
```

iii. The notes cite the reference area mapping and upstream Suite2p curation, with no additional cell-quality criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each variable-length trial begins with retained moving frames at/after corridor entry; neural columns use those exact frame indices. Stopped corridor frames are removed, so elapsed time is not represented by a contiguous neural sequence.

ii.
```python
frame_idx = extract_trial_frames(beh, t, n_neural_frames)
neural = spk_filtered[:, frame_idx].astype(np.float32)
```

iii. The AI calls corridor entry the alignment event, uses `off_start=0`, and argues that native frame-index alignment keeps all streams synchronized.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One native imaging frame is one bin, approximately 315 ms. No rebinning occurs; metadata uses the mean of session median frame intervals.

ii.
```python
frame_dt_sec = np.nanmedian(np.diff(ft)) * 24 * 3600
mean_frame_dt = np.mean(frame_dts)
'time_bin_size': mean_frame_dt * 1000
```

iii. The notes say native imaging frames are the available temporal resolution and report 314.8 ms.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from each trial's `SoundFr`, retained neural frame indices, and the median interval of `ft` timestamps.

ii.
```python
sound_fr = beh['SoundFr'][t]
frame_dt_sec = np.nanmedian(np.diff(ft)) * 24 * 3600
time_to_cue = (frame_idx - sound_fr) * frame_dt_sec
```

iii. The notes identify `SoundFr` as an interpolated frame index and `ft` as the source of seconds per frame.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The code subtracts cue frame from current frame and multiplies by a constant median frame interval. Thus it is negative before and positive after the cue—actually time since cue, not time to cue.

ii.
```python
time_to_cue = (frame_idx - sound_fr) * frame_dt_sec
```

iii. The planning notes give the opposite formula, `(SoundFr - frame_index)`, while also claiming “negative before, positive after”; the implementation follows the stated sign convention but not the variable's “time to” semantics or reference.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at the same `frame_idx` columns selected for neural data and has exactly the same number of timepoints.

ii.
```python
neural = spk_filtered[:, frame_idx].astype(np.float32)
inp[0, :] = time_to_cue.astype(np.float32)
```

iii. The AI justifies alignment through common imaging-frame indices.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It comes from `mname` and `datexp` in the master session index.

ii.
```python
mouse_dates[info['mname']].append((key, parse_date(info['datexp'])))
```

iii. The notes say session dates provide a per-mouse training timeline.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the code subtracts the earliest recording date from each session's calendar date, producing elapsed calendar days, then broadcasts that number over the trial.

ii.
```python
first_date = date_list[0][1]
day_map[key] = (dt - first_date).days
inp[1, :] = day_val
```

iii. The AI explicitly chose “days since first session.” Its sample values of 12 and 79 confirm calendar-day gaps rather than ordinal recorded training sessions.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It uses `StartFr`, retained frame indices, and the median interval from `ft`.

ii.
```python
start_fr = beh['StartFr'][t]
time_since_start = (frame_idx - start_fr) * frame_dt_sec
```

iii. The notes identify `StartFr` as the fractional corridor-entry frame.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Frame-index offsets from `StartFr` are multiplied by a constant session median frame duration and stored as seconds.

ii.
```python
time_since_start = (frame_idx - start_fr) * frame_dt_sec
inp[2, :] = time_since_start.astype(np.float32)
```

iii. The AI says this yields a continuous time-varying input starting near zero.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Values are calculated for the same retained frame indices as neural columns. Because stopped frames are removed, values can jump while neural samples remain adjacent array columns.

ii.
```python
neural = spk_filtered[:, frame_idx].astype(np.float32)
inp[2, :] = time_since_start.astype(np.float32)
```

iii. The AI relies on common frame indices for alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is read directly from the per-trial `isRew` flag.

ii.
```python
reward_avail = float(beh['isRew'][t])
```

iii. The notes describe `isRew` as the rewarded-corridor boolean.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean is cast to float and broadcast over every retained timepoint.

ii.
```python
inp[3, :] = reward_avail
```

iii. The AI treats it as a per-trial decoder input represented on the common `(variables, time)` grid.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived directly from the per-trial `WallName` string.

ii.
```python
wall_names = beh['WallName']
stim_name = wall_names[t]
```

iii. The notes prefer `WallName` because it remains available in sessions where other stimulus identifiers may be masked.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Every distinct wall name is globally sorted and assigned an index, leaving 15 crop/swap variants as separate classes rather than collapsing them to four base textures; the index is broadcast over the trial.

ii.
```python
all_stimuli = sorted(all_stimuli)
stim_to_idx = {s: i for i, s in enumerate(all_stimuli)}
out_filled[0, :] = stim_to_idx[stim_name]
```

iii. The AI planned to “use WallName directly” and reported 15 unique stimuli as a successful sanity check.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It uses `LickFr` and `LickTrind`, selecting lick events belonging to the current trial.

ii.
```python
trial_lick_mask = lick_trind == trial_idx
trial_lick_fr = lick_fr[trial_lick_mask]
```

iii. The notes describe these as frame indices and trial identities for every lick.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Fractional lick frames are rounded to the nearest integer, placed in a set, and each retained frame is labeled 1 on exact membership, else 0.

ii.
```python
lick_int_frames = np.round(trial_lick_fr).astype(int)
lick_set = set(lick_int_frames)
binary = np.array([1.0 if fi in lick_set else 0.0 for fi in frame_indices])
```

iii. The AI says this makes a binary per-frame signal; it chose nearest-frame rounding for interpolated lick indices.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Binary values are generated only for, and in the order of, the exact retained neural frame indices.

ii.
```python
licking = make_lick_binary(beh, frame_idx, t)
out[1, :] = licking.astype(np.int64)
```

iii. The AI cites shared frame numbers as the alignment mechanism.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from framewise `ft_Pos` at the retained indices.

ii.
```python
positions = beh['ft_Pos'][:n_use]
trial_positions = positions[frame_idx[frame_idx < n_use]]
```

iii. The notes state that position units are decimeters and the texture corridor spans 0–40.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Values are clipped to `[0, 40)`, discretized, and cast to integers. If a selected index exceeds available position data, the code pads with the last valid position.

ii.
```python
pos_clipped = np.clip(positions, 0, TEXTURE_LENGTH - 1e-6)
bins = np.digitize(pos_clipped, bin_edges) - 1
trial_positions = np.concatenate([trial_positions, np.full(pad_len, trial_positions[-1] ...)])
```

iii. The AI says this converts decimeters to the requested 1 m classes and uses padding defensively for length mismatches.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Fixed edges `[0,10,20,30,40]` dm yield `[0,10)`, `[10,20)`, `[20,30)`, and `[30,40]`, labeled 0–3.

ii.
```python
POS_BIN_EDGES = np.array([0, 10, 20, 30, 40])
bins = np.digitize(pos_clipped, bin_edges) - 1
bins = np.clip(bins, 0, N_POS_BINS - 1)
```

iii. The notes link ten decimeters to one meter and the four requested equal-length bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is indexed by the same retained frame indices and then inserted into an output array of neural-trial length, with last-value padding only on a source-length mismatch.

ii.
```python
trial_positions = positions[frame_idx[frame_idx < n_use]]
out[2, :] = pos_bins
```

iii. The AI's rationale is common native frame indexing.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from framewise `ft_RunSpeed`, with `ft_CorrSpc` and `ft_move` used to choose the population for global thresholds.

ii.
```python
mask = ft_CorrSpc & (ft_move > 0)
speeds = ft_RunSpeed[mask]
trial_speeds = run_speeds[frame_idx[frame_idx < n_use_speed]]
```

iii. The notes identify `ft_RunSpeed` as the direct speed measure and restrict it to running texture frames.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Up to 10,000 eligible frames per session are randomly sampled with a fixed seed; pooled 25th/50th/75th percentiles become global thresholds. Trial values are discretized, with last-value padding on mismatch.

ii.
```python
if len(speeds) > 10000:
    rng = np.random.RandomState(42)
    idx = rng.choice(len(speeds), 10000, replace=False)
quartiles = np.percentile(all_speeds, [25, 50, 75])
speed_bins = discretize_speed(trial_speeds, speed_quartiles)
```

iii. The AI chose global thresholds for “consistent binning” across sessions and subsampling for efficiency.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize` assigns 0–3 using the three sampled global percentile values (reported as about 12.2, 25.0, and 40.5 cm/s). Ties remain together, so final classes need not each contain exactly 25%.

ii.
```python
quartiles = np.percentile(all_speeds, [25, 50, 75])
bins = np.digitize(speeds, quartile_edges)
```

iii. The AI interpreted “each corresponding to 25% of the data” as global quartile thresholds.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed values are selected at the same retained frame indices as neural columns; missing tail values are padded with the final observed speed.

ii.
```python
trial_speeds = run_speeds[frame_idx[frame_idx < n_use_speed]]
out[3, :] = speed_bins
```

iii. The AI relies on shared frame indices for temporal alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Frame extraction is capped at the shorter neural/behavior frame count. Neural/retinotopy neuron mismatches are warned about then truncated to the minimum. Missing behavior skips a session; short trials/sessions are skipped; short position/speed streams are padded with their last value (or zero). NaNs in trial indices are excluded.

ii.
```python
n_frames = min(n_neural_frames, n_beh_frames)
valid = np.isfinite(ft_trInd)
min_n = min(len(iarea), n_neurons_total)
if beh is None: return None
```

iii. The notes call the one-frame behavior/neural mismatch expected and emphasize clipping to neural frames; the other branches are defensive safeguards.

## 12-a. What are the most time-consuming steps of the code?

i. Loading and concatenating roughly 434 GB of spike files dominates; building the 142 GB float32 output and pickle also costs substantial memory/I/O. The preliminary global-speed pass reloads behavior but was measured at only seconds.

ii.
```python
data = np.load(path, allow_pickle=True).item()
spk = np.concatenate(data['spks'], axis=0)
neural = spk_filtered[:, frame_idx].astype(np.float32)
pickle.dump(data, f, protocol=4)
```

iii. The notes estimate processing from spike-data size and report about five seconds per GB.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial extraction scans all session frames once per trial; lick labeling loops over retained frames; output assembly loops over every trial to copy and fill stimulus codes. These could be grouped/indexed vectorially, though spike I/O dominates.

ii.
```python
for t in range(ntrials):
    frame_idx = extract_trial_frames(...)
binary = np.array([1.0 if fi in lick_set else 0.0 for fi in frame_indices])
for out_data, stim_name in result['output']:
```

iii. The AI did not explicitly discuss these vectorization opportunities; its efficiency justification focuses on speed subsampling and I/O.

## 12-c. What processing does the code repeat multiple times?

i. `find_behavior_for_session` reloads whole behavior files for every session during speed collection and again during conversion. It can also try/reload multiple candidate experiment files. Plot mode reloads neural, retinotopy, and behavior for already processed sessions.

ii.
```python
for session_key in session_keys:
    beh, _ = find_behavior_for_session(...)
...
for i, session_key in enumerate(session_keys):
    result = process_session(...)
```

iii. The AI did not justify repeated behavior loading; it only described the global pass as necessary for common speed thresholds.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `build_output` contains a dead, always-empty stimulus loop. `collect_all_running_speeds` gathers/subsamples values only to discard them after three percentiles. `process_session` carries metadata not emitted in the final dataset, and optional plots reload and recompute data solely for diagnostics.

ii.
```python
for _, stim_name in [r for r in [(out[1]) for out in result['output']]
                     if isinstance(r, str)] if False else []:
    pass
```

iii. The diagnostic plots and global threshold pass were intended for validation/consistent labels; the dead loop has no documented rationale.
