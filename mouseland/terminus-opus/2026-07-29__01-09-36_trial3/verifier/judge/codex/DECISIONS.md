# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `data/beh/Imaging_Exp_info.npy` as a master index, builds a `session_meta` dictionary keyed by `mname_datexp_blk`, records candidate behavior files/keys for each session, then loads spikes, behavior, and retinotopy per session during processing.

ii. 
```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()
session_meta = {}
for exp_type, sessions in exp_info.items():
    for db in sessions:
        sid = f"{db['mname']}_{db['datexp']}_{db['blk']}"
```

```python
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate(spk_data['spks'], axis=0)
beh_all = np.load(beh_path, allow_pickle=True).item()
dtrans = np.load(ret_path, allow_pickle=True)
```

iii. In the trajectory and notes, the AI states that the same recording can appear under multiple experiment types, so it decided to keep each unique session once and search multiple behavior keys/files for that session.

## 1-b. How are the data split into subjects?

i. Subjects are defined by `mname`. The final `subjects` list is the sorted set of session mouse names, and each session gets a `subject_idx` from that mapping.

ii.
```python
subjects_set = set()
for sid in session_ids:
    subjects_set.add(session_meta[sid]['mname'])
subjects_list = sorted(subjects_set)
subject_to_idx = {name: idx for idx, name in enumerate(subjects_list)}
```

iii. The AI treated mouse name as the authoritative subject identifier because `Imaging_Exp_info.npy` already groups recordings by mouse and the notes report 19 unique mice.

## 1-c. How are the data split into sessions?

i. A session is one unique `(mname, datexp, blk)` triple. Duplicate appearances across experiment types are merged into one `sid`.

ii.
```python
sid = f"{db['mname']}_{db['datexp']}_{db['blk']}"
if sid not in session_meta:
    session_meta[sid] = {
        'mname': db['mname'],
        'datexp': db['datexp'],
        'blk': db['blk'],
        'exptype': db.get('exptype', ''),
        'rewType': db.get('rewType', ''),
    }
```

iii. The AI explicitly justified this in its notes and trajectory: the same recording session appears in multiple experiment-type files, but the behavioral data is the same, so each unique session should be used once.

## 1-d. How are the data split into trials?

i. Trials are split by taking every frame whose `ft_trInd` equals the trial index. The AI keeps the full variable-length frame sequence for each trial, including corridor and gray-space frames, rather than restricting to `ft_CorrSpc` or forcing a fixed length.

ii.
```python
for trial_idx in range(ntrials):
    trial_mask = (ft_trInd == trial_idx)
    frame_indices = np.where(trial_mask)[0]
```

iii. The trajectory says the AI aligned to trial start and kept the full trial, reasoning that `ft_trInd` already defines trial membership and that excluding gray-space frames would change the trial structure.

## 1-e. How are trials filtered based on quality controls?

i. There is no substantive trial-quality filter. The AI skips trials with fewer than 2 remaining frames, either initially or after clipping to the imaged frame range.

ii.
```python
if len(frame_indices) < 2:
    skipped += 1
    continue

frame_indices = frame_indices[frame_indices < n_frames_spk]
if len(frame_indices) < 2:
    skipped += 1
    continue
```

iii. In the notes, the AI describes this as edge-case handling: include all trials except ones that are too short to be useful after frame-range clipping.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes from concatenating the session’s `spks` arrays across imaging planes. `iarea` from retinotopy is used to define region indices and filtering.

ii.
```python
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate(spk_data['spks'], axis=0)
```

```python
dtrans = np.load(ret_path, allow_pickle=True)
return dtrans['iarea']
```

iii. The notes state that the source neural signal is already deconvolved fluorescence traces from Suite2p output, so the AI used those arrays directly.

## 2-b. How is the `neural` data processed?

i. The AI filters neurons by retinotopy, subsamples to at most 2000 neurons per session, then slices each trial’s selected frames and stores the result as variable-length `float32` arrays. It does not pad trials to a common length.

ii.
```python
spk, iarea_filtered, selected_idx = filter_and_subsample_neurons(spk, iarea)
trial_neural = spk[:, frame_indices].astype(np.float32)
```

iii. The trajectory shows the main justification: storing all neurons made the dataset impractically large, so the AI used the decoder’s PCA/SVD settings to justify capping sessions at 2000 neurons.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons with `iarea == -1` or `iarea == 7` are removed. If more than 2000 valid neurons remain, the AI subsamples them stratified by brain region.

ii.
```python
valid_mask = (iarea != -1) & (iarea != 7)
valid_indices = np.where(valid_mask)[0]
```

```python
if n_valid <= max_neurons:
    selected_indices = valid_indices
else:
    region_masks = neu_area_ID(iarea[valid_indices])
    n_sample = max(1, int(np.round(max_neurons * n_region / total_in_regions)))
    sampled = rng.choice(region_local_idx, size=n_sample, replace=False)
```

iii. The notes claim the exclusion filter matches the reference code; the added subsampling was justified pragmatically by file size and decoder tractability.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start by taking all frames belonging to a trial according to `ft_trInd`. The alignment window is the full available trial, not a fixed 32-frame corridor-only window.

ii.
```python
trial_mask = (ft_trInd == trial_idx)
frame_indices = np.where(trial_mask)[0]
trial_neural = spk[:, frame_indices].astype(np.float32)
```

iii. The AI’s reasoning was that trial start is corridor entry, and `ft_trInd` gives the trial-aligned frame sequence directly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses the native imaging frame rate, `FS = 3.17 Hz`, so bins are about 315.5 ms. No rebinning or resampling is applied.

ii.
```python
FS = 3.17  # Frame rate in Hz
TIME_BIN_MS = 1000.0 / FS  # ~315.5 ms
```

```python
'time_bin_size': TIME_BIN_MS,
```

iii. The notes explicitly say the frame rate is already the natural temporal grid for neural and behavioral variables.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundFr` and the integer frame indices kept for the trial.

ii.
```python
trial_sound_fr = sound_fr[trial_idx]
time_to_sound = (trial_sound_fr - frame_indices.astype(np.float64)) / FS
```

iii. The AI treated `SoundFr` as the cue time in frame units and converted frame differences to seconds with the nominal frame rate.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each kept frame in a trial, the AI computes `(SoundFr - frame_index) / FS`, producing a continuous time-to-cue value in seconds.

ii.
```python
trial_sound_fr = sound_fr[trial_idx]
time_to_sound = (trial_sound_fr - frame_indices.astype(np.float64)) / FS
trial_input[0, :] = time_to_sound.astype(np.float32)
```

iii. The notes/trajectory frame this as a straightforward time-difference calculation on the frame grid.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on exactly the same `frame_indices` used to slice neural activity for that trial.

ii.
```python
trial_neural = spk[:, frame_indices].astype(np.float32)
time_to_sound = (trial_sound_fr - frame_indices.astype(np.float64)) / FS
```

iii. The AI repeatedly states that all time-varying streams are aligned by shared frame indices.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the session metadata in `Imaging_Exp_info.npy`, using the `days` field when present and otherwise `sess#`.

ii.
```python
day_key = 'days' if 'days' in db else 'sess#'
day_val = db.get(day_key, 0)
'day_of_training': day_val,
```

iii. The notes describe this as using the explicit per-session training-day/session-number field from the experiment index.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The stored `day_of_training` value is converted to float and broadcast across every timepoint in every trial from that session.

ii.
```python
day_of_training = float(meta['day_of_training'])
trial_input[1, :] = day_of_training
```

iii. The AI did not derive this from session ordering; it assumed the metadata field already represented the correct training day.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `StartFr` and the integer frame indices used for the trial.

ii.
```python
trial_start = start_fr[trial_idx]
time_since_start = (frame_indices.astype(np.float64) - trial_start) / FS
```

iii. The AI treats `StartFr` as the trial-start time in frame units.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each kept frame in a trial, it computes `(frame_index - StartFr) / FS` and stores the result as `float32`.

ii.
```python
time_since_start = (frame_indices.astype(np.float64) - trial_start) / FS
trial_input[2, :] = time_since_start.astype(np.float32)
```

iii. The notes describe this as direct timing relative to corridor entry.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed on the same `frame_indices` used to extract the neural trace for the trial.

ii.
```python
trial_neural = spk[:, frame_indices].astype(np.float32)
time_since_start = (frame_indices.astype(np.float64) - trial_start) / FS
```

iii. The AI’s stated alignment rule is shared frame indexing across streams.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial `isRew` flag.

ii.
```python
is_rew = beh['isRew']
trial_input[3, :] = float(is_rew[trial_idx])
```

iii. The notes identify `isRew` as the reward-availability variable directly.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No transformation beyond converting to float and broadcasting the trial’s reward flag across all timepoints.

ii.
```python
trial_input[3, :] = float(is_rew[trial_idx])
```

iii. The AI treated reward availability as a per-trial constant input.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`, `UniqWalls`, and optionally session `stim_id` values when available.

ii.
```python
uniq_walls = beh['UniqWalls']
stim_id_map = beh.get('stim_id', None)
stim_name = wall_to_stim.get(wall_name[trial_idx], wall_name[trial_idx])
```

iii. The notes say the AI used `stim_id` when possible, but fell back to literal wall names for sessions with missing or `NaN` stimulus IDs.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI maps `UniqWalls` through `stim_id` to names like `circle1`, `leaf2`, `leaf1_swap1`, or leaves the wall name unchanged, then builds a global sorted vocabulary of all observed names and encodes each trial with that category at every timepoint.

ii.
```python
if stim_id_map is not None:
    for i, wall in enumerate(uniq_walls):
        if i < len(stim_id_map):
            sid_val = stim_id_map[i]
            if not np.isnan(sid_val):
                wall_to_stim[wall] = STIM_NAMES.get(int(sid_val), wall)
            else:
                wall_to_stim[wall] = wall
```

```python
stim_to_idx, stim_names_sorted = build_stimulus_mapping(all_output_raw)
output_arr[0, :] = stim_idx
```

iii. The AI justified this in the notes by pointing to swap sessions and `NaN` `stim_id` values, concluding that retaining the observed wall-name distinctions yielded 12 categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr` together with `LickTrind` to assign lick events to trials.

ii.
```python
lick_fr = beh['LickFr']
lick_trind = beh['LickTrind']
trial_lick_mask = (lick_trind == trial_idx)
trial_lick_frames = lick_fr[trial_lick_mask]
```

iii. The AI used `LickTrind` as a helper because it wanted to process licking independently within each trial.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the AI rounds each lick frame to the nearest integer frame, searches for that frame in the trial’s `frame_indices`, and sets a binary per-timepoint lick label.

ii.
```python
lick_binary = np.zeros(n_tp, dtype=np.int64)
for lf in trial_lick_frames:
    lf_int = int(np.round(lf))
    pos = np.searchsorted(frame_indices, lf_int)
    if pos < n_tp and frame_indices[pos] == lf_int:
        lick_binary[pos] = 1
    elif pos > 0 and frame_indices[pos-1] == lf_int:
        lick_binary[pos-1] = 1
```

iii. The notes justify the binary representation by noting that multiple lick events can fall within the same 315 ms frame.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned by building the binary lick vector on the same `frame_indices` used for the trial’s neural array.

ii.
```python
trial_neural = spk[:, frame_indices].astype(np.float32)
trial_output[1, :] = lick_binary
```

iii. The AI’s alignment rule is shared frame indexing across streams within each trial.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from `ft_Pos` at the frames assigned to each trial.

ii.
```python
ft_Pos = beh['ft_Pos'][:n_frames]
trial_pos = ft_Pos[frame_indices]
```

iii. The notes identify `ft_Pos` as the frame-level position variable in corridor units.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position values are clipped to `[0, 39.999]`, divided by 10, floored, and clipped to bins 0-3. This maps 1 m corridor bins but also forces gray-space frames into the final corridor bin.

ii.
```python
trial_pos = ft_Pos[frame_indices]
pos_clipped = np.clip(trial_pos, 0, 39.999)
pos_bin = np.floor(pos_clipped / 10.0).astype(np.int64)
pos_bin = np.clip(pos_bin, 0, 3)
```

iii. The trajectory shows the AI recognized the gray-space skew, but kept this rule so it could preserve full trials while still outputting only four position categories.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The thresholds are effectively four 10-unit bins: `[0,10)`, `[10,20)`, `[20,30)`, and `[30,40]`, with any higher values clipped into the last bin.

ii.
```python
pos_clipped = np.clip(trial_pos, 0, 39.999)
pos_bin = np.floor(pos_clipped / 10.0).astype(np.int64)
pos_bin = np.clip(pos_bin, 0, 3)
```

iii. The AI justified this as a direct implementation of four 1 m bins over the 4 m corridor.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is taken from the same `frame_indices` used for the neural trace of the trial.

ii.
```python
trial_neural = spk[:, frame_indices].astype(np.float32)
trial_pos = ft_Pos[frame_indices]
```

iii. The notes repeatedly say frame-level variables are aligned by selecting the same per-trial frames.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed` at the selected trial frames.

ii.
```python
ft_RunSpeed = beh['ft_RunSpeed'][:n_frames]
trial_speed = ft_RunSpeed[frame_indices]
```

iii. The notes identify `ft_RunSpeed` as the frame-level running-speed variable.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI first computes three global percentile thresholds over all frames from all selected sessions, then bins each trial’s speeds with `np.digitize`.

ii.
```python
all_speeds = []
for sid in session_ids:
    meta = session_meta[sid]
    beh = load_beh(sid, meta['beh_files'], meta['beh_keys'])
    speeds = beh['ft_RunSpeed']
    all_speeds.append(speeds)
all_speeds = np.concatenate(all_speeds)
quartiles = np.percentile(all_speeds, [25, 50, 75])
```

```python
speed_bin = np.digitize(trial_speed, speed_quartiles).astype(np.int64)
speed_bin = np.clip(speed_bin, 0, 3)
```

iii. The notes explicitly call this a “global quartile” choice and justify it from the task wording “25% of the data,” even though later notes acknowledge the resulting imbalance.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. It uses the three global percentile thresholds returned by `np.percentile(all_speeds, [25, 50, 75])`, and assigns bins with `np.digitize`, clipped to 0-3.

ii.
```python
quartiles = np.percentile(all_speeds, [25, 50, 75])
speed_bin = np.digitize(trial_speed, speed_quartiles).astype(np.int64)
speed_bin = np.clip(speed_bin, 0, 3)
```

iii. The AI intended these bins to represent dataset-wide quartiles rather than session-wise rank quartiles.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is sampled on the same `frame_indices` used for the neural data.

ii.
```python
trial_neural = spk[:, frame_indices].astype(np.float32)
trial_speed = ft_RunSpeed[frame_indices]
```

iii. The AI’s general alignment strategy is to use a shared frame subset for all streams in a trial.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI clips session data to the minimum of spike and behavioral frame counts, tries multiple behavior-file keys for sessions with suffix variants, asserts that retinotopy and spike neuron counts match, and skips trials left with fewer than 2 usable frames.

ii.
```python
n_frames = min(n_frames_spk, n_frames_beh)
for beh_file in beh_files:
    beh_path = os.path.join(DATA_ROOT, 'beh', beh_file)
    if session_id in beh_all:
        return beh_all[session_id]
    for key in beh_keys:
        if key in beh_all:
            return beh_all[key]
```

```python
assert len(iarea) == n_neurons_raw, f"Neuron count mismatch: iarea={len(iarea)}, spk={n_neurons_raw}"
```

iii. The notes call out these edge cases explicitly: swap-session key suffixes, `NaN` stimulus IDs, and spike/behavior frame-count mismatches.

## 12-a. What are the most time-consuming steps of the code?

i. The slowest work is loading and concatenating large spike files for every session. The AI also adds an extra full pass over all behavior files to compute global running-speed quartiles.

ii.
```python
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate(spk_data['spks'], axis=0)
```

```python
for sid in session_ids:
    meta = session_meta[sid]
    beh = load_beh(sid, meta['beh_files'], meta['beh_keys'])
    speeds = beh['ft_RunSpeed']
```

iii. The notes estimate that neural-data loading dominates wall time, while the quartile pre-pass is an additional cost introduced by the chosen speed discretization.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The biggest obvious candidates are the per-trial `np.where(ft_trInd == trial_idx)` scan, the per-lick loop that rounds and searches each lick separately, and the later pass that rebuilds outputs just to assign stimulus indices.

ii.
```python
for trial_idx in range(ntrials):
    trial_mask = (ft_trInd == trial_idx)
    frame_indices = np.where(trial_mask)[0]
```

```python
for lf in trial_lick_frames:
    lf_int = int(np.round(lf))
    pos = np.searchsorted(frame_indices, lf_int)
```

iii. The AI did not discuss vectorization directly, but these loops arise from its chosen per-trial/per-event processing strategy.

## 12-c. What processing does the code repeat multiple times?

i. Behavior loading is repeated: once globally inside `compute_running_speed_quartiles()` and again during `process_session()`. Stimulus encoding is also done in two stages: collect `(output_arr, stim_name)` tuples, then revisit every trial to fill in stimulus indices.

ii.
```python
speed_quartiles = compute_running_speed_quartiles(session_meta, session_ids)
beh = load_beh(sid, meta['beh_files'], meta['beh_keys'])
```

```python
output_trials.append((trial_output, stim_name))
for session_outputs in all_output_raw:
    for output_arr, stim_name in session_outputs:
        stim_idx = stim_to_idx[stim_name]
        output_arr[0, :] = stim_idx
```

iii. This repeated work follows from two AI decisions: global speed thresholds and deferred construction of the stimulus vocabulary.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes some values it never uses downstream, including `ft_CorrSpc`, `selected_idx`, and the placeholder stimulus row before it is overwritten later. It also stores extra session metadata fields that do not affect the converted dataset.

ii.
```python
spk, iarea_filtered, selected_idx = filter_and_subsample_neurons(spk, iarea)
ft_CorrSpc = beh['ft_CorrSpc'][:n_frames]
```

```python
trial_output = np.zeros((4, n_tp), dtype=np.int64)
trial_output[0, :] = -1  # placeholder
output_arr[0, :] = stim_idx
```

iii. There is no explicit justification for these discarded computations; they appear to be leftovers from the AI’s implementation strategy.
