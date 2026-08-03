# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads `Imaging_Exp_info.npy` as the master index, iterating over experiment types and session entries to build a `session_meta` dictionary. For each session, it loads neural data from `spk/<session_id>_neural_data.npy`, behavioral data from `Beh_<exp_type>.npy`, and retinotopy from `retinotopy/<mname>_<datexp>_trans.npz`. Behavioral data is re-loaded from disk for each session individually (once in `compute_running_speed_quartiles` and again in `process_session`), rather than loading each behavior file once for all sessions it contains.

ii.
```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()
# ...
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate(spk_data['spks'], axis=0)
# ...
beh_all = np.load(beh_path, allow_pickle=True).item()
# ...
dtrans = np.load(ret_path, allow_pickle=True)
return dtrans['iarea']
```

iii. The AI documented in CONVERSION_NOTES.md that it follows the same loading structure as the reference code, reading the experiment info first, then loading per-session neural, behavioral, and retinotopy data.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `mname` field in the experiment info. The AI builds a sorted set of unique subject names from all session metadata and creates an index mapping.

ii.
```python
subjects_set = set()
for sid in session_ids:
    subjects_set.add(session_meta[sid]['mname'])
subjects_list = sorted(subjects_set)
subject_to_idx = {name: idx for idx, name in enumerate(subjects_list)}
```

iii. The AI noted that 19 mice are identified, matching the reference paper.

## 1-c. How are the data split into sessions?

i. A session is identified by the composite key `mname_datexp_blk`. When the same session appears under multiple experiment types, only the first occurrence is kept via a deduplication check.

ii.
```python
sid = f"{db['mname']}_{db['datexp']}_{db['blk']}"
if sid not in session_meta:
    session_meta[sid] = { ... }
```

iii. The AI documented that 89 unique sessions are identified from 142 total entries across experiment types, matching the reference.

## 1-d. How are the data split into trials?

i. Trials are split using `ft_trInd`, which labels each frame with its trial index. For each trial, the AI finds all frames where `ft_trInd == trial_idx`. Unlike the reference, the AI does NOT filter to corridor space frames using `ft_CorrSpc`, and does NOT use a fixed trial length (N_FRAMES=32). Trials have variable lengths.

ii.
```python
for trial_idx in range(ntrials):
    trial_mask = (ft_trInd == trial_idx)
    frame_indices = np.where(trial_mask)[0]
    if len(frame_indices) < 2:
        skipped += 1
        continue
    frame_indices = frame_indices[frame_indices < n_frames_spk]
```

iii. The AI documented using `ft_trInd` for frame-to-trial mapping, consistent with the reference approach for identifying trial boundaries, but omitted the corridor space filtering.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 frames are skipped. No other quality filtering is applied. This is similar to the reference which drops trials with no corridor frames, but the reference uses a different frame selection (corridor space only).

ii.
```python
if len(frame_indices) < 2:
    skipped += 1
    continue
```

iii. The AI noted this in CONVERSION_NOTES.md under trial curation rules: "Include all trials. Skip trials with < 2 frames."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the neural data files (`spk/<session_id>_neural_data.npy`), which contains deconvolved calcium traces as a list of arrays per imaging plane. These are concatenated into a single neurons-by-frames array. The visual area of each neuron comes from `iarea` in the retinotopy files.

ii.
```python
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate(spk_data['spks'], axis=0)
# ...
iarea = load_retinotopy(mname, datexp)
```

iii. The AI correctly identified these as deconvolved fluorescence traces from Suite2p processing.

## 2-b. How is the `neural` data processed?

i. The neural data is stored as float32 (reference uses float16). The AI applies neuron subsampling to a maximum of 2000 neurons per session, stratified by brain region, which the reference does NOT do. No other processing is applied.

ii.
```python
trial_neural = spk[:, frame_indices].astype(np.float32)
```

iii. The AI justified the subsampling as an optimization because "decoder uses PCA to 100 components and SVD with max 2000 neurons."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by visual area: neurons with `iarea == -1` or `iarea == 7` are excluded. Additionally, the AI subsamples to max 2000 neurons per session, stratified by brain region proportions.

ii.
```python
def filter_and_subsample_neurons(spk, iarea, max_neurons=MAX_NEURONS, rng=None):
    valid_mask = (iarea != -1) & (iarea != 7)
    valid_indices = np.where(valid_mask)[0]
    if n_valid <= max_neurons:
        selected_indices = valid_indices
    else:
        # stratified subsampling by brain region
        for region in BRAIN_REGIONS:
            n_sample = max(1, int(np.round(max_neurons * n_region / total_in_regions)))
            sampled = rng.choice(region_local_idx, size=n_sample, replace=False)
```

iii. The AI noted the filtering matches `(arid!=-1) & (arid != 7)` from the reference code. The subsampling was justified as an efficiency measure.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI takes all frames belonging to a trial (identified by `ft_trInd`), but does NOT filter to corridor space (`ft_CorrSpc`). Trials are of variable length with no fixed window or padding. The reference aligns to corridor entry by taking only `ft_CorrSpc` frames and padding/truncating to N_FRAMES=32.

ii.
```python
trial_mask = (ft_trInd == trial_idx)
frame_indices = np.where(trial_mask)[0]
n_tp = len(frame_indices)
trial_neural = spk[:, frame_indices].astype(np.float32)
```

iii. The AI documented "Trial alignment: Align to corridor entry (trial start). Use ft_trInd for frame-to-trial mapping."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging frame rate of 3.17 Hz is used, giving ~315.5 ms time bins. No rebinning is applied. This matches the reference.

ii.
```python
FS = 3.17  # Frame rate in Hz
TIME_BIN_MS = 1000.0 / FS  # ~315.5 ms
```

iii. The AI correctly identified the frame rate and documented it matches the reference.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the sound cue frame for each trial) and the frame indices. The reference also uses `ft` (frame timestamps) for proper time conversion.

ii.
```python
trial_sound_fr = sound_fr[trial_idx]
time_to_sound = (trial_sound_fr - frame_indices.astype(np.float64)) / FS
```

iii. The AI computes this as a frame difference divided by frame rate, rather than interpolating onto a proper time axis as the reference does.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The AI computes `(SoundFr - frame_index) / FS` for each frame in the trial. This converts the frame difference to seconds by dividing by the frame rate. The reference instead interpolates `SoundFr` onto the actual timestamp axis derived from `ft` (which are MATLAB datenums), computing `cue_time - frame_time` in proper seconds.

ii.
```python
time_to_sound = (trial_sound_fr - frame_indices.astype(np.float64)) / FS
trial_input[0, :] = time_to_sound.astype(np.float32)
```

iii. The sign convention is the same as the reference (positive before the cue, negative after).

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the same `frame_indices` used for the neural data of that trial, so it is naturally aligned.

ii.
```python
frame_indices = np.where(trial_mask)[0]
# ... used for both neural and input
time_to_sound = (trial_sound_fr - frame_indices.astype(np.float64)) / FS
```

iii. Aligned via shared frame indices.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the `days` or `sess#` field in the experiment info entries. The reference derives it by counting the ordered sessions per mouse.

ii.
```python
day_key = 'days' if 'days' in db else 'sess#'
day_val = db.get(day_key, 0)
session_meta[sid]['day_of_training'] = day_val
```

iii. The AI uses the `days` field from experiment info directly, rather than computing it from session ordering.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI reads the value directly from the experiment info (`days` or `sess#` field) and broadcasts it as a constant across all timepoints in a trial. The reference counts sessions in order per mouse (0-indexed).

ii.
```python
day_of_training = float(meta['day_of_training'])
trial_input[1, :] = day_of_training
```

iii. The AI documented "Day of training range: 0-15". The reference's approach produces values 0-7 (counting recording sessions, not calendar days).

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `StartFr` (the corridor entry frame for each trial) and the frame indices.

ii.
```python
trial_start = start_fr[trial_idx]
time_since_start = (frame_indices.astype(np.float64) - trial_start) / FS
```

iii. Documented in CONVERSION_NOTES.md variable mapping.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The AI computes `(frame_index - StartFr) / FS` to get time in seconds since trial start. The reference interpolates `StartFr` onto the proper timestamp axis from `ft`.

ii.
```python
time_since_start = (frame_indices.astype(np.float64) - trial_start) / FS
trial_input[2, :] = time_since_start.astype(np.float32)
```

iii. Same approach as time_to_sound_cue: frame difference divided by frame rate instead of proper timestamp interpolation.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed from the same `frame_indices` used for neural data, so naturally aligned.

ii.
```python
time_since_start = (frame_indices.astype(np.float64) - trial_start) / FS
```

iii. Aligned via shared frame indices.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks whether each trial is in a rewarded corridor.

ii.
```python
is_rew = beh['isRew']
trial_input[3, :] = float(is_rew[trial_idx])
```

iii. Matches the reference approach.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew` is cast to float (0.0 or 1.0) and broadcast across all timepoints. This matches the reference.

ii.
```python
trial_input[3, :] = float(is_rew[trial_idx])
```

iii. No additional processing needed, same as reference.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The AI uses `stim_id` from experiment info to map `WallName` to stimulus names like 'circle1', 'leaf1', etc. The reference uses `WallName` directly, mapping through a `TEXTURE` dictionary to 4 broad categories ('circle', 'leaf', 'rock', 'wood').

ii.
```python
STIM_NAMES = {0: 'circle1', 1: 'circle2', 2: 'leaf1', 3: 'leaf2', 4: 'leaf3', 5: 'leaf1_swap1', 6: 'leaf1_swap2'}
# ...
wall_to_stim[wall] = STIM_NAMES.get(int(sid_val), wall)
stim_name = wall_to_stim.get(wall_name[trial_idx], wall_name[trial_idx])
```

iii. The AI documented using `stim_id` mapping from the reference code's data_process_script.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI produces 12 stimulus categories (individual wall names like 'circle1', 'leaf1', 'rock1', etc.) rather than the 4 broad texture categories used by the reference ('circle', 'leaf', 'rock', 'wood'). The stimulus index is broadcast across all timepoints as a per-trial constant.

ii.
```python
stim_to_idx, stim_names_sorted = build_stimulus_mapping(all_output_raw)
# Results in 12 categories
stim_idx = stim_to_idx[stim_name]
output_arr[0, :] = stim_idx
```

iii. The AI noted "12 categories" in CONVERSION_NOTES.md. The reference groups wall textures into 4 broad categories, which better aligns with the paper's experimental design where the key discrimination is between texture types, not individual crops.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (frame numbers of licks) and `LickTrind` (trial index of each lick). The reference uses only `LickFr`.

ii.
```python
lick_fr = beh['LickFr']
lick_trind = beh['LickTrind']
```

iii. The AI uses per-trial lick matching via `LickTrind`, while the reference creates a session-wide binary lick array.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The AI matches licks to trials using `LickTrind`, then uses `searchsorted` to find matching frames. The reference creates a session-wide licking array by marking frames at `LickFr` positions as 1, then slices it per trial.

ii.
```python
trial_lick_mask = (lick_trind == trial_idx)
trial_lick_frames = lick_fr[trial_lick_mask]
lick_binary = np.zeros(n_tp, dtype=np.int64)
for lf in trial_lick_frames:
    lf_int = int(np.round(lf))
    pos = np.searchsorted(frame_indices, lf_int)
    if pos < n_tp and frame_indices[pos] == lf_int:
        lick_binary[pos] = 1
```

iii. The AI's approach is more complex and may miss licks that don't exactly match frame indices due to the `searchsorted` and exact match requirement.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick frames are matched to the same `frame_indices` used for neural data, so alignment is maintained.

ii.
```python
pos = np.searchsorted(frame_indices, lf_int)
if pos < n_tp and frame_indices[pos] == lf_int:
    lick_binary[pos] = 1
```

iii. Aligned via shared frame indices.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position of the mouse at each imaging frame (in decimeters, 0-60).

ii.
```python
ft_Pos = beh['ft_Pos'][:n_frames]
trial_pos = ft_Pos[frame_indices]
```

iii. Same source variable as the reference.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is clipped to [0, 39.999] and divided by 10 to get 4 bins of 1 meter each. This matches the reference approach.

ii.
```python
pos_clipped = np.clip(trial_pos, 0, 39.999)
pos_bin = np.floor(pos_clipped / 10.0).astype(np.int64)
pos_bin = np.clip(pos_bin, 0, 3)
```

iii. The binning logic is equivalent to the reference's `np.clip(ft_Pos // 10, 0, 3)`.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four equal 1-meter bins: 0-1m (bin 0), 1-2m (bin 1), 2-3m (bin 2), 3-4m (bin 3). Positions above 4m are clipped to bin 3. The reference adds a 'none' category for padding; the AI does not because trials are variable length.

ii.
```python
pos_bin_names = ['0-1m', '1-2m', '2-3m', '3-4m']
```

iii. The bin definitions match the reference.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is indexed at the same `frame_indices` used for neural data.

ii.
```python
trial_pos = ft_Pos[frame_indices]
```

iii. Aligned via shared frame indices.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
ft_RunSpeed = beh['ft_RunSpeed'][:n_frames]
trial_speed = ft_RunSpeed[frame_indices]
```

iii. Same source as the reference.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI discretizes speed into 4 bins using global percentile-based quartiles (25th, 50th, 75th percentile thresholds computed across all sessions). The reference uses per-session rank-based quartiles over kept frames, which guarantees exactly 25% of frames in each bin.

ii.
```python
def compute_running_speed_quartiles(session_meta, session_ids):
    all_speeds = np.concatenate(all_speeds)
    quartiles = np.percentile(all_speeds, [25, 50, 75])
    return quartiles
# ...
speed_bin = np.digitize(trial_speed, speed_quartiles).astype(np.int64)
```

iii. The AI documented using "4 global quartile bins" computed across all sessions. The reference computes quartiles per session using rank-ordering.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The AI uses `np.digitize` with global percentile boundaries to assign frames to 4 speed bins. The reference uses rank-based splitting per session, which handles ties (e.g., many frames at zero speed) more gracefully.

ii.
```python
speed_bin = np.digitize(trial_speed, speed_quartiles).astype(np.int64)
speed_bin = np.clip(speed_bin, 0, 3)
```

iii. Global percentile thresholds vs per-session rank-based quartiles can produce different distributions, especially when speed distributions vary across sessions.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is indexed at the same `frame_indices` used for neural data.

ii.
```python
trial_speed = ft_RunSpeed[frame_indices]
```

iii. Aligned via shared frame indices.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: frame count mismatch between neural and behavioral data (uses minimum), trials with fewer than 2 frames (skipped), `NaN` stim_id values (falls back to wall name), and stimtype suffix variations in behavioral data keys.

ii.
```python
n_frames = min(n_frames_spk, n_frames_beh)
if len(frame_indices) < 2:
    skipped += 1
    continue
```

iii. Documented in CONVERSION_NOTES.md under edge cases. The reference handles the same frame count mismatch by cutting to the number of imaged frames.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the neural spike files, which are large (total ~405 GB). The AI also re-loads behavioral data twice (once for speed quartile computation, once for session processing), adding unnecessary I/O.

ii.
```python
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate(spk_data['spks'], axis=0)
```

iii. The AI reported ~18 minutes for full conversion.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The lick processing loop iterates over individual lick frames using Python for-loop and `searchsorted`, which could be vectorized. The reference vectorizes this by creating a session-wide binary licking array in one operation.

ii.
```python
for lf in trial_lick_frames:
    lf_int = int(np.round(lf))
    pos = np.searchsorted(frame_indices, lf_int)
    if pos < n_tp and frame_indices[pos] == lf_int:
        lick_binary[pos] = 1
```

iii. Not documented by the AI.

## 12-c. What processing does the code repeat multiple times?

i. Behavioral data is loaded twice: once in `compute_running_speed_quartiles()` and again in `process_session()`. The reference loads each behavior file once and reuses it for all sessions in that file.

ii.
```python
# First pass - in compute_running_speed_quartiles:
beh = load_beh(sid, meta['beh_files'], meta['beh_keys'])
speeds = beh['ft_RunSpeed']

# Second pass - in process_session:
beh = load_beh(sid, meta['beh_files'], meta['beh_keys'])
```

iii. Not documented by the AI.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `ft_CorrSpc` but never uses it. The neuron subsampling (to max 2000) discards neural data that the reference retains.

ii.
```python
ft_CorrSpc = beh['ft_CorrSpc'][:n_frames]  # loaded but never used
```

iii. Not documented by the AI.
