# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three directories under `data/`: `beh/` (behavior), `spk/` (neural), and `retinotopy/` (brain regions). It first loads `Imaging_Exp_info.npy` as the master index, iterates over experiment types to collect unique sessions, then for each session loads the behavior file (`Beh_<exp_type>.npy`), neural data (`<session_id>_neural_data.npy`), and retinotopy (`<mname>_<datexp>_trans.npz`).

ii.
```python
info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
# ...
spk = np.concatenate(
    [nspk for nspk in np.load(os.path.join(root, fn), allow_pickle=True).item()['spks']], 0
)
# ...
dtrans = np.load(os.path.join(root, f'{mname}_{datexp}_trans.npz'), allow_pickle=True)
```

iii. The AI's CONVERSION_NOTES document the standard data loading pattern from the reference code's `load_spk` and `load_retino` functions.

## 1-b. How are the data split into subjects?

i. The mouse name (`mname`) is extracted from each entry in the experiment info. Subjects are collected as a list during processing and indexed by their order of appearance.

ii.
```python
sessions.append({'session_id': session_id, 'mname': entry['mname'], ...})
# ...
mname = result['mname']
if mname not in all_subjects: all_subjects.append(mname)
subject_idx = all_subjects.index(mname)
```

iii. The AI notes that there are 19 subjects across 89 sessions, consistent with the paper.

## 1-c. How are the data split into sessions?

i. A session is identified by the triple `(mname, datexp, blk)`. The AI deduplicates sessions that appear under multiple experiment types by keeping only the first occurrence of each session_id.

ii.
```python
session_id = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
if session_id not in session_map:
    session_map[session_id] = (exp_type, entry, beh_key)
```

iii. The AI documents 89 unique sessions matching the paper.

## 1-d. How are the data split into trials?

i. The AI splits trials using `ft_trInd` (trial index per frame) AND `ft_move > 0` (VR movement flag). Only frames where the VR is actively moving are included. Trials with fewer than 2 valid frames are dropped.

ii.
```python
valid_mask = (ft_trInd == trial_idx) & vr_move
valid_frame_indices = np.where(valid_mask)[0]
if len(valid_frame_indices) < 2:
    continue
```

iii. The AI's CONVERSION_NOTES state: "Frame filter: `ft_move > 0` only running/VR-moving frames" and references the paper's statement that analyses only considered timepoints during running.

## 1-e. How are trials filtered based on quality controls?

i. The AI only filters trials with fewer than 2 valid (VR-moving) frames. No trial length filtering or outlier removal is applied. All 38,110 trials are retained.

ii.
```python
if len(valid_frame_indices) < 2:
    continue
# ...
if valid_count < 2:
    return None
```

iii. The CONVERSION_NOTES state "Trial curation: All trials included, frames filtered for VR-moving." No mention of removing long trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `<session_id>_neural_data.npy`, which contains deconvolved calcium traces per imaging plane, concatenated across planes. Brain region comes from `iarea` in the retinotopy files.

ii.
```python
spk = np.concatenate(
    [nspk for nspk in np.load(os.path.join(root, fn), allow_pickle=True).item()['spks']], 0
)
iarea = load_retino(mname, datexp, root=os.path.join(DATA_ROOT, 'retinotopy'))
```

iii. Documented as "Suite2p deconvolved fluorescence traces."

## 2-b. How is the `neural` data processed?

i. Neural data is not further processed beyond filtering by brain area and selecting frames. Data is stored as float32. Trials have variable length.

ii.
```python
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
```

iii. The AI notes that all analyses are based on deconvolved fluorescence, so no dF/F or further deconvolution is needed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by excluding those with `iarea == -1` or `iarea == 7`. This removes neurons outside visual cortex.

ii.
```python
neuron_mask = (iarea != -1) & (iarea != 7)
spk = spk[neuron_mask]
```

iii. The CONVERSION_NOTES reference the pattern from `neu_area_ID` in the reference code: "Neuron filter: `(iarea != -1) & (iarea != 7)` excludes non-visual cortex neurons."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (corridor entry). For each trial, the frames where `ft_trInd == trial_idx` and `ft_move > 0` are extracted. The first such frame effectively becomes the start of the trial.

ii.
```python
valid_mask = (ft_trInd == trial_idx) & vr_move
valid_frame_indices = np.where(valid_mask)[0]
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
```

iii. The metadata specifies `temporal_alignment_event: 'Trial start (corridor entry)'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The imaging frame rate of 3.17 Hz is used directly, giving a time bin of ~315.5 ms.

ii.
```python
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE
```

iii. The AI correctly identifies the frame rate from the reference code and paper.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the frame at which the sound cue was played for each trial) and `ft` (frame timestamps in MATLAB datenum format).

ii.
```python
s_fr = sound_fr[trial_idx]
# ...
sound_time = ft[s_fr_int] + s_fr_frac * (ft[s_fr_int + 1] - ft[s_fr_int])
time_to_sound = ((ft_trial - sound_time) * DAYS_TO_SEC).astype(np.float32)
```

iii. The AI notes the sound frame is fractional and interpolates between adjacent frame times.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The sound frame `SoundFr[trial]` is fractional, so it is linearly interpolated between adjacent frame timestamps. The time to sound is computed as `frame_time - sound_time`, converted from days to seconds. Note this produces a sign OPPOSITE to "time TO sound": it is negative before the cue and positive after.

ii.
```python
s_fr_int = int(np.floor(s_fr))
s_fr_frac = s_fr - s_fr_int
if 0 <= s_fr_int < len(ft) - 1:
    sound_time = ft[s_fr_int] + s_fr_frac * (ft[s_fr_int + 1] - ft[s_fr_int])
time_to_sound = ((ft_trial - sound_time) * DAYS_TO_SEC).astype(np.float32)
```

iii. The AI handles edge cases where the sound frame might be NaN or out of range.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The time to sound cue is computed at the same frame indices used for the neural data (`valid_frame_indices`), so it is inherently aligned.

ii.
```python
ft_trial = ft[valid_frame_indices]
time_to_sound = ((ft_trial - sound_time) * DAYS_TO_SEC).astype(np.float32)
```

iii. All signals use the same frame indices.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From `datexp` (the date string in each session entry), parsed as a calendar date.

ii.
```python
def compute_day_of_training(sessions):
    # ...
    dates = [(idx, datetime(int(s['datexp'].split('_')[0]), int(s['datexp'].split('_')[1]), int(s['datexp'].split('_')[2]))) for idx, s in msessions]
    dates.sort(key=lambda x: x[1])
    first_date = dates[0][1]
    for idx, date in dates:
        day_of_training[idx] = (date - first_date).days
```

iii. The AI parses the date from the session ID to compute calendar days.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI computes day of training as the number of **calendar days** since the mouse's first session, NOT the number of recording sessions. This gives values up to 92 for some mice (i.e., 92 days between first and last recording).

ii.
```python
first_date = dates[0][1]
for idx, date in dates:
    day_of_training[idx] = (date - first_date).days
```

iii. The AI treats "day of training" literally as calendar days elapsed.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `ft` (frame timestamps) at the valid frame indices of each trial.

ii.
```python
ft_trial = ft[valid_frame_indices]
time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
```

iii. Uses actual frame timestamps converted from MATLAB datenum to seconds.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The time since trial start is computed as the difference between each frame's timestamp and the first valid (VR-moving) frame's timestamp of that trial, converted from days to seconds. This always starts at exactly 0.0.

ii.
```python
time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
```

iii. Uses the first moving frame as the reference rather than the corridor entry frame (`StartFr`).

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed at the same `valid_frame_indices` as neural data, so inherently aligned.

ii.
```python
ft_trial = ft[valid_frame_indices]
time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
```

iii. Same frame indices for all signals.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks whether each trial is in a rewarded corridor.

ii.
```python
is_rew = beh['isRew']
input_trial[3] = np.float32(1.0 if is_rew[trial_idx] else 0.0)
```

iii. Direct mapping from the behavior data.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No processing. The boolean `isRew` is cast to 1.0 (rewarded) or 0.0 (not rewarded) and broadcast across all time bins of the trial.

ii.
```python
input_trial[3] = np.float32(1.0 if is_rew[trial_idx] else 0.0)
```

iii. Straightforward mapping.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, which names the texture on the corridor walls for each trial.

ii.
```python
stim_idx = stim_to_idx[str(wall_name[trial_idx])]
```

iii. Uses the wall name directly.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI uses ALL 15 unique wall names (e.g., `circle1`, `circle2`, `leaf1_swap1`, etc.) as individual categories rather than grouping them into 4 base textures (circle, leaf, rock, wood). Each wall name is mapped to a unique integer index 0-14.

ii.
```python
all_stim_names = get_all_stim_names(all_sessions)  # Returns 15 unique names
stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
# ...
'output_values': [all_stim_names, ...]  # 15 categories
```

iii. The AI's CONVERSION_NOTES list the stimulus names under "Key Decisions": the mapping treats each unique wall name as its own category.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the frame numbers of each lick event in the session.

ii.
```python
lick_binary = build_lick_per_frame(n_frames_neural, beh.get('LickFr', np.array([])))
```

iii. Direct from the behavior data.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Each lick frame is rounded to the nearest integer frame, and a binary array is constructed where 1 indicates a lick at that frame.

ii.
```python
def build_lick_per_frame(n_frames, lick_fr):
    lick_binary = np.zeros(n_frames, dtype=np.int32)
    idx = np.round(lick_fr).astype(int)
    idx = idx[(idx >= 0) & (idx < n_frames)]
    lick_binary[idx] = 1
    return lick_binary
```

iii. The AI handles edge cases where lick frames might be out of bounds or empty.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick binary array is indexed at the same `valid_frame_indices` used for neural data.

ii.
```python
lick_trial = lick_binary[valid_frame_indices]
```

iii. Frame-level alignment via shared indices.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position at each imaging frame in decimeters.

ii.
```python
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
```

iii. The AI uses position data from the behavior files.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is binned by dividing by 15 decimeters (1.5 m), giving 4 bins over the full 6 m corridor (0-1.5m, 1.5-3m, 3-4.5m, 4.5-6m).

ii.
```python
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
```

iii. The AI's CONVERSION_NOTES state: "Position bins: 4 equal bins of 15dm over full 6m corridor."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The position is divided into 4 bins of 1.5m each: [0-1.5m], [1.5-3m], [3-4.5m], [4.5-6m], covering the full 6m corridor (4m texture + 2m grey).

ii.
```python
'output_values': [..., ['0-1.5m', '1.5-3m', '3-4.5m', '4.5-6m'], ...]
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
```

iii. The AI divides over the full 6m corridor rather than just the 4m texture.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is indexed at the same `valid_frame_indices` as neural data.

ii.
```python
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
```

iii. Same frame-level alignment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
```

iii. Directly from the behavior data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Global speed quartile boundaries are computed across all sessions (only from VR-moving frames), with subsampling to 5000 frames per session. Then `np.digitize` is used with these boundaries to assign each frame to one of 4 bins.

ii.
```python
def compute_global_speed_quartiles(sessions):
    # ...
    vr_move = beh['ft_move'] > 0
    speeds = beh['ft_RunSpeed'][vr_move]
    if len(speeds) > 5000:
        speeds = np.random.RandomState(42+i).choice(speeds, 5000, replace=False)
    all_speeds.append(speeds)
    quartiles = np.percentile(all_speeds, [25, 50, 75])
    return quartiles
```

iii. The AI computes global quartile thresholds rather than per-session quartiles.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Speed is discretized using `np.digitize` with global quartile boundaries [11.79, 24.43, 40.44]. Values below the 25th percentile get bin 0, between 25th-50th get bin 1, etc.

ii.
```python
speed_quartiles = compute_global_speed_quartiles(all_sessions)
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
```

iii. Uses value-based thresholds rather than rank-based assignment.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is indexed at the same `valid_frame_indices` as neural data.

ii.
```python
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
```

iii. Frame-level alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI truncates behavior arrays to the number of neural frames (`n_frames_neural = spk.shape[1]`). Lick frames beyond the neural recording are dropped. If `LickFr` is missing, an empty array is used. Trials with fewer than 2 valid frames are skipped. Sessions with fewer than 2 trials are skipped. NaN sound frames result in zero time-to-sound values.

ii.
```python
ft = beh['ft'][:n_frames_neural]
ft_trInd = beh['ft_trInd'][:n_frames_neural]
lick_binary = build_lick_per_frame(n_frames_neural, beh.get('LickFr', np.array([])))
# ...
if np.isnan(s_fr):
    time_to_sound = np.zeros(n_t, dtype=np.float32)
```

iii. The AI handles edge cases with defensive checks.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the neural spike files, which are very large. The AI reports 73.1 minutes for full conversion. Individual session load times are 4-15 seconds each.

ii.
```python
spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
```

iii. Documented in CONVERSION_NOTES: processing time is dominated by I/O.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop within each session iterates over all trials, computing masks and extracting frames one at a time. The trial frame finding (`(ft_trInd == trial_idx) & vr_move`) scans the full frame array once per trial.

ii.
```python
for trial_idx in range(ntrials):
    valid_mask = (ft_trInd == trial_idx) & vr_move
    valid_frame_indices = np.where(valid_mask)[0]
```

iii. Could be vectorized by grouping all frames by trial in one pass using `np.unique` or similar.

## 12-c. What processing does the code repeat multiple times?

i. The behavior files are loaded multiple times: once in `compute_global_speed_quartiles`, once in `get_all_stim_names`, and once per session in `process_session`. Each session's behavior file is loaded individually rather than shared across sessions in the same file.

ii.
```python
# In compute_global_speed_quartiles:
beh_cache[exp_type] = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), ...)
# In get_all_stim_names:
beh_cache[exp_type] = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), ...)
# In process_session:
beh_all = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), ...)
```

iii. The behavior loading is repeated 3 times for preprocessing + once per session.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores `exp_type` metadata per session and various extra metadata fields (deconvolution_decay_s, corridor_length_m, etc.) that are not used by the decoder. The `stim_to_idx` mapping is stored in metadata. Temp files are written and then re-read during assembly.

ii.
```python
'metadata': {
    'deconvolution_decay_s': 0.75,
    'neuron_filter': 'Excluded neurons outside visual cortex ...',
    'frame_filter': 'Only VR-moving frames (ft_move > 0)',
    'stim_to_idx': stim_to_idx,
    ...
}
```

iii. Extra metadata is informational but not used by downstream decoder.
