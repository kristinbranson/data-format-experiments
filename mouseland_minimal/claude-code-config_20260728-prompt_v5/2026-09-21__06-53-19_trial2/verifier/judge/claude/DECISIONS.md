# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three directories: `beh/` for behavior, `spk/` for deconvolved calcium traces, and `retinotopy/` for visual area assignments. It first reads `Imaging_Exp_info.npy` as the master index, then iterates over experiment types, loading each `Beh_<exp_type>.npy` file. For each unique session, it loads the spike file and retinotopy data. Sessions are deduplicated by their key (`mname_datexp_blk`).

ii.
```python
exp_info = np.load(os.path.join(BEH_ROOT, 'Imaging_Exp_info.npy'), allow_pickle=True).item()

for exp_type in exp_info:
    beh_file = os.path.join(BEH_ROOT, f'Beh_{exp_type}.npy')
    Beh = np.load(beh_file, allow_pickle=True).item()

    for ndb in exp_info[exp_type]:
        session_key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
        if session_key in sessions:
            continue
        ...
        sessions[session_key] = {
            'ndb': ndb, 'beh': Beh[beh_key], 'exp_type': exp_type,
        }
```

iii. The agent noted: "89 unique sessions, but some sessions appear in multiple experiment types because the same recording was analyzed for different stimuli. But the neural data and trials are the same." It chose to use each recording only once, deduplicating by session key.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mname` from the experiment info entries. As sessions are processed, each unique mouse name is added to a list and assigned an index. The `subject_idx` array maps each session to its subject.

ii.
```python
if mname not in subject_to_idx:
    subject_to_idx[mname] = len(subjects_list)
    subjects_list.append(mname)
session_subject_idx.append(subject_to_idx[mname])
```

iii. The mouse name is directly available from the data index entries. No derivation is needed.

## 1-c. How are the data split into sessions?

i. A session is defined by the triple `(mname, datexp, blk)`, concatenated into a key string. When the same recording appears under multiple experiment types, only the first occurrence is kept. Sessions are processed in sorted order of session keys.

ii.
```python
session_key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
if session_key in sessions:
    continue
```

iii. The agent noted: "For deduplication, I'll just use the first behavior entry I find for each unique recording."

## 1-d. How are the data split into trials?

i. Trials are defined by `ft_trInd` (trial index per frame) and `ft_CorrSpc` (whether the frame is in the corridor/texture space). For each trial index from 0 to `ntrials-1`, the frames where both conditions match are selected. This gives corridor-only frames per trial.

ii.
```python
for trial_idx in range(ntrials):
    mask = (ft_trInd == trial_idx) & ft_CorrSpc
    frame_indices = np.where(mask)[0]

    if len(frame_indices) < 2:
        continue
```

iii. The agent reasoned: "For the decoder, I'll use ft_trInd to find each trial's frames and ft_CorrSpc to isolate just the texture corridor portion... Since the four 1m position bins only cover the texture corridor itself, I'll define each trial as just the corridor traversal, excluding gray space."

## 1-e. How are trials filtered based on quality controls?

i. The only filter is that trials with fewer than 2 frames are skipped. There is no upper bound on trial length. Sessions with fewer than 2 valid trials are also skipped.

ii.
```python
if len(frame_indices) < 2:
    continue
```
```python
if len(neural_trials) < 2:
    return None
```

iii. The agent noted that some trials have 5607 frames (~29 minutes), suggesting the mouse wasn't running, but did not implement any outlier filtering. Its only stated filter was the minimum of 2 frames.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the spike files (`spk/<session_id>_neural_data.npy`), which contains a list of arrays per imaging plane. These are concatenated across planes. The visual area of each neuron comes from `iarea` in the retinotopy files.

ii.
```python
spk_data = np.load(os.path.join(SPK_ROOT, fn), allow_pickle=True).item()
spk = np.concatenate(spk_data['spks'], axis=0)
```
```python
ret = np.load(os.path.join(RET_ROOT, fn))
brain_reg_idx = map_brain_regions(ret['iarea'])
```

iii. The agent noted: "Neural data: Concatenate all planes' spks. Use deconvolved fluorescence traces as stated in the paper."

## 2-b. How is the `neural` data processed?

i. The neural data is not further processed beyond concatenation across planes. It is stored as float32 (not float16 as in the reference). Trials are variable length and are not padded.

ii.
```python
neural_trial = spk[:, frame_indices].astype(np.float32)
```

iii. The agent reasoned the data already contains deconvolved traces, so no additional processing is needed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are filtered out. All neurons are kept, including those outside the four identified visual areas (V1, mHV, lHV, aHV). Neurons with `iarea` values of -1 or 7 are assigned to an "unassigned" brain region category rather than being dropped.

ii.
```python
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'unassigned']
BRAIN_REGION_MAP = {
    8: 0,                  # V1
    0: 1, 1: 1, 2: 1, 9: 1,  # mHV
    5: 2, 6: 2,            # lHV
    3: 3, 4: 3,            # aHV
    -1: 4, 7: 4,           # unassigned
}
```

iii. The agent explicitly decided: "For the decoder, I'm deciding to include all neurons rather than filtering by area, since different analysis functions in the codebase handle this inconsistently anyway -- the decoder can learn which neurons carry useful signal on its own."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to corridor entry (trial start). For each trial, the frames where `ft_trInd == trial_idx` and `ft_CorrSpc` is true are selected. The first such frame corresponds to corridor entry. Trials are variable length.

ii.
```python
mask = (ft_trInd == trial_idx) & ft_CorrSpc
frame_indices = np.where(mask)[0]
neural_trial = spk[:, frame_indices].astype(np.float32)
```

iii. The agent noted: "Trial start is described as corridor entry, and StartFr marks the neural frame when the animal enters each corridor, so I'll use that as the alignment point."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The native imaging frame rate is used. The time bin size is computed from the median inter-frame interval of the first session's `ft` timestamps, yielding approximately 315 ms.

ii.
```python
ft = sessions[first_key]['beh']['ft']
dt_seconds = float(np.median(np.diff(ft)) * 86400)
dt_ms = dt_seconds * 1000
```

iii. The agent used the actual frame timestamps rather than the nominal 3.17 Hz rate to compute the bin size.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr`, the frame index at which the sound cue was played in each trial, and the current frame indices of the trial.

ii.
```python
time_to_sound = ((SoundFr[trial_idx] - frame_indices) * dt_seconds).astype(np.float32)
```

iii. The agent noted: "For sound cue timing, I'll rely on the floating point SoundFr frame indices directly rather than rounding them."

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The time to sound cue is computed as `(SoundFr[trial] - frame_index) * dt_seconds`, where `dt_seconds` is the median inter-frame interval. This gives positive values before the cue and negative after, consistent with the name "time TO sound cue."

ii.
```python
time_to_sound = ((SoundFr[trial_idx] - frame_indices) * dt_seconds).astype(np.float32)
```

iii. The agent uses a constant `dt_seconds` multiplier rather than interpolating onto the actual frame time axis.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It uses the same `frame_indices` array as the neural data for that trial, so it is inherently aligned.

ii.
```python
frame_indices = np.where(mask)[0]
time_to_sound = ((SoundFr[trial_idx] - frame_indices) * dt_seconds).astype(np.float32)
neural_trial = spk[:, frame_indices].astype(np.float32)
```

iii. All data streams use the same frame indices per trial.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the `datexp` field (date string in YYYY_MM_DD format) of each session's experiment info entry.

ii.
```python
def compute_days_of_training(sessions):
    mouse_dates = {}
    for session_key, info in sessions.items():
        mname = info['ndb']['mname']
        dt = parse_date(info['ndb']['datexp'])
        mouse_dates.setdefault(mname, []).append(dt)
    mouse_first_date = {m: min(dates) for m, dates in mouse_dates.items()}
    return {
        sk: (parse_date(info['ndb']['datexp']) - mouse_first_date[info['ndb']['mname']]).days
        for sk, info in sessions.items()
    }
```

iii. The agent reasoned: "For 'day of training,' calendar days between sessions computed from the sorted recording dates for each mouse seems most accurate."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Calendar days since each mouse's first recording date. For each mouse, the earliest recording date is found, and each session's day of training is the number of calendar days elapsed from that first date. This differs from the reference, which counts the ordinal session number (0, 1, 2, ...).

ii.
```python
mouse_first_date = {m: min(dates) for m, dates in mouse_dates.items()}
return {
    sk: (parse_date(info['ndb']['datexp']) - mouse_first_date[info['ndb']['mname']]).days
    for sk, info in sessions.items()
}
```

iii. The agent chose calendar days rather than session count, noting it "seems most accurate."

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From the `frame_indices` of each trial (derived from `ft_trInd` and `ft_CorrSpc`), using the first frame index as the trial start.

ii.
```python
time_since_start = ((frame_indices - frame_indices[0]) * dt_seconds).astype(np.float32)
```

iii. The agent uses the first corridor frame as the start reference.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Time since trial start is computed as `(frame_index - first_frame_index) * dt_seconds`. This assumes uniform frame spacing and uses a constant `dt_seconds` rather than actual frame timestamps. It starts at 0 for the first corridor frame.

ii.
```python
time_since_start = ((frame_indices - frame_indices[0]) * dt_seconds).astype(np.float32)
```

iii. The agent uses frame index differences multiplied by a constant dt, rather than interpolating the actual `StartFr` onto the frame time axis.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It uses the same `frame_indices` array as the neural data, so alignment is automatic.

ii.
```python
frame_indices = np.where(mask)[0]
time_since_start = ((frame_indices - frame_indices[0]) * dt_seconds).astype(np.float32)
```

iii. All data streams share the same frame index array per trial.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks trials in the rewarded corridor.

ii.
```python
reward_avail = np.full(T, float(isRew[trial_idx]), dtype=np.float32)
```

iii. Directly available per trial from behavior data.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew` value is cast to float (0.0 or 1.0) and broadcast across all timepoints of the trial.

ii.
```python
reward_avail = np.full(T, float(isRew[trial_idx]), dtype=np.float32)
```

iii. No processing needed beyond type conversion.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, the texture name for each trial.

ii.
```python
stim_idx = STIM_TO_IDX[str(WallName[trial_idx])]
stim_out = np.full(T, stim_idx, dtype=np.int64)
```

iii. The agent noted: "For the 'visual stimulus category' output, I should use all these as possible values."

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each of the 15 unique `WallName` strings is mapped to an index in a sorted list of all 15 stimulus names. Unlike the reference, which groups the 15 textures into 4 base categories (circle, leaf, rock, wood), the AI keeps all 15 as separate categories.

ii.
```python
ALL_STIMULI = sorted([
    'circle1', 'circle2', 'circle3',
    'leaf1', 'leaf1_swap1', 'leaf1_swap2', 'leaf2', 'leaf3',
    'rock1', 'rock2',
    'wood1', 'wood1_swap1', 'wood1_swap2', 'wood2', 'wood5',
])
STIM_TO_IDX = {s: i for i, s in enumerate(ALL_STIMULI)}
```

iii. The agent treated each individual texture variant as its own category rather than grouping by base texture type.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (frame indices of licks) and `LickTrind` (trial index of each lick).

ii.
```python
def compute_lick_per_frame(beh, trial_idx, frame_indices):
    lick_mask = beh['LickTrind'] == trial_idx
    ...
    lick_frames_set = set(np.round(beh['LickFr'][lick_mask]).astype(int))
    return np.array([1.0 if f in lick_frames_set else 0.0 for f in frame_indices], dtype=np.float32)
```

iii. The agent noted: "For the licking output, I need to decide per imaging frame whether a lick occurred - since LickFr values may fall between frames, I'll assign each lick to its nearest frame index."

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, licks are filtered using `LickTrind == trial_idx`. The fractional `LickFr` values are rounded to the nearest integer frame. A binary array is created where frames with at least one lick are 1, others are 0.

ii.
```python
lick_mask = beh['LickTrind'] == trial_idx
lick_frames_set = set(np.round(beh['LickFr'][lick_mask]).astype(int))
return np.array([1.0 if f in lick_frames_set else 0.0 for f in frame_indices], dtype=np.float32)
```

iii. The agent uses per-trial lick filtering via `LickTrind` and rounds `LickFr` rather than truncating it.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick binary array is computed only for the `frame_indices` of the trial, which are the same frames used for neural data.

ii.
```python
return np.array([1.0 if f in lick_frames_set else 0.0 for f in frame_indices], dtype=np.float32)
```

iii. Alignment is through shared frame indices.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position of the mouse at each imaging frame, in decimeters.

ii.
```python
pos_out = np.clip(ft_Pos[frame_indices] // 10, 0, 3).astype(np.int64)
```

iii. Directly from the per-frame position data.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The position in decimeters is integer-divided by 10 to get meters, then clipped to [0, 3], giving four 1-meter bins.

ii.
```python
pos_out = np.clip(ft_Pos[frame_indices] // 10, 0, 3).astype(np.int64)
```

iii. The same approach as the reference solution.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Floor division by 10 (decimeters to meters), clipped to range [0, 3]. Bins: 0-1m, 1-2m, 2-3m, 3-4m.

ii.
```python
pos_out = np.clip(ft_Pos[frame_indices] // 10, 0, 3).astype(np.int64)
```

iii. Four equal 1-meter bins as specified in the instructions.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is indexed by the same `frame_indices` used for neural data, so alignment is automatic.

ii.
```python
pos_out = np.clip(ft_Pos[frame_indices] // 10, 0, 3).astype(np.int64)
```

iii. All data streams share the same frame indices.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
speed_out = np.digitize(ft_RunSpeed[frame_indices], speed_quartiles).astype(np.int64)
```

iii. Directly from the per-frame running speed data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed quartiles are computed globally across ALL corridor frames from ALL sessions using `np.percentile` at the 25th, 50th, and 75th percentiles. Then `np.digitize` assigns each frame's speed to one of 4 bins based on these thresholds. This is a percentile-threshold approach computed globally, unlike the reference which uses a rank-based approach computed per session.

ii.
```python
def compute_speed_quartiles(sessions):
    all_speeds = []
    for session_key in sorted(sessions.keys()):
        ...
        corridor_mask = ft_CorrSpc & ~np.isnan(ft_trInd)
        all_speeds.append(ft_RunSpeed[corridor_mask])
    all_speeds = np.concatenate(all_speeds)
    q25, q50, q75 = np.percentile(all_speeds, [25, 50, 75])
    return np.array([q25, q50, q75])
```
```python
speed_out = np.digitize(ft_RunSpeed[frame_indices], speed_quartiles).astype(np.int64)
```

iii. The agent noted: "Running speed quartiles computed across all corridor frames in all sessions."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Global percentile thresholds (25th, 50th, 75th) are computed from all corridor frames. `np.digitize` maps each speed to bins 0-3. This differs from the reference's rank-based approach which guarantees exactly 25% of frames in each bin per session.

ii.
```python
q25, q50, q75 = np.percentile(all_speeds, [25, 50, 75])
speed_out = np.digitize(ft_RunSpeed[frame_indices], speed_quartiles).astype(np.int64)
```

iii. The percentile approach does not guarantee equal bin sizes per session, especially when many frames have zero speed (ties at a boundary).

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is indexed by the same `frame_indices` used for neural data.

ii.
```python
speed_out = np.digitize(ft_RunSpeed[frame_indices], speed_quartiles).astype(np.int64)
```

iii. All data streams share the same frame indices.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Behavior arrays are trimmed to the number of neural frames (`nfr = spk.shape[1]`). Trials with fewer than 2 frames are skipped. Sessions with fewer than 2 valid trials are skipped. The agent noted a minor frame count discrepancy (behavior has one more frame than neural) but handles it by trimming.

ii.
```python
ft_trInd = beh['ft_trInd'][:nfr]
ft_CorrSpc = beh['ft_CorrSpc'][:nfr]
```
```python
if len(frame_indices) < 2:
    continue
```

iii. The agent noted: "I did notice a minor discrepancy where the behavior array has one more frame than the neural spike data (22476 vs 22475), but that single-frame difference is negligible."

## 12-a. What are the most time-consuming steps of the code?

i. Loading the neural spike files, which are very large (totaling hundreds of GB). The agent also noted the quartile computation initially loaded full neural data just to get frame counts, which was wasteful, and optimized it to only load the first plane's shape.

ii.
```python
spk_data = np.load(os.path.join(SPK_ROOT, fn), allow_pickle=True).item()
spk = np.concatenate(spk_data['spks'], axis=0)
```

iii. The agent noted: "The script is running but taking a long time because it's loading all the neural data files." and optimized the quartile computation to avoid unnecessary full loads.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The `compute_lick_per_frame` function uses a Python loop over frame indices instead of vectorized numpy operations:

ii.
```python
return np.array([1.0 if f in lick_frames_set else 0.0 for f in frame_indices], dtype=np.float32)
```

This could be vectorized with `np.isin(frame_indices, lick_frames_array)`.

iii. No explicit discussion of this inefficiency in the trajectory.

## 12-c. What processing does the code repeat multiple times?

i. The `get_nfr` function loads the neural data file just to get the frame count, and then `load_spk` loads it again for actual processing. This means each session's spike file is loaded twice: once during `compute_speed_quartiles` and once during `process_session`.

ii.
```python
def get_nfr(ndb):
    fn = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}_neural_data.npy"
    spk_data = np.load(os.path.join(SPK_ROOT, fn), allow_pickle=True).item()
    nfr = spk_data['spks'][0].shape[1]
    del spk_data
    return nfr
```

iii. The agent was aware of the inefficiency but chose this approach to reduce peak memory usage.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code includes all neurons (including unassigned brain regions) which adds significant data volume. These neurons outside the four visual areas are unlikely to be useful for decoding visual cortex activity. The code also stores neural data as float32 instead of float16, doubling the size of the largest data component.

ii.
```python
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'unassigned']
neural_trial = spk[:, frame_indices].astype(np.float32)
```

iii. The agent reasoned that "the decoder can learn which neurons carry useful signal on its own," but this significantly increases data size and computation.
