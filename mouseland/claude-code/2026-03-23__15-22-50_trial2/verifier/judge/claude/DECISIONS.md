# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the master index `Imaging_Exp_info.npy` from `data/beh/` to enumerate all sessions grouped by experiment type. It builds a session map keyed by `spk_key` (`mname_datexp_blk`). For each session, it loads: (1) neural data from `data/spk/{spk_key}_neural_data.npy`, (2) retinotopy from `data/retinotopy/{mname}_{datexp}_trans.npz`, and (3) behavior from the corresponding `Beh_{exp_type}.npy` file. Behavior files are loaded per-session (not cached across sessions that share the same file).

ii. Session map construction:
```python
def build_session_map():
    exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
                       allow_pickle=True).item()
    session_map = {}
    for exp_type, db_list in exp_info.items():
        for ndb in db_list:
            spk_key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
            beh_key = f"{spk_key}_{ndb['stimtype']}" if 'stimtype' in ndb else spk_key
            if spk_key not in session_map or 'stimtype' not in ndb:
                session_map[spk_key] = {
                    'exp_type': exp_type, 'beh_key': beh_key, 'db': dict(ndb)
                }
    return session_map
```

iii. The AI noted that 89 unique sessions exist across 142 total entries in the experiment info (duplicates across experiment types). It prefers entries without `stimtype` when deduplicating.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mname` from the experiment info metadata. As sessions are processed, each unique `mname` is assigned an incrementing index in the order encountered (sorted by spk_key). 19 subjects are found.

ii.
```python
if mname not in subjects_seen:
    subjects_seen[mname] = len(subjects_seen)
subject_idx_list.append(subjects_seen[mname])
subjects = sorted(subjects_seen.keys(), key=lambda x: subjects_seen[x])
```

iii. The AI verified 19 subjects matching the paper's "89 recordings in 19 mice."

## 1-c. How are the data split into sessions?

i. A session is one unique `spk_key` = `mname_datexp_blk`. The `build_session_map()` function deduplicates recordings that appear under multiple experiment types by keeping one entry per `spk_key`, preferring entries without `stimtype`. All 89 sessions are processed.

ii.
```python
spk_key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
if spk_key not in session_map or 'stimtype' not in ndb:
    session_map[spk_key] = {...}
```

iii. The AI noted that the same recording can appear under multiple experiment types and chose to keep only one entry per physical recording.

## 1-d. How are the data split into trials?

i. Trials are defined using `StartFr` boundaries: trial i runs from `StartFr[i]` to `StartFr[i+1]` (or end of session for the last trial). This includes both corridor texture space AND gray space frames. Trials with fewer than 2 frames are skipped.

ii.
```python
for i in range(ntrials):
    start = StartFr[i]
    end = StartFr[i + 1] if i < ntrials - 1 else nfr_use
    start = max(0, start)
    end = min(nfr_use, end)
    n_frames = end - start
    if n_frames < 2:
        continue
    trial_spk = spk[:, start:end].copy()
```

iii. The AI stated: "Use frames from StartFr to start of next trial (or end of session). This captures corridor + gray space." The agent did not use `ft_trInd` or `ft_CorrSpc` to restrict to corridor-only frames.

## 1-e. How are trials filtered based on quality controls?

i. The only filter is that trials with fewer than 2 frames after clipping to the neural data length are skipped. Sessions with fewer than 2 valid trials are also skipped. No trial length outlier filtering is applied.

ii.
```python
if n_frames < 2:
    continue
...
if len(neural_trials) < 2:
    return None
```

iii. The AI considered long trials (some up to 5621 frames) but chose not to filter them, reasoning that "Reference code handles variable-length trials via position interpolation; removing long trials could bias decoder training."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `data/spk/{spk_key}_neural_data.npy`, which contains a list of per-plane neuron-by-frame arrays. The visual area of each neuron comes from `iarea` in `data/retinotopy/{mname}_{datexp}_trans.npz`.

ii.
```python
spk_data = np.load(os.path.join(DATA_ROOT, 'spk', fn), allow_pickle=True).item()
planes = spk_data['spks']
...
dtrans = np.load(os.path.join(DATA_ROOT, 'retinotopy', fn), allow_pickle=True)
return dtrans['iarea']
```

iii. The AI noted these are Suite2p deconvolved calcium traces.

## 2-b. How is the `neural` data processed?

i. Neural data is filtered per-plane by brain region mask before concatenation (to save memory), then stored as float16. No additional processing (normalization, smoothing, etc.) is applied.

ii.
```python
def load_spk_filtered(mname, datexp, blk, valid_mask):
    planes = spk_data['spks']
    filtered = []
    offset = 0
    for plane in planes:
        n = plane.shape[0]
        plane_mask = valid_mask[offset:offset+n]
        filtered.append(plane[plane_mask].astype(np.float16))
        offset += n
    return np.concatenate(filtered, 0)
```

iii. The AI noted that "Neural data is deconvolved calcium traces from Suite2p (already in data files)" so no further processing is needed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are excluded if their `iarea` is -1 or 7 (outside visual cortex). All other neurons are kept and assigned to V1, mHV, lHV, or aHV based on area codes.

ii.
```python
AREA_MAP = {8: 'V1', 0: 'mHV', 1: 'mHV', 2: 'mHV', 9: 'mHV', 5: 'lHV', 6: 'lHV', 3: 'aHV', 4: 'aHV'}
EXCLUDED_AREAS = {-1, 7}

def get_brain_region_idx(iarea):
    valid_mask = np.array([int(ia) not in EXCLUDED_AREAS for ia in iarea])
    region_idx = np.array([
        BRAIN_REGIONS.index(AREA_MAP.get(int(ia), 'V1'))
        for ia in iarea[valid_mask]
    ], dtype=np.int64)
    return valid_mask, region_idx
```

iii. The AI noted this matches the reference code's `neu_area_ID` function: "Excluded neurons outside visual cortex (iarea==-1 or iarea==7)."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry (trial start). Each trial's neural data is sliced from `StartFr[i]` to `StartFr[i+1]`, so it starts at corridor entry. Trials have variable length.

ii.
```python
start = StartFr[i]
end = StartFr[i + 1] if i < ntrials - 1 else nfr_use
trial_spk = spk[:, start:end].copy()
```

iii. The AI set `temporal_alignment_event` to "corridor entry (trial start)" and `off_start` to 0.0.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The native imaging frame rate of ~3.17 Hz is used, giving a time bin of ~315.5 ms.

ii.
```python
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE  # ~315.5 ms
```

iii. The AI noted "Use native frame rate (~315 ms). No resampling needed."

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr`, the frame number of the sound cue for each trial, and the current frame index.

ii.
```python
sound_fr = SoundFr[i]
time_to_sound = ((sound_fr - frame_idx) / fs).astype(np.float32)
```

iii. The AI uses the frame number directly, converting to time by dividing by frame rate.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The time to sound cue is computed as `(SoundFr - current_frame_index) / fs`, where `fs` is the session-specific frame rate computed from median inter-frame intervals. If SoundFr is NaN, time_to_sound is set to zeros. The sign convention is positive before the cue, negative after.

ii.
```python
frame_idx = np.arange(start, end)
sound_fr = SoundFr[i]
if np.isnan(sound_fr):
    time_to_sound = np.zeros(n_frames, dtype=np.float32)
else:
    time_to_sound = ((sound_fr - frame_idx) / fs).astype(np.float32)
```

iii. The AI computes session-specific frame rate from `ft` timestamps rather than using the constant 3.17 Hz.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The frame indices used to compute time_to_sound are the same `start:end` range used for the neural data slice, ensuring alignment.

ii.
```python
frame_idx = np.arange(start, end)
trial_spk = spk[:, start:end]
```

iii. Both use the same start:end frame range.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the `days` or `sess#` field in the experiment info metadata dictionary (`db`).

ii.
```python
def get_session_day(db):
    for key in ['days', 'sess#']:
        if key in db:
            return int(db[key])
    return 0
```

iii. The AI chose to use the metadata fields rather than counting session order per mouse.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The raw value from the metadata is used directly as an integer, broadcast across all time bins of every trial. If neither `days` nor `sess#` is available, defaults to 0.

ii.
```python
day = np.float32(get_session_day(db))
...
np.full(n_frames, day, dtype=np.float32)
```

iii. The AI stated: "Use sess# or days field from exp_info if available; otherwise use session chronological order within subject." The verification output shows day_of_training ranges from 0 to 15, suggesting `days` or `sess#` contains values larger than the number of sessions per mouse.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From the frame index within the trial (0-based) and the session frame rate.

ii.
```python
(np.arange(n_frames) / fs).astype(np.float32)
```

iii. Time is computed as frame offset from trial start divided by frame rate.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The time since trial start is computed as `frame_offset / fs`, where frame_offset is 0, 1, 2, ... for each frame in the trial. The session-specific frame rate `fs` is computed from median inter-frame intervals.

ii.
```python
dt = np.median(np.diff(ft[:min(1000, len(ft))])) * 86400  # days->seconds
fs = 1.0 / dt if dt > 0 else FRAME_RATE
...
(np.arange(n_frames) / fs).astype(np.float32)
```

iii. This always starts exactly at 0.0 for every trial.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed from frame offsets starting at 0 for the same slice of frames used for neural data.

ii.
```python
(np.arange(n_frames) / fs).astype(np.float32)
```

iii. Aligned by construction since both start at the same StartFr.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, a per-trial boolean indicating whether the corridor is rewarded.

ii.
```python
np.full(n_frames, float(isRew[i]), dtype=np.float32)
```

iii. Directly from the behavior data.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew[i]` is cast to float (0.0 or 1.0) and broadcast across all frames of the trial.

ii.
```python
np.full(n_frames, float(isRew[i]), dtype=np.float32)
```

iii. No additional processing needed.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, the texture name for each trial, mapped through `STIM_CATEGORY_MAP`.

ii.
```python
stim = standardize_stim_name(str(WallName[i]))
```

iii. The AI uses WallName for stimulus identity.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Wall names are mapped through `STIM_CATEGORY_MAP` which maps rock textures to circle equivalents and wood textures to leaf equivalents (e.g., rock1→circle1, wood1→leaf1). This results in 8-9 distinct stimulus categories (circle1, circle2, circle3, leaf1, leaf2, leaf3, leaf1_swap1, leaf1_swap2, wood5) rather than 4 broad categories.

ii.
```python
STIM_CATEGORY_MAP = {
    'circle1': 'circle1', 'circle2': 'circle2',
    'leaf1': 'leaf1', 'leaf2': 'leaf2', 'leaf3': 'leaf3',
    'leaf1_swap1': 'leaf1_swap1', 'leaf1_swap2': 'leaf1_swap2',
    'rock1': 'circle1', 'rock2': 'circle2',
    'wood1': 'leaf1', 'wood2': 'leaf2',
    'wood1_swap1': 'leaf1_swap1', 'wood1_swap2': 'leaf1_swap2',
    'brick1': 'circle1', 'brick2': 'circle2',
    'brick5': 'leaf3', 'wood5': 'leaf3',
    'rock5': 'circle3',
}
all_stim_sorted = sorted(all_stim_names)
stim_to_idx = {s: i for i, s in enumerate(all_stim_sorted)}
```

iii. The AI justified that "rock/wood/brick are equivalent stimuli used for different mice" but the actual output has 9 categories (circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, wood5), not the 4 broad categories (circle, leaf, rock, wood) from the paper.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the frame numbers of lick events in the session.

ii.
```python
lick_binary = np.zeros(nfr_use, dtype=np.int64)
if 'LickFr' in beh and len(beh['LickFr']) > 0:
    lick_fr = beh['LickFr'].astype(int)
    valid_lick = (lick_fr >= 0) & (lick_fr < nfr_use)
    lick_binary[lick_fr[valid_lick]] = 1
```

iii. Lick events are converted to a binary per-frame indicator.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A binary array is created with 1 at each frame where a lick occurred and 0 otherwise. Lick frames outside the valid range are excluded.

ii.
```python
lick_binary = np.zeros(nfr_use, dtype=np.int64)
lick_fr = beh['LickFr'].astype(int)
valid_lick = (lick_fr >= 0) & (lick_fr < nfr_use)
lick_binary[lick_fr[valid_lick]] = 1
```

iii. Straightforward binary encoding of lick events.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick binary array is indexed with the same `start:end` slice as the neural data.

ii.
```python
lick_binary[start:end]
```

iii. Both use the same frame range.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position in the corridor at each imaging frame (in decimeters, 0-60).

ii.
```python
ft_Pos = beh['ft_Pos'][:nfr_use]
```

iii. Directly from behavioral position data.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The position is binned into 5 categories: 4 texture bins (0-1m, 1-2m, 2-3m, 3-4m) plus a 5th "gray" bin for the gray space (4-6m). Bins are assigned based on position thresholds in decimeters.

ii.
```python
pos_bins = np.full(nfr_use, 4, dtype=np.int64)  # default: gray
for b in range(4):
    mask = ft_CorrSpc & (ft_Pos >= b * 10) & (ft_Pos < (b + 1) * 10)
    pos_bins[mask] = b
pos_bins[ft_CorrSpc & (ft_Pos >= 40)] = 3  # edge case
```

iii. The AI noted: "4 bins in texture area + 1 bin for gray space. Gray space is 33% of total corridor (2m/6m), justifying its own bin category."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The 5 categories are: bin 0 (0-1m), bin 1 (1-2m), bin 2 (2-3m), bin 3 (3-4m), bin 4 (gray space 4-6m). The default is gray (4), and texture bins are assigned based on `ft_CorrSpc` and position ranges.

ii.
```python
POSITION = ['0-1m', '1-2m', '2-3m', '3-4m', 'gray']  # in output_values
```

iii. Instructions specify "4 equal-length, 1-m-long spatial bins" but the AI added a 5th gray bin.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. The position array is sliced with the same `start:end` range as the neural data.

ii.
```python
pos_bins[start:end]
```

iii. Same frame range as neural data.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
ft_RunSpeed = beh['ft_RunSpeed'][:nfr_use]
```

iii. Directly from behavioral speed data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speed quartile thresholds are computed globally across ALL sessions from positive-speed frames only (speed > 0). These three thresholds (25th, 50th, 75th percentile of positive speeds) are then applied to all frames including stationary ones. This results in highly unequal bin sizes: Q1 contains ~47% of frames (all stopped + slowest running frames), while Q2-Q4 each contain ~17-18%.

ii.
```python
def collect_speed_quartiles(session_map, keys):
    all_speeds = np.concatenate(all_speeds)
    valid = all_speeds > 0
    return np.percentile(all_speeds[valid], [25, 50, 75])

speed_bins = np.zeros(nfr_use, dtype=np.int64)
speed_bins[ft_RunSpeed >= speed_quartiles[0]] = 1
speed_bins[ft_RunSpeed >= speed_quartiles[1]] = 2
speed_bins[ft_RunSpeed >= speed_quartiles[2]] = 3
```

iii. The AI justified: "Compute percentile thresholds on positive speeds only, then apply categorization to all frames."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four bins: Q1_slow (speed < 25th percentile of positive speeds), Q2, Q3, Q4_fast. The thresholds are [6.9, 22.2, 39.2] cm/s. Because thresholds are computed on positive speeds only, Q1 contains ~47% of data.

ii.
```python
SPEED = ['Q1_slow', 'Q2', 'Q3', 'Q4_fast']
speed_bins[ft_RunSpeed >= speed_quartiles[0]] = 1
speed_bins[ft_RunSpeed >= speed_quartiles[1]] = 2
speed_bins[ft_RunSpeed >= speed_quartiles[2]] = 3
```

iii. The instruction says "4 bins, each corresponding to 25% of the data" but the AI's approach gives unequal bins.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. The speed array is sliced with the same `start:end` range as the neural data.

ii.
```python
speed_bins[start:end]
```

iii. Same frame range as neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The number of usable frames is set to `min(nfr, len(beh['ft']))` to handle mismatches between neural and behavioral data length. Lick frames outside valid range are excluded with bounds checking. Trials with fewer than 2 frames are skipped. NaN SoundFr values are handled by setting time_to_sound to zero.

ii.
```python
nfr_use = min(nfr, len(beh['ft']))
valid_lick = (lick_fr >= 0) & (lick_fr < nfr_use)
if np.isnan(sound_fr):
    time_to_sound = np.zeros(n_frames, dtype=np.float32)
```

iii. The AI handled edge cases defensively.

## 12-a. What are the most time-consuming steps of the code?

i. Neural data loading and filtering dominates (~15-25s per session). The full conversion takes ~21 minutes for 89 sessions. Additionally, the behavior data is loaded redundantly - once during speed quartile computation and again during per-session processing.

ii.
```python
spk = load_spk_filtered(mname, datexp, blk, valid_mask)  # ~15-25s per session
```

iii. The AI estimated ~21 minutes total and noted per-plane filtering before concatenation as a memory optimization.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The position binning uses a loop over 4 bins instead of vectorized operations. The brain region assignment uses a Python list comprehension instead of vectorized numpy operations.

ii.
```python
for b in range(4):
    mask = ft_CorrSpc & (ft_Pos >= b * 10) & (ft_Pos < (b + 1) * 10)
    pos_bins[mask] = b

region_idx = np.array([
    BRAIN_REGIONS.index(AREA_MAP.get(int(ia), 'V1'))
    for ia in iarea[valid_mask]
], dtype=np.int64)
```

iii. These are minor compared to I/O time.

## 12-c. What processing does the code repeat multiple times?

i. Behavior data is loaded twice for every session: once in `collect_speed_quartiles()` and again in `process_session()` via `load_beh()`. Each call to `load_beh()` reads the entire behavior file from disk (which contains multiple sessions).

ii.
```python
# First load in collect_speed_quartiles:
beh = load_beh(info)
# Second load in process_session:
beh = load_beh(session_info)
```

iii. The AI did not cache behavior data between the two passes.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI includes gray space frames in every trial (32% of position data), which adds a 5th position bin. Since the instructions specify only 4 position bins, the gray space frames are unnecessary overhead. Additionally, the speed quartile computation includes all frames from sessions that won't be processed in sample mode.
