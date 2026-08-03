# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three directories: `beh/` (behavior), `spk/` (neural data), and `retinotopy/` (brain area assignments). It builds a session map from `Imaging_Exp_info.npy`, iterating over experiment types and creating a mapping from spike key to session info. For each session, it loads behavior via `load_beh()`, neural data via `load_spk_filtered()`, and retinotopy via `load_retino()`. The behavior file is re-loaded for each session individually (not shared across sessions from the same behavior file).

ii.
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

iii. The AI's CONVERSION_NOTES.md documents that sessions can appear in multiple experiment types (142 entries for 89 unique sessions) and that each physical recording is used once.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mname` from the session metadata. The AI tracks subjects in order of first appearance in the sorted session keys, assigning sequential indices.

ii.
```python
if mname not in subjects_seen:
    subjects_seen[mname] = len(subjects_seen)
# ...
subjects = sorted(subjects_seen.keys(), key=lambda x: subjects_seen[x])
```

iii. The AI notes 19 subjects and 89 sessions, consistent with the paper.

## 1-c. How are the data split into sessions?

i. A session is identified by the spike key `{mname}_{datexp}_{blk}`. The `build_session_map()` function deduplicates sessions that appear under multiple experiment types by keeping entries without `stimtype` preferentially.

ii.
```python
spk_key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
if spk_key not in session_map or 'stimtype' not in ndb:
    session_map[spk_key] = {...}
```

iii. The AI documents that 89 unique sessions are found from 142 total entries in the experiment info.

## 1-d. How are the data split into trials?

i. The AI defines trials using `StartFr` boundaries: trial `i` spans from `StartFr[i]` to `StartFr[i+1]` (or end of session for the last trial). This includes both the corridor texture portion AND the gray space between corridors. Trials with fewer than 2 frames are dropped.

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
```

iii. The CONVERSION_NOTES.md states: "Trial length: Use frames from StartFr to start of next trial (or end of session). This captures corridor + gray space."

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 frames are dropped. Sessions with fewer than 2 valid trials are skipped. No other quality filtering is applied.

ii.
```python
if n_frames < 2:
    continue
# ...
if len(neural_trials) < 2:
    print(f"WARNING: <2 valid trials for {spk_key}")
    return None
```

iii. No explicit justification beyond the minimum frame count.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/{session_id}_neural_data.npy` (deconvolved calcium traces), concatenated across imaging planes. Brain area assignments come from `iarea` in `retinotopy/{mname}_{datexp}_trans.npz`.

ii.
```python
spk_data = np.load(os.path.join(DATA_ROOT, 'spk', fn), allow_pickle=True).item()
planes = spk_data['spks']
# Filter and concatenate
filtered.append(plane[plane_mask].astype(np.float16))
return np.concatenate(filtered, 0)
```

iii. CONVERSION_NOTES.md: "Neural data is deconvolved calcium traces from Suite2p (already in data files). No additional delta F/F computation needed."

## 2-b. How is the `neural` data processed?

i. Neural data is filtered per-plane by brain area mask before concatenation (to save memory), then cast to float16. No additional processing (smoothing, normalization, etc.) is applied. Each trial gets a variable-length slice of the neural data from StartFr to the next StartFr.

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

iii. CONVERSION_NOTES.md notes "Filter neurons per-plane before concatenation (saves memory and time)" and "Store neural data as float16 (halves file size)."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons outside visual cortex are excluded: those with `iarea == -1` or `iarea == 7`. Remaining neurons are assigned to V1, mHV, lHV, or aHV.

ii.
```python
EXCLUDED_AREAS = {-1, 7}
AREA_MAP = {8: 'V1', 0: 'mHV', 1: 'mHV', 2: 'mHV', 9: 'mHV', 5: 'lHV', 6: 'lHV', 3: 'aHV', 4: 'aHV'}

def get_brain_region_idx(iarea):
    valid_mask = np.array([int(ia) not in EXCLUDED_AREAS for ia in iarea])
    region_idx = np.array([
        BRAIN_REGIONS.index(AREA_MAP.get(int(ia), 'V1'))
        for ia in iarea[valid_mask]
    ], dtype=np.int64)
    return valid_mask, region_idx
```

iii. CONVERSION_NOTES.md: "Exclude neurons outside visual cortex: iarea==-1 or iarea==7."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry (`StartFr`). Each trial runs from `StartFr[i]` to `StartFr[i+1]`, resulting in variable-length trials. No fixed window or padding to a standard length is applied.

ii.
```python
start = StartFr[i]
end = StartFr[i + 1] if i < ntrials - 1 else nfr_use
trial_spk = spk[:, start:end].copy()
```

iii. The metadata sets `temporal_alignment_event` to `'corridor entry (trial start)'` and `off_end` to `None`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The native imaging frame rate of ~3.17 Hz is used, giving a time bin of ~315.5 ms.

ii.
```python
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE  # ~315.5 ms
```

iii. CONVERSION_NOTES.md: "Time bin size: Use native frame rate (~315 ms). No resampling needed."

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the frame number of the sound cue for each trial) and the frame index within the trial.

ii.
```python
sound_fr = SoundFr[i]
if np.isnan(sound_fr):
    time_to_sound = np.zeros(n_frames, dtype=np.float32)
else:
    time_to_sound = ((sound_fr - frame_idx) / fs).astype(np.float32)
```

iii. No specific justification in CONVERSION_NOTES.md beyond the variable mapping table.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The difference between `SoundFr` and the current frame index is divided by the session-specific frame rate (`fs`). This converts frame differences to seconds. If `SoundFr` is NaN, the time is set to zero. The sign convention is positive before the cue (sound_fr > frame_idx) and negative after.

ii.
```python
frame_idx = np.arange(start, end)
sound_fr = SoundFr[i]
if np.isnan(sound_fr):
    time_to_sound = np.zeros(n_frames, dtype=np.float32)
else:
    time_to_sound = ((sound_fr - frame_idx) / fs).astype(np.float32)
```

iii. No explicit justification beyond the mapping table. The AI computes a session-specific frame rate `fs` from the median inter-frame interval.

## 3-c. How is `input` *Time to sound cue* aligned with the neural data?

i. It uses the same frame indices (`np.arange(start, end)`) as the neural data for each trial.

ii.
```python
frame_idx = np.arange(start, end)
time_to_sound = ((sound_fr - frame_idx) / fs).astype(np.float32)
```

iii. Alignment is implicit through shared frame indexing.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From `days` or `sess#` fields in the session metadata dictionary (`db`), falling back to 0 if neither is available.

ii.
```python
def get_session_day(db):
    for key in ['days', 'sess#']:
        if key in db:
            return int(db[key])
    return 0
```

iii. CONVERSION_NOTES.md mapping: "Session day info -> input[1]: day_of_training. Session index within mouse or sess# field."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The raw value from the metadata is used directly as an integer, then broadcast as a float32 constant across all time bins of a trial.

ii.
```python
day = np.float32(get_session_day(db))
# ...
np.full(n_frames, day, dtype=np.float32),
```

iii. No transformation beyond type casting.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Computed from the frame count within the trial and the session-specific frame rate.

ii.
```python
(np.arange(n_frames) / fs).astype(np.float32)
```

iii. No explicit justification.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The frame index within the trial (0-based) is divided by the session frame rate to give time in seconds, starting at 0.

ii.
```python
(np.arange(n_frames) / fs).astype(np.float32)
```

iii. CONVERSION_NOTES.md mapping: "Time since StartFr -> input[2]: time_since_trial_start. (current_frame - StartFr) / fs. Continuous, starts at 0."

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It starts at 0 for the first frame of the trial and increments by 1/fs for each frame, matching the neural data frames exactly.

ii.
```python
(np.arange(n_frames) / fs).astype(np.float32)
```

iii. Alignment is implicit since both use the same frame count.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, a per-trial boolean indicating whether the trial was in the rewarded corridor.

ii.
```python
isRew = beh['isRew']
# ...
np.full(n_frames, float(isRew[i]), dtype=np.float32)
```

iii. No explicit justification needed.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew[i]` is cast to float (0.0 or 1.0) and broadcast across all frames of the trial.

ii.
```python
np.full(n_frames, float(isRew[i]), dtype=np.float32)
```

iii. No specific justification.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, the stimulus name for each trial. The AI maps stimulus names through `STIM_CATEGORY_MAP` which maps rock variants to circle variants and wood variants to leaf variants.

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
    'brick5': 'leaf3',
    'wood5': 'leaf3',
    'rock5': 'circle3',
}
```

iii. CONVERSION_NOTES.md: "Stimulus name standardization (rock->circle, wood->leaf, brick->circle/leaf3)."

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Wall names are mapped through `STIM_CATEGORY_MAP` to "standardized" names, then each unique name is assigned an integer index. This results in 9 categories (circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, wood5), broadcast as a constant across all frames of a trial.

ii.
```python
stim = standardize_stim_name(str(WallName[i]))
# ...
all_stim_sorted = sorted(all_stim_names)
stim_to_idx = {s: i for i, s in enumerate(all_stim_sorted)}
for si in range(len(output_all)):
    for ti in range(len(output_all[si])):
        output_all[si][ti][0, :] = stim_to_idx[all_stim_per_session[si][ti]]
```

iii. The AI chose to keep individual stimulus variants as separate categories rather than grouping into 4 broad texture categories.

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

iii. No explicit justification beyond the variable mapping.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A binary array is created per frame. Lick frame indices are truncated to integers and used to set the corresponding frame to 1. Invalid lick frames (negative or beyond session length) are excluded.

ii.
```python
lick_binary = np.zeros(nfr_use, dtype=np.int64)
lick_fr = beh['LickFr'].astype(int)
valid_lick = (lick_fr >= 0) & (lick_fr < nfr_use)
lick_binary[lick_fr[valid_lick]] = 1
```

iii. No explicit justification.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The licking binary array is indexed with the same frame range as the neural data (`start:end`).

ii.
```python
lick_binary[start:end]
```

iii. Alignment through shared frame indexing.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos` (position per frame in decimeters) and `ft_CorrSpc` (boolean indicating whether the frame is in the textured corridor).

ii.
```python
ft_Pos = beh['ft_Pos'][:nfr_use]
ft_CorrSpc = beh['ft_CorrSpc'][:nfr_use].astype(bool)
```

iii. No explicit justification.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is binned into 5 categories: 4 texture corridor bins (each 10 dm = 1 m) plus a gray space bin (value 4). The binning uses explicit range checks with `ft_CorrSpc` to distinguish corridor from gray space.

ii.
```python
pos_bins = np.full(nfr_use, 4, dtype=np.int64)  # default: gray
for b in range(4):
    mask = ft_CorrSpc & (ft_Pos >= b * 10) & (ft_Pos < (b + 1) * 10)
    pos_bins[mask] = b
pos_bins[ft_CorrSpc & (ft_Pos >= 40)] = 3  # edge case
```

iii. CONVERSION_NOTES.md: "4 bins in texture area...+ 1 bin for gray space."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. 5 categories: 0-1m (bin 0), 1-2m (bin 1), 2-3m (bin 2), 3-4m (bin 3), gray (bin 4). The gray space is all frames not in `ft_CorrSpc` or those in gray space between corridors.

ii.
```python
POSITION = ['0-1m', '1-2m', '2-3m', '3-4m', 'gray']
```

iii. The AI notes the instructions ask for 4 bins but adds a 5th for gray space.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. The position array is indexed with the same frame range as the neural data.

ii.
```python
pos_bins[start:end]
```

iii. Aligned through shared frame indexing.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
ft_RunSpeed = beh['ft_RunSpeed'][:nfr_use]
```

iii. No explicit justification.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speed is discretized into 4 bins using global quartile thresholds computed across ALL sessions' behavioral data. The thresholds are the 25th, 50th, and 75th percentiles of positive (>0) running speeds across the entire dataset.

ii.
```python
def collect_speed_quartiles(session_map, keys):
    all_speeds = []
    for spk_key in keys:
        info = session_map[spk_key]
        beh = load_beh(info)
        speeds = beh['ft_RunSpeed']
        all_speeds.append(speeds)
    all_speeds = np.concatenate(all_speeds)
    valid = all_speeds > 0
    return np.percentile(all_speeds[valid], [25, 50, 75])

# Application:
speed_bins = np.zeros(nfr_use, dtype=np.int64)
speed_bins[ft_RunSpeed >= speed_quartiles[0]] = 1
speed_bins[ft_RunSpeed >= speed_quartiles[1]] = 2
speed_bins[ft_RunSpeed >= speed_quartiles[2]] = 3
```

iii. CONVERSION_NOTES.md: "Speed quartile computation across all sessions."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. 4 categories based on global percentile thresholds of positive speeds: Q1_slow (below 25th percentile or zero), Q2 (25th-50th), Q3 (50th-75th), Q4_fast (above 75th).

ii.
```python
SPEED = ['Q1_slow', 'Q2', 'Q3', 'Q4_fast']
```

iii. The actual quartile values are [6.91, 22.18, 39.23] cm/s.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. The speed bin array is indexed with the same frame range as the neural data.

ii.
```python
speed_bins[start:end]
```

iii. Aligned through shared frame indexing.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled: (1) the behavior may have more frames than neural data, so arrays are truncated to the minimum; (2) lick frames outside the valid range are excluded; (3) NaN sound cue frames produce zero time_to_sound values; (4) trials with <2 frames are dropped; (5) sessions with <2 valid trials are skipped.

ii.
```python
nfr_use = min(nfr, len(beh['ft']))
spk = spk[:, :nfr_use]
# ...
valid_lick = (lick_fr >= 0) & (lick_fr < nfr_use)
# ...
if np.isnan(sound_fr):
    time_to_sound = np.zeros(n_frames, dtype=np.float32)
```

iii. CONVERSION_NOTES.md notes the behavior can run past the imaging frames.

## 12-a. What are the most time-consuming steps of the code?

i. Loading and filtering the neural spike files, which total hundreds of GB. Each session requires reading the full spike file from disk and filtering by brain area.

ii.
```python
spk_data = np.load(os.path.join(DATA_ROOT, 'spk', fn), allow_pickle=True).item()
```

iii. The conversion takes ~1870 seconds (~31 minutes) for 89 sessions.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The position binning uses an explicit loop over 4 bins when a single vectorized expression like `np.clip(ft_Pos // 10, 0, 3)` would suffice. The speed quartile collection re-loads all behavior files separately from the main processing loop.

ii.
```python
for b in range(4):
    mask = ft_CorrSpc & (ft_Pos >= b * 10) & (ft_Pos < (b + 1) * 10)
    pos_bins[mask] = b
```

iii. No explicit discussion of vectorization opportunities.

## 12-c. What processing does the code repeat multiple times?

i. The behavior files are loaded twice: once in `collect_speed_quartiles()` to compute global speed thresholds, and again in the main `process_session()` loop. This doubles the I/O for all behavior files.

ii.
```python
def collect_speed_quartiles(session_map, keys):
    for spk_key in keys:
        beh = load_beh(info)  # first load
# ...
def process_session(spk_key, session_info, speed_quartiles):
    beh = load_beh(session_info)  # second load
```

iii. No explicit justification for this repetition.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI includes gray space frames in each trial (from StartFr to next StartFr), but the gray space position bin (bin 4) and the corresponding frames are essentially noise for the decoder since the position, speed, and neural signals during gray space are not meaningfully related to the corridor task. The reference solution restricts to corridor frames only.

ii.
```python
start = StartFr[i]
end = StartFr[i + 1] if i < ntrials - 1 else nfr_use
# This includes gray space frames
```

iii. The CONVERSION_NOTES.md states: "Trial length: Use frames from StartFr to start of next trial (or end of session). This captures corridor + gray space."
