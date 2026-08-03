# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three sources: (1) `Imaging_Exp_info.npy` to build a session map linking experiment metadata to neural/behavioral files, (2) neural data from `data/spk/{mname}_{datexp}_{blk}_neural_data.npy` files, (3) behavioral data from `data/beh/Beh_{exp_type}.npy` files, and (4) retinotopy from `data/retinotopy/{mname}_{datexp}_trans.npz`. Sessions appearing in multiple experiment types are deduplicated by `spk_key` (mname_datexp_blk), preferring entries without `stimtype`.

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

iii. The AI documented in CONVERSION_NOTES Step 4 that "Sessions can appear in multiple experiment types (142 total entries for 89 unique sessions). For the decoder, each physical recording (neural data file) is used once." This matches the reference code's approach of one entry per unique neural recording.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `mname` field from the session metadata. A dictionary `subjects_seen` tracks unique mouse names and assigns sequential indices. The `subject_idx` array maps each session to its subject.

ii.
```python
if mname not in subjects_seen:
    subjects_seen[mname] = len(subjects_seen)
subject_idx_list.append(subjects_seen[mname])
subjects = sorted(subjects_seen.keys(), key=lambda x: subjects_seen[x])
```

iii. The AI noted 19 subjects matching the paper's "89 recordings in 19 mice." Subject splitting follows naturally from the session metadata structure.

## 1-c. How are the data split into sessions?

i. Each unique `spk_key` (combination of mname, datexp, blk) constitutes one session. Sessions with `stimtype` variants pointing to the same neural recording are deduplicated, resulting in 89 sessions.

ii.
```python
spk_key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
# ...
keys = sample_keys if sample_keys else sorted(session_map.keys())
for idx, spk_key in enumerate(keys):
    result = process_session(spk_key, info, speed_quartiles)
```

iii. The AI verified in CONVERSION_NOTES Step 4 that 89 unique sessions match between exp_info, neural data files, and the paper.

## 1-d. How are the data split into trials?

i. Trials are defined by the `StartFr` array from behavioral data. Trial `i` spans from `StartFr[i]` to `StartFr[i+1]` (or end of recording for the last trial). This captures both the texture corridor and gray space portions of each trial.

ii.
```python
StartFr = beh['StartFr'].astype(int)
for i in range(ntrials):
    start = StartFr[i]
    end = StartFr[i + 1] if i < ntrials - 1 else nfr_use
    start = max(0, start)
    end = min(nfr_use, end)
    n_frames = end - start
    if n_frames < 2:
        continue
```

iii. The AI noted in CONVERSION_NOTES Step 5 that "Trial length: Use frames from StartFr to start of next trial (or end of session). This captures corridor + gray space."

## 1-e. How are trials filtered based on quality controls?

i. The only trial filtering is excluding trials with fewer than 2 frames (`n_frames < 2`). Sessions with fewer than 2 valid trials are also skipped. No other trial quality filtering is applied (no filtering by running state, no trial count limits).

ii.
```python
if n_frames < 2:
    continue
# ...
if len(neural_trials) < 2:
    print(f"WARNING: <2 valid trials for {spk_key}")
    return None
```

iii. The AI noted in CONVERSION_NOTES Step 3: "No explicit trial filtering in reference code for basic loading. Reference code sometimes restricts to first 200 trials for specific analyses. For decoder: use all trials (no filtering)."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `spks` in the neural data files (`{mname}_{datexp}_{blk}_neural_data.npy`). These contain Suite2p deconvolved calcium traces (with decay timescale tau=0.75s), stored as a list of arrays per imaging plane.

ii.
```python
def load_spk_filtered(mname, datexp, blk, valid_mask):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk_data = np.load(os.path.join(DATA_ROOT, 'spk', fn), allow_pickle=True).item()
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

iii. The AI documented in CONVERSION_NOTES Step 1: "Neural data: Suite2p deconvolved calcium traces (NOT raw fluorescence). Deconvolution with tau=0.75s." This matches the reference paper which states "All analyses based on deconvolved fluorescence traces."

## 2-b. How is the `neural` data processed?

i. Neural data processing involves: (1) concatenating planes, (2) filtering out neurons outside visual cortex based on `iarea` codes, (3) converting to float16 for memory efficiency. No additional processing such as normalization, z-scoring, or running-frame filtering is applied. The reference code applies running-frame filtering (`ft_move > 0`) for its analyses, but the AI chose to include all frames.

ii.
```python
filtered.append(plane[plane_mask].astype(np.float16))
# ...
trial_spk = spk[:, start:end].copy()
```

iii. The AI noted in CONVERSION_NOTES Step 1: "Reference code only uses running frames: `VRmove = beh['ft_move'][:nfr]>0`" but chose not to apply this filtering for the decoder task. The AI described its neural data as "Suite2p deconvolved calcium traces (tau=0.75s), stored as float16."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by brain area assignment: neurons with `iarea == -1` (outside visual cortex) or `iarea == 7` are excluded. No d-prime-based selectivity filtering (d' >= 0.3) is applied — that filtering is specific to certain analyses in the reference code, not general data loading.

ii.
```python
EXCLUDED_AREAS = {-1, 7}
def get_brain_region_idx(iarea):
    valid_mask = np.array([int(ia) not in EXCLUDED_AREAS for ia in iarea])
    region_idx = np.array([
        BRAIN_REGIONS.index(AREA_MAP.get(int(ia), 'V1'))
        for ia in iarea[valid_mask]
    ], dtype=np.int64)
    return valid_mask, region_idx
```

iii. The AI noted in CONVERSION_NOTES Step 3: "Exclude neurons outside visual cortex: iarea==-1 or iarea==7." This matches the reference code's `neu_area_ID` function which uses `(arid!=-1) & (arid != 7)`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (corridor entry) using `StartFr[i]` as the first frame of each trial. Each trial's neural data spans from `StartFr[i]` to `StartFr[i+1]` (or end of session).

ii.
```python
start = StartFr[i]
end = StartFr[i + 1] if i < ntrials - 1 else nfr_use
trial_spk = spk[:, start:end].copy()
```

iii. The AI set `temporal_alignment_event` to "corridor entry (trial start)" and `off_start` to 0.0, matching the instruction "Temporally aligned based on trial start (corridor entry)."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging frame rate of ~3.17 Hz is used directly, giving a time bin size of ~315.5 ms. No temporal rebinning is applied.

ii.
```python
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE  # ~315.5 ms
```

iii. The AI noted "fs = 3.17Hz (from data_process_script)" and decided: "Time bin size: Use native frame rate (~315 ms). No resampling needed." This matches the reference code which operates at the native frame rate.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from `SoundFr` (per-trial frame index of sound cue delivery) and the current frame index within the trial.

ii.
```python
SoundFr = beh['SoundFr']
# ...
sound_fr = SoundFr[i]
if np.isnan(sound_fr):
    time_to_sound = np.zeros(n_frames, dtype=np.float32)
else:
    time_to_sound = ((sound_fr - frame_idx) / fs).astype(np.float32)
```

iii. The AI mapped "Time to SoundFr" → "input[0]: time_to_sound_cue" with transform "(SoundFr - current_frame) / fs."

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the time to sound cue is computed as `(SoundFr - current_frame_index) / frame_rate`, giving positive values before the cue and negative values after. When `SoundFr` is NaN (no sound cue for that trial), the time is set to zero for all frames.

ii.
```python
frame_idx = np.arange(start, end)
sound_fr = SoundFr[i]
if np.isnan(sound_fr):
    time_to_sound = np.zeros(n_frames, dtype=np.float32)
else:
    time_to_sound = ((sound_fr - frame_idx) / fs).astype(np.float32)
```

iii. The AI described the transform as "Negative before cue, 0 at cue, positive after" in CONVERSION_NOTES Step 5. Note: the actual code computes it as `(SoundFr - frame_idx)` which is positive before the cue (when frame_idx < SoundFr) and negative after.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The time to sound cue uses the same frame indices (`np.arange(start, end)`) as the neural data slice (`spk[:, start:end]`), ensuring perfect temporal alignment.

ii.
```python
frame_idx = np.arange(start, end)
time_to_sound = ((sound_fr - frame_idx) / fs).astype(np.float32)
# Neural uses same start:end
trial_spk = spk[:, start:end].copy()
```

iii. No explicit discussion in CONVERSION_NOTES, but alignment is inherent in using the same frame indices for all variables.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from session metadata fields `days` or `sess#` in the experiment info dictionary.

ii.
```python
def get_session_day(db):
    for key in ['days', 'sess#']:
        if key in db:
            return int(db[key])
    return 0
```

iii. The AI noted in CONVERSION_NOTES Step 5: "Use sess# or days field from exp_info if available; otherwise use session chronological order within subject."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The day value is extracted as an integer from session metadata, converted to float32, and broadcast as a constant across all frames in each trial.

ii.
```python
day = np.float32(get_session_day(db))
# ...
np.full(n_frames, day, dtype=np.float32),
```

iii. The AI described it as "Per-trial scalar" in the variable mapping table.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from the frame index within each trial (relative to trial start) and the session frame rate.

ii.
```python
(np.arange(n_frames) / fs).astype(np.float32),
```

iii. The AI mapped it as "Time since StartFr → input[2]: time_since_trial_start" with transform "(current_frame - StartFr) / fs."

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each trial, the time since trial start is computed as `frame_index_within_trial / frame_rate`, giving time in seconds starting from 0.

ii.
```python
(np.arange(n_frames) / fs).astype(np.float32),
```

iii. The AI described it as "Continuous, starts at 0" in the variable mapping.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Uses the same number of frames (`n_frames = end - start`) as the neural data, with index 0 corresponding to the first neural frame of the trial. Inherently aligned.

ii.
```python
n_frames = end - start
# Neural: spk[:, start:end] has n_frames timepoints
# Input: np.arange(n_frames) / fs also has n_frames timepoints
```

iii. Alignment is inherent in the code structure.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from `isRew` in the behavioral data, which is a per-trial boolean/binary indicator of whether the trial's corridor is rewarded.

ii.
```python
isRew = beh['isRew']
# ...
np.full(n_frames, float(isRew[i]), dtype=np.float32),
```

iii. The AI mapped "isRew → input[3]: reward_availability" as "Binary 0/1, Per-trial scalar."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The per-trial boolean `isRew[i]` is converted to float (0.0 or 1.0) and broadcast as a constant across all frames in the trial.

ii.
```python
np.full(n_frames, float(isRew[i]), dtype=np.float32),
```

iii. The AI described it as "Binary 0/1" per-trial. No further processing.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from `WallName` in the behavioral data, which contains the stimulus texture name for each trial (e.g., 'circle1', 'leaf2', 'wood1', 'rock2').

ii.
```python
WallName = beh['WallName']
# ...
stim = standardize_stim_name(str(WallName[i]))
```

iii. The AI mapped "WallName → output[0]: visual_stimulus" with "Categorical encoding."

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Wall names are standardized via `STIM_CATEGORY_MAP` which maps equivalent textures across cohorts (e.g., rock→circle, wood→leaf, brick→circle/leaf). The standardized name is encoded as an integer index, constant across all frames of a trial.

ii.
```python
STIM_CATEGORY_MAP = {
    'circle1': 'circle1', 'circle2': 'circle2',
    'rock1': 'circle1', 'rock2': 'circle2',
    'wood1': 'leaf1', 'wood2': 'leaf2',
    'wood5': 'leaf3', 'rock5': 'circle3',
    # ... etc
}
def standardize_stim_name(name):
    return STIM_CATEGORY_MAP.get(name, name)
# ...
stim = standardize_stim_name(str(WallName[i]))
out = np.stack([np.full(n_frames, 0, dtype=np.int64), ...])
# Later: output_all[si][ti][0, :] = stim_to_idx[all_stim_per_session[si][ti]]
```

iii. The AI noted in CONVERSION_NOTES: "Stimulus name standardization (rock→circle, wood→leaf, brick→circle/leaf3)." However, the data contains `circle3` as a raw WallName which is NOT in the STIM_CATEGORY_MAP, causing it to pass through unmapped. The final output has 9 categories including `wood5` and `circle3` as separate entries, suggesting either a code/data mismatch or the converted data was generated from an earlier code version.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `LickFr` in the behavioral data, which contains frame indices when licks were detected.

ii.
```python
lick_binary = np.zeros(nfr_use, dtype=np.int64)
if 'LickFr' in beh and len(beh['LickFr']) > 0:
    lick_fr = beh['LickFr'].astype(int)
    valid_lick = (lick_fr >= 0) & (lick_fr < nfr_use)
    lick_binary[lick_fr[valid_lick]] = 1
```

iii. The AI mapped "LickFr/LickTrind → output[1]: licking" as "Binary per frame."

## 8-b. What processing is involved in computing `output` *Licking*?

i. A binary array is created for the full session. `LickFr` frame indices (filtered for valid range) are used to set those frames to 1. The per-trial licking output is then sliced from this session-level array.

ii.
```python
lick_binary = np.zeros(nfr_use, dtype=np.int64)
if 'LickFr' in beh and len(beh['LickFr']) > 0:
    lick_fr = beh['LickFr'].astype(int)
    valid_lick = (lick_fr >= 0) & (lick_fr < nfr_use)
    lick_binary[lick_fr[valid_lick]] = 1
# Per trial:
lick_binary[start:end]
```

iii. The AI described licking as "Binary per frame - check if any lick occurred in that frame's time window."

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The session-level `lick_binary` array is sliced with the same `start:end` indices as the neural data, ensuring perfect temporal alignment.

ii.
```python
out = np.stack([
    np.full(n_frames, 0, dtype=np.int64),  # stim placeholder
    lick_binary[start:end],                 # same start:end as neural
    pos_bins[start:end],
    speed_bins[start:end],
], axis=0)
```

iii. Alignment is inherent in using the same frame indices.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Derived from `ft_Pos` (frame-level position in corridor, in decimeters 0-60) and `ft_CorrSpc` (boolean mask indicating whether the mouse is in the texture corridor 0-4m).

ii.
```python
ft_Pos = beh['ft_Pos'][:nfr_use]
ft_CorrSpc = beh['ft_CorrSpc'][:nfr_use].astype(bool)
```

iii. The AI mapped "ft_Pos → output[2]: position_bin" with "Discretize 0-4m into 4 bins."

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is discretized into 4 texture corridor bins (each 1m = 10 decimeters) plus a 5th bin for gray space. Frames in the texture corridor (`ft_CorrSpc == True`) are binned by position: 0-10dm → bin 0, 10-20dm → bin 1, 20-30dm → bin 2, 30-40dm → bin 3. Frames in gray space default to bin 4.

ii.
```python
pos_bins = np.full(nfr_use, 4, dtype=np.int64)  # default: gray
for b in range(4):
    mask = ft_CorrSpc & (ft_Pos >= b * 10) & (ft_Pos < (b + 1) * 10)
    pos_bins[mask] = b
pos_bins[ft_CorrSpc & (ft_Pos >= 40)] = 3  # edge case
```

iii. The AI noted: "4 bins in texture area... + 1 bin for gray space. Actually, re-reading: the task says 4 bins. I'll use 4 bins and mark gray space as a 5th bin."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four equal 1-meter bins in the texture corridor (0-1m, 1-2m, 2-3m, 3-4m) plus a 5th "gray" category for frames outside the texture corridor. The output values are labeled `['0-1m', '1-2m', '2-3m', '3-4m', 'gray']`.

ii.
```python
'output_values': [
    all_stim_sorted,
    ['no_lick', 'lick'],
    ['0-1m', '1-2m', '2-3m', '3-4m', 'gray'],
    ['Q1_slow', 'Q2', 'Q3', 'Q4_fast'],
],
```

iii. The instructions specify "4 equal-length, 1-m-long spatial bins." The AI added a 5th category for gray space frames since trials extend beyond the 4m texture corridor.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. The session-level `pos_bins` array is sliced with the same `start:end` indices as the neural data.

ii.
```python
pos_bins[start:end],
```

iii. Alignment is inherent in using the same frame indices.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Derived from `ft_RunSpeed` in the behavioral data, which contains per-frame running speed values.

ii.
```python
ft_RunSpeed = beh['ft_RunSpeed'][:nfr_use]
```

iii. The AI mapped "ft_RunSpeed → output[3]: running_speed_bin" with "Quartile discretization across all frames."

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is discretized into 4 bins based on quartile thresholds. Quartiles are computed across all sessions but only on positive speed values (`speed > 0`), excluding stationary frames. The thresholds (25th, 50th, 75th percentiles of positive speeds) are then applied to all frames including those with zero/negative speed.

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

iii. The AI noted: "Discretize into 4 quartile bins computed across ALL running frames in the dataset." The resulting distribution is heavily skewed: Q1 contains ~47% of data (all stationary + slow frames below the 25th percentile of positive speeds), while Q2-Q4 each contain ~17.6%.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three thresholds from the 25th, 50th, and 75th percentiles of positive running speeds divide frames into 4 bins: Q1_slow (speed < 25th pct), Q2 (25th-50th), Q3 (50th-75th), Q4_fast (>75th). Because quartiles are computed on positive speeds only, Q1 absorbs all non-running frames.

ii.
```python
speed_bins = np.zeros(nfr_use, dtype=np.int64)
speed_bins[ft_RunSpeed >= speed_quartiles[0]] = 1
speed_bins[ft_RunSpeed >= speed_quartiles[1]] = 2
speed_bins[ft_RunSpeed >= speed_quartiles[2]] = 3
```

iii. The AI labeled the bins `['Q1_slow', 'Q2', 'Q3', 'Q4_fast']`.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. The session-level `speed_bins` array is sliced with the same `start:end` indices as the neural data.

ii.
```python
speed_bins[start:end],
```

iii. Alignment is inherent in using the same frame indices.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled: (1) NaN `SoundFr` → time_to_sound set to zero vector. (2) Missing or empty `LickFr` → all-zero lick array. (3) Frame count mismatch between neural and behavioral data → truncated to minimum via `min(nfr, len(beh['ft']))`. (4) Trials with < 2 frames → skipped. (5) Sessions with < 2 valid trials → skipped. (6) Invalid lick frame indices (negative or beyond session length) → filtered out.

ii.
```python
nfr_use = min(nfr, len(beh['ft']))
# ...
if np.isnan(sound_fr):
    time_to_sound = np.zeros(n_frames, dtype=np.float32)
# ...
if 'LickFr' in beh and len(beh['LickFr']) > 0:
    lick_fr = beh['LickFr'].astype(int)
    valid_lick = (lick_fr >= 0) & (lick_fr < nfr_use)
    lick_binary[lick_fr[valid_lick]] = 1
# ...
if n_frames < 2:
    continue
```

iii. The AI documented frame count truncation and NaN handling across various variables. These are reasonable defensive measures.

## 12-a. What are the most time-consuming steps of the code?

i. Neural data loading is the dominant bottleneck. Each session's neural data file contains tens of thousands of neurons across thousands of frames (e.g., 58,224 neurons × 31,707 frames per plane × 3 planes). Loading, filtering, and converting to float16 takes ~15-25 seconds per session. Total conversion time for 89 sessions is ~21 minutes.

ii.
```python
def load_spk_filtered(mname, datexp, blk, valid_mask):
    spk_data = np.load(os.path.join(DATA_ROOT, 'spk', fn), allow_pickle=True).item()
    planes = spk_data['spks']
    filtered = []
    for plane in planes:
        plane_mask = valid_mask[offset:offset+n]
        filtered.append(plane[plane_mask].astype(np.float16))
    return np.concatenate(filtered, 0)
```

iii. The AI noted in CONVERSION_NOTES Step 7 time estimates: "Neural loading+filtering: ~15-25s per session, estimated total ~1600s."

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-level loop in `process_session` iterates over all trials sequentially, constructing per-trial arrays individually. The position binning loop iterates over 4 bins. Both could potentially be vectorized, though the trial loop's variable-length outputs make full vectorization difficult.

ii.
```python
for i in range(ntrials):
    # Individual trial processing
    start = StartFr[i]
    end = StartFr[i + 1] if i < ntrials - 1 else nfr_use
    # ...
for b in range(4):
    mask = ft_CorrSpc & (ft_Pos >= b * 10) & (ft_Pos < (b + 1) * 10)
    pos_bins[mask] = b
```

iii. The position binning loop (4 iterations) is minor. The trial loop is inherently sequential due to variable-length outputs. The AI noted filter-per-plane as a memory optimization.

## 12-c. What processing does the code repeat multiple times?

i. Behavioral data is loaded twice for each session: once during `collect_speed_quartiles` (to compute global speed thresholds) and once during `process_session`. This doubles the behavioral I/O.

ii.
```python
def collect_speed_quartiles(session_map, keys):
    for spk_key in keys:
        info = session_map[spk_key]
        beh = load_beh(info)            # First load
        speeds = beh['ft_RunSpeed']
# ...
def process_session(spk_key, session_info, speed_quartiles):
    beh = load_beh(session_info)         # Second load
```

iii. The AI separated speed quartile computation as a first pass for correctness (needs global statistics before per-session processing). The behavioral data is much smaller than neural data so the cost is modest (~5s total vs ~21 min overall).

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores neural data as float16 arrays for all neurons in visual cortex (17K-79K per session). The downstream decoder subsamples to 10,000 neurons per session and then projects to 2,000 dimensions via random projection before SVD to 100 PCs. The vast majority of per-neuron data is thus discarded. Additionally, the 5th position bin (gray space) adds a category that the instructions didn't request.

ii.
```python
# All neurons stored (17K-79K per session)
filtered.append(plane[plane_mask].astype(np.float16))
# But decoder only uses 10K and projects to 100 PCs
```

iii. The AI noted the memory issue in CONVERSION_NOTES Step 11: "Full dataset... requires ~623 GB... Solution: Subsample to 10,000 neurons per session before training." The full neuron storage is appropriate for a general-purpose conversion, but contributes to the 177 GB file size.
