# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI builds a session map from `Imaging_Exp_info.npy` which maps experiment types to lists of session metadata dicts. It deduplicates sessions by `spk_key` (`mname_datexp_blk`), keeping one entry per unique neural recording. For each session, it loads: (1) neural data from `data/spk/{mname}_{datexp}_{blk}_neural_data.npy`, (2) retinotopy from `data/retinotopy/{mname}_{datexp}_trans.npz`, and (3) behavioral data from `data/beh/Beh_{exp_type}.npy`. The behavioral data is accessed via a key constructed from the session metadata.

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

iii. The AI documented that there are 142 total entries for 89 unique sessions across 23 experiment types, and chose to use one entry per physical neural recording. It prefers entries without `stimtype` to avoid duplicates.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `mname` field in the session metadata. The AI tracks unique mouse names as they appear during processing and assigns sequential indices. The `subjects` list is built from sorted unique mouse names, and `subject_idx` maps each session to its subject.

ii.
```python
subjects_seen = {}
# ... in processing loop:
if mname not in subjects_seen:
    subjects_seen[mname] = len(subjects_seen)
subject_idx_list.append(subjects_seen[mname])
# ... after loop:
subjects = sorted(subjects_seen.keys(), key=lambda x: subjects_seen[x])
```

iii. The AI noted 19 unique mice across 89 sessions, matching the paper's report of "89 recordings in 19 mice."

## 1-c. How are the data split into sessions?

i. Each unique neural recording file (`mname_datexp_blk`) defines one session. The AI iterates over all 89 unique `spk_key` entries in the session map. Sessions appearing in multiple experiment types are only processed once.

ii.
```python
keys = sample_keys if sample_keys else sorted(session_map.keys())
for idx, spk_key in enumerate(keys):
    result = process_session(spk_key, info, speed_quartiles)
    if result is None:
        continue
    neural_all.append(result['neural'])
```

iii. The AI documented in CONVERSION_NOTES.md that the same session can appear under multiple experiment types (e.g., sup_train1_before_learning and naive_test1), and resolved this by deduplicating on the neural data file key.

## 1-d. How are the data split into trials?

i. Trials are defined using the `StartFr` array from the behavioral data. Trial `i` spans from `StartFr[i]` to `StartFr[i+1]` (or end of session for the last trial). This means each trial includes both the corridor traversal and the gray space period until the next trial starts.

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

iii. The AI's CONVERSION_NOTES.md states that trials use frames from StartFr to the start of the next trial, capturing corridor + gray space. The `ntrials` count comes from `beh['ntrials']`.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies minimal trial filtering: trials with fewer than 2 frames (`n_frames < 2`) are skipped. Sessions with fewer than 2 valid trials are discarded entirely. No other quality-based trial filtering is applied (e.g., no filtering based on running, stimulus type, or behavioral performance).

ii.
```python
if n_frames < 2:
    continue
# ...
if len(neural_trials) < 2:
    print(f"WARNING: <2 valid trials for {spk_key}")
    return None
```

iii. The AI noted that the reference code does not apply explicit trial filtering for basic data loading, and chose to include all trials for the decoder.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from the `spks` field in the neural data files (`{mname}_{datexp}_{blk}_neural_data.npy`). This contains Suite2p deconvolved calcium traces (with tau=0.75s) stored as a list of arrays, one per imaging plane.

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

iii. The AI documented that neural data consists of deconvolved calcium traces from Suite2p, and that no additional delta F/F computation is needed since the data is already deconvolved.

## 2-b. How is the `neural` data processed?

i. The neural data is: (1) loaded from multiple imaging planes and concatenated, (2) filtered to exclude neurons outside visual cortex (based on `iarea` codes), (3) cast to float16 for memory efficiency, and (4) sliced per trial based on `StartFr` boundaries. No additional normalization, smoothing, or binning is applied.

ii.
```python
# In load_spk_filtered:
filtered.append(plane[plane_mask].astype(np.float16))
return np.concatenate(filtered, 0)
# In process_session:
spk = spk[:, :nfr_use]
# Per trial:
trial_spk = spk[:, start:end].copy()
```

iii. The AI justified using float16 as a memory optimization, noting it halves file size. The neural data type is documented as "Suite2p deconvolved calcium traces (tau=0.75s), stored as float16".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered based on brain area assignment from retinotopy data. Neurons with `iarea == -1` or `iarea == 7` are excluded (these are outside visual cortex). No d-prime-based filtering or other quality metrics are applied.

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

iii. The AI documented that the reference code uses `(arid!=-1) & (arid != 7)` to exclude neurons outside visual cortex, and applied the same filter. The d-prime selectivity criterion (d' >= 0.3) from the paper is for specific analyses, not for general data loading.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (corridor entry) by slicing frames starting at `StartFr[i]`. The first frame of each trial corresponds to the moment of corridor entry (StartFr). Within each trial, frame 0 = corridor entry, and subsequent frames are at the native frame rate.

ii.
```python
start = StartFr[i]
end = StartFr[i + 1] if i < ntrials - 1 else nfr_use
trial_spk = spk[:, start:end].copy()
```

iii. The AI's metadata specifies `temporal_alignment_event: 'corridor entry (trial start)'` and `off_start: 0.0`, confirming alignment to trial start with no pre-trial offset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native imaging frame rate of ~3.17 Hz, corresponding to ~315.5 ms per time bin. No temporal rebinning is applied — the data is used at its native resolution.

ii.
```python
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE  # ~315.5 ms
```

iii. The AI documented that the native frame rate is used without resampling, consistent with the paper's imaging parameters.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Time to sound cue is derived from `SoundFr` (per-trial frame index of sound cue delivery) and the current frame index within the trial.

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

iii. The AI documented that the sound cue position is uniformly distributed between 0.5m and 3.5m in the corridor, and used SoundFr to compute the time-varying distance to the cue.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each frame in a trial, the time to sound cue is computed as `(SoundFr - current_frame_index) / frame_rate`, yielding positive values before the cue and negative after. When SoundFr is NaN (no sound cue), the value is set to 0 for all frames.

ii.
```python
frame_idx = np.arange(start, end)
sound_fr = SoundFr[i]
if np.isnan(sound_fr):
    time_to_sound = np.zeros(n_frames, dtype=np.float32)
else:
    time_to_sound = ((sound_fr - frame_idx) / fs).astype(np.float32)
```

iii. The AI computed a continuous, time-varying signal in seconds.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The time to sound cue is computed using absolute frame indices (`frame_idx = np.arange(start, end)`) which correspond to the same frames as the neural data slice `spk[:, start:end]`. Both use the same frame range, ensuring alignment.

ii.
```python
frame_idx = np.arange(start, end)
time_to_sound = ((sound_fr - frame_idx) / fs).astype(np.float32)
# Neural data uses same range:
trial_spk = spk[:, start:end].copy()
```

iii. Alignment is inherent since both neural and input data index the same frames.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Day of training is derived from the `days` or `sess#` field in the session metadata dictionary (`db`).

ii.
```python
def get_session_day(db):
    for key in ['days', 'sess#']:
        if key in db:
            return int(db[key])
    return 0
# ...
day = np.float32(get_session_day(db))
```

iii. The AI documented that it uses the session metadata to extract the training day, with a fallback to 0 if neither field is available.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The day value is extracted as an integer from the metadata, cast to float32, and broadcast as a constant across all frames in all trials of that session.

ii.
```python
day = np.float32(get_session_day(db))
# Per trial:
np.full(n_frames, day, dtype=np.float32)
```

iii. Day of training is a per-session constant replicated across time within each trial.

## 4-a (Environment type). What variables in the raw data is `input` *Environment type* derived from?

i. The AI did NOT include "Environment type" as a decoder input. The decoder inputs are limited to the 4 specified in the instructions: time to sound cue, day of training, time since trial start, and reward availability. The experiment type information (supervised, unsupervised, naive, grating) is available in `Imaging_Exp_info.npy` but was not included.

ii. N/A — no code for this variable.

iii. The instructions specify exactly 4 decoder inputs, and "Environment type" is not among them. The AI followed the specification.

## 4-b (Environment type). What processing is involved in computing `input` *Environment type*?

i. Not applicable — Environment type was not included as a decoder input.

ii. N/A

iii. See 4-a above.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Time since trial start is derived from the frame index within each trial and the session frame rate. The trial start is defined by `StartFr[i]`.

ii.
```python
(np.arange(n_frames) / fs).astype(np.float32)
```

iii. This is a simple ramp from 0 at trial start, incrementing by `1/fs` seconds per frame.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each trial, a linearly increasing array is created: `frame_number / frame_rate`, producing time in seconds from trial start. The first frame is 0 seconds.

ii.
```python
(np.arange(n_frames) / fs).astype(np.float32)
```

iii. Simple division of frame count by frame rate.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Alignment is inherent — frame 0 of the input corresponds to frame 0 of the neural data (both starting at `StartFr[i]`). The array has the same length (`n_frames`) as the neural data trial.

ii.
```python
# Both have n_frames = end - start length
trial_spk = spk[:, start:end].copy()  # neural
(np.arange(n_frames) / fs)  # input
```

iii. Same frame range ensures alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Reward availability is derived from the `isRew` per-trial array in the behavioral data, which indicates whether each trial is in a rewarded corridor.

ii.
```python
isRew = beh['isRew']
# Per trial:
np.full(n_frames, float(isRew[i]), dtype=np.float32)
```

iii. The AI documented that `isRew` is a boolean per-trial indicator.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew[i]` value is cast to float (0.0 or 1.0) and broadcast as a constant across all frames in the trial.

ii.
```python
np.full(n_frames, float(isRew[i]), dtype=np.float32)
```

iii. No additional processing beyond type conversion and broadcasting.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Visual stimulus category is derived from `WallName` (per-trial stimulus name) in the behavioral data.

ii.
```python
WallName = beh['WallName']
# Per trial:
stim = standardize_stim_name(str(WallName[i]))
```

iii. The AI noted that stimulus names need to be standardized because different cohorts use different texture names for equivalent stimuli.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Raw stimulus names are mapped to standardized categories using `STIM_CATEGORY_MAP` (e.g., rock1→circle1, wood1→leaf1, brick1→circle1). The standardized names are then encoded as integer indices. The stimulus index is broadcast as a constant across all frames in the trial.

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
    'brick5': 'leaf3', 'wood5': 'leaf3', 'rock5': 'circle3',
}
# Encoding:
all_stim_sorted = sorted(all_stim_names)
stim_to_idx = {s: i for i, s in enumerate(all_stim_sorted)}
output_all[si][ti][0, :] = stim_to_idx[all_stim_per_session[si][ti]]
```

iii. The AI documented the mapping in CONVERSION_NOTES.md and verified 8 unique stimulus categories in the full dataset.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `LickFr` in the behavioral data, which contains frame indices where lick events were detected.

ii.
```python
lick_binary = np.zeros(nfr_use, dtype=np.int64)
if 'LickFr' in beh and len(beh['LickFr']) > 0:
    lick_fr = beh['LickFr'].astype(int)
    valid_lick = (lick_fr >= 0) & (lick_fr < nfr_use)
    lick_binary[lick_fr[valid_lick]] = 1
```

iii. The AI documented that licking is only present in supervised (task-trained) sessions; unsupervised sessions have all zeros.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A binary array (0/1) is created at the session level with one value per frame. Frame indices from `LickFr` are marked as 1 (licking), all others as 0. Invalid frame indices (negative or beyond session length) are excluded.

ii.
```python
lick_binary = np.zeros(nfr_use, dtype=np.int64)
if 'LickFr' in beh and len(beh['LickFr']) > 0:
    lick_fr = beh['LickFr'].astype(int)
    valid_lick = (lick_fr >= 0) & (lick_fr < nfr_use)
    lick_binary[lick_fr[valid_lick]] = 1
```

iii. The AI chose a binary representation with 2 classes: no_lick and lick.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The session-level `lick_binary` array is sliced using the same frame indices (`start:end`) as the neural data, ensuring temporal alignment.

ii.
```python
# Neural:
trial_spk = spk[:, start:end].copy()
# Licking:
lick_binary[start:end]
```

iii. Same frame-level indexing ensures alignment.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is derived from `ft_Pos` (frame-level position in decimeters, 0-60) and `ft_CorrSpc` (boolean indicating whether the mouse is in the texture corridor).

ii.
```python
ft_Pos = beh['ft_Pos'][:nfr_use]
ft_CorrSpc = beh['ft_CorrSpc'][:nfr_use].astype(bool)
```

iii. Documented in CONVERSION_NOTES.md under behavioral variables.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is discretized into 5 bins: 4 spatial bins covering the 4m texture corridor (each 1m = 10 decimeters wide) plus 1 bin for gray space.

ii.
```python
pos_bins = np.full(nfr_use, 4, dtype=np.int64)  # default: gray
for b in range(4):
    mask = ft_CorrSpc & (ft_Pos >= b * 10) & (ft_Pos < (b + 1) * 10)
    pos_bins[mask] = b
pos_bins[ft_CorrSpc & (ft_Pos >= 40)] = 3  # edge case
```

iii. The AI documented that gray space frames default to bin 4, and the 4 texture bins correspond to 0-1m, 1-2m, 2-3m, and 3-4m.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Five categories are used:
- Bin 0: 0-1m (ft_Pos 0-10 dm, in corridor)
- Bin 1: 1-2m (ft_Pos 10-20 dm, in corridor)
- Bin 2: 2-3m (ft_Pos 20-30 dm, in corridor)
- Bin 3: 3-4m (ft_Pos 30-40 dm, in corridor)
- Bin 4: gray space (not in corridor, ft_CorrSpc=False)

ii.
```python
'output_values': [
    # ...
    ['0-1m', '1-2m', '2-3m', '3-4m', 'gray'],
    # ...
]
```

iii. The instructions specify "4 equal-length, 1-m-long spatial bins" but the AI added a 5th bin for gray space since trials extend beyond the texture corridor.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is computed at the session level and sliced per trial using the same frame indices as neural data.

ii.
```python
pos_bins[start:end]
```

iii. Same frame-level indexing as neural data.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ft_RunSpeed` (frame-level running speed) in the behavioral data.

ii.
```python
ft_RunSpeed = beh['ft_RunSpeed'][:nfr_use]
```

iii. Documented as a frame-level behavioral variable.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is discretized into 4 bins based on quartile thresholds computed globally across all sessions. Quartiles are computed on **positive speeds only** (ft_RunSpeed > 0), then applied to all frames.

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
```

iii. The AI documented that quartiles are computed from behavior data only (no neural loading needed) for efficiency.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four bins based on quartile thresholds of positive speeds:
- Bin 0 (Q1_slow): speed < quartile_25
- Bin 1 (Q2): quartile_25 ≤ speed < quartile_50
- Bin 2 (Q3): quartile_50 ≤ speed < quartile_75
- Bin 3 (Q4_fast): speed ≥ quartile_75

Since quartiles are computed on positive speeds only, non-running frames (speed ≤ 0) all fall into Q1, making Q1 contain ~47% of data rather than 25%.

ii.
```python
speed_bins = np.zeros(nfr_use, dtype=np.int64)
speed_bins[ft_RunSpeed >= speed_quartiles[0]] = 1
speed_bins[ft_RunSpeed >= speed_quartiles[1]] = 2
speed_bins[ft_RunSpeed >= speed_quartiles[2]] = 3
```

iii. The AI documented quartile values of [6.91, 22.18, 39.23] dm/s and noted the Q1 skew.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed bins are computed at the session level and sliced per trial using the same frame indices.

ii.
```python
speed_bins[start:end]
```

iii. Same frame-level indexing as neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases:
- NaN `SoundFr`: time_to_sound set to 0 for all frames
- Invalid `LickFr` indices (negative or beyond session): excluded via bounds checking
- Frame count mismatch between neural and behavioral data: truncated to the minimum (`nfr_use = min(nfr, len(beh['ft']))`)
- Trials with < 2 frames: skipped
- Sessions with < 2 valid trials: skipped entirely
- Missing `days`/`sess#` field: defaults to 0

ii.
```python
nfr_use = min(nfr, len(beh['ft']))
# ...
if np.isnan(sound_fr):
    time_to_sound = np.zeros(n_frames, dtype=np.float32)
# ...
if n_frames < 2:
    continue
# ...
def get_session_day(db):
    for key in ['days', 'sess#']:
        if key in db:
            return int(db[key])
    return 0
```

iii. The AI documented these edge cases in CONVERSION_NOTES.md.

## 12-a. What are the most time-consuming steps of the code?

i. The AI identified neural data loading and filtering as the dominant bottleneck (~15-25 seconds per session). The total conversion time for all 89 sessions was ~21 minutes. Speed quartile computation was fast (~5 seconds total).

ii.
```python
# Timing is printed per session:
print(f"  [{idx+1}/{len(keys)}] {spk_key}...", end=' ', flush=True)
# ...
print(f"...{time.time()-t_s:.1f}s")
```

iii. CONVERSION_NOTES.md Step 7 documents: Neural loading+filtering ~15-25s per session, processing ~5s per session, estimated total ~21 minutes.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `process_session` could potentially be vectorized by using advanced indexing to extract all trial data at once. The position binning loop over 4 bins could be replaced with `np.digitize`. The per-plane filtering loop in `load_spk_filtered` could be vectorized.

ii.
```python
# Per-trial loop:
for i in range(ntrials):
    start = StartFr[i]
    end = StartFr[i + 1] if i < ntrials - 1 else nfr_use
    # ... extract per-trial data ...

# Position binning loop:
for b in range(4):
    mask = ft_CorrSpc & (ft_Pos >= b * 10) & (ft_Pos < (b + 1) * 10)
    pos_bins[mask] = b
```

iii. The AI acknowledged efficiency as a concern but focused on correctness first. The position loop is only 4 iterations so the overhead is minimal.

## 12-c. What processing does the code repeat multiple times?

i. Behavioral data is loaded twice for each session: once during speed quartile computation (`collect_speed_quartiles`) and once during session processing (`process_session`). This is a deliberate design choice — the first pass is lightweight (behavior only) to avoid loading neural data just for speed computation.

ii.
```python
def collect_speed_quartiles(session_map, keys):
    for spk_key in keys:
        beh = load_beh(info)  # First load
        speeds = beh['ft_RunSpeed']
# ...
def process_session(spk_key, session_info, speed_quartiles):
    beh = load_beh(session_info)  # Second load
```

iii. The AI justified this as a tradeoff: computing speed quartiles requires behavioral data from all sessions before any neural data is loaded, so duplicating the behavioral load avoids holding all neural data in memory.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes session-level frame rate (`fs`) from the `ft` timestamps for each session individually, though a global constant `FRAME_RATE = 3.17` is also defined. The per-session rate is used for input computations but could use the constant since the frame rate is consistent across sessions. The stimulus name standardization creates mappings that may not be needed if downstream code handles raw names. The `nneu_total` count is computed but only used for display, not in the output data.

ii.
```python
ft = beh['ft']
dt = np.median(np.diff(ft[:min(1000, len(ft))])) * 86400  # days->seconds
fs = 1.0 / dt if dt > 0 else FRAME_RATE
```

iii. Using per-session frame rate is actually more accurate than a constant, so this is not truly unnecessary — it's a minor precision improvement.
