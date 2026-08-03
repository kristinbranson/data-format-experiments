# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent builds a session table from `data/beh/Imaging_Exp_info.npy`, uses each unique `{mname}_{datexp}_{blk}` recording as one session, then loads per-session behavior from the corresponding `Beh_{exp_type}.npy`, retinotopy from `data/retinotopy/{mname}_{datexp}_trans.npz`, and neural data from `data/spk/{mname}_{datexp}_{blk}_neural_data.npy`. It then loops over all selected sessions and converts each one with `process_session()`.

ii. ```python
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

def load_beh(session_info):
    beh_all = np.load(
        os.path.join(DATA_ROOT, 'beh', f"Beh_{session_info['exp_type']}.npy"),
        allow_pickle=True
    ).item()
    return beh_all[session_info['beh_key']]

for idx, spk_key in enumerate(keys):
    info = session_map[spk_key]
    result = process_session(spk_key, info, speed_quartiles)
```

iii. In `CONVERSION_NOTES.md` Step 4, the agent says there are 89 unique physical recordings even though experiment tables contain 142 entries, and that sessions with `stimtype` variants share the same underlying trial data. In trajectory step 45 it also explicitly reasoned that each unique neural data file should be used once.

## 1-b. How are the data split into subjects?

i. Subjects are split by the `mname` field from the experiment metadata. A new subject index is created the first time each mouse name is seen; that index is stored once per converted session in `subject_idx`.

ii. ```python
subjects_seen = {}

for idx, spk_key in enumerate(keys):
    info = session_map[spk_key]
    mname = info['db']['mname']
    ...
    if mname not in subjects_seen:
        subjects_seen[mname] = len(subjects_seen)
    ...
    subject_idx_list.append(subjects_seen[mname])

subjects = sorted(subjects_seen.keys(), key=lambda x: subjects_seen[x])
```

iii. In `CONVERSION_NOTES.md` Step 2, the agent records 19 subjects and identifies mouse names from the metadata and filenames. The same notes describe `mouse name -> subjects, subject_idx` as the intended mapping.

## 1-c. How are the data split into sessions?

i. Sessions are split by unique neural recording key `"{mname}_{datexp}_{blk}"`. If a recording appears in multiple experiment lists, the agent keeps one representative entry and processes that as a single session.

ii. ```python
for exp_type, db_list in exp_info.items():
    for ndb in db_list:
        spk_key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
        beh_key = f"{spk_key}_{ndb['stimtype']}" if 'stimtype' in ndb else spk_key
        if spk_key not in session_map or 'stimtype' not in ndb:
            session_map[spk_key] = {
                'exp_type': exp_type, 'beh_key': beh_key, 'db': dict(ndb)
            }
```

iii. `CONVERSION_NOTES.md` Step 4 states that the agent verified 89 unique `spk_key` sessions and decided to use each physical recording once because duplicate experiment entries point to the same underlying recording.

## 1-d. How are the data split into trials?

i. Within each session, trials are split using `StartFr`. Each trial starts at `StartFr[i]` and ends at the next trial's `StartFr[i+1]`, or at the end of the session for the last trial. Trials shorter than 2 frames are discarded.

ii. ```python
StartFr = beh['StartFr'].astype(int)
...
for i in range(ntrials):
    start = StartFr[i]
    end = StartFr[i + 1] if i < ntrials - 1 else nfr_use
    start = max(0, start)
    end = min(nfr_use, end)
    n_frames = end - start
    if n_frames < 2:
        continue
```

iii. In `CONVERSION_NOTES.md` Step 5, the agent says it will use frames from `StartFr` to the next trial start so that each trial contains the textured corridor plus gray-space period. Trajectory steps 69, 72, and 74 show the agent revisiting this choice while trying to control file size and deciding not to trim to corridor-only frames.

## 1-e. How are trials filtered based on quality controls?

i. The agent applies almost no trial-level quality filtering. It only drops trials with fewer than 2 frames after clipping to valid frame bounds, and later skips an entire session if fewer than 2 valid trials remain.

ii. ```python
start = max(0, start)
end = min(nfr_use, end)
n_frames = end - start
if n_frames < 2:
    continue
...
if len(neural_trials) < 2:
    print(f"WARNING: <2 valid trials for {spk_key}")
    return None
```

iii. `CONVERSION_NOTES.md` Step 3 says there is no explicit trial filtering in the reference code for basic loading and that, for the decoder, the agent would use all trials. The `<2 valid trials` check comes from the decoder-format requirement that each session needs at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes from the per-session `spks` list stored in each `*_neural_data.npy` file, after applying a neuron mask built from retinotopy `iarea`.

ii. ```python
fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
spk_data = np.load(os.path.join(DATA_ROOT, 'spk', fn), allow_pickle=True).item()
planes = spk_data['spks']
...
fn = f'{mname}_{datexp}_trans.npz'
dtrans = np.load(os.path.join(DATA_ROOT, 'retinotopy', fn), allow_pickle=True)
return dtrans['iarea']
```

iii. In `CONVERSION_NOTES.md` Step 1, the agent notes that the raw neural source is the Suite2p deconvolved traces in `{'spks': [...]}` and that retinotopy `iarea` determines area membership. Trajectory step 20 also shows the reference `utils.load_spk` concatenating `spks`.

## 2-b. How is the `neural` data processed?

i. The agent filters neurons plane-by-plane using the retinotopy-derived visual-cortex mask, concatenates planes, casts to `float16`, truncates to the minimum shared neural/behavior frame count, and then slices the continuous matrix into per-trial `(n_neurons, n_timepoints)` arrays aligned to trial start.

ii. ```python
def load_spk_filtered(mname, datexp, blk, valid_mask):
    ...
    for plane in planes:
        n = plane.shape[0]
        plane_mask = valid_mask[offset:offset+n]
        filtered.append(plane[plane_mask].astype(np.float16))
        offset += n
    return np.concatenate(filtered, 0)

spk = load_spk_filtered(mname, datexp, blk, valid_mask)
nneu, nfr = spk.shape
...
nfr_use = min(nfr, len(beh['ft']))
spk = spk[:, :nfr_use]
...
trial_spk = spk[:, start:end].copy()
```

iii. `CONVERSION_NOTES.md` Step 6 says these changes were deliberate memory optimizations: filter per plane before concatenation and store neural data as `float16`. Earlier trajectory steps show the agent explicitly deciding to keep frame-level deconvolved traces rather than adopting the reference position-interpolation pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural quality filter is anatomical: neurons with `iarea == -1` or `iarea == 7` are excluded as outside the visual cortex; remaining neurons are assigned to `V1`, `mHV`, `lHV`, or `aHV`.

ii. ```python
EXCLUDED_AREAS = {-1, 7}
...
valid_mask = np.array([int(ia) not in EXCLUDED_AREAS for ia in iarea])
region_idx = np.array([
    BRAIN_REGIONS.index(AREA_MAP.get(int(ia), 'V1'))
    for ia in iarea[valid_mask]
], dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` Steps 1 and 3 say the agent is following `utils.py neu_area_ID` and the paper's focus on neurons inside visual cortex. The trajectory includes the reference code for `neu_area_ID` and notes the excluded values `-1` and `7`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural traces are aligned to corridor entry by starting each trial at `StartFr[i]`. The first sample of each `trial_spk` matrix is therefore the first imaging frame after corridor entry.

ii. ```python
StartFr = beh['StartFr'].astype(int)
...
start = StartFr[i]
...
trial_spk = spk[:, start:end].copy()
```

iii. In `CONVERSION_NOTES.md` Step 5, the agent explicitly writes that temporal alignment is to `StartFr` / corridor entry because the decoder task specifies trial-start alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use the native imaging frame rate, approximately 3.17 Hz, corresponding to about 315.5 ms per bin. No temporal rebinning is applied.

ii. ```python
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE  # ~315.5 ms
...
dt = np.median(np.diff(ft[:min(1000, len(ft))])) * 86400
fs = 1.0 / dt if dt > 0 else FRAME_RATE
```

iii. `CONVERSION_NOTES.md` Steps 3 and 5 say the agent deliberately kept the native frame-rate time axis instead of interpolating to position bins because the decoder task is time-based.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundFr` and the frame indices inside each trial.

ii. ```python
SoundFr = beh['SoundFr']
...
frame_idx = np.arange(start, end)
sound_fr = SoundFr[i]
```

iii. `CONVERSION_NOTES.md` Step 5 maps `SoundFr` directly to `input[0]: time_to_sound_cue`.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each frame in a trial, the agent computes `(sound_fr - frame_idx) / fs`. If `SoundFr` is missing (`NaN`), it fills the whole trial with zeros.

ii. ```python
if np.isnan(sound_fr):
    time_to_sound = np.zeros(n_frames, dtype=np.float32)
else:
    time_to_sound = ((sound_fr - frame_idx) / fs).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 describes the same formula. The missing-cue zero fill is not justified in the notes; it is an implementation fallback visible only in the code.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on exactly the same trial frame indices used to slice the neural matrix, so every neural frame has a matching `time_to_sound` value.

ii. ```python
trial_spk = spk[:, start:end].copy()
frame_idx = np.arange(start, end)
...
inp = np.stack([
    time_to_sound,
    np.full(n_frames, day, dtype=np.float32),
    (np.arange(n_frames) / fs).astype(np.float32),
    np.full(n_frames, float(isRew[i]), dtype=np.float32),
], axis=0)
```

iii. In `CONVERSION_NOTES.md` Step 5, the agent says all decoder inputs are to be frame-aligned with the per-trial neural data at the native sampling rate.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from session metadata in `Imaging_Exp_info.npy`, specifically `db['days']` if present, otherwise `db['sess#']`, otherwise `0`.

ii. ```python
def get_session_day(db):
    for key in ['days', 'sess#']:
        if key in db:
            return int(db[key])
    return 0
```

iii. `CONVERSION_NOTES.md` Step 5 says the agent intended to use `sess#` or `days` from experiment metadata as the training-day variable.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The chosen metadata value is cast to `float32` and then repeated across all frames in the trial as a constant per-trial input channel.

ii. ```python
day = np.float32(get_session_day(db))
...
np.full(n_frames, day, dtype=np.float32),
```

iii. The notes describe this as a per-trial scalar input. The code does not implement the fallback to chronological ordering that the notes mention as a possibility.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from the trial start defined by `StartFr` together with the within-trial frame count and the session frame rate `fs`.

ii. ```python
StartFr = beh['StartFr'].astype(int)
...
(np.arange(n_frames) / fs).astype(np.float32),
```

iii. `CONVERSION_NOTES.md` Step 5 maps this input to `(current_frame - StartFr) / fs`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For every trial the agent generates `0, 1/fs, 2/fs, ...` up to the trial length, stores it as `float32`, and uses it as the third input channel.

ii. ```python
inp = np.stack([
    time_to_sound,
    np.full(n_frames, day, dtype=np.float32),
    (np.arange(n_frames) / fs).astype(np.float32),
    np.full(n_frames, float(isRew[i]), dtype=np.float32),
], axis=0)
```

iii. The notes explicitly say this variable starts at 0 at trial entry and increases continuously across the trial.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned one-to-one with the per-trial neural frames because it is built with the same `n_frames` used for `trial_spk`.

ii. ```python
n_frames = end - start
trial_spk = spk[:, start:end].copy()
...
(np.arange(n_frames) / fs).astype(np.float32),
```

iii. The agent's stated plan in `CONVERSION_NOTES.md` Step 5 is that all inputs and outputs share the neural frame grid.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial boolean `isRew`.

ii. ```python
isRew = beh['isRew']
...
np.full(n_frames, float(isRew[i]), dtype=np.float32),
```

iii. `CONVERSION_NOTES.md` Step 5 maps `isRew` directly to `reward_availability`.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The agent converts `isRew[i]` to `0.0` or `1.0` and repeats that same value across all frames of the trial.

ii. ```python
inp = np.stack([
    time_to_sound,
    np.full(n_frames, day, dtype=np.float32),
    (np.arange(n_frames) / fs).astype(np.float32),
    np.full(n_frames, float(isRew[i]), dtype=np.float32),
], axis=0)
```

iii. The notes justify this as a per-trial binary context variable: 1 for rewarded corridors, 0 otherwise.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from per-trial `WallName`, then standardized into canonical category names and finally encoded into integer class IDs.

ii. ```python
WallName = beh['WallName']
...
stim = standardize_stim_name(str(WallName[i]))
...
all_stim_sorted = sorted(all_stim_names)
stim_to_idx = {s: i for i, s in enumerate(all_stim_sorted)}
for si in range(len(output_all)):
    for ti in range(len(output_all[si])):
        output_all[si][ti][0, :] = stim_to_idx[all_stim_per_session[si][ti]]
```

iii. `CONVERSION_NOTES.md` Step 5 says the agent planned to use `WallName` as the source and standardize rock/wood/brick variants to canonical circle/leaf-style names.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The agent maps synonymous wall names through `STIM_CATEGORY_MAP`, collects the unique standardized stimulus names across sessions, sorts them, and writes the corresponding integer stimulus index into every time bin of the first output row for that trial.

ii. ```python
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
...
out = np.stack([
    np.full(n_frames, 0, dtype=np.int64),
    lick_binary[start:end],
    pos_bins[start:end],
    speed_bins[start:end],
], axis=0)
```

iii. In trajectory step 224 and the later session summary, the agent justifies these mappings by reference to canonical `stim_id` categories used in the paper/code and by equivalence between rock/wood/brick and circle/leaf stimulus families.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from the global lick frame index array `LickFr`.

ii. ```python
lick_binary = np.zeros(nfr_use, dtype=np.int64)
if 'LickFr' in beh and len(beh['LickFr']) > 0:
    lick_fr = beh['LickFr'].astype(int)
    valid_lick = (lick_fr >= 0) & (lick_fr < nfr_use)
    lick_binary[lick_fr[valid_lick]] = 1
```

iii. `CONVERSION_NOTES.md` Step 5 states that licking should be a binary per-frame output derived from `LickFr` / `LickTrind`.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The agent builds a session-wide binary lick raster with ones at all valid lick frames, then slices each trial's lick vector from that raster.

ii. ```python
lick_binary = np.zeros(nfr_use, dtype=np.int64)
...
lick_binary[lick_fr[valid_lick]] = 1
...
out = np.stack([
    np.full(n_frames, 0, dtype=np.int64),
    lick_binary[start:end],
    pos_bins[start:end],
    speed_bins[start:end],
], axis=0)
```

iii. The notes justify this as the simplest frame-aligned binary representation for a time-varying decoder target.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned by frame index: the agent first marks lick events in the session frame grid and then uses the same `start:end` bounds as the neural data for each trial.

ii. ```python
trial_spk = spk[:, start:end].copy()
...
lick_binary[start:end]
```

iii. In `CONVERSION_NOTES.md` Step 5, the agent says licking is stored as a time-varying output aligned to the trial-aligned neural frames.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from frame-level position `ft_Pos` together with `ft_CorrSpc`, which marks whether the mouse is in the textured corridor.

ii. ```python
ft_Pos = beh['ft_Pos'][:nfr_use]
ft_CorrSpc = beh['ft_CorrSpc'][:nfr_use].astype(bool)
```

iii. `CONVERSION_NOTES.md` Step 5 maps position directly from `ft_Pos` and notes that corridor location is encoded in decimeters.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The agent creates a frame-level categorical position vector. Frames in the textured corridor are assigned one of four 10-decimeter bins; all other frames default to a separate gray-space category.

ii. ```python
pos_bins = np.full(nfr_use, 4, dtype=np.int64)  # default: gray
for b in range(4):
    mask = ft_CorrSpc & (ft_Pos >= b * 10) & (ft_Pos < (b + 1) * 10)
    pos_bins[mask] = b
pos_bins[ft_CorrSpc & (ft_Pos >= 40)] = 3  # edge case
```

iii. `CONVERSION_NOTES.md` Step 5 shows the agent explicitly debating whether gray space should be its own category and deciding to keep a fifth category despite the task asking for four bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. It is thresholded into `0-1 m`, `1-2 m`, `2-3 m`, `3-4 m`, plus an extra `gray` category outside the textured corridor.

ii. ```python
'output_values': [
    all_stim_sorted,
    ['no_lick', 'lick'],
    ['0-1m', '1-2m', '2-3m', '3-4m', 'gray'],
    ['Q1_slow', 'Q2', 'Q3', 'Q4_fast'],
],
```

iii. The justification is explicit in `CONVERSION_NOTES.md` Step 5: the agent recognized the instructions asked for 4 bins but decided to add a fifth gray-space label rather than dropping those frames.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position categories are first computed on the session frame grid and then sliced into trials with the same `start:end` indices used for neural data.

ii. ```python
trial_spk = spk[:, start:end].copy()
...
pos_bins[start:end]
```

iii. The notes say the entire converted representation is built on the native neural frame grid aligned to corridor entry.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from frame-level `ft_RunSpeed`.

ii. ```python
ft_RunSpeed = beh['ft_RunSpeed'][:nfr_use]
```

iii. `CONVERSION_NOTES.md` Step 5 maps running speed directly from `ft_RunSpeed`.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The agent first concatenates all session `ft_RunSpeed` arrays, computes global quartile cutoffs using only strictly positive speeds, and then applies those cutoffs frame-by-frame within each session.

ii. ```python
def collect_speed_quartiles(session_map, keys):
    all_speeds = []
    for spk_key in keys:
        info = session_map[spk_key]
        beh = load_beh(info)
        speeds = beh['ft_RunSpeed']
        all_speeds.append(speeds)
    all_speeds = np.concatenate(all_speeds)
    valid = all_speeds > 0
    if valid.sum() == 0:
        return np.array([1.0, 2.0, 3.0])
    return np.percentile(all_speeds[valid], [25, 50, 75])
```

iii. `CONVERSION_NOTES.md` Step 5 says the agent intended to compute quartiles across running frames in the whole dataset. Step 10 later notes that this makes the slowest bin contain many more than 25% of all frames because zeros are excluded from the percentile calculation.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Speed is thresholded into four ordinal bins using the three percentile cutoffs. Frames below the 25th positive-speed percentile remain in bin 0; higher speeds step upward through bins 1, 2, and 3.

ii. ```python
speed_bins = np.zeros(nfr_use, dtype=np.int64)
speed_bins[ft_RunSpeed >= speed_quartiles[0]] = 1
speed_bins[ft_RunSpeed >= speed_quartiles[1]] = 2
speed_bins[ft_RunSpeed >= speed_quartiles[2]] = 3
```

iii. The notes describe these as quartile bins and later acknowledge that, because the thresholds are computed only on positive speeds, the resulting bin counts are not balanced over all frames.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed bins are computed on the session's neural frame grid and then sliced into trials using the same `start:end` bounds as the neural matrix.

ii. ```python
trial_spk = spk[:, start:end].copy()
...
speed_bins[start:end]
```

iii. The agent's stated design in `CONVERSION_NOTES.md` Step 5 is to keep all outputs aligned to the native frame-level neural time base.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent uses several local fixes: it truncates neural data to the shortest shared frame count with behavior; clips trial bounds to valid indices; ignores out-of-range lick frames; fills missing `SoundFr` with an all-zero `time_to_sound` trace; defaults missing day metadata to `0`; skips trials shorter than 2 frames; and skips sessions with fewer than 2 valid trials.

ii. ```python
nfr_use = min(nfr, len(beh['ft']))
spk = spk[:, :nfr_use]
...
start = max(0, start)
end = min(nfr_use, end)
...
valid_lick = (lick_fr >= 0) & (lick_fr < nfr_use)
...
if np.isnan(sound_fr):
    time_to_sound = np.zeros(n_frames, dtype=np.float32)
...
for key in ['days', 'sess#']:
    if key in db:
        return int(db[key])
return 0
```

iii. Most of these fixes are not justified explicitly in the notes. The only clearly documented one is the `<2 valid trials` session skip, which the agent links to decoder requirements. The others are pragmatic implementation guards visible in the code.

## 12-a. What are the most time-consuming steps of the code?

i. The dominant cost is session-by-session neural loading and trial extraction, especially loading huge `spks` arrays, filtering them per plane, and copying per-trial neural slices. Speed-quartile collection is comparatively cheap because it reads behavior only.

ii. ```python
speed_quartiles = collect_speed_quartiles(session_map, keys)
...
spk = load_spk_filtered(mname, datexp, blk, valid_mask)
...
for i in range(ntrials):
    ...
    trial_spk = spk[:, start:end].copy()
```

iii. `CONVERSION_NOTES.md` Step 7 estimates full-runtime cost and identifies neural loading plus processing as the major contributors. The code structure matches that estimate.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities are the per-plane filtering loop in `load_spk_filtered`, the per-trial loop in `process_session`, the per-bin position loop, and the nested loop that writes stimulus IDs after all sessions are processed.

ii. ```python
for plane in planes:
    ...

for b in range(4):
    mask = ft_CorrSpc & (ft_Pos >= b * 10) & (ft_Pos < (b + 1) * 10)
    pos_bins[mask] = b

for i in range(ntrials):
    ...

for si in range(len(output_all)):
    for ti in range(len(output_all[si])):
        output_all[si][ti][0, :] = stim_to_idx[all_stim_per_session[si][ti]]
```

iii. The AI does not justify these loops directly. This section is mostly inferred from the implementation and from the runtime concerns documented in `CONVERSION_NOTES.md`.

## 12-c. What processing does the code repeat multiple times?

i. It loads behavior twice per session path: once globally during `collect_speed_quartiles()` and again in `process_session()`. It also repeatedly allocates per-trial constant arrays for day, reward, and stimulus labels instead of storing those variables once per trial.

ii. ```python
def collect_speed_quartiles(session_map, keys):
    for spk_key in keys:
        info = session_map[spk_key]
        beh = load_beh(info)
        ...

def process_session(spk_key, session_info, speed_quartiles):
    ...
    beh = load_beh(session_info)
    ...
    np.full(n_frames, day, dtype=np.float32)
    np.full(n_frames, float(isRew[i]), dtype=np.float32)
    np.full(n_frames, 0, dtype=np.int64)
```

iii. The repeated behavior loading is partly justified in `CONVERSION_NOTES.md` Step 6 as a tradeoff to avoid loading neural data when computing speed quartiles. The repeated per-trial constant expansion is not discussed in the notes.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It constructs time-varying copies of trial-constant variables (`day_of_training`, `reward_availability`, `visual_stimulus`) at every frame; makes `.copy()` of each neural trial; collects `stim_names` only to rewrite stimulus IDs later; and optionally generates diagnostic plots that are not part of the decoder dataset.

ii. ```python
trial_spk = spk[:, start:end].copy()
...
inp = np.stack([
    time_to_sound,
    np.full(n_frames, day, dtype=np.float32),
    (np.arange(n_frames) / fs).astype(np.float32),
    np.full(n_frames, float(isRew[i]), dtype=np.float32),
], axis=0)
...
out = np.stack([
    np.full(n_frames, 0, dtype=np.int64),
    lick_binary[start:end],
    pos_bins[start:end],
    speed_bins[start:end],
], axis=0)
...
if show_processing:
    plot_processing(data)
```

iii. The notes justify some of this only indirectly via decoder-format compatibility and visualization. The extra per-frame replication and later stimulus rewrite are implementation choices rather than reference-driven processing.
