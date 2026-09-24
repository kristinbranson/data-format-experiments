# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads everything from the three subfolders of `/app/data`: `beh/` (behavior), `spk/` (deconvolved traces) and `retinotopy/` (visual-area label per neuron). `beh/Imaging_Exp_info.npy` is the master index: it is a dict of experiment type → list of recording entries, and the AI flattens it into `session_map`, keyed by `spk_key = mname_datexp_blk`, storing the experiment type, the behavior key and the metadata dict for each recording. For each session it then loads (1) the retinotopy `.npz` (to build the neuron mask first), (2) the per-session spike file `spk/<spk_key>_neural_data.npy` (a dict with a list of per-plane arrays, filtered per plane and concatenated), and (3) the behavior, by loading the whole `beh/Beh_<exp_type>.npy` file and indexing the session's key. Trials are then cut out of the session-level frame arrays. The behavior file is re-loaded from disk each time `load_beh` is called (once in the speed-quartile pass, once during processing), i.e. there is no caching/grouping by behavior file.

ii.
```python
def build_session_map():
    """Build mapping: spk_key -> {exp_type, beh_key, db}"""
    exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
                       allow_pickle=True).item()
    session_map = {}
    for exp_type, db_list in exp_info.items():
        for ndb in db_list:
            spk_key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
            beh_key = f"{spk_key}_{ndb['stimtype']}" if 'stimtype' in ndb else spk_key
            # Prefer entries without stimtype
            if spk_key not in session_map or 'stimtype' not in ndb:
                session_map[spk_key] = {
                    'exp_type': exp_type, 'beh_key': beh_key, 'db': dict(ndb)
                }
    return session_map
```
```python
def load_spk_filtered(mname, datexp, blk, valid_mask):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk_data = np.load(os.path.join(DATA_ROOT, 'spk', fn), allow_pickle=True).item()
    planes = spk_data['spks']
    ...
    return np.concatenate(filtered, 0)

def load_retino(mname, datexp):
    fn = f'{mname}_{datexp}_trans.npz'
    dtrans = np.load(os.path.join(DATA_ROOT, 'retinotopy', fn), allow_pickle=True)
    return dtrans['iarea']

def load_beh(session_info):
    beh_all = np.load(
        os.path.join(DATA_ROOT, 'beh', f"Beh_{session_info['exp_type']}.npy"),
        allow_pickle=True).item()
    return beh_all[session_info['beh_key']]
```
```python
keys = sample_keys if sample_keys else sorted(session_map.keys())
...
for idx, spk_key in enumerate(keys):
    result = process_session(spk_key, info, speed_quartiles)
```

iii. From CONVERSION_NOTES Step 1/2: the AI mirrored `utils.py`'s `load_spk` / `load_retino` / `load_exp_beh`. It documented that "Exp info in `Imaging_Exp_info.npy`: dict mapping exp_type -> list of session dicts", that the neural file is `{'spks': [plane0, plane1, ...]}` and that `load_spk` concatenates planes. Retinotopy is loaded *before* the spikes so that the neuron mask can be applied per plane, "avoids large intermediate array" (Step 6 notes). Loading one behavior file per session (rather than grouping sessions by file) was not discussed; the AI measured the whole quartile pass at 7.5 s and considered it negligible.

## 1-b. How are the data split into subjects (mice)?

i. The mouse identity is taken directly from `mname` in the `Imaging_Exp_info.npy` entry and carried on each session record. Subjects are collected in order of first appearance while iterating over the sorted session keys, and `subject_idx` is that index per session. Result: 19 subjects, matching the paper. No subject is excluded.

ii.
```python
mname, datexp, blk = db['mname'], db['datexp'], db['blk']
```
```python
if mname not in subjects_seen:
    subjects_seen[mname] = len(subjects_seen)
...
subject_idx_list.append(subjects_seen[mname])
...
subjects = sorted(subjects_seen.keys(), key=lambda x: subjects_seen[x])
'subject_idx': np.array(subject_idx_list, dtype=np.int64),
```

iii. CONVERSION_NOTES Step 3/4: "19 mice" from the paper ("89 recordings in 19 mice"), and the AI verified 19 unique `mname` values in `exp_info` and in the spk filenames. No derivation is needed because the index names the mouse.

## 1-c. How are the data split into sessions?

i. A session is one physical recording = one spike file = `mname_datexp_blk`. Because `exp_info` lists 142 entries for 89 recordings (the same recording appears under several experiment types, sometimes with a `stimtype` suffix), the AI de-duplicates on `spk_key`, preferring an entry that has no `stimtype` (the last such entry encountered wins). This gives exactly 89 sessions, one per file in `data/spk`. The behavior key is `spk_key` plus `_<stimtype>` when the chosen entry has one.

ii.
```python
spk_key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
beh_key = f"{spk_key}_{ndb['stimtype']}" if 'stimtype' in ndb else spk_key
# Prefer entries without stimtype
if spk_key not in session_map or 'stimtype' not in ndb:
    session_map[spk_key] = {'exp_type': exp_type, 'beh_key': beh_key, 'db': dict(ndb)}
```

iii. CONVERSION_NOTES Step 4, "Key Insight": "Sessions can appear in multiple experiment types (142 total entries for 89 unique sessions). For the decoder, each physical recording (neural data file) is used once. The behavioral data is the same regardless of which experiment type references it. For sessions with stimtype variants (swap1/swap2), the underlying trial data is identical — only stim_id mapping differs." The AI explicitly verified (trajectory step 51) that the behavior arrays of duplicate entries are the same.

## 1-d. How are the data split into trials?

i. Trials are defined by consecutive corridor-entry frames: trial *i* spans frames `StartFr[i]` (truncated to int) up to `StartFr[i+1]`, and the last trial runs to the last imaged frame. This deliberately includes the 2 m grey space / inter-trial interval as part of the trial, so a trial is a full corridor-to-corridor cycle rather than only the 4 m textured traversal. Frames are not selected with `ft_trInd`/`ft_CorrSpc`, and no frames are dropped inside the window (stationary periods included). Consequences: mean T = 59.6 frames vs 32.6 in the reference, max T = 5621 frames (~30 min), and ~32% of all bins lie in the grey space. Total 38,110 trials.

ii.
```python
for i in range(ntrials):
    # Trial frames: StartFr[i] to StartFr[i+1] or end
    start = StartFr[i]
    end = StartFr[i + 1] if i < ntrials - 1 else nfr_use
    start = max(0, start)
    end = min(nfr_use, end)
    n_frames = end - start
    if n_frames < 2:
        continue
    trial_spk = spk[:, start:end].copy()
```
```python
StartFr = beh['StartFr'].astype(int)
```

iii. CONVERSION_NOTES Step 5, Key Decision 2: "Trial length: Use frames from StartFr to start of next trial (or end of session). This captures corridor + gray space." The trajectory (step 45) shows the reasoning: the alignment event required is corridor entry, and the AI chose to keep "frames from corridor entry through gray space" so that the trial covers all of the inter-corridor period rather than only the traversal, noting the reference code instead interpolates activity onto 60 position bins and therefore does not define a time window at all.

## 1-e. How are trials filtered based on quality controls?

i. Almost none. A trial is dropped only if its window has fewer than 2 frames, and a session is dropped only if fewer than 2 trials survive (never triggered — all 89 sessions are kept). No filter is applied on trial duration, on stationary periods, on running (`ft_move` / `ft_isMoving`), or on missing behavior. Very long "trials" (a mouse standing in the corridor/grey space, up to 5621 frames) are retained.

ii.
```python
    if n_frames < 2:
        continue
...
if len(neural_trials) < 2:
    print(f"WARNING: <2 valid trials for {spk_key}")
    return None
```

iii. Trajectory step 96: "Let me also check if very long trials are a real issue and whether we should cap them. Looking at the reference code, they don't cap trial length — they use position interpolation which naturally handles variable-length trials. For the decoder, very long trials (where the mouse is stationary) add a lot of data but little information. Let me keep them for now but check if it causes problems." CONVERSION_NOTES Step 3 curation rules: "No explicit trial filtering in reference code for basic loading … For decoder: use all trials (no filtering)." The follow-up check the AI promised was never performed/documented.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spks` in `spk/<mname>_<datexp>_<blk>_neural_data.npy` — a list of (neurons × frames) arrays, one per imaging plane, concatenated along the neuron axis. The per-neuron area label `iarea` comes from `retinotopy/<mname>_<datexp>_trans.npz` and is used only for masking/labelling, not for computing activity.

ii.
```python
spk_data = np.load(os.path.join(DATA_ROOT, 'spk', fn), allow_pickle=True).item()
planes = spk_data['spks']
...
    filtered.append(plane[plane_mask].astype(np.float16))
return np.concatenate(filtered, 0)
```
```python
dtrans = np.load(os.path.join(DATA_ROOT, 'retinotopy', fn), allow_pickle=True)
return dtrans['iarea']
```

iii. CONVERSION_NOTES Step 1: "Neural data: Suite2p deconvolved calcium traces (NOT raw fluorescence). Deconvolution with tau=0.75 s", and "`load_spk` concatenates planes: `np.concatenate([nspk for nspk in spks], 0)`" — i.e. the same variable and the same concatenation as the reference `utils.load_spk`.

## 2-b. How is the `neural` data processed?

i. No processing at all: no dF/F, no deconvolution, no normalisation, no smoothing, no spatial/temporal rebinning. The traces are cast to `float16` at load time and each trial stores the raw columns `spk[:, start:end]` (a copy). Trials keep their own length; nothing is padded or truncated to a common window.

ii.
```python
filtered.append(plane[plane_mask].astype(np.float16))
...
trial_spk = spk[:, start:end].copy()  # copy to avoid referencing large array
```

iii. CONVERSION_NOTES Step 3: "Neural data is deconvolved calcium traces from Suite2p (already in data files); No additional delta F/F computation needed" — the methods state "All our analyses were based on deconvolved fluorescence traces". `float16` was chosen purely for size: the trajectory (steps 69–87) shows the AI measured 7.82 GB for two sessions in float32 and halved it, estimating ~174 GB for the full dataset ("float16 reduced file size from 7.82 GB to 3.91 GB (exactly half as expected)").

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only curation is by visual area: a neuron is kept if its retinotopy code `iarea` is not in `{-1, 7}` (i.e. it lies in one of V1, mHV, lHV, aHV), and the kept neurons get a `brain_region_idx` via the `iarea` → region map (V1=8; mHV=0,1,2,9; lHV=5,6; aHV=3,4). No additional quality/selectivity filter (e.g. the paper's d′ ≥ 0.3 criterion) is applied, and no activity-based filter. The mask is applied per plane before concatenation. Result: 4,105,393 neurons kept (V1 1,833,035 / mHV 1,108,860 / lHV 495,318 / aHV 668,180) — identical to the reference.

ii.
```python
AREA_MAP = {8: 'V1', 0: 'mHV', 1: 'mHV', 2: 'mHV', 9: 'mHV',
            5: 'lHV', 6: 'lHV', 3: 'aHV', 4: 'aHV'}
EXCLUDED_AREAS = {-1, 7}  # Outside visual cortex

def get_brain_region_idx(iarea):
    valid_mask = np.array([int(ia) not in EXCLUDED_AREAS for ia in iarea])
    region_idx = np.array([
        BRAIN_REGIONS.index(AREA_MAP.get(int(ia), 'V1'))
        for ia in iarea[valid_mask]
    ], dtype=np.int64)
    return valid_mask, region_idx
```

iii. CONVERSION_NOTES Step 1/3: "Brain area assignment via `neu_area_ID(iarea)` … Excluded: iarea==-1 (outside visual cortex) and iarea==7", matching the reference code's `(arid!=-1) & (arid!=7)`. Step 10: the AI justified the lower neuron counts (17,363–78,815 vs the paper's 20,547–89,577) by this exclusion: "This is correct — paper says 'inside visual cortex' only." The Suite2p cell classifier has already curated cells, so no further filter is applied.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's first column is the frame `int(StartFr[i])`, i.e. corridor entry (the required alignment event), and `metadata['temporal_alignment_event'] = 'corridor entry (trial start)'`, `off_start = 0.0`, `off_end = None`. Trials are variable length; the window ends at the next corridor entry, so it covers the traversal *and* the grey space. Because `StartFr` is fractional and is truncated with `astype(int)`, the first bin of a trial is the frame immediately *before* corridor entry (verified on `DR10_2022_07_12_1`, trial 5: the AI window starts at frame 262 while the first frame with `ft_CorrSpc` is 263) — a sub-bin (≤315 ms) offset that is consistent across trials.

ii.
```python
StartFr = beh['StartFr'].astype(int)
...
start = StartFr[i]
end = StartFr[i + 1] if i < ntrials - 1 else nfr_use
trial_spk = spk[:, start:end].copy()
```
```python
'temporal_alignment_event': 'corridor entry (trial start)',
'off_start': 0.0,
'off_end': None,
```

iii. CONVERSION_NOTES Step 5: "Temporal alignment: Trials need to be aligned to corridor entry (StartFr)"; all streams (neural, inputs, outputs) are indexed with the same `start:end` frame window, so they are aligned by construction. The AI kept variable-length trials rather than a fixed window because the format only requires a common *bin size*, and it verified in Step 10 that `time_since_trial_start` starts at 0 for every trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One bin = one imaging frame; no rebinning, resampling or interpolation. The metadata bin size is the fixed 1000/3.17 = 315.46 ms from the reference notebook. (A per-session frame rate `fs` is estimated from the median `diff(ft)` of the first 1000 frames, but it is used only to convert frame counts into seconds for the two time inputs, not to rebin.)

ii.
```python
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE  # ~315.5 ms
...
ft = beh['ft']
dt = np.median(np.diff(ft[:min(1000, len(ft))])) * 86400  # days->seconds
fs = 1.0 / dt if dt > 0 else FRAME_RATE
...
'time_bin_size': TIME_BIN_MS,
```

iii. CONVERSION_NOTES Step 5, Key Decision 1: "Time bin size: Use native frame rate (~315 ms). No resampling needed." Step 4 records the consistency check: code says ~3.17 Hz, data gives median dt = 0.3146 s → 3.178 Hz. The AI noted that the reference code interpolates activity onto 60 position bins for its own analyses, but that the decoder format requires time bins, so it kept the native imaging grid on which all behavior streams already live.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. `SoundFr` (the fractional imaging-frame number of the sound cue on each trial) and the frame index, converted to seconds with the session frame rate `fs` (derived from `ft`, the MATLAB datenum timestamp of each frame).

ii.
```python
SoundFr = beh['SoundFr']
...
sound_fr = SoundFr[i]
...
    time_to_sound = ((sound_fr - frame_idx) / fs).astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 variable mapping: "Time to SoundFr → input[0]: time_to_sound_cue, (SoundFr - current_frame) / fs". The sound cue is only available as a frame number, so the frame axis is the natural reference.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For every frame of the trial window, `(SoundFr[i] - frame_index) / fs`, in seconds — positive before the cue, zero at the cue, negative after it (the "time *to* the cue" convention, same sign as the reference). Frame differences are multiplied by a constant per-session period rather than differencing the actual `ft` timestamps. If `SoundFr` were NaN the whole trial would be filled with zeros (dead code: no NaN `SoundFr` exists in the dataset — checked across all 142 behavior entries). Because trial windows extend through the grey space and through long stationary periods, the values reach ±1766 s (reference: ±73 s).

ii.
```python
frame_idx = np.arange(start, end)
sound_fr = SoundFr[i]
if np.isnan(sound_fr):
    time_to_sound = np.zeros(n_frames, dtype=np.float32)
else:
    time_to_sound = ((sound_fr - frame_idx) / fs).astype(np.float32)

inp = np.stack([
    time_to_sound,
    np.full(n_frames, day, dtype=np.float32),
    (np.arange(n_frames) / fs).astype(np.float32),
    np.full(n_frames, float(isRew[i]), dtype=np.float32),
], axis=0)
```

iii. CONVERSION_NOTES Step 5 maps it as `(SoundFr - current_frame)/fs` (the same table describes the sign as "Negative before cue … positive after", which is the opposite of what the code computes — a documentation slip, not a code one). The task lists *time to sound cue* as a continuous, time-varying input, so it is stored as a per-bin time difference rather than as a binary event marker.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on exactly the same frame indices `np.arange(start, end)` used to slice the neural matrix of that trial, so it is aligned bin-for-bin and has the same length. `SoundFr` is already expressed in imaging-frame units, so no cross-stream resampling is needed.

ii.
```python
frame_idx = np.arange(start, end)
time_to_sound = ((sound_fr - frame_idx) / fs).astype(np.float32)
...
trial_spk = spk[:, start:end].copy()
```

iii. All behavioral and neural streams in this dataset are indexed by imaging frame (CONVERSION_NOTES Step 2 lists the `ft_*` frame-level arrays and the `*Fr` per-trial frame markers), so using one window index for every stream guarantees alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the `Imaging_Exp_info.npy` metadata dict of whichever entry was chosen for the session: the field `days` if present, otherwise the field `sess#`, otherwise 0. It is **not** derived from the recording date, and sessions are not ordered within a mouse. In practice `sess#` is a before/after-learning flag (0 or 1) that is present for most entries, and `days` (an actual training-day count, e.g. 6, 9, 13, 15) appears for only a handful, so the delivered values are mostly 0 or 1 with a few large numbers, range [0, 15], and several sessions of the same mouse share the same value.

ii.
```python
def get_session_day(db):
    """Extract day of training from session metadata."""
    for key in ['days', 'sess#']:
        if key in db:
            return int(db[key])
    return 0
...
day = np.float32(get_session_day(db))
```

iii. CONVERSION_NOTES Step 5, Key Decision 8: "Day of training: Use sess# or days field from exp_info if available; otherwise use session chronological order within subject." The documented fallback to chronological order within a subject was never implemented — the code falls back to 0. The AI preferred the metadata fields because it read them as the experiment's own record of training stage.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The integer is cast to `float32` and broadcast with `np.full` across every bin of every trial of the session (a per-trial constant expressed as a time series, as required by the format). No per-mouse re-indexing, no ordering by date, no normalisation.

ii.
```python
day = np.float32(get_session_day(db))
...
np.full(n_frames, day, dtype=np.float32),
```

iii. The task lists *day of training* as a continuous per-trial input; the AI stored it as a constant row so that all four inputs share the `(d_input, n_timepoints)` layout.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `StartFr` (corridor entry frame of each trial) — implicitly, since the trial window begins at `int(StartFr[i])` — and the session frame rate `fs` derived from `ft`.

ii.
```python
StartFr = beh['StartFr'].astype(int)
start = StartFr[i]
...
(np.arange(n_frames) / fs).astype(np.float32),
```

iii. CONVERSION_NOTES Step 5: "Time since StartFr → input[2]: time_since_trial_start, (current_frame - StartFr)/fs, Continuous, starts at 0."

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Elapsed time within the window: bin *k* gets `k / fs` seconds, so the value is exactly 0.0 in the first bin of every trial and increases linearly. Because the window start is `floor(StartFr[i])` rather than the fractional entry frame, the zero point can precede true corridor entry by up to one bin. Values run up to 1768 s in the longest (stationary) trial.

ii.
```python
inp = np.stack([
    time_to_sound,
    np.full(n_frames, day, dtype=np.float32),
    (np.arange(n_frames) / fs).astype(np.float32),
    np.full(n_frames, float(isRew[i]), dtype=np.float32),
], axis=0)
```

iii. The instruction asks for a continuous, time-varying "time since trial start" with the trial aligned to corridor entry; the AI's window already starts at corridor entry, so the elapsed-frame count converted to seconds is the direct implementation. The AI explicitly re-verified in Step 10 that "time_since_trial_start correctly starts at 0 for all trials" after first mis-indexing the check.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same window, same length: it is built from `np.arange(n_frames)` where `n_frames = end - start` is exactly the number of neural columns in that trial, so bin *k* of the input corresponds to bin *k* of the neural matrix.

ii.
```python
n_frames = end - start
trial_spk = spk[:, start:end].copy()
...
(np.arange(n_frames) / fs).astype(np.float32),
```

iii. As with all other streams: everything is indexed on the common imaging-frame grid with one `start:end` slice per trial.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, the per-trial boolean flag marking trials run in the rewarded corridor.

ii.
```python
isRew = beh['isRew']
...
np.full(n_frames, float(isRew[i]), dtype=np.float32),
```

iii. CONVERSION_NOTES Step 5 mapping table: "isRew → input[3]: reward_availability, Binary 0/1, Per-trial scalar." The AI checked in Step 10 that reward availability is non-zero only in the supervised sessions (TX108, TX109, TX60, TX61, VR2) and 0 throughout the unsupervised/naive sessions, which matches the paper's design.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. None beyond a cast to float and a broadcast across the trial's bins.

ii.
```python
np.full(n_frames, float(isRew[i]), dtype=np.float32),
```

iii. The flag is already the required binary per-trial variable; unsupervised and naive mice ran the same corridors with no water, so their value is 0 everywhere.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, the per-trial name of the wall texture (e.g. `circle1`, `leaf2`, `rock1`, `wood1_swap2`). `TrialStim` / `stim_id` are not used.

ii.
```python
WallName = beh['WallName']
...
stim = standardize_stim_name(str(WallName[i]))
...
stim_names.append(stim)
```

iii. `WallName` is the only field naming the texture in every session (the AI noted that `stim_id`/`TrialStim` are per-session index mappings and are masked/NaN in the swap sessions, trajectory step 223–224).

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Names are passed through a hard-coded standardisation table that maps the rock/brick cohort's textures onto the equivalent circle/leaf names by their role in the experiment (`rock1→circle1`, `rock2→circle2`, `wood1→leaf1`, `wood2→leaf2`, `wood5→leaf3`, `wood1_swap1/2→leaf1_swap1/2`, `brick*→circle*`), while different frozen crops of the same texture stay distinct. After all sessions are processed the surviving names are sorted and encoded as indices, and the per-trial index is broadcast over all bins of the trial. Result: 8 categories (`circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3`) versus 4 base textures in the reference. The role mapping agrees with the reference code's own `stim_id` convention (verified in the raw data: in rock/wood sessions `rock1` has `stim_id` 0 = the `circle1` slot, `wood1` has 2 = `leaf1`, `wood5` has 4 = `leaf3`). One exception exists in the data: in `TX88_2022_06_20_1`, `rock1` is used as a novel test stimulus (`stim_id` 4) *alongside* real `circle1` trials, so the fixed table merges two physically different textures into one label in that single session.

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

def standardize_stim_name(name):
    return STIM_CATEGORY_MAP.get(name, name)
```
```python
all_stim_sorted = sorted(all_stim_names)
stim_to_idx = {s: i for i, s in enumerate(all_stim_sorted)}
for si in range(len(output_all)):
    for ti in range(len(output_all[si])):
        output_all[si][ti][0, :] = stim_to_idx[all_stim_per_session[si][ti]]
```

iii. The instruction the AI received gave the example "circle1, leaf2", so crops are kept as separate categories. For the cross-cohort merge, the trajectory (steps 221–228) reasons from the methods — "each mouse was trained with rewards on one random pair of stimuli, such as leaf–circle or rock–brick" — and from the reference code's use of `stim_id`: "For rock/wood-trained mice, the equivalent would be rock1→circle1, rock2→circle2, wood1→leaf1, wood2→leaf2, wood5→leaf3." `circle3` was checked and kept as a genuine separate stimulus. The `wood5→leaf3` entry was added after the full run and patched into the saved pickle rather than re-running the conversion (so `verification_full_out.txt` still shows the pre-patch 9 categories while the trained data has 8).

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the (fractional) imaging-frame number of every lick in the session.

ii.
```python
lick_binary = np.zeros(nfr_use, dtype=np.int64)
if 'LickFr' in beh and len(beh['LickFr']) > 0:
    lick_fr = beh['LickFr'].astype(int)
```

iii. CONVERSION_NOTES Step 5: "LickFr/LickTrind → output[1]: licking, Binary per frame, Time-varying." `LickFr` is already on the imaging-frame grid, so no conversion from timestamps is needed.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A session-length binary vector is built: the lick frame numbers are truncated to integers and those frames are set to 1, everything else stays 0 — i.e. "at least one lick in this bin". Licks at frames outside the imaged range (or negative) are discarded. The trial's slice of this vector is stored. Overall 2.7% of bins are licks, and licking is non-zero only in the supervised sessions.

ii.
```python
lick_binary = np.zeros(nfr_use, dtype=np.int64)
if 'LickFr' in beh and len(beh['LickFr']) > 0:
    lick_fr = beh['LickFr'].astype(int)
    valid_lick = (lick_fr >= 0) & (lick_fr < nfr_use)
    lick_binary[lick_fr[valid_lick]] = 1
...
    lick_binary[start:end],
```
```python
'output_values': [..., ['no_lick', 'lick'], ...]
```

iii. The task asks for binary, time-varying licking (0 = not licking, 1 = licking), so the event list is rasterised onto the frame grid. The AI validated this in the sample run (Step 7/8: "Licking only in supervised session (TX108)", 3.2% lick frames) and plotted a lick raster in `processing_session_*.png`.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` indexes imaging frames, so the binary vector already shares the neural grid; the trial's licking row is the same `start:end` slice used for the neural columns, hence identical length and bin-for-bin correspondence.

ii.
```python
trial_spk = spk[:, start:end].copy()
...
out = np.stack([
    np.full(n_frames, 0, dtype=np.int64),  # placeholder for stim idx
    lick_binary[start:end],
    pos_bins[start:end],
    speed_bins[start:end],
], axis=0)
```

iii. Same rationale as the other streams: one frame window per trial applied to every stream.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the per-frame position in the corridor in decimeters (0–40 across the textured corridor, 40–60 in the grey space), together with `ft_CorrSpc`, the per-frame flag marking frames inside the textured corridor.

ii.
```python
ft_Pos = beh['ft_Pos'][:nfr_use]
ft_CorrSpc = beh['ft_CorrSpc'][:nfr_use].astype(bool)
```

iii. CONVERSION_NOTES Step 2 documents "`ft_Pos`: position in corridor (0-60 decimeters)" and "`ft_CorrSpc`: True when in texture corridor (0-4 m)"; Step 5 maps `ft_Pos → output[2]`.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. A session-length integer vector is built, initialised to category 4 ("gray"), and each corridor frame is assigned to one of four 10-decimeter (1 m) bins by its position; corridor frames at or beyond 40 dm are clamped into the last bin. Frames outside the corridor (the grey space, i.e. `ft_CorrSpc == False`) keep the value 4. The trial's slice of that vector is stored.

ii.
```python
# Position bins: 4 texture bins (1m each) + gray space
pos_bins = np.full(nfr_use, 4, dtype=np.int64)  # default: gray
for b in range(4):
    mask = ft_CorrSpc & (ft_Pos >= b * 10) & (ft_Pos < (b + 1) * 10)
    pos_bins[mask] = b
pos_bins[ft_CorrSpc & (ft_Pos >= 40)] = 3  # edge case
```

iii. CONVERSION_NOTES Step 5, Key Decision 4 records the AI arguing with itself: "the decoder asks for 4 bins, so gray space frames get a separate bin value? Re-reading: 'Position in corridor discretized into 4 equal-length, 1-m-long spatial bins'. So 4 bins covering 0-4 m. Gray space frames need a 5th category or can be excluded. I'll use 5 categories: 4 spatial + 1 gray." The grey category exists only because the trial window (decision 1-d) extends into the inter-trial grey space.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Fixed thresholds on `ft_Pos` at 0/10/20/30/40 decimeters, i.e. four equal 1-m bins as specified, plus a fifth "gray" category for every bin outside the textured corridor. `output_values[2] = ['0-1m','1-2m','2-3m','3-4m','gray']`. The resulting distribution is 19.4 / 15.9 / 16.0 / 16.7 / 32.0 %, so nearly a third of all labelled bins fall in the extra category and chance level for this output is 1/5 rather than 1/4.

ii.
```python
'output_values': [
    all_stim_sorted,
    ['no_lick', 'lick'],
    ['0-1m', '1-2m', '2-3m', '3-4m', 'gray'],
    ['Q1_slow', 'Q2', 'Q3', 'Q4_fast'],
],
```

iii. CONVERSION_NOTES Step 10: "Position bins: ~19% each for 4 texture bins + 32% gray space. Gray fraction is higher because mice spend more time in gray space between corridors." The AI treated the extra class as a necessary consequence of keeping the full corridor-to-corridor cycle in each trial rather than as a deviation to be removed.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` has one value per imaging frame and is truncated to the imaged frames; the trial row is the same `start:end` slice as the neural columns, so it is aligned bin-for-bin.

ii.
```python
ft_Pos = beh['ft_Pos'][:nfr_use]
...
    pos_bins[start:end],
```

iii. Same as all streams: one frame window per trial. The `--show-processing` plots include the per-trial output traces and the position histogram as a visual check.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed of the mouse at each imaging frame.

ii.
```python
ft_RunSpeed = beh['ft_RunSpeed'][:nfr_use]
...
speeds = beh['ft_RunSpeed']
```

iii. CONVERSION_NOTES Step 5: "`ft_RunSpeed` → output[3]: running_speed_bin, Quartile discretization across all frames, Time-varying."

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A single global set of three thresholds is computed once for the whole run, before any session is processed: all sessions' `ft_RunSpeed` arrays (all frames of the session, not only frames inside trials) are concatenated, frames with speed ≤ 0 are **excluded**, and the 25th/50th/75th percentiles of the remaining positive speeds are taken — [6.91, 22.18, 39.23]. These thresholds are then applied to every frame of every session, including the zero-speed frames.

ii.
```python
def collect_speed_quartiles(session_map, keys):
    """Collect running speed quartiles from behavioral data only (no neural loading)."""
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

iii. CONVERSION_NOTES Step 5, Key Decision 5: "Running speed: Discretize into 4 quartile bins computed across ALL running frames in the dataset." Computing them from behavior only was an efficiency choice ("no neural loading", 7.5 s for all 89 sessions). Using one global set of thresholds rather than per-session ones keeps the bin meaning comparable across sessions.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Hard thresholds: bin 0 for speed < 6.91, 1 for ≥ 6.91, 2 for ≥ 22.18, 3 for ≥ 39.23, labelled `['Q1_slow','Q2','Q3','Q4_fast']`. Because the thresholds were derived from positive speeds only while all frames (including the large mass of exactly-zero frames) are binned, the classes are **not** 25% each: the delivered distribution is 47.1 / 17.6 / 17.6 / 17.6 %, and in individual sessions bin 0 reaches 87%. The instruction asked for "4 bins, each corresponding to 25% of the data".

ii.
```python
# Speed bins: quartiles
speed_bins = np.zeros(nfr_use, dtype=np.int64)
speed_bins[ft_RunSpeed >= speed_quartiles[0]] = 1
speed_bins[ft_RunSpeed >= speed_quartiles[1]] = 2
speed_bins[ft_RunSpeed >= speed_quartiles[2]] = 3
```

iii. The AI found the imbalance (trajectory steps 190–193) and decided it was acceptable: "The speed quartiles are computed on `speeds > 0` … but the binning is applied to ALL frames including non-running (speed=0). So all stopped frames fall into Q1_slow. … This is actually correct behavior — frames where the mouse is not running get the lowest speed bin." CONVERSION_NOTES Step 10 repeats this: "Speed quartile Q1 contains 47.1% of frames … This is correct since quartiles are computed on positive speeds only." No rank-based split or zero-tie handling was attempted.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` gives one value per imaging frame and is truncated to the imaged frames; the binned vector is sliced with the same `start:end` window as the neural columns.

ii.
```python
ft_RunSpeed = beh['ft_RunSpeed'][:nfr_use]
...
    speed_bins[start:end],
```

iii. Same as all streams: one frame window per trial on the shared imaging-frame grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Four guards: (1) the behavior and the imaging can have different lengths, so everything is cut to `nfr_use = min(n_spike_frames, len(ft))` and the spike matrix is truncated to the same length; (2) licks whose frame index is negative or past the last imaged frame are dropped, and a session with an empty `LickFr` is handled explicitly; (3) a NaN `SoundFr` would produce an all-zero cue input (no such case exists in the data); (4) trials shorter than 2 frames are skipped and sessions with fewer than 2 surviving trials are skipped with a warning (never triggered). Trial windows are additionally clipped to `[0, nfr_use]`. A per-session frame rate is estimated from the data with a fallback to 3.17 Hz if `dt` is non-positive. There is no try/except around a session, so an unexpected failure would abort the whole run.

ii.
```python
nfr_use = min(nfr, len(beh['ft']))
spk = spk[:, :nfr_use]
ft_Pos = beh['ft_Pos'][:nfr_use]
ft_CorrSpc = beh['ft_CorrSpc'][:nfr_use].astype(bool)
ft_RunSpeed = beh['ft_RunSpeed'][:nfr_use]
```
```python
if 'LickFr' in beh and len(beh['LickFr']) > 0:
    lick_fr = beh['LickFr'].astype(int)
    valid_lick = (lick_fr >= 0) & (lick_fr < nfr_use)
    lick_binary[lick_fr[valid_lick]] = 1
```
```python
start = max(0, start); end = min(nfr_use, end)
if n_frames < 2: continue
...
if len(neural_trials) < 2:
    print(f"WARNING: <2 valid trials for {spk_key}")
    return None
fs = 1.0 / dt if dt > 0 else FRAME_RATE
```

iii. CONVERSION_NOTES Step 1 records that the reference code does the same truncation (`beh[...][:nfr]` with `nfr = spk.shape[1]`). The AI described the dataset as clean and the format-verification run reported no errors and no warnings. Step 10 notes no missing-data issues beyond these.

## 12-a. What are the most time-consuming steps of the code?

i. Reading and concatenating the spike files: ~11–29 s per session out of ~13–30 s total per session, dominated by disk I/O of ~400 GB of deconvolved traces. The whole conversion took 1870 s (~31 min); writing the 177 GB pickle is the second cost. The behavior-only quartile pass over all 89 sessions is 7.5 s, negligible. The AI printed per-session timings, so the breakdown is in `conversion_full_out.txt`.

ii.
```python
spk_data = np.load(os.path.join(DATA_ROOT, 'spk', fn), allow_pickle=True).item()
...
return np.concatenate(filtered, 0)
```
```python
print(f"{result['nneu']}({result['nneu_total']}) neurons, {result['ntrials']} trials, {time.time()-t_s:.1f}s")
...
print(f"  Size: {fsize:.2f} GB, total time: {time.time()-t0:.1f}s", flush=True)
```

iii. CONVERSION_NOTES Step 7 run-time table attributes ~15–25 s/session to "Neural loading+filtering" and ~5 s to processing, and Step 6 lists the mitigation actually applied: masking neurons per plane and casting to float16 *before* concatenation, which avoids materialising the full-precision, full-population array ("Down from 75 s to 57 s" for the 2-session sample). The AI concluded the remainder is I/O-bound.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain: (1) `get_brain_region_idx` runs two Python list comprehensions over every neuron of the session (50k–90k elements, with an `int()` call and a `list.index()` per neuron) where `np.isin` on the whole `iarea` array would do; (2) the per-plane masking loop in `load_spk_filtered` (unavoidable given the per-plane storage, and deliberate — it is what keeps peak memory down); (3) the per-trial Python loop that slices and stacks each trial (intrinsic to the ragged output format). The stimulus re-encoding pass at the end is also a double Python loop over all 38,110 trials. None of these is significant next to the file I/O.

ii.
```python
valid_mask = np.array([int(ia) not in EXCLUDED_AREAS for ia in iarea])
region_idx = np.array([
    BRAIN_REGIONS.index(AREA_MAP.get(int(ia), 'V1'))
    for ia in iarea[valid_mask]
], dtype=np.int64)
```
```python
for plane in planes:
    n = plane.shape[0]
    plane_mask = valid_mask[offset:offset+n]
    filtered.append(plane[plane_mask].astype(np.float16))
```
```python
for si in range(len(output_all)):
    for ti in range(len(output_all[si])):
        output_all[si][ti][0, :] = stim_to_idx[all_stim_per_session[si][ti]]
```

iii. CONVERSION_NOTES Step 6 lists only the optimisations that were made (per-plane filtering, float16, behavior-only quartiles); the remaining per-neuron loops were not identified as vectorisation targets, consistent with the AI's conclusion that the conversion is I/O-bound and already inside its time budget.

## 12-c. What processing does the code repeat multiple times?

i. Behavior files are loaded from disk twice per session and once per session that shares the file: `collect_speed_quartiles` calls `load_beh` for every session (which `np.load`s the entire multi-session `Beh_<exp_type>.npy` each time), and `process_session` then calls `load_beh` again for the same session. With 89 sessions drawn from 23 behavior files totalling 6.6 GB, each file is re-read on the order of four to ten times; the reference instead groups sessions by behavior file and reads each once. Minor repeats: `int(ia)` is computed twice per neuron (once in each comprehension); the stimulus row of every output array is written twice (a zero placeholder during processing, then the real index in the final pass).

ii.
```python
def collect_speed_quartiles(session_map, keys):
    for spk_key in keys:
        info = session_map[spk_key]
        beh = load_beh(info)          # loads the whole Beh_<exp_type>.npy
```
```python
    # Load behavior
    beh = load_beh(session_info)      # loads the same file again
```
```python
out = np.stack([
    np.full(n_frames, 0, dtype=np.int64),  # placeholder for stim idx
    ...
```

iii. Not discussed in CONVERSION_NOTES. The quartile pass was timed at 7.5 s total and treated as negligible ("Speed quartile collection from behavior only (no neural data loading)"), and the placeholder/second-pass design was chosen so that the global stimulus vocabulary is known before indices are assigned — the AI restructured the code for exactly this reason at trajectory step 62 ("I'm loading neural data twice per session … let me fix this by storing stim_names").

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several small ones and one large one. Small: frame-level `lick_binary`, `pos_bins` and `speed_bins` are built for the whole session including frames before the first corridor entry that no trial ever uses; `trial_spk = ...copy()` duplicates every trial's neural block (extra time and peak memory, since the session array is dropped anyway); the placeholder stimulus row is written then overwritten; `nneu_total` is tracked only for printing; the `--show-processing` plots reload and concatenate all trials of a session. Large: every trial stores all 17k–79k kept neurons (177 GB), but the decoder's own pipeline random-projects to 2000 neurons and then to 100 PCs, and the AI had to subsample to 10,000 neurons per session at training time to fit in RAM — so most of what was written was discarded downstream. Roughly a third of the stored bins are grey-space frames that are outside the corridor the outputs are defined on.

ii.
```python
trial_spk = spk[:, start:end].copy()  # copy to avoid referencing large array
```
```python
lick_binary = np.zeros(nfr_use, dtype=np.int64)
pos_bins = np.full(nfr_use, 4, dtype=np.int64)
speed_bins = np.zeros(nfr_use, dtype=np.int64)
```
```python
# from run_decoder.py, used for Step 11
Subsampling neurons to max 10000 per session...
  Session 0: 52246 -> 10000 neurons
```

iii. CONVERSION_NOTES Step 11 justifies the subsampling rather than the storage: "The decoder uses random projection to 2000 dims → SVD to 100 PCs, so 10K neurons provides more than sufficient information." The AI considered storing fewer neurons or precomputing the PCA during Step 6 (trajectory steps 72–81) but rejected it because "the format spec requires storing all neurons", and confirmed there was enough disk (1.7 TB) and RAM, so the full population was kept deliberately.
