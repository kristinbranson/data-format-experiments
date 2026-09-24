# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything is read from the three subfolders of `/app/data`: `beh/` (behavior), `spk/` (deconvolved calcium traces) and `retinotopy/` (visual-area label per neuron). `beh/Imaging_Exp_info.npy` is read first as the master index; it is a dict of experiment type → list of recording entries (`mname`, `datexp`, `blk`, …), 23 experiment types and 142 entries total. Every `Beh_<exp_type>.npy` file is then loaded eagerly into one flat in-memory dict keyed by `{mname}_{datexp}_{blk}` (`load_all_behavior`), first-occurrence wins. Neural traces and retinotopy are loaded lazily, once per session, inside the processing loop (`load_spk`, `load_area_ids`). The script actually walks the sessions **twice**: pass 1 (`collect_speeds_from_behavior`) to pool running speeds for the global quartile boundaries, pass 2 (`process_session_full`) to build the arrays.

ii.
```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
all_beh = load_all_behavior(exp_info)
sessions = build_session_list(exp_info)
```
```python
def load_all_behavior(exp_info):
    all_beh = {}
    for exp_type in exp_info.keys():
        beh_path = os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy')
        if not os.path.exists(beh_path):
            continue
        beh = np.load(beh_path, allow_pickle=True).item()
        for k, v in beh.items():
            base_key = k
            for suffix in ('_swap1', '_swap2'):
                if base_key.endswith(suffix):
                    base_key = base_key[:-len(suffix)]
                    break
            if base_key not in all_beh:
                all_beh[base_key] = v
    return all_beh
```
```python
def load_spk(mname, datexp, blk, root=DATA_ROOT):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    data = np.load(os.path.join(root, 'spk', fn), allow_pickle=True).item()
    spk = np.concatenate([nspk for nspk in data['spks']], 0)
    return spk

def load_area_ids(mname, datexp, root=DATA_ROOT):
    fn = f'{mname}_{datexp}_trans.npz'
    trans = np.load(os.path.join(root, 'retinotopy', fn), allow_pickle=True)
    return trans['iarea']
```

iii. From CONVERSION_NOTES Step 1/2: the AI traced the reference `utils.py` helpers (`load_spk`, `load_retino`, `load_exp_beh`, `neu_area_ID`) and reproduced them, noting "Neural data format: `spks` list of arrays per plane, concatenated" and that `Imaging_Exp_info.npy` is the master index with "142 total entries, but only 89 unique sessions". Behavior files are loaded once each up front because "a behavior file holds several sessions" and because pass 1 only needs behavior. The `_swap1`/`_swap2` key stripping was added after the AI discovered (trajectory steps 99–106) that `*_test3` behavior files key swap sessions with a stimulus-type suffix that does not appear in the exp_info-derived session id.

## 1-b. How are the data split into subjects?

i. The mouse is taken directly from `mname` in each index entry and carried on the session record. `subjects` is the sorted set of unique `mname` values (19 mice) and `subject_idx` is each session's index into that list. No inference is needed because the index already names the animal.

ii.
```python
subject_list = sorted(set(s['mname'] for s in sessions))
...
subject_idx_list.append(subject_list.index(sess['mname']))
...
'subjects': subject_list,
'subject_idx': np.array(subject_idx_list, dtype=np.int64),
```

iii. CONVERSION_NOTES Step 2/3: "89 unique sessions across 19 mice matches paper" ("89 recordings in 19 mice"). The split is given by the data, so no decision beyond using `mname` verbatim.

## 1-c. How are the data split into sessions?

i. A session is one mouse on one date in one block. `build_session_list` builds the key `{mname}_{datexp}_{blk}` for every one of the 142 exp_info entries and de-duplicates with a dict, keeping the first occurrence and appending every experiment type it appears under to `exp_types`. That yields 89 unique sessions. Sessions are then sorted by `(mname, datexp)`. The same key names the spike file and (after suffix stripping) keys the behavior dict.

ii.
```python
def build_session_list(exp_info):
    session_dict = {}
    for exp_type, db_list in exp_info.items():
        for ndb in db_list:
            key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
            if key not in session_dict:
                session_dict[key] = {'key': key, 'mname': ndb['mname'],
                                     'datexp': ndb['datexp'], 'blk': ndb['blk'],
                                     'exp_types': []}
            session_dict[key]['exp_types'].append(exp_type)
    sessions = list(session_dict.values())
    sessions.sort(key=lambda s: (s['mname'], s['datexp']))
```

iii. CONVERSION_NOTES Step 4: "Sessions count | 142 exp_info entries | 89 unique sessions | 89 recordings | 33 sessions appear in >1 exp type, deduplicate". The AI also verified that "the same session across exp types" has (in its words) identical behavior, so any experiment type's behavior file can be used.

## 1-d. How are the data split into trials?

i. Trials come from the behavior labelling, not from re-segmentation. For each trial index `n` in `range(beh['ntrials'])` the frames kept are those the behavior labels with that trial **and** flags as inside the textured corridor: `ft_trInd == n & ft_CorrSpc`. Both streams are first truncated to the number of imaged frames `nfr`. Trials are therefore variable length (11 to 5607 frames, median ≈ 32) and cover corridor entry → end of the 4 m texture; the 2 m grey space is excluded. No frames are dropped from the middle of a trial.

ii.
```python
def extract_trial_frames(beh, nfr):
    ft_trInd = beh['ft_trInd'][:nfr]
    ft_CorrSpc = beh['ft_CorrSpc'][:nfr]
    ntrials = beh['ntrials']
    trials = []
    for n in range(ntrials):
        frames = np.where((ft_trInd == n) & ft_CorrSpc)[0]
        if len(frames) >= 2:
            trials.append(frames)
        else:
            trials.append(None)
    return trials
```

iii. CONVERSION_NOTES Key Decision 4: "Trial = corridor entry to gray space entry: Use frames where ft_trInd==n AND ft_CorrSpc==True. This covers the texture area where stimuli, cues, and rewards occur." Key Decision 2 explains why no within-trial frame filtering is applied: "The reference filters to running frames for d' analysis, but the decoder needs continuous time series. Non-running frames still carry information about licking, position, etc." The trajectory (steps 54–63) shows the AI seriously considered the reference's `VRmove = ft_move > 0` running filter and rejected it after measuring that "about 49% of corridor licks happen during running", i.e. a running-only filter would destroy half the licking output.

## 1-e. How are trials filtered based on quality controls?

i. The only trial-level filter is a minimum length of 2 corridor frames; trials with 0 or 1 frame become `None` and are skipped. No other curation is applied — no outlier-length filter, no session-level filter (no minimum trial count or neuron count per session). All 89 sessions and all 38,110 trials with ≥2 corridor frames are kept, including a 5607-frame (≈29 min) trial in `TX88_2022_07_19_1` in which the animal is essentially stationary.

ii.
```python
        frames = np.where((ft_trInd == n) & ft_CorrSpc)[0]
        if len(frames) >= 2:
            trials.append(frames)
        else:
            trials.append(None)
```
```python
    for n in range(ntrials):
        frames = trial_frames_list[n]
        if frames is None:
            continue
```

iii. CONVERSION_NOTES Step 3 "Trial curation rules": "No explicit trial filtering in reference code; All trials used (both rewarded and unrewarded)." Step 10 issue 4 documents the outlier explicitly and dismisses it: "Max trial length 5607 frames: One outlier trial in session 80 (TX88). Animal likely stopped in corridor for extended period. **Not filtered per reference code convention.**" The ≥2-frame requirement is documented in Step 6 as "Trial = corridor frames (ft_CorrSpc==True) per trial index, minimum 2 frames".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/{mname}_{datexp}_{blk}_neural_data.npy` — a list of (neurons × frames) arrays, one per imaging plane, concatenated along the neuron axis. The region label of each neuron comes from `iarea` in `retinotopy/{mname}_{datexp}_trans.npz`.

ii.
```python
    data = np.load(path, allow_pickle=True).item()
    spk = np.concatenate([nspk for nspk in data['spks']], 0)
```
```python
    trans = np.load(path, allow_pickle=True)
    return trans['iarea']
```

iii. CONVERSION_NOTES Step 1: "Data is deconvolved traces from Suite2p (timescale of decay 0.75s). **No dF/F computation needed.**" This mirrors the reference `utils.load_spk`, which concatenates planes the same way.

## 2-b. How is the `neural` data processed?

i. Not processed at all. The columns of the concatenated `spks` array belonging to a trial's frames are sliced out and cast to `float32`. Trials keep their native, variable length; nothing is padded, smoothed, normalised, z-scored or position-interpolated.

ii.
```python
    spk = load_spk(mname, datexp, blk)
    nneu, nfr = spk.shape
    iarea = load_area_ids(mname, datexp)
    neuron_mask, region_idx = get_neuron_mask_and_regions(iarea)
    spk_filtered = spk[neuron_mask]
    ...
        neural = spk_filtered[:, frames].astype(np.float32)
```

iii. The data are already deconvolved, so no further processing is justified (Step 1). Key Decision 1 rejects the reference's position-interpolation pipeline: "The decoder requires time-varying signals with fixed time bins. Position interpolation (as in reference) creates position-based, not time-based data. We use raw frame-rate data aligned to corridor entry." `float32` was chosen despite the AI's own earlier estimate that `float16` would halve the 241 GB output (trajectory step 58/63: "I'll use float16 for neural data to reduce the size to ~114 GB"); the final script does not do this.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron filter is the retinotopic area label: neurons with `iarea == -1` (outside mapped visual cortex) or `iarea == 7` are dropped; everything else is mapped onto the four coarse areas V1 (8), mHV (0,1,2,9), lHV (5,6), aHV (3,4). This keeps 4,105,393 of 4,691,034 neurons (V1 1,833,035; mHV 1,108,860; lHV 495,318; aHV 668,180). No additional spike/SNR/activity-based curation is applied.

ii.
```python
AREA_MAP = {8: 0, 0: 1, 1: 1, 2: 1, 9: 1, 5: 2, 6: 2, 3: 3, 4: 3}
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV']
EXCLUDED_AREAS = {-1, 7}

def get_neuron_mask_and_regions(iarea):
    iarea_int = iarea.astype(int)
    mask = np.array([a not in EXCLUDED_AREAS for a in iarea_int])
    region_idx = np.array([AREA_MAP[int(a)] for a in iarea_int[mask]], dtype=np.int64)
    return mask, region_idx
```

iii. CONVERSION_NOTES Key Decision 3: "Exclude non-visual-cortex neurons (iarea==-1, ==7): Matches reference code's `Get_density_map` exclusion (`idx_neu = (arid!=-1) & (arid != 7)`). These neurons are outside retinotopically mapped visual areas." Step 3 "Neuron curation rules": "No quality filtering beyond Suite2p cell classification (data already processed)."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry (trial start). Each trial's array begins at the first imaged frame the behavior labels as belonging to that trial inside the corridor, and ends at the last such frame. Trials therefore have different lengths; there is no common window, no padding and no truncation. Metadata records `temporal_alignment_event = 'Corridor entry (trial start)'`, `off_start = 0.0`, `off_end = None`.

ii.
```python
        frames = trial_frames_list[n]          # frames of this trial, in order, from corridor entry
        ...
        neural = spk_filtered[:, frames].astype(np.float32)
        ...
        input_arr[2, :] = (frames - frames[0]) * dt_sec       # starts exactly at 0
```
```python
            'temporal_alignment_event': 'Corridor entry (trial start)',
            'off_start': 0.0,
            'off_end': None,
```

iii. The spec requires equal bin *size*, not equal trial length, and allows `off_end = None`. The AI's trajectory (step 63) notes the wide spread of trial lengths and concludes that "the decoder just concatenates timepoints per session without knowing trial boundaries, long trials simply add more timepoints rather than causing structural issues", so variable lengths were kept rather than imposing a fixed window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling. One imaging frame = one time bin. The per-session frame interval is measured from the MATLAB-datenum frame timestamps `ft` as `median(diff(ft)) * 86400` (0.3146–0.3153 s across sessions), and the median of those per-session values, **314.7 ms**, is written to `metadata['time_bin_size']` (with `frame_rate_hz = 3.17` also stored).

ii.
```python
    ft = beh['ft'][:nfr]
    dt_sec = np.median(np.diff(ft)) * 24 * 3600 if len(ft) > 1 else 1.0/FRAME_RATE
```
```python
    dt_secs = [si['dt_sec'] for si in session_info]
    median_dt = np.median(dt_secs)
    ...
            'time_bin_size': median_dt * 1000,  # in ms
            'frame_rate_hz': FRAME_RATE,
```

iii. CONVERSION_NOTES Step 3/Key Decision 1: the imaging frame rate of 3.17 Hz is quoted from the methods ("Calcium signal recording frame rate: fs = 3.17Hz") and "each frame naturally becomes one time bin, roughly 315 ms wide" (trajectory step 33). The frame is the finest resolution available and every behavior stream is already on the same grid, so there is nothing to rebin.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From the per-trial cue frame `beh['SoundFr']` (a fractional frame number), the trial's frame indices, and the per-session frame interval `dt_sec` derived from `beh['ft']`.

ii.
```python
    SoundFr = beh['SoundFr']
    ft = beh['ft'][:nfr]
    dt_sec = np.median(np.diff(ft)) * 24 * 3600 if len(ft) > 1 else 1.0/FRAME_RATE
    ...
        input_arr[0, :] = (frames - SoundFr[n]) * dt_sec     # time to sound cue
```

iii. CONVERSION_NOTES Step 5 mapping table: "`SoundFr - frame_idx` → input[0]: time_to_sound_cue, `(frame_idx - SoundFr) * dt_sec`, continuous, time-varying". The AI identified `SoundFr` as the only variable giving cue timing, and converted frames to seconds with the measured frame interval rather than the nominal 3.17 Hz.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The signed offset of every frame of the trial from the cue frame is multiplied by the frame interval to give seconds: `(frame_index − SoundFr[trial]) · dt_sec`. The sign convention is **negative before the cue and positive after** (i.e. it is *time since* the cue). No clipping, no binarisation. Because `SoundFr` is fractional it is used as-is, so the value is not forced onto a frame boundary. Range over the full dataset: [−722.7 s, +1762.0 s] (extremes come from the very long stationary trials).

ii.
```python
        input_arr[0, :] = (frames - SoundFr[n]) * dt_sec     # time to sound cue
```

iii. CONVERSION_NOTES Step 5 documents the convention explicitly: "Negative before cue, positive after". Using frame index × measured `dt_sec` rather than the raw `ft` timestamps is justified by the regularity of the imaging clock (per-session `dt` is stable to ~0.1 %).

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from exactly the same `frames` array used to slice the neural columns of the trial, so it is on the imaging-frame grid and has the same length `T` as the neural matrix by construction.

ii.
```python
        T = len(frames)
        neural = spk_filtered[:, frames].astype(np.float32)
        input_arr = np.empty((4, T), dtype=np.float32)
        input_arr[0, :] = (frames - SoundFr[n]) * dt_sec
```

iii. Every stream in this dataset is indexed by imaging frame, so re-using the trial's frame index array guarantees alignment. The `--show-processing` plots overlay time-to-cue against neural activity and position for the first three trials of two sessions to make this visually checkable.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From `datexp` (the recording date in the index entry) together with `mname`: sessions are grouped by mouse and ordered by date, and the rank in that ordering is the training day. No explicit "day" or "days since start" field is used.

ii.
```python
    mouse_sessions = {}
    for s in sessions:
        m = s['mname']
        if m not in mouse_sessions:
            mouse_sessions[m] = []
        mouse_sessions[m].append(s)
    for m, sess_list in mouse_sessions.items():
        sess_list.sort(key=lambda s: s['datexp'])
        for i, s in enumerate(sess_list):
            s['day_index'] = i
```

iii. CONVERSION_NOTES Key Decision 5: "Day of training = chronological session index per mouse: Natural ordering that captures experience progression." Step 10 sanity check: "Day of training ranges 0-7 per mouse (matching 1-8 sessions/mouse)."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. A 0-based count of recorded sessions for that mouse (0 for its first recording day, up to 7), computed over the whole 89-session dataset. It is a per-trial scalar that is broadcast across all `T` bins of every trial in the session, and stored as `float32`.

ii.
```python
        input_arr[1, :] = float(sess['day_index'])             # day of training
```

iii. Recording days are not consecutive calendar days, so the AI counts recording sessions rather than elapsed calendar days ("chronological session index per mouse"). The value is broadcast because the format requires each input row to be length `T`.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From the trial's own frame indices (which come from `ft_trInd` and `ft_CorrSpc`) and the frame interval `dt_sec` derived from `beh['ft']`. `StartFr` is *not* used; the first corridor frame of the trial is taken as the trial start.

ii.
```python
        input_arr[2, :] = (frames - frames[0]) * dt_sec       # time since trial start
```

iii. CONVERSION_NOTES Step 5 mapping table: "`frame_idx - start_frame` → input[2]: time_since_trial_start ... Always >= 0". CONVERSION_NOTES Step 4 flags that "`StartFr` values ... Fractional values (e.g. 8.467)" and that trials can alternatively be located "via ft_trInd"; the script takes the latter route, so trial start is defined identically to the neural window start.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Elapsed seconds since the first frame of the trial: `(frame_index − frames[0]) · dt_sec`. It starts at exactly 0.0 for every trial and increases monotonically; over the full dataset the range is [0, 1763.7 s]. No clipping or normalisation.

ii.
```python
        input_arr[2, :] = (frames - frames[0]) * dt_sec
```

iii. Defining t=0 at the first imaged corridor frame makes this input exactly consistent with `off_start = 0.0` and with the neural window, and avoids the sub-frame ambiguity of the fractional `StartFr` that the AI noted in Step 4.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same `frames` array as the neural slice, so it is on the imaging-frame grid, is length `T`, and its zero coincides with the first neural column of the trial.

ii.
```python
        T = len(frames)
        neural = spk_filtered[:, frames].astype(np.float32)
        input_arr[2, :] = (frames - frames[0]) * dt_sec
```

iii. All streams are indexed by imaging frame; re-using the trial's frame index array is the alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From the per-trial boolean `beh['isRew']`, which marks trials run in the rewarded corridor.

ii.
```python
    isRew = beh['isRew']
    ...
        input_arr[3, :] = 1.0 if isRew[n] else 0.0            # reward availability
```

iii. CONVERSION_NOTES Step 5 mapping: "`isRew[trial]` → input[3]: reward_availability, 0 or 1, per-trial scalar".

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. None beyond a boolean → {0.0, 1.0} cast, broadcast across all `T` bins of the trial. 35 of 89 sessions contain any `reward_availability == 1` trials; the rest (naive/unsupervised mice) are all zeros.

ii.
```python
        input_arr[3, :] = 1.0 if isRew[n] else 0.0
```

iii. CONVERSION_NOTES Step 10 issue 3: "Reward availability only in subset: 35 sessions have reward_availability=1 trials, matching the licking sessions" — documented as expected rather than as a bug, because unsupervised and naive mice ran the same corridors with no water available.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `beh['WallName']`, the per-trial name of the corridor wall texture. `TrialStim`/`stim_id` were examined and rejected.

ii.
```python
    WallName = beh['WallName']
    ...
        output_arr[0, :] = STIM_TO_IDX[str(WallName[n])]      # visual stimulus
```

iii. Trajectory step 40: "stim_id is really just a general index that points to whichever stimulus pair a given mouse was trained on rather than fixed stimulus identities — for naive_test2 the same indices correspond to rock/brick stimuli instead of circle/leaf ones. For the decoder, I should use the actual WallName as the category label rather than stim_id."

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The 15 distinct wall names found in the dataset are used as **15 separate categories**, alphabetically sorted and mapped to indices 0–14 through a hard-coded global table. They are *not* collapsed into the four base textures (circle / leaf / rock / wood). The value is per-trial and broadcast across all `T` bins, stored as `int64`. Because each session only shows 3–7 of the 15 names, most classes are absent in most sessions (dataset-wide fractions: circle1 0.264, leaf1 0.285, leaf2 0.130, rock1 0.069, wood1 0.060, … circle3 0.008, wood1_swap1 0.007).

ii.
```python
ALL_STIMULI = sorted([
    'circle1', 'circle2', 'circle3',
    'leaf1', 'leaf1_swap1', 'leaf1_swap2', 'leaf2', 'leaf3',
    'rock1', 'rock2',
    'wood1', 'wood1_swap1', 'wood1_swap2', 'wood2', 'wood5',
])
STIM_TO_IDX = {s: i for i, s in enumerate(ALL_STIMULI)}
...
        output_arr[0, :] = STIM_TO_IDX[str(WallName[n])]
...
    output_values = [ALL_STIMULI, ['no_lick', 'lick'], ['0-1m', '1-2m', '2-3m', '3-4m'], ['Q1','Q2','Q3','Q4']]
```

iii. CONVERSION_NOTES Key Decision 6: "**15 stimulus categories**: Use individual stimulus names (circle1, leaf1, etc.) rather than grouped categories, preserving maximum information." The trajectory (step 40) shows the AI explicitly weighing the alternative — "grouping into circle, leaf, rock, and wood collapses the swap variants and numbered versions into just four categories, ensuring every session has enough coverage for a decoder to learn from" — and then choosing the fine-grained option to "preserve maximum information per trial".

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `beh['LickFr']`, the (fractional) imaging-frame number of every lick in the session. `beh['LickTrind']` is also read but is never used.

ii.
```python
    lick_fr = beh['LickFr'].astype(int) if len(beh['LickFr']) > 0 else np.array([], dtype=int)
    lick_trind = beh['LickTrind'].astype(int) if len(beh['LickTrind']) > 0 else np.array([], dtype=int)
```

iii. CONVERSION_NOTES Step 5 mapping: "`LickFr` presence → output[1]: licking, Binary time-varying, 1 at lick frames, From LickFr + LickTrind" (the `LickTrind` half of that description is not implemented).

## 8-b. What processing is involved in computing `output` *Licking*?

i. A session-length binary indicator is built once: the fractional lick frame numbers are truncated to `int`, licks outside `[0, nfr)` are discarded, and those frames are set to 1. A frame is 1 if at least one lick falls in it, 0 otherwise (multiple licks in a bin are not counted). Each trial then indexes into that vector. Dataset-wide: 3.5 % of bins are licks; 35 of 89 sessions contain any licks.

ii.
```python
def build_lick_array_vectorized(beh, trial_frames_list, nfr):
    ...
    lick_indicator = np.zeros(nfr, dtype=np.int64)
    valid_mask = (lick_fr >= 0) & (lick_fr < nfr)
    if valid_mask.any():
        lick_indicator[lick_fr[valid_mask]] = 1

    lick_arrays = []
    for n, frames in enumerate(trial_frames_list):
        if frames is None:
            lick_arrays.append(None)
        else:
            lick_arrays.append(lick_indicator[frames].copy())
    return lick_arrays
```
```python
        output_arr[1, :] = lick_arrays[n]                      # licking
```

iii. The spec asks for a binary time series ("0 = not licking, 1 = licking"), so the event list is converted to a per-bin flag. Step 10 issue 2 documents the zero-lick sessions as expected: "Licking only in subset of sessions: 35 of 89 sessions have licks (mice with reward conditioning). 54 sessions are pre-training or non-reward experiments → all no_lick."

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` already indexes imaging frames, so the indicator is natively on the neural grid; the trial's array is `lick_indicator[frames]` with the same `frames` used for the neural columns, hence identical length and alignment.

ii.
```python
            lick_arrays.append(lick_indicator[frames].copy())
...
        neural = spk_filtered[:, frames].astype(np.float32)
        output_arr[1, :] = lick_arrays[n]
```

iii. All streams are indexed by imaging frame, so the same `frames` index aligns them.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `beh['ft_Pos']`, the position in the virtual corridor at each imaging frame, in decimetres (0–40 across the textured part, 40–60 through the grey space), truncated to the imaged frames.

ii.
```python
    ft_Pos = beh['ft_Pos'][:nfr]
    ...
        output_arr[2, :] = np.clip(ft_Pos[frames] // 10, 0, 3).astype(np.int64)  # position bin
```

iii. CONVERSION_NOTES Step 4: "Position units | Corridor_Length=60 | ft_Pos in [0,~60] dm | 4m+2m=6m | 1 dm = 0.1 m, position in dm".

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Integer-divide the decimetre position by 10 to get the 1 m bin index, then clip to [0, 3]. Because trial frames are restricted to `ft_CorrSpc` (textured corridor), positions are below 40 dm and the clip is essentially a guard. Stored as `int64`, time-varying. Resulting dataset-wide distribution: 0.285 / 0.233 / 0.236 / 0.246.

ii.
```python
        output_arr[2, :] = np.clip(ft_Pos[frames] // 10, 0, 3).astype(np.int64)
```
```python
    output_values = [..., ['0-1m', '1-2m', '2-3m', '3-4m'], ...]
```

iii. CONVERSION_NOTES Step 5 mapping: "4 bins of 10dm (1m) each, time-varying — [0,10)→0, [10,20)→1, [20,30)→2, [30,40)→3", which is the "4 equal-length, 1-m-long spatial bins" the Decoder Task asks for. Step 10 sanity check: "Position bins roughly 25% each (28.5%, 23.3%, 23.6%, 24.6%)".

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. By fixed, equal-length spatial thresholds at 1 m (10 dm) intervals — 0–1 m, 1–2 m, 2–3 m, 3–4 m — not by data quantiles. Boundaries are the same for every trial, session and mouse. Values ≥ 40 dm (should not occur inside `ft_CorrSpc`) would be clipped into bin 3, negative values into bin 0.

ii.
```python
        output_arr[2, :] = np.clip(ft_Pos[frames] // 10, 0, 3).astype(np.int64)
```

iii. The Decoder Task specifies equal-length 1 m bins for position (in contrast to equal-occupancy bins for speed), so a fixed threshold is used; the `--show-processing` plots overlay raw `ft_Pos` against the derived bin index to verify the discretisation visually.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` has one value per imaging frame, and the trial's values are taken with the same `frames` index as the neural columns, so it is aligned by construction and has the same length.

ii.
```python
        neural = spk_filtered[:, frames].astype(np.float32)
        output_arr[2, :] = np.clip(ft_Pos[frames] // 10, 0, 3).astype(np.int64)
```

iii. All behaviour streams are on the imaging-frame grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `beh['ft_RunSpeed']`, the running speed at each imaging frame, truncated to the imaged frames.

ii.
```python
    ft_RunSpeed = beh['ft_RunSpeed'][:nfr]
    ...
        output_arr[3, :] = np.digitize(ft_RunSpeed[frames], speed_quantiles).astype(np.int64)
```

iii. Step 5 mapping: "`ft_RunSpeed` → output[3]: speed_bin, 4 quartile bins, time-varying, Global quartiles across all corridor frames."

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A dedicated first pass over all 89 sessions pools `ft_RunSpeed` over every corridor frame (`ft_CorrSpc`, truncated to `nfr`) in the dataset and takes the 25th/50th/75th percentiles as three global thresholds, `[0.0, 8.369, 30.185]`. These are then applied to every trial in pass 2. The boundaries are global (shared by all sessions and mice) rather than per-session, and are computed over all corridor frames rather than only over the frames actually written out.

ii.
```python
        corr_mask = ft_CorrSpc
        if corr_mask.any():
            all_speeds.append(ft_RunSpeed[corr_mask])
...
    all_speeds_flat = np.concatenate(all_speeds)
    speed_quantiles = np.percentile(all_speeds_flat, [25, 50, 75])
```

iii. CONVERSION_NOTES Key Decision 7: "**Speed quartiles computed globally**: Across all corridor frames in all sessions, ensuring each bin has 25% of data." The two-pass structure exists specifically so the boundaries can be computed before any trial is written (trajectory step 63/64).

## 10-c. How is `output` *Running speed* thresholded into categories?

i. With `np.digitize(speed, [q25, q50, q75])`, i.e. fixed value thresholds derived from global percentiles, labelled `Q1..Q4`. Because `q25 == 0.0` exactly (a large share of corridor frames are stationary, and `np.digitize` assigns `x >= bins[0]` to bin 1), bin `Q1` ends up containing only frames with **negative** speed. The realised distribution is **Q1 0.098 / Q2 0.402 / Q3 0.250 / Q4 0.250**, not 25 % each.

ii.
```python
    speed_quantiles = np.percentile(all_speeds_flat, [25, 50, 75])
    ...
        output_arr[3, :] = np.digitize(ft_RunSpeed[frames], speed_quantiles).astype(np.int64)
```

iii. The AI detected the imbalance and chose not to act on it. CONVERSION_NOTES Step 10 issue 1: "**Speed bin Q1 has only 9.8%**: Not a bug — global 25th percentile is exactly 0.0 (many stopped frames), so Q1 captures the ~10% of frames with speed exactly 0." (The stated mechanism is itself slightly wrong: the zero-speed frames land in Q2, and Q1 holds the negative-speed frames.) Earlier in the trajectory (step 54) the AI had anticipated the problem — "If I include stopped frames where speed is near zero, the first quartile bin gets dominated by zeros" — and considered a running-frame filter as the remedy, but then rejected that filter for other reasons and did not revisit the binning.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` has one value per imaging frame; the trial's values are taken with the same `frames` index as the neural columns, giving identical length and alignment.

ii.
```python
        neural = spk_filtered[:, frames].astype(np.float32)
        output_arr[3, :] = np.digitize(ft_RunSpeed[frames], speed_quantiles).astype(np.int64)
```

iii. All behaviour streams are on the imaging-frame grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Four specific issues are handled:
1. **Behaviour runs one or more frames past the imaging.** Every frame-level stream is truncated to the number of imaged frames `nfr = spk.shape[1]` (`ft_Pos`, `ft_RunSpeed`, `ft`, `ft_trInd`, `ft_CorrSpc`).
2. **Licks recorded outside the imaged window** are dropped with an explicit range mask; empty `LickFr`/`LickTrind` arrays are handled by a length check.
3. **Frames with no trial assignment** (`ft_trInd` is NaN — e.g. the ~282 frames masked out in swap sessions) silently fail the `ft_trInd == n` test and are excluded.
4. **Trials with 0 or 1 corridor frame** are set to `None` and skipped.
There is no `try/except` around per-session processing, and no fallback for an unknown `WallName` or an unexpected `iarea` code (either would raise `KeyError`).

ii.
```python
    spk = load_spk(mname, datexp, blk)
    nneu, nfr = spk.shape
    ...
    ft_Pos = beh['ft_Pos'][:nfr]
    ft_RunSpeed = beh['ft_RunSpeed'][:nfr]
    ft = beh['ft'][:nfr]
```
```python
    lick_fr = beh['LickFr'].astype(int) if len(beh['LickFr']) > 0 else np.array([], dtype=int)
    valid_mask = (lick_fr >= 0) & (lick_fr < nfr)
    if valid_mask.any():
        lick_indicator[lick_fr[valid_mask]] = 1
```
```python
    dt_sec = np.median(np.diff(ft)) * 24 * 3600 if len(ft) > 1 else 1.0/FRAME_RATE
```

iii. CONVERSION_NOTES Step 4 records the discovery: "Frame count | spk has N-1 frames vs ft | spk: 20428, ft: 20429 | Truncate ft to match spk (as reference code does)", matching the reference's `beh[...][:nfr]` idiom. The swap-key collision is handled by stripping `_swap1`/`_swap2` (Step 6), on the stated grounds that "the behavior data is identical between swap variants" — which is true for `ft_Pos`, `ft` and `WallName` but not exactly for `ft_trInd`, which differs by ~1 % of frames between the two variants.

## 12-a. What are the most time-consuming steps of the code?

i. Reading the deconvolved spike files dominates: ~405 GB across 89 sessions, read with `np.load(..., allow_pickle=True).item()` followed by a `np.concatenate` that copies the whole array again. Timings printed by the full run: pass 1 186.5 s, pass 2 1318.7 s, total 1893.9 s (≈32 min) — and the final `pickle.dump` of the 241.67 GB output is a large part of the remaining ~390 s. Crucially the spike files are read **twice**, because pass 1 loads each `*_neural_data.npy` in full just to read `data['spks'][0].shape[1]`.

ii.
```python
def collect_speeds_from_behavior(sessions, all_beh):
    """Pass 1: Collect corridor running speeds from behavior data only (no neural loading)."""
    ...
        # Load just to get shape - this is fast since we only need spks[0].shape
        data = np.load(path, allow_pickle=True).item()
        nfr = data['spks'][0].shape[1]
        nneu_total = sum(s.shape[0] for s in data['spks'])
        del data
```
```python
    data = np.load(path, allow_pickle=True).item()
    spk = np.concatenate([nspk for nspk in data['spks']], 0)
```

iii. The trajectory (steps 63–64) identifies the bottleneck correctly — "The main bottleneck is loading the large spk neural data files ... have pass 1 load only behavior data to compute speed quartiles quickly, then have pass 2 load neural data and build trials, avoiding redundant spk loading" — but the implemented pass 1 still loads them, and the inline comment "this is fast since we only need spks[0].shape" is incorrect for a pickled object array.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Three:
1. `extract_trial_frames` scans the entire frame index once per trial (`np.where((ft_trInd == n) & ft_CorrSpc)` for each of up to ~800 trials over up to ~30,000 frames). A single `np.argsort`/`np.split` grouping pass would do the same work once. The same scan is duplicated inside `collect_speeds_from_behavior` just to count valid trials.
2. `get_neuron_mask_and_regions` uses two Python-level comprehensions over up to ~90,000 neurons per session; `np.isin` plus a lookup array would be orders of magnitude faster.
3. `n_included = sum(1 for a in iarea.astype(int) if a not in EXCLUDED_AREAS)` is another Python loop over all neurons, purely to print a count.
The remaining per-trial loop that builds the arrays is hard to vectorise because of the variable trial lengths, and it is dominated by the memory copy anyway.

ii.
```python
    for n in range(ntrials):
        frames = np.where((ft_trInd == n) & ft_CorrSpc)[0]
```
```python
    mask = np.array([a not in EXCLUDED_AREAS for a in iarea_int])
    region_idx = np.array([AREA_MAP[int(a)] for a in iarea_int[mask]], dtype=np.int64)
```
```python
        n_included = sum(1 for a in iarea.astype(int) if a not in EXCLUDED_AREAS)
```

iii. The AI recognised the general problem ("Vectorizing the per-trial array construction is tricky given variable trial lengths, so I'll settle for pre-allocating where possible", trajectory step 63) and pre-allocates `input_arr`/`output_arr`, but never vectorised the frame-scan or the neuron-mask loops. CONVERSION_NOTES leaves the template's "Code inefficiencies identified / Code speedups added" fields (Step 6) and the Step 7 run-time estimate tables unfilled.

## 12-c. What processing does the code repeat multiple times?

i. Per session, the two passes repeat: (a) the full `*_neural_data.npy` load (the single biggest cost), (b) the `*_trans.npz` retinotopy load, (c) the brain-area mask/count computation, (d) the `dt_sec = median(diff(ft))*86400` computation, and (e) the per-trial corridor-frame scan (pass 1 counts valid trials, pass 2 re-derives the same frame lists). In `--show-processing` mode, `plot_processing` performs a *third* full load of the spike file, retinotopy, area mask, trial-frame extraction and lick array for the sessions being plotted, immediately after `process_session_full` has just computed all of them.

ii.
```python
        data = np.load(path, allow_pickle=True).item()      # pass 1
        nfr = data['spks'][0].shape[1]
        ...
        iarea = load_area_ids(mname, datexp)                # pass 1
        n_included = sum(1 for a in iarea.astype(int) if a not in EXCLUDED_AREAS)
        ...
        for n in range(ntrials):                            # pass 1: count valid trials
            frames = np.where((ft_trInd == n) & ft_CorrSpc)[0]
```
```python
    spk = load_spk(mname, datexp, blk)                      # pass 2
    iarea = load_area_ids(mname, datexp)
    trial_frames_list = extract_trial_frames(beh, nfr)
```
```python
def plot_processing(sess, all_beh, speed_quantiles, save_path):   # pass 3
    spk = load_spk(mname, datexp, blk)
    iarea = load_area_ids(mname, datexp)
    trial_frames_list = extract_trial_frames(beh, nfr)
    lick_arrays = build_lick_array_vectorized(beh, trial_frames_list, nfr)
```

iii. The duplication is a side-effect of the two-pass design chosen so that global speed quartiles are known before any trial is written. The AI intended pass 1 to touch behaviour only ("Since corridor frame indices can be derived from behavior alone, pass 1 becomes essentially free"), but needed `nfr` to truncate the behaviour streams and took it from the spike file rather than from the behaviour arrays, which reintroduced the full read it was trying to avoid.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items:
1. **Pass 1's `session_info`**: `n_included`, `nneu_total`, `ntrials`, `ntrials_valid` and `nfr` are computed for all 89 sessions and stored, but only `dt_sec` is ever consumed (for `metadata['time_bin_size']`); the rest is only printed.
2. **Loading 405 GB of spike data in pass 1** solely to read a frame count and a neuron count that are recomputed in pass 2.
3. **`lick_trind`** is read and cast from `beh['LickTrind']` and never used.
4. **`nneu`** from `spk.shape` in `process_session_full` is unpacked and unused.
5. **`exp_types`** is accumulated per session in `build_session_list` and never read.
6. **`float32` neural storage**: the output is 241.67 GB; the traces are non-negative deconvolved values and `float16` (which the AI itself planned in the trajectory) would halve it at no cost to the decoder.

ii.
```python
        session_info.append({'n_included': n_included, 'nneu_total': nneu_total,
                             'ntrials': ntrials, 'ntrials_valid': n_valid,
                             'nfr': nfr, 'dt_sec': dt_sec})
...
    dt_secs = [si['dt_sec'] for si in session_info]     # only field ever used
    median_dt = np.median(dt_secs)
```
```python
    lick_trind = beh['LickTrind'].astype(int) if len(beh['LickTrind']) > 0 else np.array([], dtype=int)
```
```python
    nneu, nfr = spk.shape          # nneu unused
```
```python
        neural = spk_filtered[:, frames].astype(np.float32)
```

iii. The per-session statistics were added as progress/sanity reporting ("Neurons per session: min=…, max=…, mean=…" in the summary and the Step 9 consistency table), so the printing is intentional even though the stored values are not reused. No justification is given in CONVERSION_NOTES for keeping `float32` after the trajectory concluded `float16` should be used.
