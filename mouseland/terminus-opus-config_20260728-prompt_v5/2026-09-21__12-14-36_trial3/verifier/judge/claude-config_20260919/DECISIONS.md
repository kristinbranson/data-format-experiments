# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Three directories under `/app/data` are used: `beh/` (behavior, one `Beh_<exp_type>.npy` per experiment type plus the master index `Imaging_Exp_info.npy`), `spk/` (one `<mname>_<datexp>_<blk>_neural_data.npy` per recording, holding a list of per-imaging-plane deconvolved trace arrays), and `retinotopy/` (one `<mname>_<datexp>_trans.npz` per recording, holding `iarea`). `Imaging_Exp_info.npy` is read first and flattened over its 23 experiment types into a list of unique recordings. For each recording the behavior file of its experiment type is loaded (memoised in a module-level `_beh_cache` dict, cleared every 10 sessions) and the session's entry is looked up by the `<mname>_<datexp>_<blk>` key, falling back to a `startswith` prefix match for the swap sessions whose behavior keys carry a `_swap1`/`_swap2` suffix. Spikes and retinotopy are read once per session. The behavior files are additionally read in a separate first pass to compute the global running-speed quartile boundaries.

ii.
```python
def get_all_unique_recordings():
    exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
    seen = set()
    recordings = []
    for exp_type, sessions in exp_info.items():
        for s in sessions:
            key = f"{s['mname']}_{s['datexp']}_{s['blk']}"
            if key not in seen:
                seen.add(key)
                recordings.append((key, s, exp_type))
    return recordings


def load_beh_for_recording(key, exp_type):
    beh_file = f'Beh_{exp_type}.npy'
    if beh_file not in _beh_cache:
        _beh_cache[beh_file] = np.load(
            os.path.join(DATA_ROOT, 'beh', beh_file), allow_pickle=True).item()
    beh_dict = _beh_cache[beh_file]
    if key in beh_dict:
        return beh_dict[key]
    for beh_key in beh_dict:
        if beh_key.startswith(key):
            return beh_dict[beh_key]
    return None


def load_spk(mname, datexp, blk):
    fn = os.path.join(DATA_ROOT, 'spk', f'{mname}_{datexp}_{blk}_neural_data.npy')
    data = np.load(fn, allow_pickle=True).item()
    return np.concatenate(data['spks'], 0)


def load_retino(mname, datexp):
    fn = os.path.join(DATA_ROOT, 'retinotopy', f'{mname}_{datexp}_trans.npz')
    if not os.path.exists(fn):
        return None
    return np.load(fn, allow_pickle=True)['iarea']
```

iii. From CONVERSION_NOTES.md Step 1/Step 2: "`load_spk`: concatenates planes with `np.concatenate(data['spks'], 0)`" — the loaders were copied from the reference `utils.py` (`load_spk`, `load_retino`, `neu_area_ID`). Step 2 records "23 experiment types, 142 total entries, 89 unique recordings" and "Behavioral data identical for same recording across experiment types", which is the stated reason for de-duplicating on the `mname_datexp_blk` key and for caching behavior files rather than re-reading them per session.

## 1-b. How are the data split into subjects (mice)?

i. The mouse is taken directly from the `mname` field of the `Imaging_Exp_info.npy` entry. `subjects` is built in order of first appearance while iterating the recording list, and `subject_idx` stores each session's index into that list. This yields 19 subjects over 89 sessions, matching the paper.

ii.
```python
mname, datexp, blk = db_info['mname'], db_info['datexp'], db_info['blk']
...
mname = result['mname']
if mname not in subject_to_idx:
    subject_to_idx[mname] = len(subjects)
    subjects.append(mname)
...
all_subject_idx.append(subject_to_idx[mname])
...
'subjects': subjects,
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```

iii. No explicit justification is given beyond the Step 2/Step 3 consistency table: the index already names the mouse, and the resulting count (19) is checked against the paper's "89 recordings in 19 mice".

## 1-c. How are the data split into sessions?

i. A session is one recording, identified by the triple `(mname, datexp, blk)` joined into the key `<mname>_<datexp>_<blk>`. That key also names the spike file. The same recording is listed under several experiment types in the master index, so a `seen` set keeps only the first occurrence, giving 89 sessions. A session is dropped only if the behavior entry cannot be found or if fewer than 2 valid trials survive; in the full run no session was dropped.

ii.
```python
key = f"{s['mname']}_{s['datexp']}_{s['blk']}"
if key not in seen:
    seen.add(key)
    recordings.append((key, s, exp_type))
```
```python
if len(neural_trials) < 2:
    return None
```

iii. CONVERSION_NOTES.md Step 2: "23 experiment types, 142 total entries, 89 unique recordings. Behavioral data identical for same recording across experiment types." The de-duplication is justified by having verified that the behavior is the same object across experiment types. Step 9 checks the resulting 89 sessions / 19 subjects against the paper.

## 1-d. How are the data split into trials?

i. The trial membership of every imaging frame is given by `ft_trInd`. For each trial index `0 .. ntrials-1` the frames kept are those labelled with that trial AND inside the textured corridor (`ft_CorrSpc`, the 4 m texture, excluding the 2 m grey space) AND with the VR moving (`ft_move > 0`, i.e. the mouse running above the 6 cm/s threshold). Trials are variable length; nothing is padded or truncated to a common window. The frames are taken with `np.where` on the mask, so within a trial they are in increasing frame order but are not necessarily consecutive — every stationary frame in the middle of a traversal is removed.

ii.
```python
corridor_mask = ft_CorrSpc.copy()
corridor_mask[n_use:] = False  # safety
running_mask = beh['ft_move'][:n_use] > 0  # only running frames, matching reference paper

for trial in range(ntrials):
    trial_mask = (ft_trInd == trial) & corridor_mask & running_mask
    trial_frames = np.where(trial_mask)[0]

    if len(trial_frames) < 2:
        continue
```

iii. CONVERSION_NOTES.md Step 5 Key Decisions: "1. Running frames only (ft_move > 0) - matches reference paper. 2. Corridor frames only (ft_CorrSpc) - 4m textured area". From the trajectory (step ~62): "the reference code uses `ft_move > 0` which means VR is moving. The VR moves at constant 60 cm/s when the mouse runs > 6 cm/s threshold. So `ft_move > 0` is equivalent to the mouse running above threshold. For the decoder, I'll include ONLY running frames (ft_move > 0) to match the reference paper's processing." Step 3 of the notes cites the methods: "We only considered timepoints during running".

## 1-e. How are trials filtered based on quality controls?

i. Two filters only: a trial is dropped if it has fewer than 2 surviving (running, in-corridor) frames, and a session is dropped if fewer than 2 trials survive. There is no filter on trial duration, no filter on outlier traversals, and no filter based on behavior quality. In the full run neither filter ever fired: every one of the 89 sessions reports `N/N` trials kept and the total is 38,110 trials. Because the running mask removes stationary frames, trials in which the animal stopped for a long time are kept but shrink to a small number of bins; e.g. session 50 `TX88_2022_07_19_1` retains a trial whose stored bins span 1763.7 s (29 min) of wall-clock time.

ii.
```python
if len(trial_frames) < 2:
    continue
...
if len(neural_trials) < 2:
    return None
```

iii. No explicit justification for trial curation is given. CONVERSION_NOTES.md Step 3 "Curation Steps" is not filled in with trial rules, and Step 5 lists no trial-quality decision. The implicit rationale, from the notes' Step 5 Key Decisions, is that restricting to running frames is itself the curation step inherited from the paper, and the `< 2 frames` / `< 2 trials` thresholds exist to satisfy the format requirement "There needs to be at least two trials within each session in order to evaluate the decoder performance."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `spks` entry of `spk/<session>_neural_data.npy`, which is a list of (neurons × frames) deconvolved-fluorescence arrays, one per imaging plane, concatenated along the neuron axis. The visual-area label of each neuron comes from `iarea` in `retinotopy/<mname>_<datexp>_trans.npz`.

ii.
```python
def load_spk(mname, datexp, blk):
    fn = os.path.join(DATA_ROOT, 'spk', f'{mname}_{datexp}_{blk}_neural_data.npy')
    data = np.load(fn, allow_pickle=True).item()
    return np.concatenate(data['spks'], 0)
```
```python
iarea = load_retino(mname, datexp)
brain_region_idx = iarea_to_region_idx(iarea) if iarea is not None else np.full(n_neurons, 4, dtype=np.int64)
```

iii. CONVERSION_NOTES.md Step 1: "Neural data: Suite2p deconvolved fluorescence traces (no dF/F computation needed)"; "`load_spk`: concatenates planes with `np.concatenate(data['spks'], 0)`". The loader is copied from the reference `utils.py`.

## 2-b. How is the `neural` data processed?

i. Not processed at all. The columns of `spks` belonging to a trial's frame list are sliced out and cast to `float32`. No dF/F, no deconvolution, no normalisation, no z-scoring, no smoothing, no padding to a common length.

ii.
```python
neural = spk[:, trial_frames].astype(np.float32)  # keep float32 for decoder compatibility
```

iii. CONVERSION_NOTES.md Step 1: "Neural data: Suite2p deconvolved fluorescence traces (no dF/F computation needed)" — the file already contains what the paper's analyses use, so no further processing is needed. The `float32` choice is annotated in the code as "keep float32 for decoder compatibility"; the trajectory shows the agent considered `float16` ("Float16 should be fine for the neural data since PCA will lose precision anyway") but kept `float32`, producing a 161.65 GB pickle.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are filtered. All 4,691,034 neurons across the 89 sessions are kept, giving 20,547–89,577 per session. `iarea` is used only to *label* neurons: codes 8 → V1, {0,1,2,9} → mHV, {3,4} → aHV, {5,6} → lHV, and everything else (including -1 and 7) → a fifth region named `unassigned`, which holds 585,641 neurons (12.5%). No d-prime selectivity filter is applied.

ii.
```python
BRAIN_REGION_NAMES = ['V1', 'mHV', 'aHV', 'lHV', 'unassigned']
IAREA_TO_REGION = {8: 0}  # V1
for ia in [0, 1, 2, 9]: IAREA_TO_REGION[ia] = 1  # mHV
for ia in [3, 4]: IAREA_TO_REGION[ia] = 2  # aHV
for ia in [5, 6]: IAREA_TO_REGION[ia] = 3  # lHV


def iarea_to_region_idx(iarea):
    region_idx = np.full(len(iarea), 4, dtype=np.int64)
    for ia_code, reg_idx in IAREA_TO_REGION.items():
        region_idx[iarea == ia_code] = reg_idx
    return region_idx
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 3: "All neurons included - no d-prime filtering"; Step 1 note: "d-prime >= 0.3 for neuron selection in analyses only" — i.e. the reference's neuron selection is treated as belonging to the paper's figures, not to the data conversion. README.md adds "All neurons included: No d-prime filtering (decoder uses PCA internally)". Step 9's consistency table justifies keeping everything by the fact that the per-session neuron counts then reproduce the paper's quoted range exactly (20,547–89,577, mean 52,708).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The declared alignment event is trial start / corridor entry, and each trial's array begins at the first frame that is inside the corridor *and* running, and ends at the last such frame. Trials keep their own length (T from 11 to 178 bins, mean 22.25); nothing is padded or cut to a common window. Metadata records `off_start = 0.0` and `off_end = None`.

ii.
```python
trial_mask = (ft_trInd == trial) & corridor_mask & running_mask
trial_frames = np.where(trial_mask)[0]
...
neural = spk[:, trial_frames].astype(np.float32)
```
```python
'temporal_alignment_event': 'Trial start (corridor entry)',
'off_start': 0.0,
'off_end': None,
```

iii. CONVERSION_NOTES.md/README state the alignment is "Trial start (corridor entry)". The variable-length choice is implicit — the format spec allows `off_end = None` and the decoder treats each time bin as a sample, so no common window is imposed. Note the implementation aligns to the first *running* in-corridor frame rather than to `StartFr`, since non-running frames at corridor entry are removed before the trial array is built.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling. The imaging frame is the bin. Per session `dt` is computed as the median of `np.diff(ft)` converted from MATLAB datenum days to seconds; the metadata stores the median of those per-session `dt` values, 314.69 ms (≈3.18 Hz). Because non-running frames are dropped, consecutive stored bins are each 314.69 ms long but are not always 314.69 ms apart.

ii.
```python
dt = float(np.median(np.diff(ft)) * 86400)
...
median_dt_ms = float(np.median(dt_values) * 1000)
...
'time_bin_size': median_dt_ms,
```

iii. CONVERSION_NOTES.md Step 2 records "Frame rate | ~3.18 Hz (314.69 ms)"; README Key Processing Decision 4: "Native frame rate: ~3.18 Hz, no temporal rebinning". The imaging frame is the finest resolution available and every behavior stream is already sampled on the same grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. `SoundFr` (the fractional imaging-frame number at which the sound cue was played on each trial) together with the frame index of each kept bin and the per-session frame interval `dt` (itself derived from `ft`). The frame timestamps `ft` are not used directly; the time axis is reconstructed as frame index × `dt`.

ii.
```python
sound_fr = beh['SoundFr']
dt = float(np.median(np.diff(ft)) * 86400)
...
time_to_cue = ((trial_frames - sound_fr[trial]) * dt).astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 5 variable mapping: "SoundFr - frame | input[0]: time_to_sound_cue | seconds". Step 3 notes the paper's statement that the cue position is "randomly chosen ... between positions 0.5 m and 3.5 m", so the cue must be represented as a continuous per-frame time rather than a per-trial constant.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each kept frame, the signed difference between the frame index and the trial's `SoundFr`, multiplied by the session's median frame interval, in seconds. The sign convention is **elapsed time since the cue**: negative before the cue, positive after. Values span [-722.7, 1762.0] s across the dataset.

ii.
```python
time_to_cue = ((trial_frames - sound_fr[trial]) * dt).astype(np.float32)
```

iii. README.md documents the convention explicitly: "`time_to_sound_cue`: Time to sound cue in seconds (negative = before cue)". No further justification is given for the sign; the value is a monotone affine function of the true cue-relative time either way.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from exactly the same `trial_frames` index array used to slice the neural columns of that trial, so it has the same length and the same per-bin correspondence by construction. All behavior streams in this dataset are indexed by imaging frame number, so no cross-stream interpolation is needed.

ii.
```python
trial_frames = np.where(trial_mask)[0]
neural = spk[:, trial_frames].astype(np.float32)
time_to_cue = ((trial_frames - sound_fr[trial]) * dt).astype(np.float32)
```

iii. Implicit in the design: everything is derived from one `trial_frames` array. CONVERSION_NOTES.md Step 10 Check 2 reports a spot check reloading the original files and confirming the neural slice and per-frame outputs for trial 5 of a session match (`np.allclose = True`).

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Two different fields of the `Imaging_Exp_info.npy` entry, whichever is present: `days` if the entry has it, else `sess#`, else 0.0. `days` exists only on the 8 `*_train2_after_learning` entries and holds an actual number of training days (6–15). `sess#` exists on nearly all other entries and is a protocol-stage index (0 for `*_before_learning`, 1 for most test and after-learning sessions, 2–4 for repeated naive tests). Dates are not used.

ii.
```python
def get_training_day(db_info):
    if 'days' in db_info:
        return float(db_info['days'])
    if 'sess#' in db_info:
        return float(db_info['sess#'])
    return 0.0
```

iii. From the trajectory (step 43): "The `sess#` field represents the session number (0=before learning, 1=after learning, etc.), and `days` is the number of days of training for the after-learning sessions. For the decoder input 'Day of training', I can use the `sess#` or `days` field. For sessions without a `days` field, I'll use `sess#` as a proxy." CONVERSION_NOTES.md Step 5 mapping row: "sess#/days | input[1]: day_of_training | per-trial". README: "`day_of_training`: Day of training (0 = before learning, 1+ = after)".

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The scalar returned by `get_training_day` is broadcast unchanged across all bins of every trial of the session, as `float32`. There is no per-mouse ordering, no date arithmetic, and no normalisation of the two source fields onto a common scale. The resulting range over the dataset is [0.0, 15.0], but the value is not monotone in calendar time within a mouse: e.g. DR10's six sessions in date order (2022-07-12, 07-19, 07-21, 07-28, 07-29, 07-30) receive 0, 1, 1, 6, 1, 1.

ii.
```python
training_day = get_training_day(db_info)
...
day_arr = np.full(n_t, training_day, dtype=np.float32)
inp = np.stack([time_to_cue, day_arr, time_since_start, reward_arr], axis=0)
```

iii. As in 4-a: `days` is treated as authoritative where present and `sess#` is used "as a proxy" elsewhere. No justification is offered for putting the two on the same continuous axis, and the notes do not flag the resulting non-monotonicity.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. The kept-frame index array itself (`trial_frames`) and the session's median frame interval `dt` (derived from `ft`). `StartFr`, the recorded corridor-entry frame, is loaded elsewhere in the exploration but is **not** used.

ii.
```python
time_since_start = ((trial_frames - trial_frames[0]) * dt).astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 5 mapping row: "frame - start | input[2]: time_since_trial_start | seconds"; README: "`time_since_trial_start`: Time since corridor entry in seconds". No further justification for using the first kept frame rather than `StartFr`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Elapsed seconds from the **first kept (running, in-corridor) frame** of the trial: `(frame_index - first_frame_index) * dt`. It therefore starts at exactly 0.0 in every trial and is never negative (range across the dataset [0.0, 1763.7] s). Because the subtraction is on raw frame indices, any stationary frames removed in the middle of a traversal still count toward the elapsed time, which is why one trial reaches 1763.7 s.

ii.
```python
time_since_start = ((trial_frames - trial_frames[0]) * dt).astype(np.float32)
```

iii. No explicit justification; the documentation describes the quantity as "time since corridor entry", and the implementation takes the first stored bin as the trial origin.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed from the same `trial_frames` array as the neural slice, so one value per neural bin, aligned by construction. Its zero point coincides with the first neural bin of the trial.

ii.
```python
trial_frames = np.where(trial_mask)[0]
neural = spk[:, trial_frames].astype(np.float32)
time_since_start = ((trial_frames - trial_frames[0]) * dt).astype(np.float32)
```

iii. All streams share the imaging-frame index, so alignment is inherited from `trial_frames`.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. `isRew`, the boolean per-trial flag marking trials run in the rewarded corridor.

ii.
```python
is_rew = beh['isRew']
...
reward_arr = np.full(n_t, float(is_rew[trial]), dtype=np.float32)
```

iii. CONVERSION_NOTES.md Step 5 mapping row: "isRew | input[3]: reward_availability | binary". README: "`reward_availability`: 1 if rewarded corridor, 0 if not."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. None beyond a cast to float and a broadcast over the trial's bins. It is 0 for every trial of the unsupervised, naive and grating sessions and takes both values within the supervised/rewarded sessions, as the per-session ranges in `verification_full_out.txt` show.

ii.
```python
reward_arr = np.full(n_t, float(is_rew[trial]), dtype=np.float32)
```

iii. No processing is required; the flag is already per-trial and binary as the decoder spec asks.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. `WallName`, the per-trial name of the texture on the corridor walls. `TrialStim` is not used.

ii.
```python
wall_name = beh['WallName']
...
stim_idx = STIM_TO_IDX.get(str(wall_name[trial]), 0)
```

iii. From the trajectory (step ~40): the agent enumerated the unique wall names across the whole dataset and found 15. CONVERSION_NOTES.md Step 5 mapping row: "WallName | output[0]: visual_stimulus | 15 categories". (`TrialStim` is partly masked with the placeholder string `'stimulus_of_trial'` in the swap sessions, so `WallName` is the usable field.)

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. A single global dictionary maps each of the 15 distinct wall names to its index in the sorted list `['circle1','circle2','circle3','leaf1','leaf1_swap1','leaf1_swap2','leaf2','leaf3','rock1','rock2','wood1','wood1_swap1','wood1_swap2','wood2','wood5']`. There is **no** grouping into base texture families. The per-trial index is broadcast over the trial's bins as `int64`. An unrecognised name would silently fall back to index 0 (`circle1`). Most sessions contain only 2–5 of the 15 classes.

ii.
```python
ALL_STIMULI = sorted(['circle1', 'circle2', 'circle3', 'leaf1', 'leaf1_swap1', 'leaf1_swap2',
                      'leaf2', 'leaf3', 'rock1', 'rock2', 'wood1', 'wood1_swap1',
                      'wood1_swap2', 'wood2', 'wood5'])
STIM_TO_IDX = {s: i for i, s in enumerate(ALL_STIMULI)}
...
stim_idx = STIM_TO_IDX.get(str(wall_name[trial]), 0)
stim_arr = np.full(n_t, stim_idx, dtype=np.int64)
```

iii. From the trajectory (step ~40): "There are 15 unique stimulus types across all sessions. For the decoder output 'Visual stimulus category', I need to encode each stimulus as a categorical variable. Since different sessions have different subsets of stimuli, I need to create a global mapping." CONVERSION_NOTES.md Step 5 Key Decision 5: "15 stimulus categories mapped globally." The agent also recorded from the paper that the swap stimuli have "identical features to leaf1 and circle1, respectively, but were arranged in different spatial configurations", i.e. they are visually distinct.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. `LickFr` (the fractional imaging-frame number of each lick in the session) and `LickTrind` (the trial index of each lick), which are used together to build a per-trial lookup of lick frames.

ii.
```python
lick_fr = beh['LickFr']
lick_trind = beh['LickTrind']
...
lick_by_trial = defaultdict(list)
for li in range(len(lick_fr)):
    lick_by_trial[int(lick_trind[li])].append(lick_fr[li])
```

iii. CONVERSION_NOTES.md Step 5 mapping row: "LickFr | output[1]: licking | binary". The trajectory shows the agent inspected `LickFr`, `LickTrind` and `LickPos` and chose the frame-indexed fields so that licking lands on the same grid as the neural data.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Per trial, a zero vector of length T is created; for each lick belonging to that trial, the nearest kept frame is found and set to 1 provided it is within 1 frame of the lick's (fractional) frame number. So a bin is 1 if at least one lick falls within it. Licks that occurred on a frame that was removed by the running filter are dropped: the agent measured that 33% of licks in a supervised session occur while stationary. The overall result is 3.6% lick bins, with all-zero licking in the unsupervised/naive/grating sessions.

ii.
```python
lick_binary = np.zeros(n_t, dtype=np.int64)
if trial in lick_by_trial:
    for lf in lick_by_trial[trial]:
        diffs = np.abs(trial_frames - lf)
        closest = np.argmin(diffs)
        if diffs[closest] < 1.0:
            lick_binary[closest] = 1
```

iii. From the trajectory (step ~63): "33% of licks occur during stopped periods (ft_move=0) ... The task says to predict licking as a decoder output. If I filter to running frames only, I lose 33% of licks. However, the paper says they only use running frames. And the reference code uses `ft_move > 0`." The running filter was kept for consistency with the paper, accepting the loss.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` is already expressed in imaging-frame units, so the binary vector is defined directly on the trial's `trial_frames` grid — same length, same bins as the neural slice. Alignment is by nearest kept frame with a strict 1-frame tolerance.

ii.
```python
neural = spk[:, trial_frames].astype(np.float32)
...
diffs = np.abs(trial_frames - lf)
closest = np.argmin(diffs)
if diffs[closest] < 1.0:
    lick_binary[closest] = 1
```

iii. All streams are indexed by imaging frame; the tolerance exists because `LickFr` is fractional.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. `ft_Pos`, the VR position of the mouse at each imaging frame, in decimetres (0–40 across the 4 m texture, continuing to 60 through the 2 m grey space).

ii.
```python
ft_Pos = beh['ft_Pos'][:n_use]
...
pos = ft_Pos[trial_frames]
```

iii. CONVERSION_NOTES.md Step 5 mapping row: "ft_Pos / 10 | output[2]: position | 4 bins of 1m". Step 2/3 record "Corridor length | 4m + 2m gray" from the methods, which is why the position axis is in decimetres and why only `ft_CorrSpc` frames (0–40 dm) are used.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The position of each kept frame is divided by 10 (decimetres → metres), floored, and clipped to [0, 3]. No smoothing or interpolation. Result is stored as `int64`.

ii.
```python
pos = ft_Pos[trial_frames]
pos_binned = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
```

iii. Directly follows the decoder spec: "Position in corridor discretized into 4 equal-length, 1-m-long spatial bins". Restriction to `ft_CorrSpc` guarantees the raw value is already within the texture, so the clip is only a guard.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four fixed, equal-length spatial bins at 1 m boundaries: `[0,10) → 0-1m`, `[10,20) → 1-2m`, `[20,30) → 2-3m`, `[30,40] → 3-4m`. The bins are defined by geometry, not by data quantiles. The realised distribution is essentially uniform (0.250 / 0.249 / 0.250 / 0.252).

ii.
```python
pos_binned = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
...
'output_values': [
    ALL_STIMULI,
    ['no_lick', 'lick'],
    ['0-1m', '1-2m', '2-3m', '3-4m'],
    ['Q1', 'Q2', 'Q3', 'Q4'],
],
```

iii. Exactly as specified in the Decoder Task section. CONVERSION_NOTES.md Step 9 records the resulting distribution "Position distribution | ~25% each | 25.0-25.2% | ✓" as a consistency check.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` gives one value per imaging frame, and it is indexed by the same `trial_frames` array used for the neural slice, so it is aligned bin-for-bin with no interpolation.

ii.
```python
neural = spk[:, trial_frames].astype(np.float32)
pos = ft_Pos[trial_frames]
```

iii. CONVERSION_NOTES.md Step 10 Check 2 reports a direct spot check against the raw files: "Position match: `np.allclose` = True" for trial 5 of `TX83_2022_08_17_1`, reloading `ft_Pos` and re-deriving the frame mask independently of the conversion code.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. `ft_RunSpeed`, the mouse's running speed at each imaging frame. `ft_move` and `ft_CorrSpc` are used as masks when estimating the bin boundaries.

ii.
```python
ft_RunSpeed = beh['ft_RunSpeed'][:n_use]
...
speed = ft_RunSpeed[trial_frames]
```

iii. CONVERSION_NOTES.md Step 5 mapping row: "ft_RunSpeed | output[3]: running_speed | 4 quartile bins".

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A first pass over all 89 behavior files pools `ft_RunSpeed` from every running in-corridor frame of the whole dataset and takes the 25th, 50th and 75th percentiles as three **global** boundaries (12.39, 25.33, 40.83). Each kept frame's speed is then assigned to a bin with `np.digitize` against those boundaries. The boundaries are computed once and shared by every session. Raw speeds range from -19.17 to 321.89; negatives are not handled specially and fall into bin 0.

ii.
```python
def compute_running_speed_quartiles(recordings):
    all_speeds = []
    for key, db_info, exp_type in recordings:
        beh = load_beh_for_recording(key, exp_type)
        if beh is None:
            continue
        n = min(len(beh['ft_RunSpeed']), len(beh['ft_CorrSpc']), len(beh['ft_move']))
        running_corridor = beh['ft_CorrSpc'][:n] & (beh['ft_move'][:n] > 0)
        corridor_speeds = beh['ft_RunSpeed'][:n][running_corridor]
        all_speeds.append(corridor_speeds)
    all_speeds = np.concatenate(all_speeds)
    bins = np.percentile(all_speeds, [25, 50, 75])
    return bins
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 4: "Speed quartiles on running corridor frames only - ensures 25% per bin", and Step 10 Check 6: "Speed quartiles: Fixed to compute on running corridor frames only for balanced distribution". The trajectory records the bug that prompted this: computing the quartiles over *all* corridor frames (including stationary ones) put the Q1 boundary at 0 and left only 2.1% of the stored data in Q1; restricting the estimate to the same frame population that is actually stored gave 24.9–25.0% per bin globally.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize` against the three global percentile boundaries, clipped to [0, 3], giving `Q1 … Q4`. Globally the bins hold 24.9 / 25.0 / 25.0 / 25.0 % of the data. Per session, however, the distribution is very uneven because a single global threshold is applied to sessions with different speed distributions — e.g. one session has 95.6% of its bins in Q1 and 0.0% in Q4, another has 68.5% in Q4.

ii.
```python
speed = ft_RunSpeed[trial_frames]
speed_binned = np.clip(np.digitize(speed, speed_bins), 0, 3).astype(np.int64)
```

iii. Follows the Decoder Task wording "Running speed discretized into 4 bins, each corresponding to 25% of the data" read as a statement about the pooled dataset. CONVERSION_NOTES.md Step 9 checks the pooled result: "Speed distribution | ~25% each | 24.9-25.0% | ✓". The per-session imbalance is not discussed.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is sampled per imaging frame and is indexed with the same `trial_frames` array as the neural slice, giving bin-for-bin alignment with no resampling.

ii.
```python
neural = spk[:, trial_frames].astype(np.float32)
speed = ft_RunSpeed[trial_frames]
```

iii. Inherited from the single `trial_frames` index; all behavior streams live on the imaging-frame grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Four cases are handled. (a) Behavior arrays can run past the imaging: every per-frame stream is truncated to `n_use = min(n_frames, len(beh['ft']))`. (b) `ft_trInd` contains NaN for frames outside any trial; the `== trial` comparison silently excludes them. (c) A missing retinotopy file yields `iarea = None` and every neuron of that session is labelled `unassigned` (this never fired — all 89 sessions have retinotopy). (d) A behavior entry that cannot be found returns `None` and the session is skipped. Two further fallbacks are silent rather than defensive: an unknown `WallName` is mapped to index 0, i.e. relabelled as `circle1`; and a missing `days`/`sess#` field yields `day_of_training = 0.0`. Licks and trials falling outside the retained frames are dropped implicitly by the nearest-frame tolerance and the frame mask. Negative `ft_RunSpeed` values (down to -19.17) are left as-is.

ii.
```python
n_use = min(n_frames, len(beh['ft']))

ft_Pos = beh['ft_Pos'][:n_use]
ft_CorrSpc = beh['ft_CorrSpc'][:n_use]
ft_RunSpeed = beh['ft_RunSpeed'][:n_use]
ft_trInd = beh['ft_trInd'][:n_use]
ft = beh['ft'][:n_use]
```
```python
iarea = load_retino(mname, datexp)
brain_region_idx = iarea_to_region_idx(iarea) if iarea is not None else np.full(n_neurons, 4, dtype=np.int64)
```
```python
beh = load_beh_for_recording(key, exp_type)
if beh is None:
    return None
```
```python
stim_idx = STIM_TO_IDX.get(str(wall_name[trial]), 0)
```

iii. CONVERSION_NOTES.md Step 10 Check 5: "Edge cases: Off-by-one between beh/spk handled with min(n_frames, n_beh)". This mirrors the reference code's `beh[...][:nfr]` with `nfr = spk.shape[1]`. The notes describe the dataset as clean and report no errors or warnings from the format verifier.

## 12-a. What are the most time-consuming steps of the code?

i. Two steps dominate the 613.9 s full run. (1) Reading the 89 spike files: the per-session `load=` timings printed in `conversion_full_out.txt` run 1.2–7.3 s and total roughly 215 s — pure I/O plus the `np.concatenate` of the per-plane arrays, which copies the whole session's traces. (2) Writing the output: the per-session totals sum to about 356 s, leaving roughly 250 s for `pickle.dump` of the 161.65 GB file. The per-trial processing itself (`proc=`) is only 0.5–3.0 s per session, about 110 s total. The quartile pre-pass over all behavior files costs 1.8 s.

ii.
```python
t0 = time.time()
spk = load_spk(mname, datexp, blk)
n_neurons, n_frames = spk.shape
t_load = time.time() - t0
...
print(f'[{i+1}/{len(recordings)}] {key} ({exp_type}) '
      f'{result["ntrials_valid"]}/{result["ntrials_original"]} trials, '
      f'{result["n_neurons"]} neurons, '
      f'load={result["t_load"]:.1f}s proc={result["t_proc"]:.1f}s total={elapsed:.1f}s')
```
```python
with open(output_file, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. The code instruments load vs. process time per session precisely so the bottleneck is visible, and the full run finished in 10 minutes, under the 15-minute budget set by the instructions. The notes do not draw an explicit conclusion about which step dominates, nor note that keeping `float32` rather than `float16` doubles both the file size and the dump time.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Three. (1) The per-trial mask `(ft_trInd == trial) & corridor_mask & running_mask` rescans the entire frame axis once per trial, so the cost is O(n_trials × n_frames) — with up to 789 trials and ~28,000 frames per session this is ~22 M comparisons where one `np.argsort`/`np.unique` grouping pass over the valid frames would do. (2) The lick loop does an `np.abs(trial_frames - lf)` plus `np.argmin` for every lick of the trial, an O(n_licks × T) scan, where `np.searchsorted` on the sorted `trial_frames` would be O(n_licks log T); the `lick_by_trial` dict itself is built with a Python loop over all licks. (3) The trial body loop is pure Python over up to 789 trials per session, building three small arrays each; the per-frame quantities (position bin, speed bin, lick flag) could be computed once for the whole session and then sliced, as the reference does.

ii.
```python
for trial in range(ntrials):
    trial_mask = (ft_trInd == trial) & corridor_mask & running_mask
    trial_frames = np.where(trial_mask)[0]
```
```python
lick_by_trial = defaultdict(list)
for li in range(len(lick_fr)):
    lick_by_trial[int(lick_trind[li])].append(lick_fr[li])
...
    for lf in lick_by_trial[trial]:
        diffs = np.abs(trial_frames - lf)
        closest = np.argmin(diffs)
```

iii. Not discussed in CONVERSION_NOTES.md; the "Code inefficiencies identified" and "Code speedups added" placeholders in the Step 6 template were left empty in the final notes. In practice the cost is small relative to spike-file I/O (`proc` ≤ 3.0 s/session).

## 12-c. What processing does the code repeat multiple times?

i. (1) Every behavior file is read twice: once in `compute_running_speed_quartiles` and again in the main conversion loop, since `_beh_cache` is explicitly cleared between the two passes. (2) `_beh_cache.clear()` is called every 10 processed sessions, so a behavior file covering more than that span, or revisited later, is re-read from disk. (3) `load_beh_for_recording` is called twice per session in the main path (once at the top of `process_session`, and once already during the quartile pass). (4) The per-frame position, speed and lick quantities are recomputed inside the per-trial loop for each trial rather than once per session. (5) The `min(...)` frame-count reconciliation is done separately in `compute_running_speed_quartiles` and in `process_session`.

ii.
```python
speed_bins = compute_running_speed_quartiles(all_recordings)
_beh_cache.clear()
```
```python
if (i + 1) % 10 == 0:
    _beh_cache.clear()
```
```python
n = min(len(beh['ft_RunSpeed']), len(beh['ft_CorrSpc']), len(beh['ft_move']))
...
n_use = min(n_frames, len(beh['ft']))
```

iii. The caching and the periodic clearing are a deliberate memory/IO trade-off — the agent noted the full dataset must be held in one dictionary and the pickle reaches 161.65 GB, so behavior dictionaries are released to limit peak RSS. The whole quartile pre-pass costs only 1.8 s, so the duplicated reads are cheap.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) Neural data is stored as `float32` although the source is deconvolved traces and the decoder immediately reduces each session to 100 principal components; `float16` would halve the 161.65 GB file and the ~250 s pickle write, and the agent explicitly considered and rejected this. (2) All 4,691,034 neurons are stored, including the 585,641 with no retinotopic area assignment, even though the decoder projects each session onto 100 PCs. (3) `corridor_mask[n_use:] = False` is a no-op: `ft_CorrSpc` has already been sliced to `n_use`, so the assignment writes to an empty slice. (4) A per-session `dt` is computed and stored for all 89 sessions, but only the median of the 89 values is ever written to the metadata; the per-session values are discarded. (5) `lick_by_trial` is built over every lick in the session, including licks in the grey space and licks on stationary frames, most of which are then dropped by the 1-frame tolerance. (6) `t_load` / `t_proc` are carried through the result dict for printing only.

ii.
```python
neural = spk[:, trial_frames].astype(np.float32)  # keep float32 for decoder compatibility
```
```python
corridor_mask = ft_CorrSpc.copy()
corridor_mask[n_use:] = False  # safety
```
```python
dt_values.append(result['dt'])
...
median_dt_ms = float(np.median(dt_values) * 1000)
```

iii. From the trajectory the agent repeatedly weighed pruning the neural data — "the decoder uses PCA with 100 components. I could pre-compute PCA per session and store only the 100 components. But then brain_region_idx wouldn't work" — and concluded "the task says to include all neurons", so it kept everything. The `float32` choice is annotated "keep float32 for decoder compatibility" even though the agent had earlier judged that "Float16 should be fine for the neural data since PCA will lose precision anyway".
