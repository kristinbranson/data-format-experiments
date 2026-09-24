# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything is read from the three subfolders of `/app/data`: `beh/` (behavior), `spk/` (deconvolved calcium traces) and `retinotopy/` (visual area of each neuron). `beh/Imaging_Exp_info.npy` is read first as the master index; it is a dict mapping experiment type -> list of recording entries (`mname`, `datexp`, `blk`, sometimes `stimtype`). `get_all_unique_sessions()` flattens that index into one record per unique `mname_datexp_blk`. For each session, behavior is fetched by `get_session_beh()`, which re-scans the whole index, loads the entire `Beh_<exp_type>.npy` dict for the first experiment type that contains the recording, and returns the entry under `mname_datexp_blk` (falling back to `mname_datexp_blk_stimtype`). Spikes and retinotopy are loaded per session by `load_spk()` / `load_retino()`. Behavior files are **not** grouped, so the same 200–430 MB `Beh_*.npy` file is re-read from disk on every session lookup (once per session in the speed-quartile pass, once per session in the conversion pass, and again for each plotted session).

ii.
```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
all_sessions = get_all_unique_sessions(exp_info)
```
```python
def get_session_beh(session_key, exp_info):
    ...
    for exp_type, sessions in exp_info.items():
        for s in sessions:
            if s['mname'] == mname and s['datexp'] == datexp and s['blk'] == blk:
                beh_path = os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy')
                beh_all = np.load(beh_path, allow_pickle=True).item()
                for key_fmt in [f'{mname}_{datexp}_{blk}',
                                f'{mname}_{datexp}_{blk}_{s.get("stimtype", "")}']:
                    if key_fmt in beh_all:
                        return beh_all[key_fmt], exp_type, s
```
```python
def load_spk(mname, datexp, blk, root=DATA_ROOT):
    fn = os.path.join(root, 'spk', f'{mname}_{datexp}_{blk}_neural_data.npy')
    dat = np.load(fn, allow_pickle=True).item()
    spk = np.concatenate(dat['spks'], axis=0)  # (n_neurons, n_frames)
    return spk

def load_retino(mname, datexp, root=DATA_ROOT):
    fn = os.path.join(root, 'retinotopy', f'{mname}_{datexp}_trans.npz')
    return np.load(fn, allow_pickle=True)['iarea']
```

iii. From CONVERSION_NOTES Step 1/2: the agent mirrored the reference `utils.load_spk` (concatenate `spks` across imaging planes), `utils.load_retino` and `utils.load_exp_beh`. It documented that "23 experiment types, 142 total entries (many sessions appear in multiple exp types)" and that "Same physical recording has identical behavioral data across different exp types (only stim_id mapping differs)", which is why it takes whichever experiment type is found first. Key decision 6: "Session deduplication: Each physical recording (mname_datexp_blk) included once. Use first available exp_type for behavioral data."

## 1-b. How are the data split into subjects?

i. The subject is the `mname` field carried on every index entry and propagated on each session record. Subjects are collected during the conversion loop in order of first appearance (`subjects_seen` dict), and `subject_idx` is that dict's index for each session. Result: 19 subjects over 89 sessions, matching the paper. Note the list is in first-encounter order, not sorted.

ii.
```python
sessions.append({'key': key, 'mname': s['mname'], 'datexp': s['datexp'],
                 'blk': s['blk'], 'exp_type': exp_type, 'db': s})
```
```python
mname = result['mname']
if mname not in subjects_seen:
    subjects_seen[mname] = len(subjects_seen)
subj_idx = subjects_seen[mname]
...
subjects = list(subjects_seen.keys())
subject_idx = np.array(subject_idx_list, dtype=np.int64)
```

iii. CONVERSION_NOTES Step 2/3: the index already names the mouse, so no split needs to be derived. The agent verified "19 unique mouse IDs / 19 mice in spk/ / '19 mice'" as a consistency check (Step 4 table, Step 10 check 1).

## 1-c. How are the data split into sessions?

i. A session is one physical recording = one mouse on one date in one block, keyed `mname_datexp_blk`. Because the same recording is listed under several experiment types in `Imaging_Exp_info.npy`, a `seen` set keeps only the first occurrence, yielding 89 unique sessions. The session key also names the spike file and (with the date only) the retinotopy file.

ii.
```python
def get_all_unique_sessions(exp_info):
    seen = set()
    sessions = []
    for exp_type, session_list in exp_info.items():
        for s in session_list:
            key = f"{s['mname']}_{s['datexp']}_{s['blk']}"
            if key not in seen:
                seen.add(key)
                sessions.append({...})
    return sessions
```

iii. CONVERSION_NOTES Step 2: "23 experiment types, 142 total entries (many sessions appear in multiple exp types)"; Step 10 check 2: "Unique sessions from exp_info: 89 unique keys confirmed. PASS" against the paper's "We performed 89 recordings in 19 mice".

## 1-d. How are the data split into trials?

i. Trials are the ones the behavior declares (`beh['ntrials']`), indexed by the per-frame label `ft_trInd`. For each trial the agent keeps the frames satisfying **three** conditions: the frame belongs to that trial (`ft_trInd == trial_idx`), the frame is inside the textured corridor (`ft_CorrSpc`, i.e. 0–4 m, excluding the 2 m grey space), **and the VR/animal was running** (`ft_move > 0`). Trials are variable length and nothing is padded. This third condition is the main departure from the human reference, which keeps every corridor frame. It removes roughly 40% of corridor frames (median retained trial length 24 frames vs 35 for all corridor frames in `TX108_2023_03_25_1`; dataset-wide mean T = 22.25 vs 32.63 for the reference), and it makes the retained frames temporally non-contiguous — in the session I checked, 77.5% of trials contain at least one interior gap and the largest single gap is 408 frames (~2 min).

ii.
```python
RUNNING_THRESHOLD = 0     # ft_move > 0 means VR is moving
...
for trial_idx in range(ntrials):
    # Find valid frames: in corridor, running, and matching trial
    trial_mask = (ft_trInd == trial_idx) & ft_CorrSpc & (ft_move > RUNNING_THRESHOLD)
    frame_indices = np.where(trial_mask)[0]
    if len(frame_indices) < 2:
        continue  # skip trials with too few frames
```

iii. CONVERSION_NOTES Step 3 quotes the paper: "We only considered timepoints during running for analysis, which removed time periods when the task mice stopped to collect water rewards", and Step 1 notes that the reference code "consistently uses `VRmove = ft_move > 0` (running) and `ft_CorrSpc` (in corridor) as frame mask" (e.g. `Get_dprime_selective_neuron`). Key decisions 2 and 3: "Corridor-only frames: Include only ft_CorrSpc == True. Position output requires 0-4m range." and "Running-only frames: Follow reference paper". The trajectory shows the agent explicitly weighed the alternative (step 39: "including only running frames breaks uniform time spacing … I think it's better to keep all frames in the corridor") before reversing at step 47 in favour of matching the paper's stated methodology.

## 1-e. How are trials filtered based on quality controls?

i. Essentially none beyond the frame mask. A trial is dropped only if fewer than 2 frames survive the mask, or if its `WallName` is not in the hard-coded texture table. In practice neither fired: the conversion log shows `kept 38110 / total 38110` trials, and there are no WARNING lines in `conversion_full_out.txt`. A session is dropped if it ends with fewer than 2 trials (never fired; all 89 sessions kept). There is **no** trial-duration outlier removal. Consequently trials in which the animal parked for many minutes are retained: `time_since_trial_start` reaches 1765 s and `time_to_sound_cue` spans [−1763.3, 723.5] s, versus [0, 74.8] s and [−72.2, 73.4] s in the human reference, which drops trials longer than the 99th percentile (382 of 38110).

ii.
```python
if len(frame_indices) < 2:
    continue  # skip trials with too few frames
```
```python
stim_cat = STIM_CATEGORY_MAP.get(wall_name, None)
if stim_cat is None:
    print(f"  WARNING: Unknown stimulus '{wall_name}' in trial {trial_idx}, skipping")
    continue
```
```python
if len(neural_trials) < 2:
    print(f"  WARNING: {key} has {len(neural_trials)} valid trials, skipping")
    return None
```

iii. CONVERSION_NOTES Step 3 "Trial curation rules": "No explicit trial exclusion in reference code for general analysis; Only running frames (ft_move > 0) within corridor (ft_CorrSpc) are used; Trials with 0 valid frames after filtering should be excluded." The extreme trial was noticed and explicitly accepted — Step 10 check 12: "One trial with time_since_start=1765s (session 49, trial 391, 36 frames) — valid due to mouse stopping/restarting during long trial. PASS", and trajectory step 95: "This is actually correct behavior! The time_since_trial_start reflects the actual elapsed time from corridor entry … This gives the decoder information about the pacing of the trial."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<mname>_<datexp>_<blk>_neural_data.npy`, a list of one (neurons × frames) array per imaging plane, concatenated along the neuron axis. The per-neuron visual area used to curate and label neurons comes from `iarea` in `retinotopy/<mname>_<datexp>_trans.npz`.

ii.
```python
dat = np.load(fn, allow_pickle=True).item()
spk = np.concatenate(dat['spks'], axis=0)  # (n_neurons, n_frames)
```
```python
iarea = load_retino(mname, datexp)
```

iii. CONVERSION_NOTES Step 1 documents `load_spk(db, root)` in the reference `utils.py` as `np.concatenate([nspk for nspk in np.load(path).item()['spks']], 0)` and the agent copied that convention. Step 3: "All analyses based on deconvolved fluorescence traces (not dF/F, not raw fluorescence)".

## 2-b. How is the `neural` data processed?

i. No processing at all: the traces are already Suite2p-deconvolved, so no dF/F and no deconvolution are applied. The columns corresponding to the trial's retained frames are sliced out and cast to `float32`. Trials keep their native, variable length; nothing is padded or truncated. (The human reference stores `float16`; the `float32` choice is the main reason the output pickle is 151.6 GB.)

ii.
```python
neural = spk[:, frame_indices].astype(np.float32)  # (n_neurons, n_timepoints)
```

iii. CONVERSION_NOTES Step 3: "All analyses based on deconvolved fluorescence traces (not dF/F, not raw fluorescence)"; Step 5 mapping table: "spks (concatenated planes) -> neural … Deconvolved fluorescence, float32". Step 10 check 4 verified the values byte-for-byte: "Loaded original spk data for session 0 trial 0, applied same frame mask … compared with `np.allclose` — exact match."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only filter is anatomical: neurons whose retinotopic area code is `-1` (outside visual cortex) or `7` (unassigned) are dropped; everything else is kept and mapped to one of V1 (8), mHV (0,1,2,9), lHV (5,6), aHV (3,4). No firing-rate, SNR or d′ filter is applied. `iarea` in this dataset only takes values −1…9, so excluding {−1, 7} is exactly equivalent to the reference's keep-list; the resulting counts are identical to the reference solution (4,105,393 kept neurons; V1 1,833,035 / mHV 1,108,860 / lHV 495,318 / aHV 668,180). A guard also truncates `spk`/`iarea` to a common length if their neuron counts disagree (never triggered).

ii.
```python
EXCLUDED_IAREA = {-1, 7}  # outside visual cortex / unassigned

def get_neuron_mask(iarea):
    mask = np.ones(len(iarea), dtype=bool)
    for exc in EXCLUDED_IAREA:
        mask &= (iarea != exc)
    return mask
```
```python
neuron_mask = get_neuron_mask(iarea)
brain_region_idx = get_brain_region_indices(iarea, neuron_mask)
spk = spk[neuron_mask]
```

iii. CONVERSION_NOTES Step 1: "`Get_density_map()` — Filter neurons: `(arid!=-1) & (arid!=7)` to exclude non-visual-cortex neurons"; Step 3: "No additional quality filtering (no minimum firing rate, no signal-to-noise cutoff)". Step 10 check 3 spot-checked three sessions against the raw spk + retinotopy files.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is trial start / corridor entry. Each trial's array begins at its **first retained frame**, i.e. the first frame that is simultaneously inside the texture corridor and flagged as running — not necessarily the corridor-entry frame `StartFr`. Trials run to their own last retained corridor frame, are variable length, and are neither cut to a common window nor padded. Metadata records this honestly: `temporal_alignment_event = 'Trial start (corridor entry, first running frame in texture corridor)'`, `off_start = 0.0`, `off_end = None`.

ii.
```python
trial_mask = (ft_trInd == trial_idx) & ft_CorrSpc & (ft_move > RUNNING_THRESHOLD)
frame_indices = np.where(trial_mask)[0]
neural = spk[:, frame_indices].astype(np.float32)
```
```python
'temporal_alignment_event': 'Trial start (corridor entry, first running frame in texture corridor)',
'off_start': 0.0,   # alignment is at trial start
'off_end': None,    # variable trial lengths
```

iii. CONVERSION_NOTES Step 5 key decision 1/2: native imaging frames from corridor entry; the format only requires a common bin *size*, not a common trial length, so variable-length trials are kept. Trajectory step 42: "For the decoder, I want to mark trial start at corridor entry and keep only corridor frames since the position bins correspond to that 0-4m range."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling: the imaging frame is the bin. The frame rate is 3.178 Hz, so `time_bin_size` is recorded as a hard-coded 315.0 ms (the human reference records 1000/3.17 = 315.457 ms; the agent also stores `frame_rate_hz: 3.178` in metadata). Note that because non-running frames are dropped inside trials, consecutive bins within a trial are not always 315 ms apart — the bin width is constant but the sampling grid has gaps.

ii.
```python
'time_bin_size': 315.0,  # approximate, in ms (1/3.178 Hz * 1000)
'frame_rate_hz': 3.178,
```
A per-session frame interval is also computed but never used downstream:
```python
if frame_dt_s is None:
    frame_dt_s = float(np.median(np.diff(ft)) * 86400)  # days to seconds
```

iii. CONVERSION_NOTES Step 5 key decision 1: "Time bins = native imaging frames (~315ms): Matches reference code's temporal resolution. No interpolation needed." Step 4 consistency table: "Frame rate | Computed from ft: 3.178 Hz | … | '3.17 Hz' (data_process_script) | Consistent."

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the fractional imaging-frame number of the sound cue on each trial) and `ft` (the MATLAB-datenum timestamp of every imaging frame).

ii.
```python
SoundFr = beh['SoundFr']
ft = beh['ft'][:nfr]
...
sound_fr = SoundFr[trial_idx]
sound_time_s = np.interp(sound_fr, np.arange(nfr), ft[:nfr]) * 86400
```

iii. CONVERSION_NOTES Step 5 mapping table: "SoundFr, ft -> input[0]: time_to_sound_cue". Trajectory step 42: "`StartFr`, `EndFr`, `GrayFr`, `SoundFr`, `LickFr` are all stored as float frame indices (non-integer because they were interpolated from timestamps)", which is why the cue frame is interpolated onto the time axis rather than rounded.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The fractional cue frame is linearly interpolated onto the frame-time axis, timestamps are converted from days to seconds (×86400), and the input is `cue_time − frame_time` for each retained frame of the trial — positive before the cue, negative after, matching the "time **to**" convention. Stored as float32.

ii.
```python
frame_times = ft[frame_indices] * 86400  # convert days to seconds
sound_time_s = np.interp(sound_fr, np.arange(nfr), ft[:nfr]) * 86400
time_to_sound = sound_time_s - frame_times  # positive before cue, negative after
...
input_data = np.vstack([
    time_to_sound.reshape(1, -1).astype(np.float32), ...])
```

iii. CONVERSION_NOTES Step 5: "Positive before cue, negative after". Step 10 check 9: "Time to sound cue: Recomputed from original timestamps for session 0 trial 0 — exact match with `np.allclose`. Sign convention correct (positive before cue, negative after). PASS."

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at exactly the same `frame_indices` used to slice the neural columns for that trial, so it shares the neural grid and length bin-for-bin.

ii.
```python
frame_indices = np.where(trial_mask)[0]
neural = spk[:, frame_indices].astype(np.float32)
frame_times = ft[frame_indices] * 86400
time_to_sound = sound_time_s - frame_times
```

iii. Every stream in this dataset is indexed by imaging frame number, so using one `frame_indices` array for neural, inputs and outputs guarantees alignment (CONVERSION_NOTES Step 3: "Behavior data time bin — same (aligned to neural frames)").

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the `datexp` date string of each recording in `Imaging_Exp_info.npy`, grouped by `mname`. No behavior variable is used.

ii.
```python
date = datetime.strptime(s['datexp'], '%Y_%m_%d')
mouse_dates[mname].append((date, s['key']))
```

iii. CONVERSION_NOTES Step 5 mapping table: "Session dates -> input[1]: day_of_training … Computed from date strings."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, sessions are sorted by date and the value is the **calendar-day difference** from that mouse's earliest recording: `(date − first_date).days`. It is computed once over all 89 sessions (so sample and full runs agree) and then broadcast as a constant row across every bin of every trial in the session. The resulting range is [0, 92] days (the human reference instead uses the ordinal index of the recorded session, range [0, 7]).

ii.
```python
def compute_day_of_training(sessions):
    ...
    day_map = {}
    for mname, dates in mouse_dates.items():
        dates.sort(key=lambda x: x[0])
        first_date = dates[0][0]
        for date, key in dates:
            day_map[key] = (date - first_date).days
    return day_map
```
```python
day_of_training = float(day_map.get(key, 0))
...
np.full((1, n_tp), day_of_training, dtype=np.float32),
```

iii. CONVERSION_NOTES Step 5 key decision 8: "Day of training: Compute as days since first session date for each mouse." Step 10 check 8: "Day of training: Verified for 3 sessions against independently computed dates. PASS." The trajectory (step 47) shows the agent first considered "session number as day of training" before settling on the calendar-date difference.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `ft` alone — the timestamps of the trial's retained frames. `StartFr` (the corridor-entry frame, which the human reference uses) is **not** used; the origin is the trial's first retained frame.

ii.
```python
frame_times = ft[frame_indices] * 86400  # convert days to seconds
time_since_start = frame_times - frame_times[0]
```

iii. CONVERSION_NOTES Step 5 mapping table: "Frame index within trial -> input[2]: time_since_trial_start … Seconds from corridor entry." The metadata string makes the operational definition explicit: trial start = "corridor entry, first running frame in texture corridor".

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Timestamps are converted from MATLAB datenum days to seconds (×86400) and the first retained frame's time is subtracted, so the value is exactly 0.0 in the first bin of every trial and increases monotonically. Stored as float32. Because non-running frames are dropped but real elapsed time is used, the value can jump across gaps; the full-dataset range is [0, 1765.0] s.

ii.
```python
time_since_start = frame_times - frame_times[0]
...
time_since_start.reshape(1, -1).astype(np.float32),
```

iii. Trajectory step 95: "the time_since_trial_start reflects the actual elapsed time from corridor entry to the latest frame, even if the mouse wasn't always running. This gives the decoder information about the pacing of the trial." CONVERSION_NOTES Step 10 check 12 accepts the 1765 s maximum as valid.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed from `ft` at the same `frame_indices` used for the neural columns, so it is bin-for-bin aligned with the neural array and has the same length.

ii.
```python
frame_indices = np.where(trial_mask)[0]
neural = spk[:, frame_indices].astype(np.float32)
frame_times = ft[frame_indices] * 86400
time_since_start = frame_times - frame_times[0]
```

iii. Same rationale as 3-c: all streams share the imaging-frame index.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `beh['isRew']`, the per-trial boolean flag marking trials run in the rewarded corridor.

ii.
```python
isRew = beh['isRew']
...
np.full((1, n_tp), float(isRew[trial_idx]), dtype=np.float32),
```

iii. CONVERSION_NOTES Step 5 mapping table: "isRew -> input[3]: reward_availability | 1 if isRew[trial]==True, else 0. Discrete, per-trial."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. None beyond a bool→float cast and broadcast as a constant row across the trial's bins. 28 of 89 sessions contain rewarded trials; the rest (unsupervised/naive animals) are all zero.

ii.
```python
reward_avail = np.full(1, float(isRew[trial_idx]), dtype=np.float32)   # computed, then unused
...
np.full((1, n_tp), float(isRew[trial_idx]), dtype=np.float32),
```

iii. CONVERSION_NOTES Step 10 check 10: "Reward availability: 28 sessions with reward, 61 without — reasonable for mix of supervised/unsupervised experiments. PASS." (Note the `reward_avail` scalar on the first line is dead code — the broadcast row is built independently.)

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `beh['WallName']`, the per-trial name of the corridor wall texture. (`TrialStim` is not used.)

ii.
```python
WallName = beh['WallName']
...
wall_name = str(WallName[trial_idx])
stim_cat = STIM_CATEGORY_MAP.get(wall_name, None)
```

iii. CONVERSION_NOTES Step 5 mapping table: "WallName -> output[0]: stimulus_category".

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The 15 wall names present in the dataset are collapsed to four base textures via a hard-coded table — circle (circle1/2/3), leaf (leaf1/2/3, leaf1_swap1/2), rock (rock1/2), wood (wood1/2/5, wood1_swap1/2) — and stored as the index into `STIM_CATEGORIES`. The value is per-trial and is broadcast across the trial's bins as an int64 row. Resulting fractions: circle 0.311, leaf 0.470, rock 0.085, wood 0.135.

ii.
```python
STIM_CATEGORY_MAP = {}
for s in ['circle1', 'circle2', 'circle3']:                          STIM_CATEGORY_MAP[s] = 'circle'
for s in ['leaf1', 'leaf2', 'leaf3', 'leaf1_swap1', 'leaf1_swap2']:  STIM_CATEGORY_MAP[s] = 'leaf'
for s in ['rock1', 'rock2']:                                         STIM_CATEGORY_MAP[s] = 'rock'
for s in ['wood1', 'wood2', 'wood5', 'wood1_swap1', 'wood1_swap2']:  STIM_CATEGORY_MAP[s] = 'wood'
STIM_CATEGORIES = ['circle', 'leaf', 'rock', 'wood']
...
np.full((1, n_tp), stim_cat_idx, dtype=np.int64),
```

iii. CONVERSION_NOTES Step 5 key decision 5: "Stimulus categories: Use broad categories … rather than specific stimulus IDs. The 'e.g. circle, leaf, etc.' in the decoder task suggests this." The code comment records that "paper calls 4 categories 'circle, leaf, rock, brick' but actual data labels use 'wood'. We use data labels." The map was corrected after an initial version missed `wood*`, `rock2` and `circle3` (trajectory step 114). Step 10 check 7: "All 4 categories present (circle=11,619, leaf=17,761, rock=3,278, wood=5,452). Constant within each trial. PASS."

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `beh['LickFr']` (the fractional imaging-frame number of every lick) together with `beh['LickTrind']` (the trial each lick belongs to), used to select the licks of the current trial.

ii.
```python
LickFr = beh['LickFr']
LickTrind = beh['LickTrind']
...
trial_lick_mask = (LickTrind == trial_idx)
trial_lick_frs = LickFr[trial_lick_mask]
```

iii. CONVERSION_NOTES Step 5 mapping table: "LickFr, LickTrind -> output[1]: licking | Binary time series: 1 if any lick in frame's time bin, 0 otherwise."

## 8-b. What processing is involved in computing `output` *Licking*?

i. A binary row of length `n_tp` is built; for each lick belonging to the trial, `searchsorted` locates its insertion point in the trial's retained-frame list and the closer of the two neighbouring retained frames is set to 1. Indices past the end are clamped to the last bin. Because the retained-frame list excludes all non-running frames (and all grey-space frames), a large fraction of licks do not land on a retained frame and are **relocated** to the nearest one: in `TX108_2023_03_25_1`, 1277 of 2721 licks (47%) fall on a non-retained frame, and 137 (5%) occur after the trial's last retained corridor frame and are all collapsed onto the final bin. The aggregate lick fraction nevertheless comes out at 0.041, essentially identical to the human reference's 0.0414.

ii.
```python
lick_binary = np.zeros(n_tp, dtype=np.float32)
if len(trial_lick_frs) > 0:
    fi_sorted = frame_indices.astype(float)
    lick_bin_idx = np.searchsorted(fi_sorted, trial_lick_frs)
    for li in range(len(trial_lick_frs)):
        idx = lick_bin_idx[li]
        if idx >= n_tp:
            idx = n_tp - 1
        elif idx > 0:
            if abs(trial_lick_frs[li] - fi_sorted[idx-1]) < abs(trial_lick_frs[li] - fi_sorted[idx]):
                idx = idx - 1
        lick_binary[idx] = 1.0
```

iii. The agent's stated intent (CONVERSION_NOTES Step 5) was "1 if any lick in frame's time bin, 0 otherwise"; the trajectory summary at step 114 describes the implementation as "searchsorted-based nearest-frame assignment". The rationale implied by trajectory step 39 is that "excluding non-running frames would strip out useful licking information", so licks are preserved by snapping rather than dropped. Step 10 check 6 verified only that all values are strictly 0/1.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Onto the same grid: the lick row is indexed by position within `frame_indices`, the identical array used to slice the neural columns, so the row length always equals the neural trial length. The alignment of an individual lick is approximate rather than exact because of the nearest-retained-frame snapping described in 8-b.

ii.
```python
frame_indices = np.where(trial_mask)[0]
neural = spk[:, frame_indices].astype(np.float32)
fi_sorted = frame_indices.astype(float)
lick_bin_idx = np.searchsorted(fi_sorted, trial_lick_frs)
```

iii. `LickFr` is expressed in imaging-frame units, so no cross-clock conversion is needed (CONVERSION_NOTES Step 3: behavior variables are "mapped to neural frame times").

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `beh['ft_Pos']`, the per-imaging-frame position along the corridor in decimeters (0–40 across the texture, 40–60 through the grey space).

ii.
```python
ft_Pos = beh['ft_Pos'][:nfr]
...
positions = ft_Pos[frame_indices]
```

iii. CONVERSION_NOTES Step 5 mapping table: "ft_Pos -> output[2]: position". Trajectory step 42: "Position runs from 0 to 60 (Corridor_Length). The texture area spans 0-40 units (0-4m) and the grey space spans 40-60 units (4-6m)."

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Positions of the retained frames are clipped into [0, 40) and digitized into four 10-decimeter (1 m) bins, stored as int64 and broadcast into the output array. Since only `ft_CorrSpc` frames are retained, `ft_Pos` is already within [0, 39.97] and the clip is effectively a no-op. Resulting distribution is near-uniform: 0.250 / 0.249 / 0.250 / 0.252.

ii.
```python
POSITION_BIN_EDGES_DM = np.array([0, 10, 20, 30, 40])  # decimeter edges
...
positions = np.clip(positions, 0, CORRIDOR_LENGTH_DM - 0.001)
pos_bin = np.digitize(positions, POSITION_BIN_EDGES_DM[1:])  # 0,1,2,3
pos_bin = np.clip(pos_bin, 0, N_POSITION_BINS - 1)
```

iii. CONVERSION_NOTES Step 5 mapping table: "Discretize 0-40dm into 4 bins: [0-10), [10-20), [20-30), [30-40] … 4 bins of 1m each", matching the decoder task's "4 equal-length, 1-m-long spatial bins". Step 10 check 5: "Position binning: All position bins in [0,3] range across all trials. PASS."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Fixed spatial thresholds at 10, 20 and 30 decimeters (1, 2, 3 m), giving labels `['0-1m', '1-2m', '2-3m', '3-4m']`. The thresholds are absolute (not data-driven), exactly as the task specifies.

ii.
```python
N_POSITION_BINS = 4       # 4 bins of 1m each (0-10, 10-20, 20-30, 30-40 dm)
POSITION_BIN_EDGES_DM = np.array([0, 10, 20, 30, 40])
...
'output_values': [..., ['0-1m', '1-2m', '2-3m', '3-4m'], ...]
```

iii. Directly from the Decoder Task specification; the grey space (40–60 dm) is excluded from the data entirely so that the four bins tile the whole retained range (trajectory step 42: "Since grey space positions exceed 40 and don't fit these bins, I should restrict the analysis to corridor frames only").

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` has one value per imaging frame, so it is read at the same `frame_indices` as the neural columns and is bin-for-bin aligned with them.

ii.
```python
frame_indices = np.where(trial_mask)[0]
neural = spk[:, frame_indices].astype(np.float32)
positions = ft_Pos[frame_indices]
```

iii. Same frame-index grid as all other streams.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `beh['ft_RunSpeed']`, the running speed at each imaging frame (cm/s).

ii.
```python
ft_RunSpeed = beh['ft_RunSpeed'][:nfr]
...
speeds = ft_RunSpeed[frame_indices]
```

iii. CONVERSION_NOTES Step 5 mapping table: "ft_RunSpeed -> output[3]: running_speed | Discretize into 4 quartile bins (computed across all data)."

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A single **global** set of quartile edges is computed once, before conversion, over all sessions: for each session the corridor+running frames' speeds are taken, subsampled to at most 10,000 values with a fixed RNG seed, and all sessions' samples are pooled (778,898 values) to give `np.percentile(..., [0,25,50,75,100])` = `[-19.17, 12.20, 25.01, 40.51, 321.89]` cm/s. Each trial's speeds are then digitized against those edges. Globally this yields 0.246 / 0.248 / 0.252 / 0.255 of the data per bin, but per-session distributions are highly non-uniform (one session has 95.5% of its bins in the lowest quartile). The human reference instead ranks within each session, giving exactly 25% per session.

ii.
```python
def compute_speed_quartile_edges(sessions, exp_info):
    all_speeds = []
    for s in sessions:
        ...
        mask = ft_CorrSpc & (ft_move > RUNNING_THRESHOLD)
        speeds = ft_RunSpeed[mask]
        if len(speeds) > 10000:
            rng = np.random.default_rng(42)
            speeds = rng.choice(speeds, 10000, replace=False)
        all_speeds.append(speeds)
    all_speeds = np.concatenate(all_speeds)
    edges = np.percentile(all_speeds, [0, 25, 50, 75, 100])
    return edges
```

iii. CONVERSION_NOTES Step 5 key decision 7: "Running speed quartiles: Compute across ALL sessions' valid frames to ensure consistent bin edges." The edges are also stored in metadata (`speed_quartile_edges`) and used to build human-readable labels.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. By `np.digitize` against the three interior global percentile edges (12.20, 25.01, 40.51 cm/s), clipped to [0, 3]. The `output_values` labels are generated from the edges, e.g. `'-19.2-12.2 cm/s'`, `'12.2-25.0 cm/s'`, `'25.0-40.5 cm/s'`, `'40.5-321.9 cm/s'`.

ii.
```python
speed_bin = np.digitize(speeds, speed_edges[1:-1])  # 0,1,2,3
speed_bin = np.clip(speed_bin, 0, N_SPEED_BINS - 1)
```
```python
for i in range(N_SPEED_BINS):
    speed_labels.append(f'{speed_edges[i]:.1f}-{speed_edges[i+1]:.1f} cm/s')
```

iii. Value thresholds are workable here because the zero-speed pile-up has already been removed by the `ft_move > 0` frame mask; the task's requirement of "4 bins, each corresponding to 25% of the data" is met at the dataset level.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` has one value per imaging frame, so it is read at the same `frame_indices` as the neural columns and is bin-for-bin aligned.

ii.
```python
frame_indices = np.where(trial_mask)[0]
neural = spk[:, frame_indices].astype(np.float32)
speeds = ft_RunSpeed[frame_indices]
```

iii. Same frame-index grid as all other streams.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several guards, all documented:
- Behavior can run past (or short of) the imaging: every behavior stream is truncated to `nfr = min(spk.shape[1], len(beh['ft_trInd']))`.
- Neuron-count mismatch between `spks` and `iarea`: both truncated to the shorter length, with a warning (never fired on this dataset).
- Session with no behavior entry found: warned and skipped.
- Trial with fewer than 2 retained frames: skipped.
- Trial whose `WallName` is not in the texture table: warned and skipped (this fired during development for `wood*`, `rock2`, `circle3`, and the map was extended).
- Session left with fewer than 2 trials: warned and skipped.
No warnings appear in `conversion_full_out.txt`, i.e. none of these paths triggered on the final run. Licks past the trial's last retained frame are **not** dropped; they are clamped into the final bin (see 8-b).

ii.
```python
nfr = min(n_frames, nfr_beh)
ft_trInd = beh['ft_trInd'][:nfr]
...
if len(iarea) != n_neurons_raw:
    print(f"  WARNING: neuron count mismatch for {key}: spk={n_neurons_raw}, retino={len(iarea)}")
    min_n = min(n_neurons_raw, len(iarea))
    spk = spk[:min_n]; iarea = iarea[:min_n]; n_neurons_raw = min_n
```
```python
if beh is None:
    print(f"  WARNING: No behavioral data found for {key}, skipping")
    return None
```

iii. CONVERSION_NOTES Step 10 check 12: "Extreme values / data integrity: No NaN or Inf in neural, input, or output data." The truncation mirrors the reference code's `beh[...][:nfr]` convention documented in Step 1.

## 12-a. What are the most time-consuming steps of the code?

i. From `conversion_full_out.txt` (total 1386.9 s for 89 sessions): the per-session "load" phase — loading the ~4.5 GB spike file **plus** re-reading an entire `Beh_*.npy` behavior file — dominates at 577.8 s; the per-trial Python processing loop takes 394.3 s; the per-neuron brain-region assignment ("filter") takes 163.3 s; and writing the 151.6 GB pickle accounts for the remaining ~240 s. The speed-quartile pre-pass adds only 6.3 s. The dominant cost is therefore raw I/O on the ~405 GB of spike files, which is largely irreducible — but the 151.6 GB write is twice what it needs to be because neural data is stored as `float32` rather than `float16`. Total runtime (23 min) exceeds the instructions' 15-minute guideline and no optimization pass was performed.

ii. The code is instrumented per phase, which is how the above was measured:
```python
print(f"  {key}: {n_neurons} neurons, {len(neural_trials)}/{ntrials} trials, "
      f"load={t_load-t0:.1f}s, filter={t_filter-t_load:.1f}s, process={t_process-t_filter:.1f}s")
```
```python
neural = spk[:, frame_indices].astype(np.float32)   # float16 would halve the 151.6 GB output
```

iii. CONVERSION_NOTES has no bottleneck analysis: the Step 7 "Run Time Estimates" table and the Step 6 "Code inefficiencies identified / Code speedups added" fields of the template were left out of the final notes. The timing instrumentation itself satisfies the instruction to "print timing information to find bottlenecks", but the findings were never acted on.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Three clear cases:
- `get_brain_region_indices` iterates over every kept neuron in Python (4,105,393 iterations across the dataset) to do a small dictionary lookup — this is the entire 163 s "filter" cost and is a one-line `np.isin`/lookup-table operation.
- The trial loop recomputes `(ft_trInd == trial_idx) & ft_CorrSpc & (ft_move > ...)` over the whole session's frame vector once per trial (O(n_trials × n_frames)); a single grouping pass over `ft_trInd` would do.
- The nearest-frame lick assignment loops in Python over each lick; the whole thing is expressible with `np.searchsorted` plus a vectorized comparison.
(Also in plotting: `reward_vals.count(...)` and repeated list comprehensions, but those are off the critical path.)

ii.
```python
def get_brain_region_indices(iarea, neuron_mask):
    filtered_iarea = iarea[neuron_mask]
    region_idx = np.zeros(len(filtered_iarea), dtype=np.int64)
    for i, ia in enumerate(filtered_iarea):          # 4.1M Python iterations
        region_name = BRAIN_REGION_MAP.get(int(ia), None)
        ...
```
```python
for trial_idx in range(ntrials):
    trial_mask = (ft_trInd == trial_idx) & ft_CorrSpc & (ft_move > RUNNING_THRESHOLD)
    frame_indices = np.where(trial_mask)[0]          # full-session scan per trial
```
```python
for li in range(len(trial_lick_frs)):                # per-lick Python loop
    idx = lick_bin_idx[li]
    ...
```

iii. Not discussed in CONVERSION_NOTES — the "Code inefficiencies identified" field was left blank. The per-trial mask and the lick loop are at least partially defensible as clear code; the per-neuron region loop is not, and it is the second-largest CPU cost in the script.

## 12-c. What processing does the code repeat multiple times?

i. The behavior files are re-read from disk on every session lookup. `get_session_beh()` re-scans `exp_info` and calls `np.load` on a whole `Beh_<exp_type>.npy` (200–430 MB, 6.6 GB total across 23 files) each time it is called, and it is called once per session inside `compute_speed_quartile_edges` (89 calls), once per session inside `process_session` (89 calls), and again inside `plot_processing` for each plotted session — roughly 180 full-file loads where 23 would suffice. The human reference explicitly groups sessions by behavior file and loads each exactly once. Secondary repetitions: `exp_info` is linearly re-searched on every lookup; the retained-frame mask components are recomputed per trial; and `float(isRew[trial_idx])` is computed twice per trial (once into the unused `reward_avail`).

ii.
```python
# called once per session in the quartile pass and again in the conversion pass
beh, _, _ = get_session_beh(s['key'], exp_info)
...
def get_session_beh(session_key, exp_info):
    for exp_type, sessions in exp_info.items():
        for s in sessions:
            if s['mname'] == mname and ...:
                beh_all = np.load(beh_path, allow_pickle=True).item()   # whole file, every call
```

iii. Not identified in CONVERSION_NOTES. Empirically the cost is smaller than it looks (the quartile pre-pass over all 89 sessions completed in 6.3 s, so the OS page cache absorbs most of it), but it is avoidable redundancy that the reference design eliminates by construction, and it is the reason the per-session "load" timings mix behavior and spike I/O.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Mostly small dead computations, plus one substantive storage cost:
- `frame_dt_s = np.median(np.diff(ft)) * 86400` is computed for every session and never used (the metadata bin size is hard-coded to 315.0 instead).
- `reward_avail = np.full(1, float(isRew[trial_idx]))` is built and discarded; the broadcast row is constructed separately.
- `valid_trial_indices` is accumulated per session but never stored or returned.
- `n_neurons_raw` is carried in the result dict and never used.
- `plot_processing` receives `day_map` (and unpacks `datexp`, `blk`) without using them.
- Substantively: neural data is stored as `float32` rather than `float16`, doubling the pickle to 151.6 GB and adding ~2 minutes of write time, for precision the deconvolved traces do not carry meaningfully for decoding.
Nothing scientifically wasteful is computed — there is no discarded heavy processing such as position-space interpolation.

ii.
```python
if frame_dt_s is None:
    frame_dt_s = float(np.median(np.diff(ft)) * 86400)  # never used again
```
```python
reward_avail = np.full(1, float(isRew[trial_idx]), dtype=np.float32)  # unused
...
valid_trial_indices.append(trial_idx)                                  # never returned
```
```python
'n_neurons_raw': n_neurons_raw,  # total before filter (already correct)
```

iii. Not discussed in CONVERSION_NOTES. The `float32` choice is stated in the Step 5 mapping table ("Deconvolved fluorescence, float32") without a memory-cost justification, and the resulting 151.6 GB file size is reported in Step 9 without comment.
