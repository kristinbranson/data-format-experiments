# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads from the three dataset folders `beh/`, `spk/` and `retinotopy/`. `beh/Imaging_Exp_info.npy` is opened first and every experiment-type group is walked; for each entry the behavior file `Beh_<exp_type>.npy` is loaded and the session dict is pulled with the key `mname_datexp_blk` plus the optional `_stimtype` suffix (`swap1`/`swap2`). Because the same recording is listed under several experiment types, a *canonical* table keyed by the bare `mname_datexp_blk` is built: the first occurrence is kept, and every later occurrence is verified field-by-field (`ntrials`, `WallName`, `isRew`, `ft_*`, …) against the stored one, raising if they disagree. The list of sessions actually converted is taken from the filenames in `spk/` (89 `*_neural_data.npy` files), and the code asserts every one of them has behavior. Neural traces and retinotopy are then read once per session inside `convert_session`.

ii.
```python
def load_canonical_behavior_sessions():
    exp_info = np.load(EXP_INFO_PATH, allow_pickle=True).item()
    canonical = {}
    behavior_files = {}

    for exp_type, records in exp_info.items():
        behavior_path = os.path.join(BEH_DIR, f"Beh_{exp_type}.npy")
        beh = np.load(behavior_path, allow_pickle=True).item()
        behavior_files[exp_type] = beh

        for record in records:
            full_key = full_key_from_record(record)
            base_key = f"{record['mname']}_{record['datexp']}_{record['blk']}"
            session = beh[full_key]

            if base_key not in canonical:
                canonical[base_key] = {"behavior": session, "exp_type": exp_type,
                                       "full_key": full_key, "record": dict(record)}
                continue

            same, field = behavior_sessions_match(canonical[base_key]["behavior"], session)
            if not same:
                raise ValueError(...)
    return canonical
```
```python
def get_spk_session_keys():
    session_keys = []
    for filename in sorted(os.listdir(SPK_DIR)):
        if not filename.endswith("_neural_data.npy"):
            continue
        session_keys.append(filename[: -len("_neural_data.npy")])
    return session_keys
```
```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
...
retino = np.load(retino_path, allow_pickle=True)   # f"{mouse}_{date_str}_trans.npz"
return build_region_index(retino["iarea"])
```

iii. From the trajectory (step 56/60): "`Imaging_Exp_info.npy` is analysis-oriented, not a simple unique-session list. The same recording key appears in multiple experiment groupings like `unsup_test1` and `unsup_train2_before_learning`, so I need to deduplicate at the raw recording level instead of blindly concatenating every entry", and "several duplicated entries are byte-for-byte the same recording reused in different figure groupings" — hence the explicit equality check before de-duplicating. The `stimtype` suffix was discovered in step 38 ("some `test3` sessions use keys like `..._swap1`, so the converter has to honor the optional `stimtype` suffix").

## 1-b. How are the data split into subjects?

i. The mouse name is parsed out of the session key with a regex (`<mouse>_<YYYY_MM_DD>_<blk>`), i.e. taken from the `mname` field that produced the key. `subjects` is the sorted set of unique mouse names (19 mice) and `subject_idx` is each kept session's index into that list.

ii.
```python
BASE_KEY_RE = re.compile(r"^(?P<mouse>.+?)_(?P<date>\d{4}_\d{2}_\d{2})_(?P<blk>\d+)$")

def build_subject_metadata(session_keys):
    subjects = sorted({parse_base_key(key)[0] for key in session_keys})
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
    subject_idx = np.array([subject_to_idx[parse_base_key(key)[0]] for key in session_keys],
                           dtype=np.int64)
    return subjects, subject_idx
```
```python
kept_subject_idx.append(session_to_subject[session_key])
...
converted["subject_idx"] = np.asarray(kept_subject_idx, dtype=np.int64)
```

iii. No explicit justification in the trajectory; the mouse identity is already carried by the session key/`mname`, so no split has to be inferred. The regex form was chosen so that the same parse serves the mouse name, the date (for day-of-training) and the retinotopy filename.

## 1-c. How are the data split into sessions?

i. A session is one recording = one mouse, one date, one block (`mname_datexp_blk`). The 89 sessions come from the 89 spike files in `spk/`. Recordings listed under several experiment types, and the two `swap1`/`swap2` behavior views of the same recording, are collapsed to a single session; the AI first confirmed the duplicates carry identical behavior. Sessions surviving with fewer than 2 usable trials would be dropped (none were).

ii.
```python
base_key = f"{record['mname']}_{record['datexp']}_{record['blk']}"
...
same, field = behavior_sessions_match(canonical[base_key]["behavior"], session)
```
```python
session_keys = get_spk_session_keys()          # 89 files in /app/data/spk
missing_behavior = [key for key in session_keys if key not in canonical_behavior]
if missing_behavior:
    raise ValueError(f"Missing behavior for sessions: {missing_behavior[:5]}")
```

iii. Step 60/63: "The remaining question is the `swap1`/`swap2` cases, where the same raw recording may have been split into two behavior views; I'm checking those before I decide whether the final dataset should have 89 true recording sessions or the larger analysis-session count." The AI verified the swap views share `WallName`, `SoundPos` and `isRew`, and that the paper reports "89 recordings in 19 mice", so it settled on 89 raw recordings.

## 1-d. How are the data split into trials?

i. Trials are the `ntrials` traversals the behavior declares; each frame is assigned to its trial by `ft_trInd`. Within a trial the AI keeps only the frames that are **inside the texture and during virtual-reality movement**: `ft_CorrSpc & (ft_move > 0)`. Grey-space frames and all stationary frames are discarded, so a trial is a variable-length sequence of running frames starting at (approximately) corridor entry and ending at the end of the 4 m texture. Frames dropped in the middle of a traversal are simply removed, so the kept samples are not always temporally contiguous.

ii.
```python
retained_mask = ft_corr & (ft_move > 0)
...
for trial in range(n_trials):
    frame_idx = np.flatnonzero((ft_trind == trial) & retained_mask)
    if len(frame_idx) == 0:
        continue
    neural = session_matrix[:, frame_idx].astype(np.float16, copy=False)
```

iii. This mirrors the paper and the authors' own code. Methods: "We only considered timepoints during running for analysis, which removed time periods when the task mice stopped to collect water rewards." The authors' notebook/`utils.py` use exactly the same mask (`VRmove = beh['ft_move'][:nfr]>0`; `fr_valid = VRmove & isCorridor  # only use activity inside the texture area plus mouse is running`). Trajectory step 12 ("they interpolate neural traces onto corridor position and use only running frames in the corridor"), step 22, and step 44 item 3: "Match the paper's curation by restricting each trial to running frames (`ft_move > 0`) inside the corridor (`ft_CorrSpc`), which removes stationary reward-collection periods the authors excluded."

## 1-e. How are trials filtered based on quality controls?

i. Only two filters. A trial with zero retained frames (never imaged, or never run through) is skipped; a session left with fewer than two trials is dropped entirely (`return None`). There is **no** trial-length / stopped-animal outlier filter: all 38,110 trials with at least one running corridor frame are kept. The running-frame mask incidentally removes the long stationary stretches, so the longest trial is 178 frames (~56 s of running) instead of 5,607 raw frames.

ii.
```python
frame_idx = np.flatnonzero((ft_trind == trial) & retained_mask)
if len(frame_idx) == 0:
    continue
...
if len(neural_trials) < 2:
    return None
```
```python
if converted_session is None:
    print(f"  Skipping {session_key}: fewer than 2 valid trials after filtering", flush=True)
    continue
```

iii. The two-trial minimum comes straight from the format requirement ("There needs to be at least two trials within each session in order to evaluate the decoder performance"). Zero-frame trials are unusable. The AI's justification for not needing further trial curation is implicit in the running-frame decision (step 44): stationary periods — the reason trials become pathologically long — are already excluded frame-by-frame, so no explicit length cut was added. The AI did check (steps 45, 70–73) the resulting per-trial frame counts and the lag between corridor entry and the first retained frame before committing.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spks` in `spk/<session>_neural_data.npy` — a list of (neurons × frames) deconvolved-trace arrays, one per imaging plane. The visual-area label of each neuron comes from `iarea` in `retinotopy/<mouse>_<date>_trans.npz`; the code checks that the retinotopy length equals the total number of neurons across planes.

ii.
```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
n_frames = spk_planes[0].shape[1]
region_idx_full = load_retinotopy_region_idx(session_key)
if len(region_idx_full) != sum(plane.shape[0] for plane in spk_planes):
    raise ValueError(f"Retinotopy length mismatch for {session_key}")
```
```python
def build_region_index(iarea):
    region_idx = np.full(len(iarea), -1, dtype=np.int16)
    region_idx[iarea == 8] = 0                    # V1
    region_idx[np.isin(iarea, [0, 1, 2, 9])] = 1  # mHV
    region_idx[np.isin(iarea, [5, 6])] = 2        # lHV
    region_idx[np.isin(iarea, [3, 4])] = 3        # aHV
    return region_idx
```

iii. The AI inspected the spk and retinotopy schemas (step 15/18) and copied the area code mapping verbatim from the authors' `utils.neu_area_ID`. Metadata records `"neural_signal": "Suite2p deconvolved fluorescence traces"`, matching "All our analyses were based on deconvolved fluorescence traces".

## 2-b. How is the `neural` data processed?

i. No transformation at all: no dF/F, no smoothing, no normalization, no z-scoring, no rebinning. The selected neurons' columns for the trial's retained frames are sliced out and cast to `float16`. Trials keep their own length; nothing is padded or truncated to a common window.

ii.
```python
def build_selected_session_matrix(spk_planes, selected_global):
    plane_sizes = [plane.shape[0] for plane in spk_planes]
    per_plane = split_global_indices_by_plane(selected_global, plane_sizes)
    selected_parts = []
    for plane, local_idx in zip(spk_planes, per_plane):
        if len(local_idx):
            selected_parts.append(plane[local_idx])
    return np.concatenate(selected_parts, axis=0)
```
```python
neural = session_matrix[:, frame_idx].astype(np.float16, copy=False)
```

iii. The file already contains deconvolved traces, which the paper says all analyses use, so nothing further is applied. `float16` and the per-region neuron cap were the AI's two levers for keeping the output pickle tractable (step 74: "If I keep every neuron from all 89 recordings, the converted pickle may be unrealistically large").

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two stages. (1) **Area filter**: a neuron is kept only if `iarea` maps into V1, mHV, lHV or aHV (the paper's four coarse visual areas); unlabeled / non-visual neurons are dropped. (2) **Per-region cap**: within each of the four regions the AI keeps at most `MAX_NEURONS_PER_REGION = 128` neurons, ranked by their variance over the retained (corridor + running) frames, restricted to neurons whose mean activity in the corridor exceeds their mean activity in the running grey space ("responsive"); if fewer than 128 responsive neurons exist, the region is topped up with the remaining highest-variance neurons. Every session therefore contributes exactly 512 neurons (4 × 128) — about 1% of the ~20k–90k neurons recorded per session (46,128 per session in the expert conversion).

ii.
```python
MAX_NEURONS_PER_REGION = 128
```
```python
corridor_data = plane_valid[:, corridor_mask]
corridor_mean = corridor_data.mean(axis=1)
corridor_var = corridor_data.var(axis=1)
...
if gray_mask.any():
    gray_mean = plane_valid[:, gray_mask].mean(axis=1)
    responsive = corridor_mean > gray_mean
else:
    responsive = np.ones_like(corridor_mean, dtype=bool)
responsive &= np.isfinite(corridor_var)
```
```python
region_selected = sort_take_desc(resp_scores, resp_indices, max_per_region)
if len(region_selected) < max_per_region:
    ...  # top up with remaining highest-variance neurons of that region
```
```python
"neuron_selection": (
    "Restricted to retinotopically assigned visual cortex neurons (V1, mHV, lHV, aHV). "
    f"Within each region, retained up to {max_neurons_per_region} deterministic "
    "high-variance corridor-responsive neurons."),
```

iii. The area restriction follows the paper's code (`utils.neu_area_ID`, and `idx_neu = (arid!=-1) & (arid != 7)` "exclude neurons from outside of visual cortex"). The cap is justified purely on tractability: step 74 — "If I keep every neuron from all 89 recordings, the converted pickle may be unrealistically large, so I need a defensible curation step that stays consistent with the paper rather than producing an unusable artifact"; step 78 — "the full 405 GB neural matrix needs principled neuron curation to produce a usable decoder dataset". The final message states the cap explicitly: "For tractability it keeps retinotopically assigned visual-cortex neurons only and caps each session at `128` neurons per region … so `512` neurons/session."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is on corridor entry (trial start). Each trial simply begins at its own first retained frame inside the corridor and runs to its last, with no padding, no truncation and no common window; `off_start` is 0.0 and `off_end` is `None`. The AI measured empirically that the first retained frame lags true corridor entry by a median of 0.18 s, with 93% of trials within one imaging frame and 99.96% within two frames.

ii.
```python
frame_idx = np.flatnonzero((ft_trind == trial) & retained_mask)
neural = session_matrix[:, frame_idx].astype(np.float16, copy=False)
```
```python
"temporal_alignment_event": "Trial start / corridor entry, with paper-matched retention of running frames inside the corridor only.",
"off_start": 0.0,
"off_end": None,
```

iii. Step 69: "I'm quantifying how far the first kept sample tends to lag corridor entry after the running-only filter. If that lag is small, I can keep the paper's filter and still justify trial-start alignment without extra interpolation or padding." The measurement (step 72) gave median 0.178 s, 93% ≤ 1 frame, so the AI kept native frames aligned at corridor entry. Variable trial lengths were checked against the validator/decoder first (steps 42–43) to confirm ragged trials are accepted.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling: the native two-photon frames are the bins. The AI measured the median inter-frame interval across all 89 sessions (0.31439–0.31537 s, median 0.31469 s) and hard-coded the median as the shared bin size. However, the constant is written into the metadata after multiplying by 24 × 3600, so the saved `time_bin_size` is 27,189,520 ms instead of ~314.7 ms. Note also that, because non-running frames are deleted inside trials, successive samples of a trial are not always one frame apart, so the declared bin size describes the sample width rather than a guaranteed sample spacing.

ii.
```python
TIME_BIN_MS = 1000.0 * 24.0 * 3600.0 * 0.31469352543354034
...
"time_bin_size": float(TIME_BIN_MS),
```
(no resampling anywhere; trials are built by direct frame indexing)

iii. Step 64: "the native imaging frame interval is close to constant but not identical across sessions, and the target format asks for a shared bin size. I'm measuring the frame-rate spread now so I can decide whether to preserve native frames or resample everything onto one common temporal grid." Step 26: "the behavior arrays are already frame-aligned to the imaging data at about `0.315 s` per sample." The measured spread (±0.3%) was judged small enough to keep native frames and declare one bin size — but the value actually stored is off by the 86,400 s/day factor used when converting the MATLAB datenum diffs.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. `SoundTime` (absolute MATLAB datenum of the sound cue, one per trial) and `ft` (the MATLAB datenum timestamp of every imaging frame, truncated to the imaged frames).

ii.
```python
ft_time = np.asarray(behavior["ft"][:n_frames], dtype=np.float64)
sound_time = np.asarray(behavior["SoundTime"], dtype=np.float64)
```

iii. The AI inspected the behavior schema (steps 19/25/46) and found both the frame-number (`SoundFr`) and the absolute-time (`SoundTime`) representations; it used the time fields so no frame→time interpolation is needed.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Per trial, cue time minus the timestamp of each retained frame, converted from days to seconds (× 86,400) and stored as float32. Positive before the cue, negative after, as the name "time *to* sound cue" implies. No clipping is applied, so trials in which the mouse stopped for minutes give values as extreme as −1763 s / +724 s.

ii.
```python
times_to_sound = ((sound_time[trial] - ft_time[frame_idx]) * 24.0 * 3600.0).astype(np.float32)
```

iii. No separate justification recorded beyond the schema inspection; a signed time difference in seconds is the direct reading of the instruction "Time to sound cue, continuous, time-varying".

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated on `ft_time[frame_idx]`, i.e. exactly the same frame indices used to slice the neural matrix for that trial, so it has the same number of columns as the trial's neural array by construction.

ii.
```python
frame_idx = np.flatnonzero((ft_trind == trial) & retained_mask)
neural = session_matrix[:, frame_idx]...
times_to_sound = ((sound_time[trial] - ft_time[frame_idx]) * 24.0 * 3600.0).astype(np.float32)
```

iii. All streams in this dataset are on the imaging-frame grid; the AI confirmed this early (step 26: "the behavior arrays are already frame-aligned to the imaging data") and therefore derives every input/output from the same `frame_idx`.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The recording date embedded in the session key (`mname_YYYY_MM_DD_blk`), taken from the `spk/` filenames — i.e. from `datexp` in `Imaging_Exp_info.npy`.

ii.
```python
def date_from_base_key(base_key):
    _, date_str, _ = parse_base_key(base_key)
    return datetime.strptime(date_str, "%Y_%m_%d").date()
```

iii. The date string is the only field that orders sessions within a mouse; no explicit trajectory note.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse the earliest imaging date is found, and every session gets the number of **calendar days elapsed** since that first session (0 for the first day, up to 92 across the dataset). The scalar is broadcast across all bins of every trial of that session as a float32 input row.

ii.
```python
def compute_training_day_by_session(session_keys):
    by_mouse = defaultdict(list)
    for session_key in session_keys:
        mouse, _, _ = parse_base_key(session_key)
        by_mouse[mouse].append(session_key)

    first_day = {}
    for mouse, keys in by_mouse.items():
        first_day[mouse] = min(date_from_base_key(key) for key in keys)

    training_day = {}
    for session_key in session_keys:
        mouse, _, _ = parse_base_key(session_key)
        training_day[session_key] = float((date_from_base_key(session_key) - first_day[mouse]).days)
    return training_day
```
```python
training_day = np.full(len(frame_idx), day_of_training, dtype=np.float32)
```
```python
"day_of_training_definition": "Calendar days since the first imaging session for that mouse.",
```

iii. The definition is documented in the metadata. The AI examined the per-mouse session tables (step 40, listing each mouse's dates, experiment types and blocks) before choosing elapsed calendar days rather than a session ordinal.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. `Trial_start_time` (absolute datenum of corridor entry, one per trial) and `ft` (frame timestamps).

ii.
```python
trial_start_time = np.asarray(behavior["Trial_start_time"], dtype=np.float64)
ft_time = np.asarray(behavior["ft"][:n_frames], dtype=np.float64)
```

iii. Same as 3-a: the absolute-time fields avoid having to interpolate the fractional `StartFr` frame index onto the time axis.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Frame time minus the trial's start time, × 86,400 to get seconds, float32. Positive after entry, starting at a median of ~0.18 s. No clipping: for trials in which the mouse stopped in the corridor the value can reach 1765 s even though the intervening stationary frames are absent.

ii.
```python
times_since_start = ((ft_time[frame_idx] - trial_start_time[trial]) * 24.0 * 3600.0).astype(np.float32)
```

iii. Step 69/72: the AI explicitly measured this quantity for the first retained frame of every trial in the dataset (median 0.178 s; 93% ≤ one frame) to confirm that the running-only filter does not meaningfully displace the trial-start reference.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed on the same `frame_idx` as the neural slice, so it is aligned frame-for-frame and has identical length.

ii.
```python
times_since_start = ((ft_time[frame_idx] - trial_start_time[trial]) * 24.0 * 3600.0).astype(np.float32)
input_arr = np.vstack([times_to_sound, training_day, times_since_start, reward_available]).astype(np.float32)
```

iii. Every stream is indexed by the same retained-frame array; see 3-c.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. `isRew`, the per-trial flag marking trials run in the rewarded corridor.

ii.
```python
is_rew = np.asarray(behavior["isRew"], dtype=bool)
```

iii. Direct reading of the instruction "Reward availability: 1 if in rewarded corridor, 0 if not".

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast to float 0/1 and broadcast across all bins of the trial (a constant row of the input matrix).

ii.
```python
reward_available = np.full(len(frame_idx), float(is_rew[trial]), dtype=np.float32)
```

iii. None needed; it is a per-trial binary flag, held constant over the trial because the format asks for `(d_input, n_timepoints)` rows. It is 0 for all unsupervised/naive mice, which ran the same corridors with no water available.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. `WallName`, the per-trial name of the wall texture (e.g. `leaf1`, `circle2`, `wood1_swap2`).

ii.
```python
wall_names = np.asarray(behavior["WallName"])
...
stim_category = STIM_CATEGORY_TO_IDX[canonical_stimulus_category(wall_names[trial])]
```

iii. Step 49: "One decision point affects decoder difficulty a lot: whether 'visual stimulus category' means exact wall identity (`leaf1`, `leaf2`, `swap1`, etc.) or the broader corridor category … I'm enumerating the actual wall names used across sessions so I can choose a single consistent label space instead of mixing schemes." The AI enumerated all names appearing in every session (steps 50–54) before choosing.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The wall name is reduced to its base texture by stripping a trailing `_swap1`/`_swap2` and then trailing digits, giving exactly four categories: `circle`, `leaf`, `rock`, `wood`. An unknown name raises. The category index is a per-trial constant broadcast over the trial's bins and stored as int8. (The AI verified that the 15 names occurring in the dataset map onto these four.)

ii.
```python
STIM_CATEGORY_VALUES = ["circle", "leaf", "rock", "wood"]

def canonical_stimulus_category(wall_name):
    category = re.sub(r"_swap[12]$", "", str(wall_name))
    category = re.sub(r"\d+$", "", category)
    if category not in STIM_CATEGORY_TO_IDX:
        raise ValueError(f"Unexpected wall name/category: {wall_name} -> {category}")
    return category
```
```python
stim_out = np.full(len(frame_idx), stim_category, dtype=np.int8)
```

iii. Step 51's enumeration produced exactly `{'circle1': 'circle', …, 'leaf1_swap1': 'leaf', …, 'wood5': 'wood'}` — 15 names → 4 categories — which the AI adopted as the label space, consistent with the instruction's "Visual stimulus category. e.g. circle, leaf, etc." and with the paper's four texture images (circle, leaf, rock, brick/wood).

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. `LickFr` (fractional imaging-frame index of each lick), `LickTrind` (the trial each lick belongs to) and `LickPos` (corridor position of each lick), together with `Texture_Length` (40 dm).

ii.
```python
lick_trial_index = np.asarray(behavior["LickTrind"], dtype=np.float64)
lick_frame_idx = np.asarray(behavior["LickFr"], dtype=np.float64)
lick_pos = np.asarray(behavior["LickPos"], dtype=np.float64)
```

iii. Step 46 inspected `LickFr` alongside `LickTrind`, `LickPos` and `Lick_wallName`; the AI used the trial index to restrict licks to their own trial and the position to restrict them to the texture part of the corridor.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A binary per-bin flag. For each trial, the licks with `LickTrind == trial`, finite `LickFr`/`LickPos` and `0 ≤ LickPos < 40` are kept; each is mapped by binary search to the **nearest retained frame of that trial**, and the bin is set to 1 if the nearest retained frame is within **0.5 frames** of the lick. Licks farther than half a frame from any retained frame (including licks that occurred while the mouse was stationary) contribute nothing. Stored as int8, values `["no_lick", "lick"]`. The resulting positive rate is 3.47% of bins.

ii.
```python
def nearest_retained_licks(retained_frame_idx, lick_frame_idx, lick_pos, texture_length):
    lick_binary = np.zeros(len(retained_frame_idx), dtype=np.int8)
    ...
    valid = np.isfinite(lick_frame_idx) & np.isfinite(lick_pos) & (lick_pos >= 0.0) & (lick_pos < texture_length)
    lick_frame_idx = lick_frame_idx[valid]
    ...
    insert = np.searchsorted(retained, lick_frame_idx)
    candidate_left = np.clip(insert - 1, 0, len(retained) - 1)
    candidate_right = np.clip(insert, 0, len(retained) - 1)
    ...
    lick_binary[np.unique(nearest[nearest_dist <= 0.5])] = 1
    return lick_binary
```
```python
trial_lick_mask = lick_trial_index == trial
licking = nearest_retained_licks(retained_frame_idx=frame_idx,
                                 lick_frame_idx=lick_frame_idx[trial_lick_mask],
                                 lick_pos=lick_pos[trial_lick_mask],
                                 texture_length=texture_length)
```

iii. The instruction asks for a binary time-varying lick signal. Because the frame grid has holes after the running-only filter, the AI chose an explicit nearest-frame assignment with a half-frame tolerance rather than a plain `int(LickFr)` index, so that a lick is never attributed to a frame that is not adjacent in time; the trial index and corridor-position checks keep licks from leaking across trials or from outside the texture.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The flag vector is built directly on the trial's `frame_idx` array (one entry per retained frame), so it is aligned bin-for-bin with the neural matrix of that trial.

ii.
```python
licking = nearest_retained_licks(retained_frame_idx=frame_idx, ...)
output_arr = np.vstack([stim_out, licking, position_out, speed_out]).astype(np.int8, copy=False)
```

iii. `LickFr` indexes the imaging frames, so licks live on the same clock as the neural data; the nearest-frame search simply maps them onto the subset of frames the trial retains.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. `ft_Pos`, the within-corridor position at every imaging frame in decimeters (0–40 across the texture, continuing to 60 through the grey space), truncated to the imaged frames; plus `Texture_Length` (40 dm) for the bin edges.

ii.
```python
ft_pos = np.asarray(behavior["ft_Pos"][:n_frames], dtype=np.float32)
texture_length = float(behavior.get("Texture_Length", 40.0))
```

iii. The AI inspected a single trial frame-by-frame (step 35) printing `ft_Pos`, `ft_PosCum`, `ft_CorrSpc`, `ft_GraySpc` and `ft_WallID` to confirm that `ft_Pos` inside `ft_CorrSpc` spans 0–40 dm of texture.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is clipped to `[0, Texture_Length)` and digitized into 4 equal bins; only the trial's retained frames are taken. Stored as int8 with labels `["0-1m", "1-2m", "2-3m", "3-4m"]`.

ii.
```python
def make_position_bins(position, texture_length):
    pos = np.clip(np.asarray(position, dtype=np.float32), 0.0, float(texture_length) - 1e-6)
    bin_edges = np.linspace(0.0, float(texture_length), 5)
    return np.clip(np.digitize(pos, bin_edges[1:-1], right=False), 0, 3).astype(np.int8)
```

iii. Direct implementation of "Position in corridor discretized into 4 equal-length, 1-m-long spatial bins"; `Texture_Length = 40` decimeters = 4 m, so the four bins are exactly 1 m each.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four fixed, equal-length spatial bins at 0–10, 10–20, 20–30, 30–40 decimeters (i.e. 0–1, 1–2, 2–3, 3–4 m), the same edges for every trial and session. The resulting occupancy is almost uniform (25.0% / 24.9% / 25.0% / 25.2%).

ii.
```python
bin_edges = np.linspace(0.0, float(texture_length), 5)
...
"output_values": [..., ["0-1m", "1-2m", "2-3m", "3-4m"], ...]
```

iii. The instruction prescribes equal 1-m bins, so no data-driven thresholding is used.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is a per-frame variable, indexed with the trial's `frame_idx`, so it is aligned bin-for-bin with the neural array.

ii.
```python
position_out = make_position_bins(ft_pos[frame_idx], texture_length)
```

iii. Same frame grid as the neural data (see 3-c).

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. `ft_RunSpeed`, the mouse's running speed at every imaging frame (also `ft_CorrSpc` and `ft_move` to decide which frames enter the quantile estimate).

ii.
```python
ft_speed = np.asarray(behavior["ft_RunSpeed"][:n_frames], dtype=np.float32)
```

iii. Direct: the instruction asks for running speed discretized into quartile bins.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Three global cut points are computed **once for the whole dataset**, before conversion, as the 25th/50th/75th percentiles of `ft_RunSpeed` pooled over every session's corridor + moving frames. Each frame's speed is then digitized against those three edges. The edges are recorded in the metadata; bins are labelled `speed_q1…q4`. Occupancy comes out at 24.93 / 25.02 / 25.01 / 25.04%.

ii.
```python
def compute_speed_edges(session_keys, canonical_behavior):
    all_speeds = []
    for session_key in session_keys:
        sess = canonical_behavior[session_key]["behavior"]
        mask = sess["ft_CorrSpc"] & (sess["ft_move"] > 0)
        if np.any(mask):
            all_speeds.append(np.asarray(sess["ft_RunSpeed"][mask], dtype=np.float32))
    speed_values = np.concatenate(all_speeds)
    q25, q50, q75 = np.quantile(speed_values, [0.25, 0.5, 0.75])
    return np.array([q25, q50, q75], dtype=np.float32)
```
```python
"running_speed_bin_edges_cm_s": speed_edges.astype(float).tolist(),
```

iii. Step 92: "the speed-class labels will use generic quartile names in `output_values`, while the exact global cut points stay in metadata." Using one global set of edges makes a given speed label mean the same thing in every session, while still satisfying "4 bins, each corresponding to 25% of the data" at the dataset level.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize` against the three global quartile edges, clipped to 0–3, int8.

ii.
```python
def make_speed_bins(speed, speed_edges):
    return np.clip(np.digitize(np.asarray(speed, dtype=np.float32), speed_edges, right=False), 0, 3).astype(np.int8)
```

iii. As above: fixed global quartile thresholds rather than per-session ranks.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is per imaging frame and is indexed with the trial's `frame_idx`, so it is aligned bin-for-bin with the neural array.

ii.
```python
speed_out = make_speed_bins(ft_speed[frame_idx], speed_edges)
output_arr = np.vstack([stim_out, licking, position_out, speed_out]).astype(np.int8, copy=False)
```

iii. Same frame grid as the neural data (see 3-c).

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several guards. (a) The behavior streams run longer than the imaging, so every behavior array is truncated to `n_frames = spk_planes[0].shape[1]`. (b) Licks with non-finite `LickFr`/`LickPos`, or positions outside the texture, are dropped; licks with no retained frame within half a frame are dropped. (c) Trials with no retained frames are skipped; sessions with fewer than two trials are skipped with a message. (d) Hard failures are raised rather than worked around: a retinotopy/neuron count mismatch, a missing behavior entry for a spike file, an unexpected wall name, a session with no selectable neurons, and any disagreement between duplicate copies of the same recording all raise. There is no try/except around session processing, so any one failure would abort the whole run (none occurred). (e) `Texture_Length` falls back to 40.0 if absent.

ii.
```python
n_frames = spk_planes[0].shape[1]
ft_trind = np.asarray(behavior["ft_trInd"][:n_frames])
ft_pos = np.asarray(behavior["ft_Pos"][:n_frames], dtype=np.float32)
...
ft_time = np.asarray(behavior["ft"][:n_frames], dtype=np.float64)
```
```python
valid = np.isfinite(lick_frame_idx) & np.isfinite(lick_pos) & (lick_pos >= 0.0) & (lick_pos < texture_length)
```
```python
if len(region_idx_full) != sum(plane.shape[0] for plane in spk_planes):
    raise ValueError(f"Retinotopy length mismatch for {session_key}")
...
missing_behavior = [key for key in session_keys if key not in canonical_behavior]
if missing_behavior:
    raise ValueError(f"Missing behavior for sessions: {missing_behavior[:5]}")
```

iii. The truncation to imaged frames copies the authors' own convention (`nfr = spk.shape[1]`; `beh['ft_move'][:nfr]`), which the AI read in `utils.py`. The AI also ran a full sweep across all 89 sessions before writing the converter (step 31: "I'm doing a full pass over all 89 imaging sessions now. This is to catch any inconsistencies in frame counts, trial markers, or stimulus sets before I commit to a conversion that later fails validation"), which is why most anomalies are treated as assertions rather than as cases to be patched.

## 12-a. What are the most time-consuming steps of the code?

i. (1) Reading the 405 GB of `spk/*_neural_data.npy` files — 89 pickled multi-gigabyte arrays, unavoidably one full pass. (2) The neuron-selection pass, which for every plane materialises `plane_valid[:, corridor_mask]` — a fresh copy of tens of thousands of neurons × tens of thousands of frames — just to take a mean and a variance. (3) Building the selected matrix and slicing trials. The whole full run took about 8.5 minutes.

ii.
```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
```
```python
corridor_data = plane_valid[:, corridor_mask]
corridor_mean = corridor_data.mean(axis=1)
corridor_var = corridor_data.var(axis=1)
del corridor_data
```

iii. Step 74/84: "the raw files are multi-gigabyte pickled arrays"; step 94: "This will take a while because it has to read the 405 GB raw `spk` directory once, score neurons per session, and write the final pickle." The AI explicitly designed for one pass over the spike data and monitored, but did not revisit the cost of the per-neuron scoring pass.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. (1) The per-trial `np.flatnonzero((ft_trind == trial) & retained_mask)` rescans the whole frame index once per trial (O(ntrials × nframes)); grouping frames by `ft_trInd` in a single pass would do the same work once. (2) The fallback "top-up" in `select_visual_neurons` builds a Python `set` and then a per-element list comprehension over all candidate indices (`np.isin` / a boolean mask would be the vectorized form). (3) `split_global_indices_by_plane` loops over planes building masks over the full index array. (4) The per-plane accumulation of `(indices, scores)` chunks into lists that are later concatenated.

ii.
```python
for trial in range(n_trials):
    frame_idx = np.flatnonzero((ft_trind == trial) & retained_mask)
```
```python
already = set(region_selected.tolist())
fill_mask = np.array([idx not in already for idx in all_indices], dtype=bool)
```

iii. Not discussed in the trajectory; these costs are negligible next to the spike-file I/O, which is presumably why they were left as is.

## 12-c. What processing does the code repeat multiple times?

i. (1) Every `Beh_*.npy` file is loaded in full and every duplicated recording is compared field-by-field across 15 arrays (`behavior_sessions_match`) — a second full traversal of the behavior data purely as a consistency check. (2) All behavior sessions are then walked again in `compute_speed_edges` to pool running speeds. (3) `parse_base_key` re-runs the regex on the same session keys many times (subject list, subject index, training day, retinotopy path). (4) The neural data of a session is touched twice: once to score every neuron, once to extract the selected rows. (5) All behavior sessions are held in memory for the whole run rather than released per file.

ii.
```python
same, field = behavior_sessions_match(canonical[base_key]["behavior"], session)
```
```python
day_by_session = compute_training_day_by_session(session_keys)
speed_edges = compute_speed_edges(session_keys, canonical_behavior)
subjects, subject_idx_full = build_subject_metadata(session_keys)
```

iii. The duplicate verification is a deliberate safety check (step 56/60: the AI wanted proof that the repeated entries really are the same recording before de-duplicating). The global speed pass is needed because the quartile edges must be known before the first session is written.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The largest item: per-neuron corridor mean, grey-space mean and variance are computed for **every** neuron in the session (20k–90k) and then all but 512 of those neurons — and all of the statistics themselves — are thrown away; they never reach the output file. (2) The duplicate-behavior equality check compares 15 arrays per duplicated recording and its result is only used as an assertion. (3) `ft_move` is materialised as a float32 copy of the whole frame axis although only `> 0` is used; `ft_gray` is only used for the responsiveness score that is then discarded. (4) The `LickPos` validity filter is redundant with the frames already being restricted to the texture. (5) `np.unique` followed by `.sort()` on the selected indices (unique already returns sorted output). (6) Bulky metadata (`source_sessions`, `source_behavior_keys`) that the decoder never reads, and a leftover `converted_data_debug.pkl` from the trial run.

ii.
```python
corridor_mean = corridor_data.mean(axis=1)
corridor_var = corridor_data.var(axis=1)
...
gray_mean = plane_valid[:, gray_mask].mean(axis=1)
responsive = corridor_mean > gray_mean
```
```python
selected_global = np.unique(selected_global)
selected_global.sort()
```
```python
"source_sessions": session_keys,
"source_behavior_keys": {key: canonical_behavior[key]["full_key"] for key in session_keys},
```

iii. The scoring work is the unavoidable price of the AI's neuron-cap decision (2-c): to rank neurons it must first measure all of them. The verification and provenance metadata were added deliberately for auditability (steps 56–60), at a modest cost.
