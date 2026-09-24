# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All data come from three subfolders of `/app/data`: `beh/` (behavior, one `.npy` per experiment type plus the master index `Imaging_Exp_info.npy`), `spk/` (deconvolved traces, one `.npy` per session) and `retinotopy/` (one `.npz` per mouse-date, giving each neuron's visual area). `build_session_list()` reads the master index first and reorganises it from *experiment type → list of entries* into a dict keyed by `"{mname}_{datexp}_{blk}"`, recording every experiment type a recording appears under. For each session the AI then (1) finds and loads the behavior via `find_behavior_for_session()`, (2) loads the spike file and concatenates the imaging planes, (3) loads the retinotopy file. The behavior lookup prefers an experiment type **without** a `stimtype` key (plain `session_key`), and falls back to a `stimtype`-suffixed key (`"{session_key}_{stimtype}"`) for the swap ("test3") sessions. Note that `find_behavior_for_session()` re-reads the whole `Beh_*.npy` file (up to 413 MB) from disk on every call, and is called at least twice per session (once in `collect_all_running_speeds`, once in `process_session`).

ii.
```python
def build_session_list():
    """Build list of unique sessions with metadata from Imaging_Exp_info."""
    exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
                       allow_pickle=True).item()
    sessions = {}
    for exp_type, db_list in exp_info.items():
        for ndb in db_list:
            key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
            if key not in sessions:
                sessions[key] = {'mname': ndb['mname'], 'datexp': ndb['datexp'],
                                 'blk': ndb['blk'], 'exp_types': [], 'ndb_list': []}
            sessions[key]['exp_types'].append(exp_type)
            sessions[key]['ndb_list'].append(ndb)
    return sessions, exp_info
```
```python
def load_neural_data(mname, datexp, blk):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    data = np.load(os.path.join(DATA_ROOT, 'spk', fn), allow_pickle=True).item()
    spk = np.concatenate(data['spks'], axis=0)  # (n_neurons, n_frames)
    return spk

def load_retinotopy(mname, datexp):
    fn = f'{mname}_{datexp}_trans.npz'
    return np.load(os.path.join(DATA_ROOT, 'retinotopy', fn), allow_pickle=True)['iarea']
```
```python
    # in find_behavior_for_session(): prefer exp types without stimtype
    for exp_type, ndb in zip(info['exp_types'], info['ndb_list']):
        if ndb.get('stimtype', None) is not None:
            continue
        beh_data = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'),
                           allow_pickle=True).item()
        if session_key in beh_data:
            return beh_data[session_key], exp_type
```

iii. From CONVERSION_NOTES Step 1/2: the AI copied the reference `load_spk()` pattern (`np.concatenate([nspk for nspk in np.load(...)['spks']], 0)`), `load_retino()` and `load_exp_beh()`. It documented that "the same physical session appears in multiple experiment types (e.g. `naive_test1` and `naive_test2`) with different `stim_id` mappings for different analysis purposes" and concluded (trajectory step 61) that "the `stimtype` just controls which stimuli are analyzed. For the decoder, I should load each physical session once and use all the trials regardless of the analysis perspective."

## 1-b. How are the data split into subjects?

i. The mouse identity is the `mname` field of each index entry, which is also the first token of the session key. It is carried on every session result as `result['mname']`. At assembly, `subjects` is the sorted set of mouse names over the sessions that survived, and `subject_idx` is each session's index into that list. Result: 19 subjects, with per-subject session counts identical to the reference (DR10 6, DR15 5, LZ13 4, LZ16 4, TX104 2, TX105 5, TX108 7, TX109 6, TX119 8, TX123 8, TX124 3, TX139 2, TX140 1, TX60 5, TX61 5, TX83 3, TX85 2, TX88 6, VR2 7).

ii.
```python
all_subjects = sorted(set(r['mname'] for r in session_results))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
...
subject_idx.append(subject_to_idx[result['mname']])
...
'subjects': all_subjects,
'subject_idx': np.array(subject_idx, dtype=np.int64),
```

iii. No explicit justification is given beyond the fact that the master index already names the mouse for every recording, so no split has to be inferred. CONVERSION_NOTES Step 2 records "Subjects: 19 mice" as a dataset statistic and Step 9 lists "Subjects 19 / 19 / Yes" as a passed consistency check against the paper's "89 recordings in 19 mice".

## 1-c. How are the data split into sessions?

i. A session is one unique `(mname, datexp, blk)` triple. Because `build_session_list()` keys a dict on `f"{mname}_{datexp}_{blk}"` and only creates the entry the first time it is seen, recordings listed under several experiment types collapse to one session; the extra experiment types are appended to `exp_types`/`ndb_list` and used only as fallbacks for finding the behavior. This yields 89 unique sessions from 142 index entries, matching the paper's "89 recordings".

ii.
```python
key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
if key not in sessions:
    sessions[key] = {...}
sessions[key]['exp_types'].append(exp_type)
sessions[key]['ndb_list'].append(ndb)
```
```python
all_session_keys = sorted(sessions_info.keys())
print(f"  Found {len(all_session_keys)} unique sessions")
```

iii. CONVERSION_NOTES Step 4: "Same physical session appears in multiple experiment types (e.g. naive_test1 and naive_test2) with different stim_id mappings for different analysis purposes. For decoder, use each unique session once with its actual stimulus names from WallName." Step 2 cross-checks this against the 89 spike files on disk and the paper's 89 recordings.

## 1-d. Are the data correctly split into trials?

i. Trials are the `ntrials` trials the behavior declares, and the frames of a trial are selected by a per-frame mask. The AI requires **three** conditions: `ft_trInd == t` (frame belongs to the trial), `ft_CorrSpc` (frame is inside the 4 m textured corridor, excluding the 2 m grey space) and **`ft_move > 0`** (the VR was moving, i.e. the animal was running). `ft_trInd` contains NaN for frames outside any trial, so finiteness is tested explicitly. Behavior arrays are truncated to `min(n_neural_frames, n_beh_frames)`. Trials keep their own length (variable), nothing is padded.

The `ft_move > 0` condition is the one substantive divergence from the human reference, which uses only `ft_trInd == trial & ft_CorrSpc`. It removes roughly 27–32 % of in-corridor frames (checked directly on `TX60_2021_06_07_1`: 14,988 corridor frames → 10,954 running corridor frames) and is why the AI's mean trial length is 22.25 bins against the reference's 32.63. It also means the frames retained inside a trial are **not temporally contiguous**: when a mouse stops mid-corridor, the gap is silently closed, so consecutive stored columns are not always one 315 ms frame apart.

ii.
```python
def extract_trial_frames(beh, trial_idx, n_neural_frames):
    """Get frame indices for a trial where mouse is running in corridor."""
    n_beh_frames = len(beh['ft_trInd'])
    n_frames = min(n_neural_frames, n_beh_frames)

    ft_trInd   = beh['ft_trInd'][:n_frames]
    ft_CorrSpc = beh['ft_CorrSpc'][:n_frames]
    ft_move    = beh['ft_move'][:n_frames]

    valid      = np.isfinite(ft_trInd)
    trial_mask = valid & (ft_trInd.astype(float) == trial_idx)
    corr_mask  = ft_CorrSpc.astype(bool)
    move_mask  = ft_move > 0

    frame_mask = trial_mask & corr_mask & move_mask
    return np.where(frame_mask)[0]
```

iii. CONVERSION_NOTES Step 5, Key Decision 1: "Include only frames where `ft_move > 0` AND `ft_CorrSpc == True` (texture corridor only, while running). Rationale: matches reference code practice; position bins are only defined in texture area; paper says 'only considered timepoints during running'." This is grounded in the reference code, which repeatedly computes `VRmove = beh['ft_move'][:nfr]>0` and `fr_valid = VRmove & isCorridor  # only use activity inside the texture area plus mouse is running (VR moving)`, and in the methods quote recorded in Step 3: "We only considered timepoints during running for analysis." The AI also reasoned (trajectory step 35) that the four 1 m position bins cover exactly the 4 m texture length, so grey-space frames have no valid position label.

## 1-e. How are trials filtered based on quality controls?

i. Two filters only, both structural rather than quality-based: a trial is dropped if it retains fewer than 2 frames after the trial/corridor/running mask, and a whole session is dropped if fewer than 2 trials survive (no session was actually dropped). There is **no** outlier filter on trial duration. 38,110 trials are kept, against the reference's 37,728 (the reference additionally drops the 382 trials longer than the 99th percentile of traversal length).

The consequence is visible in the verification output: `time_since_trial_start` reaches **1763.9 s** (29 minutes) and `time_to_sound_cue` spans **[−722.7, 1762.0] s**, versus the reference's [0, 74.8] s and [−72.2, 73.4] s. These are trials in which the mouse entered the corridor and then stood still for many minutes; the `ft_move > 0` mask deletes the stationary frames from the neural/output streams, so the trial looks short (≤178 bins), but the surviving frame *indices* still straddle the stoppage, so the time inputs carry the full 25×-out-of-range excursion.

ii.
```python
        frame_idx = extract_trial_frames(beh, t, n_neural_frames)

        if len(frame_idx) < 2:
            continue  # Skip trials with too few frames
```
```python
    if len(neural_trials) < 2:
        print(f"  WARNING: Session {session_key} has {len(neural_trials)} valid trials, skipping")
        return None
```

iii. CONVERSION_NOTES Step 3 ("Trial curation rules: No explicit trial filtering in reference code") and Step 10 Check 5 ("Trials with < 2 frames skipped; Sessions with < 2 valid trials skipped (none occurred)"). The AI did notice the anomalous time values during sample validation (trajectory step 77: "−149.5 seconds means the cue is 149.5 seconds AFTER the current frame… A value of 149.5 seconds is way outside that range, so something in my calculation must be off") but resolved the concern by convincing itself the arithmetic was right, and never followed up on *why* a within-corridor traversal could last minutes. No trial-duration quality control was added.

## 2-a. What variables in the raw data is the final `neural` data derived from?

i. From `spks` in `spk/{mname}_{datexp}_{blk}_neural_data.npy`, a list of one (neurons × frames) array per imaging plane, concatenated along the neuron axis into a single (n_neurons, n_frames) matrix. The visual area of every neuron comes from `iarea` in `retinotopy/{mname}_{datexp}_trans.npz`, which indexes the neurons in the same concatenated plane order. This is identical to the reference.

ii.
```python
data = np.load(path, allow_pickle=True).item()
spk = np.concatenate(data['spks'], axis=0)  # (n_neurons, n_frames)
```
```python
iarea = load_retinotopy(mname, datexp)
...
neuron_mask, region_idx = get_neuron_mask_and_regions(iarea)
spk_filtered = spk[neuron_mask]
```

iii. CONVERSION_NOTES Step 1 documents the reference's `load_spk(db, root)` as "Loads neural data, concatenates planes: `np.concatenate([nspk for nspk in np.load(...)['spks']], 0)`" and `load_retino(db, root)` as returning `iarea`; Step 2 notes "Retinotopy file matches total neuron count across all planes", and Step 4 verifies this for TX108 (3 planes × 22,851 = 68,553 = len(iarea)).

## 2-b. How is the `neural` data processed?

i. No transformation at all: the columns belonging to a trial are sliced out of the (already deconvolved) trace matrix and cast to `float32`. No dF/F, no deconvolution, no z-scoring, no smoothing, no rebinning, no padding. (The reference makes the identical decision but casts to `float16`; the AI's `float32` is the main reason `converted_data.pkl` is 144 GB.)

ii.
```python
# Neural data: (n_neurons, n_timepoints)
neural = spk_filtered[:, frame_idx].astype(np.float32)
```

iii. CONVERSION_NOTES Step 3: "Neural data type | Deconvolved fluorescence | 'All our analyses were based on deconvolved fluorescence traces'" and "Deconvolution | Suite2p, decay timescale 0.75s". The metadata field records `'neural_data_type': 'Suite2p deconvolved fluorescence (decay timescale 0.75s)'`. Because the files already hold deconvolved traces, no further processing is applied. The AI did consider `float16` during sample validation (trajectory step 66, worrying about a ~267 GB estimate) but did not adopt it.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron filter is anatomical: a neuron is kept if `iarea != -1` and `iarea != 7`, i.e. it lies in one of the four retinotopically defined visual areas, and it is assigned a region index via `AREA_MAP` (V1 ← 8; mHV ← 0,1,2,9; lHV ← 5,6; aHV ← 3,4). There is no activity-, SNR- or firing-rate-based curation. The resulting counts are **exactly** those of the reference: V1 1,833,035 / mHV 1,108,860 / lHV 495,318 / aHV 668,180 = 4,105,393 neurons, 46,128 mean and 17,363–78,815 range per session. A defensive branch truncates both `spk` and `iarea` to the shorter length with a warning if their neuron counts disagree (this never fired over the 89 sessions).

ii.
```python
AREA_MAP = {8: 'V1', 0: 'mHV', 1: 'mHV', 2: 'mHV', 9: 'mHV',
            5: 'lHV', 6: 'lHV', 3: 'aHV', 4: 'aHV'}
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV']

def get_neuron_mask_and_regions(iarea):
    # Exclude iarea == -1 (unassigned) and iarea == 7
    mask = (iarea != -1) & (iarea != 7)
    region_idx = np.zeros(mask.sum(), dtype=np.int64)
    valid_iarea = iarea[mask]
    for iarea_val, region_name in AREA_MAP.items():
        region_idx[valid_iarea == iarea_val] = BRAIN_REGIONS.index(region_name)
    return mask, region_idx
```
```python
    if len(iarea) != n_neurons_total:
        print(f"  WARNING: Neuron count mismatch for {session_key}: ...")
        min_n = min(len(iarea), n_neurons_total)
        iarea, spk, n_neurons_total = iarea[:min_n], spk[:min_n], min_n
```

iii. CONVERSION_NOTES Step 3, "Neuron curation rules": "Exclude neurons with iarea=-1 (outside mapped region) and iarea=7 (not part of the 4 defined visual areas). Reference code: `(arid!=-1) & (arid != 7)` in `Get_density_map`. No additional quality filtering mentioned; Suite2p cell classification used upstream." Step 1 documents the reference's `neu_area_ID(iarea)` mapping, which the AI reproduces verbatim.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry / trial start. Each trial's neural matrix is `spk_filtered[:, frame_idx]` where `frame_idx` is the ordered list of that trial's in-corridor running frames, i.e. the trial begins at the first imaging frame after corridor entry and ends at the last frame before the grey space. Trials are variable length (11–178 bins); nothing is cut to a common window and nothing is padded. Metadata records `temporal_alignment_event = 'Corridor entry (trial start)'`, `off_start = 0.0`, `off_end = None`. Every input and output stream is indexed with the *same* `frame_idx`, so all streams are aligned by construction. The caveat inherited from 1-d is that `frame_idx` can contain gaps where the animal stopped.

ii.
```python
frame_idx = extract_trial_frames(beh, t, n_neural_frames)
...
neural = spk_filtered[:, frame_idx].astype(np.float32)
```
```python
'temporal_alignment_event': 'Corridor entry (trial start)',
'off_start': 0.0,  # trial starts at corridor entry
'off_end': None,  # variable trial length
```

iii. Trajectory step 35: "For the decoder task, I'm aligning trials to corridor entry (StartFr)… Since the target format allows each trial to have a different number of timepoints, variable-length trials should work fine for the decoder rather than forcing a fixed length." The instructions state alignment is "based on trial start (corridor entry)", and the format only requires a constant bin *size*, not a constant trial length.

## 2-e. How is the `neural` data temporally binned/resampled?

i. Not at all. The imaging frame is the bin. The bin size is computed per session as the median inter-frame interval of `ft` (MATLAB datenums → seconds) and averaged over sessions, giving `time_bin_size = 314.8 ms` (≈3.18 Hz); the reference reports 315.46 ms. No rebinning, no interpolation, no smoothing; in particular the AI deliberately did **not** use the reference's `get_interpPos_spk()` position-interpolation onto 60 spatial bins.

ii.
```python
ft = beh['ft']
frame_dt_days = np.nanmedian(np.diff(ft))
frame_dt_sec = frame_dt_days * 24 * 3600
```
```python
frame_dts = [r['frame_dt_sec'] for r in session_results]
mean_frame_dt = np.mean(frame_dts)
...
'time_bin_size': mean_frame_dt * 1000,  # in ms
'frame_rate_hz': 1.0 / mean_frame_dt,
```

iii. CONVERSION_NOTES Step 5, Key Decision 3: "Time bin size: One calcium imaging frame = ~315ms. This is the native temporal resolution." Trajectory step 31: "I'm leaning toward using raw frame-level data rather than position-interpolated bins, since the task calls for time-varying signals aligned to trial onset rather than spatial binning." Step 3 records the frame rate cross-check: "Frame rate | ~3.17 Hz | Computed from data: mean frame dt = 314.8ms".

## 3-a. What variables in the raw data is `input` *time_to_sound_cue* derived from?

i. From `SoundFr[t]`, the (fractional) imaging-frame index at which the sound cue was delivered on trial `t`, and the frame indices of the trial. The conversion from frames to seconds uses `frame_dt_sec`, the per-session median of `diff(ft)` in seconds, rather than reading the `ft` timestamps of the individual frames (the reference interpolates `SoundFr` onto the `ft` axis instead). Frame sampling is regular enough that the two agree to well under 0.1 s within a trial.

ii.
```python
ft = beh['ft']
frame_dt_sec = np.nanmedian(np.diff(ft)) * 24 * 3600
...
sound_fr = beh['SoundFr'][t]
time_to_cue = (frame_idx - sound_fr) * frame_dt_sec
```

iii. CONVERSION_NOTES Step 2 lists `SoundFr` as "frame index of sound cue per trial (float, interpolated)", and Step 4 notes "The frame indices (StartFr, SoundFr, etc.) are floating-point (interpolated), not exact integer frame indices". Trajectory step 77 confirms the AI checked that "SoundFr values are stored as global frame indices for the whole session, not trial-relative ones, and since frame_idx from the extraction step is also global, subtracting them directly actually does give the correct frame difference within the same trial".

## 3-b. What processing is involved in computing `input` *time_to_sound_cue*?

i. A single subtraction and scaling: `(frame_idx − SoundFr[t]) × frame_dt_sec`, broadcast over the trial's timepoints as row 0 of the input array. This is **time since the cue** — negative before the cue, positive after — i.e. the opposite sign convention from the human reference, which computes `cue − time` so that the value is positive before the cue and counts down to zero. Since a sign flip is an invertible monotone transform of a continuous decoder input, it carries the same information under a different name. The observed range is [−722.7, 1762.0] s, dominated by the unfiltered stopped trials described in 1-e (reference: [−72.2, 73.4] s).

Note an internal inconsistency: CONVERSION_NOTES states the formula as `(SoundFr - current_frame) * frame_dt` in both the Step 5 variable-mapping table and Key Decision 8, which is the *opposite* of what the code computes; only the accompanying prose ("negative before cue, positive after") matches the code.

ii.
```python
        sound_fr = beh['SoundFr'][t]
        time_to_cue = (frame_idx - sound_fr) * frame_dt_sec
...
        inp = np.zeros((4, n_tp), dtype=np.float32)
        inp[0, :] = time_to_cue.astype(np.float32)
```

iii. CONVERSION_NOTES Step 5, Key Decision 8: "Time to sound cue: `(SoundFr - frame_index) * frame_dt` where frame_dt = mean inter-frame interval in seconds. Negative = before cue, positive = after." No justification is offered for the sign convention itself; the AI's reasoning in trajectory step 77 treats "negative means the sound hasn't happened yet" as the intended semantics.

## 3-c. How is `input` *time_to_sound_cue* aligned with the neural data?

i. It is computed directly from the same `frame_idx` array used to slice the neural columns of that trial, so it is aligned frame-for-frame by construction and has exactly the trial's length.

ii.
```python
frame_idx = extract_trial_frames(beh, t, n_neural_frames)
neural = spk_filtered[:, frame_idx].astype(np.float32)
...
time_to_cue = (frame_idx - sound_fr) * frame_dt_sec
```

iii. All streams in this dataset are indexed by imaging frame number, so using one shared `frame_idx` per trial guarantees alignment. The AI's `--show-processing` plots (`processing_TX108_2023_03_25_1.png`, `processing_TX119_2023_12_24_1.png`) overlay neural activity, position, speed, licking and time-to-cue on a common time axis for three example trials per session; CONVERSION_NOTES Step 7 reports "No temporal misalignments visible."

## 4-a. What variables in the raw data is `input` *day_of_training* derived from?

i. From `datexp`, the recording-date string in the master index (equivalently, the date embedded in the session key), grouped by `mname`. Dates are parsed with `datetime.strptime(datexp, '%Y_%m_%d')`. The same source field the reference uses, but consumed as an actual calendar date rather than as a sort key.

ii.
```python
def parse_date(datexp):
    """Parse date string like '2022_07_12' to datetime."""
    return datetime.strptime(datexp, '%Y_%m_%d')
...
mouse_dates = defaultdict(list)
for key, info in sessions_info.items():
    mouse_dates[info['mname']].append((key, parse_date(info['datexp'])))
```

iii. Trajectory step 40: "since naive mice just have a session counter while trained mice have an explicit 'days' field, it's probably cleanest to compute the day number from the actual dates embedded in the session filenames (like TX83_2022_08_17) relative to each mouse's first session." CONVERSION_NOTES Step 5, Key Decision 7: "Day of training: Compute from dates in session names."

## 4-b. What processing is involved in computing `input` *day_of_training*?

i. Per mouse, sessions are sorted by date and the value is the **number of calendar days elapsed since that mouse's first recorded session** (`(dt - first_date).days`), computed over all 89 unique sessions. The value is a per-trial scalar broadcast across every timepoint of the trial as row 1 of the input array. The resulting range is 0–92 days. The reference instead uses the ordinal index of the session within the mouse (0–7). Two sessions of the same mouse on the same date but different blocks receive the same value here, whereas the reference numbers them separately.

ii.
```python
def compute_day_of_training(sessions_info):
    mouse_dates = defaultdict(list)
    for key, info in sessions_info.items():
        mouse_dates[info['mname']].append((key, parse_date(info['datexp'])))

    day_map = {}
    for mname, date_list in mouse_dates.items():
        date_list.sort(key=lambda x: x[1])
        first_date = date_list[0][1]
        for key, dt in date_list:
            day_map[key] = (dt - first_date).days
    return day_map
```
```python
        day_val = float(day)
        ...
        inp[1, :] = day_val  # per-trial, broadcast
```

iii. CONVERSION_NOTES Step 5, Key Decision 7: "For each mouse, day 0 = first session date, subsequent sessions = days since first session." The implicit justification (trajectory step 40) is that the calendar date is the only field common to all experiment types, and that elapsed days is the literal reading of "day of training"; the AI rejected the per-experiment-type session counters because they are not comparable across naive and trained cohorts.

## 5-a. What variables in the raw data is `input` *time_since_trial_start* derived from?

i. From `StartFr[t]`, the (fractional) imaging-frame index of corridor entry on trial `t`, and the trial's frame indices, converted to seconds with the same per-session `frame_dt_sec`.

ii.
```python
        start_fr = beh['StartFr'][t]
        time_since_start = (frame_idx - start_fr) * frame_dt_sec
```

iii. CONVERSION_NOTES Step 2 lists `StartFr` as "frame index of corridor entry per trial (float)"; Step 4 records that these frame indices are fractional. The instructions define the alignment event as corridor entry, so `StartFr` is the natural reference point.

## 5-b. What processing is involved in computing `input` *time_since_trial_start*?

i. `(frame_idx − StartFr[t]) × frame_dt_sec`, stored as row 2 of the input array. Positive after trial start, near zero at the first retained frame — the same sign convention as the reference. The per-session minimum is 0.0 everywhere. The maximum reaches 1763.9 s because of the un-dropped stopped trials (1-e); the reference's maximum is 74.8 s.

ii.
```python
        inp[2, :] = time_since_start.astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 variable-mapping table: "`frame_idx - StartFr` → input[2]: time_since_trial_start, `(current_frame - StartFr) * frame_dt` in seconds, continuous, time-varying". No further rationale is recorded; it is a direct time difference to the alignment event.

## 5-c. How is `input` *time_since_trial_start* aligned with the neural data?

i. Computed from the same per-trial `frame_idx` used for the neural columns, so it is frame-for-frame aligned and the same length.

ii.
```python
frame_idx = extract_trial_frames(beh, t, n_neural_frames)
neural = spk_filtered[:, frame_idx].astype(np.float32)
...
time_since_start = (frame_idx - start_fr) * frame_dt_sec
```

iii. Same reasoning as 3-c: all streams share the trial's frame index array. The `--show-processing` plots use `(frame_idx - StartFr[t]) * frame_dt_sec` as the common x-axis for the neural, position, speed, lick and cue panels, which is the visual check the AI cites for alignment.

## 6-a. What variables in the raw data is `input` *reward_availability* derived from?

i. From `isRew`, the per-trial boolean flag marking trials run in the rewarded corridor. Identical to the reference.

ii.
```python
        reward_avail = float(beh['isRew'][t])
```

iii. CONVERSION_NOTES Step 2 lists `isRew` as "boolean reward availability per trial". Step 5's mapping table: "isRew → input[3]: reward_availability, Direct boolean -> float (0 or 1), per-trial scalar."

## 6-b. What processing is involved in computing `input` *reward_availability*?

i. None beyond a cast to float and broadcast across the trial's timepoints as row 3 of the input array.

ii.
```python
        inp[3, :] = reward_avail  # per-trial, broadcast
```

iii. CONVERSION_NOTES Step 10/trajectory step 109 gives the sanity check: "the reward variable ranges from [0.0, 1.0] for supervised sessions (TX108, TX109, TX60/TX61, VR2) and stays at [0.0, 0.0] for unsupervised/naive sessions, which lines up correctly with which mice actually received rewards." The verification output confirms 51 of 89 sessions have `reward_availability` identically 0.

## 7-a. What variables in the raw data is `output` *visual_stimulus* derived from?

i. From `WallName`, the per-trial string naming the wall texture of the corridor. The AI explicitly did not use `stim_id`/`TrialStim` (which are experiment-type-specific remappings), reasoning that the same physical session appears under several experiment types with different `stim_id` tables.

ii.
```python
    wall_names = beh['WallName']
    ...
        stim_name = wall_names[t]
        ...
        output_trials.append((out, stim_name))
```

iii. CONVERSION_NOTES Step 4: "Same physical session appears in multiple experiment types … with different stim_id mappings for different analysis purposes. For decoder, use each unique session once with its actual stimulus names from WallName." Trajectory step 61 makes the same point about `stimtype` for the swap sessions.

## 7-b. What processing is involved in computing `output` *visual_stimulus*?

i. The raw `WallName` strings are pooled across all sessions, sorted, and each distinct string becomes its own class. This yields **15 classes**: `circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5`. The human reference instead maps these 15 names onto the **4 base textures** (`circle, leaf, rock, wood`) through a hard-coded table. The AI's choice therefore decodes texture *instance* rather than texture *category*, contrary to the instruction's example ("Visual stimulus category. e.g. circle, leaf, etc."). It has several knock-on effects: class frequencies are very uneven (0.008 to 0.264 of the data), most sessions contain only 2–4 of the 15 classes (the per-session output-range lines show ranges like `[8.0, 10.0]` or `[0.0, 3.0]`), and the paper's spatially-shuffled swap *controls* (`leaf1_swap1/2`, `wood1_swap1/2`) become categories separate from their parent texture. Chance level drops to 1/15, so the reported 0.587 validation balanced accuracy is not directly comparable with the reference's 0.664 over 4 classes.

The index is written in two stages: a `-1` placeholder is filled in per trial, then replaced with the global class index in `build_output` after all sessions have been seen.

ii.
```python
        out = np.zeros((4, n_tp), dtype=np.int64)
        out[0, :] = -1  # placeholder, will be filled with stimulus index
```
```python
    all_stimuli = set()
    for result in session_results:
        for out_data, stim_name in result['output']:
            all_stimuli.add(stim_name)
    all_stimuli = sorted(all_stimuli)
    stim_to_idx = {s: i for i, s in enumerate(all_stimuli)}
    ...
        for out_data, stim_name in result['output']:
            out_filled = out_data.copy()
            out_filled[0, :] = stim_to_idx[stim_name]
```
```python
        'output_values': [
            all_stimuli,                      # visual stimulus categories
```

iii. CONVERSION_NOTES Step 5, Key Decision 10: "Stimulus categories: Use WallName directly (e.g. 'circle1', 'leaf1', 'leaf2', etc.). Pool all unique stimuli across sessions." Step 9 records "Stimuli | circle, leaf, etc. | 15 unique stimuli | Yes" as a *passed* consistency check against the paper. Step 11 argues the resulting accuracy is fine: "0.59 val balanced accuracy across 15 classes (chance=0.067). This is excellent: 8.8x above chance." No consideration is documented of pooling the numbered variants into base textures.

## 8-a. What variables in the raw data is `output` *licking* derived from?

i. From `LickFr`, the (fractional) imaging-frame index of every lick in the session, together with `LickTrind`, the trial index of each lick, which the AI uses to restrict licks to the trial being built. The reference uses `LickFr` alone.

ii.
```python
def make_lick_binary(beh, frame_indices, trial_idx):
    lick_fr = beh['LickFr']
    lick_trind = beh['LickTrind']
    if len(lick_fr) == 0:
        return np.zeros(len(frame_indices), dtype=np.float32)
    trial_lick_mask = lick_trind == trial_idx
    trial_lick_fr = lick_fr[trial_lick_mask]
```

iii. CONVERSION_NOTES Step 2 lists `LickFr` ("frame indices of all licks") and `LickTrind` ("trial index per lick"); Step 5, Key Decision 6: "Licking: For each frame in a trial, check if any lick occurred at that frame. Use LickFr and LickTrind to create binary time series."

## 8-b. What processing is involved in computing `output` *licking*?

i. Each lick's fractional frame index is **rounded to the nearest integer frame** (`np.round`), the rounded frames of the current trial are put into a Python `set`, and each retained frame of the trial is labelled 1 if it is in that set and 0 otherwise. The reference instead **truncates** (`astype(int)`), i.e. assigns a lick to the frame interval it falls inside; on a real session roughly half the licks have a fractional part ≥0.5, so the two conventions differ by one bin for about half of all licks. Because non-running frames are dropped (1-d) and licking is concentrated at reward delivery when the animal often stops, some licks fall on discarded frames: the overall lick fraction is 0.035 versus the reference's 0.041. Sessions with no rewards have no licks at all (51 of 89 sessions are identically 0), which the AI verified.

ii.
```python
    lick_int_frames = np.round(trial_lick_fr).astype(int)
    lick_set = set(lick_int_frames)

    binary = np.array([1.0 if fi in lick_set else 0.0 for fi in frame_indices],
                      dtype=np.float32)
    return binary
```
```python
        licking = make_lick_binary(beh, frame_idx, t)
        ...
        out[1, :] = licking.astype(np.int64)
```
```python
            ['not_licking', 'licking'],        # licking
```

iii. CONVERSION_NOTES Step 10, Check 5: "LickFr rounded to nearest integer for binary licking signal." Trajectory step 109 documents the cross-check that licking appears in supervised sessions and is absent in naive/unsupervised ones, and that licking can occur on non-rewarded trials within a supervised session because `isRew` flags the corridor, not the animal's behaviour.

## 8-c. How is `output` *licking* aligned with the neural data?

i. `LickFr` indexes imaging frames, so the binary flag is already on the neural grid; `make_lick_binary` is evaluated on exactly the `frame_indices` used to slice the neural columns, so it has the trial's length and is aligned frame-for-frame.

ii.
```python
frame_idx = extract_trial_frames(beh, t, n_neural_frames)
neural = spk_filtered[:, frame_idx].astype(np.float32)
...
licking = make_lick_binary(beh, frame_idx, t)
```

iii. Same as 3-c/5-c: one shared per-trial frame index array for all streams. The lick trace is one of the rows in the `--show-processing` figure, plotted against the same trial time axis as the neural traces.

## 9-a. What variables in the raw data is `output` *position* derived from?

i. From `ft_Pos`, the per-frame position along the corridor in decimeters (0–40 across the texture, continuing to 60 through the grey space). Because frames are restricted to `ft_CorrSpc`, only the 0–40 range is ever seen.

ii.
```python
        n_beh_frames = len(beh['ft_Pos'])
        n_use = min(n_neural_frames, n_beh_frames)
        positions = beh['ft_Pos'][:n_use]
        trial_positions = positions[frame_idx[frame_idx < n_use]]
```

iii. CONVERSION_NOTES Step 2: "`ft_Pos`: position within corridor per frame (0-60)", "`Corridor_Length`: 60 (= 6m, units are decimeters)", "`Texture_Length`: 40 (= 4m)". Step 4 confirms "Corridor length | Corridor_Length=60 | Position range 0-60 | 4m+2m=6m | 60 decimeters = 6m, consistent".

## 9-b. What processing is involved in computing `output` *position*?

i. Positions are clipped to `[0, 40 − 1e-6]` and binned with `np.digitize` against the edges `[0, 10, 20, 30, 40]` decimeters, minus one to make the index 0-based, then clipped to `[0, 3]`. The result is a time-varying integer label stored as row 2 of the output array. This is arithmetically the same as the reference's `np.clip(ft_Pos // 10, 0, 3)`.

ii.
```python
POS_BIN_EDGES = np.array([0, 10, 20, 30, 40])  # bin edges in decimeters

def discretize_position(positions, bin_edges=POS_BIN_EDGES):
    """Discretize continuous position (0-40 dm) into bins.
    Bins: [0,10), [10,20), [20,30), [30,40]"""
    pos_clipped = np.clip(positions, 0, TEXTURE_LENGTH - 1e-6)
    bins = np.digitize(pos_clipped, bin_edges) - 1  # 0-indexed
    bins = np.clip(bins, 0, N_POS_BINS - 1)
    return bins.astype(np.int64)
```
```python
        out[2, :] = pos_bins
```

iii. CONVERSION_NOTES Step 5, Key Decision 4: "Position discretization: ft_Pos in [0, 40) -> 4 bins of 10 units each: [0,10), [10,20), [20,30), [30,40). Each bin = 1m." Trajectory step 35: "The four spatial bins at 1m each cover exactly 4m, which matches the texture area length, not the full 6m corridor including grey space -- so position discretization applies only to that texture region." This is exactly what the instructions ask for ("discretized into 4 equal-length, 1-m-long spatial bins").

## 9-c. How is `output` *position* thresholded into categories?

i. Fixed physical thresholds at 0, 1, 2, 3 and 4 m (0, 10, 20, 30, 40 decimeters), independent of the data distribution — equal-length bins, as instructed. The realised distribution is near-uniform because the VR advances at a constant 60 cm/s when the animal runs: 0.250 / 0.249 / 0.250 / 0.252 overall, and within 0.215–0.279 for every one of the 89 sessions. The reference's distribution is 0.254 / 0.242 / 0.247 / 0.257.

ii.
```python
N_POS_BINS = 4        # 4 bins of 1m each (10 decimeters)
POS_BIN_EDGES = np.array([0, 10, 20, 30, 40])  # bin edges in decimeters
...
            ['0-1m', '1-2m', '2-3m', '3-4m'], # position bins
```

iii. Directly from the Decoder Task specification. CONVERSION_NOTES Step 7 and Step 9 verify "Position bins ~25% each (well-balanced)" and Step 10 Check 2 records "Position bins verified to cover [0-10, 10-20, 20-30, 30-40] dm = [0-1, 1-2, 2-3, 3-4] m".

## 9-d. How is `output` *position* aligned with the neural data?

i. `ft_Pos` has one value per imaging frame, and it is indexed with the same trial `frame_idx` used for the neural columns, so it is aligned frame-for-frame. A defensive branch pads with the last valid value if fewer positions than frames were retrieved; this is unreachable in practice because `frame_idx` is constructed from `min(n_neural_frames, n_beh_frames)`.

ii.
```python
        trial_positions = positions[frame_idx[frame_idx < n_use]]
        if len(trial_positions) < len(frame_idx):
            pad_len = len(frame_idx) - len(trial_positions)
            trial_positions = np.concatenate([
                trial_positions,
                np.full(pad_len, trial_positions[-1] if len(trial_positions) > 0 else 0)])
        pos_bins = discretize_position(trial_positions)
```

iii. Same reasoning as 3-c. The position panel of the `--show-processing` figure plots `ft_Pos` on the trial time axis with the bin edges drawn as horizontal lines, which the AI cites in Step 7 as the visual proof that "Position bins correctly cover 0-4m" and that there are no misalignments.

## 10-a. What variables in the raw data is `output` *running_speed* derived from?

i. From `ft_RunSpeed`, the animal's running speed at each imaging frame. Identical source to the reference.

ii.
```python
        n_speed_frames = len(beh['ft_RunSpeed'])
        n_use_speed = min(n_neural_frames, n_speed_frames)
        run_speeds = beh['ft_RunSpeed'][:n_use_speed]
        trial_speeds = run_speeds[frame_idx[frame_idx < n_use_speed]]
```

iii. CONVERSION_NOTES Step 2 lists "`ft_RunSpeed`: running speed per frame"; Step 5's table maps it to output[3] with "Discretize into 4 quartile bins (25% each)".

## 10-b. What processing is involved in computing `output` *running_speed*?

i. A **single global** set of quartile thresholds is computed once, before any session is processed, by `collect_all_running_speeds()`: for each of the 89 sessions it loads the behavior, masks to `ft_CorrSpc & (ft_move > 0)`, subsamples to at most 10,000 frames per session with a fixed seed, pools everything, and takes `np.percentile(all_speeds, [25, 50, 75])`, giving edges `[12.20, 25.01, 40.53]`. Each trial's speeds are then assigned with `np.digitize`. The reference instead computes a **per-session rank-based** quartile split over exactly the frames it keeps.

Two consequences of the pooled approach: subsampling caps each session at 10,000 frames, so long and short sessions contribute almost equally rather than in proportion to their data; and the thresholds are estimated only from running frames (zero-speed frames are excluded by the `ft_move > 0` mask), which conveniently avoids the tie-at-zero problem that motivated the reference's rank-based split.

ii.
```python
def collect_all_running_speeds(sessions_info, exp_info, session_keys):
    all_speeds = []
    for session_key in session_keys:
        beh, _ = find_behavior_for_session(session_key, sessions_info, exp_info)
        ...
        mask = ft_CorrSpc & (ft_move > 0)
        speeds = ft_RunSpeed[mask]
        if len(speeds) > 10000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(speeds), 10000, replace=False)
            speeds = speeds[idx]
        all_speeds.append(speeds)
    all_speeds = np.concatenate(all_speeds)
    quartiles = np.percentile(all_speeds, [25, 50, 75])
    return quartiles
```

iii. CONVERSION_NOTES Step 5, Key Decision 5: "Running speed discretization: Compute quartiles across ALL sessions' running speed data (ft_RunSpeed where ft_move > 0 and ft_CorrSpc), then assign bins 0-3." Trajectory step 61: "For running speed quartiles, I'll collect all speeds in a single pass, compute the quartile bin edges at the end, then discretize using those edges." The subsampling is justified in the code comment as being "for efficiency"; the whole collection pass takes 6.6 s.

## 10-c. How is `output` *running_speed* thresholded into categories?

i. `np.digitize(speeds, [12.20, 25.01, 40.53])` → bins 0–3, with labels generated from the edges (`<12.2`, `12.2-25.0`, `25.0-40.5`, `>40.5`). Pooled over the whole dataset the split is essentially exact: 0.246 / 0.248 / 0.252 / 0.255, which satisfies the instruction "4 bins, each corresponding to 25% of the data".

Because the thresholds are global, individual sessions are far from balanced: session 0 (`DR10_2022_07_12_1`) has 0.955 / 0.043 / 0.002 / 0.000, session 34 has 0.813 / 0.182 / 0.005 / 0.000, and many sessions have essentially no frames in one of the four bins. The reference's per-session split gives 0.250 in every bin both globally and within each session. A locally unused `bin_edges` array is built inside `discretize_speed` and never used.

ii.
```python
def discretize_speed(speeds, quartile_edges):
    """Discretize running speed into 4 quartile bins."""
    bin_edges = np.array([-np.inf, quartile_edges[0], quartile_edges[1], quartile_edges[2], np.inf])
    bins = np.digitize(speeds, quartile_edges)  # 0,1,2,3
    return bins.astype(np.int64)
```
```python
    speed_labels = [
        f'<{speed_quartiles[0]:.1f}',
        f'{speed_quartiles[0]:.1f}-{speed_quartiles[1]:.1f}',
        f'{speed_quartiles[1]:.1f}-{speed_quartiles[2]:.1f}',
        f'>{speed_quartiles[2]:.1f}',
    ]
```

iii. CONVERSION_NOTES Step 5, Key Decision 5, and the stated aim of "consistent binning" across sessions (Step 6: "Computes running speed quartiles across all sessions for consistent binning"). Step 9 validates only the pooled distribution — "Speed bins | ~25% each | Yes (quartiles)" — and the per-session imbalance is never examined.

## 10-d. How is `output` *running_speed* aligned with the neural data?

i. `ft_RunSpeed` has one value per imaging frame, indexed with the trial's `frame_idx`, the same array used to slice the neural columns; it therefore has the trial's length and is aligned frame-for-frame. The same unreachable last-value padding branch as for position is present.

ii.
```python
        trial_speeds = run_speeds[frame_idx[frame_idx < n_use_speed]]
        if len(trial_speeds) < len(frame_idx):
            pad_len = len(frame_idx) - len(trial_speeds)
            trial_speeds = np.concatenate([trial_speeds,
                np.full(pad_len, trial_speeds[-1] if len(trial_speeds) > 0 else 0)])
        speed_bins = discretize_speed(trial_speeds, speed_quartiles)
        ...
        out[3, :] = speed_bins
```

iii. Same reasoning as 3-c. The `--show-processing` figure plots raw `ft_RunSpeed` for three trials with the three quartile thresholds overlaid as horizontal lines, on the same time axis as the neural data, which the AI cites in Step 7 ("Speed quartile lines visible", "No temporal misalignments visible").

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several defensive measures, most of them matching the reference's single real issue (behavior running past the imaging):
- **Behavior longer than imaging**: every frame-level stream is truncated to `min(n_neural_frames, n_beh_frames)` before use (`extract_trial_frames`, and again for `ft_Pos` and `ft_RunSpeed`). The AI verified the off-by-one on TX108 (23,193 spike frames vs 23,194 behavior frames).
- **NaN trial labels**: `ft_trInd` is NaN for frames outside any trial; finiteness is tested explicitly before the equality comparison.
- **Neuron-count mismatch** between `spks` and `iarea`: warn and truncate both to the shorter length (never triggered).
- **Missing behavior for a session**: warn and skip the session (never triggered).
- **Degenerate trials/sessions**: trials with <2 retained frames are skipped; sessions with <2 surviving trials are skipped.
- **Short position/speed vectors**: padded with the last valid value (unreachable in practice).
- Licks are only ever looked up on frames that exist, so licks recorded past the last imaged frame are implicitly dropped.

ii.
```python
    n_beh_frames = len(beh['ft_trInd'])
    n_frames = min(n_neural_frames, n_beh_frames)
    ...
    valid = np.isfinite(ft_trInd)
    trial_mask = valid & (ft_trInd.astype(float) == trial_idx)
```
```python
    if len(iarea) != n_neurons_total:
        print(f"  WARNING: Neuron count mismatch for {session_key}: "
              f"spk={n_neurons_total}, retino={len(iarea)}")
        min_n = min(len(iarea), n_neurons_total)
        iarea = iarea[:min_n]; spk = spk[:min_n]; n_neurons_total = min_n
```
```python
    if beh is None:
        print(f"  WARNING: No behavior found for {session_key}, skipping")
        return None
```

iii. CONVERSION_NOTES Step 4 documents the frame-count discrepancy ("TX108: 23193 frames, beh ft has 23194 frames … Off by 1 frame is expected; code clips to min: `spk.shape[1]` = nfr") and Step 10 Check 5 lists the edge cases handled. The full conversion log contains zero warnings, confirming none of the defensive branches fired — consistent with the reference's assessment that "this is a very clean dataset".

## 12-a. What are the most time-consuming steps of the code?

i. Reading and concatenating the spike files, which total 405 GB. The per-session log shows 11–20 s per session almost entirely tracking spike-file size, for 22.2 minutes over 89 sessions; by comparison the one-off speed-quartile pass over all 89 behavior files takes 6.6 s. Writing the 144 GB output pickle is the second cost. The AI prints per-session timing and estimated the full run from the sample.

ii.
```python
    for i, session_key in enumerate(session_keys):
        t_sess = time.time()
        print(f"\n[{i+1}/{len(session_keys)}] Processing {session_key}...")
        result = process_session(...)
        ...
            print(f"  {result['n_neurons']} neurons, {n_trials} trials, "
                  f"{total_frames} total frames, {time.time()-t_sess:.1f}s")
```
```python
    data = np.load(path, allow_pickle=True).item()
    spk = np.concatenate(data['spks'], axis=0)  # (n_neurons, n_frames)
```

iii. CONVERSION_NOTES Step 7: "Processing: ~5 s/GB of spike data; Total spike data: 434 GB; Estimated full conversion: ~36 minutes." The actual run was 22.2 minutes. Step 9 reports this as "well within 15-minute guideline", which is incorrect arithmetic (22.2 > 15) though the conversion was not further optimised.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Three, none of which the AI documented:
- `make_lick_binary` builds the lick flag with a **per-frame Python list comprehension** testing set membership. This is fully vectorizable (`np.isin`, or the reference's scatter into a session-length flag array).
- `extract_trial_frames` is called once per trial and re-scans the *entire session's* `ft_trInd`/`ft_CorrSpc`/`ft_move` arrays each time, making trial extraction O(n_trials × n_frames) — ~450 × 20,000 per session. A single pass grouping frames by trial would do. (The reference has the same structure and flags it as negligible against the I/O cost.)
- Position and speed are re-sliced and re-discretized per trial rather than once per session; the reference discretizes the whole session's `ft_Pos`/`ft_RunSpeed` once and then slices.

ii.
```python
    binary = np.array([1.0 if fi in lick_set else 0.0 for fi in frame_indices],
                      dtype=np.float32)
```
```python
    for t in range(ntrials):
        frame_idx = extract_trial_frames(beh, t, n_neural_frames)   # rescans all frames
        ...
        pos_bins = discretize_position(trial_positions)
        speed_bins = discretize_speed(trial_speeds, speed_quartiles)
```

iii. The AI's CONVERSION_NOTES contain no analysis of vectorizable loops. Step 6's "Code inefficiencies identified / Code speedups added" slots of the template were replaced with a bullet list of features rather than an efficiency discussion, and Step 7 only tabulates the speed-quartile pass and the per-GB processing rate. In fairness these loops are small against the 405 GB of spike I/O, which is the same conclusion the reference reaches about its own per-trial scan.

## 12-c. What processing does the code repeat multiple times?

i. **Behavior files are re-read from disk for every session.** `find_behavior_for_session()` calls `np.load` on a whole `Beh_*.npy` file (74–413 MB) each time it is invoked, and it is invoked at least twice per session — once inside `collect_all_running_speeds()` and once inside `process_session()` — plus once more per session in `show_processing_plots()`. When the first candidate experiment type does not contain the key, additional whole files are loaded and discarded. Over 89 sessions this is ~180+ full-file reads of data that could be read 23 times (once per experiment-type file). The reference explicitly avoids this by grouping sessions by behavior file and reading each once (`groups.setdefault(record[2], []).append(record)`).

Secondary repetitions: `show_processing_plots()` re-loads the spike file, re-loads the retinotopy, re-applies the neuron mask and re-runs `extract_trial_frames`/`make_lick_binary` for the sessions it plots, duplicating everything `process_session` just did; and the per-trial recomputation of position/speed discretization noted in 12-b.

ii.
```python
def find_behavior_for_session(session_key, sessions_info, exp_info):
    for exp_type, ndb in zip(info['exp_types'], info['ndb_list']):
        ...
        beh_file = os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy')
        if os.path.exists(beh_file):
            beh_data = np.load(beh_file, allow_pickle=True).item()   # whole file, every call
```
```python
    for session_key in session_keys:                       # in collect_all_running_speeds
        beh, _ = find_behavior_for_session(session_key, sessions_info, exp_info)
```
```python
    beh, exp_type = find_behavior_for_session(session_key, sessions_info, exp_info)  # again
```
```python
def show_processing_plots(...):
    beh, _ = find_behavior_for_session(...)      # a third time
    spk = load_neural_data(mname, datexp, blk)   # spike file re-read
    iarea = load_retinotopy(mname, datexp)
```

iii. Not identified or discussed anywhere in CONVERSION_NOTES. Step 6 asserts the script was written with efficiency in mind and Step 7 estimates ~36 minutes for the full run; the redundant behavior I/O was never measured separately, and it is masked by the much larger spike-file cost.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Mostly storage-format waste plus some dead code:
- **`float32` neural data.** The traces are stored at `float32`, producing a 144 GB pickle. The reference stores `float16` (deconvolved traces have nowhere near 24 bits of precision), which would roughly halve the file. The AI explicitly considered `float16` (trajectory step 66) and did not adopt it.
- **`int64` outputs.** All four outputs take values in 0–14 but are stored as `int64`; the reference uses `int8`, an 8× reduction on that stream.
- **`-1` stimulus placeholder plus a full array copy.** Every trial's output array is written with `out[0,:] = -1`, then `build_output` makes a full `out_data.copy()` of every trial's output array purely to overwrite row 0 with the real class index — an avoidable second pass and allocation over all 38,110 trials.
- **Literal dead code** in `build_output` (a loop over `... if False else []`) and an unused `bin_edges` array in `discretize_speed`.
- `--show-processing` recomputes an entire session's conversion just to draw the figure.
- Per-trial constants (`day_of_training`, `reward_availability`, `visual_stimulus`) are tiled across every timepoint rather than stored once — but the target format asks for this, and the reference does the same.

ii.
```python
        neural = spk_filtered[:, frame_idx].astype(np.float32)
        ...
        out = np.zeros((4, n_tp), dtype=np.int64)
        out[0, :] = -1  # placeholder, will be filled with stimulus index
```
```python
        for _, stim_name in [r for r in [(out[1]) for out in result['output']]
                             if isinstance(r, str)] if False else []:
            pass
```
```python
        for out_data, stim_name in result['output']:
            out_filled = out_data.copy()
            out_filled[0, :] = stim_to_idx[stim_name]
            session_outputs.append(out_filled)
```
```python
def discretize_speed(speeds, quartile_edges):
    bin_edges = np.array([-np.inf, quartile_edges[0], ..., np.inf])   # never used
    bins = np.digitize(speeds, quartile_edges)
```

iii. Not discussed in CONVERSION_NOTES; Step 13 simply records "`/app/converted_data.pkl` | Full dataset (89 sessions) | ~142 GB" without commenting on whether that size is necessary. The instructions' "Key Considerations" section did ask for appropriate data types ("float32 vs. float64") and memory awareness; the AI's trajectory shows it worried about total size at the sample stage and then proceeded with `float32` anyway.
