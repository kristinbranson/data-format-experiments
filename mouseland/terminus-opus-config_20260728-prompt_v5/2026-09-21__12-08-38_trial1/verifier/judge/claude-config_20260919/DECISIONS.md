# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads everything from the three subfolders of `/app/data` (`beh/`, `spk/`, `retinotopy/`), mirroring the reference repo's `utils.load_spk` / `load_retino` / `load_exp_beh`. `beh/Imaging_Exp_info.npy` is the master index; it is a dict keyed by experiment type, each value a list of recording dicts (`mname`, `datexp`, `blk`, `stimtype`, ...). The AI flattens this index into a list of unique `(exp_type, db, key)` records. Behavior comes from `beh/Beh_<exp_type>.npy` (a dict of sessions), keyed by `mname_datexp_blk` plus `_stimtype` for swap sessions. Neural data comes from `spk/<mname>_<datexp>_<blk>_neural_data.npy` (a per-plane list under `spks`, concatenated). Areas come from `retinotopy/<mname>_<datexp>_trans.npz`. Behavior files are cached per experiment type in `beh_cache`, but the cache is built and thrown away three separate times (unique-stimulus pass, speed-quartile pass, main conversion pass), so every behavior file is read from disk three times.

ii.
```python
def load_exp_info():
    info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()
    return info

def load_spk(db):
    """Load neural data for a session. Matches reference code utils.load_spk."""
    fn = '%s_%s_%s_neural_data.npy' % (db['mname'], db['datexp'], db['blk'])
    spk_path = os.path.join(DATA_ROOT, 'spk', fn)
    spk = np.concatenate([nspk for nspk in
                          np.load(spk_path, allow_pickle=True).item()['spks']], 0)
    return spk

def load_retino(db):
    dtrans = np.load(os.path.join(DATA_ROOT, 'retinotopy',
                     '%s_%s_trans.npz' % (db['mname'], db['datexp'])),
                     allow_pickle=True)
    return dtrans

def load_beh(exp_type):
    beh = np.load(os.path.join(DATA_ROOT, 'beh', 'Beh_' + exp_type + '.npy'),
                  allow_pickle=True).item()
    return beh

def get_session_key(db):
    if 'stimtype' in db:
        return '%s_%s_%s_%s' % (db['mname'], db['datexp'], db['blk'], db['stimtype'])
    else:
        return '%s_%s_%s' % (db['mname'], db['datexp'], db['blk'])
```

iii. CONVERSION_NOTES.md Step 1 tabulates `load_spk`, `load_retino`, `load_exp_beh`, `neu_area_ID` as the reference loading functions and Step 10 Check 3 states "Data loading: Uses same load_spk pattern (concatenate spks planes) ✓" and "Behavior keys: Same session key format (mname_datexp_blk) ✓". The AI's rationale is explicitly to reuse the reference loading path.

## 1-b. How are the data split into subjects (mice)?

i. The subject is `db['mname']` from the index entry. Subjects are accumulated into an insertion-ordered dict as sessions are processed, and `subject_idx` holds each kept session's index into that list. Result: 19 subjects over 89 sessions, matching the paper. (Unlike the reference, the subject list is in first-encountered order rather than sorted, and sessions are emitted in index order rather than grouped by mouse — neither affects correctness since `subject_idx` is consistent.)

ii.
```python
mname = db['mname']
if mname not in subjects_set:
    subjects_set[mname] = len(subjects_set)
subject_idx_list.append(subjects_set[mname])
...
subjects = list(subjects_set.keys())
subject_idx = np.array(subject_idx_list, dtype=np.int64)
```

iii. CONVERSION_NOTES.md Step 2/Step 9: "Subjects | 19" and "19 subjects: Matches paper ✓". The mouse name is given directly by the index, so no derivation is needed.

## 1-c. How are the data split into sessions?

i. A session is one mouse × one date × one block, keyed `mname_datexp_blk`. Because the master index lists the same recording under more than one experiment type, the AI de-duplicates with a `seen_keys` set and keeps the first occurrence, giving 89 unique sessions — exactly the paper's 89 recordings. The behavior key adds `_stimtype` for swap sessions. Sessions with fewer than 2 valid trials would be dropped, but none were.

ii.
```python
all_sessions = []
seen_keys = set()
for exp_type, sessions in info.items():
    for db in sessions:
        key = f"{db['mname']}_{db['datexp']}_{db['blk']}"
        if key not in seen_keys:
            seen_keys.add(key)
            all_sessions.append((exp_type, db, key))
print(f"Total unique sessions: {len(all_sessions)}")
...
if len(neural_trials) < 2:
    print(f"  WARNING: Session has fewer than 2 valid trials, skipping")
    continue
```

iii. CONVERSION_NOTES.md Step 4 consistency table: "Sessions | 89 unique | 89 files | 89 recordings | Consistent". The AI reasoned that the spike directory has 89 files and the paper reports 89 recordings, so first-occurrence de-duplication over the index is the right split.

## 1-d. How are the data split into trials?

i. Trials are the `ntrials` declared by the behavior file. A trial's frame window is `[floor(StartFr[tr]), floor(EndFr[tr]))` — corridor entry to corridor exit — clipped into `[0, nfr]`, where `nfr = spk.shape[1]`. This window is **contiguous and includes the 2 m grey space** after the 4 m textured corridor (`GrayFr` is never used, and `ft_CorrSpc` / `ft_trInd`, which the reference uses to keep only textured-corridor frames, are never read). Trials therefore have variable length (median ≈52 frames in a spot-checked session vs ≈31 textured-only frames), and ~27% of the retained frames are grey-space frames at 4–6 m. Trials are stored at their native length; no padding or fixed window.

ii.
```python
start_fr = np.array(beh['StartFr'])
end_fr = np.array(beh['EndFr'])
start_fr_int = np.floor(start_fr).astype(int)
end_fr_int = np.floor(end_fr).astype(int)
start_fr_int = np.clip(start_fr_int, 0, nfr - 1)
end_fr_int = np.clip(end_fr_int, 0, nfr)
...
for tr in range(ntrials):
    s_fr = start_fr_int[tr]
    e_fr = end_fr_int[tr]
    n_trial_frames = e_fr - s_fr
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 5: "**Trial boundaries**: StartFr to EndFr (including gray space)." and Decision 2: "**All frames included**: Both running and non-running frames included for temporal continuity." The trajectory (steps 43–44) shows the AI explicitly weighing the reference's running-frame filter and rejecting it because dropping frames "breaks temporal continuity" and would make `time_since_trial_start` / `time_to_sound_cue` non-uniform.

## 1-e. How are trials filtered based on quality controls?

i. The only trial filter is `n_trial_frames < 2` (trials whose imaged window has fewer than 2 frames are skipped and counted). No outlier-length filter is applied: the longest retained trial is 5621 frames ≈ 30 minutes of a stationary mouse, and 38,110 trials are kept — i.e. every trial the data declares. Sessions with <2 surviving trials would be dropped (none were).

ii.
```python
if n_trial_frames < 2:
    skipped_trials += 1
    continue
...
if skipped_trials > 0:
    print(f"    Skipped {skipped_trials} trials with < 2 frames")
```

iii. CONVERSION_NOTES.md Step 10 Check 5: "Very long trials (up to 5621 frames): Included, not filtered ✓". The trajectory (step 174) records that the AI *noticed* the problem — "The very long trials (max 5621 timepoints = ~30 minutes) are concerning... I should consider whether to cap trial length or filter these extreme trials" — but then proceeded without acting on it, reasoning only that "the data format is valid".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spks` from `spk/<mname>_<datexp>_<blk>_neural_data.npy`, a per-imaging-plane list of (neurons × frames) arrays concatenated along axis 0. Brain-area labels come from `iarea` in `retinotopy/<mname>_<datexp>_trans.npz`. Identical sources to the reference.

ii.
```python
spk = np.concatenate([nspk for nspk in
                      np.load(spk_path, allow_pickle=True).item()['spks']], 0)
...
iarea = retino['iarea']
brain_region_idx = get_brain_region_idx(iarea, brain_regions)
```

iii. CONVERSION_NOTES.md Step 3: "All analyses based on deconvolved fluorescence traces"; Step 10 Check 3: "Neural data: Uses deconvolved traces (same as reference) ✓". No dF/F or deconvolution is needed because the files already hold Suite2p-deconvolved traces.

## 2-b. How is the `neural` data processed?

i. No signal processing at all: the columns `s_fr:e_fr` of `spk` are sliced out and cast to `float16`. Before that, the neuron axis is **randomly subsampled to 5000 neurons per session** with a fixed seed (`RandomState(42)`), down from the native 20,547–89,577. If the spike and retinotopy neuron counts disagree, both are truncated to the minimum.

ii.
```python
MAX_NEURONS = 5000  # Subsample neurons for manageable file size; decoder uses PCA to 100 components
...
n_total = spk.shape[0]
if n_total > MAX_NEURONS:
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    neuron_idx = np.sort(rng.choice(n_total, MAX_NEURONS, replace=False))
    spk = spk[neuron_idx]
    iarea_arr = iarea_arr[neuron_idx]
    print(f"  Subsampled neurons: {n_total} -> {MAX_NEURONS}")
...
neural = spk[:, s_fr:e_fr].astype(np.float16)
```

iii. CONVERSION_NOTES.md Step 5 Key Decisions 1 and 6: "**Neuron subsampling**: 5000 neurons per session (from 20K-90K) to keep file size manageable. Decoder uses PCA to 100 components." and "**Float16 for neural**: Reduces file size by 2x with negligible precision loss." Trajectory steps 44/158/159 show the driver was purely practical: an all-neuron pickle was ~216 GB and took >14 min merely to load, so the AI killed the run and subsampled, arguing "5000 neurons is more than enough for good performance" since the decoder PCAs to 100 components.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No area-based filter is applied.** `neu_area_ID` maps `iarea` 8→V1, {0,1,2,9}→mHV, {5,6}→lHV, {3,4}→aHV, and everything else to the string `'unassigned'`, which is then added as a fifth entry of `brain_regions`. All neurons — including the 53,664 unassigned ones (12% of the 445,000 retained) — are kept. The only curation of the neuron axis is the random subsample to 5000 described in 2-b. No other quality filter (SNR, firing rate, Suite2p classifier) is applied.

ii.
```python
def neu_area_ID(iarea):
    area_map = {8: 'V1', 0: 'mHV', 1: 'mHV', 2: 'mHV', 9: 'mHV',
                5: 'lHV', 6: 'lHV', 3: 'aHV', 4: 'aHV'}
    regions = []
    for a in iarea:
        a_int = int(a)
        regions.append(area_map.get(a_int, 'unassigned'))
    return regions
...
brain_regions = ['V1', 'mHV', 'lHV', 'aHV', 'unassigned']
```

iii. CONVERSION_NOTES.md Step 3 "Curation Steps": "**Neuron curation rules**: No explicit filtering. All Suite2p-detected cells included." Step 10 Check 3: "Brain regions: Same neu_area_ID mapping (V1=8, mHV=0,1,2,9, lHV=5,6, aHV=3,4) ✓". The AI read the reference's area map correctly but concluded that no neuron should be discarded, so it invented an extra `'unassigned'` bucket rather than dropping out-of-area cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's first column is `floor(StartFr[tr])`, i.e. corridor entry, so every trial is aligned at t=0 to trial start. Trials keep their own length (no common end, no padding), and `metadata['temporal_alignment_event'] = 'Trial start (corridor entry)'`, `off_start = 0.0`, `off_end = None`.

ii.
```python
s_fr = start_fr_int[tr]
e_fr = end_fr_int[tr]
neural = spk[:, s_fr:e_fr].astype(np.float16)
...
'temporal_alignment_event': 'Trial start (corridor entry)',
'off_start': 0.0,
'off_end': None,
```

iii. CONVERSION_NOTES.md Step 5 Decision 5 ("Trial boundaries: StartFr to EndFr") and the `--show-processing` plots, which the AI used to check that inputs cross zero at the right place. All streams are cut with the identical `s_fr:e_fr` slice, which the AI argues guarantees alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling. The imaging frame *is* the time bin: 3.17 Hz → 315.46 ms, recorded in `metadata['time_bin_size']`. All behavioral streams (`ft_Pos`, `ft_RunSpeed`, `LickFr`, `StartFr`, `SoundFr`) are already on the frame grid, so nothing needs interpolation. Mean trial length is 59.3 bins (median 48.7, max 5621).

ii.
```python
FS = 3.17  # Frame rate in Hz
TIME_BIN_MS = 1000.0 / FS  # ~315.5 ms
...
'time_bin_size': TIME_BIN_MS,
'frame_rate_hz': FS,
```

iii. CONVERSION_NOTES.md Step 1/Step 4: "Frame rate: 3.17 Hz (calcium imaging)", sourced from the reference `data_process_script.ipynb`. The AI explicitly rejected the reference's position-interpolated (60-bin) representation: Step 10 Check 3, "We use time-series data, not position-interpolated. This is necessary for time-varying decoder outputs."

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. `beh['SoundFr']` (the fractional imaging-frame number of the cue in each trial) and the integer frame indices of the trial window. The real frame timestamps `beh['ft']` are **not** used; the nominal frame rate `FS = 3.17` converts frames to seconds instead.

ii.
```python
sound_fr = np.array(beh['SoundFr'])
...
frame_indices = np.arange(s_fr, e_fr, dtype=np.float64)
# Input 0: Time to sound cue (seconds) - negative before, positive after
time_to_cue = (frame_indices - sound_fr[tr]) / FS
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "SoundFr - frame_idx | input[0]: time_to_sound_cue | (frame - SoundFr) / fs | Seconds". The AI treats the frame index as a uniform clock at 3.17 Hz, which is what the reference's own frame rate constant implies.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Per-frame signed offset in seconds: `(frame_index - SoundFr[tr]) / 3.17`. This is *time since* the cue (negative before, positive after) — the opposite sign convention to the name and to the reference, which computes `cue_time - frame_time` (positive before). It is a continuous time-varying value, not a binary event marker. Because long stationary trials are retained, the resulting range is extreme: [-725.3, +1770.8] s.

ii.
```python
# Input 0: Time to sound cue (seconds) - negative before, positive after
time_to_cue = (frame_indices - sound_fr[tr]) / FS
...
inputs = np.stack([time_to_cue, day_input, time_since_start, reward_input],
                  axis=0).astype(np.float32)
```

iii. The AI's inline comment states the convention outright ("negative before, positive after"). CONVERSION_NOTES.md does not flag the sign mismatch with the variable name. Trajectory step 50 notes the large ranges (±200 s) as suspicious and step 174 repeats it ("very large range due to very long trials"), but the AI never traced it back to the unfiltered stationary trials.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is built from `np.arange(s_fr, e_fr)` — exactly the same frame window used to slice `spk`. So it has the same length as the trial's neural matrix and is sample-for-sample aligned.

ii.
```python
frame_indices = np.arange(s_fr, e_fr, dtype=np.float64)
time_to_cue = (frame_indices - sound_fr[tr]) / FS
...
neural = spk[:, s_fr:e_fr].astype(np.float16)
```

iii. All streams in this dataset are indexed by imaging frame, so re-using one `s_fr:e_fr` window for every stream makes alignment automatic. The `--show-processing` panel plots `time_to_cue` and `time_since_start` on the same time axis as the neural raster to make this visible.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Metadata fields of the index entry: `db['sess#']` if present, else `db['days']`, else 0.0. Session dates (`datexp`) are **not** used to order a mouse's sessions. Inspection of the raw index shows `sess#` is an index *within an experiment-type group* (0 = "before learning", 1 = "after learning", 2–4 for repeated naive tests), while `days` (present only for the `*_train2_after_learning` groups) is a genuine number of training days (6–15). Mixing the two yields a non-monotonic per-mouse sequence — e.g. DR10's six sessions in date order get 0, 1, 1, 6, 1, 1 — and 1.0 for the large majority of sessions. Range across the dataset is [0, 15].

ii.
```python
def get_day_of_training(db):
    """Get day of training from session metadata."""
    if 'sess#' in db:
        return float(db['sess#'])
    elif 'days' in db:
        return float(db['days'])
    else:
        return 0.0
```

iii. Trajectory step 40: "Most sessions have `sess#` = 0 (before learning) or 1 (after learning); Some sessions... use `days` instead of `sess#`... For the `day_of_training` input, I'll use `sess#` when available, and `days` when `sess#` is missing. This represents the day/session number in the training sequence." CONVERSION_NOTES.md Step 5 records it as "sess# or days | input[1]: day_of_training | Direct | Per-trial". The AI never checked the resulting sequence against `datexp`, and step 50 noted "day_of_training is constant at 1.0" in the sample without investigating.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The scalar is broadcast unchanged across every bin of every trial in the session (`np.full(n_trial_frames, day_of_training)`), i.e. constant per session, and stacked as input row 1.

ii.
```python
day_of_training = get_day_of_training(db)
...
# Input 1: Day of training (per-trial)
day_input = np.full(n_trial_frames, day_of_training, dtype=np.float64)
```

iii. The instructions call this a continuous, per-trial input; the AI broadcasts it to a constant time series so that all four input rows share one shape.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. `beh['StartFr']` (the fractional corridor-entry frame of each trial) and the integer frame indices of the trial window, again converted with the nominal `FS = 3.17`, not `beh['ft']`.

ii.
```python
start_fr = np.array(beh['StartFr'])
...
frame_indices = np.arange(s_fr, e_fr, dtype=np.float64)
# Input 2: Time since trial start (seconds)
time_since_start = (frame_indices - start_fr[tr]) / FS
```

iii. CONVERSION_NOTES.md Step 5: "frame_idx - StartFr | input[2]: time_since_trial_start | (frame - StartFr) / fs | Seconds". Same rationale as 3-a: frames are a uniform clock.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. `(frame_index - StartFr[tr]) / 3.17`, positive after entry, matching the reference's sign. Note the *unrounded* `start_fr[tr]` is subtracted while the window begins at `floor(StartFr[tr])`, so the first bin sits in [-0.32, 0] s rather than exactly 0 — the observed global minimum is -0.3 s, which is the intended sub-frame offset, not an error. The maximum reaches 1772.7 s because of the unfiltered 5621-frame trial.

ii.
```python
start_fr_int = np.floor(start_fr).astype(int)
...
time_since_start = (frame_indices - start_fr[tr]) / FS
```

iii. Not separately justified in CONVERSION_NOTES.md beyond the Step 5 mapping table; the `--show-processing` plot of the two time inputs against the neural raster is the AI's check.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same `np.arange(s_fr, e_fr)` window as the neural slice, so it is aligned bin-for-bin and has identical length.

ii.
```python
frame_indices = np.arange(s_fr, e_fr, dtype=np.float64)
time_since_start = (frame_indices - start_fr[tr]) / FS
neural = spk[:, s_fr:e_fr].astype(np.float16)
```

iii. One shared frame window for all streams; verified visually in `--show-processing` (row 1 of the figure) and asserted in CONVERSION_NOTES.md Step 10 Check 2 ("time_to_sound_cue matches manual calculation ✓").

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. `beh['isRew']`, the per-trial boolean flag for the rewarded corridor, cast to float.

ii.
```python
is_rew = np.array(beh['isRew']).astype(float)
...
# Input 3: Reward availability
reward_input = np.full(n_trial_frames, is_rew[tr], dtype=np.float64)
```

iii. CONVERSION_NOTES.md Step 5: "isRew | input[3]: reward_availability | Boolean to float | Per-trial"; Step 10 Check 2: "Reward availability matches original isRew ✓".

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. None beyond the bool→float cast and broadcasting to the trial's bins. It is 0 for every trial of unsupervised/naive sessions and ~50% for supervised sessions, which is what the verification shows (per-session input-3 ranges are either [0,0] or [0,1]).

ii.
```python
reward_input = np.full(n_trial_frames, is_rew[tr], dtype=np.float64)
```

iii. CONVERSION_NOTES.md Step 9: "Reward rate | ~50% | 49.8% (sup session) | ✓" and Step 10 Check 5: "Sessions with no reward (unsupervised): All reward values = 0 ✓".

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. `beh['WallName']`, the per-trial wall-texture name. The global label vocabulary is the sorted union of `beh['UniqWalls']` over every session (15 names). `TrialStim` is not used.

ii.
```python
wall_name = np.array(beh['WallName'])
stim_to_idx = {s: i for i, s in enumerate(stim_list)}
...
all_stim = set()
for exp_type, sessions in info.items():
    ...
    for w in beh_cache[exp_type][beh_key]['UniqWalls']:
        all_stim.add(str(w))
stim_list = sorted(all_stim)
```

iii. Trajectory step 39: the AI enumerated the 15 names and noted "'brick' in the paper corresponds to 'wood' in the data", so it knew the mapping to the paper's four coarse textures existed.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each trial's `WallName` is mapped to its index in the 15-element `stim_list` (with a silent `.get(..., 0)` fallback that in practice never fires — every `WallName` is in `UniqWalls`). The label is broadcast across all bins of the trial, including the grey-space bins where no texture is on screen. Crucially the AI keeps all **15 sub-textures as separate classes** (`circle1/2/3`, `leaf1/2/3`, `leaf1_swap1/2`, `rock1/2`, `wood1/2/5`, `wood1_swap1/2`) rather than collapsing them onto the four base textures. This makes chance 1/15 = 0.067, makes the class distribution highly unbalanced (circle1 0.265, leaf1 0.280 ... wood1_swap1 0.007), and leaves most sessions containing only 2–4 of the 15 labels.

ii.
```python
stim_name = str(wall_name[tr])
stim_idx = stim_to_idx.get(stim_name, 0)
stim_output = np.full(n_trial_frames, stim_idx, dtype=np.int64)
...
output_values = [
    stim_list,
    ...
]
```

iii. CONVERSION_NOTES.md Step 5: "WallName | output[0]: visual_stimulus | Map to category index | 15 categories"; Step 9 consistency table: "Stimuli | circle, leaf, rock, brick | 15 unique stimuli | ✓" — i.e. the AI knowingly reported a 15-way label set against the paper's four and marked it consistent. Step 12 Check 2 justifies the accuracy in terms of "15 stimulus categories (chance 6.7%) vs 34% achieved".

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. `beh['LickFr']` (fractional imaging-frame number of each lick) together with `beh['LickTrind']` (the trial index each lick belongs to). The reference uses `LickFr` alone.

ii.
```python
lick_fr = np.array(beh['LickFr']).astype(float) if len(beh['LickFr']) > 0 else np.array([])
lick_trind = np.array(beh['LickTrind']).astype(int) if len(beh['LickTrind']) > 0 else np.array([], dtype=int)
```

iii. CONVERSION_NOTES.md Step 5: "LickFr, LickTrind | output[1]: licking | Binary per frame | Time-varying". Using `LickTrind` lets the AI restrict the search to the current trial's licks rather than building a session-wide flag vector.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the licks with `LickTrind == tr` have their frame numbers **rounded** (`np.round`, vs the reference's truncation) to the nearest frame, shifted by `-s_fr` into trial-local indices, bounds-checked against `[0, n_trial_frames)`, and used to set a binary vector to 1. A bin is 1 if ≥1 lick falls in it. Empty `LickFr`/`LickTrind` arrays (all unsupervised sessions) are handled explicitly and produce an all-zero row. Overall 2.7% of bins are licks.

ii.
```python
lick_binary = np.zeros(n_trial_frames, dtype=np.int64)
if len(lick_trind) > 0:
    trial_lick_mask = lick_trind == tr
    if np.any(trial_lick_mask):
        trial_lick_frames = np.round(lick_fr[trial_lick_mask]).astype(int)
        trial_lick_frames = trial_lick_frames - s_fr
        valid = (trial_lick_frames >= 0) & (trial_lick_frames < n_trial_frames)
        if np.any(valid):
            lick_binary[trial_lick_frames[valid]] = 1
```

iii. CONVERSION_NOTES.md Step 10 Check 2 #6: "Licking data present in supervised sessions (280/470 trials) ✓" and Check 5: "Sessions with no licking (unsupervised): All lick values = 0 ✓". The task spec asks for a binary time series for event-type outputs, which this satisfies.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` already indexes imaging frames, so subtracting `s_fr` puts licks on the same grid and length as the trial's neural columns. The `valid` mask drops any lick that would land outside the window.

ii.
```python
trial_lick_frames = np.round(lick_fr[trial_lick_mask]).astype(int) - s_fr
valid = (trial_lick_frames >= 0) & (trial_lick_frames < n_trial_frames)
lick_binary[trial_lick_frames[valid]] = 1
...
neural = spk[:, s_fr:e_fr].astype(np.float16)
```

iii. Shared frame indexing; the `--show-processing` figure plots the lick row on the same time axis as the neural raster.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. `beh['ft_Pos']`, the VR position in decimetres at each imaging frame, truncated to the number of imaged frames `nfr`. The corridor runs 0–40 dm of texture followed by 40–60 dm of grey space.

ii.
```python
nfr = spk.shape[1]
ft_pos = np.array(beh['ft_Pos'])[:nfr]
...
pos = ft_pos[s_fr:e_fr]
pos_binned = discretize_position(pos)
```

iii. CONVERSION_NOTES.md Step 5: "ft_Pos | output[2]: position_bin | 4 bins of 1m each | Time-varying"; Step 3 records the corridor geometry from the paper ("corridors were each 4 m long, with 2 m of grey space").

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Integer division of the decimetre position by 10 (= 1 m per bin), then a clip into [0, 3]. No unit conversion or smoothing. Values are broadcast per frame and stored as int64.

ii.
```python
POS_BIN_SIZE_DM = TEXTURE_LENGTH_DM / N_POS_BINS  # 10 decimeters = 1 meter

def discretize_position(position_dm):
    bins = np.clip(position_dm // POS_BIN_SIZE_DM, 0, N_POS_BINS - 1).astype(np.int64)
    return bins
```

iii. CONVERSION_NOTES.md Step 5 Decision 3: "**Position bins**: 4 bins of 1m covering 4m corridor. Gray space (>4m) assigned to bin 3."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are at 10, 20, 30 dm, giving `0-1m / 1-2m / 2-3m / 3-4m`. Because the trial window extends through the grey space (see 1-d), every frame at 40–60 dm — the last 2 m of the traversal — is clipped into bin 3 and labelled `"3-4m"`. The consequence is visible in the verification summary: the four bins hold 0.194 / 0.159 / 0.161 / **0.486** of the data, i.e. roughly one third of all bins carry a wrong position label, and the bins are not the 4 equal-length 1-m bins the task specifies.

ii.
```python
bins = np.clip(position_dm // POS_BIN_SIZE_DM, 0, N_POS_BINS - 1).astype(np.int64)
...
output_values = [
    stim_list,
    ['not_licking', 'licking'],
    ['0-1m', '1-2m', '2-3m', '3-4m'],
    ['Q1_slowest', 'Q2', 'Q3', 'Q4_fastest'],
]
```

iii. CONVERSION_NOTES.md Step 12 "Issues Found and Resolved": "Position bin 3-4m includes gray space, making it 48.6% of data. This is the correct behavior since the corridor includes gray space." The AI saw the imbalance, named its cause, and declared it correct rather than restricting to the textured corridor (`ft_CorrSpc` / `GrayFr`) or adding grey-space bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is one value per imaging frame, sliced with the identical `s_fr:e_fr` window as the neural matrix, so it is aligned bin-for-bin and of equal length. `ft_Pos` is pre-truncated to `nfr` so the slice can never run past the imaging.

ii.
```python
ft_pos = np.array(beh['ft_Pos'])[:nfr]
pos = ft_pos[s_fr:e_fr]
neural = spk[:, s_fr:e_fr].astype(np.float16)
```

iii. Shared frame indexing; the `--show-processing` bottom row plots the position bin against the same time axis as the neural raster.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. `beh['ft_RunSpeed']`, the treadmill running speed at each imaging frame, truncated to `nfr` for the per-trial slice. For the bin boundaries it is used **untruncated and session-wide** — every frame of every session, including frames outside any trial.

ii.
```python
ft_run_speed = np.array(beh['ft_RunSpeed'])[:nfr]
...
for exp_type, sessions in info.items():
    ...
    beh = beh_cache[exp_type][beh_key]
    speeds = np.array(beh['ft_RunSpeed'])
    all_speeds.append(speeds)
all_speeds = np.concatenate(all_speeds)
```

iii. CONVERSION_NOTES.md Step 5: "ft_RunSpeed | output[3]: running_speed_bin | 4 quartile bins | Time-varying".

## 10-b. What processing is involved in computing `output` *Running speed*?

i. One global pass over all 89 sessions concatenates every `ft_RunSpeed` sample and takes the 25/50/75th percentiles, yielding boundaries `[0.0, 9.667, 31.456]` (the printed speed range is [-34.31, 383.37]). These three global thresholds are then applied with `np.digitize` to every trial. Nothing is done per session, and stationary/non-running frames are not excluded.

ii.
```python
def compute_global_speed_quartiles(info):
    ...
    all_speeds = np.concatenate(all_speeds)
    quartiles = np.percentile(all_speeds, [25, 50, 75])
    return quartiles

def discretize_speed(speed, quartiles):
    """Discretize speed into 4 quartile bins. Returns integer values 0-3."""
    bins = np.digitize(speed, quartiles)  # Returns 0,1,2,3
    return bins.astype(np.int64)
```

iii. CONVERSION_NOTES.md Step 5 Decision 4: "**Speed quartiles**: Computed across all frames in all sessions." Trajectory step 42 shows the AI first got boundaries `[0, 1.48, 20.53]`, recognised that "Q1 is basically 0 speed (not moving)", and considered restricting to running frames (`ft_move > 0`) as the reference does — but it rejected the running-frame filter (step 43) to preserve temporal continuity and kept the all-frames quartiles.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize(speed, [0.0, 9.667, 31.456])` with the default `right=False`: bin 0 is `speed < 0`, bin 1 is `0 <= speed < 9.667`, bin 2 `< 31.456`, bin 3 above. Because the 25th percentile lands exactly on 0.0 and a large mass of frames sits at exactly zero speed, the tie is broken entirely into bin 1. The realised distribution is 0.092 / 0.409 / 0.250 / 0.249 — not the "4 bins, each corresponding to 25% of the data" the task specifies, and bin 0 ends up meaning "negative (backwards) speed" rather than "slowest quartile", despite being labelled `Q1_slowest`.

ii.
```python
bins = np.digitize(speed, quartiles)  # Returns 0,1,2,3
...
['Q1_slowest', 'Q2', 'Q3', 'Q4_fastest'],
...
'speed_quartile_boundaries': speed_quartiles.tolist(),
```

iii. CONVERSION_NOTES.md Step 12 "Issues Found and Resolved": "Speed quartile Q2 has 40.9% of data because many frames have speed near 0 (not running). Quartiles are computed across all frames." The AI diagnosed the tie problem exactly and left it unaddressed; it never considered a rank-based split, which would break the tie into equal quarters.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is one value per imaging frame, truncated to `nfr`, and sliced with the same `s_fr:e_fr` window as the neural matrix — aligned bin-for-bin, equal length.

ii.
```python
ft_run_speed = np.array(beh['ft_RunSpeed'])[:nfr]
speed = ft_run_speed[s_fr:e_fr]
speed_binned = discretize_speed(speed, speed_quartiles)
neural = spk[:, s_fr:e_fr].astype(np.float16)
```

iii. Shared frame indexing, plotted together with position in the `--show-processing` figure.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several defensive measures, all of which are logged rather than silent:
  - Behavior can run past the imaging, so `ft_Pos` and `ft_RunSpeed` are truncated to `nfr = spk.shape[1]`, and `end_fr_int` is clipped to `nfr` (the same `[:nfr]` cut the reference makes).
  - `StartFr[0]` can be negative (observed: -1); `start_fr_int` is clipped to `[0, nfr-1]`.
  - Trials whose imaged window has <2 frames are skipped with a count printed.
  - A mismatch between the spike neuron count and the retinotopy `iarea` length truncates both to the minimum, with a WARNING.
  - A behavior key missing from its `Beh_<exp_type>.npy` file prints a WARNING and skips the session.
  - Empty `LickFr`/`LickTrind` arrays are special-cased; licks falling outside the window are dropped by the `valid` mask.
  - Unknown wall names fall back to index 0 (never triggered in practice).
  - Sessions ending with <2 trials are dropped so the decoder can always split train/val.
  Not handled: extreme-length trials (see 1-e), and `iarea` codes outside the four visual areas are bucketed as `'unassigned'` rather than dropped.

ii.
```python
nfr = spk.shape[1]
ft_pos = np.array(beh['ft_Pos'])[:nfr]
ft_run_speed = np.array(beh['ft_RunSpeed'])[:nfr]
start_fr_int = np.clip(start_fr_int, 0, nfr - 1)
end_fr_int = np.clip(end_fr_int, 0, nfr)
...
if n_neu_spk != n_neu_ret:
    print(f"  WARNING: Neuron count mismatch: spk={n_neu_spk}, retino={n_neu_ret}")
    min_n = min(n_neu_spk, n_neu_ret)
    spk = spk[:min_n]
...
if beh_key not in beh_cache[exp_type]:
    print(f"  WARNING: Session key {beh_key} not found in behavior data")
    continue
```

iii. CONVERSION_NOTES.md Step 10 Check 5 "Edge cases" lists negative `StartFr`, very long trials, no-lick sessions and no-reward sessions. The AI's stated principle is to clip/truncate to the imaged frame range and log anything unexpected rather than fail.

## 12-a. What are the most time-consuming steps of the code?

i. Disk I/O on the spike files dominates. Per-session logs show `load_spk` taking most of the wall clock (the 405 GB of `spk/*.npy` must be read and concatenated); after neuron subsampling, sessions dropped from 10–100+ s to 2–6 s, and the whole 89-session conversion finished in 9.8 minutes. Secondary costs are the two extra full passes over all 23 behavior files (unique-stimulus scan and `compute_global_speed_quartiles`) before any conversion starts, and the final `pickle.dump` of the 20.26 GB output. Within `process_session` the per-trial Python loop is negligible by comparison.

ii.
```python
print(f"  Loading neural data...")
t0 = time.time()
spk = load_spk(db)
print(f"  Neural data loaded: {spk.shape} in {time.time()-t0:.1f}s")
...
sess_time = time.time() - sess_start
print(f"  Session done: {len(neural_trials)} trials, {spk.shape[0]} neurons, {sess_time:.1f}s")
...
print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"Time per session: {total_time/len(neural_all):.1f}s")
```

iii. CONVERSION_NOTES.md Step 7 records the timing estimate ("~6.6s per session, ~9.8 min total"). Trajectory step 160: "Sessions are completing in 2-6 seconds instead of 10-100+ seconds... Session time is now dominated by loading neural data, not processing." The AI's conclusion is that I/O is the floor and the remaining cost is not worth optimising.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Three:
  1. `neu_area_ID` walks `iarea` one neuron at a time in Python, building a list of strings, and `get_brain_region_idx` then walks that list again to map strings back to indices. On 20k–90k neurons per session this is the slowest pure-Python loop in the script; two `np.isin` calls (as the reference does) would replace both.
  2. The `for tr in range(ntrials)` loop in `process_session` recomputes `np.arange`, `np.full`, `np.stack` and the boolean `lick_trind == tr` scan (an O(n_licks) pass per trial, so O(ntrials × n_licks) overall) for each of up to 789 trials. Position and speed could be binned once for the whole session and then sliced, and the lick flag could be built once per session with a single scatter.
  3. `compute_global_speed_quartiles` accumulates a Python list of per-session arrays and concatenates ~30 M samples at the end; a streaming histogram or reservoir would avoid materialising it.
  None of these matter next to the spike-file I/O.

ii.
```python
def neu_area_ID(iarea):
    regions = []
    for a in iarea:
        a_int = int(a)
        regions.append(area_map.get(a_int, 'unassigned'))
    return regions

def get_brain_region_idx(iarea, brain_regions):
    region_names = neu_area_ID(iarea)
    region_to_idx = {r: i for i, r in enumerate(brain_regions)}
    return np.array([region_to_idx[r] for r in region_names], dtype=np.int64)
...
for tr in range(ntrials):
    ...
    trial_lick_mask = lick_trind == tr
```

iii. CONVERSION_NOTES.md does not analyse loop vectorisation; the AI's only efficiency work was float16 storage and neuron subsampling, and it stopped once the run fit in ~10 minutes (the instructions' 15-minute budget).

## 12-c. What processing does the code repeat multiple times?

i. The behavior `.npy` files are loaded from disk **three separate times**: `beh_cache` is populated in the unique-stimulus scan, populated again inside `compute_global_speed_quartiles`, and populated a third time in the main conversion loop — with `del beh_cache` between each. Related repeats: the session de-duplication (`seen_keys`) is done independently in `main` and again inside `compute_global_speed_quartiles`; `get_session_key(db)` is recomputed in all three passes; and `stim_to_idx` / the `region_to_idx` dict are rebuilt for every session and (for `stim_to_idx`) every call of `process_session`. Reading each behavior file once and passing it through all three passes would eliminate two-thirds of the behavior I/O.

ii.
```python
    # pass 1
    for exp_type, sessions in info.items():
        if exp_type not in beh_cache:
            beh_cache[exp_type] = load_beh(exp_type)
    ...
    speed_quartiles = compute_global_speed_quartiles(info)   # pass 2, own beh_cache
    del beh_cache
    ...
    beh_cache = {}                                           # pass 3
    for sess_idx, (exp_type, db, key) in enumerate(all_sessions):
        if exp_type not in beh_cache:
            beh_cache[exp_type] = load_beh(exp_type)
```

iii. Not discussed in CONVERSION_NOTES.md. The repetition is a side effect of adding the global-stimulus and global-quartile passes as separate helpers after the main loop was already written; the AI's `del beh_cache` calls show it was managing memory rather than I/O.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
  - **Neuron count vs the decoder.** The AI subsamples to 5000 neurons per session specifically because "the decoder uses PCA to 100 components", yet `train_decoder.py` then random-projects 5000 → 2000 before its SVD on every session, so a large part of the retained neural array is not used at full resolution. Conversely, ~90% of the recorded neurons were discarded up front and can never be recovered from the pickle.
  - **Outputs stored as int64.** All four output rows are categorical with ≤15 levels but are written as 8-byte integers (the reference uses int8), inflating the output arrays 8×.
  - **`compute_global_speed_quartiles` scans the full untruncated `ft_RunSpeed`** of every session, including frames past `nfr` and frames outside any trial, none of which ever appear in the converted data — so the thresholds are fitted partly on samples that are then thrown away.
  - **The unique-stimulus pass reads all 23 behavior files** purely to build a 15-element name list.
  - **`'unassigned'` neurons** (53,664, 12% of those kept) are outside the four visual areas the paper analyses.
  - **`end_fr`/`start_fr` are re-derived and re-clipped per session** and `GrayFr`, `ft_CorrSpc`, `ft_move`, `ft` are loaded implicitly with the behavior dict but never used.
  - The `--show-processing` matplotlib figures are diagnostic only.

ii.
```python
MAX_NEURONS = 5000  # Subsample neurons for manageable file size; decoder uses PCA to 100 components
...
outputs = np.stack([stim_output, lick_binary, pos_binned, speed_binned],
                   axis=0).astype(np.int64)
...
speeds = np.array(beh['ft_RunSpeed'])      # full session, not [:nfr], not trial-restricted
all_speeds.append(speeds)
```

iii. CONVERSION_NOTES.md Step 5 Decision 1 gives the PCA argument for subsampling; Decision 6 gives the float16 argument for the neural array but no equivalent reasoning is applied to the int64 outputs. The wasted behavior passes and the unused-frame quartile fit are not discussed.
