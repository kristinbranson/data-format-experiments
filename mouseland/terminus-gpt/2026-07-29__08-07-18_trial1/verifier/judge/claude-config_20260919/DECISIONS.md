# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the dataset's master index (`beh/Imaging_Exp_info.npy`). Instead it globs every `*.npy` in `data/beh/`, loads each one fully, and keeps any top-level key that matches a session-id regex (`<mouse>_<YYYY>_<MM>_<DD>_<blk>` with an optional `_swap1/_swap2` suffix) whose value is a dict. When the same session id appears in more than one behavior file it keeps the "richer" copy, scored as `len(dict) + ft_trInd.size`. Neural data are found by a separate recursive glob for `*_neural_data.npy`, keyed by file stem. The dataset that is actually converted is the **intersection** of the two key sets. The retinotopy directory is never opened. The log reports `Loaded 100 behavior sessions, 89 neural sessions, 76 common sessions` — i.e. 13 of the 89 recordings are silently dropped because their behavior is keyed with a `_swap1`/`_swap2` suffix that never matches a spike-file stem, and that 89 → 76 gap is never reconciled.

ii.
```python
SESSION_RE = re.compile(r'^[A-Za-z0-9]+_\d{4}_\d{2}_\d{2}_\d+(?:_swap[12])?$')

def load_behavior_sessions(data_dir: Path):
    sessions = {}
    session_source = {}
    for p in sorted((data_dir / 'beh').glob('*.npy')):
        obj = np.load(p, allow_pickle=True).item()
        if not isinstance(obj, dict):
            continue
        for sess, dat in obj.items():
            if isinstance(sess, str) and SESSION_RE.match(sess) and isinstance(dat, dict):
                if sess not in sessions or _behavior_richness(dat) > _behavior_richness(sessions[sess]):
                    sessions[sess] = dat
                    session_source[sess] = p.name
    return sessions, session_source

def load_neural_files(data_dir: Path):
    files = {}
    for p in sorted(data_dir.rglob('*_neural_data.npy')):
        files[p.name.replace('_neural_data.npy', '')] = p
    return files
```
```python
common = sorted(set(beh_sessions) & set(neural_files))
...
nobj = np.load(neural_path, allow_pickle=True).item()
spk  = concat_spks(nobj['spks'])          # np.concatenate over imaging planes, float32
```

iii. From CONVERSION_NOTES.md Step 2/4: "`data/beh/Imaging_Exp_info.npy` appears to store experiment-group metadata/indices rather than individual sessions" and "Behavior files include both true session keys and metadata/group keys → Filter behavior keys to session-like IDs when matching to neural files." The richness-based de-duplication was added in Step 10 after the AI noticed that `example_bef_and_aft_learning_behavior.npy` was overwriting the real entry for `TX109_2023_03_27_1`, costing it one session. No justification is given anywhere for the remaining 13 unmatched neural files, and the trajectory shows the retinotopy folder was only ever seen in a directory listing (steps 8–9) and never read.

## 1-b. How are the data split into subjects?

i. The subject is the substring of the session id before the first underscore. `subjects` is the sorted unique set over the converted sessions, and `subject_idx` is that index per session. This yields 19 mice, matching the index and the reference.

ii.
```python
subjects = sorted({s.split('_')[0] for s in common})
subj_to_id = {s: i for i, s in enumerate(subjects)}
...
data['subject_idx'].append(subj_to_id[sess.split('_')[0]])
...
data['subject_idx'] = np.asarray(data['subject_idx'], dtype=np.int64)
```

iii. CONVERSION_NOTES.md Step 2: "Session naming convention embeds subject ID, date, and session number"; Step 2 table records "19 true animal IDs observed in session-like keys". The mouse name is part of the key, so no separate lookup is needed.

## 1-c. How are the data split into sessions?

i. A session is one behavior key that also has a spike file of the same name, i.e. one mouse / date / block. Sessions are processed in sorted order of that id. A session is dropped if it yields fewer than two usable trials. 76 sessions are written. Sessions whose behavior is stored under a swap-stimulus key (`..._swap1`, `..._swap2`) never intersect a spike-file stem and are lost — 13 recordings, including every session of the stimulus-swap experiment.

ii.
```python
common = sorted(set(beh_sessions) & set(neural_files))
...
for sess in common:
    neural_trials, input_trials, output_trials = build_session(
        beh_sessions[sess], beh_source[sess], neural_files[sess])
    if len(neural_trials) < 2:
        continue
```

iii. CONVERSION_NOTES.md Step 9/10 documents only the one session recovered by the de-duplication fix ("Corrected full conversion now includes all 76 common behavior+neural sessions"). The AI treats "76 common" as the ground-truth denominator; the mismatch against the 89 available spike files is printed by its own script but never investigated.

## 1-d. Are the data correctly split into trials / how are the data split into trials?

i. Trials come from the frame-aligned label `ft_trInd`. Frames with `NaN` trial labels are discarded; the remaining valid frames are sliced between successive first-occurrences of each unique trial id, which (verified on four sessions) reproduces the exact `ft_trInd == t` mask. Crucially the AI does **not** intersect with `ft_CorrSpc`, so each trial contains the whole traversal *including the ~2 m grey space* (≈27% of frames). Median trial length is therefore ~35 frames instead of the ~26 frames of the textured corridor, and the reported `T` statistics run to 5621 frames.

ii.
```python
valid_frame = np.isfinite(ft_tr)
ft_tr_int = np.full(n_frames, -1, dtype=int)
ft_tr_int[valid_frame] = ft_tr[valid_frame].astype(int)

valid_idx = np.flatnonzero(ft_tr_int >= 0)
valid_tr = ft_tr_int[valid_idx]
uniq_tr, starts = np.unique(valid_tr, return_index=True)
trial_frame_idx = {int(tr): valid_idx[starts[i]:starts[i+1]] if i+1 < len(starts)
                   else valid_idx[starts[i]:] for i, tr in enumerate(uniq_tr)}
```

iii. CONVERSION_NOTES.md Step 6: "Implemented frame-aligned conversion using `ft_trInd`, `ft_PosCum`, `ft_move`, ...". Trajectory step 238: "use `ft_trInd` to split trials, `ft_PosCum` to compute position, `ft_move` to optionally restrict moving frames if matching reference". The AI saw `ft_CorrSpc` and `ft_GraySpc` listed in the reference `utils.py` (trajectory steps 124–125) but never used either, and gives no reason for keeping the grey space.

## 1-e. How are trials filtered based on quality controls?

i. Essentially no quality control. A trial is kept if it has ≥ 2 valid frames and if its index is within bounds of the per-trial arrays (`WallName`, `isRew`, `SoundTime`, `Trial_start_time`). There is no trial-length outlier filter, no exclusion of stationary animals, and no check on sound/reward validity. The full-data verification reports trials up to `T = 5621` frames (≈29 min) and `time_since_trial_start` up to 1769 s, i.e. the "parked animal" trials the reference explicitly removes are retained.

ii.
```python
for tr, idx in trial_frame_idx.items():
    if idx.size < 2:
        continue
    if tr >= len(wall) or tr >= len(is_rew) or tr >= len(sound) or tr >= len(tstart):
        continue
```

iii. CONVERSION_NOTES.md Step 3 says only "Need further confirmation from code/data on any explicit exclusion of invalid trials", and Step 5 (Mapping Planning, where curation rules were to be documented) was left as the unfilled template with `**Status**: IN PROGRESS`. No justification for the absence of trial curation is given anywhere.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Exclusively `spks` in `spk/<session_id>_neural_data.npy`, a list of per-imaging-plane `(n_neurons, n_frames)` arrays, concatenated along the neuron axis. This is the same source as the reference. The companion retinotopy file (`retinotopy/<mouse>_<date>_trans.npz`, field `iarea`) that assigns each neuron a visual area is never loaded.

ii.
```python
def concat_spks(spks_list):
    return np.concatenate([np.asarray(x, dtype=np.float32) for x in spks_list], axis=0)
...
nobj = np.load(neural_path, allow_pickle=True).item()
spk = concat_spks(nobj['spks'])
```

iii. CONVERSION_NOTES.md Step 4: "Treat `spks` as the deconvolved neural activity stream for conversion"; Step 3 quotes the methods, "All our analyses were based on deconvolved fluorescence traces." Nothing is said about `iarea`/retinotopy.

## 2-b. How is the `neural` data processed?

i. No processing at all: no dF/F, no deconvolution, no normalization, no smoothing. The traces are cast to **float32**, truncated to the number of frames common to the spike matrix and the frame-aligned behavior arrays, and then sliced per trial. Trials keep their natural length; nothing is padded. The float32 choice (vs the reference's float16), combined with keeping every neuron and every grey-space frame, produced a **372 GB** `converted_data.pkl`, which is why full decoder training never completed (Step 11 left `IN PROGRESS`).

ii.
```python
n_frames = min(spk.shape[1], len(ft_tr), len(ft_pos), len(ft_move), len(ft_wall))
spk = spk[:, :n_frames]
...
tr_spk = spk[:, idx].astype(np.float32, copy=False)
```

iii. CONVERSION_NOTES.md Step 3/4: the data are already Suite2p-deconvolved traces and "Analyses are based on deconvolved fluorescence traces", so no further processing is warranted. No rationale is given for float32; Step 11 concedes "`train_decoder_full_out.txt` remained empty ... indicating that loading/training on the 344G pickle is impractically slow in the current storage format. Additional optimization/compression of the converted dataset would be required."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neural filtering whatsoever. Every ROI in `spks` is kept. Because the retinotopy is never read, all neurons are assigned a single placeholder brain region, so the required `brain_regions` / `brain_region_idx` metadata carries no information: `brain_regions = ['unknown']` and `brain_region_idx` is an all-zero vector per session. The verification log reports `unknown: 4047169 neurons` over 76 sessions; the reference keeps 4,105,393 of 4,691,034 neurons over 89 sessions after restricting to V1/mHV/lHV/aHV.

ii.
```python
'brain_regions': ['unknown'],
...
data['brain_region_idx'].append(np.zeros(neural_trials[0].shape[0], dtype=np.int64))
```

iii. CONVERSION_NOTES.md Step 3: "Suite2p cell classification was used; exact downstream neuron inclusion criteria not yet identified from current methods excerpt." That open item is never closed, and the retinotopy directory — which is exactly the missing piece — is listed in Step 0/trajectory step 8 but never opened.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is by frame index: each trial's neural matrix is `spk[:, idx]`, where `idx` is that trial's frames in `ft_trInd`, so column 0 is the first imaged frame of the trial (≈ corridor entry). Trials are variable length, not padded, not truncated to a common window. Metadata records `temporal_alignment_event = 'corridor entry / trial start'`, `off_start = 0.0`, `off_end = None`. Every other stream is sliced with the same `idx`, so the streams are mutually consistent by construction.

ii.
```python
tr_spk = spk[:, idx].astype(np.float32, copy=False)
...
'temporal_alignment_event': 'corridor entry / trial start',
'off_start': 0.0,
'off_end': None,
```

iii. Trajectory step 238: the AI found `ft_*` arrays are on the imaging-frame grid and are the same length as `spks` up to a 2-frame mismatch handled by `[:nfr]` trimming, "exactly the kind of alignment logic we need"; CONVERSION_NOTES.md Step 6 records the frame-aligned implementation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling — the imaging frame is the bin (~0.315 s). However, the AI never reads the frame clock `ft`. It **estimates** a per-session scalar `dt_frame` as the median over trials of `(Trial_end_time − Trial_start_time) / n_frames_in_trial`, falling back to 0.1 s if no trial qualifies. On a checked session this gives 0.31512 s vs the true median `ft` interval 0.31474 s. `dt_frame` is used only to build the time-valued inputs and to bin licks; it is never written out. `metadata['time_bin_size']` is left as **`None`**, violating the target-format requirement that it be a float in ms.

ii.
```python
trial_dur_sec = (tend - tstart) * 86400.0
frame_periods = []
for tr in range(min(ntrials, len(trial_dur_sec))):
    nfr = np.sum(ft_tr_int == tr)
    if nfr > 1 and np.isfinite(trial_dur_sec[tr]) and trial_dur_sec[tr] > 0:
        frame_periods.append(trial_dur_sec[tr] / nfr)
dt_frame = float(np.nanmedian(frame_periods)) if frame_periods else 0.1
```
```python
'time_bin_size': None,
```

iii. No explicit justification. The AI's notes list "Behavior data time bin | [not explicitly stated in methods excerpt]" (Step 3) and never returned to fill it in; Step 5, where binning was to be decided, is the unfilled template.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. `SoundTime` (MATLAB datenum of the cue, per trial) and `Trial_start_time` (per trial), plus the synthetic within-trial clock `arange(n) * dt_frame`. The frame-indexed cue variable `SoundFr` and the true frame timestamps `ft` — both present in the behavior dict and both used by the reference — are not used.

ii.
```python
sound  = np.asarray(beh['SoundTime'], dtype=float)
tstart = np.asarray(beh['Trial_start_time'], dtype=float)
...
sound_sec = float((sound[tr] - tstart[tr]) * 86400.0) \
    if np.isfinite(sound[tr]) and np.isfinite(tstart[tr]) else np.nan
```

iii. CONVERSION_NOTES.md Step 5 planning lists "`SoundTime`, `SoundTimeDelay`" among the fields to inspect; trajectory step 115 planned inputs "time to sound cue ... from `SoundTime`". The AI discovered the `ft_*` frame grid only later (step 238) and converted position/licking/trials to it, but left the cue on the datenum path.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Cue time is expressed in seconds after trial start (`×86400` to convert datenums to seconds), and the input is `cue_seconds − time_since_trial_start`, i.e. **positive before the cue and negative after** — the same sign convention as the reference. If `SoundTime` or `Trial_start_time` is non-finite the whole trial's row is `NaN` (no such trials exist in the files checked). Because the time axis is a synthetic uniform clock anchored on the first imaged frame while the cue is anchored on `Trial_start_time`, there is a systematic ~0.15 s (half-frame) offset plus a drift of typically < 0.05 s by trial end.

ii.
```python
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
time_to_sound = (sound_sec - time_since).astype(np.float32) if np.isfinite(sound_sec) \
    else np.full(idx.size, np.nan, dtype=np.float32)
inp = np.vstack([time_to_sound, day_arr, time_since, reward_arr]).astype(np.float32)
```

iii. No explicit written justification beyond the mapping plan. The range check in Step 7 ("time_to_sound_cue range [-293.4, 200.1]") was recorded but not questioned, even though a ±300 s cue offset only makes sense for the un-curated multi-minute trials.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is built on the same per-trial frame window `idx` as the neural matrix and has exactly `idx.size` samples, so it is aligned to the neural data bin-for-bin by construction. The residual inaccuracy is the sub-frame offset/drift of the reconstructed clock described in 3-b.

ii.
```python
for tr, idx in trial_frame_idx.items():
    ...
    time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
    time_to_sound = (sound_sec - time_since).astype(np.float32)
    inp = np.vstack([time_to_sound, day_arr, time_since, reward_arr]).astype(np.float32)
```

iii. Trajectory step 238: everything is put "on the frame grid, using per-trial arrays repeated over frames where needed."

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Not from the data at all — from the **file name** of the behavior file the session key was found in (`beh_source`). `infer_day()` pattern-matches that name: `*before_learning*`/`*before_grating*` → 1.0, `*after_learning*`/`*after_grating*` → 5.0, else `test(\d+)` → N, else `train(\d+)` → N, else 0.0. Session dates (which are in the session id, and which the reference uses to order a mouse's sessions) are ignored.

ii.
```python
def infer_day(source_name: str):
    name = source_name.lower()
    if 'before_learning' in name or 'before_grating' in name:
        return 1.0
    if 'after_learning' in name or 'after_grating' in name:
        return 5.0
    m = re.search(r'test(\d+)', name)
    if m:
        return float(m.group(1))
    m = re.search(r'train(\d+)', name)
    if m:
        return float(m.group(1))
    return 0.0
...
day = infer_day(beh_source)
```

iii. CONVERSION_NOTES.md Step 3 records from the paper "all animals started training ... and continued training for exactly 5 days", which is the source of the 1.0/5.0 anchors. Trajectory step 274 notes "Day of training is constant at 1.0 in both sample sessions because both selected sessions are 'before_learning'; that's not ideal ... the decoder input `day_of_training` will be degenerate in sample mode" — the degeneracy was observed but the derivation was not revisited.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The scalar from `infer_day()` is broadcast across every bin of every trial of the session. Across the full dataset it takes only the values {1, 5} (verification reports `day_of_training: [1.0, 5.0]`). Consequences: it is a coarse experimental-phase label, not a day count; all sessions drawn from the same behavior file share a value regardless of mouse or date; which value a session gets depends on which duplicate copy the richness heuristic selected; and `naive_test1/2/3` numbering is a test-block index, not a day.

ii.
```python
day_arr = np.full(idx.size, day, dtype=np.float32)
inp = np.vstack([time_to_sound, day_arr, time_since, reward_arr]).astype(np.float32)
```

iii. As in 4-a; no further justification is written.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From the trial's own frame count and the estimated frame period: `arange(n_frames_in_trial) * dt_frame`, where `dt_frame` derives from `Trial_start_time` / `Trial_end_time` / `ft_trInd` (see 2-e). The frame timestamps `ft` and the fractional corridor-entry frame `StartFr` — both available and both used by the reference — are not used.

ii.
```python
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
```

iii. Trajectory step 238: the AI decided to "derive time-varying inputs on the frame grid" once it found the `ft_*` arrays; it used the frame *index* as the grid but reconstructed time from a single median period rather than reading `ft`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Exactly zero at the first imaged frame of the trial and increasing linearly by `dt_frame`; always non-negative (verification: `time_since_trial_start: [0.0, 1769.4]`). The zero point is the first frame labelled with the trial rather than the interpolated `StartFr`, a ~0.15 s offset; accumulated clock drift by the end of a trial is typically < 0.05 s (checked session).

ii.
```python
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
inp = np.vstack([time_to_sound, day_arr, time_since, reward_arr]).astype(np.float32)
```

iii. Same as 5-a.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is constructed with `idx.size` samples from the same trial window `idx` used for `spk[:, idx]`, so it is bin-for-bin aligned with the neural matrix.

ii.
```python
tr_spk = spk[:, idx].astype(np.float32, copy=False)
time_since = np.arange(idx.size, dtype=np.float32) * dt_frame
```

iii. Trajectory step 238 — all streams on the imaging-frame grid.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From the per-trial boolean `isRew`, the same variable the reference uses.

ii.
```python
is_rew = np.asarray(beh['isRew']).astype(np.int64)
```

iii. CONVERSION_NOTES.md Step 4: "Reward/sound structure | Code uses `isRew`, `WallName`, `UniqWalls`, lick variables | Behavior data contain these exact fields | Methods describe rewarded vs unrewarded corridors ... | Use behavior fields directly; these are consistent across code, data, and methods."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast to int, then broadcast as a constant float32 row across all bins of the trial. Verification confirms a 0/1 range overall, and per-session ranges show entire sessions constant at 0 (the unsupervised/naive mice), as expected.

ii.
```python
reward_arr = np.full(idx.size, int(is_rew[tr]), dtype=np.float32)
inp = np.vstack([time_to_sound, day_arr, time_since, reward_arr]).astype(np.float32)
```

iii. As in 6-a. Trajectory step 262 also records the AI checking that the sample sessions were degenerate (`reward_availability` constant 0) and re-choosing the sample sessions so the flag varies.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From the per-trial string array `WallName`. (`ft_WallID` is loaded and truncated but never used; `TrialStim`, `WallType`, `stim_id` are not used.) This matches the reference's choice of `WallName`.

ii.
```python
wall = np.asarray(beh['WallName'])
...
cache.append((tr_spk, inp, str(wall[tr]), ...))
```

iii. CONVERSION_NOTES.md Step 1: "Stimulus/category structure uses fields such as `WallType`, `WallName`, `UniqWalls`, and reward/category mapping via `get_cat_id(...)`"; Step 4 resolves to "Use behavior fields directly".

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The raw wall-name strings are used as the categories with **no collapsing to base texture**. A global vocabulary is built by sorting the union of `WallName` over all converted sessions, and each trial's label is the index into that vocabulary, broadcast over all bins. On the full dataset this yields **13 classes**: `circle1 (0.284), circle2 (0.059), circle3 (0.010), leaf1 (0.298), leaf1_swap1 (0.002), leaf1_swap2 (0.003), leaf2 (0.122), leaf3 (0.041), rock1 (0.071), rock2 (0.014), wood1 (0.056), wood2 (0.029), wood5 (0.013)`. The reference maps the 15 names onto 4 base textures (`circle`, `leaf`, `rock`, `wood`), which is what "Visual stimulus category. e.g. circle, leaf, etc." asks for. Several of the AI's classes hold < 1.5% of the data, and individual sessions contain disjoint subsets of the vocabulary.

ii.
```python
wall_names = sorted({str(w) for s in common for w in np.asarray(beh_sessions[s]['WallName'])})
wall_to_id = {w: i for i, w in enumerate(wall_names)}
...
'output_values': [wall_names, ['no_lick', 'lick'], ['bin1','bin2','bin3','bin4'], ['q1','q2','q3','q4']],
...
for wall_name, out in output_trials:
    out = out.copy()
    out[0, :] = wall_to_id[wall_name]
```

iii. No written justification. Trajectory step 274 notes only the structural consequence — "Visual stimulus categories span 6 classes overall, though each of the two sample sessions covers disjoint subsets ... which is acceptable structurally but may affect decoder training" — and moves on.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickTrind` (trial index of each lick) and `LickTime` (datenum of each lick), together with `Trial_start_time` and the estimated `dt_frame`. The directly frame-indexed variable `LickFr` — which the reference uses and which needs no reconstruction — is present in the same dict but is not used.

ii.
```python
lick_tr   = np.asarray(beh.get('LickTrind', []))
lick_time = np.asarray(beh.get('LickTime', []), dtype=float)
```

iii. CONVERSION_NOTES.md Step 1: "Behavioral processing is trial-based and uses lick positions (`LickPos`) and lick trial indices (`LickTrind`) to reconstruct per-trial lick rasters" — the AI copied the reference *figure-code* convention (which works in position/trial space) rather than the frame-indexed variable needed for a frame-aligned decoder.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, lick times are converted to seconds after `Trial_start_time`, divided by `dt_frame`, floored to an integer bin, **clipped into `[0, idx.size-1]`**, de-duplicated, and used to set a binary flag in a session-length vector. Measured against the frame-exact reconstruction from `LickFr` on session `VR2_2021_04_06_1`: the reference marks 1289 lick frames, the AI marks 1291, but only 911 coincide — i.e. ~29% of the AI's lick frames are misplaced (typically by ±1 frame from the half-frame anchor offset and clock drift), and the clipping converts every lick falling after the trial's last frame into a spurious lick on that last frame. Full-dataset lick rate is 0.026.

ii.
```python
lick_series = np.zeros(n_frames, dtype=np.int64)
if lick_tr.size and lick_time.size:
    lick_tr_i = lick_tr.astype(int)
    for tr in np.unique(lick_tr_i):
        idx = trial_frame_idx.get(int(tr))
        if idx is None or idx.size == 0 or tr >= len(tstart):
            continue
        rel = (lick_time[lick_tr_i == tr] - tstart[int(tr)]) * 86400.0
        rel = rel[np.isfinite(rel)]
        if rel.size == 0:
            continue
        bins = np.clip(np.floor(rel / dt_frame).astype(int), 0, idx.size - 1)
        lick_series[idx[np.unique(bins)]] = 1
```

iii. No written justification for the clipping or for preferring `LickTime` over `LickFr`. The AI did sanity-check the *rate* (Step 7: "licking distribution no_lick 0.808, lick 0.192" on the sample) but never checked lick *placement* against the raw frame indices; Step 10's "Check 2: construct `np.allclose` sanity checks against raw files" was not performed for any stream.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick flag lives on the session frame grid and is sliced with the same trial window `idx` as the neural matrix, so shapes and bin correspondence are correct. The alignment *content* is nonetheless offset, because the mapping from lick time to frame goes through `Trial_start_time + k·dt_frame` rather than through the frame clock `ft` (or directly through `LickFr`), and because out-of-window licks are clipped onto the trial's last bin instead of being dropped.

ii.
```python
lick = lick_series[idx]
...
output_trials.append((wall_name, np.vstack([
    np.full(tr_spk.shape[1], -1, dtype=np.int64),
    lick,
    pos_bin,
    speed_bin,
])))
```

iii. Trajectory step 238: all streams on the frame grid.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_PosCum` (cumulative VR position per imaging frame, in decimetres) reduced modulo the session's `Corridor_Length`. I verified that `np.mod(ft_PosCum, 60.0)` reproduces `ft_Pos` exactly (`np.allclose` true, max diff 0.0), so the AI's source is numerically identical to the reference's `ft_Pos` — it just re-derives it.

ii.
```python
ft_pos = np.asarray(beh['ft_PosCum'], dtype=float)
...
corridor_len_arr = np.asarray(beh.get('Corridor_Length', 4.0))
corridor_len = float(corridor_len_arr.item()) if corridor_len_arr.shape == () else 4.0
...
pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
```

iii. Trajectory step 125/238: the AI found `ft_PosCum` listed in the reference `utils.py` and adopted it as the frame-aligned position source; `Corridor_Length` was read from the behavior dict rather than hard-coded.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Modulo-reduction to within-corridor position in decimetres, then linear binning into 4 bins over `[0, corridor_len)`. `Corridor_Length` is **60 dm** (40 dm textured corridor + 20 dm grey space), not 4 m, so the four bins are 15 dm = **1.5 m** wide and the last bin is entirely grey space. The value is stored as an int per bin and is time-varying.

ii.
```python
pos_bin = np.clip((pos / max(corridor_len, 1e-6) * 4).astype(int), 0, 3)
```

iii. No written justification; Step 5 (where the discretization was to be planned) is the empty template. Step 7 records the resulting distribution (`bin1 0.260, bin2 0.238, bin3 0.297, bin4 0.204`) and, because it looked roughly uniform, treated it as validated.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Equal-width thresholds at 1/4, 2/4, 3/4 of `Corridor_Length`, i.e. at 15, 30 and 45 dm → bins `[0–1.5 m), [1.5–3 m), [3–4.5 m), [4.5–6 m]`. The specification asks for "4 equal-length, **1-m-long** spatial bins", i.e. boundaries at 10, 20, 30 dm over the 4 m textured corridor (the reference's `np.clip(ft_Pos // 10, 0, 3)`). The AI's bins are 1.5 m long, the grey space is included in the label space, and bin 3 mixes the 3–4 m texture with the first 0.5 m of grey space. The `output_values` are generic (`bin1..bin4`), so the mismatch is not visible in the metadata.

ii.
```python
pos_bin = np.clip((pos / max(corridor_len, 1e-6) * 4).astype(int), 0, 3)
...
'output_values': [wall_names, ['no_lick','lick'], ['bin1','bin2','bin3','bin4'], ['q1','q2','q3','q4']],
```

iii. No justification recorded.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_PosCum` is one sample per imaging frame, and the trial window `idx` used for the neural matrix is used unchanged, so the position row is bin-for-bin aligned with the neural data and has the same length.

ii.
```python
tr_spk = spk[:, idx].astype(np.float32, copy=False)
pos = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
```

iii. Trajectory step 238; also the explicit `n_frames = min(spk.shape[1], len(ft_tr), len(ft_pos), ...)` truncation keeps the two streams on a common index.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Not from the recorded running speed. The AI differentiates the VR position it built in 9-a: `speed = diff(pos, prepend=pos[0]) / dt_frame`. The behavior dict contains `ft_RunSpeed` (and `ft_RunCum`, `RunFr`, `ft_isMoving`), which is what the reference uses; `ft_RunSpeed` never appears anywhere in the AI's code, notes, or trajectory.

ii.
```python
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
```

iii. Trajectory step 115: "running speed in 4 quantile bins from position/time derivatives". The AI committed to the derivative at planning time and never revisited it after discovering the `ft_*` frame-aligned variables at step 238.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. First-difference of the modulo-reduced position divided by `dt_frame`, with the first bin of each trial forced to 0 by `prepend=pos[0]`. Checked against `ft_RunSpeed` on `VR2_2021_04_06_1`: the two agree only moderately (r = 0.76 restricted to non-negative samples, r = 0.15 including all samples), 24% of the derived samples are exactly 0 versus 3% of `ft_RunSpeed`, and the derived signal contains large spurious negatives (down to −188) from position wrap-arounds that occur inside ~0.4% of trial windows. It also measures VR translation rather than treadmill running, so it reads 0 whenever the VR is paused even if the animal is running.

ii.
```python
pos   = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
speed = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
...
all_speeds.append(speed)
```

iii. No justification beyond the step-115 plan; no comparison against `ft_RunSpeed` was ever made.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Per session, the 25/50/75% quantiles of the concatenation of all kept trials' speeds are computed and `np.digitize(..., right=False)` assigns each sample to one of 4 bins. The intent (four bins each holding 25% of the data) matches the specification, but the implementation does not achieve it: `np.digitize` cannot split the large tie mass at speed 0, so in sessions where > 25% of samples are exactly 0 the first bin collapses. The AI's own full-dataset verification shows `running_speed_bin: {q1 (0.040), q2 (0.295), q3 (0.415), q4 (0.250)}` — 4%/30%/42%/25% rather than 25% each (sample data were even worse: 0.2%/21.6%/53.3%/25.0%). The reference handles this explicitly with a rank-based split.

ii.
```python
speed_all = np.concatenate(all_speeds)
qs = np.quantile(speed_all, [0.25, 0.5, 0.75]) if speed_all.size else np.array([0, 0, 0], dtype=np.float32)
for tr_spk, inp, wall_name, lick, pos_bin, speed in cache:
    speed_bin = np.digitize(speed, qs, right=False).astype(np.int64)
```

iii. No justification. The skewed distribution was written into CONVERSION_NOTES.md Step 7 (`q1 0.002, q2 0.216, q3 0.533, q4 0.250`) and into the full verification log, but was never flagged as inconsistent with "4 bins, each corresponding to 25% of the data", and the corresponding low decoder accuracy for this output in the sample run (0.546 validation vs 0.748 training) was not investigated.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is computed from the trial's own position samples on the trial window `idx`, so it has exactly the same number of bins as the neural matrix and is aligned bin-for-bin. The differencing means bin *k* reports the displacement between bin *k−1* and bin *k* (a half-bin lag), and bin 0 is forced to 0.

ii.
```python
tr_spk = spk[:, idx].astype(np.float32, copy=False)
pos    = np.mod(ft_pos[idx], corridor_len).astype(np.float32)
speed  = np.diff(pos, prepend=pos[0]) / max(dt_frame, 1e-6)
```

iii. Trajectory step 238 — all streams on the frame grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several defensive measures are present: (a) every stream is truncated to `n_frames = min(spk.shape[1], len(ft_trInd), len(ft_PosCum), len(ft_move), len(ft_WallID))`, matching the reference's `[:nfr]` trimming and handling the observed 2-frame behavior/imaging mismatch; (b) frames with `NaN` `ft_trInd` are dropped; (c) trials whose index exceeds the length of any per-trial array are skipped; (d) non-finite `SoundTime`/`Trial_start_time` yields a `NaN` cue row (no such trials exist in the files I checked); (e) non-finite lick times are dropped; (f) `Corridor_Length` falls back to 4.0 if not scalar; (g) `dt_frame` falls back to 0.1 s; (h) trials with < 2 frames and sessions with < 2 trials are skipped; (i) duplicate session keys across behavior files are resolved by a richness heuristic. Two handling choices are harmful rather than protective: licks outside the trial window are **clipped onto the last bin** instead of being discarded, and the 13 swap-keyed recordings are dropped silently. Two fallbacks are also latently wrong: a `Corridor_Length` fallback of 4.0 would be in the wrong unit (the data are decimetres, 60.0), and a `dt_frame` fallback of 0.1 s is three times the true frame period.

ii.
```python
n_frames = min(spk.shape[1], len(ft_tr), len(ft_pos), len(ft_move), len(ft_wall))
spk = spk[:, :n_frames]; ft_tr = ft_tr[:n_frames]; ft_pos = ft_pos[:n_frames]
...
valid_frame = np.isfinite(ft_tr)
ft_tr_int = np.full(n_frames, -1, dtype=int)
ft_tr_int[valid_frame] = ft_tr[valid_frame].astype(int)
...
if tr >= len(wall) or tr >= len(is_rew) or tr >= len(sound) or tr >= len(tstart):
    continue
...
bins = np.clip(np.floor(rel / dt_frame).astype(int), 0, idx.size - 1)
```

iii. Trajectory step 238: "the mismatch of 2 samples suggests an off-by-few frame issue that the reference code handles via `nfr` trimming (`[:nfr]`). This is exactly the kind of alignment logic we need." CONVERSION_NOTES.md Step 10 documents the duplicate-session fix in detail. Nothing is written about the clipping or the fallbacks.

## 12-a. What are the most time-consuming steps of the code?

i. The dominant cost is I/O on the neural data: `np.load` + `np.concatenate` of the per-plane `spks` arrays (the spike directory is ~405 GB, and `concat_spks` additionally forces a float32 copy of every session), followed by `pickle.dump` of the resulting 372 GB dictionary. The full run took 2177 s (~36 min) over 76 sessions, ~29 s/session, which is essentially disk throughput. Secondary, but not negligible, are `load_behavior_sessions`, which loads all ~5 GB of behavior files up front and keeps every session dict alive for the whole run, and the O(n_trials × n_frames) `frame_periods` loop. The AI did not instrument per-step timing; it only printed a total elapsed time. Its own conclusion in Step 6 was that the bottleneck was "repeated frame-index scans", which is not where the time goes; the real cost was only recognised indirectly in Step 11 when the 372 GB pickle made decoder training infeasible.

ii.
```python
def concat_spks(spks_list):
    return np.concatenate([np.asarray(x, dtype=np.float32) for x in spks_list], axis=0)
...
nobj = np.load(neural_path, allow_pickle=True).item()
spk = concat_spks(nobj['spks'])
...
with open(args.outpicklefile, 'wb') as f:
    pickle.dump(data, f)
print(f'Wrote {args.outpicklefile} with {kept} sessions in {time.time()-t0:.2f}s')
```

iii. CONVERSION_NOTES.md Step 6: "Optimized trial indexing by precomputing frame indices per trial instead of repeated `np.where` scans" / "cutting sample conversion runtime from ~160 s to ~83 s for 2 sessions". Step 7: "~41 s/session after trial-index optimization | Rough full conversion time ~52 minutes for 76 sessions."

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Three remain. (1) The `frame_periods` loop calls `np.sum(ft_tr_int == tr)` once per trial, a full scan of the frame array per trial — exactly the O(n_trials × n_frames) pattern the AI claims to have eliminated; the per-trial counts are already available for free from `trial_frame_idx` (or one `np.bincount`). (2) The lick loop iterates over unique trials and rebuilds the boolean mask `lick_tr_i == tr` each time, again a full scan per trial; a single vectorised map from lick → frame would do (and is unnecessary altogether if `LickFr` is used). (3) The main per-trial Python loop builds `np.vstack`/`np.full` arrays trial by trial; this is inherent to the ragged output format and is minor. Set against the ~29 s/session I/O cost, none of these dominate — but (1) and (2) are pure waste.

ii.
```python
for tr in range(min(ntrials, len(trial_dur_sec))):
    nfr = np.sum(ft_tr_int == tr)          # full scan per trial
    ...
for tr in np.unique(lick_tr_i):
    ...
    rel = (lick_time[lick_tr_i == tr] - tstart[int(tr)]) * 86400.0   # full scan per trial
```

iii. CONVERSION_NOTES.md Step 6 claims the scan-per-trial pattern was removed ("Precomputed `trial_frame_idx` mapping once per session, reducing repeated frame-index scans"); the two loops above show it was removed only from the trial-segmentation path.

## 12-c. What processing does the code repeat multiple times?

i. (1) Per-trial frame counting is done twice — once by `np.sum(ft_tr_int == tr)` in the `dt_frame` estimate and once by `trial_frame_idx`. (2) Every trial's outputs are built once in `build_session` with a placeholder `-1` stimulus row and then **copied again** in `main` (`out = out.copy()`) purely to overwrite row 0 with the global wall id — a full duplication of every output array. (3) The per-trial speed vectors are stored twice, once in `cache` and once in `all_speeds`. (4) `_behavior_richness` is recomputed for the incumbent session dict on every duplicate key encountered. (5) `np.mod` is applied to `ft_PosCum` per trial rather than once per session. None of these is expensive relative to I/O, but (2) doubles peak memory for the output arrays and exists only because the stimulus vocabulary is built globally instead of being known before the session loop.

ii.
```python
cache.append((tr_spk, inp, str(wall[tr]), lick.astype(np.int64), pos_bin.astype(np.int64), speed.astype(np.float32)))
all_speeds.append(speed)
```
```python
for wall_name, out in output_trials:
    out = out.copy()
    out[0, :] = wall_to_id[wall_name]
    sess_out.append(out)
```

iii. Not discussed in CONVERSION_NOTES.md.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) `ft_move` and `ft_WallID` are read, coerced and truncated every session and then never used. (2) Neural traces are stored as **float32** where float16 suffices for deconvolved traces, doubling the largest component of the output. (3) All ~53k ROIs per session are kept, including the ~12% of neurons that fall outside the four visual areas and that the reference discards — with the side effect that `brain_region_idx` carries no information. (4) The ~27% of frames in the grey space are converted, binned and stored although they lie outside the 4 m corridor the position output is defined over. (5) Multi-minute "parked animal" trials are converted in full. Together (2)–(5) are why the pickle is 372 GB; the AI's own Step 11 records that this made full decoder training impossible, so in effect the whole conversion output was discarded downstream. (6) `all_speeds` and the `out.copy()` duplication (12-c). (7) The `--show-processing` flag is parsed but implemented nowhere, so the required per-step validation plots were never produced (CONVERSION_NOTES.md Step 7 admits "`--show-processing` not yet implemented"), and `README.md` was never created.

ii.
```python
ft_move = np.asarray(beh['ft_move'], dtype=float)
ft_wall = np.asarray(beh['ft_WallID'])
...
ft_move = ft_move[:n_frames]
ft_wall = ft_wall[:n_frames]          # neither is used again
```
```python
tr_spk = spk[:, idx].astype(np.float32, copy=False)
...
data['brain_region_idx'].append(np.zeros(neural_trials[0].shape[0], dtype=np.int64))
```
```python
ap.add_argument('--show-processing', action='store_true', default=False)   # never read
```

iii. CONVERSION_NOTES.md Step 11: "`train_decoder_full_out.txt` remained empty before manual interruption, indicating that loading/training on the 344G pickle is impractically slow in the current storage format. Additional optimization/compression of the converted dataset would be required to make full decoder training practical." Steps 12 and 13 are `NOT STARTED`, so none of this was acted on.
