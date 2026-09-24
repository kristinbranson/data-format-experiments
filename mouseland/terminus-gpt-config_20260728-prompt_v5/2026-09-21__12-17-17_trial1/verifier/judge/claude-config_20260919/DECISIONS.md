# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI never reads the master index `beh/Imaging_Exp_info.npy`. Instead it (1) globs every `beh/Beh_*.npy`, loads each into memory up front, and builds a `session_id -> [experiment groups]` map from the dict keys; (2) globs `spk/*_neural_data.npy` to get the list of 89 recordings; (3) keeps only the spike sessions whose id appears verbatim as a behavior key (`matched`). Per session it concatenates the per-plane `spks` arrays along axis 0, and reads the behavior entry from a single preferred group. Retinotopy files are never opened.

Consequence: 13 of the 89 recordings are silently dropped, because in the behavior files those sessions are keyed with a stimulus-type suffix (`DR10_2022_07_30_1_swap1`, `TX108_2023_04_07_1_swap2`, …) rather than the bare session id. The conversion therefore covers 76/89 sessions (85%). Only the count `matched sessions: 76` is printed; no warning names the missing recordings.

ii.
```python
def load_behavior_maps():
    beh_maps = {}
    session_to_group = defaultdict(list)
    for f in sorted(BEH_DIR.glob('Beh_*.npy')):
        group = f.stem.replace('Beh_', '')
        d = np.load(f, allow_pickle=True).item()
        beh_maps[group] = d
        for sess in d:
            session_to_group[sess].append(group)
    return beh_maps, dict(session_to_group)


def load_spk_session(session_id):
    obj = np.load(SPK_DIR / f'{session_id}_neural_data.npy', allow_pickle=True).item()
    return np.concatenate(list(obj['spks']), axis=0).astype(np.float32)
```
```python
spk_sessions = sorted(p.stem.replace('_neural_data', '') for p in SPK_DIR.glob('*_neural_data.npy'))
matched = [s for s in spk_sessions if s in session_to_group]
```

iii. From the trajectory (steps 34–35): the plan was to "discover matched sessions between spk and behavior files" and use `/app/code`'s `load_spk`, which concatenates the three `spks` arrays along axis 0. CONVERSION_NOTES Step 4 records the (correct) conclusion that `spks` entries are neuron groups, not trials. No justification is given anywhere for excluding the 13 swap sessions; the AI appears never to have noticed the shortfall — CONVERSION_NOTES Step 2 states "89 per-session `*_neural_data.npy` files" and Step 9 (which was to check "no data was lost during conversion") was left as an empty template.

## 1-b. How are the data split into subjects?

i. The subject is the first underscore-delimited field of the session id (the mouse name). `subjects` is the sorted unique set over the converted sessions; `subject_idx` is the index of each session's mouse into that list. 19 subjects result, matching the paper.

ii.
```python
subjects = sorted({s.split('_')[0] for s in matched})
subj_to_idx = {s: i for i, s in enumerate(subjects)}
...
data['subject_idx'].append(subj_to_idx[sess.split('_')[0]])
```

iii. No explicit justification in CONVERSION_NOTES; implicit in the trajectory, where session ids such as `VR2_2021_04_11_1` were recognised early as `<mouse>_<date>_<block>` (Step 2 notes report "Subjects | 19").

## 1-c. How are the data split into sessions?

i. A session is one spike file, i.e. one mouse / one date / one block. Where a recording appears in more than one behavior group, a single group is chosen deterministically by a preference sort that favours group names containing `sup`/`unsup` over `naive`/grating groups; the trials and frame streams all come from that one behavior entry. Sessions surviving with fewer than 2 valid trials are skipped with a printed `skip` line.

ii.
```python
def find_behavior_for_session(session_id, beh_maps, session_to_group):
    groups = session_to_group.get(session_id, [])
    if not groups:
        return None, None
    # Prefer supervised/unsupervised test/train groups over naive if multiple
    pref = sorted(groups, key=lambda g: (('sup' not in g and 'unsup' not in g), g))
    g = pref[0]
    return g, beh_maps[g][session_id]
```
```python
        if len(valid) < 2:
            print('skip', sess, 'valid_trials', len(valid), 'shape', spk.shape)
            continue
```

iii. CONVERSION_NOTES Step 4 notes that behavior files are grouped by experiment name and that session ids appear inside those dicts, so "behavior dict membership/session metadata" is used to assign condition. The `sup`/`unsup` preference is not justified in the notes.

## 1-d. How are the data split into trials?

i. Trials are the trials the behavior declares (`ntrials`). The frames of trial *t* are the neural frames labelled with that trial **and** flagged as inside the textured corridor: `(ft_trInd == t) & ft_CorrSpc`. Both frame streams are first truncated to the number of imaged frames. The frames are taken as a contiguous index array; trials keep their own variable length (median ≈ 33 bins, min 11, max 5607).

ii.
```python
        ft_tr = np.asarray(beh['ft_trInd'])[:n_fr]
        ft_corr = np.asarray(beh['ft_CorrSpc'])[:n_fr].astype(bool)
        valid = []
        for tr in range(int(beh['ntrials'])):
            cols = np.flatnonzero((ft_tr == tr) & ft_corr)
            if cols.size < 2:
                continue
            valid.append((tr, cols))
```

iii. Trajectory step 53: the first implementation sliced trials with rounded `StartFr`/`EndFr`, which produced a position distribution that was 95.5% in the last bin; the AI then "patch[ed] to use `ft_trInd`/`ft_CorrSpc` segmentation", noting the reference code "work[s] with frame-level masks using `ft_trInd` … and corridor masks rather than slicing trials purely by rounded `StartFr`/`EndFr`" (step 46).

## 1-e. How are trials filtered based on quality controls?

i. The only trial filter is a minimum length of 2 corridor frames (`cols.size < 2`). There is no maximum-length filter, no stationarity check, and no other curation. Trials in which the animal entered the corridor and stopped are retained in full: the converted data contains a 5607-bin trial (≈ 29 min) and many trials of 600–1300 bins, against a median of ~33.

ii.
```python
            cols = np.flatnonzero((ft_tr == tr) & ft_corr)
            if cols.size < 2:
                continue
            valid.append((tr, cols))
```

iii. No justification is given. CONVERSION_NOTES Step 3 records the paper's statement that "only running timepoints were considered for analysis, removing periods when task mice stopped to collect water rewards" and that "exact trial exclusion criteria still need confirmation", but no such criterion was ever implemented, and Steps 9–12 (the review steps that would have surfaced the 5607-frame trial visible in `verification_full_out.txt`) were left as empty templates.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy`, a list of one (neurons × frames) array per imaging plane, concatenated along axis 0 into a single session matrix. The retinotopy files (`retinotopy/<mouse>_<date>_trans.npz`, key `iarea`) are **not** used.

ii.
```python
def load_spk_session(session_id):
    obj = np.load(SPK_DIR / f'{session_id}_neural_data.npy', allow_pickle=True).item()
    return np.concatenate(list(obj['spks']), axis=0).astype(np.float32)
```

iii. CONVERSION_NOTES Step 4: "Reference utility code concatenates `np.load(...)["spks"]` along axis 0, so the 3 list items are separate neuron groups that together form the full-session neural population, not separate trials." Step 5 Key Decision 1: "Use deconvolved traces, not raw fluorescence: Matches methods text and reference analyses."

## 2-b. How is the `neural` data processed?

i. No processing at all: the deconvolved traces are used as they are. The concatenated matrix is cast to `float32` (it is already `float32`, so this is a full-size copy), and each trial stores `spk[:, cols]`. Trials are variable length; nothing is padded, z-scored, smoothed or rebinned.

ii.
```python
            sess_neural.append(spk[:, cols].astype(np.float32))
```

iii. CONVERSION_NOTES Step 3: "Analyses are based on deconvolved fluorescence traces rather than raw fluorescence. Non-negative deconvolution used a decay timescale of 0.75 s." Metadata note in the script: "Neural traces are deconvolved activity; sessions reconstructed from full-session traces segmented by behavior StartFr/EndFr."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron filtering is applied — every Suite2p cell in the file is kept (24,383–89,577 per session; 4,047,169 total). `brain_regions` is set to three placeholder names and `brain_region_idx` labels each neuron by which imaging-plane array it came from, not by visual area. The `iarea` labels in the retinotopy files (which are exactly as long as the concatenated neuron axis) are never read.

ii.
```python
    brain_regions = ['unknown_group0', 'unknown_group1', 'unknown_group2']
```
```python
        # brain region idx: split concatenated neurons into 3 groups as placeholders
        raw = np.load(SPK_DIR / f'{sess}_neural_data.npy', allow_pickle=True).item()['spks']
        group_sizes = [arr.shape[0] for arr in raw]
        bri = np.concatenate([np.full(sz, i, dtype=np.int64) for i, sz in enumerate(group_sizes)])
```

iii. Trajectory step 13: "for TX123_2024_01_02, the neural trial shape is `(14367, 29251)` while retinotopy lengths are 43103, so retinotopy does not directly match the neural matrix dimensions for that session; perhaps retinotopy is from a broader ROI set or different preprocessing stage." CONVERSION_NOTES Step 4 repeats this: "Matching retinotopy file lengths do not equal example `spk` dimensions … Retinotopy files may refer to broader ROI sets or pre-filtered cell lists." The comparison was made against a *single plane* rather than the concatenation of all three (3 × 14,367 ≈ 43,103), so the premise is false. Step 34's plan explicitly allows "placeholder brain-region assignments if retinotopy mapping is unresolved", and the placeholder was never revisited.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry. Each trial's neural matrix is `spk[:, cols]`, where `cols` are that trial's corridor frames in ascending order, so column 0 is the first imaged frame inside the corridor. Trials keep their own length; nothing is cut or padded. `metadata['temporal_alignment_event'] = 'trial start / corridor entry'`, `off_start = 0.0`, `off_end = None`.

ii.
```python
        for tr, cols in valid:
            st, en = int(cols[0]), int(cols[-1])
            sess_neural.append(spk[:, cols].astype(np.float32))
            inp, out = build_trial_io(beh, tr, st, en, day_val, stim_to_idx, speed_edges, rewarded_wallnames)
            inp = inp[:, cols - st]
            out = out[:, cols - st]
```

iii. CONVERSION_NOTES Step 5 Key Decision 3: "Primary temporal alignment is corridor entry / trial start: Required by decoder task; cue-relative variables will be derived from behavior frame indices."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied: one column per imaging frame, i.e. ~315 ms bins (the frame times give 3.176 Hz). However, `metadata['time_bin_size']` is written as `1.0`, and the field is specified in milliseconds — off by a factor of ~315. All time-valued inputs are likewise stored in frame units, not seconds, and nothing in the metadata says so.

ii.
```python
        'metadata': {
            ...
            'time_bin_size': 1.0,
            'temporal_alignment_event': 'trial start / corridor entry',
```
```python
    time_to_cue = (sound_fr - frame_idx).astype(np.float32)
    time_since_start = np.arange(n_t, dtype=np.float32)
```

iii. CONVERSION_NOTES Step 1 notes "fig4 comments explicitly state recording frame rate is 3 Hz"; Step 3 says "Neural data time bin | based on imaging frame samples; exact frame rate not stated in methods excerpt". So the AI knew the frame rate was ~3 Hz but wrote `1.0` into the metadata; no justification is offered for the value.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr[trial]` (the fractional neural frame number of the cue) and the absolute session frame index of each bin.

ii.
```python
    frame_idx = np.arange(start_fr, end_fr + 1)
    sound_fr = int(round(float(beh['SoundFr'][trial_idx])))
    time_to_cue = (sound_fr - frame_idx).astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 mapping table: "Behavior field `SoundFr` / `SoundTime` -> input[0] time to sound cue: For each trial, compute continuous time-until-cue at each frame/time bin". The AI also observed that the reference code casts `SoundFr` to int (trajectory step 46).

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. `SoundFr` is rounded to the nearest whole frame and the bin's frame index is subtracted, giving a signed value in **frames** (positive before the cue, negative after). No conversion to seconds, and no interpolation onto the frame-time axis. Resulting range over the dataset: [−5601, 1281] frames.

ii.
```python
    sound_fr = int(round(float(beh['SoundFr'][trial_idx])))
    time_to_cue = (sound_fr - frame_idx).astype(np.float32)
```

iii. No explicit justification beyond the Step 5 mapping note ("likely `(SoundFr - frame_idx) * dt` or using times directly"); the `* dt` factor was never applied.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is built on absolute session frame indices spanning `cols[0] … cols[-1]`, then subset with `cols - st`, i.e. exactly the same frames used for that trial's neural columns. Same grid, same length.

ii.
```python
            st, en = int(cols[0]), int(cols[-1])
            inp, out = build_trial_io(beh, tr, st, en, ...)
            inp = inp[:, cols - st]
```

iii. Implicit: all behavior streams in this dataset are already on the imaging-frame grid (`ft_*` arrays and `*Fr` fields), as the AI recorded in CONVERSION_NOTES Step 4.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the date embedded in the session id. For each session, the sessions of the same mouse **that are present in the converted set** are sorted by (year, month, day) and the session's 1-based rank in that ordering is used.

ii.
```python
def infer_session_day(session_id, ordered_sessions):
    return float(ordered_sessions.index(session_id) + 1)
```
```python
        ordered_sessions = sorted([s for s in matched if s.split('_')[0] == sess.split('_')[0]],
                                  key=lambda x: tuple(x.split('_')[1:4]))
        day_val = infer_session_day(sess, ordered_sessions)
```

iii. Trajectory step 46: "derive `day_of_training` from chronological session order per subject within the selected set rather than digits in group names" — a fix after the first implementation parsed a day number out of the experiment-group name and produced a constant 1.0.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The rank (1…6) is broadcast as a constant across every bin of every trial of the session, as `float32`.

ii.
```python
    day_arr = np.full(n_t, day_val, dtype=np.float32)
    inp = np.vstack([time_to_cue, day_arr, time_since_start, reward_avail]).astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 Key Decision 4: "Represent per-trial variables as time-varying broadcasts when needed: Keeps input/output arrays shape-consistent for decoder." Note the AI itself flagged (trajectory step 58) that in `--sample` mode the rank is 1 for every session, calling this "acceptable for sample verification"; the same subset-relative behaviour also shifts the day index in the full run for the mice whose swap sessions were dropped.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From the trial's own corridor-frame index — i.e. from `ft_trInd`/`ft_CorrSpc` via `cols` — not from `StartFr` or `ft`. Time zero is the first imaged corridor frame of the trial.

ii.
```python
    n_t = end_fr - start_fr + 1
    time_since_start = np.arange(n_t, dtype=np.float32)
```
```python
            inp = inp[:, cols - st]
```

iii. CONVERSION_NOTES Step 5 mapping table: "Trial time index -> input[2] time since trial start: Continuous time-varying variable from corridor entry alignment … Alignment event is trial start / corridor entry."

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. A simple ramp `0, 1, 2, …` over the trial's span, in **frames**, subsequently subset to the corridor columns. It is therefore the frame offset from corridor entry; no seconds conversion, and no sub-frame `StartFr` offset. Range over the dataset: [0, 5606].

ii.
```python
    time_since_start = np.arange(n_t, dtype=np.float32)
```

iii. No justification recorded beyond the mapping note above.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Built on the same `start_fr … end_fr` span and subset by `cols - st`, so it is on exactly the frames of the trial's neural columns and has the same length.

ii.
```python
            inp = inp[:, cols - st]
            sess_input.append(inp)
```

iii. Same as 3-c: every stream is on the imaging-frame grid.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `WallName` together with `isRew`. Per session, the AI collects the set of wall textures on which any reward occurred (`WallName[isRew]`), then marks a trial 1 if its `WallName` is in that set. `isRew` is not used directly as the per-trial label. (`UniqWalls` is read but unused.)

ii.
```python
def infer_rewarded_wallnames(beh):
    wall = np.asarray(beh['WallName'])
    isrew = np.asarray(beh['isRew']).astype(bool)
    uniq = np.asarray(beh['UniqWalls'])
    rewarded = set(wall[isrew].tolist())
    if not rewarded:
        return set()
    return rewarded
```
```python
    wall_name = str(beh['WallName'][trial_idx])
    reward_avail = np.full(n_t, int(wall_name in rewarded_wallnames), dtype=np.float32)
```

iii. Trajectory step 45: the first implementation used `isRew` directly and produced `reward_availability` constant 0 in the (unsupervised) sample. The AI concluded (step 46) it should "define reward availability as whether the trial's `WallName` belongs to the rewarded corridor category inferred from `WallName` + `isRew` within the session, not simply `isRew`", citing that the reference code "use[s] `WallName`, `UniqWalls`, `stim_id`, and `isRew` together to identify rewarded stimulus categories".

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. A per-trial 0/1 broadcast across all bins of the trial. It equals `isRew` on 133 of 142 behavior entries; on 9 supervised entries it differs on the rewarded-corridor trials where the animal earned no reward (e.g. 34/423 in `TX108_2023_03_25_1`), which the AI's rule labels 1 and `isRew` labels 0.

ii.
```python
    reward_avail = np.full(n_t, int(wall_name in rewarded_wallnames), dtype=np.float32)
```

iii. As 6-a: the AI wanted a corridor-identity variable rather than a reward-delivery variable, matching the wording "1 if in rewarded corridor, 0 if not".

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `TrialStim`, with a fall-back to `WallName` only when `TrialStim` holds the literal placeholder string `'stimulus_of_trial'`.

ii.
```python
    stim_name = str(beh['TrialStim'][trial_idx])
    if stim_name == 'stimulus_of_trial':
        stim_name = str(beh['WallName'][trial_idx])
    stim_idx = stim_to_idx[stim_name]
    stim_arr = np.full(n_t, stim_idx, dtype=np.int64)
```

iii. CONVERSION_NOTES Step 5 mapping table: "Behavior field `TrialStim` (plus `WallName` / `WallType` if needed) -> output[0] visual stimulus category". Trajectory steps 306/373 describe a fix where the `'stimulus_of_trial'` placeholder was leaking in as a category: "The bogus `stimulus_of_trial` category is gone, replaced by meaningful categories". No check was ever made that `TrialStim` agrees with the texture actually displayed.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The raw label strings are **not** collapsed into the four base textures. A vocabulary is built by scanning every trial of the sessions being converted, sorted alphabetically, and each trial's label is stored as its index into that vocabulary, broadcast across the trial's bins. In the full run this gives 9 classes (`circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, wood5`); in the `--sample` run it gives 2 (`circle1, leaf1`), so sample and full pickles use different, incompatible encodings.

ii.
```python
    stim_names = sorted({(lambda beh,i: (str(beh['WallName'][i]) if str(beh['TrialStim'][i]) == 'stimulus_of_trial' else str(beh['TrialStim'][i])))(find_behavior_for_session(s, beh_maps, session_to_group)[1], i)
                         for s in matched
                         for i in range(int(find_behavior_for_session(s, beh_maps, session_to_group)[1]['ntrials']))})
    stim_to_idx = {s: i for i, s in enumerate(stim_names)}
```
```python
        'output_values': [
            stim_names,
            ['no_lick', 'lick'],
            ['0-1m', '1-2m', '2-3m', '3-4m'],
            ['q1', 'q2', 'q3', 'q4'],
        ],
```

iii. CONVERSION_NOTES Step 5: "Categories may include leaf/circle/rock/brick variants and probes". The AI treated each variant name as its own category; no rationale for keeping the variants separate is recorded, and no check of the resulting class distribution against the paper was performed (Step 9's consistency table is an empty template).

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (the fractional neural frame of each lick) and `LickTrind` (the trial each lick belongs to).

ii.
```python
def build_lick_vector(beh, start_fr, end_fr, trial_idx):
    n_t = end_fr - start_fr + 1
    lick = np.zeros(n_t, dtype=np.int64)
    lick_fr = np.asarray(beh['LickFr']).astype(int)
    lick_tr = np.asarray(beh['LickTrind']).astype(int)
    mask = lick_tr == int(trial_idx)
```

iii. CONVERSION_NOTES Step 5 mapping: "Lick fields `LickFr`, `LickTrind` -> output[1] licking: Binary time series per trial … Build framewise lick vector." Trajectory step 46 notes the reference code casts `LickTrind` and `LickFr` to int.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Per trial, licks belonging to that trial are truncated to whole frames, expressed relative to `start_fr`, clipped to the trial span, and the corresponding bins are set to 1; everything else is 0. (`np.unique` on the relative indices is a no-op for correctness.) Over the full dataset 3.4% of bins are licks; sessions with an empty `LickFr` (all unsupervised/naive recordings) are all-zero.

ii.
```python
    rel = lick_fr[mask] - int(start_fr)
    rel = rel[(rel >= 0) & (rel < n_t)]
    if rel.size:
        lick[np.unique(rel)] = 1
    return lick
```

iii. CONVERSION_NOTES Step 3: "Anticipatory licking before the sound cue inside the corridor is behaviorally important and should be represented." Trajectory step 58 checks the sample: "`licking` now varies with an overall 8.7% lick fraction; the supervised session shows licks while the unsupervised session does not, which is plausible."

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` is already an imaging-frame index, so the vector is on the neural grid by construction; it is built over `start_fr … end_fr` and then subset with `cols - st`, the same columns as the trial's neural matrix.

ii.
```python
    lick = build_lick_vector(beh, start_fr, end_fr, trial_idx)
    ...
            out = out[:, cols - st]
```

iii. Same as 3-c.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the per-imaging-frame corridor position (in decimetres), sliced to the trial span and then to the trial's corridor columns.

ii.
```python
    ft_pos = np.asarray(beh['ft_Pos'], dtype=float)
    pos_seg = ft_pos[start_fr:end_fr + 1]
```

iii. CONVERSION_NOTES Step 5 mapping: "Behavior fields `ft_Pos` or `VRpos` -> output[2] corridor position bin … Prefer frame-aligned `ft_Pos`; exclude gray-space frames as needed."

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The position is divided by `Corridor_Length / 4` and floored, then clipped to [0, 3]. `Corridor_Length` in these files is **60** decimetres (the 40 dm texture plus the 20 dm grey space), so the bin width is 15 dm = **1.5 m**, not the 1 m the task specifies. Because trials contain only textured-corridor frames (`ft_Pos < 40`), only bins 0, 1 and 2 are ever produced: the converted `corridor_position_bin` has 3 classes (0.409 / 0.344 / 0.247) and the declared fourth value `'3-4m'` is empty, while the labels `'0-1m'`, `'1-2m'`, `'2-3m'` mis-describe the 0–1.5 m, 1.5–3 m, 3–4 m ranges they actually hold.

ii.
```python
    corridor_len = float(beh.get('Corridor_Length', 40.0))
    bin_w = corridor_len / 4.0
    pos_bin = np.clip(np.floor(pos_seg / bin_w).astype(int), 0, 3)
```

iii. Trajectory step 53: "`ft_Pos` spans approximately 0 to 39, meaning position is in decimeters or similar scaled units across a 4 m corridor. This explains the previous position-bin bug: we had discretized by 1.0 units instead of 10-unit chunks." Step 58: "we should bin by quarters of `Corridor_Length` (likely 10-unit bins)". The AI assumed `Corridor_Length` was 40 (hence the `40.0` default) and never checked the value; the `--verify-only` output showing only three position classes was never reviewed, since Steps 9–12 of CONVERSION_NOTES were left blank.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Hard thresholds at 15, 30 and 45 dm (i.e. 1.5 m, 3 m, 4.5 m), from `floor(pos / 15)` clipped to [0, 3]. The task asked for 4 equal 1-m bins, i.e. thresholds at 10, 20 and 30 dm.

ii.
```python
    bin_w = corridor_len / 4.0
    pos_bin = np.clip(np.floor(pos_seg / bin_w).astype(int), 0, 3)
```
```python
            ['0-1m', '1-2m', '2-3m', '3-4m'],
```

iii. As 9-b: the intent recorded in the trajectory was "10-unit bins", i.e. 1 m; the implementation derives the width from `Corridor_Length`, which is 60.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is one value per imaging frame; it is sliced `[start_fr : end_fr+1]` and then indexed by `cols - st`, giving exactly the trial's neural columns.

ii.
```python
    pos_seg = ft_pos[start_fr:end_fr + 1]
    ...
            out = out[:, cols - st]
```

iii. Same as 3-c.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the per-imaging-frame running speed.

ii.
```python
    ft_speed = np.asarray(beh['ft_RunSpeed'], dtype=float)
    speed_seg = ft_speed[start_fr:end_fr + 1]
```

iii. CONVERSION_NOTES Step 5 mapping: "Behavior field `ft_RunSpeed` -> output[3] running speed bin: Discretize running speed into quartiles across dataset … Use frame-aligned speed values."

## 10-b. What processing is involved in computing `output` *Running speed*?

i. One global set of quartile edges is computed once over the whole converted dataset (not per session), pooling `ft_RunSpeed` from every session — but only over frames that are finite **and** inside the corridor **and** flagged `ft_isMoving`. Those edges are then applied to *all* frames, including the stationary ones excluded from the edge computation. Because a large fraction of frames sit at or near zero speed and were excluded from the quantile estimate, the resulting classes are far from equal: 0.549 / 0.152 / 0.150 / 0.149 over the full dataset, instead of the 25% each the task requires. (The lowest edge is also negative, −19.17.)

ii.
```python
def collect_speed_values(session_ids, beh_maps, session_to_group):
    vals = []
    for sess in session_ids:
        _, beh = find_behavior_for_session(sess, beh_maps, session_to_group)
        ...
        v = np.asarray(beh['ft_RunSpeed'], dtype=float)
        m = np.isfinite(v)
        if 'ft_CorrSpc' in beh:
            m &= np.asarray(beh['ft_CorrSpc'], dtype=bool)
        if 'ft_isMoving' in beh:
            m &= np.asarray(beh['ft_isMoving'], dtype=bool)
        vals.append(v[m])
    ...
    allv = np.concatenate(vals)
    qs = np.quantile(allv, [0, 0.25, 0.5, 0.75, 1.0])
    qs[-1] = np.nextafter(qs[-1], np.inf)
    return qs
```

iii. CONVERSION_NOTES Step 5 Key Decision 5: "Use behavior frame masks to define valid analysis periods: Methods explicitly state only running timepoints were analyzed; candidate masks include `ft_isMoving` and `ft_CorrSpc`." That is the stated reason for the `ft_isMoving` mask. No justification is given for applying edges derived from moving frames to all frames, and the resulting 55/15/15/15 split is never compared with the required 25% each (`verification_full_out.txt` prints it; Step 12's review table is an empty template).

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize` against the three interior global edges (12.15, 25.18, 40.56), right-open, then clipped to [0, 3]. Labels are `q1…q4`, implying quartiles, which they are not in the delivered data.

ii.
```python
    speed_bin = np.digitize(speed_seg, speed_edges[1:-1], right=False).astype(int)
    speed_bin = np.clip(speed_bin, 0, 3)
```
```python
            ['q1', 'q2', 'q3', 'q4'],
```

iii. As 10-b.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is one value per imaging frame; sliced `[start_fr : end_fr+1]` and indexed by `cols - st`, i.e. the trial's neural columns.

ii.
```python
    speed_seg = ft_speed[start_fr:end_fr + 1]
    ...
            out = out[:, cols - st]
```

iii. Same as 3-c.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled: (a) the behavior can be longer than the imaging, so `ft_trInd` and `ft_CorrSpc` are truncated to `spk.shape[1]` before trial segmentation, and the position/speed slices can only reach frames inside that range because they are indexed by `cols`; (b) licks outside the trial span are dropped; (c) non-finite speeds are excluded from the quantile estimate; (d) trials with fewer than 2 corridor frames are dropped, and sessions with fewer than 2 such trials are skipped with a printed message; (e) the `'stimulus_of_trial'` placeholder in `TrialStim` falls back to `WallName`; (f) `beh.get('Corridor_Length', 40.0)` guards a missing key.

Not handled: the 13 recordings whose behavior key carries a `_swapN` suffix are dropped with no message; `brain_regions` is hardcoded to 3 names while `brain_region_idx` is built from however many planes a file contains; the `Corridor_Length` default (40) differs from the value actually present in every file (60).

ii.
```python
        ft_tr = np.asarray(beh['ft_trInd'])[:n_fr]
        ft_corr = np.asarray(beh['ft_CorrSpc'])[:n_fr].astype(bool)
```
```python
    rel = rel[(rel >= 0) & (rel < n_t)]
```
```python
        if len(valid) < 2:
            print('skip', sess, 'valid_trials', len(valid), 'shape', spk.shape)
            continue
```

iii. Trajectory step 45 records finding, and then fixing, the constant `day_of_training`, constant `reward_availability`, all-zero `licking` and 95.5%-in-last-bin position problems. CONVERSION_NOTES Step 10 ("Check for edge cases", "There may be minor issues in the data that your code must handle") was never filled in.

## 12-a. What are the most time-consuming steps of the code?

i. Reading the spike files: the `spk/` tree is ~405 GB and every byte is read, and each session's file is read **twice** — once in `load_spk_session` and again a few lines later purely to recover the per-plane row counts. The `.astype(np.float32)` on an array that is already `float32` forces a second full-size copy in memory. Writing the result is also expensive: because no neurons are filtered and trials are stored as `float32`, `converted_data.pkl` is 253 GB. No timing instrumentation was added, despite the instructions asking for it.

ii.
```python
        spk = load_spk_session(sess)     # first full read of the file
        ...
        raw = np.load(SPK_DIR / f'{sess}_neural_data.npy', allow_pickle=True).item()['spks']   # second
        group_sizes = [arr.shape[0] for arr in raw]
```

iii. Nothing recorded. CONVERSION_NOTES Step 6 ("Code inefficiencies identified", "Code speedups added") and Step 7 ("Run Time Estimates") were left as empty templates.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. (1) The trial-segmentation loop rescans the whole frame index once per trial — `O(n_trials × n_frames)`, ~500 × 25,000 per session — where one pass grouping frames by `ft_trInd` would do. (2) `build_trial_io` is called once per trial and rebuilds per-trial what is constant per session: `np.asarray(beh['ft_Pos'], dtype=float)` and `np.asarray(beh['ft_RunSpeed'], dtype=float)` convert the full session arrays on every trial, and `build_lick_vector` re-materialises and re-casts the whole `LickFr`/`LickTrind` arrays on every trial. Position and speed binning could be done once per session over all frames and then indexed. (3) The stimulus-vocabulary comprehension calls `find_behavior_for_session` twice for every trial of every session.

ii.
```python
        for tr in range(int(beh['ntrials'])):
            cols = np.flatnonzero((ft_tr == tr) & ft_corr)
```
```python
    ft_pos = np.asarray(beh['ft_Pos'], dtype=float)
    ft_speed = np.asarray(beh['ft_RunSpeed'], dtype=float)
```

iii. Nothing recorded.

## 12-c. What processing does the code repeat multiple times?

i. (1) Each spike `.npy` is loaded twice per session (see 12-a). (2) The whole-session `ft_Pos` and `ft_RunSpeed` conversions, and the `LickFr`/`LickTrind` casts, are repeated once per trial. (3) `find_behavior_for_session` is re-run for every trial while building the stimulus vocabulary and again per session in `collect_speed_values` and in the main loop. (4) The per-trial input/output arrays are built over the full `start_fr … end_fr` span and only then subset to the corridor columns.

ii.
```python
        raw = np.load(SPK_DIR / f'{sess}_neural_data.npy', allow_pickle=True).item()['spks']
```
```python
    stim_names = sorted({(lambda beh,i: ...)(find_behavior_for_session(s, beh_maps, session_to_group)[1], i)
                         for s in matched
                         for i in range(int(find_behavior_for_session(s, beh_maps, session_to_group)[1]['ntrials']))})
```

iii. Nothing recorded.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) `.astype(np.float32)` on already-`float32` spikes: a full-size copy that changes nothing. (2) All ~4.05 M neurons are stored, including the ones outside the four visual areas that the reference discards (~12.5% of cells), and they are stored at `float32` rather than `float16`, so the pickle is 253 GB — roughly 24× the reference's. (3) Inputs and outputs are computed over the full `start_fr … end_fr` span and then thrown away outside the corridor columns. (4) `UniqWalls` is loaded in `infer_rewarded_wallnames` and never used; `safe_int_frames` is defined and never called; `Counter` is imported and never used. (5) The `--show-processing` flag is declared and parsed but never read anywhere in the script, so the required per-step verification plots were never produced.

ii.
```python
    return np.concatenate(list(obj['spks']), axis=0).astype(np.float32)
```
```python
    uniq = np.asarray(beh['UniqWalls'])      # never used


def safe_int_frames(x):                      # never called
    return np.asarray(np.round(x), dtype=int)
```
```python
    ap.add_argument('--show-processing', action='store_true', help='Save processing visualizations of up to 2 sessions')
```

iii. Nothing recorded; CONVERSION_NOTES Step 6 was left as an empty template and no `processing_<session_id>.png` files exist.
