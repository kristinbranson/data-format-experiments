# Decisions

> **Scope note.** The AI's run terminated before any conversion code was written. `/app/convert_data.py`,
> `/app/converted_data.pkl`, `/app/sample_data.pkl`, `/app/README.md` and all conversion/verification/
> decoder logs do **not** exist. The only artefact the AI produced is `/app/CONVERSION_NOTES.md`, which
> records Steps 0–3 as COMPLETE, Step 4 as IN PROGRESS, and Steps 5–13 as NOT STARTED (still containing
> the untouched template placeholders).
>
> The trajectory (`/logs/agent/trajectory.json`, 3196 steps, terminus-2 / gpt-5.4) shows why: the agent
> repeatedly queued several `python3 - <<'PY' ... PY` heredocs inside a single batch of keystrokes, which
> left the shell inside an unterminated quoted construct. It recovered once, but at step 313 it entered a
> `>` continuation prompt for the last time and never escaped; 2864 of the 3196 steps consist of the agent
> observing the bare `>` prompt and waiting for an "external reset" that never came. The last substantive
> terminal output is at step 3181.
>
> Consequently, for most decision points below there is **no decision and no code**. Where the AI did
> reach a documented determination during Steps 1–4 exploration, it is reported, and the code snippets
> quoted are the ad-hoc exploratory one-liners/heredocs from the trajectory (there is no script file to
> quote from).

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. No loading routine was ever implemented. During exploration the AI established that the data live in
`data/spk/` (one `<session_id>_neural_data.npy` per session, a pickled dict with the single key `spks`)
and `data/beh/` (one `Beh_<exp_type>.npy` per experiment type, each a pickled dict keyed by session id,
with `_swap1`/`_swap2` variants as extra keys). Everything is read with
`np.load(..., allow_pickle=True).item()`. Its recorded Step-4 resolution was to treat the **89 spike
files as the canonical session list** and to map behavior onto them by base session id, handling `_swap`
variants separately. It verified that all 89 spk sessions appear among the behavior base-session keys.

Two sources the reference uses were never found: `beh/Imaging_Exp_info.npy` (the master index of every
recording, grouped by experiment type) and the whole `data/retinotopy/` directory (`*_trans.npz`, holding
`iarea`, the visual area of each neuron). Neither string appears anywhere in the trajectory. A third
directory, `data/process_data/` (empty), was likewise never mentioned.

ii. The loading calls that were actually executed (exploratory only):
```python
# neural
for p in sorted(Path('data/spk').glob('*_neural_data.npy')):
    spks = np.load(p, allow_pickle=True).item()['spks']

# behavior
for p in Path('data/beh').glob('*.npy'):
    d = np.load(p, allow_pickle=True).item()
    beh_sessions.update(d.keys())
```
The session-coverage check that produced the Step-4 resolution:
```python
spk_sessions = {p.name.replace('_neural_data.npy','') for p in Path('data/spk').glob('*_neural_data.npy')}
base_beh = {s.split('_swap')[0] for s in beh_sessions}
print('spk_sessions', len(spk_sessions))
print('spk_not_in_beh_sample', sorted(spk_sessions - base_beh)[:20])
```

iii. From CONVERSION_NOTES.md Step 4: "All 89 spk sessions are present in behavior base-session keys;
extra behavior keys come from swap variants and aggregate/group labels … Use neural session list as
canonical sessions and map behavior by base session ID, handling `_swap` variants separately." The
justification is that the neural recordings are the limiting resource, so they define the session set.

## 1-b. How are the data split into subjects?

i. Subjects are the mouse name, taken as the first underscore-delimited token of the session id
(`<mname>_<YYYY>_<MM>_<DD>_<blk>`). The AI computed 19 unique subjects over the 89 spike files and
recorded the per-mouse session counts (DR10:6, DR15:5, LZ13:4, LZ16:4, TX104:2, TX105:5, TX108:7,
TX109:6, TX119:8, TX123:8, TX124:3, TX139:2, TX140:1, TX60:5, TX61:5, TX83:3, TX85:2, TX88:6, VR2:7),
which it cross-checked against the paper's "We performed 89 recordings in 19 mice".

ii.
```python
spk = sorted(Path('data/spk').glob('*_neural_data.npy'))
sessions = [p.name.replace('_neural_data.npy','') for p in spk]
subjects = [s.split('_')[0] for s in sessions]
print('n_subjects', len(set(subjects)))
print('sessions_per_subject', dict(sorted(Counter(subjects).items())))
```

iii. The filename already encodes the mouse, so no split has to be derived; the count was validated
against the paper's reported 19 mice. No `subjects` / `subject_idx` field was ever built.

## 1-c. How are the data split into sessions?

i. A session is one spike file, i.e. one mouse on one date in one block (`mname_YYYY_MM_DD_blk`), giving
89 sessions. Behavior entries that repeat the same recording under several experiment types, and the
`_swap1`/`_swap2` behavior keys, are reduced to the base session id. The AI noted 144 behavior entries
against 89 neural sessions and resolved this by taking the neural list as canonical.

ii.
```python
sessions = [p.name.replace('_neural_data.npy','') for p in spk]      # 89
base_beh = {s.split('_swap')[0] for s in beh_sessions}               # de-duplicated behavior keys
```

iii. Step-4 note: extra behavior keys "come from swap variants and aggregate/group labels", so the spike
files are the unambiguous session definition and match the paper's 89 recordings.

## 1-d. Are the data correctly split into trials?

i. **No decision was made.** Step 5 (Mapping Planning) was never started. The AI saw the behavior field
list in a terminal dump — including `ntrials`, `trInd`, `ft_trInd`, `ft_CorrSpc`, `ft_GraySpc`, `StartFr`,
`GrayFr`, `EndFr`, `RewardFr` — and recorded in Step 2 that "Trials are defined by corridor traversals
with trial-level timing variables available in behavior dicts", but it never decided which frames belong
to a trial, where a trial starts or ends, or whether trials are fixed- or variable-length. The only
trial-related quantity it computed was `ntrials` per behavior entry (min 84, mean 440.85, max 789).

ii.
```python
all_beh[(p.name, k)] = int(v['ntrials']) if isinstance(v, dict) and 'ntrials' in v else None
```
No frame-to-trial assignment code exists.

iii. CONVERSION_NOTES.md Step 3: "exact trial filtering still to be extracted." No further justification
was ever written.

## 1-e. How are trials filtered based on quality controls?

i. **No decision was made.** No trial-quality criterion was chosen, no trial-length distribution was
examined, and no trial was ever dropped. The notes state only that trial curation rules were "still to be
extracted".

ii. None.

iii. None.

## 2-a. What variables in the raw data is the final `neural` data derived from?

i. The AI identified `spks` inside `spk/<session_id>_neural_data.npy` as the neural source. It found that
`spks` is a Python list of **3** float32 arrays, each of shape (n, t) with n ≈ 11k–23k and t ≈ 17k–32k,
identical within a session. It concluded that the three arrays are **three simultaneously imaged visual
areas**, and that the session's neuron count is the sum of the three first dimensions (e.g. 3 × 19,408 =
58,224 for `DR10_2022_07_12_1`), which it checked against the paper's "20,547 to 89,577 neurons in each
recording". It never located `retinotopy/*_trans.npz` or the `iarea` codes, so no source for
`brain_region_idx` was identified.

ii.
```python
p = sorted(Path('data/spk').glob('*_neural_data.npy'))[0]
spks = np.load(p, allow_pickle=True).item()['spks']
print('SPKS_LEN', len(spks))                       # 3
for i, x in enumerate(spks[:5]):
    print('IDX', i, 'SHAPE', getattr(x, 'shape', None), 'DTYPE', getattr(x, 'dtype', None))
# IDX 0 SHAPE (19408, 31707) DTYPE float32  ... x3
```

iii. Step-4 discrepancy table: "Interpret `spks` as three area/plane-specific neuron-by-time arrays; sum
of first dimensions per session matches paper neuron-count range." The justification is purely the
agreement between the summed neuron count and the paper's reported per-recording range.

## 2-b. How is the `neural` data processed?

i. The only determination is that no ΔF/F or deconvolution step is needed: the methods say "All our
analyses were based on deconvolved fluorescence traces", obtained from Suite2p with a 0.75 s decay
timescale. Earlier the AI had wrongly written that the data are "spike-based data rather than calcium
imaging" (Step 1 note) and later corrected this in Step 3/4. Nothing else was decided — no per-trial
slicing, no concatenation across planes in code, no dtype choice, no normalisation, no padding policy.

ii. No processing code exists. The only related edit is a note rewrite:
```python
text = text.replace(
    'No obvious ΔF/F computation or spike-sorting quality filtering identified yet ...',
    'Inspection of `data/` shows session-wise spike/neural `.npy` files under `data/spk/`, so this is '
    'spike-based data rather than calcium imaging; ΔF/F computation is likely not applicable. ...')
```

iii. From CONVERSION_NOTES.md Step 3: "Paper methods state analyses used deconvolved fluorescence traces
processed with Suite2p, not raw calcium traces." So the traces are to be used as they are.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No decision was made.** The Step-3 "Neuron curation rules" entry reads: "Paper methods mention
Suite2p processing, ROI detection, cell classification, neuropil correction and spike deconvolution;
exact inclusion/exclusion criteria still to be extracted." The AI never found the retinotopy files, so
the visual-area membership filter used by the reference (keep neurons whose `iarea` is in V1/mHV/lHV/aHV)
was never considered.

ii. None.

iii. None.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. **No decision was made.** The instructions state the alignment event is trial start (corridor entry),
and the AI never chose how to realise it. The only alignment work done was a session-level sanity check:
for `TX108_2023_01_05_2` the neural arrays are (22851, 23193) while behavior `ft` has length 23194, a
one-frame difference, from which it concluded that neural arrays are on the behavior frame grid and that
"careful trimming to common length per session/trial" would be needed. No trial window, no `off_start` /
`off_end`, and no fixed-vs-variable-length choice was made.

ii.
```python
sess = 'TX108_2023_01_05_2'
spk = np.load(Path('data/spk')/(sess + '_neural_data.npy'), allow_pickle=True).item()['spks']
beh = np.load(behfile, allow_pickle=True).item()[sess]
print('neural shapes', [a.shape for a in spk])   # [(22851, 23193)] x3
print('behavior ft len', beh['ft'].shape)        # (23194,)
```

iii. Step-4 table: "Neural arrays are aligned to behavior frame stream with at most a 1-frame offset;
likely need careful trimming to common length per session/trial."

## 2-e. How is the `neural` data temporally binned/resampled?

i. **No decision was made.** The Step-3 statistics table records the neural time bin as "frame-based
deconvolved traces (exact dt pending)" and leaves "Behavior data time bin" blank. The imaging frame rate
was never determined, so `time_bin_size` was never fixed and no rebinning policy exists.

ii. None.

iii. None.

## 3-a. What variables in the raw data is `input` *time_to_sound_cue* derived from?

i. **No decision was made.** The AI saw `SoundPos`, `SoundTime`, `SoundTimeDelay` and `SoundFr` in the
behavior field dump and listed `SoundTime`/`SoundTimeDelay` among "trial-level variables" in the Step-2
notes, but never selected a source for this input. Step 5 (which is where inputs would have been mapped)
was never started.

ii. None.

iii. From Step 3: "Imaging sessions: sound cue time/randomized position per trial, uniformly between
0.5 m and 3.5 m" — recorded as background, not as a mapping decision.

## 3-b. What processing is involved in computing `input` *time_to_sound_cue*?

i. **No decision was made.** No sign convention, no unit conversion (the timestamps are MATLAB datenums),
no interpolation of the fractional cue frame onto the frame-time axis.

ii. None.

iii. None.

## 3-c. How is `input` *time_to_sound_cue* aligned with the neural data?

i. **No decision was made.** Beyond the generic Step-4 observation that behavior and neural streams share
a frame grid up to one frame, no per-trial alignment was specified.

ii. None.

iii. None.

## 4-a. What variables in the raw data is `input` *day_of_training* derived from?

i. **No decision was made.** The AI read from the methods that behavior-only training ran 5 days (passive
reward day 1, active reward days 2–5) and recorded per-mouse session counts, but never decided that the
date field of the session id orders a mouse's sessions, and never derived a training-day index.

ii. None.

iii. None.

## 4-b. What processing is involved in computing `input` *day_of_training*?

i. **No decision was made.** No counting scheme (0-based per mouse vs. calendar day), and no broadcast to
the bins of a trial.

ii. None.

iii. None.

## 5-a. What variables in the raw data is `input` *time_since_trial_start* derived from?

i. **No decision was made.** `Trial_start_time`, `StartFr` and `ft` were all visible in the field dump and
`Trial_start_time` was listed among the trial-level variables in Step 2, but no source was selected.

ii. None.

iii. None.

## 5-b. What processing is involved in computing `input` *time_since_trial_start*?

i. **No decision was made.** No sign convention, no datenum→seconds conversion, no interpolation of the
fractional `StartFr`.

ii. None.

iii. None.

## 5-c. How is `input` *time_since_trial_start* aligned with the neural data?

i. **No decision was made.**

ii. None.

iii. None.

## 6-a. What variables in the raw data is `input` *reward_availability* derived from?

i. **No decision was made.** `isRew` (bool, per trial) was enumerated in the Step-2 list of trial-level
behavior variables, and the Step-3 notes record that reward is delivered only in the rewarded corridor
after the sound cue, and that unsupervised sessions include the cue but no reward. No mapping from
`isRew` (or `RewTime`/`RewPos`/`RewardFr`/`Reward_Mode`) to the decoder input was ever chosen.

ii. None.

iii. From Step 3: "In rewarded corridors for task mice, sound cue indicated beginning of reward zone;
reward delivered after lick detected after cue. Unsupervised imaging sessions still included sound cue
for consistency, without reward."

## 6-b. What processing is involved in computing `input` *reward_availability*?

i. **No decision was made** (no cast to int, no per-trial broadcast).

ii. None.

iii. None.

## 7-a. What variables in the raw data is `output` *visual_stimulus* derived from?

i. **No decision was made.** The AI observed `WallName` (`<U7`, per trial), `WallType`, `WallIsProbe`,
`UniqWalls` (4 entries), `ft_WallID` (per frame), `TrialStim`, `StimTrial` (dict of per-trial boolean
masks) and `StimFrame` (dict of per-frame masks), and noted in step 215 that the stimulus labels include
categories "like circle1/circle2/leaf1/leaf2" and that these are "highly relevant for decoder outputs".
It never chose between `WallName`, `TrialStim` and `StimTrial`.

ii. None.

iii. None.

## 7-b. What processing is involved in computing `output` *visual_stimulus*?

i. **No decision was made.** No mapping of the texture variants (circle1/2/3, leaf1/2/3, leaf1_swap*,
rock1/2, wood1/2/5, wood1_swap*) onto four base categories, and no per-trial broadcast.

ii. None.

iii. None.

## 8-a. What variables in the raw data is `output` *licking* derived from?

i. **No decision was made.** The AI saw `LickTrind`, `LickTime`, `LickPos`, `Lick_wallName` and `LickFr`
in the field dump (in the inspected naive session these lick arrays are length 0). It listed licking as a
behavioral variable analysed by the reference figure code, but never selected a source variable.

ii. None.

iii. None.

## 8-b. What processing is involved in computing `output` *licking*?

i. **No decision was made.** No binarisation of lick events into a per-frame flag, no handling of
fractional lick frame numbers.

ii. None.

iii. None.

## 8-c. How is `output` *licking* aligned with the neural data?

i. **No decision was made.**

ii. None.

iii. None.

## 9-a. What variables in the raw data is `output` *position* derived from?

i. **No decision was made.** `ft_Pos`, `ft_PosCum`, `VRpos`, `VRposCum`, `run_pos`, `Corridor_Length`,
`Texture_Length` and `Gray_Space_length` were all visible in the field dump and `ft_Pos` was listed among
the frame-level variables in Step 2, but no source was selected. The AI did record from the methods that
the corridor is 4 m of texture followed by 2 m of grey space.

ii. None.

iii. None.

## 9-b. What processing is involved in computing `output` *position*?

i. **No decision was made.** The units of `ft_Pos` (decimetres) were never established.

ii. None.

iii. None.

## 9-c. How is `output` *position* thresholded into categories?

i. **No decision was made.** The instructions ask for four equal 1-m bins; no binning was implemented.

ii. None.

iii. None.

## 9-d. How is `output` *position* aligned with the neural data?

i. **No decision was made.**

ii. None.

iii. None.

## 10-a. What variables in the raw data is `output` *running_speed* derived from?

i. **No decision was made.** `ft_RunSpeed`, `ft_RunCum`, `ft_move` and `ft_isMoving` were enumerated in
the Step-2 notes as available frame-level variables; none was selected. The AI did record from the
methods that the VR moves at 60 cm/s when the mouse runs above a 6 cm/s threshold and that "only running
timepoints used for analysis" in the paper.

ii. None.

iii. None.

## 10-b. What processing is involved in computing `output` *running_speed*?

i. **No decision was made.**

ii. None.

iii. None.

## 10-c. How is `output` *running_speed* thresholded into categories?

i. **No decision was made.** The instructions ask for four bins each holding 25% of the data; no
quantile/rank scheme was chosen, and the large mass of exactly-zero speeds (which forces a rank-based
split) was never noticed.

ii. None.

iii. None.

## 10-d. How is `output` *running_speed* aligned with the neural data?

i. **No decision was made.**

ii. None.

iii. None.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. One issue was found and one policy sketched, but nothing was implemented. The AI compared neural and
behavior lengths for `TX108_2023_01_05_2` — neural (22851, 23193) versus behavior `ft` (23194,) — and
concluded that the streams share a frame grid with "at most a 1-frame offset" and that it would need
"careful trimming to common length per session/trial". It did not decide which stream to trim to, and it
never examined trials with no imaged frames, licks recorded after the last imaged frame, sessions with
too few trials, or per-session load failures. A stray `TypeError: float() argument must be a string or a
real number, not 'dict'` raised by one of its own exploratory scripts was never diagnosed.

ii. The check that produced the observation:
```python
print('neural shapes', [a.shape for a in spk])   # [(22851, 23193)] x3
print('behavior ft len', beh['ft'].shape)        # (23194,)
```

iii. Step-4 table: "Neural arrays are aligned to behavior frame stream with at most a 1-frame offset;
likely need careful trimming to common length per session/trial."

## 12-a. What are the most time-consuming steps of the code?

i. No timing instrumentation, no profiling and no run-time estimate exist, because no conversion script
exists; Steps 6 and 7 (where the instructions require timing information and a full-run estimate) were
never started. Empirically, the dominant cost in the AI's own exploration was repeatedly calling
`np.load(..., allow_pickle=True)` on the multi-GB `*_neural_data.npy` files — including one loop over all
89 files that only needed `len(spks)` — but this was never recognised or recorded as a bottleneck.

ii. The repeated full-file reads that dominated the exploratory phase:
```python
for p in sorted(Path('data/spk').glob('*_neural_data.npy')):
    spks = np.load(p, allow_pickle=True).item()['spks']
    counts.append(len(spks))
```

iii. None recorded.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Not applicable — there is no conversion code to vectorize, and the Step-6 "Code inefficiencies
identified" / "Code speedups added" fields of CONVERSION_NOTES.md are still template placeholders.

ii. None.

iii. None.

## 12-c. What processing does the code repeat multiple times?

i. Not applicable to a deliverable, since no conversion code exists. In the exploratory work the same
spike files and the same behavior files were loaded from scratch many times over (shapes, list lengths
and session keys were each re-derived by a fresh full `np.load` pass), and CONVERSION_NOTES.md was
rewritten by a separate `read_text`/`replace`/`write_text` heredoc after nearly every command — the
practice that ultimately wedged the shell. None of this was documented by the AI.

ii. The repeated notes-rewrite pattern, executed dozens of times:
```python
p = Path('CONVERSION_NOTES.md')
text = p.read_text()
text = text.replace(old, new)
p.write_text(text)
```

iii. None recorded.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Not applicable — no conversion code exists, so nothing is computed and discarded. The AI produced no
analysis of this question (Steps 6–13 never started).

ii. None.

iii. None.
