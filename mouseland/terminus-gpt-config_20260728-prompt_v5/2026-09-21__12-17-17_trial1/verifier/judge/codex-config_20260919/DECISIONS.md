# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent eagerly loads every `Beh_*.npy` file into `beh_maps`, indexes its keys, enumerates every spike file, and retains spike session IDs that occur as exact behavior keys. It does not load `Imaging_Exp_info.npy` or retinotopy data.

ii. ```python
for f in sorted(BEH_DIR.glob('Beh_*.npy')):
    d = np.load(f, allow_pickle=True).item()
    beh_maps[group] = d
spk_sessions = sorted(p.stem.replace('_neural_data', '') for p in SPK_DIR.glob('*_neural_data.npy'))
matched = [s for s in spk_sessions if s in session_to_group]
```

iii. The notes say to use raw-ish `spk` and `beh` (and planned retinotopy) because processed files contain derived analyses. They report 89 spike sessions and behavior dictionaries as the relevant sources.

## 1-b. How are the data split into subjects?

i. Subject IDs are inferred as the substring before the first underscore in each matched session ID; sorted unique IDs form `subjects`, and each session receives the corresponding index.

ii. ```python
subjects = sorted({s.split('_')[0] for s in matched})
subj_to_idx = {s: i for i, s in enumerate(subjects)}
data['subject_idx'].append(subj_to_idx[sess.split('_')[0]])
```

iii. The notes identify 19 mice and session names such as `VR2_2021_04_11_1`, implicitly justifying parsing the mouse from the session ID.

## 1-c. How are the data split into sessions?

i. Each `*_neural_data.npy` filename defines a session. Only IDs appearing exactly as a behavior dictionary key are retained. If several behavior groups contain an ID, a supervised/unsupervised group is preferred lexically.

ii. ```python
groups = session_to_group.get(session_id, [])
pref = sorted(groups, key=lambda g: (('sup' not in g and 'unsup' not in g), g))
g = pref[0]
```

iii. The notes concluded that `spk` contains 89 per-session files and that behavior files are aggregate dictionaries keyed by session ID.

## 1-d. How are the data split into trials?

i. For each behavior trial index, frames are those whose `ft_trInd` equals the trial and whose `ft_CorrSpc` is true. The first and last such columns are passed to behavior construction, then all arrays are subselected back to the exact corridor-frame columns.

ii. ```python
cols = np.flatnonzero((ft_tr == tr) & ft_corr)
st, en = int(cols[0]), int(cols[-1])
sess_neural.append(spk[:, cols].astype(np.float32))
inp = inp[:, cols - st]
out = out[:, cols - st]
```

iii. The notes recognized that `spks` entries are neuron groups rather than trials and planned to reconstruct trials from `StartFr`, `EndFr`, `ft_trInd`, and corridor masks, aligned to corridor entry.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than two corridor frames are removed; sessions with fewer than two remaining trials are skipped. No long-trial/outlier filter is applied.

ii. ```python
if cols.size < 2:
    continue
if len(valid) < 2:
    continue
```

iii. The documentation mentions using behavior masks and the decoder's two-trial requirement, but gives no evidence-based justification for the two-frame cutoff and leaves curation investigation unfinished.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes solely from the `spks` list in each session's spike file; its arrays are concatenated on the neuron axis. Retinotopy `iarea` is not used.

ii. ```python
obj = np.load(SPK_DIR / f'{session_id}_neural_data.npy', allow_pickle=True).item()
return np.concatenate(list(obj['spks']), axis=0).astype(np.float32)
```

iii. The notes correctly identify `spks` as deconvolved traces split into neuron groups and say they should be concatenated. Retinotopy use remained unresolved.

## 2-b. How is the `neural` data processed?

i. Plane/group arrays are concatenated, converted to float32, and sliced to corridor frames per trial. There is no dF/F, deconvolution, padding, or rebinning.

ii. ```python
spk = load_spk_session(sess)
sess_neural.append(spk[:, cols].astype(np.float32))
```

iii. The agent states that the files already contain the deconvolved activity used by the paper, so no further fluorescence processing is needed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are filtered. The three source arrays are mislabeled as placeholder brain regions.

ii. ```python
brain_regions = ['unknown_group0', 'unknown_group1', 'unknown_group2']
bri = np.concatenate([np.full(sz, i, dtype=np.int64) for i, sz in enumerate(group_sizes)])
```

iii. Notes noticed retinotopy length/linkage uncertainty but never resolved it. The final code explicitly calls the resulting region assignments placeholders.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each variable-length neural trial contains the exact `ft_trInd`/`ft_CorrSpc` frame columns, beginning at its first corridor frame (corridor entry), with no padding.

ii. ```python
cols = np.flatnonzero((ft_tr == tr) & ft_corr)
sess_neural.append(spk[:, cols].astype(np.float32))
```

iii. The notes explicitly choose corridor entry/trial start because the decoder task requires it.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied, but metadata incorrectly declares each native imaging frame to be `1.0` ms. The numeric time variables likewise count frames rather than seconds.

ii. ```python
'time_bin_size': 1.0,
time_to_cue = (sound_fr - frame_idx).astype(np.float32)
time_since_start = np.arange(n_t, dtype=np.float32)
```

iii. Notes recognized an approximately 3 Hz imaging/behavior grid but the final implementation did not convert it to the reference's roughly 315 ms bins.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundFr` and the integer indices of the selected frames; `ft` timestamps are ignored.

ii. ```python
sound_fr = int(round(float(beh['SoundFr'][trial_idx])))
frame_idx = np.arange(start_fr, end_fr + 1)
```

iii. The plan identified `SoundFr`/`SoundTime` and proposed frame differences or direct times, but did not settle the units.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. `SoundFr` is rounded to an integer and each frame index is subtracted from it, producing signed frames (positive before cue), not seconds.

ii. ```python
time_to_cue = (sound_fr - frame_idx).astype(np.float32)
```

iii. The notes call for a continuous time-until-cue variable and acknowledge that a frame difference should be multiplied by `dt`; that multiplication was omitted.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Values are made over the enclosing first-to-last corridor-frame interval, then indexed by `cols - st`, exactly matching neural columns.

ii. ```python
inp, out = build_trial_io(beh, tr, st, en, ...)
inp = inp[:, cols - st]
```

iii. The stated plan was to use behavior frame indices for frame-aligned cue-relative variables.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is inferred from each mouse's matched session IDs; the embedded date fields determine their order.

ii. ```python
ordered_sessions = sorted([s for s in matched if s.split('_')[0] == sess.split('_')[0]], key=lambda x: tuple(x.split('_')[1:4]))
```

iii. Notes proposed session training day/experiment order because no direct per-frame field supplies it.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The session's one-based position in the mouse's chronologically sorted matched sessions is cast to float and broadcast over every trial frame.

ii. ```python
return float(ordered_sessions.index(session_id) + 1)
day_arr = np.full(n_t, day_val, dtype=np.float32)
```

iii. The notes planned a continuous per-trial value broadcast across time, but did not justify using one-based rather than the reference's zero-based count.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived only from the number/order of frames between the first and last selected corridor frame, rather than `StartFr` and `ft`.

ii. ```python
n_t = end_fr - start_fr + 1
time_since_start = np.arange(n_t, dtype=np.float32)
```

iii. The plan described a trial time index aligned to corridor entry but did not finish the units decision.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Consecutive integers starting at zero are generated and treated as time, so values are frame offsets rather than elapsed seconds.

ii. ```python
time_since_start = np.arange(n_t, dtype=np.float32)
```

iii. Notes call the result continuous and time-varying, but do not justify omitting conversion from frames to seconds.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It starts at zero at the first corridor frame and is subselected with the same relative indices as neural data.

ii. ```python
inp = inp[:, cols - st]
sess_neural.append(spk[:, cols].astype(np.float32))
```

iii. The alignment follows the agent's explicit corridor-entry decision.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Rewarded wall names are inferred by pairing `WallName` with truthy `isRew`; availability for a trial is then whether its wall name belongs to that inferred set.

ii. ```python
rewarded = set(wall[isrew].tolist())
reward_avail = np.full(n_t, int(wall_name in rewarded_wallnames), dtype=np.float32)
```

iii. The plan considered `isRew`, wall identity, and trial stimulus and interpreted the target as rewarded-corridor identity.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The inferred Boolean is converted to 0/1 and broadcast over all trial frames. This indirect inference can label walls rather than using each trial's direct `isRew` value.

ii. ```python
reward_avail = np.full(n_t, int(wall_name in rewarded_wallnames), dtype=np.float32)
```

iii. Notes say reward availability is a per-trial property to be broadcast, but do not justify replacing direct `isRew` with a session-level wall-name set.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It uses `TrialStim`, except the literal placeholder `stimulus_of_trial` is replaced by `WallName`.

ii. ```python
stim_name = str(beh['TrialStim'][trial_idx])
if stim_name == 'stimulus_of_trial':
    stim_name = str(beh['WallName'][trial_idx])
```

iii. Notes considered both fields because some stimulus metadata may be masked or generic.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Every distinct selected-session stimulus string is sorted into a global vocabulary, integer-encoded, and broadcast per trial. Variant names are not collapsed to the four base textures.

ii. ```python
stim_names = sorted({...})
stim_to_idx = {s: i for i, s in enumerate(stim_names)}
stim_arr = np.full(n_t, stim_to_idx[stim_name], dtype=np.int64)
```

iii. Notes anticipated categories including texture variants/probes, rather than documenting the reference's four broad categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived jointly from `LickFr` and `LickTrind`.

ii. ```python
lick_fr = np.asarray(beh['LickFr']).astype(int)
lick_tr = np.asarray(beh['LickTrind']).astype(int)
mask = lick_tr == int(trial_idx)
```

iii. The planning table explicitly identifies both lick fields for constructing a binary time series.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Licks for the trial are converted to offsets from its first corridor frame; valid unique offsets are marked 1 and all others 0.

ii. ```python
rel = lick_fr[mask] - int(start_fr)
rel = rel[(rel >= 0) & (rel < n_t)]
lick[np.unique(rel)] = 1
```

iii. Notes planned a binary framewise lick vector and sanity checks against reference alignment helpers.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The trial-relative lick vector is subselected using the same corridor-relative columns used by neural data.

ii. ```python
out = out[:, cols - st]
sess_neural.append(spk[:, cols].astype(np.float32))
```

iii. The documented intent was framewise alignment using behavior frame indices.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from framewise `ft_Pos` and optional `Corridor_Length` (default 40).

ii. ```python
ft_pos = np.asarray(beh['ft_Pos'], dtype=float)
corridor_len = float(beh.get('Corridor_Length', 40.0))
```

iii. Notes prefer `ft_Pos` because it is already frame-aligned and identify the corridor as 4 m.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The session position stream is sliced to the trial's enclosing frame interval, then discretized; only exact corridor frames survive final subsetting.

ii. ```python
pos_seg = ft_pos[start_fr:end_fr + 1]
out = out[:, cols - st]
```

iii. Notes say gray-space frames should be excluded and that four equal 1 m bins are required.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Corridor length is divided by four, position is floored by that width, and categories are clipped to 0–3.

ii. ```python
bin_w = corridor_len / 4.0
pos_bin = np.clip(np.floor(pos_seg / bin_w).astype(int), 0, 3)
```

iii. This directly follows the notes' four equal-length-bin plan.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is indexed on the same global frame grid and later subselected with the neural corridor columns.

ii. ```python
out = out[:, cols - st]
sess_neural.append(spk[:, cols].astype(np.float32))
```

iii. The notes justify `ft_Pos` specifically as a frame-aligned source.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from framewise `ft_RunSpeed`; `ft_CorrSpc` and `ft_isMoving` select values used to estimate global thresholds.

ii. ```python
v = np.asarray(beh['ft_RunSpeed'], dtype=float)
m &= np.asarray(beh['ft_CorrSpc'], dtype=bool)
m &= np.asarray(beh['ft_isMoving'], dtype=bool)
```

iii. Notes identify `ft_RunSpeed` and propose quartiles; they also cite the paper's running-only analysis.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Values from all selected sessions that are finite, in-corridor, and moving are pooled to calculate thresholds; trial speed is sliced and digitized against them.

ii. ```python
allv = np.concatenate(vals)
qs = np.quantile(allv, [0, 0.25, 0.5, 0.75, 1.0])
speed_bin = np.digitize(speed_seg, speed_edges[1:-1], right=False).astype(int)
```

iii. The mapping plan says to discretize speed into quartiles across the dataset.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three pooled value-quantile cut points delimit four bins, with results clipped to 0–3. Threshold estimation excludes stationary frames but classification does not.

ii. ```python
qs = np.quantile(allv, [0, 0.25, 0.5, 0.75, 1.0])
speed_bin = np.clip(speed_bin, 0, 3)
```

iii. The agent justified quartiles from the task wording, but did not address ties or the reference's per-session rank split.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` uses global behavior-frame indices and is subselected with the identical corridor-relative columns as neural data.

ii. ```python
speed_seg = ft_speed[start_fr:end_fr + 1]
out = out[:, cols - st]
```

iii. Notes state that speed is frame-aligned.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Spike/behavior streams are implicitly bounded by slicing behavior vectors to `n_fr`; missing behavior matches are skipped during threshold collection, nonfinite speeds are excluded, out-of-range licks are dropped, and short trials/sessions are skipped. Other missing fields generally raise errors.

ii. ```python
ft_tr = np.asarray(beh['ft_trInd'])[:n_fr]
m = np.isfinite(v)
rel = rel[(rel >= 0) & (rel < n_t)]
if cols.size < 2: continue
```

iii. The notes do not provide a consolidated missing-data policy; several exploration items and checks remained unfinished.

## 12-a. What are the most time-consuming steps of the code?

i. The likely dominant work is loading and concatenating very large per-session spike arrays, copying them to float32, then copying trial slices into the final in-memory structure. The agent did not document timings.

ii. ```python
obj = np.load(SPK_DIR / f'{session_id}_neural_data.npy', allow_pickle=True).item()
return np.concatenate(list(obj['spks']), axis=0).astype(np.float32)
```

iii. The notes recognize tens of thousands of neurons and full-session arrays, but the “Run Time Estimates” and efficiency fields were left blank.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial search repeatedly scans the full `ft_trInd`/`ft_CorrSpc` vectors; it could group valid frame indices in one pass. Per-trial construction necessarily remains partly iterative because trials vary in length.

ii. ```python
for tr in range(int(beh['ntrials'])):
    cols = np.flatnonzero((ft_tr == tr) & ft_corr)
```

iii. The agent's notes contain only placeholder headings for inefficiencies and speedups, so this is an inference from its code rather than an explicit justification.

## 12-c. What processing does the code repeat multiple times?

i. `find_behavior_for_session` is repeatedly called inside the stimulus-vocabulary comprehension; spike files are loaded once for data and a second time to recover group sizes; session lists are repeatedly filtered and sorted to infer days.

ii. ```python
find_behavior_for_session(s, beh_maps, session_to_group)[1]
raw = np.load(SPK_DIR / f'{sess}_neural_data.npy', allow_pickle=True).item()['spks']
ordered_sessions = sorted([s for s in matched if ...])
```

iii. No justification was recorded; the efficiency documentation remained `[Note]`.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads all behavior dictionaries eagerly, builds unused `UniqWalls` in `infer_rewarded_wallnames`, creates full enclosing interval arrays that are immediately reduced to corridor columns, and stores `kept_sessions` without consuming it.

ii. ```python
uniq = np.asarray(beh['UniqWalls'])
inp = inp[:, cols - st]
out = out[:, cols - st]
kept_sessions.append((sess, group, len(valid), n_neu, n_fr))
```

iii. The agent gave no justification and did not document discarded work.
