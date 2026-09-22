# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads every `Beh_*.npy` file into memory, builds a `session_to_group` map from the behavior-dictionary keys, lists all spike-session filenames in `spk/`, and keeps only the intersection of spike-session ids and behavior keys. Neural data are then loaded per session from `*_neural_data.npy` and concatenated across the three `spks` arrays. The retinotopy files are not used to enumerate sessions or load trials.

ii. ```python
def load_behavior_maps():
    beh_maps = {}
    session_to_group = defaultdict(list)
    for f in sorted(BEH_DIR.glob('Beh_*.npy')):
        group = f.stem.replace('Beh_', '')
        d = np.load(f, allow_pickle=True).item()
        beh_maps[group] = d
        for sess in d:
            session_to_group[sess].append(group)

spk_sessions = sorted(p.stem.replace('_neural_data', '') for p in SPK_DIR.glob('*_neural_data.npy'))
matched = [s for s in spk_sessions if s in session_to_group]
```

iii. The trajectory shows the agent initially explored `Imaging_Exp_info.npy`, but the final implementation deliberately switched to direct matching between `spk/` filenames and behavior-dict keys; later notes describe this as a conservative way to reconstruct sessions when the terminal was unstable.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the session id prefix before the first underscore, and the final `subjects` list is the sorted set of those prefixes. Each kept session gets `subject_idx` from that derived mapping.

ii. ```python
subjects = sorted({s.split('_')[0] for s in matched})
subj_to_idx = {s: i for i, s in enumerate(subjects)}
...
data['subject_idx'].append(subj_to_idx[sess.split('_')[0]])
```

iii. The trajectory repeatedly treats ids such as `TX108_2023_03_13_1` as self-describing, so the mouse name is taken directly from the session string rather than from the master index.

## 1-c. How are the data split into sessions?

i. A session is any spike filename stem that also appears as an exact behavior key. If the same session id appears in multiple behavior groups, the code picks one preferred group by sorting group names so `sup*` and `unsup*` outrank `naive*`. Sessions present only through suffixed behavior keys are missed.

ii. ```python
def find_behavior_for_session(session_id, beh_maps, session_to_group):
    groups = session_to_group.get(session_id, [])
    if not groups:
        return None, None
    pref = sorted(groups, key=lambda g: (('sup' not in g and 'unsup' not in g), g))
    g = pref[0]
    return g, beh_maps[g][session_id]

matched = [s for s in spk_sessions if s in session_to_group]
```

iii. The trajectory says the agent wanted a single behavior source per spike session and preferred supervised/unsupervised groups over naive when duplicates existed.

## 1-d. How are the data split into trials?

i. Trials are reconstructed from frame-level behavior labels: for each trial index `tr`, the kept columns are the imaging frames where `ft_trInd == tr` and `ft_CorrSpc` is true. The neural, input, and output arrays for a trial all use those same corridor-frame columns.

ii. ```python
ft_tr = np.asarray(beh['ft_trInd'])[:n_fr]
ft_corr = np.asarray(beh['ft_CorrSpc'])[:n_fr].astype(bool)
...
cols = np.flatnonzero((ft_tr == tr) & ft_corr)
...
sess_neural.append(spk[:, cols].astype(np.float32))
```

iii. Step 52 of the trajectory explicitly says the earlier `StartFr`/`EndFr` segmentation looked wrong, so the agent refactored trial segmentation to use `ft_trInd` plus `ft_CorrSpc`, which it considered closer to the reference logic.

## 1-e. How are trials filtered based on quality controls?

i. The only trial-level quality control is a minimum length check: trials with fewer than 2 kept corridor frames are dropped. Sessions with fewer than 2 surviving trials are skipped entirely. There is no percentile-based long-trial removal.

ii. ```python
for tr in range(int(beh['ntrials'])):
    cols = np.flatnonzero((ft_tr == tr) & ft_corr)
    if cols.size < 2:
        continue
    valid.append((tr, cols))
if len(valid) < 2:
    print('skip', sess, 'valid_trials', len(valid), 'shape', spk.shape)
    continue
```

iii. The trajectory frames this as a conservative decoder-format requirement: keep only trials with enough columns to decode and drop sessions that cannot satisfy the decoder's minimum of two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural matrices come from the `spks` list inside each `spk/<session_id>_neural_data.npy` file. The three arrays are concatenated along axis 0. The retinotopy `iarea` values are not used to derive or filter the neural activity itself.

ii. ```python
def load_spk_session(session_id):
    obj = np.load(SPK_DIR / f'{session_id}_neural_data.npy', allow_pickle=True).item()
    return np.concatenate(list(obj['spks']), axis=0).astype(np.float32)
```

iii. The trajectory records the key inference that `spks` held three neuron groups, not three trials, so concatenating them was treated as the correct reconstruction of a full-session neuron-by-time matrix.

## 2-b. How is the `neural` data processed?

i. The neural traces are concatenated across groups, cast to `float32`, and sliced into per-trial matrices using the selected corridor-frame columns. No denoising, normalization, rebinning, padding, or interpolation is applied in the final script.

ii. ```python
return np.concatenate(list(obj['spks']), axis=0).astype(np.float32)
...
sess_neural.append(spk[:, cols].astype(np.float32))
```

iii. The notes and trajectory repeatedly state that the provided traces are already deconvolved activity, so the agent decided not to recompute fluorescence preprocessing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The neural data are not filtered by retinotopic area or any other neuron-quality criterion. All concatenated neurons are kept. The only effective filtering is indirect trial/session filtering.

ii. ```python
brain_regions = ['unknown_group0', 'unknown_group1', 'unknown_group2']
...
spk = load_spk_session(sess)
...
sess_neural.append(spk[:, cols].astype(np.float32))
```

iii. The trajectory explicitly says retinotopy mapping was unresolved, so the agent fell back to placeholder brain-region groups rather than dropping neurons by area.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned to corridor entry in the sense that each trial begins at the first frame assigned to that trial inside the corridor (`ft_CorrSpc`) and then uses all later kept corridor frames for that same trial. The implementation does not align by an interpolated timestamp; it aligns by frame membership.

ii. ```python
cols = np.flatnonzero((ft_tr == tr) & ft_corr)
...
sess_neural.append(spk[:, cols].astype(np.float32))
...
'metadata': {
    'temporal_alignment_event': 'trial start / corridor entry',
```

iii. The trajectory says corridor-entry alignment was required by the instructions and that `ft_trInd` plus `ft_CorrSpc` was the most trustworthy frame-level way to realize that alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data keep the original frame grid with no temporal rebinning, but the metadata records the bin size as `1.0` and all time-like inputs are expressed in frame units rather than seconds. So the practical resolution is one imaging frame, represented as if one frame were one unit of time.

ii. ```python
'metadata': {
    ...
    'time_bin_size': 1.0,
    ...
}
...
time_to_cue = (sound_fr - frame_idx).astype(np.float32)
time_since_start = np.arange(n_t, dtype=np.float32)
```

iii. The trajectory shows the agent focused on frame-level consistency and never reintroduced the reference notebook's 3.17 Hz timing when finalizing the code.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from the trial's `SoundFr` value and the integer frame indices spanning the selected trial window. The code does not use `ft` timestamps.

ii. ```python
frame_idx = np.arange(start_fr, end_fr + 1)
sound_fr = int(round(float(beh['SoundFr'][trial_idx])))
time_to_cue = (sound_fr - frame_idx).astype(np.float32)
```

iii. The trajectory discusses cue timing primarily in terms of frame numbers, and the final implementation stayed in frame units for simplicity.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the code rounds `SoundFr` to an integer frame, subtracts the current frame index, and stores the result as a continuous frame-offset signal. It is not interpolated onto `ft`, and it is not converted to seconds.

ii. ```python
sound_fr = int(round(float(beh['SoundFr'][trial_idx])))
time_to_cue = (sound_fr - frame_idx).astype(np.float32)
...
inp = np.vstack([time_to_cue, day_arr, time_since_start, reward_avail]).astype(np.float32)
```

iii. The notes mention `SoundFr`/`SoundTime` as candidate sources, but the final code settles on the simpler frame-difference computation.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. `time_to_sound_cue` is computed on the same trial window as the outputs and then subselected to the same corridor-frame columns used for neural data, so its final samples are aligned one-for-one with the neural columns.

ii. ```python
inp, out = build_trial_io(beh, tr, st, en, day_val, stim_to_idx, speed_edges, rewarded_wallnames)
# remap framewise outputs/inputs to corridor-frame columns only
inp = inp[:, cols - st]
out = out[:, cols - st]
sess_neural.append(spk[:, cols].astype(np.float32))
```

iii. The trajectory explicitly describes this remapping as the fix that puts framewise inputs/outputs back onto the same corridor-frame grid as the neural data.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. `day_of_training` is derived from session ids only: for each subject, the code sorts that subject's matched session ids by the date/block fields embedded in the string and uses the ordinal position of the current session. It does not use `Imaging_Exp_info.npy`.

ii. ```python
ordered_sessions = sorted([s for s in matched if s.split('_')[0] == sess.split('_')[0]], key=lambda x: tuple(x.split('_')[1:4]))
day_val = infer_session_day(sess, ordered_sessions)

def infer_session_day(session_id, ordered_sessions):
    return float(ordered_sessions.index(session_id) + 1)
```

iii. The trajectory says the agent switched away from behavior-group names after discovering they made `day_of_training` constant, and instead used chronological order of session ids within subject.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The sorted session order within a subject is converted to a 1-based float (`1.0`, `2.0`, ...), and that scalar is broadcast across all time bins of the trial. Only the matched/kept sessions for that subject are counted.

ii. ```python
def infer_session_day(session_id, ordered_sessions):
    return float(ordered_sessions.index(session_id) + 1)
...
day_arr = np.full(n_t, day_val, dtype=np.float32)
```

iii. Step 46 of the trajectory notes a failed earlier approach; the final justification was simply to use per-subject chronological ordering of available sessions.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. In the final code it is effectively derived from the trial window itself, not from `ft` timestamps: `build_trial_io` creates `np.arange(n_t)` over the contiguous window from `start_fr` to `end_fr`, then later keeps the corridor-frame subset. So the only raw trial-start dependency is via the chosen window start.

ii. ```python
n_t = end_fr - start_fr + 1
time_since_start = np.arange(n_t, dtype=np.float32)
```

iii. The trajectory shows an early plan to use `StartFr` and frame times, but after the segmentation rewrite the implemented signal became a simple per-frame offset within the selected trial window.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The code creates a frame counter starting at 0 for the contiguous trial window and keeps the corridor-frame entries corresponding to the neural columns. It is not interpolated from timestamps and not converted to seconds.

ii. ```python
time_since_start = np.arange(n_t, dtype=np.float32)
...
inp = inp[:, cols - st]
```

iii. The final script prioritizes frame alignment over physical time, which is consistent with how the agent debugged several alignment bugs in the trajectory.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned by the same corridor-frame remapping used for `time_to_sound_cue` and the outputs: `cols` select the neural frames, and `cols - st` selects the matching entries from the window-based input arrays.

ii. ```python
sess_neural.append(spk[:, cols].astype(np.float32))
...
inp = inp[:, cols - st]
```

iii. The trajectory repeatedly calls out this `cols - st` remapping as the mechanism that keeps framewise inputs and outputs synchronized with `spk[:, cols]`.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from both `WallName` and `isRew`. First the session-level set of rewarded wall names is inferred from trials where `isRew` is true, then each trial's current `WallName` is tested against that set.

ii. ```python
def infer_rewarded_wallnames(beh):
    wall = np.asarray(beh['WallName'])
    isrew = np.asarray(beh['isRew']).astype(bool)
    rewarded = set(wall[isrew].tolist())
    ...

wall_name = str(beh['WallName'][trial_idx])
reward_avail = np.full(n_t, int(wall_name in rewarded_wallnames), dtype=np.float32)
```

iii. The trajectory says the agent wanted `reward_availability` to represent rewarded-corridor identity rather than only the raw per-trial flag, so it inferred which wall textures were rewarded within a session.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The processing is: infer the set of rewarded wall textures for the session from `isRew`, assign each trial a binary rewarded-corridor label via membership in that set, and broadcast that label across every time bin of the trial.

ii. ```python
rewarded_wallnames = infer_rewarded_wallnames(beh)
...
reward_avail = np.full(n_t, int(wall_name in rewarded_wallnames), dtype=np.float32)
...
inp = np.vstack([time_to_cue, day_arr, time_since_start, reward_avail]).astype(np.float32)
```

iii. This was an intentional design choice in the trajectory after the agent noticed that some sample conversions had reward all zero when it relied too literally on raw session grouping.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The label is derived primarily from `TrialStim`, with a fallback to `WallName` only when `TrialStim` equals the placeholder string `'stimulus_of_trial'`.

ii. ```python
stim_name = str(beh['TrialStim'][trial_idx])
if stim_name == 'stimulus_of_trial':
    stim_name = str(beh['WallName'][trial_idx])
stim_idx = stim_to_idx[stim_name]
```

iii. Late in the trajectory the agent explicitly patches this fallback because the full run had produced a bogus `stimulus_of_trial` category.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The agent collects the sorted set of all observed stimulus names across the selected sessions, maps each unique string to an integer index, and broadcasts that per-trial label across all time bins of the trial. It does not collapse crops/swaps into the four base texture categories.

ii. ```python
stim_names = sorted({(lambda beh,i: (str(beh['WallName'][i]) if str(beh['TrialStim'][i]) == 'stimulus_of_trial' else str(beh['TrialStim'][i])))(find_behavior_for_session(s, beh_maps, session_to_group)[1], i)
                     for s in matched
                     for i in range(int(find_behavior_for_session(s, beh_maps, session_to_group)[1]['ntrials']))})
stim_to_idx = {s: i for i, s in enumerate(stim_names)}
...
stim_arr = np.full(n_t, stim_idx, dtype=np.int64)
```

iii. The trajectory treats removal of the placeholder category as the key fix; it never adopts the reference solution's hard-coded mapping from many wall names to four base classes.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from the lick event arrays `LickFr` and `LickTrind`. The code uses `LickTrind` to keep only lick events belonging to the current trial and `LickFr` to locate them in time.

ii. ```python
lick_fr = np.asarray(beh['LickFr']).astype(int)
lick_tr = np.asarray(beh['LickTrind']).astype(int)
mask = lick_tr == int(trial_idx)
rel = lick_fr[mask] - int(start_fr)
```

iii. The trajectory shows the agent introduced `LickTrind` after earlier sample outputs had all-zero licking, so the trial assignment was used as an explicit fix.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial window, the code allocates a zero vector, converts lick frames to trial-relative indices, clips to the window, deduplicates repeated indices with `np.unique`, and marks those bins as 1. The result is a binary time series.

ii. ```python
lick = np.zeros(n_t, dtype=np.int64)
...
rel = rel[(rel >= 0) & (rel < n_t)]
if rel.size:
    lick[np.unique(rel)] = 1
```

iii. The trajectory describes this as a debugging response to mismatched lick indexing in earlier attempts.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick vector is first defined over the same contiguous trial window as the other framewise variables, then remapped with `cols - st` so the final values correspond exactly to the corridor-frame neural columns in `spk[:, cols]`.

ii. ```python
lick = build_lick_vector(beh, start_fr, end_fr, trial_idx)
...
out = np.vstack([
    stim_arr,
    lick.astype(np.int64),
    pos_bin.astype(np.int64),
    speed_bin.astype(np.int64),
])
...
out = out[:, cols - st]
```

iii. The trajectory explicitly says the framewise remapping step was introduced to realign outputs with corridor-only neural frames.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from the frame-level position trace `ft_Pos`.

ii. ```python
ft_pos = np.asarray(beh['ft_Pos'], dtype=float)
pos_seg = ft_pos[start_fr:end_fr + 1]
```

iii. The notes and trajectory consistently identify `ft_Pos` as the frame-aligned position signal to use for corridor position.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The trial's `ft_Pos` segment is divided into four equal corridor-length chunks, floored to integer bin ids, clipped to `[0, 3]`, and then the corridor-frame subset is kept. With the default `Corridor_Length` of `40.0`, this becomes 10-unit bins.

ii. ```python
corridor_len = float(beh.get('Corridor_Length', 40.0))
bin_w = corridor_len / 4.0
pos_bin = np.clip(np.floor(pos_seg / bin_w).astype(int), 0, 3)
...
out = out[:, cols - st]
```

iii. Step 54 of the trajectory says the agent fixed an earlier position bug after realizing `ft_Pos` spans roughly 0 to 39 and therefore needs 10-unit, not 1-unit, bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The thresholding is deterministic geometry-based binning: bin width is `Corridor_Length / 4`, so the four categories are equal-length spatial bins across the corridor.

ii. ```python
corridor_len = float(beh.get('Corridor_Length', 40.0))
bin_w = corridor_len / 4.0
pos_bin = np.clip(np.floor(pos_seg / bin_w).astype(int), 0, 3)
```

iii. The trajectory explicitly justifies this using the discovered corridor scale of about 40 position units over a 4 m corridor.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is first computed over the same contiguous trial window and then subselected with `cols - st`, which puts it on the same corridor-frame grid as the neural data.

ii. ```python
pos_seg = ft_pos[start_fr:end_fr + 1]
...
out = out[:, cols - st]
sess_neural.append(spk[:, cols].astype(np.float32))
```

iii. The agent repeatedly describes the `cols`/`cols - st` pattern as its generic frame-alignment mechanism for all time-varying variables.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from the frame-level speed trace `ft_RunSpeed`.

ii. ```python
v = np.asarray(beh['ft_RunSpeed'], dtype=float)
...
ft_speed = np.asarray(beh['ft_RunSpeed'], dtype=float)
speed_seg = ft_speed[start_fr:end_fr + 1]
```

iii. The notes identify `ft_RunSpeed` as the direct frame-aligned running variable, and the final code uses it that way.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The code first computes dataset-level quartile edges from all selected sessions' speed samples, optionally masking to finite values, `ft_CorrSpc`, and `ft_isMoving`. Each trial's speed segment is then digitized against those global edges and later restricted to the corridor-frame subset.

ii. ```python
def collect_speed_values(session_ids, beh_maps, session_to_group):
    ...
    v = np.asarray(beh['ft_RunSpeed'], dtype=float)
    m = np.isfinite(v)
    if 'ft_CorrSpc' in beh:
        m &= np.asarray(beh['ft_CorrSpc'], dtype=bool)
    if 'ft_isMoving' in beh:
        m &= np.asarray(beh['ft_isMoving'], dtype=bool)
    vals.append(v[m])
    ...
    qs = np.quantile(allv, [0, 0.25, 0.5, 0.75, 1.0])

speed_bin = np.digitize(speed_seg, speed_edges[1:-1], right=False).astype(int)
```

iii. The trajectory says the agent wanted quartile bins from kept running-related frames and experimented with `ft_isMoving`; the final code bakes that into the threshold estimation.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded by `np.quantile` cut points at 0, 25, 50, 75, and 100 percent across the collected speed values, then assigned with `np.digitize` and clipped to four bins.

ii. ```python
qs = np.quantile(allv, [0, 0.25, 0.5, 0.75, 1.0])
qs[-1] = np.nextafter(qs[-1], np.inf)
...
speed_bin = np.digitize(speed_seg, speed_edges[1:-1], right=False).astype(int)
speed_bin = np.clip(speed_bin, 0, 3)
```

iii. The agent's notes and sample-run prints refer to these as running-speed quartile edges, and the full conversion output prints the chosen thresholds.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Like the other time-varying outputs, running speed is computed on the trial window and then remapped with `cols - st` so the final bins line up with the neural corridor frames.

ii. ```python
speed_seg = ft_speed[start_fr:end_fr + 1]
...
out = out[:, cols - st]
sess_neural.append(spk[:, cols].astype(np.float32))
```

iii. The trajectory uses the same justification here as for licking and position: corridor-frame remapping after building window-based arrays.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles missing or awkward data in a few ad hoc ways: sessions without an exact behavior-key match are dropped; frame-level behavior arrays used for trial segmentation are truncated to `n_fr`; licks outside the trial window are clipped away; speed-edge estimation ignores non-finite values; if no speed values exist it falls back to `[0,1,2,3,4]`; and sessions with fewer than two valid trials are skipped. There is no broader missing-data or outlier-handling logic.

ii. ```python
matched = [s for s in spk_sessions if s in session_to_group]
...
ft_tr = np.asarray(beh['ft_trInd'])[:n_fr]
ft_corr = np.asarray(beh['ft_CorrSpc'])[:n_fr].astype(bool)
...
rel = rel[(rel >= 0) & (rel < n_t)]
...
m = np.isfinite(v)
...
if not vals:
    return np.array([0, 1, 2, 3, 4], dtype=float)
...
if len(valid) < 2:
    ... continue
```

iii. The trajectory shows these checks were added reactively while debugging all-zero or misaligned sample outputs rather than as a principled curation stage.

## 12-a. What are the most time-consuming steps of the code?

i. The main costs are loading full behavior dictionaries up front and loading each full spike file per session. The code also reloads each raw spike file a second time just to recover the original group sizes for placeholder brain-region labels.

ii. ```python
for f in sorted(BEH_DIR.glob('Beh_*.npy')):
    ...
    d = np.load(f, allow_pickle=True).item()

spk = load_spk_session(sess)
...
raw = np.load(SPK_DIR / f'{sess}_neural_data.npy', allow_pickle=True).item()['spks']
```

iii. The notes do not document runtime carefully, but the code structure makes file I/O, especially repeated spike-file reads, the dominant cost.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization target is the per-trial scan over `ft_trInd` that calls `np.flatnonzero` once for every trial. The repeated session-level loops in `collect_speed_values` and the nested set-comprehension that repeatedly calls `find_behavior_for_session` are additional avoidable Python-level loops.

ii. ```python
for tr in range(int(beh['ntrials'])):
    cols = np.flatnonzero((ft_tr == tr) & ft_corr)
    ...

for sess in session_ids:
    _, beh = find_behavior_for_session(sess, beh_maps, session_to_group)
    ...

stim_names = sorted({ ... find_behavior_for_session(s, beh_maps, session_to_group) ... })
```

iii. The trajectory specifically highlights trial-frame reconstruction as the central alignment operation, but the implementation leaves it as repeated Python loops.

## 12-c. What processing does the code repeat multiple times?

i. It repeats several pieces of work: behavior lookup is recomputed many times during stimulus and speed setup; every kept session's spike file is loaded once for neural data and again for placeholder brain-region group sizes; and framewise trial arrays are built on a contiguous window before being sliced back down to corridor frames.

ii. ```python
stim_names = sorted({ ... find_behavior_for_session(s, beh_maps, session_to_group)[1] ... })
speed_edges = collect_speed_values(matched, beh_maps, session_to_group)
...
spk = load_spk_session(sess)
...
raw = np.load(SPK_DIR / f'{sess}_neural_data.npy', allow_pickle=True).item()['spks']
...
inp, out = build_trial_io(...)
inp = inp[:, cols - st]
out = out[:, cols - st]
```

iii. These repeats were not called out as a concern in `CONVERSION_NOTES.md`; they are evident from the final code path itself.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The most obvious discarded work is that `build_trial_io` computes inputs and outputs for every frame from `start_fr` to `end_fr`, but the caller immediately throws away any frame not in the corridor subset by indexing with `cols - st`. The extra second spike-file load used only to fabricate placeholder brain-region groups is also unnecessary from the perspective of downstream decoding.

ii. ```python
inp, out = build_trial_io(beh, tr, st, en, day_val, stim_to_idx, speed_edges, rewarded_wallnames)
# remap framewise outputs/inputs to corridor-frame columns only
inp = inp[:, cols - st]
out = out[:, cols - st]
...
raw = np.load(SPK_DIR / f'{sess}_neural_data.npy', allow_pickle=True).item()['spks']
```

iii. The trajectory makes clear this window-then-subsample pattern came from debugging, not from a deliberate efficiency tradeoff.
