# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by listing spike files in `spk/` and then matching each to a behavior dictionary. Behavior is loaded from `Beh_*.npy` files via `load_behavior_maps()`, which builds a mapping from session IDs to behavior group names. Spike files are matched if their session ID appears as a key in any behavior dict. This yields 76 of 89 sessions — 13 sessions are lost because their behavior keys include a `_stimtype` suffix (for swap sessions) that does not match the bare session ID.

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

spk_sessions = sorted(p.stem.replace('_neural_data', '') for p in SPK_DIR.glob('*_neural_data.npy'))
matched = [s for s in spk_sessions if s in session_to_group]
```

iii. The AI notes in conversion_full_out.txt: "spk sessions: 89, matched sessions: 76". The trajectory shows the AI acknowledged this discrepancy but considered it "plausible given 89 recordings total with only 76 matched behavior sessions." The AI did not investigate why 13 sessions were unmatched.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from the first component of the session ID (split on `_`). Unique subjects are sorted and indexed.

ii.
```python
subjects = sorted({s.split('_')[0] for s in matched})
subj_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. No explicit justification; the approach is straightforward. It yields 19 subjects, matching the paper.

## 1-c. How are the data split into sessions?

i. A session is one spike file matched to a behavior dictionary. The AI finds behavior for a session by looking up the session ID in `session_to_group` and preferring supervised/unsupervised groups over naive. Only 76 of 89 sessions are matched because swap session keys include a `_stimtype` suffix.

ii.
```python
def find_behavior_for_session(session_id, beh_maps, session_to_group):
    groups = session_to_group.get(session_id, [])
    if not groups:
        return None, None
    pref = sorted(groups, key=lambda g: (('sup' not in g and 'unsup' not in g), g))
    g = pref[0]
    return g, beh_maps[g][session_id]
```

iii. The AI's trajectory notes "76 matched sessions... plausible given 89 recordings total." The AI did not use `Imaging_Exp_info.npy` as the master index (which the reference does), instead discovering sessions bottom-up from spike files.

## 1-d. How are the data split into trials?

i. Trials are identified using `ft_trInd` and `ft_CorrSpc` from the behavior dict, same as the reference. For each trial index, frames where `ft_trInd == trial` and `ft_CorrSpc` is true are collected. Trials with fewer than 2 valid frames are skipped.

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

iii. The approach of using `ft_trInd` and `ft_CorrSpc` is consistent with the reference code's `trial_frames` function.

## 1-e. How are trials filtered based on quality controls?

i. Only trials with fewer than 2 valid corridor-space frames are dropped. No outlier trial length filtering is applied (unlike the reference which drops trials beyond the 99th percentile of length).

ii.
```python
if cols.size < 2:
    continue
```

iii. No explicit justification for omitting outlier trial filtering. The verification output shows max trial lengths of 5607 frames, confirming extremely long trials are retained.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy`, which is a list of arrays per imaging plane, concatenated along axis 0.

ii.
```python
def load_spk_session(session_id):
    obj = np.load(SPK_DIR / f'{session_id}_neural_data.npy', allow_pickle=True).item()
    return np.concatenate(list(obj['spks']), axis=0).astype(np.float32)
```

iii. The AI correctly identified that `spks` entries are neuron groups (imaging planes), not trials.

## 2-b. How is the `neural` data processed?

i. The deconvolved traces are concatenated across planes, cast to float32, and sliced to corridor-space frames per trial. No further processing (normalization, filtering, etc.) is applied.

ii.
```python
sess_neural.append(spk[:, cols].astype(np.float32))
```

iii. The AI correctly uses deconvolved traces as-is, matching the reference approach.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron filtering is applied. All neurons from all imaging planes are kept. The reference filters neurons to four visual areas (V1, mHV, lHV, aHV) using the retinotopy `iarea` field, keeping ~87% of neurons. The AI does not use retinotopy for neuron selection.

ii.
```python
# No filtering code — all neurons from concatenated spks are kept
spk = load_spk_session(sess)
```

iii. The AI assigns brain regions as placeholder groups based on imaging plane index rather than actual visual areas. From conversion notes: the AI was aware of retinotopy files but did not implement area-based filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to corridor entry by selecting frames where `ft_CorrSpc` is true for each trial. The first corridor-space frame of each trial serves as the alignment point (trial start).

ii.
```python
cols = np.flatnonzero((ft_tr == tr) & ft_corr)
st, en = int(cols[0]), int(cols[-1])
sess_neural.append(spk[:, cols].astype(np.float32))
```

iii. This is consistent with the reference's approach of using `ft_CorrSpc` frames.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied — each imaging frame is one time bin. However, the metadata `time_bin_size` is set to 1.0 ms, which is incorrect. The actual imaging rate is ~3.17 Hz, making each bin ~315 ms.

ii.
```python
'time_bin_size': 1.0,
```

iii. The AI did not compute or look up the actual frame rate. The value 1.0 appears to be a placeholder that was never corrected.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the frame number of the sound cue for each trial) and the frame indices of the trial.

ii.
```python
sound_fr = int(round(float(beh['SoundFr'][trial_idx])))
time_to_cue = (sound_fr - frame_idx).astype(np.float32)
```

iii. The AI computes time-to-cue as the difference in frame indices (integer frames), not in seconds as the reference does.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The sound frame is rounded to an integer frame number, and the time-to-cue is computed as `sound_fr - frame_idx` for each frame in the trial window. This gives time in frames (integers), not in seconds. The reference converts frame times to seconds using `ft` timestamps and computes `cue_time - frame_time`.

ii.
```python
frame_idx = np.arange(start_fr, end_fr + 1)
sound_fr = int(round(float(beh['SoundFr'][trial_idx])))
time_to_cue = (sound_fr - frame_idx).astype(np.float32)
```

iii. No justification for using frame units. The sign convention (positive before cue) matches the reference.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The time-to-cue is computed for all frames from `start_fr` to `end_fr`, then subselected to corridor-space frames via `inp = inp[:, cols - st]`. This aligns it with the neural data.

ii.
```python
inp = inp[:, cols - st]
```

iii. The alignment approach is correct — corridor-space frames are selected from the full trial window.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the session IDs of each subject, sorted by date components, giving ordinal position within a subject.

ii.
```python
def infer_session_day(session_id, ordered_sessions):
    return float(ordered_sessions.index(session_id) + 1)

ordered_sessions = sorted([s for s in matched if s.split('_')[0] == sess.split('_')[0]],
                           key=lambda x: tuple(x.split('_')[1:4]))
day_val = infer_session_day(sess, ordered_sessions)
```

iii. The AI uses 1-based indexing (first session is day 1) while the reference uses 0-based (first session is day 0). The AI only counts matched sessions (76), so subjects with missing sessions may have incorrect day counts.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Sessions for each subject are sorted by date, and the ordinal position (1-indexed) is used as the day value. It is broadcast across all time bins of every trial in that session.

ii.
```python
day_arr = np.full(n_t, day_val, dtype=np.float32)
```

iii. The 1-indexing vs 0-indexing is a minor difference. The bigger issue is that only matched sessions are counted, so if a subject has sessions that were lost in the matching step, the day count may skip values.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Computed as a simple integer sequence starting from 0, representing frame index within the trial window.

ii.
```python
time_since_start = np.arange(n_t, dtype=np.float32)
```

iii. The AI uses frame count (0, 1, 2, ...) rather than converting to seconds. The reference uses actual timestamps from `ft` to compute time in seconds since corridor entry.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. A simple `np.arange(n_t)` gives the frame index. No conversion to seconds is performed.

ii.
```python
time_since_start = np.arange(n_t, dtype=np.float32)
```

iii. The reference computes `frame_time[window] - start_time[trial]` in seconds. The AI's approach gives time in frame units.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Aligned by construction — the range starts at 0 for the first corridor-space frame and increments by 1 per frame, matching the neural data dimensions.

ii.
```python
time_since_start = np.arange(n_t, dtype=np.float32)
# later:
inp = inp[:, cols - st]
```

iii. The alignment is correct in that it matches the neural data frames, though the units are frames not seconds.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from `WallName` — the AI infers which wall names are associated with rewarded corridors by checking which `WallName` values co-occur with `isRew == True`.

ii.
```python
def infer_rewarded_wallnames(beh):
    wall = np.asarray(beh['WallName'])
    isrew = np.asarray(beh['isRew']).astype(bool)
    rewarded = set(wall[isrew].tolist())
    if not rewarded:
        return set()
    return rewarded

reward_avail = np.full(n_t, int(wall_name in rewarded_wallnames), dtype=np.float32)
```

iii. The reference simply uses `beh['isRew']` directly. The AI's approach is indirect but functionally similar within a session — it checks if a trial's `WallName` is in the set of wall names seen with rewards in that session.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. For each session, the set of rewarded wall names is inferred. Each trial's reward availability is 1 if its `WallName` is in the rewarded set, 0 otherwise. Broadcast across all frames.

ii.
```python
reward_avail = np.full(n_t, int(wall_name in rewarded_wallnames), dtype=np.float32)
```

iii. This could produce incorrect results in edge cases where the same wall name appears in both rewarded and unrewarded corridors within a session, though in practice this may not occur.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `TrialStim`, falling back to `WallName` when `TrialStim` equals `'stimulus_of_trial'` (a placeholder).

ii.
```python
stim_name = str(beh['TrialStim'][trial_idx])
if stim_name == 'stimulus_of_trial':
    stim_name = str(beh['WallName'][trial_idx])
stim_idx = stim_to_idx[stim_name]
```

iii. The AI uses raw stimulus names rather than grouping into base textures.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI collects all unique stimulus names across the dataset and assigns each an integer index. This yields 9 categories: `['circle1', 'circle2', 'circle3', 'leaf1', 'leaf1_swap1', 'leaf1_swap2', 'leaf2', 'leaf3', 'wood5']`. The reference groups these into 4 base textures (`circle`, `leaf`, `rock`, `wood`). The AI's approach misses `rock` entirely and splits `circle` and `leaf` variants into separate categories.

ii.
```python
stim_names = sorted({...})  # yields 9 raw names
stim_to_idx = {s: i for i, s in enumerate(stim_names)}
```

iii. The conversion_full_out.txt confirms: "stim categories: ['circle1', 'circle2', 'circle3', 'leaf1', 'leaf1_swap1', 'leaf1_swap2', 'leaf2', 'leaf3', 'wood5']". The AI did not group variants into base textures.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (frame number of each lick) and `LickTrind` (trial index of each lick).

ii.
```python
def build_lick_vector(beh, start_fr, end_fr, trial_idx):
    lick_fr = np.asarray(beh['LickFr']).astype(int)
    lick_tr = np.asarray(beh['LickTrind']).astype(int)
    mask = lick_tr == int(trial_idx)
    rel = lick_fr[mask] - int(start_fr)
```

iii. The AI uses `LickTrind` to filter licks to the current trial. The reference uses a session-wide approach, marking all lick frames then indexing per trial.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, lick frames belonging to that trial are identified via `LickTrind`, converted to relative frame indices within the trial window, and a binary vector is created (0 = no lick, 1 = lick).

ii.
```python
n_t = end_fr - start_fr + 1
lick = np.zeros(n_t, dtype=np.int64)
rel = lick_fr[mask] - int(start_fr)
rel = rel[(rel >= 0) & (rel < n_t)]
if rel.size:
    lick[np.unique(rel)] = 1
```

iii. Functionally similar to the reference, though the per-trial filtering via `LickTrind` adds complexity.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick vector is built for the full trial window (`start_fr` to `end_fr`), then subselected to corridor-space frames via `out = out[:, cols - st]`.

ii.
```python
out = out[:, cols - st]
```

iii. Correctly aligned to corridor-space frames.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position in decimeters at each imaging frame.

ii.
```python
ft_pos = np.asarray(beh['ft_Pos'], dtype=float)
pos_seg = ft_pos[start_fr:end_fr + 1]
```

iii. Same source variable as the reference.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is divided by `Corridor_Length / 4` (defaults to 10 dm = 1 m bins) and floored, then clipped to [0, 3].

ii.
```python
corridor_len = float(beh.get('Corridor_Length', 40.0))
bin_w = corridor_len / 4.0
pos_bin = np.clip(np.floor(pos_seg / bin_w).astype(int), 0, 3)
```

iii. The reference uses `ft_Pos // 10` clipped to [0, 3], which is equivalent when `Corridor_Length` is 40.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four 1-meter bins: 0-1m, 1-2m, 2-3m, 3-4m. However the verification output shows position only reaches bin 2 (max 2.0), suggesting the 3-4m bin is never populated. This is because `ft_CorrSpc` limits frames to the corridor space which may not include the full 4m.

ii.
```python
pos_bin = np.clip(np.floor(pos_seg / bin_w).astype(int), 0, 3)
```

iii. The verification output confirms: "corridor_position_bin: {0-1m (0.409), 1-2m (0.344), 2-3m (0.247)}" — only 3 of 4 bins populated. The reference has the same issue but reaches bin 3 in some sessions.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is extracted for the full trial window, then subselected to corridor-space frames.

ii.
```python
pos_seg = ft_pos[start_fr:end_fr + 1]
# later:
out = out[:, cols - st]
```

iii. Correctly aligned.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
ft_speed = np.asarray(beh['ft_RunSpeed'], dtype=float)
speed_seg = ft_speed[start_fr:end_fr + 1]
```

iii. Same source variable as the reference.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speed is discretized into 4 bins using global quantile edges computed across all sessions. The edges are derived from frames that are both in corridor space (`ft_CorrSpc`) and moving (`ft_isMoving`).

ii.
```python
def collect_speed_values(session_ids, beh_maps, session_to_group):
    ...
    m = np.isfinite(v)
    if 'ft_CorrSpc' in beh:
        m &= np.asarray(beh['ft_CorrSpc'], dtype=bool)
    if 'ft_isMoving' in beh:
        m &= np.asarray(beh['ft_isMoving'], dtype=bool)
    vals.append(v[m])
    allv = np.concatenate(vals)
    qs = np.quantile(allv, [0, 0.25, 0.5, 0.75, 1.0])
```

iii. The reference computes quartiles per session using rank-based splitting over kept frames. The AI uses global value-based quantile edges, and additionally filters to moving frames when computing edges, which changes the distribution.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Using `np.digitize` with the global quantile edges to assign each frame to one of 4 bins.

ii.
```python
speed_bin = np.digitize(speed_seg, speed_edges[1:-1], right=False).astype(int)
speed_bin = np.clip(speed_bin, 0, 3)
```

iii. The reference uses per-session rank-based quartiles ensuring exactly 25% of kept frames in each bin. The AI's global edges produce uneven distributions: verification shows q1 at 54.9%, q2-q4 at ~15% each.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is extracted for the full trial window, then subselected to corridor-space frames.

ii.
```python
speed_seg = ft_speed[start_fr:end_fr + 1]
# later:
out = out[:, cols - st]
```

iii. Correctly aligned.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Behavior streams are clipped to the number of neural frames (`n_fr`). Trials with fewer than 2 corridor-space frames are skipped. Sessions with fewer than 2 valid trials are skipped. Lick frames outside the neural frame range are dropped. The `TrialStim` placeholder `'stimulus_of_trial'` is handled by falling back to `WallName`.

ii.
```python
ft_tr = np.asarray(beh['ft_trInd'])[:n_fr]
if cols.size < 2:
    continue
if len(valid) < 2:
    print('skip', sess, ...)
    continue
```

iii. The handling is reasonable, though the AI does not filter outlier-length trials.

## 12-a. What are the most time-consuming steps of the code?

i. Loading spike files (89 large .npy files) and casting them to float32. The conversion_full_out.txt shows the full run completed for 76 sessions.

ii.
```python
def load_spk_session(session_id):
    obj = np.load(SPK_DIR / f'{session_id}_neural_data.npy', allow_pickle=True).item()
    return np.concatenate(list(obj['spks']), axis=0).astype(np.float32)
```

iii. I/O bound, consistent with the reference assessment.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop over `build_trial_io` computes inputs and outputs one trial at a time when many operations (position binning, lick vector construction) could be vectorized across all frames then indexed per trial.

ii.
```python
for tr, cols in valid:
    st, en = int(cols[0]), int(cols[-1])
    inp, out = build_trial_io(beh, tr, st, en, ...)
```

iii. The reference pre-computes licking, position, and speed arrays for the full session, then indexes per trial, which is more efficient.

## 12-c. What processing does the code repeat multiple times?

i. The AI reloads the spike file a second time within the main loop to get imaging plane sizes for brain region assignment, after already having loaded and concatenated spikes.

ii.
```python
# First load:
spk = load_spk_session(sess)
# Second load for brain regions:
raw = np.load(SPK_DIR / f'{sess}_neural_data.npy', allow_pickle=True).item()['spks']
group_sizes = [arr.shape[0] for arr in raw]
```

iii. This doubles the I/O cost for each session, significantly increasing total runtime.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes speed quantile edges from `ft_isMoving` and `ft_CorrSpc` filtered frames across all sessions, but then applies these edges to all trial frames including non-moving ones, creating a mismatch. Also, brain region indices are assigned as imaging plane groups which have no scientific meaning and are not used meaningfully by the decoder.

ii.
```python
# Speed edges from filtered frames:
m &= np.asarray(beh['ft_isMoving'], dtype=bool)
# Applied to all trial frames:
speed_bin = np.digitize(speed_seg, speed_edges[1:-1], right=False)
```

iii. The global speed edge computation involves loading behavior for all sessions, adding significant overhead.
