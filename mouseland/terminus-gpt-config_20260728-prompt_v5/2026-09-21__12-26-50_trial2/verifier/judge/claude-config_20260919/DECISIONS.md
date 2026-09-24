# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the master index `beh/Imaging_Exp_info.npy`. Instead it globs every `*.npy`
in `/app/data/beh`, loads each one with `allow_pickle=True`, and keeps any top-level key that
matches a session-ID regex (`MOUSE_YYYY_MM_DD_BLK`, optionally with a `_swapN` suffix). All of
these behavior dictionaries are held in one in-memory index (`{session_id: {'file':…, 'data':…}}`);
because the files are iterated in sorted order and the dict is overwritten, a session listed in
more than one behavior file keeps the **last** file's copy. Neural data is then loaded per session
from `spk/<session_id>_neural_data.npy`, and a session is processed only if that file exists.
The `retinotopy/` directory is **never opened** anywhere in the script or the trajectory.
Net effect: 100 behavior keys → 76 with a matching spike file → 75 converted sessions
(89 spike files exist). The 13 lost sessions are those whose behavior key carries a `_swapN`
suffix (e.g. `DR10_2022_07_30_1_swap1`), so `f'{sid}_neural_data.npy'` never resolves; a 14th
(`TX109_2023_03_27_1`) returned `None` from `convert_session`.

ii.
```python
SESSION_RE = re.compile(r'^[A-Z0-9]+_\d{4}_\d{2}_\d{2}_[0-9]+(?:_swap[0-9]+)?$')

def load_behavior_index():
    beh = {}
    for f in sorted(BEH_DIR.glob('*.npy')):
        obj = np.load(f, allow_pickle=True).item()
        for k, v in obj.items():
            if is_session_key(k):
                beh[k] = {'file': f.name, 'data': v}
    return beh

def load_neural_session(session_id):
    f = SPK_DIR / f'{session_id}_neural_data.npy'
    if not f.exists():
        return None
    mats = np.load(f, allow_pickle=True).item()['spks']
    return np.concatenate(mats, axis=0).astype(np.float32)
```
```python
common_sessions = [sid for sid in sorted(beh.keys())
                   if (SPK_DIR / f'{sid}_neural_data.npy').exists()]
```

iii. From CONVERSION_NOTES.md Step 4: "Behavior files include session-like keys but naive parsing
also yields labels such as `naive`, `sup`, `test1` … Filter behavior dict keys carefully and only
treat true session-ID-like keys as sessions/subjects." The trajectory (step 376) records the
discovery that a given session is not in the file you would guess, so the AI decided to
"build an index of behavior sessions across files" rather than use the experiment index.
No justification is given anywhere for ignoring `Imaging_Exp_info.npy` or `retinotopy/`, and
the 13 silently dropped `_swap` sessions are never mentioned.

## 1-b. How are the data split into subjects?

i. The subject is the first underscore-delimited token of the session ID. Unique subjects are
sorted and `subject_idx` is the index of each session's subject into that list. This yields the
19 mice (`DR10 … VR2`), matching the reference.

ii.
```python
def get_subject(session_id):
    return session_id.split('_')[0]

subjects = sorted({get_subject(sid) for sid in common_sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
data['subject_idx'].append(subject_to_idx[get_subject(sid)])
```

iii. CONVERSION_NOTES.md Step 2 notes that naive parsing of behavior keys gave 25 "subjects"
because cohort keys such as `unsup`/`naive` were being counted, and that "19 neural subjects
[are] confirmed from neural filenames". The regex key filter was introduced to fix this.

## 1-c. How are the data split into sessions?

i. A session is one behavior dictionary key that also has a spike file. The key is
`MOUSE_YYYY_MM_DD_BLK`, i.e. mouse + date + block, same granularity as the reference. Duplicates
across behavior files are resolved by *last write wins* (files iterated in sorted filename order),
whereas the reference keeps the first occurrence in the experiment index. Sessions whose behavior
key has a stimulus-type suffix are lost (see 1-a), and a session with fewer than 2 usable trials
is dropped. Final count: **75 sessions** vs the reference's **89**.

ii.
```python
common_sessions = [sid for sid in sorted(beh.keys())
                   if (SPK_DIR / f'{sid}_neural_data.npy').exists()]
...
for sid in common_sessions:
    converted = convert_session(sid, beh[sid], speed_qs, cat_map, ...)
    if converted is None:
        print('skip_session', sid)
        continue
```
```python
    if len(neural_trials) < 2:
        return None
```

iii. The stated rule is only "sessions with matched behavior and neural data" (README.md).
CONVERSION_NOTES.md Step 10 records: "One session (`TX109_2023_03_27_1`) was skipped by the
converter because `convert_session` returned `None`; this should be revisited later if needed,
but the remaining dataset verified successfully." No justification is offered for the 13 other
missing sessions, and the mismatch between 89 spike files and 75 converted sessions is never
reconciled anywhere in the notes.

## 1-d. How are the data split into trials?

i. Trials come from the per-frame trial label `ft_trInd`. Frames with `NaN` labels are excluded,
the remaining labels are cast to int, and for each unique label the trial is *every* frame with
that label — `ft_CorrSpc` is **not** used. The trial therefore runs from corridor entry all the
way through the 2 m grey space to the end of the traversal (`EndFr`), not just the 4 m textured
corridor. Trials are variable length and nothing is padded. Measured on `TX83_2022_08_31_1`,
27 % of the frames in a trial are grey-space frames and the median trial is 52 frames vs 31 for
the reference's corridor-only definition; over the full dataset mean `T` is 60.2 (reference 32.6).

ii.
```python
valid = ~np.isnan(ft_tr)
tr_ids = np.unique(ft_tr[valid].astype(int))
...
for tr in tr_ids:
    idx = np.where(ft_tr.astype(float) == float(tr))[0]
    if idx.size < 2:
        continue
```

iii. CONVERSION_NOTES.md Step 5 says only "segment trials via `ft_trInd`" and "Align all trials
to trial start / corridor entry, as required by the decoder task". The trajectory (step 377)
lists `ft_CorrSpc` and `ft_GraySpc` among the discovered frame-level fields but the AI never
discusses using them; no justification for including grey-space frames is given.

## 1-e. How are trials filtered based on quality controls?

i. Essentially none. The only trial-level exclusions are (a) trials with fewer than 2 frames, and
(b) sessions that end up with fewer than 2 trials. There is no trial-length outlier rule, no
check on stationarity, and no exclusion of probe/swap trials. Consequently trials up to
**5621 bins (~29 min, 1769 s)** survive into the dataset; the reference removes trials above the
99th percentile of traversal length (238.9 frames, 382 trials dropped). 31,359 trials are written
vs the reference's 37,728.

ii.
```python
    for tr in tr_ids:
        idx = np.where(ft_tr.astype(float) == float(tr))[0]
        if idx.size < 2:
            continue
```
```python
    if len(neural_trials) < 2:
        return None
```

iii. No justification is given. The verification log shows `T … max: 5621` and
`time_since_trial_start: [0.0, 1769.4]`, and CONVERSION_NOTES.md Step 9/Step 10 never comment on
either value. The `--sample` path additionally truncates to `max_trials=20`, but that is a speed
measure, not quality control.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy`, which is a list of three neurons-by-frames
float32 arrays (the three imaging planes). They are concatenated along the neuron axis.
No other neural source is read; in particular `retinotopy/<mouse>_<date>_trans.npz` (`iarea`) is
never loaded.

ii.
```python
mats = np.load(f, allow_pickle=True).item()['spks']
return np.concatenate(mats, axis=0).astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 4/Step 5: "Each neural session file contains `spks` as a list of 3
large 2D float32 matrices, not a flat neuron list … Interpret each `spks` entry as a
neuron-by-time block/partition; final converter should concatenate or otherwise preserve all
neurons while keeping common time axis." Key Decision 1: "Use Suite2p-derived deconvolved
fluorescence traces directly, because methods state analyses are based on deconvolved
fluorescence traces."

## 2-b. How is the `neural` data processed?

i. No processing at all: no dF/F, no deconvolution, no normalisation, no smoothing. The whole
concatenated array is cast to `float32` (it is already `float32`, so this is a redundant full
copy) and each trial is stored as `neural[:, idx].astype(np.float32)`. Trials keep their own
length. The resulting pickle is **369 GB** (the reference stores `float16`).

ii.
```python
return np.concatenate(mats, axis=0).astype(np.float32)
...
neural_trials.append(neural[:, idx].astype(np.float32))
```

iii. "Use deconvolved fluorescence traces as neural activity" (Step 5 mapping table), backed by
the methods quote "All our analyses were based on deconvolved fluorescence traces." No reasoning
is given for the dtype choice or for the file size.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neuron filtering is performed.** Every ROI in every plane is kept (mean 53,114
neurons/session vs the reference's 46,128 after visual-area selection). Because the retinotopy
is never read, `brain_regions` is filled with three invented placeholder labels
`['unknown_block0','unknown_block1','unknown_block2']` and `brain_region_idx` is generated by
splitting the neuron axis into thirds, with a hard-coded `19408` special case for session IDs
containing `2022_07_12_1`. The reported region counts (1,327,822 / 1,327,822 / 1,327,900) are
therefore just the three imaging planes relabelled as brain regions.

ii.
```python
'brain_regions': ['unknown_block0', 'unknown_block1', 'unknown_block2'],
...
data['brain_region_idx'].append(np.concatenate([
    np.full(19408 if '2022_07_12_1' in sid else nneu // 3, 0, dtype=int),
    np.full(19408 if '2022_07_12_1' in sid else nneu // 3, 1, dtype=int),
    np.full(nneu - 2 * (19408 if '2022_07_12_1' in sid else nneu // 3), 2, dtype=int),
]))
```

iii. CONVERSION_NOTES.md Step 3 states "Suite2p cell classification is part of preprocessing.
Need to determine from data/code whether only accepted cells are included in exported `spks`
arrays or whether additional filtering is needed" — this question is never answered. Trajectory
step 400 explicitly acknowledges "brain regions are placeholders", but the AI proceeded anyway
and never revisited it. The strings `iarea` and `retinotopy` appear **zero** times in the entire
852-step trajectory.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment event is corridor entry / trial start. Each trial's neural matrix is
`neural[:, idx]` where `idx` are exactly the frames labelled with that trial, in order, so the
first column is the first imaged frame after corridor entry. Trials are variable length, nothing
is padded or truncated, `off_start = 0.0` and `off_end = None`. Before slicing, the neural array
and every frame-level behavior stream are truncated to a common length
`T = min(n_neural_frames, len(ft), len(ft_trInd), …)`.

ii.
```python
T = min(neural.shape[1], len(ft), len(ft_tr), len(ft_pos), len(ft_speed), len(ft_wall))
neural = neural[:, :T]
...
idx = np.where(ft_tr.astype(float) == float(tr))[0]
...
neural_trials.append(neural[:, idx].astype(np.float32))
```
```python
'temporal_alignment_event': 'trial start / corridor entry',
'off_start': 0.0,
'off_end': None,
```

iii. Step 5 Key Decision 3: "Align all trials to trial start / corridor entry, as required by the
decoder task; derive time-to-cue from cue timing relative to this alignment." Trajectory step 377:
"the neural time dimension for session DR10_2022_07_12_1 is 31707 while behavior frame arrays are
31709, indicating near-match with a small offset that must be handled carefully … trim neural and
frame-level behavior arrays to a common length."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling. The imaging frames are the bins. `time_bin_size` is derived from
the data as the median inter-frame interval of the **first** session, converted from MATLAB
datenum days to ms: **314.69 ms** (≈ the 3.17 Hz imaging rate the reference hard-codes as
315.5 ms). The same value is reported for the whole dataset.

ii.
```python
'time_bin_size': float(np.median(np.diff(np.asarray(
    beh[common_sessions[0]]['data']['ft'])))) * 24 * 3600 * 1000 if common_sessions else None,
```

iii. Not explicitly justified in the notes; implicit in Step 5's decision to use the frame-level
`ft_*` arrays directly, which "correspond to neural timepoints" (trajectory step 377). Deriving
the bin size from `ft` rather than a constant is a data-driven choice.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From the trial-level `SoundTime` and `Trial_start_time` (both MATLAB datenums), plus the
frame-time vector `ft`. The reference instead uses `SoundFr` interpolated onto the frame-time
axis. Spot-checking `TX83_2022_08_31_1` trials 0/1/5/100, the two constructions agree to within
0.11–0.26 s, i.e. under one 315 ms bin.

ii.
```python
sound_times = np.asarray(b['SoundTime'])
trial_start_times = np.asarray(b['Trial_start_time'])
...
if tr < len(sound_times):
    cue_rel = float((sound_times[tr] - trial_start_times[tr]) * 24 * 3600)
```

iii. Step 5 mapping table: "sound cue time relative to trial start → input[time_to_sound_cue],
continuous time-varying value per bin: cue_time - current_time … Need exact source field from
behavior dict." Trajectory step 377 lists `SoundTime`/`SoundPos` among the trial-level arrays
found.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The cue time is expressed relative to the trial start time and converted from days to seconds
(`×24×3600`); the per-bin elapsed time within the trial is subtracted. The result is **positive
before the cue and negative after**, matching the reference's sign convention for
*time to* the cue. It is broadcast over all bins of the trial as a float32 row. If the trial index
exceeds the length of `SoundTime` the row is left as `NaN`. Observed range over the full dataset
is [-1767.5, 403.4] s — far wider than the reference's [-72.2, 73.4] s, entirely because of the
unfiltered multi-minute trials (1-e).

ii.
```python
times = (ft[idx] - t0) * 24 * 3600
time_to_cue = np.full(idx.size, np.nan, dtype=np.float32)
if tr < len(sound_times):
    cue_rel = float((sound_times[tr] - trial_start_times[tr]) * 24 * 3600)
    time_to_cue = (cue_rel - times).astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 10: "Initial time conversion bug: input times were left in days;
fixed by multiplying datenum differences by `24*3600`." The `cue_time - current_time` ordering is
specified in the Step 5 mapping table.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated on `ft[idx]`, i.e. exactly the same frame indices used to slice the neural
matrix, so it has the same number of bins as the trial's neural array and is sample-for-sample
aligned.

ii.
```python
idx = np.where(ft_tr.astype(float) == float(tr))[0]
t0 = ft[idx[0]]
times = (ft[idx] - t0) * 24 * 3600
...
inp = np.vstack([time_to_cue, day_val, time_since, rew_avail])
neural_trials.append(neural[:, idx].astype(np.float32))
```

iii. All streams share the imaging-frame grid after the common-length truncation
(trajectory step 377); the AI therefore indexes every stream with the same `idx`.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. **Not from the data at all** — from the *behavior file name* that the session was last found
in. The regex looks for `before_learning`/`before_grating` → 0, `after_learning`/`after_grating`
→ 1, else `train(\d+)`, else `test(\d+)`, else 0. The session date, which is present in the
session ID and is what the reference uses, is ignored.

ii.
```python
def get_day_value(session_id, beh_file):
    name = beh_file.lower()
    if 'before_learning' in name or 'before_grating' in name:
        return 0.0
    if 'after_learning' in name or 'after_grating' in name:
        return 1.0
    m = re.search(r'train(\d+)', name)
    if m:
        return float(m.group(1))
    m = re.search(r'test(\d+)', name)
    if m:
        return float(m.group(1))
    return 0.0
```

iii. Step 5 mapping table: "day/session training index … session ordering from filenames /
condition files | Must define consistent day ordering within subject." Trajectory step 400
concedes "day_of_training is a heuristic derived from filenames" and flags it as a caveat to be
"document[ed] … later"; it was never revisited or justified further.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The scalar from the filename regex is broadcast across every bin of every trial of the session
as a float32 row (per-trial constant, stored time-varying). The full-dataset range is
**[0.0, 2.0]**, against the reference's [0.0, 7.0]. Crucially the value is *not* a day count and
is not monotonic within a mouse: from the per-session input ranges in
`verification_full_out.txt`, DR10's five sessions carry days 0, 1, 0, 1, 2 — the same "day"
appears twice for the same animal, and the ordering does not follow the recording dates.

ii.
```python
day_val = np.full(idx.size, get_day_value(session_id, beh_entry['file']), dtype=np.float32)
...
inp = np.vstack([time_to_cue, day_val, time_since, rew_avail])
```

iii. As in 4-a: the only stated reason is that the condition is encoded in the file name. The
requirement the AI wrote for itself in Step 5 — "Must define consistent day ordering within
subject" — is not satisfied by the implementation, and this is never checked.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `ft` alone: the timestamp of each imaging frame minus the timestamp of the trial's first
frame (`t0 = ft[idx[0]]`). The reference instead interpolates the fractional `StartFr` onto the
frame-time axis; the difference is the sub-frame offset between corridor entry and the first
imaged frame (0–0.26 s in the trials spot-checked, i.e. under one bin).

ii.
```python
t0 = ft[idx[0]]
times = (ft[idx] - t0) * 24 * 3600
```

iii. Step 5 mapping table: "trial start / corridor entry time → input[time_since_trial_start],
continuous time-varying ramp aligned to trial start … Alignment event for all trials."

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Datenum difference × 86400 → seconds, stored as float32. It starts at exactly 0.0 for every
trial by construction and increases monotonically. Full-dataset range [0.0, 1769.4] s versus the
reference's [0.0, 74.8] s — again a consequence of the missing trial-length filter (1-e).

ii.
```python
times = (ft[idx] - t0) * 24 * 3600
...
time_since = times.astype(np.float32)
```

iii. Same as 5-a; Step 10 records the day→second unit fix that applies here as well.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed on the same `idx` frame indices as the neural slice, so it is bin-for-bin aligned and
has the trial's length.

ii.
```python
idx = np.where(ft_tr.astype(float) == float(tr))[0]
times = (ft[idx] - t0) * 24 * 3600
neural_trials.append(neural[:, idx].astype(np.float32))
```

iii. All streams are indexed with the same `idx` after the common-length truncation.
CONVERSION_NOTES.md Step 10 check 3 records the sanity check: "first trial
`time_since_trial_start` begins `[0.0, 0.2885, 0.6054, 0.9304, 1.2337]`, consistent with raw `ft`
timing converted from MATLAB datenums to seconds" — steps of ~0.3 s, the frame period.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From the trial-level boolean `isRew`, the same variable the reference uses.

ii.
```python
is_rew = np.asarray(b['isRew']).astype(int)
...
rew_avail = np.full(idx.size, int(is_rew[tr]) if tr < len(is_rew) else 0, dtype=np.float32)
```

iii. Step 5 mapping table: "rewarded corridor / reward availability → input[reward_availability]
… Interpret as 1 for rewarded corridor, 0 otherwise."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast to int, broadcast across all bins of the trial as float32 (per-trial constant stored
time-varying). Out-of-range trial indices default to 0. Range [0.0, 1.0] across the dataset, so
both values occur — consistent with the reference.

ii.
```python
rew_avail = np.full(idx.size, int(is_rew[tr]) if tr < len(is_rew) else 0, dtype=np.float32)
inp = np.vstack([time_to_cue, day_val, time_since, rew_avail])
```

iii. No further justification needed or given; the sample-session selection was deliberately
changed to pick sessions with rewarded trials so this input would vary (CONVERSION_NOTES.md
Step 10: "initial sample sessions had no licks/rewarded trials; fixed by selecting sample sessions
with richer lick/reward variation").

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From the trial-level `WallName`, with the frame-level `ft_WallID` at the trial's first frame as
an (unreachable in practice) fallback. Same source variable as the reference.

ii.
```python
wall_names = np.asarray(b['WallName']).astype(str)
...
trial_wall = wall_names[tr] if tr < len(wall_names) else ft_wall[idx[0]]
stim_cat = np.full(idx.size, global_cat_map.get(str(trial_wall), 0), dtype=np.int64)
```

iii. Step 5 mapping table: "`WallName` / `stim_id` → output[visual_stimulus_category], categorical
per-trial label … Likely one category per trial."

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The **raw** wall names are used as the categories. The set of unique names is collected over
all converted sessions, sorted alphabetically, and mapped to 0…N-1. This gives **13 classes**:
`circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1,
wood2, wood5`. Crops of the same texture (`circle1` vs `circle2`) and spatial-swap variants
(`leaf1` vs `leaf1_swap1`) are kept as *distinct categories*, so the reference's four base
textures (circle / leaf / rock / wood) are never formed. The per-trial value is broadcast across
the trial's bins. Chance level becomes 0.0769 instead of 0.25, and because most sessions contain
only 2–4 of the 13 names, part of the decoding problem reduces to session identity.

ii.
```python
def session_category_map(all_wall_names):
    cats = sorted(str(x) for x in np.unique(all_wall_names))
    return {c: i for i, c in enumerate(cats)}, cats
...
cat_map, cat_names = session_category_map(np.array(all_wall))
...
'output_values': [cat_names, ['no_lick', 'lick'], ...]
```

iii. Step 4 discrepancy table: "Methods describe four base textures (circle, leaf, rock and brick)
and grating stimuli for subset mice … converter must preserve actual per-session categories rather
than assume all global categories appear in every session." Step 5 Key Decision 4: "Use per-trial
visual category from `WallName` or `stim_id`, preserving actual session-specific categories rather
than imposing a global fixed subset." The AI's own Step 3 notes record the paper's four-texture
description but the conflict is never resolved.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the (fractional) imaging-frame number of every lick in the session — the same
variable as the reference.

ii.
```python
lick_fr = np.asarray(b.get('LickFr', []))
if lick_fr.size:
    lick_fr = lick_fr.astype(int)
```

iii. Step 5 mapping table originally planned to use `LickTime`/`LickTrind`; trajectory step 377
found the frame-indexed `LickFr` and the code uses it because it is already on the neural grid.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A session-length binary vector is zeroed, lick frame numbers are truncated to int, licks
outside `[0, T)` are discarded, and the corresponding bins are set to 1. A bin is 1 if at least
one lick falls in it. Sessions with no licks give an all-zero vector. Resulting distribution
`{no_lick 0.974, lick 0.026}` (reference: 0.959 / 0.041 — the difference comes from the extra
grey-space and stalled-trial bins in the denominator).

ii.
```python
lick = np.zeros(T, dtype=np.int64)
lick_fr = np.asarray(b.get('LickFr', []))
if lick_fr.size:
    lick_fr = lick_fr.astype(int)
    lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < T)]
    lick[lick_fr] = 1
...
lick_out = lick[idx].astype(np.int64)
```

iii. Step 5 Key Decision 5: "Convert lick timestamps to binary time bins for each trial."
Planned sanity check: "Compare one trial's binned licking output against raw
`LickTime`/`LickTrind` events from the original behavior file" (the checkbox is left unticked in
the notes).

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` already indexes imaging frames, so the flag vector shares the neural grid; the trial's
values are `lick[idx]`, the same indices used for the neural slice.

ii.
```python
lick_out = lick[idx].astype(np.int64)
out = np.vstack([stim_cat, lick_out, pos_bin, speed_bin])
neural_trials.append(neural[:, idx].astype(np.float32))
```

iii. Same frame-grid argument as the other streams; the vector is built on the truncated length
`T` so it cannot run past the imaged frames.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From the frame-level `ft_Pos`, the same variable as the reference.

ii.
```python
ft_pos = np.asarray(b['ft_Pos'])
...
pos_bin = np.clip((ft_pos[idx] // 1.0).astype(int), 0, 3).astype(np.int64)
```

iii. Step 5 mapping table: "corridor position → output[position_bin], discretize into 4 equal 1-m
spatial bins … Need exact position variable from behavior data." Trajectory step 377 identified
`ft_Pos`/`ft_PosCum` as the frame-level position arrays.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Integer-divide `ft_Pos` by 1.0, then clip to [0, 3]. **The units are wrong**: `ft_Pos` is in
decimetres (the session fields state `Texture_Length = 40`, `Gray_Space_length = 20`,
`Corridor_Length = 60`, and `ft_Pos` ranges 0–60), so dividing by 1 produces 0.1 m bins, not 1 m
bins. The reference divides by 10. No unit check was performed.

ii.
```python
pos_bin = np.clip((ft_pos[idx] // 1.0).astype(int), 0, 3).astype(np.int64)
```

iii. Step 5 Key Decision 6: "Discretize corridor position into 4 equal-length 1 m bins, matching
the task specification." No evidence anywhere in the notes or trajectory that the AI checked what
units `ft_Pos` is in.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Bin boundaries are effectively 0.1 m, 0.2 m, 0.3 m with everything beyond falling into bin 3,
instead of the intended 1 m / 2 m / 3 m. The resulting full-dataset distribution is
`{bin1 0.036, bin2 0.020, bin3 0.020, bin4 0.924}` — 92.4 % of all bins are in the last class,
against the reference's near-uniform `{0.254, 0.242, 0.247, 0.257}`. The `output_values` labels
are the uninformative `['bin1','bin2','bin3','bin4']`. This is the single largest quantitative
defect in the conversion, and it drives position decoding down to 0.374 balanced accuracy.

ii.
```python
pos_bin = np.clip((ft_pos[idx] // 1.0).astype(int), 0, 3).astype(np.int64)
...
'output_values': [cat_names, ['no_lick', 'lick'], ['bin1', 'bin2', 'bin3', 'bin4'], ...]
```

iii. The skew was observed but not diagnosed. Trajectory step 400: "position remains heavily
concentrated in bin4". CONVERSION_NOTES.md Step 10 check 4: "first trial position-bin values are
sensible discrete integers, beginning `[0, 2, 2, 2, 2]`" — treated as a pass. Step 12: "retained
current representation because it remains above chance and matches the requested 4 equal 1 m
bins", and the skew is attributed to "heavy occupancy of the final corridor bin" rather than to
the unit error.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is one value per imaging frame and is truncated to the common length `T`; the trial's
values are taken at the same `idx` as the neural slice, so alignment is exact.

ii.
```python
T = min(neural.shape[1], len(ft), len(ft_tr), len(ft_pos), len(ft_speed), len(ft_wall))
ft_pos = ft_pos[:T]
...
pos_bin = np.clip((ft_pos[idx] // 1.0).astype(int), 0, 3).astype(np.int64)
neural_trials.append(neural[:, idx].astype(np.float32))
```

iii. Same frame-grid reasoning as all other streams.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From the frame-level `ft_RunSpeed`, the same variable as the reference.

ii.
```python
ft_speed = np.asarray(b['ft_RunSpeed'])
...
speed_bin = apply_speed_bins(ft_speed[idx], speed_qs)
```

iii. Step 5 mapping table: "running speed → output[running_speed_bin], discretize into 4 quantile
bins over all valid samples … Use global quartiles over included samples."

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Three global thresholds are computed once, up front, as `np.quantile(all_speeds,
[0.25, 0.5, 0.75])` where `all_speeds` is the concatenation of **every finite `ft_RunSpeed`
sample of every session** — including frames that never make it into a trial. The thresholds come
out as `[0.0, 9.0995, 31.0195]`. The same three thresholds are applied to all sessions (the
reference computes rank-based quartiles per session, over only the kept frames).

ii.
```python
def discretize_speed(all_speeds):
    qs = np.quantile(all_speeds, [0.25, 0.5, 0.75])
    return qs
...
for sid in common_sessions:
    b = beh[sid]['data']
    all_speed.extend(np.asarray(b['ft_RunSpeed'])[
        np.isfinite(np.asarray(b['ft_RunSpeed']))].tolist())
speed_qs = discretize_speed(np.asarray(all_speed))
```

iii. Step 5 Key Decision 7: "Discretize running speed into 4 bins based on quartiles of valid
included samples." No justification is given for pooling across sessions rather than per session,
and the "included samples" qualifier is not honoured by the implementation.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize(x, qs, right=False)`, so bin 0 is `x < 0`, bin 1 is `0 <= x < 9.1`, bin 2 is
`9.1 <= x < 31.0`, bin 3 is `x >= 31.0`. Because ~44 % of frames sit at exactly zero speed, the
25th percentile *is* 0 and `digitize(0, [0, …])` returns 1: class 0 collects only the
(rare) negative speeds. The realised distribution is `{q1 0.093, q2 0.407, q3 0.250, q4 0.249}`,
which does not satisfy the instruction's "4 bins, each corresponding to 25 % of the data"
(reference: 0.2500 / 0.2500 / 0.2500 / 0.2500). Labels are `['q1','q2','q3','q4']`.

ii.
```python
def apply_speed_bins(x, qs):
    return np.digitize(x, qs, right=False).astype(np.int64)
...
speed_bin = apply_speed_bins(ft_speed[idx], speed_qs)
```

iii. The intent is stated as quartile binning (Step 5 Key Decision 7). The tie mass at zero is
never discussed and the realised fractions printed in `verification_full_out.txt` are never
compared against the intended 25 % each.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is one value per imaging frame, truncated to the common length `T`, and indexed
with the same `idx` as the neural slice — exact bin-for-bin alignment.

ii.
```python
ft_speed = ft_speed[:T]
...
speed_bin = apply_speed_bins(ft_speed[idx], speed_qs)
out = np.vstack([stim_cat, lick_out, pos_bin, speed_bin])
neural_trials.append(neural[:, idx].astype(np.float32))
```

iii. Same frame-grid reasoning as all other streams.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several defensive measures, plus some silent losses:
- **Behavior longer than imaging**: all frame-level streams and the neural matrix are truncated to
  `T = min(...)` over every stream, which subsumes the reference's `[:nfr]` cut.
- **NaN trial labels**: frames with `NaN` in `ft_trInd` are excluded from the trial-ID list.
- **Licks past the last imaged frame**: filtered with `(lick_fr >= 0) & (lick_fr < T)`.
- **Missing `LickFr`**: `b.get('LickFr', [])` defaults to empty.
- **Trial index beyond a per-trial array**: guarded with `if tr < len(...)`, defaulting to 0 /
  `NaN` / the frame-level wall ID.
- **Degenerate trials/sessions**: trials with <2 frames and sessions with <2 trials are dropped.
- **Missing spike file**: the session is silently skipped — this is what loses the 13 `_swap`
  sessions, because the behavior key does not equal the spike-file stem.
- A crash on sessions lacking `BefCueFr`/`AftCueFr` was fixed by deleting those (unused) lookups.

ii.
```python
T = min(neural.shape[1], len(ft), len(ft_tr), len(ft_pos), len(ft_speed), len(ft_wall))
...
valid = ~np.isnan(ft_tr)
tr_ids = np.unique(ft_tr[valid].astype(int))
...
lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < T)]
...
rew_avail = np.full(idx.size, int(is_rew[tr]) if tr < len(is_rew) else 0, dtype=np.float32)
trial_wall = wall_names[tr] if tr < len(wall_names) else ft_wall[idx[0]]
```

iii. CONVERSION_NOTES.md Step 10 "Issues Found and Resolved": "Full conversion crash on `BefCueFr`:
some sessions lacked `BefCueFr`/`AftCueFr`; fixed by removing those unused required lookups" and
"One session (`TX109_2023_03_27_1`) was skipped … this should be revisited later if needed."
Trajectory step 377 justifies the common-length truncation from the observed 31707 vs 31709
mismatch.

## 12-a. What are the most time-consuming steps of the code?

i. The dominant costs are (1) reading and concatenating the 89 spike files (~405 GB on disk) —
`np.concatenate` plus a redundant `.astype(np.float32)` means two full copies of a
~50,000 × 30,000 array per session; (2) the per-trial fancy-index `neural[:, idx]`, executed
~420 times per session on that array, each producing a gather over the full neuron axis; and
(3) writing the 369 GB output pickle. There is **no timing instrumentation anywhere** in the
script, despite the instructions asking for it, and the Step 7 "Run Time Estimates" tables in
CONVERSION_NOTES.md were left as empty template rows.

ii.
```python
mats = np.load(f, allow_pickle=True).item()['spks']
return np.concatenate(mats, axis=0).astype(np.float32)   # already float32: redundant copy
...
neural_trials.append(neural[:, idx].astype(np.float32))  # ~420 gathers per session
...
with open(args.outpicklefile, 'wb') as f:
    pickle.dump(data, f)                                  # 369 GB, default protocol
```

iii. CONVERSION_NOTES.md Step 6: "Full-session neural slicing is expensive due to very large
neuron-by-time matrices" and "Initial sample mode was too slow when converting all trials";
the speed-ups listed are only "Added `max_trials=20` behavior for `--sample` mode" and
"richer sample-session selection", neither of which affects the full run.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-segmentation loop. `np.where(ft_tr.astype(float) == float(tr))[0]` rescans the whole
~30,000-element frame index once per trial — and `ft_tr.astype(float)` allocates a fresh copy of
that array on *every* iteration (≈420 copies per session), which is pure waste since `ft_tr` is
already float. One `np.argsort`/`np.split` pass would group all frames by trial at once. The
reference has the same one-scan-per-trial structure but without the per-iteration cast.
Secondly, `all_speed.extend(... .tolist())` builds a Python list of ~2.6 M floats before
`np.asarray`, where `np.concatenate` over the arrays would do.

ii.
```python
for tr in tr_ids:
    idx = np.where(ft_tr.astype(float) == float(tr))[0]
```
```python
all_speed.extend(np.asarray(b['ft_RunSpeed'])[
    np.isfinite(np.asarray(b['ft_RunSpeed']))].tolist())
```

iii. Not identified in CONVERSION_NOTES.md; the only inefficiency the AI names is neural slicing.

## 12-c. What processing does the code repeat multiple times?

i. `ft_tr.astype(float)` and `float(tr)` are recomputed inside every trial iteration.
`np.asarray(b['ft_RunSpeed'])` is evaluated twice in the same expression in the pre-pass.
`load_behavior_index()` eagerly loads **all** behavior files and holds every session's behavior
dict in memory for the whole run, and the wall-name/speed pre-pass then walks that entire
structure a second time. Each per-trial array is `.astype()`-cast again even when it is already
the target dtype (`neural[:, idx].astype(np.float32)` on a float32 array).

ii.
```python
    for tr in tr_ids:
        idx = np.where(ft_tr.astype(float) == float(tr))[0]
```
```python
    all_speed.extend(np.asarray(b['ft_RunSpeed'])[np.isfinite(np.asarray(b['ft_RunSpeed']))].tolist())
```

iii. Not identified or justified anywhere in CONVERSION_NOTES.md.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items:
- `ft_wall = np.asarray(b['ft_WallID']).astype(str)` converts and truncates a ~30,000-element
  `<U30` array per session, but it is only ever read as a fallback `ft_wall[idx[0]]` that cannot
  trigger (every trial index is within `WallName`).
- `.astype(np.float32)` on already-float32 spike data (twice: once at load, once per trial slice).
- ~27 % of every trial's bins are grey-space frames that lie outside the 4 m corridor the position
  output is defined on, and the unfiltered multi-minute stationary trials add thousands more bins
  of constant labels — all of which are stored, inflating the pickle to 369 GB.
- No neuron curation, so ~13 % more neurons than the reference are stored per session and the
  fabricated `brain_region_idx` carries no usable information for downstream region analyses.
- `--show-processing` is declared in the argument parser and then never used: the required
  per-step verification plots are not produced at all.

ii.
```python
ft_wall = np.asarray(b['ft_WallID']).astype(str)   # only used as an unreachable fallback
...
ap.add_argument('--show-processing', action='store_true')   # parsed, never referenced
```
```python
'brain_regions': ['unknown_block0', 'unknown_block1', 'unknown_block2'],
```

iii. Not identified in CONVERSION_NOTES.md. The notes' Step 7 "Processing Plots Review" section is
left as the empty template placeholder, consistent with no plots having been generated.
