# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent glob-loads every `Beh_*.npy`, merges duplicate behavioral views into one recording, then loads the corresponding spike and retinotopy file per recording. It does not use `Imaging_Exp_info.npy` as the master index.

ii.
```python
for f in sorted(glob.glob(os.path.join(ROOT, 'beh', 'Beh_*.npy'))):
    B = np.load(f, allow_pickle=True).item()
blocks = np.load(spkfile, allow_pickle=True).item()['spks']
ret = np.load(os.path.join(ROOT, 'retinotopy', '%s_%s_trans.npz' % (mouse, date)))
```

iii. The trajectory says this survey found 99 behavioral keys which collapse to exactly 89 unique recordings, matching the 89 spike files one-to-one. The agent chose this route to recover and merge swap views.

## 1-b. How are the data split into subjects?

i. The mouse is parsed as the first underscore-separated component of each recording name. Subjects are accumulated in first-session order, and each session gets the corresponding index.

ii.
```python
mouse = name.split('_')[0]
if mouse not in subjects:
    subjects.append(mouse)
data['subject_idx'].append(subjects.index(mouse))
```

iii. The agent observed that recording names encode mouse, date, and block, and reported 19 mice across 89 recordings.

## 1-c. How are the data split into sessions?

i. A session is the base `mouse_year_month_day_block` recording. `_swap1` and `_swap2` suffixes and appearances in multiple behavior files are deduplicated into that base key.

ii.
```python
parts = key.split('_')
base = '_'.join(parts[:5]) if parts[-1].startswith('swap') else key
d = recs.setdefault(base, {'beh': None, 'walls': {}, 'exps': []})
```

iii. The trajectory established that swap entries are different behavioral views of the same imaging recording and that deduplication gives the exact 89 spike-file sessions.

## 1-d. How are the data split into trials?

i. Frames are grouped by integer `ft_trInd`, but only frames both in `ft_CorrSpc` and moving (`ft_move > 0`) are retained. Thus a trial is a variable-length, potentially noncontiguous sequence of running frames in the textured corridor.

ii.
```python
tr = beh['ft_trInd'][:nfr]
corr = beh['ft_CorrSpc'][:nfr].astype(bool)
moving = beh['ft_move'][:nfr] > 0
ok = corr & moving & ~np.isnan(tr)
for i in idx:
    groups[int(tr[i])].append(i)
```

iii. The agent cited the paper's statement that analyses considered only running timepoints and connected `ft_CorrSpc` to the requested 4 m corridor.

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped if they have fewer than five retained running frames or if their `WallName` lacks a merged canonical stimulus mapping (notably `circle3`). There is no long-trial percentile filter.

ii.
```python
if len(fr) < MIN_FRAMES_PER_TRIAL or wall not in walls:
    continue
```

iii. The agent reported excluding 309 `circle3` trials because the authors assigned them no canonical role, and used five running frames as a minimum usable sequence length.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from the per-plane arrays in `['spks']` of each session's neural file. `iarea` from the retinotopy file supplies neuron area assignments and determines eligible neurons.

ii.
```python
blocks = np.load(spkfile, allow_pickle=True).item()['spks']
iarea = ret['iarea']
area = np.array([IAREA_TO_AREA.get(int(a), -1) for a in iarea])
```

iii. The agent identified `spks` as Suite2p deconvolved traces and verified that plane neuron counts equal the length of `iarea`.

## 2-b. How is the `neural` data processed?

i. Eligible neurons are copied from the plane arrays, randomly limited to 2,000 per session, z-scored per neuron over the whole recording, sliced to each trial's retained frames, and stored as float32.

ii.
```python
sel = np.sort(rng.choice(valid, MAX_NEURONS, replace=False))
mu = spk.mean(axis=1, keepdims=True)
sd = spk.std(axis=1, keepdims=True)
spk = (spk - mu) / sd
neural_s.append(spk[:, fr].astype(np.float32))
```

iii. The agent linked z-scoring to `utils.get_kfold_reward_response` and capped neurons because the raw spike files total about 405 GB and the decoder's SVD also caps at 2,000.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons without one of four mapped retinotopic areas are excluded; if more than 2,000 remain, a deterministic random subset is retained.

ii.
```python
valid = np.where(area >= 0)[0]
rng = np.random.default_rng(SEED + si)
if len(valid) > MAX_NEURONS:
    sel = np.sort(rng.choice(valid, MAX_NEURONS, replace=False))
```

iii. The agent interpreted `iarea` values -1 and 7 as unassigned and described the 2,000-neuron cap as a storage/decoder practicality.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is described as corridor entry, but the stored sequence begins at the first retained moving frame in `ft_CorrSpc`; stopped frames are removed throughout. Trials remain variable length and are not padded.

ii.
```python
ok = corr & moving & ~np.isnan(tr)
neural_s.append(spk[:, fr].astype(np.float32))
'temporal_alignment_event': 'trial start = entry into the virtual corridor'
```

iii. The agent reasoned that the textured-corridor window matches the four position bins and that a running-only sequence should use a running-frame clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Each imaging frame is one bin; no rebinning or resampling is applied. The metadata uses the median observed frame interval across sessions, approximately 314.7 ms.

ii.
```python
bin_ms.append(np.median(np.diff(ft)) * 1000.0)
'time_bin_size': float(np.median(bin_ms))
```

iii. The trajectory measured a nearly uniform acquisition rate of about 3.177 Hz and retained that native grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from the trial's `SoundFr`, the retained frame indices, and the session median frame interval inferred from `ft`.

ii.
```python
bin_s = float(np.median(np.diff(ft)))
rank_cue = float(np.searchsorted(fr, beh['SoundFr'][t])) * bin_s
```

iii. The agent initially tried absolute timestamps, investigated extreme values caused by long stopped intervals, then chose a retained-running-frame clock.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Retained frames are assigned accumulated running times `k`; cue time is the insertion rank of `SoundFr` in those frames times the bin duration. The stored value is `k - rank_cue`, i.e. time since cue (negative before cue), despite the name “time to sound cue.”

ii.
```python
k = np.arange(len(fr), dtype=np.float32) * bin_s
rank_cue = float(np.searchsorted(fr, beh['SoundFr'][t])) * bin_s
inp[0] = k - rank_cue
```

iii. The agent wanted the time variable to exclude invisible stopped intervals and reported that it then crossed zero within trials with bounded values.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It has one value for every retained neural column and uses the cue's rank among the exact same retained frame indices.

ii.
```python
k = np.arange(len(fr), dtype=np.float32) * bin_s
neural_s.append(spk[:, fr].astype(np.float32))
inp = np.empty((4, len(fr)), dtype=np.float32)
```

iii. The agent explicitly chose the clock of the sequence actually passed to the decoder.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the date encoded in each recording name and the earliest recording date for that mouse.

ii.
```python
def date_of(name):
    p = name.split('_')
    return datetime.date(int(p[1]), int(p[2]), int(p[3]))
first[m] = min(first.get(m, date_of(name)), date_of(name))
```

iii. The agent interpreted “day” literally as elapsed calendar days from a mouse's first recording.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Calendar-day difference from the mouse's first recording is computed as a float and broadcast to every bin of every trial in the session.

ii.
```python
day = float((date_of(name) - first[mouse]).days)
inp[1] = day
```

iii. The trajectory did not give further justification beyond calling this “days elapsed since each mouse's first recording.”

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It uses only the count of retained frames and the median frame interval from `ft`; `StartFr` is not used directly.

ii.
```python
bin_s = float(np.median(np.diff(ft)))
k = np.arange(len(fr), dtype=np.float32) * bin_s
```

iii. Because the first retained frame was treated as corridor entry on the running-only clock, the agent made it time zero.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. It is a regularly spaced sequence starting at zero, counting only retained running frames.

ii.
```python
k = np.arange(len(fr), dtype=np.float32) * bin_s
inp[2] = k
```

iii. The agent sought to avoid wall-clock jumps during removed pauses and make the input clock match the stored sequence.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Its length equals the number of retained neural columns and element `j` corresponds to neural column `j`.

ii.
```python
neural_s.append(spk[:, fr].astype(np.float32))
inp = np.empty((4, len(fr)), dtype=np.float32)
inp[2] = k
```

iii. The agent described this as accumulated running time on the neural sequence's clock.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes directly from trial-level `isRew`.

ii.
```python
inp[3] = float(beh['isRew'][t])
```

iii. The agent treated `isRew` as the rewarded-corridor flag and did not derive it from behavior.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The scalar is cast to float and broadcast over all retained frames in the trial.

ii.
```python
inp = np.empty((4, len(fr)), dtype=np.float32)
inp[3] = float(beh['isRew'][t])
```

iii. No additional processing was considered necessary.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from each trial's `WallName` and a recording-specific mapping made by merging `UniqWalls` with `stim_id` across all duplicate views.

ii.
```python
for wall, sid in zip(beh['UniqWalls'], beh['stim_id']):
    if not np.isnan(sid):
        d['walls'][str(wall)] = int(sid)
wall = str(beh['WallName'][t])
```

iii. The agent discovered that `TrialStim` contains a placeholder in swap views and that physical wall names differ across mice, so it used the canonical author-provided role mapping.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The merged `stim_id` is used as one of seven categories (`circle1`, `circle2`, `leaf1`, `leaf2`, `leaf3`, `leaf1_swap1`, `leaf1_swap2`) and is broadcast across the trial. Unmapped walls are excluded.

ii.
```python
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2']
out[0] = walls[wall]
```

iii. The agent argued these canonical roles provide consistent cross-mouse labels and avoid inventing a placeholder category.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from session-level lick frame numbers in `LickFr`.

ii.
```python
lf = np.floor(beh['LickFr']).astype(int)
```

iii. The agent verified that the many all-zero sessions were genuine unrewarded recordings rather than a mapping bug.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Fractional lick frames are floored to integer frames, out-of-range entries are removed, and the corresponding frame flags are set true. Multiple licks in a frame remain one binary event.

ii.
```python
lick = np.zeros(nfr, dtype=bool)
lf = lf[(lf >= 0) & (lf < nfr)]
lick[lf] = True
out[1] = lick[fr].astype(np.int64)
```

iii. The requested output is binary, so the agent converted lick events to a per-frame indicator.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick indicator is indexed by the exact same retained frame array as the neural columns.

ii.
```python
neural_s.append(spk[:, fr].astype(np.float32))
out[1] = lick[fr].astype(np.int64)
```

iii. Both data streams use imaging-frame indices.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It comes from per-frame `ft_Pos`.

ii.
```python
pos = beh['ft_Pos'][:nfr]
```

iii. The agent measured the textured corridor as 0–40 decimeters, matching four 1 m bins.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is divided by 10 decimeters, converted to integer, and clipped to category 0–3.

ii.
```python
out[2] = np.clip((pos[fr] / 10.0).astype(int), 0, CORRIDOR_BINS - 1)
```

iii. This implements the requested four equal 1 m spatial bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are 0, 10, 20, 30, and 40 dm, yielding `[0,1)`, `[1,2)`, `[2,3)`, and `[3,4]` m categories; clipping catches boundaries/outliers.

ii.
```python
CORRIDOR_BINS = 4
np.clip((pos[fr] / 10.0).astype(int), 0, CORRIDOR_BINS - 1)
```

iii. The agent tied these fixed thresholds directly to the task's four equal-length bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is selected using the same retained frame indices as the neural data.

ii.
```python
neural_s.append(spk[:, fr].astype(np.float32))
out[2] = np.clip((pos[fr] / 10.0).astype(int), 0, 3)
```

iii. Position and neural activity already share the imaging-frame grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from per-frame `ft_RunSpeed` across eligible retained frames from all recordings.

ii.
```python
spd = beh['ft_RunSpeed'][:nfr]
speeds.append(spd[fr])
```

iii. The agent used the behavior's directly supplied running speed.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. All eligible running-frame speeds are concatenated in a behavior-only first pass; global 25th, 50th, and 75th percentiles are calculated and then used for all sessions.

ii.
```python
speeds = np.concatenate(speeds)
speed_edges = np.percentile(speeds, [25, 50, 75])
out[3] = np.searchsorted(speed_edges, spd[fr], side='right')
```

iii. The agent read “four bins, each corresponding to 25% of the data” as global dataset quartiles.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.searchsorted(..., side='right')` assigns category 0 below the 25th percentile, 1 between the 25th and 50th, 2 between the 50th and 75th, and 3 at or above the 75th percentile.

ii.
```python
speed_edges = np.percentile(speeds, [25, 50, 75])
out[3] = np.searchsorted(speed_edges, spd[fr], side='right')
```

iii. This was intended to make each global category contain approximately a quarter of retained frames.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is indexed with the exact retained frame array used for neural columns.

ii.
```python
neural_s.append(spk[:, fr].astype(np.float32))
out[3] = np.searchsorted(speed_edges, spd[fr], side='right')
```

iii. Both streams are frame-indexed.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural frame count limits behavior arrays; NaN trial indices are ignored; out-of-range lick frames are dropped; zero-standard-deviation neurons use a divisor of one; absent stimulus mappings and very short retained trials are excluded; neuron/retinotopy counts are asserted equal.

ii.
```python
ok = corr & moving & ~np.isnan(tr)
assert sum(counts) == len(iarea)
sd[sd == 0] = 1.0
lf = lf[(lf >= 0) & (lf < nfr)]
```

iii. The trajectory describes targeted audits of stimulus mappings, event times, and frame/trial alignment, and treats exclusions as safeguards for unusable or unassigned data.

## 12-a. What are the most time-consuming steps of the code?

i. Loading 405 GB of spike files, selecting/copying neuron traces, z-scoring them, writing the roughly 6.6 GB pickle, and the subsequent decoder validation dominate runtime.

ii.
```python
blocks = np.load(spkfile, allow_pickle=True).item()['spks']
spk = np.empty((len(sel), nfr), dtype=np.float32)
spk = (spk - mu) / sd
pickle.dump(data, f, protocol=4)
```

iii. The trajectory repeatedly identifies neural I/O as the main cost; full conversion required long background runs and produced 89 sessions and 37,801 trials.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The loop that appends every retained frame to a trial list, the per-area Python mapping of `iarea`, the plane-copy loop, the per-trial output construction, and repeated linear subject lookup could be vectorized or indexed more directly.

ii.
```python
for i in idx:
    groups[int(tr[i])].append(i)
area = np.array([IAREA_TO_AREA.get(int(a), -1) for a in iarea])
for bi, blk_arr in enumerate(blocks):
    ...
```

iii. The agent did not explicitly discuss these optimization opportunities; it focused on correctness audits and the much larger I/O cost.

## 12-c. What processing does the code repeat multiple times?

i. `trial_frames` and eligibility checks are run once in the global speed pass and again during conversion. Every behavior file is loaded in full, and duplicated views are traversed to merge mappings.

ii.
```python
frames = trial_frames(beh, nfr)  # first pass
...
frames = trial_frames(beh, nfr)  # session conversion
```

iii. The first pass is justified by the agent's choice of global speed quartiles, though it repeats grouping/filter work.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads all behavioral duplicate views even though only the first behavior object is retained; it records experiment-type provenance and several metadata fields that do not feed decoder training; it z-scores all selected frames before retaining only running-corridor frames; and it computes global speed thresholds before any optional `--limit` truncation.

ii.
```python
if d['beh'] is None:
    d['beh'] = beh
d['exps'].append(exp)
spk = (spk - mu) / sd
if args.limit:
    names = names[:args.limit]
```

iii. The agent used duplicate views to recover complete stimulus mappings and metadata for traceability. It did not identify these operations as discarded work.
