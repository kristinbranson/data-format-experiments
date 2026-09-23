# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over all `Beh_*.npy` files in the `beh/` directory using `glob.glob`, loading each file and extracting session keys. It deduplicates sessions that appear across multiple behavior files (e.g., swap1/swap2 views of the same recording) by extracting a "base" session ID (stripping swap suffixes). For each unique recording, the neural data is loaded from `spk/<session_id>_neural_data.npy` and retinotopy from `retinotopy/<mouse>_<date>_trans.npz`. The `Imaging_Exp_info.npy` index is NOT used; instead the AI discovers sessions directly from behavior files.

ii.
```python
def collect_recordings():
    """Deduplicate the Beh_*.npy views into one entry per recording."""
    recs = {}
    for f in sorted(glob.glob(os.path.join(ROOT, 'beh', 'Beh_*.npy'))):
        exp = os.path.basename(f)[4:-4]
        B = np.load(f, allow_pickle=True).item()
        for key, beh in B.items():
            parts = key.split('_')
            base = '_'.join(parts[:5]) if parts[-1].startswith('swap') else key
            d = recs.setdefault(base, {'beh': None, 'walls': {}, 'exps': []})
            ...
```

iii. The AI's trajectory shows it discovered the 89 unique recordings by iterating behavior files and deduplicating swap views, confirming a 1:1 match with the 89 spk files.

## 1-b. How are the data split into subjects?

i. The mouse name is extracted from the first component of the session name (e.g., "TX88" from "TX88_2022_07_19_1"). Subjects are accumulated in a list as encountered.

ii.
```python
mouse = name.split('_')[0]
if mouse not in subjects:
    subjects.append(mouse)
data['subject_idx'].append(subjects.index(mouse))
```

iii. The AI identified 19 mice from the session naming convention.

## 1-c. How are the data split into sessions?

i. A session is one unique recording identified by `mouse_year_month_day_block`. Swap views of the same recording are merged into a single session. The AI found 89 unique sessions.

ii.
```python
parts = key.split('_')
base = '_'.join(parts[:5]) if parts[-1].startswith('swap') else key
d = recs.setdefault(base, {'beh': None, 'walls': {}, 'exps': []})
```

iii. The AI explicitly verified that after deduplication there are exactly 89 unique recordings matching the 89 spk files.

## 1-d. How are the data split into trials?

i. Trials are identified by `ft_trInd` (frame-to-trial index), restricted to frames inside the textured corridor (`ft_CorrSpc`) AND frames where the mouse was running (`ft_move > 0`). Only running frames are kept per the paper's statement "We only considered timepoints during running for analysis."

ii.
```python
def trial_frames(beh, nfr):
    tr = beh['ft_trInd'][:nfr]
    corr = beh['ft_CorrSpc'][:nfr].astype(bool)
    moving = beh['ft_move'][:nfr] > 0
    ok = corr & moving & ~np.isnan(tr)
    ...
    return {t: np.sort(np.array(v)) for t, v in groups.items()}
```

iii. The AI justified the movement filter by quoting the paper: "We only considered timepoints during running for analysis, which removed time periods when the task mice stopped to collect water rewards."

## 1-e. How are trials filtered based on quality controls?

i. Two filters: (1) trials with fewer than `MIN_FRAMES_PER_TRIAL = 5` running frames are excluded, and (2) trials whose wall texture has no canonical stimulus role (circle3) are excluded. No upper-bound trial length filter is applied.

ii.
```python
if len(fr) < MIN_FRAMES_PER_TRIAL or wall not in walls:
    continue    # unassigned stimulus (circle3) or too little running
```

iii. The AI noted that circle3 has no canonical stim_id in any view (except TX109 where it maps to role 4), so it excluded those 309 trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy`, which contains deconvolved calcium traces as a list of arrays per imaging plane. The visual area of each neuron comes from `iarea` in `retinotopy/<mouse>_<date>_trans.npz`.

ii.
```python
spkfile = os.path.join(ROOT, 'spk', '%s_neural_data.npy' % name)
blocks = np.load(spkfile, allow_pickle=True).item()['spks']
```

iii. The AI correctly identified the deconvolved traces as the neural data source.

## 2-b. How is the `neural` data processed?

i. The neural data is z-scored per neuron across the whole recording (subtracting mean and dividing by standard deviation). Additionally, only a random subset of at most `MAX_NEURONS = 2000` neurons per session is kept due to the 405 GB dataset size. Only neurons with valid retinotopic area assignments are included. Data is stored as float32.

ii.
```python
# z-score each neuron over the whole recording (as in utils.py)
mu = spk.mean(axis=1, keepdims=True)
sd = spk.std(axis=1, keepdims=True)
sd[sd == 0] = 1.0
spk = (spk - mu) / sd
```
```python
if len(valid) > MAX_NEURONS:
    sel = np.sort(rng.choice(valid, MAX_NEURONS, replace=False))
else:
    sel = valid
```

iii. The AI justified z-scoring by citing `utils.get_kfold_reward_response` in the reference code, and neuron subsampling by citing the 405 GB dataset size and the decoder's SVD init capping at 2000.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by visual area: only those with `iarea` mapping to V1, mHV, lHV, or aHV are kept. Additionally, neurons are subsampled to at most 2000 per session.

ii.
```python
IAREA_TO_AREA = {8: 0, 0: 1, 1: 1, 2: 1, 9: 1, 5: 2, 6: 2, 3: 3, 4: 3}
area = np.array([IAREA_TO_AREA.get(int(a), -1) for a in iarea])
valid = np.where(area >= 0)[0]
```

iii. The AI noted that iarea -1 and 7 are unassigned and excluded, matching the paper's analyses.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry (trial start). Each trial's neural data consists of the running frames within the textured corridor for that trial. Trials have variable length because only running frames are kept and mice traverse the corridor at different speeds.

ii.
```python
neural_s.append(spk[:, fr].astype(np.float32))
```

iii. The AI aligned to corridor entry by selecting frames where `ft_trInd == trial & ft_CorrSpc & ft_move > 0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is the median inter-frame interval across all frames in a session, approximately 314.7 ms (~3.177 Hz). No temporal rebinning is applied; the native imaging frame rate is used.

ii.
```python
bin_s = float(np.median(np.diff(ft)))
...
'time_bin_size': float(np.median(bin_ms)),
```

iii. The AI computed the bin size empirically from the frame timestamps rather than using a fixed constant.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the frame index of the sound cue for each trial) and the retained running frames for that trial.

ii.
```python
rank_cue = float(np.searchsorted(fr, beh['SoundFr'][t])) * bin_s
inp[0] = k - rank_cue
```

iii. The AI used `SoundFr` as the cue timing source.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The cue position is found by `searchsorted` into the retained (running) frame indices, giving the cue's position in accumulated running time. Time to cue is computed as `accumulated_running_time - cue_running_time`. Note this is "time since sound cue" (negative before, positive after), not "time to sound cue" (positive before, negative after).

ii.
```python
k = np.arange(len(fr), dtype=np.float32) * bin_s
rank_cue = float(np.searchsorted(fr, beh['SoundFr'][t])) * bin_s
inp[0] = k - rank_cue
```

iii. The AI used accumulated running time rather than wall-clock time, arguing that wall-clock time would contain "large invisible jumps whenever the mouse stopped inside the corridor."

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The time is computed on the same retained running frames used for neural data, so alignment is automatic.

ii.
```python
k = np.arange(len(fr), dtype=np.float32) * bin_s
```

iii. The AI noted that since only running frames are kept, the time base is accumulated running time.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the date components of each session name, converted to `datetime.date` objects. The first recording date for each mouse is found, and the day of training is the number of calendar days since that first recording.

ii.
```python
def date_of(name):
    p = name.split('_')
    return datetime.date(int(p[1]), int(p[2]), int(p[3]))
first = {}
for name in names:
    m = name.split('_')[0]
    first[m] = min(first.get(m, date_of(name)), date_of(name))
...
day = float((date_of(name) - first[mouse]).days)
```

iii. The AI computed calendar days elapsed since each mouse's first recording.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The day of training is computed as the number of calendar days between the session date and the mouse's first recording date. This is broadcast as a constant across all time bins of each trial.

ii.
```python
day = float((date_of(name) - first[mouse]).days)
...
inp[1] = day
```

iii. The AI chose calendar days rather than counting recording sessions, which gives different values when recordings are not on consecutive days.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From the retained running frame indices for each trial and the median inter-frame interval.

ii.
```python
k = np.arange(len(fr), dtype=np.float32) * bin_s
inp[2] = k
```

iii. The AI used accumulated running time starting from 0.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Time since trial start is computed as `frame_index_within_trial * bin_size`, where frame indices count only retained running frames. This gives accumulated running time, not wall-clock time since corridor entry.

ii.
```python
k = np.arange(len(fr), dtype=np.float32) * bin_s
inp[2] = k
```

iii. The AI argued this is appropriate because non-running frames are excluded from the data, so wall-clock time would have invisible jumps.

## 5-c. How is `input` *Time since trial start* aligned with the neural data?

i. Computed on the same retained running frames, so alignment is automatic.

ii.
```python
k = np.arange(len(fr), dtype=np.float32) * bin_s
```

iii. Same running-frame time base as neural data.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, the per-trial reward flag.

ii.
```python
inp[3] = float(beh['isRew'][t])
```

iii. Direct use of the reward flag.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No processing; the flag is used directly as a float, broadcast across all time bins of the trial.

ii.
```python
inp[3] = float(beh['isRew'][t])
```

iii. No processing needed.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `UniqWalls` and `stim_id` in the behavior data, which map wall names to canonical stimulus roles (0-6). The AI uses the `stim_id` index directly, giving 7 stimulus categories rather than 4.

ii.
```python
for wall, sid in zip(beh['UniqWalls'], beh['stim_id']):
    if not np.isnan(sid):
        d['walls'][str(wall)] = int(sid)
...
out[0] = walls[wall]
```

iii. The AI argued that physical texture names differ between mice (some mice have leaf/circle, others rock/brick), so the canonical role (stim_id) provides a cross-mouse label. The AI noted that `TrialStim` contains placeholders in swap views.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The wall name of each trial is mapped to its canonical stimulus role via the pre-built `walls` dictionary (from UniqWalls + stim_id). This gives integer categories 0-6, corresponding to 7 stimulus types. The value is broadcast across all time bins.

ii.
```python
out[0] = walls[wall]
```
```python
'output_values': [STIM_NAMES, ...],
# where STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3', 'leaf1_swap1', 'leaf1_swap2']
```

iii. The AI chose 7 fine-grained categories rather than 4 coarse texture categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the frame indices of lick events in the session.

ii.
```python
lick = np.zeros(nfr, dtype=bool)
lf = np.floor(beh['LickFr']).astype(int)
lf = lf[(lf >= 0) & (lf < nfr)]
lick[lf] = True
```

iii. The AI correctly identified LickFr as the lick source.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Fractional lick frame numbers are floored to integers, filtered to valid range, and used to set a boolean array to True at those frames. The lick values for retained running frames are then extracted.

ii.
```python
lick = np.zeros(nfr, dtype=bool)
lf = np.floor(beh['LickFr']).astype(int)
lf = lf[(lf >= 0) & (lf < nfr)]
lick[lf] = True
...
out[1] = lick[fr].astype(np.int64)
```

iii. Binary lick/no-lick per frame.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick array is indexed by the same retained running frame indices used for neural data.

ii.
```python
out[1] = lick[fr].astype(np.int64)
```

iii. Same frame indexing as neural data.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position at each imaging frame in decimeters.

ii.
```python
pos = beh['ft_Pos'][:nfr]
...
out[2] = np.clip((pos[fr] / 10.0).astype(int), 0, CORRIDOR_BINS - 1)
```

iii. Direct use of ft_Pos.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimeters is divided by 10 to get meters, then integer-truncated and clipped to [0, 3], giving 4 bins of 1 m each.

ii.
```python
out[2] = np.clip((pos[fr] / 10.0).astype(int), 0, CORRIDOR_BINS - 1)
```

iii. The AI correctly mapped decimeters to 4 x 1-m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position in decimeters is divided by 10, truncated to integer, and clipped to [0, 3]. This gives bins: 0-1m, 1-2m, 2-3m, 3-4m.

ii.
```python
out[2] = np.clip((pos[fr] / 10.0).astype(int), 0, CORRIDOR_BINS - 1)
```

iii. Same as reference approach.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is indexed by the same retained running frame indices.

ii.
```python
out[2] = np.clip((pos[fr] / 10.0).astype(int), 0, CORRIDOR_BINS - 1)
```

iii. Same frame indexing.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
spd = beh['ft_RunSpeed'][:nfr]
...
out[3] = np.searchsorted(speed_edges, spd[fr], side='right')
```

iii. Direct use of ft_RunSpeed.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Global speed quartile edges are computed across all retained running frames from all sessions in a first pass. Then `np.searchsorted` assigns each frame's speed to one of 4 bins based on these global percentile edges [25th, 50th, 75th].

ii.
```python
# Pass 1: collect speeds
speeds.append(spd[fr])
speed_edges = np.percentile(speeds, [25, 50, 75])

# Pass 2: bin speeds
out[3] = np.searchsorted(speed_edges, spd[fr], side='right')
```

iii. The AI used global percentile-based edges rather than per-session rank-based quartiles.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Speed is categorized into 4 bins using global 25th, 50th, and 75th percentile edges computed across all retained running frames from all sessions.

ii.
```python
speed_edges = np.percentile(speeds, [25, 50, 75])
out[3] = np.searchsorted(speed_edges, spd[fr], side='right')
```

iii. Global edges rather than per-session quartiles.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is indexed by the same retained running frame indices.

ii.
```python
out[3] = np.searchsorted(speed_edges, spd[fr], side='right')
```

iii. Same frame indexing.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The behavior data is cut to the number of neural frames (`nfr = blocks[0].shape[1]`). Lick frames outside the valid range are filtered out. Trials with fewer than 5 running frames or with unmapped wall textures (circle3) are excluded. NaN values in `ft_trInd` are excluded from frame selection.

ii.
```python
lf = lf[(lf >= 0) & (lf < nfr)]
...
ok = corr & moving & ~np.isnan(tr)
...
if len(fr) < MIN_FRAMES_PER_TRIAL or wall not in walls:
    continue
```

iii. The AI handled edge cases by filtering invalid frames and lick events.

## 12-a. What are the most time-consuming steps of the code?

i. Reading the 89 spike files (405 GB total) is by far the most time-consuming step. The AI also performs a two-pass approach (first pass over behavior for global speed quartiles, second pass loading neural data), which adds overhead.

ii.
```python
blocks = np.load(spkfile, allow_pickle=True).item()['spks']
```

iii. I/O dominated by the large neural data files.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The `trial_frames` function uses a Python loop to group frames by trial index, which could be vectorized using `np.unique` with return_inverse or similar approaches. The area mapping also uses a list comprehension instead of vectorized operations.

ii.
```python
groups = collections.defaultdict(list)
for i in idx:
    groups[int(tr[i])].append(i)
```
```python
area = np.array([IAREA_TO_AREA.get(int(a), -1) for a in iarea])
```

iii. N/A

## 12-c. What processing does the code repeat multiple times?

i. The behavior data is processed twice: once in the first pass to compute global speed quartile edges, and again in the second pass to extract trial data. The `trial_frames` function is called in both passes.

ii.
```python
# Pass 1
for name in names:
    beh = recs[name]['beh']
    frames = trial_frames(beh, nfr)
    ...
# Pass 2
for si, name in enumerate(names):
    ...
    frames = trial_frames(beh, nfr)
```

iii. The two-pass design was chosen to compute global speed edges before processing.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The z-scoring of neural data is additional processing not present in the reference solution. The neuron subsampling (capping at 2000) discards neural data. The `ft_move` filtering removes frames that the reference retains. These processing steps change the data that reaches the decoder.

ii.
```python
spk = (spk - mu) / sd  # z-scoring
...
sel = np.sort(rng.choice(valid, MAX_NEURONS, replace=False))  # subsampling
```

iii. The AI justified z-scoring by citing reference code utils.py, and subsampling by the dataset size constraint.
