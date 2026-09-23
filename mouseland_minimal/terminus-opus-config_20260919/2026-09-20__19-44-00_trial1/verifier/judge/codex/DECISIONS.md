# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all behavior data by globbing every `Beh_*.npy` file under `/app/data/beh`, merges them into a deduplicated recording table keyed by `mouse_year_month_day_block`, and stores one representative behavior dict plus a wall-to-stimulus map per recording. It then loads retinotopy and spike data per recording from `/app/data/retinotopy` and `/app/data/spk`.

ii. 
```python
def collect_recordings():
    recs = {}
    for f in sorted(glob.glob(os.path.join(ROOT, 'beh', 'Beh_*.npy'))):
        exp = os.path.basename(f)[4:-4]
        B = np.load(f, allow_pickle=True).item()
        for key, beh in B.items():
            parts = key.split('_')
            base = '_'.join(parts[:5]) if parts[-1].startswith('swap') else key
            d = recs.setdefault(base, {'beh': None, 'walls': {}, 'exps': []})
            ...
            if d['beh'] is None:
                d['beh'] = beh
```

```python
ret = np.load(os.path.join(ROOT, 'retinotopy', '%s_%s_trans.npz' % (mouse, date)),
              allow_pickle=True)
...
blocks = np.load(spkfile, allow_pickle=True).item()['spks']
```

iii. In the trajectory, the AI said the behavior files contain multiple "views" of the same recording and decided to enumerate recordings from the `Beh_*.npy` files, deduplicate them, and make them match the 89 spike files 1:1.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the mouse name prefix of each deduplicated recording key. The code appends each mouse once to `subjects` in first-seen session order and records each session's `subject_idx` as the index of that mouse in the list.

ii. 
```python
mouse = name.split('_')[0]
...
if mouse not in subjects:
    subjects.append(mouse)
...
data['subject_idx'].append(subjects.index(mouse))
```

iii. In the trajectory, the AI reported that the 89 recordings correspond to 19 mice and treated mouse identity as the first field of the recording name.

## 1-c. How are the data split into sessions?

i. A session is one deduplicated recording named `mouse_year_month_day_block`. Swap-view suffixes such as `_swap1` and `_swap2` are collapsed into the same base recording, and repeated listings across experiment-type behavior files are merged into one session entry.

ii. 
```python
parts = key.split('_')
base = '_'.join(parts[:5]) if parts[-1].startswith('swap') else key
d = recs.setdefault(base, {'beh': None, 'walls': {}, 'exps': []})
d['exps'].append(exp)
```

iii. The AI justified this in the trajectory by noting that the behavior files re-list the same recording under multiple experiment types and swap views, and that deduplication yields exactly 89 unique recordings matching the 89 spike files.

## 1-d. How are the data split into trials?

i. Trials are defined as the running frames inside the textured corridor. The code uses `ft_trInd` to label frames by trial, intersects that with `ft_CorrSpc` and `ft_move > 0`, then groups retained frame indices by trial id. This means a trial is not all corridor frames; it is only the moving subset of them.

ii. 
```python
def trial_frames(beh, nfr):
    tr = beh['ft_trInd'][:nfr]
    corr = beh['ft_CorrSpc'][:nfr].astype(bool)
    moving = beh['ft_move'][:nfr] > 0
    ok = corr & moving & ~np.isnan(tr)
    ...
    for i in idx:
        groups[int(tr[i])].append(i)
    return {t: np.sort(np.array(v)) for t, v in groups.items()}
```

```python
frames = trial_frames(beh, nfr)
for t in sorted(frames):
    fr = frames[t]
```

iii. The AI justified this in the trajectory by citing the paper's statement that only timepoints during running were analyzed, and by arguing that the decoder should see the running-only neural sequence.

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped if they have fewer than `MIN_FRAMES_PER_TRIAL` retained running frames or if their wall texture has no canonical mapping in the merged `walls` table. In practice this excludes very short running trials and unassigned `circle3` trials. The code does not implement the reference solution's 99th-percentile long-trial filter.

ii. 
```python
MIN_FRAMES_PER_TRIAL = 5
...
if len(fr) < MIN_FRAMES_PER_TRIAL or wall not in walls:
    continue    # unassigned stimulus (circle3) or too little running
```

iii. In the trajectory, the AI said it found 309 `circle3` trials with no canonical role and chose to exclude them, and it reported a typical kept trial length of about 21.6 running frames.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the per-plane `spks` arrays in each `*_neural_data.npy` spike file, and neuron area labels come from `iarea` in the corresponding retinotopy `.npz` file.

ii. 
```python
ret = np.load(os.path.join(ROOT, 'retinotopy', '%s_%s_trans.npz' % (mouse, date)),
              allow_pickle=True)
iarea = ret['iarea']
...
blocks = np.load(spkfile, allow_pickle=True).item()['spks']
```

iii. The trajectory states that the spike files contain Suite2p deconvolved traces and the retinotopy file provides the area assignments used for filtering neurons.

## 2-b. How is the `neural` data processed?

i. The code maps neurons to four coarse visual areas, keeps only valid-area neurons, randomly subsamples to at most 2,000 neurons per session, reconstructs the selected neurons plane by plane, z-scores each neuron across the full recording, and then slices each trial's retained frames. Per-trial neural arrays are stored as `float32`.

ii. 
```python
valid = np.where(area >= 0)[0]
rng = np.random.default_rng(SEED + si)
if len(valid) > MAX_NEURONS:
    sel = np.sort(rng.choice(valid, MAX_NEURONS, replace=False))
else:
    sel = valid
...
spk = np.empty((len(sel), nfr), dtype=np.float32)
for bi, blk_arr in enumerate(blocks):
    m = (sel >= offs[bi]) & (sel < offs[bi + 1])
    if m.any():
        spk[m] = blk_arr[sel[m] - offs[bi]]
...
mu = spk.mean(axis=1, keepdims=True)
sd = spk.std(axis=1, keepdims=True)
sd[sd == 0] = 1.0
spk = (spk - mu) / sd
...
neural_s.append(spk[:, fr].astype(np.float32))
```

iii. In the trajectory, the AI said the raw neural dataset is 405 GB, so it would subsample to 2,000 neurons per session to keep the conversion feasible, and z-score per neuron because the decoder does not normalize inputs and the paper's utilities do.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by retinotopic area assignment: only neurons whose `iarea` maps to `V1`, `mHV`, `lHV`, or `aHV` are kept. After that, the code imposes an additional random cap of 2,000 neurons per session.

ii. 
```python
IAREA_TO_AREA = {8: 0, 0: 1, 1: 1, 2: 1, 9: 1, 5: 2, 6: 2, 3: 3, 4: 3}
...
area = np.array([IAREA_TO_AREA.get(int(a), -1) for a in iarea])
valid = np.where(area >= 0)[0]
...
if len(valid) > MAX_NEURONS:
    sel = np.sort(rng.choice(valid, MAX_NEURONS, replace=False))
```

iii. The trajectory explicitly justifies both steps: valid retinotopic assignment is taken from the paper's `neu_area_ID` logic, and the 2,000-neuron cap is a practical concession to the 405 GB raw dataset and decoder cost.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The code aligns trials to corridor entry conceptually, but the saved neural sequence for each trial contains only the retained running frames within the 4 m textured corridor. Gray-space frames and stopped frames are omitted, so alignment is to trial start with a compressed within-trial time axis.

ii. 
```python
"""Frames of each trial: running, inside the textured corridor."""
...
corr = beh['ft_CorrSpc'][:nfr].astype(bool)
moving = beh['ft_move'][:nfr] > 0
ok = corr & moving & ~np.isnan(tr)
```

```python
neural_s.append(spk[:, fr].astype(np.float32))
```

iii. In the trajectory, the AI said trial start should be corridor entry, but argued that only running frames should be retained because those are the timepoints the paper analyzed and the decoder actually sees.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin is the native imaging frame interval, estimated per session as the median of `np.diff(ft)` and summarized in metadata as the median across sessions. No explicit temporal rebinning is applied, but the code drops stopped frames entirely.

ii. 
```python
ft = beh['ft'][:nfr] * 86400.0
bin_s = float(np.median(np.diff(ft)))
...
bin_ms.append(np.median(np.diff(ft)) * 1000.0)
...
'time_bin_size': float(np.median(bin_ms)),
```

iii. The trajectory says the sampling rate is about 3.177 Hz and that the AI wanted to preserve that native resolution while representing time on the retained-frame clock.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundFr` for the trial's cue frame, plus the session's imaging frame period estimated from `ft`.

ii. 
```python
ft = beh['ft'][:nfr] * 86400.0
bin_s = float(np.median(np.diff(ft)))
...
rank_cue = float(np.searchsorted(fr, beh['SoundFr'][t])) * bin_s
```

iii. In the trajectory, the AI said it originally used absolute timestamps, saw implausible values caused by long pauses, and switched to cue timing derived from the retained frame sequence.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The code builds a running-frame time axis `k = 0, bin_s, 2*bin_s, ...` for the kept frames of the trial, finds where `SoundFr` would fall within that retained frame list, converts that rank to seconds, and computes `k - rank_cue`. This yields a negative value before the cue and positive after the cue.

ii. 
```python
k = np.arange(len(fr), dtype=np.float32) * bin_s
rank_cue = float(np.searchsorted(fr, beh['SoundFr'][t])) * bin_s
inp = np.empty((4, len(fr)), dtype=np.float32)
inp[0] = k - rank_cue
```

iii. The AI justified this in the trajectory by saying that wall-clock time contains long gaps when the mouse stopped, while the saved neural data contains only the retained running frames, so the input should use the running-frame clock instead.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is defined on the same retained running-frame grid as the neural data for that trial, with exactly one value per kept neural time bin.

ii. 
```python
fr = frames[t]
...
neural_s.append(spk[:, fr].astype(np.float32))
...
k = np.arange(len(fr), dtype=np.float32) * bin_s
inp[0] = k - rank_cue
```

iii. The trajectory repeatedly states that all time-varying inputs were moved onto the retained-frame clock so they match the running-only neural sequence.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the date embedded in each deduplicated session name and the mouse id prefix.

ii. 
```python
def date_of(name):
    p = name.split('_')
    return datetime.date(int(p[1]), int(p[2]), int(p[3]))
...
mouse = name.split('_')[0]
day = float((date_of(name) - first[mouse]).days)
```

iii. The AI said in the trajectory that it would treat training day as days elapsed since the first recording for each mouse.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the code finds the earliest recording date and then computes each session's day value as the calendar-day difference from that first date. The result is a constant per trial and per time bin within the trial.

ii. 
```python
first = {}
for name in names:
    m = name.split('_')[0]
    first[m] = min(first.get(m, date_of(name)), date_of(name))
...
day = float((date_of(name) - first[mouse]).days)
...
inp[1] = day
```

iii. The trajectory justifies this as a measure of how far the animal is into training, using elapsed days rather than session count.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. In the final code it is derived from the retained-frame count within the trial and the session's frame period estimated from `ft`. It is not derived from `StartFr` anymore.

ii. 
```python
ft = beh['ft'][:nfr] * 86400.0
bin_s = float(np.median(np.diff(ft)))
...
k = np.arange(len(fr), dtype=np.float32) * bin_s
...
inp[2] = k
```

iii. The trajectory says the AI initially used absolute timestamps tied to trial start, then intentionally switched to retained running-frame time because stopped periods are absent from the neural data.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The code simply counts retained frames from the start of the kept sequence and multiplies by the frame period, so the signal starts at zero and increases by one imaging frame duration per retained bin.

ii. 
```python
k = np.arange(len(fr), dtype=np.float32) * bin_s
...
inp[2] = k
```

iii. The trajectory justifies this as the "actual clock of the neural sequence the decoder sees" after stopped periods are removed.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is defined on the exact same retained running-frame sequence `fr` used to slice the neural data, so it has the same number of time bins as the neural array for that trial.

ii. 
```python
fr = frames[t]
neural_s.append(spk[:, fr].astype(np.float32))
...
k = np.arange(len(fr), dtype=np.float32) * bin_s
inp[2] = k
```

iii. The trajectory repeatedly frames this as aligning the time inputs to the running-only neural sequence.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived directly from `isRew`, the per-trial rewarded-corridor flag.

ii. 
```python
inp[3] = float(beh['isRew'][t])
```

iii. The trajectory treated zero-lick unrewarded sessions as a genuine property of the data, which implicitly relies on `isRew` being the reward-availability signal.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No substantive processing is applied beyond converting the value to float and broadcasting it across all time bins of the trial.

ii. 
```python
inp = np.empty((4, len(fr)), dtype=np.float32)
...
inp[3] = float(beh['isRew'][t])
```

iii. The trajectory does not describe any extra transformation; it treats this as a direct per-trial flag.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName` on each trial together with a canonical wall-to-stimulus-role mapping built from `UniqWalls` and `stim_id` pooled across the recording's behavior-file views.

ii. 
```python
for wall, sid in zip(beh['UniqWalls'], beh['stim_id']):
    if not np.isnan(sid):
        d['walls'][str(wall)] = int(sid)
...
wall = str(beh['WallName'][t])
...
out[0] = walls[wall]
```

iii. The trajectory says the AI intentionally avoided `TrialStim` because some views use the placeholder string `stimulus_of_trial`, and instead used the canonical role from `UniqWalls + stim_id`.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The code assigns each wall texture to one of seven canonical stimulus-role ids (`circle1`, `circle2`, `leaf1`, `leaf2`, `leaf3`, `leaf1_swap1`, `leaf1_swap2`) rather than collapsing them into four broad texture families. The per-trial class is then broadcast across all time bins of the trial. Trials whose wall name is not in the canonical map are dropped.

ii. 
```python
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2']
...
if len(fr) < MIN_FRAMES_PER_TRIAL or wall not in walls:
    continue
...
out[0] = walls[wall]
```

iii. In the trajectory, the AI argues that physical texture names can swap roles across mice, so the canonical stimulus role is the right cross-mouse label, and that `circle3` trials should be excluded because they have no canonical role in most recordings.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr`, the per-session lick frame indices.

ii. 
```python
lick = np.zeros(nfr, dtype=bool)
lf = np.floor(beh['LickFr']).astype(int)
lf = lf[(lf >= 0) & (lf < nfr)]
lick[lf] = True
```

iii. The trajectory treats the many zero-lick sessions as real unrewarded recordings, implying that the `LickFr` mapping itself was accepted as correct.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The code floors each lick frame index to an integer imaging frame, drops out-of-range indices, and marks those frames as `True`; per trial it exports the binary flag on the retained frames.

ii. 
```python
lf = np.floor(beh['LickFr']).astype(int)
lf = lf[(lf >= 0) & (lf < nfr)]
lick[lf] = True
...
out[1] = lick[fr].astype(np.int64)
```

iii. In the trajectory, the AI explicitly audited whether the many all-zero licking sessions were a bug and concluded they were genuine properties of the unrewarded mice, so it kept this direct mapping.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is converted to a per-frame session vector and then indexed with the same retained frame list `fr` used for the neural data, so it is one binary value per saved neural time bin.

ii. 
```python
neural_s.append(spk[:, fr].astype(np.float32))
...
out[1] = lick[fr].astype(np.int64)
```

iii. The trajectory treats the retained running-frame grid as the common alignment grid for time-varying signals.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from `ft_Pos`, the per-frame corridor position.

ii. 
```python
pos = beh['ft_Pos'][:nfr]
...
out[2] = np.clip((pos[fr] / 10.0).astype(int), 0, CORRIDOR_BINS - 1)
```

iii. The trajectory says the textured corridor spans 0-40 dm and should therefore map directly to the requested four 1 m bins.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The position values are divided by 10 to convert decimeters into 1 m bins, cast to integers, and clipped to `0..3`.

ii. 
```python
CORRIDOR_BINS = 4
...
out[2] = np.clip((pos[fr] / 10.0).astype(int), 0, CORRIDOR_BINS - 1)
```

iii. In the trajectory, the AI explicitly justified restricting trials to the 4 m textured corridor because that makes the requested four 1 m bins line up exactly with the retained trial window.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The thresholds are fixed spatial cutoffs at 10, 20, and 30 decimeters, implemented by integer division by 10 and clipping to four categories.

ii. 
```python
out[2] = np.clip((pos[fr] / 10.0).astype(int), 0, CORRIDOR_BINS - 1)
```

iii. The trajectory's position argument is that the retained frames span exactly the 0-40 dm textured corridor, so the 1 m categories follow directly from geometry rather than data-driven thresholds.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is sampled from `pos[fr]`, where `fr` is the same retained running-frame list used for the neural data.

ii. 
```python
fr = frames[t]
neural_s.append(spk[:, fr].astype(np.float32))
...
out[2] = np.clip((pos[fr] / 10.0).astype(int), 0, CORRIDOR_BINS - 1)
```

iii. The trajectory repeatedly says the retained running frames define the alignment grid for neural and behavioral variables.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed`, the per-frame running speed.

ii. 
```python
spd = beh['ft_RunSpeed'][:nfr]
...
speeds.append(spd[fr])
...
out[3] = np.searchsorted(speed_edges, spd[fr], side='right')
```

iii. The trajectory includes a behavior-only pass whose purpose was to summarize running speeds and choose quartile cutoffs.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The code performs a first pass over all sessions, gathers `ft_RunSpeed` values from kept running frames of kept trials, computes global 25th/50th/75th percentile cutoffs, and then bins each retained trial frame by comparing its speed to those global edges.

ii. 
```python
speeds = []
for name in names:
    beh = recs[name]['beh']
    ...
    for t, fr in frames.items():
        if len(fr) >= MIN_FRAMES_PER_TRIAL and str(beh['WallName'][t]) in recs[name]['walls']:
            speeds.append(spd[fr])
speeds = np.concatenate(speeds)
speed_edges = np.percentile(speeds, [25, 50, 75])
```

```python
out[3] = np.searchsorted(speed_edges, spd[fr], side='right')
```

iii. In the trajectory, the AI said it wanted global speed quartiles after a behavior-only dry run, and reported the resulting dataset-wide edges.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The thresholds are the global percentile edges `speed_edges`, and the bin index is `np.searchsorted(speed_edges, speed, side='right')`, yielding four ordinal bins.

ii. 
```python
speed_edges = np.percentile(speeds, [25, 50, 75])
...
out[3] = np.searchsorted(speed_edges, spd[fr], side='right')
```

iii. The trajectory explicitly reports those global quartile edges and uses them as the discretization rule.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is sampled on the same retained running frames `fr` used to slice the neural data.

ii. 
```python
neural_s.append(spk[:, fr].astype(np.float32))
...
out[3] = np.searchsorted(speed_edges, spd[fr], side='right')
```

iii. The trajectory treats the retained running-frame sequence as the common time axis for the neural and output streams.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code truncates every frame-based behavior stream to the number of neural frames, ignores NaN trial labels when grouping frames, drops lick indices outside the imaged frame range, and skips trials with unmapped wall labels or too few retained frames. It also replaces wall-clock time calculations with retained-frame timing after the AI diagnosed pathological values caused by long pauses.

ii. 
```python
tr = beh['ft_trInd'][:nfr]
corr = beh['ft_CorrSpc'][:nfr].astype(bool)
moving = beh['ft_move'][:nfr] > 0
ok = corr & moving & ~np.isnan(tr)
```

```python
lf = np.floor(beh['LickFr']).astype(int)
lf = lf[(lf >= 0) & (lf < nfr)]
lick[lf] = True
```

```python
if len(fr) < MIN_FRAMES_PER_TRIAL or wall not in walls:
    continue
```

iii. The trajectory says the AI audited the apparent timing pathology, concluded it came from genuine mid-trial pauses, and therefore changed the time inputs to use the retained-frame clock. It also says `circle3` trials lacked canonical labels and should be excluded.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are reading the 405 GB spike files, extracting selected neurons plane by plane, and z-scoring them per session. There is also a full-dataset behavior pass to compute speed quartiles, but the trajectory emphasizes neural I/O as the main cost.

ii. 
```python
blocks = np.load(spkfile, allow_pickle=True).item()['spks']
...
for bi, blk_arr in enumerate(blocks):
    m = (sel >= offs[bi]) & (sel < offs[bi + 1])
    if m.any():
        spk[m] = blk_arr[sel[m] - offs[bi]]
...
spk = (spk - mu) / sd
```

iii. In the trajectory, the AI repeatedly cites the 405 GB size of the neural data as the main runtime and memory constraint.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining obvious Python loops are the frame-grouping loop in `trial_frames`, the plane-by-plane neuron extraction loop over `blocks`, and the per-trial assembly loop that builds each session's `neural`, `input`, and `output` arrays. Unlike the human reference, the AI already avoids scanning the full frame axis once per trial.

ii. 
```python
for i in idx:
    groups[int(tr[i])].append(i)
```

```python
for bi, blk_arr in enumerate(blocks):
    m = (sel >= offs[bi]) & (sel < offs[bi + 1])
    if m.any():
        spk[m] = blk_arr[sel[m] - offs[bi]]
```

```python
for t in sorted(frames):
    fr = frames[t]
    ...
    neural_s.append(spk[:, fr].astype(np.float32))
    ...
    input_s.append(inp)
    ...
    output_s.append(out)
```

iii. The trajectory does not give a detailed vectorization discussion, but it does show the AI was primarily optimizing around the large neural I/O cost and decoder scaling.

## 12-c. What processing does the code repeat multiple times?

i. The code computes trial frame groupings twice per session conceptually: once during the first behavior-only pass used to gather speeds and again during the main conversion pass. It also performs a full preliminary behavior pass across all sessions solely to determine speed quartiles before doing the neural pass.

ii. 
```python
for name in names:
    beh = recs[name]['beh']
    nfr = len(beh['ft'])
    frames = trial_frames(beh, nfr)
    ...
    speeds.append(spd[fr])
```

```python
for si, name in enumerate(names):
    ...
    frames = trial_frames(beh, nfr)
    ...
    for t in sorted(frames):
        ...
```

iii. In the trajectory, the AI explicitly described a behavior-only dry run to compute global speed quartiles before starting the expensive neural pass.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code keeps extra bookkeeping that the decoder does not use, such as `experiment_types`, `day_of_training`, and `n_neurons_recorded` inside `session_info`, and it records `bin_ms` across sessions only to summarize a median time bin in metadata. It also carries `exps` in `recs` only for later metadata.

ii. 
```python
d = recs.setdefault(base, {'beh': None, 'walls': {}, 'exps': []})
d['exps'].append(exp)
```

```python
session_info.append({'session': name, 'mouse': mouse, 'date': date, 'block': blk,
                     'experiment_types': sorted(set(recs[name]['exps'])),
                     'day_of_training': day, 'n_trials': len(neural_s),
                     'n_neurons': len(sel),
                     'n_neurons_recorded': int(len(iarea))})
```

```python
bin_ms.append(np.median(np.diff(ft)) * 1000.0)
...
'time_bin_size': float(np.median(bin_ms)),
```

iii. The trajectory does not offer a separate justification for these extra metadata-oriented computations; they appear to be convenience bookkeeping rather than decoder-critical processing.
