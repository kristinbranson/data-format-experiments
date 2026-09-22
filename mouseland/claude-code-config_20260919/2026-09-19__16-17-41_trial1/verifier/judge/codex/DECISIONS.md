# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads `beh/Imaging_Exp_info.npy` as the master index, deduplicates its 141 experiment-type entries into 89 physical recordings, loads each `Beh_<type>.npy` once, and loads per-session spike blocks and retinotopy `iarea`. It uses two passes: a behavior-only pass over all sessions, then a neural pass over selected sessions.

ii.
```python
exp_info = np.load(os.path.join(ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=1).item()
key = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
B = np.load(os.path.join(ROOT, 'beh', 'Beh_%s.npy' % exp_type), allow_pickle=1).item()
spks = np.load(fn, allow_pickle=True).item()['spks']
```

iii. The notes justify deduplication by the paper's 89 recordings/19 mice and say loading behavior by file avoids retaining all 5 GB of behavior data. Spike blocks are kept separate to avoid a large concatenation copy.

## 1-b. How are the data split into subjects?

i. Subject identity comes from `mname`. During ordered session assembly, unique mouse names are appended to `subjects`, and each session receives the corresponding integer `subject_idx`.

ii.
```python
if r['mname'] not in subjects:
    subjects.append(r['mname'])
subject_idx.append(subjects.index(r['mname']))
```

iii. The notes state that `mname` directly identifies the 19 mice and that the 89-session/19-mouse count matches the paper.

## 1-c. How are the data split into sessions?

i. A physical session is the unique `(mname, datexp, blk)` triple. Duplicate index entries across experiment types are merged into one record; experiment types and stimulus mappings are accumulated.

ii.
```python
key = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
rec = sessions.setdefault(key, dict(key=key, mname=ndb['mname'],
    datexp=ndb['datexp'], blk=ndb['blk'], exp_types=[], beh_keys=[], ...))
```

iii. The agent verified that 141 index entries represent 89 physical recordings and used the triple because it also names the neural and retinotopy files.

## 1-d. How are the data split into trials?

i. Frames are assigned by `ft_trInd`, but only frames satisfying `ft_CorrSpc`, `ft_move > 0`, finite/in-range trial IDs, and a mapped canonical stimulus are retained. Retained frames are grouped by trial and sorted; trials remain variable length.

ii.
```python
valid = corr & move & np.isfinite(tr)
counts = np.bincount(tr_int[tr_int >= 0], minlength=ntrials)
for t in range(ntrials):
    ...
    if stim_of_trial[t] < 0:
        continue
    fi = np.sort(fi)
```

iii. The agent says this reproduces the paper/reference `fr_valid` rule for 0–4 m running frames and treats a corridor traversal as one trial aligned to entry.

## 1-e. How are trials filtered based on quality controls?

i. A trial is omitted if it has no valid running/corridor frames or its wall lacks a merged canonical `stim_id`. There is no trial-duration outlier cutoff; pauses are removed frame-by-frame through `ft_move > 0`.

ii.
```python
if n == 0:
    continue
if stim_of_trial[t] < 0:
    continue
```

iii. The notes argue that the paper used running timepoints only and that unmapped `circle3` trials were not analyzed by the paper. They report that all retained sessions still have many trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from the per-plane arrays in each session's `['spks']`; neuron anatomical labels come from retinotopy `iarea`.

ii.
```python
spks = np.load(fn, allow_pickle=True).item()['spks']
with np.load(fn, allow_pickle=True) as d:
    return d['iarea']
```

iii. The agent identified `spks` as already neuropil-corrected, cell-classifier-curated, non-negative deconvolved Suite2p traces, so it did not compute ΔF/F or deconvolve again.

## 2-b. How is the `neural` data processed?

i. Kept neuron rows and retained frame columns are selected. Each neuron is z-scored using its mean and population standard deviation over all frames in that session, then the combined matrix is split into contiguous per-trial float32 arrays.

ii.
```python
mu = s1 / nf
sd = np.sqrt(np.maximum(s2 / nf - mu * mu, 0.0))
sd[sd == 0] = 1.0
g -= mu[km, None].astype(np.float32)
g /= sd[km, None].astype(np.float32)
trials.append(np.ascontiguousarray(neural[:, a:bnd]))
```

iii. The notes justify z-scoring because fluorescence scales vary strongly across neurons/sessions and cite a reference population-analysis use of `stats.zscore`; sample decoder tests reportedly improved after scaling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are retained only if `iarea` maps to V1, mHV, lHV, or aHV; effectively `iarea` is neither -1 nor 7. No additional released quality metric is applied.

ii.
```python
reg[iarea == 8] = 0
reg[np.isin(iarea, [0, 1, 2, 9])] = 1
reg[np.isin(iarea, [5, 6])] = 2
reg[np.isin(iarea, [3, 4])] = 3
keep = reg >= 0
```

iii. The agent says Suite2p curation was already applied and this anatomical exclusion is the only further neuron filter in the reference code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial starts with retained frames after corridor entry and uses frame timestamps relative to `Trial_start_time`. Trials are variable length, unpadded, and omit stopped frames, so their stored samples can have gaps in real time.

ii.
```python
inp[2] = (ftt - tstart[t]) * SEC_PER_DAY
trials.append(np.ascontiguousarray(neural[:, a:bnd]))
```

iii. The agent defines corridor entry (`StartFr`/`Trial_start_time`) as time zero and argues variable lengths are appropriate because the format requires common bin size, not common duration.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. Each retained imaging frame is one bin; metadata uses the mean session median frame interval, approximately 315 ms (about 3.18 Hz).

ii.
```python
frame_interval_s=float(np.median(np.diff(b['ft'])) * SEC_PER_DAY)
dt = float(np.mean([s['frame_interval_s'] for s in session_info]))
'time_bin_size': dt * 1000.0
```

iii. The notes say the native imaging grid is the finest available temporal resolution and behavior is already aligned to it.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundTime` and per-frame MATLAB-datenum timestamps `ft`.

ii.
```python
tcue = np.asarray(b['SoundTime'], dtype=float)
ftt = ft[fi]
```

iii. The agent chose the direct timestamps rather than reconstructing cue time from frame numbers.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For every retained frame, frame time is subtracted from cue time and converted from days to seconds. Values are positive before and negative after the cue.

ii.
```python
inp[0] = (tcue[t] - ftt) * SEC_PER_DAY
```

iii. The notes emphasize that the task explicitly requests a continuous, time-varying quantity, not a binary onset indicator.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It uses `ft[fi]` for exactly the same `fi` frame indices selected from the neural traces.

ii.
```python
fi = np.sort(fi)
ftt = ft[fi]
inp[0] = (tcue[t] - ftt) * SEC_PER_DAY
```

iii. The agent states all behavior streams share imaging-frame indexing, enabling direct alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from each session's `datexp` and the earliest imaging-session date for the same `mname`.

ii.
```python
first_day[mn] = min(first_day.get(mn, 1 << 30), datenum(sessions[k]['datexp']))
```

iii. The notes say experiment-type-specific session counters were inconsistent, while dates were unambiguous.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Dates are converted to calendar ordinals; elapsed calendar days since that mouse's first imaging session are computed and broadcast across every frame of every trial in the session.

ii.
```python
day = datenum(sessions[k]['datexp']) - first_day[sessions[k]['mname']]
inp[1] = day_of_training
```

iii. The agent argues this continuous value tracks learning progression even when recording dates are not consecutive.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It uses per-trial `Trial_start_time` and retained-frame `ft` timestamps.

ii.
```python
tstart = np.asarray(b['Trial_start_time'], dtype=float)
ftt = ft[fi]
```

iii. The agent identifies `Trial_start_time`/`StartFr` as corridor entry, the requested alignment event.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Trial-start time is subtracted from each retained frame time and the result is converted from days to seconds.

ii.
```python
inp[2] = (ftt - tstart[t]) * SEC_PER_DAY
```

iii. The agent uses real elapsed time, preserving gaps caused by removed non-running frames.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. The calculation uses timestamps at the same frame indices used to select neural columns.

ii.
```python
frame_idx.append(fi)
ftt = ft[fi]
inp[2] = (ftt - tstart[t]) * SEC_PER_DAY
```

iii. The notes report explicit spot checks of temporal and frame-index alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is directly derived from the per-trial `isRew` array.

ii.
```python
isrew = np.asarray(b['isRew']).astype(np.float32)
```

iii. The agent interprets this as whether the current corridor is rewarded for the mouse.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The trial's 0/1 value is broadcast to every retained timepoint without further transformation.

ii.
```python
inp[3] = isrew[t]
```

iii. The notes say unrewarded/naive cohorts naturally remain zero and rewarded task trials are one.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`, `UniqWalls`, and `stim_id`, with stimulus mappings merged across all experiment-type entries for a physical recording.

ii.
```python
for w, s in zip(walls, sid):
    if not np.isnan(s):
        stim_map[key][str(w)] = int(s)
stim_of_trial[t] = smap.get(str(b['WallName'][t]), -1)
```

iii. The agent found that individual experiment types contain incomplete mappings and merged them after checking for conflicts.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Raw, mouse-specific wall names are mapped to seven canonical IDs (0–6), broadcast through a trial, and trials with no canonical ID are dropped.

ii.
```python
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3', 'leaf1_swap1', 'leaf1_swap2']
if stim_of_trial[t] < 0:
    continue
out[0] = stim_of_trial[t]
```

iii. The agent argues canonical IDs make labels comparable across mice whose physical textures were named rock/wood versus circle/leaf and follow the paper's stimulus identity scheme.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking comes from event frame numbers in `LickFr`.

ii.
```python
lick_frames = np.floor(b['LickFr']).astype(np.int64)
```

iii. The notes identify `LickFr` as the directly available lick timing stream.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Fractional lick frames are floored, out-of-range events are removed, and an imaging frame is labeled one if at least one lick falls in it, otherwise zero.

ii.
```python
lick_frames = lick_frames[(lick_frames >= 0) & (lick_frames < nfr)]
lick = np.zeros(nfr, dtype=bool)
lick[lick_frames] = True
```

iii. This follows the reference's integer-frame treatment and gives the requested binary output.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The framewise indicator is indexed by the same `fi` used for neural columns.

ii.
```python
out[1] = lick[fi]
```

iii. Both signals use imaging-frame indices, so no resampling is needed.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is derived from per-imaging-frame `ft_Pos`, in decimeters.

ii.
```python
pos = b['ft_Pos'][:nfr]
```

iii. The agent verified the retained texture corridor spans approximately 0–40 dm.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is divided by 10 dm per meter, floored, and clipped to the four requested categories.

ii.
```python
out[2] = np.clip(np.floor(pos[fi] / POS_BIN_DM), 0, N_POS_BINS - 1)
```

iii. The notes connect the four bins to the 4 m texture region and exclude the 2 m gray area.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are 0, 10, 20, 30, and 40 dm, producing labels 0–3 for 0–1, 1–2, 2–3, and 3–4 m.

ii.
```python
POS_BIN_DM = 10.0
N_POS_BINS = 4
np.clip(np.floor(pos[fi] / POS_BIN_DM), 0, N_POS_BINS - 1)
```

iii. This directly implements four equal-length 1 m spatial bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is indexed by the same retained imaging-frame indices as neural activity.

ii.
```python
out[2] = np.clip(np.floor(pos[fi] / POS_BIN_DM), 0, N_POS_BINS - 1)
```

iii. Position is already sampled on the imaging grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed` at every retained frame.

ii.
```python
speed = b['ft_RunSpeed'][:nfr]
speed=speed[frame_idx].astype(np.float32)
```

iii. The agent rejected the misleading notebook description of `RunFr` and chose the actual per-frame speed variable.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speeds from all retained frames across all 89 recordings are pooled in pass 1. Global 25th, 50th, and 75th percentile edges are computed, then `searchsorted(..., side='right')` assigns codes 0–3.

ii.
```python
speed_all = np.concatenate(speeds)
speed_edges = np.percentile(speed_all, [25, 50, 75])
sb = np.searchsorted(speed_edges, ti['speed'], side='right').astype(np.int64)
```

iii. The agent says global pooling makes each class correspond to 25% of the complete decoder dataset and keeps sample/full discretization consistent.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Values at or below/above the three global percentile edges are assigned through right-sided insertion into four ordered bins; edge values are placed in the higher bin.

ii.
```python
speed_edges = np.percentile(speed_all, [25, 50, 75])
np.searchsorted(speed_edges, ti['speed'], side='right')
```

iii. The notes report approximately/exactly quarter-balanced aggregate output classes as a sanity check.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is selected at the same `frame_idx`, split with the same per-trial bounds, and inserted into the corresponding output row.

ii.
```python
speed=speed[frame_idx].astype(np.float32)
out[3] = sb[a:bnd]
```

iii. `ft_RunSpeed` is already on the imaging-frame grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Behavior streams are truncated to neural length; nonfinite/out-of-range trial indices and out-of-range licks are excluded; incomplete `stim_id` mappings are merged across duplicate experiment entries with conflict assertions; unmapped-stimulus trials are dropped; zero-variance neural standard deviations are replaced by one. Pass 1 conservatively uses `len(ft)-2` rather than opening neural files, so it can omit up to the final two valid frames.

ii.
```python
ft = b['ft'][:nfr]
valid = corr & move & np.isfinite(tr)
lick_frames = lick_frames[(lick_frames >= 0) & (lick_frames < nfr)]
assert prev is None or prev == int(s)
sd[sd == 0] = 1.0
nfr = len(b['ft']) - 2
```

iii. The notes say behavior is consistently 1–2 frames longer than neural data, duplicate stimulus maps fill one another's NaNs without conflicts, and these checks prevent malformed indices or numerical failures.

## 12-a. What are the most time-consuming steps of the code?

i. Loading roughly 404 GB of spike files, extracting/z-scoring large neural matrices, and writing the roughly 150.8 GB pickle dominate runtime; the notes report 491 seconds total and 138 seconds writing.

ii.
```python
spks = np.load(fn, allow_pickle=True).item()['spks']
neural = extract_neural(blocks, keep, ti['frame_idx'])
pickle.dump(data, f, protocol=5)
```

iii. The agent profiled the conversion and optimized page cache behavior, block copies, threading, and prefetching based on those costs.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Most frame grouping, masks, licking, binning, and neural arithmetic are vectorized. Remaining Python loops include per-trial stimulus lookup/construction, per-session pass loops, block/chunk task construction, and splitting the session matrix into trial arrays. The per-trial stimulus lookup could be vectorized or mapped by array operations, though variable-length output construction still requires iteration.

ii.
```python
for t in range(ntrials):
    stim_of_trial[t] = smap.get(str(b['WallName'][t]), -1)
for i in range(len(ti['inputs'])):
    a, bnd = ti['bounds'][i], ti['bounds'][i + 1]
    trials.append(np.ascontiguousarray(neural[:, a:bnd]))
```

iii. The notes emphasize that the expensive neural operations were already vectorized and threaded; the remaining loops are small relative to I/O or necessary for ragged trial lists.

## 12-c. What processing does the code repeat multiple times?

i. `build_trials` is run once for every session in pass 1, but pass 2 reuses those `tinfos`; it does not rebuild with the true neural length despite comments suggesting it would. Plotting recomputes masks and concatenates input/output arrays for up to two sessions. Neural extraction is repeated only when `--no-zscore` is requested: it first computes z-scored data, then re-extracts raw data.

ii.
```python
ti = build_trials(sessions[k], b, nfr, ...)
neural = extract_neural(blocks, keep, ti['frame_idx'])
if args.no_zscore:
    neural = np.concatenate([blk[:, ti['frame_idx']][...]], axis=0)
```

iii. The two-pass behavior work is justified as necessary for global speed edges; the notes do not explicitly justify the wasted z-score work on the optional raw path.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. On the normal full run, temporary full-session gathered/z-scored neural matrices, speed/position/time diagnostic fields in `tinfos`, and metadata statistics are used during assembly but not all are stored as decoder arrays. The clearest avoidable discarded work is z-scoring under `--no-zscore`, because that result is immediately overwritten. Diagnostic plotting, when enabled, also computes distributions and figures not used by decoder training.

ii.
```python
neural = extract_neural(blocks, keep, ti['frame_idx'])
if args.no_zscore:
    neural = np.concatenate([...], axis=0)
...
allout = np.concatenate(tinfo['outputs'], axis=1)
allinp = np.concatenate(tinfo['inputs'], axis=1)
```

iii. The agent considers diagnostic plots and rich metadata validation aids; it documents the raw-path double extraction as behavior of an optional flag rather than a full-run cost.
