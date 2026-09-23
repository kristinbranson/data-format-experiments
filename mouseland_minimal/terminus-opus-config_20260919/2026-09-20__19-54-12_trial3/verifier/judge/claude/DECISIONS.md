# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads `Imaging_Exp_info.npy` as the master index, which groups recordings by experiment type. For each experiment type, the corresponding `Beh_<exp_type>.npy` behavior file is loaded. Unlike the reference which keeps only the first occurrence of each session, the AI collects ALL experiment-type entries for each session and merges their stimulus maps (WallName -> TrialStim mappings), since different experiment-type files name different subsets of stimuli. Neural spike files are loaded per session from `spk/<session_id>_neural_data.npy`, and retinotopy from `retinotopy/<mname>_<datexp>_trans.npz`.

ii.
```python
exp_info = np.load(os.path.join(ROOT, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()
sessions = {}
for exp_type in exp_info:
    Beh = np.load(os.path.join(ROOT, 'beh', 'Beh_%s.npy' % exp_type),
                  allow_pickle=True).item()
    for db in exp_info[exp_type]:
        key = '%s_%s_%s' % (db['mname'], db['datexp'], db['blk'])
        bkey = key + ('_' + db['stimtype'] if 'stimtype' in db else '')
        beh = Beh[bkey]
        s = sessions.setdefault(key, {'beh': beh, 'db': db, 'stim_map': {},
                                      'exp_types': []})
        s['exp_types'].append(exp_type)
        for w, c in zip(beh['WallName'], beh['TrialStim']):
            if str(c) != 'stimulus_of_trial':
                s['stim_map'][str(w)] = str(c)
```

iii. The agent verified that the same recording appears in multiple behavior files (with identical behavior data) and chose to collect stimulus maps across all files to resolve canonical stimulus names.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the `mname` field in the experiment info entries. A sorted unique list of mouse names forms the `subjects` list, and `subject_idx` maps each session to its index in that list.

ii.
```python
subjects = sorted(set(sessions[k]['db']['mname'] for k in keys))
data['subject_idx'].append(subjects.index(db['mname']))
```

iii. The mouse name is directly available from the experiment info.

## 1-c. How are the data split into sessions?

i. A session is uniquely identified by the triple `(mname, datexp, blk)`, which also names the spike file. The AI uses a `setdefault` pattern to ensure each unique session key appears only once. This yields 89 sessions across 19 mice.

ii.
```python
key = '%s_%s_%s' % (db['mname'], db['datexp'], db['blk'])
s = sessions.setdefault(key, {'beh': beh, 'db': db, 'stim_map': {}, 'exp_types': []})
```

iii. The agent verified that 89 unique session keys match exactly the 89 spike files.

## 1-d. How are the data split into trials?

i. Trials are corridor traversals. The AI identifies per-trial frames using `ft_trInd` (trial index per frame), `ft_CorrSpc` (inside 4-m textured corridor), AND `ft_move > 0` (mouse is running, VR is moving). Only frames satisfying all three conditions are kept for each trial.

ii.
```python
def trial_frames(beh, nfr):
    ft_tr = beh['ft_trInd'][:nfr]
    keep = (beh['ft_move'][:nfr] > 0) & beh['ft_CorrSpc'][:nfr]
    idx = np.where(keep)[0]
    tr = ft_tr[idx]
    ok = ~np.isnan(tr)
    idx, tr = idx[ok], tr[ok].astype(int)
    order = np.argsort(tr, kind='stable')
    idx, tr = idx[order], tr[order]
    bounds = np.searchsorted(tr, np.arange(int(beh['ntrials']) + 1))
    return [idx[bounds[t]:bounds[t + 1]] for t in range(int(beh['ntrials']))]
```

iii. The agent justified the running filter by quoting methods.txt: "We only considered timepoints during running for analysis" and citing the reference code `utils.py` pattern `VRmove & isCorridor`.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on two criteria: (1) fewer than 2 running-corridor frames after the ft_move/ft_CorrSpc filter, and (2) stimulus name not in the canonical list of 7 stimuli (STIM_NAMES). There is no trial length percentile filter.

ii.
```python
if len(idx) < 2:
    nskip += 1
    continue
name = smap.get(str(beh['WallName'][t]), None)
if name is None or name not in STIM_NAMES:
    nskip += 1
    continue
```

iii. The agent noted that `circle3` trials (309 total) have no canonical TrialStim name and are dropped. No long-trial filter is applied.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in each session's `<session_id>_neural_data.npy` file, which contains a list of per-plane neuron-by-frame arrays. Also from `iarea` in the retinotopy file for area filtering.

ii.
```python
spks = np.load(os.path.join(ROOT, 'spk', '%s_neural_data.npy' % k),
               allow_pickle=True).item()['spks']
ret = np.load(os.path.join(ROOT, 'retinotopy', '%s_%s_trans.npz'
                           % (db['mname'], db['datexp'])), allow_pickle=True)
aidx = area_index(ret['iarea'])
```

iii. The neural data is the deconvolved calcium traces from Suite2p.

## 2-b. How is the `neural` data processed?

i. Planes are concatenated. Valid-area neurons are identified. A random subset of at most 1000 neurons (NNEURONS) per session is selected. The selected neurons' traces are extracted plane-by-plane to limit memory usage. No additional smoothing or normalization is applied.

ii.
```python
valid = np.where(aidx >= 0)[0]
if len(valid) > args.nneurons:
    sel = np.sort(rng.choice(valid, args.nneurons, replace=False))
else:
    sel = valid
spk = np.empty((len(sel), spks[0].shape[1]), dtype=np.float32)
off, o2 = 0, 0
for p in range(len(spks)):
    n = spks[p].shape[0]
    take = sel[(sel >= off) & (sel < off + n)] - off
    spk[o2:o2 + len(take)] = spks[p][take]
    o2 += len(take)
    off += n
```

iii. The agent justified the 1000-neuron cap as "for tractability" since the raw data are 405 GB.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are kept only if their retinotopy `iarea` maps to one of the four visual area groups: V1 (iarea=8), mHV (0,1,2,9), lHV (5,6), aHV (3,4). All other neurons are excluded. Then a random subsample of at most 1000 is kept.

ii.
```python
def area_index(iarea):
    out = -np.ones(len(iarea), dtype=int)
    out[iarea == 8] = 0                # V1
    out[np.isin(iarea, [0, 1, 2, 9])] = 1  # medial HV
    out[np.isin(iarea, [5, 6])] = 2         # lateral HV
    out[np.isin(iarea, [3, 4])] = 3         # anterior HV
    return out
```

iii. The agent followed the paper's area groupings from `utils.neu_area_ID`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry (trial start). Each trial's neural data consists of the frames inside the textured corridor where the mouse was running. Variable-length trials are stored as-is with no padding.

ii.
```python
neural_s.append(np.ascontiguousarray(spk[:, idx]))
```

iii. The agent set `off_start=0.0` and `off_end=None` reflecting variable trial durations.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin is one imaging frame at ~3.17 Hz (~315 ms). No rebinning is applied. The bin size is computed empirically as the median inter-frame interval.

ii.
```python
dts.append(np.median(np.diff(ft)) * SEC_PER_DAY)
# ...
'time_bin_size': float(np.median(dts) * 1000.0),
```

iii. The imaging frame is the native temporal resolution and no resampling is needed.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `beh['SoundTime']` (the MATLAB datenum of the sound cue for each trial) and `beh['ft']` (the MATLAB datenum of each imaging frame).

ii.
```python
tcue = (beh['SoundTime'][t] - ft[idx]) * SEC_PER_DAY
```

iii. The agent used wall-clock timestamps (`SoundTime`) rather than frame indices (`SoundFr`) to compute the continuous time-to-cue variable.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the difference between the sound cue time and each frame's time is computed in seconds. `SoundTime[t] - ft[idx]` gives positive values before the cue and negative values after, multiplied by 86400 to convert from days to seconds.

ii.
```python
tcue = (beh['SoundTime'][t] - ft[idx]) * SEC_PER_DAY
inp[0] = tcue
```

iii. The sign convention matches the instruction name "time TO sound cue" (positive before, negative after).

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the same frame indices (`idx`) used for the neural data of that trial, so alignment is automatic.

ii.
```python
tcue = (beh['SoundTime'][t] - ft[idx]) * SEC_PER_DAY
neural_s.append(np.ascontiguousarray(spk[:, idx]))
```

iii. All variables use the same frame indices.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From `db['datexp']` (the date string in format `YYYY_MM_DD`) and the earliest date for each mouse across all sessions.

ii.
```python
d = datetime.date(*map(int, db['datexp'].split('_')))
if db['mname'] not in first_date or d < first_date[db['mname']]:
    first_date[db['mname']] = d
# ...
day = float((datetime.date(*map(int, db['datexp'].split('_')))
             - first_date[db['mname']]).days)
```

iii. The agent used calendar days since first recording session for each mouse.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Calendar days elapsed since the mouse's first imaging session. The first session is day 0. The value is scalar per session, broadcast to all timepoints of each trial.

ii.
```python
day = float((datetime.date(*map(int, db['datexp'].split('_')))
             - first_date[db['mname']]).days)
inp[1] = day
```

iii. The agent verified meaningful spreads across mice (e.g., TX108: days 0, 67, 76, 79, 86, 89, 92).

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `beh['Trial_start_time']` (the MATLAB datenum of corridor entry for each trial) and `beh['ft']` (frame timestamps).

ii.
```python
tt = (ft[idx] - beh['Trial_start_time'][t]) * SEC_PER_DAY
```

iii. The agent used the wall-clock trial start time rather than `StartFr`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each trial, the difference between each frame's time and the trial start time is computed in seconds. Values start near zero and increase.

ii.
```python
tt = (ft[idx] - beh['Trial_start_time'][t]) * SEC_PER_DAY
inp[2] = tt
```

iii. Simple time difference converted from MATLAB datenums to seconds.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed from the same frame indices (`idx`) as the neural data.

ii.
```python
tt = (ft[idx] - beh['Trial_start_time'][t]) * SEC_PER_DAY
neural_s.append(np.ascontiguousarray(spk[:, idx]))
```

iii. Same frame index alignment as all other variables.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `beh['isRew']`, a per-trial flag indicating whether the trial is in the rewarded corridor.

ii.
```python
inp[3] = float(beh['isRew'][t])
```

iii. Directly available per trial.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean is cast to float (0.0 or 1.0) and broadcast to all timepoints of the trial.

ii.
```python
inp[3] = float(beh['isRew'][t])
```

iii. No additional processing. For unsupervised/naive mice, isRew is 0 for all trials.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `beh['TrialStim']` (canonical/functional stimulus name per trial) via a mapping built from `(WallName, TrialStim)` pairs across all experiment-type behavior files.

ii.
```python
for w, c in zip(beh['WallName'], beh['TrialStim']):
    if str(c) != 'stimulus_of_trial':
        s['stim_map'][str(w)] = str(c)
# ...
name = smap.get(str(beh['WallName'][t]), None)
out[0] = STIM_NAMES.index(name)
```

iii. The agent argues that `TrialStim` gives canonical/functional names consistent across mice, while `WallName` gives physical texture names that vary between mice.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The 7 canonical stimulus names are used as categories: `['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3', 'leaf1_swap1', 'leaf1_swap2']`. Each trial is assigned its index into this list. Trials with stimuli not in this list (e.g., `circle3`) are dropped. The value is per-trial, broadcast to all timepoints.

ii.
```python
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2']
out[0] = STIM_NAMES.index(name)
```

iii. The agent dropped 309 `circle3` trials that had no canonical TrialStim name.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `beh['LickFr']`, the frame number of each lick in the session.

ii.
```python
lickfr = np.round(beh['LickFr']).astype(int) if len(beh['LickFr']) else np.zeros(0, int)
lickfr = lickfr[(lickfr >= 0) & (lickfr < nfr)]
is_lick = np.zeros(nfr, dtype=np.int64)
is_lick[lickfr] = 1
```

iii. Lick frame numbers are directly available.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A binary vector is created for the session (0=no lick, 1=lick). Lick frame numbers are rounded to integers (not truncated) and used to set the corresponding positions to 1. Licks outside the valid frame range are dropped.

ii.
```python
lickfr = np.round(beh['LickFr']).astype(int)
lickfr = lickfr[(lickfr >= 0) & (lickfr < nfr)]
is_lick = np.zeros(nfr, dtype=np.int64)
is_lick[lickfr] = 1
out[1] = is_lick[idx]
```

iii. Simple binary encoding of lick events.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` indexes imaging frames, so the binary lick vector is on the same grid. The same frame indices (`idx`) are used as for neural data.

ii.
```python
out[1] = is_lick[idx]
neural_s.append(np.ascontiguousarray(spk[:, idx]))
```

iii. Frame-based alignment.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `beh['ft_Pos']`, the position in the corridor at each imaging frame, in decimeters.

ii.
```python
pos = beh['ft_Pos'][:nfr]
out[2] = np.clip((pos[idx] / 10.0).astype(int), 0, 3)
```

iii. Position is directly available per frame.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimeters is divided by 10 to convert to meters, then floor-divided to get bin indices 0-3 for the four 1-m bins. Values are clipped to [0, 3].

ii.
```python
out[2] = np.clip((pos[idx] / 10.0).astype(int), 0, 3)
```

iii. The 4-m corridor maps to four 1-m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Floor division by 10 dm (=1 m) gives bins: 0-1 m -> 0, 1-2 m -> 1, 2-3 m -> 2, 3-4 m -> 3. Clipped to 3 maximum.

ii.
```python
out[2] = np.clip((pos[idx] / 10.0).astype(int), 0, 3)
```

iii. Equal 1-m bins as specified in the instructions.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Same frame indices (`idx`) as neural data.

ii.
```python
out[2] = np.clip((pos[idx] / 10.0).astype(int), 0, 3)
neural_s.append(np.ascontiguousarray(spk[:, idx]))
```

iii. Frame-based alignment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `beh['ft_RunSpeed']`, the running speed at each imaging frame.

ii.
```python
speed = beh['ft_RunSpeed'][:nfr]
out[3] = np.searchsorted(speed_edges, speed[idx])
```

iii. Running speed is directly available per frame.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is discretized into 4 quartile bins using **global** percentile edges computed from all running-corridor frames across all 89 sessions (first pass). The edges are at the 25th, 50th, and 75th percentiles. `np.searchsorted` assigns each frame's speed to a bin.

ii.
```python
# Pass 1: global speed edges
for k in keys:
    keep = (beh['ft_move'][:nfr] > 0) & beh['ft_CorrSpc'][:nfr]
    speeds.append(beh['ft_RunSpeed'][:nfr][keep])
speeds = np.concatenate(speeds)
speed_edges = np.percentile(speeds, [25, 50, 75])

# Pass 2: per trial
out[3] = np.searchsorted(speed_edges, speed[idx])
```

iii. The agent chose global quartiles to ensure consistent speed bin meanings across sessions.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Global percentile edges at 25%, 50%, 75% across all sessions' running frames. `np.searchsorted` gives bins 0 (slowest 25%), 1, 2, 3 (fastest 25%).

ii.
```python
speed_edges = np.percentile(speeds, [25, 50, 75])
out[3] = np.searchsorted(speed_edges, speed[idx])
```

iii. Global thresholds rather than per-session.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Same frame indices (`idx`) as neural data.

ii.
```python
out[3] = np.searchsorted(speed_edges, speed[idx])
neural_s.append(np.ascontiguousarray(spk[:, idx]))
```

iii. Frame-based alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The behavior is clipped to the minimum of neural and behavioral frame counts: `nfr = min(spk.shape[1], len(beh['ft']))`. Lick frames outside the valid range are dropped. Trials with fewer than 2 frames are skipped. NaN values in `ft_trInd` are excluded. Trials with unmapped stimulus names are skipped.

ii.
```python
nfr = min(spk.shape[1], len(beh['ft']))
lickfr = lickfr[(lickfr >= 0) & (lickfr < nfr)]
if len(idx) < 2:
    nskip += 1; continue
ok = ~np.isnan(tr)
```

iii. The agent handles edge cases defensively.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the spike files (89 files, 405 GB total) is by far the most time-consuming step. The plane-by-plane neuron selection within each file also has overhead.

ii.
```python
spks = np.load(os.path.join(ROOT, 'spk', '%s_neural_data.npy' % k),
               allow_pickle=True).item()['spks']
```

iii. I/O bound on the neural data files.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The plane-by-plane neuron extraction loop could be replaced by concatenating all planes first and then indexing, though this would use more memory. The trial-by-trial frame extraction is already vectorized via `searchsorted`.

ii.
```python
for p in range(len(spks)):
    n = spks[p].shape[0]
    take = sel[(sel >= off) & (sel < off + n)] - off
    spk[o2:o2 + len(take)] = spks[p][take]
    o2 += len(take)
    off += n
```

iii. The plane loop is a memory optimization trade-off.

## 12-c. What processing does the code repeat multiple times?

i. The behavior is loaded twice: once in `load_sessions()` which reads ALL behavior files for stimulus map construction, and again implicitly since the session objects persist. The first pass over all sessions computes speed edges and first dates, then the second pass processes each session fully. The behavior files are all loaded up front in `load_sessions()`.

ii.
```python
# load_sessions: loads ALL behavior files
for exp_type in exp_info:
    Beh = np.load(os.path.join(ROOT, 'beh', 'Beh_%s.npy' % exp_type), ...)

# pass 1: speed + dates
for k in keys:
    beh = sessions[k]['beh']
    ...

# pass 2: full processing
for si, k in enumerate(keys):
    beh = sessions[k]['beh']
    ...
```

iii. The two-pass design ensures global speed quartiles are computed before any trial processing, but behavior data is traversed multiple times.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads all behavior files in `load_sessions()` even though much of the data (experiment types, stimulus maps for dropped trials) is not used. The plane-by-plane extraction could be simplified if memory were not a constraint. The stimulus mapping via WallName->TrialStim is more complex than needed since TrialStim could be used directly per trial.

ii.
```python
# All exp_type files loaded even though behavior is identical across them
for exp_type in exp_info:
    Beh = np.load(os.path.join(ROOT, 'beh', 'Beh_%s.npy' % exp_type), ...)
```

iii. The extra loading builds a complete stimulus map but most of the behavior data loaded is redundant.
