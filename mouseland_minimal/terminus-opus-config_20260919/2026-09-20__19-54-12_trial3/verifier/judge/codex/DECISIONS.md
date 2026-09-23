# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the master session index from `beh/Imaging_Exp_info.npy`, then reads every `Beh_<exp_type>.npy` file in `load_sessions()` and builds one dictionary entry per unique `mouse_date_block` session. For each retained session in the main loop it then separately loads the matching retinotopy `.npz` file and the spike `.npy` file.

ii. ```python
exp_info = np.load(os.path.join(ROOT, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()
for exp_type in exp_info:
    Beh = np.load(os.path.join(ROOT, 'beh', 'Beh_%s.npy' % exp_type),
                  allow_pickle=True).item()
```
```python
key = '%s_%s_%s' % (db['mname'], db['datexp'], db['blk'])
...
ret = np.load(os.path.join(ROOT, 'retinotopy', '%s_%s_trans.npz'
                           % (db['mname'], db['datexp'])), allow_pickle=True)
spks = np.load(os.path.join(ROOT, 'spk', '%s_neural_data.npy' % k),
               allow_pickle=True).item()['spks']
```

iii. In the trajectory, the AI said it found 89 unique sessions from 19 mice and concluded that duplicate behavior entries across experiment types were identical, so it merged them into a single per-session record. It also justified loading all behavior first in order to union stimulus labels across experiment-type files before converting each session.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined by `db['mname']`. The final `subjects` list is the sorted set of mouse names across retained session keys, and each session gets `subject_idx` from its mouse name.

ii. ```python
subjects = sorted(set(sessions[k]['db']['mname'] for k in keys))
...
data['subject_idx'].append(subjects.index(db['mname']))
```

iii. In the trajectory, the AI repeatedly summarized the dataset as 89 sessions from 19 mice and treated `mname` as the canonical subject identifier, matching how it explored `Imaging_Exp_info.npy`.

## 1-c. How are the data split into sessions?

i. A session is defined as one `mname/datexp/blk` combination. The AI builds `key = '%s_%s_%s'` from those fields and stores only one entry per key in `sessions`.

ii. ```python
key = '%s_%s_%s' % (db['mname'], db['datexp'], db['blk'])
s = sessions.setdefault(key, {'beh': beh, 'db': db, 'stim_map': {},
                              'exp_types': []})
```

iii. The trajectory states that the same recording appears in multiple experiment-type behavior files, so the AI chose to collapse them into one session and only keep the per-session union of metadata it needed, especially the stimulus-name mapping.

## 1-d. How are the data split into trials?

i. Trials are split by grouping frames according to `beh['ft_trInd']`, but only after masking to frames where the animal was running (`ft_move > 0`) and inside the textured corridor (`ft_CorrSpc`). Each trial therefore contains only the kept frame indices for that trial, not every frame from corridor entry to corridor exit.

ii. ```python
def trial_frames(beh, nfr):
    ft_tr = beh['ft_trInd'][:nfr]
    keep = (beh['ft_move'][:nfr] > 0) & beh['ft_CorrSpc'][:nfr]
    idx = np.where(keep)[0]
    tr = ft_tr[idx]
    ...
    return [idx[bounds[t]:bounds[t + 1]] for t in range(int(beh['ntrials']))]
```

iii. The AI explicitly justified this in the script docstring and trajectory: it said the paper/code analyze `VRmove & ft_CorrSpc`, so it treated trials as corridor traversals aligned to corridor entry but restricted to running time points in the textured section.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops trials that have fewer than 2 kept frames after the running-and-corridor mask, and it also drops trials whose wall name cannot be mapped to one of the allowed canonical stimulus names. It does not apply the reference solution's long-trial percentile cutoff.

ii. ```python
for t, idx in enumerate(tidx):
    if len(idx) < 2:
        nskip += 1
        continue
    name = smap.get(str(beh['WallName'][t]), None)
    if name is None or name not in STIM_NAMES:
        nskip += 1
        continue
```

iii. In the trajectory, the AI said there were 309 `circle3` trials with no canonical id in the paper's analysis and decided to drop them. It also described `len(idx) < 2` trials as unusable after restricting to running frames.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data come from `spks` in each session's spike file and from `iarea` in the matching retinotopy file for region labels.

ii. ```python
ret = np.load(os.path.join(ROOT, 'retinotopy', '%s_%s_trans.npz'
                           % (db['mname'], db['datexp'])), allow_pickle=True)
aidx = area_index(ret['iarea'])

spks = np.load(os.path.join(ROOT, 'spk', '%s_neural_data.npy' % k),
               allow_pickle=True).item()['spks']
```

iii. The trajectory notes that spike files are dicts with a `spks` list of planes and that the retinotopy files supply per-neuron area labels. The AI then followed `utils.load_spk`/`utils.neu_area_ID` from the paper code.

## 2-b. How is the `neural` data processed?

i. The AI concatenates the selected neurons across imaging planes, keeps the data at one column per imaging frame, slices each trial by its kept frame indices, and stores each trial as a contiguous float32 array. It does not do any temporal rebinning or additional signal transformation.

ii. ```python
spk = np.empty((len(sel), spks[0].shape[1]), dtype=np.float32)
for p in range(len(spks)):
    n = spks[p].shape[0]
    take = sel[(sel >= off) & (sel < off + n)] - off
    spk[o2:o2 + len(take)] = spks[p][take]
```
```python
neural_s.append(np.ascontiguousarray(spk[:, idx]))
```

iii. The AI's stated rationale was that the paper used raw deconvolved Suite2p traces at the native imaging-frame resolution, so it should keep one bin per frame. It also said it chose a subset of neurons for tractability because the full spike data occupy about 405 GB on disk.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are first filtered to the four retinotopy-defined visual area groups (`V1`, `mHV`, `lHV`, `aHV`). After that, if more than `args.nneurons` neurons remain, the AI randomly subsamples at most 1000 neurons per session.

ii. ```python
valid = np.where(aidx >= 0)[0]
if len(valid) > args.nneurons:
    sel = np.sort(rng.choice(valid, args.nneurons, replace=False))
else:
    sel = valid
```

iii. The trajectory and top-of-file comments explicitly justify the area filter by the paper's area grouping and justify the random 1000-neuron cap as a tractability measure because the raw dataset is too large to materialize in full.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns trials to corridor entry conceptually, but the actual per-trial arrays contain only the running frames inside the textured corridor for that trial. Trials have variable lengths determined by how many masked frames survive.

ii. ```python
# Trials are corridor traversals, temporally aligned to corridor entry.  Only the
# 4-m textured part of the corridor is kept (beh['ft_CorrSpc'])
# Only frames in which the animal was running are kept (beh['ft_move'] > 0 ...)
```
```python
tidx = trial_frames(beh, nfr)
...
neural_s.append(np.ascontiguousarray(spk[:, idx]))
```

iii. The AI said in the trajectory that it was following the paper's `VRmove & ft_CorrSpc` analysis mask. It treated this as the correct way to align all streams to trial start while restricting analyses to running frames.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data stay at the native imaging-frame resolution. The AI estimates the bin size from the median frame interval and writes it to metadata in milliseconds; it does not rebin.

ii. ```python
dts.append(np.median(np.diff(ft)) * SEC_PER_DAY)
...
'time_bin_size': float(np.median(dts) * 1000.0),
```

iii. The trajectory repeatedly says the data are kept at the native mesoscope frame rate of about 3.17 Hz, or roughly 315 ms per bin, with one time bin per imaging frame.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-frame timestamps `ft` and the per-trial cue timestamp `SoundTime`.

ii. ```python
ft = beh['ft'][:nfr]
...
tcue = (beh['SoundTime'][t] - ft[idx]) * SEC_PER_DAY
```

iii. The trajectory shows that the AI considered both `SoundFr` and `SoundTime` during exploration, then chose `SoundTime` because it could compute the cue offset directly in seconds from the timestamp arrays.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each kept frame in a trial, the AI subtracts the frame timestamp from the trial's cue timestamp and converts the MATLAB-day difference to seconds. The result is positive before the cue and negative after it.

ii. ```python
tcue = (beh['SoundTime'][t] - ft[idx]) * SEC_PER_DAY
...
inp[0] = tcue
```

iii. In the trajectory, the AI explicitly described this variable as “seconds until the sound cue (negative after the cue)” and treated the direct time subtraction as simpler than reconstructing cue time from frame indices.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The cue-offset signal is computed on the exact same kept frame indices `idx` that are used to slice the neural array for that trial.

ii. ```python
tcue = (beh['SoundTime'][t] - ft[idx]) * SEC_PER_DAY
...
neural_s.append(np.ascontiguousarray(spk[:, idx]))
```

iii. The AI's general alignment rationale in the trajectory was that all streams should share the same running-frame trial window, so the cue input and neural data are aligned bin-for-bin by construction.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from each session's mouse name `mname` and date string `datexp` from the experiment index metadata.

ii. ```python
beh, db = sessions[k]['beh'], sessions[k]['db']
...
d = datetime.date(*map(int, db['datexp'].split('_')))
if db['mname'] not in first_date or d < first_date[db['mname']]:
    first_date[db['mname']] = d
```

iii. The trajectory says the AI inspected `exp_info` to decide how to define training day and concluded that session dates in the experiment index were the right raw source for it.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI computes the number of calendar days elapsed since the first imaging date for that mouse, converts it to float, and broadcasts that scalar across every kept frame of the trial.

ii. ```python
day = float((datetime.date(*map(int, db['datexp'].split('_')))
             - first_date[db['mname']]).days)
...
inp[1] = day
```

iii. In metadata and trajectory summaries, the AI described this variable as “days elapsed since the first imaging session of that mouse.” It did not justify switching from ordinal session count to elapsed calendar days beyond that interpretation of “day of training.”

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from per-frame timestamps `ft` and the per-trial timestamp `Trial_start_time`.

ii. ```python
ft = beh['ft'][:nfr]
...
tt = (ft[idx] - beh['Trial_start_time'][t]) * SEC_PER_DAY
```

iii. The trajectory indicates that the AI explored `StartFr`, `GrayFr`, `EndFr`, and the timestamp fields, then settled on direct timestamp subtraction for time-based inputs.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each kept frame in a trial, the AI subtracts the trial start timestamp from the frame timestamp and converts the result from MATLAB days to seconds.

ii. ```python
tt = (ft[idx] - beh['Trial_start_time'][t]) * SEC_PER_DAY
...
inp[2] = tt
```

iii. The AI's rationale was the same as for cue timing: it wanted a direct time difference on the same frame grid, and described the output as “seconds since entry into the corridor.”

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed from the same `idx` frame indices that define the neural slice for that trial, so it is aligned one-to-one with the neural bins.

ii. ```python
tt = (ft[idx] - beh['Trial_start_time'][t]) * SEC_PER_DAY
...
neural_s.append(np.ascontiguousarray(spk[:, idx]))
```

iii. The trajectory consistently states that all per-trial variables are constructed on the same kept-frame window as the neural data.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial `isRew` flag.

ii. ```python
inp[3] = float(beh['isRew'][t])
```

iii. The AI explicitly checked `isRew` in the trajectory and concluded that it marked whether the current corridor was the rewarded corridor, which it used as reward availability.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No transformation beyond casting/broadcasting is applied. The scalar `isRew` value for the trial is copied across all bins in the trial.

ii. ```python
inp = np.empty((4, T), dtype=np.float32)
...
inp[3] = float(beh['isRew'][t])
```

iii. The trajectory says the AI validated the semantics of `isRew` against task-session reward information and then used it directly.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The output stimulus label is derived from `WallName`, but only after building a wall-to-canonical-name mapping from `TrialStim` across all behavior files for the session.

ii. ```python
for w, c in zip(beh['WallName'], beh['TrialStim']):
    if str(c) != 'stimulus_of_trial':
        s['stim_map'][str(w)] = str(c)
```
```python
name = smap.get(str(beh['WallName'][t]), None)
out[0] = STIM_NAMES.index(name)
```

iii. The AI justified this strongly in the trajectory: it argued that `TrialStim` carries the paper's canonical/functional stimulus identity, while physical wall textures differ across mice, so it wanted a shared decoder label space across sessions.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI unions canonical names across duplicate behavior files, keeps only the seven canonical names in `STIM_NAMES`, drops trials with unmapped names (notably `circle3`), and writes the per-trial category index across all bins of the trial.

ii. ```python
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2']
```
```python
if name is None or name not in STIM_NAMES:
    nskip += 1
    continue
...
out[0] = STIM_NAMES.index(name)
```

iii. In the trajectory, the AI said it found 309 `circle3` trials in four sessions, judged that `circle3` “never enters any analysis in the paper,” and chose to drop those trials instead of creating an extra category.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from the session-level lick frame list `LickFr`.

ii. ```python
lickfr = np.round(beh['LickFr']).astype(int) if len(beh['LickFr']) else np.zeros(0, int)
lickfr = lickfr[(lickfr >= 0) & (lickfr < nfr)]
is_lick = np.zeros(nfr, dtype=np.int64)
is_lick[lickfr] = 1
```

iii. The trajectory shows that the AI explicitly checked lick-frame format and sanity before finalizing the converter, then used the frame list directly to form a binary lick trace.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The AI rounds each lick frame to the nearest integer frame, clips licks outside the imaged range, and creates a binary vector where a frame is 1 if at least one lick landed in that frame.

ii. ```python
lickfr = np.round(beh['LickFr']).astype(int)
lickfr = lickfr[(lickfr >= 0) & (lickfr < nfr)]
is_lick = np.zeros(nfr, dtype=np.int64)
is_lick[lickfr] = 1
```

iii. Its trajectory rationale was mostly practical: confirm `LickFr` semantics, then convert it to a per-frame binary output on the imaging grid.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The frame-wise lick vector is indexed by the same per-trial kept frame indices `idx` that slice the neural array.

ii. ```python
out[1] = is_lick[idx]
...
neural_s.append(np.ascontiguousarray(spk[:, idx]))
```

iii. The AI's general alignment rule was to put every time-varying variable on the same kept-frame window as neural activity, so licking is aligned bin-by-bin with the neural data.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position comes from the per-frame `ft_Pos` values.

ii. ```python
pos = beh['ft_Pos'][:nfr]
...
out[2] = np.clip((pos[idx] / 10.0).astype(int), 0, 3)
```

iii. The AI noted in the trajectory that corridor position is measured in decimeters, with 40 dm covering the textured corridor and 20 dm the grey space, which motivated straightforward conversion to 1 m bins.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI divides position by 10 decimeters, casts to integer bins, and clips to 0-3 to create four 1-meter position categories.

ii. ```python
out[2] = np.clip((pos[idx] / 10.0).astype(int), 0, 3)  # 4 x 1 m bins
```

iii. The top-of-file comments explicitly justify keeping only the 4 m textured part because that is exactly the corridor segment the position output is meant to decode.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. It is thresholded into four equal-length spatial bins: `[0,1)`, `[1,2)`, `[2,3)`, and `[3,4]` meters via integer division by 10 dm and clipping.

ii. ```python
'position_bin_edges_m': [0.0, 1.0, 2.0, 3.0, 4.0],
...
out[2] = np.clip((pos[idx] / 10.0).astype(int), 0, 3)
```

iii. The AI described these as the task-required “4 x 1 m bins” in both the code comments and metadata.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. The position trace is sampled on the same kept frame indices `idx` used for the neural slice.

ii. ```python
out[2] = np.clip((pos[idx] / 10.0).astype(int), 0, 3)
...
neural_s.append(np.ascontiguousarray(spk[:, idx]))
```

iii. The AI's trajectory rationale was that all continuous/frame-wise signals should be indexed by the same running-frame corridor window, which enforces alignment with neural activity.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from the per-frame `ft_RunSpeed` values.

ii. ```python
speed = beh['ft_RunSpeed'][:nfr]
...
out[3] = np.searchsorted(speed_edges, speed[idx])
```

iii. The trajectory says the AI computed running-speed statistics over all analyzed running frames before choosing the final binning rule.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. In a first pass over all sessions, the AI collects speed values from frames that satisfy `ft_move > 0` and `ft_CorrSpc`, computes global 25th/50th/75th percentile thresholds, and then discretizes each kept frame by those thresholds.

ii. ```python
keep = (beh['ft_move'][:nfr] > 0) & beh['ft_CorrSpc'][:nfr]
speeds.append(beh['ft_RunSpeed'][:nfr][keep])
...
speed_edges = np.percentile(speeds, [25, 50, 75])
```

iii. In the trajectory, the AI said it wanted global running-speed quartiles over all analyzed frames. It did not discuss the reference solution's per-session rank-based quartiles; its stated goal was a shared decoder label space.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The AI uses three global percentile edges and `np.searchsorted` to assign each kept frame to one of four speed bins.

ii. ```python
speed_edges = np.percentile(speeds, [25, 50, 75])
...
out[3] = np.searchsorted(speed_edges, speed[idx])
```

iii. The trajectory describes these as global quartile bins derived from all analyzed running frames in the dataset, not separately within each session.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is read on the imaging-frame grid and then indexed with the same per-trial frame indices `idx` used for neural data.

ii. ```python
out[3] = np.searchsorted(speed_edges, speed[idx])
...
neural_s.append(np.ascontiguousarray(spk[:, idx]))
```

iii. The AI's alignment rationale was the same as for the other time-varying signals: use the shared kept-frame trial window for every stream.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI truncates all frame-wise streams to the shared imaged length `nfr = min(spk.shape[1], len(beh['ft']))`, clips lick frames to the valid range, drops NaN trial labels inside `ft_trInd`, and skips trials that become too short or lack a usable stimulus label.

ii. ```python
nfr = min(spk.shape[1], len(beh['ft']))
...
lickfr = lickfr[(lickfr >= 0) & (lickfr < nfr)]
...
ok = ~np.isnan(tr)
idx, tr = idx[ok], tr[ok].astype(int)
```

iii. The trajectory says the AI explicitly checked for NaNs in key frame-marker variables and verified retinotopy/spike alignment, then used these range and NaN guards as the lightweight data-cleaning layer.

## 12-a. What are the most time-consuming steps of the code?

i. The dominant cost is loading each session's spike file and copying the selected neurons out plane by plane; reading all behavior files up front is minor by comparison.

ii. ```python
spks = np.load(os.path.join(ROOT, 'spk', '%s_neural_data.npy' % k),
               allow_pickle=True).item()['spks']
...
for p in range(len(spks)):
    ...
    spk[o2:o2 + len(take)] = spks[p][take]
```

iii. The trajectory explicitly reports about 8-10 seconds to load one spike file and estimates a 15-20 minute total I/O cost across the full dataset. The AI repeatedly described spike-file I/O as the main tractability issue.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining obvious Python loops are the plane-by-plane neuron gathering loop and the per-trial assembly loop. The code already vectorizes trial-frame grouping better than the human reference by masking/sorting once and slicing with `searchsorted`.

ii. ```python
for p in range(len(spks)):
    n = spks[p].shape[0]
    take = sel[(sel >= off) & (sel < off + n)] - off
    spk[o2:o2 + len(take)] = spks[p][take]
```
```python
for t, idx in enumerate(tidx):
    ...
    neural_s.append(np.ascontiguousarray(spk[:, idx]))
    input_s.append(inp)
    output_s.append(out)
```

iii. The trajectory does not explicitly dwell on vectorization, but the implemented `trial_frames()` shows that the AI had already optimized away the reference solution's repeated full scan over frames for each trial.

## 12-c. What processing does the code repeat multiple times?

i. Inside the per-trial loop it repeatedly does list/index lookups such as `STIM_NAMES.index(name)`, and for each session it repeatedly does `subjects.index(db['mname'])` instead of precomputing dictionaries. It also rebuilds per-session `stim_map` by scanning duplicate behavior entries across experiment types.

ii. ```python
out[0] = STIM_NAMES.index(name)
...
data['subject_idx'].append(subjects.index(db['mname']))
```
```python
for exp_type in exp_info:
    Beh = np.load(os.path.join(ROOT, 'beh', 'Beh_%s.npy' % exp_type),
                  allow_pickle=True).item()
    ...
    for w, c in zip(beh['WallName'], beh['TrialStim']):
        ...
```

iii. The trajectory explains the repeated `stim_map` construction as necessary to union canonical stimulus names across experiment-type files. It does not justify the repeated `.index(...)` lookups, which appear to be incidental implementation choices.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The converter builds rich session metadata (`exp_types`, `cohort`, `reward_type`, `nneurons_recorded`, `source`, descriptions, etc.) and computes per-session `dts`, but none of that affects the downstream decoder beyond a few metadata summaries. It also scans all duplicate behavior files to build `stim_map`, even though only the final wall-to-label map is used.

ii. ```python
session_info.append({'session': k, 'mouse': db['mname'], 'date': db['datexp'],
                     'block': db['blk'],
                     'exp_types': sorted(set(sessions[k]['exp_types'])),
                     'cohort': str(db.get('exptype', 'naive')),
                     'reward_type': str(db.get('rewType', 'None')),
                     'day_of_training': day,
                     'ntrials': len(neural_s),
                     'nneurons': len(sel),
                     'nneurons_recorded': int(nneu_all)})
```
```python
'sampling_rate_hz': float(1000.0 / (np.median(dts) * 1000.0)),
...
'session_info': session_info,
'source': 'Zhong et al., Unsupervised pretraining in biological neural networks',
```

iii. The trajectory justifies some of this as bookkeeping and validation support, but not as necessary for the decoder itself. These additions are mostly extra metadata rather than core conversion logic.
