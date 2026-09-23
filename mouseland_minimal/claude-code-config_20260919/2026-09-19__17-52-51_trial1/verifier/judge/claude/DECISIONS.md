# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads from the same three directories under `data`: `beh` for behavior (`Imaging_Exp_info.npy` and `Beh_<exp_type>.npy`), `spk` for deconvolved calcium traces, and `retinotopy` for visual area assignments. `Imaging_Exp_info.npy` is loaded first as the master index. Behavior files are loaded once per experiment type and keyed by session. Spike files and retinotopy are loaded per session during processing.

ii.
```python
exp_info = np.load(os.path.join(root, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()
```
```python
B = np.load(os.path.join(root, 'beh', 'Beh_%s.npy' % exp_type),
            allow_pickle=True).item()
```
```python
spks = np.load(spk_path, allow_pickle=True).item()['spks']
```
```python
ret = np.load(os.path.join(root, 'retinotopy', '%s_%s_trans.npz' %
                           (session['mname'], session['datexp'])), allow_pickle=True)
```

iii. The AI explored the data directory structure, examined the index file, and matched the loading approach to `utils.py` in the reference code. The behavior is loaded once per experiment type to avoid redundant reads.

## 1-b. How are the data split into subjects?

i. The mouse name is taken from `mname` in the index entries. Subjects are the sorted unique mouse names across all sessions.

ii.
```python
subjects = sorted({s['mname'] for s in sessions.values()})
```
```python
data['subject_idx'].append(subjects.index(s['mname']))
```

iii. The AI directly used the `mname` field from the experiment info, which already identifies each mouse.

## 1-c. How are the data split into sessions?

i. A session is one unique recording identified by `(mname, datexp, blk)`. The AI de-duplicates recordings that appear under multiple experiment types in `Imaging_Exp_info`, keeping only one entry per unique key. This yields 89 sessions.

ii.
```python
def session_key(d):
    return '%s_%s_%s' % (d['mname'], d['datexp'], d['blk'])
```
```python
sessions = OrderedDict()
for exp_type, db in exp_info.items():
    for d in db:
        kn = session_key(d)
        s = sessions.setdefault(kn, { ... })
```

iii. The AI verified that the 142 entries in the index correspond to 89 unique recordings (matching the paper's "89 recordings in 19 mice") and that behavior dicts are identical across analysis types for the same recording.

## 1-d. How are the data split into trials?

i. A trial is one traversal of the 4m textured corridor. The AI splits frames into trials using `ft_trInd`, but additionally filters to only keep frames where the mouse was running (`ft_move > 0`) AND inside the textured corridor (`ft_CorrSpc`). This "running frames only" filter drops all timepoints when the mouse was stationary.

ii.
```python
def trial_frames(beh, nfr):
    ft_trInd = beh['ft_trInd'][:nfr]
    valid = (beh['ft_move'][:nfr] > 0) & beh['ft_CorrSpc'][:nfr] & ~np.isnan(ft_trInd)
    frames = np.nonzero(valid)[0]
    tr = ft_trInd[frames].astype(np.int64)
    order = np.argsort(tr, kind='stable')
    frames, tr = frames[order], tr[order]
    bounds = np.searchsorted(tr, np.arange(int(beh['ntrials']) + 1))
    return [frames[bounds[i]:bounds[i + 1]] for i in range(int(beh['ntrials']))]
```

iii. The AI justified this by citing the paper's statement "We only considered timepoints during running for analysis" and the reference code's `fr_valid = VRmove & isCorridor` pattern from `utils.py`. The AI argued that since the decoder classifies each timepoint independently, dropping stopped frames costs no temporal structure.

## 1-e. How are trials filtered based on quality controls?

i. The AI requires at least 2 frames per trial and skips trials with non-finite `Trial_start_time` or `SoundTime`. There is no upper-bound filtering on trial length (no percentile-based cutoff). All 38,110 trials with >= 2 frames are kept.

ii.
```python
if len(fr) < 2:
    continue
```
```python
if not np.isfinite(t0) or not np.isfinite(tcue):
    continue
```

iii. The AI stated that all 38,110 trials have >= 11 frames and none were dropped. The AI did not implement a trial length cutoff.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy`, which contains a list of one neurons-by-frames array per imaging plane, concatenated into a single array. The visual area of each neuron comes from `iarea` in `retinotopy/<mouse>_<date>_trans.npz`.

ii.
```python
spks = np.load(spk_path, allow_pickle=True).item()['spks']
```
```python
ret = np.load(os.path.join(root, 'retinotopy', '%s_%s_trans.npz' %
                           (session['mname'], session['datexp'])), allow_pickle=True)
region = neu_area_idx(np.asarray(ret['iarea']))
```

iii. The AI directly used the deconvolved traces from the spike files, consistent with the paper's statement that analyses used deconvolved data.

## 2-b. How is the `neural` data processed?

i. The AI randomly subsamples neurons to a maximum of 2000 per session (with a fixed random seed). The selected neurons' traces are extracted at the frames belonging to each trial. Data is stored as float32.

ii.
```python
candidates = np.nonzero(region >= 0)[0]
rng = np.random.default_rng(seed)
if max_neurons is not None and len(candidates) > max_neurons:
    rows = np.sort(rng.choice(candidates, size=max_neurons, replace=False))
else:
    rows = candidates
```
```python
out = np.empty((len(rows), len(frames)), dtype=np.float32)
```

iii. The AI justified the 2000-neuron cap by running a pilot at various neuron counts and finding that the decoder's SVD initialization switches to a lossy random projection above 2000 neurons, which collapses accuracy. This also keeps the output file manageable at 6.6 GB vs ~175 GB.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are restricted to the four visual-area groups (V1, mHV, lHV, aHV) using the same area codes as the reference. Neurons with `iarea == -1` or `iarea == 7` are dropped. Then a random subsample of up to 2000 neurons is taken per session.

ii.
```python
def neu_area_idx(iarea):
    idx = np.full(len(iarea), -1, dtype=np.int64)
    idx[iarea == 8] = 0                                              # V1
    idx[np.isin(iarea, [0, 1, 2, 9])] = 1                            # medial HVAs
    idx[np.isin(iarea, [5, 6])] = 2                                  # lateral HVAs
    idx[np.isin(iarea, [3, 4])] = 3                                  # anterior HVAs
    return idx
```

iii. The AI cited the paper's use of these four area groups in all analyses.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry (trial start). Each trial contains variable-length frames from entry to end of traversal. `off_start = 0.0`, `off_end = None`.

ii.
```python
'off_start': 0.0,
'off_end': None,
```

iii. The AI noted that trial duration is behavior-dependent so `off_end` varies per trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The imaging frames are the time bins. The AI computes the frame rate from the median of per-session estimates derived from `ft` timestamps, yielding ~314.7 ms bins.

ii.
```python
frame_rates.append(1.0 / (np.median(np.diff(b['ft'])) * 86400.0))
```
```python
time_bin_size = 1000.0 / float(np.median(frame_rates))
```

iii. The AI computed the frame rate from the data rather than using a hardcoded constant.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime`, the absolute timestamp of the sound cue for each trial, and `ft`, the absolute timestamps of each imaging frame.

ii.
```python
tcue = beh['SoundTime'][trial]
```
```python
to_cue = (tcue - ft[fr]) * 86400.0
```

iii. The AI used `SoundTime` (absolute timestamp) rather than `SoundFr` (frame number).

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The time to sound cue is computed as `(SoundTime - ft[frame]) * 86400.0`, converting from MATLAB datenum (days) to seconds. This is positive before the cue and negative after it.

ii.
```python
to_cue = (tcue - ft[fr]) * 86400.0               # seconds until the sound cue
```
```python
inp[0] = to_cue
```

iii. The AI described this as "seconds until the sound cue (positive before the cue, negative after it)".

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed at the same frame indices (`fr`) used for the neural data of that trial.

ii.
```python
to_cue = (tcue - ft[fr]) * 86400.0
```

iii. All data streams use the same frame indices per trial.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the `datexp` field of each session, which contains the recording date string. The AI also derives the mouse's first recording date.

ii.
```python
first_date = {}
for kn, s in sessions.items():
    d = datetime.date(*map(int, s['datexp'].split('_')))
    first_date[s['mname']] = min(first_date.get(s['mname'], d), d)
day_of_training = {kn: (datetime.date(*map(int, s['datexp'].split('_')))
                        - first_date[s['mname']]).days
                   for kn, s in sessions.items()}
```

iii. The AI noted that "the released data does not contain the absolute start of training" so it measures days since the first recording of that mouse.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI computes the actual calendar day difference between each session's date and the mouse's first recording date. This gives real elapsed days (e.g., 0, 3, 7, 14...) rather than ordinal session counts (0, 1, 2, 3...).

ii.
```python
day_of_training = {kn: (datetime.date(*map(int, s['datexp'].split('_')))
                        - first_date[s['mname']]).days
                   for kn, s in sessions.items()}
```

iii. The AI stated this is "days between this recording and the first recording of the same mouse".

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time`, the absolute timestamp of corridor entry for each trial, and `ft`, the absolute timestamps of each imaging frame.

ii.
```python
t0 = beh['Trial_start_time'][trial]
```
```python
t = (ft[fr] - t0) * 86400.0                      # seconds since corridor entry
```

iii. The AI used the absolute timestamp `Trial_start_time` rather than the frame number `StartFr`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Time since trial start is computed as `(ft[frame] - Trial_start_time) * 86400.0`, converting from MATLAB datenum to seconds. This represents real elapsed time including periods when the mouse was stopped.

ii.
```python
t = (ft[fr] - t0) * 86400.0
```
```python
inp[2] = t
```

iii. The AI described this as "seconds since entry into the corridor (real time, including periods when the mouse was not running)".

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed at the same frame indices used for neural data of that trial.

ii.
```python
t = (ft[fr] - t0) * 86400.0
```

iii. Same frame-based alignment as all other data streams.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, a per-trial flag indicating whether the trial is in the rewarded corridor.

ii.
```python
inp[3] = 1.0 if beh['isRew'][trial] else 0.0
```

iii. The AI noted this is always 0 for the unsupervised and naive cohorts.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No processing needed beyond converting to float (1.0 or 0.0).

ii.
```python
inp[3] = 1.0 if beh['isRew'][trial] else 0.0
```

iii. Direct use of the existing flag.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName` (the wall texture name per trial), `UniqWalls` (unique walls in the session), and `stim_id` vectors from `Imaging_Exp_info` entries. The AI merges the `stim_id` tables across a recording's multiple experiment-type entries to build a canonical stimulus identity mapping.

ii.
```python
CANONICAL_STIM = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
                  'leaf1_swap1', 'leaf1_swap2']
EXTRA_STIM = ['circle3']
STIM_VALUES = CANONICAL_STIM + EXTRA_STIM
```
```python
def build_stim_map(session, beh):
    uniq = list(beh['UniqWalls'])
    mapping = {}
    for exp_type, key, stim_id in session['stim_id_entries']:
        ...
        for wall, sid in zip(uniq, stim_id):
            if not np.isnan(sid):
                sid = int(sid)
                ...
                mapping[wall] = sid
```

iii. The AI followed the paper's canonical stimulus identities (`utils.all_stim`) and used the `stim_id` tables to map rock/brick/wood mice onto equivalent stimuli.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI maps each wall name to one of 8 stimulus identities (circle1, circle2, leaf1, leaf2, leaf3, leaf1_swap1, leaf1_swap2, circle3) using the `stim_id` tables from the experiment info. This preserves individual texture variants rather than grouping them into 4 broad categories. `circle3` is kept as an extra category to avoid discarding trials.

ii.
```python
out[0] = stim_map[wall]
```
```python
'output_values': [STIM_VALUES, LICK_VALUES, POSITION_VALUES, SPEED_VALUES],
```

iii. The AI justified keeping 8 categories by saying the paper's `stim_id` tables distinguish them, and `circle3` was kept as an 8th category rather than discarding 390 trials.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the frame number of every lick in the session.

ii.
```python
lick = np.zeros(nfr, dtype=bool)
if len(beh['LickFr']) > 0:
    lf = np.round(np.asarray(beh['LickFr'], dtype=float))
    lf = lf[np.isfinite(lf)].astype(np.int64)
    lick[lf[(lf >= 0) & (lf < nfr)]] = True
```

iii. The AI used `LickFr` directly.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A frame is 1 if at least one lick falls in it and 0 otherwise. The AI uses `np.round` to convert fractional lick frame numbers to integers (rounding), and also filters out non-finite values.

ii.
```python
lf = np.round(np.asarray(beh['LickFr'], dtype=float))
lf = lf[np.isfinite(lf)].astype(np.int64)
lick[lf[(lf >= 0) & (lf < nfr)]] = True
```
```python
out[1] = lick[fr].astype(np.int64)
```

iii. The AI added extra safety checks for non-finite lick frames.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` indexes the imaging frames, so the lick flag is already on the same grid as neural data. It is taken at the same frame indices used for the trial.

ii.
```python
out[1] = lick[fr].astype(np.int64)
```

iii. Frame-based alignment.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position inside the corridor at each imaging frame, and `Texture_Length` for the bin size.

ii.
```python
pos = beh['ft_Pos'][:nfr]
```
```python
texture_len = float(beh['Texture_Length'])          # 40 units == 4 m
bin_len = texture_len / len(POSITION_VALUES)
```

iii. The AI used the data's own `Texture_Length` field.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The position is divided by `bin_len` (= Texture_Length / 4 = 10 dm) and clipped to [0, 3], producing four 1m bins.

ii.
```python
pos_bin = np.clip((p / bin_len).astype(np.int64), 0, len(POSITION_VALUES) - 1)
```

iii. The four 1m bins match the task specification of "4 equal-length, 1-m-long spatial bins".

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position is divided by 10 (the bin length in position units), truncated to integer, and clipped to [0, 3]. This produces four 1m bins: 0-1m, 1-2m, 2-3m, 3-4m.

ii.
```python
pos_bin = np.clip((p / bin_len).astype(np.int64), 0, len(POSITION_VALUES) - 1)
```
```python
POSITION_VALUES = ['0-1m', '1-2m', '2-3m', '3-4m']
```

iii. Functionally equivalent to the reference's `// 10` approach.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` gives one position per imaging frame. It is taken at the same frame indices as the neural data.

ii.
```python
out[2] = pos_bin
```

iii. Frame-based alignment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed of the mouse at each imaging frame.

ii.
```python
speed = beh['ft_RunSpeed'][:nfr]
```

iii. Direct use of running speed.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes **global** speed quartiles across all sessions in a first pass. It collects all running speeds from all kept frames across the entire dataset, then computes the 25th, 50th, and 75th percentile edges. These global edges are then used to bin each frame's speed via `np.searchsorted`.

ii.
```python
# pass 1 (behaviour only): global running-speed quartiles over all kept timepoints
speeds = []
for kn, s in sessions.items():
    b = beh[kn]
    nfr = len(b['ft'])
    arrays = session_behaviour_arrays(s, b, nfr, day_of_training[kn])
    speeds.append(arrays['speeds'])
speeds = np.concatenate(speeds)
speed_edges = np.percentile(speeds, [25, 50, 75])
```
```python
if speed_edges is not None:
    out[3] = np.searchsorted(speed_edges, s, side='right')
```

iii. The AI justified global quartiles so "the classes mean the same thing in every session". The edges were 10.4 / 20.2 / 32.0 cm/s.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Speed is binned into 4 categories using the global percentile edges (25th, 50th, 75th percentiles) via `np.searchsorted`. Values below 25th percentile get bin 0, etc.

ii.
```python
out[3] = np.searchsorted(speed_edges, s, side='right')
```
```python
SPEED_VALUES = ['speed_q1', 'speed_q2', 'speed_q3', 'speed_q4']
```

iii. The AI chose global quartiles rather than per-session quartiles.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` gives one speed per imaging frame. It is taken at the same frame indices as the neural data.

ii.
```python
s = speed[fr]
```

iii. Frame-based alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI clips behavior arrays to the number of neural frames (`nfr = min(p.shape[1] for p in spks)`). Trials with non-finite `Trial_start_time` or `SoundTime` are skipped. Non-finite lick frames are filtered out. Trials with fewer than 2 frames are dropped. Non-finite position or speed values cause trial exclusion.

ii.
```python
nfr = min(p.shape[1] for p in spks)
```
```python
if not np.isfinite(t0) or not np.isfinite(tcue):
    continue
```
```python
lf = lf[np.isfinite(lf)].astype(np.int64)
```
```python
if not (np.all(np.isfinite(p)) and np.all(np.isfinite(s))):
    continue
```

iii. The AI added extensive safety checks for non-finite values throughout the pipeline.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the spike files from disk, which are very large (~405 GB total). The AI uses multiprocessing (8 workers) to parallelize session processing.

ii.
```python
with Pool(args.workers, maxtasksperchild=1) as pool:
    for i, res in enumerate(pool.imap_unordered(process_session, jobs)):
```

iii. The IO cost of loading neural data dominates.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI's trial frame computation is already vectorized - it uses `np.nonzero`, `np.argsort`, and `np.searchsorted` to group all frames by trial in one pass, rather than scanning per trial. The per-trial loop for building input/output arrays could potentially be vectorized but is negligible compared to IO.

ii.
```python
frames = np.nonzero(valid)[0]
tr = ft_trInd[frames].astype(np.int64)
order = np.argsort(tr, kind='stable')
frames, tr = frames[order], tr[order]
bounds = np.searchsorted(tr, np.arange(int(beh['ntrials']) + 1))
```

iii. The vectorized trial-splitting approach is more efficient than the per-trial scanning in the reference.

## 12-c. What processing does the code repeat multiple times?

i. The AI performs a two-pass approach: a first pass over all sessions (behavior only) to compute global speed quartile edges, and then a second full pass that recomputes behavior arrays (including trial frames) during the main processing. The `session_behaviour_arrays` function is called twice per session.

ii.
```python
# pass 1 (behaviour only): global running-speed quartiles over all kept timepoints
for kn, s in sessions.items():
    b = beh[kn]
    nfr = len(b['ft'])
    arrays = session_behaviour_arrays(s, b, nfr, day_of_training[kn])
    speeds.append(arrays['speeds'])
```

iii. The first pass is necessary because global speed edges must be known before the main pass can assign speed bins. However, this means trial frame computation and some behavior processing is done twice.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI randomly subsamples neurons to 2000 per session, which is extra processing (random selection, index manipulation) not needed by the data format itself. It was motivated by the decoder's SVD limitation. Additionally, the stimulus identity mapping via `stim_id` tables is complex processing that produces 8 categories, while 4 broader categories would have been simpler and more standard.

ii.
```python
rng = np.random.default_rng(seed)
if max_neurons is not None and len(candidates) > max_neurons:
    rows = np.sort(rng.choice(candidates, size=max_neurons, replace=False))
```

iii. The neuron subsampling was motivated by decoder performance rather than data format requirements.
