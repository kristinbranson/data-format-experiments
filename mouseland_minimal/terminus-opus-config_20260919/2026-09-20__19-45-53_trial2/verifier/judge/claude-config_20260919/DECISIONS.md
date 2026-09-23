# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything is read from the three directories under `/app/data`: `beh/` (behaviour),
`spk/` (deconvolved traces) and `retinotopy/` (visual-area label of every neuron).
`beh/Imaging_Exp_info.npy` is the master index; it is a dict keyed by *experiment type*,
each holding a list of entries with `mname`, `datexp`, `blk` (and sometimes `stimtype`).
`load_behavior()` walks that index, opens each `beh/Beh_<exp_type>.npy` **once**, and for
every entry builds the behaviour key `mname_datexp_blk[_stimtype]`. A recording is
identified by `rid = mname_datexp_blk`; the first time an `rid` is seen the behaviour
fields it needs are copied into an in-memory record, and later appearances of the same
`rid` under other experiment types only contribute their wall-name → stimulus-id map.
All 89 recordings' behaviour is cached in RAM before any neural data is touched. Neural
and retinotopy data are then read one session at a time inside the main loop:
`spk/<rid>_neural_data.npy` (a dict whose `spks` entry is one array per imaging plane,
concatenated into neurons × frames) and `retinotopy/<mname>_<datexp>_trans.npz` (`iarea`).

ii.
```python
exp_info = np.load(os.path.join(ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
recs = {}
for exp_type, db in exp_info.items():
    Beh = np.load(os.path.join(ROOT, 'beh', 'Beh_%s.npy' % exp_type), allow_pickle=True).item()
    for ndb in db:
        rid = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
        key = rid + ('_' + ndb['stimtype'] if 'stimtype' in ndb else '')
        beh = Beh[key]
        if rid not in recs:
            recs[rid] = dict(rid=rid, mname=ndb['mname'], datexp=ndb['datexp'], blk=ndb['blk'],
                             ..., ntrials=int(beh['ntrials']), ft=..., ft_trInd=..., ft_CorrSpc=...,
                             ft_move=..., ft_Pos=..., ft_RunSpeed=..., LickFr=..., WallName=...,
                             isRew=..., SoundTime=..., Trial_start_time=...)
```
```python
spk_file = os.path.join(ROOT, 'spk', '%s_neural_data.npy' % rid)
spk = np.concatenate([p for p in np.load(spk_file, allow_pickle=True).item()['spks']], 0)
ret = np.load(os.path.join(ROOT, 'retinotopy', '%s_%s_trans.npz' % (r['mname'], r['datexp'])),
              allow_pickle=True)
iarea = np.asarray(ret['iarea'], dtype=float)
```

iii. The agent read `data_process_script.ipynb` (which documents every `beh` field) and
`utils.load_spk` / `utils.neu_area_ID`, and reproduced exactly the same three-file access
pattern used by the paper's own processing notebook. It explicitly checked that the index
contains 23 experiment types collapsing to 89 unique recordings in 19 mice, and that
behaviour files are the natural grouping unit so each one is opened only once.

## 1-b. How are the data split into subjects (mice)?

i. The mouse is `mname` from the index entry, carried on every record. `subjects` is the
sorted list of unique mouse names (19), and `subject_idx` is each session's index into it.
Sessions are ordered by `(mname, datexp, blk)`, so all sessions of a mouse are contiguous.

ii.
```python
rids = sorted(recs.keys(), key=lambda r: (recs[r]['mname'], recs[r]['datexp'], recs[r]['blk']))
subjects = sorted(set(recs[r]['mname'] for r in rids))
...
data['subject_idx'].append(subjects.index(r['mname']))
...
data['subject_idx'] = np.array(data['subject_idx'], dtype=int)
```

iii. The index already names the mouse of every recording, so no split has to be inferred.
The agent verified the result against the paper ("89 recordings in 19 mice") and printed
the per-subject session counts.

## 1-c. How are the data split into sessions?

i. A session is one recording = one mouse, one date, one block, keyed
`rid = mname_datexp_blk`, which is also the name of the spike file. The same recording is
listed under several experiment types (e.g. `unsup_test1` and
`unsup_train2_before_learning`, or with `stimtype` = `swap1`/`swap2`); it is kept **once**,
the behaviour taken from the first occurrence, but the wall-name → stimulus-id maps of all
occurrences are merged, because each experiment type masks (NaNs) the stimuli it does not
analyse. This yields 89 sessions.

ii.
```python
rid = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
key = rid + ('_' + ndb['stimtype'] if 'stimtype' in ndb else '')
beh = Beh[key]
if rid not in recs:
    recs[rid] = dict(...)          # behaviour stored only on first sighting
r = recs[rid]
r['exp_types'].append(exp_type)
for wall, sid in zip(beh['UniqWalls'], np.asarray(beh['stim_id'], dtype=float)):
    if not np.isnan(sid):
        r['stim_map'][str(wall)] = int(sid)
```

iii. The agent explicitly checked (step 29–32) that duplicate entries for the same `rid`
carry identical `ntrials` and behaviour, and that the per-recording `stim_id` maps never
conflict across experiment types, so merging is safe and de-duplication is lossless for the
behaviour.

## 1-d. How are the data split into trials?

i. A trial is one corridor traversal. The frames of trial *i* are the neural frames that the
behaviour labels with that trial (`ft_trInd == i`), that lie inside the 4 m texture area
(`ft_CorrSpc`), **and** during which the virtual reality was moving, i.e. the animal was
running (`ft_move > 0`). Trials are therefore variable length (11–178 frames, median 21),
and frames in which the animal stopped are removed from the *middle* of a trial, so the
kept frames of a trial are not temporally contiguous. Behaviour is truncated to the number
of imaged frames. Trials that end up with zero frames are skipped.

ii.
```python
def trial_frames(rec, nfr_spk):
    """Frame indices of every trial: running frames inside the texture corridor."""
    nfr = min(len(rec['ft']), nfr_spk)
    keep = rec['ft_CorrSpc'][:nfr] & (rec['ft_move'][:nfr] > 0)
    tr = rec['ft_trInd'][:nfr]
    idx = np.where(keep)[0]
    tr_idx = tr[idx]
    return [idx[tr_idx == i] for i in range(rec['ntrials'])], nfr
```
```python
for ti in range(r['ntrials']):
    ii = frames[ti]
    if len(ii) == 0:
        continue
```

iii. The agent quotes the Methods verbatim in the module docstring: *"We only considered
timepoints during running for analysis, which removed time periods when the task mice
stopped to collect water rewards"*, and notes that `utils.py` applies exactly this mask
(`VRmove = beh['ft_move'] > 0`; `fr_valid = VRmove & isCorridor`). It measured the effect
(38,110 trials, 821,579 running in-corridor frames, every trial keeping ≥11 frames) before
committing.

## 1-e. How are trials filtered based on quality controls?

i. Two filters only. (1) A trial with no running in-corridor frame is dropped. (2) A trial
whose wall texture has no canonical stimulus id in the merged per-recording map is dropped
(309 trials, essentially the `circle3` texture shown in a few sessions), because under the
agent's labelling scheme its category would be undefined. 37,801 of 38,110 trials survive;
no session-level or duration-based curation is applied, and no session is dropped. Sessions
keep between 84 and 789 trials.

ii.
```python
            ii = frames[ti]
            if len(ii) == 0:
                continue
            sid = r['stim_map'].get(r['WallName'][ti], None)
            if sid is None:                     # undefined stimulus category
                n_dropped_stim += 1
                continue
```
```python
        n_trials_dropped_unknown_stimulus=int(n_dropped_stim),
```

iii. The agent's justification is that the running mask already removes the pathological
stationary periods (it measured the surviving per-trial frame counts: 1st percentile 11,
99th percentile 37, max 178), so no length cut is needed; and that a trial whose texture has
no canonical id cannot be given a stimulus label. It checked afterwards that trial
*durations* are mostly sensible (median 7.1 s) and that only 1.2% exceed 60 s, and judged
these "expected given the running-only frame selection".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<rid>_neural_data.npy`, a list of one (neurons × frames) deconvolved
trace array per imaging plane, concatenated along the neuron axis. The area label of each
neuron comes from `iarea` in `retinotopy/<mname>_<datexp>_trans.npz`; the code asserts the
two have the same number of neurons.

ii.
```python
spk = np.concatenate([p for p in np.load(spk_file, allow_pickle=True).item()['spks']], 0)
nneu_all, nfr_spk = spk.shape
ret = np.load(os.path.join(ROOT, 'retinotopy', '%s_%s_trans.npz' % (r['mname'], r['datexp'])),
              allow_pickle=True)
iarea = np.asarray(ret['iarea'], dtype=float)
assert len(iarea) == nneu_all, (rid, len(iarea), nneu_all)
```

iii. This is `utils.load_spk`'s concatenation order, which is the order `iarea` indexes;
the agent verified the neuron counts match for sample sessions before relying on the assert.
The Methods state all analyses were based on deconvolved fluorescence, so no further
derivation is done.

## 2-b. How is the `neural` data processed?

i. Two operations. (1) **Sub-sampling**: only `NNEURONS = 1000` neurons per session are
kept, drawn without replacement from the area-assigned neurons with a fixed-seed
`np.random.default_rng(0)`; recordings hold 20,547–89,577 neurons, so ~98% of the neurons
are discarded. (2) **Z-scoring**: each kept neuron is standardised (mean 0, sd 1) over the
whole recording, not just the kept frames. Neurons with zero variance are dropped. The
result is stored as float32, one (1000 × T) array per trial, variable T, no padding.

ii.
```python
valid = np.where(region_of_neuron >= 0)[0]
sel = np.sort(rng.choice(valid, size=min(NNEURONS, len(valid)), replace=False))

X = spk[sel].astype(np.float32)
del spk
mu = X.mean(axis=1, keepdims=True)
sd = X.std(axis=1, keepdims=True)
good = (sd[:, 0] > 0)
X = (X[good] - mu[good]) / sd[good]          # z-score over the whole recording
sel = sel[good]
```
```python
neural_s.append(np.ascontiguousarray(X[:, ii]))
```

iii. From the docstring: the subset is kept "to keep the dataset tractable (recordings
contain 20,547-89,577 neurons)" — the agent measured 405 GB of spike files and 4.7 M
neurons and wanted an output it could hold in memory and train on (it ended at 3.3 GB).
Z-scoring is justified as "done in the reference code
(utils.get_kfold_reward_response)", which indeed calls `stats.zscore(spk, axis=1)`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only quality filter is the retinotopic area: a neuron is kept if its `iarea` falls in
one of the four groups the paper analyses — V1 = [8], mHV = [0,1,2,9], lHV = [5,6],
aHV = [3,4] — and dropped otherwise (unassigned/NaN and area 7). `brain_region_idx` records
the group of each kept neuron. In addition, neurons with zero standard deviation over the
recording are dropped (they would produce NaNs on z-scoring). The 1000-neuron random draw
(2-b) is a size cap, not a quality control.

ii.
```python
region_of_neuron = np.full(nneu_all, -1, dtype=int)
for ri, reg in enumerate(BRAIN_REGIONS):
    m = np.isin(iarea, AREA_GROUPS[reg])
    region_of_neuron[m] = ri
valid = np.where(region_of_neuron >= 0)[0]
```
```python
good = (sd[:, 0] > 0)
X = (X[good] - mu[good]) / sd[good]
sel = sel[good]
region_idx = region_of_neuron[sel].astype(int)
```

iii. The agent read `utils.neu_area_ID` and the figure code and concluded the authors group
`iarea` into exactly these four areas and apply no further neuron curation beyond the
Suite2p cell classifier that was already run; it checked the `iarea` histogram over all 89
sessions before fixing the groups.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry (trial start). Each trial's array begins at the
first kept frame after the animal entered the corridor and ends at the last kept frame
inside the 4 m texture area; nothing is cut off the end and nothing is padded, so trials
have their own lengths. `off_start = 0.0`, `off_end = None`. Every other stream is indexed
with the identical frame array `ii`, so all streams are aligned frame-by-frame.

ii.
```python
ii = frames[ti]
...
t = (r['ft'][ii] - r['Trial_start_time'][ti]) * 86400.   # s since corridor entry
...
neural_s.append(np.ascontiguousarray(X[:, ii]))
```
```python
temporal_alignment_event='trial start = entry into the virtual-reality corridor',
off_start=0.0,
off_end=None,
```

iii. The agent chose variable-length trials because the format only requires a common bin
*size*, allows `off_end = None`, and the decoder treats each bin independently; a common
window would have to pad or truncate real traversals.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling: the native two-photon frames are the bins. The bin size in
the metadata is the mean across sessions of the per-session median inter-frame interval,
314.85 ms (≈3.176 Hz), which the agent verified is uniform across sessions (0.3147 s). Note
that because non-running frames are removed from inside trials (1-d), consecutive bins of a
stored trial are not always 315 ms apart in real time.

ii.
```python
dts.append(np.median(np.diff(r['ft'][:nfr])) * 86400.)
...
time_bin_size=float(np.mean(dts) * 1000.),
sampling_rate_hz=float(1000. / (np.mean(dts) * 1000.)),
```

iii. The imaging frame is the finest resolution the data has and every behavioural stream is
already sampled on that grid (all the `ft_*` variables), so there is nothing to resample;
the agent measured the frame interval across all 89 sessions rather than hard-coding the
3.17 Hz from the notebook.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `beh['SoundTime']`, the (MATLAB datenum) time of the sound cue on each trial, and
`beh['ft']`, the timestamp of every imaging frame. The frame-number variant `SoundFr` is not
used.

ii.
```python
                    SoundTime=np.asarray(beh['SoundTime'], dtype=np.float64),
```
```python
t_cue = (r['SoundTime'][ti] - r['ft'][ii]) * 86400.      # s until sound cue
```

iii. The agent verified (step 22/26) that `SoundFr` is never NaN in any session, and worked
in the native time base rather than interpolating frame numbers. (Checking here confirms
`SoundTime` and the interpolation of `SoundFr` onto `ft` agree to ~1e-12 s, so the two
routes are equivalent.)

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The cue time minus the timestamp of each kept frame, converted from days to seconds
(× 86400). It is therefore signed and positive *before* the cue, negative after, as the
name "time to sound cue" implies, and is stored as a time-varying float32 row.

ii.
```python
t_cue = (r['SoundTime'][ti] - r['ft'][ii]) * 86400.      # s until sound cue
inp = np.stack([t_cue,
                np.full(len(ii), r['day']),
                t,
                np.full(len(ii), float(r['isRew'][ti]))]).astype(np.float32)
```

iii. Times in the behaviour files are MATLAB datenums (days), so the ×86400 conversion to
seconds; the sign convention follows the variable name in the decoder-task specification.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at exactly the frame indices `ii` used for that trial's neural columns,
so it has the same number of timepoints as the neural array, bin for bin.

ii.
```python
ii = frames[ti]
t_cue = (r['SoundTime'][ti] - r['ft'][ii]) * 86400.
...
neural_s.append(np.ascontiguousarray(X[:, ii]))
input_s.append(inp)
```

iii. All streams in this dataset live on the imaging-frame grid, so indexing every stream
with the same frame array is the alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From `datexp`, the recording date in the session id / index entry, parsed into a
`datetime.date`, together with the mouse name `mname`.

ii.
```python
    first_day = {}
    for rid in rids:
        r = recs[rid]
        d = datetime.date(*[int(x) for x in r['datexp'].split('_')])
        r['date'] = d
        if r['mname'] not in first_day or d < first_day[r['mname']]:
            first_day[r['mname']] = d
```

iii. The date string is the only field that orders sessions within a mouse; the agent noted
the `Imaging_Exp_info` entries carry no explicit training-day counter it could use.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The value is the number of **calendar days elapsed** between the mouse's first imaging
session and this session: 0 for the first recording of each mouse, growing with real
elapsed time (observed range 0–92 days). It is computed over all 89 sessions and broadcast
as a constant row across all bins of every trial of that session.

ii.
```python
    for rid in rids:
        r = recs[rid]
        r['day'] = float((r['date'] - first_day[r['mname']]).days)
```
```python
inp = np.stack([t_cue,
                np.full(len(ii), r['day']),
                t,
                np.full(len(ii), float(r['isRew'][ti]))]).astype(np.float32)
```

iii. The agent treats "day of training" literally as elapsed days of the training/imaging
programme for that animal, anchored at the animal's first recording since the true first
day of handling/training is not in the data.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `beh['Trial_start_time']` ("time when animal enters each corridor") and `beh['ft']`,
the timestamp of every imaging frame. `StartFr` is not used.

ii.
```python
                    Trial_start_time=np.asarray(beh['Trial_start_time'], dtype=np.float64),
```
```python
t = (r['ft'][ii] - r['Trial_start_time'][ti]) * 86400.   # s since corridor entry
```

iii. Corridor entry is the stated alignment event; the agent checked `StartFr` is never NaN
and used the native time variable rather than interpolating the fractional frame number.
(The two agree to ~1e-12 s.)

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Frame time minus trial start time, converted from days to seconds; positive after entry,
starting near 0. Because stationary frames are removed but *time* is not, this variable
still counts wall-clock time spent standing still, so for a small fraction of trials it
reaches very large values (observed maximum 1765 s, 99th percentile 73 s, median trial
duration 7.1 s).

ii.
```python
t = (r['ft'][ii] - r['Trial_start_time'][ti]) * 86400.   # s since corridor entry
inp = np.stack([t_cue, np.full(len(ii), r['day']), t,
                np.full(len(ii), float(r['isRew'][ti]))]).astype(np.float32)
```

iii. A plain time difference in seconds; the agent inspected the resulting duration
distribution after conversion and concluded the long tail was "expected given the
running-only frame selection" (mouse paused mid-corridor).

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Evaluated at the same frame indices `ii` as the neural columns, so identical length and
bin-for-bin correspondence.

ii.
```python
ii = frames[ti]
t = (r['ft'][ii] - r['Trial_start_time'][ti]) * 86400.
neural_s.append(np.ascontiguousarray(X[:, ii]))
```

iii. Same reasoning as 3-c: every stream is indexed by the same imaging-frame array.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `beh['isRew']`, the per-trial boolean marking trials run in the rewarded corridor.

ii.
```python
                    isRew=np.asarray(beh['isRew']).astype(bool),
```
```python
np.full(len(ii), float(r['isRew'][ti]))
```

iii. `isRew` is documented as "boolean value indicating if the trial is a reward trial", so
it is exactly the requested variable; the agent additionally noted that only 28 sessions
(the water-restricted task cohort) have rewards/licks at all.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. None beyond a cast to float (0.0/1.0) and broadcasting the per-trial value across all
bins of the trial, so the input is time-varying in shape but constant within a trial.

ii.
```python
inp = np.stack([t_cue,
                np.full(len(ii), r['day']),
                t,
                np.full(len(ii), float(r['isRew'][ti]))]).astype(np.float32)
```

iii. The format asks for all four inputs stacked into one (4, T) array, so the per-trial
scalar is repeated across timepoints; no other processing is needed.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `beh['WallName']` (the texture name of each trial) mapped through a per-recording
dictionary built from `beh['UniqWalls']` and `beh['stim_id']`, merged over every experiment
type in which that recording appears (each experiment type NaNs out the stimuli it does not
analyse, so merging fills most gaps).

ii.
```python
            for wall, sid in zip(beh['UniqWalls'], np.asarray(beh['stim_id'], dtype=float)):
                if not np.isnan(sid):
                    r['stim_map'][str(wall)] = int(sid)
```
```python
            sid = r['stim_map'].get(r['WallName'][ti], None)
```

iii. The agent found the mapping documented in `data_process_script.ipynb`
("beh['stim_id']: ID of stimuli; 0:circle1, 1:circle2, 2:leaf1, 3:leaf2, 4:leaf3,
5:leaf1_swap1, 6:leaf1_swap2") and adopted it as "the canonical stimulus ids used
throughout the paper's code". It verified that the merged per-recording maps never conflict.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The wall name of the trial is looked up in the merged map and the resulting `stim_id`
(0–6) is used directly as the categorical label, broadcast across all bins of the trial.
`output_values[0]` is `['circle1','circle2','leaf1','leaf2','leaf3','leaf1_swap1',
'leaf1_swap2']`, i.e. 7 classes. Trials whose wall is unmapped (309, mostly `circle3`) are
dropped. Resulting class fractions: circle1 0.319, circle2 0.060, leaf1 0.335, leaf2 0.170,
leaf3 0.059, leaf1_swap1 0.027, leaf1_swap2 0.029.

Note what `stim_id` actually is in this dataset: it is a *session-relative role* code, not a
texture identity. In rock/wood mice `rock1` is given id 0 and `wood1` id 2, and in mice
where circle was the rewarded texture `leaf1` is given id 0 and `circle1` id 2 (in
`utils.py`, `uniqW[stim_id==2]` is commented "get the name of reward stimulus"). Checking
every trial of the dataset against its wall name, 10,805 of 38,110 trials (28%) receive a
label whose name belongs to a different texture family than the wall actually shown
(e.g. `wood1`→"leaf1" 2517 trials, `rock1`→"circle1" 2454, `circle1`→"leaf1" 679,
`leaf1`→"circle1" 796).

ii.
```python
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2']
```
```python
            sid = r['stim_map'].get(r['WallName'][ti], None)
            if sid is None:                     # undefined stimulus category
                n_dropped_stim += 1
                continue
            ...
            out = np.stack([np.full(len(ii), sid),
                            lick[ii].astype(int),
                            pos_bin,
                            spd_bin]).astype(np.int64)
```

iii. The agent's stated reason is fidelity to the paper's own code: it wanted the labels the
authors use in every figure, and preferred the finer 7-way split (which distinguishes
leaf1/leaf2/leaf3 and the two spatial shuffles) over a coarse texture grouping. It did not
note that the ids are assigned per recording relative to the rewarded stimulus.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `beh['LickFr']`, the (fractional) neural frame number of every lick in the session.

ii.
```python
                    LickFr=np.asarray(beh['LickFr'], dtype=np.float64),
```
```python
        lick = np.zeros(nfr, dtype=bool)
        lf = np.round(r['LickFr']).astype(int)
        lick[lf[(lf >= 0) & (lf < nfr)]] = True
```

iii. `LickFr` is already expressed on the imaging-frame grid, so it is the direct source;
the agent measured that licks exist in only 28 (water-restricted task) sessions.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A session-length boolean vector is built: a frame is 1 if at least one lick rounds to it,
0 otherwise. Lick frames are rounded to the nearest frame and clipped to the imaged range
(`0 <= lf < nfr`), so licks recorded after the last imaged frame are discarded. The per-trial
output is that vector indexed by the trial's frames, cast to int. Final fraction of licking
bins: 0.035.

ii.
```python
        lick = np.zeros(nfr, dtype=bool)
        lf = np.round(r['LickFr']).astype(int)
        lick[lf[(lf >= 0) & (lf < nfr)]] = True
```
```python
            out = np.stack([np.full(len(ii), sid),
                            lick[ii].astype(int), ...
```

iii. The task asks for a binary time series, so the lick event times are rasterised onto the
frame grid. The agent explicitly measured the cost of its running-only frame selection here:
of 74,483 licks, 67,186 fall inside a corridor but only 37,106 fall on corridor frames while
the animal was running — it accepted this, noting that stopping to collect reward is exactly
what the Methods say is excluded.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` indexes imaging frames, so the flag vector is already on the neural grid; it is
indexed with the same `ii` as the neural columns, giving the same length.

ii.
```python
ii = frames[ti]
neural_s.append(np.ascontiguousarray(X[:, ii]))
... lick[ii].astype(int) ...
```

iii. Same as 3-c: one shared frame index per trial for every stream.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `beh['ft_Pos']`, the position inside the virtual corridor at every imaging frame, in
decimetres (0–40 across the 4 m texture, 40–60 through the grey space).

ii.
```python
                    ft_Pos=np.asarray(beh['ft_Pos'], dtype=np.float64),
```
```python
            pos_bin = np.digitize(r['ft_Pos'][ii], POS_BIN_EDGES)
```

iii. The agent verified in exploration that `ft_Pos` on `ft_CorrSpc` frames spans 0–40 dm
and on `ft_GraySpc` frames 40–60 dm, confirming the decimetre unit and the 4 m texture
length.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The raw decimetre position of each kept frame is bucketed into 4 bins by
`np.digitize` and clipped to 0–3, giving an integer 0–3 per timepoint, time-varying.
Resulting class fractions are essentially uniform (0.250 / 0.249 / 0.250 / 0.252).

ii.
```python
POS_BIN_EDGES = [10., 20., 30.]      # corridor position bins (dm) -> 4 x 1 m bins
```
```python
            pos_bin = np.digitize(r['ft_Pos'][ii], POS_BIN_EDGES)
            pos_bin = np.clip(pos_bin, 0, 3)
```

iii. Only texture-area frames are kept, so positions lie in 0–40 dm and the clip is a safety
net for the boundary; the bins follow the decoder-task instruction directly.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Fixed thresholds at 10, 20 and 30 dm = 1 m, 2 m and 3 m, i.e. the four equal-length 1-m
spatial bins the instructions ask for, labelled `['0-1m','1-2m','2-3m','3-4m']` and recorded
in the metadata as `position_bin_edges_m=[1.0, 2.0, 3.0]`.

ii.
```python
POS_BIN_EDGES = [10., 20., 30.]      # corridor position bins (dm) -> 4 x 1 m bins
...
output_values=[STIM_NAMES,
               ['no_lick', 'lick'],
               ['0-1m', '1-2m', '2-3m', '3-4m'],
               ['speed_q1', 'speed_q2', 'speed_q3', 'speed_q4']],
...
position_bin_edges_m=[1.0, 2.0, 3.0],
```

iii. The instruction specifies "4 equal-length, 1-m-long spatial bins", so the thresholds are
taken from the task rather than from the data distribution.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` has one value per imaging frame, so it is already on the neural grid; it is
indexed with the trial's frame array `ii`, giving one position bin per neural bin.

ii.
```python
ii = frames[ti]
pos_bin = np.digitize(r['ft_Pos'][ii], POS_BIN_EDGES)
neural_s.append(np.ascontiguousarray(X[:, ii]))
```

iii. Same as 3-c: every stream is indexed by the same frame array.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `beh['ft_RunSpeed']`, the running speed of the animal at every imaging frame (cm/s).

ii.
```python
                    ft_RunSpeed=np.asarray(beh['ft_RunSpeed'], dtype=np.float64),
```
```python
            spd_bin = np.digitize(r['ft_RunSpeed'][ii], speed_edges)
```

iii. `ft_RunSpeed` is the per-frame speed variable documented in the notebook; the agent
checked its range and that it contains no NaNs.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A **global** (whole-dataset) pre-pass collects `ft_RunSpeed` over all frames that will be
kept — running frames inside the corridor — from all 89 recordings, and takes the 25th, 50th
and 75th percentiles. Each frame is then assigned to a quartile bin with `np.digitize`.
This is done once, so the thresholds are identical for every session and each bin holds 25%
of the *pooled* data (not of each session separately). Measured edges: 12.42, 25.35 and
40.85 cm/s.

ii.
```python
    # ---- global running-speed quartiles (4 bins with 25% of the data each) ----
    speeds = []
    for rid in rids:
        r = recs[rid]
        frames, _ = trial_frames(r, len(r['ft']))
        ii = np.concatenate([f for f in frames if len(f)])
        speeds.append(r['ft_RunSpeed'][ii])
    speeds = np.concatenate(speeds)
    speed_edges = np.percentile(speeds, [25, 50, 75])
```
```python
            spd_bin = np.digitize(r['ft_RunSpeed'][ii], speed_edges)
```

iii. The instruction asks for "4 bins, each corresponding to 25% of the data", which the
agent read as a property of the dataset as a whole; it computed the quartiles from the
behaviour cache before writing the converter and recorded the edges in the metadata
(`speed_bin_edges_cm_s`) so the discretisation is documented and reproducible.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Data-derived thresholds: the pooled 25/50/75 percentiles (12.42 / 25.35 / 40.85 cm/s),
applied with `np.digitize`, giving classes named `speed_q1..speed_q4`. Because non-running
frames were already excluded, there is no large mass of exactly-zero speeds to create ties
at a threshold, and the edges are distinct.

ii.
```python
    speed_edges = np.percentile(speeds, [25, 50, 75])
    print('speed bin edges (cm/s):', speed_edges, flush=True)
```
```python
            spd_bin = np.digitize(r['ft_RunSpeed'][ii], speed_edges)
```

iii. Quartiles are the only way to satisfy "each corresponding to 25% of the data"; using
one global set of edges keeps the class meaning ("fast" vs "slow") comparable across
sessions and mice.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` has one value per imaging frame, so it is on the neural grid already, and
it is indexed with the trial's frame array `ii`.

ii.
```python
ii = frames[ti]
spd_bin = np.digitize(r['ft_RunSpeed'][ii], speed_edges)
neural_s.append(np.ascontiguousarray(X[:, ii]))
```

iii. Same as 3-c.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Four guards. (1) The behaviour can run past the imaging, so every behavioural stream is
truncated to `nfr = min(len(ft), nfr_spk)`. (2) Lick frames outside the imaged range are
dropped (`0 <= lf < nfr`). (3) Neurons with zero variance are removed before the division in
the z-score, so no NaNs are produced. (4) Trials with no surviving frame and trials with no
stimulus id are skipped. Frames with `ft_trInd = NaN` ("outside the behaviour recorded")
never match a trial index and are therefore excluded automatically. Retinotopy/neuron-count
mismatch is an `assert`, and NaN `iarea` values simply fail the `np.isin` test and are
dropped. The agent pre-checked that `SoundFr` and `StartFr` contain no NaNs anywhere in the
dataset (0 of 89 sessions), so the time inputs need no NaN handling.

ii.
```python
    nfr = min(len(rec['ft']), nfr_spk)
```
```python
        lf = np.round(r['LickFr']).astype(int)
        lick[lf[(lf >= 0) & (lf < nfr)]] = True
```
```python
        good = (sd[:, 0] > 0)
```
```python
        assert len(iarea) == nneu_all, (rid, len(iarea), nneu_all)
```

iii. The truncation mirrors the reference code's `beh[...][:nfr]` with `nfr = spk.shape[1]`.
The agent surveyed all 89 sessions for NaNs in the trial-timing variables before deciding
that no further handling was needed; it described the dataset as clean.

## 12-a. What are the most time-consuming steps of the code?

i. By far the dominant cost is reading the 89 spike files — ~405 GB in total — and the
`np.concatenate` of the per-plane arrays, which materialises a second copy of the full
(up to ~90,000 × ~100,000) session array in RAM before 1000 rows are selected from it.
Second is `load_behavior()`, which opens all 23 `Beh_<exp_type>.npy` files up front. The
global speed-quartile pre-pass and the per-trial loop are negligible by comparison.

ii.
```python
        spk = np.concatenate([p for p in np.load(spk_file, allow_pickle=True).item()['spks']], 0)
```
```python
    for exp_type, db in exp_info.items():
        Beh = np.load(os.path.join(ROOT, 'beh', 'Beh_%s.npy' % exp_type), allow_pickle=True).item()
```

iii. The agent measured the total size of `spk/` (405 GB) early on and designed around it,
running the full conversion in the background with progress logging; the run completed in a
few minutes only because the files were already in the page cache.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Two. (1) `trial_frames` builds each trial's index array with a full comparison against
the trial-index vector, one pass per trial — `idx[tr_idx == i] for i in range(ntrials)` is
O(ntrials × nframes) where a single `np.argsort`/`np.split` grouping would be O(nframes).
With up to ~800 trials per session this is the only quadratic step in the converter.
(2) The per-trial assembly loop (`np.stack` of inputs and outputs per trial) could
be computed once per session and split, rather than per trial. Both are negligible next to
the file I/O.

ii.
```python
    return [idx[tr_idx == i] for i in range(rec['ntrials'])], nfr
```
```python
        for ti in range(r['ntrials']):
            ...
            inp = np.stack([t_cue, np.full(len(ii), r['day']), t, ...])
            out = np.stack([np.full(len(ii), sid), lick[ii].astype(int), pos_bin, spd_bin])
```

iii. Not discussed by the agent; the design is I/O-bound and it optimised for I/O and memory
instead (caching only the behaviour fields it needed, `del spk` immediately after selecting
neurons).

## 12-c. What processing does the code repeat multiple times?

i. `trial_frames` is computed twice for every session: once in the global speed-quartile
pre-pass (with `nfr_spk = len(ft)`) and again in the main loop (with the true `nfr_spk` from
the spike file), so the per-trial frame grouping — the one quadratic step — is done twice.
Also, `Beh[key]` is fetched and its `UniqWalls`/`stim_id` re-scanned for every experiment
type a recording appears in (23 experiment types over 89 recordings), and
`np.mean(dts)` is recomputed for both `time_bin_size` and `sampling_rate_hz`.

ii.
```python
    for rid in rids:
        r = recs[rid]
        frames, _ = trial_frames(r, len(r['ft']))      # first pass, for the quartiles
```
```python
        frames, nfr = trial_frames(r, nfr_spk)         # second pass, for the conversion
```
```python
        time_bin_size=float(np.mean(dts) * 1000.),
        ...
        sampling_rate_hz=float(1000. / (np.mean(dts) * 1000.)),
```

iii. The repetition is a deliberate consequence of needing dataset-wide speed quartiles
before any trial can be written; the agent kept the behaviour cached in memory so the second
pass costs no I/O.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The largest by far: every session's **entire** spike array is read and concatenated
(20,547–89,577 neurons) and then 98% of it is thrown away — only 1000 neurons are kept.
(The `.npy` files are pickled dicts of per-plane arrays, so a partial read is not possible,
but the concatenate copy is avoidable.) Second, the z-score statistics and the float32 copy
`X` are computed over *all* frames of the recording although only the running in-corridor
frames (~20–25% of frames) are ever written out. Smaller items: `CORRIDOR_TEXTURE_LEN` is
defined and never used; `exp_types`/`cohorts` are accumulated per session for metadata only;
`np.ascontiguousarray` makes an extra copy of each trial slice; and the `dts` list retains a
per-session value of which only the mean is used.

ii.
```python
CORRIDOR_TEXTURE_LEN = 40.           # dm (4 m)     <- never referenced again
```
```python
        spk = np.concatenate([p for p in np.load(spk_file, allow_pickle=True).item()['spks']], 0)
        ...
        sel = np.sort(rng.choice(valid, size=min(NNEURONS, len(valid)), replace=False))
        X = spk[sel].astype(np.float32)
        del spk
```
```python
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
```

iii. The agent chose to z-score over the whole recording deliberately (that is what
`utils.get_kfold_reward_response` does), and it frees `spk` as soon as the subset is taken.
The wasted read is the unavoidable price of the source file layout; the wasted *selection*
(reading 46,000 neurons on average to keep 1000) follows from the sub-sampling decision in
2-b rather than from a coding oversight.
