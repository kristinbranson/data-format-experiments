# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Data is loaded from three subdirectories under `/app/data`: `beh/` (behavior), `spk/` (deconvolved calcium traces), and `retinotopy/` (visual area assignments). `Imaging_Exp_info.npy` is the master index listing all recordings grouped by experiment type. For each experiment type, the corresponding `Beh_<exp_type>.npy` file is loaded and iterated. Each unique recording (identified by mouse+date+block) is stored once, with stimulus mappings merged across experiment types. Spike files and retinotopy are loaded per session.

ii.
```python
exp_info = np.load(os.path.join(ROOT, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()
# ...
for exp_type, db in exp_info.items():
    Beh = np.load(os.path.join(ROOT, 'beh', 'Beh_%s.npy' % exp_type),
                  allow_pickle=True).item()
    for ndb in db:
        rid = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
        # ...
```

```python
spk = np.concatenate([p for p in np.load(spk_file, allow_pickle=True).item()['spks']], 0)
ret = np.load(os.path.join(ROOT, 'retinotopy', '%s_%s_trans.npz'
                           % (r['mname'], r['datexp'])), allow_pickle=True)
```

iii. The AI explored the data structure, found that `Imaging_Exp_info.npy` is the master index listing every recording grouped by experiment type, and that behavior files are keyed by session id. It merged stimulus mappings across experiment types per recording.

## 1-b. How are the data split into subjects?

i. Mouse name is taken from `mname` in the index entry. Subjects are the sorted unique mouse names across all recordings.

ii.
```python
subjects = sorted(set(recs[r]['mname'] for r in rids))
data['subject_idx'].append(subjects.index(r['mname']))
```

iii. The mouse name is directly available in the data index.

## 1-c. How are the data split into sessions?

i. A session is one mouse on one date in one block, identified by `mname_datexp_blk`. Recordings appearing under multiple experiment types are deduplicated (kept only once, with stimulus mappings merged).

ii.
```python
rid = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
if rid not in recs:
    recs[rid] = dict(...)
```

iii. The AI noted recordings can appear in several experiment types and chose to keep each recording once, merging stimulus information.

## 1-d. How are the data split into trials?

i. Trials are split using `ft_trInd` (trial index per frame) and `ft_CorrSpc` (inside corridor texture). Additionally, the AI applies a **running filter** using `ft_move > 0`, keeping only frames where the virtual reality was moving. Each trial consists of the frames that match all three conditions: correct trial index, inside corridor, and animal running.

ii.
```python
def trial_frames(rec, nfr_spk):
    nfr = min(len(rec['ft']), nfr_spk)
    keep = rec['ft_CorrSpc'][:nfr] & (rec['ft_move'][:nfr] > 0)
    tr = rec['ft_trInd'][:nfr]
    idx = np.where(keep)[0]
    tr_idx = tr[idx]
    return [idx[tr_idx == i] for i in range(rec['ntrials'])], nfr
```

iii. The AI cited the paper: "We only considered timepoints during running for analysis, which removed time periods when the task mice stopped to collect water rewards." The reference code in utils.py applies exactly this mask (`VRmove = beh['ft_move']>0`).

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped if: (1) they have zero frames after the running+corridor filter, or (2) their wall texture has no canonical stimulus id in the `stim_map` (specifically, 309 trials with 'circle3' texture are dropped because their stimulus category is undefined in `Imaging_Exp_info.npy`). No trial length filtering is applied.

ii.
```python
if len(ii) == 0:
    continue
sid = r['stim_map'].get(r['WallName'][ti], None)
if sid is None:                     # undefined stimulus category
    n_dropped_stim += 1
    continue
```

iii. The AI investigated unmapped wall names and found that 4 recordings have 'circle3' walls with no stim_id mapping, affecting 309 trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the spike file `spk/<session_id>_neural_data.npy`, which contains a list of neuron-by-frames arrays per imaging plane, concatenated into a single array. Visual area comes from `iarea` in the retinotopy file.

ii.
```python
spk = np.concatenate([p for p in np.load(spk_file, allow_pickle=True).item()['spks']], 0)
iarea = np.asarray(ret['iarea'], dtype=float)
```

iii. The AI confirmed these are the deconvolved traces from Suite2p.

## 2-b. How is the `neural` data processed?

i. The AI applies three processing steps: (1) filters neurons to those in V1/mHV/lHV/aHV, (2) randomly subsamples to at most 1000 neurons per session (with a fixed seed), and (3) z-scores each neuron over the entire recording (mean=0, std=1). Neurons with zero standard deviation are removed.

ii.
```python
NNEURONS = 1000
sel = np.sort(rng.choice(valid, size=min(NNEURONS, len(valid)), replace=False))
X = spk[sel].astype(np.float32)
mu = X.mean(axis=1, keepdims=True)
sd = X.std(axis=1, keepdims=True)
good = (sd[:, 0] > 0)
X = (X[good] - mu[good]) / sd[good]          # z-score over the whole recording
```

iii. The AI noted that the reference code z-scores in `get_kfold_reward_response` and that recordings contain 20,547-89,577 neurons, so a random subset of 1000 was chosen to keep the dataset tractable.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are kept only if their `iarea` falls in V1 (8), mHV (0,1,2,9), lHV (5,6), or aHV (3,4). Then a random subset of up to 1000 is selected, and any neuron with zero standard deviation is dropped.

ii.
```python
AREA_GROUPS = {'V1': [8], 'mHV': [0, 1, 2, 9], 'lHV': [5, 6], 'aHV': [3, 4]}
region_of_neuron = np.full(nneu_all, -1, dtype=int)
for ri, reg in enumerate(BRAIN_REGIONS):
    m = np.isin(iarea, AREA_GROUPS[reg])
    region_of_neuron[m] = ri
valid = np.where(region_of_neuron >= 0)[0]
sel = np.sort(rng.choice(valid, size=min(NNEURONS, len(valid)), replace=False))
```

iii. The AI identified the area groups from `utils.neu_area_ID` and applied the same grouping.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Aligned to corridor entry (trial start). Each trial's frames are the running+corridor frames for that trial. Trials are variable length, starting at corridor entry and ending when the animal leaves the 4m texture area (or stops running).

ii.
```python
neural_s.append(np.ascontiguousarray(X[:, ii]))
```
where `ii = frames[ti]` are the running+corridor frame indices for trial `ti`.

iii. The AI aligns to trial start as specified in the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning. The native imaging frame rate (~3.17 Hz, ~314.7 ms per frame) is used as is. The time bin size is computed as the mean median frame interval across sessions.

ii.
```python
dts.append(np.median(np.diff(r['ft'][:nfr])) * 86400.)
# ...
'time_bin_size': float(np.mean(dts) * 1000.),
```

iii. The AI noted the imaging frames are the finest temporal resolution available.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime`, the absolute timestamp of the sound cue for each trial, and `ft`, the timestamp of every imaging frame.

ii.
```python
SoundTime=np.asarray(beh['SoundTime'], dtype=np.float64),
# ...
t_cue = (r['SoundTime'][ti] - r['ft'][ii]) * 86400.      # s until sound cue
```

iii. The AI used `SoundTime` (the absolute time of the sound cue) rather than `SoundFr` (the frame number).

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The time to sound cue is computed as `SoundTime[trial] - ft[frame_indices]`, converted from days to seconds by multiplying by 86400. This gives positive values before the cue and negative after, consistent with "time TO sound cue."

ii.
```python
t_cue = (r['SoundTime'][ti] - r['ft'][ii]) * 86400.      # s until sound cue
```

iii. The AI computed a signed time difference in seconds.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Computed from the same frame indices (`ii = frames[ti]`) used for the neural data of that trial.

ii.
```python
t_cue = (r['SoundTime'][ti] - r['ft'][ii]) * 86400.
inp = np.stack([t_cue, ...])
```

iii. All data streams share the same frame indices per trial.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From `datexp`, the date string of each session (e.g., '2022_07_12'), parsed into a `datetime.date` object.

ii.
```python
d = datetime.date(*[int(x) for x in r['datexp'].split('_')])
r['date'] = d
if r['mname'] not in first_day or d < first_day[r['mname']]:
    first_day[r['mname']] = d
```

iii. The AI used the actual calendar date to compute elapsed days.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Day of training is computed as the number of **calendar days** elapsed since the mouse's first recording date. This differs from simply counting recording sessions: if a mouse's recordings span non-consecutive dates, the gaps count toward the day count.

ii.
```python
for rid in rids:
    r = recs[rid]
    r['day'] = float((r['date'] - first_day[r['mname']]).days)
```

iii. The AI reasoned that calendar days give a more meaningful measure of training progression.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time`, the absolute timestamp of corridor entry for each trial, and `ft`, the timestamp of each imaging frame.

ii.
```python
Trial_start_time=np.asarray(beh['Trial_start_time'], dtype=np.float64),
# ...
t = (r['ft'][ii] - r['Trial_start_time'][ti]) * 86400.   # s since corridor entry
```

iii. The AI used `Trial_start_time` rather than `StartFr`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Time since trial start is `ft[frame_indices] - Trial_start_time[trial]`, converted from days to seconds by multiplying by 86400.

ii.
```python
t = (r['ft'][ii] - r['Trial_start_time'][ti]) * 86400.   # s since corridor entry
```

iii. Direct time difference computation.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed from the same frame indices (`ii`) used for the neural data of that trial.

ii.
```python
t = (r['ft'][ii] - r['Trial_start_time'][ti]) * 86400.
```

iii. All data streams share the same frame indices per trial.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks whether each trial is in a rewarded corridor.

ii.
```python
isRew=np.asarray(beh['isRew']).astype(bool),
# ...
np.full(len(ii), float(r['isRew'][ti]))
```

iii. Directly available from the behavior data.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast to float (1.0 for rewarded, 0.0 for not), broadcast across all frames of the trial.

ii.
```python
np.full(len(ii), float(r['isRew'][ti]))
```

iii. No processing needed beyond type conversion.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName` (the wall texture name per trial) and the `stim_map` derived from `UniqWalls` and `stim_id` in the behavior data. The `stim_map` maps wall names to canonical stimulus ids (e.g., circle1=0, circle2=1, leaf1=2, etc.).

ii.
```python
for wall, sid in zip(beh['UniqWalls'], np.asarray(beh['stim_id'], dtype=float)):
    if not np.isnan(sid):
        r['stim_map'][str(wall)] = int(sid)
# ...
sid = r['stim_map'].get(r['WallName'][ti], None)
```

iii. The AI merged stimulus id mappings across experiment types to cover all wall names.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The wall name is mapped to its canonical stimulus id (an integer 0-6, corresponding to circle1, circle2, leaf1, leaf2, leaf3, leaf1_swap1, leaf1_swap2). This is stored as a per-trial integer broadcast across all frames. Trials with unmapped walls (circle3) are dropped. The output_values list contains 7 stimulus names rather than 4 broad texture categories.

ii.
```python
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2']
# ...
out = np.stack([np.full(len(ii), sid), ...])
```

iii. The AI used the paper's canonical stimulus ids rather than grouping into broad categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the neural frame number of each lick event in the session.

ii.
```python
LickFr=np.asarray(beh['LickFr'], dtype=np.float64),
```

iii. Directly available from the behavior data.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frame numbers are rounded to integers, and a binary flag is set at each frame with at least one lick. Lick frames outside the valid range are dropped.

ii.
```python
lick = np.zeros(nfr, dtype=bool)
lf = np.round(r['LickFr']).astype(int)
lick[lf[(lf >= 0) & (lf < nfr)]] = True
# ...
lick[ii].astype(int)
```

iii. The AI converted fractional frame numbers to binary lick flags per frame.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick flag array is indexed by the same frame indices (`ii`) used for neural data per trial.

ii.
```python
lick[ii].astype(int)
```

iii. All data streams share the same frame indices per trial.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position inside the corridor at each imaging frame, in decimeters.

ii.
```python
ft_Pos=np.asarray(beh['ft_Pos'], dtype=np.float64),
# ...
pos_bin = np.digitize(r['ft_Pos'][ii], POS_BIN_EDGES)
```

iii. Directly available from the behavior data.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position (in decimeters) is discretized into 4 bins of 1 m each using `np.digitize` with bin edges at [10, 20, 30] dm, then clipped to range [0, 3].

ii.
```python
POS_BIN_EDGES = [10., 20., 30.]
pos_bin = np.digitize(r['ft_Pos'][ii], POS_BIN_EDGES)
pos_bin = np.clip(pos_bin, 0, 3)
```

iii. The AI discretized into the 4 equal-length 1m bins requested by the decoder task.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Using `np.digitize` with edges [10, 20, 30] decimeters, giving bins 0-1m, 1-2m, 2-3m, 3-4m. Values are clipped to [0, 3].

ii. Same as 9-b.

iii. Equivalent to dividing by 10 dm and flooring.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position values are indexed by the same frame indices (`ii`) used for neural data.

ii.
```python
pos_bin = np.digitize(r['ft_Pos'][ii], POS_BIN_EDGES)
```

iii. All data streams share the same frame indices per trial.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
ft_RunSpeed=np.asarray(beh['ft_RunSpeed'], dtype=np.float64),
# ...
spd_bin = np.digitize(r['ft_RunSpeed'][ii], speed_edges)
```

iii. Directly available from the behavior data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is discretized into 4 bins using **global** quartile edges computed across all sessions. All running+corridor frames from all sessions are pooled, percentiles at [25, 50, 75] are computed, and `np.digitize` is used to assign each frame to a bin.

ii.
```python
speeds = []
for rid in rids:
    r = recs[rid]
    frames, _ = trial_frames(r, len(r['ft']))
    ii = np.concatenate([f for f in frames if len(f)])
    speeds.append(r['ft_RunSpeed'][ii])
speeds = np.concatenate(speeds)
speed_edges = np.percentile(speeds, [25, 50, 75])
# ...
spd_bin = np.digitize(r['ft_RunSpeed'][ii], speed_edges)
```

iii. The AI computed global speed quartiles to ensure the 25% split is across the entire dataset, not per session.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Using `np.digitize` with three global percentile edges (25th, 50th, 75th percentiles of running speed across all sessions).

ii. Same as 10-b.

iii. Global thresholds ensure consistent bin definitions across sessions.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed values are indexed by the same frame indices (`ii`) used for neural data.

ii.
```python
spd_bin = np.digitize(r['ft_RunSpeed'][ii], speed_edges)
```

iii. All data streams share the same frame indices per trial.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Behavior arrays are truncated to the number of imaged frames (`min(len(ft), nfr_spk)`). Lick frames outside the valid range are dropped (negative or >= nfr). Trials with zero frames after filtering are skipped. Trials with unmapped stimulus ids are dropped. Neurons with zero standard deviation are removed after z-scoring.

ii.
```python
nfr = min(len(rec['ft']), nfr_spk)
lf = np.round(r['LickFr']).astype(int)
lick[lf[(lf >= 0) & (lf < nfr)]] = True
# ...
good = (sd[:, 0] > 0)
X = (X[good] - mu[good]) / sd[good]
```

iii. The AI checked for edge cases including behavior running past imaging frames and lick frames outside bounds.

## 12-a. What are the most time-consuming steps of the code?

i. Reading the spike files, which total ~405 GB. Each session requires loading and concatenating all imaging planes.

ii.
```python
spk = np.concatenate([p for p in np.load(spk_file, allow_pickle=True).item()['spks']], 0)
```

iii. The AI noted the I/O cost dominates runtime.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The `trial_frames` function iterates over all trials to find frames per trial. This could be done in a single pass by grouping frames by trial index.

ii.
```python
def trial_frames(rec, nfr_spk):
    # ...
    return [idx[tr_idx == i] for i in range(rec['ntrials'])], nfr
```

iii. The loop scans the trial index once per trial rather than grouping all frames in one pass.

## 12-c. What processing does the code repeat multiple times?

i. The global speed quartiles require iterating through all sessions' behavior data to compute `trial_frames` and collect speeds, and then the main conversion loop computes `trial_frames` again for each session. This duplicates the frame selection work.

ii.
```python
# First pass for speed edges:
for rid in rids:
    frames, _ = trial_frames(r, len(r['ft']))
    ...
# Second pass for conversion:
for si, rid in enumerate(rids):
    frames, nfr = trial_frames(r, nfr_spk)
    ...
```

iii. The speed computation pre-pass repeats the trial frame extraction.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code z-scores neural data over the whole recording, but the downstream decoder (train_decoder.py) normalizes the data itself during training, making this z-scoring potentially redundant. Additionally, subsampling to 1000 neurons discards the vast majority of recorded neurons.

ii.
```python
X = (X[good] - mu[good]) / sd[good]          # z-score over the whole recording
sel = np.sort(rng.choice(valid, size=min(NNEURONS, len(valid)), replace=False))
```

iii. The AI chose to z-score following the reference code convention, and to subsample for tractability.
