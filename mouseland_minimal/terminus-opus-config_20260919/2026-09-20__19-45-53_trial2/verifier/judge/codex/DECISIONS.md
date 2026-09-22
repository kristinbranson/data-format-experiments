# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads `Imaging_Exp_info.npy`, then every `Beh_<experiment>.npy`, deduplicates recordings into a dictionary, and loads each recording's plane-wise spike file and retinotopy file. It converts all 89 unique recordings.

ii.
```python
exp_info = np.load(os.path.join(ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
Beh = np.load(os.path.join(ROOT, 'beh', 'Beh_%s.npy' % exp_type), allow_pickle=True).item()
spk = np.concatenate([p for p in np.load(spk_file, allow_pickle=True).item()['spks']], 0)
ret = np.load(os.path.join(ROOT, 'retinotopy', '%s_%s_trans.npz' % (r['mname'], r['datexp'])))
```

iii. The trajectory says it found 89 spike files/unique recordings and reorganized the experiment-indexed behavior into one entry per recording so duplicates could be merged.

## 1-b. How are the data split into subjects?

i. Subjects are the sorted unique `mname` values; every session receives the corresponding integer `subject_idx`.

ii.
```python
subjects = sorted(set(recs[r]['mname'] for r in rids))
data['subject_idx'].append(subjects.index(r['mname']))
```

iii. The agent identified `mname` as the explicit mouse identifier and reported 19 mice.

## 1-c. How are the data split into sessions?

i. A session/recording is keyed by mouse, date, and block. Duplicate appearances across experiment types are merged, including their wall-to-stimulus maps.

ii.
```python
rid = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
if rid not in recs:
    recs[rid] = dict(rid=rid, mname=ndb['mname'], datexp=ndb['datexp'], blk=ndb['blk'], ...)
```

iii. The trajectory notes that the 23 experiment types resolve to 89 unique recordings and that some recordings occur under multiple stimulus subsets.

## 1-d. How are the data split into trials?

i. It uses `ntrials` and `ft_trInd` to group frames by trial, retaining only frames both inside the texture corridor (`ft_CorrSpc`) and marked as moving (`ft_move > 0`). Trials are variable length and may be temporally non-contiguous after stopped frames are removed.

ii.
```python
keep = rec['ft_CorrSpc'][:nfr] & (rec['ft_move'][:nfr] > 0)
idx = np.where(keep)[0]
return [idx[tr[idx] == i] for i in range(rec['ntrials'])], nfr
```

iii. The agent interpreted the paper's statement that analyses considered running timepoints as requiring the `ft_move > 0` mask in this conversion.

## 1-e. How are trials filtered based on quality controls?

i. Empty trials are skipped. Trials whose `WallName` lacks a canonical `stim_map` entry are also dropped (309 `circle3` trials). There is no long-trial filter.

ii.
```python
if len(ii) == 0:
    continue
sid = r['stim_map'].get(r['WallName'][ti], None)
if sid is None:
    n_dropped_stim += 1
    continue
```

iii. The agent judged unmapped `circle3` to have an undefined target category. It believed every trial had at least 11 running frames and did not identify long stopped trials because stopped frames had already been removed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from the per-plane `spks` arrays in each `_neural_data.npy`; neuron region assignments come from retinotopy `iarea`.

ii.
```python
spk = np.concatenate([p for p in np.load(spk_file, allow_pickle=True).item()['spks']], 0)
iarea = np.asarray(ret['iarea'], dtype=float)
```

iii. The trajectory identifies these as Suite2p deconvolved traces and the paper's retinotopic area labels.

## 2-b. How is the `neural` data processed?

i. After neuron selection, each neuron is converted to float32 and z-scored over the entire recording; per-trial selected columns are stored contiguously. Up to 1,000 randomly selected neurons are retained per session.

ii.
```python
sel = np.sort(rng.choice(valid, size=min(NNEURONS, len(valid)), replace=False))
X = spk[sel].astype(np.float32)
mu = X.mean(axis=1, keepdims=True); sd = X.std(axis=1, keepdims=True)
X = (X[good] - mu[good]) / sd[good]
neural_s.append(np.ascontiguousarray(X[:, ii]))
```

iii. The agent cited z-scoring in a paper utility and chose 1,000 neurons because the 405 GB source and decoder memory made keeping millions of neurons impractical.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons outside V1/mHV/lHV/aHV are excluded, 1,000 valid neurons are sampled without replacement with seed 0, and zero-variance selected neurons are removed.

ii.
```python
valid = np.where(region_of_neuron >= 0)[0]
sel = np.sort(rng.choice(valid, size=min(NNEURONS, len(valid)), replace=False))
good = (sd[:, 0] > 0)
```

iii. The agent followed the paper's four area groups, treated other area values as unassigned, and added random subsampling for tractability.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are described as aligned to corridor entry, but neural columns are the original imaging frames in the corridor after all stopped frames are removed. Thus the first retained frame is near entry and timestamps retain elapsed time across gaps.

ii.
```python
ii = frames[ti]
t = (r['ft'][ii] - r['Trial_start_time'][ti]) * 86400.
neural_s.append(np.ascontiguousarray(X[:, ii]))
```

iii. The agent considered corridor entry the required alignment and accepted variable-length running-only sequences as matching the paper.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native imaging frames are retained with no rebinning. The metadata bin size is the mean session median frame interval, approximately 315 ms (~3.18 Hz).

ii.
```python
dts.append(np.median(np.diff(r['ft'][:nfr])) * 86400.)
time_bin_size=float(np.mean(dts) * 1000.)
```

iii. The agent found frame timing consistent across sessions and regarded imaging frames as the finest available common grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from the per-trial `SoundTime` and per-frame timestamp `ft`.

ii.
```python
t_cue = (r['SoundTime'][ti] - r['ft'][ii]) * 86400.
```

iii. The trajectory established that behavior contains sound-cue times and frame timestamps.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Frame time is subtracted from cue time and MATLAB-day units are converted to seconds, producing positive values before and negative values after the cue.

ii.
```python
t_cue = (r['SoundTime'][ti] - r['ft'][ii]) * 86400.
```

iii. The agent explicitly interpreted the requested variable as signed time remaining until the cue.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at exactly the same selected frame indices `ii` used for neural columns.

ii.
```python
t_cue = (r['SoundTime'][ti] - r['ft'][ii]) * 86400.
neural_s.append(np.ascontiguousarray(X[:, ii]))
```

iii. The agent relied on all behavior and imaging streams sharing frame indices/timestamps.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the `datexp` date embedded in each recording and the mouse identifier.

ii.
```python
d = datetime.date(*[int(x) for x in r['datexp'].split('_')])
first_day[r['mname']] = min(...)
```

iii. The agent treated calendar elapsed time from a mouse's first recording as training day.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The code subtracts each mouse's earliest recording date, yielding elapsed calendar days, and broadcasts that scalar across trial frames.

ii.
```python
r['day'] = float((r['date'] - first_day[r['mname']]).days)
np.full(len(ii), r['day'])
```

iii. The agent did not discuss gaps between recordings; its stated decision was “days since that mouse's first recording.”

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It comes from per-frame `ft` and per-trial `Trial_start_time`.

ii.
```python
t = (r['ft'][ii] - r['Trial_start_time'][ti]) * 86400.
```

iii. The agent identified `Trial_start_time`/`StartFr` as corridor entry.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Trial-start timestamp is subtracted from each retained frame timestamp and converted from days to seconds.

ii.
```python
t = (r['ft'][ii] - r['Trial_start_time'][ti]) * 86400.
```

iii. The agent intended actual elapsed time since entry, including time represented by removed stopped-frame gaps.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed for the same frame indices as each neural column.

ii.
```python
t = (r['ft'][ii] - r['Trial_start_time'][ti]) * 86400.
neural_s.append(np.ascontiguousarray(X[:, ii]))
```

iii. Shared frame indexing was considered sufficient alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes directly from per-trial `isRew`.

ii.
```python
isRew=np.asarray(beh['isRew']).astype(bool)
```

iii. The agent recognized `isRew` as marking the rewarded corridor.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The Boolean is converted to float and broadcast over all retained trial frames.

ii.
```python
np.full(len(ii), float(r['isRew'][ti]))
```

iii. No transformation beyond format/broadcasting was considered necessary.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It uses per-trial `WallName` and a wall-name-to-`stim_id` map merged from `UniqWalls`/`stim_id` across experiment entries.

ii.
```python
for wall, sid in zip(beh['UniqWalls'], np.asarray(beh['stim_id'], dtype=float)):
    r['stim_map'][str(wall)] = int(sid)
sid = r['stim_map'].get(r['WallName'][ti], None)
```

iii. The agent found seven canonical IDs in the paper code and used those as categories.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The canonical ID (0–6) is broadcast over a trial. Unmapped stimuli are dropped rather than grouped into a broader texture class.

ii.
```python
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3', 'leaf1_swap1', 'leaf1_swap2']
np.full(len(ii), sid)
```

iii. The agent said `circle3` had no canonical ID and therefore an undefined target.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr`, the frame coordinate of each lick.

ii.
```python
lf = np.round(r['LickFr']).astype(int)
```

iii. The agent identified lick frames as directly available and noted licks occur only in task sessions.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frame coordinates are rounded to the nearest integer; valid indices set a Boolean per-frame array to true. Multiple licks in one frame remain one.

ii.
```python
lick = np.zeros(nfr, dtype=bool)
lf = np.round(r['LickFr']).astype(int)
lick[lf[(lf >= 0) & (lf < nfr)]] = True
```

iii. The agent aimed to represent the requested binary time series and checked how many licks survived the running-only mask.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The per-frame lick vector is indexed by the same `ii` used for neural data, so licks during removed stopped frames disappear.

ii.
```python
out = np.stack([...,
                lick[ii].astype(int), ...])
neural_s.append(np.ascontiguousarray(X[:, ii]))
```

iii. The agent relied on neural-frame coordinates and intentionally kept only running-time outputs.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It comes from per-frame `ft_Pos`, expressed in decimeters.

ii.
```python
pos_bin = np.digitize(r['ft_Pos'][ii], POS_BIN_EDGES)
```

iii. The agent determined that 0–40 dm is the 4 m texture corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Positions at retained frames are digitized and clipped to categorical indices 0–3.

ii.
```python
POS_BIN_EDGES = [10., 20., 30.]
pos_bin = np.clip(np.digitize(r['ft_Pos'][ii], POS_BIN_EDGES), 0, 3)
```

iii. The four requested bins are each 10 dm (1 m).

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are 10, 20, and 30 dm, producing 0–1, 1–2, 2–3, and 3–4 m categories; `np.digitize` puts an exact edge into the higher bin.

ii.
```python
POS_BIN_EDGES = [10., 20., 30.]
output_values=[..., ['0-1m', '1-2m', '2-3m', '3-4m'], ...]
```

iii. The agent directly implemented the instruction's four equal 1 m bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is selected at the same original frame indices used for neural columns.

ii.
```python
pos_bin = np.digitize(r['ft_Pos'][ii], POS_BIN_EDGES)
neural_s.append(np.ascontiguousarray(X[:, ii]))
```

iii. Position and imaging were understood to share the frame grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from per-frame `ft_RunSpeed` at retained running/corridor frames.

ii.
```python
speeds.append(r['ft_RunSpeed'][ii])
```

iii. The agent identified this as the direct speed stream.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. All retained frames from all sessions are pooled, global 25th/50th/75th percentile thresholds are calculated, and each trial frame is digitized.

ii.
```python
speeds = np.concatenate(speeds)
speed_edges = np.percentile(speeds, [25, 50, 75])
spd_bin = np.digitize(r['ft_RunSpeed'][ii], speed_edges)
```

iii. The agent interpreted “each corresponding to 25% of the data” globally and reported edges near 12.4/25.4/40.9 cm/s.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Values below/between/above the three global percentile edges receive integer classes 0–3; ties follow `np.digitize` and therefore need not yield exactly equal class counts.

ii.
```python
speed_edges = np.percentile(speeds, [25, 50, 75])
spd_bin = np.digitize(r['ft_RunSpeed'][ii], speed_edges)
```

iii. Global quartile thresholds were chosen to target four bins containing a quarter of retained timepoints each.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is read at the same selected indices `ii` used for neural columns.

ii.
```python
spd_bin = np.digitize(r['ft_RunSpeed'][ii], speed_edges)
neural_s.append(np.ascontiguousarray(X[:, ii]))
```

iii. Speed and imaging were understood to share frame indices.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Behavior is truncated to the smaller of behavior and spike frame counts; out-of-range and negative lick frames are ignored; empty trials and unknown-stimulus trials are skipped; zero-variance selected neurons are removed. The retinotopy/neuron count mismatch is asserted.

ii.
```python
nfr = min(len(rec['ft']), nfr_spk)
lf[(lf >= 0) & (lf < nfr)]
if len(ii) == 0: continue
assert len(iarea) == nneu_all
good = (sd[:, 0] > 0)
```

iii. The agent surveyed missing mappings and frame consistency before conversion and explicitly treated `circle3` as undefined.

## 12-a. What are the most time-consuming steps of the code?

i. Loading and concatenating 405 GB of spike arrays, followed by per-session copying, z-scoring, and writing a 3.3 GB pickle, dominate runtime.

ii.
```python
spk = np.concatenate([p for p in np.load(spk_file, allow_pickle=True).item()['spks']], 0)
X = spk[sel].astype(np.float32)
X = (X[good] - mu[good]) / sd[good]
```

iii. The trajectory explicitly calls spike-file I/O the long step and monitored the full 89-session conversion.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. `trial_frames` scans the already-filtered trial-index array once for every trial; area assignment loops over four regions; trial assembly loops over trials. The per-trial grouping is the clearest vectorization candidate, though I/O dominates.

ii.
```python
return [idx[tr_idx == i] for i in range(rec['ntrials'])], nfr
for ri, reg in enumerate(BRAIN_REGIONS):
    region_of_neuron[np.isin(iarea, AREA_GROUPS[reg])] = ri
```

iii. The trajectory did not give an explicit optimization justification beyond focusing on I/O and memory.

## 12-c. What processing does the code repeat multiple times?

i. Trial-frame grouping is computed once during the global speed pass and again during session conversion. Behavior files are also loaded independently for each experiment type, including overlapping recordings.

ii.
```python
for rid in rids:
    frames, _ = trial_frames(r, len(r['ft']))
...
frames, nfr = trial_frames(r, nfr_spk)
```

iii. The first pass is needed to obtain global speed thresholds before outputs are built; no caching was retained in the final converter.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads and concatenates every neuron before selecting at most 1,000, computes/retains bookkeeping such as experiment/cohort metadata, and defines `CORRIDOR_TEXTURE_LEN` without using it. It also computes `sel = sel[good]` mainly for region/bookkeeping after normalization.

ii.
```python
spk = np.concatenate([...], 0)
sel = np.sort(rng.choice(valid, size=min(NNEURONS, len(valid)), replace=False))
CORRIDOR_TEXTURE_LEN = 40.
```

iii. The full concatenation is imposed by the source layout and random cross-plane selection; metadata was added for interpretability, not decoder features.
