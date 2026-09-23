# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads `Imaging_Exp_info.npy` as a master index, de-duplicates it to 89 recordings, reads each `Beh_<experiment>.npy` once, loads each recording's plane-wise spike file, and loads its retinotopy file. Duplicate index appearances are also used to merge stimulus-ID maps.

ii. ```python
exp_info = np.load(os.path.join(BEH_DIR, 'Imaging_Exp_info.npy'), allow_pickle=True).item()
B = np.load(os.path.join(BEH_DIR, 'Beh_' + exp_type + '.npy'), allow_pickle=True).item()
return np.load(os.path.join(SPK_DIR, fn), allow_pickle=True).item()['spks']
d = np.load(os.path.join(RET_DIR, '%s_%s_trans.npz' % (mname, datexp)), allow_pickle=True)
```

iii. The notes say this covers all 89 recordings/19 mice while avoiding repeated reads of large behavior files and preserving mappings supplied by duplicate experiment-type entries.

## 1-b. How are the data split into subjects?

i. Subjects are mouse names (`mname`); the sorted unique names form `subjects`, and each recording gets a `subject_idx`.

ii. ```python
subjects = sorted({r['mname'] for r in recs})
'subject_idx': np.array([subjects.index(r['mname']) for r in recs], dtype=np.int64)
```

iii. The notes report 19 unique mice and regard the explicit mouse field as authoritative.

## 1-c. How are the data split into sessions?

i. A session is the unique `(mname, datexp, blk)` recording. Repeated appearances under experiment types are de-duplicated, while their metadata are merged.

ii. ```python
key = '%s_%s_%s' % (db['mname'], db['datexp'], db['blk'])
if key not in recs:
    recs[key] = {'key': key, ...}
```

iii. The notes state that 142 index entries reduce to 89 unique recordings and that behavior is identical across duplicate appearances (checked at least by `ntrials`).

## 1-d. How are the data split into trials?

i. Retained frames are sorted by `ft_trInd`; changes in that index define trial boundaries. Only frames satisfying the moving, in-corridor mask are present, so a trial is the retained subset rather than every corridor frame.

ii. ```python
tri = b['ft_trInd'][idx].astype(np.int64)
bounds = np.flatnonzero(np.diff(tri)) + 1
starts = np.concatenate(([0], bounds))
stops = np.concatenate((bounds, [len(tri)]))
```

iii. The agent cites the paper/reference analysis mask `(ft_move > 0) & ft_CorrSpc` and says all 38,110 trials retain at least one frame.

## 1-e. How are trials filtered based on quality controls?

i. No whole trial is dropped. Instead, stationary and non-corridor frames, invalid trial indices, and 21 ambiguous wraparound boundary frames are removed.

ii. ```python
keep = (ft_move[:nfr] > 0) & ft_CorrSpc[:nfr] & ~np.isnan(tr)
keep[idx[wrap]] = False
```

iii. The notes justify the moving-frame rule from the paper's running-only analyses and state that every trial still has at least one retained frame.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from the `spks` arrays in each session's neural file; `iarea` from retinotopy determines visual-area membership and region labels.

ii. ```python
return np.load(os.path.join(SPK_DIR, fn), allow_pickle=True).item()['spks']
return d['iarea']
```

iii. The agent identifies `spks` as Suite2p deconvolved calcium traces and follows the reference area mapping.

## 2-b. How is the `neural` data processed?

i. Plane arrays are row/column-subset, concatenated, cast to float32, and each selected neuron is z-scored over retained session timepoints before trial slicing.

ii. ```python
blocks.append(p[sel][:, keep_idx])
X = np.concatenate(blocks, axis=0).astype(np.float32)
mu = X.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
sd = X.std(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
X -= mu; X /= sd
```

iii. The notes call z-scoring a deliberate reference deviation intended to condition the uncentered-SVD decoder and normalize heterogeneous deconvolution amplitudes.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons outside V1/mHV/lHV/aHV are removed; at most 2,000 visual neurons per session are retained by seeded, proportional, area-stratified random sampling. Zero-variance rows are not removed; their divisor is set to one.

ii. ```python
neu_idx, region = select_neurons(iarea, task['n_keep'], task['seed'])
if n_vis <= n_keep: ...
else: ... rng.choice(a, size=k, replace=False)
sd[sd == 0] = 1.0
```

iii. The agent says visual-area filtering matches the reference, while the 2,000 cap makes the 4.1-million-neuron dataset tractable and matches the decoder's SVD limit.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural columns use the same retained frame indices as all variables and are grouped by trial index. Trial time is computed relative to `Trial_start_time`, described as corridor entry.

ii. ```python
keep_idx = np.flatnonzero(keep)
results[r['key']]['trials'] = [np.ascontiguousarray(X[:, a:b]) for (a, b) in slices]
t_since_start = (ft - b['Trial_start_time'][tri]) * SEC_PER_DAY
```

iii. The notes say shared imaging-frame indices give direct alignment and identify trial start with the VR position reset/corridor entry.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Each native imaging frame is one bin; no temporal rebinning is applied. Metadata uses the median session frame rate, about 314.69 ms per bin.

ii. ```python
bin_ms = float(1000.0 / np.median(fs_all))
'time_bin_size': bin_ms
```

iii. The agent argues that native imaging is already slow and position rebinning would compromise decoding position.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from retained-frame timestamps `ft`, per-trial `SoundTime`, and the frame's `ft_trInd` trial ID.

ii. ```python
ft = b['ft'][idx]
t_to_cue = (b['SoundTime'][tri] - ft) * SEC_PER_DAY
```

iii. The agent chose wall-clock event timestamps, noting that frame-count/running time would confound time with position.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Cue time minus frame time is converted from MATLAB days to seconds and clipped to [-30, 30] seconds.

ii. ```python
t_to_cue = (b['SoundTime'][tri] - ft) * SEC_PER_DAY
t_to_cue = np.clip(t_to_cue, -TIME_CLIP_S, TIME_CLIP_S)
```

iii. The notes say clipping about 2% of long-pause samples avoids extreme values and improves decoder conditioning without dropping data.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at exactly the retained imaging timestamps and split with the same trial slices as neural data.

ii. ```python
idx = np.flatnonzero(keep)
ft = b['ft'][idx]
inp[0] = t_to_cue[s:e]
```

iii. The notes report checking the zero crossing against `SoundFr` and finding no offset.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It comes from the calendar date `datexp` embedded in each recording and the earliest recording date for that mouse.

ii. ```python
d0 = min(datetime.date(*map(int, r['datexp'].split('_'))) for r in rs)
r['day'] = (datetime.date(*map(int, r['datexp'].split('_'))) - d0).days
```

iii. The agent interprets “day” literally as elapsed calendar days; notes report a 0–92 range.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Dates are parsed, elapsed days from the mouse's first recording are computed, and that scalar is broadcast over every retained frame of the trial.

ii. ```python
inp[1] = rec['day']
```

iii. The decision is documented as “days elapsed since this mouse's first recording session.”

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It uses per-frame `ft`, the trial-specific `Trial_start_time`, and `ft_trInd`.

ii. ```python
t_since_start = (ft - b['Trial_start_time'][tri]) * SEC_PER_DAY
```

iii. The agent treats the timestamp as the corridor-entry/VR-reset time.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The timestamp difference is converted from days to seconds and clipped to [0, 30] seconds.

ii. ```python
t_since_start = np.clip(t_since_start, 0.0, TIME_CLIP_S)
```

iii. The notes justify clipping long pauses for numerical conditioning while retaining their samples.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is evaluated on the same retained frame timestamps and sliced by the same trial boundaries as neural data.

ii. ```python
inp[2] = t_since_start[s:e]
```

iii. Shared frame indexing is the stated alignment mechanism.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It uses `WallName` and `isRew`: the first wall marked rewarded identifies the rewarded wall, and every occurrence of that wall is labeled available; sessions with no rewarded trial are all zero.

ii. ```python
if b['isRew'].any():
    rew_stim = wall[b['isRew']][0]
    rew_avail = (wall == rew_stim).astype(np.float32)
else:
    rew_avail = np.zeros(ntr, dtype=np.float32)
```

iii. The agent says this follows reference `get_cat_id` and represents rewarded-corridor availability, not reward delivery, avoiding leakage from licking.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The rewarded wall comparison becomes a binary per-trial scalar and is broadcast through the trial.

ii. ```python
inp[3] = rew_avail[t]
```

iii. The notes emphasize that unsupervised/naive sessions remain all zero.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from per-trial `WallName`, `UniqWalls`, and `stim_id` fields merged from all index/behavior representations of a recording, with a manual fallback for `circle3`.

ii. ```python
for wall, sid in zip(beh['UniqWalls'], db['stim_id']):
    r['stim_map'][str(wall)] = int(sid)
stim_id = np.array([rec['stim_map'].get(w, STIM_EXTRA.get(w, -1)) for w in wall])
```

iii. The agent says canonical role IDs pool physical rock/brick textures onto cross-mouse experimental roles used by reference utilities.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Wall names map to eight canonical role classes (0–7), validated as nonnegative, and the per-trial code is broadcast to all trial frames.

ii. ```python
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2', 'circle3']
out[0] = stim_id[t]
```

iii. The agent argues these roles have consistent experimental meaning across mice and mirror reference category-ID logic.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It comes from `LickFr`, the fractional imaging-frame index of lick events.

ii. ```python
lick_fr = b['LickFr']
```

iii. The notes cite the reference's `LickFr.astype(int)` convention.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Nonfinite/out-of-range events are removed, frame indices are truncated to integers, and those frames are marked one in an otherwise-zero Boolean series; multiple licks in a frame remain one.

ii. ```python
lick_fr = lick_fr[np.isfinite(lick_fr)]
lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nfr)].astype(np.int64)
lick_frame[lick_fr] = True
```

iii. The agent says this directly follows reference event-frame handling.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The frame flag is indexed by the retained neural frame indices and then split with the same trial slices.

ii. ```python
licking = lick_frame[idx].astype(np.int64)
out[1] = licking[s:e]
```

iii. Alignment is by shared imaging-frame number.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It comes from per-frame `ft_Pos` on retained frames; `Texture_Length`/the fixed 40-dm geometry establishes the corridor extent.

ii. ```python
pos = b['ft_Pos'][idx]
CORRIDOR_DM = 40.0
```

iii. The notes identify raw positions as decimeters over the 0–4 m texture corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is divided by the 10-dm bin width, converted to integer, and clipped to class 0–3.

ii. ```python
POS_BIN_DM = CORRIDOR_DM / N_POS_BINS
pos_bin = np.clip((pos / POS_BIN_DM).astype(np.int64), 0, N_POS_BINS - 1)
```

iii. This directly implements four equal 1-m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are 0, 10, 20, 30, and 40 dm, producing `[0,1)`, `[1,2)`, `[2,3)`, and `[3,4]` m classes (with clipping at endpoints).

ii. ```python
'position_bin_edges_m': [0.0, 1.0, 2.0, 3.0, 4.0]
```

iii. The agent documents these as exactly four equal-length bins requested by the task.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is sampled using the same retained frame indices and trial slices as neural activity.

ii. ```python
pos = b['ft_Pos'][idx]
out[2] = pos_bin[s:e]
```

iii. Shared imaging-frame indices provide alignment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed` at every retained frame in all selected sessions.

ii. ```python
all_speed = np.concatenate([r['beh']['ft_RunSpeed'][:r['nfr']][r['keep']] for r in recs])
```

iii. The agent uses the raw speed stream directly before categorization.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Global 25th/50th/75th percentile value edges are computed from pooled retained frames; each speed is assigned with `searchsorted` and clipped to class 0–3.

ii. ```python
speed_edges = np.percentile(all_speed, [25, 50, 75])
speed_bin = np.searchsorted(speed_edges, speed, side='right').astype(np.int64)
```

iii. The notes interpret “each corresponding to 25% of the data” globally and report exactly 25% per bin after moving-frame filtering.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three global value thresholds (reported as about 12.42, 25.35, and 40.85 cm/s) define the four speed classes.

ii. ```python
'running_speed_bin_edges_cm_s': [float(x) for x in speed_edges]
```

iii. The agent preferred pooled thresholds so class meaning is consistent across sessions.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is selected at the exact retained neural frames and split by identical trial slices.

ii. ```python
speed = b['ft_RunSpeed'][idx]
out[3] = speed_bin[s:e]
```

iii. Shared imaging-frame indices provide alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural `nfr` bounds every behavioral stream; invalid trial indices and invalid lick events are removed; 21 within-trial position-wrap frames are discarded; assertions catch area-count, mapping, monotonicity, and shape inconsistencies. The code does not silently skip failed sessions.

ii. ```python
keep = (...) & ~np.isnan(tr)
lick_fr = lick_fr[np.isfinite(lick_fr)]
assert len(iarea) == n_total
assert nz.shape[1] == ip.shape[1] == op.shape[1]
```

iii. The notes describe discovering one-frame behavior/neural length mismatches and making spike-file `nfr` authoritative; boundary frames were considered label-ambiguous.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the 405 GB of spike files dominates; neural selection/z-scoring and writing the multi-GB pickle are secondary.

ii. ```python
with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(tasks))) as ex:
    for i, res in enumerate(ex.map(process_neural, tasks)):
```

iii. The notes measure about 1.2 GB/s serial versus 5.8 GB/s with six readers and identify I/O as the principal cost.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Small Python loops remain for index records, behavior entries, imaging planes, trial slices, and output assembly. Most heavy numeric work is vectorized; trial construction could be batched but variable lengths make the loop natural.

ii. ```python
for pi, p in enumerate(planes):
for s, e, t in zip(starts, stops, trial_ids):
for r in recs:
```

iii. The agent's efficiency notes focus on parallel I/O and avoiding full plane concatenation, implying these small loops are negligible beside disk reads.

## 12-c. What processing does the code repeat multiple times?

i. The retained-frame mask is computed once in each neural worker and again in the main process; selected-frame speed is traversed once to obtain global edges and again to create outputs; trial slices are iterated during creation, neural slicing, checks, and summaries.

ii. ```python
keep = task['keep_fn'](nfr)
r['keep'] = compute_frame_selection(r, r['nfr'])
assert r['keep'].sum() == results[r['key']]['X'].shape[1]
```

iii. The duplicate mask is intentional validation that worker and main-process selection agree; the notes highlight this assertion as protection against frame-count mistakes.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes diagnostic dictionaries and raw/timing/dead-neuron statistics for every session, although much is used only for logging or optional plots; `_extract_beh` also retains some fields not needed in the final pickle. Full-session `X` is later replaced by per-trial copies.

ii. ```python
raw_stats = {'mean': float(X.mean()) if X.size else 0.0,
             'max': float(X.max()) if X.size else 0.0}
diag = {'idx': idx, 'tri': tri, 'pos': pos, ...}
results[r['key']]['X'] = None
```

iii. The notes justify diagnostics as sanity checks and provenance, and the temporary session matrix as an efficient way to normalize once before trial slicing.
