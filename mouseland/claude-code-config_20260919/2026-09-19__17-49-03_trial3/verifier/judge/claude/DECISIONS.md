# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads `Imaging_Exp_info.npy` to build a table of 89 unique recordings (keyed by `mname_datexp_blk`). It then loads all 23 `Beh_<exp_type>.npy` files once each, extracting relevant behaviour arrays per recording. The canonical `stim_id` mapping is merged across all experiment types containing a recording. Neural data (`spk` files) and retinotopy files are loaded per session in parallel worker processes.

ii.
```python
exp_info = np.load(os.path.join(BEH_DIR, 'Imaging_Exp_info.npy'), allow_pickle=True).item()
# ...
key = '%s_%s_%s' % (db['mname'], db['datexp'], db['blk'])
# ...
B = np.load(os.path.join(BEH_DIR, 'Beh_' + exp_type + '.npy'), allow_pickle=True).item()
# Neural data loaded in worker processes:
fn = '%s_%s_%s_neural_data.npy' % (mname, datexp, blk)
planes = np.load(os.path.join(SPK_DIR, fn), allow_pickle=True).item()['spks']
```

iii. The AI followed the reference code's loading pattern: index from `Imaging_Exp_info`, behaviour from `Beh_*.npy`, neural from `spk/*.npy`, retinotopy from `retinotopy/*_trans.npz`. Documented in CONVERSION_NOTES Steps 1-2.

## 1-b. How are the data split into subjects?

i. The mouse name (`mname`) from each recording entry is used to identify subjects. The sorted unique set of mouse names gives 19 subjects, and `subject_idx` maps each session to its subject.

ii.
```python
subjects = sorted({r['mname'] for r in recs})
data['subject_idx'] = np.array([subjects.index(r['mname']) for r in recs], dtype=np.int64)
```

iii. The `mname` field directly identifies the subject. No derivation needed.

## 1-c. How are the data split into sessions?

i. A session is one unique recording identified by `(mname, datexp, blk)`. The 142 database entries in `Imaging_Exp_info` are deduplicated to 89 unique recordings. A recording appearing under multiple experiment types is kept once; the `stim_id` mapping is merged from all experiment types.

ii.
```python
key = '%s_%s_%s' % (db['mname'], db['datexp'], db['blk'])
if key not in recs:
    recs[key] = { ... }
r['exp_types'].append(exp_type)
```

iii. Documented in CONVERSION_NOTES Step 4, where the AI verified 89 unique recordings matching the paper's claim.

## 1-d. How are the data split into trials?

i. Trials are defined by frames where `ft_trInd` gives the trial index. The AI applies a frame mask `(ft_move > 0) & ft_CorrSpc & ~isnan(ft_trInd)` — retaining only frames inside the texture corridor while the VR is moving (mouse is running). Within these retained frames, trial boundaries are detected by changes in `ft_trInd`.

ii.
```python
def frame_mask(ft_move, ft_CorrSpc, ft_trInd, ft_Pos, nfr):
    keep = (ft_move[:nfr] > 0) & ft_CorrSpc[:nfr] & ~np.isnan(tr)
    # ...
    return keep

# Trial splitting from retained frames:
bounds = np.flatnonzero(np.diff(tri)) + 1
starts = np.concatenate(([0], bounds))
stops = np.concatenate((bounds, [len(tri)]))
trial_ids = tri[starts]
```

iii. The AI explicitly followed `Get_dprime_selective_neuron`'s `fr_valid = (ft_move > 0) & ft_CorrSpc`, the canonical frame selection in the reference code. This means non-running frames within the corridor are excluded from trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI does NOT drop any trials. All 38,110 trials are retained, as every trial has at least one retained frame after the frame mask. The AI explicitly notes "No trials are dropped" in CONVERSION_NOTES Step 5 Decision 10.

ii.
```python
# No trial filtering code — all trials with retained frames are kept
bounds = np.flatnonzero(np.diff(tri)) + 1
# All trial_ids from the retained frames are used
```

iii. The AI verified that every trial has at least one retained frame and argued that no trial-level exclusion criteria exist in the paper's main analyses.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the `spk/<session_id>_neural_data.npy` files (list of 3 plane arrays, concatenated along neuron axis) and `iarea` from `retinotopy/<mname>_<datexp>_trans.npz` for area assignment.

ii.
```python
planes = np.load(os.path.join(SPK_DIR, fn), allow_pickle=True).item()['spks']
iarea = np.load(os.path.join(RET_DIR, '%s_%s_trans.npz' % (mname, datexp)))['iarea']
```

iii. Same as reference — deconvolved Suite2p traces concatenated from imaging planes.

## 2-b. How is the `neural` data processed?

i. The AI applies three processing steps not in the reference: (1) filter to visual-cortex neurons (`iarea not in {-1, 7}`), (2) subsample to at most 2000 neurons per session (stratified proportionally across V1/mHV/lHV/aHV), and (3) z-score each neuron over the retained timepoints of the session. The data is stored as float32.

ii.
```python
# Subsampling
neu_idx, region = select_neurons(iarea, task['n_keep'], task['seed'])
# Z-scoring
mu = X.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
sd = X.std(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
sd[sd == 0] = 1.0
X -= mu
X /= sd
```

iii. The AI justified subsampling as necessary for tractability (4.1M neurons would be ~150GB), matching the decoder's `svd_max_neurons=2000` default. Z-scoring was justified as needed because raw deconvolved amplitudes have wildly heterogeneous scales that would saturate the decoder's linear layer (CONVERSION_NOTES Step 5, Decisions 4-5).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by visual area: only neurons with `iarea` in {8, 0, 1, 2, 9, 5, 6, 3, 4} (corresponding to V1, mHV, lHV, aHV) are kept. Then subsampled to 2000 per session.

ii.
```python
def neu_area_ID(iarea):
    idx = {}
    for ar in BRAIN_REGIONS:
        if ar == 'V1':   idx[ar] = iarea == 8
        elif ar == 'mHV': idx[ar] = (iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)
        elif ar == 'lHV': idx[ar] = (iarea == 5) | (iarea == 6)
        elif ar == 'aHV': idx[ar] = (iarea == 3) | (iarea == 4)
    return idx
```

iii. The area filtering matches the reference code exactly (`neu_area_ID` copied from utils.py). The 2000-neuron subsample is an additional step not in the reference.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials start at corridor entry (where `ft_trInd` becomes the trial index and `ft_Pos` restarts at 0). Within a trial, only frames satisfying `(ft_move > 0) & ft_CorrSpc` are retained. Trials have variable lengths. `off_start = 0.0`, `off_end = None`.

ii.
```python
'temporal_alignment_event': 'trial start = entry into the virtual-reality corridor'
'off_start': 0.0,
'off_end': None,
```

iii. This matches the instruction's "Temporally aligned based on trial start (corridor entry)".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning. The native imaging frame rate (~3.18 Hz, ~314.7 ms per bin) is used directly. However, non-running frames are excluded, so the time bins within a trial may not be contiguous in wall-clock time.

ii.
```python
bin_ms = float(1000.0 / np.median(fs_all))
# ~314.7 ms
```

iii. The AI correctly notes the imaging frame is the finest resolution available, and the paper's only rebinning is onto position which can't be used when position is a decoder target.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime` (per-trial wall-clock time of the sound cue, MATLAB datenum) and `ft` (per-frame wall-clock timestamps, MATLAB datenum).

ii.
```python
t_to_cue = (b['SoundTime'][tri] - ft) * SEC_PER_DAY
```

iii. The AI uses `SoundTime` directly rather than `SoundFr` (frame number). Both encode the same event but in different units.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The time to the sound cue is computed as `(SoundTime[trial] - ft[frame]) * 86400` in seconds (positive before the cue, negative after). The result is clipped to ±30 seconds.

ii.
```python
t_to_cue = (b['SoundTime'][tri] - ft) * SEC_PER_DAY
t_to_cue = np.clip(t_to_cue, -TIME_CLIP_S, TIME_CLIP_S)
```

iii. Clipping to ±30s is justified as keeping the linear decoder well-conditioned. The sign convention matches the instruction naming ("time TO sound cue" = positive before).

## 3-c. How is `input` *Time to sound cue* aligned with the neural data?

i. Both neural data and the time input use the same retained frame indices (`idx`), so they are inherently aligned.

ii.
```python
idx = np.flatnonzero(keep)
ft = b['ft'][idx]
t_to_cue = (b['SoundTime'][tri] - ft) * SEC_PER_DAY
```

iii. All data streams use the same frame indices from the frame mask.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From `datexp` (date string in the recording key) for each session of each mouse.

ii.
```python
d0 = min(datetime.date(*map(int, r['datexp'].split('_'))) for r in rs)
for r in rs:
    r['day'] = (datetime.date(*map(int, r['datexp'].split('_'))) - d0).days
```

iii. The AI parses actual dates from the session IDs.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the earliest recording date is found. The day of training for each session is the number of calendar days since that first recording (e.g., 0, 7, 9, 16, ..., up to 92). This is broadcast across all time bins within a trial.

ii.
```python
d0 = min(datetime.date(*map(int, r['datexp'].split('_'))) for r in rs)
r['day'] = (datetime.date(*map(int, r['datexp'].split('_'))) - d0).days
# In trial construction:
inp[1] = rec['day']
```

iii. This counts actual calendar days rather than recording session numbers.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time` (per-trial wall-clock time of corridor entry, MATLAB datenum) and `ft` (per-frame wall-clock timestamps).

ii.
```python
t_since_start = (ft - b['Trial_start_time'][tri]) * SEC_PER_DAY
```

iii. Uses wall-clock timestamps directly.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Time since trial start is `(ft[frame] - Trial_start_time[trial]) * 86400` in seconds. Clipped to [0, 30] seconds.

ii.
```python
t_since_start = (ft - b['Trial_start_time'][tri]) * SEC_PER_DAY
t_since_start = np.clip(t_since_start, 0.0, TIME_CLIP_S)
```

iii. Clipping to 30s prevents extreme outliers from trials where the mouse paused for a long time.

## 5-c. How is `input` *Time since trial start* aligned with the neural data?

i. Same retained frame indices as neural data, so alignment is inherent.

ii.
```python
idx = np.flatnonzero(keep)
ft = b['ft'][idx]
t_since_start = (ft - b['Trial_start_time'][tri]) * SEC_PER_DAY
```

iii. All streams share the same frame mask.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `WallName` (stimulus name per trial) and `isRew` (whether reward was delivered per trial).

ii.
```python
if b['isRew'].any():
    rew_stim = wall[b['isRew']][0]
    rew_avail = (wall == rew_stim).astype(np.float32)
else:
    rew_avail = np.zeros(ntr, dtype=np.float32)
```

iii. The AI uses `isRew` to identify which stimulus is rewarded, then marks all trials with that stimulus as reward-available, rather than using `isRew` directly.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The rewarded stimulus is identified as `WallName[isRew][0]` (the wall texture of the first trial where reward was actually delivered). Then reward availability is 1 for all trials with that wall texture, 0 otherwise. For unsupervised/naive mice (no rewards), all trials get 0. This is a per-trial scalar broadcast across time bins.

ii.
```python
rew_stim = wall[b['isRew']][0]
rew_avail = (wall == rew_stim).astype(np.float32)
# Per-trial broadcast:
inp[3] = rew_avail[t]
```

iii. The AI explicitly chose "rewarded corridor" over "reward delivered" to avoid leaking the licking output into the decoder input (CONVERSION_NOTES Step 4).

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName` (per-trial stimulus name) and the merged `stim_id` mapping from `Imaging_Exp_info`.

ii.
```python
stim_id = np.array([rec['stim_map'].get(w, STIM_EXTRA.get(w, -1)) for w in wall], dtype=np.int64)
```

iii. The AI uses the canonical stimulus role IDs from the reference code.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each wall texture is mapped to a canonical role ID (0-7): `circle1=0, circle2=1, leaf1=2, leaf2=3, leaf3=4, leaf1_swap1=5, leaf1_swap2=6, circle3=7`. These are the reference code's canonical stimulus roles, with circle3 (id 7) added for the 309 otherwise-unlabelled trials. This gives 8 stimulus categories. The value is per-trial, broadcast across time bins.

ii.
```python
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2', 'circle3']
STIM_EXTRA = {'circle3': 7}
# Per-trial broadcast:
out[0] = stim_id[t]
```

iii. The AI followed the reference code's `Get_coding_direction` docstring for role IDs 0-6 and added 7 for `circle3`.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` — the (fractional) imaging frame number of each lick event in the session.

ii.
```python
lick_fr = b['LickFr']
lick_fr = lick_fr[np.isfinite(lick_fr)]
lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nfr)].astype(np.int64)
lick_frame = np.zeros(nfr, dtype=bool)
lick_frame[lick_fr] = True
licking = lick_frame[idx].astype(np.int64)
```

iii. Same source as reference — `LickFr` with `.astype(int)` truncation.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Fractional lick frame numbers are truncated to integers (`astype(int64)`), filtered to valid range, then a binary flag per frame is created (1 = at least one lick in that frame, 0 = no lick). The AI also filters for `isfinite` to handle potential NaN values.

ii.
```python
lick_fr = b['LickFr']
lick_fr = lick_fr[np.isfinite(lick_fr)]
lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nfr)].astype(np.int64)
lick_frame = np.zeros(nfr, dtype=bool)
lick_frame[lick_fr] = True
```

iii. Follows the reference code convention in `spk_2_firstLick`: `beh['LickFr'].astype(int)`.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick flag array is indexed by the same retained frame indices (`idx`) as the neural data.

ii.
```python
licking = lick_frame[idx].astype(np.int64)
```

iii. Same frame mask alignment as all other variables.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos` — the VR position (in decimetres) at each imaging frame.

ii.
```python
pos = b['ft_Pos'][idx]
```

iii. Direct from the behaviour data.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimetres is divided by `POS_BIN_DM` (10 dm = 1 m) and truncated to integer, then clipped to [0, 3]. This gives 4 bins of 1 m each.

ii.
```python
CORRIDOR_DM = 40.0
N_POS_BINS = 4
POS_BIN_DM = CORRIDOR_DM / N_POS_BINS  # = 10
pos_bin = np.clip((pos / POS_BIN_DM).astype(np.int64), 0, N_POS_BINS - 1)
```

iii. Matches the instruction requirement of "4 equal-length, 1-m-long spatial bins".

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position (0-40 dm) is divided by 10 dm and floored to get bins 0-3: [0-1m), [1-2m), [2-3m), [3-4m]. Clipped to ensure values stay in [0, 3].

ii.
```python
pos_bin = np.clip((pos / POS_BIN_DM).astype(np.int64), 0, N_POS_BINS - 1)
```

iii. Same logic as reference (divide by 10, floor, clip).

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is indexed at the same retained frame indices as neural data.

ii.
```python
pos = b['ft_Pos'][idx]
```

iii. Same frame mask alignment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed` — the running speed (cm/s) at each imaging frame.

ii.
```python
speed = b['ft_RunSpeed'][idx]
```

iii. Direct from behaviour data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is discretized into 4 quartile bins using **global** percentile edges computed over ALL retained timepoints across ALL sessions. The 25th, 50th, and 75th percentiles are computed, then `np.searchsorted` assigns each speed to a bin.

ii.
```python
all_speed = np.concatenate([r['beh']['ft_RunSpeed'][:r['nfr']][r['keep']] for r in recs])
speed_edges = np.percentile(all_speed, [25, 50, 75])
# Per trial:
speed_bin = np.searchsorted(speed_edges, speed, side='right').astype(np.int64)
speed_bin = np.clip(speed_bin, 0, N_SPEED_BINS - 1)
```

iii. The AI computed global quartiles so each bin holds exactly 25% of the total data, as the instruction specifies.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The three global quartile edges split the retained speed values into 4 bins. `np.searchsorted` assigns each value to its bin based on these edges.

ii.
```python
speed_edges = np.percentile(all_speed, [25, 50, 75])
speed_bin = np.searchsorted(speed_edges, speed, side='right')
```

iii. Using `np.percentile` on the pooled data ensures each bin has ~25% globally.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is indexed at the same retained frame indices as neural data.

ii.
```python
speed = b['ft_RunSpeed'][idx]
```

iii. Same frame mask alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: (1) behaviour arrays are truncated to `nfr = spk.shape[1]` since behaviour can be 1 frame longer; (2) lick frames are filtered for `isfinite` and valid range; (3) a boundary wrap edge case is handled where `ft_Pos` wraps back to ~0 while `ft_trInd` still reports the previous trial (21 frames across the whole dataset are dropped); (4) `circle3` trials with no `stim_id` mapping get a new ID 7 instead of being dropped.

ii.
```python
# Boundary wrap handling in frame_mask:
wrap = np.flatnonzero((np.diff(pos) < 0) & (np.diff(tri) == 0)) + 1
if len(wrap):
    keep[idx[wrap]] = False

# Lick frame filtering:
lick_fr = lick_fr[np.isfinite(lick_fr)]
lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nfr)]

# circle3 handling:
STIM_EXTRA = {'circle3': 7}
```

iii. The AI was thorough about edge cases, particularly the corridor boundary wrap which is not handled in the reference code.

## 12-a. What are the most time-consuming steps of the code?

i. Reading the 405 GB of spike files. The AI uses 6 parallel worker processes to achieve ~5.8 GB/s aggregate throughput, completing all reads in ~75 seconds.

ii.
```python
N_WORKERS = int(os.environ.get('CONVERT_WORKERS', '6'))
with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(tasks))) as ex:
    for i, res in enumerate(ex.map(process_neural, tasks)):
```

iii. Documented in CONVERSION_NOTES Step 6.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `build_trial_variables` that constructs input/output arrays for each trial could potentially be vectorized, but it operates on variable-length slices so vectorization is limited. The overall code is already fairly well vectorized with numpy operations on full arrays before trial splitting.

ii.
```python
for s, e, t in zip(starts, stops, trial_ids):
    T = e - s
    inp = np.empty((4, T), dtype=np.float32)
    # ...
```

iii. N/A

## 12-c. What processing does the code repeat multiple times?

i. The frame mask (`frame_mask`) is computed twice: once in the worker process (`process_neural`) and once in the main process (`compute_frame_selection`), with an assertion that they agree. This is by design (to verify consistency) rather than an oversight.

ii.
```python
# In worker:
keep = task['keep_fn'](nfr)
# In main:
r['keep'] = compute_frame_selection(r, r['nfr'])
assert r['keep'].sum() == results[r['key']]['X'].shape[1]
```

iii. The duplication is intentional verification.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes detailed diagnostic information (`diag` dict) for every session even when `--show-processing` is not used. The z-scoring of neural data is extra processing not in the reference that may or may not help the decoder. The neuron subsampling discards the majority of recorded neurons (~46k to 2000 per session).
