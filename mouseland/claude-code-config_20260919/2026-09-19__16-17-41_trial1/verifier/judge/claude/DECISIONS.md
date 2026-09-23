# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three directories under `data/`: `beh/` (behavior), `spk/` (deconvolved calcium traces), and `retinotopy/` (visual area assignments). It first reads `beh/Imaging_Exp_info.npy` as the master index of all recordings grouped by experiment type. It builds a session index by deduplicating to 89 unique physical recordings. Behavior files (`Beh_<exp_type>.npy`) are loaded once per experiment type, keeping only needed fields. Spike files and retinotopy files are loaded per session.

ii. Session index construction:
```python
exp_info = np.load(os.path.join(ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=1).item()
sessions = {}
for exp_type, db in exp_info.items():
    for ndb in db:
        key = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
        beh_key = key + ('_' + ndb['stimtype'] if 'stimtype' in ndb else '')
        rec = sessions.setdefault(key, dict(...))
```
Spike loading:
```python
fn = os.path.join(ROOT, 'spk', '%s_%s_%s_neural_data.npy' % (mname, datexp, blk))
spks = np.load(fn, allow_pickle=True).item()['spks']
```
Retinotopy:
```python
fn = os.path.join(ROOT, 'retinotopy', '%s_%s_trans.npz' % (mname, datexp))
```

iii. The AI documents this in CONVERSION_NOTES.md Steps 1-2. The loading mirrors the reference code's `load_exp_beh`, `load_spk`, and `load_retino` functions. The AI also merges `stim_id` across experiment types to get complete stimulus labeling.

## 1-b. How are the data split into subjects?

i. The mouse name is taken from `mname` in the session index. Subjects are accumulated as unique names encountered during iteration over sessions.

ii.
```python
if r['mname'] not in subjects:
    subjects.append(r['mname'])
subject_idx.append(subjects.index(r['mname']))
```

iii. The session index already contains the mouse name. The AI maintains insertion order for subjects rather than sorting them.

## 1-c. How are the data split into sessions?

i. A session is one unique (mname, datexp, blk) triple. The AI deduplicates the 141 entries in `Imaging_Exp_info.npy` down to 89 physical recordings by using the triple as a key.

ii.
```python
key = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
rec = sessions.setdefault(key, dict(...))
```

iii. Documented in CONVERSION_NOTES.md. The AI notes that the same recording appears under multiple experiment types and deduplicates accordingly.

## 1-d. How are the data split into trials?

i. Trials are defined by `ft_trInd` (trial index per frame) combined with two frame masks: `ft_CorrSpc` (inside the 0-4m texture area) AND `ft_move > 0` (VR is moving / mouse is running). Only frames satisfying both conditions are kept. This matches the reference code's `fr_valid` pattern from `Get_dprime_selective_neuron`.

ii.
```python
corr = b['ft_CorrSpc'][:nfr].astype(bool)
move = b['ft_move'][:nfr] > 0
valid = corr & move & np.isfinite(tr)
```

iii. The AI justifies this by citing the reference code's `Get_dprime_selective_neuron` which uses `fr_valid = VRmove & isCorridor`, and the paper's statement "We only considered timepoints during running for analysis." Documented extensively in CONVERSION_NOTES.md Steps 1 and 5.

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped if: (1) they have zero valid frames after applying the frame mask, or (2) their wall texture has no canonical `stim_id` mapping. No trial-length-based filtering is applied.

ii.
```python
if n == 0:
    continue
if stim_of_trial[t] < 0:  # wall without a canonical stim id
    continue
```

iii. The AI notes that trials without a stim_id are never used in the reference analyses. No explicit length-based outlier removal is performed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy`, which contains a list of (n_neurons_per_plane, n_frames) float32 arrays. The visual area of each neuron comes from `iarea` in `retinotopy/<mname>_<datexp>_trans.npz`.

ii.
```python
spks = np.load(fn, allow_pickle=True).item()['spks']
iarea = load_iarea(r['mname'], r['datexp'])
```

iii. Same source as the reference code's `load_spk`.

## 2-b. How is the `neural` data processed?

i. The AI applies per-neuron z-scoring over ALL frames of the session (not just kept frames): subtracts the mean and divides by the standard deviation of each neuron's trace across all frames. The result is stored as float32.

ii.
```python
def extract_neural(blocks, keep_mask, frame_idx, zscore=True, ...):
    ...
    if zscore:
        nf = sub.shape[1]
        s1 = sub.sum(axis=1, dtype=np.float64)
        s2 = np.einsum('ij,ij->i', sub, sub, dtype=np.float64)
        mu = s1 / nf
        sd = np.sqrt(np.maximum(s2 / nf - mu * mu, 0.0))
        sd[sd == 0] = 1.0
        g -= mu[km, None].astype(np.float32)
        g /= sd[km, None].astype(np.float32)
```

iii. The AI justifies z-scoring by citing the reference code's `get_kfold_reward_response` and `Get_sort_spk` which both use `stats.zscore(spk, axis=1)`. Documented in CONVERSION_NOTES.md.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are kept if their `iarea` is in V1 (8), mHV (0,1,2,9), lHV (5,6), or aHV (3,4). Neurons with `iarea == -1` or `iarea == 7` are excluded. This keeps 4,105,393 of 4,691,034 total neurons.

ii.
```python
def neu_area_idx(iarea):
    reg = np.full(len(iarea), -1, dtype=np.int64)
    reg[iarea == 8] = 0                    # V1
    reg[np.isin(iarea, [0, 1, 2, 9])] = 1  # medial HVAs
    reg[np.isin(iarea, [5, 6])] = 2        # lateral HVAs
    reg[np.isin(iarea, [3, 4])] = 3        # anterior HVAs
    keep = reg >= 0
    return keep, reg[keep]
```

iii. Matches the reference code's `neu_area_ID` and `Get_density_map` which excludes `iarea in {-1, 7}`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry (`Trial_start_time` / `StartFr`). The valid frames of each trial start at corridor entry and run through the texture area. `off_start = 0.0`, `off_end = None` because trial lengths are variable.

ii.
```python
data['metadata'] = {
    ...
    'temporal_alignment_event': 'trial start = corridor entry ...',
    'off_start': 0.0,
    'off_end': None,
}
```

iii. The instructions specify alignment to trial start (corridor entry), which is what the AI implements.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The native imaging frame rate (~3.17 Hz, ~314.8 ms per frame) is used as the time bin. The AI computes the exact bin size as the mean of per-session median frame intervals.

ii.
```python
dt = float(np.mean([s['frame_interval_s'] for s in session_info]))
data['metadata'] = {
    'time_bin_size': dt * 1000.0,  # ms
}
```

iii. The reference never re-bins in time; all time-domain analyses use native imaging frames.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime` (the MATLAB datenum of the sound cue for each trial) and `ft` (the MATLAB datenum of each imaging frame).

ii.
```python
tcue = np.asarray(b['SoundTime'], dtype=float)
...
inp[0] = (tcue[t] - ftt) * SEC_PER_DAY  # + before the sound cue, - after
```

iii. The AI uses `SoundTime` (a MATLAB datenum) rather than `SoundFr` (a frame number), computing time differences directly from timestamps.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the MATLAB datenum of the sound cue is subtracted from the MATLAB datenum of each kept frame, and the result is multiplied by 86400 to convert from days to seconds. The sign convention is positive before the cue, negative after.

ii.
```python
inp[0] = (tcue[t] - ftt) * SEC_PER_DAY  # + before the sound cue, - after
```
where `ftt = ft[fi]` are the frame timestamps of the trial's valid frames.

iii. Direct datenum subtraction gives time in days; multiplying by SEC_PER_DAY converts to seconds.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The same frame indices (`fi`) used to extract neural data are used to index into the frame timestamps, so neural and input data are aligned frame-by-frame.

ii.
```python
fi = np.sort(fi)  # sorted frame indices for this trial
ftt = ft[fi]
inp[0] = (tcue[t] - ftt) * SEC_PER_DAY
```

iii. All data streams share the same frame index array.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From `datexp` (the date string in YYYY_MM_DD format) in each session's metadata, and the set of all sessions per mouse to determine the first imaging date.

ii.
```python
def datenum(datexp):
    y, m, d = [int(x) for x in datexp.split('_')]
    import datetime
    return datetime.date(y, m, d).toordinal()
first_day = {}
for k in keys_all:
    mn = sessions[k]['mname']
    first_day[mn] = min(first_day.get(mn, 1 << 30), datenum(sessions[k]['datexp']))
day = {k: datenum(sessions[k]['datexp']) - first_day[sessions[k]['mname']] for k in keys_all}
```

iii. The AI uses the session date strings to compute actual calendar days.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI computes the calendar date ordinal for each session and subtracts the ordinal of the mouse's first imaging session. This gives the number of calendar days since the first recording, not the ordinal session count. For example, if a mouse has sessions on days 0, 7, 9, 16, those are the day_of_training values (rather than 0, 1, 2, 3).

ii.
```python
day = {k: datenum(sessions[k]['datexp']) - first_day[sessions[k]['mname']] for k in keys_all}
```
The verification output shows values like [0, 7, 9, 16, 17, 18, ...] up to 92.

iii. The AI interprets "day of training" literally as the calendar day count. This produces values from 0 to 92, compared to the reference's 0 to 7 (session count).

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time` (MATLAB datenum of corridor entry for each trial) and `ft` (MATLAB datenum of each imaging frame).

ii.
```python
tstart = np.asarray(b['Trial_start_time'], dtype=float)
...
inp[2] = (ftt - tstart[t]) * SEC_PER_DAY
```

iii. Uses MATLAB datenums directly rather than frame-number interpolation.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The MATLAB datenum of the corridor entry is subtracted from each kept frame's datenum, and multiplied by 86400 to convert to seconds. Values start near 0 and increase.

ii.
```python
inp[2] = (ftt - tstart[t]) * SEC_PER_DAY  # time since corridor entry
```

iii. Direct datenum subtraction followed by unit conversion.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same frame indices as neural data are used.

ii.
```python
fi = np.sort(fi)
ftt = ft[fi]
inp[2] = (ftt - tstart[t]) * SEC_PER_DAY
```

iii. All streams aligned via shared frame indices.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, a per-trial boolean/integer indicating whether the corridor is the rewarded one.

ii.
```python
isrew = np.asarray(b['isRew']).astype(np.float32)
...
inp[3] = isrew[t]
```

iii. Directly from the behavior data.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The value is cast to float32. It is 1 for rewarded corridors and 0 otherwise. It is broadcast across all timepoints of the trial.

ii.
```python
inp[3] = isrew[t]
```

iii. No processing beyond type casting. The AI notes that unsupervised and naive mice always have isRew=0.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From the canonical `stim_id` values, which the AI merges from `beh['stim_id']` across all experiment types for each recording. The mapping is from `UniqWalls` (unique wall names) to a canonical integer ID (0-6).

ii.
```python
stim_of_trial = np.full(ntrials, -1, dtype=np.int64)
smap = rec['stim_map']
for t in range(ntrials):
    stim_of_trial[t] = smap.get(str(b['WallName'][t]), -1)
```
The stim_map is built by merging stim_id across experiment types:
```python
for w, s in zip(walls, sid):
    if not np.isnan(s):
        stim_map[key][str(w)] = int(s)
```

iii. The AI uses the reference code's canonical stim_id system rather than mapping wall names to broad categories.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI uses 7 canonical stimulus IDs (circle1, circle2, leaf1, leaf2, leaf3, leaf1_swap1, leaf1_swap2) as the output categories. Wall names across mice (including rock/wood variants) are mapped to these canonical IDs via the `stim_id` field. The value is per-trial and broadcast across all timepoints.

ii.
```python
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3', 'leaf1_swap1', 'leaf1_swap2']
...
out[0] = stim_of_trial[t]
```

iii. The AI chose 7 categories based on the canonical stim_id scheme from the reference code, noting that rock/wood textures map to circle/leaf equivalents across mice.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the frame number of every lick event in the session.

ii.
```python
lick_frames = np.floor(b['LickFr']).astype(np.int64)
lick_frames = lick_frames[(lick_frames >= 0) & (lick_frames < nfr)]
lick = np.zeros(nfr, dtype=bool)
lick[lick_frames] = True
```

iii. Directly from lick frame numbers.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A binary indicator per frame: 1 if at least one lick falls in that frame, 0 otherwise. The fractional frame number is floored to get the integer frame index. Lick frames beyond the neural recording length are excluded.

ii.
```python
lick_frames = np.floor(b['LickFr']).astype(np.int64)
lick_frames = lick_frames[(lick_frames >= 0) & (lick_frames < nfr)]
lick = np.zeros(nfr, dtype=bool)
lick[lick_frames] = True
...
out[1] = lick[fi]
```

iii. Simple binarization of lick events onto the frame grid.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The same frame indices used for neural data index into the per-frame lick array.

ii.
```python
out[1] = lick[fi]
```

iii. All streams share frame indices.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position in decimeters at each imaging frame (0-40 in the texture area, 40-60 in the grey space).

ii.
```python
pos = b['ft_Pos'][:nfr]
...
out[2] = np.clip(np.floor(pos[fi] / POS_BIN_DM), 0, N_POS_BINS - 1)
```

iii. Directly from the per-frame position data.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimeters is divided by 10 (= 1 meter), floored, and clipped to the range [0, 3], giving four 1-meter bins.

ii.
```python
POS_BIN_DM = 10.0  # 1 m position bins
N_POS_BINS = 4
...
out[2] = np.clip(np.floor(pos[fi] / POS_BIN_DM), 0, N_POS_BINS - 1)
```

iii. Straightforward discretization into the 4 bins specified by the instructions.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Floor division by 10 dm gives bins at [0-1m), [1-2m), [2-3m), [3-4m). Clipping to [0,3] ensures positions exactly at 40 dm (4m) are placed in bin 3.

ii.
```python
out[2] = np.clip(np.floor(pos[fi] / POS_BIN_DM), 0, N_POS_BINS - 1)
```
Categories:
```python
['0-1m', '1-2m', '2-3m', '3-4m']
```

iii. Matches the instruction for "4 equal-length, 1-m-long spatial bins."

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Same frame indices as neural data.

ii.
```python
out[2] = np.clip(np.floor(pos[fi] / POS_BIN_DM), 0, N_POS_BINS - 1)
```

iii. All streams aligned via shared frame indices.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
speed = b['ft_RunSpeed'][:nfr]
...
speeds.append(ti['speed'])  # speed of kept frames
```

iii. Directly from the per-frame running speed data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI pools running speeds from all valid frames across ALL 89 sessions (even in --sample mode) and computes global 25th, 50th, and 75th percentile edges. These edges are then applied via `np.searchsorted` to bin each frame's speed into quartiles 0-3.

ii.
```python
# Pass 1: compute global quartile edges
speed_all = np.concatenate(speeds)
speed_edges = np.percentile(speed_all, [25, 50, 75])
...
# Pass 2: apply edges
sb = np.searchsorted(speed_edges, ti['speed'], side='right').astype(np.int64)
for i in range(len(ti['inputs'])):
    a, bnd = ti['bounds'][i], ti['bounds'][i + 1]
    ti['outputs'][i][3] = sb[a:bnd]
```

iii. The AI uses global percentile edges to ensure consistency across sessions. The edges are computed in a first pass over all behavior data.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Global percentile edges at [25%, 50%, 75%] of the pooled speed distribution. `np.searchsorted(..., side='right')` assigns each frame to one of 4 bins.

ii.
```python
speed_edges = np.percentile(speed_all, [25, 50, 75])
sb = np.searchsorted(speed_edges, ti['speed'], side='right')
```
The verification output shows globally: `Q1_slowest (0.250), Q2 (0.250), Q3 (0.250), Q4_fastest (0.250)` but per-session distributions are highly uneven (e.g., session 0: 95.6% in Q1).

iii. The AI chose global quartiles for consistency across sessions.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Same frame indices as neural data.

ii.
```python
ti['outputs'][i][3] = sb[a:bnd]
```

iii. Aligned via shared frame indices.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The behavior arrays are truncated to the neural frame count (`nfr = blocks[0].shape[1]`), matching the reference code's pattern. Lick frames beyond the neural recording are excluded. Trials with no valid frames or with unknown stimulus IDs are dropped. The AI also uses `np.isfinite(tr)` to handle any NaN trial indices. The AI drops pages from the OS page cache after reading spike files to manage memory pressure.

ii.
```python
nfr_true = blocks[0].shape[1]
assert nfr_true <= len(b['ft']), (k, nfr_true, len(b['ft']))
...
lick_frames = lick_frames[(lick_frames >= 0) & (lick_frames < nfr)]
...
valid = corr & move & np.isfinite(tr)
```

iii. The AI documents that behavior arrays are 1-2 frames longer than neural recordings and must be truncated. This matches the reference code's `[:nfr]` pattern.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the 89 spike files (totaling ~404 GB) is the dominant cost. The AI reports load times of 0-7s per session and extract times of 0.3-2.8s per session. Total conversion time is ~491 seconds (~8 minutes).

ii. From the conversion log:
```
(load  2.0s extract  0.8s total  3.9s)
```

iii. I/O-bound from reading large neural data files.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `build_trials` iterates over all trials to construct frame indices and input/output arrays. The `np.argsort` + `np.bincount` approach partially vectorizes frame grouping, but the per-trial output construction remains a loop.

ii.
```python
for t in range(ntrials):
    n = counts[t]
    if n == 0:
        continue
    fi = ordered[ptr:ptr + n]
    ...
```

iii. The loop is not a bottleneck compared to I/O.

## 12-c. What processing does the code repeat multiple times?

i. The AI runs two passes over behavior data. Pass 1 computes global speed quartile edges using `build_trials` on all 89 sessions with the full behavior frame count. Pass 2 re-runs `build_trials` on each processed session with the exact neural frame count. This means trial construction logic runs twice for every session.

ii.
```python
# Pass 1: building trial structure for all recordings
for k in keys_all:
    ti = build_trials(sessions[k], beh[k], len(beh[k]['ft']), day[k])
    speeds.append(ti['speed'])
...
# Pass 2: rebuild with exact neural frame count
ti = build_trials(r, b, nfr_true, day[k])
```

iii. The duplication is intentional: pass 1 needs all sessions' speeds to compute global quartiles, and pass 2 needs the exact neural frame count.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores z-scored neural data (float32), which increases storage from ~75 GB (float16 raw) to ~150 GB. The z-scoring normalization may be redundant if the downstream decoder performs its own normalization. The two-pass behavior processing is also somewhat redundant. Additionally, the AI stores extensive per-session metadata (session_info with many fields) that is not used by the decoder.

ii.
```python
neural = extract_neural(blocks, keep, ti['frame_idx'], zscore=not args.no_zscore)
```
The output file is 150.8 GB vs what would be ~75 GB with float16 raw traces.

iii. Z-scoring during conversion pre-empts downstream normalization but doubles storage requirements.
