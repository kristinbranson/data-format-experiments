# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Loading is driven by the master index `beh/Imaging_Exp_info.npy`, which is a dict of 23 experiment types, each a list of recording entries. `load_exp_info()` walks every entry and builds one record per unique `(mname, datexp, blk)` key (142 entries collapse to 89 unique sessions, 19 mice), accumulating the list of experiment types each recording appears under. Behaviour is then loaded in a single pass over all 23 `beh/Beh_<exp_type>.npy` files found by `glob` (not by following the index): for every key in every file, the session base key is recovered as the first five underscore-separated tokens, the first behaviour dict seen for a session is retained, and the `UniqWalls -> stim_id` mapping is merged (union) across every experiment type the session appears in. Neural data and anatomy are read per session inside the worker: `spk/<key>_neural_data.npy` (dict with `spks`, a list of per-plane arrays) and `retinotopy/<mname>_<datexp>_trans.npz` (`iarea`). Sessions are processed in a `multiprocessing` fork pool of 8 workers, with the shared behaviour dict inherited copy-on-write.

ii.
```python
def load_exp_info():
    exp_info = np.load(os.path.join(BEH_DIR, 'Imaging_Exp_info.npy'), allow_pickle=True).item()
    sessions = {}
    for exp_type, dbs in exp_info.items():
        for db in dbs:
            key = session_key(db['mname'], db['datexp'], db['blk'])
            rec = sessions.setdefault(key, {...})
            rec['exp_types'].append(exp_type)
```
```python
for f in sorted(glob.glob(os.path.join(BEH_DIR, 'Beh_*.npy'))):
    B = np.load(f, allow_pickle=True).item()
    for k, beh in B.items():
        base = '_'.join(k.split('_')[:5])
        if base not in sessions: continue
        if base not in beh_by_session: beh_by_session[base] = beh
        sid = np.asarray(beh['stim_id'], dtype=float)
        for w, s in zip(list(beh['UniqWalls']), sid):
            if not np.isnan(s): stim_map[base][str(w)] = int(s)
```
```python
ret = np.load(os.path.join(RET_DIR, '%s_%s_trans.npz' % (info['mname'], info['datexp'])), allow_pickle=True)
iarea = ret['iarea']
...
planes = np.load(path, allow_pickle=True).item()['spks']
```

iii. From CONVERSION_NOTES Steps 1–2: the reference `utils.load_spk`, `utils.load_retino` and `utils.load_exp_beh` define these three reads, and `data_process_script.ipynb` uses `Imaging_Exp_info.npy` as the top-level index. The agent documented that the 142 index entries cover only 89 unique recordings (a session is listed under several experiment types, e.g. before/after learning and test1/2/3) and that there are exactly 89 spk files and 89 retinotopy files, so the unique `(mname, datexp, blk)` triple is the session. The single pass over behaviour files is described as an efficiency measure ("one pass over the 23 Beh_<exp_type>.npy files"), and the merge of `stim_id` across experiment types is justified because the `_swap1`/`_swap2` duplicate keys label complementary subsets of the stimuli.

## 1-b. How are the data split into subjects?

i. The subject is `mname` from the index entry, carried on every session record. In `assemble()` the subject list is the sorted set of `mname` over the retained sessions, and `subject_idx` is each session's index into that list. Result: 19 subjects, 1–8 sessions each, 89 sessions.

ii.
```python
subjects = sorted(set(r['mname'] for r in results))
data = {
    ...
    'subjects': subjects,
    'subject_idx': np.array([subjects.index(r['mname']) for r in results], dtype=np.int64),
```

iii. CONVERSION_NOTES Step 2 lists the 19 mouse names extracted from the index and notes this matches the paper's "89 recordings in 19 mice"; no split has to be derived because the index names the mouse directly.

## 1-c. How are the data split into sessions?

i. A session is one `(mname, datexp, blk)` recording, which also names the spike file. Duplicate listings of the same recording under different experiment types are collapsed by `dict.setdefault` in `load_exp_info()`, and the `_swap1`/`_swap2` behaviour keys are collapsed onto the same base key in `load_behaviour()` (first behaviour dict wins; the stimulus-label maps of all copies are unioned). This yields 89 sessions, all of which are kept.

ii.
```python
def session_key(mname, datexp, blk):
    return '%s_%s_%s' % (mname, datexp, blk)
...
rec = sessions.setdefault(key, {'mname': db['mname'], 'datexp': db['datexp'],
                                'blk': db['blk'], 'exp_types': [], ...})
rec['exp_types'].append(exp_type)
```
```python
base = '_'.join(k.split('_')[:5])
if base not in beh_by_session:
    beh_by_session[base] = beh
```

iii. CONVERSION_NOTES Step 4 documents the discrepancy explicitly: "142 exp_info entries over 23 exp types" vs "89 unique (mname,datexp,blk); 89 spk files; 89 retinotopy files" vs the paper's "89 recordings", resolved by using the 89 unique sessions and merging the `stim_id` labelling across every experiment type in which a session appears. The 10 extra behaviour keys are documented as duplicates of 5 sessions with identical behaviour but complementary `stim_id` labelling for the two leaf1-swap stimuli.

## 1-d. How are the data split into trials?

i. A trial is one corridor traversal. Rather than using `StartFr`/`EndFr`, the code takes the per-frame trial index `ft_trInd` and keeps only frames that pass the reference validity mask, `valid = ft_CorrSpc & (ft_move > 0)` plus finiteness checks on `ft_trInd`, `ft_Pos`, `ft_RunSpeed`. The valid frame indices are stable-sorted by trial index and split at the boundaries, so each trial is the set of *running* frames inside the 4 m textured corridor bearing that trial index. Grey-space frames (positions 40–60 dm) and frames where the VR was not moving are excluded, so the frames of a trial are not necessarily temporally contiguous. Trials keep their own length (median 21 frames, mean 21.6, max 178) — nothing is padded or truncated to a common window.

ii.
```python
valid = (ft_corr & ft_move & np.isfinite(ft_trind) & np.isfinite(ft_pos)
         & np.isfinite(ft_speed))
...
idx_valid = np.where(valid)[0]
tr_valid = ft_trind[idx_valid].astype(int)
order = np.argsort(tr_valid, kind='stable')
tr_sorted, idx_sorted = tr_valid[order], idx_valid[order]
bounds = np.where(np.diff(tr_sorted) != 0)[0] + 1
for fr, tt in zip(np.split(idx_sorted, bounds), np.split(tr_sorted, bounds)):
    tr = int(tt[0])
    if tr < 0 or tr >= ntrials: continue
    fr = np.sort(fr)
```

iii. CONVERSION_NOTES Steps 1/3/5: the mask is copied from `utils.Get_dprime_selective_neuron`, "`VRmove = beh['ft_move'][:nfr]>0`, `isCorridor = beh['ft_CorrSpc'][:nfr]`, `fr_valid = VRmove & isCorridor` (only use activity inside the texture area plus mouse is running)", and from the methods, "We only considered timepoints during running for analysis, which removed time periods when the task mice stopped to collect water rewards" and "we only selected data points inside the 0–4-m region of the corridors where the textures were shown". The agent also argues the mask "makes the 4 x 1 m position bins well defined (positions 0-4 m only)". Trial segmentation by `ft_trInd` rather than by `StartFr` is described in Step 10 Check 5 as robust to trials that are re-entered after a pause ("all valid frames of a trial index are grouped, so no trial is split or duplicated").

## 1-e. How are trials filtered based on quality controls?

i. Three trial-level rejections, applied inside `process_session`: (1) fewer than `MIN_FRAMES_PER_TRIAL = 5` valid frames; (2) the trial's `WallName` has no canonical `stim_id` in the merged map; (3) `SoundTime` or `Trial_start_time` is not finite. Sessions with fewer than 2 surviving trials would be dropped in `assemble()`. In the full run only criterion (2) ever fired: 309 `circle3` trials in 4 sessions (LZ16 ×2, TX124, TX123), leaving 37,801 of 38,110 traversals (99.2%). There is no trial-length upper bound; the running-frame mask already removes the long stationary periods that make some traversals pathologically long (max retained trial 178 frames).

ii.
```python
if len(fr) < MIN_FRAMES_PER_TRIAL:
    n_drop_short += 1
    continue
wname = str(wall[tr])
if wname not in smap:
    n_drop_stim += 1
    continue
if not np.isfinite(t_sound[tr]) or not np.isfinite(t_start[tr]):
    n_drop_cue += 1
    continue
```
```python
results = [r for r in results if len(r['trials']) >= 2]
```

iii. CONVERSION_NOTES Step 3 states "No explicit trial rejection in the reference code; trials are implicitly restricted to frames in the textured corridor while running. Trials without any running frames contribute no data." Step 10 Check 3 justifies the three filters as "drops only trials with <5 valid frames (none occurred), unlabeled stimulus (309 `circle3` trials) or non-finite cue/start time (none)", and Check 5 notes that the `circle3` drop follows from `stim_id` being NaN for that wall in every experiment type. The 2-trial session minimum is justified by the format requirement that "There needs to be at least two trials within each session in order to evaluate the decoder performance".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spks` in `spk/<mname>_<datexp>_<blk>_neural_data.npy`, a list of per-imaging-plane `(n_neurons_plane, n_frames)` float32 arrays of deconvolved fluorescence. The plane arrays are treated as one concatenated matrix in plane order (via cumulative offsets) exactly as `utils.load_spk` does, except that only the selected rows are copied out. `iarea` from `retinotopy/<mname>_<datexp>_trans.npz` supplies each neuron's visual area and is used both for curation and for `brain_region_idx`.

ii.
```python
def load_spk_rows(key, rows):
    planes = np.load(path, allow_pickle=True).item()['spks']
    sizes = np.array([p.shape[0] for p in planes])
    nfr = min(p.shape[1] for p in planes)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    out = np.empty((len(rows), nfr), dtype=np.float32)
    for i in range(len(planes)):
        m = (rows >= offsets[i]) & (rows < offsets[i + 1])
        if m.any():
            out[m] = planes[i][rows[m] - offsets[i], :nfr]
        planes[i] = None
    return out, int(offsets[-1])
```
```python
assert n_spk_neurons == n_neurons_total, 'neuron count mismatch %s: %d vs %d' % (...)
```

iii. CONVERSION_NOTES Step 1 records `utils.load_spk` and `utils.load_retino` and notes the paper's "All our analyses were based on deconvolved fluorescence traces" (Suite2p, tau = 0.75 s), so "no dF/F computation is needed and no further spike deconvolution". The agent verified that the retinotopy neuron count equals the concatenated spk neuron count (50,689 == 50,689 for TX83) and asserts this for every session at runtime.

## 2-b. How is the `neural` data processed?

i. After selecting rows and truncating to `nfr = min(n_spk_frames, len(ft))`, each neuron is z-scored using the mean and standard deviation computed over the *included* (corridor & running) frames only. Neurons with zero or non-finite s.d. are removed. Per trial, the columns of the trial's frames are taken as a contiguous copy and stored as float32. Trials remain variable length; nothing is padded or re-binned.

ii.
```python
nfr = min(spk.shape[1], len(beh['ft']))
spk = spk[:, :nfr]
...
mu = spk[:, valid].mean(axis=1, keepdims=True)
sd = spk[:, valid].std(axis=1, keepdims=True)
keep_neu = (sd[:, 0] > 0) & np.isfinite(sd[:, 0])
spk = (spk - mu) / np.where(sd > 0, sd, 1.0)
spk = spk[keep_neu]
area_idx = area_idx[keep_neu]
...
'neural': np.ascontiguousarray(spk[:, fr]),
...
neural_s.append(t['neural'].astype(np.float32))
```

iii. CONVERSION_NOTES Step 5 Key Decision 3: "Per-neuron z-scoring over the included frames: the reference z-scores deconvolved traces before population analyses (`get_kfold_reward_response`, `stats.zscore(spk, axis=1)`); it prevents a few very active neurons from dominating the decoder's PCA initialisation. Zero-variance neurons are excluded before z-scoring." Step 1 notes the decoder itself does no normalisation ("Decoder does not normalize neural data itself (just linear projection init by SVD). So I should provide reasonably scaled neural data"). No dF/F or deconvolution is applied because the released traces are already deconvolved.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. (1) Anatomical, following the reference: `iarea` is mapped to V1 (8), mHV (0,1,2,9), lHV (5,6), aHV (3,4); neurons with `iarea` in {−1, 7} get index −1 and are dropped. (2) Size: of the surviving visual-cortex neurons, at most `N_NEURONS_PER_SESSION = 1000` are kept per session, allocated proportionally across the four areas (largest-remainder rounding) and drawn without replacement with a per-session seeded `np.random.default_rng`. Zero-variance neurons are then removed. Every session ends up with exactly 1,000 neurons (89,000 total, out of 4.69 M recorded / 4.11 M in visual cortex). No further spike-quality filter is applied.

ii.
```python
AREA_NAMES = ['V1', 'mHV', 'lHV', 'aHV']
AREA_OF_IAREA = {8: 0, 0: 1, 1: 1, 2: 1, 9: 1, 5: 2, 6: 2, 3: 3, 4: 3}

def select_neurons(iarea, n_target, rng):
    area_idx = np.full(len(iarea), -1, dtype=np.int64)
    for ia, a in AREA_OF_IAREA.items():
        area_idx[iarea == ia] = a
    valid = np.where(area_idx >= 0)[0]
    if len(valid) <= n_target:
        return valid, area_idx[valid]
    counts = np.array([(area_idx[valid] == a).sum() for a in range(len(AREA_NAMES))])
    alloc = np.floor(counts / counts.sum() * n_target).astype(int)
    while alloc.sum() < n_target:
        alloc[np.argmax(counts - alloc)] += 1
    chosen = []
    for a in range(len(AREA_NAMES)):
        pool = valid[area_idx[valid] == a]
        if alloc[a] > 0:
            chosen.append(rng.choice(pool, size=min(alloc[a], len(pool)), replace=False))
    sel = np.sort(np.concatenate(chosen))
    return sel, area_idx[sel]

rng = np.random.default_rng(RNG_SEED + (abs(hash(key)) % (2 ** 31)))
rows, area_idx = select_neurons(iarea, N_NEURONS_PER_SESSION, rng)
```

iii. CONVERSION_NOTES Step 1: "No per-neuron quality filtering is performed in the reference beyond Suite2p cell classification; the only neuron curation is anatomical (exclude `iarea==-1` and `iarea==7`)", backed by the reference comment in `Get_density_map` (`idx_neu = (arid!=-1) & (arid != 7) # exclude neurons from outside of visual cortex`). Step 5 Key Decision 2 justifies the subsample: "the full dataset is 405 GB (20k-90k neurons x ~25k frames per session) and cannot be stored in a pickle or loaded by the decoder. 1,000 neurons is far more than the 100 PCs the decoder uses, so decoding is not limited by this." The per-session recorded counts (20,547–89,577) are reported in the metadata and are checked against the paper's stated range.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry (trial start). Alignment is implicit in the frame mask: `ft_CorrSpc` becomes true at corridor entry, so the first retained column of a trial is the first *running* frame inside that trial's corridor, and the last is the last running frame before the grey space. Trials keep their own length; `off_start = 0.0`, `off_end = None`, with a note that trial length varies with running speed. All streams (neural, inputs, outputs) are cut with the same `fr` index array, so they are aligned by construction.

ii.
```python
trials_out.append({
    'frames': fr,
    'neural': np.ascontiguousarray(spk[:, fr]),
    'time_to_cue': (t_sound[tr] - ft[fr]) * SEC_PER_DAY,
    'time_since_start': (ft[fr] - t_start[tr]) * SEC_PER_DAY,
    'pos_dm': ft_pos[fr], 'speed': ft_speed[fr], 'lick': lick_bin[fr], ...})
```
```python
'temporal_alignment_event': 'corridor entry (trial start; beh["Trial_start_time"]/beh["StartFr"])',
'off_start': 0.0,
'off_end': None,
```

iii. CONVERSION_NOTES Step 5: "Trial: one corridor traversal. Aligned to corridor entry (`StartFr` / `Trial_start_time`)." Step 3 records that "Temporal alignment for frame-wise analyses is by neural frame index (`StartFr`, `SoundFr`, `LickFr`, `EndFr` are frame indices into the ft/neural time base, so behaviour and neural data share one clock)". Step 10 Check 6 reports a dataset-wide verification that "The position bin is 0 at the first frame and 3 at the last frame of every trial (means 0.00 and 3.00), confirming the alignment to corridor entry", and that `time_since_trial_start` is strictly increasing within every trial (0 violations).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling. The bin is the native two-photon imaging frame; `time_bin_size` is written as the mean over sessions of the median inter-frame interval, 314.85 ms (3.176 Hz). Per-session `frame_interval_s` is also stored in `session_info`. Trials hold a median of 21 bins (mean 21.6), i.e. ~6.8 s, consistent with 4 m at 60 cm/s.

ii.
```python
'dt': float(np.median(np.diff(ft)) * SEC_PER_DAY),
...
dt_ms = float(np.mean([r['dt'] for r in results]) * 1000.0)
data['metadata'] = { ..., 'time_bin_size': dt_ms, ... }
```

iii. CONVERSION_NOTES Step 5 Key Decision 7: "Time bin = the imaging frame (3.17 Hz, ~315 ms). No re-binning: the reference analyses all frame-wise quantities on this native grid." Step 1 records the notebook's stated `fs = 3.17 Hz` and the measured median dt of 0.3149 s; Step 9 lists 314.85 ms as consistent with both.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. `beh['SoundTime']` (the cue time of each trial, a MATLAB datenum) and `beh['ft']` (the timestamp of every imaging frame, also a datenum). `SoundFr` is not used directly; the code relies on `SoundTime` being the frame-time axis evaluated at the fractional `SoundFr`.

ii.
```python
ft = np.asarray(beh['ft'], dtype=float)[:nfr]
t_sound = np.asarray(beh['SoundTime'], dtype=float)
...
'time_to_cue': (t_sound[tr] - ft[fr]) * SEC_PER_DAY,
```

iii. CONVERSION_NOTES Step 5 maps "`beh['SoundTime']`, `beh['ft']` -> `input[0]` time_to_sound_cue", noting `utils.spk_2_cue` uses `SoundFr` for the same alignment and that the two are the same clock. (Independent check for this report: `SoundTime` equals `np.interp(SoundFr, arange(n), ft)` to machine precision, so this is numerically identical to interpolating `SoundFr` onto the frame-time axis.)

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The datenum difference `SoundTime[trial] − ft[frame]` is converted to seconds by multiplying by 86,400, giving a continuous, time-varying signal that is positive before the cue and negative after it. A second, derived input `sound_cue_onset` is added: a binary time series that is 1 on the first frame of the trial at or after the cue, 0 elsewhere (0 everywhere for the ~0.04% of trials the mouse leaves before the cue).

ii.
```python
SEC_PER_DAY = 24.0 * 3600.0
'time_to_cue': (t_sound[tr] - ft[fr]) * SEC_PER_DAY,
```
```python
cue_onset = np.zeros(T, dtype=np.float32)
after = np.where(t['time_to_cue'] <= 0)[0]
if after.size:
    cue_onset[after[0]] = 1.0
inp = np.stack([t['time_to_cue'].astype(np.float32), cue_onset, ...])
```

iii. CONVERSION_NOTES Step 5 gives the sign convention explicitly ("positive before the cue, negative after"). The binary onset input is justified by the format spec, "If an input is a time such as onset of some stimulus, represent it as a binary time series", and in Step 12 as "a scale-free version of the cue timing" robust to the heavy tails of the signed time. Step 12 also reports an ablation: clipping both time inputs to ±30 s changed sample validation accuracy only marginally (stimulus 0.898→0.906, licking 0.859→0.874, position 0.830→0.833, speed 0.509→0.510), so the unclipped elapsed times were kept.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at exactly the frame indices `fr` used to slice the neural matrix for that trial, so it is on the same grid and the same length by construction.

ii.
```python
'neural': np.ascontiguousarray(spk[:, fr]),
'time_to_cue': (t_sound[tr] - ft[fr]) * SEC_PER_DAY,
```

iii. Step 3 of CONVERSION_NOTES: all streams share one clock (the imaging frame). Step 10 Check 6 verifies dataset-wide that `sound_cue_onset` is exactly the first frame with `time_to_sound_cue <= 0` in all 37,801 trials, and the sample processing plot shows `time_to_sound_cue` crossing zero exactly at the marked cue-onset frame.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The session date string `datexp` from the index entry (`YYYY_MM_DD`), parsed to a `datetime.date`. It is combined with the set of dates for the same mouse to find that mouse's first retained session.

ii.
```python
def parse_date(datexp):
    y, m, d = datexp.split('_')
    return date(int(y), int(m), int(d))

first_day = {}
for r in results:
    d = parse_date(r['datexp'])
    if r['mname'] not in first_day or d < first_day[r['mname']]:
        first_day[r['mname']] = d
```

iii. CONVERSION_NOTES Step 4/5: the agent checked `Imaging_Exp_info.npy` for an explicit field and found "'days' only in 8 entries, so day-of-training must be derived from dates" (the index does carry a `sess#` field, which was not used). The date is the only field that orders every session of every mouse.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. `day = (session_date − first_session_date_of_that_mouse).days`, i.e. calendar days elapsed since the mouse's first imaging session (0 for the first session; range 0–92 across the dataset). The scalar is broadcast across every bin of every trial of the session as a float32 constant.

ii.
```python
day = (parse_date(r['datexp']) - first_day[r['mname']]).days
...
np.full(T, float(day), dtype=np.float32),
```

iii. CONVERSION_NOTES Step 5 variable map: "`datexp` per session -> `input[1]` day_of_training: days elapsed since that mouse's first recording; continuous, per-trial (broadcast over time)". No further justification is given for calendar days rather than a session ordinal.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. `beh['Trial_start_time']` (the corridor-entry time of each trial, a datenum) and `beh['ft']` (frame timestamps).

ii.
```python
t_start = np.asarray(beh['Trial_start_time'], dtype=float)
...
'time_since_start': (ft[fr] - t_start[tr]) * SEC_PER_DAY,
```

iii. CONVERSION_NOTES Step 5: "`beh['ft']`, `beh['Trial_start_time']` -> `input[2]` time_since_trial_start". (Independent check for this report: `Trial_start_time` equals `np.interp(StartFr, arange(n), ft)` to machine precision, i.e. it is the fractional entry frame placed on the frame-time axis.)

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The datenum difference `ft[frame] − Trial_start_time[trial]` scaled by 86,400 to seconds; positive after entry, starting near zero at the first retained frame. It is stored as float32 per bin. Because non-running frames are excluded but wall-clock time is not, the value can jump across a pause: the maximum in the dataset is 1,765 s, and ~27% of trials contain a pause inside the corridor.

ii.
```python
'time_since_start': (ft[fr] - t_start[tr]) * SEC_PER_DAY,
...
t['time_since_start'].astype(np.float32),
```

iii. Step 10 Check 6 reports that `time_since_trial_start` is strictly increasing within every trial (0 violations). Step 12 documents the heavy tail explicitly ("the wall-clock `time_since_trial_start` can reach 1,765 s in 1% of trials") and reports the clipping ablation described in 3-b; the physically correct unclipped elapsed time was kept. Step 10 also records a corrected sanity check: the time-based estimate of cue *position* has a long tail for this reason, and the correct check is against the raw `SoundPos`/`ft_Pos` at `SoundFr` (0.4–3.6 m, matching the paper).

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Evaluated at the same `fr` frame indices as the neural columns of the trial, so identical grid and length.

ii.
```python
'neural': np.ascontiguousarray(spk[:, fr]),
'time_since_start': (ft[fr] - t_start[tr]) * SEC_PER_DAY,
```

iii. Same as 3-c: all streams are indexed by neural frame number, so alignment is by construction; verified by the independent raw-data `np.allclose` spot checks on all five inputs for 5 trials in each of 3 sessions (Step 10 Check 2).

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. `beh['isRew']`, the per-trial flag marking corridors in which water was available.

ii.
```python
is_rew = np.asarray(beh['isRew'], dtype=bool)
...
'is_rew': int(is_rew[tr]),
```

iii. CONVERSION_NOTES Step 5 maps "`beh['isRew']` -> `input[3]` reward_availability: 1 if the trial's corridor is the rewarded one, else 0", referencing `utils.get_cat_id` / `Get_dprime_rewPred_neuron` as the reference uses of the same flag.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast to 0/1 and broadcast as a float32 constant across the trial's bins. No other processing. It is 1 for 11.5–12.3% of all trials, 37.6% of trials in the 28 task (water-restricted) sessions, and 0 everywhere in the unsupervised and naive sessions.

ii.
```python
np.full(T, float(t['is_rew']), dtype=np.float32),
```

iii. CONVERSION_NOTES Step 9 consistency table: "Rewarded corridors ~half of trials in training sessions; `isRew` 0.26-0.51 in task sessions; 37.6% of task trials (11.5% of all trials, because 61/89 sessions have no reward)", consistent with the paper's statement that the unsupervised and naive mice were not water restricted.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. `beh['WallName']` gives the wall texture of each trial; `beh['UniqWalls']` paired with `beh['stim_id']` gives the mapping from wall name to the reference's canonical stimulus index. The mapping is built per session and unioned across every experiment type (and `_swap1`/`_swap2` key) in which the session appears, because `stim_id` is NaN for some walls in some experiment types.

ii.
```python
sid = np.asarray(beh['stim_id'], dtype=float)
for w, s in zip(list(beh['UniqWalls']), sid):
    if not np.isnan(s):
        stim_map[base][str(w)] = int(s)
...
wname = str(wall[tr])
if wname not in smap: n_drop_stim += 1; continue
...
'stim': smap[wname],
```

iii. CONVERSION_NOTES Step 2/5: "`beh['stim_id']` (parallel to `UniqWalls`) maps them onto the paper's 7 canonical stimuli: `0:circle1, 1:circle2, 2:leaf1, 3:leaf2, 4:leaf3, 5:leaf1_swap1, 6:leaf1_swap2` (rock->circle-role, wood/brick->leaf-role), i.e. exactly the pooling the paper describes ('we denote the stimuli as leaf and circle, even though other visual stimuli were also used in some mice (rock and bricks)')". The reference's own docstring `stim_ID = [2, 0]: 'circle1':0, 'circle2':1, 'leaf1':2, 'leaf2':3` is cited as the source of the canonical ids.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The per-trial wall name is looked up in the merged per-session map and stored as an integer 0–6, broadcast across every bin of the trial. `output_values[0]` names the seven classes `['circle1','circle2','leaf1','leaf2','leaf3','leaf1_swap1','leaf1_swap2']`. Trials whose wall has no canonical id anywhere (all `circle3` trials, 309 of them in 4 sessions) are dropped. Full-dataset class fractions: 0.319, 0.060, 0.335, 0.170, 0.059, 0.027, 0.029.

ii.
```python
CANONICAL_STIM = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
                  'leaf1_swap1', 'leaf1_swap2']
...
np.full(T, t['stim'], dtype=np.int64),
```

iii. CONVERSION_NOTES Step 5 Key Decision 4: "Stimulus labels are the reference's role-based canonical ids (`beh['stim_id']`), i.e. the paper's leaf/circle naming, merged over all experiment types in which a session appears so that the swap stimuli are labelled." The trajectory (step 29) records that the agent recognised the labelling is role-based — "0=unrewarded base, 1=its variant2, 2=rewarded base, 3=variant2 of rewarded, 4=variant3, 5/6=swaps, consistent across texture families (circle/leaf, rock/wood)" — and chose it deliberately over a texture-family labelling. Step 10 Check 3 justifies the dropped trials: "`circle3` has no canonical `stim_id` anywhere in the dataset, so no stimulus label can be assigned."

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. `beh['LickFr']`, the (fractional) neural frame number of every lick in the session.

ii.
```python
lick_fr = np.atleast_1d(np.asarray(beh['LickFr'], dtype=float))
```

iii. CONVERSION_NOTES Step 5 maps "`beh['LickFr']` -> `output[1]` licking: binary per frame", referencing `utils.spk_2_firstLick` / `spk_2_cue`, which build lick histograms on frame indices.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A session-length binary vector is created; each lick's frame number is floored to an integer, non-finite values and frames outside `[0, nfr)` are discarded, and the corresponding bins are set to 1. A bin is 1 if at least one lick fell in it. The trial's values are the entries at the trial's retained frames. Because only running frames survive the mask, licks emitted while the mouse was stopped (notably during reward collection) are not represented. Sessions from the unsupervised and naive cohorts have empty `LickFr` and are all-zero. Overall 3.7% of bins are licks (11.7% within the 28 task sessions).

ii.
```python
lick_bin = np.zeros(nfr, dtype=np.int64)
if lick_fr.size:
    li = np.floor(lick_fr[np.isfinite(lick_fr)]).astype(int)
    li = li[(li >= 0) & (li < nfr)]
    lick_bin[li] = 1
...
'lick': lick_bin[fr],
```

iii. CONVERSION_NOTES Step 4 documents the cohort issue: "Licking: `LickFr` empty for unsupervised/naive mice; unsupervised mice were not water restricted and received no reward -> Licking output = 0 for those sessions (mice genuinely do not lick without water); documented as a limitation." Step 10 Check 5 covers the edge cases: "`LickFr` is empty (dtype uint8) for non-water-restricted mice -> handled with `np.atleast_1d` and a finite check; lick frames outside `[0,nfr)` are dropped."

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` indexes the imaging frames, so the binary vector is already on the neural grid; it is indexed with the same `fr` array used for the neural columns.

ii.
```python
'neural': np.ascontiguousarray(spk[:, fr]),
'lick': lick_bin[fr],
```

iii. Step 3: alignment throughout the dataset is by neural frame index. Step 10 Check 2 verified licking values for 5 trials per session against the raw `LickFr` re-binned independently.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. `beh['ft_Pos']`, the position of the mouse in the 6 m VR cycle at every imaging frame, in decimeters (0–40 dm textured corridor, 40–60 dm grey space).

ii.
```python
ft_pos = np.asarray(beh['ft_Pos'], dtype=float)[:nfr]
...
'pos_dm': ft_pos[fr],
```

iii. CONVERSION_NOTES Step 4 resolved the units: "`Corridor_Length=60`, texture 0-40; `ft_Pos` in [0,60) -> `ft_Pos` is in decimeters; texture area = 0-40 dm = 0-4 m", consistent with the paper's "corridors were each 4 m long, with 2 m of grey space".

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Only the units/binning step; the raw per-frame position is carried through the trial slicing unchanged and discretised at assembly time. Because only `ft_CorrSpc` frames are retained, the values span 0–40 dm.

ii.
```python
pos_bin = np.clip(np.digitize(t['pos_dm'], POS_BIN_EDGES_DM), 0, 3)
```

iii. CONVERSION_NOTES Step 5: "`floor(ft_Pos/10)` clipped to 0..3 (ft_Pos is in decimeters; 4 bins x 1 m over the 4 m texture area)", citing the reference's 1 dm position binning in `get_interpPos_spk` as the analogous operation.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Fixed edges at 10, 20 and 30 dm, i.e. four equal 1 m bins `0-1m / 1-2m / 2-3m / 3-4m`, with a clip to [0, 3] as a guard. The resulting dataset-wide distribution is 0.250 / 0.249 / 0.250 / 0.252.

ii.
```python
POS_BIN_EDGES_DM = np.array([10.0, 20.0, 30.0])
...
pos_bin = np.clip(np.digitize(t['pos_dm'], POS_BIN_EDGES_DM), 0, 3)
...
'output_values': [..., ['0-1m', '1-2m', '2-3m', '3-4m'], ...]
'position_bin_edges_m': [0.0, 1.0, 2.0, 3.0, 4.0],
```

iii. Directly from the Decoder Task spec: "Position in corridor discretized into 4 equal-length, 1-m-long spatial bins". Step 10 Check 6 verifies the discretisation dataset-wide: "The position bin is 0 at the first frame and 3 at the last frame of every trial (means 0.00 and 3.00)", and the sample processing plot shows the bin switching at 10/20/30 dm.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` has one value per imaging frame and is indexed by the same `fr` array as the neural columns.

ii.
```python
'neural': np.ascontiguousarray(spk[:, fr]),
'pos_dm': ft_pos[fr],
```

iii. Step 3: everything shares the imaging-frame clock. Verified by the monotonicity check ("position increases monotonically inside a trial and spans ~0.1-3.9 m", Step 10 Check 2) and by the independent `np.allclose` spot checks.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. `beh['ft_RunSpeed']`, the running speed of the mouse (cm/s) at each imaging frame.

ii.
```python
ft_speed = np.asarray(beh['ft_RunSpeed'], dtype=float)[:nfr]
...
'speed': ft_speed[fr],
```

iii. CONVERSION_NOTES Step 5 maps "`beh['ft_RunSpeed']` -> `output[3]` speed bin: global quartiles over all included timepoints -> 4 bins, each 25% of the data".

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The raw per-frame speed is carried through trial slicing. At assembly, the speeds of *all* retained timepoints of *all* sessions are concatenated and the global 25/50/75th percentiles are computed once; these three edges are applied to every session. Because the frame mask already restricts to frames where the VR was moving (running above the 6 cm/s threshold), the zero-speed frames that would otherwise create massive ties are absent, and the resulting marginal is exactly uniform (0.250/0.250/0.250/0.250).

ii.
```python
all_speed = np.concatenate([np.concatenate([t['speed'] for t in r['trials']])
                            for r in results])
speed_edges = np.quantile(all_speed, [0.25, 0.5, 0.75])
print('\nGlobal running-speed quartile edges (cm/s): %s' % np.round(speed_edges, 3))
```

iii. CONVERSION_NOTES Step 5 Key Decision 6: "Speed quartiles are computed globally over all included timepoints of the converted dataset so that each bin holds 25% of the data, as required by the Decoder Task." Step 12 acknowledges the consequence: "the quartile edges are global (12.4, 25.3, 40.8 cm/s), so a large part of the speed label reflects between-session/between-mouse differences in running vigour rather than within-trial dynamics. Per-session distributions therefore vary strongly, which is the intended consequence of the 'each bin = 25% of the data' instruction."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize` against the three global quartile edges, clipped to [0, 3], giving `q1_slowest / q2 / q3 / q4_fastest`. The edges are written to metadata as `speed_bin_edges_cm_per_s`.

ii.
```python
spd_bin = np.clip(np.digitize(t['speed'], speed_edges), 0, 3)
...
'output_values': [..., ['q1_slowest', 'q2', 'q3', 'q4_fastest']],
'speed_bin_edges_cm_per_s': speed_edges.tolist(),
```

iii. Decoder Task spec: "Running speed discretized into 4 bins, each corresponding to 25% of the data". Step 7 reports the sample bins at 0.250 each and the full run at exactly 0.250 each; the `--show-processing` figure overlays the global edges on the per-session speed histogram.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` has one value per imaging frame and is indexed with the same `fr` array as the neural columns.

ii.
```python
'neural': np.ascontiguousarray(spk[:, fr]),
'speed': ft_speed[fr],
```

iii. Step 3: common imaging-frame clock; verified by the raw-data spot checks in Step 10 Check 2 ("output stimulus/licking/position-bin/speed-bin identical for 5 trials per session").

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. (a) The behaviour can run one frame past the imaging, so every stream is truncated to `nfr = min(n_spk_frames, len(ft))`, as the reference does. (b) Frames with non-finite `ft_trInd`, `ft_Pos` or `ft_RunSpeed` are excluded by the validity mask. (c) `LickFr` may be an empty `uint8` array (unsupervised/naive mice) or carry frames beyond the recording — handled with `np.atleast_1d`, a finiteness filter and a range filter. (d) Trial indices outside `[0, ntrials)` are skipped. (e) Trials with non-finite `SoundTime`/`Trial_start_time` are dropped (none occurred). (f) Neurons with zero or non-finite s.d. over the included frames are removed before z-scoring (none occurred). (g) Sessions with fewer than 2 usable trials are dropped (none occurred). (h) The concatenated spk neuron count is asserted equal to the retinotopy neuron count. Individual session failures are not caught — an exception in a worker would abort the pool.

ii.
```python
nfr = min(spk.shape[1], len(beh['ft']))
spk = spk[:, :nfr]
ft = np.asarray(beh['ft'], dtype=float)[:nfr]
...
valid = (ft_corr & ft_move & np.isfinite(ft_trind) & np.isfinite(ft_pos)
         & np.isfinite(ft_speed))
```
```python
li = np.floor(lick_fr[np.isfinite(lick_fr)]).astype(int)
li = li[(li >= 0) & (li < nfr)]
```
```python
if tr < 0 or tr >= ntrials: continue
...
keep_neu = (sd[:, 0] > 0) & np.isfinite(sd[:, 0])
...
results = [r for r in results if len(r['trials']) >= 2]
```

iii. CONVERSION_NOTES Step 10 Check 5 enumerates each of these ("Behaviour arrays can be 1 frame longer than the neural data -> everything truncated to `nfr` (as the reference does). Verified: TX83_2022_08_31_1 has 33,598 behaviour frames vs 33,597 neural frames"), and Step 4 records the truncation as matching the reference's `beh[...][:nfr]` convention.

## 12-a. What are the most time-consuming steps of the code?

i. Reading the 405 GB of spike files. The per-session log lines show load time dominating total time (e.g. `4s load, 4s tot`; `7s load, 8s tot`), and the whole full conversion takes 76 s wall clock with 8 fork workers: 2.2 s to read the 23 behaviour files, 65 s for all 89 sessions, 5.4 s to pickle 3.31 GB. The next costs are the z-scoring pass over the full session matrix and the transfer of the per-session results (≈3.3 GB of neural arrays) back through the multiprocessing pipes.

ii.
```python
t0 = time.time()
... spk, n_spk_neurons = load_spk_rows(key, rows)
t_load = time.time() - t0
...
print('  %-22s neu %5d/%6d ... [%.0fs load, %.0fs tot]' % (..., t_load, out['total_time']))
```
```python
with ctx.Pool(args.nworkers) as pool:
    results = pool.map(process_session, keys, chunksize=1)
```

iii. CONVERSION_NOTES Step 6/7: "Naively loading each session (`utils.load_spk`) would read 405 GB and hold up to 12 GB per session"; the speed-ups listed are "load only the sampled neuron rows", "multiprocessing pool (6-8 workers) ~6x" and "vectorised trial segmentation". The stated claim that "the full (up to 90k x 33k) matrix is never materialised" is only partly right — `np.load(...).item()['spks']` reads and materialises every plane before any row is selected; what is avoided is the extra concatenated copy, not the read.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Three. (1) The per-trial assembly loop in `assemble()` calls `np.digitize` on position and speed, allocates `np.full` constants and stacks arrays once per trial (37,801 iterations); both discretisations could be done once per session over the whole frame axis and then sliced, exactly as the reference does. (2) The per-trial loop in `process_session` that slices each stream separately. (3) The small loops in `select_neurons` (per-area `counts`, the largest-remainder `while`, the per-area `rng.choice`) and the per-plane loop in `load_spk_rows`. All are negligible next to the I/O; the trial *segmentation* itself is already vectorised via `argsort` + `np.split`.

ii.
```python
for t in r['trials']:
    T = len(t['frames'])
    ...
    pos_bin = np.clip(np.digitize(t['pos_dm'], POS_BIN_EDGES_DM), 0, 3)
    spd_bin = np.clip(np.digitize(t['speed'], speed_edges), 0, 3)
    outp = np.stack([...])
```
```python
counts = np.array([(area_idx[valid] == a).sum() for a in range(len(AREA_NAMES))])
while alloc.sum() < n_target:
    alloc[np.argmax(counts - alloc)] += 1
```

iii. CONVERSION_NOTES Step 6 claims "Trial segmentation is vectorised (sort by `ft_trInd`, `np.split` at the boundaries) instead of looping over trials" — true for segmentation, but the notes do not identify the remaining per-trial discretisation loops.

## 12-c. What processing does the code repeat multiple times?

i. (a) The neural array is copied twice: `np.ascontiguousarray(spk[:, fr])` in the worker and then `t['neural'].astype(np.float32)` in `assemble()`, which is a no-op cast on already-float32 data but still allocates a second full copy of the 3.3 GB of neural data. (b) `spk[:, valid]` is materialised twice (once for `mean`, once for `std`), each a large temporary. (c) `np.quantile` over the concatenated speeds requires the per-trial speed arrays to be concatenated once in `assemble()` and again inside `show_processing`. (d) Per-trial `np.digitize` re-does work that is constant per session (see 12-b). (e) Every session's full result dict — including the raw `pos_dm`, `speed`, `lick`, `frames` arrays — is pickled by the worker and unpickled by the parent, duplicating the neural data across the process boundary. (f) `subjects.index(r['mname'])` is a linear scan per session (trivial).

ii.
```python
mu = spk[:, valid].mean(axis=1, keepdims=True)
sd = spk[:, valid].std(axis=1, keepdims=True)
```
```python
neural_s.append(t['neural'].astype(np.float32))
```

iii. Not discussed in CONVERSION_NOTES; the notes only record the speed-ups that were added (row-selective loading, multiprocessing, vectorised segmentation).

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (a) The largest one is inherent to the neuron subsample: all 89 spike files are read in full (405 GB, ~98% of the rows discarded) to keep 1,000 neurons per session. Since the `.npy` files are pickled lists of plane arrays, they cannot be memory-mapped, so this is hard to avoid — but it means the dominant cost buys data that is thrown away. (b) The z-score is computed for every selected neuron over the whole session including frames that are never exported. (c) Each trial dict carries `frames`, `wall`, `trial_index`, `pos_dm` and `speed` through the multiprocessing boundary although only the discretised versions reach the pickle. (d) The redundant `astype(np.float32)` copy noted in 12-c. (e) Storing neural data as float32 rather than float16 doubles the output file (3.31 GB) for a signal the decoder immediately projects to 100 PCs. (f) `sanity_checks()` concatenates every input and output array in memory, and computes an "implied cue position" statistic that the agent itself concluded was invalid. (g) `show_processing` plotting work (only requested in sample mode).

ii.
```python
planes = np.load(path, allow_pickle=True).item()['spks']   # whole file read
out = np.empty((len(rows), nfr), dtype=np.float32)          # 1000 of ~50,000 rows kept
```
```python
inp = np.concatenate([np.concatenate(s, axis=1) for s in data['input']], axis=1)
out = np.concatenate([np.concatenate(s, axis=1) for s in data['output']], axis=1)
cue_pos = (inp[3] + inp[0]) * 0.6
```

iii. CONVERSION_NOTES Step 10 records that the implied-cue-position check was invalid and was replaced by a raw `SoundPos`/`ft_Pos` check; the other items are not discussed. Memory/precision choices are not justified in the notes beyond the subsampling rationale.
