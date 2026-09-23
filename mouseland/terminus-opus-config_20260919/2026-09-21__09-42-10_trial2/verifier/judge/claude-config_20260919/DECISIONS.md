# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything is read from three subfolders of `/app/data`: `beh/` (behaviour), `spk/` (suite2p deconvolved traces, one `<session_id>_neural_data.npy` per session, 405 GB total) and `retinotopy/` (`<mouse>_<date>_trans.npz`, giving each neuron's visual area). `beh/Imaging_Exp_info.npy` is the master index; it is a dict keyed by experiment type, each holding a list of recording entries. The AI de-duplicates those 142 entries down to 89 unique recordings, then runs **two passes**:

- **Pass 1 (behaviour, serial)**: sessions are grouped by `exp_type` so each `Beh_<exp_type>.npy` file (100–430 MB) is loaded exactly once; per-trial frame indices, inputs and raw output values are extracted for every session.
- **Pass 2 (neural, `multiprocessing.Pool`, 6–8 workers)**: each worker loads one session's retinotopy `.npz` and one `spk/*.npy`, and slices out only the selected neuron rows × retained frames.

The behaviour-key convention handles "swap" sessions by appending `stimtype` to the session id, exactly as the raw files are keyed.

ii.
```python
def load_exp_info():
    return np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()

def beh_key(db):
    kn = '%s_%s_%s' % (db['mname'], db['datexp'], db['blk'])
    if 'stimtype' in db:
        kn = kn + '_' + db['stimtype']
    return kn
```
```python
def behaviour_pass(sessions, first_date, verbose=True):
    by_exp = collections.defaultdict(list)
    for i, (key, exp_type, db, ets) in enumerate(sessions):
        by_exp[exp_type].append(i)
    results = [None] * len(sessions)
    for exp_type in sorted(by_exp):
        B = np.load(os.path.join(DATA_ROOT, 'beh', 'Beh_%s.npy' % exp_type),
                    allow_pickle=True).item()
        for i in by_exp[exp_type]:
            key, _, db, ets = sessions[i]
            beh = B[beh_key(db)]
            ...
            results[i] = process_behaviour(beh, day)
        del B
```
```python
def load_spk_rows(mname, datexp, blk, rows, frames):
    fn = os.path.join(DATA_ROOT, 'spk', '%s_%s_%s_neural_data.npy' % (mname, datexp, blk))
    spks = np.load(fn, allow_pickle=True).item()['spks']
    ...
ret = np.load(os.path.join(DATA_ROOT, 'retinotopy', '%s_%s_trans.npz' % (mname, datexp)),
              allow_pickle=True)
iarea = ret['iarea']
```

iii. From CONVERSION_NOTES Step 6: "Behaviour files loaded once per exp_type; sessions grouped by exp_type. One spk read per session; only the selected rows/columns are copied. … 6 worker processes … for the I/O-bound neural pass." Loading the behaviour once per file avoids re-reading the same 100–430 MB file up to 5×; splitting into a cheap behaviour pass and an expensive neural pass lets the global speed quartiles and training days be computed before any spike file is touched.

## 1-b. How are the data split into subjects?

i. The mouse name is the `mname` field of each index entry and is the first element of the `(mname, datexp, blk)` session key. `subjects` is the sorted set of unique mouse names over the sessions being converted (19 for the full run), and `subject_idx` is each session's index into that list.

ii.
```python
def session_key(db):
    return (db['mname'], db['datexp'], str(db['blk']))
```
```python
subjects = sorted({k[0] for k, _, _, _ in sessions})
...
data['subject_idx'].append(subjects.index(key[0]))
...
data['subject_idx'] = np.asarray(data['subject_idx'], dtype=np.int64)
```

iii. The index already names the mouse for every recording, so nothing has to be inferred. CONVERSION_NOTES Step 9 verifies "19 mice / 19 distinct `mname` / 19" against the paper's "89 recordings in 19 mice".

## 1-c. How are the data split into sessions?

i. A session is one unique `(mname, datexp, blk)` triple. `Imaging_Exp_info.npy` lists 142 entries across 23 experiment types, but a given recording appears under up to 5 experiment types; `unique_sessions` keeps the **first** experiment type in which a recording is seen, giving 89 sessions. The AI additionally verified that the duplicate behaviour dicts are identical (matching `ntrials`, `nframes`, `sum(StartFr)`, `sum(SoundFr)`, `nlicks`) before de-duplicating. The set of experiment types a recording appears in is retained in `metadata.session_info` as `exp_types`/`cohort`.

ii.
```python
def unique_sessions(exp_info):
    """One entry per unique recording, keeping the first exp_type it appears in."""
    reps = {}
    exp_types = {}
    for exp_type, dbs in exp_info.items():
        for db in dbs:
            key = session_key(db)
            exp_types.setdefault(key, []).append(exp_type)
            if key not in reps:
                reps[key] = (exp_type, db)
    ordered = sorted(reps.keys())
    return [(k, reps[k][0], reps[k][1], exp_types[k]) for k in ordered]
```

iii. CONVERSION_NOTES Step 4: "Sessions repeated across exp_types are the *same* recording. Verified all repeated entries have identical behaviour (ntrials, nframes, sum(StartFr), sum(SoundFr), nlicks: 0 inconsistencies). Convert each unique session once -> 89 sessions." This is cross-checked against 89 spk files, 89 retinotopy files and the paper's "We performed 89 recordings in 19 mice".

## 1-d. How are the data split into trials?

i. Trials are the `ntrials` traversals the behaviour declares, and each frame is assigned to its trial by the behaviour's own per-frame index `ft_trInd`. Within a trial the AI keeps **only frames that satisfy all three conditions**: `ft_trInd == t`, `ft_CorrSpc` (inside the 4 m texture corridor) and `ft_move > 0` (the VR was advancing, i.e. the mouse was running above the 6 cm/s threshold). Frames with NaN `ft_trInd` are excluded, and the behaviour is truncated to the number of imaged frames. The grouping is vectorised with a stable argsort + `searchsorted` rather than one mask scan per trial. Trials therefore have variable length (median 21 retained frames, 1st–99th percentile 15–37), and the retained frames of a trial are **not contiguous in clock time** when the mouse paused mid-corridor.

ii.
```python
move = beh['ft_move'][:nfr_beh] > 0                  # VR moved => mouse ran > 6 cm/s
corr = beh['ft_CorrSpc'][:nfr_beh].astype(bool)      # inside the 4 m texture corridor
ftr = beh['ft_trInd'][:nfr_beh].astype(float)        # trial index of each frame
...
valid = move & corr & np.isfinite(ftr)
vidx = np.where(valid)[0]
vtr = ftr[vidx].astype(int)
order = np.argsort(vtr, kind='stable')
vidx, vtr = vidx[order], vtr[order]
lo = np.searchsorted(vtr, np.arange(ntrials), side='left')
hi = np.searchsorted(vtr, np.arange(ntrials), side='right')

for t in range(ntrials):
    frames = vidx[lo[t]:hi[t]]
    if len(frames) == 0:
        continue
```

iii. CONVERSION_NOTES Step 5, Key Decision 1: "Only running (VR-moving) corridor frames are kept: matches the paper's explicit curation and the reference `fr_valid` mask; also removes the long stationary periods (reward consumption) that would otherwise dominate the time axis. Consequence: the retained frames are not contiguous in clock time, so `time_since_trial_start` is supplied as the *actual* elapsed time of each retained frame (rather than bin index)." The reference code supports this: `utils.Get_dprime_selective_neuron` uses `fr_valid = (ft_move>0) & isCorridor`, and the methods say "We only considered timepoints during running for analysis, which removed time periods when the task mice stopped to collect water rewards." The AI also notes that because the VR advances at a constant 60 cm/s, this makes trial lengths nearly uniform (median 21 ≈ 4 m / 0.6 m/s / 0.315 s) instead of up to 5,607 frames. It chose `ft_trInd` over `ceil(StartFr)..floor(GrayFr)` because "~3–5% of trials differ by one boundary frame" and `ft_trInd` is "the authoritative labelling used by `Get_coding_direction`".

## 1-e. How are trials filtered based on quality controls?

i. Almost nothing is filtered. Two rules are applied: (a) a trial with **fewer than 5 retained running frames** after truncation to the neural recording is dropped (`MIN_FRAMES_PER_TRIAL = 5`); (b) a session with fewer than 2 usable trials is skipped entirely. In practice **0 of 38,110 trials were dropped**. There is no trial-length outlier filter — the AI relies on the `ft_move` mask to strip the stationary periods that make a trial pathologically long. Residual effect: `time_since_trial_start` still reaches 1,765 s and `time_to_sound_cue` −1,763 s for the trials where a mouse stood in the corridor for minutes.

ii.
```python
MIN_FRAMES_PER_TRIAL = 5    # trials with fewer retained running frames are dropped
...
for tr in trials:
    frames = tr['frames'][tr['frames'] < nfr_spk]
    if len(frames) < MIN_FRAMES_PER_TRIAL:
        continue
```
```python
if len(trials) < 2:
    print('  skipping %s: only %d usable trials' % ('_'.join(key), len(trials)))
    continue
```

iii. CONVERSION_NOTES Step 3: "the reference code uses all trials of a session; frames are curated instead (running + corridor)". Step 5: trials with "< 5 retained frames (too short to be informative / truncated at the end of the recording)" are dropped. Step 9 notes the long-clock-time consequence and defends it: "because only running frames are kept, the *clock* time of a trial keeps running while the mouse stands still, so a few trials span minutes (2.2% of timepoints have time_since_trial_start > 30 s, 0.5% > 120 s). These are genuine elapsed times, not artefacts." The trajectory shows the agent flagged this as a risk ("heavy-tailed clock-time inputs could hurt the linear decoder", steps 59 and 62) but chose not to act on it.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spks` in `spk/<mname>_<datexp>_<blk>_neural_data.npy` — a list of one (neurons × frames) array per imaging plane — concatenated along axis 0 in plane order, exactly as `utils.load_spk` does. The per-neuron visual area comes from `iarea` in `retinotopy/<mname>_<datexp>_trans.npz`, mapped through a verbatim copy of `utils.neu_area_ID`.

ii.
```python
def neu_area_ID(iarea):
    """Copied from /app/code/utils.py (reference code) -- area masks from `iarea`."""
    area_name = ['V1', 'mHV', 'lHV', 'aHV']
    idx = {}
    for ar in area_name:
        if ar == 'V1':
            idx[ar] = iarea == 8
        elif ar == 'mHV':
            idx[ar] = (iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)
        elif ar == 'lHV':
            idx[ar] = (iarea == 5) | (iarea == 6)
        elif ar == 'aHV':
            idx[ar] = (iarea == 3) | (iarea == 4)
    return idx
```
```python
spks = np.load(fn, allow_pickle=True).item()['spks']
nfr = spks[0].shape[1]
frames = frames[frames < nfr]
out = np.empty((len(rows), len(frames)), dtype=np.float32)
offset = 0; filled = 0
for plane in spks:
    n = plane.shape[0]
    sel = rows[(rows >= offset) & (rows < offset + n)] - offset
    if len(sel):
        out[filled:filled + len(sel)] = plane[sel][:, frames]
        filled += len(sel)
    offset += n
assert filled == len(rows), (filled, len(rows))
```

iii. CONVERSION_NOTES Step 5: "deconvolved traces, no further normalisation (paper: 'All our analyses were based on deconvolved fluorescence traces')". The row-wise slicing is described as equivalent to the reference's full concatenation restricted to the selected rows, and Step 10 Check 2 verifies this by exhaustively searching the raw `spks` planes for each converted trace with `np.allclose`.

## 2-b. How is the `neural` data processed?

i. No processing at all beyond selection. No dF/F (the traces are already suite2p-deconvolved, tau = 0.75 s), no z-scoring, no smoothing, no normalisation. The selected rows × retained frames are copied into a contiguous `float32` array per trial. Trials are variable-length and are left that way — nothing is padded or truncated to a common window.

ii.
```python
neural.append(np.ascontiguousarray(act[:, cols], dtype=np.float32))
```

iii. CONVERSION_NOTES Step 5, Key Decision 6: "No z-scoring / dF-F of the neural data: the shipped traces are already suite2p deconvolved; the paper's analyses use them directly (z-scoring appears only inside specific figure analyses, and the decoder does its own projection)."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, one matching the reference and one not:

1. **Area labelling (matches reference)**: a neuron is kept only if its `iarea` maps to V1 (8), mHV (0,1,2,9), lHV (5,6) or aHV (3,4). Neurons with `iarea` in {−1, 7} (12.5% of the 4,691,034 neurons) are dropped. No further quality filter is applied — the shipped `spks` are already the suite2p-curated cells.
2. **Random subsampling (an addition)**: the remaining 4,105,393 area-labelled neurons are subsampled to **at most 2,000 per session**, allocated proportionally across the four areas (with ≥1 per non-empty area), drawn with `np.random.default_rng(2025 + session_index)` and re-sorted into recording order. This keeps **178,000 of 4,105,393 (4.3%)** area-labelled neurons.

ii.
```python
MAX_NEURONS = 2000          # per-session cap on the number of neurons kept
...
def select_neurons(iarea, rng):
    masks = neu_area_ID(iarea)
    per_region = [np.where(masks[r])[0] for r in BRAIN_REGIONS]
    counts = np.array([len(x) for x in per_region])
    total = counts.sum()
    if total > MAX_NEURONS:
        alloc = np.floor(counts / total * MAX_NEURONS).astype(int)
        alloc = np.minimum(alloc, counts)
        alloc[(counts > 0) & (alloc == 0)] = 1
        while alloc.sum() < MAX_NEURONS:
            room = counts - alloc
            if room.max() <= 0:
                break
            alloc[np.argmax(room)] += 1
        while alloc.sum() > MAX_NEURONS:
            alloc[np.argmax(alloc)] -= 1
    else:
        alloc = counts
    keep, region = [], []
    for r in range(len(BRAIN_REGIONS)):
        if alloc[r] <= 0:
            continue
        pick = rng.choice(per_region[r], size=alloc[r], replace=False)
        keep.append(pick); region.append(np.full(alloc[r], r, dtype=np.int64))
    keep = np.concatenate(keep); region = np.concatenate(region)
    order = np.argsort(keep)                      # keep neurons in recording order
    return keep[order], region[order], counts, total
```

iii. CONVERSION_NOTES Step 5, Key Decisions 4 and 5: "Neuron subsampling to 2,000/session (stratified by area, seeded): required for a tractable dataset size; justified by the decoder's own 100-PC / 2,000-neuron projection." and "Neurons without an area label (`iarea` in {−1,7}) are dropped: they are excluded from every area analysis in the reference code and cannot be given a `brain_regions` entry." Step 10 Check 3 adds: "not in the reference (which analyses all neurons); required because the full running-frame matrix would be ~183 GB. The reference decoder projects each session to 100 PCs and random-projects to at most 2,000 neurons for its SVD initialisation, so this is not a loss for decoding." The trajectory shows the machine had 1 TB RAM and 3.4 TB free disk.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry (trial start, `beh['StartFr']`). A trial's neural matrix is exactly the columns of `spks` at that trial's retained running-corridor frames, in temporal order, so the first column is the first running frame at/after corridor entry. Trials keep their own length; nothing is padded or cut to a shared window. `metadata.off_start = 0.0` and `off_end = None`. Because the alignment carrier is the frame index and every behaviour stream is natively on the same frame clock, all streams are aligned by construction.

ii.
```python
t_start = np.interp(beh['StartFr'], frame_idx, t_frame)
...
time_since_start=(t_frame[frames] - t_start[t]).astype(np.float32),
```
```python
temporal_alignment_event='trial start = entry into the virtual-reality corridor (beh["StartFr"])',
off_start=0.0,
off_end=None,
```

iii. CONVERSION_NOTES Step 5, Key Decision 8: "Alignment event = trial start (corridor entry, `StartFr`), as required by the decoder task; `off_start = 0`, `off_end = None` because trials end at corridor exit and therefore have variable duration (median 6.6 s)." Step 4: "Behaviour is already resampled to the neural frame clock (`ft_*`), so alignment = indexing by neural frame."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin is the native imaging frame. No rebinning, resampling, smoothing or position-interpolation is applied. `metadata.time_bin_size` is the median of the per-session median `diff(ft)`, 314.70 ms (3.178 Hz), matching the notebook's fs = 3.17 Hz. Caveat implied by the `ft_move` mask: successive retained columns of a trial are one frame apart only while the mouse ran; when it paused, consecutive stored bins are separated by more than 315 ms (the metadata reports a single nominal bin size, and the real gap is recoverable from `time_since_trial_start`).

ii.
```python
dt=float(np.median(np.diff(t_frame)))
...
dts = np.array([s['dt'] for s in session_info])
...
time_bin_size=float(np.median(dts) * 1000.0),
frame_rate_hz=float(1.0 / np.median(dts)),
```

iii. CONVERSION_NOTES Step 5, Key Decision 7: "Time bin size = the imaging frame interval (median 315.2 ms, fs = 3.17 Hz); no re-binning, so no temporal information is destroyed and alignment is exact by construction (behaviour is natively on the frame clock)." Step 10 Check 3(d): the reference "interpolates to 60 position bins only for the position-tuning figures"; the AI keeps native frames "because the decoder task asks for *time*-resolved data aligned to trial start".

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. `beh['SoundFr']` (the fractional neural-frame index of the sound cue in each trial) and `beh['ft']` (the MATLAB datenum timestamp of every imaging frame).

ii.
```python
t_frame = beh['ft'] * 86400.0                       # datenum (days) -> seconds
frame_idx = np.arange(nfr_beh)
t_cue = np.interp(beh['SoundFr'], frame_idx, t_frame)
```

iii. CONVERSION_NOTES Step 4: "Use `SoundFr` (true cue time) for the 'time to sound cue' input, not the delayed reward proxy [`SoundDelayFr`]." Verified that "`SoundFr` always inside `[StartFr, GrayFr]`" and `SoundPos` spans 4–36 dm, consistent with the paper's uniform 0.5–3.5 m cue position.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. `SoundFr` is fractional, so the cue time is obtained by linear interpolation of the frame-time axis at that fractional index. The input is then `t_cue − t_frame` in seconds for each retained frame of the trial — **positive before the cue, negative after** — stored as `float32`. Value range over the full dataset: [−1763.3, 723.5] s (the extremes come from trials in which the mouse paused).

ii.
```python
time_to_cue=(t_cue[t] - t_frame[frames]).astype(np.float32),
```
```python
inp[0] = tr['time_to_cue']
```

iii. CONVERSION_NOTES Step 5 variable-mapping table: "`t_cue - t_frame` in seconds, using `ft` interpolated at the fractional `SoundFr`; positive before the cue, negative after"; metadata records `input_units[0] = 's (positive before the cue)'`. The interpolation is justified in Step 10 Check 5: "Fractional frame indices (`StartFr`, `SoundFr`, `LickFr` are floats): times are interpolated on the frame clock … as the reference does."

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at `t_frame[frames]`, where `frames` is the *same* array of absolute neural frame indices used to slice the neural matrix for that trial. Alignment is therefore exact by construction and the input has the trial's own length.

ii.
```python
frames = vidx[lo[t]:hi[t]]
...
time_to_cue=(t_cue[t] - t_frame[frames]).astype(np.float32),
```
```python
cols = np.array([frame_pos[f] for f in frames], dtype=np.int64)
neural.append(np.ascontiguousarray(act[:, cols], dtype=np.float32))
```

iii. CONVERSION_NOTES Step 4: "Behaviour is already resampled to the neural frame clock (`ft_*`), so alignment = indexing by neural frame." Step 10 Check 2 verifies the inputs against raw files for the first, sixth and last trial of three sessions.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The `datexp` field of the session key (the recording date) from `Imaging_Exp_info.npy`, compared against the earliest `datexp` for that mouse over **all 89** sessions. `exp_info['days']` was inspected and rejected because it exists for only 8 of 142 entries.

ii.
```python
def first_session_dates(all_sessions):
    """Date of each mouse's first imaging session (over the whole dataset)."""
    first_date = {}
    for key, _, _, _ in all_sessions:
        m, date = key[0], key[1]
        d = datetime.date(*map(int, date.split('_')))
        if m not in first_date or d < first_date[m]:
            first_date[m] = d
    return first_date
```
```python
first_date = first_session_dates(sessions)   # over ALL sessions, not just the sample
```

iii. CONVERSION_NOTES Step 10 Check 3, difference 3: "`day_of_training` = days since the mouse's first imaging session: `exp_info['days']` exists for only 8 of 142 entries, so it cannot be used as a dataset-wide axis; elapsed calendar days is the continuous variable the paper plots training progress against (Fig. 1b, Fig. 5f)."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. `day = (session_date − first_session_date_of_that_mouse).days`, i.e. **calendar** days elapsed, giving a per-trial scalar in [0, 92] that is broadcast across every bin of every trial of the session. It is computed over all 89 sessions so that `--sample` and `--full` produce the same value (this was a bug the agent found and fixed in Step 7).

ii.
```python
day = (datetime.date(*map(int, key[1].split('_'))) - first_date[key[0]]).days
results[i] = process_behaviour(beh, day)
```
```python
day_of_training=np.float32(day_of_training),
...
inp[1] = tr['day_of_training']
```

iii. CONVERSION_NOTES Step 6: "`first_session_dates()` — per-mouse date of the first imaging session, computed over **all 89** sessions so that `day_of_training` is identical in `--sample` and `--full` runs." Step 10 Issues Found: "`day_of_training` computed over the selected sessions only (Step 7) -> fixed to use all 89 sessions".

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. `beh['StartFr']` (fractional neural-frame index of corridor entry for each trial) and `beh['ft']` (frame timestamps).

ii.
```python
t_start = np.interp(beh['StartFr'], frame_idx, t_frame)
```

iii. `StartFr` is the corridor-entry event chosen as the temporal alignment event (CONVERSION_NOTES Step 5, Key Decision 8).

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Linear interpolation of the frame-time axis at the fractional `StartFr`, then `t_frame − t_start` in seconds for each retained frame, stored as `float32`. It is ≥ 0 by construction (asserted in the script). Range over the full dataset: [0, 1765.2] s — large values arise because the clock keeps running through the dropped stationary frames.

ii.
```python
time_since_start=(t_frame[frames] - t_start[t]).astype(np.float32),
...
inp[2] = tr['time_since_start']
...
assert allin[2].min() >= 0, 'time since trial start must be >= 0'
```

iii. CONVERSION_NOTES Step 5 variable-mapping table: "`t_frame - t_start`, `t_start` = `ft` interpolated at fractional `StartFr` … 0 at corridor entry". Step 10 Check 5 handles the one session with a negative `StartFr`: "`np.interp` clamps to the first frame time and the affected frames are simply the ones present, so no negative `time_since_trial_start` occurs (asserted in the script)."

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same as 3-c: it is evaluated at `t_frame[frames]` using the identical `frames` array that indexes the neural columns of that trial, so it is sample-for-sample aligned and has the trial's length.

ii.
```python
frames = vidx[lo[t]:hi[t]]
time_since_start=(t_frame[frames] - t_start[t]).astype(np.float32),
```

iii. All streams are indexed by neural frame number (CONVERSION_NOTES Step 4).

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. `beh['isRew']`, the per-trial flag marking trials run in the rewarded corridor. It is non-zero only in the 28 task (supervised) sessions.

ii.
```python
is_rew = np.asarray(beh['isRew']).astype(bool)
```

iii. CONVERSION_NOTES Step 4: "`isRew` is exactly the set of trials of the rewarded wall; 61 sessions have none … Use `isRew` as the 'reward availability' per-trial input."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Boolean → `float32` 1.0/0.0, stored as a per-trial scalar and broadcast across every bin of the trial. No other processing.

ii.
```python
reward_available=np.float32(1.0 if is_rew[t] else 0.0),
...
inp[3] = tr['reward_available']
```

iii. Directly available from the behaviour; the unsupervised and naive cohorts (61 of 89 sessions) were not water restricted, so 0 is a correct label rather than missing data (CONVERSION_NOTES Step 5, Key Decision 2).

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. `beh['WallName']`, the per-trial name of the corridor wall texture (15 distinct names across the dataset). `beh['UniqWalls']` / `stim_id` were inspected but `WallName` is what is used.

ii.
```python
wall = np.asarray(beh['WallName'])
stim_cat = np.array([STIM_CATEGORIES.index(texture_family(w)) for w in wall])
```

iii. CONVERSION_NOTES Step 4: 15 distinct wall names from 4 texture families; the paper says "we denote the stimuli as leaf and circle, even though other visual stimuli were also used in some mice (rock and bricks)".

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each wall name is reduced to its texture family by splitting at `_` (to drop the `swap1`/`swap2` suffixes), stripping trailing digits, and renaming `wood` → `brick` to match the paper's terminology. This gives four categories `['circle', 'leaf', 'rock', 'brick']`, stored as an index and broadcast across every bin of the trial as `output[0]`. Resulting timepoint fractions: circle 0.311, leaf 0.470, rock 0.085, brick 0.135.

ii.
```python
STIM_CATEGORIES = ['circle', 'leaf', 'rock', 'brick']

def texture_family(wall_name):
    base = wall_name.split('_')[0]
    base = ''.join(ch for ch in base if not ch.isdigit())
    if base == 'wood':
        base = 'brick'
    return base
```
```python
out[0] = tr['stim_cat']
```

iii. CONVERSION_NOTES Step 5, Key Decision 3: "Stimulus category = texture family (circle / leaf / rock / brick), pooling the frozen-crop variants (leaf1/leaf2/leaf3/leaf1_swap…) exactly as the paper pools them for statistics. This makes the label comparable across mice trained on different stimulus pairs, as required by 'Visual stimulus category. e.g. circle, leaf, etc.' (`wood` in the data files is the texture the paper calls `brick`.)"

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. `beh['LickFr']`, the (fractional) neural-frame number of every detected lick in the session.

ii.
```python
lf = np.floor(np.asarray(beh['LickFr'], dtype=float))
lf = lf[np.isfinite(lf)].astype(int)
lf = lf[(lf >= 0) & (lf < nfr_beh)]
lick_bin[lf] = True
```

iii. CONVERSION_NOTES Step 5: licks are binned "by frame with `np.histogram` over integer frame bins" as `utils.spk_2_cue` / `spk_2_firstLick` do.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A session-length boolean vector is built with `True` at every frame containing at least one lick (fractional lick frames are floored; non-finite and out-of-range values are discarded). It is then indexed by the trial's retained frames to give a binary time series, `output[1]`, with `output_values[1] = ['no lick', 'lick']`. Overall 3.7% of timepoints are licking; 61 of 89 sessions are identically 0 because those mice were never water restricted.

ii.
```python
lick_bin = np.zeros(nfr_beh, dtype=bool)
...
lick=lick_bin[frames].astype(np.int64),
...
out[1] = tr['lick']
```

iii. CONVERSION_NOTES Step 5, Key Decision 2: "All 89 sessions are kept even though 61 have no licking: … the unsupervised/naive mice genuinely never licked (they were not water restricted and no reward was delivered), so lick = 0 is a correct label rather than missing data. Balanced accuracy is pooled over timepoints, so the 28 task sessions still determine lick-class-1 recall."

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` is already expressed in neural frame numbers, so the flag vector is on the same grid as `spks`; the trial's values are taken with exactly the same `frames` array used for the neural columns.

ii.
```python
lick=lick_bin[frames].astype(np.int64),
...
neural.append(np.ascontiguousarray(act[:, cols], dtype=np.float32))
```

iii. All streams are indexed by neural frame number (CONVERSION_NOTES Step 4). Step 10 Check 2 verifies `licking` against `floor(LickFr)` membership for the first, sixth and last trial of three sessions.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. `beh['ft_Pos']`, the position within the trial at each imaging frame, in decimetres (0–40 dm across the 4 m texture, continuing to 60 dm through the 2 m grey space; retained corridor frames all lie in [0, 40]).

ii.
```python
pos = beh['ft_Pos'][:nfr_beh].astype(float)          # position within trial, dm
...
pos=pos[frames],
```

iii. CONVERSION_NOTES Step 4: "corridor frames all have `ft_Pos` in [0,40] dm; wall id matches trial stimulus"; the paper's corridor geometry is "4 m long, with 2 m of grey space between corridors".

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The raw dm position of each retained frame is carried through the behaviour pass and discretised at assembly time into `output[2]`, an `int64` time series with `output_values[2] = ['0-1m','1-2m','2-3m','3-4m']`. No smoothing or interpolation.

ii.
```python
TEXTURE_LENGTH_DM = 40.0     # 4 m texture corridor, positions are in decimetres
N_POS_BINS = 4
...
out[2] = np.clip(np.floor(tr['pos'] / (TEXTURE_LENGTH_DM / N_POS_BINS)),
                 0, N_POS_BINS - 1).astype(np.int64)
```

iii. CONVERSION_NOTES Step 4: "Position bins = 4 equal 1-m bins over 0-4 m of the texture area", as mandated by the Decoder Task ("discretized into 4 equal-length, 1-m-long spatial bins").

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. `floor(ft_Pos / 10)` (10 dm = 1 m) clipped to [0, 3], i.e. fixed 1-m edges at 0/1/2/3/4 m — **not** data-driven quantiles. The clip catches frames exactly at the 4 m boundary. The realised distribution is 0.250 / 0.249 / 0.250 / 0.252, which the AI uses as a sanity check (uniform coverage is expected because the VR advances at a constant 60 cm/s while running).

ii.
```python
out[2] = np.clip(np.floor(tr['pos'] / (TEXTURE_LENGTH_DM / N_POS_BINS)),
                 0, N_POS_BINS - 1).astype(np.int64)
...
position_bin_edges_m=[0.0, 1.0, 2.0, 3.0, 4.0],
```

iii. CONVERSION_NOTES Step 10 Check 5: "Position exactly at the 4 m boundary: `np.clip(floor(pos/10), 0, 3)` keeps it in the last bin." Step 5 planned sanity check: "Position-bin distribution approximately uniform (constant VR speed) and covering all 4 bins."

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` gives one value per imaging frame, so it is already on the neural grid; the trial's values are taken with the same `frames` array used for the neural columns, giving the trial's own length.

ii.
```python
pos=pos[frames],
...
neural.append(np.ascontiguousarray(act[:, cols], dtype=np.float32))
```

iii. All streams are indexed by neural frame number (CONVERSION_NOTES Step 4); the `--show-processing` plots overlay the position trace and its bin assignment against the trial boundaries to check for temporal shift.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. `beh['ft_RunSpeed']`, the running speed in cm/s at each imaging frame.

ii.
```python
speed = beh['ft_RunSpeed'][:nfr_beh].astype(float)   # cm/s
...
speed=speed[frames],
```

iii. Directly available per frame; CONVERSION_NOTES Step 5 notes it is the same as `RunFr`.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The raw speeds of all retained frames of all converted sessions are pooled once and the 25th/50th/75th percentiles are taken as global bin edges (full run: 12.422, 25.353, 40.855 cm/s). Every frame is then assigned with `np.digitize`. Because only VR-moving frames are retained, the quartiles are over *running* frames only; negative speeds (backwards ball rotation) are not clipped and simply fall in the lowest bin. The resulting global distribution is exactly 0.250 / 0.250 / 0.250 / 0.250.

ii.
```python
speeds = np.concatenate([tr['speed'] for res in beh_results for tr in res['trials']])
speed_edges = np.percentile(speeds, [25, 50, 75])
print('speed quartile edges (cm/s): %s  [n=%d retained frames]'
      % (np.round(speed_edges, 3).tolist(), len(speeds)), flush=True)
```

iii. CONVERSION_NOTES Step 5, Key Decision 9: "Speed quartile edges are global (computed once from all retained frames of all sessions, behaviour-only pass) so that the 4 bins each hold 25% of the data across the whole dataset and are identical between `--sample` and `--full` runs." Step 10 Check 5: "`ft_RunSpeed` can be negative (backwards ball rotation): the lowest speed quartile simply contains them; no clipping."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize(speed, [p25, p50, p75])` → integer bins 0–3, stored as `output[3]` with `output_values[3] = ['Q1 (slowest 25%)', 'Q2', 'Q3', 'Q4 (fastest 25%)']`. The edges are recorded in `metadata.speed_bin_edges_cm_s`. Because the edges are global rather than per-session, the *per-session* distributions are far from uniform (e.g. DR10_2022_07_12_1 is 0.956 / 0.042 / 0.002 / 0.000), while the pooled distribution is exactly 25% per bin.

ii.
```python
out[3] = np.digitize(tr['speed'], speed_edges).astype(np.int64)
...
speed_bin_edges_cm_s=[float(x) for x in speed_edges],
```

iii. Follows the Decoder Task literally: "Running speed discretized into 4 bins, each corresponding to 25% of the data" — the AI reads "the data" as the whole dataset (CONVERSION_NOTES Step 5, Key Decision 9; Step 9 consistency table: "Speed bins … 0.25 / 0.25 / 0.25 / 0.25 … YES (by construction)").

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` gives one value per imaging frame, so it is already on the neural grid; the trial's values are taken with the same `frames` array used for the neural columns.

ii.
```python
speed=speed[frames],
...
neural.append(np.ascontiguousarray(act[:, cols], dtype=np.float32))
```

iii. All streams are indexed by neural frame number (CONVERSION_NOTES Step 4); the `--show-processing` plots overlay speed, the quartile edges and the resulting bin index per frame.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI enumerates and handles the following (CONVERSION_NOTES Step 10, Check 5):
- **Behaviour longer than the imaging** (1–3 frames): every behaviour stream is sliced `[:nfr_beh]` and the trial frame lists are additionally filtered to `frames < nfr_spk` inside `load_spk_rows` / `convert_session`.
- **Licks outside the recording**: non-finite `LickFr` and values `< 0` or `>= nfr_beh` are dropped.
- **NaN `ft_trInd`** (0.1–2.5% of frames, outside any trial): excluded by `np.isfinite(ftr)`.
- **Fractional `StartFr` / `SoundFr`**: interpolated on the frame-time axis; **negative `StartFr`** (one session) is clamped by `np.interp`, and the script asserts `time_since_trial_start >= 0`.
- **Duplicate session entries** across experiment types: converted once, after verifying the behaviour dicts are identical.
- **Swap sessions**: behaviour key gets the `stimtype` suffix.
- **Sessions with no licks / no rewards**: kept, with genuinely-zero labels.
- **Trials with < 5 retained frames** and **sessions with < 2 usable trials**: skipped (none occurred).
- **Negative running speed**: kept, falls in the lowest quartile.
- **Position at the 4 m boundary**: clipped into the last bin.

ii.
```python
nfr_beh = len(beh['ft'])
move = beh['ft_move'][:nfr_beh] > 0
corr = beh['ft_CorrSpc'][:nfr_beh].astype(bool)
ftr = beh['ft_trInd'][:nfr_beh].astype(float)
...
valid = move & corr & np.isfinite(ftr)
```
```python
lf = np.floor(np.asarray(beh['LickFr'], dtype=float))
lf = lf[np.isfinite(lf)].astype(int)
lf = lf[(lf >= 0) & (lf < nfr_beh)]
```
```python
nfr = spks[0].shape[1]
frames = frames[frames < nfr]
...
frames = tr['frames'][tr['frames'] < nfr_spk]
if len(frames) < MIN_FRAMES_PER_TRIAL:
    continue
```
```python
assert allin[2].min() >= 0, 'time since trial start must be >= 0'
```

iii. CONVERSION_NOTES Step 4: "reference truncates behaviour with `[:nfr]` … behaviour arrays are 1-3 frames longer than spk frames → Truncate behaviour to `nfr`; drop any trial whose frames would exceed `nfr`." The remaining handlers are documented individually in Step 10 Check 5 and were each traced to a concrete observation in the raw data during Steps 2–4.

## 12-a. What are the most time-consuming steps of the code?

i. Reading the 405 GB of `spk/*.npy` files is the dominant cost — `np.load(..., allow_pickle=True)` unpickles the whole per-plane list before any row can be selected. From `conversion_full_out.txt`: behaviour pass 1.4 s total, neural pass 42 s wall clock with 8 workers (1.4–4.6 s per session), pickle write 8.9 s, total 0.9 min. The AI mitigates rather than removes the cost: one read per session, materialising only the selected rows × frames, spread over a worker pool with `maxtasksperchild=1` to release the multi-GB buffers.

ii.
```python
pool = mp.Pool(nproc, maxtasksperchild=1)
it = pool.imap_unordered(convert_session, jobs)
```
```python
spks = np.load(fn, allow_pickle=True).item()['spks']
...
out[filled:filled + len(sel)] = plane[sel][:, frames]
```

iii. CONVERSION_NOTES Step 6: "Naively loading `spk` then concatenating planes into one array doubles peak memory (up to 16 GB/session). Per-trial re-indexing of the spk file would re-read the 2-8 GB file for every trial." Step 9 reports the measured split of the 0.9 min run.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Most of the hot path is already vectorised (the AI replaced a per-trial mask over all ~25k frames with a stable argsort + `searchsorted`). What remains as Python-level loops:
- the dict-based lookup that maps absolute frame numbers to columns of the loaded activity block — `frame_pos` plus a list comprehension per trial; this is `np.searchsorted(kept_frames, frames)` on an already-sorted array;
- the per-trial loops in `process_behaviour` and in the assembly block (one `np.empty` + 8 assignments per trial, ~38k times);
- `stim_cat = np.array([STIM_CATEGORIES.index(texture_family(w)) for w in wall])`, which re-parses the wall string for every trial although a session has at most a handful of distinct names;
- the plane loop in `load_spk_rows` (unavoidable, it is the concatenation).

None of these is material next to the spike-file I/O.

ii.
```python
frame_pos = {f: i for i, f in enumerate(kept_frames)}
...
cols = np.array([frame_pos[f] for f in frames], dtype=np.int64)
```
```python
stim_cat = np.array([STIM_CATEGORIES.index(texture_family(w)) for w in wall])
```
```python
for tr in trials:
    T = len(tr['frames'])
    inp = np.empty((4, T), dtype=np.float32)
    inp[0] = tr['time_to_cue']
    ...
```

iii. The AI documents the vectorisation it *did* do (Step 6: "Vectorised frame selection (`searchsorted` on the sorted per-frame trial index) instead of a per-trial mask over all ~25k frames") but does not identify the remaining loops; the run finished in 0.9 min, so it had no reason to look further.

## 12-c. What processing does the code repeat multiple times?

i. Little. The AI explicitly removed the two big repetitions (re-reading a behaviour file once per session, and re-reading the spike file once per trial). What is still repeated:
- `iarea` is re-read per *session*, so a mouse recorded in two blocks on the same date opens the same `<mouse>_<date>_trans.npz` twice;
- `np.unique(all_frames)` is applied to a concatenation that is already sorted and unique across trials;
- `frames` is filtered against the recording length twice (once inside `load_spk_rows`, once in `convert_session`);
- `first_session_dates` is run over all sessions even when only two are converted (deliberate — it is what makes `day_of_training` identical between `--sample` and `--full`).

ii.
```python
all_frames = np.concatenate([tr['frames'] for tr in trials]) if trials else np.zeros(0, int)
all_frames = np.unique(all_frames)
act, nfr_spk = load_spk_rows(mname, datexp, blk, rows, all_frames)
```
```python
frames = frames[frames < nfr]          # in load_spk_rows
...
frames = tr['frames'][tr['frames'] < nfr_spk]   # again in convert_session
```

iii. Not discussed in CONVERSION_NOTES beyond the speed-ups listed in Step 6; these residual repeats are negligible.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Modest amounts:
- `unique_sessions` collects the full list of experiment types per recording and `cohort_of` derives a cohort label; both only reach `metadata.session_info` and are unused by the decoder.
- `select_neurons` returns `region_counts` and `n_labelled`, used only for logging/metadata.
- `process_behaviour` returns `n_rew_trials` and a per-session `dt` computed from the full frame clock, used only for reporting and for `time_bin_size`.
- Raw `pos` and `speed` floats are stored per trial in the behaviour pass and kept in memory until the assembly step discretises them; only the discretised `int64` versions are saved.
- Outputs are stored as `int64` (8 bytes) for four small-cardinality categorical variables, which is 8× larger than needed, and `neural` is `float32` rather than `float16` — together these inflate the 6.62 GB pickle.
- Under `--show-processing`, `plot_processing` re-derives position bins and speed bins independently of the assembly code.

ii.
```python
out = np.empty((4, T), dtype=np.int64)
```
```python
def cohort_of(db, exp_types):
    ex = db.get('exptype', None)
    ...
```
```python
posbin = np.concatenate([np.clip(np.floor(trials[t]['pos'] / (TEXTURE_LENGTH_DM / N_POS_BINS)),
                                 0, N_POS_BINS - 1) for t in range(ntr)])
```

iii. Not discussed in CONVERSION_NOTES; the extra metadata is deliberate documentation (Step 13 lists `session_info` as part of the user-facing format), and the dtype choices are never revisited because the 6.62 GB output was judged acceptable.
