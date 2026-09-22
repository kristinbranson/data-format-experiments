# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads from the three subdirectories of `/app/data`: `beh/` (behaviour), `spk/` (deconvolved
traces, one file per recording) and `retinotopy/` (visual-area code per neuron). `beh/Imaging_Exp_info.npy`
is treated as the master index: it is a dict keyed by experiment type, each value a list of database
entries (`mname`, `datexp`, `blk`, `stimtype`, `stim_id`, `exptype`, ...). `build_recording_table()`
walks every experiment type and de-duplicates entries into 89 unique recordings keyed by
`mname_datexp_blk`. `load_behaviour()` then reads each of the 23 `Beh_<exp_type>.npy` files exactly
once and attaches the ~16 behaviour arrays each recording needs (reduced by `_extract_beh`), while
merging the canonical `stim_id` map over every experiment type in which the recording appears.
The spike file and the retinotopy file are read per recording — retinotopy in the main process
(`load_iarea`), spikes inside 6 worker processes (`load_spk_planes`), following the reference
`utils.load_spk` filename/concatenation convention.

ii.
```python
def build_recording_table():
    """Return an ordered list of the 89 unique recordings with their merged stimulus map."""
    exp_info = np.load(os.path.join(BEH_DIR, 'Imaging_Exp_info.npy'), allow_pickle=True).item()
    recs = {}
    for exp_type, dbs in exp_info.items():
        for db in dbs:
            key = '%s_%s_%s' % (db['mname'], db['datexp'], db['blk'])
            behkey = key + '_' + db['stimtype'] if 'stimtype' in db else key
            if key not in recs:
                recs[key] = {...'beh_source': (exp_type, behkey), 'stim_map': {}, ...}
```
```python
def load_spk_planes(mname, datexp, blk):
    """utils.load_spk, but returning the individual plane arrays (avoids one full copy)."""
    fn = '%s_%s_%s_neural_data.npy' % (mname, datexp, blk)
    return np.load(os.path.join(SPK_DIR, fn), allow_pickle=True).item()['spks']

def load_iarea(mname, datexp):
    """utils.load_retino, but only the area code of each neuron."""
    d = np.load(os.path.join(RET_DIR, '%s_%s_trans.npz' % (mname, datexp)), allow_pickle=True)
    return d['iarea']
```
```python
    for exp_type in exp_types:
        B = np.load(os.path.join(BEH_DIR, 'Beh_' + exp_type + '.npy'), allow_pickle=True).item()
        ...
        del B
```

iii. CONVERSION_NOTES Step 5/6: "One converted session = one of the **89 recordings**. Behaviour for a
recording is taken from the *first* experiment type that contains it (they are identical);
`stim_id` is merged over **all** experiment types containing it." Behaviour files are read once each
("1.1 s total") and reduced to ~16 arrays per session to bound memory; spk reading is parallelised
because "Reading 405 GB of `spk` files is the only real cost."

## 1-b. How are the data split into subjects?

i. The subject is `mname` from the index entry, carried on every recording record. At assembly the
subject list is the sorted unique set of `mname`, and `subject_idx` is each session's index into it.
Result: 19 mice, 1–8 sessions each (mean 4.68), matching the reference exactly. `day_of_training` is
also computed per mouse by grouping recordings with `by_mouse`.

ii.
```python
    by_mouse = defaultdict(list)
    for r in recs.values():
        by_mouse[r['mname']].append(r)
```
```python
    subjects = sorted({r['mname'] for r in recs})
    data = {
        ...
        'subjects': subjects,
        'subject_idx': np.array([subjects.index(r['mname']) for r in recs], dtype=np.int64),
```

iii. The index already names the mouse for every recording, so nothing has to be inferred. The AI's
consistency table records "Subjects | 19 | — | 19 distinct `mname` | 19 | ✔" and per-subject session
counts 1–8.

## 1-c. How are the data split into sessions?

i. A session is one unique `(mname, datexp, blk)` recording. Because the same recording is listed
under up to five experiment types in `Imaging_Exp_info.npy` (142 database entries), the AI keeps a
dict keyed by `mname_datexp_blk` and only creates a record the first time a key is seen; later
occurrences only contribute their `stim_id` mapping and their experiment-type label. Behaviour arrays
are taken from the first experiment type containing the recording, with an assertion that `ntrials`
agrees across experiment types. 89 sessions result, ordered by `(mname, datexp, blk)`.

ii.
```python
            key = '%s_%s_%s' % (db['mname'], db['datexp'], db['blk'])
            behkey = key + '_' + db['stimtype'] if 'stimtype' in db else key
            if key not in recs:
                recs[key] = {...}
            r = recs[key]
            r['exp_types'].append(exp_type)
            ...
            r['_db_list'] = r.get('_db_list', []) + [(exp_type, behkey, db)]
```
```python
                if 'beh' not in r:
                    r['beh'] = _extract_beh(beh)
                else:
                    assert r['beh']['ntrials'] == int(beh['ntrials']), \
                        'behaviour mismatch across experiment types for %s' % r['key']
```
```python
    order = sorted(recs.values(), key=lambda r: (r['mname'], r['datexp'], r['blk']))
```

iii. Step 9 consistency table: "Recordings (sessions) | 89 | 142 db entries → 89 unique keys | 89 spk
files | 89 | ✔". Step 10 Check 5 lists the edge case "The same recording appears under up to 5
experiment types with different `stim_id` vectors ... de-duplicated to one session; `stim_id` merged
with an assertion that no wall gets two different ids (none does)", and "`swap1`/`swap2` behaviour keys
share one recording ... one session each".

## 1-d. How are the data split into trials?

i. Trials are the corridor traversals the behaviour declares. First a per-frame retention mask is
built — `(ft_move > 0) & ft_CorrSpc & ~isnan(ft_trInd)`, i.e. the reference's `fr_valid` (inside the
0–4 m texture corridor *while the VR/mouse is moving*), restricted to the `nfr` frames actually
imaged, and with a handful of corridor-boundary frames removed. Trials are then the maximal runs of
constant `ft_trInd` among the retained frames, found vectorially from `np.diff(tri)`. Trials are
variable length (min 11, median ~22, max 178 bins). Importantly, non-running frames are deleted from
the *middle* of a trial, so a trial's bins are not temporally contiguous.

ii.
```python
def frame_mask(ft_move, ft_CorrSpc, ft_trInd, ft_Pos, nfr):
    tr = ft_trInd[:nfr]
    keep = (ft_move[:nfr] > 0) & ft_CorrSpc[:nfr] & ~np.isnan(tr)
    idx = np.flatnonzero(keep)
    if len(idx) > 1:
        tri = tr[idx]
        pos = ft_Pos[:nfr][idx]
        wrap = np.flatnonzero((np.diff(pos) < 0) & (np.diff(tri) == 0)) + 1
        if len(wrap):
            keep[idx[wrap]] = False
    return keep
```
```python
    assert np.all(np.diff(tri) >= 0), 'trial indices not monotonic for %s' % rec['key']
    _same = np.diff(tri) == 0
    assert not np.any((np.diff(pos) < 0) & _same), \
        'position decreases within a trial for %s' % rec['key']
    bounds = np.flatnonzero(np.diff(tri)) + 1
    starts = np.concatenate(([0], bounds))
    stops = np.concatenate((bounds, [len(tri)]))
    trial_ids = tri[starts]
```

iii. Step 5 decision 2: "**Timepoints: `(ft_move>0) & ft_CorrSpc`.** Identical to `fr_valid` in
`Get_dprime_selective_neuron` and to the Methods ('0–4 m region', 'excluded the data points in which
the animal was not running'). It also removes the reward-collection stops, exactly as the paper
intends." Methods.txt states: "We only considered timepoints during running for analysis, which removed
time periods when the task mice stopped to collect water rewards." The boundary-frame drop is
justified in Step 10 Check 5: 21 frames in 13 sessions where "VR position has already wrapped to ≈0
while `ft_trInd`/`ft_WallID` still name the previous trial ... both their stimulus label and their
position label are ambiguous."

## 1-e. How are trials filtered based on quality controls?

i. No trial is dropped. All 38,110 declared trials survive, and the AI verified that every trial
retains at least one frame (minimum trial length 11 bins) and that every session has ≥2 trials
(minimum 84). Quality control is instead applied at the *frame* level: frames where the VR was not
moving (mouse below the 6 cm/s threshold), frames outside the behaviour record (`NaN ft_trInd`), and
21 ambiguous corridor-boundary frames are removed. Because a stationary animal contributes no retained
frames, the pathological "parked-mouse" trials that the human reference removes by a 99th-percentile
length rule are instead shrunk to just their running portion.

ii.
```python
    keep = (ft_move[:nfr] > 0) & ft_CorrSpc[:nfr] & ~np.isnan(tr)
```
(there is no trial-level rejection anywhere in `convert_data.py`; every run of constant `ft_trInd`
among retained frames becomes a trial)

iii. Step 5 decision 10: "**No trials are dropped.** Every one of the 38,110 trials has ≥1 retained
frame (verified)." Step 5 decision 1: "**All 89 recordings / 19 mice are kept.** The paper analyses
all of them; no session-level quality criterion exists in code or text." Step 10 Check 3 row (c):
"trial filtering | reference: none | this conversion: none — all 38,110 trials kept | ✔".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<mname>_<datexp>_<blk>_neural_data.npy`, a list of one (neurons × frames) array
per imaging plane, exactly as `utils.load_spk` reads it. The visual-area code of each neuron comes from
`iarea` in `retinotopy/<mname>_<datexp>_trans.npz` (`utils.load_retino`). The number of imaged frames
`nfr` is taken from the spike array, not from the behaviour.

ii.
```python
def load_spk_planes(mname, datexp, blk):
    fn = '%s_%s_%s_neural_data.npy' % (mname, datexp, blk)
    return np.load(os.path.join(SPK_DIR, fn), allow_pickle=True).item()['spks']
```
```python
    planes = load_spk_planes(task['mname'], task['datexp'], task['blk'])
    nfr = planes[0].shape[1]
    n_total = sum(p.shape[0] for p in planes)
    iarea = task['iarea']
    assert len(iarea) == n_total, \
        '%s: retinotopy has %d neurons, spk has %d' % (key, len(iarea), n_total)
```

iii. Step 10 Check 3 row (a): "identical file names and concatenation order; planes are subset before
concatenation purely to save RAM (`p[sel][:, keep]` then concat), which is mathematically the same
rows". Step 10 Check 5: the `nfr` is taken from the spk file because "Behaviour arrays are longer than
the imaging arrays, and not always by exactly 1 ... `TX109_2023_04_18_1`: `len(ft) = 17,373`,
`spk.shape[1] = 17,371`".

## 2-b. How is the `neural` data processed?

i. Two operations beyond selection. (1) The selected rows and retained columns are gathered plane by
plane and concatenated into a float32 (n_neurons × T_session) matrix. (2) Each neuron is **z-scored**
over the retained timepoints of its own session, with the mean and s.d. accumulated in float64 and
zero-variance neurons given s.d. = 1. The session matrix is then sliced into per-trial
(n_neurons × T_trial) arrays. No dF/F or deconvolution is applied — the files already hold Suite2p
deconvolved traces. Output dtype is float32 (the human reference used float16).

ii.
```python
    offs = np.cumsum([0] + [p.shape[0] for p in planes])
    blocks = []
    for pi, p in enumerate(planes):
        lo, hi = offs[pi], offs[pi + 1]
        sel = neu_idx[(neu_idx >= lo) & (neu_idx < hi)] - lo
        if len(sel) == 0:
            continue
        blocks.append(p[sel][:, keep_idx])
    X = np.concatenate(blocks, axis=0).astype(np.float32) if blocks else ...
```
```python
    # z-score each neuron over the retained timepoints of this session.
    mu = X.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    sd = X.std(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    n_dead = int((sd[:, 0] == 0).sum())
    sd[sd == 0] = 1.0
    X -= mu
    X /= sd
```
```python
        results[r['key']]['trials'] = [np.ascontiguousarray(X[:, a:b]) for (a, b) in slices]
```

iii. Step 5 decision 5: "Deconvolved Suite2p amplitudes are in arbitrary, wildly heterogeneous units
(per-neuron s.d. spans 0.5 – 350 within one session; values up to 2849). The decoder initialises its
projection with an **uncentred** SVD and then feeds the projections straight into a linear layer at
lr = 1e-3, so raw amplitudes produce saturated logits. Z-scoring makes the SVD a true PCA ... This is
the one deliberate deviation from the reference, required by 'training a neural decoder'; it is a
per-neuron affine rescaling and therefore changes no relative structure." The float64 accumulation was
added after a sanity check failed at 2.2 × 10⁻³ (Step 10 iteration 2).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. (1) Area filter, identical to the reference: `neu_area_ID` maps `iarea` onto
V1 (8), mHV (0,1,2,9), lHV (5,6), aHV (3,4); neurons with `iarea ∈ {−1, 7}` are dropped. No activity
or SNR threshold is applied (the authors already curated cells with the Suite2p classifier). (2) An
additional **random stratified subsample to at most 2000 neurons per session**, allocated
proportionally over the four areas with largest-remainder rounding and a per-session seed derived from
the session id. This reduces 4,691,034 recorded neurons (4.1 M in visual areas) to 178,000 kept —
about 4% of the visual-area neurons the human reference keeps.

ii.
```python
def neu_area_ID(iarea):
    """utils.neu_area_ID -- map retinotopy area codes onto the four visual areas."""
    idx = {}
    for ar in BRAIN_REGIONS:
        if ar == 'V1':   idx[ar] = iarea == 8
        elif ar == 'mHV': idx[ar] = (iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)
        elif ar == 'lHV': idx[ar] = (iarea == 5) | (iarea == 6)
        elif ar == 'aHV': idx[ar] = (iarea == 3) | (iarea == 4)
    return idx
```
```python
def select_neurons(iarea, n_keep, seed):
    area_idx = neu_area_ID(iarea)
    rng = np.random.default_rng(seed)
    per_area = [np.flatnonzero(area_idx[ar]) for ar in BRAIN_REGIONS]
    n_vis = sum(len(a) for a in per_area)
    if n_vis <= n_keep:
        chosen = [a for a in per_area]
    else:
        exact = np.array([len(a) for a in per_area], dtype=np.float64) / n_vis * n_keep
        take = np.floor(exact).astype(int)
        rem = n_keep - take.sum()
        if rem > 0:
            order = np.argsort(-(exact - take))
            take[order[:rem]] += 1
        take = np.minimum(take, [len(a) for a in per_area])
        chosen = [rng.choice(a, size=k, replace=False) if k < len(a) else a
                  for a, k in zip(per_area, take)]
```

iii. Step 5 decision 4: "4.1 M visual neurons × 821 k timepoints is ~150 GB and would make both the
pickle and the 200-epoch decoder training intractable ... 2000 is exactly `decoder.train_decoder`'s
`svd_max_neurons` default, so the SVD initialisation is computed on the true neurons rather than on a
random Gaussian projection. Stratifying keeps each session's area composition. Seeded → reproducible."
Step 12 Check 3 reports an empirical sweep: 1000 / 2000 / 4000 neurons give stimulus accuracy
0.832 / 0.873 / 0.303 and position 0.844 / 0.894 / 0.564, because "above `svd_max_neurons = 2000`,
`decoder.train_decoder` initialises the projection through a random Gaussian matrix ... the training
loss plateaus around 12 instead of 0.15".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry (trial start). Trials keep their own length; nothing is
padded or truncated to a common window. Each trial's neural matrix is exactly the columns of the
session matrix belonging to that trial's retained frames, the first of which is the first *running*
frame after corridor entry (100% of trials begin in position bin 0). `off_start = 0.0`,
`off_end = None`. Because non-running frames are removed, consecutive columns within a trial can be
separated by more than one imaging frame.

ii.
```python
    bounds = np.flatnonzero(np.diff(tri)) + 1
    starts = np.concatenate(([0], bounds))
    stops = np.concatenate((bounds, [len(tri)]))
```
```python
        results[r['key']]['trials'] = [np.ascontiguousarray(X[:, a:b]) for (a, b) in slices]
```
```python
        'temporal_alignment_event':
            'trial start = entry into the virtual-reality corridor (beh["Trial_start_time"], '
            'the frame at which VR position resets to 0)',
        'off_start': 0.0,
        'off_end': None,
```

iii. Step 10 Check 2 records "100 % of trials start in position bin 0" and "`position_bin` never
decreases within a trial (0 violations over all 38,110 trials)". Metadata `trial_length_note`: "Trials
have variable numbers of time bins (median 22) because only timepoints inside the 0-4 m corridor during
which the virtual reality was moving ... are retained, exactly as in the reference analyses."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning. The native imaging frame is the bin. The AI measures the frame rate per session as
`1 / median(diff(ft) × 86400)` (3.1709–3.1807 Hz across sessions) and writes the median as
`time_bin_size = 314.69 ms`. The paper's own position re-binning (`get_interpPos_spk`, 60 position
bins) is explicitly rejected because position is a decoder output here.

ii.
```python
            'frame_rate_hz': float(1.0 / (np.median(np.diff(r['beh']['ft'])) * SEC_PER_DAY)),
```
```python
    fs_all = np.array([s['frame_rate_hz'] for s in session_info])
    bin_ms = float(1000.0 / np.median(fs_all))
    ...
        'time_bin_size': bin_ms,
        'frame_rate_hz': float(np.median(fs_all)),
        'frame_rate_hz_range': [float(fs_all.min()), float(fs_all.max())],
```

iii. Step 5 decision 3: "**Native imaging frames as time bins** (314.7 ms; frame rate varies by <0.4 %
across sessions, so a single `time_bin_size` is reported). No re-binning — the imaging rate is already
slow, and the paper's only re-binning is onto position, which cannot be used when position is a decoder
target."

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `beh['SoundTime']`, the MATLAB-datenum timestamp of the sound cue on each trial, and
`beh['ft']`, the timestamp of each imaging frame. (`SoundFr`, the frame number the human reference
interpolates, is loaded too but is used only in the diagnostic plots.)

ii.
```python
        'SoundTime': np.asarray(beh['SoundTime'], dtype=np.float64),
        'SoundFr': np.asarray(beh['SoundFr'], dtype=np.float64),
```
```python
    ft = b['ft'][idx]
    t_to_cue = (b['SoundTime'][tri] - ft) * SEC_PER_DAY
```

iii. Step 5 variable-mapping table: "`SoundTime`, `ft` → `input[0]` 'time_to_sound_cue_s':
`(SoundTime[trial] − ft[t]) × 86400`, clipped to ±30 s ... continuous, time-varying; >0 before the cue".

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The per-trial cue timestamp is broadcast to every retained frame of that trial, the frame timestamp
subtracted, and the difference converted from MATLAB days to seconds (×86400). The sign convention is
positive before the cue, negative after. The value is then **clipped to ±30 s**, which affects ~1.2%
of retained samples (raw range −328 … +400 s). Stored as float32.

ii.
```python
SEC_PER_DAY = 86400.0
TIME_CLIP_S = 30.0
```
```python
    t_to_cue = (b['SoundTime'][tri] - ft) * SEC_PER_DAY
    t_to_cue = np.clip(t_to_cue, -TIME_CLIP_S, TIME_CLIP_S)
```
```python
        inp[0] = t_to_cue[s:e]
```

iii. Step 5 decision 8: "**Time inputs clipped to ±30 s.** ... ~2 % of retained samples come from trials
in which the mouse paused for tens of seconds inside the corridor (max 400 s); clipping keeps the
linear decoder well conditioned without discarding any data." Step 9 table marks the changed range as
"by design". Verified in Step 10 Check 2: "`input[0]` equals `clip((SoundTime[trial] − ft) × 86400,
±30)`, max |diff| < 1 × 10⁻⁶ s".

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at exactly the frame timestamps `ft[idx]` of the retained frames, and sliced with
the same `(s, e)` trial boundaries used to slice the neural matrix, so it is bin-for-bin aligned with
`neural`. An assertion checks that neural, input and output have identical lengths on every trial.

ii.
```python
    idx = np.flatnonzero(keep)                      # frame indices, ascending
    tri = b['ft_trInd'][idx].astype(np.int64)
    ft = b['ft'][idx]
```
```python
        for k, (nz, ip, op) in enumerate(zip(trials, r['inputs'], r['outputs'])):
            assert nz.shape[1] == ip.shape[1] == op.shape[1], \
                '%s trial %d length mismatch' % (r['key'], k)
```

iii. Every stream is indexed by imaging-frame number, so alignment is by construction. The
`--show-processing` plot panel (2,1) confirms it empirically: "position at `time_to_cue = 0` equals the
position at `SoundFr` on the identity line for every trial" (Step 12, additional debugging item 2).

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From `datexp`, the recording date string in `Imaging_Exp_info.npy`, grouped by `mname`.

ii.
```python
    by_mouse = defaultdict(list)
    for r in recs.values():
        by_mouse[r['mname']].append(r)
    for mname, rs in by_mouse.items():
        d0 = min(datetime.date(*map(int, r['datexp'].split('_'))) for r in rs)
        for r in rs:
            r['day'] = (datetime.date(*map(int, r['datexp'].split('_'))) - d0).days
```

iii. Step 5 table: "`datexp` → `input[1]` 'day_of_training': days since that mouse's **first**
recording | `Imaging_Exp_info` | continuous, per-trial (broadcast over time)".

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The date string is parsed into a `datetime.date`, the mouse's earliest recording date is found, and
the value is the **calendar-day difference** (not an ordinal session count). The minimum is always
taken over all 89 recordings, before any `--sample` subsetting, so a sample run and a full run agree.
Range 0–92 days. The scalar is broadcast across every bin of every trial of the session.

ii.
```python
            r['day'] = (datetime.date(*map(int, r['datexp'].split('_'))) - d0).days
```
```python
        inp[1] = rec['day']
```

iii. The AI does not argue the point at length; Step 9's consistency table records "day_of_training
range | — | dates in `Imaging_Exp_info` | 0–92 days | [0, 92] | ✔", and the sanity checks confirm
"`input[1]` equals the days between this session's date and the mouse's first session" (Step 10
Check 2). Implicitly, elapsed calendar time is taken as the natural reading of "day of training"
because recordings are sparse and non-consecutive.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `beh['Trial_start_time']`, the timestamp at which the mouse entered the corridor on each trial,
and `beh['ft']`, the frame timestamps.

ii.
```python
        'Trial_start_time': np.asarray(beh['Trial_start_time'], dtype=np.float64),
```
```python
    t_since_start = (ft - b['Trial_start_time'][tri]) * SEC_PER_DAY
```

iii. Step 5 table: "`Trial_start_time`, `ft` → `input[2]` 'time_since_trial_start_s':
`(ft[t] − Trial_start_time[trial]) × 86400`, clipped to 30 s | `beh['Trial_start_time']` |
continuous, time-varying".

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Frame timestamp minus the trial's entry timestamp, converted to seconds (×86400), then clipped to
`[0, 30]` s. This is **wall-clock** elapsed time, so the time the animal spent stationary inside the
corridor is still counted even though those frames were dropped. ~2.2% of samples are clipped (raw
range 0 … 403 s). Stored as float32.

ii.
```python
    t_since_start = (ft - b['Trial_start_time'][tri]) * SEC_PER_DAY
    t_since_start = np.clip(t_since_start, 0.0, TIME_CLIP_S)
```
```python
        inp[2] = t_since_start[s:e]
```

iii. Step 5 decision 8: "Wall-clock time is used (not 'running time'), because that is the true time
since corridor entry and because running-frame counts would make position a deterministic function of
the input." — i.e. counting only retained frames would turn the input into a near-perfect predictor of
the position output and leak the answer.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same as 3-c: computed on `ft[idx]` for the retained frames and sliced with the same trial
boundaries as the neural matrix, with a per-trial length assertion.

ii.
```python
    idx = np.flatnonzero(keep)
    ft = b['ft'][idx]
    ...
        inp[2] = t_since_start[s:e]
```
```python
            assert nz.shape[1] == ip.shape[1] == op.shape[1], \
                '%s trial %d length mismatch' % (r['key'], k)
```

iii. All streams share the imaging-frame index, so alignment is automatic. The processing plot
(panel 2,0) shows all trials starting at position 0 at `time_since_trial_start ≈ 0`.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `beh['WallName']` together with `beh['isRew']`: the rewarded *corridor* is identified as the
wall texture of the first trial with `isRew == True`, and every trial showing that texture is marked
reward-available. Sessions with no rewarded trial get all zeros. `isRew` itself is deliberately **not**
used as the label.

ii.
```python
    if b['isRew'].any():
        rew_stim = wall[b['isRew']][0]              # utils.get_cat_id
        rew_avail = (wall == rew_stim).astype(np.float32)
    else:
        rew_avail = np.zeros(ntr, dtype=np.float32)
```
```python
        inp[3] = rew_avail[t]
```

iii. Step 10 Check 5: "`isRew` means 'reward delivered', not 'rewarded corridor', for active-reward
mice | `TX108_2023_03_25_1`: 34 `wood1` trials with `isRew == False`, and
`isRew.sum() == (~isnan(RewTime)).sum()` exactly | reward availability computed from
`WallName == WallName[isRew][0]`, avoiding leakage of the licking output into the decoder input."
Step 5 decision 6 repeats: "**Reward availability = rewarded *corridor*, not reward delivered**
... Prevents leaking the licking output into the decoder input." The rule itself is copied from
`utils.get_cat_id`.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. A per-trial 0/1 float, broadcast across the bins of the trial. 4,446 of 38,110 trials (11.67%) are
reward-available, all within 28 sessions from 5 task mice (≈48% of trials within a task session).

ii.
```python
        rew_avail = (wall == rew_stim).astype(np.float32)
```
```python
        inp[3] = rew_avail[t]
```
```python
            'n_trials_rewarded_corridor': int(r['diag']['rew_avail'].sum()),
```

iii. Step 9 table: "Rewarded-corridor trials | ~50 % of trials in task sessions | `WallName[isRew][0]` |
4,446 / 38,110 = 11.67 % overall (≈48 % within task sessions) | 4,446 (0.1167) | ✔". Sanity check:
"`input[3]` equals `WallName == WallName[isRew][0]` (0 everywhere for non-task mice)".

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `beh['WallName']` (the texture shown on each trial) combined with the canonical `stim_id`
vector in `Imaging_Exp_info.npy`, which is aligned to `beh['UniqWalls']`. The maps from every
experiment type in which a recording appears are merged into one `stim_map` per recording, with an
assertion that no wall receives two different ids within a recording.

ii.
```python
                for wall, sid in zip(beh['UniqWalls'], db['stim_id']):
                    if not np.isnan(sid):
                        wall = str(wall)
                        prev = r['stim_map'].get(wall)
                        assert prev is None or prev == int(sid), \
                            'stim_id conflict %s %s: %s vs %s' % (r['key'], wall, prev, sid)
                        r['stim_map'][wall] = int(sid)
```
```python
    stim_id = np.array([rec['stim_map'].get(w, STIM_EXTRA.get(w, -1)) for w in wall],
                       dtype=np.int64)
    assert (stim_id >= 0).all(), 'unmapped stimulus in %s: %s' % (...)
```

iii. Step 5 table: "`WallName` + merged `stim_id` → `output[0]` 'stimulus': canonical role id 0–6, plus
7 for `circle3` | `Get_coding_direction` docstring, `get_cat_id`". Step 10 Check 2 verifies
"`ft_WallID` agrees with `WallName[trial]` on **every** retained frame (independent confirmation that
trials are assigned to the right corridor)".

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The wall texture is mapped to the paper's **canonical stimulus role id** rather than to its base
texture: 0 circle1, 1 circle2, 2 leaf1, 3 leaf2, 4 leaf3, 5 leaf1_swap1, 6 leaf1_swap2, with a new
class 7 added for `circle3` when it has no id. Eight classes. Rock/brick/wood mice are folded onto the
circle/leaf roles. The per-trial id is broadcast across all bins of the trial.

ii.
```python
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2', 'circle3']
STIM_EXTRA = {'circle3': 7}
```
```python
    stim_id = np.array([rec['stim_map'].get(w, STIM_EXTRA.get(w, -1)) for w in wall],
                       dtype=np.int64)
```
```python
        out[0] = stim_id[t]
```
```python
            'stimulus': 'canonical stimulus role of the corridor, following the reference '
                        'code (0 circle1, 1 circle2, 2 leaf1, 3 leaf2, 4 leaf3, '
                        '5 leaf1_swap1, 6 leaf1_swap2; 7 circle3 added here). Role 2 is the '
                        'trained/rewarded stimulus; rock/brick mice map onto the same roles.',
```

iii. Step 5 decision 7: "**Stimulus = canonical role id** ... Pools rock/brick mice onto the
leaf/circle roles exactly as the paper does, giving one label set that means the same thing in every
session." Step 10 Check 5 for `circle3`: "`circle3` has no canonical `stim_id` in any experiment type |
309 trials in 4 sessions | new class id 7 rather than discarding the trials". Step 10 Check 3 row (h)
marks this as "✔ + one added class".

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `beh['LickFr']`, the (fractional) imaging-frame number of every detected lick in the session.

ii.
```python
        'LickFr': np.asarray(beh['LickFr'], dtype=np.float64),
```
```python
    lick_fr = b['LickFr']
```

iii. Step 5 table: "`LickFr` → `output[1]` 'licking': `1` for frames `floor(LickFr)`, else 0 |
`spk_2_firstLick` (`beh['LickFr'].astype(int)`) | binary, time-varying" — i.e. the event-frame
convention is copied from the reference code.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Non-finite lick frames are discarded, licks outside `[0, nfr)` are discarded (behaviour can run past
the imaging), the remainder are truncated to integer frames (`astype(int)`, matching the reference),
and a boolean per-frame flag is set. A bin is 1 if at least one lick fell in it. Overall 3.7% of bins
are licks; licks occur only in the 28 task sessions.

ii.
```python
    # licking: the reference indexes event frames with `.astype(int)` (floor)
    lick_fr = b['LickFr']
    lick_fr = lick_fr[np.isfinite(lick_fr)]
    lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nfr)].astype(np.int64)
    lick_frame = np.zeros(nfr, dtype=bool)
    lick_frame[lick_fr] = True
    licking = lick_frame[idx].astype(np.int64)
```
```python
        out[1] = licking[s:e]
```

iii. Sanity check (Step 10 Check 2): "`output[1]` equals the frames indexed by `floor(LickFr)` exactly
(1777 / 897 / 0 lick frames)" and "Licks occur in exactly the 28 task sessions". Step 12 Check 1
explains the licking accuracy ceiling: "licking is *physically absent* from 61 of the 89 sessions (the
unsupervised and naive mice were never water restricted, so `LickFr` is empty — verified directly from
the raw `Beh_*.npy` files)".

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` indexes imaging frames, so the flag lives on the neural grid already. It is subset with the
same retained-frame index `idx` and sliced with the same trial boundaries as the neural matrix.

ii.
```python
    licking = lick_frame[idx].astype(np.int64)
    ...
        out[1] = licking[s:e]
```
```python
        results[r['key']]['trials'] = [np.ascontiguousarray(X[:, a:b]) for (a, b) in slices]
```

iii. All streams are indexed by frame number. The diagnostic plot panel (3,0) shows the lick raster
against corridor position with the sound-cue marker per trial, confirming licks cluster after the cue
in rewarded corridors.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `beh['ft_Pos']`, the virtual-reality position at each imaging frame, in decimetres
(0–40 dm across the 4 m texture; `beh['Texture_Length'] == 40`).

ii.
```python
        'ft_Pos': np.asarray(beh['ft_Pos'], dtype=np.float64),
        'Texture_Length': float(beh['Texture_Length']),
```
```python
    pos = b['ft_Pos'][idx]
```

iii. Step 5 table: "`ft_Pos` → `output[2]` 'position_bin': `floor(ft_Pos/10)` clipped to 0–3 → 4 bins of
1 m over 0–4 m | corridor geometry (`Texture_Length=40` dm)". Step 9 table records
"`ft_CorrSpc == (ft_Pos<40)`" as verified in the data.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is taken at the retained frames and divided by the bin width in decimetres; no smoothing or
interpolation. Because retained frames are inside `ft_CorrSpc`, position stays in 0–40 dm.

ii.
```python
CORRIDOR_DM = 40.0
N_POS_BINS = 4                       # 4 equal-length 1-m bins
POS_BIN_DM = CORRIDOR_DM / N_POS_BINS
```
```python
    pos = b['ft_Pos'][idx]
    pos_bin = np.clip((pos / POS_BIN_DM).astype(np.int64), 0, N_POS_BINS - 1)
```

iii. Step 10 Check 2: "`output[2] == clip(floor(ft_Pos/10), 0, 3)` exactly"; "`position_bin` never
decreases within a trial (0 violations over all 38,110 trials)"; "100 % of trials start in position
bin 0". Resulting distribution [0.250, 0.249, 0.250, 0.252].

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four equal-length 1 m bins with edges at 0, 10, 20, 30, 40 dm — exactly the "4 equal-length,
1-m-long spatial bins" the decoder task specifies. `astype(int)` truncation implements the floor (all
retained positions are ≥ 0), and the clip guards the top edge at exactly 40 dm.

ii.
```python
    pos_bin = np.clip((pos / POS_BIN_DM).astype(np.int64), 0, N_POS_BINS - 1)
```
```python
        'output_values': [
            ...
            ['0-1m', '1-2m', '2-3m', '3-4m'],
```
```python
        'position_bin_edges_m': [0.0, 1.0, 2.0, 3.0, 4.0],
```

iii. Direct implementation of the Decoder Task specification; the geometry is confirmed from
`beh['Texture_Length'] == 40` dm and the paper's "corridors were each 4 m long".

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` gives one value per imaging frame, so it is already on the neural grid; it is subset with
the same retained-frame index and sliced with the same trial boundaries.

ii.
```python
    idx = np.flatnonzero(keep)
    pos = b['ft_Pos'][idx]
    ...
        out[2] = pos_bin[s:e]
```

iii. The `--show-processing` panel (0,0) overlays the retained frames on the raw `ft_Pos` trace and
panel (1,0) shows the position→bin mapping with the bin edges, to "visually convince the user" there is
no temporal misalignment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `beh['ft_RunSpeed']`, the running speed in cm/s at each imaging frame. The AI explicitly
checked and rejected the alternative `RunFr`.

ii.
```python
        'ft_RunSpeed': np.asarray(beh['ft_RunSpeed'], dtype=np.float64),
```
```python
    speed = b['ft_RunSpeed'][idx]
```

iii. Step 10 Check 5: "`RunFr` is *not* equal to `ft_RunSpeed` despite the notebook's comment | ratio
0.3226, r = 0.9997 | `ft_RunSpeed` used (the one in cm s⁻¹)".

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speeds at all retained timepoints of **all 89 sessions** are pooled and the 25/50/75th percentiles
are computed once; every timepoint is then assigned a bin by `searchsorted` against those global edges.
Because non-running frames were already removed, the pooled distribution has few zeros (edges 12.42 /
25.35 / 40.85 cm/s, range −19.17 … 163.49).

ii.
```python
    all_speed = np.concatenate([r['beh']['ft_RunSpeed'][:r['nfr']][r['keep']] for r in recs])
    speed_edges = np.percentile(all_speed, [25, 50, 75])
```
```python
    speed = b['ft_RunSpeed'][idx]
    speed_bin = np.searchsorted(speed_edges, speed, side='right').astype(np.int64)
    speed_bin = np.clip(speed_bin, 0, N_SPEED_BINS - 1)
```
```python
        out[3] = speed_bin[s:e]
```

iii. Step 5 decision 9: "**Running-speed bins from global quartiles** — the spec says each bin should
hold 25 % of the data, which is only exactly true if the quantiles are computed over the pooled
dataset." Step 10 Check 2 confirms "Running-speed bins each hold exactly 25.0 % of timepoints".

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four quartile bins defined by the three global percentile edges, with `side='right'` so a value
equal to an edge falls into the lower bin, and a clip at 3 for the maximum. Names `speed_q1..q4`;
the edges are recorded in the metadata. The resulting global distribution is exactly
[0.25, 0.25, 0.25, 0.25]; per-session distributions are however far from uniform (one session is
0.956 / 0.042 / 0.002 / 0.000 across the four bins).

ii.
```python
N_SPEED_BINS = 4                     # quartiles of the pooled running-speed distribution
```
```python
    speed_bin = np.searchsorted(speed_edges, speed, side='right').astype(np.int64)
    speed_bin = np.clip(speed_bin, 0, N_SPEED_BINS - 1)
```
```python
        'running_speed_bin_edges_cm_s': [float(x) for x in speed_edges],
        ...
            'running_speed_bin': 'running speed quartile, bin edges from the pooled '
                                 'distribution over every retained timepoint',
```

iii. As above (Step 5 decision 9). Step 12 adds "adjacent quartiles differ by only a few cm s⁻¹
(edges 12.4 / 25.4 / 40.9 cm s⁻¹), so a large share of the errors are off-by-one-bin", explaining the
0.5695 validation accuracy.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` has one value per imaging frame; it is subset with the same retained-frame index and
sliced with the same trial boundaries as the neural matrix.

ii.
```python
    idx = np.flatnonzero(keep)
    speed = b['ft_RunSpeed'][idx]
    ...
        out[3] = speed_bin[s:e]
```
```python
    all_speed = np.concatenate([r['beh']['ft_RunSpeed'][:r['nfr']][r['keep']] for r in recs])
```

iii. Frame-indexed, so alignment is automatic; the diagnostic panel (0,1) overlays the retained frames
and the global quartile edges on the raw speed trace.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Six issues are handled explicitly. (1) Behaviour arrays are longer than the imaging arrays, and not
always by exactly one frame, so every stream is truncated to `nfr = spk.shape[1]` taken from the spike
file, and the main process asserts its mask matches the worker's. (2) `ft_trInd` is NaN outside the
behaviour record (0.12% of frames in one session) → excluded. (3) 21 corridor-boundary frames in 13
sessions where the VR position has wrapped but `ft_trInd`/`ft_WallID` still name the previous trial →
dropped, with an assertion that position never decreases within a trial. (4) Licks with non-finite
frame numbers or frames ≥ `nfr` → dropped. (5) Zero-variance neurons → s.d. forced to 1 (count reported;
0 in all sessions). (6) `circle3` without a canonical stimulus id → assigned a new class rather than
discarding the trials. Several hard assertions (`len(iarea) == n_total`, `ntrials` agreement across
experiment types, no `stim_id` conflict, monotone trial index, per-trial length agreement) abort the run
rather than silently producing wrong data.

ii.
```python
    nfr = planes[0].shape[1]
    ...
    assert len(iarea) == n_total, \
        '%s: retinotopy has %d neurons, spk has %d' % (key, len(iarea), n_total)
```
```python
        assert r['keep'].sum() == results[r['key']]['X'].shape[1], \
            '%s: frame mask disagrees with worker' % r['key']
```
```python
    keep = (ft_move[:nfr] > 0) & ft_CorrSpc[:nfr] & ~np.isnan(tr)
    ...
        wrap = np.flatnonzero((np.diff(pos) < 0) & (np.diff(tri) == 0)) + 1
        if len(wrap):
            keep[idx[wrap]] = False
```
```python
    lick_fr = lick_fr[np.isfinite(lick_fr)]
    lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nfr)].astype(np.int64)
```
```python
    n_dead = int((sd[:, 0] == 0).sum())
    sd[sd == 0] = 1.0
```

iii. Step 10 Check 5 tabulates each edge case with the evidence that found it and the resolution, and
the "Issues Found and Resolved" log records three iterations: the `nfr` source bug (trial slices were
computed from `len(ft)−1` while the worker masked with the spk `nfr`), the float32 z-score precision
bug (max |diff| 2.2 × 10⁻³ → 4.5 × 10⁻⁶ after accumulating in float64), and the corridor-boundary frames
(found by the "position_bin never decreases within a trial" sanity check).

## 12-a. What are the most time-consuming steps of the code?

i. Reading the 405 GB of `spk/*.npy` files, which dominates everything else. The AI measured it and
parallelised it across 6 worker processes, reaching ~5.8 GB/s aggregate versus ~1.2 GB/s
single-threaded; the neural stage takes 37.6 s of the 52.7 s total full conversion. The remaining
stages are behaviour loading (1.4 s), speed quartiles, per-trial construction, and the 9.2 s pickle
write of the 6.62 GB output.

ii.
```python
N_WORKERS = int(os.environ.get('CONVERT_WORKERS', '6'))
```
```python
    with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(tasks))) as ex:
        for i, res in enumerate(ex.map(process_neural, tasks)):
```
```python
    for pi, p in enumerate(planes):
        lo, hi = offs[pi], offs[pi + 1]
        sel = neu_idx[(neu_idx >= lo) & (neu_idx < hi)] - lo
        if len(sel) == 0:
            continue
        blocks.append(p[sel][:, keep_idx])
```

iii. Step 6: "Reading 405 GB of `spk` files is the only real cost. Single-threaded `np.load` runs at
~1.2 GB/s." Speedups listed: "6 worker processes reading in parallel → measured **5.8 GB/s** aggregate
... so the whole 405 GB is read in ~75 s"; "Only the 2000 selected rows and the retained columns are
materialised (`p[sel][:, keep_idx]` per plane), so peak RAM per worker is the mmap'd file plus ~0.2 GB
instead of two full copies."

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Very little is left. Trial boundaries are found vectorially from `np.diff(tri)` rather than by
scanning the frame index once per trial, and all per-frame quantities (times, position bins, speed
bins, lick flags) are computed on whole-session arrays before slicing. The residual Python loops are:
the ~38,000-iteration per-trial loop in `build_trial_variables` that allocates and fills one (4, T)
input and one (4, T) output array per trial (could be `np.split` on pre-stacked session-wide arrays);
`subjects.index(r['mname'])` inside a list comprehension (O(n_sessions × n_subjects)); the
`for r in recs: for (et, behkey, db) in r['_db_list']` scan inside `load_behaviour`, which is
O(n_exp_types × n_recordings) instead of grouping recordings by experiment type first; and the
per-trial loops inside `plot_processing`. All are negligible against the 405 GB of I/O.

ii.
```python
    for s, e, t in zip(starts, stops, trial_ids):
        T = e - s
        inp = np.empty((4, T), dtype=np.float32)
        inp[0] = t_to_cue[s:e]
        ...
        inputs.append(inp)
        outputs.append(out)
```
```python
        'subject_idx': np.array([subjects.index(r['mname']) for r in recs], dtype=np.int64),
```
```python
    for exp_type in exp_types:
        B = np.load(...)
        for r in recs:
            for (et, behkey, db) in r['_db_list']:
                if et != exp_type:
                    continue
```

iii. Step 6 "Code inefficiencies identified" names only the I/O cost and the plane-concatenation memory
blow-up; the AI does not flag the per-trial assembly loop, presumably because the whole conversion runs
in 52.7 s.

## 12-c. What processing does the code repeat multiple times?

i. The retained-frame mask is computed twice per session: once inside the worker (`task['keep_fn'](nfr)`)
and once again in the main process (`compute_frame_selection`), with an assertion that the two agree.
This is a deliberate cross-check rather than an oversight, and it is cheap. `neu_area_ID` is likewise
called once per session inside `select_neurons` and the area membership recomputed for the
`n_visual_neurons` statistic. In `plot_processing`, `np.interp(SoundFr, arange(nfr), ft_Pos)` is
recomputed inside two separate per-trial loops. Behaviour files are *not* re-read — each of the 23
`Beh_*.npy` is loaded exactly once.

ii.
```python
    keep = task['keep_fn'](nfr)        # in the worker
```
```python
    for r in recs:
        r['nfr'] = results[r['key']]['nfr']
        r['keep'] = compute_frame_selection(r, r['nfr'])   # again in the main process
        assert r['keep'].sum() == results[r['key']]['X'].shape[1], \
            '%s: frame mask disagrees with worker' % r['key']
```
```python
        'n_visual_neurons': int((iarea != -1).sum() - (iarea == 7).sum()),
```

iii. Step 6: "Neural work is done **before** any trial slicing, so the true `nfr` from the spk file
drives every mask; the main process asserts that its own mask matches the worker's." Step 10 iteration 1
explains why this cross-check exists: an earlier version derived the mask from `len(ft)−1` and
mis-sliced trials.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Mostly diagnostics that are computed unconditionally. (1) `build_trial_variables` always builds and
returns the `diag` dict — nine full-session arrays (`idx`, `tri`, `pos`, `speed`, `licking`,
`t_since_start`, `t_to_cue`, `pos_bin`, `speed_bin`) plus the trial boundaries — and every session's
copy is held in `r['diag']` for the whole run, although only two sessions are ever plotted and only
`rew_avail.sum()` is read afterwards. (2) `raw_stats` (pre-z-score mean and max) is computed in every
worker and never used. (3) `n_visual_neurons`, `t_load`, `t_total` are computed per session for logging
only. (4) `r['_db_list']` and the full `exp_info` return value are kept after the recording table is
built. (5) `rew_avail` is materialised as a length-`ntrials` float32 array although only single elements
are indexed. (6) `SoundFr` is extracted for every session but used only in the optional plots.
None of this is large enough to matter at 52.7 s total.

ii.
```python
    diag = {'idx': idx, 'tri': tri, 'pos': pos, 'speed': speed, 'licking': licking,
            't_since_start': t_since_start, 't_to_cue': t_to_cue,
            'pos_bin': pos_bin, 'speed_bin': speed_bin, 'trial_ids': trial_ids,
            'starts': starts, 'stops': stops, 'stim_id': stim_id, 'rew_avail': rew_avail}
    return inputs, outputs, trial_ids, slices, diag
```
```python
        r['slices'], r['diag'] = slices, diag
```
```python
    raw_stats = {'mean': float(X.mean()) if X.size else 0.0,
                 'max': float(X.max()) if X.size else 0.0}
```

iii. Not discussed in CONVERSION_NOTES; the AI's efficiency discussion (Step 6, Step 7 "Run Time
Estimates") focuses entirely on I/O throughput and peak RAM, both of which it did optimise
(`X` is released after slicing: `results[r['key']]['X'] = None`).
