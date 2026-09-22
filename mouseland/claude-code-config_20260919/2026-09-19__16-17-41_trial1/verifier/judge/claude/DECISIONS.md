# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything is read from the three subdirectories of `/app/data`: `beh/` (behaviour),
`spk/` (deconvolved calcium traces) and `retinotopy/` (visual-area label per neuron).
`beh/Imaging_Exp_info.npy` is the master index; it is read first and its 141
`(exp_type, entry)` records are collapsed to the 89 unique physical recordings keyed by
`mname_datexp_blk` (`build_session_index`). The 23 `Beh_<exp_type>.npy` files are then read
in one pass (`load_behaviour`), keeping only the 13 fields actually needed so the ~5 GB of
behaviour is not held in memory; a behaviour dict key is mapped back to its physical
recording by taking the first five underscore-separated tokens. `stim_id` is merged over
*all* experiment-type entries of a recording (a wall that is NaN in one experiment type is
labelled in another), with an assertion that the merged labels never conflict. Neural data
and retinotopy are read per session in a second pass, with the next session's `spks` file
prefetched on a background thread.

ii.
```python
exp_info = np.load(os.path.join(ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=1).item()
sessions = {}
for exp_type, db in exp_info.items():
    for ndb in db:
        key = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
        beh_key = key + ('_' + ndb['stimtype'] if 'stimtype' in ndb else '')
        rec = sessions.setdefault(key, dict(key=key, mname=ndb['mname'], ...))
```
```python
for exp_type in sorted(set(sum([r['exp_types'] for r in sessions.values()], []))):
    B = np.load(os.path.join(ROOT, 'beh', 'Beh_%s.npy' % exp_type), allow_pickle=1).item()
    for bk, b in B.items():
        key = '_'.join(bk.split('_')[:5])
        ...
        for w, s in zip(walls, sid):
            if not np.isnan(s):
                prev = stim_map[key].get(str(w))
                assert prev is None or prev == int(s), ...
        if key in beh:
            continue
        rec = {f: np.asarray(b[f]) for f in BEH_FRAME_FIELDS}
```
```python
def load_spk_blocks(mname, datexp, blk):
    fn = os.path.join(ROOT, 'spk', '%s_%s_%s_neural_data.npy' % (mname, datexp, blk))
    spks = np.load(fn, allow_pickle=True).item()['spks']
    ...
def load_iarea(mname, datexp):
    fn = os.path.join(ROOT, 'retinotopy', '%s_%s_trans.npz' % (mname, datexp))
```

iii. "Deduplicate to the 89 **physical** recordings; verified all duplicate `beh` entries of
the same recording are byte-identical in `ntrials`, `len(ft)` and `UniqWalls`" (Step 4). The
merge of `stim_id` across experiment types is justified because "up to 4 walls [are] `NaN` in
a single exp_type" — merging recovers the label of the swap stimuli, which would otherwise be
masked. Behaviour is loaded once per file rather than once per session to avoid re-reading
the 5 GB of `Beh_*.npy`.

## 1-b. How are the data split into subjects?

i. The subject is the mouse name `mname` taken straight from the index entry and carried on
each session record. The subject list is built in the order the (sorted) session keys are
processed, and `subject_idx` holds each session's index into that list. Result: 19 mice over
89 sessions, matching the paper's "89 recordings in 19 mice".

ii.
```python
if r['mname'] not in subjects:
    subjects.append(r['mname'])
subject_idx.append(subjects.index(r['mname']))
```
```python
data['subjects'] = subjects
data['subject_idx'] = np.array(subject_idx, dtype=np.int64)
```

iii. No derivation is needed — the index already names the mouse. Sanity check **S3**:
"89 sessions / 19 mice, matching '89 recordings in 19 mice'".

## 1-c. How are the data split into sessions?

i. A session is one physical recording = one mouse, one date, one block
(`mname_datexp_blk`), which is also the name of the `spk` file. The 141 entries of
`Imaging_Exp_info.npy` are deduplicated onto that key, giving 89 sessions. All experiment-type
entries of a recording are remembered (`exp_types`, `beh_keys`) so that the `stim_id` label
vectors can be merged, but the behaviour arrays are taken from only the first entry seen
(the duplicates were verified identical).

ii.
```python
key = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
rec = sessions.setdefault(key, dict(key=key, mname=..., exp_types=[], beh_keys=[], ...))
rec['exp_types'].append(exp_type)
```
```python
key = '_'.join(bk.split('_')[:5])
if key not in sessions:
    continue
...
if key in beh:
    continue          # behaviour of this recording already stored
```

iii. Step 4: `exp_info` has 141 `(exp_type, session)` entries but there are only "89 unique
`mname_datexp_blk`, 89 spk files, 89 retinotopy files, 19 mice", and the paper says
"89 recordings in 19 mice". The duplicates are the same recording listed under several
experiment types (e.g. `swap1`/`swap2`, `test`/`train`), so they are collapsed. All 89
recordings are kept; no session is excluded (Step 5, decision 8).

## 1-d. How are the data split into trials?

i. A trial is one traversal of the 4 m texture corridor. The frames belonging to trial *t*
are those the behaviour labels with that trial index, that are inside the texture area, and
at which the virtual reality is moving (i.e. the mouse is running):
`(ft_trInd == t) & ft_CorrSpc & (ft_move > 0)`, all truncated to the number of frames actually
present in the neural recording (`spk.shape[1]`). Frames with NaN or out-of-range `ft_trInd`
are marked invalid. The grouping of frames into trials is vectorised with a stable
`argsort`/`bincount` rather than one scan per trial. Trials keep their own variable length
(11–178 kept frames; median ≈ 21).

ii.
```python
corr = b['ft_CorrSpc'][:nfr].astype(bool)
move = b['ft_move'][:nfr] > 0
valid = corr & move & np.isfinite(tr)
tr_int = np.where(valid, np.nan_to_num(tr, nan=-1), -1).astype(np.int64)
tr_int[(tr_int < 0) | (tr_int >= ntrials)] = -1

order   = np.argsort(tr_int, kind='stable')     # group frames by trial, time-ordered
counts  = np.bincount(tr_int[tr_int >= 0], minlength=ntrials)
ordered = order[int(np.sum(tr_int < 0)):]       # drop the -1 group (invalid frames)
for t in range(ntrials):
    n = counts[t]
    if n == 0:
        continue
    fi = ordered[ptr:ptr + n]; ptr += n
    ...
    fi = np.sort(fi)
```

iii. "Frames kept = `ft_CorrSpc & (ft_move>0)`, truncated to `spk.shape[1]` — exactly the
reference's `fr_valid` in `utils.Get_dprime_selective_neuron` ('only use activity inside the
texture area plus mouse is running (VR moving)')" and the paper's Methods: "We only considered
timepoints during running for analysis, which removed time periods when the task mice stopped
to collect water rewards." The AI adds that restricting to the 0–4 m texture area "is *also
required* by the decoder spec, whose position output is four 1-m bins covering exactly the 4-m
corridor (the 2-m grey space has no bin)". It verified that the kept frames "run contiguously
from `StartFr[t]` to `GrayFr[t]` with monotonically increasing `ft_Pos` from ~0 dm to ~40 dm"
(sanity check S4: 0/348 non-monotonic trials in the spot-checked session).

## 1-e. How are trials filtered based on quality controls?

i. Only one trial-level exclusion is applied: a trial whose wall texture has no canonical
`stim_id` after the cross-experiment-type merge is dropped. That is 309 trials (0.81 %), all
`circle3`, in 4 sessions. 37,801 of the 38,110 raw trials survive. No filtering is done on
trial length, trial duration, running behaviour, or licking; no session and no mouse is
excluded. The AI explicitly checked and concluded that no trial needed to be dropped for
having too few frames ("every one of the 38,110 raw trials has ≥ 11 valid frames, median 21").
One session (`DR10_2022_07_12`) whose `ft_RunSpeed` trace it diagnosed as corrupted was
deliberately kept.

ii.
```python
if stim_of_trial[t] < 0:                       # wall without a canonical stim id
    continue
```
```python
for t in range(ntrials):
    n = counts[t]
    if n == 0:
        continue                               # no valid frame in the neural recording
```

iii. Step 10 Check 3: "reference has no explicit trial filter; it selects stimuli via
`stim_id` and never touches walls with `stim_id = NaN`" → "drop trials whose wall has no
canonical `stim_id` (309 `circle3` trials) … ✅ equivalent". Step 10 Check 5 for the corrupted
speed session: "The session is **kept**: its neural, position, stimulus and lick data are
unaffected, it is part of the paper's n = 9 unsupervised cohort, and it contributes 0.6 % of
all timepoints. No ad-hoc correction is invented." The AI's argument for not needing a
length/duration filter is that the `ft_move > 0` mask already removes the stationary periods
that make trials pathologically long.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<mname>_<datexp>_<blk>_neural_data.npy`, which is a list of one
(neurons × frames) array per imaging plane. The blocks are *not* concatenated (to avoid a full
extra copy); they are indexed in place in plane order, which is the order the retinotopy
indexes them. The visual area of each neuron comes from `iarea` in
`retinotopy/<mname>_<datexp>_trans.npz`, and drives both the neuron filter and
`brain_region_idx`.

ii.
```python
def load_spk_blocks(mname, datexp, blk):
    fn = os.path.join(ROOT, 'spk', '%s_%s_%s_neural_data.npy' % (mname, datexp, blk))
    spks = np.load(fn, allow_pickle=True).item()['spks']
    ...
    return spks

def load_iarea(mname, datexp):
    fn = os.path.join(ROOT, 'retinotopy', '%s_%s_trans.npz' % (mname, datexp))
    with np.load(fn, allow_pickle=True) as d:
        return d['iarea']
```
```python
nneu_all = sum(x.shape[0] for x in blocks)
assert len(iarea) == nneu_all, (k, len(iarea), nneu_all)
```

iii. Step 10 Check 3: "`load_spk`: `np.concatenate(np.load(...).item()['spks'], 0)` →
`load_spk_blocks` + `extract_neural` operate on the same blocks in the same order without
materialising the concatenation — ✅ identical values (verified by Check 2)". Not
concatenating saves "−8 GB peak, −1.5 s/session". The assertion that `len(iarea)` equals the
total ROI count is the check that the plane order matches.

## 2-b. How is the `neural` data processed?

i. The released Suite2p non-negative deconvolved traces are used as-is except for one step:
each neuron is **z-scored over all frames of its session** (mean and s.d. computed over the
complete recording, not just the frames that are kept), then the kept frame columns are
gathered and split per trial. Zero-variance neurons get `sd = 1` so the z-score is 0 rather
than NaN. The mean/variance are computed in one pass with `sum` + `einsum`, the work is
chunked over a 16-thread pool, and the result is stored as **float32**
(150.8 GB for the full dataset). No dF/F and no deconvolution is computed; no padding is
applied, trials keep their own length.

ii.
```python
def work(task):
    blk, a, bnd, km, o = task
    sub = blk[a:bnd]
    g = sub[:, frame_idx][km]                 # column gather first: small temporary
    if zscore:
        nf = sub.shape[1]
        s1 = sub.sum(axis=1, dtype=np.float64)
        s2 = np.einsum('ij,ij->i', sub, sub, dtype=np.float64)
        mu = s1 / nf
        sd = np.sqrt(np.maximum(s2 / nf - mu * mu, 0.0))
        sd[sd == 0] = 1.0
        g -= mu[km, None].astype(np.float32)
        g /= sd[km, None].astype(np.float32)
    out[o:o + g.shape[0]] = g
```
```python
trials.append(np.ascontiguousarray(neural[:, a:bnd]))
```

iii. Step 5, decision 4: "Deconvolved amplitudes are in arbitrary fluorescence units whose
scale differs by ~10× between sessions and by orders of magnitude between neurons; the
reference itself z-scores (`stats.zscore(spk, axis=1)`) for its population analyses. Without
it the learned 100-d projection (which gets only 200 optimiser steps) is dominated by a
handful of very bright ROIs. Verified empirically in Step 7/8 (raw vs z-scored comparison)."
Step 10 Check 3 records this as "✅ reference-supported". No dF/F is needed because "the
release contains … Suite2p ROI detection + cell classification + neuropil correction +
non-negative deconvolution — already applied in the released `spks`" and the paper states
"All our analyses were based on deconvolved fluorescence traces".

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron filter is anatomical: a neuron is kept if its retinotopy area code `iarea`
is not −1 (unassigned) and not 7 (outside the four visual areas), i.e. it belongs to
V1 (`iarea == 8`), mHV (`0,1,2,9`), lHV (`5,6`) or aHV (`3,4`). This keeps 4,105,393 of
4,691,034 ROIs (87.5 %) — exactly the same total as the human reference. No further per-cell
quality metric is applied.

ii.
```python
def neu_area_idx(iarea):
    reg = np.full(len(iarea), -1, dtype=np.int64)
    reg[iarea == 8] = 0                                   # V1
    reg[np.isin(iarea, [0, 1, 2, 9])] = 1                 # medial HVAs
    reg[np.isin(iarea, [5, 6])] = 2                       # lateral HVAs
    reg[np.isin(iarea, [3, 4])] = 3                       # anterior HVAs
    keep = reg >= 0                                       # == (iarea!=-1)&(iarea!=7)
    return keep, reg[keep]
```

iii. Step 3: "`iarea != -1 & iarea != 7` — 'exclude neurons from outside of visual cortex'
(`utils.Get_density_map`); every area-resolved analysis uses only V1/mHV/lHV/aHV. No further
per-cell quality metric exists in the released data." Sanity check S1: "neuron counts per
session reproduce the paper's 20,547 – 89,577 range exactly", and Step 10 Check 2 compares
`brain_region_idx` element-wise against a freshly recomputed `neu_area_ID` in 5 sessions
(PASS).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry (trial start). Each trial's array starts at the first
kept frame after `Trial_start_time` / `StartFr` and ends at the last kept frame inside the
4 m texture area. Trials are left at their natural, variable length — nothing is cut to a
common window and nothing is padded. `off_start = 0.0` and `off_end = None`.
The neural columns and every input/output row are indexed with the *same* `frame_idx`, so all
streams share one frame grid by construction. (Because non-running frames are removed, the
frames inside a trial are not always consecutive imaging frames, even though they are always
in increasing time order.)

ii.
```python
'temporal_alignment_event':
    'trial start = corridor entry (beh["Trial_start_time"] / beh["StartFr"]); '
    'timepoints are the native two-photon imaging frames',
'off_start': 0.0,
'off_end': None,
```
```python
neural = extract_neural(blocks, keep, ti['frame_idx'], zscore=not args.no_zscore)
for i in range(len(ti['inputs'])):
    a, bnd = ti['bounds'][i], ti['bounds'][i + 1]
    trials.append(np.ascontiguousarray(neural[:, a:bnd]))
```

iii. Step 3: "The task specification asks for alignment to **trial start (corridor entry)**,
which is exactly `StartFr`." Step 12 validates the alignment two ways: "`corr(ft_move>0,
ft_RunSpeed)` peaks at **lag 0** in all 89 recordings" and "the population PSTH around
corridor entry is negative before the event and peaks at **+1 to +2 frames** (0.3–0.6 s) after
it — the expected GCaMP6s + deconvolution latency. Activity *follows* the visual event, so
there is no sign error or off-by-one."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native two-photon imaging frame is the time bin; no rebinning, no resampling and no
interpolation is applied. `time_bin_size` is measured from the data as the mean across
sessions of the per-session median inter-frame interval, giving **314.85 ms** (≈ 3.18 Hz).

ii.
```python
frame_interval_s=float(np.median(np.diff(b['ft'])) * SEC_PER_DAY)
...
dt = float(np.mean([s['frame_interval_s'] for s in session_info]))
data['metadata'] = {..., 'time_bin_size': dt * 1000.0, 'sampling_rate_hz': 1.0 / dt, ...}
```

iii. Step 5, decision 1: "Keep the native imaging frame as the time bin (314.7 ms) rather than
re-binning. The decoder needs 'time since trial start' and 'time to sound cue' as time-varying
inputs, and the reference never re-bins in time. Re-binning to a rounder number would only blur
3.18 Hz calcium data." Step 10 Check 3 records the reference's 60 *position* bins
(`get_interpPos_spk`) as the one deliberate departure: "the decoder task requires time-varying
signals plus 'time since trial start' and 'time to sound cue', which position binning would
destroy. The reference's own time-domain analyses also use raw frames."

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `beh['SoundTime']`, the MATLAB datenum timestamp of the sound cue on each trial, and
`beh['ft']`, the datenum timestamp of every imaging frame. (`SoundTime` is numerically
identical to interpolating the fractional cue frame `SoundFr` onto `ft`.)

ii.
```python
tcue = np.asarray(b['SoundTime'], dtype=float)
...
ftt = ft[fi]
inp[0] = (tcue[t] - ftt) * SEC_PER_DAY         # + before the sound cue, - after
```

iii. Step 5 variable-mapping table: "`beh['SoundTime']`, `beh['ft']` → `input[0]`
`time_to_sound_cue_s`; `(SoundTime[t] − ft[frame]) × 86400` s; **+ before the cue, − after**;
continuous, time-varying". Step 10 Check 3: "reference uses `SoundFr`/`SoundTime` for cue
alignment (`spk_2_cue`)". Sanity check S5 validates the cue variable against the paper:
"Sound-cue positions ≈ uniform on 0.5–3.5 m (93.3 % inside, p1–p99 = 0.53–3.72 m)".

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The two datenums are subtracted and multiplied by 86,400 to give seconds, signed so that
the value is **positive before** the cue and negative after it. It is stored as a continuous
float32 time series, one value per kept frame, not binarised.

ii.
```python
SEC_PER_DAY = 86400.0
...
inp = np.empty((4, T), dtype=np.float32)
inp[0] = (tcue[t] - ftt) * SEC_PER_DAY
```

iii. Step 5, decision 9: "**`time_to_sound_cue` is signed and continuous** (positive before the
cue). The decoder spec explicitly calls it 'continuous, time-varying', so it is not binarised."
Step 10 Check 2 re-derives it from `SoundTime`/`ft` outside the conversion code for 5 sessions
× 12 trials and compares with `np.allclose` (PASS).

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at `ft[fi]`, where `fi` is exactly the same frame index array used to gather
the neural columns of that trial, so it is aligned frame-for-frame with the neural data and has
the same length.

ii.
```python
fi  = np.sort(fi)
ftt = ft[fi]
inp[0] = (tcue[t] - ftt) * SEC_PER_DAY
...
frame_idx.append(fi)
...
neural = extract_neural(blocks, keep, ti['frame_idx'], ...)
```

iii. Every stream in this dataset lives on the imaging-frame grid, so using one `frame_idx`
for all of them guarantees alignment. Step 12: "behaviour↔behaviour: `corr(ft_move>0,
ft_RunSpeed)` peaks at **lag 0** in all 89 recordings … so the streams the conversion joins are
frame-synchronous".

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the session's calendar date `datexp` (in the index entry / session key) together with
the set of all dates of that mouse. It is the number of *calendar days* between this session
and that mouse's earliest imaging session; the per-mouse first day is computed over all 89
recordings, so the value is the same in `--sample` and `--full` mode. Range across the
dataset: 0–92.

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
...
day = {k: datenum(sessions[k]['datexp']) - first_day[sessions[k]['mname']] for k in keys_all}
```

iii. Step 5, decision 6: "**`day_of_training` = days since that mouse's first imaging session.**
The release contains no absolute training-day counter (`sess#`/`days` in `exp_info` are
experiment-type specific and inconsistent across duplicates), while `datexp` is unambiguous,
continuous, and directly tracks learning progression within a mouse."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Dates are converted to ordinals, the mouse's minimum is subtracted, and the resulting
integer is broadcast as a constant across every bin of every trial of that session (it is a
per-trial/per-session quantity stored in time-varying form so all four inputs share one
`(4, T)` array).

ii.
```python
inp[1] = day_of_training
```
```python
'day_of_training': 'days since this mouse\'s first imaging session',
```

iii. Same as 4-a. The count is over all 89 recordings so that a sample run and a full run give
identical values.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `beh['Trial_start_time']`, the datenum of corridor entry on each trial, and `beh['ft']`,
the datenum of every imaging frame. (`Trial_start_time` is numerically identical to
interpolating the fractional entry frame `StartFr` onto `ft`.)

ii.
```python
tstart = np.asarray(b['Trial_start_time'], dtype=float)
...
inp[2] = (ftt - tstart[t]) * SEC_PER_DAY       # time since corridor entry
```

iii. Step 3: "`Trial_start_time` / `StartFr` = corridor entry … The task specification asks for
alignment to **trial start (corridor entry)**, which is exactly `StartFr`." Using the
timestamp rather than the frame number also sidesteps the one trial with a negative `StartFr`
(Step 10 Check 5: "frames are selected by `ft_trInd`, not by `StartFr`, so the negative value
is never used as an index; `time_since_trial_start` stays ≥ 0 (dataset min = 0.0)").

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Subtraction of the two datenums, × 86,400 to get seconds, stored as a continuous float32
time series (positive after entry). Because only running frames are kept, the value is *real
elapsed time*, which can exceed the traversal's running time when the mouse pauses (dataset
maximum 1765 s).

ii.
```python
inp[2] = (ftt - tstart[t]) * SEC_PER_DAY
```
```python
'time_since_trial_start_s': 'seconds since corridor entry (real elapsed time)',
```
```python
'off_end_note':
    'Trials end when the mouse leaves the 4 m texture area. Duration is variable '
    'because non-running frames are excluded and mice pause: median 3.8 s, p99 61 s.',
```

iii. The metadata note above is the AI's own justification/flag for the variable duration. The
value is verified against the raw files in Step 10 Check 2 ("`time_to_sound_cue`,
`time_since_trial_start`, `reward_available` recomputed from `SoundTime`, `Trial_start_time`,
`ft`, `isRew` — PASS ×5").

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is evaluated at `ft[fi]` with the same `fi` used for the neural columns of that trial, so
it is frame-for-frame aligned and the same length.

ii.
```python
ftt = ft[fi]
inp[2] = (ftt - tstart[t]) * SEC_PER_DAY
...
neural = extract_neural(blocks, keep, ti['frame_idx'], ...)
trials.append(np.ascontiguousarray(neural[:, a:bnd]))
```

iii. As for 3-c — all streams are indexed with one shared frame-index array.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `beh['isRew']`, the per-trial flag marking the trials run in the rewarded corridor.

ii.
```python
isrew = np.asarray(b['isRew']).astype(np.float32)
...
inp[3] = isrew[t]
```

iii. Step 5 variable-mapping table: "`beh['isRew']` → `input[3]` `reward_available`; 1 if the
trial's corridor is the rewarded one, else 0; reference functions `utils.lickCount`,
`get_kfold_reward_response`; discrete, per-trial (broadcast)."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. None beyond a cast to float and broadcasting the per-trial value across the trial's bins.
It is 1 only in the 28 task-cohort sessions and 0 everywhere in the unsupervised, grating and
naive cohorts.

ii.
```python
inp[3] = isrew[t]
```
```python
n_rewarded_trials=int(np.sum([inp[3, 0] > 0 for inp in ti['inputs']])),
```

iii. Sanity check S6: "Rewarded trials and licking occur in exactly the same 28 sessions, which
are exactly the 28 task-cohort sessions (5 mice)." This cross-validates `isRew` against the
independent lick stream and against the cohort labels in `exp_info`.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `beh['WallName']` (the texture shown on each trial), looked up in a per-recording map
built from `beh['UniqWalls']` and `beh['stim_id']` merged over every experiment-type entry of
that recording. `stim_id` is the paper's canonical stimulus index 0–6, which the AI names
`circle1, circle2, leaf1, leaf2, leaf3, leaf1_swap1, leaf1_swap2`. `TrialStim` is not used.

ii.
```python
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3', 'leaf1_swap1', 'leaf1_swap2']
...
stim_of_trial = np.full(ntrials, -1, dtype=np.int64)
smap = rec['stim_map']
for t in range(ntrials):
    stim_of_trial[t] = smap.get(str(b['WallName'][t]), -1)
```
```python
for w, s in zip(walls, sid):
    if not np.isnan(s):
        prev = stim_map[key].get(str(w))
        assert prev is None or prev == int(s), 'conflicting stim_id for %s / %s' % (key, w)
        stim_map[key][str(w)] = int(s)
```

iii. Step 4: "Stimulus names — `UniqWalls` are mouse-specific (`rock1/wood1…`), 15 distinct raw
names; the paper says 'we denote the stimuli as leaf and circle, even though … rock and
bricks'. Resolution: Use the canonical `stim_id` (0–6) so labels are comparable across mice;
rock1→circle1, rock2→circle2, wood1→leaf1, wood2→leaf2, wood5→leaf3." The merge over
experiment types is needed because "up to 4 walls [are] `NaN` in a single exp_type", which is
how the two swap stimuli get labelled at all.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. `WallName` → canonical id 0…6 through the merged map, broadcast across all bins of the
trial and stored as `output[0]`. Trials whose wall never receives a canonical id (all
`circle3`, 309 trials in 4 sessions, 0.81 %) are dropped entirely rather than labelled. The
resulting dataset-wide class fractions are
circle1 0.319 / circle2 0.060 / leaf1 0.335 / leaf2 0.170 / leaf3 0.059 / leaf1_swap1 0.027 /
leaf1_swap2 0.029.

ii.
```python
out = np.empty((4, T), dtype=np.int64)
out[0] = stim_of_trial[t]
```
```python
if stim_of_trial[t] < 0:                       # wall without a canonical stim id
    continue
```
```python
'stimulus_id_scheme': ('canonical beh["stim_id"], merged over all experiment types of '
                       'a recording; per-mouse texture names are mapped onto it '
                       '(rock1->circle1, rock2->circle2, wood1->leaf1, wood2->leaf2, '
                       'wood5->leaf3)'),
```

iii. Step 5, decision 5: "**Stimulus label = canonical `stim_id` 0…6**, merged across all
exp_type entries of a recording. This makes the label comparable across mice that saw
leaf/circle vs rock/wood. Trials of `circle3` (no canonical id, 4 sessions) are dropped — the
reference never uses them." Step 12 adds that collapsing the label further "would throw away
the test-stimulus information that the paper's Figs. 2, 3 and ED 6–7 are entirely about", and
that the residual sklearn warning ("y_pred contains classes not in y_true") is unavoidable
because "each recording shows 2–8 of the 7 canonical stimuli".

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `beh['LickFr']`, the (fractional) imaging-frame number of every lick in the session.

ii.
```python
lick_frames = np.floor(b['LickFr']).astype(np.int64)
lick_frames = lick_frames[(lick_frames >= 0) & (lick_frames < nfr)]
lick = np.zeros(nfr, dtype=bool)
lick[lick_frames] = True
```

iii. Step 5 variable-mapping table: "`beh['LickFr']` → `output[1]` `licking`; 1 if ≥1 lick falls
in that imaging frame (`floor(LickFr)`), else 0; reference `utils.spk_2_firstLick` uses
`int(lickFr)`; binary, time-varying."

## 8-b. What processing is involved in computing `output` *Licking*?

i. Each lick's fractional frame number is floored to the frame it lands in; a frame is 1 if at
least one lick falls in it and 0 otherwise. Licks outside `[0, n_neural_frames)` are discarded.
The resulting flag is a per-frame binary series; dataset-wide 3.7 % of bins are "lick", and
licks occur only in the 28 task sessions.

ii.
```python
lick = np.zeros(nfr, dtype=bool)
lick[lick_frames] = True
...
out[1] = lick[fi]
```
```python
data['output_values'] = [STIM_NAMES, ['no_lick', 'lick'], ...]
```

iii. Step 10 Check 5: "licks outside the neural recording — `floor(LickFr)` clipped to
`[0, nfr)`; 0 licks were actually dropped." Step 12 Check 3 argues the 96.3 %/3.7 % imbalance
"is a real property of the experiment (only 28 of 89 recordings have a water spout at all);
`balanced_loss=True` and balanced accuracy handle this". Step 10 Check 4 further validates the
lick stream by reproducing the paper's Fig. 1c,d anticipatory-licking result from it.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` already indexes imaging frames, so the flag is on the neural grid; it is then
sub-indexed with the same `fi` used for the neural columns of that trial.

ii.
```python
out[1] = lick[fi]
...
neural = extract_neural(blocks, keep, ti['frame_idx'], ...)
```

iii. As for 3-c — every stream is indexed by the single shared frame-index array, and the
frame grid is truncated to `spk.shape[1]` before anything is computed.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `beh['ft_Pos']`, the position of the mouse in the virtual corridor at each imaging
frame, in decimetres (0–40 dm across the texture, 40–60 dm through the grey space).

ii.
```python
pos = b['ft_Pos'][:nfr]
...
out[2] = np.clip(np.floor(pos[fi] / POS_BIN_DM), 0, N_POS_BINS - 1)
```

iii. Step 5 variable-mapping table: "`beh['ft_Pos']` → `output[2]` `position_bin`;
`floor(ft_Pos/10)` clipped to 0…3 → 0–1, 1–2, 2–3, 3–4 m; corridor is 0–40 dm; time-varying."
Sanity check S4: "`ft_Pos` on kept frames spans 0–40 dm and increases monotonically within a
trial (verified: 0.0 – 40.0 dm; 0/348 non-monotonic trials in the spot-checked session)."

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Divide by 10 dm and floor, giving a 0-based index of the 1 m bin. No smoothing, no
interpolation, no re-referencing. Only frames inside the texture area are ever present, so the
grey space never contributes.

ii.
```python
CORRIDOR_DM = 40.0            # texture area length in decimetres (4 m)
POS_BIN_DM = 10.0             # 1 m position bins
N_POS_BINS = 4
...
out[2] = np.clip(np.floor(pos[fi] / POS_BIN_DM), 0, N_POS_BINS - 1)
```

iii. The decoder task asks for "4 equal-length, 1-m-long spatial bins", which is exactly
`floor(ft_Pos/10)` on a 4 m corridor. Sanity check S9: "Position class distribution
.250 / .249 / .250 / .252 — uniform as expected."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four fixed, equal-length spatial bins with edges at 0, 1, 2, 3, 4 m
(0, 10, 20, 30, 40 dm), named `0-1m`, `1-2m`, `2-3m`, `3-4m`. The clip at 3 puts a frame
sampled exactly at the 40 dm corridor end into the last bin rather than creating a fifth class.

ii.
```python
np.clip(np.floor(pos[fi] / POS_BIN_DM), 0, N_POS_BINS - 1)
```
```python
data['output_values'] = [..., ['0-1m', '1-2m', '2-3m', '3-4m'], ...]
'position_bin_edges_m': [0.0, 1.0, 2.0, 3.0, 4.0],
```

iii. Step 10 Check 5 edge case: "position exactly at 40 dm (corridor end) — `np.clip(..., 0, 3)`
keeps it in the last bin." The `--show-processing` plots overlay `ft_Pos` with the discretised
bin for four example trials to make the discretisation visually checkable.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` gives one value per imaging frame, so it is already on the neural grid; it is
sub-indexed with the same `fi` as the neural columns of that trial.

ii.
```python
out[2] = np.clip(np.floor(pos[fi] / POS_BIN_DM), 0, N_POS_BINS - 1)
...
neural = extract_neural(blocks, keep, ti['frame_idx'], ...)
```

iii. As for 3-c. The AI additionally plots "mean population activity as a function of position
bin" in the `--show-processing` figure as an alignment sanity check.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `beh['ft_RunSpeed']`, the running speed of the mouse interpolated to each imaging
frame. `beh['RunFr']` was explicitly rejected.

ii.
```python
speed = b['ft_RunSpeed'][:nfr]
...
speed=speed[frame_idx].astype(np.float32)
```

iii. Step 4 discrepancy table: "`beh['RunFr']` 'same as `ft_RunSpeed`' (notebook docs) — data
shows `RunFr = max(ft_RunSpeed,0) × 0.3226`, i.e. distance per frame, not speed. Resolution:
Use **`ft_RunSpeed`** (signed speed per imaging frame); the notebook comment is wrong."

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The speeds of every kept timepoint of **all 89 recordings** are pooled in a behaviour-only
first pass, and the 25th/50th/75th percentiles of that pooled distribution
(12.40, 25.28, 40.75 cm/s) become three global thresholds. Pass 1 is always run over all
sessions, so `--sample` and `--full` use identical edges. Each frame is then assigned to the
bin its speed falls in.

ii.
```python
speeds, ntr1 = [], 0
for k in keys_all:
    ti = build_trials(sessions[k], beh[k], len(beh[k]['ft']), day[k])
    speeds.append(ti['speed'])
speed_all = np.concatenate(speeds)
speed_edges = np.percentile(speed_all, [25, 50, 75])
```
```python
sb = np.searchsorted(speed_edges, ti['speed'], side='right').astype(np.int64)
for i in range(len(ti['inputs'])):
    a, bnd = ti['bounds'][i], ti['bounds'][i + 1]
    ti['outputs'][i][3] = sb[a:bnd]
```

iii. Step 5, decision 7: "**Running-speed quartiles are global** (pooled over every kept
timepoint of the whole dataset), so each of the 4 classes holds 25 % of the data exactly as
specified." Sanity check S8: "Each running-speed class holds exactly 25.0 % of timepoints"
(verification log confirms .250/.250/.250/.250). Step 10 Check 5 acknowledges the cost:
"with *global* quartiles the median session still has its largest speed class at only 41 % of
timepoints; only 9/89 sessions exceed 60 % and only DR10_2022_07_12 exceeds 95 %", the latter
being the recording whose speed trace it diagnosed as corrupted.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three global threshold values (the pooled 25/50/75th percentiles) and
`np.searchsorted(..., side='right')`, giving 4 classes named
`Q1_slowest, Q2, Q3, Q4_fastest`. The edges are recorded in the metadata.

ii.
```python
sb = np.searchsorted(speed_edges, ti['speed'], side='right').astype(np.int64)
```
```python
data['output_values'] = [..., ['Q1_slowest', 'Q2', 'Q3', 'Q4_fastest']]
'speed_bin_edges_cm_s': [float(x) for x in speed_edges],
```

iii. Decoder spec: "Running speed discretized into 4 bins, each corresponding to 25% of the
data" — the AI reads "the data" as the whole dataset, and verifies the realised fractions are
exactly 25 % each. The `--show-processing` plot draws the three edges over the speed histogram
and over the per-trial speed trace next to the assigned bin.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` gives one value per imaging frame, so it is already on the neural grid; it is
sub-indexed with the same `frame_idx` as the neural columns and then sliced per trial with the
same `bounds`.

ii.
```python
speed=speed[frame_idx].astype(np.float32)
...
sb = np.searchsorted(speed_edges, ti['speed'], side='right')
a, bnd = ti['bounds'][i], ti['bounds'][i + 1]
ti['outputs'][i][3] = sb[a:bnd]
```

iii. Step 12: "`corr(ft_move>0, ft_RunSpeed)` peaks at **lag 0** in all 89 recordings (median
r = 0.694), so the streams the conversion joins are frame-synchronous" — with the single
exception of DR10_2022_07_12, which is documented rather than corrected.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Seven issues are found and handled explicitly:
(1) the behaviour arrays run 1–2 frames past the imaging, so every behaviour stream is
truncated to `spk.shape[1]` and the trial structure is rebuilt with the exact neural frame
count in pass 2; (2) `ft_trInd = NaN` outside the behaviour recording is excluded with
`np.isfinite`; (3) `ft_trInd` outside `[0, ntrials)` is forced to −1; (4) licks whose frame
number falls outside `[0, nfr)` are dropped; (5) one trial with a negative `StartFr` is safe
because frames are selected by `ft_trInd` and times by `Trial_start_time`, never by `StartFr`
as an index; (6) zero-variance neurons get `sd = 1` so the z-score is 0 rather than NaN;
(7) walls with no canonical `stim_id` cause the trial to be dropped. One irreducible defect
(the corrupted `ft_RunSpeed` of DR10_2022_07_12) is diagnosed, documented and left uncorrected.

ii.
```python
nfr_true = blocks[0].shape[1]
assert all(x.shape[1] == nfr_true for x in blocks), k
assert nfr_true <= len(b['ft']), (k, nfr_true, len(b['ft']))
ti = build_trials(r, b, nfr_true, day[k])      # rebuild with the exact neural frame count
```
```python
valid = corr & move & np.isfinite(tr)
tr_int = np.where(valid, np.nan_to_num(tr, nan=-1), -1).astype(np.int64)
tr_int[(tr_int < 0) | (tr_int >= ntrials)] = -1
lick_frames = lick_frames[(lick_frames >= 0) & (lick_frames < nfr)]
```
```python
sd[sd == 0] = 1.0
```

iii. Step 4: "Length of `ft` vs neural frames — `len(ft)` = `nfr` + 1 or + 2. Truncate exactly
as the reference does." Step 10 Iteration 3 records that an earlier version used
`len(ft) − 2` as the frame count and "could silently drop up to 2 real frames per session";
it was restructured so pass 2 rebuilds trials with the exact `spk.shape[1]`. The residual
inconsistency — pass 1 (which only fixes the speed quartile edges) still uses the full
behaviour length — is documented as "a ≤2-frame difference per session". For the corrupted
speed trace: "No ad-hoc correction is invented."

## 12-a. What are the most time-consuming steps of the code?

i. Reading the 404 GB of `spk/*.npy` files (`load_spk_blocks`) and writing the 150.8 GB output
pickle. The full run takes 491 s end-to-end, of which 137.6 s is the pickle write and most of
the remainder is per-session disk I/O (up to 5 s per session when the file is cold; ~0 s when
it is already prefetched). The neuron/frame extraction itself is only ~1 s per session after
optimisation. The code prints per-session `load`, `extract`, `total`, cumulative size and
elapsed time so the bottleneck is visible in the log.

ii.
```python
print('[%3d/%3d] %-22s nneu %6d/%6d  trials %4d  T %6d  %6.2f GB  '
      '(load %4.1fs extract %4.1fs total %4.1fs)  cum %.1f GB, elapsed %.0fs' % ...)
```
```python
q = Queue(maxsize=1)
def prefetch():
    for k in keys:
        r = sessions[k]
        q.put((k, load_spk_blocks(r['mname'], r['datexp'], r['blk'])))
    q.put(None)
th = threading.Thread(target=prefetch, daemon=True); th.start()
```

iii. Step 6 table: "serial disk read then compute → background prefetch thread (queue depth 1)
→ overlaps ~2 s/session of I/O"; and "404 GB of spk files accumulating in the page cache forced
the kernel into reclaim for every new allocation → `posix_fadvise(POSIX_FADV_DONTNEED)` after
each file → removed a ~5× slowdown that grew through the run".

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop that could have been (and was) vectorised is the grouping of frames into
trials: instead of scanning the whole frame index once per trial, the AI does one stable
`argsort` plus a `bincount` and slices the result. The neuron/frame extraction loop was
replaced by chunked, multi-threaded numpy gathers with a single-pass mean/variance. What
remains as a Python loop is (a) the per-trial construction of the `(4, T)` input and output
arrays — ~37,800 iterations of cheap numpy work, which could be done with one `np.repeat` over
the concatenated frame axis and a `np.split`; (b) the per-trial `np.ascontiguousarray` slice of
the neural matrix, which is inherent to producing a list of per-trial arrays; and (c) the
per-session loops, which are I/O-bound rather than compute-bound. None of these is material
next to the 404 GB of reads.

ii.
```python
order   = np.argsort(tr_int, kind='stable')
counts  = np.bincount(tr_int[tr_int >= 0], minlength=ntrials)
ordered = order[int(np.sum(tr_int < 0)):]
```
```python
for a in range(0, n, chunk):
    bnd = min(a + chunk, n)
    if km[a:bnd].any():
        tasks.append((blk, a, bnd, km[a:bnd], oo + int(cs[a])))
...
with ThreadPoolExecutor(max_workers=nthreads) as ex:
    list(ex.map(work, tasks))
```
```python
for t in range(ntrials):        # remaining per-trial Python loop
    ...
    inp = np.empty((4, T), dtype=np.float32)
    out = np.empty((4, T), dtype=np.int64)
```

iii. Step 6: "single-threaded extraction → `ThreadPoolExecutor(16)` over 4096-row chunks (all
numpy calls release the GIL; the job is bandwidth-bound) → 33.6 s → 1.2 s per session";
"`blk.mean(1)` + `blk.std(1)` = 3 passes over the block → one pass with `sum` +
`einsum('ij,ij->i')` → −60 % of the arithmetic"; "`blk[km][:, frame_idx]` materialises an
(n_keep, n_frames) temporary → gather columns first: `blk[:, frame_idx][km]` → ~5× less
temporary memory".

## 12-c. What processing does the code repeat multiple times?

i. `build_trials` is run **twice for every session**: once in the behaviour-only pass 1 (over
all 89 recordings, to pool the running speeds and fix the global quartile edges) and again in
pass 2 with the exact neural frame count. Each run redoes the frame masking, the lick
rasterisation, the wall→stimulus lookup and the full per-trial input/output construction, and
the pass-1 results other than `speed` are discarded. A second, smaller repeat is that pass 1
is always executed even in `--sample` mode. Everything else is done once: the behaviour files
are read once for all the sessions they contain, and each `spk` and retinotopy file is read
once. `np.sort(fi)` is also applied to an already-sorted index array.

ii.
```python
# pass 1
for k in keys_all:
    ti = build_trials(sessions[k], beh[k], len(beh[k]['ft']), day[k])
    speeds.append(ti['speed'])
...
# pass 2
ti = build_trials(r, b, nfr_true, day[k])
```
```python
fi = np.sort(fi)     # `ordered` is already in increasing frame order within a trial
```

iii. Step 6: pass 1 is "behaviour only, all 89 recordings — `build_trials()` for every
recording and pooling of the running speeds to fix the three global quartile edges. Running
this over *all* recordings even in `--sample` mode guarantees the sample and the full dataset
use identical discretisation." Step 10 Iteration 3 explains why pass 2 must redo it: the exact
frame count is "only known once the spk file is opened in pass 2". Pass 1 costs 0.5 s in total,
so the repeat is deliberate and cheap.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Little, and all of it cheap: (a) pass 1 builds complete `(4, T)` input and output arrays for
all 89 recordings and throws everything away except `speed`; (b) `build_trials` returns `pos`
and `ft` per frame, which are used only by the `--show-processing` plots and are otherwise
dropped; (c) `out[3]` is allocated and filled with 0 in `build_trials` and immediately
overwritten with the speed bin in the main loop; (d) the z-score mean/variance are computed
over *all* ROIs of a chunk including the ones the area mask discards, and over all session
frames including frames no trial keeps (the latter is intentional — the statistics are meant to
be session-wide); (e) `build_session_index` collects `rewtype`, `exptype` and
`stim_id_entries`, of which only `exp_types` and the merged `stim_map` are used downstream;
(f) the redundant `np.sort(fi)`. Separately, the neural data is stored as **float32** rather
than float16, which doubles the output to 150.8 GB without adding information the decoder can
use — the largest avoidable cost in the pipeline, though it is a storage rather than a
processing decision.

ii.
```python
ti = build_trials(sessions[k], beh[k], len(beh[k]['ft']), day[k])
speeds.append(ti['speed'])          # everything else that build_trials produced is dropped
```
```python
out[3] = 0                          # speed bin filled in later
```
```python
return dict(frame_idx=..., bounds=..., inputs=..., outputs=...,
            speed=..., pos=pos[frame_idx].astype(np.float32), ft=ft[frame_idx], ...)
```
```python
out = np.empty((int(keep_mask.sum()), T), dtype=np.float32)
```

iii. The AI does not enumerate these as waste; it justifies the pass-1 repeat ("guarantees the
sample and the full dataset use identical discretisation") and reports that the whole
conversion runs in 491 s, so no further pruning was pursued. The float32 choice is stated in
the docstring ("Returns (n_kept_neurons, T) float32") but is never weighed against float16 in
the notes.
