# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything comes from the three subfolders of `/app/data`: `beh/` (behaviour), `spk/` (deconvolved traces) and `retinotopy/` (visual-area label per neuron). `beh/Imaging_Exp_info.npy` is read first as the master index: it is a dict `exp_type -> list of session entries`, and the 142 `(exp_type, session)` entries are collapsed into 89 unique recordings keyed `mname_datexp_blk` (`build_recording_table`). The 23 `Beh_<exp_type>.npy` files are then read once each, in a 12-process pool, keeping only the ~22 fields the conversion needs (`load_behaviour`); one behaviour record is retained per recording (the AI verified the arrays are identical across exp_types), while the `UniqWalls -> stim_id` maps of every appearance are merged. Spike and retinotopy files are read per session inside the parallel session workers.

ii.
```python
exp_info = np.load(os.path.join(BEH_DIR, 'Imaging_Exp_info.npy'), allow_pickle=True).item()
recs = collections.OrderedDict()
for exp_type, db in exp_info.items():
    for ndb in db:
        key = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
        behkey = key + ('_' + ndb['stimtype'] if 'stimtype' in ndb else '')
        rec = recs.setdefault(key, {...})
```
```python
def _load_beh_file(args):
    exp_type, wanted = args
    B = np.load(os.path.join(BEH_DIR, 'Beh_%s.npy' % exp_type), allow_pickle=True).item()
    out = []
    for behkey, reckey in wanted:
        beh = B[behkey]
        out.append((reckey, {f: beh[f] for f in BEH_FIELDS if f in beh}))
    return out
```
```python
d = np.load(os.path.join(SPK_DIR, '%s_neural_data.npy' % key), allow_pickle=True).item()
ret = np.load(os.path.join(RET_DIR, '%s_%s_trans.npz' % (rec['mname'], rec['datexp'])), allow_pickle=True)
```

iii. From CONVERSION_NOTES Step 1/2: this mirrors `utils.load_exp_beh`, `utils.load_spk` and `utils.load_retino`. The behaviour files total 6.6 GB and the spike files 405 GB, so behaviour is loaded once per file (in parallel, fields subset: "0.3 s instead of ~2 min") and the spike files are sliced per imaging plane so the full 3–7 GB matrix is never materialised. The AI verified that recordings duplicated across exp_types have byte-identical `ft`/`StartFr`/`WallName` (53/53 pairs), justifying one behaviour copy per recording.

## 1-b. How are the data split into subjects?

i. The subject is the `mname` field of the index entry, carried on every recording record. At assembly the subject list is the sorted set of unique mouse names (19 mice) and `subject_idx` is each session's index into that list.

ii.
```python
rec = recs.setdefault(key, {'key': key, 'mname': ndb['mname'], ...})
```
```python
subjects = sorted({r['mname'] for r in results})
...
'subject_idx': np.array([subjects.index(r['mname']) for r in results], dtype=np.int64),
```

iii. The index already names the mouse, so nothing has to be inferred. Verified against the paper: "89 recordings in 19 mice" — the conversion reports 89 sessions / 19 subjects, with 1–8 sessions per mouse.

## 1-c. How are the data split into sessions?

i. A session is one recording = one mouse, one date, one block (`mname_datexp_blk`). The 142 index entries contain duplicates (the same recording is listed under several `exp_type`s and `stimtype`s); an `OrderedDict` keyed by the recording id keeps the first behaviour record and merges the wall→`stim_id` maps of all appearances, yielding 89 sessions.

ii.
```python
key = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
behkey = key + ('_' + ndb['stimtype'] if 'stimtype' in ndb else '')
rec = recs.setdefault(key, {...'exp_types': [], 'beh_keys': [], ...})
rec['exp_types'].append(exp_type); rec['beh_keys'].append(behkey)
```
```python
w2id = rec.setdefault('wall2id', {})
for w, s in zip(beh['UniqWalls'], np.atleast_1d(beh['stim_id']).astype(float)):
    if not np.isnan(s):
        w2id[str(w)] = int(s)
if reckey not in beh_by_rec:
    beh_by_rec[reckey] = beh
```

iii. "Sessions appear up to 5 times across exp_types; deduplicating to 89 recordings avoids counting trials/neurons several times, and merging the `stim_id` maps keeps every wall labelled" (CONVERSION_NOTES Step 12). Each exp_type labels only the stimuli relevant to that comparison and NaNs the rest, hence the union.

## 1-d. How are the data split into trials?

i. A trial is one corridor traversal, as declared by the per-frame label `beh['ft_trInd']`. Within a trial the AI keeps only the frames that are (a) inside the 0–4 m texture region (`ft_CorrSpc`), (b) recorded while the VR was moving, i.e. the mouse was running (`ft_move > 0`), and (c) have a non-NaN trial label; behaviour is first truncated to `nfr = spk.shape[1]`. Frames are grouped by trial label with a lexsort/split, so trials keep their natural, variable length (11–178 bins, median 21) but are **not** temporally contiguous: stationary frames inside a traversal are removed.

ii.
```python
def trial_frames(beh, nfr):
    tr = beh['ft_trInd'][:nfr]
    valid = (beh['ft_CorrSpc'][:nfr].astype(bool) & (beh['ft_move'][:nfr] > 0)
             & ~np.isnan(tr))
    frames = np.flatnonzero(valid)
    labels = tr[frames].astype(int)
    order = np.lexsort((frames, labels))
    frames, labels = frames[order], labels[order]
    bounds = np.flatnonzero(np.diff(labels)) + 1
    groups = np.split(np.arange(len(frames)), bounds)
    trial_ids = [int(labels[g[0]]) for g in groups]
    return frames, labels, groups, trial_ids
```

iii. This is exactly the `fr_valid` mask of the reference code: `utils.Get_dprime_selective_neuron` computes `VRmove = beh['ft_move'][:nfr]>0; isCorridor = beh['ft_CorrSpc'][:nfr]; fr_valid = VRmove & isCorridor`, and the same mask is repeated at four other places in `utils.py`. It also implements the Methods text quoted in CONVERSION_NOTES Step 3: "we only selected data points inside the 0-4-m region of the corridors where the textures were shown" and "We excluded the data points in which the animal was not running". The `~isnan(ft_trInd)` term was added so that every kept frame can be assigned to a trial (documented as difference 2 in Step 10, Check 3).

## 1-e. How are trials filtered based on quality controls?

i. No trials are filtered. All 38,110 behaviour trials of the 89 sessions are converted ("all trials kept? True"). The AI verified that every trial retains at least 11 valid frames and every session has at least 84 trials, so there is no trial or session with too little data. Trials in which the mouse stood still for minutes survive, but only their running frames are stored, so their stored length stays bounded (max 178 bins); their elapsed-time inputs nevertheless reach ±1763 s.

ii.
```python
frames, labels, groups, trial_ids = trial_frames(beh, nfr)
...
for g, tid in zip(groups, trial_ids):
    T = len(g)
    neural_out.append(np.ascontiguousarray(spk[:, g]))
```
(no length/quality predicate anywhere; the only implicit filter is that a trial with zero valid frames would produce no group.)

iii. CONVERSION_NOTES Step 3/5: "No explicit trial rejection for the main analyses… For the decoder, trials must contain at least a few valid (corridor + running) frames; in the data every trial has >=11 valid frames, so no trial is lost." Step 10 Check 5 investigated the 260 trials longer than 100 s of elapsed time and confirmed in the raw data that they are genuine standing-still episodes (e.g. TX88_2022_07_19_1 trial 391: 5,621 contiguous frames of which only 50 running), concluding "the elapsed-time inputs are correct, not artefacts".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy` — a list of one (neurons × frames) `float32` array per imaging plane — concatenated over planes. The per-neuron visual area comes from `iarea` in `retinotopy/<mouse>_<date>_trans.npz`.

ii.
```python
d = np.load(os.path.join(SPK_DIR, '%s_neural_data.npy' % key), allow_pickle=True).item()
planes = d['spks']
...
for p in planes:
    n = p.shape[0]
    rows = sel[(sel >= off) & (sel < off + n)] - off
    if len(rows):
        out.append(np.asarray(p[np.ix_(rows, frames)], dtype=np.float32))
    off += n
return np.concatenate(out, 0), nneu, nfr
```
```python
ret = np.load(os.path.join(RET_DIR, '%s_%s_trans.npz' % (rec['mname'], rec['datexp'])), allow_pickle=True)
iarea = ret['iarea']
```

iii. "Equivalent to `np.concatenate(d['spks'], 0)[sel][:, frames]` (`utils.load_spk`) but without materialising the full (n_neurons × n_frames) matrix" (docstring). Sanity check in Step 10 verified `np.allclose(np.concatenate(spks,0)[sel][:,frames], data['neural'][s][t])` for 20 trials across 5 sessions.

## 2-b. How is the `neural` data processed?

i. Not processed at all beyond selection: the deconvolved traces are taken as they are, the selected neurons × selected frames are sliced out per plane, cast/kept as `float32`, and split per trial into (n_neurons, n_bins) arrays. No dF/F, no deconvolution, no smoothing, no normalisation, no padding.

ii.
```python
out.append(np.asarray(p[np.ix_(rows, frames)], dtype=np.float32))
...
neural_out.append(np.ascontiguousarray(spk[:, g]))
```

iii. CONVERSION_NOTES Step 1/3: "Data are **deconvolved** (suite2p non-negative deconvolution, 0.75 s decay). **No dF/F computation needed**, and the paper states all analyses use deconvolved traces." Metadata records `neural_signal: 'suite2p non-negative deconvolved fluorescence (0.75 s decay timescale), one value per imaging frame; no dF/F, as in the reference analyses'`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. (1) Area: a neuron is kept only if its `iarea` code maps to V1 (8), mHV (0,1,2,9), lHV (5,6) or aHV (3,4) — codes 7 and −1 are dropped; this keeps 4,105,393 of 4,691,034 neurons. (2) **Size cap**: of the in-area neurons, at most `--max-neurons` (default **2,000**) per session are kept, drawn as a simple random sample without replacement with a fixed seed (0) and sorted. Every session therefore stores exactly 2,000 neurons instead of the 20,547–89,577 recorded / mean 46,128 in-area. No spike-quality or activity-based filter is applied.

ii.
```python
def select_neurons(iarea, max_neurons, seed=RNG_SEED):
    region_idx = np.full(len(iarea), -1, dtype=np.int64)
    for r, name in enumerate(BRAIN_REGIONS):
        region_idx[np.isin(iarea, AREA_CODES[name])] = r
    in_area = np.flatnonzero(region_idx >= 0)
    if len(in_area) > max_neurons:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(in_area, size=max_neurons, replace=False))
    else:
        sel = in_area
    return sel, region_idx[sel], len(iarea), len(in_area)
```

iii. Area filter: "No neuron quality filtering is applied anywhere in the reference code beyond suite2p cell detection… Neurons with `iarea` = 7 or -1 belong to no named area and are excluded from all area-based analyses in the reference code" (Step 3). Subsample: "storing all in-area neurons would need 152 GB; the decoder projects each session to 100 PCs and its own SVD initialisation caps at 2,000 neurons (`svd_max_neurons` default), so nothing decodable is lost" (Step 10, Check 3). Step 5 additionally describes the sample as "stratified proportionally across V1/mHV/lHV/aHV", although the implementation is an unstratified uniform draw (composition is preserved only in expectation).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is trial start = corridor entry (`StartFr`). Each trial's neural matrix starts at the first kept frame of that traversal (the first running frame inside the texture, i.e. at/just after corridor entry) and ends at its last kept frame. Trials keep their own length; nothing is padded or truncated to a common window, so `off_start = 0.0` and `off_end = None`.

ii.
```python
neural_out.append(np.ascontiguousarray(spk[:, g]))   # g = kept frames of this trial, in time order
```
```python
'temporal_alignment_event':
    'trial start = entry into the virtual-reality corridor (beh["StartFr"])',
'off_start': 0.0,
'off_end': None,
```

iii. Step 5: "Trial = one corridor traversal, indexed by `beh['ft_trInd']`; alignment event = trial start = corridor entry (`StartFr`)… Trials keep their natural (variable) length". The `off_end_note` metadata field explains that "a trial ends when the mouse leaves the 4 m texture corridor, and only frames inside the corridor while the mouse was running are kept, so the number of kept bins varies".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One bin = one imaging frame. No rebinning, no resampling, no position interpolation. `time_bin_size` is written as the median inter-frame interval across sessions, 314.69 ms (3.178 Hz). Note that, because non-running frames are removed, consecutive bins inside a trial are not always 315 ms apart in real time — the actual elapsed time is carried by the `time_since_trial_start` input.

ii.
```python
'dt_s': float(np.median(np.diff(t_frames))),
...
dt = float(np.median([r['dt_s'] for r in results]))
data['metadata'] = {... 'time_bin_size': dt * 1000.0, 'frame_rate_hz': 1.0 / dt, ...}
```

iii. Step 5: "Time bin = one imaging frame (~315 ms, 3.17 Hz). No re-binning: the paper's analyses of trial dynamics use the native frame rate, and the spec requires equal bin sizes across trials/sessions." Step 5 decision 2: "the reference position-interpolation (60 bins/6 m) is only used for position-resolved figures, while all selectivity statistics are computed on native deconvolved frames." Consistency check: measured 3.171–3.181 Hz per session vs the notebook's stated fs = 3.17 Hz.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `beh['SoundFr']` (the fractional imaging-frame index of the sound cue in each trial) and `beh['ft']` (MATLAB datenum timestamp of every imaging frame).

ii.
```python
def frame_times_s(beh, nfr):
    ft = beh['ft'][:nfr]
    return (ft - ft[0]) * 86400.0
...
fr_axis = np.arange(nfr)
t_cue = np.interp(beh['SoundFr'], fr_axis, t_frames)     # sound cue
```

iii. Step 1/5: `utils.spk_2_cue` uses `SoundFr` as the cue time base, and all event variables (`StartFr`, `SoundFr`, `LickFr`) are given as (fractional) frame indices, so "alignment is exact in frame units"; `ft` supplies the conversion to seconds (timestamps are in days, hence ×86400).

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Frame times are converted to seconds relative to the first frame of the session; the fractional cue frame is interpolated onto that time axis; the input is `t_cue(trial) − t(bin)`, so it is positive before the cue and negative after, in seconds. It is time-varying and stored as `float32`. No clipping or normalisation: values range over [−1763.3, 723.5] s because elapsed time inside a trial includes long stationary pauses whose frames were dropped.

ii.
```python
t_cue = np.interp(beh['SoundFr'], fr_axis, t_frames)
...
time_to_cue = t_cue[labels] - t_sel            # > 0 before the cue, < 0 after
...
inp[0] = time_to_cue[g]
```

iii. Step 5 variable table: "signed time until cue: t_cue − t_frame (positive before cue, negative after)… cue frame index is fractional; converted to seconds with the session's frame period". The AI cross-checked the sign and the zero crossing against an independent estimate `(ft_Pos − SoundPos)/VR speed` in panel 4 of the processing plots, and verified in Step 10 Check 5 that `time_to_sound_cue` is strictly decreasing within every trial and that the cue falls inside the kept window for 38,094/38,110 trials.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated on exactly the same kept-frame indices used to slice the neural matrix of that trial, so it has the same length and the same bins.

ii.
```python
t_sel = t_frames[frames]
time_to_cue = t_cue[labels] - t_sel
...
neural_out.append(np.ascontiguousarray(spk[:, g]))
inp[0] = time_to_cue[g]
```

iii. Every stream in this dataset is indexed by imaging frame, so taking the same `frames`/`g` index guarantees alignment; verified numerically by the independent `np.allclose` re-derivation of input 0 in Step 10 Check 2 and visually in plot panel 4.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the `datexp` date string of each recording in `Imaging_Exp_info.npy`, grouped by `mname`.

ii.
```python
for key, rec in recs.items():
    y, m, d = [int(v) for v in rec['datexp'].split('_')]
    rec['date'] = datetime.date(y, m, d)
    by_mouse[rec['mname']].append(rec)
```

iii. Step 4b/5: "Day-of-training can be derived as days since each mouse's first recording"; the `days` field exists in only 8 of the 142 index entries, so it cannot be used as the general source.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the earliest recording date is day 0 and every other session is the number of **calendar days** after it. The value is a per-trial constant broadcast across all bins of every trial of the session. Range across the dataset: 0–92 days.

ii.
```python
for mname, rs in by_mouse.items():
    d0 = min(r['date'] for r in rs)
    for r in rs:
        r['day_of_training'] = float((r['date'] - d0).days)
...
day = rec['day_of_training']
inp[1] = day
```

iii. Step 5 decision 8: "Day of training = days since the mouse's first imaging session (continuous, per trial)". Step 10 Check 5 verified that every mouse has a day-0 session.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `beh['StartFr']` (the fractional frame of corridor entry for each trial) and `beh['ft']` (frame timestamps).

ii.
```python
t_start = np.interp(beh['StartFr'], fr_axis, t_frames)   # corridor entry
```

iii. Step 1: `StartFr` is documented in the reference notebook as corridor entry, and the decoder task defines trial start as corridor entry.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The fractional entry frame is interpolated onto the seconds axis and subtracted from the time of each bin, giving a non-negative, monotonically increasing time in seconds. Time-varying, `float32`, unnormalised; range [0, 1765.2] s (long values come from within-trial standing still).

ii.
```python
time_since_start = t_sel - t_start[labels]     # >= 0
...
inp[2] = time_since_start[g]
```

iii. Step 5: "t_frame − t_StartFr … continuous, time-varying; correctly accounts for the removed non-running frames" — i.e. it deliberately reports real elapsed time rather than bin count, so the gaps created by dropping stationary frames remain visible to the decoder. Step 10 Check 5: 0 trials with negative values, strictly increasing within every trial.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed on the same kept-frame indices as the neural matrix of that trial.

ii.
```python
t_sel = t_frames[frames]
time_since_start = t_sel - t_start[labels]
inp[2] = time_since_start[g]      # same g as neural_out.append(spk[:, g])
```

iii. Same frame-index alignment argument as 3-c; re-derived independently with `np.allclose` in Step 10 Check 2.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From the trial's wall texture (`beh['WallName']`) mapped through the merged `UniqWalls → stim_id` table to the canonical id, combined with a session-level flag `is_task`, which is true if any reward was actually delivered in the session (`~np.isnan(beh['RewardFr'])`). `beh['isRew']` is deliberately **not** used.

ii.
```python
rec['is_task'] = bool(np.any(~np.isnan(beh['RewardFr'])))
...
stim_ids, wall_names = stim_categories(beh, rec['wall2id'])
reward_available = (stim_ids == 2) & rec['is_task']
```

iii. Step 4 discrepancy table: the AI verified in all 89 sessions that `isRew == ~isnan(RewardFr)`, i.e. `isRew` means "reward was actually delivered", **not** "trial was in the rewarded corridor" (in one session 34 rewarded-corridor trials have `isRew == False`). The reference code identifies the rewarded corridor as the wall with canonical `stim_id == 2` (`Get_dprime_rewPred_neuron`, `get_reward_neuorns`), so the AI used that, gated on the session being a reward (task) session.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. A per-trial 0/1 constant broadcast across all bins: 1 iff the session delivered water at all and the trial's canonical stimulus id is 2 (the rewarded corridor). It is 0 for all 61 non-task (unsupervised / naive / grating-cohort) sessions. Result: 4,446 trials (11.7% of all trials, 39.7% of task-session trials).

ii.
```python
reward_available = (stim_ids == 2) & rec['is_task']
...
inp[3] = float(reward_available[tid])
```

iii. Step 5 decision 6: "Reward availability = rewarded corridor (not reward delivered), per the decoder spec" ("1 if in rewarded corridor, 0 if not"). Step 10 Check 5 verified no reward availability outside task sessions.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `beh['WallName']` (the wall texture name of each trial) mapped through a per-recording dictionary built from `beh['UniqWalls']` and `beh['stim_id']`, unioned over every exp_type in which the recording appears. Wall names that remain unlabelled after the union (only `circle3`, 309 trials) are assigned a new id 7.

ii.
```python
def stim_categories(beh, wall2id):
    wn = np.array([str(w) for w in beh['WallName']])
    ids = np.array([wall2id.get(w, UNRESOLVED_STIM_ID) for w in wn], dtype=np.int64)
    return ids, wn
```
```python
w2id = rec.setdefault('wall2id', {})
for w, s in zip(beh['UniqWalls'], np.atleast_1d(beh['stim_id']).astype(float)):
    if not np.isnan(s):
        w2id[str(w)] = int(s)
```

iii. Step 4/5: each exp_type labels only the stimuli relevant to that comparison and NaNs the others, so the union over exp_types is needed to label every wall; `circle3` has no canonical id in any exp_type of the sessions in which it appears, and "is the third crop of the *non-rewarded* texture family, i.e. the mirror of `leaf3` (id 4). It is given its own category id 7 (`nonrew3`) rather than being dropped."

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The 15 raw wall names are mapped to the dataset's canonical `stim_id` code, producing **8 classes** named `['nonrew_crop1','nonrew_crop2','rew_crop1','rew_crop2','rew_crop3','rew_crop1_swap1','rew_crop1_swap2','nonrew_crop3']`. This code is **task-relative, not physical**: the rewarded texture of a mouse is always id 2 and the non-rewarded one id 0, so `circle1` is class 0 for most mice but class 2 for TX109, and `rock1`/`wood1` share labels with `circle1`/`leaf1`. Different crops of the same texture (crop1/crop2/crop3) and the two spatial swaps are separate classes. The value is per trial, broadcast over all bins, stored as `int16`. Per-trial counts: 11964/2266/12339/6538/2279/1173/1242/309.

ii.
```python
STIM_VALUES = ['nonrew_crop1', 'nonrew_crop2', 'rew_crop1', 'rew_crop2',
               'rew_crop3', 'rew_crop1_swap1', 'rew_crop1_swap2', 'nonrew_crop3']
UNRESOLVED_STIM_ID = 7
...
out[0] = stim_ids[tid]
```
```python
'stimulus_category_note':
    'canonical beh["stim_id"] code, which is task-relative: crops 1/2/3 of the non-rewarded '
    'texture are 0/1/7 and crops 1/2/3 of the rewarded texture are 2/3/4, while 5/6 are the '
    'two spatially swapped versions of rewarded crop 1. The physical textures differ between '
    'mice (circle/leaf or rock/wood), the code does not.',
```

iii. Step 5 decision 5: "Stimulus category = canonical `stim_id` rather than raw wall names, because the physical textures differ between mice (circle/leaf vs rock/wood) while the canonical id encodes the same task role in every mouse." Trajectory step 28: "stim_id is a canonical, task-relative code… canonical ids unify them across mice."

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `beh['LickFr']`, the (fractional) imaging-frame index of every lick in the session.

ii.
```python
lick_any = np.zeros(nfr, dtype=bool)
if len(beh['LickFr']):
    lf = np.round(beh['LickFr']).astype(int)
    lf = lf[(lf >= 0) & (lf < nfr)]
    lick_any[lf] = True
```

iii. Step 1/5: `LickFr` (with `LickTrind`) is the frame-indexed lick variable used by `utils.spk_2_cue` / `spk_2_firstLick`, so it is already on the imaging-frame grid.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A session-length boolean vector is set to True at the **rounded** frame index of every lick (licks rounding outside `[0, nfr)` are dropped), then indexed by the trial's kept frames: a bin is 1 if at least one lick landed in it, 0 otherwise. Stored as `int16`. Licks are present only in the 28 task sessions; the 61 non-task recordings have empty `LickFr`, so licking is identically 0 there. Overall 3.5% of bins are class 1 (12.1% within task sessions).

ii.
```python
licking = lick_any[frames].astype(np.int64)
...
out[1] = licking[g]
```

iii. Step 4 discrepancy table: "lick arrays are empty for all 61 non-task recordings… Licking is 0 everywhere in non-task sessions. Kept those sessions… with licking = 0, which is the behaviourally correct value for mice that never received water." Step 12 discusses and rejects dropping non-task sessions or redefining licking per trial.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Via the same frame indices: the per-frame lick flag is sliced with the trial's kept-frame group, so it has the same length and bin alignment as the neural matrix. A consequence of the `ft_move>0` frame filter is that licks occurring while the mouse was stationary or in the grey space are not represented.

ii.
```python
licking = lick_any[frames].astype(np.int64)
neural_out.append(np.ascontiguousarray(spk[:, g]))
out[1] = licking[g]
```

iii. All streams share the imaging-frame index; Step 10 Check 2 re-derived `output 1` from raw `LickFr` with `np.array_equal` for 20 trials across 5 sessions.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `beh['ft_Pos']`, the per-frame position in the corridor in decimetres (0–40 dm = the 4 m texture region, 40–60 dm = grey space).

ii.
```python
pos = beh['ft_Pos'][:nfr][frames]
```

iii. Step 1/4: positions are in decimetres (`Corridor_Length` = 60 dm, `Texture_Length` = 40 dm), and the AI verified for all 89 sessions that `ft_CorrSpc == (ft_Pos < 40)`, i.e. the kept frames are exactly the 0–4 m region of the Methods.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimetres is divided by 10 (1 m), truncated to an integer and clipped to 0–3, giving four 1-m bins; stored per bin as `int16`. Distribution is essentially uniform (0.250 / 0.249 / 0.250 / 0.252) as expected for constant-speed VR traversals.

ii.
```python
POS_BIN_DM = 10.0   # 1 m = 10 dm; behaviour positions are in decimetres
...
pos_bin = np.clip((pos / POS_BIN_DM).astype(np.int64), 0, N_POS_BINS - 1)
```

iii. Step 5: "4 equal 1-m bins: [0,10),[10,20),[20,30),[30,40) dm", following the decoder spec ("4 equal-length, 1-m-long spatial bins") and the paper's 4 m textured corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Fixed edges at 0, 10, 20, 30, 40 dm (0–1, 1–2, 2–3, 3–4 m), i.e. `floor(pos/10)` clipped into [0,3]. The clip is a safety net only: because kept frames satisfy `ft_CorrSpc`, positions are always < 40 dm, and the AI verified no ≥4 m frames leak in.

ii.
```python
pos_bin = np.clip((pos / POS_BIN_DM).astype(np.int64), 0, N_POS_BINS - 1)
...
'position_bin_edges_m': [0.0, 1.0, 2.0, 3.0, 4.0],
'output_values': [..., ['0-1m', '1-2m', '2-3m', '3-4m'], ...]
```

iii. Directly from the decoder task spec; the reference-data check `ft_CorrSpc == (ft_Pos < 40)` guarantees the four bins tile the kept data.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is one value per imaging frame, sliced with the same kept-frame indices as the neural data, so it is bin-for-bin aligned.

ii.
```python
pos = beh['ft_Pos'][:nfr][frames]
pos_bin = np.clip((pos / POS_BIN_DM).astype(np.int64), 0, N_POS_BINS - 1)
out[2] = pos_bin[g]
```

iii. Same frame-index argument; plot panel 3 shows position vs time-since-trial-start for five trials coloured by bin, with bin changes exactly at the 1 m edges, and Step 10 Check 2 re-derived output 2 from the raw behaviour with `np.array_equal`.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `beh['ft_RunSpeed']`, the running speed (cm/s) at each imaging frame.

ii.
```python
speed = beh['ft_RunSpeed'][:nfr][frames]
```
```python
def session_speed_values(beh, nfr):
    frames, _, _, _ = trial_frames(beh, nfr)
    return beh['ft_RunSpeed'][:nfr][frames]
```

iii. Step 5 variable table: `ft_RunSpeed` is the frame-interpolated run speed used by the reference; no alternative source was needed.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Before converting any session, the speeds of the kept (corridor + running) frames of **all 89 sessions** are pooled and the 25/50/75th percentiles are taken as three global thresholds (12.42 / 25.35 / 40.85 cm/s). Each kept frame is then assigned a bin with `np.digitize` against those thresholds. This is a single global discretisation, so every bin holds exactly 25% of the dataset (0.250/0.250/0.250/0.250) although an individual session need not be uniform.

ii.
```python
speeds = []
for key, rec in recs.items():
    beh = beh_by_rec[key]
    speeds.append(session_speed_values(beh, len(beh['ft']) - 2))
speeds = np.concatenate(speeds)
speed_thr = np.percentile(speeds, [25, 50, 75])
...
speed_bin = np.digitize(speed, speed_thr).astype(np.int64)
out[3] = speed_bin[g]
```

iii. Step 5 decision 7: "Speed discretization by global quartiles over all kept frames (25% of data per bin), as specified" — the decoder spec asks for "4 bins, each corresponding to 25% of the data". Using one global set of thresholds also makes the class label mean the same speed range in every session.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three fixed global thresholds [12.42, 25.35, 40.85] cm/s, applied with `np.digitize` (bin 0 = speed ≤ 12.42, …, bin 3 = speed > 40.85). The thresholds are recorded in the metadata. Because the kept frames are running frames only, the mass of exactly-zero speeds that would break a threshold-based split is largely absent, and the realised distribution is exactly 25% per bin.

ii.
```python
speed_thr = np.percentile(speeds, [25, 50, 75])
speed_bin = np.digitize(speed, speed_thr).astype(np.int64)
...
'speed_bin_edges_cm_per_s': [float(x) for x in speed_thr],
'output_values': [..., ['speed_q1', 'speed_q2', 'speed_q3', 'speed_q4']]
```

iii. As above; the thresholds were also computed independently during exploration (Step 2: "quartiles 12.4 / 25.3 / 40.8") and the sample plot panel 2 overlays them on the raw speed trace coloured by bin.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is one value per imaging frame and is sliced with the same kept-frame indices as the neural data, so it is bin-for-bin aligned.

ii.
```python
speed = beh['ft_RunSpeed'][:nfr][frames]
speed_bin = np.digitize(speed, speed_thr).astype(np.int64)
out[3] = speed_bin[g]     # same g as neural_out.append(spk[:, g])
```

iii. Same frame-index argument; Step 10 Check 2 re-derived output 3 from the raw behaviour and the global thresholds with `np.array_equal`.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several specific issues are handled: (a) behaviour arrays are 1–2 frames longer than the imaging, so every behaviour stream is truncated to `nfr` taken from the spike file; (b) frames with NaN `ft_trInd` cannot be assigned to a trial and are excluded by the frame mask; (c) licks rounding to a frame outside `[0, nfr)` are discarded, and sessions with empty `LickFr` produce all-zero licking; (d) wall names with no canonical `stim_id` in any exp_type (`circle3`, 309 trials) get their own category instead of being dropped; (e) recordings duplicated across exp_types are de-duplicated and their stimulus maps merged; (f) every produced array is asserted finite, and the neuron count in the retinotopy file is asserted to equal the neuron count in the spike file. One inconsistency: the global speed-threshold pass approximates the truncation as `len(beh['ft']) - 2` instead of reading `spk.shape[1]` (conservative, affects at most a couple of frames per session out of 825k).

ii.
```python
_, n_neu_spk, nfr = load_spk_selected(key, None, None)
assert n_neu_spk == n_neurons_file, 'neuron count mismatch for %s' % key
...
valid = (beh['ft_CorrSpc'][:nfr].astype(bool) & (beh['ft_move'][:nfr] > 0) & ~np.isnan(tr))
...
lf = lf[(lf >= 0) & (lf < nfr)]
...
ids = np.array([wall2id.get(w, UNRESOLVED_STIM_ID) for w in wn], dtype=np.int64)
...
for arr, nm in ((neural_out, 'neural'), (input_out, 'input'), (output_out, 'output')):
    for a in arr:
        assert np.all(np.isfinite(a)), '%s of %s has non-finite values' % (nm, key)
```

iii. Step 4/12: "Behaviour arrays are 1-2 frames longer than the neural traces; truncating to `spk.shape[1]` (as the reference does) avoids an off-by-one at the end of each session." Step 10 Check 3: the `~isnan(ft_trInd)` term "only makes the trial assignment well defined". Step 12: "309 trials use a wall (`circle3`) that has no `stim_id` in any exp_type; instead of dropping them they were given their own category".

## 12-a. What are the most time-consuming steps of the code?

i. Reading and decoding the deconvolved spike files, which total 405 GB. The script prints total and spike-load time per session: spike load is 0.7–5.1 s and session total 3–14 s, with the rest dominated by the (uninstrumented) first read of the same file and the per-plane slicing. With 12 worker processes the 89 sessions take 59.6 s of the 70.7 s total runtime; behaviour loading is 0.3 s and pickling the 6.6 GB output 10.7 s.

ii.
```python
t_load0 = time.time()
spk, _, _ = load_spk_selected(key, sel, frames)
t_load = time.time() - t_load0
...
print('[%2d/%2d] %s: %d neurons stored ... %.1f s (spk load %.1f s) | elapsed %.1f s, est total %.1f s' % ...)
```

iii. Trajectory step 14: "Data is huge: /app/data/spk = 405 GB across 89 session files (4-7 GB each)… This makes full conversion I/O heavy; need to plan (subsample neurons, read each file once)." Step 6/7 notes list the mitigations: per-plane slicing so the full matrix is never materialised, 12 parallel session workers ("~10x wall-clock"), and parallel field-subset behaviour loading.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI vectorised all per-frame work (times, position bins, speed bins, licking are computed for the whole session at once and then split by trial). What remains looped is minor: the per-trial Python loop that assembles the three arrays of each trial (`for g, tid in zip(groups, trial_ids)`, ~428 iterations/session, each doing a fancy-index copy), the per-plane loop in `load_spk_selected`, the `np.split` group construction, the list comprehension over `WallName` in `stim_categories`, and the per-trial `np.all(np.isfinite(...))` assertion loop. None is close to the I/O cost.

ii.
```python
for g, tid in zip(groups, trial_ids):
    T = len(g)
    neural_out.append(np.ascontiguousarray(spk[:, g]))
    inp = np.empty((4, T), dtype=np.float32)
    ...
```
```python
ids = np.array([wall2id.get(w, UNRESOLVED_STIM_ID) for w in wn], dtype=np.int64)
```

iii. Step 6: "all per-frame quantities (times, position bins, speed bins, licking) computed vectorised for the whole session and then split by trial". The AI did not flag any remaining loop as a bottleneck, and its timing output supports that: total runtime was 70.7 s against a 15-minute budget, so "no further optimisation was required".

## 12-c. What processing does the code repeat multiple times?

i. The notes claim nothing is repeated ("each session read exactly once"), but the code does repeat two things. (1) **Each spike file is fully loaded twice per session**: `load_spk_selected(key, None, None)` unpickles the whole 4–7 GB `spks` dict just to read `nfr` and the neuron count, discards it, and the file is then loaded again to slice the selected neurons/frames. The instrumented `load_s` only measures the second load. (2) `trial_frames` is computed twice per session — once in `main` through `session_speed_values` for the global speed thresholds, and again inside `convert_session`. In addition, behaviour records for recordings appearing under several exp_types are extracted once per appearance (cheap, since the file is read once).

ii.
```python
# first full load, used only for nfr / n_neurons
_, n_neu_spk, nfr = load_spk_selected(key, None, None)
...
# second full load, then slicing
spk, _, _ = load_spk_selected(key, sel, frames)
```
```python
def load_spk_selected(key, sel, frames):
    d = np.load(os.path.join(SPK_DIR, '%s_neural_data.npy' % key), allow_pickle=True).item()
    planes = d['spks']
    nfr = min(p.shape[1] for p in planes)
    nneu = sum(p.shape[0] for p in planes)
    if frames is None:
        return None, nneu, nfr
```
```python
speeds.append(session_speed_values(beh, len(beh['ft']) - 2))   # calls trial_frames
...
frames, labels, groups, trial_ids = trial_frames(beh, nfr)     # again, per session
```

iii. The AI documents the opposite: Step 6, "sessions converted in parallel (`imap_unordered`, 12 workers), each session read exactly once". No justification is given for the duplicate read, and CONVERSION_NOTES never mentions it.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI documents none. In practice: the first spike-file load (12-c) is pure waste; the `np.all(np.isfinite(...))` assertion makes an extra full pass over every neural/input/output array; a number of per-session statistics that only appear in log lines or metadata are computed for every session (`n_reward_trials`, `n_rewcorridor_trials`, `n_licks`, `wall_names`, `n_frames_kept`); the end-of-run summary concatenates all inputs and outputs across sessions to print distributions; and `np.ascontiguousarray(spk[:, g])` makes a second copy of data that fancy indexing has already copied. Storing `neural` as `float32` and `output` as `int16` rather than `float16`/`int8` doubles the pickle (6.6 GB) with no downstream benefit, since the decoder recasts anyway.

ii.
```python
for arr, nm in ((neural_out, 'neural'), (input_out, 'input'), (output_out, 'output')):
    for a in arr:
        assert np.all(np.isfinite(a)), '%s of %s has non-finite values' % (nm, key)
```
```python
neural_out.append(np.ascontiguousarray(spk[:, g]))
...
allout = np.concatenate([np.concatenate(r['output'], 1) for r in results], 1)
allin = np.concatenate([np.concatenate(r['input'], 1) for r in results], 1)
```

iii. No justification is given in CONVERSION_NOTES; these are framed as validation and reporting. The finiteness assertion and the printed distributions are defensible as sanity checks (the instructions ask for them), and the per-session statistics feed `metadata['session_info']`, which is part of the delivered format.
