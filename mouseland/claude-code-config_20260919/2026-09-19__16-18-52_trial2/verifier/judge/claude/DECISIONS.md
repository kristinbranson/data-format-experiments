# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything is read from the three subdirectories of `/app/data`: `beh/` (behaviour),
`spk/` (Suite2p deconvolved traces) and `retinotopy/` (visual-area label per neuron).
`beh/Imaging_Exp_info.npy` is the master index: a dict of 23 *experiment types*, each a
list of recording records carrying `mname`, `datexp`, `blk` and (sometimes) `stimtype`.
In a single "stage 1" pass (`load_session_registry`) the AI opens each of the 23
`Beh_<exp_type>.npy` files exactly once, walks the 142 index records, and collapses them
into an `OrderedDict` of 89 unique recordings keyed `<mname>_<datexp>_<blk>`. Only the 17
behaviour fields it actually needs (`BEH_FIELDS`) are retained, to keep the registry to a
few hundred MB instead of 6.6 GB. The expensive per-session reads (`spk/*_neural_data.npy`
and `retinotopy/*_trans.npz`) happen later in "stage 2" inside `process_session`, which is
run across 6–8 `multiprocessing` fork workers.

ii.
```python
exp_info = np.load(os.path.join(root, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()
reg = OrderedDict()
for exp_type, db in exp_info.items():
    beh_all = np.load(os.path.join(root, 'beh', 'Beh_%s.npy' % exp_type),
                      allow_pickle=True).item()
    for d in db:
        key = '%s_%s_%s' % (d['mname'], d['datexp'], d['blk'])
        beh_key = key + ('_' + d['stimtype'] if 'stimtype' in d else '')
        beh = beh_all[beh_key]
        ...
        reg[key] = dict(beh={f: beh[f] for f in BEH_FIELDS if f in beh}, ...)
    del beh_all
```
```python
path = os.path.join(root, 'spk', '%s_%s_%s_neural_data.npy'
                    % (entry['mname'], entry['datexp'], entry['blk']))
d = np.load(path, allow_pickle=True).item()
planes = d['spks']
```
```python
d = np.load(os.path.join(root, 'retinotopy', '%s_%s_trans.npz' % (mname, datexp)),
            allow_pickle=True)
iarea = np.asarray(d['iarea'])
```

iii. From CONVERSION_NOTES Step 1/2: `utils.load_spk`, `utils.load_retino` and
`utils.load_exp_beh` in the reference code define exactly this layout, so the AI reused
the same file naming and the same `np.load(...).item()['spks']` +
`np.concatenate(..., 0)` idiom. It split the work into a cheap behaviour-only stage and
an expensive neural stage explicitly so that the 405 GB of `spk` files are touched exactly
once each and the global running-speed quartiles can be computed before any neural data is
read. It records in Step 6 that an earlier version re-read behaviour files once per
experiment-type record (142 reads) and that this was reduced to 23 reads.

## 1-b. How are the data split into subjects (mice)?

i. The subject is `mname` from the index record, carried on every registry entry. At
assembly the subject list is the sorted set of unique `mname` over the converted sessions
(19 mice), and `subject_idx` is each session's index into it. Sessions are ordered
`(mname, datexp, blk)`, so a subject's sessions are contiguous and in date order.

ii.
```python
return OrderedDict(sorted(reg.items(),
                          key=lambda kv: (kv[1]['mname'], kv[1]['datexp'], kv[1]['blk'])))
```
```python
subjects = sorted({sess_beh[k]['mname'] for k in keys})
...
data['subject_idx'].append(subjects.index(sb['mname']))
data['subject_idx'] = np.asarray(data['subject_idx'], dtype=np.int64)
```

iii. The index already names the mouse for every recording, so nothing has to be inferred.
The AI checked the result against the paper ("We performed 89 recordings in 19 mice") and
lists the 19 names in Step 2 of CONVERSION_NOTES.

## 1-c. How are the data split into sessions?

i. A session is one unique recording `(mname, datexp, blk)` — 89 of them, matching the 89
`spk` files and 89 retinotopy files. The 142 `Imaging_Exp_info` records are *views* of
those 89 recordings (a recording appears under up to 5 experiment types). Rather than
keeping only the first view (which would lose information), the AI merges the views: it
asserts that `UniqWalls` and `ntrials` agree, and takes the **union** of the partially
`NaN`-masked `stim_id` vectors, asserting that no two views disagree on a non-NaN entry.
The first view's behaviour payload is kept.

ii.
```python
if key not in reg:
    reg[key] = dict(beh={f: beh[f] for f in BEH_FIELDS if f in beh},
                    uniq_walls=uw, stim_id=sid.copy(), ...)
else:
    e = reg[key]
    assert list(e['uniq_walls']) == list(uw), 'wall mismatch %s' % key
    assert e['beh']['ntrials'] == beh['ntrials'], 'ntrials mismatch %s' % key
    m = ~np.isnan(sid)
    conflict = m & ~np.isnan(e['stim_id']) & (e['stim_id'] != sid)
    assert not conflict.any(), 'stim_id conflict in %s' % key
    e['stim_id'][m] = sid[m]
```

iii. CONVERSION_NOTES Step 4/5 decision 1: "The 142 `exp_info` records are views of 89
recordings; using them all would duplicate neural data 1.6×. Verified the behaviour
payloads are identical." The union of `stim_id` is needed because "each experiment type
only labels the stimuli relevant to that comparison and sets the others to `NaN`"; merging
recovers a single conflict-free canonical stimulus code per recording.

## 1-d. How are the data split into trials?

i. A trial is one corridor traversal, identified by `ft_trInd == n`, but the AI keeps only
a *subset* of that traversal's frames: samples inside the 0–4 m texture region
(`ft_CorrSpc`) **while the virtual reality was moving** (`ft_move > 0`). On top of that it
applies a boundary guard requiring the frame index to lie in
`[StartFr[n] − 1, GrayFr[n] + 1]`. Because `ft_trInd` is non-decreasing, the retained
samples of a trial are contiguous *in the retained-sample array* and are cut out with
`np.unique(..., return_index=True)`. Trials are variable length (min 11, median 21, mean
21.6, max 178 bins) and are not padded. Note the consequence: consecutive bins of a trial
are **not** contiguous in real time — every frame in which the animal stopped is deleted
from the middle of the trial.

ii.
```python
running = ft_move > 0                                   # VR moved -> mouse running
tr_of_frame = np.where(np.isfinite(ft_tr), ft_tr, -1).astype(np.int64)
keep = ft_corr & running & (tr_of_frame >= 0) & (tr_of_frame < ntrials)
kept = np.flatnonzero(keep)
kept = kept[np.isfinite(trial_stim[tr_of_frame[kept]])]
...
tr_kept = tr_of_frame[kept]
inside = (kept >= start_fr_all[tr_kept] - 1.0) & (kept <= gray_fr_all[tr_kept] + 1.0)
kept = kept[inside]
trial = tr_of_frame[kept]
```
```python
assert np.all(np.diff(trial) >= 0), 'trial indices not sorted for %s' % sb['key']
uniq, starts = np.unique(trial, return_index=True)
stops = np.append(starts[1:], len(trial))
```

iii. CONVERSION_NOTES Step 5 decision 2: the mask is taken directly from the reference
code's `Get_dprime_selective_neuron` (`fr_valid = VRmove & isCorridor`) and from the paper
("we only selected data points inside the 0–4-m region of the corridors where the textures
were shown"; "We only considered timepoints during running for analysis, which removed
time periods when the task mice stopped to collect water rewards"). The AI verified
numerically that its mask selects exactly the same frames as the reference expression for
`VR2_2021_04_11_1` (14,126 frames). The boundary guard was added in Step 10 iteration 2
after the AI found 21 trials in which the frame just before the *next* corridor entry still
carried the previous `ft_trInd` while `ft_Pos` had already wrapped to ~0.

## 1-e. How are trials filtered based on quality controls?

i. Three rules. (1) Trials whose wall texture has no canonical `stim_id` are dropped —
this is exactly `circle3`, 309 of 38,110 trials in 4 sessions (implemented by dropping
those samples, so the trial ends up with zero retained samples and never appears).
(2) Trials with fewer than `MIN_SAMPLES_PER_TRIAL = 2` retained samples are dropped (the
AI reports the observed minimum is 11, so this guard never fires). (3) Trials that were
never imaged / whose frames all fall outside the guard window are implicitly dropped for
having no samples. There is **no** filter on excessively long trials; the long-stop
problem is handled instead by deleting non-running samples. 37,801 trials survive. No
session is dropped — all 89 are kept.

ii.
```python
kept = kept[np.isfinite(trial_stim[tr_of_frame[kept]])]
```
```python
for t, a, b in zip(uniq, starts, stops):
    T = b - a
    if T < MIN_SAMPLES_PER_TRIAL:
        continue
```

iii. Step 5 decision 8: "`circle3` … has `stim_id = NaN` in *every* experiment-type view,
i.e. the reference assigns it no role and excludes it from every analysis. Keeping it
would add an 8th class present in 4/89 sessions, which the balanced-accuracy metric would
weight as heavily as `leaf1`." Step 5 decision 13 describes the ≥2-sample rule as a guard.
Step 4 notes the pathological wall-clock trial durations (median 7.1 s, 1% > 74 s, max
1765 s) and states that removing non-running samples is what deals with them.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spks` from `spk/<mname>_<datexp>_<blk>_neural_data.npy` — a list of one
`(n_neurons_plane, n_frames)` float32 array per imaging plane, concatenated over planes.
The per-neuron visual area comes from `iarea` in `retinotopy/<mname>_<datexp>_trans.npz`.
No other neural variable is used; no ΔF/F or deconvolution is computed.

ii.
```python
d = np.load(path, allow_pickle=True).item()
planes = d['spks']
nframes = planes[0].shape[1]
n_total = sum(p.shape[0] for p in planes)
...
chunks = []
while planes:
    p = planes.pop(0)
    chunks.append(p[:, frame_idx])
    del p
X = np.concatenate(chunks, axis=0)
```
```python
iarea = np.asarray(d['iarea'])
region = np.full(len(iarea), -1, dtype=np.int64)
for code, ridx in IAREA_TO_REGION.items():
    region[iarea == code] = ridx
```

iii. Step 1: "No ΔF/F computation is needed. The `spk` files already contain Suite2p
*deconvolved* traces ('All our analyses were based on deconvolved fluorescence traces',
Methods)." The AI asserts `len(region_all) == n_total` for every session, which also
confirms that the retinotopy vector is aligned 1-to-1 with the plane-concatenated neuron
order used by `utils.load_spk`.

## 2-b. How is the `neural` data processed?

i. Three operations. (1) Each plane is **column-sliced to the retained frames before**
concatenation (same result as concatenating first, ~3× less peak memory). (2) After
neuron selection, every neuron is **z-scored** across the retained samples of that session.
(3) Stored as `float32`, one contiguous `(n_neurons, T)` array per trial. Trials are
variable length and are not padded.

ii.
```python
Xs = np.ascontiguousarray(X[sel])
if zscore:
    Xs = (Xs - Xs.mean(axis=1, keepdims=True)) / Xs.std(axis=1, keepdims=True)
Xs = Xs.astype(np.float32, copy=False)
trials = [np.ascontiguousarray(Xs[:, a:b]) for (a, b) in slices]
```

iii. Step 5 decision 4: "Suite2p deconvolved amplitudes span 0 – ~8000 with wildly
different per-neuron scales; the reference itself z-scores neurons before any population
read-out (`get_kfold_reward_response`, `Get_sort_spk`). Without it the decoder's SVD
initialisation and its L1-regularised linear projection are dominated by a handful of
high-amplitude ROIs." Step 12 Check 5 reports an ablation: mean validation accuracy 0.776
(z-scored) vs 0.777 (raw), i.e. a wash, so "the choice was made on principle, not on the
metric". `--no-zscore` reproduces the ablation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Three filters. (1) **Anatomical**: keep only neurons whose retinotopy `iarea` maps to
one of V1 (8), mHV (0,1,2,9), lHV (5,6), aHV (3,4); `iarea ∈ {-1, 7}` and anything else is
dropped. This is the reference's own rule and keeps 4,105,393 of 4,691,034 neurons.
(2) **Zero-variance**: neurons with `std == 0` over the retained samples are relabelled −1
and dropped. (3) **Subsampling**: each session is then reduced to at most
`--max-neurons = 2000` neurons, drawn without replacement with a per-session-seeded RNG
and stratified so each region's quota is proportional to its size. The shipped dataset
therefore contains 178,000 neurons (exactly 2,000 per session) instead of ~4.1 M.

ii.
```python
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV']
IAREA_TO_REGION = {8: 0, 0: 1, 1: 1, 2: 1, 9: 1, 5: 2, 6: 2, 3: 3, 4: 3}
```
```python
sd_all = X.std(axis=1)
n_dead = int(np.sum((sd_all == 0) & (region_all >= 0)))
region_all[sd_all == 0] = -1
rng = np.random.default_rng(zlib.crc32(key.encode()) ^ RNG_SEED)
sel = stratified_subsample(region_all, max_neurons, rng)
```
```python
def stratified_subsample(region, max_neurons, rng):
    pool = np.flatnonzero(region >= 0)
    if len(pool) <= max_neurons:
        return np.sort(pool)
    counts = np.array([int(np.sum(region[pool] == r)) for r in range(len(BRAIN_REGIONS))])
    target = max_neurons * counts / counts.sum()
    quota = np.minimum(np.floor(target).astype(int), counts)
    ...
```

iii. Step 5 decision 3 and decision 5. The anatomical rule is quoted from
`utils.Get_density_map` ("exclude neurons from outside of visual cortex"), and the AI notes
that no SNR/quality threshold exists anywhere in the reference because Suite2p's cell
classifier has already run. The subsample is justified purely on tractability: "keeping all
4,105,393 visual-cortex neurons would produce a **151.5 GB** array; training the provided
decoder on it … would require ~4.6 × 10¹⁵ FLOPs plus 30 TB of host→device traffic … 2000 is
also the provided decoder's own `svd_max_neurons`". Step 12 Check 4 reports an ablation
over 1000/1500/2000/3000/4000 neurons per session (mean validation accuracy
0.764/0.768/0.776/0.474/0.469) and picks 2000 as the best operating point, attributing the
collapse at ≥3000 to the provided decoder's L1 penalty rather than to the data.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry (trial start, `StartFr[n]`). Every trial's array
begins at the first retained sample at or after corridor entry and runs to the last
retained sample before grey-space entry; trials keep their own length, nothing is padded or
truncated to a common window. `off_start = 0.0`, `off_end = None`. The neural, input and
output arrays of a trial are all cut with the *same* column index set (`frame_idx[a:b]`),
so they cannot drift apart. Critically, `process_session` opens the `spk` file first and
passes its frame count back into `build_session_behavior`, so the behaviour is truncated to
the imaged frames *before* the trial windows are computed.

ii.
```python
sb = build_session_behavior(key, entry, day, nframes_neural=nframes)
inputs, outputs, slices, trial_ids = finalize_trials(sb, speed_edges)
frame_idx = sb['frame_idx']
...
chunks.append(p[:, frame_idx])
...
trials = [np.ascontiguousarray(Xs[:, a:b]) for (a, b) in slices]
```
```python
temporal_alignment_event='trial start = entry into the visual corridor (position 0 m)',
off_start=0.0,
off_end=None,
```

iii. Step 5: "**Alignment event** = trial start = corridor entry (`Trial_start_time[n]` /
`StartFr[n]`), `off_start = 0.0 s`, `off_end = None` (variable-length trials)." The
`--show-processing` figures were used to confirm visually that `time_since_trial_start`
is 0 at the plotted `StartFr` line for every trial and that the retained (shaded) samples
lie strictly between corridor entry and grey-space entry.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One bin = one imaging frame. No rebinning, no resampling and no positional
interpolation. The frame interval is measured per session as
`median(diff(ft)) * 86400` (0.31439–0.31537 s across the 89 sessions) and the metadata
reports the mean, 314.85 ms (fs = 3.176 Hz). Because non-running frames are deleted,
consecutive bins within a trial are not necessarily adjacent in real time.

ii.
```python
dt = float(np.median(np.diff(ft)) * SEC_PER_DAY)       # ~0.3149 s
```
```python
dts = [s['frame_interval_s'] for s in session_info]
...
time_bin_size=float(np.mean(dts) * 1000.0),
sampling_rate_hz=float(1.0 / np.mean(dts)),
frame_interval_s_range=[float(np.min(dts)), float(np.max(dts))],
```

iii. Step 3/Step 5: the imaging frame is the native resolution ("Calcium signal recording
frame rate: fs = 3.17 Hz" from the reference notebook), every `ft_*` behaviour stream is
already sampled once per imaging frame, and the reference's d′ analyses are "computed from
original estimated deconvolved traces without interpolation". The reference's other option
— interpolating onto 60 spatial bins — was explicitly rejected because the decoder task
requires time alignment and wants position as an *output* (Step 10, Check 3, row (d)).

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. `SoundFr` (the fractional imaging-frame index of the sound cue on each trial) and the
frame time axis, which the AI builds from `ft` (for the frame interval `dt`) and `ft_move`
(to decide which frames count).

ii.
```python
grid = np.arange(n, dtype=float)
tau_cue = np.interp(np.asarray(beh['SoundFr'], dtype=float)[:ntrials], grid, tau)
```

iii. Step 5 variable-mapping table lists `SoundFr`, `ft_move` → `input[0]`, noting that the
reference's own `spk_2_cue` uses `SoundFr`, i.e. the cue is natively expressed in the
imaging-frame domain and therefore needs no cross-stream alignment.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The AI does **not** use wall-clock time. It builds a "running clock"
`tau[k] = dt × #{running frames strictly before k}` over the whole session, interpolates the
fractional `SoundFr` onto that clock, and reports `tau_cue[trial] − tau[k]` in seconds —
positive before the cue, negative after. The resulting range over the dataset is
[−51.0 s, +19.4 s]. 17 trials (0.045%) have the cue after the last retained sample, so the
value stays positive throughout; these are kept.

ii.
```python
running = ft_move > 0                                   # VR moved -> mouse running
# "running clock": elapsed running time strictly before each frame
tau = np.zeros(n)
if n > 1:
    np.cumsum(running[:-1] * dt, out=tau[1:])
```
```python
time_to_cue=(tau_cue[trial] - tau[kept]).astype(np.float32),
```
```python
inp[0] = sb['time_to_cue'][a:b]
```

iii. Step 5 decision 6: "After the paper's non-running samples are removed, wall-clock
elapsed time has a pathological tail (median trial duration 7.1 s, 99th pct 74 s, max
1765 s) produced entirely by the stops the paper excludes. I therefore accumulate time only
over running frames … For the 88% of trials without a long stop this is within ~0.4 s of
wall-clock time; it removes the outliers and keeps the input on a scale the linear decoder
can use." The `--show-processing` figure was used to confirm the trace crosses zero exactly
at the plotted `SoundFr` line.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at exactly the same retained frame indices (`kept` / `frame_idx`) used
to column-slice the neural matrix, and cut into trials with the same `(a, b)` slices, so it
is sample-for-sample aligned by construction.

ii.
```python
kept = kept[inside]
...
time_to_cue=(tau_cue[trial] - tau[kept]).astype(np.float32),
```
```python
chunks.append(p[:, frame_idx])      # neural, same frame_idx
...
inp[0] = sb['time_to_cue'][a:b]     # input, same slice
```

iii. Step 5: every behaviour stream in this dataset is sampled once per imaging frame
(`ft_*` prefix), and the cue event is given as a frame index, so a single shared index
array aligns all streams. The AI also re-derives the behaviour masks from the true `spk`
frame count inside `process_session` specifically "so the neural and behavioural time axes
can never drift apart".

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. `datexp`, the recording date in the `Imaging_Exp_info` record (and hence in the session
id), together with `mname` to group by mouse.

ii.
```python
def training_day_lookup(reg):
    """Days since each mouse's first imaging session, per session."""
    first = {}
    for entry in reg.values():
        d = date(*map(int, entry['datexp'].split('_')))
        first[entry['mname']] = min(first.get(entry['mname'], d), d)
    return {key: float((date(*map(int, e['datexp'].split('_'))) - first[e['mname']]).days)
            for key, e in reg.items()}
```

iii. Step 5 decision 12: "`exp_info` only carries an explicit `days` field for a handful of
records, so the session date is the only definition available for all 89 sessions."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Calendar days elapsed since that mouse's earliest imaging session: the first session of
a mouse is 0 and later ones count real days, giving a range of 0–92 over the dataset (with
gaps, because recordings are not on consecutive days). The value is per-trial and is
broadcast across all bins of the trial. It is computed over all 89 sessions, so a sample
run and a full run agree.

ii.
```python
inp[1] = sb['day']
```
```python
day=np.float32(day),
```

iii. Step 5 decision 12 (above). The AI notes the value is "days since that mouse's first
imaging session (0–92)".

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. `StartFr` (the fractional imaging-frame index of corridor entry for each trial), plus
`ft` and `ft_move` for the time axis.

ii.
```python
grid = np.arange(n, dtype=float)
tau_start = np.interp(start_fr_all, grid, tau)
```

iii. Step 5 variable-mapping table: `StartFr`, `ft_move` → `input[2]`. `StartFr` is the
same corridor-entry event used as the alignment event, so the input is zero at the
alignment point by construction.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Again on the running clock: `tau[k] − tau_start[trial]`, in seconds, where `StartFr` is
interpolated onto `tau` because it is fractional. Range over the dataset [0.0, 55.8] s.
Two assertions guarantee the value is non-negative and non-decreasing within every trial.

ii.
```python
time_since_start=(tau[kept] - tau_start[trial]).astype(np.float32),
```
```python
inp[2] = sb['time_since_start'][a:b]
```
```python
assert (it[2] >= -1e-4).all(), 'negative time since trial start'
assert np.all(np.diff(it[2]) >= -1e-4), 'time since trial start not monotone'
```

iii. Step 5 decision 6, the same running-clock justification as 3-b: wall-clock trial
duration has a tail to 1765 s produced entirely by stops that the paper excludes from
analysis, so time is accumulated only over running frames.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Identically to 3-c: evaluated at the retained frame indices used for the neural columns
and cut with the same per-trial slices.

ii.
```python
time_since_start=(tau[kept] - tau_start[trial]).astype(np.float32),
...
inp[2] = sb['time_since_start'][a:b]
```

iii. All streams share the imaging-frame grid; one index array (`frame_idx`) and one set of
slices are used for neural, input and output alike.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. `isRew` and `WallName`. The AI does not use `isRew` directly per trial; it first
identifies the rewarded wall texture as `WallName[isRew][0]` (the reference's
`utils.get_cat_id` rule) and then flags every trial whose wall is that texture. Sessions
where `isRew` is all-False (61 of 89 unsupervised/naive sessions) get all zeros.

ii.
```python
is_rew = np.asarray(beh['isRew'], dtype=bool)[:ntrials]
if is_rew.any():
    rew_wall = str(wall_name[is_rew][0])
    trial_rew = (wall_name == rew_wall).astype(np.float32)
else:
    rew_wall = None
    trial_rew = np.zeros(ntrials, dtype=np.float32)
```

iii. Step 5 decision 9: "`utils.get_cat_id` identifies the rewarded wall as
`WallName[isRew][0]`; a trial is 'in the rewarded corridor' iff its wall is that wall.
Unsupervised/naive sessions have `isRew` all-False → all zeros, which is correct (those
mice were never rewarded, though they still heard the cue)."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. None beyond the per-trial flag; it is a 0/1 float broadcast across all bins of the
trial. Dataset mean 0.127; 28 sessions have any 1s, exactly the 28 sessions where a reward
was ever delivered.

ii.
```python
inp[3] = sb['trial_rew'][t]
```

iii. Step 9/10 consistency check: "`reward_available == 1` ⇔ `stimulus == 2` (the canonical
leaf1/rewarded role) in every one of those 28 sessions", used as a cross-check that the
rewarded-wall identification is right.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. `WallName` (the texture on each trial), `UniqWalls` and `stim_id` — the last being the
paper's canonical *role* code, merged across all experiment-type views of the recording
(see 1-c). A `WallName → stim_id` lookup is built per session.

ii.
```python
wall2id = {w: s for w, s in zip(entry['uniq_walls'], entry['stim_id'])}
wall_name = np.asarray(beh['WallName'])[:ntrials]
trial_stim = np.array([wall2id[w] for w in wall_name], dtype=float)
```

iii. Step 1: "`beh['stim_id']` is the paper's canonical stimulus role code:
`0:circle1, 1:circle2, 2:leaf1, 3:leaf2, 4:leaf3, 5:leaf1_swap1, 6:leaf1_swap2`. It is
*role*-based, not name-based (e.g. for mouse TX109 the physical `circle1` texture carries
`stim_id == 2` … for rock/wood mice `rock1→0`, `wood1→2`)."

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The merged `stim_id` is used directly as a 7-class categorical label, per trial,
broadcast across every bin of the trial. `output_values[0]` is
`['circle1','circle2','leaf1','leaf2','leaf3','leaf1_swap1','leaf1_swap2']`. Trials whose
wall has no `stim_id` (`circle3`) are dropped entirely (see 1-e). Observed class fractions:
(0.319, 0.060, 0.335, 0.170, 0.059, 0.027, 0.029).

ii.
```python
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2']
```
```python
out[0] = int(sb['trial_stim'][t])
```

iii. Step 5 decision 7: "This is the reference's own stimulus variable and is *role*-aligned
across mice (leaf1 ≡ the rewarded exemplar), which is what makes a shared read-out across
19 mice and 3 different texture sets (circle/leaf, rock/wood) meaningful. Using raw
`WallName` would give mutually incompatible labels across mice; using only the two
categories (circle vs leaf) would throw away the exemplar information the paper's Figs. 2–3
are built on."

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. `LickFr`, the (fractional) imaging-frame index of every lick in the session.

ii.
```python
lick_bin = np.zeros(n, dtype=bool)
lf = np.floor(np.asarray(beh['LickFr'], dtype=float)).astype(np.int64)
lf = lf[(lf >= 0) & (lf < n)]
lick_bin[lf] = True
```

iii. Step 5 mapping table: `LickFr` → `output[1]`; the reference's own lick analyses
(`spk_2_firstLick`, `lickCount`) use `LickFr` in the frame domain.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Binary per bin: 1 if at least one lick falls in that frame, else 0. The fractional lick
frame index is truncated with `floor`. Licks outside the imaged range (`< 0` or `≥ n`) are
discarded. Overall 3.72% of bins are licks.

ii.
```python
lf = np.floor(np.asarray(beh['LickFr'], dtype=float)).astype(np.int64)
lf = lf[(lf >= 0) & (lf < n)]
lick_bin[lf] = True
...
out[1] = sb['lick'][a:b]
```

iii. Step 5 decision 10: "frame *k* covers `[t_k, t_{k+1})`, so a lick at fractional frame
index 140.945 belongs to frame 140." The `--show-processing` figure overlays raw `LickFr`
tick marks on the binary output to confirm the binning.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` is already on the imaging-frame grid, so the per-frame flag is indexed with the
same `frame_idx` and cut with the same per-trial slices as the neural columns.

ii.
```python
lick=lick_bin[kept].astype(np.int64),
...
out[1] = sb['lick'][a:b]
```

iii. Same as 3-c/5-c: all streams live on the imaging-frame grid, so one shared index array
aligns them.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. `ft_Pos`, the virtual-reality position at each imaging frame, in decimetres (0–40 dm
across the 4 m texture, 40–60 dm through the grey space).

ii.
```python
ft_pos = np.asarray(beh['ft_Pos'], dtype=float)[:n]
...
position=ft_pos[kept],
```

iii. Step 2: "Units: positions are in **decimetres** (corridor 0–40 dm = 0–4 m, grey space
40–60 dm)." Step 4 records the check that `ft_CorrSpc` is exactly `ft_Pos < 40`.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Only unit conversion and discretisation — no smoothing, no interpolation. Since the
retained samples are restricted to `ft_CorrSpc`, positions are always in [0, 40) dm.

ii.
```python
TEXTURE_LENGTH_DM = 40.0   # 4 m texture region; positions are in decimetres
N_POSITION_BINS = 4        # 4 equal-length 1 m bins
```
```python
pos_bin = np.clip((sb['position'] / (TEXTURE_LENGTH_DM / N_POSITION_BINS)).astype(np.int64),
                  0, N_POSITION_BINS - 1)
```

iii. The decoder task specifies "Position in corridor discretized into 4 equal-length,
1-m-long spatial bins"; the 4 m texture region of the paper maps exactly onto 4 × 10 dm
bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. `floor(ft_Pos / 10)` clipped to [0, 3], i.e. fixed 1 m edges at 0/1/2/3/4 m, labelled
`['0-1m','1-2m','2-3m','3-4m']`. Observed fractions (0.2499, 0.2488, 0.2496, 0.2517),
essentially uniform as expected from the constant 60 cm s⁻¹ VR speed. Two assertions check
that the bin never decreases within a trial and that every trial starts in bin 0.

ii.
```python
OUTPUT_VALUES = [..., ['0-1m', '1-2m', '2-3m', '3-4m'], ...]
position_bin_edges_m=[0.0, 1.0, 2.0, 3.0, 4.0],
```
```python
assert np.all(np.diff(ot[2]) >= 0), 'position bin decreases in (%d, %d)' % (s, t)
assert ot[2][0] == 0, 'trial (%d, %d) does not start at 0-1 m' % (s, t)
```

iii. Directly from the decoder-task specification; the AI added the two assertions in
Step 10 after the corridor/grey-boundary artifact was found, to guarantee no sample is
attributed to the wrong trial.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is one value per imaging frame, so it is indexed with the same `frame_idx` and
cut with the same per-trial slices as the neural columns.

ii.
```python
position=ft_pos[kept],
...
out[2] = pos_bin[a:b]
```

iii. Same shared-index argument as the other streams; the `--show-processing` figure
overlays `ft_Pos/10 dm` on the discretised output to show the steps occur exactly at the
integer crossings.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. `ft_RunSpeed`, the animal's running speed at each imaging frame (cm s⁻¹; range over the
retained samples [−19.17, 163.49]).

ii.
```python
ft_spd = np.asarray(beh['ft_RunSpeed'], dtype=float)[:n]
...
speed=ft_spd[kept],
```

iii. Step 5 mapping table: `ft_RunSpeed` → `output[3]`.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. No smoothing or averaging — the instantaneous per-frame speed is used. The only
processing is discretisation into quartiles. The quartile edges are computed **once,
globally**, over the retained samples of all 89 sessions during the cheap behaviour-only
stage 1, so the sample run and the full run use identical boundaries: edges
[12.40, 25.28, 40.75].

ii.
```python
all_speed = np.concatenate([sb['speed'] for sb in sess_beh.values()])
speed_edges = np.percentile(all_speed, [25, 50, 75])
print('global running-speed quartile edges: %s   (n = %d samples, range [%.2f, %.2f])'
      % (np.round(speed_edges, 4), len(all_speed), all_speed.min(), all_speed.max()))
```

iii. Step 5 decision 11: "Speed quartiles computed globally (over every retained sample of
the whole dataset), not per session, so that class *q2* means the same thing in every
session for the shared read-out." Step 6 adds that computing them in the behaviour-only
pass is what makes the sample and full runs share class boundaries.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize` against the three global quartile edges, giving 4 classes labelled
`['speed_q1','speed_q2','speed_q3','speed_q4']`. Globally the classes are exactly 25% each
(0.25/0.25/0.25/0.25); per session they are not, e.g. the two sample sessions gave
(0.340, 0.165, 0.175, 0.320).

ii.
```python
spd_bin = np.digitize(sb['speed'], speed_edges).astype(np.int64)
```
```python
speed_bin_edges=[float(e) for e in speed_edges],
speed_units='cm/s (beh["ft_RunSpeed"])',
```

iii. The decoder task asks for "Running speed discretized into 4 bins, each corresponding
to 25% of the data"; the AI reads "the data" as the whole dataset and documents the
per-session departure in Step 7 ("per-session, since the quartile edges are global").
Because non-running samples are already excluded, there is no large mass of exactly-zero
speeds to make the quartile split ambiguous.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is one value per imaging frame, so it is indexed with the same `frame_idx`
and cut with the same per-trial slices as the neural columns.

ii.
```python
speed=ft_spd[kept],
...
out[3] = spd_bin[a:b]
```

iii. Same shared-index argument; the `--show-processing` figure draws the global quartile
edges on the raw `ft_RunSpeed` trace next to the discretised output.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled explicitly.
* **Behaviour longer than imaging**: all `ft_*` streams are truncated to
  `n = min(len(ft), nframes_spk)`. The AI measured `len(ft) − nfr ∈ {1, 2, 3}` across all
  89 sessions.
* **Frames with NaN `ft_trInd`** (outside the behavioural record, 0–166 per session) are
  excluded.
* **Licks outside the imaged range** are discarded (`0 ≤ floor(LickFr) < n`).
* **Corridor/grey-space wrap artifact**: 21 samples across 37,801 trials carried the
  previous trial's `ft_trInd` while `ft_Pos` had already wrapped to ~0; a
  `StartFr − 1 ≤ frame ≤ GrayFr + 1` guard removes exactly those while keeping the 1,301
  legitimate last-corridor samples that exceed the interpolated `GrayFr` by < 0.1 frame.
* **Zero-variance neurons** are dropped so the z-score cannot produce 0/0.
* **Empty / too-short trials** are dropped (`MIN_SAMPLES_PER_TRIAL`).
* **Structural mismatches** are caught by assertions (retinotopy neuron count vs `spks`
  neuron count; `ft_trInd` monotone; `stim_id` conflicts; per-trial shape/finiteness).
The AI also verified there are no NaNs in `SoundFr`, `ft_RunSpeed`, `ft_Pos` or `LickFr` in
any session. There is **no** per-session `try/except`, so an unexpected failure aborts the
whole run rather than skipping one session.

ii.
```python
n = len(ft_full) if nframes_neural is None else min(len(ft_full), int(nframes_neural))
```
```python
tr_of_frame = np.where(np.isfinite(ft_tr), ft_tr, -1).astype(np.int64)
keep = ft_corr & running & (tr_of_frame >= 0) & (tr_of_frame < ntrials)
```
```python
inside = (kept >= start_fr_all[tr_kept] - 1.0) & (kept <= gray_fr_all[tr_kept] + 1.0)
kept = kept[inside]
```
```python
assert len(region_all) == n_total, \
    '%s: retinotopy has %d neurons, spks has %d' % (key, len(region_all), n_total)
```

iii. Step 10 Check 5 enumerates each edge case and its handling; the wrap artifact is
documented as "Iteration 2" with its effect quantified (815,506 → 815,485 timepoints, −21;
no other statistic changed). The `[:nfr]` truncation is justified as matching the reference
code, which always slices behaviour with `beh[...][:nfr]`, `nfr = spk.shape[1]`.

## 12-a. What are the most time-consuming steps of the code?

i. Reading and slicing the 405 GB of `spk` files. The script's own timing output shows
stage 2 (neural) = 46.7 s wall with 8 workers (≈ 310 s of CPU, ~0.5–9.4 s per session,
scaling with file size), pickling the 6.57 GB output = 8.5 s, and stage 1 (all behaviour,
89 sessions) = 1.4 s, for a total of 60.1 s. So neural file I/O plus the
`concatenate`/`z-score` on the sliced matrix dominate by roughly 5:1 over everything else.

ii.
```python
t_neural = time.time() - t0
print('stage 2 (neural): %.1f s for %d sessions (%.2f s/session)'
      % (t_neural, len(keys), t_neural / max(1, len(jobs))))
```
```python
with mp.get_context('fork').Pool(processes=min(args.workers, len(jobs))) as pool:
    for i, (key, res) in enumerate(pool.imap_unordered(_worker, jobs)):
```

iii. Step 6/Step 7: the AI profiled the sample run at "≈ 2.1 s per GB of spk file, mean
4.6 GB → ≈ 850 s serial", then cut it two ways — column-slicing each plane before
concatenation ("~2× on stage 2, and −5 GB peak RSS per worker") and an 8-process pool
("~7× on stage 2") — bringing the estimate to ≈ 120 s, well under the 15-minute budget.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The hot path is already vectorized: sample selection is a single boolean mask,
trial boundaries come from one `np.unique(..., return_index=True)` call, position and speed
discretisation are whole-array operations, and z-scoring is a single broadcast. The
remaining Python loops are all small or I/O-bound:
* the per-trial loop in `finalize_trials` (one iteration per trial, allocating two small
  `(4, T)` arrays) — could be replaced by building two `(4, n_samples)` session-wide arrays
  once and taking views;
* `while planes: p = planes.pop(0)` — intentional, it exists to free each plane's memory;
* `for code, ridx in IAREA_TO_REGION.items()` — 9 full-array comparisons instead of one
  lookup-table gather;
* the `while quota.sum() < max_neurons` largest-remainder loop and the per-region
  `counts` comprehension in `stratified_subsample`;
* the sequential `for exp_type, db in exp_info.items()` registry build and the
  `sess_beh` comprehension, which run `build_session_behavior` 89 times in one process.
None of these is material next to the file I/O.

ii.
```python
for t, a, b in zip(uniq, starts, stops):
    T = b - a
    ...
    inp = np.empty((4, T), dtype=np.float32)
```
```python
for code, ridx in IAREA_TO_REGION.items():
    region[iarea == code] = ridx
```
```python
counts = np.array([int(np.sum(region[pool] == r)) for r in range(len(BRAIN_REGIONS))])
```

iii. Step 6 lists the inefficiencies the AI identified and removed (plane-wise slicing,
one read per behaviour file, restricting the registry to `BEH_FIELDS`, computing the speed
quartiles once in the cheap pass) and the speed-ups added (`multiprocessing.Pool`,
`float32`, contiguous per-trial views). The AI did not document the residual loops above,
presumably because the measured 60 s total made further optimisation pointless.

## 12-c. What processing does the code repeat multiple times?

i. **`build_session_behavior` is run twice for every session.** Stage 1 runs it for all 89
sessions with `nframes_neural=None` (full behaviour length) purely so the global speed
quartiles and the sample-mode session picks can be computed; stage 2 runs it again inside
`process_session` with the true `spk` frame count. All of the masking, the running clock,
the lick binning and the per-trial stimulus/reward derivation are therefore computed twice.
Related repetition: the registry is built and all 89 sessions' behaviour is processed even
in `--sample` mode, where only 2 sessions are converted; and `--show-processing` calls
`process_session` a third time for the two plotted sessions (with `return_raw=True`, which
also makes an extra full copy of the neural matrix).

ii.
```python
sess_beh = OrderedDict((key, build_session_behavior(key, entry, day_lut[key]))
                       for key, entry in reg.items())          # stage 1, all 89
```
```python
sb = build_session_behavior(key, entry, day, nframes_neural=nframes)   # stage 2, again
```
```python
raw = Xs.copy() if return_raw else None
```

iii. The duplication is a deliberate consequence of two documented decisions: Step 5
decision 11 ("Speed quartiles computed globally … so that class *q2* means the same thing
in every session") and Step 6 ("the global running-speed quartiles are computed once in the
cheap behaviour-only pass, so the sample run and the full run share identical class
boundaries"). The AI measured this pass at 1.4 s for all 89 sessions, i.e. ~2% of runtime.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Mostly small, and mostly the stage-1/stage-2 duplication above: the stage-1
`sess_beh` dict computes the running clock, lick bins, per-trial stimulus ids, reward flags
and time inputs for all 89 sessions, of which only `speed` (for the quartiles), `mname`
and `rew_wall` (for `--sample` selection) are ever used — everything else is recomputed and
thrown away. Beyond that: the entire `spk` file is read from disk even though only ~1/3 of
its columns (corridor + running frames) and 2,000 of ~50,000 rows are kept; `n_dead`,
`exptypes`, `reward_mode` and `cohort` are carried through into `session_info` but are not
used by the decoder; under `--show-processing` a full extra copy of the neural matrix
(`raw`) is made just for a histogram. The truly wasteful item — computing z-scores and
region labels for neurons that the subsample will discard — is avoided: the subsample is
applied *before* z-scoring.

ii.
```python
sess_beh = OrderedDict((key, build_session_behavior(key, entry, day_lut[key]))
                       for key, entry in reg.items())
all_speed = np.concatenate([sb['speed'] for sb in sess_beh.values()])
speed_edges = np.percentile(all_speed, [25, 50, 75])
del all_speed
```
```python
sel = stratified_subsample(region_all, max_neurons, rng)
Xs = np.ascontiguousarray(X[sel])
del X
raw = Xs.copy() if return_raw else None
if zscore:
    Xs = (Xs - Xs.mean(axis=1, keepdims=True)) / Xs.std(axis=1, keepdims=True)
```

iii. The AI does not flag any of this as waste. Its Step 6 framing is that the behaviour
pass is cheap by construction (only `BEH_FIELDS` retained, 1.4 s total) and that the
reads of the `spk` files are unavoidable because the retained columns are scattered
throughout each file. The extra `raw` copy is confined to the two `--show-processing`
sessions.
