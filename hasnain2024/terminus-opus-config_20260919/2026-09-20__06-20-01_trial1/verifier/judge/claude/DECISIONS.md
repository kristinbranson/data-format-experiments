# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not glob the data folders. It hard-codes a list of 44 sessions — 25 in `Ephys_Behavior` (fixed delay, `FIXED`) and 19 in `RandomizedDelay_Ephys_Behavior` (randomized delay, `RANDOM`) — transcribed from the authors' `/app/code/DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m` files, with commented-out entries excluded. Each tuple carries `(folder, animal, date, probe list)`. Two other data folders (`DelayInhibition_BilatMC_Behavior`, `GoCueInhibition_BilatMC_Behavior`, all `MAH*` files) are excluded because they are behaviour+video only and contain no spike data. Each session's `data_structure_<ANM>_<DATE>.mat` is opened once by a `Session` class that sniffs the MATLAB version (`is_hdf5` reads the file header) and dispatches to `h5py` (v7.3, 36 files) or `scipy.io.loadmat` (v7, 11 files), exposing a single accessor API (`bpvec`, `ev`, `ev_cell`, `clusters`, `traj_trial`, `feat_names`, `vidshift`, `probe_locs`). Motion energy comes from the companion `motionEnergy_<ANM>_<DATE>.mat` via `load_motion_energy`, which handles three different file layouts. Sessions are processed in parallel with a `multiprocessing.Pool` (8 workers) and assembled in list order. All 44 sessions survive the session-level filters (>=10 units, >=2 trials).

ii.
```python
FIXED = [
    ('Ephys_Behavior', 'JEB6',  '2021-04-18', [2]),
    ...
    ('Ephys_Behavior', 'JEB19', '2023-04-21', [1]),
]
RANDOM = [
    ('RandomizedDelay_Ephys_Behavior', 'JEB11', '2022-05-10', [1]),
    ...
    ('RandomizedDelay_Ephys_Behavior', 'JEB24', '2023-11-03', [1]),
]
SESSIONS = FIXED + RANDOM
```

```python
def is_hdf5(fn):
    with open(fn, 'rb') as f:
        return b'7.3' in f.read(120)


class Session:
    """Accessor for one `data_structure_<ANM>_<DATE>.mat` file (obj struct)."""

    def __init__(self, fn):
        self.fn = fn
        self.hdf5 = is_hdf5(fn)
        if self.hdf5:
            self.f = h5py.File(fn, 'r')
            self.obj = self.f['obj']
        else:
            self.obj = sio.loadmat(fn, struct_as_record=False, squeeze_me=True)['obj']
```

```python
    nproc = min(args.nproc, len(sessions))
    if nproc > 1:
        with Pool(nproc) as pool:
            results = pool.map(_worker, list(zip(sessions, want_debug)))
```

iii. From CONVERSION_NOTES.md: "Sessions: the 44 reference ALM ephys sessions (25 fixed-delay + 19 randomized-delay), transcribed into `/app/cache/meta_sessions.py`"; "Ephys sessions = files NOT containing 'MAH'. MAH sessions are behavior+video only (bilateral MC inhibition) -> no neural data -> cannot be used for a neural decoder." The Step 4 consistency table records that 25 uncommented fixed-delay entries and 19 uncommented randomized-delay entries exactly reproduce the paper's "25 sessions" and "19 sessions", whereas the folders hold 25 and 22 files respectively (JEB23 2023-10-20 is commented out in the loading script; JEB24 2023-10-03/10-04 have no sorted clusters, a 'DUMMY' probe and no motion-energy file). "MAT file versions are mixed: 36 data_structure files are MAT v7.3 (HDF5, read with `h5py`), 11 are MAT v7 (read with `scipy.io.loadmat`) ... A unified loader (`Session` class) was written for both."

## 1-b. How are the data split into subjects?

i. The animal identifier is an explicit field of each hard-coded session tuple (identical to the prefix of the file name, e.g. `JEB19_2023-04-19` -> `JEB19`). At assembly the AI builds `subjects` as the list of unique animals **in order of first appearance** over the session list, and `subject_idx[session] = subjects.index(animal)`. This yields 14 subjects over 44 sessions (JEB6 1, JEB7 2, EKH1 1, EKH3 1, JGR2 2, JGR3 1, JEB13 5, JEB14 4, JEB15 4, JEB19 4, JEB11 2, JEB12 2, JEB23 7, JEB24 8).

ii.
```python
    d, anm, date, probes = entry
    sess_id = f'{anm}_{date}'
    ...
    res = dict(session_id=sess_id, animal=anm, date=date, ...)
```

```python
        if r['animal'] not in subjects:
            subjects.append(r['animal'])
        ...
        data['subject_idx'].append(subjects.index(r['animal']))
    data['subjects'] = subjects
    data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. CONVERSION_NOTES.md Step 5: "`subjects` = the 14 animals; `subject_idx` per session." The animal id is taken from the authors' own meta scripts / file names rather than from inside the `.mat` file; Step 2 notes that 14 animals have ephys data (EKH1, EKH3, JEB6, JEB7, JEB11-15, JEB19, JEB23, JEB24, JGR2, JGR3). Step 4 documents that the paper's "nine mice" for the fixed-delay task corresponds to `NullPotent/SessionMeta.csv` (which omits EKH1), while the 25-session loading-script list the paper's session count matches contains 10 animals; the AI kept all 25 sessions / 10 fixed-delay animals and documented the discrepancy.

## 1-c. How are the data split into sessions?

i. One session = one entry of `SESSIONS` = one `data_structure` file = one element of `neural`/`input`/`output`/`brain_region_idx`/`subject_idx`. The folder is part of the tuple, so the fixed-delay and randomized-delay experiments are pooled into one 44-session dataset rather than treated as two datasets; the task (`'fixed'`/`'randomized'`) is recorded per session in `metadata['session_info']`. A session is dropped only if it raises an error, has fewer than 10 curated units, or fewer than 2 kept trials; none of the 44 is dropped. Sessions recorded with two probes have the units of the listed probes concatenated into one population.

ii.
```python
def data_path(entry):
    d, anm, date, _ = entry
    return f'{DATA_ROOT}/{d}/data_structure_{anm}_{date}.mat'
```

```python
        if r['nunits'] < MIN_UNITS:
            print(f"  SKIP {r['session_id']}: only {r['nunits']} units (< {MIN_UNITS})")
            continue
        if r['ntrials_kept'] < 2:
            print(f"  SKIP {r['session_id']}: only {r['ntrials_kept']} trials")
            continue
```

```python
        session_info.append(dict(session_id=r['session_id'], animal=r['animal'], date=r['date'],
                                 task=r['group'], nunits=r['nunits'],
                                 ntrials=r['ntrials_kept'], ntrials_raw=r['ntrials_total'], ...))
```

iii. CONVERSION_NOTES.md Step 5 decision 9: "**Session curation**: keep a session only if it has >= 10 curated units (paper criterion) and at least 2 usable trials"; the 10-unit rule is quoted from the methods ("Recording sessions were included for analysis only if they had at least 10 units") and the 2-trial rule from the target-format requirement ("There needs to be at least two trials within each session"). Step 9 records "in practice all 44 pass".

## 1-d. How are the data split into trials?

i. A trial is a row of the Bpod table: the number of trials is `obj.bp.Ntrials`, and every per-trial field (`hit`, `miss`, `no`, `early`, `autowater`, `stim.enable`, `ev.goCue`, `ev.lickL`, `ev.lickR`) is read as a vector/cell of that length (fields are truncated with `[:ntrials]` because some are stored longer). Spikes carry their trial number in `clu.trial` (1-based, converted with `tr - 1`), and video/motion-energy data are stored as one entry per trial, so no trial boundaries need to be reconstructed. Trial indices are kept in the original 0-based numbering throughout (`trials = np.where(keep)[0]`) and used to index back into every raw array.

ii.
```python
    ntrials = s.ntrials
    gocue = s.ev(ALIGN_EVENT)
    ...
    keep = (stim[:ntrials] == 0) & (early[:ntrials] == 0) & np.isfinite(gocue[:ntrials])
    keep &= (hit[:ntrials] + miss[:ntrials] + no[:ntrials]) > 0
    trials = np.where(keep)[0]
```

```python
    @property
    def ntrials(self):
        if self.hdf5:
            return int(np.asarray(self.obj['bp/Ntrials'][()]).ravel()[0])
        return int(self.obj.bp.Ntrials)
```

```python
        ok = (tr >= 1) & (tr <= ntrials) & np.isfinite(tt)
        tr, tt = tr[ok], tt[ok]
        aligned = tt - gocue[(tr - 1).astype(int)]     # trialtm_aligned
```

iii. CONVERSION_NOTES.md Step 2 documents the `obj.bp` layout ("`Ntrials`, `hit`, `miss`, `no`, `early`, `autowater`, `R`, `L`, `stim.enable` ... `obj.bp.ev`: `bitStart`, `sample`, `delay`, `goCue`, `reward` (per trial, seconds within trial), `lickL`, `lickR` (cell arrays of lick contact times)") and `obj.clu` fields ("`trialtm` (time within trial), `trial` (trial number)"). Trial splitting therefore follows the reference's `findTrials.m`, which evaluates condition strings over the same per-trial `obj.bp` fields.

## 1-e. How are trials filtered based on quality controls?

i. Four filters, applied before anything is computed except the go-cue read:
1. photoinactivation trials dropped (`bp.stim.enable != 0`), as in every reference condition string `'~stim.enable'`;
2. early-lick trials dropped (`bp.early != 0`), as in `'~early'` and in the methods;
3. trials with a non-finite go cue dropped (alignment impossible);
4. trials that are none of hit/miss/ignore dropped (`hit+miss+no == 0`).
NaNs in any of these Bpod flags are coerced to 0 first. A fifth filter is applied after the neural data are built: trials in which **no** curated unit fires a single spike anywhere in the 5 s window are dropped, because in JEB24_2023-10-23 (trials 292-319) and JEB24_2023-11-03 (trials 301-333) the probe recording ended while Bpod kept running. **Ignore trials are deliberately retained** (they are a required output class). The result is 13,762 kept trials of 15,062 (193-474 per session).

ii.
```python
    stim = s.bpvec('stim/enable') if s.has_bp_field('stim') else np.zeros(ntrials)
    early = s.bpvec('early')
    hit = s.bpvec('hit')
    miss = s.bpvec('miss')
    no = s.bpvec('no')
    autowater = s.bpvec('autowater')
    for v in (stim, early, hit, miss, no, autowater):
        v[np.isnan(v)] = 0
    keep = (stim[:ntrials] == 0) & (early[:ntrials] == 0) & np.isfinite(gocue[:ntrials])
    # a trial must be one of hit / miss / ignore
    keep &= (hit[:ntrials] + miss[:ntrials] + no[:ntrials]) > 0
    trials = np.where(keep)[0]
```

```python
    # Trials in which no curated unit fired a single spike in the whole 5 s window
    # have no ephys coverage (in 2 JEB24 sessions the recording stopped before the
    # behavioural session ended); drop them.
    if nunits > 0:
        has_spikes = rates[trials].sum(axis=(1, 2)) > 0
        n_nospk = int((~has_spikes).sum())
        trials = trials[has_spikes]
```

iii. CONVERSION_NOTES.md Step 5 decision 6: "exclude photoinactivation trials (`bp.stim.enable`) and early-lick trials (`bp.early`), exactly as every reference condition string does ('~stim.enable&~early') and as stated in the methods ... **Ignore trials are kept** because 'ignore' and 'none' are required output classes." The zero-spike filter was added in Step 9/10 in response to 61 verification warnings: "First full run produced 61 warnings 'all neural data is zero' in JEB24_2023-10-23 (trials 292-319) and JEB24_2023-11-03 (trials 301-333). These are contiguous **trailing** trials in which the ephys recording had already stopped while Bpod kept running. Fixed by dropping trials in which no curated unit fires a single spike in the whole 5 s window." Step 10 Check 5 reports "for all 44 sessions, kept trials == (non-stim, non-early, hit|miss|no) minus no-ephys trials. **0 mismatches.**"

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` — the spike-sorted clusters of the probe(s) listed for that session in the authors' meta scripts. Three fields per cluster are used: `trial` (1-based trial number of each spike), `trialtm` (spike time relative to that trial's start) and `quality` (manual curation label). `obj.bp.ev.goCue` supplies the alignment times, and `obj.ex.probe.loc` supplies the anatomical label of each probe ('R/L ALM' -> `ALM`, '*M1TJ*' -> `tjM1`, missing/'DUMMY' -> `ALM`). Clusters from the two probes of a dual-probe session are concatenated into one population, with their region recorded per neuron in `brain_region_idx`.

ii.
```python
    def clusters(self, prb):
        """List of dicts (quality, trial, trialtm) for 0-based probe index prb."""
        ...
            qual, trial, trialtm = o['quality'][()].ravel(), o['trial'][()].ravel(), o['trialtm'][()].ravel()
            for i in range(len(qual)):
                out.append(dict(quality=_h5str(self.f, qual[i]),
                                trial=np.asarray(self.f[trial[i]][()]).ravel(),
                                trialtm=np.asarray(self.f[trialtm[i]][()]).ravel()))
```

```python
    for p in probes:
        loc_list = s.probe_locs()
        loc = loc_list[p - 1] if len(loc_list) >= p else ''
        reg = region_from_loc(loc)
        for c in s.clusters(p - 1):
            q = c['quality'].strip().lower()
            if q in BAD_QUALITY:
                continue                      # findClusters.m with quality = 'all'
            clusters.append(c)
            qualities.append(q)
            cl_region.append(reg)
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "`obj.clu{alm}.trialtm`, `.trial` -> `neural` | align to go cue (subtract `bp.ev.goCue`), bin -2.5..2.5 s at dt=30 ms, rate = count/dt, causal-Gaussian smooth ... alignSpikes.m, getSeq.m, mySmooth.m | firing rate in spikes/s, float32". Decision 5 explains probe selection and region labelling: "use exactly the probe(s) listed in the reference meta scripts (`load<ANM>_ALMVideo.m`). These are the probes the paper analysed ... for JEB15 2022-07-26/27/28 the reference uses probes [1 2] where probe 2 is 'L M1TJ' ... those neurons are labelled tjM1 so that `brain_region_idx` is truthful, rather than dropping data the reference analysed." Final counts: 2,312 ALM + 145 tjM1 = 2,457 neurons.

## 2-b. How is the `neural` data processed?

i. Aligned spike times are counted into the 30 ms bin grid with a single `np.histogram2d` over (trial, aligned time) per cluster, divided by the bin width to give spikes/s, then smoothed along time by a **causal** Gaussian kernel: `gausswin(15)` with its first half zeroed and renormalised, with the reference's `'reflect'` boundary handling (prepend the first 15 samples, convolve, trim). Nothing else is done: no z-scoring, no baseline subtraction, no normalisation; stored values are firing rates in Hz as `float32`. Smoothing is applied to all trials at once with `scipy.signal.fftconvolve`.

ii.
```python
def gausswin(N, alpha=2.5):
    """MATLAB gausswin(N, alpha)."""
    n = np.arange(N) - (N - 1) / 2.0
    return np.exp(-0.5 * (alpha * n / ((N - 1) / 2.0)) ** 2)


def causal_kernel(N=SMOOTH):
    """mySmooth.m kernel: gausswin with the first half zeroed (causal), normalized."""
    k = gausswin(N)
    k[:N // 2] = 0.0
    return k / k.sum()


def mysmooth(x, N=SMOOTH, bctype='reflect'):
    """Port of utils/mySmooth.m; smooths along axis 0."""
    ...
    if bctype == 'reflect':
        xf = np.concatenate([x[:N], x], axis=0)
        trim = N
    ...
    out = fftconvolve(xf, k.reshape((-1,) + (1,) * (xf.ndim - 1)), mode='same', axes=0)
    return out[trim:]
```

```python
        cnt, _, _ = np.histogram2d(tr, aligned, bins=[trial_edges, edges])
        meanfr[i] = cnt.sum() / (ntrials * window)
        rates[:, :, i] = mysmooth(cnt.T / DT).T.astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 5 decision 3: "**Causal Gaussian smoothing, N=15 bins, 'reflect'** exactly as `mySmooth` — causal so no information from the future leaks backwards in time (important for a decoder)." Step 1 records the reference chain `alignSpikes -> getSeq -> mySmooth` and `params.smooth = 15`, `bctype='reflect'`. Step 10 Check 2 verified the rate of 9 (trial, neuron) pairs against an independent from-scratch `np.convolve` recomputation from the raw spike times (`np.allclose`, PASS), and Check 3 records binning/smoothing as identical to `getSeq.m`. The addendum reports an empirical check that the chosen (dt=30 ms, smooth=15) setting beat (dt=10 ms, smooth=15) and (dt=30 ms, smooth=7) on every decoder output.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Three filters. (1) Manual curation label: the label is stripped and lower-cased and the cluster is dropped if it is in `{garbage, gabrga, noisy, real?, ''}` — i.e. the `findClusters.m` `quality='all'` drop list plus unlabelled clusters; everything else is kept, including `poor`, `fair` and `multi`. (2) Mean firing rate: a unit is kept only if its mean rate over the whole −2.5..2.5 s window, computed from the **raw unsmoothed** counts pooled over all trials, exceeds 1 Hz (`removeLowFRClusters.m` with `params.lowFR = 1`; paper: "all units with firing rates exceeding 1 Hz"). (3) Probe selection: only clusters on the probe(s) listed in the authors' meta scripts. A session-level filter (>=10 units) follows. Result: 2,457 units of ~14,300 raw clusters, 17-141 per session, mean 55.8.

ii.
```python
LOW_FR = 1.0          # params.lowFR (spikes/s); paper: "units with firing rates exceeding 1 Hz"
MIN_UNITS = 10        # paper: "sessions were included ... only if they had at least 10 units"
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?', ''}   # findClusters.m, quality='all'
```

```python
            q = c['quality'].strip().lower()
            if q in BAD_QUALITY:
                continue                      # findClusters.m with quality = 'all'
```

```python
    # removeLowFRClusters.m: mean rate must exceed lowFR (1 spk/s)
    keep_units = meanfr > LOW_FR
    rates = rates[:, :, keep_units]
    cl_region = [r for r, k in zip(cl_region, keep_units) if k]
```

iii. CONVERSION_NOTES.md Step 5 decision 4: "quality not in {garbage, gabrga, noisy, real?, empty} (findClusters, case-insensitive/trimmed to handle label case and typos) AND mean firing rate over the trial window > 1 Hz (removeLowFRClusters with `lowFR=1`, matching the paper's 'units with firing rates exceeding 1 Hz'). Sessions with < 10 units would be dropped (paper criterion); in practice all 44 pass." Step 2 justifies case-insensitive matching: "Quality strings differ in case/padding between sessions ('Poor' vs 'poor   '), and include typos". Step 9 notes the change to computing the FR criterion from raw counts over all trials "matching `removeLowFRClusters.m`, whose first PSTH condition `(hit|miss|no)` spans every trial (changing this altered only 1 unit in total)". Step 9's consistency table is honest that the final counts miss the paper's: 1,532 vs 1,651 fixed-delay units and 925 vs 845 randomized-delay units, with the explanation that only 1,565 clusters on the meta ALM probes survive the quality filter at all, so 1,651 is unreachable from the shared session/probe list.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to the go cue is one subtraction, exactly as in `alignSpikes.m`: `clu.trialtm` is already on the behaviour clock relative to its own trial's start, and `bp.ev.goCue` is on the same clock, so `aligned = trialtm − goCue[trial]` gives seconds from go-cue onset with no interpolation or offset. Spikes with out-of-range trial numbers or non-finite times are discarded first; spikes outside the window fall off the edges of the `histogram2d` grid. The go cue field is used unchanged in both contexts (on water-cued trials `bp.ev.goCue` holds the water-drop time).

ii.
```python
ALIGN_EVENT = 'goCue'
...
        aligned = tt - gocue[(tr - 1).astype(int)]     # trialtm_aligned
        cnt, _, _ = np.histogram2d(tr, aligned, bins=[trial_edges, edges])
```

iii. CONVERSION_NOTES.md Step 1: "alignSpikes | `trialtm_aligned = clu.trialtm - obj.bp.ev.(alignEvent)(clu.trial)`", with `params.alignEvent = 'goCue'`. Step 5 decision 1: "**Alignment to go cue** (`params.alignEvent='goCue'`), window **-2.5 to +2.5 s**, exactly as in every reference analysis script. In WC trials `bp.ev.goCue` holds the water-drop time, so the alignment is meaningful in both contexts." Step 7 gives a sanity check on the result: "Population firing rate peaks at t = +0.22 s / +0.19 s after the go cue (response-epoch activity) -> neural alignment correct."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 30 ms bins (`DT = 0.03`, `params.dt = (1/100)*3` from the paper's Figure 3 scripts), giving **166 bins per trial** for every trial of every session. The grid is built exactly as `getSeq.m` does: `edges = tmin:dt:tmax` and `time = edges + dt/2` with the last element dropped. Because 5 s is not an integer multiple of 30 ms, the edges run −2.5 to +2.48 s and the bin centres from −2.485 to +2.465 s (so the last ~20 ms of the nominal window is not covered, although `metadata['off_end']` is reported as 2.5). Spikes are binned directly at this resolution — there is no rebinning of a finer representation; the video and motion-energy streams are averaged/interpolated onto the same 166-bin axis, so all streams share one time axis. Neural smoothing is 15 bins of causal Gaussian, i.e. 450 ms at this bin width.

ii.
```python
TMIN = -2.5           # params.tmin
TMAX = 2.5            # params.tmax
DT = 0.03             # params.dt = (1/100)*3  (30 ms bins)
SMOOTH = 15           # params.smooth (causal gaussian kernel width in bins)
```

```python
def time_axis():
    """getSeq.m: edges = tmin:dt:tmax; time = edges + dt/2 with the last dropped."""
    nedges = int(np.floor((TMAX - TMIN) / DT)) + 1
    edges = TMIN + DT * np.arange(nedges)
    t = edges + DT / 2.0
    return edges, t[:-1]
```

```python
        time_bin_size=DT * 1000.0,
        ...
        off_start=TMIN, off_end=TMAX,
```

iii. CONVERSION_NOTES.md Step 5 decision 2: "**Bin size 30 ms** = `params.dt = (1/100)*3` used in the paper's Figure 3 analysis scripts (other scripts use 5-10 ms; the decoding analyses re-bin to 75 ms). 30 ms keeps ~167 bins/trial, enough temporal resolution for licking (~7 Hz) while keeping the dataset tractable (~2.3M timepoints)." Step 3 records the ambiguity in the reference ("`dt` = 5-30 ms depending on script ... Decoding analyses re-bin neural data to **75 ms**"). The addendum adds an empirical justification: converting the two sample sessions at dt=10 ms and at smooth=7 gave lower validation accuracy on **every** output than dt=30 ms / smooth=15, "so the reference parameters were kept unchanged."

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Nothing in the raw data — it is defined by the conversion as the bin-centre time axis of the alignment window (`obj.time` in the reference), i.e. the centres of the 166 30 ms bins spanning −2.5..+2.48 s around `bp.ev.goCue`. The same vector is used for every trial of every session and is also stored in `metadata['time_axis']`.

ii.
```python
    edges, taxis = time_axis()
    nbins = taxis.size
    ...
    tin = taxis.astype(np.float32).reshape(1, -1)
    for i in trials:
        ...
        inp.append(tin.copy())
```

```python
                input_names=['time_from_go_cue'],
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "time axis -> `input[0]` = `time_from_gocue` | obj.time = bin centers, seconds | getSeq.m | continuous, time-varying (1 x T)". The decoder task specification asks for "Time from go cue onset in seconds (continuous, time-varying)", so the quantity is the conversion's own time base, not a measured variable.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond constructing the axis: `edges = tmin:dt:tmax`, `t = edges + dt/2`, drop the last element, cast to `float32`, reshape to `(1, 166)` and copy once per trial. Values run −2.485 to 2.465 s.

ii.
```python
def time_axis():
    """getSeq.m: edges = tmin:dt:tmax; time = edges + dt/2 with the last dropped."""
    nedges = int(np.floor((TMAX - TMIN) / DT)) + 1
    edges = TMIN + DT * np.arange(nedges)
    t = edges + DT / 2.0
    return edges, t[:-1]
```

iii. CONVERSION_NOTES.md Step 10 Check 2 lists two passing `np.allclose` sanity checks on it: "input time axis == bin centres of `tmin:dt:tmax` | all | PASS" and "input identical for every session/trial | 0, 10, 43 | PASS". Step 10 Check 5: "Time axis: -2.485 .. 2.465 s, dt = 0.03 s to float32 precision".

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid: the input is the centre of the same `edges` array used to histogram the go-cue-aligned spike times, so bin *k* of the input and bin *k* of the neural matrix denote the same 30 ms interval. The same `edges`/`taxis` pair is also used to bin the tongue/paw frames (`np.digitize(vt, bin_edges)`) and to interpolate motion energy, so all five streams are on one axis by construction.

ii.
```python
    edges, taxis = time_axis()
    ...
        cnt, _, _ = np.histogram2d(tr, aligned, bins=[trial_edges, edges])
        rates[:, :, i] = mysmooth(cnt.T / DT).T.astype(np.float32)
    ...
        idx = np.digitize(vt0, bin_edges) - 1               # bin index for each frame
    ...
                y = interp_to_axis(vt0[:n], mev[:n], taxis)
```

iii. CONVERSION_NOTES.md Step 10 Check 3(e): "input construction | `obj.time` used as the time axis everywhere | `input[0] = obj.time` | YES". No separate alignment step is needed because the axis is shared.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The recorded lickport contact times, `obj.bp.ev.lickL` and `obj.bp.ev.lickR` (cell arrays, one array of contact times per trial), together with `bp.ev.goCue` to define "after the go cue". The outcome flags are *not* used for this output. This follows the reference helper `firstLickTime.m`.

ii.
```python
    lickL, lickR = s.ev_cell('lickL'), s.ev_cell('lickR')
    lick_dir = np.full(ntrials, 2, dtype=np.int64)          # 2 = none
    for i in range(ntrials):
        tl = np.concatenate([lickL[i] - gocue[i], lickR[i] - gocue[i]])
        side = np.concatenate([np.zeros(lickL[i].size), np.ones(lickR[i].size)])
```

```python
    def ev_cell(self, name):
        out = []
        if self.hdf5:
            for r in self.obj['bp/ev/' + name][()].ravel():
                d = self.f[r]
                if d.attrs.get('MATLAB_empty', 0) == 1:
                    out.append(np.array([]))
                else:
                    out.append(np.asarray(d[()], dtype=float).ravel())
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "first lick after go cue (`bp.ev.lickL/lickR`) -> `output[0]` = `lick_direction` | 0=left, 1=right, 2=none | firstLickTime.m". Step 4 justifies the choice against the indirect alternative: "firstLickTime.m: first lickport contact after go cue | first post-go-cue lick side agrees 100% with (hit & R -> right, miss & R -> left, no -> none) on all tested sessions | consistent; use the actual first lick."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. For each trial, left and right contact times are pooled, expressed relative to the go cue, contacts at or before the go cue are discarded (`tl > 0`, matching `firstLickTime.m`'s `licktimes(licktimes <= 0) = []`), and the side of the earliest remaining contact is taken: 0 = left, 1 = right. Trials with no post-go-cue contact get 2 = none. The per-trial code is then broadcast across all 166 bins so the output is time-varying in shape, constant within a trial. Resulting fractions: left 0.423, right 0.446, none 0.130.

ii.
```python
    for i in range(ntrials):
        tl = np.concatenate([lickL[i] - gocue[i], lickR[i] - gocue[i]])
        side = np.concatenate([np.zeros(lickL[i].size), np.ones(lickR[i].size)])
        m = tl > 0                                           # firstLickTime.m keeps post-go-cue licks
        if m.any():
            lick_dir[i] = int(side[m][np.argmin(tl[m])])     # 0 = left, 1 = right
```

```python
        o = np.empty((6, nbins), dtype=np.int64)
        o[0] = lick_dir[i]
```

```python
                output_values=[['left', 'right', 'none'], ...
```

iii. CONVERSION_NOTES.md Step 4 (agreement with the outcome-derived labels in 100% of tested trials) and Step 10 Check 2: "`lick_direction` == side of the first lickport contact after the go cue | 3 sessions | PASS" and "per-trial outputs constant across time | 3 | PASS". Step 7 adds "100% of correct trials have a lick; 100%/98% of ignore trials have lick_direction = none -> behavioural outputs correct." The value coding follows the prompt's "(left, right, none, per-trial)".

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial Bpod flag, `obj.bp.autowater`, which marks the water-cued (WC) trials where water is delivered at a random port with no cues. NaNs are coerced to 0.

ii.
```python
    autowater = s.bpvec('autowater')
    for v in (stim, early, hit, miss, no, autowater):
        v[np.isnan(v)] = 0
```

iii. CONVERSION_NOTES.md Step 1: "`autowater` == water-cued (WC) context; `~autowater` == delayed-response (DR) context" (read off the reference condition strings). Step 5 mapping table: "`bp.autowater` -> `output[1]` = `context` | 0=WC (autowater=1), 1=DR | params.condition strings".

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling: `autowater > 0` -> 0 (WC), otherwise 1 (DR), broadcast across the 166 bins. Overall fractions WC 0.097 / DR 0.903; per session WC ranges 0 to 0.39, and the randomized-delay sessions are essentially pure DR.

ii.
```python
    context = np.where(autowater[:ntrials] > 0, 0, 1).astype(np.int64)   # 0 = WC, 1 = DR
    ...
        o[1] = context[i]
```

```python
                output_values=[..., ['WC', 'DR'], ...
```

iii. Coding follows the prompt's "(WC, DR, per-trial)". CONVERSION_NOTES.md Step 10 Check 2: "`context` == `bp.autowater` | 3 sessions | PASS". Step 9 checks the distribution against the paper's block structure: "~100 DR trials then alternating blocks | autowater flag | WC fraction 0-39% per session | WC 0.097, DR 0.903 | YES (12/25 fixed-delay sessions have >5% WC trials; randomized-delay sessions are almost pure DR)".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Three mutually exclusive per-trial Bpod flags: `obj.bp.miss` (incorrect), `obj.bp.hit` (correct) and `obj.bp.no` (ignore). `bp.no` is read both for the trial filter (a trial must be one of the three) and implicitly as the default class.

ii.
```python
    hit = s.bpvec('hit')
    miss = s.bpvec('miss')
    no = s.bpvec('no')
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "`bp.hit/miss/no` -> `output[2]` = `outcome` | 0=incorrect(miss), 1=correct(hit), 2=ignore(no) | getOutcome.m". Step 1 records `getOutcome.m` as "outcome = bp.hit, NaN where bp.no (ignore)", so the AI extends the reference's binary outcome with the explicit ignore class the prompt requires.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Initialise everything to 2 (ignore), set miss trials to 0 (incorrect), then hit trials to 1 (correct) — so hit wins if flags ever overlapped. Broadcast across the 166 bins. Fractions: incorrect 0.120, correct 0.749, ignore 0.131 (per session hit fraction 0.53-0.94).

ii.
```python
    outcome = np.full(ntrials, 2, dtype=np.int64)            # 2 = ignore
    outcome[miss[:ntrials] > 0] = 0                          # 0 = incorrect
    outcome[hit[:ntrials] > 0] = 1                           # 1 = correct
    ...
        o[2] = outcome[i]
```

```python
                output_values=[..., ['incorrect', 'correct', 'ignore'], ...
```

iii. Coding follows the prompt's "(incorrect, correct, ignore, per-trial)". CONVERSION_NOTES.md Step 5 decision 6 explains why ignore trials are retained rather than dropped as in the paper's analyses: "'ignore' and 'none' are required output classes". Step 10 Check 2: "`outcome` == `bp.hit`/`miss`/`no` | 3 | PASS"; Step 9 compares to the paper's training criterion: "trained animals > 70% correct ... correct 0.749 | YES (75% correct > 70% criterion)".

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj{1}` — the **side camera only** — feature `tongue`: its per-frame x and y from `ts[:, :2, i_tongue]`, plus that camera's `frameTimes` and `NdroppedFrames`. The bottom-camera tongue markers (`top_tongue` etc.) are not used. `obj.sglx.bitcode.bitstart`, `obj.sglx.fs` and `bp.ev.bitStart` are needed for the clock correction, and `bp.ev.goCue` for the alignment. Visibility is taken from the coordinates themselves: the authors already store NaN for x and y wherever DLC likelihood is low, so `isfinite(x) & isfinite(y)` is the visibility mask (the likelihood channel `ts[:, 2, :]` is not thresholded separately).

ii.
```python
    fn0, fn1 = s.feat_names(0), s.feat_names(1)
    i_tongue = fn0.index('tongue')
```

```python
        # --- tongue (side cam) ---
        xy = ts0[:, :2, i_tongue]
        vis = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
        spd = speed_from_xy(xy) / dtf                        # pixels / s
        _accumulate(idx, inb, spd, vis, tongue_spd, tongue_vis, i, nbins)
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "side-cam `tongue` x/y (`obj.traj{1}`) -> `output[3]` = `tongue_velocity` | speed=|d(x,y)/dt| of interpolated position; 0 if < session median, 1 if >= median, 2 if tongue not visible (DLC NaN) | findPosition.m, findVelocity.m". Step 2 lists the available features per view, and Step 5 decision 7 states that "Kinematics NaNs are kept as the 'not visible' class instead of being filled/zeroed as in `findPosition`/`findVelocity`", because the decoder spec requires an explicit 'not visible' category. `tongue` is the first entry of `params.traj_features{1}` in every reference script. The notes do not explicitly argue against also using the bottom-camera tongue markers.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps. (1) Frames where x or y is NaN (DLC low likelihood) are marked invisible. (2) Within each **contiguous run of visible frames**, x and y are differentiated with `np.gradient` (one-sided at run ends) and the speed is `hypot(vx, vy)`, then divided by the trial's median frame interval to give pixels/s; positions are not smoothed (matching `findPosition.m`, which skips smoothing for tongue features). A run of length 1 gets speed 0. (3) Frames are mapped to bins with `np.digitize` on the shared `edges` and the per-frame speeds are **averaged within each bin** by `np.bincount`; a bin is marked visible if it contains at least one visible frame, otherwise it is NaN/not-visible. (4) The binned speed is discretised at the session median (7-c). No cross-camera normalisation is needed since only one view is used; values stay in raw pixels/s.

ii.
```python
def speed_from_xy(xy):
    """Speed of a DLC feature, computed like findVelocity.m (gradient of position),
    but evaluated separately within each contiguous run of frames in which the
    feature is visible.
    ...
    """
    vis = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
    spd = np.full(xy.shape[0], np.nan)
    ...
    for a, b in zip(starts, stops):
        seg = idx[a:b + 1]
        if seg.size == 1:
            spd[seg] = 0.0                      # single visible frame: no motion estimate
            continue
        vx = np.gradient(xy[seg, 0])
        vy = np.gradient(xy[seg, 1])
        spd[seg] = np.sqrt(vx ** 2 + vy ** 2)
    return spd
```

```python
def _accumulate(idx, inb, spd, vis, spd_out, vis_out, trial, nbins):
    """Average per-frame speed within each time bin; a bin is visible if any frame is."""
    good = inb & vis & np.isfinite(spd)
    if good.any():
        sums = np.bincount(idx[good], weights=spd[good], minlength=nbins)
        cnts = np.bincount(idx[good], minlength=nbins)
        ...
        spd_out[trial] = vals
        vis_out[trial] = cnts > 0
```

iii. CONVERSION_NOTES.md Step 10 Check 2 documents that the run-wise gradient was a **bug fix** found by the sanity checks: "the initial `speed_from_xy` used `np.gradient` over the whole trial, so visible frames adjacent to a NaN frame received a NaN speed and were mislabelled 'not visible' (tongue 'not visible' was 0.845 -> 0.826 after the fix; ~2% of visible timepoints were affected). Fixed by computing the gradient separately inside each contiguous run of visible frames." Step 10 Check 3(f) records the deliberate differences from the reference: "(i) DLC NaNs become the 'not visible' class instead of being filled ..., (ii) speed = |velocity| rather than separate x/y velocities, discretized at the session median (required by the decoder spec)".

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A single per-session threshold: the 50th percentile of the binned tongue speed pooled over **all kept trials and all bins in which the tongue is visible** (invisible bins are excluded from the percentile). Bins with speed < threshold -> 0, >= threshold -> 1, invisible (or outside video coverage, or trial without usable video) -> 2. The threshold is recorded per session in `metadata['session_info'][i]['thresholds']`. Result: 0.087 / 0.087 / 0.826 overall — the two visible classes are equal to three decimals, as a median split requires.

ii.
```python
    sel = np.zeros(ntrials, dtype=bool)
    sel[trials] = True
    m_t = sel[:, None] & tongue_vis & np.isfinite(tongue_spd)
    thr_t = np.nanpercentile(tongue_spd[m_t], 50) if m_t.any() else np.nan
    ...
    tongue_cls = discretize(tongue_spd, tongue_vis, thr_t)
```

```python
def discretize(values, visible, thresh):
    """0 = below threshold, 1 = at/above threshold, 2 = not visible / no video."""
    out = np.full(values.shape, 2, dtype=np.int64)
    v = visible & np.isfinite(values)
    out[v & (values < thresh)] = 0
    out[v & (values >= thresh)] = 1
    return out
```

iii. CONVERSION_NOTES.md Step 5 decision 8: "**Discretization thresholds are per session** (median over all included trials and timepoints where the feature is visible), as specified in the decoder task." Step 7: "Discretization: class 0 and class 1 fractions are equal to 3 decimals for every discretized output (median split), as expected"; Step 10 Check 2: "discretized outputs are balanced 50/50 between class 0 and 1 | 9 | PASS". Step 9 checks the not-visible fraction against the raw data: "tongue visible only during licking | DLC NaN = not visible | ~89% NaN over the whole trial | not visible 0.845 | YES" (0.826 after the run-wise-gradient fix).

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Two corrections, then the shared grid. First the video clock is put on the behaviour clock with the reference's `findVideoOffset.m` constant, computed once per session: `vidshift = mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)` (~0.49-0.50 s; NaNs removed from `bitstart` before taking the mode). Then the trial's go cue is subtracted: `vt0 = frameTimes − vidshift − goCue[trial]`. Frames are assigned to bins of the same `edges` array used for the spikes with `np.digitize`, and frames falling outside the window are dropped. The side camera's own `frameTimes` are used, since the tongue comes from the side camera.

ii.
```python
    def vidshift(self):
        """funcs/findVideoOffset.m: mode(sglx.bitcode.bitstart)/fs - mode(bp.ev.bitStart)"""
        from scipy.stats import mode as _mode
        bitStart = _mode(self.ev('bitStart'), keepdims=False).mode
        ...
        bs = bs[~np.isnan(bs)]
        return float(_mode(bs, keepdims=False).mode / fs - bitStart)
```

```python
        # video clock -> time from the alignment event
        vt0 = ft0 - vidshift - gocue[i]
        dtf = np.median(np.diff(ft0)) if ft0.size > 1 else 1 / 400.0
        has_video[i] = True
        idx = np.digitize(vt0, bin_edges) - 1               # bin index for each frame
        inb = (idx >= 0) & (idx < nbins)
```

iii. CONVERSION_NOTES.md Step 5 decision 10: "**Video/ephys synchronisation** uses `vidshift` from `findVideoOffset.m` for both DLC and motion energy, as in the reference." Step 4 confirms the magnitude across sessions ("~0.49-0.50 s for all sessions"). Step 7 gives the alignment sanity check: "Tongue visible 5% of the time before the go cue vs 28-36% after, peaking at +0.16/+0.25 s -> DLC alignment correct." Step 10 Check 3(c): "alignment | `alignSpikes.m`: `trialtm - bp.ev.goCue(trial)`; video: `frameTimes - vidshift - goCue`; ME: same | identical formulas ... | YES".

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The DeepLabCut tracking in `obj.traj{2}` — the **bottom camera** — features `top_paw` **and** `bottom_paw` (whichever are present in `featNames`), their x/y from `ts1[:, :2, j]`, plus the bottom camera's own `frameTimes` and `NdroppedFrames`. Visibility again comes from `isfinite(x) & isfinite(y)`.

ii.
```python
    i_paws = [fn1.index(f) for f in ('top_paw', 'bottom_paw') if f in fn1]
```

```python
            spds, viss = [], []
            for j in i_paws:
                pxy = ts1[:, :2, j]
                spds.append(speed_from_xy(pxy) / dtf1)
                viss.append(np.isfinite(pxy[:, 0]) & np.isfinite(pxy[:, 1]))
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "bottom-cam `top_paw`,`bottom_paw` (`obj.traj{2}`) -> `output[4]` = `paw_velocity` | mean speed of the visible paw markers; 0/1 by session median, 2 if neither paw visible | findPosition.m, findVelocity.m". Both markers are the paw features listed in the reference's `getDefaultParams.m` `params.traj_features{2}`. Step 2 notes the observed DLC NaN rate for the paw ("paw NaN 0.5-9%").

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Identical to the tongue: per-marker run-wise `np.gradient` of x and y over contiguous visible frames, `hypot` to get speed, divide by the bottom camera's median frame interval to get pixels/s. The two markers are then combined per frame with `np.nanmean` (so a frame where only one paw is tracked uses that paw alone) and the frame is visible if **either** marker is visible. The combined per-frame speed is averaged into the 30 ms bins with `np.bincount` and discretised at the session median. No normalisation.

ii.
```python
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                spd_p = np.nanmean(np.vstack(spds), axis=0)
            vis_p = np.any(np.vstack(viss), axis=0)
            _accumulate(idx1, inb1, spd_p, vis_p, paw_spd, paw_vis, i, nbins)
```

iii. Same justification as 7-b (CONVERSION_NOTES.md Step 5 decision 7, Step 10 Check 3(f) item (iii) "paw speed = mean over the two bottom-cam paw markers"). The 'not visible' class is retained rather than `fillmissing('nearest')` as the reference does, because the decoder spec requires the explicit class.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same rule as the tongue: per-session 50th percentile of the binned paw speed over kept trials and visible bins; < threshold -> 0, >= -> 1, no visible frame in the bin (or trial without usable video) -> 2. Result 0.477 / 0.477 / 0.046.

ii.
```python
    m_p = sel[:, None] & paw_vis & np.isfinite(paw_spd)
    thr_p = np.nanpercentile(paw_spd[m_p], 50) if m_p.any() else np.nan
    ...
    paw_cls = discretize(paw_spd, paw_vis, thr_p)
```

iii. Same as 7-c: CONVERSION_NOTES.md Step 5 decision 8 (per-session median, from the decoder task specification) and Step 9's check of the not-visible fraction against the raw DLC NaN rate ("paw_velocity | - | DLC NaN | 0.5-9% NaN | not visible 0.049 | YES").

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. The same session `vidshift` and the same trial go cue, but using the **bottom** camera's own `frameTimes` (`vt1 = ft1 − vidshift − goCue[i]`) and its own median frame interval, because the two cameras can report different frame counts for a trial. Frames are digitized onto the same `edges` grid as the spikes and out-of-window frames dropped. If the bottom-camera entry for a trial is empty or its `NdroppedFrames` is NaN, the paw is left entirely not-visible for that trial.

ii.
```python
        if ts1.size and ft1.size and np.isfinite(nd1):
            vt1 = ft1 - vidshift - gocue[i]
            idx1 = np.digitize(vt1, bin_edges) - 1
            inb1 = (idx1 >= 0) & (idx1 < nbins)
            dtf1 = np.median(np.diff(ft1)) if ft1.size > 1 else 1 / 400.0
```

iii. Same justification as 7-d (CONVERSION_NOTES.md Step 5 decision 10; Step 10 Check 3(c)). `nvid = min(s.ntraj_trials(0), s.ntraj_trials(1))` guards against the two views holding different numbers of trials.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<ANM>_<DATE>.mat` file next to each data structure, which holds one motion-energy trace per trial with one value per camera frame (400 Hz, the side camera's frames). `obj.me` is not used. `load_motion_energy` unwraps three different on-disk layouts recursively: a struct whose `.data` is a cell array, a struct whose `.data` is itself a struct (JEB15), and a bare cell array with no `moveThresh` (some JEB23). The authors' `moveThresh` is parsed but not used for the output (the prompt requires a median split). The side camera's `frameTimes` provide the time base.

ii.
```python
def load_motion_energy(fn):
    """Return (list of per-trial motion-energy arrays, moveThresh or None).

    Three file variants exist in the dataset: me struct with cell .data, me struct
    whose .data is itself a struct (JEB15), and a bare cell array (some JEB23).
    """
    me = sio.loadmat(fn)['me']
    ...
    def unpack(x):
        if isinstance(x, np.ndarray) and x.dtype.names is not None and 'data' in x.dtype.names:
            ...
            return unpack(d)
        return x
```

iii. CONVERSION_NOTES.md Step 2: "motionEnergy file: `me.data` = cell (nTrials x 1) of per-frame motion energy (400 Hz, same frames as video), `me.moveThresh` = per-session movement threshold." Step 4: "loadMotionEnergy.m expects `me.data` cell + `me.moveThresh` | 3 file variants: (a) struct with cell data, (b) nested struct (JEB15 x2), (c) bare cell array without moveThresh (JEB23 x4) | wrote a robust loader; ME trial counts match Ntrials in every session." The reference `loadMotionEnergy.m` has the same nested-struct guard (`if isstruct(me.data), me.data = me.data.data; end`).

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond resampling — the value is already one number per frame (the paper reduces each frame to the 99th percentile of the per-pixel frame difference upstream). The AI ports `loadMotionEnergy.m` literally: the per-frame trace is **linearly interpolated** onto the 166 bin centres (`interp_to_axis`, a MATLAB-`interp1` equivalent that returns NaN outside the source range) and then the remaining NaNs are filled with the nearest finite value (`_fill_nearest`, a `fillmissing(...,'nearest')` equivalent). It is therefore sampled at the bin centres rather than averaged within bins, and nearest-fill extends the trace over any part of the window the video does not cover — so class 2 ('no video') occurs only for trials with no usable video at all (overall fraction 0.000; it appears in one trial of JEB19_2023-04-19). Traces and frame times are truncated to their common length.

ii.
```python
def interp_to_axis(src_t, src_y, taxis):
    """Linear interpolation like MATLAB interp1 (NaN outside the source range)."""
    ...
    inside = (taxis >= st[0]) & (taxis <= st[-1])
    ...
        out[inside] = np.interp(taxis[inside], st, sy)
```

```python
        # --- motion energy (loadMotionEnergy.m: interp to taxis, fill nearest) ---
        if i < len(me_trials):
            mev = me_trials[i]
            n = min(mev.size, vt0.size)
            if n > 1:
                y = interp_to_axis(vt0[:n], mev[:n], taxis)
                y = _fill_nearest(y)
                me_binned[i] = y
```

```python
def _fill_nearest(y):
    """MATLAB fillmissing(y,'nearest') for a 1-D array."""
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "`motionEnergy_*.mat` `me.data` -> `output[5]` = `motion_energy` | interp to trial time axis (frameTimes - vidshift - goCue), fillmissing nearest; 0/1 by session median, 2 if the trial has no usable video | loadMotionEnergy.m". Step 10 Check 3(f) item (iv): "ME is averaged into 30 ms bins via interpolation onto `obj.time` exactly as the reference does". Step 10 Check 2 verified it independently: "`motion_energy` class == interp1 + fillmissing + median threshold recomputed from the raw ME file | 3 | PASS". Step 7 gives the alignment check: "Motion energy above-median fraction 0.28-0.35 before vs 0.66-0.72 after the go cue -> ME alignment correct."

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session 50th percentile of the interpolated motion energy over kept trials and finite bins; < threshold -> 0, >= -> 1, and 2 where the trial has no usable video. Because nearest-fill leaves essentially no NaNs, the split is almost exactly 0.500/0.500/0.000.

ii.
```python
    m_m = sel[:, None] & np.isfinite(me_binned)
    thr_m = np.nanpercentile(me_binned[m_m], 50) if m_m.any() else np.nan
    ...
    me_cls = discretize(me_binned, has_video[:, None] & np.isfinite(me_binned), thr_m)
```

```python
                output_values=[..., ['below_median', 'above_median', 'no_video']])
```

iii. CONVERSION_NOTES.md Step 5 decision 8 (per-session median as specified by the decoder task) and Step 9: "motion_energy | - | ME per frame | - | 0.500/0.500/0.000 | YES (exact median split)". The AI explicitly chose the prompt's median split over the authors' manual `me.moveThresh`, which it loads but does not apply. Step 10 Check 5 notes that "class 2 of motion energy occurs in the one trial with unusable video in JEB19_2023-04-19".

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same offset and same axis as the tracking, using the **side** camera's frame times (motion energy has one value per side-camera frame): `vt0 = frameTimes(view 0) − vidshift − goCue[trial]`, then linear interpolation onto the shared bin-centre axis `taxis`. This is exactly the reference's `interp1(obj.traj{1}(trix).frameTimes - vidshift - alignTimes(trix), me.data{trix}, taxis)`.

ii.
```python
        ts0, ft0, nd0 = s.traj_trial(0, i)
        ...
        vt0 = ft0 - vidshift - gocue[i]
        ...
                y = interp_to_axis(vt0[:n], mev[:n], taxis)
```

iii. CONVERSION_NOTES.md Step 10 Check 3(c): "video: `frameTimes - vidshift - goCue`; ME: same | identical formulas, `vidshift` from `findVideoOffset.m` | YES". Step 7's pre/post go-cue contrast in above-median motion energy (0.28-0.35 before vs 0.66-0.72 after) is the sanity check offered for the alignment.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Every known quirk is handled by keeping the trial/session and marking or repairing the specific gap:
- **Two MATLAB file formats**: `is_hdf5` sniffs the header; `h5py` for v7.3 (36 files), `scipy.io` for v7 (11 files), behind one accessor API; `traj_trial` transposes `ts` for the HDF5 layout.
- **Three motion-energy layouts**: recursive `unpack` in `load_motion_energy`.
- **Quality labels with inconsistent case, padding and typos** (`'Poor'`, `'poor   '`, `'gabrga'`, `'mutli'`): stripped and lower-cased before matching; unlabelled clusters (`''`) are dropped.
- **Missing/absent `obj.clu`, 'DUMMY' probes**: `nprobes`/`clusters` return empty lists; such sessions are absent from the hard-coded list anyway.
- **Missing `obj.bp.stim`**: `has_bp_field` check, defaulting to zeros.
- **NaNs in Bpod flags**: coerced to 0 before masking.
- **Non-finite go cue**: trial dropped.
- **Unusable video** (`NdroppedFrames` NaN, empty `ts`, all-NaN `frameTimes`, or fewer video trials than Bpod trials): trial skipped in the video loop, exactly as `findPosition.m` does, and its tongue/paw/ME bins come out as class 2 (1 such trial in the whole dataset, JEB19_2023-04-19).
- **Untracked DLC frames** (x/y NaN): excluded from the velocity and from the bin average; bins with no visible frame become class 2.
- **Different frame counts between the two cameras**: each feature is timed by its own camera's `frameTimes`.
- **ME trace/frame-time length mismatch**: truncated to `min` length.
- **NaN `bitstart` entries**: removed before taking the mode for `vidshift`.
- **Recording that ends before the behavioural session**: trials with no spikes in any curated unit are dropped.
- **Sessions too small**: <10 units or <2 trials would be skipped at assembly.
No interpolation or filling is used for missing DLC data (the 'not visible' class carries it instead); motion energy is nearest-filled, following the reference.

ii.
```python
        if ts0.size == 0 or ft0.size == 0 or np.all(~np.isfinite(ft0)) or not np.isfinite(nd0):
            continue                                        # findPosition.m skips these trials
```

```python
    for v in (stim, early, hit, miss, no, autowater):
        v[np.isnan(v)] = 0
```

```python
    nvid = min(s.ntraj_trials(0), s.ntraj_trials(1))
    ...
    for i in trials:
        if i >= nvid:
            continue
```

```python
        bs = bs[~np.isnan(bs)]
```

iii. CONVERSION_NOTES.md Step 10 Check 5: "Data-quirk handling verified: MAT v7 vs v7.3, 3 motion-energy file layouts, missing `obj.clu`, 'DUMMY' probes, quality-label case and typos, trials with unusable video (`NdroppedFrames` NaN), sessions where the ephys stops early." Step 5 decision 7 gives the reason for not filling DLC gaps: "Kinematics NaNs are kept as the 'not visible' class instead of being filled/zeroed as in `findPosition`/`findVelocity`. This is required by the decoder output specification, which asks for an explicit 'not visible' category."

## 11-a. What are the most time-consuming steps of the code?

i. File reading, specifically the per-trial DeepLabCut reads out of the HDF5 files: the AI's own instrumentation (`timing['neural_bin']`, `timing['video']`) put neural binning at ~0.4 s per session against ~2.5 s per session for the video/ME stage, out of 1.4-4.5 s total per session. With 8 worker processes the whole 44-session conversion takes 16.9 s wall clock.

ii.
```python
    t1 = time.time()
    rates, meanfr = bin_spikes(clusters, gocue, ntrials, edges)  # (ntrials, nbins, nunits)
    timing['neural_bin'] = time.time() - t1
    ...
    timing['video'] = time.time() - t1
```

iii. CONVERSION_NOTES.md Step 6: "Code inefficiencies identified: per-trial DLC reads from HDF5 dominate runtime (each trial is a separate dataset reference)." Step 7's table gives "neural binning ~0.4 s | video/DLC + ME ~2.5 s | whole session 2.6-3.1 s", and Step 9 reports the measured full run ("15 s wall clock, 8 workers"), well inside the 15-minute budget in the instructions.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Three loops remain. (1) The per-cluster loop in `bin_spikes` — each cluster has its own ragged spike list, but the trial dimension inside it is already vectorised by `histogram2d` and smoothing is applied to all trials at once with `fftconvolve`. (2) The per-trial video loop in `process_session` — each trial has a different number of camera frames, so there is no rectangular array to work on; within a trial the frame-to-bin reduction is vectorised with `np.digitize` + `np.bincount`. (3) The per-run loop inside `speed_from_xy`, needed so gradients never cross a visibility gap. (4) The per-trial `lick_dir` loop over ragged lick-time cells. Sessions themselves are run in parallel processes rather than serially.

ii.
```python
        cnt, _, _ = np.histogram2d(tr, aligned, bins=[trial_edges, edges])
        meanfr[i] = cnt.sum() / (ntrials * window)
        rates[:, :, i] = mysmooth(cnt.T / DT).T.astype(np.float32)
```

```python
        sums = np.bincount(idx[good], weights=spd[good], minlength=nbins)
        cnts = np.bincount(idx[good], minlength=nbins)
```

```python
        with Pool(nproc) as pool:
            results = pool.map(_worker, list(zip(sessions, want_debug)))
```

iii. CONVERSION_NOTES.md Step 6: "Spike binning vectorised with `np.histogram2d` over (trial, aligned time) per unit instead of a per-trial loop (the reference's double loop over units x trials). Smoothing done with `scipy.signal.fftconvolve` over all trials at once. Video frames binned with `np.bincount` instead of per-bin loops. Sessions processed in parallel with a `multiprocessing.Pool` (default 8 workers; memory-bound because v7.3 files are 90-250 MB each)." Step 7: "vectorised histogram2d binning + FFT smoothing + bincount + 8-way multiprocessing | ~10x vs naive loops".

## 11-c. What processing does the code repeat multiple times?

i. Little, but not nothing. Each file is read once per session; `vidshift` is computed once per session (not per trial); the bin grid is rebuilt by `time_axis()` in each session and again in `main`, which is trivial. The genuine repeats are: `s.probe_locs()` is re-read inside the probe loop rather than once before it; `causal_kernel()` is rebuilt on every `mysmooth` call, i.e. once per cluster; the `ts` array of a trial is read twice per camera-bearing stream in the sense that the full feature array is materialised and then indexed per feature; and `speed_from_xy` recomputes the visibility mask that the caller also computes. None of these is material at 17 s total runtime. Notably, per-trial velocities are computed once and reused by both the binning and the percentile, and the per-session thresholds are computed once from the already-binned arrays.

ii.
```python
    for p in probes:
        loc_list = s.probe_locs()      # re-read inside the loop
        loc = loc_list[p - 1] if len(loc_list) >= p else ''
```

```python
    k = causal_kernel(N)               # rebuilt on every call, i.e. per cluster
```

```python
        xy = ts0[:, :2, i_tongue]
        vis = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])   # also computed inside speed_from_xy
        spd = speed_from_xy(xy) / dtf
```

iii. The notes do not discuss repeated computation explicitly; the efficiency section of Step 6 lists only the vectorisations and the parallelism, and Step 7 concludes the runtime was already far under budget ("44 sessions / 8 workers ~ 1-3 min ... observed full run below"), so no further deduplication was pursued.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Four things. (1) **Rates for units that are then discarded**: `bin_spikes` histograms *and* FFT-smooths every quality-passing cluster before the >1 Hz filter is applied, so the smoothing work for the ~5% of clusters that fail is thrown away. (2) **Rates for trials that are then discarded**: `rates` is built for all `ntrials`, including the photostim/early/no-ephys trials that are never emitted (13,762 of 15,062 trials are kept, so ~9% of the binning and smoothing is wasted). (3) **Fields loaded but unused**: `NdroppedFrames` is only used as a validity flag, the full `ts` array of every feature (including `jaw`, `nose`, `trident`, `lickport`, the extra tongue markers) is read per trial although only `tongue` and the two paws are used, the DLC likelihood channel is read and never used, `me.moveThresh` is parsed and never applied, and the per-cluster `qualities` list is assembled and filtered but never written to the output. (4) **Debug payload**: with `--show-processing` the first two sessions keep a full `debug` dict (all rates, speeds, lick cells) in memory for plotting.

ii.
```python
    rates, meanfr = bin_spikes(clusters, gocue, ntrials, edges)  # all clusters, all trials
    ...
    keep_units = meanfr > LOW_FR
    rates = rates[:, :, keep_units]
```

```python
            clusters.append(c)
            qualities.append(q)       # filtered later, never emitted
```

```python
    me_trials, _ = load_motion_energy(me_path(entry))   # moveThresh discarded
```

```python
    if want_debug:
        res['debug'] = dict(taxis=taxis, rates=rates, tongue_spd=tongue_spd, ...)
```

iii. The notes do not flag these as waste; the AI's position, stated in Step 7/9, is that the conversion already runs in ~17 s so the bottleneck is file I/O rather than computation. The FR filter is deliberately applied *after* binning because `removeLowFRClusters.m` thresholds the mean of the binned PSTH, which requires the counts to exist first (Step 9: "changed the low-FR criterion to use the raw (unsmoothed) spike counts over **all** trials, matching `removeLowFRClusters.m`").
