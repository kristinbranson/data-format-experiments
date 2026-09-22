# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Sessions are **not** discovered by globbing `/app/data`. A hard-coded `SESSIONS` list of 44 tuples `(animal, date, probes, directory, task)` is transcribed from the authors' `DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m` meta-loaders; entries commented out there (`JEB23_2023-10-20`) and animals whose data are not distributed (`JEB4`, `JEB5`) are omitted, as are the three extra `data_structure` files present on disk but absent from the loaders (`JEB24_2023-10-03/04`, which have no `obj.clu`). Each session's `data_structure_<anm>_<date>.mat` is opened once by `load_session`, which sniffs the file header (`is_v73`) and dispatches to an `h5py` reader for MATLAB v7.3 files or `scipy.io.loadmat` for v7 files. Both readers return the same dict and **only pull the fields the conversion needs** (selected `bp` fields, the requested probes of `obj.clu`, three DLC features, side-camera frame times, `obj.sglx`, `obj.ex.probe.loc`). Motion energy is read from the sibling `motionEnergy_<anm>_<date>.mat` by `load_motion_energy`. Sessions are processed in a `multiprocessing.Pool` (default 12 workers).

ii.
```python
SESSIONS = [
    # (animal, date, probes, directory, task)
    ('JEB6',  '2021-04-18', [2],    DATA_FIXED, 'fixed delay'),
    ...
    ('JEB24', '2023-11-03', [1],    DATA_RAND,  'randomized delay'),
]

def is_v73(fn):
    with open(fn, 'rb') as fh:
        return b'MATLAB 7.3' in fh.read(128)

def load_session(fn, probes, feats):
    return _load_v73(fn, probes, feats) if is_v73(fn) else _load_v7(fn, probes, feats)
```
```python
fn = os.path.join(ddir, 'data_structure_%s_%s.mat' % (anm, date))
feats = [TONGUE_FEAT] + PAW_FEATS
sess = load_session(fn, probes, feats)
...
if args.nproc > 1 and len(jobs) > 1:
    with Pool(min(args.nproc, len(jobs))) as pool:
        results = pool.map(process_session, jobs)
```

iii. From CONVERSION_NOTES.md Step 1/Step 4: the `load*_ALMVideo.m` files are "the authoritative inclusion list" — they record both which sessions and **which probe is ALM** entered the paper's analyses, and commented-out entries mark sessions the authors excluded. Globbing would add three files the reference never uses (two with no sorted units, one duplicate session). The dual reader is needed because "most `data_structure` files are MATLAB v7.3 (HDF5, read with `h5py`); 11 are v7". Reading only the needed fields/features was an explicit speed-up ("~5-10x vs loading the whole `obj`").

## 1-b. How are the data split into subjects (mice)?

i. The animal ID is the first element of each `SESSIONS` tuple (identical to the filename prefix, e.g. `JEB19_2023-04-19` → `JEB19`). It is carried on each session result as `anm`; at assembly `data['subjects']` is built in first-encounter order and `subject_idx` is each session's index into that list. Result: 14 subjects (10 fixed-delay + 4 randomized-delay animals).

ii.
```python
if r['anm'] not in data['subjects']:
    data['subjects'].append(r['anm'])
data['subject_idx'].append(data['subjects'].index(r['anm']))
...
data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. The animal ID is part of the session identity in the reference meta-loaders (one loader file per animal), so it is taken from there rather than from inside the file. CONVERSION_NOTES Step 9 flags and explains the one mismatch with the paper: the loaders list 10 fixed-delay animals while the paper text says nine; since the session count matches exactly (25) the AI keeps all 10 animals, "because the code's session list is the authoritative inclusion list".

## 1-c. How are the data split into sessions?

i. One entry of `SESSIONS` = one `data_structure` file = one element of `neural`, `input`, `output`, `subject_idx` and `brain_region_idx`. The folder is stored explicitly in the tuple (`DATA_FIXED` vs `DATA_RAND`) rather than searched, so the fixed-delay and randomized-delay experiments are pooled into one uniform list of sessions (25 + 19 = 44), with the task type recorded per session in `metadata['session_info']`. A session is dropped only if it ends up with < 10 units or < 2 trials; none of the 44 was dropped.

ii.
```python
sessions = SESSIONS[:2] if args.sample else SESSIONS
jobs = [(a, d, p, dd, tk, show and i < 2) for i, (a, d, p, dd, tk) in enumerate(sessions)]
...
fn = os.path.join(ddir, 'data_structure_%s_%s.mat' % (anm, date))
```
```python
if rates.shape[1] < MIN_UNITS:
    return dict(ok=False, reason='fewer than %d units (%d)' % (MIN_UNITS, rates.shape[1]), ...)
if keep_trials.size < MIN_TRIALS:
    return dict(ok=False, reason='too few trials', ...)
```

iii. Step 5: "the 44 ephys sessions listed (uncommented) in `DataLoadingScripts/Recording and video/load*_ALMVideo.m` = 25 fixed-delay + 19 randomized-delay, exactly the counts reported in the paper". Both task variants are included (Key Decision 6) "to use all available neural data; randomized-delay sessions contribute few/no WC trials, which is a property of the experiment, not a conversion error". `MIN_UNITS = 10` comes from the paper ("Recording sessions were included for analysis only if they had at least 10 units"); `MIN_TRIALS = 2` from the decoder format requirement.

## 1-d. How are the data split into trials?

i. A trial is one row of the `obj.bp` per-trial table. `N = int(bp['Ntrials'][0])` sets the trial count, every per-trial flag is asserted to have at least `N` entries and then sliced to `[:N]` (some fields are stored longer than `Ntrials`), and `align = bp.ev.goCue` provides exactly one alignment time per trial. Spikes carry their own trial index (`clu.trial`, 1-based, converted to 0-based) and camera/motion-energy data are stored per trial, so no trial boundaries have to be reconstructed.

ii.
```python
N = int(bp['Ntrials'][0])
ev = bp['ev']
align = ev[ALIGN_EVENT].astype(float)
hit = np.nan_to_num(bp['hit']).astype(bool)
...
for a in (hit, miss, no, R, L, early, aw, stim):
    assert a.size >= N
...
keep = (~stim[:N]) & (~early[:N]) & valid_align
keep_trials = np.nonzero(keep)[0]
```
```python
tr = c['trial'][i].astype(np.int64) - 1               # -> 0-based
tm = c['trialtm'][i]
```

iii. The Bpod table defines trials directly; `Ntrials` is the authoritative count and the reference condition strings in `findTrials.m` index the same per-trial vectors. `assert a.size >= N` plus `[:N]` truncation is the AI's guard against the over-long fields.

## 1-e. How are trials filtered based on quality controls?

i. Four filters. (1) Photostimulation trials (`bp.stim.enable`) are dropped; (2) early-lick trials (`bp.early`) are dropped — these two reproduce the `~stim.enable & ~early` clause present in every reference condition string; (3) trials whose go cue is not finite are dropped; (4) after binning, trials in which **no** quality-passing unit fires a single spike anywhere in the 5 s window are dropped, because in two sessions the SpikeGLX recording stopped before the behavioural session ended. Ignore (`bp.no`) trials are deliberately **kept**, since the decoder spec requires `ignore`/`none` classes. 13,762 of 15,155 trials survive.

ii.
```python
# ---- trial curation: reference conditions use ~stim.enable & ~early -------------
valid_align = np.isfinite(align[:N])
keep = (~stim[:N]) & (~early[:N]) & valid_align
keep_trials = np.nonzero(keep)[0]
```
```python
# Drop trials with no ephys coverage: in a few sessions (e.g. JEB24_2023-10-23) the
# SpikeGLX recording ends before the behavioural session, so the trailing trials contain
# no spikes from ANY unit ...
has_spikes = rates.sum(axis=(1, 2)) > 0
n_noephys = int((~has_spikes).sum())
if n_noephys:
    rates = rates[has_spikes]
    keep_trials = keep_trials[has_spikes]
```

iii. Step 4/5: "All paper analyses use `~stim.enable` (no optogenetic stimulation) and `~early` (no early lick)"; ignore trials are retained "because the decoder specification requires an `ignore` outcome class and a `none` lick-direction class". The zero-spike rule was added in Step 9 after the first full run produced 61 "all neural data is zero" warnings; the AI traced it to the raw files — "in `JEB24_2023-10-23` spikes exist only up to trial 314/343 and in `JEB24_2023-11-03` up to trial 312/346 … (`obj.trials.bp.haveEphys` is 1 for all trials and therefore does not flag it)" — and argues it is "impossible over a 5 s window with tens of simultaneously recorded units unless the probe was no longer recording".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The spike-sorted clusters `obj.clu{probe}` of the probe(s) named in `SESSIONS`, using three fields per cluster: `trialtm` (spike time relative to its trial's start, behaviour clock), `trial` (1-based trial index of each spike) and `quality` (manual curation label). The second input is `obj.bp.ev.goCue`, which sets the alignment. Nothing else (no `clu.tm`, no waveforms) is read.

ii.
```python
clu[p] = dict(
    quality=[_h5str(f, g['quality'][i, 0]).strip() for i in range(n)],
    trialtm=[np.array(f[g['trialtm'][i, 0]]).ravel().astype(np.float64) for i in range(n)],
    trial=[np.array(f[g['trial'][i, 0]]).ravel().astype(np.int64) for i in range(n)])
```
```python
tr = c['trial'][i].astype(np.int64) - 1               # -> 0-based
tm = c['trialtm'][i]
...
aligned = tm[sel] - align_times[tr[sel]]
```

iii. Step 1/Step 2: "Neural data for ephys sessions: spike times per cluster `obj.clu{probe}.trialtm` (time within trial), `.trial`, `.quality`", which is exactly the set of fields `alignSpikes.m`, `getSeq.m` and `findClusters.m` touch. Only the ALM probe(s) given by `meta.probe` in the loaders are read.

## 2-b. How is the `neural` data processed?

i. `bin_spikes` reproduces `alignSpikes.m` + `getSeq.m` + `mySmooth.m`: spikes are aligned to the go cue, counted into 500 non-overlapping **10 ms** bins over [-2.5, 2.5] s with MATLAB's `histc` convention (`floor((t - tmin)/dt)`, last edge dropped), divided by `dt` to give spikes/s, and smoothed along time with the authors' **causal** Gaussian kernel — `gausswin(15, alpha=2.5)` with the first `floor(15/2)` taps zeroed and renormalised, convolved `'same'` after prepending the first 15 samples ('reflect' boundary as in `mySmooth.m`). The smoothed rates are then averaged in groups of 5 to give **50 ms** output bins (100 timepoints). No normalisation, z-scoring or baseline subtraction; units are spikes/s, stored `float32`. Units from both probes of a two-probe session are concatenated into one population.

ii.
```python
def gausswin(N, alpha=2.5):
    n = np.arange(N) - (N - 1) / 2.0
    return np.exp(-0.5 * (alpha * n / ((N - 1) / 2.0)) ** 2)

def causal_kernel(N=SMOOTH_N):
    """Kernel used by utils/mySmooth.m: gausswin with the first floor(N/2) taps zeroed."""
    k = gausswin(N)
    k[:N // 2] = 0.0
    return k / k.sum()

def my_smooth(x, kern=KERN, bctype=BCTYPE):
    N = len(kern)
    if bctype == 'reflect':
        xf = np.concatenate([x[..., :N], x], axis=-1)
        trim = N
    ...
    out = fftconvolve(xf, kern.reshape((1,) * (xf.ndim - 1) + (N,)), mode='same', axes=-1)
    return out[..., trim:]
```
```python
aligned = tm[sel] - align_times[tr[sel]]
b = np.floor((aligned - TMIN) / DT_FINE).astype(np.int64)
inwin = (b >= 0) & (b < NFINE)
if inwin.any():
    flat = row[sel][inwin] * NFINE + b[inwin]
    np.add.at(counts.reshape(-1), flat, 1.0)
rate = counts / DT_FINE                                # spikes / s
rate = my_smooth(rate)                                 # causal gaussian
rates.append(downsample_mean(rate).astype(np.float32))
```

iii. Step 5 Key Decision 3: "**Firing rates rather than spike counts**: the reference `trialdat` (spikes/s, smoothed) is what is fed to all reference decoders." Step 10 Check 3(d) records the line-by-line comparison with `getSeq.m` and notes the AI deliberately reproduced the `histc`-then-drop-last-bin convention (its first sanity check used `np.histogram`, which counts a spike at exactly t = +2.5 s, and the *check* was corrected, not the converter). An independent re-implementation from the raw `.mat` files reproduced every converted unit's trace to 4.5e-06 (float32 rounding).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two unit filters plus one session filter. (1) `findClusters.m`'s exclusion list — `{garbage, gabrga, noisy, real?}` — applied **case-insensitively** (the released labels are capitalised, e.g. `Garbage`, `Poor`); everything else is kept, including multi-units, `poor` and unlabeled/empty-string clusters. (2) `removeLowFRClusters.m`: mean firing rate across all kept trials and all 100 bins must exceed `LOW_FR = 1.0` Hz. (3) A session is dropped if fewer than `MIN_UNITS = 10` units survive. Result: 2,513 units pass quality, 2,456 (97.7%) also pass 1 Hz; 17–141 per session.

ii.
```python
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}   # findClusters.m
LOW_FR = 1.0                # params.lowFR (Hz)
MIN_UNITS = 10              # paper: sessions included only if they had at least 10 units
...
good = [i for i, q in enumerate(c['quality'])
        if q.strip().lower() not in BAD_QUALITY]          # findClusters.m
```
```python
# removeLowFRClusters.m: mean firing rate across trials & time must exceed lowFR
mean_fr = rates.mean(axis=(0, 2))
use = mean_fr > LOW_FR
rates = rates[:, use, :]
ids = [c for c, u in zip(ids, use) if u]
```

iii. Step 4: the labels "are capitalized (`Poor`, `Great`, ...); the reference `findClusters` compares to lowercase names, so I match **case-insensitively** (same intent, otherwise nothing would be excluded)". The 1 Hz cut is `params.lowFR = 1` and the paper's "All units with firing rates exceeding 1 Hz were included in all other analyses". Step 10 Check 3 documents one deliberate difference: `removeLowFRClusters.m` averages the *condition* PSTHs, whereas the AI takes the unweighted mean over all kept trials — "Both estimate the same quantity … mine is the unweighted trial mean, which is the more natural estimate and does not depend on the arbitrary condition list." Empty quality strings are kept "exactly as `findClusters.m` does".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. By a single subtraction, `trialtm - bp.ev.goCue[trial]`, performed per spike using that spike's own trial index. `clu.trialtm` is already on the behaviour clock relative to its trial's start and `bp.ev.goCue` is on the same clock, so no offset or interpolation is needed for the neural stream (unlike the video streams, which need `vidshift`). `ALIGN_EVENT = 'goCue'`, and on water-cued trials `bp.ev.goCue` is the water-drop time.

ii.
```python
ALIGN_EVENT = 'goCue'
...
align = ev[ALIGN_EVENT].astype(float)
...
aligned = tm[sel] - align_times[tr[sel]]
b = np.floor((aligned - TMIN) / DT_FINE).astype(np.int64)
```

iii. Step 1: `alignSpikes` — "`clu.trialtm_aligned = clu.trialtm - ev.(alignEvent)(clu.trial)`; alignEvent='goCue'". Step 5 Key Decision 1: "**Alignment to go cue** (`params.alignEvent='goCue'`) and window **[-2.5, +2.5] s** — exactly the reference `params.tmin/tmax`. On WC trials `bp.ev.goCue` is the water-drop time, which is what the paper uses." Verified indirectly: population rate peaks at +0.12 s, tongue visibility jumps after 0 in all 44 sessions (median 30× higher than before), and the `_alignment.png` plots show sample/delay onsets at fixed negative times.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The stored resolution is **50 ms**, 100 bins spanning -2.5 to +2.5 s; `metadata['time_bin_size'] = 50.0`, `off_start = -2.5`, `off_end = 2.5`. Rebinning **is** applied: spikes are first binned at the reference's `params.dt = 10 ms` (500 bins) and smoothed there, then five consecutive fine bins are averaged (`DOWNSAMPLE = 5`) into each output bin. The input time axis and all three camera streams are put on the same 10 ms grid and downsampled by the same factor, so every stream shares one time axis.

ii.
```python
DT_FINE = 0.01              # params.dt (1/100 s) used for binning + smoothing
DOWNSAMPLE = 5              # 10 ms -> 50 ms decoder bins
DT_OUT = DT_FINE * DOWNSAMPLE

def time_axes():
    edges = np.round(np.arange(TMIN, TMAX + 1e-9, DT_FINE), 6)
    centres = edges[:-1] + DT_FINE / 2.0
    nfine = len(centres)
    nout = nfine // DOWNSAMPLE
    centres_out = centres[:nout * DOWNSAMPLE].reshape(nout, DOWNSAMPLE).mean(axis=1)
    return edges, centres, centres_out

def downsample_mean(x, k=DOWNSAMPLE, nanmean=False):
    n = (x.shape[-1] // k) * k
    y = x[..., :n].reshape(x.shape[:-1] + (n // k, k))
    ...
    return y.mean(axis=-1)
```

iii. Step 4/Step 5 Key Decision 2: "`params.dt` = 1/100 (most scripts), 3/100 (Fig 3c), 1/200 (Fig 1e); decoding scripts average to **75 ms** bins … Bin spikes at 10 ms + causal Gaussian smoothing (15 bins) exactly as `getSeq`/`mySmooth`, then average into **50 ms** decoder bins (close to the reference's 75 ms decoding bins, and makes the go cue fall exactly on a bin edge)." Step 9 lists the bin size as an "intentional" difference from the reference.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not read from the raw data: it is the converter's own time axis, defined by the alignment event (`bp.ev.goCue`) and the window/bin constants `TMIN = -2.5`, `TMAX = 2.5`, `DT_FINE = 0.01`, `DOWNSAMPLE = 5`. The stored value for every trial is the vector of 50 ms bin centres, -2.475 … +2.475 s.

ii.
```python
TMIN, TMAX = -2.5, 2.5      # params.tmin / params.tmax
...
EDGES, TCENT, TOUT = time_axes()
NFINE, NOUT = len(TCENT), len(TOUT)
...
tin = TOUT.astype(np.float32)[None, :]
inputs = [tin.copy() for _ in range(kt.size)]
data['input_names'] = ['time_from_go_cue']
```

iii. Step 5 variable-mapping table: "time axis → `input[0]` = `time_from_go_cue`, bin centres in seconds, -2.475 … +2.475; only decoder input, per spec". The window is `params.tmin/tmax` from the reference figure scripts.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond constructing the axis: `getSeq.m`'s convention `edges = tmin:dt:tmax; time = edges + dt/2; time = time(1:end-1)` gives the 500 fine bin centres, and each output value is the mean of the five fine centres it covers (equivalently the centre of the 50 ms bin). The same `float32` vector is copied for every trial of every session, so the input is identical across trials by construction; sanity check 3 confirmed `[-2.475, 2.475]` for every trial of every session.

ii.
```python
centres = edges[:-1] + DT_FINE / 2.0
centres_out = centres[:nout * DOWNSAMPLE].reshape(nout, DOWNSAMPLE).mean(axis=1)
```

iii. Step 6 maps `time_axes` to `getSeq.m` ("`edges = tmin:dt:tmax`, centres at `edges + dt/2`"). Nothing else is required — the quantity is defined by the alignment, not measured.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid. Spikes are placed into the fine bins whose centres are `TCENT` and then averaged in groups of five; the input is the mean of the same five centres, so output bin *k* denotes exactly the same 50 ms interval in the input and in the neural array (and in the three camera outputs, which are interpolated onto `TCENT` and downsampled with the same function). `off_start`/`off_end` in the metadata state the window explicitly.

ii.
```python
b = np.floor((aligned - TMIN) / DT_FINE).astype(np.int64)   # neural -> fine bins
rates.append(downsample_mean(rate).astype(np.float32))      # fine -> 50 ms
...
pos[:, d] = np.interp(TCENT, ft[valid], xy[valid, d], ...)  # video -> same fine grid
tongue_ds = downsample_mean(tongue_sp, nanmean=True)        # same downsampling
...
tin = TOUT.astype(np.float32)[None, :]                      # the same bin centres
```

iii. Sanity check 3 in Step 10 ("`input[0]` equals the bin centres of [-2.5, 2.5] s in 50 ms steps for every trial of every session — PASS"). No separate justification is needed: one grid is shared by all streams.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial `obj.bp` flags: the instructed side `R` and `L`, and the outcomes `hit` and `miss`. The licked side is not recorded as such, so it is inferred from instructed side × outcome; trials that are neither hit nor miss (i.e. `bp.no`, ignore) fall through to the `none` class.

ii.
```python
hit = np.nan_to_num(bp['hit']).astype(bool)
miss = np.nan_to_num(bp['miss']).astype(bool)
R = np.nan_to_num(bp['R']).astype(bool)
L = np.nan_to_num(bp['L']).astype(bool)
```

iii. Step 5 maps this to the reference's `funcs/getPrevChoice.m` (`choice = (R&hit) | (L&miss)`), and Step 4 records an independent validation: the derived direction "is identical to the direction of the first lickport contact after the go cue on 382/382 trials of JEB6_2021-04-18" (extended in Step 10 to 924 trials across 3 sessions, 0 mismatches).

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A three-way relabelling, constant across the 100 bins of a trial: right (1) if `(R & hit) | (L & miss)`, left (0) if `(L & hit) | (R & miss)`, otherwise none (2). Codes and value names are `['left', 'right', 'none']`.

ii.
```python
lick_dir = np.full(kt.size, 2, dtype=np.int64)          # 2 = none (ignore trials)
right_lick = (R[kt] & hit[kt]) | (L[kt] & miss[kt])     # funcs/getPrevChoice.m
left_lick = (L[kt] & hit[kt]) | (R[kt] & miss[kt])
lick_dir[right_lick] = 1
lick_dir[left_lick] = 0
...
o[0] = lick_dir[i]          # broadcast over all NOUT bins
```

iii. Step 4: the definition is taken verbatim from `getPrevChoice.m`, with the reference's `NaN` for ignore trials replaced by an explicit third class because the decoder spec requires `none`. The one apparent mismatch in the independent lick-time check was investigated and attributed to the check, not the conversion: "JEB19_2023-04-20, trial 296 is a `bp.no == 1` (ignore) trial whose only post-go-cue lick is at **+4.01 s**, i.e. after the [paper's 3 s] response window, so `none` is the correct label."

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial flag, `obj.bp.autowater`, which marks trials in the water-cued block (water delivered at a random port with no cues).

ii.
```python
aw = np.nan_to_num(bp['autowater']).astype(bool)
```

iii. Step 4: "`autowater` used 'as a proxy for obtaining water-cued blocks and delayed-response blocks' (`WorkingWithDataObjs.m`)"; it also appears directly in the reference condition strings (`~autowater`). Sanity check 5 confirmed the converted `context` equals raw `bp.autowater`.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct two-class relabelling held constant over time: `autowater == 1` → WC (0), otherwise DR (1), matching the spec's `(WC, DR)` ordering. Overall 9.7% WC / 90.3% DR (≈33% WC in the two-context fixed-delay sessions, ≈0 in the randomized-delay sessions).

ii.
```python
context = np.where(aw[kt], 0, 1).astype(np.int64)       # 0 = WC (autowater), 1 = DR
...
o[1] = context[i]
```

iii. Step 4/5: "`autowater==1` → WC, `0` → DR." Step 10 Check 4 notes the low overall WC fraction is "expected since those [randomized-delay] sessions were run without WC blocks", and `metadata['session_info']` records the task type per session so the imbalance is traceable.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The per-trial flags `obj.bp.hit` and `obj.bp.miss`. `bp.no` is loaded (and used in an assertion) but is not needed to assign the class: a trial that is neither hit nor miss is an ignore by construction.

ii.
```python
hit = np.nan_to_num(bp['hit']).astype(bool)
miss = np.nan_to_num(bp['miss']).astype(bool)
no = np.nan_to_num(bp['no']).astype(bool)
```

iii. Step 5 maps this to `funcs/getOutcome.m` ("outcome = bp.hit, with bp.no (ignore) trials set to NaN"). The same two flags already determine lick direction. Sanity check 6 verified the converted `outcome` against raw `bp.hit`/`bp.miss`/`bp.no`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A three-class relabelling, constant over time, in the order the spec gives: incorrect (0) on `miss`, correct (1) on `hit`, ignore (2) otherwise. Result: 12.0% incorrect, 74.9% correct, 13.1% ignore.

ii.
```python
outcome = np.full(kt.size, 2, dtype=np.int64)           # 2 = ignore
outcome[hit[kt]] = 1
outcome[miss[kt]] = 0
...
o[2] = outcome[i]
```

iii. Step 4/5: ignore trials, which the paper omits from its behavioural analyses, are "retained as an output class" because the decoder spec lists `ignore`; the `getOutcome.m` NaN is replaced by class 2. Step 10 Check 4 compares the resulting hit fraction (74.9%) with the paper's behavioural performance.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj{1}` (side camera) for the feature named `tongue`: the x and y columns of `ts` (columns 1–2 of the `(nframes, 3, nfeat)` array; the third, likelihood, column is not read), plus `traj.frameTimes`, `traj.NdroppedFrames`, `obj.sglx.fs`, `obj.sglx.bitcode.bitstart` and `bp.ev.bitStart` (for the clock offset) and `bp.ev.goCue`. The bottom camera's tongue markers (`top_tongue`, etc.) are **not** used.

ii.
```python
TONGUE_FEAT = ('side', 'tongue')            # obj.traj{1} 'tongue'
...
fi = featNames.index(featname)
xy.append(np.array(ds[fi, :2, :]).T.astype(np.float64))   # (nframes, 2)
ft.append(np.array(f[g['frameTimes'][i, 0]]).ravel().astype(np.float64))
...
tongue_sp, _ = feature_speed(sess, TONGUE_FEAT, kt, vidshift, align, fill_missing=False)
```

iii. Step 2 lists the per-view feature names and Step 5 maps `obj.traj{1}` feature `tongue` → `output[3]`, following the reference's `findPosition`/`findVelocity`, which handle one view/feature at a time; `tongue` is the primary side-view tongue marker in `params.traj_features`. Step 4 notes "tongue NaN ~93% of frames" and that the authors already store NaN where the tongue is not tracked, so no explicit likelihood threshold is applied — the AI verified the dataset semantics ("`tongue` is NaN (not labelled by DLC) ~93% of frames = not visible"), which is exactly the likelihood ≤ 0.9 set.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps, mirroring `findPosition.m` + `findVelocity.m`. (1) Trials whose `NdroppedFrames` is NaN are skipped (all bins → not visible), as the reference does. (2) x and y are linearly interpolated from the corrected frame times onto the 10 ms grid with `np.interp` and `left/right = NaN`, **without** smoothing (the reference does not smooth tongue position) and **without** nearest-fill (`fill_missing=False`, the reference's tongue exception); a grid point is "visible" only if both interpolated coordinates are finite. (3) Velocity is `np.gradient` of the interpolated x and y (no baseline-derivative subtraction — the reference applies that only to non-tongue features) and speed is `hypot(vx, vy)`, in pixels per 10 ms sample. (4) Non-visible samples are set to NaN rather than the reference's 0, and the fine trace is averaged into 50 ms bins with `nanmean`, so a bin is visible if any fine sample in it was. No cross-camera normalisation is needed because only one view is used.

ii.
```python
pos = np.empty((NFINE, 2))
for d in range(2):
    pos[:, d] = np.interp(TCENT, ft[valid], xy[valid, d], left=np.nan, right=np.nan)
vis = np.isfinite(pos).all(axis=1)
if fill_missing:
    ...
vel = np.gradient(pos, axis=0)
if fill_missing:
    base = np.nanmedian(np.diff(pos, axis=0), axis=0)   # findVelocity.m
    vel = vel - base[0]
sp = np.hypot(vel[:, 0], vel[:, 1])
sp[~vis] = np.nan                                       # not visible
```
```python
tongue_ds = downsample_mean(tongue_sp, nanmean=True)
```

iii. Step 5 variable-mapping row and Step 4: "`findPosition` keeps tongue NaNs (not visible); `findVelocity` sets tongue velocity to 0 when not visible … Keep NaN = 'not visible' (class 2) rather than 0, as required by the decoder spec." The paper's "velocity of each feature was then calculated as the first-order derivative of the position vector" is the basis for `np.gradient`; the AI keeps the reference's convention of differentiating per sample rather than per second, which is irrelevant after a percentile split.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per session, the threshold is the 50th percentile of **all finite** (trial × bin) values of the 50 ms binned speed — i.e. pooled over the whole session and computed only over visible bins. `>= thr` → 1, `< thr` → 0, non-finite → 2. Value names `['below_median', 'above_median', 'not_visible']`. The resulting session distribution is ~5.1% / 5.1% / 89.7%, i.e. exactly half of the visible bins in each of classes 0 and 1.

ii.
```python
def discretize(values, nan_class=2):
    """0 = < 50th percentile, 1 = >= 50th percentile (per session), 2 = not visible/no video."""
    finite = np.isfinite(values)
    cls = np.full(values.shape, nan_class, dtype=np.int64)
    if finite.any():
        thr = np.percentile(values[finite], 50)
        cls[finite & (values >= thr)] = 1
        cls[finite & (values < thr)] = 0
    else:
        thr = np.nan
    return cls, thr
...
tongue_cls, tongue_thr = discretize(tongue_ds)
```

iii. Step 5 Key Decision 7: "**Discretization thresholds** are computed **per session** on the pooled (trial × bin) distribution of *visible* values, as specified by the task (50th percentile)." This is a documented deviation from the reference, which uses `me.moveThresh`/no threshold at all: "For the decoder output the task specifies a **50th-percentile** per-session threshold, which replaces `me.moveThresh` (documented deviation required by the decoder spec)." The per-session thresholds are recorded in `metadata['session_info'][i]['thr']`.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are on the video clock, which leads the behaviour clock, so a session-constant `vidshift` is removed first: `vidshift = mode(sglx.bitcode.bitstart)/sglx.fs - mode(bp.ev.bitStart)` (`findVideoOffset.m`, ≈0.490 s in every session checked). The per-trial frame time relative to the alignment event is then `frameTimes - vidshift - goCue[trial]`. If `frameTimes` is missing or all-NaN the reference's fallback is used: `(1:n)/400 - 0.5 - goCue[trial]`. The corrected times are the x-axis for the interpolation onto `TCENT`, the identical grid the spikes are binned on, and the same `downsample_mean` produces the 50 ms bins.

ii.
```python
def video_offset(sess):
    """funcs/findVideoOffset.m: mode(sglx.bitcode.bitstart)/fs - mode(bp.ev.bitStart)."""
    ...
    vid_file_offset = float(smode(bs, keepdims=False).mode) / sess['sglx']['fs']
    return vid_file_offset - float(smode(np.round(bstart, 6), keepdims=False).mode)

def frame_times_for_trial(sess, trial, vidshift, align_t):
    ft = sess['frameTimes'][trial]
    if ft.size == 0 or np.all(np.isnan(ft)):
        ft = (np.arange(1, sess['nframes'][trial] + 1) / VIDEO_FS)
        return ft - 0.5 - align_t
    return ft - vidshift - align_t
```

iii. Step 4/5 Key Decision 8: "**Video/ephys synchronisation** uses the reference `findVideoOffset` (vidshift ~0.49 s) with the reference fallback (`(1:n)/400 - 0.5`)." Sanity check 7 recomputed `vidshift` from the raw file (0.49004 s, PASS) and sanity check 9 verified the alignment behaviourally: "tongue visibility is much higher after than before the go cue in **all 44 sessions** (min ratio 6.0, median 30.0)"; the `_alignment.png` figure and the quantitative table in Step 7 show tongue visibility rising from 0.04 at t = −0.1 s to 0.49 at t = +0.2 s.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The bottom-camera tracking `obj.traj{2}` for **both** paw markers, `top_paw` and `bottom_paw` (x and y columns only), plus the same frame-time/offset variables as the tongue.

ii.
```python
PAW_FEATS = [('bottom', 'top_paw'), ('bottom', 'bottom_paw')]   # obj.traj{2}, paws only in bottom view
...
paw_sps = [feature_speed(sess, k, kt, vidshift, align, fill_missing=True)[0] for k in PAW_FEATS]
paw_sp = np.nanmean(np.stack(paw_sps, axis=0), axis=0) if paw_sps else np.full_like(tongue_sp, np.nan)
```

iii. Step 2/Step 3: "The tongue, jaw and nose were tracked using both cameras, whereas the paws were tracked using only the bottom view" (paper), so only `obj.traj{2}` is consulted. Both markers are the two paws listed in the reference's `params.traj_features` for cam1, and Step 5 states the speed is "averaged over the two paw markers".

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The same `feature_speed` pipeline as the tongue but on the reference's non-tongue branch (`fill_missing=True`): position is interpolated onto the 10 ms grid, the visibility mask is recorded **before** filling, missing grid points are nearest-filled (`np.interp` over indices, = `fillmissing(...,'nearest')`), `np.gradient` gives the velocity, the median baseline derivative is subtracted (`findVelocity.m`), speed is `hypot(vx, vy)`, and samples that were not visible before filling are set back to NaN. The two paws' fine traces are averaged with `np.nanmean` and the result is averaged into 50 ms bins with `nanmean`. No normalisation is applied. Resulting class fractions: 47.6% / 47.6% / 4.8% not visible.

ii.
```python
if fill_missing:
    # nearest-neighbour fill (fillmissing(...,'nearest')) as in findPosition.m
    if vis.any():
        idx = np.arange(NFINE)
        pos = np.stack([np.interp(idx, idx[vis], pos[vis, d]) for d in range(2)], axis=1)
    else:
        continue
vel = np.gradient(pos, axis=0)
if fill_missing:
    base = np.nanmedian(np.diff(pos, axis=0), axis=0)   # findVelocity.m
    vel = vel - base[0]
sp = np.hypot(vel[:, 0], vel[:, 1])
sp[~vis] = np.nan                                       # not visible
```

iii. Step 4: "Non-tongue features are nearest-filled as in the reference; paw NaNs (after interpolation, i.e. no DLC label in that bin) are mapped to 'not visible'", following the paper's "Missing values were filled in with the nearest available value for all features, except for the tongue" and `findVelocity.m`'s baseline-derivative subtraction. Note that the AI reproduces the reference's own quirk of subtracting `basederiv(1)` (the x-axis median difference) from *both* velocity components — `vel = vel - base[0]` matches `xvel = xvel - basederiv(1); yvel = yvel - basederiv(1)` in `findVelocity.m`. Filling is only used to avoid `np.gradient` propagating NaN across gaps; the pre-fill mask still decides visibility.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Identically to the tongue: one `discretize` call with the per-session 50th percentile of the finite 50 ms binned values, `>=` → 1, `<` → 0, NaN → 2 (`not_visible`).

ii.
```python
paw_ds = downsample_mean(paw_sp, nanmean=True)
paw_cls, paw_thr = discretize(paw_ds)
```

iii. Same as 7-c: Step 5 Key Decision 7 and the decoder spec's per-session 50th-percentile rule. The thresholds are stored per session in `metadata['session_info'][i]['thr']['paw']`.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same session `vidshift`, same per-trial `- goCue`, same interpolation onto `TCENT` and same downsampling to 50 ms bins as every other stream, via the same `frame_times_for_trial` helper. That helper always reads the **side** camera's `frameTimes` (`sess['frameTimes']`, loaded from `obj.traj{1}`) even for the bottom-camera paws; if the frame count disagrees with the bottom-camera `ts` the arrays are truncated to the shorter length, and trials with fewer than two usable frames are skipped.

ii.
```python
ft = frame_times_for_trial(sess, t, vidshift, align_times[t])
xy = tr['xy'][t]
if ft.size != xy.shape[0] or ft.size < 2:
    m = min(ft.size, xy.shape[0])
    if m < 2:
        continue
    ft, xy = ft[:m], xy[:m]
```

iii. Step 4/5: the video alignment rule is `frameTimes - vidshift - goCue` from `findPosition.m`, applied unchanged to all camera features; Step 10 Check 5 lists the frame-count/length guard among the handled edge cases ("Trials shorter than the window: video interpolation returns NaN outside the recorded frames → 'not visible' rather than fabricated values"). The two cameras are hardware-synchronised; in the sessions I inspected their `frameTimes` vectors are bit-identical, so using the side camera's times for the paws has no practical effect, but the AI does not document having checked this.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` sitting beside each data structure: `me.data`, a cell array with one 400 Hz trace per trial, and `me.moveThresh`, the authors' per-session movement threshold (loaded and recorded in the metadata but not used for discretisation). Alignment additionally uses the side camera's `frameTimes` and `bp.ev.goCue`. `obj.me` is not used.

ii.
```python
def load_motion_energy(fn):
    m = sio.loadmat(fn, struct_as_record=False, squeeze_me=True)
    me = m['me']
    thresh = np.nan
    raw = me
    # descend through nested structs until we reach the cell array of trials
    while hasattr(raw, '_fieldnames'):
        if 'moveThresh' in raw._fieldnames:
            try:
                thresh = float(np.array(raw.moveThresh).ravel()[0])
            except Exception:
                pass
        if 'data' not in raw._fieldnames:
            break
        raw = raw.data
    data = [np.atleast_1d(np.array(d).ravel()).astype(float) for d in np.atleast_1d(raw)]
    return dict(data=data, moveThresh=thresh)
```

iii. The docstring and Step 9 record the three layouts found in the released files — "`me` struct with fields {data, moveThresh}" (37 files), "`me` whose `data` field is itself a struct" (3 files, the case `loadMotionEnergy.m` guards with `if isstruct(me.data), me.data = me.data.data; end`), and "`me` saved directly as the cell array … (no moveThresh)" (4 JEB23 files) — which is why the unwrapping is a loop. This bug was found and fixed during the first full run.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Nothing beyond re-sampling: the paper's per-pixel difference and 99th-percentile spatial reduction are already baked into `me.data`, so `motion_energy_trials` reproduces `loadMotionEnergy.m` exactly — `np.interp` of the 400 Hz trace onto the 10 ms grid using `frameTimes - vidshift - goCue`, then a nearest-neighbour fill of any remaining NaNs (`fillmissing(...,'nearest')`), then averaging into 50 ms bins. Because of the nearest-fill, the trace is defined at every bin of every retained trial, so the `no_video` class is never used (0.000 of bins).

ii.
```python
v = np.interp(TCENT, ft[valid], y[valid], left=np.nan, right=np.nan)
# fillmissing(...,'nearest')
ok = np.isfinite(v)
if not ok.all() and ok.any():
    idx = np.arange(NFINE)
    v = np.interp(idx, idx[ok], v[ok])
out[r] = v
...
me_ds = downsample_mean(me_fine, nanmean=True)
```

iii. Step 1/4: `loadMotionEnergy` "interp1 onto `obj.time` using `frameTimes - vidshift - alignTime`; `fillmissing(...,'nearest')`", and Step 4's resolution is "Interpolate identically." Step 10 sanity check 8 re-derived the classes from the raw `motionEnergy_*.mat` with independent code: "0/30,200 bins differ; threshold identical to 3 decimals".

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The same `discretize` call: per-session 50th percentile of the finite 50 ms binned values, `>=` → 1 (`above_median`), `<` → 0 (`below_median`), non-finite → 2 (`no_video`). The authors' `me.moveThresh` is deliberately **not** used as the threshold, though it is stored per session in `metadata['session_info'][i]['me_thresh_session']` and drawn on the diagnostic plots. Result: 50.0% / 50.0% / 0.0%.

ii.
```python
me_cls, me_thr = discretize(me_ds)
...
me_thresh_session=(None if me is None else me['moveThresh']),
thr=dict(..., me=float(me_thr) if np.isfinite(me_thr) else None),
```

iii. Step 4: "For the decoder output the task specifies a **50th-percentile** per-session threshold, which replaces `me.moveThresh` (documented deviation required by the decoder spec)." The `no_video` class exists per the spec but is empty because the reference's nearest-fill leaves no undefined bins.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Identically to the tracking streams, using the side camera's frame times, since motion energy has exactly one value per side-camera frame: `frameTimes - vidshift - goCue[trial]` (with the `(1:n)/400 - 0.5` fallback), truncated to the shorter of the frame-time and motion-energy vectors, interpolated onto `TCENT` and downsampled by 5.

ii.
```python
def motion_energy_trials(me, sess, keep_trials, vidshift, align_times):
    """DataLoadingScripts/loadMotionEnergy.m: interpolate 400 Hz ME onto the trial time axis."""
    ...
    y = me['data'][t]
    ft = frame_times_for_trial(sess, t, vidshift, align_times[t])
    m = min(ft.size, y.size)
    if m < 2:
        continue
    ft, y = ft[:m], y[:m]
    valid = np.isfinite(ft) & np.isfinite(y)
```

iii. `loadMotionEnergy.m` uses `obj.traj{1}(trix).frameTimes` — the side camera — so the AI does the same, including the reference's `catch` fallback. Step 7's quantitative alignment table shows the expected post-go-cue rise ("fraction motion energy high: 0.14 at t = −1 s, 0.50 at −0.1 s, 0.78 at +0.2 s, peak +0.62 s").

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Eight cases, all handled by keeping the trial/session and marking or defaulting rather than failing. (1) **Mixed MATLAB versions** — `is_v73` dispatches to an HDF5 or a v7 reader, including the squeezed single-probe `obj.clu` layout. (2) **Three `motionEnergy` layouts** — unwrapped in a `while` loop. (3) **Missing `bp.stim`** — photostim mask defaults to all-False. (4) **NaN behavioural flags** — `np.nan_to_num` before casting to bool; trials with a non-finite go cue are dropped. (5) **`NdroppedFrames` is NaN** — the trial is skipped for that feature (→ `not_visible`), exactly as `findPosition.m` does. (6) **`frameTimes` missing or all-NaN** — the reference fallback `(1:n)/400 - 0.5` is used; frame-count/`ts`-length mismatches are truncated to the shorter vector and trials with < 2 usable frames are skipped. (7) **Missing `obj.ex.probe.loc`** — region defaults to `ALM`; unlabeled/empty cluster quality strings are kept. (8) **Trials with no spikes from any unit** (recording ended early) are dropped. Outside the tracked frames nothing is fabricated for the tongue and paws — those bins become class 2.

ii.
```python
if t < len(nd) and np.isnan(nd[t]):      # reference skips these trials
    continue
...
def frame_times_for_trial(sess, trial, vidshift, align_t):
    ft = sess['frameTimes'][trial]
    if ft.size == 0 or np.all(np.isnan(ft)):
        ft = (np.arange(1, sess['nframes'][trial] + 1) / VIDEO_FS)
        return ft - 0.5 - align_t
    return ft - vidshift - align_t
```
```python
stim = (np.nan_to_num(bp['stim_enable']).astype(bool)
        if bp['stim_enable'] is not None else np.zeros(N, bool))
valid_align = np.isfinite(align[:N])
...
def region_of_probe(sess, probe):
    locs = sess.get('probe_loc', [])
    if len(locs) >= probe:
        ...
    return ALM
```

iii. Step 10 Check 5 is a dedicated list of these edge cases with the reason for each: the `NdroppedFrames` skip and the `(1:n)/400 - 0.5` fallback because "the reference skips them in `findPosition.m`" / uses the same `catch`; the default `ALM` because "all loaders are `*_ALMVideo`"; empty quality strings kept "exactly as `findClusters.m` does"; and "Trials shorter than the window: video interpolation returns NaN outside the recorded frames → 'not visible' rather than fabricated values."

## 11-a. What are the most time-consuming steps of the code?

i. Reading the `.mat` files. The per-session timing printed by the script gives load 1.3–5.0 s, neural binning + smoothing 0.1–1.4 s and all video/motion-energy work ≈0.2–0.4 s; the whole 44-session conversion takes 11.5 s wall clock on 12 worker processes (12.2 s including pickling). Within loading, the cost is the many small HDF5 dereferences (one per cluster for `quality`/`trialtm`/`trial`, one per trial for `ts`, `frameTimes`, `NdroppedFrames`, `lickL`, `lickR`).

ii.
```python
t0 = time.time()
...
sess = load_session(fn, probes, feats)
t_load = time.time() - t0
...
timing=dict(load=t_load, neural=t_neural, video=t_video, total=time.time() - t0)
```

iii. Step 6/7: "Timing: 2 sample sessions took 3.9 s and 4.6 s each (dominated by file loading)", and the Step 7 run-time table lists file loading as the dominant per-session step with the neural binning second. The AI's response was to read only the needed probes/features ("~5-10x vs loading the whole `obj`") and to parallelise across sessions.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI reports having vectorised the two loops it considered important: the per-trial `histc` loop of `getSeq.m` (replaced by one `np.add.at` over a flat trial × bin index) and the per-trial smoothing loop (replaced by a single FFT convolution over the whole trial × time matrix). Loops that remain, and that it does not flag: (1) **two genuine per-spike Python loops** inside `bin_spikes` — `ok = np.array([t in keep_idx for t in tr])` and `for j in np.nonzero(ok)[0]: row[j] = keep_idx[tr[j]]` — which are O(n_spikes) in the interpreter and could be a single `np.isin` / lookup-array gather; (2) the loop over clusters in `bin_spikes`, which allocates and smooths a `(n_trials, 500)` matrix per unit instead of one 3-D array; (3) the per-trial loops in `feature_speed` and `motion_energy_trials` (unavoidable in the same sense as in the reference, since each trial has a different number of camera frames); (4) the per-trial/per-cluster HDF5 dereference loops in `_load_v73`.

ii. What was vectorised:
```python
flat = row[sel][inwin] * NFINE + b[inwin]
np.add.at(counts.reshape(-1), flat, 1.0)
...
out = fftconvolve(xf, kern.reshape((1,) * (xf.ndim - 1) + (N,)), mode='same', axes=-1)
```
What was not:
```python
ok = np.array([t in keep_idx for t in tr]) if tr.size else np.zeros(0, bool)
if tr.size:
    row = np.full(tr.shape, -1, dtype=np.int64)
    for j in np.nonzero(ok)[0]:
        row[j] = keep_idx[tr[j]]
```

iii. Step 6/7 lists the speed-ups and their estimated savings ("vectorised spike binning (`np.add.at`) ~20x vs per-trial histogram loops"; "FFT convolution smoothing for all trials at once ~10x"; "multiprocessing over sessions (12 workers) ~10x"). The remaining loops are not discussed; the implicit justification is the Step 7 conclusion that loading dominates and the total is ~12 s, far inside the 15-minute budget.

## 11-c. What processing does the code repeat multiple times?

i. Several cheap repetitions, none of which the notes identify. (1) `frame_times_for_trial` recomputes the same corrected side-camera frame times four times per trial — once each for `tongue`, `top_paw`, `bottom_paw` and motion energy. (2) The side camera's `frameTimes` are read out of the HDF5 file twice per trial: once into `traj[('side','tongue')]['frameTimes']` and again into `out['frameTimes']`; `NdroppedFrames` likewise. (3) `is_v73` opens each file to read 128 bytes and the reader then opens it again. (4) `discretize` recomputes `np.isfinite(values)` masks it has already formed. Things the AI does compute only once: `video_offset` (a session constant, computed once in `process_session` and passed down), each session's file load, and the shared `EDGES`/`TCENT`/`TOUT` grid and smoothing kernel, which are built once at module import and reused by every trial, session and stream.

ii.
```python
KERN = causal_kernel()
EDGES, TCENT, TOUT = time_axes()
...
vidshift = video_offset(sess)                     # once per session
tongue_sp, _ = feature_speed(sess, TONGUE_FEAT, kt, vidshift, align, fill_missing=False)
paw_sps = [feature_speed(sess, k, kt, vidshift, align, fill_missing=True)[0] for k in PAW_FEATS]
me_fine = motion_energy_trials(me, sess, kt, vidshift, align)   # each re-derives frame times
```
```python
out['frameTimes'] = [np.array(f[g['frameTimes'][i, 0]]).ravel().astype(np.float64)
                     for i in range(g['frameTimes'].shape[0])]   # already read for the tongue
```

iii. The notes' only statement in this area is the Step 6 efficiency list (read only what is needed, vectorise, parallelise); the AI does not claim or analyse redundancy elimination beyond hoisting `video_offset` out of the per-trial path. The repetitions are all O(n_frames) arithmetic or a second small HDF5 read, i.e. negligible against the 1.3–5 s file load.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Mostly in loading. Fields read on **every** run but used only by the `--show-processing` plots: `bp.ev.lickL` and `bp.ev.lickR` (one HDF5 dereference per trial each, ~700 per session) and `bp.ev.sample`, `bp.ev.delay`. Fields read and never used at all: `bp.ev.reward`, and `out['NdroppedFrames']`/`out['nframes']`-adjacent side-camera `NdroppedFrames` (the per-feature copy inside `traj` is the one `feature_speed` consults). `bp['no']` is loaded and only appears in an `assert`. `me['moveThresh']` is parsed but never used for discretisation (only recorded in metadata and plotted). On the compute side, the 500-bin fine-resolution rates, tongue/paw speeds and motion-energy traces are all built and then averaged away to 100 bins — a necessary intermediate given the reference's 10 ms smoothing, but 5× more data than is kept — and `bottom_paw`'s speed is computed at full resolution only to be averaged with `top_paw`'s. Everything else computed after loading reaches the output.

ii.
```python
for k in ['lickL', 'lickR']:
    refs = bp['ev'][k]
    ...
    for i in range(n):
        r = refs[0, i] if refs.ndim == 2 else refs[i]
        try:
            lst.append(np.array(f[r]).ravel().astype(float))
        except Exception:
            lst.append(np.array([]))
    ev[k] = lst
```
```python
ev = {k: np.array(bp['ev'][k]).ravel().astype(float)
      for k in ['bitStart', 'sample', 'delay', 'goCue', 'reward'] if k in bp['ev']}
...
out['NdroppedFrames'] = np.array([...])     # never consumed
```

iii. The notes' claim is the opposite emphasis — "Only the needed probes/DLC features are read out of the (100-300 MB) HDF5 files, instead of loading whole `obj` structs" (Step 6) — which is true at the level of probes and DLC features but not of the `bp.ev` cell arrays, which are always loaded even when `--show-processing` is off. The AI does not flag these as unnecessary; the implicit justification is again that the whole conversion runs in 12 s.
