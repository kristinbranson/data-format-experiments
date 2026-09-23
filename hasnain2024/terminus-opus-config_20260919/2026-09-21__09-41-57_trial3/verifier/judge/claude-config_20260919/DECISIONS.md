# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not glob the data directories. It treats the authors' own MATLAB loading
scripts (`/app/code/DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m`) as the
authoritative session list and *parses them programmatically* in a helper module
(`/app/session_meta.py`), reproducing the two MATLAB struct-array idioms
(`meta(end+1).anm = ...` and `meta(end+1) = meta(end)`) and stripping `%` comments so that
sessions the authors commented out are never picked up. Each parsed entry yields an animal,
a date, and the probe number(s) to analyse. Entries are then cross-referenced against
`/app/data/Ephys_Behavior` and `/app/data/RandomizedDelay_Ephys_Behavior`: 50 meta entries,
44 of which have a `data_structure_<anm>_<date>.mat` present (JEB4 ×3 and JEB5 ×3 are not in
the release). Three data files that are present but absent from the meta scripts
(`JEB23_2023-10-20`, `JEB24_2023-10-03`, `JEB24_2023-10-04`) are excluded. The two
behaviour-only opto directories (`DelayInhibition_BilatMC_Behavior`,
`GoCueInhibition_BilatMC_Behavior`, MAH* animals) are excluded because they contain no ephys.
Each session file is opened exactly once by `matio.load_obj`, which dispatches on the file's
magic bytes: `h5py` for the 36 MATLAB v7.3 files, `scipy.io.loadmat` for the 11 v5 files, and
returns the *same* nested-Python structure either way (struct→dict, cell→object ndarray,
char→str, numeric arrays transposed back to MATLAB column-major order). `clu.spkWavs` is
skipped at read time. Sessions are processed in a `multiprocessing.Pool` (8–16 workers), so
the whole conversion takes ~18 s.

ii.
```python
def parse_meta_scripts(meta_dir=META_DIR):
    for fn in sorted(glob.glob(os.path.join(meta_dir, '*.m'))):
        entries = []
        for line in open(fn):
            l = line.split('%')[0].strip()          # commented-out sessions disappear
            if not l: continue
            if 'meta(end+1)' in l:
                if re.search(r'meta\(end\+1\)\s*=\s*meta\(end\)', l):
                    entries.append(dict(entries[-1]) if entries else {}); continue
                entries.append({})
            ...
            m = re.search(r"\.probe\s*=\s*(\[[^\]]*\]|\d+)", l)
            if m: entries[-1]['probe'] = [int(x) for x in re.findall(r'\d+', m.group(1))]
```

```python
def get_sessions():
    """sessions listed in the reference meta scripts AND present in /app/data."""
    out = []
    for s in parse_meta_scripts():
        f = find_data_file(s['anm'], s['date'])
        if f is None: continue
        s = dict(s); s['datafile'] = f
        s['mefile'] = find_me_file(s['anm'], s['date'])
        s['dataset'] = ('randomized_delay' if 'RandomizedDelay' in f else 'fixed_delay')
        s['sessid'] = f"{s['anm']}_{s['date']}"
        out.append(s)
    return out
```

```python
SKIP_FIELDS = {'spkWavs', 'spkwavs'}

def load_obj(fn, field='obj'):
    """Load a session data object from either mat file version."""
    if is_v73(fn):
        return load_v73(fn, field)
    return load_v5(fn, field)
```

```python
jobs = [(s, args.show_processing, outdir) for s in sessions]
nw = min(args.nworkers, len(jobs))
if nw > 1:
    with Pool(nw) as pool:
        results = pool.map(_worker, jobs)
```

iii. From CONVERSION_NOTES.md: "The authoritative session list and the ALM probe per session
are given by the reference meta scripts `load<ANM>_ALMVideo.m`". The AI notes that 3 data
files are not in the meta list "and are therefore excluded, exactly as the authors did", and
that the MAH* directories are excluded because "they contain no neural data". Both `.mat`
readers are needed because "File formats are mixed: `data_structure` files are MATLAB v7.3
(HDF5) for 36 sessions and MATLAB v5 for 11". `spkWavs` is skipped because "Loading
`clu.spkWavs` (spike waveforms) is unnecessary and dominates file reading".

---

## 1-b. How are the data split into subjects?

i. The subject is the `meta.anm` field parsed out of the reference `.m` scripts (identical to
the filename prefix, e.g. `JEB19_2023-04-19` → `JEB19`). Each session carries `sess['anm']`
through processing. At assembly, `subjects` is built in first-encounter order as sessions are
appended, and `subject_idx` is that list index per session. Result: 14 mice
(EKH1, EKH3, JEB6, JEB7, JEB11, JEB12, JEB13, JEB14, JEB15, JEB19, JEB23, JEB24, JGR2, JGR3)
— 10 in the fixed-delay set and 4 in the randomized-delay set. The animal identity is never
read out of `obj.meta.anm`, which is missing in several sessions.

ii.
```python
anm = sess['anm']
if anm not in data['subjects']:
    data['subjects'].append(anm)
data['subject_idx'].append(data['subjects'].index(anm))
...
data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. The AI's Step 5 mapping table lists `meta.anm → subjects, subject_idx`, "14 mice". Its
Step 4 consistency table records the one discrepancy it found and how it resolved it: "the
reference meta scripts for the fixed-delay dataset list 10 animals; the paper text says nine
mice for the DR task. We follow the authors' own session list. (One animal, e.g. JGR3 with a
single session, may have been grouped or excluded in the text.)"

---

## 1-c. How are the data split into sessions?

i. One session = one entry of the parsed meta list = one `data_structure_<anm>_<date>.mat`
file, and becomes one element of `neural`, `input`, `output`, `brain_region_idx`, and
`subject_idx`. `find_data_file` searches both task folders, so fixed-delay and
randomized-delay sessions are handled uniformly as one pooled dataset (the `dataset` label is
retained in `metadata.session_info` only). 44 sessions: 25 fixed-delay, 19 randomized-delay.
Two extra session-level quality gates are applied at assembly time: a session is dropped if it
has fewer than 10 curated units (the paper's stated inclusion rule) or fewer than 2 trials
(the decoder format requirement). In the full run neither gate fired — `skipped: []`, all 44
sessions retained, 17–141 units each.

ii.
```python
def find_data_file(anm, date):
    for d in DATA_DIRS:
        fn = os.path.join(d, f'data_structure_{anm}_{date}.mat')
        if os.path.exists(fn):
            return fn
    return None
```

```python
PARAMS = dict(..., min_units=10,   # "sessions were included only if they had at least 10 units"
              ...)
...
if info['nunits'] < PARAMS['min_units']:
    nskipped.append((sess['sessid'], f"only {info['nunits']} units"))
    print(f"  SKIP {sess['sessid']}: {info['nunits']} units < {PARAMS['min_units']}")
    continue
if info['ntrials_used'] < 2:
    nskipped.append((sess['sessid'], 'fewer than 2 trials'))
    continue
```

iii. "All 44 ALM ephys+video sessions listed in the reference meta scripts and present in
`/app/data` (25 fixed-delay, 19 randomized-delay), further required to have >= 10 curated
units (paper: 'Recording sessions were included for analysis only if they had at least 10
units')." The 3 randomized-delay files absent from the meta list are excluded because "3 extra
files were excluded by the authors".

---

## 1-d. How are the data split into trials?

i. Trials come straight from the Bpod table. `bp.Ntrials` sets N, and every per-trial field
(`hit`, `miss`, `no`, `R`, `L`, `early`, `autowater`, `stim.enable`, `ev.goCue`) is read with
`as_vector(...)[:N]`, truncating any field stored longer than the trial count. Trial numbers
are kept 1-based throughout ("as in MATLAB") so they can index back into the raw arrays;
`obj.clu[*].trial` and `obj.traj{view}(trial)` are indexed with those same numbers
(`traj_trials[tr - 1]`, `trial_pos[tr]`). No trial boundaries are reconstructed. Each
surviving trial becomes one `(n_neurons, 500)` neural array, one `(1, 500)` input array and
one `(6, 500)` output array.

ii.
```python
bp = obj['bp']
N = int(as_vector(bp['Ntrials'])[0])
hit = as_vector(bp['hit'])[:N].astype(bool)
miss = as_vector(bp['miss'])[:N].astype(bool)
no = as_vector(bp['no'])[:N].astype(bool)
R = as_vector(bp['R'])[:N].astype(bool)
L = as_vector(bp['L'])[:N].astype(bool)
early = as_vector(bp['early'])[:N].astype(bool)
aw = as_vector(bp['autowater'])[:N].astype(bool)
stim = as_vector(bp['stim']['enable'])[:N]
stim = np.nan_to_num(stim).astype(bool)
gocue = as_vector(bp['ev'][PARAMS['alignEvent']])[:N]
...
trials = np.where(keep)[0] + 1          # 1-based trial numbers, as in MATLAB
align_times = gocue[trials - 1]
```

iii. The AI's Step 2 notes document `obj.bp` as "Bpod / trial info: `Ntrials`, `hit`, `miss`,
`no`, `R`, `L`, `early`, `autowater`, `stim.enable` … and `obj.bp.ev` with … `goCue` (all
seconds *within trial*)", and `obj.clu` as carrying `trial` ("trial index, 1-based"). The
sanity checks in Step 10 include "trial count after curation matches an independent
recomputation" (PASS on 3 sessions).

---

## 1-e. How are trials filtered based on quality controls?

i. Two curation filters plus two defensive guards.
**(1)** Early-lick trials (`bp.early`) and photostimulation trials (`bp.stim.enable`) are
dropped, following the paper's "omitted from all analyses"; `stim.enable` is `nan_to_num`'d
first because some sessions store NaN there.
**(2)** Guards: the go cue must be finite, and the trial must be flagged as exactly one of
hit/miss/ignore (`hit | miss | no`). I verified both guards are no-ops on this release (0
non-finite go cues, 0 trials with none of the three flags set).
**(3)** After spike binning, trials in which *no* quality-passing unit fired a single spike
anywhere in the trial are dropped. This was added in Step 9 in response to 61
"all neural data is zero" warnings in `JEB24_2023-10-23` (trials 293–320) and
`JEB24_2023-11-03` (trials 302–334), where the probe stopped before the behavioural session
ended.
**(4)** Ignore trials are deliberately *retained*, unlike in the paper, because the decoder
spec requires an `ignore` outcome class and a `none` lick-direction class.
Net: 14,972 raw trials → 13,762 kept (976 early-lick, 187 opto, 61 unrecorded; categories
overlap slightly).

ii.
```python
# --- trial curation: exclude early-lick and optogenetic stim trials
keep = (~early) & (~stim) & np.isfinite(gocue) & (hit | miss | no)
trials = np.where(keep)[0] + 1          # 1-based trial numbers, as in MATLAB
```

```python
# Some sessions' ephys recording stops before the behavioural session ends; those
# trials contain no spikes from any unit and carry no neural information, so they
# are dropped (data-quality curation).
recorded = spk_per_trial > 0
n_not_recorded = int((~recorded).sum())
if n_not_recorded:
    trials = trials[recorded]
    align_times = align_times[recorded]
    rates = rates[recorded]
    ntr = trials.size
```

iii. Step 4: "Early/stim trials … 'omitted from all analyses' → Exclude `early` and
`stim.enable` trials." For ignore trials: "Keep: the decoder task explicitly asks for an
`ignore` outcome class and a `none` lick direction. Documented as a required deviation."
For the recording-length cut: "In those sessions the ephys recording stopped before the
behavioural session ended, so the last trials contain no spikes from any unit. Such trials
carry no neural information and are now dropped … After the fix the verification reports no
warnings." It is listed as difference #3 in the Step 10 reference-code comparison:
"not in the reference code, but necessary because those trials carry no neural data at all."

---

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` for the probe(s) named by the reference meta script for that session — the
spike-sorted clusters, using three of their fields: `trial` (1-based trial each spike belongs
to), `trialtm` (spike time relative to the start of that trial, on the behaviour clock), and
`quality` (the manual curation label). `obj.bp.ev.goCue` supplies the alignment event, and
`obj.ex.probe.loc` / `obj.meta.probe.loc` supplies the recording location used to label each
unit's brain region. Units from both probes of a dual-probe session are concatenated into one
population.

ii.
```python
for p in probes:
    clusters = get_clusters(obj, p)
    keep = [i for i, c in enumerate(clusters)
            if str(c.get('quality', '')).strip().lower() not in BAD_QUALITY]
    ...
    for j, ci in enumerate(keep):
        c = clusters[ci]
        tr = np.asarray(c['trial'], dtype=np.int64).ravel()
        tm = np.asarray(c['trialtm'], dtype=float).ravel()
```

```python
def probe_regions(obj, probes):
    """Brain region label for each requested probe, from obj.ex/obj.meta probe.loc."""
    for key in ('ex', 'meta'):
        d = obj.get(key)
        if isinstance(d, dict) and isinstance(d.get('probe'), dict) and 'loc' in d['probe']:
            ...
    for p in probes:
        name = 'ALM'
        if locs is not None and len(locs) >= p:
            l = locs[p - 1].upper()
            if 'M1TJ' in l or 'TJM1' in l: name = 'tjM1'
            elif 'ALM' in l:               name = 'ALM'
```

iii. Step 2: "`obj.clu{probe}` struct array over sorted units: `quality` (str), `tm` (session
time), `trialtm` (time within trial), `trial` (trial index, 1-based)". The probe-location read
was added after a dedicated audit: "the reference meta includes probe 2 for the three
dual-probe JEB15 sessions and the only probe of JEB15_2022-07-29, which `obj.ex.probe.loc`
labels `L M1TJ`. We keep those units (the authors' own meta includes them) but label them
`tjM1` in `brain_regions` rather than mislabelling them ALM. 2,311 of 2,456 units are ALM."
(I confirmed directly from the files that JEB15's probe 2 is indeed `L M1TJ`.)

---

## 2-b. How is the `neural` data processed?

i. A direct port of `alignSpikes.m` + `getSeq.m` + `mySmooth.m`. Spikes are aligned by
subtracting the trial's go cue, histogrammed into the 10 ms bin grid with a single vectorised
`np.add.at` per cluster (instead of the reference's per-trial `histc` loop), divided by `dt`
to give spikes/s, and then smoothed along time with the reference's **causal** Gaussian:
`gausswin(15, 2.5)` with its first `floor(15/2)=7` taps zeroed and the rest renormalised,
applied with `'reflect'` boundary handling. Smoothing is done for all trials and units at once
by reshaping to `(n_bins, n_trials·n_units)`. No normalisation, baseline subtraction, or
z-scoring is applied; stored values are firing rates in Hz, `float32`, shape
`(n_neurons, 500)` per trial. I verified the port against `utils/mySmooth.m` line by line
(kernel formula, causal zeroing index, `conv(...,'same')`, and the `trim = N + 1` 1-based
trim ≡ `trim = N` 0-based) — it is faithful.

ii.
```python
def _gausswin(N, alpha=2.5):
    """MATLAB gausswin(N, alpha)."""
    n = np.arange(N) - (N - 1) / 2.0
    return np.exp(-0.5 * (alpha * 2 * n / (N - 1)) ** 2)

def causal_gauss_kernel(N=PARAMS['smooth']):
    """mySmooth.m kernel: gausswin with the first half zeroed (causal), normalised."""
    k = _gausswin(N)
    k[:N // 2] = 0.0
    return k / k.sum()

def my_smooth(x, N=PARAMS['smooth'], bctype=PARAMS['bctype']):
    """Port of utils/mySmooth.m. Operates along the first axis of x (time)."""
    ...
    if bctype == 'reflect':
        xf = np.concatenate([x[:N, :], x], axis=0); trim = N
    ...
    for j in range(xf.shape[1]):
        out[:, j] = np.convolve(xf[:, j], k, mode='same')
    out = out[trim:, :]
```

```python
aligned = tm[sel] - align_times[pos]
b = np.floor((aligned - edges[0]) / PARAMS['dt']).astype(np.int64)
good = (b >= 0) & (b < nb)
np.add.at(counts[:, :, j], (pos[good], b[good]), 1.0)
...
rates = counts / PARAMS['dt']
flat = rates.transpose(1, 0, 2).reshape(nb, -1)
sm = my_smooth(flat)
rates = sm.reshape(nb, ntr, -1).transpose(1, 0, 2).astype(np.float32)
```

iii. "Smoothed firing rates (causal Gaussian, 15 bins) rather than raw spike counts: this is
what `getSeq.m` produces (`obj.trialdat`) and what the paper's decoders were trained on."
`my_smooth()` is described as "a line-by-line port of `utils/mySmooth.m` (MATLAB
`gausswin(15)` with the first half zeroed = causal, normalised, `reflect` boundary handling)".
Verified numerically in Step 10 Check 2: "firing rate of 9 random (trial, unit) pairs,
recomputed with `np.histogram` + an independent causal-Gaussian smoother … PASS (max abs diff
~7e-6, float32 rounding)".

---

## 2-c. How is the `neural` data filtered based on quality controls?

i. Three filters.
**(1)** Cluster quality: the label is coerced to `str`, stripped, lower-cased, and dropped if
it is in `{garbage, gabrga, noisy, real?, ''}`. This is exactly the exclusion set of
`findClusters.m` under `params.quality = {'all'}` (with `''` added), matched case-insensitively
"because JEB7 uses 'Good'/'Multi' capitalisation". Notably `poor`, `fair`, and `multi` units
are **kept**, as `findClusters.m` keeps them. Across all 44 sessions there are 10,330 clusters;
7,817 are dropped by this rule, leaving 2,513.
**(2)** Firing rate: units whose mean smoothed rate over trials × bins is ≤ 1 Hz are dropped
(`removeLowFRClusters.m`, computed on the retained trials after the no-spike-trial cut).
2,513 → 2,456 units, 17–141 per session, mean 55.8.
**(3)** Session level: a session with fewer than 10 curated units would be dropped (none was).

ii.
```python
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?', ''}
PARAMS = dict(..., lowFR=1.0, min_units=10, ...)
...
keep = [i for i, c in enumerate(clusters)
        if str(c.get('quality', '')).strip().lower() not in BAD_QUALITY]
```

```python
# low firing rate filter (removeLowFRClusters.m; methods: FR > 1 Hz)
if rates.shape[2]:
    meanfr = rates.mean(axis=(0, 1))
    keep_u = meanfr > PARAMS['lowFR']
...
rates = rates[:, :, keep_u]
```

iii. "Neuron curation rules: drop units whose quality label is garbage/noisy/real?
(findClusters with params.quality={'all'}); then drop units with mean firing rate <= 1 Hz
(params.lowFR = 1 in the tutorial, and the methods state 'All units with firing rates
exceeding 1 Hz were included')." Step 4 resolves the conflict between the two parameter files:
"lowFR threshold | 1 Hz in WorkingWithDataObjs.m, 0.5 Hz in getDefaultParams.m | 'firing rates
exceeding 1 Hz' | Use **1 Hz** (tutorial + methods agree)." The case-insensitive comparison is
listed as difference #4: "the MATLAB `ismember` test is case-sensitive and would have kept a
'Garbage' unit. No session in the release actually uses capitalised 'Garbage', so this changes
nothing, but it is safer." The AI reports the resulting total as 2,456 against the paper's
1,651 + 845 = 2,496 (98%).

---

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. One subtraction, exactly as `alignSpikes.m`. `clu.trialtm` is already on the behaviour
clock and already relative to its own trial's start, and `bp.ev.goCue` is on the same clock, so
`trialtm − goCue[trial]` gives seconds from the go cue with no offset or interpolation. The
per-spike lookup is vectorised: a `trial_pos` scatter array maps 1-based trial number →
position in the curated trial list, and `align_times[pos]` gathers each spike's own trial's go
cue. Spikes whose bin index falls outside `[0, 500)` are discarded. The go cue is used for WC
(autowater) trials too, where it marks the water-drop time.

ii.
```python
trial_pos = -np.ones(int(max(trials.max(), 0)) + 2, dtype=np.int64)
trial_pos[trials] = np.arange(ntr)
...
pos = trial_pos[tr]
sel = pos >= 0
...
aligned = tm[sel] - align_times[pos]
b = np.floor((aligned - edges[0]) / PARAMS['dt']).astype(np.int64)
good = (b >= 0) & (b < nb)
```

iii. "Neural: spike times within a trial (`clu.trialtm`) minus the alignment event,
histogrammed on `tmin:dt:tmax` edges, divided by dt (spikes/s), smoothed with a causal
Gaussian (`mySmooth`)." Key decision 1: "**Alignment to go cue for every trial, including WC
trials**: `bp.ev.goCue` is defined on WC trials as the water-drop time; the paper always refers
to 'the go cue/water drop' and the reference code aligns all conditions with
`params.alignEvent='goCue'`." Validated empirically: "the ALM population rate peaks at
t = +0.1 s".

---

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **10 ms bins, 500 bins, spanning −2.5 to +2.5 s from the go cue.** The grid is built once
by `time_axis()`, which reproduces `getSeq.m` exactly: `edges = tmin:dt:tmax`, bin centres
`edges + dt/2` with the last edge dropped, giving centres −2.495 … +2.495 s. The same 500-bin
axis is used for every trial, session, and data stream (neural, input, tongue, paw, motion
energy), so no cross-stream resampling is ever needed. `metadata['time_bin_size'] = 10.0` ms.
No rebinning is applied after binning — the neural data is histogrammed directly at 10 ms, and
the camera streams are interpolated onto this grid once. The AI chose 10 ms over the 5 ms of
`getDefaultParams.m` because the tutorial `WorkingWithDataObjs.m` (and, as I verified, most of
the paper's own figure scripts including all the Figure 3 decoding scripts) uses
`params.dt = 1/100`.

ii.
```python
PARAMS = dict(alignEvent='goCue', tmin=-2.5, tmax=2.5,
              dt=0.01,             # 10 ms bins
              smooth=15, bctype='reflect', lowFR=1.0, min_units=10,
              advance_movement=0.0)

def time_axis():
    """getSeq.m: edges = tmin:dt:tmax; time = edges + dt/2, dropping the last edge."""
    edges = np.arange(PARAMS['tmin'], PARAMS['tmax'] + 1e-12, PARAMS['dt'])
    tm = edges[:-1] + PARAMS['dt'] / 2
    return edges, tm
```

iii. Step 4 discrepancy table: "dt | 1/100 s in tutorial, 1/200 s in getDefaultParams | Use the
tutorial value, 10 ms (also what all figure scripts use)." Key decision 2: "**Window [-2.5,
+2.5] s, dt = 10 ms (500 bins)**: the reference parameters. This covers the ITI/sample/delay
epochs before the go cue and the response epoch after it." Sanity-checked in Step 10:
"input time axis == -2.495:0.01:2.495 (500 bins) | all | PASS".

---

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Not from any raw variable — it is the conversion's own time axis, defined by the alignment
event (`bp.ev.goCue`, the event every trial is aligned to) and the window/binning parameters.
The input is the vector of the 500 bin centres, identical on every trial of every session:
−2.495 to +2.495 s in 10 ms steps. `input_names = ['time_from_go_cue']`, `d_input = 1`, stored
as `float32` with shape `(1, 500)`.

ii.
```python
INPUT_NAMES = ['time_from_go_cue']
...
edges, taxis = time_axis()
...
T = taxis.size
tin = taxis.astype(np.float32)[None, :]
for i in range(ntr):
    ...
    inp.append(tin.copy())
```

iii. Step 5 mapping table: "bin centre times → `input[sess][trial]` (1, T) | t = -2.495 …
+2.495 s relative to go cue | `getSeq.m` (obj.time) | required decoder input: 'Time from go cue
onset in seconds'". The Step 10 reference comparison notes for input construction: "(the
reference has no decoder input; `obj.time` is the time axis) … `input = obj.time` = time from
the go cue | required by the decoder spec".

---

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None beyond constructing the axis. `np.arange(-2.5, 2.5+1e-12, 0.01)` gives the 501 edges,
`edges[:-1] + dt/2` the 500 centres. The tiny `1e-12` pad guards against floating-point
truncation of the last edge. Values are cast to `float32` and the same row is copied once per
trial (`tin.copy()`), so each trial owns a contiguous `(1, 500)` array. Because the decoder
spec calls this input "continuous, time-varying", it is stored as the continuous ramp rather
than as a binary event marker.

ii.
```python
edges = np.arange(PARAMS['tmin'], PARAMS['tmax'] + 1e-12, PARAMS['dt'])
tm = edges[:-1] + PARAMS['dt'] / 2
```

iii. `time_axis()` is documented as reproducing `getSeq.m`: "`time_axis()` reproduces
`getSeq.m`: `edges = tmin:dt:tmax`, bin centres `edges+dt/2` (last dropped) → 500 bins of
10 ms covering -2.495 … +2.495 s". Planned sanity check: "Input: time vector equals
`-2.5+dt/2 : dt : 2.5-dt/2`, min/max = -2.495/2.495" — reported PASS.

---

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid. `time_axis()` returns both `edges` (passed to
`bin_spikes`, which histograms go-cue-relative spike times into them) and `taxis` (the centres,
stored as the input), so bin *k* of the input is by construction the same 10 ms interval as bin
*k* of the neural array — and of the three camera outputs, which are interpolated onto
`taxis`. No shift or interpolation can introduce a lag.

ii.
```python
edges, taxis = time_axis()
...
rates, quals, spk_per_trial, units_per_probe = bin_spikes(obj, sess['probe'], trials,
                                                          align_times, edges)
...
Xt, Yt = interp_positions(side, tongue_ix, trials, align_times, taxis, vidshift)
...
ME = motion_energy_on_axis(me_trials, side, trials, align_times, taxis, vidshift)
...
tin = taxis.astype(np.float32)[None, :]
```

iii. Implicit in the design — one grid is computed per session and handed to every stream. The
AI's Step 10 sanity checks confirm the result: "input of a random trial equals the time axis |
3 sessions | PASS", and the alignment of the other streams to it is checked by the
post-go-cue jumps in tongue visibility, motion energy, and firing rate.

---

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial fields of `obj.bp`: the instructed side `R` and `L`, and the outcome flags
`hit`, `miss`, `no`. Lick direction is not recorded as such, so it is inferred from the
instructed side combined with whether the animal was correct. `bp.ev.lickL` / `bp.ev.lickR`
(the actual lick-contact times) are *not* used for the label, but the AI did use them once as
an independent cross-check.

ii.
```python
hit = as_vector(bp['hit'])[:N].astype(bool)
miss = as_vector(bp['miss'])[:N].astype(bool)
no = as_vector(bp['no'])[:N].astype(bool)
R = as_vector(bp['R'])[:N].astype(bool)
L = as_vector(bp['L'])[:N].astype(bool)
```

iii. Step 5 mapping table: "`bp.R/L/hit/miss/no` → `output[0]` lick direction … reference code
`getPrevChoice.m` (choice definition) | verified to agree with the port of the first
post-go-cue lick contact on 99-100% of trials". The AI explicitly went to
`funcs/getPrevChoice.m` for the authors' own definition of choice (trajectory step 75).

---

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A vectorised three-way relabelling: an animal that was correct licked the instructed port,
an animal that erred licked the other port, and an ignore trial has no lick. So
`right = (R & hit) | (L & miss)`, and the class is 2 (`none`) on `no` trials, else 1 (`right`)
if `right`, else 0 (`left`). Codes follow the spec's left/right/none ordering,
`output_values[0] = ['left', 'right', 'none']`. The per-trial scalar is then broadcast across
all 500 bins so that all six outputs live in one `(6, 500)` array (the instructions prefer
time-varying outputs). Full-dataset distribution: left 0.423, right 0.446, none 0.131.

ii.
```python
OUTPUT_VALUES = [
    ['left', 'right', 'none'],
    ...
]
...
tr0 = trials - 1
right = (R[tr0] & hit[tr0]) | (L[tr0] & miss[tr0])
lick = np.where(no[tr0], 2, np.where(right, 1, 0))          # 0 left, 1 right, 2 none
...
o = np.empty((6, T), dtype=np.int64)
o[0] = lick[i]
```

iii. "right = (R&hit)|(L&miss); left = otherwise; none = `no` (ignore) trials", cross-referenced
to `getPrevChoice.m`. The AI validated the inference against the animal's actual licking:
"verified to agree with the port of the first post-go-cue lick contact on 99-100% of trials".
Step 10 Check 2 re-derives it from the raw files: "lick direction == (R&hit)|(L&miss) → right,
`no` → none, from raw bp | 3 sessions | PASS", and "per-trial outputs constant over time |
3 sessions | PASS".

---

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial flag, `obj.bp.autowater`, which marks the trials on which water was delivered
at a random port without any auditory cue — the water-cued (WC) context. The AI also checked
the alternative field `bp.autowaterBlock` where it exists and found the two agree.

ii.
```python
aw = as_vector(bp['autowater'])[:N].astype(bool)
```

iii. Step 1 notes: "autowater==1 → water-cued (WC) block; autowater==0 → delayed-response (DR)
block", read off the reference condition strings (`'R&hit&~stim.enable&autowater&~early'`
etc.). Step 5 mapping: "`bp.autowater` → `output[1]` context … equals `bp.autowaterBlock` where
that field exists."

---

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling of the boolean: autowater → WC (0), otherwise DR (1), matching the
spec's `WC, DR` ordering. Broadcast across the 500 bins like the other per-trial outputs.
Full-dataset distribution: WC 0.097, DR 0.903 — the AI cross-checked this against the raw
count (1,548/14,972 = 10.3% autowater trials). 12 of the 44 sessions are pure DR, so `context`
is constant within them.

ii.
```python
OUTPUT_VALUES = [..., ['WC', 'DR'], ...]
...
context = np.where(aw[tr0], 0, 1)                            # 0 WC, 1 DR
...
o[1] = context[i]
```

iii. Step 5 mapping: "0 = DR, 1 = WC" [sic — the code and `output_values` are WC = 0, DR = 1,
which is what the spec asks for]. Step 10 Check 2: "context == bp.autowater | 3 sessions |
PASS". Step 10 Check 5 flags the constant-context sessions and justifies keeping them: "12
sessions contain no WC trials (pure DR sessions) so `context` is constant within them; this is
a property of the experiment, not a bug, and the decoder is trained across sessions."

---

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Three per-trial flags of `obj.bp`: `hit`, `miss`, and `no`. Unlike the neighbouring
lick-direction derivation, `no` is read explicitly rather than inferred as "neither hit nor
miss". (I confirmed the three flags are mutually exclusive and sum to 1 on every trial, so the
two formulations are equivalent.) The same three arrays already loaded for lick direction are
reused.

ii.
```python
hit = as_vector(bp['hit'])[:N].astype(bool)
miss = as_vector(bp['miss'])[:N].astype(bool)
no = as_vector(bp['no'])[:N].astype(bool)
```

iii. Step 5 mapping: "`bp.hit/miss/no` → `output[2]` outcome | 0 = incorrect (miss),
1 = correct (hit), 2 = ignore (no) | `getOutcome.m`". The AI had read `funcs/getOutcome.m` and
recorded that it computes "outcome = bp.hit, with NaN on ignore (bp.no) trials", i.e. the
authors also treat ignore as a separate, non-binary state.

---

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A three-way relabelling matching the spec's incorrect/correct/ignore ordering: 2 (`ignore`)
on `no` trials, else 1 (`correct`) on hits, else 0 (`incorrect`). Broadcast across the 500
bins. Full-dataset distribution: incorrect 0.120, correct 0.749, ignore 0.131 — the AI checked
the 74.9% correct rate against the raw `bp.hit` fraction (0.75 of curated trials) and against
the paper's typical performance. Ignore trials are retained rather than dropped, a documented
deviation from the paper, so that the required `ignore` class exists.

ii.
```python
OUTPUT_VALUES = [..., ['incorrect', 'correct', 'ignore'], ...]
...
outcome = np.where(no[tr0], 2, np.where(hit[tr0], 1, 0))     # 0 incorrect,1 correct,2 ignore
...
o[2] = outcome[i]
```

iii. Step 4: "Ignore trials | excluded from the paper's choice conditions | ~10% of trials |
'ignore … omitted' from behavioural analyses | Keep: the decoder task explicitly asks for an
`ignore` outcome class and a `none` lick direction. Documented as a required deviation."
Step 10 Check 2: "outcome == bp.hit/miss/no | 3 sessions | PASS". Step 9 consistency table:
"Outcome distribution | ~70-75% correct typical | hit 0.75 of curated trials | correct 0.749,
incorrect 0.120, ignore 0.131 | YES".

---

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj{1}` — the **side camera only** — feature `tongue`.
Per trial the AI reads `ts` (shape `(n_frames, [x, y, likelihood], n_features)` after the
loader restores MATLAB's ordering), `frameTimes`, and `featNames` (to find the feature index by
name rather than by a hard-coded position). `obj.sglx.bitcode.bitstart`, `obj.sglx.fs`, and
`obj.bp.ev.bitStart` are also needed, to put the camera frames on the behaviour clock, plus
`bp.ev.goCue`. The bottom camera's `top_tongue`/`bottom_tongue` features are read for the
feature inventory but are **not** used for the tongue output.

ii.
```python
side = get_traj_view(obj, 1)
bottom = get_traj_view(obj, 2)
side_feats = get_feat_names(side[0])
bot_feats = get_feat_names(bottom[0])

tongue_ix = side_feats.index('tongue')
Xt, Yt = interp_positions(side, tongue_ix, trials, align_times, taxis, vidshift)
```

```python
t = traj_trials[tr - 1]
ts = np.asarray(t['ts'], dtype=float)
...
for k, arr in ((0, X), (1, Y)):
    v = ts[:, k, featix]
```

iii. From the trajectory (step 80), where the AI weighed the two views explicitly: "which view
for tongue? The reference kinematic feature set includes both. … Paper: 'Tongue angle and
length were found using the bottom camera'. For velocity magnitude, either. Simpler and better
justified: use the side camera 'tongue' (view 1) for tongue velocity (captures protrusion
up/down and forward)". Step 5 mapping: "`obj.traj{1}` tongue x/y (DLC) → `output[3]` tongue
velocity … tongue is NOT smoothed and NOT nan-filled, exactly as the reference does".

---

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps, ordered as in `findPosition.m` → `findVelocity.m` (interpolate first, then
differentiate).
**(1)** Per trial, the side camera's `frameTimes` are converted to seconds from the go cue and
`x`, `y` are linearly interpolated onto the shared 500-bin axis with `np.interp`, returning NaN
outside the covered range. Because `np.interp` silently bridges NaNs, a *companion*
interpolation of the NaN mask is run and used to re-impose NaN wherever the tongue was not
tracked. No smoothing of the tongue position — matching `findPosition.m`, which skips
smoothing for any feature whose name contains "tongue".
**(2)** No explicit likelihood threshold is applied; the AI relies on the authors having
already set `x`/`y` to NaN where the DLC likelihood is low. I verified this is exactly
equivalent to the likelihood > 0.9 rule: on every trial I checked, `isnan(x)` and
`likelihood <= 0.9` agree element-for-element.
**(3)** Speed is the magnitude of the central difference of the interpolated `x` and `y`
(`np.gradient`-style, unit bin spacing, so units are px/bin — irrelevant to a percentile
split). A custom `_gradient_nan` falls back to a one-sided difference where one neighbour is
NaN, so the first and last bin of every lick keep a defined velocity; where even that fails at
an isolated visible bin the speed is set to 0, following `findVelocity.m`'s "set tongue
velocity to 0 if not visible".
**(4)** Visibility is `isfinite(X) & isfinite(Y)`; non-visible bins become class 2.
Result: 9.1% of bins have a visible tongue.

ii.
```python
tt = ft - vidshift - align_times[i]
for k, arr in ((0, X), (1, Y)):
    v = ts[:, k, featix]
    arr[i] = np.interp(taxis, tt, v, left=np.nan, right=np.nan)
    # np.interp does not propagate NaNs, do it explicitly
    nanmask = np.interp(taxis, tt, np.isnan(v).astype(float),
                        left=1.0, right=1.0) > 0
    arr[i][nanmask] = np.nan
```

```python
def _gradient_nan(A):
    """Central difference along axis 1, falling back to a one-sided difference when
    one neighbour is missing (e.g. at the first/last visible frame of a lick)."""
    fwd[:, :-1] = A[:, 1:] - A[:, :-1]
    bwd[:, 1:]  = A[:, 1:] - A[:, :-1]
    central = 0.5 * (fwd + bwd)
    out = np.where(np.isfinite(central), central,
                   np.where(np.isfinite(fwd), fwd, bwd))
    out[~np.isfinite(A)] = np.nan
    return out

def speed_from_positions(X, Y):
    """findVelocity.m: velocity = gradient of the interpolated position; speed = |v|."""
    vx = _gradient_nan(X); vy = _gradient_nan(Y)
    return np.sqrt(vx ** 2 + vy ** 2)
```

```python
tongue_visible = np.isfinite(Xt) & np.isfinite(Yt)
# findVelocity.m sets the tongue velocity to 0 wherever it cannot be computed;
# here that only happens for an isolated visible frame (no neighbour to difference
# against). Those timepoints stay 'visible' with zero speed.
tongue_speed = np.where(tongue_visible & ~np.isfinite(tongue_speed), 0.0, tongue_speed)
```

iii. Key decision 6: "**Tongue 'not visible'** = DLC NaN for the tongue feature at that
timepoint, which is exactly how the reference code detects licks (`extractAllLicks.m`,
`setTongueBaselinePosition`)." The NaN-mask interpolation is documented as: "`np.interp` does
not propagate NaN, so a companion interpolation of the NaN mask is used to mark 'not visible'
timepoints." Two bugs found and fixed during development are documented: "MATLAB `gradient`
(central difference) returns NaN next to NaNs, which incorrectly labelled the first and last
timepoint of every lick as 'tongue not visible' (visible fraction 0.039 instead of 0.078)";
and "the tongue 'visible' fraction in the pickle (0.080) was lower than the raw DLC visible
fraction (0.087) because an isolated visible frame has no neighbour for the difference …
Following `findVelocity.m` … the speed is now set to 0 at visible timepoints where it cannot
be differenced. After the fix the masks match exactly."

---

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A single `discretize()` helper is used for all three movement outputs. The threshold is the
**50th percentile of the speeds at visible timepoints, pooled over all trials of that session**
(so one threshold per session, as the spec requires). Bins at or above it become 1, below it 0,
and non-visible bins become 2. The per-session threshold is recorded in
`metadata.session_info[*].tongue_thr`. Because the split is a median of the visible values, the
class-0 and class-1 fractions come out equal by construction: 0.046 / 0.046 / 0.909 over the
full dataset, and equal to within rounding in every individual session.

ii.
```python
def discretize(values, visible):
    """0 = < session median, 1 = >= session median, 2 = not visible/no video."""
    out = np.full(values.shape, 2, dtype=np.int64)
    vis = visible & np.isfinite(values)
    if np.any(vis):
        thr = np.percentile(values[vis], 50)
        out[vis] = (values[vis] >= thr).astype(np.int64)
    else:
        thr = np.nan
    return out, thr
...
tongue_cls, tongue_thr = discretize(tongue_speed, tongue_visible)
```

iii. Key decision 5: "**Discretisation of the three continuous movement outputs** uses the
per-session 50th percentile of the values at *visible* timepoints (class 2 = not visible / no
video), as specified by the decoder task." Listed as difference #2 in the reference-code
comparison: "**Median-split discretisation** of tongue/paw velocity and motion energy —
required by the decoder task. (The reference instead thresholds motion energy at a manually
chosen per-session `moveThresh`.)" Checked in Step 10: "tongue classes 0/1 are an exact median
split | 3 sessions | PASS", and in Step 9: "Movement medians | median split | tongue
0.040/0.040 … | YES (exact median split)".

---

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The camera runs on its own clock. The offset is computed **once per session** by a direct
port of `findVideoOffset.m`: the mode of `sglx.bitcode.bitstart / sglx.fs` (the bitcode pulse
on the ephys clock) minus the mode of `bp.ev.bitStart` (the same pulse on the behaviour clock),
≈ 0.49 s. Each trial's frame times then become `frameTimes − vidshift − goCue[trial]`, i.e.
seconds from that trial's go cue on the same axis as the spikes, and are interpolated onto the
identical 500-bin grid. If `findVideoOffset` fails or returns a non-finite value, the reference
code's fallback of 0.5 s is used. Correct alignment was verified behaviourally: P(tongue
visible) is ~0.00 through the whole delay and jumps to 0.15–0.27 within 100 ms of t = 0.

ii.
```python
def video_offset(obj):
    """Replicates funcs/findVideoOffset.m: offset (s) between ephys and video clocks."""
    bitStart = as_vector(obj['bp']['ev']['bitStart'])
    bitstart_sglx = as_vector(obj['sglx']['bitcode']['bitstart'])
    fs = as_vector(obj['sglx']['fs'])[0]
    m1 = _mode(bitStart[~np.isnan(bitStart)], keepdims=False).mode
    m2 = _mode(bitstart_sglx[~np.isnan(bitstart_sglx)], keepdims=False).mode
    return float(m2 / fs - m1)

def get_vidshift(obj):
    try:
        vs = video_offset(obj)
        if np.isfinite(vs): return float(vs)
    except Exception:
        pass
    return DEFAULT_VIDSHIFT      # 0.5 s, the reference code's fallback
```

```python
vidshift = get_vidshift(obj)          # once per session
...
tt = ft - vidshift - align_times[i]
arr[i] = np.interp(taxis, tt, v, left=np.nan, right=np.nan)
```

iii. Step 3: "Video: frame times must be corrected by
`vidshift = mode(sglx.bitcode.bitstart)/fs - mode(bp.ev.bitStart)` (~0.49 s) and then aligned
to the same event before interpolating onto the neural time axis." Step 5: "`vidshift =
mode(bitstart)/fs - mode(bp.ev.bitStart)`; video time axis = `frameTimes - vidshift - goCue`;
reference `findVideoOffset.m`, `loadMotionEnergy.m`, `findPosition.m`; fallback: frameTimes =
(1:n)/400 and shift 0.5 s (the reference fallback)." Validation: "Temporal alignment verified
quantitatively: P(tongue visible) goes 0.00 → 0.15-0.27 within 100 ms of t = 0 … The tongue
becomes visible, motion energy increases and ALM firing peaks immediately after t = 0, exactly
as expected for go-cue alignment."

---

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The DeepLabCut tracking in `obj.traj{2}` — the **bottom camera** — using **both** paw
features, `top_paw` *and* `bottom_paw`, which are the paw features in the reference's
`params.traj_features`. The same `frameTimes`, `bitStart`/`bitcode` offset and `goCue` are used
as for the tongue. The side camera is not used (it does not track the paws).

ii.
```python
paw_speeds = []
paw_vis = []
for f in ('top_paw', 'bottom_paw'):
    ix = bot_feats.index(f)
    Xp, Yp = interp_positions(bottom, ix, trials, align_times, taxis, vidshift)
    paw_speeds.append(speed_from_positions(Xp, Yp))
    paw_vis.append(np.isfinite(Xp) & np.isfinite(Yp))
```

iii. Step 5 mapping: "`obj.traj{2}` top_paw/bottom_paw x/y → `output[4]` paw velocity | speed
averaged over the two tracked paws … paws are tracked only on the bottom cam (paper)". From
the trajectory (step 80): "For paw, only bottom view ('top_paw','bottom_paw'). … Use 'top_paw'
(the reference's params list includes top_paw and bottom_paw). I'll use the mean of the two
paws' speeds? Simpler: use 'top_paw' only... but bottom_paw is the other paw. Defining paw
velocity as the mean speed across the two tracked paws is reasonable and uses all data."

---

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Identical to the tongue pipeline — per-trial interpolation of `x`, `y` onto the 500-bin axis
with NaN-mask propagation, then `speed_from_positions` — run separately for each paw, then the
two are combined with `np.nanmean`, so a bin uses both paws where both are tracked and
whichever one is available otherwise. A bin is "visible" if **either** paw is tracked
(`np.any` over the two masks), and a visible bin whose speed is still undefined is set to 0,
as for the tongue. Two documented deviations from `findPosition.m`/`findVelocity.m`, which
treat non-tongue features differently: the AI does **not** nearest-fill missing paw positions
or velocities (that would erase the required `not visible` class), and it does not subtract the
`median(diff(...))` baseline-drift term that `findVelocity.m` applies to non-tongue features.
No normalisation is applied — both paws come from the same camera, so they share a pixel scale.
Result: 5.9% of bins are `not visible`.

ii.
```python
paw_vis = np.any(np.stack(paw_vis), axis=0)
with np.errstate(invalid='ignore'):
    paw_speed = np.nanmean(np.stack(paw_speeds), axis=0)
paw_speed = np.where(paw_vis & ~np.isfinite(paw_speed), 0.0, paw_speed)
```

iii. "speed averaged over the two tracked paws; 2 = not visible; else 0/1 by per-session
median". Key decision 7: "**Paw 'not visible'** = both paw features NaN at that timepoint."
From the trajectory: "'not visible' for paws: DLC fills missing with nearest for non-tongue
features. Paws are usually visible; NaN values occur if DLC didn't track. I'll define paw 'not
visible' = both paws' positions are NaN at that timepoint (before filling) or the trial's video
missing." (Note: the omission of the `basederiv` baseline-drift subtraction is not mentioned
anywhere in the notes.)

---

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The same `discretize()` call: per-session 50th percentile of the combined paw speed over all
visible bins of that session; ≥ threshold → 1, < threshold → 0, not visible → 2. Threshold
stored per session as `paw_thr`. Full-dataset distribution 0.470 / 0.470 / 0.059, exactly
symmetric as a median split must be.

ii.
```python
paw_cls, paw_thr = discretize(paw_speed, paw_vis)
...
OUTPUT_VALUES = [..., ['below_median', 'above_median', 'not_visible'], ...]
```

iii. Same as 7-c — one shared helper and one rationale for all three movement outputs: the
per-session median split is "as specified by the decoder task", with class 2 reserved for
"not visible / no video". Verified by the exactly equal class fractions per session in
`verification_full_out.txt`.

---

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue, but using the bottom camera's own `frameTimes` for the trial
(each `interp_positions` call is given the view whose features it is reading, so a view-to-view
frame-count difference cannot misalign anything). The one session-wide `vidshift` from
`findVideoOffset.m` is subtracted, then the trial's `goCue`, and the frames are interpolated
onto the same 500-bin grid as the spikes.

ii.
```python
Xp, Yp = interp_positions(bottom, ix, trials, align_times, taxis, vidshift)
```
inside which:
```python
t = traj_trials[tr - 1]                # bottom-camera entry for this trial
ft = np.asarray(t.get('frameTimes', np.nan), dtype=float).ravel()
...
tt = ft - vidshift - align_times[i]
arr[i] = np.interp(taxis, tt, v, left=np.nan, right=np.nan)
```

iii. Same justification as 7-d — the offset and grid are shared by every stream, so no
per-feature treatment is needed. Confirmed in the `--show-processing` plots, which overlay
P(paw visible) with P(tongue visible) and P(ME available) on the go-cue-aligned axis.

---

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` file that sits beside each session's data
structure — one 400 Hz trace per trial, one value per side-camera frame. `matio.load_motion_energy`
unwraps the three layouts present in the release (`me` a bare cell array; `me.data` a cell plus
`me.moveThresh`; `me.data.data` nested one deeper). The AI noted that `obj.me` carries the same
data in 28 of 44 sessions and uses the standalone file instead, which exists for all 44. The
side camera's `frameTimes` provide the time base.

ii.
```python
def load_motion_energy(mefile):
    """Handles the three layouts present in the dataset:
      me.data = cell of trials, me.moveThresh = scalar
      me.data = struct with .data (cell) and .moveThresh
      me = cell of trials (no threshold stored in the file)"""
    m = sio.loadmat(mefile, struct_as_record=False, squeeze_me=True)['me']
    thresh = np.nan
    if isinstance(m, np.ndarray) and m.dtype == object:
        data = m
    else:
        data = m.data
        thresh = ... if hasattr(m, 'moveThresh') else np.nan
        if hasattr(data, 'data'):
            ...
            data = data.data
    trials = [np.asarray(x, dtype=float).ravel() for x in np.atleast_1d(data).ravel()]
    return trials, thresh
```

```python
me_trials = None
if sess.get('mefile'):
    try:
        me_trials, _thr = load_motion_energy(sess['mefile'])
    except Exception:
        me_trials = None
ME = motion_energy_on_axis(me_trials, side, trials, align_times, taxis, vidshift)
```

iii. Step 2: "Motion energy files come in three layouts (handled in
`matio.load_motion_energy`): `me.data` = cell of per-trial 400 Hz vectors + `me.moveThresh`;
`me.data.data` nested; or `me` = bare cell array (no threshold)", and "`obj.me` (present in
28/44 ephys sessions) — same motion energy as the separate `motionEnergy_*.mat` file". The AI
had explicitly compared the two copies ("compare obj.me with motionEnergy file", trajectory
step 25).

---

## 9-b. How is `output` *Motion energy* processed?

i. Nothing beyond resampling. The value is already a single scalar per frame (the paper reduces
each frame's per-pixel motion energy to its 99th percentile upstream), so it is only linearly
interpolated from the side camera's frame times onto the shared 500-bin axis, with `left=NaN`
/ `right=NaN` outside the covered range — the same `interp1` step as `loadMotionEnergy.m`,
minus that function's `fillmissing(...,'nearest')`, which is skipped so that genuinely
uncovered bins can carry the `no_video` class. No smoothing, differentiation, or per-session
normalisation. A trial with no ME file, a fewer-than-2-sample trace, or an index past the end
of the ME cell array is left entirely NaN.

ii.
```python
def motion_energy_on_axis(me_trials, traj_trials, trials, align_times, taxis, vidshift):
    """loadMotionEnergy.m: interpolate the 400 Hz motion energy onto taxis."""
    ME = np.full((ntr, taxis.size), np.nan)
    if me_trials is None:
        return ME
    for i, tr in enumerate(trials):
        if tr - 1 >= len(me_trials): continue
        d = np.asarray(me_trials[tr - 1], dtype=float).ravel()
        if d.size < 2: continue
        t = traj_trials[tr - 1]
        ft = np.asarray(t.get('frameTimes', np.nan), dtype=float).ravel()
        if ft.size != d.size or not np.any(np.isfinite(ft)):
            ft = np.arange(1, d.size + 1) / VIDEO_FS
            tt = ft - DEFAULT_VIDSHIFT - align_times[i]
        else:
            tt = ft - vidshift - align_times[i]
        ME[i] = np.interp(taxis, tt, d, left=np.nan, right=np.nan)
    return ME
```

iii. Step 5 mapping: "`motionEnergy_ANM_DATE.mat` (or `obj.me`) → `output[5]` motion energy |
400 Hz ME interpolated onto the neural time axis; 2 = no video/ME for that trial; else 0/1 by
per-session median | `loadMotionEnergy.m`". Step 3 notes the authors' own definition: "Motion
energy | 99th percentile of frame difference; per-session manual movement threshold". Key
decision 8: "**Motion energy 'no video'** = no ME sample for that trial (no ME file, empty
trial entry or interpolation entirely outside the recorded frames)."

---

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The same `discretize()` call as the other two movement streams: the per-session 50th
percentile of the interpolated ME over all bins where a sample exists; ≥ threshold → 1,
< threshold → 0, no sample → 2 (labelled `no_video` here rather than `not_visible`, matching
the spec's wording for this variable). The authors' own manually chosen `moveThresh` is read
out of the file but deliberately discarded in favour of the median, because the decoder spec
mandates a 50th-percentile split. Threshold stored per session as `me_thr`. Full-dataset
distribution 0.480 / 0.481 / 0.039.

ii.
```python
OUTPUT_VALUES = [..., ['below_median', 'above_median', 'no_video']]
...
me_visible = np.isfinite(ME)
...
me_cls, me_thr = discretize(ME, me_visible)
```

iii. Key decision 5 covers all three: per-session 50th percentile of the visible values, "as
specified by the decoder task". The departure from the authors' threshold is called out as
difference #2 in the Step 10 reference comparison: "(The reference instead thresholds motion
energy at a manually chosen per-session `moveThresh`.)" Sanity-checked from raw data in
Step 10: "motion-energy classes of a trial recomputed from the raw 400 Hz ME | 3 sessions |
PASS".

---

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same offset, same grid. Motion energy has one value per **side camera** frame, so
`motion_energy_on_axis` is passed the side-camera trajectory list and uses its `frameTimes`,
corrected by the session `vidshift` and the trial's `goCue`, then interpolated onto the 500-bin
axis. The AI guards the frame-count assumption explicitly: if the ME trace and the side
camera's `frameTimes` have different lengths (or the frame times are all NaN) it falls back to
a synthesised 400 Hz axis with the default 0.5 s shift.

ii.
```python
side = get_traj_view(obj, 1)
...
ME = motion_energy_on_axis(me_trials, side, trials, align_times, taxis, vidshift)
```
inside which:
```python
if ft.size != d.size or not np.any(np.isfinite(ft)):
    ft = np.arange(1, d.size + 1) / VIDEO_FS
    tt = ft - DEFAULT_VIDSHIFT - align_times[i]
else:
    tt = ft - vidshift - align_times[i]
```

iii. Step 3: "Motion energy is stored at 400 Hz per trial and interpolated onto the neural time
axis identically [to the DLC kinematics]." Validated by the post-cue jump: "P(ME above median)
… 0.62 at t = 0 → 0.91 at t = +0.1 s", and in the sample table "motion energy increases …
immediately after t = 0".

---

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Eight cases, handled without dropping trials except where the trial has no neural data at
all:
**(1) Mixed `.mat` versions** — 36 v7.3 + 11 v5 files behind one loader that normalises both to
the same nested Python structure, including restoring MATLAB's array orientation and decoding
`char` arrays.
**(2) Three motion-energy layouts** — unwrapped by explicit `hasattr` checks; a missing or
unreadable ME file is caught by `try/except` and the whole session's ME becomes `no_video`.
**(3) Over-long per-trial fields** — every `bp` field is truncated with `[:N]`.
**(4) NaN in `bp.stim.enable`** — `np.nan_to_num(...)` before the boolean cast, so a NaN is
read as "no stim" rather than raising or being coerced to True.
**(5) Untracked DLC frames** — the likelihood-driven NaNs are preserved through interpolation by
the companion NaN-mask interpolation and become class 2; nothing is nearest-filled (a
deliberate departure from `findPosition.m`'s `fillmissing`).
**(6) Missing or all-NaN `frameTimes`, or a frame/sample count mismatch** — the AI *synthesises*
a time base, `(1:n)/400` with a 0.5 s shift, copying `findPosition.m`'s fallback. I measured
how often this fires: 4 trial-views in the first 12 sessions (~0.03%), no count mismatches.
**(7) Missing `obj.ex`/`obj.meta`** — older sessions (JEB7, JGR2, JGR3) have no `probe.loc`, so
the brain region defaults to ALM, which the paper states for all these recordings.
**(8) Trials with no spikes** — a trial in which no curated unit fired at all (probe stopped
before the session ended) is dropped, since 500 bins of zero rate across every neuron carry no
neural information.
Two guards in the trial filter (finite go cue, exactly one outcome flag) never fire in this
release. One cosmetic artefact remains: `np.nanmean` over a trial where neither paw is tracked
emits `RuntimeWarning: Mean of empty slice` (many lines in `conversion_full_out.txt`); the
resulting NaN is correctly mapped to `not visible`.

ii.
```python
stim = as_vector(bp['stim']['enable'])[:N]
stim = np.nan_to_num(stim).astype(bool)
```

```python
ft = np.asarray(t.get('frameTimes', np.nan), dtype=float).ravel()
if ft.size != ts.shape[0] or not np.any(np.isfinite(ft)):
    # reference fallback: frameTimes = (1:nframes)/400, shift 0.5 s
    ft = (np.arange(1, ts.shape[0] + 1)) / VIDEO_FS
    tt = ft - DEFAULT_VIDSHIFT - align_times[i]
```

```python
if sess.get('mefile'):
    try:
        me_trials, _thr = load_motion_energy(sess['mefile'])
    except Exception:
        me_trials = None
```

```python
    Older sessions (JEB7, JGR2, JGR3) have no `ex`/`meta` field; the paper states that all
    of these recordings were made in ALM, so ALM is used as the default.
```

iii. Step 6: "NaN handling: `np.interp` does not propagate NaN, so a companion interpolation of
the NaN mask is used to mark 'not visible' timepoints." Step 12's issue list gathers the rest:
"61 trials with no spikes at all (recording ended early) → dropped. Tongue/paw velocity NaN at
isolated visible frames → speed set to 0 (as `findVelocity.m` does). MATLAB `gradient` NaN
propagation shrinking the visible tongue period → one-sided differences at edges. Probe 2 of
the JEB15 sessions is tjM1, not ALM → labelled correctly in `brain_regions`. Cluster-quality
labels are capitalised in some sessions → case-insensitive comparison. Three motion-energy
file layouts and two .mat versions → a uniform loader (`matio.py`)." The 400 Hz frame-time
fallback is justified by reference to the reference code: "fallback: frameTimes = (1:n)/400 and
shift 0.5 s (the reference fallback)".

---

## 11-a. What are the most time-consuming steps of the code?

i. Reading the `.mat` files dominates, and the AI measured this rather than assuming it: each
session's `process_session` records a `timing` dict that is stored in `metadata.session_info`.
Per session: load 1.5–4 s, spike binning/smoothing ~1 s, kinematics + motion energy ~2 s,
total 3–8 s. Because sessions are independent, they run in a `multiprocessing.Pool`, so the
whole 44-session conversion finishes in **18 s** wall clock (the AI's own pre-run estimate was
~4 min serial / ~1 min with 8 workers, so the measured time beat the estimate).

ii.
```python
def process_session(sess, show_processing=False, outdir='/app'):
    t0 = time.time(); timing = {}
    obj = load_obj(sess['datafile'])
    timing['load'] = time.time() - t0
    ...
    t1 = time.time()
    rates, quals, spk_per_trial, units_per_probe = bin_spikes(...)
    timing['spikes'] = time.time() - t1
    ...
    timing['kinematics'] = time.time() - t1
    ...
    timing['motion_energy'] = time.time() - t1
```

iii. Step 6: "Loading `clu.spkWavs` (spike waveforms) is unnecessary and dominates file
reading → skipped in `matio`", and "Sessions are independent → processed with a
`multiprocessing.Pool` (8 workers by default)". Step 7's tables record both the speed-ups
("skip `spkWavs` when reading the .mat files | ~2-5x faster loading"; "multiprocessing over
sessions (8 workers) | ~6x") and the per-step timings quoted above.

---

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI vectorised the two loops the reference MATLAB is slowest on: the per-trial `histc`
loop in `getSeq.m` became one `np.add.at` per cluster over all (trial, bin) pairs at once, and
the per-trial/per-unit `mySmooth` call became a single call on a `(n_bins, n_trials·n_units)`
matrix. Loops that remain, and which it did **not** flag in its notes:
 * `my_smooth` still loops over columns calling `np.convolve` one at a time — after the
   reshape that is ~`n_trials × n_units` ≈ 17,000 scalar convolutions per session, and is the
   single largest remaining vectorisation opportunity (`scipy.ndimage.convolve1d` or an FFT
   convolution would do it in one call).
 * `interp_positions` and `motion_energy_on_axis` loop over trials calling `np.interp`; this is
   hard to avoid because each trial has a different number of camera frames.
 * `bin_spikes` loops over clusters.
 * the final assembly loop builds one `(1, 500)` input copy and one `(6, 500)` output array per
   trial.
Since loading dominates and the whole run is 18 s, none of these is a practical bottleneck.

ii. What was vectorised:
```python
np.add.at(counts[:, :, j], (pos[good], b[good]), 1.0)
...
flat = rates.transpose(1, 0, 2).reshape(nb, -1)
sm = my_smooth(flat)
rates = sm.reshape(nb, ntr, -1).transpose(1, 0, 2).astype(np.float32)
```

What remains as a loop:
```python
out = np.empty_like(xf)
for j in range(xf.shape[1]):
    out[:, j] = np.convolve(xf[:, j], k, mode='same')
```
```python
for i, tr in enumerate(trials):
    ...
    arr[i] = np.interp(taxis, tt, v, left=np.nan, right=np.nan)
```

iii. Step 6: "Code inefficiencies identified: Per-cluster python loops over trials (as in the
MATLAB code) would be very slow → replaced by vectorised binning." And: "Code speedups added:
vectorised binning, vectorised smoothing (all trials/units at once), skipping `spkWavs`,
multiprocessing across sessions." `bin_spikes` is documented as vectorising `alignSpikes.m` +
`getSeq.m`: "all spikes of a unit are binned with a single `np.add.at` over (trial, bin)
instead of a per-trial histogram loop." The notes give no analysis of the loops that survive.

---

## 11-c. What processing does the code repeat multiple times?

i. The big repetitions are avoided: each `.mat` file is read exactly once per session, the
video offset is a single `get_vidshift(obj)` call reused by all four camera streams, the bin
grid is built once per session and shared by every stream, and the quality/FR filters are each
applied once. Redundancy that does remain, and which the notes do not mention:
 * `interp_positions` is called three times (side `tongue`, bottom `top_paw`, bottom
   `bottom_paw`), and `motion_energy_on_axis` a fourth time. Each call re-reads the trial's
   `ts`/`frameTimes` and recomputes the same `frameTimes − vidshift − goCue` axis for the same
   trial — so the bottom camera's time axis is built twice and the side camera's twice.
 * the NaN mask is interpolated separately for `x` and `y` although both share it.
 * the discretisation threshold is recomputed inside `plot_processing` (`np.percentile(
   tongue_speed[tongue_vis], 50)`) rather than reusing the returned `tongue_thr`.
All of these are arithmetic on already-loaded arrays and cost little next to the file read.

ii.
```python
vidshift = get_vidshift(obj)        # computed once per session
...
Xt, Yt = interp_positions(side,   tongue_ix, trials, align_times, taxis, vidshift)
for f in ('top_paw', 'bottom_paw'):
    Xp, Yp = interp_positions(bottom, bot_feats.index(f), trials, align_times, taxis, vidshift)
...
ME = motion_energy_on_axis(me_trials, side, trials, align_times, taxis, vidshift)
```
each of which recomputes, per trial:
```python
tt = ft - vidshift - align_times[i]
```

iii. Not discussed as such in CONVERSION_NOTES.md; the closest statements are the Step 6
speed-up list ("Avoid unnecessary file I/O" is satisfied by loading each session once) and the
deliberate single computation of `vidshift` per session in `process_session` rather than per
trial.

---

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Mostly in the reading and the reporting.
 * **Loading**: `matio.load_obj` walks the whole `obj` tree, so it materialises fields the
   conversion never touches — `obj.sglx`'s per-trial index arrays, `clu.tm` (session-clock
   spike times, unused since `trialtm` is what is needed), `clu.channel`, `obj.me`, `bp.ev`
   fields such as `sample`, `delay`, `reward`, `lickL`, `lickR`, and the ~14 DLC features other
   than `tongue`, `top_paw`, `bottom_paw`. The one large field that *is* excluded is
   `clu.spkWavs`, skipped on purpose.
 * **Discarded computations**: `load_motion_energy` parses the authors' `moveThresh` out of
   every ME file and it is thrown away (`_thr`) because the spec mandates a median split;
   `spikes_per_trial` is accumulated over every spike of every quality-passing unit only to
   test `> 0`; `quals` and `n_single_units` (the excellent/great/good count) are computed purely
   for the notes; `bot_feats`/`side_feats` inventories are built in full to index three names.
 * **Redundant metadata**: `metadata['time_axis']` duplicates the `input` of every trial, and
   the per-trial `input` arrays are 13,762 identical copies of the same 500-value vector.
 * **Plotting**: `plot_processing` runs only under `--show-processing`, so it costs nothing in
   the full run.

ii.
```python
me_trials, _thr = load_motion_energy(sess['mefile'])     # _thr never used again
```
```python
np.add.at(spikes_per_trial, pos, 1)                      # only ever compared to 0
```
```python
n_single_units=int(sum(1 for q, k in zip(quals, keep_u)
                       if k and q in ('excellent', 'great', 'good'))),
```
```python
time_axis=tm.astype(np.float32),                         # duplicates `input`
```
```python
SKIP_FIELDS = {'spkWavs', 'spkwavs'}                     # the one field deliberately not read
```

iii. The only item the AI documents is the deliberate one: "Loading `clu.spkWavs` (spike
waveforms) is unnecessary and dominates file reading → skipped in `matio`." The single-unit
count is explicitly for reporting only: "This affects only the reporting of single-unit counts,
not the conversion (all non-garbage units are used, exactly as `params.quality = {'all'}`
prescribes)." The remaining discarded work is not discussed.
