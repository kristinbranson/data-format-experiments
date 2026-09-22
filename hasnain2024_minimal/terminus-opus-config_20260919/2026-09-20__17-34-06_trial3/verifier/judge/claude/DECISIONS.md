# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI restricts itself to a single data folder, `/app/data/Ephys_Behavior` (the fixed-delay DR ephys+video cohort, 25 sessions). It globs `data_structure_*.mat` there, and intersects that glob with a session/probe table that it builds **programmatically** by parsing the authors' own MATLAB meta scripts (`code/DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m`): it reads uncommented `meta(..).anm`, `meta(..).date`, and `meta(..).probe` lines, skipping any line starting with `%`. A file with no meta entry is skipped. The second data folder, `RandomizedDelay_Ephys_Behavior` (19 sessions, 4 mice), is deliberately **not** loaded. Data structures are read with `h5py` (all `Ephys_Behavior` files are MATLAB v7.3 / HDF5); the `motionEnergy_*.mat` companion files are read with `scipy.io.loadmat` (v5). Per-cluster and per-trial fields are HDF5 object references that are dereferenced on demand.

ii.
```python
DATA_DIR = '/app/data/Ephys_Behavior'

def parse_meta_scripts(meta_dir='/app/code/DataLoadingScripts/Recording and video'):
    sessions = {}
    for fn in sorted(glob.glob(os.path.join(meta_dir, '*.m'))):
        anm = None
        cur_date = None
        for line in open(fn).read().split('\n'):
            s = line.strip()
            if s.startswith('%'):
                continue
            m = re.match(r"meta\(.*\)\.anm\s*=\s*'([^']+)'", s)
            if m:
                anm = m.group(1)
            m = re.match(r"meta\(.*\)\.date\s*=\s*'([^']+)'", s)
            if m:
                cur_date = m.group(1)
            m = re.match(r"meta\(.*\)\.probe\s*=\s*\[?([0-9 ,]+)\]?", s)
            if m and anm is not None and cur_date is not None:
                probes = [int(p) for p in m.group(1).replace(',', ' ').split()]
                sessions[(anm, cur_date)] = probes
    return sessions
```

```python
meta = parse_meta_scripts()
files = sorted(glob.glob(os.path.join(DATA_DIR, 'data_structure_*.mat')))
...
    probes = meta.get((anm, date))
    if probes is None:
        print(f'{base}: no meta entry, skipping')
        continue
    res = load_session(dsfile, probes)
```

```python
def load_session(dsfile, probes, verbose=True):
    f = h5py.File(dsfile, 'r')
    bp = f['obj/bp']
```

iii. From the module docstring and the trajectory (steps 32, 39, 45): the AI identified the two folders early and explicitly deliberated whether to merge them. Its stated reasons for keeping only `Ephys_Behavior`: it is "the fixed-delay recordings used for the paper's main neural analyses (25 sessions / 9 mice, 12 of which are two-context DR+WC sessions)", whereas "the randomized-delay dataset ... is a separate cohort analysed separately in the paper (Fig. 8) with a different trial structure, and its sessions contain essentially no water-cued trials, so it is not merged here." It confirmed the low-WC claim empirically by scanning `autowater` fractions across all sessions in both folders (step 39/41). For the session/probe list, the AI reasoned that the authors' `load<ANM>_ALMVideo.m` meta scripts are the authoritative record of which sessions and probes entered the analysis, and chose to parse them rather than hard-code, so that commented-out sessions are automatically excluded.

## 1-b. How are the data split into subjects?

i. The animal ID is the token between `data_structure_` and the date in the filename, extracted with a regex. Subjects are accumulated in first-seen (alphabetical-by-filename) order in a `subjects` list, and each session appends the index of its animal into `subject_idx`. The result is 10 subjects over 25 sessions (EKH1, EKH3, JEB13, JEB14, JEB15, JEB19, JEB6, JEB7, JGR2, JGR3).

ii.
```python
m = re.match(r'data_structure_([A-Za-z0-9]+)_(\d{4}-\d{2}-\d{2})\.mat', base)
anm, date = m.group(1), m.group(2)
...
if anm not in subjects:
    subjects.append(anm)
...
data['subject_idx'].append(subjects.index(anm))
...
data['subjects'] = subjects
data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. Not discussed at length in the trajectory. The filename is the only place the animal ID is reliably stored (the AI discovered in steps 48–51 that the in-file metadata is inconsistent — some sessions have `obj.ex`, others `obj.meta`), and the animal prefix is also how the authors' meta scripts key sessions (`meta(..).anm`).

## 1-c. How are the data split into sessions?

i. One session = one `data_structure_<anm>_<date>.mat` file in `Ephys_Behavior` that has a matching entry in the parsed meta table. Each becomes one element of `neural`, `input`, `output`, `subject_idx`, and `brain_region_idx`. Two extra session-level filters are applied: a session with fewer than 2 usable trials is dropped, and a session with fewer than 10 usable units is dropped (`MIN_UNITS = 10`, the paper's stated inclusion criterion). In practice neither filter removed anything; 25 sessions are written.

ii.
```python
MIN_UNITS = 10   # minimum number of units for a session to be included
...
    trials = np.where(keep)[0]            # 0-based trial indices
    if len(trials) < 2:
        f.close()
        return None
...
    rates = np.concatenate(rates, axis=0)       # (nunits, ntrials, nt)
    if rates.shape[0] < MIN_UNITS:
        if verbose:
            print(f'   session dropped: only {rates.shape[0]} units')
        f.close()
        return None
```

iii. Docstring: "Sessions with fewer than 10 usable units are excluded (paper inclusion criterion)". This is taken verbatim from the methods ("Recording sessions were included for analysis only if they had at least 10 units"). The 2-trial minimum comes from the task instructions ("There needs to be at least two trials within each session"). The restriction to the fixed-delay folder is justified as in 1-a.

## 1-d. How are the data split into trials?

i. Trials are the rows of the Bpod table `obj.bp`. `Ntrials` gives the count and every per-trial flag (`hit`, `miss`, `no`, `early`, `autowater`, `R`, `L`, `stim.enable`) and event (`ev.goCue`, `ev.lickL`, `ev.lickR`, `ev.bitStart`) is read as a flat vector/cell of that length. Trial membership is never reconstructed: each spike carries its own `clu.trial` index (1-based, converted to 0-based) and `clu.trialtm` (time within that trial), and the video arrays `traj{view}.ts` / `traj{view}.frameTimes` and the motion-energy cell array are indexed by trial. A `trial_pos` lookup maps original trial number → row in the kept-trial subset.

ii.
```python
bp = f['obj/bp']
get = lambda k: np.array(bp[k]).ravel()
ntrials = int(get('Ntrials')[0])
hit, miss, no = get('hit'), get('miss'), get('no')
early, autowater = get('early'), get('autowater')
R, L = get('R'), get('L')
stim = np.array(bp['stim/enable']).ravel()
gocue = np.array(bp['ev/goCue']).ravel()
```

```python
trial_pos = -np.ones(ntrials, dtype=np.int64)
trial_pos[trials] = np.arange(len(trials))
...
tt = np.array(f[trialtm_refs[cid]]).ravel()
tr = np.array(f[trial_refs[cid]]).ravel().astype(np.int64) - 1  # 0-based
ok = (tr >= 0) & (tr < ntrials)
tt, tr = tt[ok], tr[ok]
```

iii. Not argued explicitly; the AI spent steps 24–27 and 37–44 dumping the HDF5 tree (`obj/bp`, `obj/clu`, `obj/traj`, `obj/sglx`) and confirming per-trial field lengths and the 1-based `clu.trial` indexing before writing the converter (step 45). Spikes with out-of-range trial indices are masked out defensively (`ok`).

## 1-e. How are trials filtered based on quality controls?

i. A single boolean mask combining four conditions, applied before anything is computed: (1) the trial must be *completed*, i.e. exactly one of `hit`, `miss`, `no` is set; (2) early-lick trials (`early == 1`) are dropped; (3) optogenetic photoinactivation trials (`stim.enable == 1`) are dropped; (4) trials with a NaN go cue are dropped (no alignment point). Ignore/no-response trials are **kept** on purpose, because "ignore" is one of the required outcome categories. This yields 7,426 trials across the 25 sessions. No trial is dropped for running past the end of the ephys recording.

ii.
```python
keep = ((hit == 1) | (miss == 1) | (no == 1)) & (early != 1) & (stim != 1) & ~np.isnan(gocue)
trials = np.where(keep)[0]            # 0-based trial indices
```

iii. Docstring: early-lick removal follows the paper ("excluding early lick and ignore trials, which were omitted from all analyses"); ignore trials are kept "because 'ignore' is one of the outcome categories that the decoder must predict"; photostim trials are excluded "so that the neural data reflect unperturbed activity" (also matching the authors' condition strings, all of which contain `~stim.enable&~early`). The NaN-go-cue guard was added after the AI explicitly checked for NaN go-cue trials during exploration (step 44).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` — the spike-sorted clusters of the probe(s) the authors' meta script lists for that session. Three per-cluster fields are used: `trialtm` (spike time relative to its trial's start), `trial` (1-based trial index of each spike), and `quality` (manual curation label, a char array behind an HDF5 reference). The go cue times `obj.bp.ev.goCue` supply the alignment. Probe brain-region labels are read from `obj.ex.probe.loc` or `obj.meta.probe.loc`.

ii.
```python
clu = f[f['obj/clu'][prb - 1, 0]]
qual = [mstr(f, r).strip().lower() for r in np.array(clu['quality']).ravel()]
cluid = [i for i, q in enumerate(qual) if q not in BAD_QUALITY]
trialtm_refs = np.array(clu['trialtm']).ravel()
trial_refs = np.array(clu['trial']).ravel()
...
tt = np.array(f[trialtm_refs[cid]]).ravel()
tr = np.array(f[trial_refs[cid]]).ravel().astype(np.int64) - 1  # 0-based
ta = tt - gocue[tr]                     # align to go cue
```

```python
def get_probe_locs(f):
    """probe locations, stored in obj.ex.probe.loc (newer objs) or obj.meta.probe.loc"""
    for key in ['obj/ex/probe/loc', 'obj/meta/probe/loc']:
        if key in f:
            ...
```

iii. Trajectory steps 5–8, 25–28: the AI read `alignSpikes.m`, `getSeq.m`, `findClusters.m`, `removeLowFRClusters.m` and `WorkingWithDataObjs.m`, and confirmed that `clu.trialtm` is already on the Bpod clock relative to trial start, so a single subtraction of `goCue` aligns it. Probe-location reading was added reactively in steps 48–53 after two sessions crashed on a missing `obj/ex`; the AI decided to label regions from the data object rather than assume ALM, which let it keep `JEB15_2022-07-29` (a session whose only meta-listed probe is in tjM1).

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed into 500 non-overlapping 10 ms bins spanning −2.5 to +2.5 s from the go cue (via `np.floor((t - TMIN)/DT)` + `np.bincount` over a flattened trial×bin index), divided by the bin width to give Hz, then smoothed along time with the authors' **causal** Gaussian kernel: `gausswin(15)` with the first `floor(15/2) = 7` coefficients zeroed and the rest normalised to sum to 1, applied with MATLAB `mySmooth`'s `'reflect'` boundary handling (prepend the first 15 samples, convolve `'same'`, then trim). No normalisation, baseline subtraction or z-scoring. Units from all meta-listed probes of a session are concatenated into one population. Stored as `float32`, shape `(n_units, 500)` per trial.

ii.
```python
def gausswin(N, alpha=2.5):
    n = np.arange(N) - (N - 1) / 2.0
    return np.exp(-0.5 * (alpha * n / ((N - 1) / 2.0)) ** 2)


def causal_kernel(N=SMOOTH):
    """mySmooth.m: gaussian window, first floor(N/2) coefficients zeroed (causal)"""
    k = gausswin(N)
    k[:N // 2] = 0
    return k / k.sum()


def smooth_causal(x):
    """smooth along last axis, replicating mySmooth(x,15,'reflect')"""
    N = len(KERN)
    xp = np.concatenate([x[..., :N], x], axis=-1)
    ...
        o[i] = np.convolve(flat[i], KERN, mode='same')
    return out[..., N:]
```

```python
b = np.floor((ta - TMIN) / DT).astype(np.int64)
good = (pos >= 0) & (b >= 0) & (b < NT)
idx = pos[good] * NT + b[good]
counts = np.bincount(idx, minlength=len(trials) * NT).reshape(len(trials), NT)
sess_rates[ui] = counts / DT
sess_rates = smooth_causal(sess_rates)
```

iii. Docstring: "Binned from -2.5 s to +2.5 s in 10 ms bins (params.tmin/tmax/dt), converted to firing rate and smoothed with the authors' causal Gaussian kernel (mySmooth.m, window 15 bins, 'reflect' boundary handling)". Trajectory step 8/28: the AI read `mySmooth.m` line by line and reproduced its causal zeroing, normalisation, padding and trimming exactly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Three filters. (1) Cluster quality: the free-text `clu.quality` label is stripped and lower-cased and the cluster is dropped if it is one of `garbage`, `gabrga`, `noisy`, `real?` — exactly the drop list in `findClusters.m` under `params.quality = {'all'}`. Every other label (`good`, `fair`, `multi`, `poor`, unlabeled) is kept, including multi-units. (2) Firing rate: after binning and smoothing, any unit whose mean rate over the whole window and all kept trials is ≤ 1 Hz is dropped. (3) Session level: a session left with < 10 units is dropped entirely. The result is 1,531 units across 25 sessions (27–141 per session), of which 1,386 are labelled ALM and 145 tjM1.

ii.
```python
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR = 1.0     # Hz, minimum mean firing rate of a unit
MIN_UNITS = 10
...
qual = [mstr(f, r).strip().lower() for r in np.array(clu['quality']).ravel()]
cluid = [i for i, q in enumerate(qual) if q not in BAD_QUALITY]
...
# remove low firing rate units
mfr = sess_rates.mean(axis=(1, 2))
use = mfr > LOW_FR
sess_rates = sess_rates[use]
```

iii. Docstring: "Clusters whose quality label is garbage/noisy/real? are excluded (findClusters.m with params.quality = 'all')" and "Units with mean firing rate <= 1 Hz are excluded (paper: 'All units with firing rates exceeding 1 Hz were included in all other analyses'; removeLowFRClusters.m)". The AI printed the actual quality-label distribution during exploration (step 27/43) before choosing the drop list, and lower-cased the labels because they are written inconsistently.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. By subtracting the trial's go-cue time from each spike's within-trial time, with no interpolation or further offset, because `clu.trialtm` and `bp.ev.goCue` are already on the same Bpod clock and both relative to trial start. On water-cued (autowater) trials `ev.goCue` marks the water presentation, which the AI verified is populated and uses as the alignment point there too. The camera streams need an extra clock correction (see 7-d); the spikes do not.

ii.
```python
gocue = np.array(bp['ev/goCue']).ravel()
...
ta = tt - gocue[tr]                     # align to go cue
```

iii. Docstring: "Spikes are aligned to the go cue (which, on water-cued trials, is the time of water presentation) -- params.alignEvent = 'goCue' in the paper's scripts." Trajectory step 43: the AI explicitly checked "goCue on autowater trials" before committing to this alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **10 ms bins**, 500 bins spanning −2.5 s to +2.5 s around the go cue, identical for every trial and session and shared by the neural data, the input, and all three video-derived outputs. The grid is built once at module scope. There is no rebinning: spikes are histogrammed directly onto this grid, and the video streams (400 Hz frames) are linearly interpolated onto the same bin centres, so a single time axis is used end to end. `metadata['time_bin_size'] = 10.0` ms.

ii.
```python
TMIN = -2.5      # s relative to go cue
TMAX = 2.5
DT = 0.01        # 10 ms bins
SMOOTH = 15      # causal gaussian kernel window (bins)
...
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TAXIS = EDGES[:-1] + DT / 2
NT = len(TAXIS)
```

iii. Docstring cites "params.tmin/tmax/dt". `getDefaultParams.m` sets `params.dt = 1/200` (5 ms), but the AI noted in step 28 that the paper's analysis scripts use `dt = 1/100` (10 ms) — which is true of essentially every script under `Scripts/` (Figure 3, Figure 8, EDFigure 2/3) and of `WorkingWithDataObjs.m`. It chose 10 ms on that basis, which also halves the size of the output pickle.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Nothing from the raw files; it is defined by the conversion. It is the vector of bin centres of the shared −2.5 to +2.5 s, 10 ms grid, i.e. the same grid the spikes are counted into. `obj.bp.ev.goCue` is implicitly the origin.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TAXIS = EDGES[:-1] + DT / 2
NT = len(TAXIS)
...
time_input = TAXIS.astype(np.float32).reshape(1, NT)
```

iii. The decoder task specifies "Time from go cue onset in seconds (continuous, time-varying)" as the only decoder input; the AI supplies exactly that, with the window taken from `params.tmin/tmax`.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None. The bin-centre vector is cast to `float32`, reshaped to `(1, 500)` so `d_input = 1`, and one copy is attached to every trial of every session. Range is [−2.495, 2.495].

ii.
```python
time_input = TAXIS.astype(np.float32).reshape(1, NT)
...
input_sess = [time_input.copy() for _ in range(ntr)]
```

iii. N/A — nothing to justify beyond the format requirement of shape `(d_input, n_timepoints)`.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural time axis. Spike times are expressed relative to their trial's go cue and floored into bins of `EDGES`; the input is the centre of those same bins, so bin *k* denotes the same interval in both streams. All trials are the same length (500), so no padding or resampling is needed.

ii.
```python
ta = tt - gocue[tr]                     # align to go cue
b = np.floor((ta - TMIN) / DT).astype(np.int64)
```
```python
TAXIS = EDGES[:-1] + DT / 2
```

iii. N/A.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The **actual lick-port contact times**, `obj.bp.ev.lickL` and `obj.bp.ev.lickR` (a cell array of contact times per trial, stored as HDF5 references), together with `obj.bp.ev.goCue`. The instructed side (`bp.R`/`bp.L`) and the outcome flags are *not* used for this output (they are read into `R, L` but left unused).

ii.
```python
# lick direction: first lick port contact after the go cue
lickL = [np.array(f[r]).ravel() if np.array(f[r]).size else np.array([])
         for r in np.array(bp['ev/lickL']).ravel()]
lickR = [np.array(f[r]).ravel() if np.array(f[r]).size else np.array([])
         for r in np.array(bp['ev/lickR']).ravel()]
```

iii. Docstring: "lick_direction: side of the first lick-port contact after the go cue (left/right/none)." The AI read `funcs/firstLickTime.m` and `getOutcome.m` during exploration (step 33) and chose the directly measured lick events over an inference from outcome flags.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. For each kept trial, the earliest left-port contact and the earliest right-port contact strictly after the go cue are found; whichever is earlier gives the class (ties go to left). If there is no contact on either port after the go cue, the trial is class 2, "none". Codes: left 0, right 1, none 2. The per-trial value is then tiled across all 500 bins.

ii.
```python
lick_dir = np.full(len(trials), 2, dtype=np.int64)   # 2 = none
for i, t in enumerate(trials):
    gc = gocue[t]
    lt = lickL[t][lickL[t] > gc] if lickL[t].size else np.array([])
    rt = lickR[t][lickR[t] > gc] if lickR[t].size else np.array([])
    fl = lt.min() if lt.size else np.inf
    fr = rt.min() if rt.size else np.inf
    if np.isinf(fl) and np.isinf(fr):
        lick_dir[i] = 2
    elif fl <= fr:
        lick_dir[i] = 0          # left
    else:
        lick_dir[i] = 1          # right
```
```python
out[0] = res['lick_dir'][i]
```

iii. Same as 4-a: the AI treats the first post-go-cue port contact as the animal's choice, which is the definition the authors' `firstLickTime.m` uses. The third class exists because the decoder spec lists "none" as a lick direction.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. A single per-trial flag, `obj.bp.autowater`, which marks the water-cued block.

ii.
```python
early, autowater = get('early'), get('autowater')
...
context = (autowater[trials] == 1).astype(np.int64)   # 0 = DR, 1 = WC
```

iii. Docstring: "context: DR vs WC block, from obj.bp.autowater (the authors use autowater as the proxy for water-cued blocks, see WorkingWithDataObjs.m)." The AI confirmed this by reading the authors' condition strings, which separate `autowater` / `~autowater`, and by scanning the `autowater` fraction across all sessions (step 39) to identify the two-context sessions.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling of the boolean: `autowater == 1` → 1 (WC), otherwise → 0 (DR). The declared category order is `['DR', 'WC']`, i.e. the opposite index convention from the prompt's "(WC, DR)" listing, but `output_values` documents it consistently. The per-trial value is tiled across all 500 bins. Overall fractions: DR 0.833, WC 0.167.

ii.
```python
context = (autowater[trials] == 1).astype(np.int64)   # 0 = DR, 1 = WC
...
'output_values': [...,
                  ['DR', 'WC'],
                  ...]
...
out[1] = res['context'][i]
```

iii. Not separately argued; the AI simply reports the flag as a two-class variable and records the mapping in `output_values`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Three mutually exclusive per-trial flags of `obj.bp`: `hit`, `miss`, and `no`. Unlike the lick direction, the outcome is taken from the Bpod flags rather than re-derived.

ii.
```python
hit, miss, no = get('hit'), get('miss'), get('no')
```

iii. Docstring: "outcome: incorrect (miss) / correct (hit) / ignore (no)". The AI read `funcs/getOutcome.m` (step 33) to confirm the flag semantics.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A relabelling into three classes on the kept trials: the array is initialised to 0 (incorrect), `hit` trials are set to 1 (correct) and `no` trials to 2 (ignore); since trial selection already required exactly one of `hit|miss|no`, the remaining zeros are exactly the `miss` trials. Codes follow the prompt's order (incorrect 0, correct 1, ignore 2). Tiled across all 500 bins. Fractions: 0.134 / 0.694 / 0.172.

ii.
```python
outcome = np.full(len(trials), 0, dtype=np.int64)     # 0 incorrect
outcome[hit[trials] == 1] = 1                         # 1 correct
outcome[no[trials] == 1] = 2                          # 2 ignore
...
out[2] = res['outcome'][i]
```

iii. Class codes follow the decoder spec's listed order. Ignore trials are retained (rather than dropped as in the paper's analyses) precisely so that the third class exists.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj{1}` — the **side camera only** — specifically the feature named `tongue`. Per trial it uses `traj.ts` (x, y, likelihood per feature per frame; here only the x and y planes) and `traj.frameTimes`. The go cue `bp.ev.goCue` and the bitcode fields `obj.sglx.bitcode.bitstart` / `obj.sglx.fs` / `bp.ev.bitStart` are also required, to put the video frames on the behaviour clock. The bottom camera's `top_tongue` / `bottom_tongue` are **not** used.

ii.
```python
SIDE tongue is view 0, feature 'tongue':
for i, t in enumerate(trials):
    tongue[i] = feat_speed(0, ['tongue'], t)
```
```python
g, names = views[view]
idxs = [names.index(n) for n in featnames if n in names]
ft = np.array(f[np.array(g['frameTimes']).ravel()[t]]).ravel().astype(float)
ts = np.array(f[np.array(g['ts']).ravel()[t]])       # (feat, 3, frames)
```

iii. Docstring: "speed of the DeepLabCut-tracked tongue (side camera) ... (findVideoOffset.m, findPosition.m/findVelocity.m)". `params.traj_features` in `getDefaultParams.m` lists `tongue` as the cam-0 tongue feature, and the AI inspected `featNames` and the NaN structure of `ts` directly (steps 37–38, 42) before choosing it.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps, mirroring the authors' `findPosition.m` / `findVelocity.m`. **(1)** Frame times are converted to seconds from the go cue (see 7-d). **(2)** The x and y traces are **linearly interpolated onto the 500-bin neural time axis**, with NaN outside the video window and NaN wherever either bracketing frame is NaN. No explicit likelihood threshold is applied — the AI relies on the fact that the authors already set x and y to NaN wherever the DLC likelihood is ≤ 0.9, so NaN *is* the "not visible" marker (verified: for the side-camera tongue the NaN mask and the `likelihood <= 0.9` mask coincide exactly). **(3)** Speed is `sqrt(vx² + vy²)` with `vx = np.gradient(x)`, `vy = np.gradient(y)` taken on the uniform grid, i.e. in pixels per bin rather than pixels per second — a constant scaling, irrelevant to a percentile split. No smoothing of the tongue trace (matching `findPosition.m`, which explicitly skips smoothing for tongue features), and no cross-camera normalisation since only one view is used. **(4)** Because `np.gradient` propagates NaN to neighbouring samples, the untracked region is widened by one bin at each edge. Resulting distribution: 2.5 % / 2.5 % / 95.1 % not visible.

ii.
```python
tt = ft - vidshift - gocue[t]
sp = []
for fi in idxs:
    x = interp_nan(TAXIS, tt, ts[fi, 0, :].astype(float))
    y = interp_nan(TAXIS, tt, ts[fi, 1, :].astype(float))
    vx = np.gradient(x)
    vy = np.gradient(y)
    sp.append(np.sqrt(vx ** 2 + vy ** 2))
sp = np.array(sp)
with np.errstate(invalid='ignore'):
    out = np.nanmean(sp, axis=0) if sp.shape[0] > 1 else sp[0]
out[np.all(np.isnan(sp), axis=0)] = np.nan
```
```python
def interp_nan(xnew, x, y):
    """linear interpolation, NaN outside of range (like MATLAB interp1)"""
    out = np.interp(xnew, x, y, left=np.nan, right=np.nan)
    idx = np.searchsorted(x, xnew) - 1
    idx = np.clip(idx, 0, len(x) - 2)
    inrange = (xnew >= x[0]) & (xnew <= x[-1])
    bad = np.isnan(y[idx]) | np.isnan(y[idx + 1])
    out[inrange & bad] = np.nan
    return out
```

iii. Docstring: the velocities are "computed on the neural time axis after interpolating the trajectories with the authors' video/ephys offset (findVideoOffset.m, findPosition.m/findVelocity.m)". `findPosition.m` does exactly `interp1(frameTimes - vidshift - alignEv, ts, taxis)` and `findVelocity.m` then applies `gradient` on that interpolated grid — the AI reproduces both. In step 59 it noted "tongue not-visible is 95% which is expected (tongue only out during licks)" and judged that acceptable per the spec.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per session, the 50th percentile of all **finite** tongue-speed values pooled across every kept trial and bin is used as the threshold. Values `< threshold` → 0 (`below_median`), `>= threshold` → 1 (`above_median`), NaN (not tracked) → 2 (`not_visible`). If a session has no finite values at all, everything is class 2.

ii.
```python
def discretize(x):
    """per-session median split; NaN (not visible / no video) -> 2"""
    out = np.full(x.shape, 2, dtype=np.int64)
    finite = np.isfinite(x)
    if np.any(finite):
        thresh = np.percentile(x[finite], 50)
        out[finite & (x < thresh)] = 0
        out[finite & (x >= thresh)] = 1
    return out
...
tongue = discretize(res['tongue'])
```

iii. Docstring: "Discretized per session at the 50th percentile of the finite values (0 = below, 1 = at/above); timepoints where the feature is not tracked (tongue not out of the mouth, paw not visible) are class 2." This is the rule stated in the Decoder Task section of the instructions.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The video runs on the SpikeGLX clock, which leads the Bpod clock; the offset is computed once per session as `mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)`, exactly reproducing `findVideoOffset.m` (NaNs excluded from the mode). Each trial's frame times become `frameTimes − vidshift − goCue[trial]`, and the x/y traces are interpolated onto the identical 500-bin grid used for the spikes, so the two streams share one time axis bin for bin.

ii.
```python
def video_offset(f):
    """findVideoOffset.m : offset between neural file start and video file start"""
    def mode(x):
        x = x[~np.isnan(x)]
        vals, cnt = np.unique(x, return_counts=True)
        return vals[np.argmax(cnt)]
    fs = np.array(f['obj/sglx/fs']).ravel()[0]
    bitstart_sglx = np.array(f['obj/sglx/bitcode/bitstart']).ravel().astype(float)
    bitstart_bp = np.array(f['obj/bp/ev/bitStart']).ravel().astype(float)
    return mode(bitstart_sglx) / fs - mode(bitstart_bp)
```
```python
tt = ft - vidshift - gocue[t]
x = interp_nan(TAXIS, tt, ts[fi, 0, :].astype(float))
```

iii. Docstring cites `findVideoOffset.m`; the AI read that file in step 27 and reproduced it line for line, including the use of the mode rather than the mean.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The same `obj.traj` DLC tracking, but the **bottom camera** (`obj.traj{2}`) and **both** paw features, `top_paw` and `bottom_paw`, plus that camera's own `frameTimes`. If a session has only one camera, the paw output is all-NaN.

ii.
```python
PAW features, view 1:
for i, t in enumerate(trials):
    tongue[i] = feat_speed(0, ['tongue'], t)
    if nviews > 1:
        paw[i] = feat_speed(1, ['top_paw', 'bottom_paw'], t)
```

iii. Docstring: "paws (bottom camera, mean of top_paw and bottom_paw)". Trajectory step 59: "I used both top_paw and bottom_paw from bottom cam; the paper's default traj_features for view 2 includes top_paw/bottom_paw in getDefaultParams. Fine." `params.traj_features{2}` does list both.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Identical machinery to the tongue (`feat_speed`), applied to two features and then averaged: x and y of each paw are interpolated onto the 500-bin grid, differentiated with `np.gradient`, combined into a speed, and the two paws are combined with `np.nanmean`, so a bin is NaN only if **both** paws are untracked there. No likelihood threshold is applied explicitly (again relying on the authors' NaNs), no smoothing, no baseline-derivative subtraction, no normalisation. Resulting distribution: 47.3 % / 47.3 % / 5.3 % not visible.

ii.
```python
sp = []
for fi in idxs:
    x = interp_nan(TAXIS, tt, ts[fi, 0, :].astype(float))
    y = interp_nan(TAXIS, tt, ts[fi, 1, :].astype(float))
    vx = np.gradient(x)
    vy = np.gradient(y)
    sp.append(np.sqrt(vx ** 2 + vy ** 2))
sp = np.array(sp)
with np.errstate(invalid='ignore'):
    out = np.nanmean(sp, axis=0) if sp.shape[0] > 1 else sp[0]
out[np.all(np.isnan(sp), axis=0)] = np.nan
```

iii. As in 8-a: both paws are the authors' default cam-1 paw features, and averaging their speeds gives a single "paw velocity" summary. The AI reviewed the resulting not-visible fractions in step 59 and judged them reasonable ("paw mostly visible").

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Exactly as the tongue: the same `discretize` helper, a per-session 50th percentile of the finite paw speeds pooled over all kept trials and bins, `< threshold` → 0, `>= threshold` → 1, NaN → 2 (`not_visible`).

ii.
```python
paw = discretize(res['paw'])
```
```python
thresh = np.percentile(x[finite], 50)
out[finite & (x < thresh)] = 0
out[finite & (x >= thresh)] = 1
```

iii. Same as 7-c — the rule is specified by the instructions.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue, but using the bottom camera's own `frameTimes`: `frameTimes − vidshift − goCue[trial]`, then interpolation onto the shared 500-bin grid. The session video offset is the same constant for both cameras, but it is recomputed from the file inside `video_speeds` rather than cached. Because each view's own `frameTimes` array is used, differing frame counts between the two cameras are handled correctly.

ii.
```python
g, names = views[view]
ft = np.array(f[np.array(g['frameTimes']).ravel()[t]]).ravel().astype(float)
...
tt = ft - vidshift - gocue[t]
```

iii. Same offset and same grid as every other stream; no separate treatment is argued.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` file beside each data structure, loaded with `scipy.io.loadmat`. Inside it, `me.data` is a cell array with one trace per trial at video frame rate; three sessions wrap it one level deeper as `me.data.data`, which is unwrapped. The side camera's `frameTimes` (view 0) plus the session video offset and go cue supply the time base. `obj.me`, where present, is not used.

ii.
```python
mefile = dsfile.replace('data_structure', 'motionEnergy')
if not os.path.exists(mefile):
    return np.full((len(trials), NT), np.nan)
me = sio.loadmat(mefile)['me']
data = me['data'][0, 0]
# some sessions store me.data as a struct with its own .data field (loadMotionEnergy.m)
if data.dtype.names is not None and 'data' in data.dtype.names:
    data = data['data'][0, 0]
if data.dtype != object:
    return np.full((len(trials), NT), np.nan)
```

iii. Docstring: "motion_energy: frame-to-frame motion energy from the motionEnergy_*.mat files interpolated onto the same axis (loadMotionEnergy.m)". The nested-struct unwrap was added reactively in steps 56–58 after the AI found two JEB15 sessions coming out with no motion energy, and it matches the `if isstruct(me.data), me.data = me.data.data; end` guard in `loadMotionEnergy.m`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond resampling: the trace is already one scalar per frame. It is linearly interpolated onto the 500-bin grid with `interp_nan`, and then remaining NaNs are filled with the **nearest** finite value along time, reproducing `fillmissing(me.data,'nearest')` at the end of `loadMotionEnergy.m`. Consequently almost no bins end up in the "no video" class (fraction 0.0001, a single trial). Trials whose motion-energy length disagrees with the frame-time length, or whose frame times are all NaN, or which lie beyond the end of the motion-energy cell array, are skipped and left NaN.

ii.
```python
d = np.array(data[t, 0]).ravel().astype(float)
ft = np.array(f[ftrefs[t]]).ravel().astype(float)
if d.size < 2 or ft.size != d.size or np.all(np.isnan(ft)):
    continue
tt = ft - vidshift - gocue[t]
out[i] = interp_nan(TAXIS, tt, d)
# fill edge NaNs (time points outside of the video) with nearest value,
# as in loadMotionEnergy.m
v = out[i]
if np.any(np.isnan(v)) and np.any(~np.isnan(v)):
    good = np.where(~np.isnan(v))[0]
    idx = np.clip(np.searchsorted(good, np.arange(NT)), 0, len(good) - 1)
    prev = np.maximum(idx - 1, 0)
    choose = np.where(np.abs(good[idx] - np.arange(NT)) <= np.abs(good[prev] - np.arange(NT)),
                      good[idx], good[prev])
    out[i] = np.where(np.isnan(v), v[choose], v)
```

iii. Docstring cites `loadMotionEnergy.m`; the nearest-fill is annotated in the code as "as in loadMotionEnergy.m", which is literally what that function does after its `interp1`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The same `discretize` helper: per-session 50th percentile over all finite values, `< threshold` → 0, `>= threshold` → 1, NaN → 2 (labelled `no_video` for this output). Fractions: 0.4996 / 0.5003 / 0.0001.

ii.
```python
me = discretize(res['me'])
```
```python
'output_values': [...,
                  ['below_median', 'above_median', 'no_video']],
```

iii. The instructions specify the per-session 50th-percentile split and the "no video" third class. In step 61–63 the AI noticed the `no_video` class has only ~500 timepoints (one trial) and explicitly checked `accuracy_all_sessions` in `decoder.py` to confirm that `balanced_accuracy_score` only scores classes present in `y_true`, so keeping a near-empty class does not distort the reported metric.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy has one value per side-camera frame, so it uses view 0's `frameTimes`, corrected by the same session video offset and the trial's go cue, then interpolated onto the shared 500-bin grid. The offset is recomputed from the file (a second call to `video_offset`).

ii.
```python
vidshift = video_offset(f)
g = f[f['obj/traj'][0, 0]]
ftrefs = np.array(g['frameTimes']).ravel()
...
tt = ft - vidshift - gocue[t]
out[i] = interp_nan(TAXIS, tt, d)
```

iii. `loadMotionEnergy.m` uses `obj.traj{1}(trix).frameTimes - vidshift - alignTimes(trix)`, which the AI reproduces exactly, including using view 0.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Every anomaly the AI met is handled by degrading gracefully rather than by crashing or by inventing neural data:

* **Trials with a NaN go cue** — excluded from the trial mask (no alignment point exists).
* **Spikes with out-of-range trial indices** — masked out by `ok = (tr >= 0) & (tr < ntrials)`.
* **Missing `obj.ex.probe.loc`** — falls back to `obj.meta.probe.loc`; if neither exists, or the location string is blank, the probe is labelled ALM on the grounds that the authors' meta scripts only list ALM (and one tjM1) probes.
* **Probe location stored as a char array instead of a cell of references** (single-probe sessions) — decoded directly rather than dereferenced.
* **Missing / all-NaN `frameTimes`, or fewer than 2 frames** — that trial's tongue/paw row is left all-NaN → class 2.
* **Untracked frames** — x/y are already NaN there (the authors' likelihood ≤ 0.9 rule), interpolation propagates the NaN, and those bins become class 2.
* **Missing `motionEnergy_*.mat`, unexpected `me.data` type, trial index beyond the motion-energy array, frame-count/ME-length mismatch** — that row is left NaN → class 2; the `me.data.data` nesting is unwrapped.
* **Edge NaNs in motion energy** (bins outside the video window) — filled with the nearest finite value, following `loadMotionEnergy.m`.
* **Sessions with < 2 usable trials or < 10 usable units** — dropped entirely.

No neural sample is ever imputed, and no trial is dropped merely because its video is missing.

ii.
```python
keep = ((hit == 1) | (miss == 1) | (no == 1)) & (early != 1) & (stim != 1) & ~np.isnan(gocue)
```
```python
def get_probe_locs(f):
    for key in ['obj/ex/probe/loc', 'obj/meta/probe/loc']:
        if key in f:
            dset = f[key]
            if dset.dtype == h5py.special_dtype(ref=h5py.Reference):
                return [mstr(f, r) for r in np.array(dset).ravel()]
            # single probe sessions store the location string directly
            return [''.join(chr(int(c)) for c in np.array(dset).ravel())]
    return None
```
```python
if ft.size < 2 or np.all(np.isnan(ft)):
    return np.full(NT, np.nan)
```
```python
if d.size < 2 or ft.size != d.size or np.all(np.isnan(ft)):
    continue
```

iii. Almost all of these guards were added reactively, after a run crashed or produced an empty stream, and each is annotated with the authors' function that handles the same case (`loadMotionEnergy.m`, `findPosition.m`). The AI's stated principle (steps 49, 53, 57) is to keep the session and mark the gap rather than lose data: when `JEB15_2022-07-29` was initially dropped because its only meta-listed probe is tjM1, the AI chose to keep the session and label the region correctly instead.

## 11-a. What are the most time-consuming steps of the code?

i. The whole conversion takes ≈ 57 s for 25 sessions (≈ 2.3 s/session). The dominant costs are, in order: (1) reading the HDF5 data structures — in particular dereferencing `clu.trialtm` / `clu.trial` for every kept cluster and `traj.ts` / `traj.frameTimes` for every trial and camera, which is hundreds to thousands of small HDF5 reads per session; (2) `smooth_causal`, which loops in Python over every (unit × trial) row and calls `np.convolve` — for a 141-unit, 380-trial session that is ~54,000 individual convolutions; (3) `video_speeds`, a Python loop over trials × features performing three interpolations and two gradients each; (4) pickling the 1.1 GB result.

ii.
```python
    flat = xp.reshape(-1, xp.shape[-1])
    o = out.reshape(-1, xp.shape[-1])
    for i in range(flat.shape[0]):
        o[i] = np.convolve(flat[i], KERN, mode='same')
```
```python
    for i, t in enumerate(trials):
        tongue[i] = feat_speed(0, ['tongue'], t)
        if nviews > 1:
            paw[i] = feat_speed(1, ['top_paw', 'bottom_paw'], t)
```

iii. Not analysed in the trajectory. The AI timed its runs (`time python3 convert_data.py`, ~57 s) and evidently judged the runtime acceptable, never profiling or optimising further.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Three clear candidates. (1) `smooth_causal`'s per-row `np.convolve` loop is a one-liner with `scipy.ndimage.convolve1d(x, KERN, axis=-1)` or `scipy.signal.fftconvolve`, which would operate on the whole `(units, trials, time)` array at once; this is the single biggest avoidable cost. (2) The per-cluster loop that bins spikes could be collapsed into one `np.bincount`/`np.histogram2d` over all clusters at once by adding a cluster offset to the flattened index (all clusters' spike arrays would first have to be concatenated). (3) The per-trial loops in `video_speeds` and `motion_energy` are harder to vectorise because each trial has a different frame count, but the *interpolation* could at least be batched per feature. The lick-direction loop over trials is trivial in cost. Everything downstream of `load_session` (the `discretize` calls, the per-trial slicing) is already array-wide.

ii.
```python
    for i in range(flat.shape[0]):
        o[i] = np.convolve(flat[i], KERN, mode='same')
```
```python
        for ui, cid in enumerate(cluid):
            tt = np.array(f[trialtm_refs[cid]]).ravel()
            ...
            counts = np.bincount(idx, minlength=len(trials) * NT).reshape(len(trials), NT)
```

iii. Not discussed. No vectorisation trade-off is argued anywhere in the code or trajectory.

## 11-c. What processing does the code repeat multiple times?

i. Several things are recomputed. (1) `video_offset(f)` — a session constant — is computed twice per session, once in `video_speeds` and again in `motion_energy`, each time re-reading `sglx.bitcode.bitstart` and `bp.ev.bitStart` and taking two modes. (2) The side camera's `frameTimes` for each trial are read once inside `feat_speed` and a second time inside `motion_energy`. (3) `trial_pos` is rebuilt inside the per-probe loop rather than once per session. (4) In the lick-time list comprehensions `np.array(f[r])` is constructed twice per trial (once for the `.size` test, once for `.ravel()`). (5) `interp_nan` performs the interpolation with `np.interp` and then repeats the bracketing search with `np.searchsorted` to redo a NaN check that `np.interp` has already effectively performed. (6) The whole conversion was re-run four times end to end during development, but that is process, not code. Per-session probe-location decoding and the meta-script parse are each done once, as they should be.

ii.
```python
def video_speeds(f, trials, gocue):
    vidshift = video_offset(f)
```
```python
def motion_energy(dsfile, f, trials, gocue):
    ...
    vidshift = video_offset(f)
```
```python
    for prb in probes:
        ...
        trial_pos = -np.ones(ntrials, dtype=np.int64)
        trial_pos[trials] = np.arange(len(trials))
```

iii. Not discussed. None of the duplication is deliberate; it follows from `video_speeds` and `motion_energy` being written as independent free functions that each take the open file handle.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Four items. (1) `R` and `L` (the instructed lick side) are read for every session and then **never used**, because lick direction is derived from actual lick times instead. (2) Firing rates are binned *and causally smoothed* for every quality-passing cluster before the ≤ 1 Hz units are discarded, so the smoothing work on the dropped units is wasted (a small effect here — only 3–5 % of clusters are dropped). (3) The full `lickL` / `lickR` cell arrays are dereferenced for **all** `Ntrials` trials, including early-lick and photostim trials that were already excluded. (4) The identical 500-element time axis is `.copy()`-ed once per trial, producing 7,426 duplicate 2 KB arrays (≈ 15 MB) that all hold the same values — the format arguably requires a per-trial entry, but the copies are avoidable. Nothing else computed is thrown away: all six outputs, the rates, the regions and the session metadata reach the pickle.

ii.
```python
R, L = get('R'), get('L')          # never used again
```
```python
        sess_rates = smooth_causal(sess_rates)
        # remove low firing rate units
        mfr = sess_rates.mean(axis=(1, 2))
        use = mfr > LOW_FR
        sess_rates = sess_rates[use]
```
```python
    lickL = [np.array(f[r]).ravel() if np.array(f[r]).size else np.array([])
             for r in np.array(bp['ev/lickL']).ravel()]
```
```python
    input_sess = [time_input.copy() for _ in range(ntr)]
```

iii. Not discussed. The AI did remove one unused helper in step 67 ("Remove the unused helper") and re-ran the conversion afterwards so the pickle matched the final code, but did not audit for unused reads or redundant work beyond that.
