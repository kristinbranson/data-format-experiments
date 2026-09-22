# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not glob the data folders. It *parses the authors' own per-animal loading scripts*
(`/app/code/DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m`) with a regex, skipping any
line that starts with `%` so that sessions the authors commented out are excluded, and reads the
`anm`, `date` and `probe` fields out of each entry. That yields 50 sessions, which are then
intersected with the `.mat` files actually present in `/app/data/Ephys_Behavior` and
`/app/data/RandomizedDelay_Ephys_Behavior`, leaving 44 sessions. Each session's `data_structure_*.mat`
is opened once by `load_session`, which sniffs the file header and dispatches to an HDF5 reader
(`_load_v73`, for the 33 v7.3 files) or a `scipy.io` reader (`_load_v7`, for the 11 v7 files).
Unlike a generic loader, only the fields the conversion needs are pulled out of the file
(`bp` trial flags and events, the selected probes' `clu.quality/trial/trialtm`, `sglx.fs`,
`sglx.bitcode.bitstart`, the two DLC views restricted to the requested features, and `ex.probe.loc`).
Motion energy is read from the companion `motionEnergy_<anm>_<date>.mat`. Sessions are processed in a
12-way `multiprocessing.Pool`. Whole conversion: ~16 s wall.

ii.
```python
def parse_sessions():
    """Read the paper's per-animal loading scripts -> [(anm, date, [probes])]."""
    for fn in sorted(glob.glob(os.path.join(LOADSCRIPT_DIR, '*.m'))):
        for line in open(fn):
            s = line.strip()
            if s.startswith('%'):
                continue          # commented-out sessions are excluded by the authors
            m = re.search(r"\.anm\s*=\s*'([^']+)'", s)
            ...
            m = re.search(r"\.probe\s*=\s*\[?([0-9 ,]+)\]?\s*;", s)
```

```python
def load_session(fn, probes, traj_feats):
    if is_v73(fn):
        return _load_v73(fn, probes, traj_feats)
    return _load_v7(fn, probes, traj_feats)
```

```python
sessions = parse_sessions()
sessions = [s for s in sessions if find_files(s['anm'], s['date'])[0] is not None]
with Pool(args.nproc, maxtasksperchild=1) as p:
    res = p.map(process_session, sessions, chunksize=1)
```

iii. From the trajectory (steps 27–29, 82–83): "Load scripts specify anm/date/probe; commented lines
must be skipped. I'll parse them programmatically", and after the run it verified the three data files
*not* in the scripts: "JEB23 2023-10-20 is explicitly commented out by the authors (excluded).
JEB24 2023-10-03/04 aren't in that file at all... Either way, excluding them matches the authors'
curation." The dual reader exists because it found "Some .mat files are v7 format (not HDF5) - need
scipy fallback" (step 33).

## 1-b. How are the data split into subjects?

i. The subject is the `anm` string taken from the load script (equivalently the `<ANM>` part of the
filename), carried on each session result as `r['anm']`. `subjects` is the sorted unique set
(14 mice), and `subject_idx` is each session's index into it. No `obj.meta` field is read.

ii.
```python
subjects = sorted({r['anm'] for r in res})
...
'subjects': subjects,
'subject_idx': np.array([subjects.index(r['anm']) for r in res]),
```

iii. Not explicitly argued in the trajectory; the animal id comes for free from the loading-script
parse, and the AI verified the result (step 69 printed the 14 subjects and the `subject_idx` vector).

## 1-c. How are the data split into sessions?

i. One session = one `(anm, date)` entry of the parsed load scripts = one `data_structure_*.mat`.
`find_files` searches both task folders, so fixed-delay (`Ephys_Behavior`, 25 sessions) and
randomized-delay (`RandomizedDelay_Ephys_Behavior`, 19 sessions) recordings are handled uniformly and
end up as 44 elements of `neural`/`input`/`output`. The folder name is recorded per session in
`metadata.session_info` as `task: 'fixed delay' | 'randomized delay'`. Sessions recorded with two
probes (3 of them) have both probes' units concatenated into one population.

ii.
```python
def find_files(anm, date):
    for d in DATA_DIRS:
        f = os.path.join(d, 'data_structure_%s_%s.mat' % (anm, date))
        if os.path.exists(f):
            me = os.path.join(d, 'motionEnergy_%s_%s.mat' % (anm, date))
            return f, (me if os.path.exists(me) else None), os.path.basename(d)
    return None, None, None
```
```python
for p in probes:
    g = f[clu[p - 1, 0]]
    ...
    units.append({'quality': q, 'trial': trial, 'trialtm': trialtm, 'probe': p})
```

iii. Step 32: "Found session lists: 25 Ephys_Behavior (DR/two-context) + 19 RandomizedDelay sessions
match the paper's counts."

## 1-d. How are the data split into trials?

i. Trials are taken directly from the Bpod table: `obj.bp.Ntrials` sets the count and every per-trial
flag/event (`hit`, `miss`, `no`, `early`, `autowater`, `R`, `L`, `stim.enable`, `ev.goCue`,
`ev.bitStart`) is one entry per trial; all of these are truncated/masked to `N = Ntrials`. Spikes carry
their trial number in `clu.trial` (1-based) and their within-trial time in `clu.trialtm`, and the video
arrays `obj.traj{view}` are already one entry per trial, so no trial boundaries are reconstructed. The
AI explicitly checked the `bp.fidx` trial-offset case used in `findTrials.m` and confirmed it is absent
everywhere.

ii.
```python
N = int(np.array(bp['Ntrials']).ravel()[0])
out['Ntrials'] = N
for k in ['hit', 'miss', 'no', 'early', 'autowater', 'R', 'L']:
    out[k] = np.array(bp[k]).ravel().astype(float)
...
keep = keep[:N]
```
```python
starts = np.searchsorted(tt, trials + 1, 'left')
ends = np.searchsorted(tt, trials + 1, 'right')
```

iii. Step 72: "No session has bp.fidx, and Ntrials always matches the number of video trials — so no
trial index offset is needed."

## 1-e. How are trials filtered based on quality controls?

i. Five filters, in order:
1. **early-lick trials** (`bp.early != 0`) dropped, following the paper.
2. **photostimulation trials** (`bp.stim.enable != 0`) dropped.
3. trials with a **non-finite go cue** dropped (no alignment possible).
4. trials whose index exceeds the number of entries in the **video** array dropped
   (verified never to trigger: `Ntrials` always equals the number of video trials).
5. after binning, trials in which **no unit fired a single spike** are dropped — in two sessions the
   ephys recording stopped before the behavioral session ended, leaving a tail of all-zero trials.
   This removed 61 trials.

Sessions are also dropped if fewer than `MIN_TRIALS = 2` trials survive (never triggered).
Ignore/no-response trials are deliberately **kept**, because "ignore" is a requested outcome class.
Final count: **13,762 trials over 44 sessions** — identical to the expert solution's 13,762.

ii.
```python
keep = (d['early'] == 0) & (d['stim'] == 0) & np.isfinite(gocue)
ntraj = len(d['traj'][TONGUE_VIEW]['trials'])
keep = keep[:N]
if ntraj < N:
    keep[ntraj:] = False
trials = np.where(keep)[0]
if len(trials) < MIN_TRIALS:
    return None
```
```python
# Drop trials in which no unit fired a single spike.  In a couple of sessions
# the ephys recording stopped before the behavioral session ended, leaving a
# tail of trials with no neural data at all.
nonempty = np.any(rates != 0, axis=(0, 1))
n_no_ephys = int((~nonempty).sum())
if n_no_ephys:
    rates = rates[:, :, nonempty]
    trials = trials[nonempty]
```

iii. Docstring: "trials: early-lick and photoinactivation trials removed (the paper removes both from
all analyses). Ignore ('no') trials are kept because 'ignore' is one of the requested outcome
categories." Steps 73–74: "In 2 sessions, the last trials have no spikes at all — the ephys recording
ended before the behavioral session. These trials should be excluded."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` for the ALM probe(s) named in the authors' load script: each cluster's `quality`
(curation label), `trial` (1-based trial of each spike) and `trialtm` (spike time relative to that
trial's start). `obj.bp.ev.goCue` supplies the alignment time. `obj.ex.probe.loc` is read for the
recorded location (stored in metadata only; `brain_regions` is the single entry `'ALM'`).

ii.
```python
q = _h5str(f, g['quality'][i, 0]).strip()
if q in QUAL_EXCLUDE:
    continue
trial = np.array(f[g['trial'][i, 0]]).ravel().astype(int)
trialtm = np.array(f[g['trialtm'][i, 0]]).ravel().astype(float)
units.append({'quality': q, 'trial': trial, 'trialtm': trialtm, 'probe': p})
```

iii. Docstring: processing follows `loadSessionData/processData/getSeq/alignSpikes/
removeLowFRClusters/findClusters.m`; the probe number in `load<ANM>_ALMVideo.m` "selects the ALM
probe(s) of each recording".

## 2-b. How is the `neural` data processed?

i. Per unit, spikes are histogrammed into the 500 go-cue-aligned 10 ms bins, divided by `dt` to give
spikes/s, and smoothed along time with a **port of the paper's `utils/mySmooth.m`**: a
`gausswin(15, alpha=2.5)` kernel whose first `floor(N/2)=7` taps are zeroed (causal half-Gaussian),
normalised to sum 1, convolved `'same'`, with the `'reflect'` boundary condition (the first 15 samples
are prepended and then trimmed). No normalisation, baseline subtraction or z-scoring. Units of both
probes in two-probe sessions are concatenated. Stored as `float32` Hz.

ii.
```python
def gausswin(N, alpha=2.5):
    n = np.arange(N) - (N - 1) / 2.0
    return np.exp(-0.5 * (alpha * n / ((N - 1) / 2.0)) ** 2)


def my_smooth(x, N=SMOOTH, bctype=BCTYPE):
    """Port of utils/mySmooth.m: causal half-Gaussian smoothing along axis 0."""
    if bctype == 'reflect':
        xf = np.concatenate([x[:N], x], axis=0); trim = N
    kern = gausswin(N)
    kern[:N // 2] = 0          # causal
    kern = kern / kern.sum()
    out = np.apply_along_axis(lambda v: np.convolve(v, kern, mode='same'), 0, xf)
    return out[trim:]
```
```python
cnt[:, j] = np.histogram(tm[starts[j]:ends[j]], bins=EDGES)[0]
rates[:, iu, :] = my_smooth(cnt / DT).astype(np.float32)
```

iii. Step 39: "mySmooth: causal half-Gaussian (gausswin with alpha 2.5, first half zeroed), conv
'same', with 'reflect' boundary handling." Docstring: "smoothed with the causal half-Gaussian kernel of
utils/mySmooth.m (params.smooth = 15, params.bctype = 'reflect')."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two unit filters and one session filter.
1. **Curation label**: clusters whose stripped `quality` string is one of
   `garbage`, `gabrga`, `noisy`, `real?` are dropped — exactly the set excluded by
   `findClusters.m` under `params.quality = {'all'}`. Everything else, including `multi`, `fair` and
   `poor`, is kept. (Matching is case-sensitive, as in MATLAB's `ismember`; I verified no case variants
   of the four excluded labels exist in the dataset, so this is safe here.)
2. **Firing rate**: units whose mean rate over the whole window and all kept trials is `<= 1 Hz` are
   dropped (`params.lowFR = 1`, as set in every figure script).
3. **Session**: a session with fewer than 10 surviving units is discarded (never triggered).

Result: **2,457 units** (1,532 fixed-delay + 925 randomized-delay), 17–141 per session. The expert kept
1,954 units; the difference is mostly the expert's additional exclusion of `poor`/`Poor` clusters
(647 such clusters in the v7.3 files alone).

ii.
```python
QUAL_EXCLUDE = ('garbage', 'gabrga', 'noisy', 'real?')
...
# remove low firing rate units (params.lowFR)
mean_fr = rates.mean(axis=(0, 2))
use = mean_fr > LOW_FR
if use.sum() < MIN_UNITS:
    return None
rates = rates[:, use, :]
```

iii. Docstring: "units: cluster qualities except garbage/noisy/'real?' (params.quality = 'all'), then
units with mean firing rate <= 1 Hz removed (params.lowFR = 1)." Step 64–65 cross-checked the totals
against the paper: "1532 units (fixed delay, 25 sessions) and 925 (randomized delay, 19 sessions),
close to the paper's 1651/845 (which are counts before the 1 Hz firing-rate filter)."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. One subtraction, as in `alignSpikes.m`. `clu.trialtm` is already on the behaviour clock relative to
its own trial's start, and `bp.ev.goCue` is on the same clock, so `trialtm - goCue[trial]` gives time
from go cue in seconds. Spikes are then assigned to the 10 ms bins of the fixed
[-2.5, +2.5] s window; spikes outside the window fall outside `EDGES` and are not counted. The trial
index is clipped to `[0, N-1]` only for the go-cue look-up (spikes of out-of-range trials are excluded
later by the `searchsorted` trial selection). The AI also verified that `bp.ev.goCue` exists on
water-cued (autowater) trials, where it marks the water-drop time, so the same alignment is valid in
both contexts.

ii.
```python
tt = u['trial']
tm = u['trialtm'] - gocue[np.clip(tt - 1, 0, N - 1)]   # trialtm_aligned
```

iii. Step 54–55: "WC trials also have a goCue field (water presentation time), so aligning to goCue
works for both contexts." Docstring: "temporal alignment: go cue (obj.bp.ev.goCue; in the water-cued
context this field holds the water-drop time), params.alignEvent = 'goCue'." Step 70 verified the
alignment empirically: tongue visibility, motion energy and firing rate all jump at t = 0.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **10 ms bins**, 500 bins spanning -2.5 to +2.5 s, identical for every trial, session and data
stream. The bin centres are `edges + dt/2`, exactly as `getSeq.m` builds `obj.time`. No rebinning or
resampling after the initial binning: spikes are histogrammed straight onto this grid and the video
streams are interpolated onto it. The expert used 5 ms bins (`params.dt = 1/200` in
`getDefaultParams.m`); the AI chose `dt = 1/100`, which is the value used in most of the paper's
figure scripts (`Scripts/Figure 3/*`, `EDFigure 2/3`, `Behavior/*`, `WorkingWithDataObjs.m`).
Note the module docstring attributes `params.dt = 10 ms` to "the params struct", which is not literally
what `getDefaultParams.m` says, although 1/100 is what the majority of the analysis scripts set.

ii.
```python
TMIN = -2.5           # s, params.tmin
TMAX = 2.5            # s, params.tmax
DT = 0.01             # s, params.dt = 1/100
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TAXIS = (EDGES + DT / 2)[:-1]     # bin centres, obj.time
NT = len(TAXIS)
```

iii. Step 40: "Standard params: tmin=-2.5, tmax=2.5, smooth=15; dt varies between scripts (1/100,
(1/100)*3)." Step 50 also weighed the resulting dataset size against decoder memory before settling on
10 ms.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not derived from a raw variable: it is the time axis defined by the conversion itself — the
centres of the 500 bins of the go-cue-aligned window, i.e. the same grid the spikes are binned onto.
It implicitly depends on `bp.ev.goCue`, which defines t = 0 for each trial.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TAXIS = (EDGES + DT / 2)[:-1]     # bin centres, obj.time
```

iii. Docstring: "inputs: time from the go cue in seconds, time-varying, one value per bin." The window
and bin centres are taken from `params.tmin/tmax` and `getSeq.m`'s `obj.time = edges + params.dt/2`.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None. `TAXIS` is cast to `float32`, reshaped to `(1, 500)` and copied once per trial, so every trial
of every session carries the identical vector -2.495 … +2.495 s.

ii.
```python
time_in = TAXIS.astype(np.float32)[None, :]
for j, t in enumerate(trials):
    inputs.append(time_in.copy())
```

iii. N/A — the axis is defined by the conversion.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid. Spike times are expressed as `trialtm - goCue[trial]` and
histogrammed into `EDGES`; the input is the centre of those same bins, so column *k* of `input` and
column *k* of `neural` refer to the same 10 ms interval. The video-derived outputs are interpolated
onto `TAXIS` as well, so all four streams share one time axis.

ii.
```python
cnt[:, j] = np.histogram(tm[starts[j]:ends[j]], bins=EDGES)[0]
...
x = interp_nan(TAXIS, tt, tr['xy'][:, 0, fi])
```

iii. N/A.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial Bpod flags: `bp.R` and `bp.L` (the instructed port) and `bp.hit` / `bp.no` (outcome).
The licked side is not recorded as such, so it is inferred from instructed side × outcome; `bp.no`
(no response) provides the third class.

ii.
```python
hit, miss, no = d['hit'] > 0, d['miss'] > 0, d['no'] > 0
R, L = d['R'] > 0, d['L'] > 0
```

iii. Step 57: the AI validated the inference against the actual first lick after the go cue
(`bp.ev.lickL/lickR`) on a two-context session and found **98.9 % agreement** (98.8 % DR, 99.1 % WC),
with only 3 of 84 "no" trials containing any lick — so the derived choice is a faithful stand-in for
the observed lick.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. `right_lick = (R & hit) | (L & miss)` — i.e. the animal licked right if it was a right trial it got
correct or a left trial it got wrong. No-response trials take class 2. Codes: 0 = left, 1 = right,
2 = none. The value is constant within a trial and is broadcast across all 500 bins. Distribution over
the dataset: 42.3 % left, 44.6 % right, 13.1 % none.

ii.
```python
# lick direction: right if (right trial & correct) or (left trial & error)
# (identical to funcs/getPrevChoice.m); ignore trials have no lick
right_lick = (R & hit) | (L & miss)
lick_dir = np.where(no, 2, np.where(right_lick, 1, 0))       # 0 left,1 right,2 none
...
out[0, :] = lick_dir[t]
```

iii. The code comment cites `funcs/getPrevChoice.m` as the source of the same rule, and step 57
provides the empirical validation quoted above.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The single per-trial flag `bp.autowater`, which marks the water-cued trials (water delivered at a
random port with all auditory cues omitted).

ii.
```python
aw = d['autowater'] > 0
```

iii. Docstring: "In the water-cued (WC) task all auditory cues were omitted and a water drop was
presented at a random time at a randomly chosen port." The AI also inspected `goCue`/reward times for
autowater vs DR trials (step 54) before relying on the flag.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling: autowater → 0 (WC), everything else → 1 (DR), constant within a trial and
broadcast across bins. Dataset-wide 9.7 % WC / 90.3 % DR; `session_info` records per-session WC/DR
counts, and 14 of the 44 sessions contain no WC trials at all.

ii.
```python
context = np.where(aw, 0, 1)                                 # 0 WC, 1 DR
...
out[1, :] = context[t]
```

iii. Codes follow the prompt's WC/DR ordering (`output_values[1] = ['WC', 'DR']`).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The three mutually exclusive per-trial Bpod flags `bp.hit`, `bp.miss` and `bp.no`. Unlike the expert
(who derives "ignore" as `~hit & ~miss`), the AI reads `bp.no` explicitly.

ii.
```python
hit, miss, no = d['hit'] > 0, d['miss'] > 0, d['no'] > 0
```

iii. Step 45/48 explored "hit/miss/no by context" before choosing; step 57 also confirmed that "no"
trials essentially never contain a lick (3/84), so `no` really is the ignore class.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Relabelling into the three requested classes: 0 = incorrect (miss), 1 = correct (hit), 2 = ignore
(`no`). Constant within a trial, broadcast across bins. Ignore trials are kept in the dataset rather
than dropped (the paper omits them from its analyses) precisely because "ignore" is a requested class.
Distribution: 12.0 % incorrect, 74.9 % correct, 13.1 % ignore.

ii.
```python
outcome = np.where(no, 2, np.where(hit, 1, 0))               # 0 incorrect,1 correct,2 ignore
...
out[2, :] = outcome[t]
```

iii. Docstring: "Ignore ('no') trials are kept because 'ignore' is one of the requested outcome
categories."

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut trace of the **`tongue` feature of the side camera only** (`obj.traj{1}`, view index
0): its x and y columns (`ts[:, 0:2, feat]`), plus that view's `frameTimes` and `NdroppedFrames`.
The bottom camera's tongue markers (`top_tongue`, `bottom_tongue`, …) are not used, although the AI
listed them when it enumerated `featNames` (step 30). `sglx.fs`, `sglx.bitcode.bitstart` and
`bp.ev.bitStart` are also needed, for the video/ephys clock offset. The likelihood column is never
read — the AI relies on the fact (verified in step 44/45, and confirmed independently here) that the
authors already set x and y to NaN exactly where DeepLabCut likelihood <= 0.9.

ii.
```python
TONGUE_VIEW, TONGUE_FEATS = 0, ['tongue']
...
xy = np.stack([ts[i, 0:2, :].T for i in fidx], axis=-1)  # (nframes,2,nfeat)
```

iii. Code comment: "The tongue is tracked in the side camera (view 1) and the paws only in the bottom
camera (view 2), so each is taken from the camera that tracks it." Step 45: "DLC ts contains NaNs where
features are not visible (tongue mostly NaN)."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Exactly the paper's `findPosition.m` / `findVelocity.m` recipe, kept in speed form:
frame times are put on the go-cue clock, x and y are **linearly interpolated onto the 500-bin time
axis** with `interp_nan` (NaN outside the frames of that trial and NaN wherever DLC did not detect the
tongue, i.e. no `fillmissing` — matching the reference, which never fills the tongue), then
`vx = gradient(x)/dt`, `vy = gradient(y)/dt` and the speed `sqrt(vx² + vy²)` is taken so that a single
scalar per bin can be discretized. No smoothing is applied to the tongue position — `findPosition.m`
explicitly skips smoothing for tongue features. There is only one camera, so no cross-view
normalisation is done. Bins whose speed is NaN become class 2. About 94.7 % of bins end up
"not visible".

ii.
```python
tt = ft - vidshift - gocue[t]
for fi in feats_idx:
    x = interp_nan(TAXIS, tt, tr['xy'][:, 0, fi])
    y = interp_nan(TAXIS, tt, tr['xy'][:, 1, fi])
    vx = np.gradient(x) / DT
    vy = np.gradient(y) / DT
    sp.append(np.sqrt(vx ** 2 + vy ** 2))
```

iii. Docstring: "velocities are the gradient of the interpolated position, as in
funcs/kinematics/findVelocity.m, but expressed as a speed so that a single scalar per time point can be
discretized. Following the reference code, tongue positions are NaN whenever DeepLabCut does not detect
the tongue, and those time points are labelled 'not visible' (category 2) rather than being filled in."

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per session, over all trials at once: the threshold is the 50th percentile of the **visible
(finite)** samples only; bins `>= threshold` are class 1, `<` are class 0, NaN bins are class 2. The
threshold is recorded in `metadata.session_info[i]['tongue_speed_threshold']`. Because the split is
over visible samples only, classes 0 and 1 come out exactly balanced within the visible bins
(2.63 % / 2.63 % / 94.74 % of all bins across the dataset).

ii.
```python
def discretise(x):
    vis = np.isfinite(x)
    out = np.full(x.shape, 2, dtype=np.int64)          # 2 = not visible / no video
    if vis.sum() > 0:
        thr = np.percentile(x[vis], 50)
        out[vis] = (x[vis] >= thr).astype(np.int64)
    return out, (float(np.percentile(x[vis], 50)) if vis.sum() else None)

tongue_cat, tongue_thr = discretise(tongue_speed)
```

iii. Step 66: "the discretization defines percentile over visible samples only. The task says
'0: < 50th percentile, 1: >= 50th percentile, 2: not visible' — yes, threshold computed over visible
values, which is what I did."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The camera clock is corrected once per session with the bitcode offset of `findVideoOffset.m`:
`vidshift = median(sglx.bitcode.bitstart)/sglx.fs - median(bp.ev.bitStart)` (the reference uses `mode`;
I checked three sessions and median and mode agree to < 1 µs). Then
`t = frameTimes - vidshift - goCue[trial]` and the positions are interpolated onto `TAXIS`, the same
grid the spikes are binned onto. Bins outside a trial's video coverage return NaN → class 2.

ii.
```python
# video/ephys offset, funcs/findVideoOffset.m
vidshift = (np.median(d['bitstart_sglx']) / d['fs']) - np.median(d['bitStart'])
...
tt = ft - vidshift - gocue[t]
x = interp_nan(TAXIS, tt, tr['xy'][:, 0, fi])
```

iii. Step 53: "kinematics from DLC interpolated onto the same time axis using vidshift =
mode(bitcode.bitstart)/fs - mode(bp.ev.bitStart)". Step 70 verified the result empirically: tongue
visibility jumps from ~0.01 before the go cue to 0.26 just after it, peaking at ~0.9 s.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The bottom camera (`obj.traj{2}`, view index 1), using **both** `top_paw` and `bottom_paw`, plus that
view's `frameTimes` and `NdroppedFrames`. The two markers are two different forepaws, not two views of
one paw.

ii.
```python
PAW_VIEW, PAW_FEATS = 1, ['top_paw', 'bottom_paw']
```

iii. Code comment: "the paws only in the bottom camera (view 2), so each is taken from the camera that
tracks it"; docstring: "paw velocity the mean speed of the bottom-camera 'top_paw' and 'bottom_paw'
markers". The AI did not examine the relative tracking reliability of the two markers.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Identical machinery to the tongue — interpolate x and y onto `TAXIS`, `gradient/dt`, speed —
computed separately for `top_paw` and `bottom_paw`, and then **averaged over whichever marker is
available in that bin** (`np.nanmean`, NaN only if both are missing). No smoothing, no normalisation
between the markers, no `fillmissing` (the reference `findPosition.m` does apply `fillmissing(...,
'nearest')` to non-tongue features; the AI deliberately leaves the gaps so they can take the "not
visible" class). Values are in pixels/s. 6.5 % of bins end up "not visible".

ii.
```python
sp = np.vstack(sp)
with np.errstate(invalid='ignore'):
    spd = np.where(np.all(np.isnan(sp), axis=0), np.nan, np.nanmean(sp, axis=0))
```

iii. Docstring, as quoted in 8-a. Step 66 flagged one aspect of it: "my paw 'not visible' arises from
time outside video coverage rather than DLC non-detection. The paw is always detected (conf ~1.0).
Hmm — is that the intended meaning of 'not visible'? For paw, category 2 would effectively be 'no video
for that timepoint'. That's a justifiable mapping." (On the session I checked, `bottom_paw` is in fact
detected in only ~33 % of bins while `top_paw` reaches ~78 %, so the premise of that note is only true
of `top_paw`.)

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The same `discretise` helper as the tongue: per-session 50th percentile of the finite samples,
`>=` → 1, `<` → 0, NaN → 2; threshold stored as `paw_speed_threshold` in `session_info`.
Dataset-wide: 46.75 % / 46.75 % / 6.5 %.

ii.
```python
paw_cat, paw_thr = discretise(paw_speed)
```

iii. As in 7-c — the per-session median split is the prompt's specification.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Identically to the tongue, but using the **bottom** camera's own `frameTimes` (each view is timed by
its own frame times, so differing frame counts between views are handled):
`t = frameTimes - vidshift - goCue[trial]`, then interpolation onto `TAXIS`.

ii.
```python
for view, feats_idx, dest in ((tv, t_idx, 'tongue'), (pv, p_idx, 'paw')):
    tr = view['trials'][t] if t < len(view['trials']) else None
    ...
    ft = tr['frameTimes']
    tt = ft - vidshift - gocue[t]
```

iii. Same session-constant offset and same grid as every other stream; no separate treatment argued.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The companion file `motionEnergy_<anm>_<date>.mat`, one trace per trial with one value per camera
frame. `me.data` is read, and unwrapped a second time when it is itself a struct
(`me.data.data`). `obj.me` is not used. If the motion-energy file were missing, the whole session's
motion energy would be class 2 (all 44 sessions have one).

ii.
```python
def load_motion_energy(fn):
    m = sio.loadmat(fn, squeeze_me=True, struct_as_record=False)
    me = m['me']
    data = me.data
    if hasattr(data, '_fieldnames'):
        data = data.data
    return [np.atleast_1d(np.asarray(d, dtype=float)).ravel() for d in np.atleast_1d(data)]
```

iii. Step 42: "Need a universal loader... inspect motionEnergy file structure." The double unwrap
mirrors `loadMotionEnergy.m`'s `if isstruct(me.data), me.data = me.data.data; end`. I confirmed all 45
motion-energy files in `/app/data` load without error through this function.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. None beyond interpolation onto the common time axis — the value is already one scalar per frame
(the paper's per-pixel frame differencing reduced to the 99th percentile across pixels). The AI checked
that `len(frameTimes) == len(me[trial])` for every trial of the sessions it examined and uses that
equality as a guard before interpolating.

ii.
```python
if tr is not None and tr['frameTimes'].size == len(me_raw[t]):
    tt = tr['frameTimes'] - vidshift - gocue[t]
    me_t[:, j] = interp_nan(TAXIS, tt, me_raw[t])
elif tr is not None:
    ftimes = (np.arange(len(me_raw[t])) + 1) / 400.0
    me_t[:, j] = interp_nan(TAXIS, ftimes - 0.5 - gocue[t], me_raw[t])
```

iii. Step 58 verified the length agreement on two sessions ("lens sample [(1792, 1792, 1792), …]
mismatch 0"). The fallback branch uses a hard-coded 0.5 s shift instead of the measured `vidshift`
(~0.490 s); this is an undocumented magic number, but it never fired in the sessions the AI checked.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same `discretise` helper: per-session 50th percentile over finite samples, `>=` → 1, `<` → 0,
NaN (no video coverage for that bin, or no motion-energy file) → 2, recorded as
`motion_energy_threshold`. Dataset-wide: 48.0 % / 48.1 % / 3.9 %.

ii.
```python
me_cat, me_thr = discretise(me_t)
```

iii. As in 7-c.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is one value per **side-camera** frame, so it is timed with the side camera's
`frameTimes`, corrected by the same session `vidshift` and the trial's go cue, and interpolated onto
`TAXIS`.

ii.
```python
# motion energy (sampled with the video frames)
if me_raw is not None and t < len(me_raw):
    tr = tv['trials'][t] if t < len(tv['trials']) else None
    if tr is not None and tr['frameTimes'].size == len(me_raw[t]):
        tt = tr['frameTimes'] - vidshift - gocue[t]
        me_t[:, j] = interp_nan(TAXIS, tt, me_raw[t])
```

iii. Step 70 checked the alignment empirically: the fraction of "high motion energy" bins jumps from
~0.32 just before the go cue to ~0.83 just after, peaking at t ≈ +0.13 s.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Eight distinct guards, all of which either keep the trial and mark the gap, or skip a trial when no
alignment is possible:
- **two MATLAB file formats** — header sniffed, `h5py` or `scipy.io` used accordingly.
- **three motion-energy layouts** — `me.data`, unwrapped again when it is a struct.
- **non-finite go cue** — trial dropped (cannot be aligned).
- **`NdroppedFrames` NaN or empty** — that trial's kinematics are skipped, following `findPosition.m`'s
  identical check; the trial is kept and its tongue/paw bins become "not visible".
- **missing `frameTimes`** — replaced by `(1:nframes)/400`, the same fallback `findPosition.m` uses.
- **`ts` not 3-D** — the trial's video entry is set to `None` and skipped.
- **DLC non-detection** — x/y are already NaN, propagate through the interpolation and gradient, and
  become class 2; nothing is interpolated or filled in.
- **bins outside a trial's video coverage** — `interp_nan` returns NaN → class 2.
- **trials with no spikes at all** — dropped (see 1-e).
- **`ex.probe.loc` unreadable** — falls back to `'ALM'` inside a `try/except`.

ii.
```python
tr = view['trials'][t] if t < len(view['trials']) else None
if tr is None or tr['Ndropped'].size == 0 or np.any(~np.isfinite(tr['Ndropped'])):
    continue          # video for this trial is unusable
ft = tr['frameTimes']
if ft.size == 0 or not np.any(np.isfinite(ft)):
    ft = (np.arange(tr['xy'].shape[0]) + 1) / 400.0
```
```python
def interp_nan(xnew, xp, fp):
    """Linear interpolation that returns NaN outside the sampled range and
    propagates NaNs in fp, like MATLAB's interp1."""
```

iii. Docstring: "Following the reference code, tongue positions are NaN whenever DeepLabCut does not
detect the tongue, and those time points are labelled 'not visible' (category 2) rather than being
filled in. Time points that fall outside the video recording of a trial are likewise labelled 'not
visible'/'no video'." Step 58 measured the coverage the last rule applies to: 92.6–98.8 % of the window
is covered on average, with only 15–26 % of trials fully covered.

## 11-a. What are the most time-consuming steps of the code?

i. Reading the `.mat` files. The full conversion takes **16 s wall / 1 m 52 s CPU** across 10–12 worker
processes, and nearly all of the CPU time is `h5py`/`scipy.io` I/O and decompression of the 100–300 MB
session files. Two things keep this small: `_load_v73` reads only the fields the conversion needs
(never `spkWavs`, `clu.tm`, the untouched DLC features, or `sglx`'s per-trial index arrays), and
sessions are processed in parallel with `multiprocessing.Pool`. After loading, the per-unit × per-trial
spike histogram loop and `my_smooth` (`np.apply_along_axis` over columns) are the next largest costs.

ii.
```python
with Pool(args.nproc, maxtasksperchild=1) as p:
    res = p.map(process_session, sessions, chunksize=1)
```
```python
for v, feats in traj_feats.items():
    g = f[o['traj'][v, 0]]
    names = [_h5str(f, r) for r in np.array(f[g['featNames'][0, 0]]).ravel()]
    fidx = [names.index(ft) for ft in feats]      # only the requested features are read
```

iii. Not argued explicitly; the AI stated only that the run was "Fast (2 sessions in 4s)" and that the
full conversion "ran in 16s".

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Three:
- the **per-unit × per-trial spike histogram** (`for iu in units: for j in trials: np.histogram(...)`),
  which is ~2,500 units × ~310 trials calls to `np.histogram`. A single `np.histogram2d` over
  (trial, time) per unit — or one `bincount` over the whole session — would replace the inner loop
  entirely. This is the one clearly avoidable loop.
- `my_smooth`'s `np.apply_along_axis(np.convolve, 0, xf)`, which convolves column by column in Python;
  `scipy.signal.fftconvolve` or `oaconvolve` along axis 0 would do it in one call.
- `interp_nan` is called separately for x and y even though both share the same `xp`, so the
  `argsort`/`searchsorted` work is done twice; it could interpolate a 2-column array at once.

The per-trial video loop itself is not vectorizable, since each trial has a different number of frames.

ii.
```python
for iu, u in enumerate(units):
    ...
    cnt = np.zeros((NT, len(trials)))
    for j in range(len(trials)):
        if ends[j] > starts[j]:
            cnt[:, j] = np.histogram(tm[starts[j]:ends[j]], bins=EDGES)[0]
```

iii. Not discussed in the trajectory. In practice it does not matter much: the run is I/O-bound and
finishes in 16 s wall.

## 11-c. What processing does the code repeat multiple times?

i. A handful of small redundancies, none of them significant:
- `discretise` computes `np.percentile(x[vis], 50)` **twice** — once for the classification and again in
  the return expression.
- `interp_nan` repeats the finite-mask, `argsort` and `searchsorted` work for x and y of the same
  feature (and again for each of the two paw markers, all of which share one `frameTimes`).
- `tm = u['trialtm'] - gocue[...]` and the `argsort` are computed for **all** of a unit's spikes,
  including those of trials that were already excluded.
- the go-cue-aligned rate is computed and smoothed for every surviving cluster, including the ~30 % that
  are then removed by the 1 Hz filter.
- `vidshift` is, correctly, computed once per session, and the bin grid once at module level, so those
  are *not* repeated.

ii.
```python
    if vis.sum() > 0:
        thr = np.percentile(x[vis], 50)
        out[vis] = (x[vis] >= thr).astype(np.int64)
    return out, (float(np.percentile(x[vis], 50)) if vis.sum() else None)
```

iii. Not discussed in the trajectory.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Little, by design — the loader is field-selective, so unlike a whole-tree reader it never
materialises `clu.spkWavs`, `clu.tm`, `sglx`'s per-trial index arrays, or the DLC features other than
`tongue`, `top_paw` and `bottom_paw`. What is still computed and then thrown away:
- `bp.sample`, `bp.delay` and `has_fidx` are loaded and never used (the AI used `fidx` only for a
  one-off check); `bp.L` is used only through `(L & miss)`.
- `ex.probe.loc` is parsed per session (`'L ALM'`, `'R ALM'`, `'ALM'`) and stored in `session_info`, but
  `brain_regions` is hard-coded to the single value `'ALM'` and every `brain_region_idx` is zeros, so it
  does not affect the decoder.
- `n_single_units` is computed for metadata only.
- Smoothed firing rates are produced for all clusters and only then filtered by the 1 Hz criterion
  (~2,457 of ~3,500 kept), and for the 61 trials later dropped as having no ephys.
- `NdroppedFrames` is loaded per trial and used only as a validity flag.
- The output array is stored as `int64` although every value is in {0, 1, 2}; `int8` would cut that part
  of the pickle by 8× (the file is 1.9 GB).

ii.
```python
for k in ['goCue', 'sample', 'delay', 'bitStart']:
    out[k] = np.array(bp['ev'][k]).ravel().astype(float)
out['has_fidx'] = 'fidx' in bp
```
```python
out = np.empty((6, NT), dtype=np.int64)
```

iii. Not discussed in the trajectory; the field-selective loading was motivated by file size
("Files are large (100-300MB)", step 19).
