# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a list of 25 `(animal, date, probe(s))` tuples in `SESSIONS`, transcribed by hand from the uncommented entries of the authors' `code/DataLoadingScripts/Recording and video/load<ANM>_ALMVideo.m` files. Only the `Ephys_Behavior` folder is read (`DATA_DIR = '/app/data/Ephys_Behavior'`); the 22 `data_structure_*.mat` files in `RandomizedDelay_Ephys_Behavior` (19 of which are listed in the same authors' loading scripts) are deliberately not loaded. Each session's `data_structure_<anm>_<date>.mat` is opened once by a purpose-written HDF5 reader (`sessionio.Session`, h5py-only, reading only the fields actually needed rather than the whole `obj` tree); the matching `motionEnergy_<anm>_<date>.mat` is read separately with `scipy.io.loadmat`. There is no v7/`scipy.io` fallback for the data structures, so only MATLAB v7.3 files can be opened.

ii.
```python
DATA_DIR = '/app/data/Ephys_Behavior'
...
SESSIONS = [
    ('JEB6',  '2021-04-18', [2]),
    ('JEB7',  '2021-04-29', [1]),
    ...
    ('JEB19', '2023-04-21', [1]),
]
...
def process_session(anm, date, probes, verbose=True):
    path = os.path.join(DATA_DIR, 'data_structure_%s_%s.mat' % (anm, date))
    s = Session(path)
```
```python
# sessionio.py
class Session:
    def __init__(self, path):
        self.f = h5py.File(path, 'r')
        self.o = self.f['obj']
```
```python
for anm, date, probes in sessions:
    res = process_session(anm, date, probes)
```

iii. From the trajectory: the AI first tried `mat73`, found it returned `None` for the `clu` struct arrays, then wrote its own h5py reader, and later replaced a generic recursive converter with targeted readers because "generic recursive conversion of traj is slow" (steps 19–21, 47). For session selection it grepped the figure scripts and concluded (step 26–27, 31) "Main dataset = Ephys_Behavior 25 sessions (10 mice) used in most figure scripts. Randomized delay is a separate secondary dataset", after seeing that `loadJEB11/12/23/24_ALMVideo.m` are called only from `EDFigure2a_Right.m`, `Figure3h.m` and `Figure3i.m`. In step 53 it checked the randomized-delay files and found that 11 of the 22 are not HDF5 (`NOT-HDF5 OSError`) and that two JEB24 files have no `clu` field at all, but it never revisited the decision or added a `scipy.io` fallback. The final summary (step 82) states the dataset is "exactly the sessions listed in the paper's `load<ANM>_ALMVideo.m` meta scripts that drive the main figures".

## 1-b. How are the data split into subjects (mice)?

i. The animal is the first element of each `SESSIONS` tuple (it is also the prefix of the filename). It is carried on each session's result as `subject`. At assembly, `subjects` is built in first-encounter order and `subject_idx` holds each session's index into that list. The result is 10 subjects over 25 sessions (`JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, JEB13, JEB14, JEB15, JEB19`).

ii.
```python
return dict(neural=neural, input=inp, output=outputs, regions=regions,
            subject=anm, info=info)
```
```python
if res['subject'] not in data['subjects']:
    data['subjects'].append(res['subject'])
data['subject_idx'].append(data['subjects'].index(res['subject']))
...
data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. Not discussed explicitly in the trajectory. The animal id is taken from the authors' own meta scripts (`meta(end).anm`), which is the same identifier used in the filenames; the AI's `parse_meta.py` exploration script (step 23–24) extracted `anm`/`date`/`probe` triples from those scripts and had to "carry forward" the animal name for continuation entries, which is why the final list hard-codes the animal per session.

## 1-c. How are the data split into sessions?

i. One `(anm, date)` pair = one file = one session = one element of `neural`, `input`, `output`, `subject_idx` and `brain_region_idx`. Sessions are processed independently in a loop; a session is dropped if it has fewer than 10 curated units or fewer than 2 surviving trials. 25 sessions are produced (all 25 `data_structure_*.mat` files present in `Ephys_Behavior`; none were dropped by the unit/trial criteria). The 19 randomized-delay sessions used by the paper's Fig. 3g–i are excluded entirely.

ii.
```python
MIN_UNITS = 10        # Methods: sessions need at least 10 units
...
if fr.shape[0] < MIN_UNITS or len(trials) < 2:
    if verbose:
        print('  skipping %s %s: %d units, %d trials' % ...)
    s.close()
    return None
```
```python
for anm, date, probes in sessions:
    res = process_session(anm, date, probes)
    if res is None:
        continue
    data['neural'].append(res['neural'])
```

iii. Step 43 of the trajectory: the AI grepped for `UseInclusionCritera`/`RemoveUnwantedSessions` and checked `EDFigure2a` for per-session unit criteria; the Methods state "Recording sessions were included for analysis only if they had at least 10 units", which is the source of `MIN_UNITS = 10`. The exclusion of the randomized-delay folder is justified as in 1-a (secondary dataset used only in Fig. 3h,i and ED Fig. 2a right).

## 1-d. How are the data split into trials?

i. Trials are the rows of the Bpod table: every per-trial field of `obj.bp` (`early`, `hit`, `miss`, `no`, `R`, `L`, `autowater`, `stim.enable`, `ev.goCue`) is read as a vector of length `Ntrials`, and a trial index is a position in those vectors. Spikes already carry their trial number (`clu.trial`, 1-based) and the DLC/motion-energy data are already stored per trial (one cell per trial), so no trial boundaries are reconstructed. The kept trials are `trials = np.where(keep)[0]`, and a lookup table `trial_pos` maps a 1-based raw trial number onto its position in the output.

ii.
```python
ntrials_all = s.ntrials
gocue = s.ev('goCue')
early = s.bp_flag('early')
...
trials = np.where(keep)[0]
...
trial_pos = -np.ones(ntrials_all + 1, dtype=int)
trial_pos[trials + 1] = np.arange(len(trials))   # bp trials are 1-indexed
```
```python
# sessionio.py
@property
def ntrials(self):
    return int(vec(self.f, self.o['bp']['Ntrials'])[0])

def bp_flag(self, name):
    return vec(self.f, self.o['bp'][name]).astype(bool)
```

iii. Not argued at length; the AI verified in step 52 that the per-trial flags are self-consistent (lick direction derived from `hit`/`miss`/`R`/`L` agreed with the actual `bp.ev.lickL`/`lickR` times) and in step 50 that `goCue` exists for both DR and autowater trials, which is what makes the go cue usable as the per-trial alignment event.

## 1-e. How are trials filtered based on quality controls?

i. Three masks, combined before anything is computed: early-lick trials (`bp.early`) are dropped, photostimulation trials (`bp.stim.enable`) are dropped, and trials whose `goCue` is not finite are dropped. Ignore (no-response) trials are deliberately kept, because `ignore` is one of the required outcome classes. No filter for trials that run past the end of the ephys recording is applied (I verified that none of the 25 selected sessions contains such a trial, so this makes no difference here). 7,426 of 8,760 trials survive.

ii.
```python
# ------------------------------------------------ trial curation
keep = (~early) & (~stim) & np.isfinite(gocue)
trials = np.where(keep)[0]
```

iii. Module docstring: "trial curation: early-lick trials and photoinactivation trials are excluded (all `params.condition` strings contain `~early & ~stim.enable`; the Methods state early-lick trials were omitted from all analyses). Ignore (no-response) trials are kept because 'ignore' is one of the outcome classes to decode." The AI reached this in step 25 by grepping `params.condition` in the figure scripts, and confirmed in step 54 by checking `goCue`/NaN patterns on early trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` — the spike-sorted clusters of the probe(s) listed for that session in the authors' meta scripts. For each cluster the AI reads `trial` (1-based trial of each spike), `trialtm` (spike time relative to that trial's start) and `quality` (manual curation label). `obj.bp.ev.goCue` supplies the alignment time, and `obj.ex.probe.loc` supplies the anatomical label used to fill `brain_region_idx`.

ii.
```python
for prb in probes:
    qual = s.qualities(prb)
    cluix = [i for i, q in enumerate(qual) if q.lower() not in BAD_QUALITY]
    ...
    for ii, ci in enumerate(cluix):
        tr, ttm = s.clu_spikes(prb, ci)
        aligned = ttm - gocue[tr - 1]                 # alignSpikes.m
```
```python
# sessionio.py
def clu_spikes(self, prbnum, cluix):
    g = self.probe_group(prbnum)
    trial = np.array(self.f[np.array(g['trial'][()]).ravel()[cluix]][()]).ravel().astype(int)
    trialtm = np.array(self.f[np.array(g['trialtm'][()]).ravel()[cluix]][()]).ravel().astype(float)
    return trial, trialtm
```

iii. From step 20: the AI inspected `clu` and found "a 2x1 cell array of struct arrays ... with fields quality, site, spkWavs, tm, trial, trialtm", and from `alignSpikes.m` that `trialtm` is the field to align. Probes are taken from the meta scripts (e.g. JEB15 uses `[1 2]`, and the loader comment "removed probe 1 b/c no qualities, doesn't seem to be sorted" for 2022-07-29 is respected).

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed into 10 ms bins spanning −2.5 to +2.5 s from the go cue, divided by the bin width to give spikes/s, and then smoothed along time with a re-implementation of the authors' `mySmooth(x, 15, 'reflect')`: a `gausswin(15)` (alpha = 2.5) with its first `floor(15/2)` taps zeroed (i.e. a **causal** kernel), normalised to sum 1, applied by 'same' convolution after prepending the first 15 samples of the trace as the boundary condition. No normalisation, z-scoring or baseline subtraction. Units from both probes of a two-probe session are concatenated into one population. Values are stored as `float32` firing rates in Hz.

ii.
```python
def causal_gaussian_kernel(n=SMOOTH_N):
    """MATLAB gausswin(n) (alpha=2.5) with the acausal half zeroed, normalised."""
    k = np.arange(n)
    alpha = 2.5
    w = np.exp(-0.5 * (alpha * (k - (n - 1) / 2) / ((n - 1) / 2)) ** 2)
    w[:n // 2] = 0.0                # 'causal' step in mySmooth.m
    return w / w.sum()

def smooth_causal(x):
    """mySmooth(x, 15, 'reflect') along the last axis."""
    pad = x[..., :SMOOTH_N]
    xf = np.concatenate([pad, x], axis=-1)
    out = convolve1d(xf, KERN, axis=-1, mode='constant', cval=0.0)
    return out[..., SMOOTH_N:]
```
```python
    bi = np.floor((aligned - TMIN) / DT).astype(int)
    ti = trial_pos[tr]
    ok = (bi >= 0) & (bi < NT) & (ti >= 0)
    np.add.at(counts[ii], (ti[ok], bi[ok]), 1.0)
rate = smooth_causal(counts / DT)                 # getSeq.m
...
fr = np.concatenate(fr_list, axis=0)                  # (units, trials, time)
```

iii. Steps 34–35, 56, 72–73: the AI read `mySmooth.m`, noted that it "uses a causal gaussian window (gausswin(N), first half zeroed, normalized), with boundary handling", replicated it, and then explicitly validated it with an impulse test — "Causal smoothing verified ... matches MATLAB's mySmooth (reflect pad = first N samples prepended, 'same' convolution)", checking that the residual values before the impulse were numerical noise. Dividing counts by `params.dt` to get spikes/s is taken from `getSeq.m` (`mySmooth(N./params.dt, params.smooth, params.bctype)`).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Three filters. (1) Clusters whose manual `quality` label, lower-cased, is one of `garbage`, `gabrga`, `noisy`, `real?` are dropped — exactly the list in `findClusters.m`; every other label (including multi-units and `poor`) is kept. (2) Units whose mean firing rate over the whole −2.5 to 2.5 s window and all kept trials is ≤ 1 Hz are dropped (`params.lowFR = 1`). (3) A session with fewer than 10 surviving units (or fewer than 2 trials) is dropped entirely. 1,531 units survive across the 25 sessions (27–141 per session, mean 61), labelled 1,386 ALM and 145 tjM1.

ii.
```python
LOW_FR = 1.0          # Hz, params.lowFR used in all figure scripts
MIN_UNITS = 10        # Methods: sessions need at least 10 units
BAD_QUALITY = ('garbage', 'gabrga', 'noisy', 'real?')  # findClusters.m
...
cluix = [i for i, q in enumerate(qual) if q.lower() not in BAD_QUALITY]
...
# low firing rate units (removeLowFRClusters.m)
meanfr = fr.mean(axis=(1, 2))
use = meanfr > LOW_FR
fr = fr[use]
regions = regions[use]
if fr.shape[0] < MIN_UNITS or len(trials) < 2:
    ...
    return None
```

iii. Module docstring: "clusters labelled 'garbage'/'noisy'/'real?' are dropped (findClusters.m) and units with mean firing rate <= 1 Hz are removed (removeLowFRClusters.m with params.lowFR = 1, as in the figure scripts and the Methods: 'All units with firing rates exceeding 1 Hz were included')". Step 13 notes the default in `getDefaultParams.m` is 0.5 Hz but that the figure scripts and Methods use 1 Hz, and the AI chose 1 Hz. Step 60 sanity-checks a suspicious session: "JEB13 2022-09-21 has 115 units above 1 Hz out of 118 non-garbage—seems high but that's the data."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. By subtracting the trial's own go cue from the spike time: `clu.trialtm` is already on the behaviour clock and relative to that trial's start, and `bp.ev.goCue` is on the same clock, so one subtraction per spike puts every spike in seconds from the go cue. No interpolation or extra offset. In WC blocks `goCue` is the water-drop time, which the AI verified exists on autowater trials. Spikes landing outside [−2.5, 2.5] s or in a dropped trial are discarded by the `ok` mask.

ii.
```python
tr, ttm = s.clu_spikes(prb, ci)
aligned = ttm - gocue[tr - 1]                 # alignSpikes.m
bi = np.floor((aligned - TMIN) / DT).astype(int)
ti = trial_pos[tr]
ok = (bi >= 0) & (bi < NT) & (ti >= 0)
```

iii. Step 13/55: `params.alignEvent = 'goCue'` in the authors' scripts and `alignSpikes.m` does `trialtm_aligned = trialtm - event`. Step 50 explicitly checked "do autowater (WC) trials have goCue times" and found valid go-cue times for both DR and WC trials (`gcWC` printed per session), which is recorded in the metadata as "go cue onset (auditory go cue in DR blocks, water drop in WC blocks)".

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 10 ms bins (`DT = 0.01`), 500 bins spanning −2.5 to +2.5 s around the go cue, identical for every trial and session. Spikes are binned directly at that resolution (there is no fine binning followed by rebinning), and every other stream (input time axis, tongue, paw, motion energy) is resampled onto the same 500-bin grid, so there is a single shared time axis. `metadata['time_bin_size'] = 10.0` ms.

ii.
```python
TMIN = -2.5           # s relative to go cue (params.tmin)
TMAX = 2.5            # s relative to go cue (params.tmax)
DT = 0.01             # s, bin size (params.dt = 1/100)
...
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME = EDGES[:-1] + DT / 2          # bin centres, as in getSeq.m
NT = len(TIME)
```

iii. Step 25/55: "Standard parameters identified: alignEvent goCue, tmin -2.5, tmax 2.5, dt 1/100, smooth 15, bctype reflect, quality all, lowFR 1." The AI saw both values in the repository (step 13: "dt (1/100 in tutorial, 1/200 default)") and picked 1/100 because it is what `WorkingWithDataObjs.m` and nearly every `Scripts/Figure*/*.m` uses. It also weighed the cost of 1000 vs 500 bins against decoder memory (steps 38, 41, 49: "check decoder memory handling to pick bin size given ~8000 trials x 500 bins"). The module docstring attributes `dt = 1/100` to "getDefaultParams.m / Scripts/Figure*/*.m", which is a mis-citation — `getDefaultParams.m` actually sets `params.dt = 1/200`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Nothing in the raw data beyond the alignment event itself: the input is the vector of bin centres of the −2.5 to 2.5 s window around each trial's go cue, i.e. the same grid the spikes are binned onto. It is the sole input (`input_names = ['time_from_go_cue']`), stored as a `(1, 500)` `float32` array, ranging from −2.495 to +2.495 s.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME = EDGES[:-1] + DT / 2          # bin centres, as in getSeq.m
NT = len(TIME)
...
inp = [np.asarray(TIME, dtype=np.float32).reshape(1, NT)] * len(trials)
...
data['input_names'] = ['time_from_go_cue']
```

iii. Dictated by the task instructions ("Time from go cue onset in seconds (continuous, time-varying)"); the window and grid follow `params.tmin`/`params.tmax`/`params.dt` and `getSeq.m`'s `obj.time = edges + params.dt/2`.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. None. The array is constructed once at module level and the identical object is reused for every trial of every session (`[TIME_array] * n_trials`), because the window is the same for all trials.

ii.
```python
inp = [np.asarray(TIME, dtype=np.float32).reshape(1, NT)] * len(trials)
```

iii. N/A — no raw variable is processed. (Note: because the list holds the same object repeatedly, the input costs essentially nothing in the pickle.)

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It *is* the neural binning grid. Spike times are expressed relative to the go cue and assigned to bin `floor((t − TMIN)/DT)` of `EDGES`, and the input is `EDGES[:-1] + DT/2`, the centres of those same bins. So input bin *k* and neural bin *k* are the same 10 ms interval, by construction, for every trial.

ii.
```python
bi = np.floor((aligned - TMIN) / DT).astype(int)
```
```python
TIME = EDGES[:-1] + DT / 2          # bin centres, as in getSeq.m
```

iii. N/A.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial flags of `obj.bp`: `hit`, `miss`, `R` and `L`. The direction actually licked is not stored, so it is inferred from the instructed/rewarded side (`R`/`L`) together with whether the animal was correct (`hit`) or not (`miss`). Trials that are neither hits nor misses (ignores) form the third class.

ii.
```python
hit = s.bp_flag('hit')
miss = s.bp_flag('miss')
no = s.bp_flag('no')
R = s.bp_flag('R')
L = s.bp_flag('L')
```

iii. Step 52–53: "Lick-direction from bp flags is consistent with lick times" — the AI wrote a verification script comparing the direction derived from `hit`/`miss`/`R`/`L` against the first lick after the go cue in `bp.ev.lickL`/`lickR` (the logic of the authors' `firstLickTime.m`, which it had just read in step 27) and confirmed they agree, then used the flags rather than the lick times.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A four-way relabelling on the kept trials: hit & R → right (1), hit & L → left (0), miss & R → left (0, the animal licked the wrong port), miss & L → right (1); everything else keeps the default, none (2). The per-trial value is then broadcast across all 500 time bins. `output_values[0] = ['left', 'right', 'none']`.

ii.
```python
# lick direction: 0 left, 1 right, 2 none (no response / ignore trial)
lick_dir = np.full(len(trials), 2, dtype=np.int64)
lick_dir[hit[trials] & R[trials]] = 1
lick_dir[hit[trials] & L[trials]] = 0
lick_dir[miss[trials] & R[trials]] = 0     # error trial: licked other way
lick_dir[miss[trials] & L[trials]] = 1
...
o[0] = lick_dir[k]
```

iii. As in 4-a: the flags encode the rewarded side and the correctness, so the licked side is their combination, and the AI cross-validated this against the recorded lick times. The third class exists because the instructions require a `none` category and ignore trials are retained.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. A single per-trial flag, `obj.bp.autowater`, which marks the water-cued (WC) block trials where water is delivered from a random port with no auditory cues; all other trials are delayed-response (DR).

ii.
```python
autowater = s.bp_flag('autowater')   # 1 in water-cued (WC) blocks
```

iii. Step 29: "autowater count in JEB13 session is small (22) - need to survey all sessions to identify two-context sessions"; step 50's survey printed `aw=` counts per session and confirmed that WC trials have valid go-cue (water-drop) times. Sessions with `aw = 0` (pure DR sessions) are kept, so the context output is constant in some sessions.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling: `autowater` → 0 (WC), otherwise 1 (DR), broadcast over the 500 bins. `output_values[1] = ['WC', 'DR']`.

ii.
```python
# context: 0 WC (autowater block), 1 DR
context = np.where(autowater[trials], 0, 1).astype(np.int64)
...
o[1] = context[k]
```

iii. Code order follows the instructions' listing ("WC, DR"). Overall 16.7% of the kept trials are WC.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Two per-trial flags of `obj.bp`: `hit` (rewarded) and `no` (ignore / no response). `miss` is not read for this variable — a trial that is neither a hit nor an ignore is an incorrect lick by construction.

ii.
```python
hit = s.bp_flag('hit')
no = s.bp_flag('no')
```

iii. Step 27: the AI read the authors' `getOutcome.m`, which is `outcome = bp.hit; outcome(logical(bp.no)) = nan` — i.e. the authors themselves define outcome from `hit` with `no` marking ignores. The AI kept ignores as a third class instead of `NaN` because the instructions require an `ignore` category.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A three-way relabelling on kept trials: start everything at 0 (incorrect), set hits to 1 (correct), set no-response trials to 2 (ignore); broadcast over the 500 bins. `output_values[2] = ['incorrect', 'correct', 'ignore']`. Because ignore trials are retained rather than dropped (unlike in the paper's analyses), the resulting distribution is 13.4% incorrect / 69.4% correct / 17.2% ignore.

ii.
```python
# outcome: 0 incorrect, 1 correct, 2 ignore
outcome = np.full(len(trials), 0, dtype=np.int64)
outcome[hit[trials]] = 1
outcome[no[trials]] = 2
...
o[2] = outcome[k]
```

iii. Class codes follow the instructions (incorrect 0, correct 1, ignore 2). The module docstring records why ignores are retained: "Ignore (no-response) trials are kept because 'ignore' is one of the outcome classes to decode."

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`, specifically **view 0 (side camera) feature `tongue`** — its `ts[:, 0:2, featidx]` (x, y; the confidence channel is not read) and the trial's `frameTimes`. `obj.traj{1}.NdroppedFrames` gates whether a trial's video is usable, and `obj.bp.ev.goCue` plus `obj.sglx.bitcode.bitstart`/`obj.sglx.fs`/`obj.bp.ev.bitStart` supply the alignment. The bottom camera's tongue markers (`top_tongue`, etc.) are not used.

ii.
```python
feats0 = s.traj_featnames(0)     # side cam
feats1 = s.traj_featnames(1)     # bottom cam
tongue_ix = feats0.index('tongue')
paw_ix = feats1.index('top_paw')
ndrop = s.ndropped(0)
...
ft, xy = s.traj_feat_trial(0, tr, tongue_ix)
```
```python
# sessionio.py
def traj_feat_trial(self, view, trial, featidx):
    """(frameTimes, xy) for one DLC feature; xy shape (nframes,2)"""
    ...
    xy = np.array(ds[featidx, 0:2, :]).astype(float).T
```

iii. Step 32: "traj structure understood: view0 side cam (tongue, jaw, nose...), view1 bottom cam (tongues, paws...). ts = (frames, [x,y,conf], feature) with frameTimes at 400Hz." Step 44–45: "DLC NaNs indicate feature not visible (tongue ~92% NaN, paws ~35-47% NaN). This maps directly to the 'not visible' class" — i.e. the AI relies on the authors having already NaN'd untracked coordinates rather than re-applying a likelihood threshold. Step 57 records the final choice: "DLC features (side-cam 'tongue', bottom-cam 'top_paw')".

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Four steps, following `findPosition.m` + `findVelocity.m`. (1) Trials whose `NdroppedFrames` is NaN, or whose `frameTimes` are missing/all-NaN, are marked as having no video and left entirely NaN. (2) The raw x and y traces are linearly interpolated (`interp1d`, NaN outside the frame range and NaN wherever a neighbouring frame is untracked) from the 400 Hz frame times onto the 500-bin, 10 ms neural axis — **no smoothing**, matching `findPosition.m`'s `if ~contains(feat,'tongue')` guard. (3) Velocity is a NaN-robust difference on that grid: central where both neighbours are tracked, one-sided at the edges of a visibility bout, 0 for an isolated tracked sample, NaN where untracked; speed is `hypot(vx, vy)`. No baseline-derivative subtraction (also per `findVelocity.m`, which excludes the tongue). (4) No cross-camera normalisation, since only one view is used. Velocity is in pixels per bin, which is irrelevant because only a percentile split is taken.

ii.
```python
def nan_gradient(pos):
    """Per-sample velocity of a (T,2) position trace that contains NaNs. ..."""
    ...
    both = np.isfinite(fwd) & np.isfinite(bwd)
    vel[both] = 0.5 * (fwd[both] + bwd[both])
    only_f = np.isfinite(fwd) & ~np.isfinite(bwd)
    vel[only_f] = fwd[only_f]
    only_b = np.isfinite(bwd) & ~np.isfinite(fwd)
    vel[only_b] = bwd[only_b]
    # tracked sample with no tracked neighbour: no velocity estimate, call it 0
    isolated = np.isfinite(pos) & ~np.isfinite(vel)
    vel[isolated] = 0.0
    return vel
```
```python
for k, tr in enumerate(trials):
    if ndrop is not None and not np.isfinite(ndrop[tr]):
        continue                                        # bad video trial
    taxis = TIME + gocue[tr] + vidshift                 # into video clock
    ft, xy = s.traj_feat_trial(0, tr, tongue_ix)
    if ft is None or xy is None or not np.isfinite(ft).any():
        continue
    has_video[k] = True
    pos = interp_to_axis(ft, xy, taxis)
    vel = nan_gradient(pos)
    tongue_speed[k] = np.hypot(vel[:, 0], vel[:, 1])
```

iii. Step 74 explains the one deliberate departure from plain `np.gradient`: "for tongue/paw speed I currently use np.gradient, which propagates NaNs and erases the edges of each visibility bout, over-assigning the 'not visible' class. Visibility should be defined by whether DLC tracked the marker at that timepoint. I'll use a NaN-robust difference (central where possible, one-sided at bout edges)." Step 77 reports the effect: "Tongue visible fraction increased from 4.9% to 7.9%, paw visible from 79.8% to 81.1%, as intended." The authors' own rule of setting untracked tongue velocity to 0 (`findVelocity.m`) is deliberately not followed, because the task specification requires a separate `not visible` class.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. One threshold per session, the 50th percentile of the speed over all finite (i.e. tongue-visible) bins of all kept trials in that session: below → 0, at or above → 1, non-finite (tongue not tracked, or the trial has no video) → 2. `output_values[3] = ['below_median', 'above_median', 'not_visible']`. Result: 3.93% / 3.93% / 92.13% of bins.

ii.
```python
def discretise(x):
    """0: < session median, 1: >= session median, 2: not visible / no video."""
    out = np.full(x.shape, 2, dtype=np.int64)
    vis = np.isfinite(x)
    if vis.any():
        thr = np.percentile(x[vis], 50)
        out[vis] = (x[vis] >= thr).astype(np.int64)
    return out

tongue_d = discretise(tongue_speed)
```

iii. Directly specified by the instructions ("discretized with per-session threshold: 0: < 50th percentile, 1: >= 50th percentile, 2: not visible"). The mapping of DLC NaN onto class 2 comes from step 45: "DLC NaNs indicate feature not visible ... This maps directly to the 'not visible' class."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The camera clock leads the behaviour clock, so a per-session offset `vidshift = mode(sglx.bitcode.bitstart)/sglx.fs − mode(bp.ev.bitStart)` is computed once (`findVideoOffset.m`). Rather than shifting the frame times, the AI shifts the target axis: it evaluates the DLC trace at `TIME + goCue[trial] + vidshift` in raw video-clock seconds, which is algebraically identical to interpolating `frameTimes − vidshift − goCue[trial]` onto `TIME`. The result therefore lands exactly on the neural 10 ms bin centres.

ii.
```python
# sessionio.py
def video_offset(self):
    """vidshift = mode(sglx.bitcode.bitstart)/fs - mode(bp.ev.bitStart)"""
    bs = vec(self.f, self.o['sglx']['bitcode']['bitstart'])
    fs = vec(self.f, self.o['sglx']['fs'])[0]
    bitStart = self.ev('bitStart')
    m1 = stats.mode(bs, keepdims=False).mode / fs
    m2 = stats.mode(bitStart, keepdims=False).mode
    return float(m1 - m2)
```
```python
vidshift = s.video_offset()
...
taxis = TIME + gocue[tr] + vidshift                 # into video clock
pos = interp_to_axis(ft, xy, taxis)
```

iii. Module docstring: "video: DLC trajectories are aligned with `frameTimes - vidshift - alignTime`, `vidshift = mode(sglx.bitcode.bitstart)/fs - mode(bp.ev.bitStart)` (findVideoOffset.m, findPosition.m)". Read in steps 9–12 and confirmed in step 55.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The same `obj.traj` DLC tracking, but **view 1 (bottom camera) feature `top_paw`** — x, y and the trial's `frameTimes`. The other forepaw marker (`bottom_paw`) is not used. The same `vidshift` and `goCue` supply the alignment.

ii.
```python
feats1 = s.traj_featnames(1)     # bottom cam
paw_ix = feats1.index('top_paw')
...
ft1, xy1 = s.traj_feat_trial(1, tr, paw_ix)
```

iii. Step 45: the AI grepped the figure scripts for which kinematic features the authors use and found `Figure1e.m`'s `feat2use = {'jaw_yvel_view1','nose_yvel_view1', 'top_paw_yvel_view2'}` — i.e. `top_paw` on view 2 (the bottom camera) is the paw feature the paper itself plots. Step 50's survey confirmed `toppaw1=True` (the feature exists on view 1) for all 25 sessions.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The same pipeline as the tongue — NdroppedFrames/frameTimes gate, linear interpolation of x and y onto the 500-bin 10 ms axis, NaN-robust difference, speed = `hypot` — plus one extra step taken from `findVelocity.m`: for non-tongue features the baseline (median frame-to-frame displacement over the trial) is subtracted from the velocity before taking the magnitude, removing slow drift. The authors' `fillmissing(...,'nearest')` of untracked paw positions is deliberately **not** applied, so untracked bins stay NaN and become the `not visible` class. Note the paw is only computed for trials where the tongue's frame times were also valid (the loop `continue`s before reaching the paw otherwise).

ii.
```python
ft1, xy1 = s.traj_feat_trial(1, tr, paw_ix)
if ft1 is None or xy1 is None or not np.isfinite(ft1).any():
    continue
pos1 = interp_to_axis(ft1, xy1, taxis)
vel1 = nan_gradient(pos1)
# findVelocity.m subtracts the baseline (median) frame-to-frame
# displacement for all non-tongue features, removing slow drift
base = np.nanmedian(np.diff(pos1, axis=0), axis=0)
vel1 = vel1 - base[None, :]
paw_speed[k] = np.hypot(vel1[:, 0], vel1[:, 1])
```

iii. Step 68: "One refinement to better match the reference code: findVelocity.m subtracts the per-trial baseline derivative (median frame-to-frame displacement) for non-tongue features such as the paw. I'll add that." The AI re-ran the conversion and retrained to confirm the change was stable (steps 69–72). (`findVelocity.m` subtracts `basederiv(1)` from *both* axes, apparently a bug; the AI subtracts each axis's own baseline.)

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Identically to the tongue: the same `discretise` helper, a per-session 50th-percentile split over all finite bins of the session, with class 2 for non-finite (untracked paw, or trial with no video). `output_values[4] = ['below_median', 'above_median', 'not_visible']`. Result: 40.5% / 40.5% / 18.9%.

ii.
```python
paw_d = discretise(paw_speed)
```

iii. Specified by the instructions; the `not visible` class is populated by DLC's NaNs, which the AI measured at ~35–47% of raw frames for the paws (step 45) and ~19% of bins after the NaN-robust velocity change (step 77).

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Exactly as for the tongue, and inside the same per-trial loop: the paw's own (bottom-camera) `frameTimes` are interpolated onto `TIME + goCue[trial] + vidshift`, i.e. the neural 10 ms bin centres, using the single session-level `vidshift`. The two cameras' frame times are read separately, so a view-specific frame count is handled naturally.

ii.
```python
taxis = TIME + gocue[tr] + vidshift                 # into video clock
...
ft1, xy1 = s.traj_feat_trial(1, tr, paw_ix)
...
pos1 = interp_to_axis(ft1, xy1, taxis)
```

iii. Same source as 7-d (`findVideoOffset.m` / `findPosition.m`); no separate treatment is needed for the paw.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The standalone `motionEnergy_<anm>_<date>.mat` file next to the data structure, which holds one trace per trial at camera frame resolution (plus a `moveThresh` that the AI does not use). The wrapper struct is unwrapped repeatedly to cope with the several layouts in the files. The side camera's `frameTimes` (read via the tongue feature) supply the time base. `obj.me`, present in some sessions, is not used.

ii.
```python
mepath = os.path.join(DATA_DIR, 'motionEnergy_%s_%s.mat' % (anm, date))
if os.path.exists(mepath):
    medat = sio.loadmat(mepath)['me']['data'][0, 0]
    # loadMotionEnergy.m: 'if isstruct(me.data), me.data = me.data.data'
    while medat.dtype.names is not None and 'data' in medat.dtype.names:
        medat = medat['data'][0, 0]
    medat = medat.ravel()
```

iii. Steps 33–34: "motionEnergy files are v7 (scipy) ... Motion energy: per-trial arrays at video frame rate with a session moveThresh." Step 50 confirmed `me=True` (a motion-energy file exists) for all 25 sessions. Step 61 discovered the nested layout during the first full run — "Some motionEnergy files have me.data as a nested struct (handled in loadMotionEnergy.m by `if isstruct(me.data) me.data = me.data.data`). Need to handle that." — and patched it into a `while` loop.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. No re-derivation — the value is already one number per video frame. It is linearly interpolated from the side camera's `frameTimes` onto the 500-bin neural axis, and the resulting NaNs (bins outside the frame coverage) are filled, following `loadMotionEnergy.m`'s `fillmissing(me.data,'nearest')`. A trial is left entirely NaN (→ class 2, "no video") if it had no usable video, if the trial index exceeds the motion-energy array, or if the frame count does not match the DLC frame times. The session's `moveThresh` is ignored in favour of the instructed 50th-percentile split.

ii.
```python
for k, tr in enumerate(trials):
    if not has_video[k] or tr >= len(medat):
        continue
    m = np.asarray(medat[tr], float).ravel()
    ft, _ = s.traj_feat_trial(0, tr, tongue_ix)
    if ft is None or len(ft) != len(m):
        continue
    taxis = TIME + gocue[tr] + vidshift
    vals = interp_to_axis(ft, m, taxis)
    me_arr[k] = fill_nearest(vals)   # loadMotionEnergy.m fills nans
```
```python
def fill_nearest(y):
    """fillmissing(y,'nearest') for a 1-d array."""
    y = np.asarray(y, float)
    good = np.isfinite(y)
    if not good.any():
        return y
    idx = np.arange(len(y))
    return np.interp(idx, idx[good], y[good])
```

iii. Module docstring: "motion energy is interpolated onto the same time axis (loadMotionEnergy.m)". The AI read `loadMotionEnergy.m` in step 9 and reproduced its two operations (interp1 onto the aligned axis, then fill the NaNs). Note that `fill_nearest` actually performs *linear* interpolation between valid samples rather than nearest-neighbour fill; for the edge NaNs that dominate here ("there are some nans at the start of each trial" per the MATLAB comment) `np.interp` holds the nearest value, so the two agree, but they differ for interior gaps.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The same `discretise` helper: a per-session 50th-percentile split over all finite bins, with class 2 where the whole trial is NaN. `output_values[5] = ['below_median', 'above_median', 'no_video']`. Because the NaN fill makes almost every video trial fully defined, class 2 is very rare (0.013% of bins; the split is 49.96% / 50.03% / 0.013%). The `moveThresh` stored with the motion-energy files is not used.

ii.
```python
me_d = discretise(me_arr)
```

iii. Specified by the instructions.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same offset and same grid as the DLC streams. Motion energy has one value per side-camera frame, so it is interpolated from the side camera's `frameTimes` (re-read through `traj_feat_trial(0, tr, tongue_ix)`) onto `TIME + goCue[trial] + vidshift`. The explicit `len(ft) != len(m)` check guards against a frame-count mismatch between the motion-energy trace and the frame times.

ii.
```python
ft, _ = s.traj_feat_trial(0, tr, tongue_ix)
if ft is None or len(ft) != len(m):
    continue
taxis = TIME + gocue[tr] + vidshift
vals = interp_to_axis(ft, m, taxis)
```

iii. `loadMotionEnergy.m` uses `obj.traj{1}(trix).frameTimes-vidshift-alignTimes(trix)` — i.e. the side camera — which is what the AI reproduces.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Everything is handled by keeping the trial/session and marking the gap; nothing is dropped for bad video and nothing is invented for missing neural data. Specifically: (a) trials whose go cue is not finite are removed; (b) trials whose `NdroppedFrames` is NaN, or whose `frameTimes` are absent or all-NaN, are marked "no video" and come out as 500 bins of class 2 for tongue, paw and motion energy; (c) untracked DLC samples (already NaN in the file) propagate to class 2, and the custom `nan_gradient` keeps the first/last sample of each visibility bout instead of letting `np.gradient` erase them; (d) a paw trace missing while the tongue is present leaves the paw at class 2 only; (e) motion energy that is absent, short, or whose frame count disagrees with the frame times leaves that trial at class 2, and interior/edge NaNs after interpolation are filled; (f) the three different `motionEnergy` file layouts are unwrapped in a loop; (g) `obj.ex.probe.loc` missing (EKH1, EKH3) falls back to `ALM`; (h) a probe with no usable clusters is skipped, and a session with <10 units or <2 trials is dropped.

ii.
```python
keep = (~early) & (~stim) & np.isfinite(gocue)
...
if ndrop is not None and not np.isfinite(ndrop[tr]):
    continue                                        # bad video trial
...
if ft is None or xy is None or not np.isfinite(ft).any():
    continue
```
```python
def region_from_loc(loc):
    """Map obj.ex.probe.loc strings onto brain region names."""
    if loc is None:
        return 'ALM'
```
```python
    isolated = np.isfinite(pos) & ~np.isfinite(vel)
    vel[isolated] = 0.0
```

iii. Most of these were found empirically and patched: the nested motion-energy struct in step 61, the probe-location char-array case in step 55 ("First fix probe_locs reading for char arrays"), and the NaN-propagation issue in step 74 ("np.gradient ... erases the edges of each visibility bout, over-assigning the 'not visible' class"). The `NdroppedFrames` check is copied from `findPosition.m`. The overall philosophy — keep the trial, mark the gap — follows from the instruction that a `not visible` class exists.

## 11-a. What are the most time-consuming steps of the code?

i. Reading the files dominates: each session is a multi-hundred-MB HDF5 file, and the code performs one h5py dereference per cluster (`clu_spikes`) and two per trial per feature (`traj_feat_trial`, called three times per trial — tongue, paw, and again for motion energy). Within the computation, the per-cluster spike-binning loop with `np.add.at` is the next largest cost (`np.add.at` is an unbuffered scatter-add and is slow), followed by constructing one `scipy.interpolate.interp1d` object per trial per feature. Writing the 1.1 GB pickle is also non-trivial. The full 25-session conversion took a few minutes end to end (it was run in the background and polled).

ii.
```python
for ii, ci in enumerate(cluix):
    tr, ttm = s.clu_spikes(prb, ci)
    ...
    np.add.at(counts[ii], (ti[ok], bi[ok]), 1.0)
```
```python
def interp_to_axis(src_t, src_y, dst_t):
    f = interp1d(src_t, src_y, kind='linear', bounds_error=False,
                 fill_value=np.nan, axis=0, assume_sorted=False)
    return f(dst_t)
```

iii. The AI never profiled, but it did act on load cost: step 47 — "I need a fast HDF5 loader that reads only needed fields (generic recursive conversion of traj is slow)" — which is why `sessionio.py` exposes targeted readers instead of materialising the whole `obj`. It also ran the conversion with `nohup`/logging and polled (steps 59–63), suggesting it expected a multi-minute runtime.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Two. (1) The per-cluster spike loop: all clusters of a probe could be binned in a single pass (e.g. one `np.histogramdd`/`histogram2d` over a concatenated (cluster, trial, time) index set, or at minimum `np.bincount` on a flattened linear index instead of `np.add.at`, which is the slowest available scatter-add). (2) The motion-energy loop re-derives `taxis` and re-reads the frame times per trial; the frame times could be hoisted out of the kinematics loop and reused. The per-trial video loops themselves cannot be vectorised because each trial has a different number of camera frames. What *is* well vectorised: the Gaussian smoothing, which is applied once to the whole `(units, trials, time)` array rather than per cluster, and all six discretisations, which are whole-array operations.

ii.
```python
counts = np.zeros((len(cluix), len(trials), NT), dtype=np.float32)
for ii, ci in enumerate(cluix):
    ...
    np.add.at(counts[ii], (ti[ok], bi[ok]), 1.0)
rate = smooth_causal(counts / DT)                 # vectorized over units x trials
```

iii. Not discussed in the trajectory; the AI's stated performance concern was loading, not binning.

## 11-c. What processing does the code repeat multiple times?

i. Two clear repetitions. (1) The side camera's `frameTimes` **and** the tongue x/y are read from HDF5 twice per trial — once in the kinematics loop and again in the motion-energy loop — even though only `ft` is needed the second time. (2) `taxis = TIME + gocue[tr] + vidshift` is recomputed in both loops for every trial. Beyond that, nothing is recomputed: `vidshift` is computed once per session, feature-name lookups and `NdroppedFrames` are read once per session, the bin grid and smoothing kernel are built once at module level, and each session file is opened once.

ii.
```python
# kinematics loop
taxis = TIME + gocue[tr] + vidshift
ft, xy = s.traj_feat_trial(0, tr, tongue_ix)
...
# motion energy loop, same trial
ft, _ = s.traj_feat_trial(0, tr, tongue_ix)
...
taxis = TIME + gocue[tr] + vidshift
```
```python
vidshift = s.video_offset()          # once per session
KERN = causal_gaussian_kernel()      # once per module
```

iii. Not discussed; the motion-energy block was written as a separate section after the kinematics block and was patched twice (steps 61, 68) without being merged into the main trial loop.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Modest amounts. (1) In the motion-energy loop `traj_feat_trial` returns and therefore reads the tongue x/y array from disk, which is immediately discarded (`ft, _ = ...`). (2) The paw's velocity is computed for all trials, including the ~88% of bins where downstream only the median split matters — unavoidable. (3) `from scipy import stats` is imported in `convert_data.py` but never used there. (4) Outputs are stored as `int64` when the six values are all in {0,1,2}; an `int8` array would be 8× smaller (~178 MB → ~22 MB of the 1.1 GB file), and each per-trial scalar (lick direction, context, outcome) is materialised 500 times. (5) `region_from_loc` is evaluated for every non-garbage cluster before the firing-rate filter removes most of them. (6) Smoothing is applied to all clusters before the ≤1 Hz filter drops them. Nothing else computed is thrown away — every stream that is computed ends up in the output.

ii.
```python
ft, _ = s.traj_feat_trial(0, tr, tongue_ix)   # xy read from disk and discarded
```
```python
from scipy import stats                        # unused in convert_data.py
```
```python
o = np.empty((6, NT), dtype=np.int64)          # int8 would suffice
```
```python
rate = smooth_causal(counts / DT)
fr_list.append(rate)
...
meanfr = fr.mean(axis=(1, 2))
use = meanfr > LOW_FR                          # filter applied after smoothing
```

iii. Not discussed in the trajectory. The AI did clean up its scratch exploration scripts at the end (step 80) and kept `sessionio.py` as a required dependency of `convert_data.py`.
