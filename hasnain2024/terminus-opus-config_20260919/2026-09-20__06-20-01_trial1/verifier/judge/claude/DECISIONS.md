# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each session is one MATLAB file `data_structure_<ANM>_<DATE>.mat`, found in either `Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior`. The 44 sessions and their probe numbers are hard-coded in `FIXED` and `RANDOM` lists, transcribed from the authors' loading scripts. A `Session` class handles both v7 and v7.3 MAT files. Motion energy is loaded separately from `motionEnergy_<ANM>_<DATE>.mat`.

ii.
```python
FIXED = [
    ('Ephys_Behavior', 'JEB6',  '2021-04-18', [2]),
    ...
]
RANDOM = [
    ('RandomizedDelay_Ephys_Behavior', 'JEB11', '2022-05-10', [1]),
    ...
]
SESSIONS = FIXED + RANDOM

class Session:
    def __init__(self, fn):
        self.hdf5 = is_hdf5(fn)
        if self.hdf5:
            self.f = h5py.File(fn, 'r')
            self.obj = self.f['obj']
        else:
            self.obj = sio.loadmat(fn, struct_as_record=False, squeeze_me=True)['obj']
```

iii. The session list is transcribed from the authors' `load<ANM>_ALMVideo.m` scripts. The AI noted that commented-out entries and sessions without sorted units (JEB24 10-03/10-04) are excluded. Both MATLAB file formats are handled by checking the file header for v7.3.

## 1-b. How are the data split into subjects?

i. The animal name is the first element of each session tuple (e.g., `'JEB6'`). At assembly, subjects are collected in order of first appearance, and `subject_idx` maps each session to its subject.

ii.
```python
subjects = []
for r in results:
    if r['animal'] not in subjects:
        subjects.append(r['animal'])
    data['subject_idx'].append(subjects.index(r['animal']))
data['subjects'] = subjects
data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. The animal ID comes from the session metadata tuples, which were transcribed from the authors' loading scripts.

## 1-c. How are the data split into sessions?

i. Each entry in the `SESSIONS` list (FIXED + RANDOM) corresponds to one session file on disk. Each becomes one element in `neural`, `input`, and `output`. The result is 44 sessions (25 fixed-delay + 19 randomized-delay), though sessions with fewer than 10 units are skipped.

ii.
```python
sessions = SESSIONS
if args.sample:
    sessions = [FIXED[0], RANDOM[0]]
# ...
for r in results:
    if r['nunits'] < MIN_UNITS:
        print(f"  SKIP {r['session_id']}: only {r['nunits']} units (< {MIN_UNITS})")
        continue
```

iii. The AI applies a minimum of 10 units per session, citing the paper's criterion "sessions were included for analysis only if they had at least 10 units." In practice, all 44 sessions pass this criterion.

## 1-d. How are the data split into trials?

i. Trials are defined by `bp.Ntrials`. Per-trial fields are read from `obj.bp` using the `bpvec` and `ev` methods of the `Session` class. Spike times carry their trial number in `clu.trial`, so no trial boundary reconstruction is needed.

ii.
```python
ntrials = s.ntrials  # int(obj.bp.Ntrials)
gocue = s.ev(ALIGN_EVENT)
# ...
keep = (stim[:ntrials] == 0) & (early[:ntrials] == 0) & np.isfinite(gocue[:ntrials])
keep &= (hit[:ntrials] + miss[:ntrials] + no[:ntrials]) > 0
trials = np.where(keep)[0]
```

iii. The Bpod table directly defines trials, with one go cue per trial.

## 1-e. How are trials filtered based on quality controls?

i. Three filters are applied: (1) photostimulation trials (`bp.stim.enable`) are excluded, (2) early-lick trials (`bp.early`) are excluded, and (3) trials must have a finite go cue time and at least one of hit/miss/no set. Additionally, trials with no spikes from any curated unit are dropped (for sessions where the recording ended before behavior).

ii.
```python
stim = s.bpvec('stim/enable')
early = s.bpvec('early')
keep = (stim[:ntrials] == 0) & (early[:ntrials] == 0) & np.isfinite(gocue[:ntrials])
keep &= (hit[:ntrials] + miss[:ntrials] + no[:ntrials]) > 0
trials = np.where(keep)[0]
# ...
if nunits > 0:
    has_spikes = rates[trials].sum(axis=(1, 2)) > 0
    trials = trials[has_spikes]
```

iii. The AI documented in CONVERSION_NOTES.md that photostim and early-lick exclusion follows the reference condition strings. The finite goCue check and hit+miss+no check are additional robustness filters. The no-spikes filter handles sessions where ephys recording ended early.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` spike-sorted clusters. Each cluster has `trial` (1-based trial number), `trialtm` (spike time within trial), and `quality` (manual curation label). The go cue times `bp.ev.goCue` are used for alignment.

ii.
```python
def clusters(self, prb):
    # returns list of dicts with quality, trial, trialtm
    ...
    out.append(dict(quality=..., trial=..., trialtm=...))
```

iii. The same raw variables as the reference MATLAB pipeline (alignSpikes.m uses trialtm and goCue).

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue (`trialtm - goCue`), binned into 30ms bins from -2.5 to +2.5s (166 bins), converted to firing rate (counts/dt), and smoothed with a **causal** Gaussian kernel of 15 bins width using the reference's `mySmooth.m` approach (gausswin with first half zeroed, reflect boundary).

ii.
```python
DT = 0.03  # 30 ms bins
SMOOTH = 15
# ...
def causal_kernel(N=SMOOTH):
    k = gausswin(N)
    k[:N // 2] = 0.0
    return k / k.sum()

def mysmooth(x, N=SMOOTH, bctype='reflect'):
    # Port of utils/mySmooth.m; smooths along axis 0
    k = causal_kernel(N)
    out = fftconvolve(xf, k, mode='same', axes=0)
    ...

def bin_spikes(clusters, gocue, ntrials, edges):
    aligned = tt - gocue[(tr - 1).astype(int)]
    cnt, _, _ = np.histogram2d(tr, aligned, bins=[trial_edges, edges])
    meanfr[i] = cnt.sum() / (ntrials * window)
    rates[:, :, i] = mysmooth(cnt.T / DT).T.astype(np.float32)
```

iii. The AI chose 30ms bins citing `params.dt = (1/100)*3` from the Figure 3 scripts. The causal Gaussian smoothing is a faithful port of the reference's `mySmooth.m`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) cluster quality labels are checked case-insensitively against `BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?', ''}`, excluding clusters with those labels, and (2) units with mean firing rate <= 1 Hz are dropped.

ii.
```python
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?', ''}
LOW_FR = 1.0
# ...
for c in s.clusters(p - 1):
    q = c['quality'].strip().lower()
    if q in BAD_QUALITY:
        continue
# ...
keep_units = meanfr > LOW_FR
rates = rates[:, :, keep_units]
```

iii. The AI cited `findClusters.m` with `quality='all'` and the paper's "units with firing rates exceeding 1 Hz." The empty string `''` is excluded instead of `'poor'` (which the reference solution excludes). The AI noted in CONVERSION_NOTES that about 206 clusters have empty quality labels.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to the go cue is done by subtracting `goCue[trial]` from `trialtm` for each spike, identical to the reference's `alignSpikes.m`.

ii.
```python
aligned = tt - gocue[(tr - 1).astype(int)]     # trialtm_aligned
cnt, _, _ = np.histogram2d(tr, aligned, bins=[trial_edges, edges])
```

iii. This matches `alignSpikes.m`: `clu.trialtm_aligned = clu.trialtm - obj.bp.ev.(alignEvent)(clu.trial)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 30ms bins, giving 166 time bins spanning -2.5 to +2.5s. The time axis is computed as `edges + dt/2` with the last dropped, matching `getSeq.m`. No rebinning is applied after the initial binning.

ii.
```python
DT = 0.03  # params.dt = (1/100)*3  (30 ms bins)
TMIN = -2.5
TMAX = 2.5

def time_axis():
    nedges = int(np.floor((TMAX - TMIN) / DT)) + 1
    edges = TMIN + DT * np.arange(nedges)
    t = edges + DT / 2.0
    return edges, t[:-1]
```

iii. The AI chose 30ms citing `params.dt = (1/100)*3` from Figure 3 scripts, noting that other scripts use 5-10ms and decoding re-bins to 75ms. The reference solution uses 5ms (1000 bins).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The time axis is constructed from the bin edges, not from any raw data variable. It is the bin centers of the -2.5 to +2.5s window.

ii.
```python
def time_axis():
    nedges = int(np.floor((TMAX - TMIN) / DT)) + 1
    edges = TMIN + DT * np.arange(nedges)
    t = edges + DT / 2.0
    return edges, t[:-1]
# ...
tin = taxis.astype(np.float32).reshape(1, -1)
inp.append(tin.copy())
```

iii. The time axis is defined by the binning parameters, identical in concept to the reference.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing beyond constructing the bin center times. The same time axis is shared by all trials and sessions.

ii. See 3-a.

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is the bin center times of the same binning grid used for neural data, so they are inherently aligned.

ii.
```python
tin = taxis.astype(np.float32).reshape(1, -1)
# Same taxis used for edges in bin_spikes
```

iii. N/A

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derives lick direction from `bp.ev.lickL` and `bp.ev.lickR` (cell arrays of lick contact times), finding the first lick after the go cue.

ii.
```python
lickL, lickR = s.ev_cell('lickL'), s.ev_cell('lickR')
lick_dir = np.full(ntrials, 2, dtype=np.int64)  # 2 = none
for i in range(ntrials):
    tl = np.concatenate([lickL[i] - gocue[i], lickR[i] - gocue[i]])
    side = np.concatenate([np.zeros(lickL[i].size), np.ones(lickR[i].size)])
    m = tl > 0
    if m.any():
        lick_dir[i] = int(side[m][np.argmin(tl[m])])  # 0=left, 1=right
```

iii. The AI cited `firstLickTime.m` and verified that the first post-go-cue lick side agrees 100% with the hit/miss + R derivation on all tested sessions.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. For each trial, all lick times (left and right) are aligned to the go cue. The earliest post-go-cue lick determines the direction (0=left, 1=right). If no lick occurs after the go cue, the direction is 2 (none). This is broadcast as a per-trial constant over all time bins.

ii. See 4-a code.

iii. The AI verified this approach agrees with the hit/miss derivation used by the reference solution.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `obj.bp.autowater`, where autowater > 0 indicates water-cued (WC) context, otherwise delayed-response (DR).

ii.
```python
autowater = s.bpvec('autowater')
context = np.where(autowater[:ntrials] > 0, 0, 1).astype(np.int64)  # 0=WC, 1=DR
```

iii. Directly from the trial table, matching the reference condition strings.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct mapping: autowater > 0 -> WC (0), else DR (1). Broadcast over all time bins as a per-trial constant.

ii. See 5-a.

iii. Matches the reference.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `bp.hit` and `bp.miss` flags, with trials that are neither hit nor miss classified as ignore.

ii.
```python
hit = s.bpvec('hit')
miss = s.bpvec('miss')
outcome = np.full(ntrials, 2, dtype=np.int64)  # 2 = ignore
outcome[miss[:ntrials] > 0] = 0                # 0 = incorrect
outcome[hit[:ntrials] > 0] = 1                 # 1 = correct
```

iii. Same derivation as the reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct mapping: miss -> incorrect (0), hit -> correct (1), else ignore (2). Broadcast over time bins.

ii. See 6-a.

iii. Matches the reference.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Only the **side camera** (`obj.traj[0]`) tongue feature (`tongue`), using its x,y coordinates from `ts` and `frameTimes` for timing. `bp.ev.goCue` and `sglx.bitcode.bitstart` are used for alignment.

ii.
```python
fn0 = s.feat_names(0)  # side camera
i_tongue = fn0.index('tongue')
# ...
ts0, ft0, nd0 = s.traj_trial(0, i)  # side camera only
xy = ts0[:, :2, i_tongue]
spd = speed_from_xy(xy) / dtf
```

iii. The AI uses only the side camera for tongue velocity, unlike the reference which uses both side and bottom cameras.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The x,y position of the tongue is taken from frames where the feature is visible (non-NaN). Speed is computed using `np.gradient` on x and y within contiguous runs of visible frames, then combined as `sqrt(vx^2 + vy^2)`. The speed is divided by the median frame interval to get pixels/second. Frames are then binned by averaging into the time bins. The result is discretized at the session 50th percentile.

ii.
```python
def speed_from_xy(xy):
    vis = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
    for a, b in zip(starts, stops):
        seg = idx[a:b + 1]
        vx = np.gradient(xy[seg, 0])
        vy = np.gradient(xy[seg, 1])
        spd[seg] = np.sqrt(vx ** 2 + vy ** 2)
    return spd
# ...
spd = speed_from_xy(xy) / dtf  # pixels/s
_accumulate(idx, inb, spd, vis, tongue_spd, tongue_vis, i, nbins)
```

iii. Unlike the reference which smooths x,y with a Gaussian before differentiating, the AI computes raw gradients. The AI divides by the median frame interval uniformly rather than using per-frame intervals via `np.gradient(x, time)`.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The 50th percentile of visible tongue speed values across all kept trials in the session is computed. Values below the threshold get class 0, at/above get class 1, and not-visible bins get class 2.

ii.
```python
m_t = sel[:, None] & tongue_vis & np.isfinite(tongue_spd)
thr_t = np.nanpercentile(tongue_spd[m_t], 50) if m_t.any() else np.nan
tongue_cls = discretize(tongue_spd, tongue_vis, thr_t)

def discretize(values, visible, thresh):
    out = np.full(values.shape, 2, dtype=np.int64)
    v = visible & np.isfinite(values)
    out[v & (values < thresh)] = 0
    out[v & (values >= thresh)] = 1
    return out
```

iii. Matches the instruction specification of 50th percentile threshold per session.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Camera frame times are corrected by the video offset (from `findVideoOffset.m` logic) and aligned to the go cue. Frames are then digitized into the same bin edges as the neural data.

ii.
```python
vidshift = s.vidshift()
vt0 = ft0 - vidshift - gocue[i]
idx = np.digitize(vt0, bin_edges) - 1
```

iii. The video offset computation matches `findVideoOffset.m`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The **bottom camera** (`obj.traj[1]`) features `top_paw` and `bottom_paw` are both used, with their speeds averaged.

ii.
```python
fn1 = s.feat_names(1)  # bottom camera
i_paws = [fn1.index(f) for f in ('top_paw', 'bottom_paw') if f in fn1]
# ...
for j in i_paws:
    pxy = ts1[:, :2, j]
    spds.append(speed_from_xy(pxy) / dtf1)
spd_p = np.nanmean(np.vstack(spds), axis=0)
```

iii. The AI uses both paw features and averages them, while the reference uses only `top_paw`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same speed computation as tongue (`np.gradient` of x,y, then magnitude), divided by median frame interval. The speeds of both paws are averaged per frame (using whichever is visible). Binned into time bins and discretized at session median.

ii. See 8-a code.

iii. Unlike the reference, the AI uses two paw features rather than one.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Session 50th percentile of visible paw speed values across kept trials. Same discretization logic as tongue.

ii.
```python
m_p = sel[:, None] & paw_vis & np.isfinite(paw_spd)
thr_p = np.nanpercentile(paw_spd[m_p], 50) if m_p.any() else np.nan
paw_cls = discretize(paw_spd, paw_vis, thr_p)
```

iii. Matches instruction specification.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: bottom camera frame times corrected by video offset and aligned to go cue, then digitized into the same bin edges.

ii.
```python
vt1 = ft1 - vidshift - gocue[i]
idx1 = np.digitize(vt1, bin_edges) - 1
```

iii. Same offset and binning as neural data.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. `motionEnergy_<ANM>_<DATE>.mat`, loaded with a robust unpacker handling three file variants.

ii.
```python
def load_motion_energy(fn):
    me = sio.loadmat(fn)['me']
    def unpack(x):
        if isinstance(x, np.ndarray) and x.dtype.names is not None and 'data' in x.dtype.names:
            d = x['data']
            return unpack(d)
        return x
    data = np.asarray(unpack(me))
    trials = [np.asarray(data.ravel()[i], dtype=float).ravel() for i in range(data.size)]
    return trials, thresh[0]
```

iii. Handles the three file layout variants (bare cell, struct with data, nested struct).

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy values are interpolated to the time axis using linear interpolation (`interp_to_axis`) and then filled with nearest-neighbor values (`_fill_nearest`), matching the reference MATLAB code's `interp1` + `fillmissing('nearest')`. Then discretized at session median.

ii.
```python
y = interp_to_axis(vt0[:n], mev[:n], taxis)
y = _fill_nearest(y)
me_binned[i] = y
```

iii. This closely follows `loadMotionEnergy.m` which does `interp1(frameTimes, meData, taxis)` then `fillmissing('nearest')`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Session 50th percentile of finite motion energy values across kept trials. Same discretization as other movement signals, with class 2 meaning "no video."

ii.
```python
m_m = sel[:, None] & np.isfinite(me_binned)
thr_m = np.nanpercentile(me_binned[m_m], 50) if m_m.any() else np.nan
me_cls = discretize(me_binned, has_video[:, None] & np.isfinite(me_binned), thr_m)
```

iii. Matches instruction specification.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned using the side camera frame times corrected by the video offset and go cue, then interpolated to the same time axis as neural data.

ii.
```python
vt0 = ft0 - vidshift - gocue[i]
y = interp_to_axis(vt0[:n], mev[:n], taxis)
```

iii. Same alignment approach as other camera streams.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases handled: (1) Trials with NaN `NdroppedFrames` or empty frame times are skipped (`findPosition.m` does the same). (2) Trials with no video get class 2 for all movement outputs. (3) NaN go cue times cause trial exclusion. (4) Trials where ephys recording ended (no spikes) are dropped. (5) Motion energy uses nearest-neighbor filling for NaN bins.

ii.
```python
if ts0.size == 0 or ft0.size == 0 or np.all(~np.isfinite(ft0)) or not np.isfinite(nd0):
    continue  # findPosition.m skips these trials
# ...
if nunits > 0:
    has_spikes = rates[trials].sum(axis=(1, 2)) > 0
    trials = trials[has_spikes]
```

iii. The AI documented these edge cases in CONVERSION_NOTES.md and verified against the reference code's behavior.

## 11-a. What are the most time-consuming steps of the code?

i. The AI uses multiprocessing (`Pool`) to process sessions in parallel (default 8 workers). Loading the MAT files and binning spikes are the main costs. Total conversion ran in reasonable time for 44 sessions.

ii.
```python
nproc = min(args.nproc, len(sessions))
if nproc > 1:
    with Pool(nproc) as pool:
        results = pool.map(_worker, list(zip(sessions, want_debug)))
```

iii. The AI noted timing for each session and used parallel processing to speed things up.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-cluster loop in `bin_spikes` could potentially be vectorized but requires different trial/time counts per cluster. The per-trial video processing loop iterates over trials with variable frame counts. The `speed_from_xy` function loops over contiguous runs of visible frames.

ii.
```python
for i, c in enumerate(clusters):
    # per-cluster histogram2d
    cnt, _, _ = np.histogram2d(tr, aligned, bins=[trial_edges, edges])
    rates[:, :, i] = mysmooth(cnt.T / DT).T
# ...
for i in trials:
    # per-trial video processing
    ts0, ft0, nd0 = s.traj_trial(0, i)
```

iii. These loops are hard to vectorize due to variable-length data per cluster/trial.

## 11-c. What processing does the code repeat multiple times?

i. `s.traj_trial(0, i)` is called once per trial for the side camera, but the video offset, feature indices, and motion energy data are computed/loaded once per session. The `time_axis()` function is called multiple times but is cheap.

ii.
```python
vidshift = s.vidshift()  # computed once per session
fn0, fn1 = s.feat_names(0), s.feat_names(1)  # computed once
me_trials, _ = load_motion_energy(me_path(entry))  # loaded once
```

iii. Most quantities are computed once per session and reused.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `Session` class using `struct_as_record=False, squeeze_me=True` for v5 files loads the entire object structure. The `moveThresh` from motion energy files is loaded but not used. Debug information is optionally collected but only for `--show-processing`. Probe location strings are read for region mapping but most map to 'ALM'.

ii.
```python
self.obj = sio.loadmat(fn, struct_as_record=False, squeeze_me=True)['obj']
# loads entire obj structure including unused fields
# ...
me_trials, _ = load_motion_energy(me_path(entry))  # thresh not used
```

iii. Loading the full object is unavoidable since many fields are needed, but some (like spike waveforms) are never used.
