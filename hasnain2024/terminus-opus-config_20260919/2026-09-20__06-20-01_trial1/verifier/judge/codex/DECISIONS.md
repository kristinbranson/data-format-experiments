# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a 44-session manifest as `FIXED + RANDOM`, then loads each `data_structure_<animal>_<date>.mat` file through a custom `Session` wrapper that supports both MATLAB v7 and v7.3 layouts. Motion energy is loaded separately from the paired `motionEnergy_<animal>_<date>.mat` file.

ii. ```python
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
        self.fn = fn
        self.hdf5 = is_hdf5(fn)
        if self.hdf5:
            self.f = h5py.File(fn, 'r')
            self.obj = self.f['obj']
        else:
            self.obj = sio.loadmat(fn, struct_as_record=False, squeeze_me=True)['obj']
```
```python
def load_motion_energy(fn):
    me = sio.loadmat(fn)['me']
    ...
    return trials, thresh[0]
```

iii. `CONVERSION_NOTES.md` says the session/probe list was transcribed from the authors’ loading scripts, and the custom loaders were added because the shared files mix MATLAB formats and motion-energy file layouts.

## 1-b. How are the data split into subjects?

i. The AI treats the `anm` field from each session tuple as the subject id, preserves the first-seen order of animals in `subjects`, and stores one `subject_idx` per session.

ii. ```python
res = dict(
    session_id=sess_id, animal=anm, date=date, group=('fixed' if entry in FIXED else 'randomized'),
    neural=neural, input=inp, output=outp,
    ...
)
```
```python
subjects = []
for r in results:
    ...
    if r['animal'] not in subjects:
        subjects.append(r['animal'])
    ...
    data['subject_idx'].append(subjects.index(r['animal']))

data['subjects'] = subjects
data['subject_idx'] = np.array(data['subject_idx'], dtype=np.int64)
```

iii. The notes justify this by saying the session manifest already carries the animal id explicitly, so subject identity is taken from the manifest rather than from possibly inconsistent metadata inside the `.mat` files.

## 1-c. How are the data split into sessions?

i. Each tuple in `SESSIONS` is treated as one session. `process_session` converts one tuple into one session-level entry containing all trials for that recording.

ii. ```python
SESSIONS = FIXED + RANDOM
...
def process_session(entry, want_debug=False):
    d, anm, date, probes = entry
    sess_id = f'{anm}_{date}'
    s = Session(data_path(entry))
```
```python
for r in results:
    ...
    data['neural'].append(r['neural'])
    data['input'].append(r['input'])
    data['output'].append(r['output'])
```

iii. The notes say this mirrors the authors’ per-session loading scripts and intentionally excludes behavior-only MAH sessions and randomized-delay files that were not part of the reference session list.

## 1-d. How are the data split into trials?

i. The AI uses the Bpod trial table length `bp.Ntrials` as the master trial count. Per-trial fields such as `stim`, `early`, `hit`, `miss`, `no`, `autowater`, `goCue`, `lickL`, and `lickR` are indexed by trial number, and the kept trials are the integer indices returned by the trial mask.

ii. ```python
@property
def ntrials(self):
    if self.hdf5:
        return int(np.asarray(self.obj['bp/Ntrials'][()]).ravel()[0])
    return int(self.obj.bp.Ntrials)
```
```python
ntrials = s.ntrials
gocue = s.ev(ALIGN_EVENT)
...
keep = (stim[:ntrials] == 0) & (early[:ntrials] == 0) & np.isfinite(gocue[:ntrials])
keep &= (hit[:ntrials] + miss[:ntrials] + no[:ntrials]) > 0
trials = np.where(keep)[0]
```

iii. The notes justify this as following the Bpod structure directly rather than reconstructing trial boundaries from spikes or video.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in three stages. First, the session-level mask drops photostimulation trials, early-lick trials, trials with non-finite go-cue times, and trials that do not satisfy `hit | miss | no`. Second, after neural curation, trials with zero spikes across all curated units in the full window are removed as “no ephys coverage.” Third, sessions are skipped entirely if they end up with fewer than 10 curated units or fewer than 2 kept trials.

ii. ```python
keep = (stim[:ntrials] == 0) & (early[:ntrials] == 0) & np.isfinite(gocue[:ntrials])
keep &= (hit[:ntrials] + miss[:ntrials] + no[:ntrials]) > 0
trials = np.where(keep)[0]
```
```python
if nunits > 0:
    has_spikes = rates[trials].sum(axis=(1, 2)) > 0
    n_nospk = int((~has_spikes).sum())
    trials = trials[has_spikes]
```
```python
if r['nunits'] < MIN_UNITS:
    print(f"  SKIP {r['session_id']}: only {r['nunits']} units (< {MIN_UNITS})")
    continue
if r['ntrials_kept'] < 2:
    print(f"  SKIP {r['session_id']}: only {r['ntrials_kept']} trials")
    continue
```

iii. The notes explicitly justify the extra zero-spike removal as a fix for trailing behavioral trials after the recording stopped, and justify retaining ignore trials because the decoder spec requires an ignore / none output class.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from each cluster’s `trial` and `trialtm` arrays in `obj.clu`, with `obj.bp.ev.goCue` used to align spikes to the go cue. The cluster `quality` string is also used during curation, and probe-location metadata is read from `obj.ex.probe.loc`.

ii. ```python
for c in s.clusters(p - 1):
    q = c['quality'].strip().lower()
    if q in BAD_QUALITY:
        continue
    clusters.append(c)
```
```python
tr = np.asarray(c['trial'], dtype=float).ravel()
tt = np.asarray(c['trialtm'], dtype=float).ravel()
...
aligned = tt - gocue[(tr - 1).astype(int)]
```

iii. The notes state that this is a direct port of the reference `findClusters.m` and `alignSpikes.m` logic.

## 2-b. How is the `neural` data processed?

i. The AI bins aligned spikes into a fixed `-2.5` to `+2.5` s window using 30 ms bins, converts counts to firing rates in spikes/s, then smooths each unit’s binned firing rate with a causal Gaussian kernel of width 15 bins using `mysmooth`. The output stored per trial is a `(n_units, n_bins)` matrix.

ii. ```python
TMIN = -2.5
TMAX = 2.5
DT = 0.03
SMOOTH = 15
```
```python
cnt, _, _ = np.histogram2d(tr, aligned, bins=[trial_edges, edges])
meanfr[i] = cnt.sum() / (ntrials * window)
rates[:, :, i] = mysmooth(cnt.T / DT).T.astype(np.float32)
```
```python
for i in trials:
    neural.append(np.ascontiguousarray(rates[i].T))
```

iii. The notes justify this as following the paper’s Figure 3 parameterization: 30 ms bins, causal Gaussian smoothing, and rates in spikes/s.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered first by free-text quality label and then by mean firing rate. The AI drops qualities in `{'garbage', 'gabrga', 'noisy', 'real?', ''}`, keeps the rest, computes mean firing rate over the full aligned window, and retains only units with `meanfr > 1 Hz`. Sessions with fewer than 10 remaining units are later discarded.

ii. ```python
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?', ''}
LOW_FR = 1.0
MIN_UNITS = 10
```
```python
for c in s.clusters(p - 1):
    q = c['quality'].strip().lower()
    if q in BAD_QUALITY:
        continue
    clusters.append(c)
```
```python
keep_units = meanfr > LOW_FR
rates = rates[:, :, keep_units]
...
if r['nunits'] < MIN_UNITS:
    ...
```

iii. The notes justify the label handling as a trimmed, case-insensitive port of `findClusters.m`, and justify the `> 1 Hz` threshold and `>= 10 units/session` rule from the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time is aligned by subtracting that spike’s trial’s go-cue time: `trialtm - goCue[trial-1]`. The aligned spike times are then binned on the common decoder time axis.

ii. ```python
aligned = tt - gocue[(tr - 1).astype(int)]
cnt, _, _ = np.histogram2d(tr, aligned, bins=[trial_edges, edges])
```

iii. The notes explicitly identify this as the port of `alignSpikes.m` with `params.alignEvent = 'goCue'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 30 ms bins spanning `-2.5` to `+2.5` s, yielding 166 bins per trial. No later rebinning is applied inside the conversion script.

ii. ```python
DT = 0.03
...
def time_axis():
    nedges = int(np.floor((TMAX - TMIN) / DT)) + 1
    edges = TMIN + DT * np.arange(nedges)
    t = edges + DT / 2.0
    return edges, t[:-1]
```

iii. The notes justify 30 ms as the setting used in the paper’s Figure 3 scripts and say it was kept after small empirical comparisons against other candidate bin widths.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The decoder input is not read from a raw signal array. It is a synthetic time axis defined relative to the go cue, using the same aligned window used for neural binning.

ii. ```python
def time_axis():
    nedges = int(np.floor((TMAX - TMIN) / DT)) + 1
    edges = TMIN + DT * np.arange(nedges)
    t = edges + DT / 2.0
    return edges, t[:-1]
```
```python
tin = taxis.astype(np.float32).reshape(1, -1)
for i in trials:
    inp.append(tin.copy())
```

iii. The notes describe this as the direct analogue of the reference `obj.time` axis.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI computes bin centers from the common `[-2.5, 2.5]` s window and 30 ms bin width, casts the result to `float32`, reshapes it to `(1, n_bins)`, and copies the same array into every trial.

ii. ```python
edges, taxis = time_axis()
...
tin = taxis.astype(np.float32).reshape(1, -1)
for i in trials:
    inp.append(tin.copy())
```

iii. The notes justify this as the decoder input corresponding exactly to the time axis used everywhere else in the session.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is aligned to the neural data by construction: both use the same `time_axis()` output and the same go-cue-centered window.

ii. ```python
edges, taxis = time_axis()
...
rates, meanfr = bin_spikes(clusters, gocue, ntrials, edges)
...
tin = taxis.astype(np.float32).reshape(1, -1)
```

iii. The notes explicitly say `input[0] = obj.time` and that `obj.time` is the same axis used by the aligned neural activity.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the lick-event arrays `bp.ev.lickL` and `bp.ev.lickR`, together with `goCue` so that only post-go-cue licks are considered.

ii. ```python
lickL, lickR = s.ev_cell('lickL'), s.ev_cell('lickR')
lick_dir = np.full(ntrials, 2, dtype=np.int64)
for i in range(ntrials):
    tl = np.concatenate([lickL[i] - gocue[i], lickR[i] - gocue[i]])
    side = np.concatenate([np.zeros(lickL[i].size), np.ones(lickR[i].size)])
```

iii. The notes justify this by citing the paper code’s `firstLickTime.m` behavior and by reporting that first post-go-cue lick side matched the hit/miss-based alternative on tested sessions.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. For each trial, the AI concatenates left and right lick times relative to the go cue, keeps only licks after the go cue, and labels the earliest such lick as left (`0`) or right (`1`). If no post-go-cue lick exists, the label remains `2` for none.

ii. ```python
m = tl > 0
if m.any():
    lick_dir[i] = int(side[m][np.argmin(tl[m])])
```

iii. The notes justify this as a more direct reconstruction of lick direction than inferring it from hit/miss and instructed side.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is derived from `bp.autowater`.

ii. ```python
autowater = s.bpvec('autowater')
...
context = np.where(autowater[:ntrials] > 0, 0, 1).astype(np.int64)
```

iii. The notes say this follows the paper’s use of `autowater` to distinguish water-cued (WC) from delayed-response (DR) trials.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI maps `autowater > 0` to WC (`0`) and everything else to DR (`1`), then broadcasts that per-trial label across time bins when assembling the output tensor.

ii. ```python
context = np.where(autowater[:ntrials] > 0, 0, 1).astype(np.int64)
...
o[1] = context[i]
```

iii. The notes justify the code mapping as matching the decoder task’s requested category order.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the Bpod per-trial flags `hit`, `miss`, and `no`.

ii. ```python
hit = s.bpvec('hit')
miss = s.bpvec('miss')
no = s.bpvec('no')
```
```python
outcome = np.full(ntrials, 2, dtype=np.int64)
outcome[miss[:ntrials] > 0] = 0
outcome[hit[:ntrials] > 0] = 1
```

iii. The notes justify using these trial-outcome flags directly from the Bpod table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Trials default to ignore (`2`), then miss trials are relabeled incorrect (`0`) and hit trials correct (`1`). The per-trial class is repeated across all time bins in the final output array.

ii. ```python
outcome = np.full(ntrials, 2, dtype=np.int64)
outcome[miss[:ntrials] > 0] = 0
outcome[hit[:ntrials] > 0] = 1
...
o[2] = outcome[i]
```

iii. The notes justify retaining ignore trials because the decoder spec requires them as a third class rather than dropping them as many paper analyses did.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the side-camera DeepLabCut tongue track only: the side-view `traj` entry, its `ts` coordinates for the `tongue` feature, the side-camera `frameTimes`, and the session’s `vidshift` plus `goCue` for alignment.

ii. ```python
fn0, fn1 = s.feat_names(0), s.feat_names(1)
i_tongue = fn0.index('tongue')
```
```python
ts0, ft0, nd0 = s.traj_trial(0, i)
...
xy = ts0[:, :2, i_tongue]
vis = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
spd = speed_from_xy(xy) / dtf
_accumulate(idx, inb, spd, vis, tongue_spd, tongue_vis, i, nbins)
```

iii. The notes justify this by saying the conversion follows the side-camera tongue feature from the reference kinematics code, while preserving missing visibility as its own class.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each kept trial, the AI computes framewise tongue speed from the side-camera `x,y` track using finite-coordinate runs only, converts it to pixels/s by dividing by the median frame interval, bins framewise speed into the decoder time bins with a per-bin mean, and later discretizes the binned speed. It does not smooth position with a Gaussian, does not interpolate the track to the decoder time axis, and does not combine the second tongue view.

ii. ```python
def speed_from_xy(xy):
    vis = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
    spd = np.full(xy.shape[0], np.nan)
    ...
    for a, b in zip(starts, stops):
        seg = idx[a:b + 1]
        if seg.size == 1:
            spd[seg] = 0.0
            continue
        vx = np.gradient(xy[seg, 0])
        vy = np.gradient(xy[seg, 1])
        spd[seg] = np.sqrt(vx ** 2 + vy ** 2)
    return spd
```
```python
spd = speed_from_xy(xy) / dtf
_accumulate(idx, inb, spd, vis, tongue_spd, tongue_vis, i, nbins)
```

iii. The notes justify the segment-wise gradient as a bug fix so visible frames adjacent to missing ones are not mislabeled as not visible, and justify keeping NaNs as an explicit class rather than zero-filling them as in the paper’s tongue-position pipeline.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI computes a per-session 50th percentile over visible tongue-speed bins from the kept trials, then uses `discretize` to assign `0` below threshold, `1` at or above threshold, and `2` when the tongue is not visible.

ii. ```python
m_t = sel[:, None] & tongue_vis & np.isfinite(tongue_spd)
thr_t = np.nanpercentile(tongue_spd[m_t], 50) if m_t.any() else np.nan
...
tongue_cls = discretize(tongue_spd, tongue_vis, thr_t)
```
```python
def discretize(values, visible, thresh):
    out = np.full(values.shape, 2, dtype=np.int64)
    v = visible & np.isfinite(values)
    out[v & (values < thresh)] = 0
    out[v & (values >= thresh)] = 1
    return out
```

iii. The notes justify the session-median split as required by the decoder instructions.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Side-camera frame times are shifted onto the behavior clock with `vidshift`, then aligned to the trial’s go cue by subtracting `gocue[i]`. Those aligned frame times are digitized into the same decoder bins used for the neural data.

ii. ```python
vidshift = s.vidshift()
...
vt0 = ft0 - vidshift - gocue[i]
idx = np.digitize(vt0, bin_edges) - 1
inb = (idx >= 0) & (idx < nbins)
```

iii. The notes explicitly cite `findVideoOffset.m` and say the same `vidshift` correction is used for all video-derived streams.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom-camera DeepLabCut tracks `top_paw` and `bottom_paw`, using their `ts` coordinates, bottom-camera `frameTimes`, and the session `vidshift` plus `goCue`.

ii. ```python
fn0, fn1 = s.feat_names(0), s.feat_names(1)
i_paws = [fn1.index(f) for f in ('top_paw', 'bottom_paw') if f in fn1]
```
```python
ts1, ft1, nd1 = s.traj_trial(1, i)
...
for j in i_paws:
    pxy = ts1[:, :2, j]
    spds.append(speed_from_xy(pxy) / dtf1)
    viss.append(np.isfinite(pxy[:, 0]) & np.isfinite(pxy[:, 1]))
```

iii. The notes justify this as using the bottom-camera paw markers and averaging over whichever paw markers are visible.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each bottom-camera paw marker, the AI computes runwise framewise speed from `x,y`, converts to pixels/s, averages the available markers with `nanmean`, marks a bin visible if either marker was visible, and averages visible framewise speeds within each decoder bin.

ii. ```python
spds, viss = [], []
for j in i_paws:
    pxy = ts1[:, :2, j]
    spds.append(speed_from_xy(pxy) / dtf1)
    viss.append(np.isfinite(pxy[:, 0]) & np.isfinite(pxy[:, 1]))
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    spd_p = np.nanmean(np.vstack(spds), axis=0)
vis_p = np.any(np.vstack(viss), axis=0)
_accumulate(idx1, inb1, spd_p, vis_p, paw_spd, paw_vis, i, nbins)
```

iii. The notes justify this as a decoder-oriented simplification of the kinematic features, with “not visible” preserved explicitly instead of filling gaps.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. A per-session 50th percentile is computed over visible paw-speed bins from kept trials, and `discretize` assigns `0` below threshold, `1` at or above threshold, and `2` when no paw marker is visible.

ii. ```python
m_p = sel[:, None] & paw_vis & np.isfinite(paw_spd)
thr_p = np.nanpercentile(paw_spd[m_p], 50) if m_p.any() else np.nan
...
paw_cls = discretize(paw_spd, paw_vis, thr_p)
```

iii. The notes justify the per-session median split from the decoder task.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times are shifted by `vidshift`, aligned to each trial’s go cue, and digitized into the same decoder bins used for the neural data.

ii. ```python
vt1 = ft1 - vidshift - gocue[i]
idx1 = np.digitize(vt1, bin_edges) - 1
inb1 = (idx1 >= 0) & (idx1 < nbins)
```

iii. The notes say the same video/ephys synchronization formula is applied to all camera-derived outputs.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_<animal>_<date>.mat` file loaded by `load_motion_energy`, and it is aligned with the side-camera frame times from `traj_trial(0, i)`.

ii. ```python
def load_motion_energy(fn):
    me = sio.loadmat(fn)['me']
    ...
    if data.dtype == object:
        trials = [np.asarray(data.ravel()[i], dtype=float).ravel() for i in range(data.size)]
    else:
        trials = [np.asarray(data, dtype=float).ravel()]
    return trials, thresh[0]
```
```python
me_trials, _ = load_motion_energy(me_path(entry))
```

iii. The notes justify the custom loader because the shared motion-energy files appear in three different nested MATLAB layouts.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. For each kept trial, the AI pairs the motion-energy trace with side-camera aligned frame times, linearly interpolates it onto the decoder time axis, fills remaining gaps by nearest-neighbor propagation, and then discretizes the resulting binned time series with a session-median threshold. It does not differentiate or smooth the trace.

ii. ```python
if i < len(me_trials):
    mev = me_trials[i]
    n = min(mev.size, vt0.size)
    if n > 1:
        y = interp_to_axis(vt0[:n], mev[:n], taxis)
        y = _fill_nearest(y)
        me_binned[i] = y
```

iii. The notes justify this as following `loadMotionEnergy.m`, which interpolates motion energy onto the common time axis and fills missing bins.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes the session 50th percentile over finite motion-energy bins from kept trials, then uses `discretize` to assign `0` below threshold, `1` at or above threshold, and `2` when there is no usable video / no finite value.

ii. ```python
m_m = sel[:, None] & np.isfinite(me_binned)
thr_m = np.nanpercentile(me_binned[m_m], 50) if m_m.any() else np.nan
...
me_cls = discretize(me_binned, has_video[:, None] & np.isfinite(me_binned), thr_m)
```

iii. The notes justify the median split from the decoder spec and the class-2 handling for missing or unusable video.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by using side-camera frame times shifted with `vidshift` and then with each trial’s go cue, after which the trace is interpolated onto the same `taxis` used by neural activity.

ii. ```python
vt0 = ft0 - vidshift - gocue[i]
...
y = interp_to_axis(vt0[:n], mev[:n], taxis)
```

iii. The notes say this mirrors the reference video/ephys synchronization and uses the common decoder time axis.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI adds several robustness layers. It supports both MATLAB storage formats and three motion-energy layouts; it trims and lower-cases quality labels; it skips unusable video trials when frame times or dropped-frame metadata are invalid; it leaves untracked tongue/paw bins as explicit class `2`; it fills motion-energy gaps by nearest neighbor after interpolation; and it drops trailing trials with zero neural activity across all curated units.

ii. ```python
if self.hdf5:
    self.f = h5py.File(fn, 'r')
    self.obj = self.f['obj']
else:
    self.obj = sio.loadmat(fn, struct_as_record=False, squeeze_me=True)['obj']
```
```python
if ts0.size == 0 or ft0.size == 0 or np.all(~np.isfinite(ft0)) or not np.isfinite(nd0):
    continue
```
```python
def _fill_nearest(y):
    ...
    y[bi] = y[nearest]
    return y
```
```python
if nunits > 0:
    has_spikes = rates[trials].sum(axis=(1, 2)) > 0
    trials = trials[has_spikes]
```

iii. The notes justify these choices as necessary to survive the heterogeneous shared files and to represent missing visibility honestly as a decoder class instead of fabricating kinematics.

## 11-a. What are the most time-consuming steps of the code?

i. The AI identifies video processing, especially per-trial HDF5 DeepLabCut reads and framewise binning, as the dominant cost, with neural spike binning as a secondary cost. It also parallelizes sessions with a worker pool.

ii. ```python
t1 = time.time()
rates, meanfr = bin_spikes(clusters, gocue, ntrials, edges)
timing['neural_bin'] = time.time() - t1
...
t1 = time.time()
...
timing['video'] = time.time() - t1
```
```python
if nproc > 1:
    with Pool(nproc) as pool:
        results = pool.map(_worker, list(zip(sessions, want_debug)))
```

iii. The notes explicitly state that per-trial DLC reads from HDF5 dominate runtime and motivated the multiprocessing and vectorized binning choices.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorizes spike binning with `np.histogram2d`, bin aggregation with `np.bincount`, and smooths with convolution over arrays. The remaining obvious loops are over sessions, trials, clusters, and paw markers; the notes argue these remain because camera frames are ragged per trial and HDF5 data are stored per-trial.

ii. ```python
for i, c in enumerate(clusters):
    ...
    cnt, _, _ = np.histogram2d(tr, aligned, bins=[trial_edges, edges])
```
```python
for i in trials:
    ...
    _accumulate(idx, inb, spd, vis, tongue_spd, tongue_vis, i, nbins)
```
```python
sums = np.bincount(idx[good], weights=spd[good], minlength=nbins)
cnts = np.bincount(idx[good], minlength=nbins)
```

iii. The notes justify leaving those loops in place because most of the remaining work is shaped by variable-length per-trial video data rather than a single rectangular tensor.

## 11-c. What processing does the code repeat multiple times?

i. In the main conversion path, the AI tries to avoid repeated recomputation: it computes `time_axis()` once per session, loads event vectors once, computes `vidshift` once per session, builds full per-session neural and video matrices, then thresholds from those cached arrays. There is little intentional repeated processing beyond iterating through trials once per output stream.

ii. ```python
edges, taxis = time_axis()
...
vidshift = s.vidshift()
...
rates, meanfr = bin_spikes(clusters, gocue, ntrials, edges)
...
tongue_cls = discretize(tongue_spd, tongue_vis, thr_t)
paw_cls = discretize(paw_spd, paw_vis, thr_p)
me_cls = discretize(me_binned, has_video[:, None] & np.isfinite(me_binned), thr_m)
```

iii. The notes explicitly describe the implementation as precomputing arrays session-wise and then reusing them for thresholding and assembly.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does some optional and auxiliary work that the decoder itself does not need: debug tensors for plotting when `--show-processing` is enabled, session timing and threshold metadata, session grouping labels, and per-unit quality strings that are not written to the final pickle. It also stores a duplicate global `time_axis` in metadata even though the same information is present in each trial’s input.

ii. ```python
if want_debug:
    res['debug'] = dict(taxis=taxis, rates=rates, tongue_spd=tongue_spd, tongue_vis=tongue_vis,
                        paw_spd=paw_spd, paw_vis=paw_vis, me=me_binned, has_video=has_video,
                        lick_dir=lick_dir, context=context, outcome=outcome, gocue=gocue,
                        lickL=lickL, lickR=lickR, clusters_kept=int(nunits))
```
```python
res = dict(
    session_id=sess_id, animal=anm, date=date, group=('fixed' if entry in FIXED else 'randomized'),
    ...
    thresholds=dict(tongue=float(thr_t), paw=float(thr_p), me=float(thr_m)),
    vidshift=vidshift, timing=timing, elapsed=time.time() - t0,
    n_novideo=int(np.sum(~has_video[trials])), n_nospike=n_nospk,
)
```
```python
data['metadata'] = dict(
    ...
    time_axis=taxis.astype(np.float32),
    source='Zenodo 10.5281/zenodo.13941415; code https://github.com/economolab',
    session_info=session_info,
)
```

iii. The notes justify most of this as validation and documentation support rather than decoder-facing output.
