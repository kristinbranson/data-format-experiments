# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each session is one MATLAB v7.3 HDF5 file (`data_structure_<anm>_<date>.mat`) in the `Ephys_Behavior` folder. The AI created a `sessionio.py` helper that opens each file with `h5py` and provides accessor methods for behavior, spikes, and video data. The AI only loads sessions from `Ephys_Behavior` (25 sessions, 10 mice) and does NOT load the `RandomizedDelay_Ephys_Behavior` sessions. Motion energy is loaded from separate `motionEnergy_<anm>_<date>.mat` files using `scipy.io.loadmat`.

ii.
```python
DATA_DIR = '/app/data/Ephys_Behavior'
SESSIONS = [
    ('JEB6',  '2021-04-18', [2]),
    ('JEB7',  '2021-04-29', [1]),
    ...  # 25 sessions total, all from Ephys_Behavior
]

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

iii. The agent surveyed the figure scripts and found that the main analyses use only the Ephys_Behavior sessions (10 mice). It stated that the RandomizedDelay sessions appear only in secondary analyses (Figure3h, Figure3i, EDFigure2a_Right). When the agent tried to load a RandomizedDelay file, it failed to open, reinforcing the decision to focus on Ephys_Behavior only.

## 1-b. How are the data split into subjects?

i. The animal name is the first element in the SESSIONS tuple (e.g., `'JEB6'`). Each session's `process_session` receives the animal name directly and stores it in `res['subject']`. At assembly, subjects are built as a list of unique animal names in order of first appearance.

ii.
```python
SESSIONS = [
    ('JEB6',  '2021-04-18', [2]),
    ...
]

for anm, date, probes in sessions:
    res = process_session(anm, date, probes)
    ...
    if res['subject'] not in data['subjects']:
        data['subjects'].append(res['subject'])
    data['subject_idx'].append(data['subjects'].index(res['subject']))
```

iii. The animal name is directly encoded in the session list, matching the authors' loading scripts.

## 1-c. How are the data split into sessions?

i. One session is one entry in the `SESSIONS` list, keyed by `(anm, date, probes)`. Each becomes one element of `neural`, `input`, and `output`. Only `Ephys_Behavior` sessions (25 total) are used; `RandomizedDelay_Ephys_Behavior` sessions are excluded. Sessions with fewer than 10 units after quality filtering are skipped.

ii.
```python
SESSIONS = [
    ('JEB6',  '2021-04-18', [2]),
    ...  # 25 sessions
]

for anm, date, probes in sessions:
    res = process_session(anm, date, probes)
    if res is None:
        continue
```

```python
if fr.shape[0] < MIN_UNITS or len(trials) < 2:
    return None
```

iii. The agent cited the Methods: "Recording sessions were included for analysis only if they had at least 10 units." The session list is transcribed from the authors' `load<ANM>_ALMVideo.m` scripts.

## 1-d. How are the data split into trials?

i. Trials are defined by `bp.Ntrials`. The `bp_flag` and `ev` methods read per-trial fields. Trial indices are 0-based within the code, with spike trial numbers being 1-indexed in the raw data (adjusted with `tr - 1`). Each trial has one go cue time from `bp.ev.goCue`.

ii.
```python
@property
def ntrials(self):
    return int(vec(self.f, self.o['bp']['Ntrials'])[0])

def bp_flag(self, name):
    return vec(self.f, self.o['bp'][name]).astype(bool)
```

iii. The Bpod table defines trials directly, with one go cue per trial.

## 1-e. How are trials filtered based on quality controls?

i. Two filters are applied: early-lick trials (`bp.early`) and photostimulation trials (`bp.stim.enable`) are excluded. Additionally, trials with non-finite go cue times are excluded. No filtering based on recording length is done (unlike the reference).

ii.
```python
keep = (~early) & (~stim) & np.isfinite(gocue)
trials = np.where(keep)[0]
```

iii. The agent follows the paper's Methods, which state early-lick trials were omitted from all analyses, and photoinactivation is treated as a separate experiment.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` spike-sorted clusters. Each cluster provides `trial` (1-based trial index for each spike) and `trialtm` (spike time relative to trial start). The go cue times `bp.ev.goCue` are used for alignment.

ii.
```python
def clu_spikes(self, prbnum, cluix):
    g = self.probe_group(prbnum)
    trial = np.array(self.f[np.array(g['trial'][()]).ravel()[cluix]][()]).ravel().astype(int)
    trialtm = np.array(self.f[np.array(g['trialtm'][()]).ravel()[cluix]][()]).ravel().astype(float)
    return trial, trialtm
```

iii. Same source variables as the reference pipeline.

## 2-b. How is the `neural` data processed?

i. Spikes are binned at 10 ms (DT=0.01) into bins spanning -2.5 to +2.5 s from the go cue, yielding 500 time bins. Spike counts are converted to firing rate (Hz) by dividing by DT. The rate is then smoothed with a **causal** Gaussian kernel (15-bin window, alpha=2.5, first half zeroed, normalized), matching `mySmooth.m`. Units from multiple probes are concatenated.

ii.
```python
DT = 0.01             # s, bin size (params.dt = 1/100)
SMOOTH_N = 15         # params.smooth

def causal_gaussian_kernel(n=SMOOTH_N):
    k = np.arange(n)
    alpha = 2.5
    w = np.exp(-0.5 * (alpha * (k - (n - 1) / 2) / ((n - 1) / 2)) ** 2)
    w[:n // 2] = 0.0                # 'causal' step in mySmooth.m
    return w / w.sum()

def smooth_causal(x):
    pad = x[..., :SMOOTH_N]
    xf = np.concatenate([pad, x], axis=-1)
    out = convolve1d(xf, KERN, axis=-1, mode='constant', cval=0.0)
    return out[..., SMOOTH_N:]

rate = smooth_causal(counts / DT)
```

iii. The agent read `mySmooth.m` and identified the causal half-zeroing of the Gaussian window. The agent chose DT=0.01 because the figure scripts consistently override the default 1/200 to 1/100. The agent verified the causal kernel with a delta-function test.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) Quality labels matching `('garbage', 'gabrga', 'noisy', 'real?')` are excluded (lower-cased comparison). Note: `'poor'` is NOT excluded. (2) Units with mean firing rate <= 1 Hz are removed. (3) Sessions with fewer than 10 remaining units are excluded entirely.

ii.
```python
BAD_QUALITY = ('garbage', 'gabrga', 'noisy', 'real?')  # findClusters.m
LOW_FR = 1.0          # Hz
MIN_UNITS = 10        # Methods

qual = s.qualities(prb)
cluix = [i for i, q in enumerate(qual) if q.lower() not in BAD_QUALITY]

meanfr = fr.mean(axis=(1, 2))
use = meanfr > LOW_FR
fr = fr[use]

if fr.shape[0] < MIN_UNITS or len(trials) < 2:
    return None
```

iii. The agent read `findClusters.m` which excludes `garbage`, `gabrga`, `noisy`, and `real?`. It does not include `poor` in the exclusion list. The MIN_UNITS=10 filter comes from the Methods section.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to the go cue is done by subtracting the go cue time from each spike's trial time: `aligned = ttm - gocue[tr - 1]`. Spikes are then binned relative to the go cue.

ii.
```python
tr, ttm = s.clu_spikes(prb, ci)
aligned = ttm - gocue[tr - 1]                 # alignSpikes.m
bi = np.floor((aligned - TMIN) / DT).astype(int)
```

iii. This matches the reference's `alignSpikes.m`: `obj.clu{prb}(clu).trialtm_aligned = obj.clu{prb}(clu).trialtm - event`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 10 ms (DT = 0.01 s), producing 500 time bins over the -2.5 to +2.5 s window. No rebinning is applied; spikes are directly counted into 10 ms bins.

ii.
```python
TMIN = -2.5           # s relative to go cue
TMAX = 2.5            # s relative to go cue
DT = 0.01             # s, bin size (params.dt = 1/100)

EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME = EDGES[:-1] + DT / 2
NT = len(TIME)
```

iii. The agent chose DT=0.01 because the paper's figure scripts consistently use `params.dt = 1/100`, overriding the `getDefaultParams.m` default of `1/200`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This is a synthetic variable: the bin centers of the time grid, defined by TMIN, TMAX, and DT. It is not derived from any raw data variable.

ii.
```python
TIME = EDGES[:-1] + DT / 2
inp = [np.asarray(TIME, dtype=np.float32).reshape(1, NT)] * len(trials)
```

iii. The time axis is defined by the binning parameters.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing -- it is the bin centers of the time grid, computed once and shared across all trials.

ii. N/A

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is the neural binning grid itself. Spikes are counted into bins defined by EDGES, and the input is the center of those same bins.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME = EDGES[:-1] + DT / 2
```

iii. The same time grid is used for neural data and input.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial fields: `bp.hit`, `bp.miss`, `bp.R`, and `bp.L`. Hit/miss determine whether the animal licked correctly/incorrectly, and R/L determine the instructed side.

ii.
```python
hit = s.bp_flag('hit')
miss = s.bp_flag('miss')
R = s.bp_flag('R')
L = s.bp_flag('L')
```

iii. The lick direction is not recorded directly but can be inferred from hit/miss and instructed side.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A hit on a right-instructed trial means the animal licked right; a miss means it licked left (the wrong side). The same logic applies in reverse for left-instructed trials. No-lick trials get class 2 ("none"). Codes: left=0, right=1, none=2.

ii.
```python
lick_dir = np.full(len(trials), 2, dtype=np.int64)
lick_dir[hit[trials] & R[trials]] = 1
lick_dir[hit[trials] & L[trials]] = 0
lick_dir[miss[trials] & R[trials]] = 0     # error trial: licked other way
lick_dir[miss[trials] & L[trials]] = 1
```

iii. The derivation follows from the task structure: a hit means the animal licked the instructed side, a miss means the opposite side.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `bp.autowater` -- a per-trial flag. Autowater trials are the water-cued (WC) context; all others are delayed-response (DR).

ii.
```python
autowater = s.bp_flag('autowater')
context = np.where(autowater[trials], 0, 1).astype(np.int64)
```

iii. This field directly indicates the behavioral context.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct mapping: autowater=True -> WC (0), autowater=False -> DR (1).

ii.
```python
context = np.where(autowater[trials], 0, 1).astype(np.int64)
```

iii. Straightforward relabeling.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Three per-trial flags: `bp.hit`, `bp.miss`, and `bp.no`. Hit maps to correct, miss to incorrect, no-response to ignore.

ii.
```python
hit = s.bp_flag('hit')
miss = s.bp_flag('miss')
no = s.bp_flag('no')
```

iii. The three flags are mutually exclusive and cover all trial types.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapping: miss -> incorrect (0), hit -> correct (1), no-response -> ignore (2). Default is 0 (incorrect), then hit and no are overlaid.

ii.
```python
outcome = np.full(len(trials), 0, dtype=np.int64)
outcome[hit[trials]] = 1
outcome[no[trials]] = 2
```

iii. Matches the instruction's specification of incorrect=0, correct=1, ignore=2.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`, specifically the side camera (view 0) feature `'tongue'`. Frame times (`frameTimes`) and tracked positions (`ts`) are used, along with the video offset from `sglx.bitcode.bitstart` and `bp.ev.bitStart`.

ii.
```python
feats0 = s.traj_featnames(0)     # side cam
tongue_ix = feats0.index('tongue')
ft, xy = s.traj_feat_trial(0, tr, tongue_ix)
```

iii. The agent used only the side camera's tongue feature, consistent with how the paper's pipeline uses `params.traj_features` where side cam tongue is the primary tongue marker.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. (1) Positions are interpolated from video frame times onto the neural time axis using linear interpolation. (2) Velocity is computed using a custom `nan_gradient` function that handles NaN positions without propagating NaNs to tracked neighbors. (3) Speed is `np.hypot(vel[:,0], vel[:,1])`. (4) The speed is discretized at the session 50th percentile, with NaN timepoints assigned class 2 ("not_visible").

ii.
```python
taxis = TIME + gocue[tr] + vidshift
ft, xy = s.traj_feat_trial(0, tr, tongue_ix)
pos = interp_to_axis(ft, xy, taxis)
vel = nan_gradient(pos)
tongue_speed[k] = np.hypot(vel[:, 0], vel[:, 1])

tongue_d = discretise(tongue_speed)
```

iii. The agent used only the side camera for tongue velocity, without combining two camera views. Positions are interpolated onto the neural time axis rather than being binned.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Split at the session 50th percentile (median) over all finite values. Class 0 = below median, class 1 = at or above median, class 2 = not visible (NaN).

ii.
```python
def discretise(x):
    out = np.full(x.shape, 2, dtype=np.int64)
    vis = np.isfinite(x)
    if vis.any():
        thr = np.percentile(x[vis], 50)
        out[vis] = (x[vis] >= thr).astype(np.int64)
    return out
```

iii. Matches the instructions: 0 = < 50th percentile, 1 = >= 50th percentile, 2 = not visible.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The video offset is computed as `mode(sglx.bitcode.bitstart)/fs - mode(bp.ev.bitStart)`. The neural time axis is then shifted into video clock coordinates: `taxis = TIME + gocue[tr] + vidshift`. DLC positions are interpolated onto this axis.

ii.
```python
def video_offset(self):
    bs = vec(self.f, self.o['sglx']['bitcode']['bitstart'])
    fs = vec(self.f, self.o['sglx']['fs'])[0]
    bitStart = self.ev('bitStart')
    m1 = stats.mode(bs, keepdims=False).mode / fs
    m2 = stats.mode(bitStart, keepdims=False).mode
    return float(m1 - m2)

taxis = TIME + gocue[tr] + vidshift
pos = interp_to_axis(ft, xy, taxis)
```

iii. This follows `findVideoOffset.m` and `findPosition.m` from the reference code.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The DLC tracking from the bottom camera (view 1), feature `'top_paw'`.

ii.
```python
feats1 = s.traj_featnames(1)     # bottom cam
paw_ix = feats1.index('top_paw')
ft1, xy1 = s.traj_feat_trial(1, tr, paw_ix)
```

iii. `top_paw` from the bottom camera is the reliably tracked paw.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same as tongue: positions are interpolated onto the neural time axis, velocity is computed with `nan_gradient`, speed is `np.hypot`. Additionally, a baseline subtraction is applied: the median frame-to-frame displacement is subtracted from the velocity, following `findVelocity.m`'s handling of non-tongue features.

ii.
```python
pos1 = interp_to_axis(ft1, xy1, taxis)
vel1 = nan_gradient(pos1)
# findVelocity.m subtracts the baseline (median) frame-to-frame displacement
base = np.nanmedian(np.diff(pos1, axis=0), axis=0)
vel1 = vel1 - base[None, :]
paw_speed[k] = np.hypot(vel1[:, 0], vel1[:, 1])
```

iii. The agent read `findVelocity.m` and found that non-tongue features have their baseline derivative subtracted.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: split at session 50th percentile, with NaN -> class 2.

ii.
```python
paw_d = discretise(paw_speed)
```

iii. Same discretization logic as tongue velocity.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same video offset and interpolation approach as tongue. The bottom camera's frame times are used.

ii.
```python
ft1, xy1 = s.traj_feat_trial(1, tr, paw_ix)
pos1 = interp_to_axis(ft1, xy1, taxis)
```

iii. Same alignment mechanism for all video-derived features.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Separate `motionEnergy_<anm>_<date>.mat` files. The motion energy data is loaded using `scipy.io.loadmat`, with nested struct unwrapping matching `loadMotionEnergy.m`.

ii.
```python
mepath = os.path.join(DATA_DIR, 'motionEnergy_%s_%s.mat' % (anm, date))
medat = sio.loadmat(mepath)['me']['data'][0, 0]
while medat.dtype.names is not None and 'data' in medat.dtype.names:
    medat = medat['data'][0, 0]
```

iii. The agent referenced `loadMotionEnergy.m`: `if isstruct(me.data), me.data = me.data.data`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy (one value per frame) is interpolated from video frame times onto the neural time axis using linear interpolation, then NaN values are filled with nearest valid values (`fill_nearest`), matching `loadMotionEnergy.m`'s `fillmissing(...,'nearest')`. The result is discretized at the session 50th percentile.

ii.
```python
vals = interp_to_axis(ft, m, taxis)
me_arr[k] = fill_nearest(vals)   # loadMotionEnergy.m fills nans
me_d = discretise(me_arr)
```

iii. The agent matched `loadMotionEnergy.m`'s NaN filling behavior.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same as tongue and paw: split at session 50th percentile, with NaN -> class 2 ("no_video").

ii.
```python
me_d = discretise(me_arr)
```

iii. Same discretization as other movement variables.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy uses the side camera's frame times (same as tongue). The frame times are shifted by the video offset and go cue, and the motion energy values are interpolated onto the neural time axis.

ii.
```python
ft, _ = s.traj_feat_trial(0, tr, tongue_ix)
taxis = TIME + gocue[tr] + vidshift
vals = interp_to_axis(ft, m, taxis)
```

iii. Same video offset approach as all other video-derived features.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) Trials with non-finite go cue times are excluded. (2) Trials with no video (`ndropped` is NaN or frame times are None/all NaN) get all-NaN velocity arrays, which become class 2 in discretization. (3) DLC positions with NaN values (low likelihood) propagate as NaN through velocity computation and become "not visible" class 2. (4) Motion energy NaN values from interpolation are filled with nearest valid values. (5) The `nan_gradient` function avoids propagating NaN from untracked frames to tracked neighbors.

ii.
```python
if ndrop is not None and not np.isfinite(ndrop[tr]):
    continue                                        # bad video trial

# nan_gradient handles NaN positions without propagating
vel = nan_gradient(pos)

# motion energy NaN filling
me_arr[k] = fill_nearest(vals)
```

iii. The agent progressively refined NaN handling, creating the `nan_gradient` function to reduce over-assignment of the 'not visible' class.

## 11-a. What are the most time-consuming steps of the code?

i. Reading the HDF5 files. Each `Session(path)` opens a v7.3 MATLAB file with h5py, and reading the spike data and trajectory data dominates runtime. The overall conversion is I/O bound.

ii.
```python
class Session:
    def __init__(self, path):
        self.f = h5py.File(path, 'r')
        self.o = self.f['obj']
```

iii. File I/O is inherently the bottleneck; all computation is fast by comparison.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop over video features (`for k, tr in enumerate(trials)`) processes tongue, paw, and motion energy one trial at a time. The per-cluster spike binning loop (`for ii, ci in enumerate(cluix)`) uses `np.add.at` per cluster rather than a vectorized histogram approach. These could be partially vectorized but are complicated by varying frame counts per trial and per-cluster spike arrays.

ii.
```python
for k, tr in enumerate(trials):
    ...
    ft, xy = s.traj_feat_trial(0, tr, tongue_ix)
    pos = interp_to_axis(ft, xy, taxis)
    ...

for ii, ci in enumerate(cluix):
    tr, ttm = s.clu_spikes(prb, ci)
    ...
    np.add.at(counts[ii], (ti[ok], bi[ok]), 1.0)
```

iii. Per-trial loops are needed because each trial has different numbers of frames; the spike loop could potentially use histogram2d but the current approach works correctly.

## 11-c. What processing does the code repeat multiple times?

i. The side camera's frame times are read twice per trial: once for the tongue feature (`s.traj_feat_trial(0, tr, tongue_ix)`) and again for motion energy. The video offset is computed once per session (efficient). The `gocue + vidshift` computation is done per trial (unavoidable).

ii.
```python
# For tongue:
ft, xy = s.traj_feat_trial(0, tr, tongue_ix)
# For motion energy:
ft, _ = s.traj_feat_trial(0, tr, tongue_ix)
```

iii. The duplicate frame time read is minor since it's just accessing HDF5 references.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `sessionio.py` helper provides methods like `lick_times`, `traj_trial`, `has_traj` that are not used by the conversion script. The `ndropped` check adds a filtering step that may exclude some valid trials unnecessarily. The `fill_nearest` call on motion energy fills NaN values that are then discretized -- the NaN filling is unnecessary if those values would just be assigned class 2 anyway (though it does affect the median threshold computation).

ii.
```python
def lick_times(self, side):  # not used
    ...
def traj_trial(self, view, trial):  # not used
    ...

me_arr[k] = fill_nearest(vals)   # fills NaNs before discretization
```

iii. The unused helper methods are part of a general-purpose session reader. The `fill_nearest` matches the reference MATLAB code's behavior.
