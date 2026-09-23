# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a list of 44 sessions (with their ALM probe numbers) in the `SESSIONS` list, transcribed from the authors' `load*_ALMVideo.m` scripts. Each session's `.mat` file is loaded with a format-detecting loader (`is_v73` checks the header, then dispatches to `_load_v73` or `_load_v7`). Only the needed fields (bp, clu for the specified probes, traj for specific features, sglx) are read. Motion energy is loaded separately from `motionEnergy_*.mat` files. Sessions are processed in parallel using `multiprocessing.Pool`.

ii.
```python
SESSIONS = [
    ('JEB6',  '2021-04-18', [2],    DATA_FIXED, 'fixed delay'),
    ...
    ('JEB24', '2023-11-03', [1],    DATA_RAND,  'randomized delay'),
]

def load_session(fn, probes, feats):
    return _load_v73(fn, probes, feats) if is_v73(fn) else _load_v7(fn, probes, feats)
```

iii. The AI documents that the session list was transcribed from the authors' loader scripts (commented-out entries excluded, JEB4/JEB5 data not available). The selective loading of only needed fields was done for performance.

## 1-b. How are the data split into subjects?

i. The animal ID is the first element of each session tuple (e.g., `'JEB6'`). Subjects are collected from all processed sessions and assigned indices in the order they appear.

ii.
```python
SESSIONS = [
    ('JEB6',  '2021-04-18', [2],    DATA_FIXED, 'fixed delay'),
    ...
]
# In main():
if r['anm'] not in data['subjects']:
    data['subjects'].append(r['anm'])
data['subject_idx'].append(data['subjects'].index(r['anm']))
```

iii. The animal ID comes directly from the session list which mirrors the authors' loader scripts.

## 1-c. How are the data split into sessions?

i. One session = one entry in the `SESSIONS` list = one `.mat` file on disk. The two task folders (fixed-delay and randomized-delay) are handled by storing the directory path in each session tuple. This yields 44 sessions (25 fixed-delay + 19 randomized-delay).

ii.
```python
SESSIONS = [
    ('JEB6',  '2021-04-18', [2],    DATA_FIXED, 'fixed delay'),
    ...
]
fn = os.path.join(ddir, 'data_structure_%s_%s.mat' % (anm, date))
```

iii. Documented in CONVERSION_NOTES Steps 1-4 as matching the paper's 25 + 19 session count.

## 1-d. How are the data split into trials?

i. Each session has `bp.Ntrials` trials. Per-trial fields are indexed by trial number. The number of trials (`N`) is read from `bp['Ntrials']` and used to truncate all per-trial arrays.

ii.
```python
bp = sess['bp']
N = int(bp['Ntrials'][0])
...
hit = np.nan_to_num(bp['hit']).astype(bool)
miss = np.nan_to_num(bp['miss']).astype(bool)
# etc., all arrays indexed up to N
```

iii. The Bpod trial structure directly defines trials; no inference of trial boundaries is needed.

## 1-e. How are trials filtered based on quality controls?

i. Three filters: (1) photostimulation trials (`stim.enable`) removed, (2) early-lick trials (`early`) removed, (3) trials with NaN go-cue alignment time removed. Additionally, after neural binning, trials where no unit fires any spike (ephys recording ended early) are dropped. Ignore trials are kept (required by the decoder spec). Sessions with fewer than 2 remaining trials or fewer than 10 units are excluded.

ii.
```python
valid_align = np.isfinite(align[:N])
keep = (~stim[:N]) & (~early[:N]) & valid_align
keep_trials = np.nonzero(keep)[0]
# ...
# After neural binning:
has_spikes = rates.sum(axis=(1, 2)) > 0
if n_noephys:
    rates = rates[has_spikes]
    keep_trials = keep_trials[has_spikes]
```

iii. From CONVERSION_NOTES: "All paper analyses use `~stim.enable` and `~early`." Ignore trials kept because they are a required decoder output class. No-ephys trial removal documented in Step 9.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` spike-sorted clusters: per-cluster `.trialtm` (spike time within trial), `.trial` (trial number, 1-based), `.quality` (curation label). Also `bp.ev.goCue` for alignment.

ii.
```python
for p in probes:
    c = clu.get(p)
    ...
    tr = c['trial'][i].astype(np.int64) - 1  # -> 0-based
    tm = c['trialtm'][i]
    ...
    aligned = tm[sel] - align_times[tr[sel]]
```

iii. The AI reads only the specified ALM probes per session as defined in the SESSIONS list, matching `meta.probe` in the reference loaders.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to the go cue, binned into 10 ms bins (matching `params.dt = 1/100`) over [-2.5, 2.5] s, converted to spikes/s by dividing by dt, then smoothed with a **causal** Gaussian kernel (matching `mySmooth.m`: `gausswin(15)` with the first `floor(15/2) = 7` taps zeroed, then normalized). The smoothed rates are then averaged into 50 ms bins (5x downsampling), yielding 100 timepoints per trial.

ii.
```python
def causal_kernel(N=SMOOTH_N):
    k = gausswin(N)
    k[:N // 2] = 0.0
    return k / k.sum()

def my_smooth(x, kern=KERN, bctype=BCTYPE):
    from scipy.signal import fftconvolve
    N = len(kern)
    if bctype == 'reflect':
        xf = np.concatenate([x[..., :N], x], axis=-1)
        trim = N
    ...

# In bin_spikes:
counts = np.zeros((nkeep, NFINE), dtype=np.float32)
# ... spike counting via np.add.at ...
rate = counts / DT_FINE                          # spikes / s
rate = my_smooth(rate)                           # causal gaussian
rates.append(downsample_mean(rate).astype(np.float32))
```

iii. The AI explicitly documents matching the reference code's `getSeq.m` and `mySmooth.m`: "10 ms binning + causal Gaussian smoothing (15 bins) exactly as `getSeq`/`mySmooth`". The 50 ms downsampling is documented as "a compromise between the reference decoding bin (75 ms) and temporal resolution".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) cluster quality labels matching {garbage, gabrga, noisy, real?} are excluded (case-insensitive comparison, matching `findClusters.m`). (2) Units whose mean firing rate (over all kept trials and the full time window) is <= 1 Hz are dropped (`removeLowFRClusters.m`, `params.lowFR = 1`). Additionally, sessions with fewer than 10 surviving units are excluded.

ii.
```python
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}
...
good = [i for i, q in enumerate(c['quality'])
        if q.strip().lower() not in BAD_QUALITY]
...
# removeLowFRClusters.m:
mean_fr = rates.mean(axis=(0, 2))
use = mean_fr > LOW_FR
rates = rates[:, use, :]
...
if rates.shape[1] < MIN_UNITS:
    return dict(ok=False, ...)
```

iii. From CONVERSION_NOTES: "`findClusters` excludes lowercase 'garbage','gabrga','noisy','real?'" and "All units with firing rates exceeding 1 Hz were included" (paper quote). Session minimum of 10 units from: "Recording sessions were included for analysis only if they had at least 10 units."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times (`trialtm`) are subtracted by the go cue time (`bp.ev.goCue`) for their respective trial, matching `alignSpikes.m`. This puts spikes in seconds relative to the go cue onset.

ii.
```python
aligned = tm[sel] - align_times[tr[sel]]
b = np.floor((aligned - TMIN) / DT_FINE).astype(np.int64)
```

iii. From CONVERSION_NOTES: "identical, including the `(1:n)/400 - 0.5` fallback" when compared with the reference `alignSpikes.m`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Spikes are initially binned at 10 ms (DT_FINE = 0.01), matching `params.dt = 1/100` in the reference. After smoothing, the 500 fine bins are averaged into groups of 5 to produce 100 output bins at 50 ms resolution (DT_OUT = 0.05). This is applied consistently to neural data and all video-derived outputs.

ii.
```python
DT_FINE = 0.01              # params.dt (1/100 s)
DOWNSAMPLE = 5              # 10 ms -> 50 ms decoder bins
DT_OUT = DT_FINE * DOWNSAMPLE

def downsample_mean(x, k=DOWNSAMPLE, nanmean=False):
    n = (x.shape[-1] // k) * k
    y = x[..., :n].reshape(x.shape[:-1] + (n // k, k))
    ...
    return y.mean(axis=-1)
```

iii. The AI explains the 50 ms bin size as "a compromise between the reference decoding bin (75 ms) and temporal resolution for time-varying outputs, and it puts the go cue exactly on a bin edge."

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This is a synthetic variable defined by the time axis. It equals the bin centres of the 50 ms output bins, spanning approximately -2.475 to +2.475 s.

ii.
```python
edges = np.round(np.arange(TMIN, TMAX + 1e-9, DT_FINE), 6)
centres = edges[:-1] + DT_FINE / 2.0
...
centres_out = centres[:nout * DOWNSAMPLE].reshape(nout, DOWNSAMPLE).mean(axis=1)
...
tin = TOUT.astype(np.float32)[None, :]
inputs = [tin.copy() for _ in range(kt.size)]
```

iii. N/A - this is defined by the decoder specification.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The bin centres are computed once from the time axis parameters. No processing of raw data is needed; this variable is the shared time grid for all trials.

ii. Same as 3-a above.

iii. N/A.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is the bin centres of the same grid used for neural data, so alignment is automatic by construction.

ii.
```python
EDGES, TCENT, TOUT = time_axes()
# TOUT is used as both the input variable and defines the neural time axis
```

iii. N/A.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Three per-trial fields: `bp.R` (right-instructed), `bp.hit`, and `bp.miss`. The actual lick direction is inferred from the combination of instructed side and outcome.

ii.
```python
hit = np.nan_to_num(bp['hit']).astype(bool)
miss = np.nan_to_num(bp['miss']).astype(bool)
R = np.nan_to_num(bp['R']).astype(bool)
L = np.nan_to_num(bp['L']).astype(bool)
```

iii. From CONVERSION_NOTES: "Verified in data: identical to the direction of the first lickport contact after the go cue on 382/382 trials of JEB6_2021-04-18."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A hit on a right-instructed trial means the animal licked right; a miss on a right-instructed trial means it licked left (and vice versa). Ignore trials get class 2 (none). Codes: left=0, right=1, none=2. The value is constant across all time bins within a trial.

ii.
```python
lick_dir = np.full(kt.size, 2, dtype=np.int64)          # 2 = none
right_lick = (R[kt] & hit[kt]) | (L[kt] & miss[kt])
left_lick = (L[kt] & hit[kt]) | (R[kt] & miss[kt])
lick_dir[right_lick] = 1
lick_dir[left_lick] = 0
```

iii. Matches `getPrevChoice.m`: `choice = (R&hit)|(L&miss)`.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `bp.autowater`: 1 = water-cued (WC) trial, 0 = delayed-response (DR) trial.

ii.
```python
aw = np.nan_to_num(bp['autowater']).astype(bool)
context = np.where(aw[kt], 0, 1).astype(np.int64)       # 0 = WC, 1 = DR
```

iii. From CONVERSION_NOTES: "`autowater` used 'as a proxy for obtaining water-cued blocks and delayed-response blocks' (WorkingWithDataObjs.m)".

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct mapping: autowater=1 -> WC (0), autowater=0 -> DR (1). The value is constant across all time bins. No further processing.

ii. Same as 5-a above.

iii. Matches the reference code's condition strings.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `bp.hit` and `bp.miss`. Ignore trials are identified as those that are neither hit nor miss.

ii.
```python
hit = np.nan_to_num(bp['hit']).astype(bool)
miss = np.nan_to_num(bp['miss']).astype(bool)
outcome = np.full(kt.size, 2, dtype=np.int64)           # 2 = ignore
outcome[hit[kt]] = 1
outcome[miss[kt]] = 0
```

iii. From CONVERSION_NOTES: "outcome = bp.hit, with bp.no (ignore) trials set to NaN" in the reference `getOutcome.m`; here ignore trials are kept as class 2.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct relabelling: miss -> incorrect (0), hit -> correct (1), neither -> ignore (2). The value is constant across all time bins within a trial.

ii. Same as 6-a above.

iii. Matches the decoder specification ordering.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. DeepLabCut tracking from the **side camera only**: feature `tongue` in `obj.traj{1}`. The tracked x, y coordinates and frame times are extracted. `bp.ev.goCue` and `sglx.bitcode.bitstart` are used for clock alignment.

ii.
```python
TONGUE_FEAT = ('side', 'tongue')
...
tongue_sp, _ = feature_speed(sess, TONGUE_FEAT, kt, vidshift, align, fill_missing=False)
```

iii. The AI documents: "The tongue, jaw and nose were tracked using both cameras, whereas the paws were tracked using only the bottom view" (paper quote), but only extracts the side camera tongue feature. The bottom camera's `top_tongue` feature is not used for tongue velocity.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. (1) Position is interpolated from frame times onto the 10 ms fine grid using `np.interp`. (2) NaN-marked frames (DLC not visible) remain NaN (no filling, matching the reference `findPosition.m` exception for tongue). (3) Velocity is computed as `np.gradient` of the interpolated position. (4) Speed = `np.hypot(vx, vy)`. (5) Non-visible bins remain NaN. (6) Speed is downsampled to 50 ms bins using `nanmean`. (7) Discretized at the per-session 50th percentile; not-visible bins -> class 2.

ii.
```python
def feature_speed(sess, key, keep_trials, vidshift, align_times, fill_missing):
    ...
    pos = np.empty((NFINE, 2))
    for d in range(2):
        pos[:, d] = np.interp(TCENT, ft[valid], xy[valid, d], left=np.nan, right=np.nan)
    vis = np.isfinite(pos).all(axis=1)
    ...
    vel = np.gradient(pos, axis=0)
    sp = np.hypot(vel[:, 0], vel[:, 1])
    sp[~vis] = np.nan
    ...
tongue_ds = downsample_mean(tongue_sp, nanmean=True)
tongue_cls, tongue_thr = discretize(tongue_ds)
```

iii. Follows `findPosition.m` (interp1 onto time axis) and `findVelocity.m` (gradient). Tongue NaNs are preserved as in the reference. No normalization since only one camera view is used.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per-session 50th percentile of all finite (visible) tongue speed values across all trials and time bins. Below threshold = 0, at or above = 1, not visible (NaN) = 2.

ii.
```python
def discretize(values, nan_class=2):
    finite = np.isfinite(values)
    cls = np.full(values.shape, nan_class, dtype=np.int64)
    if finite.any():
        thr = np.percentile(values[finite], 50)
        cls[finite & (values >= thr)] = 1
        cls[finite & (values < thr)] = 0
    ...
    return cls, thr
```

iii. Matches the decoder specification: "0: < 50th percentile, 1: >= 50th percentile, 2: not visible".

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by the video offset (`findVideoOffset.m` logic: `mode(sglx.bitcode.bitstart)/fs - mode(bp.ev.bitStart)`) and the trial's go cue time. Then interpolated onto the same 10 ms grid as neural data and downsampled to the same 50 ms bins.

ii.
```python
def video_offset(sess):
    vid_file_offset = float(smode(bs, keepdims=False).mode) / sess['sglx']['fs']
    return vid_file_offset - float(smode(np.round(bstart, 6), keepdims=False).mode)

def frame_times_for_trial(sess, trial, vidshift, align_t):
    ft = sess['frameTimes'][trial]
    ...
    return ft - vidshift - align_t
```

iii. From CONVERSION_NOTES: "Video/ephys synchronisation uses the reference `findVideoOffset`".

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. DeepLabCut tracking from the **bottom camera**: features `top_paw` and `bottom_paw` in `obj.traj{2}`. Both paw features are extracted and their speeds are averaged.

ii.
```python
PAW_FEATS = [('bottom', 'top_paw'), ('bottom', 'bottom_paw')]
...
paw_sps = [feature_speed(sess, k, kt, vidshift, align, fill_missing=True)[0] for k in PAW_FEATS]
paw_sp = np.nanmean(np.stack(paw_sps, axis=0), axis=0)
```

iii. From CONVERSION_NOTES: "The tongue, jaw and nose were tracked using both cameras, whereas the paws were tracked using only the bottom view."

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same interpolation pipeline as tongue velocity, but with `fill_missing=True`: (1) Interpolate position onto 10 ms grid, (2) nearest-neighbor fill for missing values (matching `fillmissing('nearest')` in `findPosition.m`), (3) gradient of position, (4) baseline derivative subtraction (median of diff, matching `findVelocity.m`), (5) speed = hypot, (6) non-visible bins set to NaN, (7) average both paw speeds, (8) downsample to 50 ms bins, (9) discretize at session 50th percentile.

ii.
```python
# In feature_speed with fill_missing=True:
if fill_missing:
    if vis.any():
        idx = np.arange(NFINE)
        pos = np.stack([np.interp(idx, idx[vis], pos[vis, d]) for d in range(2)], axis=1)
vel = np.gradient(pos, axis=0)
if fill_missing:
    base = np.nanmedian(np.diff(pos, axis=0), axis=0)
    vel = vel - base[0]
```

iii. Follows `findPosition.m` (nearest-fill for non-tongue features) and `findVelocity.m` (baseline subtraction).

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: per-session 50th percentile of finite values. Below = 0, above = 1, not visible = 2.

ii. Same `discretize` function as in 7-c.

iii. Matches the decoder specification.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same alignment as tongue: frame times corrected by video offset and go cue, interpolated onto the 10 ms grid, downsampled to 50 ms.

ii. Same `frame_times_for_trial` and `feature_speed` functions.

iii. Same clock correction and grid as all other streams.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Separate `motionEnergy_ANM_DATE.mat` files containing `me.data` (cell array of per-trial 400 Hz vectors) and `me.moveThresh`. Three file layouts are handled (bare cell, struct with data+moveThresh, nested struct).

ii.
```python
def load_motion_energy(fn):
    m = sio.loadmat(fn, struct_as_record=False, squeeze_me=True)
    me = m['me']
    ...
    while hasattr(raw, '_fieldnames'):
        if 'moveThresh' in raw._fieldnames:
            thresh = float(np.array(raw.moveThresh).ravel()[0])
        if 'data' not in raw._fieldnames:
            break
        raw = raw.data
    data = [np.atleast_1d(np.array(d).ravel()).astype(float) for d in np.atleast_1d(raw)]
    return dict(data=data, moveThresh=thresh)
```

iii. From CONVERSION_NOTES: matches `loadMotionEnergy.m` which handles nested struct variants.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The per-trial 400 Hz motion energy values are interpolated onto the 10 ms fine grid using frame times (corrected by video offset and go cue). Missing values are filled with nearest-neighbor interpolation (matching `fillmissing('nearest')` in `loadMotionEnergy.m`). Then downsampled to 50 ms bins and discretized at the session 50th percentile.

ii.
```python
def motion_energy_trials(me, sess, keep_trials, vidshift, align_times):
    ...
    v = np.interp(TCENT, ft[valid], y[valid], left=np.nan, right=np.nan)
    ok = np.isfinite(v)
    if not ok.all() and ok.any():
        idx = np.arange(NFINE)
        v = np.interp(idx, idx[ok], v[ok])
    out[r] = v
    ...
```

iii. Directly follows `loadMotionEnergy.m`: interpolate and nearest-fill.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same per-session 50th percentile split as other movement variables. Class 2 is for "no video" (NaN bins), though in practice the nearest-fill means very few or no bins remain NaN.

ii. Same `discretize` function.

iii. Matches the decoder specification; the AI's nearest-fill approach (following the MATLAB reference) means effectively no bins get class 2.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy uses the side camera's frame times (corrected by video offset and go cue), interpolated onto the same 10 ms grid, then downsampled to 50 ms bins.

ii.
```python
ft = frame_times_for_trial(sess, t, vidshift, align_times[t])
...
v = np.interp(TCENT, ft[valid], y[valid], left=np.nan, right=np.nan)
```

iii. Same alignment pipeline as all video-derived outputs.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled: (1) Trials with `NdroppedFrames = NaN` are skipped for video features (matching reference `findPosition.m`). (2) Frame time / position array length mismatches are truncated to the shorter length. (3) Trials with fewer than 2 valid frames are skipped (NaN output). (4) All-NaN frame times use a fallback: `(1:n)/400 - 0.5 - align_t`. (5) Trials with no ephys coverage (recording ended early) are dropped. (6) Sessions without sorted units are excluded. (7) Three different motion-energy file layouts are handled.

ii.
```python
# NdroppedFrames check:
if t < len(nd) and np.isnan(nd[t]):
    continue
# Frame time fallback:
if ft.size == 0 or np.all(np.isnan(ft)):
    ft = (np.arange(1, sess['nframes'][trial] + 1) / VIDEO_FS)
    return ft - 0.5 - align_t
# Length mismatch:
if ft.size != xy.shape[0] or ft.size < 2:
    m = min(ft.size, xy.shape[0])
    ...
```

iii. Each edge case is documented in CONVERSION_NOTES Step 10 Check 5 with specific examples.

## 11-a. What are the most time-consuming steps of the code?

i. File loading dominates runtime (3-5 s per session). Neural spike binning takes 0.1-1.4 s, and video processing takes 0.2-0.4 s. With 12-worker parallelism, total conversion is ~12 s for all 44 sessions.

ii.
```python
# Timing output example:
# JEB6_2021-04-18 (fixed delay): 29 units, 302/382 trials, load 3.2s neural 0.5s video 0.2s total 3.9s
```

iii. From CONVERSION_NOTES: "Sample sessions took 3.9 s and 4.6 s each (dominated by file loading)."

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop over clusters (`for i in good:`) processes one unit at a time with `np.add.at`. This could potentially be vectorized into a 3D histogram. The video feature processing loops over trials (`for r, t in enumerate(keep_trials):`). These stay as loops because each trial has different numbers of frames.

ii.
```python
# Per-unit spike counting:
for i in good:
    tr = c['trial'][i].astype(np.int64) - 1
    tm = c['trialtm'][i]
    ...
    np.add.at(counts.reshape(-1), flat, 1.0)
```

iii. The AI notes that spike binning is "vectorised over spikes" within each unit, but still loops over units. The parallelism across sessions (12 workers) compensates.

## 11-c. What processing does the code repeat multiple times?

i. The motion energy frame times are recomputed per trial using the same `frame_times_for_trial` function that was already called for the tongue/paw features (since they share the side camera frame times). However, the video offset is computed only once per session.

ii.
```python
# In feature_speed:
ft = frame_times_for_trial(sess, t, vidshift, align_times[t])
# And again in motion_energy_trials:
ft = frame_times_for_trial(sess, t, vidshift, align_times[t])
```

iii. The duplicate frame-time computation is minor; loading dominates the overall runtime.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `lickL` and `lickR` event arrays, `probe_loc` strings, and `NdroppedFrames` arrays that are not directly needed for the conversion (though they serve diagnostic/validation purposes). The `bottom_paw` feature is loaded and processed even though it could be argued only `top_paw` is needed. The `moveThresh` from the motion-energy file is loaded but not used (the decoder spec requires the 50th percentile instead).

ii.
```python
# lickL/lickR loaded but only used in diagnostic plots:
for k in ['lickL', 'lickR']:
    refs = bp['ev'][k]
    ...
# moveThresh loaded but not used for discretization:
me = load_motion_energy(mefn)  # returns dict with moveThresh
# discretization uses 50th percentile instead:
me_cls, me_thr = discretize(me_ds)
```

iii. These serve diagnostic purposes documented in the processing plots (`--show-processing` mode).
