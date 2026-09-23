# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads only the **12 two-context (DR + WC) sessions** from the `Ephys_Behavior` folder, not all 44 sessions across both folders. Each session is loaded from a `data_structure_<ANM>_<DATE>.mat` file using `h5py` (all files in `Ephys_Behavior` are HDF5/v7.3). Motion energy is loaded separately from `motionEnergy_<ANM>_<DATE>.mat` using `scipy.io.loadmat`. The 12 sessions and probe numbers are hard-coded in the `SESSIONS` list, taken from the authors' `load<ANM>_ALMVideo.m` scripts (specifically those loaded by `Scripts/Figure 8` and `Scripts/EDFigure 2a-left`).

ii. Session list and loading:
```python
SESSIONS = [
    ('JEB6',  '2021-04-18', 2),
    ('JEB7',  '2021-04-29', 1),
    ...  # 12 sessions total
    ('JEB19', '2023-04-21', 1),
]

def session_paths(anm, date):
    return (os.path.join(DATA_DIR, f'data_structure_{anm}_{date}.mat'),
            os.path.join(DATA_DIR, f'motionEnergy_{anm}_{date}.mat'))

f = h5py.File(data_path, 'r')
o = f['obj']
```

iii. The AI justified using only 12 sessions because "behavioural context (WC, DR)" is a required decoder output, and only these sessions contain both contexts. The AI noted these sessions reproduce the paper's 522-unit / 214-single-unit counts.

## 1-b. How are the data split into subjects?

i. The animal name is the first element of the session tuple (e.g., `'JEB6'`). Subjects are accumulated as sessions are processed, with `subject_idx` built as an index into the subjects list. The 12 sessions come from 7 animals.

ii.
```python
if anm not in data['subjects']:
    data['subjects'].append(anm)
data['subject_idx'].append(data['subjects'].index(anm))
```

iii. The animal ID comes from the session tuple which is derived from the filename, matching the authors' loading scripts.

## 1-c. How are the data split into sessions?

i. One session = one entry in the `SESSIONS` list = one `.mat` file. All 12 sessions are from the `Ephys_Behavior` folder (fixed-delay task). Each session becomes one element of `neural`, `input`, and `output`.

ii.
```python
for k, (anm, date, probe) in enumerate(sessions):
    neural, inp, out, info, extras = convert_session(anm, date, probe)
```

iii. These are the 12 two-context sessions loaded by the authors' Figure 8 scripts.

## 1-d. How are the data split into trials?

i. Trials are defined by `obj.bp`, which has `Ntrials` entries. Each trial has one go cue (`bp.ev.goCue`). The `load_bpod` function reads all per-trial fields truncated to `Ntrials`.

ii.
```python
def load_bpod(f, o):
    bp = o['bp']
    n = int(np.array(bp['Ntrials'])[0, 0])
    def g(name):
        return np.array(bp[name]).flatten()[:n]
    ...
```

iii. The Bpod table directly defines trials; each has exactly one go cue.

## 1-e. How are trials filtered based on quality controls?

i. Three filters: (1) early-lick trials (`bp.early`) are dropped, (2) photostimulation trials (`bp.stim.enable`) are dropped, (3) trials without usable video are dropped. Also, NaN go cue trials are excluded. Ignore trials (`bp.no`) are retained (deliberate deviation from the paper, since the decoder task requires an "ignore" outcome class). Additionally, sessions are skipped if they have fewer than 2 usable trials or fewer than 10 units.

ii.
```python
keep = (bp['hit'] | bp['miss'] | bp['no']) & ~bp['early'] & ~bp['stim'] & ~np.isnan(bp['goCue'])
...
keep = keep & kin['has_video']
```

iii. The AI documented that early-lick and photostim removal follows every `params.condition` in the reference code. Ignore trials are retained because the decoder task defines "ignore" as an outcome class. Trials without video are dropped because three of six outputs are undefined for them.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` spike-sorted clusters: each cluster has `trialtm` (spike times relative to trial start), `trial` (1-based trial index), and `quality` (manual curation label). The go cue times `bp.ev.goCue` are used for alignment.

ii.
```python
clu = f[o['clu'][probe - 1, 0]]
...
tm = np.array(f[clu['trialtm'][ci, 0]]).flatten()
tr = np.array(f[clu['trial'][ci, 0]]).flatten().astype(int)
aligned = tm - gocue[tr - 1]
```

iii. These are the same variables used by the reference's `alignSpikes.m` and `getSeq.m`.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into **10 ms** bins (not 5 ms) spanning [-2.5, 2.5] s (500 bins), divided by `DT` to get firing rates (Hz), then smoothed with a **causal** Gaussian kernel matching the reference's `mySmooth.m`: `gausswin(15)` with the first 7 taps zeroed, normalized, convolved with 'reflect' boundary conditions. No normalization, baseline subtraction, or z-scoring.

ii.
```python
DT = 0.01  # 10 ms bins
edges = np.arange(TMIN, TMAX + DT / 2, DT)
...
rate = bin_spikes(f, clu, qual_keep, bp['goCue'], n, edges)
rate_s = my_smooth(rate.reshape(t, -1)).reshape(rate.shape).astype(np.float32)
```

The causal kernel:
```python
def make_causal_kernel(n=SMOOTH):
    k = gausswin(n)
    k[:n // 2] = 0.0
    return k / k.sum()
```

iii. The AI chose 10 ms because `dt = 1/100` is the value used by every published figure script (not `getDefaultParams.m`'s `1/200`). The causal Gaussian is a direct port of `mySmooth.m`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. First, clusters with quality strings `garbage`, `gabrga`, `noisy`, or `real?` are discarded (case-sensitive after `strtrim`, matching `findClusters.m`). `poor` is NOT dropped. Second, clusters with condition-averaged mean firing rate <= 1 Hz are discarded, using the 7 conditions from `Figure8a_thru_c.m` (matching `removeLowFRClusters.m`).

ii.
```python
BAD_QUALITY = ('garbage', 'gabrga', 'noisy', 'real?')
quality = [matstr(f, r).strip() for r in np.array(clu['quality']).flatten()]
qual_keep = [i for i, q in enumerate(quality) if q not in BAD_QUALITY]
...
fr_mask, mean_fr = low_fr_mask(rate_s, fig8_conditions(bp))
```

The condition-averaged firing rate filter:
```python
def low_fr_mask(rate_smooth, conditions):
    psth = np.full((rate_smooth.shape[0], rate_smooth.shape[1], len(conditions)), np.nan)
    for j, c in enumerate(conditions):
        if c.sum() == 0:
            continue
        psth[:, :, j] = rate_smooth[:, :, c].mean(axis=2)
    mean_fr = np.nanmean(np.nanmean(psth, axis=2), axis=0)
    return mean_fr > LOW_FR, mean_fr
```

iii. The AI documented that these filters match `findClusters.m` (quality) and `removeLowFRClusters.m` (rate), using the condition list from `Figure8a_thru_c.m`. Reproduces 521 units (vs paper's 522, differing by 1 unit at the window edge) and exactly 214 single units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue by subtracting `goCue[trial]` from each spike's `trialtm`, then counted into bins spanning [-2.5, 2.5] s.

ii.
```python
aligned = tm - gocue[tr - 1]
inwin = (aligned >= edges[0]) & (aligned < edges[-1])
h, _, _ = np.histogram2d(aligned[inwin], tr[inwin], bins=[edges, trial_bins])
rate[:, k, :] = h / DT
```

iii. This matches `alignSpikes.m`: `trialtm_aligned = trialtm - ev.goCue(trial)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Spikes are counted into **10 ms** non-overlapping bins spanning [-2.5, 2.5] s from the go cue, giving **500 timepoints**. No rebinning is applied after the initial binning.

ii.
```python
DT = 0.01  # 10 ms
TMIN = -2.5
TMAX = 2.5
edges = np.arange(TMIN, TMAX + DT / 2, DT)
taxis = (edges[:-1] + edges[1:]) / 2.0  # 500 bin centres
```

iii. The AI chose `dt = 1/100` (10 ms) because this is the value used by every published figure script. `getDefaultParams.m` has `1/200` (5 ms) but this is unused.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This input is the bin centres of the time axis, defined as the midpoints of the 10 ms bins from -2.5 to 2.5 s.

ii.
```python
edges = np.arange(TMIN, TMAX + DT / 2, DT)
taxis = (edges[:-1] + edges[1:]) / 2.0
input_trial = taxis.astype(np.float32)[None, :]
```

iii. The time axis is defined by the processing parameters, not derived from raw data variables.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing beyond defining the bin centres from the grid parameters.

ii. N/A (see 3-a).

iii. N/A.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is the bin-centre grid itself. Spikes are counted into the same bins, so bin k of the neural data and bin k of the input correspond to the same time interval.

ii.
```python
input_trial = taxis.astype(np.float32)[None, :]  # same taxis used for spike binning
```

iii. By construction, both share the same time axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial fields of `obj.bp`: `L` (left-instructed), `R` (right-instructed), `hit` (correct response), and `miss` (incorrect response).

ii.
```python
lick_dir = np.full(n, -1, dtype=np.int64)
lick_dir[(bp['L'] & bp['hit']) | (bp['R'] & bp['miss'])] = 0      # licked left
lick_dir[(bp['R'] & bp['hit']) | (bp['L'] & bp['miss'])] = 1      # licked right
lick_dir[bp['no']] = 2                                            # no response
```

iii. This matches the reference `getPrevChoice.m`.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A hit means the animal licked the instructed port; a miss means it licked the other port; `no` means no response. Left = `(L & hit) | (R & miss)` = 0, right = `(R & hit) | (L & miss)` = 1, none = `no` = 2. The per-trial value is broadcast across all time bins.

ii. (see 4-a code)

iii. Follows the reference's `getPrevChoice.m` logic.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The per-trial field `obj.bp.autowater`.

ii.
```python
context = np.where(bp['autowater'], 0, 1).astype(np.int64)  # 0 = WC, 1 = DR
```

iii. `autowater = True` marks WC trials; everything else is DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct mapping: `autowater` -> WC (0), not autowater -> DR (1). Broadcast across time bins.

ii. (see 5-a)

iii. Follows the reference's `getBlockNum_AltContextTask.m`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Three per-trial fields: `bp.hit`, `bp.miss`, `bp.no`.

ii.
```python
outcome = np.full(n, -1, dtype=np.int64)
outcome[bp['miss']] = 0
outcome[bp['hit']] = 1
outcome[bp['no']] = 2
```

iii. Follows the reference's `getOutcome.m`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct relabelling: miss -> incorrect (0), hit -> correct (1), no -> ignore (2). Broadcast across time bins.

ii. (see 6-a)

iii. The paper excludes ignore trials; here they are retained as required by the decoder task's "ignore" outcome class.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking from `obj.traj`, specifically the **side camera only** (view index 0), feature `tongue`. The `ts` array holds x, y, and confidence per frame. `frameTimes` provides the video timing, and `sglx.bitcode.bitstart`/`sglx.fs`/`bp.ev.bitStart` provide the video clock offset.

ii.
```python
v1 = f[o['traj'][0, 0]]      # side camera
n1 = feature_names(f, v1)
i_tongue = n1.index('tongue')
...
ts1 = np.array(f[v1['ts'][tr, 0]])
```

iii. The AI uses only the side camera for tongue tracking, following the reference's `findPosition.m` which operates on one view at a time.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The tongue x and y positions are **interpolated** onto the 10 ms time grid using `np.interp` (linear interpolation). Frames where DLC did not detect the tongue (x = NaN) are marked as not visible. Velocity is computed as `np.gradient` of the `nearest_fill`-ed interpolated position, and speed is the magnitude `np.hypot(vx, vy)`. The speed is then discretized at the per-session 50th percentile of visible time points.

ii.
```python
x = resample(ts1[i_tongue, 0])  # np.interp onto taxis
y = resample(ts1[i_tongue, 1])
vis = ~np.isnan(x)
if vis.any():
    vx = np.gradient(nearest_fill(x))
    vy = np.gradient(nearest_fill(y))
    tongue_speed[:, tr] = np.hypot(vx, vy)
tongue_vis[:, tr] = vis
```

```python
def discretize(values, visible, keep_trials):
    thresh = np.percentile(vals[vis], 50)
    out[vis & (vals < thresh)] = 0
    out[vis & (vals >= thresh)] = 1
    # non-visible = 2
```

iii. The AI's approach follows `findPosition.m` (interpolation) and `findVelocity.m` (gradient of interpolated position), but only for the side camera. Tongue NaN values are not nearest-filled (matching the reference's special handling of the tongue feature).

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per-session 50th percentile of the **visible** time points of **kept trials**. Below threshold = 0, at/above threshold = 1, not visible = 2.

ii. (see `discretize` in 7-b)

iii. The 50th percentile split follows the task instructions.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The video clock is aligned to the behavior clock using `findVideoOffset`: `vidshift = mode(sglx.bitcode.bitstart)/fs - mode(bp.ev.bitStart)`. Frame times are then `frameTimes - vidshift - goCue(trial)`, and positions are interpolated onto the same 10 ms time axis as the neural data.

ii.
```python
def video_offset(f, o, bp):
    fs = float(np.array(o['sglx']['fs'])[0, 0])
    bitstart = np.array(o['sglx']['bitcode']['bitstart']).flatten()
    return mode_value(bitstart) / fs - mode_value(bp['bitStart'])
...
def resample(y, times=None):
    return np.interp(taxis, times[:k], y[:k], left=np.nan, right=np.nan)
```

iii. Matches the reference's `findVideoOffset.m`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The DeepLabCut tracking from the **bottom camera** (view index 1), using **both** `top_paw` and `bottom_paw` features.

ii.
```python
v2 = f[o['traj'][1, 0]]      # bottom camera
n2 = feature_names(f, v2)
i_paws = [n2.index(p) for p in ('top_paw', 'bottom_paw') if p in n2]
```

iii. The AI uses both paw markers from the bottom camera, averaged together.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Each paw's x and y positions are interpolated onto the 10 ms grid, nearest-filled, and the velocity is computed as `gradient()` minus the baseline drift `median(diff(pos))` (matching `findVelocity.m`). Speed is `hypot(vx, vy)`. The two paw speeds are averaged (nanmean). Discretized at the per-session 50th percentile of visible time points.

ii.
```python
pxf, pyf = nearest_fill(px), nearest_fill(py)
vx = np.gradient(pxf) - np.median(np.diff(pxf))
vy = np.gradient(pyf) - np.median(np.diff(pyf))
s = np.hypot(vx, vy)
...
# average over both paws
paw_speed[:, tr] = np.where(cnt > 0, np.nansum(sm, axis=0) / np.maximum(cnt, 1), np.nan)
```

iii. Follows `findPosition.m` (nearest-fill for non-tongue features) and `findVelocity.m` (baseline drift removal). Uses both paw markers averaged.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: per-session 50th percentile of visible time points. 0 = below, 1 = at/above, 2 = not visible.

ii. (see `discretize` function)

iii. Follows the task instructions.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same video offset correction as tongue, interpolated onto the same 10 ms time axis, using the bottom camera's frame times.

ii. (see 7-d for video offset; interpolation uses `resample` with `tt2` = bottom camera frame times)

iii. Same alignment as all video-derived features.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Loaded from the separate `motionEnergy_<ANM>_<DATE>.mat` file. Contains one trace per trial at the camera frame rate (~400 Hz).

ii.
```python
def load_motion_energy(me_path):
    m = sio.loadmat(me_path, struct_as_record=False, squeeze_me=False)
    me = m['me'][0, 0]
    data = me.data
    if not isinstance(data, np.ndarray) or data.dtype != object:
        data = data[0, 0].data
    cells = [np.asarray(data[i, 0]).flatten() for i in range(data.shape[0])]
    ...
```

iii. Uses the separate file, matching the reference's `loadMotionEnergy.m`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy values are **interpolated** onto the 10 ms time axis using `np.interp`, then **nearest-filled** (`fillmissing('nearest')`) to cover the few samples at trial edges not covered by video. Discretized at the per-session 50th percentile.

ii.
```python
me[:, tr] = nearest_fill(
    np.interp(taxis, tt[:k], mv[:k], left=np.nan, right=np.nan))
```

iii. Matches `loadMotionEnergy.m`: `interp1` then `fillmissing(...,'nearest')`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session 50th percentile. 0 = below, 1 = at/above, 2 = no video (trial-level, only for trials without video, which are dropped).

ii.
```python
me_vis = ~np.isnan(kin['me']) & kin['has_video'][None, :]
me_cat, me_thresh = discretize(kin['me'], me_vis, keep_idx)
```

iii. Since trials without video are dropped, class 2 has zero instances in the final data.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same video offset and interpolation as tongue/paw. Motion energy uses the side camera frame times.

ii.
```python
me[:, tr] = nearest_fill(
    np.interp(taxis, tt[:k], mv[:k], left=np.nan, right=np.nan))
```

iii. Same alignment approach as all video-derived features.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases: (1) Trials with no usable video (`frameTimes` empty or all NaN) are detected and the entire trial is dropped from the dataset. (2) DeepLabCut NaN positions (tongue not visible) are preserved as NaN and become class 2 ("not visible") after discretization. (3) Motion energy at trial edges not covered by video is nearest-filled, matching `loadMotionEnergy.m`. (4) Sessions without `obj.bp.stim` are handled with try/except. (5) Bottom camera frame times used independently from side camera (different frame counts in some trials). (6) NaN `goCue` trials are excluded.

ii.
```python
try:
    out['stim'] = np.array(bp['stim']['enable']).flatten()[:n].astype(bool)
except (KeyError, TypeError):
    out['stim'] = np.zeros(n, bool)
...
if tt is None:
    continue
has_video[tr] = True
```

iii. The AI documented these edge cases in detail in CONVERSION_NOTES.md Step 10, Check 5.

## 11-a. What are the most time-consuming steps of the code?

i. The AI reports the kinematics/motion energy extraction (1.3-2.2 s per session, dominated by HDF5 reads of DLC trajectories) as the bottleneck, followed by spike binning (0.1-0.35 s) and smoothing (0.07-0.32 s). Total for all 12 sessions: ~25 s.

ii.
```python
timing['kinematics'] = time.time() - t0  # reported as the slowest step
```

iii. The AI timed each processing step and documented this in Step 7 of CONVERSION_NOTES.md.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop is over trials for kinematics extraction (`for tr in range(n):`), which processes each trial's DLC tracking individually. This is difficult to vectorize because each trial has a different number of frames. The spike binning uses `np.histogram2d` which is already vectorized over trials. The smoothing is done in one call on a reshaped matrix.

ii.
```python
for tr in range(n):  # per-trial kinematic extraction
    tt = trial_frame_times(f, v1, tr, vidshift, bp['goCue'][tr])
    ...
```

iii. The AI noted that per-trial loops remain because trials have different numbers of camera frames, so there is no rectangular array to operate on.

## 11-c. What processing does the code repeat multiple times?

i. Frame times and DLC data are read for each trial. The same `f[v1['ts'][tr, 0]]` and `f[v1['frameTimes'][tr, 0]]` arrays are read once per trial. Within a trial, the side camera data is read for both tongue and motion energy alignment. The AI does not appear to cache the side camera frame times across these uses within the same trial.

ii.
```python
# Inside extract_kinematics, for each trial:
tt = trial_frame_times(f, v1, tr, vidshift, bp['goCue'][tr])  # side camera times
# Later in the same trial:
# motion energy also uses side camera times via tt
```

iii. The AI mitigated this by reading all features from the same `ts` array read per trial.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several pieces: (1) The `nearest_fill` function is computed for tongue positions even though the tongue's nearest-filled values are only used to compute velocity which is then masked by visibility anyway. (2) The `extras` dict returned by `convert_session` contains large intermediate arrays (raw rates, kinematics, etc.) used only for plotting diagnostics and then deleted. (3) The `basederiv` subtraction for paw velocity computation is arguably unnecessary since the median derivative is approximately zero. (4) The `lickL`/`lickR` event times are loaded but only used in diagnostic plots, not in the main conversion.

ii.
```python
out['lickL'] = [np.array(f[r]).flatten() for r in np.array(ev['lickL']).flatten()[:n]]
out['lickR'] = [np.array(f[r]).flatten() for r in np.array(ev['lickR']).flatten()[:n]]
...
extras = dict(bp=bp, kin=kin, ...)  # large intermediate data for diagnostics
```

iii. The AI prioritized matching the reference code's processing over minimizing unnecessary computation.
