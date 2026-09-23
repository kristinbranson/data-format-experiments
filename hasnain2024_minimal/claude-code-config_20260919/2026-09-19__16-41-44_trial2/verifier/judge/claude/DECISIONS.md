# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads only the 25 fixed-delay sessions from `data/Ephys_Behavior/`, excluding the 19 randomized-delay sessions in `data/RandomizedDelay_Ephys_Behavior/`. Each session is loaded via `h5py.File()` directly (HDF5/v7.3 format only). The 25 session names and their probe IDs are hard-coded in the `SESSIONS` list, transcribed from the authors' `load<ANM>_ALMVideo.m` scripts. Motion energy is loaded separately from `motionEnergy_<anm>_<date>.mat` files via `scipy.io.loadmat`.

ii.
```python
DATA_DIR = '/app/data/Ephys_Behavior'

SESSIONS = [
    ('EKH1',  '2021-08-07', [2]),
    ('EKH3',  '2021-08-11', [2]),
    ...
    ('JGR3',  '2021-11-18', [1]),
]

def convert_session(anm, date, probes, verbose=True):
    path = os.path.join(DATA_DIR, f'data_structure_{anm}_{date}.mat')
    f = h5py.File(path, 'r')
    obj = f['obj']
```

iii. The agent explicitly chose to exclude the randomized-delay dataset, reasoning that "the delay varies 0.3-3.6 s, so the pre-go-cue window would contain a different mixture of task epochs across trials than in the fixed-delay sessions, and those mice never experienced the WC context." The agent also noted that the optogenetic directories have no neural data. The agent explored the data loading scripts and file structure extensively before settling on this list.

## 1-b. How are the data split into subjects?

i. The subject (mouse) is the first element of each session tuple in `SESSIONS` (e.g., `'EKH1'`). The unique sorted subjects are collected at the end and `subject_idx` maps each session to its subject.

ii.
```python
subjects = sorted({s['session_info']['subject'] for s in sessions})
subject_idx = np.array([subjects.index(s['session_info']['subject'])
                        for s in sessions], dtype=np.int64)
```

iii. The subject ID is part of the session filename and is stored explicitly in the `SESSIONS` list.

## 1-c. How are the data split into sessions?

i. Each entry in the `SESSIONS` list is one session, identified by `(anm, date, probes)`. Only the 25 fixed-delay sessions from `Ephys_Behavior/` are included. Each session produces one element in the `neural`, `input`, and `output` lists. Sessions that yield no valid units or fewer than 2 trials are skipped.

ii.
```python
for anm, date, probes in SESSIONS:
    s = convert_session(anm, date, probes)
    if s is None:
        print(f'skipping {anm}_{date}')
        continue
    sessions.append(s)
```

iii. The agent transcribed the session list from the authors' loading scripts, focusing on the fixed-delay experiment.

## 1-d. How are the data split into trials?

i. Trials are defined by the behavioral protocol fields in `obj.bp`. The number of trials is `bp.Ntrials`. Each behavioral variable is read as a vector of length `Ntrials`. Trial indices are 1-based MATLAB IDs, converted to 0-based Python indices where needed.

ii.
```python
def load_behavior(f, obj):
    bp = obj['bp']
    ev = bp['ev']
    n = int(vec(bp, 'Ntrials')[0])
    b = {
        'n': n,
        'R': np.nan_to_num(vec(bp, 'R')[:n]) > 0,
        ...
        'goCue': vec(ev, ALIGN_EVENT)[:n],
    }
    return b
```

iii. The trial structure follows the standard Bpod protocol structure, with one entry per trial.

## 1-e. How are trials filtered based on quality controls?

i. Two filters: early-lick trials (`bp.early`) and photostimulation trials (`bp.stim.enable`) are excluded. Additionally, trials with non-finite go cue times are excluded. Hit, miss, and ignore trials are all retained.

ii.
```python
keep = (~b['early']) & (~b['stim']) & np.isfinite(b['goCue'])
trials = np.where(keep)[0] + 1
```

iii. The agent identified that every analysis condition in the paper's scripts uses the `~early & ~stim.enable` mask. The `isfinite(goCue)` check ensures trials with missing alignment events are excluded.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The spike-sorted clusters in `obj.clu`, accessed per-probe. For each cluster, `trialtm` (spike times relative to trial start) and `trial` (trial ID for each spike) are used, along with `quality` for curation. The go cue times from `bp.ev.goCue` provide alignment.

ii.
```python
groups = clu_groups(f, obj)
...
tm = np.array(f[g['trialtm'][icell, 0]]).flatten()
tr = np.array(f[g['trial'][icell, 0]]).flatten().astype(np.int64)
...
aligned = tm - align[tr - 1]
```

iii. The agent explored the cluster structure in the HDF5 files and identified `trialtm` and `trial` as the relevant spike time variables.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 10 ms bins (DT = 1/100), converted to firing rates (spikes/s), then smoothed with a causal Gaussian kernel of 15 bins. The smoothing uses `lfilter` to implement a causal version of `gausswin(15)` with the acausal half zeroed, matching the reference code's `mySmooth.m`. The boundary condition is `'reflect'`.

ii.
```python
DT = 1.0 / 100.0     # params.dt  (10 ms bins)
SMOOTH_N = 15        # params.smooth

def _causal_gauss_kernel(n=SMOOTH_N):
    k = np.arange(n)
    alpha = 2.5
    w = np.exp(-0.5 * (alpha * (2 * k - (n - 1)) / (n - 1)) ** 2)
    w[: n // 2] = 0.0          # causal
    return w / w.sum()

def my_smooth(x):
    if BCTYPE == 'reflect':
        pad = x[:SMOOTH_N]
        xf = np.concatenate([pad, x], axis=0)
        return lfilter(_TAPS, [1.0], xf, axis=0)[SMOOTH_N:]
    return lfilter(_TAPS, [1.0], x, axis=0)
```

The firing rate criterion uses the all-trials PSTH:
```python
psth = my_smooth((counts.sum(axis=0) / len(trials) / DT)[:, None])[:, 0]
if psth.mean() > LOW_FR:
    keep_rates.append(counts)
```

iii. The agent explicitly states in the docstring: "binning + smoothing -> getSeq.m (tmin=-2.5, tmax=2.5, dt=1/100, causal Gaussian kernel of 15 bins, 'reflect' boundary)". The agent ported `mySmooth.m` directly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. First, cluster quality labels matching `{'garbage', 'gabrga', 'noisy', 'real?'}` are excluded (matched case-insensitively). Then units whose mean firing rate (computed on the all-trials PSTH after smoothing) is at or below 1 Hz are removed.

ii.
```python
BAD_QUALITY = {'garbage', 'gabrga', 'noisy', 'real?'}

for icell, q in enumerate(qualities):
    if q.lower() in BAD_QUALITY:
        continue
    ...
    psth = my_smooth((counts.sum(axis=0) / len(trials) / DT)[:, None])[:, 0]
    if psth.mean() > LOW_FR:
        keep_rates.append(counts)
```

iii. The agent cites `findClusters.m` with `params.quality = 'all'` for the quality filter and `removeLowFRClusters.m` with `params.lowFR = 1` for the firing rate criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times (`trialtm`) are aligned to the go cue by subtracting the go cue time for each spike's trial: `aligned = tm - align[tr - 1]`, where `align` is `bp.ev.goCue`.

ii.
```python
align = b['goCue']
...
aligned = tm - align[tr - 1]
ok = (row >= 0) & (aligned >= TMIN) & (aligned < TMAX)
```

iii. This follows `alignSpikes.m` with `params.alignEvent = 'goCue'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 10 ms bins (DT = 1/100 s), yielding 500 bins from -2.5 to +2.5 s. No rebinning is applied; spikes are binned directly at this resolution.

ii.
```python
DT = 1.0 / 100.0     # params.dt  (10 ms bins)
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)          # 501 edges
TAXIS = EDGES[:-1] + DT / 2                          # 500 bin centres
NBINS = len(TAXIS)
```

iii. The agent references `params.dt = 1/100` from `getDefaultParams.m`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the time axis itself -- the bin centres of the time grid, defined analytically from TMIN, TMAX, and DT. It is not derived from any raw data variable.

ii.
```python
time_input = TAXIS.astype(np.float32)[None, :]
```

iii. The time axis is constructed to match the neural binning grid.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing. The bin centres are computed as `EDGES[:-1] + DT / 2` and used directly.

ii.
```python
TAXIS = EDGES[:-1] + DT / 2
```

iii. The time input is purely definitional.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is the bin centres of the same time grid used for the neural data, so they are inherently aligned.

ii.
```python
time_input = TAXIS.astype(np.float32)[None, :]
```

iii. Alignment is guaranteed by construction since both use the same time axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Derived from `bp.R`, `bp.L`, `bp.hit`, and `bp.miss`. The instructed side (R or L) combined with the outcome (hit or miss) determines which direction the animal licked.

ii.
```python
licked_right = (b['R'][idx] & b['hit'][idx]) | (b['L'][idx] & b['miss'][idx])
licked_left = (b['L'][idx] & b['hit'][idx]) | (b['R'][idx] & b['miss'][idx])
```

iii. The lick direction is not directly recorded, so it must be inferred from the instructed side and outcome.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Three classes: left (0), right (1), none (2). A hit on a right-instructed trial or a miss on a left-instructed trial means the animal licked right, and vice versa. Ignore trials get the "none" class.

ii.
```python
lick = np.full(len(idx), 2, dtype=np.int8)                     # 2 = none
licked_right = (b['R'][idx] & b['hit'][idx]) | (b['L'][idx] & b['miss'][idx])
licked_left = (b['L'][idx] & b['hit'][idx]) | (b['R'][idx] & b['miss'][idx])
lick[licked_left] = 0
lick[licked_right] = 1
```

iii. The logic is equivalent to the reference's approach of combining instructed side with outcome.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derived from `bp.autowater`. Autowater trials are the water-cued (WC) context; non-autowater trials are the delayed-response (DR) context.

ii.
```python
context = np.where(b['autowater'][idx], 0, 1).astype(np.int8)  # 0 = WC, 1 = DR
```

iii. Read directly from the trial table.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct relabelling: autowater=True -> WC (0), autowater=False -> DR (1).

ii.
```python
context = np.where(b['autowater'][idx], 0, 1).astype(np.int8)
```

iii. Straightforward mapping matching the instruction's WC/DR encoding.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `bp.hit` and `bp.miss`. Trials that are neither hit nor miss are classified as ignore.

ii.
```python
outcome = np.full(len(idx), 2, dtype=np.int8)                  # 2 = ignore
outcome[b['miss'][idx]] = 0                                    # 0 = incorrect
outcome[b['hit'][idx]] = 1                                     # 1 = correct
```

iii. The three-class scheme follows the instructions: incorrect (0), correct (1), ignore (2).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A relabelling: miss -> incorrect (0), hit -> correct (1), everything else -> ignore (2).

ii.
```python
outcome = np.full(len(idx), 2, dtype=np.int8)
outcome[b['miss'][idx]] = 0
outcome[b['hit'][idx]] = 1
```

iii. Direct mapping from the behavioral flags.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`, specifically the side camera's `tongue` feature (view index 1 = side camera). The x,y positions and `frameTimes` are used, along with the video offset from `sglx.bitcode`.

ii.
```python
TONGUE_FEATURE = ('tongue', 1)
...
pos, has_video = load_traj(f, obj, trials, align, vidshift)
tspeed, tvis = tongue_speed(pos[TONGUE_FEATURE[0]])
```

iii. The agent uses only the side camera's tongue marker, not both cameras.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The tongue x,y positions are interpolated onto the time axis (TAXIS) using linear interpolation. NaN positions (tongue not visible) are filled with the session's mean lick-onset position (matching `setTongueBaselinePosition()` in `getKinematicsFromVideo.m`). Then `np.gradient` is applied to compute velocity, and the speed (magnitude) is computed. The speed is discretized at the session's 50th percentile into two classes, with a third class for frames where the tongue was not visible.

ii.
```python
def tongue_speed(pos):
    visible = np.isfinite(pos[..., 0]) | np.isfinite(pos[..., 1])
    starts_x, starts_y = [], []
    for i in range(pos.shape[0]):
        for s in runs_of_true(visible[i]):
            starts_x.append(pos[i, s, 0])
            starts_y.append(pos[i, s, 1])
    mux = np.nanmean(starts_x) if starts_x else 0.0
    muy = np.nanmean(starts_y) if starts_y else 0.0
    filled = pos.copy()
    filled[..., 0] = np.where(np.isfinite(filled[..., 0]), filled[..., 0], mux)
    filled[..., 1] = np.where(np.isfinite(filled[..., 1]), filled[..., 1], muy)
    vx = np.gradient(filled[..., 0], axis=1)
    vy = np.gradient(filled[..., 1], axis=1)
    return np.sqrt(vx ** 2 + vy ** 2), visible
```

iii. The agent explicitly states this follows `setTongueBaselinePosition()` and `findVelocity.m`.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The session's 50th percentile of valid tongue speed values is computed, and values are discretized: 0 = below threshold, 1 = at or above threshold, 2 = not visible (tongue not tracked).

ii.
```python
def discretize(values, valid, missing_code=2):
    out = np.full(values.shape, missing_code, dtype=np.int8)
    v = values[valid]
    v = v[np.isfinite(v)]
    thresh = np.percentile(v, 50)
    finite = valid & np.isfinite(values)
    out[finite & (values < thresh)] = 0
    out[finite & (values >= thresh)] = 1
    return out
```

iii. Follows the instructions' discretization scheme.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by the video offset (from `findVideoOffset.m`) and the trial's go cue time, then the tracking positions are linearly interpolated onto the same TAXIS used by the neural data.

ii.
```python
t_src = ft - vidshift - align[j]
pos[name][i] = interp_to_taxis(t_src, xy, TAXIS)
```

iii. The video offset is computed from the bitcode synchronization signal, matching `findVideoOffset.m`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Both bottom-camera paw markers: `top_paw` and `bottom_paw` from view 2 (bottom camera).

ii.
```python
PAW_FEATURES = [('top_paw', 2), ('bottom_paw', 2)]
...
pspeed, pvis = paw_speed([pos[n] for n, _ in PAW_FEATURES])
```

iii. The agent uses both paw markers and averages their speeds.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw marker, positions are filled with nearest-neighbor interpolation (`fill_nearest`), and the per-trial median frame-to-frame displacement is subtracted (drift removal, following `findPosition.m`/`findVelocity.m`). Speed is computed as the magnitude of the gradient. The two paw speeds are averaged where both are visible. Discretized at the session's 50th percentile.

ii.
```python
def paw_speed(pos_list):
    for pos in pos_list:
        visible = np.isfinite(pos[..., 0]) | np.isfinite(pos[..., 1])
        for i in range(pos.shape[0]):
            if not visible[i].any():
                continue
            x = fill_nearest(pos[i, :, 0])
            y = fill_nearest(pos[i, :, 1])
            vx = np.gradient(x) - np.nanmedian(np.diff(x))
            vy = np.gradient(y) - np.nanmedian(np.diff(y))
            sp[i] = np.sqrt(vx ** 2 + vy ** 2)
    speeds = np.stack(speeds)
    mean_speed = np.nansum(...) / np.maximum(visibles.sum(axis=0), 1)
    return mean_speed, any_visible
```

iii. The agent cites `findPosition.m` and `findVelocity.m` for the nearest-fill and median-drift-removal approach.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: session's 50th percentile threshold, with 0 = below, 1 = at or above, 2 = not visible.

ii.
```python
paw_code = discretize(pspeed, pvis)
```

iii. Follows the instructions' discretization scheme.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: frame times corrected by video offset and go cue, then interpolated onto TAXIS.

ii.
```python
t_src = ft - vidshift - align[j]
pos[name][i] = interp_to_taxis(t_src, xy, TAXIS)
```

iii. Same alignment mechanism as all video-derived outputs.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The `motionEnergy_<anm>_<date>.mat` files. The per-trial motion energy trace is read from these files (one value per camera frame).

ii.
```python
def load_motion_energy(anm, date, f, obj, trials, align, vidshift):
    fn = os.path.join(DATA_DIR, f'motionEnergy_{anm}_{date}.mat')
    m = scipy.io.loadmat(fn, struct_as_record=False, squeeze_me=False)
    me = m['me'][0, 0]
    data = me.data
    while hasattr(data, '_fieldnames') or ...:
        data = data.data ...
```

iii. The motion energy files are loaded via scipy.io for the v5 MATLAB format.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The per-frame motion energy values are interpolated onto TAXIS using linear interpolation, then missing values are filled with nearest-neighbor. When frame times don't match the motion energy trace length, a fallback of 400 Hz with 0.5 s offset is used. Discretized at the session's 50th percentile.

ii.
```python
out[i] = interp_to_taxis(t_src, y, TAXIS)
out[i] = fill_nearest(out[i])
```

iii. The fallback timing follows `loadMotionEnergy.m`'s catch branch.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same percentile-based discretization: 0 = below 50th percentile, 1 = at or above, 2 = no video.

ii.
```python
me_code = discretize(me, np.isfinite(me))
```

iii. Follows the instructions' discretization scheme.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy uses the side camera's frame times, corrected by the video offset and go cue, then interpolated onto TAXIS. Fallback timing (400 Hz, 0.5 s offset) is used when frame times don't match.

ii.
```python
if ft.size != y.size or not np.isfinite(ft).any():
    t_src = np.arange(1, y.size + 1) / 400.0 - 0.5 - align[j]
else:
    t_src = ft - vidshift - align[j]
out[i] = interp_to_taxis(t_src, y, TAXIS)
```

iii. The fallback matches `loadMotionEnergy.m`.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) Trials with fewer than 10 frame times or all-NaN frame times are skipped for video data. (2) Trials flagged with `NdroppedFrames = NaN` are skipped. (3) Tongue NaN positions are filled with the session's mean lick-onset position. (4) Paw NaN positions are filled with nearest-neighbor interpolation. (5) Motion energy missing values are filled with nearest-neighbor. (6) `np.nan_to_num` is used when reading behavioral variables.

ii.
```python
if ft.size < 10 or not np.isfinite(ft).any():
    continue
try:
    nd = np.array(f[v0['NdroppedFrames'][j, 0]]).flatten()
    if nd.size and np.isnan(nd[0]):
        continue
except Exception:
    pass
```

iii. The filling strategies follow the reference MATLAB code's approach: `setTongueBaselinePosition()` for tongue, `fillmissing(..., 'nearest')` for paw/motion energy.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the HDF5 files via `h5py.File` and reading the cluster data (spike times, trial IDs, quality labels) for each session. The neural matrix computation (binning and smoothing across all clusters) is also time-consuming.

ii.
```python
f = h5py.File(path, 'r')
...
tm = np.array(f[g['trialtm'][icell, 0]]).flatten()
tr = np.array(f[g['trial'][icell, 0]]).flatten().astype(np.int64)
```

iii. HDF5 dereferencing of object arrays requires many small reads.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-cluster loop in `neural_matrix` processes each cluster individually with `np.bincount`. The per-trial loops in `load_traj` and `load_motion_energy` iterate over trials for interpolation. The paw speed computation loops over trials for nearest-fill and gradient computation.

ii.
```python
for icell, q in enumerate(qualities):
    ...
    counts = np.bincount(row * NBINS + b, minlength=len(trials) * NBINS).reshape(...)
```

iii. The trial loop is harder to vectorize because each trial may have different numbers of frames.

## 11-c. What processing does the code repeat multiple times?

i. Frame times are read multiple times for the same trial -- once in `load_traj` for DLC features and once in `load_motion_energy`. The smoothing in `neural_matrix` is applied to the all-trials PSTH for the firing rate criterion, then applied again to the full per-trial rates.

ii.
```python
# In neural_matrix:
psth = my_smooth((counts.sum(axis=0) / len(trials) / DT)[:, None])[:, 0]
...
flat = my_smooth(flat)
```

iii. The PSTH smoothing is done per-cluster to decide whether to keep it, then the full matrix is smoothed again.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The behavioral fields `sample` and `delay` event times are loaded but never used. (2) The `NdroppedFrames` check adds processing for a rarely-triggered edge case. (3) The `fill_nearest` on motion energy fills NaN values that may then be marked as "no video" class 2 anyway in the discretization step (though the valid mask is `np.isfinite(me)` which would be True after filling).

ii.
```python
'sample': vec(ev, 'sample')[:n],
'delay': vec(ev, 'delay')[:n],
```

iii. These event times were likely loaded for potential use but are not referenced in the output construction.
