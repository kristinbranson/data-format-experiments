# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses `discover_sessions()` to glob all `data_structure_*.mat` and `motionEnergy_*.mat` files from subdirectories of the `data/` folder, building a dictionary keyed by session stem. For HDF5 files, it uses a custom `load_data_structure()` function with `h5py`. For motion energy files, it uses `scipy.io.loadmat`. However, it can only read HDF5 (v7.3) MAT files for data structures — when `h5py.File()` raises `OSError`, it returns `None` and skips the session. This causes it to skip 11 sessions that are stored in v5 MAT format.

ii.
```python
def discover_sessions(data_root):
    sessions = {}
    for subdir in Path(data_root).iterdir():
        if not subdir.is_dir():
            continue
        for f in subdir.glob('data_structure_*.mat'):
            stem = f.stem.replace('data_structure_', '')
            sessions.setdefault(stem, {})['data_structure'] = f
        for f in subdir.glob('motionEnergy_*.mat'):
            stem = f.stem.replace('motionEnergy_', '')
            sessions.setdefault(stem, {})['motion_energy'] = f
    return sessions

def load_data_structure(path):
    out = {}
    try:
        hfile = h5py.File(path, 'r')
    except OSError:
        return None
    ...
```

iii. The AI's CONVERSION_NOTES.md states: "Mixed MATLAB formats are present: `data_structure_*.mat` are MATLAB v7.3/HDF5, while at least `motionEnergy_*.mat` are older MAT files readable with `scipy.io.loadmat`." The AI assumed all data_structure files were HDF5 and did not implement a fallback to `scipy.io.loadmat` for v5 files.

## 1-b. How are the data split into subjects?

i. The subject is extracted from the session stem by splitting on underscore and taking the first part, same approach as the reference. Subjects are collected as they appear during processing.

ii.
```python
def infer_subject_session(stem):
    parts = stem.split('_')
    return parts[0], '_'.join(parts[1:])
```

iii. The AI identified subject from the filename pattern. This matches the reference approach.

## 1-c. How are the data split into sessions?

i. Sessions are discovered by globbing the data directory for files matching `data_structure_*.mat`. All discovered sessions with both a data_structure and motionEnergy file are processed. This differs from the reference, which uses a hard-coded session list from the authors' loading scripts. The AI processes sessions from all four subdirectories (including `DelayInhibition_BilatMC_Behavior` and `GoCueInhibition_BilatMC_Behavior`), though these may not have matching files. The AI ends up with 33 sessions after skipping 11 unreadable v5 files and 1 session with missing spike fields.

ii.
```python
sessions = discover_sessions('data')
keys = sorted(k for k,v in sessions.items() if 'data_structure' in v and 'motion_energy' in v)
```

iii. The AI chose discovery over a hard-coded list. Its CONVERSION_NOTES.md does not explain this choice.

## 1-d. How are the data split into trials?

i. Trials are identified by the `Ntrials` field in `bp`. A boolean mask is created for valid trials. Each trial's spikes are extracted by matching `clu.trial` indices. Per-trial behavioral labels and trajectories are indexed by the raw trial index.

ii.
```python
def valid_trials(bp):
    n = int(np.array(bp['Ntrials']).reshape(-1)[0])
    mask = np.ones(n, dtype=bool)
    for name in ['early', 'no']:
        if name in bp:
            arr = np.array(bp[name], dtype=float).reshape(-1)
            if arr.size == n:
                mask &= (arr == 0)
    return mask
```

iii. The AI uses `bp.Ntrials` to determine total trial count and iterates over valid trial indices.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters out early-lick trials (`bp.early`) and ignore trials (`bp.no`). It does NOT filter out photostimulation trials (`bp.stim.enable`). The reference filters early-lick and photostimulation trials but keeps ignore trials (assigning them a third outcome class).

ii.
```python
def valid_trials(bp):
    n = int(np.array(bp['Ntrials']).reshape(-1)[0])
    mask = np.ones(n, dtype=bool)
    for name in ['early', 'no']:
        if name in bp:
            arr = np.array(bp[name], dtype=float).reshape(-1)
            if arr.size == n:
                mask &= (arr == 0)
    return mask
```

iii. The AI's CONVERSION_NOTES.md states "Early lick and ignore trials are omitted from analyses" citing the methods. The paper says early lick trials are omitted; ignore trials are sometimes excluded from specific analyses but the reference keeps them as a third outcome class. Photostim trials should be excluded per the paper.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses `obj.clu.trialtm` (spike times relative to trial start) and `obj.clu.trial` (trial assignment for each spike). It only reads the first probe entry (`obj['clu'][()].reshape(-1)[0]`), missing multi-probe sessions.

ii.
```python
clu_ref = obj['clu'][()].reshape(-1)[0]
clu = deref(h, clu_ref)
clu_out = {}
for name in ['tm', 'site', 'quality', 'trialtm', 'trial']:
    if name in clu:
        ds = clu[name]
        ...
        clu_out[name] = [np.array(deref(h, r)).reshape(-1) for r in ds[()].reshape(-1)]
```

iii. The AI identifies `clu.trialtm` and `clu.trial` as the source variables. It does not discuss multi-probe handling.

## 2-b. How is the `neural` data processed?

i. The AI bins spike times into time bins using `np.histogram` per unit per trial. The data is stored as raw spike counts (`float32`) with NO conversion to firing rates and NO Gaussian smoothing. The reference converts to Hz (dividing by bin width) and applies a 14ms Gaussian smoothing kernel.

ii.
```python
for raw_t in np.where(trial_mask)[0]:
    counts, _ = np.histogram(st[tr == raw_t], bins=edges)
    unit_trials.append(counts.astype(np.float32))
```

iii. The AI's CONVERSION_NOTES.md mentions the reference code uses smoothing but does not apply it. The code simply histograms spike times into bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies only a firing rate filter: it estimates the firing rate as `spike_count / approx_duration` and excludes units with FR <= 1 Hz. It does NOT filter by the manual curation quality labels (`clu.quality`). The reference filters by quality labels (excluding `garbage`, `gabrga`, `noisy`, `real?`, `poor`) AND by mean rate > 1 Hz.

ii.
```python
approx_duration = (edges[-1] - edges[0]) * max(trial_mask.sum(), 1)
fr = st.size / max(approx_duration, 1e-9)
if fr <= 1.0:
    continue
```

iii. The AI reads `clu.quality` from the file but never uses it for filtering. The CONVERSION_NOTES.md mentions "all units with firing rates > 1 Hz were included" from the paper but does not discuss quality label filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI does NOT align spike times to the go cue. It bins raw `clu.trialtm` values directly into a fixed time grid from -2.5 to 2.5s without subtracting the go cue time. The `trialtm` values are relative to trial start, not the go cue, so the neural data is aligned to trial start rather than go cue onset.

ii.
```python
def bin_spikes_for_session(obj, trial_mask, edges):
    ...
    for ui, (st, tr) in enumerate(zip(trialtm, trialid)):
        st = np.array(st, dtype=float).reshape(-1)
        tr = np.array(tr, dtype=float).reshape(-1).astype(int) - 1
        ...
        for raw_t in np.where(trial_mask)[0]:
            counts, _ = np.histogram(st[tr == raw_t], bins=edges)
```

iii. The AI's CONVERSION_NOTES.md says "use go cue as the temporal alignment event" but the code does not subtract `bp.ev.goCue` from spike times. The reference does `trialtm - goCue[trial]`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 75ms bins, resulting in 66 time points per trial. The reference uses 5ms bins (1000 time points per trial). 75ms was chosen based on the DLC decoding scripts which use `rez.binSize = 75`.

ii.
```python
def build_time_grid(bin_size_s=0.075, t_start=-2.5, t_end=2.5):
    edges = np.arange(t_start, t_end + 1e-9, bin_size_s)
    centers = edges[:-1] + bin_size_s / 2
    return edges, centers
```

iii. The AI's CONVERSION_NOTES.md notes: "DLC_ContextDecoding.m explicitly sets `rez.binSize = 75` ms", using this as justification. However, the reference native time resolution is `params.dt = 1/200` = 5ms, and 75ms was only used for the DLC decoding analysis, not as the native data representation.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the bin centers of the time grid, which span -2.5 to 2.5s. It is derived purely from the time grid definition, not from any raw data variable.

ii.
```python
edges, centers = build_time_grid()
inputs = [centers[None, :].astype(np.float32) for _ in idx]
```

iii. This matches the reference approach conceptually — the time axis is constructed, not derived from data.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The bin centers are computed from the time grid edges. No further processing.

ii.
```python
centers = edges[:-1] + bin_size_s / 2
```

iii. Same as reference, though with different bin size (75ms vs 5ms).

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The same time grid edges are used for both spike binning and input construction, so they share the same temporal axis by construction.

ii.
```python
edges, centers = build_time_grid()
# Used for both:
neural, kept_units = bin_spikes_for_session(obj, trial_mask, edges)
inputs = [centers[None, :].astype(np.float32) for _ in idx]
```

iii. Consistent within the AI's framework, though the neural data is not actually aligned to go cue (see 2-d).

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI uses `bp.L` and `bp.R` (instructed lick port indicators) to determine lick direction. It does NOT use `bp.hit` and `bp.miss` to infer actual lick direction. The reference uses `bp.R`, `bp.hit`, and `bp.miss` to derive actual lick direction.

ii.
```python
L = np.array(bp['L'], dtype=float).reshape(-1)
R = np.array(bp['R'], dtype=float).reshape(-1)
lick_dir = np.where(R[idx] > L[idx], 1, 0).astype(np.int64)
```

iii. The AI treats `R > L` as lick direction, which is actually the instructed side, not the direction the animal licked. On miss trials, the animal licks the opposite side.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Binary comparison `R > L` assigns right (1) or left (0). There is no "no lick" class for ignore trials. The reference has 3 classes: left (0), right (1), no lick (2), where the actual lick direction is inferred from hit/miss and instructed side.

ii.
```python
lick_dir = np.where(R[idx] > L[idx], 1, 0).astype(np.int64)
```

iii. The AI's output_values list `['left', 'right']` with only 2 classes. Ignore trials (which the AI excludes) would not need a "no lick" class in the AI's framework, but the AI is also mislabeling miss trials.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The AI uses `bp.autowater` to determine context.

ii.
```python
autowater = np.array(bp['autowater'], dtype=float).reshape(-1)
context = np.where(autowater[idx] > 0, 0, 1).astype(np.int64)
```

iii. Matches the reference: autowater > 0 maps to WC (0), otherwise DR (1).

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct thresholding of `autowater > 0` to assign WC (0) vs DR (1).

ii.
```python
context = np.where(autowater[idx] > 0, 0, 1).astype(np.int64)
```

iii. Matches the reference approach.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The AI uses `bp.hit` and `bp.miss`.

ii.
```python
hit = np.array(bp['hit'], dtype=float).reshape(-1)
miss = np.array(bp['miss'], dtype=float).reshape(-1)
outcome = np.where(hit[idx] > miss[idx], 1, 0).astype(np.int64)
```

iii. The AI uses hit and miss as the source, same as the reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI uses `hit > miss` to assign correct (1) or incorrect (0). There is NO ignore class (2) — ignore trials are excluded entirely. The reference keeps ignore trials as a third class.

ii.
```python
outcome = np.where(hit[idx] > miss[idx], 1, 0).astype(np.int64)
```

iii. The AI's output_values list `['incorrect', 'correct']` has only 2 classes. Since the AI excludes ignore trials entirely, the binary encoding is internally consistent but differs from the reference which uses 3 classes including ignore.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The AI uses `traj[0]` (side camera) tracking data: `ts` (trajectory coordinates) and `frameTimes`. It identifies the tongue feature by name from `featNames`. Only the side camera is used, unlike the reference which uses both cameras.

ii.
```python
tongue_idx = pick_feature(side_names, ['tongue', 'left_tongue', 'right_tongue'])
...
ts_side = side['ts'][tr]
ft_side = side['frameTimes'][tr]
tongue.append(interp_feature_velocity(ts_side, ft_side, go[tr], centers, tongue_idx, tongue=True))
```

iii. The AI uses only the side camera view for tongue tracking. The reference uses both side and bottom cameras, normalizes each by its 90th percentile, and averages them.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI interpolates x,y coordinates to the bin centers, then computes the gradient of the interpolated positions to get velocity. NaN values in the velocity are replaced with 0 (for tongue). There is no likelihood filtering, no per-run smoothing, and no normalization. The result is thresholded at the session median into 2 classes (low/high) with no "not visible" class.

ii.
```python
def interp_feature_velocity(ts, frame_times, align_time, centers, feat_idx, tongue=False):
    ...
    xy = arr[:, :2, feat_idx]
    ...
    rel_t = (ft - 0.5) - float(align_time)
    x = np.interp(centers, rel_t, xy[:, 0], left=np.nan, right=np.nan)
    y = np.interp(centers, rel_t, xy[:, 1], left=np.nan, right=np.nan)
    xv = np.gradient(x)
    yv = np.gradient(y)
    if tongue:
        xv[np.isnan(xv)] = 0
        yv[np.isnan(yv)] = 0
    return np.sqrt(xv**2 + yv**2).astype(np.float32)
```

Discretization:
```python
tongue_disc = (tongue > np.nanmedian(tongue)).astype(np.int64)
```

iii. The AI chose interpolation to bin centers rather than frame-level velocity computation. No likelihood filtering is applied. The reference applies a 0.9 likelihood cut, Gaussian smoothing within contiguous valid runs, and handles "not visible" as a third class.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Thresholded at the session median (`np.nanmedian`) into 2 classes: low (0) and high (1). No "not visible" class (2) as specified in the instructions. The reference uses the 50th percentile with 3 classes including "not visible".

ii.
```python
tongue_disc = (tongue > np.nanmedian(tongue)).astype(np.int64)
```

iii. The instructions explicitly specify 3 classes: 0 (< 50th percentile), 1 (>= 50th percentile), 2 (not visible). The AI only implements 2 classes.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI uses `(frameTimes - 0.5) - goCue` to compute relative frame times, then interpolates to the bin centers. The subtraction of 0.5 is unexplained and does not correspond to the bitcode-based video offset correction used in the reference.

ii.
```python
rel_t = (ft - 0.5) - float(align_time)
x = np.interp(centers, rel_t, xy[:, 0], left=np.nan, right=np.nan)
```

iii. The reference computes the video offset from bitcode pulse times (`sglx.bitcode.bitstart / sglx.fs - bp.ev.bitStart`), which is a session-specific constant typically around 0.5s but varying. The AI uses a fixed 0.5s offset, which is an approximation.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The AI uses `traj[1]` (bottom camera) tracking data for the paw, looking for features named `top_paw`, `bottom_paw`, or `paw`.

ii.
```python
paw_idx = pick_feature(bottom_names, ['top_paw', 'bottom_paw', 'paw'])
...
ts_bot = bottom['ts'][tr]
ft_bot = bottom['frameTimes'][tr]
paw.append(interp_feature_velocity(ts_bot, ft_bot, go[tr], centers, paw_idx, tongue=False))
```

iii. The feature selection matches the reference's preference for `top_paw` from the bottom camera.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same interpolation approach as tongue: interpolate x,y to bin centers, compute gradient, compute speed magnitude. For paw (tongue=False), NaN values are replaced with the nanmedian, and the velocities are median-subtracted. No likelihood filtering, no per-run smoothing.

ii.
```python
def interp_feature_velocity(ts, frame_times, align_time, centers, feat_idx, tongue=False):
    ...
    if np.all(np.isnan(xv)) or np.all(np.isnan(yv)):
        return np.zeros(len(centers), dtype=np.float32)
    xv = np.where(np.isnan(xv), np.nanmedian(xv), xv)
    yv = np.where(np.isnan(yv), np.nanmedian(yv), yv)
    xv = xv - np.nanmedian(xv)
    yv = yv - np.nanmedian(yv)
    return np.sqrt(xv**2 + yv**2).astype(np.float32)
```

iii. The median subtraction of velocity components is not done in the reference and alters the velocity semantics. The reference computes speed within contiguous tracked runs after Gaussian smoothing.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Thresholded at the session median into 2 classes (low/high). No "not visible" class as specified in the instructions.

ii.
```python
paw_disc = (paw > np.nanmedian(paw)).astype(np.int64)
```

iii. Same issue as tongue: instructions specify 3 classes but only 2 are implemented.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same approach as tongue: `(frameTimes - 0.5) - goCue`, then interpolation to bin centers. Same fixed 0.5s offset issue.

ii.
```python
rel_t = (ft - 0.5) - float(align_time)
```

iii. Same concern as 7-d about the fixed offset.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI loads motion energy from `motionEnergy_*.mat` files using `scipy.io.loadmat`. The `me` struct is unwrapped through nested dict/array structures to get per-trial arrays.

ii.
```python
def load_motion_energy(path):
    return sio.loadmat(path, simplify_cells=True)['me']

def normalize_motion_energy_data(me):
    data = me
    if isinstance(data, dict):
        data = data.get('data', data)
    ...
```

iii. The loading approach handles the various wrapper formats, similar to the reference.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI rebins per-trial motion energy traces to the target number of bins using linear interpolation (`np.interp` with `np.linspace`). This does not use frame timestamps for temporal alignment — it simply stretches/compresses each trial's trace to fit the bin count. Thresholded at session median into 2 classes.

ii.
```python
def rebin_variable_trace(values, n_bins):
    arr = np.array(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.zeros(n_bins, dtype=np.float32)
    xp = np.linspace(0, 1, arr.size)
    xnew = np.linspace(0, 1, n_bins)
    return np.interp(xnew, xp, arr).astype(np.float32)
```

iii. The reference uses frame timestamps aligned to the go cue via the bitcode offset, then bins into the 5ms grid. The AI's approach ignores temporal structure entirely, stretching whatever data exists to fill the time window.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Thresholded at the session median into 2 classes. No "no video" class (2) as specified in the instructions.

ii.
```python
me_disc = (me_stack > np.nanmedian(me_stack)).astype(np.int64)
```

iii. Instructions specify 3 classes (0: < 50th, 1: >= 50th, 2: no video). Only 2 are implemented.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is NOT properly aligned to the go cue. The `rebin_variable_trace` function uses `np.linspace(0, 1, arr.size)` to normalize the trace length, then interpolates to the target bins. No frame timestamps or go cue alignment is used.

ii.
```python
xp = np.linspace(0, 1, arr.size)
xnew = np.linspace(0, 1, n_bins)
return np.interp(xnew, xp, arr).astype(np.float32)
```

iii. The reference computes frame times in go-cue-relative coordinates using the bitcode offset, then bins frame values into the 5ms grid. The AI's approach assumes the trace covers the full trial window, which is not guaranteed.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases: (1) Sessions with unreadable data_structure files are skipped entirely. (2) Sessions without motion energy files are skipped. (3) Sessions with fewer than 10 units or 2 trials are skipped. (4) For tongue velocity, NaN values in interpolated velocity are replaced with 0. (5) For paw velocity, NaN values are replaced with the nanmedian, and velocities are median-subtracted. (6) Empty motion energy traces are filled with zeros.

ii.
```python
if obj is None:
    print('  skip_reason: unreadable_data_structure', flush=True)
    return None
...
if len(neural) < 2 or len(kept_units) < 10:
    print(f'  skip_reason: insufficient_neural ...', flush=True)
    return None
...
# In interp_feature_velocity for tongue:
xv[np.isnan(xv)] = 0
yv[np.isnan(yv)] = 0
# For paw:
xv = np.where(np.isnan(xv), np.nanmedian(xv), xv)
```

iii. The reference handles missing data by keeping trials but marking gaps with a "not visible" class rather than filling with zeros or medians. The AI's approach of filling NaN with 0 or median introduces fabricated velocity values.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the HDF5 data structure files is the most time-consuming step, as it involves reading and dereferencing many HDF5 references. The per-trial spike binning loop (nested unit x trial) is also relatively slow due to lack of vectorization.

ii.
```python
def load_data_structure(path):
    ...
    with hfile as h:
        obj = h['obj']
        # Many deref calls for each field
```

iii. No timing information is printed by the AI's code to measure bottlenecks.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop iterates over each unit and each trial individually, calling `np.histogram` once per unit-trial pair. The reference vectorizes this with a single `np.histogram2d` call per unit across all trials simultaneously.

ii.
```python
for ui, (st, tr) in enumerate(zip(trialtm, trialid)):
    ...
    for raw_t in np.where(trial_mask)[0]:
        counts, _ = np.histogram(st[tr == raw_t], bins=edges)
        unit_trials.append(counts.astype(np.float32))
```

iii. The nested loop (units x trials) is O(n_units * n_trials) histogram calls, which could be reduced to O(n_units) with histogram2d.

## 11-c. What processing does the code repeat multiple times?

i. The `valid_trials` mask is computed once per session. The time grid is built once. Frame times and trajectory data are accessed per trial in the trajectory output construction. No obvious major repeated computations.

ii. N/A

iii. The code structure is relatively straightforward with no major repeated processing.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads many fields from the HDF5 file that are never used (e.g., `meta`, `tm`, `site`, `NdroppedFrames`, `fn`). The `clu.quality` field is read but never used for filtering. The `stim` field in `bp` is not read at all, leading to photostim trials being included.

ii.
```python
for name in ['tm', 'site', 'quality', 'trialtm', 'trial']:
    if name in clu:
        ...
        clu_out[name] = [np.array(deref(h, r)).reshape(-1) for r in ds[()].reshape(-1)]
```

iii. Reading unused fields adds I/O overhead. The `quality` field is loaded but not used for curation despite the paper specifying quality-based filtering.
