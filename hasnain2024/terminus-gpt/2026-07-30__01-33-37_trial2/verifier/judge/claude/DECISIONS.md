# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by globbing the `data/` directory for `data_structure_*.mat` and `motionEnergy_*.mat` files, pairing them by stem name. Each `data_structure` file is opened with `h5py` (HDF5/v7.3 format only); if `h5py` raises an `OSError`, the session is skipped entirely. This means sessions stored in MATLAB v5 format are silently dropped. Motion energy is loaded separately via `scipy.io.loadmat`.

ii. Session discovery:
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
```

Loading:
```python
def load_data_structure(path):
    try:
        hfile = h5py.File(path, 'r')
    except OSError:
        return None
```

iii. The AI noted in CONVERSION_NOTES.md that sessions with unreadable data structures are skipped with logging. The trajectory shows the AI discovered the data files by exploring the directory structure and used `h5py` for reading. The AI did not cross-reference against the authors' loading scripts to determine which sessions should be included.

## 1-b. How are the data split into subjects?

i. The subject is extracted from the session stem by splitting on underscore and taking the first part (e.g., `JEB11` from `JEB11_2022-05-10`). Subjects are accumulated in order of first appearance.

ii.
```python
def infer_subject_session(stem):
    parts = stem.split('_')
    return parts[0], '_'.join(parts[1:])
```

iii. The AI noted this naming convention from exploring the filenames. This is the same approach as the reference.

## 1-c. How are the data split into sessions?

i. One session is one `data_structure_*.mat` file discovered by globbing. Sessions from all subdirectories (`Ephys_Behavior`, `RandomizedDelay_Ephys_Behavior`, and also inhibition folders) are discovered. Sessions that fail to load via `h5py` or lack required fields are skipped. The result is 33 sessions instead of the reference's 44, because the AI's loader cannot read v5 MATLAB files and does not use the author's session list.

ii.
```python
keys = sorted(k for k,v in sessions.items() if 'data_structure' in v and 'motion_energy' in v)
```

iii. The conversion log shows 11 sessions skipped as "unreadable_data_structure" (all JEB23/JEB24 sessions stored in v5 format) and 1 skipped as "missing_trial_aligned_spikes" (JEB6). The AI did not attempt to add `scipy.io.loadmat` as a fallback reader.

## 1-d. How are the data split into trials?

i. Trials are indexed by the `Ntrials` field in `bp`. The `valid_trials` function creates a boolean mask of length `Ntrials` and filters based on `early` and `no` (ignore) fields.

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

iii. The AI documented in CONVERSION_NOTES that early lick and ignore trials are omitted, citing the methods text.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by excluding early-lick trials (`bp.early`) and ignore/no-response trials (`bp.no`). Photostimulation trials (`bp.stim.enable`) are NOT filtered. There is no check for trials extending past the end of the recording.

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

iii. The AI cited the methods text: "early lick and ignore trials are omitted from analyses." However, the reference also excludes photostimulation trials, and the AI's filtering of ignore trials differs from the reference which keeps ignore trials but assigns them a third outcome/lick-direction class.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data comes from `obj.clu`, specifically the `trialtm` (spike times relative to trial) and `trial` (trial assignment) arrays for each unit. Only probe 1 is read (`clu[()].reshape(-1)[0]`).

ii.
```python
clu_ref = obj['clu'][()].reshape(-1)[0]
clu = deref(h, clu_ref)
clu_out = {}
for name in ['tm', 'site', 'quality', 'trialtm', 'trial']:
    if name in clu:
        ds = clu[name]
        if ds.dtype == h5py.ref_dtype or str(ds.dtype) == 'object':
            clu_out[name] = [np.array(deref(h, r)).reshape(-1) for r in ds[()].reshape(-1)]
```

iii. The AI's trajectory (Step 25) confirmed it read `clu.trialtm` and `clu.trial` as per-unit spike time arrays. The loader only dereferences the first element of `obj['clu']`, so multi-probe sessions (e.g., JEB15 with probes [1,2]) lose all units from probe 2.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into time bins using `np.histogram` per unit per trial. No conversion to firing rates (Hz), no Gaussian smoothing. Raw spike counts (as float32) are stored directly.

ii.
```python
def bin_spikes_for_session(obj, trial_mask, edges):
    ...
    for ui, (st, tr) in enumerate(zip(trialtm, trialid)):
        ...
        for raw_t in np.where(trial_mask)[0]:
            counts, _ = np.histogram(st[tr == raw_t], bins=edges)
            unit_trials.append(counts.astype(np.float32))
```

iii. The AI's CONVERSION_NOTES do not mention converting to firing rates or smoothing. The reference converts counts to Hz by dividing by bin width and applies a 14ms Gaussian smooth.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered only by an approximate firing rate threshold: `fr = st.size / approx_duration; if fr <= 1.0: continue`. The `quality` field is read from the data but never used for filtering. No manual curation labels are checked.

ii.
```python
approx_duration = (edges[-1] - edges[0]) * max(trial_mask.sum(), 1)
fr = st.size / max(approx_duration, 1e-9)
if fr <= 1.0:
    continue
```

iii. The AI noted in CONVERSION_NOTES Step 5 "Filter to sessions with >=10 units and units with FR >1 Hz". The `quality` field is loaded but the code never references it for filtering. The reference filters by quality labels (excluding garbage, gabrga, noisy, real?, poor) AND by mean rate > 1 Hz.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are NOT explicitly aligned to the go cue before binning. The `bin_spikes_for_session` function bins `trialtm` values (spike times relative to trial start) directly into the time grid `[-2.5, 2.5]`. There is no subtraction of `goCue` from `trialtm`.

ii.
```python
def bin_spikes_for_session(obj, trial_mask, edges):
    ...
    for raw_t in np.where(trial_mask)[0]:
        counts, _ = np.histogram(st[tr == raw_t], bins=edges)
```

The edges are built without go-cue alignment:
```python
def build_time_grid(bin_size_s=0.075, t_start=-2.5, t_end=2.5):
    edges = np.arange(t_start, t_end + 1e-9, bin_size_s)
```

iii. The AI discussed go-cue alignment in planning (CONVERSION_NOTES Steps 4-5) but the code bins `trialtm` directly without subtracting `goCue`. The reference subtracts `goCue[trial]` from `trialtm` before binning.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 75 ms, producing 66 time bins over the [-2.5, 2.5] s window. The metadata records `time_bin_size: 75.0`.

ii.
```python
def build_time_grid(bin_size_s=0.075, t_start=-2.5, t_end=2.5):
    edges = np.arange(t_start, t_end + 1e-9, bin_size_s)
    centers = edges[:-1] + bin_size_s / 2
    return edges, centers
```

iii. The AI's trajectory (Step 4) noted that `DLC_ContextDecoding.m` uses `rez.binSize = 75` ms. However, the reference's native bin size is 5 ms (`params.dt = 1/200`), and the 75 ms was only used for the DLC decoding analysis, not for data storage. The reference uses 5 ms bins (1000 time points).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the center of each time bin in the fixed grid, constructed from the bin edges. It is not derived from any raw data variable.

ii.
```python
inputs = [centers[None, :].astype(np.float32) for _ in idx]
```

Where `centers` comes from:
```python
edges, centers = build_time_grid()
```

iii. Same approach as reference: a fixed time vector representing bin centers.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is constructed from the bin edges as `edges[:-1] + bin_size_s / 2`. No processing of raw data is involved.

ii.
```python
centers = edges[:-1] + bin_size_s / 2
```

iii. Same conceptual approach as reference, but at 75ms resolution instead of 5ms.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The same `centers` array is used as the input for every trial. However, since neural data is not properly aligned to go cue (see 2-d), the input time labels may not correspond to the actual time from go cue in the neural data.

ii.
```python
inputs = [centers[None, :].astype(np.float32) for _ in idx]
```

iii. The time grid is shared between neural and input, but the neural data alignment to go cue is missing, so the correspondence is broken.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Derived from `bp.L` and `bp.R`, which indicate the instructed lick direction (left or right port).

ii.
```python
L = np.array(bp['L'], dtype=float).reshape(-1)
R = np.array(bp['R'], dtype=float).reshape(-1)
lick_dir = np.where(R[idx] > L[idx], 1, 0).astype(np.int64)
```

iii. The AI uses `R > L` to determine lick direction. This gives the *instructed* direction, not the *actual* lick direction. The reference derives actual lick direction from the combination of instructed side and hit/miss outcome.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A simple comparison: if R > L then right (1), else left (0). Only 2 classes. No "no lick" class for ignore trials (those are already filtered out).

ii.
```python
lick_dir = np.where(R[idx] > L[idx], 1, 0).astype(np.int64)
```

iii. The reference uses 3 classes (left=0, right=1, no lick=2), deriving actual lick direction from hit/miss combined with instructed side. The AI's approach conflates instructed and actual direction -- on miss trials the animal licked the opposite direction.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derived from `bp.autowater`. Trials with `autowater > 0` are WC (water-cued, coded 0), others are DR (delayed-response, coded 1).

ii.
```python
autowater = np.array(bp['autowater'], dtype=float).reshape(-1)
context = np.where(autowater[idx] > 0, 0, 1).astype(np.int64)
```

iii. Same source variable and mapping as reference.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct relabeling: autowater > 0 maps to WC (0), else DR (1).

ii.
```python
context = np.where(autowater[idx] > 0, 0, 1).astype(np.int64)
```

iii. Matches the reference's WC=0, DR=1 encoding.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `bp.hit` and `bp.miss`.

ii.
```python
hit = np.array(bp['hit'], dtype=float).reshape(-1)
miss = np.array(bp['miss'], dtype=float).reshape(-1)
outcome = np.where(hit[idx] > miss[idx], 1, 0).astype(np.int64)
```

iii. Same source variables as reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. If `hit > miss` then correct (1), else incorrect (0). Only 2 classes. Ignore trials (where both hit and miss are 0) are already excluded by `valid_trials` filtering `bp.no`. The comparison `hit > miss` would assign ignore trials to class 0 (incorrect) if they were present.

ii.
```python
outcome = np.where(hit[idx] > miss[idx], 1, 0).astype(np.int64)
```

iii. The reference uses 3 classes (incorrect=0, correct=1, ignore=2) and keeps ignore trials. The AI filters them out entirely. The instructions specify "incorrect = 0, correct = 1" with only 2 classes, so this is somewhat consistent with the instructions, though the reference chose to add a third class.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Derived from `obj.traj` (side camera view only), using the `tongue` feature identified by name in `featNames`. Uses `ts` for positions and `frameTimes` for timing.

ii.
```python
tongue_idx = pick_feature(side_names, ['tongue', 'left_tongue', 'right_tongue'])
...
tongue.append(interp_feature_velocity(ts_side, ft_side, go[tr], centers, tongue_idx, tongue=True))
```

iii. The reference uses both side and bottom cameras for tongue tracking (`tongue` and `top_tongue`), normalizes each by its 90th percentile, and averages them. The AI uses only the side camera.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI interpolates x,y positions to the bin centers, then computes velocity as `np.gradient` of the interpolated positions. For tongue, NaN values in the gradient are replaced with 0. Speed is `sqrt(xv^2 + yv^2)`. No likelihood filtering, no per-run smoothing. Discretized by `> median` into 2 classes.

ii.
```python
def interp_feature_velocity(ts, frame_times, align_time, centers, feat_idx, tongue=False):
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

iii. Major differences from reference: (1) no likelihood filtering (reference uses > 0.9), (2) interpolation to bin centers instead of binning frame-level velocities, (3) no Gaussian smoothing of positions before differentiation, (4) no normalization of camera views, (5) only 2 classes instead of 3 (no "not visible" class), (6) uses `> median` instead of `>= 50th percentile`.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Split at the global median (across all trials and time bins for the session) using strict `>`. Two classes: 0 (below or equal to median) and 1 (above median).

ii.
```python
tongue_disc = (tongue > np.nanmedian(tongue)).astype(np.int64)
```

iii. The reference uses `np.nanpercentile(tongue, 50)` with `>=` threshold and has 3 classes (below, above, not visible). The AI uses strict `>` which biases class sizes slightly.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are adjusted by subtracting a hardcoded offset of 0.5 seconds and the go cue time: `rel_t = (ft - 0.5) - float(align_time)`. The positions are then interpolated to bin centers.

ii.
```python
rel_t = (ft - 0.5) - float(align_time)
x = np.interp(centers, rel_t, xy[:, 0], left=np.nan, right=np.nan)
```

iii. The reference computes the video offset dynamically from bitcode timestamps (`findVideoOffset.m`). The AI uses a hardcoded 0.5s offset, which is likely incorrect for most sessions since the offset varies per session.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Derived from `obj.traj` (bottom camera view), using the `top_paw` feature (or fallback to `bottom_paw` or `paw`).

ii.
```python
paw_idx = pick_feature(bottom_names, ['top_paw', 'bottom_paw', 'paw'])
...
paw.append(interp_feature_velocity(ts_bot, ft_bot, go[tr], centers, paw_idx, tongue=False))
```

iii. Same source as reference (`top_paw` from bottom camera).

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same interpolation-based approach as tongue velocity but with different NaN handling: NaN values are replaced with `nanmedian` and the velocity is median-subtracted. Discretized by `> median` into 2 classes.

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

iii. Same issues as tongue: no likelihood filtering, no per-run smoothing, interpolation instead of binning, 2 classes instead of 3. Additionally, the median subtraction of velocity components is not done in the reference.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: split at global session median using strict `>`. Two classes.

ii.
```python
paw_disc = (paw > np.nanmedian(paw)).astype(np.int64)
```

iii. Same differences as tongue thresholding.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same hardcoded 0.5s offset as tongue, with interpolation to bin centers.

ii.
```python
rel_t = (ft - 0.5) - float(align_time)
```

iii. Same issue as tongue alignment: hardcoded offset instead of per-session bitcode-derived offset.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Derived from `motionEnergy_*.mat` files, loaded via `scipy.io.loadmat`. The `me` struct is unwrapped through potentially nested `data` keys.

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

iii. Same source file as reference.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The per-trial motion energy trace is rebinned to match the number of time bins using linear interpolation (`rebin_variable_trace`). Then discretized at the session median with strict `>`.

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

iii. The reference bins frame-level values into 5ms bins using actual frame times (aligned to go cue). The AI's interpolation does not use frame times at all -- it just stretches the trace to fit the bins, losing temporal alignment.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Split at the global session median using strict `>`. Two classes.

ii.
```python
me_disc = (me_stack > np.nanmedian(me_stack)).astype(np.int64)
```

iii. Reference uses `np.nanpercentile` with `>=` and 3 classes (below, above, not visible). The AI uses 2 classes.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is NOT aligned to the go cue. The `rebin_variable_trace` function linearly interpolates the per-trial trace to 66 bins without reference to frame times, go cue, or any temporal anchor.

ii.
```python
me_trials = [rebin_variable_trace(me_data[int(i)], n_bins) for i in idx]
```

iii. The reference aligns motion energy using the side camera's frame times, corrected by the video offset and go cue time. The AI's approach ignores temporal alignment entirely.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several missing-data cases are handled: sessions with unreadable files are skipped; sessions with fewer than 10 units or fewer than 2 trials are skipped; if trajectory data is missing or has no paw feature, the session is skipped; empty motion energy data skips the session. For tongue velocity, NaN values in derivatives are set to 0. For paw velocity, NaN values are replaced with the nanmedian.

ii.
```python
if obj is None:
    return None
if len(neural) < 2 or len(kept_units) < 10:
    return None
if tongue_disc is None or paw_disc is None:
    return None
```

iii. The AI logs skip reasons for debugging. The reference handles missing data at finer granularity (per-trial, per-bin) with a "not visible" class rather than skipping entire sessions.

## 11-a. What are the most time-consuming steps of the code?

i. File I/O dominates: loading HDF5 files with full dereferencing of MATLAB object references. The per-trial, per-unit spike binning loop is also expensive due to lack of vectorization.

ii.
```python
def load_data_structure(path):
    ...
    with hfile as h:
        obj = h['obj']
        # extensive per-field dereferencing
```

```python
for ui, (st, tr) in enumerate(zip(trialtm, trialid)):
    ...
    for raw_t in np.where(trial_mask)[0]:
        counts, _ = np.histogram(st[tr == raw_t], bins=edges)
```

iii. The nested unit-by-trial loop for spike binning is O(units * trials) histogram calls, which is much slower than the reference's vectorized `histogram2d` approach.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop iterates over each unit and each trial separately, calling `np.histogram` once per unit per trial. This could be vectorized using `np.histogram2d` (as the reference does) to bin all trials for a unit in one call.

ii.
```python
for ui, (st, tr) in enumerate(zip(trialtm, trialid)):
    ...
    for raw_t in np.where(trial_mask)[0]:
        counts, _ = np.histogram(st[tr == raw_t], bins=edges)
        unit_trials.append(counts.astype(np.float32))
```

iii. The reference bins all trials for each unit in a single `histogram2d` call, which is significantly faster.

## 11-c. What processing does the code repeat multiple times?

i. The go cue array is loaded multiple times in `build_trial_labels` and `build_traj_outputs` rather than being computed once. The trajectory data (`_traj`) is accessed repeatedly for each feature/trial. The `trial_mask` application and `np.where(trial_mask)[0]` is computed in multiple places.

ii.
```python
# In build_trial_labels:
go = np.array(obj['bp']['ev']['goCue'], dtype=float).reshape(-1)
# Same in build_traj_outputs:
go = np.array(obj['bp']['ev']['goCue'], dtype=float).reshape(-1)
```

iii. Minor redundancy. The reference computes go cue once per session.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `load_data_structure` function reads fields like `meta`, `L`, `R`, `bitRand`, `lickL`, `lickR`, `reward`, `sample`, `site`, `tm` that are never used by the conversion. The `no` field is read for trial filtering but downstream analyses wouldn't need ignore trial filtering since the reference keeps them.

ii.
```python
for name in ['L', 'R', 'Ntrials', 'autowater', 'bitRand', 'early', 'hit', 'miss', 'no']:
    if name in bp:
        bp_out[name] = np.array(bp[name])
```

iii. Reading unused fields adds I/O overhead but doesn't affect correctness.
