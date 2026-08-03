# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by globbing `data/Ephys_Behavior/data_structure_*.mat` files, then filters to "context-capable" sessions (those with both autowater=0 and autowater=1 trials). It only looks in the `Ephys_Behavior` folder, not `RandomizedDelay_Ephys_Behavior`. Sessions are loaded using either h5py (for HDF5/v7.3 files) or scipy.io.loadmat. Motion energy files are discovered by matching filenames in the same folder.

ii.
```python
def discover_sessions() -> List[SessionRecord]:
    base = Path('data/Ephys_Behavior')
    data_files = sorted(base.glob('data_structure_*.mat'))
    ...
    sessions = []
    for df in data_files:
        m = re.match(r'^data_structure_([A-Za-z0-9]+)_(\d{4}-\d{2}-\d{2})\.mat$', df.name)
        ...
        sessions.append(SessionRecord(subj, day, df, motion_map.get((subj, day))))
    return sessions
```

```python
def load_mat_obj(path: Path):
    if is_hdf5_mat(path):
        return h5py.File(path, 'r')
    x = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    return x['obj'].flat[0]
```

iii. The AI explored the data directory structure and found four folders. It decided to use only `Ephys_Behavior` sessions that have both DR and WC trial types (autowater states), identifying 22 "context-capable" sessions. The CONVERSION_NOTES.md mentions awareness of the randomized delay folder but the code never includes it. The AI acknowledged the discrepancy with the paper's session counts but did not fully resolve it.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from filenames using a regex pattern `([A-Za-z0-9]+)_(\d{4}-\d{2}-\d{2})`, taking the first capture group as the subject ID. Unique subjects are sorted and assigned indices.

ii.
```python
for df in data_files:
    m = re.match(r'^data_structure_([A-Za-z0-9]+)_(\d{4}-\d{2}-\d{2})\.mat$', df.name)
    if not m:
        continue
    subj, day = m.group(1), m.group(2)
    sessions.append(SessionRecord(subj, day, df, motion_map.get((subj, day))))
```

```python
subjects = sorted({s.subject for s in sessions})
subj_to_idx = {s:i for i,s in enumerate(subjects)}
```

iii. The AI uses the filename to identify subjects, which is a reasonable approach consistent with the reference.

## 1-c. How are the data split into sessions?

i. Each `.mat` file in `Ephys_Behavior` is treated as one session. The AI processes only sessions from this single folder, yielding 21 sessions (after skipping one with unsupported clu structure). The reference includes both `Ephys_Behavior` (25 sessions) and `RandomizedDelay_Ephys_Behavior` (19 sessions) for a total of 44 sessions.

ii.
```python
def discover_sessions() -> List[SessionRecord]:
    base = Path('data/Ephys_Behavior')
    data_files = sorted(base.glob('data_structure_*.mat'))
```

iii. The AI was aware of multiple data folders but chose to focus on `Ephys_Behavior` as the primary dataset. The CONVERSION_NOTES.md shows the AI struggled to match the paper's session counts and never resolved the issue of including randomized delay sessions.

## 1-d. How are the data split into trials?

i. Trials are defined by the go cue event array `bp.ev.goCue`, with `n_trials = go.shape[0]`. Each trial index corresponds to one entry in the behavioral and neural data arrays.

ii.
```python
go = get_event(obj, ALIGN_EVENT)
n_trials = go.shape[0]
```

iii. This is a standard approach consistent with the reference's use of `Ntrials` and the go cue array.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several filters: early lick trials are removed, stim-enabled trials are removed, and additionally it requires trials to be either hit or miss (excluding ignore trials), requires `right ^ left` (exactly one direction set), and requires autowater to be 0 or 1. This is more restrictive than the reference, which only removes early lick and photostim trials but keeps ignore trials.

ii.
```python
valid = np.ones(n_trials, dtype=bool)
if early is not None:
    valid &= ~early
if hit is not None and miss is not None:
    valid &= (hit | miss)
if stim_enable is not None:
    valid &= ~stim_enable
if right is not None and left is not None:
    valid &= (right ^ left)
if autowater is not None:
    valid &= np.isin(autowater.astype(int), [0, 1])
```

iii. The AI's CONVERSION_NOTES.md acknowledges the tradeoff: "enforcing the exact hit-only/no-stim conditions from `getDefaultParams.m` makes the `outcome` output degenerate (all correct), which conflicts with the decoder task requirement to predict incorrect vs correct outcome." Despite this awareness, the code still filters out ignore trials (keeping only hit|miss).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `obj.clu` spike cluster data, specifically the `trial` and `trialtm` fields of each cluster. Go cue times from `bp.ev.goCue` are used for alignment.

ii.
```python
if isinstance(obj, h5py.File):
    clu_root = obj['obj']['clu']
    clu = obj[clu_root[0,0]]
    ...
    n_clu = clu['trial'].shape[0]
    for ci in range(n_clu):
        tr_ref = clu['trial'][ci,0]
        tm_ref = clu['trialtm'][ci,0]
        tr_arr = np.asarray(obj[tr_ref][()]).reshape(-1).astype(int)
        tm_arr = np.asarray(obj[tm_ref][()]).reshape(-1).astype(float)
```

iii. The AI correctly identified `clu.trial` and `clu.trialtm` as the source of neural data, consistent with the reference.

## 2-b. How is the `neural` data processed?

i. The AI bins spike counts into 75ms bins spanning [-1.5, 1.5]s from the go cue. No conversion to firing rate (Hz), no Gaussian smoothing. Raw spike counts are used directly. The reference converts to Hz (divides by bin width) and smooths with a 14ms Gaussian.

ii.
```python
binedges = np.concatenate([time_bins - BIN_SIZE_S/2, [time_bins[-1] + BIN_SIZE_S/2]]).astype(float)
...
for ci, (tr_arr, tm_arr) in enumerate(zip(clu_trial, clu_trialtm)):
    spikes = tm_arr[tr_arr == tr1]
    if spikes.size:
        mat[ci], _ = np.histogram(spikes, bins=binedges)
neural_trials.append(mat)
```

iii. The AI chose 75ms bins based on the decoder scripts' `rez.binSize = 75` and a [-1.5, 1.5]s window. This differs from the reference which uses 5ms bins and [-2.5, 2.5]s (from `params.dt = 1/200` and `params.tmin/tmax`). The 75ms bin size was the decoder's *rebinning* resolution, not the native processing resolution.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies no quality filtering on clusters. All clusters from the first probe reference (`clu[0,0]`) are included regardless of quality label or firing rate. The reference filters by quality label (dropping "garbage", "gabrga", "noisy", "real?", "poor") and by mean firing rate (>1 Hz threshold).

ii.
```python
n_clu = clu['trial'].shape[0]
for ci in range(n_clu):
    tr_ref = clu['trial'][ci,0]
    ...
    clu_trial.append(tr_arr)
    clu_trialtm.append(tm_arr)
```

iii. The AI's CONVERSION_NOTES.md mentions being aware of `removeLowFRClusters.m` and the paper's 1 Hz threshold, and planned to apply filtering, but the final code does not implement any quality or rate filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times (`trialtm`) are not explicitly aligned to the go cue. Instead, the code computes bin edges relative to the go cue onset using `np.histogram` with edges centered on `time_bins` (which start at -1.5s). However, the actual subtraction of go cue from spike times is missing — the code histograms raw `trialtm` values (which are relative to trial start) against go-cue-aligned bin edges, which is incorrect.

ii.
```python
binedges = np.concatenate([time_bins - BIN_SIZE_S/2, [time_bins[-1] + BIN_SIZE_S/2]]).astype(float)
...
spikes = tm_arr[tr_arr == tr1]
if spikes.size:
    mat[ci], _ = np.histogram(spikes, bins=binedges)
```

The `time_bins` are defined as:
```python
time_bins = np.arange(T_START, T_END + 1e-9, BIN_SIZE_S, dtype=np.float32)
```
where `T_START = -1.5` and `T_END = 1.5`.

iii. The `trialtm` values are relative to trial start, but the bin edges are relative to the go cue. The AI does not subtract the go cue time from the spike times before histogramming. The reference correctly does `spike_time = trialtm - go_cue[trial]`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 75ms bins, producing 41 time points per trial in a [-1.5, 1.5]s window. The reference uses 5ms bins, producing 1000 time points in a [-2.5, 2.5]s window. No additional rebinning is applied by the AI.

ii.
```python
BIN_SIZE_S = 0.075
T_START = -1.5
T_END = 1.5
...
time_bins = np.arange(T_START, T_END + 1e-9, BIN_SIZE_S, dtype=np.float32)
```

iii. The AI chose 75ms because the decoder scripts in the reference code use `rez.binSize = 75`. However, this is the decoder's analysis binning, not the native data processing resolution. The reference code uses `params.dt = 1/200` (5ms) as the fundamental time step.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is a synthetic time vector constructed from the bin parameters, not from any raw data variable. It represents the center of each time bin.

ii.
```python
time_bins = np.arange(T_START, T_END + 1e-9, BIN_SIZE_S, dtype=np.float32)
...
inp = time_bins[None, :].astype(np.float32)
```

iii. This is conceptually correct — the time-from-go-cue input is defined by the binning grid, same as in the reference.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time bins are generated using `np.arange` from T_START to T_END with step BIN_SIZE_S. This produces bin edges, and the values represent the start of each bin (not bin centers as in the reference).

ii.
```python
time_bins = np.arange(T_START, T_END + 1e-9, BIN_SIZE_S, dtype=np.float32)
```

iii. The reference uses bin centers (`BIN_EDGES[:-1] + BIN/2`), while the AI uses `np.arange` which gives the left edge of each bin as the time value.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The same `time_bins` array defines both the neural histogram edges and the input time vector, so they are implicitly aligned by construction.

ii.
```python
binedges = np.concatenate([time_bins - BIN_SIZE_S/2, [time_bins[-1] + BIN_SIZE_S/2]]).astype(float)
...
inp = time_bins[None, :].astype(np.float32)
```

iii. The alignment is consistent by construction, same approach as the reference.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derives lick direction from the `R` (right) field of `bp`. It does not use hit/miss to infer the actual lick direction — it directly uses the instructed side.

ii.
```python
right = get_trial_bool(bp, 'R')
...
lick_dir = 1 if bool(right[tr]) else 0
```

iii. The AI uses the instructed direction as a proxy for lick direction. The reference infers actual lick direction from the combination of instructed side and outcome (hit = licked instructed side, miss = licked opposite side, ignore = no lick).

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI assigns left=0 if `R` is false, right=1 if `R` is true. Only two classes. The reference uses three classes: left=0, right=1, no lick=2. Since the AI also removes ignore trials, the instructed side happens to match the actual lick direction for the remaining hit/miss trials.

ii.
```python
lick_dir = 1 if bool(right[tr]) else 0
```

Reference:
```python
lick_direction = np.full(n_trials, LICK['no lick'])
lick_direction[hit] = np.where(right[hit], LICK['right'], LICK['left'])
lick_direction[miss] = np.where(right[miss], LICK['left'], LICK['right'])
```

iii. The AI's approach conflates instructed side with lick direction. For hit trials this is correct, but for miss trials the animal licked the opposite side. Since ignore trials are filtered out, miss trials still have the wrong lick direction assigned.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The `autowater` field of `bp` is used. WC (autowater) = 0, DR (non-autowater) = 1.

ii.
```python
autowater = get_trial_bool(bp, 'autowater')
...
context = 0 if bool(autowater[tr]) else 1
```

iii. This matches the reference's approach.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct mapping: autowater=True becomes WC=0, autowater=False becomes DR=1. This matches the reference.

ii.
```python
context = 0 if bool(autowater[tr]) else 1
```

iii. Consistent with the reference and instruction specification (WC=0, DR=1).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The AI uses `hit` from `bp`. If hit is true, outcome=1 (correct), otherwise outcome=0 (incorrect). Since ignore trials are filtered out, the remaining trials are only hits and misses.

ii.
```python
outcome = 1 if bool(hit[tr]) else 0
```

iii. The reference uses three classes: incorrect=0, correct=1, ignore=2. The AI uses only two since ignore trials are excluded.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Binary classification: hit=correct(1), everything else=incorrect(0). The reference has a third "ignore" class for trials where the animal did not respond.

ii.
```python
outcome = 1 if bool(hit[tr]) else 0
```

iii. The AI's CONVERSION_NOTES.md acknowledges this tradeoff but chose to exclude ignore trials to have a nontrivial outcome variable.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The AI uses `obj.traj` trajectory data, searching for tongue-related features with candidates `['top_tongue', 'topleft_tongue', 'bottom_tongue', 'bottomleft_tongue', 'tongue', 'left_tongue', 'right_tongue']`.

ii.
```python
tongue_vals = extract_hdf5_traj_velocity(obj, tr,
    ['top_tongue', 'topleft_tongue', 'bottom_tongue', 'bottomleft_tongue',
     'tongue', 'left_tongue', 'right_tongue'], time_bins)
```

iii. The AI identifies the correct source but uses a broad candidate list. The reference specifically uses `tongue` (side camera) and `top_tongue` (bottom camera) and combines both views.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI computes velocity as `sqrt(dx^2 + dy^2) / dt` using raw `np.diff` on x,y coordinates. No likelihood filtering is applied. The speed is interpolated to the time bins using `np.interp`, and NaN gaps are filled with nearest-neighbor interpolation. Only one camera view is used (the first matching feature). The reference uses likelihood-based filtering (>0.9), per-run Gaussian smoothing, combines two camera views normalized by their 90th percentiles, and uses proper binning.

ii.
```python
xy = ts[feat_idx, :2, :]
dx = np.diff(xy[0], prepend=xy[0,0])
dy = np.diff(xy[1], prepend=xy[1,0])
dt = np.diff(ft, prepend=ft[0])
dt[dt <= 0] = np.nanmedian(dt[dt > 0]) if np.any(dt > 0) else 1.0
speed = np.sqrt(dx*dx + dy*dy) / dt
go = float(np.asarray(obj['obj']['bp']['ev']['goCue'][()]).reshape(-1)[trial_index0])
rel_t = ft - go
vals = np.interp(time_bins, rel_t, speed, left=np.nan, right=np.nan)
if np.isfinite(vals).any():
    idx = np.arange(vals.size)
    good = np.isfinite(vals)
    vals = np.interp(idx, idx[good], vals[good])
```

iii. The AI's velocity computation is much simpler than the reference's, lacking likelihood filtering, Gaussian smoothing, multi-camera fusion, and normalization. The NaN interpolation fills gaps rather than marking them as "not visible".

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI uses 50th percentile threshold across all trials in a session, producing two classes (0=low, 1=high). The reference also uses 50th percentile but has a third class (2="not visible") for bins where the tongue was not tracked.

ii.
```python
tongue_thr = np.nanpercentile(tongue_all, 50) if tongue_all.size and np.isfinite(tongue_all).any() else np.nan
...
tmp[finite] = (tong[finite] >= tongue_thr).astype(np.int64)
```

iii. The AI uses only 2 classes while the reference uses 3. The reference's "not visible" class accounts for ~88% of bins where the tongue is out of view.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI subtracts the go cue time from frame times directly (`ft - go`), without computing the video-behavior clock offset via bitcode. The reference computes a session-wide video offset from `sglx.bitcode.bitstart` and `bp.ev.bitStart` to correct the camera clock before aligning to the go cue.

ii.
```python
go = float(np.asarray(obj['obj']['bp']['ev']['goCue'][()]).reshape(-1)[trial_index0])
rel_t = ft - go
vals = np.interp(time_bins, rel_t, speed, left=np.nan, right=np.nan)
```

iii. Without the bitcode-based clock correction, the camera frame times may be misaligned with the behavioral events by a constant offset per session.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The AI uses `obj.traj` with feature candidates `['paw', 'left_paw', 'right_paw', 'top_paw', 'bottom_paw']`. The reference uses only `top_paw` from the bottom camera.

ii.
```python
paw_vals = extract_hdf5_traj_velocity(obj, tr,
    ['paw', 'left_paw', 'right_paw', 'top_paw', 'bottom_paw'], time_bins)
```

iii. The AI uses a broad feature candidate list and takes the first match. The reference specifically chose `top_paw` because it stays reliably tracked through the delay epoch.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same processing as tongue velocity: raw diff-based speed, no likelihood filtering, no Gaussian smoothing, nearest-neighbor interpolation of NaN gaps. The reference applies the same processing as tongue (likelihood cut, per-run smoothing) but without cross-camera normalization since only one camera is used.

ii. Same `extract_hdf5_traj_velocity` function as tongue.

iii. The AI does not distinguish between paw and tongue processing approaches.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: 50th percentile, two classes. The reference uses 50th percentile with a third "not visible" class.

ii.
```python
paw_thr = np.nanpercentile(paw_all, 50)
...
tmp[finite] = (paw[finite] >= paw_thr).astype(np.int64)
```

iii. Same issue as tongue — missing the "not visible" class.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same approach as tongue: go cue subtracted from frame times, no bitcode clock correction.

ii. Same `extract_hdf5_traj_velocity` function.

iii. Same issue as tongue — missing video clock offset correction.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI loads motion energy from `motionEnergy_*.mat` sidecar files, extracting `me.data`.

ii.
```python
def load_motion_energy(path: Optional[Path], n_trials: int) -> Optional[List[np.ndarray]]:
    ...
    x = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    me = x['me'].flat[0]
    data = np.asarray(unwrap_me_data(me)).reshape(-1)
```

iii. Consistent with the reference's source data.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI interpolates motion energy to the time bins using `np.interp` with linearly spaced time points (`np.linspace(T_START, T_END, raw.size)`), assuming the frames span the trial window uniformly. The reference uses actual camera frame times (corrected by the bitcode offset) and bins by averaging frames within each 5ms bin.

ii.
```python
if raw.size > 1:
    x_old = np.linspace(T_START, T_END, raw.size)
    me_binned = np.interp(time_bins, x_old, raw)
```

iii. The assumption that frames are linearly spaced is incorrect — camera frame times are not uniformly distributed and need the actual `frameTimes` field with clock correction.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The AI uses 50th percentile across all trials, two classes. Before thresholding, NaN values are replaced with the median of all motion energy values. The reference uses 50th percentile with a third "not visible" class for NaN bins.

ii.
```python
all_me = np.concatenate([np.nan_to_num(x, nan=np.nanmedian(np.concatenate(me_binned_all))) for x in me_binned_all])
thr = np.nanpercentile(all_me, 50)
...
tmp[finite] = (meb[finite] >= thr).astype(np.int64)
```

iii. The NaN replacement with median inflates the "below threshold" class and loses information about untracked frames.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI assumes frame times are linearly spaced within the trial window and uses interpolation. The reference uses actual camera frame times corrected by the session-wide bitcode offset.

ii.
```python
x_old = np.linspace(T_START, T_END, raw.size)
me_binned = np.interp(time_bins, x_old, raw)
```

iii. This is an approximation that could introduce temporal misalignment.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing data by: (1) skipping sessions with unsupported clu structure (e.g., JEB6), (2) filling NaN trajectory values with nearest-neighbor interpolation, (3) replacing NaN motion energy with the session median, (4) padding motion energy if fewer trials exist than expected. Sessions that fail processing are silently skipped with a warning.

ii.
```python
except Exception as e:
    print(f'[warn] skipping {sess.data_file.name}: {e}', flush=True)
    continue
```

```python
# NaN interpolation for trajectories
if np.isfinite(vals).any():
    idx = np.arange(vals.size)
    good = np.isfinite(vals)
    vals = np.interp(idx, idx[good], vals[good])
```

iii. The reference preserves missing data information through a "not visible" class rather than imputing values. The AI's approach of imputing NaNs with interpolation or medians masks genuinely missing data.

## 11-a. What are the most time-consuming steps of the code?

i. The trajectory velocity extraction dominates, since `extract_hdf5_traj_velocity` is called once per trial per feature, with extensive HDF5 dereferencing for each call. The conversion log shows sessions taking 5-20 seconds each, with 21 sessions total.

ii.
```python
def extract_hdf5_traj_velocity(obj: Any, trial_index0: int, feature_candidates: List[str], time_bins: np.ndarray) -> np.ndarray:
    ...
    for i in range(traj_ds.shape[0]):
        ...
        names_arr = obj[traj['featNames'][trial_index0,0]][()]
```

iii. The per-trial HDF5 dereferencing is very expensive compared to loading the whole structure once.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop over trials in `build_session` processes each trial individually, including spike histogramming, trajectory extraction, and output construction. The spike counting could be vectorized with `np.histogram2d` as in the reference. The trajectory extraction calls `extract_hdf5_traj_velocity` per trial, which repeats HDF5 feature name resolution for every trial.

ii.
```python
for tr in trial_idx:
    ...
    for ci, (tr_arr, tm_arr) in enumerate(zip(clu_trial, clu_trialtm)):
        spikes = tm_arr[tr_arr == tr1]
        if spikes.size:
            mat[ci], _ = np.histogram(spikes, bins=binedges)
```

iii. The reference uses `np.histogram2d` to bin all spikes across all trials at once, avoiding the per-trial loop.

## 11-c. What processing does the code repeat multiple times?

i. The trajectory velocity extraction is called **three times** per trial — once during the initial trial loop, once for computing thresholds (all trials concatenated), and once again for applying thresholds. Each call repeats all HDF5 dereferencing and velocity computation.

ii.
```python
# First pass in trial loop:
tongue_vals = extract_hdf5_traj_velocity(obj, tr, [...], time_bins)
paw_vals = extract_hdf5_traj_velocity(obj, tr, [...], time_bins)

# Second pass for thresholds:
tongue_all = np.concatenate([... extract_hdf5_traj_velocity(obj, tr, [...], time_bins) for tr in trial_idx])
paw_all = np.concatenate([... extract_hdf5_traj_velocity(obj, tr, [...], time_bins) for tr in trial_idx])

# Third pass for applying thresholds:
for i, tr in enumerate(trial_idx):
    tong = extract_hdf5_traj_velocity(obj, tr, [...], time_bins)
    paw = extract_hdf5_traj_velocity(obj, tr, [...], time_bins)
```

iii. Each extraction involves HDF5 I/O and velocity computation, so tripling the work is very expensive.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The first pass of tongue/paw velocity extraction in the trial loop computes `tongue_vals` and `paw_vals` but stores them as zeros in the output array (the placeholder values are overwritten in the third pass). The second pass concatenates all velocities just to compute thresholds, then discards the arrays. Also, the initial output array is constructed with all-zeros for tongue/paw/motion_energy slots, which are then overwritten.

ii.
```python
output_trials.append(np.vstack([
    ...
    np.zeros(time_bins.shape, dtype=np.int64),  # tongue placeholder
    np.zeros(time_bins.shape, dtype=np.int64),  # paw placeholder
    np.zeros(time_bins.shape, dtype=np.int64),  # motion_energy placeholder
]))
```

iii. The redundant extraction passes waste significant compute time.
