# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by globbing only `data/Ephys_Behavior/data_structure_*.mat` files. It does NOT include the `RandomizedDelay_Ephys_Behavior` directory. It then filters sessions to only those with both autowater states present ("context-capable" sessions). Each session is loaded via `load_mat_obj()`, which detects whether a file is HDF5 (v7.3) or standard MATLAB and loads accordingly. Motion energy is loaded from sidecar `motionEnergy_*.mat` files via `load_motion_energy()`.

ii. Session discovery:
```python
def discover_sessions() -> List[SessionRecord]:
    base = Path('data/Ephys_Behavior')
    data_files = sorted(base.glob('data_structure_*.mat'))
    ...
```

Context filtering:
```python
def select_context_sessions(sessions: List[SessionRecord]) -> List[SessionRecord]:
    ...
    vals = set(np.unique(autowater.astype(int)).tolist())
    if vals == {0, 1}:
        keep.append(s)
    ...
```

iii. The AI chose to glob only `Ephys_Behavior` and filter for "context-capable" sessions based on the presence of both autowater states, reasoning that the two-context task is the relevant subset. The CONVERSION_NOTES.md documents this decision and notes the discrepancy with the paper's session counts.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from the filename using a regex pattern `([A-Za-z0-9]+)_(\d{4}-\d{2}-\d{2})`. Unique subjects are sorted and a `subject_idx` array maps each session to its subject.

ii.
```python
for df in data_files:
    m = re.match(r'^data_structure_([A-Za-z0-9]+)_(\d{4}-\d{2}-\d{2})\.mat$', df.name)
    ...
    subj, day = m.group(1), m.group(2)
    sessions.append(SessionRecord(subj, day, df, motion_map.get((subj, day))))
...
subjects = sorted({s.subject for s in sessions})
subj_to_idx = {s:i for i,s in enumerate(subjects)}
```

iii. Subject IDs are parsed from filenames, which is the same approach used by the reference (and the authors' own loading scripts).

## 1-c. How are the data split into sessions?

i. One session corresponds to one `data_structure_*.mat` file from `Ephys_Behavior`. Only sessions with both autowater states (context-capable) are kept. After context filtering, 22 sessions are found, but one (JEB6) is skipped due to unsupported `clu` structure, yielding 21 sessions. The `RandomizedDelay_Ephys_Behavior` directory is not searched.

ii.
```python
def discover_sessions() -> List[SessionRecord]:
    base = Path('data/Ephys_Behavior')
    data_files = sorted(base.glob('data_structure_*.mat'))
```

iii. The AI's CONVERSION_NOTES.md acknowledges the discrepancy with the paper's counts (21 vs the paper's reported numbers), noting that "additional curation is required before the conversion can be considered reference-matched."

## 1-d. How are the data split into trials?

i. Trials are determined by the number of go cue events: `n_trials = go.shape[0]`. Each trial index is used to access per-trial fields from the `bp` structure and per-trial spike data from `clu`.

ii.
```python
go = get_event(obj, ALIGN_EVENT)
n_trials = go.shape[0]
...
for tr in trial_idx:
    ...
```

iii. Trials correspond to rows of the Bpod behavior table, indexed by go cue events.

## 1-e. How are trials filtered based on quality controls?

i. Four filters are applied: (1) early lick trials removed, (2) photostimulation trials removed, (3) ignore trials removed (only hit OR miss trials kept), (4) trials must have valid R xor L direction labels. Additionally, sessions are filtered to only include those with both autowater states present.

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

iii. The AI noted (in CONVERSION_NOTES.md) a tension between the reference code (which uses hit-only trials for some analyses) and the decoder task (which requires predicting outcome). The AI chose to keep both hit and miss trials but exclude ignore trials, stating that this prevents the outcome output from being degenerate.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `obj.clu` cluster data. Specifically, for HDF5 files, it reads `clu.trial` (spike trial assignments, 1-based) and `clu.trialtm` (spike times relative to trial start). Go cue times `bp.ev.goCue` are also used.

ii.
```python
clu_root = obj['obj']['clu']
clu = obj[clu_root[0,0]]
n_clu = clu['trial'].shape[0]
for ci in range(n_clu):
    tr_ref = clu['trial'][ci,0]
    tm_ref = clu['trialtm'][ci,0]
    tr_arr = np.asarray(obj[tr_ref][()]).reshape(-1).astype(int)
    tm_arr = np.asarray(obj[tm_ref][()]).reshape(-1).astype(float)
```

iii. The AI correctly identified `clu.trial` and `clu.trialtm` as the source variables, matching the reference.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into time bins using `np.histogram`. The bin edges are centered around `time_bins` with half-bin-width offsets. The resulting counts are stored directly as float32 spike counts per bin, with NO conversion to firing rate (Hz) and NO temporal smoothing applied.

ii.
```python
binedges = np.concatenate([time_bins - BIN_SIZE_S/2, [time_bins[-1] + BIN_SIZE_S/2]]).astype(float)
...
for ci, (tr_arr, tm_arr) in enumerate(zip(clu_trial, clu_trialtm)):
    spikes = tm_arr[tr_arr == tr1]
    if spikes.size:
        mat[ci], _ = np.histogram(spikes, bins=binedges)
```

iii. The AI stores raw spike counts without converting to firing rates or applying Gaussian smoothing. The reference converts counts to Hz (divides by bin width) and applies a 14 ms Gaussian smooth.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No quality-based filtering is applied. All clusters in the `clu` structure are used regardless of their quality label. No firing rate threshold is applied. The only filtering is that sessions with unsupported `clu` structure (like JEB6) are skipped entirely.

ii.
```python
n_clu = clu['trial'].shape[0]
for ci in range(n_clu):
    ...
    clu_trial.append(tr_arr)
    clu_trialtm.append(tm_arr)
```

iii. The AI's CONVERSION_NOTES.md discusses neuron curation rules from the paper (>1 Hz, quality labels) but the code does not implement them.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data is NOT properly aligned to the go cue. The code selects spikes for a given trial using `tm_arr[tr_arr == tr1]`, which gives `trialtm` values (times relative to trial start). These are directly histogrammed into bins spanning -1.5 to 1.5 seconds WITHOUT subtracting the go cue time. Since `trialtm` is relative to trial start and the go cue typically occurs several seconds into the trial, the spikes are misaligned.

ii.
```python
tr1 = tr + 1
for ci, (tr_arr, tm_arr) in enumerate(zip(clu_trial, clu_trialtm)):
    spikes = tm_arr[tr_arr == tr1]
    if spikes.size:
        mat[ci], _ = np.histogram(spikes, bins=binedges)
```

Compare with reference:
```python
spike_time = np.asarray(cluster['trialtm'], float) - self.go_cue[spike_trial]
```

iii. The AI does not subtract the go cue time from `trialtm` before binning. The reference explicitly does `trialtm - goCue[trial]` to align spikes to the go cue.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 75 ms time bins spanning -1.5 to 1.5 seconds (41 time points). No rebinning is applied.

ii.
```python
BIN_SIZE_S = 0.075
T_START = -1.5
T_END = 1.5
...
time_bins = np.arange(T_START, T_END + 1e-9, BIN_SIZE_S, dtype=np.float32)
```

iii. The AI chose 75 ms based on `rez.binSize = 75` in the reference decoder scripts (`NeuralContextDecoding.m`). The reference solution uses 5 ms bins (matching `params.dt = 1/200`) spanning -2.5 to 2.5 s (1000 time points).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is derived from the time bin centers, computed as `np.arange(T_START, T_END + 1e-9, BIN_SIZE_S)`, giving 41 values from -1.5 to 1.5 s at 75 ms spacing.

ii.
```python
time_bins = np.arange(T_START, T_END + 1e-9, BIN_SIZE_S, dtype=np.float32)
...
inp = time_bins[None, :].astype(np.float32)
```

iii. Same concept as the reference - the input is the time axis itself.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is defined directly as bin centers from -1.5 to 1.5 s at 75 ms spacing. No additional processing.

ii.
```python
time_bins = np.arange(T_START, T_END + 1e-9, BIN_SIZE_S, dtype=np.float32)
```

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is the same time axis used for neural binning, so they share the same grid by construction.

ii.
```python
binedges = np.concatenate([time_bins - BIN_SIZE_S/2, [time_bins[-1] + BIN_SIZE_S/2]]).astype(float)
...
inp = time_bins[None, :].astype(np.float32)
```

iii. Both are defined from the same `time_bins` array.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived solely from `bp.R` (right trial indicator). The AI does not use `bp.hit` or `bp.miss` to determine actual lick direction.

ii.
```python
right = get_trial_bool(bp, 'R')
...
lick_dir = 1 if bool(right[tr]) else 0
```

iii. The AI treats the instructed side (`R`) as the lick direction, which is only correct for hit trials. On miss trials, the animal licked the opposite direction.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A direct mapping: right=1 if `bp.R` is True, left=0 otherwise. Only two classes are used (no "no lick" class). Since ignore trials are already filtered out, all remaining trials are assumed to have licked.

ii.
```python
lick_dir = 1 if bool(right[tr]) else 0
...
output_trials.append(np.vstack([
    np.full(time_bins.shape, lick_dir, dtype=np.int64),
    ...
]))
```

iii. The AI uses only 2 classes (left=0, right=1) vs. the reference's 3 classes (left=0, right=1, no lick=2). The AI does not derive lick direction from the combination of instructed side and outcome.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derived from `bp.autowater`. Autowater=True means WC context (0), autowater=False means DR context (1).

ii.
```python
autowater = get_trial_bool(bp, 'autowater')
...
context = 0 if bool(autowater[tr]) else 1
```

iii. Same source variable as the reference.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct relabelling: autowater=True -> WC=0, autowater=False -> DR=1.

ii.
```python
context = 0 if bool(autowater[tr]) else 1
```

iii. Same logic as the reference.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `bp.hit` only. Since ignore trials are already filtered out (the code requires `hit | miss`), the outcome is binary: hit=correct=1, miss=incorrect=0.

ii.
```python
hit = get_trial_bool(bp, 'hit')
...
outcome = 1 if bool(hit[tr]) else 0
```

iii. The AI uses `bp.hit` as the sole determinant.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A binary mapping: hit=1 (correct), not-hit=0 (incorrect). Only two classes are used (no "ignore" class) because ignore trials are filtered out during trial selection.

ii.
```python
outcome = 1 if bool(hit[tr]) else 0
```

iii. The reference uses 3 classes (incorrect=0, correct=1, ignore=2), keeping ignore trials with a third class. The AI removes ignore trials instead.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Derived from `obj.traj` DeepLabCut tracking data. The AI searches for tongue features by trying a list of candidate names: `['top_tongue', 'topleft_tongue', 'bottom_tongue', 'bottomleft_tongue', 'tongue', 'left_tongue', 'right_tongue']`. It uses `ts` (tracked coordinates), `featNames`, and `frameTimes`.

ii.
```python
tongue_vals = extract_hdf5_traj_velocity(obj, tr, ['top_tongue', 'topleft_tongue', 'bottom_tongue', 'bottomleft_tongue', 'tongue', 'left_tongue', 'right_tongue'], time_bins)
```

iii. The AI tries to find any tongue-like feature, using the first match from the candidate list. The reference uses both the side camera (`tongue`) and bottom camera (`top_tongue`) views and averages them.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI computes velocity as speed = sqrt(dx^2 + dy^2)/dt using frame-to-frame differences (np.diff with prepend). No likelihood filtering is applied. The speed is then interpolated onto the time bins using `np.interp`. Missing values (NaN) are filled using nearest-neighbor interpolation. Only one camera view is used (the first matching feature name). Only 2 discrete classes (low=0, high=1) are used with per-session 50th percentile threshold.

ii.
```python
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

iii. Major differences from reference: (1) No likelihood threshold filtering (reference uses >0.9), (2) No Gaussian smoothing of x,y before differentiation (reference uses 5ms Gaussian), (3) No per-run computation to avoid differentiating across gaps, (4) Only one camera view used (reference averages both after normalizing), (5) NaN filling by nearest interpolation rather than "not visible" class, (6) Only 2 output classes vs reference's 3.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per-session 50th percentile threshold applied. Values below threshold = 0 ("low"), values above = 1 ("high"). Only 2 classes (no "not visible" class).

ii.
```python
tongue_thr = np.nanpercentile(tongue_all, 50)
...
tmp = np.zeros(tong.shape, dtype=np.int64)
finite = np.isfinite(tong)
tmp[finite] = (tong[finite] >= tongue_thr).astype(np.int64)
output_trials[i][3] = tmp
```

iii. The reference uses 3 classes: below threshold=0, above threshold=1, not visible=2. The AI uses only 2 classes and assigns 0 to bins where the tongue is not tracked (since NaN is filled by interpolation or set to 0).

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are aligned to the go cue by subtracting `goCue[trial]` from `frameTimes`. However, the video-behavior clock offset (computed from bitcode in the reference) is NOT applied. The aligned velocities are then interpolated onto the neural time bins.

ii.
```python
go = float(np.asarray(obj['obj']['bp']['ev']['goCue'][()]).reshape(-1)[trial_index0])
rel_t = ft - go
vals = np.interp(time_bins, rel_t, speed, left=np.nan, right=np.nan)
```

iii. The reference computes a video offset using bitcode timestamps (`findVideoOffset.m`) and subtracts it before applying go cue alignment. The AI skips this offset correction.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Derived from `obj.traj` DeepLabCut tracking, searching candidate names: `['paw', 'left_paw', 'right_paw', 'top_paw', 'bottom_paw']`.

ii.
```python
paw_vals = extract_hdf5_traj_velocity(obj, tr, ['paw', 'left_paw', 'right_paw', 'top_paw', 'bottom_paw'], time_bins)
```

iii. The reference uses only `top_paw` from the bottom camera. The AI tries multiple candidates.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same processing as tongue velocity: frame-to-frame speed, no likelihood filtering, no Gaussian smoothing, nearest-neighbor NaN fill, interpolation onto time bins, 2-class discretization at session 50th percentile.

ii. Same `extract_hdf5_traj_velocity` function is used for both tongue and paw.

iii. Same issues as tongue velocity (7-b).

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: per-session 50th percentile, 2 classes (low=0, high=1), no "not visible" class.

ii.
```python
paw_thr = np.nanpercentile(paw_all, 50)
...
tmp[finite] = (paw[finite] >= paw_thr).astype(np.int64)
output_trials[i][4] = tmp
```

iii. Reference uses 3 classes including "not visible" = 2.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue velocity: `frameTimes - goCue`, then interpolated onto time bins. No video offset correction.

ii. Same `extract_hdf5_traj_velocity` function.

iii. Missing video offset correction, same as tongue.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Derived from `motionEnergy_*.mat` sidecar files. The `me.data` field is unwrapped through nested structs.

ii.
```python
def load_motion_energy(path: Optional[Path], n_trials: int) -> Optional[List[np.ndarray]]:
    ...
    x = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    me = x['me'].flat[0]
    data = np.asarray(unwrap_me_data(me)).reshape(-1)
```

iii. Same source file as the reference, though the unwrapping logic differs.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The raw per-trial motion energy (one value per frame) is mapped onto the time bins using linear interpolation via `np.interp`, assuming the raw data spans linearly from T_START to T_END. This is incorrect because the actual frame times are not evenly distributed over -1.5 to 1.5 s. Then discretized at per-session 50th percentile into 2 classes.

ii.
```python
x_old = np.linspace(T_START, T_END, raw.size)
me_binned = np.interp(time_bins, x_old, raw)
```

iii. The reference uses actual camera frame times (corrected by the video offset) to bin the motion energy. The AI fabricates a linear time axis, which is incorrect.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session 50th percentile threshold. Values below = 0, above = 1. NaN values are replaced by the session median before thresholding. Only 2 classes (no "no video" class).

ii.
```python
all_me = np.concatenate([np.nan_to_num(x, nan=np.nanmedian(np.concatenate(me_binned_all))) for x in me_binned_all])
thr = np.nanpercentile(all_me, 50)
...
tmp[finite] = (meb[finite] >= thr).astype(np.int64)
output_trials[i][5] = tmp
```

iii. Reference uses 3 classes including "no video" = 2. The AI fills NaN with the session median instead.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is mapped to time bins using `np.linspace(T_START, T_END, raw.size)` as a fabricated time axis, then linearly interpolated. No actual frame times are used, and no video offset is applied.

ii.
```python
x_old = np.linspace(T_START, T_END, raw.size)
me_binned = np.interp(time_bins, x_old, raw)
```

iii. The reference uses actual camera frame times with video offset correction, then bins values into the 5ms grid.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several approaches: (1) Sessions with unsupported `clu` structure are skipped entirely (JEB6). (2) Sessions with <2 valid trials are skipped. (3) Missing motion energy files result in NaN-filled arrays, which are then filled with the session median. (4) For trajectory extraction, broad try/except blocks catch and suppress errors, returning NaN arrays. (5) NaN values in velocity are filled by nearest-neighbor interpolation.

ii.
```python
except Exception as e:
    print(f'[warn] skipping {sess.data_file.name}: {e}', flush=True)
    continue
...
if np.isfinite(vals).any():
    idx = np.arange(vals.size)
    good = np.isfinite(vals)
    vals = np.interp(idx, idx[good], vals[good])
```

iii. The reference handles missing data by assigning a "not visible" class rather than interpolating/filling. The AI's approach of NaN-filling can fabricate values where data is genuinely missing.

## 11-a. What are the most time-consuming steps of the code?

i. The trajectory velocity extraction is by far the most expensive: each trial calls `extract_hdf5_traj_velocity` which accesses HDF5 references, decodes feature names, extracts coordinates, and computes velocity. This is called 3 times per trial (tongue computation is done twice: once for threshold computation and once for assignment), and each session takes 5-20 seconds. Total conversion takes ~200s for 21 sessions.

ii.
```python
tongue_all = np.concatenate([np.nan_to_num(extract_hdf5_traj_velocity(obj, tr, [...], time_bins), ...) for tr in trial_idx])
...
for i, tr in enumerate(trial_idx):
    tong = extract_hdf5_traj_velocity(obj, tr, [...], time_bins)
```

iii. The trajectory extraction is called redundantly (see 11-c).

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The main per-trial loop in `build_session` processes each trial sequentially, including neural spike histogram computation (one per cluster per trial) and trajectory velocity extraction. The spike counting could be vectorized using `np.histogram2d` across all trials at once (as the reference does). The trajectory extraction could potentially be vectorized per-session rather than per-trial.

ii.
```python
for ci, (tr_arr, tm_arr) in enumerate(zip(clu_trial, clu_trialtm)):
    spikes = tm_arr[tr_arr == tr1]
    if spikes.size:
        mat[ci], _ = np.histogram(spikes, bins=binedges)
```

iii. The reference uses `np.histogram2d` to count all spikes for a cluster across all trials in one call.

## 11-c. What processing does the code repeat multiple times?

i. Trajectory velocity extraction is computed TWICE for both tongue and paw features. First, it's computed for all trials to determine the per-session threshold (the `tongue_all` / `paw_all` concatenation). Then it's computed again for each trial to apply the threshold. This doubles the most expensive computation.

ii.
```python
# First computation for threshold
tongue_all = np.concatenate([... extract_hdf5_traj_velocity(obj, tr, [...], time_bins) ... for tr in trial_idx])
...
# Second computation for assignment
for i, tr in enumerate(trial_idx):
    tong = extract_hdf5_traj_velocity(obj, tr, [...], time_bins)
```

iii. This is a clear inefficiency - the per-trial velocities should be cached from the first pass.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `tongue_vals` and `paw_vals` inside the per-trial loop (lines 288-289) and stores them in the output array as zeros (lines 294-295), only to be overwritten later by the second pass of trajectory extraction. These initial computations are completely wasted.

ii.
```python
tongue_vals = extract_hdf5_traj_velocity(obj, tr, [...], time_bins) if isinstance(obj, h5py.File) else ...
paw_vals = extract_hdf5_traj_velocity(obj, tr, [...], time_bins) if isinstance(obj, h5py.File) else ...
output_trials.append(np.vstack([
    ...
    np.zeros(time_bins.shape, dtype=np.int64),  # tongue placeholder, overwritten later
    np.zeros(time_bins.shape, dtype=np.int64),  # paw placeholder, overwritten later
    np.zeros(time_bins.shape, dtype=np.int64),  # ME placeholder, overwritten later
]))
```

iii. The `tongue_vals` and `paw_vals` variables computed in the loop are never used - the output is initially set to zeros and overwritten by later code.
