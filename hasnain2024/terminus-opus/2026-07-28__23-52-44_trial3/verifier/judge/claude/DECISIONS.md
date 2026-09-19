# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each session is one MATLAB file, `data_structure_<anm>_<date>.mat`, stored in one of two folders (`Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior`). The 44 sessions and their probe numbers are hard-coded in `SESSION_META`, transcribed from the authors' loading scripts. The AI auto-detects the MATLAB format: it tries h5py first (v7.3 HDF5), and falls back to scipy.io (v5). Motion energy is loaded from separate `motionEnergy_*.mat` files. Unlike the reference, the AI reads specific fields from each file rather than recursively walking the entire data structure.

ii. Session list (excerpt):
```python
SESSION_META = [
    {'anm': 'EKH1', 'date': '2021-08-07', 'probe': 2, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    ...
    {'anm': 'JEB24', 'date': '2023-11-03', 'probe': 1, 'dir': 'RandomizedDelay_Ephys_Behavior', 'task': 'RandDelay'},
]
```

Loading:
```python
def load_session(filepath, probe_num):
    try:
        f = h5py.File(filepath, 'r')
        f.close()
        return load_session_h5(filepath, probe_num)
    except:
        return load_session_v5(filepath, probe_num)
```

iii. The AI notes it followed the loading scripts in `code/DataLoadingScripts/Recording and video/` to determine which sessions to include and which probes to use.

## 1-b. How are the data split into subjects?

i. The subject (animal) ID is taken from the `anm` field of each session's metadata entry in `SESSION_META`. Subjects are accumulated into a list as sessions are processed. The subject list is ordered by first appearance rather than sorted.

ii.
```python
subj = result['subject']
if subj not in all_subjects:
    all_subjects.append(subj)
subject_idx.append(all_subjects.index(subj))
```

iii. Each session in `SESSION_META` has an explicit `anm` field so no parsing is needed.

## 1-c. How are the data split into sessions?

i. Each entry in `SESSION_META` is one session, identified by `<anm>_<date>`. The AI stores the directory (`Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior`) directly in the metadata rather than searching both folders. Each session becomes one element of `neural`, `input`, and `output`.

ii.
```python
data_path = os.path.join('data', data_dir, f'data_structure_{session_id}.mat')
session_data = load_session(data_path, probe)
```

iii. Sessions are determined from the loading scripts, with commented-out sessions excluded.

## 1-d. How are the data split into trials?

i. Trials are indexed 0 to ntrials-1 within each session. The number of trials is read from `bp.Ntrials`. All per-trial fields are read with that length. Spike times carry a trial number, so no trial boundaries need to be reconstructed.

ii.
```python
ntrials = int(bp['Ntrials'][0, 0])
data['ntrials'] = ntrials
data['hit'] = bp['hit'][0, :].astype(bool)
...
```

iii. The AI reads the trial count from `bp.Ntrials` and uses it to slice all per-trial fields.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies three filters: (1) early-lick trials are excluded, (2) photostimulation trials (`stim.enable`) are excluded, and (3) **ignore/no-response trials are excluded** — only hit or miss trials are kept. This is more aggressive than the reference, which keeps ignore trials.

ii.
```python
valid_trial_mask = ~session_data['early'] & ~session_data['no'] & ~session_data['stim_enable']
valid_trial_mask = valid_trial_mask & (session_data['hit'] | session_data['miss'])
```

iii. The AI's CONVERSION_NOTES states: "Exclude early, no-response, and stim trials." The AI interpreted the task as requiring only hit/miss trials, but the instructions specify "ignore" as an outcome class, which requires keeping ignore trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data comes from `obj.clu{probe}`, specifically the `trialtm` (spike times relative to trial start), `trial` (trial number), and `quality` (curation label) fields. Go cue times `bp.ev.goCue` are used for alignment.

ii.
```python
trialtm = clu['trialtm']
trial_nums = clu['trial']
aligned_times = trialtm[spike_mask] - goCue[trial_idx]
counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. Same variables as the reference.

## 2-b. How is the `neural` data processed?

i. Spike counts are computed in 5 ms bins, converted to firing rates (Hz) by dividing by dt, then smoothed with a **causal** Gaussian kernel of window size 15. The causal kernel zeros out the first half of a `gausswin(15)` and normalizes. This differs from the reference, which uses a symmetric Gaussian with sigma=14ms (scipy's `gaussian_filter1d`).

ii.
```python
def make_causal_gaussian_kernel(N):
    kern = np.array(sig_windows.gaussian(N, std=np.std(np.arange(N))))
    kern[:N // 2] = 0  # causal: zero out first half
    kern = kern / kern.sum()
    return kern

fr = counts.astype(np.float32) / dt
fr_smooth = smooth_causal(fr, params['smooth_window'], bctype='none')
```

iii. The AI's CONVERSION_NOTES says: "causal Gaussian, N=15" matching `mySmooth.m`. The reference code's `mySmooth.m` does implement causal smoothing, so this is a reasonable interpretation. However, the reference solution uses symmetric smoothing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied. First, clusters with quality labels in `{garbage, gabrga, noisy, real?}` are excluded. The AI does NOT exclude `poor` quality units, unlike the reference. Second, units with mean firing rate below 0.5 Hz are removed (the reference uses 1.0 Hz). Sessions with fewer than 10 units after filtering are skipped.

ii.
```python
excluded = params['excluded_qualities']  # {'garbage', 'gabrga', 'noisy', 'real?'}
...
if q in excluded:
    continue
...
mean_fr = np.mean(np.mean(trialdat, axis=2), axis=0)
keep_mask = mean_fr > params['low_fr']  # 0.5 Hz
```

iii. The AI notes: "FR threshold: 0.5 Hz (code default)" and "Quality filter matches findClusters.m (keeps empty quality strings)." The reference uses 1.0 Hz from the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue by subtracting `goCue[trial_idx]` from each spike's `trialtm`. This is the same approach as the reference.

ii.
```python
aligned_times = trialtm[spike_mask] - goCue[trial_idx]
counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. Follows the reference's `alignSpikes.m`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 5 ms bins spanning -2.5 to 2.5 s from the go cue, producing 1000 time points. Same as the reference.

ii.
```python
tmin, tmax, dt = params['tmin'], params['tmax'], params['dt']  # -2.5, 2.5, 0.005
edges = np.arange(tmin, tmax + dt, dt)
time_axis = edges[:-1] + dt / 2
```

iii. Matches `params.dt = 1/200` and `params.tmin/tmax` from `getDefaultParams.m`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the time axis itself — the bin centers of the 1000 bins spanning -2.5 to 2.5 s, defined by the binning parameters. Same as the reference.

ii.
```python
time_axis = edges[:-1] + dt / 2
time_input = time_axis.reshape(1, -1).astype(np.float32)
```

iii. No raw data variable needed; it is derived from the binning grid.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing — the time axis is defined directly from the bin edges.

ii. N/A

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input IS the binning grid, so it is inherently aligned. The same `time_axis` defines both the neural bins and the input.

ii.
```python
time_input = time_axis.reshape(1, -1).astype(np.float32)
```

iii. Same approach as the reference.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derives lick direction from `bp.R` (the instructed right-lick indicator), using it directly as the output. This means the output is the **instructed direction**, NOT the **actual lick direction**. The reference derives actual lick direction from the combination of `bp.R`, `bp.hit`, and `bp.miss`.

ii.
```python
lick_direction = session_data['R'][valid_trial_indices].astype(int)
```

iii. The AI's CONVERSION_NOTES says: "bp.R → output[0] (lick_direction): L=0, R=1, per-trial." This conflates instructed direction with lick direction.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI simply takes `bp.R` as an integer (0=left, 1=right). Since ignore trials are already excluded, there is no "no lick" class. The reference derives the actual lick direction: hit+right = right lick, hit+left = left lick, miss+right = left lick, miss+left = right lick, ignore = no lick.

ii.
```python
lick_direction = session_data['R'][valid_trial_indices].astype(int)
```

iii. The AI only has 2 classes, while the instructions specify 3 classes (left, right, none) for lick direction.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derived from `bp.autowater`. Same variable as the reference.

ii.
```python
context = (session_data['autowater'][valid_trial_indices] == 0).astype(int)
```

iii. Matches the reference approach.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. `autowater == 0` maps to DR (1), and autowater trials to WC (0). This is the same encoding as the reference.

ii.
```python
context = (session_data['autowater'][valid_trial_indices] == 0).astype(int)
```

iii. WC=0, DR=1 matches the instruction specification.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `bp.hit`. Since the AI excludes ignore trials, outcome is simply whether the trial was a hit.

ii.
```python
outcome = session_data['hit'][valid_trial_indices].astype(int)
```

iii. The reference uses both `bp.hit` and `bp.miss` and includes a third class for ignore.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps hit=1 (correct) and not-hit=0 (incorrect). Since ignore trials are excluded, this is a binary classification. The instructions specify three classes: incorrect (0), correct (1), ignore (2).

ii.
```python
outcome = session_data['hit'][valid_trial_indices].astype(int)
```

iii. Missing the "ignore" class specified in the instructions.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Derived from `obj.traj` — specifically the side camera (camera 0) tongue feature. The AI only uses one camera view, while the reference uses both the side camera (`tongue`) and bottom camera (`top_tongue`).

ii.
```python
cam0 = session_data['traj'][0]  # side cam
feat_names = cam0['featNames']
tongue_idx = None
for fi, fn in enumerate(feat_names):
    if fn.lower() == 'tongue':
        tongue_idx = fi
```

iii. The AI's CONVERSION_NOTES says "Tongue velocity: Fill NaN positions with baseline, set NaN velocity to 0."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI interpolates x, y positions from frame times to the neural time axis using `interp1d`, fills NaN positions with the session mean ("baseline"), computes velocity as `np.gradient` of the filled positions, sets velocity to 0 where the tongue was originally not visible, then computes speed as `sqrt(xvel^2 + yvel^2)`. This differs significantly from the reference, which: (1) computes velocity at frame resolution within contiguous tracked runs, (2) applies Gaussian smoothing before differentiating, (3) uses both camera views and normalizes by the 90th percentile before averaging, (4) bins into 5ms bins, and (5) uses a "not visible" class (value 2) instead of setting velocity to 0.

ii.
```python
# Fill NaN tongue positions with session mean (baseline)
all_x_filled[np.isnan(all_x_filled)] = mean_x
all_y_filled[np.isnan(all_y_filled)] = mean_y

xvel = np.gradient(all_x_filled[:, trial_idx])
yvel = np.gradient(all_y_filled[:, trial_idx])
nan_mask = np.isnan(all_x[:, trial_idx])
xvel[nan_mask] = 0.0
yvel[nan_mask] = 0.0
speed = np.sqrt(xvel**2 + yvel**2)
```

iii. The AI's approach of filling NaN with baseline and setting untracked velocity to 0 is problematic because (1) it creates artificial zero-velocity values that affect the percentile threshold, and (2) it lacks the "not visible" category. The tongue threshold is 0.00 for all sessions in the output, indicating that the velocity is essentially always 0 when the tongue isn't visible, making the discretization degenerate.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI uses the 50th percentile of non-NaN values to split into 2 classes (0=low, 1=high). NaN values default to 0. There is no "not visible" (value 2) class. The output shows tongue_velocity as 90.4% "low" and 9.6% "high", suggesting the threshold is degenerate since the tongue is only visible ~10% of the time.

ii.
```python
def discretize_velocity(vel_data):
    threshold = np.percentile(valid_vals, 50)
    if threshold <= np.min(valid_vals) + 1e-10:
        discretized = (vel_data > threshold).astype(int)
    else:
        discretized = (vel_data >= threshold).astype(int)
    discretized[np.isnan(vel_data)] = 0  # default for NaN
    return discretized, threshold
```

iii. Only 2 classes instead of the 3 specified in the instructions (0: < 50th percentile, 1: >= 50th percentile, 2: not visible).

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI interpolates camera frame data directly onto the neural time axis using `interp1d`, after correcting frame times with the video offset and go cue time. The reference instead bins frame-resolution data into 5ms bins.

ii.
```python
aligned_ft = ft[:len(x)] - vidshift - goCue[trial_idx]
fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
fy = interp1d(aligned_ft, y, kind='linear', bounds_error=False, fill_value=np.nan)
all_x[:, trial_idx] = fx(time_axis)
all_y[:, trial_idx] = fy(time_axis)
```

iii. The video offset computation uses `nanmedian` for bitStart instead of `mode`, which may differ slightly.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Derived from `obj.traj` bottom camera (camera 1), specifically the `top_paw` feature. Same feature as the reference.

ii.
```python
cam1 = session_data['traj'][1]  # top cam
for fi, fn in enumerate(feat_names):
    if 'top_paw' in fn.lower():
        paw_idx = fi
```

iii. Uses the same feature as the reference.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The AI interpolates x, y positions to the neural time axis, fills NaN with nearest neighbor interpolation, computes velocity via `np.gradient`, subtracts a baseline derivative (median of diff), and computes speed. This differs from the reference, which computes velocity at frame resolution within contiguous tracked runs with Gaussian smoothing.

ii.
```python
fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
fy = interp1d(aligned_ft, y, kind='linear', bounds_error=False, fill_value=np.nan)
x_interp = fx(time_axis)
y_interp = fy(time_axis)
for arr in [x_interp, y_interp]:
    nans = np.isnan(arr)
    if np.any(nans) and np.any(~nans):
        arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans])
xvel = np.gradient(x_interp)
yvel = np.gradient(y_interp)
basederiv_x = np.nanmedian(np.diff(x_interp))
basederiv_y = np.nanmedian(np.diff(y_interp))
xvel -= basederiv_x
yvel -= basederiv_y
```

iii. The baseline derivative subtraction is not present in the reference code. Filling NaN with nearest neighbor removes information about when the paw was not visible.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue velocity — 2 classes (0=low, 1=high) with no "not visible" class. NaN defaults to 0. The instructions specify 3 classes with "not visible" (value 2).

ii.
```python
paw_disc, paw_thresh = discretize_velocity(valid_paw_vel)
```

iii. Missing the "not visible" class.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same approach as tongue velocity — interpolated to neural time axis using `interp1d` with video offset correction.

ii.
```python
aligned_ft = ft[:len(x)] - vidshift - goCue[trial_idx]
fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
```

iii. Uses interpolation rather than binning.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Loaded from separate `motionEnergy_<anm>_<date>.mat` files. Handles multiple formats (standard, nested, direct cell array). Same source as the reference.

ii.
```python
def load_motion_energy(me_filepath):
    d = sio.loadmat(me_filepath, squeeze_me=False)
    me_raw = d['me']
    # handles 3 formats...
```

iii. The AI correctly handles multiple ME file formats.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI interpolates motion energy from frame times onto the neural time axis using `interp1d`, then fills remaining NaN values with nearest-neighbor interpolation. The reference bins frame-resolution data into 5ms bins using averaging and leaves NaN where no frames fall.

ii.
```python
f_interp = interp1d(aligned_ft, trial_me,
                     kind='linear', bounds_error=False, fill_value=np.nan)
me_aligned[:, trial_idx] = f_interp(time_axis)

# Fill NaN with nearest
col[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), col[not_nans])
```

iii. The NaN filling means there is no "no video" class for motion energy, whereas the reference preserves NaN for bins with no camera frames and uses a third class (value 2).

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same as other movement outputs — 2 classes (0=low, 1=high) with no "no video" class. The instructions specify 3 classes with "no video" (value 2).

ii.
```python
me_disc, me_thresh = discretize_velocity(valid_me)
```

iii. Missing the "no video" class.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Interpolated to neural time axis from camera frame times, corrected with video offset and go cue. When ME length doesn't match frame times length, an alternative alignment at 400 Hz is attempted. The reference uses bin averaging.

ii.
```python
aligned_ft = ft - vidshift - goCue[trial_idx]
if len(trial_me) == len(ft):
    f_interp = interp1d(aligned_ft, trial_me, kind='linear', bounds_error=False, fill_value=np.nan)
    me_aligned[:, trial_idx] = f_interp(time_axis)
else:
    me_times = np.arange(len(trial_me)) / 400.0
    aligned_me_times = me_times - 0.5 - goCue[trial_idx] + session_data['bitStart'][trial_idx]
    f_interp = interp1d(aligned_me_times, trial_me, ...)
```

iii. Uses interpolation rather than bin averaging.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: (1) trials with all-zero neural data are removed (30 total across 2 sessions), (2) NaN frame times cause trials to be skipped for camera data, (3) NaN in tongue velocity is filled with zeros, (4) NaN in paw velocity is filled via nearest-neighbor interpolation, (5) NaN in motion energy is filled via nearest-neighbor interpolation. The reference instead preserves NaN gaps and assigns them the "not visible" class.

ii.
```python
# Remove trials with all-zero neural data
if np.all(ni == 0):
    n_removed += 1
    continue

# Fill NaN with nearest for motion energy
col[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), col[not_nans])
```

iii. The AI's approach of filling/removing missing data differs from the reference's approach of preserving the information about data absence through the "not visible" class.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the MATLAB files dominates runtime. The full conversion runs in ~248 seconds. The spike binning is also slow because it loops over neurons AND trials individually rather than using vectorized histogram2d.

ii.
```python
for neuron_idx, clu in enumerate(valid_clusters):
    for trial_idx in range(ntrials):
        spike_mask = trial_nums == trial_num
        aligned_times = trialtm[spike_mask] - goCue[trial_idx]
        counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. The AI's CONVERSION_NOTES estimates ~6s per session.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop iterates over every neuron AND every trial individually with a boolean mask, which is very inefficient. The reference uses a single `np.histogram2d` call per cluster to bin all trials at once. The tongue and paw velocity loops over trials could also be partially vectorized.

ii.
```python
for neuron_idx, clu in enumerate(valid_clusters):
    for trial_idx in range(ntrials):
        spike_mask = trial_nums == trial_num
        aligned_times = trialtm[spike_mask] - goCue[trial_idx]
        counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. The double loop (neurons x trials) is the most obvious vectorization opportunity.

## 11-c. What processing does the code repeat multiple times?

i. The `_traj` function re-reads trajectory data for each feature and trial. The go cue alignment is computed separately for neural, tongue, paw, and motion energy streams rather than being computed once. The spike masking `trial_nums == trial_num` is recomputed for every neuron on every trial.

ii.
```python
# Go cue read multiple places:
goCue = session_data['goCue']
# Then used in spike binning, tongue, paw, ME processing separately
```

iii. The repeated trajectory reading could be factored out.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads lick times (`lickL`, `lickR`) from the data but never uses them for any output variable. It also loads `sample` and `delay` event times that are not used. The `NdroppedFrames` field is loaded but only used as a check (skipping if NaN), not for computation. The baseline derivative subtraction for paw velocity is an unnecessary processing step not in the reference.

ii.
```python
data['lickL'] = []
data['lickR'] = []
data['sample'] = ev['sample'][0, :]
data['delay'] = ev['delay'][0, :]
```

iii. These fields are loaded but never contribute to the final output.
