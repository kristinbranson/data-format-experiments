# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by globbing the two data directories (`Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior`) for files matching `data_structure_*.mat`. It skips one known duplicate (`data_structure_JEB23_2023-10-20.mat`). Each session file is loaded using either `h5py` (for HDF5/v7.3 MATLAB files) or `scipy.io.loadmat` (for older MATLAB files), with the format auto-detected by trying h5py first and falling back to scipy on failure. Motion energy is loaded from separate `motionEnergy_*.mat` files when available, or from embedded `obj.me` within the data structure.

ii. Session discovery:
```python
def discover_sessions(sample=False):
    sessions = []
    for dir_key, data_dir in DATA_DIRS.items():
        for fn in sorted(os.listdir(data_dir)):
            if not fn.startswith('data_structure_'):
                continue
            if fn in SKIP_FILES:
                continue
            ...
```

Loading:
```python
def load_session(fpath):
    try:
        session = load_session_h5(fpath)
        return session
    except:
        pass
    try:
        session = load_session_scipy(fpath)
        return session
    except Exception as e:
        print(f"  ERROR loading {fpath}: {e}")
        return None
```

iii. The AI documented in CONVERSION_NOTES.md that it found 47 total sessions across the two directories (25 Ephys + 22 RandomizedDelay), then excluded the duplicate and two behavior-only sessions (JEB24_10-03, JEB24_10-04) that had no neural data, arriving at 44 sessions. The approach of globbing the directories differs from the reference solution, which hard-codes the 44 session names and probe assignments from the authors' `load<ANM>_ALMVideo.m` scripts.

## 1-b. How are the data split into subjects?

i. The animal name is extracted from the filename by splitting `data_structure_<animal>_<date>.mat` on underscores. A sorted set of unique animal names forms the `subjects` list, and `subject_idx` maps each session to its subject.

ii.
```python
parts = fn.replace('.mat', '').split('_')
animal = parts[2]
...
subjects_set = sorted(set(s['animal'] for s in sessions_info))
subject_to_idx = {s: i for i, s in enumerate(subjects_set)}
```

iii. The AI documented finding 14 subjects with neural data across both directories. This matches the reference.

## 1-c. How are the data split into sessions?

i. One session is one `data_structure_*.mat` file. The AI discovers them by directory listing. Both Ephys_Behavior and RandomizedDelay_Ephys_Behavior directories are searched. Sessions without neural data are skipped, and a duplicate file is excluded via `SKIP_FILES`. The result is 44 sessions.

ii.
```python
SKIP_FILES = {
    'data_structure_JEB23_2023-10-20.mat',  # duplicate of JEB23_2023-10-19
}
```

iii. The AI documented detecting the duplicate via data comparison and excluding it. The final count of 44 sessions (25 + 19) matches the paper.

## 1-d. How are the data split into trials?

i. Trials are defined by `Ntrials` in `obj.bp`. The AI reads `session['Ntrials']` and uses indices 0 to Ntrials-1. Each per-trial field (hit, miss, early, etc.) is read as a flat array and indexed by trial number.

ii.
```python
n_trials = session['Ntrials']
...
valid_trials = np.ones(n_trials, dtype=bool)
valid_trials[session['early']] = False
valid_trials[session['stim_enable']] = False
valid_trials[np.isnan(goCue)] = False
trial_indices = np.where(valid_trials)[0]
```

iii. The trial definition follows the standard approach of using the Bpod trial table. The AI also excludes trials with NaN go cue times, which is a reasonable safety check.

## 1-e. How are trials filtered based on quality controls?

i. Three filters are applied: (1) early lick trials are excluded, (2) photostimulation trials (stim.enable) are excluded, (3) trials with NaN go cue times are excluded. Additionally, sessions with fewer than 10 units after neural quality filtering are excluded entirely. Unlike the reference, the AI does NOT filter out trials that occur after the recording ends, which results in all-zero neural data for late trials in two sessions (36 and 43).

ii.
```python
valid_trials[session['early']] = False
valid_trials[session['stim_enable']] = False
valid_trials[np.isnan(goCue)] = False
...
if n_units < MIN_UNITS:
    print(f"  {session_id}: Too few units after filtering ({n_units}), skipping")
    return None
```

iii. The AI documented in CONVERSION_NOTES.md that early lick and stim trials are excluded per the paper, and that a minimum of 10 units is required per session. The AI noted that sessions 36 and 43 have all-zero late trials but did not implement a fix, calling them "data artifacts."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses `obj.clu` spike-sorted clusters. For each neuron, it reads `trialtm` (spike times relative to trial start), `trial` (which trial each spike belongs to), and `quality` (curation label). The go cue times `bp.ev.goCue` provide the alignment event.

ii.
```python
neurons.append({
    'quality': quality,
    'trialtm': trialtm,
    'trial': trial,
})
```

iii. These are the same source variables as the reference.

## 2-b. How is the `neural` data processed?

i. Spikes are binned at 10 ms resolution into time bins from -2.5 to 2.5 s around the go cue, producing 500 bins. The binned counts are converted to firing rates (Hz) by dividing by the bin width (0.01 s). The firing rates are then smoothed with a causal Gaussian kernel of 15 samples (150 ms window at 10 ms bins).

ii.
```python
DT = 0.01            # 10 ms time bins (params.dt = 1/100)
SMOOTH_WIN = 15       # causal Gaussian window size (samples)
...
fr = counts.astype(np.float32) / dt
fr = smooth_causal(fr, KERNEL, 'reflect')
```

Causal kernel:
```python
def causal_gaussian_kernel(N):
    alpha = 2.5
    n = np.arange(N)
    w = np.exp(-0.5 * ((n - (N - 1) / 2) / (alpha * (N - 1) / 2)) ** 2)
    w[:N // 2] = 0  # Zero out first half (causal)
    w = w / w.sum()
    return w
```

iii. The AI interpreted `params.dt = 1/100` as 10 ms bins (100 Hz). The reference code parameter is actually `params.dt = 1/200` (5 ms bins, 200 Hz). The smoothing is implemented as a causal Gaussian matching the MATLAB `mySmooth.m` function, which zeroes out the first half of the kernel. The reference solution instead uses a symmetric Gaussian with sigma=14 ms via `scipy.ndimage.gaussian_filter1d`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons with quality labels in `{'garbage', 'noisy', 'gabrga', 'real?', ''}` are excluded. Then neurons with mean firing rate below 1 Hz (`>= 1.0 Hz` threshold) are removed. This produces 3,066 units across 44 sessions.

ii.
```python
EXCLUDE_QUALITY = {'garbage', 'noisy', 'gabrga', 'real?', ''}
LOW_FR_THRESH = 1.0
...
if quality in EXCLUDE_QUALITY:
    continue
...
keep_neurons = mean_fr >= LOW_FR_THRESH
```

iii. The AI's quality filter differs from the reference in two ways: (1) it excludes empty string labels but does not exclude `'poor'`, while the reference excludes `'poor'` but keeps empty strings; (2) the firing rate threshold uses `>=` instead of `>`. The net result is 3,066 neurons vs. the reference's ~1,954, a significant discrepancy.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times (`trialtm`) are aligned to the go cue by subtracting the trial's go cue time. The aligned times are then binned into the time bin edges. This is done per-neuron, per-trial.

ii.
```python
aligned = trialtm[spike_mask] - gc
counts, _ = np.histogram(aligned, bins=edges)
```

iii. The alignment is correct: subtracting go cue from trialtm is the standard approach matching `alignSpikes.m`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 10 ms time bins (DT = 0.01), producing 500 bins over the [-2.5, 2.5] s window. No rebinning is applied; spikes are directly counted into these bins.

ii.
```python
DT = 0.01            # 10 ms time bins (params.dt = 1/100)
EDGES = np.arange(TMIN, TMAX + DT, DT)
N_TIMEBINS = len(TIME_CENTERS)  # 500
```

iii. The AI justified this as matching the reference code default `params.dt = 1/100`. However, the reference code actually uses `params.dt = 1/200 = 0.005` (5 ms bins), which the reference solution correctly implements with 1000 bins.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This is a synthetic variable: the time axis of the binning grid, computed as the centers of the 500 time bins spanning -2.5 to 2.5 s.

ii.
```python
TIME_CENTERS = EDGES[:-1] + DT / 2
...
time_input = TIME_CENTERS.astype(np.float32)
result['input'].append(time_input[np.newaxis, :])
```

iii. Identical approach to the reference, just with different bin size (10 ms vs 5 ms).

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing beyond computing bin centers from the bin edges. The time axis is defined by the binning parameters.

ii. N/A

iii. N/A

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time input is the center of the same bins used for neural data, so alignment is intrinsic. The same `TIME_CENTERS` array is used for both.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT, DT)
TIME_CENTERS = EDGES[:-1] + DT / 2
```

iii. Same approach as the reference.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial fields: `bp.hit`, `bp.miss`, `bp.L`, and `bp.R`. Hit/miss indicate outcome, L/R indicate instructed side. The lick direction is inferred from the combination.

ii.
```python
if session['hit'][t]:
    if session['L'][t]:
        lick_dir[out_idx] = 0  # left
    elif session['R'][t]:
        lick_dir[out_idx] = 1  # right
elif session['miss'][t]:
    if session['L'][t]:
        lick_dir[out_idx] = 1  # licked right (wrong)
    elif session['R'][t]:
        lick_dir[out_idx] = 0  # licked left (wrong)
```

iii. The AI correctly derives lick direction from the combination of instructed side and outcome, matching the reference approach.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A hit means the animal licked the instructed port (so lick = instructed side). A miss means the animal licked the opposite port (so lick = opposite of instructed side). Ignore trials get the default "none" class (2). Codes: left=0, right=1, none=2.

ii.
```python
lick_dir = np.full(n_valid, 2, dtype=np.int32)  # default: none
for out_idx, t in enumerate(trial_indices):
    if session['hit'][t]:
        if session['L'][t]:
            lick_dir[out_idx] = 0
        elif session['R'][t]:
            lick_dir[out_idx] = 1
    elif session['miss'][t]:
        if session['L'][t]:
            lick_dir[out_idx] = 1
        elif session['R'][t]:
            lick_dir[out_idx] = 0
```

iii. The logic matches the reference solution. The AI uses a per-trial loop instead of vectorized operations but the result is equivalent.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The per-trial field `bp.autowater`. When autowater is 1, the trial is in the water-cued (WC) context; otherwise it is delayed-response (DR).

ii.
```python
session['autowater'] = np.array(bp['autowater']).flatten().astype(int)
...
if session['autowater'][t] == 1:
    context[out_idx] = 0
```

iii. Same source variable as the reference.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct mapping: autowater=1 maps to WC (0), everything else maps to DR (1).

ii.
```python
context = np.full(n_valid, 1, dtype=np.int32)  # default: DR
for out_idx, t in enumerate(trial_indices):
    if session['autowater'][t] == 1:
        context[out_idx] = 0
```

iii. Matches the reference logic.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Two per-trial fields: `bp.hit` and `bp.miss`. Neither hit nor miss implies ignore.

ii.
```python
session['hit'] = np.array(bp['hit']).flatten().astype(bool)
session['miss'] = np.array(bp['miss']).flatten().astype(bool)
```

iii. Same source variables as the reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Three classes: hit maps to correct (1), miss maps to incorrect (0), and everything else maps to ignore (2).

ii.
```python
outcome = np.full(n_valid, 2, dtype=np.int32)  # default: ignore
for out_idx, t in enumerate(trial_indices):
    if session['hit'][t]:
        outcome[out_idx] = 1
    elif session['miss'][t]:
        outcome[out_idx] = 0
```

iii. Matches the reference logic and category codes.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DLC tracking data from `obj.traj`, specifically the `tongue` feature from camera 0 (one camera only). The AI uses the `ts` array (features x [x,y,confidence] x frames), `frameTimes`, and `featNames` from a single camera view.

ii.
```python
tongue_vel, tongue_vis = compute_velocity_timeseries(
    session, cam_idx=0, feat_name='tongue',
    trial_indices=trial_indices, vidshift=vidshift)
```

iii. The reference solution uses BOTH cameras (side camera 'tongue' and bottom camera 'top_tongue'), normalizes each by its 90th percentile, and averages them. The AI only uses one camera view.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial: (1) Extract x, y, confidence from the tracking data. (2) Compute velocity as `sqrt(diff(x)^2 + diff(y)^2) * 400 Hz` (raw pixel differences times frame rate). (3) Set velocity to NaN where confidence < 0.5. (4) Interpolate velocity to the neural time bin centers using `np.interp`. (5) Discretize per-session at the 50th percentile of visible values: 0=below, 1=above, 2=not visible.

ii.
```python
dx = np.diff(x)
dy = np.diff(y)
v = np.sqrt(dx**2 + dy**2) * VIDEO_FPS  # pixels/second
v = np.concatenate([[0], v])
low_conf = conf < DLC_CONF_THRESH
v[low_conf] = np.nan
...
v_interp = np.interp(taxis, aligned_ft[valid], v[valid], left=np.nan, right=np.nan)
```

iii. Several differences from the reference: (1) No Gaussian smoothing of x,y before differentiating (reference uses 5 ms Gaussian). (2) Uses `np.diff` instead of `np.gradient` for derivatives. (3) Multiplies by constant FPS instead of dividing by actual frame time differences. (4) Interpolates to bin centers instead of bin-averaging. (5) Confidence threshold is 0.5 instead of 0.9. (6) Does not handle contiguous runs of valid frames separately.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The 50th percentile of all visible (confidence >= 0.5) velocity values across the session is computed. Values below the threshold get class 0, at or above get class 1, and non-visible bins get class 2.

ii.
```python
def discretize_velocity(velocity, visible):
    result = np.full((n_time, n_trials), 2, dtype=np.int32)
    vis_vals = velocity[visible]
    thresh = np.nanpercentile(vis_vals, 50)
    result[visible & (velocity < thresh)] = 0
    result[visible & (velocity >= thresh)] = 1
    return result
```

iii. The discretization logic matches the reference in principle (50th percentile split), but the percentile is computed only over visible values rather than over all values including NaN (the reference uses `nanpercentile` which already ignores NaN). The visibility definition differs due to the different confidence threshold (0.5 vs 0.9).

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by subtracting the video offset (`vidshift`) and the trial's go cue time. The corrected frame times are then used to interpolate velocity values to the neural time bin centers.

ii.
```python
aligned_ft = ft - vidshift - gc
...
v_interp = np.interp(taxis, aligned_ft[valid], v[valid], left=np.nan, right=np.nan)
```

iii. The video offset computation matches the reference (mode of sglx bitcode start / fs minus mode of bp bitStart). However, interpolation to bin centers differs from the reference's approach of averaging all frames within each bin.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The DLC tracking data from `obj.traj`, specifically the `top_paw` feature from camera 1 (bottom camera).

ii.
```python
paw_vel, paw_vis = compute_velocity_timeseries(
    session, cam_idx=1, feat_name='top_paw',
    trial_indices=trial_indices, vidshift=vidshift)
```

iii. The reference also uses `top_paw` from the bottom camera. The camera index assignment matches.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same pipeline as tongue velocity: raw pixel differences times frame rate, confidence threshold at 0.5, interpolation to bin centers, then discretization at 50th percentile.

ii.
```python
paw_vel, paw_vis = compute_velocity_timeseries(
    session, cam_idx=1, feat_name='top_paw', ...)
paw_disc = discretize_velocity(paw_vel, paw_vis)
```

iii. Same differences from reference as tongue velocity: no Gaussian smoothing, `np.diff` instead of `np.gradient`, constant FPS instead of actual time differences, interpolation instead of binning, and lower confidence threshold.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: 50th percentile of visible values, split into below (0), above (1), not visible (2).

ii.
```python
paw_disc = discretize_velocity(paw_vel, paw_vis)
```

iii. Same approach as tongue discretization. Matches reference in concept.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: frame times corrected by video offset and go cue, then interpolated to bin centers.

ii.
```python
aligned_ft = ft - vidshift - gc
v_interp = np.interp(taxis, aligned_ft[valid], v[valid], left=np.nan, right=np.nan)
```

iii. Same approach as tongue alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from separate `motionEnergy_*.mat` files when available, or from embedded `obj.me` within the data structure. The AI tries `obj.me` first (from the h5py/scipy loading), then falls back to the separate file.

ii.
```python
me_trials = session.get('me_embedded', None)
has_me_file = me_path is not None and os.path.exists(me_path)
if has_me_file:
    me_trials_loaded, me_thresh = load_motion_energy_file(me_path)
    me_aligned = compute_motion_energy_timeseries(session, trial_indices, me_trials_loaded, vidshift)
elif me_trials is not None:
    me_aligned = compute_motion_energy_timeseries(session, trial_indices, me_trials, vidshift)
```

iii. The reference always loads from the separate `motionEnergy_*.mat` file. The AI's approach of preferring the file when available is similar, though the loading code handles the nested dict structure differently.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy values (already one number per frame) are aligned to the go-cue clock using frame times, then interpolated to the neural time bin centers. No additional smoothing or processing is applied.

ii.
```python
me_interp = np.interp(taxis, aligned_ft[valid], me_data[valid], left=np.nan, right=np.nan)
```

iii. Same as the reference in that no additional processing is done. The difference is interpolation vs bin-averaging.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. 50th percentile of non-NaN values across the session. Below = 0, above = 1, NaN (no video) = 2.

ii.
```python
def discretize_motion_energy(me_aligned):
    result = np.full((n_time, n_trials), 2, dtype=np.int32)
    valid = ~np.isnan(me_aligned)
    thresh = np.nanpercentile(valid_vals, 50)
    result[valid & (me_aligned < thresh)] = 0
    result[valid & (me_aligned >= thresh)] = 1
    return result
```

iii. Matches the reference approach.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Camera 0 frame times are used, corrected by video offset and go cue time, then motion energy is interpolated to bin centers.

ii.
```python
cam = session['traj'][0]  # Use camera 0 (bottom) frame times
aligned_ft = ft - vidshift - gc
me_interp = np.interp(taxis, aligned_ft[valid], me_data[valid], left=np.nan, right=np.nan)
```

iii. The reference uses side camera (index 0) frame times and bin-averages rather than interpolates. The camera index matches (both use 0), though the AI's comment incorrectly labels it as "bottom."

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: (1) Trials with NaN go cue times are excluded. (2) Missing video tracking data (None ts or frameTimes) results in NaN velocity, which becomes the "not visible" class. (3) Sessions with fewer than 10 units are skipped. (4) Missing motion energy results in all-NaN, becoming "no video" class. However, the AI does NOT handle the case where behavioral trials continue past the recording end, resulting in all-zero neural data for late trials in two sessions.

ii.
```python
if ts is None or ft is None or np.isnan(gc):
    continue
if np.all(np.isnan(ft)):
    continue
...
valid_trials[np.isnan(goCue)] = False
```

iii. The AI acknowledged the all-zero trials in sessions 36 and 43 in CONVERSION_NOTES.md but did not fix them, calling them "data artifacts." The reference solution explicitly detects and drops these trials by finding the last trial with any spike data.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the MATLAB files is the dominant cost, as the AI noted. The spike binning is done per-neuron, per-trial in a nested Python loop, which is also very slow compared to the reference's vectorized `histogram2d` approach.

ii.
```python
for ni, neuron in enumerate(neurons):
    for t in range(n_trials):
        ...
        counts, _ = np.histogram(aligned, bins=edges)
```

iii. The AI documented timing information showing load time and binning time separately, identifying loading as the bottleneck. However, the nested loop for spike binning is also a significant bottleneck that wasn't identified.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop is a major candidate: it loops over every neuron and every trial individually, calling `np.histogram` once per neuron-trial combination. The reference solution vectorizes this with a single `np.histogram2d` call per neuron (all trials at once). The velocity computation and motion energy alignment also use per-trial loops with interpolation.

ii.
```python
for ni, neuron in enumerate(neurons):
    for t in range(n_trials):
        spike_mask = trial == trial_num
        aligned = trialtm[spike_mask] - gc
        counts, _ = np.histogram(aligned, bins=edges)
```

iii. The AI did not specifically identify the spike binning loop as a vectorization target. The reference uses `np.histogram2d` to bin all trials at once per neuron.

## 11-c. What processing does the code repeat multiple times?

i. The code loads each session file once. However, the per-trial fields (hit, miss, L, R, etc.) are read and flattened during loading, and some are read again during loading (e.g., goCue is read both in h5py loading and used later). The velocity computation function is called separately for tongue and paw, each recomputing frame times from scratch. The session loading functions (`load_session_h5` and `load_session_scipy`) both parse many fields that could overlap if a session was tried with both loaders.

ii. No explicit repeated computation, but the architecture doesn't cache intermediate results like frame times across features from the same camera.

iii. The AI did not specifically note repeated computations in CONVERSION_NOTES.md.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several unnecessary fields are loaded: (1) Lick times (`lickL`, `lickR`) are loaded but never used in the conversion. (2) `sample` and `delay` event times are loaded but not used. (3) The `moveThresh` from motion energy files is loaded but not used. (4) The `no` field (ignore trials) is loaded but outcome is derived from hit/miss only.

ii.
```python
session['lickL'] = []
session['lickR'] = []
...
session['sample'] = np.array(ev['sample']).flatten()
session['delay'] = np.array(ev['delay']).flatten()
...
thresh = me.get('moveThresh', 10)
```

iii. The AI loaded these fields during the exploration phase and kept them in the loading code even though they aren't used in the conversion pipeline.
