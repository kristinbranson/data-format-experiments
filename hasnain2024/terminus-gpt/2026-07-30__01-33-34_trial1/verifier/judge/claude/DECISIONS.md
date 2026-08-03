# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI only loads sessions from the `RandomizedDelay_Ephys_Behavior` directory, processing 19 randomized-delay sessions. It hardcodes a list `RAND_SESSIONS` of 19 session names. Each session is loaded from paired files: `data_structure_<subject>_<date>.mat` and `motionEnergy_<subject>_<date>.mat`. The data_structure files are loaded using `h5py` (for v7.3 HDF5) with fallback to `scipy.io.loadmat` (for older MAT). Motion energy is always loaded via `scipy.io.loadmat`.

ii.
```python
RAND_SESSIONS = [
    'JEB11_2022-05-10','JEB11_2022-05-11',
    'JEB12_2022-05-12','JEB12_2022-05-13',
    'JEB23_2023-10-10','JEB23_2023-10-11','JEB23_2023-10-12','JEB23_2023-10-13','JEB23_2023-10-18','JEB23_2023-10-19','JEB23_2023-10-21',
    'JEB24_2023-10-23','JEB24_2023-10-24','JEB24_2023-10-25','JEB24_2023-10-26','JEB24_2023-10-27','JEB24_2023-10-31','JEB24_2023-11-02','JEB24_2023-11-03'
]
...
base = Path('data/RandomizedDelay_Ephys_Behavior')
keys = RAND_SESSIONS[:2] if args.sample else RAND_SESSIONS
```

iii. The AI's CONVERSION_NOTES.md notes that the paper reports "19 sessions using four mice" for the randomized delay task, and the AI chose to convert only that subset. However, the reference code's loading scripts (`load<ANM>_ALMVideo.m`) enumerate sessions from both `Ephys_Behavior` (25 fixed-delay sessions) and `RandomizedDelay_Ephys_Behavior` (19 randomized-delay sessions), totaling 44 sessions from 14 subjects.

## 1-b. How are the data split into subjects?

i. The subject is extracted from the session name by splitting on underscore: `key.split('_')[0]`. Unique subjects are accumulated in order of appearance. Only 4 subjects are present (JEB11, JEB12, JEB23, JEB24) because only randomized-delay sessions are loaded.

ii.
```python
subj = key.split('_')[0]
if subj not in subject_names:
    subject_names.append(subj)
...
subject_idx.append(subject_names.index(subj))
```

iii. The approach of extracting the subject from the filename is correct, matching the reference approach.

## 1-c. How are the data split into sessions?

i. Each entry in `RAND_SESSIONS` is one session. Only the `RandomizedDelay_Ephys_Behavior` folder is searched. The 25 fixed-delay sessions from `Ephys_Behavior` are entirely omitted.

ii.
```python
base = Path('data/RandomizedDelay_Ephys_Behavior')
for key in keys:
    df = base / f'data_structure_{key}.mat'
    mf = base / f'motionEnergy_{key}.mat'
    neural, inp, out, bri, info = process_session(df, mf)
```

iii. The AI identified both directories in Step 2 of CONVERSION_NOTES.md but chose to only process the randomized-delay subset.

## 1-d. How are the data split into trials?

i. Trials are determined by `bp['Ntrials']`, which gives the number of trials in the session. Each trial index from 0 to Ntrials-1 is processed. Neural spike data is binned per trial using trial indices from cluster data.

ii.
```python
n_trials = bp['Ntrials']
...
for trial_idx in range(1, n_trials + 1):
    mask = tr == trial_idx
    if np.any(mask):
        counts, _ = np.histogram(ttm[mask], bins=time_edges)
        per_trial[trial_idx - 1][ui] = counts.astype(np.float32)
```

iii. The trial splitting approach is correct in principle, using `Ntrials` to define the trial count, matching the reference.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials by `(bp['early'] == 0) & (bp['no'] == 0)`, removing early-lick trials and "no-response" (ignore) trials. Additionally, trials where all neural activity is zero are removed. Photostimulation trials (`bp.stim.enable`) are NOT filtered.

ii.
```python
valid = (bp['early'] == 0) & (bp['no'] == 0)
...
keep_idx = np.where(valid)[0]
keep_idx = [i for i in keep_idx if not np.all(neural[i] == 0)]
```

iii. The CONVERSION_NOTES.md mentions "Early lick and ignore trials are omitted from analyses" based on the paper. However, the reference filters early-lick and photostimulation trials (`stim.enable`), NOT ignore/no-response trials. The AI removes ignore trials but keeps photostim trials, which is the wrong filtering choice.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `obj.clu` cluster data, specifically the `trialtm` (spike times relative to trial start) and `trial` (trial assignment) fields. For HDF5 files, these are extracted via reference dereferencing from `obj/clu`. For older MAT files, they are extracted from `obj.clu` directly.

ii.
```python
def extract_clu_h5(f):
    clu = deref(f, f['obj/clu'][()].flat[0])
    trialtm = [np.asarray(deref(f, r)[()]).ravel().astype(np.float32) for r in clu['trialtm'][()].flat]
    trial = [np.asarray(deref(f, r)[()]).ravel().astype(int) for r in clu['trial'][()].flat]
    n_units = len(trialtm)
    return trialtm, trial, n_units
```

iii. The AI correctly identifies the spike data source variables, matching the reference.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 75ms time bins from -2.5 to 2.5s around the go cue. The spikes are aligned by subtracting the go cue time. The resulting values are raw spike counts (not converted to firing rates). No smoothing is applied.

ii.
```python
BIN_SIZE = 0.075
...
def bin_unit_spikes_for_trials(trialtm_list, trial_list, n_trials, time_edges):
    ...
    for ui, (ttm, tr) in enumerate(zip(trialtm_list, trial_list)):
        for trial_idx in range(1, n_trials + 1):
            mask = tr == trial_idx
            if np.any(mask):
                counts, _ = np.histogram(ttm[mask], bins=time_edges)
                per_trial[trial_idx - 1][ui] = counts.astype(np.float32)
```

iii. The reference code uses 5ms bins (matching `params.dt = 1/200`), converts counts to Hz by dividing by the bin width, and applies Gaussian smoothing with 14ms sigma. The AI's 75ms bins come from the decoding script's `rez.binSize = 75`, which is the analysis bin for decoding, not the native temporal resolution of the data.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No quality filtering is applied to neural units. All clusters from `obj.clu` are included regardless of their quality labels or firing rates. The AI does not read or check the `quality` field of clusters, nor does it apply a minimum firing rate threshold.

ii.
```python
def extract_clu_h5(f):
    clu = deref(f, f['obj/clu'][()].flat[0])
    trialtm = [np.asarray(deref(f, r)[()]).ravel().astype(np.float32) for r in clu['trialtm'][()].flat]
    trial = [np.asarray(deref(f, r)[()]).ravel().astype(int) for r in clu['trial'][()].flat]
    n_units = len(trialtm)
    return trialtm, trial, n_units
```

iii. The reference drops clusters with quality labels in `{garbage, gabrga, noisy, real?, poor}` and then drops units with mean firing rate <= 1 Hz. The AI includes all clusters, which would include noisy and garbage units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue by subtracting `bp['goCue'][ti]` (the go cue time for trial `ti`) from each spike's `trialtm`. This is done when binning the behavioral features and implicitly when binning neural spikes (the spikes are binned using time edges relative to the go cue).

ii.
```python
go = bp['goCue'][ti]
...
candidate = nanbin_mean(tmid - go, spd[idx], time_edges)
```
For neural data, spikes are binned using `time_edges` that span from T_START to T_END, but the go cue subtraction for neural spikes happens implicitly in the histogram call since `trialtm` stores times relative to trial start. The code does NOT subtract the go cue from spike times before binning.

Actually, looking more carefully:
```python
counts, _ = np.histogram(ttm[mask], bins=time_edges)
```
where `ttm` is raw `trialtm` without go cue subtraction. The `time_edges` span -2.5 to 2.5. Since `trialtm` is relative to trial start (not go cue), the neural alignment is INCORRECT - spikes are not aligned to the go cue.

iii. The AI subtracts the go cue for behavioral features but NOT for neural spike times. This is a critical alignment bug.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 75ms bins, producing 67 time bins from -2.5 to 2.5s. No rebinning is applied - spikes are directly counted into 75ms bins.

ii.
```python
BIN_SIZE = 0.075
T_START = -2.5
T_END = 2.5
...
time_edges = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE, dtype=np.float32)
```

iii. The AI chose 75ms based on the decoding scripts' `rez.binSize = 75`. However, the reference uses 5ms bins matching `params.dt = 1/200` from the reference code's `getDefaultParams.m`. The 75ms is the decoding analysis bin, not the data storage resolution.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is constructed from the time bin centers, computed from `time_edges`. It represents time from go cue onset.

ii.
```python
time_centers = (time_edges[:-1] + time_edges[1:]) / 2
...
def make_time_input(n_trials, time_centers):
    return [time_centers[None, :].astype(np.float32).copy() for _ in range(n_trials)]
```

iii. This matches the reference approach conceptually.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time centers are computed as the midpoints of the time bin edges. No other processing is involved.

ii.
```python
time_centers = (time_edges[:-1] + time_edges[1:]) / 2
```

iii. Straightforward computation, same approach as reference.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time bin edges that define the input are the same edges used for binning neural spikes and behavioral features. However, since the neural spikes are not go-cue-aligned (see 2-d), the alignment between input and neural data is broken.

ii. The same `time_edges` array is used for both:
```python
time_edges = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE, dtype=np.float32)
```

iii. The intent is correct (same grid for all), but the neural alignment bug means the neural data is not actually aligned to the go cue.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `bp['R']` and `bp['L']`, the instructed side fields.

ii.
```python
lick_dir = np.where(bp['R'] > bp['L'], 1, 0).astype(np.int64)
```

iii. The AI uses only the instructed side to determine lick direction, rather than the combination of instructed side and outcome (hit/miss) that the reference uses to infer actual lick direction.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A simple comparison: if R > L then right (1), otherwise left (0). This gives the instructed direction, not the actual lick direction. The AI does not have a "no lick" class for ignore trials.

ii.
```python
lick_dir = np.where(bp['R'] > bp['L'], 1, 0).astype(np.int64)
```

iii. The reference derives actual lick direction: hit + right = licked right, hit + left = licked left, miss + right = licked left, miss + left = licked right, ignore = no lick. The AI's approach conflates instructed direction with actual direction and has only 2 classes instead of 3. Since the AI also drops ignore trials (via `bp['no'] == 0` filter), the missing "no lick" class is partially mitigated but miss trials still get the wrong direction.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `bp['autowater']`.

ii.
```python
context = np.where(bp['autowater'] > 0, 0, 1).astype(np.int64)
```

iii. Correctly identifies the source variable, matching the reference.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater > 0 maps to WC (0), otherwise DR (1).

ii.
```python
context = np.where(bp['autowater'] > 0, 0, 1).astype(np.int64)
```

iii. This matches the reference's mapping: WC = 0, DR = 1.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived solely from `bp['hit']`.

ii.
```python
outcome = np.where(bp['hit'] > 0, 1, 0).astype(np.int64)
```

iii. The reference uses both `bp.hit` and `bp.miss` to create three classes (correct, incorrect, ignore). The AI only uses `hit` for a binary classification.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Simple binary: hit > 0 maps to correct (1), everything else to incorrect (0). No "ignore" class.

ii.
```python
outcome = np.where(bp['hit'] > 0, 1, 0).astype(np.int64)
```

iii. The reference has three classes: hit = correct (1), miss = incorrect (0), everything else = ignore (2). The AI collapses miss and ignore into one "incorrect" class. However, since the AI also filters out ignore trials (`bp['no'] == 0`), the remaining incorrect trials are actually miss trials. But the output_values only lists `['incorrect', 'correct']` with 2 classes, not 3.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from `obj.traj`, specifically the tracked feature `tongue` (from the side camera, stream 0) or `top_tongue` (from the bottom camera, stream 1). The `ts` (tracking data: x, y, likelihood per feature per frame) and `frameTimes` are used.

ii.
```python
tongue_idx0 = choose_feature(names0, ['tongue', 'left_tongue', 'right_tongue'])
tongue_idx1 = choose_feature(names1, ['top_tongue', 'bottom_tongue', 'topleft_tongue', 'bottomleft_tongue'])
...
for idx, getter in [(tongue_idx1, get_stream1), (tongue_idx0, get_stream0)]:
    if idx is None:
        continue
    ts, ft = getter(ti)
    spd, tmid = speed_from_ts(ts, ft)
    candidate = nanbin_mean(tmid - go, spd[idx], time_edges)
    if np.any(np.isfinite(candidate)):
        tongue_bin = candidate
        break
```

iii. The AI identifies the correct source variables from `traj` for the tongue.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Speed is computed as the magnitude of frame-to-frame position differences divided by frame-to-frame time differences. No Gaussian smoothing is applied to positions before differentiation. No likelihood filtering is applied (all frames are used regardless of tracking confidence). Only ONE camera view is used per trial (the first one that produces finite values), rather than combining both views.

ii.
```python
def speed_from_ts(ts, frame_times):
    xy = ts[:, :2, :].astype(np.float32)
    dt = np.diff(frame_times).astype(np.float32)
    dt[dt <= 0] = np.nan
    dxy = np.diff(xy, axis=2)
    spd = np.sqrt(np.nansum(dxy ** 2, axis=1)) / dt[None, :]
    tmid = (frame_times[:-1] + frame_times[1:]) / 2
    return spd, tmid
```

iii. The reference applies likelihood > 0.9 filtering, 5ms Gaussian smoothing of x/y within valid runs, uses both camera views normalized by their 90th percentile, and averages them. The AI's approach is much simpler and missing several processing steps.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The session median of all finite tongue velocity values is used as the threshold. Values >= threshold are "high" (1), values below are "low" (0). The conversion output shows `tongue_threshold: 0.0` for all sessions, meaning the threshold is 0, which makes almost everything "high".

ii.
```python
def threshold_session_bins(arr_list):
    finite_chunks = [a[np.isfinite(a)] for a in arr_list if np.any(np.isfinite(a))]
    allv = np.concatenate(finite_chunks) if finite_chunks else np.array([])
    thr = np.nanmedian(allv) if allv.size else np.nan
    ...
    b[valid] = (a[valid] >= thr).astype(np.int64)
```

iii. The tongue_threshold being 0.0 across all sessions indicates a bug: the median of tongue velocity values is 0, meaning most values are 0 (tongue not visible). The reference handles this by having a "not visible" class for NaN bins and computing the threshold over actual velocity values. The AI does not distinguish between "not visible" (no tongue tracked) and "below threshold", collapsing them into one class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are aligned by subtracting the go cue time: `tmid - go`. The go-cue-aligned times are then binned into the same time edges as the neural data. However, the video-to-behavior clock offset is NOT computed or applied.

ii.
```python
go = bp['goCue'][ti]
...
candidate = nanbin_mean(tmid - go, spd[idx], time_edges)
```

iii. The reference computes a video offset using bitcode signals (`findVideoOffset.m` equivalent) to correct for the clock difference between the video and behavior systems. The AI subtracts only the go cue without this offset correction, leading to temporal misalignment.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from `obj.traj` stream 1 (bottom camera), using the `top_paw` or `bottom_paw` feature (whichever is found first).

ii.
```python
paw_idx1 = choose_feature(names1, ['top_paw', 'bottom_paw'])
...
ts, ft = get_stream1(ti)
spd, tmid = speed_from_ts(ts, ft)
paw_bin = nanbin_mean(tmid - go, spd[paw_idx1], time_edges)
```

iii. The reference also uses `top_paw` from the bottom camera, matching the AI's first choice.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same as tongue: raw frame-to-frame speed without smoothing or likelihood filtering, binned into 75ms bins.

ii.
```python
spd, tmid = speed_from_ts(ts, ft)
paw_bin = nanbin_mean(tmid - go, spd[paw_idx1], time_edges)
```

iii. The reference applies likelihood filtering, Gaussian smoothing, and bins into 5ms bins. The AI omits all of these processing steps.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Session median threshold, same as tongue. Values >= threshold are "high" (1), below are "low" (0). No "not visible" class.

ii.
```python
paw_bin, paw_thr = threshold_session_bins(paw_vals)
```

iii. The reference uses session 50th percentile with a "not visible" class for untracked bins. The AI uses median (equivalent to 50th percentile) but lacks the "not visible" class.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: frame times minus go cue, no video offset correction.

ii.
```python
paw_bin = nanbin_mean(tmid - go, spd[paw_idx1], time_edges)
```

iii. Missing video offset correction, same issue as tongue alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from `motionEnergy_<subject>_<date>.mat`, reading the `me` struct and extracting the `data` field (with nested unwrapping for different formats).

ii.
```python
def load_motion_energy(path):
    me = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)['me']
    ...
    while hasattr(raw, '_fieldnames') and hasattr(raw, 'data'):
        raw = raw.data
    arr = np.asarray(raw, dtype=object)
    out = [normalize_motion_elem(x) for x in arr.flat]
    return out, move_thresh
```

iii. The source is correct, matching the reference.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy values are binned into time bins by aligning frame times to the go cue. No additional processing beyond binning.

ii.
```python
me = np.asarray(motion_data[ti]).ravel().astype(np.float32)
_, ft0 = get_stream0(ti)
me_bin = nanbin_mean(ft0 - go, me, time_edges) if len(ft0) == len(me) else np.full(...)
```

iii. The reference also just bins the pre-computed motion energy values, so the processing is similar in principle.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Session median threshold, same as tongue and paw. Values >= threshold are "high" (1), below are "low" (0).

ii.
```python
me_bin, me_thr = threshold_session_bins(me_vals)
```

iii. The reference uses session 50th percentile (equivalent to median). The AI's approach is functionally similar but lacks a "not visible" class for NaN bins.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Frame times from stream 0 (side camera) are used, with go cue subtraction. No video offset correction.

ii.
```python
_, ft0 = get_stream0(ti)
me_bin = nanbin_mean(ft0 - go, me, time_edges)
```

iii. Missing video offset correction, same as tongue and paw. The reference uses the side camera frame times with the video offset correction applied.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing data is handled by filling with NaN and then using `nanbin_mean` which ignores NaN values in bins. If motion energy length doesn't match frame times, the entire trial is set to NaN. The `normalize_motion_elem` function handles empty arrays. After thresholding, NaN bins become 0 (below threshold) since `np.isfinite(a)` is False for NaN and the default is 0.

ii.
```python
me_bin = nanbin_mean(ft0 - go, me, time_edges) if len(ft0) == len(me) else np.full((len(time_edges)-1,), np.nan, dtype=np.float32)
...
b = np.zeros_like(a, dtype=np.int64)
if np.isfinite(thr):
    valid = np.isfinite(a)
    b[valid] = (a[valid] >= thr).astype(np.int64)
```

iii. The reference uses a "not visible" class (value 2) for bins where the camera did not track the feature. The AI assigns NaN bins to class 0 ("low"), which conflates missing data with low velocity/energy.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the MAT files and the per-trial looping for behavioral feature extraction. The full conversion of 19 sessions produces output in a few minutes based on the conversion log.

ii. The main loop processes sessions sequentially:
```python
for key in keys:
    ...
    neural, inp, out, bri, info = process_session(df, mf)
```

iii. Similar to the reference, file I/O dominates.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop iterates over every unit and every trial individually, which is very inefficient. The reference uses a single `np.histogram2d` call per cluster to bin all trials at once. The behavioral feature extraction also loops per trial.

ii.
```python
for ui, (ttm, tr) in enumerate(zip(trialtm_list, trial_list)):
    for trial_idx in range(1, n_trials + 1):
        mask = tr == trial_idx
        if np.any(mask):
            counts, _ = np.histogram(ttm[mask], bins=time_edges)
            per_trial[trial_idx - 1][ui] = counts.astype(np.float32)
```

iii. The nested loop over units and trials is O(n_units * n_trials), while the reference's histogram2d approach is O(n_units) with vectorized trial handling.

## 11-c. What processing does the code repeat multiple times?

i. In `extract_binned_behavior_generic`, `get_stream1(ti)` is called twice for the same trial when both tongue and paw are extracted from stream 1: once in the tongue loop and once for paw extraction.

ii.
```python
for idx, getter in [(tongue_idx1, get_stream1), (tongue_idx0, get_stream0)]:
    ...
    ts, ft = getter(ti)
    spd, tmid = speed_from_ts(ts, ft)
    ...
if paw_idx1 is not None:
    ts, ft = get_stream1(ti)
    spd, tmid = speed_from_ts(ts, ft)
```

iii. The stream data and speed computation are redundantly recomputed for the same trial and camera.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes speed for ALL tracked features in `ts` (all features in the tracking array), but only uses one feature index per stream. The `speed_from_ts` function computes speed across all features and returns the full array, even though only one feature's speed is used.

ii.
```python
def speed_from_ts(ts, frame_times):
    xy = ts[:, :2, :].astype(np.float32)
    ...
    spd = np.sqrt(np.nansum(dxy ** 2, axis=1)) / dt[None, :]
    # spd has shape (n_features, n_frames-1), but only one feature index is used
```

iii. Computing speed for all features when only one is needed wastes computation.
