# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a `RAND_SESSIONS` list and loads only the `data/RandomizedDelay_Ephys_Behavior` folder. For each session key it constructs paired `data_structure_<session>.mat` and `motionEnergy_<session>.mat` paths and calls `process_session`. Mixed MAT formats are handled by trying `h5py` first and falling back to `scipy.io.loadmat`.

ii. ```python
BIN_SIZE = 0.075
RAND_SESSIONS = [
    'JEB11_2022-05-10', ... 'JEB24_2023-11-03'
]
...
base = Path('data/RandomizedDelay_Ephys_Behavior')
keys = RAND_SESSIONS[:2] if args.sample else RAND_SESSIONS
for key in keys:
    df = base / f'data_structure_{key}.mat'
    mf = base / f'motionEnergy_{key}.mat'
    neural, inp, out, bri, info = process_session(df, mf)
```

iii. The justification recorded in `CONVERSION_NOTES.md`, `README.md`, and trajectory step 30 is that the agent inferred the randomized-delay ALM subset from the paper/methods and decoder scripts. It also justified the mixed-format readers from Step 2 notes about HDF5 `data_structure_*.mat` files and older `motionEnergy_*.mat` files.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the session key prefix before the underscore, e.g. `JEB23_2023-10-10 -> JEB23`. The script keeps subjects in first-seen order and stores one `subject_idx` entry per session.

ii. ```python
subj = key.split('_')[0]
if subj not in subject_names:
    subject_names.append(subj)
...
subject_idx.append(subject_names.index(subj))
```

iii. There is no deeper recorded rationale beyond the filename convention. The trajectory repeatedly notes that session names encode subject and date.

## 1-c. How are the data split into sessions?

i. Each string in `RAND_SESSIONS` is treated as one session. Each session becomes one element of `neural`, `input`, `output`, and `brain_region_idx`.

ii. ```python
for key in keys:
    df = base / f'data_structure_{key}.mat'
    mf = base / f'motionEnergy_{key}.mat'
    neural, inp, out, bri, info = process_session(df, mf)
    all_neural.append(neural)
    all_input.append(inp)
    all_output.append(out)
```

iii. The recorded justification is again the agent's decision to focus on randomized-delay sessions only. `README.md` explicitly says the output contains 19 randomized-delay ALM sessions.

## 1-d. How are the data split into trials?

i. Trials are taken directly from `bp['Ntrials']`. Trial-indexed outputs are built by looping `ti in range(n_trials)`, and neural spikes are assigned to trials by matching `cluster['trial'] == trial_idx`.

ii. ```python
def bin_unit_spikes_for_trials(trialtm_list, trial_list, n_trials, time_edges):
    per_trial = [np.zeros((n_units, n_bins), dtype=np.float32) for _ in range(n_trials)]
    for ui, (ttm, tr) in enumerate(zip(trialtm_list, trial_list)):
        for trial_idx in range(1, n_trials + 1):
            mask = tr == trial_idx
            if np.any(mask):
                counts, _ = np.histogram(ttm[mask], bins=time_edges)
                per_trial[trial_idx - 1][ui] = counts.astype(np.float32)
```

iii. The trajectory notes that `obj.bp` contains per-trial arrays and `obj.clu` contains per-spike trial labels, so no trial-boundary reconstruction was attempted.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered with two rules: keep only trials with `early == 0` and `no == 0`, then discard any kept trial whose neural matrix is all zeros.

ii. ```python
valid = (bp['early'] == 0) & (bp['no'] == 0)
...
keep_idx = np.where(valid)[0]
keep_idx = [i for i in keep_idx if not np.all(neural[i] == 0)]
```

iii. The trajectory and `README.md` justify this as excluding early and no-response trials. There is no recorded justification for not removing photostimulation trials or for dropping ignore trials instead of keeping them as an outcome class.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from `obj.clu` spike times and spike trial labels, specifically `trialtm` and `trial`. The code does not use cluster `quality` or any probe-selection metadata.

ii. ```python
def extract_clu_h5(f):
    clu = deref(f, f['obj/clu'][()].flat[0])
    trialtm = [np.asarray(deref(f, r)[()]).ravel().astype(np.float32) for r in clu['trialtm'][()].flat]
    trial = [np.asarray(deref(f, r)[()]).ravel().astype(int) for r in clu['trial'][()].flat]
```

iii. Trajectory step 41 says the agent concluded `trialtm` and `trial` were enough to reconstruct per-trial spike trains. No explicit rationale is recorded for omitting quality labels or the go-cue subtraction from the neural derivation itself.

## 2-b. How is the `neural` data processed?

i. The script bins raw spike times into fixed-width histograms and stores the counts directly. It does not convert counts to Hz, smooth, normalize, baseline-subtract, or z-score.

ii. ```python
def bin_unit_spikes_for_trials(trialtm_list, trial_list, n_trials, time_edges):
    ...
    counts, _ = np.histogram(ttm[mask], bins=time_edges)
    per_trial[trial_idx - 1][ui] = counts.astype(np.float32)
```

iii. No scientific justification is recorded beyond getting decoder-compatible matrices. The trajectory shows the agent prioritized making the format pass verification before revisiting scientific fidelity.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is effectively no unit-level neural QC. The code keeps every cluster returned by `obj.clu`, applies no `quality` filter, and applies no minimum firing-rate threshold. The only later removal is dropping whole trials whose neural matrices are all zero.

ii. ```python
trialtm, trial, n_units = extract_clu_h5(f)
...
neural = bin_unit_spikes_for_trials(trialtm, trial, n_trials, time_edges)
...
keep_idx = [i for i in keep_idx if not np.all(neural[i] == 0)]
```

iii. No explicit unit-curation rationale is recorded. This appears to be an omission rather than a defended choice.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It is not aligned to go cue in the neural path. The code histograms `trialtm` directly into bins from `-2.5` to `2.5` without subtracting `bp['goCue']`.

ii. ```python
def bin_unit_spikes_for_trials(trialtm_list, trial_list, n_trials, time_edges):
    ...
    counts, _ = np.histogram(ttm[mask], bins=time_edges)
```

iii. The trajectory discusses go-cue alignment for video-derived outputs, but there is no recorded neural-specific justification. This looks like a missed step rather than an intentional alternative.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The script uses 75 ms bins over a nominal `-2.5` to `2.5` s window, producing 67 bins. There is no further rebinning after the initial histogram/binning step.

ii. ```python
BIN_SIZE = 0.075
T_START = -2.5
T_END = 2.5
...
time_edges = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE, dtype=np.float32)
time_centers = (time_edges[:-1] + time_edges[1:]) / 2
```

iii. The recorded justification is explicit: Step 1 notes say the reference decoding scripts use `rez.binSize = 75` ms, and trajectory steps 47 and 49 say the agent intentionally binned all streams onto the same 75 ms grid.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is derived from a synthetic bin-center vector defined by `T_START`, `T_END`, and `BIN_SIZE`; the code does not read a raw variable to construct the input per trial.

ii. ```python
time_edges = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE, dtype=np.float32)
time_centers = (time_edges[:-1] + time_edges[1:]) / 2
...
def make_time_input(n_trials, time_centers):
    return [time_centers[None, :].astype(np.float32).copy() for _ in range(n_trials)]
```

iii. The trajectory treats this as the common decoder time axis rather than a measurement read from file. No separate rationale beyond the chosen alignment grid is recorded.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code computes bin centers from the fixed session-independent time edges and repeats that 1D vector for every trial.

ii. ```python
time_centers = (time_edges[:-1] + time_edges[1:]) / 2
...
input_trials = make_time_input(n_trials, time_centers)
```

iii. The recorded rationale is that all modalities should share one common 75 ms grid. No additional processing or transformation is justified.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input uses the same `time_edges`/`time_centers` grid that the neural histograms use, so it is aligned to the neural matrices in the code's own coordinate system.

ii. ```python
counts, _ = np.histogram(ttm[mask], bins=time_edges)
...
time_centers = (time_edges[:-1] + time_edges[1:]) / 2
input_trials = make_time_input(n_trials, time_centers)
```

iii. Trajectory step 49 explicitly says the plan was to bin all streams onto the common grid relative to go cue. In the implementation, the shared grid is real, even though the neural stream never actually subtracts go cue.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The code derives lick direction only from the `bp['R']` and `bp['L']` side flags.

ii. ```python
lick_dir = np.where(bp['R'] > bp['L'], 1, 0).astype(np.int64)
```

iii. The trajectory notes that `L`, `R`, `hit`, `miss`, and `no` are available in `obj.bp`, but there is no explicit recorded argument for ignoring hit/miss when constructing lick direction.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The script encodes each trial as right if `R > L`, otherwise left, and then repeats that label across all time bins. It does not create a third `none` class.

ii. ```python
lick_dir = np.where(bp['R'] > bp['L'], 1, 0).astype(np.int64)
...
np.full((len(time_centers),), lick_dir[ti], dtype=np.int64)
```

iii. No justification is recorded beyond using the available trial flags. The README also advertises only `left/right` output values.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `bp['autowater']`.

ii. ```python
context = np.where(bp['autowater'] > 0, 0, 1).astype(np.int64)
```

iii. This is a straightforward field mapping. The trajectory explicitly identified `autowater` as the key context flag.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code maps `autowater > 0` to class `0` and all other trials to class `1`, then repeats that label across time bins.

ii. ```python
context = np.where(bp['autowater'] > 0, 0, 1).astype(np.int64)
...
np.full((len(time_centers),), context[ti], dtype=np.int64)
```

iii. The trajectory and notes justify this as the WC versus DR split directly available in Bpod metadata.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived only from the `bp['hit']` flag.

ii. ```python
outcome = np.where(bp['hit'] > 0, 1, 0).astype(np.int64)
```

iii. The trajectory notes that `hit`, `miss`, and `no` were available, but there is no recorded reason for collapsing everything non-hit into a single class.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code labels hit trials as `1` and every other trial as `0`, then repeats that per-trial value across all time bins. It does not represent a separate ignore class.

ii. ```python
outcome = np.where(bp['hit'] > 0, 1, 0).astype(np.int64)
...
np.full((len(time_centers),), outcome[ti], dtype=np.int64)
```

iii. No specific justification is recorded. The README similarly documents only `incorrect/correct` output values.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from `obj.traj` feature names, trajectory arrays `ts`, frame times `frameTimes`, and per-trial `bp['goCue']`. The code looks for tongue-like feature names in both camera streams and uses the first stream with finite data.

ii. ```python
tongue_idx0 = choose_feature(names0, ['tongue', 'left_tongue', 'right_tongue'])
tongue_idx1 = choose_feature(names1, ['top_tongue', 'bottom_tongue', 'topleft_tongue', 'bottomleft_tongue'])
...
ts, ft = getter(ti)
spd, tmid = speed_from_ts(ts, ft)
candidate = nanbin_mean(tmid - go, spd[idx], time_edges)
```

iii. Trajectory step 49 gives the clearest justification: stream 0 contains tongue-like features, stream 1 contains tongue and paw features, and the agent decided to choose a preferred tongue feature from those available and bin it relative to go cue.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The code computes frame-to-frame speed from x/y coordinates with simple finite differences, converts to midpoint timestamps, bins by within-bin mean, and takes the first tongue view that has any finite values. It does not use likelihood values, smoothing, per-run gap handling, or two-view averaging.

ii. ```python
def speed_from_ts(ts, frame_times):
    xy = ts[:, :2, :].astype(np.float32)
    dt = np.diff(frame_times).astype(np.float32)
    dt[dt <= 0] = np.nan
    dxy = np.diff(xy, axis=2)
    spd = np.sqrt(np.nansum(dxy ** 2, axis=1)) / dt[None, :]
    tmid = (frame_times[:-1] + frame_times[1:]) / 2
    return spd, tmid
...
for idx, getter in [(tongue_idx1, get_stream1), (tongue_idx0, get_stream0)]:
    ...
    if np.any(np.isfinite(candidate)):
        tongue_bin = candidate
        break
```

iii. Trajectory step 49 explicitly says the plan was to compute speed from x/y coordinates and `frameTimes`, then threshold per session. No richer justification is recorded.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. All finite tongue bins from the session are pooled, the session median is used as the threshold, and bins are encoded as `0` below threshold and `1` at or above threshold. Missing bins remain `0`; there is no `2 = not visible` class.

ii. ```python
def threshold_session_bins(arr_list):
    finite_chunks = [a[np.isfinite(a)] for a in arr_list if np.any(np.isfinite(a))]
    allv = np.concatenate(finite_chunks) if finite_chunks else np.array([])
    thr = np.nanmedian(allv) if allv.size else np.nan
    out = []
    for a in arr_list:
        b = np.zeros_like(a, dtype=np.int64)
        if np.isfinite(thr):
            valid = np.isfinite(a)
            b[valid] = (a[valid] >= thr).astype(np.int64)
```

iii. Trajectory steps 47 and 49 justify the session-median threshold. No recorded rationale explains why missing bins were folded into class `0` instead of the requested `2`.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The code aligns tongue midpoint times by subtracting the per-trial `goCue`, then bins them onto the same `time_edges` grid as the neural data. It does not apply any video-to-behavior clock offset.

ii. ```python
go = bp['goCue'][ti]
...
spd, tmid = speed_from_ts(ts, ft)
candidate = nanbin_mean(tmid - go, spd[idx], time_edges)
```

iii. Trajectory step 49 explicitly says the agent intended to align `frameTimes` to go cue using per-trial `bp.ev.goCue`. There is no recorded discussion of the reference `bitStart`/`sglx` offset correction.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from `obj.traj` stream 1 feature names and trajectory arrays, using `top_paw` if available and otherwise `bottom_paw`, together with `frameTimes` and per-trial `goCue`.

ii. ```python
paw_idx1 = choose_feature(names1, ['top_paw', 'bottom_paw'])
...
ts, ft = get_stream1(ti)
spd, tmid = speed_from_ts(ts, ft)
paw_bin = nanbin_mean(tmid - go, spd[paw_idx1], time_edges)
```

iii. Trajectory step 49 records this decision directly: pick a paw feature, preferably `top_paw`, and compute speed from x/y coordinates and `frameTimes`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Paw velocity uses the same finite-difference speed calculation as tongue velocity, then per-bin averaging. There is no smoothing, no likelihood masking, and no special handling of tracking gaps.

ii. ```python
def speed_from_ts(ts, frame_times):
    xy = ts[:, :2, :].astype(np.float32)
    dt = np.diff(frame_times).astype(np.float32)
    dt[dt <= 0] = np.nan
    dxy = np.diff(xy, axis=2)
    spd = np.sqrt(np.nansum(dxy ** 2, axis=1)) / dt[None, :]
```

iii. The recorded rationale is the same as for tongue velocity: use the available tracked coordinates and put everything on the 75 ms decoder grid.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity is thresholded with the same session-wide median rule as tongue velocity, producing only classes `0` and `1`. Missing bins default to `0`.

ii. ```python
paw_bin, paw_thr = threshold_session_bins(paw_vals)
```

iii. Trajectory step 49 explicitly says the agent intended session-median thresholds for the kinematic outputs. No separate justification is recorded for omitting the requested `2 = not visible` class.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw midpoint times are aligned by subtracting the per-trial `goCue` and then binning onto the same grid as neural data. No camera-to-behavior offset is applied.

ii. ```python
go = bp['goCue'][ti]
...
paw_bin = nanbin_mean(tmid - go, spd[paw_idx1], time_edges)
```

iii. The trajectory records the go-cue subtraction plan, but not the reference offset-correction step.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the standalone `motionEnergy_<session>.mat` file and paired with frame times from trajectory stream 0 plus per-trial `goCue`.

ii. ```python
def load_motion_energy(path):
    me = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)['me']
    ...
    out = [normalize_motion_elem(x) for x in arr.flat]
    return out, move_thresh
...
me = np.asarray(motion_data[ti]).ravel().astype(np.float32)
_, ft0 = get_stream0(ti)
```

iii. Step 2 notes explicitly identify `me.data` as the per-trial motion-energy source and note the old-style MAT wrapper structure. Trajectory step 49 says those traces were already available and only needed binning.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The code averages the per-frame motion-energy trace into the session's fixed time bins. It does not smooth, differentiate, or otherwise transform the trace before thresholding.

ii. ```python
if ti < len(motion_data):
    me = np.asarray(motion_data[ti]).ravel().astype(np.float32)
    _, ft0 = get_stream0(ti)
    me_bin = nanbin_mean(ft0 - go, me, time_edges) if len(ft0) == len(me) else np.full((len(time_edges)-1,), np.nan, dtype=np.float32)
```

iii. The trajectory justification is minimal but explicit: the motion-energy traces already exist per trial, so the agent decided to bin them onto the common decoder grid.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is thresholded with the same session-wide median rule as the other movement outputs, producing only classes `0` and `1`. Missing or unusable bins become `0`.

ii. ```python
me_bin, me_thr = threshold_session_bins(me_vals)
```

iii. The recorded rationale is again the session-median thresholding plan from trajectory step 49. There is no justification for omitting the required `2 = no video` class.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion-energy frame times from stream 0 are aligned by subtracting the per-trial `goCue`, then binned onto the same fixed grid as the neural data. No video-clock offset is applied.

ii. ```python
_, ft0 = get_stream0(ti)
me_bin = nanbin_mean(ft0 - go, me, time_edges)
```

iii. The trajectory justifies only the go-cue subtraction and common-grid binning. There is no recorded consideration of the reference's `bitStart`-based offset.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The code explicitly handles mixed MAT-file formats and nested motion-energy wrappers. Missing or malformed motion-energy trials are converted to all-NaN vectors, and length mismatches between motion energy and frame times also become all-NaN. More generally, invalid or missing kinematic bins are left as NaN until thresholding, where they collapse to class `0` rather than a dedicated missing-data class.

ii. ```python
def normalize_motion_elem(x):
    while True:
        if hasattr(x, '_fieldnames'):
            ...
        if isinstance(x, np.ndarray) and x.dtype == object:
            if x.size == 0:
                return np.array([], dtype=np.float32)
...
me_bin = nanbin_mean(ft0 - go, me, time_edges) if len(ft0) == len(me) else np.full((len(time_edges)-1,), np.nan, dtype=np.float32)
...
b = np.zeros_like(a, dtype=np.int64)
if np.isfinite(thr):
    valid = np.isfinite(a)
    b[valid] = (a[valid] >= thr).astype(np.int64)
```

iii. The notes justify the mixed-format handling from data exploration. There is no explicit recorded rationale for collapsing missing behavioral/video bins into the low class.

## 11-a. What are the most time-consuming steps of the code?

i. By structure, the most time-consuming steps are likely the nested per-unit/per-trial spike histogram loops, the per-trial trajectory extraction and speed computation, and then file loading. The code is dominated by Python loops rather than by a single vectorized counting step.

ii. ```python
for ui, (ttm, tr) in enumerate(zip(trialtm_list, trial_list)):
    for trial_idx in range(1, n_trials + 1):
        mask = tr == trial_idx
        if np.any(mask):
            counts, _ = np.histogram(ttm[mask], bins=time_edges)
```

iii. The AI did not record a runtime analysis. This answer is inferred from the implementation it wrote.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization targets are the nested neural loop in `bin_unit_spikes_for_trials`, the per-bin loop in `nanbin_mean`, and the repeated per-trial behavior loop in `extract_binned_behavior_generic`.

ii. ```python
for bi in range(len(out)):
    m = idx == bi
    if np.any(m):
        vv = values[m]
        if np.any(np.isfinite(vv)):
            out[bi] = np.nanmean(vv)
```

iii. No explicit efficiency rationale is recorded. The trajectory shows the agent was optimizing primarily for getting the converter to work and pass format verification.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats some work. It recomputes `speed_from_ts` separately for tongue and paw even when both come from stream 1 on the same trial, rebuilds `time_edges` inside every `process_session` call, and rescans every bin in `nanbin_mean` for each stream and trial.

ii. ```python
for ti in range(n_trials):
    ...
    for idx, getter in [(tongue_idx1, get_stream1), (tongue_idx0, get_stream0)]:
        ts, ft = getter(ti)
        spd, tmid = speed_from_ts(ts, ft)
    if paw_idx1 is not None:
        ts, ft = get_stream1(ti)
        spd, tmid = speed_from_ts(ts, ft)
```

iii. No explicit rationale is recorded for these repetitions. They appear to be implementation shortcuts.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads more fields than it ultimately needs, including unused Bpod fields like `bitRand`, `sample`, `delay`, and `reward`. It also reads full trajectory tensors and feature-name lists before discarding all non-selected features, and stores `moveThresh` only in metadata.

ii. ```python
for k in ['L', 'R', 'autowater', 'bitRand', 'early', 'hit', 'miss', 'no']:
    out[k] = np.asarray(bp[k][()]).astype(np.float32).ravel()
...
for k in ['bitStart', 'sample', 'delay', 'goCue', 'reward']:
    out[k] = np.asarray(ev[k][()]).astype(np.float32).ravel()
```

iii. No explicit justification is recorded. This looks like broad extraction for convenience rather than a deliberate downstream need.
