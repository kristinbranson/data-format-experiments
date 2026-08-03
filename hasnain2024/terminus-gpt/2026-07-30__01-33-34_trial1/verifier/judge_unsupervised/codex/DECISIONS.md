# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script loads only a hard-coded 19-session randomized-delay subset. For each session key, it opens a paired `data_structure_<session>.mat` file and `motionEnergy_<session>.mat` file from `data/RandomizedDelay_Ephys_Behavior`. Within each session it supports both HDF5/v7.3 MAT files and older MAT files, then builds neural/input/output trial lists session by session.

ii. ```python
RAND_SESSIONS = [
    'JEB11_2022-05-10', 'JEB11_2022-05-11',
    'JEB12_2022-05-12', 'JEB12_2022-05-13',
    'JEB23_2023-10-10', ..., 'JEB24_2023-11-03'
]

base = Path('data/RandomizedDelay_Ephys_Behavior')
for key in keys:
    df = base / f'data_structure_{key}.mat'
    mf = base / f'motionEnergy_{key}.mat'
    neural, inp, out, bri, info = process_session(df, mf)
```

```python
try:
    with h5py.File(data_path, 'r') as f:
        bp = extract_bp_h5(f)
        trialtm, trial, n_units = extract_clu_h5(f)
        names0, get0 = get_traj_stream_h5(f, 0)
        names1, get1 = get_traj_stream_h5(f, 1)
except OSError:
    obj = scipy.io.loadmat(data_path, squeeze_me=True, struct_as_record=False)['obj']
    bp = extract_bp_old(obj)
    trialtm, trial, n_units = extract_clu_old(obj)
    names0, get0 = get_traj_stream_old(obj, 0)
    names1, get1 = get_traj_stream_old(obj, 1)
```

iii. The notes say the agent resolved the paper-versus-raw-count mismatch by following the randomized-delay loader session list from the reference code: 19 sessions total across JEB11, JEB12, JEB23, and JEB24, rather than scanning all raw files.

## 1-b. How are the data split into subjects (mice)?

i. Subject identity is inferred from the session key prefix before the first underscore, e.g. `JEB23` from `JEB23_2023-10-10`. Sessions are mapped to integer subject indices in first-seen order.

ii. ```python
subj = key.split('_')[0]
if subj not in subject_names:
    subject_names.append(subj)
...
subject_idx.append(subject_names.index(subj))
```

iii. The notes justify using the randomized-delay ALM reference loaders, which enumerate four mice. The code then derives the mouse ID directly from each loader-style session name.

## 1-c. How are the data split into sessions?

i. Each string in `RAND_SESSIONS` defines one session. The main loop processes one paired `.mat` dataset per session and appends one element to `neural`, `input`, `output`, and `brain_region_idx`.

ii. ```python
for key in keys:
    df = base / f'data_structure_{key}.mat'
    mf = base / f'motionEnergy_{key}.mat'
    neural, inp, out, bri, info = process_session(df, mf)
    all_neural.append(neural)
    all_input.append(inp)
    all_output.append(out)
    brain_region_idx.append(bri)
```

iii. The notes explicitly say the agent chose the session split to match the reference randomized-delay loaders: 2 sessions for JEB11, 2 for JEB12, 7 for JEB23, and 8 for JEB24.

## 1-d. How are the data split into trials?

i. Trials are indexed by `bp['Ntrials']`. Neural spikes are split by per-unit trial labels in `obj.clu.*.trial`, and behavioral streams are iterated trial-by-trial from `0..Ntrials-1`. After construction, only kept trial indices are retained.

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

```python
n_trials = bp['Ntrials']
for ti in range(n_trials):
    ...
keep_idx = np.where(valid)[0]
```

iii. The agent’s notes map `obj.bp.Ntrials` and `obj.clu` trial fields to the trial structure, and they planned sanity checks around trial counts matching event-array lengths.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept if they are not early-lick trials and not `no` trials, then further filtered to remove trials whose neural matrix is entirely zero. No additional trial-balance or performance-based curation is enforced.

ii. ```python
valid = (bp['early'] == 0) & (bp['no'] == 0)
...
keep_idx = np.where(valid)[0]
keep_idx = [i for i in keep_idx if not np.all(neural[i] == 0)]
```

iii. The notes cite the paper/methods rule that early lick and ignore trials were omitted from analyses. The extra all-zero-neural filter was added later to silence verifier warnings rather than because of a documented reference rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural matrices are derived from the spike-time and spike-trial fields inside `obj.clu`: specifically per-unit `trialtm` and `trial`. Unit count is the number of `obj.clu` entries.

ii. ```python
def extract_clu_h5(f):
    clu = deref(f, f['obj/clu'][()].flat[0])
    trialtm = [np.asarray(deref(f, r)[()]).ravel().astype(np.float32) for r in clu['trialtm'][()].flat]
    trial = [np.asarray(deref(f, r)[()]).ravel().astype(int) for r in clu['trial'][()].flat]
    n_units = len(trialtm)
    return trialtm, trial, n_units
```

```python
def extract_clu_old(obj):
    clu_arr = np.asarray(obj.clu, dtype=object).ravel()
    trialtm = [np.asarray(unwrap_obj(c).trialtm).ravel().astype(np.float32) for c in clu_arr]
    trial = [np.asarray(unwrap_obj(c).trial).ravel().astype(int) for c in clu_arr]
```

iii. The notes identify `obj.clu` as the neural source and specifically call out `trialtm`/`trial` as the fields needed to reconstruct trialized spikes.

## 2-b. How is the `neural` data processed?

i. For each unit and each trial, spikes are histogrammed into fixed 75 ms bins spanning `[-2.5, 2.5]`. The result is a per-trial neuron-by-time spike-count matrix. There is no smoothing, firing-rate normalization, or z-scoring.

ii. ```python
BIN_SIZE = 0.075
T_START = -2.5
T_END = 2.5
...
counts, _ = np.histogram(ttm[mask], bins=time_edges)
per_trial[trial_idx - 1][ui] = counts.astype(np.float32)
```

iii. The notes justify the 75 ms bin size from the reference decoding scripts (`rez.binSize = 75` ms). Beyond that, the code makes a simplifying decision to use raw binned spike counts directly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is effectively not filtered at the neuron level. The script loads all units present in `obj.clu` and never uses raw quality labels, site/channel metadata, or firing-rate thresholds.

ii. ```python
trialtm, trial, n_units = extract_clu_h5(f)
...
brain_region_idx = np.zeros((n_units,), dtype=int)
```

```python
# No references to clu['quality'], firing rate, or >1 Hz filtering appear later.
```

iii. The notes themselves say the reference methods used unit curation, sessions with at least 10 units, and often a >1 Hz firing-rate threshold. The code never implemented those rules.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The code does not explicitly align neural spikes to go cue. It bins raw `trialtm` values against fixed edges and never subtracts per-trial `bp['goCue']`. This implicitly assumes `trialtm` is already go-cue-aligned.

ii. ```python
time_edges = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE, dtype=np.float32)
...
neural = bin_unit_spikes_for_trials(trialtm, trial, n_trials, time_edges)
```

```python
# In contrast, behavior is aligned with explicit subtraction:
candidate = nanbin_mean(tmid - go, spd[idx], time_edges)
```

iii. The notes repeatedly say go cue should be the alignment event, but the implementation never applies `goCue` to the neural stream. Raw event arrays show `goCue` varies by trial, so treating raw `trialtm` as already aligned is not justified by the files.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 75 ms bins (`0.075` s) over a 5 s window. The only temporal rebinning is the initial binning/averaging into those bins; there is no second rebinning stage.

ii. ```python
BIN_SIZE = 0.075
T_START = -2.5
T_END = 2.5
time_edges = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE, dtype=np.float32)
time_centers = (time_edges[:-1] + time_edges[1:]) / 2
```

iii. The notes explicitly cite the reference decoder bin size of 75 ms as the basis for this choice.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. In the final code it is not derived from any raw variable. It is generated from hard-coded bin edges and centers, not from `obj.bp.ev.goCue` or any other session field.

ii. ```python
def make_time_input(n_trials, time_centers):
    return [time_centers[None, :].astype(np.float32).copy() for _ in range(n_trials)]
```

```python
time_centers = (time_edges[:-1] + time_edges[1:]) / 2
input_trials = make_time_input(n_trials, time_centers)
```

iii. The notes describe time from go cue conceptually, but the implementation shortcut was to reuse the same fixed time vector for every trial.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The script computes bin centers from the fixed `[-2.5, 2.5]` window and copies that same 1D vector into every trial.

ii. ```python
time_centers = (time_edges[:-1] + time_edges[1:]) / 2
...
return [time_centers[None, :].astype(np.float32).copy() for _ in range(n_trials)]
```

iii. The justification is implicit: once everything is assumed to live on a common go-cue-centered grid, the same time vector can be reused for every trial.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is aligned only by assumption: the input uses the same 67-bin grid as the neural matrices, but the neural matrices are not explicitly shifted by `goCue`. So the input is go-cue-relative while the neural data remain in raw `trialtm` coordinates.

ii. ```python
input_trials = make_time_input(n_trials, time_centers)
neural = bin_unit_spikes_for_trials(trialtm, trial, n_trials, time_edges)
```

iii. The notes say go cue should align everything, but the actual code only shares bin count and keep indices across modalities.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the trial-level behavior flags `bp['L']` and `bp['R']`.

ii. ```python
lick_dir = np.where(bp['R'] > bp['L'], 1, 0).astype(np.int64)
```

iii. The notes map `obj.bp.L` and `obj.bp.R` directly to the lick-direction target.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The code compares the right and left trial flags and assigns `1` if `R > L`, otherwise `0`. The resulting scalar is then repeated across all time bins of the trial.

ii. ```python
lick_dir = np.where(bp['R'] > bp['L'], 1, 0).astype(np.int64)
...
np.full((len(time_centers),), lick_dir[ti], dtype=np.int64)
```

iii. The notes justify a per-trial categorical choice variable, and the code turns that into a time-varying constant to match the decoder format.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `bp['autowater']`, treating water-cued/autowater trials as WC and other trials as DR.

ii. ```python
context = np.where(bp['autowater'] > 0, 0, 1).astype(np.int64)
```

iii. The notes explicitly justify the WC/DR mapping from `autowater`, citing figure scripts that use `autowater` for WC and `~autowater` for DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code binarizes `autowater`: `autowater > 0` becomes WC=`0`, otherwise DR=`1`. The per-trial label is then repeated across the trial’s time bins.

ii. ```python
context = np.where(bp['autowater'] > 0, 0, 1).astype(np.int64)
...
np.full((len(time_centers),), context[ti], dtype=np.int64)
```

iii. The justification is the same `autowater`-to-context mapping documented in the notes.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the trial-level `bp['hit']` flag after early and `no` trials are filtered. Miss trials remain as `hit == 0`.

ii. ```python
outcome = np.where(bp['hit'] > 0, 1, 0).astype(np.int64)
```

iii. The notes say outcome should be correct versus incorrect on the valid trials after omission of early and ignore trials.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code converts `hit > 0` to correct=`1`, otherwise incorrect=`0`, then repeats the per-trial value across all time bins.

ii. ```python
outcome = np.where(bp['hit'] > 0, 1, 0).astype(np.int64)
...
np.full((len(time_centers),), outcome[ti], dtype=np.int64)
```

iii. The notes justify outcome as a per-trial categorical variable on filtered trials.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity comes from pose trajectories in `obj.traj`. The code prefers a tongue feature from stream 1 (`top_tongue`, `bottom_tongue`, etc.) and falls back to stream 0 (`tongue`, `left_tongue`, `right_tongue`).

ii. ```python
tongue_idx0 = choose_feature(names0, ['tongue', 'left_tongue', 'right_tongue'])
tongue_idx1 = choose_feature(names1, ['top_tongue', 'bottom_tongue', 'topleft_tongue', 'bottomleft_tongue'])
```

iii. The notes planned to map tongue-related `obj.traj` features to this output and cited the DLC decoding scripts as the reference behavior source.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The code computes 2D speed from successive x/y positions, ignoring the third trajectory channel, then averages speed within 75 ms bins relative to go cue. It takes the first tongue feature stream that yields any finite values.

ii. ```python
def speed_from_ts(ts, frame_times):
    xy = ts[:, :2, :].astype(np.float32)
    dt = np.diff(frame_times).astype(np.float32)
    dxy = np.diff(xy, axis=2)
    spd = np.sqrt(np.nansum(dxy ** 2, axis=1)) / dt[None, :]
    tmid = (frame_times[:-1] + frame_times[1:]) / 2
    return spd, tmid
```

```python
candidate = nanbin_mean(tmid - go, spd[idx], time_edges)
if np.any(np.isfinite(candidate)):
    tongue_bin = candidate
    break
```

iii. The justification is mostly implicit: use time-resolved DLC kinematics and bin them onto the decoder time axis. The notes wanted time-varying outputs when possible.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. All finite tongue-speed bins from the session are pooled, the session median is taken, and each finite bin is labeled low/high by whether it is below or at/above that median. Non-finite bins default to 0.

ii. ```python
def threshold_session_bins(arr_list):
    finite_chunks = [a[np.isfinite(a)] for a in arr_list if np.any(np.isfinite(a))]
    allv = np.concatenate(finite_chunks) if finite_chunks else np.array([])
    thr = np.nanmedian(allv) if allv.size else np.nan
    ...
    b[valid] = (a[valid] >= thr).astype(np.int64)
```

```python
tongue_bin, tongue_thr = threshold_session_bins(tongue_vals)
```

iii. The notes explicitly say session-level 50th-percentile thresholding should be used for the decoder outputs.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue velocity is aligned to go cue by subtracting `goCue` from frame-time midpoints before binning. Neural data are not shifted by `goCue`, so tongue and neural signals end up on inconsistent time references even though they share the same bin grid.

ii. ```python
go = bp['goCue'][ti]
...
candidate = nanbin_mean(tmid - go, spd[idx], time_edges)
```

iii. The notes intended all modalities to be go-cue aligned. In code, only the behavioral stream actually is.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from stream-1 paw features in `obj.traj`, with `top_paw` preferred over `bottom_paw` because `choose_feature` returns the first match.

ii. ```python
paw_idx1 = choose_feature(names1, ['top_paw', 'bottom_paw'])
```

iii. The notes planned to use paw-related `obj.traj` features as the raw source, again based on the DLC feature groups seen in the reference code.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. As with tongue, the code computes 2D speed from the chosen paw trajectory and averages it in go-cue-relative 75 ms bins.

ii. ```python
if paw_idx1 is not None:
    ts, ft = get_stream1(ti)
    spd, tmid = speed_from_ts(ts, ft)
    paw_bin = nanbin_mean(tmid - go, spd[paw_idx1], time_edges)
```

iii. The justification is implicit reuse of the same kinematic extraction pipeline for another DLC feature group.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity uses the same session-median binarization as tongue velocity.

ii. ```python
paw_bin, paw_thr = threshold_session_bins(paw_vals)
```

iii. The notes say continuous kinematic outputs should be discretized per session at the 50th percentile.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw velocity is aligned to go cue using `tmid - go`, but the neural data remain in raw `trialtm` coordinates, so paw and neural streams are not truly aligned to one another.

ii. ```python
paw_bin = nanbin_mean(tmid - go, spd[paw_idx1], time_edges)
```

iii. The notes intended common go-cue alignment, but the implementation only applies it to the behavioral side.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_<session>.mat` file, specifically the `me.data` entries loaded by `load_motion_energy`.

ii. ```python
def load_motion_energy(path):
    me = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)['me']
    ...
    out = [normalize_motion_elem(x) for x in arr.flat]
    return out, move_thresh
```

iii. The notes explicitly mapped `me.data` to the motion-energy decoder output.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The code normalizes several possible MAT representations into 1D float arrays, then bins each trial’s motion-energy trace using frame times borrowed from trajectory stream 0. If the motion trace length does not equal the frame-time length, the whole trial becomes NaN before thresholding.

ii. ```python
def normalize_motion_elem(x):
    while True:
        if hasattr(x, '_fieldnames'):
            if hasattr(x, 'data'):
                x = x.data
                continue
            x = getattr(x, x._fieldnames[0])
            continue
        if isinstance(x, np.ndarray) and x.dtype == object:
            ...
        return np.asarray(x).ravel().astype(np.float32)
```

```python
_, ft0 = get_stream0(ti)
me_bin = nanbin_mean(ft0 - go, me, time_edges) if len(ft0) == len(me) else np.full((len(time_edges)-1,), np.nan, dtype=np.float32)
```

iii. The notes justify robust mixed-format loading because the motion files had heterogeneous MATLAB representations. Using the video frame times is an implementation choice rather than a documented reference rule.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is thresholded with the same per-session median rule used for tongue and paw velocity.

ii. ```python
me_bin, me_thr = threshold_session_bins(me_vals)
```

iii. The notes and task specification both called for a per-session 50th-percentile threshold.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to go cue via `ft0 - go`, where `ft0` comes from trajectory stream 0 frame times. Neural data are not go-cue-aligned, so motion energy and neural activity are not actually aligned in the same time frame.

ii. ```python
me_bin = nanbin_mean(ft0 - go, me, time_edges) if len(ft0) == len(me) else ...
```

iii. The notes intended go-cue alignment across modalities, but the implementation only does that for behavior/motion.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles format heterogeneity explicitly: it falls back from HDF5 to `scipy.io.loadmat`, recursively unwraps nested MATLAB objects, and normalizes odd motion-energy element types. Missing or invalid kinematic/motion bins stay NaN until thresholding, where non-finite values silently become class `0`. Whole motion trials with frame-length mismatches also become all-NaN/all-0 after thresholding.

ii. ```python
try:
    with h5py.File(data_path, 'r') as f:
        ...
except OSError:
    obj = scipy.io.loadmat(data_path, squeeze_me=True, struct_as_record=False)['obj']
```

```python
me_bin = nanbin_mean(ft0 - go, me, time_edges) if len(ft0) == len(me) else np.full((len(time_edges)-1,), np.nan, dtype=np.float32)
...
b = np.zeros_like(a, dtype=np.int64)
if np.isfinite(thr):
    valid = np.isfinite(a)
    b[valid] = (a[valid] >= thr).astype(np.int64)
```

iii. The notes explicitly justify the mixed-format loaders and motion normalization because the raw MATLAB files varied. There is no recorded reference-based justification for imputing missing values to the low class.

## 11-a. What are the most time-consuming steps of the code?

i. The main costs are the per-session MAT-file parsing/dereferencing and the nested trial-by-unit spike binning. The behavior extraction also repeatedly loads trial trajectories and bins them one trial at a time.

ii. ```python
for ui, (ttm, tr) in enumerate(zip(trialtm_list, trial_list)):
    for trial_idx in range(1, n_trials + 1):
        ... np.histogram(...)
```

```python
for ti in range(n_trials):
    ...
    ts, ft = getter(ti)
    spd, tmid = speed_from_ts(ts, ft)
    candidate = nanbin_mean(...)
```

iii. This is a direct reading of the implemented control flow rather than a separately justified modeling choice.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The inner `for trial_idx` loop in `bin_unit_spikes_for_trials`, the per-bin loop inside `nanbin_mean`, and repeated per-trial recomputation of stream speeds could all be vectorized or cached.

ii. ```python
for ui, (ttm, tr) in enumerate(zip(trialtm_list, trial_list)):
    for trial_idx in range(1, n_trials + 1):
        mask = tr == trial_idx
        if np.any(mask):
            counts, _ = np.histogram(ttm[mask], bins=time_edges)
```

```python
for bi in range(len(out)):
    m = idx == bi
    if np.any(m):
        ...
```

iii. This is an efficiency judgment from the code path itself; the notes also mention speedups were still pending.

## 11-c. What processing does the code repeat multiple times?

i. It repeatedly dereferences the same trajectory streams within a trial, recomputes speed from those streams for tongue and paw separately, and constructs long time-constant categorical output rows for every trial.

ii. ```python
for idx, getter in [(tongue_idx1, get_stream1), (tongue_idx0, get_stream0)]:
    ts, ft = getter(ti)
    spd, tmid = speed_from_ts(ts, ft)
...
if paw_idx1 is not None:
    ts, ft = get_stream1(ti)
    spd, tmid = speed_from_ts(ts, ft)
```

```python
output_trials.append(np.vstack([
    np.full((len(time_centers),), lick_dir[ti], dtype=np.int64),
    np.full((len(time_centers),), context[ti], dtype=np.int64),
    np.full((len(time_centers),), outcome[ti], dtype=np.int64),
    ...
]))
```

iii. This is a factual description of the current code organization.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads and stores `moveThresh` only in metadata without using it in the conversion, builds neural/input/output arrays for invalid trials before filtering them out, and expands trial-constant labels into 67 time bins even though they carry no extra temporal information.

ii. ```python
motion_data, move_thresh = load_motion_energy(motion_path)
...
info = {
    ...
    'moveThresh': move_thresh,
}
```

```python
output_trials.append(np.vstack([
    np.full((len(time_centers),), lick_dir[ti], dtype=np.int64),
    np.full((len(time_centers),), context[ti], dtype=np.int64),
    np.full((len(time_centers),), outcome[ti], dtype=np.int64),
    ...
]))
...
keep_idx = np.where(valid)[0]
```

iii. This follows from the final code path; the notes do not provide a separate justification for keeping these redundant computations.
