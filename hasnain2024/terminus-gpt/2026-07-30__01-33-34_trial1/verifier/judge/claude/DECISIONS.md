# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data exclusively from the `RandomizedDelay_Ephys_Behavior` subdirectory. It processes a hardcoded list of 19 sessions (RAND_SESSIONS) from 4 mice. For each session, it loads two paired MAT files: `data_structure_<session>.mat` (neural/behavioral data) and `motionEnergy_<session>.mat` (motion energy data). It handles mixed MAT file formats (HDF5/v7.3 via h5py, older format via scipy.io.loadmat) with a try/except fallback.

ii.
```python
RAND_SESSIONS = [
    'JEB11_2022-05-10','JEB11_2022-05-11',
    'JEB12_2022-05-12','JEB12_2022-05-13',
    'JEB23_2023-10-10','JEB23_2023-10-11','JEB23_2023-10-12','JEB23_2023-10-13','JEB23_2023-10-18','JEB23_2023-10-19','JEB23_2023-10-21',
    'JEB24_2023-10-23','JEB24_2023-10-24','JEB24_2023-10-25','JEB24_2023-10-26','JEB24_2023-10-27','JEB24_2023-10-31','JEB24_2023-11-02','JEB24_2023-11-03'
]
# ...
base = Path('data/RandomizedDelay_Ephys_Behavior')
keys = RAND_SESSIONS[:2] if args.sample else RAND_SESSIONS
for key in keys:
    df = base / f'data_structure_{key}.mat'
    mf = base / f'motionEnergy_{key}.mat'
    neural, inp, out, bri, info = process_session(df, mf)
```

iii. The AI identified in CONVERSION_NOTES Step 4 that the paper reports "19 sessions using four mice" for the randomized delay task, and extracted the session list from the reference loading scripts. The AI chose to process only the randomized delay subset rather than all data directories (which also include `Ephys_Behavior` with 25 two-context sessions, and two behavior-only directories).

## 1-b. How are the data split into subjects (mice)?

i. Subject names are extracted from the session key by splitting on underscore and taking the first element. Four subjects are identified: JEB11, JEB12, JEB23, JEB24. A list of unique subject names is maintained, and each session is mapped to its subject index.

ii.
```python
for key in keys:
    subj = key.split('_')[0]
    if subj not in subject_names:
        subject_names.append(subj)
    # ...
    subject_idx.append(subject_names.index(subj))
```

iii. This matches the reference code's session loading scripts, which organize data by animal ID (e.g., `loadJEB11_ALMVideo`).

## 1-c. How are the data split into sessions?

i. Each MAT file pair corresponds to one session. The hardcoded list of 19 sessions defines which sessions to process. Sessions are processed sequentially in a loop, and each session's data is appended to the aggregate lists.

ii.
```python
keys = RAND_SESSIONS[:2] if args.sample else RAND_SESSIONS
for key in keys:
    # ... process each session
    all_neural.append(neural)
    all_input.append(inp)
    all_output.append(out)
```

iii. The session list was derived from the reference loading scripts and matches the paper's reported 19 sessions.

## 1-d. How are the data split into trials?

i. Within each session, the number of trials is determined from `bp['Ntrials']`. Trials are indexed sequentially from 1 to Ntrials. Neural spikes are assigned to trials via the `trial` array in `obj.clu`, and behavioral data is indexed per-trial from arrays in `obj.bp`.

ii.
```python
n_trials = bp['Ntrials']
# Neural: spikes assigned to trials via trial identity
for trial_idx in range(1, n_trials + 1):
    mask = tr == trial_idx
    # ...
# Behavioral: per-trial indexing
for ti in range(n_trials):
    go = bp['goCue'][ti]
```

iii. This follows the reference data structure where `obj.bp.Ntrials` defines the total trial count and `obj.clu.trial` maps spikes to trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by two criteria: (1) excluding early lick and ignore/no-go trials using `(bp['early'] == 0) & (bp['no'] == 0)`, and (2) removing trials where all neural activity across all neurons is zero (no spikes detected). The AI does NOT filter out stimulation trials (`stim.enable`) or restrict to hit-only trials.

ii.
```python
valid = (bp['early'] == 0) & (bp['no'] == 0)
# ...
keep_idx = np.where(valid)[0]
keep_idx = [i for i in keep_idx if not np.all(neural[i] == 0)]
neural = [neural[i] for i in keep_idx]
input_trials = [input_trials[i] for i in keep_idx]
output_trials = [output_trials[i] for i in keep_idx]
```

iii. The CONVERSION_NOTES document that "Early lick and ignore trials are omitted from analyses" per the paper. The instruction also specifies "Outcome (incorrect = 0, correct = 1)" as a decoder output, requiring both hit and miss trials. The reference code's default conditions (`'R&hit&~stim.enable&~autowater&~early'`) are more restrictive, filtering to hit-only and excluding stimulation trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `obj.clu.trialtm` (spike times per neuron) and `obj.clu.trial` (trial assignment per spike). These are extracted from either HDF5 references (for v7.3 MAT files) or MATLAB struct fields (for older MAT files).

ii.
```python
def extract_clu_h5(f):
    clu = deref(f, f['obj/clu'][()].flat[0])
    trialtm = [np.asarray(deref(f, r)[()]).ravel().astype(np.float32) for r in clu['trialtm'][()].flat]
    trial = [np.asarray(deref(f, r)[()]).ravel().astype(int) for r in clu['trial'][()].flat]
    n_units = len(trialtm)
    return trialtm, trial, n_units
```

iii. The AI identified that `obj.clu` contains cluster/neuron information from its data exploration in Step 2.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 75 ms time bins spanning -2.5 to +2.5 seconds using `np.histogram`. The output is raw spike **counts** per bin, NOT firing rates. There is no smoothing applied, no conversion to firing rates (dividing by bin size or trial count), and no baseline normalization.

ii.
```python
BIN_SIZE = 0.075
T_START = -2.5
T_END = 2.5

def bin_unit_spikes_for_trials(trialtm_list, trial_list, n_trials, time_edges):
    n_units = len(trialtm_list)
    n_bins = len(time_edges) - 1
    per_trial = [np.zeros((n_units, n_bins), dtype=np.float32) for _ in range(n_trials)]
    for ui, (ttm, tr) in enumerate(zip(trialtm_list, trial_list)):
        for trial_idx in range(1, n_trials + 1):
            mask = tr == trial_idx
            if np.any(mask):
                counts, _ = np.histogram(ttm[mask], bins=time_edges)
                per_trial[trial_idx - 1][ui] = counts.astype(np.float32)
    return per_trial
```

iii. The AI chose 75 ms bins based on the reference code's decoder binning (`rez.binSize = 75`). However, the reference code's `getSeq.m` uses 5 ms bins (`dt = 1/200`), converts counts to firing rates (`N./params.dt`), and applies causal Gaussian smoothing (`mySmooth(N./params.dt, params.smooth, params.bctype)` with smooth=15 samples ~75 ms kernel). The decoder scripts then downsample to 75 ms bins for decoding.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does NOT filter neurons by quality or firing rate. All neurons (units) in `obj.clu` are included regardless of their quality label or firing rate. The only filtering is at the trial level (removing all-zero trials).

ii.
```python
# No neuron filtering code exists. All units from clu are used:
n_units = len(trialtm)
brain_region_idx = np.zeros((n_units,), dtype=int)
```

iii. The CONVERSION_NOTES document that the reference code filters neurons by quality (excluding 'garbage', 'noisy', etc. via `findClusters.m`) and removes low firing rate neurons (< 0.5 Hz via `removeLowFRClusters.m`). The AI acknowledged these steps but did not implement them.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI bins the raw `trialtm` values directly into time edges from -2.5 to 2.5 seconds WITHOUT subtracting the per-trial go cue time. The reference code explicitly creates `trialtm_aligned = trialtm - goCue(trial)` via `alignSpikes.m`. In contrast, the behavioral data IS aligned to go cue by subtracting `bp['goCue'][ti]` from frame times.

ii.
```python
# Neural: trialtm used directly without goCue subtraction
counts, _ = np.histogram(ttm[mask], bins=time_edges)

# Behavioral: goCue IS subtracted
go = bp['goCue'][ti]
tongue_bin = nanbin_mean(tmid - go, spd[idx], time_edges)
```

iii. The reference code in `alignSpikes.m` explicitly aligns spikes: `obj.clu{prbnum}(clu).trialtm_aligned = obj.clu{prbnum}(clu).trialtm - event`. The AI does not perform this step. Whether this causes errors depends on whether `trialtm` in the data files is already relative to goCue (pre-aligned) or not.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 75 ms (`BIN_SIZE = 0.075`), producing 67 time bins from -2.5 to +2.5 seconds. No rebinning is applied -- spikes are directly binned at 75 ms resolution. The reference code bins at 5 ms (`dt = 1/200`), producing 1000 time bins, and the decoder scripts downsample to 75 ms for decoding.

ii.
```python
BIN_SIZE = 0.075
time_edges = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE, dtype=np.float32)
# Produces 67 bins
```

iii. The AI chose 75 ms directly based on the decoder bin size from the reference decoding scripts (`rez.binSize = 75` ms). The reference pipeline uses a finer 5 ms resolution with Gaussian smoothing.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not derived from any raw data variable. It is constructed as the centers of the time bins, representing time from go cue onset in seconds.

ii.
```python
def make_time_input(n_trials, time_centers):
    return [time_centers[None, :].astype(np.float32).copy() for _ in range(n_trials)]

# time_centers computed from time_edges:
time_edges = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE, dtype=np.float32)
time_centers = (time_edges[:-1] + time_edges[1:]) / 2
```

iii. This is a synthetic input that provides the decoder with temporal context (time relative to go cue). The same time vector is used for all trials.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time bin centers are computed as the midpoints of the 75 ms time edges. No further processing is applied. The input is shape (1, n_timepoints) for each trial.

ii.
```python
time_centers = (time_edges[:-1] + time_edges[1:]) / 2
# For each trial:
time_centers[None, :].astype(np.float32).copy()  # shape (1, 67)
```

iii. This is a straightforward construction matching the instruction requirement for "Time from go cue onset in seconds (continuous, time-varying)."

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input uses the same time bin centers as the neural data, so alignment is inherent -- both use the same 67-bin time grid from -2.5 to +2.5 seconds with 75 ms spacing.

ii.
```python
# Same time_edges used for both:
time_edges = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE, dtype=np.float32)
time_centers = (time_edges[:-1] + time_edges[1:]) / 2
neural = bin_unit_spikes_for_trials(trialtm, trial, n_trials, time_edges)
input_trials = make_time_input(n_trials, time_centers)
```

iii. Alignment is automatic since both use the same time grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `obj.bp.R` (right lick indicator) and `obj.bp.L` (left lick indicator).

ii.
```python
# Extracted from bp:
for k in ['L', 'R', 'autowater', 'bitRand', 'early', 'hit', 'miss', 'no']:
    out[k] = np.asarray(bp[k][()]).astype(np.float32).ravel()
# Used as:
lick_dir = np.where(bp['R'] > bp['L'], 1, 0).astype(np.int64)
```

iii. The reference code uses `R` and `L` as boolean condition flags in `findTrials.m` to separate right and left trials.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI compares the R and L values: if R > L, lick direction is 1 (right); otherwise 0 (left). This is a per-trial scalar that is broadcast across all time bins. The output follows the instruction mapping: left = 0, right = 1.

ii.
```python
lick_dir = np.where(bp['R'] > bp['L'], 1, 0).astype(np.int64)
# Broadcast to all time bins:
np.full((len(time_centers),), lick_dir[ti], dtype=np.int64)
```

iii. The reference code uses `R` and `L` as boolean flags in trial conditions (e.g., `'R&hit&~stim.enable&~autowater&~early'`). The AI's comparison approach (`R > L`) achieves a similar result when R and L are binary indicators.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj.bp.autowater`.

ii.
```python
out['autowater'] = np.asarray(bp[k][()]).astype(np.float32).ravel()
# ...
context = np.where(bp['autowater'] > 0, 0, 1).astype(np.int64)
```

iii. The reference code and paper use `autowater` to distinguish WC (autowater on) from DR (autowater off) contexts, consistent with the AI's mapping.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. If `autowater > 0`, context is WC (0); otherwise DR (1). This is a per-trial scalar broadcast to all time bins.

ii.
```python
context = np.where(bp['autowater'] > 0, 0, 1).astype(np.int64)
# Broadcast:
np.full((len(time_centers),), context[ti], dtype=np.int64)
```

iii. This matches the instruction mapping: WC = 0, DR = 1. However, for the randomized delay task, nearly all trials are DR (98.7%), making this output nearly degenerate.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit`.

ii.
```python
out['hit'] = np.asarray(bp[k][()]).astype(np.float32).ravel()
# ...
outcome = np.where(bp['hit'] > 0, 1, 0).astype(np.int64)
```

iii. The reference code uses `hit` and `miss` flags in trial conditions.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. If `hit > 0`, outcome is correct (1); otherwise incorrect (0). After filtering out early lick and no-go trials, the remaining non-hit trials are misses (incorrect). This is a per-trial scalar broadcast to all time bins.

ii.
```python
outcome = np.where(bp['hit'] > 0, 1, 0).astype(np.int64)
np.full((len(time_centers),), outcome[ti], dtype=np.int64)
```

iii. Matches instruction mapping: incorrect = 0, correct = 1.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from `obj.traj` video tracking data. The AI searches for tongue-related features in both video streams (stream 0: side view, stream 1: bottom view), looking for feature names like 'tongue', 'left_tongue', 'right_tongue', 'top_tongue', 'bottom_tongue', etc.

ii.
```python
tongue_idx0 = choose_feature(names0, ['tongue', 'left_tongue', 'right_tongue'])
tongue_idx1 = choose_feature(names1, ['top_tongue', 'bottom_tongue', 'topleft_tongue', 'bottomleft_tongue'])
```

iii. The reference code's `getDefaultParams.m` specifies tongue features from both camera views, matching the AI's feature selection.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Speed is computed from the x,y position tracking data as `sqrt(dx^2 + dy^2) / dt`, where dx and dy are the differences in the first two position dimensions and dt is the frame time difference. The speed is then binned into the 75 ms neural time bins using `nanbin_mean` (averaging within each bin, ignoring NaN values).

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

iii. The reference paper describes: "The velocity of each feature was then calculated as the first-order derivative of the position vector." The AI's approach computes velocity magnitude (speed) from the position derivative.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The 50th percentile (median) of all finite tongue speed values across all trials in a session is used as the threshold. Values below the threshold are 0 (low), values >= threshold are 1 (high). However, the actual threshold is 0.0 for ALL sessions, meaning nearly all non-zero values are categorized as "high" (86% high, 14% low).

ii.
```python
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
        out.append(b)
    return out, thr
```

iii. The instructions specify "50th percentile" per-session threshold. The threshold of 0.0 suggests that the median tongue speed (including all time bins, even when tongue is invisible and speed is 0) is 0. This creates a heavily imbalanced distribution.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue frame times are aligned to the go cue by subtracting `bp['goCue'][ti]` from the video frame times. The aligned times are then binned into the same 75 ms time edges used for neural data.

ii.
```python
go = bp['goCue'][ti]
tongue_bin = nanbin_mean(tmid - go, spd[idx], time_edges)
```

iii. Behavioral data is aligned to go cue, consistent with the instructions.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from `obj.traj` video tracking data from stream 1 (bottom view camera), looking for features named 'top_paw' or 'bottom_paw'.

ii.
```python
paw_idx1 = choose_feature(names1, ['top_paw', 'bottom_paw'])
```

iii. The reference `getDefaultParams.m` includes paw features in stream 1: `{'top_paw','bottom_paw'}`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same as tongue: speed is computed from x,y position differences using `speed_from_ts`, then binned into 75 ms bins using `nanbin_mean`.

ii.
```python
if paw_idx1 is not None:
    ts, ft = get_stream1(ti)
    spd, tmid = speed_from_ts(ts, ft)
    paw_bin = nanbin_mean(tmid - go, spd[paw_idx1], time_edges)
```

iii. Same velocity computation as tongue.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Per-session 50th percentile threshold, same as tongue. Paw thresholds are non-zero (ranging ~72 to ~175), producing a more balanced distribution (57% low, 43% high).

ii.
```python
paw_bin, paw_thr = threshold_session_bins(paw_vals)
```

iii. Matches the instruction specification of 50th percentile per-session threshold.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same alignment as tongue: frame times minus go cue time, binned into neural time edges.

ii.
```python
paw_bin = nanbin_mean(tmid - go, spd[paw_idx1], time_edges)
```

iii. Consistent go-cue alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from `motionEnergy_<session>.mat` files, specifically from the `me.data` field. The `moveThresh` field is also loaded but not used for thresholding in the final code.

ii.
```python
def load_motion_energy(path):
    me = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)['me']
    move_thresh = getattr(me, 'moveThresh', None) if hasattr(me, '_fieldnames') else None
    raw = me
    while hasattr(raw, '_fieldnames') and hasattr(raw, 'data'):
        raw = raw.data
    arr = np.asarray(raw, dtype=object)
    out = [normalize_motion_elem(x) for x in arr.flat]
    return out, move_thresh
```

iii. The reference code loads motion energy from the same files via `loadMotionEnergy.m`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Raw motion energy values (per trial, at video frame rate) are binned into 75 ms neural time bins using `nanbin_mean`. The frame times from video stream 0 are used for temporal alignment. No interpolation to a neural time grid is performed (unlike the reference code which interpolates to 200 Hz). No video offset correction is applied (unlike the reference code which uses `findVideoOffset`). No missing value filling is performed (unlike the reference code which uses `fillmissing('nearest')`).

ii.
```python
if ti < len(motion_data):
    me = np.asarray(motion_data[ti]).ravel().astype(np.float32)
    _, ft0 = get_stream0(ti)
    me_bin = nanbin_mean(ft0 - go, me, time_edges) if len(ft0) == len(me) else np.full((len(time_edges)-1,), np.nan, dtype=np.float32)
```

iii. The reference `loadMotionEnergy.m` performs: (1) video offset correction via `findVideoOffset`, (2) interpolation to neural time grid at 200 Hz using `interp1`, and (3) NaN filling with nearest values. The AI skips all three of these steps.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session 50th percentile threshold, same mechanism as tongue and paw. Motion energy thresholds range ~22 to ~54 across sessions, producing a relatively balanced distribution (54% low, 46% high).

ii.
```python
me_bin, me_thr = threshold_session_bins(me_vals)
```

iii. Matches the instruction specification. The thresholding uses the 50th percentile across all finite values per session.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Frame times from video stream 0 are subtracted by the go cue time, then binned into the neural time edges. This assumes the motion energy time series corresponds to the frame times from stream 0.

ii.
```python
me_bin = nanbin_mean(ft0 - go, me, time_edges)
```

iii. The reference code uses `interp1(frameTimes - vidshift - alignTimes(trix), me.data{trix}, taxis)` which includes video offset correction. The AI's approach is simpler and omits the video offset.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several mechanisms handle missing data: (1) NaN-filled arrays are used as defaults for behavioral data when features are not found or data is missing. (2) `nanbin_mean` ignores NaN values when computing bin averages. (3) Motion energy trials beyond the available data length are filled with NaN. (4) Frame time length mismatches result in NaN-filled bins. (5) Zero dt values in velocity computation are set to NaN to avoid division by zero. (6) Trials with all-zero neural activity are removed.

ii.
```python
# NaN defaults
tongue_bin = np.full((len(time_edges)-1,), np.nan, dtype=np.float32)
paw_bin = np.full((len(time_edges)-1,), np.nan, dtype=np.float32)

# NaN-aware binning
def nanbin_mean(times, values, edges):
    out = np.full((len(edges) - 1,), np.nan, dtype=np.float32)
    # ...
    if np.any(np.isfinite(vv)):
        out[bi] = np.nanmean(vv)

# Zero dt handling
dt[dt <= 0] = np.nan

# Threshold: NaN values mapped to 0
b = np.zeros_like(a, dtype=np.int64)
if np.isfinite(thr):
    valid = np.isfinite(a)
    b[valid] = (a[valid] >= thr).astype(np.int64)
```

iii. The AI handles NaN propagation systematically. NaN values in behavioral outputs are ultimately mapped to 0 (low category) during thresholding, since only valid (finite) values above threshold get assigned 1. The reference code fills NaN values with nearest values (`fillmissing('nearest')`), which preserves more data.

## 11-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is `bin_unit_spikes_for_trials`, which has a doubly-nested loop over all units and all trials. For each unit-trial pair, it filters spikes by trial identity and computes a histogram. With ~500 units and ~350 trials per session, this is ~175,000 iterations per session. The behavioral data extraction (`extract_binned_behavior_generic`) also loops over all trials.

ii.
```python
def bin_unit_spikes_for_trials(trialtm_list, trial_list, n_trials, time_edges):
    for ui, (ttm, tr) in enumerate(zip(trialtm_list, trial_list)):
        for trial_idx in range(1, n_trials + 1):
            mask = tr == trial_idx
            if np.any(mask):
                counts, _ = np.histogram(ttm[mask], bins=time_edges)
```

iii. The CONVERSION_NOTES acknowledge the need for efficiency but don't document specific timing analysis.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The `bin_unit_spikes_for_trials` function could vectorize the inner trial loop by pre-sorting spikes by trial and using array operations. The `nanbin_mean` function loops over bins and could use `np.digitize` + `np.bincount` for vectorized binning. The behavioral extraction loop over trials could use vectorized operations on the trajectory arrays.

ii.
```python
# nanbin_mean: per-bin loop
for bi in range(len(out)):
    m = idx == bi
    if np.any(m):
        vv = values[m]
        if np.any(np.isfinite(vv)):
            out[bi] = np.nanmean(vv)
```

iii. No explicit vectorization improvements are documented.

## 11-c. What processing does the code repeat multiple times?

i. The trajectory stream data (`get_stream0(ti)` / `get_stream1(ti)`) is loaded multiple times per trial: once for tongue (potentially from both streams), once for paw, and once for motion energy (which uses stream 0 frame times). The speed computation (`speed_from_ts`) may be called twice for the same stream data if tongue features are checked in both streams.

ii.
```python
# In extract_binned_behavior_generic:
for idx, getter in [(tongue_idx1, get_stream1), (tongue_idx0, get_stream0)]:
    ts, ft = getter(ti)
    spd, tmid = speed_from_ts(ts, ft)
# ...
if paw_idx1 is not None:
    ts, ft = get_stream1(ti)  # Potentially re-loads stream1
    spd, tmid = speed_from_ts(ts, ft)  # Recomputes speed
```

iii. No caching or deduplication is implemented.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) All behavioral data (tongue, paw, motion energy) is computed for ALL trials before trial filtering. Trials that are later removed (early lick, no-go, all-zero neural) have their behavioral data discarded. (2) The `moveThresh` value from motion energy files is loaded and stored in session info but never used for thresholding (the code uses 50th percentile instead). (3) The `bitRand` field is loaded from behavioral data but never used.

ii.
```python
# Behavioral computed before filtering:
tongue_vals, paw_vals, me_vals = extract_binned_behavior_generic(...)
# Then filtering removes some trials:
keep_idx = np.where(valid)[0]
keep_idx = [i for i in keep_idx if not np.all(neural[i] == 0)]
```

iii. Computing behavioral data for trials that will be filtered out is wasteful but doesn't affect correctness.
