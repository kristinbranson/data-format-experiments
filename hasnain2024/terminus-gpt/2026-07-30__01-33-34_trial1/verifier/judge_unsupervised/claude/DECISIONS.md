# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data exclusively from the `data/RandomizedDelay_Ephys_Behavior/` directory using a hardcoded list of 19 session identifiers (`RAND_SESSIONS`). For each session, two MAT files are loaded: `data_structure_<session>.mat` (neural/behavioral data) and `motionEnergy_<session>.mat` (motion energy data). The code handles two MATLAB file formats: HDF5/v7.3 (via h5py) and older MAT (via scipy.io.loadmat), with a try/except fallback.

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
for key in keys:
    df = base / f'data_structure_{key}.mat'
    mf = base / f'motionEnergy_{key}.mat'
    neural, inp, out, bri, info = process_session(df, mf)
```

iii. The AI identified that the reference code loading scripts enumerate specific sessions for each mouse (JEB11, JEB12, JEB23, JEB24) and that the paper reports 19 sessions from 4 mice for the randomized delay task. The AI resolved a discrepancy between 22 raw files in the directory and 19 sessions reported in the paper by adopting the session list from the reference loading scripts.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by parsing the session key prefix (e.g., "JEB11" from "JEB11_2022-05-10"). A list of unique subject names is built incrementally, and each session is assigned a subject index.

ii.
```python
for key in keys:
    subj = key.split('_')[0]
    if subj not in subject_names:
        subject_names.append(subj)
    # ...
    subject_idx.append(subject_names.index(subj))
```

iii. The AI determined subjects from session identifiers, which is consistent with how the reference code organizes sessions by animal name (JEB11, JEB12, JEB23, JEB24). The resulting 4 subjects match the paper's report.

## 1-c. How are the data split into sessions?

i. Each entry in the `RAND_SESSIONS` list corresponds to one session. The session identifier combines subject name and date (e.g., "JEB11_2022-05-10"). Each session produces one entry in the neural, input, and output lists.

ii.
```python
for key in keys:
    # ... each iteration processes one session
    all_neural.append(neural)
    all_input.append(inp)
    all_output.append(out)
```

iii. The session list was derived from the reference code loading scripts, matching the paper's 19 sessions for the randomized delay task.

## 1-d. How are the data split into trials?

i. Trials are determined from `bp['Ntrials']` (from `obj.bp.Ntrials` in the MAT file). Spike data is binned per trial using 1-indexed trial identifiers from `obj.clu.trial`. Behavioral data is similarly indexed per trial.

ii.
```python
n_trials = bp['Ntrials']
# In bin_unit_spikes_for_trials:
for trial_idx in range(1, n_trials + 1):
    mask = tr == trial_idx
```

iii. The AI uses the trial count from the behavioral parameters structure, consistent with how the reference code accesses trial information.

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded if they have early licks (`bp['early'] == 1`) or are ignore/no-response trials (`bp['no'] == 1`). Additionally, trials with all-zero neural activity are removed. No filtering based on minimum trial counts per condition (e.g., 40 correct DR trials per direction) is applied. No session-level filtering based on minimum number of units (>=10) is applied.

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

iii. The AI documented in CONVERSION_NOTES.md Step 3 that "Early lick and ignore trials are omitted from analyses" per the paper methods. The additional zero-neural-activity filter is a reasonable quality control not explicitly mentioned in the reference but prevents degenerate trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `obj.clu`, which contains per-unit spike time data. Specifically, `clu.trialtm` (spike times within trial) and `clu.trial` (trial assignment for each spike) are extracted for each unit.

ii.
```python
# HDF5 path:
def extract_clu_h5(f):
    clu = deref(f, f['obj/clu'][()].flat[0])
    trialtm = [np.asarray(deref(f, r)[()]).ravel().astype(np.float32) for r in clu['trialtm'][()].flat]
    trial = [np.asarray(deref(f, r)[()]).ravel().astype(int) for r in clu['trial'][()].flat]

# Old MAT path:
def extract_clu_old(obj):
    clu_arr = np.asarray(obj.clu, dtype=object).ravel()
    trialtm = [np.asarray(unwrap_obj(c).trialtm).ravel().astype(np.float32) for c in clu_arr]
    trial = [np.asarray(unwrap_obj(c).trial).ravel().astype(int) for c in clu_arr]
```

iii. The AI identified `obj.clu` as containing cluster/neuron information with spike times and trial assignments, consistent with the reference code's use of aligned spike data.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into spike count histograms using 75 ms time bins spanning -2.5 to +2.5 seconds relative to go cue. The result is raw spike counts per bin (not firing rates). No further processing (e.g., smoothing, normalization) is applied.

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

iii. The AI chose 75 ms bins based on the reference decoding scripts (`rez.binSize = 75` ms) and a -2.5 to +2.5 s window based on the reference code's alignment logic (`aligntimes = mode(obj(sessix).bp.ev.goCue) - 2.5`).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality filtering is applied. The code does not filter neurons by firing rate (>1 Hz threshold mentioned in the paper/methods), does not check for unit isolation quality, and does not enforce a minimum number of units per session (>=10 as mentioned in the paper). All units from `obj.clu` are included.

ii.
```python
# No neuron filtering code exists. All units from clu are used:
trialtm, trial, n_units = extract_clu_h5(f)  # or extract_clu_old
# n_units = len(trialtm) -- all units included
brain_region_idx = np.zeros((n_units,), dtype=int)
```

iii. The AI documented in CONVERSION_NOTES.md Step 3 that the paper requires "units with firing rates > 1 Hz" and "sessions only if they had at least 10 units," but did not implement these filters in the conversion script. Step 10 (Critical Review 1) was marked as NOT STARTED, meaning these checks were never performed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data alignment relies on the assumption that `clu.trialtm` spike times are already expressed relative to the go cue. The time bins are defined from -2.5 to +2.5 s, and spike times from `trialtm` are directly histogrammed into these bins without subtracting any additional event time.

ii.
```python
time_edges = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE, dtype=np.float32)
# ...
counts, _ = np.histogram(ttm[mask], bins=time_edges)
```

iii. The AI's CONVERSION_NOTES.md references the alignment logic from `getCodingDimensions_2afc.m`: `aligntimes = mode(obj(sessix).bp.ev.goCue) - 2.5`, suggesting that the reference code uses the mode of go cue times minus 2.5 as the alignment reference. The AI appears to assume `trialtm` is already in a go-cue-centered time frame, which needs verification against the reference `alignSpikes.m` function.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 75 ms (0.075 s). No rebinning is applied -- spikes are directly binned from continuous spike times into 75 ms bins. The time window spans -2.5 to +2.5 s, producing 67 time bins per trial.

ii.
```python
BIN_SIZE = 0.075
time_edges = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE, dtype=np.float32)
# 67 bins total
```

iii. The 75 ms bin size matches `rez.binSize = 75` in the reference decoding scripts. The metadata stores `time_bin_size: 0.075` (in seconds), though the instructions specify the field should be in ms.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not derived from any raw data variable. It is synthetically constructed as the center of each time bin, which are defined relative to go cue onset.

ii.
```python
def make_time_input(n_trials, time_centers):
    return [time_centers[None, :].astype(np.float32).copy() for _ in range(n_trials)]
# ...
time_centers = (time_edges[:-1] + time_edges[1:]) / 2
input_trials = make_time_input(n_trials, time_centers)
```

iii. The time input is a deterministic function of the bin edges, representing time from go cue in seconds. This is consistent with the instruction to provide "Time from go cue onset in seconds" as a continuous, time-varying decoder input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Time bin centers are computed as the midpoint of each bin edge pair. The values range from approximately -2.4625 to +2.4625 s. The same time vector is replicated for every trial.

ii.
```python
time_centers = (time_edges[:-1] + time_edges[1:]) / 2
# Result shape: (1, 67) for each trial
```

iii. No justification is explicitly provided; this is a standard approach for representing time-varying input.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input uses the same time bin edges as the neural data, so they are inherently aligned. Both use bins from -2.5 to +2.5 s relative to go cue.

ii.
```python
time_edges = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE, dtype=np.float32)
time_centers = (time_edges[:-1] + time_edges[1:]) / 2
# Same time_edges used for neural binning and input construction
```

iii. Alignment is implicit through shared time bin definitions.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `obj.bp.L` (left lick flag) and `obj.bp.R` (right lick flag).

ii.
```python
# Extracted from bp:
out['L'] = np.asarray(bp['L'][()]).astype(np.float32).ravel()
out['R'] = np.asarray(bp['R'][()]).astype(np.float32).ravel()
# ...
lick_dir = np.where(bp['R'] > bp['L'], 1, 0).astype(np.int64)
```

iii. The AI used `bp.L` and `bp.R` trial flags, consistent with reference code condition groupings that use these fields for left/right classification.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. If `R > L`, lick direction is 1 (right); otherwise 0 (left). This is a per-trial constant value that is broadcast across all time bins.

ii.
```python
lick_dir = np.where(bp['R'] > bp['L'], 1, 0).astype(np.int64)
# ...
np.full((len(time_centers),), lick_dir[ti], dtype=np.int64)
```

iii. This matches the instruction specification of left=0, right=1, per-trial. The use of `R > L` comparison is a reasonable interpretation of lick direction from the behavioral flags.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj.bp.autowater`.

ii.
```python
out['autowater'] = np.asarray(bp['autowater'][()]).astype(np.float32).ravel()
# ...
context = np.where(bp['autowater'] > 0, 0, 1).astype(np.int64)
```

iii. The AI documented that WC (water-cued) trials correspond to autowater trials and DR (delay-response) to non-autowater, consistent with the reference code and methods.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. If `autowater > 0`, context is 0 (WC); otherwise 1 (DR). This is a per-trial constant broadcast across all time bins. The resulting distribution is extremely imbalanced: 98.7% DR, 1.3% WC, because the randomized delay task is primarily DR with occasional WC catch trials.

ii.
```python
context = np.where(bp['autowater'] > 0, 0, 1).astype(np.int64)
np.full((len(time_centers),), context[ti], dtype=np.int64)
```

iii. The mapping matches the instruction WC=0, DR=1.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit`.

ii.
```python
out['hit'] = np.asarray(bp['hit'][()]).astype(np.float32).ravel()
# ...
outcome = np.where(bp['hit'] > 0, 1, 0).astype(np.int64)
```

iii. The AI uses the `hit` flag to determine correct (1) vs incorrect (0) trials.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. If `hit > 0`, outcome is 1 (correct); otherwise 0 (incorrect). This is per-trial and broadcast across all time bins. The `miss` field is loaded but not directly used -- the logic assumes any non-hit trial that passes the early/no filter is incorrect.

ii.
```python
outcome = np.where(bp['hit'] > 0, 1, 0).astype(np.int64)
np.full((len(time_centers),), outcome[ti], dtype=np.int64)
```

iii. This matches the instruction specification of incorrect=0, correct=1.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from `obj.traj` trajectory data. The code searches for tongue-related features across two video streams: stream 0 features like 'tongue', 'left_tongue', 'right_tongue'; and stream 1 features like 'top_tongue', 'bottom_tongue', 'topleft_tongue', 'bottomleft_tongue'.

ii.
```python
tongue_idx0 = choose_feature(names0, ['tongue', 'left_tongue', 'right_tongue'])
tongue_idx1 = choose_feature(names1, ['top_tongue', 'bottom_tongue', 'topleft_tongue', 'bottomleft_tongue'])
```

iii. Feature names match those listed in the reference behavioral analysis scripts (e.g., `LickRaster_DR_WC.m` feature lists).

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Velocity is computed as Euclidean speed from x,y coordinates of tongue DLC tracking: `sqrt(dx^2 + dy^2) / dt`. The code uses `speed_from_ts()` which takes the first 2 dimensions (x,y) of the trajectory time series, computes frame-to-frame differences, and divides by frame time intervals. The resulting speed values are then binned into the same time bins as neural data using `nanbin_mean()`, which averages speed values within each bin.

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

iii. The velocity computation follows standard kinematics (Euclidean speed from position differences).

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The 50th percentile (median) of ALL finite tongue velocity values across ALL trials and ALL time bins within a session is computed. Values >= threshold become 1 (high), values < threshold become 0 (low). NaN values (missing data) default to 0. Critically, the tongue velocity threshold is 0.0 for ALL 19 sessions, indicating that the median tongue velocity is exactly 0 -- most time bins have zero tongue velocity because the tongue is not visible/moving in most time bins. This results in a highly skewed distribution: ~14% low, ~86% high.

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

iii. The instructions specify "discretized into two bins with per-session threshold: 0: < 50th percentile, 1: >= 50th percentile." The code implements this correctly in terms of the median-based threshold, but the threshold being 0.0 means any non-zero velocity is "high," creating a very imbalanced distribution.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue velocity is aligned by subtracting the go cue time from the frame timestamps before binning into the same time edges used for neural data.

ii.
```python
candidate = nanbin_mean(tmid - go, spd[idx], time_edges)
# where go = bp['goCue'][ti]
```

iii. The go cue subtraction ensures temporal alignment with the neural data bins.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from `obj.traj` trajectory data, specifically from stream 1 features named 'top_paw' or 'bottom_paw'.

ii.
```python
paw_idx1 = choose_feature(names1, ['top_paw', 'bottom_paw'])
```

iii. Feature names match the reference behavioral analysis scripts.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Identical to tongue velocity: Euclidean speed from x,y coordinates, then binned into 75 ms time bins using `nanbin_mean()`.

ii.
```python
if paw_idx1 is not None:
    ts, ft = get_stream1(ti)
    spd, tmid = speed_from_ts(ts, ft)
    paw_bin = nanbin_mean(tmid - go, spd[paw_idx1], time_edges)
```

iii. Same velocity computation as tongue.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue velocity: per-session median thresholding. Unlike tongue, the paw velocity thresholds are non-zero (ranging from ~73 to ~175), producing a more balanced distribution (~57% low, ~43% high).

ii.
```python
paw_bin, paw_thr = threshold_session_bins(paw_vals)
```

iii. Follows the instruction specification for 50th percentile per-session thresholding.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue velocity: frame times are shifted by go cue time before binning into shared time edges.

ii.
```python
paw_bin = nanbin_mean(tmid - go, spd[paw_idx1], time_edges)
```

iii. Go-cue-aligned binning ensures temporal alignment with neural data.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from `motionEnergy_<session>.mat` files, specifically `me.data` (per-trial motion energy time series). The file also contains `me.moveThresh` (a session-level motion threshold, stored but not used for discretization).

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

iii. The AI correctly identified motion energy as coming from separate `motionEnergy_*.mat` files.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The raw motion energy values per trial are binned into the same 75 ms time bins as neural data using `nanbin_mean()`. The motion energy is aligned using frame times from trajectory stream 0.

ii.
```python
if ti < len(motion_data):
    me = np.asarray(motion_data[ti]).ravel().astype(np.float32)
    _, ft0 = get_stream0(ti)
    me_bin = nanbin_mean(ft0 - go, me, time_edges) if len(ft0) == len(me) else np.full((len(time_edges)-1,), np.nan, dtype=np.float32)
```

iii. Using frame times from stream 0 for motion energy alignment is reasonable since motion energy is typically computed from the same video source.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same per-session median thresholding as tongue and paw velocity. Motion energy thresholds range from ~22 to ~54, producing a relatively balanced distribution (~54% low, ~47% high).

ii.
```python
me_bin, me_thr = threshold_session_bins(me_vals)
```

iii. Follows the instruction specification.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by subtracting go cue time from frame times of trajectory stream 0, then binning into the same time edges.

ii.
```python
me_bin = nanbin_mean(ft0 - go, me, time_edges)
```

iii. Uses go-cue-centered time bins consistent with neural data alignment.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies are used:
- Missing/NaN behavioral values: `nanbin_mean()` uses `np.nanmean()` to skip NaN values during binning, and bins with no valid data remain NaN.
- Motion energy length mismatch: If `len(ft0) != len(me)`, the trial's motion energy is set to all NaN.
- Trials with all-zero neural activity: Excluded from the final dataset.
- NaN values in discretized outputs: Default to 0 (low) due to the thresholding logic initializing with zeros.
- Zero/negative frame time differences: Set to NaN in `speed_from_ts()` to avoid division errors.
- Mixed MATLAB formats: Try/except fallback between HDF5 and old MAT readers.
- Missing tongue features: Falls back from stream 1 to stream 0 features.

ii.
```python
# NaN handling in speed computation:
dt[dt <= 0] = np.nan

# NaN handling in binning:
def nanbin_mean(times, values, edges):
    out = np.full((len(edges) - 1,), np.nan, dtype=np.float32)
    # ...
    if np.any(np.isfinite(vv)):
        out[bi] = np.nanmean(vv)

# Motion energy length check:
me_bin = nanbin_mean(ft0 - go, me, time_edges) if len(ft0) == len(me) else np.full(..., np.nan, ...)

# NaN-to-zero in thresholding:
b = np.zeros_like(a, dtype=np.int64)
b[valid] = (a[valid] >= thr).astype(np.int64)
```

iii. The AI documented edge cases with mixed MATLAB formats and motion energy normalization issues in CONVERSION_NOTES.md Step 6.

## 11-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is likely `bin_unit_spikes_for_trials()`, which iterates over every unit and every trial to create spike count histograms. For sessions with many units (up to 560) and many trials (up to 450), this involves many iterations. The behavioral extraction (`extract_binned_behavior_generic()`) also loops over all trials and calls `speed_from_ts()` and `nanbin_mean()` for each.

ii.
```python
def bin_unit_spikes_for_trials(trialtm_list, trial_list, n_trials, time_edges):
    for ui, (ttm, tr) in enumerate(zip(trialtm_list, trial_list)):
        for trial_idx in range(1, n_trials + 1):
            # ...
```

iii. CONVERSION_NOTES.md Step 6 does not document timing information or speedup attempts. The Step 7 timing estimates section is left empty.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
- `bin_unit_spikes_for_trials()`: The nested loop over units and trials could potentially be replaced with a single vectorized operation using `np.histogram2d` or groupby operations.
- `nanbin_mean()`: The loop over bins could be replaced with `scipy.stats.binned_statistic` or vectorized groupby operations using `np.digitize` followed by `np.bincount`.
- `extract_binned_behavior_generic()`: The trial loop with individual `speed_from_ts()` and `nanbin_mean()` calls could potentially batch process multiple trials.

ii.
```python
# nanbin_mean loop:
for bi in range(len(out)):
    m = idx == bi
    if np.any(m):
        vv = values[m]
        if np.any(np.isfinite(vv)):
            out[bi] = np.nanmean(vv)
```

iii. No vectorization optimizations are documented.

## 11-c. What processing does the code repeat multiple times?

i. The code calls `get_stream0(ti)` and `get_stream1(ti)` multiple times for the same trial within `extract_binned_behavior_generic()` -- once for tongue and once for paw (stream 1 is fetched twice when both tongue and paw features come from stream 1). Stream 0 is also fetched for motion energy alignment in addition to tongue. Each call re-reads and re-processes the trajectory data from the MAT file.

ii.
```python
# Stream 1 fetched for tongue:
ts, ft = getter(ti)  # get_stream1
# Stream 1 fetched again for paw:
ts, ft = get_stream1(ti)
# Stream 0 fetched for motion energy:
_, ft0 = get_stream0(ti)
```

iii. No caching of trajectory data is documented.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads and stores `me.moveThresh` from motion energy files, but this threshold is never used for any processing -- the actual thresholding uses the 50th percentile computed from the data. The `bitRand`, `bitStart`, `sample`, `delay`, and `reward` event fields are extracted but never used in the conversion (only `goCue` is used for alignment). The `miss` behavioral flag is also extracted but never used.

ii.
```python
# moveThresh stored but unused:
move_thresh = getattr(me, 'moveThresh', None)
info['moveThresh'] = move_thresh  # stored in metadata only

# Unused event fields:
for k in ['bitStart', 'sample', 'delay', 'goCue', 'reward']:
    out[k] = np.asarray(ev[k][()]).astype(np.float32).ravel()

# Unused behavioral fields:
for k in ['L', 'R', 'autowater', 'bitRand', 'early', 'hit', 'miss', 'no']:
    out[k] = ...
# 'bitRand' and 'miss' are never referenced after extraction
```

iii. No justification for extracting unused fields is provided.
