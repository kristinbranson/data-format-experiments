# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI only loads sessions from the `RandomizedDelay_Ephys_Behavior` subdirectory. A hard-coded list of 19 randomized-delay session names (`RAND_SESSIONS`) is used. The AI does NOT load the 25 fixed-delay sessions from `Ephys_Behavior`. Each session's data structure and motion energy are loaded from paired `.mat` files. The AI handles both HDF5 (v7.3) and older MAT formats by trying `h5py` first and falling back to `scipy.io.loadmat`.

ii.
```python
RAND_SESSIONS = [
    'JEB11_2022-05-10','JEB11_2022-05-11',
    'JEB12_2022-05-12','JEB12_2022-05-13',
    ...
    'JEB24_2023-11-03'
]

base = Path('data/RandomizedDelay_Ephys_Behavior')
keys = RAND_SESSIONS[:2] if args.sample else RAND_SESSIONS
for key in keys:
    df = base / f'data_structure_{key}.mat'
    mf = base / f'motionEnergy_{key}.mat'
    neural, inp, out, bri, info = process_session(df, mf)
```

iii. The AI's CONVERSION_NOTES.md acknowledges finding all data directories but chose to focus on the randomized-delay sessions. No explicit justification is given for excluding the 25 fixed-delay Ephys_Behavior sessions.

## 1-b. How are the data split into subjects?

i. Subject is extracted from the session name by splitting on underscore and taking the first part. Subjects are accumulated into a list as new ones are encountered. The AI uses 4 subjects (from the randomized-delay subset) rather than 14.

ii.
```python
subj = key.split('_')[0]
if subj not in subject_names:
    subject_names.append(subj)
subject_idx.append(subject_names.index(subj))
```

iii. No specific justification given. The subject extraction method itself is correct.

## 1-c. How are the data split into sessions?

i. Each session corresponds to one entry in the `RAND_SESSIONS` list and one pair of data files. Only 19 randomized-delay sessions are processed, not the full 44. The AI hardcodes the path to the randomized-delay directory only.

ii.
```python
base = Path('data/RandomizedDelay_Ephys_Behavior')
keys = RAND_SESSIONS[:2] if args.sample else RAND_SESSIONS
```

iii. The AI's CONVERSION_NOTES.md notes that the reference code's loader scripts enumerate 19 randomized-delay sessions, but does not address the 25 fixed-delay sessions.

## 1-d. How are the data split into trials?

i. Trials are identified using `bp['Ntrials']` to determine the number of trials per session. Each trial index from 1 to Ntrials is processed. The AI uses the `Ntrials` field from the behavior protocol data.

ii.
```python
n_trials = bp['Ntrials']
neural = bin_unit_spikes_for_trials(trialtm, trial, n_trials, time_edges)
```

iii. No specific justification. The approach of using `Ntrials` is consistent with the reference.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters out trials where `early == 0` AND `no == 0` (keeping non-early, non-ignore trials). This drops ignore trials but does NOT drop photostimulation trials (`stim.enable`). Additionally, trials where all neural data is zero are dropped. The reference drops early-lick and photostim trials but keeps ignore trials.

ii.
```python
valid = (bp['early'] == 0) & (bp['no'] == 0)
keep_idx = np.where(valid)[0]
keep_idx = [i for i in keep_idx if not np.all(neural[i] == 0)]
```

iii. The AI's CONVERSION_NOTES.md mentions "omit early lick and ignore trials" based on the paper methods, but the paper actually says to omit early lick trials. Ignore trials are kept in the reference as a third outcome class.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses `obj.clu` spike-sorted cluster data, specifically `trialtm` (spike times relative to trial start) and `trial` (trial assignment for each spike). For HDF5 files, these are accessed via `extract_clu_h5`; for older MAT files via `extract_clu_old`.

ii.
```python
def extract_clu_h5(f):
    clu = deref(f, f['obj/clu'][()].flat[0])
    trialtm = [np.asarray(deref(f, r)[()]).ravel().astype(np.float32) for r in clu['trialtm'][()].flat]
    trial = [np.asarray(deref(f, r)[()]).ravel().astype(int) for r in clu['trial'][()].flat]
    n_units = len(trialtm)
    return trialtm, trial, n_units
```

iii. No specific justification beyond using the available cluster data.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 75ms time bins spanning -2.5 to 2.5s relative to go cue. The AI does NOT convert counts to firing rates (Hz), does NOT apply Gaussian smoothing, and stores raw spike counts as float32. The reference converts to Hz (divides by bin width) and applies 14ms Gaussian smoothing.

ii.
```python
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

iii. The AI's CONVERSION_NOTES.md mentions "bin aligned neural activity to common time bins (likely 75 ms)" based on seeing `rez.binSize = 75` in the decoding scripts. However, the reference uses 5ms bins (matching `params.dt = 1/200`).

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does NOT filter neurons based on quality labels or minimum firing rate. All clusters from `obj.clu` are included regardless of quality. The reference filters by cluster quality labels (excluding garbage, gabrga, noisy, real?, poor) and minimum mean firing rate (>1 Hz).

ii.
```python
# No quality filtering code - all clusters used
trialtm = [np.asarray(deref(f, r)[()]).ravel().astype(np.float32) for r in clu['trialtm'][()].flat]
```

iii. No justification provided. The AI's CONVERSION_NOTES.md notes the paper mentions "units with firing rates > 1 Hz" but this was not implemented.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns to the go cue by subtracting the go cue time from spike times. Spikes are binned relative to go cue using `trialtm[mask]` directly (which are already relative to trial start), then the time edges are defined relative to go cue onset. However, the alignment is done incorrectly: `trialtm` is relative to trial start, but the AI does NOT subtract `goCue` from `trialtm` before binning. The go cue subtraction happens only for the behavioral data.

ii.
```python
time_edges = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE, dtype=np.float32)
# Neural: spikes binned using time_edges directly from trialtm
counts, _ = np.histogram(ttm[mask], bins=time_edges)
# Behavioral: go cue subtracted
tongue_bin = nanbin_mean(tmid - go, spd[idx], time_edges)
```

iii. The AI's CONVERSION_NOTES.md mentions go cue alignment but the neural binning code uses `trialtm` directly without subtracting the go cue time. This means neural data is aligned to trial start, not go cue, while behavioral data is aligned to go cue -- a temporal misalignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 75ms bins (`BIN_SIZE = 0.075`). This produces approximately 67 bins for the -2.5 to 2.5s window. The reference uses 5ms bins (1000 bins). The AI's `time_bin_size` metadata is stored as 0.075 (seconds) rather than in ms as required by the format specification.

ii.
```python
BIN_SIZE = 0.075
time_edges = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE, dtype=np.float32)
# metadata:
'time_bin_size': BIN_SIZE,  # Should be in ms per format spec
```

iii. The AI chose 75ms based on `rez.binSize = 75` in the decoding analysis scripts, which is the bin size for the decoding analysis, not the underlying data resolution. The paper's underlying temporal resolution is 5ms (`params.dt = 1/200`).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is computed from the time bin centers of the binning grid. It is not derived from any raw data variable but from the defined time window and bin size.

ii.
```python
time_centers = (time_edges[:-1] + time_edges[1:]) / 2
input_trials = make_time_input(n_trials, time_centers)
```

iii. No specific justification. This is the correct approach -- the input is the time axis itself.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time centers are computed as the midpoints of the bin edges. Each trial gets the same time vector, shaped as `(1, n_bins)`.

ii.
```python
def make_time_input(n_trials, time_centers):
    return [time_centers[None, :].astype(np.float32).copy() for _ in range(n_trials)]
```

iii. No special processing needed.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. Both neural data and the input use the same time bin edges, so they are aligned by construction. However, as noted in 2-d, the neural data may not actually be aligned to the go cue (trialtm not adjusted for goCue), while the time input assumes go cue alignment.

ii.
```python
time_edges = np.arange(T_START, T_END + BIN_SIZE, BIN_SIZE, dtype=np.float32)
# Used for both neural binning and time input
```

iii. No specific justification.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derives lick direction from `bp['R']` and `bp['L']` only, comparing which is greater. It does NOT use hit/miss to determine actual lick direction.

ii.
```python
lick_dir = np.where(bp['R'] > bp['L'], 1, 0).astype(np.int64)
```

iii. No justification. This gives the INSTRUCTED direction (which port was cued), not the actual direction the animal licked.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A simple comparison: if R > L, direction is 1 (right), else 0 (left). Only 2 classes are used (left=0, right=1). The reference has 3 classes (left=0, right=1, no lick=2) and derives actual lick direction from the combination of instructed side and outcome.

ii.
```python
lick_dir = np.where(bp['R'] > bp['L'], 1, 0).astype(np.int64)
```
```python
'output_values': [['left', 'right'], ...]
```

iii. No justification for using only 2 classes or for using instructed direction instead of actual lick direction.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The AI uses `bp['autowater']` to determine behavioral context.

ii.
```python
context = np.where(bp['autowater'] > 0, 0, 1).astype(np.int64)
```

iii. No specific justification. This is the correct source variable.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater > 0 maps to WC (0), else DR (1). This matches the reference.

ii.
```python
context = np.where(bp['autowater'] > 0, 0, 1).astype(np.int64)
```

iii. The AI's CONVERSION_NOTES.md notes "Map WC=autowater, DR=not autowater".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The AI uses only `bp['hit']`. The reference uses both `bp['hit']` and `bp['miss']` to construct 3 classes.

ii.
```python
outcome = np.where(bp['hit'] > 0, 1, 0).astype(np.int64)
```

iii. No justification. The AI maps hit to correct (1) and everything else to incorrect (0), losing the ignore class.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Binary classification: hit > 0 maps to correct (1), else incorrect (0). The reference has 3 classes (incorrect=0, correct=1, ignore=2). Since the AI already filtered out ignore trials, it treats miss and all other trials as incorrect.

ii.
```python
outcome = np.where(bp['hit'] > 0, 1, 0).astype(np.int64)
```
```python
'output_values': [..., ['incorrect', 'correct'], ...]
```

iii. The AI's trial filtering removed ignore trials (bp.no > 0), so only hit and miss trials remain, making the binary encoding somewhat consistent with its own filtering. However, the instructions specify 3 outcome classes.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The AI uses `obj.traj` trajectory data. It identifies tongue features by name from `featNames` (looking for `tongue` in stream 0 and `top_tongue` in stream 1). The `ts` (tracking timeseries) and `frameTimes` are used.

ii.
```python
tongue_idx0 = choose_feature(names0, ['tongue', 'left_tongue', 'right_tongue'])
tongue_idx1 = choose_feature(names1, ['top_tongue', 'bottom_tongue', 'topleft_tongue', 'bottomleft_tongue'])
```

iii. Feature names are selected from the reference code's trajectory feature lists.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI computes speed from raw x,y differences divided by time differences. There is NO likelihood filtering (all frames used regardless of tracking quality), NO Gaussian smoothing of positions, and NO normalization between camera views. Only one camera view is used for tongue (the first one found with valid data, not both). Speed is binned into 75ms bins using `nanbin_mean`.

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

iii. No justification for omitting likelihood filtering or smoothing. The reference filters frames with likelihood <= 0.9 and smooths with a 5ms Gaussian.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Thresholded at the session median (50th percentile) into 2 classes: low (0) and high (1). There is NO "not visible" class (class 2). The reference uses 3 classes including "not visible" for bins where tracking was unavailable.

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

iii. The instructions specify 3 classes (0: < 50th percentile, 1: >= 50th percentile, 2: not visible), but the AI only implements 2.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The go cue time is subtracted from frame times before binning: `tmid - go`. However, the AI does NOT correct for the video-to-behavior clock offset. The reference computes this offset from bitcode signals (`sglx.bitcode.bitstart`).

ii.
```python
go = bp['goCue'][ti]
candidate = nanbin_mean(tmid - go, spd[idx], time_edges)
```

iii. No justification for omitting video clock offset correction.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The AI uses `obj.traj` stream 1 (bottom camera), looking for `top_paw` or `bottom_paw` features.

ii.
```python
paw_idx1 = choose_feature(names1, ['top_paw', 'bottom_paw'])
```

iii. No specific justification.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same as tongue: raw speed from position differences, no likelihood filtering, no smoothing, binned into 75ms bins.

ii.
```python
if paw_idx1 is not None:
    ts, ft = get_stream1(ti)
    spd, tmid = speed_from_ts(ts, ft)
    paw_bin = nanbin_mean(tmid - go, spd[paw_idx1], time_edges)
```

iii. No justification.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: session median threshold, 2 classes (low/high), no "not visible" class.

ii.
```python
paw_bin, paw_thr = threshold_session_bins(paw_vals)
```

iii. Same issue as tongue -- instructions specify 3 classes.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: go cue subtracted from frame times, no video clock offset correction.

ii.
```python
paw_bin = nanbin_mean(tmid - go, spd[paw_idx1], time_edges)
```

iii. No justification.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The AI loads motion energy from separate `motionEnergy_*.mat` files. The loading function handles nested struct formats by unwrapping.

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

iii. The AI correctly handles the nested motion energy file format.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy values are binned into 75ms bins using the side camera's frame times. No additional processing (the motion energy is already computed per frame).

ii.
```python
if ti < len(motion_data):
    me = np.asarray(motion_data[ti]).ravel().astype(np.float32)
    _, ft0 = get_stream0(ti)
    me_bin = nanbin_mean(ft0 - go, me, time_edges) if len(ft0) == len(me) else np.full(...)
```

iii. No specific justification.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same as tongue/paw: session median threshold, 2 classes (low/high), no "no video" class.

ii.
```python
me_bin, me_thr = threshold_session_bins(me_vals)
```

iii. Instructions specify 3 classes (0: < 50th percentile, 1: >= 50th percentile, 2: no video).

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Frame times from stream 0 (side camera) are used with go cue subtraction. No video clock offset correction applied.

ii.
```python
_, ft0 = get_stream0(ti)
me_bin = nanbin_mean(ft0 - go, me, time_edges)
```

iii. No justification for omitting video clock offset.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles some edge cases: motion energy data shorter than trial count results in NaN-filled bins. Zero-neural trials are filtered out. The `nanbin_mean` function handles NaN values. However, there is no handling of DLC likelihood (no tracking quality filtering), no handling of mismatched frame counts between cameras, and no explicit "not visible" class for missing video data.

ii.
```python
if ti < len(motion_data):
    me = np.asarray(motion_data[ti]).ravel().astype(np.float32)
    ...
else:
    me_bin = np.full((len(time_edges)-1,), np.nan, dtype=np.float32)
```
```python
keep_idx = [i for i in keep_idx if not np.all(neural[i] == 0)]
```

iii. No justification for the missing data handling approach.

## 11-a. What are the most time-consuming steps of the code?

i. The AI does not profile or report timing information in the code. Based on structure, the most time-consuming step is likely the nested loop in `bin_unit_spikes_for_trials` which loops over all units and all trials individually, performing a histogram for each unit-trial pair.

ii.
```python
for ui, (ttm, tr) in enumerate(zip(trialtm_list, trial_list)):
    for trial_idx in range(1, n_trials + 1):
        mask = tr == trial_idx
        if np.any(mask):
            counts, _ = np.histogram(ttm[mask], bins=time_edges)
```

iii. No timing analysis documented.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop (`bin_unit_spikes_for_trials`) iterates over every unit and every trial, performing a separate histogram for each. The reference uses `np.histogram2d` to bin all trials of a unit at once. The behavioral binning loop (`nanbin_mean`) iterates over each bin individually using `np.digitize`, which could be vectorized with `np.bincount`.

ii.
```python
# Could be vectorized:
for ui, (ttm, tr) in enumerate(zip(trialtm_list, trial_list)):
    for trial_idx in range(1, n_trials + 1):
        ...

def nanbin_mean(times, values, edges):
    out = np.full((len(edges) - 1,), np.nan, dtype=np.float32)
    idx = np.digitize(times, edges) - 1
    for bi in range(len(out)):
        m = idx == bi
        ...
```

iii. No efficiency analysis documented.

## 11-c. What processing does the code repeat multiple times?

i. The AI calls `get_stream1(ti)` (which loads trajectory data for a trial) twice per trial when both tongue and paw are being processed -- once for the tongue fallback and once for the paw. Similarly `get_stream0(ti)` is called for tongue and again for motion energy.

ii.
```python
for idx, getter in [(tongue_idx1, get_stream1), (tongue_idx0, get_stream0)]:
    ...
    ts, ft = getter(ti)
    ...
if paw_idx1 is not None:
    ts, ft = get_stream1(ti)  # called again
```

iii. No justification.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes speed for ALL tracked features via `speed_from_ts(ts, frame_times)` which processes the full `ts` array (all features x y likelihood x frames), even though only one feature index is used. Behavioral outputs are computed for ALL trials before filtering, meaning outputs for early/ignore trials are computed then discarded.

ii.
```python
spd, tmid = speed_from_ts(ts, ft)  # computes speed for all features
candidate = nanbin_mean(tmid - go, spd[idx], time_edges)  # only uses one feature
```

iii. No justification.
