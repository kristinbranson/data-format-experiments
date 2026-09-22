# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI only loads data from `RandomizedDelay_Ephys_Behavior`, ignoring the `Ephys_Behavior` folder entirely. Sessions are discovered by globbing `data_structure_*.mat` in that one folder, rather than using a hard-coded session list. Each file is opened with either `h5py` (for v7.3 files) or `scipy.io.loadmat` (for v5 files), selected by trying `h5py` first and falling back on `scipy.io`.

ii.
```python
BASE = Path('/app/data/RandomizedDelay_Ephys_Behavior')
...
def convert(sample=False):
    files = sorted(BASE.glob('data_structure_*.mat'))
    if sample:
        files = files[:2]
    sessions = [load_session(f) for f in files]
```

iii. The CONVERSION_NOTES.md only mentions data in `RandomizedDelay_Ephys_Behavior`. The AI did not discover or process the `Ephys_Behavior` folder, which contains 25 additional fixed-delay sessions. The AI noted the paper mentions "19 sessions using four mice" for the randomized delay task, but did not recognize the paper also describes a fixed-delay task with 25 additional sessions.

## 1-b. How are the data split into subjects?

i. The subject is extracted from the filename by splitting on underscores and taking the third element (index 2): `path.stem.split('_')[2]`. Since filenames are `data_structure_<subject>_<date>.mat`, index 2 corresponds to the subject name. The subjects list is the sorted unique set across sessions.

ii.
```python
'subject': path.stem.split('_')[2],
...
subjects = sorted({s['subject'] for s in sessions})
subject_map = {s: i for i, s in enumerate(subjects)}
```

iii. No explicit justification given; this follows from the filename convention.

## 1-c. How are the data split into sessions?

i. Each `.mat` file in the `RandomizedDelay_Ephys_Behavior` folder is one session. Sessions without neural data (`clu`) are excluded, and sessions with fewer than 10 curated units are excluded. This yields ~20 sessions from 4 subjects, all from the randomized-delay task only.

ii.
```python
sessions = [s for s in sessions if s['has_clu']]
sessions = [s for s in sessions if len(select_units(s)) >= 10]
```

iii. The AI noted in CONVERSION_NOTES.md that 2 of 22 files lacked `clu` and were excluded. The >=10 units threshold is mentioned in the paper's methods. However, the AI missed that the reference also includes 25 fixed-delay sessions from `Ephys_Behavior`.

## 1-d. How are the data split into trials?

i. All trials from 1 to `session['ntrials']` are included without any filtering. The trial loop iterates over all trial indices.

ii.
```python
for tr in range(1, session['ntrials'] + 1):
    arr = np.zeros((len(units), len(centers)), dtype=np.float32)
    for i, u in enumerate(units):
        mask = (u['trial'] == tr)
```

iii. No justification for including all trials. The AI's CONVERSION_NOTES.md mentions "early lick and ignore trials were omitted from many paper analyses" but the code does not implement any trial filtering.

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filtering is applied. Early-lick trials and photostimulation trials are not excluded. The `early` and `stim.enable` fields are loaded but never used for filtering. All trials (1 through `Ntrials`) are processed.

ii.
```python
# early is loaded but never used for filtering:
'early': np.array(bp['early'][()]).squeeze().astype(bool) if 'early' in bp else np.zeros(ntrials, dtype=bool),
# No filtering code exists - all trials from 1 to ntrials are processed
for tr in range(1, session['ntrials'] + 1):
```

iii. The AI noted in CONVERSION_NOTES.md Step 3 that "Behavioral analyses in the paper excluded early-lick and ignore trials from many analyses" and in Step 5 listed "Early trials: Investigate whether early trials should be excluded" as a key decision. However, this was never implemented.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `obj.clu` cluster fields: `trial` (which trial each spike belongs to), `trialtm` (spike time relative to trial start), and `quality` (curation label). The go cue times from `obj.bp.ev.goCue` are loaded but NOT used for alignment.

ii.
```python
units.append({'quality': q, 'tm': tm, 'trial': trial, 'trialtm': trialtm, 'site': site})
```

iii. The AI correctly identified the source variables but failed to use `goCue` for alignment.

## 2-b. How is the `neural` data processed?

i. Spike times (`trialtm`) are histogrammed into 20 ms bins from -2.4 to 2.0 s (220 bins), divided by `dt` to get firing rates in Hz, then smoothed with a Gaussian of sigma = 2.0 bins (40 ms). Critically, `trialtm` is used directly without subtracting the go cue time, so the data is aligned to trial start, not go cue onset.

ii.
```python
def build_neural_trials(session, units, tmin=-2.4, tmax=2.0, dt=0.02, smooth_sigma=2.0):
    edges = np.arange(tmin, tmax + dt, dt)
    centers = edges[:-1] + dt / 2
    neural_trials = []
    for tr in range(1, session['ntrials'] + 1):
        arr = np.zeros((len(units), len(centers)), dtype=np.float32)
        for i, u in enumerate(units):
            mask = (u['trial'] == tr)
            if np.any(mask):
                counts, _ = np.histogram(u['trialtm'][mask], bins=edges)
                arr[i] = gaussian_smooth(counts.astype(np.float32) / dt, smooth_sigma)
        neural_trials.append(arr)
    return neural_trials, centers
```

iii. The AI's CONVERSION_NOTES.md states it uses "goCue alignment" but the code does not subtract `goCue` from `trialtm`. The bin size of 20 ms and the time window of [-2.4, 2.0] differ from the reference's 5 ms bins and [-2.5, 2.5] window.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept if their quality label (normalized to lowercase) is in the set `{'multi', 'fair', 'good', 'great', 'excellent'}` AND their firing rate exceeds 1 Hz. The firing rate is calculated as `tm.size / (tm.max() - tm.min())`, which is the number of spikes divided by the time span of the `tm` array (not the mean rate over the trial window).

ii.
```python
KEEP_QUALITY = {'multi', 'fair', 'good', 'great', 'excellent'}

def normq(q):
    q = str(q).strip().lower().replace('\x00', '')
    if q == 'mutli':
        q = 'multi'
    return q

def unit_fr_gt1(unit):
    tm = unit['tm']
    if tm.size < 2:
        return False
    dur = float(tm.max() - tm.min())
    if dur <= 0:
        return False
    return (tm.size / dur) > 1.0

def select_units(session):
    keep = [u for u in session['units'] if u['quality'] in KEEP_QUALITY and unit_fr_gt1(u)]
    return keep
```

iii. The AI noted the paper states "all units with firing rates exceeding 1 Hz were included" and that quality filtering is needed. The keep-list approach differs from the reference's drop-list but aims to achieve similar results (874 vs 845 units). The firing rate calculation differs from the reference (which uses mean rate over the analysis window).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data is NOT aligned to the go cue. `trialtm` (time relative to trial start) is used directly without subtracting `goCue`. The go cue times are loaded into the session dict but never subtracted from spike times.

ii.
```python
# In build_neural_trials:
counts, _ = np.histogram(u['trialtm'][mask], bins=edges)
# goCue is loaded but never used:
'goCue': np.array(bp['ev']['goCue'][()]).squeeze().astype(float),
```

iii. The CONVERSION_NOTES.md claims "goCue alignment" is used, but the code does not implement it.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 20 ms (`dt=0.02`), with a window from -2.4 to 2.0 s, yielding 220 bins. The metadata reports `time_bin_size: 20.0` ms.

ii.
```python
def build_neural_trials(session, units, tmin=-2.4, tmax=2.0, dt=0.02, smooth_sigma=2.0):
    edges = np.arange(tmin, tmax + dt, dt)
```

```python
'time_bin_size': 20.0,
'off_start': -2.4,
'off_end': 2.0,
```

iii. No explicit justification given for the 20 ms bin size. The reference uses 5 ms bins (`params.dt = 1/200`) and a [-2.5, 2.5] window (1000 bins).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is defined as the bin centers of the neural time grid, computed from the binning parameters. It is not derived from any raw data variable.

ii.
```python
def build_inputs(centers, ntrials):
    arr = centers.astype(np.float32)[None, :]
    return [arr.copy() for _ in range(ntrials)]
```

iii. This is a constructed time axis, as expected.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The centers are computed as `edges[:-1] + dt/2` where edges span -2.4 to 2.0 in steps of 0.02. The result is the midpoint of each 20 ms bin.

ii.
```python
edges = np.arange(tmin, tmax + dt, dt)
centers = edges[:-1] + dt / 2
```

iii. However, since the neural data is not actually aligned to the go cue (trialtm is relative to trial start), this time axis does not truly represent "time from go cue onset."

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The same bin centers are used for both neural data and the input, so they share the same time axis by construction.

ii.
```python
neural_trials, centers = build_neural_trials(sess, units)
inputs = build_inputs(centers, sess['ntrials'])
```

iii. While the input and neural arrays are aligned with each other, neither is actually aligned to the go cue due to the missing goCue subtraction.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `obj.bp.L` and `obj.bp.R`, the instructed side fields, not from the actual lick direction.

ii.
```python
lick = np.full(ntr, 2, dtype=np.int64)
lick[session['L']] = 0
lick[session['R']] = 1
```

iii. No justification given. The AI appears to have interpreted `L` and `R` as the animal's lick direction rather than the instructed reward side.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Trials where `L` is true are labeled 0 (left), trials where `R` is true are labeled 1 (right), and remaining trials default to 2 (none). This encodes the instructed side, not the actual lick direction. The reference derives actual lick direction from the combination of instructed side and outcome (hit = licked the instructed side, miss = licked the opposite side).

ii.
```python
lick = np.full(ntr, 2, dtype=np.int64)
lick[session['L']] = 0
lick[session['R']] = 1
```

iii. This will produce incorrect lick direction labels for miss trials (where the animal licked the opposite of the instructed side).

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derived from `obj.bp.autowater`. Trials with `autowater=True` are WC (water-cued, class 0), others are DR (delayed-response, class 1).

ii.
```python
context = np.where(session['autowater'], 0, 1).astype(np.int64)  # WC, DR
```

iii. This matches the reference approach.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct mapping: autowater -> WC (0), not autowater -> DR (1). This is correct.

ii.
```python
context = np.where(session['autowater'], 0, 1).astype(np.int64)
```

iii. Matches the reference.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `obj.bp.hit`, `obj.bp.miss`, and `obj.bp.no`.

ii.
```python
outcome = np.full(ntr, 2, dtype=np.int64)
outcome[session['miss']] = 0
outcome[session['hit']] = 1
outcome[session['no']] = 2
```

iii. This is consistent with the reference (incorrect=0, correct=1, ignore=2).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. `miss` -> incorrect (0), `hit` -> correct (1), `no` -> ignore (2). The default fill is 2 (ignore), and then miss and hit override. This matches the reference's encoding.

ii.
```python
outcome = np.full(ntr, 2, dtype=np.int64)
outcome[session['miss']] = 0
outcome[session['hit']] = 1
outcome[session['no']] = 2
```

iii. Matches the reference, except that early-lick trials (which have their own outcome behavior) are not excluded.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from `obj.traj` DLC tracking data. The AI uses `speed_from_ts` which extracts x/y coordinates and likelihood for features named `tongue`, `left_tongue`, or `right_tongue` from the first camera view (view 0 / side camera only).

ii.
```python
feat0, ts0, ft0 = extract_h5_traj_view(h, view0, tr)
spd0, valid0 = speed_from_ts(ts0, feat0, ['tongue','left_tongue','right_tongue'])
```

iii. Only the side camera view is used for tongue, unlike the reference which uses both side (`tongue`) and bottom (`top_tongue`) camera views and averages them after normalization.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Speed is computed as `sqrt(dx^2 + dy^2)` using `np.diff` on x and y coordinates, with a likelihood threshold of 0.5 (vs reference's 0.9). No Gaussian smoothing of x/y positions is applied before differentiation. NaN values are filled with the median before resampling. The speed is then resampled to neural time bins using linear interpolation (`np.interp`) rather than proper temporal binning. Finally, discretized at the session median.

ii.
```python
def speed_from_ts(ts, feat_names, feature_candidates, like_thresh=0.5):
    ...
    dx = np.diff(xy[0], prepend=np.nan)
    dy = np.diff(xy[1], prepend=np.nan)
    spd = np.sqrt(dx**2 + dy**2)
    spd[~valid] = np.nan
    return spd, valid
```

```python
rr = resample_trace_to_bins(np.nan_to_num(spd0, nan=np.nanmedian(spd0[np.isfinite(spd0)]) if np.isfinite(spd0).any() else 0.0), ntime)
```

iii. Multiple differences from reference: no x/y smoothing, lower likelihood threshold (0.5 vs 0.9), `np.diff` instead of `np.gradient`, NaN-filling with median, linear interpolation resampling instead of mean-within-bin, and only one camera view instead of two.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Discretized at the session median (50th percentile) of valid speed values. Below median = 0, above/equal = 1, not visible = 2.

ii.
```python
if len(tongue_vals) > 0:
    med = np.nanmedian(np.asarray(tongue_vals, dtype=float))
    for tr, item in enumerate(tongue_res):
        if item is not None:
            rr, valid = item
            tongue_out[tr, valid] = (rr[valid] >= med).astype(np.int64)
```

iii. The median split matches the reference's 50th percentile split, though applied to differently computed values.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame-level speed is resampled to neural bins using linear interpolation (`np.interp` via `resample_trace_to_bins`), mapping the old indices to new indices on a 0-to-1 normalized scale. There is no temporal alignment using the video offset (bitcode correction) or go cue subtraction.

ii.
```python
def resample_trace_to_bins(trace, ntime):
    trace = np.asarray(trace, dtype=float).squeeze()
    x_old = np.linspace(0.0, 1.0, trace.size)
    x_new = np.linspace(0.0, 1.0, ntime)
    return np.interp(x_new, x_old, trace).astype(np.float32)
```

iii. This resampling assumes the frame-level data covers the same time window as the neural bins, which is only approximately true and does not account for the video clock offset. The reference computes the offset from bitcode pulses and properly bins frames into 5 ms bins.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from `obj.traj` DLC tracking, looking for `top_paw` or `bottom_paw` features in the second camera view (view 1 / bottom camera).

ii.
```python
feat1, ts1, ft1 = extract_h5_traj_view(h, view1, tr)
spd1, valid1 = speed_from_ts(ts1, feat1, ['top_paw','bottom_paw'])
```

iii. The reference only uses `top_paw` from the bottom camera; the AI falls back to `bottom_paw` if `top_paw` is not found.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same as tongue velocity: speed from `np.diff` of x/y, no smoothing, likelihood threshold of 0.5, NaN-filling with median, resampling via interpolation, and discretization at session median.

ii.
```python
spd1, valid1 = speed_from_ts(ts1, feat1, ['top_paw','bottom_paw'])
if spd1 is not None:
    rr = resample_trace_to_bins(np.nan_to_num(spd1, ...), ntime)
```

iii. Same issues as tongue velocity processing.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: session median split. Below median = 0, above/equal = 1, not visible = 2.

ii.
```python
if len(paw_vals) > 0:
    med = np.nanmedian(np.asarray(paw_vals, dtype=float))
    for tr, item in enumerate(paw_res):
        if item is not None:
            rr, valid = item
            paw_out[tr, valid] = (rr[valid] >= med).astype(np.int64)
```

iii. Matches the reference's 50th percentile split conceptually.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: resampled via linear interpolation without video clock offset correction.

ii. Same `resample_trace_to_bins` function.

iii. Same issues as tongue velocity alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from sidecar files `motionEnergy_<subject>_<date>.mat`, reading the `me` struct's `data` field. Only files in `RandomizedDelay_Ephys_Behavior` are checked.

ii.
```python
def load_motion_energy_sidecar(session_name):
    f = BASE / f'motionEnergy_{session_name}.mat'
    if not f.exists():
        return None, None
    try:
        me = sio.loadmat(str(f), squeeze_me=True, struct_as_record=False)['me']
        if hasattr(me, 'data') and hasattr(me, 'moveThresh'):
            return np.array(me.data, dtype=object), float(me.moveThresh)
```

iii. The reference also loads from sidecar files, with similar unwrapping logic.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Per-trial motion energy traces are resampled to neural time bins using linear interpolation (`resample_trace_to_bins`), then discretized at the session median.

ii.
```python
for trc in trial_traces:
    rr = resample_trace_to_bins(trc, ntime)
    if rr is None:
        resampled.append(None)
    else:
        resampled.append(rr)
        session_vals.extend(rr.tolist())
if len(session_vals) > 0:
    med = np.nanmedian(np.asarray(session_vals, dtype=float))
    for i, rr in enumerate(resampled):
        if rr is not None:
            motion[i] = (rr >= med).astype(np.int64)
```

iii. The reference bins frame-level values into 5 ms bins by averaging within each bin. The AI uses linear interpolation resampling instead.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Session median split: below median = 0, above/equal median = 1, no video = 2.

ii.
```python
med = np.nanmedian(np.asarray(session_vals, dtype=float))
motion[i] = (rr >= med).astype(np.int64)
```

iii. Matches the reference's 50th percentile conceptually.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Resampled via linear interpolation from the raw frame count to the number of neural time bins, without any temporal alignment using frame times or video offset.

ii.
```python
rr = resample_trace_to_bins(trc, ntime)
```

iii. No video offset correction or proper temporal binning. The reference uses the side camera frame times, corrected by the bitcode offset and go cue, and averages values within each 5 ms bin.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The code uses broad `try/except Exception: pass` blocks around video processing, so any errors during tongue/paw extraction are silently swallowed and the trial defaults to class 2 (not visible). Sessions without `clu` are excluded. Sessions with fewer than 10 units are excluded. Missing motion energy sidecar files return `None`, and those sessions get all class 2.

ii.
```python
try:
    t_out, p_out = build_video_outputs_h5(session, ntime)
    if t_out is not None:
        tongue = t_out
except Exception:
    pass
```

iii. The broad exception handling is concerning as it could mask real bugs. The reference handles specific cases (missing frame times, low likelihood) explicitly rather than catching all exceptions.

## 11-a. What are the most time-consuming steps of the code?

i. The per-trial, per-unit spike histogram loop in `build_neural_trials` iterates over every unit and every trial sequentially, making it O(n_units * n_trials). Loading files and HDF5 video extraction are also slow steps.

ii.
```python
for tr in range(1, session['ntrials'] + 1):
    arr = np.zeros((len(units), len(centers)), dtype=np.float32)
    for i, u in enumerate(units):
        mask = (u['trial'] == tr)
        if np.any(mask):
            counts, _ = np.histogram(u['trialtm'][mask], bins=edges)
```

iii. The reference avoids this double loop by using `np.histogram2d` to bin all trials at once per cluster.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning has nested loops over trials and units. The reference vectorizes this with `histogram2d(spike_trial, spike_time, bins=[trial_edges, BIN_EDGES])` to bin all trials of a cluster in one call. The per-trial video extraction loops could also benefit from batch processing.

ii.
```python
# AI's nested loop:
for tr in range(1, session['ntrials'] + 1):
    for i, u in enumerate(units):
        mask = (u['trial'] == tr)
        counts, _ = np.histogram(u['trialtm'][mask], bins=edges)

# Reference's vectorized approach:
counts, _, _ = np.histogram2d(spike_trial, spike_time, bins=[trial_edges, BIN_EDGES])
```

iii. The nested loop is the biggest efficiency issue.

## 11-c. What processing does the code repeat multiple times?

i. The video data file is re-opened for the video output extraction (`build_video_outputs_h5` opens the same file that `load_session_h5` already opened). The `_traj` extraction per trial is done separately for tongue and paw.

ii.
```python
# load_session opens the file once:
with h5py.File(path, 'r') as h:
    return load_session_h5(path, h)
# build_video_outputs_h5 re-opens it:
with h5py.File(BASE / f"data_structure_{session['session_name']}.mat", 'r') as h:
```

iii. The double file open is a clear redundancy.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads many fields that are never used: `sample`, `reward`, `site`, `tm` (spike times on the recording clock), `traj` and `me_obj` from the v5 loader, and `has_clu`. The `stim` field from `bp.stim.enable` is not loaded at all, and `early` is loaded but not used for filtering. The motion energy `moveThresh` is loaded but not used (the code computes its own median threshold).

ii.
```python
'sample': np.array(bp['ev']['sample'][()]).squeeze().astype(float) if 'sample' in bp['ev'] else None,
'reward': np.array(bp['ev']['reward'][()]).squeeze().astype(float) if 'reward' in bp['ev'] else None,
'site': int(np.array(h[clu['site'][i, 0]][()]).squeeze()) if 'site' in clu else -1,
```

iii. These extra fields add loading time but are never used in the conversion.
