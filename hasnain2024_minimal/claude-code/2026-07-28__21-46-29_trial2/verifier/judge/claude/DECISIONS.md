# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from `.mat` files located in `data/Ephys_Behavior/` and `data/RandomizedDelay_Ephys_Behavior/` directories. It auto-detects whether each file is HDF5 (MATLAB v7.3) or v5 format and uses the appropriate loader (`h5py` or `scipy.io.loadmat`). A hardcoded list `ALL_SESSIONS` defines all 44 sessions with animal name, date, directory, and probe numbers. Each session's data structure file is named `data_structure_{anm}_{date}.mat`. Motion energy is loaded from separate `motionEnergy_{anm}_{date}.mat` files.

ii.
```python
ALL_SESSIONS = [
    {'dir': EB, 'anm': 'EKH1', 'date': '2021-08-07', 'probes': [2]},
    # ... 44 sessions total
]

data_path = os.path.join(data_dir, sess['dir'], f"data_structure_{sname}.mat")
me_path = os.path.join(data_dir, sess['dir'], f"motionEnergy_{sname}.mat")

if _is_hdf5(data_path):
    sd = _load_session_h5(data_path, sess)
else:
    sd = _load_session_v5(data_path, sess)
```

iii. From CONVERSION_NOTES.md: "Sessions were identified from the `DataLoadingScripts/Recording and video/` MATLAB loading scripts. Each `loadANM_ALMVideo.m` file specifies which sessions and probes to use for each animal." The AI identified 44 sessions (25 Ephys_Behavior, 19 RandomizedDelay) across 14 mice.

## 1-b. How are the data split into subjects (mice)?

i. Subjects (mice) are identified by the `anm` field in the session definitions. As sessions are processed, unique animal names are accumulated in a dictionary mapping name to index. The final `subjects` list preserves the order of first encounter.

ii.
```python
anm = result['anm']
if anm not in subjects:
    subjects[anm] = len(subjects)
all_subject_idx.append(subjects[anm])
```

iii. The AI used the `anm` field from each session definition, which was derived from the reference loading scripts (e.g., `loadJEB13_ALMVideo.m`).

## 1-c. How are the data split into sessions?

i. Each entry in `ALL_SESSIONS` represents one recording session, identified by animal + date + directory. Each session is processed independently by `process_session()`, producing separate neural, input, and output lists. Sessions with fewer than `MIN_TRIALS=2` usable trials or fewer than `MIN_UNITS=10` neurons are skipped.

ii.
```python
for sess in sessions:
    result, err = process_session(sess, data_dir)
    if result is None:
        print(f"  SKIPPED: {err}")
        continue
    all_neural.append(result['neural'])
```

iii. From CONVERSION_NOTES.md, sessions were identified from the loading scripts. Excluded sessions include those with missing data files (JEB4, JEB5) and commented-out sessions (JEB23 2023-10-20).

## 1-d. How are the data split into trials?

i. Each session contains `n_trials` (from `bp.Ntrials`). Trials are indexed 1-based in the MATLAB data. The AI extracts behavioral markers (hit, miss, early, no_resp, L, R, autowater, stim_en) for all trials, then filters to usable trials.

ii.
```python
n_trials = sd['n_trials']
# ... extract hit, miss, early, etc.
use = ((hit == 1) | (miss == 1)) & (stim_en == 0) & (early == 0)
trial_idx = np.where(use)[0]
```

iii. Trials correspond to individual behavioral trials within a session. The `bp.Ntrials` field gives the total count.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials using the condition `(hit|miss) & ~stim.enable & ~early`. This includes both correct (hit) and incorrect (miss) trials, excludes stimulation trials and early-lick trials. No-response ("ignore") trials are also excluded since they are neither hit nor miss. Both DR (autowater=0) and WC (autowater=1) trials are included.

ii.
```python
use = ((hit == 1) | (miss == 1)) & (stim_en == 0) & (early == 0)
trial_idx = np.where(use)[0]
if len(trial_idx) < MIN_TRIALS:
    return None, f"Only {len(trial_idx)} usable trials"
```

iii. From CONVERSION_NOTES.md: "Matches `findTrials.m` condition `'(hit|miss)&~stim.enable&~early'`". The AI also requires at least 2 trials per session for the decoder.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from sorted spike times in `obj.clu{probe}(cluster).trialtm` (spike time within trial) and `obj.clu{probe}(cluster).trial` (trial assignment). Go cue times from `obj.bp.ev.goCue` are used for alignment.

ii.
```python
spike_data.append({
    'trial': probe_arr['trial'][0, ci].flatten().astype(float),
    'trialtm': probe_arr['trialtm'][0, ci].flatten().astype(float),
})
# ...
aligned = spk['trialtm'][mask] - goCue[ti]
counts, _ = np.histogram(aligned, bins=EDGES)
```

iii. The AI references the standard data structure where `obj.clu` contains sorted spike data with `trialtm` and `trial` fields.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to go cue by subtracting `goCue[ti]` from `trialtm`. Aligned spikes are binned into 10ms bins from -2.5s to +2.5s (500 bins). Counts are converted to firing rates by dividing by dt (0.01s). Firing rates are smoothed with a causal Gaussian kernel (window=15, reflect boundary condition).

ii.
```python
aligned = spk['trialtm'][mask] - goCue[ti]
counts, _ = np.histogram(aligned, bins=EDGES)
fr[ni, :, ti] = causal_smooth(counts.astype(np.float64) / DT)
```

The smoothing function:
```python
def causal_smooth(x, N=SMOOTH_N, bctype=BC_TYPE):
    kern = gausswin(N)
    kern[:N // 2] = 0   # causal: zero out first half
    kern /= kern.sum()
    if bctype == 'reflect':
        x_padded = np.concatenate([x[:N], x])
        trim = N
    out = np.convolve(x_padded, kern, mode='same')
    return out[trim:]
```

iii. From CONVERSION_NOTES.md: "Matches `getSeq.m` and `mySmooth.m`". Parameters match `WorkingWithDataObjs.m`: dt=1/100, smooth=15, bctype='reflect'.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) Cluster quality filter: excludes clusters with quality labels 'garbage', 'gabrga', 'noisy', 'real?', and empty strings. (2) Low firing rate filter: removes neurons with mean firing rate <= 1 Hz, computed as the mean across all time bins and all trials. Sessions with fewer than 10 neurons after filtering are skipped.

ii.
```python
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
valid_clu = [i for i, q in enumerate(qualities)
             if q.lower() not in EXCLUDED_QUALITIES and q != '']
# ...
mean_fr = np.mean(fr, axis=(1, 2))
keep = mean_fr > LOW_FR
fr = fr[keep]
```

iii. From CONVERSION_NOTES.md: "Matches `findClusters.m` with `'all'` quality parameter" and "Matches `removeLowFRClusters.m`".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue by subtracting the per-trial go cue time (`goCue[ti]`) from each spike's time-in-trial (`trialtm`). The aligned spikes are then binned into the time axis centered on the go cue.

ii.
```python
aligned = spk['trialtm'][mask] - goCue[ti]
counts, _ = np.histogram(aligned, bins=EDGES)
```

iii. This matches `alignSpikes.m` which computes `trialtm_aligned = trialtm - event` where event is `goCue`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 10ms (DT = 0.01s), matching `params.dt = 1/100` in `WorkingWithDataObjs.m`. The time axis spans -2.5s to +2.5s (500 bins). No rebinning is applied; spike data is directly binned at this resolution.

ii.
```python
DT = 0.01           # 10ms bins (params.dt = 1/100)
TMIN = -2.5
TMAX = 2.5
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
```

iii. The AI explicitly references `WorkingWithDataObjs.m` for the `params.dt = 1/100` parameter.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not directly derived from a raw data variable. It is the time axis itself (bin centers), representing the time from the go cue onset in seconds.

ii.
```python
TIME_AXIS = EDGES[:-1] + DT / 2  # bin centers
# ...
input_list.append(TIME_AXIS[np.newaxis, :].astype(np.float32).copy())  # (1, T)
```

iii. The time axis is constructed from the bin edges, with each value representing the center of a time bin relative to the go cue onset.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is computed as bin centers: `EDGES[:-1] + DT/2`, giving values from -2.495s to +2.495s in 10ms steps. The same time axis is used for all trials and sessions.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
if len(EDGES) > 501:
    EDGES = EDGES[:501]
TIME_AXIS = EDGES[:-1] + DT / 2
```

iii. This matches the reference `getSeq.m` which computes `obj.time = edges + params.dt/2; obj.time = obj.time(1:end-1)`.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time axis is identically the bin centers of the neural data, so alignment is automatic. Both neural data and input share the exact same temporal grid.

ii.
```python
neural_list.append(fr[:, :, ti].copy())                     # (n_neurons, T)
input_list.append(TIME_AXIS[np.newaxis, :].astype(np.float32).copy())  # (1, T)
```

iii. The AI uses the same TIME_AXIS for constructing both the neural data bins and the input.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `obj.bp.R` (right lick indicator). R=1 means right, R=0 means left.

ii.
```python
R = bp['R'][0, 0].flatten()[:n_trials].astype(int)
# ...
out[0, :] = int(R[ti])  # lick direction: L=0, R=1
```

iii. The `R` field from `obj.bp` is a binary indicator for right-choice trials. Left trials have R=0.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The per-trial R value is broadcast across all time bins as a constant per-trial value. Left=0, Right=1.

ii.
```python
out[0, :] = int(R[ti])  # lick direction: L=0, R=1
```

iii. No transformation is needed; the raw binary indicator directly maps to the required encoding.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj.bp.autowater`. When autowater=1, the trial is water-cued (WC); when autowater=0, it is delayed-response (DR).

ii.
```python
autowater = bp['autowater'][0, 0].flatten()[:n_trials].astype(int)
# ...
out[1, :] = 1 - int(autowater[ti])  # context: WC=0, DR=1
```

iii. From `WorkingWithDataObjs.m`: "obj.bp.autowater=1 when water was delivered regardless of animal choice (0 otherwise). this field can be used as a proxy for obtaining water-cued blocks and delayed-response blocks of trials".

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The autowater value is inverted (1 - autowater) so that WC=0 and DR=1, matching the instructions. The value is broadcast across all time bins.

ii.
```python
out[1, :] = 1 - int(autowater[ti])  # context: WC=0, DR=1
```

iii. The instructions specify WC=0, DR=1. Since autowater=1 for WC trials, the inversion correctly maps the encoding.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit`. hit=1 indicates a correct trial, hit=0 (which for included trials means miss=1) indicates an incorrect trial.

ii.
```python
hit = bp['hit'][0, 0].flatten()[:n_trials].astype(int)
# ...
out[2, :] = int(hit[ti])  # outcome: incorrect=0, correct=1
```

iii. Since only hit and miss trials are included, hit=1 corresponds to correct and hit=0 to incorrect (miss).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The per-trial hit value is directly used as the outcome (0=incorrect, 1=correct) and broadcast across all time bins.

ii.
```python
out[2, :] = int(hit[ti])  # outcome: incorrect=0, correct=1
```

iii. No additional processing is needed.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the bottom camera DLC tracking data: `obj.traj{2}(trial).ts` for the `top_tongue` feature (x, y coordinates and confidence), and `obj.traj{2}(trial).frameTimes` for temporal alignment.

ii.
```python
tongue_fi = feat_names.index('top_tongue') if 'top_tongue' in feat_names else 0
# v5 format: ts[:, 0, tongue_fi], ts[:, 1, tongue_fi] (x, y)
# h5 format: ts[tongue_fi, 0], ts[tongue_fi, 1]
t_spd = compute_speed(x, y, ft, confidence, DLC_CONF, fill_missing=True)
```

iii. The AI identifies the `top_tongue` feature from the bottom camera's feature names, matching the reference code's `params.traj_features` which includes `'top_tongue'` for the bottom camera.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI: (1) extracts x, y, and confidence for `top_tongue`; (2) marks frames with NaN positions or confidence < 0.9 as invalid; (3) fills invalid positions with nearest valid position; (4) computes instantaneous speed as `sqrt(dx/dt^2 + dy/dt^2)` using finite differences; (5) aligns to go cue using video offset; (6) interpolates speed to neural time axis; (7) fills remaining NaN with nearest valid value.

ii.
```python
def compute_speed(x, y, ft, confidence=None, conf_thresh=DLC_CONF, fill_missing=True):
    valid = ~np.isnan(x) & ~np.isnan(y)
    if confidence is not None:
        valid = valid & (confidence >= conf_thresh)
    if np.any(valid) and not np.all(valid):
        x, y = fill_positions_nearest(x, y, valid)
    dt_vid = np.diff(ft)
    vx = np.diff(x) / dt_vid
    vy = np.diff(y) / dt_vid
    speed = np.sqrt(vx**2 + vy**2)
    speed = np.concatenate([[0.0], speed])
    return speed

# Alignment and interpolation
aln = ft - vidshift - goCue[ti]
interp_fn = interp1d(aln, t_spd, kind='linear', bounds_error=False, fill_value=np.nan)
tongue_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. CONVERSION_NOTES.md: "Fill invalid positions (NaN or low confidence < 0.9) with nearest valid position. Compute instantaneous speed: sqrt(dx/dt^2 + dy/dt^2). Align to go cue using video offset. Interpolate to neural time axis."

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI computes the 50th percentile threshold on **non-zero** tongue velocity values from usable trials only. Time bins with velocity >= threshold get value 1 ("high"), below get 0 ("low").

ii.
```python
tongue_usable = tongue_vel[:, trial_idx].flatten()
tongue_nonzero = tongue_usable[tongue_usable > 0]
if len(tongue_nonzero) > 0:
    tongue_thresh = np.percentile(tongue_nonzero, 50)
else:
    tongue_thresh = 1.0
out[3, :] = (tongue_vel[:, ti] >= tongue_thresh).astype(np.int64)
```

iii. CONVERSION_NOTES.md: "Tongue is only visible during licking (~4-8% of frames). After filling invalid positions with nearest valid, velocity is near-zero when tongue is not visible. The 50th percentile threshold is computed on non-zero values only."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The tongue speed (computed in video frame time) is aligned by subtracting the video offset and go cue time from the frame times, then linearly interpolating to the neural time axis (TIME_AXIS). Remaining NaN values are filled with nearest valid values.

ii.
```python
aln = ft - vidshift - goCue[ti]
interp_fn = interp1d(aln, t_spd, kind='linear', bounds_error=False, fill_value=np.nan)
tongue_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The video offset (`vidshift`) is computed from `mode(sglx.bitcode.bitstart) / sglx.fs - mode(bp.ev.bitStart)`, matching `findVideoOffset.m`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom camera DLC tracking data for the `top_paw` or `bottom_paw` feature (x, y coordinates and confidence), plus frame times.

ii.
```python
for pname in ['top_paw', 'bottom_paw']:
    if pname in feat_names:
        paw_fi = feat_names.index(pname)
        break
```

iii. The AI checks for `top_paw` first, then `bottom_paw` as fallback, consistent with the reference `params.traj_features` which lists both.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same pipeline as tongue velocity: extract x/y/confidence, fill invalid positions, compute speed via finite differences, align to go cue, interpolate to neural time axis, fill remaining NaN.

ii.
```python
p_spd = compute_speed(ts_p[:, 0, paw_fi], ts_p[:, 1, paw_fi], ft_p,
                      ts_p[:, 2, paw_fi], DLC_CONF, fill_missing=True)
interp_fn = interp1d(aln, p_spd, kind='linear', bounds_error=False, fill_value=np.nan)
paw_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The same `compute_speed` function is used for both tongue and paw, applying the same confidence threshold and position filling.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The 50th percentile of all paw velocity values across usable trials is used as the threshold. Values >= threshold are "high" (1), below are "low" (0).

ii.
```python
paw_thresh = np.percentile(paw_vel[:, trial_idx], 50)
out[4, :] = (paw_vel[:, ti] >= paw_thresh).astype(np.int64)
```

iii. This follows the instructions for per-session 50th percentile thresholding.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same alignment as tongue: paw speed in video frame time is aligned using video offset and go cue times, then interpolated to the neural time axis.

ii.
```python
aln = ft - vidshift - goCue[ti]
interp_fn = interp1d(aln, p_spd, kind='linear', bounds_error=False, fill_value=np.nan)
paw_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. Uses the same video offset and interpolation approach as tongue velocity.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from separate `motionEnergy_{anm}_{date}.mat` files. The data is in `me.data` (per-trial cell array). Side camera frame times (`obj.traj{1}(trial).frameTimes`) are used for alignment.

ii.
```python
me_path = os.path.join(data_dir, sess['dir'], f"motionEnergy_{sname}.mat")
me_mat = scipy.io.loadmat(me_path, squeeze_me=False)
me_struct = me_mat['me']
me_trials = me_struct['data'][0, 0]
```

iii. CONVERSION_NOTES.md: "Loaded from separate `motionEnergy_*.mat` files (always MATLAB v5 format). Aligned using side camera frame times and video offset."

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Per-trial motion energy values are aligned to the go cue using side camera frame times (with video offset), linearly interpolated to the neural time axis, and NaN values filled with nearest valid values. When side camera frame times are unavailable, synthetic frame times at 400 Hz with a 0.5s offset are used.

ii.
```python
for ti in range(min(n_trials, me_trials.shape[0])):
    trial_me = me_trials[ti, 0].flatten().astype(np.float64)
    if ti in side_ft:
        aln = side_ft[ti] - vidshift - goCue[ti]
        if len(trial_me) != len(aln):
            aln = np.arange(len(trial_me)) / 400.0 - 0.5 - goCue[ti]
    else:
        aln = np.arange(len(trial_me)) / 400.0 - 0.5 - goCue[ti]
    interp_fn = interp1d(aln, trial_me, kind='linear', bounds_error=False, fill_value=np.nan)
    me_data[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. This matches `loadMotionEnergy.m` which uses `interp1(frameTimes-vidshift-alignTimes(trix), me.data{trix}, taxis)` and falls back to `(1:nFrames)/400 - 0.5` when frameTimes are unavailable.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The 50th percentile of all motion energy values across usable trials is used as the threshold. Values >= threshold are "high" (1), below are "low" (0).

ii.
```python
me_thresh = np.percentile(me_data[:, trial_idx], 50)
out[5, :] = (me_data[:, ti] >= me_thresh).astype(np.int64)
```

iii. This follows the instructions for per-session 50th percentile thresholding. Note that the reference code uses a manually set `me.moveThresh` instead.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned using side camera frame times minus video offset minus go cue time, then interpolated to the neural time axis. When frame times are unavailable, synthetic times at 400 Hz with 0.5s offset are used.

ii.
```python
aln = side_ft[ti] - vidshift - goCue[ti]
interp_fn = interp1d(aln, trial_me, kind='linear', bounds_error=False, fill_value=np.nan)
me_data[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. Matches `loadMotionEnergy.m` alignment approach.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing data in several ways:
- **Missing .mat files**: Sessions with missing data files are skipped.
- **Missing motion energy**: Sessions without motion energy files get all-zero motion energy data.
- **NaN in interpolated video data**: Filled with nearest valid value via `fill_nan_nearest`.
- **Invalid DLC positions**: Low-confidence (< 0.9) or NaN positions are filled with nearest valid position before velocity computation.
- **Missing frame times**: Synthetic frame times at 400 Hz with 0.5s offset are generated.
- **All-zero neural data**: Late trials with no spikes pass through without special handling.
- **Try/except blocks**: Extensive exception handling around DLC and motion energy processing to prevent crashes.

ii.
```python
def fill_nan_nearest(arr):
    nans = np.isnan(arr)
    if np.all(nans):
        arr[:] = 0
        return arr
    valid_idx = np.where(~nans)[0]
    f_interp = interp1d(valid_idx, arr[valid_idx], kind='nearest',
                        fill_value='extrapolate', bounds_error=False)
    arr[nans] = f_interp(np.where(nans)[0])
    return arr
```

iii. CONVERSION_NOTES.md documents specific warnings about all-zero neural data, motion energy loading failures, and paw velocity threshold issues.

## 11-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading HDF5 data files, especially reading spike data with object references. (2) Spike binning: a nested loop over all neurons and all trials, computing histograms for each. (3) DLC velocity computation: looping over all trials, computing speed, and interpolating for each trial.

ii.
```python
# Spike binning - O(n_neurons * n_trials)
for ni, spk in enumerate(spike_data):
    for j in range(n_trials):
        mask = spk['trial'] == (ti + 1)
        aligned = spk['trialtm'][mask] - goCue[ti]
        counts, _ = np.histogram(aligned, bins=EDGES)
        fr[ni, :, ti] = causal_smooth(counts.astype(np.float64) / DT)
```

iii. The spike binning loop is the primary bottleneck, iterating over every neuron-trial pair and doing a linear scan (`mask = spk['trial'] == (ti + 1)`) for each.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. (1) The spike binning loop over trials could be vectorized by grouping spikes by trial upfront using `np.searchsorted` or similar. (2) The trial mask computation `spk['trial'] == (ti + 1)` scans all spikes for each trial; pre-grouping spikes by trial would avoid repeated scans. (3) The DLC velocity computation could process all trials at once for each feature rather than looping.

ii.
```python
# This linear scan happens for every neuron-trial pair:
mask = spk['trial'] == (ti + 1)
```

iii. No specific justification was provided by the AI for this implementation choice.

## 11-c. What processing does the code repeat multiple times?

i. (1) The video offset (`vidshift`) computation is done once per session, which is efficient. (2) The smoothing kernel is recreated for every call to `causal_smooth` (once per neuron per trial), when it could be computed once and reused. (3) The trial mask `spk['trial'] == (ti + 1)` is recomputed for every trial across all neurons, though the trial assignments are the same.

ii.
```python
def causal_smooth(x, N=SMOOTH_N, bctype=BC_TYPE):
    kern = gausswin(N)  # recreated every call
    kern[:N // 2] = 0
    kern /= kern.sum()
```

iii. No discussion of optimization was provided by the AI.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) Neural data is computed for ALL trials (including filtered ones), then only usable trials are selected. The firing rates for excluded trials (early, stim, no-response) are computed but never used. (2) DLC velocities and motion energy are also computed for all trials, not just usable ones. (3) The smoothing is applied to all trials' spike data before the low-FR filter, meaning some neurons' data is smoothed and then discarded.

ii.
```python
# fr computed for all n_trials
fr = np.zeros((n_raw, T_BINS, n_trials), dtype=np.float32)
for ni, spk in enumerate(spike_data):
    for ti in range(n_trials):  # ALL trials, not just usable ones
        # ...

# Only usable trials are selected later
for ti in trial_idx:
    neural_list.append(fr[:, :, ti].copy())
```

iii. The AI computes firing rates for all trials to use them in the low-FR filter (which averages across all trials). However, the reference code also computes trialdat for all trials.
