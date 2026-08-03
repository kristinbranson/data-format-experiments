# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from `.mat` files located in `data/Ephys_Behavior/` and `data/RandomizedDelay_Ephys_Behavior/` directories. It auto-detects whether each file is MATLAB v7.3 (HDF5) or v5 format and uses the appropriate loader (`h5py` vs `scipy.io.loadmat`). Each session is loaded from a file named `data_structure_<animal>_<date>.mat`. Motion energy is loaded separately from `motionEnergy_<animal>_<date>.mat` files. Session definitions (animal, date, probes) are hardcoded in the `ALL_SESSIONS` list, derived from the MATLAB loading scripts.

ii.
```python
ALL_SESSIONS = [
    {'dir': EB, 'anm': 'EKH1', 'date': '2021-08-07', 'probes': [2]},
    # ... 44 total session definitions
]

def process_session(sess, data_dir):
    data_path = os.path.join(data_dir, sess['dir'], f"data_structure_{sname}.mat")
    me_path = os.path.join(data_dir, sess['dir'], f"motionEnergy_{sname}.mat")
    if _is_hdf5(data_path):
        sd = _load_session_h5(data_path, sess)
    else:
        sd = _load_session_v5(data_path, sess)
```

iii. The agent identified session definitions from the MATLAB loading scripts in `DataLoadingScripts/Recording and video/` directory. It read each animal's loading script to determine which sessions and probes to include. The dual-format handling was added after the initial conversion crashed on JEB23/JEB24 sessions that were MATLAB v5 format.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `anm` (animal name) field in each session definition. Unique animal names are collected as sessions are processed, building an ordered list. Each session is assigned a `subject_idx` pointing into this list.

ii.
```python
subjects = {}  # name -> index
for sess in sessions:
    result, err = process_session(sess, data_dir)
    anm = result['anm']
    if anm not in subjects:
        subjects[anm] = len(subjects)
    all_subject_idx.append(subjects[anm])
```

iii. The agent followed the MATLAB loading scripts which organize data by animal. 14 unique subjects were identified across the 44 sessions.

## 1-c. How are the data split into sessions?

i. Each session corresponds to one recording day for one animal. Sessions are defined by the combination of animal name, date, and directory (Ephys_Behavior or RandomizedDelay_Ephys_Behavior). Each session is processed independently and results in one entry in the output lists. 44 sessions total were successfully processed (25 Ephys_Behavior + 19 RandomizedDelay_Ephys_Behavior).

ii.
```python
ALL_SESSIONS = [
    {'dir': EB, 'anm': 'EKH1', 'date': '2021-08-07', 'probes': [2]},
    # ...
]
for sess in sessions:
    result, err = process_session(sess, data_dir)
    if result is None:
        print(f"  SKIPPED: {err}")
        continue
    all_neural.append(result['neural'])
```

iii. The agent noted that JEB4 and JEB5 were referenced in loading scripts but data files were not provided. JEB23 2023-10-20 was excluded (commented out in loading script). JEB24's early dates (2023-10-03, 2023-10-04) were excluded as they weren't referenced in loading scripts.

## 1-d. How are the data split into trials?

i. Each session's `.mat` file contains `bp.Ntrials` indicating total trials. The neural data (spike times per cluster) is organized by trial number. For each trial, spikes are extracted by matching `clu.trial == trial_number`. A trial filter is applied before assembling the final output (see 1-e).

ii.
```python
n_trials = int(bp['Ntrials'][0, 0])
# For each neuron:
for ti in range(n_trials):
    mask = spk['trial'] == (ti + 1)
    aligned = spk['trialtm'][mask] - goCue[ti]
    counts, _ = np.histogram(aligned, bins=EDGES)
```

iii. The agent followed the data structure in the `.mat` files, where trial information is stored per-cluster with trial number indices.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered using the condition `(hit|miss) & ~stim.enable & ~early`, matching the reference `findTrials.m`. This includes only hit and miss trials, excludes stimulation trials, and excludes early-lick trials. No-response trials are implicitly excluded (they are neither hit nor miss). Sessions with fewer than 2 usable trials are skipped entirely.

ii.
```python
use = ((hit == 1) | (miss == 1)) & (stim_en == 0) & (early == 0)
trial_idx = np.where(use)[0]
if len(trial_idx) < MIN_TRIALS:
    return None, f"Only {len(trial_idx)} usable trials"
```

iii. The agent read `findTrials.m` and identified the trial filter expression used in the reference processing scripts. The condition `'(hit|miss)&~stim.enable&~early'` was documented in CONVERSION_NOTES.md.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the cluster spike times stored in `obj.clu{probe}(cluster).trial` (which trial each spike belongs to) and `obj.clu{probe}(cluster).trialtm` (the time of each spike within that trial). These are combined with the go cue event times `obj.bp.ev.goCue` for alignment.

ii.
```python
spike_data.append({
    'trial': probe_arr['trial'][0, ci].flatten().astype(float),
    'trialtm': probe_arr['trialtm'][0, ci].flatten().astype(float),
})
```

iii. The agent read `getSeq.m` and `alignSpikes.m` to understand how spikes are aligned and binned.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to go cue onset, binned into 10ms bins from -2.5s to +2.5s (500 bins), converted to firing rate by dividing by bin size (dt=0.01s), then smoothed with a causal Gaussian kernel (window size N=15, reflect boundary condition). This matches `getSeq.m` and `mySmooth.m`.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)  # -2.5 to 2.5
TIME_AXIS = EDGES[:-1] + DT / 2  # bin centers
# Per neuron, per trial:
aligned = spk['trialtm'][mask] - goCue[ti]
counts, _ = np.histogram(aligned, bins=EDGES)
fr[ni, :, ti] = causal_smooth(counts.astype(np.float64) / DT)
```

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

iii. The agent studied `getSeq.m`, `mySmooth.m`, and `alignSpikes.m` to replicate the exact processing pipeline, including the causal Gaussian kernel with reflect boundary handling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) Cluster quality filter matching `findClusters.m` with 'all' mode — excludes clusters labeled 'garbage', 'gabrga', 'noisy', or 'real?'. (2) Firing rate filter matching `removeLowFRClusters.m` — removes neurons with mean firing rate <= 1 Hz. Sessions with fewer than 10 valid neurons are skipped.

ii.
```python
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
valid_clu = [i for i, q in enumerate(qualities)
             if q.lower() not in EXCLUDED_QUALITIES and q != '']

# After binning/smoothing:
mean_fr = np.mean(fr, axis=(1, 2))
keep = mean_fr > LOW_FR  # LOW_FR = 1.0
fr = fr[keep]
```

iii. The agent referenced `findClusters.m` for quality labels and `removeLowFRClusters.m` for the firing rate threshold. The agent noted that the FR filter is computed over all trials (including non-selected ones) and all time bins, rather than using condition-averaged PSTHs as in the reference code, but stated this should yield comparable results.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to go cue onset. For each trial, spike times (`trialtm`) are shifted by subtracting the go cue time for that trial, then histogrammed into bins relative to go cue = 0.

ii.
```python
aligned = spk['trialtm'][mask] - goCue[ti]
counts, _ = np.histogram(aligned, bins=EDGES)
```

iii. The agent followed `alignSpikes.m` which subtracts the alignment event time from spike times. The instructions specified go cue alignment, matching `params.alignEvent = 'goCue'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 10ms bins (DT = 0.01s). No rebinning is applied — spikes are directly histogrammed into 10ms bins. The time window spans -2.5s to +2.5s giving 500 time bins per trial.

ii.
```python
DT = 0.01  # 10ms bins (params.dt = 1/100)
TMIN = -2.5
TMAX = 2.5
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
```

iii. The agent matched the reference code parameter `params.dt = 1/100` for 10ms bins.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is derived from the time axis itself — the bin centers of the neural data bins, which are computed from the fixed time window parameters (TMIN, TMAX, DT). No raw data variable is used; it is a synthetic variable representing the time relative to go cue onset.

ii.
```python
TIME_AXIS = EDGES[:-1] + DT / 2  # bin centers
# Per trial:
input_list.append(TIME_AXIS[np.newaxis, :].astype(np.float32).copy())  # (1, T)
```

iii. The instructions specify "Time from go cue onset in seconds" as a continuous, time-varying decoder input. The agent implemented this as the time axis itself (bin centers from -2.495 to +2.495 seconds).

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing — it is simply the bin center times computed from the fixed parameters. The same time vector is used for every trial in every session.

ii.
```python
TIME_AXIS = EDGES[:-1] + DT / 2
input_list.append(TIME_AXIS[np.newaxis, :].astype(np.float32).copy())
```

iii. Since the neural data is already aligned to go cue onset, the time axis directly represents time from go cue.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis is identical to the neural data time axis (same bin centers), so they are inherently aligned. Both use the same EDGES and TIME_AXIS computed from TMIN, TMAX, and DT.

ii.
```python
# Same TIME_AXIS used for both neural binning and input
counts, _ = np.histogram(aligned, bins=EDGES)  # neural uses EDGES
input_list.append(TIME_AXIS[np.newaxis, :])     # input uses TIME_AXIS = EDGES bin centers
```

iii. No additional alignment is needed since both share the same time grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the `R` (right lick) field in `obj.bp.R`. The `L` (left lick) field also exists but is not directly used — if `R == 1`, direction is right (1); otherwise left (0).

ii.
```python
R = bp['R'][0, 0].flatten()[:n_trials].astype(int)
# Per trial:
out[0, :] = int(R[ti])  # lick direction: L=0, R=1
```

iii. The agent identified `bp.R` and `bp.L` as binary trial-level indicators. Using `R` directly gives 1 for right licks and 0 for left licks.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. No processing — the raw `R` value is used directly as a binary label. It is tiled across all time bins for that trial (same value at every time point).

ii.
```python
out[0, :] = int(R[ti])  # constant across time bins
```

iii. The instructions specify left = 0, right = 1. Since `R` is already a binary indicator (1 for right), it maps directly.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the `autowater` field in `obj.bp.autowater`. A value of 1 indicates water-cued (WC) trials, 0 indicates delayed-response (DR) trials.

ii.
```python
autowater = bp['autowater'][0, 0].flatten()[:n_trials].astype(int)
# Per trial:
out[1, :] = 1 - int(autowater[ti])  # context: WC=0, DR=1
```

iii. The agent identified `autowater` as the trial-level indicator for WC vs DR contexts. Inverting it (`1 - autowater`) maps WC=0 and DR=1 as specified.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The `autowater` value is inverted (1 - autowater) to match the specified encoding (WC=0, DR=1). The value is tiled across all time bins.

ii.
```python
out[1, :] = 1 - int(autowater[ti])
```

iii. Direct inversion of the binary field with no other processing.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the `hit` field in `obj.bp.hit`. A value of 1 indicates correct (hit) trials, 0 indicates incorrect (miss) trials.

ii.
```python
hit = bp['hit'][0, 0].flatten()[:n_trials].astype(int)
# Per trial:
out[2, :] = int(hit[ti])  # outcome: incorrect=0, correct=1
```

iii. Since the trial filter only keeps hit and miss trials, `hit` directly provides the binary outcome label.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. No processing — the raw `hit` value is used directly. Hit=1 maps to correct=1, miss (hit=0) maps to incorrect=0. Tiled across all time bins.

ii.
```python
out[2, :] = int(hit[ti])
```

iii. Direct use of the existing binary field.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the bottom camera DLC tracking data: specifically the `top_tongue` feature's (x, y, confidence) coordinates stored in `obj.traj{2}(trial).ts` and frame times `obj.traj{2}(trial).frameTimes`. The video offset (from `findVideoOffset`) and go cue times are used for alignment.

ii.
```python
tongue_fi = feat_names.index('top_tongue') if 'top_tongue' in feat_names else 0
# v5 format: ts shape (nframes, 3, nfeat) → ts[:, 0, tongue_fi], ts[:, 1, tongue_fi]
# h5 format: ts shape (nfeat, 3, nframes) → ts[tongue_fi, 0], ts[tongue_fi, 1]
t_spd = compute_speed(x, y, ft, confidence, DLC_CONF, fill_missing=True)
```

iii. The agent identified `top_tongue` as the DLC feature name from the bottom camera tracking data.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Processing steps: (1) Extract (x, y, confidence) for `top_tongue` feature. (2) Mark frames with confidence < 0.9 or NaN positions as invalid. (3) Fill invalid positions with nearest valid position. (4) Compute instantaneous speed: `sqrt((dx/dt)^2 + (dy/dt)^2)`. (5) Align frame times to go cue: `frameTimes - vidshift - goCue`. (6) Interpolate speed to neural time axis using linear interpolation. (7) Fill remaining NaN values with nearest non-NaN.

ii.
```python
def compute_speed(x, y, ft, confidence=None, conf_thresh=DLC_CONF, fill_missing=True):
    valid = ~np.isnan(x) & ~np.isnan(y)
    if confidence is not None:
        valid = valid & (confidence >= conf_thresh)
    if np.any(valid) and not np.all(valid):
        x, y = fill_positions_nearest(x, y, valid)
    vx = np.diff(x) / dt_vid
    vy = np.diff(y) / dt_vid
    speed = np.sqrt(vx**2 + vy**2)
    speed = np.concatenate([[0.0], speed])
    return speed

aln = ft - vidshift - goCue[ti]
interp_fn = interp1d(aln, t_spd, kind='linear', bounds_error=False, fill_value=np.nan)
tongue_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The agent studied the paper's DLC processing pipeline and implemented velocity computation from raw tracking coordinates, including confidence-based filtering and nearest-neighbor interpolation for missing data.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The tongue velocity is discretized using the 50th percentile of **non-zero** values across all usable trials in a session. Time bins with velocity below the threshold are labeled 0 (low), >= threshold labeled 1 (high). This deviates from a standard 50th percentile — the agent specifically excluded zero values because the tongue is only visible ~4% of frames, and including zeros would make the threshold ~0, classifying nearly everything as "high."

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

iii. The agent documented this decision in CONVERSION_NOTES.md: "Tongue is only visible during licking (~4-8% of frames). After filling invalid positions with nearest valid, velocity is near-zero when tongue is not visible. The 50th percentile threshold is computed on non-zero values only." The resulting distribution is ~93% low / ~7% high.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue velocity is aligned using the video offset and go cue time. Camera frame times are adjusted: `frameTimes - vidshift - goCue`, then the velocity time series is interpolated onto the neural time axis using linear interpolation.

ii.
```python
aln = ft - vidshift - goCue[ti]
interp_fn = interp1d(aln, t_spd, kind='linear', bounds_error=False, fill_value=np.nan)
tongue_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The agent followed `findVideoOffset.m` to compute the video-to-neural clock offset, then subtracted both the video offset and go cue time to align to the same time base as neural data.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom camera DLC tracking data for the `top_paw` (or `bottom_paw` as fallback) feature, using the same (x, y, confidence) format as tongue.

ii.
```python
paw_fi = -1
for pname in ['top_paw', 'bottom_paw']:
    if pname in feat_names:
        paw_fi = feat_names.index(pname)
        break
```

iii. The agent identified paw features from the DLC feature names in the bottom camera data.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same processing as tongue velocity: extract positions, fill invalid positions with nearest valid, compute instantaneous speed, align to go cue, interpolate to neural time axis.

ii.
```python
p_spd = compute_speed(ts_p[:, 0, paw_fi], ts_p[:, 1, paw_fi], ft_p,
                      ts_p[:, 2, paw_fi], DLC_CONF, fill_missing=True)
interp_fn = interp1d(aln, p_spd, kind='linear', bounds_error=False, fill_value=np.nan)
paw_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The agent used the same velocity computation pipeline for both tongue and paw.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity uses a standard 50th percentile threshold computed over all values (including zeros) across all usable trials in a session. Unlike tongue, no non-zero filtering is applied.

ii.
```python
paw_thresh = np.percentile(paw_vel[:, trial_idx], 50)
out[4, :] = (paw_vel[:, ti] >= paw_thresh).astype(np.int64)
```

iii. The agent noted that for some sessions (JEB13 2022-09-13 and 2022-09-21), paw velocity threshold was 0, making all values "high."

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same alignment as tongue velocity — using video offset and go cue time, interpolated to neural time axis.

ii.
```python
aln = ft - vidshift - goCue[ti]
interp_fn = interp1d(aln, p_spd, kind='linear', bounds_error=False, fill_value=np.nan)
paw_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. Same approach as tongue velocity alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from separate `motionEnergy_<animal>_<date>.mat` files. The data is in `me.data`, containing per-trial time series. Frame times come from the side camera (`obj.traj{1}(trial).frameTimes`).

ii.
```python
me_mat = scipy.io.loadmat(me_path, squeeze_me=False)
me_struct = me_mat['me']
me_trials = me_struct['data'][0, 0]
trial_me = me_trials[ti, 0].flatten().astype(np.float64)
```

iii. The agent read `loadMotionEnergy.m` to understand the motion energy data structure and loading process.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy is loaded from pre-computed files (not computed from raw video). It is aligned to go cue using side camera frame times and video offset, then interpolated to the neural time axis. Missing values are filled with nearest neighbor interpolation. When side camera frame times are unavailable, a fallback alignment is used assuming 400 Hz frame rate with 0.5s offset.

ii.
```python
if ti in side_ft:
    aln = side_ft[ti] - vidshift - goCue[ti]
    if len(trial_me) != len(aln):
        aln = np.arange(len(trial_me)) / 400.0 - 0.5 - goCue[ti]
else:
    aln = np.arange(len(trial_me)) / 400.0 - 0.5 - goCue[ti]
interp_fn = interp1d(aln, trial_me, kind='linear', bounds_error=False, fill_value=np.nan)
me_data[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The agent followed `loadMotionEnergy.m` which uses `interp1(frameTimes-vidshift-alignTimes, me.data, taxis)` with a fallback using synthetic frame times at 400 Hz.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy uses a standard 50th percentile threshold computed over all values in usable trials for each session.

ii.
```python
me_thresh = np.percentile(me_data[:, trial_idx], 50)
out[5, :] = (me_data[:, ti] >= me_thresh).astype(np.int64)
```

iii. For sessions where motion energy loading failed (resulting in all-zero data), the threshold is 0 and all values are classified as "high" (>= 0). This affects 7 sessions.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned using side camera frame times, video offset, and go cue time, then interpolated to the neural time axis.

ii.
```python
aln = side_ft[ti] - vidshift - goCue[ti]
interp_fn = interp1d(aln, trial_me, kind='linear', bounds_error=False, fill_value=np.nan)
me_data[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The reference `loadMotionEnergy.m` uses `taxis = obj.time + params.advance_movement` for interpolation. The AI does not account for `params.advance_movement`, interpolating directly to `TIME_AXIS`. If `advance_movement` is non-zero, this would introduce a temporal misalignment.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several types of missing/problematic data are handled:
- **Missing data files**: Sessions with missing `.mat` files are skipped.
- **NaN positions in DLC tracking**: Filled with nearest valid neighbor before computing velocity.
- **Low-confidence DLC tracking** (< 0.9): Treated same as NaN, filled with nearest valid.
- **NaN values after interpolation**: Filled with nearest non-NaN value (`fill_nan_nearest`).
- **Empty quality strings**: Clusters with empty quality strings are excluded.
- **Null-byte quality strings** (`\x00\x00`): Treated as valid (non-excluded) quality.
- **All-zero neural data**: Late trials in JEB24 sessions 36 and 43 have all-zero neural data but are NOT filtered out.
- **Motion energy loading failures**: Sessions with incompatible data formats get all-zero motion energy, resulting in all values classified as "high."
- **Missing video offset data**: Falls back to 0.5s default offset.

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

iii. The agent documented these issues in CONVERSION_NOTES.md and the conversion output logs.

## 11-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading large HDF5 files and extracting spike data per cluster. (2) The triple-nested loop for spike binning: iterating over neurons x trials, computing histograms for each. (3) DLC velocity computation per trial with position filling, speed computation, and interpolation. (4) Motion energy interpolation per trial.

ii.
```python
# Triple nested: neurons x trials for spike binning
fr = np.zeros((n_raw, T_BINS, n_trials), dtype=np.float32)
for ni, spk in enumerate(spike_data):
    for ti in range(n_trials):
        mask = spk['trial'] == (ti + 1)
        aligned = spk['trialtm'][mask] - goCue[ti]
        counts, _ = np.histogram(aligned, bins=EDGES)
        fr[ni, :, ti] = causal_smooth(counts.astype(np.float64) / DT)
```

iii. The agent did not explicitly document performance considerations, focusing on correctness over efficiency.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop (neurons x trials) could be partially vectorized by grouping spikes by trial first, then using vectorized histogram operations. The DLC velocity computation loop over trials could be vectorized using array operations. The smoothing operation could use `scipy.ndimage` or matrix multiplication instead of per-neuron convolution.

ii.
```python
# This loop could be vectorized:
for ni, spk in enumerate(spike_data):
    for ti in range(n_trials):
        mask = spk['trial'] == (ti + 1)
        aligned = spk['trialtm'][mask] - goCue[ti]
        counts, _ = np.histogram(aligned, bins=EDGES)
        fr[ni, :, ti] = causal_smooth(counts.astype(np.float64) / DT)
```

iii. No justification given — the agent prioritized code clarity and correctness.

## 11-c. What processing does the code repeat multiple times?

i. (1) The smoothing kernel (`gausswin(N)` and causal masking) is recomputed for every call to `causal_smooth` rather than being precomputed once. (2) For DLC data, the same trial's tracking data is loaded twice — once for tongue and once for paw — though paw reuses the already-loaded `ts, ft` variables. (3) The video offset (`vidshift`) is computed once per session, which is correct.

ii.
```python
def causal_smooth(x, N=SMOOTH_N, bctype=BC_TYPE):
    kern = gausswin(N)  # Recomputed every call
    kern[:N // 2] = 0
    kern /= kern.sum()
```

iii. The kernel computation is cheap but called thousands of times (neurons x trials per session).

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) Neural data is computed for ALL trials (including filtered-out trials like early, stim, no-response), but only a subset is used in the final output. The full firing rate matrix `fr` has shape `(n_neurons, T_BINS, n_trials)` but only `trial_idx` trials are kept. (2) The PSTH computation in the reference code is not replicated, but the agent computes single-trial data for all trials before filtering. (3) DLC velocities and motion energy are also computed for all trials, not just the usable ones.

ii.
```python
# Computes for ALL n_trials:
fr = np.zeros((n_raw, T_BINS, n_trials), dtype=np.float32)
for ni, spk in enumerate(spike_data):
    for ti in range(n_trials):  # ALL trials
        ...

# But only uses trial_idx in final output:
for ti in trial_idx:
    neural_list.append(fr[:, :, ti].copy())
```

iii. Processing all trials first is simpler to implement and ensures the firing rate filter uses all available data (matching the reference approach where PSTHs are computed over selected conditions).
