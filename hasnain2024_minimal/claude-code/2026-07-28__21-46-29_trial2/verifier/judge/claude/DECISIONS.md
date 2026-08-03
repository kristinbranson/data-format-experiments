# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each session is one MATLAB file, `data_structure_<anm>_<date>.mat`, located in one of two folders (`Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior`). Motion energy is loaded from a separate `motionEnergy_<anm>_<date>.mat` file. The 44 sessions are hard-coded in `ALL_SESSIONS`, transcribed from the authors' loading scripts. The script auto-detects whether each file is HDF5 (v7.3) or v5 format and uses `h5py` or `scipy.io.loadmat` accordingly, with separate loader functions `_load_session_h5` and `_load_session_v5`.

ii. Session list:
```python
ALL_SESSIONS = [
    {'dir': EB, 'anm': 'EKH1', 'date': '2021-08-07', 'probes': [2]},
    ...
    {'dir': RD, 'anm': 'JEB24', 'date': '2023-11-03', 'probes': [1]},
]
```

Loading dispatch:
```python
if _is_hdf5(data_path):
    sd = _load_session_h5(data_path, sess)
else:
    sd = _load_session_v5(data_path, sess)
```

iii. The AI noted in CONVERSION_NOTES.md: "Sessions were identified from the DataLoadingScripts/Recording and video/ MATLAB loading scripts." The two-format handling was needed because some sessions are v7.3 HDF5 and others are v5.

## 1-b. How are the data split into subjects?

i. The animal name is taken from the `anm` field of the session definition dict. Subjects are assigned indices in encounter order (not sorted). Each session's animal is looked up in a dict mapping animal name to index.

ii.
```python
anm = result['anm']
if anm not in subjects:
    subjects[anm] = len(subjects)
all_subject_idx.append(subjects[anm])
```

iii. The AI used the session definition's `anm` field directly, which comes from the hard-coded session list.

## 1-c. How are the data split into sessions?

i. One session is one entry in `ALL_SESSIONS`, identified by animal name, date, and directory. Each becomes one element of the `neural`, `input`, and `output` lists. The result is 44 sessions: 25 fixed-delay and 19 randomized-delay.

ii.
```python
for sess in sessions:
    result, err = process_session(sess, data_dir)
    if result is None:
        print(f"  SKIPPED: {err}")
        continue
    all_neural.append(result['neural'])
```

iii. Session selection matches the authors' loading scripts, as documented in CONVERSION_NOTES.md.

## 1-d. How are the data split into trials?

i. Each session has `Ntrials` trials from `bp.Ntrials`. All per-trial fields (hit, miss, early, etc.) are read and truncated to `n_trials`. Spike data carries trial assignment via the `trial` field.

ii.
```python
n_trials = int(bp['Ntrials'][0, 0])
hit = bp['hit'][0, 0].flatten()[:n_trials].astype(int)
```

iii. The trial structure is defined by the Bpod behavioral data, with one entry per trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials to only keep **hit or miss** trials, excluding early-lick and photostimulation trials. This means **no-response/ignore trials are excluded**. The filter condition is `(hit == 1) | (miss == 1)) & (stim_en == 0) & (early == 0)`. Additionally, sessions with fewer than 2 usable trials or fewer than 10 neurons are skipped entirely.

ii.
```python
use = ((hit == 1) | (miss == 1)) & (stim_en == 0) & (early == 0)
trial_idx = np.where(use)[0]

if len(trial_idx) < MIN_TRIALS:
    return None, f"Only {len(trial_idx)} usable trials"
```

iii. The AI stated in CONVERSION_NOTES.md: "Matches findTrials.m condition '(hit|miss)&~stim.enable&~early'" and noted "No-response trials are also excluded (they are neither hit nor miss)."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The spike-sorted clusters from `obj.clu{probe}`, specifically the `trial` and `trialtm` fields for each cluster. The `quality` field is used for filtering. Go cue times from `bp.ev.goCue` provide the alignment.

ii.
```python
spike_data.append({
    'trial': probe_arr['trial'][0, ci].flatten().astype(float),
    'trialtm': probe_arr['trialtm'][0, ci].flatten().astype(float),
})
```

iii. Same raw variables as the reference: spike times relative to trial start (`trialtm`) and trial assignment (`trial`).

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue by subtracting `goCue[trial]` from `trialtm`. They are then binned into **10 ms bins** (500 bins from -2.5 to +2.5 s), converted to firing rates by dividing by `DT`, and smoothed with a **causal Gaussian kernel** (window size 15, first half zeroed out), with reflect boundary conditions.

ii.
```python
DT = 0.01           # 10ms bins (params.dt = 1/100)
SMOOTH_N = 15        # smoothing window (params.smooth = 15)
...
aligned = spk['trialtm'][mask] - goCue[ti]
counts, _ = np.histogram(aligned, bins=EDGES)
fr[ni, :, ti] = causal_smooth(counts.astype(np.float64) / DT)
```

Causal smoothing:
```python
def causal_smooth(x, N=SMOOTH_N, bctype=BC_TYPE):
    kern = gausswin(N)
    kern[:N // 2] = 0   # causal: zero out first half
    kern /= kern.sum()
    ...
    out = np.convolve(x_padded, kern, mode='same')
```

iii. The AI interpreted `params.dt = 1/100` as 10 ms bins and implemented the smoothing as a causal Gaussian matching `mySmooth.m`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. First, clusters whose quality label (lower-cased) is in `{garbage, gabrga, noisy, real?}` are excluded. Empty quality strings are also excluded. Then neurons with mean firing rate <= 1 Hz are removed. Sessions with fewer than 10 neurons after filtering are skipped entirely.

ii.
```python
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
...
valid_clu = [i for i, q in enumerate(qualities)
             if q.lower() not in EXCLUDED_QUALITIES and q != '']
...
mean_fr = np.mean(fr, axis=(1, 2))
keep = mean_fr > LOW_FR
```

iii. The AI stated this matches `findClusters.m` with `'all'` quality parameter. Notably, `poor` is NOT in the exclusion set.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue by subtracting `goCue[trial_index]` from `trialtm` before binning.

ii.
```python
aligned = spk['trialtm'][mask] - goCue[ti]
counts, _ = np.histogram(aligned, bins=EDGES)
```

iii. This matches the reference's `alignSpikes.m`: `trialtm_aligned = trialtm - goCue`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses **10 ms bins** (DT = 0.01, 500 bins), interpreting `params.dt = 1/100` from `WorkingWithDataObjs.m`. No rebinning is applied after the initial binning.

ii.
```python
DT = 0.01           # 10ms bins (params.dt = 1/100)
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
T_BINS = len(TIME_AXIS)  # 500
```

iii. The AI interpreted the reference code parameter `params.dt = 1/100` as specifying 10 ms bins. The comment in the code says "params.dt = 1/100".

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the bin centers of the time axis, defined from -2.5 to +2.5 s around the go cue in 10 ms steps.

ii.
```python
TIME_AXIS = EDGES[:-1] + DT / 2  # bin centers
...
input_list.append(TIME_AXIS[np.newaxis, :].astype(np.float32).copy())
```

iii. The time axis is constructed from the binning parameters, not from raw data.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is computed as the centers of the bins used for spike binning. It is the same for every trial.

ii.
```python
TIME_AXIS = EDGES[:-1] + DT / 2
```

iii. N/A - this is a constructed variable.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis is the bin centers of the same binning grid used for the neural data, so they are inherently aligned.

ii.
```python
input_list.append(TIME_AXIS[np.newaxis, :].astype(np.float32).copy())  # (1, T)
```

iii. Same grid definition ensures alignment.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI uses `bp.R` directly as the lick direction, where R=1 means right and R=0 (i.e., left) means left. Since only hit and miss trials are kept, every trial has a lick.

ii.
```python
R = bp['R'][0, 0].flatten()[:n_trials].astype(int)
...
out[0, :] = int(R[ti])  # lick direction: L=0, R=1
```

iii. The AI treats R as the lick direction directly, with only 2 classes (left=0, right=1).

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The value of `R[ti]` is used directly. Since the AI only keeps hit and miss trials, R indicates the instructed side. On hit trials, the animal licked the instructed side, so R is the lick direction. On miss trials, the animal licked the opposite side, but the AI uses R (the instructed side) rather than the actual lick direction.

ii.
```python
out[0, :] = int(R[ti])  # lick direction: L=0, R=1
```

iii. The AI did not derive actual lick direction from hit/miss + R. It simply used R as the lick direction.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `bp.autowater` is used. Autowater trials are WC (0), others are DR (1).

ii.
```python
autowater = bp['autowater'][0, 0].flatten()[:n_trials].astype(int)
...
out[1, :] = 1 - int(autowater[ti])  # context: WC=0, DR=1
```

iii. Same source variable as the reference.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. `1 - autowater` gives WC=0 and DR=1.

ii.
```python
out[1, :] = 1 - int(autowater[ti])  # context: WC=0, DR=1
```

iii. Direct mapping matches the instructions' WC=0, DR=1.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The `hit` field from `bp` is used directly. Since only hit and miss trials are kept, hit=1 means correct and hit=0 means incorrect (miss).

ii.
```python
hit = bp['hit'][0, 0].flatten()[:n_trials].astype(int)
...
out[2, :] = int(hit[ti])  # outcome: incorrect=0, correct=1
```

iii. With the trial filter excluding ignore trials, hit directly gives the binary outcome.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The value of `hit[ti]` is used directly as the outcome. Only 2 classes: incorrect=0, correct=1. No ignore class exists because those trials are filtered out.

ii.
```python
out[2, :] = int(hit[ti])  # outcome: incorrect=0, correct=1
```

iii. The AI's output_values lists only `['incorrect', 'correct']`, not including an ignore class.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DLC tracking from `obj.traj`, specifically the **bottom camera only** (`top_tongue` feature). The side camera tongue is not used. Frame times and the video offset (from `sglx.bitcode.bitstart` and `bp.ev.bitStart`) are used for alignment.

ii.
```python
tongue_fi = feat_names.index('top_tongue') if 'top_tongue' in feat_names else 0
...
t_spd = compute_speed(
    ts[:, 0, tongue_fi], ts[:, 1, tongue_fi], ft,
    ts[:, 2, tongue_fi], DLC_CONF, fill_missing=True)
```

iii. The AI used only the bottom camera for tongue tracking. The CONVERSION_NOTES.md describes "DLC Velocities (Bottom Camera)" without mentioning the side camera's tongue feature.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI fills invalid positions (NaN or low confidence < 0.9) with nearest valid position using `fill_positions_nearest`, then computes instantaneous speed as `sqrt(dx/dt^2 + dy/dt^2)` using `np.diff`. Speed is then interpolated to the neural time axis using linear interpolation, and remaining NaN values are filled with nearest neighbor. The result is discretized at the 50th percentile of **non-zero** values only.

ii.
```python
def compute_speed(x, y, ft, confidence=None, conf_thresh=DLC_CONF, fill_missing=True):
    valid = ~np.isnan(x) & ~np.isnan(y)
    if confidence is not None:
        valid = valid & (confidence >= conf_thresh)
    if np.any(valid) and not np.all(valid):
        x, y = fill_positions_nearest(x, y, valid)
    ...
    vx = np.diff(x) / dt_vid
    vy = np.diff(y) / dt_vid
    speed = np.sqrt(vx**2 + vy**2)
    speed = np.concatenate([[0.0], speed])
    return speed
```

Discretization:
```python
tongue_nonzero = tongue_usable[tongue_usable > 0]
tongue_thresh = np.percentile(tongue_nonzero, 50)
out[3, :] = (tongue_vel[:, ti] >= tongue_thresh).astype(np.int64)
```

iii. The AI's CONVERSION_NOTES.md explains: "Fill invalid positions (NaN or low confidence < 0.9) with nearest valid position... Tongue is only visible during licking (~4-8% of frames)... 50th percentile threshold is computed on non-zero values only."

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The 50th percentile of non-zero tongue velocity values across the session's used trials is computed. Values >= threshold are "high" (1), below are "low" (0). Only 2 categories; no "not visible" class.

ii.
```python
tongue_nonzero = tongue_usable[tongue_usable > 0]
tongue_thresh = np.percentile(tongue_nonzero, 50)
out[3, :] = (tongue_vel[:, ti] >= tongue_thresh).astype(np.int64)
```

iii. The AI modified the thresholding to use non-zero values because "the tongue is only visible during licking (~4% of frames)" and using all values would give a threshold of 0.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Camera frame times are corrected by the video offset and go cue: `ft - vidshift - goCue[ti]`. The resulting speed trace is then linearly interpolated onto the neural time axis (`TIME_AXIS`), and any remaining NaN values are filled with nearest neighbor interpolation.

ii.
```python
aln = ft - vidshift - goCue[ti]
interp_fn = interp1d(aln, t_spd, kind='linear',
                     bounds_error=False, fill_value=np.nan)
tongue_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The AI uses interpolation rather than binning (averaging frames within each bin).

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The DLC tracking from the bottom camera, using the `top_paw` feature (or `bottom_paw` as fallback).

ii.
```python
paw_fi = -1
for pname in ['top_paw', 'bottom_paw']:
    if pname in feat_names:
        paw_fi = feat_names.index(pname)
        break
```

iii. Same feature choice as the reference (top_paw from bottom camera).

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same as tongue velocity: fill invalid positions with nearest valid, compute speed from `np.diff`, interpolate to neural time axis, fill remaining NaN with nearest neighbor.

ii.
```python
p_spd = compute_speed(
    ts_p[:, 0, paw_fi], ts_p[:, 1, paw_fi], ft_p,
    ts_p[:, 2, paw_fi], DLC_CONF, fill_missing=True)
interp_fn = interp1d(aln, p_spd, kind='linear',
                     bounds_error=False, fill_value=np.nan)
paw_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. Same pipeline as tongue velocity, applied to paw tracking.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Standard 50th percentile of all paw velocity values across the session's used trials. Values >= threshold are "high" (1), below are "low" (0). Only 2 categories.

ii.
```python
paw_thresh = np.percentile(paw_vel[:, trial_idx], 50)
out[4, :] = (paw_vel[:, ti] >= paw_thresh).astype(np.int64)
```

iii. Uses `np.percentile` on all values (not non-zero like tongue).

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: video offset correction + go cue subtraction, then linear interpolation to neural time axis, then nearest-fill.

ii.
```python
aln = ft - vidshift - goCue[ti]
interp_fn = interp1d(aln, p_spd, kind='linear',
                     bounds_error=False, fill_value=np.nan)
paw_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. Same alignment pipeline as tongue.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. A separate file `motionEnergy_<anm>_<date>.mat`, loaded with `scipy.io.loadmat`. The data is in `me.data` (a cell array of per-trial traces). Side camera frame times are used for temporal alignment.

ii.
```python
me_mat = scipy.io.loadmat(me_path, squeeze_me=False)
me_struct = me_mat['me']
me_trials = me_struct['data'][0, 0]
```

iii. Same source as the reference. However, the AI's loading code does not handle the double-wrapped struct format that occurs in 3 sessions.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The per-trial motion energy trace is aligned to the go cue using side camera frame times and the video offset, then linearly interpolated to the neural time axis. Remaining NaN values are filled with nearest neighbor. Discretized at the session's 50th percentile.

ii.
```python
trial_me = me_trials[ti, 0].flatten().astype(np.float64)
aln = side_ft[ti] - vidshift - goCue[ti]
interp_fn = interp1d(aln, trial_me, kind='linear',
                     bounds_error=False, fill_value=np.nan)
me_data[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. Same approach as the velocity features: interpolation rather than binning.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Standard 50th percentile of all motion energy values across the session's used trials. Values >= threshold are "high" (1), below are "low" (0). Only 2 categories.

ii.
```python
me_thresh = np.percentile(me_data[:, trial_idx], 50)
out[5, :] = (me_data[:, ti] >= me_thresh).astype(np.int64)
```

iii. Standard thresholding. However, sessions where motion energy loading failed have all-zero values, leading to a threshold of 0 and all values classified as "high".

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side camera frame times are corrected by the video offset and go cue, then the motion energy trace is interpolated to the neural time axis. When side camera frame times are unavailable, a synthetic time axis at 400 Hz is used as fallback.

ii.
```python
if ti in side_ft:
    aln = side_ft[ti] - vidshift - goCue[ti]
    if len(trial_me) != len(aln):
        aln = np.arange(len(trial_me)) / 400.0 - 0.5 - goCue[ti]
else:
    aln = np.arange(len(trial_me)) / 400.0 - 0.5 - goCue[ti]
```

iii. The fallback time axis at 400 Hz with -0.5 offset is a heuristic when frame times are unavailable.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI fills missing/invalid DLC positions with nearest valid positions (`fill_positions_nearest`). NaN values after interpolation are filled with `fill_nan_nearest`. If no valid DLC data exists for a trial, speed defaults to zero. Motion energy loading failures result in all-zero values for affected sessions. Sessions where the recording ends before behavior (all-zero neural data) are noted as a warning but trials are kept. The AI does NOT have a "not visible" class for movement features.

ii.
```python
def fill_positions_nearest(x, y, valid_mask):
    ...
    x[bad_idx] = x[valid_idx[final]]
    y[bad_idx] = y[valid_idx[final]]
    return x, y

def fill_nan_nearest(arr):
    ...
    f_interp = interp1d(valid_idx, arr[valid_idx], kind='nearest',
                        fill_value='extrapolate', bounds_error=False)
    arr[nans] = f_interp(np.where(nans)[0])
    return arr
```

iii. The AI's CONVERSION_NOTES.md documents several data issues including all-zero neural data in late trials, motion energy loading failures, and paw velocity threshold=0 in some sessions.

## 11-a. What are the most time-consuming steps of the code?

i. Loading and parsing the MATLAB files, especially HDF5 files which require reading individual references for each cluster. The per-trial loop over all neurons for spike binning (`n_neurons * n_trials` histogram operations) is also potentially expensive.

ii.
```python
for ni, spk in enumerate(spike_data):
    for ti in range(n_trials):
        mask = spk['trial'] == (ti + 1)
        aligned = spk['trialtm'][mask] - goCue[ti]
        counts, _ = np.histogram(aligned, bins=EDGES)
        fr[ni, :, ti] = causal_smooth(counts.astype(np.float64) / DT)
```

iii. File I/O dominates. The nested neuron-by-trial loop for spike binning is a secondary cost.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning uses a double loop over neurons and trials (`for ni` ... `for ti`), computing one histogram per neuron per trial. This could be vectorized using `np.histogram2d` (as the reference does) to bin all trials for one neuron in a single call. The DLC velocity computation also loops per trial.

ii.
```python
for ni, spk in enumerate(spike_data):
    for ti in range(n_trials):
        mask = spk['trial'] == (ti + 1)
        ...
        counts, _ = np.histogram(aligned, bins=EDGES)
```

iii. The reference solution uses `np.histogram2d` to bin all trials of a cluster in one call, avoiding the inner trial loop entirely.

## 11-c. What processing does the code repeat multiple times?

i. The spike binning recomputes the trial mask `spk['trial'] == (ti + 1)` for every trial within the inner loop, effectively scanning the full spike array `n_trials` times per neuron. The smoothing convolution is also applied separately to each (neuron, trial) pair rather than vectorized.

ii.
```python
for ti in range(n_trials):
    mask = spk['trial'] == (ti + 1)
```

iii. This could be avoided by sorting spikes by trial once, or using histogram2d.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes per-trial velocity and motion energy for ALL trials (including early-lick and stim trials), then only uses the filtered subset (`trial_idx`). The spike binning is also done for all trials before filtering. Additionally, the `no_resp` field is loaded but not used in any computation.

ii.
```python
fr = np.zeros((n_raw, T_BINS, n_trials), dtype=np.float32)
for ni, spk in enumerate(spike_data):
    for ti in range(n_trials):  # all trials, not just trial_idx
        ...
```

```python
tongue_vel = np.zeros((T_BINS, n_trials), dtype=np.float32)
...
for ti in range(n_trials):  # all trials
```

iii. Processing all trials before filtering wastes computation on trials that will be discarded.
