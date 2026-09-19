# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each session is one MATLAB file `data_structure_<anm>_<date>.mat` in one of two folders (`Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior`), with motion energy in a separate `motionEnergy_<anm>_<date>.mat` file. The 44 sessions are hard-coded in `ALL_SESSIONS`, transcribed from the authors' loading scripts. The code auto-detects HDF5 (v7.3) vs v5 MATLAB format and uses separate loading functions (`_load_session_h5` and `_load_session_v5`) for each.

ii.
```python
ALL_SESSIONS = [
    {'dir': EB, 'anm': 'EKH1', 'date': '2021-08-07', 'probes': [2]},
    ...
    {'dir': RD, 'anm': 'JEB24', 'date': '2023-11-03', 'probes': [1]},
]

if _is_hdf5(data_path):
    sd = _load_session_h5(data_path, sess)
else:
    sd = _load_session_v5(data_path, sess)
```

iii. The agent identified the 44 sessions from the authors' `load<ANM>_ALMVideo.m` scripts and noted that both MATLAB file formats occur across the dataset, requiring two loading code paths.

## 1-b. How are the data split into subjects?

i. The animal name is taken from the `anm` field in the session definition (e.g. `'EKH1'`). Subjects are accumulated in insertion order into a dict during processing, and the subject index for each session is assigned as sessions are processed.

ii.
```python
anm = result['anm']
if anm not in subjects:
    subjects[anm] = len(subjects)
all_subject_idx.append(subjects[anm])
```

iii. The agent derived the animal name from the session definition rather than from the filename or internal metadata.

## 1-c. How are the data split into sessions?

i. One session is one entry in `ALL_SESSIONS`, corresponding to one `.mat` file on disk. Both fixed-delay and randomized-delay sessions are treated uniformly. Each becomes one element of `neural`, `input`, and `output`. The result is 44 sessions.

ii.
```python
for sess in sessions:
    result, err = process_session(sess, data_dir)
    if result is None:
        print(f"  SKIPPED: {err}")
        continue
    all_neural.append(result['neural'])
```

iii. The agent followed the authors' loading scripts to determine which sessions to include.

## 1-d. How are the data split into trials?

i. Each session has `Ntrials` from `bp['Ntrials']`. Per-trial fields (hit, miss, early, etc.) are read and truncated to `n_trials`. Trials are indexed by their position in these arrays.

ii.
```python
n_trials = int(bp['Ntrials'][0, 0])
hit = bp['hit'][0, :n_trials].astype(int)
miss = bp['miss'][0, :n_trials].astype(int)
...
```

iii. The agent used the Bpod trial table which defines trials directly.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to keep only hit or miss trials that are not early-lick and not photostimulation trials. Ignore trials (where the animal did not respond) are **excluded**. This is the filter `(hit|miss) & ~stim & ~early`. There is also a minimum trial count filter (`MIN_TRIALS = 2`) and minimum neuron count filter (`MIN_UNITS = 10`) that can cause an entire session to be skipped.

ii.
```python
use = ((hit == 1) | (miss == 1)) & (stim_en == 0) & (early == 0)
trial_idx = np.where(use)[0]
if len(trial_idx) < MIN_TRIALS:
    return None, f"Only {len(trial_idx)} usable trials"
```

iii. The agent's reasoning (step 47) states: "For the decoder outputs, I need trials that support each variable... I'll filter for trials that are unstimulated and not early, then split by context... and include both correct and incorrect trials." The decision to exclude ignore trials was made to ensure all trials have a valid lick direction.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The spike-sorted clusters from `obj.clu{probe}`. Each cluster carries `trial` (the trial each spike belongs to, 1-based) and `trialtm` (the spike time relative to trial start). The go cue times `bp.ev.goCue` are used for alignment.

ii.
```python
spike_data.append({
    'trial': probe_arr['trial'][0, ci].flatten().astype(float),
    'trialtm': probe_arr['trialtm'][0, ci].flatten().astype(float),
})
...
aligned = spk['trialtm'][mask] - goCue[ti]
counts, _ = np.histogram(aligned, bins=EDGES)
```

iii. The agent identified `trial` and `trialtm` as the key fields for spike data.

## 2-b. How is the `neural` data processed?

i. Spike counts are computed in 10 ms bins from -2.5 to 2.5 s around the go cue, converted to firing rates (Hz) by dividing by the bin width (0.01 s), then smoothed with a **causal** Gaussian kernel of window size 15 bins with `reflect` boundary conditions. The causal kernel zeros out the first half of the Gaussian window.

ii.
```python
DT = 0.01           # 10ms bins
SMOOTH_N = 15

def causal_smooth(x, N=SMOOTH_N, bctype=BC_TYPE):
    kern = gausswin(N)
    kern[:N // 2] = 0   # causal: zero out first half
    kern /= kern.sum()
    if bctype == 'reflect':
        x_padded = np.concatenate([x[:N], x])
        trim = N
    out = np.convolve(x_padded, kern, mode='same')
    return out[trim:]

fr[ni, :, ti] = causal_smooth(counts.astype(np.float64) / DT)
```

iii. The agent's reasoning (step 59) states: "For the causal Gaussian kernel, I'm implementing the same approach as the MATLAB code: creating a 15-sample Gaussian window, zeroing out the first half to make it causal."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. First, clusters whose quality label (lower-cased) matches `{'garbage', 'gabrga', 'noisy', 'real?'}` or is empty are excluded. Then units whose mean firing rate over all bins and all trials is at or below 1 Hz are removed. There is also a minimum unit count (`MIN_UNITS = 10`) that can cause the session to be skipped entirely.

ii.
```python
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR = 1.0
MIN_UNITS = 10

valid_clu = [i for i, q in enumerate(qualities)
             if q.lower() not in EXCLUDED_QUALITIES and q != '']

mean_fr = np.mean(fr, axis=(1, 2))
keep = mean_fr > LOW_FR
```

iii. The agent followed the reference code's `findClusters.m` quality filter and the paper's 1 Hz firing rate threshold.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. For each cluster, spike times (`trialtm`) within each trial are aligned to the go cue by subtracting `goCue[ti]`, then histogrammed into time bins.

ii.
```python
aligned = spk['trialtm'][mask] - goCue[ti]
counts, _ = np.histogram(aligned, bins=EDGES)
```

iii. The agent correctly identified that `trialtm` is already relative to trial start on the behavior clock, so subtracting the go cue time aligns spikes to the go cue.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 10 ms bins (`DT = 0.01`), producing 500 time bins from -2.5 to 2.5 s. No rebinning is applied — spikes are directly counted into these bins.

ii.
```python
DT = 0.01           # 10ms bins (params.dt = 1/100)
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
if len(EDGES) > 501:
    EDGES = EDGES[:501]
TIME_AXIS = EDGES[:-1] + DT / 2  # bin centers
T_BINS = len(TIME_AXIS)
```

iii. The agent commented `params.dt = 1/100` to justify the 10 ms choice, though this is a misreading of the reference code where `params.dt = 1/200` (5 ms bins).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the bin centres of the time axis, defined by the binning parameters. It is not derived from any raw data variable — it is the time grid itself.

ii.
```python
TIME_AXIS = EDGES[:-1] + DT / 2  # bin centers
input_list.append(TIME_AXIS[np.newaxis, :].astype(np.float32).copy())
```

iii. The agent defined the time axis from the binning parameters.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is defined by the binning parameters. No processing of raw data is involved.

ii. N/A

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is the same time grid used for binning spikes, so alignment is inherent — bin k in the input corresponds to bin k in the neural data.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
```

iii. The agent ensured input and neural data share the same time axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The per-trial field `bp.R` (right-instructed side). The AI uses `R` directly as the lick direction code: `R=0` maps to left (0), `R=1` maps to right (1).

ii.
```python
R = bp['R'][0, :n_trials].astype(int)
...
out[0, :] = int(R[ti])  # lick direction: L=0, R=1
```

iii. The agent treated R as equivalent to the lick direction, which is only correct for hit trials. For miss trials, the animal licked the opposite direction from R.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. `R[ti]` is directly used as the lick direction code. There is no logic to derive the actual lick direction from the combination of instructed side and outcome. There is also **no "no lick" category** — since ignore trials are filtered out in 1-e, only hit and miss trials remain, and `R` is used directly for all of them.

ii.
```python
out[0, :] = int(R[ti])  # lick direction: L=0, R=1
```

iii. The agent's comment says "lick direction: L=0, R=1" but this is actually the instructed side, not the actual lick direction.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The per-trial field `bp.autowater`. Autowater=1 indicates water-cued (WC) context, autowater=0 indicates delayed-response (DR) context.

ii.
```python
autowater = bp['autowater'][0, :n_trials].astype(int)
...
out[1, :] = 1 - int(autowater[ti])  # context: WC=0, DR=1
```

iii. The agent correctly derived context from the autowater field.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling: `1 - autowater` gives WC=0 and DR=1.

ii.
```python
out[1, :] = 1 - int(autowater[ti])  # context: WC=0, DR=1
```

iii. The mapping follows the instructions.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The per-trial field `bp.hit`. The AI uses `hit` directly as the outcome code.

ii.
```python
hit = bp['hit'][0, :n_trials].astype(int)
...
out[2, :] = int(hit[ti])  # outcome: incorrect=0, correct=1
```

iii. The agent used hit directly since ignore trials are already excluded by the trial filter.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. `hit[ti]` is directly used: 1 for correct (hit), 0 for incorrect (miss). Since ignore trials are filtered out in 1-e, the remaining trials are either hits or misses, so `hit` directly maps to the outcome.

ii.
```python
out[2, :] = int(hit[ti])  # outcome: incorrect=0, correct=1
```

iii. The mapping is correct given the trial filter excludes ignore trials.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DLC tracking from `obj.traj`, specifically the **bottom camera only** (`traj[1]`). The feature `top_tongue` is used. Frame times are from the bottom camera's `frameTimes`.

ii.
```python
tongue_fi = feat_names.index('top_tongue') if 'top_tongue' in feat_names else 0
...
t_spd = compute_speed(
    ts[tongue_fi, 0], ts[tongue_fi, 1], ft,
    ts[tongue_fi, 2], DLC_CONF, fill_missing=True)
```

iii. The agent chose to use only the bottom camera's `top_tongue` feature for tongue velocity.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Several steps: (1) Low-confidence positions (below 0.9) are filled with the nearest valid position using `fill_positions_nearest`. (2) Speed is computed as the magnitude of the velocity vector: `sqrt(vx^2 + vy^2)` where `vx = diff(x)/diff(ft)` and `vy = diff(y)/diff(ft)`. (3) The speed is interpolated to the neural time bins using linear interpolation (`interp1d`). (4) NaN values after interpolation are filled with nearest non-NaN values (`fill_nan_nearest`). (5) The per-session 50th percentile is computed on **non-zero values only**, and values are thresholded into low (0) and high (1).

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

tongue_nonzero = tongue_usable[tongue_usable > 0]
if len(tongue_nonzero) > 0:
    tongue_thresh = np.percentile(tongue_nonzero, 50)
```

iii. The agent's reasoning explains the fill-missing approach: "fills NaN/low-confidence positions with nearest valid position before computing velocity... For tongue, positions are NaN when not visible; filling gives near-zero velocity during non-licking periods and meaningful velocity during licking." The non-zero percentile was chosen because the tongue is only visible ~4% of frames, making the overall 50th percentile 0.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The 50th percentile is computed on non-zero velocity values only. Values >= threshold are "high" (1), below are "low" (0). There is **no "not visible" category** — all bins get a value of 0 or 1 because missing positions are filled before computing velocity.

ii.
```python
tongue_nonzero = tongue_usable[tongue_usable > 0]
if len(tongue_nonzero) > 0:
    tongue_thresh = np.percentile(tongue_nonzero, 50)
else:
    tongue_thresh = 1.0
...
out[3, :] = (tongue_vel[:, ti] >= tongue_thresh).astype(np.int64)
```

iii. The agent computed the percentile on non-zero values to get a meaningful threshold when most frames have zero velocity.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by subtracting the video offset (`vidshift`) and the trial's go cue time. The resulting velocity is then linearly interpolated onto the neural time axis using `interp1d`, and NaN values (outside the frame time range) are filled with the nearest non-NaN value.

ii.
```python
aln = ft - vidshift - goCue[ti]
interp_fn = interp1d(aln, t_spd, kind='linear',
                     bounds_error=False, fill_value=np.nan)
tongue_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The agent computed the video offset from the bitcode timing, matching the reference's `findVideoOffset.m`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The DLC tracking from the bottom camera (`obj.traj[1]`), using the `top_paw` feature (or `bottom_paw` as fallback).

ii.
```python
paw_fi = -1
for pname in ['top_paw', 'bottom_paw']:
    if pname in feat_names:
        paw_fi = feat_names.index(pname)
        break
```

iii. The agent preferred `top_paw` and used `bottom_paw` as fallback.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same pipeline as tongue velocity: fill missing positions with nearest valid, compute speed from position differences, interpolate to neural time bins, fill NaN with nearest. The threshold is the 50th percentile of **all** paw velocity values (not just non-zero).

ii.
```python
p_spd = compute_speed(
    ts_p[paw_fi, 0], ts_p[paw_fi, 1], ft_p,
    ts_p[paw_fi, 2], DLC_CONF, fill_missing=True)
interp_fn = interp1d(aln, p_spd, kind='linear',
                     bounds_error=False, fill_value=np.nan)
paw_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
...
paw_thresh = np.percentile(paw_vel[:, trial_idx], 50)
```

iii. The agent used the same velocity computation for paw as for tongue.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The 50th percentile of all paw velocity values is used as the threshold. Values >= threshold are "high" (1), below are "low" (0). There is **no "not visible" category**.

ii.
```python
paw_thresh = np.percentile(paw_vel[:, trial_idx], 50)
out[4, :] = (paw_vel[:, ti] >= paw_thresh).astype(np.int64)
```

iii. The agent used the overall 50th percentile for paw since the paw is tracked in most frames.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: frame times corrected by video offset and go cue, then linearly interpolated to neural time bins.

ii.
```python
interp_fn = interp1d(aln, p_spd, kind='linear',
                     bounds_error=False, fill_value=np.nan)
paw_vel[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. Same alignment approach as all camera-derived signals.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. A separate file `motionEnergy_<anm>_<date>.mat`, loaded with `scipy.io.loadmat`. The `me.data` field contains one trace per trial.

ii.
```python
me_mat = scipy.io.loadmat(me_path, squeeze_me=False)
me_struct = me_mat['me']
me_trials = me_struct['data'][0, 0]
trial_me = me_trials[ti, 0].flatten().astype(np.float64)
```

iii. The agent loaded motion energy from the separate file.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. No processing beyond alignment and interpolation. The motion energy trace is interpolated onto the neural time bins using the side camera frame times (corrected by video offset and go cue), then NaN values are filled with the nearest non-NaN value. If side camera frame times are unavailable or length mismatches, a synthetic time axis assuming 400 Hz is used.

ii.
```python
if ti in side_ft:
    aln = side_ft[ti] - vidshift - goCue[ti]
    if len(trial_me) != len(aln):
        aln = np.arange(len(trial_me)) / 400.0 - 0.5 - goCue[ti]
else:
    aln = np.arange(len(trial_me)) / 400.0 - 0.5 - goCue[ti]
interp_fn = interp1d(aln, trial_me, kind='linear',
                     bounds_error=False, fill_value=np.nan)
me_data[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. The agent used side camera frame times when available and fell back to a synthetic 400 Hz time axis.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The 50th percentile of all motion energy values across the session's usable trials is used as the threshold. Values >= threshold are "high" (1), below are "low" (0). There is **no "no video" category**.

ii.
```python
me_thresh = np.percentile(me_data[:, trial_idx], 50)
out[5, :] = (me_data[:, ti] >= me_thresh).astype(np.int64)
```

iii. The agent used the overall 50th percentile.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side camera frame times are corrected by the video offset and go cue, then the motion energy trace is linearly interpolated onto the neural time axis. If frame times are unavailable, a synthetic 400 Hz axis is used.

ii.
```python
aln = side_ft[ti] - vidshift - goCue[ti]
interp_fn = interp1d(aln, trial_me, kind='linear',
                     bounds_error=False, fill_value=np.nan)
me_data[:, ti] = fill_nan_nearest(interp_fn(TIME_AXIS))
```

iii. Same alignment approach as other camera signals.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) Low-confidence DLC positions are filled with nearest valid positions before computing velocity (`fill_positions_nearest`). (2) If all positions are NaN, velocity is set to zero. (3) After interpolation to neural time bins, NaN values are filled with the nearest non-NaN value (`fill_nan_nearest`). (4) If motion energy frame times don't match, a synthetic 400 Hz axis is used. (5) Exceptions during DLC or motion energy processing are caught with try/except and silently pass, leaving zeros.

ii.
```python
def fill_positions_nearest(x, y, valid_mask):
    ...
def fill_nan_nearest(arr):
    nans = np.isnan(arr)
    if np.all(nans):
        arr[:] = 0
        return arr
    ...

try:
    ...
except Exception:
    pass
```

iii. The agent chose to fill missing data rather than leave it as a separate category, and used broad exception handling to avoid crashes.

## 11-a. What are the most time-consuming steps of the code?

i. Two main bottlenecks: (1) Reading the MATLAB files, especially the HDF5 format ones. (2) The per-trial, per-neuron spike binning loop, which iterates over all neurons and all trials to histogram spikes individually.

ii.
```python
for ni, spk in enumerate(spike_data):
    for ti in range(n_trials):
        mask = spk['trial'] == (ti + 1)
        aligned = spk['trialtm'][mask] - goCue[ti]
        counts, _ = np.histogram(aligned, bins=EDGES)
        fr[ni, :, ti] = causal_smooth(counts.astype(np.float64) / DT)
```

iii. The agent did not discuss performance optimization in the trajectory.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike counting loop iterates over every neuron and every trial, masking spikes by trial number and histogramming each individually. This could be vectorized using `np.histogram2d` over all trials at once per neuron, as the reference does. The DLC velocity computation also loops per trial.

ii.
```python
for ni, spk in enumerate(spike_data):
    for ti in range(n_trials):
        mask = spk['trial'] == (ti + 1)
        ...
```

iii. The nested loop is O(n_neurons * n_trials * n_spikes) due to the per-trial masking, which is significantly slower than the vectorized approach.

## 11-c. What processing does the code repeat multiple times?

i. The spike binning applies causal smoothing separately for each trial of each neuron. The DLC processing reads and processes each trial's tracking data within a single pass, but the `compute_speed` function is called separately for tongue and paw on each trial, re-reading the same tracking data from the stored dict/closure. The bot_cam trial data is accessed twice per trial (once for tongue, once for paw) when using the `get_trial` closure in v5 format, which re-reads from the scipy struct each time.

ii.
```python
# tongue
t_spd = compute_speed(ts[:, 0, tongue_fi], ts[:, 1, tongue_fi], ft, ...)
# paw - re-accesses same ts, ft
p_spd = compute_speed(ts_p[:, 0, paw_fi], ts_p[:, 1, paw_fi], ft_p, ...)
```

iii. The tracking data is read twice per trial for v5 files via the `get_trial` closure.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code fills all NaN/low-confidence positions with nearest valid values before computing velocity, even though the reference approach keeps NaN positions and uses a "not visible" category for bins without valid data. The `fill_nan_nearest` function is applied after interpolation to neural bins, which fills in values at times where no camera data exists. The code also loads `no_resp` (no-response flag) from the data but doesn't use it, since ignore trials are filtered out by the `(hit|miss)` condition. Additionally, the code processes all `n_trials` for neural data (including early/stim/ignore trials) before selecting only the `trial_idx` subset, wasting computation on trials that are discarded.
