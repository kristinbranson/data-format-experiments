# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from MATLAB `.mat` files (both v5 and HDF5/v7.3 formats) located in two directories: `data/Ephys_Behavior/` and `data/RandomizedDelay_Ephys_Behavior/`. It uses a hardcoded list of 44 session definitions (`SESSION_DEFS`) specifying animal, date, probe numbers, and data directory. Each session file is `data_structure_<animal>_<date>.mat`. Motion energy is loaded from separate `motionEnergy_<animal>_<date>.mat` files. The AI auto-detects the file format (v5 vs HDF5) and uses the appropriate loader.

ii.
```python
SESSION_DEFS = [
    ('EKH1', '2021-08-07', [2], 'ephys'),
    ('EKH3', '2021-08-11', [2], 'ephys'),
    # ... 44 sessions total
]

DATA_DIRS = {
    'ephys': os.path.join(DATA_ROOT, 'Ephys_Behavior'),
    'random': os.path.join(DATA_ROOT, 'RandomizedDelay_Ephys_Behavior'),
}

def _detect_file_format(data_fn):
    try:
        f = h5py.File(data_fn, 'r')
        f.close()
        return 'hdf5'
    except Exception:
        return 'v5'
```

iii. The AI derived the session list from the reference code's loading scripts in `DataLoadingScripts/Recording and video/`. Sessions without corresponding data files (JEB4, JEB5) were excluded. Three data files without loading scripts (JEB23_2023-10-20, JEB24_2023-10-03/04) were also excluded.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the animal name from the session definitions. The AI collects all unique animal names across loaded sessions, sorts them alphabetically, and creates a mapping from animal name to index.

ii.
```python
all_animals = sorted(set(s['animal'] for s in all_sessions))
animal_to_idx = {a: i for i, a in enumerate(all_animals)}
# ...
subject_idx.append(animal_to_idx[sess['animal']])
```

iii. The AI identified 14 unique subjects across the 44 sessions. The paper reports 9 mice for DR and 4 for randomized delay (13 unique). The AI noted the extra subject likely comes from a DR animal not explicitly mentioned in the paper but present in the data.

## 1-c. How are the data split into sessions?

i. Each `.mat` file corresponds to one recording session. The AI hardcodes 44 sessions: 25 from `Ephys_Behavior` and 19 from `RandomizedDelay_Ephys_Behavior`, matching the session loading scripts in the reference code.

ii.
```python
for i, (anm, date, probes, ddir) in enumerate(session_defs):
    result = load_session(anm, date, probes, ddir, PARAMS)
    if result is not None:
        all_sessions.append(result)
```

iii. The session counts (25 DR + 19 randomized) match the paper exactly. Sessions are processed sequentially, one file per session.

## 1-d. How are the data split into trials?

i. Trials are identified from `obj.bp.Ntrials` in each session's data file. Spike data includes trial membership via `obj.clu.trial`. Behavioral variables (hit, miss, R, L, etc.) are arrays of length Ntrials.

ii.
```python
bp['Ntrials'] = int(bp_group['Ntrials'][()].flat[0])
ntrials = bp['Ntrials']
# Each behavioral variable truncated to ntrials:
for field in ['hit', 'miss', 'no', 'early', 'L', 'R', 'autowater']:
    bp[field] = bp_group[field][()].flatten().astype(np.float64)[:ntrials]
```

iii. The AI uses the session's reported trial count to consistently index all per-trial arrays.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters out early lick trials (`early == 1`) and photostimulation trials (`stim.enable == 1`). All other trials are retained, including hit, miss, and ignore/no-response trials.

ii.
```python
trial_mask = (bp['early'] == 0) & (bp['stim_enable'] == 0)
valid_trials = np.where(trial_mask)[0] + 1  # 1-indexed
```

iii. The AI justified including all non-early, non-stim trials because the decoder needs to predict outcome (correct/incorrect), requiring exposure to both hit and miss trials. However, the AI did not explicitly exclude ignore/no-response trials, which the paper states "were omitted from all analyses."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `obj.clu` — the sorted spike cluster data. Specifically, `trialtm` (spike times within trial) and `trial` (trial membership) for each unit, from the specified probe(s).

ii.
```python
# For HDF5:
trialtm = f[trialtm_refs[i]][()].flatten().astype(np.float64)
trial = f[trial_refs[i]][()].flatten().astype(np.float64)
# For v5:
trialtm = unit['trialtm'].flatten().astype(np.float64)
trial = unit['trial'].flatten().astype(np.float64)
```

iii. This matches the reference code which accesses `obj.clu{prbnum}(clunum).trialtm` and `obj.clu{prbnum}(clunum).trial`.

## 2-b. How is the `neural` data processed?

i. The AI aligns spike times to the go cue, bins spikes into 10ms bins over [-2.5, 2.5]s, converts to firing rate (spks/s by dividing by dt), then applies causal Gaussian smoothing with window=15 bins.

ii.
```python
edges = np.arange(params['tmin'], params['tmax'] + params['dt'], params['dt'])
time_axis = edges[:-1] + params['dt'] / 2
# For each neuron and trial:
spk_times = spike_tm[spk_mask] - align_times[trial_num - 1]
counts, _ = np.histogram(spk_times, bins=edges)
fr = causal_gaussian_smooth(counts.astype(np.float64) / params['dt'],
                           params['smooth_window'],
                           params['smooth_bctype'])
```

iii. Parameters match the WorkingWithDataObjs.m tutorial: `dt=1/100` (10ms), `smooth=15`, `bctype='reflect'`, aligned to goCue, time window [-2.5, 2.5]s.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage neuron filtering: (1) Quality filter excludes neurons labeled 'garbage', 'noisy', 'gabrga', 'real?', or with empty/NaN quality. (2) Low firing rate filter removes neurons with mean FR <= 1.0 Hz. Sessions with fewer than 10 remaining units are dropped.

ii.
```python
quality_exclude = {'garbage', 'noisy', 'gabrga', 'real?'}
keep_mask = np.array([q not in quality_exclude and q != '' and q != 'nan'
                      for q in all_qualities])
# After binning:
mean_fr = np.mean(trialdat, axis=(1, 2))
fr_mask = mean_fr > params['low_fr']
trialdat = trialdat[fr_mask, :, :]
```

iii. Quality labels match `findClusters.m` ('all' mode excludes garbage, noisy, gabrga, real?). The lowFR=1.0 Hz matches the tutorial (not getDefaultParams.m which uses 0.5 Hz). However, the FR is computed from mean of single-trial data (`trialdat`), whereas the reference code computes from the condition-averaged `psth`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time is aligned to the go cue by subtracting the trial's goCue event time from the spike's within-trial time.

ii.
```python
align_times = ev[params['align_event']]  # 'goCue'
# Per neuron, per trial:
spk_times = spike_tm[spk_mask] - align_times[trial_num - 1]
counts, _ = np.histogram(spk_times, bins=edges)
```

iii. This matches the reference `alignSpikes.m` which creates `trialtm_aligned = trialtm - ev.goCue(trial)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 10ms (dt=1/100). Spikes are directly binned at this resolution; no further rebinning is applied after the initial binning.

ii.
```python
PARAMS = {
    'dt': 1.0 / 100,  # 10 ms bins
    # ...
}
edges = np.arange(params['tmin'], params['tmax'] + params['dt'], params['dt'])
```

iii. The 10ms bin matches `params.dt = 1/100` in WorkingWithDataObjs.m. The reference `getDefaultParams.m` uses `dt=1/200` (5ms), but the tutorial overrides this. No rebinning is performed.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is derived from the time axis itself, which is constructed from the binning parameters (tmin, tmax, dt). It is not derived from a raw data variable but from the processing parameters.

ii.
```python
time_axis = edges[:-1] + params['dt'] / 2
# Per trial:
time_input = time_axis.astype(np.float32).reshape(1, -1)
sess_input.append(time_input)
```

iii. The time axis runs from -2.495 to 2.495 seconds relative to goCue, with 500 time points at 10ms resolution. Every trial gets the same input vector.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing is needed — the time from go cue is simply the time axis constructed during spike binning, which is centered on the go cue event (time=0).

ii.
```python
time_input = time_axis.astype(np.float32).reshape(1, -1)
```

iii. Since neural data is aligned to goCue, the time axis directly represents time from go cue onset.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis is identical to the neural data time axis — both use the same `time_axis` array. They are inherently aligned.

ii.
```python
# Same time_axis used for both neural binning and input
time_axis = edges[:-1] + params['dt'] / 2
# Neural: np.histogram(spk_times, bins=edges)
# Input: time_input = time_axis.reshape(1, -1)
```

iii. Perfect alignment because both derive from the same edge/bin definitions.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `obj.bp.R` (right lick indicator), a binary per-trial variable.

ii.
```python
lick_direction = bp['R'][valid_trial_indices].copy()
```

iii. In the raw data, `bp.R=1` for right trials and `bp.R=0` for left trials. This maps directly to the decoder specification (left=0, right=1).

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. No processing — the raw `bp.R` value is used directly. The value is broadcast across all time bins to create a (1, n_timebins) array for each trial.

ii.
```python
lick_dir = np.full((1, n_timebins), int(sess['lick_direction'][t]), dtype=np.int64)
```

iii. R=1 maps to right=1, R=0 maps to left=0, matching the decoder specification.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj.bp.autowater`, which indicates whether water was auto-delivered (WC task) or not (DR task).

ii.
```python
behavioral_context = 1.0 - bp['autowater'][valid_trial_indices]
```

iii. `autowater=1` indicates WC (water-cued) trials; `autowater=0` indicates DR (delayed-response) trials.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The autowater field is inverted: `1 - autowater`. This maps autowater=1 (WC) to 0 and autowater=0 (DR) to 1, matching the decoder specification (WC=0, DR=1). The value is broadcast across time bins.

ii.
```python
behavioral_context = 1.0 - bp['autowater'][valid_trial_indices]
context = np.full((1, n_timebins), int(sess['behavioral_context'][t]), dtype=np.int64)
```

iii. The mapping WC=0, DR=1 matches the decoder task specification.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit`, a binary per-trial variable indicating correct responses.

ii.
```python
outcome = bp['hit'][valid_trial_indices].copy()
```

iii. `hit=1` for correct trials, `hit=0` for incorrect or ignore trials.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. No processing — `bp.hit` maps directly to the decoder specification (incorrect=0, correct=1). The value is broadcast across time bins.

ii.
```python
outc = np.full((1, n_timebins), int(sess['outcome'][t]), dtype=np.int64)
```

iii. For ignore/no-response trials (where both hit=0 and miss=0), the outcome is 0 (incorrect).

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from `obj.traj{2}` (bottom camera) DLC tracking data for the `top_tongue` feature. Specifically, the x,y position coordinates (`ts`) and video frame times (`frameTimes`) are used.

ii.
```python
tongue_vel = _compute_feature_velocity_generic(
    traj_data, ntrials, view=2, feat_name='top_tongue',
    vidshift=vidshift, align_times=align_times,
    time_axis=time_axis, is_tongue=True)
```

iii. The reference code uses bottom camera features for tongue tracking. The `top_tongue` feature from the bottom camera corresponds to the tip of the tongue as viewed from below.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Processing steps: (1) Extract x,y positions of `top_tongue` from DLC data. (2) Align video frame times using vidshift and goCue time. (3) Interpolate positions to neural time axis. (4) Compute velocity as gradient of position (separately for x and y). (5) Set NaN velocities to 0 (tongue not visible). (6) Compute Euclidean speed: `sqrt(xvel^2 + yvel^2)`.

ii.
```python
# Interpolate position to neural time axis
xpos = interp1d(ft_a[valid], x_r[valid], kind='linear',
                bounds_error=False, fill_value=np.nan)(time_axis)
ypos = interp1d(ft_a[valid], y_r[valid], kind='linear',
                bounds_error=False, fill_value=np.nan)(time_axis)
# Compute velocity
xvel = np.gradient(xpos)
yvel = np.gradient(ypos)
# Tongue: NaN → 0
if is_tongue:
    xvel[np.isnan(xvel)] = 0
    yvel[np.isnan(yvel)] = 0
# Euclidean speed
speed[:, trix] = np.sqrt(xvel**2 + yvel**2).astype(np.float32)
```

iii. The processing matches the reference `findPosition.m` and `findVelocity.m` for tongue: position not smoothed, missing values not filled with nearest (unlike non-tongue features), NaN velocity set to 0 when tongue is not visible. Note: the reference does NOT subtract baseline velocity for tongue features, and the AI matches this.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The 50th percentile (median) of all non-NaN tongue velocity values across the session is used as the threshold. Values >= threshold are 1 ("high"), < threshold are 0 ("low"). NaN values default to 0.

ii.
```python
def _discretize_velocity(data, n_timebins, n_trials):
    valid_vals = data[~np.isnan(data)]
    threshold = np.percentile(valid_vals, 50)
    disc = (data >= threshold).astype(np.int64)
    disc[np.isnan(data)] = 0
    return disc
```

iii. This follows the decoder task specification: "discretized into two bins with per-session threshold: 0: < 50th percentile, 1: >= 50th percentile."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Video frame times are corrected by subtracting vidshift (video-neural timing offset) and the trial's goCue time. The position is then interpolated to the neural time axis using linear interpolation.

ii.
```python
ft_aligned = ft - vidshift - align_times[trix]
xpos = interp1d(ft_aligned[valid], x_r[valid], kind='linear',
                bounds_error=False, fill_value=np.nan)(time_axis)
```

iii. This matches `findPosition.m`: `interp1(traj(trix).frameTimes - vidshift - obj.bp.ev.(alignEv)(trix), ts, taxis)`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from `obj.traj{2}` (bottom camera) DLC tracking data for the `top_paw` feature.

ii.
```python
paw_vel = _compute_feature_velocity_generic(
    traj_data, ntrials, view=2, feat_name='top_paw',
    vidshift=vidshift, align_times=align_times,
    time_axis=time_axis, is_tongue=False)
```

iii. The reference code tracks paw position from the bottom camera. Only `top_paw` is used (not `bottom_paw`).

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Processing steps: (1) Extract x,y positions from DLC data. (2) Align and interpolate to neural time axis (same as tongue). (3) Fill missing positions with nearest value. (4) Compute velocity via gradient. (5) Subtract baseline drift velocity (median of position differences). (6) Fill NaN velocities with nearest value. (7) Compute Euclidean speed.

ii.
```python
if not is_tongue:
    xpos = _fill_nearest(xpos)
    ypos = _fill_nearest(ypos)
# ...
xvel = np.gradient(xpos)
yvel = np.gradient(ypos)
# Subtract baseline drift
base_xvel = np.nanmedian(np.diff(xpos))
base_yvel = np.nanmedian(np.diff(ypos))
xvel -= (base_xvel if not np.isnan(base_xvel) else 0)
yvel -= (base_yvel if not np.isnan(base_yvel) else 0)
xvel = _fill_nearest(xvel)
yvel = _fill_nearest(yvel)
speed[:, trix] = np.sqrt(xvel**2 + yvel**2).astype(np.float32)
```

iii. The processing matches the reference `findVelocity.m` pattern: position filled with nearest, baseline subtracted, NaN velocities filled with nearest. However, there is a difference: the reference code subtracts `basederiv(1)` (x-baseline) from BOTH xvel and yvel (lines 13-14 of findVelocity.m), while the AI correctly subtracts x-baseline from xvel and y-baseline from yvel. This appears to be a bug in the reference code that the AI "fixed."

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue velocity: 50th percentile per session, values >= threshold are 1, < threshold are 0.

ii.
```python
paw_vel_disc = _discretize_velocity(paw_vel, n_timebins, n_valid_trials)
```

iii. Matches decoder task specification.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same alignment method as tongue velocity: video frame times corrected by vidshift and goCue time, then interpolated to neural time axis.

ii.
```python
ft_aligned = ft - vidshift - align_times[trix]
xpos = interp1d(ft_aligned[valid], x_r[valid], ...)(time_axis)
```

iii. Matches reference alignment approach.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from separate `motionEnergy_<animal>_<date>.mat` files. The raw data is a cell array (`me.data`) containing per-trial motion energy time series captured at 400 Hz (same as video frame rate).

ii.
```python
me_fn = os.path.join(data_dir, f'motionEnergy_{anm}_{date}.mat')
# Load via scipy.io or h5py:
me_struct = me_file['me']
me_cell = me_struct['data'].item()
```

iii. Matches `loadMotionEnergy.m` which loads from `motionEnergy*.mat` files.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Processing steps: (1) Load per-trial ME arrays from .mat file. (2) Get frame times from side camera (view 1) trajectory data. (3) Align frame times by subtracting vidshift and goCue time. (4) Interpolate ME to neural time axis using linear interpolation. (5) Fill remaining NaN values with nearest non-NaN value.

ii.
```python
ft_aligned = ft - vidshift - align_times[trix]
me_aligned[:, trix] = interp1d(ft_aligned[valid], me_trial[valid],
                               kind='linear', bounds_error=False,
                               fill_value=np.nan)(time_axis).astype(np.float32)
# Fill NaNs:
me_aligned[:, trix] = _fill_nearest(col)
```

iii. This matches `loadMotionEnergy.m` which interpolates ME using `interp1(frameTimes - vidshift - alignTimes(trix), me.data{trix}, taxis)` and then fills NaN with `fillmissing(..., 'nearest')`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same discretization as tongue and paw velocity: 50th percentile per session, >= threshold is 1, < threshold is 0.

ii.
```python
me_disc = _discretize_velocity(me_data, n_timebins, n_valid_trials)
```

iii. Matches decoder task specification.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. ME is aligned using side camera frame times, corrected by vidshift and goCue time, then interpolated to the neural time axis. If frame times are unavailable, synthetic frame times are generated at 400 Hz.

ii.
```python
side_cam = traj_data['views'][0]  # side cam for frame times
ft = side_cam['trials'][trix]['frameTimes']
ft_aligned = ft - vidshift - align_times[trix]
me_aligned[:, trix] = interp1d(ft_aligned[valid], me_trial[valid], ...)(time_axis)
```

iii. Matches reference: `loadMotionEnergy.m` uses `obj.traj{1}(trix).frameTimes` (side cam) for ME alignment.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several categories of missing/problematic data:
- **Missing DLC detections (tongue)**: NaN positions → velocity set to 0 when tongue not visible
- **Missing DLC detections (non-tongue)**: Positions filled with nearest non-NaN value before computing velocity
- **Missing frame times**: Synthetic frame times generated at 400 Hz
- **Corrupted ME files**: Graceful fallback to all-zero ME (4 JEB23 sessions)
- **All-zero neural data**: Retained as-is (61/13823 trials in late sessions)
- **Missing data files**: Sessions skipped with warning
- **NaN in interpolated data**: Filled with nearest non-NaN value

ii.
```python
# Tongue velocity NaN handling:
xvel[np.isnan(xvel)] = 0
yvel[np.isnan(yvel)] = 0
# Non-tongue fill nearest:
xpos = _fill_nearest(xpos)
# ME fallback for corrupted files:
if data is None:
    return np.zeros((n_timebins, n_trials), dtype=np.int64)
# Missing frame times:
if ft is None or ft.size == 0:
    ft = np.arange(1, me_trial.size + 1) / 400.0
```

iii. The approaches match the reference code patterns: `fillmissing(..., 'nearest')` for non-tongue features, tongue NaN → 0, synthetic frame times as fallback.

## 11-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the nested loop over neurons and trials for spike binning and smoothing (lines 460-473 of convert_data.py). For each neuron and each trial, the code: finds matching spikes, computes histogram, applies Gaussian smoothing. This is O(n_neurons * n_trials) per session.

ii.
```python
for neuron_idx, clu_idx in enumerate(good_indices):
    spike_tm = all_spike_times[clu_idx]
    spike_trial = all_spike_trials[clu_idx]
    for t_idx, trial_num in enumerate(valid_trials):
        spk_mask = spike_trial == trial_num
        spk_times = spike_tm[spk_mask] - align_times[trial_num - 1]
        counts, _ = np.histogram(spk_times, bins=edges)
        fr = causal_gaussian_smooth(counts.astype(np.float64) / params['dt'],
                                   params['smooth_window'], params['smooth_bctype'])
```

iii. The AI estimated ~5s per session, ~220s total for 44 sessions. This is within acceptable limits.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops could be vectorized:
1. The neuron x trial spike binning loop (lines 460-473): Trial selection (`spike_trial == trial_num`) could be vectorized using numpy advanced indexing or `np.searchsorted`.
2. The `causal_gaussian_smooth` function loops over columns (line 188-189): `np.convolve` is called per column instead of using `scipy.signal.fftconvolve` or `np.apply_along_axis`.

ii.
```python
# Loop over columns in smoothing:
for j in range(x_filt.shape[1]):
    out[:, j] = np.convolve(x_filt[:, j], kern, mode='same')
```

iii. The AI acknowledged efficiency concerns but kept the iterative approach since total runtime was ~3 minutes.

## 11-c. What processing does the code repeat multiple times?

i. Several processing steps are repeated:
1. The smoothing kernel is reconstructed on every call to `causal_gaussian_smooth` (same parameters each time).
2. Video offset (`vidshift`) computation is performed per session (same calculation pattern).
3. The `_fill_nearest` function is called multiple times per trial for different signals.
4. Format detection (`_detect_file_format`) performs a trial open of each file.

ii.
```python
# Kernel rebuilt every call:
def causal_gaussian_smooth(x, N, bctype='reflect'):
    kern = np.array(sig_windows.gaussian(N, std=(N - 1) / (2 * 2.5)))
    kern[:N // 2] = 0
    kern = kern / kern.sum()
```

iii. The repeated kernel computation is wasteful but has negligible performance impact since the kernel is small.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores raw (continuous) velocity and motion energy data (`tongue_vel_raw`, `paw_vel_raw`, `me_raw`) per session, but only the discretized versions are included in the final output. These raw values are only used for processing plots in `--show-processing` mode.

ii.
```python
# Raw values stored but not in final output:
'tongue_vel_raw': tongue_vel,
'paw_vel_raw': paw_vel,
'me_raw': me_data,
# Only discretized versions go into final output:
tongue_v = sess['tongue_vel_disc'][:, t].astype(np.int64).reshape(1, -1)
```

iii. This is a minor inefficiency — the raw values consume memory during processing but are discarded when building the final pickle. They serve a useful purpose for diagnostic plots.
