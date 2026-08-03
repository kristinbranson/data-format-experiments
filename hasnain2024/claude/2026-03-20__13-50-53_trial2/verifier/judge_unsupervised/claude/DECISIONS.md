# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hardcodes a list of 44 session definitions (SESSION_DEFS) as tuples of (animal, date, probes, data_dir_key). It iterates through each session, loading the corresponding `data_structure_<animal>_<date>.mat` file from either `Ephys_Behavior/` or `RandomizedDelay_Ephys_Behavior/` directories. It auto-detects HDF5 vs MATLAB v5 format and uses separate loading functions for each. Motion energy is loaded from separate `motionEnergy_<animal>_<date>.mat` files.

ii.
```python
SESSION_DEFS = [
    ('EKH1', '2021-08-07', [2], 'ephys'),
    # ... 44 sessions total
    ('JEB24', '2023-11-03', [1], 'random'),
]

DATA_DIRS = {
    'ephys': os.path.join(DATA_ROOT, 'Ephys_Behavior'),
    'random': os.path.join(DATA_ROOT, 'RandomizedDelay_Ephys_Behavior'),
}

for i, (anm, date, probes, ddir) in enumerate(session_defs):
    result = load_session(anm, date, probes, ddir, PARAMS)
```

iii. The AI constructed SESSION_DEFS by examining the loading scripts in `DataLoadingScripts/Recording and video/` which specify which sessions and probes to load. Sessions without corresponding data files (JEB4, JEB5) were excluded. Sessions with data files but no loading scripts (JEB23_2023-10-20, JEB24_2023-10-03/04) were also excluded.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the animal name field from each session definition. After loading, unique animal names are sorted and each session is assigned a `subject_idx` mapping to the sorted list. The AI found 14 unique subjects (10 from Ephys_Behavior, 4 from RandomizedDelay).

ii.
```python
all_animals = sorted(set(s['animal'] for s in all_sessions))
animal_to_idx = {a: i for i, a in enumerate(all_animals)}
subject_idx.append(animal_to_idx[sess['animal']])
```

iii. The AI noted the paper reports 9 DR mice and 4 randomized mice (13 unique), while it found 14 unique subjects. The discrepancy was attributed to one extra DR animal not filtered by the paper's criteria. The AI chose to include all animals with data, letting the per-session filtering decide inclusion.

## 1-c. How are the data split into sessions?

i. Each .mat file corresponds to one session. The AI processes 44 sessions total (25 Ephys_Behavior + 19 RandomizedDelay). Sessions are iterated in SESSION_DEFS order. Sessions that fail loading or have too few units after filtering are skipped.

ii.
```python
for i, (anm, date, probes, ddir) in enumerate(session_defs):
    result = load_session(anm, date, probes, ddir, PARAMS)
    if result is not None:
        all_sessions.append(result)
```

iii. The session list matches the 25 DR + 19 randomized sessions described in the paper and defined in the reference loading scripts.

## 1-d. How are the data split into trials?

i. Within each session, the total number of trials is read from `obj.bp.Ntrials`. All spike data, behavioral data, and trajectory data are indexed by trial number. The AI uses 1-indexed trial numbers (matching MATLAB convention) for spike filtering, then converts to 0-indexed for array access.

ii.
```python
bp['Ntrials'] = int(bp_group['Ntrials'][()].flat[0])
ntrials = bp['Ntrials']
# ...
valid_trials = np.where(trial_mask)[0] + 1  # 1-indexed
# For spike binning:
for t_idx, trial_num in enumerate(valid_trials):
    spk_mask = spike_trial == trial_num
```

iii. Trial splitting follows the data structure where each session has Ntrials trials with associated behavioral variables and spike data.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to exclude early lick trials (`early==0`) and photostimulation trials (`stim_enable==0`). No filtering by outcome (hit/miss/no) - all trial outcomes are retained. Sessions with fewer than 2 valid trials are skipped.

ii.
```python
trial_mask = (bp['early'] == 0) & (bp['stim_enable'] == 0)
valid_trials = np.where(trial_mask)[0] + 1  # 1-indexed
if len(valid_trials) < 2:
    print(f"  WARNING: {session_id} has < 2 valid trials, skipping")
    return None
```

iii. The AI followed the reference code's trial filtering conventions (`~early`, `~stim.enable`). It included all outcome types because the decoder needs to predict outcome, requiring both correct and incorrect trials. The reference code's condition(1) in WorkingWithDataObjs.m is `'(hit|miss|no)'` for all trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the spike times in `obj.clu`. For each unit (cluster), `trialtm` provides spike times relative to trial start, and `trial` provides the trial number for each spike.

ii.
```python
# From HDF5 loading:
trialtm = f[trialtm_refs[i]][()].flatten().astype(np.float64)
trial = f[trial_refs[i]][()].flatten().astype(np.float64)
```

iii. This matches the reference code's `alignSpikes.m` which uses `obj.clu{prbnum}(clu).trialtm` and aligns to the specified event.

## 2-b. How is the `neural` data processed?

i. The processing pipeline follows the reference code: (1) filter clusters by quality, (2) align spikes to goCue, (3) bin spikes in 10ms bins from -2.5 to 2.5s, (4) convert to firing rate (Hz) by dividing by dt, (5) smooth with causal Gaussian kernel (window=15), (6) remove neurons with mean FR < 1 Hz.

ii.
```python
# Align to goCue
spk_times = spike_tm[spk_mask] - align_times[trial_num - 1]
# Bin
counts, _ = np.histogram(spk_times, bins=edges)
# Convert to FR and smooth
fr = causal_gaussian_smooth(counts.astype(np.float64) / params['dt'],
                           params['smooth_window'], params['smooth_bctype'])
```

iii. The AI documented following the pipeline: loadObjs -> findClusters -> alignSpikes -> getSeq -> removeLowFRClusters, matching the reference code tutorial.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filtering steps: (1) Quality filter: exclude clusters with quality labels 'garbage', 'noisy', 'gabrga', 'real?', or empty strings. (2) Low firing rate filter: remove neurons with mean FR across all timepoints and trials < 1 Hz. Sessions with fewer than 10 units after filtering are excluded.

ii.
```python
quality_exclude = {'garbage', 'noisy', 'gabrga', 'real?'}
keep_mask = np.array([q not in quality_exclude and q != '' and q != 'nan'
                      for q in all_qualities])
# ...
mean_fr = np.mean(trialdat, axis=(1, 2))
fr_mask = mean_fr > params['low_fr']
```

iii. Quality labels match `findClusters.m` with quality='all'. The 1 Hz threshold matches the WorkingWithDataObjs.m tutorial. The min 10 units threshold matches the paper's statement.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue by subtracting the goCue event time for each trial from each spike's trial-relative time. Then spikes are binned in a window of [-2.5, 2.5] seconds relative to goCue.

ii.
```python
align_times = ev[params['align_event']]  # 'goCue'
edges = np.arange(params['tmin'], params['tmax'] + params['dt'], params['dt'])
# ...
spk_times = spike_tm[spk_mask] - align_times[trial_num - 1]
counts, _ = np.histogram(spk_times, bins=edges)
```

iii. This matches the reference `alignSpikes.m` which computes `trialtm_aligned = trialtm - event` where event is the goCue time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 10 ms (dt = 1/100 s). This produces 500 time bins for the 5-second window [-2.5, 2.5]s. No temporal rebinning is applied after the initial binning.

ii.
```python
PARAMS = {
    'dt': 1.0 / 100,  # 10 ms bins
    'tmin': -2.5,
    'tmax': 2.5,
}
edges = np.arange(params['tmin'], params['tmax'] + params['dt'], params['dt'])
time_axis = edges[:-1] + params['dt'] / 2
```

iii. This matches the WorkingWithDataObjs.m tutorial: `params.dt = 1/100`. The metadata stores `time_bin_size: 10.0` (ms).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not derived from raw data variables directly. It is the time axis used for binning, which represents time from go cue onset (the alignment event).

ii.
```python
time_input = time_axis.astype(np.float32).reshape(1, -1)
sess_input.append(time_input)
```

iii. The time axis is computed from the bin edges and represents the center of each 10ms bin relative to goCue onset, ranging from -2.495 to 2.495 seconds.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is computed as the center of each time bin: `edges[:-1] + dt/2`. This is a fixed array that is the same for every trial and session.

ii.
```python
edges = np.arange(params['tmin'], params['tmax'] + params['dt'], params['dt'])
time_axis = edges[:-1] + params['dt'] / 2
# ...
time_input = time_axis.astype(np.float32).reshape(1, -1)
```

iii. No special processing needed - this is a deterministic time vector.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input uses the exact same time axis as the neural data, so alignment is implicit. Both share the same 500-timepoint axis.

ii.
```python
# Neural uses same edges/time_axis for binning
# Input reuses time_axis directly
time_input = time_axis.astype(np.float32).reshape(1, -1)
```

iii. Since neural data is aligned to goCue and binned on the same time axis, the input time vector is inherently aligned.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `obj.bp.R` (right lick indicator).

ii.
```python
lick_direction = bp['R'][valid_trial_indices].copy()
```

iii. The reference code uses `obj.bp.R` and `obj.bp.L` as binary indicators. R=1 means right trial, L=1 means left trial.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The R field is used directly: R=1 maps to right=1, R=0 maps to left=0. The value is broadcast to all timepoints as a per-trial constant.

ii.
```python
lick_direction = bp['R'][valid_trial_indices].copy()
# Later:
lick_dir = np.full((1, n_timebins), int(sess['lick_direction'][t]), dtype=np.int64)
```

iii. The instructions specify left=0, right=1, which matches using bp.R directly.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj.bp.autowater`.

ii.
```python
behavioral_context = 1.0 - bp['autowater'][valid_trial_indices]
```

iii. The reference code and paper describe autowater=1 as water-cued (WC) trials and autowater=0 as delayed-response (DR) trials.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The autowater field is inverted: `1 - autowater`. This maps autowater=1 (WC) to 0, and autowater=0 (DR) to 1. The value is broadcast to all timepoints.

ii.
```python
behavioral_context = 1.0 - bp['autowater'][valid_trial_indices]
# Later:
context = np.full((1, n_timebins), int(sess['behavioral_context'][t]), dtype=np.int64)
```

iii. The instructions specify WC=0, DR=1, which matches 1-autowater.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit`.

ii.
```python
outcome = bp['hit'][valid_trial_indices].copy()
```

iii. The hit field is a binary indicator of correct responses.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The hit field is used directly: hit=1 maps to correct=1, hit=0 maps to incorrect=0. The value is broadcast to all timepoints.

ii.
```python
outcome = bp['hit'][valid_trial_indices].copy()
# Later:
outc = np.full((1, n_timebins), int(sess['outcome'][t]), dtype=np.int64)
```

iii. The instructions specify incorrect=0, correct=1. Note that hit=0 includes both miss trials (wrong lick) and no-response trials (no lick), both mapped to incorrect=0.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from DLC trajectories in `obj.traj{2}` (bottom camera, view index 2 in 1-based). Specifically, the 'top_tongue' feature's x,y position time series and associated frame times.

ii.
```python
tongue_vel = _compute_feature_velocity_generic(
    traj_data, ntrials, view=2, feat_name='top_tongue',
    vidshift=vidshift, align_times=align_times,
    time_axis=time_axis, is_tongue=True)
```

iii. The reference code uses bottom camera features for tongue tracking. The AI selected 'top_tongue' from the bottom camera's feature list.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Processing: (1) Extract x,y position from DLC trajectory for the 'top_tongue' feature. (2) Adjust frame times by subtracting video offset and goCue alignment time. (3) Interpolate positions to the neural time axis. (4) Compute velocity via `np.gradient()` for both x and y. (5) For tongue, NaN velocities are set to 0 (tongue not visible = not moving). (6) Compute Euclidean speed: `sqrt(xvel^2 + yvel^2)`.

ii.
```python
xpos = interp1d(ft_a[valid], x_r[valid], kind='linear',
                bounds_error=False, fill_value=np.nan)(time_axis)
ypos = interp1d(ft_a[valid], y_r[valid], kind='linear',
                bounds_error=False, fill_value=np.nan)(time_axis)
xvel = np.gradient(xpos)
yvel = np.gradient(ypos)
if is_tongue:
    xvel[np.isnan(xvel)] = 0
    yvel[np.isnan(yvel)] = 0
speed[:, trix] = np.sqrt(xvel**2 + yvel**2).astype(np.float32)
```

iii. This follows the reference code pipeline: findPosition -> findVelocity. The tongue-specific NaN handling (set to 0) matches `findVelocity.m`.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Discretized using the 50th percentile of all valid (non-NaN) values across all timepoints and trials within each session. Values >= threshold map to 1 ("high"), values < threshold map to 0 ("low"). NaN entries are set to 0.

ii.
```python
def _discretize_velocity(data, n_timebins, n_trials):
    valid_vals = data[~np.isnan(data)]
    threshold = np.percentile(valid_vals, 50)
    disc = (data >= threshold).astype(np.int64)
    disc[np.isnan(data)] = 0
    return disc
```

iii. The instructions specify 50th percentile per-session threshold with 0 = below, 1 = above.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue velocity is aligned by interpolating the DLC position data (at ~400 Hz) to the neural time axis (10ms bins) using linear interpolation, after adjusting frame times for the video offset and goCue alignment.

ii.
```python
ft_aligned = ft - vidshift - align_times[trix]
xpos = interp1d(ft_a[valid], x_r[valid], kind='linear',
                bounds_error=False, fill_value=np.nan)(time_axis)
```

iii. This matches the reference code's approach in `findPosition.m`: `interp1(traj(trix).frameTimes-vidshift-obj.bp.ev.(alignEv)(trix), ts, taxis)`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from DLC trajectories in `obj.traj{2}` (bottom camera), specifically the 'top_paw' feature.

ii.
```python
paw_vel = _compute_feature_velocity_generic(
    traj_data, ntrials, view=2, feat_name='top_paw',
    vidshift=vidshift, align_times=align_times,
    time_axis=time_axis, is_tongue=False)
```

iii. The reference code extracts paw kinematics from the bottom camera.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Processing: (1) Extract x,y position for 'top_paw'. (2) Adjust frame times. (3) Interpolate to neural time axis. (4) Fill NaN positions with nearest values. (5) Compute velocity via gradient. (6) Subtract baseline derivative (median of diff) from velocity. (7) Fill NaN velocities with nearest values. (8) Compute Euclidean speed.

ii.
```python
if not is_tongue:
    xpos = _fill_nearest(xpos)
    ypos = _fill_nearest(ypos)
xvel = np.gradient(xpos)
yvel = np.gradient(ypos)
if not is_tongue:
    base_xvel = np.nanmedian(np.diff(xpos))
    base_yvel = np.nanmedian(np.diff(ypos))
    xvel -= (base_xvel if not np.isnan(base_xvel) else 0)
    yvel -= (base_yvel if not np.isnan(base_yvel) else 0)
    xvel = _fill_nearest(xvel)
    yvel = _fill_nearest(yvel)
speed[:, trix] = np.sqrt(xvel**2 + yvel**2).astype(np.float32)
```

iii. This follows `findVelocity.m` which subtracts baseline derivative for non-tongue features and fills missing values with nearest. However, the AI subtracts `base_xvel` from x-velocity and `base_yvel` from y-velocity, while the reference code subtracts `basederiv(1)` (the x baseline) from both x and y velocities.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue velocity: 50th percentile per-session threshold. Values >= threshold = 1, values < threshold = 0.

ii.
```python
paw_vel_disc = _discretize_velocity(paw_vel, n_timebins, n_valid_trials)
```

iii. Follows the instructions for 50th percentile discretization.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same alignment approach as tongue velocity: interpolation of DLC data to neural time axis after video offset correction.

ii.
```python
ft_aligned = ft - vidshift - align_times[trix]
xpos = interp1d(ft_a[valid], x_r[valid], kind='linear',
                bounds_error=False, fill_value=np.nan)(time_axis)
```

iii. Matches the reference code's interpolation approach in `findPosition.m`.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from separate `motionEnergy_<animal>_<date>.mat` files. The variable `me.data` contains per-trial motion energy arrays at 400 Hz.

ii.
```python
me_fn = os.path.join(data_dir, f'motionEnergy_{anm}_{date}.mat')
# ...
me_struct = me_file['me']
me_cell = me_struct['data'].item()
```

iii. This matches the reference `loadMotionEnergy.m` which loads `motionEnergy_*.mat` files.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Processing: (1) Load per-trial ME arrays. (2) Get frame times from side camera (view 1) trajectories. (3) Adjust frame times for video offset and goCue alignment. (4) Interpolate ME to neural time axis using linear interpolation. (5) Fill NaN values with nearest non-NaN value.

ii.
```python
ft_aligned = ft - vidshift - align_times[trix]
me_aligned[:, trix] = interp1d(ft_aligned[valid], me_trial[valid],
                               kind='linear', bounds_error=False,
                               fill_value=np.nan)(time_axis).astype(np.float32)
# Fill NaNs
me_aligned[:, trix] = _fill_nearest(col)
```

iii. This matches `loadMotionEnergy.m`: `interp1(obj.traj{1}(trix).frameTimes-vidshift-alignTimes(trix), me.data{trix}, taxis)` followed by `fillmissing(me.data,'nearest')`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same as tongue/paw velocity: 50th percentile per-session threshold.

ii.
```python
me_disc = _discretize_velocity(me_data, n_timebins, n_valid_trials)
```

iii. Follows the instructions for 50th percentile discretization.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by interpolating from the video frame time axis (corrected for video offset and goCue time) to the neural time axis.

ii.
```python
ft_aligned = ft - vidshift - align_times[trix]
me_aligned[:, trix] = interp1d(ft_aligned[valid], me_trial[valid],
                               kind='linear', bounds_error=False,
                               fill_value=np.nan)(time_axis)
```

iii. The reference code performs the same alignment in `loadMotionEnergy.m`.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled: (1) Missing data files: sessions are skipped with a warning. (2) Missing motion energy: corrupt ME files (4 JEB23 sessions) produce all-zero discretized output. (3) Missing frame times: fallback to generating `np.arange(1, size+1) / 400.0`. (4) NaN values in trajectories: filtered before interpolation, filled afterward with nearest valid value. (5) All-zero neural data: 61 late trials across 2 sessions have zero spikes in the time window. (6) Trials with NaN video data: skipped during trajectory processing.

ii.
```python
# Missing frame times fallback
if ft is None or ft.size == 0 or np.all(np.isnan(ft)):
    ft = np.arange(1, me_trial.size + 1) / 400.0
# NaN filling
me_aligned[:, trix] = _fill_nearest(col)
# Missing ME -> zeros
if data is None:
    return np.zeros((n_timebins, n_trials), dtype=np.int64)
```

iii. The AI documented these edge cases in CONVERSION_NOTES.md and verified they affect a small fraction of the data (0.44% for zero-neural trials).

## 11-a. What are the most time-consuming steps of the code?

i. The main bottleneck is the spike binning and smoothing loop in `load_session`, which iterates over all neurons and trials to bin spikes and apply Gaussian smoothing. The full conversion takes ~163 seconds for 44 sessions.

ii.
```python
for neuron_idx, clu_idx in enumerate(good_indices):
    for t_idx, trial_num in enumerate(valid_trials):
        # ... histogram + smooth per neuron per trial
```

iii. The AI estimated ~5s per session based on sample conversion and tracked timing in the output.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The nested neuron x trial loop for spike binning is the main candidate for vectorization. Each call to `np.histogram` and `causal_gaussian_smooth` is done individually per neuron per trial. This could potentially be vectorized by building a sparse spike matrix and applying binning in bulk.

ii.
```python
for neuron_idx, clu_idx in enumerate(good_indices):
    for t_idx, trial_num in enumerate(valid_trials):
        spk_mask = spike_trial == trial_num
        spk_times = spike_tm[spk_mask] - align_times[trial_num - 1]
        counts, _ = np.histogram(spk_times, bins=edges)
        fr = causal_gaussian_smooth(counts / params['dt'], ...)
```

iii. The per-column convolution in `causal_gaussian_smooth` could also be vectorized using `scipy.ndimage.convolve1d` or FFT-based convolution for the full matrix.

## 11-c. What processing does the code repeat multiple times?

i. The spike masking `spike_trial == trial_num` is computed separately for each neuron-trial combination. For a given trial, this mask could be precomputed once and reused across neurons. Similarly, the Gaussian smoothing kernel is recreated on each call to `causal_gaussian_smooth`.

ii.
```python
# This is computed inside the inner loop for every neuron x trial:
spk_mask = spike_trial == trial_num
# The kernel is rebuilt every call:
kern = np.array(sig_windows.gaussian(N, std=(N - 1) / (2 * 2.5)))
```

iii. Precomputing trial masks per neuron and caching the kernel would avoid redundant computation.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores raw (continuous) tongue velocity, paw velocity, and motion energy data (`tongue_vel_raw`, `paw_vel_raw`, `me_raw`) which are only used for plotting in `--show-processing` mode but are included in the session result dictionary during full conversion. These are not saved to the final pickle file but consume memory during processing. Additionally, it computes kinematic data for all trials (including invalid ones) before subsetting to valid trials.

ii.
```python
# Raw velocities stored in session result but not saved to output
'tongue_vel_raw': tongue_vel,
'paw_vel_raw': paw_vel,
'me_raw': me_data,
# Computed for all ntrials, then subset to valid:
tongue_vel = tongue_vel[:, valid_0idx]
```

iii. Computing kinematics for all trials before filtering is wasteful since only valid trials are needed in the output.
