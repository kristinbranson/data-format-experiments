# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from MATLAB `.mat` files stored in `data/Ephys_Behavior/` and `data/RandomizedDelay_Ephys_Behavior/` directories. Each session's data is stored in a file named `data_structure_{anm}_{date}.mat`. The AI auto-detects the MATLAB file format (v7.3/HDF5 vs v5) and loads using `h5py` or `scipy.io.loadmat` accordingly. Motion energy is loaded from separate `motionEnergy_{anm}_{date}.mat` files. A hardcoded list of 44 sessions (`SESSION_META`) specifies which files to load with which probe number.

ii.
```python
SESSION_META = [
    {'anm': 'EKH1', 'date': '2021-08-07', 'probe': 2, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    ...  # 25 DR sessions + 19 RandDelay sessions = 44 total
]

def load_session(filepath, probe_num):
    """Load session data, auto-detecting file format."""
    try:
        f = h5py.File(filepath, 'r')
        f.close()
        return load_session_h5(filepath, probe_num)
    except:
        return load_session_v5(filepath, probe_num)
```

iii. The AI documented in CONVERSION_NOTES.md that it identified 44 sessions (25 DR + 19 RandDelay) from the reference loading scripts in `code/DataLoadingScripts/Recording and video/`. JEB23 2023-10-20 was excluded because it was commented out in the loading script.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `anm` field in the `SESSION_META` list. Each unique animal name becomes an entry in the `subjects` list. The AI found 14 unique subjects (10 DR mice + 4 RandDelay mice, with no overlap).

ii.
```python
subj = result['subject']
if subj not in all_subjects:
    all_subjects.append(subj)
subject_idx.append(all_subjects.index(subj))
```

iii. The AI noted that the paper says 9 DR mice but data has 10 (EKH1 is included in data but may not be counted in the paper's "nine mice" for DR). This was noted as an acceptable discrepancy.

## 1-c. How are the data split into sessions?

i. Each entry in `SESSION_META` defines one session. Sessions are processed sequentially; each session corresponds to one `.mat` file with one probe. The result is a list of 44 sessions in the `neural`, `input`, and `output` lists.

ii.
```python
for sess_meta in sessions_to_process:
    result = process_session(sess_meta, PARAMS, show_processing=args.show_processing)
    if result is None:
        continue
    all_neural.append(result['neural'])
    all_input.append(result['input'])
    all_output.append(result['output'])
```

iii. The AI determined session definitions from the per-animal loading scripts (e.g., `loadJEB7_ALMVideo.m`) which specify which dates and probes to use for each animal.

## 1-d. How are the data split into trials?

i. Within each session, the total number of trials (`ntrials`) is read from `bp.Ntrials`. Spike data is binned per trial using the cluster's `trial` field (1-indexed trial numbers). Valid trials are then selected based on filtering criteria.

ii.
```python
ntrials = session_data['ntrials']
# ...
for trial_idx in range(ntrials):
    trial_num = trial_idx + 1  # 1-indexed
    spike_mask = trial_nums == trial_num
    # ...
    counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. The AI uses `bp.Ntrials` from the data structure to determine total trial count, matching the reference code's approach.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to exclude: early lick trials (`early`), no-response trials (`no`), stimulation trials (`stim_enable`). Only trials that are hits or misses are kept (`hit | miss`). Additionally, trials with all-zero neural data are removed.

ii.
```python
valid_trial_mask = ~session_data['early'] & ~session_data['no'] & ~session_data['stim_enable']
valid_trial_mask = valid_trial_mask & (session_data['hit'] | session_data['miss'])
# ...
# Later: filter out all-zero neural data
if np.all(ni == 0):
    n_removed += 1
    continue
```

iii. The AI based this filtering on the reference code's `findTrials.m` conditions which use expressions like `R&hit&~stim.enable&~autowater&~early`. However, the reference conditions are for PSTH computation (specific combinations of R/L, hit, autowater), not for trial exclusion from `trialdat`. The `trialdat` in the reference code is computed for ALL trials (1 to Ntrials). The AI's filtering is a reasonable choice for the decoder task but goes beyond what the reference code does for `trialdat`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the cluster spike times: `clu.trialtm` (spike times relative to trial start) and `clu.trial` (trial assignment for each spike), aligned to `bp.ev.goCue` (go cue event times).

ii.
```python
trialtm = clu['trialtm']
trial_nums = clu['trial']
aligned_times = trialtm[spike_mask] - goCue[trial_idx]
```

iii. The AI correctly identified that `clu.trialtm` contains spike times and `clu.trial` contains trial assignments, matching the reference code's `alignSpikes.m`.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to go cue, binned into 5ms bins (dt=1/200), converted to firing rates (counts/dt), and smoothed with a causal Gaussian kernel (window=15). This produces a (n_timepoints, n_neurons, n_trials) array.

ii.
```python
edges = np.arange(tmin, tmax + dt, dt)
counts, _ = np.histogram(aligned_times, bins=edges)
fr = counts.astype(np.float32) / dt
fr_smooth = smooth_causal(fr, params['smooth_window'], bctype='none')
```

iii. The AI documented this as matching `getSeq.m`: bin spikes into 5ms bins, convert to firing rate, smooth with causal Gaussian kernel of window 15.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering: (1) Clusters are filtered by quality label - excluding 'garbage', 'gabrga', 'noisy', 'real?' (empty quality strings are kept). (2) Clusters with mean firing rate <= 0.5 Hz are removed. Sessions with fewer than 10 remaining units are skipped entirely.

ii.
```python
excluded = params['excluded_qualities']  # {'garbage', 'gabrga', 'noisy', 'real?'}
for i, clu in enumerate(session_data['clusters']):
    q = clu['quality'].lower().strip().replace('\x00', '')
    if q in excluded:
        continue
    valid_clusters.append(clu)

# Low FR filter
mean_fr = np.mean(np.mean(trialdat, axis=2), axis=0)
keep_mask = mean_fr > params['low_fr']  # 0.5 Hz
```

iii. Quality filtering matches `findClusters.m` which excludes 'garbage', 'gabrga', 'noisy', 'real?'. The low FR threshold of 0.5 Hz matches `getDefaultParams.m` (`params.lowFR = 0.5`). However, the reference `removeLowFRClusters.m` computes mean FR from the PSTH (trial-averaged by condition), not from all individual trials: `meanFRs = mean(mean(obj.psth{prbnum},3,'omitnan'),'omitnan')`. The AI's approach averages over all trials directly, which is mathematically different.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue by subtracting `goCue` time from spike times: `aligned_times = trialtm - goCue[trial_idx]`. This matches the instruction to "temporally align based on Go cue onset."

ii.
```python
aligned_times = trialtm[spike_mask] - goCue[trial_idx]
counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. Matches `alignSpikes.m`: `obj.clu{prbnum}(clu).trialtm_aligned = obj.clu{prbnum}(clu).trialtm - event` where event is `goCue`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 5ms (dt = 1/200 seconds), producing 1000 time bins over the [-2.5, 2.5] second window. No rebinning is applied - data is binned once at this resolution.

ii.
```python
PARAMS = {
    'tmin': -2.5,
    'tmax': 2.5,
    'dt': 1.0 / 200.0,  # 5ms bins
}
edges = np.arange(tmin, tmax + dt, dt)
time_axis = edges[:-1] + dt / 2
```

iii. Matches `getDefaultParams.m`: `params.dt = 1/200` and the time window `[-2.5, 2.5]`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the time axis itself - a linearly spaced vector from -2.4975 to 2.4975 seconds (bin centers), representing time from go cue onset. It is not derived from any raw data variable; it is computed from the binning parameters.

ii.
```python
time_axis = edges[:-1] + dt / 2
time_input = time_axis.reshape(1, -1).astype(np.float32)
input_trials.append(time_input)
```

iii. The AI correctly identified that time from go cue is simply the time axis, identical across all trials.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing - the time axis is computed once from the bin edges and reused for every trial. Shape is (1, 1000).

ii.
```python
time_input = time_axis.reshape(1, -1).astype(np.float32)
```

iii. Straightforward computation, consistent with the instruction for a continuous time-varying input.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time axis is the same for both neural data and input - both use the bin centers of the same edges array. They are inherently aligned.

ii.
```python
edges = np.arange(tmin, tmax + dt, dt)
time_axis = edges[:-1] + dt / 2
# Used for both neural binning and input
```

iii. Since neural spikes are binned using the same edges, and the time input is the bin centers, alignment is exact.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction comes from `bp.R` - a boolean array indicating right lick trials. Left lick trials are inferred as `~R` among valid trials.

ii.
```python
data['R'] = bp['R'][0, :].astype(bool)
# ...
lick_direction = session_data['R'][valid_trial_indices].astype(int)
```

iii. The AI maps R=1 (right) and L=0 (left), matching the instruction specification.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Per-trial binary value: R trials get value 1, L trials get value 0. This is broadcast across all time bins for a trial (shape: output[0, :] = constant).

ii.
```python
lick_direction = session_data['R'][valid_trial_indices].astype(int)
output[0, :] = lick_direction[i]  # per-trial, broadcast
```

iii. Simple binary encoding matching instructions (left=0, right=1).

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context comes from `bp.autowater` - a variable indicating whether autowater (WC context) is active on each trial.

ii.
```python
data['autowater'] = bp['autowater'][0, :]
# ...
context = (session_data['autowater'][valid_trial_indices] == 0).astype(int)
```

iii. The AI interprets `autowater == 0` as DR context (value 1) and `autowater != 0` as WC context (value 0).

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Per-trial binary value: `autowater == 0` maps to DR=1, `autowater != 0` maps to WC=0. Broadcast across all time bins.

ii.
```python
context = (session_data['autowater'][valid_trial_indices] == 0).astype(int)
output[1, :] = context[i]  # per-trial, broadcast
```

iii. The reference code conditions include `~autowater` for DR and `autowater` for WC. The AI's logic `autowater == 0` is equivalent to `~autowater`, correctly identifying DR context.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes from `bp.hit` - a boolean array indicating correct (hit) trials.

ii.
```python
data['hit'] = bp['hit'][0, :].astype(bool)
# ...
outcome = session_data['hit'][valid_trial_indices].astype(int)
```

iii. Maps hit=1 (correct) and miss=0 (incorrect), matching instruction specification.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Per-trial binary value from `hit` field. Broadcast across all time bins.

ii.
```python
outcome = session_data['hit'][valid_trial_indices].astype(int)
output[2, :] = outcome[i]  # per-trial, broadcast
```

iii. Simple binary encoding matching instructions (incorrect=0, correct=1).

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from DLC tracking data in `traj{1}` (side camera). The AI extracts tongue x,y positions from `traj.ts` using the feature name 'tongue', aligned using `traj.frameTimes` and the video offset.

ii.
```python
cam0 = session_data['traj'][0]  # side cam
feat_names = cam0['featNames']
for fi, fn in enumerate(feat_names):
    if fn.lower() == 'tongue':
        tongue_idx = fi
        break
# ...
ts = trial_info['ts']
if ts.ndim == 3:
    if ts.shape[0] == len(feat_names):
        x = ts[tongue_idx, 0, :].copy()
        y = ts[tongue_idx, 1, :].copy()
    elif ts.shape[2] == len(feat_names):
        x = ts[:, 0, tongue_idx].copy()
        y = ts[:, 1, tongue_idx].copy()
```

iii. The AI identified tongue as a DLC-tracked feature from camera 0 (side camera), matching the reference code's use of `findPosition` and `findVelocity` for tongue kinematics.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The processing pipeline:
1. Extract tongue x,y positions from DLC tracking data
2. Align frame times to go cue using video offset: `aligned_ft = ft - vidshift - goCue[trial_idx]`
3. Interpolate positions to the neural time axis using linear interpolation
4. Fill NaN positions with session mean (baseline position)
5. Compute velocity using `np.gradient` on filled positions
6. Set velocity to 0 where tongue was originally not visible (NaN positions)
7. Compute speed as `sqrt(xvel^2 + yvel^2)`

ii.
```python
aligned_ft = ft[:len(x)] - vidshift - goCue[trial_idx]
fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
all_x[:, trial_idx] = fx(time_axis)
# Fill NaN with mean
all_x_filled[np.isnan(all_x_filled)] = mean_x
# Compute velocity
xvel = np.gradient(all_x_filled[:, trial_idx])
yvel = np.gradient(all_y_filled[:, trial_idx])
# Zero out where tongue not visible
xvel[nan_mask] = 0.0
yvel[nan_mask] = 0.0
speed = np.sqrt(xvel**2 + yvel**2)
```

iii. The AI's approach broadly matches the reference `findPosition.m` and `findVelocity.m`. However, there are differences: (1) The reference `findPosition` fills NaN tongue positions NOT with the mean - it does NOT fill missing for tongue (only non-tongue features get `fillmissing`). Instead the reference just leaves NaN positions for tongue. (2) The AI fills NaN positions with session mean before computing gradient, which introduces artificial zero velocities at NaN->mean transitions. The reference code computes gradient on the NaN-containing data and then sets NaN velocities to 0. (3) The reference code does NOT smooth tongue positions (`if ~contains(feat,'tongue')` check skips smoothing for tongue), which the AI also does not smooth.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per-session 50th percentile threshold on all non-NaN velocity values. Values >= threshold get category 1 (high), values < threshold get category 0 (low). Special handling when threshold equals minimum value.

ii.
```python
def discretize_velocity(vel_data):
    valid_vals = vel_data[~np.isnan(vel_data)]
    threshold = np.percentile(valid_vals, 50)
    if threshold <= np.min(valid_vals) + 1e-10:
        discretized = (vel_data > threshold).astype(int)
    else:
        discretized = (vel_data >= threshold).astype(int)
    discretized[np.isnan(vel_data)] = 0
    return discretized, threshold
```

iii. The tongue velocity thresholds are ALL 0.00 across all sessions (visible in conversion_full_out.txt). This results in 90.4% "low" and 9.6% "high" across the dataset, a very skewed distribution. This strongly suggests a bug in tongue velocity computation - likely the tongue position extraction is failing (due to incorrect array indexing of the ts data in different MATLAB file formats), resulting in mostly-zero or mostly-NaN velocities where the median is 0.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue positions are interpolated from video frame times to the neural time axis using linear interpolation, with frame times adjusted by video offset and go cue time.

ii.
```python
aligned_ft = ft[:len(x)] - vidshift - goCue[trial_idx]
fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
all_x[:, trial_idx] = fx(time_axis)
```

iii. Alignment approach matches reference `findPosition.m`: `interp1(traj(trix).frameTimes-vidshift-obj.bp.ev.(alignEv)(trix), ts, taxis)`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from DLC tracking data in `traj{2}` (top camera), using the 'top_paw' feature (or falling back to any feature containing 'paw').

ii.
```python
cam1 = session_data['traj'][1]  # top cam
feat_names = cam1['featNames']
for fi, fn in enumerate(feat_names):
    if 'top_paw' in fn.lower():
        paw_idx = fi
        break
```

iii. The AI identified paw tracking from camera 1 (top camera), matching the reference code's `params.traj_features` which lists 'top_paw' for the second camera.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The processing pipeline:
1. Extract paw x,y positions from DLC tracking data (top camera)
2. Align frame times using video offset and go cue
3. Interpolate to neural time axis
4. Fill NaN with nearest non-NaN value
5. Compute velocity using `np.gradient`
6. Subtract baseline derivative (median of diff) - matching reference for non-tongue features
7. Fill NaN velocities with nearest non-NaN
8. Compute speed as `sqrt(xvel^2 + yvel^2)`

ii.
```python
fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
x_interp = fx(time_axis)
# Fill missing
for arr in [x_interp, y_interp]:
    nans = np.isnan(arr)
    if np.any(nans) and np.any(~nans):
        arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans])
# Velocity
xvel = np.gradient(x_interp)
yvel = np.gradient(y_interp)
# Subtract baseline
basederiv_x = np.nanmedian(np.diff(x_interp))
xvel -= basederiv_x
```

iii. This broadly matches the reference `findPosition.m` and `findVelocity.m` for non-tongue features. Key differences: (1) The reference smooths non-tongue positions with `mySmooth(ts, 1, 'reflect')` before interpolation - the AI does not smooth positions. (2) The reference subtracts `basederiv` from both x and y velocity, but uses `basederiv(1)` for both x and y velocity (a likely bug in the reference or intentional - the reference uses `basederiv = median(diff(tsinterp),'omitnan')` where `tsinterp` is 2D, so `basederiv` has 2 elements, but applies `basederiv(1)` to both). The AI uses separate basederiv for x and y.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue velocity - per-session 50th percentile threshold. Paw velocity distributions are close to 50/50 across sessions (0.509 low, 0.491 high overall).

ii.
```python
paw_disc, paw_thresh = discretize_velocity(valid_paw_vel)
```

iii. The paw velocity distribution is much more balanced than tongue velocity, suggesting the paw velocity computation is working more correctly.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same approach as tongue - interpolated from video frame times to neural time axis, adjusted for video offset and go cue alignment.

ii.
```python
aligned_ft = ft[:len(x)] - vidshift - goCue[trial_idx]
fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
```

iii. Consistent with reference `findPosition.m` alignment approach.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from separate `motionEnergy_{anm}_{date}.mat` files. The raw variable is `me.data` (cell array of per-trial motion energy time series) and `me.moveThresh` (movement threshold).

ii.
```python
def load_motion_energy(me_filepath):
    d = sio.loadmat(me_filepath, squeeze_me=False)
    me_raw = d['me']
    # Handles 3 formats: standard, nested, direct
    me = me_raw[0, 0]
    me_data = me['data']
    me_thresh = float(me['moveThresh'].flat[0])
```

iii. The AI handles 3 different formats for motion energy files, matching the variety found in the data. The reference `loadMotionEnergy.m` also handles the nested format: `if isstruct(me.data); me.data = me.data.data; end`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The processing pipeline:
1. Load motion energy data from separate .mat files
2. Compute video offset using `findVideoOffset`
3. Align using `frameTimes - vidshift - goCue` and interpolate to neural time axis
4. Fall back to 400 Hz frame timing if frameTimes don't match ME length
5. Fill NaN values with nearest non-NaN interpolation

ii.
```python
aligned_ft = ft - vidshift - goCue[trial_idx]
f_interp = interp1d(aligned_ft, trial_me, kind='linear', bounds_error=False, fill_value=np.nan)
me_aligned[:, trial_idx] = f_interp(time_axis)
# Fill NaN
col[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), col[not_nans])
```

iii. The reference `loadMotionEnergy.m` uses `interp1(obj.traj{1}(trix).frameTimes-vidshift-alignTimes(trix), me.data{trix}, taxis)` where `taxis = obj.time + params.advance_movement`. The AI does NOT account for `params.advance_movement`, which introduces a temporal offset. The reference also uses `fillmissing('nearest')` to fill NaNs, which the AI replicates.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session 50th percentile threshold, same as tongue and paw velocity. Motion energy distribution is very balanced (0.499 low, 0.501 high).

ii.
```python
me_disc, me_thresh = discretize_velocity(valid_me)
```

iii. The instructions specify 50th percentile threshold for motion energy. The reference code uses `me.moveThresh` (a pre-computed threshold) for binarization: `me.move = me.data > me.moveThresh`. The AI uses a per-session 50th percentile instead, which follows the instructions but differs from the reference code.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is interpolated from video frame times to the neural time axis using the same alignment as tongue/paw: `frameTimes - vidshift - goCue`.

ii.
```python
aligned_ft = ft - vidshift - goCue[trial_idx]
f_interp = interp1d(aligned_ft, trial_me, kind='linear', bounds_error=False, fill_value=np.nan)
me_aligned[:, trial_idx] = f_interp(time_axis)
```

iii. The reference adds `params.advance_movement` to the time axis for motion energy alignment. The AI omits this parameter, which could cause a temporal misalignment.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies:
- Missing/NaN tongue positions: filled with session mean, then velocity set to 0 where tongue was not visible
- Missing/NaN paw positions: filled with nearest non-NaN interpolation
- Missing motion energy: filled with nearest non-NaN interpolation
- Empty spike data: trials with all-zero neural data are removed (30 trials across 2 sessions)
- Missing motion energy files: session still processed but ME data is all NaN -> discretized to 0
- Dropped video frames: trials with NaN `NdroppedFrames` are skipped for kinematic processing
- Empty quality strings: kept (not excluded), matching `findClusters.m`

ii.
```python
# Fill NaN tongue positions with mean
all_x_filled[np.isnan(all_x_filled)] = mean_x
# Fill NaN ME with nearest
col[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), col[not_nans])
# Remove all-zero neural trials
if np.all(ni == 0):
    n_removed += 1
    continue
```

iii. The AI documented edge cases including multiple MATLAB file formats, 3 ME loading formats, and null quality strings. Removing all-zero neural trials (19 + 11 = 30 total) is a reasonable quality control not present in the reference code.

## 11-a. What are the most time-consuming steps of the code?

i. Based on timing output, the most time-consuming steps are:
- Data loading (2-4 seconds per session for I/O)
- Spike binning and smoothing (nested loops over neurons and trials)
- Trajectory interpolation (loop over trials for each kinematic feature)
Total processing time: ~248 seconds for 44 sessions (~5.6s per session average).

ii.
```python
# Nested loop over neurons and trials
for neuron_idx, clu in enumerate(valid_clusters):
    for trial_idx in range(ntrials):
        # ...
        counts, _ = np.histogram(aligned_times, bins=edges)
        fr_smooth = smooth_causal(fr, params['smooth_window'], bctype='none')
```

iii. The AI estimated and documented timing. The conversion completes in ~4 minutes, well under the 15-minute limit.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
1. **Spike binning loop** (neuron x trial): Could vectorize by concatenating all spikes per neuron and using 2D histogram
2. **Smoothing loop** in `smooth_causal`: Loops over columns for 2D smoothing, could use `scipy.signal.fftconvolve`
3. **Tongue velocity loop**: Iterates over trials individually; position extraction could be batched
4. **Paw velocity loop**: Same as tongue
5. **Motion energy alignment loop**: Iterates over trials individually

ii.
```python
# Example: spike binning loops over each neuron and trial
for neuron_idx, clu in enumerate(valid_clusters):
    for trial_idx in range(ntrials):
        spike_mask = trial_nums == trial_num
        counts, _ = np.histogram(aligned_times, bins=edges)

# Smoothing loops over columns
for j in range(x_filt.shape[1]):
    out[:, j] = np.convolve(x_filt[:, j], kern, mode='same')
```

iii. The AI estimated full conversion would take ~4.4 minutes and decided the existing speed was acceptable. No major vectorization was applied.

## 11-c. What processing does the code repeat multiple times?

i. Several processing steps are repeated:
1. **Video offset computation**: Called once per session but could be computed once globally
2. **Time axis computation**: `edges` and `time_axis` are recomputed in every session call, though they're identical
3. **Kernel creation**: `make_causal_gaussian_kernel` is called for every neuron-trial smoothing operation within `smooth_causal` (though this is fast)
4. **Feature name lookup**: The tongue and paw feature index search is repeated per session

ii.
```python
# Time axis computed in every process_session call
edges = np.arange(tmin, tmax + dt, dt)
time_axis = edges[:-1] + dt / 2

# Kernel created in every smooth_causal call
kern = make_causal_gaussian_kernel(N)
```

iii. These redundancies are minor and don't significantly impact performance.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items:
1. **Processing ALL trials for neural data** before filtering to valid trials: `trialdat` is computed for all `ntrials` but only valid trial indices are used in the output
2. **Processing kinematic data for all trials** then selecting valid trials: tongue, paw, and ME are computed for all trials before subsetting
3. **Computing lick times** (lickL, lickR): loaded but never used in the output
4. **Loading event times** (sample, delay, bitStart): loaded but only bitStart is used for video offset; sample and delay are unused
5. **All-trial PSTH-like computation** for low FR filtering when only the mean FR is needed

ii.
```python
# All ntrials processed, but only valid_trial_indices used
trialdat = np.zeros((n_timepts, n_neurons, ntrials), dtype=np.float32)
for neuron_idx, clu in enumerate(valid_clusters):
    for trial_idx in range(ntrials):  # processes ALL trials
        # ...

# Only valid trials used later
valid_me = me_aligned[:, valid_trial_indices]
```

iii. Processing all trials first and filtering later is inefficient but ensures no data is missed. This matches the reference code's approach where `trialdat` is computed for all trials.
