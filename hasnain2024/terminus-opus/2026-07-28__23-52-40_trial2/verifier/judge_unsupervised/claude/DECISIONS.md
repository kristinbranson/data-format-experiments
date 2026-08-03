# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from HDF5 (.mat v7.3) files in the `data/Ephys_Behavior` directory. Each session has a `data_structure_<animal>_<date>.mat` file for neural and behavioral data, and a separate `motionEnergy_<animal>_<date>.mat` file for motion energy. Session metadata (animal, date, probe numbers) is hardcoded in a `SESSION_META` list of 25 sessions. The AI iterates over sessions, calling `process_session()` for each, using `h5py.File()` to read HDF5 data.

ii.
```python
DATA_DIR = 'data/Ephys_Behavior'

SESSION_META = [
    ('EKH1', '2021-08-07', [1]),
    ('EKH3', '2021-08-11', [1, 2]),
    ...  # 25 sessions total
]

def process_session(animal, date, probes, ...):
    data_file = os.path.join(DATA_DIR, f'data_structure_{session_id}.mat')
    me_file = os.path.join(DATA_DIR, f'motionEnergy_{session_id}.mat')
    f = h5py.File(data_file, 'r')
```

iii. The AI documented this in CONVERSION_NOTES.md Step 1, identifying the reference code's `loadObjs.m` and `loadSessionData.m` as the data loading functions, and noting the data is MATLAB v7.3 (HDF5 format). The session list was derived from the `loadXXX_ALMVideo.m` scripts.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `animal` field from `SESSION_META`. After processing all sessions, unique subject names are collected and sorted. A `subject_idx` array maps each session to its subject index.

ii.
```python
subjects = sorted(list(set(r['animal'] for r in all_results)))
subj_idx = subjects.index(result['animal'])
subject_idx_list.append(subj_idx)
```

iii. The AI noted 10 animals in the data (vs. 9 in the paper), and decided to include all 10. The subjects are: EKH1, EKH3, JEB6, JEB7, JEB13, JEB14, JEB15, JEB19, JGR2, JGR3.

## 1-c. How are the data split into sessions?

i. Sessions are defined by the hardcoded `SESSION_META` list, with each entry being a unique (animal, date) pair. Each session is processed independently via `process_session()`. The resulting list of sessions becomes the top-level structure in the output data dictionary.

ii.
```python
SESSION_META = [
    ('EKH1', '2021-08-07', [1]),
    ...
]
for idx, (animal, date, probes) in enumerate(sessions):
    result = process_session(animal, date, probes, ...)
    if result is not None:
        all_results.append(result)
```

iii. The session definitions match the reference `loadXXX_ALMVideo.m` scripts. The AI correctly identifies 25 sessions, matching the paper's stated 25 sessions.

## 1-d. How are the data split into trials?

i. Trial count per session is read from `obj.bp.Ntrials`. Individual trial data is indexed by trial number (1-indexed in MATLAB convention, converted to 0-indexed). Only valid trials (after filtering) are included in the output.

ii.
```python
bp = f['obj']['bp']
Ntrials = int(bp['Ntrials'][0, 0])
# ...
for trial_idx in valid_trials:
    neural_trials.append(trialdat[:, :, trial_idx].T.astype(np.float32))
```

iii. The AI used the behavioral data structure (`obj.bp`) to determine trial count, consistent with the reference code's approach.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to exclude: early lick trials (`early==1`), no-response/ignore trials (`no==1`), stimulation trials (`stim.enable==1`). Only hit and miss trials are included (`hit==1 | miss==1`). A minimum of 2 valid trials per session is required.

ii.
```python
valid_mask = (early == 0) & (no == 0) & (stim_enable == 0) & ((hit == 1) | (miss == 1))
valid_trials = np.where(valid_mask)[0]
if len(valid_trials) < 2:
    # skip session
```

iii. The AI documented these filtering criteria in CONVERSION_NOTES.md Steps 3 and 5, citing the paper's methods: "Early lick and ignore trials excluded from all analyses." The reference code's `findTrials.m` uses condition strings like `'R&hit&~stim.enable&~autowater&~early'`, which filter by similar criteria but also separate by direction and context. The AI's approach of including both hit and miss trials and not further filtering by direction or context is reasonable for the decoder task, though the reference code's condition strings also exclude `autowater` (WC) trials in some analyses. However, the instructions ask for WC as an output variable, so including autowater trials is correct.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from spike times stored in `obj.clu{probe}(cluster).trialtm` (spike times relative to trial start), `obj.clu{probe}(cluster).trial` (trial assignment for each spike), and `obj.bp.ev.goCue` (go cue event times for alignment).

ii.
```python
spike_trialtm = f[tm_ref][:].flatten()    # spike times relative to trial start
spike_trial = f[trial_ref][:].flatten()    # trial numbers
goCue = bp['ev']['goCue'][:].flatten()     # go cue times
```

iii. The AI identified these as the key variables from the reference code's `alignSpikes.m` and `getSeq.m` functions.

## 2-b. How is the `neural` data processed?

i. Processing pipeline: (1) Spike times are aligned to go cue by subtracting `goCue(trial)` from `trialtm`. (2) Aligned spikes are binned into 10ms time bins from -2.5s to 2.5s using `np.histogram`. (3) Counts are converted to firing rate (spks/s) by dividing by dt. (4) Firing rates are smoothed with a causal Gaussian kernel (window=15 samples, reflect boundary condition).

ii.
```python
spike_aligned = spike_trialtm_valid - spike_goCue
counts, _ = np.histogram(trial_spikes, bins=EDGES)
fr = counts / PARAMS['dt']  # spks/sec
fr_smooth = causal_gaussian_smooth(fr, PARAMS['smooth'], PARAMS['bctype'])
```

iii. The AI documented this as matching `getSeq.m` (binning and smoothing) and `alignSpikes.m` (alignment). The smoothing implementation uses `mySmooth.m`'s logic: causal Gaussian with `gausswin(N)`, zero first half, reflect boundary.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering: (1) Cluster quality filter: excludes clusters with quality labels 'garbage', 'gabrga', 'noisy', 'real?'. Clusters with empty/null quality strings are included. (2) Low firing rate filter: removes neurons with mean firing rate <= 1 Hz. Sessions with fewer than 10 neurons after filtering are skipped.

ii.
```python
# Quality filter
cluid = find_clusters(qualities, PARAMS['quality_exclude'])

# Low FR filter
meanFRs = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
fr_mask = meanFRs > PARAMS['lowFR']  # lowFR = 1.0
trialdat = trialdat[:, fr_mask, :]

if n_neurons < 10:
    # skip session
```

iii. The AI matched `findClusters.m` (quality='all' excludes garbage/noisy) and `removeLowFRClusters.m`. The low FR threshold of 1.0 Hz was chosen to match the paper's statement, though the default in `getDefaultParams.m` is 0.5 Hz. The minimum 10 neurons per session threshold matches the paper's stated criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue onset. For each spike, the aligned time is computed as `trialtm - goCue(trial)`, where `trialtm` is the spike time relative to trial start and `goCue(trial)` is the go cue time for that trial. Spikes are then binned relative to this aligned time.

ii.
```python
spike_goCue = goCue[spike_trial_valid - 1]
spike_aligned = spike_trialtm_valid - spike_goCue
counts, _ = np.histogram(trial_spikes, bins=EDGES)
```

iii. This matches `alignSpikes.m`: `obj.clu{prbnum}(clu).trialtm_aligned = obj.clu{prbnum}(clu).trialtm - event` where event is `obj.bp.ev.goCue`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 10ms (`dt = 1/100`). Bins span from -2.5s to 2.5s relative to go cue, yielding 500 time bins. No additional temporal rebinning is applied after the initial binning.

ii.
```python
PARAMS = {
    'dt': 1/100,         # 10ms time bins
    'tmin': -2.5,
    'tmax': 2.5,
}
EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
```

iii. The AI chose 10ms bins matching `params.dt = 1/100` used in most reference analysis scripts. The reference code's `getDefaultParams.m` also defaults to `1/100`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not derived from raw data variables. It is constructed as the time axis itself: the center of each 10ms bin, ranging from -2.495s to 2.495s.

ii.
```python
time_input = TIME_AXIS.astype(np.float32).reshape(1, -1)
session_inputs.append(time_input)
```

iii. The AI constructed this from the binning parameters, which matches the instruction to provide "Time from go cue onset in seconds" as a continuous, time-varying input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing of raw data. The time axis is computed from binning parameters: `edges = arange(tmin, tmax+dt, dt)`, then bin centers = `edges[:-1] + dt/2`. This produces a 500-element vector from -2.495 to 2.495.

ii.
```python
EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
```

iii. Same time axis construction as `getSeq.m`: `obj.time = edges + params.dt/2; obj.time = obj.time(1:end-1)`.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time input is identical for every trial - it is the same time axis used for binning neural data, so alignment is inherent. Both neural data and the time input use the same bin centers.

ii.
```python
# Same TIME_AXIS used for both neural binning and input
time_input = TIME_AXIS.astype(np.float32).reshape(1, -1)
```

iii. Since the time axis is defined by the binning parameters and is identical for all trials, alignment is guaranteed by construction.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `obj.bp.R` - a boolean array indicating right-lick trials (1=right, 0=left).

ii.
```python
R = bp['R'][:].flatten()
lick_direction = R.copy()  # 1=right, 0=left
```

iii. The AI noted this maps to left=0, right=1 per the decoder task specification.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The raw `R` variable is directly used as the lick direction (right=1, left=0). It is broadcast to all time bins as a per-trial constant, creating a (n_timepoints,) array with the same value at every time point.

ii.
```python
lick_dir = np.full(N_TIMEBINS, info['lick_direction'], dtype=np.int64)
```

iii. No transformation needed. The value is the same for all time bins within a trial since lick direction is a per-trial variable.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj.bp.autowater` - a boolean array where autowater=1 indicates water-cued (WC) trials.

ii.
```python
autowater = bp['autowater'][:].flatten()
context = 1 - autowater  # DR=1, WC=0
```

iii. The AI inverted `autowater` so that WC=0 and DR=1, matching the decoder task specification.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The `autowater` variable is inverted: `context = 1 - autowater`. This maps autowater=1 (WC) to context=0, and autowater=0 (DR) to context=1. Like lick direction, it is broadcast to all time bins.

ii.
```python
context = 1 - autowater
# Later:
context_arr = np.full(N_TIMEBINS, info['context'], dtype=np.int64)
```

iii. Simple inversion to match the specification WC=0, DR=1.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit` - a boolean array where hit=1 indicates correct trials.

ii.
```python
hit = bp['hit'][:].flatten()
outcome = hit.copy()  # 1=correct, 0=incorrect
```

iii. Since the AI already filters to only include hit and miss trials (excluding no-response), `hit` directly maps to outcome.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The `hit` variable is directly used as outcome (correct=1, incorrect=0). Broadcast to all time bins as a per-trial constant.

ii.
```python
outcome = hit.copy()
# Later:
outcome_arr = np.full(N_TIMEBINS, info['outcome'], dtype=np.int64)
```

iii. No transformation needed since hit already encodes correct=1, incorrect=0 within the filtered trial set.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from DLC tracking data in `obj.traj` (view 1 / side view, index 0), specifically the 'tongue' feature. The x,y positions are extracted from `traj.ts` (trajectory time series), aligned using `traj.frameTimes` and `findVideoOffset()`.

ii.
```python
traj_ref = f['obj']['traj'][0, 0]  # view 1 (side)
# Extract x, y for tongue feature
x = ts_data[tongue_idx, 0, :]
y = ts_data[tongue_idx, 1, :]
```

iii. The AI identified the tongue feature in the side view, matching `findPosition.m` which takes view and feature parameters.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Processing: (1) Extract tongue x,y positions from DLC data. (2) Compute video-to-neural offset using `findVideoOffset()`. (3) Align frame times to go cue: `aligned_times = frameTimes - vidshift - goCue[trial]`. (4) Interpolate positions to neural time axis using `scipy.interpolate.interp1d`. (5) Compute velocity using `np.gradient` on position. (6) Set NaN velocities to 0 (tongue not visible = no movement). (7) Compute speed: `sqrt(xvel^2 + yvel^2)`.

ii.
```python
aligned_times = frameTimes - vidshift - goCue[trix]
fx = interp1d(aligned_times[valid], x[valid], bounds_error=False, fill_value=np.nan)
xpos[:, trix] = fx(taxis)
# Velocity
xv = np.gradient(xpos[:, trix])
xv[np.isnan(xv)] = 0  # tongue: NaN -> 0
speed = np.sqrt(xvel**2 + yvel**2)
```

iii. The AI matched `findPosition.m` (interpolation to neural time axis) and `findVelocity.m` (gradient-based velocity, NaN->0 for tongue). However, the AI does NOT apply any smoothing to tongue position before computing velocity (no `mySmooth(ts, 1, 'reflect')` call), which matches the reference code since tongue uses smooth window of 1 (no-op).

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Tongue velocity is discretized per-session using the 50th percentile (median) of all non-NaN values as threshold. Values >= threshold are labeled 1 (high), below are 0 (low). Special handling: if the threshold is 0, the median of positive values is used instead.

ii.
```python
def discretize_continuous(values_per_trial, threshold_percentile=50):
    all_values = np.concatenate([v.flatten() for v in values_per_trial])
    all_values = all_values[~np.isnan(all_values)]
    threshold = np.percentile(all_values, threshold_percentile)
    if threshold == 0:
        pos_values = all_values[all_values > 0]
        if len(pos_values) > 0:
            threshold = np.percentile(pos_values, threshold_percentile)
    result = [(v >= threshold).astype(np.float32) for v in values_per_trial]
    return result
```

iii. The AI noted the edge case where tongue velocity is mostly 0 (tongue not visible), causing the median to be 0. The fallback to median of positive values creates a heavily imbalanced distribution (97.3% low, 2.7% high).

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue velocity is aligned to the neural time axis by interpolating DLC frame-time positions onto the same bin centers used for neural data, after accounting for the video offset and go cue time.

ii.
```python
taxis = TIME_AXIS + PARAMS['advance_movement']  # advance_movement = 0
aligned_times = frameTimes - vidshift - goCue[trix]
fx = interp1d(aligned_times[valid], x[valid], bounds_error=False, fill_value=np.nan)
xpos[:, trix] = fx(taxis)
```

iii. This matches `findPosition.m`'s approach: `interp1(frameTimes - vidshift - alignEvent, ts, taxis)`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from DLC tracking data in `obj.traj` (view 2 / top view, index 1), using features containing 'paw' in their name (typically 'top_paw' and 'bottom_paw').

ii.
```python
traj_ref = f['obj']['traj'][1, 0]  # view 2 (top)
for i, name in enumerate(feat_names):
    if 'paw' in name.lower():
        paw_indices.append(i)
```

iii. The AI identified paw features from the top view, consistent with the reference code's use of view 2 for paw tracking.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Processing: (1) Extract x,y positions for each paw feature. (2) Align and interpolate to neural time axis (same as tongue). (3) Fill missing (NaN) positions with nearest valid value. (4) Compute velocity using `np.gradient`. (5) Subtract baseline drift: `basederiv = median(diff(position))`. (6) Fill NaN velocities with nearest. (7) Compute speed: `sqrt(xvel^2 + yvel^2)`. (8) Average speeds across paw features (top_paw and bottom_paw).

ii.
```python
# Fill missing positions
xp = np.interp(indices, indices[mask], xp[mask])
# Velocity with baseline subtraction
xv = np.gradient(xpos[:, trix])
basederiv_x = np.nanmedian(np.diff(xpos[:, trix]))
xv = xv - basederiv_x
# Average across paw features
avg_speed = np.mean(all_speeds, axis=0)
```

iii. The AI matched `findVelocity.m`'s logic for non-tongue features: subtract baseline derivative, fill missing. The averaging of top_paw and bottom_paw speeds is the AI's own decision.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue velocity: 50th percentile per-session threshold. Values >= threshold = 1 (high), below = 0 (low). Same edge case handling for zero threshold.

ii.
```python
paw_disc = discretize_continuous(result['paw_vel'])
```

iii. Uses the same `discretize_continuous` function. The distribution is more balanced (52.9% low, 47.1% high) than tongue velocity.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same alignment approach as tongue velocity: interpolation from video frame times to neural time axis, with video offset and go cue alignment.

ii.
```python
aligned_times = frameTimes - vidshift - goCue[trix]
fx = interp1d(aligned_times[valid], x[valid], bounds_error=False, fill_value=np.nan)
xp = fx(taxis)
```

iii. Same approach as tongue, matching `findPosition.m`.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from separate `motionEnergy_<animal>_<date>.mat` files. The data is in `me.data` (a cell array of per-trial motion energy time series).

ii.
```python
me_file = os.path.join(DATA_DIR, f'motionEnergy_{session_id}.mat')
me_raw = sio.loadmat(me_file)
me_struct = me_raw['me'][0, 0]
me_data = me_struct['data']
if me_data.dtype.names is not None and 'data' in me_data.dtype.names:
    me_data = me_data[0, 0]['data']
```

iii. The AI matched `loadMotionEnergy.m`'s handling of the nested struct case (`if isstruct(me.data), me.data = me.data.data`).

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Processing: (1) Load motion energy time series per trial. (2) Get frame times from `traj.frameTimes` (falling back to `(1:Nframes)/400`). (3) Compute aligned times: `frameTimes - vidshift - goCue[trial]`. (4) Interpolate to neural time axis. (5) Fill NaN values with nearest valid value.

ii.
```python
aligned_times = frameTimes - vidshift - goCue[trix]
f_interp = interp1d(aligned_times[valid], me_trial[valid], bounds_error=False, fill_value=np.nan)
me_interp = f_interp(taxis)
# Fill NaN with nearest
me_interp = np.interp(indices, indices[mask], me_interp[mask])
```

iii. The AI matched `loadMotionEnergy.m`'s interpolation and NaN-filling approach.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same as other continuous outputs: 50th percentile per-session threshold. Values >= threshold = 1, below = 0.

ii.
```python
me_disc = discretize_continuous(result['motion_energy'])
```

iii. Uses the same `discretize_continuous` function. The distribution is well-balanced (50.0% low, 50.0% high).

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by interpolating from video frame times (adjusted for video offset and go cue) to the neural time axis. This is the same approach as for kinematic variables.

ii.
```python
taxis = TIME_AXIS + PARAMS['advance_movement']
aligned_times = frameTimes - vidshift - goCue[trix]
f_interp = interp1d(aligned_times[valid], me_trial[valid], bounds_error=False, fill_value=np.nan)
me_interp = f_interp(taxis)
```

iii. Matches `loadMotionEnergy.m`: `interp1(frameTimes - vidshift - alignTimes(trix), me.data{trix}, taxis)`.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled: (1) Missing data files: session is skipped with a warning. (2) Missing/NaN frame times: trial skipped for kinematic computation. (3) NaN velocity values: for tongue, set to 0; for paw, filled with nearest valid value. (4) NaN motion energy: filled with nearest valid value. (5) Dropped video frames (`NdroppedFrames`): trial skipped if NaN. (6) Sessions with < 2 valid trials: skipped. (7) Sessions with < 10 neurons: skipped. (8) Probe indexing: handles single vs dual probe sessions. (9) Nested motion energy struct: unwrapped. (10) Null quality strings in clusters: treated as valid (included). (11) Zero threshold for discretization: falls back to median of positive values.

ii.
```python
# Tongue: NaN -> 0
xv[np.isnan(xv)] = 0
# Paw: fill with nearest
xp = np.interp(indices, indices[mask], xp[mask])
# Motion energy: fill with nearest
me_interp = np.interp(indices, indices[mask], me_interp[mask])
# Dropped frames check
if np.isnan(ndrop).any():
    continue
```

iii. The AI documented edge cases in CONVERSION_NOTES.md Step 10, particularly the JEB15 null quality strings and JEB7 probe indexing issues.

## 11-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is spike binning and smoothing in `process_session()`, which loops over every cluster and every trial to histogram spikes and apply causal Gaussian smoothing. For the dual-probe session EKH3, processing takes 16.4s. The full conversion takes 188.9s for 25 sessions.

ii.
```python
for ci, clu_idx in enumerate(cluid):
    for trial_num in range(1, Ntrials + 1):
        counts, _ = np.histogram(trial_spikes, bins=EDGES)
        fr = counts / PARAMS['dt']
        fr_smooth = causal_gaussian_smooth(fr, PARAMS['smooth'], PARAMS['bctype'])
        trialdat[:, ci, trial_num - 1] = fr_smooth
```

iii. The nested loop over clusters and trials is the primary bottleneck, along with the per-column convolution in `causal_gaussian_smooth`.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized: (1) The spike histogram loop over trials could use vectorized operations with trial indexing. (2) The smoothing convolution loop over columns in `causal_gaussian_smooth` could use `scipy.ndimage.convolve1d` or `np.apply_along_axis`. (3) The tongue/paw velocity computation loops over trials could be partially vectorized using `np.gradient` on the full 2D array at once. (4) The discretization could use a single vectorized comparison.

ii.
```python
# Per-column convolution loop (could be vectorized)
for j in range(x_padded.shape[1]):
    result[:, j] = convolve(x_padded[:, j], kern, mode='same')

# Per-trial velocity loop (could be vectorized)
for trix in range(Ntrials):
    xv = np.gradient(xpos[:, trix])
```

iii. The AI noted timing information and estimated full conversion time in CONVERSION_NOTES.md Step 7, but did not heavily optimize vectorization since the total time was under 3 minutes.

## 11-c. What processing does the code repeat multiple times?

i. (1) `find_video_offset()` is called separately in `compute_tongue_velocity`, `compute_paw_velocity`, and `load_motion_energy` for the same session - computing it three times. (2) `scipy.interpolate.interp1d` is imported inside loops (cosmetic but repeated). (3) Frame times for the same trial are read multiple times across tongue, paw, and motion energy processing. (4) The discretization function is called during plotting (`plot_processing`) and again during final output construction.

ii.
```python
# Video offset computed 3 times per session:
# In compute_tongue_velocity:
vidshift = find_video_offset(f)
# In compute_paw_velocity:
vidshift = find_video_offset(f)
# In load_motion_energy:
vidshift = find_video_offset(f)
```

iii. The AI did not explicitly document this redundancy.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The code computes absolute spike times (`tm_abs_ref = clu_group['tm'][clu_idx, 0]`) but never uses them. (2) Processing is done for ALL trials (including invalid ones) during spike binning, then only valid trials are selected afterward. This wastes computation on early-lick, no-response, and stimulation trials. (3) Tongue/paw velocity and motion energy are computed for ALL trials but only valid trials are used. (4) When `show_processing` is enabled, the `trialdat_full` array is saved but only used for plotting.

ii.
```python
# Absolute spike times loaded but unused
tm_abs_ref = clu_group['tm'][clu_idx, 0]
spike_tm_abs = f[tm_abs_ref][:].flatten()

# All trials processed, then filtered
trialdat = np.zeros((N_TIMEBINS, len(cluid), Ntrials), dtype=np.float32)
# ... bins all Ntrials
for trial_idx in valid_trials:  # only valid trials used
    neural_trials.append(trialdat[:, :, trial_idx].T)
```

iii. Processing all trials (rather than just valid ones) is a significant source of unnecessary computation, especially for sessions with many excluded trials.
