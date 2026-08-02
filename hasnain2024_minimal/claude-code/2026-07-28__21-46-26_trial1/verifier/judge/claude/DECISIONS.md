# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans two directories (`/app/data/Ephys_Behavior` and `/app/data/RandomizedDelay_Ephys_Behavior`) for files matching `data_structure_*.mat`, pairs them with corresponding `motionEnergy_*.mat` files, and loads each session's `.mat` file using HDF5 (h5py) or scipy.io as a fallback. Motion energy is loaded from separate files.

ii.
```python
DATA_DIRS = [
    '/app/data/Ephys_Behavior',
    '/app/data/RandomizedDelay_Ephys_Behavior',
]

def find_session_files(data_dirs):
    for data_dir in data_dirs:
        data_files = sorted(glob.glob(os.path.join(data_dir, 'data_structure_*.mat')))
        for df in data_files:
            parts = basename.replace('data_structure_', '').replace('.mat', '')
            me_file = os.path.join(data_dir, f'motionEnergy_{parts}.mat')
            sessions.append({'data_file': df, 'me_file': me_file, 'animal_date': parts})

def load_session_data(filepath):
    try:
        with h5py.File(filepath, 'r') as f:
            ...
        return _load_session_data_h5(filepath)
    except (OSError, ValueError):
        pass
    return load_session_data_v5(filepath)
```

iii. The AI documented that MATLAB v7.3 files are loaded with h5py and v5/v7 with scipy.io fallback, and that 2 behavior-only sessions were skipped.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are extracted from the filename of each data file (e.g., `data_structure_JEB13_2022-09-13.mat` -> animal = `JEB13`). Unique subjects are collected and a subject index is assigned per session.

ii.
```python
parts = basename.replace('data_structure_', '').replace('.mat', '').split('_')
animal = parts[0]
# ...
unique_subjects = sorted(set(all_animals))
subject_idx = np.array([unique_subjects.index(a) for a in all_animals], dtype=np.int64)
```

iii. The AI's CONVERSION_NOTES notes the subject counts (10 mice in Ephys_Behavior, 4 in RandomizedDelay) and acknowledges a discrepancy with the paper (9 vs 10 mice for Ephys_Behavior).

## 1-c. How are the data split into sessions?

i. Each `data_structure_*.mat` file represents one session. Sessions are processed individually and results are collected into lists.

ii.
```python
for idx, sf in enumerate(session_files):
    result = process_session(sf['data_file'], sf['me_file'], idx)
    if result is None:
        continue
    all_neural.append(result['neural'])
```

iii. Documented in CONVERSION_NOTES as 25 sessions in Ephys_Behavior (matching paper) and 20 in RandomizedDelay (paper says 19).

## 1-d. How are the data split into trials?

i. Trials are defined by the `Ntrials` field in the session data. Each spike is assigned to a trial via `clu.trial`. Valid trials are selected by the trial filter mask, and data is extracted per trial by iterating over valid trial indices.

ii.
```python
Ntrials = session['Ntrials']
valid_trials = np.where(trial_mask)[0]
# In align_and_bin_spikes:
for ti, trial_idx in enumerate(valid_trials):
    trial_num = trial_idx + 1  # MATLAB 1-indexed
    spike_mask = spike_trials == trial_num
    aligned_times = spike_times[spike_mask] - go_time
```

iii. The AI processes only valid (filtered) trials, which is documented in CONVERSION_NOTES.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to include only non-stimulation, non-early-lick trials where the animal responded (hit or miss). This includes both correct (hit) and incorrect (miss) trials.

ii.
```python
trial_mask = (
    (session['stim_enable'] == 0) &
    (session['early'] == 0) &
    ((session['hit'] == 1) | (session['miss'] == 1))
)
```

iii. The AI documented matching reference code conditions: `~stim.enable & ~early & (hit | miss)`. The AI includes miss trials (which the reference `getDefaultParams.m` comments out) because the decoder task requires outcome prediction.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `obj.clu{probe}` which contains per-unit spike data: `trialtm` (spike times relative to trial start) and `trial` (trial assignment for each spike).

ii.
```python
# In _load_session_data_h5:
unit['trialtm'] = f[tm_ref][()].flatten()
unit['trial'] = f[trial_ref][()].flatten().astype(int)
```

iii. Documented as following `getDefaultParams.m` and `getSeq.m`.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to go cue, binned into 5ms bins, converted to firing rate (counts/dt), then smoothed with a causal Gaussian kernel (N=15 samples, half-kernel zeroed for causality).

ii.
```python
aligned_times = spike_times[spike_mask] - go_time
counts, _ = np.histogram(aligned_times, bins=edges)
fr = counts / dt
fr_smooth = causal_gaussian_smooth(fr, smooth_samples)

def causal_gaussian_smooth(x, kernel_width_samples):
    N = kernel_width_samples  # 15
    sigma = N / 6  # MATLAB gausswin default: alpha=2.5, sigma = (N-1)/(2*alpha)
    kern = np.exp(-0.5 * ((n - center) / sigma) ** 2)
    kern[:N // 2] = 0  # causal
    kern = kern / kern.sum()
    return np.convolve(x, kern, mode='same')
```

iii. The AI documented following `getSeq.m` and `mySmooth.m`, with causal Gaussian kernel matching the MATLAB implementation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered in two stages: (1) quality filter excludes 'garbage', 'gabrga', 'noisy', 'real?' labels; (2) firing rate filter removes units with mean FR <= 0.5 Hz. Additionally, only ALM probes are included.

ii.
```python
def filter_clusters(clusters, quality_filter='all'):
    excluded = {'garbage', 'gabrga', 'noisy', 'real?'}
    for i, unit in enumerate(clusters):
        q = unit.get('quality', '').strip().lower()
        if q not in excluded and q != '':
            indices.append(i)
        elif q == '':
            indices.append(i)

# FR filter:
mean_frs = np.mean(np.mean(trialdat, axis=2), axis=0)
fr_mask = mean_frs > LOW_FR  # 0.5 Hz
```

iii. Documented as matching `findClusters.m` with quality='all' and `removeLowFRClusters.m` with default lowFR=0.5.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned by subtracting the go cue time for each trial: `aligned_times = spike_times - go_time`. These aligned times are then binned relative to the go cue.

ii.
```python
go_time = go_cue_times[trial_idx]
aligned_times = spike_times[spike_mask] - go_time
counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. Documented as alignment to go cue onset, matching `alignSpikes.m`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 5ms (1/200 s), spanning -2.5 to +2.5 seconds, giving 1000 time bins per trial. No rebinning is applied; spikes are binned directly at this resolution.

ii.
```python
DT = 1 / 200  # 5 ms bins
TMIN = -2.5
TMAX = 2.5
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
```

iii. Documented as matching `getDefaultParams.m`: `params.dt = 1/200`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not derived from raw data variables. It is the time axis constructed from the binning parameters (TMIN, TMAX, DT), representing the center of each time bin.

ii.
```python
TIME_AXIS = EDGES[:-1] + DT / 2
# In process_session:
input_trial = TIME_AXIS.reshape(1, -1).astype(np.float32)
```

iii. Documented as time from go cue onset: continuous time axis [-2.4975, ..., 2.4975] in 5ms steps.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing is involved. The time axis is computed once from the binning parameters and is the same for every trial. It is broadcast as a (1, n_timepoints) array for each trial.

ii.
```python
input_trial = TIME_AXIS.reshape(1, -1).astype(np.float32)
```

iii. The AI documented this as a direct computation from parameters.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time axis is inherently aligned with the neural data because both use the same time bins defined by EDGES, which are centered on the go cue (time=0).

ii.
```python
# Both neural and input use the same TIME_AXIS derived from EDGES
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
```

iii. Alignment is by construction since both the neural data binning and the time input use the same time axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Derived from `obj.bp.R` (right-trial indicator), where R=1 means the trial is a right-lick trial.

ii.
```python
session['R'] = bp['R'][()].flatten().astype(float)
# ...
lick_direction = session['R'][valid_trials].astype(np.int32)  # R=1, L=0
```

iii. Documented as R=1 (right), L=0 (left) from `obj.bp.R` field.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The R field is directly used as the lick direction: right=1, left=0. The value is tiled across all time bins as a per-trial constant.

ii.
```python
lick_direction = session['R'][valid_trials].astype(np.int32)
# ...
output_trial[0, :] = lick_direction[ti]  # per-trial, tiled across time
```

iii. No complex processing; direct mapping from the R indicator.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derived from `obj.bp.autowater`, where autowater=1 indicates WC (water-cued) trials.

ii.
```python
session['autowater'] = bp['autowater'][()].flatten().astype(float)
# ...
context = (1 - session['autowater'][valid_trials]).astype(np.int32)  # DR=1, WC=0
```

iii. Documented as: autowater=1 means WC, so DR=1 is computed as `1 - autowater`.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The autowater flag is inverted (1 - autowater) so that DR=1 and WC=0, matching the decoder task specification. The value is tiled across all time bins.

ii.
```python
context = (1 - session['autowater'][valid_trials]).astype(np.int32)
output_trial[1, :] = context[ti]
```

iii. Simple inversion of the autowater flag.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `obj.bp.hit`, where hit=1 indicates a correct trial.

ii.
```python
session['hit'] = bp['hit'][()].flatten().astype(float)
# ...
outcome = session['hit'][valid_trials].astype(np.int32)
```

iii. Documented as correct=1, incorrect=0 from `obj.bp.hit`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The hit field is directly used as the outcome value (1=correct, 0=incorrect). Tiled across all time bins.

ii.
```python
outcome = session['hit'][valid_trials].astype(np.int32)
output_trial[2, :] = outcome[ti]
```

iii. Direct mapping from the hit indicator.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Derived from `obj.traj{1}` (side camera, cam index 0), specifically the 'tongue' feature's x,y position time series (`ts` field with shape (n_features, 3, n_frames) where 3=[x, y, likelihood]).

ii.
```python
tongue_vel = extract_kinematic_feature(session, 0, 'tongue', valid_trials, go_cue_times)
# In extract_kinematic_feature:
x_pos = ts[feat_idx, 0, :].copy()
y_pos = ts[feat_idx, 1, :].copy()
```

iii. Documented as extracted from side camera (cam 0), 'tongue' feature.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Velocity magnitude is computed as `sqrt(dx^2 + dy^2)` using `np.gradient` at native video frame rate (400 Hz). When the tongue is not visible (NaN positions), velocity is set to 0. The velocity is then interpolated to the neural time axis using linear interpolation.

ii.
```python
dx = np.gradient(x_pos, dt_vid)  # dt_vid = 1/400
dy = np.gradient(y_pos, dt_vid)
vel = np.sqrt(dx**2 + dy**2)
if is_tongue:
    nan_mask = np.isnan(x_pos) | np.isnan(y_pos)
    vel[nan_mask] = 0.0
# Interpolate to neural time axis
aligned_frame_times = frame_times[:n_frames] - vidshift - go_time
interp_fn = interp1d(aligned_frame_times, vel, kind='linear', bounds_error=False, fill_value=np.nan)
velocities[:, ti] = interp_fn(TIME_AXIS)
```

iii. The AI documented matching `findPosition.m` and paper methods, setting tongue velocity to 0 when not visible.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Discretized per session using the 50th percentile (median) as threshold: values >= threshold -> 1 ("high"), values < threshold -> 0 ("low").

ii.
```python
def discretize_per_session(values, percentile=50):
    valid = values[~np.isnan(values)]
    threshold = np.percentile(valid, percentile)
    discretized = (values >= threshold).astype(np.int32)
    return discretized
```

iii. Documented as per decoder task specification. The AI noted that tongue velocity is degenerate because the tongue is only visible ~10-20% of the time, making the 50th percentile threshold = 0, resulting in nearly all values classified as "high".

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Video frame times are aligned to go cue by subtracting vidshift (video-neural offset) and go cue time, then velocity is interpolated from video frame times to the neural time axis using linear interpolation.

ii.
```python
aligned_frame_times = frame_times[:n_frames] - vidshift - go_time
interp_fn = interp1d(aligned_frame_times, vel, kind='linear', bounds_error=False, fill_value=np.nan)
velocities[:, ti] = interp_fn(TIME_AXIS)
```

iii. Documented as matching video offset computation from `findVideoOffset.m`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Derived from `obj.traj{2}` (bottom camera, cam index 1), specifically the 'top_paw' and 'bottom_paw' feature positions.

ii.
```python
paw_top = extract_kinematic_feature(session, 1, 'top_paw', valid_trials, go_cue_times)
paw_bot = extract_kinematic_feature(session, 1, 'bottom_paw', valid_trials, go_cue_times)
```

iii. Documented as extracted from bottom camera (cam 1).

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw feature, velocity magnitude is computed as `sqrt(dx^2 + dy^2)` at native video frame rate. Missing position values are filled with linear interpolation (nearest valid values). The top_paw and bottom_paw velocities are averaged. The result is interpolated to the neural time axis.

ii.
```python
# Fill missing for non-tongue features:
if not is_tongue:
    for pos in [x_pos, y_pos]:
        mask = np.isnan(pos)
        if mask.any() and not mask.all():
            valid_idx = np.where(~mask)[0]
            pos[mask] = np.interp(np.where(mask)[0], valid_idx, pos[valid_idx])

# Average top and bottom paw:
if paw_top is not None and paw_bot is not None:
    paw_vel = (paw_top + paw_bot) / 2.0
```

iii. Documented as averaging top_paw and bottom_paw, with missing values filled with nearest available value.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue velocity: discretized per session at 50th percentile. Values >= threshold -> 1, < threshold -> 0.

ii.
```python
paw_vel_disc = discretize_per_session(paw_vel)
```

iii. Per decoder task specification.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue velocity: video frame times minus vidshift minus go cue time, then linear interpolation to neural time axis.

ii.
```python
aligned_frame_times = frame_times[:n_frames] - vidshift - go_time
interp_fn = interp1d(aligned_frame_times, vel, kind='linear', bounds_error=False, fill_value=np.nan)
```

iii. Same alignment approach as tongue velocity.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Derived from `motionEnergy_*.mat` files, specifically the `me.data` field (per-trial motion energy time series at 400 Hz) and `me.moveThresh` (loaded but not used for discretization).

ii.
```python
def load_motion_energy(filepath):
    mat = scipy.io.loadmat(filepath, squeeze_me=False)
    me = mat['me']
    data_field = me['data'][0, 0]
    # ...
    threshold = float(thresh_field.flatten()[0])
    return {'data': me_data, 'moveThresh': threshold}
```

iii. Documented as loaded from separate `motionEnergy_*.mat` files.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Per-trial motion energy is interpolated to the neural time axis using video frame times (from `traj{1}` camera data) aligned to go cue. NaN values are filled with nearest valid values or set to 0 if all NaN.

ii.
```python
def interpolate_motion_energy(me_data, session, valid_trials, go_cue_times):
    for ti, trial_idx in enumerate(valid_trials):
        aligned_frame_times = frame_times - vidshift - go_time
        interp_fn = interp1d(aligned_frame_times, me_trial, kind='linear', bounds_error=False, fill_value=np.nan)
        me_interp[:, ti] = interp_fn(TIME_AXIS)
    # Fill NaN with nearest:
    for ti in range(n_trials):
        col = me_interp[:, ti]
        mask = np.isnan(col)
        if mask.any() and not mask.all():
            valid_idx = np.where(~mask)[0]
            col[mask] = np.interp(np.where(mask)[0], valid_idx, col[valid_idx])
```

iii. Documented as matching `loadMotionEnergy.m` logic, with video offset alignment.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same as other continuous outputs: discretized per session at 50th percentile.

ii.
```python
me_disc = discretize_per_session(me_interp)
```

iii. Per decoder task specification.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Uses video frame times from `traj{1}` (first camera), subtracting vidshift and go cue time, then interpolating to neural time axis. Falls back to constructed frame times if unavailable.

ii.
```python
if frame_times is None:
    frame_times = np.arange(1, n_frames + 1) / 400.0
    aligned_frame_times = frame_times - 0.5 - go_time
else:
    aligned_frame_times = frame_times - vidshift - go_time
```

iii. Documented as matching `loadMotionEnergy.m`, with the same fallback using 0.5 offset.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple strategies are used:
- Sessions without neural data (behavior-only) are skipped entirely.
- Sessions with fewer than 2 valid trials are skipped.
- Sessions/probes with 0 valid units after filtering are skipped.
- Missing tongue velocity, paw velocity, or motion energy are replaced with zeros.
- NaN values in kinematic/ME data are filled with nearest-neighbor interpolation or set to 0 for all-NaN trials.
- Trials with NaN or 0 go cue times are skipped in spike processing.
- Dropped video frames (NaN NdroppedFrames) cause the trial's kinematic data to be skipped.

ii.
```python
if n_valid < 2:
    print(f"  SKIPPING: fewer than 2 valid trials")
    return None

if np.isnan(go_time) or go_time == 0:
    continue

if tongue_vel is None:
    tongue_vel_disc = np.zeros((N_TIMEBINS, n_trials), dtype=np.int32)

# NaN filling:
if mask.any() and not mask.all():
    valid_idx = np.where(~mask)[0]
    col[mask] = np.interp(np.where(mask)[0], valid_idx, col[valid_idx])
elif mask.all():
    velocities[:, ti] = 0.0
```

iii. Documented in CONVERSION_NOTES under "Known Issues": missing motion energy for 8 sessions (set to zeros), all-zero neural data in some late trials, degenerate tongue velocity.

## 11-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading HDF5/MAT files (especially large files with trajectory data requiring dereferencing of many HDF5 object references).
2. Spike binning and smoothing: triple-nested loops over units, trials, and time bins in `align_and_bin_spikes`.
3. Kinematic feature extraction: per-trial interpolation with separate loops for each camera/feature.

ii.
```python
# Spike processing loop (unit x trial):
for ui, ci in enumerate(cluster_indices):
    for ti, trial_idx in enumerate(valid_trials):
        counts, _ = np.histogram(aligned_times, bins=edges)
        fr_smooth = causal_gaussian_smooth(fr, smooth_samples)

# Kinematic extraction loop (trial):
for ti, trial_idx in enumerate(valid_trials):
    interp_fn = interp1d(aligned_frame_times, vel, ...)
```

iii. Not explicitly discussed in CONVERSION_NOTES.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
1. The spike binning loop over units and trials in `align_and_bin_spikes` could use vectorized histogram operations or sparse matrix construction.
2. The convolution loop in `causal_gaussian_smooth` over columns could use `scipy.ndimage.convolve1d` with axis parameter.
3. The NaN filling loops over trials could use vectorized `fillna`-like operations across the 2D array.
4. The per-trial output construction loop could use array broadcasting/stacking.

ii.
```python
# Could vectorize the inner convolution loop:
for j in range(x.shape[1]):
    out[:, j] = np.convolve(x[:, j], kern, mode='same')

# Could vectorize NaN filling:
for ti in range(n_trials):
    col = velocities[:, ti]
    mask = np.isnan(col)
    ...
```

iii. Not discussed in CONVERSION_NOTES.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats several operations:
1. The `extract_kinematic_feature` function is called separately for tongue, top_paw, and bottom_paw, each time re-parsing the trajectory data structure and re-computing video alignment for the same trials.
2. The NaN filling pattern (nearest-neighbor interpolation) is duplicated identically in `extract_kinematic_feature` and `interpolate_motion_energy`.
3. Video offset computation accesses the same session data multiple times.

ii.
```python
tongue_vel = extract_kinematic_feature(session, 0, 'tongue', valid_trials, go_cue_times)
paw_top = extract_kinematic_feature(session, 1, 'top_paw', valid_trials, go_cue_times)
paw_bot = extract_kinematic_feature(session, 1, 'bottom_paw', valid_trials, go_cue_times)
```

iii. Not discussed in CONVERSION_NOTES.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several things are loaded/processed but not used in the final output:
1. `me.moveThresh` is loaded from motion energy files but never used (discretization uses 50th percentile instead).
2. The `sample` and `delay` event times are loaded but never used.
3. The `no` (no-response) indicator is loaded but not used in trial filtering (already excluded by `hit | miss`).
4. The code loads trajectory `NdroppedFrames` and `featNames` for all cameras even when only specific features are needed.
5. Non-ALM probe data is loaded before being skipped in the filtering step.

ii.
```python
session['sample'] = ev['sample'][()].flatten()
session['delay'] = ev['delay'][()].flatten()
session['no'] = bp['no'][()].flatten().astype(float)
threshold = float(thresh_field.flatten()[0])  # moveThresh loaded but unused
```

iii. Not discussed in CONVERSION_NOTES.
