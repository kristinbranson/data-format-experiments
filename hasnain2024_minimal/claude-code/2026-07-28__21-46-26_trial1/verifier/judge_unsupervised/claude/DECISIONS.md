# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans two data directories (`Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior`) for `data_structure_*.mat` files and paired `motionEnergy_*.mat` files. It tries HDF5 (h5py) first, then falls back to scipy.io (MATLAB v5/v7). Each session file is loaded individually, extracting behavioral fields (`R`, `L`, `hit`, `miss`, `early`, `autowater`, `stim`, events), cluster/spike data, trajectory data, and video offset. Motion energy is loaded separately from `motionEnergy_*.mat` files.

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
            ...

def load_session_data(filepath):
    try:
        with h5py.File(filepath, 'r') as f:
            ...
        return _load_session_data_h5(filepath)
    except (OSError, ValueError):
        pass
    return load_session_data_v5(filepath)
```

iii. The AI documented: "MATLAB v7.3 (HDF5) files loaded with h5py; v5/v7 files loaded with scipy.io as fallback." It identified 47 total session files across both directories, with 2 skipped (behavior-only, no neural data).

## 1-b. How are the data split into subjects (mice)?

i. Animal identity is extracted from the filename (`data_structure_ANIMAL_DATE.mat`) by parsing the prefix before the date. Unique subjects are collected across all sessions and stored as a sorted list. A `subject_idx` array maps each session to its subject.

ii.
```python
parts = basename.replace('data_structure_', '').replace('.mat', '').split('_')
animal = parts[0]

unique_subjects = sorted(set(all_animals))
subject_idx = np.array([unique_subjects.index(a) for a in all_animals], dtype=np.int64)
```

iii. The AI noted 14 subjects across both datasets (10 from Ephys_Behavior vs. paper's 9, 4 from RandomizedDelay matching paper). The extra mouse in Ephys_Behavior was noted as a discrepancy.

## 1-c. How are the data split into sessions?

i. Each `data_structure_*.mat` file is treated as one session. Sessions are processed sequentially and indexed in order. Sessions with fewer than 2 valid trials or no valid neural clusters are skipped. The final dataset has 45 sessions.

ii.
```python
for idx, sf in enumerate(session_files):
    result = process_session(sf['data_file'], sf['me_file'], idx)
    if result is None:
        continue
    all_neural.append(result['neural'])
    ...
```

iii. The AI documented "2 sessions skipped (JEB24_2023-10-03, JEB24_2023-10-04): behavior-only, no neural data." Final count: 25 from Ephys_Behavior, 20 from RandomizedDelay (paper reports 19).

## 1-d. How are the data split into trials?

i. Trials within each session are identified by the `Ntrials` field and per-trial arrays (`R`, `L`, `hit`, `miss`, `early`, etc.). Spike data is assigned to trials via the `trial` field in each cluster. Valid trials are selected by a boolean mask, and their indices are used to extract neural and behavioral data.

ii.
```python
Ntrials = session['Ntrials']
trial_mask = (
    (session['stim_enable'] == 0) &
    (session['early'] == 0) &
    ((session['hit'] == 1) | (session['miss'] == 1))
)
valid_trials = np.where(trial_mask)[0]
```

iii. The AI documented trial counts ranging from 137 to 472 valid trials per session.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by three criteria: (1) no optogenetic stimulation (`stim.enable == 0`), (2) no early licks (`early == 0`), (3) the mouse responded (`hit == 1 | miss == 1`). This includes both correct and incorrect trials (but excludes ignore/no-response trials). Sessions with fewer than 2 valid trials are skipped entirely.

ii.
```python
trial_mask = (
    (session['stim_enable'] == 0) &
    (session['early'] == 0) &
    ((session['hit'] == 1) | (session['miss'] == 1))
)
valid_trials = np.where(trial_mask)[0]
if n_valid < 2:
    return None
```

iii. The AI justified including miss trials: "This includes both correct (hit) and incorrect (miss) trials for outcome decoding." The reference code default conditions only include `hit` trials, but the decoder task requires predicting outcome, necessitating both hit and miss trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `clu` (cluster) field in each session's `obj` structure. Specifically, `trialtm` (spike times within trial, relative to trial start), `trial` (trial assignment for each spike), and `quality` (unit quality label) are used per unit, per probe.

ii.
```python
# HDF5 path
unit['trialtm'] = f[tm_ref][()].flatten()
unit['trial'] = f[trial_ref][()].flatten().astype(int)
unit['quality'] = h5_read_string(f, f[q_ref])
```

iii. The AI noted this follows `getSeq.m` which bins `clu.trialtm` aligned to events.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue, binned into 5ms bins over [-2.5, +2.5]s (1000 bins), converted to firing rate (counts/dt), and smoothed with a causal Gaussian kernel. The kernel uses `gausswin(N)` with N=15 samples, left half zeroed for causality, then normalized.

ii.
```python
TMIN, TMAX, DT = -2.5, 2.5, 1/200
SMOOTH_MS = 15
EDGES = np.arange(TMIN, TMAX + DT/2, DT)

# In align_and_bin_spikes:
aligned_times = spike_times[spike_mask] - go_time
counts, _ = np.histogram(aligned_times, bins=edges)
fr = counts / dt
fr_smooth = causal_gaussian_smooth(fr, smooth_samples)

# In causal_gaussian_smooth:
N = kernel_width_samples  # 15
sigma = N / 6
kern = np.exp(-0.5 * ((n - center) / sigma) ** 2)
kern[:N // 2] = 0
kern = kern / kern.sum()
return np.convolve(x, kern, mode='same')
```

iii. The AI documented: "Bin size: 5 ms (dt = 1/200 s), giving 1000 time bins per trial. Smoothing: Causal Gaussian kernel with width N=15 samples (matching mySmooth.m)."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) Quality filter matching `findClusters.m` with `quality='all'`: excludes units labeled `'garbage'`, `'gabrga'`, `'noisy'`, or `'real?'`; unlabeled units are included. (2) Firing rate filter matching `removeLowFRClusters.m`: removes units with mean FR <= 0.5 Hz. Additionally, only ALM probes are included (non-ALM probes like M1TJ and brainstem are skipped).

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
    return indices

# Firing rate filter:
mean_frs = np.mean(np.mean(trialdat, axis=2), axis=0)
fr_mask = mean_frs > LOW_FR  # LOW_FR = 0.5

# ALM-only filter:
if 'ALM' not in loc_upper:
    continue
```

iii. The AI documented matching `findClusters.m` and `removeLowFRClusters.m`. However, the mean FR computation differs from the reference: the AI averages across all trials then time, while the reference code averages the PSTH (condition-averaged firing rates) across conditions then time.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes are aligned by subtracting the go cue time (`goCue`) from each spike's within-trial time (`trialtm`), then binning into edges from -2.5 to +2.5 s.

ii.
```python
go_time = go_cue_times[trial_idx]
aligned_times = spike_times[spike_mask] - go_time
counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. The AI documented: "Alignment: Spikes aligned to go cue onset (goCue event)" matching `params.alignEvent = 'goCue'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 5 ms (DT = 1/200 s), producing 1000 time bins per trial over the [-2.5, 2.5] s window. No rebinning is applied after the initial binning.

ii.
```python
DT = 1 / 200  # 5 ms bins
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
```

iii. The AI documented: "Bin size: 5 ms (dt = 1/200 s), giving 1000 time bins per trial."

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not derived from raw data variables. It is constructed as a deterministic time axis based on the bin edges and bin width, representing the center of each 5ms bin from -2.5 to +2.5 seconds.

ii.
```python
TIME_AXIS = EDGES[:-1] + DT / 2  # center of each bin
input_trial = TIME_AXIS.reshape(1, -1).astype(np.float32)
```

iii. The AI documented: "Time from go cue onset (seconds): continuous time axis [-2.4975, ..., 2.4975] in 5ms steps."

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is computed as `edges[:-1] + DT/2`, giving the center of each bin. This is the same for every trial and session. No trial-specific processing is needed.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
```

iii. No specific justification provided beyond matching `getSeq.m`'s time axis construction: `obj.time = edges + params.dt/2; obj.time = obj.time(1:end-1);`

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The same `TIME_AXIS` is used for both the neural data (as bin centers for spike histograms) and the input. They are inherently aligned since they share the same time vector.

ii.
```python
# Neural binning uses EDGES -> TIME_AXIS
# Input uses TIME_AXIS directly
input_trial = TIME_AXIS.reshape(1, -1).astype(np.float32)
```

iii. No explicit justification needed; alignment is inherent.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `obj.bp.R`, which is a per-trial binary indicator (1 = right lick, 0 = left lick).

ii.
```python
session['R'] = bp['R'][()].flatten().astype(float)
lick_direction = session['R'][valid_trials].astype(np.int32)  # R=1, L=0
```

iii. The AI documented: "Lick direction: R=1 (right), L=0 (left) from obj.bp.R field."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The `R` field is directly used as the lick direction (right=1, left=0). It is tiled across all time bins as a per-trial constant.

ii.
```python
lick_direction = session['R'][valid_trials].astype(np.int32)
output_trial[0, :] = lick_direction[ti]  # per-trial, tiled across time
```

iii. Matches decoder task specification: "left = 0, right = 1, per-trial."

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj.bp.autowater`, where `autowater=1` indicates water-cued (WC) context and `autowater=0` indicates delayed-response (DR) context.

ii.
```python
session['autowater'] = bp['autowater'][()].flatten().astype(float)
context = (1 - session['autowater'][valid_trials]).astype(np.int32)  # DR=1, WC=0
```

iii. The AI documented: "Behavioral context: DR=1, WC=0 from obj.bp.autowater (autowater=1 means WC)."

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The `autowater` field is inverted (`1 - autowater`) so that DR=1 and WC=0, matching the decoder task specification. It is tiled across time bins as a per-trial constant.

ii.
```python
context = (1 - session['autowater'][valid_trials]).astype(np.int32)
output_trial[1, :] = context[ti]  # per-trial
```

iii. Matches decoder task specification: "WC = 0, DR = 1, per-trial."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit`, where `hit=1` means the trial was correct and `hit=0` (with `miss=1`) means incorrect.

ii.
```python
session['hit'] = bp['hit'][()].flatten().astype(float)
outcome = session['hit'][valid_trials].astype(np.int32)
```

iii. The AI documented: "Outcome: correct=1, incorrect=0 from obj.bp.hit."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The `hit` field is used directly as the outcome variable. It is tiled across time bins as a per-trial constant.

ii.
```python
outcome = session['hit'][valid_trials].astype(np.int32)
output_trial[2, :] = outcome[ti]  # per-trial
```

iii. Matches decoder task specification: "incorrect = 0, correct = 1, per-trial."

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the trajectory data (`obj.traj`) for the side camera (camera index 0), specifically the `'tongue'` feature. The `ts` array provides x,y positions, and `frameTimes` provides timing. `vidshift` (from `sglx` and `bp.ev.bitStart`) is used for temporal alignment.

ii.
```python
tongue_vel = extract_kinematic_feature(session, 0, 'tongue', valid_trials, go_cue_times)
# Inside extract_kinematic_feature:
x_pos = ts[feat_idx, 0, :].copy()  # x position
y_pos = ts[feat_idx, 1, :].copy()  # y position
```

iii. The AI documented: "Tongue velocity: Extracted from side camera (cam 0), 'tongue' feature."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Velocity is computed at 400 Hz from x,y positions using `np.gradient`, then magnitude is taken as `sqrt(dx^2 + dy^2)`. For the tongue feature, velocity is set to 0 where tongue is not visible (NaN positions). The velocity signal is then interpolated to the neural time axis using linear interpolation, aligned to the go cue with video offset correction.

ii.
```python
dx = np.gradient(x_pos, dt_vid)  # dt_vid = 1/400
dy = np.gradient(y_pos, dt_vid)
vel = np.sqrt(dx**2 + dy**2)

if is_tongue:
    nan_mask = np.isnan(x_pos) | np.isnan(y_pos)
    vel[nan_mask] = 0.0

aligned_frame_times = frame_times[:n_frames] - vidshift - go_time
interp_fn = interp1d(aligned_frame_times, vel, kind='linear', bounds_error=False, fill_value=np.nan)
velocities[:, ti] = interp_fn(TIME_AXIS)
```

iii. The AI documented: "Set to 0 when tongue not visible (NaN positions), matching paper: 'Missing values were filled in with the nearest available value for all features, except for the tongue'."

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per-session 50th percentile (median) threshold is used. Values >= threshold are class 1 ("high"), values < threshold are class 0 ("low"). However, since tongue velocity is 0 whenever the tongue is not visible (majority of time), the median is 0, making all values >= 0 classify as class 1. This results in a degenerate all-high output.

ii.
```python
def discretize_per_session(values, percentile=50):
    threshold = np.percentile(valid, percentile)
    discretized = (values >= threshold).astype(np.int32)
    return discretized

tongue_vel_disc = discretize_per_session(tongue_vel)
```

iii. The AI documented this as a known issue: "The tongue is only visible during licks (~10-20% of time). When invisible, velocity = 0. The 50th percentile threshold is 0, making all values >= 0 classify as 'high' (class 1)."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Video frame times are corrected by the video offset (`vidshift`) and aligned to the go cue, then the velocity signal is interpolated to the neural time axis (`TIME_AXIS`) using linear interpolation. Remaining NaN values after interpolation are filled with nearest valid values, or 0 if all values are NaN.

ii.
```python
aligned_frame_times = frame_times[:n_frames] - vidshift - go_time
interp_fn = interp1d(aligned_frame_times, vel, kind='linear', bounds_error=False, fill_value=np.nan)
velocities[:, ti] = interp_fn(TIME_AXIS)

# Fill NaN with nearest
for ti in range(n_trials):
    col = velocities[:, ti]
    mask = np.isnan(col)
    if mask.any() and not mask.all():
        valid_idx = np.where(~mask)[0]
        col[mask] = np.interp(np.where(mask)[0], valid_idx, col[valid_idx])
```

iii. The AI documented following `findPosition.m` logic for frame time alignment and video offset correction.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from trajectory data for the bottom camera (camera index 1), using `'top_paw'` and `'bottom_paw'` features from the `ts` array.

ii.
```python
paw_top = extract_kinematic_feature(session, 1, 'top_paw', valid_trials, go_cue_times)
paw_bot = extract_kinematic_feature(session, 1, 'bottom_paw', valid_trials, go_cue_times)
```

iii. The AI documented: "Paw velocity: Extracted from bottom camera (cam 1), averaged 'top_paw' and 'bottom_paw'."

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw feature, x,y positions are extracted, NaN values are filled with nearest valid values (non-tongue feature handling), velocity is computed at 400 Hz using `np.gradient`, and the velocity magnitude is taken. The two paw velocities are then averaged. The result is interpolated to the neural time axis.

ii.
```python
# Non-tongue NaN filling:
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

iii. The AI documented matching the paper's method of filling missing values for non-tongue features with nearest values.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue velocity: per-session 50th percentile threshold. Values >= threshold are class 1, values < threshold are class 0.

ii.
```python
paw_vel_disc = discretize_per_session(paw_vel)
```

iii. Matches decoder task specification: "discretized into two bins with per-session threshold: 0: < 50th percentile, 1: >= 50th percentile."

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same approach as tongue velocity: video frame times corrected by `vidshift`, aligned to go cue, then interpolated to neural time axis. NaN values filled with nearest valid values.

ii.
```python
aligned_frame_times = frame_times[:n_frames] - vidshift - go_time
interp_fn = interp1d(aligned_frame_times, vel, kind='linear', bounds_error=False, fill_value=np.nan)
velocities[:, ti] = interp_fn(TIME_AXIS)
```

iii. Same as tongue velocity alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from separate `motionEnergy_*.mat` files containing per-trial motion energy arrays (`me.data`) and a per-session movement threshold (`me.moveThresh`). Frame timing comes from the trajectory data's `frameTimes` field.

ii.
```python
def load_motion_energy(filepath):
    mat = scipy.io.loadmat(filepath, squeeze_me=False)
    me = mat['me']
    data_field = me['data'][0, 0]
    thresh_field = me['moveThresh'][0, 0]
    ...
    return {'data': me_data, 'moveThresh': threshold}
```

iii. The AI documented: "Motion energy: From motionEnergy_*.mat files."

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The per-trial motion energy arrays are loaded as-is (no recomputation). They are interpolated to the neural time axis using video frame times, with video offset correction. NaN values are filled with nearest valid values or set to 0 if all NaN. The `moveThresh` from the file is loaded but not used (the decoder task specifies 50th percentile thresholding instead).

ii.
```python
def interpolate_motion_energy(me_data, session, valid_trials, go_cue_times):
    aligned_frame_times = frame_times - vidshift - go_time
    interp_fn = interp1d(aligned_frame_times, me_trial,
                        kind='linear', bounds_error=False, fill_value=np.nan)
    me_interp[:, ti] = interp_fn(TIME_AXIS)
```

iii. The AI documented matching `loadMotionEnergy.m` logic for interpolation and alignment.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same as other continuous outputs: per-session 50th percentile threshold. The original `moveThresh` from the motion energy file (which is a manually set threshold at the trough of the bimodal ME distribution) is not used for discretization.

ii.
```python
me_disc = discretize_per_session(me_interp)
```

iii. The AI follows the decoder task specification rather than the paper's manual per-session threshold.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is interpolated from video frame times (400 Hz) to the neural time axis, aligned to the go cue with video offset correction. If trajectory frame times are unavailable, synthetic frame times at 400 Hz with a 0.5s offset are used as fallback.

ii.
```python
if frame_times is None:
    frame_times = np.arange(1, n_frames + 1) / 400.0
    aligned_frame_times = frame_times - 0.5 - go_time  # fallback offset
else:
    aligned_frame_times = frame_times - vidshift - go_time
```

iii. The AI documented matching `loadMotionEnergy.m` for interpolation.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several types of missing data are handled:
- **Missing motion energy files**: 8 sessions lack loadable ME files; motion energy set to all zeros, discretized as all-low.
- **Missing trajectory data**: If a camera or feature is not available, the corresponding velocity is set to all zeros.
- **NaN in frame times**: Trials with all-NaN frame times are skipped or use synthetic frame times.
- **NaN in positions**: For non-tongue features, NaN values are filled with nearest valid values. For tongue, NaN positions result in velocity = 0.
- **NaN in go cue times**: Trials with `go_time == 0` or `go_time == NaN` are skipped during spike binning.
- **Dropped frames**: Trials with `NdroppedFrames == NaN` are skipped for kinematic extraction.
- **All-zero neural data**: Late trials in 2 sessions (37 and 44) have all-zero neural data (go cue times beyond recording window); these are kept in the dataset.
- **Missing probe location**: Defaults to 'ALM' if location cannot be read.

ii.
```python
# Missing ME:
if me_interp is not None:
    me_disc = discretize_per_session(me_interp)
else:
    me_disc = np.zeros((N_TIMEBINS, n_trials), dtype=np.int32)

# NaN go cue:
if np.isnan(go_time) or go_time == 0:
    continue

# Dropped frames:
if isinstance(ndf, float) and np.isnan(ndf):
    continue

# Fill NaN with nearest:
col[mask] = np.interp(np.where(mask)[0], valid_idx, col[valid_idx])
```

iii. The AI documented known issues including degenerate tongue velocity, format issues with some v5/v7 sessions' paw data, and missing motion energy for 8 sessions.

## 11-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading HDF5 files**: Each session file contains large arrays of spike data with dereferenced object references that must be read individually per unit.
2. **Spike binning**: Triple-nested loop over units, trials, and spikes (masking spikes by trial, aligning, histogramming).
3. **Kinematic extraction**: Per-trial velocity computation, NaN handling, and interpolation for multiple features across two cameras.
4. **Motion energy interpolation**: Per-trial interpolation from 400 Hz to 5 ms bins.

ii.
```python
# Spike binning - innermost loop:
for ui, ci in enumerate(cluster_indices):
    for ti, trial_idx in enumerate(valid_trials):
        spike_mask = spike_trials == trial_num
        aligned_times = spike_times[spike_mask] - go_time
        counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. No explicit discussion of performance in the AI's notes.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several nested loops could be vectorized:
1. **Spike binning**: The inner loop over trials for each unit could use vectorized groupby-histogram operations.
2. **Causal Gaussian smoothing**: Loops over columns (`for j in range(x.shape[1])`) when input is 2D.
3. **NaN filling**: Per-trial NaN filling loops could use vectorized operations.
4. **Kinematic feature extraction**: Per-trial velocity computation and interpolation.

ii.
```python
# Smoothing loop over columns:
for j in range(x.shape[1]):
    out[:, j] = np.convolve(x[:, j], kern, mode='same')

# NaN filling loop:
for ti in range(n_trials):
    col = velocities[:, ti]
    mask = np.isnan(col)
    ...
```

iii. No discussion of vectorization in the AI's notes.

## 11-c. What processing does the code repeat multiple times?

i.
1. **Kinematic extraction** is called separately for tongue, top_paw, and bottom_paw, each running very similar per-trial loops.
2. **NaN filling** logic is duplicated between `extract_kinematic_feature` and `interpolate_motion_energy`.
3. **Video offset computation** is done once during session loading but the shift is applied multiple times (once per kinematic feature, once for ME).

ii.
```python
# Called three times with different features:
tongue_vel = extract_kinematic_feature(session, 0, 'tongue', ...)
paw_top = extract_kinematic_feature(session, 1, 'top_paw', ...)
paw_bot = extract_kinematic_feature(session, 1, 'bottom_paw', ...)
```

iii. No discussion of code repetition in the AI's notes.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
1. **`moveThresh` from motion energy files**: Loaded but never used; the decoder task specifies 50th percentile thresholding instead.
2. **`delay` event times**: Loaded from the session data but never used.
3. **`sample` event times**: Loaded but never used.
4. **`no` (no-response) field**: Loaded but not used in trial filtering (implicitly excluded by `hit | miss`).
5. **`L` (left lick) field**: Loaded but not used; lick direction uses only `R`.
6. **Non-ALM probe data**: Cluster data is fully loaded for all probes before filtering by location.
7. **All trajectory feature names**: All feature names are loaded but only 'tongue', 'top_paw', and 'bottom_paw' are used.

ii.
```python
session['delay'] = ev['delay'][()].flatten()  # never used
session['sample'] = ev['sample'][()].flatten()  # never used
session['no'] = bp['no'][()].flatten()  # never used
session['L'] = bp['L'][()].flatten()  # never used

# moveThresh loaded but not used for discretization:
return {'data': me_data, 'moveThresh': threshold}
```

iii. No discussion of unnecessary processing in the AI's notes.
