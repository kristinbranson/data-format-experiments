# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI defines a session registry (`EPHYS_SESSIONS` and `RANDOMIZED_DELAY_SESSIONS`) listing all 44 sessions (25 Ephys_Behavior + 19 RandomizedDelay_Ephys_Behavior) with animal name, date, probe numbers, and data directory. Each session is loaded individually via `load_session_data()`, which uses `mat73.loadmat()` for MATLAB v7.3 files and `scipy.io.loadmat()` (via `_load_v5_session()`) for MATLAB v5 files. Motion energy is loaded from separate `motionEnergy_*.mat` files. Sessions are processed sequentially in a loop.

ii.
```python
ALL_SESSIONS = EPHYS_SESSIONS + RANDOMIZED_DELAY_SESSIONS

def load_session_data(anm, date, data_dir):
    data_path = os.path.join(DATA_ROOT, data_dir, f'data_structure_{anm}_{date}.mat')
    try:
        obj = mat73.loadmat(data_path)['obj']
    except TypeError:
        obj = _load_v5_session(data_path)
    # Load motion energy
    me_path = os.path.join(DATA_ROOT, data_dir, f'motionEnergy_{anm}_{date}.mat')
    ...
```

iii. The AI documented in CONVERSION_NOTES Step 2 that data is organized in two directories (Ephys_Behavior and RandomizedDelay_Ephys_Behavior), identified per-animal loading scripts in the reference code, and built the session registry to match those scripts, excluding sessions that were commented out (e.g., JEB23_2023-10-20) or not listed in the loading scripts (e.g., JEB24_2023-10-03).

## 1-b. How are the data split into subjects (mice)?

i. Subjects are extracted from the animal name field in the session registry. A set of unique animal names is collected during processing and sorted alphabetically to create the `subjects` list. Each session's animal name is mapped to an index into this list via `subject_idx`.

ii.
```python
subjects_set = set()
for i, (anm, date, probes, data_dir) in enumerate(sessions):
    ...
    subjects_set.add(anm)
subjects = sorted(subjects_set)
subject_idx = np.array([subjects.index(anm) for anm in all_animals], dtype=np.int64)
```

iii. The AI identified 14 unique subjects across both datasets: 10 from Ephys_Behavior (EKH1, EKH3, JEB6, JEB7, JEB13, JEB14, JEB15, JEB19, JGR2, JGR3) and 4 from RandomizedDelay (JEB11, JEB12, JEB23, JEB24).

## 1-c. How are the data split into sessions?

i. Each `.mat` file corresponds to one recording session. The session registry defines which sessions to include based on the reference code's per-animal loading scripts. All 44 sessions (25 Ephys_Behavior + 19 RandomizedDelay) are included. Sessions with fewer than 2 valid trials or fewer than 10 neurons after filtering are skipped.

ii.
```python
EPHYS_SESSIONS = [
    ('EKH1', '2021-08-07', [2], 'Ephys_Behavior'),
    ...  # 25 sessions
]
RANDOMIZED_DELAY_SESSIONS = [
    ('JEB11', '2022-05-10', [1], 'RandomizedDelay_Ephys_Behavior'),
    ...  # 19 sessions
]
ALL_SESSIONS = EPHYS_SESSIONS + RANDOMIZED_DELAY_SESSIONS
```

iii. The AI documented that it included all sessions from both datasets based on the reference loading scripts. It noted the paper reports "25 sessions, nine mice" for DR and "19 sessions, four mice" for randomized delay.

## 1-d. How are the data split into trials?

i. Trial count is read from `bp.Ntrials`. The code processes all trials (0 to Ntrials-1), bins spikes for every trial, then filters to valid trials using `valid_mask`. Behavioral variables are extracted per trial from the behavioral data arrays.

ii.
```python
ntrials_total = int(bp['Ntrials'])
# Process spikes for ALL trials
trialdat = np.zeros((n_time, total_neurons, ntrials_total), dtype=np.float32)
for j in range(ntrials_total):
    trial_num = j + 1  # MATLAB 1-indexed
    ...
# Extract valid trials only
trialdat_valid = trialdat[:, :, valid_trials]
```

iii. The AI processes all trials first for neural data (including invalid ones) then subsets to valid trials, which ensures alignment between trial indices in the spike data and behavioral variables.

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded if they are: (1) early lick trials (`early=1`), (2) stimulation trials (`stim.enable=1`), or (3) no-response/ignore trials (`no=1`). Additionally, trials must have a lick response (`hit | miss`). The AI does NOT apply session-level behavioral criteria from the paper (minimum 40 correct DR trials/direction, minimum 20 correct WC trials/direction).

ii.
```python
valid_mask = ~early & ~stim_enable & ~no
valid_mask = valid_mask & (hit | miss)
valid_trials = np.where(valid_mask)[0]
```

iii. The AI documented in CONVERSION_NOTES Step 3 that the paper specifies behavioral inclusion criteria (>= 40 correct DR trials/direction, >= 20 correct WC trials/direction), and in Step 5 that trial exclusion should remove early, stim.enable, and no-response trials while keeping hit and miss. The session-level behavioral criteria were noted but not implemented in code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from cluster spike data in `obj.clu[probe]`. Specifically, `clu.trial` (trial membership for each spike) and `clu.trialtm` (spike times relative to trial start) are used. The alignment times come from `obj.bp.ev.goCue` (go cue event times).

ii.
```python
trial_arr = np.array(clu_probe['trial'][clu_idx]).flatten()
trialtm_arr = np.array(clu_probe['trialtm'][clu_idx]).flatten()
aligned = trialtm_arr[spk_mask] - align_times_all[j]
```

iii. The AI identified from the reference code that `alignSpikes.m` computes `trialtm_aligned = trialtm - event_time` and `getSeq.m` uses `histc` to bin the aligned spike times.

## 2-b. How is the `neural` data processed?

i. Processing follows the reference pipeline: (1) Align spike times to goCue by subtracting the goCue time, (2) Bin spikes using histogram into 10ms bins from -2.5s to 2.5s (500 bins), (3) Convert counts to firing rate by dividing by dt, (4) Apply causal Gaussian smoothing with window=15 bins and reflect boundary condition.

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
counts, _ = np.histogram(aligned, bins=edges)
fr = counts.astype(np.float32) / DT
trialdat[:, neuron_offset + i, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)
```

iii. The AI documented in CONVERSION_NOTES Step 1 that the reference pipeline is: alignSpikes -> getSeq (bin + smooth) -> removeLowFRClusters. Parameters were chosen from WorkingWithDataObjs.m: dt=1/100, smooth=15, bctype='reflect'.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering: (1) Cluster quality filtering excludes clusters with quality labels in {'garbage', 'gabrga', 'noisy', 'real?'}. (2) Low firing rate filtering removes neurons with mean firing rate <= 1 Hz (computed as mean across all trials and time points).

ii.
```python
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}

def get_valid_cluster_indices(clu_probe, excluded_qualities=EXCLUDED_QUALITIES):
    for i, q in enumerate(qualities):
        if q_stripped.lower() not in {e.lower() for e in excluded_qualities}:
            valid.append(i)

def remove_low_fr_neurons(trialdat, low_fr):
    mean_fr = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
    keep = mean_fr > low_fr
    return trialdat[:, keep, :], keep
```

iii. The AI documented that `findClusters.m` uses quality='all' which excludes garbage, gabrga, noisy, and real?. The paper states "firing rates exceeding 1 Hz" which informed the 1 Hz threshold. The AI noted a ~7% discrepancy in neuron counts vs the paper (1,532 vs 1,651 for DR sessions).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Per-trial neural data is aligned to the go cue onset. For each trial, the go cue time is read from `bp.ev.goCue`, and each spike time is shifted by subtracting the go cue time. The aligned spike times are then binned into the time window [-2.5, 2.5] seconds.

ii.
```python
ALIGN_EVENT = 'goCue'
goCue = np.array(ev['goCue']).flatten()
align_times_all = goCue
aligned = trialtm_arr[spk_mask] - align_times_all[j]
```

iii. The AI documented that `alignEvent = 'goCue'` is the default in the reference code, and the instructions explicitly specify alignment to go cue onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 10 ms (dt = 1/100 s), giving 500 time bins from -2.5s to 2.5s. No temporal rebinning is applied -- spikes are binned directly from spike times at this resolution. The reference code's default dt is 5 ms (1/200), but WorkingWithDataObjs.m uses 10 ms (1/100).

ii.
```python
DT = 1.0 / 100  # 10 ms bins
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
```

iii. The AI noted in CONVERSION_NOTES Step 4 that there is a discrepancy between the default dt (1/200 = 5ms) and WorkingWithDataObjs.m (1/100 = 10ms), and chose 10ms "as in WorkingWithDataObjs.m".

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is simply the time axis vector, which is derived from the parameters TMIN (-2.5), TMAX (2.5), and DT (0.01). It is the same for every trial and represents seconds relative to go cue onset.

ii.
```python
time_axis = edges[:-1] + DT / 2  # center of each bin
input_trials.append(time_axis.reshape(1, -1).astype(np.float32))
```

iii. The instructions specify "Time from go cue onset in seconds" as a decoder input, and the time axis already represents this since neural data is aligned to go cue.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing is needed -- the time axis is computed directly from the bin edges. Each value is the center of its time bin (edge + dt/2), ranging from approximately -2.495 to 2.495 seconds.

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
```

iii. The AI noted this is a straightforward continuous time vector that serves as the decoder input.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input uses the exact same time axis as the neural data, so they are inherently aligned. Both use `time_axis = edges[:-1] + DT/2` with the same TMIN, TMAX, DT parameters.

ii.
```python
# Neural and input share the same time_axis
neural_trials.append(trialdat_valid[:, :, t_idx].T.astype(np.float32))  # (n_neurons, n_time)
input_trials.append(time_axis.reshape(1, -1).astype(np.float32))  # (1, n_time)
```

iii. The AI ensured alignment by using the same time axis for all data streams.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from four behavioral variables: `bp.R` (right trial), `bp.L` (left trial), `bp.hit` (correct response), and `bp.miss` (incorrect response). These are combined to determine the actual direction the mouse licked.

ii.
```python
R = np.array(bp['R']).flatten().astype(bool)
L = np.array(bp['L']).flatten().astype(bool)
hit = np.array(bp['hit']).flatten().astype(bool)
miss = np.array(bp['miss']).flatten().astype(bool)
lick_right = (R & hit) | (L & miss)
lick_direction = lick_right[valid_trials].astype(np.int32)
```

iii. The AI reasoned that `R&hit` = mouse on a right trial responded correctly (licked right), and `L&miss` = mouse on a left trial responded incorrectly (licked right). The complement gives left licks.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Lick direction is computed as a binary per-trial label: right=1 if (R&hit)|(L&miss), left=0 otherwise. This is the actual lick direction (what the mouse did), not the instructed direction. The label is broadcast to all time points in the output array.

ii.
```python
lick_right = (R & hit) | (L & miss)
lick_direction = lick_right[valid_trials].astype(np.int32)
out[0, :] = lick_direction[t_idx]  # broadcast per-trial to time
```

iii. The instructions specify "Lick direction (left = 0, right = 1, per-trial)", which the AI implements correctly.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `bp.autowater`, which indicates whether the trial used automatic water delivery (water-cued, WC) or required a delayed response (DR).

ii.
```python
autowater = np.array(bp['autowater']).flatten().astype(bool)
context = (~autowater[valid_trials]).astype(np.int32)
```

iii. The AI documented that autowater=1 indicates WC (water-cued) context and autowater=0 indicates DR (delayed response) context, matching the reference code's usage.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Context is computed as a binary per-trial label: DR=1 (not autowater), WC=0 (autowater). The label is broadcast to all time points.

ii.
```python
context = (~autowater[valid_trials]).astype(np.int32)
out[1, :] = context[t_idx]
```

iii. The instructions specify "Behavioral context (WC = 0, DR = 1, per-trial)", which the AI implements correctly.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `bp.hit` (correct response) and `bp.miss` (incorrect response).

ii.
```python
hit = np.array(bp['hit']).flatten().astype(bool)
outcome = hit[valid_trials].astype(np.int32)
```

iii. The AI uses only the `hit` variable since the trial filter already ensures only hit or miss trials are included.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome is a binary per-trial label: correct=1 (hit), incorrect=0 (miss). Broadcast to all time points.

ii.
```python
outcome = hit[valid_trials].astype(np.int32)
out[2, :] = outcome[t_idx]
```

iii. The instructions specify "Outcome (incorrect = 0, correct = 1, per-trial)".

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from DLC tracking data in the bottom camera view (`obj.traj[1]`). Specifically, the `top_tongue` feature's x and y coordinates are extracted from `traj.ts` (trajectory time series). Frame timing comes from `traj.frameTimes`.

ii.
```python
traj_bottom = obj['traj'][1]  # bottom cam
# Find tongue feature index (top_tongue)
for i, name in enumerate(feat_names):
    if name == 'top_tongue':
        tongue_idx = i
        break
x = ts[:, 0, tongue_idx].copy()
y = ts[:, 1, tongue_idx].copy()
```

iii. The AI identified from the reference code that bottom camera features include tongue-related DLC markers and chose `top_tongue` as the primary tongue feature.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Processing: (1) Extract x,y positions of top_tongue from DLC data, (2) Compute velocity using `np.gradient()` scaled by VIDEO_FR (400 Hz), (3) Compute speed as magnitude sqrt(vx^2 + vy^2), (4) Set speed to NaN where tongue is not visible (NaN positions), (5) Interpolate to neural time axis using video offset correction, (6) Replace remaining NaN values with 0 (tongue not visible = no movement).

ii.
```python
vx_all = np.gradient(x) * VIDEO_FR
vy_all = np.gradient(y) * VIDEO_FR
speed = np.sqrt(vx_all**2 + vy_all**2)
speed[~valid] = np.nan
# Interpolate to neural time axis
f_interp = interp1d(old_time, speed, kind='linear', bounds_error=False, fill_value=np.nan)
tongue_vel[:, trix] = f_interp(time_axis)
# Set NaN to 0
tongue_vel = np.nan_to_num(tongue_vel, nan=0.0)
```

iii. The AI noted "Per paper methods: tongue missing values are NOT filled with nearest" and "Velocity is 0 where tongue is not visible." This differs from the reference code's `getKinematics.m` which applies standardization and PCA to all features. The AI computed raw velocity magnitude instead.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per-session 50th percentile thresholding. All non-NaN values across the session are pooled, the 50th percentile is computed, and values >= threshold get label 1 ("high"), values < threshold get label 0 ("low"). Special case: if threshold is 0 (common for tongue data with many zeros), it is replaced with machine epsilon so that exact zeros are "low" and any positive value is "high".

ii.
```python
def discretize_per_session(data_2d, percentile=50):
    all_vals = data_2d[~np.isnan(data_2d)]
    threshold = np.percentile(all_vals, percentile)
    if threshold == 0:
        threshold = np.finfo(np.float32).eps
    result = (data_2d >= threshold).astype(np.int32)
    result[np.isnan(data_2d)] = 0
    return result
```

iii. The instructions specify "discretized into two bins with per-session threshold: 0: < 50th percentile, 1: >= 50th percentile". The verification output shows tongue_velocity is ~90% "low" and ~10% "high", reflecting that the tongue is not visible most of the time (values are 0).

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue velocity is aligned via the video offset correction and interpolation. Video frame times are adjusted by subtracting `vidshift` (video-neural time offset) and the trial's go cue time. The resulting time series is linearly interpolated to the neural time axis.

ii.
```python
ft = np.array(traj_bottom['frameTimes'][trix]).flatten()
old_time = ft - vidshift - align_times[trix]
f_interp = interp1d(old_time, speed, kind='linear', bounds_error=False, fill_value=np.nan)
tongue_vel[:, trix] = f_interp(time_axis)
```

iii. The AI documented the video offset calculation matching `findVideoOffset.m`: `vidshift = mode(sglx.bitcode.bitstart)/sglx.fs - mode(bp.ev.bitStart)`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from DLC tracking data in the bottom camera view (`obj.traj[1]`). Features with "paw" in their name are used (typically `top_paw` and `bottom_paw`).

ii.
```python
traj_bottom = obj['traj'][1]
paw_indices = []
for i, name in enumerate(feat_names):
    if 'paw' in name.lower():
        paw_indices.append(i)
```

iii. The AI identified paw features from the bottom camera DLC feature list.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Processing: (1) Extract x,y positions of paw features, (2) Fill NaN values with nearest neighbor, (3) Compute velocity using `np.gradient()` scaled by VIDEO_FR, (4) Compute speed as magnitude, (5) Average speeds across paw features, (6) Interpolate to neural time axis with video offset correction, (7) Fill remaining NaN with nearest neighbor.

ii.
```python
for pidx in paw_indices:
    x = ts[:, 0, pidx].copy()
    y = ts[:, 1, pidx].copy()
    for arr in [x, y]:
        nans = np.isnan(arr)
        if nans.any() and not nans.all():
            valid = np.where(~nans)[0]
            nan_pos = np.where(nans)[0]
            nearest = np.searchsorted(valid, nan_pos).clip(0, len(valid)-1)
            arr[nans] = arr[valid[nearest]]
    vx = np.gradient(x) * VIDEO_FR
    vy = np.gradient(y) * VIDEO_FR
    speeds.append(np.sqrt(vx**2 + vy**2))
avg_speed = np.mean(speeds, axis=0)
```

iii. Unlike tongue, paw NaN values ARE filled with nearest neighbor before velocity computation. The AI averages velocities across paw features.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue velocity: per-session 50th percentile threshold via `discretize_per_session()`. The verification output shows paw_velocity is ~51% "low" and ~49% "high", which is well-balanced.

ii.
```python
paw_vel_disc = discretize_per_session(paw_vel_valid)
```

iii. Instructions specify 50th percentile per-session threshold.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same alignment as tongue velocity: frame times adjusted by video offset and go cue time, then linearly interpolated to neural time axis.

ii.
```python
ft = np.array(traj_bottom['frameTimes'][trix]).flatten()
old_time = ft - vidshift - align_times[trix]
f_interp = interp1d(old_time, avg_speed, kind='linear', bounds_error=False, fill_value=np.nan)
paw_vel[:, trix] = f_interp(time_axis)
```

iii. Same video offset correction approach as for tongue velocity.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from separate pre-computed files (`motionEnergy_ANM_DATE.mat`). The variable `me.data` (or nested `me.data.data`) contains per-trial motion energy time series at 400 Hz.

ii.
```python
me_path = os.path.join(DATA_ROOT, data_dir, f'motionEnergy_{anm}_{date}.mat')
me_file = scipy.io.loadmat(me_path)
me_var = me_file['me']
if me_data.dtype.names and 'data' in me_data.dtype.names:
    me_raw = inner['data']
```

iii. The AI documented that motion energy files are separate from the main data structure files, matching the reference code's `loadMotionEnergy.m`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Processing: (1) Load pre-computed motion energy, (2) Get video frame times from side camera (`obj.traj[0]`), (3) Apply video offset correction, (4) Interpolate to neural time axis using linear interpolation, (5) Fill NaN values with nearest neighbor (matching `fillmissing(,'nearest')` in MATLAB).

ii.
```python
def get_motion_energy_aligned(me_raw, obj, align_times, time_axis, vidshift, ntrials):
    traj_view0 = obj['traj'][0]  # side cam
    me_trial = np.array(me_raw[trix, 0]).flatten()
    ft = np.array(traj_view0['frameTimes'][trix]).flatten()
    old_time = ft - vidshift - align_times[trix]
    f_interp = interp1d(old_time, me_trial, kind='linear', bounds_error=False, fill_value=np.nan)
    me_aligned[:, trix] = f_interp(time_axis)
    # Fill NaN with nearest
    ...
```

iii. The AI documented that the reference `loadMotionEnergy.m` aligns motion energy using `interp1(frameTimes - vidshift - alignTime, me.data, taxis)`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same as tongue/paw velocity: per-session 50th percentile threshold via `discretize_per_session()`. The verification output shows motion_energy is exactly 50%/50% split, confirming the percentile-based threshold works correctly.

ii.
```python
me_disc = discretize_per_session(me_valid)
```

iii. The instructions specify 50th percentile per-session threshold. The AI ignores the pre-computed `me.moveThresh` threshold from the motion energy file.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned using the side camera's frame times, adjusted by video offset and go cue time, then linearly interpolated to the neural time axis. NaN values from out-of-bounds interpolation are filled with nearest neighbor values.

ii.
```python
ft = np.array(traj_view0['frameTimes'][trix]).flatten()
old_time = ft - vidshift - align_times[trix]
f_interp = interp1d(old_time, me_trial, kind='linear', bounds_error=False, fill_value=np.nan)
me_aligned[:, trix] = f_interp(time_axis)
```

iii. Alignment uses `traj[0]` (side camera) frame times for motion energy, matching the reference code's approach.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple strategies: (1) Missing motion energy files: session still processed with zero motion energy. (2) NaN in DLC tracking: tongue NaN kept (set to 0 velocity), paw NaN filled with nearest neighbor. (3) Sessions with < 2 valid trials or < 10 neurons: skipped entirely. (4) MATLAB v5 vs v7.3 file format: automatic detection and fallback loading. (5) All-zero neural data in late trials: retained with warning. (6) NaN in motion energy after interpolation: filled with nearest neighbor.

ii.
```python
if n_valid < 2:
    print(f"  WARNING: Only {n_valid} valid trials, skipping session")
    return None
if n_neurons_final < 10:
    print(f"  WARNING: Only {n_neurons_final} neurons after filtering, skipping session")
    return None
# Tongue: set NaN to 0
tongue_vel = np.nan_to_num(tongue_vel, nan=0.0)
# Motion energy: fill NaN with nearest
```

iii. The AI documented in CONVERSION_NOTES Step 9 that sessions 36 and 43 had all-zero neural data in late trials, which were retained as warnings.

## 11-a. What are the most time-consuming steps of the code?

i. Based on the conversion output timing, the most time-consuming steps are: (1) Loading .mat files (3-9 seconds per session, ~44% of total time), (2) Spike binning and smoothing (0.5-2.8 seconds per session, ~20% of total time). The full conversion of 44 sessions took ~300 seconds (~7 seconds/session average).

ii.
```python
# Timing from conversion_full_out.txt:
# "Loaded in 6.1s" ... "Spike binning: 1.1s" ... "Behavioral processing: 0.2s"
```

iii. The AI printed timing information and estimated full conversion time based on sample runs.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loops that could be vectorized: (1) The triple-nested loop in spike binning (over neurons, trials, then histogram per neuron-trial pair). This could be vectorized using np.searchsorted or sparse matrix operations. (2) The per-column Gaussian smoothing loop in `causal_gaussian_smooth`. (3) The per-trial interpolation loops for tongue/paw velocity and motion energy.

ii.
```python
# Inner spike binning loop:
for i, clu_idx in enumerate(valid_clu):
    for j in range(ntrials_total):
        spk_mask = trial_arr == trial_num
        counts, _ = np.histogram(aligned, bins=edges)
        fr = counts.astype(np.float32) / DT
        trialdat[:, neuron_offset + i, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)

# Per-column smoothing:
for j in range(x_filt.shape[1]):
    out[:, j] = np.convolve(x_filt[:, j], kern, mode='same')
```

iii. The AI noted in CONVERSION_NOTES Step 6 that it aimed to write efficient code, but the spike binning loop remains unvectorized.

## 11-c. What processing does the code repeat multiple times?

i. (1) Behavioral variables (R, L, hit, miss, etc.) are loaded for ALL trials, then subsetted to valid trials -- the processing of invalid trials is wasted. (2) The video offset `compute_video_offset()` is called once per session, which is correct. (3) The smoothing function is called independently for each neuron-trial pair, recomputing the kernel each time (though the kernel is the same across calls).

ii.
```python
# Spike binning done for ALL trials, then subsetted:
trialdat = np.zeros((n_time, total_neurons, ntrials_total), dtype=np.float32)
...  # process all ntrials_total
trialdat_valid = trialdat[:, :, valid_trials]  # subset

# Smoothing kernel recomputed each call:
def causal_gaussian_smooth(x, N, bctype='reflect'):
    kern = np.array(scipy_windows.gaussian(N, std=N/6.0))
```

iii. The AI processes all trials first for neural data then subsets. This is redundant since only valid trials are used downstream.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) Neural data is binned for ALL trials (including early, stim, no-response), but only valid trials are kept. (2) Per-trial outputs (lick direction, context, outcome) are broadcast to all 500 time points, duplicating the same value 500 times, which increases memory but is required by the output format. (3) The `align_and_bin_spikes` function is defined but never called (process_session reimplements the logic inline). (4) The `remove_low_fr_clusters` function is defined but `remove_low_fr_neurons` is used instead.

ii.
```python
# Unused function:
def align_and_bin_spikes(clu_probe, cluster_indices, align_times, ntrials, ...):
    ...

# Per-trial broadcast:
out[0, :] = lick_direction[t_idx]  # same value repeated 500 times
out[1, :] = context[t_idx]
out[2, :] = outcome[t_idx]
```

iii. The AI's code contains some dead code (unused functions) and processes data for invalid trials that is discarded.
