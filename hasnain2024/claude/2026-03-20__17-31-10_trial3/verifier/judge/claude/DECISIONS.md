# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each session is one MATLAB file `data_structure_<animal>_<date>.mat` in one of two directories (`Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior`). The 44 sessions are enumerated in a hardcoded list `EPHYS_SESSIONS` with directory, animal, date, and probe numbers. Files are loaded via `load_mat_file()` which tries HDF5 first (h5py), falling back to `scipy.io.loadmat`. Motion energy is loaded separately from `motionEnergy_<animal>_<date>.mat` via `load_motion_energy()`.

ii.
```python
EPHYS_SESSIONS = [
    ('data/Ephys_Behavior', 'JEB6', '2021-04-18', [2]),
    ...
    ('data/RandomizedDelay_Ephys_Behavior', 'JEB24', '2023-11-03', [1]),
]

def load_mat_file(filepath):
    try:
        f = h5py.File(filepath, 'r')
        return f, 'h5'
    except Exception:
        mat = scipy.io.loadmat(filepath, squeeze_me=False)
        return mat, 'v5'
```

iii. From CONVERSION_NOTES.md: "Only ephys sessions have neural data: Ephys_Behavior and RandomizedDelay_Ephys_Behavior directories." Session list derived from the authors' DataLoadingScripts loader scripts.

## 1-b. How are the data split into subjects?

i. The animal name is provided as part of the session tuple in `EPHYS_SESSIONS`. Unique subjects are collected after processing and sorted. `subject_idx` maps each session to its subject.

ii.
```python
all_animals.append(animal)
...
unique_subjects = sorted(set(all_animals))
subject_idx = np.array([unique_subjects.index(a) for a in all_animals])
```

iii. From CONVERSION_NOTES.md: "Subjects: 14 (EKH1, EKH3, JEB6, JEB7, JEB11, JEB12, JEB13, JEB14, JEB15, JEB19, JEB23, JEB24, JGR2, JGR3)"

## 1-c. How are the data split into sessions?

i. One session = one entry in `EPHYS_SESSIONS`, corresponding to one `.mat` file. Both fixed-delay and randomized-delay sessions are processed uniformly in a single loop. Each produces one element of `neural`, `input`, `output`. Total: 44 sessions (25 fixed + 19 randomized).

ii.
```python
for sess_idx, (dirpath, animal, date, probes) in enumerate(sessions):
    result = process_session(dirpath, animal, date, probes, ...)
    if result is None:
        continue
    all_neural.append(result['neural'])
```

iii. From CONVERSION_NOTES.md: "Sessions (total): 44 (25 standard + 19 randomized)"

## 1-d. How are the data split into trials?

i. Trials are determined by `Ntrials` from `obj.bp`. Each trial has corresponding behavioral fields (hit, miss, R, L, etc.) and event times. Trial indices are 0-based in the code (but 1-based in the raw data).

ii.
```python
ntrials = get_ntrials(data, fmt)
# ...
for j in range(ntrials):
    trial_num = j + 1  # 1-indexed
    spike_mask = trial == trial_num
```

iii. From CONVERSION_NOTES.md: "obj.bp fields: hit, miss, R, L, autowater, early, no, stim.enable, ev (events)"

## 1-e. How are trials filtered based on quality controls?

i. Three filters: (1) Exclude photostimulation trials (`stim.enable=1`). (2) Exclude early-lick trials (`early=1`). (3) Exclude trials where go cue alignment time is NaN or 0. (4) Exclude trials beyond the last spike trial (recording ended before session).

ii.
```python
valid_trials = ~stim_enable & ~early
valid_trials &= ~np.isnan(align_times) & (align_times > 0)
# ...
max_spike_trial = max(int(np.max(c['trial'])) for c in all_clusters if len(c['trial']) > 0)
valid_trial_indices = valid_trial_indices[valid_trial_indices < max_spike_trial]
```

iii. From CONVERSION_NOTES.md: "Trial filter: exclude stim.enable=1 and early=1 trials" and "Recording may end before session ends."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Spike data from `obj.clu{probe}` clusters. Each cluster has `trialtm` (spike times relative to trial start), `trial` (trial number), and `quality` (curation label). Go cue times from `obj.bp.ev.goCue` are used for alignment.

ii.
```python
trialtm = probe_data['trialtm'][i, 0]
trialtm = f[trialtm_ref][:].flatten()
trial_ref = probe_data['trial'][i, 0]
trial = f[trial_ref][:].flatten().astype(int)
```

iii. From CONVERSION_NOTES.md: "Spike data: obj.clu{probe}(cluster).tm (session time), .trialtm (trial time), .trial (trial number), .quality (string)"

## 2-b. How is the `neural` data processed?

i. Spikes are binned into time bins (10 ms, DT=1/100), converted to firing rates by dividing by DT, then smoothed with a causal Gaussian kernel (N=15, reflect boundary). The causal Gaussian zeros out the first half of the kernel, so only past and current time points contribute.

ii.
```python
DT = 1.0 / 100  # 10 ms time bins
SMOOTH_N = 15
# ...
counts = np.histogram(aligned_times, bins=edges)[0]
rate = counts.astype(np.float64) / DT
smoothed = causal_gaussian_smooth(rate, SMOOTH_N, SMOOTH_BC)
```

```python
def causal_gaussian_smooth(x, N, bctype='reflect'):
    t = np.arange(N)
    mu = (N - 1) / 2
    sigma = N / 6
    kernel = np.exp(-0.5 * ((t - mu) / sigma) ** 2)
    kernel[:int(np.ceil(N / 2))] = 0  # Make causal
    kernel = kernel / kernel.sum()
```

iii. From CONVERSION_NOTES.md: "Smoothing: Causal Gaussian N=15, reflect boundary, as in code" and "dt = 1/100 = 10ms, as in WorkingWithDataObjs.m"

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) Quality label filter: exclude clusters with quality in {'garbage', 'gabrga', 'noisy', 'real?'}. (2) Low firing rate filter: exclude neurons with mean FR <= 1 Hz. Additionally, sessions with fewer than 10 units after filtering are skipped entirely.

ii.
```python
EXCLUDE_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR_THRESHOLD = 1.0
MIN_UNITS = 10
# ...
quality = h5_deref_string(f, q_ref).strip().lower()
if quality in EXCLUDE_QUALITIES:
    continue
# ...
mean_frs = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
keep = mean_frs > LOW_FR_THRESHOLD
```

iii. From CONVERSION_NOTES.md: "Quality filter: keep all except 'garbage', 'noisy', 'real?', 'gabrga'" and "Low FR filter: > 1 Hz mean firing rate"

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to go cue by subtracting the go cue event time from each spike's trial-relative time: `aligned_times = trialtm - align_times[trial]`.

ii.
```python
align_times = get_event_times(data, fmt, ALIGN_EVENT)  # ALIGN_EVENT = 'goCue'
# ...
aligned_times = trialtm[spike_mask] - align_times[j]
```

iii. From CONVERSION_NOTES.md: "Align to goCue event"

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Time bin size is 10 ms (DT = 1/100), yielding 500 bins over the -2.5 to 2.5 s window. No rebinning is applied.

ii.
```python
DT = 1.0 / 100  # 10 ms time bins
TMIN = -2.5
TMAX = 2.5
# ...
edges = np.arange(TMIN, TMAX + DT/2, DT)
time_axis = edges[:-1] + DT / 2
```

iii. From CONVERSION_NOTES.md: "dt = 1/100 = 10ms, as in WorkingWithDataObjs.m and consistent with main analysis" and noted the ambiguity: "WorkingWithDataObjs: 1/100=10ms; getDefaultParams: 1/200=5ms"

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the time axis itself, defined by the bin centers of the [-2.5, 2.5] s window with 10 ms bins.

ii.
```python
edges = np.arange(TMIN, TMAX + DT/2, DT)
time_axis = edges[:-1] + DT / 2
# ...
input_data = time_axis.astype(np.float32).reshape(1, -1)
```

iii. Same time axis for all trials, centered on go cue.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing. The time axis is directly computed from the bin parameters and used as-is. Each trial gets the identical time axis.

ii.
```python
input_data = time_axis.astype(np.float32).reshape(1, -1)
input_trials.append(input_data)
```

iii. N/A

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time axis IS the binning grid used for neural data, so they are inherently aligned. The same edges are used for spike binning and for computing bin centers.

ii.
```python
time_axis, edges = compute_time_axis()
# edges used for spike binning:
counts = np.histogram(aligned_times, bins=edges)[0]
# time_axis used for input:
input_data = time_axis.astype(np.float32).reshape(1, -1)
```

iii. N/A

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Only `obj.bp.R` (and implicitly `obj.bp.L`). The AI uses R directly as the lick direction, treating it as the instruction/stimulus side rather than deriving the actual lick direction.

ii.
```python
R = get_bp_field(data, fmt, 'R').astype(bool)
# ...
lick_direction = R_valid.astype(np.float32)
```

iii. From CONVERSION_NOTES.md Step 5: "Actually, we should keep it simple: R=1, L=0 as the instruction/stimulus direction. The outcome variable captures whether they got it right."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Simple binary encoding: R=1 (right instruction), L=0 (left instruction). This is the instruction side, not the actual lick direction. There are only 2 classes (left=0, right=1), with no "no lick" class for ignore trials.

ii.
```python
lick_direction = R_valid.astype(np.float32)
# ...
out[0, :] = int(lick_direction[t_idx])
```

```python
'output_values': [
    ['left', 'right'],           # lick_direction: 0=left, 1=right
    ...
]
```

iii. The AI considered deriving actual lick direction from hit/miss but decided to use instruction side for simplicity: "Let me just use R=1, L=0 as the trial type variable."

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `obj.bp.autowater`. When autowater is true, the trial is WC (water-cued) context; otherwise DR (delayed-response).

ii.
```python
autowater = get_bp_field(data, fmt, 'autowater').astype(bool)
# ...
behavioral_context = (~autowater_valid).astype(np.float32)
```

iii. From CONVERSION_NOTES.md: "WC(aw=1)=0, DR(aw=0)=1"

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct mapping: autowater=True -> WC=0, autowater=False -> DR=1.

ii.
```python
behavioral_context = (~autowater_valid).astype(np.float32)
# ...
out[1, :] = int(behavioral_context[t_idx])
```

iii. Matches the instruction specification: WC=0, DR=1.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Only `obj.bp.hit`. The AI uses hit as a binary outcome indicator.

ii.
```python
hit = get_bp_field(data, fmt, 'hit').astype(bool)
# ...
outcome = hit_valid.astype(np.float32)
```

iii. From CONVERSION_NOTES.md: "correct (hit=1) -> 1, incorrect (miss=1 or no=1) -> 0"

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Binary encoding: hit=1 -> correct=1, everything else (miss and ignore) -> incorrect=0. There are only 2 classes, with no separate "ignore" class.

ii.
```python
outcome = hit_valid.astype(np.float32)
out[2, :] = int(outcome[t_idx])
```

```python
'output_values': [
    ...
    ['incorrect', 'correct'],    # outcome: 0=incorrect, 1=correct
    ...
]
```

iii. From CONVERSION_NOTES.md Step 5: "Outcome: correct (hit=1) -> 1, incorrect (miss=1 or no=1) -> 0"

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. DLC tracking data from `obj.traj`, specifically the 'tongue' feature from the side camera (view=0). Only one camera view is used for the tongue.

ii.
```python
tongue_speed = extract_velocity_from_traj(
    data, fmt, view=0, feat_name='tongue',
    ntrials=ntrials, align_times=align_times,
    time_axis=time_axis, vidshift=vidshift
)
```

iii. From CONVERSION_NOTES.md Step 5: "Tongue velocity: Compute from jaw feature on side cam (y-velocity), or from tongue if available... the task says 'tongue velocity' - use tongue velocity from side cam."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For tongue: no position smoothing is applied. Velocity is computed as `np.gradient(x)` and `np.gradient(y)` (no time step used). NaN velocities are set to 0. Speed is `sqrt(xvel^2 + yvel^2)`. Remaining NaNs filled with nearest neighbor. Speed is then interpolated to the neural time axis via `np.interp`.

ii.
```python
# For tongue: no smoothing
xpos_smooth = xpos.copy()
ypos_smooth = ypos.copy()

xvel = np.gradient(xpos_smooth)
yvel = np.gradient(ypos_smooth)

# For tongue: set NaN velocities to 0
xvel[np.isnan(xvel)] = 0
yvel[np.isnan(yvel)] = 0

spd = np.sqrt(xvel**2 + yvel**2)

# Fill missing with nearest
# ...
# Interpolate to time axis
interpolated = np.interp(time_axis, aligned_ft, spd)
```

iii. From CONVERSION_NOTES.md: "DLC velocity extraction matching findVelocity.m" and "When tongue not visible, velocity = 0"

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Discretized into 2 classes using per-session 50th percentile threshold: below threshold -> 0, at or above threshold -> 1. Special handling for near-zero thresholds (threshold < 1e-10 is replaced with 1e-10).

ii.
```python
tongue_thresh = np.nanpercentile(tongue_speed_valid, 50)
tongue_disc = discretize_time_series(tongue_speed_valid, tongue_thresh)

def discretize_time_series(values, threshold):
    if threshold < 1e-10:
        threshold = 1e-10
    return (values >= threshold).astype(np.float32)
```

iii. From CONVERSION_NOTES.md: "Per-session 50th percentile threshold. Below median -> 0, >= median -> 1."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are aligned by subtracting the video offset and go cue time. The resulting speed values are interpolated (`np.interp`) onto the neural time axis.

ii.
```python
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
# ...
interpolated = np.interp(time_axis, aligned_ft, spd)
```

iii. From CONVERSION_NOTES.md: "DLC kinematics: interpolated from video time to neural time axis, video offset subtracted"

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. DLC tracking from `obj.traj`, the 'top_paw' feature from the bottom camera (view=1).

ii.
```python
paw_speed = extract_velocity_from_traj(
    data, fmt, view=1, feat_name='top_paw',
    ntrials=ntrials, align_times=align_times,
    time_axis=time_axis, vidshift=vidshift
)
```

iii. From CONVERSION_NOTES.md: "Paw velocity: Use top_paw from bottom cam."

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For non-tongue features (including paw), position is smoothed with causal Gaussian (N=21, reflect). Velocity is computed via `np.gradient`. Baseline velocity (median) is subtracted. NaN values are filled with nearest neighbor. Speed is interpolated to neural time axis.

ii.
```python
xpos_smooth = causal_gaussian_smooth(xpos, 21, 'reflect')
ypos_smooth = causal_gaussian_smooth(ypos, 21, 'reflect')
xvel = np.gradient(xpos_smooth)
yvel = np.gradient(ypos_smooth)
xvel = xvel - np.nanmedian(xvel)
yvel = yvel - np.nanmedian(yvel)
spd = np.sqrt(xvel**2 + yvel**2)
```

iii. From CONVERSION_NOTES.md: "Velocity: computed as gradient of position"

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue velocity: discretized into 2 classes using per-session 50th percentile. Below threshold -> 0, at or above -> 1.

ii.
```python
paw_thresh = np.nanpercentile(paw_speed_valid, 50)
paw_disc = discretize_time_series(paw_speed_valid, paw_thresh)
```

iii. Same discretization method as all continuous outputs.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: frame times corrected by video offset and go cue, then interpolated to neural time axis.

ii.
```python
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
interpolated = np.interp(time_axis, aligned_ft, spd)
```

iii. Same alignment procedure as tongue velocity.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Loaded from separate `motionEnergy_<animal>_<date>.mat` files. Handles multiple nested formats (struct with 'data' field, direct cell array, doubly-nested struct).

ii.
```python
def load_motion_energy(dirpath, animal, date):
    me_file = os.path.join(dirpath, f'motionEnergy_{animal}_{date}.mat')
    me_mat = scipy.io.loadmat(me_file, squeeze_me=False)
    me_raw = me_mat['me']
    # Handles Format 1 (struct), Format 2 (cell array), Format 3 (nested struct)
```

iii. From CONVERSION_NOTES.md: "Motion energy: loaded from separate files, interpolated to neural time axis"

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. No processing beyond interpolation. Per-trial motion energy traces are interpolated to the neural time axis using `np.interp`. NaN values are filled with nearest neighbor or zeros.

ii.
```python
me_interp[:, trial_idx] = np.interp(
    time_axis, aligned_ft[:n_frames], me_trial[:n_frames]
).astype(np.float32)
```

iii. From CONVERSION_NOTES.md: "Motion energy from separate files at 400Hz original rate"

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same as other continuous outputs: per-session 50th percentile threshold, 2 classes (low=0, high=1).

ii.
```python
me_thresh_50 = np.nanpercentile(me_valid, 50)
me_disc = discretize_time_series(me_valid, me_thresh_50)
```

iii. Same discretization as tongue and paw velocity.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Frame times from the side camera (view=0) are used. Video offset is subtracted and go cue time subtracted. Motion energy is interpolated to neural time axis.

ii.
```python
ts, frame_times, _ = get_traj_data(data, fmt, 0, trial_idx)  # side cam
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
me_interp[:, trial_idx] = np.interp(
    time_axis, aligned_ft[:n_frames], me_trial[:n_frames]
).astype(np.float32)
```

iii. Uses side camera frame times since motion energy corresponds to side camera frames.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) NaN frame times: synthetic frame times created at 400 Hz with 0.5s offset. (2) NaN velocities for tongue: set to 0. (3) NaN speed values: filled with nearest neighbor. (4) Fully NaN trial columns: filled with zeros. (5) Motion energy NaN: filled with nearest or zeros. (6) Trials beyond recording: excluded by detecting max spike trial.

ii.
```python
if len(frame_times) == 0 or np.all(np.isnan(frame_times)):
    frame_times = np.arange(n_frames) / 400.0
    ft_offset = 0.5
# ...
nan_cols = np.all(np.isnan(speed), axis=0)
speed[:, nan_cols] = 0.0
```

iii. From CONVERSION_NOTES.md Step 10: "Recording likely ended before these trials" and filling strategies are implicit in the code.

## 11-a. What are the most time-consuming steps of the code?

i. DLC velocity extraction (tongue and paw) and spike binning are the most time-consuming, each taking several seconds per session. The total processing time is ~6s per session, ~270s total.

ii.
```python
t1 = time.time()
trialdat = bin_and_smooth_spikes(...)
print(f"    Spike binning: {time.time()-t1:.1f}s")
# ...
tongue_speed = extract_velocity_from_traj(...)
print(f"    Tongue velocity: {time.time()-t2:.1f}s")
```

iii. From CONVERSION_NOTES.md: "Total: 269 seconds (~6.1s/session average)"

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loops over both neurons and trials individually (`for j in range(ntrials)` inside `for i, clu in enumerate(clusters)`), using per-trial `np.histogram`. This could be vectorized with `np.histogram2d` over all trials at once per cluster. The velocity extraction also loops over trials.

ii.
```python
for i, clu in enumerate(clusters):
    for j in range(ntrials):
        spike_mask = trial == trial_num
        counts = np.histogram(aligned_times, bins=edges)[0]
```

iii. From CONVERSION_NOTES.md: "Spike binning loops over neurons and trials (vectorized with np.histogram)" — though the code actually uses per-trial histogram, not a fully vectorized approach.

## 11-c. What processing does the code repeat multiple times?

i. The `get_traj_data` function is called separately for tongue velocity extraction and motion energy interpolation, reading the same camera data twice for the same trials. Feature names are re-extracted each trial call.

ii.
```python
# Called for tongue:
ts, frame_times, feat_names = get_traj_data(data, fmt, view=0, trial_idx)
# Called again for motion energy:
ts, frame_times, _ = get_traj_data(data, fmt, 0, trial_idx)
```

iii. Not explicitly documented in CONVERSION_NOTES.md.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads the full MATLAB file including all fields (sglx, ex, etc.) that are not used. The HDF5 files are kept open during processing, materializing data on access. For velocity, baseline subtraction (median subtraction) is applied to paw but not tongue, and NaN filling with nearest neighbor is applied before interpolation, which then smooths the signal anyway.

ii.
```python
f = h5py.File(filepath, 'r')
return f, 'h5'
# Full file handle kept open
```

iii. Not explicitly documented in CONVERSION_NOTES.md.
