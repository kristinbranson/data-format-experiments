# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from .mat files (both HDF5 v7.3 and MATLAB v5 formats) from two directories: `data/Ephys_Behavior/` (25 sessions) and `data/RandomizedDelay_Ephys_Behavior/` (19 sessions, excluding 3 not in loader scripts). Each file is named `data_structure_<animal>_<date>.mat`. A hardcoded list `EPHYS_SESSIONS` enumerates all 44 sessions with directory, animal name, date, and probe numbers. Motion energy is loaded from separate `motionEnergy_<animal>_<date>.mat` files. Behavior-only directories (`DelayInhibition_BilatMC_Behavior`, `GoCueInhibition_BilatMC_Behavior`) are excluded (no neural data).

ii.
```python
EPHYS_SESSIONS = [
    ('data/Ephys_Behavior', 'JEB6', '2021-04-18', [2]),
    ('data/Ephys_Behavior', 'JEB7', '2021-04-29', [1]),
    # ... 44 total entries
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

iii. The AI identified from the reference loader scripts (e.g., `loadSessionData.m`, `loadANM_ALMVideo()`) which sessions and probe numbers to include, excluding 3 randomized delay sessions not in the loader scripts (JEB23_2023-10-20, JEB24_2023-10-03, JEB24_2023-10-04) and all behavior-only sessions.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the animal name field in `EPHYS_SESSIONS`. Unique subjects are collected from all processed sessions and sorted alphabetically. A `subject_idx` array maps each session to its subject index. 14 unique subjects are identified.

ii.
```python
unique_subjects = sorted(set(all_animals))
subject_idx = np.array([unique_subjects.index(a) for a in all_animals])
```

iii. The AI noted that the paper says "nine mice" for 25 fixed-delay sessions, but data shows 10 unique animals. The AI decided to include all sessions from the loader scripts rather than trying to match the paper's count exactly.

## 1-c. How are the data split into sessions?

i. Each entry in `EPHYS_SESSIONS` is one session, identified by animal+date combination. The data file is `data_structure_<animal>_<date>.mat`. For multi-probe sessions (e.g., JEB15 with probes [1,2]), neurons from all probes are concatenated into a single session.

ii.
```python
filepath = os.path.join(dirpath, f"data_structure_{session_id}.mat")
# ...
for p in probe_nums:
    clusters = get_clusters(data, fmt, p - 1)  # 0-indexed
    all_clusters.extend(clusters)
```

iii. Session definitions follow the reference loader scripts. Sessions with fewer than 10 neurons after filtering or fewer than 2 valid trials are skipped.

## 1-d. How are the data split into trials?

i. Trial count is read from `obj.bp.Ntrials`. Each cluster (neuron) has spike times tagged with trial numbers (`clu.trial`). Behavioral variables (`hit`, `miss`, `R`, `L`, `autowater`, `early`, etc.) are per-trial arrays indexed 0 to Ntrials-1. Go cue event times (`obj.bp.ev.goCue`) provide alignment for each trial.

ii.
```python
def get_ntrials(data, fmt):
    if fmt == 'h5':
        return int(data['obj']['bp']['Ntrials'][0, 0])
    else:
        return int(data['obj']['bp'][0, 0]['Ntrials'][0, 0])
```

iii. The AI followed the reference data structure where trials are already defined by the behavioral protocol.

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded if: (1) `stim.enable == 1` (optogenetic stimulation trials), (2) `early == 1` (early lick trials), (3) go cue time is NaN or 0, (4) trial index exceeds the maximum trial number in spike data (recording ended before session). No minimum trial count filtering per session is applied (only minimum neuron count).

ii.
```python
valid_trials = ~stim_enable & ~early
valid_trials &= ~np.isnan(align_times) & (align_times > 0)
# ...
if all_clusters:
    max_spike_trial = max(int(np.max(c['trial'])) for c in all_clusters if len(c['trial']) > 0)
    beyond = np.sum(valid_trial_indices >= max_spike_trial)
    if beyond > 0:
        valid_trial_indices = valid_trial_indices[valid_trial_indices < max_spike_trial]
```

iii. The AI based filtering on the reference code's `findTrials.m` which excludes stim and early trials. The recording-extent check was added after discovering sessions where spike recording ended before the behavioral session (sessions 36 and 43).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `obj.clu{probe}(cluster).trialtm` (spike times relative to trial start) and `obj.clu{probe}(cluster).trial` (trial number for each spike). The quality label `obj.clu{probe}(cluster).quality` is used for filtering.

ii.
```python
trialtm = f[trialtm_ref][:].flatten()
trial = f[trial_ref][:].flatten().astype(int)
quality = h5_deref_string(f, q_ref).strip().lower()
```

iii. This directly follows the reference code's `findClusters.m` and `alignSpikes.m` which access these fields.

## 2-b. How is the `neural` data processed?

i. Processing pipeline: (1) Quality filter clusters, (2) Align spike times to go cue event, (3) Bin spikes into 10ms time bins over [-2.5, 2.5]s, (4) Convert counts to firing rates (divide by dt), (5) Apply causal Gaussian smoothing (N=15, reflect boundary), (6) Remove neurons with mean FR <= 1 Hz.

ii.
```python
# Bin spikes
counts = np.histogram(aligned_times, bins=edges)[0]
rate = counts.astype(np.float64) / DT
smoothed = causal_gaussian_smooth(rate, SMOOTH_N, SMOOTH_BC)

# Causal Gaussian: zero out first half of kernel
kernel[:int(np.ceil(N / 2))] = 0
kernel = kernel / kernel.sum()
```

iii. The AI documented matching `getSeq.m` for binning and `mySmooth.m` for smoothing. Parameters (dt=1/100, N=15, tmin=-2.5, tmax=2.5) were taken from `WorkingWithDataObjs.m`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two levels of filtering: (1) Cluster quality filter: exclude clusters with quality labels in `{'garbage', 'gabrga', 'noisy', 'real?'}`, (2) Low firing rate filter: remove neurons with mean FR <= 1 Hz (mean across all time bins and all trials). Sessions with fewer than 10 neurons after filtering are skipped.

ii.
```python
EXCLUDE_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR_THRESHOLD = 1.0
MIN_UNITS = 10

# Quality filter in get_clusters:
if quality in EXCLUDE_QUALITIES:
    continue

# Low FR filter:
mean_frs = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
keep = mean_frs > LOW_FR_THRESHOLD
```

iii. Quality exclusion labels match `findClusters.m`. Low FR threshold of 1 Hz matches the paper statement. The AI noted a discrepancy between `WorkingWithDataObjs.m` (1 Hz) and `getDefaultParams.m` (0.5 Hz) and chose 1 Hz per the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to go cue onset by subtracting the go cue event time from each spike's trial-relative time: `aligned_times = trialtm - align_times[trial]`. The time axis spans -2.5 to 2.5 seconds from go cue.

ii.
```python
ALIGN_EVENT = 'goCue'
align_times = get_event_times(data, fmt, ALIGN_EVENT)
# ...
aligned_times = trialtm[spike_mask] - align_times[j]
counts = np.histogram(aligned_times, bins=edges)[0]
```

iii. Follows the reference code's `alignSpikes.m` which performs `trialtm_aligned = trialtm - event_time`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Time bin size is 10 ms (dt = 1/100 seconds). No temporal rebinning is applied beyond the initial binning of raw spike times. The time axis has 500 bins spanning [-2.495, 2.495] seconds (bin centers).

ii.
```python
DT = 1.0 / 100  # 10 ms time bins
TMIN = -2.5
TMAX = 2.5
edges = np.arange(TMIN, TMAX + DT/2, DT)
time_axis = edges[:-1] + DT / 2
```

iii. The AI chose 10ms over 5ms, noting `WorkingWithDataObjs.m` uses dt=1/100 while `getDefaultParams.m` uses dt=1/200. The AI chose 10ms as matching the main analysis tutorial.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not derived from any raw data variable. It is the pre-computed time axis: the center of each time bin relative to go cue onset. It is identical for all trials.

ii.
```python
input_data = time_axis.astype(np.float32).reshape(1, -1)
input_trials.append(input_data)
```

iii. This is a synthetic variable representing elapsed time from the alignment event, as specified in the instructions.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is computed as bin centers: `edges + dt/2` where edges span from TMIN to TMAX in steps of dt. The last edge is dropped. The result is a 1D array of 500 values from -2.495 to 2.495 seconds.

ii.
```python
def compute_time_axis():
    edges = np.arange(TMIN, TMAX + DT/2, DT)
    time_axis = edges[:-1] + DT / 2
    return time_axis, edges
```

iii. Matches `getSeq.m` bin center computation.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis is the same time axis used to bin neural spikes, so they are inherently aligned. Both use the same edges and bin centers relative to go cue onset.

ii.
```python
# Same time_axis used for both:
trialdat = bin_and_smooth_spikes(clusters, ntrials, align_times, time_axis, edges)
# ...
input_data = time_axis.astype(np.float32).reshape(1, -1)
```

iii. No explicit alignment step is needed since both share the same temporal grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `obj.bp.R` (right trial indicator). R=1 maps to lick_direction=1 (right), R=0 (implying L=1) maps to lick_direction=0 (left).

ii.
```python
R = get_bp_field(data, fmt, 'R').astype(bool)
L = get_bp_field(data, fmt, 'L').astype(bool)
lick_direction = R_valid.astype(np.float32)
```

iii. The AI considered whether to use actual lick direction (which differs on error trials) versus instruction/stimulus direction, and decided to use R/L as the trial type variable (instruction direction), noting that the outcome variable captures correctness.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Simple binary conversion: R=1 -> 1 (right), R=0 -> 0 (left). The value is per-trial (constant across all time bins). Output is stored as int64 and broadcast to all time bins.

ii.
```python
lick_direction = R_valid.astype(np.float32)
out[0, :] = int(lick_direction[t_idx])
```

iii. The AI noted this represents the instructed/correct direction, not necessarily the actual lick direction on error trials.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj.bp.autowater`. Autowater=1 indicates Water-Cued (WC) context, autowater=0 indicates Delayed-Response (DR) context.

ii.
```python
autowater = get_bp_field(data, fmt, 'autowater').astype(bool)
behavioral_context = (~autowater_valid).astype(np.float32)
```

iii. The AI identified that `autowater` distinguishes WC (autowater=1) from DR (autowater=0) contexts, mapping to WC=0, DR=1 as specified in the instructions.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Logical negation of autowater: `~autowater` gives DR=1, WC=0. The value is per-trial, broadcast to all time bins.

ii.
```python
behavioral_context = (~autowater_valid).astype(np.float32)
out[1, :] = int(behavioral_context[t_idx])
```

iii. Direct mapping consistent with instructions (WC=0, DR=1).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit`. Hit=1 indicates correct outcome, hit=0 (miss or no-response) indicates incorrect outcome.

ii.
```python
hit = get_bp_field(data, fmt, 'hit').astype(bool)
outcome = hit_valid.astype(np.float32)
```

iii. The AI chose to use `hit` directly rather than combining `miss` and `no` (ignore) explicitly, since `hit=0` captures both miss and no-response trials.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Simple binary conversion: hit=1 -> correct=1, hit=0 -> incorrect=0. Per-trial, broadcast to all time bins.

ii.
```python
outcome = hit_valid.astype(np.float32)
out[2, :] = int(outcome[t_idx])
```

iii. Consistent with instructions (incorrect=0, correct=1).

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from DLC tracking data in `obj.traj{1}` (side camera, view=0), specifically the 'tongue' feature. Position data (x, y coordinates) and frame times are extracted per trial.

ii.
```python
tongue_speed = extract_velocity_from_traj(
    data, fmt, view=0, feat_name='tongue',
    ntrials=ntrials, align_times=align_times,
    time_axis=time_axis, vidshift=vidshift
)
```

iii. The AI debated between using 'jaw' and 'tongue' features, ultimately choosing 'tongue' from the side camera as specified in the task instructions.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Processing: (1) Extract x, y positions for 'tongue' feature, (2) For tongue: no position smoothing, (3) Compute velocity using `np.gradient`, (4) Set NaN velocities to 0 (tongue not visible), (5) Compute speed as sqrt(xvel^2 + yvel^2), (6) Interpolate from video frame times to neural time axis using `np.interp`.

ii.
```python
if is_tongue:
    xpos_smooth = xpos.copy()
    ypos_smooth = ypos.copy()
xvel = np.gradient(xpos_smooth)
yvel = np.gradient(ypos_smooth)
if is_tongue:
    xvel[np.isnan(xvel)] = 0
    yvel[np.isnan(yvel)] = 0
spd = np.sqrt(xvel**2 + yvel**2)
interpolated = np.interp(time_axis, aligned_ft, spd)
```

iii. Processing follows `findVelocity.m` from the reference code, with tongue NaN handling as a special case.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per-session 50th percentile threshold across all valid timepoints and trials. Values >= threshold -> 1 (high), < threshold -> 0 (low). When threshold is near 0 (degenerate case where tongue is invisible ~90% of the time), a small epsilon (1e-10) is used so that 0-velocity maps to class 0.

ii.
```python
tongue_thresh = np.nanpercentile(tongue_speed_valid, 50)
tongue_disc = discretize_time_series(tongue_speed_valid, tongue_thresh)

def discretize_time_series(values, threshold):
    if threshold < 1e-10:
        threshold = 1e-10
    return (values >= threshold).astype(np.float32)
```

iii. The AI discovered the degenerate case during sample validation and added the epsilon threshold to separate zero (not visible) from positive (visible and moving) tongue velocity. This results in ~91% low / 9% high distribution.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Video frame times are aligned to go cue by subtracting video offset and go cue time: `aligned_ft = frame_times - vidshift - align_times[trial]`. Then tongue speed is interpolated from video time to the neural time axis using `np.interp`.

ii.
```python
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
interpolated = np.interp(time_axis, aligned_ft, spd)
```

iii. Video offset is computed from bitcode synchronization (`findVideoOffset.m`) or defaults to 0.5 seconds.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from DLC tracking data in `obj.traj{2}` (bottom camera, view=1), specifically the 'top_paw' feature.

ii.
```python
paw_speed = extract_velocity_from_traj(
    data, fmt, view=1, feat_name='top_paw',
    ntrials=ntrials, align_times=align_times,
    time_axis=time_axis, vidshift=vidshift
)
```

iii. The AI identified 'top_paw' from the bottom camera feature list as the appropriate paw tracking feature.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Processing: (1) Extract x, y positions for 'top_paw', (2) Smooth positions with causal Gaussian (N=21, reflect), (3) Compute velocity using `np.gradient`, (4) Subtract baseline velocity (median), (5) Compute speed as sqrt(xvel^2 + yvel^2), (6) Interpolate to neural time axis.

ii.
```python
if not is_tongue:
    xpos_smooth = causal_gaussian_smooth(xpos, 21, 'reflect')
    ypos_smooth = causal_gaussian_smooth(ypos, 21, 'reflect')
xvel = np.gradient(xpos_smooth)
yvel = np.gradient(ypos_smooth)
if not is_tongue:
    xvel = xvel - np.nanmedian(xvel)
    yvel = yvel - np.nanmedian(yvel)
spd = np.sqrt(xvel**2 + yvel**2)
```

iii. Non-tongue features get position smoothing and baseline subtraction, following `findVelocity.m` / `getKinematicsFromVideo.m`.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Per-session 50th percentile threshold across all valid timepoints and trials. Values >= threshold -> 1, < threshold -> 0. Results in approximately 50/50 split by design.

ii.
```python
paw_thresh = np.nanpercentile(paw_speed_valid, 50)
paw_disc = discretize_time_series(paw_speed_valid, paw_thresh)
```

iii. Follows the instruction specification for 50th percentile per-session threshold.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue velocity: video frame times aligned via video offset and go cue subtraction, then interpolated to neural time axis.

ii.
```python
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
interpolated = np.interp(time_axis, aligned_ft, spd)
```

iii. Same video synchronization pipeline as tongue velocity.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from separate files `motionEnergy_<animal>_<date>.mat`. The data is in `me.data` (or nested `me.data.data` for some sessions), containing per-trial time series at ~400 Hz.

ii.
```python
def load_motion_energy(dirpath, animal, date):
    me_file = os.path.join(dirpath, f'motionEnergy_{animal}_{date}.mat')
    me_mat = scipy.io.loadmat(me_file, squeeze_me=False)
    me_raw = me_mat['me']
    # Handle multiple formats (struct with data field, direct cell array, nested struct)
```

iii. The AI handled multiple .mat file formats for motion energy data, including a nested structure case discovered during full conversion.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Per-trial motion energy is extracted from the .mat file. Frame times from the side camera (view=0) are used for temporal alignment. Motion energy is interpolated from video frame times to the neural time axis using `np.interp`. Missing trials are filled with zeros.

ii.
```python
def interpolate_motion_energy(me_data, ntrials, data, fmt, align_times, time_axis, vidshift):
    aligned_ft = frame_times - ft_offset - align_times[trial_idx]
    n_frames = min(len(me_trial), len(aligned_ft))
    me_interp[:, trial_idx] = np.interp(
        time_axis, aligned_ft[:n_frames], me_trial[:n_frames]
    ).astype(np.float32)
```

iii. Follows `loadMotionEnergy.m` which interpolates ME to neural time axis using video frame times for alignment.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session 50th percentile threshold. Values >= threshold -> 1, < threshold -> 0. Results in approximately 50/50 split.

ii.
```python
me_thresh_50 = np.nanpercentile(me_valid, 50)
me_disc = discretize_time_series(me_valid, me_thresh_50)
```

iii. Follows instruction specification.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned using the same video frame times as trajectory data (side camera). Frame times are adjusted by video offset and go cue time, then ME is interpolated to neural time axis.

ii.
```python
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
me_interp[:, trial_idx] = np.interp(time_axis, aligned_ft[:n_frames], me_trial[:n_frames])
```

iii. Follows the reference code's `loadMotionEnergy.m` temporal alignment approach.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) Trials beyond spike recording extent are excluded (sessions 36 and 43), (2) NaN frame times trigger synthetic frame times at 400 Hz, (3) NaN tongue velocities set to 0, (4) Missing motion energy trials filled with zeros, (5) NaN values in speed/ME arrays filled with nearest valid value, (6) Entirely NaN columns filled with zeros, (7) Failed DLC extraction falls through silently (speed stays 0).

ii.
```python
# Recording extent check
if beyond > 0:
    valid_trial_indices = valid_trial_indices[valid_trial_indices < max_spike_trial]

# NaN frame times
if len(frame_times) == 0 or np.all(np.isnan(frame_times)):
    frame_times = np.arange(n_frames) / 400.0

# NaN tongue velocity
if is_tongue:
    xvel[np.isnan(xvel)] = 0
    yvel[np.isnan(yvel)] = 0

# Missing ME
if me_data is None:
    me_valid = np.zeros(...)
```

iii. The AI discovered and documented these edge cases during conversion, adding handling iteratively.

## 11-a. What are the most time-consuming steps of the code?

i. Per the timing output, the most time-consuming steps per session are: (1) Paw velocity extraction (~2-5s), (2) Tongue velocity extraction (~1-2s), (3) Motion energy loading/interpolation (~1-2s), (4) Spike binning (~0.5-2.5s). Total ~6s/session, ~270s for all 44 sessions.

ii.
```python
# Timing printed per session:
print(f"    Spike binning: {time.time()-t1:.1f}s")
print(f"    Tongue velocity: {time.time()-t2:.1f}s")
print(f"    Paw velocity: {time.time()-t3:.1f}s")
print(f"    Motion energy: {time.time()-t4:.1f}s")
```

iii. The AI estimated ~5-7 minutes total and noted the actual time was ~4.5 minutes.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning has nested loops (over neurons and trials) that could be partially vectorized. The DLC velocity extraction loops over trials individually. The NaN filling loops (nearest neighbor replacement) iterate per-element.

ii.
```python
# Spike binning: nested loop over neurons and trials
for i, clu in enumerate(clusters):
    for j in range(ntrials):
        counts = np.histogram(aligned_times, bins=edges)[0]

# NaN filling: element-wise loop
for idx in np.where(nan_mask)[0]:
    nearest = valid[np.argmin(np.abs(valid - idx))]
    spd[idx] = spd[nearest]
```

iii. The AI noted the spike binning uses `np.histogram` (vectorized within each trial) but the outer loops remain. DLC extraction must loop over trials due to variable frame times.

## 11-c. What processing does the code repeat multiple times?

i. The code loads trajectory data (`get_traj_data`) separately for tongue velocity, paw velocity, and motion energy alignment. For motion energy, it re-loads side camera data just to get frame times. Feature names are re-parsed for every trial. The video offset is computed once per session but DLC data is accessed multiple times.

ii.
```python
# Tongue: loads traj data for all trials (view=0)
tongue_speed = extract_velocity_from_traj(data, fmt, view=0, ...)
# Paw: loads traj data for all trials (view=1)
paw_speed = extract_velocity_from_traj(data, fmt, view=1, ...)
# ME: loads traj data again (view=0) just for frame times
ts, frame_times, _ = get_traj_data(data, fmt, 0, trial_idx)
```

iii. Frame times from the side camera are loaded 3 times (tongue, motion energy interpolation, and indirectly through the separate velocity extraction calls).

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) Spike binning and smoothing is done for ALL trials including invalid ones (stim/early), then only valid trials are selected. (2) The `L` variable is loaded but never used (only `R` is used for lick direction). (3) The `miss` variable is loaded but not used for outcome (only `hit`). (4) The `no` variable from the behavior protocol is not loaded at all. (5) Motion energy threshold from the ME file (`me_thresh`) is loaded but not used (replaced by 50th percentile computation).

ii.
```python
# All-trial processing before filtering to valid:
trialdat = bin_and_smooth_spikes(all_clusters, ntrials, align_times, time_axis, edges)
# Then extract only valid:
trialdat_valid = trialdat[:, :, valid_trial_indices]

# L loaded but unused:
L = get_bp_field(data, fmt, 'L').astype(bool)
# miss loaded but unused:
miss = get_bp_field(data, fmt, 'miss').astype(bool)
```

iii. Processing all trials before filtering wastes computation on ~20% of trials that are later excluded.
