# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI defines all 44 sessions (25 from `Ephys_Behavior/` and 19 from `RandomizedDelay_Ephys_Behavior/`) in a hardcoded list `EPHYS_SESSIONS`. Each entry specifies the directory, animal ID, date, and probe number(s). The two behavior-only directories (`DelayInhibition_BilatMC_Behavior/`, `GoCueInhibition_BilatMC_Behavior/`) are excluded as they have no neural data. Each session is loaded individually from a `.mat` file via `load_mat_file()`, which handles both HDF5 (v7.3) and MATLAB v5 formats. The 3 excluded randomized-delay sessions (JEB23_2023-10-20, JEB24_2023-10-03, JEB24_2023-10-04) match the reference loader scripts.

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

iii. The AI documented the session list based on reference loader scripts in `code/DataLoadingScripts/Recording and video/`. The CONVERSION_NOTES explain that only ephys directories are used since the other directories lack neural data.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the animal ID string in each session definition. Unique subjects are collected and sorted alphabetically. A `subject_idx` array maps each session to its subject. The AI identifies 14 unique subjects across the 44 sessions (10 from fixed delay, 4 from randomized delay).

ii.
```python
unique_subjects = sorted(set(all_animals))
subject_idx = np.array([unique_subjects.index(a) for a in all_animals])
```

iii. The CONVERSION_NOTES discuss a discrepancy: the paper reports 9 mice for the 25 fixed-delay sessions, but the data files show 10 unique animals. The AI chose to include all animals found in the loader scripts.

## 1-c. How are the data split into sessions?

i. Each entry in `EPHYS_SESSIONS` represents one session, identified by a unique (animal, date) pair. Sessions are processed sequentially, and each session produces a separate entry in the output lists. Sessions with fewer than `MIN_UNITS=10` neurons after quality and low-FR filtering are skipped.

ii.
```python
for sess_idx, (dirpath, animal, date, probes) in enumerate(sessions):
    result = process_session(dirpath, animal, date, probes, time_axis, edges, ...)
    if result is None:
        continue
    all_neural.append(result['neural'])
```

iii. The session list and MIN_UNITS threshold are justified by the reference loader scripts and the paper statement: "Recording sessions were included for analysis only if they had at least 10 units."

## 1-d. How are the data split into trials?

i. The total number of trials per session is read from `obj.bp.Ntrials`. Each trial is identified by a 1-based index. Spike data, behavioral variables, and video data are all indexed by trial number. After filtering, only valid trial indices are used to extract data.

ii.
```python
ntrials = get_ntrials(data, fmt)
valid_trial_indices = np.where(valid_trials)[0]
trialdat_valid = trialdat[:, :, valid_trial_indices]
```

iii. The AI follows the same trial indexing as the reference code where trials are numbered 1 to Ntrials. This matches the MATLAB convention in `getSeq.m` which iterates `j = 1:obj.bp.Ntrials`.

## 1-e. How are trials filtered based on quality controls?

i. The AI excludes trials where: (1) `stim.enable = 1` (optogenetic stimulation), (2) `early = 1` (early lick), (3) goCue alignment time is NaN or 0, and (4) the trial number exceeds the maximum trial in the spike data (recording ended before session). Importantly, the AI does NOT exclude ignore/no-response trials.

ii.
```python
valid_trials = ~stim_enable & ~early
valid_trials &= ~np.isnan(align_times) & (align_times > 0)

# Also check recording extent
max_spike_trial = max(int(np.max(c['trial'])) for c in all_clusters if len(c['trial']) > 0)
beyond = np.sum(valid_trial_indices >= max_spike_trial)
if beyond > 0:
    valid_trial_indices = valid_trial_indices[valid_trial_indices < max_spike_trial]
```

iii. The CONVERSION_NOTES document excluding stim and early trials as matching the reference conditions `~stim.enable&~early`. The recording-extent check was added during Step 10 to handle sessions where recording ended before the behavioral session. The AI does not discuss exclusion of ignore trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `obj.clu{probe}` spike cluster data. Specifically, the `trialtm` (spike times within each trial) and `trial` (trial assignment for each spike) fields are used, along with `quality` for filtering.

ii.
```python
clusters.append({
    'trialtm': trialtm,
    'trial': trial,
    'quality': quality,
})
```

iii. This matches the reference code's `alignSpikes.m` and `getSeq.m` which use `obj.clu{prbnum}(curClu).trialtm_aligned` and `.trial`.

## 2-b. How is the `neural` data processed?

i. Processing follows this pipeline: (1) Filter clusters by quality, (2) Align spike times to goCue by subtracting the goCue event time, (3) Bin spikes into 10ms bins using `np.histogram`, (4) Convert counts to firing rate by dividing by dt, (5) Smooth with a causal Gaussian kernel (N=15, boundary='reflect'), (6) Remove neurons with mean FR ≤ 1 Hz.

ii.
```python
aligned_times = trialtm[spike_mask] - align_times[j]
counts = np.histogram(aligned_times, bins=edges)[0]
rate = counts.astype(np.float64) / DT
smoothed = causal_gaussian_smooth(rate, SMOOTH_N, SMOOTH_BC)
```

iii. The AI documents this as matching `getSeq.m` and `alignSpikes.m`. The dt=1/100 (10ms) is from `WorkingWithDataObjs.m`, and the causal Gaussian smoothing matches `mySmooth.m`. However, note: the default in `getDefaultParams.m` is dt=1/200 (5ms), and processData.m defaults bctype to 'none'.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering: (1) Quality filter: exclude clusters with quality labels in `{'garbage', 'gabrga', 'noisy', 'real?'}` (case-insensitive, whitespace-stripped). (2) Low-FR filter: remove neurons with mean firing rate ≤ 1 Hz. The low FR is computed as the mean across all trials and all time bins. Sessions with <10 neurons after filtering are skipped.

ii.
```python
EXCLUDE_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
quality = h5_deref_string(f, q_ref).strip().lower()
if quality in EXCLUDE_QUALITIES:
    continue

# Low FR filter
mean_frs = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
keep = mean_frs > LOW_FR_THRESHOLD
```

iii. Quality labels match `findClusters.m` with `quality={'all'}`. The 1 Hz threshold matches `WorkingWithDataObjs.m` and the paper. The CONVERSION_NOTES note a discrepancy with `getDefaultParams.m` (0.5 Hz) but chose 1 Hz based on the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the goCue event by subtracting the per-trial goCue time: `aligned_times = trialtm - align_times[trial]`. The aligned spike times are then binned into time bins spanning [-2.5, 2.5] seconds.

ii.
```python
ALIGN_EVENT = 'goCue'
align_times = get_event_times(data, fmt, ALIGN_EVENT)
aligned_times = trialtm[spike_mask] - align_times[j]
```

iii. Aligning to goCue matches both the instructions ("Temporally align based on Go cue onset") and the reference code's default `params.alignEvent = 'goCue'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 10 ms (dt = 1/100 s). No temporal rebinning is applied after the initial binning. The time axis spans [-2.5, 2.5] seconds from goCue, producing 500 time bins. Bin edges are computed as `tmin:dt:tmax` and bin centers as `edges[:-1] + dt/2`.

ii.
```python
DT = 1.0 / 100  # 10 ms time bins
def compute_time_axis():
    edges = np.arange(TMIN, TMAX + DT/2, DT)
    time_axis = edges[:-1] + DT / 2
    return time_axis, edges
```

iii. The AI chose 10ms based on `WorkingWithDataObjs.m` (dt=1/100). The CONVERSION_NOTES acknowledge that `getDefaultParams.m` uses dt=1/200 (5ms) but chose 10ms as the "main analysis" value.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the time axis itself (bin centers relative to goCue), not derived from any raw data variable. It is the same for every trial: a linearly spaced array from approximately -2.495 to 2.495 seconds.

ii.
```python
input_data = time_axis.astype(np.float32).reshape(1, -1)
input_trials.append(input_data)
```

iii. The AI treats this as a continuous time variable identical across trials, consistent with the decoder input specification "Time from go cue onset in seconds (continuous, time-varying)."

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing is required. The time axis is computed once from the binning parameters and reused for every trial. The values are the bin centers of the neural data time bins.

ii.
```python
time_axis, edges = compute_time_axis()
# edges = np.arange(TMIN, TMAX + DT/2, DT)
# time_axis = edges[:-1] + DT / 2
```

iii. This is a direct consequence of the temporal binning scheme and requires no transformation.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis is identical to the neural data time axis by construction. Both use the same bin centers, so alignment is trivially correct.

ii.
```python
# Same time_axis used for both neural binning and input
input_data = time_axis.astype(np.float32).reshape(1, -1)
```

iii. This is straightforward since the input IS the time axis of the neural data.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `obj.bp.R` and `obj.bp.L`, which indicate the correct response direction (right or left) for each trial. The AI uses R directly: R=1 (right), not-R=0 (left).

ii.
```python
R = get_bp_field(data, fmt, 'R').astype(bool)
lick_direction = R_valid.astype(np.float32)
```

iii. The CONVERSION_NOTES extensively discuss this decision, noting that R/L indicates the CORRECT response direction, not the actual lick direction. For hit trials, actual lick = instructed side. For miss trials, actual lick = opposite side. The AI chose to use the instruction/stimulus direction.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Minimal processing: the boolean R field is cast to float (R=True→1.0, R=False→0.0). This value is broadcast across all time bins as a per-trial constant.

ii.
```python
lick_direction = R_valid.astype(np.float32)
out[0, :] = int(lick_direction[t_idx])
```

iii. The AI treats lick direction as per-trial (constant across time), which matches the instruction specification "per-trial."

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj.bp.autowater`. When autowater is True (1), the context is WC (water-cued) = 0. When autowater is False (0), the context is DR (delayed-response) = 1.

ii.
```python
autowater = get_bp_field(data, fmt, 'autowater').astype(bool)
behavioral_context = (~autowater_valid).astype(np.float32)
```

iii. The mapping WC=0, DR=1 matches the instructions. The use of `autowater` to distinguish contexts matches the reference code conditions where `autowater` separates WC from DR trials.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Simple boolean inversion and cast: autowater=True→WC=0, autowater=False→DR=1. Broadcast across all time bins as per-trial constant.

ii.
```python
behavioral_context = (~autowater_valid).astype(np.float32)
out[1, :] = int(behavioral_context[t_idx])
```

iii. Straightforward mapping consistent with the instruction specification.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit`. Hit trials (correct) are coded as 1, all others (miss, no/ignore) as 0.

ii.
```python
hit = get_bp_field(data, fmt, 'hit').astype(bool)
outcome = hit_valid.astype(np.float32)
```

iii. The AI defines correct=hit=1 and incorrect=(not hit)=0, which includes both miss trials and ignore ("no") trials as incorrect.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Simple boolean cast: hit=True→1, hit=False→0. Broadcast across all time bins as per-trial constant.

ii.
```python
outcome = hit_valid.astype(np.float32)
out[2, :] = int(outcome[t_idx])
```

iii. Per-trial constant as specified in instructions.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the 'tongue' feature tracked by DeepLabCut in the side camera (view=0). Specifically, `obj.traj{1}(trial).ts` provides x and y coordinates, and `obj.traj{1}(trial).frameTimes` provides the video frame timestamps.

ii.
```python
tongue_speed = extract_velocity_from_traj(
    data, fmt, view=0, feat_name='tongue',
    ntrials=ntrials, align_times=align_times,
    time_axis=time_axis, vidshift=vidshift
)
```

iii. The AI chose to use the 'tongue' feature from the side camera, reasoning that the task specifies "tongue velocity."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial: (1) Extract x,y positions from DLC data. (2) For tongue features, no position smoothing is applied. (3) Compute velocity using `np.gradient()`. (4) Set NaN velocities to 0 (tongue not visible). (5) Compute speed as `sqrt(xvel^2 + yvel^2)`. (6) Fill remaining NaNs with nearest. (7) Interpolate speed from video frame times to neural time axis using `np.interp()`.

ii.
```python
xvel = np.gradient(xpos_smooth)
yvel = np.gradient(ypos_smooth)
if is_tongue:
    xvel[np.isnan(xvel)] = 0
    yvel[np.isnan(yvel)] = 0
spd = np.sqrt(xvel**2 + yvel**2)
interpolated = np.interp(time_axis, aligned_ft, spd)
```

iii. The AI claims this matches `findVelocity.m` logic. However, note that the AI computes velocity at video frame rate and then interpolates speed, whereas the reference code first interpolates position to neural time axis then computes velocity.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per-session 50th percentile threshold computed across all valid timepoints and trials. Values below threshold → 0, values >= threshold → 1. A minimum threshold of 1e-10 is used to handle degenerate cases where median is 0.

ii.
```python
tongue_thresh = np.nanpercentile(tongue_speed_valid, 50)
tongue_disc = discretize_time_series(tongue_speed_valid, tongue_thresh)

def discretize_time_series(values, threshold):
    if threshold < 1e-10:
        threshold = 1e-10
    return (values >= threshold).astype(np.float32)
```

iii. This matches the instruction specification: "discretized into two bins with per-session threshold: 0: < 50th percentile, 1: >= 50th percentile."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Video frame times are aligned to goCue by subtracting the video offset and the per-trial goCue time: `aligned_ft = frame_times - vidshift - align_times[trial]`. The velocity (speed) is then interpolated from these aligned frame times to the neural time axis using linear interpolation.

ii.
```python
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
interpolated = np.interp(time_axis, aligned_ft, spd)
```

iii. The video offset is computed via `find_video_offset()` matching `findVideoOffset.m`. The interpolation to neural time axis matches `loadMotionEnergy.m` and `findPosition.m` approaches.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the 'top_paw' feature tracked by DeepLabCut in the bottom camera (view=1).

ii.
```python
paw_speed = extract_velocity_from_traj(
    data, fmt, view=1, feat_name='top_paw',
    ntrials=ntrials, align_times=align_times,
    time_axis=time_axis, vidshift=vidshift
)
```

iii. The reference code lists 'top_paw' as a tracked feature from the bottom camera, consistent with the paper's statement that paws were tracked using only the bottom view.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same pipeline as tongue velocity, but for non-tongue features: (1) Extract x,y positions. (2) Apply causal Gaussian smoothing to positions (N=21, boundary='reflect'). (3) Compute velocity via gradient. (4) Subtract baseline velocity (median of each velocity component). (5) Compute speed. (6) Fill NaNs with nearest. (7) Interpolate to neural time axis.

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
```

iii. The AI applies position smoothing with N=21 for non-tongue features and subtracts median velocity as baseline.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue velocity: per-session 50th percentile threshold across all valid timepoints and trials.

ii.
```python
paw_thresh = np.nanpercentile(paw_speed_valid, 50)
paw_disc = discretize_time_series(paw_speed_valid, paw_thresh)
```

iii. Matches instruction specification for per-session 50th percentile discretization.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue velocity: video frame times aligned to goCue, speed interpolated to neural time axis.

ii.
```python
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
interpolated = np.interp(time_axis, aligned_ft, spd)
```

iii. Same alignment approach as tongue velocity.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from separate `motionEnergy_<animal>_<date>.mat` files. The motion energy data is a per-trial time series at approximately 400 Hz.

ii.
```python
def load_motion_energy(dirpath, animal, date):
    me_file = os.path.join(dirpath, f'motionEnergy_{animal}_{date}.mat')
    me_mat = scipy.io.loadmat(me_file, squeeze_me=False)
    me_raw = me_mat['me']
```

iii. This matches the reference `loadMotionEnergy.m` which loads motion energy from separate files.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. For each trial: (1) Load per-trial motion energy array. (2) Get video frame times from the DLC trajectory data (side camera). (3) Align frame times to goCue (subtract video offset and goCue time). (4) Interpolate motion energy to neural time axis using `np.interp()`. (5) Fill NaNs with nearest neighbor.

ii.
```python
me.newdata(:,trix) = interp1(obj.traj{1}(trix).frameTimes-vidshift-alignTimes(trix), me.data{trix}, taxis);
# AI equivalent:
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
me_interp[:, trial_idx] = np.interp(time_axis, aligned_ft[:n_frames], me_trial[:n_frames])
```

iii. The interpolation approach matches `loadMotionEnergy.m` which uses `interp1(frameTimes-vidshift-alignTimes, me.data, taxis)`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session 50th percentile threshold across all valid timepoints and trials.

ii.
```python
me_thresh_50 = np.nanpercentile(me_valid, 50)
me_disc = discretize_time_series(me_valid, me_thresh_50)
```

iii. Follows instruction specification for 50th percentile discretization. Note: the paper describes a per-session manual threshold separating a bimodal distribution, but the instructions explicitly specify 50th percentile.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned using the same video frame times as DLC data. The frame times are shifted by the video offset and goCue time, then motion energy is interpolated to the neural time axis.

ii.
```python
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
me_interp[:, trial_idx] = np.interp(time_axis, aligned_ft[:n_frames], me_trial[:n_frames])
```

iii. Matches `loadMotionEnergy.m` alignment logic.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several missing data scenarios are handled:
- **Missing goCue times**: Trials with NaN or 0 goCue times are excluded.
- **Recording ended early**: Trials beyond max spike trial number are excluded (found in 2 sessions: JEB24_2023-10-23 and JEB24_2023-11-03).
- **Missing stim.enable field**: Defaults to all-zeros (no stim).
- **Missing motion energy files**: Fills with zeros.
- **Missing DLC video data**: Fills velocity with NaN, then zeros for trials with no data.
- **NaN velocities**: Non-tongue features filled with nearest neighbor; tongue NaN velocities set to 0.
- **Missing frame times**: Synthetic frame times created at 400 Hz with default 0.5s offset.
- **Zero median threshold**: Minimum threshold of 1e-10 to prevent degenerate discretization.

ii.
```python
valid_trials &= ~np.isnan(align_times) & (align_times > 0)

if beyond > 0:
    valid_trial_indices = valid_trial_indices[valid_trial_indices < max_spike_trial]

try:
    stim_enable = get_bp_field(data, fmt, 'stim.enable').astype(bool)
except Exception:
    stim_enable = np.zeros(ntrials, dtype=bool)
```

iii. The CONVERSION_NOTES document the recording-extent issue in Steps 9-10 and describe the fix applied.

## 11-a. What are the most time-consuming steps of the code?

i. Based on timing information in the code output, the most time-consuming steps per session are: (1) Spike binning and smoothing (~2-4s per session), which loops over all neurons and all trials. (2) DLC velocity extraction for tongue and paw (~1-2s each per session), which loops over all trials with per-trial DLC data loading and interpolation.

ii.
```python
t1 = time.time()
trialdat = bin_and_smooth_spikes(all_clusters, ntrials, align_times, time_axis, edges)
print(f"    Spike binning: {time.time()-t1:.1f}s")
```

iii. The CONVERSION_NOTES report ~6-7s per session total, with the full conversion taking ~269 seconds for 44 sessions.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The main nested loop is in `bin_and_smooth_spikes`: it loops over neurons (outer) and trials (inner), calling `np.histogram` and `causal_gaussian_smooth` per neuron-trial pair. The smoothing could potentially be vectorized using `scipy.ndimage.convolve1d` across all neurons/trials at once. The DLC velocity extraction also has a per-trial loop that could potentially batch-process trials.

ii.
```python
for i, clu in enumerate(clusters):  # loop over neurons
    for j in range(ntrials):  # loop over trials
        counts = np.histogram(aligned_times, bins=edges)[0]
        rate = counts.astype(np.float64) / DT
        smoothed = causal_gaussian_smooth(rate, SMOOTH_N, SMOOTH_BC)
```

iii. The CONVERSION_NOTES mention that `np.histogram` is already vectorized per call, but the outer loops remain.

## 11-c. What processing does the code repeat multiple times?

i. The DLC trajectory data is loaded 3 times per session: once for tongue velocity, once for paw velocity, and once for motion energy frame times. Each call to `extract_velocity_from_traj` and `interpolate_motion_energy` independently calls `get_traj_data` for each trial. The code also bins spikes for ALL trials (including invalid ones) and then discards invalid trials afterward, wasting computation on stim/early/invalid trials.

ii.
```python
# Called separately for tongue, paw, and ME:
tongue_speed = extract_velocity_from_traj(data, fmt, view=0, feat_name='tongue', ...)
paw_speed = extract_velocity_from_traj(data, fmt, view=1, feat_name='top_paw', ...)
# ME also loads frame times from traj:
ts, frame_times, _ = get_traj_data(data, fmt, 0, trial_idx)
```

iii. The AI noted this as a potential inefficiency but chose simplicity over optimization since total runtime was acceptable (~5 minutes).

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The code bins and smooths spikes for ALL trials (including stim, early, and invalid trials), then only uses valid trials. This wastes computation on ~8-10% of trials. (2) The code computes velocity for all trials including invalid ones, then selects only valid trials. (3) The causal Gaussian smoothing is applied to every neuron-trial pair, but neurons that fail the low-FR filter are subsequently discarded. Filtering neurons first would avoid smoothing discarded neurons.

ii.
```python
# Bins ALL ntrials, then selects valid subset:
trialdat = bin_and_smooth_spikes(all_clusters, ntrials, align_times, time_axis, edges)
trialdat, filtered_clusters, keep_mask = remove_low_fr_neurons(trialdat, all_clusters)
trialdat_valid = trialdat[:, :, valid_trial_indices]
```

iii. The AI prioritized code simplicity and correctness over efficiency, noting that the total runtime was within acceptable bounds (~5 minutes).
