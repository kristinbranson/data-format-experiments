# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hardcodes a list of 45 sessions (EPHYS_SESSIONS) with animal name, date, probe numbers, and data directory. For each session, it loads a `.mat` file (`data_structure_{anm}_{date}.mat`) from the appropriate data directory (`Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior`). It uses a unified loader class (`SessionData`) that detects whether the file is HDF5 (v7.3) or v5 MATLAB format and provides a common interface for accessing trial info, cluster data, trajectory data, and motion energy.

ii.
```python
EPHYS_SESSIONS = [
    ('EKH1', '2021-08-07', [2], 'Ephys_Behavior'),
    ('EKH3', '2021-08-11', [2], 'Ephys_Behavior'),
    ...
    ('JEB24', '2023-11-03', [1], 'RandomizedDelay_Ephys_Behavior'),
]

class SessionData:
    def __init__(self, fpath):
        self.is_h5 = self._check_h5(fpath)
        if self.is_h5:
            self.f = h5py.File(fpath, 'r')
            self.obj = self.f['obj']
        else:
            mat = sio.loadmat(fpath, squeeze_me=True, struct_as_record=False)
            self.obj_v5 = mat['obj']
```

iii. The AI derived the session list from the reference loading scripts (`loadJEB7_ALMVideo.m`, etc.) and the available data files. It handles both MATLAB file formats as needed by the data.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the animal name (`anm`) from the hardcoded session list. Unique subject names are collected in order of first appearance and stored in a `subjects` list. Each session is assigned a `subject_idx` pointing into this list.

ii.
```python
subjects_set = []
for anm, date, probes, data_dir in available:
    result = process_session(anm, date, probes, data_dir, time_edges, time_axis)
    if result is not None:
        all_sessions.append(result)
        if anm not in subjects_set:
            subjects_set.append(anm)

subjects = subjects_set
subject_idx = np.array([subjects.index(s['anm']) for s in all_sessions])
```

iii. Subject identity comes directly from the animal name field in the session definitions, matching the reference code's `meta.anm` field.

## 1-c. How are the data split into sessions?

i. Each entry in the EPHYS_SESSIONS list defines one session by (animal, date, probes, data_dir). Each session is processed independently by `process_session()`. Multi-probe sessions list multiple probe numbers (e.g., `[1, 2]` for JEB15), and all probes are loaded together as a single session.

ii.
```python
for anm, date, probes, data_dir in available:
    result = process_session(anm, date, probes, data_dir, time_edges, time_axis)
```

iii. This matches the reference code's approach where each `meta` entry defines a session and multi-probe sessions concatenate cluster data across probes (see `loadSessionData.m` lines 47-68).

## 1-d. How are the data split into trials?

i. The number of trials per session is obtained from `obj.bp.Ntrials`. Spike data is binned per trial by filtering cluster spike times by trial number. Only trials passing quality filters are included in the final output.

ii.
```python
ntrials = sd.get_ntrials()
# ...
for j in range(ntrials):
    trial_mask = (trial_nums == (j + 1))
    # ...
    aligned_times = spike_times[trial_mask] - gocue[j]
    counts, _ = np.histogram(aligned_times, bins=time_edges)
```

iii. This matches the reference code's approach in `getSeq.m` where single-trial data is computed for all trials (1 to Ntrials).

## 1-e. How are trials filtered based on quality controls?

i. Valid trials must be hit OR miss (excluding ignore/no-response), must not have stimulation enabled, and must not be early-lick trials. Sessions with fewer than 5 valid trials or fewer than 2 final valid trials are skipped.

ii.
```python
hit = sd.get_trial_array('hit').astype(bool)
miss = sd.get_trial_array('miss').astype(bool)
stim_enable = sd.get_stim_enable().astype(bool)
early = sd.get_trial_array('early').astype(bool)

valid_trials = (hit | miss) & ~stim_enable & ~early
valid_idx = np.where(valid_trials)[0]
```

iii. The AI explains this matches the reference code conditions: excluding stim-enabled, early-lick, and ignore trials. Including both hit and miss trials is needed for the decoder's "outcome" output variable.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from sorted spike clusters (`obj.clu`), specifically the `trial`, `trialtm`, and `quality` fields for each cluster on each probe.

ii.
```python
def get_clusters(self, probe_nums):
    # For each cluster: quality, trial (spike trial assignments), trialtm (spike times within trial)
    clusters.append((q, trial, trialtm))
```

iii. This matches the reference code's use of `obj.clu{prbnum}(curClu).trial` and `obj.clu{prbnum}(curClu).trialtm` in `alignSpikes.m` and `getSeq.m`.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue by subtracting the go cue time for each trial. Aligned spikes are binned into time bins from -2.5 to 2.5 seconds in 10ms bins. Spike counts are converted to firing rates (counts/dt). Firing rates are smoothed with a causal Gaussian kernel (window=15 bins, reflect boundary condition).

ii.
```python
aligned_times = spike_times[trial_mask] - gocue[j]
counts, _ = np.histogram(aligned_times, bins=time_edges)
fr = counts / DT
trialdat[i, :, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)
```

iii. The AI states this matches `getSeq.m` and `mySmooth.m`. The parameters (10ms bins, 15-bin Gaussian, reflect BC) come from `WorkingWithDataObjs.m`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering: (1) Clusters with quality labels 'garbage', 'gabrga', 'noisy', or 'real?' are excluded. (2) After binning and smoothing, units with mean firing rate <= 1 Hz across valid trials are removed. Sessions with <10 remaining units are skipped.

ii.
```python
EXCLUDE_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
filtered_clusters = [(q, t, tt) for q, t, tt in clusters if q not in EXCLUDE_QUALITIES]

# After binning:
mean_frs = np.mean(np.mean(trialdat[:, :, valid_idx], axis=2), axis=1)
keep_units = mean_frs > LOW_FR_THRESH  # LOW_FR_THRESH = 1.0
```

iii. The AI states this matches `findClusters.m` (quality='all' excludes garbage/noisy) and `removeLowFRClusters.m`. The 1 Hz threshold comes from `WorkingWithDataObjs.m`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes are aligned to go cue onset by subtracting `bp.ev.goCue` for each trial from spike times within that trial, then binning the aligned times.

ii.
```python
gocue = sd.get_event_times('goCue')
# ...
aligned_times = spike_times[trial_mask] - gocue[j]
counts, _ = np.histogram(aligned_times, bins=time_edges)
```

iii. This matches `alignSpikes.m` which computes `trialtm_aligned = trialtm - event` where event is `obj.bp.ev.goCue`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 10ms (DT = 1/100). This is applied directly when binning spikes; no rebinning is performed after initial binning.

ii.
```python
DT = 1/100  # 10 ms
time_edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = time_edges[:-1] + DT / 2
```

iii. The AI states this matches the reference code parameters. The value 1/100 (10ms) matches `WorkingWithDataObjs.m` line 213 (`params.dt = 1/100`), though `getDefaultParams.m` uses `params.dt = 1/200` (5ms).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is derived directly from the time axis (bin centers of the spike-binning time grid), not from any raw data variable. The time axis is computed relative to the go cue alignment.

ii.
```python
time_axis = time_edges[:-1] + DT / 2
# In process_session:
input_data = time_axis.reshape(1, -1).copy()
```

iii. The time axis represents bin centers from -2.495 to 2.495 seconds relative to go cue onset.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing beyond computing the bin center times. The time axis is a linearly spaced vector from TMIN+DT/2 to TMAX-DT/2.

ii.
```python
time_edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = time_edges[:-1] + DT / 2
input_data = time_axis.reshape(1, -1).copy()
```

iii. This is consistent with the reference code's `obj.time = edges + params.dt/2; obj.time = obj.time(1:end-1)`.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time axis is identical for both neural data and the input, since both use the same bin centers. The input is simply the time axis copied for each trial.

ii.
```python
input_data = time_axis.reshape(1, -1).copy()  # shape (1, n_timebins)
```

iii. Perfect alignment since both are defined on the same time grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `obj.bp.R` (right lick indicator). If R is true for a trial, lick direction = 1 (right); otherwise = 0 (left).

ii.
```python
R = sd.get_trial_array('R').astype(bool)
lick_dir = 1 if R[trial_idx] else 0
```

iii. The AI states this uses bp.R as the right-lick indicator, matching the instructions (left=0, right=1).

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A simple binary mapping: R=True maps to 1 (right), R=False maps to 0 (left). This is broadcast across all time bins as a per-trial constant.

ii.
```python
lick_dir = 1 if R[trial_idx] else 0
out[0, :] = t['lick_dir']  # constant across time
```

iii. Matches the instruction specification: left=0, right=1, per-trial.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj.bp.autowater`. When autowater=1, the context is WC (water-cued); otherwise, it is DR (delayed-response).

ii.
```python
autowater = sd.get_trial_array('autowater')
context = 0 if autowater[trial_idx] == 1 else 1
```

iii. The AI references `WorkingWithDataObjs.m` which states: "obj.bp.autowater=1 when water was delivered regardless of animal choice...this field can be used as a proxy for obtaining water-cued blocks and delayed-response blocks."

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A simple mapping: autowater==1 maps to 0 (WC), autowater!=1 maps to 1 (DR). Broadcast across all time bins as a per-trial constant.

ii.
```python
context = 0 if autowater[trial_idx] == 1 else 1
out[1, :] = t['context']
```

iii. Matches instructions: WC=0, DR=1.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit`. Hit (correct) trials map to 1, miss (incorrect) trials map to 0.

ii.
```python
hit = sd.get_trial_array('hit').astype(bool)
outcome = 1 if hit[trial_idx] else 0
```

iii. Since only hit and miss trials pass the trial filter, this binary mapping covers all included trials.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A simple binary mapping: hit=True maps to 1, hit=False maps to 0. Broadcast across all time bins.

ii.
```python
outcome = 1 if hit[trial_idx] else 0
out[2, :] = t['outcome']
```

iii. Matches instructions: incorrect=0, correct=1.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from DLC tracking data for the `top_tongue` feature from the bottom camera (`obj.traj{2}` in MATLAB, view=1 in the AI's 0-indexed scheme). Specifically, the x and y coordinates from `traj.ts` and frame timing from `traj.frameTimes`.

ii.
```python
bottom_feats = sd.get_traj_feature_names(1)  # bottom camera
# Find 'top_tongue' feature index
for idx, name in enumerate(bottom_feats):
    if name == 'top_tongue' and tongue_feat_idx is None:
        tongue_feat_idx = idx

ft, x, y = sd.get_trial_traj(1, trial_idx, tongue_feat_idx)
```

iii. The AI uses the `top_tongue` feature from the bottom camera, consistent with the reference code's `params.traj_features` which lists `top_tongue` for the bottom camera (cam1).

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI computes velocity magnitude from raw video coordinates: (1) compute `diff(x)` and `diff(y)` at video frame rate, (2) compute speed as `sqrt(dx^2 + dy^2) / dt_vid` where dt_vid = 1/400, (3) assign velocity to midpoint times between frames, (4) align to go cue by subtracting video offset and go cue time, (5) interpolate to neural time bins using `np.interp`.

ii.
```python
dt_vid = 1.0 / VIDEO_FPS
vel = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt_vid
vel_t = ft[:-1] + dt_vid / 2
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
tongue_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. The AI computes Euclidean speed from raw DLC tracking, then interpolates to neural time bins. This differs from the reference code which first interpolates position to neural time bins, then computes velocity using `gradient()` (central differences).

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per session, the 50th percentile of all tongue velocity values (across valid trials, including NaN timepoints excluded via nanpercentile) is computed. Values >= threshold map to 1, values < threshold map to 0. NaN values (outside video coverage) are mapped to 0.

ii.
```python
all_tongue = np.concatenate([t['tongue_vel'] for t in raw_outputs])
tongue_thresh = np.nanpercentile(all_tongue, 50) if not np.all(np.isnan(all_tongue)) else 0

tongue_disc = np.where(np.isnan(t['tongue_vel']), 0,
                      np.where(t['tongue_vel'] >= tongue_thresh, 1, 0)).astype(np.int64)
```

iii. Matches instructions: per-session threshold at 50th percentile, 0 for < 50th percentile, 1 for >= 50th percentile.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Video frame times are aligned to go cue by subtracting the video offset and go cue time: `vel_t_aligned = vel_t - vidshift - goCue[trial]`. The aligned velocity is then interpolated to the neural time axis.

ii.
```python
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
tongue_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. The video offset is computed as `mode(sglx.bitcode.bitstart)/sglx.fs - mode(bp.ev.bitStart)`, matching `findVideoOffset.m`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from DLC tracking data for the `bottom_paw` feature from the bottom camera (view=1), using the x and y coordinates from `traj.ts`.

ii.
```python
for idx, name in enumerate(bottom_feats):
    if name == 'bottom_paw' and paw_feat_idx is None:
        paw_feat_idx = idx

ft, x, y = sd.get_trial_traj(1, trial_idx, paw_feat_idx)
```

iii. Uses `bottom_paw` from the bottom camera, consistent with the reference code's feature list.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Identical to tongue velocity: compute Euclidean speed from raw DLC coordinates at 400 Hz using `np.diff`, then interpolate to neural time bins.

ii.
```python
vel = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt_vid
vel_t = ft[:-1] + dt_vid / 2
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
paw_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. Same approach as tongue velocity. Note: the reference code applies `fillmissing('nearest')` for paw position before computing velocity and subtracts baseline derivative; the AI does neither.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same approach as tongue velocity: per-session 50th percentile threshold, NaN mapped to 0.

ii.
```python
all_paw = np.concatenate([t['paw_vel'] for t in raw_outputs])
paw_thresh = np.nanpercentile(all_paw, 50) if not np.all(np.isnan(all_paw)) else 0

paw_disc = np.where(np.isnan(t['paw_vel']), 0,
                   np.where(t['paw_vel'] >= paw_thresh, 1, 0)).astype(np.int64)
```

iii. Matches instructions for per-session 50th percentile thresholding.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue velocity: frame times aligned via video offset and go cue subtraction, then interpolated to neural time axis.

ii.
```python
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
paw_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. Consistent alignment approach using `findVideoOffset`.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from separate `.mat` files (`motionEnergy_{anm}_{date}.mat`). The `me` variable contains per-trial motion energy data (one value per video frame) and a `moveThresh` threshold.

ii.
```python
def load_motion_energy(data_dir, anm, date):
    me_fn = f'motionEnergy_{anm}_{date}.mat'
    me_path = os.path.join(DATA_ROOT, data_dir, me_fn)
    me_mat = sio.loadmat(me_path, squeeze_me=False, struct_as_record=True)
    me_var = me_mat['me']
    # Extract me.data (per-trial cell array) and me.moveThresh
```

iii. Matches the reference `loadMotionEnergy.m` which loads from `motionEnergy_*.mat` files.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Per-trial motion energy (at 400 Hz) is aligned to go cue using bottom camera frame times and video offset, then interpolated to neural time bins.

ii.
```python
ft = sd.get_trial_frame_times(1, trial_idx)  # bottom camera frame times
ft_aligned = ft - vidshift - gocue[trial_idx]
me_interp = np.interp(time_axis, ft_aligned, me_data[trial_idx], left=np.nan, right=np.nan)
```

iii. The reference code (`loadMotionEnergy.m` line 71) uses side camera frame times (`obj.traj{1}`) for alignment, while the AI uses bottom camera frame times (view=1 in 0-indexed Python). The reference also applies `fillmissing('nearest')` after interpolation; the AI does not.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session 50th percentile threshold. NaN values mapped to 0.

ii.
```python
all_me = np.concatenate([t['motion_energy'] for t in raw_outputs])
me_thresh_50 = np.nanpercentile(all_me, 50) if not np.all(np.isnan(all_me)) else 0

me_disc = np.where(np.isnan(t['motion_energy']), 0,
                  np.where(t['motion_energy'] >= me_thresh_50, 1, 0)).astype(np.int64)
```

iii. The instructions specify per-session 50th percentile thresholding. Note: the reference code uses a manually-set `moveThresh` (bimodal threshold) for motion energy binarization, not a percentile threshold. The AI follows the decoder instructions rather than the reference code's `me.move = me.data > me.moveThresh`.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Bottom camera frame times are aligned via video offset and go cue subtraction, then motion energy is interpolated to neural time axis.

ii.
```python
ft = sd.get_trial_frame_times(1, trial_idx)
ft_aligned = ft - vidshift - gocue[trial_idx]
me_interp = np.interp(time_axis, ft_aligned, me_data[trial_idx], left=np.nan, right=np.nan)
```

iii. Uses the same alignment approach as kinematic features, but with the bottom camera frame times instead of the reference code's side camera frame times.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple strategies:
- Sessions with missing data files are skipped.
- Sessions with <5 valid trials or <10 units after filtering are skipped.
- Tongue/paw trajectory extraction failures are caught by try/except and default to NaN arrays.
- NaN values in velocity/motion energy are mapped to category 0 during discretization.
- Missing motion energy files result in NaN arrays for the session.
- HDF5 vs v5 MATLAB format differences are auto-detected and handled.

ii.
```python
if not os.path.exists(fpath):
    print(f'  WARNING: File not found: {fpath}')
    return None

tongue_vel = np.full(n_timebins, np.nan)
if tongue_feat_idx is not None:
    try:
        ft, x, y = sd.get_trial_traj(1, trial_idx, tongue_feat_idx)
        # ...
    except:
        pass

tongue_disc = np.where(np.isnan(t['tongue_vel']), 0, ...)
```

iii. The AI handles missing data gracefully but maps NaN to category 0 rather than filling with nearest values as the reference code does for non-tongue features.

## 11-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading large HDF5/MATLAB files for each session, (2) The triple-nested loop for spike binning (units x trials x spikes per trial), (3) Gaussian smoothing applied per-unit per-trial, (4) Video trajectory extraction and interpolation for each trial.

ii.
```python
for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
    for j in range(ntrials):
        trial_mask = (trial_nums == (j + 1))
        aligned_times = spike_times[trial_mask] - gocue[j]
        counts, _ = np.histogram(aligned_times, bins=time_edges)
        fr = counts / DT
        trialdat[i, :, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)
```

iii. The spike binning and smoothing loops dominate processing time since they iterate over all units and ALL trials (not just valid ones).

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike histogram computation (inner loop over trials for each unit) could be partially vectorized by grouping spikes by trial first and using vectorized histogram operations. The Gaussian smoothing loop over columns could use `scipy.ndimage.convolve1d` for the whole matrix. The per-trial velocity computation loop could be vectorized with array operations.

ii.
```python
# This nested loop could be vectorized:
for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
    for j in range(ntrials):
        trial_mask = (trial_nums == (j + 1))
        # ...

# Smoothing loop over columns:
for j in range(x_filt.shape[1]):
    out[:, j] = np.convolve(x_filt[:, j], kern, mode='same')
```

iii. No justification given by the AI for the loop-based approach vs vectorized alternatives.

## 11-c. What processing does the code repeat multiple times?

i. The video offset (`get_video_offset()`) is computed once per session, which is efficient. However, the frame times for the bottom camera are loaded separately for tongue, paw, and motion energy processing, repeating the same HDF5 read operations. Also, `get_trial_traj` and `get_trial_frame_times` are called separately for each trial in loops, rather than loading all trial data at once.

ii.
```python
# For each valid trial, these are called separately:
ft, x, y = sd.get_trial_traj(1, trial_idx, tongue_feat_idx)
ft, x, y = sd.get_trial_traj(1, trial_idx, paw_feat_idx)
ft = sd.get_trial_frame_times(1, trial_idx)
```

iii. Each call to `get_trial_traj` accesses the HDF5 file, which could be batched.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `trialdat` (spike-binned, smoothed firing rates) for ALL trials including invalid ones (stim, early, ignore), but only valid trials are used in the output. This wastes significant computation, especially for smoothing. The `me_thresh` value loaded from the motion energy file is never used (the AI computes its own 50th percentile threshold instead).

ii.
```python
trialdat = np.zeros((n_units, n_timebins, ntrials))  # ALL trials
for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
    for j in range(ntrials):  # loops over ALL ntrials, not just valid_idx
        # ...bins and smooths for every trial

# Only valid trials are used later:
for trial_idx in valid_idx:
    neural = trialdat[:, :, trial_idx]
```

iii. Processing all trials matches the reference code (`getSeq.m` also processes all trials), but it's unnecessary computation for the decoder task which only uses valid trials.
