# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI defines a hardcoded list `EPHYS_SESSIONS` of 45 session tuples `(animal, date, probes, data_dir)` covering two directories: `Ephys_Behavior` (25 sessions) and `RandomizedDelay_Ephys_Behavior` (20 sessions). For each session, it loads the `.mat` file `data_structure_{anm}_{date}.mat`. It auto-detects HDF5 (v7.3) vs v5 MAT format and uses the appropriate loader (`h5py` or `scipy.io.loadmat`). All data fields (behavior, spikes, trajectory, motion energy) are loaded from the `obj` struct in each file.

ii.
```python
EPHYS_SESSIONS = [
    ('EKH1', '2021-08-07', [2], 'Ephys_Behavior'),
    ('EKH3', '2021-08-11', [2], 'Ephys_Behavior'),
    ...
    ('JEB24', '2023-11-03', [1], 'RandomizedDelay_Ephys_Behavior'),
]

# In SessionData.__init__:
if self.is_h5:
    self.f = h5py.File(fpath, 'r')
    self.obj = self.f['obj']
else:
    mat = sio.loadmat(fpath, squeeze_me=True, struct_as_record=False)
    self.obj_v5 = mat['obj']
```

iii. The AI identified the session list by examining the available data files and matching them to sessions described in the paper. The dual-format loader was needed because some sessions use MATLAB v7.3 HDF5 format while others use v5 format.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the animal name (`anm`) field in each session tuple. A unique ordered list of subjects is built by iterating through successfully processed sessions. Each session is assigned a `subject_idx` mapping it to the subjects list.

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

iii. The AI built the subjects list from the animal names in the session definitions. 14 unique subjects were identified.

## 1-c. How are the data split into sessions?

i. Each entry in `EPHYS_SESSIONS` defines one session. A session corresponds to one `.mat` file (one recording day from one animal). When multiple probes are specified, clusters from all probes are combined into a single session. Each session is processed independently via `process_session()`.

ii.
```python
for anm, date, probes, data_dir in available:
    result = process_session(anm, date, probes, data_dir, time_edges, time_axis)
    if result is not None:
        all_sessions.append(result)
```

iii. The AI treated each date-animal combination as a session, matching the reference code structure. Multi-probe sessions (e.g., JEB15 with probes [1,2]) concatenate clusters from both probes.

## 1-d. How are the data split into trials?

i. The total number of trials per session is obtained from `bp.Ntrials`. Spikes are binned for ALL trials, then valid trials are selected based on filtering criteria. Valid trial indices are stored in `valid_idx`.

ii.
```python
ntrials = sd.get_ntrials()
# ...
valid_trials = (hit | miss) & ~stim_enable & ~early
valid_idx = np.where(valid_trials)[0]
# ...
trialdat = np.zeros((n_units, n_timebins, ntrials))
for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
    for j in range(ntrials):
        # bin spikes for each trial
```

iii. Spikes are binned for all trials first (matching the reference `getSeq.m` which loops over `1:obj.bp.Ntrials`), then only valid trials are extracted for the output.

## 1-e. How are trials filtered based on quality controls?

i. Trials are included if they are hit (correct) or miss (incorrect), and excluded if stimulation was enabled, or if the animal licked early. Ignore/no-response trials (`no`) are implicitly excluded by requiring `hit | miss`. Sessions with fewer than 5 valid trials or fewer than 2 valid trials after processing are skipped.

ii.
```python
hit = sd.get_trial_array('hit').astype(bool)
miss = sd.get_trial_array('miss').astype(bool)
stim_enable = sd.get_stim_enable().astype(bool)
early = sd.get_trial_array('early').astype(bool)

valid_trials = (hit | miss) & ~stim_enable & ~early
valid_idx = np.where(valid_trials)[0]

if len(valid_idx) < 5:
    print(f'  WARNING: Too few valid trials ({len(valid_idx)}), skipping')
```

iii. The filtering matches the reference code's conditions 2-9 which all require `(hit|miss) & ~stim.enable & ~early`. The exclusion of `no` (ignore) trials is implicit since `no` trials are neither `hit` nor `miss`. The AI also excludes autowater trials from the `no` category through the `hit|miss` requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `obj.clu` (sorted spike cluster data). Each cluster has fields: `quality` (cluster quality label), `trial` (which trial each spike belongs to), and `trialtm` (spike time within each trial). Spikes are aligned to go cue times from `obj.bp.ev.goCue`.

ii.
```python
clusters = sd.get_clusters(probe_nums)
# Each cluster: (quality, trial_array, trialtm_array)

# Alignment:
aligned_times = spike_times[trial_mask] - gocue[j]
```

iii. The AI correctly identified that spike data is stored in `obj.clu` as described in the reference code `WorkingWithDataObjs.m`.

## 2-b. How is the `neural` data processed?

i. Processing pipeline:
1. Filter clusters by quality (exclude garbage, gabrga, noisy, real?)
2. For each cluster and trial, align spike times to go cue
3. Bin spikes into 10ms bins from -2.5 to 2.5s
4. Convert spike counts to firing rates (counts / dt)
5. Apply causal Gaussian smoothing (window=15 bins, reflect boundary)
6. Remove low firing rate units (mean FR <= 1 Hz)

ii.
```python
# Binning
aligned_times = spike_times[trial_mask] - gocue[j]
counts, _ = np.histogram(aligned_times, bins=time_edges)
fr = counts / DT

# Smoothing
trialdat[i, :, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)

# Low FR removal
mean_frs = np.mean(np.mean(trialdat[:, :, valid_idx], axis=2), axis=1)
keep_units = mean_frs > LOW_FR_THRESH
```

iii. The processing matches the reference code pipeline: `findClusters` -> `alignSpikes` -> `getSeq` (binning + smoothing) -> `removeLowFRClusters`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two quality filters are applied:
1. **Cluster quality**: Clusters with quality labels 'garbage', 'gabrga', 'noisy', or 'real?' are excluded (matching `findClusters.m` with `quality='all'`).
2. **Firing rate threshold**: Units with mean firing rate <= 1 Hz across valid trials are removed.
3. Sessions with fewer than 10 remaining units are skipped entirely.

ii.
```python
EXCLUDE_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
filtered_clusters = [(q, t, tt) for q, t, tt in clusters if q not in EXCLUDE_QUALITIES]

# Low FR removal
mean_frs = np.mean(np.mean(trialdat[:, :, valid_idx], axis=2), axis=1)
keep_units = mean_frs > LOW_FR_THRESH
```

iii. The quality exclusion list matches `findClusters.m` exactly. The low FR threshold of 1 Hz matches `params.lowFR = 1` from the reference.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes are aligned to go cue onset by subtracting `goCue[trial]` from each spike's `trialtm`. This alignment happens inline during binning (not as a separate pre-alignment step like the reference code).

ii.
```python
gocue = sd.get_event_times('goCue')
# ...
aligned_times = spike_times[trial_mask] - gocue[j]
counts, _ = np.histogram(aligned_times, bins=time_edges)
```

iii. The reference code separates alignment (`alignSpikes.m`) and binning (`getSeq.m`), but the AI combines them in one step. The result is equivalent: `trialtm - goCue[trial]` is the same whether done beforehand or inline.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 10ms (`DT = 1/100`). Bins span from -2.5 to 2.5 seconds, yielding 500 time bins per trial. No rebinning is applied - this is the native binning.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1/100  # 10 ms
time_edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = time_edges[:-1] + DT / 2
```

iii. The reference code uses `params.dt = 1/100` (10ms) and the same time range. The comment in the reference says "use a 5 ms bin width" but the actual code uses `1/100 = 10ms`. The AI correctly followed the code rather than the comment.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not derived from raw data variables. It is the time axis itself - the center of each time bin relative to go cue onset.

ii.
```python
time_axis = time_edges[:-1] + DT / 2  # bin centers
# ...
input_data = time_axis.reshape(1, -1).copy()
```

iii. The instructions specify "Time from go cue onset in seconds" as a decoder input. The AI correctly generates this as the time axis of each bin center.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing is needed. The time axis is computed as bin centers from -2.495 to 2.495 seconds (500 bins of 10ms). It is the same for every trial.

ii.
```python
time_edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = time_edges[:-1] + DT / 2
# ...
input_data = time_axis.reshape(1, -1).copy()
```

iii. The time axis is deterministic from the binning parameters and identical across all trials.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input uses the same time axis as the neural data. Both use bin centers from the same `time_edges` array, so they are inherently aligned.

ii.
```python
# Same time_axis used for both neural and input
time_edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = time_edges[:-1] + DT / 2
# Neural: counts, _ = np.histogram(aligned_times, bins=time_edges)
# Input: input_data = time_axis.reshape(1, -1).copy()
```

iii. By construction, both share the same temporal grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `obj.bp.R` - a boolean array indicating right lick trials.

ii.
```python
R = sd.get_trial_array('R').astype(bool)
# ...
lick_dir = 1 if R[trial_idx] else 0
```

iii. The reference code uses `obj.bp.R` and `obj.bp.L` for right and left trials. The AI uses R=1 for right, else 0 for left, matching the instruction (left=0, right=1).

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. It is a binary per-trial variable: 1 if the trial has `R=True` (right lick), 0 otherwise (left lick). It is broadcast across all time bins for the trial.

ii.
```python
lick_dir = 1 if R[trial_idx] else 0
# ...
out[0, :] = t['lick_dir']  # broadcast to all time bins
```

iii. The instructions specify left=0, right=1 as a per-trial variable. The AI implements this correctly.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj.bp.autowater`. The reference code notes that `autowater=1` corresponds to water-cued (WC) blocks.

ii.
```python
autowater = sd.get_trial_array('autowater')
# ...
context = 0 if autowater[trial_idx] == 1 else 1
```

iii. The AI uses `autowater==1` as WC (0) and everything else as DR (1), matching the reference description in `WorkingWithDataObjs.m`.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. It is a binary per-trial variable: 0 for WC (`autowater==1`), 1 for DR (`autowater!=1`). It is broadcast across all time bins.

ii.
```python
context = 0 if autowater[trial_idx] == 1 else 1
# ...
out[1, :] = t['context']
```

iii. Matches the instructions: WC=0, DR=1.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit` - a boolean array indicating correct (hit) trials.

ii.
```python
hit = sd.get_trial_array('hit').astype(bool)
# ...
outcome = 1 if hit[trial_idx] else 0
```

iii. Since only hit and miss trials are included (from trial filtering), hit=1 means correct, and not-hit means miss (incorrect).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. It is a binary per-trial variable: 1 for correct (hit), 0 for incorrect (miss). Broadcast across all time bins.

ii.
```python
outcome = 1 if hit[trial_idx] else 0
out[2, :] = t['outcome']
```

iii. Matches the instructions: incorrect=0, correct=1.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from DLC (DeepLabCut) tracking data in `obj.traj{2}` (bottom camera). Specifically, the `top_tongue` feature's x,y coordinates are used. Frame times come from `obj.traj{2}(trial).frameTimes`. Video offset is computed from `obj.bp.ev.bitStart`, `obj.sglx.bitcode.bitstart`, and `obj.sglx.fs`.

ii.
```python
bottom_feats = sd.get_traj_feature_names(1)  # view=1 for bottom cam
# Find 'top_tongue' feature index
for idx, name in enumerate(bottom_feats):
    if name == 'top_tongue' and tongue_feat_idx is None:
        tongue_feat_idx = idx

ft, x, y = sd.get_trial_traj(1, trial_idx, tongue_feat_idx)
```

iii. The AI used bottom camera DLC tracking matching the reference code's approach for kinematic features.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Processing steps:
1. Get x,y coordinates from DLC tracking of `top_tongue` at 400 Hz
2. Compute velocity as `sqrt(dx^2 + dy^2) / dt` where `dt = 1/400`
3. Compute velocity timestamps at midpoints of position samples
4. Align to go cue: `vel_t - vidshift - goCue[trial]`
5. Interpolate to the 10ms neural time axis
6. NaN values for timepoints outside video coverage

ii.
```python
dt_vid = 1.0 / VIDEO_FPS  # 1/400
vel = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt_vid
vel_t = ft[:-1] + dt_vid / 2
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
tongue_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. The processing follows standard velocity computation from position data.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per session, the 50th percentile of all tongue velocity values across all valid trials is computed (ignoring NaNs). Values >= threshold are assigned class 1, values < threshold are assigned class 0. NaN values are assigned class 0.

ii.
```python
all_tongue = np.concatenate([t['tongue_vel'] for t in raw_outputs])
tongue_thresh = np.nanpercentile(all_tongue, 50)
# ...
tongue_disc = np.where(np.isnan(t['tongue_vel']), 0,
                      np.where(t['tongue_vel'] >= tongue_thresh, 1, 0)).astype(np.int64)
```

iii. Matches the instructions: discretized into two bins with per-session 50th percentile threshold. The NaN-to-0 assignment is a reasonable handling of missing data.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue velocity is aligned by subtracting `vidshift + goCue[trial]` from the video frame times, then interpolating to the neural time axis (bin centers).

ii.
```python
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
tongue_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. The alignment uses the same video offset computation as the reference code's `findVideoOffset.m`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from DLC tracking data in `obj.traj{2}` (bottom camera), specifically the `bottom_paw` feature's x,y coordinates.

ii.
```python
for idx, name in enumerate(bottom_feats):
    if name == 'bottom_paw' and paw_feat_idx is None:
        paw_feat_idx = idx

ft, x, y = sd.get_trial_traj(1, trial_idx, paw_feat_idx)
```

iii. Same source as tongue velocity but different tracked body part.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Identical to tongue velocity processing: compute velocity from x,y position differences at 400 Hz, align to go cue via video offset, interpolate to neural time axis.

ii.
```python
vel = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt_vid
vel_t = ft[:-1] + dt_vid / 2
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
paw_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. Same processing pipeline as tongue velocity.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Per session, the 50th percentile of all paw velocity values across all valid trials is computed. Values >= threshold are class 1, < threshold are class 0. NaN values are assigned class 0.

ii.
```python
all_paw = np.concatenate([t['paw_vel'] for t in raw_outputs])
paw_thresh = np.nanpercentile(all_paw, 50)
paw_disc = np.where(np.isnan(t['paw_vel']), 0,
                   np.where(t['paw_vel'] >= paw_thresh, 1, 0)).astype(np.int64)
```

iii. Same thresholding approach as tongue velocity, per the instructions.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same alignment as tongue velocity - subtract video offset and go cue time from frame times, then interpolate.

ii.
```python
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
paw_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. Uses the same video offset and alignment procedure.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from separate files `motionEnergy_{anm}_{date}.mat`. The data comes from `me.data` (cell array of per-trial motion energy time series) and the threshold from `me.moveThresh`. Frame times for alignment come from `obj.traj{1}` (side camera, view index 1 in the AI code which maps to bottom camera view index 1).

ii.
```python
def load_motion_energy(data_dir, anm, date):
    me_fn = f'motionEnergy_{anm}_{date}.mat'
    me_path = os.path.join(DATA_ROOT, data_dir, me_fn)
    me_mat = sio.loadmat(me_path, squeeze_me=False, struct_as_record=True)
    me_var = me_mat['me']
    # Handle struct format with data and moveThresh
    me_struct = me_var[0, 0]
    thresh = float(me_struct['moveThresh'].flatten()[0])
    data_field = me_struct['data']
```

iii. The AI correctly identified the separate motion energy files and their format.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Processing steps:
1. Load per-trial motion energy data from separate `.mat` files
2. Get frame times from the bottom camera trajectory data
3. Align frame times to go cue: `frameTimes - vidshift - goCue[trial]`
4. Interpolate motion energy to the neural time axis
5. NaN for timepoints outside video coverage

ii.
```python
ft = sd.get_trial_frame_times(1, trial_idx)  # bottom cam frame times
ft_aligned = ft - vidshift - gocue[trial_idx]
me_interp = np.interp(time_axis, ft_aligned, me_data[trial_idx], left=np.nan, right=np.nan)
```

iii. The reference `loadMotionEnergy.m` uses `interp1(obj.traj{1}(trix).frameTimes-vidshift-alignTimes(trix), me.data{trix}, taxis)`. The AI replicates this approach, though there's a question about which camera view's frame times to use.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per session, the 50th percentile of all motion energy values is computed. Values >= threshold are class 1, < threshold are class 0. NaN values are assigned class 0. Note: the AI computes its own 50th percentile threshold and does NOT use the `moveThresh` from the `.mat` file.

ii.
```python
all_me = np.concatenate([t['motion_energy'] for t in raw_outputs])
me_thresh_50 = np.nanpercentile(all_me, 50)
me_disc = np.where(np.isnan(t['motion_energy']), 0,
                  np.where(t['motion_energy'] >= me_thresh_50, 1, 0)).astype(np.int64)
```

iii. The instructions specify 50th percentile per-session threshold, which the AI follows. The `moveThresh` from the data is loaded but not used for the final discretization.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by using the bottom camera frame times, subtracting video offset and go cue time, then interpolating to the neural time axis.

ii.
```python
ft = sd.get_trial_frame_times(1, trial_idx)
ft_aligned = ft - vidshift - gocue[trial_idx]
me_interp = np.interp(time_axis, ft_aligned, me_data[trial_idx], left=np.nan, right=np.nan)
```

iii. The reference code uses `obj.traj{1}` (side cam in MATLAB 1-indexed) frame times for motion energy alignment. The AI uses view index 1, which in its 0-indexed scheme corresponds to the bottom camera. This is a discrepancy from the reference code.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple fallback strategies:
1. Missing `.mat` files: session is skipped with a warning
2. Missing cluster data: session is skipped
3. Too few valid trials (<5): session is skipped
4. Too few units after filtering (<10): session is skipped
5. Missing trajectory data: try/except catches errors, fills with NaN
6. Missing motion energy files: returns None, fills with NaN
7. NaN values in velocity/ME: assigned to class 0 during discretization
8. All-zero neural data for some trials: allowed through (noted in verification warnings)

ii.
```python
if not os.path.exists(fpath):
    print(f'  WARNING: File not found: {fpath}')
    return None

# Tongue velocity with fallback
tongue_vel = np.full(n_timebins, np.nan)
if tongue_feat_idx is not None:
    try:
        # ... compute velocity ...
    except:
        pass

# NaN handling in discretization
tongue_disc = np.where(np.isnan(t['tongue_vel']), 0,
                      np.where(t['tongue_vel'] >= tongue_thresh, 1, 0))
```

iii. The AI documented known issues in CONVERSION_NOTES.md including all-zero neural data in some late trials and sessions with unavailable motion energy.

## 11-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is spike binning and smoothing in `process_session()`. For each session, it loops over every cluster and every trial (including invalid ones), computing histograms and applying Gaussian smoothing. With ~50-140 units and ~150-470 trials per session across 45 sessions, this is O(n_units * n_trials * n_timebins) per session. Video trajectory loading and interpolation is the second most expensive step.

ii.
```python
# Most expensive loop - all units x all trials
trialdat = np.zeros((n_units, n_timebins, ntrials))
for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
    for j in range(ntrials):
        trial_mask = (trial_nums == (j + 1))
        aligned_times = spike_times[trial_mask] - gocue[j]
        counts, _ = np.histogram(aligned_times, bins=time_edges)
        fr = counts / DT
        trialdat[i, :, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)
```

iii. The nested loop over units and trials with per-element smoothing is inherently slow in Python.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
1. **Smoothing**: The `causal_gaussian_smooth` function already operates on columns but is called per-trial. Could smooth the entire `(n_timebins, n_trials)` matrix at once per unit.
2. **Spike binning**: The inner loop over trials could potentially use vectorized sparse-matrix approaches.
3. **Velocity computation**: The per-trial trajectory processing could batch multiple trials.

ii.
```python
# Current: per-trial smoothing
for j in range(ntrials):
    trialdat[i, :, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)

# Could be: smooth all trials at once
# trialdat[i, :, :] = causal_gaussian_smooth(fr_all_trials, SMOOTH_WINDOW, BC_TYPE)
```

iii. The smoothing function already supports multi-column input but is called on individual columns.

## 11-c. What processing does the code repeat multiple times?

i.
1. The video offset (`get_video_offset()`) is computed once per session, which is correct.
2. Feature name lookup (`get_traj_feature_names()`) is done once per session.
3. Smoothing is applied independently to each unit-trial combination rather than batched.
4. The `trial_mask = (trial_nums == (j + 1))` search is repeated for every trial for every unit - this could be precomputed once.

ii.
```python
# Repeated per unit per trial
for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
    for j in range(ntrials):
        trial_mask = (trial_nums == (j + 1))  # recomputed for each unit
```

iii. The trial mask computation is redundant across units since `trial_nums` differs per unit but the pattern `(trial_nums == j+1)` is recomputed.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
1. **Spike binning for invalid trials**: The code bins spikes for ALL trials (`for j in range(ntrials)`) but only valid trials (hit|miss, no stim, no early) are used in the output. This wastes computation on trials that are discarded.
2. **Computing PSTH-equivalent data**: The reference code computes PSTHs (`obj.psth`) for multiple conditions, but the AI doesn't compute separate PSTHs - only single-trial data is needed for the decoder.
3. **Loading `moveThresh` from motion energy files**: This threshold is loaded but never used; the 50th percentile is computed fresh.

ii.
```python
# Bins ALL trials, but only valid_idx are used
trialdat = np.zeros((n_units, n_timebins, ntrials))
for j in range(ntrials):  # loops over ALL trials
    # ... binning ...

# Only valid trials used in output
for trial_idx in valid_idx:
    neural = trialdat[:, :, trial_idx]
```

iii. Binning all trials matches the reference code behavior (which also bins all trials for `trialdat`), but for this decoder task, only valid trials are needed. The reference code needs all trials for the PSTH computation.
