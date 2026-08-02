# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hardcodes a list of 44 sessions in `SESSION_META` (lines 26-74), each specifying animal name, date, probe number, data directory, and task type. For each session, it constructs the file path `data/<dir>/data_structure_<anm>_<date>.mat` and loads it using either h5py (MATLAB v7.3/HDF5) or scipy.io.loadmat (MATLAB v5), auto-detecting the format. Motion energy is loaded from separate `motionEnergy_<session_id>.mat` files.

ii.
```python
SESSION_META = [
    {'anm': 'EKH1', 'date': '2021-08-07', 'probe': 2, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    # ... 43 more entries
]

def load_session(filepath, probe_num):
    try:
        f = h5py.File(filepath, 'r')
        f.close()
        return load_session_h5(filepath, probe_num)
    except:
        return load_session_v5(filepath, probe_num)
```

iii. The AI documented in CONVERSION_NOTES.md that it identified the session list from the loading scripts in `code/DataLoadingScripts/Recording and video/` and handles both MATLAB file formats found in the data.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `anm` field in `SESSION_META`. As sessions are processed, unique animal names are collected into the `all_subjects` list, and `subject_idx` maps each session to its subject index.

ii.
```python
subj = result['subject']
if subj not in all_subjects:
    all_subjects.append(subj)
subject_idx.append(all_subjects.index(subj))
```

iii. The AI noted 14 unique animals (10 DR + 4 RandDelay) from the data files, while the paper reports 9 DR mice. The AI chose to include all animals found in the data/loading scripts.

## 1-c. How are the data split into sessions?

i. Each entry in `SESSION_META` represents one session, defined by a unique combination of animal and date. Sessions are processed sequentially, with each session's data stored as a separate entry in the output lists.

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

iii. The AI identified 25 DR sessions and 19 RandDelay sessions from the loading scripts, totaling 44 sessions, matching the paper's reported numbers.

## 1-d. How are the data split into trials?

i. Within each session, the number of trials (`ntrials`) is read from `obj.bp.Ntrials`. Spike data is organized per-trial using the `trial` field of each cluster, and behavioral variables (`hit`, `miss`, `R`, `L`, etc.) are stored per-trial. Only valid trials (after filtering) are included in the output.

ii.
```python
ntrials = session_data['ntrials']
# ...
for trial_idx in range(ntrials):
    trial_num = trial_idx + 1  # 1-indexed
    spike_mask = trial_nums == trial_num
```

iii. The AI loads all trials from the data files and then applies quality filters to select valid trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded if they are: early lick (`early`), no response (`no`), stimulation enabled (`stim_enable`), or neither hit nor miss. This keeps only hit and miss trials without stimulation or behavioral violations.

ii.
```python
valid_trial_mask = ~session_data['early'] & ~session_data['no'] & ~session_data['stim_enable']
valid_trial_mask = valid_trial_mask & (session_data['hit'] | session_data['miss'])
```

iii. The AI documented this decision in CONVERSION_NOTES.md Step 5: "Trial filtering: Exclude early, no-response, stim trials; keep hit+miss." Including miss trials is necessary because the instructions require decoding "Outcome" (incorrect=0, correct=1).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `obj.clu{probe}(i).trialtm` (spike times relative to trial start) and `obj.clu{probe}(i).trial` (trial assignment for each spike), along with `obj.bp.ev.goCue` for temporal alignment.

ii.
```python
trialtm = clu['trialtm']
trial_nums = clu['trial']
aligned_times = trialtm[spike_mask] - goCue[trial_idx]
```

iii. The AI identified these variables from the reference code's `alignSpikes.m` and `getSeq.m` functions.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue, binned into 5ms bins over [-2.5, 2.5]s, converted to firing rates (spikes/s), and smoothed with a causal Gaussian kernel (window size 15).

ii.
```python
edges = np.arange(tmin, tmax + dt, dt)
counts, _ = np.histogram(aligned_times, bins=edges)
fr = counts.astype(np.float32) / dt
fr_smooth = smooth_causal(fr, params['smooth_window'], bctype='none')
```

iii. The AI documented matching `getSeq.m` for binning (`histc` equivalent) and `mySmooth.m` for smoothing (causal Gaussian kernel with first floor(N/2) elements zeroed).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering: (1) Cluster quality filter excludes clusters labeled 'garbage', 'gabrga', 'noisy', or 'real?'. (2) Low firing rate filter removes clusters with mean firing rate <= 0.5 Hz. Sessions with fewer than 10 remaining units are skipped.

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

iii. The AI matched `findClusters.m` for quality exclusions and `removeLowFRClusters.m` for the FR threshold. However, the AI noted a discrepancy: the paper mentions 1 Hz for some analyses while the code uses 0.5 Hz; the AI chose to follow the code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue by subtracting `goCue[trial_idx]` from each spike's `trialtm`. This matches the reference `alignSpikes.m` logic: `trialtm_aligned = trialtm - event`.

ii.
```python
aligned_times = trialtm[spike_mask] - goCue[trial_idx]
```

iii. The AI documented that alignment follows `alignSpikes.m` with `params.alignEvent = 'goCue'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 5ms (dt = 1/200). No temporal rebinning is applied - spikes are binned directly at 5ms resolution matching `getDefaultParams.m`. The time window spans [-2.5, 2.5]s, yielding 1000 time bins.

ii.
```python
PARAMS = {
    'dt': 1.0 / 200.0,  # 5ms bins
    'tmin': -2.5,
    'tmax': 2.5,
}
```

iii. The AI confirmed this matches `getDefaultParams.m`: `params.dt = 1/200`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is derived from the time axis itself, which is constructed from the binning parameters (tmin, tmax, dt). It is a deterministic variable based on the temporal alignment to the go cue.

ii.
```python
edges = np.arange(tmin, tmax + dt, dt)
time_axis = edges[:-1] + dt / 2
time_input = time_axis.reshape(1, -1).astype(np.float32)
```

iii. The AI specified in the mapping plan that input[0] is "Time from goCue in seconds" as required by the instructions.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is computed as the centers of the 5ms bins: `edges[:-1] + dt/2`. This produces values from approximately -2.4975 to 2.4975 seconds. The same time vector is used for every trial.

ii.
```python
time_input = time_axis.reshape(1, -1).astype(np.float32)
input_trials.append(time_input)
```

iii. No special processing needed; this is a deterministic time axis matching the neural data bins.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time input is inherently aligned with the neural data because both use the same time axis (derived from the same bin edges). Each time point in the input corresponds directly to the same time bin in the neural data.

ii.
```python
# Same time_axis used for both neural binning and input
time_input = time_axis.reshape(1, -1).astype(np.float32)
```

iii. Alignment is trivial since both use the same time axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `obj.bp.R`, a boolean array indicating whether each trial was a right-lick trial.

ii.
```python
data['R'] = bp['R'][0, :].astype(bool)
# ...
lick_direction = session_data['R'][valid_trial_indices].astype(int)
```

iii. The AI mapped R=1 (right) and L=0 (left) as specified in the instructions.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The boolean `R` field is cast to int: True (right) becomes 1, False (left) becomes 0. This is a per-trial value broadcast across all time bins.

ii.
```python
lick_direction = session_data['R'][valid_trial_indices].astype(int)
output[0, :] = lick_direction[i]  # per-trial, broadcast
```

iii. Matches the instruction specification: left=0, right=1.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj.bp.autowater`, which indicates whether autowater (water-cued task) is enabled for each trial.

ii.
```python
data['autowater'] = bp['autowater'][0, :]
# ...
context = (session_data['autowater'][valid_trial_indices] == 0).astype(int)
```

iii. The AI documented in CONVERSION_NOTES.md that DR context has `~autowater` (autowater=0) and WC context has `autowater` (autowater nonzero), matching the reference code conditions.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. When `autowater == 0` (DR task), context = 1. When `autowater != 0` (WC task), context = 0. This is a per-trial value broadcast across all time bins.

ii.
```python
context = (session_data['autowater'][valid_trial_indices] == 0).astype(int)
output[1, :] = context[i]
```

iii. Matches instructions: WC=0, DR=1. The reference code's condition strings use `~autowater` for DR and `autowater` for WC.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit`, a boolean array indicating whether each trial was a correct (hit) trial.

ii.
```python
data['hit'] = bp['hit'][0, :].astype(bool)
# ...
outcome = session_data['hit'][valid_trial_indices].astype(int)
```

iii. The AI mapped hit=1 (correct) and miss=0 (incorrect).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The boolean `hit` field is cast to int: True (correct) becomes 1, False (incorrect) becomes 0. This is a per-trial value broadcast across all time bins.

ii.
```python
outcome = session_data['hit'][valid_trial_indices].astype(int)
output[2, :] = outcome[i]
```

iii. Matches instructions: incorrect=0, correct=1.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from DeepLabCut tracking data in `obj.traj{1}` (side camera, camera 0), specifically the 'tongue' feature's x,y position time series (`ts` field), aligned using `frameTimes` and corrected by the video offset.

ii.
```python
cam0 = session_data['traj'][0]  # side cam
feat_names = cam0['featNames']
for fi, fn in enumerate(feat_names):
    if fn.lower() == 'tongue':
        tongue_idx = fi
        break
# ...
x = ts[tongue_idx, 0, :]
y = ts[tongue_idx, 1, :]
```

iii. The AI identified the tongue feature from the side camera DLC tracking data, matching the reference `findPosition.m` and `findVelocity.m` functions.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Processing steps: (1) Extract tongue x,y from DLC tracking. (2) Align frame times using video offset and go cue. (3) Interpolate to neural time axis. (4) Fill NaN positions with session mean. (5) Compute velocity as gradient. (6) Set velocity to 0 where tongue was not visible (NaN in original). (7) Compute speed as sqrt(xvel^2 + yvel^2).

ii.
```python
aligned_ft = ft[:len(x)] - vidshift - goCue[trial_idx]
fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
all_x[:, trial_idx] = fx(time_axis)
# Fill NaN with session mean
all_x_filled[np.isnan(all_x_filled)] = mean_x
# Compute velocity
xvel = np.gradient(all_x_filled[:, trial_idx])
yvel = np.gradient(all_y_filled[:, trial_idx])
nan_mask = np.isnan(all_x[:, trial_idx])
xvel[nan_mask] = 0.0
yvel[nan_mask] = 0.0
speed = np.sqrt(xvel**2 + yvel**2)
```

iii. The AI noted following `findPosition.m` and `findVelocity.m`. However, the reference code does NOT fill NaN positions with session mean for tongue - it leaves them as NaN and only sets the velocity to 0. The AI's approach of filling with session mean before computing gradient could affect values near NaN boundaries.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per-session 50th percentile threshold applied to all tongue velocity values across valid trials. Values >= threshold get category 1, values < threshold get category 0. Special handling when threshold equals minimum (uses strict > to avoid all-1 output).

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

iii. Matches the instructions: discretize into two bins using per-session 50th percentile threshold.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue positions are interpolated from video frame times to the neural time axis using `interp1d`. Frame times are corrected by subtracting the video offset and go cue time: `frameTimes - vidshift - goCue[trial_idx]`.

ii.
```python
aligned_ft = ft[:len(x)] - vidshift - goCue[trial_idx]
fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
all_x[:, trial_idx] = fx(time_axis)
```

iii. Matches `findPosition.m`: `interp1(traj(trix).frameTimes - vidshift - obj.bp.ev.(alignEv)(trix), ts, taxis)`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from DLC tracking data in `obj.traj{2}` (top/bottom camera, camera 1), specifically the 'top_paw' feature's x,y positions.

ii.
```python
cam1 = session_data['traj'][1]  # top cam
for fi, fn in enumerate(feat_names):
    if 'top_paw' in fn.lower():
        paw_idx = fi
        break
```

iii. The AI identified `top_paw` from the second camera, matching the reference `params.traj_features` which includes `'top_paw'` for the second camera.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Processing steps: (1) Extract paw x,y from DLC. (2) Align and interpolate to neural time axis with video offset correction. (3) Fill missing values with nearest. (4) Compute velocity with gradient. (5) Subtract baseline derivative (median of diff). (6) Fill missing velocity with nearest. (7) Compute speed as sqrt(xvel^2 + yvel^2).

ii.
```python
fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
x_interp = fx(time_axis)
# Fill missing
arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans])
# Velocity with baseline subtraction
xvel = np.gradient(x_interp)
basederiv_x = np.nanmedian(np.diff(x_interp))
basederiv_y = np.nanmedian(np.diff(y_interp))
xvel -= basederiv_x
yvel -= basederiv_y
speed = np.sqrt(xvel**2 + yvel**2)
```

iii. The AI followed `findPosition.m` (interpolation, fill missing) and `findVelocity.m` (gradient, baseline subtraction). Note: the reference code has a bug where both xvel and yvel subtract `basederiv(1)` (x component); the AI correctly subtracts the respective components. Also, the reference smooths non-tongue positions with `mySmooth(ts, 1, 'reflect')` before interpolation, but with N=1 this is a no-op.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue velocity: per-session 50th percentile threshold with the same `discretize_velocity` function.

ii.
```python
paw_disc, paw_thresh = discretize_velocity(valid_paw_vel)
```

iii. Matches instructions: 50th percentile threshold per session.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same alignment as tongue velocity: frame times corrected by video offset and go cue time, then interpolated to neural time axis.

ii.
```python
aligned_ft = ft[:len(x)] - vidshift - goCue[trial_idx]
fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
```

iii. Matches reference `findPosition.m` alignment approach.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from separate `motionEnergy_<session_id>.mat` files containing `me.data` (cell array of per-trial ME time series) and `me.moveThresh` (per-session movement threshold).

ii.
```python
me_path = os.path.join('data', data_dir, f'motionEnergy_{session_id}.mat')
me = load_motion_energy(me_path)
# Returns {'data': trials_list, 'moveThresh': threshold}
```

iii. The AI identified the motion energy files and documented handling of 3 different ME file formats found in the data.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Processing steps: (1) Load per-trial ME data from .mat file. (2) Align using side camera frame times, video offset, and go cue. (3) Interpolate to neural time axis. (4) Fill NaN with nearest value interpolation.

ii.
```python
aligned_ft = ft - vidshift - goCue[trial_idx]
f_interp = interp1d(aligned_ft, trial_me, kind='linear', bounds_error=False, fill_value=np.nan)
me_aligned[:, trial_idx] = f_interp(time_axis)
# Fill NaN with nearest
col[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), col[not_nans])
```

iii. Matches reference `loadMotionEnergy.m`: `interp1(obj.traj{1}(trix).frameTimes - vidshift - alignTimes(trix), me.data{trix}, taxis)` followed by `fillmissing(...,'nearest')`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same as tongue/paw velocity: per-session 50th percentile threshold using the `discretize_velocity` function.

ii.
```python
me_disc, me_thresh = discretize_velocity(valid_me)
```

iii. The instructions specify 50th percentile threshold. The reference code uses `me.moveThresh` (a manually-set per-session threshold based on bimodal distribution), which differs from the 50th percentile. The AI followed the instructions rather than the reference code.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. ME is aligned using the side camera's frame times, corrected by video offset and go cue time, then interpolated to the neural time axis.

ii.
```python
aligned_ft = ft - vidshift - goCue[trial_idx]
f_interp = interp1d(aligned_ft, trial_me, kind='linear', bounds_error=False, fill_value=np.nan)
me_aligned[:, trial_idx] = f_interp(time_axis)
```

iii. Matches `loadMotionEnergy.m` alignment.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) Missing video data (NaN dropped frames) causes trial to be skipped for kinematic processing. (2) NaN tongue positions are filled with session mean, then velocity set to 0 where tongue was invisible. (3) NaN paw positions filled with nearest. (4) NaN motion energy filled with nearest interpolation. (5) Trials with all-zero neural data are removed. (6) Empty quality strings in cluster data are kept (treated as unlabeled). (7) The code handles 3 different motion energy file formats. (8) Sessions with fewer than 10 units or fewer than 2 valid trials are skipped.

ii.
```python
# Missing video data
if np.isnan(trial_info['NdroppedFrames']):
    continue

# NaN tongue positions
all_x_filled[np.isnan(all_x_filled)] = mean_x

# Trials with all-zero neural
if np.all(ni == 0):
    n_removed += 1
    continue

# Empty cluster quality
if q in excluded:
    continue
valid_clusters.append(clu)
```

iii. The AI documented edge cases in CONVERSION_NOTES.md Step 10, including handling of JEB15 sessions with null quality strings and JEB6 with dual probes.

## 11-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading MATLAB files, especially large HDF5 files with cluster and trajectory data. (2) The nested loop over neurons and trials for spike binning and smoothing. (3) Processing DLC trajectory data for tongue and paw velocity (interpolation per trial).

ii.
```python
t_load_start = time.time()
session_data = load_session(data_path, probe)
print(f"    Data loaded in {time.time()-t_load_start:.1f}s")
# ...
for neuron_idx, clu in enumerate(valid_clusters):
    for trial_idx in range(ntrials):
        counts, _ = np.histogram(aligned_times, bins=edges)
        fr_smooth = smooth_causal(fr, params['smooth_window'], bctype='none')
```

iii. The AI noted ~6-8s per session, with ~4.4 minutes estimated for the full dataset (44 sessions).

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The main candidate for vectorization is the nested neuron-trial loop for spike binning (lines 556-575). Instead of looping over each neuron and trial separately, spike times could be pre-grouped and binned using vectorized operations. The smoothing loop in `smooth_causal` (line 131) iterates over columns individually. The tongue/paw velocity computation loops over trials individually.

ii.
```python
for neuron_idx, clu in enumerate(valid_clusters):
    for trial_idx in range(ntrials):
        spike_mask = trial_nums == trial_num
        aligned_times = trialtm[spike_mask] - goCue[trial_idx]
        counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. The AI mentioned efficiency improvements in CONVERSION_NOTES.md but did not implement significant vectorization.

## 11-c. What processing does the code repeat multiple times?

i. The code does not significantly repeat processing. Each session is processed once. However, the `spike_mask = trial_nums == trial_num` check inside the inner loop creates a boolean mask for every trial for every neuron, which is redundant - the trial assignments could be pre-grouped once per neuron using `np.searchsorted` or dictionary grouping.

ii.
```python
for neuron_idx, clu in enumerate(valid_clusters):
    trialtm = clu['trialtm']
    trial_nums = clu['trial']
    for trial_idx in range(ntrials):
        spike_mask = trial_nums == trial_num  # repeated for every trial
```

iii. Not explicitly documented by the AI.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes neural data (`trialdat`) for ALL trials before applying trial filtering, meaning spike binning and smoothing is performed for trials that are subsequently excluded (early, no-response, stim trials). Similarly, tongue velocity, paw velocity, and motion energy are computed for all trials before filtering to valid trials only. The code also removes trials with all-zero neural data after all processing is complete.

ii.
```python
# Neural data computed for ALL trials
trialdat = np.zeros((n_timepts, n_neurons, ntrials), dtype=np.float32)
for neuron_idx, clu in enumerate(valid_clusters):
    for trial_idx in range(ntrials):
        # ... processing all trials

# But only valid trials are used
valid_trial_indices = np.where(valid_trial_mask)[0]
valid_me = me_aligned[:, valid_trial_indices]
```

iii. The AI did not explicitly document this inefficiency, but it follows the same pattern as the reference code (`getSeq.m` processes all trials in `trialdat`).
