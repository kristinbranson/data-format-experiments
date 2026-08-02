# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from HDF5 (.mat v7.3) files in the `data/Ephys_Behavior` directory. Each session has a `data_structure_{animal}_{date}.mat` file and a separate `motionEnergy_{animal}_{date}.mat` file. A hardcoded `SESSION_META` list defines all 25 sessions with (animal, date, probe_numbers). The code iterates through this list, calling `process_session()` for each. Files are opened with `h5py.File()`.

ii.
```python
SESSION_META = [
    ('EKH1', '2021-08-07', [1]),
    ('EKH3', '2021-08-11', [1, 2]),
    ('JEB6', '2021-04-18', [2]),
    ...
]
DATA_DIR = 'data/Ephys_Behavior'

for idx, (animal, date, probes) in enumerate(sessions):
    result = process_session(animal, date, probes, ...)

def process_session(animal, date, probes, ...):
    data_file = os.path.join(DATA_DIR, f'data_structure_{session_id}.mat')
    me_file = os.path.join(DATA_DIR, f'motionEnergy_{session_id}.mat')
    f = h5py.File(data_file, 'r')
```

iii. The AI identified 25 sessions from the `loadXXX_ALMVideo.m` loader scripts in the reference code. The AI documented this in CONVERSION_NOTES.md under the probe assignments table and Step 1 notes.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `animal` field in `SESSION_META`. The AI extracts unique animal names from all processed sessions and creates a sorted list of subjects. Each session maps to its subject via `subjects.index(result['animal'])`.

ii.
```python
subjects = sorted(list(set(r['animal'] for r in all_results)))
subj_idx = subjects.index(result['animal'])
subject_idx_list.append(subj_idx)
```

iii. The AI noted 10 subjects in the data vs 9 reported in the paper, and decided to include all 10. Documented in CONVERSION_NOTES.md Step 4 discrepancies.

## 1-c. How are the data split into sessions?

i. Each entry in `SESSION_META` defines one session as a unique (animal, date) combination. Each session is processed independently by `process_session()`. For multi-probe sessions, probes are concatenated within the session.

ii.
```python
for idx, (animal, date, probes) in enumerate(sessions):
    result = process_session(animal, date, probes, ...)
    if result is not None:
        all_results.append(result)

# For multi-probe, concatenate across probes within session:
trialdat = np.concatenate(all_trialdat, axis=1)  # (time, neurons, trials)
```

iii. The AI followed the reference code's loadSessionData.m pattern where each session has a separate data file and probes within a session are concatenated.

## 1-d. How are the data split into trials?

i. Trials within a session are indexed by their position in the behavioral data arrays (0-indexed). The total number of trials comes from `bp['Ntrials']`. Only trials passing quality filters are included in the final output.

ii.
```python
Ntrials = int(bp['Ntrials'][0, 0])
valid_mask = (early == 0) & (no == 0) & (stim_enable == 0) & ((hit == 1) | (miss == 1))
valid_trials = np.where(valid_mask)[0]

for trial_idx in valid_trials:
    neural_trials.append(trialdat[:, :, trial_idx].T.astype(np.float32))
```

iii. The AI documented trial filtering criteria in CONVERSION_NOTES.md Step 5.

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded if they have: early licks (`early == 1`), no response / ignore (`no == 1`), or stimulation enabled (`stim.enable == 1`). Only hit and miss trials are kept. Sessions with fewer than 2 valid trials are skipped entirely.

ii.
```python
valid_mask = (early == 0) & (no == 0) & (stim_enable == 0) & ((hit == 1) | (miss == 1))
valid_trials = np.where(valid_mask)[0]

if len(valid_trials) < 2:
    print(f'    Skipping {session_id}: only {len(valid_trials)} valid trials')
    f.close()
    return None
```

iii. The AI's justification: "Exclude early, no, stim trials: Per paper methods" (CONVERSION_NOTES Step 5). The paper states "excluding early lick and ignore trials, which were omitted from all analyses." Including miss (error) trials is necessary for the decoder to predict outcome. The reference code conditions for PSTHs only use hit trials, but trialdat is computed for all Ntrials, and the decoder task requires miss trials to decode outcome.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from spike times stored in `obj.clu{probe}` structures. Specifically, for each cluster: `clu.trialtm` (spike times relative to trial start), `clu.trial` (trial assignments), and `clu.tm` (absolute spike times). The cluster quality strings (`clu.quality`) are used for filtering.

ii.
```python
clu_ref = f['obj']['clu'][probe_idx, 0]
clu_group = f[clu_ref]

tm_ref = clu_group['trialtm'][clu_idx, 0]
trial_ref = clu_group['trial'][clu_idx, 0]
spike_trialtm = f[tm_ref][:].flatten()
spike_trial = f[trial_ref][:].flatten().astype(int)
```

iii. The AI identified these as the spike time variables from the reference code's `alignSpikes.m` and `getSeq.m` functions.

## 2-b. How is the `neural` data processed?

i. Processing pipeline: (1) Spike times are aligned to the go cue event. (2) Aligned spikes are binned into 10ms time bins from -2.5s to +2.5s. (3) Bin counts are converted to firing rates (spikes/sec) by dividing by dt. (4) Firing rates are smoothed with a causal Gaussian kernel (window=15 samples, boundary=reflect). (5) Low firing rate neurons (mean FR <= 1 Hz) are removed.

ii.
```python
# Align to goCue
spike_goCue = goCue[spike_trial_valid - 1]
spike_aligned = spike_trialtm_valid - spike_goCue

# Bin and smooth
counts, _ = np.histogram(trial_spikes, bins=EDGES)
fr = counts / PARAMS['dt']  # spks/sec
fr_smooth = causal_gaussian_smooth(fr, PARAMS['smooth'], PARAMS['bctype'])

# Causal Gaussian smoothing
kern = sig_windows.gaussian(N, std=(N-1)/(2*2.5))
kern[:N//2] = 0  # causal
kern = kern / kern.sum()
```

iii. The AI documented following `getSeq.m` for binning/smoothing and `mySmooth.m` for the causal Gaussian kernel implementation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering: (1) Cluster quality filter: exclude clusters labeled 'garbage', 'gabrga', 'noisy', or 'real?' (matching `findClusters.m` with quality='all'). Clusters with empty/null quality strings are included. (2) Low firing rate filter: remove neurons with mean FR <= 1 Hz across all trials and time bins. Sessions with fewer than 10 neurons after filtering are skipped.

ii.
```python
PARAMS['quality_exclude'] = ['garbage', 'gabrga', 'noisy', 'real?']
PARAMS['lowFR'] = 1.0

# Quality filter
cluid = find_clusters(qualities, PARAMS['quality_exclude'])

# FR filter
meanFRs = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
fr_mask = meanFRs > PARAMS['lowFR']
trialdat = trialdat[:, fr_mask, :]

# Min neuron threshold
if n_neurons < 10:
    return None
```

iii. CONVERSION_NOTES Step 3: "Quality filter: exclude garbage, gabrga, noisy, real? clusters" and "Low FR threshold: 1 Hz" based on paper statement "firing rates exceeding 1 Hz."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue onset. For each spike, the go cue time of its trial is subtracted from the spike time relative to trial start: `aligned_time = trialtm - goCue(trial)`. This aligned time is then binned.

ii.
```python
spike_goCue = goCue[spike_trial_valid - 1]
spike_aligned = spike_trialtm_valid - spike_goCue
counts, _ = np.histogram(trial_spikes, bins=EDGES)
```

iii. The AI followed the instruction to align to "Go cue onset" and the reference code's `alignSpikes.m` which computes `trialtm_aligned = trialtm - event`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 10ms (`dt = 1/100`). Time bins span from -2.5s to +2.5s relative to go cue, yielding 500 time bins. No rebinning is applied; spikes are directly binned at this resolution.

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

iii. The AI chose 10ms based on its claim that `params.dt = 1/100` is used in "most scripts." However, the reference `getDefaultParams.m` specifies `params.dt = 1/200` (5ms).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not derived from any raw data variable. It is constructed as the time axis itself (bin centers of the spike histograms), which represents seconds relative to go cue onset.

ii.
```python
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
time_input = TIME_AXIS.astype(np.float32).reshape(1, -1)
session_inputs.append(time_input)
```

iii. The AI constructed this as the time axis with values from approximately -2.495 to 2.495 in 10ms steps, consistent with the decoder task specification of "time from go cue onset in seconds."

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing is needed. The time axis is computed directly as bin centers from -2.5 to 2.5 seconds at 10ms resolution.

ii.
```python
EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
time_input = TIME_AXIS.astype(np.float32).reshape(1, -1)
```

iii. Straightforward construction from the parameters; no justification needed.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis is identical to the neural time axis, since both use the same TIME_AXIS array. Neural data is binned using the same EDGES, so the time axis perfectly aligns with the neural data.

ii.
```python
# Neural binning uses:
counts, _ = np.histogram(trial_spikes, bins=EDGES)
# Input uses:
time_input = TIME_AXIS.astype(np.float32).reshape(1, -1)
# Both use the same EDGES/TIME_AXIS
```

iii. No explicit justification needed; the alignment is trivially correct since the same time axis is shared.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `obj.bp.R`, which is a binary array indicating right-lick trials (1 = right, 0 = left).

ii.
```python
R = bp['R'][:].flatten()
lick_direction = R.copy()  # 1=right, 0=left
```

iii. The AI maps R directly: left=0, right=1, matching the instruction specification.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. No processing beyond copying the R variable. The value is broadcast to all time bins as a per-trial constant.

ii.
```python
lick_direction = R.copy()
# In output construction:
lick_dir = np.full(N_TIMEBINS, info['lick_direction'], dtype=np.int64)
```

iii. The AI treats this as a per-trial variable broadcast to time-varying shape for the decoder format.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj.bp.autowater`. When `autowater == 1`, the trial is a water-cued (WC) trial; when `autowater == 0`, it is a delayed-response (DR) trial.

ii.
```python
autowater = bp['autowater'][:].flatten()
context = 1 - autowater  # DR=1, WC=0
```

iii. The AI correctly inverts autowater so that WC=0 and DR=1, matching the instruction specification.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Simple inversion: `context = 1 - autowater`. Broadcast to all time bins as a per-trial constant.

ii.
```python
context = 1 - autowater
context_val = np.full(N_TIMEBINS, info['context'], dtype=np.int64)
```

iii. Direct mapping from autowater flag.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit`, a binary array where 1 = correct (hit) and 0 = incorrect (miss).

ii.
```python
hit = bp['hit'][:].flatten()
outcome = hit.copy()  # 1=correct, 0=incorrect
```

iii. The AI uses hit directly, matching the instruction's incorrect=0, correct=1 encoding.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. No processing beyond copying the hit variable. Broadcast to all time bins.

ii.
```python
outcome = hit.copy()
outcome_val = np.full(N_TIMEBINS, info['outcome'], dtype=np.int64)
```

iii. Direct mapping; no additional processing needed.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from DLC (DeepLabCut) trajectory data in `obj.traj{1}` (side view / view 1). The 'tongue' feature's x and y positions are extracted from `traj.ts`, with timing from `traj.frameTimes`.

ii.
```python
traj_ref = f['obj']['traj'][0, 0]  # view 1 (side)
# Extract tongue feature
ts_data = f[ts_ref][:]
x = ts_data[tongue_idx, 0, :]
y = ts_data[tongue_idx, 1, :]
# Frame times for alignment
ft_ref = traj_group['frameTimes'][trix, 0]
frameTimes = f[ft_ref][:].flatten()
```

iii. The AI identified tongue as coming from the side view (view 1, index 0 in Python) based on the reference code's `findPosition.m` and `getDefaultParams.m` feature lists.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Processing steps: (1) Extract x, y positions of tongue for each trial. (2) Align to neural time axis by subtracting video offset and go cue time from frame times. (3) Interpolate positions to the neural time axis using `interp1d`. (4) Compute velocity as gradient of position. (5) Set NaN velocities to 0 (tongue not visible). (6) Compute speed as `sqrt(xvel^2 + yvel^2)`.

ii.
```python
aligned_times = frameTimes - vidshift - goCue[trix]
fx = interp1d(aligned_times[valid], x[valid], bounds_error=False, fill_value=np.nan)
xpos[:, trix] = fx(taxis)

xv = np.gradient(xpos[:, trix])
xv[np.isnan(xv)] = 0  # tongue: NaN -> 0
speed = np.sqrt(xvel**2 + yvel**2)
```

iii. The AI followed `findPosition.m` and `findVelocity.m` logic. For tongue specifically: no smoothing of position, NaN velocity set to 0, no baseline subtraction.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Tongue velocity is discretized per session using the 50th percentile (median) as threshold. Values >= threshold are labeled 1, values < threshold are labeled 0. Special handling: if threshold is 0, the median of positive values is used instead.

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

iii. The AI applied per-session 50th percentile thresholding as specified in the instructions. The threshold=0 edge case handling was the AI's own design decision.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue position data is interpolated from video frame times to the neural time axis. Frame times are aligned by subtracting the video offset and go cue time, then `interp1d` maps these to the neural time bin centers.

ii.
```python
taxis = TIME_AXIS + PARAMS['advance_movement']
aligned_times = frameTimes - vidshift - goCue[trix]
fx = interp1d(aligned_times[valid], x[valid], bounds_error=False, fill_value=np.nan)
xpos[:, trix] = fx(taxis)
```

iii. Matches `findPosition.m`: `interp1(traj(trix).frameTimes-vidshift-obj.bp.ev.(alignEv)(trix), ts, taxis)`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from DLC trajectory data in `obj.traj{2}` (bottom/top view / view 2). Features named 'top_paw' and 'bottom_paw' are used, identified by searching feature names containing 'paw'.

ii.
```python
traj_ref = f['obj']['traj'][1, 0]  # view 2 (top)
paw_indices = []
for i, name in enumerate(feat_names):
    if 'paw' in name.lower():
        paw_indices.append(i)
```

iii. The AI identified paw features from the bottom camera (view 2) per the reference code's `getDefaultParams.m` which lists `'top_paw','bottom_paw'` in view 2 features.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Processing: (1) For each paw feature: extract x, y positions, align and interpolate to neural time axis. (2) Fill missing position values with nearest. (3) Compute velocity via gradient. (4) Subtract baseline derivative (median of diff) from both x and y velocity. (5) Fill missing velocity with nearest. (6) Compute speed. (7) Average speed across paw features.

ii.
```python
# Fill missing positions
xp = np.interp(indices, indices[mask], xp[mask])
# Velocity with baseline subtraction
xv = np.gradient(xpos[:, trix])
basederiv_x = np.nanmedian(np.diff(xpos[:, trix]))
basederiv_y = np.nanmedian(np.diff(ypos[:, trix]))
xv = xv - basederiv_x
yv = yv - basederiv_y
# Average across paw features
avg_speed = np.mean(all_speeds, axis=0)
```

iii. The AI followed `findVelocity.m` for baseline subtraction and `findPosition.m` for position filling. Note: the MATLAB reference subtracts `basederiv(1)` (x baseline) from both x and y velocity, while the AI subtracts the respective baselines.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue velocity: per-session 50th percentile thresholding with the same `discretize_continuous` function.

ii.
```python
paw_disc = discretize_continuous(result['paw_vel'])
```

iii. Follows the instruction specification for 50th percentile discretization.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same alignment approach as tongue: interpolate from video frame times (corrected by video offset and go cue time) to neural time axis.

ii.
```python
aligned_times = frameTimes - vidshift - goCue[trix]
fx = interp1d(aligned_times[valid], x[valid], bounds_error=False, fill_value=np.nan)
xpos[:, trix] = fx(taxis)
```

iii. Uses the same time alignment pipeline as tongue, matching `findPosition.m`.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from separate `motionEnergy_{animal}_{date}.mat` files. The data is in `me.data`, which may be nested as `me.data.data` (handled by unwrapping).

ii.
```python
me_file = os.path.join(DATA_DIR, f'motionEnergy_{session_id}.mat')
me_raw = sio.loadmat(me_file)
me_struct = me_raw['me'][0, 0]
me_data = me_struct['data']
if me_data.dtype.names is not None and 'data' in me_data.dtype.names:
    me_data = me_data[0, 0]['data']
```

iii. The AI followed `loadMotionEnergy.m`: `if isstruct(me.data), me.data = me.data.data`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Processing: (1) Load raw motion energy per trial. (2) Get frame times from `traj{1}.frameTimes`. (3) Align by subtracting video offset and go cue time. (4) Interpolate to neural time axis using `interp1d`. (5) Fill NaN with nearest valid values.

ii.
```python
me.newdata(:,trix) = interp1(frameTimes-vidshift-alignTimes(trix), me.data{trix}, taxis)
# AI equivalent:
aligned_times = frameTimes - vidshift - goCue[trix]
f_interp = interp1d(aligned_times[valid], me_trial[valid], bounds_error=False, fill_value=np.nan)
me_interp = f_interp(taxis)
me_interp = np.interp(indices, indices[mask], me_interp[mask])  # fill NaN
```

iii. Matches `loadMotionEnergy.m` interpolation and NaN filling logic.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same as tongue and paw: per-session 50th percentile thresholding with `discretize_continuous`.

ii.
```python
me_disc = discretize_continuous(result['motion_energy'])
```

iii. The instructions specify 50th percentile. The reference code uses a manually-set `moveThresh` per session, but the AI follows the instruction specification instead.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is interpolated from video frame times to the neural time axis, same as tongue and paw.

ii.
```python
aligned_times = frameTimes - vidshift - goCue[trix]
f_interp = interp1d(aligned_times[valid], me_trial[valid], bounds_error=False, fill_value=np.nan)
me_interp = f_interp(taxis)
```

iii. Matches `loadMotionEnergy.m` alignment approach.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several missing data scenarios are handled:
- **Missing video data**: If `NdroppedFrames` is NaN, the trial's video data is skipped, and zeros are used for tongue/paw velocity.
- **Missing frame times**: If `frameTimes` doesn't exist, synthetic frame times are created at 400 Hz.
- **Missing tongue/paw/ME data**: If computation fails for a session, zeros are used as fallback.
- **NaN in velocity**: For tongue, NaN velocity is set to 0. For paw, NaN is filled with nearest value.
- **NaN in motion energy**: Filled with nearest value after interpolation.
- **Empty quality strings**: Included (not excluded) per MATLAB `findClusters.m` behavior.
- **Probe shape edge case**: When clu has only 1 entry, index 0 is used regardless of probe number.
- **Nested ME struct**: Unwrapped when `me.data` contains a sub-struct.
- **Discretization threshold=0**: Uses median of positive values instead.

ii.
```python
# Missing video data
if np.isnan(ndrop).any():
    continue

# Missing frame times fallback
frameTimes = np.arange(1, n_frames + 1) / 400.0

# Tongue NaN -> 0
xv[np.isnan(xv)] = 0

# Paw fill missing
xp = np.interp(indices, indices[mask], xp[mask])

# Session-level fallback
if tongue_vel is None:
    tongue_vel_trials.append(np.zeros(N_TIMEBINS, dtype=np.float32))
```

iii. The AI documented edge case handling in CONVERSION_NOTES Step 10 Check 5, noting specific issues found (JEB15 null quality strings, JEB7 probe shape, ME nested struct).

## 11-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is neural data processing: iterating over every cluster, for every trial, binning spikes and applying Gaussian smoothing. This is an O(n_clusters x n_trials) loop with histogram and convolution operations at each step. Video/kinematic processing (tongue, paw, motion energy) is also time-intensive due to per-trial interpolation loops.

ii.
```python
for ci, clu_idx in enumerate(cluid):
    for trial_num in range(1, Ntrials + 1):
        counts, _ = np.histogram(trial_spikes, bins=EDGES)
        fr = counts / PARAMS['dt']
        fr_smooth = causal_gaussian_smooth(fr, PARAMS['smooth'], PARAMS['bctype'])
        trialdat[:, ci, trial_num - 1] = fr_smooth
```

iii. The AI estimated ~6-7 seconds per session, ~180 seconds total (CONVERSION_NOTES Step 7).

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several nested loops could be vectorized:
- The spike binning loop iterates per-cluster, per-trial. Spikes could be grouped by trial using vectorized operations and binned in batch.
- The smoothing loop in `causal_gaussian_smooth` iterates per-column; `scipy.ndimage.convolve1d` could handle all columns at once.
- The tongue/paw velocity computation loops per-trial for gradient and NaN handling.
- The discretization loop iterates per-trial for thresholding.

ii.
```python
# Inner loop that could be vectorized
for ci, clu_idx in enumerate(cluid):
    for trial_num in range(1, Ntrials + 1):
        # ... histogram + smooth per cluster per trial

# Smoothing loop
for j in range(x_padded.shape[1]):
    result[:, j] = convolve(x_padded[:, j], kern, mode='same')

# Velocity per trial
for trix in range(Ntrials):
    xv = np.gradient(xpos[:, trix])
```

iii. The AI acknowledged timing but did not implement major vectorization optimizations.

## 11-c. What processing does the code repeat multiple times?

i. Several computations are repeated:
- **Video offset**: `find_video_offset(f)` is called separately in `compute_tongue_velocity`, `compute_paw_velocity`, and `load_motion_energy` for the same session file.
- **Feature name extraction**: Feature names from DLC trajectory are parsed independently in tongue and paw functions.
- **Smoothing kernel construction**: The Gaussian kernel is recreated for every single call to `causal_gaussian_smooth`, which is called once per cluster per trial.
- **Discretization in plotting**: `discretize_continuous` is called both in `plot_processing` and in the main data assembly loop.

ii.
```python
# Called 3 times per session:
vidshift = find_video_offset(f)  # in compute_tongue_velocity
vidshift = find_video_offset(f)  # in compute_paw_velocity
vidshift = find_video_offset(f)  # in load_motion_energy

# Kernel recreated each call:
kern = sig_windows.gaussian(N, std=(N-1)/(2*2.5))
```

iii. No explicit justification provided for the repetition.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations are unnecessary:
- **PSTH computation**: The reference code computes PSTHs (condition-averaged firing rates), but the AI's code does not compute PSTHs. However, the AI computes trialdat for ALL trials (including invalid ones) and then filters to valid trials afterward, wasting computation on invalid trials.
- **Absolute spike times** (`clu.tm`): Read but never used in the actual alignment (alignment uses `trialtm` and `goCue` instead).
- **Processing plots during `--show-processing`**: Creates detailed visualizations including discretized outputs that are recomputed in the main loop.
- **trialdat_full stored for plotting**: Full trialdat array is stored in result dict when show_processing is enabled.

ii.
```python
# Absolute spike times read but unused
tm_abs_ref = clu_group['tm'][clu_idx, 0]
spike_tm_abs = f[tm_abs_ref][:].flatten()  # never used after this

# trialdat computed for ALL trials, then filtered
trialdat = np.zeros((N_TIMEBINS, len(cluid), Ntrials), dtype=np.float32)
# ... processes all Ntrials ...
# Then only valid_trials are extracted
for trial_idx in valid_trials:
    neural_trials.append(trialdat[:, :, trial_idx].T)
```

iii. No explicit justification. Computing trialdat for all trials follows the reference pattern (getSeq.m also computes for all Ntrials).
