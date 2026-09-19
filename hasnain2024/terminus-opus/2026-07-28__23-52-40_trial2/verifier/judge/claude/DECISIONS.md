# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads only sessions from the `Ephys_Behavior` directory (fixed-delay task), totaling 25 sessions from 10 subjects. Sessions are listed in a hard-coded `SESSION_META` list derived from the authors' `load<ANM>_ALMVideo.m` scripts. Each session is opened once via `h5py.File()` and read directly from the HDF5 structure. Motion energy is loaded from a separate `motionEnergy_*.mat` file via `scipy.io.loadmat`. The AI does NOT include the 19 sessions from `RandomizedDelay_Ephys_Behavior`.

ii.
```python
SESSION_META = [
    ('EKH1', '2021-08-07', [1]),
    ('EKH3', '2021-08-11', [1, 2]),
    ...
    ('JGR3', '2021-11-18', [1]),
]
DATA_DIR = 'data/Ephys_Behavior'

for idx, (animal, date, probes) in enumerate(sessions):
    result = process_session(animal, date, probes, ...)

f = h5py.File(data_file, 'r')
```

iii. CONVERSION_NOTES.md states "Data source: Ephys_Behavior directory (fixed delay DR+WC two-context task)". The AI chose to only process fixed-delay sessions, not the randomized-delay sessions.

## 1-b. How are the data split into subjects?

i. The animal name is taken from the first element of each `SESSION_META` tuple (e.g., `'EKH1'`). Unique subjects are collected as a sorted set at assembly time.

ii.
```python
subjects = sorted(list(set(r['animal'] for r in all_results)))
subj_idx = subjects.index(result['animal'])
```

iii. The animal ID is explicitly listed in SESSION_META rather than parsed from filenames. This is straightforward and correct.

## 1-c. How are the data split into sessions?

i. One session is one entry in `SESSION_META`, keyed by `(animal, date, probes)`. Only the `Ephys_Behavior` directory is searched. Each session corresponds to one `.mat` file and becomes one element of `neural`, `input`, and `output`. The result is 25 sessions.

ii.
```python
data_file = os.path.join(DATA_DIR, f'data_structure_{session_id}.mat')
```

iii. Sessions come from the authors' loading scripts. Only the Ephys_Behavior folder is used.

## 1-d. How are the data split into trials?

i. Each trial is one row of the behavioral arrays read from `obj.bp`. `Ntrials` is read from `bp['Ntrials']` and used to index into arrays like `hit`, `miss`, `R`, etc. Spike times carry a trial number (`clu.trial`) which is used for binning.

ii.
```python
Ntrials = int(bp['Ntrials'][0, 0])
L = bp['L'][:].flatten()
R = bp['R'][:].flatten()
hit = bp['hit'][:].flatten()
...
```

iii. The Bpod table defines the trials directly.

## 1-e. How are trials filtered based on quality controls?

i. The AI excludes three categories of trials: early-lick (`early == 1`), no-response/ignore (`no == 1`), and photostimulation (`stim_enable == 1`). Only hit and miss trials are retained. This differs from the reference, which keeps ignore trials (as a third outcome class). The AI also does NOT check for trials that extend past the end of the neural recording.

ii.
```python
valid_mask = (early == 0) & (no == 0) & (stim_enable == 0) & ((hit == 1) | (miss == 1))
valid_trials = np.where(valid_mask)[0]
```

iii. CONVERSION_NOTES.md says "Exclude early lick, ignore/no-response, stimulation trials. Include: hit and miss trials." The AI considered this consistent with the paper's methods.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` spike-sorted clusters. Each cluster carries `trial` (1-based trial number), `trialtm` (spike time relative to trial start), `tm` (absolute spike time), and `quality` (manual curation label). The go cue times `bp.ev.goCue` provide alignment.

ii.
```python
spike_trialtm = f[tm_ref][:].flatten()
spike_trial = f[trial_ref][:].flatten().astype(int)
spike_goCue = goCue[spike_trial_valid - 1]
spike_aligned = spike_trialtm_valid - spike_goCue
```

iii. The AI reads the same raw variables as the reference.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to the go cue by subtracting `goCue[trial]` from `trialtm`. They are binned into 10 ms bins (not 5 ms as in the reference) spanning -2.5 to 2.5 s (500 bins). Spike counts are converted to firing rates (spks/sec) by dividing by `dt`. Firing rates are then smoothed with a **causal** Gaussian kernel of window=15 samples, with the first half zeroed and reflect boundary conditions.

ii.
```python
PARAMS = {
    'dt': 1/100,         # 10ms time bins
    'smooth': 15,        # smoothing window (samples)
    'bctype': 'reflect',
}

fr = counts / PARAMS['dt']
fr_smooth = causal_gaussian_smooth(fr, PARAMS['smooth'], PARAMS['bctype'])
```

The causal smoothing:
```python
kern = sig_windows.gaussian(N, std=(N-1)/(2*2.5))
kern[:N//2] = 0  # causal: zero out first half
kern = kern / kern.sum()
```

iii. CONVERSION_NOTES.md says "Smoothing: causal Gaussian kernel (gausswin), window=15, first half zeroed for causality, boundary=reflect" and "dt = 10ms: Standard across analysis scripts".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. First, clusters are filtered by quality label — excluding `garbage`, `gabrga`, `noisy`, and `real?` (matching the MATLAB `findClusters.m`). Unlike the reference, `poor` is NOT excluded. Then, neurons with mean firing rate <= 1 Hz are removed. The AI also skips sessions with fewer than 10 neurons.

ii.
```python
PARAMS['quality_exclude'] = ['garbage', 'gabrga', 'noisy', 'real?']

cluid = find_clusters(qualities, PARAMS['quality_exclude'])

meanFRs = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
fr_mask = meanFRs > PARAMS['lowFR']

if n_neurons < 10:
    print(f'    Skipping {session_id}: only {n_neurons} neurons (need >= 10)')
    return None
```

iii. CONVERSION_NOTES.md documents the quality filter as matching `findClusters.m`. The 10-neuron minimum is mentioned in the paper: "at least 10 units".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Aligned to the go cue by subtracting `goCue[trial]` from `trialtm` for each spike. This matches the reference approach.

ii.
```python
spike_goCue = goCue[spike_trial_valid - 1]
spike_aligned = spike_trialtm_valid - spike_goCue
counts, _ = np.histogram(trial_spikes, bins=EDGES)
```

iii. Follows `alignSpikes.m` from the reference code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 10 ms time bins (`dt = 1/100`), producing 500 bins over the -2.5 to 2.5 s window. This differs from the reference which uses 5 ms bins (`dt = 1/200`, 1000 bins).

ii.
```python
PARAMS = {
    'dt': 1/100,         # 10ms time bins
}
EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
N_TIMEBINS = len(TIME_AXIS)  # 500
```

iii. CONVERSION_NOTES.md says "Neural data time bin: 10ms, params.dt = 1/100" and in the consistency check: "dt: default 1/200, N/A, 1/100 in most scripts. Using 1/100 (10ms)." The AI chose 10 ms believing it was more standard across the analysis scripts.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The time axis is computed from the bin edges, as the center of each 10 ms bin spanning -2.5 to 2.5 s. No raw data variable is used; it is defined by the binning grid.

ii.
```python
EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
```

iii. Same approach as the reference — it is a computed quantity.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing beyond computing the bin centers. The time axis is defined by the binning grid.

ii. N/A (same as 3-a)

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time axis is the neural binning grid itself. Spike times are expressed relative to the go cue and counted into `EDGES`, and the input is the center of those same bins.

ii.
```python
time_input = TIME_AXIS.astype(np.float32).reshape(1, -1)
```

iii. N/A

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI uses `bp.R` (right trial indicator) directly as the lick direction. A value of 1 means right, 0 means left.

ii.
```python
R = bp['R'][:].flatten()
lick_direction = R.copy()  # 1=right, 0=left
```

iii. CONVERSION_NOTES.md says "R (right trial): left=0, right=1, Per trial".

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI directly uses `R` as lick direction without considering the relationship between instructed side and actual lick. Since ignore trials are excluded, the AI treats `R=1` as right lick and `R=0` as left lick. There is no "no lick" class — only two categories (left=0, right=1). On hit trials `R` matches the lick direction; on miss trials the lick is the *opposite* of `R`, so the AI's assignment is incorrect for miss trials.

ii.
```python
lick_direction = R.copy()  # 1=right, 0=left
# output_values: ['left', 'right']
```

iii. The AI treated `R` as lick direction directly. It does not derive lick direction from the combination of instructed side and outcome, which would be needed for miss trials.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `bp.autowater` — the autowater flag indicates WC context.

ii.
```python
autowater = bp['autowater'][:].flatten()
context = 1 - autowater  # DR=1, WC=0
```

iii. Direct relabeling of the autowater flag.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabeling: `autowater=1` becomes WC (0), `autowater=0` becomes DR (1). The mapping `context = 1 - autowater` produces WC=0, DR=1.

ii.
```python
context = 1 - autowater  # DR=1, WC=0
```

iii. Codes follow the same convention as the reference (WC=0, DR=1).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `bp.hit` directly — the AI uses the hit flag as the outcome value (1=correct, 0=incorrect). Since ignore trials are excluded, only hit and miss trials remain.

ii.
```python
hit = bp['hit'][:].flatten()
outcome = hit.copy()  # 1=correct, 0=incorrect
```

iii. CONVERSION_NOTES.md: "hit: incorrect=0, correct=1, Per trial".

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI uses `hit` directly as outcome: 1 = correct, 0 = incorrect. There is no "ignore" class because ignore trials are excluded during trial filtering. This produces only 2 classes instead of the 3 specified in the instructions (incorrect=0, correct=1, ignore=2).

ii.
```python
outcome = hit.copy()  # 1=correct, 0=incorrect
# output_values: ['incorrect', 'correct']
```

iii. The AI excluded ignore trials, so there was no need for a third class.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The AI uses `obj.traj[0]` (side view, index 0) to get the tongue position from the DLC tracking. It extracts the `tongue` feature's x and y coordinates from the `ts` array, and uses `frameTimes` for temporal alignment. It uses only the side camera view, not both cameras.

ii.
```python
traj_ref = f['obj']['traj'][0, 0]  # view 1 (side)
# Find tongue feature index
for i, name in enumerate(feat_names):
    if name == 'tongue':
        tongue_idx = i
x = ts_data[tongue_idx, 0, :]
y = ts_data[tongue_idx, 1, :]
```

iii. CONVERSION_NOTES.md: "Tongue velocity: Side view (view 1), tongue feature, speed = sqrt(xvel^2 + yvel^2)".

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI: (1) Extracts x, y positions from DLC tracking. (2) **Interpolates** them to the neural time axis using `scipy.interpolate.interp1d`. (3) Computes velocity via `np.gradient` on the interpolated positions. (4) Sets NaN velocities to 0. (5) Computes speed as `sqrt(xvel^2 + yvel^2)`. No likelihood filtering is applied — all frames are interpolated regardless of DLC confidence. No per-run smoothing is done. No normalization across cameras (only one camera used).

ii.
```python
fx = interp1d(aligned_times[valid], x[valid], bounds_error=False, fill_value=np.nan)
fy = interp1d(aligned_times[valid], y[valid], bounds_error=False, fill_value=np.nan)
xpos[:, trix] = fx(taxis)
ypos[:, trix] = fy(taxis)

xv = np.gradient(xpos[:, trix])
yv = np.gradient(ypos[:, trix])
xv[np.isnan(xv)] = 0
yv[np.isnan(yv)] = 0
speed = np.sqrt(xvel**2 + yvel**2)
```

iii. CONVERSION_NOTES.md describes this as "matching findVelocity.m".

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI discretizes using a 50th percentile (median) threshold across all time bins and trials in the session. Values below the threshold are 0 ("low"), values at or above are 1 ("high"). There is NO "not visible" class — NaN values from missing tracking are replaced with 0 before discretization. If the threshold is 0 (common since tongue is mostly invisible), the AI uses the median of positive values instead.

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
    result = []
    for v in values_per_trial:
        disc = (v >= threshold).astype(np.float32)
        result.append(disc)
    return result
```

iii. CONVERSION_NOTES.md mentions the edge case handling: "When threshold=0, use median of positive values".

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI computes the video offset using `findVideoOffset` (matching the reference's `findVideoOffset.m`), subtracts it and the go cue time from `frameTimes`, then **interpolates** the positions to the neural time axis using `interp1d`. This contrasts with the reference which bins frame-level values into the same 5 ms grid.

ii.
```python
vidshift = find_video_offset(f)
aligned_times = frameTimes - vidshift - goCue[trix]
fx = interp1d(aligned_times[valid], x[valid], bounds_error=False, fill_value=np.nan)
xpos[:, trix] = fx(taxis)
```

iii. The video offset computation matches the reference.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The AI uses `obj.traj[1]` (bottom/top view, index 1) and finds ALL features with 'paw' in the name — both `top_paw` and `bottom_paw`. The reference uses only `top_paw`.

ii.
```python
traj_ref = f['obj']['traj'][1, 0]  # view 2 (top)
paw_indices = []
for i, name in enumerate(feat_names):
    if 'paw' in name.lower():
        paw_indices.append(i)
```

iii. CONVERSION_NOTES.md: "Paw velocity: Top view (view 2), top_paw + bottom_paw averaged".

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw feature: (1) Extract x, y positions. (2) Interpolate to neural time axis. (3) Fill NaN positions with nearest-neighbor interpolation. (4) Compute velocity via `np.gradient`. (5) Subtract baseline derivative (`nanmedian(diff(position))`). (6) Fill NaN velocities with nearest-neighbor interpolation. (7) Compute speed. Then the speeds from both paw features are averaged. This is substantially different from the reference, which uses a simpler pipeline (likelihood filter, per-run Gaussian smoothing, gradient within runs).

ii.
```python
# Fill missing with nearest
mask = ~np.isnan(xp)
if np.any(mask):
    xp = np.interp(indices, indices[mask], xp[mask])

# Subtract baseline
basederiv_x = np.nanmedian(np.diff(xpos[:, trix]))
xv = xv - basederiv_x

# Average across paw features
avg_speed = np.mean(all_speeds, axis=0)
```

iii. CONVERSION_NOTES.md: "paw subtract baseline, matching findVelocity.m".

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue velocity — 50th percentile threshold producing 2 categories ("low" and "high"), no "not visible" class.

ii.
```python
paw_disc = discretize_continuous(result['paw_vel'])
```

iii. Same discretization function used for all continuous outputs.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: video offset subtracted, go cue subtracted, then interpolated to the neural time axis.

ii. Same code as tongue velocity alignment.

iii. Same approach as tongue alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. A separate `motionEnergy_*.mat` file. The AI loads it with `scipy.io.loadmat` and handles the nested struct (matching the MATLAB `loadMotionEnergy.m` pattern of checking `if isstruct(me.data), me.data = me.data.data`).

ii.
```python
me_raw = sio.loadmat(me_file)
me_struct = me_raw['me'][0, 0]
me_data = me_struct['data']
if me_data.dtype.names is not None and 'data' in me_data.dtype.names:
    me_data = me_data[0, 0]['data']
```

iii. CONVERSION_NOTES.md: "Motion energy: loaded separately, interpolated to neural time axis using video frameTimes".

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI: (1) Gets frame times from the side camera's `traj[0].frameTimes`. (2) Aligns by subtracting video offset and go cue. (3) Interpolates to neural time axis using `interp1d`. (4) Fills NaN with nearest-neighbor interpolation. The reference instead bins the frame-level values into the 5 ms grid without interpolation.

ii.
```python
f_interp = interp1d(aligned_times[valid], me_trial[valid],
                   bounds_error=False, fill_value=np.nan)
me_interp = f_interp(taxis)
# Fill NaN with nearest
mask = ~np.isnan(me_interp)
if np.any(mask):
    me_interp = np.interp(indices, indices[mask], me_interp[mask])
```

iii. CONVERSION_NOTES.md: "Motion energy loading and interpolation matching loadMotionEnergy.m".

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same as tongue and paw — 50th percentile threshold, 2 categories ("low" and "high"), no "no video" class.

ii.
```python
me_disc = discretize_continuous(result['motion_energy'])
```

iii. Same discretization function used for all continuous outputs.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same as the tracking: video offset and go cue subtracted from frame times, then interpolated to the neural time axis. The reference bins the values instead of interpolating.

ii. See 9-b code.

iii. See 9-b justification.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several cases: (1) If `NdroppedFrames` contains NaN, the trial is skipped for velocity computation (tongue/paw set to 0). (2) If `frameTimes` are all NaN, the trial is skipped. (3) If interpolation fails, the trial is skipped and set to 0. (4) For paw, NaN positions are filled with nearest-neighbor interpolation. (5) For motion energy, NaN values are filled with nearest-neighbor interpolation. (6) Broad try/except blocks catch unexpected errors and set values to 0.

ii.
```python
ndrop = f[ndrop_ref][:].flatten()
if np.isnan(ndrop).any():
    continue

if np.all(np.isnan(frameTimes)):
    continue

# Fill NaN with nearest
mask = ~np.isnan(me_interp)
if np.any(mask):
    me_interp = np.interp(indices, indices[mask], me_interp[mask])
```

iii. The AI uses nearest-neighbor filling for missing data rather than the reference's approach of marking bins as "not visible".

## 11-a. What are the most time-consuming steps of the code?

i. The AI's code is dominated by the per-trial, per-cluster spike binning loop, and the per-trial DLC processing loops. The conversion of 25 sessions takes a total of about 210 seconds based on the output logs. Loading and processing each session ranges from 4 to 17 seconds.

ii.
```python
for ci, clu_idx in enumerate(cluid):
    for trial_num in range(1, Ntrials + 1):
        counts, _ = np.histogram(trial_spikes, bins=EDGES)
        fr_smooth = causal_gaussian_smooth(fr, PARAMS['smooth'], PARAMS['bctype'])
```

iii. The AI identified the per-trial per-neuron loop as a bottleneck in CONVERSION_NOTES.md.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The inner loop over trials within the spike binning (line 301: `for trial_num in range(1, Ntrials + 1)`) could be vectorized using `np.histogram2d` as the reference does. The per-trial smoothing could also be vectorized by smoothing the entire (time, neurons, trials) array at once. The per-trial velocity computation loops could similarly be vectorized.

ii.
```python
# This inner loop is the main bottleneck:
for trial_num in range(1, Ntrials + 1):
    trial_mask = spike_trial_valid == trial_num
    trial_spikes = spike_aligned[trial_mask]
    counts, _ = np.histogram(trial_spikes, bins=EDGES)
    fr = counts / PARAMS['dt']
    fr_smooth = causal_gaussian_smooth(fr, PARAMS['smooth'], PARAMS['bctype'])
    trialdat[:, ci, trial_num - 1] = fr_smooth
```

iii. The reference avoids this by using `np.histogram2d` to bin all trials at once per cluster.

## 11-c. What processing does the code repeat multiple times?

i. The video offset (`find_video_offset(f)`) is computed separately in `compute_tongue_velocity`, `compute_paw_velocity`, and `load_motion_energy` — three times per session instead of once. The `frameTimes` and `featNames` are also re-read from the HDF5 file in each function.

ii.
```python
# In compute_tongue_velocity:
vidshift = find_video_offset(f)
# In compute_paw_velocity:
vidshift = find_video_offset(f)
# In load_motion_energy:
vidshift = find_video_offset(f)
```

iii. The reference computes the video offset once in `Camera.__init__` and reuses it.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI reads `tm` (absolute spike times) for each cluster but never uses them — only `trialtm` is needed for alignment. The code also reads and checks `NdroppedFrames` for the tongue and paw, which is not used by the reference. Additionally, the code computes velocities for ALL trials (not just valid ones) and then selects valid trials at the end, wasting computation on excluded trials.

ii.
```python
# tm_abs is read but never used for anything
tm_abs_ref = clu_group['tm'][clu_idx, 0]
spike_tm_abs = f[tm_abs_ref][:].flatten()

# Velocity computed for all Ntrials, then only valid_trials selected
trialdat = np.zeros((N_TIMEBINS, len(cluid), Ntrials), dtype=np.float32)
# ... later ...
for trial_idx in valid_trials:
    neural_trials.append(trialdat[:, :, trial_idx].T)
```

iii. Computing velocities and spike counts for excluded trials is wasted work.
