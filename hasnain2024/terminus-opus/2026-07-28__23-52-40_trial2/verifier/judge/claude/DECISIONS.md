# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from HDF5 `.mat` files using `h5py` only. A hardcoded list `SESSION_META` of 25 sessions (animal, date, probes) defines which sessions to process, all from a single directory `data/Ephys_Behavior`. The randomized delay sessions in `data/RandomizedDelay_Ephys_Behavior` are not included. Each session's data structure file and motion energy file are loaded separately.

ii.
```python
DATA_DIR = 'data/Ephys_Behavior'

SESSION_META = [
    ('EKH1', '2021-08-07', [1]),
    ('EKH3', '2021-08-11', [1, 2]),
    ...
    ('JGR3', '2021-11-18', [1]),
]

f = h5py.File(data_file, 'r')
```

iii. The AI's CONVERSION_NOTES.md states it focused on "Ephys_Behavior directory (fixed delay DR+WC two-context task)". The AI only loads HDF5 files with `h5py` and does not handle the v5 MATLAB format (no `scipy.io.loadmat` fallback).

## 1-b. How are the data split into subjects?

i. The animal name is taken directly from the `SESSION_META` tuple (first element). Unique subjects are collected and sorted for the `subjects` list.

ii.
```python
SESSION_META = [
    ('EKH1', '2021-08-07', [1]),
    ...
]
subjects = sorted(list(set(r['animal'] for r in all_results)))
subj_idx = subjects.index(result['animal'])
```

iii. The AI identifies 10 subjects from the 25 sessions. The reference identifies 14 subjects from 44 sessions. The paper says 9 mice for the DR task.

## 1-c. How are the data split into sessions?

i. Each entry in `SESSION_META` is one session. Only `Ephys_Behavior` sessions are included (25 total). The `RandomizedDelay_Ephys_Behavior` folder (19 additional sessions) is ignored.

ii.
```python
DATA_DIR = 'data/Ephys_Behavior'
data_file = os.path.join(DATA_DIR, f'data_structure_{session_id}.mat')
```

iii. The AI's CONVERSION_NOTES.md says "25 sessions" and "Subjects (DR): 9 mice" from the paper. The AI chose to exclude the randomized delay sessions.

## 1-d. How are the data split into trials?

i. Trials are indexed by the `Ntrials` field from `obj.bp`. Per-trial behavioral fields (`L`, `R`, `hit`, `miss`, `no`, `autowater`, `early`, `stim.enable`) are read as flat arrays. Trial indices are 0-based in the code.

ii.
```python
Ntrials = int(bp['Ntrials'][0, 0])
L = bp['L'][:].flatten()
R = bp['R'][:].flatten()
hit = bp['hit'][:].flatten()
...
```

iii. This is straightforward indexing from the behavioral data structure.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies three filters: excludes early-lick trials (`early == 0`), excludes no-response/ignore trials (`no == 0`), and excludes stimulation trials (`stim_enable == 0`). Additionally, only hit and miss trials are kept (`(hit == 1) | (miss == 1)`). This is more aggressive filtering than the reference, which keeps ignore trials.

ii.
```python
valid_mask = (early == 0) & (no == 0) & (stim_enable == 0) & ((hit == 1) | (miss == 1))
valid_trials = np.where(valid_mask)[0]
```

iii. The CONVERSION_NOTES.md says "Exclude: early lick, ignore/no-response, stimulation. Include: hit and miss trials." The AI justified this by wanting only "valid" trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Spike times from `obj.clu` clusters: `trialtm` (spike times relative to trial start), `trial` (which trial each spike belongs to, 1-indexed), and `quality` (curation label). Go cue times from `obj.bp.ev.goCue`.

ii.
```python
spike_trialtm = f[tm_ref][:].flatten()
spike_trial = f[trial_ref][:].flatten().astype(int)
spike_goCue = goCue[spike_trial_valid - 1]
spike_aligned = spike_trialtm_valid - spike_goCue
```

iii. Same raw variables as the reference.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to go cue, binned into 10ms time bins (not 5ms as in reference), converted to firing rate (spks/s), then smoothed with a **causal** Gaussian kernel (window=15 samples, first half zeroed). The reference uses a symmetric Gaussian with sigma=14ms via `gaussian_filter1d`.

ii.
```python
PARAMS = {
    'dt': 1/100,         # 10ms time bins
    'smooth': 15,        # smoothing window (samples)
}

# Causal Gaussian smoothing
kern = sig_windows.gaussian(N, std=(N-1)/(2*2.5))
kern[:N//2] = 0  # causal: zero out first half
kern = kern / kern.sum()
```

iii. The AI's CONVERSION_NOTES.md says "params.dt = 1/100 in most scripts" and "causal Gaussian kernel, window=15, first half zeroed for causality". The AI chose 10ms bins and causal smoothing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) Quality filter excluding clusters labeled 'garbage', 'gabrga', 'noisy', 'real?' (does NOT exclude 'poor', unlike reference). (2) Firing rate filter: mean FR > 1 Hz.

ii.
```python
'quality_exclude': ['garbage', 'gabrga', 'noisy', 'real?'],

cluid = find_clusters(qualities, PARAMS['quality_exclude'])

meanFRs = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
fr_mask = meanFRs > PARAMS['lowFR']
```

iii. The AI also adds a minimum neuron count per session (< 10 neurons = skip session). The reference does not have this filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned by subtracting the go cue time for each spike's trial: `spike_aligned = trialtm - goCue[trial]`. This matches the reference approach.

ii.
```python
spike_goCue = goCue[spike_trial_valid - 1]
spike_aligned = spike_trialtm_valid - spike_goCue
counts, _ = np.histogram(trial_spikes, bins=EDGES)
```

iii. Matches `alignSpikes.m` from the reference code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 10ms bins (`dt = 1/100`), producing 500 time bins over the -2.5 to 2.5s window. The reference uses 5ms bins (`dt = 1/200`), producing 1000 time bins.

ii.
```python
PARAMS = {
    'dt': 1/100,         # 10ms time bins
}
EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
N_TIMEBINS = len(TIME_AXIS)  # 500
```

iii. The AI's CONVERSION_NOTES.md step 4 notes "dt: default 1/200 | N/A | 1/100 in most scripts | Using 1/100 (10ms)". The AI chose 10ms over the reference's 5ms.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the time axis itself, constructed from the bin edges, representing time from go cue in seconds.

ii.
```python
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
time_input = TIME_AXIS.astype(np.float32).reshape(1, -1)
```

iii. This is a synthetic variable defined by the bin grid.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is computed as bin centers from the edges. No additional processing.

ii.
```python
EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
```

iii. Straightforward bin center computation.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time axis IS the neural binning grid. Neural spikes are binned into the same edges, so alignment is inherent.

ii.
```python
counts, _ = np.histogram(trial_spikes, bins=EDGES)
```

iii. Same approach as the reference.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI uses only `obj.bp.R` (right trial indicator) directly as lick direction. The reference uses `R`, `hit`, and `miss` together to infer actual lick direction.

ii.
```python
R = bp['R'][:].flatten()
lick_direction = R.copy()  # 1=right, 0=left
```

iii. The AI treats `R` as lick direction. However, `R` indicates the instructed/correct side, not the actual lick direction. For miss trials, the animal licked the opposite side.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Direct copy of `R` field. No derivation from hit/miss outcomes. This means on miss trials, the lick direction is recorded as the instructed side (wrong), not the actual lick side.

ii.
```python
lick_direction = R.copy()  # 1=right, 0=left
```

iii. The AI did not account for the fact that on miss trials the animal licked the opposite direction from the instructed side. The reference correctly derives: hit+R=right, hit+L=left, miss+R=left, miss+L=right.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `obj.bp.autowater` flag. Autowater=1 indicates WC context, autowater=0 indicates DR context.

ii.
```python
autowater = bp['autowater'][:].flatten()
context = 1 - autowater  # DR=1, WC=0
```

iii. Matches the reference approach. WC=0, DR=1.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Simple inversion: `1 - autowater`. Autowater=1 -> WC=0, autowater=0 -> DR=1.

ii.
```python
context = 1 - autowater  # DR=1, WC=0 (autowater=1 means WC)
```

iii. Equivalent to the reference's `np.where(autowater, CONTEXT['WC'], CONTEXT['DR'])`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `obj.bp.hit` directly. Since ignore trials are filtered out, hit=1 maps to correct and hit=0 (i.e., miss) maps to incorrect.

ii.
```python
hit = bp['hit'][:].flatten()
outcome = hit.copy()  # 1=correct, 0=incorrect
```

iii. The AI uses only 2 classes (correct/incorrect) because ignore trials are excluded. The reference uses 3 classes (correct/incorrect/ignore).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct copy of `hit` field. Since the AI filtered out ignore trials, the remaining values are 0 (miss=incorrect) and 1 (hit=correct).

ii.
```python
outcome = hit.copy()  # 1=correct, 0=incorrect
```

iii. For the trials the AI keeps (only hit and miss), this encoding is functionally correct. But the reference keeps ignore trials as a third class.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. DLC tracking data from `obj.traj`, specifically view 1 (side camera, index 0) only. The 'tongue' feature's x, y positions and frame times are used. The reference uses both side and bottom camera tongue features.

ii.
```python
traj_ref = f['obj']['traj'][0, 0]  # view 1 (side) only
# Finds 'tongue' feature index
```

iii. The AI uses only the side camera. The reference uses both side ('tongue') and bottom ('top_tongue') cameras, normalizes each by its 90th percentile, and averages them.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI: (1) extracts x, y positions from DLC tracking, (2) **interpolates** positions to the neural time axis using `interp1d`, (3) computes velocity as `np.gradient` of the interpolated positions, (4) sets NaN velocities to 0, (5) computes speed as magnitude. No likelihood filtering is applied before interpolation. No per-run smoothing.

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

iii. The AI follows the reference code's `findPosition.m` (interpolation to neural time axis) and `findVelocity.m` (gradient). However, it does not apply likelihood filtering (>0.9), does not smooth within contiguous runs of valid frames, and fills NaN with 0 instead of using a "not visible" class.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. 2-class discretization at the 50th percentile. A special case: if threshold==0, uses median of positive values instead. No "not visible" class.

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

iii. The reference uses 3 classes (below threshold=0, above threshold=1, not visible=2). The AI uses 2 classes (low=0, high=1). The special threshold==0 handling is non-standard.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by subtracting the video offset and go cue time, then positions are interpolated to the neural time axis using `interp1d`.

ii.
```python
vidshift = find_video_offset(f)
aligned_times = frameTimes - vidshift - goCue[trix]
fx = interp1d(aligned_times[valid], x[valid], bounds_error=False, fill_value=np.nan)
xpos[:, trix] = fx(taxis)
```

iii. The video offset computation matches `findVideoOffset.m`. The interpolation-based alignment differs from the reference's bin-averaging approach.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. DLC tracking from view 2 (bottom camera, index 1). The AI uses **both** 'top_paw' and 'bottom_paw' features and averages their speeds. The reference uses only 'top_paw'.

ii.
```python
traj_ref = f['obj']['traj'][1, 0]  # view 2 (top)
paw_indices = []
for i, name in enumerate(feat_names):
    if 'paw' in name.lower():
        paw_indices.append(i)
# Average across paw features
avg_speed = np.mean(all_speeds, axis=0)
```

iii. The AI averages both paw features. The reference notes that `bottom_paw` has poor tracking reliability during the delay epoch and uses only `top_paw`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The AI: (1) interpolates x, y to neural time axis, (2) fills NaN with nearest-neighbor interpolation, (3) computes velocity via `np.gradient`, (4) subtracts baseline derivative (median of diff), (5) fills remaining NaN with interpolation, (6) computes speed. Applied to each paw feature, then averaged.

ii.
```python
fx = interp1d(aligned_times[valid], x[valid], bounds_error=False, fill_value=np.nan)
xp = fx(taxis)
# Fill missing with nearest
xp = np.interp(indices, indices[mask], xp[mask])
# Subtract baseline
basederiv_x = np.nanmedian(np.diff(xpos[:, trix]))
xv = xv - basederiv_x
```

iii. The baseline subtraction and nearest-fill are non-standard additions not in the reference code. The reference does not interpolate positions or fill missing values.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same 2-class discretization as tongue velocity using `discretize_continuous`. No "not visible" class.

ii.
```python
paw_disc = discretize_continuous(result['paw_vel'])
```

iii. Same issue as tongue: reference uses 3 classes. The AI's special threshold==0 logic could also apply here.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same interpolation approach as tongue velocity: frame times corrected by video offset and go cue, then positions interpolated to neural time axis.

ii.
```python
aligned_times = frameTimes - vidshift - goCue[trix]
fx = interp1d(aligned_times[valid], x[valid], bounds_error=False, fill_value=np.nan)
```

iii. Same approach as tongue, using `interp1d` instead of bin-averaging.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Separate `motionEnergy_<animal>_<date>.mat` files loaded with `scipy.io.loadmat`. Handles nested struct wrapping (matching `loadMotionEnergy.m`).

ii.
```python
me_raw = sio.loadmat(me_file)
me_struct = me_raw['me'][0, 0]
me_data = me_struct['data']
if me_data.dtype.names is not None and 'data' in me_data.dtype.names:
    me_data = me_data[0, 0]['data']
```

iii. The AI correctly handles the nested struct format.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy values are interpolated to the neural time axis using `interp1d`, then NaN values are filled with nearest-neighbor interpolation. The reference bins frame-level values using mean.

ii.
```python
f_interp = interp1d(aligned_times[valid], me_trial[valid],
                   bounds_error=False, fill_value=np.nan)
me_interp = f_interp(taxis)
# Fill NaN with nearest
me_interp = np.interp(indices, indices[mask], me_interp[mask])
```

iii. The AI uses interpolation and NaN filling rather than bin-averaging. The reference uses `_bin_frames` which computes the mean of valid frames in each bin.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same 2-class discretization as other movement variables.

ii.
```python
me_disc = discretize_continuous(result['motion_energy'])
```

iii. Same 2-class vs 3-class difference as tongue and paw.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Frame times from the side camera's trajectory data are used. Video offset correction applied. Interpolated to neural time axis.

ii.
```python
traj_ref = f['obj']['traj'][0, 0]
ft_ref = traj_group['frameTimes'][trix, 0]
frameTimes = f[ft_ref][:].flatten()
aligned_times = frameTimes - vidshift - goCue[trix]
f_interp = interp1d(aligned_times[valid], me_trial[valid], ...)
```

iii. Uses the same offset correction as other camera streams. Interpolation rather than binning.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several cases: (1) Missing data files: skips session. (2) NaN frame times: skips trial for that stream. (3) NaN in tracking: attempts nearest-fill interpolation. (4) Sessions with < 10 neurons: skips entirely. (5) Motion energy nested struct: unwraps. (6) Paw with NaN positions: fills with nearest interpolation. Broad `try/except` blocks silently skip trials/streams on any error.

ii.
```python
if np.all(np.isnan(frameTimes)):
    continue
...
except Exception as e:
    continue
```

iii. The AI uses aggressive NaN filling (nearest-neighbor interpolation) rather than preserving missing-ness. The reference keeps NaN and assigns a "not visible" class. The broad exception handling risks masking real bugs.

## 11-a. What are the most time-consuming steps of the code?

i. The main bottleneck is the per-trial loop for spike binning: the AI bins spikes one cluster at a time, one trial at a time, using `np.histogram` inside a double loop (over clusters and trials). This is much slower than the reference's vectorized `np.histogram2d` across all trials at once.

ii.
```python
for ci, clu_idx in enumerate(cluid):
    ...
    for trial_num in range(1, Ntrials + 1):
        trial_mask = spike_trial_valid == trial_num
        counts, _ = np.histogram(trial_spikes, bins=EDGES)
        fr = counts / PARAMS['dt']
        fr_smooth = causal_gaussian_smooth(fr, PARAMS['smooth'], PARAMS['bctype'])
        trialdat[:, ci, trial_num - 1] = fr_smooth
```

iii. Total processing time was ~189s for 25 sessions. The double loop over clusters and trials is the primary bottleneck.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning double loop (over clusters and trials) could be vectorized using `np.histogram2d` as the reference does. The per-trial smoothing could also be vectorized across trials. The velocity computation loops over all `Ntrials` (not just valid trials), wasting computation.

ii.
```python
# Per-cluster, per-trial loop that could use histogram2d
for ci, clu_idx in enumerate(cluid):
    for trial_num in range(1, Ntrials + 1):
        counts, _ = np.histogram(trial_spikes, bins=EDGES)
```

iii. The reference's `histogram2d` approach bins all trials for a cluster in one call.

## 11-c. What processing does the code repeat multiple times?

i. The video offset (`find_video_offset`) is computed separately for tongue, paw, and motion energy processing — three times per session instead of once. The `interp1d` import is also done inside loops.

ii.
```python
# Called in compute_tongue_velocity:
vidshift = find_video_offset(f)
# Called again in compute_paw_velocity:
vidshift = find_video_offset(f)
# Called again in load_motion_energy:
vidshift = find_video_offset(f)
```

iii. The reference computes the offset once in `Camera.__init__`.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI processes ALL trials (including filtered-out ones) for neural data, then selects valid trials afterward. Spike counts are computed for all `Ntrials` including early-lick, ignore, and stim trials, then only `valid_trials` are kept. Similarly, velocity and motion energy are computed for all trials.

ii.
```python
trialdat = np.zeros((N_TIMEBINS, len(cluid), Ntrials), dtype=np.float32)
for trial_num in range(1, Ntrials + 1):  # processes ALL trials
    ...
# Later, only valid trials are selected:
for trial_idx in valid_trials:
    neural_trials.append(trialdat[:, :, trial_idx].T)
```

iii. This wastes computation on trials that are immediately discarded.
