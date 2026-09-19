# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads each session from a MATLAB `.mat` file (`data_structure_<animal>_<date>.mat`) located in one of two directories (`Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior`). The 44 sessions and their probe assignments are hard-coded in `EPHYS_SESSIONS`. Each file is opened with either `h5py` (for v7.3 HDF5 files) or `scipy.io.loadmat` (for v5 files). Motion energy is loaded from separate `motionEnergy_<animal>_<date>.mat` files. The h5py file handle is kept open during processing and closed afterward.

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

iii. The AI identified the session list from the authors' `load<ANM>_ALMVideo.m` scripts and noted that only Ephys_Behavior and RandomizedDelay_Ephys_Behavior directories contain neural data. CONVERSION_NOTES.md documents that 3 sessions were excluded from randomized delay loaders.

## 1-b. How are the data split into subjects?

i. The animal name is passed directly as a field in the session definition tuple. All unique animal names are collected and sorted to form the `subjects` list, with `subject_idx` mapping each session.

ii.
```python
EPHYS_SESSIONS = [
    ('data/Ephys_Behavior', 'JEB6', '2021-04-18', [2]),
    ...
]
unique_subjects = sorted(set(all_animals))
subject_idx = np.array([unique_subjects.index(a) for a in all_animals])
```

iii. The AI documented 14 unique subjects across both datasets in CONVERSION_NOTES.md.

## 1-c. How are the data split into sessions?

i. Each entry in `EPHYS_SESSIONS` is one session, identified by (directory, animal, date, probes). Each becomes one element of `neural`, `input`, and `output`. 44 sessions total: 25 fixed-delay and 19 randomized-delay.

ii.
```python
for sess_idx, (dirpath, animal, date, probes) in enumerate(sessions):
    result = process_session(dirpath, animal, date, probes, ...)
```

iii. The AI identified sessions from the authors' loader scripts and noted which sessions were excluded.

## 1-d. How are the data split into trials?

i. Trials are indexed by the `Ntrials` field in `obj.bp`. Each behavioral variable is read as a flat array of length `Ntrials`. Trial indices are 0-based internally (with spike data using 1-based trial numbers). Spikes carry their trial assignment in `clu.trial`.

ii.
```python
def get_ntrials(data, fmt):
    if fmt == 'h5':
        return int(data['obj']['bp']['Ntrials'][0, 0])
    else:
        return int(data['obj']['bp'][0, 0]['Ntrials'][0, 0])
```

iii. The AI reads the total trial count from the Bpod structure and uses it to index all per-trial variables.

## 1-e. How are trials filtered based on quality controls?

i. Three filters are applied: (1) early-lick trials (`bp.early`) are excluded, (2) photostimulation trials (`bp.stim.enable`) are excluded, (3) trials where the alignment time (go cue) is NaN or zero are excluded, (4) trials beyond the last trial with spike data are excluded. Additionally, sessions with fewer than 10 neurons or fewer than 2 valid trials are skipped entirely.

ii.
```python
valid_trials = ~stim_enable & ~early
valid_trials &= ~np.isnan(align_times) & (align_times > 0)
...
if all_clusters:
    max_spike_trial = max(int(np.max(c['trial'])) for c in all_clusters ...)
    valid_trial_indices = valid_trial_indices[valid_trial_indices < max_spike_trial]
...
if len(all_clusters) < MIN_UNITS:
    return None
```

iii. CONVERSION_NOTES.md documents these filters and notes the exclusion of trials beyond recording extent in sessions 36 and 43.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Spike-sorted cluster data from `obj.clu{probe}`: specifically `trialtm` (spike times relative to trial start), `trial` (trial assignment), and `quality` (manual curation label). Go cue times from `bp.ev.goCue` are used for alignment.

ii.
```python
trialtm = f[trialtm_ref][:].flatten()
trial = f[trial_ref][:].flatten().astype(int)
...
aligned_times = trialtm[spike_mask] - align_times[j]
counts = np.histogram(aligned_times, bins=edges)[0]
```

iii. Documented in CONVERSION_NOTES.md Step 1.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue, binned into 10 ms bins (DT=1/100), converted to firing rates by dividing by DT, then smoothed with a **causal** Gaussian kernel of window size N=15. The causal kernel zeros out the first half of a standard Gaussian window.

ii.
```python
DT = 1.0 / 100  # 10 ms time bins
SMOOTH_N = 15

def causal_gaussian_smooth(x, N, bctype='reflect'):
    t = np.arange(N)
    mu = (N - 1) / 2
    sigma = N / 6
    kernel = np.exp(-0.5 * ((t - mu) / sigma) ** 2)
    kernel[:int(np.ceil(N / 2))] = 0  # Make causal
    kernel = kernel / kernel.sum()
    ...

rate = counts.astype(np.float64) / DT
smoothed = causal_gaussian_smooth(rate, SMOOTH_N, SMOOTH_BC)
```

iii. CONVERSION_NOTES.md says "Causal Gaussian smoothing matching getSeq.m" and "dt=1/100=10ms, as in WorkingWithDataObjs.m."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) cluster quality labels matching `{'garbage', 'gabrga', 'noisy', 'real?'}` are excluded (case-insensitive, stripped), (2) neurons with mean firing rate <= 1 Hz are removed. Sessions with fewer than 10 units after filtering are skipped.

ii.
```python
EXCLUDE_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR_THRESHOLD = 1.0
MIN_UNITS = 10

quality = h5_deref_string(f, q_ref).strip().lower()
if quality in EXCLUDE_QUALITIES:
    continue
...
mean_frs = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
keep = mean_frs > LOW_FR_THRESHOLD
```

iii. CONVERSION_NOTES.md documents quality filter from `findClusters.m` and low FR from the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times (`trialtm`) are aligned to go cue by subtraction: `aligned_times = trialtm - goCue[trial]`. This is done per-neuron, per-trial in a nested loop.

ii.
```python
for j in range(ntrials):
    trial_num = j + 1
    spike_mask = trial == trial_num
    aligned_times = trialtm[spike_mask] - align_times[j]
    counts = np.histogram(aligned_times, bins=edges)[0]
```

iii. CONVERSION_NOTES.md says "Align spike times to event (goCue): trialtm_aligned = trialtm - event_time."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 10 ms bins (DT = 1/100 = 0.01 s), resulting in 500 time bins spanning -2.5 to 2.5 s. No rebinning is applied — spikes are directly binned at this resolution.

ii.
```python
DT = 1.0 / 100  # 10 ms time bins
TMIN = -2.5
TMAX = 2.5

def compute_time_axis():
    edges = np.arange(TMIN, TMAX + DT/2, DT)
    time_axis = edges[:-1] + DT / 2
    return time_axis, edges
```

iii. CONVERSION_NOTES.md says "dt = 1/100 = 10ms, as in WorkingWithDataObjs.m and consistent with main analysis." The AI chose 10 ms over 5 ms, noting the ambiguity between `WorkingWithDataObjs.m` (dt=1/100) and `getDefaultParams.m` (dt=1/200).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Not derived from raw data variables. The input is the time axis itself — the center of each time bin, computed from the binning parameters.

ii.
```python
def compute_time_axis():
    edges = np.arange(TMIN, TMAX + DT/2, DT)
    time_axis = edges[:-1] + DT / 2
    return time_axis, edges
...
input_data = time_axis.astype(np.float32).reshape(1, -1)
```

iii. No justification needed — this is the defined time axis.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is computed as bin centers: `edges[:-1] + DT/2`. It is the same for all trials and sessions.

ii.
```python
edges = np.arange(TMIN, TMAX + DT/2, DT)
time_axis = edges[:-1] + DT / 2
```

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is the bin-center grid of the neural data itself. The same edges used to bin spikes define the time axis input.

ii.
```python
counts = np.histogram(aligned_times, bins=edges)[0]
...
input_data = time_axis.astype(np.float32).reshape(1, -1)
```

iii. N/A

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derives lick direction from `obj.bp.R` (right-trial indicator) only. It does NOT use hit/miss to infer actual lick direction.

ii.
```python
R = get_bp_field(data, fmt, 'R').astype(bool)
...
lick_direction = R_valid.astype(np.float32)
```

iii. CONVERSION_NOTES.md initially considered deriving actual lick direction from hit/miss + R/L but ultimately decided to use R/L directly as "instruction direction," reasoning that "the outcome variable captures whether they got it right."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A direct cast of the R flag to float: R=1 (right), not-R=0 (left). Only 2 classes. No "no lick" class for ignore trials.

ii.
```python
lick_direction = R_valid.astype(np.float32)
...
'output_values': [
    ['left', 'right'],  # lick_direction: 0=left, 1=right
]
```

iii. The AI chose to encode instruction direction rather than actual lick direction, and omitted the "no lick" class.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `obj.bp.autowater` — trials where autowater=1 are WC context, others are DR.

ii.
```python
autowater = get_bp_field(data, fmt, 'autowater').astype(bool)
behavioral_context = (~autowater_valid).astype(np.float32)
```

iii. Documented in CONVERSION_NOTES.md.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct negation: autowater=True maps to WC=0, autowater=False maps to DR=1.

ii.
```python
behavioral_context = (~autowater_valid).astype(np.float32)
```

iii. Matches the specification WC=0, DR=1.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Only `obj.bp.hit` is used. Hit trials become correct (1), everything else (miss + ignore) becomes incorrect (0).

ii.
```python
hit = get_bp_field(data, fmt, 'hit').astype(bool)
outcome = hit_valid.astype(np.float32)
```

iii. CONVERSION_NOTES.md says "Outcome: correct (hit=1) -> 1, incorrect (miss=1 or no=1) -> 0."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct cast of hit to float: hit=1 is correct, everything else is incorrect. Only 2 classes — no separate "ignore" class. Ignore trials are lumped with incorrect.

ii.
```python
outcome = hit_valid.astype(np.float32)
...
'output_values': [
    ...
    ['incorrect', 'correct'],  # outcome: 0=incorrect, 1=correct
]
```

iii. The instructions specify three outcome classes: incorrect, correct, ignore. The AI used only two.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. DLC tracking from the side camera (view=0), feature `'tongue'`. Uses `obj.traj` with `ts` (tracked positions), `frameTimes`, and `featNames`.

ii.
```python
tongue_speed = extract_velocity_from_traj(
    data, fmt, view=0, feat_name='tongue',
    ntrials=ntrials, align_times=align_times,
    time_axis=time_axis, vidshift=vidshift
)
```

iii. CONVERSION_NOTES.md: "Tongue velocity: Compute from... tongue if available."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For tongue: positions are NOT smoothed (the `is_tongue` branch copies x,y without smoothing). Velocity is computed via `np.gradient` on raw positions. NaN velocities are set to 0 for tongue. Speed = sqrt(xvel^2 + yvel^2). NaN values are filled with nearest neighbor interpolation. The speed is then interpolated (`np.interp`) from frame times to the neural time axis. Only one camera (side) is used.

ii.
```python
is_tongue = 'tongue' in feat_name.lower()
if not is_tongue:
    xpos_smooth = causal_gaussian_smooth(xpos, 21, 'reflect')
    ypos_smooth = causal_gaussian_smooth(ypos, 21, 'reflect')
else:
    xpos_smooth = xpos.copy()
    ypos_smooth = ypos.copy()

xvel = np.gradient(xpos_smooth)
yvel = np.gradient(ypos_smooth)

if is_tongue:
    xvel[np.isnan(xvel)] = 0
    yvel[np.isnan(yvel)] = 0

spd = np.sqrt(xvel**2 + yvel**2)
...
interpolated = np.interp(time_axis, aligned_ft, spd)
```

iii. The AI chose not to smooth tongue positions and to set NaN velocities to 0 rather than marking them as "not visible."

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Discretized at the session 50th percentile into 2 classes: 0 (below threshold) and 1 (at or above threshold). No "not visible" (class 2) category.

ii.
```python
tongue_thresh = np.nanpercentile(tongue_speed_valid, 50)
tongue_disc = discretize_time_series(tongue_speed_valid, tongue_thresh)

def discretize_time_series(values, threshold):
    if threshold < 1e-10:
        threshold = 1e-10
    return (values >= threshold).astype(np.float32)
```

iii. The instructions specify 3 categories (0: <50th, 1: >=50th, 2: not visible). The AI omitted class 2.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are aligned by subtracting the video offset and go cue time, then `np.interp` interpolates the speed from camera frame times to the neural time axis.

ii.
```python
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
interpolated = np.interp(time_axis, aligned_ft, spd)
```

iii. The video offset is computed using `np.nanmedian` rather than the reference code's `mode`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. DLC tracking from the bottom camera (view=1), feature `'top_paw'`.

ii.
```python
paw_speed = extract_velocity_from_traj(
    data, fmt, view=1, feat_name='top_paw',
    ntrials=ntrials, align_times=align_times,
    time_axis=time_axis, vidshift=vidshift
)
```

iii. Documented in CONVERSION_NOTES.md.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For paw (non-tongue features): positions are smoothed with a causal Gaussian (N=21), velocity is computed via `np.gradient`, median baseline is subtracted from velocity, speed = sqrt(xvel^2 + yvel^2). NaN values are filled with nearest neighbor. Speed is interpolated to the neural time axis.

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

iii. The AI applies a causal Gaussian to paw positions and subtracts median velocity as baseline.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: discretized at session 50th percentile into 2 classes (0/1). No "not visible" (class 2) category.

ii.
```python
paw_thresh = np.nanpercentile(paw_speed_valid, 50)
paw_disc = discretize_time_series(paw_speed_valid, paw_thresh)
```

iii. The instructions specify 3 categories including "not visible." The AI omitted it.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: frame times corrected by video offset and go cue, then interpolated to neural time axis with `np.interp`.

ii.
```python
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
interpolated = np.interp(time_axis, aligned_ft, spd)
```

iii. Same approach as tongue velocity alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Loaded from separate `motionEnergy_<animal>_<date>.mat` files. Handles multiple file formats (struct with data/moveThresh, direct cell array, nested struct).

ii.
```python
def load_motion_energy(dirpath, animal, date):
    me_file = os.path.join(dirpath, f'motionEnergy_{animal}_{date}.mat')
    me_mat = scipy.io.loadmat(me_file, squeeze_me=False)
    me_raw = me_mat['me']
    # Handles Format 1 (struct), Format 2 (cell array), Format 3 (nested struct)
    ...
```

iii. Documented in CONVERSION_NOTES.md.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The per-trial motion energy trace is interpolated from camera frame times to the neural time axis using `np.interp`. NaN values are filled with nearest neighbor or zeros. Then discretized at session 50th percentile.

ii.
```python
me_interp[:, trial_idx] = np.interp(
    time_axis, aligned_ft[:n_frames], me_trial[:n_frames]
).astype(np.float32)
...
me_thresh_50 = np.nanpercentile(me_valid, 50)
me_disc = discretize_time_series(me_valid, me_thresh_50)
```

iii. Documented in CONVERSION_NOTES.md.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same as other continuous outputs: 2 classes split at session 50th percentile. No "no video" (class 2) category.

ii.
```python
me_disc = discretize_time_series(me_valid, me_thresh_50)
...
'output_values': [
    ...
    ['low', 'high'],  # motion_energy: 0=<50th, 1=>=50th
]
```

iii. The instructions specify 3 categories including "no video." The AI omitted it.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned using the side camera's frame times, corrected by the video offset and go cue, then interpolated to the neural time axis.

ii.
```python
ts, frame_times, _ = get_traj_data(data, fmt, 0, trial_idx)  # side cam
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
me_interp[:, trial_idx] = np.interp(
    time_axis, aligned_ft[:n_frames], me_trial[:n_frames]
).astype(np.float32)
```

iii. Documented in CONVERSION_NOTES.md.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple strategies: (1) If frame times are all NaN, synthetic frame times at 400 Hz are created with a fixed 0.5s offset. (2) NaN velocities for tongue are set to 0. (3) NaN values in speed arrays are filled with nearest-neighbor interpolation. (4) Entirely NaN trial columns are filled with zeros. (5) Trials beyond recording extent are excluded. (6) Sessions with too few neurons or trials are skipped.

ii.
```python
if len(frame_times) == 0 or np.all(np.isnan(frame_times)):
    frame_times = np.arange(n_frames) / 400.0
    ft_offset = 0.5
...
if is_tongue:
    xvel[np.isnan(xvel)] = 0
    yvel[np.isnan(yvel)] = 0
...
nan_cols = np.all(np.isnan(speed), axis=0)
speed[:, nan_cols] = 0.0
```

iii. The AI's approach fills missing data with zeros or nearest-neighbor values rather than using a dedicated "not visible" class.

## 11-a. What are the most time-consuming steps of the code?

i. Spike binning is the most time-consuming step because it loops over every neuron and every trial individually. The AI reports ~6-9 seconds per session, with total conversion taking ~269 seconds for 44 sessions.

ii.
```python
for i, clu in enumerate(clusters):
    for j in range(ntrials):
        spike_mask = trial == trial_num
        aligned_times = trialtm[spike_mask] - align_times[j]
        counts = np.histogram(aligned_times, bins=edges)[0]
```

iii. CONVERSION_NOTES.md: "Spike binning loops over neurons and trials (vectorized with np.histogram)" — though the loop is only partially vectorized (per-bin histogram is vectorized, but the neuron x trial loop is not).

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning double loop (over neurons and trials) could be vectorized using `np.histogram2d` to count all trials at once for each neuron, as the reference code does. The velocity extraction also loops per-trial but this is harder to vectorize due to variable frame counts.

ii.
```python
# Current: nested loop
for i, clu in enumerate(clusters):
    for j in range(ntrials):
        counts = np.histogram(aligned_times, bins=edges)[0]

# Reference uses: np.histogram2d(spike_trial, spike_time, bins=[trial_edges, BIN_EDGES])
```

iii. CONVERSION_NOTES.md acknowledges the loop but doesn't discuss vectorization opportunities.

## 11-c. What processing does the code repeat multiple times?

i. Frame times and trajectory data are loaded multiple times for the same trial — once for tongue velocity and once for motion energy alignment — because `get_traj_data` is called independently in each extraction function. The video offset could also be reused but is computed once.

ii.
```python
# Called for tongue:
ts, frame_times, feat_names = get_traj_data(data, fmt, view=0, trial_idx)
# Called again for motion energy:
ts, frame_times, _ = get_traj_data(data, fmt, 0, trial_idx)
```

iii. Not discussed in CONVERSION_NOTES.md.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code bins and smooths spikes for ALL trials (including filtered ones) before selecting valid trials. The `bin_and_smooth_spikes` function processes all `ntrials` trials, and only afterward are `valid_trial_indices` used to select the subset. Similarly, velocity is extracted for all trials before filtering.

ii.
```python
trialdat = bin_and_smooth_spikes(all_clusters, ntrials, align_times, time_axis, edges)
...
trialdat_valid = trialdat[:, :, valid_trial_indices]  # Only valid trials selected here
...
tongue_speed_valid = tongue_speed[:, valid_trial_indices]
```

iii. Not discussed in CONVERSION_NOTES.md. This is a significant inefficiency since ~10% of trials are filtered out.
