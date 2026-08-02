# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from two directories: `Ephys_Behavior/` (25 sessions) and `RandomizedDelay_Ephys_Behavior/` (19 sessions), totaling 44 sessions. Each session has a `data_structure_ANM_DATE.mat` file loaded via `mat73.loadmat()` (for HDF5/v7.3 format) or a custom `_load_v5_session()` using `scipy.io.loadmat()` (for MATLAB v5 format). Motion energy is loaded separately from `motionEnergy_ANM_DATE.mat` files via `scipy.io.loadmat()`. The AI hardcodes all session identifiers (animal, date, probe numbers, data directory) in two lists: `EPHYS_SESSIONS` and `RANDOMIZED_DELAY_SESSIONS`, derived from the reference loading scripts.

ii.
```python
ALL_SESSIONS = EPHYS_SESSIONS + RANDOMIZED_DELAY_SESSIONS

def load_session_data(anm, date, data_dir):
    data_path = os.path.join(DATA_ROOT, data_dir, f'data_structure_{anm}_{date}.mat')
    try:
        obj = mat73.loadmat(data_path)['obj']
    except TypeError:
        obj = _load_v5_session(data_path)
    # Load motion energy
    me_path = os.path.join(DATA_ROOT, data_dir, f'motionEnergy_{anm}_{date}.mat')
    ...
```

iii. The AI documented in CONVERSION_NOTES.md Step 2 that the data was organized in the two data directories, and in Step 1 that `loadObjs.m` and per-animal loading scripts were the reference for session selection. The AI explicitly excluded behavior-only sessions (MAH mice) and sessions not listed in loading scripts (JEB23_2023-10-20, JEB24_2023-10-03, JEB24_2023-10-04).

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the animal name string (e.g., 'EKH1', 'JEB7') from the hardcoded session registry. The AI collects unique animal names across all processed sessions and builds a sorted `subjects` list. A `subject_idx` array maps each session to its subject index. The AI identifies 14 unique subjects (10 from Ephys_Behavior, 4 from RandomizedDelay).

ii.
```python
subjects_set = set()
for i, (anm, date, probes, data_dir) in enumerate(sessions):
    ...
    all_animals.append(anm)
    subjects_set.add(anm)
subjects = sorted(subjects_set)
subject_idx = np.array([subjects.index(anm) for anm in all_animals], dtype=np.int64)
```

iii. The AI noted in CONVERSION_NOTES.md Step 2 that there were 10 mice from Ephys_Behavior and 4 from RandomizedDelay. The AI acknowledged a discrepancy with the paper (which reports 9 DR mice) in Step 9, attributing it to "minor discrepancy."

## 1-c. How are the data split into sessions?

i. Each `.mat` file constitutes one session. The AI processes each session independently via the `process_session()` function, which loads the data, processes neural and behavioral variables, and returns per-trial arrays. Sessions with fewer than 2 valid trials or fewer than 10 neurons after filtering are skipped.

ii.
```python
for i, (anm, date, probes, data_dir) in enumerate(sessions):
    result = process_session(anm, date, probes, data_dir, ...)
    if result is None:
        continue
    all_neural.append(result['neural'])
    ...
```

iii. The AI documented in CONVERSION_NOTES.md Step 3 that the paper requires "at least 10 units" per session for inclusion.

## 1-d. How are the data split into trials?

i. Within each session, `obj.bp.Ntrials` gives the total number of trials. The AI creates boolean masks for trial properties (R, L, hit, miss, no, autowater, early, stim.enable) and selects valid trials based on filtering criteria. Each valid trial produces one entry in the neural, input, and output lists.

ii.
```python
ntrials_total = int(bp['Ntrials'])
R = np.array(bp['R']).flatten().astype(bool)
L = np.array(bp['L']).flatten().astype(bool)
hit = np.array(bp['hit']).flatten().astype(bool)
miss = np.array(bp['miss']).flatten().astype(bool)
no = np.array(bp['no']).flatten().astype(bool)
autowater = np.array(bp['autowater']).flatten().astype(bool)
early = np.array(bp['early']).flatten().astype(bool)
```

iii. The AI documented in CONVERSION_NOTES.md Step 5 that trials are indexed from 1 in MATLAB and that each trial has associated behavioral metadata.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies the following trial filters: exclude early lick trials (`early=1`), exclude stim trials (`stim.enable=1`), exclude no-response/ignore trials (`no=1`), and require a lick response (`hit | miss`). Notably, the AI keeps both hit (correct) and miss (incorrect) trials, unlike the reference code default conditions which only keep hit trials.

ii.
```python
valid_mask = ~early & ~stim_enable & ~no
valid_mask = valid_mask & (hit | miss)
valid_trials = np.where(valid_mask)[0]
```

iii. The AI documented in CONVERSION_NOTES.md Steps 3 and 5: "Trial filtering: exclude early, stim.enable trials" and "Exclude early, stim.enable, and no-response trials. Keep hit and miss." The inclusion of miss trials is justified by the decoder task requiring an "Outcome" output (correct vs incorrect).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `obj.clu`, which contains sorted spike clusters. For each cluster, the spike times are in `clu.trialtm` (time within trial) and `clu.trial` (trial assignment). The AI selects specific probe(s) per session based on the hardcoded session registry (matching the reference loading scripts' probe assignments for ALM).

ii.
```python
clu_data = obj['clu']
for probe_num in probes:
    probe_idx = probe_num - 1
    clu_probe = clu_data[probe_idx]
    ...
    trial_arr = np.array(clu_probe['trial'][clu_idx]).flatten()
    trialtm_arr = np.array(clu_probe['trialtm'][clu_idx]).flatten()
```

iii. The AI documented in CONVERSION_NOTES.md Step 1 that `obj.clu` contains spike data with fields quality, site, tm, trial, trialtm, and spkWavs.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to the go cue by subtracting `goCue` time from each spike's `trialtm`. Aligned spikes are binned into 10 ms bins (dt=1/100) over a [-2.5, 2.5] s window using `np.histogram`. Bin counts are converted to firing rates (counts/dt) and smoothed with a causal Gaussian kernel (window=15 bins, bctype='reflect').

ii.
```python
DT = 1.0 / 100  # 10 ms bins
SMOOTH_WINDOW = 15
BC_TYPE = 'reflect'

aligned = trialtm_arr[spk_mask] - align_times_all[j]
counts, _ = np.histogram(aligned, bins=edges)
fr = counts.astype(np.float32) / DT
trialdat[:, neuron_offset + i, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)
```

iii. The AI documented in CONVERSION_NOTES.md Step 1 the pipeline: "Align spikes to goCue: trialtm_aligned = trialtm - goCue_time. Bin spikes: histc(trialtm_aligned, edges). Smooth: mySmooth(N/dt, smooth, bctype)."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering: (1) Cluster quality filtering excludes clusters labeled 'garbage', 'gabrga', 'noisy', or 'real?'. (2) Low firing rate filtering removes neurons with mean firing rate <= 1 Hz across all trials and time points.

ii.
```python
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR = 1.0

def get_valid_cluster_indices(clu_probe, excluded_qualities=EXCLUDED_QUALITIES):
    for i, q in enumerate(qualities):
        if q_stripped.lower() not in {e.lower() for e in excluded_qualities}:
            valid.append(i)

def remove_low_fr_neurons(trialdat, low_fr):
    mean_fr = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
    keep = mean_fr > low_fr
    return trialdat[:, keep, :], keep
```

iii. The AI documented in CONVERSION_NOTES.md Step 3: "Neuron curation: Exclude quality labels: garbage, gabrga, noisy, real?. Then remove neurons with mean FR < 1 Hz." The AI cited the paper: "firing rates exceeding 1 Hz."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's trial-relative time (`trialtm`) has the go cue onset time (`goCue`) subtracted from it. The resulting aligned times are then binned into histogram bins centered on the go cue. The time axis is defined as `edges[:-1] + dt/2`, ranging from approximately -2.495 to 2.495 s.

ii.
```python
ALIGN_EVENT = 'goCue'
goCue = np.array(ev['goCue']).flatten()
align_times_all = goCue
aligned = trialtm_arr[spk_mask] - align_times_all[j]
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
counts, _ = np.histogram(aligned, bins=edges)
```

iii. The AI documented in CONVERSION_NOTES.md Step 1: "alignEvent = 'goCue'" and "Align spikes to goCue: trialtm_aligned = trialtm - goCue_time."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 10 ms (dt = 1/100 s), producing 500 time bins over the [-2.5, 2.5] s window. No rebinning is applied; spikes are directly binned at this resolution from raw spike times. The bin centers define the time axis.

ii.
```python
DT = 1.0 / 100  # 10 ms bins
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
```

iii. The AI documented in CONVERSION_NOTES.md Step 4: "Use 1/100 (10 ms) as in WorkingWithDataObjs.m" and Step 5: "Bin size 10 ms (dt=1/100): As in WorkingWithDataObjs.m, gives time axis from -2.5 to 2.5 s = 500 time bins."

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is derived from the time axis computed from the binning parameters (TMIN, TMAX, DT), not directly from a raw data variable. It represents the center of each time bin relative to the go cue onset.

ii.
```python
input_trials.append(time_axis.reshape(1, -1).astype(np.float32))
```

iii. The AI documented in CONVERSION_NOTES.md Step 5: "Continuous time axis, ranges from -2.5 to 2.5 s."

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is computed as bin centers: `edges[:-1] + dt/2`. This produces values from approximately -2.495 to 2.495 s in 0.01 s increments. The same time vector is used for every trial (since all trials have the same window).

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
input_trials.append(time_axis.reshape(1, -1).astype(np.float32))
```

iii. No specific justification given beyond it being a straightforward time axis computation.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis is identical to the neural data time axis, since both use the same bin centers computed from the same edges. The neural data is binned using these edges, and the input is exactly these bin centers. Both are implicitly aligned to the go cue onset (t=0).

ii.
```python
# Same time_axis used for both:
time_axis = edges[:-1] + DT / 2
# Neural: binned using edges
counts, _ = np.histogram(aligned, bins=edges)
# Input: the time axis itself
input_trials.append(time_axis.reshape(1, -1).astype(np.float32))
```

iii. No specific justification needed; alignment is inherent in the construction.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from four behavioral variables: `obj.bp.R` (right trial indicator), `obj.bp.L` (left trial indicator), `obj.bp.hit` (correct response), and `obj.bp.miss` (incorrect response).

ii.
```python
R = np.array(bp['R']).flatten().astype(bool)
L = np.array(bp['L']).flatten().astype(bool)
hit = np.array(bp['hit']).flatten().astype(bool)
miss = np.array(bp['miss']).flatten().astype(bool)
lick_right = (R & hit) | (L & miss)
lick_direction = lick_right[valid_trials].astype(np.int32)
```

iii. The AI documented in CONVERSION_NOTES.md Step 5: "R&hit or L&miss -> right(1); L&hit or R&miss -> left(0)."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI determines actual lick direction (not instructed direction) by combining trial type (R/L) with outcome (hit/miss). On a hit (correct) trial, the mouse licked in the instructed direction. On a miss (incorrect) trial, the mouse licked in the opposite direction. Thus: right lick = (R&hit) | (L&miss), left lick = (L&hit) | (R&miss). The result is broadcast as a per-trial constant across all time bins (shape: (6, n_time) with lick_direction at index 0).

ii.
```python
lick_right = (R & hit) | (L & miss)
lick_direction = lick_right[valid_trials].astype(np.int32)
out[0, :] = lick_direction[t_idx]  # broadcast per-trial to time
```

iii. The AI documented in CONVERSION_NOTES.md Step 5 the mapping and noted it as per-trial output.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj.bp.autowater`, a per-trial boolean flag indicating whether the trial was a water-cued (WC) trial.

ii.
```python
autowater = np.array(bp['autowater']).flatten().astype(bool)
context = (~autowater[valid_trials]).astype(np.int32)
```

iii. The AI documented in CONVERSION_NOTES.md Step 5: "autowater=1 -> WC(0); autowater=0 -> DR(1)."

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Simple boolean inversion: autowater=True maps to WC=0, autowater=False maps to DR=1. Broadcast as per-trial constant across all time bins.

ii.
```python
context = (~autowater[valid_trials]).astype(np.int32)
out[1, :] = context[t_idx]
```

iii. The AI documented in CONVERSION_NOTES.md Step 5 and noted that randomized delay sessions have context=DR for all trials.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit`, a per-trial boolean flag.

ii.
```python
hit = np.array(bp['hit']).flatten().astype(bool)
outcome = hit[valid_trials].astype(np.int32)
```

iii. The AI documented in CONVERSION_NOTES.md Step 5: "hit -> correct(1); miss -> incorrect(0)."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct mapping: hit=True maps to correct=1, hit=False (which is miss, given the trial filter requires hit|miss) maps to incorrect=0. Broadcast as per-trial constant across all time bins.

ii.
```python
outcome = hit[valid_trials].astype(np.int32)
out[2, :] = outcome[t_idx]
```

iii. Straightforward mapping documented in CONVERSION_NOTES.md Step 5.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from `obj.traj[1]` (bottom camera view), specifically the `top_tongue` feature's x,y coordinates from the DeepLabCut tracking data (`traj.ts[:, 0:2, tongue_idx]`), along with `traj.frameTimes` for temporal alignment.

ii.
```python
traj_bottom = obj['traj'][1]  # bottom cam
# Find tongue feature index (top_tongue)
for i, name in enumerate(feat_names):
    if name == 'top_tongue':
        tongue_idx = i
        break
x = ts[:, 0, tongue_idx].copy()
y = ts[:, 1, tongue_idx].copy()
```

iii. The AI documented in CONVERSION_NOTES.md Step 5: "Tongue velocity: Use tip-of-tongue displacement from bottom cam view, compute velocity as 1st derivative, take magnitude."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI computes tongue velocity at the native 400 Hz video frame rate: (1) extracts x,y position of `top_tongue` from bottom camera DLC data, (2) computes gradient (first derivative) at 400 Hz, scaled by VIDEO_FR to get pixels/second, (3) computes speed as sqrt(vx^2 + vy^2), (4) sets speed to NaN where tongue is not visible (NaN positions), (5) interpolates speed to neural time axis using `interp1d`, (6) sets remaining NaN values to 0 (tongue not visible = no movement).

ii.
```python
vx_all = np.gradient(x) * VIDEO_FR
vy_all = np.gradient(y) * VIDEO_FR
speed = np.sqrt(vx_all**2 + vy_all**2)
speed[~valid] = np.nan
# Interpolate to neural time axis
f_interp = interp1d(old_time, speed, kind='linear', bounds_error=False, fill_value=np.nan)
tongue_vel[:, trix] = f_interp(time_axis)
# Set NaN to 0
tongue_vel = np.nan_to_num(tongue_vel, nan=0.0)
```

iii. The AI documented in CONVERSION_NOTES.md Step 6: "Tongue NaN values NOT filled (per paper methods), velocity computed only where visible."

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Tongue velocity is discretized using per-session 50th percentile thresholding: values below the threshold are labeled 0 ("low"), values at or above are labeled 1 ("high"). If the threshold is 0 (many zeros from tongue-invisible periods), it is set to a tiny epsilon to ensure only exact zeros are "low."

ii.
```python
def discretize_per_session(data_2d, percentile=50):
    all_vals = data_2d[~np.isnan(data_2d)]
    threshold = np.percentile(all_vals, percentile)
    if threshold == 0:
        threshold = np.finfo(np.float32).eps
    result = (data_2d >= threshold).astype(np.int32)
    return result

tongue_vel_disc = discretize_per_session(tongue_vel_valid)
```

iii. The AI followed the decoder task instructions: "discretized into two bins with per-session threshold: 0: < 50th percentile, 1: >= 50th percentile."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue velocity is aligned to the neural time axis by computing video frame times relative to the go cue, accounting for the video-neural offset (vidshift): `old_time = frameTimes - vidshift - alignTime`. The speed signal is then linearly interpolated from these aligned frame times to the neural time axis bin centers.

ii.
```python
ft = np.array(traj_bottom['frameTimes'][trix]).flatten()
old_time = ft - vidshift - align_times[trix]
f_interp = interp1d(old_time, speed, kind='linear', bounds_error=False, fill_value=np.nan)
tongue_vel[:, trix] = f_interp(time_axis)
```

iii. The AI documented in CONVERSION_NOTES.md Step 1 the video offset computation: "vidshift = mode(sglx.bitcode.bitstart)/sglx.fs - mode(bp.ev.bitStart)."

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from `obj.traj[1]` (bottom camera), using all features with 'paw' in the name (typically `top_paw` and `bottom_paw`), their x,y coordinates from DLC tracking, and `frameTimes` for alignment.

ii.
```python
traj_bottom = obj['traj'][1]  # bottom cam
paw_indices = []
for i, name in enumerate(feat_names):
    if 'paw' in name.lower():
        paw_indices.append(i)
```

iii. The AI documented in CONVERSION_NOTES.md Step 5: "Paw velocity: Use paw position from bottom cam, compute velocity, take magnitude."

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw feature: (1) extract x,y positions, (2) fill NaN values with nearest neighbor (unlike tongue), (3) compute gradient at 400 Hz scaled by VIDEO_FR, (4) compute speed as sqrt(vx^2+vy^2). Speeds from multiple paw features are averaged. The averaged speed is interpolated to the neural time axis. Remaining NaN values are filled with nearest neighbor.

ii.
```python
for pidx in paw_indices:
    x = ts[:, 0, pidx].copy()
    y = ts[:, 1, pidx].copy()
    # Fill NaN with nearest
    for arr in [x, y]:
        nans = np.isnan(arr)
        if nans.any() and not nans.all():
            ...
            arr[nans] = arr[valid[nearest]]
    vx = np.gradient(x) * VIDEO_FR
    vy = np.gradient(y) * VIDEO_FR
    speeds.append(np.sqrt(vx**2 + vy**2))
avg_speed = np.mean(speeds, axis=0)
```

iii. The AI documented in CONVERSION_NOTES.md Step 6: "Paw NaN values filled with nearest neighbor before velocity computation."

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue velocity: per-session 50th percentile threshold. Values below → 0 ("low"), at or above → 1 ("high").

ii.
```python
paw_vel_disc = discretize_per_session(paw_vel_valid)
```

iii. Follows decoder task instructions for 50th percentile discretization.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same alignment method as tongue velocity: frame times adjusted by vidshift and go cue time, then interpolated to neural time axis.

ii.
```python
ft = np.array(traj_bottom['frameTimes'][trix]).flatten()
old_time = ft - vidshift - align_times[trix]
f_interp = interp1d(old_time, avg_speed, kind='linear', bounds_error=False, fill_value=np.nan)
paw_vel[:, trix] = f_interp(time_axis)
```

iii. Same justification as tongue velocity alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from separate `motionEnergy_ANM_DATE.mat` files. The raw data is in `me.data` (a cell array of per-trial motion energy time series at 400 Hz) and `me.moveThresh` (a per-session threshold). The `me.data` field may be nested inside a struct (`me.data.data`).

ii.
```python
me_path = os.path.join(DATA_ROOT, data_dir, f'motionEnergy_{anm}_{date}.mat')
me_file = scipy.io.loadmat(me_path)
me_var = me_file['me']
if me_data.dtype.names and 'data' in me_data.dtype.names:
    inner = me_data[0, 0]
    me_raw = inner['data']
    me_thresh = float(inner['moveThresh'].flat[0])
```

iii. The AI documented in CONVERSION_NOTES.md Step 1: "Motion energy aligned: interp1(frameTimes - vidshift - alignTime, me.data, taxis)."

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Per-trial motion energy is interpolated from native 400 Hz video rate to the neural time axis. Video-neural offset is applied. NaN values at trial boundaries are filled with nearest neighbor values. The signal is then discretized.

ii.
```python
def get_motion_energy_aligned(me_raw, obj, align_times, time_axis, vidshift, ntrials):
    for trix in range(ntrials):
        me_trial = np.array(me_raw[trix, 0]).flatten()
        ft = np.array(traj_view0['frameTimes'][trix]).flatten()
        old_time = ft - vidshift - align_times[trix]
        f_interp = interp1d(old_time, me_trial, kind='linear', bounds_error=False, fill_value=np.nan)
        me_aligned[:, trix] = f_interp(time_axis)
    # Fill NaN with nearest
    ...
```

iii. The AI documented in CONVERSION_NOTES.md Step 1 that this matches `loadMotionEnergy.m`: "interp1(frameTimes - vidshift - alignTime, me.data, taxis)" and "fillmissing(,'nearest')."

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized using per-session 50th percentile thresholding, identical to tongue and paw velocity. The reference code instead uses `me.moveThresh` (a manually set per-session threshold separating bimodal distribution modes), but the AI follows the decoder task instructions which specify 50th percentile.

ii.
```python
me_disc = discretize_per_session(me_valid)
```

iii. The AI followed the decoder task instructions: "discretized into two bins with per-session threshold: 0: < 50th percentile, 1: >= 50th percentile."

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy uses the side camera's (`obj.traj[0]`) `frameTimes` for alignment, with the same vidshift and go cue time offset. The signal is interpolated to the neural time axis.

ii.
```python
traj_view0 = obj['traj'][0]  # side cam
ft = np.array(traj_view0['frameTimes'][trix]).flatten()
old_time = ft - vidshift - align_times[trix]
f_interp = interp1d(old_time, me_trial, kind='linear', bounds_error=False, fill_value=np.nan)
me_aligned[:, trix] = f_interp(time_axis)
```

iii. The AI's approach matches the reference `loadMotionEnergy.m` which uses `obj.traj{1}` (MATLAB side cam = view index 1) frameTimes.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several types of missing/problematic data:
- **Tongue NaN (not visible)**: Velocity set to 0 where tongue is not visible, consistent with the paper.
- **Paw/motion energy NaN**: Filled with nearest neighbor values before further processing.
- **MATLAB format differences**: Handles both HDF5 (mat73) and v5 (scipy.io) format files with a try/except fallback.
- **Missing motion energy files**: Returns zeros if file doesn't exist.
- **Missing frameTimes**: Falls back to generating synthetic frame times at 400 Hz.
- **All-zero neural data**: Some trials at session ends have all-zero neural data (sessions 36, 43); these are retained.
- **Sessions with too few trials/neurons**: Skipped if < 2 valid trials or < 10 neurons.

ii.
```python
# Tongue NaN handling
tongue_vel = np.nan_to_num(tongue_vel, nan=0.0)

# Paw NaN filling
for arr in [x, y]:
    nans = np.isnan(arr)
    if nans.any() and not nans.all():
        ...
        arr[nans] = arr[valid[nearest]]

# MATLAB format fallback
try:
    obj = mat73.loadmat(data_path)['obj']
except TypeError:
    obj = _load_v5_session(data_path)

# Fallback for missing frameTimes
except:
    nframes = len(me_trial)
    ft = np.arange(1, nframes + 1) / VIDEO_FR
    old_time = ft - 0.5 - align_times[trix]
```

iii. The AI documented missing data handling in CONVERSION_NOTES.md Step 9 (data warnings about all-zero trials) and Step 6 (tongue NaN policy, MATLAB format handling).

## 11-a. What are the most time-consuming steps of the code?

i. The spike binning step is the most time-consuming, as documented in the conversion output. It involves a triple nested loop over neurons, trials, and spike times. For a session with 85 neurons and ~300 trials, this takes ~2.8s. File loading (mat73/scipy.io) is also significant at 3-9s per session.

ii.
```python
for i, clu_idx in enumerate(valid_clu):
    trial_arr = np.array(clu_probe['trial'][clu_idx]).flatten()
    trialtm_arr = np.array(clu_probe['trialtm'][clu_idx]).flatten()
    for j in range(ntrials_total):
        ...
        counts, _ = np.histogram(aligned, bins=edges)
        fr = counts.astype(np.float32) / DT
        trialdat[:, neuron_offset + i, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)
```

iii. The AI documented timing in the conversion output: e.g., "Spike binning: 2.8s for 85 neurons."

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
- **Spike binning loop**: The inner trial loop could be vectorized by grouping all spikes by trial using `np.searchsorted` or similar, then computing all histograms at once.
- **Causal Gaussian smoothing**: Called per-neuron per-trial; could be vectorized to operate on the full (time x neurons x trials) array at once using `scipy.ndimage.convolve1d`.
- **NaN filling loops**: The nearest-neighbor NaN filling loops over trials could use vectorized operations.
- **Velocity computation loops**: The per-trial tongue/paw velocity loops iterate over trials; position extraction and gradient could be batched.

ii.
```python
# Current: nested loop
for i, clu_idx in enumerate(valid_clu):
    for j in range(ntrials_total):
        ...
        trialdat[:, neuron_offset + i, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)

# Smoothing also loops per column:
for j in range(x_filt.shape[1]):
    out[:, j] = np.convolve(x_filt[:, j], kern, mode='same')
```

iii. The AI noted in CONVERSION_NOTES.md Step 6 efficiency concerns but did not fully vectorize these loops.

## 11-c. What processing does the code repeat multiple times?

i. The code processes neural data for ALL trials (including invalid ones that will be filtered out), then selects only valid trials. This means spike alignment, binning, smoothing, and FR filtering are performed on early lick, stim, and no-response trials that are discarded. Also, the smoothing function is called once per neuron per trial rather than batched.

ii.
```python
# Processes ALL trials
trialdat = np.zeros((n_time, total_neurons, ntrials_total), dtype=np.float32)
for j in range(ntrials_total):  # ALL trials
    ...
# Then filters to valid only
trialdat_valid = trialdat[:, :, valid_trials]
```

iii. No specific justification given for processing all trials before filtering. This simplifies the code but wastes computation.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) Neural data is computed for all trials including invalid ones (early, stim, no-response) then filtered to valid trials only. (2) Behavioral outputs (tongue velocity, paw velocity, motion energy) are also computed for all trials then filtered. (3) The low-FR filtering operates on all trials rather than just valid trials, which could change which neurons are retained. (4) The code loads and processes per-trial motion energy data from the `motionEnergy_*.mat` file even for trials that will be excluded.

ii.
```python
# Compute for all trials
tongue_vel = compute_tongue_velocity(obj, align_times_all, time_axis, vidshift, ntrials_total)
# Then filter
tongue_vel_valid = tongue_vel[:, valid_trials]

# Low-FR filtering on all trials (not just valid)
trialdat, kept_mask = remove_low_fr_neurons(trialdat, LOW_FR)
# Then extract valid
trialdat_valid = trialdat[:, :, valid_trials]
```

iii. No justification given. Processing all trials before filtering is a code simplification choice that trades efficiency for clarity.
