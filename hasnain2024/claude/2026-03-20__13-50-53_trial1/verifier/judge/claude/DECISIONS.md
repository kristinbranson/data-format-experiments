# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads each session from a MATLAB `.mat` file named `data_structure_<anm>_<date>.mat` located in one of two folders (`Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior`). The 44 sessions are hard-coded in two lists (`EPHYS_SESSIONS` and `RANDOMIZED_DELAY_SESSIONS`) with animal name, date, probe list, and data directory. For v7.3 HDF5 files, `mat73.loadmat()` is used; for v5 files, `scipy.io.loadmat()` with a custom `_load_v5_session()` converter. Motion energy is loaded separately from `motionEnergy_<anm>_<date>.mat` via `scipy.io.loadmat()`.

ii.
```python
EPHYS_SESSIONS = [
    ('EKH1', '2021-08-07', [2], 'Ephys_Behavior'),
    ...
]
RANDOMIZED_DELAY_SESSIONS = [
    ('JEB11', '2022-05-10', [1], 'RandomizedDelay_Ephys_Behavior'),
    ...
]
ALL_SESSIONS = EPHYS_SESSIONS + RANDOMIZED_DELAY_SESSIONS

def load_session_data(anm, date, data_dir):
    data_path = os.path.join(DATA_ROOT, data_dir, f'data_structure_{anm}_{date}.mat')
    try:
        obj = mat73.loadmat(data_path)['obj']
    except TypeError:
        obj = _load_v5_session(data_path)
```

iii. The AI documented in CONVERSION_NOTES.md that the session list was derived from the per-animal loading scripts in `code/DataLoadingScripts/Recording and video/`. Excluded sessions are JEB4/JEB5 (no data files), JEB23_2023-10-20 (commented out), and JEB24_2023-10-03/2023-10-04 (not in loading scripts).

## 1-b. How are the data split into subjects?

i. The animal name is passed from the session tuple (first element) through the processing function. At assembly, the sorted unique set of animal names forms the `subjects` list, and each session gets an index into it via `subjects.index(anm)`.

ii.
```python
for i, (anm, date, probes, data_dir) in enumerate(sessions):
    result = process_session(anm, date, probes, data_dir, ...)
    all_animals.append(anm)
    subjects_set.add(anm)

subjects = sorted(subjects_set)
subject_idx = np.array([subjects.index(anm) for anm in all_animals], dtype=np.int64)
```

iii. The animal name is part of the session tuple and is tracked through processing. This produces 14 unique subjects.

## 1-c. How are the data split into sessions?

i. Each entry in `ALL_SESSIONS` (a tuple of animal, date, probes, data_dir) represents one session and maps to one file on disk. Fixed-delay and randomized-delay sessions are combined into a single list. Each session becomes one element of the `neural`, `input`, and `output` lists. 44 sessions total (25 fixed-delay + 19 randomized-delay).

ii.
```python
ALL_SESSIONS = EPHYS_SESSIONS + RANDOMIZED_DELAY_SESSIONS  # 25 + 19 = 44

for i, (anm, date, probes, data_dir) in enumerate(sessions):
    result = process_session(anm, date, probes, data_dir, ...)
    if result is None:
        continue
    all_neural.append(result['neural'])
```

iii. From CONVERSION_NOTES.md: "Include all 44 ephys sessions: Both Ephys_Behavior (25) and RandomizedDelay (19)."

## 1-d. How are the data split into trials?

i. Trials are defined by the per-trial fields of `obj.bp`. The total number of trials comes from `bp['Ntrials']`. Each per-trial field (R, L, hit, miss, no, autowater, early, stim.enable, goCue) is flattened to this length. Trial indexing is 0-based in Python, 1-based in the MATLAB spike data. Valid trials are selected by a boolean mask.

ii.
```python
ntrials_total = int(bp['Ntrials'])
goCue = np.array(ev['goCue']).flatten()
R = np.array(bp['R']).flatten().astype(bool)
...
valid_mask = ~early & ~stim_enable & ~no
valid_mask = valid_mask & (hit | miss)
valid_trials = np.where(valid_mask)[0]
```

iii. The AI reads all per-trial fields directly from the behavioral data structure.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies three filters: (1) early-lick trials (`early=1`) are excluded, (2) photostimulation trials (`stim.enable=1`) are excluded, and (3) **ignore/no-response trials (`no=1`) are excluded**, keeping only trials with a lick response (hit or miss). This is more aggressive filtering than the reference, which keeps ignore trials.

ii.
```python
valid_mask = ~early & ~stim_enable & ~no
valid_mask = valid_mask & (hit | miss)
valid_trials = np.where(valid_mask)[0]
```

iii. From CONVERSION_NOTES.md Step 5: "Trial exclusion: Exclude early, stim.enable, and no-response trials. Keep hit and miss." The AI noted the paper mentions excluding early and stim trials, but also chose to exclude ignore trials, which the reference does not do. The AI does NOT filter trials based on recording length (unlike reference which drops trials past the last spike).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu` spike-sorted clusters, specifically the `trial`, `trialtm`, and `quality` fields per cluster. The go cue times from `bp.ev.goCue` are used for alignment.

ii.
```python
clu_probe = clu_data[probe_idx]
trial_arr = np.array(clu_probe['trial'][clu_idx]).flatten()
trialtm_arr = np.array(clu_probe['trialtm'][clu_idx]).flatten()
aligned = trialtm_arr[spk_mask] - align_times_all[j]
```

iii. Same source variables as the reference code's `alignSpikes.m` and `getSeq.m`.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue, binned into **10 ms bins** (DT=1/100), and smoothed with a **causal Gaussian** kernel of window size 15 bins (std=15/6=2.5 bins). The smoothing zeroes out the first half of the kernel, making it causal. Firing rates are in spikes/s (counts/DT). Units from multiple probes are concatenated.

ii.
```python
DT = 1.0 / 100  # 10 ms bins
SMOOTH_WINDOW = 15
edges = np.arange(TMIN, TMAX + DT, DT)

def causal_gaussian_smooth(x, N, bctype='reflect'):
    kern = np.array(scipy_windows.gaussian(N, std=N/6.0))
    kern[:N//2] = 0  # causal: zero out first half
    kern = kern / kern.sum()
    ...

fr = counts.astype(np.float32) / DT
trialdat[:, neuron_offset + i, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)
```

iii. From CONVERSION_NOTES.md: "Bin size 10 ms (dt=1/100): As in WorkingWithDataObjs.m." The AI chose `WorkingWithDataObjs.m`'s dt=1/100 rather than the `getDefaultParams.m` value of dt=1/200 (5 ms). The causal Gaussian smoothing was the AI's interpretation of `mySmooth.m`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. (1) Quality label filter: excludes clusters with quality labels in `{'garbage', 'gabrga', 'noisy', 'real?'}` (case-insensitive). Note: `'poor'` is NOT excluded (unlike reference). (2) Mean firing rate filter: removes neurons with mean FR <= 1 Hz across all trials and time points.

ii.
```python
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}

def get_valid_cluster_indices(clu_probe, excluded_qualities=EXCLUDED_QUALITIES):
    for i, q in enumerate(qualities):
        if q_stripped.lower() not in {e.lower() for e in excluded_qualities}:
            valid.append(i)

def remove_low_fr_neurons(trialdat, low_fr):
    mean_fr = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
    keep = mean_fr > low_fr
    return trialdat[:, keep, :], keep
```

iii. From CONVERSION_NOTES.md: "Cluster quality: 'all' (exclude garbage, gabrga, noisy, real?)" and "FR threshold = 1 Hz: Matches paper." The AI followed `findClusters.m` but omitted the 'poor' quality label.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue by subtracting the go cue time of each trial from `trialtm`: `aligned = trialtm - goCue[trial]`. This is done per-neuron per-trial in a loop.

ii.
```python
ALIGN_EVENT = 'goCue'
align_times_all = goCue

for j in range(ntrials_total):
    trial_num = j + 1
    spk_mask = trial_arr == trial_num
    aligned = trialtm_arr[spk_mask] - align_times_all[j]
    counts, _ = np.histogram(aligned, bins=edges)
```

iii. Matches the reference's `alignSpikes.m`: `trialtm_aligned = trialtm - event_time`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is **10 ms** (DT = 1/100), producing **500 time bins** over the -2.5 to 2.5 s window. No rebinning is applied; data is binned directly at this resolution. This differs from the reference which uses 5 ms bins (1000 time bins).

ii.
```python
DT = 1.0 / 100  # 10 ms bins
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
```

iii. From CONVERSION_NOTES.md: "Bin size 10 ms (dt=1/100): As in WorkingWithDataObjs.m, gives time axis from -2.5 to 2.5 s = 500 time bins." The AI noted that different reference scripts use different bin sizes and chose 1/100 from `WorkingWithDataObjs.m` rather than 1/200 from `getDefaultParams.m`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not derived from any raw data variable. It is a synthetic time axis constructed from the bin edges, representing the center of each time bin relative to the go cue.

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2

input_trials.append(time_axis.reshape(1, -1).astype(np.float32))
```

iii. The time axis is defined by the binning parameters.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is computed as the centers of the time bins: `edges[:-1] + DT/2`, ranging from -2.495 to 2.495 s in 10 ms steps. It is the same for every trial and session.

ii.
```python
time_axis = edges[:-1] + DT / 2
input_trials.append(time_axis.reshape(1, -1).astype(np.float32))
```

iii. No processing beyond defining the bin centers.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis IS the neural binning grid. Spikes are binned using the same `edges` that define the time axis, so bin k in the neural data corresponds to bin k in the input.

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
# Same edges used for spike histogram:
counts, _ = np.histogram(aligned, bins=edges)
# Same time_axis used for input:
input_trials.append(time_axis.reshape(1, -1).astype(np.float32))
```

iii. Alignment is guaranteed by using the same bin grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Derived from `obj.bp.R` (right-instructed), `obj.bp.L` (left-instructed), `obj.bp.hit`, and `obj.bp.miss`. Since ignore trials are excluded, only hit and miss trials remain.

ii.
```python
R = np.array(bp['R']).flatten().astype(bool)
L = np.array(bp['L']).flatten().astype(bool)
hit = np.array(bp['hit']).flatten().astype(bool)
miss = np.array(bp['miss']).flatten().astype(bool)

lick_right = (R & hit) | (L & miss)
lick_direction = lick_right[valid_trials].astype(np.int32)
```

iii. The AI derives lick direction from the combination of instructed side and outcome.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A hit means the animal licked the instructed port, a miss means it licked the other one. Right lick = (R & hit) | (L & miss). Left is everything else among valid trials (which are only hit or miss). This produces **2 classes** (left=0, right=1), unlike the reference which has 3 classes (left=0, right=1, no lick=2) because the reference keeps ignore trials.

ii.
```python
lick_right = (R & hit) | (L & miss)
lick_direction = lick_right[valid_trials].astype(np.int32)  # 0=left, 1=right

out[0, :] = lick_direction[t_idx]  # broadcast per-trial to time
```

iii. The AI's `output_values` lists `['left', 'right']` for lick_direction, consistent with having only 2 classes.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derived from `obj.bp.autowater`, a per-trial flag indicating water-cued (WC) context.

ii.
```python
autowater = np.array(bp['autowater']).flatten().astype(bool)
context = (~autowater[valid_trials]).astype(np.int32)  # 0=WC, 1=DR
```

iii. The AI reads autowater directly from the trial table.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct relabelling: autowater=True -> WC (0), autowater=False -> DR (1). This matches the prompt's specification of WC=0, DR=1.

ii.
```python
context = (~autowater[valid_trials]).astype(np.int32)
```

iii. Straightforward mapping from the autowater flag.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `obj.bp.hit` and `obj.bp.miss`. Since ignore trials are excluded by trial filtering, outcome is simply the hit flag.

ii.
```python
hit = np.array(bp['hit']).flatten().astype(bool)
miss = np.array(bp['miss']).flatten().astype(bool)
outcome = hit[valid_trials].astype(np.int32)  # 0=incorrect(miss), 1=correct(hit)
```

iii. Since the AI drops ignore trials, outcome is binary (incorrect/correct).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Directly mapped: hit -> correct (1), miss -> incorrect (0). Produces **2 classes**, unlike the reference which has 3 classes (incorrect=0, correct=1, ignore=2).

ii.
```python
outcome = hit[valid_trials].astype(np.int32)
out[2, :] = outcome[t_idx]
```

iii. The AI's `output_values` lists `['incorrect', 'correct']` for outcome.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Derived from `obj.traj[1]` (bottom camera only) DLC tracking data. The feature `top_tongue` is used. Frame times from `traj[1].frameTimes` and the video offset from `obj.sglx` are used for temporal alignment.

ii.
```python
traj_bottom = obj['traj'][1]  # bottom cam
# Find tongue feature index (top_tongue)
tongue_idx = None
for i, name in enumerate(feat_names):
    if name == 'top_tongue':
        tongue_idx = i
        break
```

iii. The AI uses only the bottom camera view for the tongue, while the reference uses both side ('tongue') and bottom ('top_tongue') camera views.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial: (1) Extract x, y coordinates from `ts` array for the tongue feature. (2) Compute velocity as `np.gradient(x) * VIDEO_FR` and `np.gradient(y) * VIDEO_FR`, then speed as magnitude. (3) Set speed to NaN where x or y is NaN. (4) Interpolate to the neural time axis using `interp1d`. (5) Set remaining NaN to 0 (tongue not visible = no movement).

Notable differences from reference: no likelihood filtering (just NaN check), no per-run Gaussian smoothing before differentiation, no normalization by percentile, no combining two camera views, gradient uses fixed VIDEO_FR=400 rather than actual frame time differences, NaN set to 0 instead of a "not visible" class.

ii.
```python
vx_all = np.gradient(x) * VIDEO_FR
vy_all = np.gradient(y) * VIDEO_FR
speed = np.sqrt(vx_all**2 + vy_all**2)
speed[~valid] = np.nan

f_interp = interp1d(old_time, speed, kind='linear', bounds_error=False, fill_value=np.nan)
tongue_vel[:, trix] = f_interp(time_axis)

tongue_vel = np.nan_to_num(tongue_vel, nan=0.0)
```

iii. From CONVERSION_NOTES.md: "Tongue NaN values NOT filled (per paper methods), velocity computed only where visible." However, the code does set NaN to 0 at the end (line 522).

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The per-session 50th percentile of all non-NaN values is computed, and values >= threshold get class 1 ("high"), values < threshold get class 0 ("low"). Produces **2 classes** (low=0, high=1), unlike the reference which has 3 classes (below threshold=0, above threshold=1, not visible=2). Since NaN is set to 0 before discretization, untracked bins get folded into "low" rather than having a separate class.

ii.
```python
def discretize_per_session(data_2d, percentile=50):
    all_vals = data_2d[~np.isnan(data_2d)]
    threshold = np.percentile(all_vals, percentile)
    if threshold == 0:
        threshold = np.finfo(np.float32).eps
    result = (data_2d >= threshold).astype(np.int32)
    result[np.isnan(data_2d)] = 0
    return result
```

iii. The instructions say "discretized into two bins with per-session threshold", so 2 classes is a valid interpretation.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by subtracting the video offset and the go cue time, then the speed trace is interpolated (`interp1d`, linear) onto the neural time axis. The video offset is computed from `sglx.bitcode.bitstart / sglx.fs` vs `bp.ev.bitStart` using scipy `mode`.

ii.
```python
def compute_video_offset(obj):
    bitStart = scipy_stats.mode(np.array(obj['bp']['ev']['bitStart']).flatten(), keepdims=False).mode
    bc_bitstart = np.array(obj['sglx']['bitcode']['bitstart']).flatten()
    bc_mode = scipy_stats.mode(bc_bitstart, keepdims=False).mode
    fs = float(obj['sglx']['fs'])
    vidshift = bc_mode / fs - bitStart
    return vidshift

ft = np.array(traj_bottom['frameTimes'][trix]).flatten()
old_time = ft - vidshift - align_times[trix]
f_interp = interp1d(old_time, speed, kind='linear', bounds_error=False, fill_value=np.nan)
tongue_vel[:, trix] = f_interp(time_axis)
```

iii. The video offset calculation matches `findVideoOffset.m`. The AI uses interpolation to the neural time axis rather than bin averaging as the reference does.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Derived from `obj.traj[1]` (bottom camera) DLC tracking data. The AI uses **all features containing 'paw'** in the name, which includes both `top_paw` and `bottom_paw`. The reference uses only `top_paw`.

ii.
```python
traj_bottom = obj['traj'][1]  # bottom cam
paw_indices = []
for i, name in enumerate(feat_names):
    if 'paw' in name.lower():
        paw_indices.append(i)
```

iii. The AI uses both paw features from the bottom camera, while the reference selects only the reliably tracked one (`top_paw`).

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each trial and each paw feature: (1) Extract x, y from `ts`. (2) Fill NaN with nearest neighbor. (3) Compute velocity as `np.gradient * VIDEO_FR`, then speed as magnitude. (4) Average speeds across paw features. (5) Interpolate to neural time axis. (6) Fill remaining NaN with nearest neighbor.

Differences from reference: uses both paws averaged (not just top_paw), fills NaN with nearest rather than having "not visible" class, no likelihood filtering, no per-run Gaussian smoothing, gradient uses fixed VIDEO_FR.

ii.
```python
for pidx in paw_indices:
    x = ts[:, 0, pidx].copy()
    y = ts[:, 1, pidx].copy()
    for arr in [x, y]:
        nans = np.isnan(arr)
        if nans.any() and not nans.all():
            arr[nans] = arr[valid[nearest]]  # nearest fill
    vx = np.gradient(x) * VIDEO_FR
    vy = np.gradient(y) * VIDEO_FR
    speeds.append(np.sqrt(vx**2 + vy**2))
avg_speed = np.mean(speeds, axis=0)
```

iii. The AI chose to average both paw features rather than selecting one.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue velocity: per-session 50th percentile threshold, producing 2 classes (low=0, high=1). No "not visible" class since NaN was filled with nearest neighbor.

ii.
```python
paw_vel_disc = discretize_per_session(paw_vel_valid)
```

iii. Same discretization function as other movement outputs.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue velocity: frame times corrected by video offset and go cue, then interpolated to the neural time axis.

ii.
```python
ft = np.array(traj_bottom['frameTimes'][trix]).flatten()
old_time = ft - vidshift - align_times[trix]
f_interp = interp1d(old_time, avg_speed, kind='linear', bounds_error=False, fill_value=np.nan)
paw_vel[:, trix] = f_interp(time_axis)
```

iii. Uses the bottom camera's frame times for alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Loaded from the separate `motionEnergy_<anm>_<date>.mat` file. Handles three formats: structured array with `.data` and `.moveThresh`, nested struct with `.data.data`, and direct cell array.

ii.
```python
me_file = scipy.io.loadmat(me_path)
me_var = me_file['me']
if me_var.dtype.names:
    me_struct = me_var[0, 0]
    me_data = me_struct['data']
    if me_data.dtype.names and 'data' in me_data.dtype.names:
        inner = me_data[0, 0]
        me_raw = inner['data']
    else:
        me_raw = me_data
elif me_var.dtype == object:
    me_raw = me_var
```

iii. Handles the same three file layouts as the reference.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The motion energy trace is interpolated to the neural time axis using `interp1d` (linear), then NaN values are filled with nearest neighbor (`fillmissing(,'nearest')` equivalent). This differs from the reference which uses bin averaging.

ii.
```python
f_interp = interp1d(old_time, me_trial, kind='linear', bounds_error=False, fill_value=np.nan)
me_aligned[:, trix] = f_interp(time_axis)

# Fill NaN with nearest
for trix in range(ntrials):
    col = me_aligned[:, trix]
    nans = np.isnan(col)
    if nans.any() and not nans.all():
        nearest = np.searchsorted(valid_idx, nan_idx).clip(0, len(valid_idx)-1)
        col[nans] = col[valid_idx[nearest]]
```

iii. The AI references `loadMotionEnergy.m` which uses `interp1` and `fillmissing(,'nearest')`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same as tongue and paw velocity: per-session 50th percentile threshold, producing 2 classes (low=0, high=1).

ii.
```python
me_disc = discretize_per_session(me_valid)
```

iii. Follows the instructions' specification of 50th percentile threshold.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Frame times from the side camera (`traj[0].frameTimes`) are corrected by the video offset and go cue time, then the motion energy trace is interpolated to the neural time axis.

ii.
```python
traj_view0 = obj['traj'][0]  # side cam
ft = np.array(traj_view0['frameTimes'][trix]).flatten()
old_time = ft - vidshift - align_times[trix]
f_interp = interp1d(old_time, me_trial, kind='linear', bounds_error=False, fill_value=np.nan)
me_aligned[:, trix] = f_interp(time_axis)
```

iii. Correctly uses side camera frame times for motion energy alignment.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several approaches: (1) For tongue velocity, NaN values (where tongue is not visible) are set to 0, which then gets classified as "low" during discretization. (2) For paw velocity, NaN in x/y coordinates is filled with nearest neighbor before computing velocity. (3) For motion energy, NaN after interpolation is filled with nearest neighbor. (4) Frame time fallback: if frame times can't be loaded, a fallback generates them at 400 Hz. (5) Sessions with fewer than 2 valid trials or fewer than 10 neurons are skipped entirely. (6) Trials past the recording end are NOT filtered (unlike reference), resulting in all-zero neural data warnings for 30 trials in sessions 36 and 43.

ii.
```python
# Tongue: NaN -> 0
tongue_vel = np.nan_to_num(tongue_vel, nan=0.0)

# Paw: nearest fill
for arr in [x, y]:
    nans = np.isnan(arr)
    if nans.any() and not nans.all():
        arr[nans] = arr[valid[nearest]]

# Frame time fallback
except:
    nframes = len(me_trial)
    ft = np.arange(1, nframes + 1) / VIDEO_FR
    old_time = ft - 0.5 - align_times[trix]
```

iii. From CONVERSION_NOTES.md: "Session 36 (JEB24_2023-10-23, 17 neurons): 19 trials with all-zero neural data at end of session... These are likely recording artifacts; trials retained for completeness."

## 11-a. What are the most time-consuming steps of the code?

i. File loading and spike binning are the most time-consuming steps. The full conversion takes about 300 seconds. Spike binning involves a triple-nested loop (probes x neurons x trials), which is the main computational bottleneck.

ii.
```python
for probe_idx, valid_clu in all_cluster_indices:
    for i, clu_idx in enumerate(valid_clu):
        for j in range(ntrials_total):
            aligned = trialtm_arr[spk_mask] - align_times_all[j]
            counts, _ = np.histogram(aligned, bins=edges)
            trialdat[:, neuron_offset + i, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)
```

iii. From CONVERSION_NOTES.md Step 7: the AI tracked per-session timing and total estimated time.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop is the main candidate. It iterates per-neuron per-trial, calling `np.histogram` and `causal_gaussian_smooth` for each. The reference uses `np.histogram2d` to bin all trials at once per cluster. The tongue/paw velocity loops (per-trial) and motion energy alignment (per-trial) also iterate individually.

ii.
```python
# Current: triple loop
for probe_idx, valid_clu in all_cluster_indices:
    for i, clu_idx in enumerate(valid_clu):
        for j in range(ntrials_total):
            counts, _ = np.histogram(aligned, bins=edges)

# Reference uses histogram2d:
counts, _, _ = np.histogram2d(spike_trial, spike_time, bins=[trial_edges, BIN_EDGES])
```

iii. The spike binning loop calls `np.histogram` and the smoothing function for every neuron-trial pair, which is significantly less efficient than the reference's vectorized approach.

## 11-c. What processing does the code repeat multiple times?

i. The smoothing function `causal_gaussian_smooth` is called once per neuron per trial (total_neurons x ntrials_total times per session), and it rebuilds the Gaussian kernel each time. The video offset is computed once per session, which is efficient. The per-trial loops for tongue/paw/motion energy each independently compute frame times.

ii.
```python
# Kernel rebuilt each call:
def causal_gaussian_smooth(x, N, bctype='reflect'):
    kern = np.array(scipy_windows.gaussian(N, std=N/6.0))
    kern[:N//2] = 0
    kern = kern / kern.sum()
```

iii. The Gaussian kernel could be precomputed once and reused across all calls.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI bins spikes for ALL trials (including invalid ones) into `trialdat` of shape `(n_time, total_neurons, ntrials_total)`, then selects only valid trials afterwards with `trialdat_valid = trialdat[:, :, valid_trials]`. This means spike binning and smoothing are performed for early-lick, stim, and ignore trials that are subsequently discarded. Similarly, tongue and paw velocities are computed for all trials before selecting valid ones.

ii.
```python
# Bin ALL trials:
trialdat = np.zeros((n_time, total_neurons, ntrials_total), dtype=np.float32)
for j in range(ntrials_total):
    ...
# Then select valid:
trialdat_valid = trialdat[:, :, valid_trials]

# Same for behavioral:
tongue_vel = compute_tongue_velocity(obj, align_times_all, time_axis, vidshift, ntrials_total)
tongue_vel_valid = tongue_vel[:, valid_trials]
```

iii. Processing all trials first and filtering later is simpler but wastes computation on trials that are discarded. With ~25-40% of trials being filtered out, this represents significant unnecessary work.
