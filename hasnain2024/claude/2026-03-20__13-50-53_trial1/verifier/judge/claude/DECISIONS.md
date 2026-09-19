# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from 44 sessions across two folders (`Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior`). Sessions are hard-coded in two lists (`EPHYS_SESSIONS` and `RANDOMIZED_DELAY_SESSIONS`) as tuples of `(animal, date, probes, data_dir)`, transcribed from the authors' loading scripts. Each session's `.mat` file is loaded with `mat73.loadmat()` for v7.3 files, falling back to a custom `_load_v5_session()` using `scipy.io.loadmat` for v5 files. Motion energy is loaded separately from `motionEnergy_*.mat` files using `scipy.io.loadmat`.

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

iii. The AI documented in CONVERSION_NOTES.md that sessions were identified from the authors' `load<ANM>_ALMVideo.m` scripts, excluding commented-out sessions and those without data files. The AI correctly identified 25 fixed-delay and 19 randomized-delay sessions totaling 44.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the animal name (`anm`) from the session tuple. The set of unique animal names is sorted to form the `subjects` list, and `subject_idx` maps each session to its subject index.

ii.
```python
subjects_set = set()
for i, (anm, date, probes, data_dir) in enumerate(sessions):
    ...
    subjects_set.add(anm)
subjects = sorted(subjects_set)
subject_idx = np.array([subjects.index(anm) for anm in all_animals], dtype=np.int64)
```

iii. The animal ID is taken directly from the session registry tuple, consistent with the filename convention `data_structure_<anm>_<date>.mat`.

## 1-c. How are the data split into sessions?

i. Each tuple in `ALL_SESSIONS` is one session. Each session corresponds to one `.mat` file on disk. Fixed-delay and randomized-delay sessions are treated uniformly from the combined `ALL_SESSIONS` list. The result is 44 sessions: 25 fixed-delay and 19 randomized-delay.

ii.
```python
ALL_SESSIONS = EPHYS_SESSIONS + RANDOMIZED_DELAY_SESSIONS

for i, (anm, date, probes, data_dir) in enumerate(sessions):
    result = process_session(anm, date, probes, data_dir, ...)
```

iii. Sessions are defined by the loading scripts, same as the reference.

## 1-d. How are the data split into trials?

i. The AI reads `bp['Ntrials']` to get the total trial count, then uses arrays from `bp` (hit, miss, no, early, stim, etc.) indexed by trial number. Trial boundaries are implicit in the Bpod data structure where each per-trial field has `Ntrials` entries.

ii.
```python
bp = obj['bp']
ntrials_total = int(bp['Ntrials'])
R = np.array(bp['R']).flatten().astype(bool)
L = np.array(bp['L']).flatten().astype(bool)
hit = np.array(bp['hit']).flatten().astype(bool)
...
```

iii. The AI reads the trial count and per-trial arrays directly from the Bpod table. However, unlike the reference, the AI does not use a `trial_column` helper that truncates to `Ntrials` (some fields may be stored longer). The AI simply takes `flatten()` which may not handle this edge case.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters out early-lick trials (`early`), photostimulation trials (`stim.enable`), AND ignore/no-response trials (`no`). It further requires that each trial is a hit or miss (`hit | miss`). This is more aggressive filtering than the reference, which keeps ignore trials.

ii.
```python
valid_mask = ~early & ~stim_enable & ~no
valid_mask = valid_mask & (hit | miss)
```

iii. The AI's CONVERSION_NOTES.md states: "Trial exclusion: Exclude early, stim.enable, and no-response trials. Keep hit and miss." The paper excludes early lick trials from analyses, but ignore trials are typically kept (with their own outcome category). The AI also does NOT filter trials that extend past the end of the recording (the reference does), resulting in all-zero neural data warnings for sessions 36 and 43.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses `obj.clu` (spike-sorted clusters) with fields `trial` (trial number, 1-based), `trialtm` (spike time relative to trial start), and `quality` (curation label). Go cue times from `bp.ev.goCue` provide alignment.

ii.
```python
trial_arr = np.array(clu_probe['trial'][clu_idx]).flatten()
trialtm_arr = np.array(clu_probe['trialtm'][clu_idx]).flatten()
aligned = trialtm_arr[spk_mask] - align_times_all[j]
```

iii. Same raw variables as the reference.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to the go cue, then binned into 10 ms bins (DT=1/100) spanning -2.5 to 2.5 s (500 bins). Bin counts are converted to firing rates by dividing by DT. The rates are then smoothed with a **causal** Gaussian kernel of window size 15 bins, implemented as a half-Gaussian convolution with boundary condition 'reflect'.

ii.
```python
DT = 1.0 / 100  # 10 ms bins
SMOOTH_WINDOW = 15  # bins, causal Gaussian

edges = np.arange(TMIN, TMAX + DT, DT)
counts, _ = np.histogram(aligned, bins=edges)
fr = counts.astype(np.float32) / DT
fr_smooth = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)

def causal_gaussian_smooth(x, N, bctype='reflect'):
    kern = np.array(scipy_windows.gaussian(N, std=N/6.0))
    kern[:N//2] = 0  # causal: zero out first half
    kern = kern / kern.sum()
```

iii. The AI chose 10 ms bins based on `WorkingWithDataObjs.m` which uses `dt = 1/100`, and the causal Gaussian smoothing based on `mySmooth.m`. The reference uses 5 ms bins (`params.dt = 1/200`) and a symmetric Gaussian with sigma=14 ms (matching `gausswin(15)` at 5 ms bins). The AI's smoothing is causal (half the kernel zeroed), while the reference's is symmetric.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. First, clusters with quality labels in `{'garbage', 'gabrga', 'noisy', 'real?'}` are excluded. Then units with mean firing rate <= 1 Hz are removed. The AI does NOT include 'poor' in the exclusion set (the reference does).

ii.
```python
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}

def get_valid_cluster_indices(clu_probe, excluded_qualities=EXCLUDED_QUALITIES):
    for i, q in enumerate(qualities):
        if q_stripped.lower() not in {e.lower() for e in excluded_qualities}:
            valid.append(i)

mean_fr = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
keep = mean_fr > low_fr
```

iii. The exclusion set matches `findClusters.m` exactly (garbage, gabrga, noisy, real?). The reference additionally drops 'poor'. The 1 Hz threshold matches the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to the go cue is done by subtracting the go cue time from each spike's trial time: `aligned = trialtm - goCue[trial]`. This is the same approach as the reference.

ii.
```python
goCue = np.array(ev['goCue']).flatten()
aligned = trialtm_arr[spk_mask] - align_times_all[j]
```

iii. Standard approach matching `alignSpikes.m`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 10 ms bins (DT = 1/100), producing 500 time bins spanning -2.5 to 2.5 s. No rebinning is applied. The reference uses 5 ms bins producing 1000 time bins.

ii.
```python
DT = 1.0 / 100  # 10 ms bins
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
```

iii. The AI cited `WorkingWithDataObjs.m` which uses `dt = 1/100`. However, the reference code and other analysis scripts use `dt = 1/200` (5 ms), and the instructions say to match the reference processing.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the time axis itself, defined by the bin edges centered at each bin's midpoint, spanning -2.5 to 2.5 s around the go cue.

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
input_trials.append(time_axis.reshape(1, -1).astype(np.float32))
```

iii. Same conceptual approach as the reference.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is computed from the bin edges. No further processing. It is the same for every trial.

ii. N/A

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time axis IS the neural binning grid itself. Bin centers are used as the input values, so they align by construction.

ii.
```python
time_axis = edges[:-1] + DT / 2  # used for both input and neural binning
```

iii. Same approach as the reference.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Derived from `bp.R`, `bp.L`, `bp.hit`, and `bp.miss`. The combination of instructed side and outcome determines lick direction.

ii.
```python
R = np.array(bp['R']).flatten().astype(bool)
L = np.array(bp['L']).flatten().astype(bool)
hit = np.array(bp['hit']).flatten().astype(bool)
miss = np.array(bp['miss']).flatten().astype(bool)
```

iii. Same variables as the reference.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A hit on the instructed side means the animal licked that side; a miss means it licked the opposite side. The AI computes: `lick_right = (R & hit) | (L & miss)`, then converts to int (0=left, 1=right). Since ignore trials are excluded, there are only 2 classes. The reference keeps 3 classes (left=0, right=1, no lick=2).

ii.
```python
lick_right = (R & hit) | (L & miss)
lick_direction = lick_right[valid_trials].astype(np.int32)
```

iii. The AI's `output_values` for lick_direction is `['left', 'right']` (2 classes), while the reference has `['left', 'right', 'no lick']` (3 classes). The instructions specify "left, right, none" as the output values.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derived from `bp.autowater`. When autowater is True, the context is WC (water-cued); otherwise DR (delayed-response).

ii.
```python
autowater = np.array(bp['autowater']).flatten().astype(bool)
context = (~autowater[valid_trials]).astype(np.int32)
```

iii. Same as reference.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct mapping: autowater=True -> WC (0), autowater=False -> DR (1). This matches the reference.

ii.
```python
context = (~autowater[valid_trials]).astype(np.int32)  # 0=WC, 1=DR
```

iii. Consistent with the reference.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `bp.hit` and `bp.miss`. Since ignore trials are excluded from the dataset, only hit and miss remain.

ii.
```python
hit = np.array(bp['hit']).flatten().astype(bool)
miss = np.array(bp['miss']).flatten().astype(bool)
outcome = hit[valid_trials].astype(np.int32)
```

iii. Same raw variables as the reference, but the reference keeps ignore trials as a third class.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hit trials are coded as correct (1), miss trials as incorrect (0). The AI has only 2 outcome classes (`['incorrect', 'correct']`). The reference has 3 classes (`['incorrect', 'correct', 'ignore']`) since it keeps ignore trials.

ii.
```python
outcome = hit[valid_trials].astype(np.int32)  # 0=incorrect (miss), 1=correct (hit)
```

iii. The instructions specify "incorrect, correct, ignore" as the output values. The AI's 2-class approach omits the ignore class.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Derived from `obj.traj[1]` (bottom camera only), using the `top_tongue` feature. The AI uses only one camera view, whereas the reference combines both the side camera's `tongue` and the bottom camera's `top_tongue`.

ii.
```python
traj_bottom = obj['traj'][1]  # bottom cam
tongue_idx = None
for i, name in enumerate(feat_names):
    if name == 'top_tongue':
        tongue_idx = i
        break
```

iii. The AI's CONVERSION_NOTES says "Tongue velocity: Use tip-of-tongue displacement from bottom cam view." The reference uses both cameras to recover more tongue visibility.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI computes x/y gradients multiplied by the video frame rate (400 Hz) to get velocity in pixels/second. No Gaussian smoothing of positions is applied before differentiation. No likelihood-based filtering is applied (it relies on NaN values already present in positions). The velocity is computed over ALL frames (including NaN transitions), then NaN bins are set to 0. No normalization is applied.

ii.
```python
vx_all = np.gradient(x) * VIDEO_FR
vy_all = np.gradient(y) * VIDEO_FR
speed = np.sqrt(vx_all**2 + vy_all**2)
speed[~valid] = np.nan
...
tongue_vel = np.nan_to_num(tongue_vel, nan=0.0)
```

iii. The reference applies a 5 ms Gaussian smooth to x,y positions, computes velocity only within contiguous runs of tracked frames (likelihood > 0.9), normalizes each camera view by its 90th percentile, averages the two views, and uses a "not visible" class for untracked bins. The AI's approach differs significantly: no smoothing, no run-based computation, no normalization, single camera, and NaN mapped to 0 instead of a separate class.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI discretizes at the 50th percentile per session into 2 classes (0=low, 1=high). NaN values (tongue not visible) are set to 0 (low class) before discretization. The reference uses 3 classes with a separate "not visible" class (code 2).

ii.
```python
tongue_vel = np.nan_to_num(tongue_vel, nan=0.0)
tongue_vel_disc = discretize_per_session(tongue_vel_valid)

def discretize_per_session(data_2d, percentile=50):
    threshold = np.percentile(all_vals, percentile)
    result = (data_2d >= threshold).astype(np.int32)
    result[np.isnan(data_2d)] = 0
    return result
```

iii. The instructions specify: "0: < 50th percentile, 1: >= 50th percentile, 2: not visible". The AI maps untracked bins to 0 (low) instead of a separate class 2, and `output_values` lists only `['low', 'high']`. This means ~88% of tongue bins that should be "not visible" are incorrectly labeled as "low".

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI uses `interp1d` (linear interpolation) to resample frame-resolution velocity to the neural time axis. The reference bins frame values into the 5 ms bins by averaging. Both use the same video offset correction.

ii.
```python
ft = np.array(traj_bottom['frameTimes'][trix]).flatten()
old_time = ft - vidshift - align_times[trix]
f_interp = interp1d(old_time, speed, kind='linear', bounds_error=False, fill_value=np.nan)
tongue_vel[:, trix] = f_interp(time_axis)
```

iii. The AI uses interpolation while the reference uses binned averaging. The video offset computation follows the same logic as the reference's `findVideoOffset.m`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The AI uses `obj.traj[1]` (bottom camera) with ALL paw features (both `top_paw` and `bottom_paw`). The reference uses only `top_paw`.

ii.
```python
paw_indices = []
for i, name in enumerate(feat_names):
    if 'paw' in name.lower():
        paw_indices.append(i)
```

iii. The AI averages velocities from all paw features found. The reference deliberately uses only `top_paw` because `bottom_paw` drops out during the delay epoch.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The AI fills NaN positions with nearest-neighbor interpolation, then computes x/y gradients multiplied by VIDEO_FR, and averages speeds across paw features. No Gaussian smoothing of positions. After interpolation to the neural time axis, remaining NaN values are filled with nearest neighbor.

ii.
```python
for arr in [x, y]:
    nans = np.isnan(arr)
    if nans.any() and not nans.all():
        nearest = np.searchsorted(valid, nan_pos).clip(0, len(valid)-1)
        arr[nans] = arr[valid[nearest]]
vx = np.gradient(x) * VIDEO_FR
vy = np.gradient(y) * VIDEO_FR
speeds.append(np.sqrt(vx**2 + vy**2))
avg_speed = np.mean(speeds, axis=0)
```

iii. The reference computes velocity within contiguous runs of tracked frames using a 5 ms Gaussian smooth, and uses a "not visible" class for untracked bins. The AI fills in missing data instead, which fabricates velocity values.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: 50th percentile per session, 2 classes (low/high). NaN set to 0. The reference uses 3 classes with "not visible" for untracked bins.

ii.
```python
paw_vel_disc = discretize_per_session(paw_vel_valid)
```

iii. The instructions specify 3 classes including "not visible" (code 2). The AI only uses 2.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same approach as tongue: linear interpolation from frame times to neural time axis, with video offset correction.

ii.
```python
f_interp = interp1d(old_time, avg_speed, kind='linear', bounds_error=False, fill_value=np.nan)
paw_vel[:, trix] = f_interp(time_axis)
```

iii. Uses interpolation rather than binned averaging.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Loaded from separate `motionEnergy_<anm>_<date>.mat` files. The AI handles multiple file formats (struct with data/moveThresh, nested struct, or direct cell array).

ii.
```python
me_path = os.path.join(DATA_ROOT, data_dir, f'motionEnergy_{anm}_{date}.mat')
me_file = scipy.io.loadmat(me_path)
me_var = me_file['me']
if me_var.dtype.names:
    me_struct = me_var[0, 0]
    me_data = me_struct['data']
    if me_data.dtype.names and 'data' in me_data.dtype.names:
        me_raw = me_data[0, 0]['data']  # nested struct
```

iii. Same source as the reference, with similar handling of the multiple file formats.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy is aligned to the neural time axis using linear interpolation, then NaN values are filled with nearest neighbor. It is then discretized at the 50th percentile into 2 classes. The reference bins frame values by averaging and uses 3 classes including "no video".

ii.
```python
f_interp = interp1d(old_time, me_trial, kind='linear', bounds_error=False, fill_value=np.nan)
me_aligned[:, trix] = f_interp(time_axis)
# Fill NaN with nearest
for trix in range(ntrials):
    col[nans] = col[valid_idx[nearest]]
```

iii. The reference does no interpolation or NaN-filling -- it bins the raw frame values into 5 ms bins and uses a "no video" class for empty bins.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same as tongue/paw: 50th percentile per session, 2 classes (low/high). The reference uses 3 classes with "no video" for bins without camera data.

ii.
```python
me_disc = discretize_per_session(me_valid)
```

iii. The instructions specify 3 classes including "no video" (code 2). The AI only uses 2.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Linear interpolation from frame times to neural time axis, with video offset correction and nearest-neighbor filling of NaN values.

ii.
```python
def get_motion_energy_aligned(me_raw, obj, align_times, time_axis, vidshift, ntrials):
    old_time = ft - vidshift - align_times[trix]
    f_interp = interp1d(old_time, me_trial, kind='linear', bounds_error=False, fill_value=np.nan)
    me_aligned[:, trix] = f_interp(time_axis)
```

iii. The video offset computation follows the reference. Uses interpolation rather than binned averaging.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several cases: (1) Missing frame times are handled by a try/except fallback that generates frame times at 400 Hz. (2) NaN positions for the tongue are left as NaN until discretization, where they become 0 (low). (3) NaN positions for the paw are filled with nearest-neighbor interpolation before velocity computation. (4) NaN motion energy values are filled with nearest-neighbor. (5) Trials past the end of recording are NOT detected and result in all-zero neural data (warnings for sessions 36 and 43).

ii.
```python
# Fallback for missing frame times:
except:
    nframes = len(me_trial)
    ft = np.arange(1, nframes + 1) / VIDEO_FR
    old_time = ft - 0.5 - align_times[trix]

# Tongue NaN -> 0:
tongue_vel = np.nan_to_num(tongue_vel, nan=0.0)

# Paw NaN fill:
nearest = np.searchsorted(valid, nan_pos).clip(0, len(valid)-1)
arr[nans] = arr[valid[nearest]]
```

iii. The reference handles missing data by keeping trials and using the "not visible" / "no video" class for untracked bins, never filling in fabricated values. The reference also detects and drops trials past the end of the recording.

## 11-a. What are the most time-consuming steps of the code?

i. Loading MATLAB files and spike binning. The AI notes loading times of 3-9 seconds per session and spike binning of 0.5-5 seconds per session. Total processing for 44 sessions was ~300 seconds.

ii.
```python
# Per-session timing output:
print(f"  Loaded in {t_load:.1f}s")
print(f"  Spike binning: {t_bin:.1f}s for {total_neurons} neurons")
```

iii. Loading is the dominant cost, similar to the reference. The spike binning is slower than the reference because the AI uses per-trial, per-neuron loops instead of vectorized histogram2d.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning uses a triple-nested loop (probes -> clusters -> trials) with per-trial histogramming. The reference uses a single `np.histogram2d` call per cluster to bin all trials at once. The tongue/paw velocity computation also uses per-trial loops. The smoothing is done per-trial per-neuron with a Python loop and `np.convolve`.

ii.
```python
for i, clu_idx in enumerate(valid_clu):
    for j in range(ntrials_total):
        spk_mask = trial_arr == trial_num
        counts, _ = np.histogram(aligned, bins=edges)
        fr_smooth = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)
```

iii. The per-trial spike loop could be replaced with `np.histogram2d` as the reference does. The per-column smoothing loop inside `causal_gaussian_smooth` could use `scipy.ndimage.convolve1d`.

## 11-c. What processing does the code repeat multiple times?

i. The AI computes firing rates for ALL `ntrials_total` trials (including filtered ones), then selects valid trials afterward. This means spike binning and smoothing are done for early-lick, stim, and ignore trials that are later discarded. Motion energy alignment and video processing are also done for all trials.

ii.
```python
trialdat = np.zeros((n_time, total_neurons, ntrials_total), dtype=np.float32)
for j in range(ntrials_total):  # processes ALL trials
    ...
trialdat_valid = trialdat[:, :, valid_trials]  # then selects valid ones
```

iii. Processing all trials first and filtering later is wasteful but simpler. The reference filters trials before neural processing.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) Processing spikes for all trials including those that will be filtered out. (2) The `_load_v5_session` function reads ALL fields from the MATLAB file including unused ones like `ex`, `spkWavs`. (3) Paw NaN-filling computation is done even though the filled values may not be needed. (4) Motion energy `moveThresh` is loaded but not used (50th percentile is used instead).

ii.
```python
# Loading unused fields:
ex_raw = o['ex'][0, 0]
ex = {}
for field in ex_raw.dtype.names:
    ...
obj['ex'] = ex
```

iii. The reference also materializes unused fields during loading but avoids processing filtered-out trials.
