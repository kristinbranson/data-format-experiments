# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-coded the 44 included sessions into two Python lists, `EPHYS_SESSIONS` and `RD_SESSIONS`, then iterated through them session-by-session. For each session it opened `data_structure_<anm>_<date>.mat` from either `/app/data/Ephys_Behavior` or `/app/data/RandomizedDelay_Ephys_Behavior`, using `h5py` for v7.3 files and `scipy.io.loadmat` otherwise. Motion energy was loaded separately from `motionEnergy_<anm>_<date>.mat`.

ii.
```python
EPHYS_SESSIONS = [
    ('EKH1', '2021-08-07', [2]),
    ...
]

RD_SESSIONS = [
    ('JEB11', '2022-05-10', [1]),
    ...
]
```

```python
def load_mat_file(fpath):
    try:
        f = h5py.File(fpath, 'r')
        return f, 'h5py'
    except:
        data = sio.loadmat(fpath, squeeze_me=False)
        return data, 'scipy'
```

```python
all_sessions = [(a, d, p, EPHYS_DATA_DIR) for a, d, p in EPHYS_SESSIONS] + \
               [(a, d, p, RD_DATA_DIR) for a, d, p in RD_SESSIONS]
```

iii. In `CONVERSION_NOTES.md`, the AI justified this by saying the dataset mixes HDF5 and older MATLAB formats, combines both ephys datasets into one output, and excludes extra files that were not in the authors' loading scripts.

## 1-b. How are the data split into subjects?

i. The AI treated the animal id (`anm`) from each hard-coded session tuple as the subject id. It built `subjects` incrementally in first-seen order and stored `subject_idx` as the index of each session's animal in that list.

ii.
```python
subjects, subject_idx, session_info = [], [], []
...
if anm not in subjects:
    subjects.append(anm)
subject_idx.append(subjects.index(anm))
```

iii. The notes justify this by counting unique animals separately in the two session groups and stating that the combined dataset has 14 distinct animals.

## 1-c. How are the data split into sessions?

i. Each tuple in `EPHYS_SESSIONS` or `RD_SESSIONS` is treated as one recording session. `process_session` converts one session at a time, and each returned session contributes one element to `neural`, `input`, and `output`.

ii.
```python
for si, (anm, date, probes, ddir) in enumerate(all_sessions):
    result = process_session(anm, date, probes, ddir, args.show_processing)
    if result is None:
        continue
    neural_all.append(result['neural'])
    input_all.append(result['input'])
    output_all.append(result['output'])
```

iii. The AI's notes say it combined 25 fixed-delay sessions and 19 randomized-delay sessions because both are relevant ALM electrophysiology datasets.

## 1-d. How are the data split into trials?

i. The AI used `bp['Ntrials']` as the trial count, read trial-level behavioral arrays by flattening them, and looped over trial numbers `1..n_trials` when assigning spikes to trials. Video and motion energy were also processed one trial index at a time.

ii.
```python
info = {
    'n_trials': int(bp['Ntrials'][()].flatten()[0]),
    'hit': bp['hit'][()].flatten().astype(bool),
    'miss': bp['miss'][()].flatten().astype(bool),
    ...
}
```

```python
for ci, clu in enumerate(clusters):
    for trial_num in range(1, n_trials + 1):
        spk_mask = clu['trial'] == trial_num
        ...
```

```python
for t in range(n_trials):
    ...
```

iii. There is no explicit justification beyond the notes' description of `obj.bp` as a per-trial structure containing `Ntrials`, hit/miss labels, and events.

## 1-e. How are trials filtered based on quality controls?

i. The AI mostly did not filter trials. It loaded `early` and `stim_enable`, but it never dropped early-lick trials, photostimulation trials, or late trials after the recording stopped. Instead it kept early trials and only relabeled their `lick_direction` and `outcome`. The only session-level exclusion was dropping entire sessions with fewer than 10 retained neurons.

ii.
```python
if 'stim' in bp:
    stim = bp['stim']
    info['stim_enable'] = stim['enable'][()].flatten().astype(bool) if 'enable' in stim else np.zeros(info['n_trials'], dtype=bool)
else:
    info['stim_enable'] = np.zeros(info['n_trials'], dtype=bool)
```

```python
early = trial_info['early']
...
lick_direction[early] = 2
...
outcome[early] = 2
```

```python
if n_kept < 10:
    print(f"    WARNING: Too few neurons ({n_kept}), skipping")
    ...
    return None
```

iii. The notes explicitly say "All trials included" and say conditions were kept as output labels instead of dropping trials. Later notes acknowledge all-zero neural trials caused by recording gaps but treat them as expected warnings rather than filtering them out.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derived neural data from each cluster's `trialtm` and `trial` arrays, aligned using per-trial `goCue` times from `obj.bp.ev.goCue`. It also used cluster `quality` for filtering.

ii.
```python
trialtm = f[clu['trialtm'][i, 0]][()].flatten()
trial = f[clu['trial'][i, 0]][()].flatten().astype(int)
clusters.append({'quality': q_str, 'trialtm': trialtm, 'trial': trial, 'probe': probe_num})
```

```python
goCue = trial_info['goCue']
...
aligned_times = clu['trialtm'][spk_mask] - goCue[trial_num - 1]
```

iii. The notes say spike alignment should match `alignSpikes.m` and quality filtering should match `findClusters.m`.

## 2-b. How is the `neural` data processed?

i. For each cluster and each trial, the AI histogrammed aligned spike times into a fixed window from `-2.5` to `2.5` s using 10 ms bins, converted counts to firing rates, and smoothed each trial's rate vector with a custom causal Gaussian filter using a 15-sample window and reflective padding. It did no additional normalization or baseline subtraction.

ii.
```python
DT = 1.0 / 100  # 10ms bins
SMOOTH_WINDOW = 15
BC_TYPE = 'reflect'
```

```python
N, _ = np.histogram(aligned_times, bins=edges)
fr = N.astype(np.float64) / DT
trialdat[:, ci, trial_num - 1] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE).astype(np.float32)
```

```python
def causal_gaussian_smooth(x, N, bctype='reflect'):
    ...
    kern[:int(N // 2)] = 0
    ...
```

iii. The notes say this was meant to match the MATLAB `getSeq.m` and `mySmooth.m` pipeline and justify the 10 ms choice by citing `WorkingWithDataObjs.m` as "more standard for decoder applications."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI first removed clusters whose quality string matched `garbage`, `noisy`, `gabrga`, or `real?`, and also discarded empty or corrupted quality strings. After binning/smoothing, it removed units with mean firing rate `<= 1 Hz` over all trials and time bins. It then dropped entire sessions with fewer than 10 kept neurons.

ii.
```python
EXCLUDED_QUALITIES = {'garbage', 'noisy', 'gabrga', 'real?'}
...
if q_str in EXCLUDED_QUALITIES or q_str == '' or '\x00' in q_str:
    continue
```

```python
mean_fr = trialdat.mean(axis=(0, 2))
keep_mask = mean_fr > LOW_FR_THRESHOLD
...
if n_kept < 10:
    ...
    return None
```

iii. The notes justify this as matching `findClusters.m` and `removeLowFRClusters.m`, and cite the paper's 1 Hz firing-rate threshold.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes were aligned to go-cue onset by subtracting the trial's `goCue` time from each spike's `trialtm` within that trial.

ii.
```python
aligned_times = clu['trialtm'][spk_mask] - goCue[trial_num - 1]
N, _ = np.histogram(aligned_times, bins=edges)
```

iii. The notes explicitly say this matches `alignSpikes.m` and the task requirement to align to `goCue`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI used a uniform 10 ms grid from `-2.5` to `2.5` s around the go cue. Neural data were binned directly on that grid; video-derived outputs and the time input were resampled onto the same grid. No secondary neural rebinning was applied.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 100  # 10ms bins
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
```

iii. The notes say the AI deliberately chose 10 ms instead of 5 ms because it found both in the reference materials and considered 10 ms more standard for decoder use.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is derived from the synthetic analysis time grid defined by `TMIN`, `TMAX`, and `DT`, with the go cue as the alignment event. The raw data contribution is the choice of `goCue` as the zero point.

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
...
time_input = time_axis.astype(np.float32).reshape(1, -1)
```

iii. The notes say the time-from-go-cue input is simply the continuous decoder input corresponding to the aligned analysis window.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI computed bin centers for the fixed analysis grid, cast them to `float32`, reshaped to `(1, n_time)`, and copied that same vector into every trial.

ii.
```python
time_axis = edges[:-1] + DT / 2
...
time_input = time_axis.astype(np.float32).reshape(1, -1)
...
input_trials.append(time_input.copy())
```

iii. The notes do not give extra justification beyond choosing the `[-2.5, 2.5]` go-cue-centered window and 10 ms bin size.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time input is exactly the same bin-center grid used for neural binning, so each input time point corresponds to the same analysis bin as the neural activity.

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
...
N, _ = np.histogram(aligned_times, bins=edges)
...
time_input = time_axis.astype(np.float32).reshape(1, -1)
```

iii. The notes describe the time variable as "time from go cue" and present it as the common decoder input axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derived lick direction from trial-level `hit`, `miss`, `R`, and `L` fields, plus `early` for an override that forces early-lick trials to the "none" class.

ii.
```python
hit = trial_info['hit']
miss = trial_info['miss']
R = trial_info['R']
L = trial_info['L']
early = trial_info['early']
```

iii. The notes map lick direction to `bp.hit`, `bp.miss`, `bp.R`, and `bp.L`, and later justify using `early` because some early-lick trials were also marked hit or miss.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI assigned code `1` to right licks and `0` to left licks based on instructed side and hit/miss status. Trials not assigned by hit or miss stayed at code `2` ("none"), and all early-lick trials were explicitly reset to `2`.

ii.
```python
lick_direction = np.full(n_trials, 2, dtype=int)  # 2=none
lick_direction[hit & R] = 1  # right
lick_direction[hit & L] = 0  # left
lick_direction[miss & R] = 0  # licked left (wrong on right trial)
lick_direction[miss & L] = 1  # licked right (wrong on left trial)
lick_direction[early] = 2
```

iii. The notes justify the early-trial override by saying early licks can co-occur with hit/miss labels and therefore should become "none" in this representation.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The AI derived context entirely from `bp.autowater`.

ii.
```python
autowater = trial_info['autowater']
```

iii. The notes say context is a direct binary mapping from `bp.autowater`.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI converted `autowater` directly to integer labels and paired those labels with `output_values = ['DR', 'WC']`. In effect, `autowater == False` becomes `0` (`DR`) and `autowater == True` becomes `1` (`WC`).

ii.
```python
context = autowater.astype(int)  # 0=DR, 1=WC
```

```python
'output_values': [
    ['left', 'right', 'none'], ['DR', 'WC'],
    ...
]
```

iii. The notes explicitly justify this as a binary context variable with `0=DR, 1=WC`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The AI derived outcome from `hit` and `miss`, with `early` used as an override to force early-lick trials to `ignore`.

ii.
```python
hit = trial_info['hit']
miss = trial_info['miss']
early = trial_info['early']
```

iii. The notes map outcome to `bp.hit`, `bp.miss`, and `bp.early`, and justify the `early` override as part of the critical review fix.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome was initialized to `2` (`ignore`), then set to `1` for hits and `0` for misses. Early-lick trials were finally forced back to `2`.

ii.
```python
outcome = np.full(n_trials, 2, dtype=int)  # 2=ignore
outcome[hit] = 1  # correct
outcome[miss] = 0  # incorrect
outcome[early] = 2
```

iii. The notes justify this by saying early lick trials can also be marked hit/miss and should be recoded to ignore in the decoder outputs.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The AI used DeepLabCut tracking from the side camera only: feature name `tongue` in `obj.traj`, plus side-camera `frameTimes`, `ts`, the session-wide `vidshift`, and per-trial `goCue`.

ii.
```python
ft_side, ts_side, feat_side = get_traj_trial_h5(file_data, obj, t, 0)
...
if 'tongue' in feat_side:
    ti = feat_side.index('tongue')
    tx, ty, tc = ts_side[ti, 0, :], ts_side[ti, 1, :], ts_side[ti, 2, :]
```

iii. The notes explicitly say "Tongue velocity from side cam" and justify visibility with a confidence threshold of 0.5.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI computed frame-to-frame speed from raw x/y positions and raw frame times, masked frames whose DLC confidence was below `0.5`, then linearly interpolated the speed trace onto the common time grid. It also interpolated the visibility mask and re-applied it after interpolation. It did not smooth position, split tracked runs, normalize across cameras, or combine both tongue views.

ii.
```python
def compute_velocity(x, y, ft):
    dt = np.diff(ft)
    dt[dt == 0] = 1e-6
    dx = np.diff(x)
    dy = np.diff(y)
    speed = np.sqrt(dx**2 + dy**2) / dt
    return np.concatenate([[speed[0]], speed])
```

```python
tongue_visible = tc >= DLC_CONFIDENCE_THRESHOLD
tspeed = compute_velocity(tx, ty, ft_side)
tspeed[~tongue_visible] = np.nan
tv_interp = np.interp(time_axis, ft_side_aligned, tspeed)
tv_vis = np.interp(time_axis, ft_side_aligned, tongue_visible.astype(float)) >= 0.5
tv_interp[~tv_vis] = np.nan
tongue_vel_all[t] = tv_interp
```

iii. The notes justify this only briefly: use the side-camera tongue feature, treat confidence `>= 0.5` as visible, and then discretize per session.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI pooled all non-NaN tongue-velocity bins within a session, computed the 50th percentile, assigned bins below that threshold to `0`, bins at or above it to `1`, and left NaN bins as `2`.

ii.
```python
tongue_valid = ~np.isnan(tongue_vel_all)
tongue_thresh = np.nanpercentile(tongue_vel_all[tongue_valid], 50) if tongue_valid.any() else 0
tongue_disc = np.full(tongue_vel_all.shape, 2, dtype=int)
tongue_disc[tongue_valid & (tongue_vel_all < tongue_thresh)] = 0
tongue_disc[tongue_valid & (tongue_vel_all >= tongue_thresh)] = 1
```

iii. The notes explicitly say the discretization is a per-session 50th-percentile split with a third class for not visible bins.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI estimated a session-wide video shift from SpikeGLX bitcodes and Bpod `bitStart`, subtracted that shift and the trial's `goCue` from side-camera frame times, then interpolated tongue speed to the common neural time grid.

ii.
```python
info['vidshift'] = sp_stats.mode(bitstart_sglx, keepdims=False).mode / fs - sp_stats.mode(bitStart_bp, keepdims=False).mode
```

```python
ft_side_aligned = ft_side - vidshift - goCue[t]
tv_interp = np.interp(time_axis, ft_side_aligned, tspeed)
```

iii. The notes justify the offset computation by citing `findVideoOffset.m` and say the outputs should share the same decoder time axis as neural data.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The AI derived paw velocity from the bottom-camera `top_paw` feature in `obj.traj`, along with its `frameTimes`, the session `vidshift`, and per-trial `goCue`.

ii.
```python
ft_bot, ts_bot, feat_bot = get_traj_trial_h5(file_data, obj, t, 1)
...
if 'top_paw' in feat_bot:
    pi = feat_bot.index('top_paw')
    px, py, pc = ts_bot[pi, 0, :], ts_bot[pi, 1, :], ts_bot[pi, 2, :]
```

iii. The notes explicitly say "Paw velocity from bottom cam: using `top_paw` feature."

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Paw velocity used the same processing as tongue velocity but on the bottom camera: raw frame-to-frame speed from x/y, DLC visibility threshold `0.5`, NaN masking, and interpolation onto the common time grid. There was no smoothing of position and no run-wise handling of gaps.

ii.
```python
paw_visible = pc >= DLC_CONFIDENCE_THRESHOLD
pspeed = compute_velocity(px, py, ft_bot)
pspeed[~paw_visible] = np.nan
pv_interp = np.interp(time_axis, ft_bot_aligned, pspeed)
pv_vis = np.interp(time_axis, ft_bot_aligned, paw_visible.astype(float)) >= 0.5
pv_interp[~pv_vis] = np.nan
paw_vel_all[t] = pv_interp
```

iii. The notes justify this only at a high level: use the bottom-camera paw feature and discretize it per session.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The AI applied a per-session median split over all non-NaN paw-velocity bins: below-threshold bins became `0`, above-or-equal bins became `1`, and NaN bins stayed `2`.

ii.
```python
paw_valid = ~np.isnan(paw_vel_all)
paw_thresh = np.nanpercentile(paw_vel_all[paw_valid], 50) if paw_valid.any() else 0
paw_disc = np.full(paw_vel_all.shape, 2, dtype=int)
paw_disc[paw_valid & (paw_vel_all < paw_thresh)] = 0
paw_disc[paw_valid & (paw_vel_all >= paw_thresh)] = 1
```

iii. The notes explicitly say velocity variables use per-session 50th-percentile discretization.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. The AI aligned bottom-camera frame times with `vidshift` and `goCue`, then interpolated paw speed onto the common neural time grid.

ii.
```python
ft_bot_aligned = ft_bot - vidshift - goCue[t]
pv_interp = np.interp(time_axis, ft_bot_aligned, pspeed)
```

iii. The notes justify this with the same video-offset logic used for the tongue and motion-energy signals.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy came from `motionEnergy_<anm>_<date>.mat` in the session directory. The AI unwrapped a few possible MATLAB layouts into a list of per-trial motion-energy vectors.

ii.
```python
def load_motion_energy(anm, date, data_dir):
    me_path = os.path.join(data_dir, f'motionEnergy_{anm}_{date}.mat')
    ...
    me_data = sio.loadmat(me_path, squeeze_me=False)
    me_raw = me_data['me']
```

```python
if me_raw.dtype.names and 'data' in me_raw.dtype.names:
    ...
elif me_raw.dtype == object:
    ...
```

iii. The notes justify this by saying motion-energy files exist in multiple struct/cell-array layouts and therefore need wrapper handling.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. For each trial, the AI interpolated the motion-energy trace onto the common time grid. If its length matched the side-camera frame times, it used those aligned frame times directly; otherwise it fabricated evenly spaced frame times between the first and last side-camera timestamps and interpolated from those. It did not smooth the trace before discretization.

ii.
```python
if me_trials is not None and t < len(me_trials) and len(me_trials[t]) > 0:
    me_trial = me_trials[t].astype(float)
    if len(me_trial) == len(ft_side):
        me_interp = np.interp(time_axis, ft_side_aligned, me_trial)
    elif len(me_trial) > 0:
        me_ft = np.linspace(ft_side[0], ft_side[-1], len(me_trial))
        me_ft_aligned = me_ft - vidshift - goCue[t]
        me_interp = np.interp(time_axis, me_ft_aligned, me_trial)
    else:
        me_interp = np.full(n_time, np.nan)
```

iii. The notes justify this only indirectly by saying motion energy was properly loaded for all sessions and should be discretized per session like the other movement outputs.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The AI median-split all non-NaN motion-energy bins within a session: values below threshold became `0`, values at or above threshold became `1`, and NaN bins remained `2`.

ii.
```python
me_valid = ~np.isnan(me_all)
me_thresh_disc = np.nanpercentile(me_all[me_valid], 50) if me_valid.any() else 0
me_disc = np.full(me_all.shape, 2, dtype=int)
me_disc[me_valid & (me_all < me_thresh_disc)] = 0
me_disc[me_valid & (me_all >= me_thresh_disc)] = 1
```

iii. The notes justify this as the same per-session 50th-percentile discretization used for movement variables.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligned motion energy using the same `vidshift` and go-cue correction as the camera features, then interpolated to the neural time grid.

ii.
```python
me_interp = np.interp(time_axis, ft_side_aligned, me_trial)
...
me_ft_aligned = me_ft - vidshift - goCue[t]
me_interp = np.interp(time_axis, me_ft_aligned, me_trial)
```

iii. The notes treat motion energy as another time-varying output that should share the common aligned decoder grid.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI used permissive fallback handling. Missing or malformed motion-energy files returned `(None, None)`. Per-trial video-processing failures were swallowed by a blanket `try/except`, leaving that trial's velocity and motion-energy arrays as NaN. Zero `dt` values in video frame times were clamped to `1e-6`. Missing bins later became category `2` during discretization. The AI did not remove trials with all-zero neural data after the recording ended.

ii.
```python
if not os.path.exists(me_path):
    return None, None
...
except Exception as e:
    print(f"    WARNING: Failed to load ME for {anm}_{date}: {e}")
    return None, None
```

```python
dt = np.diff(ft)
dt[dt == 0] = 1e-6  # avoid division by zero
```

```python
for t in range(n_trials):
    try:
        ...
    except Exception as e:
        pass  # Leave as NaN
```

iii. The notes justify some of this as edge-case handling for mixed MATLAB formats and note that zero-neural trials were left in and only reported as warnings.

## 11-a. What are the most time-consuming steps of the code?

i. The most expensive implemented steps are the nested neural loop over every cluster and every trial, and the per-trial video/motion-energy interpolation loop. These dominate runtime more than the small assembly steps.

ii.
```python
for ci, clu in enumerate(clusters):
    for trial_num in range(1, n_trials + 1):
        ...
        N, _ = np.histogram(aligned_times, bins=edges)
        ...
```

```python
for t in range(n_trials):
    try:
        ...
        tv_interp = np.interp(time_axis, ft_side_aligned, tspeed)
        ...
        pv_interp = np.interp(time_axis, ft_bot_aligned, pspeed)
        ...
        me_interp = np.interp(time_axis, ft_side_aligned, me_trial)
```

iii. The notes do not explicitly identify bottlenecks, but they report about 148 s total runtime for the full conversion, which is consistent with these nested loops being the main cost.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest candidate is the neural `cluster x trial` loop, which could have been collapsed into a single 2D counting pass per cluster or a session-wide counting strategy. The per-trial camera interpolation loops are also repeated scalar operations that were not vectorized.

ii.
```python
for ci, clu in enumerate(clusters):
    for trial_num in range(1, n_trials + 1):
        spk_mask = clu['trial'] == trial_num
        ...
```

```python
for t in range(n_trials):
    ...
```

iii. There is no explicit justification in the notes for keeping these loops; the notes only say the implementation handles both MATLAB formats and mixed trajectory layouts.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats several operations: it recomputes a histogram separately for every `(cluster, trial)` pair, separately interpolates both the continuous signal and a visibility mask for tongue and paw, and repeats static trial labels across every time bin when building the output arrays.

ii.
```python
for ci, clu in enumerate(clusters):
    for trial_num in range(1, n_trials + 1):
        ...
        N, _ = np.histogram(aligned_times, bins=edges)
```

```python
tv_interp = np.interp(time_axis, ft_side_aligned, tspeed)
tv_vis = np.interp(time_axis, ft_side_aligned, tongue_visible.astype(float)) >= 0.5
...
pv_interp = np.interp(time_axis, ft_bot_aligned, pspeed)
pv_vis = np.interp(time_axis, ft_bot_aligned, paw_visible.astype(float)) >= 0.5
```

```python
output[0, :] = lick_direction[t]
output[1, :] = context[t]
output[2, :] = outcome[t]
```

iii. No explicit justification is given for these repeated computations.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI reads and stores some fields it never uses downstream (`sample`, `delay`, `L`, `me_thresh`), computes a `show_processing` argument it never uses, and materializes per-trial constant outputs as full-length time series even though only three outputs vary within trial. It also builds visibility interpolants only to throw them away after masking the interpolated signals.

ii.
```python
info = {
    ...
    'L': bp['L'][()].flatten().astype(bool),
    ...
    'sample': ev['sample'][()].flatten(),
    'delay': ev['delay'][()].flatten(),
}
```

```python
me_trials, me_thresh = load_motion_energy(anm, date, data_dir)
```

```python
parser.add_argument('--show-processing', action='store_true')
...
result = process_session(anm, date, probes, ddir, args.show_processing)
```

```python
output[0, :] = lick_direction[t]
output[1, :] = context[t]
output[2, :] = outcome[t]
```

iii. The notes do not justify these extra computations; they mainly justify broader design choices such as bin size, inclusion of all trials, and support for multiple MATLAB layouts.
