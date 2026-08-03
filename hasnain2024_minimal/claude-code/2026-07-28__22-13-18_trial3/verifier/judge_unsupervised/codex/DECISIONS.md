# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-coded a session manifest in `EPHYS_SESSIONS`, then iterated over that list, opening each `data_structure_<animal>_<date>.mat` file from `data/<data_dir>/`. It did not use the reference `DataLoadingScripts/loadSessionData.m` pipeline at runtime; it reimplemented loading in Python.

ii. 
```python
EPHYS_SESSIONS = [
    ('EKH1', '2021-08-07', [2], 'Ephys_Behavior'),
    ...
    ('JEB23', '2023-10-20', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB24', '2023-11-03', [1], 'RandomizedDelay_Ephys_Behavior'),
]

available = []
for anm, date, probes, data_dir in EPHYS_SESSIONS:
    fn = f'data_structure_{anm}_{date}.mat'
    fpath = os.path.join(DATA_ROOT, data_dir, fn)
    if os.path.exists(fpath):
        available.append((anm, date, probes, data_dir))
```

iii. In `CONVERSION_NOTES.md`, the agent says it matched the reference recording-session lists from the code and excluded only sessions lacking neural data. The trajectory shows it read the reference loading scripts to assemble this manual manifest.

## 1-b. How are the data split into subjects?

i. Subjects are split by animal name (`anm`). After sessions are processed, the code builds a unique subject list in first-seen order and maps each session to that subject list with `subject_idx`.

ii. 
```python
subjects_set = []
...
if anm not in subjects_set:
    subjects_set.append(anm)

subjects = subjects_set
subject_idx = np.array([subjects.index(s['anm']) for s in all_sessions])
```

iii. The agent’s notes justify this as a direct subject mapping from the session metadata; no additional reasoning beyond preserving session order is given.

## 1-c. How are the data split into sessions?

i. Each tuple in `EPHYS_SESSIONS` defines one session by `(animal, date, probes, data_dir)`. A processed session becomes one element of `all_sessions` and later one element of `data['neural']`, `data['input']`, and `data['output']`.

ii. 
```python
def process_session(anm, date, probe_nums, data_dir, time_edges, time_axis):
    fn = f'data_structure_{anm}_{date}.mat'
    fpath = os.path.join(DATA_ROOT, data_dir, fn)
    ...

for anm, date, probes, data_dir in available:
    result = process_session(anm, date, probes, data_dir, time_edges, time_axis)
    if result is not None:
        all_sessions.append(result)
```

iii. `CONVERSION_NOTES.md` says the session list was meant to reproduce the paper’s electrophysiology sessions, using the probe assignments from the reference loading scripts.

## 1-d. How are the data split into trials?

i. Raw trials are indexed from `bp.Ntrials`. For converted output, the code keeps only `valid_idx` trials and creates one `(neurons, time)` matrix, one input array, and one output array per kept trial.

ii. 
```python
ntrials = sd.get_ntrials()
...
valid_idx = np.where(valid_trials)[0]
...
for trial_idx in valid_idx:
    neural = trialdat[:, :, trial_idx]
    input_data = time_axis.reshape(1, -1).copy()
    ...
    neural_trials.append(neural)
    input_trials.append(input_data)
```

iii. The justification in the notes is that the decoder should only see analysis-valid trials, matching the reference trial selection rules.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to include only `hit` or `miss` trials and to exclude stimulation and early-lick trials. Ignore trials are excluded implicitly because they are neither hit nor miss.

ii. 
```python
hit = sd.get_trial_array('hit').astype(bool)
miss = sd.get_trial_array('miss').astype(bool)
early = sd.get_trial_array('early').astype(bool)
stim_enable = sd.get_stim_enable().astype(bool)

valid_trials = (hit | miss) & ~stim_enable & ~early
valid_idx = np.where(valid_trials)[0]
```

iii. `CONVERSION_NOTES.md` explicitly says this was chosen to match the reference conditions: include hit/miss, exclude stim-enabled, early-lick, and ignore/no-response trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the cluster-level spike-trial assignments and within-trial spike times in `obj.clu`: specifically cluster `quality`, `trial`, and `trialtm`, restricted to the chosen probes.

ii. 
```python
for i in range(n_clu):
    ...
    trial = np.array(self.f[t_ref]).flatten()
    trialtm = np.array(self.f[tt_ref]).flatten()
    clusters.append((q, trial, trialtm))
```

iii. The notes cite `findClusters.m`, `alignSpikes.m`, and `getSeq.m` as the reference for using cluster quality plus `trial` and `trialtm` to build neural sequences.

## 2-b. How is the `neural` data processed?

i. For each kept cluster and each raw trial, spike times are aligned to the trial’s go cue, binned from `-2.5` to `2.5` s in `10` ms bins, converted to firing rate, and smoothed with a causal Gaussian kernel using reflect padding.

ii. 
```python
aligned_times = spike_times[trial_mask] - gocue[j]
counts, _ = np.histogram(aligned_times, bins=time_edges)
fr = counts / DT
trialdat[i, :, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)
```

iii. `CONVERSION_NOTES.md` says this was intended to match `alignSpikes.m`, `getSeq.m`, and `mySmooth.m`: go-cue alignment, `dt = 1/100`, smoothing window `15`, and reflect boundary handling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code first excludes clusters with labels in `{'garbage', 'gabrga', 'noisy', 'real?'}`. It then removes units with mean firing rate `<= 1 Hz`, and finally skips whole sessions with fewer than `10` remaining units.

ii. 
```python
filtered_clusters = [(q, t, tt) for q, t, tt in clusters if q not in EXCLUDE_QUALITIES]
...
mean_frs = np.mean(np.mean(trialdat[:, :, valid_idx], axis=2), axis=1)
keep_units = mean_frs > LOW_FR_THRESH

if keep_units.sum() < 10:
    ...
```

iii. The agent justifies the quality-label filter as matching `findClusters.m` with `quality = {'all'}` and the 1 Hz cutoff as matching `removeLowFRClusters.m`; the notes also say the 10-unit session minimum comes from the Methods.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every trial is aligned to that trial’s `goCue` timestamp by subtracting `gocue[j]` from that trial’s spike times before binning.

ii. 
```python
gocue = sd.get_event_times('goCue')
...
aligned_times = spike_times[trial_mask] - gocue[j]
counts, _ = np.histogram(aligned_times, bins=time_edges)
```

iii. The notes say alignment to go cue was chosen because both the decoder instructions and the reference code used `params.alignEvent = 'goCue'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use `10 ms` bins (`DT = 1/100`) from `-2.5` to `2.5` s. No later temporal rebinning is applied.

ii. 
```python
DT = 1/100  # 10 ms
...
time_edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = time_edges[:-1] + DT / 2
```

iii. `CONVERSION_NOTES.md` states this was meant to match `getSeq.m` (`dt = 1/100`).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not derived from a raw time series field; it is the synthetic analysis time axis centered on the raw `goCue` event. The only raw event it depends on conceptually is `bp.ev.goCue`.

ii. 
```python
gocue = sd.get_event_times('goCue')
...
time_axis = time_edges[:-1] + DT / 2
input_data = time_axis.reshape(1, -1).copy()
```

iii. The notes justify this as the required decoder input specified by the task: continuous time from go cue.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code constructs bin centers from the fixed window `[-2.5, 2.5]` s using `DT = 10 ms`, then copies that same vector into every kept trial as a `1 x T` input array.

ii. 
```python
time_edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = time_edges[:-1] + DT / 2
...
input_data = time_axis.reshape(1, -1).copy()
```

iii. The agent’s notes describe this as a direct representation of time relative to go cue, with no extra preprocessing.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is aligned by construction: the same `time_axis` used for neural bin centers is copied into each trial’s input.

ii. 
```python
n_timebins = len(time_axis)
...
neural = trialdat[:, :, trial_idx]
input_data = time_axis.reshape(1, -1).copy()
```

iii. The notes explicitly say the input is the go-cue-relative time axis used for the neural bins.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the raw behavioral side-choice flags, specifically `bp.R` (with `bp.L` loaded but unused in the mapping).

ii. 
```python
R = sd.get_trial_array('R').astype(bool)
L = sd.get_trial_array('L').astype(bool)
...
lick_dir = 1 if R[trial_idx] else 0
```

iii. `CONVERSION_NOTES.md` says this output was taken from `bp.R`, mapping left to `0` and right to `1`.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. There is no transformation beyond a binary map: `R=True` becomes `1`, otherwise `0`, and that scalar is repeated across all time bins for the trial.

ii. 
```python
lick_dir = 1 if R[trial_idx] else 0
...
out[0, :] = t['lick_dir']
```

iii. The notes justify this as the task-specified categorical output.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `bp.autowater`.

ii. 
```python
autowater = sd.get_trial_array('autowater')
...
context = 0 if autowater[trial_idx] == 1 else 1
```

iii. The trajectory shows the agent relied on `WorkingWithDataObjs.m`, which says `obj.bp.autowater` can be used as a proxy for water-cued versus delayed-response blocks.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code maps `autowater == 1` to water-cued (`0`) and everything else to delayed-response (`1`), then repeats that scalar across time bins.

ii. 
```python
context = 0 if autowater[trial_idx] == 1 else 1
...
out[1, :] = t['context']
```

iii. `CONVERSION_NOTES.md` explicitly states the rule `WC=0 (autowater=1), DR=1 (autowater=0)`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `bp.hit` and, implicitly, `bp.miss` because only hit/miss trials are kept.

ii. 
```python
hit = sd.get_trial_array('hit').astype(bool)
miss = sd.get_trial_array('miss').astype(bool)
...
outcome = 1 if hit[trial_idx] else 0
```

iii. The notes justify this as `incorrect=0 (miss), correct=1 (hit)`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code maps hit to `1` and non-hit kept trials to `0`, then repeats that per-trial value over time bins.

ii. 
```python
outcome = 1 if hit[trial_idx] else 0
...
out[2, :] = t['outcome']
```

iii. The agent’s notes say this was the task-specified binary outcome variable.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the bottom-camera DeepLabCut trajectory `ts` and `frameTimes`, using the feature named `top_tongue` if present, otherwise the first feature containing `"tongue"`.

ii. 
```python
bottom_feats = sd.get_traj_feature_names(1)
...
if name == 'top_tongue' and tongue_feat_idx is None:
    tongue_feat_idx = idx
...
ft, x, y = sd.get_trial_traj(1, trial_idx, tongue_feat_idx)
```

iii. `CONVERSION_NOTES.md` says tongue velocity was computed from bottom-camera DLC tracking of the `top_tongue` feature.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The code takes first differences of the raw `x` and `y` positions, converts them to Euclidean speed at `400 Hz`, timestamps them at derivative midpoints, aligns them by `vidshift` and `goCue`, and interpolates them onto the neural `10 ms` time grid.

ii. 
```python
dt_vid = 1.0 / VIDEO_FPS
vel = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt_vid
vel_t = ft[:-1] + dt_vid / 2
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
tongue_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. The notes cite the Methods: velocity is the first-order derivative of tracked position, with the tongue treated specially relative to missing-value filling.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. After collecting all trials in a session, the code computes the session-wide 50th percentile of all tongue-velocity bins and labels each time bin as `1` if it is at or above that threshold and `0` otherwise. `NaN` bins are forced to class `0`.

ii. 
```python
tongue_thresh = np.nanpercentile(all_tongue, 50) if not np.all(np.isnan(all_tongue)) else 0
...
tongue_disc = np.where(np.isnan(t['tongue_vel']), 0,
                      np.where(t['tongue_vel'] >= tongue_thresh, 1, 0)).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` says this followed the decoder instructions to discretize at the per-session 50th percentile; it also notes that NaNs outside video coverage were assigned to class 0.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. It is aligned with the same go-cue-centered clock as the neural data by subtracting the video offset and the trial’s go cue from video timestamps, then interpolating onto `time_axis`.

ii. 
```python
vidshift = sd.get_video_offset()
...
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
tongue_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. The notes say this was intended to match `findVideoOffset.m` and the frame-time alignment shown in the reference code.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom-camera DeepLabCut `ts` and `frameTimes`, preferring the feature named `bottom_paw` and otherwise the first feature containing `"paw"`.

ii. 
```python
if name == 'bottom_paw' and paw_feat_idx is None:
    paw_feat_idx = idx
...
ft, x, y = sd.get_trial_traj(1, trial_idx, paw_feat_idx)
```

iii. `CONVERSION_NOTES.md` says paw velocity was computed the same way as tongue velocity but using `bottom_paw`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The code computes Euclidean speed from raw bottom-camera `x` and `y` position differences at `400 Hz`, aligns derivative timestamps by video offset and go cue, and interpolates onto neural bins.

ii. 
```python
dt_vid = 1.0 / VIDEO_FPS
vel = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt_vid
vel_t = ft[:-1] + dt_vid / 2
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
paw_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. The notes justify this as “same as tongue but using `bottom_paw`”; there is no separate discussion of the paper’s nearest-value fill for non-tongue features.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The code thresholds all paw-velocity bins within a session at the session median (50th percentile), assigns `1` to bins at or above threshold and `0` otherwise, and maps `NaN` bins to `0`.

ii. 
```python
paw_thresh = np.nanpercentile(all_paw, 50) if not np.all(np.isnan(all_paw)) else 0
...
paw_disc = np.where(np.isnan(t['paw_vel']), 0,
                   np.where(t['paw_vel'] >= paw_thresh, 1, 0)).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` says this followed the decoder task’s per-session 50th-percentile discretization.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw velocity uses the same alignment recipe as tongue velocity: `frameTimes - vidshift - goCue`, then interpolation to the neural time grid.

ii. 
```python
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
paw_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. The notes justify this with the same video-offset logic used for all video-derived variables.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_<animal>_<date>.mat` file, specifically its `me` variable and nested `data` field. Frame times come from the trajectory object for interpolation.

ii. 
```python
me_fn = f'motionEnergy_{anm}_{date}.mat'
...
me_var = me_mat['me']
...
data_field = me_struct['data']
...
ft = sd.get_trial_frame_times(1, trial_idx)
```

iii. `CONVERSION_NOTES.md` says motion energy was loaded from separate motion-energy files and that the loader was written to handle multiple MAT-file formats.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The code unwraps `me.data` if necessary, leaves it continuous, aligns per-trial frame times by `vidshift` and `goCue`, and interpolates onto the neural `10 ms` grid. It loads the stored `moveThresh` but does not use it downstream.

ii. 
```python
thresh = float(me_struct['moveThresh'].flatten()[0])
...
me_interp = np.full(n_timebins, np.nan)
if me_data is not None and trial_idx < len(me_data):
    ft = sd.get_trial_frame_times(1, trial_idx)
    ft_aligned = ft - vidshift - gocue[trial_idx]
    me_interp = np.interp(time_axis, ft_aligned, me_data[trial_idx], left=np.nan, right=np.nan)
```

iii. The notes say the agent intentionally used a per-session 50th-percentile discretization for decoder output rather than the paper’s manual movement threshold, but otherwise tried to preserve the reference alignment logic.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The code computes the session-wide 50th percentile of all interpolated motion-energy bins and labels bins above or equal to that threshold as `1`; `NaN` bins are assigned to `0`.

ii. 
```python
me_thresh_50 = np.nanpercentile(all_me, 50) if not np.all(np.isnan(all_me)) else 0
...
me_disc = np.where(np.isnan(t['motion_energy']), 0,
                  np.where(t['motion_energy'] >= me_thresh_50, 1, 0)).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` explicitly says this 50th-percentile rule was chosen to satisfy the decoder instructions.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by taking trial frame times from the trajectory object, subtracting `vidshift` and the trial’s `goCue`, and interpolating onto `time_axis`.

ii. 
```python
vidshift = sd.get_video_offset()
...
ft = sd.get_trial_frame_times(1, trial_idx)
ft_aligned = ft - vidshift - gocue[trial_idx]
me_interp = np.interp(time_axis, ft_aligned, me_data[trial_idx], left=np.nan, right=np.nan)
```

iii. The notes cite `findVideoOffset.m` as the source of the alignment rule, though they do not discuss the camera-view choice used for the frame times.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles many irregularities with broad `try/except` blocks. Missing neural files, missing clusters, too few valid trials, or too few units cause sessions to be skipped. Missing tongue/paw/motion-energy values become `NaN`, and during discretization `NaN` bins are converted to category `0`.

ii. 
```python
try:
    clusters = sd.get_clusters(probe_nums)
except (KeyError, AttributeError) as e:
    ...
    return None

...
tongue_vel = np.full(n_timebins, np.nan)
...
except:
    pass

...
tongue_disc = np.where(np.isnan(t['tongue_vel']), 0,
                      np.where(t['tongue_vel'] >= tongue_thresh, 1, 0)).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` mentions handling multiple motion-energy formats and assigning missing video-derived values to class 0; the trajectory shows the agent favored permissive error handling over failing hard.

## 11-a. What are the most time-consuming steps of the code?

i. The most expensive work is the nested per-unit, per-trial spike binning and smoothing loop, followed by the per-trial video-feature loading and interpolation for tongue, paw, and motion energy.

ii. 
```python
for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
    for j in range(ntrials):
        ...
        trialdat[i, :, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)

for trial_idx in valid_idx:
    ...
    ft, x, y = sd.get_trial_traj(1, trial_idx, tongue_feat_idx)
    ...
```

iii. There is no explicit written justification; this is simply where the code performs the bulk of its repeated numeric work.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit/per-trial spike histogram loop, the per-trial tongue and paw velocity loops, the smoothing loop over columns, and the repeated per-trial interpolation loops could all have been vectorized or batch-processed more efficiently.

ii. 
```python
for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
    for j in range(ntrials):
        ...

for j in range(x_filt.shape[1]):
    out[:, j] = np.convolve(x_filt[:, j], kern, mode='same')
```

iii. No separate justification is given; the agent prioritized a straightforward translation of the reference logic.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats nearly identical logic for tongue and paw velocity, repeatedly scans feature names to find indices, and repeatedly interpolates aligned video signals trial by trial.

ii. 
```python
if tongue_feat_idx is not None:
    ...
    tongue_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)

if paw_feat_idx is not None:
    ...
    paw_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. The trajectory indicates the agent chose duplicated, explicit code paths rather than abstracting the shared video-feature logic.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `me_thresh` from the motion-energy files but never uses it. It also loads `L` but never uses it. In addition, it expands scalar trial labels (`lick_dir`, `context`, `outcome`) into full time-by-trial arrays, which is redundant work/storage for downstream decoding.

ii. 
```python
L = sd.get_trial_array('L').astype(bool)
...
me_data, me_thresh = load_motion_energy(data_dir, anm, date)
...
out = np.zeros((6, n_timebins), dtype=np.int64)
out[0, :] = t['lick_dir']
out[1, :] = t['context']
out[2, :] = t['outcome']
```

iii. No explicit justification is given. The notes only explain why the agent ignored the stored motion-energy threshold in favor of the task-required 50th-percentile discretization.
