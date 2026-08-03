# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a session list in `EPHYS_SESSIONS`, where each entry contains animal, date, probe list, and source subdirectory. It then checks which listed `data_structure_*.mat` files exist under `data/`, loads each session through a `SessionData` wrapper that branches between MATLAB v7.3 HDF5 and older MAT files, and separately loads motion-energy files with `load_motion_energy`.

ii. 
```python
EPHYS_SESSIONS = [
    ('EKH1', '2021-08-07', [2], 'Ephys_Behavior'),
    ...
    ('JEB23', '2023-10-20', [1], 'RandomizedDelay_Ephys_Behavior'),
    ('JEB23', '2023-10-21', [1], 'RandomizedDelay_Ephys_Behavior'),
    ...
]
```

```python
available = []
for anm, date, probes, data_dir in EPHYS_SESSIONS:
    fn = f'data_structure_{anm}_{date}.mat'
    fpath = os.path.join(DATA_ROOT, data_dir, fn)
    if os.path.exists(fpath):
        available.append((anm, date, probes, data_dir))
```

```python
class SessionData:
    def __init__(self, fpath):
        self.fpath = fpath
        self.is_h5 = self._check_h5(fpath)
        if self.is_h5:
            self.f = h5py.File(fpath, 'r')
            self.obj = self.f['obj']
        else:
            mat = sio.loadmat(fpath, squeeze_me=True, struct_as_record=False)
            self.obj_v5 = mat['obj']
```

```python
def load_motion_energy(data_dir, anm, date):
    me_fn = f'motionEnergy_{anm}_{date}.mat'
    me_path = os.path.join(DATA_ROOT, data_dir, me_fn)
```

iii. In `CONVERSION_NOTES.md`, the AI says it is matching the paper code while using a mixed-format MAT loader and handling multiple motion-energy layouts. The trajectory summary also says the final dataset contains 45 sessions from 14 mice.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `anm` field of each hard-coded session tuple. The script appends each unique animal to `subjects_set` in first-seen order and builds `subject_idx` by locating each session’s `anm` in that list.

ii. 
```python
for anm, date, probes, data_dir in available:
    result = process_session(anm, date, probes, data_dir, time_edges, time_axis)
    if result is not None:
        all_sessions.append(result)
        if anm not in subjects_set:
            subjects_set.append(anm)
```

```python
subjects = subjects_set
subject_idx = np.array([subjects.index(s['anm']) for s in all_sessions])
```

iii. There is no separate justification beyond the implementation and the notes’ statement that the converted dataset contains 14 subjects.

## 1-c. How are the data split into sessions?

i. Each tuple in `EPHYS_SESSIONS` is treated as one session, corresponding to one `data_structure_<anm>_<date>.mat` file in either `Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior`. Each successfully processed tuple becomes one element in the final session lists.

ii. 
```python
def process_session(anm, date, probe_nums, data_dir, time_edges, time_axis):
    fn = f'data_structure_{anm}_{date}.mat'
    fpath = os.path.join(DATA_ROOT, data_dir, fn)
```

```python
neural = [s['neural'] for s in all_sessions]
inputs = [s['input'] for s in all_sessions]
outputs = [s['output'] for s in all_sessions]
```

iii. `CONVERSION_NOTES.md` explicitly describes 25 `Ephys_Behavior` sessions and 20 `RandomizedDelay_Ephys_Behavior` sessions, and says two extra JEB24 dates were excluded because they lack neural data.

## 1-d. How are the data split into trials?

i. The AI treats trial index `0..ntrials-1` as the trial partition. Trial-level behavioral arrays come from `bp`, and spike times are assigned to trials using the cluster `trial` numbers. The final per-session trial lists are built by iterating over `valid_idx`.

ii. 
```python
ntrials = sd.get_ntrials()
...
valid_trials = (hit | miss) & ~stim_enable & ~early
valid_idx = np.where(valid_trials)[0]
```

```python
for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
    for j in range(ntrials):
        trial_mask = (trial_nums == (j + 1))
        ...
```

```python
for trial_idx in valid_idx:
    neural = trialdat[:, :, trial_idx]
    input_data = time_axis.reshape(1, -1).copy()
    ...
    neural_trials.append(neural)
```

iii. The AI does not separately justify the trial split in prose; the code implies it trusts the Bpod trial count and cluster `trial` labels.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only hit or miss trials and excludes photostimulation and early-lick trials. Ignore/no-response trials are dropped because they fail `(hit | miss)`. Sessions with fewer than 5 valid trials are skipped entirely. The code does not remove trials that occur after recording has effectively stopped, which leads to all-zero neural trials noted in validation.

ii. 
```python
valid_trials = (hit | miss) & ~stim_enable & ~early
valid_idx = np.where(valid_trials)[0]

if len(valid_idx) < 5:
    print(f'  WARNING: Too few valid trials ({len(valid_idx)}), skipping')
    sd.close()
    return None
```

iii. `CONVERSION_NOTES.md` says the AI was “matching reference code conditions” by including hit and miss trials and excluding stim, early lick, and ignore/no-response trials. The same file later explains all-zero neural trials as late trials from degraded recordings rather than fixing them upstream.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from cluster quality labels, cluster trial numbers, cluster spike times relative to trial start, and go-cue times. In the loader these are exposed as `(quality, trial_array, trialtm_array)` plus `bp.ev.goCue`.

ii. 
```python
def get_clusters(self, probe_nums):
    """Get list of (quality, trial_array, trialtm_array) for each cluster."""
```

```python
clusters = sd.get_clusters(probe_nums)
...
gocue = sd.get_event_times('goCue')
```

```python
for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
    ...
    aligned_times = spike_times[trial_mask] - gocue[j]
```

iii. `CONVERSION_NOTES.md` says spikes are aligned to go cue, clusters are filtered by quality, and low firing-rate units are removed.

## 2-b. How is the `neural` data processed?

i. The AI bins spikes into 10 ms bins over `[-2.5, 2.5]`, converts counts to firing rates by dividing by `DT`, and smooths each per-trial trace with a causal Gaussian kernel implemented in `causal_gaussian_smooth`.

ii. 
```python
TMIN = -2.5
TMAX = 2.5
DT = 1/100  # 10 ms
SMOOTH_WINDOW = 15
BC_TYPE = 'reflect'
```

```python
counts, _ = np.histogram(aligned_times, bins=time_edges)
fr = counts / DT
trialdat[i, :, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)
```

```python
def causal_gaussian_smooth(x, N, bctype='reflect'):
    ...
    kern = windows.gaussian(N, std=(N-1)/(2*2.5))
    kern[:N//2] = 0
```

iii. The notes explicitly justify this as matching `getSeq.m`: 10 ms bins, causal Gaussian window 15, reflect padding, and MATLAB `gausswin` parameters.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI drops clusters whose quality is in `{'garbage', 'gabrga', 'noisy', 'real?'}` and then removes units with mean firing rate `<= 1 Hz` over valid trials and time bins. It also skips whole sessions if fewer than 10 units survive.

ii. 
```python
LOW_FR_THRESH = 1.0
EXCLUDE_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
```

```python
filtered_clusters = [(q, t, tt) for q, t, tt in clusters if q not in EXCLUDE_QUALITIES]
```

```python
mean_frs = np.mean(np.mean(trialdat[:, :, valid_idx], axis=2), axis=1)
keep_units = mean_frs > LOW_FR_THRESH

if keep_units.sum() < 10:
    print(f'  WARNING: Only {keep_units.sum()} units with FR>{LOW_FR_THRESH} Hz, skipping')
```

iii. `CONVERSION_NOTES.md` says this matches `findClusters.m` with `quality='all'` and `removeLowFRClusters.m`, and explicitly says poor and multi units are included.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. For each spike, the AI subtracts the trial’s go-cue time from the spike’s `trialtm` value, then bins the aligned spikes.

ii. 
```python
aligned_times = spike_times[trial_mask] - gocue[j]
counts, _ = np.histogram(aligned_times, bins=time_edges)
```

iii. The notes state that alignment event is go-cue onset and present this as a direct match to the reference alignment choice.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 10 ms bins, producing 500 bins from `-2.5` to `2.5` s. Spikes are binned directly onto this grid; no additional rebinning step is applied after spike counting.

ii. 
```python
DT = 1/100  # 10 ms
...
time_edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = time_edges[:-1] + DT / 2
n_timebins = len(time_axis)
```

iii. The notes and trajectory repeatedly claim that 10 ms is the reference bin size.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The AI derives this input from a synthetic time axis defined around the go cue rather than from a dedicated raw signal. Conceptually it uses the same alignment choice as neural data, but the stored values are the bin centers in `time_axis`.

ii. 
```python
time_edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = time_edges[:-1] + DT / 2
```

```python
input_data = time_axis.reshape(1, -1).copy()
```

iii. `CONVERSION_NOTES.md` says the decoder input is “Time axis relative to go cue onset.”

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI computes bin centers for the fixed `[-2.5, 2.5]` window with 10 ms spacing and copies that vector into every trial as a `(1, T)` array.

ii. 
```python
time_axis = time_edges[:-1] + DT / 2
...
input_data = time_axis.reshape(1, -1).copy()
```

iii. The only stated justification is that this is the continuous time-from-go-cue input requested by the decoder task.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is exactly the same time grid used to bin neural spikes, so each input timepoint corresponds to one neural bin center.

ii. 
```python
counts, _ = np.histogram(aligned_times, bins=time_edges)
...
input_data = time_axis.reshape(1, -1).copy()
```

iii. No extra justification is given beyond using a shared time axis for all trial arrays.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derives lick direction from the trial-side indicator `bp.R`, with `R=True` mapped to right and otherwise left. Although it also loads `L`, `hit`, and `miss`, those are not used for the lick-direction label.

ii. 
```python
R = sd.get_trial_array('R').astype(bool)
L = sd.get_trial_array('L').astype(bool)
...
lick_dir = 1 if R[trial_idx] else 0
```

iii. `CONVERSION_NOTES.md` states that lick direction is “left=0, right=1 (per-trial, from bp.R).”

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI performs a direct binary mapping from the instructed/right-side flag to `0/1` and repeats that single trial label across all time bins. It does not infer actual lick side from hit versus miss, and it does not include a no-lick category because ignore trials were already removed.

ii. 
```python
lick_dir = 1 if R[trial_idx] else 0
...
out = np.zeros((6, n_timebins), dtype=np.int64)
out[0, :] = t['lick_dir']
```

iii. The notes justify the binary coding by the decoder-task specification `left=0, right=1`.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `bp.autowater`.

ii. 
```python
autowater = sd.get_trial_array('autowater')
...
context = 0 if autowater[trial_idx] == 1 else 1
```

iii. `CONVERSION_NOTES.md` explicitly says WC is `autowater=1` and DR is `autowater=0`.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI remaps `autowater` to a binary class: `0` for WC and `1` for DR, then tiles the same label across all time bins of the trial.

ii. 
```python
context = 0 if autowater[trial_idx] == 1 else 1
...
out[1, :] = t['context']
```

iii. The notes say this is the requested coding for behavioral context.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from hit versus miss trial flags. In practice the code reads both `hit` and `miss`, but after trial filtering it only needs `hit` because remaining trials are either hit or miss.

ii. 
```python
hit = sd.get_trial_array('hit').astype(bool)
miss = sd.get_trial_array('miss').astype(bool)
...
outcome = 1 if hit[trial_idx] else 0
```

iii. `CONVERSION_NOTES.md` says outcome is incorrect `0` for miss and correct `1` for hit.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI collapses outcome to a binary label, assigning `1` to hits and `0` otherwise, then repeats that label across time bins. Because it already excluded ignore trials, there is no third class for no response.

ii. 
```python
outcome = 1 if hit[trial_idx] else 0
...
out[2, :] = t['outcome']
```

iii. The notes justify this as matching the decoder-task labels incorrect `0`, correct `1`.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from bottom-camera trajectory data, primarily the `top_tongue` feature in `obj.traj`. The code uses frame times and tracked `x, y` positions from that feature, plus video offset and go-cue times for alignment.

ii. 
```python
bottom_feats = sd.get_traj_feature_names(1)
...
if name == 'top_tongue' and tongue_feat_idx is None:
    tongue_feat_idx = idx
```

```python
ft, x, y = sd.get_trial_traj(1, trial_idx, tongue_feat_idx)
...
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
```

iii. `CONVERSION_NOTES.md` explicitly says tongue velocity is computed from the bottom camera’s `top_tongue` feature.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI computes frame-to-frame speed as `sqrt(dx^2 + dy^2) / dt` at 400 Hz, uses midpoint times for those velocities, aligns them by subtracting the video offset and go cue, and interpolates them onto the neural time axis. It does not apply a likelihood threshold, gap-aware smoothing, per-run differentiation, two-camera averaging, or per-view normalization.

ii. 
```python
ft, x, y = sd.get_trial_traj(1, trial_idx, tongue_feat_idx)
dt_vid = 1.0 / VIDEO_FPS
vel = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt_vid
vel_t = ft[:-1] + dt_vid / 2
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
tongue_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. The notes justify this as a 400 Hz velocity magnitude interpolated to 10 ms bins, with missing values later assigned to class 0.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI concatenates all tongue-velocity timepoints in a session, takes the 50th percentile ignoring NaNs, and assigns class `1` to bins at or above threshold and class `0` below threshold. NaN bins are also forced to class `0`.

ii. 
```python
all_tongue = np.concatenate([t['tongue_vel'] for t in raw_outputs])
...
tongue_thresh = np.nanpercentile(all_tongue, 50) if not np.all(np.isnan(all_tongue)) else 0
```

```python
tongue_disc = np.where(np.isnan(t['tongue_vel']), 0,
                      np.where(t['tongue_vel'] >= tongue_thresh, 1, 0)).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` says tongue velocity is discretized at the per-session 50th percentile and that NaNs outside video coverage are assigned to class 0.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI computes a session-wide video offset from bitcode start times, subtracts that offset and the trial go cue from frame times, then interpolates the resulting velocity trace onto the same `time_axis` used for neural data.

ii. 
```python
def get_video_offset(self):
    ...
    return sglx_mode / sglx_fs - bs_mode
```

```python
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
tongue_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. `CONVERSION_NOTES.md` explicitly cites `findVideoOffset.m` and `WorkingWithDataObjs.m` for this choice.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from bottom-camera trajectory data as well, using a paw feature chosen from `featNames`. The intended feature is `bottom_paw`, with a fallback to the first feature whose name contains `paw`.

ii. 
```python
if name == 'bottom_paw' and paw_feat_idx is None:
    paw_feat_idx = idx
...
if paw_feat_idx is None:
    for idx, name in enumerate(bottom_feats):
        if 'paw' in name.lower():
            paw_feat_idx = idx
            break
```

```python
ft, x, y = sd.get_trial_traj(1, trial_idx, paw_feat_idx)
```

iii. `CONVERSION_NOTES.md` says paw velocity is computed “using `bottom_paw` feature.”

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The AI uses the same frame-to-frame speed and interpolation pipeline as for tongue velocity: finite differences in `x` and `y`, divide by fixed frame interval, align by offset and go cue, and interpolate onto the neural bins. It does not apply the paper’s tracking-quality threshold or smoothing.

ii. 
```python
dt_vid = 1.0 / VIDEO_FPS
vel = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt_vid
vel_t = ft[:-1] + dt_vid / 2
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
paw_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. The notes say paw velocity is processed “same as tongue.”

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The AI pools all paw-velocity values within a session, thresholds at the session median, assigns `1` above threshold and `0` below threshold, and converts NaN bins to `0`.

ii. 
```python
all_paw = np.concatenate([t['paw_vel'] for t in raw_outputs])
paw_thresh = np.nanpercentile(all_paw, 50) if not np.all(np.isnan(all_paw)) else 0
```

```python
paw_disc = np.where(np.isnan(t['paw_vel']), 0,
                   np.where(t['paw_vel'] >= paw_thresh, 1, 0)).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` says paw velocity is discretized at the per-session 50th percentile.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw velocity uses the same session video offset and go-cue subtraction as tongue velocity, then the aligned velocity is interpolated onto `time_axis`.

ii. 
```python
vidshift = sd.get_video_offset()
...
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
paw_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. The notes justify this with the same reference to `findVideoOffset.m`.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from separate `motionEnergy_<anm>_<date>.mat` files. The loader handles either a struct-with-data layout or a direct cell-array layout, and returns per-trial traces plus an optional `moveThresh` value.

ii. 
```python
def load_motion_energy(data_dir, anm, date):
    me_fn = f'motionEnergy_{anm}_{date}.mat'
    me_path = os.path.join(DATA_ROOT, data_dir, me_fn)
```

```python
me_mat = sio.loadmat(me_path, squeeze_me=False, struct_as_record=True)
me_var = me_mat['me']
...
if me_var.dtype.names is not None and 'data' in me_var.dtype.names:
    ...
    thresh = float(me_struct['moveThresh'].flatten()[0])
```

iii. `CONVERSION_NOTES.md` says the standalone motion-energy files are used and that multiple file formats are handled.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI takes the per-frame motion-energy trace and interpolates it onto the neural time axis using aligned frame times. It does not perform any additional smoothing or derivative calculation.

ii. 
```python
ft = sd.get_trial_frame_times(1, trial_idx)
ft_aligned = ft - vidshift - gocue[trial_idx]
me_interp = np.interp(time_axis, ft_aligned, me_data[trial_idx], left=np.nan, right=np.nan)
```

iii. The notes say motion energy is loaded from separate files and interpolated from the 400 Hz video frame rate to 10 ms bins.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The AI uses the per-session 50th percentile of all interpolated motion-energy values and maps bins to `0/1`, again forcing NaNs to class `0`.

ii. 
```python
all_me = np.concatenate([t['motion_energy'] for t in raw_outputs])
me_thresh_50 = np.nanpercentile(all_me, 50) if not np.all(np.isnan(all_me)) else 0
```

```python
me_disc = np.where(np.isnan(t['motion_energy']), 0,
                  np.where(t['motion_energy'] >= me_thresh_50, 1, 0)).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` says motion energy is discretized at the per-session 50th percentile.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI uses the session video offset and trial go cue to align motion-energy frame times, then interpolates onto the shared neural `time_axis`. The code uses `get_trial_frame_times(1, trial_idx)`, i.e. view index `1`.

ii. 
```python
ft = sd.get_trial_frame_times(1, trial_idx)
ft_aligned = ft - vidshift - gocue[trial_idx]
me_interp = np.interp(time_axis, ft_aligned, me_data[trial_idx], left=np.nan, right=np.nan)
```

iii. The notes justify this as matching the video-alignment procedure from the reference code.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles many failures by falling back to missing-valued arrays or skipping sessions. Missing motion-energy files return `(None, None)`. Failures when reading trajectories or frame times are swallowed with bare `except`, leaving the corresponding velocity arrays as all-NaN. Those NaNs are later converted to class `0`. For missing or low-quality sessions, the script may skip the whole session if files are absent, cluster data are unavailable, too few valid trials remain, or fewer than 10 units survive. It does not repair the late-recording problem that produces all-zero neural trials.

ii. 
```python
if not os.path.exists(me_path):
    return None, None
```

```python
tongue_vel = np.full(n_timebins, np.nan)
if tongue_feat_idx is not None:
    try:
        ...
    except:
        pass
```

```python
if len(valid_idx) < 5:
    ...
if keep_units.sum() < 10:
    ...
```

```python
tongue_disc = np.where(np.isnan(t['tongue_vel']), 0,
                      np.where(t['tongue_vel'] >= tongue_thresh, 1, 0)).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` explicitly says NaNs outside video coverage are assigned to class 0, notes missing motion energy for session 35, and acknowledges warnings about all-zero neural trials without adding a corrective filter.

## 11-a. What are the most time-consuming steps of the code?

i. The AI did not explicitly document runtime hotspots. From the implementation, the most expensive steps are likely the nested spike-binning loop over every cluster and every trial, plus per-trial trajectory extraction and interpolation for tongue, paw, and motion energy.

ii. 
```python
for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
    for j in range(ntrials):
        trial_mask = (trial_nums == (j + 1))
        ...
        counts, _ = np.histogram(aligned_times, bins=time_edges)
```

```python
for trial_idx in valid_idx:
    ...
    tongue_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
    ...
    paw_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
    ...
    me_interp = np.interp(time_axis, ft_aligned, me_data[trial_idx], left=np.nan, right=np.nan)
```

iii. No explicit justification appears in the notes or trajectory.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit/per-trial spike loop is an obvious vectorization target, since the reference solution counts spikes across all trials in one call. The per-trial interpolation loops for video-derived outputs could also potentially be reduced or cached more aggressively, although the varying frame counts make that harder.

ii. 
```python
for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
    for j in range(ntrials):
        trial_mask = (trial_nums == (j + 1))
        ...
```

```python
for trial_idx in valid_idx:
    ...
    tongue_vel = np.interp(...)
    paw_vel = np.interp(...)
    me_interp = np.interp(...)
```

iii. The AI did not provide an explicit efficiency justification.

## 11-c. What processing does the code repeat multiple times?

i. The code repeatedly loops over trials to interpolate separate tongue, paw, and motion-energy traces, and it repeatedly scans each cluster against every trial when binning spikes. It also recomputes aligned midpoints and interpolation separately for tongue and paw even though both come from the same view and use the same per-trial offset and go cue.

ii. 
```python
for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
    for j in range(ntrials):
        ...
```

```python
ft, x, y = sd.get_trial_traj(1, trial_idx, tongue_feat_idx)
...
ft, x, y = sd.get_trial_traj(1, trial_idx, paw_feat_idx)
```

```python
ft = sd.get_trial_frame_times(1, trial_idx)
ft_aligned = ft - vidshift - gocue[trial_idx]
```

iii. No explicit justification is given; these repetitions are only visible in the implementation.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads or computes several values that are then unused downstream: `L` is read but never used; `me_thresh` from motion-energy files is returned but ignored; full continuous tongue, paw, and motion-energy traces are built only to be immediately thresholded; and exceptions are silently swallowed rather than producing structured missing-data metadata.

ii. 
```python
L = sd.get_trial_array('L').astype(bool)
```

```python
me_data, me_thresh = load_motion_energy(data_dir, anm, date)
```

```python
raw_outputs.append({
    'lick_dir': lick_dir,
    'context': context,
    'outcome': outcome,
    'tongue_vel': tongue_vel,
    'paw_vel': paw_vel,
    'motion_energy': me_interp,
})
...
final_output_trials = []
for t in raw_outputs:
    tongue_disc = ...
    paw_disc = ...
    me_disc = ...
```

iii. The AI did not justify these extra computations in the notes or trajectory.
