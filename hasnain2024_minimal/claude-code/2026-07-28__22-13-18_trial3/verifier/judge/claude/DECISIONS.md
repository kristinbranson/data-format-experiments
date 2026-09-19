# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each session is one MATLAB file `data_structure_<anm>_<date>.mat` located in one of two folders (`Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior`). The 45 session names and their probes are hard-coded in `EPHYS_SESSIONS`. A `SessionData` class handles both v7.3 (HDF5) and v5 MATLAB formats by checking the file header. Motion energy is loaded separately from `motionEnergy_<anm>_<date>.mat` files via `load_motion_energy`.

ii.
```python
EPHYS_SESSIONS = [
    ('EKH1', '2021-08-07', [2], 'Ephys_Behavior'),
    ...
    ('JEB24', '2023-11-03', [1], 'RandomizedDelay_Ephys_Behavior'),
]

class SessionData:
    def __init__(self, fpath):
        self.is_h5 = self._check_h5(fpath)
        if self.is_h5:
            self.f = h5py.File(fpath, 'r')
            self.obj = self.f['obj']
        else:
            mat = sio.loadmat(fpath, squeeze_me=True, struct_as_record=False)
            self.obj_v5 = mat['obj']
```

iii. The agent examined the data directory structure, identified the two task folders, and built a session list by reading the authors' loading scripts. The agent noted it needed to handle both MATLAB file formats.

## 1-b. How are the data split into subjects?

i. The animal name is the first element of each session tuple (e.g., `'EKH1'`). Subjects are collected in encounter order (not sorted) into `subjects_set`.

ii.
```python
subjects_set = []
for anm, date, probes, data_dir in available:
    result = process_session(anm, date, probes, data_dir, time_edges, time_axis)
    if result is not None:
        all_sessions.append(result)
        if anm not in subjects_set:
            subjects_set.append(anm)

subjects = subjects_set
subject_idx = np.array([subjects.index(s['anm']) for s in all_sessions])
```

iii. The agent derived subject identity from the session tuple rather than from within the data files.

## 1-c. How are the data split into sessions?

i. One session is one tuple in `EPHYS_SESSIONS`, corresponding to one file on disk. The AI lists 45 sessions (25 fixed-delay + 20 randomized-delay), including `JEB23_2023-10-20` which is not in the reference's 44-session list.

ii.
```python
for anm, date, probes, data_dir in available:
    result = process_session(anm, date, probes, data_dir, time_edges, time_axis)
```

iii. The agent built the session list from the data loading scripts. It included one extra session (`JEB23_2023-10-20`) that the reference excludes.

## 1-d. How are the data split into trials?

i. Trials are indexed from 0 to `ntrials-1` where `ntrials` comes from `bp.Ntrials`. Per-trial fields (hit, miss, R, L, etc.) are read as flat arrays indexed by trial number.

ii.
```python
ntrials = sd.get_ntrials()
hit = sd.get_trial_array('hit').astype(bool)
miss = sd.get_trial_array('miss').astype(bool)
...
valid_idx = np.where(valid_trials)[0]
```

iii. The agent used the trial count from `bp.Ntrials` and accessed per-trial fields directly.

## 1-e. How are trials filtered based on quality controls?

i. Three filters: early-lick trials (`bp.early`), photostimulation trials (`bp.stim.enable`), and **ignore trials** (trials that are neither hit nor miss) are all excluded. The filter is `(hit | miss) & ~stim_enable & ~early`. The AI does NOT check for trials past the end of the recording.

ii.
```python
valid_trials = (hit | miss) & ~stim_enable & ~early
valid_idx = np.where(valid_trials)[0]
```

iii. The agent described filtering out stim, early lick, and ignore trials. The exclusion of ignore trials is a deviation from the reference, which keeps ignore trials as a third outcome class.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` spike-sorted clusters, specifically `trial` (1-based trial index for each spike), `trialtm` (spike time relative to trial start), and `quality` (curation label). Go cue times from `bp.ev.goCue` are used for alignment.

ii.
```python
clusters = sd.get_clusters(probe_nums)
# Each cluster is (quality, trial_array, trialtm_array)
aligned_times = spike_times[trial_mask] - gocue[j]
counts, _ = np.histogram(aligned_times, bins=time_edges)
```

iii. The agent read the cluster data fields to extract spike times and trial assignments.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 10 ms bins (DT = 1/100), converted to firing rates (counts / DT), then smoothed with a **causal** Gaussian filter. The causal Gaussian uses `gausswin(15)` with sigma = (15-1)/(2*2.5) = 2.8 bins, but zeros out the first half of the kernel so only past values contribute. Boundary condition is 'reflect'.

ii.
```python
DT = 1/100  # 10 ms
SMOOTH_WINDOW = 15
...
def causal_gaussian_smooth(x, N, bctype='reflect'):
    kern = windows.gaussian(N, std=(N-1)/(2*2.5))
    kern[:N//2] = 0
    kern = kern / kern.sum()
    out[:, j] = np.convolve(x_filt[:, j], kern, mode='same')
```

iii. The agent implemented causal Gaussian smoothing matching the reference code's `mySmooth.m`. It noted the window size of 15 and boundary condition 'reflect'.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. Clusters with quality labels in `{'garbage', 'gabrga', 'noisy', 'real?'}` are excluded (matched case-insensitively). Then units with mean firing rate <= 1 Hz (over valid trials only) are removed. The AI does NOT exclude `'poor'` quality clusters.

ii.
```python
EXCLUDE_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
filtered_clusters = [(q, t, tt) for q, t, tt in clusters if q not in EXCLUDE_QUALITIES]
...
mean_frs = np.mean(np.mean(trialdat[:, :, valid_idx], axis=2), axis=1)
keep_units = mean_frs > LOW_FR_THRESH
```

iii. The agent followed the reference code's `findClusters.m` exclusion list. It did not include `'poor'` in the exclusion set.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned by subtracting the go cue time: `aligned_times = spike_times[trial_mask] - gocue[j]`. This is done per-trial in a loop.

ii.
```python
for j in range(ntrials):
    trial_mask = (trial_nums == (j + 1))
    aligned_times = spike_times[trial_mask] - gocue[j]
    counts, _ = np.histogram(aligned_times, bins=time_edges)
```

iii. The agent aligned to go cue as specified in the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is **10 ms** (DT = 1/100), yielding 500 time bins from -2.5 to 2.5 s. No rebinning is applied.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1/100  # 10 ms
time_edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = time_edges[:-1] + DT / 2
```

iii. The agent chose 10 ms based on the reference code's `params.dt = 1/200`, but interpreted it as 1/100 = 10 ms instead of 1/200 = 5 ms.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the time axis itself: bin centers from -2.5 to 2.5 s in 10 ms steps. It is not derived from raw data variables but defined by the binning grid.

ii.
```python
time_axis = time_edges[:-1] + DT / 2
input_data = time_axis.reshape(1, -1).copy()
```

iii. The agent defined the time axis from the binning parameters.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing. The time axis is the bin centers of the neural binning grid.

ii. N/A

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time axis is defined by the same bin edges used for spike binning, so alignment is inherent.

ii.
```python
time_edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = time_edges[:-1] + DT / 2
# same time_edges used in:
counts, _ = np.histogram(aligned_times, bins=time_edges)
```

iii. The agent used the same binning grid for both neural data and the time input.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The per-trial field `bp.R` (right-instructed) is used directly. A right-instructed trial is coded as 1, otherwise 0.

ii.
```python
R = sd.get_trial_array('R').astype(bool)
lick_dir = 1 if R[trial_idx] else 0
```

iii. The agent used the instructed side `R` directly as lick direction, without combining it with hit/miss to determine actual lick direction.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A simple binary: `R=1` maps to right (1), `R=0` maps to left (0). There is no third "no lick" class. The lick direction is equated with the instructed side, not derived from the combination of instructed side and outcome (hit/miss).

ii.
```python
lick_dir = 1 if R[trial_idx] else 0

out[0, :] = t['lick_dir']
```

iii. The agent noted the instruction specified left=0, right=1 and implemented it as the instructed side.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The per-trial field `bp.autowater`. Autowater=1 indicates the WC (water-cued) context.

ii.
```python
autowater = sd.get_trial_array('autowater')
context = 0 if autowater[trial_idx] == 1 else 1
```

iii. The agent read the autowater field directly.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct relabelling: autowater=1 becomes WC (0), otherwise DR (1).

ii.
```python
context = 0 if autowater[trial_idx] == 1 else 1
```

iii. Straightforward mapping matching the instructions.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The per-trial field `bp.hit`. Since ignore trials are filtered out, only hit and miss remain.

ii.
```python
hit = sd.get_trial_array('hit').astype(bool)
outcome = 1 if hit[trial_idx] else 0
```

iii. Because the AI filters out ignore trials, outcome is binary (hit=correct=1, miss=incorrect=0).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Binary mapping: hit=1 becomes correct (1), otherwise incorrect (0). There is no third "ignore" class because ignore trials were already excluded.

ii.
```python
outcome = 1 if hit[trial_idx] else 0
```

iii. The agent's filtering choice (exclude ignore trials) simplifies outcome to a binary.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. DLC tracking from `obj.traj`, specifically the bottom camera view only (`view=1`), using the `top_tongue` feature. The tracked x,y positions and frame times are used.

ii.
```python
tongue_feat_idx = None
for idx, name in enumerate(bottom_feats):
    if name == 'top_tongue' and tongue_feat_idx is None:
        tongue_feat_idx = idx
...
ft, x, y = sd.get_trial_traj(1, trial_idx, tongue_feat_idx)
```

iii. The agent identified `top_tongue` from the bottom camera as the tongue tracking feature.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Simple finite differences of raw (unsmoothed) x,y positions divided by a fixed `dt_vid = 1/400`. Speed is `sqrt(dx^2 + dy^2) / dt_vid`. No likelihood filtering, no per-run smoothing. The velocity is then interpolated to the neural time bins using `np.interp`. No normalization across cameras (only one camera used). Discretized at session 50th percentile; NaN bins map to class 0 (below threshold).

ii.
```python
dt_vid = 1.0 / VIDEO_FPS
vel = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt_vid
vel_t = ft[:-1] + dt_vid / 2
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
tongue_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
...
tongue_disc = np.where(np.isnan(t['tongue_vel']), 0,
                      np.where(t['tongue_vel'] >= tongue_thresh, 1, 0))
```

iii. The agent computed velocity as simple finite differences without DLC likelihood filtering or Gaussian smoothing of positions.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Session 50th percentile of all non-NaN tongue velocity values across trials. Values >= threshold get class 1, below get class 0. NaN values (outside video range) map to class 0, not a separate "not visible" class.

ii.
```python
tongue_thresh = np.nanpercentile(all_tongue, 50)
tongue_disc = np.where(np.isnan(t['tongue_vel']), 0,
                      np.where(t['tongue_vel'] >= tongue_thresh, 1, 0))
```

iii. The agent's instructions mentioned only two bins (0 and 1), though the full instructions specified a third "not visible" class (class 2).

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by subtracting the video offset and go cue time, then velocity is interpolated to the neural time axis using `np.interp`.

ii.
```python
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
tongue_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. The agent used the video offset computed from bitcode alignment, matching the reference's approach.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. DLC tracking from `obj.traj`, bottom camera (`view=1`), using `bottom_paw` feature.

ii.
```python
paw_feat_idx = None
for idx, name in enumerate(bottom_feats):
    if name == 'bottom_paw' and paw_feat_idx is None:
        paw_feat_idx = idx
```

iii. The agent chose `bottom_paw` from the bottom camera.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same as tongue: simple finite differences of raw x,y divided by fixed dt_vid, interpolated to neural time bins. No likelihood filtering or smoothing.

ii.
```python
ft, x, y = sd.get_trial_traj(1, trial_idx, paw_feat_idx)
dt_vid = 1.0 / VIDEO_FPS
vel = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt_vid
paw_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. Same approach as tongue velocity.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: session 50th percentile, NaN maps to class 0 (not a separate "not visible" class).

ii.
```python
paw_disc = np.where(np.isnan(t['paw_vel']), 0,
                   np.where(t['paw_vel'] >= paw_thresh, 1, 0))
```

iii. Same approach as tongue velocity.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: frame times corrected by video offset and go cue, then interpolated.

ii.
```python
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
paw_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. Same alignment approach as tongue.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Separate `motionEnergy_<anm>_<date>.mat` files. The `me` variable is unwrapped through potential nested struct layers to get per-trial arrays. Also extracts `moveThresh` if available.

ii.
```python
def load_motion_energy(data_dir, anm, date):
    me_mat = sio.loadmat(me_path, squeeze_me=False, struct_as_record=True)
    me_var = me_mat['me']
    if me_var.dtype.names is not None and 'data' in me_var.dtype.names:
        me_struct = me_var[0, 0]
        data_field = me_struct['data']
        if data_field.dtype.names is not None and 'data' in data_field.dtype.names:
            data_field = data_field[0, 0]['data']
```

iii. The agent handled multiple motion energy file formats.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy values are interpolated to the neural time axis using `np.interp`. No further processing. Discretized at session 50th percentile.

ii.
```python
ft = sd.get_trial_frame_times(1, trial_idx)
ft_aligned = ft - vidshift - gocue[trial_idx]
me_interp = np.interp(time_axis, ft_aligned, me_data[trial_idx], left=np.nan, right=np.nan)
```

iii. The agent treated motion energy as already processed and only needed to align and discretize it.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Session 50th percentile, NaN maps to class 0. No separate "no video" class.

ii.
```python
me_disc = np.where(np.isnan(t['motion_energy']), 0,
                  np.where(t['motion_energy'] >= me_thresh_50, 1, 0))
```

iii. Same approach as tongue and paw.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Frame times from the bottom camera (`view=1`) are corrected by video offset and go cue, then motion energy is interpolated to neural time bins.

ii.
```python
ft = sd.get_trial_frame_times(1, trial_idx)
ft_aligned = ft - vidshift - gocue[trial_idx]
me_interp = np.interp(time_axis, ft_aligned, me_data[trial_idx], left=np.nan, right=np.nan)
```

iii. The agent used the bottom camera frame times for motion energy alignment.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Broad `try/except` blocks catch errors in tongue velocity, paw velocity, and motion energy extraction. On failure, the values default to NaN arrays which then get discretized to class 0. If motion energy files don't exist, `load_motion_energy` returns `None`. Sessions with fewer than 5 valid trials or fewer than 10 units are skipped entirely.

ii.
```python
try:
    ft, x, y = sd.get_trial_traj(1, trial_idx, tongue_feat_idx)
    ...
except:
    pass

if len(valid_idx) < 5:
    print(f'  WARNING: Too few valid trials ({len(valid_idx)}), skipping')
    sd.close()
    return None
```

iii. The agent used broad exception handling to be robust against data format issues.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the MATLAB files and the per-trial, per-cluster spike binning loop. The spike binning uses a triple-nested loop (over clusters, then trials within each cluster), which is much slower than the reference's vectorized `histogram2d` approach.

ii.
```python
for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
    for j in range(ntrials):
        trial_mask = (trial_nums == (j + 1))
        aligned_times = spike_times[trial_mask] - gocue[j]
        counts, _ = np.histogram(aligned_times, bins=time_edges)
```

iii. The agent did not attempt to vectorize the spike counting.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop iterates over all clusters and all trials individually. The reference uses `np.histogram2d` to bin all trials at once per cluster. The per-trial video velocity computation could also be vectorized in principle.

ii.
```python
# Current: per-trial loop
for j in range(ntrials):
    trial_mask = (trial_nums == (j + 1))
    counts, _ = np.histogram(aligned_times, bins=time_edges)

# Reference: vectorized
counts, _, _ = np.histogram2d(spike_trial, spike_time, bins=[trial_edges, BIN_EDGES])
```

iii. The agent did not address vectorization.

## 11-c. What processing does the code repeat multiple times?

i. The spike binning computes `trial_mask = (trial_nums == (j + 1))` for every cluster-trial combination, even when many clusters share the same trial structure. Also, the go cue times and trial arrays are re-read for each processing step (neural, video) rather than being computed once.

ii.
```python
for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
    for j in range(ntrials):
        trial_mask = (trial_nums == (j + 1))
```

iii. No explicit caching of intermediate results.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code bins spikes for ALL trials (including filtered ones) before selecting valid trials. `trialdat` is built for all `ntrials`, then only `valid_idx` trials are used. The smoothing is also applied to all trials. Also, a `sample_data.pkl` file is created that is not needed.

ii.
```python
trialdat = np.zeros((n_units, n_timebins, ntrials))
# ... fills all ntrials ...
trialdat = trialdat[keep_units, :, :]
# Later, only valid_idx trials are used:
for trial_idx in valid_idx:
    neural = trialdat[:, :, trial_idx]
```

iii. The agent did not optimize to only process valid trials.
